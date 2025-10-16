# ===============================================================
# SARIMAX FEATURE SELECTION PIPELINE
# ===============================================================

set.seed(1234)

# --- SETTINGS ---

target_col <- "total_consumption_kWh"

# SARIMA parameters
ORDER    <- c(2, 0, 2)
SEASONAL <- c(0, 1, 1)
PERIOD   <- 168L

# Filter thresholds
CORR_MIN <- 0.30
VIF_MAX  <- 5

# Domain knowledge features (ALWAYS INCLUDED in final selection)
domain_features <- c(
  "business_hours",
  "holiday",
  "dst"
)

# --- DATA PREPARATION ---

# Use only train + validation data (before first test period)
tz_ref <- attr(model_data$interval, "tzone")
train_val_data <- model_data[
  interval >= as.POSIXct("2023-07-01 00:00:00", tz = tz_ref) &
    interval <  as.POSIXct("2024-10-01 00:00:00", tz = tz_ref)
]
cat(sprintf("Data: %d samples (%s to %s)\n\n",
            nrow(train_val_data),
            min(as.Date(train_val_data$interval)),
            max(as.Date(train_val_data$interval))))

# --- STEP 1: CORRELATION FILTER ---

corr_filter <- function(dt, target_col, corr_min = 0.30) {
  
  cat("--- Step 1: Correlation Filter ---\n")
  
  # Columns to exclude
  exclude_cols <- c(
    "interval", "total_consumption_kWh", "date",
    "occ_co2", "occ_temp",
    "lag_24", "lag_48", "lag_72", "lag_168", "lag_336", "lag_504",
    "rollmean_24", "rollmean_168"
  )
  
  feature_cols <- setdiff(names(dt), exclude_cols)
  y <- dt[[target_col]]
  
  # Compute correlations
  cors <- sapply(feature_cols, function(v) {
    cor(dt[[v]], y, use = "complete.obs")
  })
  
  # Sort by absolute correlation
  cors_sorted <- cors[order(-abs(cors))]
  
  # Filter by threshold
  kept <- names(cors_sorted[!is.na(cors_sorted) & abs(cors_sorted) >= corr_min])
  
  cat(sprintf("Threshold |r| >= %.2f: kept %d/%d features\n",
              corr_min, length(kept), length(feature_cols)))
  cat("Kept:", paste(kept, collapse = ", "), "\n\n")
  
  return(kept)
}

# --- STEP 2: VIF FILTER ---

vif_prune <- function(dt, vars, vif_max = 5) {
  
  cat("--- Step 2: VIF Filter ---\n")
  
  # Function to compute VIF
  vif_vec <- function(M) {
    p <- ncol(M)
    out <- rep(NA_real_, p)
    names(out) <- colnames(M)
    
    if (p == 1L) {
      out[] <- 1
      return(out)
    }
    
    for (j in seq_len(p)) {
      xj <- M[, j]
      Xmj <- M[, -j, drop = FALSE]
      df <- data.frame(xj = xj, Xmj, check.names = FALSE)
      
      fit <- tryCatch(lm(xj ~ ., data = df), error = function(e) NULL)
      
      if (is.null(fit)) {
        out[j] <- Inf
      } else {
        r2 <- max(0, min(1, summary(fit)$r.squared))
        out[j] <- 1 / (1 - r2 + 1e-12)
      }
    }
    out
  }
  
  Xv <- as.data.frame(dt[, ..vars])
  removed_features <- c()
  
  # Iteratively remove highest VIF
  repeat {
    vifs <- vif_vec(Xv)
    vmax <- max(vifs, na.rm = TRUE)
    
    if (!is.finite(vmax) || vmax <= vif_max || ncol(Xv) <= 1L) break
    
    worst <- names(which.max(vifs))
    cat(sprintf("Removing '%s' (VIF = %.2f)\n", worst, vmax))
    removed_features <- c(removed_features, worst)
    Xv <- Xv[, setdiff(names(Xv), worst), drop = FALSE]
  }
  
  final <- names(Xv)
  
  cat(sprintf("Result: %d features remain after VIF pruning\n", length(final)))
  cat("Kept:", paste(final, collapse = ", "), "\n\n")
  
  return(final)
}

# --- STEP 3: FORWARD SELECTION WITH AIC ---

forward_sarimax_select <- function(train_val_data, target_col, candidate_features,
                                   order, seasonal, period = 168) {
  y <- ts(train_val_data[[target_col]], frequency = period)
  
  cat("BASELINE SARIMA MODEL (no exogenous variables)\n")
  
  baseline_css <- tryCatch(
    Arima(y,
          order = order,
          seasonal = list(order = seasonal, period = period),
          method = "CSS"),
    error = function(e) NULL
  )
  
  if (!is.null(baseline_css)) {
    baseline_fit <- tryCatch(
      Arima(y,
            order = order,
            seasonal = list(order = seasonal, period = period),
            method = "CSS-ML",
            fixed = coef(baseline_css)),
      error = function(e) NULL
    )
  } else {
    baseline_fit <- NULL
  }
  
  if (is.null(baseline_fit)) {
    cat("Baseline SARIMA fit failed.\n")
  } else {
    cat("\nBaseline SARIMA summary:\n")
    cat(sprintf("AIC: %.2f | BIC: %.2f | LogLik: %.2f\n",
                baseline_fit$aic, baseline_fit$bic, baseline_fit$loglik))
    cat("\nCoefficients:\n")
    print((summary(baseline_fit))
  }
  
  baseline_aic <- if (!is.null(baseline_fit)) baseline_fit$aic else Inf
  
  cat("FORWARD SARIMAX FEATURE SELECTION\n")
  
  best_aic <- baseline_aic
  selected <- c()
  best_model <- NULL
  for (feat in candidate_features) {
    feats_try <- c(selected, feat)
    xreg <- scale(as.matrix(train_val_data[, ..feats_try]))
    fit_css <- tryCatch(
      Arima(y, order = order,
            seasonal = list(order = seasonal, period = period),
            xreg = xreg, method = "CSS"),
      error = function(e) NULL
    )
    if (is.null(fit_css)) next
    fit_ml <- tryCatch(
      Arima(y, order = order,
            seasonal = list(order = seasonal, period = period),
            xreg = xreg, method = "CSS-ML", fixed = coef(fit_css)),
      error = function(e) NULL
    )
    fit <- if (!is.null(fit_ml)) fit_ml else fit_css
    aic <- AIC(fit)
    delta <- aic - best_aic
    if (length(selected) == 0 || delta < -2) {
      best_aic <- aic
      selected <- feats_try
      best_model <- fit
      cat(sprintf("Accepted %s | AIC=%.2f\n", feat, aic))
    } else {
      cat(sprintf("Rejected %s | AIC=%.2f (Δ=%.2f)\n", feat, aic, delta))
    }
    print(summary(best_model))
  }
  cat("\nFinal AIC:", round(best_aic, 2), "\n")
  cat("Selected features:", paste(selected, collapse = ", "), "\n\n")
  list(selected = selected, model = best_model, best_aic = best_aic)
}


# --- MAIN EXECUTION ---

start_time <- Sys.time()

# Step 1: Correlation filter
corr_vars <- corr_filter(train_val_data, target_col, corr_min = CORR_MIN)

# Step 2: VIF pruning
vif_vars <- vif_prune(train_val_data, corr_vars, vif_max = VIF_MAX)

# Step 3: Combine with domain features for candidate set
cat("--- Combining Results ---\n")
cat("VIF-filtered:", paste(vif_vars, collapse = ", "), "\n")
cat("Domain features:", paste(domain_features, collapse = ", "), "\n")

candidate_vars <- unique(c(vif_vars, domain_features))
cat(sprintf("Combined candidates: %d features\n\n", length(candidate_vars)))

# Step 4: Forward AIC selection
selection_result <- forward_sarimax_select(
  train_val_data = train_val_data,
  target_col = target_col,
  candidate_features = candidate_vars,
  order = ORDER,
  seasonal = SEASONAL,
  period = PERIOD
)

# --- ENSURE DOMAIN FEATURES ARE INCLUDED ---

cat("--- Finalizing Feature Set ---\n")

# Start with selected features from forward selection
final_features <- selection_result$selected

# Add any missing domain features
missing_domain <- setdiff(domain_features, final_features)

if (length(missing_domain) > 0) {
  cat("Adding domain features not selected by AIC:\n")
  cat(paste("  +", missing_domain, collapse = "\n"), "\n")
  final_features <- c(final_features, missing_domain)
} else {
  cat("All domain features already included\n")
}

selected_xreg <- final_features

cat(sprintf("\nFinal feature set: %d features\n", length(selected_xreg)))
cat(paste(selected_xreg, collapse = ", "), "\n\n")

# --- FINAL SUMMARY ---

total_time <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))

cat(sprintf("Selected features: %d\n", length(selected_xreg)))
cat(sprintf("Final AIC: %.2f\n", selection_result$best_aic))
cat(sprintf("Runtime: %.2f minutes\n\n", total_time))

cat("--- Selected Features ---\n")
cat("selected_xreg <- c(\n")
cat(paste0('  "', selected_xreg, '"', collapse = ",\n"))
