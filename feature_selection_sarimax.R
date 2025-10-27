# ==============================================================
# Author: Pauline Cox
# Script: feature_selection_sarimax.R
#
# Description: Implements a three-stage feature selection pipeline
# for SARIMAX models, combining correlation filtering, VIF pruning,
# and forward stepwise selection based on AIC. Domain-relevant 
# predictors (holidays, business hours, DST) are always included.
#
# Input:
#   - Preprocessed and feature-engineered dataset (model_data)
#
# Output:
#   - Final selected feature set
#   - Diagnostic summaries for each selection stage
# ==============================================================

set.seed(1234)

# --- Settings ---
target_col <- "total_consumption_kWh"
ORDER    <- c(1, 0, 2)
SEASONAL <- c(1, 1, 1)
PERIOD   <- 168L

CORR_MIN <- 0.30
VIF_MAX  <- 5

domain_features <- c("business_hours", "holiday", "dst")

# --- Data Preparation ---
tz_ref <- attr(model_data$interval, "tzone")
train_val_data <- model_data[
  interval >= as.POSIXct("2023-07-01 00:00:00", tz = tz_ref) &
    interval <  as.POSIXct("2024-10-01 00:00:00", tz = tz_ref)
]
cat(sprintf("Training/Validation data: %d samples (%s to %s)\n\n",
            nrow(train_val_data),
            min(as.Date(train_val_data$interval)),
            max(as.Date(train_val_data$interval))))


# --- STEP 1: Correlation Filter ---

corr_filter <- function(dt, target_col, corr_min = 0.30) {
  exclude_cols <- c(
    "interval", "total_consumption_kWh", "date",
    "occ_co2", "occ_temp",
    "lag_24", "lag_48", "lag_72", "lag_168", "lag_336", "lag_504",
    "rollmean_24", "rollmean_168"
  )
  feature_cols <- setdiff(names(dt), exclude_cols)
  y <- dt[[target_col]]
  
  cors <- sapply(feature_cols, function(v) cor(dt[[v]], y, use = "complete.obs"))
  cors_sorted <- cors[order(-abs(cors))]
  kept <- names(cors_sorted[!is.na(cors_sorted) & abs(cors_sorted) >= corr_min])
  
  cat(sprintf("Correlation threshold |r| >= %.2f: retained %d features\n\n",
              corr_min, length(kept)))
  return(kept)
}


# --- STEP 2: VIF Pruning ---

vif_prune <- function(dt, vars, vif_max = 5) {
  vif_vec <- function(M) {
    p <- ncol(M); out <- rep(NA_real_, p); names(out) <- colnames(M)
    if (p == 1L) { out[] <- 1; return(out) }
    for (j in seq_len(p)) {
      xj <- M[, j]; Xmj <- M[, -j, drop = FALSE]
      df <- data.frame(xj = xj, Xmj, check.names = FALSE)
      fit <- tryCatch(lm(xj ~ ., data = df), error = function(e) NULL)
      r2 <- if (!is.null(fit)) max(0, min(1, summary(fit)$r.squared)) else 1
      out[j] <- 1 / (1 - r2 + 1e-12)
    }
    out
  }
  
  Xv <- as.data.frame(dt[, ..vars])
  repeat {
    vifs <- vif_vec(Xv)
    vmax <- max(vifs, na.rm = TRUE)
    if (!is.finite(vmax) || vmax <= vif_max || ncol(Xv) <= 1L) break
    worst <- names(which.max(vifs))
    Xv <- Xv[, setdiff(names(Xv), worst), drop = FALSE]
  }
  final <- names(Xv)
  cat(sprintf("VIF pruning (VIF ≤ %.1f): retained %d features\n\n", vif_max, length(final)))
  return(final)
}


# --- STEP 3: Forward Selection ---

forward_sarimax_select <- function(train_val_data, target_col, candidate_features,
                                   order, seasonal, period = 168) {
  y <- ts(train_val_data[[target_col]], frequency = period)
  
  baseline_fit <- tryCatch(
    Arima(y,
          order = order,
          seasonal = list(order = seasonal, period = period),
          method = "CSS-ML"),
    error = function(e) NULL
  )
  
  baseline_aic <- if (!is.null(baseline_fit)) baseline_fit$aic else Inf
  best_aic <- baseline_aic
  selected <- c()
  best_model <- baseline_fit
  
  for (feat in candidate_features) {
    feats_try <- c(selected, feat)
    xreg <- scale(as.matrix(train_val_data[, ..feats_try]))
    fit_css <- tryCatch(
      Arima(y, order = order,
            seasonal = list(order = seasonal, period = period),
            xreg = xreg, method = "CSS-ML"),
      error = function(e) NULL
    )
    if (is.null(fit_css)) next
    aic <- AIC(fit_css)
    delta <- aic - best_aic
    if (length(selected) == 0 || delta < -2) {
      best_aic <- aic
      selected <- feats_try
      best_model <- fit_css
    }
  }
  
  cat(sprintf("Forward selection complete - Final AIC: %.2f\n", best_aic))
  cat("Selected features:\n")
  cat(paste(selected, collapse = ", "), "\n\n")
  
  list(selected = selected, model = best_model, best_aic = best_aic)
}

# --- Main execution ---

start_time <- Sys.time()

corr_vars <- corr_filter(train_val_data, target_col, corr_min = CORR_MIN)
vif_vars  <- vif_prune(train_val_data, corr_vars, vif_max = VIF_MAX)

candidate_vars <- unique(c(vif_vars, domain_features))
cat(sprintf("Combined candidate features: %d\n\n", length(candidate_vars)))

selection_result <- forward_sarimax_select(
  train_val_data = train_val_data,
  target_col = target_col,
  candidate_features = candidate_vars,
  order = ORDER,
  seasonal = SEASONAL,
  period = PERIOD
)

# --- Ensure domain features are included ---
final_features <- selection_result$selected
missing_domain <- setdiff(domain_features, final_features)
final_features <- c(final_features, missing_domain)

selected_xreg <- unique(final_features)
cat(sprintf("Final feature set: %d features\n", length(selected_xreg)))
cat(paste(selected_xreg, collapse = ", "), "\n\n")

# --- Summary ---
total_time <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
cat(sprintf("Feature selection completed in %.2f minutes.\n", total_time))
cat(sprintf("Final AIC: %.2f\n\n", selection_result$best_aic))

cat("selected_xreg <- c(\n")
cat(paste0('  "', selected_xreg, '"', collapse = ",\n"))
cat(")\n")
