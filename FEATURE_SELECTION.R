# FEATURE SELECTION PIPELINE FOR SARIMAX

# --- CORRELATION FILTER ---

corr_filter <- function(dt,
                        target_col = "total_consumption_kWh",
                        corr_min = 0.30) {
  
  # Columns to exclude
  exclude_cols <- c(
    "interval", "total_consumption_kWh", "date",
    "occ_co2", "occ_temp",
    "lag_24", "lag_72", "lag_168", "lag_336", "lag_504",
    "rollmean_24", "rollmean_168"
  )
  feature_cols <- setdiff(names(dt), exclude_cols)
  y <- dt[[target_col]]
  
  # Compute correlations
  cors <- sapply(feature_cols, function(v) cor(dt[[v]], y, use = "complete.obs"))
  cors <- sort(cors, decreasing = TRUE)
  
  # Print and filter
  print(round(cors[order(-abs(cors))], 3))
  
  kept <- names(cors[!is.na(cors) & abs(cors) >= corr_min])
  cat(sprintf("\nCorrelation filter (|r| ≥ %.2f): kept %d/%d predictors\n",
              corr_min, length(kept), length(cors)))
  cat("Kept:", paste(kept, collapse = ", "), "\n\n")
  
  return(kept)
}

# --- VIF FILTER (ITERATIVE) ---

vif_prune <- function(dt, vars, vif_max = 5) {
  vif_vec <- function(M) {
    p <- ncol(M)
    out <- rep(NA_real_, p)
    names(out) <- colnames(M)
    if (p == 1L) { out[] <- 1; return(out) }
    for (j in seq_len(p)) {
      xj <- M[, j]
      Xmj <- M[, -j, drop = FALSE]
      df <- data.frame(xj = xj, Xmj, check.names = FALSE)
      fit <- tryCatch(lm(xj ~ ., data = df), error = function(e) NULL)
      if (is.null(fit)) out[j] <- Inf else {
        r2 <- max(0, min(1, summary(fit)$r.squared))
        out[j] <- 1 / (1 - r2 + 1e-12)
      }
    }
    out
  }
  
  Xv <- as.data.frame(dt[, ..vars])
  step <- 0L
  repeat {
    vifs <- vif_vec(Xv)
    vmax <- max(vifs, na.rm = TRUE)
    cat(sprintf("VIF iteration %d: max VIF = %.2f\n", step, vmax))
    cat("  -> VIF values:", paste(names(vifs), sprintf("(%.2f)", vifs), collapse = ", "), "\n")
    
    if (!is.finite(vmax) || vmax <= vif_max || ncol(Xv) <= 1L) break
    worst <- names(which.max(vifs))
    cat(sprintf("  Removing '%s' (VIF = %.2f)\n\n", worst, vmax))
    Xv <- Xv[, setdiff(names(Xv), worst), drop = FALSE]
    step <- step + 1L
  }
  
  final <- names(Xv)
  cat("\nFinal VIF-pruned features:", paste(final, collapse = ", "), "\n")
  return(final)
}


# --- FORWARD AIC-BASED SARIMAX SELECTION (TRAIN + VALIDATION) ---

forward_sarimax_select <- function(train_val_data, target_col, candidate_features,
                                   order = c(2,0,2), seasonal = c(0,1,1), period = 168) {
  
  y <- ts(train_val_data[[target_col]], frequency = period)
  best_aic <- Inf
  selected <- c()
  best_model <- NULL
  aic_log <- data.table(step = integer(), feature = character(), AIC = numeric())
  total_feats <- length(candidate_features)
  
  cat(sprintf("\n Forward AIC selection with %d features\n", total_feats))
  
  for (i in seq_along(candidate_features)) {
    feat <- candidate_features[i]
    feats_try <- c(selected, feat)
    cat(sprintf("\n[%02d/%02d] Testing feature: %s\n", i, total_feats, feat))
    xreg <- as.matrix(train_val_data[, ..feats_try])
    
    # CSS initialization
    fit_css <- tryCatch(
      Arima(y, order = order,
            seasonal = list(order = seasonal, period = period),
            xreg = scale(xreg), method = "CSS"),
      error = function(e) NULL
    )
    if (is.null(fit_css)) {
      next
    }
    
    # Refine with CSS-ML starting from CSS coefficients
    fit_ml <- tryCatch(
      Arima(y, order = order,
            seasonal = list(order = seasonal, period = period),
            xreg = scale(xreg), method = "CSS-ML",
            fixed = coef(fit_css)),
      error = function(e) NULL
    )
    fit <- if (!is.null(fit_ml)) fit_ml else fit_css
    
    aic <- suppressWarnings(AIC(fit))
    if (is.null(aic) || !is.finite(aic)) {
      next
    }
    
    aic_log <- rbind(aic_log, data.table(step = nrow(aic_log) + 1, feature = feat, AIC = aic))
    cat(sprintf("  AIC = %.2f\n", aic))
    
    if (length(selected) == 0) {
      best_aic <- aic
      selected <- feats_try
      best_model <- fit
      cat("  Accepted first feature:", feat, "\n")
    } else if (aic + 2 < best_aic) {
      best_aic <- aic
      selected <- feats_try
      best_model <- fit
      cat("  Accepted:", feat, "| new best AIC =", round(best_aic, 2), "\n")
    } else {
      cat("  Rejected:", feat, "\n")
    }
  }
  
  cat("\n Forward selection complete.\n")
  cat("Final selected predictors:", paste(selected, collapse = ", "), "\n\n")
  
  return(list(selected = selected, model = best_model, aic_log = aic_log))
}


# Example main usage
sarimax_feature_selection <- function(model_data, domain_features, corr_min = 0.30, vif_max = 5) {
  target_col <- "total_consumption_kWh"
  
  # Split your dataset (Train + Validation only, exclude test)
  train_val <- model_data[as.Date(interval) < as.Date("2024-10-01")]  # adjust cutoff
  y <- train_val[[target_col]]
  
  cat("=== STEP 1: Correlation filter ===\n")
  corr_vars <- corr_filter(train_val, target_col, corr_min)
  
  cat("\n=== STEP 2: VIF pruning ===\n")
  vif_vars <- vif_prune(train_val, corr_vars, vif_max)
  
  cat("\n=== STEP 3: Forward SARIMAX selection ===\n")
  candidate_vars <- unique(c(vif_vars, domain_features))
  sel <- forward_sarimax_select(train_val, target_col, candidate_vars,
                                order = c(2,0,2), seasonal = c(0,1,1), period = 168)
  
  return(sel$selected)
}