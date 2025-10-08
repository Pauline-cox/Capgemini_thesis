# ==============================================================================
# Feature Selection for SARIMAX (Verbose Version)
# ==============================================================================

sarimax_feature_selection <- function(corr_min = 0.30, vif_max = 5) {
  # Prepare data
  y <- model_data$total_consumption_kWh
  
  exclude_cols <- c(
    "interval", "total_consumption_kWh", "date",
    "occ_co2", "occ_temp",
    "lag_24", "lag_72", "lag_168", "lag_336", "lag_504",
    "rollmean_24", "rollmean_168"
  )
  
  feature_cols <- setdiff(names(model_data), exclude_cols)
  X <- as.matrix(model_data[, ..feature_cols])
  
  cat("\n=== FEATURE SELECTION START ===\n")
  cat("Initial features:", ncol(X), "\n")
  cat("Feature names:", paste(colnames(X), collapse = ", "), "\n\n")
  
  # Step 1: Drop non-finite or constant columns
  nonfinite <- apply(X, 2, function(z) !all(is.finite(z)))
  nzv <- apply(X, 2, function(z) var(z, na.rm = TRUE) <= 1e-12)
  drop0 <- nonfinite | nzv
  
  if (any(drop0)) {
    cat("Dropping", sum(drop0), "non-finite/constant columns:\n  ->",
        paste(colnames(X)[drop0], collapse = ", "), "\n\n")
    X <- X[, !drop0, drop = FALSE]
  } else {
    cat("No non-finite or constant columns found.\n\n")
  }
  
  # Step 2: Correlation filter
  cors <- suppressWarnings(cor(X, y, use = "pairwise.complete.obs"))
  cors <- as.numeric(cors)
  names(cors) <- colnames(X)
  
  keep_idx <- which(!is.na(cors) & abs(cors) >= corr_min)
  cat(sprintf("Correlation filter (|r| >= %.2f): kept %d / %d predictors\n",
              corr_min, length(keep_idx), ncol(X)))
  
  if (length(keep_idx) == 0) {
    cat("No features passed the correlation threshold. Exiting.\n")
    return(list(selected_features = character(), audit = data.frame()))
  }
  
  kept_names_corr <- colnames(X)[keep_idx]
  Xc <- X[, kept_names_corr, drop = FALSE]
  cors_kept <- cors[kept_names_corr]
  
  cat("Kept after correlation filter:\n  ->",
      paste(kept_names_corr, collapse = ", "), "\n\n")
  
  # Step 3: Define VIF function
  vif_vec <- function(M) {
    p <- ncol(M)
    out <- rep(NA_real_, p)
    names(out) <- colnames(M)
    if (p == 1L) { out[] <- 1; return(out) }
    for (j in seq_len(p)) {
      xj <- M[, j]
      Xmj <- M[, -j, drop = FALSE]
      if (ncol(Xmj) == 0L) { out[j] <- 1; next }
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
  
  # Step 4: VIF pruning loop
  Xv <- Xc
  step <- 0L
  repeat {
    vifs <- vif_vec(Xv)
    vmax <- max(vifs, na.rm = TRUE)
    cat(sprintf("VIF iteration %d: max VIF = %.2f\n", step, vmax))
    cat("  -> VIF values:", paste(names(vifs), sprintf("(%.2f)", vifs), collapse = ", "), "\n")
    
    if (!is.finite(vmax) || vmax <= vif_max || ncol(Xv) <= 1L) break
    worst <- names(which.max(vifs))
    cat(sprintf("  Removing '%s' (VIF = %.2f)\n\n", worst, vmax))
    Xv <- Xv[, setdiff(colnames(Xv), worst), drop = FALSE]
    step <- step + 1L
    if (step > 1000) { warning("VIF pruning exceeded 1000 steps; stopping."); break }
  }
  
  final_cols <- colnames(Xv)
  
  # Step 5: Create audit summary
  audit <- data.frame(
    predictor = kept_names_corr,
    corr_with_y = as.numeric(cors_kept[kept_names_corr]),
    kept_after_vif = kept_names_corr %in% final_cols,
    stringsAsFactors = FALSE
  )
  audit <- audit[order(-abs(audit$corr_with_y)), , drop = FALSE]
  
  # Final summary
  cat("\n=== FEATURE SELECTION COMPLETE ===\n")
  cat("Final features:", length(final_cols), "\n")
  cat("Selected predictors:\n  ->", paste(final_cols, collapse = ", "), "\n\n")
  
  cat("Top 10 features by correlation:\n")
  print(head(audit, 10))
  cat("===================================\n\n")
  
  return(list(
    selected_features = final_cols,
    audit = audit
  ))
}
