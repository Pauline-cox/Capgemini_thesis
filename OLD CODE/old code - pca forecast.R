# ===============================================================
# PCA-BASED SARIMAX ROLLING WINDOW FORECAST
# Using ALL Environmental Variables - 24H AHEAD FORECAST
# ===============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(forecast)
  library(ggplot2)
})

set.seed(1234)

# --- SETTINGS ---

ORDER    <- c(2, 0, 2)
SEASONAL <- c(0, 1, 1)
PERIOD   <- 168L
target_col <- "total_consumption_kWh"

# PCA settings
PCA_VAR_THRESHOLD <- 0.75  # Retain 75% of variance

# ALL environmental variables (testing if combined info helps)
env_vars <- c(
  # Indoor environmental
  "tempC", "humidity", "co2", "sound", "lux", "total_occupancy",
  
  # Outdoor weather  
  "temperature", "wind_speed", "sunshine_minutes", 
  "global_radiation", "humidity_percent",
  
  # Weather conditions
  "fog", "rain", "snow", "thunder", "ice"
)

# Selected temporal features (from feature selection results)
temporal_vars <- c("business_hours", "hour_sin", "hour_cos", "holiday", "dst")

# Test periods
testA_start <- as.Date("2024-10-01")
testA_end   <- as.Date("2024-10-14")
trainA_end  <- testA_start - 1

testB_start <- as.Date("2024-12-18")
testB_end   <- as.Date("2024-12-31")
trainB_end  <- testB_start - 1

# --- HELPER FUNCTIONS ---

mape <- function(actual, pred) {
  mean(abs((actual - pred) / pmax(actual, 1e-6)), na.rm = TRUE) * 100
}

evaluate_forecast <- function(actual, pred, model_name) {
  rmse <- sqrt(mean((pred - actual)^2, na.rm = TRUE))
  mae  <- mean(abs(pred - actual), na.rm = TRUE)
  r2   <- 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
  map  <- mape(actual, pred)
  data.table(Model = model_name, RMSE = rmse, MAE = mae, R2 = r2, MAPE = map)
}

# --- PCA FEATURE EXTRACTION ---

extract_pca_features <- function(train_data, test_data, var_threshold = 0.90) {
  
  cat("\n--- PCA Feature Extraction ---\n")
  cat(sprintf("Using %d environmental variables\n", length(env_vars)))
  
  # Keep only available variables
  available_env <- intersect(env_vars, names(train_data))
  available_temp <- intersect(temporal_vars, names(train_data))
  
  cat(sprintf("Available: %d environmental + %d temporal\n", 
              length(available_env), length(available_temp)))
  cat("Environmental:", paste(available_env, collapse = ", "), "\n\n")
  
  # Prepare environmental data
  X_train_env <- as.matrix(train_data[, ..available_env])
  X_test_env <- as.matrix(test_data[, ..available_env])
  
  # Handle missing values
  for (i in seq_len(ncol(X_train_env))) {
    mu <- mean(X_train_env[, i], na.rm = TRUE)
    X_train_env[is.na(X_train_env[, i]), i] <- mu
    X_test_env[is.na(X_test_env[, i]), i] <- mu
  }
  
  # Run PCA
  cat("Running PCA on environmental variables...\n")
  pca_model <- prcomp(X_train_env, center = TRUE, scale. = TRUE)
  
  # Variance explained
  var_explained <- pca_model$sdev^2 / sum(pca_model$sdev^2)
  cum_var <- cumsum(var_explained)
  
  # Number of components for threshold
  n_components <- which(cum_var >= var_threshold)[1]
  if (is.na(n_components)) n_components <- length(available_env)
  
  cat(sprintf("\nComponents for %.0f%% variance: %d/%d\n", 
              var_threshold * 100, n_components, length(available_env)))
  
  # Show component contributions
  cat("\nVariance explained by components:\n")
  for (i in 1:min(n_components, 5)) {
    cat(sprintf("  PC%d: %.1f%% (cumulative: %.1f%%)\n", 
                i, var_explained[i] * 100, cum_var[i] * 100))
  }
  
  # Show top loadings for PC1
  cat("\nTop 5 loadings for PC1 (main environmental pattern):\n")
  loadings_pc1 <- pca_model$rotation[, 1]
  top_loadings_idx <- order(abs(loadings_pc1), decreasing = TRUE)[1:5]
  for (idx in top_loadings_idx) {
    cat(sprintf("  %s: %.3f\n", rownames(pca_model$rotation)[idx], 
                loadings_pc1[idx]))
  }
  cat("\n")
  
  # Transform data
  train_pcs <- predict(pca_model, X_train_env)[, 1:n_components, drop = FALSE]
  test_pcs <- predict(pca_model, X_test_env)[, 1:n_components, drop = FALSE]
  colnames(train_pcs) <- paste0("PC", 1:n_components)
  colnames(test_pcs) <- paste0("PC", 1:n_components)
  
  # Combine PCs with temporal features
  train_features <- cbind(
    as.data.table(train_pcs),
    train_data[, ..available_temp]
  )
  
  test_features <- cbind(
    as.data.table(test_pcs),
    test_data[, ..available_temp]
  )
  
  cat(sprintf("Final features: %d PCs + %d temporal = %d total\n",
              n_components, length(available_temp), ncol(train_features)))
  cat("Features:", paste(names(train_features), collapse = ", "), "\n\n")
  
  list(
    train = train_features,
    test = test_features,
    pca_model = pca_model,
    n_components = n_components,
    variance_explained = cum_var[n_components],
    feature_names = names(train_features),
    var_explained_individual = var_explained[1:n_components],
    loadings_pc1 = loadings_pc1
  )
}

# --- ROLLING WINDOW SARIMAX WITH PCA (24H AHEAD) ---

rolling_sarimax_pca <- function(train_data, test_data, pca_features, 
                                order, seasonal, period) {
  
  cat("\n>>> Starting PCA-SARIMAX Rolling Window Forecast (24h ahead)...\n")
  cat("Strategy: Start 23 hours before test to produce aligned 24h forecasts\n\n")
  
  n_test <- nrow(test_data)
  forecasts <- rep(NA_real_, n_test)
  
  overall_start <- Sys.time()
  
  # --- STEP 1: TRAIN MODEL ONCE ---
  
  cat("--- Step 1: Training SARIMAX-PCA model ---\n")
  train_start <- Sys.time()
  
  y_train <- train_data[[target_col]]
  x_train <- as.matrix(pca_features$train)
  
  # Scale features
  x_train_scaled <- scale(x_train)
  center_vec <- attr(x_train_scaled, "scaled:center")
  scale_vec <- attr(x_train_scaled, "scaled:scale")
  
  y_train_ts <- ts(y_train, frequency = period)
  
  cat(sprintf("Training samples: %d\n", length(y_train)))
  cat(sprintf("PCA components: %d (%.1f%% variance)\n", 
              pca_features$n_components, 
              pca_features$variance_explained * 100))
  cat(sprintf("Total features: %d\n", ncol(x_train)))
  
  sarimax_model <- tryCatch(
    Arima(y_train_ts,
          order = order,
          seasonal = list(order = seasonal, period = period),
          xreg = x_train_scaled,
          method = "CSS"),
    error = function(e) {
      cat("ERROR: Model training failed\n")
      return(NULL)
    }
  )
  
  if (is.null(sarimax_model)) {
    stop("SARIMAX-PCA model training failed")
  }
  
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  cat(sprintf("Training complete! Time: %.2f minutes\n", train_time))
  cat(sprintf("AIC: %.2f | BIC: %.2f\n\n", sarimax_model$aic, sarimax_model$bic))
  
  # --- STEP 2: ROLLING PREDICTIONS (Starting 23h before test) ---
  
  cat("--- Step 2: Rolling 24h-ahead predictions ---\n")
  predict_start <- Sys.time()
  
  # Combine train + test
  all_y <- c(y_train, test_data[[target_col]])
  all_x <- rbind(x_train, as.matrix(pca_features$test))
  
  # Scale all features using training statistics
  all_x_scaled <- scale(all_x, center = center_vec, scale = scale_vec)
  
  train_size <- length(y_train)
  
  # Start from h = -22 (23 hours before test) to get first forecast at test hour 1
  # Loop through until we've forecasted all test hours
  for (h in seq(-22, n_test)) {
    
    # Progress indicator
    if (h %% 24 == 0 || h == -22 || h == 1) {
      elapsed <- as.numeric(difftime(Sys.time(), predict_start, units = "secs"))
      if (h > 0) {
        pct_complete <- (h / n_test) * 100
        cat(sprintf("  Hour %d/%d (%.1f%%) | Elapsed: %.1fs\n",
                    h, n_test, pct_complete, elapsed))
      } else {
        cat(sprintf("  Pre-roll hour %d | Elapsed: %.1fs\n", h, elapsed))
      }
    }
    
    # Get historical data up to current point
    current_idx <- train_size + h - 1
    
    # Skip if we don't have enough history
    if (current_idx < 100) next
    
    history_y <- all_y[1:current_idx]
    history_x <- all_x_scaled[1:current_idx, , drop = FALSE]
    
    # Get future 24 hours of features
    future_start <- current_idx + 1
    future_end <- min(current_idx + 24, nrow(all_x_scaled))
    future_x <- all_x_scaled[future_start:future_end, , drop = FALSE]
    
    # Need full 24 hours of future features
    if (nrow(future_x) < 24) next
    
    # Update model
    y_ts <- ts(history_y, frequency = period)
    
    updated_model <- tryCatch(
      Arima(y_ts,
            model = sarimax_model,
            xreg = history_x),
      error = function(e) NULL
    )
    
    if (is.null(updated_model)) next
    
    # Forecast 24 hours ahead
    fc <- forecast(updated_model, xreg = future_x, h = 24)
    
    # Calculate which test hour this 24h-ahead forecast corresponds to
    forecast_hour <- current_idx + 24 - train_size
    
    # Store if it falls within the test period
    if (forecast_hour >= 1 && forecast_hour <= n_test) {
      forecasts[forecast_hour] <- fc$mean[24]
    }
  }
  
  predict_time <- as.numeric(difftime(Sys.time(), predict_start, units = "mins"))
  total_time <- as.numeric(difftime(Sys.time(), overall_start, units = "mins"))
  
  cat(sprintf("\nPrediction complete! Time: %.2f minutes\n", predict_time))
  cat(sprintf("Total runtime: %.2f minutes (Train: %.2f + Predict: %.2f)\n\n",
              total_time, train_time, predict_time))
  
  # Check for any missing forecasts
  n_missing <- sum(is.na(forecasts))
  if (n_missing > 0) {
    cat(sprintf("Warning: %d forecasts are missing (will use last observed value)\n", n_missing))
    # Fill any remaining NAs with last observation
    for (i in which(is.na(forecasts))) {
      if (i == 1) {
        forecasts[i] <- tail(y_train, 1)
      } else {
        forecasts[i] <- forecasts[i-1]
      }
    }
  }
  
  return(list(
    forecasts = forecasts,
    runtime = total_time,
    train_time = train_time,
    predict_time = predict_time,
    model = sarimax_model
  ))
}

# --- RUN FOR BOTH TEST PERIODS ---

run_pca_sarimax <- function(train, test, period_label) {
  
  cat(sprintf("\n==================== %s ====================\n", period_label))
  
  # Extract PCA features
  pca_result <- extract_pca_features(train, test, var_threshold = PCA_VAR_THRESHOLD)
  
  # Run forecast
  forecast_result <- rolling_sarimax_pca(
    train_data = train,
    test_data = test,
    pca_features = pca_result,
    order = ORDER,
    seasonal = SEASONAL,
    period = PERIOD
  )
  
  # Evaluate
  actual <- test[[target_col]]
  
  eval_result <- evaluate_forecast(actual, forecast_result$forecasts, "SARIMAX_PCA_24h")
  eval_result[, Runtime_min := forecast_result$runtime]
  eval_result[, Train_min := forecast_result$train_time]
  eval_result[, Predict_min := forecast_result$predict_time]
  eval_result[, Period := period_label]
  eval_result[, N_Components := pca_result$n_components]
  eval_result[, Var_Explained := pca_result$variance_explained]
  
  cat("--- Evaluation Results (24h-ahead) ---\n")
  print(eval_result)
  cat("\n")
  
  # Create forecast dataframe
  forecast_dt <- data.table(
    Time = seq_along(actual),
    Actual = actual,
    SARIMAX_PCA_24h = forecast_result$forecasts
  )
  
  # Plot results
  p <- ggplot(forecast_dt, aes(x = Time)) +
    geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
    geom_line(aes(y = SARIMAX_PCA_24h, color = "SARIMAX-PCA (24h ahead)"), 
              linewidth = 0.7, alpha = 0.8) +
    scale_color_manual(values = c("Actual" = "black", 
                                  "SARIMAX-PCA (24h ahead)" = "purple")) +
    labs(
      title = paste("SARIMAX-PCA 24h-Ahead Forecast -", period_label),
      subtitle = sprintf("%d PCs from %d env vars (%.1f%% variance)", 
                         pca_result$n_components,
                         length(intersect(env_vars, names(train))),
                         pca_result$variance_explained * 100),
      x = "Time (hours)",
      y = "Energy Consumption (kWh)",
      color = "Series"
    ) +
    theme_minimal(base_size = 12)
  
  print(p)
  
  return(list(
    eval = eval_result,
    forecasts = forecast_dt,
    plot = p,
    pca_info = pca_result,
    model = forecast_result$model
  ))
}

# --- PREPARE DATA ---

cat("\n========== PREPARING DATA ==========\n")

trainA <- model_data[as.Date(interval) <= trainA_end]
testA  <- model_data[as.Date(interval) >= testA_start & as.Date(interval) <= testA_end]

trainB <- model_data[as.Date(interval) <= trainB_end]
testB  <- model_data[as.Date(interval) >= testB_start & as.Date(interval) <= testB_end]

cat(sprintf("Period A - Train: %d | Test: %d\n", nrow(trainA), nrow(testA)))
cat(sprintf("Period B - Train: %d | Test: %d\n", nrow(trainB), nrow(testB)))

# --- MAIN EXECUTION ---

cat("\n========== PCA-BASED SARIMAX FORECASTING (24H AHEAD) ==========\n")
cat("Research Question: Does capturing full environmental context\n")
cat("through PCA improve 24h-ahead forecasts vs individually-selected features?\n")

resultsA_pca <- run_pca_sarimax(trainA, testA, "Period A (Stable)")
resultsB_pca <- run_pca_sarimax(trainB, testB, "Period B (Not Stable)")

# --- FINAL SUMMARY ---

cat("\n==================== FINAL SUMMARY ====================\n")

all_eval_pca <- rbind(resultsA_pca$eval, resultsB_pca$eval)
print(all_eval_pca)

cat("\n--- PCA Analysis ---\n")
cat(sprintf("Period A: %d components (%.1f%% variance)\n",
            resultsA_pca$pca_info$n_components,
            resultsA_pca$pca_info$variance_explained * 100))

cat("\nIndividual component contributions (Period A):\n")
for (i in 1:resultsA_pca$pca_info$n_components) {
  cat(sprintf("  PC%d: %.1f%%\n", i, 
              resultsA_pca$pca_info$var_explained_individual[i] * 100))
}

cat(sprintf("\nPeriod B: %d components (%.1f%% variance)\n",
            resultsB_pca$pca_info$n_components,
            resultsB_pca$pca_info$variance_explained * 100))

cat("\n--- Performance Comparison (24h-ahead) ---\n")
cat(sprintf("Period A: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f\n",
            resultsA_pca$eval$RMSE,
            resultsA_pca$eval$MAE,
            resultsA_pca$eval$MAPE,
            resultsA_pca$eval$R2))

cat(sprintf("Period B: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f\n",
            resultsB_pca$eval$RMSE,
            resultsB_pca$eval$MAE,
            resultsB_pca$eval$MAPE,
            resultsB_pca$eval$R2))

# --- INTERPRETATION ---

cat("\n--- Interpretation Guide ---\n")
cat("Compare these 24h-ahead results with other approaches:\n")
cat("  - If PCA performs BETTER: Combined environmental patterns add value\n")
cat("  - If PCA performs SIMILAR: Different representation, same information\n")
cat("  - If PCA performs WORSE: Individual features sufficient\n\n")

cat("Key insight: Check PC1 loadings to understand main environmental pattern\n")

# --- SAVE RESULTS ---

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
results_file <- sprintf("SARIMAX_PCA_24h_Results_%s.rds", timestamp)

pca_sarimax_results <- list(
  period_A = resultsA_pca,
  period_B = resultsB_pca,
  evaluations = all_eval_pca,
  parameters = list(
    order = ORDER,
    seasonal = SEASONAL,
    period = PERIOD,
    pca_var_threshold = PCA_VAR_THRESHOLD,
    env_vars_used = env_vars,
    temporal_vars_used = temporal_vars,
    forecast_horizon = 24
  )
)

saveRDS(pca_sarimax_results, file = results_file)
cat(sprintf("\nResults saved to: %s\n", results_file))

cat("\nPCA-SARIMAX 24h-ahead pipeline complete!\n")