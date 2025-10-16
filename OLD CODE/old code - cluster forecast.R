# ===============================================================
# CLUSTERING-BASED SARIMAX ROLLING WINDOW FORECAST (FIXED)
# Using ALL Environmental Variables - 24H AHEAD
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

# Clustering settings
N_CLUSTERS <- 3  # Number of environmental regimes

# ALL environmental variables (testing if regimes help)
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

# --- ENVIRONMENTAL CLUSTERING ---

create_env_clusters <- function(train_data, n_clusters = 3) {
  
  cat("\n--- Environmental Clustering ---\n")
  cat(sprintf("Using %d environmental variables\n", length(env_vars)))
  
  # Keep only available variables
  available_env <- intersect(env_vars, names(train_data))
  
  cat(sprintf("Available: %d environmental variables\n", length(available_env)))
  cat("Variables:", paste(available_env, collapse = ", "), "\n")
  cat(sprintf("Number of clusters: %d\n\n", n_clusters))
  
  # Prepare data
  X_env <- as.matrix(train_data[, ..available_env])
  
  # Handle missing values
  for (i in seq_len(ncol(X_env))) {
    mu <- mean(X_env[, i], na.rm = TRUE)
    X_env[is.na(X_env[, i]), i] <- mu
  }
  
  # Scale and cluster
  X_scaled <- scale(X_env)
  km <- kmeans(X_scaled, centers = n_clusters, nstart = 25)
  
  cat("Cluster sizes:\n")
  cluster_table <- table(km$cluster)
  print(cluster_table)
  cat(sprintf("Cluster proportions: %s\n\n", 
              paste(sprintf("%.1f%%", prop.table(cluster_table) * 100), 
                    collapse = ", ")))
  
  # Interpret clusters - unscaled centers
  centers_unscaled <- km$centers * attr(X_scaled, "scaled:scale") + 
    attr(X_scaled, "scaled:center")
  
  cat("Cluster centers (selected key variables):\n")
  key_vars <- intersect(c("temperature", "co2", "total_occupancy", 
                          "lux", "global_radiation", "humidity"), 
                        available_env)
  if (length(key_vars) > 0) {
    print(round(centers_unscaled[, key_vars, drop = FALSE], 2))
  }
  cat("\n")
  
  # Identify key differentiating variables
  cat("Top 5 variables differentiating clusters (highest variance):\n")
  cluster_variance <- sapply(as.data.frame(centers_unscaled), var, na.rm = TRUE)
  
  top_vars <- sort(cluster_variance, decreasing = TRUE)[1:min(5, length(cluster_variance))]
  for (var in names(top_vars)) {
    cat(sprintf("  %s: variance = %.2f\n", var, top_vars[var]))
  }
  cat("\n")
  
  # Interpret clusters (simple heuristic)
  cat("Cluster interpretation (heuristic):\n")
  for (k in 1:n_clusters) {
    temp_val <- if ("temperature" %in% available_env) centers_unscaled[k, "temperature"] else NA
    occ_val <- if ("total_occupancy" %in% available_env) centers_unscaled[k, "total_occupancy"] else NA
    rad_val <- if ("global_radiation" %in% available_env) centers_unscaled[k, "global_radiation"] else NA
    
    interpretation <- sprintf("Cluster %d (n=%d): ", k, cluster_table[k])
    if (!is.na(temp_val)) interpretation <- paste0(interpretation, sprintf("Temp=%.1f°C ", temp_val))
    if (!is.na(occ_val)) interpretation <- paste0(interpretation, sprintf("Occ=%.0f ", occ_val))
    if (!is.na(rad_val)) interpretation <- paste0(interpretation, sprintf("Rad=%.0f", rad_val))
    
    cat(interpretation, "\n")
  }
  cat("\n")
  
  list(
    clusters = km$cluster,
    centers_scaled = km$centers,
    centers_unscaled = centers_unscaled,
    env_vars = available_env,
    n_clusters = n_clusters,
    cluster_sizes = as.numeric(cluster_table),
    top_differentiating_vars = names(top_vars)
  )
}

assign_env_clusters <- function(test_data, train_data, cluster_obj) {
  
  env_vars <- cluster_obj$env_vars
  
  # Prepare data
  X_train <- as.matrix(train_data[, ..env_vars])
  X_test <- as.matrix(test_data[, ..env_vars])
  
  # Handle missing values using training means
  for (i in seq_len(ncol(X_test))) {
    mu <- mean(X_train[, i], na.rm = TRUE)
    X_test[is.na(X_test[, i]), i] <- mu
  }
  
  # Scale using training statistics
  X_train_scaled <- scale(X_train)
  center_vec <- attr(X_train_scaled, "scaled:center")
  scale_vec <- attr(X_train_scaled, "scaled:scale")
  X_test_scaled <- scale(X_test, center = center_vec, scale = scale_vec)
  
  # Assign to nearest cluster
  centers <- cluster_obj$centers_scaled
  cluster_ids <- apply(X_test_scaled, 1, function(x) {
    dists <- colSums((t(centers) - x)^2)
    which.min(dists)
  })
  
  return(cluster_ids)
}

# --- CREATE CLUSTER FEATURES ---

create_cluster_features <- function(train_data, test_data, cluster_obj) {
  
  cat("--- Creating Cluster Features ---\n")
  
  # Add cluster assignments
  train_dt <- copy(train_data)
  test_dt <- copy(test_data)
  
  train_dt[, env_cluster := cluster_obj$clusters]
  test_dt[, env_cluster := assign_env_clusters(test_dt, train_dt, cluster_obj)]
  
  cat("Test set cluster distribution:\n")
  test_cluster_table <- table(test_dt$env_cluster)
  print(test_cluster_table)
  cat(sprintf("Test proportions: %s\n\n", 
              paste(sprintf("%.1f%%", prop.table(test_cluster_table) * 100), 
                    collapse = ", ")))
  
  # Create one-hot encoding for clusters (DROP LAST CLUSTER to avoid multicollinearity)
  for (k in 1:(cluster_obj$n_clusters - 1)) {  # ← CHANGED: Use n_clusters - 1
    var_name <- paste0("cluster_", k)
    train_dt[, (var_name) := as.integer(env_cluster == k)]
    test_dt[, (var_name) := as.integer(env_cluster == k)]
  }
  
  # Add temporal features
  available_temp <- intersect(temporal_vars, names(train_data))
  
  # Combine cluster indicators with temporal features
  cluster_vars <- paste0("cluster_", 1:(cluster_obj$n_clusters - 1))  # ← CHANGED
  feature_vars <- c(cluster_vars, available_temp)
  
  cat(sprintf("Final: %d cluster indicators + %d temporal = %d features\n",
              cluster_obj$n_clusters - 1, length(available_temp), length(feature_vars)))  # ← CHANGED
  cat("Features:", paste(feature_vars, collapse = ", "), "\n")
  cat("Note: Dropped cluster_%d as reference category to avoid multicollinearity\n\n",
      cluster_obj$n_clusters)
  
  list(
    train = train_dt[, ..feature_vars],
    test = test_dt[, ..feature_vars],
    feature_names = feature_vars,
    train_clusters = train_dt$env_cluster,
    test_clusters = test_dt$env_cluster
  )
}
# --- ROLLING WINDOW SARIMAX WITH CLUSTERING (24H AHEAD - FIXED) ---

rolling_sarimax_cluster <- function(train_data, test_data, cluster_features,
                                    order, seasonal, period) {
  
  cat("\n>>> Starting Cluster-SARIMAX 24h-Ahead Rolling Window Forecast...\n")
  cat("Strategy: Start 23 hours before test to produce aligned 24h forecasts\n\n")
  
  n_test <- nrow(test_data)
  forecasts <- rep(NA_real_, n_test)
  
  overall_start <- Sys.time()
  
  # --- STEP 1: TRAIN MODEL ONCE ---
  
  cat("--- Step 1: Training SARIMAX-Cluster model ---\n")
  train_start <- Sys.time()
  
  y_train <- train_data[[target_col]]
  x_train <- as.matrix(cluster_features$train)
  
  # Scale features
  x_train_scaled <- scale(x_train)
  center_vec <- attr(x_train_scaled, "scaled:center")
  scale_vec <- attr(x_train_scaled, "scaled:scale")
  
  y_train_ts <- ts(y_train, frequency = period)
  
  cat(sprintf("Training samples: %d\n", length(y_train)))
  cat(sprintf("Features: %d\n", ncol(x_train)))
  
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
    stop("SARIMAX-Cluster model training failed")
  }
  
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  cat(sprintf("Training complete! Time: %.2f minutes\n", train_time))
  cat(sprintf("AIC: %.2f | BIC: %.2f\n\n", sarimax_model$aic, sarimax_model$bic))
  
  # --- STEP 2: ROLLING PREDICTIONS (Starting 23h before test) ---
  
  cat("--- Step 2: Rolling 24h-ahead predictions ---\n")
  predict_start <- Sys.time()
  
  # Combine train + test
  all_y <- c(y_train, test_data[[target_col]])
  all_x <- rbind(x_train, as.matrix(cluster_features$test))
  
  # Scale all features
  all_x_scaled <- scale(all_x, center = center_vec, scale = scale_vec)
  
  train_size <- length(y_train)
  
  # Start from h = -22 (23 hours before test) to get aligned 24h-ahead forecasts
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
    
    # Current position in combined data
    current_idx <- train_size + h - 1
    
    # Skip if not enough history
    if (current_idx < 100) next
    
    # Get historical data
    history_y <- all_y[1:current_idx]
    history_x <- all_x_scaled[1:current_idx, , drop = FALSE]
    
    # Get future 24 hours
    future_start <- current_idx + 1
    future_end <- min(current_idx + 24, nrow(all_x_scaled))
    
    if (future_end - future_start + 1 < 24) next
    
    future_x <- all_x_scaled[future_start:future_end, , drop = FALSE]
    
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
    fc <- forecast(updated_model, xreg = future_x[1:24, , drop = FALSE], h = 24)
    
    # Calculate which test hour this 24h-ahead forecast is for
    forecast_hour <- current_idx + 24 - train_size
    
    # Store if within test period
    if (forecast_hour >= 1 && forecast_hour <= n_test) {
      forecasts[forecast_hour] <- fc$mean[24]
    }
  }
  
  predict_time <- as.numeric(difftime(Sys.time(), predict_start, units = "mins"))
  total_time <- as.numeric(difftime(Sys.time(), overall_start, units = "mins"))
  
  cat(sprintf("\nPrediction complete! Time: %.2f minutes\n", predict_time))
  cat(sprintf("Total runtime: %.2f minutes (Train: %.2f + Predict: %.2f)\n\n",
              total_time, train_time, predict_time))
  
  # Fill any remaining NAs
  n_missing <- sum(is.na(forecasts))
  if (n_missing > 0) {
    cat(sprintf("Warning: %d forecasts are missing (will use last observed value)\n", n_missing))
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

run_cluster_sarimax <- function(train, test, period_label) {
  
  cat(sprintf("\n==================== %s ====================\n", period_label))
  
  # Create environmental clusters
  cluster_obj <- create_env_clusters(train, n_clusters = N_CLUSTERS)
  
  # Create cluster features
  cluster_features <- create_cluster_features(train, test, cluster_obj)
  
  # Run forecast
  forecast_result <- rolling_sarimax_cluster(
    train_data = train,
    test_data = test,
    cluster_features = cluster_features,
    order = ORDER,
    seasonal = SEASONAL,
    period = PERIOD
  )
  
  # Evaluate
  actual <- test[[target_col]]
  
  eval_result <- evaluate_forecast(actual, forecast_result$forecasts, "SARIMAX_Cluster_24h")
  eval_result[, Runtime_min := forecast_result$runtime]
  eval_result[, Train_min := forecast_result$train_time]
  eval_result[, Predict_min := forecast_result$predict_time]
  eval_result[, Period := period_label]
  eval_result[, N_Clusters := cluster_obj$n_clusters]
  
  cat("--- Evaluation Results (24h-ahead) ---\n")
  print(eval_result)
  cat("\n")
  
  # Create forecast dataframe
  forecast_dt <- data.table(
    Time = seq_along(actual),
    Actual = actual,
    SARIMAX_Cluster_24h = forecast_result$forecasts,
    Cluster = cluster_features$test_clusters
  )
  
  # Plot results
  p <- ggplot(forecast_dt, aes(x = Time)) +
    geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
    geom_line(aes(y = SARIMAX_Cluster_24h, color = "SARIMAX-Cluster (24h)"), 
              linewidth = 0.7, alpha = 0.8) +
    scale_color_manual(values = c("Actual" = "black", "SARIMAX-Cluster (24h)" = "orange")) +
    labs(
      title = paste("SARIMAX-Cluster 24h-Ahead Forecast -", period_label),
      subtitle = sprintf("%d environmental clusters from %d variables", 
                         cluster_obj$n_clusters,
                         length(cluster_obj$env_vars)),
      x = "Time (hours)",
      y = "Energy Consumption (kWh)",
      color = "Series"
    ) +
    theme_minimal(base_size = 12)
  
  print(p)
  
  # Plot with cluster background
  p2 <- ggplot(forecast_dt, aes(x = Time)) +
    geom_rect(aes(xmin = Time - 0.5, xmax = Time + 0.5,
                  ymin = -Inf, ymax = Inf, fill = factor(Cluster)),
              alpha = 0.2) +
    geom_line(aes(y = Actual), color = "black", linewidth = 1) +
    geom_line(aes(y = SARIMAX_Cluster_24h), color = "orange", linewidth = 0.7) +
    scale_fill_discrete(name = "Cluster") +
    labs(
      title = paste("Cluster Assignment Over Time -", period_label),
      subtitle = "24h-ahead forecasts with environmental regimes",
      x = "Time (hours)",
      y = "Energy Consumption (kWh)"
    ) +
    theme_minimal(base_size = 12)
  
  print(p2)
  
  return(list(
    eval = eval_result,
    forecasts = forecast_dt,
    plot = p,
    cluster_plot = p2,
    cluster_info = cluster_obj,
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

cat("\n========== CLUSTERING-BASED SARIMAX FORECASTING (24H AHEAD) ==========\n")
cat("Research Question: Do environmental regimes (clusters) capture\n")
cat("useful patterns for 24h-ahead forecasting vs individually-selected features?\n")

resultsA_cluster <- run_cluster_sarimax(trainA, testA, "Period A (Stable)")
resultsB_cluster <- run_cluster_sarimax(trainB, testB, "Period B (Not Stable)")

# --- FINAL SUMMARY ---

cat("\n==================== FINAL SUMMARY ====================\n")

all_eval_cluster <- rbind(resultsA_cluster$eval, resultsB_cluster$eval)
print(all_eval_cluster)

cat("\n--- Cluster Analysis ---\n")
cat(sprintf("Number of clusters: %d\n", N_CLUSTERS))
cat(sprintf("Environmental variables used: %d\n", 
            length(resultsA_cluster$cluster_info$env_vars)))

cat("\nPeriod A cluster characteristics:\n")
cat("Top differentiating variables:", 
    paste(resultsA_cluster$cluster_info$top_differentiating_vars, collapse = ", "), "\n")

cat("\nPeriod B cluster characteristics:\n")
cat("Top differentiating variables:", 
    paste(resultsB_cluster$cluster_info$top_differentiating_vars, collapse = ", "), "\n")

cat("\n--- Performance Comparison (24h-ahead) ---\n")
cat(sprintf("Period A: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f\n",
            resultsA_cluster$eval$RMSE,
            resultsA_cluster$eval$MAE,
            resultsA_cluster$eval$MAPE,
            resultsA_cluster$eval$R2))

cat(sprintf("Period B: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f\n",
            resultsB_cluster$eval$RMSE,
            resultsB_cluster$eval$MAE,
            resultsB_cluster$eval$MAPE,
            resultsB_cluster$eval$R2))

# --- INTERPRETATION ---

cat("\n--- Interpretation Guide ---\n")
cat("Compare these 24h-ahead results with other approaches:\n")
cat("  - If Clustering performs BETTER: Environmental regimes add value\n")
cat("  - If Clustering performs SIMILAR: Regimes capture same info as features\n")
cat("  - If Clustering performs WORSE: Individual features more informative\n\n")

cat("Key insight: Check cluster assignments to see if regimes are stable\n")
cat("or if rapid switching between clusters indicates regime shifts\n")

# --- SAVE RESULTS ---

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
results_file <- sprintf("SARIMAX_Cluster_24h_Results_%s.rds", timestamp)

cluster_sarimax_results <- list(
  period_A = resultsA_cluster,
  period_B = resultsB_cluster,
  evaluations = all_eval_cluster,
  parameters = list(
    order = ORDER,
    seasonal = SEASONAL,
    period = PERIOD,
    n_clusters = N_CLUSTERS,
    env_vars_used = env_vars,
    temporal_vars_used = temporal_vars,
    forecast_horizon = 24
  )
)

saveRDS(cluster_sarimax_results, file = results_file)
cat(sprintf("\nResults saved to: %s\n", results_file))

cat("\nCluster-SARIMAX 24h-ahead pipeline complete!\n")