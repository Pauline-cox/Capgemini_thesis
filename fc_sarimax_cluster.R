# ---------------------------------------------------------------
# MODEL: SARIMAX (Cluster-Based Exogenous Variables)
# 24-hour rolling forecast using environmental regimes
# ---------------------------------------------------------------

set.seed(1234)
library(data.table)
library(forecast)

# --- Settings ---
N_CLUSTERS <- 3
env_vars <- c(
  "tempC", "humidity", "co2", "sound", "lux", "total_occupancy",
  "temperature", "wind_speed", "sunshine_minutes",
  "global_radiation", "humidity_percent"
)
temporal_vars <- c("business_hours", "hour_sin", "hour_cos", "dow_cos", "holiday", "dst")

# --- Create Clusters (with robust numeric handling) ---
create_env_clusters <- function(train_data, n_clusters = 3) {
  available_env <- intersect(env_vars, names(train_data))
  if (length(available_env) == 0) stop("No environmental variables found.")
  
  X_env <- as.data.frame(train_data[, ..available_env])
  X_env[] <- lapply(X_env, function(col) as.numeric(as.character(col)))  # ensure numeric
  
  for (i in seq_len(ncol(X_env))) {
    mu <- mean(X_env[[i]], na.rm = TRUE)
    X_env[[i]][is.na(X_env[[i]])] <- mu
  }
  
  km <- kmeans(scale(X_env), centers = n_clusters, nstart = 25)
  
  list(
    clusters = km$cluster,
    centers = km$centers,
    env_vars = available_env,
    n_clusters = n_clusters
  )
}

# --- Assign Clusters to Test Data ---
assign_env_clusters <- function(test_data, train_data, cluster_obj) {
  X_train <- as.data.frame(train_data[, ..cluster_obj$env_vars])
  X_test  <- as.data.frame(test_data[, ..cluster_obj$env_vars])
  
  X_train[] <- lapply(X_train, function(col) as.numeric(as.character(col)))
  X_test[]  <- lapply(X_test,  function(col) as.numeric(as.character(col)))
  
  for (i in seq_len(ncol(X_train))) {
    mu <- mean(X_train[[i]], na.rm = TRUE)
    X_train[[i]][is.na(X_train[[i]])] <- mu
    X_test[[i]][is.na(X_test[[i]])]  <- mu
  }
  
  X_train_scaled <- scale(X_train)
  X_test_scaled  <- scale(
    X_test,
    center = attr(X_train_scaled, "scaled:center"),
    scale  = attr(X_train_scaled, "scaled:scale")
  )
  
  apply(X_test_scaled, 1, function(x) which.min(colSums((t(cluster_obj$centers) - x)^2)))
}

# --- Create Cluster Feature Dummies ---
create_cluster_features <- function(train_data, test_data, cluster_obj) {
  train_dt <- copy(train_data)
  test_dt <- copy(test_data)
  
  train_dt[, env_cluster := cluster_obj$clusters]
  test_dt[, env_cluster := assign_env_clusters(test_dt, train_dt, cluster_obj)]
  
  for (k in 1:(cluster_obj$n_clusters - 1)) {
    var <- paste0("cluster_", k)
    train_dt[, (var) := as.integer(env_cluster == k)]
    test_dt[, (var) := as.integer(env_cluster == k)]
  }
  
  available_temp <- intersect(temporal_vars, names(train_data))
  feature_vars <- c(paste0("cluster_", 1:(cluster_obj$n_clusters - 1)), available_temp)
  
  list(
    train = train_dt[, ..feature_vars],
    test = test_dt[, ..feature_vars],
    feature_names = feature_vars,
    n_clusters = cluster_obj$n_clusters
  )
}

# --- Rolling SARIMAX with Cluster Features ---
rolling_sarimax_cluster_24h <- function(train_data, test_data, cluster_features, order, seasonal, period) {
  n_test <- nrow(test_data)
  n_train <- nrow(train_data)
  forecasts <- rep(NA_real_, n_test)
  overall_start <- Sys.time()
  
  # --- Train initial model ---
  train_start <- Sys.time()
  y_train <- train_data[[target_col]]
  x_train <- as.matrix(cluster_features$train)
  x_train_scaled <- scale(x_train)
  center_vec <- attr(x_train_scaled, "scaled:center")
  scale_vec  <- attr(x_train_scaled, "scaled:scale")
  
  y_train_ts <- ts(y_train, frequency = period)
  model <- Arima(y_train_ts,
                 order = order,
                 seasonal = list(order = seasonal, period = period),
                 xreg = x_train_scaled,
                 method = "CSS-ML")
  
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  cat(sprintf("Trained SARIMAX-Cluster: %d clusters | AIC=%.2f | Time=%.2fmin\n",
              cluster_features$n_clusters, model$aic, train_time))
  print(summary(model))
  
  # --- Rolling 24-hour forecasts ---
  predict_start <- Sys.time()
  all_y <- c(y_train, test_data[[target_col]])
  all_x <- rbind(x_train, as.matrix(cluster_features$test))
  all_x_scaled <- scale(all_x, center = center_vec, scale = scale_vec)
  
  cat(sprintf("Starting 24h rolling forecasts (%d test hours)...\n", n_test))
  filled <- 0
  
  for (h in seq(-22, n_test - 23)) {
    current_idx <- n_train + h - 1
    if (current_idx < 100) next
    
    hist_y <- all_y[1:current_idx]
    hist_x <- all_x_scaled[1:current_idx, , drop = FALSE]
    future_x <- all_x_scaled[(current_idx + 1):(current_idx + 24), , drop = FALSE]
    if (nrow(future_x) < 24) next
    
    updated <- tryCatch(
      Arima(ts(hist_y, frequency = period),
            model = model,
            xreg = hist_x),
      error = function(e) NULL
    )
    if (is.null(updated)) next
    
    fc <- forecast(updated, xreg = future_x, h = 24)
    idx <- current_idx + 24 - n_train
    
    if (idx >= 1 && idx <= n_test) {
      forecasts[idx] <- fc$mean[24]
      filled <- filled + 1
    }
    
    # --- Standardized progress print ---
    if (filled %% 24 == 0 || filled == n_test) {
      elapsed <- as.numeric(difftime(Sys.time(), predict_start, units = "mins"))
      pct <- (filled / n_test) * 100
      cat(sprintf("Progress: %3d/%d (%.1f%%) | Elapsed: %.2f min\n",
                  filled, n_test, pct, elapsed))
      flush.console()
    }
  }
  
  cat("Forecasting complete.\n")
  
  predict_time <- as.numeric(difftime(Sys.time(), predict_start, units = "mins"))
  total_time <- as.numeric(difftime(Sys.time(), overall_start, units = "mins"))
  
  for (i in which(is.na(forecasts))) {
    forecasts[i] <- ifelse(i == 1, tail(y_train, 1), forecasts[i - 1])
  }
  
  list(forecasts = forecasts,
       runtime = total_time,
       train_time = train_time,
       predict_time = predict_time)
}

# --- Runner Function ---
run_sarimax_cluster <- function(train, test, label) {
  cat(sprintf("\n--- %s ---\n", label))
  cluster_obj <- create_env_clusters(train, n_clusters = N_CLUSTERS)
  cluster_features <- create_cluster_features(train, test, cluster_obj)
  res <- rolling_sarimax_cluster_24h(train, test, cluster_features, ORDER, SEASONAL, PERIOD)
  
  actual <- test[[target_col]]
  eval <- evaluate_forecast(actual, res$forecasts, "SARIMAX_Cluster_24h")
  eval[, `:=`(Runtime_min = res$runtime, Train_min = res$train_time,
              Predict_min = res$predict_time, Period = label)]
  
  dt <- data.table(Time = seq_along(actual), Actual = actual, Forecast = res$forecasts)
  p <- plot_forecast(dt, "SARIMAX_Cluster_24h", label, color = "orange")
  print(p)
  
  list(eval = eval, forecasts = dt, plot = p)
}

# --- Execution ---
splits <- split_periods(model_data)
resultsA_sarimax_cluster <- run_sarimax_cluster(splits$trainA, splits$testA, "Period A (Stable)")
resultsB_sarimax_cluster <- run_sarimax_cluster(splits$trainB, splits$testB, "Period B (Not Stable)")

all_eval_sarimax_cluster <- rbind(resultsA_sarimax_cluster$eval, resultsB_sarimax_cluster$eval)
print(all_eval_sarimax_cluster)

# --- Summary Output ---
cat("\n--- Summary ---\n")
cat("Model: SARIMAX_Cluster_24h\n")

cat(sprintf(
  "Period A: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R²=%.4f | Time=%.2fmin (Train=%.2f + Predict=%.2f)\n",
  resultsA_sarimax_cluster$eval$RMSE,
  resultsA_sarimax_cluster$eval$MAE,
  resultsA_sarimax_cluster$eval$MAPE,
  resultsA_sarimax_cluster$eval$R2,
  resultsA_sarimax_cluster$eval$Runtime_min,
  resultsA_sarimax_cluster$eval$Train_min,
  resultsA_sarimax_cluster$eval$Predict_min
))

cat(sprintf(
  "Period B: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R²=%.4f | Time=%.2fmin (Train=%.2f + Predict=%.2f)\n",
  resultsB_sarimax_cluster$eval$RMSE,
  resultsB_sarimax_cluster$eval$MAE,
  resultsB_sarimax_cluster$eval$MAPE,
  resultsB_sarimax_cluster$eval$R2,
  resultsB_sarimax_cluster$eval$Runtime_min,
  resultsB_sarimax_cluster$eval$Train_min,
  resultsB_sarimax_cluster$eval$Predict_min
))

# --- Save Results ---
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Results_SARIMAX_Cluster_24h_%s.rds", timestamp)
saveRDS(list(model = "SARIMAX_Cluster_24h",
             period_A = resultsA_sarimax_cluster,
             period_B = resultsB_sarimax_cluster,
             evaluations = all_eval_sarimax_cluster,
             parameters = list(order = ORDER, seasonal = SEASONAL,
                               period = PERIOD, clusters = N_CLUSTERS)),
        file = save_name)
cat(sprintf("\nResults saved to: %s\n", save_name))
cat("SARIMAX (Cluster) 24-hour forecast complete!\n")