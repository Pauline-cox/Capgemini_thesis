# ---------------------------------------------------------------
# MODEL: SARIMAX (PCA-Based Exogenous Variables)
# 24-hour rolling forecast using principal components
# ---------------------------------------------------------------

set.seed(1234)

# --- Settings ---
PCA_VAR_THRESHOLD <- 0.90
env_vars <- c(
  "tempC", "humidity", "co2", "sound", "lux", "total_occupancy",
  "temperature", "wind_speed", "sunshine_minutes",
  "global_radiation", "humidity_percent"
)
temporal_vars <- c("business_hours", "hour_sin", "hour_cos", "dow_cos", "holiday", "dst")

# --- PCA Feature Extraction ---
extract_pca_features <- function(train_data, test_data, var_threshold = PCA_VAR_THRESHOLD) {
  available_env <- intersect(env_vars, names(train_data))
  available_temp <- intersect(temporal_vars, names(train_data))
  if (length(available_env) == 0) stop("No environmental variables available for PCA.")
  
  X_train <- as.matrix(train_data[, ..available_env])
  X_test  <- as.matrix(test_data[, ..available_env])
  
  for (i in seq_len(ncol(X_train))) {
    mu <- mean(X_train[, i], na.rm = TRUE)
    X_train[is.na(X_train[, i]), i] <- mu
    X_test[is.na(X_test[, i]), i] <- mu
  }
  
  pca_model <- prcomp(X_train, center = TRUE, scale. = TRUE)
  var_exp <- pca_model$sdev^2 / sum(pca_model$sdev^2)
  cum_var <- cumsum(var_exp)
  n_comp <- which(cum_var >= var_threshold)[1]
  if (is.na(n_comp)) n_comp <- ncol(X_train)
  
  cat(sprintf("PCA Components Retained: %d (%.1f%% variance)\n", n_comp, cum_var[n_comp] * 100))
  
  train_pcs <- predict(pca_model, X_train)[, 1:n_comp, drop = FALSE]
  test_pcs  <- predict(pca_model, X_test)[, 1:n_comp, drop = FALSE]
  colnames(train_pcs) <- paste0("PC", 1:n_comp)
  colnames(test_pcs)  <- paste0("PC", 1:n_comp)
  
  list(
    train = cbind(as.data.table(train_pcs), train_data[, ..available_temp]),
    test  = cbind(as.data.table(test_pcs), test_data[, ..available_temp]),
    n_comp = n_comp,
    var_explained = cum_var[n_comp]
  )
}

# --- Rolling SARIMAX with PCA Features ---
rolling_sarimax_pca_24h <- function(train_data, test_data, pca_obj, order, seasonal, period) {
  n_test <- nrow(test_data)
  n_train <- nrow(train_data)
  forecasts <- rep(NA_real_, n_test)
  overall_start <- Sys.time()
  
  # --- Train model once ---
  train_start <- Sys.time()
  y_train <- train_data[[target_col]]
  x_train <- as.matrix(pca_obj$train)
  x_train_scaled <- scale(x_train)
  center_vec <- attr(x_train_scaled, "scaled:center")
  scale_vec  <- attr(x_train_scaled, "scaled:scale")
  
  y_train_ts <- ts(y_train, frequency = period)
  model <- Arima(
    y_train_ts,
    order = order,
    seasonal = list(order = seasonal, period = period),
    xreg = x_train_scaled,
    method = "CSS-ML"
  )
  
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  cat(sprintf("Trained SARIMAX-PCA: %d PCs | AIC=%.2f | Time=%.2fmin\n",
              pca_obj$n_comp, model$aic, train_time))
  print(summary(model))
  
  # --- Rolling 24-hour forecasts ---
  predict_start <- Sys.time()
  all_y <- c(y_train, test_data[[target_col]])
  all_x <- rbind(x_train, as.matrix(pca_obj$test))
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
      Arima(ts(hist_y, frequency = period), model = model, xreg = hist_x),
      error = function(e) NULL
    )
    if (is.null(updated)) next
    
    fc <- forecast(updated, xreg = future_x, h = 24)
    idx <- current_idx + 24 - n_train
    
    if (idx >= 1 && idx <= n_test) {
      forecasts[idx] <- fc$mean[24]
      filled <- filled + 1
    }
    
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
  
  # Fill missing forecasts
  for (i in which(is.na(forecasts))) {
    forecasts[i] <- ifelse(i == 1, tail(y_train, 1), forecasts[i - 1])
  }
  
  list(
    forecasts = forecasts,
    model = model,  
    runtime = total_time,
    train_time = train_time,
    predict_time = predict_time
  )
}

# --- Runner for both periods ---
run_sarimax_pca <- function(train, test, label) {
  cat(sprintf("\n--- %s ---\n", label))
  pca_obj <- extract_pca_features(train, test, var_threshold = PCA_VAR_THRESHOLD)
  res <- rolling_sarimax_pca_24h(train, test, pca_obj, ORDER, SEASONAL, PERIOD)
  
  actual <- test[[target_col]]
  eval <- evaluate_forecast(actual, res$forecasts, "SARIMAX_PCA_24h")
  eval[, `:=`(Runtime_min = res$runtime,
              Train_min = res$train_time,
              Predict_min = res$predict_time,
              Period = label,
              N_Components = pca_obj$n_comp,
              Var_Explained = pca_obj$var_explained)]
  
  dt <- data.table(Time = seq_along(actual), Actual = actual, Forecast = res$forecasts)
  p <- plot_forecast(dt, "SARIMAX_PCA_24h", label, color = "purple")
  print(p)
  
  list(
    eval = eval,
    forecasts = dt,
    plot = p,
    model = res$model,  
    pca_obj = pca_obj      
  )
}

# --- Data splits and execution ---
splits <- split_periods(model_data)
resultsA_sarimax_pca <- run_sarimax_pca(splits$trainA, splits$testA, "Period A (Stable)")
resultsB_sarimax_pca <- run_sarimax_pca(splits$trainB, splits$testB, "Period B (Not Stable)")

all_eval_sarimax_pca <- rbind(resultsA_sarimax_pca$eval, resultsB_sarimax_pca$eval)
print(all_eval_sarimax_pca)

# --- Print results ---
cat("\n--- Summary ---\n")
cat(sprintf("Model: %s\n", resultsA_sarimax_pca$eval$Model[1]))

cat(sprintf(
  "Period A: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | Time=%.2fmin (Train=%.2f + Predict=%.2f)\n",
  resultsA_sarimax_pca$eval$RMSE,
  resultsA_sarimax_pca$eval$MAE,
  resultsA_sarimax_pca$eval$MAPE,
  resultsA_sarimax_pca$eval$R2,
  resultsA_sarimax_pca$eval$Runtime_min,
  resultsA_sarimax_pca$eval$Train_min,
  resultsA_sarimax_pca$eval$Predict_min
))

cat(sprintf(
  "Period B: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | Time=%.2fmin (Train=%.2f + Predict=%.2f)\n",
  resultsB_sarimax_pca$eval$RMSE,
  resultsB_sarimax_pca$eval$MAE,
  resultsB_sarimax_pca$eval$MAPE,
  resultsB_sarimax_pca$eval$R2,
  resultsB_sarimax_pca$eval$Runtime_min,
  resultsB_sarimax_pca$eval$Train_min,
  resultsB_sarimax_pca$eval$Predict_min
))

# --- Save results ---
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Results_SARIMAX_PCA_24h_%s.rds", timestamp)

saveRDS(
  list(
    model = "SARIMAX_PCA_24h",
    period_A = resultsA_sarimax_pca, 
    period_B = resultsB_sarimax_pca, 
    evaluations = all_eval_sarimax_pca,
    parameters = list(
      order = ORDER,
      seasonal = SEASONAL,
      period = PERIOD,
      pca_vars = env_vars,
      pca_var_threshold = PCA_VAR_THRESHOLD
    )
  ),
  file = save_name
)
cat(sprintf("\nResults and trained models saved to: %s\n", save_name))
cat("SARIMAX (PCA) 24-hour forecast complete!\n")
