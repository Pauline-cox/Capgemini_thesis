# ==============================================================
# Author: Pauline Cox
# Script: fc_sarimax.R
#
# Description: Implements a SARIMAX model using raw exogenous 
# variables for 24-hour rolling energy consumption forecasts. 
# The model is trained once per period and generates sequential
# forecasts across two evaluation periods (A and B) without retraining.
#
# Input: 
#   - Preprocessed and feature-engineered dataset (model_data)
#   - Functions from forecast_preparations.R
#   - Selected features
#
# Output: 
#   - Forecast results, evaluation metrics and diagnostic plots
# ==============================================================

set.seed(1234)

# --- Selected Exogenous Variables ---

selected_xreg <- c(
  "co2", "total_occupancy", "lux",
  "business_hours", "hour_sin", "hour_cos", "dow_cos",
  "holiday", "dst"
)

# --- Rolling 24h SARIMAX Forecast ---

rolling_sarimax_24h <- function(train_data, test_data, order, seasonal, period, xreg_vars) {
  n_test <- nrow(test_data)
  n_train <- nrow(train_data)
  forecasts <- rep(NA_real_, n_test)
  
  overall_start <- Sys.time()
  
  # --- Train model once ---
  train_start <- Sys.time()
  y_train <- train_data[[target_col]]
  x_train <- as.matrix(train_data[, ..xreg_vars])
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
  cat(sprintf("Trained SARIMAX: %d features | AIC=%.2f | Time=%.2f min\n",
              length(xreg_vars), model$aic, train_time))
  print(summary(model))
  
  # --- Rolling forecasts ---
  predict_start <- Sys.time()
  all_y <- c(y_train, test_data[[target_col]])
  all_x <- rbind(x_train, as.matrix(test_data[, ..xreg_vars]))
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
    
    # --- Print progress every 24 filled forecasts ---
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
  
  # Fill NAs if needed
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

run_sarimax_raw <- function(train, test, label) {
  cat(sprintf("\n--- %s ---\n", label))
  res <- rolling_sarimax_24h(train, test, ORDER, SEASONAL, PERIOD, selected_xreg)
  actual <- test[[target_col]]
  
  eval <- evaluate_forecast(actual, res$forecasts, "SARIMAX_Raw_24h")
  eval[, `:=`(
    Runtime_min = res$runtime,
    Train_min = res$train_time,
    Predict_min = res$predict_time,
    Period = label
  )]
  
  dt <- data.table(Time = seq_along(actual), Actual = actual, Forecast = res$forecasts)
  p <- plot_forecast(dt, "SARIMAX_Raw_24h", label, color = "red")
  print(p)
  
  # --- Return results ---
  list(
    eval = eval,
    forecasts = dt,
    plot = p,
    model = res$model
  )
}

# --- Data splits and execution ---

splits <- split_periods(model_data)
resultsA_sarimax_raw <- run_sarimax_raw(splits$trainA, splits$testA, "Period A")
resultsB_sarimax_raw <- run_sarimax_raw(splits$trainB, splits$testB, "Period B")

all_eval_sarimax_raw <- rbind(resultsA_sarimax_raw$eval, resultsB_sarimax_raw$eval)
print(all_eval_sarimax_raw)

# --- Print results ---

cat(sprintf("Model: %s\n", resultsA_sarimax_raw$eval$Model[1]))

cat(sprintf(
  "Period A: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | Time=%.2fmin (Train=%.2f + Predict=%.2f)\n",
  resultsA_sarimax_raw$eval$RMSE,
  resultsA_sarimax_raw$eval$MAE,
  resultsA_sarimax_raw$eval$MAPE,
  resultsA_sarimax_raw$eval$R2,
  resultsA_sarimax_raw$eval$Runtime_min,
  resultsA_sarimax_raw$eval$Train_min,
  resultsA_sarimax_raw$eval$Predict_min
))

cat(sprintf(
  "Period B: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | Time=%.2fmin (Train=%.2f + Predict=%.2f)\n",
  resultsB_sarimax_raw$eval$RMSE,
  resultsB_sarimax_raw$eval$MAE,
  resultsB_sarimax_raw$eval$MAPE,
  resultsB_sarimax_raw$eval$R2,
  resultsB_sarimax_raw$eval$Runtime_min,
  resultsB_sarimax_raw$eval$Train_min,
  resultsB_sarimax_raw$eval$Predict_min
))

# --- Save results  ---

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Results_SARIMAX_Raw_24h_%s.rds", timestamp)

saveRDS(
  list(
    model = "SARIMAX_Raw_24h",
    period_A = resultsA_sarimax_raw,
    period_B = resultsB_sarimax_raw,
    evaluations = all_eval_sarimax_raw,
    parameters = list(
      order = ORDER,
      seasonal = SEASONAL,
      period = PERIOD,
      features = selected_xreg
    )
  ),
  file = save_name
)

cat(sprintf("\nResults and models saved to: %s\n", save_name))
