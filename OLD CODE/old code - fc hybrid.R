# ---------------------------------------------------------------
# MODEL: Hybrid (SARIMAX + LSTM Residual Correction)
# ---------------------------------------------------------------

set.seed(1234)

# --- Reuse SARIMAX (raw) variables ---
xreg_vars <- c("co2", "business_hours", "total_occupancy",
               "lux", "hour_cos", "hour_sin", "holiday", "dst")

# --- Hyperparameters from Bayesian opt ---
LOOKBACK = 168         
UNITS1 = 104
UNITS2 = 30
DROPOUT = 0.05
LR = 0.004108306
BATCH = 37
EPOCHS = 50          
HORIZON = 24

# --- Hybrid forecast function ---
hybrid_forecast_24h <- function(train_data, test_data, order, seasonal, period, xreg_vars) {
  cat("\n>>> Hybrid model: SARIMAX base + LSTM residual <<<\n")
  overall_start <- Sys.time()
  
  # --- Step 1: SARIMAX base forecast ---
  base_model <- Arima(
    ts(train_data[[target_col]], frequency = period),
    order = order,
    seasonal = list(order = seasonal, period = period),
    xreg = as.matrix(scale(train_data[, ..xreg_vars])),
    method = "CSS"
  )
  
  cat("Base SARIMAX model trained.\n")
  
  # Forecast base values
  all_x <- rbind(train_data[, ..xreg_vars], test_data[, ..xreg_vars])
  all_x_scaled <- scale(all_x, center = attr(scale(train_data[, ..xreg_vars]), "scaled:center"),
                        scale = attr(scale(train_data[, ..xreg_vars]), "scaled:scale"))
  
  all_y <- c(train_data[[target_col]], test_data[[target_col]])
  n_train <- nrow(train_data)
  n_test <- nrow(test_data)
  base_fc <- rep(NA_real_, n_test)
  
  filled <- 0
  for (h in seq(-22, n_test - 23)) {
    current_idx <- n_train + h - 1
    if (current_idx < 100) next
    hist_y <- all_y[1:current_idx]
    hist_x <- all_x_scaled[1:current_idx, , drop = FALSE]
    future_x <- all_x_scaled[(current_idx + 1):(current_idx + 24), , drop = FALSE]
    updated <- tryCatch(Arima(ts(hist_y, frequency = period), model = base_model, xreg = hist_x),
                        error = function(e) NULL)
    if (is.null(updated)) next
    fc <- forecast(updated, xreg = future_x, h = 24)
    idx <- current_idx + 24 - n_train
    if (idx >= 1 && idx <= n_test) {
      base_fc[idx] <- fc$mean[24]
      filled <- filled + 1
    }
    if (filled %% 24 == 0 || filled == n_test) {
      pct <- (filled / n_test) * 100
      cat(sprintf("SARIMAX progress: %3d/%d (%.1f%%)\n", filled, n_test, pct))
    }
  }
  
  base_fc <- zoo::na.locf0(base_fc)
  residuals_train <- train_data[[target_col]] - fitted(base_model)
  
  # --- Step 2: LSTM on residuals ---
  cat("\nTraining LSTM on residuals...\n")
  train_resid <- copy(train_data)
  train_resid$residual <- residuals_train
  test_resid <- copy(test_data)
  test_resid$residual <- 0
  
  res_lstm <- lstm_pure_24h_only(train_resid, test_resid, feature_columns, LOOKBACK,
                                 UNITS1, UNITS2, DROPOUT, LR, BATCH, EPOCHS, HORIZON)
  
  hybrid_forecast <- base_fc + res_lstm$forecasts
  runtime_total <- as.numeric(difftime(Sys.time(), overall_start, units = "mins"))
  
  list(forecasts = hybrid_forecast, runtime = runtime_total)
}

# --- Runner ---
run_hybrid_model <- function(train, test, label) {
  res <- hybrid_forecast_24h(train, test, ORDER, SEASONAL, PERIOD, xreg_vars)
  actual <- test[[target_col]]
  eval <- evaluate_forecast(actual, res$forecasts, "Hybrid_SARIMAX_LSTM_24h")
  eval[, `:=`(Runtime_min = res$runtime, Period = label)]
  
  dt <- data.table(Time = seq_along(actual), Actual = actual, Forecast = res$forecasts)
  p <- plot_forecast(dt, "Hybrid_SARIMAX_LSTM_24h", label, color = "green")
  print(p)
  
  list(eval = eval, forecasts = dt, plot = p)
}

# --- Execution ---
splits <- split_periods(model_data)
resultsA_hybrid <- run_hybrid_model(splits$trainA, splits$testA, "Period A (Stable)")
resultsB_hybrid <- run_hybrid_model(splits$trainB, splits$testB, "Period B (Not Stable)")

all_eval_hybrid <- rbind(resultsA_hybrid$eval, resultsB_hybrid$eval)
print(all_eval_hybrid)

cat("\n--- Summary ---\n")
cat(sprintf("Model: %s\n", resultsA_hybrid$eval$Model[1]))
cat(sprintf(
  "Period A: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | Time=%.2fmin\n",
  resultsA_hybrid$eval$RMSE, resultsA_hybrid$eval$MAE,
  resultsA_hybrid$eval$MAPE, resultsA_hybrid$eval$R2, resultsA_hybrid$eval$Runtime_min
))
cat(sprintf(
  "Period B: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | Time=%.2fmin\n",
  resultsB_hybrid$eval$RMSE, resultsB_hybrid$eval$MAE,
  resultsB_hybrid$eval$MAPE, resultsB_hybrid$eval$R2, resultsB_hybrid$eval$Runtime_min
))

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Results_Hybrid_SARIMAX_LSTM_24h_%s.rds", timestamp)
saveRDS(
  list(model = "Hybrid_SARIMAX_LSTM_24h",
       period_A = resultsA_hybrid, period_B = resultsB_hybrid,
       evaluations = all_eval_hybrid),
  file = save_name
)
cat(sprintf("\nResults saved to: %s\n", save_name))
cat("Hybrid (SARIMAX + LSTM residual) forecast complete!\n")