# ---------------------------------------------------------------
# MODEL: SARIMA (No Exogenous Variables)
# 24-hour rolling forecast without retraining
# ---------------------------------------------------------------

set.seed(1234)

# --- Rolling 24h forecast function ---

rolling_sarima_24h <- function(train_data, test_data, order, seasonal, period) {
  n_test <- nrow(test_data)
  n_train <- nrow(train_data)
  forecasts <- rep(NA_real_, n_test)
  
  overall_start <- Sys.time()
  
  # --- Train model once ---
  train_start <- Sys.time()
  y_train <- ts(train_data[[target_col]], frequency = period)
  model <- Arima(y_train, order = order, seasonal = list(order = seasonal, period = period), method = "CSS-ML")
  
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  cat(sprintf("Trained SARIMA: AIC=%.2f | Time=%.2f min\n", model$aic, train_time))
  print(summary(model))
  
  # --- Rolling forecasts ---
  predict_start <- Sys.time()
  all_y <- c(train_data[[target_col]], test_data[[target_col]])
  cat(sprintf("Starting 24h rolling forecasts (%d test hours)...\n", n_test))
  
  filled <- 0
  
  for (h in seq(-22, n_test - 23)) {
    current_idx <- n_train + h - 1
    if (current_idx < 100) next
    
    hist_y <- ts(all_y[1:current_idx], frequency = period)
    updated <- tryCatch(Arima(hist_y, model = model), error = function(e) NULL)
    if (is.null(updated)) next

    fc <- forecast(updated, h = 24)
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
  
  # Fill NAs in forecast vector
  for (i in which(is.na(forecasts))) {
    forecasts[i] <- ifelse(i == 1, tail(train_data[[target_col]], 1), forecasts[i - 1])
  }
  
  list(
    forecasts = forecasts,
    model = model,  # --- keep the fitted SARIMA model
    runtime = total_time,
    train_time = train_time,
    predict_time = predict_time
  )
}

# --- Runner for both periods ---

run_sarima <- function(train, test, label) {
  cat(sprintf("\n--- %s ---\n", label))
  res <- rolling_sarima_24h(train, test, ORDER, SEASONAL, PERIOD)
  actual <- test[[target_col]]
  
  eval <- evaluate_forecast(actual, res$forecasts, "SARIMA_24h")
  eval[, `:=`(Runtime_min = res$runtime,
              Train_min = res$train_time,
              Predict_min = res$predict_time,
              Period = label)]
  
  dt <- data.table(Time = seq_along(actual), Actual = actual, Forecast = res$forecasts)
  p <- plot_forecast(dt, "SARIMA_24h", label, color = "blue")
  print(p)
  
  # --- Return full result bundle including model ---
  list(
    eval = eval,
    forecasts = dt,
    plot = p,
    model = res$model  
  )
}

# --- Data splits and execution ---

splits <- split_periods(model_data)
resultsA_sarima <- run_sarima(splits$trainA, splits$testA, "Period A")
resultsB_sarima <- run_sarima(splits$trainB, splits$testB, "Period B")

all_eval_sarima <- rbind(resultsA_sarima$eval, resultsB_sarima$eval)
print(all_eval_sarima)

# --- Print results ---

cat("\n--- Summary ---\n")
cat(sprintf("Model: %s\n", resultsA_sarima$eval$Model[1]))

cat(sprintf(
  "Period A: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | Time=%.2fmin (Train=%.2f + Predict=%.2f)\n",
  resultsA_sarima$eval$RMSE,
  resultsA_sarima$eval$MAE,
  resultsA_sarima$eval$MAPE,
  resultsA_sarima$eval$R2,
  resultsA_sarima$eval$Runtime_min,
  resultsA_sarima$eval$Train_min,
  resultsA_sarima$eval$Predict_min
))

cat(sprintf(
  "Period B: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | Time=%.2fmin (Train=%.2f + Predict=%.2f)\n",
  resultsB_sarima$eval$RMSE,
  resultsB_sarima$eval$MAE,
  resultsB_sarima$eval$MAPE,
  resultsB_sarima$eval$R2,
  resultsB_sarima$eval$Runtime_min,
  resultsB_sarima$eval$Train_min,
  resultsB_sarima$eval$Predict_min
))

# --- Save results (including trained models) ---

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Results_SARIMA_24h_%s.rds", timestamp)

saveRDS(
  list(
    model = "SARIMA_24h",
    period_A = resultsA_sarima,
    period_B = resultsB_sarima,  
    evaluations = all_eval_sarima,
    parameters = list(
      order = ORDER,
      seasonal = SEASONAL,
      period = PERIOD
    )
  ),
  file = save_name
)

cat(sprintf("\nResults and models saved to: %s\n", save_name))
cat("SARIMA 24-hour forecast complete!\n")
