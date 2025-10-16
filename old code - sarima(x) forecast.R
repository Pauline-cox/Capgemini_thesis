# ===============================================================
# 24-HOUR ROLLING FORECAST - SARIMA & SARIMAX (NO RETRAINING)
# Fair comparison with LSTM: Train once, predict many times
# ===============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(forecast)
  library(ggplot2)
})

# ===============================================================
# SETTINGS
# ===============================================================
set.seed(1234)

ORDER    <- c(2, 0, 2)
SEASONAL <- c(0, 1, 1)
PERIOD   <- 168L
target_col <- "total_consumption_kWh"

selected_xreg <- c(
  "co2", "business_hours", "total_occupancy",
  "lux", "hour_cos", "hour_sin",
  "holiday", "dst"
)

# Test periods
testA_start <- as.Date("2024-10-01")
testA_end   <- as.Date("2024-10-14")
trainA_end  <- testA_start - 1

testB_start <- as.Date("2024-12-18")
testB_end   <- as.Date("2024-12-31")
trainB_end  <- testB_start - 1

# ===============================================================
# HELPER FUNCTIONS
# ===============================================================

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

# ===============================================================
# 24-HOUR ROLLING FORECAST FUNCTIONS (NO RETRAINING)
# ===============================================================

rolling_sarima_24h_no_retrain <- function(train_data, test_data, order, seasonal, period) {
  cat("\n>>> Starting 24h Rolling SARIMA Forecast (NO RETRAINING)...\n")
  cat("Strategy: Train once, then roll through test set\n\n")
  
  n_test <- nrow(test_data)
  n_train <- nrow(train_data)
  forecasts <- rep(NA_real_, n_test)
  
  overall_start <- Sys.time()
  
  # --- STEP 1: TRAIN MODEL ONCE ---
  
  cat("--- Step 1: Training SARIMA model (ONE TIME) ---\n")
  train_start <- Sys.time()
  
  y_train <- train_data[[target_col]]
  y_train_ts <- ts(y_train, frequency = period)
  
  cat(sprintf("Training samples: %d\n", length(y_train)))
  
  # Train the model once
  sarima_model <- tryCatch(
    Arima(y_train_ts, 
          order = order,
          seasonal = list(order = seasonal, period = period),
          method = "CSS"),
    error = function(e) {
      cat("ERROR: Model training failed\n")
      return(NULL)
    }
  )
  
  if (is.null(sarima_model)) {
    stop("SARIMA model training failed")
  }
  
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  cat(sprintf("Training complete! Time: %.2f minutes\n", train_time))
  cat(sprintf("AIC: %.2f | BIC: %.2f\n\n", sarima_model$aic, sarima_model$bic))
  
  # --- STEP 2: ROLLING PREDICTIONS (NO RETRAINING) ---
  
  cat("--- Step 2: Rolling predictions (no retraining) ---\n")
  predict_start <- Sys.time()
  
  # Combine train + test
  all_y <- c(y_train, test_data[[target_col]])
  
  # Pre-roll: Start from -22 to get aligned 24h-ahead forecasts
  # Stop at n_test - 23 since later iterations don't produce valid forecasts
  for (h in seq(-22, n_test - 23)) {
    
    # Progress indicator
    if (h %% 24 == 0 || h == -22 || h == 1) {
      elapsed <- as.numeric(difftime(Sys.time(), predict_start, units = "secs"))
      if (h > 0) {
        pct_complete <- (h / (n_test - 23)) * 100
        cat(sprintf("  Hour %d/%d (%.1f%%) | Elapsed: %.1fs\n", 
                    h, n_test - 23, pct_complete, elapsed))
      } else {
        cat(sprintf("  Pre-roll hour %d | Elapsed: %.1fs\n", h, elapsed))
      }
    }
    
    # Current position in combined data
    current_idx <- n_train + h - 1
    
    # Skip if not enough history
    if (current_idx < 100) next
    
    # Get history up to current point
    history_y <- all_y[1:current_idx]
    history_ts <- ts(history_y, frequency = period)
    
    # Update model with new observations (NOT retraining from scratch)
    updated_model <- tryCatch(
      Arima(history_ts,
            model = sarima_model),  # ← KEY: Use existing model structure
      error = function(e) NULL
    )
    
    if (is.null(updated_model)) next
    
    # Forecast 24 hours ahead
    fc <- forecast(updated_model, h = 24)
    
    # Calculate which test hour this 24h-ahead forecast is for
    forecast_hour <- current_idx + 24 - n_train
    
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
  for (i in which(is.na(forecasts))) {
    if (i == 1) {
      forecasts[i] <- tail(y_train, 1)
    } else {
      forecasts[i] <- forecasts[i-1]
    }
  }
  
  return(list(
    forecasts = forecasts, 
    runtime = total_time,
    train_time = train_time,
    predict_time = predict_time
  ))
}

rolling_sarimax_24h_no_retrain <- function(train_data, test_data, order, seasonal, period, xreg_vars) {
  cat("\n>>> Starting 24h Rolling SARIMAX Forecast (NO RETRAINING)...\n")
  cat("Strategy: Train once, then roll through test set\n\n")
  
  n_test <- nrow(test_data)
  n_train <- nrow(train_data)
  forecasts <- rep(NA_real_, n_test)
  
  overall_start <- Sys.time()
  
  # --- STEP 1: TRAIN MODEL ONCE ---
  
  cat("--- Step 1: Training SARIMAX model (ONE TIME) ---\n")
  train_start <- Sys.time()
  
  y_train <- train_data[[target_col]]
  x_train <- as.matrix(train_data[, ..xreg_vars])
  
  # Scale features
  x_train_scaled <- scale(x_train)
  center_vec <- attr(x_train_scaled, "scaled:center")
  scale_vec <- attr(x_train_scaled, "scaled:scale")
  
  y_train_ts <- ts(y_train, frequency = period)
  
  cat(sprintf("Training samples: %d\n", length(y_train)))
  cat(sprintf("Features: %d\n", length(xreg_vars)))
  
  # Train the model once
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
    stop("SARIMAX model training failed")
  }
  
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  cat(sprintf("Training complete! Time: %.2f minutes\n", train_time))
  cat(sprintf("AIC: %.2f | BIC: %.2f\n\n", sarimax_model$aic, sarimax_model$bic))
  
  # --- STEP 2: ROLLING PREDICTIONS (NO RETRAINING) ---
  
  cat("--- Step 2: Rolling predictions (no retraining) ---\n")
  predict_start <- Sys.time()
  
  # Combine train + test
  all_y <- c(y_train, test_data[[target_col]])
  all_x <- rbind(x_train, as.matrix(test_data[, ..xreg_vars]))
  
  # Scale all features using training statistics
  all_x_scaled <- scale(all_x, center = center_vec, scale = scale_vec)
  
  # Pre-roll: Start from -22 to get aligned 24h-ahead forecasts
  # Stop at n_test - 23 since later iterations don't produce valid forecasts
  for (h in seq(-22, n_test - 23)) {
    
    # Progress indicator
    if (h %% 24 == 0 || h == -22 || h == 1) {
      elapsed <- as.numeric(difftime(Sys.time(), predict_start, units = "secs"))
      if (h > 0) {
        pct_complete <- (h / (n_test - 23)) * 100
        cat(sprintf("  Hour %d/%d (%.1f%%) | Elapsed: %.1fs\n", 
                    h, n_test - 23, pct_complete, elapsed))
      } else {
        cat(sprintf("  Pre-roll hour %d | Elapsed: %.1fs\n", h, elapsed))
      }
    }
    
    # Current position in combined data
    current_idx <- n_train + h - 1
    
    # Skip if not enough history
    if (current_idx < 100) next
    
    # Get history up to current point
    history_y <- all_y[1:current_idx]
    history_x <- all_x_scaled[1:current_idx, , drop = FALSE]
    
    # Get future 24 hours of features
    future_start <- current_idx + 1
    future_end <- min(current_idx + 24, nrow(all_x_scaled))
    
    if (future_end - future_start + 1 < 24) next
    
    future_x <- all_x_scaled[future_start:future_end, , drop = FALSE]
    
    # Create time series
    history_ts <- ts(history_y, frequency = period)
    
    # Update model with new observations (NOT retraining from scratch)
    updated_model <- tryCatch(
      Arima(history_ts,
            model = sarimax_model,  # ← KEY: Use existing model structure
            xreg = history_x),
      error = function(e) NULL
    )
    
    if (is.null(updated_model)) next
    
    # Forecast 24 hours ahead with future exogenous variables
    fc <- forecast(updated_model, xreg = future_x[1:24, , drop = FALSE], h = 24)
    
    # Calculate which test hour this 24h-ahead forecast is for
    forecast_hour <- current_idx + 24 - n_train
    
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
  for (i in which(is.na(forecasts))) {
    if (i == 1) {
      forecasts[i] <- tail(y_train, 1)
    } else {
      forecasts[i] <- forecasts[i-1]
    }
  }
  
  return(list(
    forecasts = forecasts, 
    runtime = total_time,
    train_time = train_time,
    predict_time = predict_time
  ))
}

# ===============================================================
# RUN ROLLING FORECASTS FOR BOTH PERIODS
# ===============================================================

run_rolling_forecasts <- function(train, test, period_label) {
  cat(sprintf("\n==================== %s ====================\n", period_label))
  
  # Run SARIMA rolling forecast (no retraining)
  sarima_result <- rolling_sarima_24h_no_retrain(train, test, ORDER, SEASONAL, PERIOD)
  
  # Run SARIMAX rolling forecast (no retraining)
  sarimax_result <- rolling_sarimax_24h_no_retrain(train, test, ORDER, SEASONAL, PERIOD, selected_xreg)
  
  # Evaluate both models
  actual <- test[[target_col]]
  
  eval_sarima <- evaluate_forecast(actual, sarima_result$forecasts, "SARIMA_24h")
  eval_sarima[, Runtime_min := sarima_result$runtime]
  eval_sarima[, Train_min := sarima_result$train_time]
  eval_sarima[, Predict_min := sarima_result$predict_time]
  
  eval_sarimax <- evaluate_forecast(actual, sarimax_result$forecasts, "SARIMAX_24h")
  eval_sarimax[, Runtime_min := sarimax_result$runtime]
  eval_sarimax[, Train_min := sarimax_result$train_time]
  eval_sarimax[, Predict_min := sarimax_result$predict_time]
  
  eval_combined <- rbind(eval_sarima, eval_sarimax)
  eval_combined[, Period := period_label]
  
  cat("\n--- Evaluation Results (24h-ahead, no retraining) ---\n")
  print(eval_combined)
  
  # Create forecast comparison dataframe
  forecast_dt <- data.table(
    Time = seq_along(actual),
    Actual = actual,
    SARIMA_24h = sarima_result$forecasts,
    SARIMAX_24h = sarimax_result$forecasts
  )
  
  # Plot results
  p <- ggplot(forecast_dt, aes(x = Time)) +
    geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
    geom_line(aes(y = SARIMA_24h, color = "SARIMA"), linewidth = 0.7, alpha = 0.8) +
    geom_line(aes(y = SARIMAX_24h, color = "SARIMAX"), linewidth = 0.7, alpha = 0.8) +
    scale_color_manual(values = c("Actual" = "black", "SARIMA" = "blue", "SARIMAX" = "red")) +
    labs(
      title = paste("24h-Ahead Rolling Forecast (No Retraining) -", period_label),
      subtitle = "Fair comparison: Train once, predict many times",
      x = "Time (hours)", 
      y = "Energy Consumption (kWh)",
      color = "Series"
    ) +
    theme_minimal(base_size = 12)
  
  print(p)
  
  return(list(
    eval = eval_combined,
    forecasts = forecast_dt,
    plot = p
  ))
}

# ===============================================================
# MAIN EXECUTION
# ===============================================================

# Prepare data splits
trainA <- model_data[as.Date(interval) <= trainA_end]
testA  <- model_data[as.Date(interval) >= testA_start & as.Date(interval) <= testA_end]

trainB <- model_data[as.Date(interval) <= trainB_end]
testB  <- model_data[as.Date(interval) >= testB_start & as.Date(interval) <= testB_end]

# Run rolling forecasts for both periods
cat("\n========== STARTING 24-HOUR ROLLING FORECASTS (NO RETRAINING) ==========\n")
cat("Fair comparison with LSTM: Train once, then roll through predictions\n\n")

resultsA <- run_rolling_forecasts(trainA, testA, "Period A (Stable)")
resultsB <- run_rolling_forecasts(trainB, testB, "Period B (Not Stable)")

# ===============================================================
# FINAL SUMMARY
# ===============================================================

cat("\n==================== FINAL SUMMARY ====================\n")

all_eval <- rbind(resultsA$eval, resultsB$eval)
print(all_eval)

cat("\n--- Performance by Period (24h-ahead) ---\n")

cat("\nPeriod A:\n")
for (i in 1:nrow(resultsA$eval)) {
  cat(sprintf("  %s: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f\n",
              resultsA$eval$Model[i],
              resultsA$eval$RMSE[i],
              resultsA$eval$MAE[i],
              resultsA$eval$MAPE[i],
              resultsA$eval$R2[i]))
  cat(sprintf("    Runtime: %.2f min (Train: %.2f + Predict: %.2f)\n",
              resultsA$eval$Runtime_min[i],
              resultsA$eval$Train_min[i],
              resultsA$eval$Predict_min[i]))
}

cat("\nPeriod B:\n")
for (i in 1:nrow(resultsB$eval)) {
  cat(sprintf("  %s: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f\n",
              resultsB$eval$Model[i],
              resultsB$eval$RMSE[i],
              resultsB$eval$MAE[i],
              resultsB$eval$MAPE[i],
              resultsB$eval$R2[i]))
  cat(sprintf("    Runtime: %.2f min (Train: %.2f + Predict: %.2f)\n",
              resultsB$eval$Runtime_min[i],
              resultsB$eval$Train_min[i],
              resultsB$eval$Predict_min[i]))
}

cat("\n--- Best Models by Period ---\n")
cat(sprintf("Period A Best: %s (RMSE=%.2f, MAPE=%.2f%%)\n", 
            resultsA$eval[which.min(RMSE), Model],
            resultsA$eval[which.min(RMSE), RMSE],
            resultsA$eval[which.min(RMSE), MAPE]))

cat(sprintf("Period B Best: %s (RMSE=%.2f, MAPE=%.2f%%)\n", 
            resultsB$eval[which.min(RMSE), Model],
            resultsB$eval[which.min(RMSE), RMSE],
            resultsB$eval[which.min(RMSE), MAPE]))

cat("\n24h-ahead rolling forecast pipeline complete (no retraining)!\n")
cat("This approach is directly comparable to LSTM:\n")
cat("  - Train once on training data\n")
cat("  - Roll through test set making predictions\n")
cat("  - Model structure stays fixed\n")