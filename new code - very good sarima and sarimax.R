# ===============================================================
# SARIMAX vs SARIMA COMPARISON SCRIPT (with custom xreg features)
# ===============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(forecast)
  library(ggplot2)
  library(lubridate)
})

# ===============================================================
# 0) SETTINGS
# ===============================================================
ORDER    <- c(2,0,2)
SEASONAL <- c(0,1,1)
PERIOD   <- 168L  # weekly seasonality (hourly data)

testA_start <- as.Date("2024-10-01")
testA_end   <- as.Date("2024-10-14")
trainA_end  <- testA_start - 1

testB_start <- as.Date("2024-12-18")
testB_end   <- as.Date("2024-12-31")
trainB_end  <- testB_start - 1

target_col <- "total_consumption_kWh"

# Your selected regressors
selected_xreg <- c("co2", "business_hours", "total_occupancy",
                   "lux", "dow_cos", "hour_cos", "hour_sin",
                   "holiday", "dst")

# ===============================================================
# 1) Define MAPE function
# ===============================================================
mape <- function(actual, pred) {
  mean(abs((actual - pred) / actual), na.rm = TRUE) * 100
}

# ===============================================================
# 2) Comparison Function
# ===============================================================
compare_sarimax_sarima <- function(dt, target_col, features,
                                   order, seasonal, period,
                                   train_end, test_start, test_end,
                                   plot_title = "Forecast Comparison: SARIMAX vs SARIMA") {
  cat("\n===============================================================\n")
  cat(sprintf("Running comparison for period: %s → %s\n",
              format(test_start), format(test_end)))
  cat("===============================================================\n")
  
  train_data <- dt[as.Date(interval) <= train_end]
  test_data  <- dt[as.Date(interval) >= test_start & as.Date(interval) <= test_end]
  
  y_train <- ts(train_data[[target_col]], frequency = period)
  y_test  <- test_data[[target_col]]
  
  # --- SARIMAX ---
  x_train <- scale(as.matrix(train_data[, ..features]))
  x_test  <- scale(as.matrix(test_data[, ..features]),
                   center = attr(x_train, "scaled:center"),
                   scale  = attr(x_train, "scaled:scale"))
  
  cat("Fitting SARIMAX model...\n")
  fit_sarimax <- Arima(y_train, order = order,
                       seasonal = list(order = seasonal, period = period),
                       xreg = x_train, method = "CSS")
  fc_sarimax <- forecast(fit_sarimax, xreg = x_test, h = length(y_test))
  
  # --- Pure SARIMA ---
  cat("Fitting SARIMA model (no exogenous variables)...\n")
  fit_sarima <- Arima(y_train, order = order,
                      seasonal = list(order = seasonal, period = period),
                      method = "CSS")
  fc_sarima <- forecast(fit_sarima, h = length(y_test))
  
  # --- Results ---
  results <- data.table(
    interval = test_data$interval,
    actual   = y_test,
    sarimax  = as.numeric(fc_sarimax$mean),
    sarima   = as.numeric(fc_sarima$mean)
  )
  
  # --- Metrics ---
  cat("\nModel Comparison Metrics:\n")
  for (m in c("sarimax", "sarima")) {
    rmse <- sqrt(mean((results[[m]] - results$actual)^2, na.rm = TRUE))
    mae  <- mean(abs(results[[m]] - results$actual), na.rm = TRUE)
    r2   <- 1 - sum((results[[m]] - results$actual)^2) /
      sum((results$actual - mean(results$actual))^2)
    map  <- mape(results$actual, results[[m]])
    cat(sprintf("%-8s | RMSE: %.2f | MAE: %.2f | R²: %.3f | MAPE: %.2f%%\n",
                toupper(m), rmse, mae, r2, map))
  }
  
  # --- Plot ---
  p <- ggplot(results, aes(x = interval)) +
    geom_line(aes(y = actual, color = "Actual"), linewidth = 0.8) +
    geom_line(aes(y = sarimax, color = "SARIMAX"), linewidth = 0.8, alpha = 0.8) +
    geom_line(aes(y = sarima, color = "SARIMA"), linewidth = 0.8, alpha = 0.6, linetype = "dashed") +
    labs(title = plot_title,
         y = "Energy Consumption (kWh)", x = NULL) +
    scale_color_manual(values = c("Actual"="black", "SARIMAX"="blue", "SARIMA"="red")) +
    theme_minimal(base_size = 13) +
    theme(legend.title = element_blank())
  
  print(p)
  return(results)
}

# ===============================================================
# 3) Run comparisons
# ===============================================================
cat("\n=== Test A Comparison ===\n")
compare_sarimax_sarima(model_data, target_col, selected_xreg,
                       ORDER, SEASONAL, PERIOD,
                       trainA_end, testA_start, testA_end,
                       plot_title = "Forecast Comparison: Test A (SARIMAX vs SARIMA)")

cat("\n=== Test B Comparison ===\n")
compare_sarimax_sarima(model_data, target_col, selected_xreg,
                       ORDER, SEASONAL, PERIOD,
                       trainB_end, testB_start, testB_end,
                       plot_title = "Forecast Comparison: Test B (SARIMAX vs SARIMA)")
