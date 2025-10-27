# ==============================================================
# Author: Pauline Cox
# Script: forecast_preparations.R
#
# Description: Defines common forecasting settings, data splits,
# evaluation metrics, and plotting utilities used across all 
# models
#
# Input: 
#   - Feature-engineered dataset
#
# Output: 
#   - Data split objects (trainA, testA, trainB, testB)
#   - Evaluation metric and plotting helper functions.
# ==============================================================

# --- Global Settings ---
ORDER    <- c(1, 0, 2)
SEASONAL <- c(1, 1, 1)
PERIOD   <- 168L
target_col <- "total_consumption_kWh"

# --- Data Split Function ---
split_periods <- function(data) {
  
  tz_ref <- attr(data$interval, "tzone")
  list(
    trainA = data[interval >= as.POSIXct("2023-07-01 00:00:00", tz = tz_ref) &
                    interval <  as.POSIXct("2024-10-01 00:00:00", tz = tz_ref)],
    testA  = data[interval >= as.POSIXct("2024-10-01 00:00:00", tz = tz_ref) &
                    interval <  as.POSIXct("2024-10-15 00:00:00", tz = tz_ref)],
    trainB = data[interval >= as.POSIXct("2023-07-01 00:00:00", tz = tz_ref) &
                    interval <  as.POSIXct("2024-12-18 00:00:00", tz = tz_ref)],
    testB  = data[interval >= as.POSIXct("2024-12-18 00:00:00", tz = tz_ref) &
                    interval <  as.POSIXct("2025-01-01 00:00:00", tz = tz_ref)]
  )
}
# --- Evaluation Metrics ---
mape <- function(actual, pred) {
  mean(abs((actual - pred) / pmax(actual, 1e-6)), na.rm = TRUE) * 100
}

evaluate_forecast <- function(actual, pred, model_name) {
  rmse <- sqrt(mean((pred - actual)^2, na.rm = TRUE))
  mae  <- mean(abs(pred - actual), na.rm = TRUE)
  r2   <- 1 - sum((actual - pred)^2, na.rm = TRUE) / sum((actual - mean(actual))^2, na.rm = TRUE)
  map  <- mape(actual, pred)
  data.table(Model = model_name, RMSE = rmse, MAE = mae, R2 = r2, MAPE = map)
}

# --- Plot Helper ---
plot_forecast <- function(dt, model_name, label, color = "blue") {
  ggplot(dt, aes(x = Time)) +
    geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
    geom_line(aes(y = Forecast, color = model_name), linewidth = 0.8) +
    scale_color_manual(values = c("Actual" = "black", model_name = color)) +
    labs(
      title = paste(model_name, "- 24h Ahead Forecast -", label),
      subtitle = "Train once, predict many times",
      x = "Time (hours)",
      y = "Energy Consumption (kWh)",
      color = "Series"
    ) +
    theme_minimal(base_size = 12)
}