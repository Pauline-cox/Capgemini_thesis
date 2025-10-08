# ==============================================================================
# SARIMAX FORECAST FUNCTION
# ==============================================================================

sarimax_forecast <- function(model_name, train_data, test_data, xreg_train, xreg_test, seasonal_period = 168,
                             order, seasonal) {
  
  y_train <- train_data$total_consumption_kWh
  y_test     <- test_data$total_consumption_kWh
  y_tv_ts    <- ts(y_train, frequency = seasonal_period)
  
  fit <- Arima(y_tv_ts, order = order,
               seasonal = list(order = seasonal, period = seasonal_period),
               xreg = xreg_train, method = "CSS")
  
  fc <- forecast(fit, xreg = xreg_test, h = length(y_test))$mean
  
  rmse <- sqrt(mean((y_test - fc)^2))
  mae  <- mean(abs(y_test - fc))
  mape <- mean(abs((y_test - fc) / y_test)) * 100
  r2   <- 1 - sum((y_test - fc)^2) / sum((y_test - mean(y_test))^2)
  
  metrics <- data.table(Model=model_name, RMSE=rmse, MAE=mae, MAPE=mape, R2=r2)
  print(metrics)
  
  list(forecast = ts(fc, start = 1, frequency = 1), metrics = metrics)
}
