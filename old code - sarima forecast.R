# SARIMA forecast function

sarima_forecast <- function(train_data, test_data, order, seasonal_order) {
  t0 <- Sys.time()
  
  # Prepare data  
  y_train <- train_data$total_consumption_kWh
  y_test  <- test_data$total_consumption_kWh
  y_train_ts <- ts(y_train, frequency = 168)
  
  # Fit SARIMA model  
  fit <- Arima(y_train_ts,
               order = order,
               seasonal = list(order = seasonal_order, period = 168),
               method = "CSS")
  
  # Forecast  
  fc <- forecast(fit, h = length(y_test))$mean
  
  # Runtime  
  runtime <- difftime(Sys.time(), t0, units = "secs")
  
  # Return forecast, model, and runtime  
  list(
    forecast = as.numeric(fc),
    model    = fit,
    runtime  = as.numeric(runtime)
  )
}

