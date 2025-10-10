# Function to evaluate forecasts

# --- Evaluation metrics ---

evaluate_all <- function(results_list, actual_values) {
  metrics <- lapply(names(results_list), function(name) {
    res <- results_list[[name]]
    
    if (is.null(res$forecast)) return(NULL)
    
    pred <- as.numeric(res$forecast)
    act  <- as.numeric(actual_values[1:length(pred)])
    
    rmse <- sqrt(mean((pred - act)^2, na.rm = TRUE))
    mae  <- mean(abs(pred - act), na.rm = TRUE)
    mape <- mean(abs((pred - act) / pmax(act, 1e-6)), na.rm = TRUE) * 100
    ss_res <- sum((act - pred)^2, na.rm = TRUE)
    ss_tot <- sum((act - mean(act, na.rm = TRUE))^2, na.rm = TRUE)
    r2 <- 1 - ss_res / ss_tot
    
    runtime <- if (!is.null(res$runtime)) as.numeric(res$runtime) else NA
    
    data.table(
      Model = name,
      RMSE = round(rmse, 3),
      MAE  = round(mae, 3),
      MAPE = round(mape, 3),
      R2   = round(r2, 4),
      Runtime_sec = round(runtime, 2)
    )
  })
  
  results <- rbindlist(metrics, fill = TRUE)
  setorder(results, RMSE)
  return(results)
}

# --- Plots forecast vs actual ---

plot_forecasts <- function(forecast_list, actual_values, title = "Forecast Comparison") {
 
  df_list <- lapply(names(forecast_list), function(name) {
    preds <- as.numeric(forecast_list[[name]]$forecast)
    data.table(Time = seq_along(preds), Forecast = preds, Model = name)
  })
  
  df <- rbindlist(df_list)
  df_actual <- data.table(Time = seq_along(actual_values), Actual = actual_values)
  
  ggplot() +
    geom_line(data = df_actual, aes(x = Time, y = Actual),
              color = "black", size = 1, linetype = "solid") +
    geom_line(data = df, aes(x = Time, y = Forecast, color = Model), size = 0.8) +
    labs(title = title,
         x = "Time (hours)",
         y = "Energy Consumption (kWh)",
         color = "Model") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "bottom")
}

