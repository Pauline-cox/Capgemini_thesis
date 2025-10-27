# ==============================================================
# Author: Pauline Cox
# Script: fc_hybrid.R
#
# Description: Implements a hybrid forecasting model combining a 
# pretrained SARIMAX baseline with an LSTM network trained on 
# residuals for 24-hour-ahead energy consumption prediction. 
# The hybrid approach captures both linear and nonlinear 
#
# Input: 
#   - Pretrained SARIMAX model results (RESULT_FC_SARIMAX)
#   - Functions from forecast_preparations.R
#   - Preprocessed and feature-engineered dataset (model_data)
#   - Optimal hyperparamters from BOA
#
# Output: 
#   - Hybrid forecasts, evaluation metrics and diagnostic plots
# ==============================================================

set.seed(1234)
tensorflow::set_random_seed(1234)

# --- Load pretrained SARIMAX results ---
sarimax_res <- RESULT_FC_SARIMAX
target_col <- "total_consumption_kWh"

# --- Hyperparameters (Bayesian optimization results) ---
LOOKBACK <- 168L
UNITS1   <- 78L
UNITS2   <- 29L
DROPOUT  <- 0.09418245
LR       <- 0.0005
BATCH    <- 58L
EPOCHS   <- 50L
HORIZON  <- 24L

# --- Evaluation helpers ---
mape <- function(actual, pred) mean(abs((actual - pred) / pmax(actual, 1e-6))) * 100

evaluate_forecast <- function(actual, pred, model_name) {
  rmse <- sqrt(mean((pred - actual)^2, na.rm = TRUE))
  mae  <- mean(abs(pred - actual), na.rm = TRUE)
  r2   <- 1 - sum((actual - pred)^2, na.rm = TRUE) / sum((actual - mean(actual))^2, na.rm = TRUE)
  map  <- mape(actual, pred)
  data.table(Model = model_name, RMSE = rmse, MAE = mae, R2 = r2, MAPE = map)
}

plot_forecast <- function(dt, model_name, label, color = "darkgreen") {
  ggplot(dt, aes(x = Time)) +
    geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
    geom_line(aes(y = Forecast, color = model_name), linewidth = 0.8) +
    scale_color_manual(values = c("Actual" = "black", model_name = color)) +
    labs(
      title = paste(model_name, "- 24h Ahead Forecast -", label),
      x = "Time (hours)",
      y = "Energy Consumption (kWh)",
      color = "Series"
    ) +
    theme_minimal(base_size = 12)
}

# --- Residual extraction and preprocessing ---
get_full_residuals <- function(period_obj, y_train_ts) {
  model <- period_obj$model
  fitted_vals <- as.numeric(fitted(model))
  actual_vals <- as.numeric(y_train_ts)
  n <- min(length(fitted_vals), length(actual_vals))
  train_resid <- actual_vals[1:n] - fitted_vals[1:n]
  f <- as.data.table(period_obj$forecasts)
  test_resid <- f$Actual - f$Forecast
  c(train_resid, test_resid)
}

add_residual_lags <- function(df, lags = c(24, 168)) {
  for (i in lags) df[[paste0("resid_lag", i)]] <- dplyr::lag(df$Residual, i)
  df
}

minmax_scale <- function(train, test) {
  mins <- apply(train, 2, min)
  maxs <- apply(train, 2, max)
  range <- pmax(maxs - mins, 1e-6)
  train_scaled <- sweep(train, 2, mins, "-")
  train_scaled <- sweep(train_scaled, 2, range, "/")
  test_scaled  <- sweep(test, 2, mins, "-")
  test_scaled  <- sweep(test_scaled, 2, range, "/")
  train_scaled[is.nan(train_scaled)] <- 0
  test_scaled[is.nan(test_scaled)] <- 0
  list(train = train_scaled, test = test_scaled)
}

# --- LSTM residual trainer with convergence plot ---
train_lstm_residual <- function(train_data, test_data, feature_columns) {
  x_train <- as.matrix(train_data[, ..feature_columns])
  y_train <- train_data$residual
  x_test  <- as.matrix(test_data[, ..feature_columns])
  
  sds <- apply(x_train, 2, sd)
  keep <- sds > 0
  x_train <- x_train[, keep, drop = FALSE]
  x_test  <- x_test[, keep, drop = FALSE]
  feature_columns <- feature_columns[keep]
  
  scaled <- minmax_scale(x_train, x_test)
  x_train <- scaled$train
  x_test  <- scaled$test
  
  n_features <- ncol(x_train)
  n_timesteps <- 1
  
  model <- keras_model_sequential() |>
    layer_lstm(units = UNITS1, input_shape = c(n_timesteps, n_features)) |>
    layer_dropout(DROPOUT) |>
    layer_dense(units = UNITS2, activation = "relu") |>
    layer_dropout(DROPOUT) |>
    layer_dense(units = 1)
  
  model |> compile(
    optimizer = optimizer_adam(learning_rate = LR, clipnorm = 1.0),
    loss = "mae",
    metrics = "mae"
  )
  
  cat("Training LSTM model on residuals...\n")
  train_start <- Sys.time()
  history <- model |> fit(
    array(x_train, dim = c(nrow(x_train), 1, n_features)),
    y_train,
    epochs = EPOCHS, batch_size = BATCH,
    validation_split = 0.1, verbose = 2,
    callbacks = list(callback_early_stopping(patience = 8, restore_best_weights = TRUE))
  )
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  
  # --- Convergence plot ---
  df <- data.frame(
    epoch = seq_along(history$metrics$loss),
    loss = history$metrics$loss,
    val_loss = history$metrics$val_loss
  )
  p_conv <- ggplot(df, aes(x = epoch)) +
    geom_line(aes(y = loss, color = "Training Loss"), size = 1) +
    geom_line(aes(y = val_loss, color = "Validation Loss"), linetype = "dashed", size = 1) +
    labs(
      title = "LSTM Residual Training Convergence",
      x = "Epoch",
      y = "Loss (MAE)",
      color = "Legend"
    ) +
    theme_minimal(base_size = 13) +
    theme(legend.position = "bottom")
  
  preds <- as.numeric(model |> predict(array(x_test, dim = c(nrow(x_test), 1, n_features))))
  runtime <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  
  list(
    forecasts = preds,
    model = model,
    runtime = runtime,
    train_time = train_time,
    convergence_plot = p_conv
  )
}

# --- Hybrid forecast wrapper ---
run_hybrid_residual_24h <- function(train, test, sarimax_period_obj, y_train_ts, label) {
  cat(sprintf("\n--- %s ---\n", label))
  overall_start <- Sys.time()
  
  full_resid <- get_full_residuals(sarimax_period_obj, y_train_ts)
  df <- rbind(train, test)
  df <- df[seq_len(length(full_resid)), ]
  df[, Residual := full_resid]
  df <- add_residual_lags(df, c(24, 168))
  df <- df[complete.cases(df)]
  
  n_train <- nrow(train)
  train_data <- df[1:(n_train - HORIZON)]
  test_data  <- df[(n_train - LOOKBACK + 1):nrow(df)]
  train_data$residual <- train_data$Residual
  test_data$residual  <- 0
  
  feature_columns <- c(
    "total_occupancy", "humidity", "co2", "sound", "lux",
    "wind_speed", "sunshine_minutes", "global_radiation", "humidity_percent",
    "weekend", "business_hours", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos", "month_cos",
    "lag_24", "lag_168", "rollmean_24",
    "holiday", "dst", "resid_lag24", "resid_lag168"
  )
  
  lstm_res <- train_lstm_residual(train_data, test_data, feature_columns)
  
  sarimax_fc <- sarimax_period_obj$forecasts$Forecast
  hybrid_fc  <- sarimax_fc + lstm_res$forecasts[seq_along(sarimax_fc)]
  actual     <- sarimax_period_obj$forecasts$Actual[seq_along(hybrid_fc)]
  
  eval <- evaluate_forecast(actual, hybrid_fc, "Hybrid_SARIMAX_LSTM_24h")
  eval[, `:=`(Runtime_min = lstm_res$runtime, Train_min = lstm_res$train_time, Period = label)]
  
  dt <- data.table(Time = seq_along(actual), Actual = actual, Forecast = hybrid_fc)
  p <- plot_forecast(dt, "Hybrid_SARIMAX_LSTM_24h", label, color = "darkgreen")
  
  print(p)
  print(lstm_res$convergence_plot)
  
  list(
    eval = eval,
    forecasts = dt,
    plot = p,
    convergence_plot = lstm_res$convergence_plot
  )
}

# --- Run both periods ---
splits <- split_periods(model_data)
resultsA_hybrid <- run_hybrid_residual_24h(
  splits$trainA, splits$testA,
  sarimax_res$period_A, sarimax_res$period_A$model$x,
  "Period A"
)
resultsB_hybrid <- run_hybrid_residual_24h(
  splits$trainB, splits$testB,
  sarimax_res$period_B, sarimax_res$period_B$model$x,
  "Period B"
)

all_eval_hybrid <- rbind(resultsA_hybrid$eval, resultsB_hybrid$eval)
print(all_eval_hybrid)

cat(sprintf("Model: %s\n", resultsA_hybrid$eval$Model[1]))
cat(sprintf(
  "Period A: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | Time=%.2fmin\n",
  resultsA_hybrid$eval$RMSE, resultsA_hybrid$eval$MAE, resultsA_hybrid$eval$MAPE,
  resultsA_hybrid$eval$R2, resultsA_hybrid$eval$Runtime_min
))
cat(sprintf(
  "Period B: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | Time=%.2fmin\n",
  resultsB_hybrid$eval$RMSE, resultsB_hybrid$eval$MAE, resultsB_hybrid$eval$MAPE,
  resultsB_hybrid$eval$R2, resultsB_hybrid$eval$Runtime_min
))

# --- Save results ---
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Results_Hybrid_SARIMAX_LSTM_24h_%s.rds", timestamp)

saveRDS(
  list(
    model = "Hybrid_SARIMAX_LSTM_24h",
    period_A = resultsA_hybrid,
    period_B = resultsB_hybrid,
    evaluations = all_eval_hybrid,
    parameters = list(
      lookback = LOOKBACK,
      horizon = HORIZON,
      units1 = UNITS1,
      units2 = UNITS2,
      dropout = DROPOUT,
      lr = LR,
      batch = BATCH,
      epochs = EPOCHS
    )
  ),
  file = save_name
)

cat(sprintf("\nResults and models saved to: %s\n", save_name))
