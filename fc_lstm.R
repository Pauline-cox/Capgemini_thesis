# ==============================================================
# Author: Pauline Cox
# Script: fc_lstm_pure24.R
#
# Description: Implements an LSTM model for 24-hour-ahead direct 
# energy consumption forecasting. The model uses preprocessed and 
# feature-engineered inputs, is trained once per period, and generates 
# sequential forecasts for evaluation periods A and B.
#
# Input: 
#   - Preprocessed and feature-engineered dataset (model_data)
#   - Functions from forecast_preparations.R
#   - Optimal hyperparamaters from bayesian optimzation
#
# Output: 
#   - Forecast results, evaluation metrics and diagnostic plots
# ==============================================================

set.seed(1234)
tensorflow::set_random_seed(1234)

# --- Hyperparameters (BOA results) ---
LOOKBACK <- 168L
UNITS1   <- 130
UNITS2   <- 32
DROPOUT  <- 0.269
LR       <- 0.00441
BATCH    <- 32
EPOCHS   <- 50L
HORIZON  <- 24L

feature_columns <- c(
  "total_occupancy", "humidity", "co2", "sound", "lux",
  "wind_speed", "sunshine_minutes", "global_radiation", "humidity_percent",
  "weekend", "business_hours", "hour_sin", "hour_cos",
  "dow_sin", "dow_cos", "month_cos",
  "lag_24", "lag_168", "rollmean_24",
  "holiday", "dst"
)

# --- Build LSTM model ---

build_lstm_model <- function(input_shape, units1, units2, dropout, lr) {
  keras_model_sequential() %>%
    layer_lstm(units = units1, input_shape = input_shape, return_sequences = TRUE) %>%
    layer_dropout(rate = dropout) %>%
    layer_lstm(units = units2, return_sequences = FALSE) %>%
    layer_dense(units = 1, activation = "linear") %>%
    compile(
      optimizer = optimizer_adam(learning_rate = lr),
      loss = "mse",
      metrics = "mae"
    )
}

# --- Plot training convergence ---

plot_convergence <- function(history, title = "LSTM Training Convergence") {
  df <- data.frame(
    epoch = seq_along(history$metrics$loss),
    loss = history$metrics$loss,
    val_loss = history$metrics$val_loss
  )
  ggplot(df, aes(x = epoch)) +
    geom_line(aes(y = loss, color = "Training Loss"), size = 1) +
    geom_line(aes(y = val_loss, color = "Validation Loss"), size = 1, linetype = "dashed") +
    labs(title = title, x = "Epoch", y = "Loss (MSE)", color = "Legend") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "bottom")
}

# --- 24h ahead LSTM forecast ---

lstm_pure_24h_only <- function(train_data, test_data, feature_cols,
                               lookback, units1, units2, dropout, lr,
                               batch_size, epochs, horizon = 24) {
  cat("\n--- LSTM PURE 24-Hour-Ahead Forecast ---\n")
  
  overall_start <- Sys.time()
  train_y <- train_data[[target_col]]
  test_y  <- test_data[[target_col]]
  all_y   <- c(train_y, test_y)
  
  train_x <- as.matrix(train_data[, ..feature_cols])
  test_x  <- as.matrix(test_data[, ..feature_cols])
  all_x   <- rbind(train_x, test_x)
  
  # --- Min–Max scaling for predictors and target ---
  x_min <- apply(train_x, 2, min, na.rm = TRUE)
  x_max <- apply(train_x, 2, max, na.rm = TRUE)
  x_scaled <- sweep(all_x, 2, x_min, "-")
  x_scaled <- sweep(x_scaled, 2, (x_max - x_min + 1e-6), "/")
  x_scaled <- pmax(pmin(x_scaled, 1), 0)
  
  y_min <- min(train_y, na.rm = TRUE)
  y_max <- max(train_y, na.rm = TRUE)
  y_scaled <- (all_y - y_min) / (y_max - y_min + 1e-6)
  
  n_train <- nrow(train_data)
  n_test <- nrow(test_data)
  n_seq <- n_train - lookback - horizon
  cat(sprintf("Training sequences: %d | Features: %d\n", n_seq, length(feature_cols)))
  
  # --- Build training sequences ---
  X_train <- array(NA_real_, dim = c(n_seq, lookback + 1, length(feature_cols)))
  y_train <- numeric(n_seq)
  for (i in seq_len(n_seq)) {
    X_train[i, , ] <- rbind(
      x_scaled[i:(i + lookback - 1), ],
      x_scaled[i + lookback + horizon - 1, , drop = FALSE]
    )
    y_train[i] <- y_scaled[i + lookback + horizon - 1]
  }
  
  # --- Build and fit model ---
  model <- build_lstm_model(c(lookback + 1, length(feature_cols)), units1, units2, dropout, lr)
  train_start <- Sys.time()
  history <- model %>% fit(
    X_train, y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0.2,
    verbose = 2,
    callbacks = list(callback_early_stopping(patience = 10, restore_best_weights = TRUE))
  )
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  cat(sprintf("Training complete in %.2f min\n", train_time))
  
  # --- Plot convergence ---
  p_conv <- plot_convergence(history, "LSTM Pure 24h Training Convergence")
  print(p_conv)
  
  # --- Forecast generation ---
  cat("Generating 24-hour-ahead forecasts...\n")
  start_idx <- n_train - (horizon - 1)
  end_idx   <- nrow(x_scaled) - horizon
  n_windows <- end_idx - start_idx + 1
  forecasts <- rep(NA_real_, n_test)
  filled <- 0
  
  for (i in seq_len(n_windows)) {
    idx <- start_idx + i - 1
    window_start <- idx - lookback + 1
    window_end   <- idx
    X_pred <- rbind(
      x_scaled[window_start:window_end, , drop = FALSE],
      x_scaled[idx + horizon, , drop = FALSE]
    )
    pred <- predict(model, array(X_pred, dim = c(1, lookback + 1, ncol(x_scaled))), verbose = 0)
    y_pred <- pred[1] * (y_max - y_min + 1e-6) + y_min
    target_idx <- idx - n_train + horizon
    if (target_idx >= 1 && target_idx <= n_test) {
      forecasts[target_idx] <- y_pred
      filled <- filled + 1
    }
    if (filled %% 24 == 0 || filled == n_test) {
      elapsed <- as.numeric(difftime(Sys.time(), overall_start, units = "mins"))
      pct <- (filled / n_test) * 100
      cat(sprintf("Progress: %3d/%d (%.1f%%) | Elapsed: %.2f min\n", filled, n_test, pct, elapsed))
      flush.console()
    }
  }
  
  forecasts <- zoo::na.locf0(forecasts)
  total_time <- as.numeric(difftime(Sys.time(), overall_start, units = "mins"))
  k_clear_session()
  
  list(
    forecasts = forecasts,
    runtime = total_time,
    train_time = train_time,
    history = list(
      loss = history$metrics$loss,
      val_loss = history$metrics$val_loss,
      epochs = seq_along(history$metrics$loss)
    ),
    convergence_plot = p_conv
  )
}

# --- Runner for each period ---

run_lstm_pure24 <- function(train, test, label) {
  cat(sprintf("\n--- %s ---\n", label))
  res <- lstm_pure_24h_only(train, test, feature_columns, LOOKBACK, UNITS1, UNITS2,
                            DROPOUT, LR, BATCH, EPOCHS, HORIZON)
  actual <- test[[target_col]]
  
  eval <- evaluate_forecast(actual, res$forecasts, "LSTM_Pure_24h")
  eval[, `:=`(Runtime_min = res$runtime, Train_min = res$train_time, Period = label)]
  
  dt <- data.table(Time = seq_along(actual), Actual = actual, Forecast = res$forecasts)
  p_forecast <- plot_forecast(dt, "LSTM_Pure_24h", label, color = "blue")
  
  print(p_forecast)
  print(res$convergence_plot)
  
  list(
    eval = eval,
    forecasts = dt,
    plot_forecast = p_forecast,
    plot_convergence = res$convergence_plot,
    history = res$history
  )
}

# --- Execution ---

splits <- split_periods(model_data)
resultsA_lstm <- run_lstm_pure24(splits$trainA, splits$testA, "Period A")
resultsB_lstm <- run_lstm_pure24(splits$trainB, splits$testB, "Period B")

all_eval_lstm <- rbind(resultsA_lstm$eval, resultsB_lstm$eval)
print(all_eval_lstm)

cat(sprintf("Model: %s\n", resultsA_lstm$eval$Model[1]))
cat(sprintf(
  "Period A: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | Time=%.2fmin\n",
  resultsA_lstm$eval$RMSE, resultsA_lstm$eval$MAE, resultsA_lstm$eval$MAPE,
  resultsA_lstm$eval$R2, resultsA_lstm$eval$Runtime_min
))
cat(sprintf(
  "Period B: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | Time=%.2fmin\n",
  resultsB_lstm$eval$RMSE, resultsB_lstm$eval$MAE, resultsB_lstm$eval$MAPE,
  resultsB_lstm$eval$R2, resultsB_lstm$eval$Runtime_min
))

# --- Save results ---

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Results_LSTM_Pure_24h_%s.rds", timestamp)

saveRDS(
  list(
    model = "LSTM_Pure_24h",
    period_A = resultsA_lstm,
    period_B = resultsB_lstm,
    evaluations = all_eval_lstm,
    parameters = list(
      lookback = LOOKBACK,
      horizon = HORIZON,
      units1 = UNITS1,
      units2 = UNITS2,
      dropout = DROPOUT,
      lr = LR,
      batch = BATCH,
      epochs = EPOCHS
    ),
    history_A = resultsA_lstm$history,
    history_B = resultsB_lstm$history
  ),
  file = save_name
)

cat(sprintf("\nResults and models saved to: %s\n", save_name))
