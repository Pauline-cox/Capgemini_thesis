# Best hyperparameters from Bayesian Optimization

# ---------------------------------------------------------------
# MODEL: LSTM (Pure 24-Hour-Ahead Direct Forecast)
# Trains a single LSTM optimized for 24-hour horizon
# ---------------------------------------------------------------

set.seed(1234)
tensorflow::set_random_seed(1234)

# --- Hyperparameters ---
# LOOKBACK <- 168L
# UNITS1   <- 51L
# UNITS2   <- 63L
# DROPOUT  <- 0.208
# LR       <- 0.00243
# BATCH    <- 32L
# EPOCHS   <- 50L
# HORIZON  <- 24L
# 
# Model     RMSE      MAE        R2     MAPE Runtime_min Train_min                Period
# <char>    <num>    <num>     <num>    <num>       <num>     <num>                <char>
#   1: LSTM_Pure_24h 43.77738 31.39975 0.9364986 22.00875    15.16665  14.75475     Period A (Stable)
# 2: LSTM_Pure_24h 86.12249 54.27386 0.6916632 29.42742    22.95137  22.58912 Period B (Not Stable)

# LOOKBACK = 168
# UNITS1 = 78
# UNITS2 = 29
# DROPOUT = 0.09418245
# LR = 0.003896599
# BATCH = 58
# EPOCHS = 50       
# HORIZON = 24      
target_col <- "total_consumption_kWh"

# print(all_eval_lstm)
# Model     RMSE      MAE        R2     MAPE Runtime_min Train_min                Period
# <char>    <num>    <num>     <num>    <num>       <num>     <num>                <char>
#   1: LSTM_Pure_24h 37.64422 26.69364 0.9530452 18.29058    4.761067  4.302590     Period A (Stable)
# 2: LSTM_Pure_24h 67.85111 49.56896 0.8086158 26.71098    6.102128  5.739295 Period B (Not Stable)
# > 

LOOKBACK <- 168L
UNITS1   <- 128L
UNITS2   <- 64L
DROPOUT  <- 0.208
LR       <- 0.00243
BATCH    <- 32L
EPOCHS   <- 50L
HORIZON  <- 24L

# Model     RMSE      MAE        R2     MAPE Runtime_min Train_min                Period
# <char>    <num>    <num>     <num>    <num>       <num>     <num>                <char>
#   1: LSTM_Pure_24h 39.18153 23.31528 0.9491318 12.50146    26.58198  25.99752     Period A (Stable)
# 2: LSTM_Pure_24h 61.71443 41.54636 0.8416691 20.64878    33.18730  32.61440 Period B (Not Stable)

feature_columns <- c(
  "total_occupancy", "humidity", "co2", "sound", "lux",
  "wind_speed", "sunshine_minutes", "global_radiation", "humidity_percent",
  "weekend", "business_hours", "hour_sin", "hour_cos",
  "dow_sin", "dow_cos", "month_cos",
  "lag_24", "lag_168", "rollmean_24",
  "holiday", "dst"
)

LOOKBACK <- 168L
UNITS1   <- 130
UNITS2   <- 32
DROPOUT  <- 0.269
LR       <- 0.00241 #0.00441
BATCH    <- 63
EPOCHS   <- 50L
HORIZON  <- 24L

# Model     RMSE      MAE        R2     MAPE Runtime_min Train_min                Period
# <char>    <num>    <num>     <num>    <num>       <num>     <num>                <char>
#   1: LSTM_Pure_24h 39.64530 31.42511 0.9479177 28.84405    12.63486  12.14587     Period A (Stable)
# 2: LSTM_Pure_24h 57.69054 40.05431 0.8614274 20.34963    26.70902  26.22786 Period B (Not Stable)

# --- Build model ---
build_lstm_model <- function(input_shape, units1, units2, dropout, lr) {
  keras_model_sequential() %>%
    layer_lstm(units = units1, input_shape = input_shape, return_sequences = TRUE) %>%
    layer_dropout(rate = dropout) %>%
    layer_lstm(units = units2, return_sequences = FALSE) %>%
    layer_dense(units = 1, activation = "linear") %>%
    compile(optimizer = optimizer_adam(learning_rate = lr), loss = "mse", metrics = "mae")
}

# --- LSTM 24h direct forecast ---
lstm_pure_24h_only <- function(train_data, test_data, feature_cols,
                               lookback, units1, units2, dropout, lr,
                               batch_size, epochs, horizon = 24) {
  cat("\n>>> Starting LSTM PURE 24-Hour-Ahead Forecast <<<\n")
  
  overall_start <- Sys.time()
  train_y <- train_data[[target_col]]
  test_y  <- test_data[[target_col]]
  all_y   <- c(train_y, test_y)
  
  train_x <- as.matrix(train_data[, ..feature_cols])
  test_x  <- as.matrix(test_data[, ..feature_cols])
  all_x   <- rbind(train_x, test_x)
  
  x_scaled <- scale(all_x, center = colMeans(train_x), scale = apply(train_x, 2, sd))
  y_min <- min(train_y)
  y_max <- max(train_y)
  y_scaled <- (all_y - y_min) / (y_max - y_min + 1e-6)
  
  n_train <- nrow(train_data)
  n_test <- nrow(test_data)
  n_seq <- n_train - lookback - horizon
  
  cat(sprintf("Training sequences: %d | Features: %d\n", n_seq, length(feature_cols)))
  
  X_train <- array(NA_real_, dim = c(n_seq, lookback + 1, length(feature_cols)))
  y_train <- numeric(n_seq)
  for (i in seq_len(n_seq)) {
    X_train[i, , ] <- rbind(
      x_scaled[i:(i + lookback - 1), ],
      x_scaled[i + lookback + horizon - 1, , drop = FALSE]
    )
    y_train[i] <- y_scaled[i + lookback + horizon - 1]
  }
  
  model <- build_lstm_model(c(lookback + 1, length(feature_cols)), units1, units2, dropout, lr)
  train_start <- Sys.time()
  history <- model %>% fit(
    X_train, y_train,
    epochs = epochs, batch_size = batch_size, validation_split = 0.2, verbose = 1,
    callbacks = list(callback_early_stopping(patience = 10, restore_best_weights = TRUE))
  )
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  cat(sprintf("Training complete in %.2f min\n", train_time))
  
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
    pred <- predict(model, array(X_pred, dim = c(1, lookback + 1, ncol(x_scaled))), verbose = 1)
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
  list(forecasts = forecasts, runtime = total_time, train_time = train_time)
}

# --- Runner ---
run_lstm_pure24 <- function(train, test, label) {
  res <- lstm_pure_24h_only(train, test, feature_columns, LOOKBACK, UNITS1, UNITS2,
                            DROPOUT, LR, BATCH, EPOCHS, HORIZON)
  actual <- test[[target_col]]
  eval <- evaluate_forecast(actual, res$forecasts, "LSTM_Pure_24h")
  eval[, `:=`(Runtime_min = res$runtime, Train_min = res$train_time, Period = label)]
  
  dt <- data.table(Time = seq_along(actual), Actual = actual, Forecast = res$forecasts)
  p <- plot_forecast(dt, "LSTM_Pure_24h", label, color = "blue")
  print(p)
  
  list(eval = eval, forecasts = dt, plot = p)
}

# --- Execution ---
splits <- split_periods(model_data)
resultsA_lstm <- run_lstm_pure24(splits$trainA, splits$testA, "Period A (Stable)")
resultsB_lstm <- run_lstm_pure24(splits$trainB, splits$testB, "Period B (Not Stable)")

all_eval_lstm <- rbind(resultsA_lstm$eval, resultsB_lstm$eval)
print(all_eval_lstm)

cat("\n--- Summary ---\n")
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

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Results_LSTM_Pure_24h_%s.rds", timestamp)
saveRDS(
  list(model = "LSTM_Pure_24h", period_A = resultsA_lstm, period_B = resultsB_lstm,
       evaluations = all_eval_lstm, parameters = list(lookback = LOOKBACK, horizon = HORIZON)),
  file = save_name
)
cat(sprintf("\nResults saved to: %s\n", save_name))
cat("LSTM (Pure 24h) forecast complete!\n")