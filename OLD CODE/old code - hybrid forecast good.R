# ===============================================================
# HYBRID SARIMAX–LSTM (LSTM on Residuals)
# ===============================================================

suppressPackageStartupMessages({
  library(forecast)
  library(data.table)
  library(keras)
  library(tensorflow)
  library(ggplot2)
  library(zoo)
})

set.seed(1234); tensorflow::set_random_seed(1234)

# ===============================================================
# PARAMETERS
# ===============================================================
ORDER    <- c(2, 0, 2)
SEASONAL <- c(0, 1, 1)
PERIOD   <- 168L
target_col <- "total_consumption_kWh"

xreg_cols <- c(
  "co2", "business_hours", "total_occupancy",
  "lux", "hour_cos", "hour_sin", "holiday", "dst"
)

LOOKBACK <- 168L
UNITS1 <- 32L
UNITS2 <- 32L
DROPOUT <- 0.2
LR <- 0.001
BATCH <- 32L
EPOCHS <- 40L

# ===============================================================
# EVALUATION HELPERS
# ===============================================================
mape <- function(actual, pred) mean(abs((actual - pred) / pmax(actual, 1e-6))) * 100
evaluate <- function(actual, pred, name) {
  data.table(
    Model = name,
    RMSE = sqrt(mean((pred - actual)^2, na.rm = TRUE)),
    MAE = mean(abs(pred - actual), na.rm = TRUE),
    R2 = 1 - sum((actual - pred)^2, na.rm = TRUE) / sum((actual - mean(actual))^2, na.rm = TRUE),
    MAPE = mape(actual, pred)
  )
}

# ===============================================================
# LSTM BUILDER
# ===============================================================
build_lstm_model <- function(input_shape, units1, units2, dropout, lr) {
  keras_model_sequential() %>%
    layer_lstm(units = units1, input_shape = input_shape, return_sequences = TRUE) %>%
    layer_dropout(dropout) %>%
    layer_lstm(units = units2, return_sequences = FALSE) %>%
    layer_dense(units = 1, activation = "linear") %>%
    compile(optimizer = optimizer_adam(lr), loss = "mse", metrics = "mae")
}

# ===============================================================
# MAIN HYBRID PIPELINE
# ===============================================================
run_hybrid_sarimax_lstm <- function(train, test, label) {
  cat(sprintf("\n==================== HYBRID MODEL: %s ====================\n", label))
  
  # ---------- Step 1: SARIMAX Forecast ----------
  y_train <- train[[target_col]]
  x_train <- as.matrix(train[, ..xreg_cols])
  x_test  <- as.matrix(test[, ..xreg_cols])
  
  x_train_s <- scale(x_train)
  center <- attr(x_train_s, "scaled:center"); scalev <- attr(x_train_s, "scaled:scale")
  x_test_s <- scale(x_test, center = center, scale = scalev)
  
  y_train_ts <- ts(y_train, frequency = PERIOD)
  
  cat("Training SARIMAX...\n")
  sarimax_model <- Arima(
    y_train_ts, order = ORDER, 
    seasonal = list(order = SEASONAL, period = PERIOD),
    xreg = x_train_s, method = "CSS"
  )
  fc_sarimax <- forecast(sarimax_model, xreg = x_test_s, h = nrow(test))
  sarimax_pred <- as.numeric(fc_sarimax$mean)
  
  # ---------- Step 2: Residual Calculation ----------
  train$resid <- residuals(sarimax_model)
  test$resid <- NA_real_
  
  # ---------- Step 3: LSTM on Residuals ----------
  cat("Training LSTM on residuals...\n")
  
  y_resid <- train$resid
  all_x <- rbind(x_train, x_test)
  all_x_s <- scale(all_x, center = center, scale = scalev)
  y_resid_all <- c(y_resid, rep(NA, nrow(test)))
  
  n_train <- nrow(train); n_test <- nrow(test)
  n_feat <- ncol(all_x_s)
  
  # Create training sequences
  n_seq <- n_train - LOOKBACK - 24
  X_train <- array(NA_real_, c(n_seq, LOOKBACK, n_feat))
  y_train_resid <- numeric(n_seq)
  for (i in seq_len(n_seq)) {
    X_train[i,,] <- all_x_s[i:(i+LOOKBACK-1),]
    y_train_resid[i] <- y_resid[i + LOOKBACK]
  }
  
  model <- build_lstm_model(c(LOOKBACK, n_feat), UNITS1, UNITS2, DROPOUT, LR)
  history <- model %>% fit(
    X_train, y_train_resid, 
    epochs = EPOCHS, batch_size = BATCH, 
    validation_split = 0.2, verbose = 0,
    callbacks = list(callback_early_stopping(patience = 10, restore_best_weights = TRUE))
  )
  
  # ---------- Step 4: Predict residuals on test ----------
  X_test <- array(NA_real_, c(n_test, LOOKBACK, n_feat))
  for (i in seq_len(n_test)) {
    start_idx <- n_train - LOOKBACK + i
    X_test[i,,] <- all_x_s[start_idx:(start_idx + LOOKBACK - 1),]
  }
  resid_pred <- as.numeric(predict(model, X_test))
  
  # ---------- Step 5: Hybrid forecast ----------
  hybrid_pred <- sarimax_pred + resid_pred
  
  # ---------- Step 6: Evaluate ----------
  actual <- test[[target_col]]
  eval_sarimax <- evaluate(actual, sarimax_pred, "SARIMAX")
  eval_hybrid  <- evaluate(actual, hybrid_pred, "HYBRID_SARIMAX+LSTM")
  
  eval <- rbind(eval_sarimax, eval_hybrid)
  eval[, Period := label]
  print(eval)
  
  dt_plot <- data.table(
    Time = seq_along(actual),
    Actual = actual,
    SARIMAX = sarimax_pred,
    HYBRID = hybrid_pred
  )
  
  p <- ggplot(dt_plot, aes(x = Time)) +
    geom_line(aes(y = Actual, color = "Actual")) +
    geom_line(aes(y = SARIMAX, color = "SARIMAX")) +
    geom_line(aes(y = HYBRID, color = "Hybrid")) +
    scale_color_manual(values = c("Actual" = "black", "SARIMAX" = "red", "Hybrid" = "blue")) +
    labs(title = paste("Hybrid SARIMAX–LSTM:", label),
         subtitle = "LSTM trained on SARIMAX residuals",
         x = "Time (hours)", y = "Energy Consumption (kWh)") +
    theme_minimal(base_size = 12)
  
  print(p)
  
  k_clear_session()
  return(list(eval = eval, forecasts = dt_plot, plot = p))
}

# ===============================================================
# RUN FOR PERIOD A AND B
# ===============================================================

testA_start <- as.Date("2024-10-01")
testA_end   <- as.Date("2024-10-14")
trainA_end  <- testA_start - 1

testB_start <- as.Date("2024-12-18")
testB_end   <- as.Date("2024-12-31")
trainB_end  <- testB_start - 1

trainA <- model_data[as.Date(interval) <= trainA_end]
testA  <- model_data[as.Date(interval) >= testA_start & as.Date(interval) <= testA_end]

trainB <- model_data[as.Date(interval) <= trainB_end]
testB  <- model_data[as.Date(interval) >= testB_start & as.Date(interval) <= testB_end]

resA <- run_hybrid_sarimax_lstm(trainA, testA, "Period A (Stable)")
resB <- run_hybrid_sarimax_lstm(trainB, testB, "Period B (Not Stable)")

all_eval <- rbind(resA$eval, resB$eval)
print(all_eval)
