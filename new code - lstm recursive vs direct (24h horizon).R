# ================================================================
# LSTM FORECAST COMPARISON — DIRECT vs RECURSIVE (24h Horizon)
# ================================================================

suppressPackageStartupMessages({
  library(keras)
  library(tensorflow)
  library(recipes)
  library(data.table)
  library(ggplot2)
})

set.seed(42)
cat("\n>>> STARTING LSTM FORECAST COMPARISON <<<\n")

# ================================================================
# PARAMETERS
# ================================================================
LOOKBACK <- 168L     # past 7 days (hourly data)
HORIZON  <- 24L      # forecast 24 hours
UNITS1   <- 64L
UNITS2   <- 32L
DROPOUT  <- 0.3
LR       <- 0.005
BATCH    <- 16L
EPOCHS   <- 100L

target_col <- "total_consumption_kWh"

# ================================================================
# CHOOSE PERIOD (Test A or Test B)
# ================================================================
# test_start <- as.Date("2024-10-01")
# test_end   <- as.Date("2024-10-14")
# train_end  <- test_start - 1

test_start <- as.Date("2024-12-18")
test_end   <- as.Date("2024-12-31")
train_end  <- test_start - 1

train <- model_data[date <= train_end]
test  <- model_data[date >= test_start & date <= test_end]

cat(sprintf("\nSelected period: %s–%s (train ≤ %s)\n",
            test_start, test_end, train_end))

# ================================================================
# FEATURE SELECTION
# ================================================================
feature_columns <- setdiff(names(model_data),
                           c("interval", "date", target_col))
feature_columns <- intersect(feature_columns, names(model_data))

cat("Using", length(feature_columns), "features.\n")

# ================================================================
# DATA PREPROCESSING
# ================================================================
cat("\n[STEP 1] Preparing data and scaling...\n")

rec <- recipe(as.formula(paste(target_col, "~ .")),
              data = train[, c(target_col, feature_columns), with = FALSE]) %>%
  step_string2factor(all_nominal(), -all_outcomes()) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  step_range(all_numeric(), -all_outcomes()) %>%
  prep(training = train, retain = TRUE)

X_train <- bake(rec, new_data = train)[, setdiff(names(bake(rec, new_data = train)), target_col), with = FALSE]
X_test  <- bake(rec, new_data = test)[, setdiff(names(bake(rec, new_data = test)), target_col), with = FALSE]

y_train <- train[[target_col]]
y_test  <- test[[target_col]]

y_min <- min(y_train); y_max <- max(y_train)
scale_y <- function(y) (y - y_min) / (y_max - y_min + 1e-6)
invert_y <- function(y) y * (y_max - y_min) + y_min

y_train_scaled <- scale_y(y_train)

cat("Data preprocessing complete.\n")

# ================================================================
# [STEP 2] BUILD TRAINING SEQUENCES — DIRECT (multi-output)
# ================================================================
cat("\n[STEP 2] Building training sequences for DIRECT model...\n")

n_seq <- nrow(X_train) - LOOKBACK - HORIZON + 1
X_dir <- array(NA_real_, dim = c(n_seq, LOOKBACK, ncol(X_train)))
Y_dir <- array(NA_real_, dim = c(n_seq, HORIZON))

for (i in 1:n_seq) {
  X_dir[i,,] <- as.matrix(X_train[i:(i + LOOKBACK - 1), ])
  Y_dir[i, ] <- y_train_scaled[(i + LOOKBACK):(i + LOOKBACK + HORIZON - 1)]
}

cat("Built", n_seq, "training sequences.\n")

# ================================================================
# [STEP 3] TRAIN DIRECT MULTI-OUTPUT MODEL
# ================================================================
cat("\n[STEP 3] Training DIRECT LSTM model...\n")

model_direct <- keras_model_sequential() %>%
  layer_lstm(units = UNITS1, input_shape = c(LOOKBACK, ncol(X_train)), return_sequences = TRUE) %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_lstm(units = UNITS2, return_sequences = FALSE) %>%
  layer_dense(units = HORIZON, activation = "linear") %>%
  compile(optimizer = optimizer_nadam(learning_rate = LR),
          loss = "mse", metrics = "mae")

history_direct <- model_direct %>%
  fit(X_dir, Y_dir,
      epochs = EPOCHS, batch_size = BATCH,
      validation_split = 0.1, verbose = 1,
      callbacks = list(
        callback_early_stopping(patience = 12, restore_best_weights = TRUE),
        callback_reduce_lr_on_plateau(factor = 0.5, patience = 6)
      ))

cat("Training complete for DIRECT model.\n")
# ================================================================
# [STEP 4] FORECAST USING DIRECT MODEL — correctly aligned
# ================================================================
cat("\n[STEP 4] Forecasting with DIRECT model (correctly aligned 24h blocks)...\n")

history_all <- X_train
pred_direct <- numeric(0)
n_hist <- nrow(X_train)
n_test <- nrow(X_test)

for (i in seq(1, n_test, by = HORIZON)) {
  # Use the last LOOKBACK hours before the current test block
  hist_idx <- (n_hist + i - LOOKBACK):(n_hist + i - 1)
  xin <- array(as.matrix(rbind(X_train, X_test)[hist_idx, , drop = FALSE]),
               dim = c(1, LOOKBACK, ncol(X_train)))
  pred_block <- as.numeric(model_direct %>% predict(xin, verbose = 0))
  pred_direct <- c(pred_direct, invert_y(pred_block))
}
pred_direct <- pred_direct[1:n_test]


# ================================================================
# [STEP 5] TRAIN RECURSIVE (single-output) MODEL
# ================================================================
cat("\n[STEP 5] Building and training RECURSIVE model...\n")

n_seq2 <- nrow(X_train) - LOOKBACK
X_rec <- array(NA_real_, dim = c(n_seq2, LOOKBACK, ncol(X_train)))
Y_rec <- array(NA_real_, dim = c(n_seq2, 1))

for (i in 1:n_seq2) {
  X_rec[i,,] <- as.matrix(X_train[i:(i + LOOKBACK - 1), ])
  Y_rec[i, ] <- y_train_scaled[i + LOOKBACK]
}

model_rec <- keras_model_sequential() %>%
  layer_lstm(units = UNITS1, input_shape = c(LOOKBACK, ncol(X_train)), return_sequences = TRUE) %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_lstm(units = UNITS2, return_sequences = FALSE) %>%
  layer_dense(units = 1, activation = "linear") %>%
  compile(optimizer = optimizer_nadam(learning_rate = LR),
          loss = "mse", metrics = "mae")

history_rec <- model_rec %>%
  fit(X_rec, Y_rec,
      epochs = EPOCHS, batch_size = BATCH,
      validation_split = 0.1, verbose = 1,
      callbacks = list(
        callback_early_stopping(patience = 12, restore_best_weights = TRUE),
        callback_reduce_lr_on_plateau(factor = 0.5, patience = 6)
      ))

cat("Training complete for RECURSIVE model.\n")

# ================================================================
# [STEP 6] FORECAST USING RECURSIVE MODEL — continuous 24h recursion
# ================================================================
cat("\n[STEP 6] Forecasting with RECURSIVE model (proper hourly recursion)...\n")

X_all2 <- rbind(X_train, X_test)
n_hist <- nrow(X_train)
n_test <- nrow(X_test)
pred_rec <- numeric(0)

for (i in 1:n_test) {
  hist_idx <- (n_hist + i - LOOKBACK):(n_hist + i - 1)
  current_window <- as.matrix(X_all2[hist_idx, , drop = FALSE])
  
  # predict 1 step ahead
  xin_array <- array(current_window, dim = c(1, LOOKBACK, ncol(X_all2)))
  p <- model_rec %>% predict(xin_array, verbose = 0)
  p_val <- as.numeric(invert_y(p))
  pred_rec <- c(pred_rec, p_val)
  
  # update input window for next step:
  if (i < n_test) {
    next_row <- as.numeric(X_test[i, ])  # real exogenous vars at t+i
    current_window <- rbind(current_window[-1, ], next_row)
    X_all2 <- rbind(X_all2, next_row)    # expand context progressively
  }
}
pred_rec <- pred_rec[1:n_test]

# ================================================================
# [STEP 7] EVALUATION METRICS
# ================================================================
cat("\n[STEP 7] Evaluating models...\n")

safe_metrics <- function(actual, forecast) {
  actual <- as.numeric(actual)
  forecast <- as.numeric(forecast)
  bad <- is.na(actual) | is.na(forecast) | is.nan(actual) | is.nan(forecast)
  actual <- actual[!bad]; forecast <- forecast[!bad]
  RMSE <- sqrt(mean((actual - forecast)^2))
  MAE  <- mean(abs(actual - forecast))
  MAPE <- mean(abs((actual - forecast) / pmax(actual, 1e-3))) * 100
  R2   <- 1 - sum((actual - forecast)^2) / sum((actual - mean(actual))^2)
  data.table(RMSE = RMSE, MAE = MAE, MAPE = MAPE, R2 = R2)
}

metrics_direct <- safe_metrics(y_test, pred_direct)
metrics_rec    <- safe_metrics(y_test, pred_rec)

cat("\n=== DIRECT MODEL RESULTS ===\n"); print(metrics_direct)
cat("\n=== RECURSIVE MODEL RESULTS ===\n"); print(metrics_rec)

# ================================================================
# [STEP 8] PLOTTING
# ================================================================
cat("\n[STEP 8] Plotting forecasts...\n")

plot_data <- rbind(
  data.table(interval = test$interval,
             actual = y_test,
             forecast = pred_direct,
             Model = "LSTM_Direct"),
  data.table(interval = test$interval,
             actual = y_test,
             forecast = pred_rec,
             Model = "LSTM_Recursive")
)

ggplot(plot_data) +
  geom_line(aes(x = interval, y = actual), colour = "black", size = 1) +
  geom_line(aes(x = interval, y = forecast, colour = Model), size = 0.8) +
  labs(title = sprintf("LSTM Direct vs Recursive — %s to %s",
                       test_start, test_end),
       x = NULL, y = "Energy Consumption (kWh)",
       colour = "Model") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

cat("\n>>> SCRIPT COMPLETE ✅ — Both LSTM models trained, evaluated, and plotted.\n")
