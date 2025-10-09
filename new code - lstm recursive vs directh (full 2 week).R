# ================================================================
# LSTM FORECAST COMPARISON — DIRECT vs RECURSIVE (Full 2-Week Horizon)
# ================================================================
suppressPackageStartupMessages({
  library(keras)
  library(tensorflow)
  library(recipes)
  library(data.table)
  library(ggplot2)
})

set.seed(42)
cat("\n>>> STARTING LSTM FORECAST COMPARISON (Full-Horizon, Timed) <<<\n")

# ================================================================
# PARAMETERS
# ================================================================
LOOKBACK <- 168L     # 7 days lookback (hourly)
HORIZON  <- 24L      # 24-hour forecast horizon
UNITS1   <- 64L
UNITS2   <- 32L
DROPOUT  <- 0.3
LR       <- 0.005
BATCH    <- 16L
EPOCHS   <- 100L
target_col <- "total_consumption_kWh"

# ================================================================
# PERIOD SELECTION
# ================================================================
test_start <- as.Date("2024-10-01")
test_end   <- as.Date("2024-10-14")

test_start <- as.Date("2024-12-18")
test_end   <- as.Date("2024-12-31")

train_end  <- test_start - 1

train <- model_data[date <= train_end]
test  <- model_data[date >= test_start & date <= test_end]

cat(sprintf("\nForecast period: %s–%s (train ≤ %s)\n", test_start, test_end, train_end))

# ================================================================
# FEATURE PREPARATION + SCALING
# ================================================================
feature_columns <- setdiff(names(model_data), c("interval", "date", target_col))
cat("Using", length(feature_columns), "input features.\n")

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
scale_y  <- function(y) (y - y_min) / (y_max - y_min + 1e-6)
invert_y <- function(y) y * (y_max - y_min) + y_min
y_train_scaled <- scale_y(y_train)

# ================================================================
# [STEP 1] TRAINING SEQUENCES — DIRECT MODEL
# ================================================================
cat("\n[STEP 1] Building training sequences for DIRECT model...\n")
n_seq <- nrow(X_train) - LOOKBACK - HORIZON + 1
X_dir <- array(NA_real_, dim = c(n_seq, LOOKBACK, ncol(X_train)))
Y_dir <- array(NA_real_, dim = c(n_seq, HORIZON))
for (i in 1:n_seq) {
  X_dir[i,,] <- as.matrix(X_train[i:(i + LOOKBACK - 1), ])
  Y_dir[i, ] <- y_train_scaled[(i + LOOKBACK):(i + LOOKBACK + HORIZON - 1)]
}
cat("Built", n_seq, "training sequences.\n")

# ================================================================
# [STEP 2] TRAIN DIRECT MODEL
# ================================================================
cat("\n[STEP 2] Training DIRECT LSTM model...\n")
t_start <- Sys.time()

model_direct <- keras_model_sequential() %>%
  layer_lstm(units = UNITS1, input_shape = c(LOOKBACK, ncol(X_train)), return_sequences = TRUE) %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_lstm(units = UNITS2, return_sequences = FALSE) %>%
  layer_dense(units = HORIZON, activation = "linear") %>%
  compile(optimizer = optimizer_nadam(learning_rate = LR),
          loss = "mse", metrics = "mae")

history_direct <- model_direct %>% fit(
  X_dir, Y_dir,
  epochs = EPOCHS, batch_size = BATCH,
  validation_split = 0.1, verbose = 1,
  callbacks = list(
    callback_early_stopping(patience = 12, restore_best_weights = TRUE),
    callback_reduce_lr_on_plateau(factor = 0.5, patience = 6)
  )
)

t_end <- Sys.time()
cat(sprintf("Training complete for DIRECT model. Runtime: %.2f min\n",
            as.numeric(difftime(t_end, t_start, units = "mins"))))

# ================================================================
# [STEP 3] FORECAST — DIRECT MODEL (correct full 336h coverage)
# ================================================================
cat("\n[STEP 3] Forecasting with DIRECT model (correct full 336h coverage)...\n")
t_start <- Sys.time()

X_all  <- rbind(X_train, X_test)
n_hist <- nrow(X_train)
n_test <- nrow(X_test)

pred_direct <- rep(NA_real_, n_test)

# start indices of each 24h block inside the test set
for (s in seq(1, n_test - HORIZON + 1, by = HORIZON)) {
  # end of lookback window just before hour s of the test
  hist_end  <- n_hist + s - 1
  hist_idx  <- (hist_end - LOOKBACK + 1):hist_end
  xin <- array(as.matrix(X_all[hist_idx, , drop = FALSE]),
               dim = c(1, LOOKBACK, ncol(X_all)))
  pred_block <- as.numeric(model_direct %>% predict(xin, verbose = 0))
  pred_direct[s:(s + HORIZON - 1)] <- invert_y(pred_block)
}

# if any tiny remainder (should not happen with 336 and 24), fill safely
if (anyNA(pred_direct)) {
  last_idx <- max(which(!is.na(pred_direct)))
  remaining <- n_test - last_idx
  if (remaining > 0) {
    hist_end <- n_hist + last_idx
    hist_idx <- (hist_end - LOOKBACK + 1):hist_end
    xin <- array(as.matrix(X_all[hist_idx, , drop = FALSE]),
                 dim = c(1, LOOKBACK, ncol(X_all)))
    pred_tail <- as.numeric(model_direct %>% predict(xin, verbose = 0))
    pred_direct[(last_idx + 1):n_test] <- invert_y(pred_tail[1:remaining])
  }
}

t_end <- Sys.time()
cat(sprintf("Forecast complete for DIRECT model. Runtime: %.2f sec\n",
            as.numeric(difftime(t_end, t_start, units = "secs"))))

# sanity check
cat("Direct forecast length:", length(pred_direct), 
    " | NAs:", sum(is.na(pred_direct)), "\n")

# ================================================================
# [STEP 4] TRAINING SEQUENCES — RECURSIVE MODEL
# ================================================================
cat("\n[STEP 4] Building training sequences for RECURSIVE model...\n")
n_seq2 <- nrow(X_train) - LOOKBACK
X_rec <- array(NA_real_, dim = c(n_seq2, LOOKBACK, ncol(X_train)))
Y_rec <- array(NA_real_, dim = c(n_seq2, 1))
for (i in 1:n_seq2) {
  X_rec[i,,] <- as.matrix(X_train[i:(i + LOOKBACK - 1), ])
  Y_rec[i, ] <- y_train_scaled[i + LOOKBACK]
}

# ================================================================
# [STEP 5] TRAIN RECURSIVE MODEL
# ================================================================
cat("\n[STEP 5] Training RECURSIVE LSTM model...\n")
t_start <- Sys.time()

model_rec <- keras_model_sequential() %>%
  layer_lstm(units = UNITS1, input_shape = c(LOOKBACK, ncol(X_train)), return_sequences = TRUE) %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_lstm(units = UNITS2, return_sequences = FALSE) %>%
  layer_dense(units = 1, activation = "linear") %>%
  compile(optimizer = optimizer_nadam(learning_rate = LR),
          loss = "mse", metrics = "mae")

history_rec <- model_rec %>% fit(
  X_rec, Y_rec,
  epochs = EPOCHS, batch_size = BATCH,
  validation_split = 0.1, verbose = 0,
  callbacks = list(
    callback_early_stopping(patience = 12, restore_best_weights = TRUE),
    callback_reduce_lr_on_plateau(factor = 0.5, patience = 6)
  )
)

t_end <- Sys.time()
cat(sprintf("Training complete for RECURSIVE model. Runtime: %.2f min\n",
            as.numeric(difftime(t_end, t_start, units = "mins"))))

# ================================================================
# [STEP 6] FORECAST — RECURSIVE MODEL (continuous 2-week recursion)
# ================================================================
cat("\n[STEP 6] Forecasting with RECURSIVE model (continuous recursion)...\n")
t_start <- Sys.time()

X_all2 <- rbind(X_train, X_test)
n_hist <- nrow(X_train)
n_test <- nrow(X_test)
pred_rec <- numeric(0)
pb <- txtProgressBar(min = 0, max = n_test, style = 3)

for (i in 1:n_test) {
  hist_idx <- (n_hist + i - LOOKBACK):(n_hist + i - 1)
  xin <- array(as.matrix(X_all2[hist_idx, , drop = FALSE]),
               dim = c(1, LOOKBACK, ncol(X_all2)))
  pred <- model_rec %>% predict(xin, verbose = 0)
  pred_rec <- c(pred_rec, invert_y(pred))
  setTxtProgressBar(pb, i)
}
close(pb)

t_end <- Sys.time()
cat(sprintf("Forecast complete for RECURSIVE model. Runtime: %.2f sec\n",
            as.numeric(difftime(t_end, t_start, units = "secs"))))

# ================================================================
# [STEP 7] EVALUATION
# ================================================================
safe_metrics <- function(actual, forecast) {
  RMSE <- sqrt(mean((actual - forecast)^2))
  MAE  <- mean(abs(actual - forecast))
  MAPE <- mean(abs((actual - forecast) / pmax(actual, 1e-3))) * 100
  R2   <- 1 - sum((actual - forecast)^2) / sum((actual - mean(actual))^2)
  data.table(RMSE=RMSE, MAE=MAE, MAPE=MAPE, R2=R2)
}w

metrics_direct <- safe_metrics(y_test, pred_direct)
metrics_rec    <- safe_metrics(y_test, pred_rec)

cat("\n=== DIRECT MODEL RESULTS ===\n"); print(metrics_direct)
cat("\n=== RECURSIVE MODEL RESULTS ===\n"); print(metrics_rec)

# ================================================================
# [STEP 8] PLOTTING
# ================================================================
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
  geom_line(aes(x = interval, y = actual), colour = "black", linewidth = 1) +
  geom_line(aes(x = interval, y = forecast, colour = Model), linewidth = 0.8) +
  labs(title = sprintf("LSTM Direct vs Recursive — %s to %s", test_start, test_end),
       x = NULL, y = "Energy Consumption (kWh)", colour = "Model") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "bottom")

cat("\n>>> SCRIPT COMPLETE ✅ — Full 2-week forecasts generated with runtimes.\n")
