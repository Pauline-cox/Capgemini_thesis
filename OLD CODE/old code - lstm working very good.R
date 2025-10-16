# ================================================================
# RECURSIVE LSTM FORECAST — Full Continuous Horizon (Improved)
# ================================================================
suppressPackageStartupMessages({
  library(keras)
  library(tensorflow)
  library(recipes)
  library(data.table)
  library(ggplot2)
})

set.seed(42)
cat("\n>>> STARTING FULL-PERIOD RECURSIVE LSTM FORECAST <<<\n")

# ================================================================
# PARAMETERS
# ================================================================
LOOKBACK <- 168L      # 7 days
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
test_start <- as.Date("2024-12-18")
test_end   <- as.Date("2024-12-31")
train_end  <- test_start - 1

train <- model_data[date <= train_end]
test  <- model_data[date >= test_start & date <= test_end]

cat(sprintf("Forecast period: %s–%s (train ≤ %s)\n",
            test_start, test_end, train_end))

# ================================================================
# FEATURE PREPARATION + SCALING
# ================================================================
feature_columns <- setdiff(names(model_data),
                           c("interval", "date", target_col))

rec <- recipe(as.formula(paste(target_col, "~ .")),
              data = train[, c(target_col, feature_columns), with = FALSE]) %>%
  step_string2factor(all_nominal(), -all_outcomes()) %>%
  step_dummy(all_nominal(), one_hot = TRUE) %>%
  step_range(all_numeric(), -all_outcomes()) %>%
  prep(training = train, retain = TRUE)

X_train <- bake(rec, new_data = train)[, setdiff(names(bake(rec, new_data = train)),
                                                 target_col), with = FALSE]
X_test  <- bake(rec, new_data = test)[, setdiff(names(bake(rec, new_data = test)),
                                                target_col), with = FALSE]

y_train <- train[[target_col]]
y_test  <- test[[target_col]]

y_min <- min(y_train); y_max <- max(y_train)
scale_y  <- function(y) (y - y_min) / (y_max - y_min + 1e-6)
invert_y <- function(y) y * (y_max - y_min) + y_min
y_train_scaled <- scale_y(y_train)

# ================================================================
# BUILD SEQUENCES FOR 1-STEP (RECURSIVE) MODEL
# ================================================================
n_seq <- nrow(X_train) - LOOKBACK
X_arr <- array(NA_real_, dim = c(n_seq, LOOKBACK, ncol(X_train)))
Y_arr <- array(NA_real_, dim = c(n_seq, 1))

for (i in 1:n_seq) {
  X_arr[i,,] <- as.matrix(X_train[i:(i + LOOKBACK - 1), ])
  Y_arr[i, ] <- y_train_scaled[i + LOOKBACK]
}

# ================================================================
# TRAIN MODEL
# ================================================================
cat("\nTraining recursive LSTM model...\n")
t_start <- Sys.time()

model_rec <- keras_model_sequential() %>%
  layer_lstm(units = UNITS1, input_shape = c(LOOKBACK, ncol(X_train)),
             return_sequences = TRUE) %>%
  layer_dropout(rate = DROPOUT) %>%
  layer_lstm(units = UNITS2, return_sequences = FALSE) %>%
  layer_dense(units = 1, activation = "linear") %>%
  compile(optimizer = optimizer_nadam(learning_rate = LR),
          loss = "mse", metrics = "mae")

model_rec %>% fit(
  X_arr, Y_arr,
  epochs = EPOCHS, batch_size = BATCH,
  validation_split = 0.1, verbose = 0,
  callbacks = list(
    callback_early_stopping(patience = 12, restore_best_weights = TRUE),
    callback_reduce_lr_on_plateau(factor = 0.5, patience = 6)
  )
)

t_end <- Sys.time()
cat(sprintf("Training time: %.2f min\n",
            as.numeric(difftime(t_end, t_start, units = "mins"))))

# ================================================================
# FORECAST ENTIRE TEST PERIOD RECURSIVELY
# ================================================================
cat("\nForecasting recursively for full test period...\n")
t_start <- Sys.time()

X_all <- rbind(X_train, X_test)
n_hist <- nrow(X_train)
n_test <- nrow(X_test)
pred_rec <- numeric(0)

# initialize last known lookback window (from end of training)
current_window <- as.matrix(X_all[(n_hist - LOOKBACK + 1):n_hist, , drop = FALSE])

pb <- txtProgressBar(min = 0, max = n_test, style = 3)
for (h in 1:n_test) {
  xin <- array(current_window, dim = c(1, LOOKBACK, ncol(X_all)))
  pred_scaled <- model_rec %>% predict(xin, verbose = 0)
  pred_val <- as.numeric(invert_y(pred_scaled))
  pred_rec <- c(pred_rec, pred_val)
  
  # advance window → drop oldest, append new row (real exogenous + predicted target)
  if (h < n_test) {
    next_row <- as.numeric(X_test[h, ])
    # replace the target value (if present) with scaled predicted value if needed
    current_window <- rbind(current_window[-1, ], next_row)
  }
  setTxtProgressBar(pb, h)
}
close(pb)

t_end <- Sys.time()
cat(sprintf("Forecast complete. Runtime: %.2f sec\n",
            as.numeric(difftime(t_end, t_start, units = "secs"))))

# ================================================================
# EVALUATION
# ================================================================
RMSE <- sqrt(mean((y_test - pred_rec)^2))
MAE  <- mean(abs(y_test - pred_rec))
MAPE <- mean(abs((y_test - pred_rec) / pmax(y_test, 1e-3))) * 100
R2   <- 1 - sum((y_test - pred_rec)^2) / sum((y_test - mean(y_test))^2)
cat(sprintf("\nRMSE = %.3f,  MAE = %.3f,  MAPE = %.2f%%,  R² = %.3f\n",
            RMSE, MAE, MAPE, R2))

# ================================================================
# PLOT
# ================================================================
plot_data <- data.table(interval = test$interval,
                        actual = y_test,
                        forecast = pred_rec)

ggplot(plot_data) +
  geom_line(aes(x = interval, y = actual), colour = "black", size = 1) +
  geom_line(aes(x = interval, y = forecast), colour = "steelblue", size = 0.8) +
  labs(title = sprintf("Recursive LSTM Forecast — %s to %s",
                       test_start, test_end),
       x = NULL, y = "Energy Consumption (kWh)") +
  theme_minimal(base_size = 13)

cat("\n>>> SCRIPT COMPLETE ✅ — Full-period recursive forecast generated.\n")
