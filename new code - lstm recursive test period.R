# =====================================================================
# LSTM RECURSIVE FORECAST — Two Test Periods (A and B)
# =====================================================================
suppressPackageStartupMessages({
  library(keras)
  library(tensorflow)
  library(recipes)
  library(data.table)
  library(ggplot2)
})

set.seed(42)
cat("\n>>> STARTING RECURSIVE LSTM FORECAST FOR TEST A & B <<<\n")

# =====================================================================
# PARAMETERS
# =====================================================================
LOOKBACK <- 168L    # 1 week lookback
HORIZON  <- 336L    # forecast 2 weeks (for convenience)
UNITS1   <- 64L
UNITS2   <- 32L
DROPOUT  <- 0.3
LR       <- 0.005
BATCH    <- 16L
EPOCHS   <- 100L
target_col <- "total_consumption_kWh"

# =====================================================================
# DEFINE TEST PERIODS
# =====================================================================
testA_start <- as.Date("2024-10-01")
testA_end   <- as.Date("2024-10-14")
testB_start <- as.Date("2024-12-18")
testB_end   <- as.Date("2024-12-31")

trainA_end  <- testA_start - 1
trainB_end  <- testB_start - 1

cat(sprintf("Test A: %s–%s (train ≤ %s)\n", testA_start, testA_end, trainA_end))
cat(sprintf("Test B: %s–%s (train ≤ %s)\n\n", testB_start, testB_end, trainB_end))

# =====================================================================
# FUNCTION TO RUN ONE PERIOD
# =====================================================================
run_recursive_period <- function(train, test, label) {
  cat(sprintf("\n================ %s ================\n", label))
  
  # --------------------------------------------------------------
  # Feature preprocessing
  # --------------------------------------------------------------
  feature_columns <- setdiff(names(model_data), c("interval", "date", target_col))
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
  
  # --------------------------------------------------------------
  # Build sequences
  # --------------------------------------------------------------
  n_seq <- nrow(X_train) - LOOKBACK
  X_arr <- array(NA_real_, dim = c(n_seq, LOOKBACK, ncol(X_train)))
  Y_arr <- array(NA_real_, dim = c(n_seq, 1))
  
  for (i in 1:n_seq) {
    X_arr[i,,] <- as.matrix(X_train[i:(i + LOOKBACK - 1), ])
    Y_arr[i, ] <- y_train_scaled[i + LOOKBACK]
  }
  
  # --------------------------------------------------------------
  # Train model
  # --------------------------------------------------------------
  cat("Training recursive model...\n")
  t_start <- Sys.time()
  
  model_rec <- keras_model_sequential() %>%
    layer_lstm(units = UNITS1, input_shape = c(LOOKBACK, ncol(X_train)), return_sequences = TRUE) %>%
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
  cat(sprintf("Training time: %.2f min\n", as.numeric(difftime(t_end, t_start, units = "mins"))))
  
  # --------------------------------------------------------------
  # Forecast recursively (entire test period)
  # --------------------------------------------------------------
  cat("Forecasting recursively for full 2-week horizon...\n")
  t_start <- Sys.time()
  
  X_all <- rbind(X_train, X_test)
  n_hist <- nrow(X_train)
  n_test <- nrow(X_test)
  
  pred_rec <- numeric(0)
  current_window <- as.matrix(X_all[(n_hist - LOOKBACK + 1):n_hist, , drop = FALSE])
  
  for (h in 1:n_test) {
    xin_array <- array(current_window, dim = c(1, LOOKBACK, ncol(X_all)))
    p <- model_rec %>% predict(xin_array, verbose = 0)
    p_val <- as.numeric(invert_y(p))
    pred_rec <- c(pred_rec, p_val)
    
    # advance window using real exogenous vars for t+h
    if (h < n_test) {
      next_row <- as.numeric(X_test[h, ])
      current_window <- rbind(current_window[-1, ], next_row)
    }
  }
  
  t_end <- Sys.time()
  cat(sprintf("Forecast time: %.2f sec\n", as.numeric(difftime(t_end, t_start, units = "secs"))))
  
  # --------------------------------------------------------------
  # Evaluation
  # --------------------------------------------------------------
  RMSE <- sqrt(mean((y_test - pred_rec)^2))
  MAE  <- mean(abs(y_test - pred_rec))
  MAPE <- mean(abs((y_test - pred_rec) / pmax(y_test, 1e-3))) * 100
  R2   <- 1 - sum((y_test - pred_rec)^2) / sum((y_test - mean(y_test))^2)
  metrics <- data.table(Period = label, RMSE = RMSE, MAE = MAE, MAPE = MAPE, R2 = R2)
  print(metrics)
  
  # --------------------------------------------------------------
  # Plot
  # --------------------------------------------------------------
  plot_data <- data.table(interval = test$interval,
                          actual = y_test,
                          forecast = pred_rec)
  
  p <- ggplot(plot_data) +
    geom_line(aes(x = interval, y = actual), colour = "black", size = 1) +
    geom_line(aes(x = interval, y = forecast), colour = "steelblue", size = 0.8) +
    labs(title = sprintf("Recursive LSTM — %s", label),
         x = NULL, y = "Energy Consumption (kWh)") +
    theme_minimal(base_size = 13)
  
  print(p)
  
  return(metrics)
}

# =====================================================================
# RUN BOTH PERIODS
# =====================================================================
trainA <- model_data[date <= trainA_end]
testA  <- model_data[date >= testA_start & date <= testA_end]

trainB <- model_data[date <= trainB_end]
testB  <- model_data[date >= testB_start & date <= testB_end]

resultsA <- run_recursive_period(trainA, testA, "Test A (Oct 2024)")
resultsB <- run_recursive_period(trainB, testB, "Test B (Dec 2024)")

# =====================================================================
# FINAL COMPARISON
# =====================================================================
cat("\n==================== FINAL COMPARISON ====================\n")
results_all <- rbind(resultsA, resultsB)
print(results_all)

cat("\n>>> SCRIPT COMPLETE ✅ — Recursive forecasts for both test periods done.\n")
