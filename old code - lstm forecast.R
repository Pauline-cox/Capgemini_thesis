# ===============================================================
# LSTM 24-HOUR RECURSIVE FORECAST
# ===============================================================

set.seed(1234)
tensorflow::set_random_seed(1234)


# ===============================================================
# PASTE OPTIMAL HYPERPARAMETERS HERE (FROM BAYESIAN OPTIMIZATION)
# ===============================================================

LOOKBACK <- 168L
UNITS1   <- 51L         # CHANGE THIS
UNITS2   <- 63L         # CHANGE THIS
DROPOUT  <- 0.208       # CHANGE THIS
LR       <- 0.00243     # CHANGE THIS
BATCH    <- 32L         # CHANGE THIS
EPOCHS   <- 50L
HORIZON  <- 24L

target_col <- "total_consumption_kWh"

# ===============================================================

feature_columns <- c(
  "total_occupancy", "humidity", "co2", "sound", "lux",
  "wind_speed", "sunshine_minutes", "global_radiation", "humidity_percent",
  "weekend", "business_hours", "hour_sin", "hour_cos",
  "dow_sin", "dow_cos", "month_cos",
  "lag_24", "lag_168", "rollmean_24",
  "holiday", "dst"
)

# --- TEST PERIOD DEFINITIONS ---

testA_start <- as.Date("2024-10-01")
testA_end   <- as.Date("2024-10-14")
trainA_end  <- testA_start - 1

testB_start <- as.Date("2024-12-18")
testB_end   <- as.Date("2024-12-31")
trainB_end  <- testB_start - 1

# --- HELPER FUNCTIONS ---

mape <- function(actual, pred) {
  mean(abs((actual - pred) / pmax(actual, 1e-6)), na.rm = TRUE) * 100
}

evaluate_forecast <- function(actual, pred, model_name) {
  rmse <- sqrt(mean((pred - actual)^2, na.rm = TRUE))
  mae  <- mean(abs(pred - actual), na.rm = TRUE)
  r2   <- 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
  map  <- mape(actual, pred)
  data.table(Model = model_name, RMSE = rmse, MAE = mae, R2 = r2, MAPE = map)
}

build_lstm_model <- function(input_shape, units1, units2, dropout, lr) {
  model <- keras_model_sequential() %>%
    layer_lstm(units = units1, input_shape = input_shape, return_sequences = TRUE) %>%
    layer_dropout(rate = dropout) %>%
    layer_lstm(units = units2, return_sequences = FALSE) %>%
    layer_dense(units = 1, activation = "linear") %>%
    compile(
      optimizer = optimizer_adam(learning_rate = lr),
      loss = "mse",
      metrics = "mae"
    )
  return(model)
}

# --- RECURSIVE 24-HOUR ROLLING FORECAST ---

rolling_lstm_24h_recursive <- function(train_data, test_data, feature_cols, 
                                       lookback, units1, units2, dropout, lr, 
                                       batch_size, epochs, horizon = 24) {
  
  cat("\n>>> Starting RECURSIVE 24h-Ahead Rolling LSTM Forecast...\n")
  cat("Strategy: Train once for 1-step ahead, then recursively forecast 24 steps\n\n")
  
  n_test <- nrow(test_data)
  forecasts <- rep(NA_real_, n_test)
  
  overall_start <- Sys.time()
  
  # --- STEP 1: PREPARE TRAINING DATA ---
  
  cat("--- Step 1: Preparing training data ---\n")
  
  train_y <- train_data[[target_col]]
  train_x <- as.matrix(train_data[, ..feature_cols])
  
  # Scale features
  x_scaled <- scale(train_x)
  center_vec <- attr(x_scaled, "scaled:center")
  scale_vec  <- attr(x_scaled, "scaled:scale")
  
  # Scale target
  y_min <- min(train_y)
  y_max <- max(train_y)
  y_scaled <- (train_y - y_min) / (y_max - y_min + 1e-6)
  
  cat(sprintf("Training samples: %d\n", length(train_y)))
  cat(sprintf("Features: %d\n", length(feature_cols)))
  cat(sprintf("Target range: [%.2f, %.2f]\n", y_min, y_max))
  
  # --- STEP 2: CREATE TRAINING SEQUENCES (1-STEP AHEAD) ---
  
  cat("\n--- Step 2: Creating training sequences (1-step ahead) ---\n")
  
  n_seq <- length(y_scaled) - lookback
  
  if (n_seq <= 0) {
    stop("Not enough training data for the specified lookback")
  }
  
  X_train <- array(NA_real_, dim = c(n_seq, lookback, length(feature_cols)))
  y_train <- array(NA_real_, dim = c(n_seq, 1))
  
  for (i in seq_len(n_seq)) {
    X_train[i, , ] <- x_scaled[i:(i + lookback - 1), ]
    y_train[i, ] <- y_scaled[i + lookback]
  }
  
  cat(sprintf("Training sequences created: %d (1-step ahead)\n", n_seq))
  
  # --- STEP 3: TRAIN MODEL ONCE (FOR 1-STEP PREDICTION) ---
  
  cat("\n--- Step 3: Training LSTM model (1-STEP AHEAD) ---\n")
  train_start <- Sys.time()
  
  model <- build_lstm_model(
    input_shape = c(lookback, length(feature_cols)),
    units1 = units1,
    units2 = units2,
    dropout = dropout,
    lr = lr
  )
  
  cat(sprintf("Model architecture: LSTM(%d) -> Dropout(%.2f) -> LSTM(%d) -> Dense(1)\n",
              units1, dropout, units2))
  cat(sprintf("Training with %d epochs, batch size %d\n", epochs, batch_size))
  
  history <- model %>% fit(
    X_train, y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0.2,
    verbose = 0,
    callbacks = list(
      callback_early_stopping(patience = 10, restore_best_weights = TRUE),
      callback_reduce_lr_on_plateau(factor = 0.5, patience = 5)
    )
  )
  
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  cat(sprintf("\nTraining complete! Time: %.2f minutes\n", train_time))
  
  final_train_mae <- tail(history$metrics$mae, 1)
  final_val_mae <- tail(history$metrics$val_mae, 1)
  cat(sprintf("Final train MAE: %.4f | Final val MAE: %.4f\n", 
              final_train_mae, final_val_mae))
  
  # --- STEP 4: RECURSIVE ROLLING PREDICTIONS ON TEST SET ---
  
  cat("\n--- Step 4: Recursive 24h-ahead predictions through test set ---\n")
  predict_start <- Sys.time()
  
  # Combine train + test for rolling window
  all_y <- c(train_y, test_data[[target_col]])
  all_x <- rbind(train_x, as.matrix(test_data[, ..feature_cols]))
  
  # Scale all features using training statistics
  all_x_scaled <- scale(all_x, center = center_vec, scale = scale_vec)
  
  train_size <- length(train_y)
  
  # --- Identify target feature and lag indices ---
  # Feed back predicted energy consumption (not occupancy)
  target_in_features <- target_col %in% feature_cols
  if (target_in_features) {
    target_feature_idx <- which(feature_cols == target_col)
  }
  
  lag_24_idx       <- if ("lag_24"       %in% feature_cols) which(feature_cols == "lag_24")       else NULL
  lag_168_idx      <- if ("lag_168"      %in% feature_cols) which(feature_cols == "lag_168")      else NULL
  rollmean_24_idx  <- if ("rollmean_24"  %in% feature_cols) which(feature_cols == "rollmean_24")  else NULL
  
  # --- Main recursive forecast loop ---
  for (h in seq_len(n_test)) {
    # Progress indicator
    if (h %% 24 == 0 || h == 1) {
      elapsed <- as.numeric(difftime(Sys.time(), predict_start, units = "secs"))
      pct_complete <- (h / n_test) * 100
      cat(sprintf("  Forecasting day %d/%d (%.1f%%) | Elapsed: %.1fs\n", 
                  ceiling(h/24), ceiling(n_test/24), pct_complete, elapsed))
    }
    
    current_idx <- train_size + h
    start_idx <- current_idx - lookback
    
    if (start_idx < 1) {
      forecasts[h] <- tail(train_y, 1)
      next
    }
    
    # Window of past observations (scaled)
    recursive_window <- all_x_scaled[start_idx:(current_idx - 1), , drop = FALSE]
    recursive_predictions <- numeric(horizon)
    
    # RECURSIVE MULTI-STEP FORECAST
    for (step in seq_len(horizon)) {
      
      # 1-step prediction
      X_pred <- array(recursive_window, dim = c(1, lookback, length(feature_cols)))
      pred_scaled <- predict(model, X_pred, verbose = 0)
      pred_value_scaled <- pred_scaled[1, 1]
      recursive_predictions[step] <- pred_value_scaled
      
      # Prepare next input window
      if (step < horizon) {
        new_window <- recursive_window[2:lookback, , drop = FALSE]
        next_idx <- current_idx - 1 + step
        
        if (next_idx < nrow(all_x_scaled)) {
          new_features <- all_x_scaled[next_idx + 1, , drop = FALSE]
        } else {
          # If we run past data end, replicate last row
          new_features <- recursive_window[lookback, , drop = FALSE]
        }
        
        # --- FEEDBACK CORRECTIONS HERE ---
        # 1. Feed back predicted consumption if target is in features
        if (target_in_features) {
          new_features[1, target_feature_idx] <- pred_value_scaled
        }
        
        # 2. Update lagged features
        if (!is.null(lag_24_idx) && step >= 24) {
          new_features[1, lag_24_idx] <- recursive_predictions[step - 23] # last 24h pred
        }
        
        if (!is.null(lag_168_idx) && step >= 168) {
          new_features[1, lag_168_idx] <- recursive_predictions[step - 167] # last week pred
        }
        
        # 3. Update rolling mean of last 24 predictions
        if (!is.null(rollmean_24_idx)) {
          recent_steps <- tail(recursive_predictions[seq_len(step)], 24)
          new_features[1, rollmean_24_idx] <- mean(recent_steps, na.rm = TRUE)
        }
        
        # Add updated row into the sliding window
        recursive_window <- rbind(new_window, new_features)
      }
    }
    
    # Keep only the 24th forecast
    forecast_24h_scaled <- recursive_predictions[horizon]
    forecasts[h] <- forecast_24h_scaled * (y_max - y_min + 1e-6) + y_min
  }
  
  
  predict_time <- as.numeric(difftime(Sys.time(), predict_start, units = "mins"))
  total_time <- as.numeric(difftime(Sys.time(), overall_start, units = "mins"))
  
  cat(sprintf("\nPrediction complete! Time: %.2f minutes\n", predict_time))
  cat(sprintf("Total runtime: %.2f minutes (Train: %.2f min + Predict: %.2f min)\n",
              total_time, train_time, predict_time))
  
  # Fill any remaining NAs
  n_missing <- sum(is.na(forecasts))
  if (n_missing > 0) {
    cat(sprintf("Warning: %d forecasts missing, filling with last value\n", n_missing))
    for (i in which(is.na(forecasts))) {
      if (i == 1) {
        forecasts[i] <- tail(train_y, 1)
      } else {
        forecasts[i] <- forecasts[i - 1]
      }
    }
  }
  
  # Clean up
  k_clear_session()
  
  return(list(
    forecasts = forecasts, 
    runtime = total_time,
    train_time = train_time,
    predict_time = predict_time,
    history = history
  ))
}

# --- RUN ROLLING FORECASTS FOR BOTH PERIODS ---

run_lstm_rolling <- function(train, test, period_label) {
  cat(sprintf("\n==================== %s ====================\n", period_label))
  
  # Run LSTM recursive forecast
  lstm_result <- rolling_lstm_24h_recursive(
    train_data = train,
    test_data = test,
    feature_cols = feature_columns,
    lookback = LOOKBACK,
    units1 = UNITS1,
    units2 = UNITS2,
    dropout = DROPOUT,
    lr = LR,
    batch_size = BATCH,
    epochs = EPOCHS,
    horizon = HORIZON
  )
  
  # Evaluate
  actual <- test[[target_col]]
  
  eval_lstm <- evaluate_forecast(actual, lstm_result$forecasts, "LSTM_24h_Recursive")
  eval_lstm[, Runtime_min := lstm_result$runtime]
  eval_lstm[, Train_min := lstm_result$train_time]
  eval_lstm[, Predict_min := lstm_result$predict_time]
  eval_lstm[, Period := period_label]
  
  cat("\n--- Evaluation Results (24h-ahead RECURSIVE) ---\n")
  print(eval_lstm)
  
  # Create forecast dataframe
  forecast_dt <- data.table(
    Time = seq_along(actual),
    Actual = actual,
    LSTM_24h_Recursive = lstm_result$forecasts
  )
  
  # Plot results
  p <- ggplot(forecast_dt, aes(x = Time)) +
    geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
    geom_line(aes(y = LSTM_24h_Recursive, color = "LSTM (24h Recursive)"), 
              linewidth = 0.7, alpha = 0.8) +
    scale_color_manual(values = c("Actual" = "black", "LSTM (24h Recursive)" = "blue")) +
    labs(
      title = paste("LSTM 24h-Ahead Recursive Forecast -", period_label),
      subtitle = "Recursive 24-hour ahead predictions (feeding predictions back)",
      x = "Time (hours)", 
      y = "Energy Consumption (kWh)",
      color = "Series"
    ) +
    theme_minimal(base_size = 12)
  
  print(p)
  
  # Plot training history
  history_df <- data.frame(
    epoch = seq_len(length(lstm_result$history$metrics$loss)),
    train_loss = lstm_result$history$metrics$loss,
    val_loss = lstm_result$history$metrics$val_loss
  )
  
  p_history <- ggplot(history_df, aes(x = epoch)) +
    geom_line(aes(y = train_loss, color = "Training"), linewidth = 1) +
    geom_line(aes(y = val_loss, color = "Validation"), linewidth = 1) +
    scale_color_manual(values = c("Training" = "blue", "Validation" = "red")) +
    labs(
      title = paste("LSTM Training History -", period_label),
      x = "Epoch",
      y = "Loss (MSE)",
      color = "Dataset"
    ) +
    theme_minimal(base_size = 12)
  
  print(p_history)
  
  return(list(
    eval = eval_lstm,
    forecasts = forecast_dt,
    plot = p,
    history_plot = p_history,
    history = lstm_result$history
  ))
}

# --- PREPARE DATA SPLITS ---

cat("\n========== PREPARING DATA ==========\n")

trainA <- model_data[as.Date(interval) <= trainA_end]
testA  <- model_data[as.Date(interval) >= testA_start & as.Date(interval) <= testA_end]

trainB <- model_data[as.Date(interval) <= trainB_end]
testB  <- model_data[as.Date(interval) >= testB_start & as.Date(interval) <= testB_end]

cat(sprintf("Period A - Train: %d hours | Test: %d hours\n", nrow(trainA), nrow(testA)))
cat(sprintf("Period B - Train: %d hours | Test: %d hours\n", nrow(trainB), nrow(testB)))

# --- MAIN EXECUTION ---

cat("\n========== STARTING LSTM 24-HOUR AHEAD RECURSIVE FORECASTS ==========\n")

resultsA_lstm <- run_lstm_rolling(trainA, testA, "Period A (Stable)")
resultsB_lstm <- run_lstm_rolling(trainB, testB, "Period B (Not Stable)")

# --- FINAL SUMMARY ---

cat("\n==================== FINAL SUMMARY ====================\n")

all_eval_lstm <- rbind(resultsA_lstm$eval, resultsB_lstm$eval)
print(all_eval_lstm)

cat("\n--- Performance by Period (24h-ahead RECURSIVE) ---\n")
cat(sprintf("Period A: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f\n",
            resultsA_lstm$eval$RMSE,
            resultsA_lstm$eval$MAE,
            resultsA_lstm$eval$MAPE,
            resultsA_lstm$eval$R2))
cat(sprintf("  Runtime: %.2f min (Train: %.2f min + Predict: %.2f min)\n",
            resultsA_lstm$eval$Runtime_min,
            resultsA_lstm$eval$Train_min,
            resultsA_lstm$eval$Predict_min))

cat(sprintf("\nPeriod B: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f\n",
            resultsB_lstm$eval$RMSE,
            resultsB_lstm$eval$MAE,
            resultsB_lstm$eval$MAPE,
            resultsB_lstm$eval$R2))
cat(sprintf("  Runtime: %.2f min (Train: %.2f min + Predict: %.2f min)\n",
            resultsB_lstm$eval$Runtime_min,
            resultsB_lstm$eval$Train_min,
            resultsB_lstm$eval$Predict_min))

# --- SAVE RESULTS ---

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
results_file <- sprintf("LSTM_24h_Recursive_Forecast_%s.rds", timestamp)

lstm_rolling_results <- list(
  period_A = resultsA_lstm,
  period_B = resultsB_lstm,
  evaluations = all_eval_lstm,
  parameters = list(
    lookback = LOOKBACK,
    units1 = UNITS1,
    units2 = UNITS2,
    dropout = DROPOUT,
    lr = LR,
    batch_size = BATCH,
    epochs = EPOCHS,
    horizon = HORIZON,
    forecast_method = "recursive"
  )
)

saveRDS(lstm_rolling_results, file = results_file)
cat(sprintf("\nResults saved to: %s\n", results_file))

cat("\nLSTM 24h-ahead recursive forecast pipeline complete!\n")
