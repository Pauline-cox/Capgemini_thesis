# ===============================================================
# LSTM BAYESIAN OPTIMIZATION (FIXED LOOKBACK=168, OPTIMIZER=ADAM)
# ===============================================================

set.seed(1234)

# --- SETTINGS --- 

HORIZON <- 24  # 24-hour ahead forecast
LOOKBACK <- 168  # fixed: 7 days of hourly data
OPTIMIZER_FIXED <- "adam"

feature_columns <- c(
  "total_occupancy", "humidity", "co2", "sound", "lux",
  "wind_speed", "sunshine_minutes", "global_radiation", "humidity_percent",
  "weekend", "business_hours", "hour_sin", "hour_cos",
  "dow_sin", "dow_cos", "month_cos",
  "lag_24", "lag_168", "rollmean_24",
  "holiday", "dst"
)

# --- DATA SPLIT ---

train_data <- model_data[interval >= "2023-07-01" & interval <= "2024-06-30"]
val_data   <- model_data[interval >= "2024-07-01" & interval < "2024-09-30"]

cat(sprintf("Training samples: %d\n", nrow(train_data)))
cat(sprintf("Validation samples: %d\n", nrow(val_data)))

train_data[, target := total_consumption_kWh]
val_data[, target := total_consumption_kWh]

# --- SCALING ---

rec <- recipe(target ~ ., data = rbind(train_data, val_data)[, c("target", feature_columns), with = FALSE]) %>%
  step_range(all_numeric(), -all_outcomes()) %>%
  prep(training = train_data)

Xtr <- bake(rec, train_data)[, feature_columns, with = FALSE]
Xva <- bake(rec, val_data)[, feature_columns, with = FALSE]
ytr <- train_data$target
yva <- val_data$target

y_min <- min(ytr)
y_max <- max(ytr)
scale_y <- function(y) (y - y_min) / (y_max - y_min + 1e-6)
unscale_y <- function(y_scaled) y_scaled * (y_max - y_min + 1e-6) + y_min
ytr_scaled <- scale_y(ytr)
yva_scaled <- scale_y(yva)

cat(sprintf("Target range: [%.2f, %.2f]\n", y_min, y_max))

# --- SEQUENCE GENERATION ---

make_train_sequences <- function(X, y, lookback, horizon) {
  n <- nrow(X) - lookback - horizon + 1
  if (n <= 0) stop("Not enough data for training sequences.")
  
  Xarr <- array(NA_real_, dim = c(n, lookback, ncol(X)))
  Yarr <- array(NA_real_, dim = c(n, horizon))
  
  for (i in seq_len(n)) {
    Xarr[i, , ] <- as.matrix(X[i:(i + lookback - 1), ])
    Yarr[i, ] <- y[(i + lookback):(i + lookback + horizon - 1)]
  }
  list(X = Xarr, y = Yarr)
}

make_val_sequences <- function(X_train, X_val, y_train, y_val, lookback, horizon) {
  X_full <- rbind(X_train, X_val)
  y_full <- c(y_train, y_val)
  start_idx <- nrow(X_train) + 1
  end_idx <- nrow(X_full)
  n <- end_idx - start_idx - horizon + 1
  if (n <= 0) stop("Not enough validation data for sequences.")
  
  Xarr <- array(NA_real_, dim = c(n, lookback, ncol(X_full)))
  Yarr <- array(NA_real_, dim = c(n, horizon))
  
  for (i in seq_len(n)) {
    actual_idx <- start_idx + i - 1
    Xarr[i, , ] <- as.matrix(X_full[(actual_idx - lookback + 1):actual_idx, ])
    Yarr[i, ] <- y_full[(actual_idx + 1):(actual_idx + horizon)]
  }
  list(X = Xarr, y = Yarr)
}

# --- MODEL BUILDER ---

build_lstm_model <- function(input_shape, units1, units2, dropout, lr) {
  opt <- optimizer_adam(learning_rate = lr)
  keras_model_sequential() %>%
    layer_lstm(units = as.integer(units1), input_shape = input_shape, return_sequences = TRUE) %>%
    layer_dropout(rate = dropout) %>%
    layer_lstm(units = as.integer(units2), return_sequences = FALSE) %>%
    layer_dense(units = 1, activation = "linear") %>%
    compile(optimizer = opt, loss = "mse", metrics = "mae")
}

# --- BAYESIAN OPTIMIZATION SETUP ---

cat("STARTING LSTM BAYESIAN OPTIMIZATION (Fixed Lookback=168, Optimizer=Adam)\n")

trial_counter <- 0
all_trial_results <- data.table(
  Trial = integer(),
  units1 = integer(),
  units2 = integer(),
  dropout = numeric(),
  lr = numeric(),
  batch_size = integer(),
  val_mae = numeric(),
  runtime_sec = numeric()
)

# --- Objective Function ---

bo_objective <- function(units1, units2, dropout, lr, batch_size) {
  trial_counter <<- trial_counter + 1
  trial_start <- Sys.time()
  
  units1 <- as.integer(round(units1))
  units2 <- as.integer(round(units2))
  batch_sz <- as.integer(round(batch_size))
  
  cat(sprintf("\n>>> Trial %d\n", trial_counter))
  cat(sprintf("    units1=%d | units2=%d | dropout=%.3f | lr=%.5f | batch=%d\n", 
              units1, units2, dropout, lr, batch_sz))
  flush.console()
  
  train_seq <- make_train_sequences(Xtr, ytr_scaled, LOOKBACK, HORIZON)
  val_seq <- make_val_sequences(Xtr, Xva, ytr_scaled, yva_scaled, LOOKBACK, HORIZON)
  
  model <- build_lstm_model(
    input_shape = c(LOOKBACK, ncol(Xtr)),
    units1 = units1,
    units2 = units2,
    dropout = dropout,
    lr = lr
  )
  
  history <- model %>% fit(
    train_seq$X, train_seq$y,
    epochs = 50,
    batch_size = batch_sz,
    validation_data = list(val_seq$X, val_seq$y),
    verbose = 0,
    callbacks = list(
      callback_early_stopping(patience = 5, restore_best_weights = TRUE),
      callback_reduce_lr_on_plateau(factor = 0.5, patience = 3)
    )
  )
  
  val_mae <- tail(history$metrics$val_mae, 1)
  runtime_sec <- as.numeric(difftime(Sys.time(), trial_start, units = "secs"))
  
  cat(sprintf("    Val MAE: %.4f | Runtime: %.1fs\n", val_mae, runtime_sec))
  
  all_trial_results <<- rbind(
    all_trial_results,
    data.table(
      Trial = trial_counter,
      units1 = units1,
      units2 = units2,
      dropout = dropout,
      lr = lr,
      batch_size = batch_sz,
      val_mae = val_mae,
      runtime_sec = runtime_sec
    )
  )
  
  k_clear_session()
  gc()
  
  return(list(Score = -as.numeric(val_mae)))
}

# --- Search Space ---

bounds <- list(
  units1     = c(32L, 128L),
  units2     = c(16L, 64L),
  dropout    = c(0.05, 0.4),
  lr         = c(1e-4, 5e-3),
  batch_size = c(16L, 64L)
)

cat("\nSearch space:\n")
print(bounds)
cat("\n")

# --- Run Optimization ---

optimization_start <- Sys.time()

opt_results <- tryCatch({
  bayesOpt(
    FUN = bo_objective,
    bounds = bounds,
    initPoints = 10,
    iters.n = 20,
    acq = "ei",
    verbose = 1
  )
}, error = function(e) {
  cat("Optimization failed:", conditionMessage(e), "\n")
  return(NULL)
})

optimization_runtime <- as.numeric(difftime(Sys.time(), optimization_start, units = "mins"))

# --- Results Summary ---

cat("\nOPTIMIZATION COMPLETE\n")
cat(sprintf("Total runtime: %.2f minutes\n\n", optimization_runtime))
setorder(all_trial_results, val_mae)
print(all_trial_results)

best_params <- tryCatch(getBestPars(opt_results), error = function(e) NULL)
best_score <- tryCatch(max(opt_results$scoreSummary$Score), error = function(e) NA)
best_val_mae <- -best_score
best_trial <- all_trial_results[which.min(val_mae)]

cat("\n--- BEST PARAMETERS ---\n")
print(best_params)
cat(sprintf("Best Validation MAE: %.4f\n", best_val_mae))
cat("\n--- BEST TRIAL DETAILS ---\n")
print(best_trial)

# --- Visualization ---

p1 <- ggplot(all_trial_results, aes(x = Trial, y = val_mae)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "darkblue", size = 2) +
  geom_hline(yintercept = best_val_mae, linetype = "dashed", color = "red") +
  labs(title = "Validation MAE Over Trials", x = "Trial", y = "Validation MAE") +
  theme_minimal(base_size = 12)
print(p1)

param_importance <- data.table(
  Parameter = c("units1", "units2", "dropout", "lr", "batch_size"),
  Correlation = c(
    cor(all_trial_results$units1, all_trial_results$val_mae),
    cor(all_trial_results$units2, all_trial_results$val_mae),
    cor(all_trial_results$dropout, all_trial_results$val_mae),
    cor(all_trial_results$lr, all_trial_results$val_mae),
    cor(all_trial_results$batch_size, all_trial_results$val_mae)
  )
)

p2 <- ggplot(param_importance, aes(x = reorder(Parameter, abs(Correlation)), 
                                   y = Correlation, fill = Correlation > 0)) +
  geom_col() +
  coord_flip() +
  labs(title = "Hyperparameter Correlation with Val MAE",
       x = "Parameter", y = "Correlation") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "none")
print(p2)

# --- Final Output ---

cat("\nLSTM BAYESIAN OPTIMIZATION COMPLETE\n")

lstm_opt <- list(
  opt_results = opt_results,
  all_trials = all_trial_results,
  best_params = best_params,
  best_score = best_score,
  best_val_mae = best_val_mae,
  runtime_min = optimization_runtime
)

cat("\n--- SUMMARY ---\n")
cat(sprintf("Total trials: %d\n", nrow(all_trial_results)))
cat(sprintf("Best MAE: %.4f\n", best_val_mae))
cat(sprintf("Total runtime: %.2f minutes\n", optimization_runtime))
cat(sprintf("Avg time per trial: %.1f seconds\n", mean(all_trial_results$runtime_sec)))
cat("\nFixed parameters: LOOKBACK=168 | OPTIMIZER=Adam\n")
cat("\nOptimization complete! Use 'lstm_opt$best_params' for final model.\n")
