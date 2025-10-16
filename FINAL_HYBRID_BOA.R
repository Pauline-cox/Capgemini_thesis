# ===============================================================
# HYBRID MODEL BAYESIAN OPTIMIZATION (SARIMAX + LSTM Residuals)
# ===============================================================

set.seed(1234)

# --- Settings ---
LOOKBACK <- 168L
HORIZON <- 24L
ORDER <- c(2, 0, 2)
SEASONAL <- c(0, 1, 1)
PERIOD <- 168L
target_col <- "total_consumption_kWh"

xreg_vars <- c("co2", "business_hours", "total_occupancy",
               "lux", "hour_cos", "hour_sin", "holiday", "dst")

feature_columns <- c(
  "total_occupancy", "humidity", "co2", "sound", "lux",
  "wind_speed", "sunshine_minutes", "global_radiation", "humidity_percent",
  "weekend", "business_hours", "hour_sin", "hour_cos",
  "dow_sin", "dow_cos", "month_cos",
  "lag_24", "lag_168", "rollmean_24",
  "holiday", "dst"
)

# --- Split Data ---

# --- DATA SPLIT ---

tz_ref <- attr(model_data$interval, "tzone")

train_data <- model_data[
  interval >= as.POSIXct("2023-07-01 00:00:00", tz = tz_ref) &
    interval <  as.POSIXct("2024-07-01 00:00:00", tz = tz_ref)
]

val_data <- model_data[
  interval >= as.POSIXct("2024-07-01 00:00:00", tz = tz_ref) &
    interval <  as.POSIXct("2024-09-30 00:00:00", tz = tz_ref)
]

cat(sprintf("Training samples: %d\n", nrow(train_data)))
cat(sprintf("Validation samples: %d\n", nrow(val_data)))

train_data[, target := total_consumption_kWh]
val_data[, target := total_consumption_kWh]

cat(sprintf("Training samples: %d | Validation samples: %d\n",
            nrow(train_data), nrow(val_data)))

# --- Train base SARIMAX model ---
cat("Training fixed SARIMAX base model...\n")
y_train <- train_data[[target_col]]
x_train <- as.matrix(scale(train_data[, ..xreg_vars]))
sarimax_base <- Arima(ts(y_train, frequency = PERIOD),
                      order = ORDER,
                      seasonal = list(order = SEASONAL, period = PERIOD),
                      xreg = x_train,
                      method = "CSS")

cat(sprintf("SARIMAX AIC=%.2f | Coefficients=%d\n",
            sarimax_base$aic, length(coef(sarimax_base))))

# --- Compute residuals for training and validation ---
base_fc_train <- fitted(sarimax_base)
residuals_train <- y_train - base_fc_train

x_val <- as.matrix(scale(val_data[, ..xreg_vars],
                         center = attr(x_train, "scaled:center"),
                         scale = attr(x_train, "scaled:scale")))
fc_val <- forecast(sarimax_base, xreg = x_val, h = nrow(val_data))
residuals_val <- val_data[[target_col]] - fc_val$mean

train_resid <- copy(train_data)
val_resid <- copy(val_data)
train_resid$residual <- residuals_train
val_resid$residual <- residuals_val

# --- Prepare scaled features for LSTM residual model ---
rec <- recipe(residual ~ ., data = rbind(train_resid, val_resid)[, c("residual", feature_columns), with = FALSE]) %>%
  step_range(all_numeric(), -all_outcomes()) %>%
  prep(training = train_resid)

Xtr <- bake(rec, train_resid)[, feature_columns, with = FALSE]
Xva <- bake(rec, val_resid)[, feature_columns, with = FALSE]
ytr <- train_resid$residual
yva <- val_resid$residual

y_min <- min(ytr)
y_max <- max(ytr)
scale_y <- function(y) (y - y_min) / (y_max - y_min + 1e-6)
ytr_scaled <- scale_y(ytr)
yva_scaled <- scale_y(yva)

# --- Sequence builder ---
make_train_sequences <- function(X, y, lookback, horizon) {
  n <- nrow(X) - lookback - horizon + 1
  Xarr <- array(NA_real_, dim = c(n, lookback, ncol(X)))
  Yarr <- array(NA_real_, dim = c(n, 1))
  for (i in seq_len(n)) {
    Xarr[i, , ] <- as.matrix(X[i:(i + lookback - 1), ])
    Yarr[i, ] <- y[i + lookback + horizon - 1]
  }
  list(X = Xarr, y = Yarr)
}

make_val_sequences <- function(X_train, X_val, y_train, y_val, lookback, horizon) {
  X_full <- rbind(X_train, X_val)
  y_full <- c(y_train, y_val)
  start_idx <- nrow(X_train) + 1
  n <- nrow(X_val) - lookback - horizon + 1
  Xarr <- array(NA_real_, dim = c(n, lookback, ncol(X_full)))
  Yarr <- array(NA_real_, dim = c(n, 1))
  for (i in seq_len(n)) {
    idx <- start_idx + i - 1
    Xarr[i, , ] <- as.matrix(X_full[(idx - lookback):(idx - 1), ])
    Yarr[i, ] <- y_full[idx + horizon - 1]
  }
  list(X = Xarr, y = Yarr)
}

# --- LSTM builder ---
build_lstm_model <- function(input_shape, units1, units2, dropout, lr) {
  keras_model_sequential() %>%
    layer_lstm(units = as.integer(units1), input_shape = input_shape, return_sequences = TRUE) %>%
    layer_dropout(rate = dropout) %>%
    layer_lstm(units = as.integer(units2), return_sequences = FALSE) %>%
    layer_dense(units = 1, activation = "linear") %>%
    compile(optimizer = optimizer_adam(learning_rate = lr), loss = "mse", metrics = "mae")
}

# --- Bayesian Optimization ---
cat("STARTING HYBRID (SARIMAX + LSTM residual) BAYESIAN OPTIMIZATION\n")

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

bo_objective <- function(units1, units2, dropout, lr, batch_size) {
  trial_counter <<- trial_counter + 1
  tensorflow::set_random_seed(1234 + trial_counter)
  trial_start <- Sys.time()
  
  units1 <- as.integer(round(units1))
  units2 <- as.integer(round(units2))
  batch_sz <- as.integer(round(batch_size))
  
  cat(sprintf("\n>>> Trial %d | units1=%d | units2=%d | dropout=%.3f | lr=%.5f | batch=%d\n",
              trial_counter, units1, units2, dropout, lr, batch_sz))
  flush.console()
  
  train_seq <- make_train_sequences(Xtr, ytr_scaled, LOOKBACK, HORIZON)
  val_seq <- make_val_sequences(Xtr, Xva, ytr_scaled, yva_scaled, LOOKBACK, HORIZON)
  
  model <- build_lstm_model(c(LOOKBACK, ncol(Xtr)), units1, units2, dropout, lr)
  history <- model %>% fit(
    train_seq$X, train_seq$y,
    epochs = 50,
    batch_size = batch_sz,
    validation_data = list(val_seq$X, val_seq$y),
    verbose = 0,
    callbacks = list(callback_early_stopping(patience = 5, restore_best_weights = TRUE))
  )
  
  val_mae <- min(history$metrics$val_mae)
  runtime_sec <- as.numeric(difftime(Sys.time(), trial_start, units = "secs"))
  cat(sprintf("Val MAE: %.4f | Runtime: %.1fs\n", val_mae, runtime_sec))
  
  all_trial_results <<- rbind(all_trial_results,
                              data.table(Trial = trial_counter, units1 = units1,
                                         units2 = units2, dropout = dropout, lr = lr,
                                         batch_size = batch_sz, val_mae = val_mae,
                                         runtime_sec = runtime_sec))
  
  k_clear_session(); gc()
  list(Score = -as.numeric(val_mae))
}

bounds <- list(
  units1     = c(32L, 128L),
  units2     = c(16L, 64L),
  dropout    = c(0.05, 0.4),
  lr         = c(1e-4, 5e-3),
  batch_size = c(16L, 64L)
)

cat("\nSearch space:\n"); print(bounds); cat("\n")

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

# --- Results ---
setorder(all_trial_results, val_mae)
best_trial <- all_trial_results[which.min(val_mae)]

cat("\n--- HYBRID OPTIMIZATION RESULTS ---\n")
print(best_trial)
cat(sprintf("Total runtime: %.2f min | Best MAE: %.4f\n", optimization_runtime, best_trial$val_mae))

timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Results_Hybrid_LSTM_Residual_BOA_%s.rds", timestamp)
saveRDS(list(trials = all_trial_results, best_trial = best_trial,
             runtime_min = optimization_runtime, opt_results = opt_results),
        file = save_name)
cat(sprintf("\nResults saved to: %s\n", save_name))
cat("Hybrid (SARIMAX + LSTM residual) Bayesian optimization complete!\n")
