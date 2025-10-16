# ===============================================================
# FULL FORECAST PIPELINE — LSTM & HYBRID SARIMA+LSTM
# ===============================================================

# Assumes libraries are already loaded: keras, tensorflow, data.table, recipes, ggplot2, forecast, lubridate

# ===============================================================
# 0) SETTINGS
# ===============================================================
set.seed(42)
LOOKBACK <- 168L
UNITS1   <- 64L
UNITS2   <- 32L
DROPOUT  <- 0.3
LR       <- 0.005
BATCH    <- 16L
EPOCHS   <- 50L
target_col <- "total_consumption_kWh"

ORDER    <- c(2,0,2)
SEASONAL <- c(0,1,1)
PERIOD   <- 168L

# --- Features ---
env_vars <- c("tempC","humidity","co2","sound","lux","temperature",
              "wind_speed","sunshine_minutes","global_radiation",
              "humidity_percent","fog","rain","snow","thunder","ice")

temporal_vars <- c("hour","weekday","month","weekend","business_hours",
                   "hour_sin","hour_cos","dow_sin","dow_cos","holiday","dst")

# Use all engineered + exogenous + lags
feature_columns <- c("total_occupancy","tempC","humidity","co2","sound","lux",
                     "temperature","wind_speed","sunshine_minutes","global_radiation",
                     "humidity_percent","fog","rain","snow","thunder","ice","hour",
                     "weekday","month","weekend","business_hours","hour_sin","hour_cos",
                     "dow_sin","dow_cos",
                     "lag_24","lag_72","lag_168","lag_336","lag_504",
                     "rollmean_24","rollmean_168",
                     "holiday","dst")

feature_columns <- c("total_occupancy","tempC","humidity","co2","sound","lux",
                     "temperature","wind_speed","sunshine_minutes","global_radiation",
                     "humidity_percent",
                     "weekday","weekend","business_hours","hour_sin","hour_cos",
                     "dow_sin","dow_cos", "month_cos",
                     "holiday","dst")


# Date ranges
testA_start <- as.Date("2024-10-01")
testA_end   <- as.Date("2024-10-14")
trainA_end  <- testA_start - 1

testB_start <- as.Date("2024-12-18")
testB_end   <- as.Date("2024-12-31")
trainB_end  <- testB_start - 1

# ===============================================================
# 1) HELPER FUNCTIONS
# ===============================================================

# --- Scaling ---
scale_features <- function(train, test, feature_cols, target_col) {
  rec <- recipe(as.formula(paste(target_col, "~ .")),
                data = train[, c(target_col, feature_cols), with = FALSE]) %>%
    step_string2factor(all_nominal(), -all_outcomes()) %>%
    step_dummy(all_nominal(), one_hot = TRUE) %>%
    step_range(all_numeric(), -all_outcomes()) %>%
    prep(training = train, retain = TRUE)
  list(
    X_train = bake(rec, new_data = train)[, setdiff(names(bake(rec, new_data = train)),
                                                    target_col), with = FALSE],
    X_test  = bake(rec, new_data = test)[, setdiff(names(bake(rec, new_data = test)),
                                                   target_col), with = FALSE],
    rec = rec
  )
}

# --- Evaluation ---
evaluate_models <- function(actual, preds) {
  res <- lapply(names(preds), function(name) {
    p <- preds[[name]]
    rmse <- sqrt(mean((actual - p)^2, na.rm = TRUE))
    mae  <- mean(abs(actual - p), na.rm = TRUE)
    r2   <- 1 - sum((actual - p)^2) / sum((actual - mean(actual))^2)
    mape <- mean(abs((actual - p)/pmax(actual,1e-6)), na.rm=TRUE)*100
    data.table(Model=name, RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)
  })
  rbindlist(res)[order(RMSE)]
}

# --- Plot Faceted ---
plot_forecasts_facet <- function(actual, preds, title) {
  df_list <- lapply(names(preds), \(n) data.table(Time=seq_along(actual),
                                                  Forecast=preds[[n]], Model=n))
  df <- rbindlist(df_list)
  df_actual <- data.table(Time=seq_along(actual), Actual=actual)
  df_melt <- merge(df, df_actual, by="Time")
  ggplot(df_melt, aes(x=Time)) +
    geom_line(aes(y=Actual), color="black", linewidth=1) +
    geom_line(aes(y=Forecast, color=Model), linewidth=0.8) +
    facet_wrap(~Model, ncol=1, scales="free_y") +
    labs(title=title, x="Time (hours)", y="Energy (kWh)") +
    theme_minimal(base_size=12) +
    theme(legend.position="none", strip.text=element_text(face="bold"))
}

# --- Build sequences ---
build_sequences <- function(X, y_scaled, lookback) {
  n_seq <- nrow(X) - lookback
  X_arr <- array(NA_real_, dim=c(n_seq, lookback, ncol(X)))
  Y_arr <- array(NA_real_, dim=c(n_seq,1))
  for (i in 1:n_seq) {
    X_arr[i,,] <- as.matrix(X[i:(i+lookback-1),])
    Y_arr[i,]  <- y_scaled[i+lookback]
  }
  list(X=X_arr, Y=Y_arr)
}

# --- LSTM model ---
build_lstm_model <- function(input_dim, lookback) {
  keras_model_sequential() %>%
    layer_lstm(units=UNITS1, input_shape=c(lookback, input_dim),
               return_sequences=TRUE) %>%
    layer_dropout(rate=DROPOUT) %>%
    layer_lstm(units=UNITS2, return_sequences=FALSE) %>%
    layer_dense(units=1, activation="linear") %>%
    compile(optimizer=optimizer_nadam(learning_rate=LR),
            loss="mse", metrics="mae")
}

# --- SARIMA helper for hybrid ---
sarima_forecast <- function(train, test) {
  y_train <- ts(train[[target_col]], frequency=PERIOD)
  fit <- Arima(y_train, order=ORDER,
               seasonal=list(order=SEASONAL, period=PERIOD), method="CSS")
  fc <- forecast(fit, h=nrow(test))
  list(fit=fit, forecast=fc$mean)
}

# ===============================================================
# 2) CORE MODEL FUNCTIONS
# ===============================================================

# --- Recursive LSTM Forecast ---
run_lstm_recursive <- function(train, test, features, label) {
  cat(sprintf("\n>>> TRAINING %s\n", label))
  
  scaled <- scale_features(train, test, features, target_col)
  X_train <- scaled$X_train; X_test <- scaled$X_test
  y_train <- train[[target_col]]; y_test <- test[[target_col]]
  
  y_min <- min(y_train); y_max <- max(y_train)
  scale_y <- \(y) (y - y_min)/(y_max - y_min + 1e-6)
  invert_y <- \(y) y*(y_max - y_min) + y_min
  y_train_scaled <- scale_y(y_train)
  
  seqs <- build_sequences(X_train, y_train_scaled, LOOKBACK)
  model <- build_lstm_model(ncol(X_train), LOOKBACK)
  model %>% fit(
    seqs$X, seqs$Y, epochs=EPOCHS, batch_size=BATCH,
    validation_split=0.1, verbose=0,
    callbacks=list(
      callback_early_stopping(patience=10, restore_best_weights=TRUE),
      callback_reduce_lr_on_plateau(factor=0.5, patience=5)
    )
  )
  
  cat("Recursive forecasting...\n")
  X_all <- rbind(X_train, X_test)
  n_hist <- nrow(X_train); n_test <- nrow(X_test)
  preds <- numeric(0)
  current <- as.matrix(X_all[(n_hist-LOOKBACK+1):n_hist, , drop=FALSE])
  pb <- txtProgressBar(min=0, max=n_test, style=3)
  for (i in 1:n_test) {
    xin <- array(current, dim=c(1,LOOKBACK,ncol(X_all)))
    pred_scaled <- model %>% predict(xin, verbose=0)
    pred_val <- invert_y(as.numeric(pred_scaled))
    preds <- c(preds, pred_val)
    if (i < n_test) current <- rbind(current[-1,], as.numeric(X_test[i,]))
    setTxtProgressBar(pb,i)
  }
  close(pb)
  
  list(pred=preds, actual=y_test, model=model, label=label)
}

# --- Hybrid SARIMA + LSTM ---
run_hybrid_sarima_lstm <- function(train, test, features) {
  cat("\n>>> TRAINING HYBRID SARIMA + LSTM\n")
  sarima_fit <- sarima_forecast(train, test)
  sarima_pred_train <- fitted(sarima_fit$fit)
  resid_train <- as.numeric(train[[target_col]] - sarima_pred_train)
  
  train_resid <- copy(train)
  train_resid[, residual := resid_train]
  
  # Build lagged residual features
  for (lag in c(24,72,168)) {
    train_resid[, paste0("resid_lag_",lag) := shift(residual, n=lag, fill=0)]
  }
  test_resid <- copy(test)
  test_resid[, residual := 0]
  for (lag in c(24,72,168)) test_resid[, paste0("resid_lag_",lag) := 0]
  
  # Train LSTM on residuals
  feat_lstm <- c(features, paste0("resid_lag_",c(24,72,168)))
  scaled <- scale_features(train_resid, test_resid, feat_lstm, "residual")
  X_train <- scaled$X_train; X_test <- scaled$X_test
  y_train <- train_resid$residual; y_test <- test[[target_col]]
  
  y_min <- min(y_train); y_max <- max(y_train)
  scale_y <- \(y) (y - y_min)/(y_max - y_min + 1e-6)
  invert_y <- \(y) y*(y_max - y_min) + y_min
  y_train_scaled <- scale_y(y_train)
  
  seqs <- build_sequences(X_train, y_train_scaled, LOOKBACK)
  model <- build_lstm_model(ncol(X_train), LOOKBACK)
  model %>% fit(
    seqs$X, seqs$Y, epochs=EPOCHS, batch_size=BATCH,
    validation_split=0.1, verbose=0,
    callbacks=list(
      callback_early_stopping(patience=10, restore_best_weights=TRUE),
      callback_reduce_lr_on_plateau(factor=0.5, patience=5)
    )
  )
  
  # Recursive forecast residuals
  cat("Recursive forecasting residuals...\n")
  X_all <- rbind(X_train, X_test)
  n_hist <- nrow(X_train); n_test <- nrow(X_test)
  preds_resid <- numeric(0)
  current <- as.matrix(X_all[(n_hist-LOOKBACK+1):n_hist, , drop=FALSE])
  pb <- txtProgressBar(min=0, max=n_test, style=3)
  for (i in 1:n_test) {
    xin <- array(current, dim=c(1,LOOKBACK,ncol(X_all)))
    pred_scaled <- model %>% predict(xin, verbose=0)
    pred_val <- invert_y(as.numeric(pred_scaled))
    preds_resid <- c(preds_resid, pred_val)
    if (i < n_test) current <- rbind(current[-1,], as.numeric(X_test[i,]))
    setTxtProgressBar(pb,i)
  }
  close(pb)
  
  final_pred <- as.numeric(sarima_fit$forecast) + preds_resid
  list(pred=final_pred, actual=y_test, model=model, label="Hybrid_SARIMA_LSTM")
}

# ===============================================================
# 3) RUN PIPELINE FOR A TEST PERIOD
# ===============================================================
run_all_models <- function(train, test, label_period) {
  cat(sprintf("\n==================== %s ====================\n", label_period))
  
  lstm_full <- run_lstm_recursive(train, test, feature_columns, "LSTM_Full")
  lstm_exog <- run_lstm_recursive(train, test, env_vars, "LSTM_ExogOnly")
  lstm_temp <- run_lstm_recursive(train, test, temporal_vars, "LSTM_Temporal")
  hybrid    <- run_hybrid_sarima_lstm(train, test, feature_columns)
  
  preds <- list(
    LSTM_Full = lstm_full$pred,
    LSTM_ExogOnly = lstm_exog$pred,
    LSTM_Temporal = lstm_temp$pred,
    Hybrid_SARIMA_LSTM = hybrid$pred
  )
  actual <- test[[target_col]]
  eval <- evaluate_models(actual, preds)
  print(eval)
  
  plot <- plot_forecasts_facet(actual, preds,
                               paste0("Forecast Comparison — ", label_period))
  print(plot)
  
  list(eval=eval)
}

# ===============================================================
# 4) RUN FOR BOTH PERIODS
# ===============================================================
trainA <- model_data[as.Date(interval) <= trainA_end]
testA  <- model_data[as.Date(interval) >= testA_start & as.Date(interval) <= testA_end]
trainB <- model_data[as.Date(interval) <= trainB_end]
testB  <- model_data[as.Date(interval) >= testB_start & as.Date(interval) <= testB_end]

resA <- run_all_models(trainA, testA, "Test Period A")
resB <- run_all_models(trainB, testB, "Test Period B")

# ===============================================================
# 5) SUMMARY
# ===============================================================
cat("\n==================== SUMMARY ====================\n")
cat(sprintf("Best Model Period A: %s (RMSE=%.2f)\n", resA$eval[1,Model], resA$eval[1,RMSE]))
cat(sprintf("Best Model Period B: %s (RMSE=%.2f)\n", resB$eval[1,Model], resB$eval[1,RMSE]))
cat("\nPipeline complete ✅\n")

