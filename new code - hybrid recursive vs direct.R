# ================================================================
# HYBRID SARIMA + LSTM (Train LSTM on training residuals)
# ================================================================
suppressPackageStartupMessages({
  library(data.table)
  library(forecast)
  library(keras)
  library(tensorflow)
  library(recipes)
  library(Metrics)
  library(ggplot2)
})

set.seed(42)

# ================================================================
# PARAMETERS
# ================================================================
ORDER    <- c(1,0,1)
SEASONAL <- c(1,1,1)
PERIOD   <- 168L        # weekly seasonality (hourly data)
LOOKBACK <- 168L
HORIZON  <- 24L
UNITS1   <- 64L
UNITS2   <- 32L
DROPOUT  <- 0.3
LR       <- 0.005
BATCH    <- 16L
EPOCHS   <- 100L
target_col <- "total_consumption_kWh"

# ================================================================
# TRAIN / TEST SPLIT
# ================================================================
test_start <- as.Date("2024-12-18")
test_end   <- as.Date("2024-12-31")
train_end  <- test_start - 1

train <- model_data[date <= train_end]
test  <- model_data[date >= test_start & date <= test_end]

cat(sprintf("\nTraining ≤ %s | Test %s–%s\n", train_end, test_start, test_end))

# ================================================================
# 1️⃣ FIT SARIMA ON TRAINING PERIOD
# ================================================================
y_train <- ts(train[[target_col]], frequency = PERIOD)
sarima_fit <- Arima(y_train,
                    order = ORDER,
                    seasonal = list(order = SEASONAL, period = PERIOD),
                    method = "CSS")
summary(sarima_fit)

# Fitted values + residuals (TRAIN only)
train[, SARIMA_fitted := fitted(sarima_fit)]
train[, SARIMA_resid := as.numeric(y_train - SARIMA_fitted)]

# Forecast baseline for TEST
sarima_forecast <- as.numeric(forecast(sarima_fit, h = nrow(test))$mean)
cat("\nSARIMA baseline forecast complete.\n")

# ================================================================
# 2️⃣ PREP LSTM DATA (TRAIN ON TRAINING RESIDUALS)
# ================================================================
# Add lagged residuals
train[, resid_lag1   := shift(SARIMA_resid, 1)]
train[, resid_lag24  := shift(SARIMA_resid, 24)]
train[, resid_lag168 := shift(SARIMA_resid, 168)]
train <- na.omit(train)

# Automatically include ALL engineered predictors
# Exclude interval/date/target and SARIMA columns
feature_columns <- setdiff(
  names(train),
  c("interval","date",target_col,"SARIMA_fitted","SARIMA_resid")
)
cat("\nIncluding", length(feature_columns), "features for LSTM residual model.\n")

# Scale data
rec <- recipe(SARIMA_resid ~ ., data = train[, c("SARIMA_resid", feature_columns), with=FALSE]) %>%
  step_range(all_numeric(), -all_outcomes()) %>%
  prep(training = train, retain = TRUE)

X_train <- bake(rec, new_data = train)[, feature_columns, with=FALSE]
y_train <- bake(rec, new_data = train)[["SARIMA_resid"]]

# ================================================================
# 3️⃣ BUILD TRAINING SEQUENCES
# ================================================================
n <- nrow(X_train) - LOOKBACK - HORIZON + 1
Xarr <- array(NA_real_, dim = c(n, LOOKBACK, ncol(X_train)))
Yarr <- array(NA_real_, dim = c(n, HORIZON))
for (i in 1:n) {
  Xarr[i,,] <- as.matrix(X_train[i:(i+LOOKBACK-1), ])
  Yarr[i, ] <- y_train[(i+LOOKBACK):(i+LOOKBACK+HORIZON-1)]
}

cat(sprintf("Built %d training sequences for LSTM.\n", n))

# ================================================================
# 4️⃣ TRAIN LSTM ON TRAIN RESIDUALS
# ================================================================
cat("\nTraining LSTM on training residuals...\n")

model_lstm <- keras_model_sequential() %>%
  layer_lstm(units=UNITS1, input_shape=c(LOOKBACK, ncol(X_train)), return_sequences=TRUE) %>%
  layer_dropout(rate=DROPOUT) %>%
  layer_lstm(units=UNITS2, return_sequences=FALSE) %>%
  layer_dense(units=HORIZON, activation="linear") %>%
  compile(optimizer=optimizer_nadam(learning_rate=LR),
          loss="mse", metrics="mae")

history <- model_lstm %>% fit(
  Xarr, Yarr,
  epochs=EPOCHS, batch_size=BATCH,
  validation_split=0.1, verbose=1,
  callbacks=list(
    callback_early_stopping(patience=12, restore_best_weights=TRUE),
    callback_reduce_lr_on_plateau(factor=0.5, patience=6)
  )
)

cat("LSTM residual model training complete.\n")

# ================================================================
# 5️⃣ PREP TEST INPUTS FOR RESIDUAL FORECASTING
# ================================================================
# Build residual lag features for TEST period
# (Extend training residuals into test)
resid_hist <- c(train$SARIMA_resid, rep(NA, nrow(test)))
test[, resid_lag1   := shift(resid_hist, 1)[(nrow(train)+1):(nrow(train)+nrow(test))]]
test[, resid_lag24  := shift(resid_hist, 24)[(nrow(train)+1):(nrow(train)+nrow(test))]]
test[, resid_lag168 := shift(resid_hist, 168)[(nrow(train)+1):(nrow(train)+nrow(test))]]

# Scale test inputs using same recipe
X_test <- bake(rec, new_data = test[, feature_columns, with=FALSE])

# ================================================================
# 6️⃣ FORECAST RESIDUALS WITH LSTM
# ================================================================
cat("\nForecasting residuals for test period...\n")

X_all <- rbind(X_train, X_test)
n_hist <- nrow(X_train)
n_test <- nrow(X_test)
pred_resid <- numeric(0)

for (i in seq(1, n_test - LOOKBACK + 1, by = HORIZON)) {
  end_idx <- min(n_hist + i - 1, n_hist + n_test - HORIZON)
  hist_idx <- (end_idx - LOOKBACK + 1):end_idx
  xin <- array(as.matrix(X_all[hist_idx, , drop=FALSE]),
               dim = c(1, LOOKBACK, ncol(X_all)))
  pred <- model_lstm %>% predict(xin, verbose=0)
  pred_resid <- c(pred_resid, as.numeric(pred))
}
pred_resid <- pred_resid[1:n_test]

cat("Residual forecasts complete.\n")

# ================================================================
# 7️⃣ COMBINE FORECASTS → HYBRID
# ================================================================
hybrid_forecast <- sarima_forecast + pred_resid
y_test <- test[[target_col]]

# ================================================================
# 8️⃣ EVALUATE
# ================================================================
safe_metrics <- function(actual, forecast) {
  RMSE <- sqrt(mean((actual - forecast)^2))
  MAE  <- mean(abs(actual - forecast))
  MAPE <- mean(abs((actual - forecast) / pmax(actual, 1e-3))) * 100
  R2   <- 1 - sum((actual - forecast)^2) / sum((actual - mean(actual))^2)
  data.table(RMSE=RMSE, MAE=MAE, MAPE=MAPE, R2=R2)
}

metrics_sarima <- safe_metrics(y_test, sarima_forecast)
metrics_hybrid <- safe_metrics(y_test, hybrid_forecast)

cat("\n=== SARIMA PERFORMANCE ===\n"); print(metrics_sarima)
cat("\n=== HYBRID PERFORMANCE ===\n"); print(metrics_hybrid)

# ================================================================
# 9️⃣ PLOTS
# ================================================================
cat("\nPlotting...\n")

# --- Forecast comparison ---
plot_data1 <- rbind(
  data.table(interval=test$interval, actual=y_test,
             forecast=sarima_forecast, Model="SARIMA"),
  data.table(interval=test$interval, actual=y_test,
             forecast=hybrid_forecast, Model="Hybrid")
)
p1 <- ggplot(plot_data1) +
  geom_line(aes(x=interval, y=actual), colour="black", size=1) +
  geom_line(aes(x=interval, y=forecast, colour=Model), size=0.8) +
  labs(title="Hybrid SARIMA–LSTM Forecast vs Actual",
       y="Energy Consumption (kWh)", x=NULL, colour=NULL) +
  theme_minimal(base_size=13) +
  theme(legend.position="bottom")
print(p1)

# --- Residuals comparison ---
plot_data2 <- data.table(
  interval = test$interval,
  forecast_resid = pred_resid
)
plot_data2$actual_resid <- tail(train$SARIMA_resid, nrow(test))

p2 <- ggplot(plot_data2) +
  geom_line(aes(x=interval, y=actual_resid), colour="black", size=1) +
  geom_line(aes(x=interval, y=forecast_resid), colour="red", size=0.8) +
  labs(title="Residuals: Actual vs LSTM-Predicted",
       y="Residual (kWh)", x=NULL) +
  theme_minimal(base_size=13)
print(p2)

cat("\n>>> HYBRID SARIMA–LSTM COMPLETE ✅\n")
