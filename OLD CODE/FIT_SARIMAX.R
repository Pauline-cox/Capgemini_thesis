# ---------------------------------------------------------------
# SARIMAX MODEL FIT — TRAIN DATA ONLY (NO FORECASTING)
# ---------------------------------------------------------------

set.seed(1234)

# --- SETTINGS ---
ORDER    <- c(2, 0, 2)
SEASONAL <- c(0, 1, 1)
PERIOD   <- 168L
target_col <- "total_consumption_kWh"

selected_xreg <- c(
  "co2", "total_occupancy", "lux",
  "business_hours", "hour_sin", "hour_cos", "dow_cos",
  "holiday", "dst"
)

# --- DATA PREP ---
tz_ref <- attr(model_data$interval, "tzone")
train_val_data <- model_data[
  interval >= as.POSIXct("2023-07-01 00:00:00", tz = tz_ref) &
    interval <  as.POSIXct("2024-10-01 00:00:00", tz = tz_ref)
]

y_train <- train_val_data[[target_col]]
x_train <- as.matrix(train_val_data[, ..selected_xreg])
x_train_scaled <- scale(x_train)
y_train_ts <- ts(y_train, frequency = PERIOD)

# --- MODEL FITTING ---
cat("\n--- Step 1: CSS → ML (Warm Start) Estimation ---\n")
train_start <- Sys.time()

sarimax_fit <-Arima(
    y_train_ts,
    order = ORDER,
    seasonal = list(order = SEASONAL, period = PERIOD),
    xreg = x_train_scaled,
    method = "CSS-ML"  
  )

train_time <- as.numeric(difftime(Sys.time(), train_start, units = "secs"))
cat(sprintf("\nModel training completed in %.2f seconds\n", train_time))

cat(sprintf("AIC: %.2f | BIC: %.2f | LogLik: %.2f\n",
            sarimax_fit$aic, sarimax_fit$bic, sarimax_fit$loglik))
print(summary(sarimax_fit))

# --- RESIDUAL DIAGNOSTICS ---
residuals_fit <- residuals(sarimax_fit)

cat("\n--- Residual Diagnostics ---\n")
lb_test <- Box.test(residuals_fit, lag = 24, type = "Ljung-Box",
                    fitdf = length(ORDER) + length(SEASONAL))
print(lb_test)

par(mfrow = c(2, 2))
plot(residuals_fit, main = "Residuals", col = "steelblue")
acf(residuals_fit, main = "ACF of Residuals")
pacf(residuals_fit, main = "PACF of Residuals")
qqnorm(residuals_fit); qqline(residuals_fit, col = "red")
par(mfrow = c(1, 1))

if (lb_test$p.value < 0.05) {
  cat("\nResiduals are likely *not* white noise (p < 0.05).\n")
} else {
  cat("\nResiduals appear to be white noise (p ≥ 0.05).\n")
}

