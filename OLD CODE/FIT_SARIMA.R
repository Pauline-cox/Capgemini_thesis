# ===============================================================
# PURE SARIMA MODEL FIT — TRAIN DATA ONLY (NO FORECASTING)
# ===============================================================

set.seed(1234)

# --- SETTINGS ---
ORDER    <- c(2, 0, 2)
SEASONAL <- c(0, 1, 1)
PERIOD   <- 168L
target_col <- "total_consumption_kWh"

# --- DATA PREP ---
tz_ref <- attr(model_data$interval, "tzone")
train_val_data <- model_data[
  interval >= as.POSIXct("2023-07-01 00:00:00", tz = tz_ref) &
    interval <  as.POSIXct("2024-10-01 00:00:00", tz = tz_ref)
]
y_train_ts <- ts(train_val_data$total_consumption_kWh, frequency = PERIOD)

# --- MODEL FITTING ---
cat("\n--- Step 1: CSS → ML (Warm Start) Estimation ---\n")
train_start <- Sys.time()

sarima_fit <- Arima(
    y_train_ts,
    order = ORDER,
    seasonal = list(order = SEASONAL, period = PERIOD),
    method = "CSS-ML"
  )

train_time <- as.numeric(difftime(Sys.time(), train_start, units = "secs"))
cat(sprintf("\nModel training completed in %.2f seconds\n", train_time))

cat(sprintf("AIC: %.2f | BIC: %.2f | LogLik: %.2f\n",
            sarima_fit$aic, sarima_fit$bic, sarima_fit$loglik))
print(summary(sarima_fit))


# --- RESIDUAL DIAGNOSTICS — LJUNG-BOX TEST ---
residuals_fit <- residuals(sarima_fit)

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


# RESULTS:
#   sarima_fit 
# Series: y_train_ts 
# ARIMA(2,0,2)(0,1,1)[168] 
# 
# Coefficients:
#   ar1      ar2      ma1     ma2     sma1
# 1.2144  -0.2938  -0.3626  0.0661  -0.7349
# s.e.  0.2381   0.2135   0.2376  0.0129   0.0079
# 
# sigma^2 = 745.9:  log likelihood = -48829
# AIC=97670   AICc=97670.01   BIC=97713.45
