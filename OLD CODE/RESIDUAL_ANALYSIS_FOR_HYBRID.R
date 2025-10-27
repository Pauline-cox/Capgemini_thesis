# ===============================================================
# RESIDUAL DIAGNOSTICS — SARIMAX MODEL
# ===============================================================

source("SARIMAX_MODEL_FIT")

# --- Extract residuals ---
resid_ts <- residuals(sarimax_fit)

# --- Basic summary ---
cat("\n--- Residual Summary ---\n")
print(summary(resid_ts))

# --- Ljung-Box test (autocorrelation check) ---
cat("\n--- Ljung-Box Test ---\n")
print(Box.test(resid_ts, lag = 24, type = "Ljung-Box"))

# --- Plot residuals as a time series ---
par(mfrow = c(3, 1))

plot(
  resid_ts, type = "l", col = "#2E86AB", lwd = 1.2,
  main = "SARIMAX Residuals (Time Series)",
  ylab = "Residual", xlab = "Time"
)
abline(h = 0, col = "red", lty = 2)

# --- ACF and PACF of residuals ---
acf(resid_ts, main = "ACF of Residuals", col = "#2E86AB", lwd = 2)
pacf(resid_ts, main = "PACF of Residuals", col = "#2E86AB", lwd = 2)

par(mfrow = c(1, 1))

# --- Normality check ---
tseries::jarque.bera.test(resid_ts)

# --- Histogram + QQ plot ---
par(mfrow = c(1, 2))
hist(resid_ts, breaks = 40, col = "#0072B2", main = "Residuals Histogram", xlab = "Residuals")
qqnorm(resid_ts, main = "Q-Q Plot of Residuals")
qqline(resid_ts, col = "red", lwd = 2)
par(mfrow = c(1, 1))

# ---------------------------------------------------------------
# 1) RESIDUAL TIME-SERIES OBJECT
# ---------------------------------------------------------------
resid_ts <- ts(residuals(sarimax_fit), frequency = 168)  # 168 = 24×7 for hourly data
plot(resid_ts, col="#2E86AB", main="In-sample SARIMAX Residuals", ylab="Residual (kWh)")
abline(h=0, col="red", lty=2)
# ---------------------------------------------------------------
# 2) STL DECOMPOSITION
# ---------------------------------------------------------------
resid_decomp <- stl(resid_ts, s.window = "periodic", robust = TRUE)
plot(resid_decomp, main = "STL Decomposition of SARIMAX Residuals")

comp <- resid_decomp$time.series
var_total <- var(resid_ts)
var_season <- var(comp[,"seasonal"])
var_trend  <- var(comp[,"trend"])
cat(sprintf("Seasonal = %.1f%%, Trend = %.1f%%, Noise = %.1f%%\n",
            100*var_season/var_total,
            100*var_trend/var_total,
            100*(1 - (var_season+var_trend)/var_total)))
spec.pgram(resid_ts, log="no", col="#2E86AB",
           main="Spectral Density of SARIMAX Residuals",
           xlab="Frequency (cycles/hour)", ylab="Spectral Power")

