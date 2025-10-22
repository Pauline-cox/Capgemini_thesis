# ---------------------------------------------------------------
# CLUSTER-BASED SARIMAX MODEL FIT — TRAIN DATA ONLY (NO FORECASTING)
# ---------------------------------------------------------------

suppressPackageStartupMessages({
  library(forecast)
  library(data.table)
  library(tseries)
  library(stats)
})

set.seed(1234)

# --- SETTINGS ---
ORDER    <- c(2, 0, 2)
SEASONAL <- c(0, 1, 1)
PERIOD   <- 168L
N_CLUSTERS <- 3
target_col <- "total_consumption_kWh"

env_vars <- c(
  "tempC", "humidity", "co2", "sound", "lux", "total_occupancy",
  "temperature", "wind_speed", "sunshine_minutes",
  "global_radiation", "humidity_percent",
  "fog", "rain", "snow", "thunder", "ice"
)

temporal_vars <- c("business_hours", "hour_sin", "hour_cos", "dow_cos", "holiday", "dst")

# --- TRAINING DATA ---
tz_ref <- attr(model_data$interval, "tzone")
train_data <- model_data[
  interval < as.POSIXct("2024-10-01 00:00:00", tz = tz_ref)
]

# --- ENVIRONMENTAL CLUSTERING ---
cat("\n--- Environmental Clustering ---\n")

X_env <- as.matrix(train_data[, ..env_vars])
for (i in seq_len(ncol(X_env))) {
  mu <- mean(X_env[, i], na.rm = TRUE)
  X_env[is.na(X_env[, i]), i] <- mu
}
X_scaled <- scale(X_env)

km <- kmeans(X_scaled, centers = N_CLUSTERS, nstart = 25)
train_data[, env_cluster := km$cluster]

cat("Cluster sizes:\n")
print(table(km$cluster))

# --- FEATURE ENGINEERING ---
for (k in 1:(N_CLUSTERS - 1)) {
  train_data[, paste0("cluster_", k) := as.integer(env_cluster == k)]
}

cluster_vars <- paste0("cluster_", 1:(N_CLUSTERS - 1))
available_temp <- intersect(temporal_vars, names(train_data))
feature_vars <- c(cluster_vars, available_temp)

cat(sprintf("Final: %d cluster indicators + %d temporal = %d total features\n",
            length(cluster_vars), length(available_temp), length(feature_vars)))

# --- MODEL TRAINING ---
cat("\n--- Step 1: CSS → ML (Warm Start) Estimation ---\n")
train_start <- Sys.time()

y_train <- train_data[[target_col]]
x_train <- as.matrix(train_data[, ..feature_vars])
x_train_scaled <- scale(x_train)
y_train_ts <- ts(y_train, frequency = PERIOD)

sarimax_cluster_fit <- Arima(
    y_train_ts,
    order = ORDER,
    seasonal = list(order = SEASONAL, period = PERIOD),
    xreg = x_train_scaled,
    method = "CSS-ML"  
  )

train_time <- as.numeric(difftime(Sys.time(), train_start, units = "secs"))
cat(sprintf("\nModel training completed in %.2f seconds\n", train_time))

cat(sprintf("AIC: %.2f | BIC: %.2f | LogLik: %.2f\n",
            sarimax_cluster_fit$aic, sarimax_cluster_fit$bic, sarimax_cluster_fit$loglik))
print(summary(sarimax_cluster_fit))

# --- RESIDUAL DIAGNOSTICS ---
residuals_fit <- residuals(sarimax_cluster_fit)

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

cat("\n--- Cluster-SARIMAX Model Complete ---\n")
