# ---------------------------------------------------------------
# PCA-BASED SARIMAX MODEL FIT — TRAIN DATA ONLY (NO FORECASTING)
# ---------------------------------------------------------------

suppressPackageStartupMessages({
  library(forecast)
  library(data.table)
  library(tseries)
})

set.seed(1234)

# --- SETTINGS ---
ORDER    <- c(2, 0, 2)
SEASONAL <- c(0, 1, 1)
PERIOD   <- 168L
PCA_VAR_THRESHOLD <- 0.75
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

# --- PCA FEATURE EXTRACTION ---
cat("\n--- PCA Feature Extraction ---\n")

X_env <- as.matrix(train_data[, ..env_vars])
X_env[is.na(X_env)] <- apply(X_env, 2, function(x) mean(x, na.rm = TRUE))

pca_model <- prcomp(X_env, center = TRUE, scale. = TRUE)
var_expl <- pca_model$sdev^2 / sum(pca_model$sdev^2)
cum_var <- cumsum(var_expl)
n_components <- which(cum_var >= PCA_VAR_THRESHOLD)[1]

cat(sprintf("Selected %d PCs explaining %.1f%% of variance\n",
            n_components, cum_var[n_components] * 100))

train_pcs <- predict(pca_model, X_env)[, 1:n_components, drop = FALSE]
colnames(train_pcs) <- paste0("PC", 1:n_components)

available_temp <- intersect(temporal_vars, names(train_data))
train_features <- cbind(as.data.table(train_pcs),
                        train_data[, ..available_temp])

cat(sprintf("Final features: %d PCs + %d temporal = %d total\n",
            n_components, length(available_temp), ncol(train_features)))

# --- MODEL TRAINING ---
cat("\n--- Step 1: CSS → ML (Warm Start) Estimation ---\n")
train_start <- Sys.time()

y_train <- train_data[[target_col]]
x_train <- as.matrix(train_features)
x_train_scaled <- scale(x_train)
y_train_ts <- ts(y_train, frequency = PERIOD)

sarimax_pca_fit <- Arima(
    y_train_ts,
    order = ORDER,
    seasonal = list(order = SEASONAL, period = PERIOD),
    xreg = x_train_scaled,
    method = "CSS-ML"  
  )

train_time <- as.numeric(difftime(Sys.time(), train_start, units = "secs"))
cat(sprintf("\nModel training completed in %.2f seconds\n", train_time))

cat(sprintf("AIC: %.2f | BIC: %.2f | LogLik: %.2f\n",
            sarimax_pca_fit$aic, sarimax_pca_fit$bic, sarimax_pca_fit$loglik))
print(summary(sarimax_pca_fit))

# --- RESIDUAL DIAGNOSTICS ---
residuals_fit <- residuals(sarimax_pca_fit)

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

cat("\n--- PCA-SARIMAX Model Complete ---\n")
