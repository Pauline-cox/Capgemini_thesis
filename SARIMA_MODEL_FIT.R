# ===============================================================
# PURE SARIMA MODEL FIT — TRAIN DATA ONLY (NO FORECASTING)
# ===============================================================

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
target_col <- "total_consumption_kWh"

# --- DATA PREP ---
train_data <- model_data[interval >= "2023-07-01" & interval <= "2024-09-30"]
y_train <- train_data[[target_col]]
y_train_ts <- ts(y_train, frequency = PERIOD)

# ===============================================================
# MODEL FITTING — CSS → CSS-ML (Aligned Method)
# ===============================================================

cat("\n--- Step 1: CSS Estimation ---\n")
train_start <- Sys.time()

sarima_css <- tryCatch(
  Arima(y_train_ts,
        order = ORDER,
        seasonal = list(order = SEASONAL, period = PERIOD),
        method = "CSS"),
  error = function(e) {
    cat("ERROR: CSS fit failed:\n", e$message, "\n")
    return(NULL)
  }
)

if (!is.null(sarima_css)) {
  cat("CSS estimation successful — proceeding to ML refinement.\n")
  sarima_fit <- tryCatch(
    Arima(y_train_ts,
          order = ORDER,
          seasonal = list(order = SEASONAL, period = PERIOD),
          method = "CSS-ML",
          fixed = coef(sarima_css)),
    error = function(e) {
      cat("ERROR: CSS-ML refinement failed:\n", e$message, "\n")
      return(NULL)
    }
  )
} else {
  sarima_fit <- NULL
}

# Fallback if both failed
if (is.null(sarima_fit)) {
  cat("Final fallback: CSS-only estimation.\n")
  sarima_fit <- Arima(y_train_ts,
                      order = ORDER,
                      seasonal = list(order = SEASONAL, period = PERIOD),
                      method = "CSS")
}

train_time <- as.numeric(difftime(Sys.time(), train_start, units = "secs"))
cat(sprintf("\nModel training completed in %.2f seconds\n", train_time))

if (!is.null(sarima_fit)) {
  cat(sprintf("AIC: %.2f | BIC: %.2f | LogLik: %.2f\n",
              sarima_fit$aic, sarima_fit$bic, sarima_fit$loglik))
  print(summary(sarima_fit))
}

# ===============================================================
# RESIDUAL DIAGNOSTICS — LJUNG-BOX TEST
# ===============================================================

if (!is.null(sarima_fit)) {
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
}

