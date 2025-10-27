# ==============================================================
# Author: Pauline Cox
# Script: gridsearch_sarima.R
#
# Description: Performs a comprehensive SARIMA grid search over 
# specified seasonal and non-seasonal orders to identify the 
# optimal model for hourly energy consumption data. 
# The script includes ADF stationarity testing, ACF/PACF 
# visualization, parameter tuning, model diagnostics, 
# and automatic saving of the best configuration.
#
# Input:
#   - Preprocessed and feature-engineered dataset (model_data)
#
# Output:
#   - Best SARIMA model (by AIC)
#   - Full parameter grid search results
#   - ACF/PACF diagnostic plots
# ==============================================================

set.seed(1234)

# --- Settings ---
period <- 168  # Weekly seasonality 

# --- Data Split ---
tz_ref <- attr(model_data$interval, "tzone")
train_val_data <- model_data[
  interval >= as.POSIXct("2023-07-01 00:00:00", tz = tz_ref) &
    interval <  as.POSIXct("2024-10-01 00:00:00", tz = tz_ref)
]
y_full <- ts(train_val_data$total_consumption_kWh, frequency = period)

# --- Stationarity Test ---
cat("ADF test for stationarity:\n")
suppressWarnings(print(adf.test(y_full, alternative = "stationary")))

# --- ACF/PACF Diagnostics ---
acf_obj <- Acf(y_full, lag.max = 168, plot = FALSE)
pacf_obj <- Pacf(y_full, lag.max = 168, plot = FALSE)

acf_df <- data.frame(lag = acf_obj$lag, acf = acf_obj$acf)
pacf_df <- data.frame(lag = pacf_obj$lag, pacf = pacf_obj$acf)
conf_level <- 1.96 / sqrt(length(y_full))

p_acf <- ggplot(acf_df, aes(x = lag, y = acf)) +
  geom_segment(aes(xend = lag, yend = 0), color = "black", linewidth = 0.7) +
  geom_hline(yintercept = c(-conf_level, 0, conf_level),
             linetype = c("dashed", "solid", "dashed"),
             color = c("grey40", "black", "grey40")) +
  labs(title = "Autocorrelation Function (ACF)", x = "Lag", y = "ACF") +
  theme_minimal(base_size = 13)

p_pacf <- ggplot(pacf_df, aes(x = lag, y = pacf)) +
  geom_segment(aes(xend = lag, yend = 0), color = "black", linewidth = 0.7) +
  geom_hline(yintercept = c(-conf_level, 0, conf_level),
             linetype = c("dashed", "solid", "dashed"),
             color = c("grey40", "black", "grey40")) +
  labs(title = "Partial Autocorrelation Function (PACF)", x = "Lag", y = "PACF") +
  theme_minimal(base_size = 13)

grid.arrange(p_acf, p_pacf, ncol = 1)

# --- Parameter Grid Definition ---
param_grid <- expand.grid(
  p = 0:2, d = 0, q = 0:2,
  P = 0:1, D = 1, Q = 0:1,
  seasonal = period
)

results <- data.table(
  p = integer(), d = integer(), q = integer(),
  P = integer(), D = integer(), Q = integer(),
  AIC = numeric(), AICc = numeric(), BIC = numeric(),
  loglik = numeric(), sigma2 = numeric(),
  convergence = logical(), error_msg = character()
)

start_time <- Sys.time()
best_model <- NULL
best_aic <- Inf

# --- Grid Search Loop ---
for (i in seq_len(nrow(param_grid))) {
  gi <- param_grid[i, ]
  cat(sprintf("Fitting SARIMA(%d,%d,%d)(%d,%d,%d)[%d] (%d of %d)\n",
              gi$p, gi$d, gi$q, gi$P, gi$D, gi$Q, period, i, nrow(param_grid)))
  
  fit <- try(
    suppressWarnings(
      Arima(y_full,
            order = c(gi$p, gi$d, gi$q),
            seasonal = list(order = c(gi$P, gi$D, gi$Q), period = period),
            method = "CSS-ML")
    ),
    silent = TRUE
  )
  
  if (inherits(fit, "try-error") || is.null(fit)) {
    results <- rbind(results, data.table(
      p = gi$p, d = gi$d, q = gi$q,
      P = gi$P, D = gi$D, Q = gi$Q,
      AIC = NA, AICc = NA, BIC = NA,
      loglik = NA, sigma2 = NA,
      convergence = FALSE, error_msg = "fit failed"
    ), fill = TRUE)
    next
  }
  
  results <- rbind(results, data.table(
    p = gi$p, d = gi$d, q = gi$q,
    P = gi$P, D = gi$D, Q = gi$Q,
    AIC = fit$aic, AICc = fit$aicc, BIC = fit$bic,
    loglik = fit$loglik, sigma2 = fit$sigma2,
    convergence = TRUE, error_msg = ""
  ), fill = TRUE)
  
  if (!is.null(fit$aic) && fit$aic < best_aic) {
    best_aic <- fit$aic
    best_model <- fit
    best <- gi
  }
}

# --- Process Results ---
total_time <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 2)
cat("\nGrid search completed in", total_time, "minutes.\n")

results_clean <- results[convergence == TRUE]
if (nrow(results_clean) == 0) stop("No models converged successfully.")
results_sorted <- results_clean[order(AIC)]

# --- Best Model Diagnostics ---
cat("\nBest model (by AIC):\n")
cat(sprintf("SARIMA(%d,%d,%d)(%d,%d,%d)[%d]\n",
            best$p, best$d, best$q, best$P, best$D, best$Q, period))
cat(sprintf("AIC: %.2f | BIC: %.2f | LogLik: %.2f\n\n",
            best_model$aic, best_model$bic, best_model$loglik))

lb_test <- Box.test(residuals(best_model), lag = 10, type = "Ljung-Box")
cat("Ljung-Box p-value:", round(lb_test$p.value, 4), "\n")
cat("Residuals independent:", ifelse(lb_test$p.value > 0.05, "YES", "NO"), "\n")

# --- Save Best Model ---
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Best_SARIMA_CSSML_%s.rds", timestamp)
saveRDS(list(
  best_model = best_model,
  best_params = list(
    order = c(best$p, best$d, best$q),
    seasonal = c(best$P, best$D, best$Q),
    period = period
  ),
  results = results_sorted,
  runtime_min = total_time
), file = save_name)

cat("\nBest model saved to:", save_name, "\n")
