# ===============================================================
# SARIMA GRID SEARCH 
# ===============================================================

# --- SETTINGS ---
period <- 168
set.seed(1234)

# --- DATA SPLIT ---
tz_ref <- attr(model_data$interval, "tzone")

train_val_data <- model_data[
  interval >= as.POSIXct("2023-07-01 00:00:00", tz = tz_ref) &
    interval <  as.POSIXct("2024-10-01 00:00:00", tz = tz_ref)
]
y_full <- ts(train_val_data$total_consumption_kWh,frequency = period)

# --- Stationarity test ---
cat("ADF test for stationarity:\n")
suppressWarnings(print(adf.test(y_full, alternative = "stationary")))
cat("\n")

# Compute ACF and PACF
acf_obj <- Acf(y_full, lag.max = 168, plot = FALSE)
pacf_obj <- Pacf(y_full, lag.max = 168, plot = FALSE)

# Convert to data frames for ggplot
acf_df <- data.frame(lag = acf_obj$lag, acf = acf_obj$acf)
pacf_df <- data.frame(lag = pacf_obj$lag, pacf = pacf_obj$acf)

# Confidence interval
conf_level <- 1.96 / sqrt(length(y_full))

# --- ACF Plot ---
p1 <- ggplot(acf_df, aes(x = lag, y = acf)) +
  geom_segment(aes(xend = lag, yend = 0), color = "black", linewidth = 0.7) +
  geom_hline(yintercept = 0, color = "black") +
  geom_hline(yintercept = c(-conf_level, conf_level),
             linetype = "dashed", color = "grey40") +
  labs(title = "Autocorrelation Function (ACF)",
       x = "Lag", y = "ACF") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    panel.grid.minor = element_blank()
  )

# --- PACF Plot ---
p2 <- ggplot(pacf_df, aes(x = lag, y = pacf)) +
  geom_segment(aes(xend = lag, yend = 0), color = "black", linewidth = 0.7) +
  geom_hline(yintercept = 0, color = "black") +
  geom_hline(yintercept = c(-conf_level, conf_level),
             linetype = "dashed", color = "grey40") +
  labs(title = "Partial Autocorrelation Function (PACF)",
       x = "Lag", y = "PACF") +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    panel.grid.minor = element_blank()
  )

# Display stacked vertically
grid.arrange(p1, p2, ncol = 1)

# --- PARAMETER GRID ---
param_grid <- expand.grid(
  p = 0:2, d = 0, q = 0:2,
  P = 0:1, D = 1, Q = 0:1,
  seasonal = period
)
# --- GRID FOR TESTING ---
# param_grid <- data.frame(
#   p = c(1, 2, 1),
#   d = c(0, 0, 0),
#   q = c(1, 2, 1),
#   P = c(1, 0, 0),
#   D = c(1, 1, 1),
#   Q = c(1, 1, 1)
# )
cat("Total combinations:", nrow(param_grid), "\n")


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

# --- GRID SEARCH LOOP ---
for (i in seq_len(nrow(param_grid))) {
  gi <- param_grid[i, ]
  cat(sprintf("Progress: %d / %d -> SARIMA(%d,%d,%d)(%d,%d,%d)[%d]\n",
              i, nrow(param_grid), gi$p, gi$d, gi$q, gi$P, gi$D, gi$Q, period))
  
  iter_start <- Sys.time()
  
  # --- Direct CSS-ML fit ---
  fit <- try(
    suppressWarnings(
      Arima(y_full,
            order = c(gi$p, gi$d, gi$q),
            seasonal = list(order = c(gi$P, gi$D, gi$Q), period = period),
            method = "CSS-ML")
    ),
    silent = TRUE
  )
  
  # --- Handle failures ---
  if (inherits(fit, "try-error") || is.null(fit)) {
    cat("Model failed to converge, skipping.\n\n")
    results <- rbind(
      results,
      data.table(
        p = gi$p, d = gi$d, q = gi$q,
        P = gi$P, D = gi$D, Q = gi$Q,
        AIC = NA, AICc = NA, BIC = NA,
        loglik = NA, sigma2 = NA,
        convergence = FALSE, error_msg = "fit failed"
      ),
      fill = TRUE
    )
    next
  }
  
  # --- Record metrics ---
  results <- rbind(
    results,
    data.table(
      p = gi$p, d = gi$d, q = gi$q,
      P = gi$P, D = gi$D, Q = gi$Q,
      AIC = fit$aic, AICc = fit$aicc, BIC = fit$bic,
      loglik = fit$loglik, sigma2 = fit$sigma2,
      convergence = TRUE, error_msg = ""
    ),
    fill = TRUE
  )
  
  cat(sprintf("Done | BIC: %.2f | AIC: %.2f\n", fit$bic, fit$aic))
  
  # --- Save best model immediately ---
  if (!is.null(fit$aic) && fit$aic < best_aic) {
    best_aic <- fit$aic
    best_model <- fit
    best <- gi
    cat("✓ New best model found.\n")
  }
  
  iter_time <- round(as.numeric(difftime(Sys.time(), iter_start, units = "secs")), 1)
  elapsed   <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 2)
  remaining <- round((nrow(param_grid) - i) * (elapsed / i), 2)
  cat(sprintf("  Time: %.1fs | Elapsed: %.2fm | Est. Remaining: %.2fm\n\n",
              iter_time, elapsed, remaining))
}

# --- PROCESS RESULTS ---
total_time <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 2)
cat("\nGrid Search Complete. Total time:", total_time, "minutes\n\n")

results_clean <- results[convergence == TRUE]
if (nrow(results_clean) == 0) stop("No models converged successfully.")
results_sorted <- results_clean[order(AIC)]

# --- PRINT BEST MODEL INFO ---
cat("Selected best model by AIC:\n")
cat(sprintf("SARIMA(%d,%d,%d)(%d,%d,%d)[%d]\n",
            best$p, best$d, best$q, best$P, best$D, best$Q, period))
cat(sprintf("AIC: %.2f | BIC: %.2f | LogLik: %.2f\n\n",
            best_model$aic, best_model$bic, best_model$loglik))

# --- DIAGNOSTICS ---
lb_test <- Box.test(residuals(best_model), lag = 10, type = "Ljung-Box")
cat("Ljung-Box p-value:", round(lb_test$p.value, 4), "\n")
cat("Residuals independent:", ifelse(lb_test$p.value > 0.05, "YES", "NO"), "\n")

# --- PRINT RESULTS ---
cat("\n--- SARIMA GRID SEARCH RESULTS ---\n")
print(results_sorted)
cat("\n--- BEST MODEL SUMMARY ---\n")
print(summary(best_model))

# --- AUTO-SAVE BEST MODEL ---
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Best_SARIMA_CSSML_%s.rds", timestamp)

saveRDS(
  list(
    best_model = best_model,
    best_params = list(
      order = c(best$p, best$d, best$q),
      seasonal = c(best$P, best$D, best$Q),
      period = period
    ),
    results = results_sorted,
    runtime_min = total_time
  ),
  file = save_name
)

cat(sprintf("\nBest model saved to: %s\n", save_name))
cat("SARIMA grid search complete!\n")