# Function for grid search for SARIMA order selection (minimal fix version + fallback retry)

sarima_grid_search <- function(period = 168) {
  # Combine training + validation
  y_full <- ts(c(train_data$total_consumption_kWh, val_data$total_consumption_kWh),
               frequency = period)
  
  # Stationarity test
  cat("ADF test for stationarity:\n")
  suppressWarnings(print(adf.test(y_full, alternative = "stationary")))
  cat("\n")
  
  # Parameter grid
  # param_grid <- expand.grid(
  #   p = 0:2, d = 0, q = 0:2,
  #   P = 0:1, D = 1, Q = 0:1,
  #   seasonal = period
  # )
  
  # Small grid for testing
  param_grid <- expand.grid(
    p = 1, d = 0, q = 1,
    P = 1, D = 1, Q = 1
  )
  
  cat("Total combinations:", nrow(param_grid), "\n")
  
  results <- data.table(
    p = integer(), d = integer(), q = integer(),
    P = integer(), D = integer(), Q = integer(),
    AIC = numeric(), AICc = numeric(), BIC = numeric(),
    loglik = numeric(), sigma2 = numeric(),
    convergence = logical(), error_msg = character()
  )
  
  start_time <- Sys.time()
  
  # Grid Search Loop
  for (i in seq_len(nrow(param_grid))) {
    gi <- param_grid[i, ]
    cat(sprintf("Progress: %d / %d -> SARIMA(%d,%d,%d)(%d,%d,%d)[%d]\n",
                i, nrow(param_grid), gi$p, gi$d, gi$q, gi$P, gi$D, gi$Q, period))
    
    iter_start <- Sys.time()
    
    fit <- try(
      suppressWarnings(
        Arima(y_full,
              order = c(gi$p, gi$d, gi$q),
              seasonal = list(order = c(gi$P, gi$D, gi$Q), period = period),
              method = "CSS-ML")
      ),
      silent = TRUE
    )
    
    # Retry with CSS→ML warm start if CSS-ML fails
    if (inherits(fit, "try-error")) {
      cat("Failed initial fit, retrying with CSS→ML warm start...\n")
      fit <- try(
        suppressWarnings(
          Arima(y_full,
                model = Arima(y_full,
                              order = c(gi$p, gi$d, gi$q),
                              seasonal = list(order = c(gi$P, gi$D, gi$Q), period = period),
                              method = "CSS",
                              include.mean = FALSE),
                method = "ML",
                include.mean = FALSE,
                transform.pars = FALSE,
                optim.control = list(maxit = 5000, reltol = 1e-8))
        ),
        silent = TRUE
      )
    }
    
    if (inherits(fit, "try-error")) {
      cat("Failed: ", conditionMessage(attr(fit, "condition")), "\n")
      results <- rbind(
        results,
        data.table(
          p = gi$p, d = gi$d, q = gi$q,
          P = gi$P, D = gi$D, Q = gi$Q,
          AIC = NA, AICc = NA, BIC = NA,
          loglik = NA, sigma2 = NA,
          convergence = FALSE, error_msg = as.character(fit)
        ),
        fill = TRUE
      )
      next
    }
    
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
    
    iter_time <- round(as.numeric(difftime(Sys.time(), iter_start, units = "secs")), 1)
    elapsed   <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 2)
    remaining <- round((nrow(param_grid) - i) * (elapsed / i), 2)
    cat(sprintf("  Time: %.1fs | Elapsed: %.2fm | Est. Remaining: %.2fm\n\n",
                iter_time, elapsed, remaining))
  }
  
  total_time <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 2)
  cat("Grid Search Complete. Total time:", total_time, "minutes\n\n")
  
  # Process and Select
  results_clean <- results[convergence == TRUE]
  if (nrow(results_clean) == 0) stop("No models converged successfully.")
  results_sorted <- results_clean[order(AIC)]
  
  best <- results_sorted[1, ]
  cat("Selected best model by AIC:\n")
  cat(sprintf("SARIMA(%d,%d,%d)(%d,%d,%d)[%d]\n",
              best$p, best$d, best$q, best$P, best$D, best$Q, period))
  cat("AIC:", round(best$AIC, 2), "\n\n")
  
  # Fit final model
  best_model <- Arima(y_full,
                      order = c(best$p, best$d, best$q),
                      seasonal = list(order = c(best$P, best$D, best$Q), period = period),
                      method = "CSS")
  
  # Diagnostics
  lb_test <- Box.test(residuals(best_model), lag = 10, type = "Ljung-Box")
  cat("Ljung-Box p-value:", round(lb_test$p.value, 4), "\n")
  cat("Residuals independent:", ifelse(lb_test$p.value > 0.05, "YES", "NO"), "\n")
  
  # Return results
  return(list(
    results = results_sorted,
    best_model = best_model,
    best_order = c(best$p, best$d, best$q),
    best_seasonal = c(best$P, best$D, best$Q),
    runtime_min = total_time
  ))
}
