# ===============================================================
# CLUSTERING-BASED SARIMA MODEL FIT — TRAIN DATA ONLY
# ===============================================================

cat("\n===============================================================\n")
cat("CLUSTER-BASED SARIMA MODEL FIT\n")
cat("===============================================================\n")

suppressPackageStartupMessages({
  library(stats)
  library(forecast)
  library(tseries)
})

set.seed(1234)

# --- SETTINGS ---
N_CLUSTERS <- 3

# --- CLUSTERING ON ENVIRONMENTAL VARIABLES ---
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

# One-hot encode clusters
for (k in 1:(N_CLUSTERS - 1)) {
  train_data[, paste0("cluster_", k) := as.integer(env_cluster == k)]
}

cluster_vars <- paste0("cluster_", 1:(N_CLUSTERS - 1))
available_temp <- intersect(temporal_vars, names(train_data))
feature_vars <- c(cluster_vars, available_temp)

cat(sprintf("Final: %d cluster indicators + %d temporal = %d total features\n",
            length(cluster_vars), length(available_temp), length(feature_vars)))

# --- MODEL TRAINING ---
cat("\n========== TRAINING CLUSTER-SARIMA MODEL ==========\n")

y_train <- train_data[[target_col]]
x_train <- as.matrix(train_data[, ..feature_vars])
x_train_scaled <- scale(x_train)
y_train_ts <- ts(y_train, frequency = PERIOD)

# --- Step 1: CSS estimation ---
cat("\n--- Step 1: CSS Estimation ---\n")
sarima_cluster_css <- tryCatch(
  Arima(y_train_ts,
        order = ORDER,
        seasonal = list(order = SEASONAL, period = PERIOD),
        xreg = x_train_scaled,
        method = "CSS"),
  error = function(e) {
    cat("ERROR: CSS fit failed:\n", e$message, "\n")
    return(NULL)
  }
)

# --- Step 2: CSS-ML refinement ---
if (!is.null(sarima_cluster_css)) {
  cat("CSS estimation successful — proceeding to ML refinement.\n")
  sarima_cluster_fit <- tryCatch(
    Arima(y_train_ts,
          order = ORDER,
          seasonal = list(order = SEASONAL, period = PERIOD),
          xreg = x_train_scaled,
          method = "CSS-ML",
          fixed = sarima_cluster_css$coef),
    error = function(e) {
      cat("ERROR: CSS-ML fit failed:\n", e$message, "\n")
      return(NULL)
    }
  )
} else {
  sarima_cluster_fit <- NULL
}

# --- Summary ---
if (!is.null(sarima_cluster_fit)) {
  cat(sprintf("\nAIC: %.2f | BIC: %.2f | LogLik: %.2f\n",
              sarima_cluster_fit$aic, sarima_cluster_fit$bic, sarima_cluster_fit$loglik))
  print(summary(sarima_cluster_fit))
  
  residuals_fit <- residuals(sarima_cluster_fit)
  cat("\n--- Residual Diagnostics ---\n")
  lb_test <- Box.test(residuals_fit, lag = 24, type = "Ljung-Box",
                      fitdf = length(sarima_cluster_fit$coef))
  print(lb_test)
  
  par(mfrow = c(2, 2))
  plot(residuals_fit, main = "Residuals", col = "steelblue")
  acf(residuals_fit, main = "ACF of Residuals")
  pacf(residuals_fit, main = "PACF of Residuals")
  qqnorm(residuals_fit); qqline(residuals_fit, col = "red")
  par(mfrow = c(1, 1))
  
  if (lb_test$p.value < 0.05) {
    cat("\n⚠️ Residuals are likely *not* white noise (p < 0.05)\n")
  } else {
    cat("\n✅ Residuals appear to be white noise (p ≥ 0.05)\n")
  }
}

cat("\n========== CLUSTER-SARIMA MODEL COMPLETE ==========\n")