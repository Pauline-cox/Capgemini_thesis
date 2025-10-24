# ---------------------------------------------------------------
# MODEL: SARIMAX (PCA Exogenous, Outdoor-only)
# 24-hour rolling forecast without retraining
# ---------------------------------------------------------------

library(data.table)
library(forecast)

set.seed(1234)

# --- Settings ---
PCA_VAR_THRESHOLD <- 0.90
env_outdoor <- c(
  "temperature","wind_speed","sunshine_minutes",
  "global_radiation","humidity_percent","fog","rain","snow","thunder","ice"
)
temporal_vars <- c("business_hours","hour_sin","hour_cos","dow_cos","holiday","dst")
occ_var <- "total_occupancy"   # single name as requested

# --- PCA Feature Extraction (OUTDOOR + occupancy + temporal) ---
extract_pca_features_outdoor <- function(train_data, test_data, var_threshold = PCA_VAR_THRESHOLD) {
  avail_env  <- intersect(env_outdoor, names(train_data))
  avail_temp <- intersect(temporal_vars, names(train_data))
  if (length(avail_env) == 0) stop("No outdoor variables available for PCA.")
  
  X_train <- as.matrix(train_data[, ..avail_env])
  X_test  <- as.matrix(test_data[,  ..avail_env])
  
  # impute with train means
  for (j in seq_len(ncol(X_train))) {
    mu <- mean(X_train[, j], na.rm = TRUE)
    X_train[is.na(X_train[, j]), j] <- mu
    X_test [is.na(X_test [, j]), j] <- mu
  }
  
  pca_model <- prcomp(X_train, center = TRUE, scale. = TRUE)
  var_exp <- pca_model$sdev^2 / sum(pca_model$sdev^2)
  cum_var <- cumsum(var_exp)
  n_comp <- which(cum_var >= var_threshold)[1]
  if (is.na(n_comp)) n_comp <- ncol(X_train)
  
  cat(sprintf("PCA(OUT): retained %d PCs (%.2f%% cumulative variance)\n", n_comp, 100*cum_var[n_comp]))
  
  train_pcs <- predict(pca_model, X_train)[, 1:n_comp, drop = FALSE]
  test_pcs  <- predict(pca_model, X_test )[ , 1:n_comp, drop = FALSE]
  colnames(train_pcs) <- paste0("PC", 1:n_comp)
  colnames(test_pcs)  <- paste0("PC", 1:n_comp)
  
  # bind occupancy + temporal
  extras_train <- cbind(train_data[, ..occ_var], train_data[, ..avail_temp])
  extras_test  <- cbind(test_data[,  ..occ_var], test_data[,  ..avail_temp])
  
  # simple NA guards
  if (anyNA(extras_train[[occ_var]])) {
    mu_occ <- mean(extras_train[[occ_var]], na.rm = TRUE)
    extras_train[[occ_var]][is.na(extras_train[[occ_var]])] <- mu_occ
  }
  if (anyNA(extras_test[[occ_var]])) {
    mu_occ <- mean(extras_test[[occ_var]], na.rm = TRUE)
    extras_test[[occ_var]][is.na(extras_test[[occ_var]])] <- mu_occ
  }
  for (v in avail_temp) {
    if (anyNA(extras_train[[v]])) extras_train[[v]][is.na(extras_train[[v]])] <- 0
    if (anyNA(extras_test [[v]])) extras_test [[v]][is.na(extras_test [[v]])] <- 0
  }
  
  list(
    train = cbind(as.data.table(train_pcs), extras_train),
    test  = cbind(as.data.table(test_pcs),  extras_test),
    n_comp = n_comp,
    var_explained = cum_var[n_comp],
    avail_env = avail_env
  )
}

# --- Rolling 24h SARIMAX Forecast (PCA-xreg) ---
rolling_sarimax_pca_24h <- function(train_data, test_data, order, seasonal, period) {
  # Build PCA xreg once
  pca_obj <- extract_pca_features_outdoor(train_data, test_data, PCA_VAR_THRESHOLD)
  
  n_test <- nrow(test_data)
  n_train <- nrow(train_data)
  forecasts <- rep(NA_real_, n_test)
  
  overall_start <- Sys.time()
  
  # --- Train model once ---
  train_start <- Sys.time()
  y_train <- train_data[[target_col]]
  x_train <- as.matrix(pca_obj$train)
  
  x_train_scaled <- scale(x_train)
  center_vec <- attr(x_train_scaled, "scaled:center")
  scale_vec  <- attr(x_train_scaled, "scaled:scale")
  
  y_train_ts <- ts(y_train, frequency = period)
  model <- Arima(
    y_train_ts,
    order = order,
    seasonal = list(order = seasonal, period = period),
    xreg = x_train_scaled,
    method = "CSS-ML"
  )
  
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  cat(sprintf(
    "Trained SARIMAX-PCA: %d PCs (+%d extras) | AIC=%.2f | Time=%.2fmin\n",
    pca_obj$n_comp, ncol(pca_obj$train) - pca_obj$n_comp, model$aic, train_time
  ))
  cat(sprintf("PCA specifics: PCs=%d | CumVar=%.2f%% | EnvVars={%s}\n",
              pca_obj$n_comp, 100*pca_obj$var_explained, paste(pca_obj$avail_env, collapse=", ")))
  print(summary(model))
  
  # --- Rolling 24-hour forecasts ---
  predict_start <- Sys.time()
  all_y <- c(y_train, test_data[[target_col]])
  all_x <- rbind(x_train, as.matrix(pca_obj$test))
  all_x_scaled <- scale(all_x, center = center_vec, scale = scale_vec)
  
  cat(sprintf("Starting 24h rolling forecasts (%d test hours)...\n", n_test))
  filled <- 0
  
  for (h in seq(-22, n_test - 23)) {
    current_idx <- n_train + h - 1
    if (current_idx < 100) next
    
    hist_y <- all_y[1:current_idx]
    hist_x <- all_x_scaled[1:current_idx, , drop = FALSE]
    future_x <- all_x_scaled[(current_idx + 1):(current_idx + 24), , drop = FALSE]
    if (nrow(future_x) < 24) next
    
    updated <- tryCatch(
      Arima(ts(hist_y, frequency = period), model = model, xreg = hist_x),
      error = function(e) NULL
    )
    if (is.null(updated)) next
    
    fc <- forecast(updated, xreg = future_x, h = 24)
    idx <- current_idx + 24 - n_train
    if (idx >= 1 && idx <= n_test) { forecasts[idx] <- fc$mean[24]; filled <- filled + 1 }
    
    if (filled %% 24 == 0 || filled == n_test) {
      elapsed <- as.numeric(difftime(Sys.time(), predict_start, units = "mins"))
      pct <- (filled / n_test) * 100
      cat(sprintf("Progress: %3d/%d (%.1f%%) | Elapsed: %.2f min\n", filled, n_test, pct, elapsed))
      flush.console()
    }
  }
  
  cat("Forecasting complete.\n")
  
  predict_time <- as.numeric(difftime(Sys.time(), predict_start, units = "mins"))
  total_time <- as.numeric(difftime(Sys.time(), overall_start, units = "mins"))
  
  # Fill NAs
  for (i in which(is.na(forecasts))) {
    forecasts[i] <- ifelse(i == 1, tail(y_train, 1), forecasts[i - 1])
  }
  
  list(
    forecasts = forecasts,
    model = model,
    runtime = total_time,
    train_time = train_time,
    predict_time = predict_time,
    pca_obj = pca_obj
  )
}

# --- Runner for both periods ---
cat(sprintf("\n--- %s [SARIMAX_PCA_OUT] ---\n", label))
res <- rolling_sarimax_pca_24h(train, test, ORDER, SEASONAL, PERIOD)
run_sarimax_pca_outdoor <- function(train, test, label) {
  actual <- test[[target_col]]
  
  eval <- evaluate_forecast(actual, res$forecasts, "SARIMAX_PCA_OUT_24h")
  eval[, `:=`(Runtime_min = res$runtime,
              Train_min = res$train_time,
              Predict_min = res$predict_time,
              Period = label,
              N_Components = res$pca_obj$n_comp,
              Var_Explained = res$pca_obj$var_explained)]
  
  dt <- data.table(Time = seq_along(actual), Actual = actual, Forecast = res$forecasts)
  p <- plot_forecast(dt, "SARIMAX_PCA_OUT_24h", label, color = "purple")
  print(p)
  
  # Coefficients print
  cat("\n--- Coefficients (train fit) ---\n")
  print(round(coef(res$model), 6))
  
  list(eval = eval, forecasts = dt, plot = p, model = res$model, pca_obj = res$pca_obj)
}

# --- Execute ---
splits <- split_periods(model_data)
resultsA_sarimax_pca_out <- run_sarimax_pca_outdoor(splits$trainA, splits$testA, "Period A")
resultsB_sarimax_pca_out <- run_sarimax_pca_outdoor(splits$trainB, splits$testB, "Period B")

all_eval_sarimax_pca_out <- rbind(resultsA_sarimax_pca_out$eval, resultsB_sarimax_pca_out$eval)
print(all_eval_sarimax_pca_out)

# --- Print summary ---
cat("\n--- Summary (SARIMAX_PCA_OUT) ---\n")
cat(sprintf("Model: %s\n", resultsA_sarimax_pca_out$eval$Model[1]))
cat(sprintf(
  "Period A: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | PCs=%d (%.2f%%) | Time=%.2f (Train=%.2f, Predict=%.2f)\n",
  resultsA_sarimax_pca_out$eval$RMSE,
  resultsA_sarimax_pca_out$eval$MAE,
  resultsA_sarimax_pca_out$eval$MAPE,
  resultsA_sarimax_pca_out$eval$R2,
  resultsA_sarimax_pca_out$pca_obj$n_comp, 100*resultsA_sarimax_pca_out$pca_obj$var_explained,
  resultsA_sarimax_pca_out$eval$Runtime_min,
  resultsA_sarimax_pca_out$eval$Train_min,
  resultsA_sarimax_pca_out$eval$Predict_min
))
cat(sprintf(
  "Period B: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | PCs=%d (%.2f%%) | Time=%.2f (Train=%.2f, Predict=%.2f)\n",
  resultsB_sarimax_pca_out$eval$RMSE,
  resultsB_sarimax_pca_out$eval$MAE,
  resultsB_sarimax_pca_out$eval$MAPE,
  resultsB_sarimax_pca_out$eval$R2,
  resultsB_sarimax_pca_out$pca_obj$n_comp, 100*resultsB_sarimax_pca_out$pca_obj$var_explained,
  resultsB_sarimax_pca_out$eval$Runtime_min,
  resultsB_sarimax_pca_out$eval$Train_min,
  resultsB_sarimax_pca_out$eval$Predict_min
))

# --- Save results ---
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Results_SARIMAX_PCA_OUT_24h_%s.rds", timestamp)
saveRDS(
  list(
    model = "SARIMAX_PCA_OUT_24h",
    period_A = resultsA_sarimax_pca_out,
    period_B = resultsB_sarimax_pca_out,
    evaluations = all_eval_sarimax_pca_out,
    parameters = list(
      order = ORDER, seasonal = SEASONAL, period = PERIOD,
      pca_vars = env_outdoor, pca_var_threshold = PCA_VAR_THRESHOLD
    )
  ),
  file = save_name
)
cat(sprintf("\nResults saved to: %s\n", save_name))
cat("SARIMAX (PCA OUT) 24-hour forecast complete!\n")