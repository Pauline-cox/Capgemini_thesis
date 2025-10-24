# ---------------------------------------------------------------
# MODEL: SARIMAX (Cluster Exogenous, Outdoor-only)
# 24-hour rolling forecast without retraining
# ---------------------------------------------------------------

library(data.table)
library(forecast)
library(cluster)

set.seed(1234)

# --- Settings ---
env_outdoor <- c(
  "temperature","wind_speed","sunshine_minutes",
  "global_radiation","humidity_percent","fog","rain","snow","thunder","ice"
)
temporal_vars <- c("business_hours","hour_sin","hour_cos","dow_cos","holiday","dst")
occ_var <- "total_occupancy"
K_RANGE <- 2:8  # silhouette search

# --- Cluster Feature Extraction (OUTDOOR + occupancy + temporal) ---
build_cluster_features_outdoor <- function(train_data, test_data) {
  avail_env  <- intersect(env_outdoor, names(train_data))
  avail_temp <- intersect(temporal_vars, names(train_data))
  if (length(avail_env) == 0) stop("No outdoor variables for clustering.")
  
  Xtr <- as.matrix(train_data[, ..avail_env])
  Xte <- as.matrix(test_data[,  ..avail_env])
  
  # impute with train means
  for (j in seq_len(ncol(Xtr))) {
    mu <- mean(Xtr[, j], na.rm = TRUE)
    Xtr[is.na(Xtr[, j]), j] <- mu
    Xte[is.na(Xte[, j]), j] <- mu
  }
  
  # standardize with train stats
  cvec <- colMeans(Xtr); svec <- apply(Xtr, 2, sd); svec[svec == 0] <- 1
  Ztr <- scale(Xtr, center = cvec, scale = svec)
  Zte <- scale(Xte, center = cvec, scale = svec)
  
  # choose k by silhouette + collect elbow stats
  set.seed(1234)
  dist_tr <- dist(Ztr)
  km_stats <- lapply(K_RANGE, function(k) {
    fit <- kmeans(Ztr, centers = k, nstart = 50, iter.max = 100)
    sil <- mean(cluster::silhouette(fit$cluster, dist_tr)[, "sil_width"])
    data.table(k = k,
               Avg_Silhouette = sil,
               TotWithin = fit$tot.withinss)
  })
  sel_tbl <- rbindlist(km_stats)
  sel_tbl[, `:=`(
    Avg_Silhouette = round(Avg_Silhouette, 4),
    TotWithin = round(TotWithin, 2)
  )]
  FINAL_K <- sel_tbl$k[which.max(sel_tbl$Avg_Silhouette)]
  AVG_SIL <- sel_tbl[k == FINAL_K, Avg_Silhouette][1]
  
  cat("Silhouette selection table:\n"); print(sel_tbl)
  cat(sprintf("Selected k=%d (avg silhouette=%.3f)\n", FINAL_K, AVG_SIL))
  
  # ---- NEW: Scree/Elbow & Silhouette plots ----
  p_elbow <- ggplot(sel_tbl, aes(x = k, y = TotWithin)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    geom_vline(xintercept = FINAL_K, linetype = "dashed") +
    labs(title = "Elbow (Total Within-Cluster SS vs K)",
         x = "K (clusters)", y = "Total within-cluster SS") +
    theme_minimal(base_size = 11)
  
  p_sil <- ggplot(sel_tbl, aes(x = k, y = Avg_Silhouette)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    geom_vline(xintercept = FINAL_K, linetype = "dashed") +
    labs(title = "Average Silhouette vs K",
         x = "K (clusters)", y = "Average silhouette width") +
    theme_minimal(base_size = 11)
  
  print(p_elbow)
  print(p_sil)
  # ---------------------------------------------
  
  # final kmeans & assign test
  set.seed(1234)
  km <- kmeans(Ztr, centers = FINAL_K, nstart = 100, iter.max = 200)
  tr_cl <- km$cluster; centers <- km$centers
  dmat <- sapply(1:nrow(centers), function(ci) rowSums((Zte - centers[ci,])^2))
  te_cl <- max.col(-dmat)
  
  # one-hot; drop one cluster dummy to avoid dummy trap
  Dtr <- as.data.table(matrix(0L, nrow(Ztr), FINAL_K)); setnames(Dtr, paste0("cluster_", 1:FINAL_K))
  for (k in 1:FINAL_K) Dtr[[k]] <- as.integer(tr_cl == k)
  ref_dummy <- paste0("cluster_", FINAL_K); Dtr[, (ref_dummy) := NULL]
  
  Dte <- as.data.table(matrix(0L, nrow(Zte), FINAL_K)); setnames(Dte, paste0("cluster_", 1:FINAL_K))
  for (k in 1:FINAL_K) Dte[[k]] <- as.integer(te_cl == k)
  Dte[, (ref_dummy) := NULL]
  
  # occupancy + temporal
  extras_tr <- cbind(train_data[, ..occ_var], train_data[, ..avail_temp])
  extras_te <- cbind(test_data[,  ..occ_var], test_data[,  ..avail_temp])
  
  # NA guards
  if (anyNA(extras_tr[[occ_var]])) {
    mu_occ <- mean(extras_tr[[occ_var]], na.rm = TRUE); extras_tr[[occ_var]][is.na(extras_tr[[occ_var]])] <- mu_occ
  }
  if (anyNA(extras_te[[occ_var]])) {
    mu_occ <- mean(extras_te[[occ_var]], na.rm = TRUE); extras_te[[occ_var]][is.na(extras_te[[occ_var]])] <- mu_occ
  }
  for (v in avail_temp) {
    if (anyNA(extras_tr[[v]])) extras_tr[[v]][is.na(extras_tr[[v]])] <- 0
    if (anyNA(extras_te [[v]])) extras_te [[v]][is.na(extras_te [[v]])] <- 0
  }
  
  # sizes for print
  sizes <- table(tr_cl)
  size_tbl <- data.table(Cluster = as.integer(names(sizes)),
                         Train_N = as.integer(sizes),
                         Train_pct = round(100*as.integer(sizes)/sum(sizes), 2))
  
  cat("Cluster sizes (train):\n"); print(size_tbl)
  cat("Reference (omitted) dummy:", ref_dummy, "\n")
  
  list(
    train = cbind(Dtr, extras_tr),
    test  = cbind(Dte, extras_te),
    K = FINAL_K,
    Avg_Silhouette = AVG_SIL,
    size_tbl = size_tbl,
    ref_dummy = ref_dummy,
    avail_env = avail_env,
    k_selection = sel_tbl        # <- return the table for saving/logging
  )
}

# --- Rolling 24h SARIMAX Forecast (Cluster-xreg) ---
rolling_sarimax_cluster_24h <- function(train_data, test_data, order, seasonal, period) {
  cl_obj <- build_cluster_features_outdoor(train_data, test_data)
  
  n_test <- nrow(test_data)
  n_train <- nrow(train_data)
  forecasts <- rep(NA_real_, n_test)
  
  overall_start <- Sys.time()
  
  # --- Train model once ---
  train_start <- Sys.time()
  y_train <- train_data[[target_col]]
  x_train <- as.matrix(cl_obj$train)
  
  # Drop zero-variance cols if any (rare, but holiday/dst can be constant)
  nzv <- apply(x_train, 2, function(z) sd(as.numeric(z)) == 0)
  if (any(nzv)) {
    cat("Dropping zero-variance columns:", paste(colnames(x_train)[nzv], collapse=", "), "\n")
    x_train <- x_train[, !nzv, drop = FALSE]
    cl_obj$test <- cl_obj$test[, !nzv, with = FALSE]
  }
  
  x_train_scaled <- scale(x_train)
  center_vec <- attr(x_train_scaled, "scaled:center")
  scale_vec  <- attr(x_train_scaled, "scaled:scale")
  
  y_train_ts <- ts(y_train, frequency = period)
  model <- Arima(
    y_train_ts,
    order = order,
    seasonal = list(order = seasonal, period = period),
    xreg = x_train_scaled,
    method = "CSS"
  )
  
  train_time <- as.numeric(difftime(Sys.time(), train_start, units = "mins"))
  cat(sprintf(
    "Trained SARIMAX-CLUST: K=%d (ref=%s) | extras=%d | AIC=%.2f | Time=%.2fmin\n",
    cl_obj$K, cl_obj$ref_dummy, ncol(cl_obj$train) - (cl_obj$K - 1), model$aic, train_time
  ))
  cat(sprintf("Cluster specifics: K=%d | Avg silhouette=%.3f | EnvVars={%s}\n",
              cl_obj$K, cl_obj$Avg_Silhouette, paste(cl_obj$avail_env, collapse=", ")))
  print(summary(model))
  
  # --- Rolling 24-hour forecasts ---
  predict_start <- Sys.time()
  all_y <- c(y_train, test_data[[target_col]])
  all_x <- rbind(x_train, as.matrix(cl_obj$test))
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
    cl_obj = cl_obj
  )
}

# --- Runner for both periods ---
run_sarimax_cluster_outdoor <- function(train, test, label) {
  cat(sprintf("\n--- %s [SARIMAX_CLUSTER_OUT] ---\n", label))
  res <- rolling_sarimax_cluster_24h(train, test, ORDER, SEASONAL, PERIOD)
  actual <- test[[target_col]]
  
  eval <- evaluate_forecast(actual, res$forecasts, "SARIMAX_CLUSTER_OUT_24h")
  eval[, `:=`(Runtime_min = res$runtime,
              Train_min = res$train_time,
              Predict_min = res$predict_time,
              Period = label,
              K = res$cl_obj$K,
              Avg_Silhouette = res$cl_obj$Avg_Silhouette)]
  dt <- data.table(Time = seq_along(actual), Actual = actual, Forecast = res$forecasts)
  p <- plot_forecast(dt, "SARIMAX_CLUSTER_OUT_24h", label, color = "darkgreen")
  print(p)
  
  cat("\n--- Coefficients (train fit) ---\n")
  print(round(coef(res$model), 6))
  
  list(eval = eval, forecasts = dt, plot = p, model = res$model, cl_obj = res$cl_obj)
}

# --- Execute ---
splits <- split_periods(model_data)
resultsA_sarimax_cluster_out <- run_sarimax_cluster_outdoor(splits$trainA, splits$testA, "Period A")
resultsB_sarimax_cluster_out <- run_sarimax_cluster_outdoor(splits$trainB, splits$testB, "Period B")

all_eval_sarimax_cluster_out <- rbind(resultsA_sarimax_cluster_out$eval, resultsB_sarimax_cluster_out$eval)
print(all_eval_sarimax_cluster_out)

# --- Print summary ---
cat("\n--- Summary (SARIMAX_CLUSTER_OUT) ---\n")
cat(sprintf("Model: %s\n", resultsA_sarimax_cluster_out$eval$Model[1]))
cat(sprintf(
  "Period A: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | K=%d (sil=%.3f) | Time=%.2f (Train=%.2f, Predict=%.2f)\n",
  resultsA_sarimax_cluster_out$eval$RMSE,
  resultsA_sarimax_cluster_out$eval$MAE,
  resultsA_sarimax_cluster_out$eval$MAPE,
  resultsA_sarimax_cluster_out$eval$R2,
  resultsA_sarimax_cluster_out$cl_obj$K, resultsA_sarimax_cluster_out$cl_obj$Avg_Silhouette,
  resultsA_sarimax_cluster_out$eval$Runtime_min,
  resultsA_sarimax_cluster_out$eval$Train_min,
  resultsA_sarimax_cluster_out$eval$Predict_min
))
cat(sprintf(
  "Period B: RMSE=%.2f | MAE=%.2f | MAPE=%.2f%% | R2=%.4f | K=%d (sil=%.3f) | Time=%.2f (Train=%.2f, Predict=%.2f)\n",
  resultsB_sarimax_cluster_out$eval$RMSE,
  resultsB_sarimax_cluster_out$eval$MAE,
  resultsB_sarimax_cluster_out$eval$MAPE,
  resultsB_sarimax_cluster_out$eval$R2,
  resultsB_sarimax_cluster_out$cl_obj$K, resultsB_sarimax_cluster_out$cl_obj$Avg_Silhouette,
  resultsB_sarimax_cluster_out$eval$Runtime_min,
  resultsB_sarimax_cluster_out$eval$Train_min,
  resultsB_sarimax_cluster_out$eval$Predict_min
))

# --- Save results ---
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
save_name <- sprintf("Results_SARIMAX_CLUSTER_OUT_24h_%s.rds", timestamp)
saveRDS(
  list(
    model = "SARIMAX_CLUSTER_OUT_24h",
    period_A = resultsA_sarimax_cluster_out,
    period_B = resultsB_sarimax_cluster_out,
    evaluations = all_eval_sarimax_cluster_out,
    parameters = list(
      order = ORDER, seasonal = SEASONAL, period = PERIOD,
      cluster_vars = env_outdoor, k_range = K_RANGE
    )
  ),
  file = save_name
)
cat(sprintf("\nResults saved to: %s\n", save_name))
cat("SARIMAX (CLUSTER OUT) 24-hour forecast complete!\n")