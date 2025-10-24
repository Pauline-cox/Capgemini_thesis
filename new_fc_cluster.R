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
  
  # choose k by silhouette
  dist_tr <- dist(Ztr)
  sil_vals <- sapply(K_RANGE, function(k) {
    cl <- kmeans(Ztr, centers = k, nstart = 50, iter.max = 100)$cluster
    mean(silhouette(cl, dist_tr)[, "sil_width"])
  })
  sel_tbl <- data.table(k = K_RANGE, Avg_Silhouette = round(sil_vals, 4))
  FINAL_K <- sel_tbl$k[which.max(sel_tbl$Avg_Silhouette)]
  AVG_SIL <- sel_tbl[ k == FINAL_K, Avg_Silhouette ][1]
  cat("Silhouette selection table:\n"); print(sel_tbl)
  cat(sprintf("Selected k=%d (avg silhouette=%.3f)\n", FINAL_K, AVG_SIL))
  
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
    avail_env = avail_env
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
    method = "CSS-ML"
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
  all_x_scaled
  