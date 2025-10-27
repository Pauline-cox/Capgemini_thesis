# ==============================================================
# CLUSTER FEATURES (OUTDOOR) — Period A (train fit, test assign)
# ==============================================================

library(data.table)
library(cluster)

set.seed(1234)

# 0) Choose split ------------------------------------------------
train_data <- splits$trainA
test_data  <- splits$testA

# 1) Variables ---------------------------------------------------
env_outdoor  <- c("temperature","wind_speed","sunshine_minutes",
                  "global_radiation","humidity_percent","fog",
                  "rain","snow","thunder","ice")
temporal_vars <- c("business_hours","hour_sin","hour_cos","dow_cos","holiday","dst")
occ_var <- "total_occupancy"

available_env <- intersect(env_outdoor, names(train_data))
stopifnot(length(available_env) > 0)

# 2) Build env matrices; impute NAs with TRAIN means ------------
Xtr <- as.matrix(train_data[, ..available_env])
Xte <- as.matrix(test_data[,  ..available_env])
for (j in seq_len(ncol(Xtr))) {
  mu <- mean(Xtr[, j], na.rm = TRUE)
  Xtr[is.na(Xtr[, j]), j] <- mu
  Xte[is.na(Xte[, j]), j] <- mu
}

# 3) Standardize with TRAIN stats -------------------------------
cvec <- colMeans(Xtr)
svec <- apply(Xtr, 2, sd); svec[svec == 0] <- 1
Ztr <- scale(Xtr, center = cvec, scale = svec)
Zte <- scale(Xte, center = cvec, scale = svec)

# 4) Fit k-means on TRAIN (k = 4 fixed) -------------------------
K <- 4
km <- kmeans(Ztr, centers = K, nstart = 100, iter.max = 200)
tr_cl <- km$cluster
centers <- km$centers

# 5) Assign TEST by nearest centroid ----------------------------
dmat <- sapply(1:nrow(centers), function(ci) rowSums((Zte - centers[ci,])^2))
te_cl <- max.col(-dmat)

# 6) One-hot dummies; DROP ONE (reference) ----------------------
# TRAIN
Dtr <- as.data.table(matrix(0L, nrow(Ztr), K)); setnames(Dtr, paste0("cluster_", 1:K))
for (k in 1:K) Dtr[[k]] <- as.integer(tr_cl == k)
ref_dummy <- paste0("cluster_", K)   # drop cluster_4 as reference
Dtr[, (ref_dummy) := NULL]

# TEST
Dte <- as.data.table(matrix(0L, nrow(Zte), K)); setnames(Dte, paste0("cluster_", 1:K))
for (k in 1:K) Dte[[k]] <- as.integer(te_cl == k)
Dte[, (ref_dummy) := NULL]

# 7) Bind occupancy + temporal (impute simple NAs if any) -------
extras_tr <- cbind(train_data[, ..occ_var], train_data[, ..temporal_vars])
extras_te <- cbind(test_data[,  ..occ_var], test_data[,  ..temporal_vars])

if (anyNA(extras_tr[[occ_var]])) {
  mu_occ <- mean(extras_tr[[occ_var]], na.rm = TRUE)
  extras_tr[[occ_var]][is.na(extras_tr[[occ_var]])] <- mu_occ
}
if (anyNA(extras_te[[occ_var]])) {
  mu_occ <- mean(extras_te[[occ_var]], na.rm = TRUE)
  extras_te[[occ_var]][is.na(extras_te[[occ_var]])] <- mu_occ
}
for (v in temporal_vars) {
  if (anyNA(extras_tr[[v]])) extras_tr[[v]][is.na(extras_tr[[v]])] <- 0
  if (anyNA(extras_te[[v]])) extras_te[[v]][is.na(extras_te[[v]])] <- 0
}

# 8) Final xreg tables ------------------------------------------
xreg_train_A <- cbind(Dtr, extras_tr)  # used for model fit
xreg_test_A  <- cbind(Dte,  extras_te) # used for forecasting

cat("Built cluster features. Reference (omitted) dummy:", ref_dummy, "\n")
