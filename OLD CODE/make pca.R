# ==============================================================
# PCA FEATURES (OUTDOOR) — Period A (train fit, test apply)
# ==============================================================

library(data.table)

set.seed(1234)

# 0) Choose split -----------------------------------------------
train_data <- splits$trainA
test_data  <- splits$testA

# 1) Variables ---------------------------------------------------
env_outdoor  <- c("temperature","wind_speed","sunshine_minutes",
                  "global_radiation","humidity_percent","fog",
                  "rain","snow","thunder","ice")
temporal_vars <- c("business_hours","hour_sin","hour_cos","dow_cos","holiday","dst")
occ_var <- "total_occupancy"
PCA_VAR_THRESHOLD <- 0.90

available_env <- intersect(env_outdoor, names(train_data))
stopifnot(length(available_env) > 0)

# 2) Build matrices; impute NAs with TRAIN means ----------------
Xtr <- as.matrix(train_data[, ..available_env])
Xte <- as.matrix(test_data[,  ..available_env])
for (j in seq_len(ncol(Xtr))) {
  mu <- mean(Xtr[, j], na.rm = TRUE)
  Xtr[is.na(Xtr[, j]), j] <- mu
  Xte[is.na(Xte[, j]), j] <- mu
}

# 3) PCA on TRAIN (center+scale), pick k by >= 90% --------------
pca_fit <- prcomp(Xtr, center = TRUE, scale. = TRUE)
var_ratio <- pca_fit$sdev^2 / sum(pca_fit$sdev^2)
cum_ratio <- cumsum(var_ratio)
k <- which(cum_ratio >= PCA_VAR_THRESHOLD)[1]
if (is.na(k)) k <- ncol(Xtr)
cat(sprintf("PCA retained %d PCs (%.2f%% cumulative variance)\n", k, 100*cum_ratio[k]))

pcs_tr <- predict(pca_fit, Xtr)[, 1:k, drop = FALSE]
pcs_te <- predict(pca_fit, Xte)[, 1:k, drop = FALSE]
colnames(pcs_tr) <- paste0("PC", 1:k)
colnames(pcs_te) <- paste0("PC", 1:k)

# 4) Bind occupancy + temporal (impute simple NAs if any) -------
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

# 5) Final xreg tables ------------------------------------------
xreg_train_A <- cbind(as.data.table(pcs_tr), extras_tr)  # train-only fit
xreg_test_A  <- cbind(as.data.table(pcs_te),  extras_te) # forecasting on test

cat("Built PCA features with PCs:", paste(colnames(pcs_tr), collapse = ", "), "\n")
