# ==============================================================
# Author: Pauline Cox
# Script: diagnostics_pca.R
#
# Description: Performs PCA diagnostics on  environmental 
# variables to determine the optimal number of components for 
# dimensionality reduction. Includes variance analysis, loadings, 
# communalities, and visualization (scree plot).
#
# Input: 
#   - Preprocessed and feature-engineered dataset (model_data)
#   - Split data (splits$trainA / splits$trainB)
#
# Output: 
#   - PCA summary tables and diagnostic plots
# ==============================================================

set.seed(1234)

# --- Data Split ---
train_data <- splits$trainA
test_data  <- splits$testA
# train_data <- splits$trainB
# test_data <- splits$testB

# --- Settings ---
PCA_VAR_THRESHOLD <- 0.75
env_variables <- c(
  "temperature","wind_speed","sunshine_minutes","global_radiation",
  "humidity_percent","fog","rain","snow","thunder","ice",
  "lux","co2","tempC","humidity","sound"
)
MAKE_BIPLOT <- TRUE

# --- Prepare Matrices ---
available_env <- intersect(env_variables, names(train_data))
stopifnot(length(available_env) > 0)

X_train <- as.matrix(train_data[, ..available_env])
X_test  <- as.matrix(test_data[,  ..available_env])

for (i in seq_len(ncol(X_train))) {
  mu <- mean(X_train[, i], na.rm = TRUE)
  X_train[is.na(X_train[, i]), i] <- mu
  X_test [is.na(X_test [, i]), i] <- mu
}

# --- Fit PCA (Train only) ---
pca_fit <- prcomp(X_train, center = TRUE, scale. = TRUE)
pca_var <- pca_fit$sdev^2
pca_ratio <- pca_var / sum(pca_var)
pca_cumratio <- cumsum(pca_ratio)
pca_n_ret <- which(pca_cumratio >= PCA_VAR_THRESHOLD)[1]
if (is.na(pca_n_ret)) pca_n_ret <- ncol(X_train)

cat(sprintf("Variables used (%d): %s\n", length(available_env), paste(available_env, collapse = ", ")))
cat(sprintf("Retained Components: %d (%.2f%% cumulative variance ≥ %.0f%%)\n",
            pca_n_ret, 100*pca_cumratio[pca_n_ret], 100*PCA_VAR_THRESHOLD))

# --- Variance Explained ---
dt_pca_var <- data.table(
  PC = paste0("PC", seq_along(pca_ratio)),
  StdDev = round(pca_fit$sdev, 6),
  Var = round(pca_var, 6),
  Var_Explained = round(pca_ratio, 6),
  Cum_Var_Explained = round(pca_cumratio, 6)
)
print(dt_pca_var)

# --- Loadings (retained PCs) ---
loadings_ret <- pca_fit$rotation[, 1:pca_n_ret, drop = FALSE]
dt_loadings <- as.data.table(loadings_ret, keep.rownames = "Variable")
cat("\n--- Loadings (first ", pca_n_ret, " PCs) ---\n", sep = "")
print(dt_loadings)

# --- Variable Contributions per PC ---
contrib <- sweep(loadings_ret^2, 2, colSums(loadings_ret^2), "/") * 100
dt_contrib <- as.data.table(contrib, keep.rownames = "Variable")
cat("\n--- Variable Contributions per PC (% of PC variance) ---\n")
print(dt_contrib)

# --- Communalities (variance explained per variable) ---
communalities <- rowSums(loadings_ret^2)
dt_communalities <- data.table(
  Variable = rownames(loadings_ret),
  Communality = round(communalities, 6)
)
setorder(dt_communalities, -Communality)
cat("\n--- Communalities ---\n")
print(dt_communalities)

# --- Correlations (Variable ↔ PC) ---
corr_mat <- sweep(loadings_ret, 2, pca_fit$sdev[1:pca_n_ret], "*")
dt_corr <- as.data.table(corr_mat, keep.rownames = "Variable")
cat("\n--- Correlations: Variables vs Retained PCs ---\n")
print(dt_corr)

# --- Scree + Cumulative Plot ---
dv <- data.table(PC = factor(seq_along(pca_ratio)), VarExplained = pca_ratio, CumVar = pca_cumratio)
p_scree <- ggplot(dv, aes(x = PC)) +
  geom_col(aes(y = VarExplained), fill = "grey") +
  geom_line(aes(y = CumVar, group = 1), color = "black") +
  geom_point(aes(y = CumVar), color = "black", size = 2) +
  labs(x = "Principal Component", y = "Variance Explained") +
  theme_bw(base_size = 11)
print(p_scree)
