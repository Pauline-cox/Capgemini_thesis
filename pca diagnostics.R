# ==============================================================
# PCA DIAGNOSTICS — OUTDOOR (decide #PCs + loadings etc.)
# ==============================================================

library(data.table)
library(ggplot2)

set.seed(1234)

# 0) Pick period -------------------------------------------------
train_data <- splits$trainA
test_data  <- splits$testA
# train_data <- splits$trainB; test_data <- splits$testB  # (switch if needed)

# 1) Settings ----------------------------------------------------
PCA_VAR_THRESHOLD <- 0.90
env_outdoor <- c("temperature","wind_speed","sunshine_minutes",
                 "global_radiation","humidity_percent","fog",
                 "rain","snow","thunder","ice")
MAKE_BIPLOT <- TRUE

# 2) Build matrices (train means for NA imputation) -------------
available_env <- intersect(env_outdoor, names(train_data))
stopifnot(length(available_env) > 0)

X_train <- as.matrix(train_data[, ..available_env])
X_test  <- as.matrix(test_data[,  ..available_env])

for (i in seq_len(ncol(X_train))) {
  mu <- mean(X_train[, i], na.rm = TRUE)
  X_train[is.na(X_train[, i]), i] <- mu
  X_test [is.na(X_test [, i]), i] <- mu
}

# 3) Fit PCA on TRAIN (center+scale) ----------------------------
pca_fit <- prcomp(X_train, center = TRUE, scale. = TRUE)

# 4) Variance explained & retained components -------------------
pca_var      <- pca_fit$sdev^2
pca_ratio    <- pca_var / sum(pca_var)
pca_cumratio <- cumsum(pca_ratio)
pca_n_ret    <- which(pca_cumratio >= PCA_VAR_THRESHOLD)[1]
if (is.na(pca_n_ret)) pca_n_ret <- ncol(X_train)

cat("\n================ PCA REPORT (OUTDOOR) ================\n")
cat(sprintf("Variables used (%d): %s\n", length(available_env), paste(available_env, collapse = ", ")))
cat(sprintf("Centering: %s | Scaling: %s\n", TRUE, TRUE))
cat(sprintf("Retained Components: %d (%.2f%% cumulative variance ≥ %.0f%%)\n",
            pca_n_ret, 100*pca_cumratio[pca_n_ret], 100*PCA_VAR_THRESHOLD))

# 5) Variance explained table -----------------------------------
dt_pca_var <- data.table(
  PC = paste0("PC", seq_along(pca_ratio)),
  StdDev = round(pca_fit$sdev, 6),
  Var = round(pca_var, 6),
  Var_Explained = round(pca_ratio, 6),
  Cum_Var_Explained = round(pca_cumratio, 6)
)
print(dt_pca_var)

# 6) Loadings (first pca_n_ret PCs) -----------------------------
loadings_ret <- pca_fit$rotation[, 1:pca_n_ret, drop = FALSE]
colnames(loadings_ret) <- paste0("PC", 1:pca_n_ret)
dt_loadings <- as.data.table(loadings_ret, keep.rownames = "Variable")
cat("\n--- Loadings (first ", pca_n_ret, " PCs) ---\n", sep = "")
print(dt_loadings)

# 7) Contributions per PC (%) -----------------------------------
contrib <- sweep(loadings_ret^2, 2, colSums(loadings_ret^2), "/") * 100
dt_contrib <- as.data.table(contrib, keep.rownames = "Variable")
cat("\n--- Variable Contributions per PC (% of PC variance) ---\n")
print(dt_contrib)

# 8) Communalities (variance explained per variable) ------------
communalities <- rowSums(loadings_ret^2)
dt_communalities <- data.table(Variable = rownames(loadings_ret),
                               Communality = round(communalities, 6))
setorder(dt_communalities, -Communality)
cat("\n--- Communalities ---\n")
print(dt_communalities)

# 9) Correlations (vars ↔ PCs): loading * sdev ------------------
corr_mat <- sweep(loadings_ret, 2, pca_fit$sdev[1:pca_n_ret], "*")
dt_corr <- as.data.table(corr_mat, keep.rownames = "Variable")
cat("\n--- Correlations: Variables vs Retained PCs ---\n")
print(dt_corr)

# 10) Top 5 absolute loadings per PC ----------------------------
cat("\n--- Top 5 absolute loadings per PC ---\n")
for (j in seq_len(pca_n_ret)) {
  ord <- order(abs(loadings_ret[, j]), decreasing = TRUE)
  top_idx <- ord[1:min(5, nrow(loadings_ret))]
  top_dt <- data.table(
    Variable = rownames(loadings_ret)[top_idx],
    Loading  = round(loadings_ret[top_idx, j], 6)
  )
  cat(sprintf("\nPC%02d:\n", j)); print(top_dt)
}

# 11) Scree + cumulative plot -----------------------------------
dv <- data.table(
  PC = factor(seq_along(pca_ratio), levels = seq_along(pca_ratio)),
  VarExplained = pca_ratio, CumVar = pca_cumratio
)
p_scree <- ggplot(dv, aes(x = PC, y = VarExplained)) +
  geom_col() +
  geom_point(aes(y = CumVar), size = 2) +
  geom_line(aes(y = CumVar, group = 1)) +
  labs(x = NULL, y = "Proportion of Variance (bars) / Cumulative (line)") +
  theme_bw(base_size = 11) + theme(legend.position = "none")
print(p_scree)

# 12) Loadings heatmap ------------------------------------------
dt_heat <- melt(as.data.table(loadings_ret, keep.rownames = "Variable"),
                id.vars = "Variable", variable.name = "PC", value.name = "Loading")
p_heat <- ggplot(dt_heat, aes(x = PC, y = Variable, fill = Loading)) +
  geom_tile() +
  scale_fill_gradient2() +
  labs(x = NULL, y = NULL, fill = "Loading", title = "Loadings Heatmap (retained PCs)") +
  theme_bw(base_size = 11)
print(p_heat)

# 13) Optional biplot -------------------------------------------
if (MAKE_BIPLOT && pca_n_ret >= 2) {
  arrows_dt <- data.table(
    xend = loadings_ret[,1] * pca_fit$sdev[1] * sqrt(nrow(X_train)),
    yend = loadings_ret[,2] * pca_fit$sdev[2] * sqrt(nrow(X_train)),
    Variable = rownames(loadings_ret)
  )
  pc12 <- data.table(PC1 = predict(pca_fit, X_train)[,1],
                     PC2 = predict(pca_fit, X_train)[,2])
  p_biplot <- ggplot() +
    geom_point(data = pc12, aes(PC1, PC2), alpha = 0.2) +
    geom_segment(data = arrows_dt,
                 aes(x = 0, y = 0, xend = xend, yend = yend),
                 arrow = arrow(length = unit(0.02, "npc"))) +
    geom_text(data = arrows_dt, aes(x = xend, y = yend, label = Variable),
              hjust = 0, vjust = 0, size = 3) +
    labs(title = "PCA Biplot (Train, outdoor)", x = "PC1", y = "PC2") +
    theme_bw(base_size = 11)
  print(p_biplot)
}

cat("\n=============== END PCA REPORT ===============\n")

