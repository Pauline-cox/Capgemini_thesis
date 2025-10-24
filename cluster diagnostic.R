# ==============================================================
# CLUSTERING DIAGNOSTICS — OUTDOOR (decide K + profiles)
# ==============================================================

library(data.table)
library(ggplot2)
library(cluster)

set.seed(1234)

# 0) Pick period -------------------------------------------------
train_data <- splits$trainA
test_data  <- splits$testA
# train_data <- splits$trainB; test_data <- splits$testB  # (switch if needed)

# 1) Settings ----------------------------------------------------
env_outdoor <- c("temperature","wind_speed","sunshine_minutes",
                 "global_radiation","humidity_percent","fog",
                 "rain","snow","thunder","ice")
K_RANGE <- 2:8
MAKE_PCA_SCATTER <- TRUE  # visualization only

# 2) Build env matrices; impute NAs with TRAIN means ------------
available_env <- intersect(env_outdoor, names(train_data))
stopifnot(length(available_env) > 0)

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

# 4) Model selection: elbow + silhouette ------------------------
wss <- numeric(length(K_RANGE))
avg_sil <- numeric(length(K_RANGE))
dist_tr <- dist(Ztr, method = "euclidean")

for (i in seq_along(K_RANGE)) {
  k <- K_RANGE[i]
  fit <- kmeans(Ztr, centers = k, nstart = 50, iter.max = 100)
  wss[i] <- fit$tot.withinss
  sil <- silhouette(fit$cluster, dist_tr)
  avg_sil[i] <- mean(sil[, "sil_width"])
}

sel_tbl <- data.table(k = K_RANGE, WSS = wss, Avg_Silhouette = round(avg_sil, 4))
print(sel_tbl)

# 5) Choose K (peak silhouette) ---------------------------------
FINAL_K <- sel_tbl$k[which.max(sel_tbl$Avg_Silhouette)]
cat(sprintf("\n[Selection] FINAL_K = %d | Avg silhouette = %.3f\n",
            FINAL_K, sel_tbl[k == FINAL_K, Avg_Silhouette]))

# 6) Final fit on TRAIN + assign TEST ---------------------------
set.seed(1234)
km <- kmeans(Ztr, centers = FINAL_K, nstart = 100, iter.max = 200)
train_clusters <- km$cluster
centers <- km$centers

dist_mat <- sapply(1:nrow(centers), function(ci) rowSums((Zte - centers[ci,])^2))
test_clusters <- max.col(-dist_mat)

# 7) Cluster sizes & proportions --------------------------------
sizes <- table(train_clusters)
size_tbl <- data.table(
  Cluster  = as.integer(names(sizes)),
  Train_N  = as.integer(sizes),
  Train_pct = round(100 * as.integer(sizes) / sum(sizes), 2)
)
print(size_tbl)

# 8) Cluster profiles (z-scores) --------------------------------
Z_dt <- as.data.table(Ztr)
Z_dt[, .cluster := train_clusters]
profile_z <- Z_dt[, lapply(.SD, mean), by = .cluster, .SDcols = available_env]
setorder(profile_z, .cluster)
setnames(profile_z, ".cluster", "Cluster")
print(profile_z)

# 9) Cluster profiles (original units) --------------------------
X_dt <- as.data.table(Xtr)
X_dt[, .cluster := train_clusters]
profile_raw <- X_dt[, lapply(.SD, mean), by = .cluster, .SDcols = available_env]
setorder(profile_raw, .cluster)
setnames(profile_raw, ".cluster", "Cluster")
print(profile_raw)

# 10) Center distances (z-space) --------------------------------
center_dist <- as.matrix(dist(centers))
print(round(center_dist, 3))

# 11) Silhouette details for FINAL_K ----------------------------
sil_final <- silhouette(train_clusters, dist_tr)
sil_tbl <- data.table(Cluster = sil_final[, "cluster"],
                      Sil_Width = sil_final[, "sil_width"])
sil_summary <- sil_tbl[, .(Avg_Sil = mean(Sil_Width),
                           Q25 = quantile(Sil_Width, 0.25),
                           Q50 = median(Sil_Width),
                           Q75 = quantile(Sil_Width, 0.75)), by = Cluster]
print(sil_summary)

# 12) Plots ------------------------------------------------------
# A) Elbow (WSS)
p_elbow <- ggplot(sel_tbl, aes(k, WSS)) +
  geom_point() + geom_line() +
  labs(x = "k", y = "Total within-cluster SS", title = "Elbow Plot (WSS vs k)") +
  theme_bw(base_size = 11)
print(p_elbow)

# B) Silhouette vs k
p_sil <- ggplot(sel_tbl, aes(k, Avg_Silhouette)) +
  geom_point() + geom_line() +
  geom_vline(xintercept = FINAL_K, linetype = 2) +
  labs(x = "k", y = "Average silhouette width", title = "Silhouette Curve") +
  theme_bw(base_size = 11)
print(p_sil)

# C) Silhouette histogram for FINAL_K
sil_df <- data.frame(sil_final)
p_sil_hist <- ggplot(sil_df, aes(sil_width)) +
  geom_histogram(bins = 40) +
  geom_vline(xintercept = mean(sil_df$sil_width), linetype = 2) +
  labs(x = "Silhouette width", y = "Count", title = sprintf("Silhouette Widths (k = %d)", FINAL_K)) +
  theme_bw(base_size = 11)
print(p_sil_hist)

# D) Heatmap: cluster profiles in z-scores
prof_m <- as.matrix(profile_z[, ..available_env])
rownames(prof_m) <- paste0("C", profile_z$Cluster)
dt_heat <- as.data.table(prof_m, keep.rownames = "Cluster")
dt_heat <- melt(dt_heat, id.vars = "Cluster", variable.name = "Variable", value.name = "Z_Mean")
p_heat <- ggplot(dt_heat, aes(x = Cluster, y = Variable, fill = Z_Mean)) +
  geom_tile() +
  scale_fill_gradient2() +
  labs(x = NULL, y = NULL, fill = "z-mean", title = "Cluster Profiles (standardized)") +
  theme_bw(base_size = 11)
print(p_heat)

# E) Optional 2D scatter (PCA only for visualization)
if (MAKE_PCA_SCATTER) {
  pc <- prcomp(Ztr, center = FALSE, scale. = FALSE)
  pc_dt <- data.table(PC1 = pc$x[,1], PC2 = pc$x[,2], Cluster = factor(train_clusters))
  p_scatter <- ggplot(pc_dt, aes(PC1, PC2, color = Cluster)) +
    geom_point(alpha = 0.35) +
    labs(title = "Clusters projected on first two PCs (viz only)") +
    theme_bw(base_size = 11) + theme(legend.position = "bottom")
  print(p_scatter)
}

cat("\n=============== END CLUSTERING REPORT ===============\n")

