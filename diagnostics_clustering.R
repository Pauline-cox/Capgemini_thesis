# ==============================================================
# Author: Pauline Cox
# Script: diagnostics_cluster_outdoor.R
#
# Description: Performs clustering diagnostics on 
# environmental variables to determine the optimal number of clusters 
# (K) based on silhouette width and elbow criteria. Includes 
# visualization of WSS and silhouette metrics 
# Provides a summary of selected clusters for 
# interpretability and later SARIMAX integration.
#
# Input: 
#   - Preprocessed and feature-engineered dataset (model_data)
#   - Split data (splits$trainA / splits$trainB)
#
# Output: 
#   - Cluster summary tables, diagnostics, and visualizations
# ==============================================================

set.seed(1234)

# --- Data Split ---
splits <- split_periods(model_data)
train_data <- splits$trainA
test_data  <- splits$testA
# train_data <- splits$trainB
# test_data  <- splits$testB

# --- Settings ---
env_variables <- c(
  "temperature","wind_speed","sunshine_minutes","global_radiation",
  "humidity_percent","fog","rain","snow","thunder","ice",
  "lux","co2","tempC","humidity","sound"
)

temporal_vars <- c("business_hours","hour_sin","hour_cos","dow_cos","holiday","dst")
occ_var <- "total_occupancy"

K_RANGE <- 2:8
MAKE_PCA_SCATTER <- TRUE

# --- Prepare Environmental Matrices ---
available_env <- intersect(env_variables, names(train_data))
stopifnot(length(available_env) > 0)

Xtr <- as.matrix(train_data[, ..available_env])
Xte <- as.matrix(test_data[,  ..available_env])

# Impute missing values using training means
for (j in seq_len(ncol(Xtr))) {
  mu <- mean(Xtr[, j], na.rm = TRUE)
  Xtr[is.na(Xtr[, j]), j] <- mu
  Xte[is.na(Xte[, j]), j] <- mu
}

# --- Standardize with Training Statistics ---
cvec <- colMeans(Xtr)
svec <- apply(Xtr, 2, sd); svec[svec == 0] <- 1
Ztr <- scale(Xtr, center = cvec, scale = svec)
Zte <- scale(Xte, center = cvec, scale = svec)

# --- Model Selection: Elbow + Silhouette Analysis ---
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

# --- Select Optimal K ---
FINAL_K <- sel_tbl$k[which.max(sel_tbl$Avg_Silhouette)]
cat(sprintf("\n FINAL_K = %d | Avg silhouette = %.3f\n",
            FINAL_K, sel_tbl[k == FINAL_K, Avg_Silhouette]))

# --- Final K-Means Fit and Assignment ---
set.seed(1234)
km <- kmeans(Ztr, centers = FINAL_K, nstart = 100, iter.max = 200)
train_clusters <- km$cluster
centers <- km$centers

# Assign clusters to test set (nearest centroid)
dist_mat <- sapply(1:nrow(centers), function(ci) rowSums((Zte - centers[ci,])^2))
test_clusters <- max.col(-dist_mat)

# --- Cluster Sizes & Proportions ---
sizes <- table(train_clusters)
size_tbl <- data.table(
  Cluster  = as.integer(names(sizes)),
  Train_N  = as.integer(sizes),
  Train_pct = round(100 * as.integer(sizes) / sum(sizes), 2)
)
print(size_tbl)

# --- Silhouette Statistics for Selected K ---
sil_final <- silhouette(train_clusters, dist_tr)
sil_tbl <- data.table(Cluster = sil_final[, "cluster"], Sil_Width = sil_final[, "sil_width"])
sil_summary <- sil_tbl[, .(
  Avg_Sil = mean(Sil_Width),
  Q25 = quantile(Sil_Width, 0.25),
  Q50 = median(Sil_Width),
  Q75 = quantile(Sil_Width, 0.75)
), by = Cluster]
print(sil_summary)

# --- Visualization: Elbow Plot ---
p_elbow <- ggplot(sel_tbl, aes(k, WSS)) +
  geom_point() + geom_line() +
  labs(x = "k", y = "Total Within-Cluster SS", title = "Elbow Plot") +
  theme_bw(base_size = 11)
print(p_elbow)

# --- Visualization: Silhouette vs K ---
p_sil <- ggplot(sel_tbl, aes(k, Avg_Silhouette)) +
  geom_point() + geom_line() +
  geom_vline(xintercept = FINAL_K, linetype = 2, color = "black") +
  labs(x = "k", y = "Average Silhouette Width", title = "Silhouette Width by k") +
  theme_bw(base_size = 11)
print(p_sil)

# ---  PCA Visualization (2D Scatter) ---
pc <- prcomp(Ztr, center = FALSE, scale. = FALSE)
pc_dt <- data.table(PC1 = pc$x[,1], PC2 = pc$x[,2], Cluster = factor(train_clusters))
p_scatter <- ggplot(pc_dt, aes(PC1, PC2, color = Cluster)) +
  geom_point(alpha = 0.35) +
  labs(title = "Clusters Projected on First Two PCs (Visualization Only)") +
  theme_bw(base_size = 11) + theme(legend.position = "bottom")
print(p_scatter)
