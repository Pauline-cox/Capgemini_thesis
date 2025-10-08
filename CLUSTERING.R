# ==============================================================================
# ENVIRONMENTAL CLUSTERING FUNCTION (VERBOSE + DESCRIPTIVE STATS)
# ==============================================================================

env_clustering <- function(model_data, k = 2) {
  suppressPackageStartupMessages({
    library(data.table)
    library(cluster)
    library(factoextra) # optional: for silhouette and visualization metrics
  })
  
  cat("\n=== ENVIRONMENTAL CLUSTERING (k =", k, ") ===\n")
  
  # ---------------------------------------------------------------------------
  # 1️⃣ Select Environmental Variables
  # ---------------------------------------------------------------------------
  env_vars <- c(
    "tempC", "humidity", "co2", "sound", "lux",
    "temperature", "wind_speed", "sunshine_minutes",
    "global_radiation", "humidity_percent",
    "fog", "rain", "snow", "thunder", "ice"
  )
  env_vars <- intersect(env_vars, names(model_data))
  
  if (length(env_vars) == 0) {
    stop("No environmental variables found in model_data.")
  }
  
  cat("Variables used for clustering (", length(env_vars), "):\n  ",
      paste(env_vars, collapse = ", "), "\n", sep = "")
  
  X <- model_data[, ..env_vars]
  
  # Remove columns with near-zero variance
  nzv <- which(apply(X, 2, sd, na.rm = TRUE) < 1e-6)
  if (length(nzv) > 0) {
    cat("Removed near-zero variance variables:", paste(names(X)[nzv], collapse = ", "), "\n")
    X <- X[, -nzv, drop = FALSE]
    env_vars <- setdiff(env_vars, names(X)[nzv])
  }
  
  # Handle missing values
  for (v in env_vars) {
    if (anyNA(X[[v]])) {
      X[[v]][is.na(X[[v]])] <- mean(X[[v]], na.rm = TRUE)
    }
  }
  
  X_scaled <- scale(X)
  
  # ---------------------------------------------------------------------------
  # 2️⃣ Run K-Means Clustering
  # ---------------------------------------------------------------------------
  set.seed(42)
  km <- kmeans(X_scaled, centers = k, nstart = 25)
  model_data[, env_cluster := factor(km$cluster)]
  
  cat("✓ Environmental clustering complete\n")
  cat("  • Total observations:", nrow(model_data), "\n")
  cat("  • Cluster sizes:\n")
  print(table(model_data$env_cluster))
  
  # ---------------------------------------------------------------------------
  # 3️⃣ Cluster Centroids (means of each variable by cluster)
  # ---------------------------------------------------------------------------
  centers_scaled <- km$centers
  centers_unscaled <- centers_scaled * attr(X_scaled, "scaled:scale") + attr(X_scaled, "scaled:center")
  
  cat("\n--- Cluster Centers (Unscaled Environmental Values) ---\n")
  print(round(centers_unscaled, 2))
  
  # ---------------------------------------------------------------------------
  # 4️⃣ Within-cluster variance, between-cluster ratio, silhouette (optional)
  # ---------------------------------------------------------------------------
  tot_withinss <- km$tot.withinss
  betweenss   <- km$betweenss
  ratio_bw     <- round(betweenss / (betweenss + tot_withinss), 3)
  cat("\n--- Cluster Fit Statistics ---\n")
  cat(sprintf("  • Total within-cluster SS: %.2f\n", tot_withinss))
  cat(sprintf("  • Between-cluster SS: %.2f\n", betweenss))
  cat(sprintf("  • Between/Total SS ratio: %.3f (higher = better separation)\n", ratio_bw))
  
  # Optional: silhouette measure of cluster separation
  sil <- silhouette(km$cluster, dist(X_scaled))
  mean_sil <- mean(sil[, 3])
  cat(sprintf("  • Average silhouette width: %.3f\n", mean_sil))
  
  # ---------------------------------------------------------------------------
  # 5️⃣ Variable Contribution by Cluster (ANOVA-style importance)
  # ---------------------------------------------------------------------------
  cat("\n--- Variable Importance by Cluster (ANOVA F-statistic) ---\n")
  var_imp <- sapply(env_vars, function(v) {
    summary(aov(X_scaled[, v] ~ km$cluster))[[1]][["F value"]][1]
  })
  var_imp <- sort(var_imp, decreasing = TRUE)
  print(round(var_imp, 3))
  
  cat("\nTop 5 contributing variables:\n")
  print(head(var_imp, 5))
  
  # ---------------------------------------------------------------------------
  # 6️⃣ Return enriched result
  # ---------------------------------------------------------------------------
  list(
    data = model_data,
    centers_scaled = centers_scaled,
    centers_unscaled = centers_unscaled,
    env_vars = env_vars,
    within_ss = tot_withinss,
    between_ss = betweenss,
    between_total_ratio = ratio_bw,
    silhouette_mean = mean_sil,
    var_importance = var_imp
  )
}
