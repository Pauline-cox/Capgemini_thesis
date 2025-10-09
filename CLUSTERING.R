env_clustering <- function(model_data, k = 2) {
  suppressPackageStartupMessages({library(data.table); library(cluster)})
  cat("\n=== ENVIRONMENTAL CLUSTERING (k =", k, ") ===\n")
  
  env_vars <- c("tempC","humidity","co2","sound","lux","temperature","wind_speed",
                "sunshine_minutes","global_radiation","humidity_percent",
                "fog","rain","snow","thunder","ice")
  env_vars <- intersect(env_vars, names(model_data))
  if (length(env_vars) == 0) stop("No environmental variables found in model_data.")
  
  X <- model_data[, ..env_vars]
  X <- as.data.table(lapply(X, function(col) { col[is.na(col)] <- mean(col, na.rm = TRUE); col }))
  X_scaled <- scale(X)
  
  set.seed(42)
  km <- kmeans(X_scaled, centers = k, nstart = 25)
  model_data[, env_cluster := factor(km$cluster)]
  
  centers_scaled   <- km$centers
  centers_unscaled <- centers_scaled * attr(X_scaled, "scaled:scale") + attr(X_scaled, "scaled:center")
  
  cat("Clustering complete | Cluster sizes:\n")
  print(table(model_data$env_cluster))
  
  list(
    data = model_data,
    centers_scaled = centers_scaled,
    centers_unscaled = centers_unscaled,
    env_vars = env_vars
  )
}
