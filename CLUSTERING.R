env_clustering <- function(model_data, k = 2) {
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

# --- Function to assign clusters to test set ---
assign_env_clusters <- function(train_data, test_data, clus_obj) {
  vars <- intersect(clus_obj$env_vars, names(test_data))
  if (length(vars) == 0)
    stop("No matching environmental variables found in test_data.")
  
  # Select and clean numeric data
  clean_numeric <- function(dt, vars) {
    X <- copy(dt[, vars, with = FALSE])
    for (v in vars) {
      if (!is.numeric(X[[v]])) X[[v]] <- as.numeric(as.character(X[[v]]))
      if (anyNA(X[[v]])) X[[v]][is.na(X[[v]])] <- mean(X[[v]], na.rm = TRUE)
    }
    return(as.matrix(X))
  }
  
  X_test <- clean_numeric(test_data, vars)
  
  # Ensure centers are numeric
  centers <- as.matrix(clus_obj$centers_scaled)
  
  # Compute nearest cluster for each row
  cluster_ids <- apply(X_test, 1, function(x) {
    dists <- colSums((t(centers) - x)^2)
    which.min(dists)
  })
  
  factor(cluster_ids)
}
