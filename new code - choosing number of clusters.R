# ============================================================================
# OPTIMAL NUMBER OF CLUSTERS SELECTION FOR ENVIRONMENTAL DATA
# ============================================================================

library(data.table)
library(ggplot2)
library(cluster)
library(factoextra)

# Function to determine optimal number of clusters
determine_optimal_clusters <- function(model_data, k_max = 10, method = "all") {
  
  cat("\n=== DETERMINING OPTIMAL NUMBER OF CLUSTERS ===\n\n")
  
  # Define environmental variables
  env_vars <- c("tempC","humidity","co2","sound","lux","temperature","wind_speed",
                "sunshine_minutes","global_radiation","humidity_percent",
                "fog","rain","snow","thunder","ice")
  env_vars <- intersect(env_vars, names(model_data))
  
  if (length(env_vars) == 0) {
    stop("No environmental variables found in model_data.")
  }
  
  # Prepare and scale data
  X <- model_data[, ..env_vars]
  X <- as.data.table(lapply(X, function(col) {
    col[is.na(col)] <- mean(col, na.rm = TRUE)
    col
  }))
  X_scaled <- scale(X)
  
  # Initialize results storage
  results <- list()
  
  # ======================== METHOD 1: ELBOW METHOD ========================
  if (method %in% c("all", "elbow")) {
    cat("1. Computing Within-Cluster Sum of Squares (Elbow Method)...\n")
    
    wss <- sapply(1:k_max, function(k) {
      set.seed(42)
      km <- kmeans(X_scaled, centers = k, nstart = 25)
      km$tot.withinss
    })
    
    # Calculate percentage of variance explained
    variance_explained <- (1 - wss / wss[1]) * 100
    
    results$wss <- data.table(
      k = 1:k_max,
      WSS = wss,
      Variance_Explained = variance_explained
    )
    
    cat("   WSS values:\n")
    print(results$wss)
    cat("\n")
  }
  
  # ==================== METHOD 2: SILHOUETTE METHOD ====================
  if (method %in% c("all", "silhouette")) {
    cat("2. Computing Average Silhouette Width...\n")
    
    silhouette_avg <- sapply(2:k_max, function(k) {
      set.seed(42)
      km <- kmeans(X_scaled, centers = k, nstart = 25)
      sil <- silhouette(km$cluster, dist(X_scaled))
      mean(sil[, 3])
    })
    
    results$silhouette <- data.table(
      k = 2:k_max,
      Avg_Silhouette = silhouette_avg
    )
    
    cat("   Average Silhouette values:\n")
    print(results$silhouette)
    cat("\n")
    
    optimal_sil <- results$silhouette[which.max(Avg_Silhouette), k]
    cat(sprintf("   → Optimal k by Silhouette: %d (Avg Sil = %.3f)\n\n", 
                optimal_sil, max(silhouette_avg)))
  }
  
  # ==================== METHOD 3: GAP STATISTIC ====================
  if (method %in% c("all", "gap")) {
    cat("3. Computing Gap Statistic (this may take a while)...\n")
    
    set.seed(42)
    gap_stat <- clusGap(X_scaled, FUN = kmeans, nstart = 25, K.max = k_max, B = 50)
    
    results$gap <- data.table(
      k = 1:k_max,
      Gap = gap_stat$Tab[, "gap"],
      SE = gap_stat$Tab[, "SE.sim"]
    )
    
    cat("   Gap Statistic values:\n")
    print(results$gap)
    cat("\n")
    
    # Find optimal k using "firstmax" method
    optimal_gap <- maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"], method = "firstmax")
    cat(sprintf("   → Optimal k by Gap Statistic: %d\n\n", optimal_gap))
  }
  
  # ==================== METHOD 4: CALINSKI-HARABASZ INDEX ====================
  if (method %in% c("all", "ch")) {
    cat("4. Computing Calinski-Harabasz Index...\n")
    
    ch_index <- sapply(2:k_max, function(k) {
      set.seed(42)
      km <- kmeans(X_scaled, centers = k, nstart = 25)
      
      # Between-cluster sum of squares
      bss <- sum(sapply(1:k, function(i) {
        cluster_points <- X_scaled[km$cluster == i, , drop = FALSE]
        nrow(cluster_points) * sum((colMeans(cluster_points) - colMeans(X_scaled))^2)
      }))
      
      # Within-cluster sum of squares
      wss <- km$tot.withinss
      
      # CH index
      n <- nrow(X_scaled)
      ch <- (bss / (k - 1)) / (wss / (n - k))
      return(ch)
    })
    
    results$ch <- data.table(
      k = 2:k_max,
      CH_Index = ch_index
    )
    
    cat("   Calinski-Harabasz Index values:\n")
    print(results$ch)
    cat("\n")
    
    optimal_ch <- results$ch[which.max(CH_Index), k]
    cat(sprintf("   → Optimal k by CH Index: %d (CH = %.2f)\n\n", 
                optimal_ch, max(ch_index)))
  }
  
  # ==================== VISUALIZATIONS ====================
  cat("5. Creating visualizations...\n\n")
  
  plots <- list()
  
  # Plot 1: Elbow plot
  if (!is.null(results$wss)) {
    plots$elbow <- ggplot(results$wss, aes(x = k, y = WSS)) +
      geom_line(color = "steelblue", size = 1) +
      geom_point(color = "steelblue", size = 3) +
      geom_vline(xintercept = 3, linetype = "dashed", color = "red", alpha = 0.5) +
      labs(title = "Elbow Method",
           subtitle = "Look for the 'elbow' where WSS starts to flatten",
           x = "Number of Clusters (k)",
           y = "Within-Cluster Sum of Squares") +
      theme_minimal(base_size = 12) +
      scale_x_continuous(breaks = 1:k_max)
    
    print(plots$elbow)
  }
  
  # Plot 2: Silhouette plot
  if (!is.null(results$silhouette)) {
    optimal_sil <- results$silhouette[which.max(Avg_Silhouette), k]
    plots$silhouette <- ggplot(results$silhouette, aes(x = k, y = Avg_Silhouette)) +
      geom_line(color = "darkgreen", size = 1) +
      geom_point(color = "darkgreen", size = 3) +
      geom_vline(xintercept = optimal_sil, linetype = "dashed", color = "red") +
      labs(title = "Silhouette Method",
           subtitle = sprintf("Optimal k = %d (highest average silhouette)", optimal_sil),
           x = "Number of Clusters (k)",
           y = "Average Silhouette Width") +
      theme_minimal(base_size = 12) +
      scale_x_continuous(breaks = 2:k_max)
    
    print(plots$silhouette)
  }
  
  # Plot 3: Gap statistic
  if (!is.null(results$gap)) {
    plots$gap <- ggplot(results$gap, aes(x = k, y = Gap)) +
      geom_line(color = "purple", size = 1) +
      geom_point(color = "purple", size = 3) +
      geom_errorbar(aes(ymin = Gap - SE, ymax = Gap + SE), width = 0.2, alpha = 0.5) +
      geom_vline(xintercept = optimal_gap, linetype = "dashed", color = "red") +
      labs(title = "Gap Statistic",
           subtitle = sprintf("Optimal k = %d (first local maximum)", optimal_gap),
           x = "Number of Clusters (k)",
           y = "Gap Statistic") +
      theme_minimal(base_size = 12) +
      scale_x_continuous(breaks = 1:k_max)
    
    print(plots$gap)
  }
  
  # Plot 4: CH Index
  if (!is.null(results$ch)) {
    optimal_ch <- results$ch[which.max(CH_Index), k]
    plots$ch <- ggplot(results$ch, aes(x = k, y = CH_Index)) +
      geom_line(color = "orange", size = 1) +
      geom_point(color = "orange", size = 3) +
      geom_vline(xintercept = optimal_ch, linetype = "dashed", color = "red") +
      labs(title = "Calinski-Harabasz Index",
           subtitle = sprintf("Optimal k = %d (maximum CH index)", optimal_ch),
           x = "Number of Clusters (k)",
           y = "CH Index (higher is better)") +
      theme_minimal(base_size = 12) +
      scale_x_continuous(breaks = 2:k_max)
    
    print(plots$ch)
  }
  
  # ==================== RECOMMENDATIONS ====================
  cat("\n=== RECOMMENDATIONS ===\n")
  
  recommendations <- c()
  if (!is.null(results$silhouette)) {
    recommendations <- c(recommendations, 
                         sprintf("Silhouette method: k = %d", 
                                 results$silhouette[which.max(Avg_Silhouette), k]))
  }
  if (!is.null(results$gap)) {
    recommendations <- c(recommendations, 
                         sprintf("Gap statistic: k = %d", optimal_gap))
  }
  if (!is.null(results$ch)) {
    recommendations <- c(recommendations, 
                         sprintf("Calinski-Harabasz: k = %d", 
                                 results$ch[which.max(CH_Index), k]))
  }
  
  cat(paste(recommendations, collapse = "\n"))
  cat("\n\n")
  
  cat("INTERPRETATION GUIDE:\n")
  cat("• Elbow Method: Look for the 'elbow' where adding clusters gives diminishing returns\n")
  cat("• Silhouette: Higher is better (range: -1 to 1). Values > 0.5 indicate good clustering\n")
  cat("• Gap Statistic: Choose k where Gap(k) ≥ Gap(k+1) - SE(k+1)\n")
  cat("• CH Index: Higher is better. Peak indicates optimal separation\n\n")
  
  cat("FOR YOUR ENERGY DATA:\n")
  cat("• k=2-3: Simple (e.g., 'hot vs cold', 'day vs night vs weekend')\n")
  cat("• k=4-5: Moderate (e.g., seasonal + occupancy patterns)\n")
  cat("• k>5: Complex (may overfit or be hard to interpret)\n\n")
  
  cat("RECOMMENDED: Start with k=2 or k=3 for interpretability,\n")
  cat("then test if higher k improves forecast accuracy.\n\n")
  
  return(list(
    results = results,
    plots = plots,
    env_vars = env_vars,
    data_scaled = X_scaled
  ))
}

# ==================== EXAMPLE USAGE ====================

# Run cluster analysis
cluster_analysis <- determine_optimal_clusters(
  model_data = trainA,  # Use your training data
  k_max = 10,           # Test up to 10 clusters
  method = "all"        # Use all methods: "elbow", "silhouette", "gap", "ch", or "all"
)

# Access results
cluster_analysis$results$silhouette  # Silhouette scores
cluster_analysis$results$gap         # Gap statistics
cluster_analysis$plots$elbow         # Elbow plot

# ==================== ADDITIONAL: VISUALIZE ACTUAL CLUSTERS ====================

visualize_clusters <- function(model_data, k_chosen = 3) {
  
  env_vars <- c("tempC","humidity","co2","sound","lux","temperature","wind_speed",
                "sunshine_minutes","global_radiation","humidity_percent",
                "fog","rain","snow","thunder","ice")
  env_vars <- intersect(env_vars, names(model_data))
  
  X <- model_data[, ..env_vars]
  X <- as.data.table(lapply(X, function(col) {
    col[is.na(col)] <- mean(col, na.rm = TRUE)
    col
  }))
  X_scaled <- scale(X)
  
  # Fit k-means with chosen k
  set.seed(42)
  km <- kmeans(X_scaled, centers = k_chosen, nstart = 25)
  
  # PCA for visualization
  pca <- prcomp(X_scaled)
  pca_data <- data.table(
    PC1 = pca$x[, 1],
    PC2 = pca$x[, 2],
    Cluster = factor(km$cluster)
  )
  
  # Plot clusters in 2D
  p <- ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster)) +
    geom_point(alpha = 0.5, size = 1) +
    stat_ellipse(level = 0.95, size = 1) +
    labs(title = sprintf("Cluster Visualization (k = %d)", k_chosen),
         subtitle = sprintf("Variance explained: PC1=%.1f%%, PC2=%.1f%%",
                            summary(pca)$importance[2,1]*100,
                            summary(pca)$importance[2,2]*100),
         x = "First Principal Component",
         y = "Second Principal Component") +
    theme_minimal(base_size = 12) +
    scale_color_brewer(palette = "Set1")
  
  print(p)
  
  # Print cluster statistics
  cat("\n=== CLUSTER STATISTICS ===\n")
  model_data[, cluster := km$cluster]
  
  for (i in 1:k_chosen) {
    cat(sprintf("\nCluster %d (n=%d, %.1f%%):\n", 
                i, sum(km$cluster == i), 
                sum(km$cluster == i) / length(km$cluster) * 100))
    
    cluster_data <- X[km$cluster == i]
    means <- colMeans(cluster_data)
    cat("  Characteristics:\n")
    
    # Show top 3 distinctive features
    overall_means <- colMeans(X)
    deviations <- abs(means - overall_means) / apply(X, 2, sd)
    top_features <- names(sort(deviations, decreasing = TRUE)[1:3])
    
    for (feat in top_features) {
      cat(sprintf("    %s: %.2f (overall: %.2f)\n", 
                  feat, means[feat], overall_means[feat]))
    }
  }
  
  return(list(
    kmeans = km,
    pca = pca,
    plot = p
  ))
}

# Visualize chosen number of clusters
vis <- visualize_clusters(trainA, k_chosen = 3)