# ===============================================================
# CORRELATION INSPECTION (linear, squared, cubic, interactions)
# ===============================================================

inspect_poly_corr <- function(model_data, target_col = "total_consumption_kWh",
                              top_n = 25, max_features = 12) {
  suppressPackageStartupMessages(library(data.table))
  setDT(model_data)
  
  y <- model_data[[target_col]]
  
  # keep only numeric predictors
  exclude_cols <- c("interval", target_col, "date")
  feats <- setdiff(names(model_data), exclude_cols)
  num_cols <- feats[sapply(model_data[, ..feats], is.numeric)]
  
  # limit for speed
  if (length(num_cols) > max_features) {
    cat(sprintf("Using first %d numeric features for inspection.\n", max_features))
    num_cols <- num_cols[1:max_features]
  }
  
  X <- model_data[, ..num_cols]
  results <- list()
  
  # ---- 1. Linear correlations ----
  c1 <- sapply(X, function(x) cor(x, y, use="complete.obs"))
  c1 <- data.table(term = names(c1), type = "linear", corr = c1)
  results[[1]] <- c1
  
  # ---- 2. Squared terms ----
  cat("\nComputing squared correlations...\n")
  c2 <- sapply(X, function(x) cor(x^2, y, use="complete.obs"))
  c2 <- data.table(term = paste0(names(c2), "^2"), type = "squared", corr = c2)
  results[[2]] <- c2
  
  # ---- 3. Cubic terms ----
  cat("Computing cubic correlations...\n")
  c3 <- sapply(X, function(x) cor(x^3, y, use="complete.obs"))
  c3 <- data.table(term = paste0(names(c3), "^3"), type = "cubic", corr = c3)
  results[[3]] <- c3
  
  # ---- 4. Pairwise interactions ----
  cat("Computing pairwise interaction correlations...\n")
  pairs <- combn(num_cols, 2, simplify = FALSE)
  c4 <- lapply(pairs, function(p) {
    term <- paste(p, collapse="*")
    v <- X[[p[1]]] * X[[p[2]]]
    cxy <- cor(v, y, use="complete.obs")
    data.table(term = term, type = "interaction", corr = cxy)
  })
  results[[4]] <- rbindlist(c4)
  
  # ---- Combine & rank ----
  all_corr <- rbindlist(results)
  all_corr[, abs_corr := abs(corr)]
  setorder(all_corr, -abs_corr)
  
  cat("\n=== TOP CORRELATED TERMS (any order) ===\n")
  print(head(all_corr, top_n))
  
  return(all_corr)
}

# Example run:
corr_table <- inspect_poly_corr(model_data, target_col = "total_consumption_kWh")


library(ggplot2)
ggplot(corr_table[1:30], aes(x = reorder(term, abs_corr), y = abs_corr, fill = type)) +
  geom_col() +
  coord_flip() +
  labs(x = "Feature / Interaction", y = "|Correlation with target|", fill = "Type") +
  theme_minimal()
