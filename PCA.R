# ==============================================================================
# PCA FEATURE TRANSFORMATION FUNCTION (VERBOSE + TRAIN/TEST)
# ==============================================================================

pca_features <- function(train_data, test_data, var_threshold = 0.9) {
  suppressPackageStartupMessages({
    library(data.table)
  })
  
  cat("\n=== PCA FEATURE EXTRACTION ===\n")
  
  # Identify numeric candidate columns
  is_num <- function(x) is.numeric(x) || is.integer(x)
  exclude_patterns <- c("^interval$", "^date$", "^holiday$", "^total_consumption_kWh$",
                        "^lag_", "^rollmean", "^consumption_roll")
  exclude_cols <- unique(unlist(lapply(exclude_patterns, function(p) grep(p, names(train_data), value = TRUE))))
  candidate_cols <- setdiff(names(train_data), exclude_cols)
  xreg_cols <- candidate_cols[sapply(train_data[, ..candidate_cols], is_num)]
  
  cat("Candidate variables for PCA:", length(xreg_cols), "\n")
  
  # Impute missing values (mean imputation)
  X_train <- train_data[, ..xreg_cols]
  X_test  <- test_data[, ..xreg_cols]
  for (cl in xreg_cols) {
    mu <- mean(X_train[[cl]], na.rm = TRUE)
    if (anyNA(X_train[[cl]])) X_train[[cl]][is.na(X_train[[cl]])] <- mu
    if (anyNA(X_test[[cl]]))  X_test[[cl]][is.na(X_test[[cl]])]  <- mu
  }
  
  # ---------------------------------------------------------------------------
  # 1️⃣ PCA on ALL VARIABLES
  # ---------------------------------------------------------------------------
  cat("\n--- PCA on ALL VARIABLES ---\n")
  pca_all <- prcomp(as.matrix(X_train), center = TRUE, scale. = TRUE)
  var_exp_all <- summary(pca_all)$importance[2, ]     # Proportion of variance
  cum_var_all <- summary(pca_all)$importance[3, ]     # Cumulative variance
  K_all <- which(cum_var_all >= var_threshold)[1]
  if (is.na(K_all)) K_all <- ncol(X_train)
  
  cat(sprintf("  • Retained PCs: %d (threshold %.2f)\n", K_all, var_threshold))
  cat(sprintf("  • Variance explained by first %d PCs: %.2f%%\n", K_all, cum_var_all[K_all] * 100))
  cat("  • Individual variance per retained PC:\n")
  print(round(var_exp_all[1:K_all] * 100, 2))
  
  X_train_pca_all <- predict(pca_all, newdata = as.matrix(X_train))[, 1:K_all, drop = FALSE]
  X_test_pca_all  <- predict(pca_all, newdata = as.matrix(X_test))[, 1:K_all, drop = FALSE]
  
  # ---------------------------------------------------------------------------
  # 2️⃣ PCA on ENVIRONMENTAL VARIABLES (+ occupancy + holiday)
  # ---------------------------------------------------------------------------
  cat("\n--- PCA on ENVIRONMENTAL VARIABLES + OCC + HOL ---\n")
  env_vars <- c("tempC","humidity","co2","sound","lux",
                "temperature","wind_speed","sunshine_minutes",
                "global_radiation","humidity_percent",
                "fog","rain","snow","thunder","ice")
  env_vars <- intersect(env_vars, xreg_cols)
  
  if (length(env_vars) == 0) {
    stop("No matching environmental variables found in data.")
  }
  
  pca_env <- prcomp(as.matrix(train_data[, ..env_vars]), center = TRUE, scale. = TRUE)
  var_exp_env <- summary(pca_env)$importance[2, ]
  cum_var_env <- summary(pca_env)$importance[3, ]
  K_env <- which(cum_var_env >= var_threshold)[1]
  if (is.na(K_env)) K_env <- length(env_vars)
  
  cat(sprintf("  • Retained PCs: %d (threshold %.2f)\n", K_env, var_threshold))
  cat(sprintf("  • Variance explained by first %d PCs: %.2f%%\n", K_env, cum_var_env[K_env] * 100))
  cat("  • Individual variance per retained PC:\n")
  print(round(var_exp_env[1:K_env] * 100, 2))
  
  X_train_envplus <- cbind(
    predict(pca_env, newdata = as.matrix(train_data[, ..env_vars]))[, 1:K_env, drop = FALSE],
    occ = train_data$total_occupancy,
    hol = train_data$holiday
  )
  X_test_envplus <- cbind(
    predict(pca_env, newdata = as.matrix(test_data[, ..env_vars]))[, 1:K_env, drop = FALSE],
    occ = test_data$total_occupancy,
    hol = test_data$holiday
  )
  
  cat(sprintf("\n✓ PCA complete | Components kept: %d (ALL), %d (ENV)\n", K_all, K_env))
  
  list(
    pca_all = list(train = X_train_pca_all, test = X_test_pca_all),
    pca_env = list(train = X_train_envplus, test = X_test_envplus)
  )
}
