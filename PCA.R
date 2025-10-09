pca_features <- function(train_data, test_data, var_threshold = 0.9) {
  suppressPackageStartupMessages(library(data.table))
  cat("\n=== PCA FEATURE EXTRACTION ===\n")
  
  is_num <- function(x) is.numeric(x) || is.integer(x)
  exclude_patterns <- c("^interval$", "^date$", "^holiday$", "^total_consumption_kWh$", 
                        "^lag_", "^rollmean", "^consumption_roll")
  exclude_cols <- unique(unlist(lapply(exclude_patterns, grep, x = names(train_data), value = TRUE)))
  candidate_cols <- setdiff(names(train_data), exclude_cols)
  xreg_cols <- candidate_cols[sapply(train_data[, ..candidate_cols], is_num)]
  
  # Impute missing values (mean)
  for (col in xreg_cols) {
    mu <- mean(train_data[[col]], na.rm = TRUE)
    train_data[[col]][is.na(train_data[[col]])] <- mu
    test_data[[col]][is.na(test_data[[col]])] <- mu
  }
  
  # --- PCA: All numeric vars ---
  pca_all <- prcomp(train_data[, ..xreg_cols], center = TRUE, scale. = TRUE)
  cum_var <- summary(pca_all)$importance[3, ]
  K_all <- which(cum_var >= var_threshold)[1]
  if (is.na(K_all)) K_all <- length(xreg_cols)
  
  X_train_pca_all <- predict(pca_all, train_data[, ..xreg_cols])[, 1:K_all, drop = FALSE]
  X_test_pca_all  <- predict(pca_all, test_data[, ..xreg_cols])[, 1:K_all, drop = FALSE]
  colnames(X_train_pca_all) <- colnames(X_test_pca_all) <- paste0("PC_all_", seq_len(K_all))
  
  # --- PCA: Environmental subset ---
  env_vars <- c("tempC","humidity","co2","sound","lux","temperature","wind_speed",
                "sunshine_minutes","global_radiation","humidity_percent",
                "fog","rain","snow","thunder","ice")
  env_vars <- intersect(env_vars, xreg_cols)
  pca_env <- prcomp(train_data[, ..env_vars], center = TRUE, scale. = TRUE)
  cum_var_env <- summary(pca_env)$importance[3, ]
  K_env <- which(cum_var_env >= var_threshold)[1]
  if (is.na(K_env)) K_env <- length(env_vars)
  
  X_train_env <- predict(pca_env, train_data[, ..env_vars])[, 1:K_env, drop = FALSE]
  X_test_env  <- predict(pca_env, test_data[, ..env_vars])[, 1:K_env, drop = FALSE]
  colnames(X_train_env) <- colnames(X_test_env) <- paste0("PC_env_", seq_len(K_env))
  
  list(
    pca_all = list(train = as.data.table(X_train_pca_all), test = as.data.table(X_test_pca_all)),
    pca_env = list(train = as.data.table(cbind(X_train_env, occ = train_data$total_occupancy, hol = train_data$holiday)),
                   test  = as.data.table(cbind(X_test_env,  occ = test_data$total_occupancy,  hol = test_data$holiday)))
  )
}
