# Main pipeline for final forecast for both test periods

# Set seeds for reproducibility
set.seed(1234)  # For R
tensorflow::set_random_seed(1234)  # For TensorFlow/Keras

source("FEATURE_SELECTION.R")
source("CLUSTERING.R")
source("PCA.R")
source("SARIMA_FORECAST.R")
source("SARIMAX_FORECAST.R")
source("LSTM_FORECAST.R")
source("HYBRID_FORECAST.R")
source("EVALUATION_FUNCTIONS.R")


testA_start <- as.Date("2024-10-01")
testA_end   <- as.Date("2024-10-14")
testB_start <- as.Date("2024-12-18")
testB_end   <- as.Date("2024-12-31")
trainA_end  <- testA_start - 1
trainB_end  <- testB_start - 1

trainA <- model_data[interval <= trainA_end]
testA  <- model_data[interval >= testA_start & interval <= testA_end]
trainB <- model_data[interval <= trainB_end]
testB  <- model_data[interval >= testB_start & interval <= testB_end]

cat(sprintf("Test A: %s–%s (train ≤ %s)\n", testA_start, testA_end, trainA_end))
cat(sprintf("Test B: %s–%s (train ≤ %s)\n\n", testB_start, testB_end, trainB_end))

# ---------------------- PREPROCESS FOR SARIMAX -----------------------------o

# OPTIMAL ORDER FROM ORDER SELECTION
order          <- c(2,0,2)
seasonal_order <- c(0,1,1)

# FEATURE SELECTION
domain_vars <- c("holiday", "weekend", "dst")  # domain knowledge vars
train_val <- model_data[as.Date(interval) < as.Date("2024-10-01")]
features_selected <- sarimax_feature_selection(train_val, domain_vars, corr_min = 0.3, vif_max = 5)

# CLUSTERING
clusA <- env_clustering(copy(trainA))
clusB <- env_clustering(copy(trainB))

# Use cluster assignment as an additional regressor
trainA_cl <- clusA$data
testA_cl  <- testA
testA_cl[, env_cluster := assign_env_clusters(trainA_cl, testA, clusA)]
sarimax_envclust_A <- sarimax_forecast(trainA_cl, testA_cl, order, seasonal_order, "env_cluster_num")

trainB_cl <- clusB$data
testB_cl  <- testB
testB_cl[, env_cluster := assign_env_clusters(trainB_cl, testB, clusB)]
trainB_cl[, env_cluster_num := as.numeric(as.character(env_cluster))]
testB_cl[, env_cluster_num  := as.numeric(as.character(env_cluster))]


# PCA
pcaA <- pca_features(trainA, testA)
pcaB <- pca_features(trainB, testB)

trainA_pca <- cbind(trainA, pcaA$pca_all$train)
testA_pca  <- cbind(testA,  pcaA$pca_all$test)
trainB_pca <- cbind(trainB, pcaB$pca_all$train)
testB_pca  <- cbind(testB,  pcaB$pca_all$test)


# ---------------------------- FORECASTS ---------------------------------------

# --- SARIMA ---

sarima_fc_A <- sarima_forecast(trainA, testA, order, seasonal_order)
sarima_fc_B <- sarima_forecast(trainB, testB, order, seasonal_order)

# --- SARIMAX (raw xreg) ---

# results from forward selection
sarimax_raw_A <- sarimax_forecast(trainA, testA, order, seasonal_order, selected_features)
sarimax_raw_B <- sarimax_forecast(trainB, testB, order, seasonal_order, selected_features)

# --- SARIMAX (enviromental clustering) ---

sarimax_envclust_A <- sarimax_forecast(trainA_cl, testA_cl, order, seasonal_order, "env_cluster_num")
sarimax_envclust_B <- sarimax_forecast(trainB_cl, testB_cl, order, seasonal_order, "env_cluster_num")

# --- SARIMAX (enviromental clustering + occ + holiday) ---

sarimax_envclust_occ_A <- sarimax_forecast(trainA_cl, testA_cl, order, seasonal_order, c("env_cluster_num", "total_occupancy", "holiday"))
sarimax_envclust_occ_B <- sarimax_forecast(trainB_cl, testB_cl, order, seasonal_order, c("env_cluster_num", "total_occupancy", "holiday"))

# --- SARIMAX (enviromental PCA (all)) ---

sarimax_pcaall_A <- sarimax_forecast(trainA_pca, testA_pca, order, seasonal_order, colnames(pcaA$pca_all$train))
sarimax_pcaall_B <- sarimax_forecast(trainB_pca, testB_pca, order, seasonal_order, colnames(pcaB$pca_all$train))

# --- SARIMAX (enviromental PCA (environmental) +  occ + holiday) ---

sarimax_pcaenv_A <- sarimax_forecast(trainA, testA, order, seasonal_order, colnames(pcaA$pca_env$train))
sarimax_pcaenv_B <- sarimax_forecast(trainB, testB, order, seasonal_order, colnames(pcaB$pca_env$train))

# --- LSTM ---

selected_features <- ...
lstm_fc_A <- forecast_lstm(trainA, testA, lstm_parameters_best, selected_features)
lstm_fc_B <- forecast_lstm(trainB, testB, lstm_parameters_best, selected_features)

# --- HYBRID ---

hybrid_fc_A <- forecast_hybrid(trainA, testA, hybrid_parameters_best, selected_features)
hybrid_fc_B <- forecast_hybrid(trainB, testB, hybrid_parameters_best, selected_features)

# ---------------------------- EVALUATION --------------------------------------

eval_A <- evaluate_all(list(
  SARIMA               = sarima_fc_A,
  SARIMAX_RAW          = sarimax_raw_A,
  SARIMAX_ENVCLUST     = sarimax_envclust_A,
  SARIMAX_ENVCLUST_OCC = sarimax_envclust_occ_A,
  SARIMAX_PCA_ALL      = sarimax_pcaall_A,
  SARIMAX_PCA_ENV      = sarimax_pcaenv_A
), actual_values = testA$total_consumption_kWh)

eval_B <- evaluate_all(list(
  SARIMA               = sarima_fc_B,
  SARIMAX_RAW          = sarimax_raw_B,
  SARIMAX_ENVCLUST     = sarimax_envclust_B,
  SARIMAX_ENVCLUST_OCC = sarimax_envclust_occ_B,
  SARIMAX_PCA_ALL      = sarimax_pcaall_B,
  SARIMAX_PCA_ENV      = sarimax_pcaenv_B
), actual_values = testB$total_consumption_kWh)

cat("\n=== Test Period A ===\n"); print(eval_A)
cat("\n=== Test Period B ===\n"); print(eval_B)

# --- Plots ---
plot_forecasts(list(
  SARIMA=sarima_fc_A,
  SARIMAX_RAW=sarimax_raw_A,
  SARIMAX_ENVCLUST=sarimax_envclust_A,
  SARIMAX_PCA_ALL=sarimax_pcaall_A
), actual_values=testA$total_consumption_kWh,
title="Forecast Comparison – Period A")

plot_forecasts(list(
  SARIMA=sarima_fc_B,
  SARIMAX_RAW=sarimax_raw_B,
  SARIMAX_ENVCLUST=sarimax_envclust_B,
  SARIMAX_PCA_ALL=sarimax_pcaall_B
), actual_values=testB$total_consumption_kWh,
title="Forecast Comparison – Period B")
