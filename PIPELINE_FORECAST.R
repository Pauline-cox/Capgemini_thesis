# Main pipeline for final forecast for both test periods

# Set seeds for reproducibility
set.seed(1234)  # For R
tensorflow::set_random_seed(1234)  # For TensorFlow/Keras

source("SARIMA_FORECAST.R")
source("SARIMA_FORECAST.R")
source("LSTM_FORECAST.R")
source("HYBRID_FORECAST.R")

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

# ---------------------- PREPROCESS FOR SARIMAX --------------------------------

# FORWARD SELECTION 


# CLUSTERING
clusA <- env_clustering(copy(trainA))
clusB <- env_clustering(copy(trainB))

# Use cluster assignment as an additional regressor
trainA_cl <- clusA$data
testA_cl  <- testA
testA_cl[, env_cluster := factor(predict(kmeans(scale(trainA_cl[, ..clusA$env_vars]),
                                                centers = clusA$centers_scaled), 
                                         newdata = scale(testA[, ..clusA$env_vars])))]
trainB_cl <- clusB$data
testB_cl  <- testB
testB_cl[, env_cluster := factor(predict(kmeans(scale(trainB_cl[, ..clusB$env_vars]),
                                                centers = clusB$centers_scaled),
                                         newdata = scale(testB[, ..clusB$env_vars])))]
# PCA
pcaA <- pca_features(trainA, testA)
pcaB <- pca_features(trainB, testB)


# ---------------------------- FORECASTS ---------------------------------------

# --- SARIMA ---

sarima_fc_A <- forecast_sarima(trainA, testA, order_best, seasonal_best)
sarima_fc_B <- forecast_sarima(trainB, testB, order_best, seasonal_best)

# --- SARIMAX (raw xreg) ---

# results from forward selection
sarimax_raw_xreg_fc_A <- forecast_sarimax(trainA, testA, order_best, seasonal_best, xregA)
sarimax_raw_xreg_fc_B <- forecast_sarimax(trainB, testB, order_best, seasonal_best, xregB)

# --- SARIMAX (enviromental clustering) ---


sarimax_envclust_A <- sarimax_forecast(trainA_cl, testA_cl, order_best, seasonal_best, c("env_cluster"))
sarimax_envclust_B <- sarimax_forecast(trainB_cl, testB_cl, order_best, seasonal_best, c("env_cluster"))

# --- SARIMAX (enviromental clustering + occ + holiday) ---

sarimax_envclust_occ_A <- sarimax_forecast(trainA_cl, testA_cl, order_best, seasonal_best, c("env_cluster", "total_occupancy", "holiday"))
sarimax_envclust_occ_B <- sarimax_forecast(trainB_cl, testB_cl, order_best, seasonal_best, c("env_cluster", "total_occupancy", "holiday"))

# --- SARIMAX (enviromental PCA (all)) ---

sarimax_pcaall_A <- sarimax_forecast(trainA, testA, order_best, seasonal_best, colnames(pcaA$pca_all$train))
sarimax_pcaall_B <- sarimax_forecast(trainB, testB, order_best, seasonal_best, colnames(pcaB$pca_all$train))


# --- SARIMAX (enviromental PCA (environmental) +  occ + holiday) ---

sarimax_pcaenv_A <- sarimax_forecast(trainA, testA, order_best, seasonal_best, colnames(pcaA$pca_env$train))
sarimax_pcaenv_B <- sarimax_forecast(trainB, testB, order_best, seasonal_best, colnames(pcaB$pca_env$train))

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
  SARIMAX_PCA_ENV      = sarimax_pcaenv_A,
  LSTM                 = lstm_fc_A,
  HYBRID               = hybrid_fc_A
))

eval_B <- evaluate_all(list(
  SARIMA               = sarima_fc_B,
  SARIMAX_RAW          = sarimax_raw_B,
  SARIMAX_ENVCLUST     = sarimax_envclust_B,
  SARIMAX_ENVCLUST_OCC = sarimax_envclust_occ_B,
  SARIMAX_PCA_ALL      = sarimax_pcaall_B,
  SARIMAX_PCA_ENV      = sarimax_pcaenv_B,
  LSTM                 = lstm_fc_B,
  HYBRID               = hybrid_fc_B
))

cat("\n=== Test Period A ===\n"); print(eval_A)
cat("\n=== Test Period B ===\n"); print(eval_B)

plot_forecasts(list(
  SARIMA=sarima_fc_A, SARIMAX_RAW=sarimax_raw_A,
  SARIMAX_ENVCLUST=sarimax_envclust_A,
  SARIMAX_PCA_ALL=sarimax_pcaall_A,
  LSTM=lstm_fc_A, HYBRID=hybrid_fc_A),
  title="Forecast Comparison – Period A"
)

plot_forecasts(list(
  SARIMA=sarima_fc_B, SARIMAX_RAW=sarimax_raw_B,
  SARIMAX_ENVCLUST=sarimax_envclust_B,
  SARIMAX_PCA_ALL=sarimax_pcaall_B,
  LSTM=lstm_fc_B, HYBRID=hybrid_fc_B),
  title="Forecast Comparison – Period B"
)