source("SARIMA_GRID_SEARCH.R")
source("LSTM_HYPERPARAM_TUNING.R")
source("HYBRID_HYPERPARAM_TUNING.R")

# Split data
train_data <- model_data[interval >= "2023-07-01" & interval < "2024-06-30"]
val_data   <- model_data[interval >= "2024-07-01" & interval < "2024-09-30"]

feature_columns <- c(
  "total_occupancy", "co2", "tempC", "humidity", "sound", "lux",
  "temperature", "global_radiation", "sunshine_minutes", "wind_speed", "humidity_percent",
  "holiday", "weekend", "business_hours",
  "hour_sin", "hour_cos", "dow_sin", "dow_cos",
  "lag_24", "lag_72", "lag_168", "rollmean_24", "rollmean_168"
)

# --- SARIMA ORDER SELECTION ---
cat("Running SARIMA grid search")
sarima_opt <- sarima_grid_search()

# --- LSTM HYPERPARAMETER OPTIMIZATION ---
cat("Running LSTM Bayesian Optimization")
lstm_opt <- lstm_bayesopt_train(train_data, val_data, feature_columns)

# --- HYBRID SARIMA–LSTM OPTIMIZATION ---
cat("Running Hybrid SARIMA–LSTM Bayesian Optimization")
hybrid_opt <- hybrid_bayesopt_train(train_data, val_data, feature_columns, sarima_order = c(2,0,2), sarima_seasonal = c(1,1,1))

# --- PRINT RESULTS ---

cat("SARIMA result")
print(sarima_opt)

cat("LSTM result")
print(lstm_opt) 

cat("Hybrid result")
print(hybrid_opt)
