# ==============================================================
# Author: Pauline Cox
# Script: main_pipeline_data_preparation.R
#
# Description: Main pipeline for loading, cleaning, and preparing energy,
# occupancy, and weather data for forecasting. It performs exploratory data 
# analysis, and constructs the final feature-engineered dataset used.
#
# Input: 
#   - initialize.R
#   - load_data.R
#   - eda.R
#   - feature_engineering.R
#   - Raw datasets (Eneco, sensor, KNMI)
#
# Output: Cleaned and feature-engineered dataset ready for model training.
# ==============================================================

# Clear workspace
rm(list = ls())

#  --- Initialization ---

source("initialize.R")
initialize_environment()

# --- Data loading --- 

source("load_data.R")
eneco_data <- load_eneco_data()
sensor_data <- load_sensor_data()
knmi_data <- load_knmi_data()
raw_data <- combine_data()

# raw_data <- load_all_data()

cat("Raw data dimensions:", nrow(raw_data), "rows x", ncol(raw_data), "columns\n")

# --- Target variable exploration & processing --- 

source("eda.R")
eneco_data_processed <- explore_target()

cat("Processed target dimensions:", nrow(eneco_data_processed), "observations\n")

# --- Full data exploration & processing --- 

step_start <- Sys.time()
clean_data <- explore_full_data()

# --- Feature engineering --- 

source("feature_engineering.R")
model_data <- add_engineered_features(clean_data)
model_data <- model_data[interval >= as.POSIXct("2023-07-01 00:00:00") & 
                           interval <= as.POSIXct("2024-12-31 23:59:59")]

cat("Feature-engineered data dimensions:", nrow(model_data), "rows x", ncol(model_data), "columns\n")

# --- Finished data preparation pipeline ---

