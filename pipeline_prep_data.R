# ==============================================================================
# MAIN PIPELINE: Energy Consumption Forecasting
# ==============================================================================
# Description: Complete pipeline for loading, processing, and forecasting
#              energy consumption data using SARIMA, SARIMAX, and LSTM models
# Author: [Your Name]
# Date: September 30, 2025
# ==============================================================================

# Clear workspace
rm(list = ls())

# Set seeds for reproducibility
set.seed(1234)

# Start overall timer
pipeline_start <- Sys.time()
cat("\n", rep("=", 80), "\n", sep = "")
cat("ENERGY CONSUMPTION FORECASTING PIPELINE\n")
cat("Started at:", format(pipeline_start, "%Y-%m-%d %H:%M:%S"), "\n")
cat(rep("=", 80), "\n\n", sep = "")

#  --- 1. ENVIRONMENT INITIALIZATION --- 
step_start <- Sys.time()

source("initialize.R")
initialize_environment()

step_duration <- round(difftime(Sys.time(), step_start, units = "secs"), 2)
cat("    Completed in:", step_duration, "seconds\n\n")

# --- DATA LOADING --- 

cat("STEP 2: Loading Data\n")
step_start <- Sys.time()

source("load_data.R")
eneco_data <- load_eneco_data()
sensor_data <- load_sensor_data()
knmi_data <- load_knmi_data()
raw_data <- combine_data()

# raw_data <- load_all_data()

cat("    Raw data dimensions:", nrow(raw_data), "rows x", ncol(raw_data), "columns\n")
step_duration <- round(difftime(Sys.time(), step_start, units = "secs"), 2)
cat("    Completed in:", step_duration, "seconds\n\n")


# --- TARGET VARIABLE EXPLORATION & PROCESSING --- 

cat("STEP 3: Exploring and Processing Target Variable\n")
step_start <- Sys.time()

source("eda.R")
eneco_data_processed <- explore_target()

cat("    Processed target dimensions:", nrow(eneco_data_processed), "observations\n")
step_duration <- round(difftime(Sys.time(), step_start, units = "secs"), 2)
cat("    Completed in:", step_duration, "seconds\n\n")


# --- FULL DATA EXPLORATION & CLEANING --- 

cat("STEP 4: Exploring Full Dataset with Features\n")
step_start <- Sys.time()

clean_data <- explore_full_data()

# Verification checks
cat("    Verifying data alignment...\n")
if (identical(clean_data$interval, eneco_data_processed$interval)) {
  cat("    Interval alignment verified\n")
} else {
  warning("    Interval mismatch detected!")
}

cat("    Target variable summary:\n")
print(summary(clean_data$total_consumption_kWh))

step_duration <- round(difftime(Sys.time(), step_start, units = "secs"), 2)
cat("    Completed in:", step_duration, "seconds\n\n")

# --- FEATURE ENGINEERING --- 

cat("STEP 5: Engineering Additional Features\n")
step_start <- Sys.time()

source("feature_engineering.R")
model_data <- add_engineered_features(clean_data)
model_data <- model_data[interval >= as.POSIXct("2023-07-01 00:00:00") & 
                           interval <= as.POSIXct("2024-12-31 23:59:59")]

cat("    Feature-engineered data dimensions:", nrow(model_data), "rows x", ncol(model_data), "columns\n")
step_duration <- round(difftime(Sys.time(), step_start, units = "secs"), 2)
cat("    Completed in:", step_duration, "seconds\n\n")

step_duration <- round(difftime(Sys.time(), step_start, units = "secs"), 2)
cat("    Completed in:", step_duration, "seconds\n\n")

