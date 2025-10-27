# ==============================================================
# Author: Pauline Cox
# Script: run_all_models.R
#
# Description: Main  execution pipeline for the forecasting.
# Sequentially sources all component scripts required to perform
# data preparation, model training, forecasting, and evaluation.
# ==============================================================

# Prepare forecast functions and splits
source("forecast_preperations.R")

# Run forecast script for all models
source("fc_sarima.R")
source("fc_sarimax.R")
source("fc_sarimax_pca.R")
source("fc_sarimax_cluster.R")
source("fc_lstm.R")
source("fc_hybrid.R")

# Run evaluation script
source("fc_evaluation.R")

