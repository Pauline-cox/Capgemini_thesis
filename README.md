Energy Forecasting Thesis - Code Repository
Author: Pauline Cox
Title: Forecasting Building Energy Consumption 24 Hours Ahead: A Comparative Analysis of SARIMA, SARIMAX, LSTM, and Hybrid Models

Overview
This repository contains the complete implementation for a master's thesis comparing statistical, deep learning, and hybrid approaches for 24-hour ahead energy consumption forecasting in an office building. The analysis evaluates model performance under both stable operational periods and volatile conditions (holidays, irregular occupancy, seasonal transitions) to assess regime-dependent forecasting behavior.

Key Finding
No single model dominates universally across all conditions. Statistical models (SARIMA, SARIMAX) excel during stable operational periods with predictable patterns, while LSTM neural networks maintain superior accuracy during volatile periods with irregular occupancy and non-standard events. PCA-based dimensionality reduction approaches neural network performance while preserving model interpretability, offering a practical middle ground for operational deployment.

Repository Structure
code_thesis/
├── initialize.R                          # Environment setup
├── main_pipeline_data_preparation.R      # Data processing pipeline
├── main_pipeline_forecast.R              # Forecasting pipeline
│
├── Data Loading & Preprocessing
│   ├── load_data.R                       # Load energy, sensor, weather data
│   ├── eda.R                             # Exploratory analysis
│   ├── feature_engineering.R             # Feature creation
│   └── forecast_preperations.R           # Train/test splits
│
├── Model Development
│   ├── gridsearch_sarima.R               # SARIMA order selection
│   ├── feature_selection_sarimax.R       # Feature selection
│   ├── tune_lstm_bayes.R                 # LSTM hyperparameter tuning
│   └── tune_hybrid_bayes.R               # Hybrid tuning
│
├── Forecasting Models
│   ├── fc_sarima.R                       # SARIMA baseline
│   ├── fc_sarimax.R                      # SARIMAX with features
│   ├── fc_sarimax_pca.R                  # SARIMAX with PCA
│   ├── fc_sarimax_cluster.R              # SARIMAX with clustering
│   ├── fc_lstm.R                         # LSTM neural network
│   └── fc_hybrid.R                       # SARIMAX→LSTM hybrid
│
└── Evaluation & Diagnostics
    ├── evaluation_forecasts.R            # Model comparison
    ├── diagnostics_residuals.R           # Residual analysis
    ├── diagnostics_pca.R                 # PCA diagnostics
    └── diagnostics_clustering.R          # Clustering diagnostics

Requirements
R: 4.0+ with packages forecast, data.table, dplyr, ggplot2, keras, reticulate, ParBayesianOptimization, cluster, tseries, caret
Python: 3.10 with tensorflow>=2.10, keras, pandas
Data: Energy consumption (Eneco Excel files), sensor data (CSV), weather data (KNMI text format)

Quick Start

# 1. Configure environment
source("initialize.R")
initialize_environment()

# 2. Run data preparation
source("main_pipeline_data_preparation.R")

# 3. Run forecasting pipeline
source("main_pipeline_forecast.R")

Models

SARIMA (Baseline)
Pure autoregressive model without exogenous features
Seasonal ARIMA(1,0,1)(1,0,1)[168] structure
Use case: Stable conditions with regular patterns

SARIMAX
SARIMA extended with exogenous regressors
Features selected via forward stepwise selection
Use case: Stable conditions with known predictors

SARIMAX-PCA
SARIMAX with PCA-transformed environmental features
Orthogonalizes correlated predictors to reduce multicollinearity
Use case: Volatile conditions requiring robust linear features

SARIMAX-Clustering
SARIMAX with k-means regime indicators
Binary encoding of weather/occupancy clusters
Use case: Moderate volatility with identifiable regimes

LSTM
Deep learning sequence model with 168-hour lookback
Two LSTM layers (128, 64 units) plus dropout regularization
Direct 24-hour ahead prediction (no recursive forecasting)
Use case: Volatile conditions requiring adaptability

Hybrid (SARIMAX→LSTM)
Sequential decomposition: SARIMAX baseline plus LSTM residual correction
LSTM learns from SARIMAX residuals
Use case: Balancing transparency and robustness

Evaluation Framework

Test Periods:
Period A (Stable): October 1-14, 2024 (336 hours)
Period B (Volatile): December 18-31, 2024 (336 hours, includes holidays)

Metrics:
Point accuracy: RMSE, MAE, MAPE, R²
Statistical testing: Diebold-Mariano tests with HAC correction
Error diagnostics: Distributional analysis (skewness, kurtosis, normality)
Temporal patterns: Hour-of-day and day-of-week accuracy profiles

Key Results:
During stable periods: All models achieve similar accuracy (RMSE 33-39 kWh)
During volatility: LSTM achieves 26% improvement over SARIMA
PCA-SARIMAX approaches LSTM accuracy with only 4.5% gap while maintaining interpretability
Hybrid model shows statistically significant and consistent improvements
