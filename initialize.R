# ==============================================================
# Author: Pauline Cox
# Script: initialize.R
#
# Description: Configures the computational environment for the energy forecasting
# project. Installs and loads required R and Python packages, sets 
# random seeds for reproducibility, and defines utility functions.
#
# Input: None 
#
# Output: Loaded libraries and configured environment for subsequent scripts.
# ==============================================================

initialize_environment <- function() {
  
  # Python version
  message("Configuring Python environment...")
  library(reticulate)
  use_python("C:/Users/pauli/AppData/Local/Programs/Python/Python310/python.exe", required = TRUE)
  
  # Install required R packages
  packages <- c(
    "readxl", "data.table", "dplyr", "tidyr", "tibble", "stringr", "MASS", "tseries",
    "lubridate", "furrr", "future", "corrplot", "ggplot2", "forecast", "urca",
    "randomForest", "caret", "recipes", "Metrics", "xgboost", "zoo", "purrr", "ranger",
    "SHAPforxgboost", "reshape2", "psych", "viridis", "Amelia", "VIM", "tsibble", "car", 
    "ParBayesianOptimization", "gridExtra", "stats", "rBayesianOptimization",
    "ggpubr", "patchwork", "grid", "knitr", "nortest", "e1071", "RColorBrewer", "cluster"
  )
  
  new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
  if (length(new_packages)) install.packages(new_packages, dependencies = TRUE)
  lapply(packages, library, character.only = TRUE)
  
  # Install required Python packages
  if (!py_module_available("tensorflow")) {
    py_install(c(
      "tensorflow", "keras", "tensorflow-hub", "tensorflow-datasets",
      "scipy", "pandas", "h5py", "pillow", "requests"
    ))
  }
  
  # Load tensorflow
  library(tensorflow)
  library(keras)
  
  # Null-coalescing helper
  if (!exists("%||%", mode = "function")) {
    `%||%` <<- function(a, b) if (!is.null(a)) a else b
  }
  
  # Set seed
  set.seed(1234)
  tensorflow::set_random_seed(1234)
}