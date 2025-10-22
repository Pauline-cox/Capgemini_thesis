# Environment Initialization

# color pallet 
# colors <- c("#0072B2", "#E69F00", "#009E73", "#CC79A7", 
#             "#F0E442", "#56B4E9", "#D55E00")

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
    "ggpubr", "patchwork", "grid", "knitr"
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
  
  # Load tensorflow and verify
  library(tensorflow)
  library(keras)
  
  # Null-coalescing helper
  if (!exists("%||%", mode = "function")) {
    `%||%` <<- function(a, b) if (!is.null(a)) a else b
  }
}