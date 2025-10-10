# ===============================================================
# ENHANCED PIPELINE — SARIMA + MULTI-SARIMAX (RAW, BASE, TEMPORAL, CLUSTER, CLUSTER+TEMP, PCA)
# ===============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(forecast)
  library(ggplot2)
  library(lubridate)
  library(gridExtra)
  library(stats)
})


add saving forecasts and runtimes and also model fit

# ===============================================================
# 0) SETTINGS
# ===============================================================
set.seed(1234)

ORDER    <- c(2, 0, 2)
SEASONAL <- c(0, 1, 1)
PERIOD   <- 168L  # weekly seasonality
target_col <- "total_consumption_kWh"

# Feature sets
selected_xreg <- c(
  "co2", "business_hours", "total_occupancy",
  "lux", "dow_cos", "hour_cos", "hour_sin",
  "holiday", "dst"
)

base_xreg     <- c("co2","total_occupancy","lux")
temporal_vars <- c("business_hours","dow_cos","hour_cos","hour_sin","holiday","dst")

env_vars <- c("tempC","humidity","co2","sound","lux","temperature","wind_speed",
              "sunshine_minutes","global_radiation","humidity_percent",
              "fog","rain","snow","thunder","ice")

# Date ranges
testA_start <- as.Date("2024-10-01")
testA_end   <- as.Date("2024-10-14")
trainA_end  <- testA_start - 1

testB_start <- as.Date("2024-12-18")
testB_end   <- as.Date("2024-12-31")
trainB_end  <- testB_start - 1

# ===============================================================
# 1) HELPER FUNCTIONS
# ===============================================================

scale_xreg <- function(train_data, test_data, xreg_vars) {
  x_train <- as.matrix(train_data[, ..xreg_vars])
  x_test  <- as.matrix(test_data[, ..xreg_vars])
  x_train_scaled <- scale(x_train)
  x_test_scaled  <- scale(
    x_test,
    center = attr(x_train_scaled, "scaled:center"),
    scale  = attr(x_train_scaled, "scaled:scale")
  )
  list(train = x_train_scaled, test = x_test_scaled)
}

mape <- function(actual, pred) {
  mean(abs((actual - pred) / pmax(actual, 1e-6)), na.rm = TRUE) * 100
}

sarima_forecast <- function(train_data, test_data, order, seasonal, period) {
  y_train <- ts(train_data[[target_col]], frequency = period)
  y_test  <- test_data[[target_col]]
  fit <- Arima(y_train, order = order,
               seasonal = list(order = seasonal, period = period),
               method = "CSS")
  fc <- forecast(fit, h = length(y_test))
  list(pred = as.numeric(fc$mean), fit = fit)
}

sarimax_forecast <- function(train_data, test_data, order, seasonal, period, xreg_vars) {
  y_train <- ts(train_data[[target_col]], frequency = period)
  y_test  <- test_data[[target_col]]
  scaled <- scale_xreg(train_data, test_data, xreg_vars)
  fit <- Arima(y_train,
               order = order,
               seasonal = list(order = seasonal, period = period),
               xreg = scaled$train,
               method = "CSS")
  fc <- forecast(fit, xreg = scaled$test, h = length(y_test))
  list(pred = as.numeric(fc$mean), fit = fit)
}

evaluate_models <- function(actual, preds) {
  res <- lapply(names(preds), function(name) {
    pred <- preds[[name]]
    rmse <- sqrt(mean((pred - actual)^2, na.rm = TRUE))
    mae  <- mean(abs(pred - actual), na.rm = TRUE)
    r2   <- 1 - sum((actual - pred)^2) / sum((actual - mean(actual))^2)
    map  <- mape(actual, pred)
    data.table(Model = name, RMSE = rmse, MAE = mae, R2 = r2, MAPE = map)
  })
  res <- rbindlist(res)
  setorder(res, RMSE)
  return(res)
}

plot_forecasts_facet <- function(actual, preds, title) {
  df_list <- lapply(names(preds), function(name) {
    data.table(Time = seq_along(actual), Forecast = preds[[name]], Model = name)
  })
  df <- rbindlist(df_list)
  df_actual <- data.table(Time = seq_along(actual), Actual = actual)
  df_melt <- merge(df, df_actual, by = "Time")
  
  ggplot(df_melt, aes(x = Time)) +
    geom_line(aes(y = Actual), color = "black", linewidth = 1) +
    geom_line(aes(y = Forecast, color = Model), linewidth = 0.8) +
    facet_wrap(~Model, ncol = 1, scales = "free_y") +
    labs(title = title, x = "Time (hours)", y = "Energy Consumption (kWh)") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none",
          strip.text = element_text(size = 11, face = "bold"))
}

env_clustering <- function(dt, env_vars, k = 3) {
  env_scaled <- scale(dt[, ..env_vars])
  km <- kmeans(env_scaled, centers = k, nstart = 25)
  dt[, env_cluster := factor(km$cluster)]
  list(data = dt, centers_scaled = km$centers, env_vars = env_vars)
}

assign_clusters <- function(new_data, env_vars, train_data, centers_scaled) {
  train_scaled <- scale(train_data[, ..env_vars])
  center_vec <- attr(train_scaled, "scaled:center")
  scale_vec  <- attr(train_scaled, "scaled:scale")
  new_scaled <- scale(as.matrix(new_data[, ..env_vars]),
                      center = center_vec, scale = scale_vec)
  distances <- apply(new_scaled, 1, function(row)
    sqrt(colSums((t(centers_scaled) - row)^2)))
  clusters <- apply(distances, 2, which.min)
  return(factor(clusters))
}

pca_features <- function(train_data, test_data, vars, ncomp = 3) {
  pca <- prcomp(train_data[, ..vars], scale. = TRUE)
  train_pca <- as.data.table(predict(pca, train_data[, ..vars])[, 1:ncomp, drop = FALSE])
  test_pca  <- as.data.table(predict(pca, test_data[, ..vars])[, 1:ncomp, drop = FALSE])
  names(train_pca) <- paste0("PCA", 1:ncomp)
  names(test_pca)  <- paste0("PCA", 1:ncomp)
  list(train = train_pca, test = test_pca)
}

# ===============================================================
# 2) FUNCTION TO RUN ALL MODELS FOR A GIVEN PERIOD
# ===============================================================
run_all_models <- function(train, test, period_label) {
  cat(sprintf("\n==================== %s ====================\n", period_label))
  
  # --- CLUSTER CREATION ---
  clus <- env_clustering(copy(train), env_vars)
  train_cl <- clus$data
  test_cl  <- copy(test)
  test_cl[, env_cluster := assign_clusters(test_cl, env_vars, train_cl, clus$centers_scaled)]
  train_cl[, env_cluster_num := as.numeric(env_cluster)]
  test_cl[, env_cluster_num := as.numeric(env_cluster)]
  
  # --- PCA FEATURES ---
  pca <- pca_features(train, test, vars = selected_xreg, ncomp = 3)
  train_pca <- cbind(train, pca$train)
  test_pca  <- cbind(test, pca$test)
  
  # --- FORECASTS ---
  sarima_mod        <- sarima_forecast(train, test, ORDER, SEASONAL, PERIOD)
  sarimax_full      <- sarimax_forecast(train, test, ORDER, SEASONAL, PERIOD, selected_xreg)
  sarimax_base      <- sarimax_forecast(train, test, ORDER, SEASONAL, PERIOD, base_xreg)
  sarimax_temporal  <- sarimax_forecast(train, test, ORDER, SEASONAL, PERIOD, temporal_vars)
  sarimax_cluster   <- sarimax_forecast(train_cl, test_cl, ORDER, SEASONAL, PERIOD, c("env_cluster_num"))
  sarimax_clustemp  <- sarimax_forecast(train_cl, test_cl, ORDER, SEASONAL, PERIOD,
                                        c("env_cluster_num", temporal_vars))
  sarimax_pca       <- sarimax_forecast(train_pca, test_pca, ORDER, SEASONAL, PERIOD,
                                        c("PCA1","PCA2","PCA3"))
  
  # --- EVALUATION ---
  actual <- test[[target_col]]
  preds <- list(
    SARIMA = sarima_mod$pred,
    SARIMAX_FULL = sarimax_full$pred,
    SARIMAX_BASE = sarimax_base$pred,
    SARIMAX_TEMPORAL = sarimax_temporal$pred,
    SARIMAX_CLUSTER = sarimax_cluster$pred,
    SARIMAX_CLUSTEMP = sarimax_clustemp$pred,
    SARIMAX_PCA = sarimax_pca$pred
  )
  
  eval <- evaluate_models(actual, preds)
  print(eval)
  
  # --- PLOT ---
  plot <- plot_forecasts_facet(actual, preds, paste0("Forecast Comparison — ", period_label))
  print(plot)
  
  return(list(eval = eval))
}

# ===============================================================
# 3) RUN BOTH TEST PERIODS
# ===============================================================

trainA <- model_data[as.Date(interval) <= trainA_end]
testA  <- model_data[as.Date(interval) >= testA_start & as.Date(interval) <= testA_end]

trainB <- model_data[as.Date(interval) <= trainB_end]
testB  <- model_data[as.Date(interval) >= testB_start & as.Date(interval) <= testB_end]

resA <- run_all_models(trainA, testA, "Test Period A")
resB <- run_all_models(trainB, testB, "Test Period B")

# ===============================================================
# 4) SUMMARY
# ===============================================================
cat("\n==================== SUMMARY ====================\n")
cat(sprintf("Best Model Period A: %s (RMSE=%.2f)\n", resA$eval[1, Model], resA$eval[1, RMSE]))
cat(sprintf("Best Model Period B: %s (RMSE=%.2f)\n", resB$eval[1, Model], resB$eval[1, RMSE]))
cat("\nPipeline complete!\n")
