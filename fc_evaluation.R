# ==============================================================================
# COMPREHENSIVE MODEL EVALUATION SCRIPT
# ==============================================================================

theme_set(theme_bw(base_size = 11) +
            theme(panel.grid.minor = element_blank(),
                  plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
                  strip.background = element_rect(fill = "grey92"),
                  strip.text = element_text(face = "bold", size = 10),
                  legend.position = "bottom"))

# --- Load results ---
cat("Loading forecast results...\n")
SARIMA  <- readRDS("RESULTS_FC_SARIMA.rds")
SARIMAX <- readRDS("RESULTS_FC_SARIMAX.rds")
LSTM    <- readRDS("RESULTS_FC_LSTM.rds")
HYBRID  <- readRDS("RESULTS_FC_HYBRID.rds")

models <- list(SARIMA, SARIMAX, LSTM, HYBRID)
model_names <- c("SARIMA", "SARIMAX", "LSTM", "HYBRID")
names(models) <- model_names

# --- Performance Comparison ---
cat("\n\nTABLE 2: FORECASTING PERFORMANCE COMPARISON\n\n")

perf_list <- list()
for (i in seq_along(models)) {
  if (!is.null(models[[i]]$evaluations)) {
    perf <- as.data.table(models[[i]]$evaluations)
    perf[, Model := model_names[i]]
    perf_list[[model_names[i]]] <- perf
  }
}
performance <- rbindlist(perf_list, fill = TRUE)
performance[, Period_Clean := gsub("Period | \\(.*\\)", "", Period)]

perf_long <- melt(performance,
                  id.vars = c("Model", "Period_Clean"),
                  measure.vars = c("RMSE", "MAE", "R2", "MAPE"),
                  variable.name = "Metric", value.name = "Value")
perf_wide <- dcast(perf_long, Model ~ Metric + Period_Clean, value.var = "Value")

col_order <- intersect(c("Model", "RMSE_A", "RMSE_B", "MAE_A", "MAE_B",
                         "R2_A", "R2_B", "MAPE_A", "MAPE_B"), names(perf_wide))
setcolorder(perf_wide, col_order)
print(kable(perf_wide, format = "pipe", digits = 2))

cat("\n\nModel Rankings (1 = best):\n")
performance[, `:=`(
  RMSE_Rank = rank(RMSE), 
  MAE_Rank = rank(MAE),
  R2_Rank = rank(-R2),
  MAPE_Rank = rank(MAPE)
), by = Period_Clean]
print(kable(performance[, .(Model, Period = Period_Clean, RMSE_Rank, MAE_Rank, R2_Rank, MAPE_Rank)],
            format = "pipe"))

cat("\n\nBest Model by Metric:\n")
best_models <- performance[, .SD[which.min(RMSE)], by = Period_Clean][, .(Period = Period_Clean, Model, RMSE)]
setnames(best_models, "RMSE", "Best_RMSE")
best_models[performance[, .SD[which.min(MAE)], by = Period_Clean], Best_MAE := i.Model, on = "Period"]
best_models[performance[, .SD[which.max(R2)], by = Period_Clean], Best_R2 := i.Model, on = "Period"]
best_models[performance[, .SD[which.min(MAPE)], by = Period_Clean], Best_MAPE := i.Model, on = "Period"]
print(best_models)

# --- Forecast Extraction ---
cat("\n\nExtracting forecast data...\n")
extract_fc <- function(model, name, period) {
  fc <- model[[period]]$forecasts
  if (!is.null(fc)) as.data.table(fc)[, Model := name]
}
forecasts_A <- lapply(seq_along(models), \(i) extract_fc(models[[i]], model_names[i], "period_A"))
forecasts_B <- lapply(seq_along(models), \(i) extract_fc(models[[i]], model_names[i], "period_B"))
all_forecasts_A <- rbindlist(forecasts_A, fill = TRUE)
all_forecasts_B <- rbindlist(forecasts_B, fill = TRUE)
cat("✓ Forecast data extracted\n")

# --- Forecast vs Actual Plots ---
cat("\n\nPLOT 1: FORECAST VS ACTUAL COMPARISON\n\n")

# Ensure facet order matches model_names
all_forecasts_A[, Model := factor(Model, levels = model_names)]
all_forecasts_B[, Model := factor(Model, levels = model_names)]

# Convert numeric Time to POSIXct datetime using real period start dates
start_A <- as.POSIXct("2024-10-01 01:00:00", tz = "UTC")
start_B <- as.POSIXct("2024-12-18 01:00:00", tz = "UTC")
all_forecasts_A[, Datetime := start_A + as.difftime(Time - 1, units = "hours")]
all_forecasts_B[, Datetime := start_B + as.difftime(Time - 1, units = "hours")]

plot_fc <- function(df, title) {
  ggplot(df, aes(x = Datetime)) +
    geom_line(aes(y = Actual), color = "black", linewidth = 0.8, alpha = 0.8) +
    geom_line(aes(y = Forecast, color = Model), linewidth = 0.8, alpha = 0.9) +
    facet_wrap(~Model, ncol = 1, scales = "fixed") +
    scale_color_manual(values = c(
      "SARIMA" = "#E41A1C", "SARIMAX" = "#377EB8",
      "LSTM" = "#4DAF4A", "HYBRID" = "#984EA3")) +
    scale_x_datetime(date_labels = "%d %b", date_breaks = "2 days") +
    labs(
      x = "Date",
      y = "Energy Consumption (kWh)"
    ) +
    theme(legend.position = "none",
          strip.text = element_text(face = "bold", size = 9))
}

print(plot_fc(all_forecasts_A, "Forecast vs Actual – Period A (Oct 1 - Oct 10)"))
print(plot_fc(all_forecasts_B, "Forecast vs Actual – Period B (Dec 18 - Dec 31)"))


# --- Error Metrics ---
cat("\n\nCalculating forecast errors...\n")
for (df in list(all_forecasts_A, all_forecasts_B)) {
  df[, Error := Actual - Forecast]
  df[, AbsError := abs(Error)]
  df[, SquaredError := Error^2]
  df[, PercentError := 100 * abs(Error) / Actual]
}

# --- Error Distributions ---
cat("\n\nPLOT 2: ERROR DISTRIBUTION COMPARISON\n\n")
plot_error_density <- function(df, title) {
  ggplot(df, aes(x = Error, fill = Model, color = Model)) +
    geom_density(alpha = 0.3, linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    scale_fill_brewer(palette = "Set1") + scale_color_brewer(palette = "Set1") +
    labs(x = "Forecast Error (kWh)", y = "Density")
}

print(plot_error_density(all_forecasts_A, "Error Distribution - Period A"))
print(plot_error_density(all_forecasts_B, "Error Distribution - Period B"))
# --- Error Distributions with Gaussian Comparison ---
cat("\n\nPLOT 2: ERROR DISTRIBUTION COMPARISON WITH GAUSSIAN\n\n")

plot_error_density <- function(df, title) {
  # Calculate statistics for each model
  model_stats <- df[, .(
    mean_err = mean(Error),
    sd_err = sd(Error)
  ), by = Model]
  
  p <- ggplot(df, aes(x = Error, fill = Model, color = Model)) +
    geom_density(alpha = 0.3, linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    scale_fill_brewer(palette = "Set1") + 
    scale_color_brewer(palette = "Set1") +
    labs(x = "Forecast Error (kWh)", y = "Density")
  
  # Add Gaussian overlays for each model
  colors <- RColorBrewer::brewer.pal(max(3, length(unique(df$Model))), "Set1")
  
  # Get the range for x-axis
  x_range <- range(df$Error)
  x_vals <- seq(x_range[1], x_range[2], length.out = 200)
  
  # Create data frame for Gaussian curves
  gaussian_data <- data.frame()
  for (i in seq_len(nrow(model_stats))) {
    model_name <- model_stats$Model[i]
    mean_val <- model_stats$mean_err[i]
    sd_val <- model_stats$sd_err[i]
    
    temp_df <- data.frame(
      x = x_vals,
      y = dnorm(x_vals, mean = mean_val, sd = sd_val),
      Model = model_name,
      Type = "Gaussian"
    )
    gaussian_data <- rbind(gaussian_data, temp_df)
  }
  
  # Add Gaussian curves as geom_line
  p <- p + geom_line(
    data = gaussian_data,
    aes(x = x, y = y, color = Model),
    linetype = "dotted",
    linewidth = 0.8,
    inherit.aes = FALSE
  )
  
  return(p)
}

# Create plots
print(plot_error_density(all_forecasts_A, "Error Distribution - Period A"))
print(plot_error_density(all_forecasts_B, "Error Distribution - Period B"))

# --- Normality Tests ---
cat("\n\nNORMALITY TESTS\n")
cat(rep("=", 60), "\n")

test_normality <- function(df, period_name) {
  cat("\n", period_name, "\n")
  cat(rep("-", 40), "\n")
  
  for (model in unique(df$Model)) {
    errors <- df[Model == model, Error]
    
    cat("\nModel:", model, "\n")
    cat("  Mean:", round(mean(errors), 2), "\n")
    cat("  SD:", round(sd(errors), 2), "\n")
    cat("  Skewness:", round(e1071::skewness(errors), 3), "\n")
    cat("  Kurtosis:", round(e1071::kurtosis(errors), 3), "\n")
    
    # Shapiro-Wilk test (for n < 5000)
    if (length(errors) < 5000) {
      sw_test <- shapiro.test(errors)
      cat("  Shapiro-Wilk p-value:", round(sw_test$p.value, 4), "\n")
    }
    
    # Kolmogorov-Smirnov test
    ks_test <- ks.test(errors, "pnorm", mean(errors), sd(errors))
    cat("  KS test p-value:", round(ks_test$p.value, 4), "\n")
    
    # Anderson-Darling test
    ad_test <- nortest::ad.test(errors)
    cat("  Anderson-Darling p-value:", round(ad_test$p.value, 4), "\n")
  }
}

# Ensure required packages are loaded
if (!require("e1071")) install.packages("e1071")
if (!require("nortest")) install.packages("nortest")

test_normality(all_forecasts_A, "Period A")
test_normality(all_forecasts_B, "Period B")

# --- Q-Q Plots ---
cat("\n\nPLOT 3: Q-Q PLOTS\n\n")

plot_qq <- function(df, title) {
  ggplot(df, aes(sample = Error, color = Model)) +
    stat_qq() +
    stat_qq_line(linetype = "dashed") +
    facet_wrap(~Model, scales = "free") +
    scale_color_brewer(palette = "Set1") +
    labs( x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_minimal()
}

print(plot_qq(all_forecasts_A, "Q-Q Plots - Period A"))
print(plot_qq(all_forecasts_B, "Q-Q Plots - Period B"))

# --- Accuracy by Hour of Day ---
cat("\n\nFORECAST ACCURACY BY HOUR OF DAY\n\n")
for (df in list(all_forecasts_A, all_forecasts_B)) df[, Hour := ((Time - 1) %% 24) + 1]
calc_hourly <- function(df) df[, .(
  N = .N, RMSE = sqrt(mean(SquaredError, na.rm = TRUE)),
  MAE = mean(AbsError, na.rm = TRUE),
  MAPE = mean(PercentError, na.rm = TRUE)
), by = .(Model, Hour)][order(Model, Hour)]
by_hour_A <- calc_hourly(all_forecasts_A)
by_hour_B <- calc_hourly(all_forecasts_B)
print(kable(by_hour_A, format = "pipe", digits = 2))
print(kable(by_hour_B, format = "pipe", digits = 2))

plot_hour <- function(df, title) {
  ggplot(df, aes(x = Hour, y = MAPE, color = Model, group = Model)) +
    geom_line(linewidth = 1, alpha = 0.8) + geom_point(size = 2) +
    scale_color_brewer(palette = "Set1") +
    scale_x_continuous(breaks = seq(1, 24, 2)) +
    labs(x = "Hour of Day", y = "MAPE (%)") +
    theme_minimal(base_size = 12) 
}
print(plot_hour(by_hour_A, "Forecast Accuracy by Hour of Day - Period A"))
print(plot_hour(by_hour_B, "Forecast Accuracy by Hour of Day - Period B"))

# --- Accuracy by Day of Week ---
cat("\n\nFORECAST ACCURACY BY DAY OF WEEK\n\n")
dow_labels <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
for (df in list(all_forecasts_A, all_forecasts_B)) {
  df[, Day := ceiling(Time / 24)]
  df[, DayOfWeek := ((Day - 1) %% 7) + 1]
}
calc_dow <- function(df) df[, .(
  N = .N, RMSE = sqrt(mean(SquaredError, na.rm = TRUE)),
  MAE = mean(AbsError, na.rm = TRUE),
  MAPE = mean(PercentError, na.rm = TRUE)
), by = .(Model, DayOfWeek)][, DayName := dow_labels[DayOfWeek]]
by_dow_A <- calc_dow(all_forecasts_A)
by_dow_B <- calc_dow(all_forecasts_B)
print(kable(by_dow_A[, .(Model, DayName, RMSE, MAE, MAPE)], format = "pipe", digits = 2))
print(kable(by_dow_B[, .(Model, DayName, RMSE, MAE, MAPE)], format = "pipe", digits = 2))

plot_dow <- function(df, title) {
  df$DayName <- factor(df$DayName, levels = dow_labels)
  ggplot(df, aes(x = DayName, y = MAPE, color = Model, group = Model)) +
    geom_line(linewidth = 1, alpha = 0.8) + geom_point(size = 2) +
    scale_color_brewer(palette = "Set1") +
    labs(x = "Day of Week", y = "MAPE (%)") +
    theme_minimal(base_size = 12)
}
print(plot_dow(by_dow_A, "Forecast Accuracy by Day of Week - Period A"))
print(plot_dow(by_dow_B, "Forecast Accuracy by Day of Week - Period B"))

# --- Plot RMSE and MAE by Hour of Day ---
plot_hour_metric <- function(df, metric, title, ylab) {
  ggplot(df, aes_string(x = "Hour", y = metric, color = "Model", group = "Model")) +
    geom_line(linewidth = 1, alpha = 0.8) +
    geom_point(size = 2) +
    scale_color_brewer(palette = "Set1") +
    scale_x_continuous(breaks = seq(1, 24, 2)) +
    labs(x = "Hour of Day", y = ylab, title = title) +
    theme_minimal(base_size = 12)
}

print(plot_hour_metric(by_hour_A, "RMSE", "RMSE by Hour of Day – Period A", "RMSE (kWh)"))
print(plot_hour_metric(by_hour_B, "RMSE", "RMSE by Hour of Day – Period B", "RMSE (kWh)"))
print(plot_hour_metric(by_hour_A, "MAE", "MAE by Hour of Day – Period A", "MAE (kWh)"))
print(plot_hour_metric(by_hour_B, "MAE", "MAE by Hour of Day – Period B", "MAE (kWh)"))

# --- Plot RMSE and MAE by Day of Week ---
plot_dow_metric <- function(df, metric, title, ylab) {
  df$DayName <- factor(df$DayName, levels = dow_labels)
  ggplot(df, aes_string(x = "DayName", y = metric, color = "Model", group = "Model")) +
    geom_line(linewidth = 1, alpha = 0.8) +
    geom_point(size = 2) +
    scale_color_brewer(palette = "Set1") +
    labs(x = "Day of Week", y = ylab, title = title) +
    theme_minimal(base_size = 12)
}

print(plot_dow_metric(by_dow_A, "RMSE", "RMSE by Day of Week – Period A", "RMSE (kWh)"))
print(plot_dow_metric(by_dow_B, "RMSE", "RMSE by Day of Week – Period B", "RMSE (kWh)"))
print(plot_dow_metric(by_dow_A, "MAE", "MAE by Day of Week – Period A", "MAE (kWh)"))
print(plot_dow_metric(by_dow_B, "MAE", "MAE by Day of Week – Period B", "MAE (kWh)"))

library(ggplot2)
library(patchwork)

# ============================================================
# Use your global theme
# ============================================================
theme_set(theme_bw(base_size = 11) +
            theme(panel.grid.minor = element_blank(),
                  plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
                  strip.background = element_rect(fill = "grey92"),
                  strip.text = element_text(face = "bold", size = 10),
                  legend.position = "bottom"))

# ============================================================
# Helper plotting functions
# ============================================================
plot_hour_metric <- function(df, metric, subtitle) {
  ggplot(df, aes_string(x = "Hour", y = metric, color = "Model", group = "Model")) +
    geom_line(linewidth = 1, alpha = 0.8) +
    geom_point(size = 2) +
    scale_color_brewer(palette = "Set1") +
    scale_x_continuous(breaks = seq(1, 24, 3)) +
    labs(x = "Hour of Day", y = paste0(metric, " (kWh)"), subtitle = subtitle) +
    theme(legend.position = "bottom",
          plot.subtitle = element_text(hjust = 0.5, face = "bold"))
}

plot_dow_metric <- function(df, metric, subtitle, dow_labels) {
  df$DayName <- factor(df$DayName, levels = dow_labels)
  ggplot(df, aes_string(x = "DayName", y = metric, color = "Model", group = "Model")) +
    geom_line(linewidth = 1, alpha = 0.8) +
    geom_point(size = 2) +
    scale_color_brewer(palette = "Set1") +
    labs(x = "Day of Week", y = paste0(metric, " (kWh)"), subtitle = subtitle) +
    theme(legend.position = "bottom",
          plot.subtitle = element_text(hjust = 0.5, face = "bold"))
}

# ============================================================
# Define day labels
# ============================================================
dow_labels <- c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")

# ============================================================
# FIGURE 1: Hour of Day (Period A + Period B side by side)
# ============================================================
p_hour_A_MAE <- plot_hour_metric(by_hour_A, "MAE", "Period A")
p_hour_B_MAE <- plot_hour_metric(by_hour_B, "MAE", "Period B")

mae_hour_fig <- p_hour_A_MAE + p_hour_B_MAE +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

# ============================================================
# FIGURE 2: Day of Week (Period A + Period B side by side)
# ============================================================
p_dow_A_MAE <- plot_dow_metric(by_dow_A, "MAE", "Period A", dow_labels)
p_dow_B_MAE <- plot_dow_metric(by_dow_B, "MAE", "Period B", dow_labels)

mae_dow_fig <- p_dow_A_MAE + p_dow_B_MAE +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

# ============================================================
# Display or save
# ============================================================
print(mae_hour_fig)
print(mae_dow_fig)

# --- Diebold-Mariano Test for Forecast Comparison ---

# Function to perform DM test between two models
perform_dm_test <- function(actual, forecast1, forecast2, 
                            alternative = "two.sided", 
                            h = 24) {
  # Calculate forecast errors
  e1 <- actual - forecast1
  e2 <- actual - forecast2
  
  # Perform DM test with squared error loss
  dm_result <- dm.test(e1, e2, alternative = alternative, h = h, loss)
  
  return(dm_result)
}


# ============================================================================
# Period A Tests
# ============================================================================
cat("\n=== PERIOD A (Stable) ===\n")

# Extract forecasts for Period A
actual_A <- SARIMA$period_A$forecasts$Actual
sarima_A <- SARIMA$period_A$forecasts$Forecast
sarimax_A <- SARIMAX$period_A$forecasts$Forecast
lstm_A <- LSTM$period_A$forecasts$Forecast
hybrid_A <- HYBRID$period_A$forecasts$Forecast

# LSTM vs SARIMA
cat("\nLSTM vs SARIMA:\n")
dm_lstm_sarima_A <- perform_dm_test(actual_A, lstm_A, sarima_A)
print(dm_lstm_sarima_A)

# LSTM vs SARIMAX
cat("\nLSTM vs SARIMAX:\n")
dm_lstm_sarimax_A <- perform_dm_test(actual_A, lstm_A, sarimax_A)
print(dm_lstm_sarimax_A)

# LSTM vs Hybrid
cat("\nLSTM vs Hybrid:\n")
dm_lstm_hybrid_A <- perform_dm_test(actual_A, lstm_A, hybrid_A)
print(dm_lstm_hybrid_A)

# SARIMAX vs SARIMA
cat("\nSARIMAX vs SARIMA:\n")
dm_sarimax_sarima_A <- perform_dm_test(actual_A, sarimax_A, sarima_A)
print(dm_sarimax_sarima_A)

# Hybrid vs SARIMAX
cat("\nHybrid vs SARIMAX:\n")
dm_hybrid_sarimax_A <- perform_dm_test(actual_A, hybrid_A, sarimax_A)
print(dm_hybrid_sarimax_A)

# ============================================================================
# Period B Tests
# ============================================================================
cat("\n=== PERIOD B (Volatile) ===\n")

# Extract forecasts for Period B
actual_B <- SARIMA$period_B$forecasts$Actual
sarima_B <- SARIMA$period_B$forecasts$Forecast
sarimax_B <- SARIMAX$period_B$forecasts$Forecast
lstm_B <- LSTM$period_B$forecasts$Forecast
hybrid_B <- HYBRID$period_B$forecasts$Forecast

# LSTM vs SARIMA
cat("\nLSTM vs SARIMA:\n")
dm_lstm_sarima_B <- perform_dm_test(actual_B, lstm_B, sarima_B)
print(dm_lstm_sarima_B)

# LSTM vs SARIMAX
cat("\nLSTM vs SARIMAX:\n")
dm_lstm_sarimax_B <- perform_dm_test(actual_B, lstm_B, sarimax_B)
print(dm_lstm_sarimax_B)

# LSTM vs Hybrid
cat("\nLSTM vs Hybrid:\n")
dm_lstm_hybrid_B <- perform_dm_test(actual_B, lstm_B, hybrid_B)
print(dm_lstm_hybrid_B)

# SARIMAX vs SARIMA
cat("\nSARIMAX vs SARIMA:\n")
dm_sarimax_sarima_B <- perform_dm_test(actual_B, sarimax_B, sarima_B)
print(dm_sarimax_sarima_B)

# Hybrid vs SARIMAX
cat("\nHybrid vs SARIMAX:\n")
dm_hybrid_sarimax_B <- perform_dm_test(actual_B, hybrid_B, sarimax_B)
print(dm_hybrid_sarimax_B)

# ============================================================================
# Create summary table
# ============================================================================
create_dm_summary <- function(dm_test) {
  data.table(
    Statistic = round(dm_test$statistic, 3),
    p_value = round(dm_test$p.value, 4),
    Significant = ifelse(dm_test$p.value < 0.05, "Yes*", 
                         ifelse(dm_test$p.value < 0.10, "Yes†", "No"))
  )
}

cat("\n=== SUMMARY TABLE ===\n")
dm_summary <- data.table(
  Comparison = c(
    "LSTM vs SARIMA (A)", "LSTM vs SARIMAX (A)", "LSTM vs Hybrid (A)",
    "SARIMAX vs SARIMA (A)", "Hybrid vs SARIMAX (A)",
    "LSTM vs SARIMA (B)", "LSTM vs SARIMAX (B)", "LSTM vs Hybrid (B)",
    "SARIMAX vs SARIMA (B)", "Hybrid vs SARIMAX (B)"
  ),
  DM_Stat = c(
    dm_lstm_sarima_A$statistic, dm_lstm_sarimax_A$statistic, 
    dm_lstm_hybrid_A$statistic, dm_sarimax_sarima_A$statistic,
    dm_hybrid_sarimax_A$statistic,
    dm_lstm_sarima_B$statistic, dm_lstm_sarimax_B$statistic,
    dm_lstm_hybrid_B$statistic, dm_sarimax_sarima_B$statistic,
    dm_hybrid_sarimax_B$statistic
  ),
  p_value = c(
    dm_lstm_sarima_A$p.value, dm_lstm_sarimax_A$p.value,
    dm_lstm_hybrid_A$p.value, dm_sarimax_sarima_A$p.value,
    dm_hybrid_sarimax_A$p.value,
    dm_lstm_sarima_B$p.value, dm_lstm_sarimax_B$p.value,
    dm_lstm_hybrid_B$p.value, dm_sarimax_sarima_B$p.value,
    dm_hybrid_sarimax_B$p.value
  )
)

dm_summary[, `:=`(
  DM_Stat = round(DM_Stat, 3),
  p_value = round(p_value, 4),
  Significant = ifelse(p_value < 0.01, "***",
                       ifelse(p_value < 0.05, "**",
                              ifelse(p_value < 0.10, "*", "")))
)]

print(dm_summary)

# Note: Negative DM statistic means first model is better
# Significance: *** p<0.01, ** p<0.05, * p<0.10