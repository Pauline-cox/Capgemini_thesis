# ==============================================================================
# COMPREHENSIVE MODEL EVALUATION SCRIPT
# Separates Residual Diagnostics (for model development) from 
# Forecast Error Analysis (for comparative evaluation)
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

# ==============================================================================
# PART 1: RESIDUAL DIAGNOSTICS (FOR MODEL DEVELOPMENT SECTIONS)
# Use training residuals to validate model specification
# ==============================================================================

cat("\n\n", paste(rep("=", 80), collapse = ""), "\n")
cat("PART 1: RESIDUAL DIAGNOSTICS (Training Data)\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

# --- Extract Training Residuals ---
cat("Extracting training residuals from fitted models...\n")

extract_residuals <- function(model_obj, model_name) {
  # Check different possible locations for residuals
  residuals <- NULL
  
  if (!is.null(model_obj$model)) {
    # For SARIMA/SARIMAX: residuals() function on fitted model
    if (any(class(model_obj$model) %in% c("Arima", "forecast_ARIMA"))) {
      residuals <- as.numeric(residuals(model_obj$model))
    }
  }
  
  if (!is.null(model_obj$training_residuals)) {
    # If explicitly stored
    residuals <- model_obj$training_residuals
  }
  
  if (!is.null(model_obj$history)) {
    # For LSTM: training history/residuals if stored
    if (!is.null(model_obj$history$residuals)) {
      residuals <- model_obj$history$residuals
    }
  }
  
  if (!is.null(residuals)) {
    return(data.table(
      Time = seq_along(residuals),
      Residual = residuals,
      Model = model_name
    ))
  } else {
    cat("Warning: No training residuals found for", model_name, "\n")
    return(NULL)
  }
}

residuals_list <- lapply(seq_along(models), function(i) {
  extract_residuals(models[[i]], model_names[i])
})
names(residuals_list) <- model_names

# Filter out NULL results
residuals_list <- residuals_list[!sapply(residuals_list, is.null)]

if (length(residuals_list) > 0) {
  all_residuals <- rbindlist(residuals_list, fill = TRUE)
  
  # --- Residual Diagnostics Table ---
  cat("\n\nTABLE 1: TRAINING RESIDUAL DIAGNOSTICS\n")
  cat("(For validating model specification during development)\n\n")
  
  calc_residual_diagnostics <- function(df) {
    df[, .(
      N         = .N,
      Mean      = mean(Residual, na.rm = TRUE),
      SD        = sd(Residual, na.rm = TRUE),
      Skewness  = skewness(Residual, na.rm = TRUE),
      Kurtosis  = kurtosis(Residual, na.rm = TRUE),
      JB_p      = tryCatch(jarque.bera.test(Residual)$p.value, error = function(e) NA),
      LB_p      = tryCatch(Box.test(Residual, lag = 24, type = "Ljung-Box")$p.value, error = function(e) NA)
    ), by = Model][order(Model)]
  }
  
  residual_diagnostics <- calc_residual_diagnostics(all_residuals)
  
  # Format for display
  residual_diagnostics_fmt <- residual_diagnostics[, .(
    Model,
    N         = sprintf("%d", N),
    Mean      = sprintf("%.2f", Mean),
    SD        = sprintf("%.2f", SD),
    Skewness  = sprintf("%.3f", Skewness),
    Kurtosis  = sprintf("%.3f", Kurtosis),
    JB_p      = ifelse(JB_p < 0.001, "<0.001", sprintf("%.3f", JB_p)),
    LB_p      = ifelse(LB_p < 0.001, "<0.001", sprintf("%.3f", LB_p))
  )]
  
  print(kable(residual_diagnostics_fmt, format = "pipe"))
  
  cat("\nInterpretation:")
  cat("\n- Mean near 0: unbiased model")
  cat("\n- Skewness near 0: symmetric residuals")
  cat("\n- Kurtosis near 3: normal-like tails")
  cat("\n- JB p > 0.05: residuals are normal (good)")
  cat("\n- LB p > 0.05: no autocorrelation (good)\n\n")
  
  # --- Residual Distribution Plots ---
  cat("\n\nPLOT 1: TRAINING RESIDUAL DISTRIBUTIONS\n")
  cat("(Should be approximately normal with mean near zero)\n\n")
  
  plot_residual_density <- function(df) {
    ggplot(df, aes(x = Residual, fill = Model, color = Model)) +
      geom_density(alpha = 0.3, linewidth = 0.8) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
      scale_fill_brewer(palette = "Set1") + 
      scale_color_brewer(palette = "Set1") +
      labs(
        title = "Training Residual Distributions",
        x = "Residual (kWh)", 
        y = "Density"
      )
  }
  
  print(plot_residual_density(all_residuals))
  
  # --- ACF/PACF Plots ---
  cat("\n\nPLOT 2: ACF/PACF OF TRAINING RESIDUALS\n")
  cat("(Should show no significant autocorrelation for well-specified models)\n\n")
  
  # Create ACF/PACF plots for each model
  for (model_name in names(residuals_list)) {
    model_residuals <- residuals_list[[model_name]]$Residual
    
    if (length(model_residuals) > 50) {  # Need sufficient data for ACF/PACF
      cat(paste0("\n", model_name, ":\n"))
      
      par(mfrow = c(1, 2))
      acf(model_residuals, main = paste(model_name, "- ACF"), lag.max = 48)
      pacf(model_residuals, main = paste(model_name, "- PACF"), lag.max = 48)
      par(mfrow = c(1, 1))
    }
  }
  
  # --- QQ Plots ---
  cat("\n\nPLOT 3: Q-Q PLOTS OF TRAINING RESIDUALS\n")
  cat("(Points should follow diagonal line for normal residuals)\n\n")
  
  plot_qq <- function(df) {
    ggplot(df, aes(sample = Residual, color = Model)) +
      stat_qq() +
      stat_qq_line() +
      facet_wrap(~Model, scales = "free") +
      scale_color_brewer(palette = "Set1") +
      labs(
        title = "Q-Q Plots: Training Residuals vs Normal Distribution",
        x = "Theoretical Quantiles",
        y = "Sample Quantiles"
      ) +
      theme(legend.position = "none")
  }
  
  print(plot_qq(all_residuals))
  
  # --- Residuals vs Fitted ---
  cat("\n\nPLOT 4: RESIDUALS VS TIME\n")
  cat("(Should show random scatter with constant variance)\n\n")
  
  plot_residuals_time <- function(df) {
    ggplot(df, aes(x = Time, y = Residual, color = Model)) +
      geom_point(alpha = 0.4, size = 0.8) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
      geom_smooth(method = "loess", se = FALSE, linewidth = 0.8) +
      facet_wrap(~Model, scales = "free_x") +
      scale_color_brewer(palette = "Set1") +
      labs(
        title = "Training Residuals Over Time",
        x = "Time Index",
        y = "Residual (kWh)"
      ) +
      theme(legend.position = "none")
  }
  
  print(plot_residuals_time(all_residuals))
  
} else {
  cat("WARNING: No training residuals could be extracted from model objects.\n")
  cat("Make sure your models save residuals during training.\n\n")
}

# ==============================================================================
# PART 2: FORECAST ERROR ANALYSIS (FOR COMPARATIVE EVALUATION)
# Use out-of-sample forecast errors to compare operational performance
# ==============================================================================

cat("\n\n", paste(rep("=", 80), collapse = ""), "\n")
cat("PART 2: FORECAST ERROR ANALYSIS (Test Data)\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

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
cat("\n\nPLOT 5: FORECAST VS ACTUAL COMPARISON\n\n")

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
      title = title,
      x = "Date",
      y = "Energy Consumption (kWh)"
    ) +
    theme(legend.position = "none",
          strip.text = element_text(face = "bold", size = 9))
}

print(plot_fc(all_forecasts_A, "Forecast vs Actual – Period A (Oct 1 - Oct 10)"))
print(plot_fc(all_forecasts_B, "Forecast vs Actual – Period B (Dec 18 - Dec 31)"))

# --- Calculate Forecast Errors ---
cat("\n\nCalculating forecast errors...\n")
for (df in list(all_forecasts_A, all_forecasts_B)) {
  df[, Error := Actual - Forecast]
  df[, AbsError := abs(Error)]
  df[, SquaredError := Error^2]
  df[, PercentError := 100 * abs(Error) / Actual]
}

# --- Forecast Error Distributions ---
cat("\n\nPLOT 6: FORECAST ERROR DISTRIBUTION COMPARISON\n\n")
plot_error_density <- function(df, title) {
  ggplot(df, aes(x = Error, fill = Model, color = Model)) +
    geom_density(alpha = 0.3, linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    scale_fill_brewer(palette = "Set1") + scale_color_brewer(palette = "Set1") +
    labs(
      title = title,
      x = "Forecast Error (kWh)", 
      y = "Density"
    )
}

print(plot_error_density(all_forecasts_A, "Forecast Error Distribution - Period A"))
print(plot_error_density(all_forecasts_B, "Forecast Error Distribution - Period B"))

# --- Accuracy by Hour of Day ---
cat("\n\nPLOT 7: FORECAST ACCURACY BY HOUR OF DAY\n\n")
for (df in list(all_forecasts_A, all_forecasts_B)) df[, Hour := ((Time - 1) %% 24) + 1]
calc_hourly <- function(df) df[, .(
  N = .N, RMSE = sqrt(mean(SquaredError, na.rm = TRUE)),
  MAE = mean(AbsError, na.rm = TRUE),
  MAPE = mean(PercentError, na.rm = TRUE)
), by = .(Model, Hour)][order(Model, Hour)]
by_hour_A <- calc_hourly(all_forecasts_A)
by_hour_B <- calc_hourly(all_forecasts_B)

plot_hour <- function(df, title) {
  ggplot(df, aes(x = Hour, y = MAPE, color = Model, group = Model)) +
    geom_line(linewidth = 1, alpha = 0.8) + geom_point(size = 2) +
    scale_color_brewer(palette = "Set1") +
    scale_x_continuous(breaks = seq(1, 24, 2)) +
    labs(
      title = title,
      x = "Hour of Day", 
      y = "MAPE (%)"
    ) +
    theme_minimal(base_size = 12) 
}
print(plot_hour(by_hour_A, "Forecast Accuracy by Hour of Day - Period A"))
print(plot_hour(by_hour_B, "Forecast Accuracy by Hour of Day - Period B"))

# --- Accuracy by Day of Week ---
cat("\n\nPLOT 8: FORECAST ACCURACY BY DAY OF WEEK\n\n")
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

plot_dow <- function(df, title) {
  df$DayName <- factor(df$DayName, levels = dow_labels)
  ggplot(df, aes(x = DayName, y = MAPE, color = Model, group = Model)) +
    geom_line(linewidth = 1, alpha = 0.8) + geom_point(size = 2) +
    scale_color_brewer(palette = "Set1") +
    labs(
      title = title,
      x = "Day of Week", 
      y = "MAPE (%)"
    ) +
    theme_minimal(base_size = 12)
}
print(plot_dow(by_dow_A, "Forecast Accuracy by Day of Week - Period A"))
print(plot_dow(by_dow_B, "Forecast Accuracy by Day of Week - Period B"))

# ==============================================================================
# SUMMARY
# ==============================================================================

cat("\n\n", paste(rep("=", 80), collapse = ""), "\n")
cat("EVALUATION COMPLETE\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")
cat("PART 1 outputs (for Model Development sections):\n")
cat("  - Training residual diagnostics table\n")
cat("  - Residual distribution plots\n")
cat("  - ACF/PACF plots\n")
cat("  - Q-Q plots\n")
cat("  - Residuals vs time plots\n\n")
cat("PART 2 outputs (for Comparative Analysis section):\n")
cat("  - Performance comparison table\n")
cat("  - Forecast vs actual plots\n")
cat("  - Forecast error distributions\n")
cat("  - Hourly/daily accuracy patterns\n\n")