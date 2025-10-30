# ==============================================================
# Author: Pauline Cox
# Script: evaluation_forecasts.R
#
# Description: Comprehensive evaluation pipeline comparing all
# forecasting models (SARIMA, SARIMAX, LSTM, Hybrid, PCA, Cluster). 
# Produces unified tables and figures for performance metrics, 
# forecast visualization, error distributions, temporal patterns, 
# and pairwise Diebold–Mariano significance testing.
#
# Input:
#   - Saved model result files
#
# Output:
#   - Performance comparison table
#   - Forecast vs actual plots (Periods A and B)
#   - Error distribution and density plots
#   - Hourly and daily accuracy profiles
#   - Diebold–Mariano significance results
# ==============================================================

# --- Global theme ---
theme_set(theme_bw(base_size = 11) +
            theme(
              panel.grid.minor = element_blank(),
              plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
              strip.background = element_rect(fill = "grey92"),
              strip.text = element_text(face = "bold", size = 10),
              legend.position = "bottom"
            ))

# --- Color palette and model order ---
model_colors <- c(
  SARIMA          = "#E41A1C", 
  SARIMAX         = "#377EB8", 
  LSTM            = "#4DAF4A",  
  HYBRID          = "#984EA3", 
  SARIMAX_CLUSTER = "#D95F02",
  SARIMAX_PCA     = "#1B9E77"  
)
model_order <- c("SARIMA", "SARIMAX", "LSTM", "HYBRID", "SARIMAX_CLUSTER", "SARIMAX_PCA")

# --- Helpers ---
safe_read <- function(path) {
  if (!file.exists(path)) stop(sprintf("File not found: %s", path))
  readRDS(path)
}
msg <- function(...) cat(sprintf(...), "\n")

# --- Load results ---
models <- list(
  SARIMA          = safe_read("RESULTS/RESULTS_FC_SARIMA.rds"),
  SARIMAX         = safe_read("RESULTS/RESULTS_FC_SARIMAX.rds"),
  LSTM            = safe_read("RESULTS/RESULTS_FC_LSTM.rds"),
  HYBRID          = safe_read("RESULTS/RESULTS_FC_HYBRID.rds"),
  SARIMAX_CLUSTER = safe_read("RESULTS/RESULTS_FC_CLUSTER.rds"),
  SARIMAX_PCA     = safe_read("RESULTS/RESULTS_FC_PCA.rds")
)
model_names <- names(models)


# --- Performance table ---

msg("\nForecast performance comparison\n")
perf_list <- lapply(model_names, function(nm) {
  if (!is.null(models[[nm]]$evaluations)) {
    dt <- as.data.table(models[[nm]]$evaluations)
    dt[, Model := nm]
    return(dt)
  }
})
performance <- rbindlist(perf_list, fill = TRUE)
performance[, Period_Clean := gsub("Period | \\(.*\\)", "", Period)]

perf_long <- melt(
  performance,
  id.vars = c("Model", "Period_Clean"),
  measure.vars = c("RMSE", "MAE", "R2", "MAPE"),
  variable.name = "Metric", value.name = "Value"
)
perf_wide <- dcast(perf_long, Model ~ Metric + Period_Clean, value.var = "Value")

print(kable(perf_wide, format = "pipe", digits = 3))

# --- Extract forecast ---
extract_fc <- function(model, name, period) {
  fc <- model[[period]]$forecasts
  if (is.null(fc)) return(NULL)
  dt <- as.data.table(fc)
  dt[, Model := name]
}

forecasts_A <- lapply(model_names, \(nm) extract_fc(models[[nm]], nm, "period_A"))
forecasts_B <- lapply(model_names, \(nm) extract_fc(models[[nm]], nm, "period_B"))
all_forecasts_A <- rbindlist(forecasts_A, fill = TRUE)
all_forecasts_B <- rbindlist(forecasts_B, fill = TRUE)

# Ensure consistent order
all_forecasts_A[, Model := factor(Model, levels = model_order)]
all_forecasts_B[, Model := factor(Model, levels = model_order)]

# --- Plot forecasts vs actual ---
start_A <- as.POSIXct("2024-10-01 01:00:00", tz = "UTC")
start_B <- as.POSIXct("2024-12-18 01:00:00", tz = "UTC")
all_forecasts_A[, Datetime := start_A + as.difftime(Time - 1, units = "hours")]
all_forecasts_B[, Datetime := start_B + as.difftime(Time - 1, units = "hours")]

plot_fc <- function(df, title) {
  ggplot(df, aes(x = Datetime)) +
    geom_line(aes(y = Actual), color = "black", linewidth = 0.8, alpha = 0.8) +
    geom_line(aes(y = Forecast, color = Model), linewidth = 0.8, alpha = 0.9) +
    facet_wrap(~Model, ncol = 1, scales = "fixed") +
    scale_color_manual(values = model_colors) +
    scale_x_datetime(date_labels = "%d %b", date_breaks = "2 days") +
    labs(title = title, x = "Date", y = "Energy Consumption (kWh)") +
    theme(legend.position = "none",
          strip.text = element_text(face = "bold", size = 9))
}
print(plot_fc(all_forecasts_A, "Forecast vs Actual – Period A"))
print(plot_fc(all_forecasts_B, "Forecast vs Actual – Period B"))

# --- Error metrics and distribution --- 
add_err_cols <- function(df) {
  df[, Error := Actual - Forecast]
  df[, AbsError := abs(Error)]
  df[, SquaredError := Error^2]
  df[, PercentError := fifelse(Actual == 0, NA_real_, 100 * abs(Error) / abs(Actual))]
  df
}
all_forecasts_A <- add_err_cols(all_forecasts_A)
all_forecasts_B <- add_err_cols(all_forecasts_B)

plot_error_overlay <- function(df) {
  stats <- df[, .(mean_err = mean(Error, na.rm = TRUE), sd_err = sd(Error, na.rm = TRUE)), by = Model]
  x_lim <- max(abs(range(df$Error, na.rm = TRUE)))
  x_vals <- seq(-x_lim, x_lim, length.out = 400)
  gauss <- rbindlist(lapply(1:nrow(stats), function(i)
    data.table(x = x_vals, y = dnorm(x_vals, stats$mean_err[i], stats$sd_err[i]), Model = stats$Model[i])
  ))
  ggplot(df, aes(x = Error, fill = Model, color = Model)) +
    geom_density(alpha = 0.25, linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.6) +
    geom_line(data = gauss, aes(x = x, y = y, color = Model),
              inherit.aes = FALSE, linetype = "dotted", linewidth = 0.9) +
    scale_color_manual(values = model_colors) +
    scale_fill_manual(values = model_colors) +
    coord_cartesian(xlim = c(-x_lim, x_lim)) +
    labs(x = "Forecast Error (kWh)", y = "Density") +
    theme_bw(base_size = 11) +
    theme(legend.position = "bottom",
          legend.title = element_blank())
}

plot_error_facet <- function(df) {
  stats <- df[, .(mean_err = mean(Error, na.rm = TRUE), sd_err = sd(Error, na.rm = TRUE)), by = Model]
  x_lim <- max(abs(range(df$Error, na.rm = TRUE)))
  x_vals <- seq(-x_lim, x_lim, length.out = 400)
  gauss <- rbindlist(lapply(1:nrow(stats), function(i)
    data.table(x = x_vals, y = dnorm(x_vals, stats$mean_err[i], stats$sd_err[i]), Model = stats$Model[i])
  ))
  ggplot(df, aes(x = Error)) +
    geom_density(aes(fill = Model), color = "black", alpha = 0.25, linewidth = 0.5) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "grey30", linewidth = 0.6) +
    geom_line(data = gauss, aes(x = x, y = y), color = "red", linetype = "dotted", linewidth = 0.8) +
    facet_wrap(~Model, ncol = 2, scales = "fixed") +
    scale_fill_manual(values = model_colors) +
    coord_cartesian(xlim = c(-x_lim, x_lim)) +
    labs(x = "Forecast Error (kWh)", y = "Density") +
    theme_bw(base_size = 11) +
    theme(strip.text = element_text(face = "bold", size = 9),
          legend.position = "none")
}

print(plot_error_overlay(all_forecasts_A))
print(plot_error_facet(all_forecasts_A))
print(plot_error_overlay(all_forecasts_B))
print(plot_error_facet(all_forecasts_B))

# --- QQ plots of forecast errors ---
plot_qq_facet <- function(df, title) {
  ggplot(df, aes(sample = Error)) +
    stat_qq(aes(color = Model), alpha = 0.5, size = 1) +
    stat_qq_line(aes(color = Model), linetype = "dashed") +
    facet_wrap(~Model, ncol = 2, scales = "fixed") +
    scale_color_manual(values = model_colors) +
    labs(x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_bw(base_size = 11) +
    theme(strip.text = element_text(face = "bold", size = 9),
          legend.position = "none")
}

# Produce QQ plots for both periods
print(plot_qq_facet(all_forecasts_A, "QQ Plot of Forecast Errors – Period A"))
print(plot_qq_facet(all_forecasts_B, "QQ Plot of Forecast Errors – Period B"))

# --- Time-Series Plots of Forecast Errors ---

plot_ts_error <- function(df, period_label) {
  ggplot(df, aes(x = Datetime, y = Error, color = Model)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey40", linewidth = 0.6) +
    geom_line(linewidth = 0.8, alpha = 0.9) +
    facet_wrap(~Model, ncol = 1, scales = "fixed") +
    scale_color_manual(values = model_colors) +
    scale_x_datetime(date_labels = "%d %b", date_breaks = "2 days") +
    labs(
      x = "Date",
      y = "Forecast Error (kWh)"
    ) +
    theme_bw(base_size = 11) +
    theme(
      legend.position = "none",
      strip.text = element_text(face = "bold", size = 9),
      panel.grid.minor = element_blank()
    )
}

# Generate plots for both periods
print(plot_ts_error(all_forecasts_A, "A"))
print(plot_ts_error(all_forecasts_B, "B"))

# --- Hourly/daily error analysis --- 
for (df in list(all_forecasts_A, all_forecasts_B)) df[, Hour := ((Time - 1) %% 24) + 1]
calc_hourly <- function(df) df[, .(
  RMSE = sqrt(mean(SquaredError, na.rm = TRUE)),
  MAE  = mean(AbsError, na.rm = TRUE),
  MAPE = mean(PercentError, na.rm = TRUE)
), by = .(Model, Hour)][order(Model, Hour)]

by_hour_A <- calc_hourly(all_forecasts_A)
by_hour_B <- calc_hourly(all_forecasts_B)

plot_hour <- function(df, title) {
  ggplot(df, aes(x = Hour, y = MAE, color = Model, group = Model)) +
    geom_line(linewidth = 1, alpha = 0.8) +
    geom_point(size = 2) +
    scale_color_manual(values = model_colors) +
    scale_x_continuous(breaks = seq(1, 24, 3)) +
    labs(title = title, x = "Hour of Day", y = "MAPE (%)") +
    theme_minimal(base_size = 12)
}
print(plot_hour(by_hour_A, "Forecast Accuracy by Hour – Period A"))
print(plot_hour(by_hour_B, "Forecast Accuracy by Hour – Period B"))

# --- Hourly/daily error analysis --- 

# Calculate Hour for both datasets
all_forecasts_A[, Hour := ((Time - 1) %% 24) + 1]
all_forecasts_B[, Hour := ((Time - 1) %% 24) + 1]

# Calculate Day of Week for both datasets
all_forecasts_A[, Day := ceiling(Time / 24)]
all_forecasts_A[, DayOfWeek := ((Day - 1) %% 7) + 1]

all_forecasts_B[, Day := ceiling(Time / 24)]
all_forecasts_B[, DayOfWeek := ((Day - 1) %% 7) + 1]

dow_labels <- c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Calculate hourly metrics
calc_hourly <- function(df) df[, .(
  RMSE = sqrt(mean(SquaredError, na.rm = TRUE)),
  MAE  = mean(AbsError, na.rm = TRUE),
  MAPE = mean(PercentError, na.rm = TRUE)
), by = .(Model, Hour)][order(Model, Hour)]

# Calculate daily metrics
calc_daily <- function(df) {
  result <- df[, .(
    RMSE = sqrt(mean(SquaredError, na.rm = TRUE)),
    MAE  = mean(AbsError, na.rm = TRUE),
    MAPE = mean(PercentError, na.rm = TRUE)
  ), by = .(Model, DayOfWeek)]
  result[, DayName := factor(dow_labels[DayOfWeek], levels = dow_labels)]
  result[order(Model, DayOfWeek)]
}

by_hour_A <- calc_hourly(all_forecasts_A)
by_hour_B <- calc_hourly(all_forecasts_B)
by_day_A <- calc_daily(all_forecasts_A)
by_day_B <- calc_daily(all_forecasts_B)

# Hour of Day comparison (Period A vs Period B)
plot_hourly_comparison <- function(hour_A, hour_B) {
  # Determine shared y-axis limits
  y_max <- max(c(hour_A$MAE, hour_B$MAE), na.rm = TRUE)
  y_min <- min(c(hour_A$MAE, hour_B$MAE), na.rm = TRUE)
  
  p_hour_A <- ggplot(hour_A, aes(x = Hour, y = MAE, color = Model, group = Model)) +
    geom_line(linewidth = 1, alpha = 0.8) +
    geom_point(size = 2) +
    scale_color_manual(values = model_colors) +
    scale_x_continuous(breaks = seq(1, 24, 3)) +
    coord_cartesian(ylim = c(y_min, y_max)) +
    labs(title = "Period A", x = "Hour of Day", y = "MAE (kWh)") +
    theme_minimal(base_size = 12)
  
  p_hour_B <- ggplot(hour_B, aes(x = Hour, y = MAE, color = Model, group = Model)) +
    geom_line(linewidth = 1, alpha = 0.8) +
    geom_point(size = 2) +
    scale_color_manual(values = model_colors) +
    scale_x_continuous(breaks = seq(1, 24, 3)) +
    coord_cartesian(ylim = c(y_min, y_max)) +
    labs(title = "Period B", x = "Hour of Day", y = "MAE (kWh)") +
    theme_minimal(base_size = 12)
  
  combined <- p_hour_A + p_hour_B + 
    plot_layout(guides = "collect") & 
    theme(legend.position = "bottom") &
    plot_annotation(theme = theme(plot.title = element_text(face = "bold", hjust = 0.5)))
  
  return(combined)
}

# Day of Week comparison (Period A vs Period B)
plot_daily_comparison <- function(day_A, day_B) {
  # Determine shared y-axis limits
  y_max <- max(c(day_A$MAE, day_B$MAE), na.rm = TRUE)
  y_min <- min(c(day_A$MAE, day_B$MAE), na.rm = TRUE)
  
  p_day_A <- ggplot(day_A, aes(x = DayName, y = MAE, color = Model, group = Model)) +
    geom_line(linewidth = 1, alpha = 0.8) +
    geom_point(size = 2) +
    scale_color_manual(values = model_colors) +
    coord_cartesian(ylim = c(y_min, y_max)) +
    labs(title = "Period A", x = "Day of Week", y = "MAE (kWh)") +
    theme_minimal(base_size = 12)
  
  p_day_B <- ggplot(day_B, aes(x = DayName, y = MAE, color = Model, group = Model)) +
    geom_line(linewidth = 1, alpha = 0.8) +
    geom_point(size = 2) +
    scale_color_manual(values = model_colors) +
    coord_cartesian(ylim = c(y_min, y_max)) +
    labs(title = "Period B", x = "Day of Week", y = "MAE (kWh)") +
    theme_minimal(base_size = 12)
  
  combined <- p_day_A + p_day_B + 
    plot_layout(guides = "collect") & 
    theme(legend.position = "bottom") &
    plot_annotation( theme = theme(plot.title = element_text(face = "bold", hjust = 0.5)))
  
  return(combined)
}

# Print both comparison plots
print(plot_hourly_comparison(by_hour_A, by_hour_B))
print(plot_daily_comparison(by_day_A, by_day_B))
# --- Dm test ---
perform_dm <- function(actual, f1, f2, h = 24) {
  e1 <- actual - f1; e2 <- actual - f2
  forecast::dm.test(e1, e2, h = h, power = 2)
}

dm_for_period <- function(dt, period_label, h = 24) {
  ms <- levels(dt$Model)
  res <- list(); k <- 1L
  for (i in 1:(length(ms)-1)) {
    for (j in (i+1):length(ms)) {
      m1 <- ms[i]; m2 <- ms[j]
      pair <- merge(
        dt[Model == m1, .(Time, Actual, F1 = Forecast)],
        dt[Model == m2, .(Time, Actual2 = Actual, F2 = Forecast)],
        by = "Time", all = FALSE
      )
      actual <- pair$Actual
      dm <- perform_dm(actual, pair$F1, pair$F2, h)
      better <- if (!is.na(dm$statistic) && as.numeric(dm$statistic) < 0) m1 else m2
      res[[k]] <- data.table(
        Period = period_label,
        Comparison = sprintf("%s vs %s", m1, m2),
        DM_Stat = round(as.numeric(dm$statistic), 3),
        p_value = round(as.numeric(dm$p.value), 4),
        Better = better
      )
      k <- k + 1L
    }
  }
  rbindlist(res)
}

dm_A <- dm_for_period(all_forecasts_A, "A")
dm_B <- dm_for_period(all_forecasts_B, "B")
dm_all <- rbind(dm_A, dm_B)[, Sig := fifelse(p_value < 0.01, "***",
                                             fifelse(p_value < 0.05, "**",
                                                     fifelse(p_value < 0.10, "*", "")))]

msg("\n Dm summary (negative DM -> first model better)")
print(dm_all[order(Period, p_value, -abs(DM_Stat))])

