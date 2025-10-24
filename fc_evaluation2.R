# ==============================================================================
# COMPREHENSIVE MODEL EVALUATION SCRIPT — with SARIMAX_CLUSTER & SARIMAX_PCA
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(knitr)
  library(RColorBrewer)
  library(patchwork)
  library(tseries)    # dm.test
  library(e1071)      # skewness/kurtosis
  library(nortest)    # ad.test
})

theme_set(theme_bw(base_size = 11) +
            theme(panel.grid.minor = element_blank(),
                  plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
                  strip.background = element_rect(fill = "grey92"),
                  strip.text = element_text(face = "bold", size = 10),
                  legend.position = "bottom"))

# ---------- helpers ----------
safe_read <- function(path) {
  if (!file.exists(path)) stop(sprintf("File not found: %s", path))
  readRDS(path)
}

msg <- function(...) cat(sprintf(...), "\n")

# ---------- load results ----------
msg("Loading forecast results...")
SARIMA           <- safe_read("RESULTS_FC_SARIMA.rds")
SARIMAX          <- safe_read("RESULTS_FC_SARIMAX.rds")
LSTM             <- safe_read("RESULTS_FC_LSTM.rds")
HYBRID           <- safe_read("RESULTS_FC_HYBRID.rds")
SARIMAX_CLUSTER  <- safe_read("RESULTS_FC_SARIMAX_CLUSTER.rds")
SARIMAX_PCA      <- safe_read("RESULTS_FC_SARIMAX_PCA.rds")

models <- list(
  SARIMA  = SARIMA,
  SARIMAX = SARIMAX,
  LSTM    = LSTM,
  HYBRID  = HYBRID,
  SARIMAX_CLUSTER = SARIMAX_CLUSTER,
  SARIMAX_PCA     = SARIMAX_PCA
)
model_names <- names(models)

# ---------- performance comparison ----------
msg("\nTABLE 2: FORECASTING PERFORMANCE COMPARISON\n")

perf_list <- list()
for (nm in model_names) {
  if (!is.null(models[[nm]]$evaluations)) {
    perf <- as.data.table(models[[nm]]$evaluations)
    perf[, Model := nm]
    perf_list[[nm]] <- perf
  }
}
performance <- rbindlist(perf_list, fill = TRUE)
performance[, Period_Clean := gsub("Period | \\(.*\\)", "", Period)]

# long/wide
perf_long <- melt(performance,
                  id.vars = c("Model", "Period_Clean"),
                  measure.vars = intersect(c("RMSE","MAE","R2","MAPE"), names(performance)),
                  variable.name = "Metric", value.name = "Value")
perf_wide <- dcast(perf_long, Model ~ Metric + Period_Clean, value.var = "Value")

col_order <- intersect(c("Model", "RMSE_A", "RMSE_B", "MAE_A", "MAE_B",
                         "R2_A", "R2_B", "MAPE_A", "MAPE_B"), names(perf_wide))
setcolorder(perf_wide, col_order)
print(kable(perf_wide, format = "pipe", digits = 3))

msg("\nModel Rankings (1 = best):")
performance[, `:=`(
  RMSE_Rank = rank(RMSE, ties.method = "min"),
  MAE_Rank  = rank(MAE,  ties.method = "min"),
  R2_Rank   = rank(-R2,  ties.method = "min"),
  MAPE_Rank = rank(MAPE, ties.method = "min")
), by = Period_Clean]
print(kable(performance[, .(Model, Period = Period_Clean, RMSE_Rank, MAE_Rank, R2_Rank, MAPE_Rank)],
            format = "pipe"))

msg("\nBest Model by Metric:")
best_tbl <- performance[, .SD[which.min(RMSE)], by = Period_Clean][, .(Period = Period_Clean, Best_RMSE = Model)]
best_tbl[performance[, .SD[which.min(MAE)], by = Period_Clean], Best_MAE := i.Model, on = "Period"]
best_tbl[performance[, .SD[which.max(R2)],  by = Period_Clean], Best_R2  := i.Model, on = "Period"]
best_tbl[performance[, .SD[which.min(MAPE)],by = Period_Clean], Best_MAPE:= i.Model, on = "Period"]
print(best_tbl)

# ---------- forecast extraction ----------
msg("\nExtracting forecast data...")
extract_fc <- function(model, name, period_name) {
  # expect model[[period_name]]$forecasts with columns Time, Actual, Forecast
  pe <- model[[period_name]]
  if (is.null(pe) || is.null(pe$forecasts)) return(NULL)
  dt <- as.data.table(pe$forecasts)
  req <- c("Time","Actual","Forecast")
  if (!all(req %in% names(dt))) stop(sprintf("Missing columns in %s/%s", name, period_name))
  dt[, Model := name]
}

forecasts_A <- lapply(model_names, \(nm) extract_fc(models[[nm]], nm, "period_A"))
forecasts_B <- lapply(model_names, \(nm) extract_fc(models[[nm]], nm, "period_B"))
all_forecasts_A <- rbindlist(forecasts_A, fill = TRUE)
all_forecasts_B <- rbindlist(forecasts_B, fill = TRUE)
msg("✓ Forecast data extracted")

# ---------- plots: Forecast vs Actual ----------
msg("\nPLOT 1: FORECAST VS ACTUAL COMPARISON\n")

all_forecasts_A[, Model := factor(Model, levels = model_names)]
all_forecasts_B[, Model := factor(Model, levels = model_names)]

start_A <- as.POSIXct("2024-10-01 01:00:00", tz = "UTC")
start_B <- as.POSIXct("2024-12-18 01:00:00", tz = "UTC")
all_forecasts_A[, Datetime := start_A + as.difftime(Time - 1, units = "hours")]
all_forecasts_B[, Datetime := start_B + as.difftime(Time - 1, units = "hours")]

plot_fc <- function(df, title) {
  ggplot(df, aes(x = Datetime)) +
    geom_line(aes(y = Actual), color = "black", linewidth = 0.8, alpha = 0.8) +
    geom_line(aes(y = Forecast, color = Model), linewidth = 0.8, alpha = 0.9) +
    facet_wrap(~Model, ncol = 1, scales = "fixed") +
    scale_color_brewer(palette = "Set1") +
    scale_x_datetime(date_labels = "%d %b", date_breaks = "2 days") +
    labs(title = title, x = "Date", y = "Energy Consumption (kWh)") +
    theme(legend.position = "none",
          strip.text = element_text(face = "bold", size = 9))
}
print(plot_fc(all_forecasts_A, "Forecast vs Actual – Period A (Oct 1 - Oct 10)"))
print(plot_fc(all_forecasts_B, "Forecast vs Actual – Period B (Dec 18 - Dec 31)"))

# ---------- errors ----------
msg("\nCalculating forecast errors...")
add_err_cols <- function(df) {
  df[, Error := Actual - Forecast]
  df[, AbsError := abs(Error)]
  df[, SquaredError := Error^2]
  df[, PercentError := fifelse(Actual == 0, NA_real_, 100 * abs(Error) / abs(Actual))]
  df
}
all_forecasts_A <- add_err_cols(all_forecasts_A)
all_forecasts_B <- add_err_cols(all_forecasts_B)

# ---------- error distributions (with Gaussian overlay) ----------
msg("\nPLOT 2: ERROR DISTRIBUTION COMPARISON (with Gaussian overlay)\n")

plot_error_density <- function(df, title) {
  stats <- df[, .(mean_err = mean(Error, na.rm = TRUE),
                  sd_err   = sd(Error, na.rm = TRUE)), by = Model]
  x_range <- range(df$Error, na.rm = TRUE)
  x_vals <- seq(x_range[1], x_range[2], length.out = 300)
  gauss <- rbindlist(lapply(1:nrow(stats), function(i) {
    data.table(x = x_vals,
               y = dnorm(x_vals, stats$mean_err[i], stats$sd_err[i]),
               Model = stats$Model[i])
  }))
  
  ggplot(df, aes(x = Error, fill = Model, color = Model)) +
    geom_density(alpha = 0.25, linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    geom_line(data = gauss, aes(x = x, y = y, color = Model),
              inherit.aes = FALSE, linetype = "dotted", linewidth = 0.9) +
    scale_fill_brewer(palette = "Set1") + scale_color_brewer(palette = "Set1") +
    labs(title = title, x = "Forecast Error (kWh)", y = "Density")
}
print(plot_error_density(all_forecasts_A, "Error Distribution - Period A"))
print(plot_error_density(all_forecasts_B, "Error Distribution - Period B"))

# ---------- normality tests ----------
msg("\nNORMALITY TESTS\n")
cat(strrep("=", 60), "\n")

test_normality <- function(df, period_name) {
  cat("\n", period_name, "\n", strrep("-", 40), "\n", sep = "")
  for (m in levels(df$Model)) {
    errors <- df[Model == m, Error]
    cat("\nModel:", m, "\n")
    cat("  Mean:", round(mean(errors), 3), "\n")
    cat("  SD:", round(sd(errors), 3), "\n")
    cat("  Skewness:", round(e1071::skewness(errors), 3), "\n")
    cat("  Kurtosis:", round(e1071::kurtosis(errors), 3), "\n")
    
    if (length(errors) < 5000) {
      sw <- shapiro.test(errors)
      cat("  Shapiro-Wilk p-value:", round(sw$p.value, 4), "\n")
    }
    ks <- ks.test(errors, "pnorm", mean(errors), sd(errors))
    cat("  KS test p-value:", round(ks$p.value, 4), "\n")
    ad <- nortest::ad.test(errors)
    cat("  Anderson-Darling p-value:", round(ad$p.value, 4), "\n")
  }
}
test_normality(all_forecasts_A, "Period A")
test_normality(all_forecasts_B, "Period B")

# ---------- Q-Q plots ----------
msg("\nPLOT 3: Q-Q PLOTS\n")
plot_qq <- function(df, title) {
  ggplot(df, aes(sample = Error, color = Model)) +
    stat_qq() +
    stat_qq_line(linetype = "dashed") +
    facet_wrap(~Model, scales = "free") +
    scale_color_brewer(palette = "Set1") +
    labs(title = title, x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_minimal()
}
print(plot_qq(all_forecasts_A, "Q-Q Plots - Period A"))
print(plot_qq(all_forecasts_B, "Q-Q Plots - Period B"))

# ---------- accuracy by hour of day ----------
msg("\nFORECAST ACCURACY BY HOUR OF DAY\n")
for (df in list(all_forecasts_A, all_forecasts_B)) df[, Hour := ((Time - 1) %% 24) + 1]
calc_hourly <- function(df) df[, .(
  N = .N,
  RMSE = sqrt(mean(SquaredError, na.rm = TRUE)),
  MAE  = mean(AbsError, na.rm = TRUE),
  MAPE = mean(PercentError, na.rm = TRUE)
), by = .(Model, Hour)][order(Model, Hour)]
by_hour_A <- calc_hourly(all_forecasts_A)
by_hour_B <- calc_hourly(all_forecasts_B)
print(kable(by_hour_A, format = "pipe", digits = 3))
print(kable(by_hour_B, format = "pipe", digits = 3))

plot_hour <- function(df, title) {
  ggplot(df, aes(x = Hour, y = MAPE, color = Model, group = Model)) +
    geom_line(linewidth = 1, alpha = 0.8) + geom_point(size = 2) +
    scale_color_brewer(palette = "Set1") +
    scale_x_continuous(breaks = seq(1, 24, 2)) +
    labs(title = title, x = "Hour of Day", y = "MAPE (%)") +
    theme_minimal(base_size = 12)
}
print(plot_hour(by_hour_A, "Forecast Accuracy by Hour of Day - Period A"))
print(plot_hour(by_hour_B, "Forecast Accuracy by Hour of Day - Period B"))

# ---------- accuracy by day of week ----------
msg("\nFORECAST ACCURACY BY DAY OF WEEK\n")
dow_labels <- c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")
for (df in list(all_forecasts_A, all_forecasts_B)) {
  df[, Day := ceiling(Time / 24)]
  df[, DayOfWeek := ((Day - 1) %% 7) + 1]
}
calc_dow <- function(df) df[, .(
  N = .N,
  RMSE = sqrt(mean(SquaredError, na.rm = TRUE)),
  MAE  = mean(AbsError, na.rm = TRUE),
  MAPE = mean(PercentError, na.rm = TRUE)
), by = .(Model, DayOfWeek)][, DayName := dow_labels[DayOfWeek]]
by_dow_A <- calc_dow(all_forecasts_A)
by_dow_B <- calc_dow(all_forecasts_B)
print(kable(by_dow_A[, .(Model, DayName, RMSE, MAE, MAPE)], format = "pipe", digits = 3))
print(kable(by_dow_B[, .(Model, DayName, RMSE, MAE, MAPE)], format = "pipe", digits = 3))

plot_dow <- function(df, title) {
  df$DayName <- factor(df$DayName, levels = dow_labels)
  ggplot(df, aes(x = DayName, y = MAPE, color = Model, group = Model)) +
    geom_line(linewidth = 1, alpha = 0.8) + geom_point(size = 2) +
    scale_color_brewer(palette = "Set1") +
    labs(title = title, x = "Day of Week", y = "MAPE (%)") +
    theme_minimal(base_size = 12)
}
print(plot_dow(by_dow_A, "Forecast Accuracy by Day of Week - Period A"))
print(plot_dow(by_dow_B, "Forecast Accuracy by Day of Week - Period B"))

# ---------- side-by-side figs (MAE) ----------
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
plot_dow_metric <- function(df, metric, subtitle) {
  df$DayName <- factor(df$DayName, levels = dow_labels)
  ggplot(df, aes_string(x = "DayName", y = metric, color = "Model", group = "Model")) +
    geom_line(linewidth = 1, alpha = 0.8) +
    geom_point(size = 2) +
    scale_color_brewer(palette = "Set1") +
    labs(x = "Day of Week", y = paste0(metric, " (kWh)"), subtitle = subtitle) +
    theme(legend.position = "bottom",
          plot.subtitle = element_text(hjust = 0.5, face = "bold"))
}
p_hour_A_MAE <- plot_hour_metric(by_hour_A, "MAE", "Period A")
p_hour_B_MAE <- plot_hour_metric(by_hour_B, "MAE", "Period B")
mae_hour_fig <- p_hour_A_MAE + p_hour_B_MAE + plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
p_dow_A_MAE <- plot_dow_metric(by_dow_A, "MAE", "Period A")
p_dow_B_MAE <- plot_dow_metric(by_dow_B, "MAE", "Period B")
mae_dow_fig <- p_dow_A_MAE + p_dow_B_MAE + plot_layout(guides = "collect") &
  theme(legend.position = "bottom")
print(mae_hour_fig)
print(mae_dow_fig)

# ---------- Diebold-Mariano tests (pairwise across ALL models) ----------
msg("\nDIEBOLD–MARIANO TESTS (pairwise, h=24, SE loss)\n")
perform_dm <- function(actual, f1, f2, h = 24, alternative = "two.sided") {
  e1 <- actual - f1
  e2 <- actual - f2
  # tseries::dm.test uses power (2=SE). We’ll use power=2 to match squared error loss.
  tseries::dm.test(e1, e2, h = h, power = 2, alternative = alternative)
}

dm_for_period <- function(dt, period_label) {
  # dt must contain Model, Forecast, Actual with aligned Time
  res <- list()
  ms <- levels(dt$Model)
  for (i in 1:(length(ms)-1)) {
    for (j in (i+1):length(ms)) {
      m1 <- ms[i]; m2 <- ms[j]
      seg1 <- dt[Model == m1][order(Time)]
      seg2 <- dt[Model == m2][order(Time)]
      # ensure same length / alignment
      n <- min(nrow(seg1), nrow(seg2))
      seg1 <- seg1[1:n]; seg2 <- seg2[1:n]
      dm <- perform_dm(seg1$Actual, seg1$Forecast, seg2$Forecast, h = 24)
      res[[paste(m1, m2, sep = " vs ")]] <- data.table(
        Period = period_label,
        Comparison = paste(m1, "vs", m2),
        DM_Stat = as.numeric(dm$statistic),
        p_value = as.numeric(dm$p.value),
        Better = ifelse(dm$statistic < 0, m1, m2) # negative => first better
      )
    }
  }
  rbindlist(res)
}

dm_A <- dm_for_period(all_forecasts_A, "A")
dm_B <- dm_for_period(all_forecasts_B, "B")
dm_all <- rbind(dm_A, dm_B)[, `:=`(
  DM_Stat = round(DM_Stat, 3),
  p_value = round(p_value, 4),
  Sig = fifelse(p_value < 0.01, "***",
                fifelse(p_value < 0.05, "**",
                        fifelse(p_value < 0.10, "*", "")))
)]
msg("\n=== DM SUMMARY (Note: Negative DM_Stat ⇒ first model better) ===")
print(dm_all[order(Period, p_value, -abs(DM_Stat))])

msg("\nDone.")
