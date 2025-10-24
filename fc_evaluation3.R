# ==============================================================================
# TWO-MODEL EVALUATION: SARIMAX_PCA vs SARIMAX_CLUSTER
# (custom colors not used by base models)
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(knitr)
  library(patchwork)
  library(e1071)   # skewness/kurtosis
  library(nortest) # ad.test
  library(forecast) # dm.test
})

# ---------------- Theme ----------------
theme_set(theme_bw(base_size = 11) +
            theme(panel.grid.minor = element_blank(),
                  plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
                  strip.background = element_rect(fill = "grey92"),
                  strip.text = element_text(face = "bold", size = 10),
                  legend.position = "bottom"))

# Distinct colors NOT used by base-4:
COLS <- c("SARIMAX_PCA" = "#1B9E77",    # teal
          "SARIMAX_CLUSTER" = "#D95F02")# orange

# ---------------- Helpers ----------------
msg <- function(...) cat(sprintf(...), "\n")
safe_read <- function(path) {
  if (!file.exists(path)) stop(sprintf("File not found: %s", path))
  readRDS(path)
}

# ---------------- Load results ----------------
msg("Loading PCA & CLUSTER result files...")
SARIMAX_PCA     <- safe_read("RESULTS_FC_SARIMAX_PCA.rds")
SARIMAX_CLUSTER <- safe_read("RESULTS_FC_SARIMAX_CLUSTER.rds")

models <- list(
  SARIMAX_PCA     = SARIMAX_PCA,
  SARIMAX_CLUSTER = SARIMAX_CLUSTER
)
model_names <- names(models)

# ---------------- Performance table ----------------
msg("\nTABLE: Performance comparison (two models)\n")

perf_list <- lapply(model_names, function(nm) {
  ev <- models[[nm]]$evaluations
  if (is.null(ev)) return(NULL)
  dt <- as.data.table(ev)
  dt[, Model := nm]
  dt
})
performance <- rbindlist(perf_list, fill = TRUE)
performance[, Period_Clean := gsub("Period | \\(.*\\)", "", Period)]

perf_long <- melt(
  performance,
  id.vars = c("Model", "Period_Clean"),
  measure.vars = intersect(c("RMSE","MAE","R2","MAPE"), names(performance)),
  variable.name = "Metric", value.name = "Value"
)
perf_wide <- dcast(perf_long, Model ~ Metric + Period_Clean, value.var = "Value")
col_order <- intersect(c("Model","RMSE_A","RMSE_B","MAE_A","MAE_B","R2_A","R2_B","MAPE_A","MAPE_B"),
                       names(perf_wide))
setcolorder(perf_wide, col_order)
print(kable(perf_wide, format = "pipe", digits = 3))

msg("\nBest model by metric per period:")
best_tbl <- performance[, .SD[which.min(RMSE)], by = Period_Clean][, .(Period = Period_Clean, Best_RMSE = Model)]
best_tbl[performance[, .SD[which.min(MAE)], by = Period_Clean], Best_MAE := i.Model, on = "Period"]
best_tbl[performance[, .SD[which.max(R2)],  by = Period_Clean], Best_R2  := i.Model, on = "Period"]
best_tbl[performance[, .SD[which.min(MAPE)],by = Period_Clean], Best_MAPE:= i.Model, on = "Period"]
print(best_tbl)

# ---------------- Forecast extraction ----------------
msg("\nExtracting forecasts...")
extract_fc <- function(model, name, period_name) {
  pe <- model[[period_name]]
  if (is.null(pe) || is.null(pe$forecasts)) return(NULL)
  dt <- as.data.table(pe$forecasts)
  stopifnot(all(c("Time","Actual","Forecast") %in% names(dt)))
  dt[, Model := name]
  dt
}
fa <- lapply(model_names, \(nm) extract_fc(models[[nm]], nm, "period_A"))
fb <- lapply(model_names, \(nm) extract_fc(models[[nm]], nm, "period_B"))
all_forecasts_A <- rbindlist(fa, fill = TRUE)
all_forecasts_B <- rbindlist(fb, fill = TRUE)
msg("✓ Forecasts ready.")

# ---------------- Datetimes ----------------
start_A <- as.POSIXct("2024-10-01 01:00:00", tz = "UTC")
start_B <- as.POSIXct("2024-12-18 01:00:00", tz = "UTC")
for (df in list(all_forecasts_A, all_forecasts_B)) {
  df[, Model := factor(Model, levels = model_names)]
}
all_forecasts_A[, Datetime := start_A + as.difftime(Time - 1, units = "hours")]
all_forecasts_B[, Datetime := start_B + as.difftime(Time - 1, units = "hours")]

# ---------------- Plot: Forecast vs Actual ----------------
msg("\nPLOT 1: Forecast vs Actual (two models)\n")
plot_fc <- function(df, title) {
  ggplot(df, aes(x = Datetime)) +
    geom_line(aes(y = Actual), color = "black", linewidth = 0.8, alpha = 0.8) +
    geom_line(aes(y = Forecast, color = Model), linewidth = 0.9, alpha = 0.95) +
    facet_wrap(~Model, ncol = 1, scales = "fixed") +
    scale_color_manual(values = COLS) +
    scale_x_datetime(date_labels = "%d %b", date_breaks = "2 days") +
    labs(title = title, x = "Date", y = "Energy Consumption (kWh)") +
    theme(legend.position = "none",
          strip.text = element_text(face = "bold", size = 9))
}
print(plot_fc(all_forecasts_A, "Forecast vs Actual – Period A (Oct 1–10)"))
print(plot_fc(all_forecasts_B, "Forecast vs Actual – Period B (Dec 18–31)"))

# ---------------- Errors & derived metrics ----------------
msg("\nCompute errors...")
add_errs <- function(df) {
  df[, Error := Actual - Forecast]
  df[, AbsError := abs(Error)]
  df[, SquaredError := Error^2]
  df[, PercentError := fifelse(Actual == 0, NA_real_, 100 * abs(Error) / abs(Actual))]
  df
}
all_forecasts_A <- add_errs(all_forecasts_A)
all_forecasts_B <- add_errs(all_forecasts_B)

# ---------------- Error density with Gaussian overlay ----------------
msg("\nPLOT 2: Error density (with Gaussian overlay)\n")
plot_error_density <- function(df, title) {
  stats <- df[, .(mean_err = mean(Error, na.rm = TRUE),
                  sd_err   = sd(Error,   na.rm = TRUE)), by = Model]
  xr <- range(df$Error, na.rm = TRUE); xv <- seq(xr[1], xr[2], length.out = 300)
  gauss <- rbindlist(lapply(1:nrow(stats), function(i)
    data.table(x = xv, y = dnorm(xv, stats$mean_err[i], stats$sd_err[i]),
               Model = stats$Model[i])))
  ggplot(df, aes(x = Error, fill = Model, color = Model)) +
    geom_density(alpha = 0.25, linewidth = 0.8) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    geom_line(data = gauss, aes(x = x, y = y, color = Model),
              inherit.aes = FALSE, linetype = "dotted", linewidth = 0.9) +
    scale_color_manual(values = COLS) + scale_fill_manual(values = COLS) +
    labs(title = title, x = "Forecast Error (kWh)", y = "Density")
}
print(plot_error_density(all_forecasts_A, "Error Distribution – Period A"))
print(plot_error_density(all_forecasts_B, "Error Distribution – Period B"))

# ---------------- Normality tests ----------------
msg("\nNORMALITY TESTS"); cat(strrep("=", 60), "\n")
test_normality <- function(df, period_name) {
  cat("\n", period_name, "\n", strrep("-", 40), "\n", sep = "")
  for (m in levels(df$Model)) {
    e <- df[Model == m, Error]
    cat("\nModel:", m, "\n")
    cat("  Mean:", round(mean(e), 3), " SD:", round(sd(e), 3),
        " Skew:", round(e1071::skewness(e), 3),
        " Kurt:", round(e1071::kurtosis(e), 3), "\n")
    if (length(e) < 5000) cat("  Shapiro-Wilk p:", round(shapiro.test(e)$p.value, 4), "\n")
    cat("  KS p:", round(ks.test(e, "pnorm", mean(e), sd(e))$p.value, 4), "\n")
    cat("  AD p:", round(nortest::ad.test(e)$p.value, 4), "\n")
  }
}
test_normality(all_forecasts_A, "Period A")
test_normality(all_forecasts_B, "Period B")

# ---------------- Q-Q plots ----------------
msg("\nPLOT 3: Q-Q plots\n")
plot_qq <- function(df, title) {
  ggplot(df, aes(sample = Error, color = Model)) +
    stat_qq() + stat_qq_line(linetype = "dashed") +
    facet_wrap(~Model, scales = "free") +
    scale_color_manual(values = COLS) +
    labs(title = title, x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_minimal()
}
print(plot_qq(all_forecasts_A, "Q-Q Plots – Period A"))
print(plot_qq(all_forecasts_B, "Q-Q Plots – Period B"))

# ---------------- Accuracy by Hour of Day ----------------
msg("\nAccuracy by hour of day")
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

plot_hour_metric <- function(df, metric, title, ylab) {
  ggplot(df, aes_string(x = "Hour", y = metric, color = "Model", group = "Model")) +
    geom_line(linewidth = 1, alpha = 0.85) + geom_point(size = 2) +
    scale_color_manual(values = COLS) +
    scale_x_continuous(breaks = seq(1, 24, 2)) +
    labs(title = title, x = "Hour of Day", y = ylab) +
    theme_minimal(base_size = 12)
}
print(plot_hour_metric(by_hour_A, "MAPE", "MAPE by Hour – Period A", "MAPE (%)"))
print(plot_hour_metric(by_hour_B, "MAPE", "MAPE by Hour – Period B", "MAPE (%)"))

# ---------------- Accuracy by Day of Week ----------------
msg("\nAccuracy by day of week")
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

plot_dow_metric <- function(df, metric, title, ylab) {
  df$DayName <- factor(df$DayName, levels = dow_labels)
  ggplot(df, aes_string(x = "DayName", y = metric, color = "Model", group = "Model")) +
    geom_line(linewidth = 1, alpha = 0.85) + geom_point(size = 2) +
    scale_color_manual(values = COLS) +
    labs(title = title, x = "Day of Week", y = ylab) +
    theme_minimal(base_size = 12)
}
print(plot_dow_metric(by_dow_A, "MAPE", "MAPE by Day – Period A", "MAPE (%)"))
print(plot_dow_metric(by_dow_B, "MAPE", "MAPE by Day – Period B", "MAPE (%)"))

# ---------------- Side-by-side (MAE) optional ----------------
p_hour_A_MAE <- plot_hour_metric(by_hour_A, "MAE", "MAE by Hour – Period A", "MAE (kWh)")
p_hour_B_MAE <- plot_hour_metric(by_hour_B, "MAE", "MAE by Hour – Period B", "MAE (kWh)")
print((p_hour_A_MAE + p_hour_B_MAE) + plot_layout(guides = "collect"))

# ---------------- Diebold–Mariano: PCA vs CLUSTER only ----------------
msg("\nDIEBOLD–MARIANO: SARIMAX_PCA vs SARIMAX_CLUSTER (h=24, SE loss)\n")

perform_dm <- function(actual, f1, f2, h = 24, alternative = "two.sided", power = 2) {
  idx <- complete.cases(actual, f1, f2)
  e1 <- as.numeric(actual[idx] - f1[idx])
  e2 <- as.numeric(actual[idx] - f2[idx])
  forecast::dm.test(e1, e2, h = h, power = power, alternative = alternative)
}

dm_period <- function(dt) {
  # dt has rows for both models; align by Time
  pca <- dt[Model == "SARIMAX_PCA",     .(Time, Actual, F1 = Forecast)][order(Time)]
  clu <- dt[Model == "SARIMAX_CLUSTER", .(Time, F2 = Forecast)][order(Time)]
  m   <- merge(pca, clu, by = "Time")
  perform_dm(m$Actual, m$F1, m$F2, h = 24)
}

dm_A <- dm_period(all_forecasts_A)
dm_B <- dm_period(all_forecasts_B)

dm_summary <- data.table(
  Period   = c("A","B"),
  Comparison = c("SARIMAX_PCA vs SARIMAX_CLUSTER","SARIMAX_PCA vs SARIMAX_CLUSTER"),
  DM_Stat  = c(as.numeric(dm_A$statistic), as.numeric(dm_B$statistic)),
  p_value  = c(as.numeric(dm_A$p.value),   as.numeric(dm_B$p.value))
)[, `:=`(
  DM_Stat = round(DM_Stat, 3),
  p_value = round(p_value, 4),
  Sig = fifelse(p_value < 0.01, "***",
                fifelse(p_value < 0.05, "**",
                        fifelse(p_value < 0.10, "*", ""))),
  Better = ifelse(DM_Stat < 0, "SARIMAX_PCA", "SARIMAX_CLUSTER")
)]

msg("=== DM SUMMARY (Negative DM_Stat ⇒ first model better) ===")
print(dm_summary)

msg("\nDone.")
