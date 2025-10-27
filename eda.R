# ==============================================================
# Author: Pauline Cox
# Script: eda.R
#
# Description: Performs exploratory data analysis and preprocessing
# for the Eneco energy consumption dataset and merged feature set. 
# Includes anomaly detection, missing value handling, interpolation, 
# outlier detection, winsorization, correlation analysis, and visualization.
#
# Input: 
#   - Raw Eneco energy data 
#   - Combined raw dataset 
#
# Output: 
#   - Cleaned and aggregated energy consumption data (energy_hourly_complete)
#   - Preprocessed, feature-complete dataset (processed_data)
# ==============================================================s

# --- Target Variable Exploration & Processing ---

explore_target <- function() {
  
  cat("Loaded Eneco data:", nrow(eneco_data), "rows\n")
  
  # Build full 15-min timeline
  raw_seq <- tibble(`Start Datum` = seq(
    min(eneco_data$`Start Datum`),
    max(eneco_data$`Start Datum`),
    by = "15 min"
  ))
  
  # Flag missing / zero values
  check_dt <- raw_seq %>%
    left_join(
      dplyr::select(eneco_data, `Start Datum`, total_consumption),
      by = "Start Datum"
    ) %>%
    arrange(`Start Datum`) %>%
    mutate(
      issue_type = case_when(
        is.na(total_consumption)    ~ "missing",
        total_consumption == 0      ~ "zero",
        TRUE                        ~ "ok"
      ),
      flag_issue = ifelse(issue_type == "ok", 0, 1)
    )
  
  # Collapse into issue blocks
  issue_blocks <- check_dt %>%
    mutate(block = cumsum(c(0, diff(flag_issue)) != 0)) %>%
    group_by(block) %>%
    summarise(
      start = min(`Start Datum`),
      end   = max(`Start Datum`),
      n     = sum(flag_issue),
      issue_types = paste(unique(issue_type[issue_type != "ok"]), collapse = ", "),
      .groups = "drop"
    ) %>%
    filter(n > 0)
  
  cat("Found", nrow(issue_blocks), "anomalous periods\n")
  
  # Plot anomalies
  p_highlighted <- ggplot(check_dt, aes(x = `Start Datum`, y = total_consumption)) +
    geom_rect(data = issue_blocks,
              aes(xmin = start, xmax = end, ymin = -Inf, ymax = Inf),
              inherit.aes = FALSE, fill = "black", alpha = 0.4) +
    geom_line(color = "skyblue3", alpha = 0.6) +
    labs(
         x = "Datetime", y = "Consumption (kWh)") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  print(p_highlighted)
  
  # Fill anomalies with profile
  fill_anomalies_with_profile <- function(df, min_valid = 0) {
    df <- df %>%
      arrange(`Start Datum`) %>%
      mutate(date = as.Date(`Start Datum`),
             weekday = wday(date, week_start = 1),
             hour = hour(`Start Datum`),
             minute = minute(`Start Datum`))
    
    anomaly_idx <- which(df$total_consumption <= min_valid)
    cat("Filling", length(anomaly_idx), "anomalous values\n")
    
    for (i in anomaly_idx) {
      ts <- df$`Start Datum`[i]
      wday_i <- df$weekday[i]; h_i <- df$hour[i]; m_i <- df$minute[i]
      window_start <- ts - weeks(2); window_end <- ts + weeks(2)
      
      ref_vals <- df %>%
        filter(`Start Datum` >= window_start,
               `Start Datum` <= window_end,
               weekday == wday_i, hour == h_i, minute == m_i,
               total_consumption > min_valid) %>%
        pull(total_consumption)
      
      if (length(ref_vals) > 0) {
        df$total_consumption[i] <- mean(ref_vals, na.rm = TRUE)
      } else {
        df$total_consumption[i] <- NA
      }
    }
    df
  }
  
  energy_clean <- fill_anomalies_with_profile(eneco_data, min_valid = 0)
  
  # Fill short gaps with interpolation
  full_seq <- tibble(`Start Datum` = seq(min(energy_clean$`Start Datum`),
                                         max(energy_clean$`Start Datum`),
                                         by = "15 min"))
  
  energy_full <- full_seq %>%
    left_join(energy_clean %>% dplyr::select(`Start Datum`, total_consumption),
              by = "Start Datum") %>%
    arrange(`Start Datum`)
  
  n_before <- sum(is.na(energy_full$total_consumption))
  energy_full <- energy_full %>%
    mutate(total_consumption = na.approx(total_consumption, na.rm = FALSE))
  n_after <- sum(is.na(energy_full$total_consumption))
  cat("Interpolated", n_before - n_after, "missing values\n")
  
  # Aggregate to hourly
  energy_hourly <- energy_full %>%
    mutate(hour = floor_date(`Start Datum`, "hour")) %>%
    group_by(hour) %>%
    summarise(total_consumption_kWh = sum(total_consumption, na.rm = TRUE),
              .groups = "drop") %>%
    rename(interval = hour) %>%
    as.data.table()
  
  # Add time features
  energy_hourly[, date := as.Date(interval)]
  energy_hourly[, hour := hour(interval)]
  energy_hourly[, month_str := format(interval, "%Y-%m")]
  energy_hourly[, weekday_en := wday(date, label = TRUE, abbr = FALSE,
                                     week_start = 1, locale = "C")]
  
  # Filter from 1 July 2023 onward
  energy_hourly_complete <- energy_hourly[interval >= as.POSIXct("2023-07-01 00:00:00", tz = "UTC")]
  
  # Summary stats
  cat("\nDescriptive statistics (after July 2023):\n")
  print(psych::describe(as.data.frame(energy_hourly_complete[, .(total_consumption_kWh)])))
  
  # Seasonal decomposition 
  cat("\nSeasonal decomposition (trend, seasonality, residual):\n")
  
  ts_energy <- ts(energy_hourly_complete$total_consumption_kWh, frequency = 168)
  decomp <- stl(ts_energy, s.window = "periodic")
  
  # Create tidy data frame for plotting
  decomp_df <- data.frame(
    interval = energy_hourly_complete$interval,
    Observed = energy_hourly_complete$total_consumption_kWh,
    Trend = decomp$time.series[, "trend"],
    Seasonal = decomp$time.series[, "seasonal"],
    Remainder = decomp$time.series[, "remainder"]
  )
  
  # Reshape for ggplot
  decomp_long <- tidyr::pivot_longer(
    decomp_df,
    cols = c("Observed", "Trend", "Seasonal", "Remainder"),
    names_to = "Component",
    values_to = "Value"
  )
  
  # Order panels so Observed appears first
  decomp_long$Component <- factor(
    decomp_long$Component,
    levels = c("Observed", "Trend", "Seasonal", "Remainder")
  )
  
  # Plot stacked facets with one shared x-axis
  ggplot(decomp_long, aes(x = interval, y = Value)) +
    geom_line(linewidth = 0.5, color = "black") +
    facet_wrap(~ Component, ncol = 1, scales = "free_y", strip.position = "right") +
    labs(
      x = "Date", y = "kWh"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      strip.text = element_text(face = "bold"),
      panel.spacing.y = unit(0.2, "lines"),
      axis.title.x = element_text(margin = ggplot2::margin(t = 5)),
      axis.title.y = element_text(margin = ggplot2::margin(r = 5)),
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.minor = element_blank()
    )
  
  
  # Daily totals
  daily_kWh <- energy_hourly_complete[, .(daily_kWh = sum(total_consumption_kWh, na.rm = TRUE)), by = date]
  daily_kWh[, month_str := format(date, "%Y-%m")]
  daily_kWh[, weekday_en := wday(date, label = TRUE, abbr = FALSE, week_start = 1, locale = "C")]
  
  # Visualizations (only post-July 2023)
  print(ggplot(energy_hourly_complete, aes(interval, total_consumption_kWh)) +
          geom_line(color = "#0072B2") +
          labs( x = "Datetime", y = "kWh") +
          theme_minimal())
  
  print(ggplot(daily_kWh, aes(date, daily_kWh)) +
          geom_line(color = "#E69F00") +
          labs( x = "Date", y = "kWh") +
          theme_minimal())
  
  print(ggplot(daily_kWh, aes(weekday_en, daily_kWh)) +
          geom_boxplot(fill = "#E69F00") +
          labs( x = NULL, y = "kWh") +
          theme_minimal())
  
  # Create month index (1–12) and label
  daily_kWh[, month_num := month(date)]
  daily_kWh[, month_label := factor(month_num, 
                                    levels = 1:12, 
                                    labels = month.abb)]
  
  print(
    ggplot(daily_kWh, aes(x = month_label, y = daily_kWh)) +
      geom_boxplot(fill = "#0072B2") +
      labs(x = "Month", y = "kWh") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  )
  
  print(ggplot(energy_hourly_complete, aes(factor(hour), total_consumption_kWh)) +
          geom_boxplot(fill = "#009E73") +
          labs( x = "Hour", y = "kWh") +
          theme_minimal())
  
  # --- Subset test periods ---
  oct_period <- energy_hourly_complete[
    interval >= as.POSIXct("2024-10-01 00:00:00", tz = "UTC") &
      interval <  as.POSIXct("2024-10-15 00:00:00", tz = "UTC")
  ]
  
  dec_period <- energy_hourly_complete[
    interval >= as.POSIXct("2024-12-17 00:00:00", tz = "UTC") &
      interval <= as.POSIXct("2024-12-31 23:59:59", tz = "UTC")
  ]
  
  # --- Plot for early October (black line) ---
  p_oct <- ggplot(oct_period, aes(x = interval, y = total_consumption_kWh)) +
    geom_line(color = "black", linewidth = 0.5) +
    labs(x = "Datetime", y = "kWh") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.minor = element_blank()
    )
  
  # --- Plot for late December (black line) ---
  p_dec <- ggplot(dec_period, aes(x = interval, y = total_consumption_kWh)) +
    geom_line(color = "black", linewidth = 0.5) +
    labs(x = "Datetime", y = "kWh") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.minor = element_blank()
    )
  
  # --- Display separately ---
  print(p_oct)
  print(p_dec)
  
  return(energy_hourly_complete)
}


# --- Full data exploration & preprocessing ---

explore_full_data <- function() {
  
  # Filter to 2023-2024
  processed_data <- as.data.table(raw_data)[!is.na(interval)] %>%
    filter(interval >= as.POSIXct("2023-01-07 00:00:00", tz = "UTC"),
           interval <  as.POSIXct("2025-01-01 00:00:00", tz = "UTC"))
  
  # --- FIX: Scale KNMI temperature from 1/10 °C to °C ---
  if ("temperature" %in% names(processed_data)) {
    processed_data[, temperature := temperature / 10]
    cat("Converted KNMI temperature from 1/10 °C to °C scale\n")
  }

  numeric_cols <- names(processed_data)[sapply(processed_data, is.numeric)]
  
  cat("\nDescriptive statistics (before preprocessing):\n")
  summary_before <- psych::describe(as.data.frame(processed_data[, ..numeric_cols]))
  print(
    round(
      summary_before[, c("mean", "sd", "median", "min", "max", "skew", "kurtosis")],
      2
    )
  )
  # Replace target with cleaned version from explore_target
  processed_data[, total_consumption_kWh := eneco_data_processed[.SD, on = "interval", total_consumption_kWh]]
  
  # Outlier detection
  detect_outliers <- function(x, k = 1.5) {
    Q1 <- quantile(x, 0.25, na.rm = TRUE)
    Q3 <- quantile(x, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower <- Q1 - k*IQR
    upper <- Q3 + k*IQR
    return(x < lower | x > upper)
  }
  
  non_binary_vars <- numeric_cols[
    sapply(processed_data[, ..numeric_cols], function(x) length(unique(na.omit(x))) > 2)
  ]
  
  outlier_plot_dt <- rbindlist(lapply(non_binary_vars, function(var) {
    flags <- detect_outliers(processed_data[[var]])
    data.table(
      interval = processed_data$interval,
      Variable = var,
      Value = processed_data[[var]],
      Outlier = flags
    )
  }), fill = TRUE)
  
  print(
    ggplot(outlier_plot_dt, aes(x = interval, y = Value)) +
      geom_line(color = "skyblue3", linewidth = 0.3, na.rm = TRUE) +
      geom_point(data = outlier_plot_dt[Outlier == TRUE],
                 aes(x = interval, y = Value),
                 color = "red", size = 0.8, na.rm = TRUE) +
      facet_wrap(~ Variable, scales = "free_y", ncol = 3) +
      labs() +
      theme_minimal()
  )
  
  # Missing value analysis
  cat("\nMissing values:\n")
  missing_counts <- processed_data[, lapply(.SD, function(x) sum(is.na(x)))]
  missing_perc   <- processed_data[, lapply(.SD, function(x) mean(is.na(x)) * 100)]
  missing_summary <- data.table(
    Variable        = names(missing_counts),
    Missing_Count   = as.numeric(missing_counts[1, ]),
    Missing_Percent = as.numeric(missing_perc[1, ])
  )[Missing_Count > 0][order(-Missing_Percent)]
  print(missing_summary)
  
  # Handle missing values
  cat("\nHandling missing values...\n")
  
  # Indoor sensors: interpolate then extend
  indoor_vars <- c("tempC","humidity","co2","sound","lux")
  for (var in intersect(indoor_vars, names(processed_data))) {
    processed_data[, (var) := na.approx(get(var), maxgap = 6, na.rm = FALSE)]
    processed_data[, (var) := na.fill(get(var), c("extend","extend"))]
  }
  
  # Occupancy: assume 0 when missing
  if ("total_occupancy" %in% names(processed_data)) {
    processed_data[is.na(total_occupancy), total_occupancy := 0]
  }
  
  # Weather: forward fill
  weather_vars <- c("temperature","wind_speed","sunshine_minutes","global_radiation",
                    "humidity_percent","fog","rain","snow","thunder","ice")
  for (var in intersect(weather_vars, names(processed_data))) {
    processed_data[, (var) := na.fill(get(var), c("extend","extend"))]
  }
  
  # Winsorization (CO2 + Sound)
  if ("co2" %in% names(processed_data)) {
    co2_upper <- quantile(processed_data$co2, 0.999, na.rm = TRUE)
    processed_data[co2 > co2_upper, co2 := co2_upper]
  }
  if ("sound" %in% names(processed_data)) {
    sound_lower <- quantile(processed_data$sound, 0.001, na.rm = TRUE)
    sound_upper <- quantile(processed_data$sound, 0.999, na.rm = TRUE)
    processed_data[sound > sound_upper, sound := sound_upper]
    processed_data[sound < sound_lower, sound := sound_lower]
  }
  
  
  cat("\nDescriptive statistics (after preprocessing):\n")
  summary_after <- psych::describe(as.data.frame(processed_data[, ..numeric_cols]))
  print(
    round(
      summary_after[, c("mean", "sd", "median", "min", "max", "skew", "kurtosis")],
      2
    )
  )
  
  # Correlation analysis
  cor_matrix <- cor(processed_data[, ..numeric_cols], use = "pairwise.complete.obs")
  cor_dt <- as.data.table(as.table(cor_matrix))
  setnames(cor_dt, old = names(cor_dt), new = c("Var1", "Var2", "Correlation"))
  
  var_order <- colnames(processed_data[, ..numeric_cols])
  # total energy consumption first
  var_order <- c("total_consumption_kWh", setdiff(var_order, "total_consumption_kWh"))
  
  cor_dt[, Var1 := factor(Var1, levels = var_order)]
  cor_dt[, Var2 := factor(Var2, levels = rev(var_order))]  # reverse for heatmap orientation
  
  # Plot
  print(
    ggplot(cor_dt, aes(x = Var1, y = Var2, fill = Correlation)) +
      geom_tile(color = "white") +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
      labs(fill = "Correlation") +
      theme_minimal() +
      theme(
        axis.title.x = element_blank(),
        axis.title.y = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank()
      )
  )
  
  
  # Correlation with energy
  non_binary_vars <- setdiff(non_binary_vars, "total_consumption_kWh")
  energy_corr <- sapply(processed_data[, ..non_binary_vars], function(x) {
    cor(processed_data$total_consumption_kWh, x, use = "pairwise.complete.obs")
  })
  energy_corr_dt <- data.table(Variable = names(energy_corr), Correlation = as.numeric(energy_corr))
  
  print(
    ggplot(energy_corr_dt, aes(x = reorder(Variable, Correlation), y = Correlation)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(title = "Correlation with Energy Consumption", x = NULL) +
      theme_minimal()
  )
  
  # Time series visualization
  plot_vars <- setdiff(names(processed_data), c("interval", "total_consumption_kWh", "date", "hour", "month_str", "weekday_en"))
  plot_dt <- melt(
    processed_data,
    id.vars = "interval",
    measure.vars = plot_vars,
    variable.name = "Variable", value.name = "Value"
  )
  
  print(
    ggplot(plot_dt, aes(interval, Value)) +
      geom_line(linewidth = 0.25, color = "#0072B2", na.rm = TRUE) +
      facet_wrap(~ Variable, scales = "free_y", ncol = 3) +
      labs() +
      theme_minimal()
  )

  return(processed_data)
}