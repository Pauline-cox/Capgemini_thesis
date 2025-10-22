# Function to add engeneerged features to the dataset

add_engineered_features <- function(dt) {
  dt <- copy(dt)
  
  # Ensure proper datetime format and timezone
  dt[, interval := as.POSIXct(interval, tz = "Europe/Amsterdam")]
  
  # Time-based features
  dt[, hour := hour(interval)]
  dt[, weekday := wday(interval, week_start = 1)]
  dt[, month := month(interval)]
  dt[, weekend := as.integer(weekday %in% c(6, 7))]
  dt[, business_hours := as.integer(hour >= 7 & hour <= 19 & weekend == 0)]
  
  # Cyclical encoding
  dt[, hour_sin := sin(2 * pi * hour / 24)]
  dt[, hour_cos := cos(2 * pi * hour / 24)]
  dt[, dow_sin  := sin(2 * pi * weekday / 7)]
  dt[, dow_cos  := cos(2 * pi * weekday / 7)]
  dt[, month_sin := sin(2 * pi * month / 12)]
  dt[, month_cos := cos(2 * pi * month / 12)]
  
  # Lags
  dt[, lag_24 := shift(total_consumption_kWh, 24)] # 1 day
  dt[, lag_48 := shift(total_consumption_kWh, 48)] # 2 days
  dt[, lag_72 := shift(total_consumption_kWh, 72)] # 3 days
  dt[, lag_168 := shift(total_consumption_kWh, 168)] # 1 week
  dt[, lag_336 := shift(total_consumption_kWh, 336)] # 2 weeks
  dt[, lag_504 := shift(total_consumption_kWh, 504)] # 3 weeks
  
  # Long rolling means
  dt[, rollmean_24 := shift(frollmean(total_consumption_kWh, n = 24, align = "right", fill = NA), 1)]
  dt[, rollmean_168 := shift(frollmean(total_consumption_kWh, n = 168, align = "right", fill = NA), 1)]
  
  # Holidays (Netherlands + school + bridge days)
  nl_holidays <- as.Date(c(
    "2023-01-01", "2023-04-07", "2023-04-09", "2023-04-10", "2023-04-27",
    "2023-05-05", "2023-05-18", "2023-05-28", "2023-05-29", "2023-12-25", "2023-12-26",
    "2024-01-01", "2024-03-29", "2024-03-31", "2024-04-01", "2024-04-27",
    "2024-05-05", "2024-05-09", "2024-05-19", "2024-05-20", "2024-12-25", "2024-12-26"
  ))
  
  school_holidays <- as.Date(c(
    seq(as.Date("2023-04-29"), as.Date("2023-05-07"), by = "day"),
    seq(as.Date("2024-04-27"), as.Date("2024-05-05"), by = "day"),
    seq(as.Date("2023-07-08"), as.Date("2023-08-20"), by = "day"),
    seq(as.Date("2024-07-13"), as.Date("2024-08-25"), by = "day"),
    seq(as.Date("2023-12-23"), as.Date("2024-01-07"), by = "day"),
    seq(as.Date("2024-12-21"), as.Date("2025-01-05"), by = "day"),
    seq(as.Date("2024-02-17"), as.Date("2024-02-25"), by = "day")
  ))
  
  mandatory_days_off <- as.Date(c("2023-04-28", "2023-05-19", "2023-12-27",
                                  "2024-05-10", "2024-12-27"))
  
  all_holidays <- unique(c(nl_holidays, school_holidays, mandatory_days_off))
  
  dt[, date := as.Date(interval)]
  dt[, holiday := as.integer(date %in% all_holidays)]
  
  # Daylight Saving Time indicator (DST)
  dt[, dst := as.integer(dst(interval))]
  
  # Remove rows with any NA (from lags or rolling means)
  dt <- na.omit(dt)
  
  return(dt)
}
