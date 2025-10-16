# ===========================================================
# Cyclical Encoding Visualization Script
# ===========================================================

# --- Hour of day ---
hour_dt <- data.table(hour = 0:23)
hour_dt[, hour_sin := sin(2 * pi * hour / 24)]
hour_dt[, hour_cos := cos(2 * pi * hour / 24)]

p_hour <- ggplot(hour_dt, aes(x = hour)) +
  geom_line(aes(y = hour_sin, color = "sin(hour)"), linewidth = 1) +
  geom_line(aes(y = hour_cos, color = "cos(hour)"), linewidth = 1) +
  labs(
    x = "Hour of Day",
    y = "Value"
  ) +
  scale_color_manual(values = c("sin(hour)" = "#1f77b4", "cos(hour)" = "#ff7f0e")) +
  theme_minimal(base_size = 13) +
  theme(legend.title = element_blank())

# --- Day of week ---
dow_dt <- data.table(weekday = 1:7)
dow_dt[, dow_sin := sin(2 * pi * weekday / 7)]
dow_dt[, dow_cos := cos(2 * pi * weekday / 7)]

p_dow <- ggplot(dow_dt, aes(x = weekday)) +
  geom_line(aes(y = dow_sin, color = "sin(dow)"), linewidth = 1) +
  geom_line(aes(y = dow_cos, color = "cos(dow)"), linewidth = 1) +
  scale_x_continuous(breaks = 1:7, labels = c("Mon","Tue","Wed","Thu","Fri","Sat","Sun")) +
  labs(
    x = "Day of Week",
    y = "Value"
  ) +
  scale_color_manual(values = c("sin(dow)" = "#2ca02c", "cos(dow)" = "#d62728")) +
  theme_minimal(base_size = 13) +
  theme(legend.title = element_blank())

# --- Month of year ---
month_dt <- data.table(month = 1:12)
month_dt[, month_sin := sin(2 * pi * month / 12)]
month_dt[, month_cos := cos(2 * pi * month / 12)]

p_month <- ggplot(month_dt, aes(x = month)) +
  geom_line(aes(y = month_sin, color = "sin(month)"), linewidth = 1) +
  geom_line(aes(y = month_cos, color = "cos(month)"), linewidth = 1) +
  scale_x_continuous(breaks = 1:12,
                     labels = c("Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec")) +
  labs(
    x = "Month of Year",
    y = "Value"
  ) +
  scale_color_manual(values = c("sin(month)" = "#9467bd", "cos(month)" = "#8c564b")) +
  theme_minimal(base_size = 13) +
  theme(legend.title = element_blank())

# --- Display plots ---
print(p_hour)
print(p_dow)
print(p_month)
