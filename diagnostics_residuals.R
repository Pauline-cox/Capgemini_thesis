# ==============================================================
# Author: Pauline Cox
# Script: diagnostics_residuals.R
#
# Description: Computes and visualizes residual diagnostics for 
# SARIMA and SARIMAX models across two test periods. Includes 
# distributional statistics, normality and independence tests, 
# and visual assessment via histograms, density plots, and Q–Q plots.
#
# Input:
#   - Trained model result objects
#
# Output:
#   - Residual diagnostic table (summary statistics)
#   - Diagnostic plots (distribution, Q–Q, and ACF)
# ==============================================================

# Set theme
theme_set(theme_bw(base_size = 11) +
            theme(panel.grid.minor = element_blank(),
                  plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
                  strip.background = element_rect(fill = "grey92"),
                  strip.text = element_text(face = "bold", size = 10),
                  legend.position = "bottom"))

# --- Load Model Results ---
SARIMA  <- readRDS("RESULTS/RESULTS_FC_SARIMA.rds")
SARIMAX <- readRDS("RESULTS/RESULTS_FC_SARIMAX.rds")

models <- list(SARIMA = SARIMA, SARIMAX = SARIMAX)

# Function to extract residuals
extract_residuals <- function(model, name, period) {
  res <- model[[period]]$residuals
  if (is.null(res)) res <- model[[period]]$model$residuals
  
  if (is.vector(res) || is.numeric(res)) {
    dt <- data.table(Residual = as.numeric(res))
  } else if (is.data.frame(res)) {
    dt <- as.data.table(res)
    res_col <- grep("resid|error", names(dt), ignore.case = TRUE, value = TRUE)
    setnames(dt, res_col[1], "Residual")
  }
  
  dt[, Model := name]
  return(dt)
}


# --- Diagnostic Statistics ---
calc_diag <- function(df) {
  df[, .(
    Mean      = mean(Residual, na.rm = TRUE),
    SD        = sd(Residual, na.rm = TRUE),
    Skewness  = skewness(Residual, na.rm = TRUE),
    Kurtosis  = kurtosis(Residual, na.rm = TRUE),
    JB_p      = jarque.bera.test(Residual)$p.value,
    LB_p      = Box.test(Residual, lag = 24, type = "Ljung-Box")$p.value
  ), by = Model][order(Model)]
}

# Extract residuals for each period
res_A <- lapply(names(models), function(n) extract_residuals(models[[n]], n, "period_A"))
res_B <- lapply(names(models), function(n) extract_residuals(models[[n]], n, "period_B"))

all_res_A <- rbindlist(res_A, fill = TRUE)
all_res_B <- rbindlist(res_B, fill = TRUE)

# Compute diagnostics 
diag_A <- calc_diag(all_res_A)
diag_B <- calc_diag(all_res_B)

diag_A[, Period := "A"]
diag_B[, Period := "B"]
res_diag <- rbind(diag_A, diag_B)

res_diag_fmt <- res_diag[, .(
  Period,
  Model,
  Mean      = sprintf("%.3f", Mean),
  SD        = sprintf("%.3f", SD),
  Skewness  = sprintf("%.3f", Skewness),
  Kurtosis  = sprintf("%.3f", Kurtosis),
  JB_p      = sprintf("%.6f", JB_p),
  LB_p      = sprintf("%.6f", LB_p)
)]

# Display residual summary
print(kable(res_diag_fmt, format = "pipe"))

# --- Residual distribution and QQ plots ---
plot_residuals <- function(df, title) {
  p1 <- ggplot(df, aes(x = Residual, fill = Model, color = Model)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.4, position = "identity") +
    geom_density(alpha = 0.3, linewidth = 0.8) +
    labs(x = "Residuals", y = "Density", title = paste(title, "- Histogram & Density")) +
    theme_minimal(base_size = 11)
  
  p2 <- ggplot(df, aes(sample = Residual, color = Model)) +
    stat_qq(alpha = 0.7) +
    stat_qq_line(linewidth = 0.6) +
    labs(title = paste(title, "- Q–Q Plot"), x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_minimal(base_size = 11)
  
  grid.arrange(p1, p2, ncol = 2)
}

plot_residuals(all_res_A, "Period A")
plot_residuals(all_res_B, "Period B")
