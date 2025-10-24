# ==============================================================================
# RESIDUAL DIAGNOSTICS: SARIMA & SARIMAX
# ==============================================================================

library(data.table)
library(ggplot2)
library(moments)
library(tseries)
library(knitr)
library(gridExtra)

theme_set(theme_bw(base_size = 11) +
            theme(panel.grid.minor = element_blank(),
                  plot.title = element_text(face = "bold", size = 12, hjust = 0.5),
                  strip.background = element_rect(fill = "grey92"),
                  strip.text = element_text(face = "bold", size = 10),
                  legend.position = "bottom"))

cat("Loading SARIMA and SARIMAX model results...\n")
SARIMA  <- readRDS("RESULTS_FC_SARIMA.rds")
SARIMAX <- readRDS("RESULTS_FC_SARIMAX.rds")

models <- list(SARIMA, SARIMAX)
model_names <- c("SARIMA", "SARIMAX")
names(models) <- model_names

# ------------------------------------------------------------------------------
# Residual extraction helper
# ------------------------------------------------------------------------------
extract_residuals <- function(model, name, period) {
  # Try different possible locations of residuals
  res <- NULL
  if (!is.null(model[[period]]$residuals)) res <- model[[period]]$residuals
  else if (!is.null(model[[period]]$model$residuals)) res <- model[[period]]$model$residuals
  
  if (is.null(res)) {
    warning(sprintf("No residuals found for %s - %s", name, period))
    return(NULL)
  }
  
  # Convert to a consistent data.table format
  if (is.vector(res) || is.numeric(res)) {
    dt <- data.table(Residual = as.numeric(res))
  } else if (is.data.frame(res)) {
    dt <- as.data.table(res)
    res_col <- grep("resid|error", names(dt), ignore.case = TRUE, value = TRUE)
    if (length(res_col) == 0) stop("No residual column found in residual object")
    setnames(dt, res_col[1], "Residual")
  } else {
    stop("Unknown residual format")
  }
  
  dt[, Model := name]
  return(dt)
}

# ------------------------------------------------------------------------------
# Compute diagnostic statistics
# ------------------------------------------------------------------------------
calc_diag <- function(df) {
  if (!"Model" %in% names(df)) stop("Input data must include a 'Model' column.")
  df[, .(
    Mean      = mean(Residual, na.rm = TRUE),
    SD        = sd(Residual, na.rm = TRUE),
    Skewness  = skewness(Residual, na.rm = TRUE),
    Kurtosis  = kurtosis(Residual, na.rm = TRUE),
    JB_p      = tryCatch(jarque.bera.test(Residual)$p.value, error = function(e) NA),
    LB_p      = tryCatch(Box.test(Residual, lag = 24, type = "Ljung-Box")$p.value, error = function(e) NA)
  ), by = Model][order(Model)]
}

# ------------------------------------------------------------------------------
# Extract residuals for both periods
# ------------------------------------------------------------------------------
res_A <- lapply(seq_along(models), \(i) extract_residuals(models[[i]], model_names[i], "period_A"))
res_B <- lapply(seq_along(models), \(i) extract_residuals(models[[i]], model_names[i], "period_B"))

# Combine
all_res_A <- rbindlist(res_A, fill = TRUE)
all_res_B <- rbindlist(res_B, fill = TRUE)

cat("✓ Residuals extracted successfully\n")

# ------------------------------------------------------------------------------
# Residual diagnostics summary table
# ------------------------------------------------------------------------------
cat("\n\nTABLE 1: RESIDUAL DIAGNOSTICS SUMMARY\n\n")

diag_A <- calc_diag(all_res_A)
diag_B <- calc_diag(all_res_B)

diag_A[, Period := "A (Stable)"]
diag_B[, Period := "B (Volatile)"]
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

print(kable(res_diag_fmt, format = "pipe"))

# ------------------------------------------------------------------------------
# PLOTS
# ------------------------------------------------------------------------------
cat("\n\nPLOT 1: RESIDUAL DISTRIBUTIONS & ACF\n\n")

plot_residuals <- function(df, title) {
  p1 <- ggplot(df, aes(x = Residual, fill = Model, color = Model)) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, alpha = 0.4, position = "identity") +
    geom_density(alpha = 0.3, linewidth = 0.8) +
    labs(x = "Residuals", y = "Density", title = paste(title, "- Histogram & Density")) +
    theme_minimal(base_size = 11)
  
  p2 <- ggplot(df, aes(sample = Residual, color = Model)) +
    stat_qq(alpha = 0.7) + stat_qq_line(linewidth = 0.6) +
    labs(title = paste(title, "- Q–Q Plot"), x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_minimal(base_size = 11)
  
  grid.arrange(p1, p2, ncol = 2)
}

plot_residuals(all_res_A, "Period A")
plot_residuals(all_res_B, "Period B")

# ------------------------------------------------------------------------------
# Residual time series plots
# ------------------------------------------------------------------------------
plot_acf <- function(df, title) {
  ggplot(df, aes(x = seq_along(Residual), y = Residual, color = Model)) +
    geom_line(alpha = 0.7) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = paste(title, "- Residual Time Series"), x = "Time", y = "Residual") +
    theme_minimal(base_size = 11)
}

print(plot_acf(all_res_A, "Residuals – Period A"))
print(plot_acf(all_res_B, "Residuals – Period B"))
