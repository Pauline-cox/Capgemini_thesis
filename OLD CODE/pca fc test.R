# ==============================================================
# SARIMAX (CSS) — Train A fit + short 24h rolling forecast on Test A
# ==============================================================

library(data.table)
library(forecast)

# ---- helpers to clean design matrix ---------------------------
drop_zero_var <- function(DT) {
  stopifnot(is.data.table(DT))
  keep <- vapply(DT, function(z) sd(as.numeric(z)) > 0, logical(1))
  if (any(!keep)) {
    cat("Dropping zero-variance:", paste(names(DT)[!keep], collapse = ", "), "\n")
  }
  cols <- names(DT)[keep]
  DT[, ..cols]
}

drop_collinear <- function(DT, tol = 1e-7) {
  stopifnot(is.data.table(DT))
  X <- as.matrix(DT)
  q <- qr(X, tol = tol)
  cols <- colnames(X)[q$pivot[seq_len(q$rank)]]
  DT[, ..cols]
}

# 0) Target & frequency -----------------------------------------
y_train <- splits$trainA[[target_col]]
y_test  <- splits$testA[[target_col]]
y_ts    <- ts(y_train, frequency = PERIOD)

# 1) Short test slice for a quick sanity check ------------------
TEST_LIMIT <- 24                               # change to taste
test_idx   <- seq_len(min(TEST_LIMIT, nrow(splits$testA)))

xreg_tr <- copy(xreg_train_A)
xreg_te <- copy(xreg_test_A)[test_idx, ]

# 2) Clean xreg: drop zero-variance & collinear, keep same cols --
# (PCs are orthogonal to each other, but temporal/occupancy can create redundancies)
xreg_tr <- drop_zero_var(xreg_tr)
cols_train <- names(xreg_tr)
xreg_te <- xreg_te[, ..cols_train]

xreg_tr <- drop_collinear(xreg_tr)
cols_train2 <- names(xreg_tr)
xreg_te <- xreg_te[, ..cols_train2]

# 3) Scale with TRAIN stats -------------------------------------
Xtr <- as.matrix(xreg_tr)
Xte <- as.matrix(xreg_te)
Xtr_s <- scale(Xtr)
ctr  <- attr(Xtr_s, "scaled:center")
scl  <- attr(Xtr_s, "scaled:scale")
Xte_s <- scale(Xte, center = ctr, scale = scl)

# 4) Fit SARIMAX on TRAIN with CSS only -------------------------
fit_css <- Arima(
  y_ts,
  order    = ORDER,
  seasonal = list(order = SEASONAL, period = PERIOD),
  xreg     = Xtr_s,
  method   = "CSS"
)

cat("\n=== TRAIN A — SARIMAX (CSS, PCA-xreg) summary ===\n")
print(summary(fit_css))
cat("\n--- Coefficients ---\n")
print(round(coef(fit_css), 6))

# 5) Short 24h rolling forecast on the TEST slice ---------------
n_train <- length(y_train)
n_test  <- length(test_idx)
all_y   <- c(y_train, y_test[test_idx])

# build unified xreg matrix and scale with TRAIN params
all_X  <- rbind(Xtr, Xte)
all_Xs <- scale(all_X, center = ctr, scale = scl)

forecasts <- rep(NA_real_, n_test)
cat(sprintf("\nStarting 24h rolling forecast on first %d test hours...\n", n_test))

filled <- 0L
for (h in seq(-22L, n_test - 23L)) {
  cur <- n_train + h - 1L
  if (cur < 100L) next
  
  hist_y <- all_y[1:cur]
  hist_X <- all_Xs[1:cur, , drop = FALSE]
  fut_X  <- all_Xs[(cur+1):(cur+24), , drop = FALSE]
  if (nrow(fut_X) < 24) next
  
  fit_upd <- Arima(ts(hist_y, frequency = PERIOD),
                   model = fit_css, xreg = hist_X, method = "CSS")
  
  fc <- forecast(fit_upd, xreg = fut_X, h = 24)
  
  idx <- cur + 24 - n_train
  if (idx >= 1 && idx <= n_test) {
    forecasts[idx] <- fc$mean[24]; filled <- filled + 1L
  }
  
  if (filled %% 24 == 0 || filled == n_test) {
    cat(sprintf("Progress: %3d/%d (%.1f%%)\n", filled, n_test, 100*filled/n_test))
    flush.console()
  }
}

# 6) Backfill any NA forecasts (if early indices were skipped) --
for (i in which(is.na(forecasts))) forecasts[i] <- ifelse(i == 1, tail(y_train, 1), forecasts[i-1])

# 7) Quick peek --------------------------------------------------
dt_eval <- data.table(
  Time = test_idx,
  Actual = y_test[test_idx],
  Forecast = forecasts
)
print(head(dt_eval, 10))
s