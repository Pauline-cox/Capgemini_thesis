# Function for hybrid SARIMA–LSTM hyperparameter tuning on residuals 
hybrid_bayesopt_train <- function(train_data, val_data,
                                  feature_columns,
                                  sarima_order = c(2,0,2),
                                  sarima_seasonal = c(1,1,1),
                                  period = 168,
                                  HORIZON = 24) {
  
  cat("\n=== Running Hybrid SARIMA–LSTM Optimization ===\n")
  cat(sprintf("SARIMA order: (%s)(%s)[%d]\n", paste(sarima_order, collapse=","),
              paste(sarima_seasonal, collapse=","), period))

  # Start runtime tracking
  start_time <- Sys.time()
  
  # Fit SARIMA and compute residuals
  y_train <- ts(train_data$total_consumption_kWh, frequency = period)
  sarima_model <- try(
    Arima(y_train,
          order = sarima_order,
          seasonal = list(order = sarima_seasonal, period = period),
          method = "CSS"),
    silent = TRUE
  )
  
  if (inherits(sarima_model, "try-error")) {
    cat("ARIMA fitting failed, using default residuals = target.\n")
    train_data[, resid := total_consumption_kWh]
    val_data[,   resid := total_consumption_kWh]
  } else {
    residuals_train <- as.numeric(residuals(sarima_model))
    residuals_val   <- val_data$total_consumption_kWh -
      as.numeric(forecast(sarima_model, h = nrow(val_data))$mean)
    train_data[, resid := residuals_train]
    val_data[,   resid := residuals_val]
  }
  
  # Add lagged residuals
  train_data[, `:=`(
    resid_lag_24  = shift(resid, 24,  NA, "lag"),
    resid_lag_48  = shift(resid, 48,  NA, "lag"),
    resid_lag_168 = shift(resid, 168, NA, "lag")
  )]
  
  last_train <- tail(train_data$resid, 168)
  val_data[, `:=`(
    resid_lag_24  = shift(resid, 24,  NA, "lag"),
    resid_lag_48  = shift(resid, 48,  NA, "lag"),
    resid_lag_168 = shift(resid, 168, NA, "lag")
  )]
  
  val_data[1:168, resid_lag_24  := rep(tail(last_train, 24),  length.out = 168)]
  val_data[1:168, resid_lag_48  := rep(tail(last_train, 48),  length.out = 168)]
  val_data[1:168, resid_lag_168 := rep(last_train,          length.out = 168)]
  
  # Preprocess geatures for LSTM
  feature_columns <- intersect(feature_columns, names(train_data))
  train_data[, target := resid]
  val_data[,   target := resid]
  
  rec <- recipe(target ~ ., data = rbind(train_data, val_data)[, c("target", feature_columns), with=FALSE]) %>%
    step_range(all_numeric(), -all_outcomes()) %>%
    prep(training = train_data)
  
  Xtr <- bake(rec, train_data)[, feature_columns, with=FALSE]
  Xva <- bake(rec, val_data)[, feature_columns, with=FALSE]
  ytr <- train_data$target
  yva <- val_data$target
  
  # Scale residual target
  y_min <- min(ytr); y_max <- max(ytr)
  scale_y  <- function(y) (y - y_min) / (y_max - y_min + 1e-6)
  ytr_s <- scale_y(ytr); yva_s <- scale_y(yva)
  
  # Sequence maker
  make_seq <- function(X, y, lookback, horizon) {
    n <- nrow(X) - lookback - horizon + 1
    if (n <= 0) return(NULL)
    Xarr <- array(NA_real_, dim = c(n, lookback, ncol(X)))
    Yarr <- array(NA_real_, dim = c(n, horizon))
    for (i in 1:n) {
      Xarr[i,,] <- as.matrix(X[i:(i+lookback-1),])
      Yarr[i, ] <- y[(i+lookback):(i+lookback+horizon-1)]
    }
    list(X = Xarr, y = Yarr)
  }
  
  # LSTM model builder
  build_model <- function(input_shape, units1, units2, dropout, lr, opt_name="nadam") {
    opt <- switch(opt_name,
                  "adam"  = optimizer_adam(learning_rate=lr),
                  "nadam" = optimizer_nadam(learning_rate=lr),
                  "sgd"   = optimizer_sgd(learning_rate=lr, momentum=0.9),
                  optimizer_nadam(learning_rate=lr))
    
    keras_model_sequential() %>%
      layer_lstm(units=as.integer(units1), input_shape=input_shape, return_sequences=TRUE) %>%
      layer_dropout(rate=dropout) %>%
      layer_lstm(units=as.integer(units2), return_sequences=FALSE) %>%
      layer_dense(units=HORIZON, activation="linear") %>%
      compile(optimizer=opt, loss="mse", metrics="mae")
  }
  
  # Bayesian optimization objective
  .trial_counter <- 0
  bo_objective <- function(units1, units2, dropout, lr, lookback_id, batch_size_id, opt_id) {
    .trial_counter <<- .trial_counter + 1
    t0 <- Sys.time()
    
    lookback <- LOOKBACKS[round(lookback_id)]
    batch_sz <- c(16, 32, 64)[round(batch_size_id)]
    opt_name <- c("adam", "nadam", "sgd")[round(opt_id)]
    
    cat(sprintf(
      "Trial %d | units1=%d, units2=%d, dropout=%.3f, lr=%.5f, lookback=%d, batch=%d, opt=%s\n",
      .trial_counter, as.integer(units1), as.integer(units2),
      dropout, lr, lookback, batch_sz, opt_name
    ))
    flush.console()
    
    tr <- make_seq(Xtr, ytr_s, lookback, HORIZON)
    va <- make_seq(rbind(Xtr, Xva), c(ytr_s, yva_s), lookback, HORIZON)
    if (is.null(tr) || is.null(va)) return(list(Score = -1e6))
    
    model <- build_model(c(lookback, ncol(Xtr)), units1, units2, dropout, lr, opt_name)
    
    history <- model %>% fit(
      tr$X, tr$y,
      epochs = 50, batch_size = as.integer(batch_sz),
      validation_data = list(va$X, va$y),
      verbose = 0,
      callbacks = list(
        callback_early_stopping(patience = 6, restore_best_weights = TRUE),
        callback_reduce_lr_on_plateau(factor = 0.5, patience = 3)
      )
    )
    
    val_mae <- tail(history$metrics$val_mae, 1)
    elapsed <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 2)
    cat(sprintf("Done in %ss | Val MAE=%.4f\n", elapsed, val_mae))
    
    k_clear_session(); gc()
    list(Score = -as.numeric(val_mae))
  }
  
  # Define search space
  bounds <- list(
    units1     = c(32, 128),
    units2     = c(16, 64),
    dropout    = c(0.0, 0.3),
    lr         = c(1e-4, 5e-3),
    lookback   = c(96, 168, 336),   
    batch_size = c(16, 128),                         
    opt_id     = c(1, 3)                              
  )
  
  # Run Bayesian optimization
  cat("Starting Bayesian Optimization (Hybrid SARIMA–LSTM)")
  opt_res <- bayesOpt(
    FUN = bo_objective,
    bounds = bounds,
    initPoints = 10,
    iters.n = 20,
    acq = "ei",
    verbose = 1
  )
  
  total_time <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 2)
  cat("Optimization complete. Total optimization time:", total_time, "minutes\n\n")
  
  best_pars  <- getBestPars(opt_res)
  best_score <- getBestScore(opt_res)
  
  cat("Best parameters (Hybrid SARIMA–LSTM)")
  print(best_pars)
  cat("Best (negative) Val MAE:", best_score, "\n")
  
  # -------------------------------------------------------------
  # 10. Return Results
  # -------------------------------------------------------------
  list(
    opt_results = opt_res,
    best_params = best_pars,
    best_score  = best_score,
    runtime_min = total_time
  )
}
