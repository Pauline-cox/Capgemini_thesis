SARIMA

> print(summary(best_model))
Series: y_full 
ARIMA(2,0,2)(0,1,1)[168] 

Coefficients:
  ar1      ar2      ma1     ma2     sma1
1.5874  -0.6285  -0.7058  0.0677  -0.7019
s.e.  0.0638   0.0571   0.0645  0.0109   0.0079

sigma^2 = 781.6:  log likelihood = -48781.5

Training set error measures:
  ME     RMSE      MAE       MPE     MAPE      MASE          ACF1
Training set 0.1480216 27.72383 15.77464 -1.921473 9.424473 0.4512325 -0.0006226916

--- Residual Diagnostics ---
  > lb_test <- Box.test(residuals_fit, lag = 24, type = "Ljung-Box", 
                        +                     fitdf = length(ORDER) + length(SEASONAL))
> print(lb_test)

Box-Ljung test

data:  residuals_fit
X-squared = 649.36, df = 18, p-value < 2.2e-16

SARIMAX

AIC: 97381.06 | BIC: 97388.30 | LogLik: -48689.53
Series: y_train_ts 
Regression with ARIMA(2,0,2)(0,1,1)[168] errors 

Coefficients:
  ar1              ar2              ma1              ma2             sma1              co2   business_hours  total_occupancy         hour_sin         hour_cos  
0.9721          -0.0882          -0.1522           0.0475          -0.7625           8.4199           7.7771          21.4920         150.8226          80.9982  
lux          holiday              dst  
1.1119          -8.0207         -12.2514  

sigma^2 = 723.3:  log likelihood = -48689.53
AIC=97381.06   AICc=97381.06   BIC=97388.3

Training set error measures:
  ME    RMSE      MAE       MPE    MAPE      MASE          ACF1
Training set 0.02423233 26.6772 15.66943 -1.874092 9.74826 0.4618467 -0.0003762868
> 
  --- Residual Diagnostics ---
  
  Box-Ljung test

data:  residuals_fit
X-squared = 572.29, df = 18, p-value < 2.2e-16


Residuals are likely *not* white noise (p < 0.05). Consider adjusting model order.



sarimax with pca
AIC: 97447.31 | BIC: 97454.55 | LogLik: -48722.65
Series: y_train_ts 
Regression with ARIMA(2,0,2)(0,1,1)[168] errors 

Coefficients:
  ar1             ar2             ma1             ma2            sma1             PC1             PC2             PC3             PC4             PC5             PC6  
0.0969          0.6939          0.7301         -0.0404         -0.7645         -6.0373        -13.4547         -4.6142         -5.9499         -3.3371          1.7082  
PC7  business_hours        hour_sin        hour_cos         holiday             dst  
-2.1191          6.7462        144.3919         63.9852         -8.1501         -9.7948  

sigma^2 = 727.8:  log likelihood = -48722.65
AIC=97447.31   AICc=97447.31   BIC=97454.55

Training set error measures:
  ME    RMSE      MAE      MPE     MAPE      MASE         ACF1
Training set 0.08688318 26.7614 15.72785 -1.77247 9.876306 0.4635684 -0.001872431

--- Residual Diagnostics ---
  
  Box-Ljung test

data:  residuals_fit
X-squared = 626.04, df = 7, p-value < 2.2e-16

SARIMA with clustering
AIC: 97549.19 | BIC: 97556.43 | LogLik: -48773.60
Series: y_train_ts 
Regression with ARIMA(2,0,2)(0,1,1)[168] errors 

Coefficients:
  ar1             ar2             ma1             ma2            sma1       cluster_1       cluster_2  business_hours        hour_sin        hour_cos         holiday  
0.1197          0.6853          0.7231         -0.0323         -0.7607          1.1187          1.0956          5.6661        138.2019         51.4264         -9.2749  
dst  
-12.6600  

sigma^2 = 735.2:  log likelihood = -48773.6
AIC=97549.19   AICc=97549.19   BIC=97556.43

Training set error measures:
  ME    RMSE      MAE       MPE     MAPE      MASE         ACF1
Training set 0.09360403 26.8967 15.53811 -1.725713 9.563256 0.4579762 -0.001542435

--- Residual Diagnostics ---
  
  Box-Ljung test

data:  residuals_fit
X-squared = 667.85, df = 12, p-value < 2.2e-16
