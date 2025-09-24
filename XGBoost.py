# XGBoost.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import NormalizeMult, DenormalizeMult, evaluation_metric, create_dataset_multivariate, data_split
from models import walk_forward_validation
from Config import Config
import xgboost as xgb

# Load data
data = pd.read_csv('./601988.SH.csv')
data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data = data.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]

# residuals & merge
residuals = pd.read_csv('./ARIMA_residuals1.csv', parse_dates=['trade_date'], index_col='trade_date')
merge = data.join(residuals, how='inner').select_dtypes(include=[np.number])

# Use only last part (starting from TRAIN_SPLIT_INDEX)
start_idx = Config.TRAIN_SPLIT_INDEX
horizon = Config.PREDICTION_HORIZON

# Ensure we have enough rows
if start_idx + horizon > len(merge):
    # fallback: use last available horizon rows
    merge_part = merge.iloc[-(horizon + Config.TIME_STEPS):]
else:
    merge_part = merge.iloc[start_idx - Config.TIME_STEPS:start_idx + horizon]

# Create supervised dataset for XGBoost residual prediction (n_in = 6 by previous code)
supervised = create_dataset_multivariate(merge_part.values, time_steps=6)
X, Y = supervised  # X shape (samples, 6, features), Y shape (samples, features)
# We'll flatten X to (samples, 6*features)
X_flat = X.reshape(X.shape[0], -1)
Y_close = Y[:, 3]  # assuming close index 3

# Split last 'horizon' rows as test
trainX = X_flat[:-horizon]
trainY = Y_close[:-horizon]
testX = X_flat[-horizon:]
testY = Y_close[-horizon:]

# Fit XGBoost one-step per sample (walk-forward style)
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=Config.XGB_N_ESTIMATORS)
model.fit(trainX, trainY)
yhat = model.predict(testX)

# Combine with ARIMA predictions
arima = pd.read_csv('./ARIMA.csv', parse_dates=['trade_date'], index_col='trade_date')
arima_vals = arima['close'].iloc[:len(yhat)].values  # align length

final_pred = arima_vals + yhat

# Ground truth close
truth_close = merge['close'].iloc[start_idx:start_idx + horizon].values

evaluation_metric(truth_close[:len(final_pred)], final_pred)

dates = merge.index[start_idx:start_idx + horizon]
plt.plot(dates, truth_close[:len(final_pred)], label='Actual')
plt.plot(dates, final_pred, label='ARIMA+XGB')
plt.legend()
plt.show()
