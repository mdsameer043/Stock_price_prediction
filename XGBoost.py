import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from model import walk_forward_validation

# Load stock data
data = pd.read_csv('./601988.SH.csv')
data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data = data.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]
# data = pd.DataFrame(data, dtype=np.float64)
close = data.pop('close')
data.insert(5, 'close', close)
data1 = data.iloc[3501:, 5]

# Load residuals
residuals = pd.read_csv('./ARIMA_residuals1.csv')
residuals.index = pd.to_datetime(residuals['trade_date'])
residuals.pop('trade_date')

# Merge data and residuals
merge_data = pd.merge(data, residuals, on='trade_date')

# Drop any non-numeric columns to avoid errors in XGBoost
merge_data = merge_data.select_dtypes(include=[np.number])

#merge_data = merge_data.drop(labels='2007-01-04', axis=0)
time = pd.Series(data.index[3501:])

# Load ARIMA predictions
Lt = pd.read_csv('./ARIMA.csv')
Lt = Lt.drop('trade_date', axis=1)
Lt = np.array(Lt)
Lt = Lt.flatten().tolist()

# Prepare train/test datasets for supervised learning
train, test = prepare_data(merge_data, n_test=180, n_in=6, n_out=1)

# Perform walk-forward validation using XGBoost
y, yhat = walk_forward_validation(train, test)

# Plot residuals and predicted residuals
plt.figure(figsize=(10, 6))
plt.plot(time, y, label='Residuals')
plt.plot(time, yhat, label='Predicted Residuals')
plt.title('ARIMA+XGBoost: Residuals Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Residuals', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

# Combine ARIMA and predicted residuals to get final stock price prediction
finalpredicted_stock_price = [i + j for i, j in zip(Lt, yhat)]
#print('final', finalpredicted_stock_price)

# Evaluate prediction accuracy
evaluation_metric(data1, finalpredicted_stock_price)

# Plot real vs predicted stock prices
plt.figure(figsize=(10, 6))
plt.plot(time, data1, label='Stock Price')
plt.plot(time, finalpredicted_stock_price, label='Predicted Stock Price')
plt.title('ARIMA+XGBoost: Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()
