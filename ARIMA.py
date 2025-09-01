import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from pandas.plotting import autocorrelation_plot
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import metrics
from utils import *
from utils import evaluation_metric

# Load data
data = pd.read_csv('./601988.SH.csv')
test_set2 = data.loc[3501:, :]

# Set datetime index
data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data = data.drop(['ts_code', 'trade_date'], axis=1)
data = pd.DataFrame(data, dtype=np.float64)
# data.index.freq = 'B'  # Optional: business day frequency
data = data.asfreq('B').fillna(method='ffill')

# Split into training and test sets
training_set = data.loc['2007-01-04':'2021-06-21', :].copy()
test_set = data.loc['2021-06-22':, :]

# Plot raw close prices
plt.figure(figsize=(10, 6))
plt.plot(training_set['close'], label='training_set')
plt.plot(test_set['close'], label='test_set')
plt.title('Close price')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

# First-order differencing
training_set.loc[:, 'diff_1'] = training_set['close'].diff(1)
plt.figure(figsize=(10, 6))
training_set['diff_1'].plot()
plt.title('First-order diff')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_1', fontsize=14, horizontalalignment='center')
plt.show()

# Second-order differencing
training_set.loc[:, 'diff_2'] = training_set['diff_1'].diff(1)
plt.figure(figsize=(10, 6))
training_set['diff_2'].plot()
plt.title('Second-order diff')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_2', fontsize=14, horizontalalignment='center')
plt.show()

# White noise test
temp1 = np.diff(training_set['close'], n=1)
print(acorr_ljungbox(temp1, lags=2, boxpierce=True))  # White noise test

# ACF & PACF plots
acf_pacf_plot(training_set['close'], acf_lags=160)

# Create differenced dataframe
price = list(temp1)
data2 = {
    'trade_date': training_set['diff_1'].index[1:], 
    'close': price
}
df = pd.DataFrame(data2)
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y-%m-%d')
training_data_diff = df.set_index('trade_date')
print('&', training_data_diff)

acf_pacf_plot(training_data_diff)

# Train ARIMA model
model = sm.tsa.ARIMA(endog=training_set['close'], order=(2, 1, 0)).fit()

# Rolling Forecast
history = [x for x in training_set['close']]
predictions = []
for t in range(test_set.shape[0]):
    model1 = sm.tsa.ARIMA(history, order=(2, 1, 0))
    model_fit = model1.fit()
    yhat = model_fit.forecast()
    yhat = float(yhat[0])  # FIXED: removed deprecated np.float
    predictions.append(yhat)
    obs = test_set2.iloc[t, 5]
    history.append(obs)

# Save predictions
predictions1 = pd.DataFrame({
    'trade_date': test_set.index,
    'close': predictions
})
predictions1 = predictions1.set_index('trade_date')
predictions1.to_csv('./ARIMA.csv')

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(test_set['close'], label='Stock Price')
plt.plot(predictions1, label='Predicted Stock Price')
plt.title('ARIMA: Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

# Residual analysis
model2 = sm.tsa.ARIMA(endog=data['close'], order=(2, 1, 0)).fit()
residuals = pd.DataFrame(model2.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()
residuals.to_csv('./ARIMA_residuals1.csv')

# Evaluate model
evaluation_metric(test_set['close'], predictions)

# Stationarity tests
adf_test(np.array(training_set['close']))
adf_test(temp1)

# Fit on diff_1 values
predictions_ARIMA_diff = pd.Series(model.fittedvalues, copy=True)
predictions_ARIMA_diff = predictions_ARIMA_diff[3479:]

plt.figure(figsize=(10, 6))
plt.plot(training_data_diff, label="diff_1")
plt.plot(predictions_ARIMA_diff, label="prediction_diff_1")
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('diff_1', fontsize=14, horizontalalignment='center')
plt.title('DiffFit')
plt.legend()
plt.show()
