# ARIMA.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from utils import acf_pacf_plot, evaluation_metric, adf_test
from Config import Config

# Load data
data = pd.read_csv('./601988.SH.csv')
data.index = pd.to_datetime(data['trade_date'], format='%Y%m%d')
data = data.drop(['ts_code', 'trade_date'], axis=1)
data = data.select_dtypes(include=[np.number]).astype(np.float64)
data = data.asfreq('B').fillna(method='ffill')

# Split
training_set = data.loc['2007-01-04':'2021-06-21', :].copy()
test_set = data.loc['2021-06-22':, :]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(training_set['close'], label='training_set')
plt.plot(test_set['close'], label='test_set')
plt.legend()
plt.show()

# differencing
training_set['diff_1'] = training_set['close'].diff(1)
acf_pacf_plot(training_set['close'], acf_lags=80, pacf_lags=80)

# Fit ARIMA on training close
order = Config.ARIMA_ORDER
model = sm.tsa.ARIMA(endog=training_set['close'], order=order).fit()

# Rolling forecast: produce predictions only for required horizon or entire test set depending usage.
# Here we produce one-step forecasts for the length of test_set (downstream code slices)
history = list(training_set['close'].values)
predictions = []
for t in range(len(test_set)):
    model_temp = sm.tsa.ARIMA(history, order=order)
    model_fit = model_temp.fit()
    yhat = model_fit.forecast(steps=1)
    predictions.append(float(yhat[0]))
    # append actual observation from test_set to history for next step
    history.append(test_set['close'].iloc[t])

predictions_df = pd.DataFrame({'trade_date': test_set.index, 'close': predictions}).set_index('trade_date')
predictions_df.to_csv('./ARIMA.csv')

# Residuals from ARIMA fitted on full data (for downstream residual modeling)
model_full = sm.tsa.ARIMA(endog=data['close'], order=order).fit()
residuals = pd.DataFrame(model_full.resid)
residuals.index = data.index
residuals.to_csv('./ARIMA_residuals1.csv', index_label='trade_date')

# Evaluate
evaluation_metric(test_set['close'], predictions)

# Stationarity
adf_test(training_set['close'].dropna())
