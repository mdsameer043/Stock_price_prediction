import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Step 1: Load your original dataset
df = pd.read_csv("601988.SH.csv")
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df = df.set_index('trade_date')
df = df.sort_index()

# We'll use only 'close' price for ARIMA
close_prices = df['close']

# Step 2: Fit ARIMA model on training data
model = ARIMA(close_prices, order=(5,1,0))  # You can try different orders
model_fit = model.fit()

# Step 3: Make predictions
arima_pred = model_fit.predict(start=1, end=len(close_prices)-1, typ='levels')
arima_pred.index = close_prices.index[1:]

# Step 4: Calculate residuals
residuals = close_prices[1:] - arima_pred

# Step 5: Save predictions and residuals
arima_df = pd.DataFrame({'trade_date': arima_pred.index, 'close': arima_pred.values})
arima_df.to_csv('ARIMA.csv', index=False)

residuals_df = df.iloc[1:].copy()
residuals_df['close'] = residuals.values
residuals_df.reset_index(inplace=True)
residuals_df.to_csv('ARIMA_residuals1.csv', index=False)

# (Optional) Plot to visualize
plt.figure(figsize=(10,5))
plt.plot(close_prices, label='Actual')
plt.plot(arima_pred, label='ARIMA Predicted')
plt.legend()
plt.title('ARIMA Forecast vs Actual')
plt.show()
