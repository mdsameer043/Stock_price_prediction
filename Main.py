# Main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from Config import Config
from utils import NormalizeMult, create_dataset_multivariate, evaluation_metric
from models import attention_model
import os

# Load base stock data
df = pd.read_csv("./601988.SH.csv")
df.index = pd.to_datetime(df['trade_date'], format='%Y%m%d')
base = df.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]

# load residuals (from ARIMA)
residuals = pd.read_csv('./ARIMA_residuals1.csv', parse_dates=['trade_date'], index_col='trade_date')
residuals = residuals.select_dtypes(include=[np.number])

# merge on index
data = base.join(residuals, how='inner').select_dtypes(include=[np.number])

# Splits
train_idx = Config.TRAIN_SPLIT_INDEX
time_steps = Config.TIME_STEPS
horizon = Config.PREDICTION_HORIZON

# get training area (we will reserve last 'horizon' rows for short-term forecasting)
train_area = data.iloc[1:train_idx, :].values
test_area = data.iloc[train_idx:train_idx + horizon, :].values

# normalize
train_norm, meta = NormalizeMult(train_area)

# create sequence dataset
X, Y = create_dataset_multivariate(train_norm, time_steps)
close_col = list(data.columns).index('close')
Y_close = Y[:, close_col]

# Build model
INPUT_DIMS = X.shape[2]
m = attention_model(INPUT_DIMS=INPUT_DIMS, TIME_STEPS=time_steps, lstm_units=64)
adam = Adam(learning_rate=Config.LEARNING_RATE)
m.compile(optimizer=adam, loss='mse')
m.summary()

# Train
history = m.fit(X, Y_close, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, validation_split=0.1)

# Save model weights
if not os.path.exists('./models'):
    os.makedirs('./models')
m.save_weights('./models/attention_weights.h5')
np.save('normalize_meta.npy', meta)

# Predict next 'horizon' days
last_window = train_norm[-time_steps:].copy()
preds = []
current = last_window.copy()
for i in range(horizon):
    inp = np.expand_dims(current, axis=0)
    yhat = m.predict(inp).flatten()[0]
    preds.append(yhat)
    dummy = np.zeros((INPUT_DIMS,))
    dummy[close_col] = yhat
    current = np.vstack([current[1:], dummy])

# Denormalize predictions
meta_close = meta[close_col]
minc, maxc = meta_close[0], meta_close[1]
denorm_preds = np.array(preds) * (maxc - minc) + minc

# Actual close prices
true_close = data['close'].iloc[train_idx:train_idx + horizon].values
dates = data.iloc[train_idx:train_idx + horizon].index

# Print predictions with dates
print("\n🔮 Predicted Close Prices:")
for d, p in zip(dates, denorm_preds):
    print(f"{d.strftime('%Y-%m-%d')}: {p:.4f}")

# Evaluate
evaluation_metric(true_close[:len(denorm_preds)], denorm_preds)

# Plot
plt.plot(dates, true_close[:len(denorm_preds)], label='Actual')
plt.plot(dates, denorm_preds, label='Predicted (Attention)')
plt.title(f'Predictions for next {horizon} days')
plt.legend()
plt.show()
