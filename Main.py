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

# 🔥 Ask user for prediction horizon
while True:
    try:
        horizon = int(input("Enter number of days to predict (e.g., 1, 2, 7): "))
        if horizon > 0:
            break
        else:
            print("Please enter a positive number.")
    except ValueError:
        print("Invalid input. Please enter a number.")

# get training area (we will reserve last 'horizon' rows for short-term forecasting)
train_area = data.iloc[1:train_idx, :].values

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

# Print predictions with fake dates (since horizon can be arbitrary)
print("\n🔮 Predicted Close Prices:")
last_known_date = data.index[train_idx - 1]
future_dates = [last_known_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]

for i, p in enumerate(denorm_preds, start=1):
    print(f"{future_dates[i-1].strftime('%Y-%m-%d')}: {p:.4f}")

# Plot predictions
plt.figure(figsize=(8, 4))

if horizon == 1:
    # Just one point - show a dot with text
    plt.scatter(future_dates, denorm_preds, color='red', label='Predicted Close Price')
    plt.title("Prediction for Next Day")
    plt.ylabel("Close Price")
    plt.xticks([])  # Remove x-axis ticks
    # Annotate with date
    plt.text(0, denorm_preds[0], future_dates[0].strftime('%Y-%m-%d'),
             ha='center', va='bottom', fontsize=10)
else:
    # Multiple days - plot normal line chart
    plt.plot(future_dates, denorm_preds, label='Predicted (Attention)', marker='o')
    plt.title(f'Predictions for next {horizon} days')
    plt.xticks(rotation=45)

plt.legend()
plt.tight_layout()
plt.show()

