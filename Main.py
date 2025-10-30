# Main.py (Fluctuation-Preserving Version)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.losses import Huber
from Config import Config
from utils import NormalizeMult, create_dataset_multivariate
from models import attention_model
import os

# ---------------------- Helper Functions ----------------------
def standardize_columns(df):
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {
        'prev close': 'pre_close',
        'last': 'close',
        'volume': 'vol',
        'turnover': 'amount',
        '%deliverble': 'percent_deliverable'
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    return df

def compute_returns(df, col='close'):
    return df[col].pct_change().fillna(0).values

# ---------------------- User Input ----------------------
stock_name = input("Enter stock name (e.g., HDFC, TCS, ICICIBANK): ").upper()
file_path = f"./{stock_name}.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found!")

df = pd.read_csv(file_path)
df = standardize_columns(df)
df.index = pd.to_datetime(df['trade_date'] if 'trade_date' in df.columns else df['date'], dayfirst=False)

while True:
    try:
        horizon = int(input("Enter number of days to predict (e.g., 1, 2, 7, 30, 60): "))
        if horizon > 0:
            break
        print("Enter a positive number.")
    except ValueError:
        print("Invalid input. Enter a number.")

# ---------------------- Feature Selection ----------------------
required_cols = ['open', 'high', 'low', 'close', 'vol', 'amount', 'deliverable volume', 'percent_deliverable']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in CSV.")

base = df.loc[:, required_cols].copy()
base['return'] = compute_returns(base, col='close')
data = base.dropna().select_dtypes(include=[np.number])
train_idx = len(data)
print(f"Using full dataset: train_idx={train_idx}, dataset size={len(data)}")

# ---------------------- Shorter Sequence for Fluctuations ----------------------
time_steps = 20  # smaller window to capture short-term movements
features = data.drop(columns=['return']).values
returns = data['return'].values

# ---------------------- Normalize Features ----------------------
train_norm, meta = NormalizeMult(features)
mean_vec = np.array(meta[0])
std_vec = np.array(meta[1])

# Compute return mean/std separately
mean_return = returns.mean()
std_return = returns.std()

# ---------------------- Create Sequence Dataset ----------------------
full_data = np.hstack([train_norm, returns.reshape(-1, 1)])
X, Y_full = create_dataset_multivariate(full_data, time_steps)
Y_returns = Y_full[:, -1]  # last column is return

# Optional: add small Gaussian noise to features
noise_level = 0.001
X += np.random.normal(0, noise_level, X.shape)

# ---------------------- Build & Train Model ----------------------
INPUT_DIMS = X.shape[2]
model = attention_model(INPUT_DIMS=INPUT_DIMS, TIME_STEPS=time_steps, lstm_units=64)
model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE), loss=Huber())
model.summary()
model.fit(X, Y_returns, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, validation_split=0.1)

# ---------------------- Predict Next 'horizon' Days ----------------------
last_window = full_data[-time_steps:, :]  # shape: (time_steps, num_features)
pred_returns = []
current = last_window.copy()
last_price = data['close'].iloc[-1]

for _ in range(horizon):
    inp = np.expand_dims(current, axis=0)  # shape: (1, time_steps, num_features)
    yhat_norm = model.predict(inp, verbose=0).flatten()[0]

    # Denormalize return
    denorm_return = yhat_norm * std_return + mean_return
    denorm_return = np.clip(denorm_return, -0.05, 0.05)
    pred_returns.append(denorm_return)

    # Update sequence: shift left and append new row
    next_row = current[-1, :].copy()
    next_row[-1] = yhat_norm  # update return column
    current = np.vstack([current[1:], next_row])  # shape stays (time_steps, num_features)

# ---------------------- Reconstruct Predicted Prices ----------------------
pred_prices = []
price = last_price
for r in pred_returns:
    price = price * (1 + r)
    pred_prices.append(price)

# ---------------------- Display Predictions ----------------------
print(f"\n🔮 Predicted Close Prices for {stock_name}:")
last_known_date = data.index[-1]
future_dates = [last_known_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]
for d, p in zip(future_dates, pred_prices):
    print(f"{d.strftime('%Y-%m-%d')}: {p:.4f}")

# ---------------------- Plot ----------------------
plt.figure(figsize=(10, 5))
plt.plot(future_dates, pred_prices, label='Predicted Close', marker='o', color='red')
plt.title(f'{stock_name} Predicted Close Prices for next {horizon} days')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
