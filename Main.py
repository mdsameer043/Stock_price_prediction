# Main.py (Fixed & Tuned for HDFCBANK.csv)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.losses import Huber
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
        '%deliverble': 'percent_deliverable',
        '%deliverable': 'percent_deliverable',
        'deliverable volume': 'deliverable_volume',
        'date': 'trade_date'
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    return df


def compute_returns(df, col='close'):
    return df[col].pct_change().fillna(0).values


# ---------------------- User Input ----------------------
stock_name = input("Enter stock name (e.g., HDFCBANK, HDFC, TCS): ").upper()
file_path = f"./{stock_name}.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found!")

df = pd.read_csv(file_path)
df = standardize_columns(df)

# Detect date column
date_col = "trade_date" if "trade_date" in df.columns else "date"
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])
df.index = df[date_col]

while True:
    try:
        horizon = int(input("Enter number of days to predict (e.g., 1, 3, 7): "))
        if horizon > 0:
            break
        print("Enter a positive number.")
    except ValueError:
        print("Invalid input. Enter a number.")


# ---------------------- Feature Selection ----------------------
# Only keep numeric columns that actually exist
possible_cols = ['open', 'high', 'low', 'close', 'vol', 'amount']
available_cols = [c for c in possible_cols if c in df.columns]

if len(available_cols) < 4:
    raise ValueError(f"Not enough numeric columns found in {file_path}. Found: {available_cols}")

base = df[available_cols].copy()
base['return'] = compute_returns(base, col='close')
data = base.dropna().select_dtypes(include=[np.number])

# ---------------------- Parameters ----------------------
time_steps = 20
split_ratio = 0.8  # 80% train, 20% test

# ---------------------- Normalize Features ----------------------
features = data.drop(columns=['return']).values
returns = data['return'].values

features_norm, meta = NormalizeMult(features)
mean_return = returns.mean()
std_return = returns.std() if returns.std() != 0 else 1.0

full_data = np.hstack([features_norm, returns.reshape(-1, 1)])
X, Y_full = create_dataset_multivariate(full_data, time_steps)
Y_returns = Y_full[:, -1]

# ---------------------- Train/Test Split ----------------------
train_size = int(len(X) * split_ratio)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y_returns[:train_size], Y_returns[train_size:]

print(f"📊 Dataset size={len(X)}, Train={len(X_train)}, Test={len(X_test)}")

# ---------------------- Build & Train Model ----------------------
INPUT_DIMS = X.shape[2]
model = attention_model(INPUT_DIMS=INPUT_DIMS, TIME_STEPS=time_steps, lstm_units=64)
model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber())

history = model.fit(
    X_train, Y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)


# ---------------------- Performance Metrics ----------------------
def evaluate_model(model, X, Y, std_return, mean_return, label):
    pred = model.predict(X, verbose=0).flatten()
    pred_denorm = pred * std_return + mean_return
    Y_denorm = Y * std_return + mean_return

    mae = mean_absolute_error(Y_denorm, pred_denorm)
    rmse = np.sqrt(mean_squared_error(Y_denorm, pred_denorm))
    r2 = r2_score(Y_denorm, pred_denorm)
    acc = 1 - (mae / (np.max(Y_denorm) - np.min(Y_denorm) + 1e-6))

    print(f"\n📈 {label} Performance:")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"R²   : {r2:.6f}")
    print(f"Accuracy (approx): {acc * 100:.2f}%")


evaluate_model(model, X_train, Y_train, std_return, mean_return, "Training")
evaluate_model(model, X_test, Y_test, std_return, mean_return, "Testing")

# ---------------------- Predict Next 'horizon' Days ----------------------
last_window = full_data[-time_steps:, :]
pred_returns = []
current = last_window.copy()
last_price = float(data['close'].iloc[-1])

for _ in range(horizon):
    inp = np.expand_dims(current, axis=0)
    yhat_norm = model.predict(inp, verbose=0).flatten()[0]
    denorm_return = yhat_norm * std_return + mean_return
    denorm_return = np.clip(denorm_return, -0.05, 0.05)
    pred_returns.append(denorm_return)

    next_row = current[-1, :].copy()
    next_row[-1] = yhat_norm
    current = np.vstack([current[1:], next_row])

# ---------------------- Reconstruct Predicted Prices ----------------------
pred_prices = []
price = last_price
for r in pred_returns:
    price *= (1 + r)
    pred_prices.append(price)

# ---------------------- Save Model Weights ----------------------
os.makedirs("./models", exist_ok=True)
model.save_weights("./models/hdfc_attention_weights.h5")
print("✅ Model weights saved at ./models/hdfc_attention_weights.h5")

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
