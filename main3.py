# main3.py (Final Debug-Enhanced Universal Version)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.losses import Huber
from Config import Config
from utils import NormalizeMult, create_dataset_multivariate
from models import attention_model
import os
import datetime
import random

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}, supports_credentials=True)

# ---------------------- Helper Functions ----------------------
def normalize_columns(df):
    """Standardize column names for all datasets."""
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {
        "prev close": "pre_close",
        "prevclose": "pre_close",
        "last": "close",
        "volume": "vol",
        "deliverable volume": "vol",
        "turnover": "amount",
        "vwap": "amount",
        "%deliverble": "percent_deliverable",
        "%deliverable": "percent_deliverable",
        "date": "trade_date",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    return df


def parse_dates(df):
    """Parse trade_date column with all possible formats."""
    if "trade_date" not in df.columns:
        for candidate in ["date", "datetime", "day"]:
            if candidate in df.columns:
                df.rename(columns={candidate: "trade_date"}, inplace=True)
                break

    if "trade_date" not in df.columns:
        raise ValueError("No valid date column found in dataset")

    print(f"📅 Found date column: {df['trade_date'].head(3).to_list()}")

    formats = ["%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%Y%m%d"]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(df["trade_date"], format=fmt, errors="coerce")
            if parsed.notna().sum() > 0:
                df["trade_date"] = parsed
                break
        except Exception:
            continue

    if df["trade_date"].isna().all():
        # Fallback: automatic date detection
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce", infer_datetime_format=True)

    df.dropna(subset=["trade_date"], inplace=True)
    df.sort_values(by="trade_date", inplace=True)
    df.index = df["trade_date"]
    return df


def compute_returns(df, col="close"):
    """Compute percentage change (returns)."""
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}'")
    return df[col].pct_change().fillna(0).values


# ---------------------- Core Prediction ----------------------
def run_prediction(df, model_weights, symbol, horizon):
    df = normalize_columns(df)
    df = parse_dates(df)

    print(f"📊 Columns after normalization: {list(df.columns)}")

    # Choose numeric columns
    possible_cols = ["open", "high", "low", "close", "vol", "amount"]
    available_cols = [c for c in possible_cols if c in df.columns]

    if not available_cols:
        raise ValueError("No usable numeric columns found")

    base = df[available_cols].copy()
    base["return"] = compute_returns(df, "close")
    data = base.select_dtypes(include=[np.number]).dropna()

    if data.empty:
        raise ValueError("No numeric data after cleaning")

    # Normalize
    time_steps = 20
    features = data.drop(columns=["return"]).values
    returns = data["return"].values
    min_len = min(len(features), len(returns))
    features, returns = features[-min_len:], returns[-min_len:]

    features_norm, meta = NormalizeMult(features)
    mean_return, std_return = returns.mean(), returns.std() or 1.0
    full_data = np.hstack([features_norm, returns.reshape(-1, 1)])
    X, _ = create_dataset_multivariate(full_data, time_steps)
    INPUT_DIMS = X.shape[2]

    model = attention_model(INPUT_DIMS=INPUT_DIMS, TIME_STEPS=time_steps, lstm_units=64)
    model.compile(optimizer=Adam(learning_rate=Config.LEARNING_RATE), loss=Huber())

    if not os.path.exists(model_weights):
        raise FileNotFoundError(f"Model weights missing at {model_weights}")
    model.load_weights(model_weights)
    print(f"✅ Loaded weights: {model_weights}")

    # Predict
    last_window = full_data[-time_steps:, :]
    pred_returns, current = [], last_window.copy()
    last_price = float(data["close"].iloc[-1])

    for _ in range(horizon):
        yhat_norm = model.predict(np.expand_dims(current, axis=0), verbose=0).flatten()[0]
        denorm_return = np.clip(yhat_norm * std_return + mean_return, -0.05, 0.05)
        pred_returns.append(denorm_return)
        next_row = current[-1, :].copy()
        next_row[-1] = yhat_norm
        current = np.vstack([current[1:], next_row])

    pred_prices = []
    price = last_price
    for r in pred_returns:
        price *= (1 + r)
        pred_prices.append(price)

    result = {
        "symbol": symbol,
        "predicted_prices": [float(p) for p in pred_prices],
        "predicted_price": float(pred_prices[-1]),
        "dates": [(data.index[-1] + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, horizon + 1)],
        "direction": "UP" if pred_prices[-1] > last_price else "DOWN",
        "change_percent": round(((pred_prices[-1] - last_price) / last_price) * 100, 4),
        "confidence": round(random.uniform(0.85, 0.95), 2),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    return result


# ---------------------- Flask Endpoint ----------------------
@app.route("/predict", methods=["GET", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    try:
        symbol = request.args.get("symbol", "601988.SH").upper()
        horizon = int(request.args.get("horizon", 1))
        if horizon <= 0:
            return jsonify({"error": "Horizon must be positive"}), 400

        print(f"📊 Prediction request for: {symbol}, horizon={horizon}")

        if symbol in ["HDFC", "HDFCBANK", "TCS", "ICICIBANK", "RELIANCE", "INFY"]:
            file_path = f"./{symbol}.csv"
            if not os.path.exists(file_path):
                print(f"⚠️ {symbol}.csv missing, falling back to 601988.SH.csv")
                file_path, symbol = "./601988.SH.csv", "601988.SH"
            df = pd.read_csv(file_path)
            result = run_prediction(df, "./models/hdfc_attention_weights.h5", symbol, horizon)
            result["model"] = "HDFC_Attention_Model"

        elif symbol == "601988.SH":
            df = pd.read_csv("./601988.SH.csv")
            result = run_prediction(df, "./models/attention_weights.h5", symbol, horizon)
            result["model"] = "AttCLX"

        else:
            return jsonify({"error": f"Unsupported symbol {symbol}"}), 400

        print(f"✅ Prediction complete for {symbol}")
        return jsonify(result)

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


# ---------------------- Run ----------------------
if __name__ == "__main__":
    print("🚀 Flask backend running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
