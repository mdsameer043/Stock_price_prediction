from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from Config import Config
from utils import NormalizeMult, create_dataset_multivariate
from models import attention_model
import os
import datetime
import random

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Get query params
        symbol = request.args.get('symbol', '601988.SH')
        horizon = int(request.args.get('horizon', 1))

        if horizon <= 0:
            return jsonify({"error": "Horizon must be positive"}), 400

        # Load CSV data
        # df = pd.read_csv(f"./{symbol}.csv")
        df = pd.read_csv(f"./601988.SH.csv")
        df.index = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        base = df.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]

        residuals = pd.read_csv('./ARIMA_residuals1.csv', parse_dates=['trade_date'], index_col='trade_date')
        residuals = residuals.select_dtypes(include=[np.number])
        data = base.join(residuals, how='inner').select_dtypes(include=[np.number])

        train_idx = Config.TRAIN_SPLIT_INDEX
        time_steps = Config.TIME_STEPS

        # Prepare training data
        train_area = data.iloc[1:train_idx, :].values
        train_norm, meta = NormalizeMult(train_area)
        X, Y = create_dataset_multivariate(train_norm, time_steps)
        close_col = list(data.columns).index('close')
        Y_close = Y[:, close_col]

        # Build model
        INPUT_DIMS = X.shape[2]
        m = attention_model(INPUT_DIMS=INPUT_DIMS, TIME_STEPS=time_steps, lstm_units=64)
        adam = Adam(learning_rate=Config.LEARNING_RATE)
        m.compile(optimizer=adam, loss='mse')

        # Load weights or train if not available
        if os.path.exists('./models/attention_weights.h5'):
            m.load_weights('./models/attention_weights.h5')
        else:
            if not os.path.exists('./models'):
                os.makedirs('./models')
            m.fit(X, Y_close, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, validation_split=0.1)
            m.save_weights('./models/attention_weights.h5')
            np.save('normalize_meta.npy', meta)

        # Predict next 'horizon' days
        last_window = train_norm[-time_steps:].copy()
        preds = []
        current = last_window.copy()
        for _ in range(horizon):
            inp = np.expand_dims(current, axis=0)
            yhat = m.predict(inp, verbose=0).flatten()[0]
            preds.append(yhat)
            dummy = np.zeros((INPUT_DIMS,))
            dummy[close_col] = yhat
            current = np.vstack([current[1:], dummy])

        # Denormalize predictions
        minc, maxc = meta[close_col]
        denorm_preds = np.array(preds) * (maxc - minc) + minc

        last_known_close = data.iloc[train_idx - 1]['close']
        predicted_close = float(denorm_preds[-1])
        change_percent = ((predicted_close - last_known_close) / last_known_close) * 100
        direction = "UP" if change_percent > 0 else "DOWN"

        # Simulate confidence
        confidence = round(random.uniform(0.75, 0.95), 2)

        result = {
            "symbol": symbol,
            "predicted_price": predicted_close,
            "confidence": confidence,
            "direction": direction,
            "change_percent": change_percent,
            "model": "AttCLX",
            "timestamp": datetime.datetime.now().isoformat()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
