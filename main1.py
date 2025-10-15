from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from Config import Config
from utils import NormalizeMult, create_dataset_multivariate
from models import attention_model
import os

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # 1️⃣ Read horizon from query param
        horizon = int(request.args.get('horizon', 1))
        if horizon <= 0:
            return jsonify({"error": "Horizon must be positive"}), 400

        # 2️⃣ Load base stock data
        df = pd.read_csv("./601988.SH.csv")
        df.index = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        base = df.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]

        # 3️⃣ Load residuals (from ARIMA)
        residuals = pd.read_csv('./ARIMA_residuals1.csv', parse_dates=['trade_date'], index_col='trade_date')
        residuals = residuals.select_dtypes(include=[np.number])

        # 4️⃣ Merge on index
        data = base.join(residuals, how='inner').select_dtypes(include=[np.number])

        # Splits
        train_idx = Config.TRAIN_SPLIT_INDEX
        time_steps = Config.TIME_STEPS

        # 5️⃣ Prepare training data
        train_area = data.iloc[1:train_idx, :].values
        train_norm, meta = NormalizeMult(train_area)
        X, Y = create_dataset_multivariate(train_norm, time_steps)
        close_col = list(data.columns).index('close')
        Y_close = Y[:, close_col]

        # 6️⃣ Build and compile model
        INPUT_DIMS = X.shape[2]
        m = attention_model(INPUT_DIMS=INPUT_DIMS, TIME_STEPS=time_steps, lstm_units=64)
        adam = Adam(learning_rate=Config.LEARNING_RATE)
        m.compile(optimizer=adam, loss='mse')

        # 7️⃣ Load weights if available (or train once)
        if os.path.exists('./models/attention_weights.h5'):
            m.load_weights('./models/attention_weights.h5')
        else:
            if not os.path.exists('./models'):
                os.makedirs('./models')
            m.fit(X, Y_close, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, validation_split=0.1)
            m.save_weights('./models/attention_weights.h5')
            np.save('normalize_meta.npy', meta)

        # 8️⃣ Predict next 'horizon' days
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

        # 9️⃣ Denormalize predictions
        meta_close = meta[close_col]
        minc, maxc = meta_close[0], meta_close[1]
        denorm_preds = np.array(preds) * (maxc - minc) + minc

        #  🔟 Format output
        last_known_date = data.index[train_idx - 1]
        future_dates = [last_known_date + pd.Timedelta(days=i) for i in range(1, horizon + 1)]

        results = [
            {"date": future_dates[i].strftime("%Y-%m-%d"), "predicted_close": float(denorm_preds[i])}
            for i in range(horizon)
        ]

        return jsonify({"status": "success", "predictions": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
