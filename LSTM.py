# LSTM.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.optimizers import Adam
from numpy.random import seed
from utils import data_split, NormalizeMult, DenormalizeMult, evaluation_metric
from Config import Config

# GPU memory setting omitted for clarity (keep if needed)

seed(1)
np.random.seed(1)

# params
n_timestamp = Config.TIME_STEPS
n_epochs = Config.EPOCHS
batch_size = Config.BATCH_SIZE
lr = Config.LEARNING_RATE
model_type_choice = Config.MODEL_NAME  # "BiLSTM" or "LSTM"
train_idx = Config.TRAIN_SPLIT_INDEX
horizon = Config.PREDICTION_HORIZON

# Load
yuan_data = pd.read_csv('./601988.SH.csv')
yuan_data.index = pd.to_datetime(yuan_data['trade_date'], format='%Y%m%d')
yuan_data = yuan_data.loc[:, ['open', 'high', 'low', 'close', 'amount']]

residuals = pd.read_csv('./ARIMA_residuals1.csv', parse_dates=['trade_date'], index_col='trade_date')
residuals = residuals.select_dtypes(include=[np.number])

# split according to train_idx and horizon - ensure enough rows exist
train_res = residuals.iloc[1:train_idx, :].values
test_res = residuals.iloc[train_idx:train_idx + horizon, :].values

yuan_train = yuan_data.iloc[1:train_idx, :].values
yuan_test = yuan_data.iloc[train_idx:train_idx + horizon, :].values

# normalize
train_res_norm, res_meta = NormalizeMult(train_res)
yuan_train_norm, yuan_meta = NormalizeMult(yuan_train)

# create sequences
X_train, y_train = data_split(train_res_norm, n_timestamp)
X_yuan_train, y_yuan_train = data_split(yuan_train_norm, n_timestamp)

# build model
if model_type_choice.lower().startswith('bi'):
    model = Sequential()
    model.add(Bidirectional(LSTM(50), input_shape=(n_timestamp, X_train.shape[2])))
else:
    model = Sequential()
    model.add(LSTM(50, input_shape=(n_timestamp, X_train.shape[2])))

model.add(Dense(X_train.shape[2]))  # predict multivariate residuals
adam = Adam(learning_rate=lr)
model.compile(optimizer=adam, loss='mse')
model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

# Predict residuals for the horizon using sliding window from end of training set
# Build input window from last n_timestamp rows of training normalized data
last_window = train_res_norm[-n_timestamp:].copy()
predicted_residuals = []
current_window = last_window.copy()
for i in range(horizon):
    input_window = np.expand_dims(current_window, axis=0)  # shape (1, time_steps, features)
    pred = model.predict(input_window)
    predicted_residuals.append(pred.flatten())
    # slide
    current_window = np.vstack([current_window[1:], pred])

predicted_residuals = np.array(predicted_residuals)  # shape (horizon, features)

# Now combine ARIMA predictions + residuals
arima_df = pd.read_csv('./ARIMA.csv', parse_dates=['trade_date'], index_col='trade_date')
arima_preds = arima_df.iloc[:].loc[residuals.index[train_idx:train_idx + horizon], 'close'].values
# if arima length shorter, take last horizon values
if len(arima_preds) < horizon:
    arima_preds = arima_df['close'].iloc[-horizon:].values

# We'll assume the residual for 'close' column is at the same positional index as 'close' in residuals dataframe:
close_col_index = list(residuals.columns).index('close') if 'close' in residuals.columns else 0
pred_close_residuals = predicted_residuals[:, close_col_index]

final_pred = arima_preds + pred_close_residuals

# evaluate - use yuan_test close values
true_close = yuan_test[:, 3]  # index 3 is 'close' per selection above
evaluation_metric(true_close[:len(final_pred)], final_pred)

# Plot
dates = residuals.index[train_idx:train_idx + horizon]
plt.plot(dates, true_close[:len(final_pred)], label='Actual Close')
plt.plot(dates, final_pred, label='ARIMA + LSTM Residuals')
plt.legend()
plt.show()
