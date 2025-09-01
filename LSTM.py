import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras.optimizers import Adam
from numpy.random import seed
from utils import data_split, evaluation_metric

# --- Define lstm model builder ---
def lstm(model_type, X_train, yuan_X_train):
    n_timestamp = X_train.shape[1]
    n_features = X_train.shape[2]

    model = Sequential()
    yuan_model = Sequential()

    if model_type == 1:
        model.add(LSTM(units=50, input_shape=(n_timestamp, n_features)))
        yuan_model.add(LSTM(units=50, input_shape=(n_timestamp, yuan_X_train.shape[2])))
    elif model_type == 2:
        model.add(LSTM(units=50, return_sequences=True, input_shape=(n_timestamp, n_features)))
        model.add(LSTM(units=50))
        yuan_model.add(LSTM(units=50, return_sequences=True, input_shape=(n_timestamp, yuan_X_train.shape[2])))
        yuan_model.add(LSTM(units=50))
    elif model_type == 3:
        model.add(Bidirectional(LSTM(units=50), input_shape=(n_timestamp, n_features)))
        yuan_model.add(Bidirectional(LSTM(units=50), input_shape=(n_timestamp, yuan_X_train.shape[2])))

    model.add(Dense(1))
    yuan_model.add(Dense(1))
    return model, yuan_model

# --- GPU memory control ---
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.set_visible_devices([gpus[0]], "GPU")

# --- Set random seed ---
seed(1)
tf.random.set_seed(1)

# --- Parameters ---
n_timestamp = 10
n_epochs = 10
model_type = 3

# --- Load and preprocess data ---
yuan_data = pd.read_csv('./601988.SH.csv')
yuan_data.index = pd.to_datetime(yuan_data['trade_date'], format='%Y%m%d')
yuan_data = yuan_data.loc[:, ['open', 'high', 'low', 'close', 'amount']]

data = pd.read_csv('./ARIMA_residuals1.csv')
data.index = pd.to_datetime(data['trade_date'])
data = data.drop('trade_date', axis=1)

Lt = pd.read_csv('./ARIMA.csv')

idx = 3500
training_set = data.iloc[1:idx, :]
test_set = data.iloc[idx:, :]
yuan_training_set = yuan_data.iloc[1:idx, :]
yuan_test_set = yuan_data.iloc[idx:, :]

sc = MinMaxScaler(feature_range=(0, 1))
yuan_sc = MinMaxScaler(feature_range=(0, 1))

training_set_scaled = sc.fit_transform(training_set.select_dtypes(include=[np.number]))
testing_set_scaled = sc.transform(test_set.select_dtypes(include=[np.number]))

yuan_training_set_scaled = yuan_sc.fit_transform(yuan_training_set)
yuan_testing_set_scaled = yuan_sc.transform(yuan_test_set)

X_train, y_train = data_split(training_set_scaled, n_timestamp)
X_test, y_test = data_split(testing_set_scaled, n_timestamp)

yuan_X_train, yuan_y_train = data_split(yuan_training_set_scaled, n_timestamp)
yuan_X_test, yuan_y_test = data_split(yuan_testing_set_scaled, n_timestamp)

# Reshape yuan data for multivariate
yuan_X_train = yuan_X_train.reshape(yuan_X_train.shape[0], yuan_X_train.shape[1], 5)
yuan_X_test = yuan_X_test.reshape(yuan_X_test.shape[0], yuan_X_test.shape[1], 5)

# --- Model creation ---
model, yuan_model = lstm(model_type, X_train, yuan_X_train)
print(model.summary())

# Separate optimizers
adam1 = Adam(learning_rate=0.01)
adam2 = Adam(learning_rate=0.01)
model.compile(optimizer=adam1, loss='mse')
yuan_model.compile(optimizer=adam2, loss='mse')

# --- Training ---
history = model.fit(X_train, y_train, batch_size=32, epochs=n_epochs, validation_data=(X_test, y_test))
yuan_history = yuan_model.fit(yuan_X_train, yuan_y_train, batch_size=32, epochs=n_epochs,
                              validation_data=(yuan_X_test, yuan_y_test))

# --- Predictions and inverse transform ---
yuan_predicted = yuan_model.predict(yuan_X_test)
yuan_predicted_full = np.zeros((yuan_predicted.shape[0], 5))
yuan_predicted_full[:, 3] = yuan_predicted[:, 0]
yuan_predicted_stock_price = yuan_sc.inverse_transform(yuan_predicted_full)[:, 3]

yuan_real = yuan_y_test.reshape(-1, 1)
yuan_real_full = np.zeros((yuan_real.shape[0], 5))
yuan_real_full[:, 3] = yuan_real[:, 0]
yuan_real_stock_price = yuan_sc.inverse_transform(yuan_real_full)[:, 3]

# --- Residual model predictions ---
predicted = model.predict(X_test)

# Get the correct shape that scaler was fit on
n_scaled_features = training_set_scaled.shape[1]  # likely 19

# Create dummy with same number of columns
dummy_pred = np.zeros((predicted.shape[0], n_scaled_features))

# Insert predicted values in the correct column (assuming first one holds residual)
dummy_pred[:, 0] = predicted[:, 0]

# Now apply inverse_transform safely
predicted_stock_price = sc.inverse_transform(dummy_pred)[:, 0]

# Properly aligned time indices for predictions
aligned_index_residual = data.index[idx + n_timestamp: idx + n_timestamp + len(predicted_stock_price)]
aligned_index_yuan = yuan_data.index[idx + n_timestamp: idx + n_timestamp + len(yuan_real_stock_price)]

predicted_stock_price1 = pd.DataFrame({
    'trade_date': aligned_index_residual,
    'close': predicted_stock_price
}).set_index('trade_date')

# Align lengths in case of mismatch
min_len_pred_yuan = min(len(aligned_index_yuan), len(yuan_predicted_stock_price))
yuan_predicted_stock_price1 = pd.DataFrame({
    'trade_date': aligned_index_yuan[:min_len_pred_yuan],
    'close': yuan_predicted_stock_price[:min_len_pred_yuan]
}).set_index('trade_date')

min_len_real_yuan = min(len(aligned_index_yuan), len(yuan_real_stock_price))
yuan_real_stock_price1 = pd.DataFrame({
    'trade_date': aligned_index_yuan[:min_len_real_yuan],
    'close': yuan_real_stock_price[:min_len_real_yuan]
}).set_index('trade_date')

real_stock_price = sc.inverse_transform(y_test)

# --- Combine ARIMA and Residuals ---
final_predicted = pd.concat([Lt, predicted_stock_price1])
final_predicted = final_predicted.groupby('trade_date')['close'].sum().reset_index()
final_predicted.index = pd.to_datetime(final_predicted['trade_date'])
final_predicted = final_predicted.drop('trade_date', axis=1)

# --- Plot predictions ---
plt.figure(figsize=(10, 6))
plt.plot(yuan_data.loc['2021-06-22':, 'close'], label='Actual Stock Price')
plt.plot(final_predicted['close'], label='Predicted Stock Price (ARIMA + Residual LSTM)')
plt.title('BiLSTM: Final Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Close')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(yuan_real_stock_price1['close'], label='Actual Stock Price')
plt.plot(yuan_predicted_stock_price1['close'], label='Predicted Stock Price (LSTM only)')
plt.title('LSTM: Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Close')
plt.legend()
plt.show()

# --- Evaluation ---
# --- Evaluation (align by date index) ---
yhat = yuan_data.loc['2021-06-22':, 'close']
yhat.index = pd.to_datetime(yhat.index)

# Align predictions with ground truth using intersection of dates
common_index = final_predicted.index.intersection(yhat.index)
evaluation_metric(final_predicted.loc[common_index, 'close'], yhat.loc[common_index])
