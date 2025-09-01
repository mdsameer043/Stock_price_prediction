from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from utils import *
from model import *

# Load and prepare base stock data
data1 = pd.read_csv("./601988.SH.csv")
data1.index = pd.to_datetime(data1['trade_date'], format='%Y%m%d')
data1 = data1.loc[:, ['open', 'high', 'low', 'close', 'vol', 'amount']]
data_yuan = data1.copy()


# Load ARIMA residuals and merge on trade_date
residuals = pd.read_csv('./ARIMA_residuals1.csv')
residuals.index = pd.to_datetime(residuals['trade_date'])
residuals.pop('trade_date')

# Merge and remove suffixes
data1 = pd.merge(data1, residuals, left_index=True, right_index=True)


# print(data1.head(5))
# a=data1.head(1)
# for items in a:
#     print(items)
    
# exit()


# Rename columns after merge (if residuals caused name collisions)
data1.columns = [col.replace('_x', '').replace('_y', '') for col in data1.columns]

# Split data into train/test and ensure all columns are numeric
data = data1.iloc[1:3500, :].select_dtypes(include=[np.number])
data2 = data1.iloc[3500:, :].select_dtypes(include=[np.number])

TIME_STEPS = 20

# Normalize training data
data, normalize = NormalizeMult(data)
print('#', normalize)

# Extract 'close' column for prediction (Y label)
pollution_data = data[:, 3].reshape(len(data), 1)  # Assuming 'close' is at index 3

# Create training sequences
train_X, _ = create_dataset(data, TIME_STEPS)
_, train_Y = create_dataset(pollution_data, TIME_STEPS)

print(train_X.shape, train_Y.shape)

# Build and compile the attention model
m = attention_model(INPUT_DIMS=data.shape[1])  # Dynamically use feature count
m.summary()
adam = Adam(learning_rate=0.01)
m.compile(optimizer=adam, loss='mse')

# Train the model
history = m.fit([train_X], train_Y, epochs=10, batch_size=64, validation_split=0.1)

# Save model and normalization parameters
m.save("./stock_model.h5")
np.save("stock_normalize.npy", normalize)

# Plot loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Prediction and evaluation
class Config:
    def __init__(self):
        self.dimname = 'close'  # Now 'close' is restored in the merged DataFrame

config = Config()
name = config.dimname

y_hat, y_test = PredictWithData(data2, data_yuan, name, 'stock_model.h5', data.shape[1])
y_hat = np.array(y_hat, dtype='float64')
y_test = np.array(y_test, dtype='float64')

# Evaluation
# evaluation_metric(y_test, y_hat)


# Additional Evaluation Metrics
mae = metrics.mean_absolute_error(y_test, y_hat)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_hat))
r2 = metrics.r2_score(y_test, y_hat)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot predicted vs actual prices
time = pd.Series(data1.index[3499:])
plt.plot(time, y_test, label='True')
plt.plot(time, y_hat, label='Prediction')
plt.title('Hybrid model prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Price', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()


