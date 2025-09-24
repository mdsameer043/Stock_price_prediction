# utils.py
import numpy as np
import pandas as pd
from sklearn import metrics
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import matplotlib.pyplot as plt

def adf_test(series):
    t = adfuller(series)
    output = {
        'Test Statistic': t[0],
        'p-value': t[1],
        'Lags Used': t[2],
        'Number of Observations': t[3],
        'Critical Values': t[4]
    }
    print(pd.Series(output))
    return output

def acf_pacf_plot(seq, acf_lags=40, pacf_lags=40):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(seq, lags=acf_lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(seq, lags=pacf_lags, ax=ax2)
    plt.show()

def create_dataset_multivariate(data, time_steps):
    """
    Create sequences for multivariate time-series.
    data: numpy array shape (samples, features)
    returns X (samples-time_steps, time_steps, features) and y (samples-time_steps, features)
    """
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = data.shape[1] if hasattr(data, 'shape') else 1
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # inputs
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # outputs
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def prepare_data(supervised_df, n_test):
    """
    Given a supervised dataframe (from series_to_supervised),
    split into train and test based on index positions.
    """
    train = supervised_df.iloc[:n_test].copy()
    test = supervised_df.iloc[n_test:].copy()
    return train, test

def evaluation_metric(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    MSE = metrics.mean_squared_error(y_true, y_pred)
    RMSE = np.sqrt(MSE)
    MAE = metrics.mean_absolute_error(y_true, y_pred)
    R2 = metrics.r2_score(y_true, y_pred)
    print('MSE: %.5f' % MSE)
    print('RMSE: %.5f' % RMSE)
    print('MAE: %.5f' % MAE)
    print('R2: %.5f' % R2)
    return {'MSE': MSE, 'RMSE': RMSE, 'MAE': MAE, 'R2': R2}

def NormalizeMult(data):
    """
    Vectorized normalization (min-max) column-wise.
    data: numpy array or pandas DataFrame
    returns normalized array and the min/max array for inverse
    """
    arr = np.array(data, dtype='float64')
    mins = np.nanmin(arr, axis=0)
    maxs = np.nanmax(arr, axis=0)
    denom = (maxs - mins)
    denom[denom == 0] = 1.0
    norm = (arr - mins) / denom
    meta = np.stack([mins, maxs], axis=1)
    return norm, meta

def DenormalizeMult(normed, meta):
    mins = meta[:, 0]
    maxs = meta[:, 1]
    denom = (maxs - mins)
    denom[denom == 0] = 1.0
    return normed * denom + mins

def data_split(sequence, n_timestamp):
    """
    Safe data_split that handles 1D or 2D arrays.
    Returns X (samples, n_timestamp, features) and y (samples, features)
    """
    X, y = [], []
    seq = np.array(sequence)
    for i in range(len(seq) - n_timestamp):
        X.append(seq[i:i + n_timestamp])
        y.append(seq[i + n_timestamp])
    return np.array(X), np.array(y)
