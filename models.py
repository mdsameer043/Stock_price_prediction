# models.py
import numpy as np
import xgboost as xgb
from keras.layers import (
    Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply,
    Permute, RepeatVector, Flatten, Lambda
)
from keras.models import Model, Sequential
from keras import backend as K
from utils import create_dataset_multivariate

# Attention block
def attention_3d_block(inputs, single_attention_vector=False):
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]

    a = Permute((2, 1))(inputs)                      # shape: (batch, features, time)
    a = Dense(time_steps, activation='softmax')(a)   # softmax over time
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

# Attention-based Conv1D + BiLSTM model for regression
def attention_model(INPUT_DIMS=13, TIME_STEPS=20, lstm_units=64, dropout=0.3):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x = Dropout(dropout)(x)

    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(dropout)(lstm_out)

    attention_mul = attention_3d_block(lstm_out)
    attention_flat = Flatten()(attention_mul)

    output = Dense(1, activation='linear')(attention_flat)
    model = Model(inputs=[inputs], outputs=output)
    return model

# Simple LSTM wrapper (returns Keras model)
def build_lstm_model(time_steps, input_dims, bidirectional=False, lstm_units=50):
    model = Sequential()
    if bidirectional:
        model.add(Bidirectional(LSTM(lstm_units, activation='tanh'), input_shape=(time_steps, input_dims)))
    else:
        model.add(LSTM(lstm_units, activation='tanh', input_shape=(time_steps, input_dims)))
    model.add(Dense(1))
    return model

# XGBoost helper used after attention (to refine predictions)
def xgboost_forecast(train, testX, n_estimators=50):
    train = np.asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators)
    model.fit(trainX, trainy)
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]

def walk_forward_validation(train_df, test_df, xgb_estimators=50):
    predictions = []
    history = [x for x in train_df.values]
    for i in range(len(test_df)):
        testX = test_df.iloc[i, :-1]
        testy = test_df.iloc[i, -1]
        yhat = xgboost_forecast(history, testX, n_estimators=xgb_estimators)
        predictions.append(yhat)
        history.append(test_df.iloc[i, :].values)
    return test_df.iloc[:, -1].values, predictions
