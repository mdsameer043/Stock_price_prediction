# Config.py - single source of truth for hyperparams & indices
class Config:
    # Model selection: "Attention", "LSTM" (single), "BiLSTM"
    MODEL_NAME = "Attention"

    # General training params
    TIME_STEPS = 20            # look-back window
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001

    # Data split (index where test set begins). Set according to your CSV length
    TRAIN_SPLIT_INDEX = 3500   # previously 3500

    # Prediction horizon: 1 (next day), 2 (two days), 7 (one week)
    PREDICTION_HORIZON = 2

    # ARIMA order (if you want to change)
    ARIMA_ORDER = (2, 1, 0)

    # XGBoost params
    XGB_N_ESTIMATORS = 50
