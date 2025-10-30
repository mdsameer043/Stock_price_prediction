class Config:
    MODEL_NAME = "Attention"
    TIME_STEPS = 20          # keep 10-20 for fluctuations
    EPOCHS = 25              # increase slightly
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    TRAIN_SPLIT_INDEX = 1238 # full dataset if CSV has ~1200 rows
    PREDICTION_HORIZON = 2
    ARIMA_ORDER = (2, 1, 0)
    XGB_N_ESTIMATORS = 50
    NOISE_LEVEL = 0.001      # optional small noise
