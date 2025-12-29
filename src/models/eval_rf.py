import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.features.rolling_features import make_rolling_features

def model_path(fd: str, window: int) -> str:
    return f"models/rf_{fd}_w{window}.joblib"

def eval_test(fd="FD001", window=30):
    model = joblib.load(model_path(fd, window))
    test_df = pd.read_parquet(f"data/processed/test_{fd}.parquet")
    
    feats = make_rolling_features(test_df, window=window)
    X_test = feats.drop(columns=["RUL", "engine_id", "cycle"])
    y_test = feats["RUL"].to_numpy()
    
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    
    print(f"{fd} TEST | MAE: {mae:.2f} RMSE: {rmse:.2f}")
    
if __name__ == "__main__":
    eval_test()