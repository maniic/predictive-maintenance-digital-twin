import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.features.rolling_features import make_rolling_features

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray, a1: float = 10.0, a2: float = 13.0) -> float:
    delta = y_pred - y_true
    score = np.where(
        delta >= 0,
        np.exp(delta / a1) - 1.0,
        np.exp(-delta / a2) - 1.0
    )
    return float(np.sum(score))

def model_path(fd: str, window: int, include_slope: bool) -> str:
    suffix = "_slope" if include_slope else ""
    return f"models/rf_{fd}_w{window}{suffix}.joblib"

def eval_test(fd="FD001", window=30, include_slope=False):
    model = joblib.load(model_path(fd, window, include_slope))
    test_df = pd.read_parquet(f"data/processed/test_{fd}.parquet")
    
    feats = make_rolling_features(test_df, window=window, include_slope=include_slope)
    X_test = feats.drop(columns=["RUL", "engine_id", "cycle"])
    y_test = feats["RUL"].to_numpy()
    
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    score = nasa_score(y_test, pred)
    
    print(f"{fd} TEST | MAE: {mae:.2f} RMSE: {rmse:.2f} NASA Score: {score:.2f}")
    
if __name__ == "__main__":
    eval_test()