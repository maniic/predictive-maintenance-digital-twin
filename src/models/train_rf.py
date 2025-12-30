from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

from src.features.rolling_features import make_rolling_features

def model_path(fd: str, window: int, include_slope: bool) -> str:
    suffix = "_slope" if include_slope else ""
    return f"models/rf_{fd}_w{window}{suffix}.joblib"

def train(fd="FD001", window=30, include_slope=False):
    train_df = pd.read_parquet(f"data/processed/train_{fd}.parquet")
    feats = make_rolling_features(train_df, window=window, include_slope=include_slope)
    
    X = feats.drop(columns=["RUL"])
    y = feats["RUL"].to_numpy()
    groups = feats["engine_id"].to_numpy()
    
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, va_idx = next(splitter.split(X, y, groups=groups))
    
    X_train = X.iloc[tr_idx].drop(columns=["engine_id", "cycle"])
    y_train = y[tr_idx]
    X_val = X.iloc[va_idx].drop(columns=["engine_id", "cycle"])
    y_val = y[va_idx]
    
    model = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    
    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    print(f"{fd} VAL | MAE: {mae:.2f} RMSE: {rmse:.2f}")
    
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, model_path(fd, window, include_slope))
    print("Saved model to", model_path(fd, window, include_slope))

if __name__ == "__main__":
    train()