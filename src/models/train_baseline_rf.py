from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

SENSORS = [f"sensor_{i}" for i in range(1, 22)]
SETTINGS = [f"setting_{i}" for i in range(1, 4)]
FEATURE_COLS = SENSORS + SETTINGS

def make_rolling_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Rolling mean/std over last `windows` cycles for each engine.
    """
    df = df.sort_values(["engine_id", "cycle"]).copy()
    
    mean_df = (
        df.groupby("engine_id")[FEATURE_COLS]
        .rolling(window=window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
        .add_suffix(f"_mean{window}")
    )
    
    std_df = (
        df.groupby("engine_id")[FEATURE_COLS]
        .rolling(window=window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
        .add_suffix(f"_std{window}")
    )
    
    out = pd.concat([df[["engine_id", "cycle", "RUL"]], mean_df, std_df], axis=1)
    return out.dropna().reset_index(drop=True)

def main():
    fd = "FD001"
    window = 30
    
    train = pd.read_parquet(f"data/processed/train_{fd}.parquet")
    feats = make_rolling_features(train, window=window)
    
    X = feats.drop(columns=["RUL"])
    y = feats["RUL"].to_numpy()
    groups = feats["engine_id"].to_numpy()
    
    # Split by engine so validation engines are unseen
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    
    X_train = X.iloc[train_idx].drop(columns=["engine_id", "cycle"])
    y_train = y[train_idx]
    X_val = X.iloc[val_idx].drop(columns=["engine_id", "cycle"])
    y_val = y[val_idx]
    
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
    
    print(f"{fd} | window={window}")
    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation RMSE: {rmse:.2f}")
    
    Path("models").mkdir(exist_ok=True)
    out_path = f"models/rf_{fd}_w{window}.joblib"
    joblib.dump(model, out_path)
    print(f"Saved model to {out_path}")
    
if __name__ == "__main__":
    main()
    