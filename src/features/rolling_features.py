import pandas as pd

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