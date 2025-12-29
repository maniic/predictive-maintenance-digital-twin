from pathlib import Path
import pandas as pd

COLS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_cmapss_txt(file_path: str | Path) -> pd.DataFrame:
    """Load CMAPSS train/test text file into a pandas DataFrame."""
    path = Path(file_path)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    
    df = df.dropna(axis=1, how="all")
    
    if df.shape[1] != len(COLS):
        raise ValueError(
            f"{file_path.name}: expected {len(COLS)} columns after cleanup, got {df.shape[1]}"
        )
        
    df.columns = COLS
    
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    
    return df

def load_cmapss_rul(file_path: str | Path) -> pd.Series:
    """Load CMAPSS RUL text file into a pandas Series."""
    path = Path(file_path)
    s = pd.read_csv(path, sep=r"\s+", header=None).dropna(axis=1, how="all").iloc[:, 0]
    return s.astype(int)

if __name__ == "__main__":
    raw = Path("data/raw")
    
    train_path = raw / "train_FD001.txt"
    test_path = raw / "test_FD001.txt"
    rul_path = raw / "RUL_FD001.txt"
    
    train = load_cmapss_txt(train_path)
    test = load_cmapss_txt(test_path)
    rul = load_cmapss_rul(rul_path)
    
    print("Train :", train.shape, "engines:", train["engine_id"].nunique())
    print("Test :", test.shape, "engines:", test["engine_id"].nunique())
    print("RUL :", rul.shape)
    
    print("\nTrain head:")
    print(train.head())