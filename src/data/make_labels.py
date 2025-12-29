from pathlib import Path
import pandas as pd
from load_cmapss import load_cmapss_txt, load_cmapss_rul

FD_SETS = ["FD001", "FD002", "FD003", "FD004"]

def add_train_rul(train: pd.DataFrame) -> pd.DataFrame:
    # max cycle per engine = failure time in train data
    max_cycle = train.groupby("engine_id")["cycle"].max().rename("max_cycle")
    train = train.merge(max_cycle, on="engine_id", how="left")
    train["RUL"] = train["max_cycle"] - train["cycle"]
    return train.drop(columns=["max_cycle"])

def add_test_rul(test: pd.DataFrame, rul_test: pd.Series) -> pd.DataFrame:
    # last cycle per engine in test file (test file ends before failure)
    last_cycle = test.groupby("engine_id")["cycle"].max().rename("last_cycle")
    test = test.merge(last_cycle, on="engine_id", how="left")
    
    rul_test = pd.Series(rul_test).astype(int).reset_index(drop=True)
    
    engine_ids = sorted(test["engine_id"].unique())
    if len(engine_ids) != len(rul_test):
        raise ValueError(
            f"Mismatch: test has {len(engine_ids)} engines but RUL file has {len(rul_test)} entries"
        )
    
    mapping = dict(zip(engine_ids, rul_test))
    test["RUL_end"] = test["engine_id"].map(mapping)
    
    if test["RUL_end"].isna().any():
        missing = sorted(test.loc[test["RUL_end"].isna(), "engine_id"].unique())
        raise ValueError(f"Some test engine_ids were not mapped to RUL_end: {missing[:10]} ...")
    
    # RUL at each row = remaining cycles to the end of the file + RUL_end
    test["RUL"] = (test["last_cycle"] - test["cycle"]) + test["RUL_end"]
    return test.drop(columns=["last_cycle", "RUL_end"])

def process_fd(fd: str, raw: Path, processed: Path) -> None:
    train_path = raw / f"train_{fd}.txt"
    test_path = raw / f"test_{fd}.txt"
    rul_path = raw / f"RUL_{fd}.txt"
    
    for p in [train_path, test_path, rul_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
    
    train = load_cmapss_txt(train_path)
    test = load_cmapss_txt(test_path)
    rul_test = load_cmapss_rul(rul_path)
    
    train_labeled = add_train_rul(train)
    test_labeled = add_test_rul(test, rul_test)
    
    train_labeled.to_parquet(processed / f"train_{fd}.parquet", index=False)
    test_labeled.to_parquet(processed / f"test_{fd}.parquet", index=False)
    
    print(f"\n[{fd}] Saved processed parquet files.")
    print("Train RUL range:", train_labeled["RUL"].min(), "to", train_labeled["RUL"].max())
    print("Test RUL range:", test_labeled["RUL"].min(), "to", test_labeled["RUL"].max())

if __name__ == "__main__":
    raw = Path("data/raw")
    processed = Path("data/processed")
    processed.mkdir(parents=True, exist_ok=True)
    for fd in FD_SETS:
        print(f"Processing {fd}...")
        process_fd(fd, raw, processed)