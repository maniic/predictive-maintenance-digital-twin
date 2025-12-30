from src.models.train_rf import train

FD_SETS = ["FD001", "FD002", "FD003", "FD004"]

def main():
    window = 30
    for fd in FD_SETS:
        print(f"\n=== TRAIN {fd} (window={window}) ===")
        train(fd=fd, window=window)
        
if __name__ == "__main__":
    main()