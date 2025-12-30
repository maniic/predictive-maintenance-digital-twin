from src.models.train_rf import train

FD_SETS = ["FD001", "FD002", "FD003", "FD004"]

def main():
    window = 30
    include_slope = True
    for fd in FD_SETS:
        print(f"\n=== TRAIN {fd} (window={window}) ===")
        train(fd=fd, window=window, include_slope=include_slope)
        
if __name__ == "__main__":
    main()