from src.models.eval_rf import eval_test

FD_SETS = ["FD001", "FD002", "FD003", "FD004"]

def main():
    window = 30
    for fd in FD_SETS:
        print(f"\n=== EVAL {fd} (window={window}) ===")
        eval_test(fd=fd, window=window)

if __name__ == "__main__":
    main()