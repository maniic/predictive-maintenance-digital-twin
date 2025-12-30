from src.models.eval_rf import eval_test

FD_SETS = ["FD001", "FD002", "FD003", "FD004"]

def main():
    window = 30
    include_slope = True
    for fd in FD_SETS:
        print(f"\n=== EVAL {fd} (window={window}) ===")
        eval_test(fd=fd, window=window, include_slope=include_slope)

if __name__ == "__main__":
    main()