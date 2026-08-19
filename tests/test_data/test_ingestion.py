"""Tests for data ingestion and RUL computation."""

import pandas as pd
import pytest

from src.data.ingestion import compute_test_rul, compute_train_rul


def make_train_df(engines):
    """Create a synthetic training DataFrame.

    Args:
        engines: dict mapping engine_id -> number of cycles
    """
    rows = []
    for eid, n_cycles in engines.items():
        for c in range(1, n_cycles + 1):
            rows.append({"engine_id": eid, "cycle": c, "sensor_1": 500.0})
    return pd.DataFrame(rows)


def make_test_df(engines):
    """Create a synthetic test DataFrame.

    Args:
        engines: dict mapping engine_id -> number of observed cycles
    """
    rows = []
    for eid, n_cycles in engines.items():
        for c in range(1, n_cycles + 1):
            rows.append({"engine_id": eid, "cycle": c, "sensor_1": 500.0})
    return pd.DataFrame(rows)


class TestComputeTrainRUL:
    def test_rul_at_last_cycle_is_zero(self):
        df = make_train_df({1: 10})
        result = compute_train_rul(df)
        last_row = result[result["cycle"] == 10].iloc[0]
        assert last_row["RUL"] == 0

    def test_rul_at_first_cycle(self):
        df = make_train_df({1: 10})
        result = compute_train_rul(df)
        first_row = result[result["cycle"] == 1].iloc[0]
        assert first_row["RUL"] == 9  # max_cycle(10) - cycle(1) = 9

    def test_rul_decreases_monotonically(self):
        df = make_train_df({1: 50})
        result = compute_train_rul(df)
        ruls = result.sort_values("cycle")["RUL"].values
        assert all(ruls[i] > ruls[i + 1] for i in range(len(ruls) - 1))

    def test_multiple_engines_independent(self):
        df = make_train_df({1: 10, 2: 20})
        result = compute_train_rul(df)

        # Engine 1: last cycle RUL should be 0
        e1_last = result[(result["engine_id"] == 1) & (result["cycle"] == 10)]
        assert e1_last.iloc[0]["RUL"] == 0

        # Engine 2: first cycle RUL should be 19
        e2_first = result[(result["engine_id"] == 2) & (result["cycle"] == 1)]
        assert e2_first.iloc[0]["RUL"] == 19

    def test_does_not_modify_original(self):
        df = make_train_df({1: 5})
        assert "RUL" not in df.columns
        compute_train_rul(df)
        assert "RUL" not in df.columns


class TestComputeTestRUL:
    def test_rul_at_last_cycle_equals_ground_truth(self):
        df = make_test_df({1: 10})
        rul_values = pd.Series([25])
        result = compute_test_rul(df, rul_values)
        last_row = result[result["cycle"] == 10].iloc[0]
        assert last_row["RUL"] == 25

    def test_rul_at_earlier_cycles(self):
        df = make_test_df({1: 10})
        rul_values = pd.Series([25])
        result = compute_test_rul(df, rul_values)
        # At cycle 1: (10 - 1) + 25 = 34
        first_row = result[result["cycle"] == 1].iloc[0]
        assert first_row["RUL"] == 34

    def test_multiple_engines(self):
        df = make_test_df({1: 5, 2: 8})
        rul_values = pd.Series([10, 20])
        result = compute_test_rul(df, rul_values)

        # Engine 1 last cycle: RUL = 10
        e1_last = result[(result["engine_id"] == 1) & (result["cycle"] == 5)]
        assert e1_last.iloc[0]["RUL"] == 10

        # Engine 2 last cycle: RUL = 20
        e2_last = result[(result["engine_id"] == 2) & (result["cycle"] == 8)]
        assert e2_last.iloc[0]["RUL"] == 20

    def test_mismatched_engines_raises(self):
        df = make_test_df({1: 5, 2: 8})
        rul_values = pd.Series([10])  # Only 1 value for 2 engines
        with pytest.raises(ValueError, match="doesn't match"):
            compute_test_rul(df, rul_values)

    def test_does_not_modify_original(self):
        df = make_test_df({1: 5})
        rul_values = pd.Series([10])
        assert "RUL" not in df.columns
        compute_test_rul(df, rul_values)
        assert "RUL" not in df.columns
