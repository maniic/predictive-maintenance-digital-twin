"""Tests for sliding-window construction and the train/validation split.

These two pieces decide what every model actually sees. A one-cycle shift
between a window and its target, or a split that lets the same engine appear
on both sides, would quietly invalidate every metric this project reports and
nothing else in the codebase would notice.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import (
    CMAPSSInferenceDataset,
    CMAPSSSequenceDataset,
    train_val_split,
)

FEATURES = ["sensor_1", "sensor_2"]


def make_df(engines: dict[int, int], rul_from_end: bool = True) -> pd.DataFrame:
    """Build a frame where every value is derivable from (engine, cycle).

    sensor_1 encodes the cycle and sensor_2 encodes the engine, so a misaligned
    window is visible in the data itself rather than only in a summary statistic.
    """
    rows = []
    for engine_id, n_cycles in engines.items():
        for cycle in range(1, n_cycles + 1):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "sensor_1": float(cycle),
                    "sensor_2": float(engine_id),
                    "RUL": float(n_cycles - cycle) if rul_from_end else 0.0,
                }
            )
    return pd.DataFrame(rows)


class TestWindowShape:
    def test_window_count_is_n_minus_length_plus_one(self):
        ds = CMAPSSSequenceDataset(make_df({1: 40}), FEATURES, sequence_length=30, stride=1)
        assert len(ds) == 40 - 30 + 1

    def test_sequence_shape(self):
        ds = CMAPSSSequenceDataset(make_df({1: 40}), FEATURES, sequence_length=30)
        x, y = ds[0]
        assert x.shape == (30, len(FEATURES))
        assert y.ndim == 0

    def test_stride_reduces_window_count(self):
        df = make_df({1: 100})
        dense = CMAPSSSequenceDataset(df, FEATURES, sequence_length=30, stride=1)
        sparse = CMAPSSSequenceDataset(df, FEATURES, sequence_length=30, stride=5)
        assert len(sparse) == len(range(0, 100 - 30 + 1, 5))
        assert len(sparse) < len(dense)

    def test_n_features_property(self):
        ds = CMAPSSSequenceDataset(make_df({1: 40}), FEATURES, sequence_length=30)
        assert ds.n_features == len(FEATURES)


class TestWindowAlignment:
    """The window must end on the cycle whose RUL is used as the target."""

    def test_first_window_covers_cycles_1_to_seq_len(self):
        ds = CMAPSSSequenceDataset(make_df({1: 40}), FEATURES, sequence_length=30)
        x, _ = ds[0]
        # sensor_1 encodes the cycle number
        assert x[:, 0].tolist() == [float(c) for c in range(1, 31)]

    def test_target_is_rul_at_last_cycle_of_window(self):
        # 40 cycles, so RUL at cycle 30 is 10 and at cycle 40 is 0.
        ds = CMAPSSSequenceDataset(make_df({1: 40}), FEATURES, sequence_length=30)
        _, first = ds[0]
        _, last = ds[len(ds) - 1]
        assert float(first) == pytest.approx(10.0)
        assert float(last) == pytest.approx(0.0)

    def test_no_off_by_one_across_every_window(self):
        seq_len, n_cycles = 5, 20
        ds = CMAPSSSequenceDataset(make_df({1: n_cycles}), FEATURES, sequence_length=seq_len)
        for i in range(len(ds)):
            x, y = ds[i]
            end_cycle = i + seq_len
            assert x[-1, 0].item() == pytest.approx(float(end_cycle))
            assert x[0, 0].item() == pytest.approx(float(end_cycle - seq_len + 1))
            assert float(y) == pytest.approx(float(n_cycles - end_cycle))

    def test_unsorted_input_is_sorted_before_windowing(self):
        df = make_df({1: 40}).sample(frac=1.0, random_state=0)
        ds = CMAPSSSequenceDataset(df, FEATURES, sequence_length=30)
        x, _ = ds[0]
        assert x[:, 0].tolist() == [float(c) for c in range(1, 31)]


class TestEngineBoundaries:
    def test_windows_never_span_two_engines(self):
        ds = CMAPSSSequenceDataset(make_df({1: 35, 2: 35}), FEATURES, sequence_length=30)
        for i in range(len(ds)):
            x, _ = ds[i]
            # sensor_2 encodes the engine id and must be constant within a window
            assert len(set(x[:, 1].tolist())) == 1

    def test_window_count_is_the_sum_over_engines(self):
        ds = CMAPSSSequenceDataset(make_df({1: 35, 2: 40}), FEATURES, sequence_length=30)
        assert len(ds) == (35 - 30 + 1) + (40 - 30 + 1)

    def test_engine_shorter_than_window_is_excluded(self):
        # This is why 6 FD002 and 11 FD004 test engines are not evaluated.
        ds = CMAPSSSequenceDataset(make_df({1: 29}), FEATURES, sequence_length=30)
        assert len(ds) == 0

    def test_engine_exactly_window_length_yields_one_window(self):
        ds = CMAPSSSequenceDataset(make_df({1: 30}), FEATURES, sequence_length=30)
        assert len(ds) == 1

    def test_short_engine_does_not_remove_the_others(self):
        ds = CMAPSSSequenceDataset(make_df({1: 10, 2: 35}), FEATURES, sequence_length=30)
        assert len(ds) == 35 - 30 + 1


class TestInferenceDataset:
    def test_one_window_per_valid_endpoint(self):
        ds = CMAPSSInferenceDataset(make_df({1: 40}), FEATURES, sequence_length=30)
        assert len(ds) == 40 - 30 + 1

    def test_metadata_tracks_cycle_and_last_flag(self):
        ds = CMAPSSInferenceDataset(make_df({1: 40}), FEATURES, sequence_length=30)
        meta = ds.get_metadata()
        assert meta["cycle"].tolist() == list(range(30, 41))
        assert meta["is_last"].sum() == 1
        assert meta[meta["is_last"]]["cycle"].iloc[0] == 40

    def test_window_ends_on_its_metadata_cycle(self):
        ds = CMAPSSInferenceDataset(make_df({1: 40}), FEATURES, sequence_length=30)
        meta = ds.get_metadata()
        for i in range(len(ds)):
            assert ds[i][-1, 0].item() == pytest.approx(float(meta["cycle"].iloc[i]))


class TestTrainValSplit:
    def test_no_engine_appears_on_both_sides(self):
        df = make_df({i: 40 for i in range(1, 21)})
        train, val = train_val_split(df, val_ratio=0.2, seed=42)
        assert set(train["engine_id"]) & set(val["engine_id"]) == set()

    def test_every_engine_is_kept(self):
        df = make_df({i: 40 for i in range(1, 21)})
        train, val = train_val_split(df, val_ratio=0.2, seed=42)
        assert set(train["engine_id"]) | set(val["engine_id"]) == set(range(1, 21))
        assert len(train) + len(val) == len(df)

    def test_split_size_follows_the_ratio(self):
        df = make_df({i: 40 for i in range(1, 21)})
        _, val = train_val_split(df, val_ratio=0.25, seed=42)
        assert val["engine_id"].nunique() == 5

    def test_rows_of_an_engine_are_never_divided(self):
        df = make_df({i: 40 for i in range(1, 21)})
        train, val = train_val_split(df, val_ratio=0.2, seed=1)
        for part in (train, val):
            counts = part.groupby("engine_id").size()
            assert (counts == 40).all()

    def test_same_seed_gives_the_same_split(self):
        df = make_df({i: 40 for i in range(1, 21)})
        _, a = train_val_split(df, val_ratio=0.2, seed=7)
        _, b = train_val_split(df, val_ratio=0.2, seed=7)
        assert sorted(a["engine_id"].unique()) == sorted(b["engine_id"].unique())

    def test_different_seeds_generally_differ(self):
        df = make_df({i: 40 for i in range(1, 51)})
        _, a = train_val_split(df, val_ratio=0.2, seed=1)
        _, b = train_val_split(df, val_ratio=0.2, seed=2)
        assert set(a["engine_id"]) != set(b["engine_id"])

    def test_windows_built_from_the_split_share_no_engine(self):
        """The leakage that matters is at the window level, not the row level."""
        df = make_df({i: 40 for i in range(1, 21)})
        train, val = train_val_split(df, val_ratio=0.2, seed=42)
        train_ds = CMAPSSSequenceDataset(train, FEATURES, sequence_length=30)
        val_ds = CMAPSSSequenceDataset(val, FEATURES, sequence_length=30)
        # sensor_2 carries the engine id through the window
        train_engines = {float(train_ds[i][0][0, 1]) for i in range(len(train_ds))}
        val_engines = {float(val_ds[i][0][0, 1]) for i in range(len(val_ds))}
        assert train_engines & val_engines == set()

    def test_does_not_mutate_the_input(self):
        df = make_df({i: 40 for i in range(1, 21)})
        before = df.copy()
        train_val_split(df, val_ratio=0.2, seed=42)
        pd.testing.assert_frame_equal(df, before)


class TestSampleWeights:
    def test_weights_are_normalised_to_the_sample_count(self):
        ds = CMAPSSSequenceDataset(make_df({1: 100}), FEATURES, sequence_length=30)
        weights = ds.get_sample_weights("linear")
        assert len(weights) == len(ds)
        assert float(weights.sum()) == pytest.approx(len(ds), rel=1e-5)

    def test_low_rul_windows_are_weighted_higher(self):
        ds = CMAPSSSequenceDataset(make_df({1: 100}), FEATURES, sequence_length=30)
        weights = ds.get_sample_weights("linear").numpy()
        targets = ds.targets
        # windows are ordered by increasing end-cycle, so RUL decreases
        assert targets[0] > targets[-1]
        assert weights[-1] > weights[0]

    def test_unknown_method_falls_back_to_uniform(self):
        ds = CMAPSSSequenceDataset(make_df({1: 100}), FEATURES, sequence_length=30)
        weights = ds.get_sample_weights("nonsense").numpy()
        assert np.allclose(weights, weights[0])
