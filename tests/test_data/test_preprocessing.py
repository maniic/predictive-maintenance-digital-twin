"""Tests for RUL capping and the fit/transform boundary.

The piecewise RUL cap defines the target every model is trained against, and
the scalers are the one place where test data could leak into training. Both
are load-bearing for the reported metrics and neither was covered.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    CMAPSSPreprocessor,
    PreprocessingConfig,
    analyze_sensors,
    find_zero_variance_sensors,
)


def make_df(n_engines: int = 4, n_cycles: int = 50, seed: int = 0) -> pd.DataFrame:
    """A C-MAPSS-shaped frame: three settings, a few sensors, a RUL column."""
    rng = np.random.RandomState(seed)
    rows = []
    for engine_id in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            rows.append(
                {
                    "engine_id": engine_id,
                    "cycle": cycle,
                    "setting_1": rng.uniform(-0.005, 0.005),
                    "setting_2": rng.uniform(0, 0.001),
                    "setting_3": 100.0,
                    "sensor_1": 518.67,  # zero variance, as in the real data
                    "sensor_2": 640.0 + cycle * 0.05 + rng.normal(0, 0.1),
                    "sensor_3": 1580.0 - cycle * 0.4 + rng.normal(0, 1.0),
                    "RUL": float(n_cycles - cycle),
                }
            )
    return pd.DataFrame(rows)


class TestRULCap:
    def test_rul_is_clipped_at_the_cap(self):
        config = PreprocessingConfig(rul_cap=125, cluster_by_regime=False, n_regimes=1)
        df = make_df(n_cycles=300)
        out = CMAPSSPreprocessor(config).fit_transform(df)
        assert out["RUL"].max() == 125

    def test_values_below_the_cap_are_untouched(self):
        config = PreprocessingConfig(rul_cap=125, cluster_by_regime=False, n_regimes=1)
        df = make_df(n_engines=1, n_cycles=50)
        out = CMAPSSPreprocessor(config).fit_transform(df)
        # 50 cycles, so every RUL is below 125 and must survive intact
        assert sorted(out["RUL"].tolist()) == sorted(float(v) for v in range(50))

    def test_cap_is_only_an_upper_bound(self):
        config = PreprocessingConfig(rul_cap=30, cluster_by_regime=False, n_regimes=1)
        out = CMAPSSPreprocessor(config).fit_transform(make_df(n_cycles=100))
        assert out["RUL"].min() == 0
        assert out["RUL"].max() == 30

    def test_a_different_cap_is_respected(self):
        config = PreprocessingConfig(rul_cap=80, cluster_by_regime=False, n_regimes=1)
        out = CMAPSSPreprocessor(config).fit_transform(make_df(n_cycles=200))
        assert out["RUL"].max() == 80

    def test_test_data_is_capped_the_same_way(self):
        """Train and test must share a target definition or the metrics mean nothing."""
        config = PreprocessingConfig(rul_cap=125, cluster_by_regime=False, n_regimes=1)
        pre = CMAPSSPreprocessor(config)
        pre.fit_transform(make_df(n_cycles=200))
        out = pre.transform(make_df(n_cycles=300, seed=1))
        assert out["RUL"].max() == 125

    def test_frames_without_a_rul_column_pass_through(self):
        config = PreprocessingConfig(rul_cap=125, cluster_by_regime=False, n_regimes=1)
        pre = CMAPSSPreprocessor(config)
        pre.fit_transform(make_df())
        out = pre.transform(make_df().drop(columns=["RUL"]))
        assert "RUL" not in out.columns


class TestNoLeakage:
    def test_transform_reuses_the_fitted_scaler(self):
        """Test data must be scaled by train statistics, not its own."""
        config = PreprocessingConfig(
            normalization="minmax", cluster_by_regime=False, n_regimes=1, drop_strategy="none"
        )
        pre = CMAPSSPreprocessor(config)
        pre.fit_transform(make_df(n_cycles=50, seed=0))
        fitted_min = pre.scalers_[0].data_min_.copy()
        fitted_max = pre.scalers_[0].data_max_.copy()

        # Transform something with a very different range
        shifted = make_df(n_cycles=50, seed=1)
        shifted["sensor_2"] += 1000.0
        pre.transform(shifted)

        assert np.allclose(pre.scalers_[0].data_min_, fitted_min)
        assert np.allclose(pre.scalers_[0].data_max_, fitted_max)

    def test_out_of_range_test_values_escape_zero_one(self):
        """Proof the scaler is not refitting: min-max output leaves [0, 1]."""
        config = PreprocessingConfig(
            normalization="minmax", cluster_by_regime=False, n_regimes=1, drop_strategy="none"
        )
        pre = CMAPSSPreprocessor(config)
        pre.fit_transform(make_df(n_cycles=50, seed=0))

        shifted = make_df(n_cycles=50, seed=1)
        shifted["sensor_2"] += 500.0
        out = pre.transform(shifted)
        assert out["sensor_2"].max() > 1.0

    def test_training_output_is_inside_zero_one(self):
        config = PreprocessingConfig(
            normalization="minmax", cluster_by_regime=False, n_regimes=1, drop_strategy="none"
        )
        out = CMAPSSPreprocessor(config).fit_transform(make_df())
        for col in ("sensor_2", "sensor_3"):
            assert out[col].min() >= -1e-9
            assert out[col].max() <= 1 + 1e-9

    def test_regime_clusters_are_fitted_once(self):
        config = PreprocessingConfig(cluster_by_regime=True, n_regimes=2, drop_strategy="none")
        pre = CMAPSSPreprocessor(config)
        pre.fit_transform(make_df(n_cycles=60))
        centers = pre.regime_model_.cluster_centers_.copy()
        pre.transform(make_df(n_cycles=60, seed=99))
        assert np.allclose(pre.regime_model_.cluster_centers_, centers)

    def test_transform_before_fit_is_refused(self):
        pre = CMAPSSPreprocessor(PreprocessingConfig(cluster_by_regime=False, n_regimes=1))
        with pytest.raises(RuntimeError, match="fitted"):
            pre.transform(make_df())

    def test_fit_does_not_mutate_the_input(self):
        df = make_df()
        before = df.copy()
        CMAPSSPreprocessor(PreprocessingConfig(cluster_by_regime=False, n_regimes=1)).fit_transform(
            df
        )
        pd.testing.assert_frame_equal(df, before)


class TestSensorSelection:
    def test_zero_variance_sensor_is_detected(self):
        assert "sensor_1" in find_zero_variance_sensors(make_df())

    def test_varying_sensors_are_kept(self):
        dropped = find_zero_variance_sensors(make_df())
        assert "sensor_2" not in dropped and "sensor_3" not in dropped

    def test_auto_strategy_drops_only_zero_variance(self):
        pre = CMAPSSPreprocessor(
            PreprocessingConfig(drop_strategy="auto", cluster_by_regime=False, n_regimes=1)
        )
        out = pre.fit_transform(make_df())
        assert pre.get_dropped_sensors() == ["sensor_1"]
        assert "sensor_1" not in out.columns

    def test_none_strategy_keeps_everything(self):
        pre = CMAPSSPreprocessor(
            PreprocessingConfig(drop_strategy="none", cluster_by_regime=False, n_regimes=1)
        )
        out = pre.fit_transform(make_df())
        assert pre.get_dropped_sensors() == []
        assert "sensor_1" in out.columns

    def test_manual_strategy_uses_the_given_list(self):
        pre = CMAPSSPreprocessor(
            PreprocessingConfig(
                drop_strategy="manual",
                drop_sensors=["sensor_3"],
                cluster_by_regime=False,
                n_regimes=1,
            )
        )
        out = pre.fit_transform(make_df())
        assert "sensor_3" not in out.columns

    def test_analysis_covers_every_sensor(self):
        analyses = analyze_sensors(make_df())
        assert {a.name for a in analyses} == {"sensor_1", "sensor_2", "sensor_3"}


class TestFeatureNames:
    def test_regime_is_a_feature_when_clustering_is_on(self):
        pre = CMAPSSPreprocessor(PreprocessingConfig(cluster_by_regime=True, n_regimes=2))
        pre.fit_transform(make_df())
        assert "regime" in pre.get_feature_names()

    def test_regime_is_absent_for_single_regime_datasets(self):
        pre = CMAPSSPreprocessor(PreprocessingConfig(cluster_by_regime=True, n_regimes=1))
        pre.fit_transform(make_df())
        assert "regime" not in pre.get_feature_names()

    def test_feature_names_match_the_transformed_columns(self):
        pre = CMAPSSPreprocessor(PreprocessingConfig(cluster_by_regime=True, n_regimes=2))
        out = pre.fit_transform(make_df())
        for name in pre.get_feature_names():
            assert name in out.columns

    def test_train_and_test_agree_on_features(self):
        pre = CMAPSSPreprocessor(PreprocessingConfig(cluster_by_regime=True, n_regimes=2))
        pre.fit_transform(make_df())
        out = pre.transform(make_df(seed=5))
        assert list(out[pre.get_feature_names()].columns) == pre.get_feature_names()


class TestRoundTrip:
    def test_saved_preprocessor_reproduces_its_output(self, tmp_path):
        pre = CMAPSSPreprocessor(PreprocessingConfig(cluster_by_regime=False, n_regimes=1))
        pre.fit_transform(make_df())
        expected = pre.transform(make_df(seed=3))

        path = tmp_path / "pre.pkl"
        pre.save(path)
        restored = CMAPSSPreprocessor.load(path)

        pd.testing.assert_frame_equal(restored.transform(make_df(seed=3)), expected)

    def test_saving_before_fit_is_refused(self, tmp_path):
        pre = CMAPSSPreprocessor(PreprocessingConfig())
        with pytest.raises(RuntimeError):
            pre.save(tmp_path / "pre.pkl")
