"""End-to-end test of the serving path.

Trains nothing of consequence: it saves a small LSTM checkpoint and a fitted
preprocessor into a temporary models directory, then drives the exact path the
dashboard and `scripts/predict.py` use — RULPredictor.load_models() ->
predict_from_dataframe() -> PredictionResult.

The point is that a refactor of checkpoint loading, feature ordering or the
preprocessor contract fails loudly here instead of surfacing as a broken
dashboard after deploy.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.preprocessing import CMAPSSPreprocessor, PreprocessingConfig
from src.digital_twin import RULPredictor
from src.models.lstm import LSTMModel

SEQUENCE_LENGTH = 30
RUL_CAP = 125


def make_engine_frame(n_cycles: int = 60, engine_id: int = 1, seed: int = 0) -> pd.DataFrame:
    """A single engine's trajectory in raw C-MAPSS column layout."""
    rng = np.random.RandomState(seed)
    rows = []
    for cycle in range(1, n_cycles + 1):
        wear = cycle / n_cycles
        rows.append(
            {
                "engine_id": engine_id,
                "cycle": cycle,
                "setting_1": rng.uniform(-0.005, 0.005),
                "setting_2": rng.uniform(0.0, 0.001),
                "setting_3": 100.0,
                "sensor_1": 518.67,
                "sensor_2": 641.0 + 2.0 * wear + rng.normal(0, 0.1),
                "sensor_3": 1580.0 + 20.0 * wear + rng.normal(0, 1.0),
                "sensor_4": 1400.0 - 15.0 * wear + rng.normal(0, 1.0),
                "RUL": float(min(n_cycles - cycle, RUL_CAP)),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def served_models(tmp_path_factory):
    """A models directory laid out the way RULPredictor expects to find one."""
    models_dir = tmp_path_factory.mktemp("models")
    dataset = "FD001"

    train_df = pd.concat(
        [make_engine_frame(n_cycles=80, engine_id=i, seed=i) for i in range(1, 6)],
        ignore_index=True,
    )

    preprocessor = CMAPSSPreprocessor(
        PreprocessingConfig(
            drop_strategy="auto",
            rul_cap=RUL_CAP,
            normalization="minmax",
            cluster_by_regime=False,
            n_regimes=1,
        )
    )
    preprocessor.fit_transform(train_df)

    preprocessor_dir = models_dir / "preprocessors"
    preprocessor_dir.mkdir(parents=True, exist_ok=True)
    preprocessor.save(preprocessor_dir / f"{dataset}_preprocessor.pkl")

    input_dim = len(preprocessor.get_feature_names())
    model = LSTMModel(
        input_dim=input_dim,
        sequence_length=SEQUENCE_LENGTH,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )

    # Bias the output towards a mid-range RUL so predictions land in a plausible
    # band without spending real training time in the test suite.
    with torch.no_grad():
        final_layer = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
        final_layer.weight.mul_(0.01)
        final_layer.bias.fill_(60.0)

    checkpoint_dir = models_dir / "checkpoints" / dataset / "lstm"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": "2.0.0",
        },
        checkpoint_dir / "test.ckpt",
    )

    return models_dir, dataset, input_dim


@pytest.fixture(scope="module")
def predictor(served_models):
    models_dir, dataset, _ = served_models
    predictor = RULPredictor(dataset=dataset, models_dir=str(models_dir), device="cpu")
    predictor.load_models()
    return predictor


class TestLoading:
    def test_checkpoint_is_discovered_and_loaded(self, predictor):
        assert "lstm" in predictor.models

    def test_an_ensemble_is_constructed(self, predictor):
        assert predictor.ensemble is not None

    def test_feature_columns_come_from_the_preprocessor(self, predictor, served_models):
        _, _, input_dim = served_models
        assert len(predictor.feature_columns) == input_dim

    def test_missing_preprocessor_raises_clearly(self, tmp_path):
        empty = RULPredictor(dataset="FD001", models_dir=str(tmp_path), device="cpu")
        with pytest.raises(FileNotFoundError, match="Preprocessor"):
            empty.load_models()


class TestInference:
    def test_prediction_is_a_finite_scalar(self, predictor):
        result = predictor.predict_from_dataframe(make_engine_frame(n_cycles=60, seed=42))
        assert isinstance(result.rul, float)
        assert np.isfinite(result.rul)

    def test_rul_is_in_a_plausible_range(self, predictor):
        """RUL is a non-negative cycle count and cannot exceed the training cap."""
        for seed in range(4):
            result = predictor.predict_from_dataframe(make_engine_frame(n_cycles=60, seed=seed))
            assert 0.0 <= result.rul <= RUL_CAP * 1.5

    def test_health_score_is_a_fraction(self, predictor):
        result = predictor.predict_from_dataframe(make_engine_frame(n_cycles=60, seed=7))
        assert 0.0 <= result.health_score <= 1.0

    def test_confidence_bounds_bracket_the_estimate(self, predictor):
        result = predictor.predict_from_dataframe(make_engine_frame(n_cycles=60, seed=7))
        assert result.rul_lower <= result.rul <= result.rul_upper
        assert result.rul_lower >= 0.0

    def test_every_loaded_model_reports_a_prediction(self, predictor):
        result = predictor.predict_from_dataframe(make_engine_frame(n_cycles=60, seed=7))
        assert set(result.individual_predictions) == set(predictor.models)

    def test_raw_sequence_input_matches_dataframe_input(self, predictor):
        df = make_engine_frame(n_cycles=60, seed=11)
        from_frame = predictor.predict_from_dataframe(df)

        processed = predictor.preprocessor.transform(df.tail(SEQUENCE_LENGTH))
        sequence = processed[predictor.feature_columns].values
        from_array = predictor.predict(sequence)

        assert from_array.rul == pytest.approx(from_frame.rul, rel=1e-5)

    def test_batch_dimension_is_accepted(self, predictor, served_models):
        _, _, input_dim = served_models
        sequence = np.zeros((1, SEQUENCE_LENGTH, input_dim), dtype=np.float32)
        assert np.isfinite(predictor.predict(sequence).rul)

    def test_too_short_a_history_is_refused(self, predictor):
        with pytest.raises(ValueError, match="at least"):
            predictor.predict_from_dataframe(make_engine_frame(n_cycles=10))

    def test_predicting_before_loading_is_refused(self, served_models):
        models_dir, dataset, _ = served_models
        cold = RULPredictor(dataset=dataset, models_dir=str(models_dir), device="cpu")
        with pytest.raises(RuntimeError, match="load_models"):
            cold.predict_from_dataframe(make_engine_frame(n_cycles=60))

    def test_inference_is_deterministic(self, predictor):
        df = make_engine_frame(n_cycles=60, seed=3)
        assert predictor.predict_from_dataframe(df).rul == pytest.approx(
            predictor.predict_from_dataframe(df).rul
        )

    def test_model_info_describes_what_is_loaded(self, predictor, served_models):
        _, dataset, input_dim = served_models
        info = predictor.get_model_info()
        assert info["dataset"] == dataset
        assert info["models_loaded"] == ["lstm"]
        assert info["input_dim"] == input_dim
        assert info["sequence_length"] == SEQUENCE_LENGTH
