"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import cmapss_score, mae, rmse


class TestRMSE:
    def test_perfect_predictions(self):
        y = np.array([10.0, 20.0, 30.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_pred = np.array([12.0, 18.0])
        y_true = np.array([10.0, 20.0])
        # errors: [2, -2], squared: [4, 4], mean: 4, sqrt: 2.0
        assert rmse(y_pred, y_true) == pytest.approx(2.0)

    def test_single_sample(self):
        assert rmse(np.array([15.0]), np.array([10.0])) == pytest.approx(5.0)


class TestMAE:
    def test_perfect_predictions(self):
        y = np.array([10.0, 20.0, 30.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_pred = np.array([12.0, 17.0])
        y_true = np.array([10.0, 20.0])
        # abs errors: [2, 3], mean: 2.5
        assert mae(y_pred, y_true) == pytest.approx(2.5)

    def test_symmetric(self):
        y_pred = np.array([15.0])
        y_true = np.array([10.0])
        # MAE(pred, true) == MAE(true, pred) for absolute error
        assert mae(y_pred, y_true) == pytest.approx(mae(y_true, y_pred))


class TestCMAPSSScore:
    def test_perfect_predictions_score_zero(self):
        y = np.array([50.0, 100.0, 25.0])
        assert cmapss_score(y, y) == pytest.approx(0.0)

    def test_late_penalized_more_than_early(self):
        y_true = np.array([50.0])
        # Late prediction: predicted 60, true 50 → d = +10
        late_score = cmapss_score(np.array([60.0]), y_true)
        # Early prediction: predicted 40, true 50 → d = -10
        early_score = cmapss_score(np.array([40.0]), y_true)
        # Late predictions should have higher (worse) score
        assert late_score > early_score

    def test_early_prediction_formula(self):
        # d = pred - true = 40 - 50 = -10
        # score = exp(-(-10)/13) - 1 = exp(10/13) - 1
        y_pred = np.array([40.0])
        y_true = np.array([50.0])
        expected = np.exp(10 / 13) - 1
        assert cmapss_score(y_pred, y_true) == pytest.approx(expected)

    def test_late_prediction_formula(self):
        # d = pred - true = 60 - 50 = 10
        # score = exp(10/10) - 1 = e - 1
        y_pred = np.array([60.0])
        y_true = np.array([50.0])
        expected = np.exp(1.0) - 1
        assert cmapss_score(y_pred, y_true) == pytest.approx(expected)

    def test_multiple_samples_sum(self):
        y_pred = np.array([40.0, 60.0])
        y_true = np.array([50.0, 50.0])
        # Should be sum of individual scores
        s1 = cmapss_score(np.array([40.0]), np.array([50.0]))
        s2 = cmapss_score(np.array([60.0]), np.array([50.0]))
        assert cmapss_score(y_pred, y_true) == pytest.approx(s1 + s2)


class TestCMAPSSArgumentOrder:
    """Regression guard for a real bug in this repo's history.

    Two training scripts called `cmapss_score(targets, preds)`, reversing the
    documented `(y_pred, y_true)` order. Because the metric is deliberately
    asymmetric, that silently inverts its meaning: over-optimistic predictions
    came out looking cheap. RMSE and MAE are symmetric and hid the mistake.
    """

    def test_swapping_the_arguments_changes_the_score(self):
        y_pred = np.array([60.0, 45.0, 80.0])
        y_true = np.array([50.0, 50.0, 50.0])
        assert cmapss_score(y_pred, y_true) != pytest.approx(cmapss_score(y_true, y_pred))

    def test_first_argument_is_the_prediction(self):
        # Predicting 70 when the truth is 50 is a late (expensive) call.
        late = cmapss_score(np.array([70.0]), np.array([50.0]))
        # Reversing the arguments describes an early call and must score lower.
        reversed_order = cmapss_score(np.array([50.0]), np.array([70.0]))
        assert late > reversed_order

    def test_evaluate_predictions_uses_the_documented_order(self):
        from src.evaluation.metrics import evaluate_predictions

        y_pred = np.array([70.0, 30.0])
        y_true = np.array([50.0, 50.0])
        assert evaluate_predictions(y_pred, y_true).cmapss_score == pytest.approx(
            cmapss_score(y_pred, y_true)
        )


class TestNormalizedScore:
    def test_is_the_mean_of_the_summed_score(self):
        from src.evaluation.metrics import cmapss_score_normalized

        y_pred = np.array([40.0, 60.0, 55.0])
        y_true = np.array([50.0, 50.0, 50.0])
        assert cmapss_score_normalized(y_pred, y_true) == pytest.approx(
            cmapss_score(y_pred, y_true) / 3
        )


class TestTensorInputs:
    def test_metrics_accept_torch_tensors(self):
        import torch

        y_pred = torch.tensor([12.0, 18.0])
        y_true = torch.tensor([10.0, 20.0])
        assert rmse(y_pred, y_true) == pytest.approx(2.0)
        assert mae(y_pred, y_true) == pytest.approx(2.0)

    def test_tensor_and_array_agree(self):
        import torch

        pred = np.array([40.0, 60.0])
        true = np.array([50.0, 50.0])
        assert cmapss_score(torch.tensor(pred), torch.tensor(true)) == pytest.approx(
            cmapss_score(pred, true)
        )


class TestPerEngineMetrics:
    def test_splits_results_by_engine(self):
        from src.evaluation.metrics import compute_per_engine_metrics

        y_pred = np.array([10.0, 12.0, 30.0, 33.0])
        y_true = np.array([10.0, 10.0, 30.0, 30.0])
        engines = np.array([1, 1, 2, 2])
        per_engine = compute_per_engine_metrics(y_pred, y_true, engines)

        assert set(per_engine) == {1, 2}
        assert per_engine[1].n_samples == 2
        assert per_engine[2].mae == pytest.approx(1.5)
