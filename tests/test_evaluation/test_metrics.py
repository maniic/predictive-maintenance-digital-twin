"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluation.metrics import rmse, mae, cmapss_score


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
