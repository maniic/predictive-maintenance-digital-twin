"""Evaluation metrics and explainability for RUL prediction."""

from src.evaluation.metrics import (
    rmse,
    mae,
    cmapss_score,
    cmapss_score_normalized,
    evaluate_predictions,
    compute_per_engine_metrics,
    EvaluationResults,
)
from src.evaluation.explainability import (
    RULExplainer,
    FeatureImportance,
    AttentionAnalysis,
    SensorContribution,
    create_shap_summary_plot,
)

__all__ = [
    # Metrics
    "rmse",
    "mae",
    "cmapss_score",
    "cmapss_score_normalized",
    "evaluate_predictions",
    "compute_per_engine_metrics",
    "EvaluationResults",
    # Explainability
    "RULExplainer",
    "FeatureImportance",
    "AttentionAnalysis",
    "SensorContribution",
    "create_shap_summary_plot",
]

