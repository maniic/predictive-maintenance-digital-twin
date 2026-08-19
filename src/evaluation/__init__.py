"""Evaluation metrics and explainability for RUL prediction.

Metrics import cleanly with only the core dependencies. The explainability
helpers need SHAP and matplotlib (`pip install -e ".[explain]"`), so they are
resolved lazily — importing this package, or `src.evaluation.metrics`, never
requires them.
"""

from src.evaluation.metrics import (
    EvaluationResults,
    cmapss_score,
    cmapss_score_normalized,
    compute_per_engine_metrics,
    evaluate_predictions,
    mae,
    rmse,
)

_EXPLAIN_EXPORTS = {
    "RULExplainer",
    "FeatureImportance",
    "AttentionAnalysis",
    "SensorContribution",
    "create_shap_summary_plot",
}

__all__ = [
    # Metrics
    "rmse",
    "mae",
    "cmapss_score",
    "cmapss_score_normalized",
    "evaluate_predictions",
    "compute_per_engine_metrics",
    "EvaluationResults",
    # Explainability (lazy — see __getattr__)
    *sorted(_EXPLAIN_EXPORTS),
]


def __getattr__(name: str):
    """Resolve explainability exports on first access.

    Keeps SHAP and matplotlib out of the import path for anyone who only wants
    metrics, while `from src.evaluation import RULExplainer` still works.
    """
    if name in _EXPLAIN_EXPORTS:
        try:
            from src.evaluation import explainability
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                f"{name} needs the optional explainability dependencies. "
                'Install them with: pip install -e ".[explain]"'
            ) from exc
        return getattr(explainability, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
