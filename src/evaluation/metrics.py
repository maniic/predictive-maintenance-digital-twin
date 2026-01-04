"""
Evaluation Metrics for RUL Prediction

Includes standard metrics (RMSE, MAE) and the official C-MAPSS
asymmetric scoring function from the PHM08 competition.

"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


def rmse(y_pred: np.ndarray | Tensor, y_true: np.ndarray | Tensor) -> float:
    """Compute Root Mean Square Error.
    
    Args:
        y_pred: Predicted RUL values
        y_true: True RUL values
        
    Returns:
        RMSE value
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_pred: np.ndarray | Tensor, y_true: np.ndarray | Tensor) -> float:
    """Compute Mean Absolute Error.
    
    Args:
        y_pred: Predicted RUL values
        y_true: True RUL values
        
    Returns:
        MAE value
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    
    return float(np.mean(np.abs(y_pred - y_true)))


def cmapss_score(y_pred: np.ndarray | Tensor, y_true: np.ndarray | Tensor) -> float:
    """Compute the official C-MAPSS asymmetric scoring function.
    
    The scoring function penalizes late predictions more heavily than
    early predictions, reflecting the real-world cost of missing a
    failure versus being overly cautious.
    
    Score = Σ exp(-d/13) - 1  for d < 0 (early prediction)
    Score = Σ exp(d/10) - 1   for d >= 0 (late prediction)
    
    where d = predicted_RUL - true_RUL
    
    Lower scores are better. A perfect prediction (d=0) gives score=0.
    
    Args:
        y_pred: Predicted RUL values
        y_true: True RUL values
        
    Returns:
        C-MAPSS score (lower is better)
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    
    # Compute error: d = predicted - true
    # Positive d = late prediction (predicted higher RUL than actual)
    # Negative d = early prediction (predicted lower RUL than actual)
    d = y_pred - y_true
    
    # Apply asymmetric penalty
    # Early predictions (d < 0): less penalty, a1 = 13
    # Late predictions (d >= 0): more penalty, a2 = 10
    score = np.where(
        d < 0,
        np.exp(-d / 13) - 1,  # Early: exp(-d/13) - 1
        np.exp(d / 10) - 1,   # Late: exp(d/10) - 1
    )
    
    return float(np.sum(score))


def cmapss_score_normalized(y_pred: np.ndarray | Tensor, y_true: np.ndarray | Tensor) -> float:
    """Compute normalized C-MAPSS score (per-sample average).
    
    Useful for comparing across datasets of different sizes.
    
    Args:
        y_pred: Predicted RUL values
        y_true: True RUL values
        
    Returns:
        Normalized C-MAPSS score (lower is better)
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    
    return cmapss_score(y_pred, y_true) / len(y_pred)


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    rmse: float
    mae: float
    cmapss_score: float
    cmapss_score_normalized: float
    n_samples: int
    
    def __repr__(self) -> str:
        return (
            f"EvaluationResults(\n"
            f"  RMSE: {self.rmse:.2f}\n"
            f"  MAE: {self.mae:.2f}\n"
            f"  C-MAPSS Score: {self.cmapss_score:.2f}\n"
            f"  C-MAPSS Score (normalized): {self.cmapss_score_normalized:.4f}\n"
            f"  Samples: {self.n_samples}\n"
            f")"
        )


def evaluate_predictions(
    y_pred: np.ndarray | Tensor,
    y_true: np.ndarray | Tensor,
) -> EvaluationResults:
    """Evaluate RUL predictions with all metrics.
    
    Args:
        y_pred: Predicted RUL values
        y_true: True RUL values
        
    Returns:
        EvaluationResults with all metrics
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, Tensor):
        y_true = y_true.cpu().numpy()
    
    return EvaluationResults(
        rmse=rmse(y_pred, y_true),
        mae=mae(y_pred, y_true),
        cmapss_score=cmapss_score(y_pred, y_true),
        cmapss_score_normalized=cmapss_score_normalized(y_pred, y_true),
        n_samples=len(y_pred),
    )


def compute_per_engine_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    engine_ids: np.ndarray,
) -> dict[int, EvaluationResults]:
    """Compute metrics per engine for detailed analysis.
    
    Args:
        y_pred: Predicted RUL values
        y_true: True RUL values
        engine_ids: Engine ID for each prediction
        
    Returns:
        Dictionary mapping engine ID to EvaluationResults
    """
    unique_engines = np.unique(engine_ids)
    results = {}
    
    for engine_id in unique_engines:
        mask = engine_ids == engine_id
        results[engine_id] = evaluate_predictions(
            y_pred[mask],
            y_true[mask],
        )
    
    return results

