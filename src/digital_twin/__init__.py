"""
Digital Twin for Predictive Maintenance

Provides real-time engine simulation, health monitoring, and RUL prediction
for turbofan jet engines based on C-MAPSS data.
"""

from src.digital_twin.state import (
    EngineState,
    EngineHistory,
    OperatingConditions,
)
from src.digital_twin.simulator import (
    DegradationSimulator,
    DegradationConfig,
    FaultMode,
)
from src.digital_twin.predictor import (
    RULPredictor,
    PredictionResult,
)

__all__ = [
    # State
    "EngineState",
    "EngineHistory",
    "OperatingConditions",
    # Simulation
    "DegradationSimulator",
    "DegradationConfig",
    "FaultMode",
    # Prediction
    "RULPredictor",
    "PredictionResult",
]
