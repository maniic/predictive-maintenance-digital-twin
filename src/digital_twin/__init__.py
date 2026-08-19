"""
Digital Twin for Predictive Maintenance

Provides real-time engine simulation, health monitoring, and RUL prediction
for turbofan jet engines based on C-MAPSS data.
"""

from src.digital_twin.predictor import (
    PredictionResult,
    RULPredictor,
)
from src.digital_twin.simulator import (
    DegradationConfig,
    DegradationSimulator,
    FaultMode,
)
from src.digital_twin.state import (
    EngineHistory,
    EngineState,
    OperatingConditions,
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
