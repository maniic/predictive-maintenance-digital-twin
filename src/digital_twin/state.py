"""
Engine State Management for Digital Twin

Provides dataclasses for representing engine state, sensor readings,
and historical tracking for visualization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class OperatingConditions:
    """Operating conditions for the engine.

    Maps to C-MAPSS operational settings.
    """
    altitude: float = 0.0          # setting_1
    mach_number: float = 0.0       # setting_2
    throttle_resolver: float = 0.0 # setting_3


@dataclass
class EngineState:
    """Current state of an engine in the digital twin.

    Attributes:
        engine_id: Unique identifier for this engine
        cycle: Current operating cycle
        operating_conditions: Current flight conditions
        sensor_readings: Dictionary of sensor_id -> value
        predicted_rul: Predicted remaining useful life (cycles)
        uncertainty: Standard deviation of RUL prediction
        health_score: Overall health score (0.0 = failed, 1.0 = healthy)
        timestamp: When this state was recorded
    """
    engine_id: str
    cycle: int
    operating_conditions: OperatingConditions
    sensor_readings: dict[str, float]
    predicted_rul: float
    uncertainty: float
    health_score: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def rul_lower_bound(self) -> float:
        """Lower bound of 95% confidence interval."""
        return max(0, self.predicted_rul - 1.96 * self.uncertainty)

    @property
    def rul_upper_bound(self) -> float:
        """Upper bound of 95% confidence interval."""
        return self.predicted_rul + 1.96 * self.uncertainty

    def to_dict(self) -> dict:
        """Convert state to dictionary for serialization."""
        return {
            "engine_id": self.engine_id,
            "cycle": self.cycle,
            "operating_conditions": {
                "altitude": self.operating_conditions.altitude,
                "mach_number": self.operating_conditions.mach_number,
                "throttle_resolver": self.operating_conditions.throttle_resolver,
            },
            "sensor_readings": self.sensor_readings.copy(),
            "predicted_rul": self.predicted_rul,
            "uncertainty": self.uncertainty,
            "health_score": self.health_score,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EngineState":
        """Create state from dictionary."""
        return cls(
            engine_id=data["engine_id"],
            cycle=data["cycle"],
            operating_conditions=OperatingConditions(
                altitude=data["operating_conditions"]["altitude"],
                mach_number=data["operating_conditions"]["mach_number"],
                throttle_resolver=data["operating_conditions"]["throttle_resolver"],
            ),
            sensor_readings=data["sensor_readings"],
            predicted_rul=data["predicted_rul"],
            uncertainty=data["uncertainty"],
            health_score=data["health_score"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class EngineHistory:
    """Track engine state history for visualization.

    Maintains a sliding window of recent states.
    """
    engine_id: str
    max_length: int = 500
    _states: list[EngineState] = field(default_factory=list)

    def add(self, state: EngineState) -> None:
        """Add a state to history."""
        self._states.append(state)
        # Trim if over max length
        if len(self._states) > self.max_length:
            self._states = self._states[-self.max_length:]

    def get_recent(self, n: Optional[int] = None) -> list[EngineState]:
        """Get n most recent states."""
        if n is None:
            return self._states.copy()
        return self._states[-n:]

    def clear(self) -> None:
        """Clear history."""
        self._states = []

    @property
    def cycles(self) -> list[int]:
        """Get list of cycle numbers."""
        return [s.cycle for s in self._states]

    @property
    def rul_values(self) -> list[float]:
        """Get list of RUL predictions."""
        return [s.predicted_rul for s in self._states]

    @property
    def uncertainty_values(self) -> list[float]:
        """Get list of uncertainty values."""
        return [s.uncertainty for s in self._states]

    @property
    def health_scores(self) -> list[float]:
        """Get list of health scores."""
        return [s.health_score for s in self._states]

    def get_sensor_series(self, sensor_id: str) -> list[float]:
        """Get time series for a specific sensor."""
        return [s.sensor_readings.get(sensor_id, 0.0) for s in self._states]

    def __len__(self) -> int:
        return len(self._states)
