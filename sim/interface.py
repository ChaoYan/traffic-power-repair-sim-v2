from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


@dataclass
class SimTask:
    """Task view exposed by simulator."""

    task_id: str
    location: str
    required_resources: Dict[str, float]
    earliest_start: int
    latest_end: int
    risk_score: float = 0.0


@dataclass
class SimState:
    """Minimal state expected by RL environment."""

    local_traffic: Dict[str, float]
    nearby_tasks: List[SimTask]
    remaining_resources: Dict[str, float]
    current_time: int
    time_window: Dict[str, int]
    keypoint_status: Dict[str, float]
    weather_blocked_locations: List[str] = field(default_factory=list)
    night_operation_forbidden: bool = False
    reachable_task_ids: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=lambda: {"PK": 0.0, "AK": 0.0, "LSD": 0.0})


@dataclass
class SimStepResult:
    next_state: SimState
    done: bool
    info: Dict[str, float] = field(default_factory=dict)


class SimulatorAdapter(Protocol):
    """Adapter to decouple RL env from concrete implementation under sim/."""

    def reset(self) -> SimState:
        ...

    def step(self, task_id: Optional[str]) -> SimStepResult:
        ...

    def estimate_counterfactual_baseline(self, state: SimState, task_id: Optional[str]) -> float:
        """Optional baseline used for cooperative credit assignment."""
        ...
