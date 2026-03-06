from __future__ import annotations

import random
from copy import deepcopy
from typing import Dict, List, Optional

from sim.interface import SimState, SimStepResult, SimTask


class MockSimulator:
    """Tiny deterministic-ish simulator for env shakedown.

    This stands in for real sim integration until production backend is wired.
    """

    def __init__(self, seed: int = 7):
        self._rng = random.Random(seed)
        self._base_tasks = [
            SimTask("T1", "N1", {"crew": 1, "truck": 1}, 0, 8, 0.2),
            SimTask("T2", "N2", {"crew": 1, "drone": 1}, 1, 10, 0.6),
            SimTask("T3", "N3", {"crew": 2}, 2, 12, 0.4),
        ]
        self._state = self._initial_state()

    def _initial_state(self) -> SimState:
        return SimState(
            local_traffic={"N1": 0.4, "N2": 0.7, "N3": 0.5},
            nearby_tasks=deepcopy(self._base_tasks),
            remaining_resources={"crew": 2, "truck": 1, "drone": 1},
            current_time=0,
            time_window={"start": 0, "end": 14},
            keypoint_status={"KP_A": 0.2, "KP_B": 0.0},
            weather_blocked_locations=[],
            night_operation_forbidden=True,
            reachable_task_ids=["T1", "T2", "T3"],
            metrics={"PK": 0.0, "AK": 0.0, "LSD": 0.0},
        )

    def reset(self) -> SimState:
        self._state = self._initial_state()
        return deepcopy(self._state)

    def step(self, task_id: Optional[str]) -> SimStepResult:
        state = deepcopy(self._state)
        info: Dict[str, float] = {"risk_penalty": 0.0, "empty_mileage_penalty": 0.0, "night_violation_penalty": 0.0}

        if task_id is None:
            state.current_time += 1
            info["empty_mileage_penalty"] = 0.1
        else:
            task = next((t for t in state.nearby_tasks if t.task_id == task_id), None)
            if task is not None:
                state.current_time += 1
                state.nearby_tasks = [t for t in state.nearby_tasks if t.task_id != task_id]
                for k, v in task.required_resources.items():
                    state.remaining_resources[k] = max(0.0, state.remaining_resources.get(k, 0.0) - v)
                state.metrics["PK"] += 1.0
                state.metrics["AK"] += max(0.0, 1.0 - task.risk_score)
                state.metrics["LSD"] += 0.6
                state.keypoint_status["KP_A"] = min(1.0, state.keypoint_status["KP_A"] + 0.2)
                info["risk_penalty"] = task.risk_score
                if state.current_time > 10 and state.night_operation_forbidden:
                    info["night_violation_penalty"] = 1.0
            else:
                state.current_time += 1
                info["empty_mileage_penalty"] = 0.2

        if self._rng.random() < 0.15:
            state.weather_blocked_locations = ["N2"]
        else:
            state.weather_blocked_locations = []

        state.reachable_task_ids = [
            t.task_id
            for t in state.nearby_tasks
            if t.location not in state.weather_blocked_locations and state.local_traffic.get(t.location, 1.0) < 0.95
        ]

        done = state.current_time >= state.time_window["end"] or not state.nearby_tasks
        self._state = state
        return SimStepResult(next_state=deepcopy(state), done=done, info=info)

    def estimate_counterfactual_baseline(self, state: SimState, task_id: Optional[str]) -> float:
        """Simple baseline: expected one-step gain under waiting action."""
        _ = state, task_id
        return 0.15
