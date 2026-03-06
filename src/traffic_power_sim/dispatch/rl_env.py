from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from sim.interface import SimState, SimulatorAdapter


@dataclass
class RewardConfig:
    pk_weight: float = 1.0
    ak_weight: float = 0.8
    lsd_weight: float = 0.5
    risk_penalty_weight: float = 0.4
    empty_mileage_weight: float = 0.3
    night_violation_weight: float = 1.0
    counterfactual_alpha: float = 1.0


class TrafficPowerRepairEnv:
    """Task-level RL env integrated with simulator adapter under sim/.

    Action space:
      - 0: wait
      - 1..N: choose task in current state's nearby_tasks order
    """

    WAIT_ACTION = 0

    def __init__(self, sim: SimulatorAdapter, reward_config: Optional[RewardConfig] = None, seed: int = 0):
        self.sim = sim
        self.reward_config = reward_config or RewardConfig()
        self.rng = random.Random(seed)
        self._state: Optional[SimState] = None
        self._prev_metrics: Dict[str, float] = {"PK": 0.0, "AK": 0.0, "LSD": 0.0}

    def reset(self) -> Dict[str, Any]:
        self._state = self.sim.reset()
        self._prev_metrics = deepcopy(self._state.metrics)
        return self._build_observation(self._state)

    def get_action_mask(self) -> List[bool]:
        state = self._require_state()
        # wait + each nearby task
        mask = [True]
        for task in state.nearby_tasks:
            legal = self._is_task_action_legal(state, task.task_id)
            mask.append(legal)
        return mask

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        state = self._require_state()
        task_id = self._action_to_task_id(state, action)

        # hard mask guard
        if action != self.WAIT_ACTION and not self._is_task_action_legal(state, task_id):
            step_result = self.sim.step(None)
            info = dict(step_result.info)
            info["invalid_action"] = 1.0
            info["selected_task_id"] = task_id
        else:
            step_result = self.sim.step(task_id)
            info = dict(step_result.info)
            info["invalid_action"] = 0.0
            info["selected_task_id"] = task_id

        reward = self._compute_reward(step_result.next_state, info, task_id)
        self._state = step_result.next_state
        obs = self._build_observation(self._state)
        return obs, reward, step_result.done, info

    def random_policy_action(self) -> int:
        mask = self.get_action_mask()
        legal_actions = [idx for idx, ok in enumerate(mask) if ok]
        return self.rng.choice(legal_actions)

    def run_random_policy(self, max_steps: int = 100) -> List[Dict[str, Any]]:
        if self._state is None:
            self.reset()

        trajectory = []
        for _ in range(max_steps):
            action = self.random_policy_action()
            obs, reward, done, info = self.step(action)
            trajectory.append({"action": action, "reward": reward, "done": done, "info": info, "obs": obs})
            if done:
                break
        return trajectory

    def replay_baseline_policy(
        self, baseline_policy: Callable[[Dict[str, Any], List[bool]], int], max_steps: int = 100
    ) -> List[Dict[str, Any]]:
        """Replay deterministic/stochastic baseline policy for A/B checks."""
        if self._state is None:
            self.reset()

        logs = []
        for _ in range(max_steps):
            obs = self._build_observation(self._require_state())
            mask = self.get_action_mask()
            action = baseline_policy(obs, mask)
            if action < 0 or action >= len(mask):
                action = self.WAIT_ACTION
            next_obs, reward, done, info = self.step(action)
            logs.append(
                {
                    "obs": obs,
                    "mask": mask,
                    "action": action,
                    "next_obs": next_obs,
                    "reward": reward,
                    "done": done,
                    "info": info,
                }
            )
            if done:
                break
        return logs

    def _build_observation(self, state: SimState) -> Dict[str, Any]:
        return {
            "local_traffic": deepcopy(state.local_traffic),
            "nearby_tasks": [
                {
                    "task_id": t.task_id,
                    "location": t.location,
                    "required_resources": deepcopy(t.required_resources),
                    "earliest_start": t.earliest_start,
                    "latest_end": t.latest_end,
                    "risk_score": t.risk_score,
                }
                for t in state.nearby_tasks
            ],
            "remaining_resources": deepcopy(state.remaining_resources),
            "time_window": {"current": state.current_time, **deepcopy(state.time_window)},
            "keypoint_status": deepcopy(state.keypoint_status),
        }

    def _compute_reward(self, next_state: SimState, info: Dict[str, Any], task_id: Optional[str]) -> float:
        cfg = self.reward_config
        pk_delta = next_state.metrics.get("PK", 0.0) - self._prev_metrics.get("PK", 0.0)
        ak_delta = next_state.metrics.get("AK", 0.0) - self._prev_metrics.get("AK", 0.0)
        lsd_delta = next_state.metrics.get("LSD", 0.0) - self._prev_metrics.get("LSD", 0.0)

        main_reward = cfg.pk_weight * pk_delta + cfg.ak_weight * ak_delta + cfg.lsd_weight * lsd_delta
        penalty = (
            cfg.risk_penalty_weight * float(info.get("risk_penalty", 0.0))
            + cfg.empty_mileage_weight * float(info.get("empty_mileage_penalty", 0.0))
            + cfg.night_violation_weight * float(info.get("night_violation_penalty", 0.0))
        )

        # Cooperative credit assignment: differential reward with counterfactual baseline.
        try:
            baseline = float(self.sim.estimate_counterfactual_baseline(next_state, task_id))
        except Exception:
            baseline = 0.0
        cooperative_term = cfg.counterfactual_alpha * (main_reward - baseline)

        reward = main_reward - penalty + cooperative_term
        self._prev_metrics = deepcopy(next_state.metrics)
        return reward

    def _action_to_task_id(self, state: SimState, action: int) -> Optional[str]:
        if action == self.WAIT_ACTION:
            return None
        task_idx = action - 1
        if 0 <= task_idx < len(state.nearby_tasks):
            return state.nearby_tasks[task_idx].task_id
        return None

    def _is_task_action_legal(self, state: SimState, task_id: Optional[str]) -> bool:
        if task_id is None:
            return True

        task = next((x for x in state.nearby_tasks if x.task_id == task_id), None)
        if task is None:
            return False

        if state.reachable_task_ids and task_id not in state.reachable_task_ids:
            return False

        # Weather lockdown check.
        if task.location in state.weather_blocked_locations:
            return False

        # Resource feasibility check.
        for res_name, need in task.required_resources.items():
            if state.remaining_resources.get(res_name, 0.0) < need:
                return False

        # Basic time-window check.
        if state.current_time > task.latest_end:
            return False

        return True

    def _require_state(self) -> SimState:
        if self._state is None:
            raise RuntimeError("env not reset")
        return self._state
