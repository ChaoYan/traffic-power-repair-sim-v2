from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable


UnifiedAction = dict[str, Any]
StrategyFn = Callable[[dict[str, Any], int], UnifiedAction]


@dataclass
class SimulationState:
    roads_remaining: dict[str, float]
    power_remaining: dict[str, float]
    weather_windows: list[bool]
    kappa_walk: float


@dataclass
class ScenarioIndex:
    road_dependencies: dict[str, str | None]
    critical_nodes: set[str]


def _init_state(scenario: dict[str, Any]) -> tuple[SimulationState, ScenarioIndex]:
    roads_remaining = {r["id"]: float(r["repair_hours"]) for r in scenario["roads"]}
    power_remaining = {n["id"]: float(n["repair_hours"]) for n in scenario["power_nodes"]}
    dependencies = {n["id"]: n.get("road_dependency") for n in scenario["power_nodes"]}
    critical = {n["id"] for n in scenario["power_nodes"] if n.get("critical", False)}

    state = SimulationState(
        roads_remaining=roads_remaining,
        power_remaining=power_remaining,
        weather_windows=list(scenario["weather_windows"]),
        kappa_walk=float(scenario.get("kappa_walk", 1.5)),
    )
    index = ScenarioIndex(road_dependencies=dependencies, critical_nodes=critical)
    return state, index


def _active_roads(state: SimulationState) -> list[str]:
    return [rid for rid, remaining in state.roads_remaining.items() if remaining > 0]


def _active_power(state: SimulationState) -> list[str]:
    return [pid for pid, remaining in state.power_remaining.items() if remaining > 0]


def _is_accessible(power_id: str, state: SimulationState, idx: ScenarioIndex) -> bool:
    dep = idx.road_dependencies.get(power_id)
    if dep is None:
        return True
    return state.roads_remaining.get(dep, 0) <= 0


def _assign_ground_task(
    team: dict[str, Any],
    target_type: str,
    target_id: str,
    state: SimulationState,
    idx: ScenarioIndex,
    allow_walk: bool,
) -> dict[str, Any]:
    if target_type == "road":
        work = min(float(team["road_repair_rate"]), state.roads_remaining[target_id])
        state.roads_remaining[target_id] -= work
        effective_rate = float(team["road_repair_rate"])
        mode = "drive"
    else:
        accessible = _is_accessible(target_id, state, idx)
        base = float(team["power_repair_rate"])
        if accessible:
            effective_rate = base
            mode = "drive"
        elif allow_walk:
            effective_rate = base / state.kappa_walk
            mode = "walk"
        else:
            effective_rate = 0.0
            mode = "blocked"

        work = min(effective_rate, state.power_remaining[target_id])
        state.power_remaining[target_id] -= work

    return {
        "team_id": team["id"],
        "target_type": target_type,
        "target_id": target_id,
        "mode": mode,
        "effective_rate": round(effective_rate, 3),
        "work_done": round(work, 3),
    }


def _build_air_manifest(
    strategy: str,
    air_units: list[dict[str, Any]],
    state: SimulationState,
    idx: ScenarioIndex,
    weather_open: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not air_units:
        return [], []

    if not weather_open:
        waits = [
            {
                "unit_id": u["id"],
                "reason": "weather_window_closed",
                "action": "wait",
            }
            for u in air_units
        ]
        return [], waits

    if strategy not in {"S3", "S4"}:
        return [], []

    manifest: list[dict[str, Any]] = []
    active_power = _active_power(state)
    critical_active = [pid for pid in active_power if pid in idx.critical_nodes]
    noncritical_active = [pid for pid in active_power if pid not in idx.critical_nodes]

    for i, unit in enumerate(air_units):
        target = None
        mission = None

        if i < len(critical_active):
            target = critical_active[i]
            if _is_accessible(target, state, idx):
                mission = "transfer"  # 空中转运备件
                effect = float(unit["transfer_rate"])
            else:
                mission = "delivery"  # 空投工具/备件
                effect = float(unit["delivery_rate"])
        elif noncritical_active:
            target = noncritical_active[0]
            mission = "recon"  # 侦察补充态势
            effect = float(unit["recon_rate"])
        else:
            continue

        if target is not None:
            work = min(effect, state.power_remaining[target])
            if mission != "recon":
                state.power_remaining[target] -= work
            manifest.append(
                {
                    "unit_id": unit["id"],
                    "task_type": mission,
                    "target_id": target,
                    "work_done": round(work, 3),
                }
            )

    return manifest, []


def _pick_next_target(ids: list[str]) -> str | None:
    return ids[0] if ids else None


def _strategy_s1(
    scenario: dict[str, Any],
    step: int,
    state: SimulationState,
    idx: ScenarioIndex,
) -> UnifiedAction:
    ground: list[dict[str, Any]] = []
    roads = _active_roads(state)
    powers = _active_power(state)

    for team in scenario["ground_teams"]:
        if roads:
            target = _pick_next_target(roads)
            if target is not None:
                ground.append(_assign_ground_task(team, "road", target, state, idx, allow_walk=False))
            roads = _active_roads(state)
        elif powers:
            target = _pick_next_target(powers)
            if target is not None:
                ground.append(_assign_ground_task(team, "power", target, state, idx, allow_walk=False))
            powers = _active_power(state)

    return {
        "strategy": "S1",
        "step": step,
        "ground_assignments": ground,
    }


def _strategy_s2(
    scenario: dict[str, Any],
    step: int,
    state: SimulationState,
    idx: ScenarioIndex,
) -> UnifiedAction:
    ground: list[dict[str, Any]] = []
    roads = _active_roads(state)
    powers = _active_power(state)

    for i, team in enumerate(scenario["ground_teams"]):
        if i % 2 == 0 and roads:
            target = _pick_next_target(roads)
            if target:
                ground.append(_assign_ground_task(team, "road", target, state, idx, allow_walk=False))
            roads = _active_roads(state)
        elif powers:
            target = _pick_next_target(powers)
            if target:
                ground.append(_assign_ground_task(team, "power", target, state, idx, allow_walk=True))
            powers = _active_power(state)
        elif roads:
            target = _pick_next_target(roads)
            if target:
                ground.append(_assign_ground_task(team, "road", target, state, idx, allow_walk=False))
            roads = _active_roads(state)

    return {
        "strategy": "S2",
        "step": step,
        "ground_assignments": ground,
    }


def _strategy_s3(
    scenario: dict[str, Any],
    step: int,
    state: SimulationState,
    idx: ScenarioIndex,
) -> UnifiedAction:
    ground: list[dict[str, Any]] = []
    roads = _active_roads(state)
    powers = _active_power(state)

    for team in scenario["ground_teams"]:
        target = _pick_next_target(roads if roads else powers)
        if target is None:
            continue
        target_type = "road" if roads else "power"
        ground.append(_assign_ground_task(team, target_type, target, state, idx, allow_walk=False))
        roads = _active_roads(state)
        powers = _active_power(state)

    return {
        "strategy": "S3",
        "step": step,
        "ground_assignments": ground,
    }


def _strategy_s4(
    scenario: dict[str, Any],
    step: int,
    state: SimulationState,
    idx: ScenarioIndex,
) -> UnifiedAction:
    # 地面沿用 S2 的边路边电；空中由主流程统一加上 S3 任务单
    action = _strategy_s2(scenario, step, state, idx)
    action["strategy"] = "S4"
    return action


STRATEGY_MAP: dict[str, StrategyFn] = {
    "S1": _strategy_s1,
    "S2": _strategy_s2,
    "S3": _strategy_s3,
    "S4": _strategy_s4,
}


def run_strategy(scenario: dict[str, Any], strategy: str) -> dict[str, Any]:
    if strategy not in STRATEGY_MAP:
        raise ValueError(f"Unsupported strategy: {strategy}")

    state, idx = _init_state(scenario)
    scenario_local = deepcopy(scenario)
    horizon = int(scenario_local["time_horizon"])
    total_power_nodes = len(scenario_local["power_nodes"])

    action_log: list[UnifiedAction] = []
    recovery_curve: list[dict[str, Any]] = []

    for step in range(horizon):
        planner = STRATEGY_MAP[strategy]
        action = planner(scenario_local, step, state, idx)
        weather_open = state.weather_windows[step % len(state.weather_windows)]

        air_manifest, waits = _build_air_manifest(
            strategy=strategy,
            air_units=scenario_local.get("air_units", []),
            state=state,
            idx=idx,
            weather_open=weather_open,
        )
        action["air_manifest"] = air_manifest
        action["wait_actions"] = waits

        repaired = sum(1 for remain in state.power_remaining.values() if remain <= 0)
        recovery_curve.append(
            {
                "strategy": strategy,
                "step": step,
                "repaired_nodes": repaired,
                "total_nodes": total_power_nodes,
                "recovery_ratio": round(repaired / total_power_nodes, 4),
            }
        )
        action_log.append(action)

        if repaired == total_power_nodes:
            break

    return {
        "strategy": strategy,
        "actions": action_log,
        "recovery_curve": recovery_curve,
        "final_step": action_log[-1]["step"] if action_log else 0,
    }


def run_all_strategies(scenario: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s: run_strategy(scenario, s) for s in ["S1", "S2", "S3", "S4"]}
