from __future__ import annotations

from .core import FacilityState, StrategyTimeline, Task, evaluate_and_export


def _states(powered, reachable, weights):
    return [
        FacilityState(facility_id=f"F{i+1}", weight=w, powered=p, reachable=a)
        for i, (p, a, w) in enumerate(zip(powered, reachable, weights, strict=True))
    ]


def build_demo_strategies() -> list[StrategyTimeline]:
    weights = [0.45, 0.35, 0.20]
    times = [0, 2, 5, 8, 12]

    s1_states = [
        _states([False, False, False], [False, False, False], weights),
        _states([True, False, False], [True, False, False], weights),
        _states([True, True, False], [True, True, False], weights),
        _states([True, True, True], [True, True, False], weights),
        _states([True, True, True], [True, True, True], weights),
    ]

    s2_states = [
        _states([False, False, False], [False, False, False], weights),
        _states([False, True, False], [False, True, False], weights),
        _states([True, True, False], [True, True, False], weights),
        _states([True, True, True], [True, True, True], weights),
        _states([True, True, True], [True, True, True], weights),
    ]

    s3_states = [
        _states([False, False, False], [False, False, False], weights),
        _states([False, False, True], [False, False, True], weights),
        _states([True, False, True], [True, False, True], weights),
        _states([True, True, True], [True, True, True], weights),
        _states([True, True, True], [True, True, True], weights),
    ]

    return [
        StrategyTimeline(
            name="strategy_alpha",
            times=times,
            critical_states=s1_states,
            restored_load=[0, 35, 70, 100, 100],
            tasks=[Task("A1", 2), Task("A2", 5), Task("A3", 8), Task("A4", 12)],
        ),
        StrategyTimeline(
            name="strategy_beta",
            times=times,
            critical_states=s2_states,
            restored_load=[0, 25, 72, 100, 100],
            tasks=[Task("B1", 2), Task("B2", 5), Task("B3", 8), Task("B4", 12)],
        ),
        StrategyTimeline(
            name="strategy_gamma",
            times=times,
            critical_states=s3_states,
            restored_load=[0, 20, 60, 100, 100],
            tasks=[Task("C1", 2), Task("C2", 5), Task("C3", 8), Task("C4", 12)],
        ),
    ]


def main() -> None:
    strategies = build_demo_strategies()
    evaluate_and_export(strategies, out_dir="outputs")


if __name__ == "__main__":
    main()
