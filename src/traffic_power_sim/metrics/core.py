from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class FacilityState:
    """State of one critical facility at time t."""

    facility_id: str
    weight: float
    powered: bool
    reachable: bool


@dataclass(frozen=True)
class Task:
    """Repair task metadata used for makespan computation."""

    task_id: str
    completed_at: float


@dataclass
class StrategyTimeline:
    """Step-wise strategy trajectory for resilience evaluation."""

    name: str
    times: Sequence[float]
    critical_states: Sequence[Sequence[FacilityState]]
    restored_load: Sequence[float]
    tasks: Sequence[Task]

    def __post_init__(self) -> None:
        if not (len(self.times) == len(self.critical_states) == len(self.restored_load)):
            raise ValueError(
                "times, critical_states, restored_load must have identical lengths"
            )
        if sorted(self.times) != list(self.times):
            raise ValueError("times must be non-decreasing")


def _weighted_ratio(states: Sequence[FacilityState], attr: str) -> float:
    total_weight = sum(s.weight for s in states)
    if total_weight <= 0:
        return 0.0
    active_weight = sum(s.weight for s in states if getattr(s, attr))
    return active_weight / total_weight


def pk_t(states: Sequence[FacilityState]) -> float:
    """PK(t): weighted power-supply rate for critical facilities."""

    return _weighted_ratio(states, "powered")


def ak_t(states: Sequence[FacilityState]) -> float:
    """AK(t): weighted accessibility rate for critical facilities."""

    return _weighted_ratio(states, "reachable")


def discrete_auc(times: Sequence[float], values: Sequence[float]) -> float:
    """Right-continuous step-wise discrete integral area."""

    if len(times) != len(values):
        raise ValueError("times and values must have same length")
    if len(times) < 2:
        return 0.0

    area = 0.0
    for i in range(len(times) - 1):
        dt = times[i + 1] - times[i]
        if dt < 0:
            raise ValueError("times must be non-decreasing")
        area += values[i] * dt
    return area


def lsd_curve(restored_load: Sequence[float]) -> list[float]:
    """LSD: recovered-load curve values."""

    return list(restored_load)


def makespan(tasks: Sequence[Task]) -> float:
    """All-tasks-completed timestamp."""

    if not tasks:
        return 0.0
    return max(task.completed_at for task in tasks)


def evaluate_strategy(strategy: StrategyTimeline) -> pd.DataFrame:
    """Compute per-time-step metric details for one strategy."""

    df = pd.DataFrame(
        {
            "t": strategy.times,
            "PK": [pk_t(states) for states in strategy.critical_states],
            "AK": [ak_t(states) for states in strategy.critical_states],
            "LSD": lsd_curve(strategy.restored_load),
        }
    )
    df["delta_t"] = df["t"].shift(-1, fill_value=df["t"].iloc[-1]) - df["t"]
    df["auc_contrib"] = df["LSD"] * df["delta_t"]
    return df


def evaluate_and_export(
    strategies: Iterable[StrategyTimeline], out_dir: str | Path = "outputs"
) -> pd.DataFrame:
    """Export per-strategy details + summary table + stepped load plot.

    Chapter-10 consistency rule: when final restored load is equal across strategies,
    compare early-stage recovery using AUC (same terminal outcome, different trajectory).
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = []
    for strategy in strategies:
        detail = evaluate_strategy(strategy)
        auc = discrete_auc(detail["t"].tolist(), detail["LSD"].tolist())
        ms = makespan(strategy.tasks)
        final_load = detail["LSD"].iloc[-1]

        detail.to_csv(out_path / f"{strategy.name}_metrics.csv", index=False)

        rows.append(
            {
                "strategy": strategy.name,
                "AUC": auc,
                "Makespan": ms,
                "FinalRecoveredLoad": final_load,
            }
        )

    summary = pd.DataFrame(rows).sort_values("AUC", ascending=False)

    same_final = summary["FinalRecoveredLoad"].nunique() == 1
    summary["Chapter10Comparable"] = same_final
    if same_final:
        summary["EarlyRecoveryBenefit"] = summary["AUC"] - summary["AUC"].min()
    else:
        summary["EarlyRecoveryBenefit"] = pd.NA

    summary.to_csv(out_path / "strategy_comparison_summary.csv", index=False)
    _plot_lsd_curves_svg(list(strategies), out_path / "lsd_curves.svg")
    return summary


def _plot_lsd_curves_svg(strategies: Sequence[StrategyTimeline], out_file: Path) -> None:
    """Export a diff-friendly step-curve SVG (no binary artifact)."""

    width, height = 900, 520
    pad_left, pad_right, pad_top, pad_bottom = 80, 30, 40, 70
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_times = [t for s in strategies for t in s.times]
    all_loads = [v for s in strategies for v in s.restored_load]
    t_min, t_max = min(all_times), max(all_times)
    y_min, y_max = 0.0, max(max(all_loads), 1)

    def sx(t: float) -> float:
        if t_max == t_min:
            return float(pad_left)
        return pad_left + (t - t_min) / (t_max - t_min) * plot_w

    def sy(v: float) -> float:
        return pad_top + (1 - (v - y_min) / (y_max - y_min)) * plot_h

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18">Recovered Load Curves by Strategy (Step-wise)</text>')

    # axes
    x0, y0 = pad_left, pad_top + plot_h
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{pad_left+plot_w}" y2="{y0}" stroke="#222"/>')
    lines.append(f'<line x1="{x0}" y1="{pad_top}" x2="{x0}" y2="{y0}" stroke="#222"/>')

    # y ticks
    for tick in [0, 25, 50, 75, 100]:
        y = sy(float(tick))
        lines.append(f'<line x1="{x0-5}" y1="{y}" x2="{x0}" y2="{y}" stroke="#666"/>')
        lines.append(f'<text x="{x0-10}" y="{y+4}" text-anchor="end" font-size="12">{tick}</text>')

    # x ticks
    for t in sorted(set(all_times)):
        x = sx(float(t))
        lines.append(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="{y0+5}" stroke="#666"/>')
        lines.append(f'<text x="{x}" y="{y0+22}" text-anchor="middle" font-size="12">{t:g}</text>')

    # labels
    lines.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="14">t</text>')
    lines.append(f'<text x="20" y="{height/2}" text-anchor="middle" font-size="14" transform="rotate(-90 20 {height/2})">Restored Load (LSD)</text>')

    # step curves
    legend_y = pad_top + 10
    for idx, s in enumerate(strategies):
        c = palette[idx % len(palette)]
        points: list[tuple[float,float]] = []
        for i in range(len(s.times)):
            x = sx(float(s.times[i]))
            y = sy(float(s.restored_load[i]))
            points.append((x, y))
            if i < len(s.times) - 1:
                nx = sx(float(s.times[i+1]))
                points.append((nx, y))
        path = " ".join(f"{x:.2f},{y:.2f}" for x,y in points)
        lines.append(f'<polyline fill="none" stroke="{c}" stroke-width="2.5" points="{path}"/>')

        ly = legend_y + idx * 20
        lx = pad_left + plot_w - 180
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" stroke="{c}" stroke-width="3"/>')
        lines.append(f'<text x="{lx+32}" y="{ly+4}" font-size="12">{s.name}</text>')

    lines.append('</svg>')
    out_file.write_text("\n".join(lines), encoding="utf-8")
