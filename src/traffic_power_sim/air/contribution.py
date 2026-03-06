from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(slots=True, frozen=True)
class MarginalContribution:
    threshold_lead_time: float
    auc_gain: float


def threshold_hit_time(times: Sequence[float], values: Sequence[float], threshold: float) -> float | None:
    for t, v in zip(times, values):
        if v >= threshold:
            return t
    return None


def trapezoid_auc(times: Sequence[float], values: Sequence[float]) -> float:
    if len(times) != len(values):
        raise ValueError("times and values length mismatch")
    if len(times) < 2:
        return 0.0

    area = 0.0
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt < 0:
            raise ValueError("times must be non-decreasing")
        area += dt * (values[i] + values[i - 1]) / 2
    return area


def compute_marginal_contribution(
    times: Sequence[float],
    baseline_alpha: Sequence[float],
    with_air_alpha: Sequence[float],
    threshold: float,
) -> MarginalContribution:
    if not (len(times) == len(baseline_alpha) == len(with_air_alpha)):
        raise ValueError("input series length mismatch")

    baseline_hit = threshold_hit_time(times, baseline_alpha, threshold)
    with_air_hit = threshold_hit_time(times, with_air_alpha, threshold)

    if baseline_hit is None or with_air_hit is None:
        lead = 0.0
    else:
        lead = baseline_hit - with_air_hit

    auc_gain = trapezoid_auc(times, with_air_alpha) - trapezoid_auc(times, baseline_alpha)
    return MarginalContribution(threshold_lead_time=lead, auc_gain=auc_gain)


def apply_night_alpha_boost(alpha: Iterable[float], boost: float) -> list[float]:
    if boost <= 0:
        return list(alpha)
    return [min(1.0, v + boost) for v in alpha]
