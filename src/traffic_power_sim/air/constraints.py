from __future__ import annotations

from .mission_card import (
    ConstraintCheckResult,
    EnvironmentState,
    FlightLimits,
    MissionCard,
)


def check_mission_constraints(
    card: MissionCard,
    limits: FlightLimits,
    env: EnvironmentState,
) -> ConstraintCheckResult:
    violations: list[str] = []

    if limits.fuel_remaining < limits.fuel_min:
        violations.append("fuel_below_minimum")

    if limits.payload > limits.payload_max:
        violations.append("payload_exceeds_max")

    if limits.t_load < 0 or limits.t_unload < 0:
        violations.append("invalid_load_unload_time")

    if env.weather_score < limits.weather_min_score:
        violations.append("weather_restriction")

    if env.is_night and not limits.allow_night:
        violations.append("night_flight_restriction")

    if card.cargo_weight > limits.payload_max:
        violations.append("mission_cargo_exceeds_capacity")

    return ConstraintCheckResult(is_feasible=not violations, violations=violations)
