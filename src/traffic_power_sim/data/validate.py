"""Validation helpers for scenario data integrity checks."""

from __future__ import annotations

from collections import Counter

from .schema import ScenarioData


class ScenarioValidationError(ValueError):
    """Raised when one or more scenario validation checks fail."""


def _assert_unique(ids: list[str], label: str, errors: list[str]) -> None:
    dupes = [value for value, count in Counter(ids).items() if count > 1]
    if dupes:
        errors.append(f"{label} primary keys are not unique: {dupes}")


def validate_scenario(data: ScenarioData) -> None:
    """Validate key constraints required by appendix A.2 data model."""

    errors: list[str] = []

    _assert_unique([r.edge_id for r in data.road_edges], "road_edges.edge_id", errors)
    _assert_unique([p.line_id for p in data.power_lines], "power_lines.line_id", errors)
    _assert_unique([e.event_id for e in data.faults_events], "faults_events.event_id", errors)
    _assert_unique([c.power_element_id for c in data.cross_layer_map], "cross_layer_map.power_element_id", errors)
    _assert_unique([c.crew_id for c in data.crews], "crews.crew_id", errors)
    _assert_unique([s.site_id for s in data.critical_sites], "critical_sites.site_id", errors)
    _assert_unique([a.heli_id for a in data.air_assets], "air_assets.heli_id", errors)

    road_edge_ids = {r.edge_id for r in data.road_edges}
    road_nodes = {r.u for r in data.road_edges} | {r.v for r in data.road_edges}
    power_line_ids = {p.line_id for p in data.power_lines}
    power_nodes = {p.u for p in data.power_lines} | {p.v for p in data.power_lines}

    for event in data.faults_events:
        missing_edges = set(event.affected_edges) - road_edge_ids
        if missing_edges:
            errors.append(f"fault event {event.event_id} references unknown road edges: {sorted(missing_edges)}")
        missing_lines = set(event.affected_lines) - power_line_ids
        if missing_lines:
            errors.append(f"fault event {event.event_id} references unknown power lines: {sorted(missing_lines)}")

    for crew in data.crews:
        if crew.start_node not in road_nodes:
            errors.append(f"crew {crew.crew_id} start_node does not exist in road graph: {crew.start_node}")
        if crew.shift_start is None or crew.shift_end is None or crew.shift_end <= crew.shift_start:
            errors.append(f"crew {crew.crew_id} has invalid shift window")

    for site in data.critical_sites:
        if site.road_node not in road_nodes:
            errors.append(f"critical site {site.site_id} road_node missing in road graph: {site.road_node}")
        if site.power_node not in power_nodes:
            errors.append(f"critical site {site.site_id} power_node missing in power graph: {site.power_node}")

    mapped_power_elements = {m.power_element_id for m in data.cross_layer_map}
    unmapped_lines = sorted(power_line_ids - mapped_power_elements)
    if unmapped_lines:
        errors.append(f"cross_layer_map incomplete for power lines: {unmapped_lines}")

    for mapping in data.cross_layer_map:
        if mapping.power_element_id not in power_line_ids:
            errors.append(
                f"cross_layer_map references unknown power element: {mapping.power_element_id}"
            )
        if not mapping.road_node and not mapping.area:
            errors.append(
                f"cross_layer_map {mapping.power_element_id} must define road_node or area"
            )
        if mapping.road_node and mapping.road_node not in road_nodes:
            errors.append(
                f"cross_layer_map {mapping.power_element_id} road_node does not exist: {mapping.road_node}"
            )

    if errors:
        raise ScenarioValidationError("; ".join(errors))
