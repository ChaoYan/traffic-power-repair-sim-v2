"""Scenario file loaders for CSV/JSON data sources."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .schema import (
    AirAsset,
    Crew,
    CrewType,
    CriticalSite,
    CrossLayerMap,
    FaultEvent,
    FaultType,
    PowerLine,
    PowerLineStatus,
    RoadEdge,
    RoadStatus,
    ScenarioData,
    SiteType,
)


TABLES = [
    "road_edges",
    "power_lines",
    "faults_events",
    "cross_layer_map",
    "crews",
    "critical_sites",
    "air_assets",
]


def _parse_dt(value: str | None) -> datetime | None:
    if value in (None, ""):
        return None
    return datetime.fromisoformat(value)


def _parse_json_value(value: str | None, default: Any) -> Any:
    if value in (None, ""):
        return default
    return json.loads(value)


def _read_table(path: Path, table_name: str) -> list[dict[str, Any]]:
    csv_path = path / f"{table_name}.csv"
    json_path = path / f"{table_name}.json"
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, list):
                msg = f"{json_path} must contain a JSON array"
                raise ValueError(msg)
            return data
    msg = f"Missing table file for {table_name}: expected .csv or .json under {path}"
    raise FileNotFoundError(msg)


def load_scenario(path: str | Path) -> ScenarioData:
    base = Path(path)

    road_edges = [
        RoadEdge(
            edge_id=row["edge_id"],
            u=row["u"],
            v=row["v"],
            length=float(row["length"]),
            status=RoadStatus(row["status"]),
            base_speed=float(row["base_speed"]),
            travel_time=float(row["travel_time"]),
            repair_time=float(row["repair_time"]),
        )
        for row in _read_table(base, "road_edges")
    ]

    power_lines = [
        PowerLine(
            line_id=row["line_id"],
            u=row["u"],
            v=row["v"],
            weight=float(row["weight"]),
            status=PowerLineStatus(row["status"]),
            repair_time=float(row["repair_time"]),
            critical_weight=float(row["critical_weight"]),
        )
        for row in _read_table(base, "power_lines")
    ]

    faults_events = [
        FaultEvent(
            event_id=row["event_id"],
            type=FaultType(row["type"]),
            geo=row["geo"],
            affected_edges=_parse_json_value(row.get("affected_edges"), []),
            affected_lines=_parse_json_value(row.get("affected_lines"), []),
            start_time=_parse_dt(row.get("start_time")),
        )
        for row in _read_table(base, "faults_events")
    ]

    cross_layer_map = [
        CrossLayerMap(
            power_element_id=row["power_element_id"],
            road_node=row.get("road_node") or None,
            area=row.get("area") or None,
            secondary_effect_params=_parse_json_value(row.get("secondary_effect_params"), {}),
        )
        for row in _read_table(base, "cross_layer_map")
    ]

    crews = [
        Crew(
            crew_id=row["crew_id"],
            type=CrewType(row["type"]),
            start_node=row["start_node"],
            shift_start=_parse_dt(row["shift_start"]),
            shift_end=_parse_dt(row["shift_end"]),
            skill=row["skill"],
            speed_profile=row["speed_profile"],
        )
        for row in _read_table(base, "crews")
    ]

    critical_sites = [
        CriticalSite(
            site_id=row["site_id"],
            type=SiteType(row["type"]),
            road_node=row["road_node"],
            power_node=row["power_node"],
            demand=float(row["demand"]),
            weight=float(row["weight"]),
            threshold_eta=float(row["threshold_eta"]),
        )
        for row in _read_table(base, "critical_sites")
    ]

    air_assets = [
        AirAsset(
            heli_id=row["heli_id"],
            base=row["base"],
            fuel=float(row["fuel"]),
            v_air=float(row["v_air"]),
            payload=float(row["payload"]),
            load_unload=float(row["load_unload"]),
            weather_limits=row["weather_limits"],
            lz_set=_parse_json_value(row.get("lz_set"), []),
        )
        for row in _read_table(base, "air_assets")
    ]

    return ScenarioData(
        road_edges=road_edges,
        power_lines=power_lines,
        faults_events=faults_events,
        cross_layer_map=cross_layer_map,
        crews=crews,
        critical_sites=critical_sites,
        air_assets=air_assets,
    )
