"""Schema definitions for the traffic-power coupled repair simulator.

The field set follows appendix A.2 in ``报告v7.pdf``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RoadStatus(str, Enum):
    OPEN = "open"
    DAMAGED = "damaged"
    BLOCKED = "blocked"
    UNDER_REPAIR = "under_repair"
    REPAIRED = "repaired"


class PowerLineStatus(str, Enum):
    ENERGIZED = "energized"
    DEGRADED = "degraded"
    OUTAGE = "outage"
    UNDER_REPAIR = "under_repair"
    RESTORED = "restored"


class FaultType(str, Enum):
    ROAD = "road"
    POWER = "power"
    COUPLED = "coupled"
    WEATHER = "weather"


class CrewType(str, Enum):
    ROAD = "road"
    POWER = "power"
    JOINT = "joint"
    AIR = "air"


class SiteType(str, Enum):
    HOSPITAL = "hospital"
    SHELTER = "shelter"
    SUBSTATION = "substation"
    WATER_PLANT = "water_plant"
    OTHER = "other"


@dataclass(slots=True)
class RoadEdge:
    edge_id: str
    u: str
    v: str
    length: float
    status: RoadStatus
    base_speed: float
    travel_time: float
    repair_time: float


@dataclass(slots=True)
class PowerLine:
    line_id: str
    u: str
    v: str
    weight: float
    status: PowerLineStatus
    repair_time: float
    critical_weight: float


@dataclass(slots=True)
class FaultEvent:
    event_id: str
    type: FaultType
    geo: str
    affected_edges: list[str] = field(default_factory=list)
    affected_lines: list[str] = field(default_factory=list)
    start_time: datetime | None = None


@dataclass(slots=True)
class CrossLayerMap:
    power_element_id: str
    road_node: str | None = None
    area: str | None = None
    secondary_effect_params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Crew:
    crew_id: str
    type: CrewType
    start_node: str
    shift_start: datetime
    shift_end: datetime
    skill: str
    speed_profile: str


@dataclass(slots=True)
class CriticalSite:
    site_id: str
    type: SiteType
    road_node: str
    power_node: str
    demand: float
    weight: float
    threshold_eta: float


@dataclass(slots=True)
class AirAsset:
    heli_id: str
    base: str
    fuel: float
    v_air: float
    payload: float
    load_unload: float
    weather_limits: str
    lz_set: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ScenarioData:
    road_edges: list[RoadEdge]
    power_lines: list[PowerLine]
    faults_events: list[FaultEvent]
    cross_layer_map: list[CrossLayerMap]
    crews: list[Crew]
    critical_sites: list[CriticalSite]
    air_assets: list[AirAsset]
