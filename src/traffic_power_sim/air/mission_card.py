from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Mapping, Optional


class MissionType(str, Enum):
    """空中任务类型。"""

    PERSONNEL_DELIVERY = "personnel_delivery"  # 人员投送
    EQUIPMENT_DELIVERY = "equipment_delivery"  # 设备投送
    RECON = "recon"  # 侦察
    TRANSFER = "transfer"  # 转运


@dataclass(slots=True, frozen=True)
class FlightLimits:
    """飞行约束参数。"""

    fuel_remaining: float
    fuel_min: float
    payload: float
    payload_max: float
    t_load: float
    t_unload: float
    allow_night: bool = False
    weather_min_score: float = 0.0


@dataclass(slots=True, frozen=True)
class MissionCard:
    """统一任务单结构。"""

    mission_id: str
    mission_type: MissionType
    origin: str
    destination: str
    departure_t: float
    cargo_weight: float = 0.0
    personnel_count: int = 0
    equipment: Mapping[str, int] = field(default_factory=dict)
    notes: str = ""
    unlock_island_task_ids: tuple[str, ...] = field(default_factory=tuple)
    boost_alpha_night: float = 0.0

    def equipment_total(self) -> int:
        return sum(self.equipment.values())


@dataclass(slots=True, frozen=True)
class EnvironmentState:
    """用于约束检查的环境状态。"""

    weather_score: float
    is_night: bool


@dataclass(slots=True)
class ConstraintCheckResult:
    """约束检查结果。"""

    is_feasible: bool
    violations: list[str]


@dataclass(slots=True)
class AirCoordinationEffect:
    """空地协同效果。"""

    unlocked_task_ids: set[str] = field(default_factory=set)
    night_alpha_boost: float = 0.0

    @classmethod
    def combine(cls, effects: Iterable["AirCoordinationEffect"]) -> "AirCoordinationEffect":
        merged = cls()
        for effect in effects:
            merged.unlocked_task_ids.update(effect.unlocked_task_ids)
            merged.night_alpha_boost += effect.night_alpha_boost
        return merged


def coordination_effect_from_mission(card: MissionCard) -> AirCoordinationEffect:
    """从任务单推导协同接口输出。"""

    return AirCoordinationEffect(
        unlocked_task_ids=set(card.unlock_island_task_ids),
        night_alpha_boost=max(card.boost_alpha_night, 0.0),
    )


def make_equipment_map(items: Optional[Dict[str, int]] = None) -> Mapping[str, int]:
    return {} if items is None else dict(items)
