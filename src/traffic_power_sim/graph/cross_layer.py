"""Cross-layer mapping: power elements to road nodes/areas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Optional

PowerElement = Hashable
RoadNode = Hashable
Area = Hashable


@dataclass(frozen=True)
class AccessPoint:
    road_node: RoadNode
    area: Optional[Area] = None


class CrossLayerMap:
    def __init__(self) -> None:
        self._element_to_access: Dict[PowerElement, AccessPoint] = {}

    def bind(self, power_element: PowerElement, road_node: RoadNode, area: Optional[Area] = None) -> None:
        self._element_to_access[power_element] = AccessPoint(road_node=road_node, area=area)

    def road_node_for(self, power_element: PowerElement) -> RoadNode:
        return self._element_to_access[power_element].road_node

    def area_for(self, power_element: PowerElement) -> Optional[Area]:
        return self._element_to_access[power_element].area

    def get(self, power_element: PowerElement) -> AccessPoint:
        return self._element_to_access[power_element]
