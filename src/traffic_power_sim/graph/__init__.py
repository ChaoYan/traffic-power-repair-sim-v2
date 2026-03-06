"""Unified graph interfaces for the scheduling layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable

from .cross_layer import CrossLayerMap
from .power_graph import PowerGraph
from .road_graph import RoadGraph


@dataclass(frozen=True)
class Crew:
    crew_id: Hashable
    road_node: Hashable


@dataclass(frozen=True)
class Task:
    task_id: Hashable
    power_element: Hashable


class DispatchGraphInterface:
    def __init__(
        self,
        road_graph: RoadGraph,
        power_graph: PowerGraph,
        cross_layer: CrossLayerMap,
        source_buses: Iterable[Hashable],
    ) -> None:
        self.road_graph = road_graph
        self.power_graph = power_graph
        self.cross_layer = cross_layer
        self.source_buses = tuple(source_buses)

    def is_reachable(self, task: Task, crew: Crew, t: float) -> bool:
        task_node = self.cross_layer.road_node_for(task.power_element)
        return self.road_graph.distR(crew.road_node, task_node, t) < float("inf")

    def eta_to_task(self, task: Task, crew: Crew, t: float) -> float:
        task_node = self.cross_layer.road_node_for(task.power_element)
        return self.road_graph.distR(crew.road_node, task_node, t)

    def power_restorable(self, line_id: Hashable, t: float) -> bool:
        return self.power_graph.power_restorable(line_id, t, self.source_buses)


__all__ = [
    "Crew",
    "Task",
    "CrossLayerMap",
    "PowerGraph",
    "RoadGraph",
    "DispatchGraphInterface",
]
