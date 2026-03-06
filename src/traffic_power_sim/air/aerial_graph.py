from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Mapping

WeatherFactor = Callable[[float], float]


@dataclass(slots=True, frozen=True)
class AirNode:
    node_id: str
    x: float
    y: float


@dataclass(slots=True, frozen=True)
class AirEdge:
    source: str
    target: str
    distance_km: float

    def flight_time(self, v_air_kmh: float, t: float, weather_factor: WeatherFactor) -> float:
        if v_air_kmh <= 0:
            raise ValueError("v_air_kmh must be positive")
        kappa = weather_factor(t)
        if kappa <= 0:
            raise ValueError("weather factor must be positive")
        return self.distance_km / v_air_kmh * kappa


@dataclass(slots=True)
class AerialGraph:
    """GH = (VH, EH)。"""

    nodes: Dict[str, AirNode] = field(default_factory=dict)
    edges: Dict[tuple[str, str], AirEdge] = field(default_factory=dict)

    def add_node(self, node: AirNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: AirEdge, bidirectional: bool = False) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise KeyError("edge endpoints must exist in nodes")
        self.edges[(edge.source, edge.target)] = edge
        if bidirectional:
            self.edges[(edge.target, edge.source)] = AirEdge(
                source=edge.target,
                target=edge.source,
                distance_km=edge.distance_km,
            )

    def neighbors(self, node_id: str) -> Iterable[str]:
        for source, target in self.edges:
            if source == node_id:
                yield target

    def edge_flight_time(
        self,
        source: str,
        target: str,
        v_air_kmh: float,
        t: float,
        weather_factor: WeatherFactor,
    ) -> float:
        edge = self.edges[(source, target)]
        return edge.flight_time(v_air_kmh=v_air_kmh, t=t, weather_factor=weather_factor)


def constant_weather_factor(kappa: float = 1.0) -> WeatherFactor:
    if kappa <= 0:
        raise ValueError("kappa must be positive")

    def _weather_factor(_: float) -> float:
        return kappa

    return _weather_factor


def build_graph(
    nodes: Mapping[str, tuple[float, float]],
    edges: Iterable[tuple[str, str, float]],
    bidirectional: bool = True,
) -> AerialGraph:
    graph = AerialGraph()
    for node_id, (x, y) in nodes.items():
        graph.add_node(AirNode(node_id=node_id, x=x, y=y))
    for source, target, distance_km in edges:
        graph.add_edge(
            AirEdge(source=source, target=target, distance_km=distance_km),
            bidirectional=bidirectional,
        )
    return graph
