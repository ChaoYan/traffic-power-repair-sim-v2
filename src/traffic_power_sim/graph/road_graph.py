"""Road-network model with dynamic edge states and shortest path query."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Hashable, Iterable, List, Optional, Tuple

from .algorithms import dijkstra_dynamic

Node = Hashable


class RoadState(str, Enum):
    OPEN = "open"
    BLOCKED = "blocked"
    RESTORED = "restored"


@dataclass
class RoadEdgeTimeline:
    """State timeline for one road segment.

    events: sorted list of (time, state), state takes effect from that time.
    """

    base_travel_time: float
    events: List[Tuple[float, RoadState]] = field(default_factory=list)

    def state_at(self, t: float) -> RoadState:
        cur = RoadState.OPEN
        for event_t, state in self.events:
            if event_t <= t:
                cur = state
            else:
                break
        return cur

    def travel_time_at(self, t: float) -> Optional[float]:
        state = self.state_at(t)
        if state == RoadState.BLOCKED:
            return None
        return self.base_travel_time


class RoadGraph:
    """Dynamic road graph with shortest path distR(u, v; t)."""

    def __init__(self) -> None:
        self._adj: Dict[Node, List[Node]] = {}
        self._edges: Dict[Tuple[Node, Node], RoadEdgeTimeline] = {}

    def add_edge(
        self,
        u: Node,
        v: Node,
        travel_time: float,
        *,
        bidirectional: bool = True,
        initial_state: RoadState = RoadState.OPEN,
    ) -> None:
        self._adj.setdefault(u, []).append(v)
        self._edges[(u, v)] = RoadEdgeTimeline(
            base_travel_time=travel_time,
            events=[(0.0, initial_state)],
        )

        if bidirectional:
            self._adj.setdefault(v, []).append(u)
            self._edges[(v, u)] = RoadEdgeTimeline(
                base_travel_time=travel_time,
                events=[(0.0, initial_state)],
            )

    def set_edge_state(self, u: Node, v: Node, t: float, state: RoadState) -> None:
        timeline = self._edges[(u, v)]
        timeline.events.append((t, state))
        timeline.events.sort(key=lambda x: x[0])

    def edge_state(self, u: Node, v: Node, t: float) -> RoadState:
        return self._edges[(u, v)].state_at(t)

    def neighbors(self, u: Node) -> Iterable[Node]:
        return self._adj.get(u, [])

    def distR(self, source: Node, target: Node, t: float) -> float:
        """Shortest travel time between nodes at time t."""

        def weight_fn(u: Node, v: Node) -> Optional[float]:
            timeline = self._edges.get((u, v))
            if timeline is None:
                return None
            return timeline.travel_time_at(t)

        dist, _ = dijkstra_dynamic(self._adj, source, target, weight_fn=weight_fn)
        return dist.get(target, float("inf"))
