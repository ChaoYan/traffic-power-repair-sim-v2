"""Power-network model with dynamic line states and restorability checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple

from .algorithms import bfs_reachable

Node = Hashable
LineID = Hashable


class LineState(str, Enum):
    UP = "up"
    DOWN = "down"
    REPAIRED = "repaired"


@dataclass
class LineTimeline:
    u: Node
    v: Node
    events: List[Tuple[float, LineState]] = field(default_factory=list)

    def state_at(self, t: float) -> LineState:
        cur = LineState.UP
        for event_t, state in self.events:
            if event_t <= t:
                cur = state
            else:
                break
        return cur

    def is_energizable(self, t: float) -> bool:
        return self.state_at(t) != LineState.DOWN


class PowerGraph:
    """Power graph with line states and load supply computation."""

    def __init__(self) -> None:
        self._lines: Dict[LineID, LineTimeline] = {}
        self._loads: Dict[Node, float] = {}

    def add_line(
        self,
        line_id: LineID,
        u: Node,
        v: Node,
        *,
        initial_state: LineState = LineState.UP,
    ) -> None:
        self._lines[line_id] = LineTimeline(u=u, v=v, events=[(0.0, initial_state)])

    def set_line_state(self, line_id: LineID, t: float, state: LineState) -> None:
        line = self._lines[line_id]
        line.events.append((t, state))
        line.events.sort(key=lambda x: x[0])

    def line_state(self, line_id: LineID, t: float) -> LineState:
        return self._lines[line_id].state_at(t)

    def set_load(self, bus: Node, demand: float) -> None:
        self._loads[bus] = demand

    def active_adjacency(self, t: float) -> Dict[Node, List[Node]]:
        adj: Dict[Node, List[Node]] = {}
        for line in self._lines.values():
            if not line.is_energizable(t):
                continue
            adj.setdefault(line.u, []).append(line.v)
            adj.setdefault(line.v, []).append(line.u)
        return adj

    def energizable_buses(self, source_buses: Iterable[Node], t: float) -> Set[Node]:
        adj = self.active_adjacency(t)
        powered: Set[Node] = set()
        for source in source_buses:
            if source in powered:
                continue
            powered |= bfs_reachable(adj, source)
            powered.add(source)
        return powered

    def load_supply_status(self, source_buses: Iterable[Node], t: float) -> Dict[Node, bool]:
        powered = self.energizable_buses(source_buses, t)
        return {bus: (bus in powered) for bus in self._loads}

    def total_served_load(self, source_buses: Iterable[Node], t: float) -> float:
        status = self.load_supply_status(source_buses, t)
        return sum(self._loads[bus] for bus, is_on in status.items() if is_on)

    def power_restorable(self, line_id: LineID, t: float, source_buses: Iterable[Node]) -> bool:
        """Check if repairing a currently-down line could restore additional load."""
        line = self._lines[line_id]
        if line.state_at(t) != LineState.DOWN:
            return False

        served_before = self.total_served_load(source_buses, t)

        # Simulate a temporary repair at the same time.
        original_events = list(line.events)
        self.set_line_state(line_id, t, LineState.REPAIRED)
        served_after = self.total_served_load(source_buses, t)
        line.events = original_events

        return served_after > served_before
