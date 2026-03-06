"""Core graph algorithms used by the traffic-power simulator."""

from __future__ import annotations

import heapq
from collections import defaultdict, deque
from typing import Callable, Dict, Hashable, Iterable, List, Optional, Set, Tuple

Node = Hashable
WeightFn = Callable[[Node, Node], Optional[float]]
Adjacency = Dict[Node, Iterable[Node]]


def dijkstra_dynamic(
    graph: Adjacency,
    source: Node,
    target: Optional[Node] = None,
    *,
    weight_fn: WeightFn,
) -> Tuple[Dict[Node, float], Dict[Node, Optional[Node]]]:
    """Dijkstra that supports dynamic edge weights.

    ``weight_fn(u, v)`` returns the current weight of edge ``u -> v``.
    Return ``None`` for unavailable edges (e.g., blocked roads).
    """
    dist: Dict[Node, float] = {source: 0.0}
    prev: Dict[Node, Optional[Node]] = {source: None}
    pq: List[Tuple[float, Node]] = [(0.0, source)]

    while pq:
        cur_dist, u = heapq.heappop(pq)
        if cur_dist > dist.get(u, float("inf")):
            continue
        if target is not None and u == target:
            break

        for v in graph.get(u, []):
            weight = weight_fn(u, v)
            if weight is None:
                continue
            cand = cur_dist + weight
            if cand < dist.get(v, float("inf")):
                dist[v] = cand
                prev[v] = u
                heapq.heappush(pq, (cand, v))

    return dist, prev


def bfs_reachable(graph: Adjacency, source: Node) -> Set[Node]:
    """Return nodes reachable from source with BFS."""
    seen: Set[Node] = {source}
    q: deque[Node] = deque([source])
    while q:
        u = q.popleft()
        for v in graph.get(u, []):
            if v in seen:
                continue
            seen.add(v)
            q.append(v)
    return seen


def dfs_component(graph: Adjacency, source: Node, seen: Optional[Set[Node]] = None) -> Set[Node]:
    """Return the connected component of ``source`` with DFS."""
    if seen is None:
        seen = set()
    comp: Set[Node] = set()

    def _dfs(u: Node) -> None:
        seen.add(u)
        comp.add(u)
        for v in graph.get(u, []):
            if v not in seen:
                _dfs(v)

    _dfs(source)
    return comp


def connected_components(graph: Adjacency) -> List[Set[Node]]:
    """Return connected components for an undirected-like adjacency map."""
    visited: Set[Node] = set()
    components: List[Set[Node]] = []
    nodes: Set[Node] = set(graph.keys())
    for neighs in graph.values():
        nodes.update(neighs)

    for node in nodes:
        if node in visited:
            continue
        comp = dfs_component(graph, node, visited)
        components.append(comp)
    return components


def find_bridges_undirected(graph: Adjacency) -> List[Tuple[Node, Node]]:
    """Find bridge edges in an undirected graph using Tarjan DFS."""
    timer = 0
    tin: Dict[Node, int] = {}
    low: Dict[Node, int] = {}
    visited: Set[Node] = set()
    bridges: List[Tuple[Node, Node]] = []

    undirected: Dict[Node, Set[Node]] = defaultdict(set)
    all_nodes: Set[Node] = set(graph.keys())
    for u, neighs in graph.items():
        for v in neighs:
            undirected[u].add(v)
            undirected[v].add(u)
            all_nodes.add(v)

    def dfs(u: Node, parent: Optional[Node]) -> None:
        nonlocal timer
        visited.add(u)
        timer += 1
        tin[u] = low[u] = timer

        for v in undirected[u]:
            if v == parent:
                continue
            if v in visited:
                low[u] = min(low[u], tin[v])
                continue

            dfs(v, u)
            low[u] = min(low[u], low[v])
            if low[v] > tin[u]:
                bridges.append((u, v))

    for node in all_nodes:
        if node not in visited:
            dfs(node, None)

    return bridges
