from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import heapq
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class EventType(str, Enum):
    """离散事件模拟中的核心事件类型。"""

    ARRIVAL = "arrival"  # 到达
    START_JOB = "start_job"  # 开始作业
    COMPLETE_JOB = "complete_job"  # 作业完成
    RETURN_RESUPPLY = "return_resupply"  # 返航补给
    WEATHER_WINDOW_CHANGE = "weather_window_change"  # 天气窗变化
    DAY_NIGHT_SWITCH = "day_night_switch"  # 昼夜切换


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"


@dataclass(order=True)
class Event:
    """事件队列中的条目。"""

    time: float
    priority: int
    event_type: EventType = field(compare=False)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass
class SimState:
    """系统状态：队伍、任务、道路/线路、天气窗、效率。"""

    now: float = 0.0
    team_position: Dict[str, str] = field(default_factory=dict)
    task_status: Dict[str, TaskStatus] = field(default_factory=dict)
    road_line_status: Dict[str, str] = field(default_factory=dict)
    weather_window: str = "open"  # Ω(t)
    alpha: float = 1.0  # α(t)
    is_daytime: bool = True


@dataclass
class Action:
    """策略返回动作。可携带状态更新和后续事件。"""

    name: str
    state_updates: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    new_events: Sequence[Event] = field(default_factory=list)


@dataclass
class MetricSnapshot:
    """用于阶梯恢复曲线输出的指标。"""

    lsd: float
    pk: float
    ak: float


@dataclass
class LogEntry:
    timestamp: float
    event_type: EventType
    action_names: List[str]
    state_changes: Dict[str, Any]
    metrics: MetricSnapshot


@dataclass
class SimulationResult:
    logs: List[LogEntry]
    time_series: List[Tuple[float, MetricSnapshot]]
    final_state: SimState


StrategyHook = Callable[[SimState, Event], Sequence[Action]]


class CoreEventLoop:
    """交通-电力抢修核心事件循环。"""

    def __init__(self, initial_state: Optional[SimState] = None):
        self.state = initial_state or SimState()
        self._queue: List[Event] = []
        self.logs: List[LogEntry] = []
        self.time_series: List[Tuple[float, MetricSnapshot]] = []

    def schedule(self, event: Event) -> None:
        heapq.heappush(self._queue, event)

    def run(self, strategy_hook: StrategyHook, until: Optional[float] = None) -> SimulationResult:
        while self._queue:
            event = heapq.heappop(self._queue)
            if until is not None and event.time > until:
                break

            self.state.now = event.time
            state_changes = self._apply_event(event)

            actions = list(strategy_hook(self.state, event) or [])
            action_changes = self._apply_actions(actions)
            if action_changes:
                state_changes["actions"] = action_changes

            metrics = self._snapshot_metrics()
            self.time_series.append((self.state.now, metrics))
            self.logs.append(
                LogEntry(
                    timestamp=self.state.now,
                    event_type=event.event_type,
                    action_names=[a.name for a in actions],
                    state_changes=state_changes,
                    metrics=metrics,
                )
            )

        return SimulationResult(
            logs=self.logs,
            time_series=self.time_series,
            final_state=self.state,
        )

    def _apply_event(self, event: Event) -> Dict[str, Any]:
        changes: Dict[str, Any] = {}
        payload = event.payload

        if event.event_type == EventType.ARRIVAL:
            team_id = payload.get("team_id")
            node = payload.get("node")
            if team_id and node:
                before = self.state.team_position.get(team_id)
                self.state.team_position[team_id] = node
                changes["team_position"] = {"team_id": team_id, "before": before, "after": node}

        elif event.event_type == EventType.START_JOB:
            task_id = payload.get("task_id")
            if task_id:
                before = self.state.task_status.get(task_id, TaskStatus.PENDING)
                self.state.task_status[task_id] = TaskStatus.IN_PROGRESS
                changes["task_status"] = {"task_id": task_id, "before": before, "after": TaskStatus.IN_PROGRESS}

        elif event.event_type == EventType.COMPLETE_JOB:
            task_id = payload.get("task_id")
            line_id = payload.get("line_id")
            if task_id:
                before = self.state.task_status.get(task_id, TaskStatus.PENDING)
                self.state.task_status[task_id] = TaskStatus.DONE
                changes["task_status"] = {"task_id": task_id, "before": before, "after": TaskStatus.DONE}
            if line_id:
                before_line = self.state.road_line_status.get(line_id, "damaged")
                self.state.road_line_status[line_id] = "recovered"
                changes["road_line_status"] = {
                    "line_id": line_id,
                    "before": before_line,
                    "after": "recovered",
                }

        elif event.event_type == EventType.RETURN_RESUPPLY:
            team_id = payload.get("team_id")
            depot = payload.get("depot", "base")
            if team_id:
                before = self.state.team_position.get(team_id)
                self.state.team_position[team_id] = depot
                changes["team_position"] = {"team_id": team_id, "before": before, "after": depot}

        elif event.event_type == EventType.WEATHER_WINDOW_CHANGE:
            before = self.state.weather_window
            weather = payload.get("weather_window", before)
            alpha = payload.get("alpha")
            self.state.weather_window = weather
            changes["weather_window"] = {"before": before, "after": weather}
            if alpha is not None:
                alpha_before = self.state.alpha
                self.state.alpha = float(alpha)
                changes["alpha"] = {"before": alpha_before, "after": self.state.alpha}

        elif event.event_type == EventType.DAY_NIGHT_SWITCH:
            before = self.state.is_daytime
            is_daytime = bool(payload.get("is_daytime", not before))
            self.state.is_daytime = is_daytime
            if "alpha" in payload:
                alpha_before = self.state.alpha
                self.state.alpha = float(payload["alpha"])
                changes["alpha"] = {"before": alpha_before, "after": self.state.alpha}
            changes["day_night"] = {"before": before, "after": is_daytime}

        return changes

    def _apply_actions(self, actions: Sequence[Action]) -> Dict[str, Any]:
        if not actions:
            return {}

        action_changes: Dict[str, Any] = {}
        for action in actions:
            team_updates = action.state_updates.get("team_position", {})
            if team_updates:
                for team_id, node in team_updates.items():
                    self.state.team_position[team_id] = node
                action_changes.setdefault("team_position", {}).update(team_updates)

            task_updates = action.state_updates.get("task_status", {})
            if task_updates:
                for task_id, status in task_updates.items():
                    self.state.task_status[task_id] = TaskStatus(status)
                action_changes.setdefault("task_status", {}).update(task_updates)

            line_updates = action.state_updates.get("road_line_status", {})
            if line_updates:
                self.state.road_line_status.update(line_updates)
                action_changes.setdefault("road_line_status", {}).update(line_updates)

            if "weather_window" in action.state_updates:
                new_weather = action.state_updates["weather_window"].get("value")
                if new_weather:
                    self.state.weather_window = new_weather
                    action_changes["weather_window"] = new_weather

            if "alpha" in action.state_updates:
                new_alpha = action.state_updates["alpha"].get("value")
                if new_alpha is not None:
                    self.state.alpha = float(new_alpha)
                    action_changes["alpha"] = self.state.alpha

            for next_event in action.new_events:
                self.schedule(next_event)

        return action_changes

    def _snapshot_metrics(self) -> MetricSnapshot:
        total_tasks = max(len(self.state.task_status), 1)
        completed_tasks = sum(1 for s in self.state.task_status.values() if s == TaskStatus.DONE)
        lsd = completed_tasks / total_tasks

        total_lines = max(len(self.state.road_line_status), 1)
        healthy_lines = sum(1 for s in self.state.road_line_status.values() if s == "recovered")
        pk = healthy_lines / total_lines

        daylight_factor = 1.0 if self.state.is_daytime else 0.85
        weather_factor = 1.0 if self.state.weather_window == "open" else 0.7
        ak = self.state.alpha * daylight_factor * weather_factor

        return MetricSnapshot(lsd=lsd, pk=pk, ak=ak)
