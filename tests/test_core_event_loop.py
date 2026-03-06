from traffic_power_sim.sim import Action, CoreEventLoop, Event, EventType, SimState, TaskStatus


def test_event_loop_generates_logs_and_time_series():
    state = SimState(
        task_status={"t1": TaskStatus.PENDING, "t2": TaskStatus.PENDING},
        road_line_status={"l1": "damaged", "l2": "damaged"},
        team_position={"teamA": "base"},
    )
    loop = CoreEventLoop(state)
    loop.schedule(Event(time=0.0, priority=0, event_type=EventType.ARRIVAL, payload={"team_id": "teamA", "node": "n1"}))
    loop.schedule(Event(time=1.0, priority=0, event_type=EventType.START_JOB, payload={"task_id": "t1"}))
    loop.schedule(Event(time=2.0, priority=0, event_type=EventType.COMPLETE_JOB, payload={"task_id": "t1", "line_id": "l1"}))

    def strategy(sim_state, event):
        if event.event_type == EventType.COMPLETE_JOB:
            return [
                Action(
                    name="dispatch-next",
                    new_events=[
                        Event(
                            time=sim_state.now + 1,
                            priority=0,
                            event_type=EventType.START_JOB,
                            payload={"task_id": "t2"},
                        )
                    ],
                )
            ]
        return []

    result = loop.run(strategy)

    assert len(result.logs) == 4
    assert len(result.time_series) == 4
    assert result.final_state.task_status["t1"].value == "done"
    assert result.time_series[-1][1].lsd >= 0.5
