from sim.mock_sim import MockSimulator
from src.traffic_power_sim.dispatch.rl_env import TrafficPowerRepairEnv


def greedy_baseline(obs, mask):
    for idx in range(1, len(mask)):
        if mask[idx]:
            return idx
    return 0


def test_random_and_baseline_replay_smoke():
    env = TrafficPowerRepairEnv(sim=MockSimulator(seed=1), seed=1)
    obs = env.reset()
    assert "local_traffic" in obs

    random_traj = env.run_random_policy(max_steps=20)
    assert len(random_traj) > 0

    env.reset()
    baseline_logs = env.replay_baseline_policy(greedy_baseline, max_steps=20)
    assert len(baseline_logs) > 0
    assert all("mask" in step for step in baseline_logs)
