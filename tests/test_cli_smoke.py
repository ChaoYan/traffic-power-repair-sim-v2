from traffic_power_sim.bootstrap import initialize_modules
from traffic_power_sim.config.settings import load_scenario


def test_load_demo_scenario():
    scenario = load_scenario("scenarios/demo.yaml")
    assert scenario["scenario_name"] == "demo"


def test_initialize_modules_has_all_entries():
    messages = initialize_modules()
    assert len(messages) == 7
