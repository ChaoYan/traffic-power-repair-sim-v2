"""Command-line entrypoint for dry-run simulation startup."""

from __future__ import annotations

import argparse

from traffic_power_sim.bootstrap import initialize_modules
from traffic_power_sim.config.settings import load_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Traffic-power repair simulation")
    parser.add_argument(
        "--scenario",
        required=True,
        help="Path to scenario YAML file",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scenario = load_scenario(args.scenario)
    print(f"Loaded scenario: {scenario['scenario_name']}")
    print(f"Time step (minutes): {scenario['time_step_minutes']}")

    print("Module initialization summary:")
    for message in initialize_modules():
        print(f"- {message}")


if __name__ == "__main__":
    main()
