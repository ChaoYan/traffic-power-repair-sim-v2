from __future__ import annotations

import csv
import json
from pathlib import Path

from traffic_power_sim.dispatch.baselines import run_all_strategies


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "scenarios" / "minimal_case" / "scenario.json"
OUT_DIR = ROOT / "scenarios" / "minimal_case" / "results"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenario = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))

    results = run_all_strategies(scenario)

    actions_path = OUT_DIR / "actions.json"
    actions_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    curve_rows = []
    summary_rows = []

    for strategy, payload in results.items():
        curve = payload["recovery_curve"]
        curve_rows.extend(curve)
        final = curve[-1]
        summary_rows.append(
            {
                "strategy": strategy,
                "final_step": payload["final_step"],
                "repaired_nodes": final["repaired_nodes"],
                "total_nodes": final["total_nodes"],
                "final_recovery_ratio": final["recovery_ratio"],
            }
        )

    with (OUT_DIR / "recovery_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "step", "repaired_nodes", "total_nodes", "recovery_ratio"],
        )
        writer.writeheader()
        writer.writerows(curve_rows)

    with (OUT_DIR / "strategy_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "final_step", "repaired_nodes", "total_nodes", "final_recovery_ratio"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
