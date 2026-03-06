# AGENTS.md

## Cursor Cloud specific instructions

This is a pure-Python CLI simulation tool (no web UI, no database, no Docker). Python >= 3.10 is required.

### Key commands

| Action | Command |
|---|---|
| Run tests | `python3 -m pytest tests/ -v` |
| Run CLI | `PYTHONPATH=src python3 -m traffic_power_sim.cli --scenario scenarios/demo.yaml` |
| Run minimal case | `PYTHONPATH=src python3 scripts/run_minimal_case.py` |
| Lint | `python3 -m ruff check src/ tests/ scripts/ sim/` |

### Caveats

- `pandas` is an **undeclared runtime dependency** (used in `src/traffic_power_sim/metrics/core.py`). The update script installs it explicitly; if it goes missing, `evaluate_strategy()` and `evaluate_and_export()` will raise `ImportError`.
- The root-level `sim/` directory (mock simulator adapter) is separate from `src/traffic_power_sim/sim/` (core event loop). Test `test_rl_env_smoke.py` imports from both via `conftest.py` adding the repo root to `sys.path`.
- `pyproject.toml` sets `pythonpath = ["src"]` for pytest, so `python3 -m pytest` from the repo root works out of the box. Running the CLI or scripts outside pytest requires `PYTHONPATH=src`.
- There are 2 pre-existing ruff lint warnings (unused imports in `sim/mock_sim.py` and `src/traffic_power_sim/graph/power_graph.py`); these are in the existing codebase.
