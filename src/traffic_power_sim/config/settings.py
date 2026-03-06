"""Scenario configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    """Raised when scenario configuration is invalid."""


def load_scenario(path: str | Path) -> dict[str, Any]:
    """Load scenario yaml and perform lightweight validation."""
    scenario_path = Path(path)
    if not scenario_path.exists():
        raise ConfigError(f"Scenario file not found: {scenario_path}")

    with scenario_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    if not isinstance(payload, dict):
        raise ConfigError("Scenario root must be a mapping object")

    payload.setdefault("scenario_name", scenario_path.stem)
    payload.setdefault("time_step_minutes", 5)
    return payload
