"""Bootstrap routines for module initialization."""

from __future__ import annotations

MODULES = [
    "config",
    "data",
    "graph",
    "sim",
    "dispatch",
    "metrics",
    "air",
]


def initialize_modules() -> list[str]:
    """Return initialization messages for each major module."""
    return [f"[{name}] initialized" for name in MODULES]
