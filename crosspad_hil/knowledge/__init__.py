"""Firmware-coupled knowledge: boot markers, fatal patterns, SysEx catalog, CDC grammar.

The YAML files version with the firmware, not with the MCP server. ``load()`` is the only
accessor; it hands out deep copies of a cached parse so nobody can mutate shared state.
"""
from __future__ import annotations

import copy
from functools import cache
from importlib import resources
from typing import Any

import yaml

from crosspad_hil.errors import ENV, HilError

KNOWLEDGE_NAMES: tuple[str, ...] = ("markers", "sysex", "cdc")


@cache
def _load_cached(name: str) -> dict[str, Any]:
    path = resources.files("crosspad_hil.knowledge").joinpath(f"{name}.yaml")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise HilError(ENV, f"knowledge file {name}.yaml did not parse to a mapping",
                       name=name)
    return data


def load(name: str) -> dict[str, Any]:
    """Return a fresh copy of ``crosspad_hil/knowledge/<name>.yaml``."""
    if name not in KNOWLEDGE_NAMES:
        raise HilError(
            ENV,
            f"unknown knowledge file {name!r}",
            hint=f"one of {', '.join(KNOWLEDGE_NAMES)}",
            available=list(KNOWLEDGE_NAMES),
        )
    return copy.deepcopy(_load_cached(name))


__all__ = ["KNOWLEDGE_NAMES", "load"]
