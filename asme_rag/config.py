"""Config loading.

Purpose: read config.toml into a plain dict and resolve paths.
Inputs: path to config.toml (default: ./config.toml). Outputs: dict plus helpers.
Data flow: used by every other module. No hidden global state: callers pass the dict.
How to change: add keys to config.toml; read them with get(). Nothing tunable is hard-coded.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path = "config.toml") -> dict[str, Any]:
    """Load config.toml and remember its folder under key '_root' for relative paths."""
    p = Path(path).resolve()
    with open(p, "rb") as fh:
        cfg = tomllib.load(fh)
    cfg["_root"] = str(p.parent)
    return cfg


def resolve_path(cfg: dict[str, Any], rel: str) -> Path:
    """Resolve a config path relative to the config file folder (absolute paths pass through)."""
    p = Path(rel)
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Read 'section.key' from the config dict, returning default if missing."""
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
