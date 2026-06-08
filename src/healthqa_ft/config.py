from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def route_config(cfg: dict[str, Any], route_name: str) -> dict[str, Any]:
    if route_name not in cfg.get("routes", {}):
        known = ", ".join(sorted(cfg.get("routes", {})))
        raise KeyError(f"Unknown route '{route_name}'. Known routes: {known}")

    route = deepcopy(cfg["routes"][route_name])
    train_cfg = deep_update(cfg.get("training_defaults", {}), route.get("training", {}))
    route["training"] = train_cfg
    route["route_name"] = route_name
    return route


def routes_to_train(cfg: dict[str, Any], selected: list[str] | None = None) -> list[str]:
    all_routes = list(cfg.get("routes", {}).keys())
    if not selected:
        return all_routes
    unknown = [r for r in selected if r not in all_routes]
    if unknown:
        raise KeyError(f"Unknown routes: {unknown}. Known routes: {all_routes}")
    return selected

