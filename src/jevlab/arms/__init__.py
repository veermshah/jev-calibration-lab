"""Arm registry: builds `DecisionModel`s from configs/arms.toml."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from ..budget import Budget
from ..cache import ResponseCache
from ..pricing import PriceTable
from . import anthropic_arm, jev, openai_compat, synthetic
from .base import DecisionModel, RemoteDecisionModel
from .logreg import LogRegModel

__all__ = ["DecisionModel", "RemoteDecisionModel", "build_arms", "load_arm_config", "ConfigError"]


class ConfigError(RuntimeError):
    pass


def load_arm_config(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    arms: dict[str, dict[str, Any]] = raw.get("arms", {})
    return arms


def _key(cfg: dict[str, Any], name: str, required: bool) -> str | None:
    env = cfg.get("api_key_env")
    value = os.environ.get(env) if env else None
    if required and not value:
        raise ConfigError(
            f"arm {name!r} needs the {env} environment variable for a real run. "
            "Set it (see .env.example) or pass --dry-run to use synthetic replies."
        )
    return value


def build_arm(
    name: str,
    cfg: dict[str, Any],
    *,
    dry_run: bool,
    prices: PriceTable,
    cache: ResponseCache,
    budget: Budget,
    data_dir: Path,
) -> DecisionModel:
    kind = cfg.get("type")
    model = cfg.get("model", name)
    prices.get(model)  # fail early if the cost model has no entry
    concurrency = int(cfg.get("concurrency", 4))
    profile = synthetic.PROFILES[cfg.get("synthetic_profile", "generic")]
    common: dict[str, Any] = {
        "prices": prices,
        "cache": cache,
        "budget": budget,
        "synthetic": dry_run,
        "concurrency": concurrency,
    }
    if kind == "logreg":
        return LogRegModel(name, model, data_dir)
    if kind == "jev":
        transport = (
            synthetic.jev_transport(profile)
            if dry_run
            else jev.http_transport(
                cfg.get("base_url", "https://api.typesafe.ai/v1/systemone"),
                _key(cfg, name, required=True) or "",
                float(cfg.get("timeout_s", 30)),
            )
        )
        return jev.JevModel(name, model, transport, est_output_tokens=64, **common)
    if kind == "anthropic":
        transport = (
            synthetic.claude_transport(profile)
            if dry_run
            else anthropic_arm.sdk_transport(_key(cfg, name, required=True))
        )
        return anthropic_arm.ClaudeModel(
            name,
            model,
            transport,
            effort=str(cfg.get("effort", "low")),
            max_tokens=int(cfg.get("max_tokens", 2048)),
            **common,
        )
    if kind == "openai_compatible":
        if "base_url" not in cfg:
            raise ConfigError(f"arm {name!r} needs base_url")
        transport = (
            synthetic.chat_transport(profile)
            if dry_run
            else openai_compat.http_transport(cfg["base_url"], _key(cfg, name, required=False))
        )
        return openai_compat.ChatCompletionsModel(
            name, model, transport, max_tokens=int(cfg.get("max_tokens", 512)), **common
        )
    raise ConfigError(f"arm {name!r} has unknown type {kind!r}")


def build_arms(
    names: list[str],
    config: dict[str, dict[str, Any]],
    **kwargs: Any,
) -> list[DecisionModel]:
    missing = [n for n in names if n not in config]
    if missing:
        raise ConfigError(f"unknown arms {missing}; defined: {sorted(config)}")
    return [build_arm(n, config[n], **kwargs) for n in names]
