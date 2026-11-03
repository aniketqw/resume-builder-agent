"""Configuration loading. Model IDs and limits come from config.yaml, secrets from .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _project_root() -> Path:
    """Where `config.yaml` and `.env` live.

    ``parents[2]`` is correct for a source checkout and for ``pip install -e``,
    because `src/rba/config.py` sits two levels under the repo root. It is
    **wrong** for a normal wheel install, where the file lands in
    ``site-packages/rba/`` and ``parents[2]`` points somewhere outside the
    package entirely -- so a container that installs the wheel cannot find its
    own config. `RBA_ROOT` is the override that makes a clean multi-stage
    Docker build possible.
    """
    override = os.environ.get("RBA_ROOT")
    return Path(override).expanduser() if override else Path(__file__).resolve().parents[2]


def _default_config() -> Path:
    override = os.environ.get("RBA_CONFIG")
    return Path(override).expanduser() if override else _project_root() / "config.yaml"


# Kept as module attributes for backwards compatibility: they are read at
# import time, so anything setting RBA_ROOT must do so before importing rba.
PROJECT_ROOT = _project_root()
DEFAULT_CONFIG = _default_config()


@dataclass(frozen=True)
class Limits:
    rpm: int
    rpd: int
    tpm: int
    tpd: int


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        # Re-read the overrides on every load rather than trusting the values
        # captured at import: a server sets them in its lifespan, and tests set
        # them with monkeypatch after `rba.config` is already imported.
        load_dotenv(_project_root() / ".env", override=False)
        p = Path(path) if path else _default_config()
        with p.open() as fh:
            return cls(raw=yaml.safe_load(fh), path=p)

    # -- navigation -------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    # -- models -----------------------------------------------------------
    def model_for_tier(self, tier: str) -> str:
        tiers = self.get("groq.tiers", {})
        if tier not in tiers:
            raise KeyError(f"unknown tier {tier!r}; known: {sorted(tiers)}")
        return tiers[tier]

    def model_for_node(self, node: str) -> str:
        tier = self.get(f"routing.{node}")
        if tier is None:
            raise KeyError(f"no routing entry for node {node!r}")
        return self.model_for_tier(tier)

    def limits_for(self, model: str) -> Limits:
        table = self.get("groq.limits", {})
        entry = table.get(model) or table["default"]
        return Limits(**entry)

    @property
    def groq_defaults(self) -> dict[str, Any]:
        return dict(self.get("groq.defaults", {}))


# -- secrets --------------------------------------------------------------
REQUIRED_KEYS = {
    "GROQ_API_KEY": "Groq (summaries, resumes, planning)",
    "TYPESAFE_API_KEY": "TypeSafe/Jev (routing decisions)",
    "LANGSMITH_API_KEY": "LangSmith (tracing + evals)",
    "GITHUB_PERSONAL_ACCESS_TOKEN": "GitHub MCP server (read-only)",
}


def secret(name: str, required: bool = True) -> str:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    val = os.environ.get(name, "")
    if required and not val:
        raise RuntimeError(
            f"{name} is not set. Copy .env.sample to .env and fill it in "
            f"({REQUIRED_KEYS.get(name, 'see README')})."
        )
    return val


def enable_tracing() -> str | None:
    """Turn on LangSmith tracing if it is configured.

    Returns the project name when tracing is active, else None. Tracing is a
    diagnostic: a misconfigured LangSmith must never stop a run.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    if os.environ.get("LANGSMITH_TRACING", "").lower() not in ("1", "true", "yes"):
        return None
    if not os.environ.get("LANGSMITH_API_KEY"):
        return None
    project = os.environ.setdefault("LANGSMITH_PROJECT", "resume-builder-agent")
    # LangChain still reads the LANGCHAIN_* names in places; mirror them.
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    os.environ.setdefault("LANGCHAIN_API_KEY", os.environ["LANGSMITH_API_KEY"])
    return project


def missing_secrets() -> list[str]:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return [k for k in REQUIRED_KEYS if not os.environ.get(k)]
