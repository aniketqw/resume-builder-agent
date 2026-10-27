"""Configuration loading. Model IDs and limits come from config.yaml, secrets from .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


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
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        p = Path(path) if path else DEFAULT_CONFIG
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
