"""TypeSafe "Jev" System One client — typed decisions, not prose.

Jev answers typed questions about a state blob and returns probabilities. It is
used here *only* for routing and selection:

  * ``claimable``  - is this repo worth putting on a resume?
  * ``tier``       - which Groq tier should summarize it?

Two reasons this is worth a second provider. Jev's output tokens are free and
input is $0.042/Mtok, so scoring all ~100 repos costs fractions of a cent; and
it draws on a completely separate quota, leaving the scarce Groq 8K TPM budget
for work that actually has to produce text.

API shape verified live against https://api.typesafe.ai/v1/systemone:
a ``choice`` question **requires** a ``criteria`` map describing each option —
passing a bare ``options`` list returns HTTP 422.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Config, secret


class JevError(RuntimeError):
    pass


def noul(question: str, true_def: str, false_def: str) -> dict[str, Any]:
    """A yes/no question. The answer comes back as a probability in [0, 1]."""
    return {
        "type": "noul",
        "question": question,
        "criteria": {"true": true_def, "false": false_def},
    }


def choice(question: str, options: dict[str, str]) -> dict[str, Any]:
    """A pick-one question. ``options`` maps each option to its definition."""
    if not options:
        raise ValueError("choice questions need at least one option")
    return {"type": "choice", "question": question, "criteria": dict(options)}


def score(question: str, levels: list[str]) -> dict[str, Any]:
    """An ordered rating question. ``levels`` is ordered low -> high."""
    return {"type": "score", "question": question, "criteria": list(levels)}


@dataclass
class JevUsage:
    input_tokens: int = 0
    output_tokens: int = 0   # always free, per TypeSafe pricing
    calls: int = 0

    @property
    def cost_usd(self) -> float:
        return self.input_tokens / 1_000_000 * 0.042


@dataclass
class JevAnswers:
    raw: dict[str, Any] = field(default_factory=dict)

    def probability(self, name: str) -> float:
        a = self.raw.get(name) or {}
        return float(a.get("noul", 0.0))

    def choice(self, name: str) -> tuple[str, float]:
        a = self.raw.get(name) or {}
        return a.get("choice", ""), float(a.get("confidence", 0.0))

    def score(self, name: str) -> float:
        a = self.raw.get(name) or {}
        return float(a.get("score", 0.0))


class JevClient:
    def __init__(self, config: Config | None = None, api_key: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.config = config or Config.load()
        self.base_url = self.config.get("jev.base_url", "https://api.typesafe.ai")
        self.model = self.config.get("jev.model", "jev-latest")
        self.enabled = bool(self.config.get("jev.enabled", True))
        self._api_key = api_key if api_key is not None else secret("TYPESAFE_API_KEY", required=False)
        self._client = httpx.AsyncClient(timeout=90.0, transport=transport)
        self.usage = JevUsage()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    @property
    def available(self) -> bool:
        return self.enabled and bool(self._api_key)

    async def ask(self, state: Any, questions: dict[str, dict[str, Any]]) -> JevAnswers:
        if not self.available:
            raise JevError("Jev is disabled or TYPESAFE_API_KEY is unset")
        resp = await self._client.post(
            f"{self.base_url}/v1/systemone",
            headers={"Authorization": f"Bearer {self._api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "state": state, "questions": questions},
        )
        if resp.status_code == 422:
            raise JevError(
                f"Jev rejected the question shape (422). A `choice` question needs a "
                f"`criteria` map, not an `options` list. Detail: {resp.text[:300]}"
            )
        if resp.status_code >= 400:
            raise JevError(f"{resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        u = data.get("usage", {}) or {}
        self.usage.input_tokens += u.get("input_tokens", 0)
        self.usage.output_tokens += u.get("output_tokens", 0)
        self.usage.calls += 1
        return JevAnswers(data.get("answers", {}) or {})
