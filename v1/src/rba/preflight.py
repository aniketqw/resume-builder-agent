"""Check every credential once, before a run spends anything.

This exists because of a real failure. A Groq key typed into the UI silently
overrode a perfectly good one in `.env`, and the run did not say so. It went on
to fetch repositories, served several of them from the response cache without
touching the network at all, and only then surfaced the problem as:

    summarize failed for vd_p2: 401: Invalid API Key

One repository named, partway through, after minutes of work -- with the cached
repositories making it look as though the key worked for everything else. The
same run reported `Jev unavailable; using the heuristic fallback`, which is the
same class of fault wearing different clothes: a credential that is absent or
wrong, discovered late and reported somewhere nobody is looking.

So: one cheap call per service, before anything else, naming the service, the
verdict and **which source the key came from**. The last part matters most --
silently preferring a typed key over a working `.env` is what caused this.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx

from .config import PROJECT_ROOT, Config

# Each service: the endpoint that proves a key works, and what it costs us.
# All of these are free metadata reads -- no tokens, no rate-limit pressure.
TIMEOUT = 12.0


@dataclass(frozen=True)
class Check:
    service: str
    ok: bool
    source: str                  # "argument" | ".env" | "environment" | "absent"
    detail: str = ""

    @property
    def icon(self) -> str:
        return "ok" if self.ok else "failed"

    def line(self) -> str:
        where = f" (from {self.source})" if self.source != "absent" else ""
        return f"{self.service}: {self.icon}{where}" + (
            f" — {self.detail}" if self.detail else "")


def _resolve(name: str, override: str | None) -> tuple[str, str]:
    """The key actually used, and an honest label for where it came from."""
    if override:
        return override, "argument"
    from dotenv import dotenv_values

    env_file = dotenv_values(PROJECT_ROOT / ".env")
    if os.environ.get(name) and os.environ.get(name) != env_file.get(name):
        return os.environ[name], "environment"
    if env_file.get(name):
        return env_file[name] or "", ".env"
    if os.environ.get(name):
        return os.environ[name], "environment"
    return "", "absent"


async def _probe(client: httpx.AsyncClient, service: str, name: str,
                 url: str, override: str | None,
                 required: bool = True) -> Check:
    key, source = _resolve(name, override)
    if not key:
        return Check(service, not required, "absent",
                     f"{name} is not set" if required else
                     f"{name} is not set; this step will be skipped")
    try:
        resp = await client.get(url, headers={"Authorization": f"Bearer {key}"},
                                timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        return Check(service, False, source, f"could not reach {url}: {exc}")

    if resp.status_code == 200:
        return Check(service, True, source)
    if resp.status_code in (401, 403):
        return Check(service, False, source,
                     f"the key was rejected ({resp.status_code}). This key "
                     f"overrides any other source, so a stale value here will "
                     f"mask a working one.")
    if resp.status_code == 429:
        # Authentication clearly worked, or we would have had a 401.
        return Check(service, True, source,
                     "rate-limited right now, but the key is valid")
    return Check(service, False, source, f"unexpected {resp.status_code}")


async def check_credentials(*, groq_key: str | None = None,
                            typesafe_key: str | None = None,
                            github_token: str | None = None,
                            config: Config | None = None) -> list[Check]:
    """Validate every credential a run needs. Never raises; returns verdicts."""
    cfg = config or Config.load()
    groq_base = cfg.get("groq.base_url", "https://api.groq.com/openai/v1")
    typesafe_base = os.environ.get("TYPESAFE_BASE_URL") or \
        cfg.get("jev.base_url", "https://api.typesafe.ai")

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            _probe(client, "Groq", "GROQ_API_KEY",
                   f"{groq_base}/models", groq_key),
            _probe(client, "TypeSafe/Jev", "TYPESAFE_API_KEY",
                   f"{typesafe_base}/v1/models", typesafe_key, required=False),
            _github(client, github_token),
        )
    return list(results)


async def _github(client: httpx.AsyncClient, override: str | None) -> Check:
    """GitHub uses its own header scheme and is usable without a token."""
    key, source = _resolve("GITHUB_PERSONAL_ACCESS_TOKEN", override)
    if not key:
        return Check("GitHub", True, "absent",
                     "no token: GitHub allows 60 requests an hour, shared "
                     "between everyone on this machine's IP")
    try:
        resp = await client.get("https://api.github.com/user",
                                headers={"Authorization": f"Bearer {key}",
                                         "Accept": "application/vnd.github+json"},
                                timeout=TIMEOUT)
    except httpx.HTTPError as exc:
        return Check("GitHub", False, source, f"could not reach GitHub: {exc}")
    if resp.status_code == 200:
        login = (resp.json() or {}).get("login", "")
        remaining = resp.headers.get("x-ratelimit-remaining", "?")
        return Check("GitHub", True, source,
                     f"authenticated as {login}; {remaining} requests left this hour")
    if resp.status_code == 401:
        return Check("GitHub", False, source,
                     "the token was rejected — it may have expired")
    return Check("GitHub", False, source, f"unexpected {resp.status_code}")


def blocking_failures(checks: list[Check]) -> list[Check]:
    """Checks that should stop a run rather than merely warn it."""
    return [c for c in checks if not c.ok and c.service == "Groq"]


def summary(checks: list[Check]) -> str:
    return "\n".join(c.line() for c in checks)
