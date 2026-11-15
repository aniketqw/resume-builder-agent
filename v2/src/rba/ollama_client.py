"""A self-hosted alternative to Groq, with the same structured-output shape.

The only method anything in `nodes/` ever calls on `GroqClient` is
`structured(messages, schema, ...) -> (T, Usage)` (confirmed: `grep -rn
"client\\.structured\\(" src/rba/` turns up every caller, and none call
`complete()` directly) -- so this client only has to match that one shape.
Nothing in `nodes/summarize.py`/`nodes/build.py`/`nodes/plan.py` needs to
change; `graph.py`'s `Runner` is the only place that picks which client to
build.

No token-bucket rate limiter here, unlike `GroqClient` -- a self-hosted
Ollama instance has no quota to protect. That is the whole point of this
client existing: Groq's 8,000 TPM free-tier limit was a repeated, measured
bottleneck (see `trace.md`), and a local model removes it entirely.

Verified directly against a real Ollama 0.20.2 instance before writing
this: `POST /api/chat` with `format` set to a full JSON Schema (not just
`"json"`) returns strictly schema-conforming JSON, and the response body
carries `prompt_eval_count`/`eval_count` as the token-count equivalents.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .cache import Cache
from .config import Config
from .groq_client import GroqError, Usage

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_TIMEOUT = 180.0


@dataclass
class _Result:
    content: str
    model: str
    usage: Usage


class OllamaClient:
    """Structural twin of `GroqClient`'s `structured()` -- see the module
    docstring for why that is the only method that needs to match."""

    def __init__(self, config: Config | None = None, base_url: str | None = None,
                 model: str | None = None, cache: Cache | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.config = config or Config.load()
        # Env override beats config.yaml beats the hardcoded fallback --
        # the per-environment knob the config.yaml comment promises.
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL")
                         or self.config.get("ollama.base_url")
                         or "http://localhost:11434").rstrip("/")
        self.model = (model or os.environ.get("OLLAMA_MODEL")
                     or self.config.get("ollama.model", "qwen2.5:14b-instruct"))
        self.cache = cache if cache is not None else Cache()
        self._client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, transport=transport)
        self.usage_by_node: dict[str, Usage] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    # -- the one method anything in nodes/ actually calls ------------------
    async def structured(self, messages: list[dict[str, str]], schema: type[T], *,
                         node: str | None = None, tier: str | None = None,
                         model: str | None = None, cache_key: str | None = None,
                         **overrides) -> tuple[T, Usage]:
        """Return a validated Pydantic object. Retries once on a schema
        violation, the same repair `GroqClient.structured` uses -- a local
        model is more reliable than Groq's small tier (measured: clean,
        correctly-shaped JSON on the first attempt in every smoke test run
        so far), but "more reliable" is not "never wrong"."""
        use_model = model or self.model
        last: Exception | None = None
        total = Usage()
        for attempt in range(2):
            if cache_key and self.cache and attempt == 0:
                hit = self.cache.get(cache_key, use_model)
                if hit is not None:
                    usage = Usage(cached=True)
                    self._record(node, usage)
                    try:
                        return schema.model_validate_json(hit), usage
                    except (ValidationError, json.JSONDecodeError):
                        pass  # a stale/bad cache entry falls through to a real call

            result = await self._call(messages, schema, use_model, **overrides)
            total = total + result.usage
            self._record(node, result.usage)
            try:
                parsed = schema.model_validate_json(result.content)
                if cache_key and self.cache:
                    self.cache.put(cache_key, use_model, result.content)
                return parsed, total
            except (ValidationError, json.JSONDecodeError) as exc:
                last = exc
                messages = [
                    *messages,
                    {"role": "assistant", "content": result.content},
                    {"role": "user",
                     "content": f"That did not match the schema: {exc}. "
                                "Return only valid JSON matching the schema."},
                ]
        raise GroqError(f"schema validation failed after retry: {last}")

    async def _call(self, messages: list[dict[str, str]], schema: type[BaseModel],
                    model: str, **overrides) -> _Result:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "format": schema.model_json_schema(),
            "stream": False,
        }
        body.update(self.config.get("ollama.defaults", {}))
        body.update(overrides)

        resp = await self._client.post(f"{self.base_url}/api/chat", json=body)
        if resp.status_code >= 400:
            raise GroqError(f"ollama {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        content = ((data.get("message") or {}).get("content") or "").strip()
        if not content:
            raise GroqError(f"ollama returned no content: {repr(data)[:500]}")

        usage = Usage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            requests=1,
        )
        return _Result(content=content, model=model, usage=usage)

    def _record(self, node: str | None, usage: Usage) -> None:
        key = node or "unattributed"
        self.usage_by_node[key] = self.usage_by_node.get(key, Usage()) + usage

    def usage_report(self) -> dict[str, dict[str, int]]:
        return {k: {"prompt": v.prompt_tokens, "completion": v.completion_tokens,
                    "total": v.total_tokens, "reasoning": v.reasoning_tokens,
                    "requests": v.requests, "cached_tokens": v.cached_tokens}
                for k, v in self.usage_by_node.items()}

    def usage_total(self) -> Usage:
        out = Usage()
        for v in self.usage_by_node.values():
            out = out + v
        return out
