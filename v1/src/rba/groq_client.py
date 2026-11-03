"""Groq chat client: tier routing, rate limiting, strict JSON, caching.

Two non-obvious behaviours this wraps, both found by probing the live API:

1. **The reasoning trap.** ``openai/gpt-oss-*`` models spend the completion
   budget on an internal ``reasoning`` field. With a small ``max_tokens`` they
   return ``content == ""`` and a populated ``reasoning`` — a 200 OK that looks
   like a model failure. Every request therefore forces ``reasoning_effort:
   "low"`` and a floor on ``max_tokens``.

2. **Structured output excludes tools and streaming.** A request may use
   ``response_format: json_schema`` *or* tool calling, never both. Callers that
   need tools must use a different node.
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
from .config import Config, enable_tracing, secret
from .limiter import Limiter, estimate_tokens
from .schema import response_format

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MIN_MAX_TOKENS = 800
"""Floor on max_tokens. Below roughly this, reasoning starves `content`."""

# Models where langchain-groq / Groq honour strict constrained decoding.
STRICT_CAPABLE = {"openai/gpt-oss-20b", "openai/gpt-oss-120b", "openai/gpt-oss-safeguard-20b"}


def _make_tracer():
    """A LangSmith client, or None when tracing is off or unavailable."""
    if enable_tracing() is None:
        return None
    try:
        from langsmith import Client

        return Client()
    except Exception:
        return None


class GroqError(RuntimeError):
    pass


class EmptyContentError(GroqError):
    """200 OK but `content` was empty — almost always the reasoning trap."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached: bool = False
    # Requests that actually hit the Groq API. This is the quantity the 1,000
    # requests/day limit counts -- retries included, local cache hits excluded.
    requests: int = 0
    # Prompt tokens Groq served from its own prefix cache. Groq's docs say these
    # are exempt from rate limits, so the billable-against-quota prompt total is
    # `prompt_tokens - cached_tokens`.
    cached_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
            self.cached or other.cached,
            self.requests + other.requests,
            self.cached_tokens + other.cached_tokens,
        )


@dataclass
class Result:
    content: str
    model: str
    usage: Usage
    raw: dict[str, Any]

    def parsed(self) -> Any:
        return json.loads(self.content)


class GroqClient:
    def __init__(self, config: Config | None = None, limiter: Limiter | None = None,
                 cache: Cache | None = None, api_key: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.config = config or Config.load()
        self.limiter = limiter or Limiter(self.config)
        self.cache = cache if cache is not None else Cache()
        self._api_key = api_key if api_key is not None else secret("GROQ_API_KEY")
        self.base_url = self.config.get("groq.base_url")
        self.max_retries = self.config.get("groq.max_retries", 5)
        self.margin = self.config.get("groq.token_estimate_margin", 0.10)
        self._client = httpx.AsyncClient(timeout=180.0, transport=transport)
        self.usage_by_node: dict[str, Usage] = {}
        self._tracer = _make_tracer()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    # -- request construction --------------------------------------------
    def _build(self, model: str, messages: list[dict[str, str]],
               schema: type[BaseModel] | None, **overrides) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "messages": messages}
        body.update(self.config.groq_defaults)
        body.update(overrides)
        # Guard the reasoning trap: never let max_tokens starve `content`.
        body["max_tokens"] = max(int(body.get("max_tokens", MIN_MAX_TOKENS)), MIN_MAX_TOKENS)
        if schema is not None:
            if model in STRICT_CAPABLE:
                body["response_format"] = response_format(schema)
            else:
                # qwen and friends: json_object + Pydantic validation + retry.
                body["response_format"] = {"type": "json_object"}
        if "tools" in body and "response_format" in body:
            raise GroqError(
                "Groq cannot combine structured outputs with tool use in one call. "
                "Split these into separate graph nodes."
            )
        return body

    # -- core call --------------------------------------------------------
    async def complete(self, messages: list[dict[str, str]], *, node: str | None = None,
                       tier: str | None = None, model: str | None = None,
                       schema: type[BaseModel] | None = None,
                       cache_key: str | None = None, **overrides) -> Result:
        if model is None:
            model = self.config.model_for_tier(tier) if tier else self.config.model_for_node(node or "")
        body = self._build(model, messages, schema, **overrides)

        if cache_key and self.cache:
            hit = self.cache.get(cache_key, model)
            if hit is not None:
                usage = Usage(cached=True)
                self._record(node, usage)
                return Result(hit, model, usage, {"cached": True})

        prompt_text = "".join(m.get("content", "") for m in messages)
        est = estimate_tokens(prompt_text, self.margin) + int(body["max_tokens"] * 0.5)
        bucket = self.limiter.bucket(model)

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            await bucket.reserve(est)
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}",
                             "Content-Type": "application/json"},
                    json=body,
                )
            except httpx.HTTPError as exc:          # network flake
                last_exc = exc
                import asyncio
                await asyncio.sleep(bucket.retry_delay(attempt))
                continue

            headers = dict(resp.headers)
            if resp.status_code == 429:
                delay = bucket.retry_delay(attempt, headers)
                log.warning("429 on %s, sleeping %.1fs (attempt %d)", model, delay, attempt + 1)
                bucket.reconcile(est, 0, headers)
                import asyncio
                await asyncio.sleep(delay)
                continue
            if resp.status_code >= 500:
                last_exc = GroqError(f"{resp.status_code}: {resp.text[:300]}")
                import asyncio
                await asyncio.sleep(bucket.retry_delay(attempt))
                continue
            if resp.status_code >= 400:
                bucket.reconcile(est, 0, headers)
                body_text = resp.text
                # Strict-schema generation that runs out of completion budget
                # comes back as a 400, not a truncated 200. Retrying with a
                # larger budget is the documented fix, so do that rather than
                # surfacing it as a hard failure.
                if ("json_validate_failed" in body_text
                        or "max completion tokens reached" in body_text):
                    if body["max_tokens"] < 8192:
                        body["max_tokens"] = min(int(body["max_tokens"] * 2), 8192)
                        log.info("schema truncated on %s; retrying with max_tokens=%d",
                                 model, body["max_tokens"])
                        continue
                raise GroqError(f"{resp.status_code}: {body_text[:500]}")

            data = resp.json()
            u = data.get("usage", {}) or {}
            usage = Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
                reasoning_tokens=(u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                requests=1,
                cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            )
            bucket.reconcile(est, usage.total_tokens, headers)
            self._record(node, usage)

            msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
            content = (msg.get("content") or "").strip()
            if not content:
                # The reasoning trap. Raise the budget and try once more.
                last_exc = EmptyContentError(
                    f"{model} returned empty content with "
                    f"{usage.reasoning_tokens} reasoning tokens; "
                    f"max_tokens={body['max_tokens']}"
                )
                body["max_tokens"] = min(int(body["max_tokens"] * 2), 8192)
                body["reasoning_effort"] = "low"
                continue

            result = Result(content, model, usage, data)
            if cache_key and self.cache:
                self.cache.put(cache_key, model, content)
            self._trace(node, model, messages, result)
            return result

        raise last_exc or GroqError(f"exhausted {self.max_retries} attempts on {model}")

    # -- typed call -------------------------------------------------------
    async def structured(self, messages: list[dict[str, str]], schema: type[T], *,
                         node: str | None = None, tier: str | None = None,
                         model: str | None = None, cache_key: str | None = None,
                         **overrides) -> tuple[T, Usage]:
        """Return a validated Pydantic object. Retries once on a validation failure."""
        last: Exception | None = None
        total = Usage()
        for attempt in range(2):
            res = await self.complete(messages, node=node, tier=tier, model=model,
                                      schema=schema, cache_key=cache_key if attempt == 0 else None,
                                      **overrides)
            total = total + res.usage
            try:
                return schema.model_validate_json(res.content), total
            except (ValidationError, json.JSONDecodeError) as exc:
                last = exc
                messages = [
                    *messages,
                    {"role": "assistant", "content": res.content},
                    {"role": "user",
                     "content": f"That did not match the schema: {exc}. "
                                "Return only valid JSON matching the schema."},
                ]
        raise GroqError(f"schema validation failed after retry: {last}")

    def _trace(self, node: str | None, model: str,
               messages: list[dict[str, str]], result: Result) -> None:
        if self._tracer is None:
            return
        try:
            self._tracer.create_run(
                name=node or "groq",
                run_type="llm",
                inputs={"messages": messages, "model": model},
                outputs={"content": result.content,
                         "usage": {"prompt": result.usage.prompt_tokens,
                                   "completion": result.usage.completion_tokens,
                                   "total": result.usage.total_tokens}},
                project_name=os.environ.get("LANGSMITH_PROJECT"),
            )
        except Exception as exc:  # tracing must never break a run
            log.debug("langsmith trace failed: %s", exc)

    def _record(self, node: str | None, usage: Usage) -> None:
        key = node or "unattributed"
        self.usage_by_node[key] = self.usage_by_node.get(key, Usage()) + usage

    def usage_report(self) -> dict[str, dict[str, int]]:
        return {k: {"prompt": v.prompt_tokens, "completion": v.completion_tokens,
                    "total": v.total_tokens, "reasoning": v.reasoning_tokens,
                    "requests": v.requests, "cached_tokens": v.cached_tokens}
                for k, v in self.usage_by_node.items()}

    def usage_total(self) -> Usage:
        """Every node summed -- the quantities the daily quotas are spent in."""
        out = Usage()
        for v in self.usage_by_node.values():
            out = out + v
        return out
