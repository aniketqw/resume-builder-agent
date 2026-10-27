"""Offline test doubles. No network, no API key needed."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx


def fake_secret(prefix: str, length: int = 36) -> str:
    """Build a credential-shaped string at runtime.

    The literal must never appear in source: scripts/scan_secrets.sh scans this
    repository too, and a realistic-looking token in a fixture is exactly what
    it is supposed to catch. Characters vary so the result does not trip the
    scanner's placeholder rule ("xxxxxxxx" reads as documentation, not a key).
    """
    import string

    alphabet = string.ascii_letters + string.digits
    body = "".join(alphabet[(i * 7 + 13) % len(alphabet)] for i in range(length))
    return prefix + body


RATE_HEADERS = {
    "x-ratelimit-limit-requests": "1000",
    "x-ratelimit-limit-tokens": "8000",
    "x-ratelimit-remaining-requests": "999",
    "x-ratelimit-remaining-tokens": "7902",
    "x-ratelimit-reset-requests": "1m26.4s",
    "x-ratelimit-reset-tokens": "735ms",
}


def chat_response(content: str, *, model: str = "openai/gpt-oss-20b",
                  prompt_tokens: int = 1623, completion_tokens: int = 451,
                  reasoning_tokens: int = 6) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


def reasoning_trap_response(model: str = "openai/gpt-oss-20b") -> dict[str, Any]:
    """The real failure mode: 200 OK, empty content, budget spent on reasoning.

    Reproduced live with max_tokens=20 on gpt-oss-20b.
    """
    return {
        "id": "chatcmpl-trap",
        "model": model,
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "",
            "reasoning": "The user says: ... So we should reply"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 19, "completion_tokens": 20, "total_tokens": 39,
                  "completion_tokens_details": {"reasoning_tokens": 20}},
    }


class FakeGroq:
    """Scripted Groq transport. Records every request body it sees."""

    def __init__(self, responses: list[dict[str, Any] | tuple[int, dict[str, Any]]]):
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.content))
            item = self._responses.pop(0) if self._responses else chat_response("{}")
            status, payload = item if isinstance(item, tuple) else (200, item)
            return httpx.Response(status, json=payload, headers=RATE_HEADERS)

        return httpx.MockTransport(handler)


class FakeJev:
    def __init__(self, answers: dict[str, Any] | Callable[[dict], dict],
                 status: int = 200, input_tokens: int = 465):
        self._answers = answers
        self._status = status
        self._input_tokens = input_tokens
        self.requests: list[dict[str, Any]] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.requests.append(body)
            if self._status != 200:
                return httpx.Response(self._status, json={"detail": "bad shape"})
            answers = self._answers(body) if callable(self._answers) else self._answers
            return httpx.Response(200, json={
                "model": "jev-1.14.0", "answers": answers,
                "usage": {"input_tokens": self._input_tokens, "output_tokens": 56}})

        return httpx.MockTransport(handler)
