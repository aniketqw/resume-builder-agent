"""The Ollama client: a structural twin of `GroqClient.structured()`.

No rate limiter to test here -- that is the entire point of this client
existing (see `trace.md` Iteration 15). What has to be proven instead: the
request shape (a full JSON Schema in `format`, not just `"json"`), the
schema-violation retry, and that `Usage` is populated from Ollama's real
response fields (`prompt_eval_count`/`eval_count`).
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel

from fakes import FakeOllama, ollama_chat_response
from rba.config import Config
from rba.groq_client import GroqError
from rba.ollama_client import OllamaClient


class Summary(BaseModel):
    one_line: str
    tech: list[str]


def client(fake, **kw):
    cfg = Config.load()
    return OllamaClient(config=cfg, base_url="http://fake", model="qwen2.5:14b-instruct",
                       transport=fake.transport(), **kw)


async def test_structured_sends_a_full_json_schema_not_just_json_mode():
    fake = FakeOllama([ollama_chat_response('{"one_line":"x","tech":["Python"]}')])
    async with client(fake) as c:
        obj, usage = await c.structured([{"role": "user", "content": "hi"}], Summary)
    assert obj.tech == ["Python"]
    body = fake.requests[0]
    assert body["format"] == Summary.model_json_schema()
    assert body["stream"] is False
    assert usage.prompt_tokens == 50 and usage.completion_tokens == 130


async def test_structured_retries_once_on_schema_violation():
    fake = FakeOllama([ollama_chat_response('{"one_line": 42}'),
                       ollama_chat_response('{"one_line":"ok","tech":[]}')])
    async with client(fake) as c:
        obj, _usage = await c.structured([{"role": "user", "content": "hi"}], Summary)
    assert obj.one_line == "ok"
    assert len(fake.requests) == 2
    assert "did not match the schema" in fake.requests[1]["messages"][-1]["content"]


async def test_structured_gives_up_after_the_retry():
    fake = FakeOllama([ollama_chat_response('{"bad":1}'),
                       ollama_chat_response('{"also":"bad"}')])
    async with client(fake) as c:
        with pytest.raises(GroqError, match="schema validation failed"):
            await c.structured([{"role": "user", "content": "hi"}], Summary)


async def test_a_4xx_is_not_retried_by_this_method_itself():
    fake = FakeOllama([(400, {"error": "model not found"})])
    async with client(fake) as c:
        with pytest.raises(GroqError, match="400"):
            await c.structured([{"role": "user", "content": "hi"}], Summary)
    assert len(fake.requests) == 1


async def test_cache_hit_costs_zero_tokens(tmp_path):
    from rba.cache import Cache

    cache = Cache(tmp_path / "c.sqlite")
    fake = FakeOllama([ollama_chat_response('{"one_line":"x","tech":[]}')])
    cfg = Config.load()
    async with OllamaClient(config=cfg, base_url="http://fake", transport=fake.transport(),
                            cache=cache) as c:
        await c.structured([{"role": "user", "content": "hi"}], Summary, cache_key="k1")
        _obj, usage = await c.structured([{"role": "user", "content": "hi"}], Summary,
                                         cache_key="k1")
    assert len(fake.requests) == 1, "the second call must be served from cache"
    assert usage.cached and usage.total_tokens == 0


async def test_usage_report_groups_by_node():
    fake = FakeOllama([ollama_chat_response('{"one_line":"x","tech":[]}')])
    async with client(fake) as c:
        await c.structured([{"role": "user", "content": "hi"}], Summary, node="summarize_repo")
    report = c.usage_report()
    assert report["summarize_repo"]["total"] == 180
