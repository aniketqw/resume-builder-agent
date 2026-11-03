
import pytest
from pydantic import BaseModel

from fakes import FakeGroq, chat_response, reasoning_trap_response
from rba.cache import Cache
from rba.config import Config
from rba.groq_client import MIN_MAX_TOKENS, GroqClient, GroqError
from rba.limiter import Limiter


class Summary(BaseModel):
    one_line: str
    tech: list[str]


def client(fake, **kw):
    cfg = Config.load()
    return GroqClient(config=cfg, limiter=Limiter(cfg), cache=Cache(enabled=False),
                      api_key="test-key", transport=fake.transport(), **kw)


async def test_basic_completion_records_usage():
    fake = FakeGroq([chat_response('{"ok": true}')])
    async with client(fake) as c:
        res = await c.complete([{"role": "user", "content": "hi"}],
                               node="summarize_repo")
        assert res.parsed() == {"ok": True}
        assert res.usage.total_tokens == 2074  # the measured real-world figure
        assert c.usage_report()["summarize_repo"]["total"] == 2074


async def test_reasoning_effort_and_max_tokens_floor_are_always_sent():
    """Regression: without these, gpt-oss returns empty content."""
    fake = FakeGroq([chat_response("{}")])
    async with client(fake) as c:
        await c.complete([{"role": "user", "content": "hi"}], tier="small",
                         max_tokens=20)
    body = fake.requests[0]
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] >= MIN_MAX_TOKENS, "a caller must not be able to starve content"


async def test_empty_content_is_retried_with_a_bigger_budget():
    """The reasoning trap: 200 OK + empty content -> retry, don't return junk."""
    fake = FakeGroq([reasoning_trap_response(), chat_response('{"ok":1}')])
    async with client(fake) as c:
        res = await c.complete([{"role": "user", "content": "hi"}], tier="small")
    assert res.parsed() == {"ok": 1}
    assert len(fake.requests) == 2
    assert fake.requests[1]["max_tokens"] > fake.requests[0]["max_tokens"]


async def test_persistent_empty_content_eventually_raises():
    fake = FakeGroq([reasoning_trap_response() for _ in range(5)])
    async with client(fake) as c:
        with pytest.raises(GroqError, match="empty content"):
            await c.complete([{"role": "user", "content": "hi"}], tier="small")


async def test_strict_schema_used_for_gpt_oss():
    fake = FakeGroq([chat_response('{"one_line":"x","tech":["Python"]}')])
    async with client(fake) as c:
        obj, _usage = await c.structured([{"role": "user", "content": "hi"}], Summary,
                                        tier="small")
    assert obj.tech == ["Python"]
    rf = fake.requests[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["additionalProperties"] is False


async def test_non_strict_model_falls_back_to_json_object():
    """langchain-groq/Groq silently ignore strict=True off gpt-oss, so don't ask."""
    fake = FakeGroq([chat_response('{"one_line":"x","tech":[]}')])
    async with client(fake) as c:
        await c.structured([{"role": "user", "content": "hi"}], Summary,
                           model="qwen/qwen3.8-27b")
    assert fake.requests[0]["response_format"] == {"type": "json_object"}


async def test_structured_retries_once_on_schema_violation():
    fake = FakeGroq([chat_response('{"one_line": 42}'),
                     chat_response('{"one_line":"ok","tech":[]}')])
    async with client(fake) as c:
        obj, _usage = await c.structured([{"role": "user", "content": "hi"}], Summary,
                                        tier="small")
    assert obj.one_line == "ok"
    assert len(fake.requests) == 2
    # The retry must tell the model what it got wrong.
    assert "did not match the schema" in fake.requests[1]["messages"][-1]["content"]


async def test_structured_gives_up_after_the_retry():
    fake = FakeGroq([chat_response('{"bad":1}'), chat_response('{"also":"bad"}')])
    async with client(fake) as c:
        with pytest.raises(GroqError, match="schema validation failed"):
            await c.structured([{"role": "user", "content": "hi"}], Summary, tier="small")


async def test_tools_plus_structured_output_is_refused_up_front():
    """Groq cannot do both in one call; fail loudly at build time."""
    fake = FakeGroq([chat_response("{}")])
    async with client(fake) as c:
        with pytest.raises(GroqError, match="separate graph nodes"):
            await c.complete([{"role": "user", "content": "hi"}], tier="small",
                             schema=Summary, tools=[{"type": "function"}])


async def test_429_is_retried_honouring_retry_after(monkeypatch):
    import asyncio
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    fake = FakeGroq([(429, {"error": "rate limited"}), chat_response('{"ok":1}')])
    async with client(fake) as c:
        res = await c.complete([{"role": "user", "content": "hi"}], tier="small")
    assert res.parsed() == {"ok": 1}
    assert slept, "a 429 must cause a sleep"


async def test_4xx_is_not_retried():
    fake = FakeGroq([(400, {"error": {"message": "bad request"}})])
    async with client(fake) as c:
        with pytest.raises(GroqError, match="400"):
            await c.complete([{"role": "user", "content": "hi"}], tier="small")
    assert len(fake.requests) == 1


async def test_cache_hit_costs_zero_tokens(tmp_path):
    cfg = Config.load()
    cache = Cache(tmp_path / "c.sqlite")
    fake = FakeGroq([chat_response('{"ok":1}')])
    c = GroqClient(config=cfg, limiter=Limiter(cfg), cache=cache, api_key="k",
                   transport=fake.transport())
    msgs = [{"role": "user", "content": "hi"}]
    first = await c.complete(msgs, tier="small", cache_key="repo@sha@v1")
    second = await c.complete(msgs, tier="small", cache_key="repo@sha@v1")
    await c.aclose()
    assert first.usage.total_tokens == 2074
    assert second.usage.total_tokens == 0 and second.usage.cached
    assert len(fake.requests) == 1, "second call must not hit the network"
    assert cache.hit_rate == 0.5
