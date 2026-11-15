
import pytest
from pydantic import BaseModel

from fakes import FakeGroq, chat_response, reasoning_trap_response, schema_rejected_response
from rba.cache import Cache
from rba.config import Config
from rba.groq_client import MIN_MAX_TOKENS, GroqClient, GroqError, SchemaRejectedError
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


async def test_a_shape_violation_400_raises_schema_rejected_not_groqerror():
    """`complete()` must not try to repair this one itself: growing
    `max_tokens` does nothing for a shape violation, and `structured()`
    cannot apply its correction-message repair to a plain `GroqError` it
    can't distinguish from any other 4xx."""
    fake = FakeGroq([schema_rejected_response()])
    async with client(fake) as c:
        with pytest.raises(SchemaRejectedError) as exc_info:
            await c.complete([{"role": "user", "content": "hi"}], tier="small",
                             schema=Summary)
    assert "missing properties" in str(exc_info.value)
    assert exc_info.value.failed_generation == '{"claims":[{"evidence":[]}]}'
    assert len(fake.requests) == 1, "complete() must not retry this itself"


async def test_structured_recovers_from_a_real_schema_rejected_400():
    """The live failure this guards against: three real GitHub-import repos
    failed exactly this way (`summarize_repo[...]`: 400, missing `text`),
    because the old code treated this identically to the max-tokens case --
    retrying with a bigger budget and an unchanged prompt, which cannot fix
    a shape violation. `structured()` must now show the model its own bad
    generation and ask again, the same repair already used for a
    client-side `ValidationError`."""
    fake = FakeGroq([schema_rejected_response(),
                     chat_response('{"one_line":"ok","tech":[]}')])
    async with client(fake) as c:
        obj, _usage = await c.structured([{"role": "user", "content": "hi"}], Summary,
                                        tier="small")
    assert obj.one_line == "ok"
    assert len(fake.requests) == 2
    retry_prompt = fake.requests[1]["messages"][-1]["content"]
    assert "did not match the schema" in retry_prompt
    # The corrected prompt must include the model's own prior (bad) answer,
    # not just an abstract description of the problem -- the same repair
    # `structured()` already does for a client-side ValidationError.
    assert fake.requests[1]["messages"][-2]["content"] == '{"claims":[{"evidence":[]}]}'


async def test_structured_gives_up_after_a_second_schema_rejected_400():
    fake = FakeGroq([schema_rejected_response(), schema_rejected_response()])
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


# -- tracing: nest under the active graph run, not a flat root -------------
class StubRunTree:
    """Stands in for `langsmith.run_trees.RunTree` -- just enough surface
    for `_trace` to exercise: `create_child`, `post`, `end`, `patch`."""

    def __init__(self, name="root", inputs=None):
        self.name = name
        self.inputs = inputs
        self.outputs = None
        self.posted = False
        self.ended = False
        self.patched = False
        self.children: list[StubRunTree] = []

    def create_child(self, name, run_type="chain", inputs=None, start_time=None):
        child = StubRunTree(name, inputs)
        self.children.append(child)
        return child

    def post(self):
        self.posted = True

    def end(self, outputs=None, end_time=None):
        self.outputs = outputs
        self.ended = True

    def patch(self):
        self.patched = True


class StubTracerClient:
    """Stands in for `langsmith.Client` -- only `create_run` is used."""

    def __init__(self):
        self.root_runs: list[dict] = []

    def create_run(self, **kw):
        self.root_runs.append(kw)


async def test_trace_nests_under_the_active_graph_run(monkeypatch):
    """Before this, every call was a flat `create_run` with no parent -- a
    disconnected root next to whatever trace tree LangGraph already built
    for the node that called it. `get_current_run_tree()` is how to find
    that in-progress parent; this is what it looks like when one exists."""
    fake = FakeGroq([chat_response('{"ok": true}')])
    c = client(fake)
    c._tracer = StubTracerClient()
    parent = StubRunTree("connect_github")
    monkeypatch.setattr("langsmith.run_helpers.get_current_run_tree", lambda: parent)

    await c.complete([{"role": "user", "content": "hi"}], node="summarize_repo")
    await c.aclose()

    assert not c._tracer.root_runs, "must not also create a disconnected root"
    assert len(parent.children) == 1
    child = parent.children[0]
    assert child.name == "summarize_repo"
    assert child.posted and child.ended and child.patched
    assert child.outputs["usage"]["total"] == 2074


async def test_trace_falls_back_to_a_root_run_outside_a_graph_context(monkeypatch):
    """A `GroqClient` used directly (a unit test, a script) has no active
    LangGraph node run to nest under -- the trace must still be recorded,
    just as a root, not silently dropped."""
    fake = FakeGroq([chat_response('{"ok": true}')])
    c = client(fake)
    c._tracer = StubTracerClient()
    monkeypatch.setattr("langsmith.run_helpers.get_current_run_tree", lambda: None)

    await c.complete([{"role": "user", "content": "hi"}], node="summarize_repo")
    await c.aclose()

    assert len(c._tracer.root_runs) == 1
    run = c._tracer.root_runs[0]
    assert run["name"] == "summarize_repo"
    assert run["start_time"] is not None and run["end_time"] is not None


async def test_a_cache_hit_is_traced_as_a_zero_token_run(monkeypatch, tmp_path):
    """Before this, `_trace` was only called on the success path *after*
    the cache-hit early return -- a cached completion never produced a run
    at all, under-reporting usage in anything that reads LangSmith for it."""
    cfg = Config.load()
    cache = Cache(tmp_path / "c.sqlite")
    fake = FakeGroq([chat_response('{"ok":1}')])
    c = GroqClient(config=cfg, limiter=Limiter(cfg), cache=cache, api_key="k",
                   transport=fake.transport())
    c._tracer = StubTracerClient()
    monkeypatch.setattr("langsmith.run_helpers.get_current_run_tree", lambda: None)

    msgs = [{"role": "user", "content": "hi"}]
    await c.complete(msgs, tier="small", cache_key="repo@sha@v1")
    await c.complete(msgs, tier="small", cache_key="repo@sha@v1")
    await c.aclose()

    assert len(c._tracer.root_runs) == 2
    assert c._tracer.root_runs[1]["outputs"]["usage"]["cached"] is True
    assert c._tracer.root_runs[1]["outputs"]["usage"]["total"] == 0


async def test_tracing_failure_never_breaks_the_call(monkeypatch):
    """The one invariant that must survive any change here."""
    fake = FakeGroq([chat_response('{"ok": true}')])
    c = client(fake)

    class ExplodingTracer:
        def create_run(self, **kw):
            raise RuntimeError("langsmith is down")

    c._tracer = ExplodingTracer()
    monkeypatch.setattr("langsmith.run_helpers.get_current_run_tree", lambda: None)

    res = await c.complete([{"role": "user", "content": "hi"}], tier="small")
    await c.aclose()
    assert res.parsed() == {"ok": True}
