"""Limiter tests. All use a fake clock so nothing actually sleeps."""
import asyncio

import pytest

from rba.config import Config, Limits
from rba.limiter import (
    DailyBudgetExhausted,
    Limiter,
    ModelBucket,
    estimate_tokens,
    parse_reset,
)


class FakeClock:
    """Monotonic clock we control, wired into asyncio.sleep by the fixture."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()

    async def fake_sleep(seconds):
        c.advance(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    return c


LIMITS = Limits(rpm=30, rpd=1000, tpm=8000, tpd=200000)


def bucket(clock, limits=LIMITS, stop=0.9):
    return ModelBucket("openai/gpt-oss-20b", limits, stop, clock)


# -- reset header parsing -------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1m26.4s", 86.4),   # x-ratelimit-reset-requests, observed live
        ("735ms", 0.735),    # x-ratelimit-reset-tokens, observed live
        ("7.66s", 7.66),
        ("2m59.56s", 179.56),
        ("2", 2.0),          # bare retry-after seconds
        ("", 0.0),
        (None, 0.0),
    ],
)
def test_parse_reset(raw, expected):
    assert parse_reset(raw) == pytest.approx(expected)


def test_parse_reset_ms_is_not_read_as_minutes():
    # '735ms' must not parse as 735 minutes. Regression guard.
    assert parse_reset("735ms") < 1.0


# -- token window ---------------------------------------------------------
async def test_reserve_under_limit_does_not_wait(clock):
    b = bucket(clock)
    waited = await b.reserve(2000)
    assert waited == 0.0
    assert b.tokens.remaining(clock()) == 6000


async def test_tpm_is_the_binding_constraint(clock):
    """8000 TPM / ~2100 tokens per summary => the 4th call must wait."""
    b = bucket(clock)
    for _ in range(3):
        assert await b.reserve(2100) == 0.0
    waited = await b.reserve(2100)
    assert waited > 0, "4th call should block on TPM, not sail through"
    # It waits until the first call's tokens age out of the 60s window.
    assert waited == pytest.approx(60.0, abs=1.0)


async def test_window_frees_up_after_a_minute(clock):
    b = bucket(clock)
    await b.reserve(8000)
    clock.advance(61)
    assert await b.reserve(8000) == 0.0


async def test_rpm_limit_blocks_even_with_tiny_calls(clock):
    b = bucket(clock, Limits(rpm=3, rpd=1000, tpm=8000, tpd=200000))
    for _ in range(3):
        assert await b.reserve(10) == 0.0
    assert await b.reserve(10) > 0


# -- reconciliation -------------------------------------------------------
def test_reconcile_corrects_an_underestimate(clock):
    b = bucket(clock)
    asyncio.get_event_loop_policy()
    asyncio.run(b.reserve(1000))
    before = b.day_tokens
    b.reconcile(est_tokens=1000, actual_tokens=2074)  # real measured usage
    assert b.day_tokens == before + 1074


def test_headers_override_local_accounting(clock):
    b = bucket(clock)
    asyncio.run(b.reserve(1000))
    b.reconcile(
        est_tokens=1000,
        actual_tokens=2074,
        headers={
            "x-ratelimit-remaining-tokens": "7902",
            "x-ratelimit-remaining-requests": "999",
        },
    )
    assert b.tokens.remaining(clock()) == 7902
    # remaining-requests is Requests Per DAY, not per minute.
    assert b.day_requests == 1


async def test_header_override_expires_and_does_not_stall_the_run(clock):
    """Regression: the header snapshot was cached forever.

    After one response reported a low remaining-tokens figure, the limiter
    believed it had no budget for the rest of the process and slept a full
    minute before every subsequent call -- turning a 3-second eval into one
    that never finished.
    """
    b = bucket(clock)
    await b.reserve(100)
    b.reconcile(100, 100, {"x-ratelimit-remaining-tokens": "200",
                           "x-ratelimit-reset-tokens": "3s"})
    # The header already accounts for the call we just made, so it is taken as-is.
    assert b.tokens.remaining(clock()) == 200

    # A big call cannot fit while the snapshot is current...
    assert b.tokens.seconds_until_room(clock(), 5000) > 0
    # ...and the wait is bounded by the reset, not a blanket 60 seconds.
    assert b.tokens.seconds_until_room(clock(), 5000) <= 3.0

    clock.advance(4)
    # Once it expires, local accounting takes over and the budget is real again.
    assert b.tokens.remaining(clock()) == pytest.approx(7900, abs=200)
    assert await b.reserve(5000) == 0.0


def test_override_is_never_trusted_beyond_a_minute(clock):
    b = bucket(clock)
    b.tokens.set_override(clock(), 10.0, resets_in=9999.0)
    clock.advance(61)
    assert b.tokens.remaining(clock()) == b.limits.tpm


def test_header_remaining_requests_is_treated_as_daily(clock):
    b = bucket(clock)
    asyncio.run(b.reserve(100))
    b.reconcile(100, 100, {"x-ratelimit-remaining-requests": "500"})
    assert b.day_requests == 500  # 1000 rpd - 500 remaining


# -- retry ----------------------------------------------------------------
def test_retry_after_header_is_honoured_exactly(clock):
    b = bucket(clock)
    assert b.retry_delay(0, {"retry-after": "2"}) == 2.0
    assert b.retry_delay(3, {"retry-after": "7.5"}) == 7.5


def test_retry_falls_back_to_reset_tokens_then_backoff(clock):
    b = bucket(clock)
    assert b.retry_delay(0, {"x-ratelimit-reset-tokens": "7.66s"}) == pytest.approx(7.66)
    assert b.retry_delay(3, None) == 8.0
    assert b.retry_delay(10, None) == 60.0  # capped


# -- daily guard ----------------------------------------------------------
async def test_daily_request_budget_stops_cleanly(clock):
    b = bucket(clock, Limits(rpm=10000, rpd=10, tpm=10**9, tpd=10**9), stop=0.9)
    for _ in range(9):
        await b.reserve(1)
    with pytest.raises(DailyBudgetExhausted, match="90%"):
        await b.reserve(1)


async def test_daily_token_budget_stops_cleanly(clock):
    b = bucket(clock, Limits(rpm=10000, rpd=10000, tpm=10**9, tpd=1000), stop=0.9)
    await b.reserve(800)
    with pytest.raises(DailyBudgetExhausted):
        await b.reserve(500)


# -- per-model isolation --------------------------------------------------
async def test_buckets_are_per_model(clock):
    lim = Limiter(Config.load(), clock)
    small = lim.bucket("openai/gpt-oss-20b")
    big = lim.bucket("openai/gpt-oss-120b")
    assert small is not big
    await small.reserve(8000)
    # Exhausting one model must not block another: buckets are per-model.
    assert await big.reserve(8000) == 0.0
    assert lim.totals()["tokens"] == 16000


def test_safeguard_model_gets_its_own_tighter_limits():
    cfg = Config.load()
    assert cfg.limits_for("openai/gpt-oss-safeguard-20b").tpm == 2000
    assert cfg.limits_for("openai/gpt-oss-120b").tpm == 8000


# -- estimation -----------------------------------------------------------
def test_estimate_includes_margin():
    text = "x" * 4000  # ~1000 tokens
    assert estimate_tokens(text, margin=0.10) == 1100


def test_estimate_never_returns_zero():
    assert estimate_tokens("") >= 1
