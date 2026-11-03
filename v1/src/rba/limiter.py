"""Per-model rate limiting for Groq's free tier.

The binding constraint is 8,000 tokens per minute (confirmed from
``x-ratelimit-limit-tokens``), not the 30 requests per minute. Buckets are
per-model, so routing a step to a different tier also buys throughput.

Flow for every call:
  1. ``reserve()``   - estimate tokens, wait until the minute window has room.
  2. the HTTP call.
  3. ``reconcile()`` - replace the estimate with ``usage.total_tokens`` and
     overwrite the window from the ``x-ratelimit-remaining-*`` headers, which
     are authoritative.

Daily counters stop the run cleanly at ``daily_stop_fraction`` so a long sweep
checkpoints instead of dying on a 429 wall.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from .config import Limits

_DURATION = re.compile(r"(?:(?P<m>[\d.]+)m)?(?:(?P<s>[\d.]+)s)?(?:(?P<ms>[\d.]+)ms)?")


def parse_reset(value: str | None) -> float:
    """Parse Groq's reset format ('1m26.4s', '735ms', '7.66s') into seconds."""
    if not value:
        return 0.0
    v = value.strip()
    # 'ms' must be checked before 's', since 'ms' ends in 's'.
    if v.endswith("ms") and "m" not in v[:-2]:
        try:
            return float(v[:-2]) / 1000.0
        except ValueError:
            return 0.0
    m = _DURATION.fullmatch(v)
    if not m or not any(m.groupdict().values()):
        try:
            return float(v)
        except ValueError:
            return 0.0
    total = 0.0
    if m.group("m"):
        total += float(m.group("m")) * 60
    if m.group("s"):
        total += float(m.group("s"))
    if m.group("ms"):
        total += float(m.group("ms")) / 1000.0
    return total


class DailyBudgetExhausted(RuntimeError):
    """Raised when the run would cross the configured fraction of RPD/TPD."""


@dataclass
class _Window:
    """A rolling one-minute window of (timestamp, amount) events.

    Response headers are more authoritative than local accounting, but only
    for as long as they describe. ``x-ratelimit-remaining-tokens`` is a
    snapshot of the current minute and the bucket refills after
    ``x-ratelimit-reset-tokens``; an override that never expires is worse than
    no override at all, because it pins the window at whatever the last
    response reported. The limiter then concludes it has no budget and sleeps
    a full minute before every subsequent call, indefinitely.
    """

    limit: int
    events: list[tuple[float, float]] = field(default_factory=list)
    override: float | None = None       # from response headers
    override_until: float = 0.0         # when that snapshot stops being true

    def _active_override(self, now: float) -> float | None:
        if self.override is None:
            return None
        if now >= self.override_until:
            self.override = None        # expired: fall back to local accounting
            return None
        return self.override

    def set_override(self, now: float, value: float, resets_in: float) -> None:
        self.override = max(0.0, value)
        # Never trust a snapshot for longer than the window it describes.
        self.override_until = now + max(0.0, min(resets_in, 60.0))

    def prune(self, now: float) -> None:
        cutoff = now - 60.0
        self.events = [(t, a) for t, a in self.events if t > cutoff]

    def used(self, now: float) -> float:
        self.prune(now)
        return sum(a for _, a in self.events)

    def remaining(self, now: float) -> float:
        override = self._active_override(now)
        if override is not None:
            return override
        return self.limit - self.used(now)

    def add(self, now: float, amount: float) -> None:
        self.events.append((now, amount))
        if self._active_override(now) is not None:
            self.override = max(0.0, self.override - amount)

    def adjust(self, now: float, delta: float) -> None:
        """Correct the most recent event by ``delta`` (estimate -> actual)."""
        if self.events:
            t, a = self.events[-1]
            self.events[-1] = (t, max(0.0, a + delta))
        if self._active_override(now) is not None:
            self.override = max(0.0, self.override - delta)

    def seconds_until_room(self, now: float, need: float) -> float:
        """How long to wait before ``need`` units fit in the window."""
        if self.remaining(now) >= need:
            return 0.0
        # If a header snapshot is what is blocking us, wait for it to expire
        # rather than for local events that may not exist.
        if self._active_override(now) is not None:
            return max(0.0, self.override_until - now)
        self.prune(now)
        # Expire oldest events until enough room frees up.
        freed = 0.0
        shortfall = need - self.remaining(now)
        for t, a in self.events:
            freed += a
            if freed >= shortfall:
                return max(0.0, (t + 60.0) - now)
        return 60.0


class ModelBucket:
    """TPM + RPM windows and RPD + TPD daily counters for one model."""

    def __init__(self, model: str, limits: Limits, stop_fraction: float = 0.9,
                 clock=time.monotonic):
        self.model = model
        self.limits = limits
        self.stop_fraction = stop_fraction
        self._clock = clock
        self.tokens = _Window(limits.tpm)
        self.requests = _Window(limits.rpm)
        self.day_requests = 0
        self.day_tokens = 0
        self._lock = asyncio.Lock()

    # -- daily guard ------------------------------------------------------
    def _check_daily(self, need_tokens: float) -> None:
        if self.day_requests + 1 > self.limits.rpd * self.stop_fraction:
            raise DailyBudgetExhausted(
                f"{self.model}: {self.day_requests} requests is at "
                f"{self.stop_fraction:.0%} of the {self.limits.rpd}/day limit"
            )
        if self.day_tokens + need_tokens > self.limits.tpd * self.stop_fraction:
            raise DailyBudgetExhausted(
                f"{self.model}: {self.day_tokens} tokens is at "
                f"{self.stop_fraction:.0%} of the {self.limits.tpd}/day limit"
            )

    # -- main API ---------------------------------------------------------
    async def reserve(self, est_tokens: int) -> float:
        """Block until ``est_tokens`` fit in the minute window. Returns wait time."""
        waited = 0.0
        async with self._lock:
            while True:
                now = self._clock()
                self._check_daily(est_tokens)
                wait = max(
                    self.tokens.seconds_until_room(now, est_tokens),
                    self.requests.seconds_until_room(now, 1),
                )
                if wait <= 0:
                    self.tokens.add(now, est_tokens)
                    self.requests.add(now, 1)
                    self.day_requests += 1
                    self.day_tokens += est_tokens
                    return waited
                waited += wait
                await asyncio.sleep(wait)

    def reconcile(self, est_tokens: int, actual_tokens: int,
                  headers: dict[str, str] | None = None) -> None:
        """Replace the estimate with the real usage; headers win over local maths."""
        now = self._clock()
        delta = actual_tokens - est_tokens
        self.tokens.adjust(now, delta)
        self.day_tokens += delta
        if not headers:
            return
        h = {k.lower(): v for k, v in headers.items()}
        rem_t = h.get("x-ratelimit-remaining-tokens")
        if rem_t is not None:
            try:
                resets_in = parse_reset(h.get("x-ratelimit-reset-tokens")) or 60.0
                self.tokens.set_override(now, float(rem_t), resets_in)
            except ValueError:
                pass
        rem_r = h.get("x-ratelimit-remaining-requests")
        if rem_r is not None:
            try:
                # This header is Requests Per Day, not per minute.
                self.day_requests = self.limits.rpd - int(float(rem_r))
            except ValueError:
                pass

    def retry_delay(self, attempt: int, headers: dict[str, str] | None = None) -> float:
        """Honour ``retry-after`` if present, else exponential backoff."""
        if headers:
            h = {k.lower(): v for k, v in headers.items()}
            ra = h.get("retry-after")
            if ra:
                try:
                    return float(ra)
                except ValueError:
                    return parse_reset(ra)
            reset = h.get("x-ratelimit-reset-tokens")
            if reset:
                return parse_reset(reset)
        return min(60.0, 2.0 ** attempt)

    def snapshot(self) -> dict[str, float]:
        now = self._clock()
        return {
            "model": self.model,
            "tpm_remaining": self.tokens.remaining(now),
            "rpm_remaining": self.requests.remaining(now),
            "day_requests": self.day_requests,
            "day_tokens": self.day_tokens,
        }


class Limiter:
    """Holds one :class:`ModelBucket` per model."""

    def __init__(self, config, clock=time.monotonic):
        self._config = config
        self._clock = clock
        self._buckets: dict[str, ModelBucket] = {}
        self.stop_fraction = config.get("groq.daily_stop_fraction", 0.9)

    def bucket(self, model: str) -> ModelBucket:
        if model not in self._buckets:
            self._buckets[model] = ModelBucket(
                model, self._config.limits_for(model), self.stop_fraction, self._clock
            )
        return self._buckets[model]

    def snapshot(self) -> list[dict[str, float]]:
        return [b.snapshot() for b in self._buckets.values()]

    def totals(self) -> dict[str, int]:
        return {
            "requests": sum(b.day_requests for b in self._buckets.values()),
            "tokens": sum(b.day_tokens for b in self._buckets.values()),
        }


def estimate_tokens(text: str, margin: float = 0.10) -> int:
    """Cheap pre-call estimate: ~4 chars/token for English + JSON, plus margin."""
    base = max(1, len(text) // 4)
    return int(base * (1.0 + margin))
