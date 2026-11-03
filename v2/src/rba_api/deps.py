"""Process-wide singletons, created once in the app lifespan.

What must be app-scoped, and why each one bites if it is not:

* **Limiter** -- `GroqClient.__init__` does ``limiter or Limiter(config)``, so a
  per-request client gets a private rate limiter and each concurrent job then
  believes it owns the whole 8,000 TPM budget. Keyed per API key, because
  that is how Groq's quota is scoped.
* **Cache** -- one sqlite connection, opened once, shared. See `rba.cache`.
* **Pools** -- see `scoring_pool`.
* **The embedding model** -- warmed at startup so the first user request does
  not pay the 1.5 s cold load.
"""
from __future__ import annotations

import hashlib
import logging
import os

from rba.cache import Cache
from rba.config import Config
from rba.limiter import Limiter

from .jobs import JobStore
from .scoring_pool import Pools
from .sessions import SessionStore

log = logging.getLogger(__name__)


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class AppState:
    def __init__(self, *, testing: bool = False) -> None:
        self.testing = testing
        self.config = Config.load()
        self.pools = Pools()
        self.sessions = SessionStore(
            ttl=float(os.environ.get("RBA_SESSION_TTL", "7200")))
        self.cache = Cache(enabled=not testing)
        self.jobs = JobStore()
        self._limiters: dict[str, Limiter] = {}
        self.dense_ready = False

    def limiter_for(self, api_key: str) -> Limiter:
        key = hashlib.sha256((api_key or "anon").encode()).hexdigest()[:16]
        if key not in self._limiters:
            self._limiters[key] = Limiter(self.config)
        return self._limiters[key]

    def warm(self) -> None:
        """Load the embedding model and the fixed corpora.

        Deliberately tolerant: `dense.load()` already degrades to None and the
        other channels carry on. But a deployment that *intends* to score
        properly can set RBA_REQUIRE_DENSE=1 and get a hard failure at startup
        instead of silently serving numbers missing half their weight.
        """
        from rba.ats import dense
        from rba.ats.corpus import background_corpus

        if self.testing:
            return
        local_only = _flag("RBA_MODEL_LOCAL_ONLY", False)
        self.dense_ready = dense.load(local_only=local_only) is not None
        if self.dense_ready:
            dense.embed(["warm"])
        elif _flag("RBA_REQUIRE_DENSE"):
            raise RuntimeError(
                "the dense channel failed to load and RBA_REQUIRE_DENSE is set; "
                "refusing to serve scores missing 50% of their weight")
        else:
            log.warning("dense channel unavailable; scores will use the "
                        "remaining channels with weights redistributed")
        background_corpus()

    def shutdown(self) -> None:
        self.pools.shutdown()
        self.cache.close()


def session_cookie_name() -> str:
    return "rba_session"
