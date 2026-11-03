"""SQLite response cache keyed by (key, model).

The key a caller builds normally includes the repo name, its HEAD SHA and the
prompt version, so an unchanged repo costs zero tokens on a re-run. That matters
a lot at 8,000 tokens per minute.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

from .config import PROJECT_ROOT

DEFAULT_PATH = PROJECT_ROOT / ".cache" / "responses.sqlite"


def make_key(*parts: object) -> str:
    """Stable cache key from arbitrary parts (repo, sha, prompt version, ...)."""
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode()).hexdigest()[:32]


class Cache:
    """Thread-safe by construction.

    v1 ran this on one thread, so the default ``check_same_thread=True`` was
    free correctness. v2 offloads scoring and rendering to worker threads and a
    `GroqClient` can be driven from any of them, at which point sqlite3 raises
    ``ProgrammingError: SQLite objects created in a thread can only be used in
    that same thread``. Allowing cross-thread use means we own the serialising,
    hence the lock around every statement -- `put` commits immediately, so two
    concurrent writers without it would interleave transactions.
    """

    def __init__(self, path: str | Path | None = None, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(path) if path else DEFAULT_PATH
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                " key TEXT, model TEXT, content TEXT, created REAL,"
                " PRIMARY KEY (key, model))"
            )
            self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, key: str, model: str) -> str | None:
        if not self._conn:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT content FROM responses WHERE key=? AND model=?",
                (key, model)
            ).fetchone()
        if row:
            self.hits += 1
            return row[0]
        self.misses += 1
        return None

    def put(self, key: str, model: str, content: str) -> None:
        if not self._conn:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?)",
                (key, model, content, time.time()),
            )
            self._conn.commit()

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def close(self) -> None:
        if self._conn:
            with self._lock:
                self._conn.close()
            self._conn = None
            self._conn = None
