"""The cache is driven from worker threads in v2, so cross-thread use and
concurrent writes have to be real tests, not assumptions."""
from __future__ import annotations

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from rba.cache import Cache, make_key


@pytest.fixture
def cache(tmp_path):
    c = Cache(path=tmp_path / "responses.sqlite")
    yield c
    c.close()


def test_a_cache_opened_on_one_thread_is_usable_on_another(cache):
    """v1's default would raise ProgrammingError here."""
    cache.put("k", "m", "value")
    result: list = []

    def read():
        try:
            result.append(cache.get("k", "m"))
        except sqlite3.ProgrammingError as exc:   # pragma: no cover
            result.append(exc)

    thread = threading.Thread(target=read)
    thread.start()
    thread.join()
    assert result == ["value"]


def test_concurrent_writers_do_not_lose_entries(cache):
    keys = [make_key("repo", i) for i in range(60)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda k: cache.put(k, "model", f"body-{k}"), keys))

    assert all(cache.get(k, "model") == f"body-{k}" for k in keys)


def test_concurrent_readers_and_writers_do_not_raise(cache):
    cache.put("warm", "model", "x")
    errors: list = []

    def churn(i):
        try:
            cache.put(f"k{i}", "model", str(i))
            cache.get("warm", "model")
        except Exception as exc:                   # pragma: no cover
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(churn, range(100)))
    assert errors == []


def test_a_disabled_cache_is_inert(tmp_path):
    c = Cache(path=tmp_path / "x.sqlite", enabled=False)
    c.put("k", "m", "v")
    assert c.get("k", "m") is None
    c.close()


def test_close_is_idempotent(tmp_path):
    c = Cache(path=tmp_path / "x.sqlite")
    c.close()
    c.close()
