"""Batch scoring must agree with `rescore` exactly, not approximately.

This is the load-bearing suite of v2. The whole UI premise is that a delta
shown beside a control is the real fused score, so if `score_many` and
`rescore` can disagree by even a rounding step, every number on the page is
suspect. "Close enough" is not a passing condition here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rba.ats.batch import BatchStats, Candidate, DenseCache, score_many
from rba.ats.dense import available, truncated_ids
from rba.resume.attribution import rescore
from rba.resume.ingest import parse_resume

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
JD = ("AI Engineer. Python, LangGraph, RAG, retrieval, evaluation, Docker, "
      "embeddings, vector search, tool-calling, structured output.")


@pytest.fixture
def doc():
    return parse_resume((FIXTURES / "aniket_bits.txt").read_bytes(),
                        filename="aniket_bits.txt")


def _candidate(doc, ident="c0"):
    return Candidate(id=ident, text=doc.as_text(), skills=doc.skills())


# ---------------------------------------------------------------- exactness --
def test_one_candidate_equals_rescore_without_dense(doc):
    expected, expected_channels = rescore(doc, JD, use_dense=False)
    got = score_many([_candidate(doc)], JD, use_dense=False)[0]
    assert got.raw == expected
    assert got.channels == expected_channels


def test_many_candidates_each_equal_their_own_rescore(doc):
    """Batching must not change any individual answer."""
    variants = []
    for bullet in doc.enabled()[:6]:
        bullet.enabled = False
        variants.append((bullet.id, doc.as_text(), doc.skills()))
        bullet.enabled = True

    candidates = [Candidate(id=i, text=t, skills=s) for i, t, s in variants]
    batched = score_many(candidates, JD, use_dense=False)

    for (bullet_id, _text, _skills), scored in zip(variants, batched, strict=True):
        bullet = next(b for b in doc.all_bullets() if b.id == bullet_id)
        bullet.enabled = False
        expected, _ = rescore(doc, JD, use_dense=False)
        bullet.enabled = True
        assert scored.raw == expected, f"batch disagreed on {bullet_id}"


def test_order_is_preserved(doc):
    candidates = [Candidate(id=f"c{i}", text=f"{doc.as_text()} extra{i}",
                            skills=doc.skills()) for i in range(5)]
    got = score_many(candidates, JD, use_dense=False)
    assert [s.id for s in got] == [f"c{i}" for i in range(5)]


def test_empty_input_is_not_an_error():
    assert score_many([], JD) == []


def test_required_and_preferred_skills_reach_the_scorer(doc):
    plain = score_many([_candidate(doc)], JD, use_dense=False)[0]
    with_skills = score_many([_candidate(doc)], JD, use_dense=False,
                             required_skills=["python", "docker"],
                             preferred_skills=["langgraph"])[0]
    assert "skills" in with_skills.channels
    assert with_skills.channels != plain.channels


# ------------------------------------------------------------------ dedup --
def test_truncation_dedup_is_exact(doc):
    """Candidates sharing a 512-token prefix must score identically.

    Not an optimisation detail: if this were approximate, the dedup would be
    silently inventing scores for most of the resume.
    """
    enabled = doc.enabled()
    texts, ids = [doc.as_text()], ["full"]
    for bullet in enabled:
        bullet.enabled = False
        texts.append(doc.as_text())
        ids.append(bullet.id)
        bullet.enabled = True

    prefixes = truncated_ids(texts)
    if prefixes is None:
        pytest.skip("embedding model unavailable")
    assert len(set(prefixes)) < len(prefixes), \
        "fixture no longer exercises truncation; the dedup is untested"

    groups: dict[tuple, list[str]] = {}
    for ident, prefix in zip(ids, prefixes, strict=True):
        groups.setdefault(prefix, []).append(ident)
    shared = [g for g in groups.values() if len(g) > 1]
    assert shared, "expected at least one shared prefix group"

    if not available():
        pytest.skip("embedding model unavailable")

    candidates = [Candidate(id=i, text=t, skills=doc.skills())
                  for i, t in zip(ids, texts, strict=True)]
    scored = {s.id: s for s in score_many(candidates, JD, use_dense=True)}

    # The actual claim: everyone sharing a prefix got the same dense value.
    # Not merely "we embedded fewer texts" -- that would be true of a wrong
    # implementation too.
    for group in shared:
        values = {scored[ident].channels.get("dense") for ident in group}
        assert len(values) == 1, \
            f"candidates sharing a 512-token prefix disagreed on dense: {group}"


@pytest.mark.slow
def test_dedup_matches_the_naive_path_with_dense_on(doc):
    """The expensive proof: dedup on == dedup off, to float equality."""
    if not available():
        pytest.skip("embedding model unavailable")

    enabled = doc.enabled()[:8]
    candidates = []
    for bullet in enabled:
        bullet.enabled = False
        candidates.append(Candidate(id=bullet.id, text=doc.as_text(),
                                    skills=doc.skills()))
        bullet.enabled = True

    batched = score_many(candidates, JD, use_dense=True)
    for candidate, scored in zip(candidates, batched, strict=True):
        bullet = next(b for b in doc.all_bullets() if b.id == candidate.id)
        bullet.enabled = False
        expected, _ = rescore(doc, JD, use_dense=True)
        bullet.enabled = True
        assert scored.raw == expected, f"dedup changed the score for {candidate.id}"


# ------------------------------------------------------------------ cache --
def test_cache_hits_do_not_change_the_answer(doc):
    cache = DenseCache()
    first = score_many([_candidate(doc)], JD, use_dense=False, cache=cache)[0]
    second = score_many([_candidate(doc)], JD, use_dense=False, cache=cache)[0]
    assert first.raw == second.raw


def test_cache_is_bounded():
    cache = DenseCache(maxsize=3)
    for i in range(10):
        cache.put((0, (i,)), float(i))
    assert len(cache) == 3
    assert cache.get((0, (0,))) is None      # evicted
    assert cache.get((0, (9,))) == 9.0       # kept


def test_cache_evicts_least_recently_used():
    cache = DenseCache(maxsize=2)
    cache.put((0, ("a",)), 1.0)
    cache.put((0, ("b",)), 2.0)
    cache.get((0, ("a",)))                   # 'a' is now the fresher entry
    cache.put((0, ("c",)), 3.0)
    assert cache.get((0, ("b",))) is None
    assert cache.get((0, ("a",))) == 1.0


def test_a_different_jd_does_not_reuse_the_cached_cosine(doc):
    """The cache key must include the job description, or editing the JD
    would silently keep scoring against the previous one."""
    cache = DenseCache()
    cache.put((hash(JD), (1, 2, 3)), 0.9)
    assert cache.get((hash("a completely different job"), (1, 2, 3))) is None


# ------------------------------------------------------------------ stats --
def test_stats_report_what_happened(doc):
    stats = BatchStats()
    score_many([_candidate(doc)], JD, use_dense=False, stats=stats)
    assert stats.candidates == 1
    assert stats.to_dict()["candidates"] == 1
