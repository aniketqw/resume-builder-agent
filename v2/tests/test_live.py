"""The scoring path has to be fast enough to feel like a control, not a report.

A toggle that takes half a second reads as a button press with a wait. The
budget here is one document embedding and nothing else, which is why the job
description's embedding is cached and kept out of the batch.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest

from rba.ats.dense import embed
from rba.resume.attribution import _jd_vector, attribute, rescore
from rba.resume.ingest import parse_bits

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "aniket_bits.txt"
JD = ("AI Engineer. Required: Python, LangChain, LangGraph, RAG, retrieval, "
      "evaluation, Docker. Build and evaluate LLM agent systems.")


@pytest.fixture(scope="module")
def doc():
    return parse_bits(FIXTURE.read_text())


def test_splitting_the_jd_out_of_the_batch_does_not_change_the_vectors(doc):
    """The optimisation must be free.

    `encode_batch` pads every text in a chunk to the longest one, so a short job
    description batched with a long resume is padded out to the resume's length.
    Splitting them is faster, but only legitimate if the numbers are identical:
    attention masking should make padding irrelevant, and this asserts it does.
    """
    text = doc.as_text()
    together = embed([text, JD])
    apart_doc, apart_jd = embed([text]), embed([JD])
    if together is None or apart_doc is None or apart_jd is None:
        pytest.skip("embedding model unavailable")

    assert float(np.abs(together[0] - apart_doc[0]).max()) < 1e-6
    assert float(np.abs(together[1] - apart_jd[0]).max()) < 1e-6
    batched = float(np.dot(together[0], together[1]))
    split = float(np.dot(apart_doc[0], apart_jd[0]))
    assert batched == pytest.approx(split, abs=1e-6)


def test_the_jd_embedding_is_computed_once(doc):
    _jd_vector.cache_clear()
    rescore(doc, JD)
    rescore(doc, JD)
    rescore(doc, JD)
    info = _jd_vector.cache_info()
    assert info.misses == 1, "the job description should be embedded once"
    assert info.hits >= 2


def test_a_different_jd_is_not_served_from_cache(doc):
    _jd_vector.cache_clear()
    first, _ = rescore(doc, JD)
    second, _ = rescore(doc, "Pastry chef. Laminated doughs, viennoiserie.")
    assert _jd_vector.cache_info().misses == 2
    assert first != second, "a different job must produce a different score"


@pytest.mark.slow
def test_one_toggle_stays_inside_the_interaction_budget(doc):
    """Measured, not asserted in a comment.

    The ceiling is deliberately loose (CI machines vary) but far below the
    590 ms this cost when the job description was re-embedded every call.
    """
    rescore(doc, JD)                      # warm the model and the JD cache
    start = time.perf_counter()
    rescore(doc, JD)
    elapsed = (time.perf_counter() - start) * 1000
    assert elapsed < 450, f"one re-score took {elapsed:.0f} ms"


def test_attribution_and_rescore_agree_exactly(doc):
    """Two code paths compute the same number; they must not drift apart."""
    raw, _ = rescore(doc, JD)
    att = attribute(doc, JD)
    assert round(raw, 10) == round(att.raw_total, 10)


def test_toggling_moves_the_score_by_exactly_the_reported_delta(doc):
    """The invariant the whole UI rests on, re-asserted after the cache change."""
    att = attribute(doc, JD)
    target = att.ranked()[0]
    before, _ = rescore(doc, JD)
    doc.toggle(target.bullet_id, False)
    try:
        after, _ = rescore(doc, JD)
    finally:
        doc.toggle(target.bullet_id, True)
    assert round((before - after) * 100, 4) == round(target.delta, 4)
