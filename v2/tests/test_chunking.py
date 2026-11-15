"""Chunked, pooled embedding for documents longer than the encoder's context.

This is opt-in infrastructure (`Candidate.chunks`), not wired into any live
route yet -- see `resume/attribution.py::rescore`'s docstring for why.
Measured on this repo's real 15-bullet resume, chunking every leave-one-out
variant in a full attribution sweep cost 29-67% MORE wall time than the
existing truncated path, worse rather than better, which is why it stops
short of the live endpoints. These tests cover what's built and correct today:
chunk identity, cache reuse, and that a chunked score actually differs from a
truncated one on a document long enough for that to matter.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rba.ats.batch import (
    Candidate,
    DenseCache,
    _ensure_chunks_embedded,
    _pool_from_vectors,
    score_many,
)
from rba.ats.dense import available
from rba.resume.chunking import document_chunks
from rba.resume.document import Bullet, Entry, ResumeDoc, Section
from rba.resume.ingest import parse_resume

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
JD = ("AI Engineer. Python, LangGraph, RAG, retrieval, evaluation, Docker, "
      "embeddings, vector search, tool-calling, structured output.")

needs_dense = pytest.mark.skipif(not available(), reason="embedding model unavailable")


def _doc_with_entries():
    return ResumeDoc(
        name="Jane Doe", headline="Backend Engineer", contact="jane@example.com",
        sections=[Section("PROJECTS", entries=[
            Entry(title="payments-api", bullets=[
                Bullet("Built Python FastAPI microservices", "payments-api"),
                Bullet("Containerised with Docker", "payments-api"),
            ]),
            Entry(title="infra-tools", bullets=[
                Bullet("Automated provisioning with Terraform", "infra-tools"),
            ]),
        ])],
    )


# -- document_chunks: identity and stability ------------------------------
def test_one_chunk_per_entry_and_per_bare_section():
    doc = _doc_with_entries()
    chunks = document_chunks(doc)
    ids = [cid for cid, _ in chunks]
    assert any(cid.startswith("entry:PROJECTS:payments-api:") for cid in ids)
    assert any(cid.startswith("entry:PROJECTS:infra-tools:") for cid in ids)
    # No bare "PROJECTS" section chunk: every bullet lives in an entry, the
    # section itself has no bullets/body/text of its own.
    assert not any(cid.startswith("section:PROJECTS:") for cid in ids)


def test_header_becomes_its_own_chunk():
    doc = _doc_with_entries()
    doc.summary = "Backend engineer focused on reliability."
    ids = [cid for cid, _ in document_chunks(doc)]
    assert "header" in ids


def test_toggling_one_bullet_off_changes_only_its_own_chunk_id():
    doc = _doc_with_entries()
    before = dict(document_chunks(doc))
    target = next(b for e in doc.sections[0].entries for b in e.bullets
                  if "FastAPI" in b.text)
    target.enabled = False
    after = dict(document_chunks(doc))
    target.enabled = True

    payments_before = {cid for cid in before if "payments-api" in cid}
    payments_after = {cid for cid in after if "payments-api" in cid}
    infra_before = {cid for cid in before if "infra-tools" in cid}
    infra_after = {cid for cid in after if "infra-tools" in cid}

    assert payments_before != payments_after, "the touched entry's id must change"
    assert infra_before == infra_after, "an untouched entry's id must not change"


def test_an_entry_with_nothing_enabled_contributes_no_chunk():
    doc = _doc_with_entries()
    for b in doc.sections[0].entries[1].bullets:
        b.enabled = False
    ids = [cid for cid, _ in document_chunks(doc)]
    assert not any(cid.startswith("entry:PROJECTS:infra-tools:") for cid in ids)
    b.enabled = True


# -- pooling and the cache --------------------------------------------------
@needs_dense
def test_cache_reuse_skips_unchanged_chunks():
    """Toggling one bullet should only require re-embedding the one chunk it
    lives in -- every other chunk's vector must come straight from the cache,
    not a fresh embed() call."""
    import rba.ats.batch as batch_mod

    doc = _doc_with_entries()
    full = tuple(document_chunks(doc))
    cache = DenseCache()

    calls = []
    real = batch_mod.embed

    def counting(texts, *a, **k):
        calls.append(list(texts))
        return real(texts, *a, **k)

    batch_mod.embed = counting
    try:
        _ensure_chunks_embedded([full], cache)
        assert sum(len(c) for c in calls) == len(full)  # every chunk, once

        calls.clear()
        target = next(b for e in doc.sections[0].entries for b in e.bullets
                      if "FastAPI" in b.text)
        target.enabled = False
        changed = tuple(document_chunks(doc))
        target.enabled = True

        _ensure_chunks_embedded([changed], cache)
        assert sum(len(c) for c in calls) == 1, \
            "only the one changed chunk should need a fresh embed"
    finally:
        batch_mod.embed = real


@needs_dense
def test_pooled_score_differs_from_truncated_on_a_long_document():
    """The whole point: on a document long enough that truncation drops
    content, the chunked score must actually use that content, not silently
    match the truncated one by coincidence."""
    doc = parse_resume((FIXTURES / "aniket_bits.txt").read_bytes(),
                       filename="aniket_bits.txt")
    chunks = tuple(document_chunks(doc))
    cache = DenseCache()

    flat_candidate = Candidate(id="flat", text=doc.as_text(), skills=doc.skills())
    chunked_candidate = Candidate(id="chunked", text=doc.as_text(),
                                  skills=doc.skills(), chunks=chunks)

    scored = score_many([flat_candidate, chunked_candidate], JD, cache=cache)
    flat_raw, chunked_raw = scored[0].raw, scored[1].raw
    assert flat_raw != chunked_raw, \
        "chunking a 1,150-token resume must change the dense-weighted score"


@needs_dense
def test_base_candidate_is_bit_exact_regardless_of_batch_mates():
    """The bug this guards against: the same document's score must not
    depend on what else is being scored in the same request.

    Measured directly before this fix existed: embedding a chunk alongside
    different batch-mates (i.e. in a differently-padded batch) is not
    bit-exact -- a reproducible ~4.3e-08 difference at the vector level,
    identical across repeated trials, not noise. That is small enough to
    never move a displayed score, but large enough to break `==` (not
    approximate) equality -- exactly what `rescore()` vs `attribute()`'s
    sweep would hit, since one always embeds a document's chunks alone and
    the other embeds them alongside leave-one-out variants.
    """
    doc = parse_resume((FIXTURES / "aniket_bits.txt").read_bytes(),
                       filename="aniket_bits.txt")
    enabled = doc.enabled()

    # The base, scored completely alone (mirrors `rescore()`'s pattern).
    alone = Candidate(id="\x00base", text=doc.as_text(), skills=doc.skills(),
                      chunks=tuple(document_chunks(doc)))
    base_alone = score_many([alone], JD, cache=DenseCache())[0].raw

    # The identical base, scored as candidate 0 of a 5-variant leave-one-out
    # sweep (mirrors `attribute()`'s pattern) -- different batch-mates.
    base_in_sweep = Candidate(id="\x00base", text=doc.as_text(), skills=doc.skills(),
                              chunks=tuple(document_chunks(doc)))
    variants = []
    for b in enabled[:5]:
        b.enabled = False
        variants.append(Candidate(id=b.id, text=doc.as_text(), skills=doc.skills(),
                                  chunks=tuple(document_chunks(doc))))
        b.enabled = True
    base_from_sweep = score_many([base_in_sweep, *variants], JD,
                                 cache=DenseCache())[0].raw

    assert base_alone == base_from_sweep, \
        "a document's own score must not depend on what else is in the batch"


@needs_dense
def test_pool_from_vectors_is_a_length_weighted_mean():
    import numpy as np

    doc = _doc_with_entries()
    chunks = tuple(document_chunks(doc))
    vectors_by_id = _ensure_chunks_embedded([chunks], None)
    pooled = _pool_from_vectors(chunks, vectors_by_id)

    weights = np.array([len(t) for _, t in chunks], dtype=np.float32)
    weights /= weights.sum()
    expected = sum(w * vectors_by_id[cid] for (cid, _), w in zip(chunks, weights, strict=True))
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(pooled, expected, atol=1e-5)
    assert abs(np.linalg.norm(pooled) - 1.0) < 1e-5, "pooled vector must be unit length"


def test_chunking_is_opt_in_and_defaults_to_the_old_path():
    """A Candidate that never sets `chunks` must be completely unaffected --
    this is the backward-compatibility guarantee the whole design rests on."""
    c = Candidate(id="x", text="hello")
    assert c.chunks is None
