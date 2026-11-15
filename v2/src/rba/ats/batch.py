"""Scoring many candidate documents in one pass.

The v2 UI shows a signed score delta on *every* control -- each segmented
version button, each improvement step, each matrix cell. A full simulator view
wants roughly 145 of them. At one `rescore` each (290 ms warm, measured) that
is 70 seconds, so the naive port is unusable and the UI would have to invent
numbers instead. It does not, because of this module.

Four exact levers, no approximation anywhere:

1. **Truncation dedup.** ``dense.MAX_TOKENS`` is 512 and a real one-page resume
   is ~1,150 tokens, so candidates that differ only after the cut have
   bit-identical embeddings. Measured on the attribution path: sixteen variants
   collapse to four forward passes.
2. **One batched call.** The remaining unique texts go through a single
   ``embed()``, which is what `attribution.attribute` already relies on -- the
   tokenizer pads a batch to its longest member, so one batch of twenty-one
   costs far less than twenty-one calls of one.
3. **A dense memo.** Toggling a line off and back on revisits a document we
   have already embedded.
4. **The job description is embedded once**, via `attribution._jd_array`, which
   is `lru_cache`d for exactly this reason.

What is deliberately *not* here: composing a document's score from cached
per-bullet embeddings. `attribution` documents measuring that approximation and
finding it agreed on *sign* for only 14 of 20 bullets. The fusion arithmetic is
likewise not reimplemented -- `_score_from_parts` is imported, so there is one
fusion code path in the process and a batch result cannot drift from `rescore`.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from ..resume.attribution import _jd_array, _score_from_parts
from .corpus import background_corpus
from .dense import embed, truncated_ids
from .fusion import ChannelScores, convex_fuse

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    """One document to score, already flattened to what scoring needs."""

    id: str
    text: str
    skills: list[str] = field(default_factory=list)
    # (chunk_id, chunk_text) pairs from `resume.chunking.document_chunks`.
    # Opt-in: `None` (the default) is the original behaviour -- `text` is
    # embedded as one truncated blob. When set, the dense channel instead
    # embeds each chunk once and pools them, so the whole document -- not
    # just the first ~512 tokens -- contributes to the dense score. See
    # docs/literature-survey.md §8.1.
    chunks: tuple[tuple[str, str], ...] | None = None


@dataclass(frozen=True)
class Scored:
    """What one candidate scored, on the same scale as `rescore`."""

    id: str
    raw: float
    channels: dict
    #: The weights that were *actually* applied. Not the nominal
    #: CHANNEL_WEIGHTS: `convex_fuse` redistributes a missing channel's weight
    #: across the rest, so with no skills data dense is effectively 0.714, not
    #: 0.5. Reporting the nominal value would mislabel the UI.
    weights_used: dict = field(default_factory=dict)

    @property
    def score_100(self) -> float:
        return round(self.raw * 100, 2)


@dataclass
class BatchStats:
    """Where the time went. Surfaced to the UI's debug strip, not decoration --
    it is how a regression in the dedup shows up as a number rather than a
    vague feeling that the page got slower."""

    candidates: int = 0
    dense_cache_hits: int = 0
    unique_prefixes: int = 0
    embedded: int = 0

    def to_dict(self) -> dict:
        return {"candidates": self.candidates,
                "dense_cache_hits": self.dense_cache_hits,
                "unique_prefixes": self.unique_prefixes,
                "embedded": self.embedded}


class DenseCache:
    """A bounded (jd_key, token_prefix) -> cosine memo, plus a bounded
    chunk_id -> embedding memo for chunked candidates.

    The first is keyed on the *truncated token ids*, not the text, so two
    documents that the model cannot tell apart share an entry. The second is
    keyed on `chunk_id` alone: a chunk id already encodes its own content (via
    the enabled bullet ids that built it -- see `resume.chunking`), so the
    same id always means the same text. Both bounded because a long editing
    session would otherwise grow them without limit.
    """

    def __init__(self, maxsize: int = 4096) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[tuple, float] = OrderedDict()
        self._chunks: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, key: tuple) -> float | None:
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: tuple, value: float) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def get_chunk(self, chunk_id: str) -> np.ndarray | None:
        if chunk_id not in self._chunks:
            return None
        self._chunks.move_to_end(chunk_id)
        return self._chunks[chunk_id]

    def put_chunk(self, chunk_id: str, vector: np.ndarray) -> None:
        self._chunks[chunk_id] = vector
        self._chunks.move_to_end(chunk_id)
        while len(self._chunks) > self.maxsize:
            self._chunks.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


def _ensure_chunks_embedded(
        chunk_sets: list[tuple[tuple[str, str], ...]],
        cache: DenseCache | None) -> dict[str, np.ndarray] | None:
    """One batched `embed()` call for every distinct chunk across every
    *chunk set passed in this call* that is not already cached -- not one
    call per candidate.

    This is the whole reason `score_many` can afford a 145-variant sweep in
    the first place ("one batch of twenty-one costs far less than twenty-one
    calls of one", see the module docstring); a first version of this
    function called `embed()` once per candidate and measured *slower* than
    the plain whole-document path it was meant to replace on a real 15-bullet
    resume (1,473 ms vs 1,269 ms) -- entirely because of that, not because
    chunking itself is more expensive. Collecting every candidate's missing
    chunks first and embedding them in one call fixes it.

    `_dense_values` calls this **twice** -- once for the first chunked
    candidate alone, once for the rest batched together -- not because that
    optimisation stopped mattering, but because embedding the same chunk
    alongside different batch-mates is not bit-exact (see the comment at that
    call site). Splitting the call in two trades a little of this batching
    for a reproducibility guarantee the test suite depends on.
    """
    local: dict[str, np.ndarray] = {}
    missing: dict[str, str] = {}
    for chunks in chunk_sets:
        for chunk_id, text in chunks:
            if chunk_id in local or chunk_id in missing:
                continue
            vector = cache.get_chunk(chunk_id) if cache is not None else None
            if vector is not None:
                local[chunk_id] = vector
            else:
                missing[chunk_id] = text
    if missing:
        ids = list(missing)
        vectors = embed([missing[cid] for cid in ids])
        if vectors is None:
            return None
        for chunk_id, vector in zip(ids, vectors, strict=True):
            local[chunk_id] = vector
            if cache is not None:
                cache.put_chunk(chunk_id, vector)
    return local


def _pool_from_vectors(chunks: tuple[tuple[str, str], ...],
                       vectors_by_id: dict[str, np.ndarray]) -> np.ndarray | None:
    """Length-weighted mean of this candidate's chunk vectors, re-normalised
    to unit length so cosine similarity against the job description stays
    meaningful.

    Mean-pooling, not max-pooling: every other channel in this scorer
    (lexical, skills) already judges the *whole* document, and this keeps
    the dense channel consistent with that rather than picking a single
    "best" chunk -- which would be exactly the kind of invented, unvalidated
    weighting this scorer otherwise avoids. There is no ranking ground truth
    in this repository to choose between them empirically (see
    docs/literature-survey.md §9), so this follows the already-justified
    precedent instead of guessing.
    """
    if not chunks:
        return None
    weights = np.array([len(text) for _, text in chunks], dtype=np.float32)
    weights = weights / weights.sum()
    try:
        ordered = np.stack([vectors_by_id[cid] for cid, _ in chunks])
    except KeyError:
        return None
    pooled = (ordered * weights[:, None]).sum(axis=0, keepdims=True)
    norm = np.linalg.norm(pooled, axis=1, keepdims=True)
    return (pooled / np.clip(norm, 1e-9, None))[0]


def _dense_values(candidates: list[Candidate], jd_text: str,
                  cache: DenseCache | None,
                  stats: BatchStats) -> list[float | None]:
    """The dense channel for every candidate, or all-None if unavailable.

    `None` means "this channel could not be computed", which `convex_fuse`
    treats as genuinely unknown and redistributes -- not as zero.
    """
    none_result: list[float | None] = [None] * len(candidates)
    if not any(c.text.strip() for c in candidates):
        return none_result

    jd_vector = _jd_array(jd_text)
    if jd_vector is None:
        return none_result

    values: list[float | None] = [None] * len(candidates)

    # Chunked candidates (opted in via Candidate.chunks) use the pooled path;
    # everyone else keeps the original whole-text truncation-dedup path
    # below, completely unchanged.
    chunked = [i for i, c in enumerate(candidates) if c.chunks]
    flat = [i for i in range(len(candidates)) if i not in chunked]

    if chunked:
        # The FIRST chunked candidate is embedded in its own dedicated call,
        # separate from every other candidate's chunks -- not merged into one
        # big batch with them.
        #
        # This looks like it undoes the "one batched call" optimisation, but
        # it fixes a real bug: the same chunk embedded alongside DIFFERENT
        # batch-mates (i.e. in a differently-padded batch) is not bit-exact.
        # Measured directly: a document's own score, computed alone, differed
        # from the same document's score computed as the base of a 5-variant
        # `attribute()` sweep by a reproducible (not random) ~4.3e-08 at the
        # vector level -- identical across 10 repeated trials, so it is a
        # floating-point padding artefact, not noise. That is small enough to
        # never matter for a displayed score, but large enough to break
        # `test_score_equals_rescore_exactly`'s `==` (not approximate)
        # assertion, because `rescore()` always embeds a document's chunks
        # alone while `attribute()`'s sweep embeds the same document's chunks
        # alongside its own leave-one-out variants. Embedding candidate 0 (by
        # convention, always "the document as it currently stands" --
        # `attribute`/`score_batch` always put it first) in isolation makes
        # that score reproducible regardless of what else is being scored in
        # the same request. The remaining chunked candidates (variants) still
        # share one batched call among themselves, so the sweep stays fast.
        base_index = chunked[0]
        base_vectors = _ensure_chunks_embedded([candidates[base_index].chunks], cache)
        rest_vectors = (_ensure_chunks_embedded(
            [candidates[i].chunks for i in chunked[1:]], cache)
            if len(chunked) > 1 else {})
        vectors_by_id = None
        if base_vectors is not None and rest_vectors is not None:
            vectors_by_id = {**base_vectors, **rest_vectors}
        if vectors_by_id is not None:
            for index in chunked:
                vector = _pool_from_vectors(candidates[index].chunks, vectors_by_id)
                if vector is not None:
                    values[index] = max(0.0, float(np.dot(vector, jd_vector)))

    if not flat:
        return values

    texts = [candidates[i].text for i in flat]
    prefixes = truncated_ids(texts)
    if prefixes is None:          # model unavailable; degrade, do not fail
        return values if chunked else none_result

    jd_key = hash(jd_text)

    # Which distinct prefixes still need embedding, and who is waiting on each.
    pending: OrderedDict[tuple, list[int]] = OrderedDict()
    for local_i, prefix in zip(flat, prefixes, strict=True):
        key = (jd_key, prefix)
        hit = cache.get(key) if cache is not None else None
        if hit is not None:
            values[local_i] = hit
            stats.dense_cache_hits += 1
        else:
            pending.setdefault(prefix, []).append(local_i)

    stats.unique_prefixes = len(pending)
    if not pending:
        return values

    # One text per distinct prefix, in one call.
    unique_texts = [candidates[holders[0]].text for holders in pending.values()]
    stats.embedded = len(unique_texts)
    vectors = embed(unique_texts)
    if vectors is None:
        return values

    for (prefix, holders), vector in zip(pending.items(), vectors, strict=True):
        cosine = max(0.0, float(np.dot(vector, jd_vector)))
        if cache is not None:
            cache.put((jd_key, prefix), cosine)
        for index in holders:
            values[index] = cosine
    return values


def score_many(candidates: list[Candidate], jd_text: str, *,
               required_skills: list[str] | None = None,
               preferred_skills: list[str] | None = None,
               use_dense: bool = True,
               cache: DenseCache | None = None,
               stats: BatchStats | None = None) -> list[Scored]:
    """Score every candidate. Order matches the input.

    The invariant that makes this safe to put behind a UI, and which
    `tests/test_batch.py` asserts to float equality:

        score_many([Candidate(i, doc.as_text(), doc.skills())], jd)[0].raw
            == rescore(doc, jd)[0]
    """
    if not candidates:
        return []

    stats = stats if stats is not None else BatchStats()
    stats.candidates = len(candidates)
    required = required_skills or []
    preferred = preferred_skills or []
    corpus = list(background_corpus())

    if use_dense:
        dense_values = _dense_values(candidates, jd_text, cache, stats)
    else:
        dense_values = [None] * len(candidates)

    results: list[Scored] = []
    for candidate, dense_value in zip(candidates, dense_values, strict=True):
        raw, channels = _score_from_parts(
            dense_value, candidate.text, jd_text, candidate.skills,
            required, preferred, corpus)
        # Same function, same inputs, so this cannot disagree with `raw`; it
        # only recovers the effective weights that `_score_from_parts` drops.
        fused = convex_fuse(ChannelScores(**{k: channels.get(k)
                                             for k in ("dense", "lexical", "skills")}))
        results.append(Scored(id=candidate.id, raw=raw, channels=channels,
                              weights_used=dict(fused.weights_used)))
    return results
