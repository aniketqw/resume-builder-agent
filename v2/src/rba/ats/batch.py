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
    """A bounded (jd_key, token_prefix) -> cosine memo.

    Keyed on the *truncated token ids*, not the text, so two documents that the
    model cannot tell apart share an entry. Bounded because a long editing
    session would otherwise grow it without limit.
    """

    def __init__(self, maxsize: int = 4096) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[tuple, float] = OrderedDict()

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

    def __len__(self) -> int:
        return len(self._data)


def _dense_values(candidates: list[Candidate], jd_text: str,
                  cache: DenseCache | None,
                  stats: BatchStats) -> list[float | None]:
    """The dense channel for every candidate, or all-None if unavailable.

    `None` means "this channel could not be computed", which `convex_fuse`
    treats as genuinely unknown and redistributes -- not as zero.
    """
    none_result: list[float | None] = [None] * len(candidates)
    texts = [c.text for c in candidates]
    if not any(t.strip() for t in texts):
        return none_result

    jd_vector = _jd_array(jd_text)
    if jd_vector is None:
        return none_result

    prefixes = truncated_ids(texts)
    if prefixes is None:          # model unavailable; degrade, do not fail
        return none_result

    jd_key = hash(jd_text)
    values: list[float | None] = [None] * len(candidates)

    # Which distinct prefixes still need embedding, and who is waiting on each.
    pending: OrderedDict[tuple, list[int]] = OrderedDict()
    for index, prefix in enumerate(prefixes):
        key = (jd_key, prefix)
        hit = cache.get(key) if cache is not None else None
        if hit is not None:
            values[index] = hit
            stats.dense_cache_hits += 1
        else:
            pending.setdefault(prefix, []).append(index)

    stats.unique_prefixes = len(pending)
    if not pending:
        return values

    # One text per distinct prefix, in one call.
    unique_texts = [texts[holders[0]] for holders in pending.values()]
    stats.embedded = len(unique_texts)
    vectors = embed(unique_texts)
    if vectors is None:
        return none_result

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
