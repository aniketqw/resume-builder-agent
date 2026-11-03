"""A background corpus so BM25's IDF means something.

BM25 weights a term by how rare it is. Rarity is measured against a corpus —
and with only the resume and the job description to hand, every term looks
equally rare and the channel collapses. Measured on one pair: the same resume
scored **0.180** with no corpus and **0.476** against a 6-document background,
while an irrelevant resume stayed at 0.000.

These job descriptions are a *document-frequency background*, nothing more.
They are not training data, they carry no labels, and no part of the score is
fitted to them. They are deliberately spread across non-software roles too, so
that "Python" is rare in the corpus the way it is rare in the world, rather
than universal the way it would be in an all-engineering sample.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).parent / "data" / "jd_corpus.json"


@lru_cache(maxsize=1)
def background_corpus() -> tuple[str, ...]:
    """Job-description texts used as the IDF background."""
    if not DATA.is_file():
        return ()
    payload = json.loads(DATA.read_text())
    return tuple(job["text"] for job in payload.get("jobs", []))


@lru_cache(maxsize=1)
def families() -> tuple[str, ...]:
    if not DATA.is_file():
        return ()
    payload = json.loads(DATA.read_text())
    return tuple(job["family"] for job in payload.get("jobs", []))
