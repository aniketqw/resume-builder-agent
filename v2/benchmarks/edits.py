"""Controlled bullet edits and their scoring.

Pure helpers shared by `run_edits.py` and the tests. The edits go through v2's
own `selection.apply_ops` (the same ops the web UI sends), so what is measured
is the product's add/remove path, not a re-implementation.
"""
from __future__ import annotations

import random
import re

from rba.ats.lexical import tokenize
from rba.resume.document import ResumeDoc
from rba.resume.selection import Op, apply_ops, copy_doc, entry_key

SECTION = "Resume"


def last_entry_key(doc: ResumeDoc) -> str:
    entries = doc.sections[0].entries
    return entry_key(SECTION, len(entries) - 1)


def add_bullets(doc: ResumeDoc, texts: list[str]) -> ResumeDoc:
    """`doc` plus `texts` appended to its last entry (new document)."""
    key = last_entry_key(doc)
    return apply_ops(doc, [Op("add_bullet", entry_key=key, text=t) for t in texts])


def disable_indices(doc: ResumeDoc, indices) -> ResumeDoc:
    """Copy of `doc` with the bullets at these `all_bullets()` positions off.

    Index-addressed on purpose: two identical sentences share a content id, so
    id addressing could switch off the wrong one.
    """
    work = copy_doc(doc)
    bullets = work.all_bullets()
    for i in indices:
        bullets[i].enabled = False
    return work


def nearest_length(pool: list[str], words: int, rng: random.Random,
                   exclude: set[str] = frozenset()) -> str:
    """A pool bullet whose word count is closest to `words` (ties random)."""
    cands = [b for b in pool if b not in exclude] or pool
    best = min(abs(len(b.split()) - words) for b in cands)
    return rng.choice([b for b in cands if abs(len(b.split()) - words) == best])


_STOP = set("the and for with that this are you our will have from your their "
            "all can has not but who any may per etc new use work able well "
            "such more other also into these those been being was were its".split())


def stuffed_bullet(jd_text: str, n_terms: int = 12) -> str:
    """A degenerate 'bullet' made only of the JD's most frequent terms.

    Used as a gaming control: if the scorer rewards this more than a genuine
    relevant bullet, it can be gamed by keyword stuffing.
    """
    counts: dict[str, int] = {}
    for tok in tokenize(jd_text):
        if len(tok) > 3 and tok not in _STOP and not re.fullmatch(r"[\d.]+", tok):
            counts[tok] = counts.get(tok, 0) + 1
    terms = [t for t, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return " ".join(terms[:n_terms])


def greedy_order(gains_fn, n: int, budget: int) -> list[int]:
    """Forward greedy: repeatedly add the index whose inclusion scores highest.

    `gains_fn(selected: list[int], candidates: list[int]) -> list[float]`
    returns the score of selected+[c] for each candidate c.
    """
    selected: list[int] = []
    remaining = list(range(n))
    while remaining and len(selected) < budget:
        scores = gains_fn(selected, remaining)
        best = max(range(len(remaining)), key=lambda i: (scores[i], -remaining[i]))
        selected.append(remaining.pop(best))
    return selected


def auc_planted(deltas: list[float], planted: list[bool], want_high: bool) -> float:
    """P(a planted bullet's delta ranks above/below an original bullet's).

    `want_high` True for planted-relevant (should help: higher delta),
    False for planted-irrelevant (should hurt: lower delta). 0.5 = chance.
    """
    pos = [d for d, p in zip(deltas, planted) if p]
    neg = [d for d, p in zip(deltas, planted) if not p]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            if a == b:
                wins += 0.5
            elif (a > b) == want_high:
                wins += 1.0
    return wins / (len(pos) * len(neg))
