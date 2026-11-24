"""Bridge between free-text dataset resumes and v2's `ResumeDoc`.

v2's own `parse_resume_text` needs bullet markers and returns flat sections;
these datasets are mostly unbulleted prose. So each resume is split into
sentence-level "bullets" and grouped into entries of `ENTRY_SIZE`. Grouping
matters: the scorer's dense channel embeds one chunk per entry, and a single
giant entry would be truncated at 512 tokens, hiding most of the resume.

Everything scores through v2's own `score_many`, so numbers are what the
product would show (no re-implemented fusion).
"""
from __future__ import annotations

import re

from rba.ats.batch import Candidate, DenseCache, score_many
from rba.ats.fusion import ChannelScores, convex_fuse
from rba.resume.chunking import document_chunks
from rba.resume.document import Bullet, Entry, ResumeDoc, Section
from rba.resume.document import _skill_candidates

ENTRY_SIZE = 6
MAX_WORDS = 60
MIN_WORDS = 3
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def segment(text: str) -> list[str]:
    """Resume text -> bullet-sized strings (sentences / short lines)."""
    out: list[str] = []
    for line in re.split(r"\n+", text):
        for piece in _SENT.split(line.strip()):
            words = piece.split()
            for i in range(0, len(words), MAX_WORDS):
                chunk = words[i:i + MAX_WORDS]
                if len(chunk) >= MIN_WORDS:
                    out.append(" ".join(chunk))
    return out


def to_doc(text: str, entry_size: int = ENTRY_SIZE) -> ResumeDoc:
    items = segment(text)
    entries = []
    for i in range(0, len(items), entry_size):
        entries.append(Entry(
            title=f"Part {i // entry_size + 1}",
            bullets=[Bullet(text=t, source_repo="bench") for t in items[i:i + entry_size]]))
    return ResumeDoc(sections=[Section(title="Resume", entries=entries)])


def jd_skills(jd_text: str) -> list[str]:
    """Deterministic required-skill list: every vocabulary skill the JD names.

    No LLM. Uses v2's own alias table, so the skills channel is the same code
    path as in the product; it simply treats everything the JD mentions as
    required. Empty for JDs with no recognised skill (channel then drops out).
    """
    return sorted(set(_skill_candidates(jd_text)))


def candidate(doc: ResumeDoc, cid: str = "") -> Candidate:
    return Candidate(id=cid, text=doc.as_text(), skills=doc.skills(),
                     chunks=tuple(document_chunks(doc)))


CACHE = DenseCache(maxsize=200_000)


def score_docs(docs: list[ResumeDoc], jd_text: str, *, use_dense: bool = True):
    """Score documents against one JD in a single batch. Returns `Scored`s."""
    req = jd_skills(jd_text)
    return score_many([candidate(d, str(i)) for i, d in enumerate(docs)], jd_text,
                      required_skills=req, preferred_skills=[],
                      use_dense=use_dense, cache=CACHE)


FUSIONS = {
    "lexical": ("lexical",),
    "dense": ("dense",),
    "dense+lexical": ("dense", "lexical"),
    "full": ("dense", "lexical", "skills"),
}


def fuse(channels: dict, which: str) -> float | None:
    """Re-fuse a scored candidate's channels with a subset (ablation)."""
    use = FUSIONS[which]
    values = {k: channels.get(k) for k in use}
    if all(v is None for v in values.values()):
        return None
    return convex_fuse(ChannelScores(**{k: values.get(k) for k in
                                        ("dense", "lexical", "skills")})).raw
