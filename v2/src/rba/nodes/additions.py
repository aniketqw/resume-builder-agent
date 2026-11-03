"""Proposing points a résumé could add, without repeating what it already says.

The third version of a résumé: after the author's own text and after grounding
it against the repositories, what *else* does the evidence support that is not
on the page yet?

Two rules make this honest rather than padding.

**Nothing may be added that the repository does not show.** Every candidate is
checked against the same evidence bundle the grounding pass uses, by the same
code, so an addition is held to exactly the standard an existing line is.

**Nothing may restate a point already there.** A résumé with two lines saying
the same thing in different words is worse than one line, and the score it
earns is illusory. When a candidate *is* a restatement it is not thrown away --
it is turned into a proposed rewrite of the line it duplicates, because if a
better wording exists for something already claimed, the right move is to
improve that line rather than add a second one.

Duplicate detection is algorithmic and calibrated, not a model's opinion.
Measured on real bullets paired with hand-written rewordings of themselves:

    same claim, reworded   cosine 0.846-0.972   word overlap 0.636-0.750
    genuinely different    cosine 0.359-0.739   word overlap 0.000-0.222

A gap of +0.106 on cosine and +0.414 on overlap, so the thresholds sit at the
midpoints rather than anywhere convenient. Either signal firing is enough: a
missed duplicate costs the user a repeated line, which is worse than a
suppressed suggestion they can still reach through the rewrite control.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ..cache import make_key
from ..config import Config
from ..groq_client import GroqClient
from ..prompts import render
from ..resume.attribution import rescore
from ..resume.document import Bullet, Entry, ResumeDoc
from ..resume.grounding import unsupported_claims
from ..resume.matching import PROJECT_SECTIONS

log = logging.getLogger(__name__)

PROMPT = "addition.v1"

# Midpoints of the measured gaps. See the module docstring, and
# `scripts/calibrate_novelty.py` to reproduce them on other data.
DUPLICATE_COSINE = 0.79
DUPLICATE_OVERLAP = 0.43

_WORD = re.compile(r"[a-z0-9][a-z0-9+#.-]*")
STOPWORDS = {"a", "an", "the", "and", "or", "with", "of", "to", "for", "in",
             "on", "that", "it", "its", "by", "from", "as", "at", "is", "was",
             "then", "this", "these", "their", "into", "over", "per"}

ADDED = "added"            # new, grounded, and says something not already said
DUPLICATE = "duplicate"    # restates an existing point -> offer as a rewrite
UNSUPPORTED = "unsupported"  # the repository does not show it


@dataclass(frozen=True)
class Addition:
    """One proposed extra point, and what became of it."""

    project_key: str
    text: str
    status: str
    gain: float = 0.0
    duplicate_of: str = ""      # bullet id it restates, when DUPLICATE
    similarity: float = 0.0
    reason: str = ""

    @property
    def usable(self) -> bool:
        return self.status == ADDED and self.gain > 0

    def to_dict(self) -> dict:
        return {"project_key": self.project_key, "text": self.text,
                "status": self.status, "gain": self.gain,
                "duplicate_of": self.duplicate_of,
                "similarity": self.similarity, "reason": self.reason}


@dataclass
class AdditionSet:
    items: list[Addition] = field(default_factory=list)
    baseline: float = 0.0

    def added(self) -> list[Addition]:
        return sorted([a for a in self.items if a.status == ADDED],
                      key=lambda a: -a.gain)

    def duplicates(self) -> list[Addition]:
        return [a for a in self.items if a.status == DUPLICATE]

    def rejected(self) -> list[Addition]:
        return [a for a in self.items if a.status == UNSUPPORTED]

    def for_project(self, key: str) -> list[Addition]:
        return [a for a in self.added() if a.project_key == key]

    def to_dict(self) -> dict:
        return {"baseline": self.baseline,
                "items": [a.to_dict() for a in self.items]}


def content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower())
            if w not in STOPWORDS and len(w) > 2}


def word_overlap(a: str, b: str) -> float:
    """Shared content words over the smaller set.

    Normalising by the *smaller* side on purpose: a short candidate wholly
    contained in a longer existing line is a duplicate, and dividing by the
    union would hide that behind the longer line's extra words.
    """
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def find_duplicate(text: str, existing: list[Bullet],
                   vectors: dict[str, np.ndarray] | None = None
                   ) -> tuple[Bullet | None, float]:
    """The existing point this restates, if any, and how close it is."""
    best: Bullet | None = None
    best_score = 0.0
    candidate_vector = (vectors or {}).get(text)
    for bullet in existing:
        overlap = word_overlap(text, bullet.text)
        cosine = 0.0
        other = (vectors or {}).get(bullet.text)
        if candidate_vector is not None and other is not None:
            cosine = float(np.dot(candidate_vector, other))
        score = max(cosine, overlap * (DUPLICATE_COSINE / DUPLICATE_OVERLAP))
        if (cosine >= DUPLICATE_COSINE or overlap >= DUPLICATE_OVERLAP) \
                and score > best_score:
            best, best_score = bullet, max(cosine, overlap)
    return best, round(best_score, 4)


# -- generation -----------------------------------------------------------
class _Additions(BaseModel):
    points: list[str] = Field(description="New resume points, grounded in the evidence")


@dataclass(frozen=True)
class ProjectContext:
    key: str
    title: str
    tech: tuple[str, ...]
    bullets: tuple[Bullet, ...]
    evidence: str


def contexts(doc: ResumeDoc, evidence_by_key: dict[str, str]) -> list[ProjectContext]:
    out: list[ProjectContext] = []
    for section in doc.sections:
        if section.title.upper() not in PROJECT_SECTIONS:
            continue
        for index, entry in enumerate(section.entries):
            key = f"{section.title}#{index}"
            evidence = evidence_by_key.get(key, "")
            if evidence.strip():
                out.append(ProjectContext(key, entry.heading() or entry.title,
                                          tuple(entry.tech),
                                          tuple(entry.bullets), evidence))
    return out


async def propose_for_project(client: GroqClient, context: ProjectContext, *,
                              keywords: list[str] | None = None, k: int = 4,
                              tier: str = "small") -> list[str]:
    """One structured call. Returns candidate points as plain strings."""
    messages, version = render(
        PROMPT, project=context.title,
        tech=", ".join(context.tech) or "(not declared)",
        keywords=", ".join(keywords or []) or "(none)",
        existing="\n".join(f"- {b.text}" for b in context.bullets) or "(none)",
        evidence=context.evidence[:5000], k=k)
    key = make_key(PROMPT, version, k, context.key,
                   *(b.id for b in context.bullets))
    obj, _usage = await client.structured(
        messages, _Additions, node="propose_additions", tier=tier, cache_key=key)
    return [p.strip() for p in obj.points if p.strip()]


def _entry_for_key(doc: ResumeDoc, key: str) -> Entry | None:
    for section in doc.sections:
        for index, entry in enumerate(section.entries):
            if f"{section.title}#{index}" == key:
                return entry
    return None


def evaluate(doc: ResumeDoc, jd_text: str,
             proposals: dict[str, list[str]],
             evidence_by_key: dict[str, str], *,
             required_skills: list[str] | None = None,
             preferred_skills: list[str] | None = None,
             use_dense: bool = True) -> AdditionSet:
    """Verify, de-duplicate and measure. No model involvement past this point."""
    base_raw, _ = rescore(doc, jd_text, required_skills=required_skills,
                          preferred_skills=preferred_skills, use_dense=use_dense)
    result = AdditionSet(baseline=round(base_raw * 100, 4))

    existing = doc.enabled()
    candidate_texts = [t for texts in proposals.values() for t in texts]
    vectors: dict[str, np.ndarray] = {}
    if use_dense and candidate_texts:
        from ..ats.dense import embed

        corpus = [b.text for b in existing] + candidate_texts
        matrix = embed(corpus)
        if matrix is not None:
            vectors = {text: matrix[i] for i, text in enumerate(corpus)}

    for key, texts in proposals.items():
        entry = _entry_for_key(doc, key)
        if entry is None:
            continue
        evidence = evidence_by_key.get(key, "")
        for raw in texts:
            text = " ".join(raw.split())
            bad_numbers, bad_tech = unsupported_claims(text, evidence)
            if bad_numbers or bad_tech:
                result.items.append(Addition(
                    key, text, UNSUPPORTED,
                    reason="not shown in the repository: "
                           + ", ".join([*bad_numbers, *bad_tech])))
                continue

            twin, similarity = find_duplicate(text, existing, vectors)
            if twin is not None:
                result.items.append(Addition(
                    key, text, DUPLICATE, 0.0, twin.id, similarity,
                    reason="says what an existing point already says; offered "
                           "as a rewrite of that line instead"))
                continue

            gain = _measure(doc, entry, text, jd_text, base_raw,
                            required_skills, preferred_skills, use_dense)
            result.items.append(Addition(key, text, ADDED, gain))
    return result


def _measure(doc: ResumeDoc, entry: Entry, text: str, jd_text: str,
             base_raw: float, required: list[str] | None,
             preferred: list[str] | None, use_dense: bool) -> float:
    """The real gain: insert the point, score the document, take it out."""
    bullet = Bullet(text=text, origin="repo")
    entry.bullets.append(bullet)
    try:
        raw, _ = rescore(doc, jd_text, required_skills=required,
                         preferred_skills=preferred, use_dense=use_dense)
    finally:
        entry.bullets.remove(bullet)
    return round((raw - base_raw) * 100, 4)


async def generate_additions(doc: ResumeDoc, jd_text: str, *,
                             client: GroqClient,
                             evidence_by_key: dict[str, str],
                             config: Config | None = None,
                             required_skills: list[str] | None = None,
                             preferred_skills: list[str] | None = None,
                             k: int = 4, concurrency: int = 2,
                             tier: str = "small",
                             use_dense: bool = True) -> AdditionSet:
    """One call per project with evidence, then pure-Python evaluation."""
    cfg = config or Config.load()
    tier = cfg.get("routing.propose_additions", tier)
    keywords = list(required_skills or []) + list(preferred_skills or [])
    semaphore = asyncio.Semaphore(concurrency)

    async def one(context: ProjectContext) -> tuple[str, list[str]]:
        async with semaphore:
            try:
                return context.key, await propose_for_project(
                    client, context, keywords=keywords, k=k, tier=tier)
            except Exception as exc:        # one project must not sink the rest
                log.warning("additions failed for %s: %s", context.key, exc)
                return context.key, []

    results = await asyncio.gather(
        *(one(c) for c in contexts(doc, evidence_by_key)))
    proposals = {key: texts for key, texts in results if texts}
    return evaluate(doc, jd_text, proposals, evidence_by_key,
                    required_skills=required_skills,
                    preferred_skills=preferred_skills, use_dense=use_dense)
