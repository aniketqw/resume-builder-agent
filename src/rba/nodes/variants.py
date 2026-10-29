"""Rephrase-only bullet variants, scored against the real scorer.

Three separate jobs, deliberately kept apart because only one of them may
involve a model:

* **Generation** is the model's job and nothing else. One structured call per
  *project* returns K rewrites for every bullet in that project. Per bullet
  would be the obvious shape and is wrong: a ten-project resume with four
  bullets each is 40 requests instead of 10, and the measured budget
  (``evals/budget.md``) says tokens bind long before requests do — every extra
  call re-sends the 600-token system prompt for nothing.

* **Verification** is code (:func:`reject_reason`), never a prompt. The prompt
  asks for rephrasing only; this module *enforces* it, reusing the same
  helpers that police generated claims in :mod:`rba.nodes.verify`. A variant
  that smuggles in a number, a technology or an outcome the evidence does not
  support is rejected with a reason, not quietly shown to the user.

* **Scoring** is :func:`rba.resume.attribution.rescore` and nothing else. The
  model is never asked what a rewrite is worth: the variant is substituted
  into the document, the document is re-scored, and the gain reported is the
  measured difference. An estimated delta is the one number this project may
  not print.

Cost. ``rescore`` embeds the whole document each time -- 291 ms measured on a
real one-page resume, 36 ms on the small three-bullet document the tests use --
so the baseline is computed once and each surviving variant costs exactly one
more embedding. At the one-page figure that is roughly 14 s for 45 variants
(measured here: 0.14 s for three variants on the small document). Rejected
variants are never scored, which is why verification runs first. That is fine
for a one-off background job behind a spinner; it would not be fine in a
keystroke handler. Disabled bullets are skipped entirely -- they are absent
from ``as_text()``, so every variant of one would measure a gain of zero.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..cache import make_key
from ..config import Config
from ..groq_client import GroqClient
from ..prompts import render
from ..resume.attribution import rescore
from ..resume.document import Bullet, ResumeDoc, _skill_candidates
from .verify import (
    _IMPACT,
    _number_supported,
    filter_tech,
    normalise,
    numbers_in,
    quote_relevant,
)

log = logging.getLogger(__name__)

PROMPT = "variant.v1"
MAX_EVIDENCE_CHARS = 600      # per bullet; keeps one project call near 1.5k tokens

# Punctuation that hides a technology name from the alias matcher: it looks for
# whole words, so "Redis-cached" and "Postgres." would otherwise slip past.
# `+` and `#` survive, because "c++" and "c#" are technology names.
_WORD_BREAK = re.compile(r"[^A-Za-z0-9+#]+")


# -- what comes back ------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    """One candidate rewrite of one bullet.

    ``gain`` is in score points (the 0-100 display scale), measured, and is
    0.0 for a rejected variant because a rejected variant is never scored.
    """

    bullet_id: str
    text: str
    gain: float = 0.0
    verified: bool = False
    rejected_reason: str = ""

    def to_dict(self) -> dict:
        return {"bullet_id": self.bullet_id, "text": self.text, "gain": self.gain,
                "verified": self.verified, "rejected_reason": self.rejected_reason}


@dataclass
class VariantSet:
    """``bullet_id -> [Variant]``, kept ones first and ranked by measured gain."""

    by_bullet: dict[str, list[Variant]] = field(default_factory=dict)
    baseline: float = 0.0      # the unmodified document's score, in points

    def best(self, bullet_id: str) -> Variant | None:
        """The highest-gaining surviving rewrite, or None if all were rejected."""
        for variant in self.by_bullet.get(bullet_id, []):
            if variant.verified:
                return variant
        return None

    def kept(self) -> list[Variant]:
        return [v for vs in self.by_bullet.values() for v in vs if v.verified]

    def rejected(self) -> list[Variant]:
        return [v for vs in self.by_bullet.values() for v in vs if not v.verified]

    def to_dict(self) -> dict:
        return {"baseline": self.baseline,
                "variants": {bid: [v.to_dict() for v in vs]
                             for bid, vs in self.by_bullet.items()}}


# -- projects -------------------------------------------------------------
@dataclass(frozen=True)
class Project:
    """One unit of fan-out: a dated entry, or a section's own loose bullets."""

    key: str
    title: str
    tech: tuple[str, ...]
    bullets: tuple[Bullet, ...]


def projects(doc: ResumeDoc) -> list[Project]:
    """Group enabled bullets into the units that each get one model call."""
    out: list[Project] = []
    for section in doc.sections:
        for index, entry in enumerate(section.entries):
            bullets = tuple(entry.enabled_bullets())
            if bullets:
                out.append(Project(key=f"{section.title}#{index}",
                                   title=entry.heading() or section.title,
                                   tech=tuple(entry.tech), bullets=bullets))
        loose = tuple(b for b in section.bullets if b.enabled)
        if loose:
            out.append(Project(key=section.title, title=section.title,
                               tech=(), bullets=loose))
    return out


# -- enforcement ----------------------------------------------------------
def reject_reason(text: str, bullet: Bullet, tech: tuple[str, ...] = ()) -> str:
    """Why this rewrite may not be shown, or "" if it is an honest rephrasing.

    The reference a variant is judged against is the *original* bullet plus
    its evidence plus the project's declared technologies -- everything the
    bullet was already entitled to say, and nothing more.
    """
    candidate = " ".join(text.split())
    if not candidate:
        return "empty"
    if normalise(candidate).strip() == normalise(bullet.text).strip():
        return "identical to the original"

    reference = "\n".join([bullet.text, bullet.evidence, " ".join(tech)])

    # `numbers_in` already masks versions and years, so "Python 3.11" is not
    # mistaken for a metric. One number finder, shared with claim verification.
    unsupported = [n for n in numbers_in(candidate)
                   if not _number_supported(n, reference)]
    if unsupported:
        return "introduces number(s) absent from the evidence: " + ", ".join(unsupported)

    # Only recognised skill names are checked. Scanning prose for
    # "technology-shaped" tokens was measured in verify.py to reject 39 of 48
    # honest claims over words like "fallback." -- the same trap applies here.
    #
    # Both sides are resolved to canonical names first, because `filter_tech`
    # compares surface strings: "Postgres" in the evidence canonicalises to
    # "postgresql", which does not appear in the evidence verbatim, and the
    # check would reject the very word the bullet already used.
    known = set(_tech_mentioned(reference))
    new_tech = [t for t in _tech_mentioned(candidate) if t not in known]
    _keep, drop = filter_tech(new_tech, reference)
    if drop:
        return "names technology absent from the evidence: " + ", ".join(drop)

    if _IMPACT.search(candidate) and not _IMPACT.search(reference):
        return "asserts an outcome or metric the evidence does not contain"

    # A "rewrite" sharing no content word with the original is a different
    # bullet, however true it may be.
    if not quote_relevant(candidate, reference):
        return "does not restate the original bullet"
    return ""


def _tech_mentioned(text: str) -> list[str]:
    """Canonical names of recognised technologies named in this text."""
    return _skill_candidates(f" {_WORD_BREAK.sub(' ', text)} ")


# -- scoring --------------------------------------------------------------
def score_variants(doc: ResumeDoc, jd_text: str,
                   proposals: dict[str, list[str]], *,
                   required_skills: list[str] | None = None,
                   preferred_skills: list[str] | None = None,
                   use_dense: bool = True) -> VariantSet:
    """Verify, measure and rank proposed rewrites. No model involvement.

    Pure Python and separately testable: the generation step hands over plain
    strings, and everything that decides what the user sees happens here.
    """
    base_raw, _channels = rescore(doc, jd_text, required_skills=required_skills,
                                  preferred_skills=preferred_skills,
                                  use_dense=use_dense)
    tech_by_bullet = {b.id: p.tech for p in projects(doc) for b in p.bullets}

    result = VariantSet(baseline=round(base_raw * 100, 4))
    for bullet_id, texts in proposals.items():
        bullet = doc.by_id(bullet_id)
        if bullet is None:
            log.warning("variants proposed for unknown bullet %s", bullet_id)
            continue
        tech = tech_by_bullet.get(bullet_id, ())
        kept: list[Variant] = []
        dropped: list[Variant] = []
        seen: set[str] = set()
        for raw_text in texts:
            text = " ".join(raw_text.split())
            if text in seen:
                continue
            seen.add(text)
            reason = reject_reason(text, bullet, tech)
            if reason:
                dropped.append(Variant(bullet_id, text, 0.0, False, reason))
                continue
            kept.append(Variant(bullet_id, text,
                                _measure_gain(doc, bullet, text, jd_text,
                                              base_raw, required_skills,
                                              preferred_skills, use_dense),
                                True, ""))
        kept.sort(key=lambda v: -v.gain)
        result.by_bullet[bullet_id] = kept + dropped
    return result


def _measure_gain(doc: ResumeDoc, bullet: Bullet, text: str, jd_text: str,
                  base_raw: float, required_skills: list[str] | None,
                  preferred_skills: list[str] | None, use_dense: bool) -> float:
    """Substitute, re-score, restore. The gain is a measurement, not a guess.

    The bullet is mutated in place rather than deep-copying the document:
    `as_text()` and `skills()` both read live objects, so a copy would have to
    be exact anyway, and the restore is in a `finally`.
    """
    original = bullet.text
    try:
        bullet.text = text
        raw, _channels = rescore(doc, jd_text, required_skills=required_skills,
                                 preferred_skills=preferred_skills,
                                 use_dense=use_dense)
    finally:
        bullet.text = original
    return round((raw - base_raw) * 100, 4)


# -- generation -----------------------------------------------------------
class _BulletRewrites(BaseModel):
    index: int = Field(description="1-based index of the bullet, exactly as given")
    variants: list[str] = Field(description="Distinct rephrasings of that bullet")


class _ProjectVariants(BaseModel):
    bullets: list[_BulletRewrites] = Field(
        description="One entry per bullet in the project, in the order given")


def _bullet_block(bullets: tuple[Bullet, ...]) -> str:
    lines = []
    for index, bullet in enumerate(bullets, start=1):
        evidence = " ".join(bullet.evidence.split())[:MAX_EVIDENCE_CHARS]
        lines.append(f"[{index}] BULLET: {bullet.text}")
        lines.append(f"    EVIDENCE: {evidence or '(none supplied)'}")
    return "\n".join(lines)


async def propose_for_project(client: GroqClient, project: Project, *,
                              keywords: list[str] | None = None, k: int = 3,
                              tier: str = "small") -> dict[str, list[str]]:
    """One structured call for a whole project. Returns bullet_id -> rewrites.

    The model answers by index rather than by bullet id: a 12-hex id is
    something a model can plausibly mistype, and a mistyped id would silently
    attach a rewrite to the wrong line.
    """
    messages, version = render(
        PROMPT, project=project.title,
        tech=", ".join(project.tech) or "(not declared)",
        keywords=", ".join(keywords or []) or "(none)",
        bullets=_bullet_block(project.bullets), k=k)
    key = make_key(PROMPT, version, k, project.key,
                   *(b.id for b in project.bullets))
    obj, _usage = await client.structured(
        messages, _ProjectVariants, node="rephrase_bullets", tier=tier,
        cache_key=key)

    out: dict[str, list[str]] = {}
    for rewrite in obj.bullets:
        position = rewrite.index - 1
        if not 0 <= position < len(project.bullets):
            log.warning("variant call returned index %d for a %d-bullet project",
                        rewrite.index, len(project.bullets))
            continue
        bullet_id = project.bullets[position].id
        out.setdefault(bullet_id, []).extend(t for t in rewrite.variants if t.strip())
    return out


def project_for(doc: ResumeDoc, bullet_id: str) -> Project | None:
    """The fan-out unit a given bullet belongs to, for its tech and title."""
    for project in projects(doc):
        if any(b.id == bullet_id for b in project.bullets):
            return project
    return None


async def improve_bullet(doc: ResumeDoc, jd_text: str, bullet_id: str, *,
                         client: GroqClient, config: Config | None = None,
                         required_skills: list[str] | None = None,
                         preferred_skills: list[str] | None = None,
                         k: int = 5, tier: str = "small",
                         use_dense: bool = True) -> VariantSet:
    """Rewrite one line, on demand, and measure every rewrite.

    The whole-document pass asks for three rewrites of every bullet at once.
    This spends an entire call on a single line, so it asks for more of them --
    the usual reason somebody wants this is that the line is *costing* them
    points, and a line that is hurting needs more than three attempts to find
    one that helps.

    Everything after generation is the shared path: the same `reject_reason`
    guard and the same real re-scoring, so a gain shown here means exactly what
    it means everywhere else.
    """
    cfg = config or Config.load()
    tier = cfg.get("routing.rephrase_bullets", tier)
    bullet = doc.by_id(bullet_id)
    if bullet is None:
        raise KeyError(f"no bullet {bullet_id!r}")

    parent = project_for(doc, bullet_id)
    focused = Project(key=f"{(parent.key if parent else 'bullet')}!{bullet_id}",
                      title=parent.title if parent else "",
                      tech=parent.tech if parent else (),
                      bullets=(bullet,))
    proposals = await propose_for_project(
        client, focused,
        keywords=list(required_skills or []) + list(preferred_skills or []),
        k=k, tier=tier)
    return score_variants(doc, jd_text, proposals,
                          required_skills=required_skills,
                          preferred_skills=preferred_skills,
                          use_dense=use_dense)


async def generate_variants(doc: ResumeDoc, jd_text: str, *, client: GroqClient,
                            config: Config | None = None,
                            required_skills: list[str] | None = None,
                            preferred_skills: list[str] | None = None,
                            k: int = 3, concurrency: int = 2,
                            tier: str = "small",
                            use_dense: bool = True) -> VariantSet:
    """Generate, verify and measure rephrasings for every enabled bullet.

    One model call per project, bounded by `concurrency` because the binding
    limit is 8,000 tokens per minute, not CPU. A project whose call fails is
    logged and skipped; the rest of the document still gets its variants.
    """
    cfg = config or Config.load()
    tier = cfg.get("routing.rephrase_bullets", tier)
    keywords = list(required_skills or []) + list(preferred_skills or [])
    groups = projects(doc)
    sem = asyncio.Semaphore(concurrency)

    async def one(project: Project) -> dict[str, list[str]]:
        async with sem:
            try:
                return await propose_for_project(client, project,
                                                 keywords=keywords, k=k, tier=tier)
            except Exception as exc:          # one bad project must not sink the job
                log.warning("variant generation failed for %s: %s", project.key, exc)
                return {}

    proposals: dict[str, list[str]] = {}
    for group in await asyncio.gather(*(one(p) for p in groups)):
        for bullet_id, texts in group.items():
            proposals.setdefault(bullet_id, []).extend(texts)

    result = score_variants(doc, jd_text, proposals,
                            required_skills=required_skills,
                            preferred_skills=preferred_skills,
                            use_dense=use_dense)
    log.info("variants: %d calls, %d proposed, %d kept, %d rejected",
             len(groups), sum(len(v) for v in proposals.values()),
             len(result.kept()), len(result.rejected()))
    return result
