"""The skills channel: weighted Jaccard over normalised skill ids.

No model runs here. Given two sets of skills this is arithmetic a reader can
verify by hand, which is the point — it is the one channel that can be fully
explained to a candidate who asks why they scored what they scored.

Required skills weigh more than preferred ones, mirroring how real systems
separate the two: Greenhouse Talent Filtering has Required (AND) and Preferred
(OR) terms, and Workday grades on "basic" versus "preferred" qualifications.

Surface forms are canonicalised first via :mod:`rba.ats.normalise`, so "JS",
"JavaScript" and "ECMAScript" count once. That module explains why a curated
alias table beat the ESCO taxonomy here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .normalise import canonical

REQUIRED_WEIGHT = 1.0
PREFERRED_WEIGHT = 0.5


@dataclass(frozen=True)
class SkillMatch:
    """A breakdown a human can audit line by line."""

    matched_required: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    matched_preferred: list[str] = field(default_factory=list)
    missing_preferred: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def summary(self) -> str:
        req_total = len(self.matched_required) + len(self.missing_required)
        pref_total = len(self.matched_preferred) + len(self.missing_preferred)
        parts = [f"{len(self.matched_required)} of {req_total} required"]
        if pref_total:
            parts.append(f"{len(self.matched_preferred)} of {pref_total} preferred")
        return ", ".join(parts)


def _norm(skill: str) -> str:
    return canonical(skill)


def weighted_jaccard(resume_skills: set[str] | list[str],
                     required: set[str] | list[str],
                     preferred: set[str] | list[str] | None = None) -> SkillMatch:
    """Coverage of the job's skills by the resume's, weighted by importance.

    This is coverage rather than symmetric Jaccard on purpose. A candidate is
    not penalised for knowing things the job did not ask about — symmetric
    Jaccard would punish a broad resume, which is the opposite of useful.
    """
    have = {_norm(s) for s in resume_skills if _norm(s)}
    req = {_norm(s) for s in required if _norm(s)}
    pref = {_norm(s) for s in (preferred or []) if _norm(s)} - req

    matched_req = sorted(req & have)
    missing_req = sorted(req - have)
    matched_pref = sorted(pref & have)
    missing_pref = sorted(pref - have)

    denominator = REQUIRED_WEIGHT * len(req) + PREFERRED_WEIGHT * len(pref)
    if denominator <= 0:
        # The job description listed no skills. That is unknown, not zero —
        # the caller drops this channel and redistributes its weight.
        return SkillMatch(score=0.0)

    numerator = (REQUIRED_WEIGHT * len(matched_req)
                 + PREFERRED_WEIGHT * len(matched_pref))
    return SkillMatch(
        matched_required=matched_req,
        missing_required=missing_req,
        matched_preferred=matched_pref,
        missing_preferred=missing_pref,
        score=max(0.0, min(1.0, numerator / denominator)),
    )


def has_skills(required: set[str] | list[str],
               preferred: set[str] | list[str] | None = None) -> bool:
    """Whether the skills channel is computable at all."""
    return bool({_norm(s) for s in required if _norm(s)}
                | {_norm(s) for s in (preferred or []) if _norm(s)})
