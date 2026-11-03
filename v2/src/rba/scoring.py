"""Deterministic ATS scoring. No model is involved anywhere in this module.

The 100 points split as:

===============  ====  ====================================================
component        pts   what it measures
===============  ====  ====================================================
keywords          44   coverage of the role's weighted keyword set
role_fit          18   cosine similarity to the role's keyword profile
numbers           12   how many bullets carry a real quantity
sections          10   the sections a parser and a screener both expect
contact            6   email, GitHub, phone
length             6   word count inside the role's target band
verbs              4   bullets opening with a strong action verb
===============  ====  ====================================================

Being deterministic is the point. The improvement planner quotes an "ATS gain"
for every suggested step, and that number is only honest if the same resume
always scores the same. It also makes the whole rubric unit-testable.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from .state import Resume, RoleScore

ROLES_PATH = Path(__file__).parent / "roles.yaml"

WEIGHTS = {
    "keywords": 44.0,
    "role_fit": 18.0,
    "numbers": 12.0,
    "sections": 10.0,
    "contact": 6.0,
    "length": 6.0,
    "verbs": 4.0,
}

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE = re.compile(r"(?<!\d)(\+?\d[\d\s().-]{7,}\d)(?!\d)")
GITHUB = re.compile(r"(?i)github\.com/[\w-]+")
# A quantity that means something on a resume: not a year, not a version.
QUANTITY = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?\s*(?:%|x\b|ms\b|s\b|k\b|m\b)?")
YEAR = re.compile(r"\b(19|20)\d{2}\b")
MEASURE_TAG = re.compile(r"\[MEASURE:[^\]]*\]")


@dataclass(frozen=True)
class Role:
    key: str
    title: str
    headline: str
    keywords: dict[str, float]
    required_sections: list[str]
    target_words: tuple[int, int]
    action_verbs: set[str]


@lru_cache(maxsize=1)
def load_roles(path: str | None = None) -> dict[str, Role]:
    data = yaml.safe_load(Path(path or ROLES_PATH).read_text())
    defaults = data.get("defaults", {})
    verbs: set[str] = set()
    for line in defaults.get("action_verbs", []):
        verbs.update(v.strip().lower() for v in line.split(",") if v.strip())
    band = defaults.get("target_words", [450, 850])
    roles = {}
    for key, spec in data["roles"].items():
        roles[key] = Role(
            key=key,
            title=spec.get("title", key),
            headline=spec.get("headline", ""),
            keywords={k.lower(): float(v) for k, v in spec["keywords"].items()},
            required_sections=list(spec.get("required_sections",
                                            defaults.get("required_sections", []))),
            target_words=tuple(spec.get("target_words", band)),
            action_verbs=verbs,
        )
    return roles


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9+#/.\- ]+", " ", text.lower())


def _contains(term: str, text: str) -> bool:
    """Whole-term match, so 'go' does not match 'going' and 'r' matches alone."""
    return re.search(rf"(?<![\w/+#-]){re.escape(term)}(?![\w/+#-])", text) is not None


# -- components -----------------------------------------------------------
def score_keywords(text: str, role: Role) -> tuple[float, list[str], list[str]]:
    """Weighted keyword coverage, saturating.

    Saturating matters: a resume that says "agent" twenty times is not twice as
    relevant as one that says it twice, and rewarding repetition would push the
    generator toward keyword stuffing.
    """
    norm = _normalise(text)
    matched, missing, got, total = [], [], 0.0, 0.0
    for term, weight in role.keywords.items():
        total += weight
        if _contains(term, norm):
            got += weight
            matched.append(term)
        else:
            missing.append(term)
    coverage = got / total if total else 0.0
    return WEIGHTS["keywords"] * coverage, matched, missing


def score_role_fit(text: str, role: Role) -> float:
    """Cosine similarity between the resume's term frequencies and the role profile."""
    norm = _normalise(text)
    counts = {t: len(re.findall(rf"(?<![\w/+#-]){re.escape(t)}(?![\w/+#-])", norm))
              for t in role.keywords}
    if not any(counts.values()):
        return 0.0
    # Dampen repetition so emphasis counts but stuffing does not.
    resume_vec = {t: math.log1p(c) for t, c in counts.items()}
    role_vec = role.keywords
    dot = sum(resume_vec[t] * role_vec[t] for t in role_vec)
    nr = math.sqrt(sum(v * v for v in resume_vec.values()))
    nq = math.sqrt(sum(v * v for v in role_vec.values()))
    return WEIGHTS["role_fit"] * (dot / (nr * nq) if nr and nq else 0.0)


def bullets_of(resume: Resume) -> list[str]:
    return [b for bullets in resume.sections.values() for b in bullets]


def score_numbers(bullets: list[str]) -> float:
    """Share of bullets carrying a real quantity, capped at a realistic target.

    Demanding a number in every bullet would be a licence to invent them — the
    exact failure this project exists to prevent. A `[MEASURE: ...]` placeholder
    scores nothing, so an unmeasured claim is never rewarded.
    """
    if not bullets:
        return 0.0
    quantified = 0
    for b in bullets:
        stripped = MEASURE_TAG.sub("", b)
        candidates = [m.group(0) for m in QUANTITY.finditer(YEAR.sub(" ", stripped))]
        if any(c.strip() for c in candidates):
            quantified += 1
    share = quantified / len(bullets)
    target = 0.5                      # half the bullets quantified is strong
    return WEIGHTS["numbers"] * min(1.0, share / target)


def score_sections(resume: Resume, role: Role) -> float:
    present = {s.lower() for s in resume.sections}
    if not role.required_sections:
        return WEIGHTS["sections"]
    hits = sum(1 for s in role.required_sections if s.lower() in present)
    return WEIGHTS["sections"] * (hits / len(role.required_sections))


def score_contact(text: str) -> float:
    found = sum([bool(EMAIL.search(text)), bool(GITHUB.search(text)),
                 bool(PHONE.search(text))])
    return WEIGHTS["contact"] * (found / 3)


def score_length(text: str, role: Role) -> float:
    words = len(text.split())
    low, high = role.target_words
    if low <= words <= high:
        return WEIGHTS["length"]
    if words < low:
        # Proportional for short resumes: an almost-empty one should score
        # almost nothing, while 300 of a 450 target still clearly beats 50.
        return WEIGHTS["length"] * (words / low)
    # Absolute falloff for long ones, zero at twice the band width past it.
    span = max(1, high - low) * 2
    return WEIGHTS["length"] * max(0.0, 1.0 - (words - high) / span)


def score_verbs(bullets: list[str], role: Role) -> float:
    if not bullets:
        return 0.0
    strong = 0
    for b in bullets:
        first = re.sub(r"^[^A-Za-z]+", "", b).split(" ")[0].lower().rstrip(".,:;")
        if first in role.action_verbs:
            strong += 1
    return WEIGHTS["verbs"] * (strong / len(bullets))


# -- top level ------------------------------------------------------------
def score_resume(resume: Resume, role_key: str, roles: dict[str, Role] | None = None,
                 contact_text: str = "") -> RoleScore:
    roles = roles or load_roles()
    role = roles[role_key]
    text = "\n".join([resume.as_text(), contact_text])
    bullets = bullets_of(resume)

    kw, matched, missing = score_keywords(text, role)
    components = {
        "keywords": round(kw, 2),
        "role_fit": round(score_role_fit(text, role), 2),
        "numbers": round(score_numbers(bullets), 2),
        "sections": round(score_sections(resume, role), 2),
        "contact": round(score_contact(text), 2),
        "length": round(score_length(text, role), 2),
        "verbs": round(score_verbs(bullets, role), 2),
    }
    return RoleScore(
        role=role_key,
        total=round(sum(components.values()), 2),
        components=components,
        matched_keywords=sorted(matched),
        # Highest-weight gaps first: that ordering drives the improvement plan.
        missing_keywords=sorted(missing, key=lambda t: -role.keywords[t]),
    )


def max_gain(role_key: str, missing: list[str], roles: dict[str, Role] | None = None,
             limit: int = 5) -> float:
    """Points recoverable by covering the top `limit` missing keywords.

    This is what the planner reports as the expected ATS gain for a step, so it
    has to come from the same arithmetic as the score itself.
    """
    roles = roles or load_roles()
    role = roles[role_key]
    total = sum(role.keywords.values())
    if not total:
        return 0.0
    recoverable = sum(role.keywords.get(t, 0.0) for t in missing[:limit])
    return round(WEIGHTS["keywords"] * recoverable / total, 2)
