"""Build the percentile reference set.

The reference set decides what a percentile *means*, so its composition is the
whole argument. An earlier version paired every synthetic resume with every job
description: 462 of its 484 pairs were cross-family nonsense (a nurse against a
Kubernetes role), so the bar was "better than a random pairing" and a mediocre
raw score of 0.6 came out at the 92nd percentile. Flattering and useless.

This builds **plausible** pairs instead: three resume strengths per job family,
each scored against its own family's job description. A percentile then means
"better than N% of plausible candidates for this kind of role", which is the
question someone applying for a job is actually asking.

Known limitation, stated rather than hidden: PJB (arXiv 2603.17386) found
performance heterogeneity across industry domains exceeds the gains from any
module upgrade, so one global distribution will misrank some sectors. The
reference set is small and synthetic; it fixes the *scale*, not the ranking.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rba.ats import (  # noqa: E402
    ChannelScores,
    bm25_similarity,
    convex_fuse,
    weighted_jaccard,
)
from rba.ats.calibrate import build_calibration, save_calibration  # noqa: E402
from rba.ats.corpus import background_corpus, families  # noqa: E402
from rba.ats.dense import cosine_similarity  # noqa: E402

# Three strengths per family. "strong" echoes the requirements, "weak" shares
# only generic professional language, "medium" sits between.
TEMPLATES = {
    "strong": ("Experienced practitioner. {req} Delivered projects end to end, "
               "owned systems in production and mentored colleagues."),
    "medium": ("Professional with exposure to {first}. Contributed to team "
               "projects, wrote documentation and supported delivery."),
    "weak": ("Motivated professional seeking new challenges. Strong "
             "communication, teamwork and problem solving. Delivered tasks "
             "to deadline and collaborated across departments."),
}


def required_skills(jd: str) -> list[str]:
    """The skills a job description lists as required."""
    if "Required:" not in jd:
        return []
    tail = jd.split("Required:")[-1].split("Preferred:")[0]
    tail = tail.split(".")[0]
    return [s.strip() for s in tail.split(",") if s.strip()]


def synthetic_resumes(jd: str) -> dict[str, str]:
    requirements = jd.split("Required:")[-1]
    first = requirements.split(",")[0].strip() or "the field"
    return {
        name: template.format(req=requirements, first=first)
        for name, template in TEMPLATES.items()
    }


# How much of the required skill list each strength evidences. The reference
# set must exercise the same three channels a real run does, or the percentile
# is not like-for-like.
SKILL_COVERAGE = {"strong": 1.0, "medium": 0.5, "weak": 0.0}


def main() -> int:
    jds = list(background_corpus())
    fams = list(families())
    if not jds:
        print("no background corpus; nothing to calibrate")
        return 2

    raw_scores: list[float] = []
    by_strength: dict[str, list[float]] = {k: [] for k in TEMPLATES}

    for family, jd in zip(fams, jds, strict=False):
        required = required_skills(jd)
        for strength, resume in synthetic_resumes(jd).items():
            lexical = bm25_similarity(resume, jd, corpus=[*jds, jd, resume])
            dense = cosine_similarity(resume, jd)
            held = required[:round(len(required) * SKILL_COVERAGE[strength])]
            skills = (weighted_jaccard(held, required=required).score
                      if required else None)
            fused = convex_fuse(ChannelScores(dense=dense, lexical=lexical,
                                              skills=skills))
            raw_scores.append(fused.raw)
            by_strength[strength].append(fused.raw)
        _ = family

    calibration = build_calibration(raw_scores, quantiles=100,
                                    channels=("dense", "lexical", "skills"), description=(
        f"{len(raw_scores)} plausible resume/job pairs: {len(TEMPLATES)} resume "
        f"strengths for each of {len(jds)} job families, each scored against "
        f"its OWN family's description. Cross-family pairs are excluded on "
        f"purpose -- including them would make the bar 'better than a random "
        f"pairing', which flatters every real application. Synthetic and small: "
        f"this fixes the scale, not the ranking. No outcome labels exist and "
        f"nothing in the scorer is fitted to these pairs."))

    path = save_calibration(calibration)
    print(f"pairs: {len(raw_scores)}  breakpoints: {len(calibration.breakpoints)}  "
          f"usable: {calibration.usable}")
    for strength, scores in by_strength.items():
        mean = sum(scores) / len(scores)
        print(f"  {strength:<7} mean raw {mean:.3f} -> "
              f"{calibration.percentile(mean):.0f}th percentile")
    print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
