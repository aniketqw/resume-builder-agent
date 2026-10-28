"""The public entry point: score one resume against one job description.

Everything here is arithmetic over the three channels. No LLM call happens in
this module — extraction runs upstream and hands in structured skills, so the
number can be recomputed and checked by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .calibrate import load_calibration
from .corpus import background_corpus
from .dense import cosine_similarity
from .fusion import ChannelScores, FusedScore, convex_fuse
from .lexical import bm25_similarity, matched_terms
from .normalise import coverage, normalise_all
from .skills import SkillMatch, has_skills, weighted_jaccard


@dataclass
class ScoreReport:
    """Everything needed to justify the number to a sceptical reader."""

    fused: FusedScore
    skills: SkillMatch
    present_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    recognised_skills: list[str] = field(default_factory=list)
    unrecognised_skills: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def score_100(self) -> float:
        """The headline number.

        A percentile against the reference set when one exists, because a raw
        similarity of 0.63 shown as "63/100" implies a calibration nobody did.
        """
        if self.fused.percentile is not None:
            return round(self.fused.percentile, 1)
        return round(self.fused.raw * 100, 1)

    @property
    def is_calibrated(self) -> bool:
        return self.fused.percentile is not None

    @property
    def scale_note(self) -> str:
        if self.is_calibrated:
            return (f"Percentile against {self.fused.reference_size} plausible "
                    f"resume/job pairs — not an ATS score.")
        return "Raw fused score, uncalibrated — not an ATS score."

    @property
    def band(self) -> str:
        return self.fused.band

    def explain(self) -> list[str]:
        """Plain-language reasons, in the order a reader would want them."""
        lines: list[str] = []
        channels = self.fused.channels
        weights = self.fused.weights_used
        for name in ("dense", "lexical", "skills"):
            if name in channels:
                label = {"dense": "Semantic similarity",
                         "lexical": "Keyword/BM25 match",
                         "skills": "Skill coverage"}[name]
                lines.append(
                    f"{label}: {channels[name] * 100:.0f}/100 "
                    f"(weight {weights[name]:.0%})")
        if self.skills.missing_required:
            lines.append("Required skills not evidenced: "
                         + ", ".join(self.skills.missing_required))
        if self.missing_terms:
            lines.append("Job-description terms absent from the resume: "
                         + ", ".join(self.missing_terms[:12]))
        lines.extend(self.notes)
        return lines


def score_resume_against_jd(resume_text: str, jd_text: str, *,
                            resume_skills: list[str] | None = None,
                            required_skills: list[str] | None = None,
                            preferred_skills: list[str] | None = None,
                            use_dense: bool = True,
                            corpus: list[str] | None = None,
                            calibrate: bool = True) -> ScoreReport:
    """Score a resume against a job description.

    ``resume_skills`` and ``required_skills`` normally come from the LLM
    extraction step. When they are absent the skills channel is dropped and its
    weight is redistributed — the score is still meaningful, just built from
    fewer signals, and the report says so.
    """
    notes: list[str] = []

    # Without a background corpus BM25's IDF is computed over two documents
    # and every term looks equally rare, which flattens the channel.
    background = list(corpus) if corpus else list(background_corpus())
    lexical = bm25_similarity(resume_text, jd_text,
                              corpus=[*background, jd_text, resume_text])
    present, missing = matched_terms(resume_text, jd_text)

    dense: float | None = None
    if use_dense:
        dense = cosine_similarity(resume_text, jd_text)
        if dense is None:
            notes.append("Semantic channel unavailable; its weight was "
                         "redistributed across the remaining channels.")
    else:
        notes.append("Semantic channel disabled.")

    skills_score: float | None = None
    match = SkillMatch()
    if has_skills(required_skills or [], preferred_skills or []):
        match = weighted_jaccard(
            normalise_all(resume_skills or []),
            required=normalise_all(required_skills or []),
            preferred=normalise_all(preferred_skills or []),
        )
        skills_score = match.score
    else:
        notes.append("No structured skills supplied, so the skill channel was "
                     "skipped rather than scored zero.")

    recognised, unrecognised = coverage(resume_skills or [])
    fused = convex_fuse(ChannelScores(dense=dense, lexical=lexical,
                                      skills=skills_score))

    if calibrate:
        calibration = load_calibration()
        if not calibration.usable:
            notes.append("No usable reference set, so the score is the raw "
                         "fused value rather than a percentile.")
        elif not calibration.matches(fused.channels):
            notes.append(
                "This run used a different set of channels than the reference "
                "set, so a percentile would not be like-for-like. Showing the "
                "raw fused score instead.")
        else:
            fused = replace(fused, percentile=calibration.percentile(fused.raw),
                            reference_size=calibration.reference_size)

    return ScoreReport(
        fused=fused,
        skills=match,
        present_terms=present,
        missing_terms=missing,
        recognised_skills=recognised,
        unrecognised_skills=unrecognised,
        notes=notes,
    )
