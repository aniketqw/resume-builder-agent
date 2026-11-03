"""Combining the channels into one score, and saying honestly what it means.

**Why convex combination rather than Reciprocal Rank Fusion.** RRF with k=60
is the usual answer for hybrid retrieval. Bruch et al., *An Analysis of Fusion
Functions for Hybrid Retrieval* (arXiv 2210.11934, TOIS), tested it against a
simple weighted sum and found the weighted sum won on every dataset they
tried — nDCG@1000 of 0.454 vs 0.425 on MS MARCO, 0.542 vs 0.514 on Natural
Questions, and similar gaps zero-shot on HotpotQA and FEVER. They also found
RRF's tuned parameters *fail to generalise out of domain*, which is precisely
the situation here: there is no in-domain data to tune on.

**Why theoretical rather than observed min-max.** The same paper shows the
choice of linear normalisation is rank-equivalent, and recommends normalising
by theoretical bounds because it carries one fewer data-dependent statistic and
so survives distribution shift better. Observed min-max also has a nasty
property for a single-document tool: with one candidate, min == max and the
normalised score is undefined.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Channel weights. The *form* (convex combination) is evidence-backed; these
# particular numbers are a judgement call and are reported as such in the UI.
# The dense:lexical ratio of 0.5:0.2 gives dense a 71% share, inside the
# [0.6, 0.8] band Bruch et al. found consistently effective for that pair.
CHANNEL_WEIGHTS: dict[str, float] = {
    "dense": 0.5,
    "lexical": 0.2,
    "skills": 0.3,
}

# Ordinal bands. Real applicant tracking systems report ordinal grades, not
# percentages: Greenhouse Talent Matching uses Strong / Good / Partial /
# Limited, and Workday HiredScore uses A-D. Showing a band beside the number
# keeps the output recognisable to anyone who has seen a real one.
ORDINAL_BANDS: tuple[tuple[float, str], ...] = (
    (0.80, "Strong match"),
    (0.60, "Good match"),
    (0.40, "Partial match"),
    (0.00, "Limited match"),
)


@dataclass(frozen=True)
class ChannelScores:
    """Each channel's similarity, already in [0, 1]."""

    dense: float | None = None
    lexical: float | None = None
    skills: float | None = None

    def available(self) -> dict[str, float]:
        """Only the channels that produced a score.

        A channel returns ``None`` when it cannot be computed — no embedding
        model loaded, or a job description with no extractable skills. Those
        are genuinely unknown, not zero, and scoring them as zero would punish
        a candidate for our missing inputs.
        """
        return {
            name: value
            for name, value in (("dense", self.dense), ("lexical", self.lexical),
                                ("skills", self.skills))
            if value is not None
        }


@dataclass(frozen=True)
class FusedScore:
    raw: float
    band: str
    channels: dict[str, float] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    percentile: float | None = None
    reference_size: int = 0

    @property
    def display(self) -> str:
        """What to show a human. Never a bare number with no provenance."""
        if self.percentile is not None:
            return (f"{self.percentile:.0f}th percentile vs "
                    f"{self.reference_size} reference pairs — {self.band}")
        return f"{self.raw * 100:.0f}/100 (uncalibrated) — {self.band}"


def theoretical_min_max(value: float, lower: float, upper: float) -> float:
    """Normalise to [0, 1] using known bounds rather than observed ones."""
    if upper <= lower:
        return 0.0
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))


def convex_fuse(channels: ChannelScores,
                weights: dict[str, float] | None = None) -> FusedScore:
    """Weighted sum of whichever channels are available.

    Missing channels have their weight redistributed proportionally across the
    rest, so the result stays on the same 0-1 scale instead of being silently
    depressed. (This redistribution idea is the one genuinely good piece of
    design in ``srbhr/Resume-Matcher``; its *weights* are asserted rather than
    fitted and are not used here.)
    """
    weights = weights or CHANNEL_WEIGHTS
    present = channels.available()
    if not present:
        return FusedScore(raw=0.0, band=ordinal_band(0.0))

    total = sum(weights[name] for name in present)
    if total <= 0:
        return FusedScore(raw=0.0, band=ordinal_band(0.0))

    used = {name: weights[name] / total for name in present}
    raw = sum(present[name] * used[name] for name in present)
    raw = max(0.0, min(1.0, raw))
    return FusedScore(raw=raw, band=ordinal_band(raw),
                      channels=dict(present), weights_used=used)


def ordinal_band(raw: float) -> str:
    for threshold, label in ORDINAL_BANDS:
        if raw >= threshold:
            return label
    return ORDINAL_BANDS[-1][1]
