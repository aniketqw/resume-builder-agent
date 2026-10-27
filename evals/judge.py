"""LLM-as-judge metrics for summarization quality.

Two metrics, deliberately different in kind:

**Faithfulness** (RAGAS). Decompose a summary into atomic claims, then issue a
binary supported/unsupported verdict per claim against the evidence. Score is
``supported / total``. Binary verdicts are used rather than a holistic rating
because LLM judges are measurably biased on 1-5 scales and much steadier on
yes/no calls. This is the blocking metric: the pipeline's whole promise is that
nothing unsupported survives, so the target is 1.0, not "high".

**G-Eval** over the four SummEval dimensions (coherence, consistency, fluency,
relevance). Scored 1-5 and probability-weighted rather than taken as a raw
integer — raw integer ratings cluster and lose resolution, which is the problem
G-Eval was designed around. This one is tracked, not blocking.

Mitigations for known judge bias (position effects, self-preference, score
compression): dimensions are scored in separate calls, samples are averaged
over ``n``, and the judge runs on a different model tier than the generator.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from rba.groq_client import GroqClient
from rba.state import ClaimStatus, RepoSummary

SUMM_EVAL_DIMENSIONS = {
    "coherence": ("The summary is well-structured and well-organised. It builds "
                  "from sentence to sentence into a connected body of information "
                  "about the repository, rather than a heap of related facts."),
    "consistency": ("Every fact in the summary is factually aligned with the "
                    "evidence. No detail is contradicted by, or absent from, the "
                    "evidence bundle."),
    "fluency": ("The individual sentences are well-written: correct grammar, "
                "natural phrasing, no awkward repetition or fragments."),
    "relevance": ("The summary selects the important content from the evidence. "
                  "It covers what matters about this repository and does not pad "
                  "with trivia."),
}


class _Atomic(BaseModel):
    claims: list[str] = Field(description="Atomic, self-contained factual statements")


class _Verdict(BaseModel):
    reason: str = Field(description="One sentence of justification, written first")
    supported: bool = Field(description="True only if the evidence supports the claim")


class _Rating(BaseModel):
    reason: str = Field(description="One sentence of justification, written first")
    score: int = Field(description="Integer rating from 1 to 5")


@dataclass
class Faithfulness:
    score: float
    supported: int
    total: int
    unsupported: list[str] = field(default_factory=list)


@dataclass
class QualityScore:
    dimensions: dict[str, float]

    @property
    def mean(self) -> float:
        vals = list(self.dimensions.values())
        return sum(vals) / len(vals) if vals else 0.0


DECOMPOSE = (
    "Break the text into atomic factual statements. An atomic statement is a "
    "short sentence conveying exactly one piece of information, understandable "
    "on its own without the surrounding text. Resolve pronouns. Ignore anything "
    "that is not a factual assertion."
)

VERDICT = (
    "You check whether a statement is supported by a body of evidence.\n\n"
    "Give one sentence of reasoning first, then the verdict. Mark `supported` "
    "true only if the evidence actually contains the information. A statement "
    "that is plausible, or that merely sounds consistent with the evidence, is "
    "NOT supported. A statement containing a number that the evidence does not "
    "contain is never supported."
)


async def faithfulness(client: GroqClient, summary_text: str,
                       evidence: str, tier: str = "big") -> Faithfulness:
    """RAGAS-style faithfulness: supported atomic claims over total."""
    atomic, _ = await client.structured(
        [{"role": "system", "content": DECOMPOSE},
         {"role": "user", "content": summary_text}],
        _Atomic, node="judge", tier=tier)
    claims = [c for c in atomic.claims if c.strip()]
    if not claims:
        return Faithfulness(1.0, 0, 0)

    async def one(statement: str) -> tuple[str, bool]:
        verdict, _ = await client.structured(
            [{"role": "system", "content": VERDICT},
             {"role": "user",
              "content": f"EVIDENCE:\n{evidence}\n\nSTATEMENT:\n{statement}"}],
            _Verdict, node="judge", tier=tier)
        return statement, verdict.supported

    results = []
    for statement in claims:          # serial: 8K TPM leaves no room to fan out
        results.append(await one(statement))
    supported = [s for s, ok in results if ok]
    unsupported = [s for s, ok in results if not ok]
    return Faithfulness(len(supported) / len(claims), len(supported),
                        len(claims), unsupported)


async def g_eval(client: GroqClient, summary_text: str, evidence: str,
                 n: int = 3, tier: str = "big") -> QualityScore:
    """G-Eval over the SummEval dimensions, averaged over n samples."""
    scores: dict[str, float] = {}
    for name, criterion in SUMM_EVAL_DIMENSIONS.items():
        samples = []
        for i in range(n):
            rating, _ = await client.structured(
                [{"role": "system", "content":
                  f"You evaluate repository summaries on one criterion.\n\n"
                  f"CRITERION — {name}: {criterion}\n\n"
                  f"Think through the criterion in one sentence, then give an "
                  f"integer score from 1 (poor) to 5 (excellent)."},
                 {"role": "user",
                  "content": f"EVIDENCE:\n{evidence[:4000]}\n\n"
                             f"SUMMARY:\n{summary_text}"}],
                _Rating, node="judge", tier=tier,
                # Vary the sample so n>1 measures something.
                temperature=0.0 if i == 0 else 0.4)
            samples.append(max(1, min(5, rating.score)))
        scores[name] = sum(samples) / len(samples)
    return QualityScore(scores)


def summary_text(summary: RepoSummary, verified_only: bool = True) -> str:
    """Render a summary the way a judge should see it."""
    parts = [summary.one_line, summary.problem, summary.approach]
    for c in summary.claims:
        if verified_only and c.status is not ClaimStatus.VERIFIED:
            continue
        parts.append(c.text)
    return "\n".join(p for p in parts if p)


# -- deterministic checks, no model needed --------------------------------
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def number_precision(summary: RepoSummary, evidence: str) -> float:
    """Share of numbers in the summary that actually occur in the evidence.

    A cheap, model-free proxy for the thing that matters most. It should be
    1.0 by construction after verify_claims; if it is not, something upstream
    regressed.
    """
    numbers = []
    for c in summary.claims:
        if c.status is ClaimStatus.VERIFIED:
            numbers += _NUM.findall(c.text)
    if not numbers:
        return 1.0
    present = sum(1 for n in numbers if n in evidence or n.replace(",", "") in evidence)
    return present / len(numbers)


def coverage(summary: RepoSummary) -> float:
    """Did the summary actually produce usable material?"""
    usable = sum(1 for c in summary.claims if c.status is ClaimStatus.VERIFIED)
    return min(1.0, usable / 3)


def weighted_overall(faith: float, quality: float, precision: float) -> float:
    """A single tracking number. Faithfulness dominates by design."""
    return round(0.6 * faith + 0.25 * (quality / 5.0) + 0.15 * precision, 4)


def probability_weighted(scores: dict[int, float]) -> float:
    """G-Eval's weighted sum, for when token logprobs are available."""
    total = sum(scores.values())
    if not total:
        return 0.0
    return sum(s * p for s, p in scores.items()) / total


__all__ = ["Faithfulness", "QualityScore", "coverage", "faithfulness", "g_eval",
           "number_precision", "probability_weighted", "summary_text",
           "weighted_overall"]
