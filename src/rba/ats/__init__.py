"""Evidence-grounded resume-to-job scoring.

This package replaces a rubric whose weights I invented. The design follows the
second-place system at **TalentCLEF 2026** (arXiv 2606.31692), a CLEF shared
task whose Task A is exactly this problem: rank resumes against a vacancy. That
system — LLM parses documents into structured JSON, then rank fusion across
several retrieval views — scored MAP 0.6632 against the winner's 0.7140, i.e.
93% of the best result, with no training.

Three independent channels, fused:

* **dense**   semantic similarity from a sentence embedding model
* **lexical** BM25, which keeps exact tokens like "AWS", "CPA", "Kubernetes"
              and version strings that a 384-dimension embedding blurs away
* **skills**  weighted Jaccard over ESCO skill ids — no model, fully auditable

What this package deliberately does *not* do:

* It does not call an LLM. Extraction happens upstream; scoring is arithmetic
  you can check by hand.
* It does not use Reciprocal Rank Fusion. RRF is the reflex choice, but Bruch
  et al. (arXiv 2210.11934, TOIS) found convex combination beats it on all five
  datasets tested, and that RRF's tuned parameters do not transfer across
  domains. See :mod:`rba.ats.fusion`.
* It does not claim to reproduce any real ATS. Greenhouse exposes
  Strong/Good/Partial buckets and Workday A-D letter grades, both driven by
  criteria the employer sets per job. No major ATS shows a candidate a 0-100
  number, so neither should we without saying whose number it is.
"""
from .fusion import (
    CHANNEL_WEIGHTS,
    ChannelScores,
    FusedScore,
    convex_fuse,
    ordinal_band,
    theoretical_min_max,
)
from .lexical import bm25_similarity, tokenize
from .skills import weighted_jaccard

__all__ = [
    "CHANNEL_WEIGHTS",
    "ChannelScores",
    "FusedScore",
    "bm25_similarity",
    "convex_fuse",
    "ordinal_band",
    "theoretical_min_max",
    "tokenize",
    "weighted_jaccard",
]
