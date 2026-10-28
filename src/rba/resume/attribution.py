"""What each bullet is actually worth, by exact leave-one-out.

The user-facing question is "what happens to my score if I remove this line?",
and the only answer that is exactly right is: remove it and score again. We can
afford that because **scoring makes no LLM calls** — it is one ONNX embedding
plus arithmetic.

Measured on a 20-bullet resume (bge-small, CPU):

======================================  =========  ==================
approach                                cost       accuracy
======================================  =========  ==================
exact leave-one-out, all 20 deltas      4.9 s      exact
re-score after one toggle               153 ms     exact
mean-pool approximation                 1.15 ms    **sign wrong 6/20**
======================================  =========  ==================

The approximation — embed each bullet once, mean-pool the kept ones, never
re-embed — is 4,253x faster and was the obvious design. Measured against ground
truth it scored Pearson 0.643 and agreed on *sign* for only 14 of 20 bullets,
meaning it would tell a user a line helps when it hurts. That is the failure
this project exists to avoid, so it is not used.

So: compute every delta once, behind a spinner, which is nothing next to the
minutes of LLM work that produced the bullets. After that each toggle costs one
embedding.

Deltas do **not** sum to the total, because bullets interact. They answer the
question asked and no other; :func:`rescore` is the authority on any particular
selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..ats.corpus import background_corpus
from ..ats.dense import embed
from ..ats.fusion import ChannelScores, convex_fuse, ordinal_band
from ..ats.lexical import bm25_similarity
from ..ats.skills import has_skills, weighted_jaccard
from .document import ResumeDoc


@dataclass(frozen=True)
class BulletImpact:
    bullet_id: str
    text: str
    delta: float               # score points lost if this bullet is removed
    source_repo: str = ""
    verified: bool = True

    @property
    def helps(self) -> bool:
        return self.delta > 0

    @property
    def label(self) -> str:
        return f"{self.delta:+.1f}"


@dataclass
class Attribution:
    total: float
    band: str
    # Unrounded 0-1 fused score. `total` is rounded for display, and comparing
    # a rounded total against an unrounded rescore introduces up to 0.05 of
    # apparent error -- enough to make an exactness test look broken.
    raw_total: float = 0.0
    impacts: list[BulletImpact] = field(default_factory=list)
    channels: dict[str, float] = field(default_factory=dict)

    def ranked(self) -> list[BulletImpact]:
        return sorted(self.impacts, key=lambda i: -i.delta)

    def harmful(self) -> list[BulletImpact]:
        """Bullets whose removal would *raise* the score.

        A grounded scorer can say this; one that asks a model to guess an
        impact cannot. Measured example: a backend JD scored 0.66 higher
        without "Ran incident response and wrote postmortems".
        """
        return [i for i in self.impacts if i.delta < 0]

    def by_id(self, bullet_id: str) -> BulletImpact | None:
        for impact in self.impacts:
            if impact.bullet_id == bullet_id:
                return impact
        return None


def _score_from_parts(dense: float | None, text: str, jd_text: str,
                      skills: list[str], required: list[str],
                      preferred: list[str], corpus: list[str]) -> tuple[float, dict]:
    lexical = bm25_similarity(text, jd_text, corpus=[*corpus, jd_text, text])
    skills_score = None
    if has_skills(required, preferred):
        skills_score = weighted_jaccard(skills, required=required,
                                        preferred=preferred).score
    fused = convex_fuse(ChannelScores(dense=dense, lexical=lexical,
                                      skills=skills_score))
    return fused.raw, dict(fused.channels)


def rescore(doc: ResumeDoc, jd_text: str, *, required_skills: list[str] | None = None,
            preferred_skills: list[str] | None = None,
            use_dense: bool = True) -> tuple[float, dict]:
    """Score the document as it currently stands. One embedding; ~153 ms."""
    text = doc.as_text()
    corpus = list(background_corpus())
    dense = None
    if use_dense and text.strip():
        vectors = embed([text, jd_text])
        if vectors is not None:
            dense = max(0.0, float(np.dot(vectors[0], vectors[1])))
    raw, channels = _score_from_parts(
        dense, text, jd_text, doc.skills(),
        required_skills or [], preferred_skills or [], corpus)
    return raw, channels


def attribute(doc: ResumeDoc, jd_text: str, *,
              required_skills: list[str] | None = None,
              preferred_skills: list[str] | None = None,
              use_dense: bool = True) -> Attribution:
    """Exact leave-one-out impact for every currently enabled bullet.

    All variants are embedded in a single batched call, which is what makes
    this affordable: twenty-one separate inferences would cost far more than
    one batch of twenty-one.
    """
    enabled = doc.enabled()
    corpus = list(background_corpus())
    required = required_skills or []
    preferred = preferred_skills or []

    full_text = doc.as_text()
    variant_texts: list[str] = []
    for bullet in enabled:
        bullet.enabled = False
        variant_texts.append(doc.as_text())
        bullet.enabled = True

    dense_values: list[float | None] = [None] * (len(enabled) + 1)
    if use_dense and full_text.strip():
        vectors = embed([full_text, *variant_texts, jd_text])
        if vectors is not None:
            jd_vector = vectors[-1]
            dense_values = [max(0.0, float(np.dot(v, jd_vector)))
                            for v in vectors[:-1]]

    full_raw, channels = _score_from_parts(
        dense_values[0], full_text, jd_text, doc.skills(), required, preferred,
        corpus)

    impacts: list[BulletImpact] = []
    for index, bullet in enumerate(enabled):
        bullet.enabled = False
        variant_raw, _ = _score_from_parts(
            dense_values[index + 1], variant_texts[index], jd_text,
            doc.skills(), required, preferred, corpus)
        bullet.enabled = True
        impacts.append(BulletImpact(
            bullet_id=bullet.id, text=bullet.text,
            delta=round((full_raw - variant_raw) * 100, 4),
            source_repo=bullet.source_repo, verified=bullet.verified))

    return Attribution(total=round(full_raw * 100, 1), raw_total=full_raw,
                       band=ordinal_band(full_raw), impacts=impacts,
                       channels={k: round(v, 4) for k, v in channels.items()})
