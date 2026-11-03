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

Cost scales with how much text each embedding carries, not just with the
bullet count. On a real 662-word one-page resume with 15 bullets the same
code takes 582 ms for one re-score and 5.4 s for a full attribution -- slower
per bullet than the 20-bullet figures above, because every variant embeds the
whole document.

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
from functools import lru_cache

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


@lru_cache(maxsize=8)
def _jd_vector(jd_text: str) -> tuple[float, ...] | None:
    """The job description's embedding, computed once per session.

    Two separate wins, both measured on a 662-word resume:

    * The job description does not change while the user edits, so re-embedding
      it on every toggle is pure waste -- 10 ms of it.
    * More importantly, the tokeniser pads a batch to its longest member. Sending
      a short job description alongside a long resume pads it out to the resume's
      length, so ``embed([doc, jd])`` cost 590 ms where ``embed([doc])`` alone
      costs 291 ms and ``embed([jd])`` costs 10 ms. Splitting them is faster even
      on the very first call.

    Returned as a tuple because `lru_cache` needs a hashable value; callers turn
    it back into an array.
    """
    vectors = embed([jd_text])
    if vectors is None:
        return None
    return tuple(float(x) for x in vectors[0])


def _jd_array(jd_text: str) -> np.ndarray | None:
    cached = _jd_vector(jd_text)
    return None if cached is None else np.asarray(cached, dtype=np.float32)


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
    """Score the document as it currently stands.

    One embedding of the whole document -- the job description's is cached, see
    :func:`_jd_vector`. Measured on a real 662-word one-page resume: 291 ms,
    down from 582 ms before the job description was split out of the batch.
    """
    text = doc.as_text()
    corpus = list(background_corpus())
    dense = None
    if use_dense and text.strip():
        jd_vector = _jd_array(jd_text)
        vectors = embed([text])
        if vectors is not None and jd_vector is not None:
            dense = max(0.0, float(np.dot(vectors[0], jd_vector)))
    raw, channels = _score_from_parts(
        dense, text, jd_text, doc.skills(),
        required_skills or [], preferred_skills or [], corpus)
    return raw, channels


def attribute(doc: ResumeDoc, jd_text: str, *,
              required_skills: list[str] | None = None,
              preferred_skills: list[str] | None = None,
              use_dense: bool = True, cache=None) -> Attribution:
    """Exact leave-one-out impact for every currently enabled bullet.

    All variants are embedded in a single batched call, which is what makes
    this affordable: twenty-one separate inferences would cost far more than
    one batch of twenty-one.
    """
    # Delegated to `ats.batch.score_many`, which does what this function used
    # to do inline -- one batched embed for all variants -- plus the truncation
    # dedup. On the 15-bullet fixture that turns 16 forward passes into 4 and
    # the call from 5.07s into 0.34s, with every number bit-identical. The
    # batch module imports `_score_from_parts` from here, so there is still
    # exactly one fusion code path.
    from ..ats.batch import Candidate, score_many

    enabled = doc.enabled()
    required = required_skills or []
    preferred = preferred_skills or []

    full_text = doc.as_text()
    candidates = [Candidate(id="", text=full_text, skills=doc.skills())]
    for bullet in enabled:
        bullet.enabled = False
        candidates.append(Candidate(id=bullet.id, text=doc.as_text(),
                                    skills=doc.skills()))
        bullet.enabled = True

    scored = score_many(candidates, jd_text, required_skills=required,
                        preferred_skills=preferred, use_dense=use_dense,
                        cache=cache)
    full_raw, channels = scored[0].raw, scored[0].channels

    impacts: list[BulletImpact] = []
    for bullet, variant in zip(enabled, scored[1:], strict=True):
        impacts.append(BulletImpact(
            bullet_id=bullet.id, text=bullet.text,
            delta=round((full_raw - variant.raw) * 100, 4),
            source_repo=bullet.source_repo, verified=bullet.verified))

    return Attribution(total=round(full_raw * 100, 1), raw_total=full_raw,
                       band=ordinal_band(full_raw), impacts=impacts,
                       channels={k: round(v, 4) for k, v in channels.items()})
