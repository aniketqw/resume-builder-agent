"""Everything the page needs to describe the scorer -- from the scorer.

The "How the score works" tab is driven entirely by this router. Nothing about
the weights, the bands or the token limit is written down in the frontend,
because a hardcoded 0.5 in JavaScript is a number that can silently stop being
true.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

from rba.ats.dense import MAX_TOKENS, MODEL_ID
from rba.ats.fusion import CHANNEL_WEIGHTS, ORDINAL_BANDS
from rba.scoring import WEIGHTS, load_roles

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    state = request.app.state.rba
    return {"ok": True, "version": "2.0.0", "model": MODEL_ID,
            "dense": bool(state.dense_ready), "sessions": len(state.sessions)}


@router.get("/meta/scoring")
async def scoring_meta() -> dict:
    return {
        "weights": dict(CHANNEL_WEIGHTS),
        "bands": [{"at_least": threshold, "label": label}
                  for threshold, label in ORDINAL_BANDS],
        "max_tokens": MAX_TOKENS,
        "model": MODEL_ID,
        "delta_basis": "raw",
        "notes": {
            "headline": (
                "The headline number is a convex combination of three "
                "channels: a dense embedding similarity, BM25 lexical overlap, "
                "and a weighted skills match. A channel that cannot be "
                "computed is dropped and its weight redistributed -- it is "
                "treated as unknown, not as zero."
            ),
            "no_rubric": (
                "There is deliberately no seven-bar rubric summing to 100. "
                "Those weights were invented -- nobody measured that a missing "
                "phone number is worth exactly six points -- so they are used "
                "for advice (which keyword is missing, is the resume too long) "
                "and never as the score."
            ),
            "truncation": (
                f"The embedding model reads at most {MAX_TOKENS} tokens, and a "
                "full one-page resume is about 1,150. Lines past that cut do "
                "not change the dense channel at all, so their delta reflects "
                "the lexical and skills channels only."
            ),
            "deltas": (
                "Every delta is computed on the raw fused score, not on the "
                "calibrated percentile: percentile differences are non-linear "
                "and a real improvement can round to +0.0."
            ),
        },
        # The advice-only rubric, published so the page can show what it is
        # and label it accordingly.
        "advice_rubric": dict(WEIGHTS),
    }


@router.get("/meta/roles")
async def roles_meta() -> dict:
    roles = load_roles()
    return {"roles": [
        {"key": key,
         "title": role.title,
         "headline": role.headline,
         "target_words": list(role.target_words),
         "required_sections": list(role.required_sections),
         "keywords": dict(role.keywords)}
        for key, role in roles.items()
    ]}
