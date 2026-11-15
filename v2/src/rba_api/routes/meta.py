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
                f"The embedding model reads at most {MAX_TOKENS} tokens at a "
                "time, and a full one-page resume is about 1,150 -- too long "
                "for one pass. The dense channel no longer truncates the "
                "document to fit: it splits the resume into its own "
                "sections and entries, embeds each chunk separately (all "
                "well under the limit), and combines them into one "
                "length-weighted vector. Every line is read by the dense "
                "channel now, not just the first page's worth."
            ),
            "chunking": (
                "Why chunk-and-pool instead of one bigger model: mean-"
                "pooling independently-embedded chunks is established "
                "practice for long-document sentence-embedding matching. "
                "The alternative -- embedding the whole document in one "
                "pass before any splitting (\"late chunking\") -- needs a "
                "long-context encoder, which would reopen the memory-"
                "footprint tradeoff this project already settled by "
                "choosing bge-small over a larger model (see "
                "docs/literature-survey.md). The one measured cost: a "
                "session's very first score pays for embedding every "
                "chunk once, before any caching kicks in -- every score "
                "after that only re-embeds the one chunk that changed."
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
