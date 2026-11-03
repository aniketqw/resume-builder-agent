"""Scoring. The one rule: every number here comes out of `rba.ats`.

Nothing in this module implements a formula. `/score` is `rescore`, `/score/batch`
is `score_many`, `/attribution` is `attribute`. The endpoints exist to batch and
to cache, never to compute.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from rba.ats.batch import BatchStats, Candidate, score_many
from rba.ats.fusion import CHANNEL_WEIGHTS, ordinal_band
from rba.resume.attribution import attribute
from rba.resume.document import ResumeDoc
from rba.resume.role_target import target_for_key
from rba.resume.selection import UnknownTarget, apply_ops, signature

from ..models import BatchScoreRequest, ScoreRequest
from ..sessions import SessionNotFound

router = APIRouter()

#: One request should describe one view. Past this the client is almost
#: certainly looping when it should be paging, and the batch stops being a
#: batch and starts being a denial of service against its own event loop.
MAX_VARIANTS = 256


def _resolve(request: Request, body):
    """Either a session's document or an inline one. Returns (session, doc, target)."""
    state = request.app.state.rba
    session = None
    if body.session:
        try:
            session = state.sessions.get(body.session)
        except SessionNotFound:
            raise HTTPException(404, f"no session {body.session!r}") from None

    if body.document is not None:
        try:
            doc = ResumeDoc.from_dict(body.document)
        except Exception as exc:
            raise HTTPException(422, f"not a valid document: {exc}") from None
    elif session is not None and session.doc is not None:
        doc = session.doc
    else:
        raise HTTPException(400, "supply either `session` with a document, or `document`")

    if body.target is not None:
        jd = body.target.jd_text
        required = body.target.required_skills
        preferred = body.target.preferred_skills
        role_key = body.target.role_key
    elif session is not None:
        jd, required, preferred = (session.jd_text, session.required_skills,
                                   session.preferred_skills)
        role_key = session.role_key
    else:
        jd, required, preferred, role_key = "", [], [], ""

    # A chosen role is a scoreable target even with the job-description box
    # empty: synthesise a posting from its keyword profile. Done HERE rather
    # than in `set_target` so the session keeps saying what the user actually
    # typed -- `snapshot()` still reports an empty `jd_text`, and the page can
    # tell a real posting from a stand-in.
    synthetic = None
    if not jd.strip() and role_key:
        synthetic = target_for_key(role_key)
        if synthetic is not None:
            jd = synthetic.jd_text
            # Only supply synthesised skills when the user gave none of their
            # own; their explicit list must always win.
            required = required or synthetic.required_skills
            preferred = preferred or synthetic.preferred_skills

    if not jd.strip():
        raise HTTPException(
            400, "set a job description or pick a role to score against")
    return session, doc, (jd, required, preferred, synthetic)


def _block(scored) -> dict:
    """`weights_used` is the EFFECTIVE weighting, not the nominal table.

    When a channel cannot be computed, `convex_fuse` shares its weight out
    across the others. Showing 0.5 beside a dense channel that is really
    carrying 0.714 of the result would be a quietly wrong label.
    """
    return {"raw": scored.raw, "score_100": round(scored.raw * 100, 2),
            "band": ordinal_band(scored.raw),
            "channels": {k: round(v, 6) for k, v in scored.channels.items()},
            "weights_used": {k: round(v, 4) for k, v in scored.weights_used.items()},
            "weights_nominal": dict(CHANNEL_WEIGHTS)}


@router.post("/score")
async def score_one(request: Request, body: ScoreRequest) -> dict:
    state = request.app.state.rba
    session, doc, (jd, required, preferred, synthetic) = _resolve(request, body)
    cache = session.dense_cache if session is not None else None

    candidates = [Candidate(id="base", text=doc.as_text(), skills=doc.skills())]
    scored = await state.pools.score(
        score_many, candidates, jd, required_skills=required,
        preferred_skills=preferred, use_dense=body.use_dense, cache=cache)

    payload = _block(scored[0])
    payload["signature"] = signature(doc)
    payload["delta_basis"] = "raw"
    if synthetic is not None:
        payload["target"] = synthetic.to_dict()
    if session is not None:
        payload["rev"] = session.rev
    return payload


@router.post("/score/batch")
async def score_batch(request: Request, body: BatchScoreRequest) -> dict:
    """Score a base selection plus N one-off variants in a single pass.

    This is what makes the UI's per-control deltas possible: a whole view's
    worth of numbers arrives in one request, and the truncation dedup inside
    `score_many` usually collapses them to a handful of forward passes.
    """
    if len(body.variants) > MAX_VARIANTS:
        raise HTTPException(413,
                            f"{len(body.variants)} variants exceeds the {MAX_VARIANTS} limit")

    state = request.app.state.rba
    session, doc, (jd, required, preferred, synthetic) = _resolve(request, body)
    cache = session.dense_cache if session is not None else None

    try:
        base_doc = (apply_ops(doc, [op.model_dump() for op in body.base_ops])
                    if body.base_ops else doc)
    except UnknownTarget as exc:
        raise HTTPException(422, f"base_ops: {exc}") from None

    candidates = [Candidate(id="\x00base", text=base_doc.as_text(),
                            skills=base_doc.skills())]
    keys: list[str] = []
    for variant in body.variants:
        try:
            patched = apply_ops(base_doc, [op.model_dump() for op in variant.ops])
        except UnknownTarget as exc:
            raise HTTPException(422, f"variant {variant.key!r}: {exc}") from None
        except ValueError as exc:
            raise HTTPException(400, f"variant {variant.key!r}: {exc}") from None
        candidates.append(Candidate(id=variant.key, text=patched.as_text(),
                                    skills=patched.skills()))
        keys.append(variant.key)

    stats = BatchStats()
    started = time.perf_counter()
    scored = await state.pools.score(
        score_many, candidates, jd, required_skills=required,
        preferred_skills=preferred, use_dense=body.use_dense, cache=cache,
        stats=stats)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

    base = scored[0]
    out = []
    for key, result in zip(keys, scored[1:], strict=True):
        out.append({"key": key, "raw": result.raw,
                    "score_100": round(result.raw * 100, 2),
                    # The delta is on the RAW score, never a percentile:
                    # percentile differences are non-linear and a genuine
                    # improvement can round away to +0.0.
                    "delta": round((result.raw - base.raw) * 100, 2)})

    payload = stats.to_dict() | {"ms": elapsed_ms}
    return {"base": _block(base), "variants": out,
            "stats": payload, "signature": signature(base_doc),
            "delta_basis": "raw",
            "target": synthetic.to_dict() if synthetic is not None else None,
            "rev": session.rev if session is not None else 0}


@router.post("/attribution")
async def attribution(request: Request, body: ScoreRequest) -> dict:
    """Exact leave-one-out impact for every enabled line."""
    state = request.app.state.rba
    session, doc, (jd, required, preferred, _synthetic) = _resolve(request, body)
    cache = session.dense_cache if session is not None else None

    current = signature(doc)
    if (session is not None and session.attribution is not None
            and session.attribution_signature == current):
        result = session.attribution
    else:
        result = await state.pools.score(
            attribute, doc, jd, required_skills=required,
            preferred_skills=preferred, use_dense=body.use_dense, cache=cache)
        if session is not None:
            session.attribution = result
            session.attribution_signature = current

    return {
        "total": result.total, "raw_total": result.raw_total,
        "band": result.band, "channels": result.channels,
        "signature": current, "delta_basis": "raw",
        "impacts": [{"bullet_id": i.bullet_id, "text": i.text,
                     "delta": i.delta, "source_repo": i.source_repo,
                     "verified": i.verified} for i in result.impacts],
    }
