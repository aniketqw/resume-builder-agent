"""Importing a GitHub account into the working document.

The rule this endpoint exists to uphold, from the project's README: *the agent
never invents an achievement or a number*. Everything it adds is traceable to
repository evidence, and anything that is not gets dropped upstream in
`verify_claims` before it ever reaches here.

The second rule is in `merge_candidates`: generated lines arrive **switched
off**. The resume the user uploaded is the baseline and is never rewritten --
they accept lines one at a time, each showing what it is actually worth.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rba.resume.failures import diagnose_failures, redact
from rba.resume.merge import baseline, candidates, merge_candidates
from rba.resume.pipeline import build_from_github

from ..sessions import SessionNotFound

log = logging.getLogger(__name__)
router = APIRouter()


class ImportRequest(BaseModel):
    username: str
    github_token: str = ""
    groq_key: str = ""
    jev_key: str = ""
    # Measured at roughly 3,000 tokens and one request per repository, so this
    # is a spend control, not a preference.
    max_repos: int = Field(default=6, ge=1, le=25)


def _session(request: Request, session_id: str):
    try:
        return request.app.state.rba.sessions.get(session_id)
    except SessionNotFound:
        raise HTTPException(404, f"no session {session_id!r}") from None


def _explain(exc: Exception) -> str:
    """An exception a user can act on, with credentials stripped.

    `_diagnose` in v1 existed because the first version printed only the
    exception class, so a missing GitHub token surfaced as a bare
    "RuntimeError" -- true and useless. Redaction matters more here than it did
    in v1: this string goes into an HTTP response body.
    """
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "invalid_api_key" in lowered or "invalid api key" in lowered or "401" in lowered:
        return ("A provider rejected the API key. Groq keys are revoked as soon "
                "as they appear in a public repository — paste a current one.")
    if "rate" in lowered and "limit" in lowered:
        return ("Rate limited. Without a GitHub token you get 60 requests an "
                "hour; adding one raises it to 5,000.")
    # What GitHub actually says for a username that does not exist is
    # "Validation Failed / The listed users and repositories cannot be
    # searched either because the resources do not exist or you do not have
    # permission" -- which is accurate and completely unhelpful.
    if ("not found" in lowered or "404" in lowered
            or "cannot be searched" in lowered
            or "validation failed" in lowered):
        return ("No public repositories found for that username. Check the "
                "spelling — and note that private repositories are never read.")
    return redact(text)[:400]


async def _run_import(app_state, session, body: ImportRequest, job) -> dict:
    """The job body. Returns what the client needs to re-render."""
    job.state = "running"
    result = await build_from_github(
        body.username,
        github_token=body.github_token,
        groq_key=body.groq_key,
        jev_key=body.jev_key,
        max_repos=body.max_repos,
        progress=job.report,
        config=app_state.config,
    )

    async with session.lock:
        generated = result.doc
        if session.doc is None or (not session.doc.sections and not session.doc.name):
            # Nothing to merge into: the fetched repositories *are* the resume.
            # Re-stamp the origin so the provenance colouring still tells the
            # truth about where every line came from.
            merged = generated
            for bullet in merged.all_bullets():
                bullet.origin = "repo"
            base = baseline(merged)
        else:
            # Mutates and returns `session.doc`. enabled=False is the whole
            # point: existing bullets keep their ids and their order, so a
            # score taken before the merge still describes the same document
            # until the user switches something on.
            merged = merge_candidates(session.doc, generated, enabled=False)
            base = session.baseline_doc or baseline(merged)

        session.doc = merged
        session.baseline_doc = base
        session.bump()          # bumps rev and drops derived artefacts

        # Everything the pipeline used to throw away.
        session.facts = result.facts
        session.summaries = result.summaries
        session.repos = result.repos
        session.excluded = result.excluded
        session.stats = result.stats

    failure = diagnose_failures(result.stats)
    return {
        "session_id": session.id,
        "rev": session.rev,
        "document": session.doc.to_dict(),
        "signature": session.signature,
        "added": len(candidates(session.doc)),
        "stats": {k: v for k, v in result.stats.items() if k != "errors"},
        "failure": failure.to_dict() if failure else None,
        "excluded": [{"name": r.name, "reason": r.selection_reason}
                     for r in result.excluded],
    }


@router.post("/session/{session_id}/github")
async def start_import(request: Request, session_id: str,
                       body: ImportRequest) -> dict:
    if not body.username.strip():
        raise HTTPException(400, "a GitHub username is required")
    session = _session(request, session_id)
    state = request.app.state.rba
    job = state.jobs.create()

    async def runner():
        try:
            job.finish(await _run_import(state, session, body, job))
        except asyncio.CancelledError:
            raise
        except Exception as exc:          # reported to the user, not raised
            # exc_info: an unexpected failure here must leave a traceback in the
            # server log, or the only evidence is a one-line message in a toast.
            log.warning("github import failed: %s", exc, exc_info=True)
            job.fail(_explain(exc))

    job.task = asyncio.create_task(runner())
    return {"job_id": job.id, "state": job.state}


@router.get("/jobs/{job_id}")
async def poll_job(request: Request, job_id: str) -> dict:
    job = request.app.state.rba.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id!r}")
    return job.to_dict()


@router.delete("/jobs/{job_id}")
async def cancel_job(request: Request, job_id: str) -> dict:
    job = request.app.state.rba.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id!r}")
    job.cancel()
    return job.to_dict(include_result=False)


@router.get("/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str) -> StreamingResponse:
    """Server-sent events, one frame per pipeline checkpoint."""
    import json

    job = request.app.state.rba.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id!r}")

    async def frames():
        async for event, payload in job.stream():
            yield f"event: {event}\ndata: {json.dumps(payload)}\n\n"

    return StreamingResponse(frames(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})
