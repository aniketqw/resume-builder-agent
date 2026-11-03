"""Getting a resume in, and editing it once it is there."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi import File as FileParam

from rba.resume.ingest import parse_resume
from rba.resume.merge import baseline
from rba.resume.pipeline import parse_resume_text
from rba.resume.selection import UnknownTarget, apply_ops

from ..models import ParseRequest, PatchRequest, Target
from ..sessions import RevisionMismatch, SessionLimit, SessionNotFound

router = APIRouter()

#: Matches `.streamlit/config.toml`'s maxUploadSize in v1. A resume that does
#: not fit in 5 MB is not a resume.
MAX_UPLOAD = 5 * 1024 * 1024


def get_session(request: Request, session_id: str):
    try:
        return request.app.state.rba.sessions.get(session_id)
    except SessionNotFound:
        raise HTTPException(404, f"no session {session_id!r}") from None


def _store(request: Request, session, doc) -> dict:
    rev = session.set_document(doc, baseline=baseline(doc))
    return {"session_id": session.id, "rev": rev,
            "document": doc.to_dict(), "signature": session.signature}


@router.post("/session")
async def create_session(request: Request, target: Target | None = None) -> dict:
    try:
        session = request.app.state.rba.sessions.create()
    except SessionLimit as exc:
        raise HTTPException(503, str(exc)) from None
    if target is not None:
        session.set_target(jd_text=target.jd_text, role_key=target.role_key,
                           required=target.required_skills,
                           preferred=target.preferred_skills)
    return session.snapshot()


@router.get("/session/{session_id}")
async def read_session(request: Request, session_id: str) -> dict:
    session = get_session(request, session_id)
    payload = session.snapshot()
    payload["document"] = session.doc.to_dict() if session.doc else None
    return payload


@router.delete("/session/{session_id}", status_code=204)
async def drop_session(request: Request, session_id: str) -> None:
    request.app.state.rba.sessions.delete(session_id)


@router.patch("/session/{session_id}/target")
async def set_target(request: Request, session_id: str, target: Target) -> dict:
    """Changing the target invalidates derived numbers but keeps the document:
    a user retargeting at a new job should not lose their edits."""
    session = get_session(request, session_id)
    async with session.lock:
        session.set_target(jd_text=target.jd_text, role_key=target.role_key,
                           required=target.required_skills,
                           preferred=target.preferred_skills)
    return session.snapshot()


@router.post("/upload")
async def upload(request: Request, session_id: str = "",
                 file: UploadFile = FileParam(...)) -> dict:
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"file is larger than {MAX_UPLOAD} bytes")
    if not data:
        raise HTTPException(400, "the uploaded file is empty")
    try:
        doc = parse_resume(data, filename=file.filename or "")
    except Exception as exc:
        # Parsing is deterministic and local; a failure here is about the file,
        # so say which file and why rather than returning a bare 500.
        raise HTTPException(422, f"could not parse {file.filename!r}: {exc}") from None

    session = (get_session(request, session_id) if session_id
               else request.app.state.rba.sessions.create())
    async with session.lock:
        return _store(request, session, doc)


@router.post("/parse")
async def parse_text(request: Request, body: ParseRequest,
                     session_id: str = "") -> dict:
    if not body.text.strip():
        raise HTTPException(400, "no text supplied")
    doc = parse_resume_text(body.text, name=body.name)
    session = (get_session(request, session_id) if session_id
               else request.app.state.rba.sessions.create())
    async with session.lock:
        return _store(request, session, doc)


@router.put("/session/{session_id}/document")
async def replace_document(request: Request, session_id: str,
                           payload: dict) -> dict:
    from rba.resume.document import ResumeDoc

    session = get_session(request, session_id)
    try:
        doc = ResumeDoc.from_dict(payload)
    except Exception as exc:
        raise HTTPException(422, f"not a valid document: {exc}") from None
    async with session.lock:
        return _store(request, session, doc)


@router.patch("/session/{session_id}/document")
async def patch_document(request: Request, session_id: str,
                         body: PatchRequest) -> dict:
    """Apply ops to the working document.

    `if_match` is the optimistic-concurrency guard: two tabs editing the same
    session would otherwise silently overwrite each other.
    """
    session = get_session(request, session_id)
    if session.doc is None:
        raise HTTPException(409, "this session has no document yet")
    async with session.lock:
        try:
            session.check_rev(body.if_match)
        except RevisionMismatch as exc:
            raise HTTPException(409, {"error": str(exc), "rev": exc.expected,
                                      "document": session.doc.to_dict()}) from None
        try:
            doc = apply_ops(session.doc, [op.model_dump() for op in body.ops])
        except UnknownTarget as exc:
            raise HTTPException(422, str(exc)) from None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        session.doc = doc
        rev = session.bump()
    return {"session_id": session.id, "rev": rev,
            "document": doc.to_dict(), "signature": session.signature}
