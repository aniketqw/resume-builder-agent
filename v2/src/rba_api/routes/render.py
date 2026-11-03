"""Preview and export, server-side.

v1's reference artifact built PDFs in the browser with jsPDF plus a hand-rolled
font-metrics table, and zipped with JSZip. Both came from a CDN, which an
offline container cannot reach and a strict CSP should not allow -- and the
browser copy mapped anything outside ASCII to "?".

None of that is needed: `rba.resume.render` already has both paper styles, a
real shrink-to-fit pass, and provenance colouring. Rendering here also means
the score shown beside the download and the bytes inside it come from the same
document, rather than from two implementations that have to be kept in step.
"""
from __future__ import annotations

import io
import zipfile

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from rba.resume.document import ResumeDoc
from rba.resume.render.html import to_html
from rba.resume.render.latex import to_latex
from rba.resume.render.options import RenderOptions
from rba.resume.render.pdf import pages_used, to_pdf
from rba.resume.render.plain import to_plain

from ..models import RenderRequest
from ..sessions import SessionNotFound

router = APIRouter()


def _doc(request: Request, session_id: str, inline: dict | None) -> ResumeDoc:
    if inline is not None:
        try:
            return ResumeDoc.from_dict(inline)
        except Exception as exc:
            raise HTTPException(422, f"not a valid document: {exc}") from None
    if not session_id:
        raise HTTPException(400, "supply either `session` or `document`")
    try:
        session = request.app.state.rba.sessions.get(session_id)
    except SessionNotFound:
        raise HTTPException(404, f"no session {session_id!r}") from None
    if session.doc is None:
        raise HTTPException(409, "this session has no document yet")
    return session.doc


def _options(body: RenderRequest) -> RenderOptions:
    return RenderOptions(style=body.style, show_numbers=body.show_numbers,
                         colour_provenance=body.colour_provenance,
                         include_disabled=body.include_disabled,
                         max_pages=body.max_pages)


@router.post("/render/html")
async def render_html(request: Request, body: RenderRequest) -> dict:
    """The live paper preview. Pure and fast -- no scoring involved."""
    doc = _doc(request, body.session, body.document)
    html = await request.app.state.rba.pools.render(to_html, doc, _options(body))
    return {"html": html, "style": body.style}


@router.post("/render/pdf")
async def render_pdf(request: Request, body: RenderRequest) -> Response:
    doc = _doc(request, body.session, body.document)
    data = await request.app.state.rba.pools.render(to_pdf, doc, _options(body))
    return Response(
        content=data, media_type="application/pdf",
        headers={
            # So the UI can say "this spilled to two pages" instead of the
            # user discovering it after they send it.
            "X-RBA-Pages": str(pages_used(data)),
            "Content-Disposition": f'attachment; filename="resume-{body.style}.pdf"',
        })


@router.post("/render/latex")
async def render_latex(request: Request, body: RenderRequest) -> Response:
    doc = _doc(request, body.session, body.document)
    text = await request.app.state.rba.pools.render(to_latex, doc, _options(body))
    return Response(content=text, media_type="text/x-tex")


@router.post("/render/text")
async def render_text(request: Request, body: RenderRequest) -> Response:
    doc = _doc(request, body.session, body.document)
    text = await request.app.state.rba.pools.render(to_plain, doc, _options(body))
    return Response(content=text, media_type="text/plain; charset=utf-8")


@router.post("/export")
async def export_bundle(request: Request, body: RenderRequest) -> Response:
    """Both paper styles plus the sources, in one zip.

    Mirrors what the reference artifact downloaded, built with stdlib
    `zipfile` and the renderers that already exist -- no new dependency.
    """
    doc = _doc(request, body.session, body.document)
    pools = request.app.state.rba.pools

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        for style in ("bits", "clean"):
            options = RenderOptions(
                style=style, show_numbers=body.show_numbers,
                colour_provenance=body.colour_provenance,
                include_disabled=body.include_disabled,
                max_pages=body.max_pages)
            pdf = await pools.render(to_pdf, doc, options)
            bundle.writestr(f"resume-{style}.pdf", pdf)
            bundle.writestr(f"latex/resume-{style}.tex",
                            await pools.render(to_latex, doc, options))
        plain_options = _options(body)
        bundle.writestr("resume.txt", await pools.render(to_plain, doc, plain_options))
        bundle.writestr("resume.json", doc.to_json())
        bundle.writestr("README.txt", _BUNDLE_README)

    return Response(
        content=buffer.getvalue(), media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="resumes.zip"'})


_BUNDLE_README = """\
What is in this bundle
======================

resume-bits.pdf    The one-page academic layout, with its tables.
resume-clean.pdf   A plainer single-column CV, the one to upload to an ATS.
latex/*.tex        The LaTeX sources for both, if you want to tweak them.
resume.txt         Plain text, for pasting into a form that mangles PDFs.
resume.json        The structured document, so you can re-import it later.

Both PDFs are shrink-to-fit onto one page. If a line reads `[N]` it is a number
that has not been measured yet -- the tool refuses to invent one for you.
"""
