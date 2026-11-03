"""The LaTeX export, compiled.

Every other test in this suite checks the *source* we emit. That is not the
same claim as "the LaTeX export works": a `.tex` file can contain every needle
you look for and still refuse to compile, or compile into a page with text
hanging off the right margin. So these tests run a real engine over the output
and look at the PDF and the log that come back.

tectonic is used because it is a single binary that pulls the packages it
needs on demand -- a full TeX Live is ~4GB and could not be assumed on a
contributor's machine, let alone in CI.
"""
from __future__ import annotations

import io
import os
import pathlib
import shutil
import subprocess

import pytest

from rba.resume.document import ResumeDoc
from rba.resume.ingest import parse_bits
from rba.resume.render import RenderOptions, to_latex

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "aniket_bits.txt"

TECTONIC = shutil.which("tectonic")
NO_TECTONIC = ("no LaTeX engine on PATH; install one with `brew install "
               "tectonic` (macOS) or `cargo install tectonic`, or see "
               "https://tectonic-typesetting.github.io/install.html")

# tectonic fetches each `.sty` and `.tfm` the first time it sees it, so the
# very first compile on a clean machine needs the network. Afterwards it works
# from its cache offline. A machine without either is not a failing renderer,
# so the compile is skipped rather than failed -- but only for that one reason.
_OFFLINE = ("connection", "network", "resolve", "dns", "timed out",
            "failed to download", "unable to connect", "io error")


@pytest.fixture(scope="module")
def doc() -> ResumeDoc:
    return parse_bits(FIXTURE.read_text())


def compile_tex(source: str, tmp_path: pathlib.Path) -> tuple[bytes, str]:
    """Typeset `source` and return the PDF bytes alongside the engine log."""
    tex = tmp_path / "resume.tex"
    tex.write_text(source, encoding="utf-8")
    result = subprocess.run(
        [TECTONIC, "-X", "compile", str(tex), "--keep-logs", "--untrusted"],
        capture_output=True, text=True, cwd=tmp_path, timeout=300,
        # Share the user's cache so the packages are downloaded once, not once
        # per test.
        env={**os.environ, "HOME": os.path.expanduser("~")},
    )
    if result.returncode != 0:
        blame = (result.stderr + result.stdout).lower()
        if any(phrase in blame for phrase in _OFFLINE):
            pytest.skip("tectonic could not fetch its support files; it needs "
                        f"network access on first run. Engine said: "
                        f"{result.stderr.strip()[-300:]}")
        pytest.fail(f"tectonic failed:\n{result.stdout}\n{result.stderr}")
    pdf = tmp_path / "resume.pdf"
    assert pdf.exists(), f"engine reported success but wrote no PDF\n{result.stdout}"
    log = (tmp_path / "resume.log").read_text(encoding="utf-8", errors="replace")
    return pdf.read_bytes(), log


def assert_no_complaints(log: str) -> None:
    """No boxes over- or underfull, and no silently substituted font.

    The count is zero rather than a tolerated baseline. Each of these means
    something visible on the page -- text past the right margin, a stretched
    line, or a glyph dropped entirely -- and the renderer's preamble is tuned
    so that none of them occur on the fixture.
    """
    complaints = [line for line in log.splitlines()
                  if line.startswith(("Overfull", "Underfull"))
                  or "Font Warning" in line
                  or "Missing character" in line]
    assert complaints == [], "\n".join(complaints)


def page_count(pdf: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(pdf)).pages)


def pdf_text(pdf: bytes) -> str:
    from pypdf import PdfReader

    return "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)


pytestmark = pytest.mark.skipif(TECTONIC is None, reason=NO_TECTONIC)


@pytest.mark.parametrize("options", [
    RenderOptions(style="bits"),
    RenderOptions(style="clean"),
    RenderOptions(show_numbers=False),
], ids=["bits", "clean", "no-numbers"])
def test_latex_compiles_cleanly(doc, tmp_path, options):
    pdf, log = compile_tex(to_latex(doc, options), tmp_path)
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 10_000, "a resume this long cannot fit in a few KB"
    assert page_count(pdf) >= 1
    assert_no_complaints(log)


def test_compiled_pdf_still_carries_the_content(doc, tmp_path):
    """A PDF that typesets but loses a section is worse than one that fails."""
    pdf, _ = compile_tex(to_latex(doc, RenderOptions()), tmp_path)
    text = pdf_text(pdf)
    for needle in ("ANIKET SAXENA", "Resume Builder Agent", "PostgreSQL",
                   "Marvell Semiconductors"):
        assert needle in text, needle


def test_empty_document_still_produces_a_page(tmp_path):
    """TeX writes no file at all for a document with no pages, so an export
    with nothing in it must not look like a crashed compile."""
    pdf, log = compile_tex(to_latex(ResumeDoc(), RenderOptions()), tmp_path)
    assert pdf[:4] == b"%PDF"
    assert page_count(pdf) == 1
    assert_no_complaints(log)


def test_links_are_hyperlinks_not_bare_text(doc, tmp_path):
    source = to_latex(doc, RenderOptions())
    assert r"\url{github.com/aniketqw}" in source
    assert "github.com/aniketqw}" not in source.replace(r"\url{github.com/aniketqw}", "")
    # A bare pipe is a maths-mode delimiter, not a separator.
    assert " | " not in source
    assert r"\textbar{}" in source
    pdf, _ = compile_tex(source, tmp_path)
    assert "github.com/aniketqw" in pdf_text(pdf)


def test_version_numbers_are_not_mistaken_for_links():
    """`\\url{Next.js}` would turn a library name into a dead hyperlink."""
    from rba.resume.render.latex import esc_prose

    assert esc_prose("Built with Next.js and Node.js") == \
        "Built with Next.js and Node.js"
    assert esc_prose("see github.com/a/b for more") == \
        r"see \url{github.com/a/b} for more"
    assert esc_prose("docs at https://example.com/x.") == \
        r"docs at \url{https://example.com/x}."


def test_headings_share_one_case(doc):
    """`Summary` is invented by the parser; the rest come from the source
    document in upper case. Mixing the two looks like a rendering bug."""
    import re

    headings = re.findall(r"\\section\{(.+?)\}", to_latex(doc, RenderOptions()))
    assert headings, "no sections rendered"
    assert headings == [h.upper() for h in headings]
