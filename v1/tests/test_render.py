"""Every format must carry the whole document.

The failure these guard against is quiet: a renderer that drops the skills
table still produces a plausible-looking resume, and the user only finds out
when an employer's parser sees no skills.
"""
from __future__ import annotations

import pathlib

import pytest

from rba.resume.document import Bullet, Entry, ResumeDoc, Section
from rba.resume.ingest import parse_bits
from rba.resume.render import RenderOptions, to_html, to_latex, to_pdf, to_plain
from rba.resume.render.pdf import pages_used

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "aniket_bits.txt"

# Things that appear in exactly one structure each: a bullet, a `Tech:` line, a
# key-value row, a table cell, a header field and the summary. If a renderer
# forgets one structure, one of these disappears.
MUST_APPEAR = [
    "ANIKET SAXENA",
    "h20250169@pilani.bits-pilani.ac.in",
    "seeking an AI engineering internship",
    "VIT Amravati",
    "ICSE",
    "PostgreSQL",
    "Software Testing",
    "Resume Builder Agent",
    "Oct 2026 - Present",
    "Marvell Semiconductors",
    "Containers & Kubernetes Essentials",
    "double-accept races",
]


@pytest.fixture(scope="module")
def doc() -> ResumeDoc:
    return parse_bits(FIXTURE.read_text())


@pytest.mark.parametrize("needle", MUST_APPEAR)
def test_plain_keeps_every_structure(doc, needle):
    assert needle in to_plain(doc, RenderOptions())


@pytest.mark.parametrize("needle", MUST_APPEAR)
def test_html_keeps_every_structure(doc, needle):
    from html import escape

    assert escape(needle, quote=False) in to_html(doc, RenderOptions())


def test_latex_keeps_every_structure(doc):
    out = to_latex(doc, RenderOptions())
    for needle in MUST_APPEAR:
        # LaTeX escapes `&`, so compare on the escaped form.
        assert needle.replace("&", r"\&") in out, needle
    assert out.rstrip().endswith(r"\end{document}")


def test_latex_escapes_specials_without_double_escaping():
    from rba.resume.render.latex import esc

    assert esc("C# & 100% _x_") == r"C\# \& 100\% \_x\_"
    assert esc(r"a\b") == r"a\textbackslash{}b"


@pytest.mark.parametrize("style", ["bits", "clean"])
def test_pdf_is_a_real_single_page_file(doc, style):
    data = to_pdf(doc, RenderOptions(style=style))
    assert data[:4] == b"%PDF"
    assert len(data) > 2000
    assert pages_used(data) == 1


def test_pdf_text_survives_extraction(doc):
    """A PDF that looks right but extracts as nothing is useless to a parser."""
    from rba.resume.ingest import read_pdf

    text = read_pdf(to_pdf(doc, RenderOptions()))
    for needle in ("ANIKET SAXENA", "Resume Builder Agent", "PostgreSQL",
                   "Marvell Semiconductors"):
        assert needle in text, needle


def test_shrink_to_fit_only_engages_when_needed(doc):
    """A document that already fits must not be shrunk."""
    small = ResumeDoc(name="Jane Doe", summary="Engineer.",
                      sections=[Section("EXPERIENCE",
                                        bullets=[Bullet("Built a thing.")])])
    from rba.resume.render.pdf import _render

    assert to_pdf(small, RenderOptions()) == _render(small, RenderOptions(), 1.0)
    # ...and the long one is not left overflowing.
    assert pages_used(to_pdf(doc, RenderOptions())) == 1
    assert pages_used(to_pdf(doc, RenderOptions(max_pages=0))) > 1


def test_disabled_bullets_are_absent_everywhere(doc):
    target = doc.all_bullets()[0]
    phrase = target.text[:40]
    doc.toggle(target.id, False)
    try:
        assert phrase not in to_plain(doc, RenderOptions())
        assert phrase not in to_latex(doc, RenderOptions())
        from rba.resume.ingest import read_pdf
        assert phrase not in read_pdf(to_pdf(doc, RenderOptions()))
    finally:
        doc.toggle(target.id, True)


def test_numbers_can_be_withheld_without_breaking_the_sentence():
    doc = ResumeDoc(sections=[Section("PROJECTS", entries=[Entry(
        title="Thing",
        bullets=[Bullet("Cut latency by 37% across 12 services.")])])])
    on = to_plain(doc, RenderOptions(show_numbers=True))
    off = to_plain(doc, RenderOptions(show_numbers=False))
    assert "37%" in on and "12" in on
    assert "37%" not in off and "Cut latency by" in off


def test_years_are_not_treated_as_metrics():
    """`numbers_in` excludes years, so a date must survive the numbers switch."""
    doc = ResumeDoc(sections=[Section("PROJECTS", entries=[
        Entry(title="Thing", dates="Jan 2026 - May 2026",
              bullets=[Bullet("Shipped it in 2026.")])])])
    off = to_plain(doc, RenderOptions(show_numbers=False))
    assert "Jan 2026 - May 2026" in off
    assert "2026." in off


def test_provenance_colouring_is_opt_in(doc):
    plain_html = to_html(doc, RenderOptions(colour_provenance=False))
    coloured = to_html(doc, RenderOptions(colour_provenance=True))
    assert "from repository evidence" not in plain_html
    assert "from repository evidence" in coloured


def test_empty_document_renders_without_raising():
    empty = ResumeDoc()
    assert to_plain(empty, RenderOptions()).strip() == ""
    assert to_pdf(empty, RenderOptions())[:4] == b"%PDF"
    assert r"\end{document}" in to_latex(empty, RenderOptions())
