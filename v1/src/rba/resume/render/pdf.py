"""PDF output, drawn directly with fpdf2.

Why not LaTeX or a headless browser: Streamlit Community Cloud gives us a
Python environment and nothing else. There is no `pdflatex`, and WeasyPrint
needs cairo and pango as system libraries. fpdf2 is a pure-Python wheel, so it
is the only one of the three that can actually produce a file on the box this
runs on. The `.tex` export in `latex.py` exists for anyone who wants the
typesetting to be exact.

The layout is redrawn, not converted, so it will be close to the original
rather than identical to it.
"""
from __future__ import annotations

import io

from ..document import KeyValues, ResumeDoc, Table
from .model import (
    keyvalue_rows,
    table_rows,
    text_of,
    visible_bullets,
    visible_entries,
    visible_sections,
)
from .options import ORIGIN_COLOURS, RenderOptions

PAGE_MARGIN = 12.0
BODY = 9.0
SMALL = 8.0
# Tried in order until the document fits `max_pages`. A resume that silently
# runs onto a second page is a different document from the one the user
# approved in the preview, so this is measured, not assumed.
FIT_SCALES = (1.0, 0.95, 0.9, 0.86, 0.82, 0.78, 0.74, 0.7)


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    value = hex_colour.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


class _Resume:
    """A thin layout helper over fpdf, so the drawing code reads top to bottom."""

    def __init__(self, serif: bool, scale: float = 1.0):
        from fpdf import FPDF

        self.family = "Times" if serif else "Helvetica"
        self.scale = scale
        margin = PAGE_MARGIN * (0.8 + 0.2 * scale)
        self.pdf = FPDF(orientation="P", unit="mm", format="A4")
        self.pdf.set_auto_page_break(auto=True, margin=margin)
        self.pdf.set_margins(margin, margin, margin)
        self.pdf.add_page()
        self.margin = margin
        self.width = self.pdf.w - 2 * margin

    def s(self, value: float) -> float:
        """Scale a size. Fonts and leading shrink together, or lines collide."""
        return value * self.scale

    # -- primitives -------------------------------------------------------
    def font(self, size: float = BODY, style: str = "") -> None:
        self.pdf.set_font(self.family, style, size)

    def colour(self, rgb: tuple[int, int, int] = (17, 17, 17)) -> None:
        self.pdf.set_text_color(*rgb)

    def title(self, text: str) -> None:
        self.font(self.s(16), "B")
        self.colour()
        self.pdf.cell(self.width, self.s(7.5), text.upper(), align="C",
                      new_x="LMARGIN", new_y="NEXT")

    def heading(self, text: str) -> None:
        self.pdf.ln(self.s(2.0))
        self.font(self.s(9.2), "B")
        self.colour()
        self.pdf.cell(self.width, self.s(4.4), text.upper(), new_x="LMARGIN", new_y="NEXT")
        y = self.pdf.get_y()
        self.pdf.set_draw_color(34, 34, 34)
        self.pdf.set_line_width(0.3)
        self.pdf.line(self.margin, y, self.margin + self.width, y)
        self.pdf.ln(self.s(1.4))

    def paragraph(self, text: str, size: float = BODY, style: str = "") -> None:
        self.font(self.s(size), style)
        self.colour()
        self.pdf.multi_cell(self.width, self.s(4.0), text, align="J",
                            new_x="LMARGIN", new_y="NEXT")

    def entry_head(self, left: str, right: str) -> None:
        self.pdf.ln(self.s(1.2))
        self.font(self.s(BODY), "B")
        self.colour()
        dates_width = self.s(34.0) if right else 0.0
        self.pdf.cell(self.width - dates_width, self.s(4.3), left)
        if right:
            self.font(self.s(SMALL))
            self.pdf.set_text_color(68, 68, 68)
            self.pdf.cell(dates_width, self.s(4.3), right, align="R")
        self.pdf.ln(self.s(4.3))

    def tech(self, text: str) -> None:
        self.font(self.s(SMALL), "I")
        self.pdf.set_text_color(51, 51, 51)
        self.pdf.multi_cell(self.width, self.s(3.6), text, new_x="LMARGIN", new_y="NEXT")

    def bullet(self, text: str, rgb: tuple[int, int, int] | None = None) -> None:
        self.font(self.s(BODY))
        self.colour(rgb or (17, 17, 17))
        indent = self.s(3.6)
        self.pdf.cell(indent, self.s(3.9), chr(149))
        self.pdf.multi_cell(self.width - indent, self.s(3.9), text, align="J",
                            new_x="LMARGIN", new_y="NEXT")

    def keyvalues(self, rows: list[tuple[str, str]]) -> None:
        label_width = self.s(30.0)
        for label, value in rows:
            top = self.pdf.get_y()
            self.font(self.s(SMALL), "B")
            self.colour()
            self.pdf.multi_cell(label_width, self.s(3.8), label, new_x="RIGHT", new_y="TOP")
            self.pdf.set_xy(self.margin + label_width, top)
            self.font(self.s(SMALL))
            self.pdf.multi_cell(self.width - label_width, self.s(3.8), value,
                                new_x="LMARGIN", new_y="NEXT")

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        columns = max(len(headers), max((len(r) for r in rows), default=0))
        if not columns:
            return
        # The first column carries the long cell (a specialisation, a
        # certification name), so it gets the slack.
        weights = [2.2] + [1.0] * (columns - 1) if columns > 1 else [1.0]
        if columns >= 4:
            weights = [0.8, 1.9, 1.3, 1.0] + [0.8] * (columns - 4)
        total = sum(weights)
        widths = [self.width * w / total for w in weights]

        if headers:
            self.font(self.s(7.2), "B")
            self.pdf.set_text_color(51, 51, 51)
            for i, head in enumerate(headers[:columns]):
                self.pdf.cell(widths[i], self.s(3.8), head.upper(), border="B")
            self.pdf.ln(self.s(3.8))
        self.font(self.s(SMALL))
        self.colour()
        for row in rows:
            cells = list(row) + [""] * (columns - len(row))
            height = max(self._lines(cells[i], widths[i]) for i in range(columns)) * self.s(3.6)
            top = self.pdf.get_y()
            for i in range(columns):
                self.pdf.set_xy(self.margin + sum(widths[:i]), top)
                self.pdf.multi_cell(widths[i], self.s(3.6), cells[i], new_x="RIGHT",
                                    new_y="TOP", max_line_height=self.s(3.6))
            self.pdf.set_xy(self.margin, top + height)

    def _lines(self, text: str, width: float) -> int:
        if not text:
            return 1
        return max(1, len(self.pdf.multi_cell(width, self.s(3.6), text, dry_run=True,
                                              output="LINES")))

    def output(self) -> bytes:
        buffer = io.BytesIO()
        self.pdf.output(buffer)
        return buffer.getvalue()


def to_pdf(doc: ResumeDoc, options: RenderOptions | None = None) -> bytes:
    """Render the document, shrinking it until it fits `options.max_pages`.

    Returns the first rendering that fits. If nothing in `FIT_SCALES` does,
    the smallest is returned rather than raising -- a slightly long resume is
    still useful, and `pages_used` lets the UI say so plainly.
    """
    opts = options or RenderOptions()
    rendered = b""
    for scale in FIT_SCALES:
        rendered = _render(doc, opts, scale)
        if not opts.max_pages or _page_count(rendered) <= opts.max_pages:
            return rendered
    return rendered


def pages_used(data: bytes) -> int:
    """How many pages a rendered PDF actually has."""
    return _page_count(data)


def _page_count(data: bytes) -> int:
    from pypdf import PdfReader

    return len(PdfReader(io.BytesIO(data)).pages)


def _render(doc: ResumeDoc, opts: RenderOptions, scale: float) -> bytes:
    page = _Resume(serif=opts.is_bits, scale=scale)

    if doc.name:
        page.title(doc.name)

    if doc.header is not None and doc.header.fields:
        page.keyvalues([(label, value) for label, value in doc.header.fields])
    elif doc.contact:
        page.paragraph(doc.contact, SMALL)
    if doc.headline:
        page.paragraph(doc.headline, SMALL, "I")

    if doc.summary:
        page.heading("Summary")
        page.paragraph(text_of(doc.summary, opts))

    for section in visible_sections(doc, opts):
        page.heading(section.title)
        if isinstance(section.body, KeyValues):
            page.keyvalues([(k, text_of(v, opts))
                            for k, v in keyvalue_rows(section.body)])
        elif isinstance(section.body, Table):
            page.table(section.body.headers,
                       [[text_of(c, opts) for c in row]
                        for row in table_rows(section.body)])
        # A section note (electives, say) reads as a footnote to the table it
        # qualifies, so it follows the body rather than preceding the headers.
        if section.text:
            page.paragraph(text_of(section.text, opts), SMALL)

        for entry in visible_entries(section, opts):
            page.entry_head(
                " - ".join(p for p in (entry.title, entry.subtitle) if p),
                entry.dates)
            if entry.tech and opts.is_bits:
                page.tech("Tech: " + ", ".join(entry.tech))
            for bullet in visible_bullets(entry, opts):
                page.bullet(text_of(bullet.text, opts), _colour_for(bullet, opts))

        for bullet in visible_bullets(section, opts):
            page.bullet(text_of(bullet.text, opts), _colour_for(bullet, opts))

    return page.output()


def _colour_for(bullet, opts: RenderOptions) -> tuple[int, int, int] | None:
    if not opts.colour_provenance:
        return None
    return _rgb(ORIGIN_COLOURS.get(bullet.origin, ORIGIN_COLOURS["assumed"]))
