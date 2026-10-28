"""LaTeX source, for when the typesetting has to be exact.

fpdf2 redraws the layout; this hands the document to a real typesetter
instead. It is plain `article` with no custom class file, so it compiles on
Overleaf as-is without chasing a missing `.cls`.

`tests/test_latex.py` compiles the output with tectonic and asserts the
engine emits no overfull or underfull boxes, so the preamble below is tuned
against a real run rather than guessed at.
"""
from __future__ import annotations

import re

from ..document import KeyValues, ResumeDoc, Table
from .model import (
    keyvalue_rows,
    table_rows,
    text_of,
    visible_bullets,
    visible_entries,
    visible_sections,
)
from .options import RenderOptions

# `extarticle` rather than `article`, at 8pt rather than 10pt, because 10pt
# ran this resume onto a second page while the fpdf2 renderer fitted it on one.
# The two must agree: the same document cannot be one page as a PDF and two as
# LaTeX. fpdf2 gets there by compiling and shrinking until it fits (see
# `pdf.py`), which is impossible here -- measuring a page count needs a TeX
# engine, and Streamlit Community Cloud has none. So the density is chosen
# rather than discovered: 8pt against fpdf2's measured 8.55pt effective size,
# close enough that the two outputs read alike.
PREAMBLE = r"""\documentclass[a4paper,8pt]{extarticle}
\usepackage[margin=1.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{array}
\usepackage{tabularx}
\usepackage{hyperref}
\hypersetup{colorlinks=true,urlcolor=black,linkcolor=black}
\pagestyle{empty}
% A tabularx is already \textwidth wide, so starting a paragraph with one
% pushes it \parindent past the margin and TeX reports an overfull box.
\setlength{\parindent}{0pt}
\setlist[itemize]{leftmargin=1.1em,topsep=1pt,itemsep=0pt,parsep=0pt}
% Bold small caps: Latin Modern has no `bx/sc', and asking for it only
% earns a font-substitution warning. Headings are uppercased already, so
% small caps would change nothing anyway.
\titleformat{\section}{\normalsize\bfseries}{}{0pt}{}[\titlerule]
\titlespacing{\section}{0pt}{7pt}{3pt}
% Flexible columns have to wrap ragged-right. Justified, a six-column table
% leaves each X narrow enough that one long word overflows the cell.
% The \hspace{0pt} matters: TeX will not hyphenate a word that opens a
% paragraph, so without it the first word of a cell overflows rather than
% breaking.
\newcolumntype{R}{>{\raggedright\arraybackslash\hspace{0pt}}X}
\newcommand{\entry}[2]{\noindent\textbf{#1}\hfill{\small #2}\par}
"""

# Characters LaTeX would otherwise interpret.
#
# This has to be a single pass. Replacing them one after another re-escapes the
# braces that `\textbackslash{}` itself introduces, turning a stray backslash
# into `\textbackslash\{\}`.
ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    # Legal in text mode but typeset as the wrong glyph, or as a line break
    # inside a tabularx cell.
    "|": r"\textbar{}", "<": r"\textless{}", ">": r"\textgreater{}",
    # Punctuation that survives ingestion from a PDF or a word processor. The
    # T1-encoded Latin Modern fonts have no slot for any of it, so left alone
    # it is dropped from the page with only a log warning -- including the em
    # dash `text_of` substitutes for a withheld number.
    "\u2014": r"\textemdash{}", "\u2013": r"\textendash{}",
    "\u2018": "`", "\u2019": "'", "\u201c": "``", "\u201d": "''",
    "\u2026": r"\ldots{}", "\u2022": r"\textbullet{}", "\xa0": "~",
}
_SPECIALS = re.compile("|".join(re.escape(c) for c in ESCAPES))

# A bare host only counts as a link once it carries a path, otherwise
# "Next.js" and "Node.js" would be turned into hyperlinks.
_URL = re.compile(
    r"(?:https?://|www\.)[^\s|)\]}]+"
    r"|(?:[\w-]+\.)+[A-Za-z]{2,}/[^\s|)\]}]*"
)
_URL_TRAILING = ".,;:!?"

# `INSTITUTE/COLLEGE` is one unbreakable word to TeX -- hyphenation stops at
# the slash -- and it is wider than a column in a six-column table.
_SLASH = re.compile(r"(?<=\S)/(?=\S)")


def _breakable(latex: str) -> str:
    return _SLASH.sub(r"/\\allowbreak{}", latex)


def esc(text: str) -> str:
    return _SPECIALS.sub(lambda m: ESCAPES[m.group(0)], text)


def esc_prose(text: str) -> str:
    """Escape running text, handing any URL to `\\url` untouched.

    `\\url` tokenises its argument itself -- `~`, `_` and `%` are literal in
    there -- so a URL that has been through `esc()` first comes out with the
    backslashes visible.
    """
    out: list[str] = []
    last = 0
    for match in _URL.finditer(text):
        url = match.group(0).rstrip(_URL_TRAILING)
        if "{" in url or "}" in url or "\\" in url:
            continue    # `\url` cannot balance those; leave it to esc().
        out.append(_breakable(esc(text[last:match.start()])))
        out.append(r"\url{" + url + "}")
        last = match.start() + len(url)
    out.append(_breakable(esc(text[last:])))
    return "".join(out)


def to_latex(doc: ResumeDoc, options: RenderOptions | None = None) -> str:
    opts = options or RenderOptions()
    out: list[str] = [PREAMBLE, r"\begin{document}"]
    start = len(out)

    if doc.name:
        out.append(r"\begin{center}{\LARGE\bfseries " + esc(doc.name)
                   + r"}\end{center}")
    if doc.header is not None and doc.header.fields:
        out.append(r"\begin{tabularx}{\textwidth}{@{}lR@{}}")
        out += [f"{esc(label)} & {esc_prose(value)} \\\\"
                for label, value in doc.header.fields]
        out.append(r"\end{tabularx}")
    elif doc.contact:
        out.append(r"\begin{center}" + esc_prose(doc.contact) + r"\end{center}")

    if doc.summary:
        out += [_section("Summary"), esc_prose(text_of(doc.summary, opts))]

    for section in visible_sections(doc, opts):
        out.append(_section(section.title))
        if isinstance(section.body, KeyValues):
            out.append(r"\begin{tabularx}{\textwidth}{@{}lR@{}}")
            out += [f"\\textbf{{{esc(k)}}} & {esc_prose(text_of(v, opts))} \\\\"
                    for k, v in keyvalue_rows(section.body)]
            out.append(r"\end{tabularx}")
        elif isinstance(section.body, Table):
            rows = table_rows(section.body)
            columns = max(len(section.body.headers),
                          max((len(r) for r in rows), default=1))
            # Every column after the first has to flex: with four fixed `l`
            # columns the single X is squeezed below the width of its own
            # header and the row runs past the margin.
            spec = "@{}l" + "R" * max(columns - 1, 0) + "@{}"
            out.append(r"\begin{tabularx}{\textwidth}{" + spec + "}")
            if section.body.headers:
                out.append(" & ".join(f"\\textbf{{{_breakable(esc(h))}}}"
                                      for h in section.body.headers) + r" \\")
            for row in rows:
                cells = list(row) + [""] * (columns - len(row))
                out.append(" & ".join(esc_prose(text_of(c, opts))
                                      for c in cells) + r" \\")
            out.append(r"\end{tabularx}")
        if section.text:
            out.append(esc_prose(text_of(section.text, opts)))

        for entry in visible_entries(section, opts):
            heading = " - ".join(p for p in (entry.title, entry.subtitle) if p)
            out.append(r"\entry{" + esc(heading) + "}{" + esc(entry.dates) + "}")
            if entry.tech:
                out.append(r"{\small\itshape Tech: " + esc(", ".join(entry.tech))
                           + r"}\par")
            out += _itemize(visible_bullets(entry, opts), opts)

        out += _itemize(visible_bullets(section, opts), opts)

    # TeX writes no PDF at all for a document with no pages, so an empty
    # resume has to still ask for one.
    if len(out) == start:
        out.append(r"\null")

    out.append(r"\end{document}")
    return "\n".join(out) + "\n"


def _section(title: str) -> str:
    """Headings are uppercased so the one the parser invented for the summary
    matches the ones that came out of the source document."""
    return r"\section{" + esc(title.upper()) + "}"


def _itemize(bullets, opts: RenderOptions) -> list[str]:
    if not bullets:
        return []
    return [r"\begin{itemize}",
            *[r"\item " + esc_prose(text_of(b.text, opts)) for b in bullets],
            r"\end{itemize}"]
