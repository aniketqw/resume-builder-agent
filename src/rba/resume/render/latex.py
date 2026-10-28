"""LaTeX source, for when the typesetting has to be exact.

fpdf2 redraws the layout; this hands the document to a real typesetter
instead. It is plain `article` with no custom class file, so it compiles on
Overleaf as-is without chasing a missing `.cls`.
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

PREAMBLE = r"""\documentclass[a4paper,10pt]{article}
\usepackage[margin=1.4cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{tabularx}
\usepackage{hyperref}
\hypersetup{colorlinks=true,urlcolor=black,linkcolor=black}
\pagestyle{empty}
\setlist[itemize]{leftmargin=1.1em,topsep=1pt,itemsep=0pt,parsep=0pt}
\titleformat{\section}{\normalsize\bfseries\scshape}{}{0pt}{}[\titlerule]
\titlespacing{\section}{0pt}{7pt}{3pt}
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
}
_SPECIALS = re.compile("|".join(re.escape(c) for c in ESCAPES))


def esc(text: str) -> str:
    return _SPECIALS.sub(lambda m: ESCAPES[m.group(0)], text)


def to_latex(doc: ResumeDoc, options: RenderOptions | None = None) -> str:
    opts = options or RenderOptions()
    out: list[str] = [PREAMBLE, r"\begin{document}"]

    if doc.name:
        out.append(r"\begin{center}{\LARGE\bfseries\scshape " + esc(doc.name)
                   + r"}\end{center}")
    if doc.header is not None and doc.header.fields:
        out.append(r"\begin{tabularx}{\textwidth}{@{}lX@{}}")
        out += [f"{esc(label)} & {esc(value)} \\\\"
                for label, value in doc.header.fields]
        out.append(r"\end{tabularx}")
    elif doc.contact:
        out.append(r"\begin{center}" + esc(doc.contact) + r"\end{center}")

    if doc.summary:
        out += [r"\section{Summary}", esc(text_of(doc.summary, opts))]

    for section in visible_sections(doc, opts):
        out.append(r"\section{" + esc(section.title) + "}")
        if isinstance(section.body, KeyValues):
            out.append(r"\begin{tabularx}{\textwidth}{@{}lX@{}}")
            out += [f"\\textbf{{{esc(k)}}} & {esc(text_of(v, opts))} \\\\"
                    for k, v in keyvalue_rows(section.body)]
            out.append(r"\end{tabularx}")
        elif isinstance(section.body, Table):
            rows = table_rows(section.body)
            columns = max(len(section.body.headers),
                          max((len(r) for r in rows), default=1))
            spec = "@{}lX" + "l" * max(columns - 2, 0) + "@{}"
            out.append(r"\begin{tabularx}{\textwidth}{" + spec + "}")
            if section.body.headers:
                out.append(" & ".join(f"\\textbf{{{esc(h)}}}"
                                      for h in section.body.headers) + r" \\")
            for row in rows:
                cells = list(row) + [""] * (columns - len(row))
                out.append(" & ".join(esc(text_of(c, opts)) for c in cells) + r" \\")
            out.append(r"\end{tabularx}")
        if section.text:
            out.append(esc(text_of(section.text, opts)))

        for entry in visible_entries(section, opts):
            heading = " - ".join(p for p in (entry.title, entry.subtitle) if p)
            out.append(r"\entry{" + esc(heading) + "}{" + esc(entry.dates) + "}")
            if entry.tech:
                out.append(r"{\small\itshape Tech: " + esc(", ".join(entry.tech))
                           + r"}\par")
            out += _itemize(visible_bullets(entry, opts), opts)

        out += _itemize(visible_bullets(section, opts), opts)

    out.append(r"\end{document}")
    return "\n".join(out) + "\n"


def _itemize(bullets, opts: RenderOptions) -> list[str]:
    if not bullets:
        return []
    return [r"\begin{itemize}",
            *[r"\item " + esc(text_of(b.text, opts)) for b in bullets],
            r"\end{itemize}"]
