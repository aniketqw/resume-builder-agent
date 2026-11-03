"""Unstyled text: the format least likely to be mangled on the way in.

No tables, no columns, no glyph bullets. Headings are uppercase on their own
line and every bullet starts with a hyphen.

Stated honestly, because it is easy to overclaim here: this is a *deterministic*
property, not a measured one. Plain ASCII with one item per line has nothing for
a text extractor to reorder, drop or interleave -- which is the actual argument.
What published evidence exists is about parsing accuracy on the *extractor*
side, not about hyphens: a layout-aware pipeline lifts resume-parsing F1 from
0.919 to 0.959 (arXiv 2510.09722, Table 2). The widely-repeated advice about
ATS-friendly formatting is largely vendor guidance rather than studies -- see
`docs/literature-survey.md` section 2.2.
"""
from __future__ import annotations

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


def to_plain(doc: ResumeDoc, options: RenderOptions | None = None) -> str:
    opts = options or RenderOptions()
    lines: list[str] = []

    if doc.name:
        lines.append(doc.name)
    if doc.header is not None:
        lines += [f"{label}: {value}" for label, value in doc.header.fields]
    elif doc.contact:
        lines.append(doc.contact)
    if doc.headline:
        lines.append(doc.headline)

    if doc.summary:
        lines += ["", "SUMMARY", text_of(doc.summary, opts)]

    for section in visible_sections(doc, opts):
        lines += ["", section.title.upper()]
        if isinstance(section.body, KeyValues):
            lines += [f"{label}: {text_of(value, opts)}"
                      for label, value in keyvalue_rows(section.body)]
        elif isinstance(section.body, Table):
            # A grid cannot survive as a grid here, so each row becomes one
            # readable line rather than a column layout a parser would shred.
            for row in table_rows(section.body):
                lines.append(" | ".join(text_of(c, opts) for c in row if c.strip()))
        if section.text:
            lines.append(text_of(section.text, opts))

        for entry in visible_entries(section, opts):
            heading = " - ".join(p for p in (entry.title, entry.subtitle) if p)
            if entry.dates:
                heading = f"{heading} ({entry.dates})" if heading else entry.dates
            lines.append(heading)
            if entry.tech:
                lines.append(f"Tech: {', '.join(entry.tech)}")
            lines += [f"- {text_of(b.text, opts)}"
                      for b in visible_bullets(entry, opts)]

        lines += [f"- {text_of(b.text, opts)}"
                  for b in visible_bullets(section, opts)]

    return "\n".join(lines).strip() + "\n"
