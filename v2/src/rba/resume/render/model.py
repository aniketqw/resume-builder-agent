"""The one place that decides what a renderer is allowed to show.

Every format walks the same flattened view, so turning numbers off or hiding a
disabled bullet cannot behave differently in the PDF than it does in the
preview the user approved.
"""
from __future__ import annotations

import re

from ..document import Bullet, Entry, KeyValues, ResumeDoc, Section, Table
from .options import RenderOptions


def visible_bullets(holder: Entry | Section, options: RenderOptions) -> list[Bullet]:
    """Bullets held directly by an entry or a section -- never both levels."""
    return [b for b in holder.bullets if options.include_disabled or b.enabled]


def visible_entries(section: Section, options: RenderOptions) -> list[Entry]:
    return [e for e in section.entries if visible_bullets(e, options)]


def visible_sections(doc: ResumeDoc, options: RenderOptions) -> list[Section]:
    out: list[Section] = []
    for section in doc.sections:
        if section.body is not None or section.text:
            out.append(section)
            continue
        if visible_entries(section, options) or visible_bullets(section, options):
            out.append(section)
    return out


def bullet_text(bullet: Bullet, options: RenderOptions) -> str:
    return text_of(bullet.text, options)


def text_of(text: str, options: RenderOptions) -> str:
    """Apply the numbers switch.

    Dropping a number has to leave a sentence that still reads, so the number
    is replaced by the measurement placeholder rather than deleted -- and the
    same `numbers_in` the verifier uses decides what counts as a number, so the
    two can never disagree about which tokens are metrics.
    """
    if options.show_numbers:
        return text
    from ...nodes.verify import numbers_in

    out = text
    for token in numbers_in(text):
        out = re.sub(rf"(?<![\w.]){re.escape(token)}(?![\w])", "—", out)
    return out


def table_rows(body: Table) -> list[list[str]]:
    return [row for row in body.rows if any(cell.strip() for cell in row)]


def keyvalue_rows(body: KeyValues) -> list[tuple[str, str]]:
    return [(k, v) for k, v in body.rows if k or v]


def word_count(doc: ResumeDoc, options: RenderOptions) -> int:
    return len(to_words(doc, options))


def to_words(doc: ResumeDoc, options: RenderOptions) -> list[str]:
    from .plain import to_plain

    return to_plain(doc, options).split()
