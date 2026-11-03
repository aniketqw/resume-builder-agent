"""Reading a resume PDF back into an editable document.

Deliberately deterministic. An LLM asked to "parse this resume" will happily
smooth over what it cannot read -- inventing a date, merging two projects,
inferring a degree -- and that is the same failure this project exists to
prevent on the generation side. So this is plain parsing: what it cannot prove
from the text, it leaves empty for the user to fill in.

The layout it understands is the BITS one-page format. Anything else should
fall back to :func:`rba.resume.pipeline.parse_resume_text`, which makes weaker
assumptions.

The hard part is that PDF text extraction destroys column boundaries and wraps
long lines. Two rules recover most of it:

* A line that continues the previous one is detected structurally, not by
  guessing -- bullets end in a full stop, entry headings end in a date.
* Table rows are accumulated until the row looks complete (ends in a year),
  because a wrapped cell would otherwise read as its own row.
"""
from __future__ import annotations

import io
import re

from .document import Bullet, Entry, Header, KeyValues, ResumeDoc, Section, Table

# Headings are the one thing this format is strict about: short, all-caps,
# on their own line.
KNOWN_HEADINGS = (
    "SUMMARY",
    "ACADEMIC DETAILS",
    "TECHNICAL SKILLS",
    "INTERNSHIPS",
    "PROJECTS",
    "POSITION OF RESPONSIBILITY",
    "CERTIFICATIONS",
    "ACHIEVEMENTS",
    "PUBLICATIONS",
    "EXPERIENCE",
)

_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
# Hyphen, en dash and em dash: a PDF may carry any of the three.
_DASH = "[-\u2013\u2014]"
# "Oct 2026 - Present", "May 2026 - Jul 2026", "2024 - 2025"
# A heading's date. Two shapes in the wild, and only the first was handled:
# a range ("Jun 2026 - Oct 2026") and a single completion date ("Nov 2025").
# Missing the second meant those headings failed to match and were appended to
# the previous project as bullets -- one entry swallowed nine of them.
_DATE_RANGE = rf"(?:{_MONTH}\s+)?\d{{4}}\s*{_DASH}\s*(?:Present|(?:{_MONTH}\s+)?\d{{4}})"
_DATE_SINGLE = rf"{_MONTH}\s+\d{{4}}"
DATE_RANGE = re.compile(rf"(?P<dates>{_DATE_RANGE}|{_DATE_SINGLE})\s*$")

# Title and subtitle are separated by a dash in one layout and a pipe in
# another: "Sandboxed Code-Execution Agent  |  LLM Agents".
_TITLE_SPLIT = re.compile(r"\s+[|\u2013\u2014]\s+|\s+-\s+")
_YEAR_END = re.compile(r"\b(19|20)\d{2}\s*$")
_SCORE = re.compile(r"(\d+(?:\.\d+)?\s*(?:%|CGPA|GPA))\s*$", re.I)
_HEADER_FIELD = re.compile(r"^(?P<label>[A-Z][A-Za-z /&]{1,22})\s*:\s*(?P<value>.+)$")


def read_pdf(data: bytes) -> str:
    """Extract text from a PDF. Raises if the file is not readable as one."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def looks_like_bits(text: str) -> bool:
    """Whether the strict parser is likely to do better than the flat one."""
    found = sum(1 for h in KNOWN_HEADINGS if _heading_line(h, text))
    return found >= 3


def _heading_line(heading: str, text: str) -> bool:
    return any(line.strip() == heading for line in text.splitlines())


def _split_sections(lines: list[str]) -> tuple[list[str], list[tuple[str, list[str]]]]:
    """Header lines, then (heading, body lines) for each known heading."""
    head: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line in KNOWN_HEADINGS:
            current = (line, [])
            sections.append(current)
            continue
        if current is None:
            head.append(line)
        else:
            current[1].append(line)
    return head, sections


def _parse_header(lines: list[str]) -> Header:
    name = ""
    fields: list[tuple[str, str]] = []
    for line in lines:
        match = _HEADER_FIELD.match(line)
        if match:
            fields.append((match.group("label").strip(), match.group("value").strip()))
        elif not name:
            name = line
        elif fields:
            # a wrapped continuation of the last field
            label, value = fields[-1]
            fields[-1] = (label, f"{value} {line}".strip())
    return Header(name=name, fields=fields)


def _is_heading(line: str) -> bool:
    """Could this line start an entry?

    Needed because a lone "Nov 2025" is a far weaker signal than a full range:
    without this, any bullet that happened to end in a month and year would be
    promoted to a heading and would steal the bullets that followed it. In this
    format a bullet is a sentence and ends in a full stop; a heading does not,
    and is short.
    """
    stripped = line.strip()
    return bool(stripped) and not stripped.endswith(".") and len(stripped) <= 120


def _join_wrapped(lines: list[str]) -> list[str]:
    """Rejoin bullets split across lines by the PDF's line breaks.

    A bullet in this format is a sentence ending in a full stop. So a line that
    does not end a sentence is continued by the next one -- unless that next
    line starts a new entry (it carries a date range) or is a `Tech:` line.
    """
    out: list[str] = []
    for line in lines:
        starts_new = (_is_heading(line) and bool(DATE_RANGE.search(line))) \
            or line.startswith("Tech:")
        if out and not starts_new and not out[-1].endswith((".", ":")) \
                and not (_is_heading(out[-1]) and DATE_RANGE.search(out[-1])) \
                and not out[-1].startswith("Tech:"):
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return out


def _parse_entries(lines: list[str], origin: str = "yours") -> list[Entry]:
    """Dated entries with an optional `Tech:` line and unmarked bullets."""
    entries: list[Entry] = []
    for line in _join_wrapped(lines):
        match = DATE_RANGE.search(line) if _is_heading(line) else None
        if match:
            heading = line[: match.start()].strip(" .,-|\u2013\u2014")
            parts = _TITLE_SPLIT.split(heading, maxsplit=1)
            entries.append(Entry(title=parts[0].strip(),
                                 subtitle=(parts[1].strip() if len(parts) > 1 else ""),
                                 dates=match.group("dates").strip()))
            continue
        if not entries:
            continue          # stray text before the first entry; never guess
        if line.startswith("Tech:"):
            entries[-1].tech = [t.strip() for t in line[5:].split(",") if t.strip()]
            continue
        entries[-1].bullets.append(Bullet(text=line, origin=origin))
    return entries


def _parse_keyvalues(lines: list[str]) -> KeyValues:
    """`Label  value` rows, where a wrapped value continues on the next line.

    A continuation is recognised by starting lowercase: every label in this
    format is capitalised, so "regression" after "... logistic" is a wrap.
    """
    rows: list[tuple[str, str]] = []
    for line in lines:
        if rows and line[:1].islower():
            label, value = rows[-1]
            rows[-1] = (label, f"{value} {line}")
            continue
        label, value = _split_label(line)
        rows.append((label, value))
    return KeyValues(rows=rows)


def _split_label(line: str) -> tuple[str, str]:
    """Split "Cloud & DevOps Docker, Nginx" into its label and its value.

    The two columns were a table cell boundary that extraction threw away, so
    it has to be inferred. The shortest split wins that leaves a label ending
    on a real word -- not on a connector like "/" or "&" -- and a value
    starting on a capital or a digit, which is what a technology name looks
    like. Taking the *shortest* such split is what stops "Data stores" from
    swallowing "PostgreSQL".
    """
    words = line.split()
    for cut in range(1, min(5, len(words)) + 1):
        label = " ".join(words[:cut])
        rest = " ".join(words[cut:])
        if not rest or len(label) > 24:
            break
        if label[-1].isalnum() and (rest[0].isupper() or rest[0].isdigit()):
            return label, rest
    return line, ""


def _parse_table(lines: list[str]) -> tuple[Table, list[tuple[str, str]]]:
    """A grid whose rows end in a year, plus any trailing label-value lines.

    `Subjects / Electives ...` sits under ACADEMIC DETAILS but is not a table
    row, so it is handed back separately rather than forced into the grid.
    """
    headers: list[str] = []
    rows: list[list[str]] = []
    extras: list[tuple[str, str]] = []
    buffer = ""
    for line in lines:
        if not headers and line.isupper() and not _YEAR_END.search(line):
            headers = _split_headers(line)
            continue
        if line.startswith(("Subjects", "Electives")):
            extras.append(_split_label(line))
            continue
        if len(headers) <= 2:
            # No trailing year to mark the end of a row, so one line is one
            # row. Accumulating would fuse every certification into one cell.
            rows.append(_split_row(line, len(headers)))
            continue
        buffer = f"{buffer} {line}".strip() if buffer else line
        if _YEAR_END.search(buffer):
            rows.append(_split_row(buffer, len(headers)))
            buffer = ""
    if buffer:
        rows.append(_split_row(buffer, len(headers)))
    return Table(headers=headers, rows=rows), extras


def _split_headers(line: str) -> list[str]:
    """COURSE SPECIALIZATION INSTITUTE/COLLEGE BOARD/UNIVERSITY SCORE YEAR."""
    known = ["COURSE", "SPECIALIZATION", "INSTITUTE/COLLEGE", "BOARD/UNIVERSITY",
             "SCORE", "YEAR", "CERTIFICATION", "CERTIFYING AUTHORITY"]
    found = [(line.find(k), k) for k in known if k in line]
    if found:
        return [k for _, k in sorted(found)]
    return line.split()


_COURSE = re.compile(
    r"^(UG|PG|CLASS\s+XII|CLASS\s+X|B\.?\s?Tech|M\.?\s?Tech|B\.?E|M\.?E"
    r"|B\.?Sc|M\.?Sc|PhD|Diploma)\b", re.I)
_PARENS = re.compile(r"^(.*?\))\s+(.*)$")


def _split_row(row: str, columns: int) -> list[str]:
    """Recover what the column positions can no longer tell us.

    Both ends are provable: the year is last, the score next to last, and the
    course label is a known leading token. The board is a trailing acronym.
    What is left over in the middle is the specialisation and the institute,
    and those only separate cleanly when the specialisation is parenthesised.
    Anything this cannot prove stays in one cell rather than being guessed at
    -- the UI makes these editable, and a wrong split is worse than an honest
    one.
    """
    if columns <= 2:
        return _split_two(row)
    year = ""
    score = ""
    match = _YEAR_END.search(row)
    if match:
        year = row[match.start():].strip()
        row = row[: match.start()].strip()
    match = _SCORE.search(row)
    if match:
        score = match.group(1).strip()
        row = row[: match.start()].strip()

    course = ""
    match = _COURSE.match(row)
    if match:
        course = match.group(0).strip()
        row = row[match.end():].strip()

    board = ""
    words = row.split()
    if len(words) > 1 and words[-1].isupper() and len(words[-1]) <= 10:
        board = words[-1]
        row = " ".join(words[:-1])

    specialisation, institute = "", row
    match = _PARENS.match(row)
    if match:
        specialisation, institute = match.group(1).strip(), match.group(2).strip()

    cells = [course, specialisation, institute, board][: max(columns - 2, 1)]
    return [*cells, score, year]


def _split_two(row: str) -> list[str]:
    """Two-column rows (CERTIFICATIONS): the authority is the trailing phrase."""
    for marker in (" Amazon Web Services", " IBM", " Google", " Microsoft",
                   " Coursera", " NPTEL", " Udemy"):
        idx = row.find(marker)
        if idx != -1:
            return [row[:idx].strip(), row[idx:].strip()]
    words = row.split()
    return [" ".join(words[:-1]), words[-1]] if len(words) > 1 else [row, ""]


def parse_bits(text: str, origin: str = "yours") -> ResumeDoc:
    """Parse the BITS one-page format into an editable document."""
    head_lines, raw_sections = _split_sections(text.splitlines())
    header = _parse_header(head_lines)

    doc = ResumeDoc(name=header.name, header=header)
    for title, lines in raw_sections:
        if title == "SUMMARY":
            doc.summary = " ".join(_join_wrapped(lines))
            continue
        if title == "TECHNICAL SKILLS":
            doc.sections.append(Section(title=title, body=_parse_keyvalues(lines)))
            continue
        if title in ("ACADEMIC DETAILS", "CERTIFICATIONS"):
            table, extras = _parse_table(lines)
            # Electives sit under ACADEMIC DETAILS in the original, so they
            # stay there rather than becoming a heading of their own.
            note = " ".join(f"{label}: {value}" for label, value in extras)
            doc.sections.append(Section(title=title, body=table, text=note))
            continue
        entries = _parse_entries(lines, origin=origin)
        if entries:
            doc.sections.append(Section(title=title, entries=entries))
        else:
            doc.sections.append(Section(
                title=title,
                bullets=[Bullet(text=line, origin=origin)
                         for line in _join_wrapped(lines)]))
    return doc


def parse_resume(data: bytes | str, *, filename: str = "") -> ResumeDoc:
    """Front door: bytes from an upload, or already-extracted text."""
    if isinstance(data, bytes):
        text = read_pdf(data) if data[:4] == b"%PDF" else data.decode("utf-8", "replace")
    else:
        text = data
    if looks_like_bits(text):
        return parse_bits(text)
    from .pipeline import parse_resume_text

    return parse_resume_text(text)
