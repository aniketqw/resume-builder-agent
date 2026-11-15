"""The BITS parser, checked against a real resume.

The fixture is the text pypdf extracts from a genuine one-page BITS resume, so
these assert against real wrapping and real lost column boundaries rather than
a tidied-up sample that would hide both.
"""
from __future__ import annotations

import pathlib

import pytest

from rba.resume.document import KeyValues, Table
from rba.resume.ingest import (
    _pick_table,
    _split_label,
    looks_like_bits,
    parse_bits,
    parse_resume,
    read_pdf_tables,
)

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "aniket_bits.txt"


@pytest.fixture(scope="module")
def text() -> str:
    return FIXTURE.read_text()


@pytest.fixture(scope="module")
def doc(text):
    return parse_bits(text)


def section(doc, title):
    for candidate in doc.sections:
        if candidate.title == title:
            return candidate
    raise AssertionError(f"no section {title!r} in {[s.title for s in doc.sections]}")


def test_detects_the_format(text):
    assert looks_like_bits(text)
    assert not looks_like_bits("Jane Doe\nSome prose about a job.\n")


def test_header_fields_are_labelled(doc):
    assert doc.name == "ANIKET SAXENA"
    labels = [k for k, _ in doc.header.fields]
    assert labels == ["Course", "Email", "Mobile", "CGPA", "Links"]
    values = dict(doc.header.fields)
    assert values["Email"] == "h20250169@pilani.bits-pilani.ac.in"
    # The full handle, not a line-wrapped prefix of it.
    assert values["Links"].endswith("linkedin.com/in/aniket-saxena-8a5328233")


def test_summary_is_rejoined_across_the_line_break(doc):
    assert doc.summary.startswith("M.E. Software Systems student")
    assert "seeking an AI engineering internship" in doc.summary


def test_every_project_survives_with_its_metadata(doc):
    projects = section(doc, "PROJECTS").entries
    assert [p.title for p in projects] == [
        "Resume Builder Agent",
        "Sandboxed Code-Execution Agent",
        "VisionDev",
        "Multi-University Bounty Platform",
    ]
    first = projects[0]
    assert first.subtitle == "LLM Agents"
    assert first.dates == "Oct 2026 - Present"
    assert "LangGraph" in first.tech
    assert len(first.bullets) == 3
    # A bullet that the PDF wrapped over two lines comes back as one sentence.
    assert first.bullets[0].text.endswith("evidence-linked facts.")


def test_bullet_counts_are_exact(doc):
    counts = {e.title: len(e.bullets) for e in section(doc, "PROJECTS").entries}
    assert counts == {
        "Resume Builder Agent": 3,
        "Sandboxed Code-Execution Agent": 2,
        "VisionDev": 2,
        "Multi-University Bounty Platform": 4,
    }


def test_internship_and_position_are_entries(doc):
    internships = section(doc, "INTERNSHIPS").entries
    assert len(internships) == 1
    assert internships[0].title == "Software Firmware Engineer, Marvell Semiconductors"
    assert internships[0].dates == "May 2026 - Jul 2026"
    assert len(internships[0].bullets) == 3

    position = section(doc, "POSITION OF RESPONSIBILITY").entries
    assert len(position) == 1
    assert position[0].dates == "Aug 2026 - Present"


def test_skills_split_into_five_label_value_rows(doc):
    body = section(doc, "TECHNICAL SKILLS").body
    assert isinstance(body, KeyValues)
    assert [k for k, _ in body.rows] == [
        "Languages", "AI / LLM", "ML / Data", "Data stores", "Cloud & DevOps",
    ]
    values = dict(body.rows)
    # The label must not eat the first technology of its own value.
    assert values["Languages"].startswith("Python")
    assert values["Data stores"].startswith("PostgreSQL")
    # ...and a value wrapped onto the next line is rejoined.
    assert values["ML / Data"].endswith("logistic regression")


@pytest.mark.parametrize("line, expected", [
    ("Languages Python, C, Java", ("Languages", "Python, C, Java")),
    ("AI / LLM LangChain, RAG", ("AI / LLM", "LangChain, RAG")),
    ("Cloud & DevOps Docker, Nginx", ("Cloud & DevOps", "Docker, Nginx")),
    ("Data stores PostgreSQL (replicas)", ("Data stores", "PostgreSQL (replicas)")),
])
def test_label_split_takes_the_shortest_defensible_cut(line, expected):
    assert _split_label(line) == expected


def test_academic_table_recovers_all_six_columns(doc):
    body = section(doc, "ACADEMIC DETAILS").body
    assert isinstance(body, Table)
    assert body.headers == ["COURSE", "SPECIALIZATION", "INSTITUTE/COLLEGE",
                            "BOARD/UNIVERSITY", "SCORE", "YEAR"]
    assert body.rows == [
        ["UG", "CSE (AI AND MACHINE LEARNING)", "VIT Amravati", "VIT",
         "8.71 CGPA", "2025"],
        ["CLASS XII", "", "Stella Maris School", "ICSE", "91.4%", "2020"],
        ["CLASS X", "", "Stella Maris School", "ICSE", "92.8%", "2018"],
    ]


def test_electives_stay_under_academic_details(doc):
    """They are a note on that table, not a table row and not a heading."""
    academic = section(doc, "ACADEMIC DETAILS")
    assert "Software Testing" in academic.text
    assert all("Software Testing" not in " ".join(r) for r in academic.body.rows)
    assert "Subjects / Electives" not in [s.title for s in doc.sections]


def test_certifications_are_one_row_each(doc):
    body = section(doc, "CERTIFICATIONS").body
    assert isinstance(body, Table)
    assert len(body.rows) == 3
    assert body.rows[1] == ["Containers & Kubernetes Essentials", "IBM"]
    assert body.rows[0][1] == "Amazon Web Services (AWS)"


def test_everything_parsed_is_marked_as_the_users_own(doc):
    assert {b.origin for b in doc.all_bullets()} == {"yours"}
    assert all(b.enabled for b in doc.all_bullets())


def test_scored_text_contains_skills_and_education(doc):
    """The scorer sees `as_text()`. If tables were dropped, so would these be."""
    text = doc.as_text()
    for needle in ("PostgreSQL", "LangGraph", "VIT Amravati", "ICSE",
                   "Amazon Web Services", "Software Testing"):
        assert needle in text, needle


def test_json_round_trip_preserves_structure(doc):
    from rba.resume.document import ResumeDoc

    restored = ResumeDoc.from_json(doc.to_json())
    assert restored.as_text() == doc.as_text()
    assert restored.header.fields == doc.header.fields
    original = section(doc, "ACADEMIC DETAILS").body
    assert section(restored, "ACADEMIC DETAILS").body.rows == original.rows
    assert [e.dates for e in section(restored, "PROJECTS").entries] == \
           [e.dates for e in section(doc, "PROJECTS").entries]


def test_unknown_layout_falls_back_without_inventing_structure():
    plain = "Jane Doe\njane@example.com\n\nEXPERIENCE\n- Built a thing\n"
    result = parse_resume(plain)
    assert result.all_bullets()
    assert "Built a thing" in result.as_text()


# -- a second real layout: pipe separators and single completion dates ------
# Both files are text extracted from genuine PDFs. The Clean one broke the
# parser: its headings end in a lone "Nov 2025" rather than a range, and split
# title from subtitle with "|". Neither matched, so two whole projects were
# appended to the previous entry as bullets -- it held nine instead of two.
AGENTS_BITS = pathlib.Path(__file__).parent.parent / "fixtures" / "aniket_agents_bits.txt"
AGENTS_CLEAN = pathlib.Path(__file__).parent.parent / "fixtures" / "aniket_agents_clean.txt"


@pytest.fixture(scope="module")
def clean_doc():
    return parse_bits(AGENTS_CLEAN.read_text())


@pytest.fixture(scope="module")
def bits_doc():
    return parse_bits(AGENTS_BITS.read_text())


def test_single_date_headings_are_entries_not_bullets(clean_doc):
    projects = section(clean_doc, "PROJECTS").entries
    assert [p.title for p in projects] == [
        "Sandboxed Code-Execution Agent",
        "CodeAct Fine-tuning with Execution Feedback",
        "Live Debugging Partner",
        "VisionDev",
    ]
    # The exact regression: one entry had swallowed the other two projects.
    assert [len(p.bullets) for p in projects] == [2, 2, 3, 2]


def test_a_lone_month_and_year_is_kept_as_the_date(clean_doc):
    dates = {p.title: p.dates for p in section(clean_doc, "PROJECTS").entries}
    assert dates["CodeAct Fine-tuning with Execution Feedback"] == "Nov 2025"
    assert dates["Live Debugging Partner"] == "Oct 2026"
    assert dates["VisionDev"] == "Jan 2026 - May 2026"


def test_pipe_separates_title_from_subtitle(clean_doc):
    subtitles = {p.title: p.subtitle for p in section(clean_doc, "PROJECTS").entries}
    assert subtitles["Sandboxed Code-Execution Agent"] == "LLM Agents"
    assert subtitles["CodeAct Fine-tuning with Execution Feedback"] == "LLM Fine-tuning"
    assert subtitles["VisionDev"] == "Computer Vision"


def test_the_dash_layout_still_parses(bits_doc):
    """The looser date rule must not cost us the layout that already worked."""
    projects = section(bits_doc, "PROJECTS").entries
    assert len(projects) == 4
    assert [len(p.bullets) for p in projects] == [3, 2, 2, 2]
    assert projects[0].subtitle == "Interaction Simulation"


@pytest.mark.parametrize("line", [
    "Shipped the migration in Mar 2024.",
    "Reduced p99 latency during the incident of Oct 2025.",
])
def test_a_bullet_ending_in_a_date_does_not_become_a_heading(line):
    """The risk the looser rule introduces, closed deliberately.

    A sentence ends in a full stop; a heading does not. Without this, any
    bullet mentioning a month would be promoted and would steal the bullets
    after it -- the same failure, with a new cause.
    """
    from rba.resume.ingest import _is_heading

    assert not _is_heading(line)
    assert _is_heading("CodeAct Fine-tuning with Execution Feedback | X Nov 2025")


# ---------------------------------------------------------- pdfplumber path --
# `pypdf`'s flat `extract_text()` is reading-order dependent: it happens to
# preserve this fixture's table correctly (see `test_academic_table_recovers_
# all_six_columns` above), but that is not guaranteed for every PDF producer
# -- a live import saw exactly this fragility break (headers and values
# landing in the wrong `Table` cells entirely). These tests reproduce the
# general bug *class* -- a scrambled, column-major-ish extraction order --
# rather than the user's exact original PDF bytes, which were not available
# to build a fixture from; see `PLAN_PIPELINE_FIXES.md` Fix 3 for why that is
# the honest thing to say about this test's provenance.

#: What a badly-ordered PDF extraction can plausibly produce: each header
#: and value on its own line instead of one line per table row. Reproduced
#: directly below (`test_the_old_heuristic_mishandles_a_scrambled_extraction`)
#: to prove this is a real failure mode of the text-only path, not a
#: hypothetical one.
_SCRAMBLED_ACADEMIC_TABLE = """ACADEMIC DETAILS
COURSE
SPECIALIZATION
INSTITUTE/COLLEGE
BOARD/UNIVERSITY
SCORE
YEAR
UG
CSE (AI AND MACHINE LEARNING)
VIT Amravati
VIT
8.71 CGPA
2025
TECHNICAL SKILLS
Languages Python
"""


def _bordered_academic_table_pdf() -> bytes:
    """A real PDF with an actual ruled grid -- built with `fpdf2` (already a
    project dependency) rather than loaded from a file no one has, so this
    test is fully reproducible without external fixtures."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 8)
    headers = ["COURSE", "SPECIALIZATION", "INSTITUTE/COLLEGE",
              "BOARD/UNIVERSITY", "SCORE", "YEAR"]
    col_w = 32
    for h in headers:
        pdf.cell(col_w, 8, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for cell in ["UG", "AI/ML", "VIT Amravati", "VIT", "8.71 CGPA", "2025"]:
        pdf.cell(col_w, 8, cell, border=1)
    pdf.ln()
    return bytes(pdf.output())


def test_the_old_heuristic_mishandles_a_scrambled_extraction():
    """Proves the failure mode this fix addresses is real: a plausible
    column-major-ish extraction order defeats `_parse_table`'s row-boundary
    heuristic (it has no page geometry to fall back on), producing a
    one-column `Table` with every value split across the wrong cells --
    structurally the same shape the live import's broken render showed."""
    doc = parse_bits(_SCRAMBLED_ACADEMIC_TABLE)
    body = section(doc, "ACADEMIC DETAILS").body
    assert body.headers == ["COURSE"], \
        "sanity check: the heuristic really does fail on this input"


def test_pdfplumber_recovers_the_table_a_scrambled_extraction_loses():
    """The fix: when a pdfplumber-detected grid matches this section's
    known header tokens, it is used directly instead of the text heuristic
    -- so the *same* scrambled flat text that the test above shows breaks
    `_parse_table` now still produces a correct `Table`, because the real
    table geometry came from the PDF's page layout, not from guessing at
    reordered text."""
    grids = read_pdf_tables(_bordered_academic_table_pdf())
    assert len(grids) == 1

    doc = parse_bits(_SCRAMBLED_ACADEMIC_TABLE, detected_tables=grids)
    body = section(doc, "ACADEMIC DETAILS").body
    assert body.headers == ["COURSE", "SPECIALIZATION", "INSTITUTE/COLLEGE",
                            "BOARD/UNIVERSITY", "SCORE", "YEAR"]
    assert body.rows == [["UG", "AI/ML", "VIT Amravati", "VIT", "8.71 CGPA", "2025"]]


def _academic_table_pdf_with_merged_heading() -> bytes:
    """A real live-upload failure, reproduced: pdfplumber frequently groups
    a section's own heading line into the *same* detected table as the
    header row directly below it, when nothing in the PDF's geometry
    visually separates them (confirmed against the user's actual resume
    upload, not guessed). `grid[0]` is then the heading, not the header
    row `_pick_table` needs to match against."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 7)
    col_w = 32
    pdf.cell(col_w * 6, 8, "ACADEMIC DETAILS", border=1)
    pdf.ln()
    headers = ["COURSE", "SPECIALIZATION", "INSTITUTE/COLLEGE",
              "BOARD/UNIVERSITY", "SCORE", "YEAR"]
    for h in headers:
        pdf.cell(col_w, 8, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 7)
    for cell in ["UG", "AI/ML", "VIT Amravati", "VIT", "8.71 CGPA", "2025"]:
        pdf.cell(col_w, 8, cell, border=1)
    pdf.ln()
    return bytes(pdf.output())


def test_pdfplumber_recovers_the_table_when_a_heading_is_merged_into_it():
    """The live bug this plan's Fix 3 missed on its first pass: the
    original synthetic test built a table with no heading row above it, so
    `_pick_table` checking only `grid[0]` happened to work there -- but a
    real resume's PDF merges the section heading into the same detected
    grid, and the first fix silently fell back to the broken text
    heuristic for every real upload as a result. `_pick_table` must search
    a few leading rows, not assume row 0 is always the header."""
    grids = read_pdf_tables(_academic_table_pdf_with_merged_heading())
    assert len(grids) == 1
    assert grids[0][0] == ["ACADEMIC DETAILS", "", "", "", "", ""], \
        "sanity check: the heading really is merged in as row 0"

    doc = parse_bits(_SCRAMBLED_ACADEMIC_TABLE, detected_tables=grids)
    body = section(doc, "ACADEMIC DETAILS").body
    assert body.headers == ["COURSE", "SPECIALIZATION", "INSTITUTE/COLLEGE",
                            "BOARD/UNIVERSITY", "SCORE", "YEAR"]
    assert body.rows == [["UG", "AI/ML", "VIT Amravati", "VIT", "8.71 CGPA", "2025"]]


def test_pick_table_will_not_steal_an_unrelated_grid():
    """A detected grid that doesn't look like an academic-details header
    row must not be forced onto this section -- that would silently invent
    structure just as badly as the heuristic it's replacing."""
    unrelated_grid = [["Skill", "Years"], ["Python", "5"]]
    assert _pick_table([unrelated_grid], {"COURSE", "SPECIALIZATION"}) is None


def test_parse_resume_detects_tables_from_real_pdf_bytes():
    """The actual entry point a browser upload goes through: `parse_resume`
    must call `read_pdf_tables` itself when the upload is a PDF, not require
    a caller to do it."""
    pdf_bytes = _bordered_academic_table_pdf()
    # A minimal but real one-page BITS-shaped text body so `looks_like_bits`
    # recognises the format -- `parse_resume` only extracts *text* via
    # pypdf from these bytes (the fpdf2 PDF above), so build the matching
    # scrambled-table text directly rather than relying on pypdf's own
    # extraction of a table this test doesn't actually need text for.
    import rba.resume.ingest as ingest_module

    original_read_pdf = ingest_module.read_pdf
    try:
        ingest_module.read_pdf = lambda data: (
            "SUMMARY\nBuilds things.\n" + _SCRAMBLED_ACADEMIC_TABLE
            + "\nPROJECTS\nThing - X\nJan 2025 - Present\nDid a thing.\n")
        doc = parse_resume(pdf_bytes)
    finally:
        ingest_module.read_pdf = original_read_pdf

    body = section(doc, "ACADEMIC DETAILS").body
    assert body.headers == ["COURSE", "SPECIALIZATION", "INSTITUTE/COLLEGE",
                            "BOARD/UNIVERSITY", "SCORE", "YEAR"], \
        "parse_resume must thread pdfplumber's detected grid through on its own"
