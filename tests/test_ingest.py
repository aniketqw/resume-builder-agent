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
    _split_label,
    looks_like_bits,
    parse_bits,
    parse_resume,
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
