"""Proposing extra points without repeating what the resume already says.

Offline. The thresholds exercised here were measured by
`scripts/calibrate_novelty.py` on real bullets paired with rewordings of
themselves; that script reports 0 missed and 0 false duplicates.
"""
from __future__ import annotations

import pathlib

import pytest

from rba.nodes.additions import (
    ADDED,
    DUPLICATE,
    DUPLICATE_COSINE,
    DUPLICATE_OVERLAP,
    UNSUPPORTED,
    evaluate,
    find_duplicate,
    word_overlap,
)
from rba.resume.document import Bullet
from rba.resume.ingest import parse_bits

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "aniket_agents_clean.txt"

EVIDENCE = (
    "README: LangGraph agent runs code in a Docker sandbox. Tools: shell, "
    "python, file I/O, HTTP, Playwright. Provider-agnostic LLM layer with "
    "Anthropic and Ollama. 14 integration tests. Adds a retry policy and a "
    "streaming output channel."
)
JD = "AI Engineer. Python, LangGraph, Docker, agents, tooling, evaluation."


@pytest.fixture(scope="module")
def doc():
    return parse_bits(FIXTURE.read_text())


@pytest.fixture(scope="module")
def project_key(doc):
    for section in doc.sections:
        if section.title != "PROJECTS":
            continue
        for index, entry in enumerate(section.entries):
            if "Sandboxed" in entry.title:
                return f"{section.title}#{index}"
    raise AssertionError("fixture changed")


def run(doc, project_key, texts):
    result = evaluate(doc, JD, {project_key: texts}, {project_key: EVIDENCE},
                      use_dense=True)
    return {a.text: a for a in result.items}


def test_a_reworded_existing_point_is_caught_as_a_duplicate(doc, project_key):
    reworded = ("Engineered a LangGraph agent executing code inside an isolated "
                "Docker sandbox with shell and Playwright tooling.")
    item = run(doc, project_key, [reworded])[reworded]
    assert item.status == DUPLICATE
    assert item.duplicate_of, "it must name the line it restates"
    assert item.similarity > DUPLICATE_COSINE


def test_a_duplicate_becomes_a_rewrite_not_a_deletion(doc, project_key):
    """Your reasoning: if two lines say the same thing, improve the one there."""
    reworded = ("Engineered a LangGraph agent executing code inside an isolated "
                "Docker sandbox with shell and Playwright tooling.")
    item = run(doc, project_key, [reworded])[reworded]
    assert "rewrite" in item.reason
    assert item.gain == 0.0, "a duplicate is never scored as an addition"


def test_a_genuinely_new_point_is_added_and_really_scored(doc, project_key):
    fresh = "Added a retry policy and a streaming output channel to the agent runtime."
    item = run(doc, project_key, [fresh])[fresh]
    assert item.status == ADDED
    assert not item.duplicate_of
    # The gain is a real re-score, so it is allowed to be negative. What it may
    # not be is invented: `usable` gates on it actually helping.
    assert item.usable == (item.gain > 0)


def test_an_unsupported_addition_is_refused(doc, project_key):
    invented = "Scaled the sandbox to 2 million executions per day."
    item = run(doc, project_key, [invented])[invented]
    assert item.status == UNSUPPORTED
    assert "not shown in the repository" in item.reason


def test_the_gain_equals_a_real_rescore(doc, project_key):
    """No estimate anywhere: insert the point and score the document."""
    from rba.resume.attribution import rescore

    fresh = "Added a retry policy and a streaming output channel to the agent runtime."
    item = run(doc, project_key, [fresh])[fresh]

    before, _ = rescore(doc, JD)
    entry = next(e for s in doc.sections if s.title == "PROJECTS"
                 for e in s.entries if "Sandboxed" in e.title)
    bullet = Bullet(text=fresh, origin="repo")
    entry.bullets.append(bullet)
    try:
        after, _ = rescore(doc, JD)
    finally:
        entry.bullets.remove(bullet)
    assert round((after - before) * 100, 4) == pytest.approx(item.gain, abs=1e-4)


def test_the_document_is_left_exactly_as_it_was(doc, project_key):
    before = doc.as_text()
    run(doc, project_key, ["Added a retry policy to the agent runtime.",
                           "Scaled the sandbox to 2 million executions per day."])
    assert doc.as_text() == before


def test_overlap_is_normalised_by_the_smaller_side():
    """A short point wholly inside a longer one is a duplicate, not a sliver."""
    short = "Built a Docker sandbox for untrusted code."
    longer = ("Built a Docker sandbox for untrusted code with tools for shell, "
              "Python execution, file I/O, HTTP requests and browser automation.")
    assert word_overlap(short, longer) >= DUPLICATE_OVERLAP


def test_nothing_matches_when_there_is_nothing_to_match():
    twin, score = find_duplicate("Anything at all.", [])
    assert twin is None and score == 0.0
