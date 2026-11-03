"""Matching a résumé project to the repository it is actually about.

The failure this guards against is specific and severe: attaching a project to
the wrong repository makes every later step verify real claims against
unrelated evidence, which strips true statements and keeps false ones. So the
tests care far more about *not matching wrongly* than about matching often.

Offline: repositories are constructed, never fetched. The thresholds they
exercise were measured on a real account by
`scripts/calibrate_matching.py` -- see that script's output.
"""
from __future__ import annotations

import pathlib

import pytest

from rba.resume.ingest import parse_bits
from rba.resume.matching import (
    STRONG_CONFIDENCE,
    STRONG_MARGIN,
    _name_bonus,
    match_projects,
    repo_text,
)
from rba.state import RepoRef

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "aniket_agents_clean.txt"

# A slice of the real account, including the two repositories that caused
# trouble: `code`, whose name substring-matches two unrelated projects, and
# `sandbox-agent`, whose name matches nothing by string comparison at all.
REPOS = [
    RepoRef(name="live-debugging-partner", description="", language="Python"),
    RepoRef(name="sandbox-agent", description="", language="Python"),
    RepoRef(name="CodeActImplementation", description="", language="Python"),
    RepoRef(name="vision_dev_project", description="", language="Python"),
    RepoRef(name="code", description="", language="Python"),
    RepoRef(name="agentic", description="", language="Python"),
    RepoRef(name="webdev100xdevNotes", description="", language="JavaScript"),
    RepoRef(name="interview-prep", description="Interview prep: 200 Q&A"),
]


@pytest.fixture(scope="module")
def matches():
    doc = parse_bits(FIXTURE.read_text())
    return {m.project_title: m for m in match_projects(doc, REPOS)}


def test_the_generic_repo_never_wins(matches):
    """`code` substring-matches two project titles and is about neither.

    This is the precise false positive that made name matching unusable.
    """
    assert matches["Sandboxed Code-Execution Agent"].repo != "code"
    assert matches["CodeAct Fine-tuning with Execution Feedback"].repo != "code"


def test_projects_whose_names_do_not_match_are_still_found(matches):
    """None of these three share a name with their repository."""
    assert matches["Sandboxed Code-Execution Agent"].repo == "sandbox-agent"
    assert matches["VisionDev"].repo == "vision_dev_project"
    assert matches["CodeAct Fine-tuning with Execution Feedback"].repo == \
        "CodeActImplementation"


def test_an_exact_name_match_is_the_most_confident(matches):
    best = matches["Live Debugging Partner"]
    assert best.repo == "live-debugging-partner"
    assert best.strong
    assert best.confidence > 0.9


def test_non_project_sections_are_never_matched():
    """An internship has no repository.

    Including them was actively harmful: in calibration the Marvell internship
    matched `live-debugging-partner` at 0.658 -- higher than genuine projects
    that have no repo at all.
    """
    doc = parse_bits(FIXTURE.read_text())
    titles = {m.project_title for m in match_projects(doc, REPOS)}
    assert not any("Marvell" in t for t in titles)
    assert not any("Teaching Assistant" in t for t in titles)


def test_a_weak_match_is_not_auto_accepted():
    """Strong requires confidence AND margin. Either alone is insufficient.

    Measured: true matches ran 0.697-0.957 confidence while projects with no
    repository at all ran 0.686-0.696. A gap of 0.001 is not a decision
    procedure, so a second, independent signal is required.
    """
    from rba.resume.matching import Match

    clear = Match("k", "p", "r", 0.90, runners_up=(("other", 0.70),))
    high_but_ambiguous = Match("k", "p", "r", 0.90, runners_up=(("other", 0.89),))
    clear_but_low = Match("k", "p", "r", 0.60, runners_up=(("other", 0.30),))

    assert clear.strong
    assert not high_but_ambiguous.strong, "a near-tie must be confirmed by a human"
    assert not clear_but_low.strong, "low similarity must be confirmed by a human"
    assert high_but_ambiguous.margin == pytest.approx(0.01)
    assert STRONG_CONFIDENCE > 0.5 and STRONG_MARGIN > 0.0


def test_name_bonus_refuses_short_generic_names():
    assert _name_bonus("Sandboxed Code-Execution Agent", "code") == 0.0
    assert _name_bonus("Live Debugging Partner", "live-debugging-partner") > 0.0
    assert _name_bonus("CodeAct Fine-tuning", "CodeActImplementation") == 0.0
    # Token containment, not substring: the repo starts with the project slug.
    assert _name_bonus("CodeAct", "CodeActImplementation") > 0.0


def test_repo_text_splits_names_into_words():
    """`vision_dev_project` must read as words, or it matches nothing."""
    text = repo_text(RepoRef(name="vision_dev_project"))
    assert "vision dev project" in text


def test_every_project_gets_a_row_even_with_no_repositories():
    """A project with no match is a fact to show, not a row to drop."""
    doc = parse_bits(FIXTURE.read_text())
    results = match_projects(doc, [])
    assert len(results) == 4
    assert all(not m.repo and not m.strong for m in results)
    assert all("nothing scored high enough" in m.why() for m in results)
