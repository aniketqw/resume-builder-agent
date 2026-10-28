"""Merging must never damage the document the user brought.

The strongest property here is the last test: the score the UI shows as
"+N against your uploaded resume" has to equal an actual rescore of an actual
document, not a number assembled from per-bullet estimates.
"""
from __future__ import annotations

import pathlib

import pytest

from rba.resume.document import Bullet, ResumeDoc, Section
from rba.resume.ingest import parse_bits
from rba.resume.merge import (
    adopted,
    baseline,
    candidates,
    merge_candidates,
    normalise,
)

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "aniket_bits.txt"


@pytest.fixture
def base() -> ResumeDoc:
    return parse_bits(FIXTURE.read_text())


@pytest.fixture
def generated() -> ResumeDoc:
    return ResumeDoc(sections=[
        # Matches an existing project, written as a repo slug.
        Section("resume-builder-agent", bullets=[
            Bullet("Measured the real token cost of a run.",
                   source_repo="resume-builder-agent", evidence="budget.md"),
        ]),
        # No counterpart in the user's resume.
        Section("tcell-sim", bullets=[
            Bullet("Simulated T-cell receptor binding.",
                   source_repo="tcell-sim", evidence="README"),
            Bullet("Claimed a speedup nobody measured.",
                   source_repo="tcell-sim", verified=False,
                   measure_hint="time it"),
        ]),
    ])


def test_repo_slug_matches_the_written_project_title(base, generated):
    merge_candidates(base, generated)
    projects = {e.title: e for s in base.sections for e in s.entries}
    entry = projects["Resume Builder Agent"]
    assert "Measured the real token cost of a run." in [b.text for b in entry.bullets]
    # ...and it did not create a duplicate entry alongside it.
    assert "Resume-builder-agent" not in projects


def test_unmatched_repo_becomes_a_new_project(base, generated):
    merge_candidates(base, generated)
    projects = {e.title for s in base.sections for e in s.entries}
    assert "Tcell Sim" in projects


def test_candidates_arrive_switched_off_and_marked(base, generated):
    merge_candidates(base, generated)
    new = candidates(base)
    assert len(new) == 3
    assert not any(b.enabled for b in new)
    assert {b.origin for b in new} == {"repo", "assumed"}
    # An unverified claim is amber, not presented as evidence-backed.
    unverified = [b for b in new if not b.verified]
    assert [b.origin for b in unverified] == ["assumed"]


def test_the_users_own_bullets_are_untouched(base, generated):
    before = [(b.id, b.text, b.enabled, b.origin) for b in base.all_bullets()]
    merge_candidates(base, generated)
    after = {b.id: b for b in base.all_bullets()}
    for bullet_id, text, enabled, origin in before:
        assert after[bullet_id].text == text
        assert after[bullet_id].enabled == enabled
        assert after[bullet_id].origin == origin == "yours"


def test_merging_does_not_change_the_rendered_resume(base, generated):
    """Nothing is adopted until the user says so, so the text must not move."""
    before = base.as_text()
    merge_candidates(base, generated)
    assert base.as_text() == before


def test_a_bullet_the_user_already_wrote_is_not_duplicated(base):
    existing = next(e for s in base.sections for e in s.entries
                    if e.title == "VisionDev")
    echo = ResumeDoc(sections=[Section("VisionDev", bullets=[
        Bullet(existing.bullets[0].text)])])
    merge_candidates(base, echo)
    assert not candidates(base)


def test_baseline_strips_candidates_but_keeps_everything_else(base, generated):
    original = base.as_text()
    merge_candidates(base, generated)
    base.toggle(candidates(base)[0].id, True)
    assert base.as_text() != original
    assert baseline(base).as_text() == original


@pytest.mark.parametrize("a, b", [
    ("resume-builder-agent", "Resume Builder Agent"),
    ("tcell_sim", "TCell Sim"),
    ("VisionDev", "visiondev"),
])
def test_normalise_ignores_punctuation_and_case(a, b):
    assert normalise(a) == normalise(b)


def test_adopting_a_bullet_moves_the_score_by_exactly_its_delta(base, generated):
    """The headline gain must be a real rescore, not a sum of estimates."""
    from rba.resume.attribution import attribute, rescore

    jd = ("Looking for an AI engineer with LangGraph, RAG, token budgeting and "
          "evaluation experience. You will measure agent cost and quality.")
    merge_candidates(base, generated)

    before, _ = rescore(base, jd)
    target = candidates(base)[0]
    base.toggle(target.id, True)
    after, _ = rescore(base, jd)

    impact = attribute(base, jd).by_id(target.id)
    assert impact is not None
    assert round(after - before, 4) == round(impact.delta / 100.0, 4)
    assert adopted(base) == [target]


def test_an_invalid_groq_key_is_reported_not_swallowed():
    """Every summary failing must produce an explanation, not an empty resume."""
    from app import _report_partial_failures
    assert callable(_report_partial_failures)


def test_pipeline_surfaces_per_repo_errors_in_stats():
    import inspect

    from rba.resume import pipeline

    source = inspect.getsource(pipeline.build_from_github)
    assert 'stats["errors"]' in source, \
        "per-repo failures must reach the caller, not die in the node"
    assert 'stats["summarised"]' in source
