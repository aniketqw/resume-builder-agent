"""The three résumés and the scores that compare them.

The property under test throughout: a version's score is never carried over or
estimated, it is always a fresh `rescore()` of that exact document. If
`build_versions` ever started caching or approximating a score, the
`test_each_version_score_is_a_real_rescore` test below would be the one to
catch it.
"""
from __future__ import annotations

import pathlib

import pytest

from rba.nodes.additions import ADDED, DUPLICATE, UNSUPPORTED, Addition, AdditionSet
from rba.resume.attribution import rescore
from rba.resume.document import Bullet, Entry, ResumeDoc, Section
from rba.resume.grounding import ground_document
from rba.resume.ingest import parse_bits
from rba.resume.versions import (
    GROUNDED,
    IMPROVED,
    UPLOADED,
    Version,
    VersionSet,
    apply_additions,
    build_versions,
)

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "aniket_agents_clean.txt"
JD = "AI Engineer. Python, LangGraph, Docker, agents, tooling, evaluation."

# A figure the job description cares about, but the repository never claims.
# Used to make the honesty cost genuinely negative -- cutting it both changes
# the document and removes something the JD was rewarding. Long enough prose
# to clear grounding.MIN_README_CHARS, otherwise the evidence is judged too
# thin to rule on anything and every point comes back unverifiable.
OVERCLAIM_EVIDENCE = (
    "README: Flask API serves model predictions, deployed on Kubernetes with "
    "autoscaling. Covered by 12 unit tests. The service exposes a single "
    "/predict endpoint behind a load balancer, with health checks and a "
    "readiness probe wired into the cluster so a failed pod is taken out of "
    "rotation automatically. Configuration is supplied through environment "
    "variables, and the Docker image is built in CI on every merge to main."
)
OVERCLAIM_JD = ("Backend engineer. Kubernetes, autoscaling, high-throughput "
               "request handling, 10000 requests per second.")


@pytest.fixture(scope="module")
def doc():
    return parse_bits(FIXTURE.read_text())


def overclaiming_doc() -> ResumeDoc:
    """A document whose project carries a figure the evidence never shows.

    The figure sits in a parenthetical, so grounding gets a clean cut
    (TRIMMED, not FLAGGED), and it names something the job description
    rewards, so the cut is guaranteed to cost real score -- not just change
    the text incidentally.
    """
    return ResumeDoc(sections=[Section("PROJECTS", entries=[Entry(
        title="InferenceService",
        bullets=[
            Bullet("Built a Flask API that serves model predictions."),
            Bullet("Deployed the service on Kubernetes with autoscaling "
                   "(processing 10,000 requests per second)."),
        ])])])


# -- scores are real rescores, never carried over --------------------------

def test_each_version_score_is_a_real_rescore(doc):
    """Every score in a VersionSet must equal a direct rescore of that document.

    Catches the regression of a version's score being copied from another
    version, or estimated, rather than computed from its own text.
    """
    uploaded = doc
    report = ground_document(uploaded, {})
    improved = ResumeDoc.from_dict(uploaded.to_dict())

    versions = build_versions(uploaded, JD, report=report, improved=improved,
                              required_skills=["python"],
                              preferred_skills=["docker"])

    for version in versions.versions:
        expected_raw, _ = rescore(version.document, JD,
                                  required_skills=["python"],
                                  preferred_skills=["docker"])
        assert version.score == round(expected_raw * 100, 2)


# -- ordering and deltas ----------------------------------------------------

def test_versions_are_ordered_uploaded_grounded_improved(doc):
    report = ground_document(doc, {})
    improved = ResumeDoc.from_dict(doc.to_dict())
    versions = build_versions(doc, JD, report=report, improved=improved)
    assert [v.name for v in versions.versions] == [UPLOADED, GROUNDED, IMPROVED]


def test_delta_from_computes_the_gap_between_consecutive_versions():
    earlier = Version("a", ResumeDoc(), score=40.0)
    later = Version("b", ResumeDoc(), score=47.5)
    assert later.delta_from(earlier) == 7.5
    assert earlier.delta_from(later) == -7.5
    assert earlier.delta_from(None) == 0.0


# -- honesty cost: grounding removing unsupported claims ---------------------

def test_honesty_cost_is_reported_when_grounding_changes_the_document():
    """An overclaiming résumé must show its honesty cost, not a silent zero.

    The bullet carries "40%" which EVIDENCE never states, so grounding trims
    it; the grounded document differs from the uploaded one and the cost must
    reflect that -- not whatever sign happens to be convenient.
    """
    doc = overclaiming_doc()
    report = ground_document(doc, {"PROJECTS#0": OVERCLAIM_EVIDENCE},
                             {"PROJECTS#0": "inference_service"})
    assert "10,000" not in report.document.as_text()
    assert "10,000" in doc.as_text(), "the document passed in must stay untouched"

    versions = build_versions(doc, OVERCLAIM_JD, report=report)
    uploaded_score, _ = rescore(doc, OVERCLAIM_JD)
    grounded_score, _ = rescore(report.document, OVERCLAIM_JD)
    expected = round(round(grounded_score * 100, 2) - round(uploaded_score * 100, 2), 2)

    assert versions.honesty_cost() == expected
    assert versions.honesty_cost() != 0.0, \
        "grounding changed the document, so the cost must be non-zero"
    assert versions.honesty_cost() < 0, \
        "cutting an unsupported figure should not be rewarded with a higher score"


def test_honesty_cost_is_zero_without_both_uploaded_and_grounded():
    versions = VersionSet(versions=[Version(UPLOADED, ResumeDoc(), score=50.0)])
    assert versions.honesty_cost() == 0.0


# -- improvement gain ---------------------------------------------------------

def test_improvement_gain_is_improved_minus_grounded(doc):
    report = ground_document(doc, {})
    improved = ResumeDoc.from_dict(doc.to_dict())
    projects = next(s for s in improved.sections if s.title == "PROJECTS")
    projects.entries[0].bullets.append(
        Bullet("An extra point not in the original.", origin="repo"))

    versions = build_versions(doc, JD, report=report, improved=improved)
    expected = versions.improved.delta_from(versions.grounded)
    assert versions.improvement_gain() == expected
    assert versions.improvement_gain() == round(
        versions.improved.score - versions.grounded.score, 2)


def test_improvement_gain_is_zero_without_an_improved_version(doc):
    report = ground_document(doc, {})
    versions = build_versions(doc, JD, report=report)
    assert versions.improvement_gain() == 0.0


# -- apply_additions: only usable additions reach the document --------------

def grounded_doc_for_additions() -> ResumeDoc:
    return ResumeDoc(sections=[Section("PROJECTS", entries=[Entry(
        title="VisionDev",
        bullets=[Bullet("Built a PyTorch Lightning callback pipeline.")],
    )])])


def test_apply_additions_only_inserts_usable_additions():
    """usable == status ADDED and gain > 0 -- everything else must be dropped."""
    grounded = grounded_doc_for_additions()
    additions = AdditionSet(items=[
        Addition("PROJECTS#0", "A genuinely new, scoring point.", ADDED, gain=5.0),
        Addition("PROJECTS#0", "An added point that did not actually help.",
                 ADDED, gain=0.0),
        Addition("PROJECTS#0", "A point that restates an existing one.",
                 DUPLICATE, 0.0, "some-bullet-id", 0.9, "rewrite instead"),
        Addition("PROJECTS#0", "A point the repository does not support.",
                 UNSUPPORTED, reason="not shown in the repository"),
    ])

    improved = apply_additions(grounded, additions)
    text = improved.as_text()

    assert "A genuinely new, scoring point." in text
    assert "An added point that did not actually help." not in text
    assert "A point that restates an existing one." not in text
    assert "A point the repository does not support." not in text


def test_apply_additions_does_not_mutate_the_input_document():
    grounded = grounded_doc_for_additions()
    before = grounded.as_text()
    additions = AdditionSet(items=[
        Addition("PROJECTS#0", "A genuinely new, scoring point.", ADDED, gain=5.0),
    ])

    apply_additions(grounded, additions)

    assert grounded.as_text() == before


def test_apply_additions_with_non_addition_set_returns_an_unchanged_copy():
    """Defensive branch: anything that is not an AdditionSet adds nothing."""
    grounded = grounded_doc_for_additions()
    improved = apply_additions(grounded, None)
    assert improved.as_text() == grounded.as_text()
    assert improved is not grounded


# -- partial version sets -----------------------------------------------------

def test_build_versions_with_no_report_yields_only_uploaded(doc):
    versions = build_versions(doc, JD)
    assert [v.name for v in versions.versions] == [UPLOADED]
    assert versions.get(GROUNDED) is None
    assert versions.get(IMPROVED) is None


def test_build_versions_with_report_but_no_improved_yields_two(doc):
    report = ground_document(doc, {})
    versions = build_versions(doc, JD, report=report)
    assert [v.name for v in versions.versions] == [UPLOADED, GROUNDED]
    assert versions.get(IMPROVED) is None


def test_build_versions_with_a_report_whose_document_is_none_yields_only_uploaded(doc):
    """A report that never ran (no document) must not fabricate a grounded version."""
    from rba.resume.grounding import GroundingReport

    report = GroundingReport(checks=[], document=None)
    versions = build_versions(doc, JD, report=report)
    assert [v.name for v in versions.versions] == [UPLOADED]


def test_get_returns_none_for_an_absent_name(doc):
    versions = build_versions(doc, JD)
    assert versions.get("nonexistent") is None
