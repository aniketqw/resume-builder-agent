"""Checking a résumé against its own repositories.

The property under test throughout: this code may delete, never write. Every
sentence it emits is the author's sentence or the author's sentence minus a
span, and `is_subsequence_edit` makes that mechanical rather than a matter of
judgement.
"""
from __future__ import annotations

import pytest

from rba.resume.document import Bullet, Entry, ResumeDoc, Section
from rba.resume.grounding import (
    FLAGGED,
    TRIMMED,
    UNVERIFIABLE,
    VERIFIED,
    check_point,
    ground_document,
    is_subsequence_edit,
    strip_unsupported,
)

EVIDENCE = (
    "README: PyTorch Lightning callback logs CIFAR-10 misclassifications and "
    "clusters them by distortion type: blur, JPEG, pixelation, noise. "
    "A YOLO classifier labels failures. Covered by 34 unit tests."
)


def check(text: str, evidence: str = EVIDENCE):
    return check_point(Bullet(text=text), evidence, repo="vision_dev_project")


def test_a_fully_supported_point_is_untouched():
    text = "Built a PyTorch Lightning callback pipeline that logs CIFAR-10 misclassifications."
    result = check(text)
    assert result.status == VERIFIED
    assert result.text == text


def test_a_number_present_in_the_evidence_survives():
    """Verification must not punish a true figure."""
    result = check("Covered the engine with 34 unit tests across the pipeline.")
    assert result.status == VERIFIED
    assert "34" in result.text


@pytest.mark.parametrize("text, gone", [
    ("Built a PyTorch Lightning pipeline that clusters failures; "
     "blur caused 40% of failures.", "40%"),
    ("Built a PyTorch Lightning pipeline (blur caused 40% of failures) "
     "for CIFAR-10.", "40%"),
])
def test_an_unsupported_figure_is_cut_out_cleanly(text, gone):
    result = check(text)
    assert result.status == TRIMMED
    assert gone not in result.text
    assert result.text != result.original
    assert is_subsequence_edit(result.original, result.text)


def test_a_claim_with_no_clean_cut_is_flagged_not_mangled():
    """The honest fallback.

    "blur caused 40% of failures." has no parenthetical and no trailing clause
    to remove; cutting the figure alone would leave "blur caused of failures".
    Reporting the problem beats emitting a broken sentence.
    """
    result = check("blur caused 40% of failures.")
    assert result.status == FLAGGED
    assert result.text == result.original
    assert "40" in result.reason()


def test_an_unevidenced_technology_is_caught():
    result = check("Built a PyTorch Lightning pipeline on Kubernetes that "
                   "logs CIFAR-10 misclassifications.")
    assert result.status == FLAGGED
    assert "kubernetes" in result.reason()


def test_no_evidence_means_unverifiable_not_wrong():
    """A project with no repository keeps every word it had."""
    text = "Built something the repositories cannot speak to."
    result = check(text, evidence="")
    assert result.status == UNVERIFIABLE
    assert result.text == text
    assert result.ok, "unverifiable is not a failure, it is an absence"


# -- the invariant ---------------------------------------------------------
def test_subsequence_edit_accepts_removal_and_rejects_addition():
    original = "Built a pipeline that clusters failures by distortion type."
    assert is_subsequence_edit(original, "Built a pipeline that clusters failures.")
    assert is_subsequence_edit(original, original)
    assert not is_subsequence_edit(original, original + " It doubled throughput.")
    assert not is_subsequence_edit(original, "Engineered a pipeline that clusters.")


def test_stripping_never_invents_a_word():
    """Fuzz the stripper over the real bullets and assert the contract holds."""
    import pathlib

    from rba.resume.ingest import parse_bits

    fixture = pathlib.Path(__file__).parent.parent / "fixtures" / "aniket_agents_clean.txt"
    doc = parse_bits(fixture.read_text())
    for bullet in doc.all_bullets():
        for offender in ("40%", "1,200", "18.7", "Kubernetes"):
            out = strip_unsupported(bullet.text, [offender])
            if out is not None:
                assert is_subsequence_edit(bullet.text, out), bullet.text


def test_grounding_a_document_reports_and_rewrites_in_place():
    doc = ResumeDoc(sections=[Section("PROJECTS", entries=[Entry(
        title="VisionDev",
        bullets=[
            Bullet("Built a PyTorch Lightning callback pipeline that logs "
                   "CIFAR-10 misclassifications."),
            Bullet("Clustered failures by distortion type; blur caused 40% "
                   "of failures."),
        ])])])
    report = ground_document(doc, {"PROJECTS#0": EVIDENCE},
                             {"PROJECTS#0": "vision_dev_project"})

    assert len(report.checks) == 2
    assert [c.status for c in report.checks] == [VERIFIED, TRIMMED]
    assert "40%" not in report.document.as_text()
    # The original document is not mutated.
    assert "40%" in doc.as_text()
    assert "2 verified" not in report.summary()
    assert "1 trimmed" in report.summary()


def test_a_project_without_a_repo_keeps_all_its_points():
    doc = ResumeDoc(sections=[Section("PROJECTS", entries=[Entry(
        title="Bounty Platform",
        bullets=[Bullet("Built a platform with Next.js and Express in a team.")],
    )])])
    report = ground_document(doc, {})
    assert [c.status for c in report.checks] == [UNVERIFIABLE]
    assert "Next.js and Express" in report.document.as_text()


# -- absence of evidence is not evidence of absence ------------------------
# The first version of this module got it badly wrong on real repositories.
# `vision_dev_project` has 4,884 files, no README and no parsed dependency
# manifest, so nothing in its evidence mentions PyTorch -- and the checker
# reported that a true PyTorch claim was unsupported. Acting on that means
# deleting things that are true, which is worse than not checking at all.
def bundle(readme: str = "", deps: list[str] | None = None,
           commits: list[str] | None = None) -> str:
    import json

    payload: dict = {"repo": "example"}
    if readme:
        payload["readme"] = {"path": "README.md", "text": readme}
    if deps:
        payload["dependencies"] = deps
    if commits:
        payload["commits"] = [{"sha": "a", "message": m} for m in commits]
    return json.dumps(payload)


def test_a_repo_with_nothing_in_it_judges_nothing():
    from rba.resume.grounding import evidence_strength

    strength = evidence_strength(bundle(commits=["fix", "wip", "update"]))
    assert not strength.numbers
    assert not strength.technologies
    assert "no README and no dependency manifest" in strength.why


def test_a_true_claim_is_not_flagged_by_an_empty_repository():
    """The exact false positive, as a regression test."""
    evidence = bundle(commits=["refactor pipeline", "tidy logging"])
    result = check_point(
        Bullet(text="Built a PyTorch Lightning callback pipeline that logs "
                    "CIFAR-10 misclassifications; blur caused 40% of failures."),
        evidence, repo="vision_dev_project")
    assert result.status == UNVERIFIABLE
    assert result.text == result.original
    assert "cannot confirm or deny" in result.reason() or \
        "confirm or deny" in result.reason()


def test_a_manifest_licenses_technology_checks_but_not_figures():
    """A dependency list names technologies; it says nothing about results."""
    from rba.resume.grounding import evidence_strength

    strength = evidence_strength(bundle(deps=["langgraph", "docker"]))
    assert strength.technologies
    assert not strength.numbers, "a manifest cannot confirm a measured figure"

    checked = check_point(
        Bullet(text="Built a LangGraph agent in Docker across 14 test cases."),
        bundle(deps=["langgraph", "docker"]))
    # 14 is unjudgeable here, and langgraph/docker are both present.
    assert checked.status == VERIFIED
    assert not checked.unsupported_numbers


def test_a_readme_licenses_both_checks():
    from rba.resume.grounding import evidence_strength

    readme = ("This project builds a LangGraph agent. " * 12)
    strength = evidence_strength(bundle(readme=readme))
    assert strength.numbers and strength.technologies
    assert strength.why == ""


def test_an_unsupported_figure_is_still_caught_when_there_is_a_readme():
    """The gate must not become a blanket excuse."""
    readme = ("The simulator compares four interaction channels for a "
              "debugging agent across scripted sessions. " * 6)
    result = check_point(
        Bullet(text="Built a simulator comparing four channels; it ran "
                    "99,999 sessions."),
        bundle(readme=readme))
    assert result.status in (TRIMMED, FLAGGED)
    assert "99,999" in " ".join(result.unsupported_numbers) or \
        "99999" in " ".join(result.unsupported_numbers)
