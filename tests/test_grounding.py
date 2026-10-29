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
