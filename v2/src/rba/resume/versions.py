"""The three résumés: what you wrote, what the evidence supports, what scores.

One pipeline, three documents, each scored by the same code so the numbers are
comparable:

* **uploaded** -- the résumé as parsed, untouched.
* **grounded** -- every project point checked against its repository, with
  unsupported spans cut out and un-cuttable ones flagged. This score is often
  *lower* than uploaded, and that is the point: the gap is what the résumé was
  claiming without support.
* **improved** -- grounded, plus the additions the evidence supports that are
  not already said.

The ordering matters. Improving before grounding would stack suggestions on top
of claims the repositories do not back, so the gain would be measured against a
document that should not have existed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .attribution import rescore
from .document import ResumeDoc
from .grounding import GroundingReport

UPLOADED = "uploaded"
GROUNDED = "grounded"
IMPROVED = "improved"


@dataclass
class Version:
    """One résumé, its score, and why it differs from the one before it."""

    name: str
    document: ResumeDoc
    score: float = 0.0
    note: str = ""

    def delta_from(self, other: Version | None) -> float:
        return 0.0 if other is None else round(self.score - other.score, 2)


@dataclass
class VersionSet:
    versions: list[Version] = field(default_factory=list)
    report: GroundingReport | None = None

    def get(self, name: str) -> Version | None:
        return next((v for v in self.versions if v.name == name), None)

    @property
    def uploaded(self) -> Version | None:
        return self.get(UPLOADED)

    @property
    def grounded(self) -> Version | None:
        return self.get(GROUNDED)

    @property
    def improved(self) -> Version | None:
        return self.get(IMPROVED)

    def honesty_cost(self) -> float:
        """What grounding cost. Negative means the résumé was overclaiming."""
        if not (self.uploaded and self.grounded):
            return 0.0
        return self.grounded.delta_from(self.uploaded)

    def improvement_gain(self) -> float:
        if not (self.grounded and self.improved):
            return 0.0
        return self.improved.delta_from(self.grounded)


def _score(doc: ResumeDoc, jd_text: str, required: list[str] | None,
           preferred: list[str] | None) -> float:
    raw, _channels = rescore(doc, jd_text, required_skills=required,
                             preferred_skills=preferred)
    return round(raw * 100, 2)


def build_versions(uploaded: ResumeDoc, jd_text: str, *,
                   report: GroundingReport | None = None,
                   improved: ResumeDoc | None = None,
                   required_skills: list[str] | None = None,
                   preferred_skills: list[str] | None = None) -> VersionSet:
    """Score whichever versions exist. Each score is a real `rescore`."""
    required, preferred = required_skills or [], preferred_skills or []
    versions = [Version(UPLOADED, uploaded,
                        _score(uploaded, jd_text, required, preferred),
                        "as you supplied it")]

    if report is not None and report.document is not None:
        trimmed = len(report.by_status("trimmed"))
        flagged = len(report.by_status("flagged"))
        note = report.summary()
        if trimmed or flagged:
            note += " — the gap below is what was being claimed unsupported"
        versions.append(Version(GROUNDED, report.document,
                                _score(report.document, jd_text, required,
                                       preferred), note))

    if improved is not None:
        versions.append(Version(IMPROVED, improved,
                                _score(improved, jd_text, required, preferred),
                                "grounded, plus what the evidence also shows"))

    return VersionSet(versions=versions, report=report)


def apply_additions(grounded: ResumeDoc, additions) -> ResumeDoc:
    """A copy of `grounded` with every usable addition inserted.

    Additions arrive enabled, because this document exists to answer "what is
    the best this évidence supports?" -- the editor is where individual lines
    get switched off.
    """
    from ..nodes.additions import AdditionSet
    from .document import Bullet

    improved = ResumeDoc.from_dict(grounded.to_dict())
    if not isinstance(additions, AdditionSet):
        return improved

    for section in improved.sections:
        for index, entry in enumerate(section.entries):
            key = f"{section.title}#{index}"
            for addition in additions.for_project(key):
                if not addition.usable:
                    continue
                entry.bullets.append(Bullet(
                    text=addition.text, origin="repo", verified=True,
                    evidence="proposed from repository evidence"))
    return improved
