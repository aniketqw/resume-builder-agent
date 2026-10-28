"""The editable resume document and its per-bullet scoring.

Two pieces:

* :mod:`rba.resume.document` — the document itself. Projects clubbed into
  sections of bullets, each bullet carrying a stable id and the evidence it came
  from, so the UI can toggle one and we can still prove where it came from.
* :mod:`rba.resume.attribution` — what each bullet is worth, by exact
  leave-one-out.
"""
from .attribution import Attribution, BulletImpact, attribute, rescore
from .document import Bullet, ResumeDoc, Section, from_summaries
from .pipeline import build_from_github, parse_resume_text

__all__ = [
    "Attribution",
    "Bullet",
    "BulletImpact",
    "ResumeDoc",
    "Section",
    "attribute",
    "build_from_github",
    "from_summaries",
    "parse_resume_text",
    "rescore",
]
