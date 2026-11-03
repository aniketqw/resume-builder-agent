"""The editable resume document and its per-bullet scoring.

Two pieces:

* :mod:`rba.resume.document` — the document itself. Projects clubbed into
  sections of bullets, each bullet carrying a stable id and the evidence it came
  from, so the UI can toggle one and we can still prove where it came from.
* :mod:`rba.resume.attribution` — what each bullet is worth, by exact
  leave-one-out.
* :mod:`rba.resume.ingest` — a resume PDF back into that document, without an
  LLM, so nothing is smoothed over or invented on the way in.
* :mod:`rba.resume.merge` — generated lines added as opt-in candidates, never
  rewriting what the user brought.
* :mod:`rba.resume.render` — the same document as plain text, HTML, PDF or
  LaTeX.
* :mod:`rba.resume.diagnostics` — advice, kept deliberately out of the score.
"""
from .attribution import Attribution, BulletImpact, attribute, rescore
from .diagnostics import Diagnostic, diagnose
from .document import (
    Bullet,
    Entry,
    Header,
    KeyValues,
    ResumeDoc,
    Section,
    Table,
    from_summaries,
)
from .ingest import parse_bits, parse_resume, read_pdf
from .merge import baseline, candidates, merge_candidates
from .pipeline import build_from_github, parse_resume_text

__all__ = [
    "Attribution",
    "Bullet",
    "BulletImpact",
    "Diagnostic",
    "Entry",
    "Header",
    "KeyValues",
    "ResumeDoc",
    "Section",
    "Table",
    "attribute",
    "baseline",
    "build_from_github",
    "candidates",
    "diagnose",
    "from_summaries",
    "merge_candidates",
    "parse_bits",
    "parse_resume",
    "parse_resume_text",
    "read_pdf",
    "rescore",
]
