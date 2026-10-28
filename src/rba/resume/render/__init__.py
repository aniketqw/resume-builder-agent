"""Rendering one document to the formats a job application actually needs.

Every renderer is a pure function of `(doc, options)`. Nothing here fetches,
scores or calls a model, which is what lets the UI re-render on every keystroke
without spending a token.

The set exists because the two audiences want opposite things: an applicant
tracking system wants unstyled text in a predictable order, and a human wants
the one-page layout the candidate actually uses.
"""
from __future__ import annotations

from .html import to_html
from .latex import to_latex
from .options import RenderOptions
from .pdf import to_pdf
from .plain import to_plain

__all__ = ["RenderOptions", "to_html", "to_latex", "to_pdf", "to_plain"]
