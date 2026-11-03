"""Turning a run's per-repository errors into something a user can act on.

Lifted out of the Streamlit app, which decided *and* rendered in one function
(`app._report_partial_failures`). v2 serves a browser over HTTP, so the two
have to separate: this module decides, the API serialises, the page renders.

The failure that prompted the original: an expired Groq key made every summary
return 401, and the app showed an empty document with no explanation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Credential-shaped text must never reach the screen -- and in v2 it travels
# over the wire to a browser first, so redaction happens here, server-side,
# before the string is ever put in a response body.
TOKENISH = re.compile(r"\b(?:gsk|ghp|gho|hf|lsv2_pt)[_-][A-Za-z0-9_-]+")

#: Longest error line we echo back. Tracebacks can be enormous and the useful
#: part is always at the front.
MAX_LINE = 300
MAX_LINES = 20

AUTH_MARKERS = ("invalid_api_key", "invalid api key", "401")


def redact(text: str) -> str:
    """Replace anything credential-shaped. Applied to every outgoing line."""
    return TOKENISH.sub("[redacted]", text)


@dataclass(frozen=True)
class FailureReport:
    """What to tell the user about a partially-failed run."""

    level: str              # "error" | "warning"
    message: str
    detail: list[str] = field(default_factory=list)
    count: int = 0

    def to_dict(self) -> dict:
        return {"level": self.level, "message": self.message,
                "detail": list(self.detail), "count": self.count}


def diagnose_failures(stats: dict) -> FailureReport | None:
    """`None` when the run was clean, otherwise what went wrong and how badly.

    Three cases, in descending severity, matching the original's branches:
    the credential was rejected (nothing could work), nothing was summarised
    for some other reason, or some repositories failed and the rest are fine.
    """
    errors = list(stats.get("errors") or [])
    if not errors:
        return None

    joined = " ".join(errors).lower()
    if any(marker in joined for marker in AUTH_MARKERS):
        level = "error"
        message = ("Groq rejected the API key, so no repository could be "
                   "summarised. Paste a current key — keys are revoked when "
                   "they leak into a public repository.")
    elif not stats.get("summarised"):
        level = "error"
        message = "No repository could be summarised. The detail is below."
    else:
        level = "warning"
        message = (f"{len(errors)} repository/repositories could not be read; "
                   f"the rest were summarised.")

    detail = [redact(line)[:MAX_LINE] for line in errors[:MAX_LINES]]
    return FailureReport(level=level, message=message, detail=detail,
                         count=len(errors))
