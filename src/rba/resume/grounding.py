"""Checking a résumé's own claims against the repositories behind them.

The rest of this project stops the *generator* inventing things. This stops the
résumé itself carrying claims its own repositories do not support -- a number
that drifted upward between drafts, a technology that was considered and never
used, a result nobody measured.

The governing constraint here is narrower than anywhere else in the codebase:
**this code may delete, never write.** Every sentence it produces is either the
author's sentence unchanged or the author's sentence with a span removed. There
is no model in this path at all, which is what makes that property checkable
rather than hoped for -- `is_subsequence_edit` asserts it mechanically.

Why deletion and not rephrasing: a rewrite that "minimally" restates a claim is
still the tool putting words in the author's mouth, and those words then go out
under their name to an employer. Cutting is the only edit that cannot say
something new. Where no clean cut exists the line is flagged and left exactly
as written, because mangling a sentence is worse than reporting a problem.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..nodes.verify import (
    _number_supported,
    filter_tech,
    normalise,
    numbers_in,
)
from .document import Bullet, Entry, ResumeDoc

# A span that can be cut without wrecking the sentence: a parenthetical, or a
# trailing clause introduced by a semicolon or a dash. Anything else -- a
# number welded into the middle of a clause -- has no clean cut.
_PARENTHETICAL = re.compile(r"\s*\([^()]*\)")
_TRAILING_CLAUSE = re.compile(r"\s*[;—-]\s+[^;]*$")
_CONTENT = re.compile(r"[a-z0-9][a-z0-9.+#/_-]*")

#: Status of one point after checking it against its repository.
VERIFIED = "verified"          # every claim supported by the evidence
TRIMMED = "trimmed"            # an unsupported span was cut out
FLAGGED = "flagged"            # unsupported, but no safe cut exists
UNVERIFIABLE = "unverifiable"  # no repository to check against


@dataclass
class PointCheck:
    """What became of one résumé point, and why."""

    bullet_id: str
    original: str
    text: str                       # original, or original minus a span
    status: str
    unsupported_numbers: list[str] = field(default_factory=list)
    unsupported_tech: list[str] = field(default_factory=list)
    repo: str = ""

    @property
    def changed(self) -> bool:
        return self.text != self.original

    @property
    def ok(self) -> bool:
        return self.status in (VERIFIED, UNVERIFIABLE)

    def reason(self) -> str:
        bits = []
        if self.unsupported_numbers:
            bits.append("figure(s) not in the repository: "
                        + ", ".join(self.unsupported_numbers))
        if self.unsupported_tech:
            bits.append("technology not evidenced: "
                        + ", ".join(self.unsupported_tech))
        if self.status == UNVERIFIABLE:
            return "no repository matched, so nothing could be checked"
        if not bits:
            return "every claim appears in the repository"
        joined = "; ".join(bits)
        if self.status == TRIMMED:
            return f"{joined} — that span was removed, the rest is verbatim"
        return f"{joined} — left as written; no part of it could be cut cleanly"

    def to_dict(self) -> dict:
        return {"bullet_id": self.bullet_id, "original": self.original,
                "text": self.text, "status": self.status, "repo": self.repo,
                "unsupported_numbers": list(self.unsupported_numbers),
                "unsupported_tech": list(self.unsupported_tech)}


@dataclass
class GroundingReport:
    """Every point, and the document that results from honouring the checks."""

    checks: list[PointCheck] = field(default_factory=list)
    document: ResumeDoc | None = None

    def by_status(self, status: str) -> list[PointCheck]:
        return [c for c in self.checks if c.status == status]

    def summary(self) -> str:
        counts = {s: len(self.by_status(s))
                  for s in (VERIFIED, TRIMMED, FLAGGED, UNVERIFIABLE)}
        return (f"{counts[VERIFIED]} verified · {counts[TRIMMED]} trimmed · "
                f"{counts[FLAGGED]} flagged · {counts[UNVERIFIABLE]} unverifiable")


def tech_terms(text: str) -> list[str]:
    """Technology names a point mentions, using the scorer's own vocabulary.

    Reuses the curated alias table rather than guessing which words are
    technologies -- the same table the skills channel scores against, so the
    two cannot disagree about what counts as a technology.
    """
    from ..ats.normalise import ALIASES

    lowered = f" {normalise(text)} "
    found: list[str] = []
    for canon, aliases in ALIASES.items():
        for surface in (canon, *aliases):
            if f" {surface} " in lowered:
                found.append(surface)
                break
    return found


def unsupported_claims(text: str, evidence: str) -> tuple[list[str], list[str]]:
    """Figures and technologies in `text` that `evidence` does not support."""
    if not evidence.strip():
        return [], []
    bad_numbers = [n for n in numbers_in(text) if not _number_supported(n, evidence)]
    _keep, drop = filter_tech(tech_terms(text), evidence)
    return bad_numbers, drop


def _cut_candidates(text: str) -> list[tuple[int, int]]:
    """Spans that can be removed without leaving a broken sentence."""
    spans = [m.span() for m in _PARENTHETICAL.finditer(text)]
    trailing = _TRAILING_CLAUSE.search(text)
    if trailing:
        spans.append(trailing.span())
    return spans


def _still_a_sentence(text: str) -> bool:
    """Enough words left, and no dangling connective at the end."""
    stripped = text.strip().rstrip(".,;:")
    words = stripped.split()
    if len(words) < 5:
        return False
    return words[-1].lower() not in {
        "and", "or", "with", "of", "the", "a", "an", "to", "by", "for", "in",
        "on", "that", "which", "using", "via", "from",
    }


def strip_unsupported(text: str, offenders: list[str]) -> str | None:
    """Remove the smallest clean span containing an unsupported claim.

    Returns the shortened sentence, or None when no cut leaves something
    readable -- in which case the caller flags the line and changes nothing.
    This never rewrites: the result is always a subsequence of the input.
    """
    if not offenders:
        return text
    current = text
    for _ in range(len(offenders) + 1):
        remaining = [o for o in offenders if o.lower() in current.lower()]
        if not remaining:
            break
        cut_made = False
        # Shortest viable cut first, so a sentence loses as little as possible.
        for start, end in sorted(_cut_candidates(current),
                                 key=lambda s: s[1] - s[0]):
            span = current[start:end]
            if not any(o.lower() in span.lower() for o in remaining):
                continue
            candidate = (current[:start] + current[end:]).strip()
            candidate = re.sub(r"\s+([.,;])", r"\1", candidate)
            if _still_a_sentence(candidate):
                current = candidate if candidate.endswith(".") else candidate + "."
                cut_made = True
                break
        if not cut_made:
            return None
    return current


def _words(text: str) -> list[str]:
    """Content words, without the punctuation that ends a sentence.

    The token pattern deliberately allows `.` inside a word so that `node.js`
    and `scikit-learn` survive intact -- but that also swallowed the full stop
    at the end of a sentence, so `failures.` compared unequal to `failures` and
    a perfectly good trim was rejected as if it had invented a word.
    """
    return [w for w in (t.rstrip(".,;:") for t in _CONTENT.findall(text.lower())) if w]


def is_subsequence_edit(original: str, edited: str) -> bool:
    """Is `edited` the `original` with spans removed, and nothing added?

    The invariant that makes this module trustworthy. Checked as a real
    subsequence over content words so that punctuation tidying is allowed but
    a single new word is not.
    """
    source = _words(original)
    result = _words(edited)
    index = 0
    for word in result:
        while index < len(source) and source[index] != word:
            index += 1
        if index == len(source):
            return False
        index += 1
    return True


def check_point(bullet: Bullet, evidence: str, repo: str = "") -> PointCheck:
    """Verify one point and, if needed, cut what the evidence cannot support."""
    if not evidence.strip():
        return PointCheck(bullet.id, bullet.text, bullet.text, UNVERIFIABLE,
                          repo=repo)

    bad_numbers, bad_tech = unsupported_claims(bullet.text, evidence)
    if not bad_numbers and not bad_tech:
        return PointCheck(bullet.id, bullet.text, bullet.text, VERIFIED,
                          repo=repo)

    trimmed = strip_unsupported(bullet.text, [*bad_numbers, *bad_tech])
    if trimmed is None or trimmed == bullet.text:
        return PointCheck(bullet.id, bullet.text, bullet.text, FLAGGED,
                          bad_numbers, bad_tech, repo)
    # Cutting must not have introduced a word. If it somehow did, keep the
    # original: a flagged true sentence beats a quietly altered one.
    if not is_subsequence_edit(bullet.text, trimmed):
        return PointCheck(bullet.id, bullet.text, bullet.text, FLAGGED,
                          bad_numbers, bad_tech, repo)
    return PointCheck(bullet.id, bullet.text, trimmed, TRIMMED,
                      bad_numbers, bad_tech, repo)


def ground_document(doc: ResumeDoc, evidence_by_key: dict[str, str],
                    repo_by_key: dict[str, str] | None = None) -> GroundingReport:
    """Check every project point against its repository's evidence.

    `evidence_by_key` maps a project key (as produced by
    :func:`rba.resume.matching.match_projects`) to that repository's evidence
    bundle. A project with no entry is reported `unverifiable` and left
    untouched -- removing real work because no repository was found would be a
    worse error than leaving it unchecked.
    """
    repo_by_key = repo_by_key or {}
    grounded = ResumeDoc.from_dict(doc.to_dict())
    checks: list[PointCheck] = []

    for section in grounded.sections:
        for index, entry in enumerate(section.entries):
            key = f"{section.title}#{index}"
            evidence = evidence_by_key.get(key, "")
            repo = repo_by_key.get(key, "")
            for bullet in entry.bullets:
                check = check_point(bullet, evidence, repo)
                checks.append(check)
                if check.status == TRIMMED:
                    bullet.text = check.text
                    bullet.origin = "yours"
                    bullet.evidence = f"trimmed against {repo}" if repo else ""
                bullet.verified = check.status == VERIFIED
                if check.status == FLAGGED:
                    bullet.measure_hint = check.reason()
    return GroundingReport(checks=checks, document=grounded)


def entries_of(doc: ResumeDoc) -> list[tuple[str, Entry]]:
    """Project entries with the keys the matcher uses."""
    from .matching import PROJECT_SECTIONS

    out: list[tuple[str, Entry]] = []
    for section in doc.sections:
        if section.title.upper() not in PROJECT_SECTIONS:
            continue
        for index, entry in enumerate(section.entries):
            out.append((f"{section.title}#{index}", entry))
    return out
