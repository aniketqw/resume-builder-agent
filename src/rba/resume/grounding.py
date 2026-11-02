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
UNVERIFIABLE = "unverifiable"  # nothing solid enough to check against


# What a repository's evidence is actually good enough to decide.
#
# This gate exists because the first version got it badly wrong on real data.
# `vision_dev_project` has 4,884 files, no README and no parsed dependency
# manifest, so its evidence bundle contains no mention of PyTorch -- and the
# checker duly reported that a true claim about PyTorch was unsupported. That
# is absence of evidence treated as evidence of absence, and acting on it means
# deleting things that are true.
#
# So the two claim types are gated on the evidence that could possibly carry
# them. A figure like "34 unit tests" is stated in prose, so judging it needs a
# README or real commit messages. A technology is named in a dependency
# manifest or in prose, so judging it needs one of those.
MIN_README_CHARS = 300
MIN_COMMIT_CHARS = 400
MIN_RAW_CHARS = 60


@dataclass(frozen=True)
class Strength:
    """Which checks this evidence can support."""

    numbers: bool
    technologies: bool
    why: str = ""

    @property
    def any(self) -> bool:
        return self.numbers or self.technologies


def evidence_strength(evidence: str) -> Strength:
    """Decide what this bundle is entitled to rule on."""
    import json as _json

    if not evidence.strip():
        return Strength(False, False, "no repository matched")
    try:
        bundle = _json.loads(evidence)
    except ValueError:
        # A raw string rather than the structured bundle. There is no way to
        # tell a README from a commit log here, so it is taken at face value as
        # curated evidence -- a caller who hand-assembles a string has already
        # decided what belongs in it. The size floor only rejects the trivially
        # empty; the strict README rule below applies to real bundles, which is
        # what the fetcher always produces.
        enough = len(evidence.strip()) >= MIN_RAW_CHARS
        return Strength(enough, enough, "" if enough
                        else "the evidence is too thin to judge")

    readme = (bundle.get("readme") or {}).get("text", "") or ""
    commits = " ".join(c.get("message", "")
                       for c in (bundle.get("commits") or []))
    deps = bundle.get("dependencies") or []

    # Each claim type is gated on the source that would actually carry it.
    #
    # A measured result ("blur caused 40% of failures") is written down in
    # documentation, not in commit subjects, so judging a figure needs a README.
    # A technology is named in a dependency manifest or a README; commit
    # messages mention neither reliably. Letting commits alone license both
    # checks is what left `vision_dev_project` -- 4,884 files, no README, no
    # manifest -- confidently reporting that its true PyTorch claim was
    # unsupported.
    documented = len(readme) >= MIN_README_CHARS
    manifest = bool(deps)
    why = ""
    if not documented and not manifest:
        why = (f"this repository has no README and no dependency manifest"
               f"{f' (only {len(commits)} characters of commit messages)' if commits else ''}"
               ", so nothing in it can confirm or deny a claim")
    elif not documented:
        why = ("this repository has no README, so its figures cannot be "
               "checked — only the technologies its manifest names")
    return Strength(numbers=documented, technologies=documented or manifest,
                    why=why)


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
    note: str = ""

    @property
    def changed(self) -> bool:
        return self.text != self.original

    @property
    def ok(self) -> bool:
        return self.status in (VERIFIED, UNVERIFIABLE)

    def reason(self) -> str:
        bits = []
        if self.unsupported_numbers:
            bits.append("figure(s) stated nowhere in the repository: "
                        + ", ".join(self.unsupported_numbers))
        if self.unsupported_tech:
            bits.append("technology not named anywhere in the repository: "
                        + ", ".join(self.unsupported_tech))
        if self.status == UNVERIFIABLE:
            return self.note or "no repository matched, so nothing was checked"
        if not bits:
            return "every claim appears in the repository"
        joined = "; ".join(bits)
        if self.status == TRIMMED:
            return (f"{joined} — that span was removed, the rest is verbatim. "
                    "Restore it if you can point to where it is recorded.")
        return (f"{joined}. It may still be true — the repository simply does "
                "not say so. Left exactly as written; no part of it could be "
                "cut cleanly.")

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
    """Figures and technologies in `text` that `evidence` contradicts or omits.

    Only claims the evidence is *competent to judge* are returned; see
    :func:`evidence_strength`. A repository with nothing in it proves nothing
    about the résumé, and saying otherwise would be the same overreach this
    project exists to prevent -- just pointed at the author instead of the
    model.
    """
    strength = evidence_strength(evidence)
    bad_numbers: list[str] = []
    drop: list[str] = []
    if strength.numbers:
        bad_numbers = [n for n in numbers_in(text)
                       if not _number_supported(n, evidence)]
    if strength.technologies:
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
    strength = evidence_strength(evidence)
    if not strength.any:
        return PointCheck(bullet.id, bullet.text, bullet.text, UNVERIFIABLE,
                          repo=repo, note=strength.why)

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
