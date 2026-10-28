"""Folding generated bullets into the resume the user already has.

The rule that governs everything here: **the user's resume is the baseline and
is never edited.** Generated lines arrive switched off, marked as coming from
repository evidence, and are adopted only when the user turns one on. That is
what makes "+23.0 against your uploaded resume" a measurement rather than a
claim -- the baseline is a real document, and the delta is a real rescore of it.

The alternative, rewriting the user's bullets in place, would make the before
and after incomparable, and would quietly discard the things the repositories
cannot prove: an internship, a teaching assistantship, a grade.
"""
from __future__ import annotations

import re

from .document import Bullet, Entry, ResumeDoc, Section

PROJECT_SECTIONS = ("PROJECTS", "PROJECT", "PERSONAL PROJECTS", "SELECTED PROJECTS")


def normalise(name: str) -> str:
    """Compare `resume-builder-agent` with `Resume Builder Agent` as equal."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _entry_for(doc: ResumeDoc, repo: str) -> Entry | None:
    """The user's own entry describing this repository, if they wrote one."""
    target = normalise(repo)
    if not target:
        return None
    for section in doc.sections:
        for entry in section.entries:
            if normalise(entry.title) == target:
                return entry
    # A looser pass: a repo slug is often a contraction of the written title.
    for section in doc.sections:
        for entry in section.entries:
            title = normalise(entry.title)
            if title and (title in target or target in title):
                return entry
    return None


def _projects_section(doc: ResumeDoc) -> Section:
    for section in doc.sections:
        if section.title.upper() in PROJECT_SECTIONS:
            return section
    section = Section(title="PROJECTS")
    doc.sections.append(section)
    return section


def _known(entry: Entry) -> set[str]:
    return {normalise(b.text) for b in entry.bullets}


def merge_candidates(base: ResumeDoc, generated: ResumeDoc, *,
                     enabled: bool = False) -> ResumeDoc:
    """Add every generated bullet to `base` as an opt-in candidate.

    Returns the same object, mutated. Existing bullets keep their text, their
    order and their ids, so a score computed before the merge still describes
    the same document afterwards as long as nothing is switched on.
    """
    for section in generated.sections:
        for source in _candidate_entries(section):
            repo = source.title or section.title
            target = _entry_for(base, repo)
            if target is None:
                target = Entry(title=_titleise(repo), subtitle=source.subtitle,
                               dates=source.dates, tech=list(source.tech))
                _projects_section(base).entries.append(target)
            seen = _known(target)
            for bullet in source.bullets:
                key = normalise(bullet.text)
                if key in seen:
                    continue      # the user already said this in their own words
                seen.add(key)
                target.bullets.append(Bullet(
                    text=bullet.text,
                    source_repo=bullet.source_repo or repo,
                    evidence=bullet.evidence,
                    verified=bullet.verified,
                    measure_hint=bullet.measure_hint,
                    enabled=enabled,
                    origin="repo" if bullet.verified else "assumed",
                ))
    return base


def _candidate_entries(section: Section) -> list[Entry]:
    """Generated documents use one section per repo, with bullets directly on it."""
    if section.entries:
        return section.entries
    if section.bullets:
        return [Entry(title=section.title, bullets=list(section.bullets))]
    return []


def _titleise(repo: str) -> str:
    """`resume-builder-agent` -> `Resume Builder Agent`, leaving prose alone."""
    if " " in repo:
        return repo
    words = re.split(r"[-_]+", repo)
    return " ".join(w[:1].upper() + w[1:] for w in words if w)


def candidates(doc: ResumeDoc) -> list[Bullet]:
    """Every bullet the user has not adopted from their own resume."""
    return [b for b in doc.all_bullets() if b.origin != "yours"]


def adopted(doc: ResumeDoc) -> list[Bullet]:
    return [b for b in candidates(doc) if b.enabled]


def baseline(doc: ResumeDoc) -> ResumeDoc:
    """A copy of the document as the user supplied it, with no candidates.

    This is what "vs your uploaded resume" is measured against, so every line
    the user arrived with is restored to *enabled* regardless of how they have
    since toggled it. Keeping their current switches here was wrong: turning
    off one of their own lines moved the baseline with it, so the comparison
    reported no change on a document that had plainly changed.

    Rebuilt rather than remembered, so that edits to the *wording* of an
    original line are still reflected.
    """
    clone = ResumeDoc.from_dict(doc.to_dict())
    for section in clone.sections:
        section.bullets = [b for b in section.bullets if b.origin == "yours"]
        for entry in section.entries:
            entry.bullets = [b for b in entry.bullets if b.origin == "yours"]
        section.entries = [e for e in section.entries if e.bullets]
    for bullet in clone.all_bullets():
        bullet.enabled = True
    return clone
