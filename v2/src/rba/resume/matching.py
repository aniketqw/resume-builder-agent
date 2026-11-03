"""Deciding which repository a résumé project is actually about.

This is the step that has to be right before anything else can be. Grounding a
project against the wrong repository would verify real claims against unrelated
evidence -- stripping true statements and keeping false ones. It is the most
damaging mistake this tool could make, which is why the result is shown to the
user before a single claim is checked against it.

Name matching does not work. Measured against a real account of 58 public
repositories and a real résumé of six projects:

===================================  ===========================  ==========
project                              repository                   by name?
===================================  ===========================  ==========
Live Debugging Partner               live-debugging-partner       exact
Sandboxed Code-Execution Agent       sandbox-agent                **no**
CodeAct Fine-tuning with Exec…       CodeActImplementation        **no**
VisionDev                            vision_dev_project           **no**
Multi-University Bounty Platform     (none)                       --
Resume Builder Agent                 (private)                    --
===================================  ===========================  ==========

One in six. Worse, relaxing to substring matching actively invents matches: a
generic repository named ``code`` substring-matches both *Sandboxed
**Code**-Execution Agent* and ***Code**Act Fine-tuning*, two different
projects. A loose matcher is more dangerous than a strict one here.

So matching is semantic, using the same ONNX embedding model the scorer already
uses -- no LLM, no tokens. Name evidence is kept as a *bonus* rather than the
decision, because when a name does match exactly it is very strong evidence,
and when it does not it says nothing either way.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..ats.dense import embed
from .document import Entry, ResumeDoc
from .merge import normalise

# Measured on a real account (58 repos) against a hand-checked ground truth,
# projects only -- see `scripts/calibrate_matching.py`:
#
#     true matches        confidence 0.697-0.957   margin 0.021-0.304
#     projects with none  confidence 0.686-0.696   margin 0.004-0.026
#
# Confidence alone separates the two by **0.001**, and the margin over the
# runner-up overlaps outright. There is no single automatic threshold on this
# data, and pretending otherwise would attach projects to unrelated repos.
#
# Requiring *both* to be strong auto-accepts 3 of the 4 true matches with zero
# false accepts; everything else is handed to the user. Erring toward asking is
# the whole point: a wrong repository verifies real claims against unrelated
# evidence, stripping true statements and keeping false ones.
MIN_CONFIDENCE = 0.55
STRONG_CONFIDENCE = 0.75
STRONG_MARGIN = 0.10

# An exact normalised name match is strong evidence on its own. Anything looser
# is not: substring matching is what produced the `code` false positives.
EXACT_NAME_BONUS = 0.25

# Repository names this short carry no signal -- `code`, `ads`, `agentic`.
# They are the ones that substring-match everything.
GENERIC_NAME_CHARS = 6


@dataclass(frozen=True)
class Match:
    """One project, and the repository it most likely describes."""

    project_key: str
    project_title: str
    repo: str
    confidence: float
    runners_up: tuple[tuple[str, float], ...] = ()

    @property
    def margin(self) -> float:
        """How far clear of the next candidate. Near zero means a coin toss."""
        return round(self.confidence - (self.runners_up[0][1]
                                        if self.runners_up else 0.0), 4)

    @property
    def strong(self) -> bool:
        """Safe to accept without asking. Deliberately conservative."""
        return (bool(self.repo) and self.confidence >= STRONG_CONFIDENCE
                and self.margin >= STRONG_MARGIN)

    @property
    def confident(self) -> bool:
        return bool(self.repo) and self.confidence >= MIN_CONFIDENCE

    def why(self) -> str:
        if self.strong:
            return f"clear match ({self.confidence:.2f}, {self.margin:+.2f} clear)"
        if not self.repo:
            return "nothing scored high enough to suggest"
        return (f"needs your confirmation — {self.confidence:.2f} and only "
                f"{self.margin:+.2f} clear of {self.runners_up[0][0]}"
                if self.runners_up else "needs your confirmation")


def project_text(entry: Entry) -> str:
    """What a project *is*, for comparison: its title, stack and claims."""
    parts = [entry.title, entry.subtitle, " ".join(entry.tech)]
    parts += [b.text for b in entry.bullets]
    return " ".join(p for p in parts if p)


def repo_text(repo) -> str:
    """What a repository *is*: its name, blurb, languages and README opening.

    The name is split on separators so `vision_dev_project` reads as three
    words, which is what lets it match "VisionDev" at all.
    """
    name = getattr(repo, "name", "") or ""
    words = " ".join(w for w in re.split(r"[-_./]+", name) if w)
    parts = [name, words, getattr(repo, "description", "") or "",
             " ".join((getattr(repo, "languages", None) or {}).keys()),
             (getattr(repo, "readme", "") or "")[:1200]]
    return " ".join(p for p in parts if p)


def _name_bonus(title: str, repo_name: str) -> float:
    """Credit only an unambiguous name match.

    Substring matching is deliberately absent. `code` is a substring of two
    unrelated project titles, and trusting that would attach a project to the
    wrong repository -- the exact failure this module exists to prevent.
    """
    project, repo = normalise(title), normalise(repo_name)
    if not project or not repo:
        return 0.0
    if project == repo:
        return EXACT_NAME_BONUS
    if len(repo) <= GENERIC_NAME_CHARS:
        return 0.0
    # Token-level containment: "codeactimplementation" starts with "codeact".
    if repo.startswith(project) or project.startswith(repo):
        return EXACT_NAME_BONUS * 0.6
    return 0.0


PROJECT_SECTIONS = ("PROJECTS", "PROJECT", "PERSONAL PROJECTS",
                    "SELECTED PROJECTS", "ACADEMIC PROJECTS")


def match_projects(doc: ResumeDoc, repos: list, *,
                   min_confidence: float = MIN_CONFIDENCE) -> list[Match]:
    """Rank every repository against every project. One batched embedding call.

    Only project sections are considered. An internship or a teaching role has
    no repository, and including them was actively harmful in calibration: the
    Marvell internship matched `live-debugging-partner` at 0.658 and a teaching
    assistantship matched `vision_dev_project` at 0.659, both higher than
    genuine projects that happen to have no repo.

    Returns one `Match` per project, in document order, including the ones that
    found nothing -- a project with no repository is a fact the user needs to
    see, not a row to omit.
    """
    entries: list[tuple[str, Entry]] = []
    for section in doc.sections:
        if section.title.upper() not in PROJECT_SECTIONS:
            continue
        for index, entry in enumerate(section.entries):
            entries.append((f"{section.title}#{index}", entry))
    if not entries or not repos:
        return [Match(key, e.title, "", 0.0) for key, e in entries]

    project_texts = [project_text(e) for _key, e in entries]
    repo_texts = [repo_text(r) for r in repos]
    vectors = embed([*project_texts, *repo_texts])
    if vectors is None:
        return [Match(key, e.title, "", 0.0) for key, e in entries]

    split = len(project_texts)
    project_vectors, repo_vectors = vectors[:split], vectors[split:]
    similarity = project_vectors @ repo_vectors.T

    out: list[Match] = []
    for row, (key, entry) in enumerate(entries):
        scored = [
            (getattr(repos[col], "name", ""),
             float(similarity[row, col])
             + _name_bonus(entry.title, getattr(repos[col], "name", "")))
            for col in range(len(repos))
        ]
        scored.sort(key=lambda pair: -pair[1])
        best_name, best_score = scored[0]
        confident = best_score >= min_confidence
        out.append(Match(
            project_key=key,
            project_title=entry.title,
            repo=best_name if confident else "",
            confidence=round(best_score, 4),
            runners_up=tuple((n, round(s, 4)) for n, s in scored[1:4]),
        ))
    return out


def entry_for_key(doc: ResumeDoc, key: str) -> Entry | None:
    for section in doc.sections:
        for index, entry in enumerate(section.entries):
            if f"{section.title}#{index}" == key:
                return entry
    return None
