"""Repo selection: cheap decisions first, expensive ones never.

Summarizing every repo is not affordable. At ~2,074 tokens per summary and
8,000 tokens per minute, 98 repos is ~26 minutes of pure rate-limit wall time.
So selection runs in two stages, and no Groq token is spent in either:

1. **Hard rules** (free) throw out forks, archived repos, empty repos and
   anything below a commit floor. These are facts, not judgements, so a model
   should not be asked about them.
2. **Jev** scores what remains for "would this be worth claiming on a resume",
   and picks the Groq tier that should summarize it. Jev's output tokens are
   free and it draws on a separate quota, so this costs fractions of a cent and
   leaves the Groq budget intact.

If Jev is unavailable the heuristic fallback keeps the graph running.
"""
from __future__ import annotations

import asyncio
import logging

from ..config import Config
from ..jev_client import JevClient, JevError, choice, noul
from ..signals import RepoSignals, gather_signals
from ..state import AgentState, RepoRef

log = logging.getLogger(__name__)

CLAIMABLE_Q = noul(
    "Based on the evidence provided, is this repository substantial, original "
    "engineering work that the author could legitimately present as a portfolio "
    "project on a resume?",
    true_def=("The evidence shows real engineering: source code the author wrote, a "
              "README that explains an actual system, and signals of effort such as a "
              "test suite, CI configuration, dependency manifests, or a structured "
              "multi-directory codebase. Judge the evidence, not the repository name."),
    false_def=("The evidence shows a course archive, tutorial follow-along, lecture "
               "notes, a collection of notebooks or documents with little original "
               "code, an empty or placeholder repository, or work that is clearly "
               "someone else's with the author as a minor contributor."),
)

TIER_Q = choice(
    "Given this evidence, how much model capability is needed to summarize this "
    "repository accurately?",
    {
        "small": "Small and self-explanatory: few files, a short README, one obvious purpose.",
        "mid": "A normal project: a real README, moderate codebase, several components.",
        "big": "Large or architecturally complex: many modules, subtle design, "
               "or a README that needs careful interpretation.",
    },
)

# Repos whose names alone signal course/scratch work. Used only as a tiebreaker
# in the fallback path — Jev gets to judge these on their actual content.
_NOISE_HINTS = ("prep", "cohort", "course", "tutorial", "practice", "demo",
                "sample", "test", "sandbox", "temp", "scratch", "dotfiles")


def _state_for(repo: RepoRef, sig: RepoSignals | None) -> dict:
    """The evidence blob Jev reasons over.

    Metadata alone is not enough: most repositories here have no description, so
    a model asked to judge them returns uniform uncertainty — the correct answer
    to a question containing no evidence. :mod:`rba.signals` supplies the README
    excerpt, tree shape and commit history that make the question answerable.
    """
    if sig is None:
        return {
            "name": repo.name,
            "description": repo.description or None,
            "primary_language": repo.language or None,
            "repo_size_kb": repo.size_kb,
            "stars": repo.stars,
            "evidence_available": False,
        }
    return sig.as_state(repo)


def apply_hard_rules(repos: list[RepoRef], config: Config) -> tuple[list[RepoRef], list[RepoRef]]:
    """Split into (candidates, rejected) using only metadata we can trust.

    Emptiness is deliberately *not* judged here. ``size_kb`` is absent from the
    MCP ``search_repositories`` response, so treating 0 as "empty" would reject
    every repository. Emptiness is a fact about files and commits, and it is
    decided in :func:`drop_empty` once signals have been gathered.
    """
    exclude_forks = config.get("selection.exclude_forks", True)
    candidates: list[RepoRef] = []
    rejected: list[RepoRef] = []
    for r in repos:
        reason = ""
        if exclude_forks and r.is_fork:
            reason = "fork — not the author's original work"
        elif r.is_archived:
            reason = "archived"
        if reason:
            r.selection_reason = reason
            rejected.append(r)
        else:
            candidates.append(r)
    return candidates, rejected


def drop_empty(repos: list[RepoRef], signals: dict[str, RepoSignals],
               min_commits: int = 2) -> tuple[list[RepoRef], list[RepoRef]]:
    """Reject repos with no retrievable content. Needs signals, not metadata."""
    keep: list[RepoRef] = []
    drop: list[RepoRef] = []
    for r in repos:
        s = signals.get(r.name)
        if s is None:
            keep.append(r)
            continue
        if s.file_count == 0 and s.commit_count == 0:
            r.selection_reason = "empty repository (no files, no commits)"
            drop.append(r)
        elif s.commit_count and s.commit_count < min_commits:
            r.selection_reason = f"only {s.commit_count} commit(s) — not claimable"
            drop.append(r)
        else:
            keep.append(r)
    return keep, drop


def heuristic_score(repo: RepoRef) -> float:
    """Fallback used only when Jev is unreachable. Intentionally crude."""
    score = 0.35
    if repo.description:
        score += 0.15
    if repo.language:
        score += 0.10
    if repo.topics:
        score += 0.05
    if repo.stars:
        score += 0.10
    if repo.size_kb > 200:
        score += 0.10
    if repo.size_kb > 2000:
        score += 0.05
    name = repo.name.lower()
    if any(h in name for h in _NOISE_HINTS):
        score -= 0.20
    return max(0.0, min(1.0, score))


async def score_with_jev(repos: list[RepoRef], jev: JevClient,
                         signals: dict[str, RepoSignals] | None = None,
                         concurrency: int = 8) -> None:
    """Annotate each repo in place with ``claimable`` and ``tier``."""
    sem = asyncio.Semaphore(concurrency)
    signals = signals or {}

    async def one(repo: RepoRef) -> None:
        async with sem:
            try:
                ans = await jev.ask(_state_for(repo, signals.get(repo.name)),
                                    {"claimable": CLAIMABLE_Q, "tier": TIER_Q})
                repo.claimable = ans.probability("claimable")
                repo.tier, conf = ans.choice("tier")
                repo.selection_reason = f"jev claimable={repo.claimable:.2f} tier={repo.tier}@{conf:.2f}"
            except JevError as exc:
                repo.claimable = heuristic_score(repo)
                repo.tier = "small"
                repo.selection_reason = f"jev failed ({str(exc)[:60]}); heuristic"

    await asyncio.gather(*(one(r) for r in repos))


async def select_repos(state: AgentState, *, config: Config | None = None,
                       jev: JevClient | None = None, backend=None) -> AgentState:
    cfg = config or Config.load()
    repos = state.get("repos", [])
    candidates, rejected = apply_hard_rules(repos, cfg)
    if rejected:
        log.info("hard rules excluded %d/%d repos (%s)", len(rejected), len(repos),
                 ", ".join(sorted({r.selection_reason.split(" —")[0] for r in rejected})))

    # Gather real evidence first. Free of LLM tokens, and reused by fetch_facts.
    signals: dict[str, RepoSignals] = state.get("signals") or {}
    if backend is not None and not signals:
        owner = state.get("username", "")
        signals = await gather_signals(backend, candidates, owner)
        n_readme = sum(1 for s in signals.values() if s.readme_chars)
        log.info("gathered signals for %d repos (%d have a README)",
                 len(signals), n_readme)
        candidates, empties = drop_empty(
            candidates, signals, cfg.get("selection.min_commits", 2))
        if empties:
            log.info("dropped %d empty or single-commit repos", len(empties))
            rejected += empties

    owned_jev = jev is None
    jev = jev or JevClient(cfg)
    try:
        if jev.available:
            await score_with_jev(candidates, jev, signals)
            log.info("Jev scored %d repos for $%.5f (%d input tokens)",
                     len(candidates), jev.usage.cost_usd, jev.usage.input_tokens)
        else:
            log.warning("Jev unavailable; using the heuristic fallback.")
            for r in candidates:
                r.claimable = heuristic_score(r)
                r.tier = "small"
                r.selection_reason = "heuristic (jev unavailable)"
    finally:
        if owned_jev:
            await jev.aclose()

    threshold = cfg.get("selection.min_claimable", 0.45)
    max_repos = cfg.get("selection.max_repos", 25)
    ranked = sorted(candidates, key=lambda r: (-r.claimable, -r.size_kb, r.name))
    selected = [r for r in ranked if r.claimable >= threshold][:max_repos]

    return {
        "selected": selected,
        "excluded": rejected,
        "signals": signals,
        "usage": {**state.get("usage", {}),
                  "jev": {"calls": jev.usage.calls,
                          "input_tokens": jev.usage.input_tokens,
                          "cost_usd": round(jev.usage.cost_usd, 6)}},
        "errors": [],
    } | ({} if selected else {"errors": ["select_repos: nothing cleared the claimable threshold"]})
