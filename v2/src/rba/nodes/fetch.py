"""Deterministic fact gathering. No model touches anything in this module.

Everything a summary is later allowed to assert must trace back to a field
produced here. That is what makes ``verify_claims`` able to reject an invented
number: the evidence set is finite, known, and was collected without a model's
involvement.

This node uses MCP tools, so it must stay separate from ``summarize_repo`` —
Groq cannot combine tool use with structured outputs in one request.
"""
from __future__ import annotations

import asyncio
import logging
import re

from ..config import Config
from ..signals import RepoSignals, classify_tree
from ..state import AgentState, RepoFacts, RepoRef

log = logging.getLogger(__name__)

KEY_FILES = ("pyproject.toml", "requirements.txt", "package.json", "go.mod",
             "Cargo.toml", "pom.xml", "Dockerfile", "docker-compose.yml",
             "Makefile", "langgraph.json")

#: How many ranked source files to pull per repo for the code-structure
#: adapter (`rba.repobrain`). Python-only for now -- see repobrain's module
#: docstring for why. Bounded to keep this a handful of extra calls, not a
#: full-repo walk.
SOURCE_FILE_LIMIT = 12

#: A safety ceiling, not an expected-to-be-hit truncation point. This used
#: to be 4000 and applied unconditionally -- but `source_files` feeds
#: `repobrain.index()`'s `ast.parse()` (`nodes/prepare_evidence.py`) and
#: `verify_claims`' quote matching (`nodes/verify.py`), both of which need
#: syntactically/textually complete files. Cutting a file off at 4000 chars
#: landed mid-token on anything longer, producing a guaranteed
#: `SyntaxError` on every such file (reproduced live: a repo's own source
#: files, truncated mid-string-literal, filled `partial_source` with
#: spurious parse errors that had nothing to do with the code's real
#: quality) -- and silently made any genuine claim whose quoted evidence
#: fell past that cutoff unverifiable. 60,000 chars is generous enough that
#: ordinary hand-written source essentially never hits it; it exists only
#: to stop one pathological huge generated/vendored file from bloating
#: `AgentState` (which flows through every graph node, see `trace.md`'s
#: note on this) without bound.
SOURCE_FILE_MAX_CHARS = 60_000

DEP_PATTERNS = {
    "requirements.txt": re.compile(r"^\s*([A-Za-z0-9_.\-]+)", re.M),
    "pyproject.toml": re.compile(r'"([A-Za-z0-9_.\-]+)\s*[><=~!]'),
    "package.json": re.compile(r'"([@A-Za-z0-9_./\-]+)"\s*:\s*"[\^~>=<0-9]'),
    "go.mod": re.compile(r"^\s+([\w.\-/]+)\s+v", re.M),
}


def extract_dependencies(key_files: dict[str, str], limit: int = 40) -> list[str]:
    """Pull dependency names out of manifests. Strings only, never versions."""
    deps: list[str] = []
    for name, pattern in DEP_PATTERNS.items():
        content = key_files.get(name)
        if not content:
            continue
        for m in pattern.finditer(content):
            dep = m.group(1).strip()
            if dep and dep not in deps and not dep.startswith("#"):
                deps.append(dep)
    return deps[:limit]


async def fetch_repo_facts(backend, repo: RepoRef, owner: str, config: Config,
                           signals: RepoSignals | None = None) -> RepoFacts:
    readme_max = config.get("github.fetch.readme_max_chars", 6000)
    max_tree = config.get("github.fetch.max_tree_files", 120)
    max_commits = config.get("github.fetch.max_commits", 30)

    facts = RepoFacts(repo=repo.name)

    # Reuse what selection already paid for.
    if signals is not None and signals.file_count:
        tree_known = True
        facts.file_count = signals.file_count
        facts.has_tests = signals.has_tests
        facts.has_ci = signals.has_ci
        facts.commit_count = signals.commit_count
        facts.author_commits = signals.author_commits
    else:
        tree_known = False

    try:
        tree, commits = await asyncio.gather(
            backend.get_tree(repo.name, repo.default_branch),
            backend.list_commits(repo.name, max_commits),
            return_exceptions=True,
        )
        if isinstance(tree, list) and tree:
            sig = RepoSignals(repo=repo.name)
            classify_tree(tree, sig)
            facts.file_count = sig.file_count
            facts.has_tests = sig.has_tests
            facts.has_ci = sig.has_ci
            # Keep the most informative paths rather than an arbitrary prefix.
            facts.tree = _interesting_paths(tree, max_tree)
        elif not tree_known:
            log.debug("no tree for %s", repo.name)

        if isinstance(commits, list):
            facts.commits = commits
            facts.commit_count = len(commits)
            facts.author_commits = sum(
                1 for c in commits if (c.get("author") or "").lower() == owner.lower())
            if commits:
                facts.head_sha = commits[0].get("sha", "")

        # Pin every file fetch below to the exact commit the evidence will
        # claim it came from. Falling back to the branch name here would let
        # a push between "list the tree" and "read the file" silently
        # desync a permalink from the content it supposedly points at.
        ref = facts.head_sha or repo.default_branch

        readme_path = (signals.readme_path if signals else "") or _find_readme(facts.tree)
        if readme_path:
            # Its own try/except: a README fetch failing must not also lose
            # the manifests and source files gathered below -- those are
            # independent facts about the repo, not contingent on this one.
            try:
                text = await backend.get_file(repo.name, readme_path, ref)
            except Exception as exc:
                log.debug("readme fetch failed for %s: %s", repo.name, exc)
                text = None
            if text:
                facts.readme = text[:readme_max]
                facts.readme_path = readme_path

        present = [f for f in KEY_FILES if f in facts.tree]
        if present:
            contents = await asyncio.gather(
                *(backend.get_file(repo.name, f, ref) for f in present),
                return_exceptions=True)
            for name, content in zip(present, contents, strict=False):
                if isinstance(content, str) and content:
                    facts.key_files[name] = content[:3000]
        facts.dependencies = extract_dependencies(facts.key_files)
        facts.languages = repo.languages or await _safe_languages(backend, repo.name)

        source_paths = _ranked_source_paths(facts.tree, SOURCE_FILE_LIMIT)
        if source_paths:
            contents = await asyncio.gather(
                *(backend.get_file(repo.name, p, ref) for p in source_paths),
                return_exceptions=True)
            for path, content in zip(source_paths, contents, strict=False):
                if isinstance(content, str) and content:
                    facts.source_files[path] = content[:SOURCE_FILE_MAX_CHARS]
    except Exception as exc:  # a single bad repo must not abort the sweep
        log.warning("fetch_facts failed for %s: %s", repo.name, exc)
    return facts


async def _safe_languages(backend, repo: str) -> dict[str, int]:
    try:
        return await backend.get_languages(repo)
    except Exception:
        return {}


def _find_readme(tree: list[str]) -> str:
    for p in tree:
        if p.lower().startswith("readme"):
            return p
    return ""


_NOISE = re.compile(r"(^|/)(node_modules|\.git|venv|\.venv|dist|build|__pycache__)/")
_SIGNAL = re.compile(r"(^|/)(src|app|lib|tests?|docs?|\.github)/|"
                     r"\.(py|ts|tsx|js|jsx|go|java|rs|c|cpp|md|ya?ml|toml)$", re.I)


def _interesting_paths(tree: list[str], limit: int) -> list[str]:
    """Pick paths that say something about the project, not vendored bulk."""
    candidates = [p for p in tree if not _NOISE.search(p)]
    scored = sorted(candidates,
                    key=lambda p: (not _SIGNAL.search(p), p.count("/"), len(p)))
    return scored[:limit]


def _ranked_source_paths(tree: list[str], limit: int) -> list[str]:
    """The same ranking as `_interesting_paths`, restricted to files the
    code-structure adapter can actually parse. Python-only today -- see
    `rba.repobrain`'s module docstring. Test files are deprioritised, not
    excluded: a test is still real evidence of what the project does, just
    less often the clearest one."""
    candidates = [p for p in tree if p.endswith(".py") and not _NOISE.search(p)]
    scored = sorted(candidates,
                    key=lambda p: (bool(re.search(r"(^|/)tests?/", p)),
                                   p.count("/"), len(p)))
    return scored[:limit]


async def fetch_facts(state: AgentState, *, backend, config: Config | None = None,
                      concurrency: int = 6) -> AgentState:
    cfg = config or Config.load()
    owner = state.get("username", "")
    signals = state.get("signals") or {}
    selected = state.get("selected", [])
    sem = asyncio.Semaphore(concurrency)

    async def one(repo: RepoRef):
        async with sem:
            return repo.name, await fetch_repo_facts(
                backend, repo, owner, cfg, signals.get(repo.name))

    pairs = await asyncio.gather(*(one(r) for r in selected))
    facts = dict(pairs)
    log.info("fetched facts for %d repos (%d with a README, %d with tests)",
             len(facts),
             sum(1 for f in facts.values() if f.readme),
             sum(1 for f in facts.values() if f.has_tests))
    return {"facts": facts}
