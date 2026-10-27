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

        readme_path = (signals.readme_path if signals else "") or _find_readme(facts.tree)
        if readme_path:
            text = await backend.get_file(repo.name, readme_path, repo.default_branch)
            if text:
                facts.readme = text[:readme_max]
                facts.readme_path = readme_path

        present = [f for f in KEY_FILES if f in facts.tree]
        if present:
            contents = await asyncio.gather(
                *(backend.get_file(repo.name, f, repo.default_branch) for f in present),
                return_exceptions=True)
            for name, content in zip(present, contents, strict=False):
                if isinstance(content, str) and content:
                    facts.key_files[name] = content[:3000]
        facts.dependencies = extract_dependencies(facts.key_files)
        facts.languages = repo.languages or await _safe_languages(backend, repo.name)
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
