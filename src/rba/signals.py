"""Cheap, deterministic evidence gathered before any model is consulted.

Repository metadata alone (name, size, language) is far too thin to judge
whether work is claimable — most repositories have no description at all, and a
500 MB course-notes dump looks exactly like a 500 MB real project. Asking a
model to guess from that produces uniform uncertainty, which is the correct
answer to a question with no evidence in it.

So before selection we pull a small, bounded set of real signals per repo:
the README head, the file tree shape, and recent commits. Three REST calls,
no tokens, and they are reused by the full fact fetch later.
"""
from __future__ import annotations

import asyncio
import logging
import re

from .state import RepoRef

log = logging.getLogger(__name__)

TEST_PAT = re.compile(r"(^|/)(tests?|spec|__tests__)/|(^|/)test_[^/]+\.py$|[._-]test\.[jt]sx?$|_test\.go$")
CI_PAT = re.compile(r"^\.github/workflows/|^\.gitlab-ci\.yml$|^\.circleci/|^Jenkinsfile$|^\.travis\.yml$")
DOC_PAT = re.compile(r"^docs?/|\.md$", re.I)
DEP_FILES = ("requirements.txt", "pyproject.toml", "package.json", "go.mod",
             "pom.xml", "build.gradle", "Cargo.toml", "Gemfile")
README_CANDIDATES = ("README.md", "README.rst", "README.txt", "readme.md", "README")

# Files that mean "this is a notebook dump / course archive", not a project.
ARCHIVE_PAT = re.compile(r"\.(zip|tar\.gz|pdf|pptx?|docx?|mp4|csv)$", re.I)


class RepoSignals:
    """Bounded, deterministic facts about one repo."""

    __slots__ = (
        "archive_files",
        "author_commits",
        "code_files",
        "commit_count",
        "dep_files",
        "error",
        "file_count",
        "first_commit",
        "has_ci",
        "has_docs",
        "has_tests",
        "last_commit",
        "notebook_files",
        "readme_chars",
        "readme_head",
        "readme_path",
        "repo",
        "top_dirs",
    )

    def __init__(self, repo: str):
        self.repo = repo
        self.readme_head = ""
        self.readme_chars = 0
        self.readme_path = ""
        self.file_count = 0
        self.code_files = 0
        self.notebook_files = 0
        self.archive_files = 0
        self.has_tests = False
        self.has_ci = False
        self.has_docs = False
        self.dep_files: list[str] = []
        self.commit_count = 0
        self.author_commits = 0
        self.first_commit = ""
        self.last_commit = ""
        self.top_dirs: list[str] = []
        self.error = ""

    def as_state(self, repo: RepoRef) -> dict:
        """The evidence blob handed to Jev. Facts only — no scoring, no adjectives."""
        return {
            "name": repo.name,
            "description": repo.description or None,
            "primary_language": repo.language or None,
            "readme_present": bool(self.readme_chars),
            "readme_length_chars": self.readme_chars,
            "readme_excerpt": self.readme_head or None,
            "total_files": self.file_count,
            "source_code_files": self.code_files,
            "notebook_files": self.notebook_files,
            "binary_or_document_files": self.archive_files,
            "top_level_directories": self.top_dirs or None,
            "has_test_suite": self.has_tests,
            "has_ci_config": self.has_ci,
            "has_docs": self.has_docs,
            "dependency_manifests": self.dep_files or None,
            "commits_sampled": self.commit_count,
            "commits_by_account_owner": self.author_commits,
            "repo_size_kb": repo.size_kb,
            "stars": repo.stars,
            "created": repo.created_at[:10],
            "last_push": repo.pushed_at[:10],
        }


CODE_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".java", ".c", ".h", ".cpp",
            ".hpp", ".rs", ".rb", ".cs", ".swift", ".kt", ".scala", ".sh", ".sql",
            ".vue", ".php", ".m", ".r"}


def classify_tree(paths: list[str], sig: RepoSignals) -> None:
    dirs: dict[str, int] = {}
    for p in paths:
        sig.file_count += 1
        lower = p.lower()
        dot = lower.rfind(".")
        ext = lower[dot:] if dot > 0 else ""
        if ext in CODE_EXT:
            sig.code_files += 1
        elif ext == ".ipynb":
            sig.notebook_files += 1
        elif ARCHIVE_PAT.search(lower):
            sig.archive_files += 1
        if TEST_PAT.search(p):
            sig.has_tests = True
        if CI_PAT.search(p):
            sig.has_ci = True
        if DOC_PAT.search(p):
            sig.has_docs = True
        base = p.rsplit("/", 1)[-1]
        if base in DEP_FILES and base not in sig.dep_files:
            sig.dep_files.append(base)
        if "/" in p:
            dirs[p.split("/", 1)[0]] = dirs.get(p.split("/", 1)[0], 0) + 1
    sig.top_dirs = [d for d, _ in sorted(dirs.items(), key=lambda kv: -kv[1])[:8]]


async def fetch_signals(backend, repo: RepoRef, owner: str,
                        readme_chars: int = 1200,
                        commit_sample: int = 30) -> RepoSignals:
    sig = RepoSignals(repo.name)
    try:
        tree_task = backend.get_tree(repo.name, repo.default_branch)
        commits_task = backend.list_commits(repo.name, commit_sample)
        tree, commits = await asyncio.gather(tree_task, commits_task,
                                             return_exceptions=True)
        if isinstance(tree, list):
            classify_tree(tree, sig)
            for cand in README_CANDIDATES:
                if cand in tree:
                    sig.readme_path = cand
                    break
            if not sig.readme_path:
                for p in tree:
                    if p.lower().startswith("readme"):
                        sig.readme_path = p
                        break
        if isinstance(commits, list):
            sig.commit_count = len(commits)
            sig.author_commits = sum(
                1 for c in commits if (c.get("author") or "").lower() == owner.lower()
            )
            if commits:
                sig.last_commit = commits[0].get("date", "")[:10]
                sig.first_commit = commits[-1].get("date", "")[:10]
        if sig.readme_path:
            text = await backend.get_file(repo.name, sig.readme_path, repo.default_branch)
            if text:
                sig.readme_chars = len(text)
                sig.readme_head = text[:readme_chars]
    except Exception as exc:  # one bad repo must not stop the sweep
        sig.error = str(exc)[:120]
        log.debug("signals failed for %s: %s", repo.name, exc)
    return sig


async def gather_signals(backend, repos: list[RepoRef], owner: str,
                         concurrency: int = 10) -> dict[str, RepoSignals]:
    sem = asyncio.Semaphore(concurrency)

    async def one(r: RepoRef):
        async with sem:
            return r.name, await fetch_signals(backend, r, owner)

    pairs = await asyncio.gather(*(one(r) for r in repos))
    return dict(pairs)
