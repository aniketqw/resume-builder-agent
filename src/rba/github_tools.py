"""GitHub access.

Preferred path is the official **GitHub MCP server** (``ghcr.io/github/github-mcp-server``)
launched over stdio in ``--read-only`` mode, which is a hard gate: write tools are
stripped from the toolset regardless of what asks for them.

A REST client implements the identical interface so the agent still runs when
Docker is unavailable, and so the offline test suite has something to fake.
Both satisfy :class:`GitHubBackend`.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
from typing import Any, Protocol

import httpx

from .config import Config, secret
from .state import RepoRef

log = logging.getLogger(__name__)

class GitHubBackend(Protocol):
    async def list_repos(self, username: str) -> list[RepoRef]: ...
    async def get_file(self, repo: str, path: str, ref: str = "") -> str | None: ...
    async def get_tree(self, repo: str, ref: str = "") -> list[str]: ...
    async def list_commits(self, repo: str, limit: int = 30) -> list[dict[str, str]]: ...
    async def get_languages(self, repo: str) -> dict[str, int]: ...
    async def aclose(self) -> None: ...


# ---------------------------------------------------------------- REST ----
class RestBackend:
    """Dependency-free fallback over api.github.com."""

    def __init__(self, token: str | None = None, config: Config | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        self.config = config or Config.load()
        self._token = token if token is not None else secret("GITHUB_PERSONAL_ACCESS_TOKEN")
        self._client = httpx.AsyncClient(
            base_url="https://api.github.com",
            timeout=60.0,
            transport=transport,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        self.owner = ""
        self.calls = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def _get(self, path: str, **params) -> Any:
        self.calls += 1
        r = await self._client.get(path, params=params or None)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    async def whoami(self) -> str:
        data = await self._get("/user")
        return (data or {}).get("login", "")

    async def list_repos(self, username: str) -> list[RepoRef]:
        """All repos for ``username``. Uses /user/repos when it is the token owner,
        since that is the only endpoint that returns private repositories."""
        self.owner = self.owner or await self.whoami()
        out: list[RepoRef] = []
        page = 1
        while True:
            if username == self.owner:
                batch = await self._get("/user/repos", per_page=100, page=page,
                                        affiliation="owner", sort="pushed")
            else:
                batch = await self._get(f"/users/{username}/repos", per_page=100,
                                        page=page, sort="pushed")
            if not batch:
                break
            for r in batch:
                out.append(RepoRef(
                    name=r["name"],
                    full_name=r.get("full_name", ""),
                    description=r.get("description") or "",
                    language=(r.get("language") or ""),
                    is_fork=bool(r.get("fork")),
                    is_private=bool(r.get("private")),
                    is_archived=bool(r.get("archived")),
                    stars=r.get("stargazers_count", 0),
                    size_kb=r.get("size", 0),
                    pushed_at=r.get("pushed_at") or "",
                    created_at=r.get("created_at") or "",
                    default_branch=r.get("default_branch") or "main",
                    topics=r.get("topics") or [],
                ))
            if len(batch) < 100:
                break
            page += 1
        return out

    async def get_file(self, repo: str, path: str, ref: str = "") -> str | None:
        owner = self.owner or await self.whoami()
        data = await self._get(f"/repos/{owner}/{repo}/contents/{path}",
                               **({"ref": ref} if ref else {}))
        if not data or isinstance(data, list):
            return None
        if data.get("encoding") == "base64":
            try:
                return base64.b64decode(data["content"]).decode("utf-8", "replace")
            except Exception:
                return None
        return data.get("content")

    async def get_tree(self, repo: str, ref: str = "") -> list[str]:
        owner = self.owner or await self.whoami()
        ref = ref or "HEAD"
        data = await self._get(f"/repos/{owner}/{repo}/git/trees/{ref}", recursive="1")
        if not data:
            return []
        return [n["path"] for n in data.get("tree", []) if n.get("type") == "blob"]

    async def list_commits(self, repo: str, limit: int = 30) -> list[dict[str, str]]:
        owner = self.owner or await self.whoami()
        data = await self._get(f"/repos/{owner}/{repo}/commits", per_page=min(limit, 100))
        if not data:
            return []
        out = []
        for c in data[:limit]:
            out.append({
                "sha": c.get("sha", "")[:10],
                "message": ((c.get("commit") or {}).get("message") or "").split("\n")[0][:200],
                "author": ((c.get("author") or {}) or {}).get("login", "")
                          or ((c.get("commit") or {}).get("author") or {}).get("name", ""),
                "date": ((c.get("commit") or {}).get("author") or {}).get("date", ""),
            })
        return out

    async def get_languages(self, repo: str) -> dict[str, int]:
        owner = self.owner or await self.whoami()
        return await self._get(f"/repos/{owner}/{repo}/languages") or {}


# ----------------------------------------------------------------- MCP ----
# Tool names on the GitHub MCP server. Verified live against
# https://api.githubcopilot.com/mcp/x/repos/readonly on 2026-10-28.
# There is no `list_repositories`: listing a user's repos goes through
# search_repositories with a `user:` qualifier. There is also no
# `get_repository_tree` on the hosted endpoint (it lives in the `git` toolset,
# which hosted does not expose), so the tree is walked with get_file_contents.
MCP_TOOLS = {
    "list_repos": "search_repositories",
    "file": "get_file_contents",
    "commits": "list_commits",
    "search_code": "search_code",
    "branches": "list_branches",
    "tree": "get_repository_tree",   # `git` toolset; absent on the hosted endpoint
}

# Directories never worth walking — they explode the call count and contain no
# authored code.
SKIP_DIRS = {"node_modules", ".git", "venv", ".venv", "env", "__pycache__",
             "dist", "build", "site-packages", ".next", "target", "vendor",
             ".pytest_cache", ".mypy_cache", "coverage", ".idea", ".vscode"}


# Errors that mean "the session is gone", as opposed to "this repo has no such
# file". Seen live after a few hundred calls against the hosted endpoint.
_SESSION_ERRORS = ("nesting counter", "timed out", "session", "closed",
                   "connection", "anyio", "cancel")


def _is_session_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    text = str(exc).lower()
    return any(s in text for s in _SESSION_ERRORS)


def _unwrap(result: Any) -> Any:
    """MCP tools return a list of content blocks; the payload is the last text one."""
    if isinstance(result, list):
        texts = [b.get("text") for b in result
                 if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
        if not texts:
            return None
        result = texts[-1]
    if isinstance(result, str):
        stripped = result.strip()
        if stripped[:1] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return result
        return result
    return result


class McpBackend:
    """GitHub through the official MCP server.

    Default target is GitHub's hosted endpoint with the read-only repos
    toolset. That choice is deliberate: the plain hosted endpoint
    (``/mcp/``) exposes write tools — ``delete_file``, ``push_files``,
    ``create_repository`` — which this agent must never hold. The
    ``/x/repos/readonly`` path returns 13 tools and zero write tools, verified
    by enumerating both.

    ``mode: docker`` runs ``ghcr.io/github/github-mcp-server`` over stdio with
    ``--read-only`` instead, for anyone who prefers a local container.
    """

    def __init__(self, config: Config | None = None, token: str | None = None,
                 mode: str | None = None, concurrency: int = 4):
        self.config = config or Config.load()
        self._token = token if token is not None else secret("GITHUB_PERSONAL_ACCESS_TOKEN")
        self.mode = mode or self.config.get("github.mcp.mode", "remote")
        self._adapter = None
        self._tools: dict[str, Any] = {}
        self.owner = ""
        self.calls = 0
        self.retries = 0
        # The hosted endpoint degrades under bursty load: trees start coming
        # back empty and the session eventually times out. Cap in-flight calls.
        self._sem = asyncio.Semaphore(concurrency)
        self._rest: RestBackend | None = None
        self._lock = asyncio.Lock()
        self.reconnects = 0

    @staticmethod
    def docker_available() -> bool:
        if not shutil.which("docker"):
            return False
        return subprocess_run_ok(["docker", "info"])

    def _target(self) -> dict:
        """A FastMCP MCPConfig dict, which is what MCPAdapter accepts."""
        if self.mode == "docker":
            image = self.config.get("github.mcp.image")
            args = ["run", "-i", "--rm",
                    "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                    image]
            toolsets = ",".join(self.config.get("github.mcp.toolsets", []))
            if toolsets:
                args += ["--toolsets", toolsets]
            if self.config.get("github.mcp.read_only", True):
                args.append("--read-only")
            return {"mcpServers": {"github": {
                "command": "docker", "args": args,
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": self._token}}}}
        url = self.config.get("github.mcp.remote_url")
        return {"mcpServers": {"github": {
            "url": url, "headers": {"Authorization": f"Bearer {self._token}"}}}}

    async def connect(self) -> None:
        from langchain.mcp import MCPAdapter  # optional extra: langchain[mcp]

        self._adapter = MCPAdapter(self._target())
        await self._adapter.__aenter__()
        self._tools = {t.name: t for t in await self._adapter.list_tools()}
        writes = {"create_or_update_file", "delete_file", "push_files",
                  "create_repository", "delete_repository", "fork_repository",
                  "merge_pull_request", "create_branch"} & set(self._tools)
        if writes:
            # Refuse to proceed with write capability we never asked for.
            await self.aclose()
            raise RuntimeError(
                f"MCP server exposed write tools {sorted(writes)}. Point "
                f"github.mcp.remote_url at the read-only endpoint "
                f"(.../mcp/x/repos/readonly) or run the container with --read-only."
            )
        log.info("MCP connected (%s): %d read-only tools", self.mode, len(self._tools))

    async def aclose(self) -> None:
        if self._rest is not None:
            await self._rest.aclose()
            self._rest = None
        await self._close_adapter()

    async def __aenter__(self):
        if self._adapter is None:
            await self.connect()
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    def has_tool(self, verb: str) -> bool:
        return MCP_TOOLS.get(verb, "") in self._tools

    async def _call(self, verb: str, attempts: int = 3, **kwargs) -> Any:
        name = MCP_TOOLS[verb]
        tool = self._tools.get(name)
        if tool is None:
            raise RuntimeError(f"MCP server does not expose {name!r}; "
                               f"available: {sorted(self._tools)}")
        last: Exception | None = None
        for attempt in range(attempts):
            async with self._sem:
                self.calls += 1
                try:
                    return _unwrap(await tool.ainvoke(kwargs))
                except Exception as exc:
                    last = exc
                    self.retries += 1
            # A long sweep outlives the server's session: once it dies, every
            # later call fails with the same error until the client reconnects.
            # Treat a failure as a possibly-dead session and rebuild it.
            if _is_session_error(last):
                await self._reconnect()
                tool = self._tools.get(name, tool)
            await asyncio.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"MCP {name} failed after {attempts} attempts: {last}")

    async def _reconnect(self) -> None:
        async with self._lock:
            self.reconnects += 1
            log.info("rebuilding MCP session (#%d)", self.reconnects)
            await self._close_adapter()
            try:
                await self.connect()
            except Exception as exc:
                log.warning("MCP reconnect failed: %s", exc)

    def _rest_backend(self) -> RestBackend:
        """A REST client used only where MCP has no single-call equivalent."""
        if self._rest is None:
            self._rest = RestBackend(token=self._token, config=self.config)
            self._rest.owner = self.owner
        return self._rest

    async def _close_adapter(self) -> None:
        if self._adapter is not None:
            try:
                await self._adapter.__aexit__(None, None, None)
            except Exception as exc:  # a timing-out session must not mask results
                log.debug("MCP session close failed: %s", exc)
            self._adapter = None

    async def whoami(self) -> str:
        # get_me lives in the `context` toolset, which the read-only repos
        # endpoint does not include. The username is configuration, not discovery.
        if not self.owner:
            self.owner = os.environ.get("GITHUB_USERNAME", "")
        return self.owner

    async def list_repos(self, username: str) -> list[RepoRef]:
        self.owner = self.owner or username
        out: list[RepoRef] = []
        page = 1
        while page <= 10:
            data = await self._call("list_repos", query=f"user:{username}",
                                    page=page, perPage=100)
            items = data.get("items", []) if isinstance(data, dict) else data
            if not items:
                break
            for r in items:
                out.append(RepoRef(
                    name=r.get("name", ""),
                    full_name=r.get("full_name", ""),
                    description=r.get("description") or "",
                    language=r.get("language") or "",
                    is_fork=bool(r.get("fork")),
                    is_private=bool(r.get("private")),
                    is_archived=bool(r.get("archived")),
                    stars=r.get("stargazers_count", 0),
                    size_kb=r.get("size", 0),
                    pushed_at=r.get("pushed_at") or "",
                    created_at=r.get("created_at") or "",
                    default_branch=r.get("default_branch") or "main",
                    topics=r.get("topics") or [],
                ))
            if len(items) < 100:
                break
            page += 1
        return out

    async def get_file(self, repo: str, path: str, ref: str = "") -> str | None:
        owner = self.owner or await self.whoami()
        data = await self._call("file", owner=owner, repo=repo, path=path,
                                **({"ref": ref} if ref else {}))
        if isinstance(data, str):
            return data
        if isinstance(data, dict) and data.get("content"):
            if data.get("encoding") == "base64":
                return base64.b64decode(data["content"]).decode("utf-8", "replace")
            return data["content"]
        return None

    async def _list_dir(self, repo: str, path: str, ref: str = "") -> list[dict]:
        owner = self.owner or await self.whoami()
        data = await self._call("file", owner=owner, repo=repo,
                                path=path or "/",
                                fields=["type", "name", "path", "size"],
                                **({"ref": ref} if ref else {}))
        return data if isinstance(data, list) else []

    async def get_tree(self, repo: str, ref: str = "", max_depth: int = 3,
                       max_entries: int = 400) -> list[str]:
        """Return every file path in the repo, in as few calls as possible.

        Three strategies, best first:

        1. ``get_repository_tree`` — one call. Lives in the ``git`` toolset,
           which the local container exposes but the hosted endpoint does not.
        2. The REST trees API — also one call, and the reason ``tree_via_rest``
           defaults to true. A recursive directory walk over MCP costs one call
           per directory; doing that for ~100 repos issued ~500 calls and drove
           the hosted endpoint into empty responses and session timeouts.
        3. A bounded breadth-first walk with ``get_file_contents`` — the
           last resort, used only when REST is disabled.
        """
        if self.has_tool("tree"):
            data = await self._call("tree", owner=self.owner or await self.whoami(),
                                    repo=repo, **({"ref": ref} if ref else {}))
            nodes = data.get("tree", data) if isinstance(data, dict) else data
            if isinstance(nodes, list):
                return [n["path"] for n in nodes
                        if isinstance(n, dict) and n.get("type") in ("blob", "file")
                        and n.get("path")]

        if self.config.get("github.mcp.tree_via_rest", True):
            try:
                rest = self._rest_backend()
                rest.owner = self.owner or await self.whoami()
                return await rest.get_tree(repo, ref)
            except Exception as exc:
                log.debug("REST tree fallback failed for %s: %s", repo, exc)

        return await self._walk_tree(repo, ref, max_depth, max_entries)

    async def _walk_tree(self, repo: str, ref: str, max_depth: int,
                         max_entries: int) -> list[str]:
        files: list[str] = []
        frontier = [("", 0)]
        while frontier and len(files) < max_entries:
            path, depth = frontier.pop(0)
            try:
                entries = await self._list_dir(repo, path, ref)
            except Exception as exc:  # one unreadable directory is not fatal
                log.debug("tree walk failed at %s/%s: %s", repo, path, exc)
                continue
            for e in entries:
                if not isinstance(e, dict):
                    continue
                name, kind = e.get("name", ""), e.get("type")
                full = e.get("path") or (f"{path}/{name}" if path else name)
                if kind == "dir":
                    if name not in SKIP_DIRS and depth + 1 <= max_depth:
                        frontier.append((full, depth + 1))
                elif kind == "file":
                    files.append(full)
                    if len(files) >= max_entries:
                        break
        return files

    async def list_commits(self, repo: str, limit: int = 30) -> list[dict[str, str]]:
        owner = self.owner or await self.whoami()
        data = await self._call("commits", owner=owner, repo=repo,
                                perPage=min(limit, 100))
        items = data.get("commits", []) if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        out = []
        for c in items[:limit]:
            commit = c.get("commit") or {}
            out.append({
                "sha": (c.get("sha") or "")[:10],
                "message": (commit.get("message") or "").split("\n")[0][:200],
                "author": (c.get("author") or {}).get("login", "")
                          or (commit.get("author") or {}).get("name", ""),
                "date": (commit.get("author") or {}).get("date", ""),
            })
        return out

    async def get_languages(self, repo: str) -> dict[str, int]:
        # No MCP equivalent exists; the REST endpoint is one cheap call.
        if not self.config.get("github.mcp.tree_via_rest", True):
            return {}
        try:
            rest = self._rest_backend()
            rest.owner = self.owner or await self.whoami()
            return await rest.get_languages(repo)
        except Exception:
            return {}

    async def search_code(self, query: str, repo: str = "") -> list[dict]:
        q = f"{query} repo:{self.owner}/{repo}" if repo else query
        data = await self._call("search_code", query=q, perPage=20)
        return data.get("items", []) if isinstance(data, dict) else (data or [])


def subprocess_run_ok(cmd: list[str]) -> bool:
    import subprocess
    return subprocess.run(cmd, capture_output=True, check=False, timeout=20).returncode == 0


async def make_backend(config: Config | None = None, mode: str | None = None):
    """Build a backend, falling back from MCP to REST with a clear log line."""
    cfg = config or Config.load()
    mode = mode or cfg.get("github.mcp.mode", "remote")
    if mode == "rest":
        return RestBackend(config=cfg)
    if mode == "docker" and not McpBackend.docker_available():
        log.warning("Docker daemon unavailable; using GitHub's hosted MCP endpoint.")
        mode = "remote"
    backend = McpBackend(config=cfg, mode=mode)
    try:
        await backend.connect()
        return backend
    except Exception as exc:
        log.warning("MCP backend unavailable (%s); falling back to REST.", exc)
        await backend.aclose()
        return RestBackend(config=cfg)


__all__ = ["MCP_TOOLS", "GitHubBackend", "McpBackend", "RestBackend", "make_backend"]
