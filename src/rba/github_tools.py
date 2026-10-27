"""GitHub access.

Preferred path is the official **GitHub MCP server** (``ghcr.io/github/github-mcp-server``)
launched over stdio in ``--read-only`` mode, which is a hard gate: write tools are
stripped from the toolset regardless of what asks for them.

A REST client implements the identical interface so the agent still runs when
Docker is unavailable, and so the offline test suite has something to fake.
Both satisfy :class:`GitHubBackend`.
"""
from __future__ import annotations

import base64
import json
import logging
import shutil
from typing import Any, Protocol

import httpx

from .config import Config, secret
from .state import RepoRef

log = logging.getLogger(__name__)

# Tool names in github-mcp-server. There is no `list_repositories`: listing a
# user's repos goes through search_repositories with a `user:` qualifier.
MCP_TOOLS = {
    "list_repos": "search_repositories",
    "file": "get_file_contents",
    "tree": "get_repository_tree",
    "commits": "list_commits",
    "search_code": "search_code",
    "me": "get_me",
}


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
class McpBackend:
    """GitHub MCP server over stdio (docker) or HTTP (GitHub-hosted).

    Tools are discovered at connect time; we map our verbs onto the server's
    tool names via :data:`MCP_TOOLS`.
    """

    def __init__(self, config: Config | None = None, token: str | None = None):
        self.config = config or Config.load()
        self._token = token if token is not None else secret("GITHUB_PERSONAL_ACCESS_TOKEN")
        self.mode = self.config.get("github.mcp.mode", "docker")
        self._adapter = None
        self._tools: dict[str, Any] = {}
        self.owner = ""

    @staticmethod
    def docker_available() -> bool:
        if not shutil.which("docker"):
            return False
        try:
            return subprocess_run_ok(["docker", "info"])
        except Exception:
            return False

    def _server_spec(self) -> Any:
        image = self.config.get("github.mcp.image")
        toolsets = ",".join(self.config.get("github.mcp.toolsets", []))
        args = ["run", "-i", "--rm",
                "-e", f"GITHUB_PERSONAL_ACCESS_TOKEN={self._token}",
                image]
        if toolsets:
            args += ["--toolsets", toolsets]
        if self.config.get("github.mcp.read_only", True):
            args.append("--read-only")
        return {"command": "docker", "args": args, "transport": "stdio"}

    async def connect(self) -> None:
        from langchain.mcp import MCPAdapter  # imported lazily: optional extra

        if self.mode == "remote":
            target: Any = self.config.get("github.mcp.remote_url")
            self._adapter = MCPAdapter(
                target, headers={"Authorization": f"Bearer {self._token}"}
            )
        else:
            self._adapter = MCPAdapter(self._server_spec())
        await self._adapter.__aenter__()
        tools = await self._adapter.list_tools()
        self._tools = {t.name: t for t in tools}
        log.info("MCP connected, %d tools: %s", len(self._tools), sorted(self._tools)[:8])

    async def aclose(self) -> None:
        if self._adapter is not None:
            await self._adapter.__aexit__(None, None, None)
            self._adapter = None

    def has_tool(self, verb: str) -> bool:
        return MCP_TOOLS.get(verb, "") in self._tools

    async def _call(self, verb: str, **kwargs) -> Any:
        name = MCP_TOOLS[verb]
        tool = self._tools.get(name)
        if tool is None:
            raise RuntimeError(f"MCP server does not expose {name!r}; "
                               f"available: {sorted(self._tools)}")
        result = await tool.ainvoke(kwargs)
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result
        return result

    async def whoami(self) -> str:
        data = await self._call("me")
        if isinstance(data, dict):
            return data.get("login", "")
        return ""

    async def list_repos(self, username: str) -> list[RepoRef]:
        out: list[RepoRef] = []
        page = 1
        while True:
            data = await self._call("list_repos", query=f"user:{username}",
                                    page=page, perPage=100)
            items = data.get("items", data) if isinstance(data, dict) else data
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
                ))
            if len(items) < 100:
                break
            page += 1
        return out

    async def get_file(self, repo: str, path: str, ref: str = "") -> str | None:
        owner = self.owner or await self.whoami()
        data = await self._call("file", owner=owner, repo=repo, path=path,
                                **({"ref": ref} if ref else {}))
        if isinstance(data, dict):
            if data.get("encoding") == "base64" and data.get("content"):
                return base64.b64decode(data["content"]).decode("utf-8", "replace")
            return data.get("content") or data.get("text")
        return data if isinstance(data, str) else None

    async def get_tree(self, repo: str, ref: str = "") -> list[str]:
        owner = self.owner or await self.whoami()
        data = await self._call("tree", owner=owner, repo=repo,
                                **({"ref": ref} if ref else {}))
        nodes = data.get("tree", data) if isinstance(data, dict) else data
        if not isinstance(nodes, list):
            return []
        return [n["path"] for n in nodes
                if isinstance(n, dict) and n.get("type") in ("blob", "file")]

    async def list_commits(self, repo: str, limit: int = 30) -> list[dict[str, str]]:
        owner = self.owner or await self.whoami()
        data = await self._call("commits", owner=owner, repo=repo, perPage=min(limit, 100))
        items = data.get("commits", data) if isinstance(data, dict) else data
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
        # Not exposed as an MCP tool; the language is already on the repo record.
        return {}


def subprocess_run_ok(cmd: list[str]) -> bool:
    import subprocess
    return subprocess.run(cmd, capture_output=True, check=False, timeout=20).returncode == 0


async def make_backend(config: Config | None = None, mode: str | None = None):
    """Build a backend, falling back from MCP to REST with a clear log line."""
    cfg = config or Config.load()
    mode = mode or cfg.get("github.mcp.mode", "docker")
    if mode == "rest":
        return RestBackend(config=cfg)
    backend = McpBackend(config=cfg)
    if mode == "docker" and not McpBackend.docker_available():
        log.warning("Docker daemon unavailable; falling back to the REST backend.")
        return RestBackend(config=cfg)
    try:
        await backend.connect()
        return backend
    except Exception as exc:
        log.warning("MCP backend unavailable (%s); falling back to REST.", exc)
        await backend.aclose()
        return RestBackend(config=cfg)


__all__ = ["MCP_TOOLS", "GitHubBackend", "McpBackend", "RestBackend", "make_backend"]
