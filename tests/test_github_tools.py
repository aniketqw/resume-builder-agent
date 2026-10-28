import json

import httpx
import pytest

# MCP ships as a base dependency, but skip rather than error if an environment
# somehow lacks it -- the REST backend still works without it.
lcmcp = pytest.importorskip("langchain.mcp")

from rba.config import Config
from rba.github_tools import (
    MCP_TOOLS,
    McpBackend,
    RestBackend,
    _is_session_error,
    _unwrap,
)


# -- MCP response unwrapping ---------------------------------------------
def test_unwrap_takes_the_last_text_block():
    """get_file_contents returns a status block first, the content second."""
    blocks = [{"type": "text", "text": "successfully downloaded text file (SHA: abc)"},
              {"type": "text", "text": "# My Project\nreal content"}]
    assert _unwrap(blocks) == "# My Project\nreal content"


def test_unwrap_parses_json_payloads():
    blocks = [{"type": "text", "text": json.dumps([{"name": "a.py", "type": "file"}])}]
    assert _unwrap(blocks) == [{"name": "a.py", "type": "file"}]


def test_unwrap_leaves_plain_text_alone():
    assert _unwrap([{"type": "text", "text": "# not json"}]) == "# not json"


def test_unwrap_handles_empty_results():
    assert _unwrap([]) is None
    assert _unwrap([{"type": "image"}]) is None


# -- session-failure detection -------------------------------------------
@pytest.mark.parametrize("msg", [
    "Internal error: nesting counter should be 0 when starting new session, got 1",
    "Timed out while waiting for response.",
    "peer closed connection",
])
def test_session_errors_trigger_a_reconnect(msg):
    """These all appeared live after a few hundred calls on the hosted endpoint."""
    assert _is_session_error(RuntimeError(msg))


def test_a_missing_file_is_not_a_session_error():
    assert not _is_session_error(RuntimeError("404 Not Found: no such path"))
    assert not _is_session_error(None)


# -- tool mapping ---------------------------------------------------------
def test_tool_map_matches_the_live_server():
    """Verified against api.githubcopilot.com/mcp/x/repos/readonly."""
    # There is no `list_repositories` tool; listing uses a search qualifier.
    assert MCP_TOOLS["list_repos"] == "search_repositories"
    assert MCP_TOOLS["file"] == "get_file_contents"
    assert MCP_TOOLS["commits"] == "list_commits"


# -- write-tool refusal ---------------------------------------------------
class FakeTool:
    def __init__(self, name, result=None):
        self.name = name
        self._result = result
        self.calls = []

    async def ainvoke(self, kwargs):
        self.calls.append(kwargs)
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


async def test_write_tools_abort_the_connection(monkeypatch):
    """The plain hosted endpoint exposes delete_file and push_files. Refuse it."""
    b = McpBackend(config=Config.load(), token="t", mode="remote")

    class Adapter:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def list_tools(self):
            return [FakeTool("get_file_contents"), FakeTool("delete_file"),
                    FakeTool("push_files")]

    monkeypatch.setattr(lcmcp, "MCPAdapter", lambda target: Adapter())
    with pytest.raises(RuntimeError, match="write tools"):
        await b.connect()


async def test_read_only_toolset_connects(monkeypatch):
    b = McpBackend(config=Config.load(), token="t", mode="remote")

    class Adapter:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def list_tools(self):
            return [FakeTool(n) for n in
                    ("get_file_contents", "list_commits", "search_repositories")]

    monkeypatch.setattr(lcmcp, "MCPAdapter", lambda target: Adapter())
    await b.connect()
    assert b.has_tool("file") and not b.has_tool("tree")


# -- target construction --------------------------------------------------
def test_remote_target_points_at_the_readonly_endpoint():
    b = McpBackend(config=Config.load(), token="tok", mode="remote")
    server = b._target()["mcpServers"]["github"]
    assert server["url"].endswith("/readonly"), "write tools must be unreachable"
    assert server["headers"]["Authorization"] == "Bearer tok"


def test_docker_target_passes_read_only_flag():
    b = McpBackend(config=Config.load(), token="tok", mode="docker")
    server = b._target()["mcpServers"]["github"]
    assert server["command"] == "docker"
    assert "--read-only" in server["args"]
    assert "--toolsets" in server["args"]


# -- REST backend ---------------------------------------------------------
def rest_with(handler):
    return RestBackend(token="t", config=Config.load(),
                       transport=httpx.MockTransport(handler))


async def test_rest_lists_private_repos_via_user_endpoint():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "me"})
        return httpx.Response(200, json=[
            {"name": "secret", "private": True, "fork": False, "size": 10,
             "default_branch": "main"}])

    b = rest_with(handler)
    repos = await b.list_repos("me")
    await b.aclose()
    assert repos[0].is_private
    # /user/repos is the only endpoint that returns private repos.
    assert any("/user/repos" in u for u in seen)


async def test_rest_decodes_base64_file_contents():
    import base64

    def handler(request):
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "me"})
        return httpx.Response(200, json={
            "encoding": "base64",
            "content": base64.b64encode(b"# hello").decode()})

    b = rest_with(handler)
    assert await b.get_file("r", "README.md") == "# hello"
    await b.aclose()


async def test_rest_returns_none_for_a_missing_file():
    def handler(request):
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "me"})
        return httpx.Response(404, json={"message": "Not Found"})

    b = rest_with(handler)
    assert await b.get_file("r", "nope.md") is None
    await b.aclose()


async def test_rest_tree_returns_only_blobs():
    def handler(request):
        if request.url.path == "/user":
            return httpx.Response(200, json={"login": "me"})
        return httpx.Response(200, json={"tree": [
            {"path": "src", "type": "tree"},
            {"path": "src/a.py", "type": "blob"}]})

    b = rest_with(handler)
    assert await b.get_tree("r") == ["src/a.py"]
    await b.aclose()
