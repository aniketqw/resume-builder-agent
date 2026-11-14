"""End-to-end graph test with no network: fake GitHub, fake Groq, fake Jev."""
import json

import pytest

from fakes import FakeGroq, FakeJev, chat_response
from rba.cache import Cache
from rba.config import Config
from rba.graph import Runner, build_graph
from rba.groq_client import GroqClient
from rba.jev_client import JevClient
from rba.limiter import Limiter
from rba.state import ClaimStatus, RepoRef

README = ("# Demo Service\n"
          "A FastAPI service with 34 tests that run in about 2 s.\n"
          "Built with Python and Docker.\n")


class FakeGitHub:
    def __init__(self):
        self.closed = False

    async def list_repos(self, username):
        return [RepoRef(name="demo", language="Python", size_kb=500,
                        default_branch="main", description="A demo service")]

    async def get_tree(self, repo, ref=""):
        return ["README.md", "src/app.py", "tests/test_app.py",
                ".github/workflows/ci.yml", "requirements.txt"]

    async def list_commits(self, repo, limit=30):
        return [{"sha": f"sha{i:07d}", "author": "aniketqw", "message": f"commit {i}",
                 "date": "2026-10-01T00:00:00Z"} for i in range(12)]

    async def get_file(self, repo, path, ref=""):
        return {"README.md": README,
                "requirements.txt": "fastapi==0.110.0\nhttpx>=0.27\n"}.get(path)

    async def get_languages(self, repo):
        return {"Python": 1000}

    async def aclose(self):
        self.closed = True


SUMMARY = json.dumps({
    "one_line": "A FastAPI demo service.",
    "problem": "Needs an HTTP interface.",
    "approach": "Implements endpoints with FastAPI and tests them.",
    "tech": ["Python", "FastAPI", "Docker"],
    "claims": [
        {"text": "Ships 34 tests that run in about 2 s",
         "evidence": [{"path": "README.md", "quote": "34 tests that run in about 2 s"}]},
        {"text": "Reduced latency by 40%",
         "evidence": [{"path": "README.md", "quote": "A FastAPI service"}]},
    ],
    "not_claimable_reason": "",
})

RESUME = json.dumps({
    "headline": "Software Engineer",
    "summary": "Builds and tests HTTP services.",
    "projects": ["Shipped 34 tests covering a FastAPI service",
                 "Built a service handling 9,999 requests per second"],
    "skills": ["Python", "FastAPI", "Docker"],
    "used_claim_ids": ["demo#0"],
})

PLAN = json.dumps({"steps": [{
    "title": "Add container orchestration",
    "gap": "No Kubernetes evidence",
    "evidence_found": "demo has a Dockerfile",
    "action": "Add a Helm chart to demo",
    "hours": 8.0,
    "measurable_output": "A deployed chart with a readiness probe",
    "repo": "demo",
}]})


@pytest.fixture
def runner(monkeypatch):
    cfg = Config.load()
    cfg.raw["report"]["roles"] = ["balanced"]

    github = FakeGitHub()
    # One summary, one resume, one plan — then a safe default for anything else.
    groq = FakeGroq([chat_response(SUMMARY), chat_response(RESUME),
                     chat_response(PLAN), chat_response(PLAN)])
    jev = FakeJev({"claimable": {"noul": 0.9},
                   "tier": {"choice": "small", "confidence": 0.9}})

    r = Runner(cfg, contact="a@b.com | github.com/x")

    async def fake_connect(state):
        r.backend = github
        r.client = GroqClient(config=cfg, limiter=Limiter(cfg), cache=Cache(enabled=False),
                              api_key="k", transport=groq.transport())
        return {"username": state.get("username", "")}

    monkeypatch.setattr(r, "connect", fake_connect)

    async def fake_select(state):
        from rba.nodes.select import select_repos
        client = JevClient(cfg, api_key="k", transport=jev.transport())
        try:
            return await select_repos(state, config=cfg, jev=client, backend=github)
        finally:
            await client.aclose()

    monkeypatch.setattr(r, "select", fake_select)
    r._fakes = (github, groq, jev)
    return r


async def test_full_graph_runs_offline(runner, tmp_path):
    """The CLI branch: full_run=True walks every node through to the report."""
    runner.config.raw["report"]["out_dir"] = str(tmp_path)
    graph = build_graph(runner)
    result = await graph.ainvoke({"username": "aniketqw", "full_run": True},
                                 {"recursion_limit": 50})
    await runner.aclose()

    assert [r.name for r in result["selected"]] == ["demo"]
    assert "demo" in result["facts"] and result["facts"]["demo"].has_ci

    statuses = {c.text: c.status for c in result["claims"]}
    assert statuses["Ships 34 tests that run in about 2 s"] is ClaimStatus.VERIFIED
    # 40% appears nowhere in the evidence.
    assert statuses["Reduced latency by 40%"] is ClaimStatus.NEEDS_MEASURE

    # The browser/CLI join point ran too, on the way to the role resumes.
    assert result["document"].all_bullets()

    assert "balanced" in result["resumes"]
    assert result["scores"]["balanced"].total > 0
    assert result["comparison"]["best_role"] == "balanced"
    assert result["plan"]

    # The fabricated throughput bullet must not survive into the resume.
    bullets = result["resumes"]["balanced"].sections["projects"]
    assert not any("9,999" in b for b in bullets), "invented number reached the resume"
    assert any("34 tests" in b for b in bullets)

    assert (tmp_path / "report.md").exists()
    md = (tmp_path / "report.md").read_text()
    assert "Measure these" in md


async def test_browser_import_stops_at_the_document(runner, tmp_path):
    """The thing that proves CLI and browser share one graph: the same
    fixtures, the same graph, and omitting `full_run` is the only
    difference. No role resumes, no plan, no report file -- just the
    evidence-backed document the browser actually shows."""
    runner.config.raw["report"]["out_dir"] = str(tmp_path)
    graph = build_graph(runner)
    result = await graph.ainvoke({"username": "aniketqw"}, {"recursion_limit": 50})
    await runner.aclose()

    doc = result["document"]
    bullets = doc.all_bullets()
    assert bullets and any("34 tests" in b.text for b in bullets)
    # Every surviving bullet must carry enough to prove itself.
    verified = [b for b in bullets if b.verified]
    assert verified and all(b.evidence_sha for b in verified)

    assert "resumes" not in result
    assert "plan" not in result
    assert not (tmp_path / "report.md").exists()


def test_node_progress_covers_every_browser_branch_node(runner):
    """Guards the one real regression risk in wiring NODE_PROGRESS by hand in
    pipeline.py: a node renamed in build_graph() must not silently lose its
    SSE progress frame."""
    from rba.resume.pipeline import NODE_PROGRESS

    graph = build_graph(runner)
    cli_only = {"build_resumes", "compare", "plan_improvements", "report"}
    browser_nodes = (set(graph.get_graph().nodes)
                    - {"__start__", "__end__"} - cli_only)
    assert set(NODE_PROGRESS) == browser_nodes
