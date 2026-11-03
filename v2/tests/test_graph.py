"""End-to-end graph test with no network: fake GitHub, fake Groq, fake Jev."""
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

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

    r = Runner(cfg, contact="a@b.com | github.com/x", review=False)

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
    runner.config.raw["report"]["out_dir"] = str(tmp_path)
    graph = build_graph(runner, InMemorySaver())
    result = await graph.ainvoke({"username": "aniketqw"},
                                 {"configurable": {"thread_id": "t1"}})
    await runner.aclose()

    assert [r.name for r in result["selected"]] == ["demo"]
    assert "demo" in result["facts"] and result["facts"]["demo"].has_ci

    statuses = {c.text: c.status for c in result["claims"]}
    assert statuses["Ships 34 tests that run in about 2 s"] is ClaimStatus.VERIFIED
    # 40% appears nowhere in the evidence.
    assert statuses["Reduced latency by 40%"] is ClaimStatus.NEEDS_MEASURE

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


async def test_graph_pauses_for_human_review(runner, tmp_path):
    runner.review = True
    runner.config.raw["report"]["out_dir"] = str(tmp_path)
    graph = build_graph(runner, InMemorySaver())
    result = await graph.ainvoke({"username": "aniketqw"},
                                 {"configurable": {"thread_id": "t2"}})
    await runner.aclose()
    assert "__interrupt__" in result, "review must stop the graph before the report"
    payload = result["__interrupt__"][0].value
    assert payload["best_role"] == "balanced"
    assert not (tmp_path / "report.md").exists(), "no report before approval"


async def test_review_resumes_and_writes_the_report(runner, tmp_path):
    from langgraph.types import Command

    runner.review = True
    runner.config.raw["report"]["out_dir"] = str(tmp_path)
    graph = build_graph(runner, InMemorySaver())
    cfg = {"configurable": {"thread_id": "t3"}}
    await graph.ainvoke({"username": "aniketqw"}, cfg)
    result = await graph.ainvoke(Command(resume={"approved": True, "note": "ship it"}),
                                 cfg)
    await runner.aclose()
    assert result["review"]["approved"] is True
    assert (tmp_path / "report.md").exists()
