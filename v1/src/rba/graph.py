"""The LangGraph wiring.

Node order matters for one non-obvious reason: Groq cannot combine tool use
with structured outputs in a single request, so every node that touches MCP
tools (`fetch_facts`) is strictly separate from every node that asks for strict
JSON (`summarize_repo`, `build_resumes`, `plan_improvements`).

A SQLite checkpointer is not optional here. At 8,000 tokens per minute a full
run stalls repeatedly, and `human_review` interrupts the graph outright — both
need durable state to resume from.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .config import PROJECT_ROOT, Config
from .github_tools import make_backend
from .groq_client import GroqClient
from .nodes.build import build_resumes, comparison_table
from .nodes.fetch import fetch_facts
from .nodes.plan import plan_improvements
from .nodes.safety import safety_scan
from .nodes.select import select_repos
from .nodes.summarize import summarize_repos
from .nodes.verify import verify_claims
from .report import write_reports
from .scoring import load_roles
from .state import AgentState

log = logging.getLogger(__name__)

CHECKPOINT_PATH = PROJECT_ROOT / ".cache" / "checkpoints.sqlite"


class Runner:
    """Holds the clients the nodes need, so nodes stay plain functions."""

    def __init__(self, config: Config | None = None, contact: str = "",
                 review: bool = True):
        self.config = config or Config.load()
        self.contact = contact or self.config.get("report.contact", "") or ""
        self.review = review
        self.backend = None
        self.client: GroqClient | None = None
        self.started = time.time()

    # -- nodes ------------------------------------------------------------
    async def connect(self, state: AgentState) -> AgentState:
        self.backend = await make_backend(self.config)
        self.client = GroqClient(self.config)
        username = state.get("username") or ""
        log.info("connected via %s for user %s",
                 type(self.backend).__name__, username)
        return {"username": username}

    async def list_repos(self, state: AgentState) -> AgentState:
        repos = await self.backend.list_repos(state["username"])
        log.info("found %d repositories", len(repos))
        return {"repos": repos}

    async def select(self, state: AgentState) -> AgentState:
        return await select_repos(state, config=self.config, backend=self.backend)

    async def fetch(self, state: AgentState) -> AgentState:
        return await fetch_facts(state, backend=self.backend, config=self.config)

    async def scan(self, state: AgentState) -> AgentState:
        return await safety_scan(state)

    async def summarize(self, state: AgentState) -> AgentState:
        return await summarize_repos(state, client=self.client, config=self.config)

    async def verify(self, state: AgentState) -> AgentState:
        return await verify_claims(state)

    async def build(self, state: AgentState) -> AgentState:
        return await build_resumes(state, client=self.client, config=self.config,
                                   contact=self.contact)

    async def compare(self, state: AgentState) -> AgentState:
        # The table is computed, never generated: these are the rubric's numbers.
        table = comparison_table(state.get("scores") or {}, load_roles())
        return {"comparison": table}

    async def plan(self, state: AgentState) -> AgentState:
        return await plan_improvements(state, client=self.client, config=self.config)

    async def human_review(self, state: AgentState) -> AgentState:
        if not self.review:
            return {"review": {"skipped": True}}
        decision = interrupt({
            "question": "Review the resumes and plan. Approve, or send edits.",
            "best_role": (state.get("comparison") or {}).get("best_role", ""),
            "scores": {k: v.total for k, v in (state.get("scores") or {}).items()},
            "plan_titles": [s.title for s in (state.get("plan") or [])],
            "needs_measure": [c.text for c in (state.get("claims") or [])
                              if c.status.value == "needs_measure"],
        })
        return {"review": decision if isinstance(decision, dict)
                else {"response": decision}}

    async def report(self, state: AgentState) -> AgentState:
        usage = dict(state.get("usage") or {})
        if self.client:
            report = self.client.usage_report()
            usage["groq"] = {
                "total": sum(v["total"] for v in report.values()),
                "calls": len(report),
                "by_node": report,
            }
            if self.client.cache:
                usage["cache_hit_rate"] = self.client.cache.hit_rate
        usage["elapsed_s"] = time.time() - self.started
        payload = dict(state)
        payload["usage"] = usage
        payload["contact"] = self.contact
        written = write_reports(payload, self.config.get("report.out_dir", "out"))
        log.info("wrote %s", ", ".join(written.values()))
        return {"usage": usage, "report_files": written}

    async def aclose(self) -> None:
        if self.backend is not None:
            await self.backend.aclose()
        if self.client is not None:
            await self.client.aclose()


def build_graph(runner: Runner, checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("connect_github", runner.connect)
    g.add_node("list_repos", runner.list_repos)
    g.add_node("select_repos", runner.select)
    g.add_node("fetch_facts", runner.fetch)
    g.add_node("safety_scan", runner.scan)
    g.add_node("summarize_repo", runner.summarize)
    g.add_node("verify_claims", runner.verify)
    g.add_node("build_resumes", runner.build)
    g.add_node("compare", runner.compare)
    g.add_node("plan_improvements", runner.plan)
    g.add_node("human_review", runner.human_review)
    g.add_node("report", runner.report)

    g.add_edge(START, "connect_github")
    for a, b in [
        ("connect_github", "list_repos"),
        ("list_repos", "select_repos"),
        ("select_repos", "fetch_facts"),
        ("fetch_facts", "safety_scan"),
        ("safety_scan", "summarize_repo"),
        ("summarize_repo", "verify_claims"),
        ("verify_claims", "build_resumes"),
        ("build_resumes", "compare"),
        ("compare", "plan_improvements"),
        ("plan_improvements", "human_review"),
        ("human_review", "report"),
        ("report", END),
    ]:
        g.add_edge(a, b)
    return g.compile(checkpointer=checkpointer)


async def make_checkpointer(path: str | Path | None = None) -> AsyncSqliteSaver:
    """Durable checkpoints for an async graph.

    It has to be the async saver: the sync ``SqliteSaver`` raises
    NotImplementedError the moment ``ainvoke`` asks it for a checkpoint, and
    the whole graph is async because every node does network I/O.
    """
    import aiosqlite

    p = Path(path or CHECKPOINT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(p))
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver
