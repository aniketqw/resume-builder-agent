"""The LangGraph wiring.

One graph, two entry points. The CLI runs every node and writes a report; the
browser import runs the identical graph and stops at ``build_document`` --
``full_run`` in the input state is the only thing that differs, and it drives
a single conditional edge. There is no second, hand-sequenced copy of this
pipeline anywhere else in the codebase.

Node order matters for one non-obvious reason: Groq cannot combine tool use
with structured outputs in a single request, so every node that touches MCP
tools (`fetch_facts`) is strictly separate from every node that asks for strict
JSON (`summarize_repo`, `build_resumes`, `plan_improvements`).

No checkpointer and no human-review interrupt: both were judged more than
this project needs right now. Removing the checkpointer also removes the
requirement that everything in `AgentState` be msgpack-serialisable, which is
what makes it safe to carry a `RepoBrain` (plain dataclasses) and a
`ResumeDoc` through state.
"""
from __future__ import annotations

import logging
import os
import time

from langgraph.graph import END, START, StateGraph

from .config import Config, enable_tracing
from .github_tools import make_backend
from .groq_client import GroqClient
from .jev_client import JevClient
from .nodes.build import build_resumes, comparison_table
from .nodes.fetch import fetch_facts
from .nodes.plan import plan_improvements
from .nodes.prepare_evidence import prepare_evidence
from .nodes.safety import safety_scan
from .nodes.select import select_repos
from .nodes.summarize import summarize_repos
from .nodes.verify import verify_claims
from .ollama_client import OllamaClient
from .report import write_reports
from .resume.document import from_summaries
from .scoring import load_roles
from .state import AgentState

log = logging.getLogger(__name__)


class Runner:
    """Holds the clients the nodes need, so nodes stay plain functions.

    Every credential defaults to empty/``None``, and every client below
    treats that as "read it from the environment" -- so a bare ``Runner()``
    behaves exactly as the CLI always has, and the API supplies per-request
    credentials by passing them here instead.
    """

    def __init__(self, config: Config | None = None, contact: str = "",
                 github_token: str = "", groq_key: str = "", jev_key: str = "",
                 limiter=None, cache=None, max_repos: int = 0):
        # Enabling tracing here, before any node runs, is what lets
        # `connect_github` itself show up in the trace tree -- GroqClient
        # used to be the only thing that called this, which meant the very
        # first node's own run was never covered because nothing had set
        # LANGCHAIN_TRACING_V2 yet by the time it executed.
        enable_tracing()
        self.config = config or Config.load()
        self.contact = contact or self.config.get("report.contact", "") or ""
        self.github_token = github_token
        self.groq_key = groq_key
        self.jev_key = jev_key
        self.limiter = limiter
        self.cache = cache
        if max_repos:
            self.config.raw.setdefault("selection", {})["max_repos"] = max_repos
        self.backend = None
        self.client: GroqClient | OllamaClient | None = None
        self.started = time.time()

    # -- nodes ------------------------------------------------------------
    async def connect(self, state: AgentState) -> AgentState:
        self.backend = await make_backend(self.config, token=self.github_token or None)
        # "ollama" trades Groq's 8,000 TPM free-tier rate limit (a repeated,
        # measured bottleneck -- see trace.md Iteration 15) for a self-hosted
        # model with no quota at all. Both clients expose the same
        # `structured()` shape, so nothing downstream of this branch knows
        # or cares which one it got.
        if os.environ.get("LLM_PROVIDER", "groq").strip().lower() == "ollama":
            self.client = OllamaClient(self.config, cache=self.cache)
        else:
            self.client = GroqClient(self.config, api_key=self.groq_key or None,
                                     limiter=self.limiter, cache=self.cache)
        username = state.get("username") or ""
        log.info("connected via %s for user %s",
                 type(self.backend).__name__, username)
        return {"username": username}

    async def list_repos(self, state: AgentState) -> AgentState:
        repos = await self.backend.list_repos(state["username"])
        public = [r for r in repos if not r.is_private]
        log.info("found %d repositories (%d public)", len(repos), len(public))
        return {"repos": public, "repos_found": len(repos)}

    async def select(self, state: AgentState) -> AgentState:
        jev = JevClient(self.config, api_key=self.jev_key or None)
        try:
            return await select_repos(state, config=self.config,
                                      backend=self.backend, jev=jev)
        finally:
            await jev.aclose()

    async def fetch(self, state: AgentState) -> AgentState:
        return await fetch_facts(state, backend=self.backend, config=self.config)

    async def prepare(self, state: AgentState) -> AgentState:
        return await prepare_evidence(state)

    async def scan(self, state: AgentState) -> AgentState:
        return await safety_scan(state)

    async def summarize(self, state: AgentState) -> AgentState:
        return await summarize_repos(state, client=self.client, config=self.config)

    async def verify(self, state: AgentState) -> AgentState:
        return await verify_claims(state)

    async def document(self, state: AgentState) -> AgentState:
        """The CLI/browser join point. Deterministic -- no LLM call -- and
        the last node the browser import needs: the thing the user actually
        sees is an editable resume with evidence-backed bullets, not a
        role-scored comparison table."""
        doc = from_summaries(state.get("summaries") or {},
                             name=state.get("username", ""), headline="",
                             contact=f"github.com/{state.get('username', '')}",
                             facts=state.get("facts"))
        return {"document": doc}

    async def build(self, state: AgentState) -> AgentState:
        return await build_resumes(state, client=self.client, config=self.config,
                                   contact=self.contact)

    async def compare(self, state: AgentState) -> AgentState:
        # Role-fit advice from the hand-written rubric -- NOT the ATS score.
        # The real score is the deterministic three-channel scorer in
        # rba.ats, computed against a job description the user actually
        # pasted (see routes/score.py). This table has no JD to score
        # against, so it must never be presented as the same number.
        table = comparison_table(state.get("scores") or {}, load_roles())
        return {"comparison": table}

    async def plan(self, state: AgentState) -> AgentState:
        return await plan_improvements(state, client=self.client, config=self.config)

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


def _after_document(state: AgentState) -> str:
    """The one branch in the graph: the CLI asks for `full_run` and gets the
    role resumes, the comparison, the plan and a report. The browser import
    omits it and stops at the document it just built."""
    return "build_resumes" if state.get("full_run") else END


def build_graph(runner: Runner):
    g = StateGraph(AgentState)
    g.add_node("connect_github", runner.connect)
    g.add_node("list_repos", runner.list_repos)
    g.add_node("select_repos", runner.select)
    g.add_node("fetch_facts", runner.fetch)
    g.add_node("prepare_evidence", runner.prepare)
    g.add_node("safety_scan", runner.scan)
    g.add_node("summarize_repo", runner.summarize)
    g.add_node("verify_claims", runner.verify)
    g.add_node("build_document", runner.document)
    g.add_node("build_resumes", runner.build)
    g.add_node("compare", runner.compare)
    g.add_node("plan_improvements", runner.plan)
    g.add_node("report", runner.report)

    g.add_edge(START, "connect_github")
    for a, b in [
        ("connect_github", "list_repos"),
        ("list_repos", "select_repos"),
        ("select_repos", "fetch_facts"),
        ("fetch_facts", "prepare_evidence"),
        ("prepare_evidence", "safety_scan"),
        ("safety_scan", "summarize_repo"),
        ("summarize_repo", "verify_claims"),
        ("verify_claims", "build_document"),
        ("build_resumes", "compare"),
        ("compare", "plan_improvements"),
        ("plan_improvements", "report"),
        ("report", END),
    ]:
        g.add_edge(a, b)
    g.add_conditional_edges("build_document", _after_document,
                            ["build_resumes", END])
    return g.compile()
