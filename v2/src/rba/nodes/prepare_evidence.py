"""Build a code-structure index from the source already fetched.

Deterministic, zero LLM calls -- `summarize_repo` is the only node allowed to
spend a Groq request. This node exists purely to turn `facts[repo].source_files`
into a `rba.repobrain.RepoBrain` the summarizer can cite real structure from,
and the verifier can use to locate a claim's quote to an actual line range.
"""
from __future__ import annotations

import logging

from ..repobrain import index
from ..state import AgentState, RepoFacts

log = logging.getLogger(__name__)


async def prepare_evidence(state: AgentState) -> AgentState:
    facts: dict[str, RepoFacts] = state.get("facts") or {}
    repo_graph = {}
    gaps: list[str] = []
    for repo, f in facts.items():
        brain = index(f.source_files)
        repo_graph[repo] = brain
        gaps.extend(f"{repo}:{g}" for g in brain.partial_source)

    if gaps:
        log.info("partial source for %d file(s) across %d repo(s)",
                 len(gaps), len(facts))
    log.info("indexed structure for %d repo(s): %d symbol(s), %d call edge(s)",
             len(repo_graph),
             sum(len(b.all_symbols()) for b in repo_graph.values()),
             sum(len(b.all_calls()) for b in repo_graph.values()))
    return {"repo_graph": repo_graph, "partial_source": gaps}
