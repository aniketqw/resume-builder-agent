"""Turn a fact bundle into evidence-linked claims.

One repo per call, strict JSON out. No tools are used here — Groq cannot
combine structured outputs with tool calling, and all the fetching already
happened in :mod:`rba.nodes.fetch`.

The evidence bundle handed to the model is built here rather than passed
through raw, for two reasons: the prompt budget is small (8,000 tokens per
minute across the whole run), and the verifier later needs the exact same text
the model saw in order to decide whether a claim is supported.
"""
from __future__ import annotations

import asyncio
import json
import logging

from pydantic import BaseModel, Field

from ..cache import make_key
from ..config import Config
from ..groq_client import GroqClient
from ..prompts import render
from ..state import AgentState, Claim, ClaimStatus, Evidence, RepoFacts, RepoSummary

log = logging.getLogger(__name__)

PROMPT = "summarize.v2"


class _Evidence(BaseModel):
    path: str = Field(description="File path, or commit:<sha>")
    quote: str = Field(description="Exact text copied from the evidence bundle")


class _Claim(BaseModel):
    text: str = Field(description="What the author built or did, in one sentence")
    evidence: list[_Evidence] = Field(
        description="One supporting excerpt, quoted exactly from the evidence")


class _Summary(BaseModel):
    one_line: str = Field(description="What this repository is, in one sentence")
    problem: str = Field(description="The problem it addresses, per the evidence")
    approach: str = Field(description="How it works, per the evidence")
    tech: list[str] = Field(description="Technologies named in the evidence")
    claims: list[_Claim] = Field(
        description="At most 6 resume-worthy, evidence-backed claims")
    not_claimable_reason: str = Field(
        description="If the evidence is too thin to summarize, why. Else empty.")


def build_evidence(facts: RepoFacts, max_chars: int = 7000) -> str:
    """A compact JSON bundle. This exact text is what the verifier checks against."""
    bundle: dict[str, object] = {"repo": facts.repo}
    if facts.readme:
        bundle["readme"] = {"path": facts.readme_path or "README.md",
                            "text": facts.readme}
    if facts.tree:
        bundle["files"] = facts.tree[:80]
    if facts.commits:
        bundle["commits"] = [{"sha": c.get("sha", ""), "message": c.get("message", "")}
                             for c in facts.commits[:20]]
    if facts.dependencies:
        bundle["dependencies"] = facts.dependencies
    if facts.languages:
        bundle["languages"] = facts.languages
    bundle["signals"] = {
        "file_count": facts.file_count,
        "commits_sampled": facts.commit_count,
        "commits_by_author": facts.author_commits,
        "has_test_suite": facts.has_tests,
        "has_ci_config": facts.has_ci,
    }
    text = json.dumps(bundle, indent=1, ensure_ascii=False)
    if len(text) > max_chars:
        # Drop the least information-dense parts first, never the README.
        bundle.pop("files", None)
        text = json.dumps(bundle, indent=1, ensure_ascii=False)
        if len(text) > max_chars and isinstance(bundle.get("readme"), dict):
            bundle["readme"]["text"] = bundle["readme"]["text"][: max_chars // 2]
            text = json.dumps(bundle, indent=1, ensure_ascii=False)
    return text


async def summarize_one(client: GroqClient, facts: RepoFacts,
                        tier: str = "small",
                        max_evidence_chars: int = 7000) -> tuple[RepoSummary, str]:
    """Summarize one repo. Returns the summary and the evidence text used."""
    evidence = build_evidence(facts, max_evidence_chars)
    messages, version = render(PROMPT, repo=facts.repo, evidence=evidence)
    key = make_key(facts.repo, facts.head_sha, PROMPT, version)
    obj, _usage = await client.structured(
        messages, _Summary, node="summarize_repo", tier=tier, cache_key=key)

    summary = RepoSummary(
        repo=facts.repo,
        one_line=obj.one_line,
        problem=obj.problem,
        approach=obj.approach,
        tech=obj.tech,
        not_claimable_reason=obj.not_claimable_reason,
        claims=[
            Claim(
                id=f"{facts.repo}#{i}",
                repo=facts.repo,
                text=c.text,
                evidence=[Evidence(path=e.path, quote=e.quote) for e in c.evidence],
                status=ClaimStatus.VERIFIED,  # provisional; verify_claims decides
            )
            for i, c in enumerate(obj.claims)
        ],
    )
    return summary, evidence


async def summarize_repos(state: AgentState, *, client: GroqClient,
                          config: Config | None = None,
                          concurrency: int = 2) -> AgentState:
    """Fan out over selected repos, bounded by the token budget rather than CPU."""
    cfg = config or Config.load()
    # ~4 chars per token: this keeps one call near 2k input tokens, which is
    # what the measured 8,000 TPM ceiling allows about four of per minute.
    max_evidence = cfg.get("summarize.max_evidence_chars", 7000)
    facts: dict[str, RepoFacts] = state.get("facts") or {}
    tiers = {r.name: (r.tier or "small") for r in state.get("selected", [])}
    sem = asyncio.Semaphore(concurrency)

    async def one(name: str, f: RepoFacts):
        async with sem:
            try:
                summary, evidence = await summarize_one(
                    client, f, tiers.get(name, "small"), max_evidence)
                return name, summary, evidence, None
            except Exception as exc:
                log.warning("summarize failed for %s: %s", name, exc)
                return name, None, "", str(exc)

    results = await asyncio.gather(*(one(n, f) for n, f in facts.items()))
    summaries: dict[str, RepoSummary] = {}
    evidence_texts: dict[str, str] = {}
    errors: list[str] = []
    for name, summary, evidence, err in results:
        if summary is not None:
            summaries[name] = summary
            evidence_texts[name] = evidence
        if err:
            errors.append(f"summarize_repo[{name}]: {err}")

    log.info("summarized %d repos, %d raw claims", len(summaries),
             sum(len(s.claims) for s in summaries.values()))
    return {"summaries": summaries, "evidence_texts": evidence_texts, "errors": errors}
