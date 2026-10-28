"""Improvement planning.

The model proposes actions. It never states the ATS gain — that comes from
:func:`rba.scoring.max_gain`, the same arithmetic that produced the score, so
a plan step and a score movement cannot disagree.

Every step is also checked for repository provenance: a step referencing a
repository the candidate does not have is dropped, because the whole value of
the plan is that it starts from real work.
"""
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from ..config import Config
from ..groq_client import GroqClient
from ..prompts import render
from ..scoring import load_roles
from ..state import AgentState, Claim, ClaimStatus, PlanStep, RoleScore

log = logging.getLogger(__name__)

PROMPT = "plan.v1"


class _Step(BaseModel):
    title: str = Field(description="Short imperative title")
    gap: str = Field(description="What the resume cannot currently evidence")
    evidence_found: str = Field(description="Existing work this builds on, or 'none'")
    action: str = Field(description="Concrete task, naming the repository to change")
    hours: float = Field(description="Realistic estimate between 1 and 40")
    measurable_output: str = Field(description="The artefact or number proving it is done")
    repo: str = Field(description="Repository the step touches, or empty for a new one")


class _Plan(BaseModel):
    steps: list[_Step] = Field(description="Improvement steps, highest impact first")


def _claims_digest(claims: list[Claim], limit: int = 18) -> str:
    usable = [c for c in claims if c.status is ClaimStatus.VERIFIED]
    return "\n".join(f"- [{c.repo}] {c.text}" for c in usable[:limit]) or "- (none)"


def attach_ats_gain(steps: list[PlanStep], score: RoleScore, roles=None) -> list[PlanStep]:
    """Distribute the recoverable keyword points across the steps.

    The total is the rubric's own :func:`max_gain`, so the plan cannot promise
    points the scorer would not actually award.
    """
    from ..scoring import max_gain

    roles = roles or load_roles()
    role = roles[score.role]
    if not steps:
        return steps
    budget = max_gain(score.role, score.missing_keywords, roles, limit=len(steps) * 2)
    # Weight each step by the keywords it plausibly covers.
    weights = []
    for step in steps:
        blob = f"{step.title} {step.gap} {step.action} {step.measurable_output}".lower()
        w = sum(role.keywords.get(k, 0.0) for k in score.missing_keywords if k in blob)
        weights.append(w)
    total = sum(weights)
    for step, w in zip(steps, weights, strict=False):
        share = (w / total) if total else (1.0 / len(steps))
        step.ats_gain = round(budget * share, 2)
    return steps


async def plan_one(client: GroqClient, score: RoleScore, claims: list[Claim],
                   max_steps: int = 5, roles=None) -> list[PlanStep]:
    roles = roles or load_roles()
    role = roles[score.role]
    known_repos = {c.repo for c in claims if c.repo}
    messages, _v = render(
        PROMPT,
        role_title=role.title,
        total=score.total,
        components=", ".join(f"{k} {v}" for k, v in score.components.items()),
        missing=", ".join(score.missing_keywords[:10]) or "(none)",
        claims=_claims_digest(claims),
        max_steps=max_steps,
    )
    obj, _usage = await client.structured(messages, _Plan, node="plan_improvements",
                                          max_tokens=2000)

    steps: list[PlanStep] = []
    for s in obj.steps[:max_steps]:
        if s.repo and s.repo not in known_repos:
            # A plan that starts from a repository you do not have is fiction.
            log.debug("plan step names unknown repo %r; treating as new work", s.repo)
            s.repo = ""
            s.evidence_found = "none"
        steps.append(PlanStep(
            title=s.title, gap=s.gap, evidence_found=s.evidence_found,
            action=s.action, hours=max(1.0, min(40.0, s.hours)),
            measurable_output=s.measurable_output, role=score.role, repo=s.repo))
    return attach_ats_gain(steps, score, roles)


async def plan_improvements(state: AgentState, *, client: GroqClient,
                            config: Config | None = None,
                            max_steps: int | None = None) -> AgentState:
    cfg = config or Config.load()
    max_steps = max_steps or cfg.get("plan.max_steps_per_role", 5)
    scores: dict[str, RoleScore] = state.get("scores") or {}
    claims = state.get("claims") or []
    comparison = state.get("comparison") or {}
    focus = comparison.get("best_role") or (next(iter(scores), ""))

    plan: list[PlanStep] = []
    errors: list[str] = []
    # Plan for the best-fit role plus the weakest, which is where the biggest
    # recoverable gap usually sits.
    targets = [k for k in (focus, _weakest(scores)) if k]
    for key in dict.fromkeys(targets):
        try:
            plan += await plan_one(client, scores[key], claims, max_steps)
        except Exception as exc:
            log.warning("planning failed for %s: %s", key, exc)
            errors.append(f"plan_improvements[{key}]: {exc}")

    plan.sort(key=lambda s: (-s.ats_gain, s.hours))
    log.info("planned %d steps across %d roles", len(plan), len(targets))
    return {"plan": plan, "errors": errors}


def _weakest(scores: dict[str, RoleScore]) -> str:
    if not scores:
        return ""
    return min(scores.items(), key=lambda kv: kv[1].total)[0]
