"""Graph state and the core domain types.

The central idea is the **claim**: a single assertion about the author's work,
carrying the evidence that supports it. A claim with no surviving evidence never
reaches a resume — it becomes a `needs_measure` task instead.
"""
from __future__ import annotations

import operator
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class ClaimStatus(StrEnum):
    VERIFIED = "verified"           # every number and technology traced to evidence
    NEEDS_MEASURE = "needs_measure"  # plausible but unquantified -> [MEASURE: ...]
    REJECTED = "rejected"           # contradicted by, or absent from, the evidence


class Evidence(BaseModel):
    """A pointer into a repository that supports a claim."""
    path: str = Field(description="File path, or 'commit:<sha>' / 'api:<field>'")
    quote: str = Field(description="Verbatim excerpt from that location")


class Claim(BaseModel):
    id: str = ""
    repo: str = ""
    text: str
    evidence: list[Evidence] = Field(default_factory=list)
    status: ClaimStatus = ClaimStatus.VERIFIED
    reason: str = ""
    measure_hint: str = ""

    @property
    def is_usable(self) -> bool:
        return self.status is ClaimStatus.VERIFIED


class RepoRef(BaseModel):
    """What we know about a repo before spending any LLM tokens on it."""
    name: str
    full_name: str = ""
    description: str = ""
    language: str = ""
    languages: dict[str, int] = Field(default_factory=dict)
    is_fork: bool = False
    is_private: bool = False
    is_archived: bool = False
    stars: int = 0
    size_kb: int = 0
    pushed_at: str = ""
    created_at: str = ""
    default_branch: str = "main"
    topics: list[str] = Field(default_factory=list)
    # filled in by select_repos
    claimable: float = 0.0
    tier: str = ""
    selection_reason: str = ""


class RepoFacts(BaseModel):
    """Deterministically gathered. No model has touched any of this."""
    repo: str
    head_sha: str = ""
    readme: str = ""
    readme_path: str = ""
    tree: list[str] = Field(default_factory=list)
    file_count: int = 0
    languages: dict[str, int] = Field(default_factory=dict)
    commits: list[dict[str, str]] = Field(default_factory=list)
    commit_count: int = 0
    author_commits: int = 0
    has_tests: bool = False
    has_ci: bool = False
    dependencies: list[str] = Field(default_factory=list)
    key_files: dict[str, str] = Field(default_factory=dict)

    @property
    def author_share(self) -> float:
        return self.author_commits / self.commit_count if self.commit_count else 0.0


class SafetyFinding(BaseModel):
    repo: str
    path: str
    kind: str
    detail: str = ""
    severity: Literal["low", "medium", "high"] = "medium"


class RepoSummary(BaseModel):
    repo: str = ""
    one_line: str = ""
    problem: str = ""
    approach: str = ""
    tech: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    not_claimable_reason: str = ""


class RoleScore(BaseModel):
    role: str
    total: float = 0.0
    components: dict[str, float] = Field(default_factory=dict)
    matched_keywords: list[str] = Field(default_factory=list)
    missing_keywords: list[str] = Field(default_factory=list)


class Resume(BaseModel):
    role: str
    headline: str = ""
    summary: str = ""
    sections: dict[str, list[str]] = Field(default_factory=dict)
    used_claim_ids: list[str] = Field(default_factory=list)

    def as_text(self) -> str:
        parts = [self.headline, self.summary]
        for name, bullets in self.sections.items():
            parts.append(name)
            parts.extend(bullets)
        return "\n".join(p for p in parts if p)


class PlanStep(BaseModel):
    title: str
    gap: str = ""
    evidence_found: str = ""
    action: str = ""
    hours: float = 0.0
    measurable_output: str = ""
    ats_gain: float = 0.0
    role: str = ""
    repo: str = ""


def _merge_dicts(left: dict, right: dict) -> dict:
    out = dict(left)
    out.update(right)
    return out


class AgentState(TypedDict, total=False):
    """LangGraph state. Fan-out nodes use reducers so parallel writes merge."""
    username: str
    repos: list[RepoRef]
    selected: list[RepoRef]
    excluded: list[RepoRef]
    signals: dict[str, Any]
    facts: Annotated[dict[str, RepoFacts], _merge_dicts]
    summaries: Annotated[dict[str, RepoSummary], _merge_dicts]
    evidence_texts: Annotated[dict[str, str], _merge_dicts]
    claims: Annotated[list[Claim], operator.add]
    safety: Annotated[list[SafetyFinding], operator.add]
    scores: dict[str, RoleScore]
    resumes: dict[str, Resume]
    comparison: dict[str, Any]
    plan: list[PlanStep]
    review: dict[str, Any]
    usage: dict[str, Any]
    report_files: dict[str, str]
    contact: str
    errors: Annotated[list[str], operator.add]
