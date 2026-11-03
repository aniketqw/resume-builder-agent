"""Resume generation from verified claims, and comparison across roles.

The generator sees only claims that already survived verification. It cannot
reach the repositories, so it has nothing to invent *from*: the worst it can do
is misuse a claim, and :func:`enforce_provenance` catches that by re-checking
every bullet's numbers against the claims it says it used.
"""
from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from ..config import Config
from ..groq_client import GroqClient
from ..prompts import render
from ..scoring import load_roles, max_gain, score_resume
from ..state import AgentState, Claim, ClaimStatus, Resume, RoleScore
from .verify import numbers_in

log = logging.getLogger(__name__)

PROMPT = "resume.v2"
SECTIONS = ["summary", "experience", "projects", "skills", "education"]
MEASURE_TAG = re.compile(r"\[MEASURE:[^\]]*\]")

# Claim-id citations the generator appends to bullets, e.g.
# "(AiAgnoSample#0, #1, #2)" or "(used_claim_ids: AiAgentDemo#0)".
# These have to be stripped before the number check: the digits in an id are
# not claims about the world, and reading them as such silently destroyed 19
# legitimate bullets in a live run. They are also removed from the output,
# since an internal id has no business being on a resume.
# Claim-id citations the generator appends to bullets, e.g.
# "(AiAgnoSample#0, #1, #2)", "[lecturelens#3]" or the full-width
# "【AOA_Controller_ty#0】" that gpt-oss reaches for unprompted.
# These must be stripped before the number check: the digits in an id are not
# claims about the world, and reading them as such silently destroyed 19
# legitimate bullets in one live run and 8 more in the next, once with each
# bracket style. They are also removed from the output, since an internal id
# has no business being on a resume.
_ID_LIST = (r"(?:used_claim_ids:\s*)?"
            r"[A-Za-z0-9_.\-]*#\d+"
            r"(?:\s*,\s*(?:[A-Za-z0-9_.\-]*)?#\d+)*")
CITATION = re.compile(
    rf"\s*(?:\({_ID_LIST}\)|\[{_ID_LIST}\]|【{_ID_LIST}】)")


def strip_citations(text: str) -> str:
    """Remove trailing claim-id references from a bullet."""
    return CITATION.sub("", text).strip()


class _Resume(BaseModel):
    headline: str = Field(description="One line: role and focus")
    summary: str = Field(description="Two or three factual sentences")
    projects: list[str] = Field(description="Bullets drawn from the claims")
    skills: list[str] = Field(description="Technologies evidenced by the claims")
    used_claim_ids: list[str] = Field(description="Ids of every claim used")


def claims_block(claims: list[Claim], limit: int = 24) -> str:
    """Render claims for the prompt, flagging the ones that lack a measurement."""
    lines = []
    for c in claims[:limit]:
        tag = "" if c.status is ClaimStatus.VERIFIED else f" [MEASURE: {c.measure_hint}]"
        lines.append(f"- id={c.id} repo={c.repo}: {c.text}{tag}")
    return "\n".join(lines)


def usable_claims(claims: list[Claim]) -> list[Claim]:
    """Verified claims, plus needs-measure ones which carry a MEASURE tag."""
    return [c for c in claims
            if c.status in (ClaimStatus.VERIFIED, ClaimStatus.NEEDS_MEASURE)]


def enforce_provenance(resume: Resume, claims: list[Claim]) -> tuple[Resume, list[str]]:
    """Drop any bullet asserting a number its source claims do not contain.

    The generator is a second chance to invent: it can merge two claims and
    produce a total that appeared in neither. This re-runs the number check at
    the bullet level, against the union of the claims the resume says it used.
    """
    by_id = {c.id: c for c in claims}
    allowed_numbers: set[str] = set()
    for cid in resume.used_claim_ids:
        c = by_id.get(cid)
        if c:
            allowed_numbers.update(numbers_in(c.text))
    for c in claims:
        allowed_numbers.update(numbers_in(c.text))

    dropped: list[str] = []
    for section, bullets in list(resume.sections.items()):
        kept = []
        for raw in bullets:
            bullet = strip_citations(raw)
            checkable = MEASURE_TAG.sub(" ", bullet)
            invented = [n for n in numbers_in(checkable) if n not in allowed_numbers]
            if invented:
                dropped.append(f"{bullet}  (invented: {', '.join(invented)})")
                continue
            kept.append(bullet)
        resume.sections[section] = kept
    resume.summary = strip_citations(resume.summary)
    return resume, dropped


async def build_one(client: GroqClient, role_key: str, claims: list[Claim],
                    roles=None) -> tuple[Resume, list[str]]:
    roles = roles or load_roles()
    role = roles[role_key]
    usable = usable_claims(claims)
    messages, _version = render(
        PROMPT,
        role_title=role.title,
        role_headline=role.headline,
        keywords=", ".join(sorted(role.keywords, key=lambda k: -role.keywords[k])[:15]),
        claims=claims_block(usable),
        sections=", ".join(SECTIONS),
    )
    # A resume payload is small; the generous global max_tokens is sized for
    # summarization. Keeping this tight matters because each reserved token
    # comes out of the same 8,000-per-minute budget, and an oversized reserve
    # throttles the whole stage.
    obj, _usage = await client.structured(messages, _Resume, node="build_resumes",
                                          max_tokens=1500)

    resume = Resume(
        role=role_key,
        headline=obj.headline,
        summary=obj.summary,
        sections={"summary": [obj.summary], "projects": obj.projects,
                  "skills": obj.skills},
        used_claim_ids=obj.used_claim_ids,
    )
    return enforce_provenance(resume, usable)


async def build_resumes(state: AgentState, *, client: GroqClient,
                        config: Config | None = None,
                        contact: str = "") -> AgentState:
    cfg = config or Config.load()
    roles = load_roles()
    claims = state.get("claims") or []
    role_keys = cfg.get("report.roles") or list(roles)

    resumes: dict[str, Resume] = {}
    scores: dict[str, RoleScore] = {}
    errors: list[str] = []
    # Serial: five role calls at ~2k tokens each would otherwise collide with
    # the 8,000 tokens-per-minute ceiling all at once.
    for key in role_keys:
        try:
            resume, dropped = await build_one(client, key, claims, roles)
            if dropped:
                log.warning("role %s: dropped %d bullet(s) with invented numbers: %s",
                            key, len(dropped), dropped[0][:90])
                errors += [f"build_resumes[{key}]: dropped bullet — {d}" for d in dropped]
            resumes[key] = resume
            scores[key] = score_resume(resume, key, roles, contact)
        except Exception as exc:
            log.warning("resume build failed for %s: %s", key, exc)
            errors.append(f"build_resumes[{key}]: {exc}")
    log.info("built %d resumes; scores: %s", len(resumes),
             {k: v.total for k, v in scores.items()})
    return {"resumes": resumes, "scores": scores, "errors": errors}


def comparison_table(scores: dict[str, RoleScore], roles=None) -> dict:
    """Built in Python so the numbers are the rubric's, not a model's."""
    roles = roles or load_roles()
    rows = []
    for key, s in sorted(scores.items(), key=lambda kv: -kv[1].total):
        rows.append({
            "role": key,
            "title": roles[key].title,
            "total": s.total,
            "components": s.components,
            "top_gaps": s.missing_keywords[:5],
            "potential_gain": max_gain(key, s.missing_keywords, roles, limit=5),
        })
    best = rows[0] if rows else None
    return {
        "rows": rows,
        "best_role": best["role"] if best else "",
        "spread": round(rows[0]["total"] - rows[-1]["total"], 2) if len(rows) > 1 else 0.0,
    }
