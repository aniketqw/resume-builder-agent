"""Turn resume prose into structured fields, then throw away what it invented.

This is the only step in scoring that uses a language model, and it is confined
to *parsing*. The split matters: published evaluations put zero-shot resume
parsing at roughly **0.96 F1** (arXiv 2510.09722), while LLM *skill extraction*
is markedly weaker — GPT-4 scored **27.8** against a supervised model's
**62.6** on strict-span F1 (arXiv 2402.03832). Parsing is close to solved;
deciding what counts as a skill is not.

So the model proposes and this module disposes. Every extracted skill carries
the text it came from, and :func:`source_verify` drops any whose evidence is
not actually in the document. That matters because the characteristic failure
is not a miss but an inflation: a model reading "built a REST API" will happily
also report Docker, CI/CD and microservices, and each invented skill raises the
match score against a job description that asked for them.
"""
from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field

from ..groq_client import GroqClient
from ..prompts import render

log = logging.getLogger(__name__)

PROMPT = "extract.v1"


class ExtractedSkill(BaseModel):
    name: str = Field(description="Skill exactly as written in the resume")
    evidence: str = Field(description="Verbatim text the skill was taken from")


class Position(BaseModel):
    title: str = Field(description="Job title as written")
    company: str = Field(description="Employer as written")
    start_date: str = Field(description="Start date verbatim, or empty")
    end_date: str = Field(description="End date verbatim, or empty")


class Extraction(BaseModel):
    name: str = Field(description="Candidate name, or empty")
    headline: str = Field(description="Headline or current title, or empty")
    email: str = Field(description="Email, or empty")
    summary: str = Field(description="Summary section verbatim, or empty")
    skills: list[ExtractedSkill] = Field(description="Skills, each with evidence")
    work: list[Position] = Field(description="Positions held")
    education: list[str] = Field(description="Education lines, verbatim")


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def source_verify(extraction: Extraction, source: str) -> tuple[Extraction, list[str]]:
    """Drop anything whose evidence is not in the source document.

    Returns the cleaned extraction and the list of rejected skill names, which
    the caller shows rather than hides — a reader should be able to see what
    the model tried to add.
    """
    haystack = _normalise_whitespace(source)
    kept: list[ExtractedSkill] = []
    rejected: list[str] = []

    for skill in extraction.skills:
        evidence = _normalise_whitespace(skill.evidence)
        name = _normalise_whitespace(skill.name)
        # Accept if either the quoted evidence or the skill name itself is
        # present. A model often quotes a whole bullet; requiring an exact
        # evidence match alone would reject honest extractions over whitespace.
        if (evidence and evidence in haystack) or (name and name in haystack):
            kept.append(skill)
        else:
            rejected.append(skill.name)

    dates_dropped: list[str] = []
    for position in extraction.work:
        for field_name in ("start_date", "end_date"):
            value = getattr(position, field_name)
            if value and _normalise_whitespace(value) not in haystack:
                setattr(position, field_name, "")
                dates_dropped.append(f"{position.title}:{value}")

    if rejected:
        log.info("source verification dropped %d skill(s): %s",
                 len(rejected), ", ".join(rejected[:8]))
    if dates_dropped:
        log.info("source verification cleared %d invented date(s)", len(dates_dropped))

    cleaned = extraction.model_copy(update={"skills": kept})
    return cleaned, rejected


async def extract_resume(client: GroqClient, resume_text: str,
                         tier: str = "big") -> tuple[Extraction, list[str]]:
    """Extract structured fields, then verify them against the source."""
    messages, _version = render(PROMPT, resume=resume_text[:12000])
    extraction, _usage = await client.structured(
        messages, Extraction, node="extract_resume", tier=tier, max_tokens=2500)
    return source_verify(extraction, resume_text)


def skill_names(extraction: Extraction) -> list[str]:
    return [skill.name for skill in extraction.skills if skill.name.strip()]
