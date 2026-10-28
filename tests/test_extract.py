"""Extraction tests. The important one is that invention gets discarded.

Published numbers put zero-shot resume *parsing* near 0.96 F1, but LLM *skill
extraction* at 27.8 strict-span F1 against a supervised model's 62.6. The
characteristic failure is inflation: a model reading "built a REST API" also
reports Docker, CI/CD and microservices, each of which raises the match score
against a job description that asked for them.
"""
import json

import pytest

from fakes import FakeGroq, chat_response
from rba.ats.extract import (
    ExtractedSkill,
    Extraction,
    Position,
    extract_resume,
    skill_names,
    source_verify,
)
from rba.cache import Cache
from rba.config import Config
from rba.groq_client import GroqClient
from rba.limiter import Limiter

RESUME = """Jane Doe — Software Engineer
jane@example.com

Built a REST API in Python with FastAPI. Wrote the PostgreSQL schema.

Engineer, Acme Ltd, Jan 2021 - Present
"""


def _extraction(**kw) -> Extraction:
    base = {"name": "Jane Doe", "headline": "Software Engineer",
            "email": "jane@example.com", "summary": "", "skills": [],
            "work": [], "education": []}
    base.update(kw)
    return Extraction(**base)


# -- the guarantee --------------------------------------------------------
def test_invented_skills_are_discarded():
    """Docker and CI/CD appear nowhere in the resume."""
    extraction = _extraction(skills=[
        ExtractedSkill(name="Python", evidence="Built a REST API in Python"),
        ExtractedSkill(name="Docker", evidence="containerised the service"),
        ExtractedSkill(name="CI/CD", evidence="automated the pipeline"),
    ])
    cleaned, rejected = source_verify(extraction, RESUME)
    assert skill_names(cleaned) == ["Python"]
    assert set(rejected) == {"Docker", "CI/CD"}


def test_a_skill_named_in_the_text_survives_a_loose_quote():
    """Models often quote a whole bullet. Rejecting honest extractions over
    whitespace would be worse than the problem being solved."""
    extraction = _extraction(skills=[
        ExtractedSkill(name="PostgreSQL", evidence="wrote   the  PostgreSQL schema"),
    ])
    cleaned, rejected = source_verify(extraction, RESUME)
    assert skill_names(cleaned) == ["PostgreSQL"] and rejected == []


def test_rejected_skills_are_returned_not_silently_dropped():
    extraction = _extraction(skills=[
        ExtractedSkill(name="Kubernetes", evidence="ran things on Kubernetes")])
    _cleaned, rejected = source_verify(extraction, RESUME)
    assert rejected == ["Kubernetes"]


def test_invented_dates_are_cleared_rather_than_kept():
    """'2021 - Present' must not become a specific end date."""
    extraction = _extraction(work=[Position(
        title="Engineer", company="Acme Ltd",
        start_date="Jan 2021", end_date="Dec 2023")])
    cleaned, _ = source_verify(extraction, RESUME)
    assert cleaned.work[0].start_date == "Jan 2021"
    assert cleaned.work[0].end_date == "", "a date absent from the source must go"


def test_verification_is_case_insensitive():
    extraction = _extraction(skills=[
        ExtractedSkill(name="python", evidence="BUILT A REST API IN PYTHON")])
    cleaned, _ = source_verify(extraction, RESUME)
    assert len(cleaned.skills) == 1


def test_empty_extraction_is_handled():
    cleaned, rejected = source_verify(_extraction(), RESUME)
    assert cleaned.skills == [] and rejected == []


# -- the call -------------------------------------------------------------
async def test_extract_runs_the_model_then_verifies():
    payload = json.dumps({
        "name": "Jane Doe", "headline": "Software Engineer",
        "email": "jane@example.com", "summary": "",
        "skills": [
            {"name": "Python", "evidence": "Built a REST API in Python"},
            {"name": "Kubernetes", "evidence": "deployed to Kubernetes"},
        ],
        "work": [{"title": "Engineer", "company": "Acme Ltd",
                  "start_date": "Jan 2021", "end_date": ""}],
        "education": [],
    })
    fake = FakeGroq([chat_response(payload)])
    cfg = Config.load()
    client = GroqClient(config=cfg, limiter=Limiter(cfg), cache=Cache(enabled=False),
                        api_key="k", transport=fake.transport())
    extraction, rejected = await extract_resume(client, RESUME)
    await client.aclose()
    assert skill_names(extraction) == ["Python"]
    assert rejected == ["Kubernetes"]


async def test_extraction_asks_for_strict_json():
    payload = json.dumps({"name": "", "headline": "", "email": "", "summary": "",
                          "skills": [], "work": [], "education": []})
    fake = FakeGroq([chat_response(payload)])
    cfg = Config.load()
    client = GroqClient(config=cfg, limiter=Limiter(cfg), cache=Cache(enabled=False),
                        api_key="k", transport=fake.transport())
    await extract_resume(client, RESUME)
    await client.aclose()
    body = fake.requests[0]
    assert body["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize("prompt_rule", [
    "Copy, do not infer", "evidence", "verbatim",
])
def test_the_prompt_states_the_rules_it_is_judged_on(prompt_rule):
    from rba.prompts import load
    system, _user, _version = load("extract.v1")
    assert prompt_rule.lower() in system.lower()
