"""Rephrase-only variants: the model proposes, the code disposes.

Two properties matter here and both are tested against the real
implementations rather than mocks of them: a variant that adds a fact never
reaches the user, and the gain shown next to a variant is a real `rescore`
of the document with that variant in it.

Dense embedding is switched off throughout (`use_dense=False`) so the suite
stays offline and fast; the arithmetic under test is identical either way,
because the same flag is passed to the baseline and to every variant.
"""
from __future__ import annotations

import json

import pytest

from fakes import FakeGroq, chat_response
from rba.cache import Cache
from rba.config import Config
from rba.groq_client import GroqClient
from rba.limiter import Limiter
from rba.nodes.variants import (
    Variant,
    generate_variants,
    projects,
    reject_reason,
    score_variants,
)
from rba.resume.attribution import rescore
from rba.resume.document import Bullet, Entry, ResumeDoc, Section

JD = ("Backend engineer. You will build Python services with FastAPI, run them "
      "on Kubernetes, and own the Postgres data model and caching layer.")
REQUIRED = ["Python", "FastAPI", "Kubernetes"]

EVIDENCE_API = ("README: the service is a FastAPI application backed by Postgres. "
                "Deployment manifests target Kubernetes. The test suite has 42 tests.")
EVIDENCE_ETL = ("README: a nightly batch job written in Python that reads CSV "
                "exports and writes them into Postgres.")


def _doc() -> ResumeDoc:
    """Two projects, three bullets -- so 'one call per project' is falsifiable."""
    api = Entry(title="Order API", subtitle="side project", dates="2026",
                tech=["Python", "FastAPI", "Postgres"], bullets=[
                    Bullet(text="Worked on a FastAPI service backed by Postgres.",
                           source_repo="order-api", evidence=EVIDENCE_API),
                    Bullet(text="Was responsible for the Kubernetes deployment manifests.",
                           source_repo="order-api", evidence=EVIDENCE_API),
                ])
    etl = Entry(title="Nightly ETL", dates="2025", tech=["Python"], bullets=[
        Bullet(text="Did a batch job in Python that loads CSV exports into Postgres.",
               source_repo="etl", evidence=EVIDENCE_ETL),
    ])
    return ResumeDoc(name="Jane Doe", headline="Backend Engineer",
                     sections=[Section(title="PROJECTS", entries=[api, etl])])


def _score(doc: ResumeDoc) -> float:
    raw, _ = rescore(doc, JD, required_skills=REQUIRED, use_dense=False)
    return raw


# -- the rule: rephrase only ----------------------------------------------
def test_a_variant_that_invents_a_number_is_rejected():
    doc = _doc()
    bullet = doc.sections[0].entries[0].bullets[0]
    poisoned = "Built a FastAPI service on Postgres serving 12000 requests per second."
    result = score_variants(doc, JD, {bullet.id: [poisoned]},
                            required_skills=REQUIRED, use_dense=False)
    variant = result.by_bullet[bullet.id][0]
    assert variant.verified is False
    assert "12000" in variant.rejected_reason
    assert result.best(bullet.id) is None


def test_a_number_already_in_the_evidence_survives():
    """42 is in the evidence, so quoting it is reporting, not inventing."""
    doc = _doc()
    bullet = doc.sections[0].entries[0].bullets[0]
    honest = "Built a FastAPI service on Postgres covered by 42 tests."
    assert reject_reason(honest, bullet, ("Python", "FastAPI", "Postgres")) == ""


def test_a_variant_that_only_rephrases_is_kept():
    doc = _doc()
    bullet = doc.sections[0].entries[0].bullets[0]
    rewrite = "Built a FastAPI service backed by Postgres."
    result = score_variants(doc, JD, {bullet.id: [rewrite]},
                            required_skills=REQUIRED, use_dense=False)
    variant = result.by_bullet[bullet.id][0]
    assert variant.verified is True
    assert variant.rejected_reason == ""
    assert result.best(bullet.id) is variant


@pytest.mark.parametrize("text,fragment", [
    ("Built a FastAPI service on Postgres, cutting latency by 40%.", "40"),
    ("Built a Redis-cached FastAPI service backed by Postgres.", "technology"),
    ("Built a FastAPI service on Postgres that improved throughput.", "outcome"),
    ("Mentored interns and ran the weekly design review.", "restate"),
    ("Worked on a FastAPI service backed by Postgres.", "identical"),
])
def test_every_invention_route_is_closed(text, fragment):
    bullet = _doc().sections[0].entries[0].bullets[0]
    reason = reject_reason(text, bullet, ("Python", "FastAPI", "Postgres"))
    assert fragment in reason, f"expected {fragment!r} in {reason!r}"


def test_a_technology_the_evidence_names_is_allowed():
    """Kubernetes is in the evidence bundle, so naming it is not invention."""
    doc = _doc()
    bullet = doc.sections[0].entries[0].bullets[1]
    rewrite = "Authored the Kubernetes deployment manifests for the service."
    assert reject_reason(rewrite, bullet, ("Python", "FastAPI")) == ""


# -- the number next to the variant ---------------------------------------
def test_reported_gain_equals_an_independent_rescore():
    doc = _doc()
    bullet = doc.sections[0].entries[0].bullets[0]
    rewrite = "Built a FastAPI service backed by Postgres."
    result = score_variants(doc, JD, {bullet.id: [rewrite]},
                            required_skills=REQUIRED, use_dense=False)
    variant = result.by_bullet[bullet.id][0]

    before = _score(doc)
    original = bullet.text
    bullet.text = rewrite
    after = _score(doc)
    bullet.text = original

    assert variant.gain == pytest.approx(round((after - before) * 100, 4), abs=1e-4)
    assert result.baseline == pytest.approx(round(before * 100, 4), abs=1e-4)


def test_scoring_restores_the_document_exactly():
    doc = _doc()
    bullet = doc.sections[0].entries[0].bullets[0]
    before = _score(doc)
    score_variants(doc, JD, {bullet.id: ["Built a FastAPI service on Postgres."]},
                   required_skills=REQUIRED, use_dense=False)
    assert bullet.text == "Worked on a FastAPI service backed by Postgres."
    assert _score(doc) == pytest.approx(before)


def test_surviving_variants_are_ranked_by_measured_gain():
    doc = _doc()
    bullet = doc.sections[0].entries[0].bullets[0]
    result = score_variants(doc, JD, {bullet.id: [
        "Built a FastAPI service backed by Postgres.",
        "Built and shipped a FastAPI service with a Postgres data model, "
        "deployed on Kubernetes.",
        "Delivered 12000 FastAPI endpoints.",          # rejected, must sort last
    ]}, required_skills=REQUIRED, use_dense=False)
    ordered = result.by_bullet[bullet.id]
    kept = [v for v in ordered if v.verified]
    assert len(kept) == 2
    assert [v.gain for v in kept] == sorted((v.gain for v in kept), reverse=True)
    assert ordered[-1].verified is False


def test_a_rejected_variant_is_never_scored():
    """Rejected means unscored: the 291 ms embedding is not spent on it."""
    doc = _doc()
    bullet = doc.sections[0].entries[0].bullets[0]
    result = score_variants(doc, JD, {bullet.id: ["Served 99.95% uptime for 5m users."]},
                            required_skills=REQUIRED, use_dense=False)
    assert result.by_bullet[bullet.id][0].gain == 0.0


def test_the_set_is_serialisable():
    doc = _doc()
    bullet = doc.sections[0].entries[0].bullets[0]
    result = score_variants(doc, JD, {bullet.id: ["Built a FastAPI service on Postgres."]},
                            required_skills=REQUIRED, use_dense=False)
    payload = json.loads(json.dumps(result.to_dict()))
    assert set(payload["variants"][bullet.id][0]) == {
        "bullet_id", "text", "gain", "verified", "rejected_reason"}


# -- the fan-out ----------------------------------------------------------
def test_projects_are_the_unit_of_fan_out():
    doc = _doc()
    groups = projects(doc)
    assert [len(p.bullets) for p in groups] == [2, 1]


def test_disabled_bullets_are_not_offered_variants():
    doc = _doc()
    doc.sections[0].entries[1].bullets[0].enabled = False
    assert [len(p.bullets) for p in projects(doc)] == [2]


def _client(fake: FakeGroq) -> GroqClient:
    cfg = Config.load()
    return GroqClient(config=cfg, limiter=Limiter(cfg), cache=Cache(enabled=False),
                      api_key="k", transport=fake.transport())


def _reply(*entries) -> dict:
    return chat_response(json.dumps({"bullets": [
        {"index": i, "variants": list(v)} for i, v in entries]}))


async def test_one_structured_call_per_project_not_per_bullet():
    doc = _doc()
    fake = FakeGroq([
        _reply((1, ["Built a FastAPI service backed by Postgres."]),
               (2, ["Authored the Kubernetes deployment manifests."])),
        _reply((1, ["Built a nightly Python job loading CSV exports into Postgres."])),
    ])
    client = _client(fake)
    result = await generate_variants(doc, JD, client=client, required_skills=REQUIRED,
                                     k=1, concurrency=1, use_dense=False)
    await client.aclose()

    # Three bullets, two projects: one call each, not one per bullet.
    assert len(fake.requests) == 2
    assert len(result.kept()) == 3
    assert all(v.verified for v in result.kept())


async def test_generation_poisoned_by_the_model_is_still_rejected():
    """The prompt forbids inventing numbers; the code is what enforces it."""
    doc = _doc()
    fake = FakeGroq([
        _reply((1, ["Built a FastAPI service handling 3000 orders a day."]),
               (2, ["Authored the Kubernetes deployment manifests."])),
        _reply((1, ["Built a nightly Python job loading CSV exports into Postgres."])),
    ])
    client = _client(fake)
    result = await generate_variants(doc, JD, client=client, required_skills=REQUIRED,
                                     k=1, concurrency=1, use_dense=False)
    await client.aclose()
    poisoned = doc.sections[0].entries[0].bullets[0].id
    assert result.best(poisoned) is None
    assert "3000" in result.by_bullet[poisoned][0].rejected_reason


async def test_a_failing_project_does_not_sink_the_others():
    doc = _doc()
    fake = FakeGroq([
        (400, {"error": {"message": "bad request"}}),
        _reply((1, ["Built a nightly Python job loading CSV exports into Postgres."])),
    ])
    client = _client(fake)
    result = await generate_variants(doc, JD, client=client, required_skills=REQUIRED,
                                     k=1, concurrency=1, use_dense=False)
    await client.aclose()
    assert len(result.kept()) == 1


async def test_the_project_facts_go_last_in_the_user_message():
    """Prefix stability is the whole reason the stable rules live in `# system`."""
    doc = _doc()
    fake = FakeGroq([_reply((1, ["Built a FastAPI service backed by Postgres."])),
                     _reply((1, ["Built a nightly Python job into Postgres."]))])
    client = _client(fake)
    await generate_variants(doc, JD, client=client, required_skills=REQUIRED,
                            k=1, concurrency=1, use_dense=False)
    await client.aclose()
    system, user = fake.requests[0]["messages"]
    assert system["role"] == "system" and "rephrase" in system["content"].lower()
    assert "Order API" in user["content"] and "FastAPI" in user["content"]
    assert EVIDENCE_API[:40] in user["content"]


@pytest.mark.parametrize("rule", ["never invent a number", "never invent a technology",
                                  "never invent an outcome", "never invent scale"])
def test_the_prompt_states_the_rules_the_code_enforces(rule):
    from rba.prompts import load

    system, _user, _version = load("variant.v1")
    assert rule in system.lower()


def test_variant_defaults_are_the_safe_ones():
    """An unexamined variant is not verified. Fail closed."""
    assert Variant(bullet_id="x", text="y").verified is False


# -- the guard, exercised against a real resume bullet ----------------------
# A live run produced 45 rewrites and rejected none of them. That is the right
# outcome only if the guard actually fires, so these poison a genuine bullet
# from the fixture resume along every route an invention could take.
def _real_bullet():
    import pathlib

    from rba.resume.ingest import parse_bits

    fixture = pathlib.Path(__file__).parent.parent / "fixtures" / "aniket_bits.txt"
    doc = parse_bits(fixture.read_text())
    return doc.by_id(next(b.id for b in doc.all_bullets()
                          if "LangGraph agent (12-node" in b.text))


@pytest.mark.parametrize("route, text", [
    ("percentage", "Built a LangGraph agent that cut review time by 40%."),
    ("scale", "Built a LangGraph agent serving 2 million requests per day."),
    ("technology", "Built a LangGraph agent on Kubernetes with Terraform and Kafka."),
    ("outcome", "Built a LangGraph agent that doubled throughput versus the baseline."),
    ("unrelated", "Led a team of fifteen engineers across three continents."),
])
def test_every_invention_route_is_closed_on_a_real_bullet(route, text):
    assert reject_reason(text, _real_bullet()), f"{route} should be rejected"


def test_an_honest_rephrase_of_a_real_bullet_survives():
    kept = ("Engineered a 12-node LangGraph agent with SQLite checkpointing and "
            "a human-review interrupt, reading GitHub through the read-only MCP "
            "server.")
    assert reject_reason(kept, _real_bullet()) == ""
