import pytest

from rba.nodes.build import (
    comparison_table,
    enforce_provenance,
    strip_citations,
    usable_claims,
)
from rba.scoring import load_roles
from rba.state import Claim, ClaimStatus, Resume, RoleScore

ROLES = load_roles()


def c(cid, text, status=ClaimStatus.VERIFIED, hint=""):
    return Claim(id=cid, repo="r", text=text, status=status, measure_hint=hint)


def test_rejected_claims_never_reach_the_generator():
    claims = [c("a", "Ships 34 tests"),
              c("b", "Reduced latency by 40%", ClaimStatus.NEEDS_MEASURE, "time it"),
              c("x", "Invented", ClaimStatus.REJECTED)]
    assert [u.id for u in usable_claims(claims)] == ["a", "b"]


def test_bullet_with_an_invented_number_is_dropped():
    """The generator can merge two claims into a total that was in neither."""
    claims = [c("a", "Ships 34 tests"), c("b", "Runs 1,200 simulations")]
    resume = Resume(role="balanced", used_claim_ids=["a", "b"], sections={
        "projects": ["Built a simulator running 1,234 scenarios",
                     "Shipped 34 tests"]})
    out, dropped = enforce_provenance(resume, claims)
    assert out.sections["projects"] == ["Shipped 34 tests"]
    assert len(dropped) == 1 and "1,234" in dropped[0]


def test_numbers_from_any_verified_claim_are_allowed():
    claims = [c("a", "Ships 34 tests"), c("b", "Runs 1,200 simulations")]
    resume = Resume(role="balanced", used_claim_ids=["a"], sections={
        "projects": ["Built a harness covering 1,200 runs across 34 tests"]})
    out, dropped = enforce_provenance(resume, claims)
    assert dropped == [] and len(out.sections["projects"]) == 1


def test_measure_tag_is_not_treated_as_an_invented_number():
    claims = [c("a", "Reduced latency", ClaimStatus.NEEDS_MEASURE, "time it")]
    resume = Resume(role="balanced", used_claim_ids=["a"], sections={
        "projects": ["Reduced latency by [MEASURE: time it before and after]"]})
    out, dropped = enforce_provenance(resume, claims)
    assert dropped == []
    assert "[MEASURE:" in out.sections["projects"][0]


def test_versions_in_bullets_are_not_flagged():
    claims = [c("a", "Built with Python")]
    resume = Resume(role="balanced", used_claim_ids=["a"], sections={
        "projects": ["Built a service with Python 3.11"]})
    _out, dropped = enforce_provenance(resume, claims)
    assert dropped == []


# -- claim-id citations ---------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("Built X for agents. (AiAgentDemo#0)", "Built X for agents."),
    ("Did Y (AiAgnoSample#0, AiAgnoSample#1)", "Did Y"),
    ("Did Z (AOA_Controller_ty#3, #4, #5).", "Did Z."),
    ("Agentic retrieval. (used_claim_ids: AiAgentDemo#0)", "Agentic retrieval."),
    ("Shipped 34 tests covering the API", "Shipped 34 tests covering the API"),
    # gpt-oss reaches for full-width brackets unprompted; a second live run
    # lost 8 bullets to this variant after the parenthesis form was fixed.
    ("Did X\u3010AiAgnoSample#0\u3011", "Did X"),
    ("Multi-threaded C\u3010AOA#0\u3011\u3010AOA#1\u3011", "Multi-threaded C"),
    ("Did Y [lecturelens#3]", "Did Y"),
])
def test_claim_id_citations_are_stripped(text, expected):
    assert strip_citations(text) == expected


def test_citation_digits_are_not_read_as_invented_numbers():
    """Regression: '(AiAgnoSample#0, #4, #5)' was parsed as the numbers 0, 4, 5
    and silently destroyed 19 legitimate bullets in a live run."""
    claims = [c("AiAgnoSample#4", "Validates each extracted fact against its chunk")]
    resume = Resume(role="balanced", used_claim_ids=["AiAgnoSample#4"], sections={
        "projects": ["Built validation scripts for each pipeline stage "
                     "(AiAgnoSample#4, #5)"]})
    out, dropped = enforce_provenance(resume, claims)
    assert dropped == [], "a claim id is not a claim about the world"
    assert out.sections["projects"] == [
        "Built validation scripts for each pipeline stage"]


def test_a_real_invented_number_still_dies_alongside_a_citation():
    claims = [c("demo#0", "Ships tests")]
    resume = Resume(role="balanced", used_claim_ids=["demo#0"], sections={
        "projects": ["Served 9,999 requests per second (demo#0)"]})
    _out, dropped = enforce_provenance(resume, claims)
    assert len(dropped) == 1 and "9,999" in dropped[0]


# -- comparison -----------------------------------------------------------
def test_comparison_ranks_roles_and_reports_the_spread():
    scores = {
        "balanced": RoleScore(role="balanced", total=55.0, missing_keywords=["docker"]),
        "ml_engineer": RoleScore(role="ml_engineer", total=71.5,
                                 missing_keywords=["pytorch", "training"]),
    }
    table = comparison_table(scores, ROLES)
    assert table["best_role"] == "ml_engineer"
    assert [r["role"] for r in table["rows"]] == ["ml_engineer", "balanced"]
    assert table["spread"] == 16.5


def test_comparison_includes_recoverable_points():
    scores = {"ml_engineer": RoleScore(role="ml_engineer", total=40.0,
                                       missing_keywords=["pytorch", "training"])}
    row = comparison_table(scores, ROLES)["rows"][0]
    assert row["potential_gain"] > 0
    assert row["top_gaps"] == ["pytorch", "training"]


def test_comparison_of_a_single_role_has_no_spread():
    scores = {"balanced": RoleScore(role="balanced", total=50.0)}
    assert comparison_table(scores, ROLES)["spread"] == 0.0
