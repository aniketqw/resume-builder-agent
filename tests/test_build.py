from rba.nodes.build import comparison_table, enforce_provenance, usable_claims
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
