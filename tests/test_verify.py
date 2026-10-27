"""The anti-invention tests. If these pass, a fabricated number cannot reach a resume."""
import pytest

from rba.nodes.verify import (
    filter_tech,
    is_impact_claim,
    numbers_in,
    quote_supported,
    verify_claim,
    verify_claims,
)
from rba.state import Claim, ClaimStatus, Evidence, RepoSummary

EVIDENCE = """{
 "repo": "live-debugging-partner",
 "readme": {"path": "README.md", "text":
   "# Live Debugging Partner\\n34 tests, about 2 s\\n1,200 runs, about 15 s\\nFour arms: text, text_interrupt, basic_voice, interactive_voice.\\nBuilt with Python 3.11 and unittest."},
 "dependencies": ["pytest", "httpx"],
 "signals": {"file_count": 37, "commits_sampled": 20, "has_test_suite": true}
}"""


def claim(text, path="README.md", quote="34 tests, about 2 s"):
    return Claim(id="c1", repo="r", text=text,
                 evidence=[Evidence(path=path, quote=quote)])


# -- the core guarantee ---------------------------------------------------
def test_invented_number_is_never_verified():
    """The headline guarantee: a fabricated metric cannot become a resume bullet."""
    c = verify_claim(claim("Reduced latency by 40% across the pipeline"), EVIDENCE)
    assert c.status is ClaimStatus.NEEDS_MEASURE
    assert "40" in c.reason
    assert c.measure_hint, "a rejected number must become an actionable task"


def test_real_number_from_the_evidence_is_verified():
    c = verify_claim(claim("Ships 34 tests that run in about 2 s"), EVIDENCE)
    assert c.status is ClaimStatus.VERIFIED


def test_formatted_thousands_are_matched():
    c = verify_claim(claim("Executes 1,200 simulated runs", quote="1,200 runs, about 15 s"),
                     EVIDENCE)
    assert c.status is ClaimStatus.VERIFIED


def test_a_number_that_is_merely_plausible_is_still_rejected():
    """35 is close to the real 34. Closeness is not evidence."""
    c = verify_claim(claim("Ships 35 tests"), EVIDENCE)
    assert c.status is ClaimStatus.NEEDS_MEASURE


# -- impact language ------------------------------------------------------
def test_unquantified_effect_claim_needs_measurement():
    c = verify_claim(claim("Improved developer productivity"), EVIDENCE)
    assert c.status is ClaimStatus.NEEDS_MEASURE
    assert "without a measured quantity" in c.reason


@pytest.mark.parametrize("text", [
    "MongoDB is used for saving and searching documents",
    "Implements an execution engine that routes orders",
    "Tracks the portfolio with real-time updates",
    "Built a parser for the configuration format",
])
def test_ordinary_descriptions_are_not_treated_as_performance_claims(text):
    """Regression: 'saving' and 'performance' as bare nouns caused false flags."""
    assert not is_impact_claim(text)
    assert verify_claim(claim(text), EVIDENCE).status is ClaimStatus.VERIFIED


@pytest.mark.parametrize("text", [
    "Reduced latency by 40%", "Improved accuracy from 0.81 to 0.92",
    "Made the build 3x faster",
])
def test_comparative_claims_are_recognised(text):
    assert is_impact_claim(text)


# -- quotes ---------------------------------------------------------------
def test_fabricated_quote_rejects_the_claim():
    # No numbers here, so the quote check is what decides the outcome.
    c = verify_claim(claim("Supports interactive voice input",
                           quote="This sentence appears nowhere in the bundle"),
                     EVIDENCE)
    assert c.status is ClaimStatus.REJECTED
    assert "absent from the evidence" in c.reason


def test_empty_quote_is_rejected():
    c = verify_claim(claim("The repository contains a Dockerfile",
                           path="Dockerfile", quote=""), EVIDENCE)
    assert c.status is ClaimStatus.REJECTED


def test_claim_with_no_evidence_at_all_is_rejected():
    c = Claim(id="x", repo="r", text="Did something impressive", evidence=[])
    assert verify_claim(c, EVIDENCE).status is ClaimStatus.REJECTED


def test_reformatted_quote_still_counts():
    """Models normalise whitespace; that should not fail an honest quote."""
    assert quote_supported("Four arms:   text,\n text_interrupt", EVIDENCE)


def test_only_the_supported_quotes_survive():
    c = Claim(id="x", repo="r", text="Ships 34 tests", evidence=[
        Evidence(path="README.md", quote="34 tests, about 2 s"),
        Evidence(path="fake.md", quote="entirely made up supporting sentence here"),
    ])
    out = verify_claim(c, EVIDENCE)
    assert out.status is ClaimStatus.VERIFIED
    assert len(out.evidence) == 1 and out.evidence[0].path == "README.md"


# -- numbers that should be ignored ---------------------------------------
def test_versions_and_years_are_not_treated_as_metrics():
    """'Python 3.11' must not be rejected for lacking a 3.11 metric."""
    assert numbers_in("Built with Python 3.11 in 2026") == []
    assert numbers_in("Spring Boot 3.1.5 backend") == []
    assert numbers_in("v2.1.0 release") == []
    assert verify_claim(claim("Built with Python 3.11"), EVIDENCE).status \
        is ClaimStatus.VERIFIED


def test_decimal_metrics_are_not_mistaken_for_versions():
    """Regression: `\\d+\\.\\d+` masking swallowed '0.81 accuracy' and '49.8%',
    which are precisely the numbers this module exists to police."""
    assert numbers_in("Improved accuracy from 0.81 to 0.92") == ["0.81", "0.92"]
    assert numbers_in("Win rate of 49.8%") == ["49.8"]
    assert numbers_in("MAE of 1315.20 vs 1309.50") == ["1315.20", "1309.50"]
    # And an unsupported decimal metric is still caught.
    assert verify_claim(claim("Reached 0.97 accuracy"), EVIDENCE).status \
        is ClaimStatus.NEEDS_MEASURE


# -- technology filtering -------------------------------------------------
def test_technologies_present_in_evidence_are_kept():
    keep, drop = filter_tech(["Python", "pytest", "httpx"], EVIDENCE)
    assert set(keep) == {"Python", "pytest", "httpx"} and drop == []


def test_invented_technologies_are_dropped():
    keep, drop = filter_tech(["Python", "Kubernetes", "Kafka"], EVIDENCE)
    assert keep == ["Python"]
    assert set(drop) == {"Kubernetes", "Kafka"}


def test_tech_mismatch_does_not_invalidate_a_claim():
    """Regression: scanning claim prose for tech-shaped words rejected 39 of 48
    real claims over words like 'fallback.' and 'scripts.'"""
    c = verify_claim(claim("Provides a local-model fallback and export scripts."),
                     EVIDENCE)
    assert c.status is ClaimStatus.VERIFIED


# -- node -----------------------------------------------------------------
async def test_node_partitions_claims_and_prunes_tech():
    summary = RepoSummary(repo="r", tech=["Python", "Kubernetes"], claims=[
        claim("Ships 34 tests"),
        claim("Reduced latency by 40%"),
        claim("Invented", quote="nothing like this exists in the bundle at all"),
    ])
    out = await verify_claims({"summaries": {"r": summary},
                               "evidence_texts": {"r": EVIDENCE}})
    statuses = [c.status for c in out["claims"]]
    assert statuses == [ClaimStatus.VERIFIED, ClaimStatus.NEEDS_MEASURE,
                        ClaimStatus.REJECTED]
    assert out["summaries"]["r"].tech == ["Python"]
