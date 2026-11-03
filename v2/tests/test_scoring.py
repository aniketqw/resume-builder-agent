"""The rubric is deterministic, so every number here is exact, not approximate."""
import pytest

from rba.scoring import (
    WEIGHTS,
    load_roles,
    max_gain,
    score_contact,
    score_keywords,
    score_length,
    score_numbers,
    score_resume,
    score_sections,
    score_verbs,
)
from rba.state import Resume

ROLES = load_roles()
CONTACT = "aniket@example.com | github.com/aniketqw | +91 98765 43210"


def resume(**sections):
    return Resume(role="balanced", headline="Software Engineer",
                  summary="Builds things.", sections=sections or {})


def test_weights_sum_to_one_hundred():
    assert sum(WEIGHTS.values()) == 100.0


def test_all_five_roles_load():
    assert set(ROLES) == {"ai_llm_agent_engineer", "ml_engineer",
                          "backend_platform_engineer", "data_scientist", "balanced"}


# -- keywords -------------------------------------------------------------
def test_full_keyword_coverage_scores_full_marks():
    role = ROLES["ai_llm_agent_engineer"]
    text = " ".join(role.keywords)
    score, matched, missing = score_keywords(text, role)
    assert score == pytest.approx(WEIGHTS["keywords"])
    assert missing == []


def test_no_coverage_scores_zero():
    score, _, missing = score_keywords("entirely unrelated prose about gardening",
                                       ROLES["ai_llm_agent_engineer"])
    assert score == 0.0
    assert len(missing) == len(ROLES["ai_llm_agent_engineer"].keywords)


def test_keyword_scoring_saturates_so_stuffing_does_not_pay():
    """Repetition must not beat coverage, or the generator learns to stuff."""
    role = ROLES["ai_llm_agent_engineer"]
    stuffed, _, _ = score_keywords("langgraph " * 50, role)
    once, _, _ = score_keywords("langgraph", role)
    assert stuffed == once


def test_substrings_do_not_count_as_matches():
    role = ROLES["backend_platform_engineer"]
    _, matched, _ = score_keywords("the meeting is ongoing", role)
    assert "go" not in matched


def test_weighted_keywords_contribute_proportionally():
    role = ROLES["ai_llm_agent_engineer"]
    heavy, _, _ = score_keywords("langgraph", role)   # weight 3.0
    light, _, _ = score_keywords("python", role)      # weight 1.0
    assert heavy == pytest.approx(light * 3.0)


# -- numbers --------------------------------------------------------------
def test_quantified_bullets_score():
    half = score_numbers(["Shipped 34 tests", "Built a parser"])
    none = score_numbers(["Built a parser", "Wrote some code"])
    assert half > none == 0.0


def test_half_the_bullets_quantified_is_full_marks():
    assert score_numbers(["Cut p95 to 120ms", "Built a parser"]) == pytest.approx(
        WEIGHTS["numbers"])


def test_measure_placeholders_score_nothing():
    """An unmeasured claim must never earn the points a real number earns."""
    real = score_numbers(["Reduced latency by 40%"])
    placeholder = score_numbers(["Reduced latency by [MEASURE: time it]"])
    assert real == pytest.approx(WEIGHTS["numbers"])
    assert placeholder == 0.0


def test_years_are_not_quantities():
    assert score_numbers(["Worked on this in 2026"]) == 0.0


# -- sections, contact, length, verbs -------------------------------------
def test_sections_score_proportionally():
    role = ROLES["balanced"]
    full = resume(summary=["s"], experience=["e"], projects=["p"], skills=["k"],
                  education=["d"])
    assert score_sections(full, role) == pytest.approx(WEIGHTS["sections"])
    assert score_sections(resume(projects=["p"]), role) == pytest.approx(
        WEIGHTS["sections"] / 5)


def test_contact_needs_all_three_channels():
    assert score_contact(CONTACT) == pytest.approx(WEIGHTS["contact"])
    assert score_contact("aniket@example.com") == pytest.approx(WEIGHTS["contact"] / 3)
    assert score_contact("no contact details here") == 0.0


def test_length_inside_the_band_scores_full_marks():
    role = ROLES["balanced"]
    low, high = role.target_words
    assert score_length("word " * ((low + high) // 2), role) == pytest.approx(
        WEIGHTS["length"])


def test_length_falls_off_outside_the_band():
    role = ROLES["balanced"]
    short = score_length("word " * 50, role)
    very_short = score_length("word", role)
    assert 0.0 <= very_short < short < WEIGHTS["length"]


def test_action_verbs_scored_on_the_first_word():
    role = ROLES["balanced"]
    assert score_verbs(["Built a parser", "Shipped a service"], role) == pytest.approx(
        WEIGHTS["verbs"])
    assert score_verbs(["Responsible for the parser"], role) == 0.0
    assert score_verbs(["- Built a parser"], role) == pytest.approx(WEIGHTS["verbs"])


# -- whole resume ---------------------------------------------------------
def test_scoring_is_deterministic():
    r = resume(projects=["Built an agent with LangGraph and MCP"])
    first = score_resume(r, "ai_llm_agent_engineer", ROLES, CONTACT)
    second = score_resume(r, "ai_llm_agent_engineer", ROLES, CONTACT)
    assert first.total == second.total
    assert first.components == second.components


def test_a_resume_scores_higher_for_the_role_it_targets():
    agent_resume = resume(projects=[
        "Built a LangGraph agent with MCP tool-calling and structured-output guardrails",
        "Shipped an evaluation harness measuring hallucination rate with LangSmith",
    ])
    targeted = score_resume(agent_resume, "ai_llm_agent_engineer", ROLES, CONTACT)
    mismatched = score_resume(agent_resume, "data_scientist", ROLES, CONTACT)
    assert targeted.total > mismatched.total


def test_total_never_exceeds_one_hundred():
    role = ROLES["ai_llm_agent_engineer"]
    stuffed = resume(summary=["Built " + " ".join(role.keywords) + " 42 times"],
                     experience=["Shipped 10 things"], projects=["Designed 3 systems"],
                     skills=["Led 5 projects"], education=["Built 1 degree"])
    assert score_resume(stuffed, "ai_llm_agent_engineer", ROLES, CONTACT).total <= 100.0


def test_empty_resume_scores_near_zero():
    assert score_resume(resume(), "balanced", ROLES, "").total < 10.0


def test_missing_keywords_are_ordered_by_weight():
    s = score_resume(resume(projects=["Built something"]), "ai_llm_agent_engineer",
                     ROLES, CONTACT)
    weights = [ROLES["ai_llm_agent_engineer"].keywords[k] for k in s.missing_keywords]
    assert weights == sorted(weights, reverse=True), "the planner relies on this order"


# -- planner arithmetic ---------------------------------------------------
def test_max_gain_matches_the_rubric_arithmetic():
    """The planner quotes this as 'ATS gain', so it must be the same maths."""
    role_key = "ai_llm_agent_engineer"
    before = score_resume(resume(projects=["Built something"]), role_key, ROLES, CONTACT)
    top = before.missing_keywords[:5]
    gain = max_gain(role_key, before.missing_keywords, ROLES, limit=5)
    after = score_resume(resume(projects=["Built something " + " ".join(top)]),
                         role_key, ROLES, CONTACT)
    assert after.components["keywords"] - before.components["keywords"] == pytest.approx(
        gain, abs=0.01)
