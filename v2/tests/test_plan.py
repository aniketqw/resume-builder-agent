from rba.nodes.plan import _weakest, attach_ats_gain
from rba.scoring import load_roles, max_gain
from rba.state import PlanStep, RoleScore

ROLES = load_roles()


def step(title, **kw):
    kw.setdefault("action", "")
    kw.setdefault("gap", "")
    kw.setdefault("measurable_output", "")
    return PlanStep(title=title, **kw)


def test_ats_gain_comes_from_the_rubric_not_the_model():
    score = RoleScore(role="ai_llm_agent_engineer", total=40.0,
                      missing_keywords=["langgraph", "mcp", "evaluation"])
    steps = [step("Add a LangGraph agent"), step("Wire up MCP tools"),
             step("Build an evaluation harness")]
    out = attach_ats_gain(steps, score, ROLES)
    budget = max_gain("ai_llm_agent_engineer", score.missing_keywords, ROLES,
                      limit=len(steps) * 2)
    assert sum(s.ats_gain for s in out) == budget
    assert all(s.ats_gain > 0 for s in out)


def test_steps_covering_heavier_keywords_get_more_credit():
    score = RoleScore(role="ai_llm_agent_engineer", total=40.0,
                      missing_keywords=["langgraph", "python"])  # 3.0 vs 1.0
    heavy = step("Add langgraph orchestration")
    light = step("Add python typing")
    attach_ats_gain([heavy, light], score, ROLES)
    assert heavy.ats_gain > light.ats_gain


def test_steps_matching_nothing_split_the_budget_evenly():
    score = RoleScore(role="balanced", total=50.0, missing_keywords=["docker"])
    a, b = step("Do something vague"), step("Do another thing")
    attach_ats_gain([a, b], score, ROLES)
    assert a.ats_gain == b.ats_gain


def test_no_steps_is_handled():
    assert attach_ats_gain([], RoleScore(role="balanced"), ROLES) == []


def test_weakest_role_is_identified_for_targeting():
    scores = {"a": RoleScore(role="a", total=70.0), "b": RoleScore(role="b", total=31.0)}
    assert _weakest(scores) == "b"
    assert _weakest({}) == ""
