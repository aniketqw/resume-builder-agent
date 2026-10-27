import pytest

from fakes import FakeJev
from rba.config import Config
from rba.jev_client import JevClient
from rba.nodes.select import (
    CLAIMABLE_Q,
    TIER_Q,
    apply_hard_rules,
    drop_empty,
    heuristic_score,
    select_repos,
)
from rba.signals import RepoSignals
from rba.state import RepoRef


def repo(name, **kw):
    kw.setdefault("size_kb", 500)
    return RepoRef(name=name, **kw)


# -- hard rules (free, deterministic) -------------------------------------
def test_forks_and_archived_are_excluded_without_a_model():
    cfg = Config.load()
    repos = [repo("real"), repo("forked", is_fork=True), repo("old", is_archived=True)]
    keep, drop = apply_hard_rules(repos, cfg)
    assert [r.name for r in keep] == ["real"]
    assert {r.name for r in drop} == {"forked", "old"}
    assert all(r.selection_reason for r in drop), "every rejection needs a reason"


def test_zero_size_is_not_treated_as_empty():
    """MCP search_repositories omits `size`; judging emptiness on it rejected
    all 98 repos in a live run. Emptiness is decided from signals instead."""
    cfg = Config.load()
    keep, drop = apply_hard_rules([repo("unknown-size", size_kb=0)], cfg)
    assert [r.name for r in keep] == ["unknown-size"]
    assert drop == []


def test_drop_empty_uses_files_and_commits():
    sigs = {}
    for name, files, commits in [("empty", 0, 0), ("one-commit", 5, 1), ("real", 20, 12)]:
        s = RepoSignals(name)
        s.file_count, s.commit_count = files, commits
        sigs[name] = s
    keep, drop = drop_empty([repo("empty"), repo("one-commit"), repo("real")], sigs,
                            min_commits=2)
    assert [r.name for r in keep] == ["real"]
    assert {r.name for r in drop} == {"empty", "one-commit"}
    assert all(r.selection_reason for r in drop)


def test_drop_empty_keeps_repos_with_no_signal_data():
    """A failed signal fetch must not silently delete a repo."""
    keep, drop = drop_empty([repo("unknown")], {}, min_commits=2)
    assert [r.name for r in keep] == ["unknown"] and drop == []


def test_private_repos_are_not_excluded():
    cfg = Config.load()
    keep, _ = apply_hard_rules([repo("secret", is_private=True)], cfg)
    assert [r.name for r in keep] == ["secret"]


# -- question shape -------------------------------------------------------
def test_questions_use_criteria_maps():
    """A bare options list is a 422 from the live API."""
    assert set(CLAIMABLE_Q["criteria"]) == {"true", "false"}
    assert set(TIER_Q["criteria"]) == {"small", "mid", "big"}
    assert "options" not in TIER_Q


# -- scoring --------------------------------------------------------------
async def test_jev_scores_drive_selection_and_ranking():
    scores = {"alpha": 0.81, "beta": 0.55, "gamma": 0.20}

    def answer(body):
        name = body["state"]["name"]
        return {"claimable": {"noul": scores[name]},
                "tier": {"choice": "mid", "confidence": 0.9}}

    fake = FakeJev(answer)
    jev = JevClient(Config.load(), api_key="k", transport=fake.transport())
    out = await select_repos(
        {"repos": [repo("gamma"), repo("alpha"), repo("beta")], "username": "u"},
        jev=jev)
    await jev.aclose()
    assert [r.name for r in out["selected"]] == ["alpha", "beta"]  # gamma below 0.45
    assert out["selected"][0].claimable == 0.81


async def test_max_repos_caps_the_groq_budget():
    """25 repos at ~2.1k tokens is ~10 min at 8k TPM. 93 would be ~40."""
    fake = FakeJev({"claimable": {"noul": 0.9}, "tier": {"choice": "small", "confidence": 1.0}})
    jev = JevClient(Config.load(), api_key="k", transport=fake.transport())
    out = await select_repos({"repos": [repo(f"r{i}") for i in range(60)], "username": "u"},
                             jev=jev)
    await jev.aclose()
    assert len(out["selected"]) == Config.load().get("selection.max_repos")


async def test_jev_cost_is_reported():
    fake = FakeJev({"claimable": {"noul": 0.9}, "tier": {"choice": "small", "confidence": 1.0}},
                   input_tokens=800)
    jev = JevClient(Config.load(), api_key="k", transport=fake.transport())
    out = await select_repos({"repos": [repo("a"), repo("b")], "username": "u"}, jev=jev)
    await jev.aclose()
    usage = out["usage"]["jev"]
    assert usage["calls"] == 2 and usage["input_tokens"] == 1600
    assert usage["cost_usd"] < 0.001


async def test_jev_failure_degrades_to_heuristic_instead_of_crashing():
    fake = FakeJev({}, status=500)
    jev = JevClient(Config.load(), api_key="k", transport=fake.transport())
    out = await select_repos(
        {"repos": [repo("good", description="a real thing", language="Python",
                        size_kb=3000, stars=2)], "username": "u"},
        jev=jev)
    await jev.aclose()
    assert len(out["selected"]) == 1
    assert "jev failed" in out["selected"][0].selection_reason


async def test_unavailable_jev_uses_heuristic_path():
    jev = JevClient(Config.load(), api_key="")
    out = await select_repos(
        {"repos": [repo("x", description="d", language="Go", size_kb=3000)],
         "username": "u"}, jev=jev)
    await jev.aclose()
    assert out["selected"][0].selection_reason == "heuristic (jev unavailable)"


async def test_empty_selection_is_reported_as_an_error():
    fake = FakeJev({"claimable": {"noul": 0.01}, "tier": {"choice": "small", "confidence": 1.0}})
    jev = JevClient(Config.load(), api_key="k", transport=fake.transport())
    out = await select_repos({"repos": [repo("a")], "username": "u"}, jev=jev)
    await jev.aclose()
    assert out["selected"] == []
    assert any("claimable threshold" in e for e in out["errors"])


# -- heuristic fallback ---------------------------------------------------
def test_heuristic_prefers_substantial_repos():
    rich = repo("engine", description="d", language="Go", size_kb=3000, stars=5)
    thin = repo("walmartPrep", size_kb=10)
    assert heuristic_score(rich) > heuristic_score(thin)


@pytest.mark.parametrize("name", ["walmartPrep", "100xDevCohort4", "my-tutorial"])
def test_heuristic_penalises_obvious_course_names(name):
    assert heuristic_score(repo(name)) < heuristic_score(repo("payments-service"))
