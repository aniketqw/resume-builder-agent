"""A role has to be scoreable on its own, and honestly labelled."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rba.resume.role_target import (
    PREFERRED_TERMS,
    REQUIRED_TERMS,
    ranked_keywords,
    role_to_target,
    target_for_key,
)
from rba.scoring import load_roles
from rba_api.app import create_app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
RESUME = (FIXTURES / "aniket_bits.txt").read_bytes()


# ------------------------------------------------------------ synthesis --
def test_every_role_in_roles_yaml_produces_a_target():
    for key in load_roles():
        target = target_for_key(key)
        assert target is not None
        assert target.jd_text.strip()
        assert target.required_skills


def test_keywords_are_ranked_by_descending_weight():
    role = load_roles()["balanced"]
    ranked = ranked_keywords(role)
    weights = [role.keywords[t] for t in ranked]
    assert weights == sorted(weights, reverse=True)


def test_ranking_is_stable_so_the_score_does_not_drift():
    """Ties break alphabetically. An unstable job description would change the
    score for no reason and invalidate every cached delta."""
    role = load_roles()["balanced"]
    assert ranked_keywords(role) == ranked_keywords(role)


def test_required_and_preferred_do_not_overlap():
    target = target_for_key("balanced")
    assert not set(target.required_skills) & set(target.preferred_skills)
    assert len(target.required_skills) <= REQUIRED_TERMS
    assert len(target.preferred_skills) <= PREFERRED_TERMS - REQUIRED_TERMS


def test_the_jd_text_names_the_role():
    role = load_roles()["balanced"]
    target = role_to_target(role, "balanced")
    assert role.title in target.jd_text
    assert role.headline in target.jd_text


def test_it_is_always_flagged_as_synthesised():
    """The UI keys its caveat off this. If it were ever False a keyword
    profile would be presented as if it were a real posting."""
    target = target_for_key("balanced")
    assert target.synthesised is True
    assert target.caveat
    assert "stand-in" in target.caveat


def test_an_unknown_role_degrades_instead_of_raising():
    assert target_for_key("no_such_role") is None


# ----------------------------------------------------------------- API --
@pytest.fixture
def client():
    with TestClient(create_app(testing=True)) as c:
        yield c


@pytest.fixture
def session(client):
    payload = client.post("/api/upload",
                          files={"file": ("r.txt", RESUME, "text/plain")}).json()
    return payload["session_id"]


def test_a_role_with_no_jd_now_scores(client, session):
    """The regression: this returned 400 'no job description set'."""
    client.patch(f"/api/session/{session}/target",
                 json={"jd_text": "", "role_key": "balanced"})
    response = client.post("/api/score", json={"session": session})
    assert response.status_code == 200
    assert response.json()["score_100"] > 0


def test_a_role_activates_the_skills_channel(client, session):
    """Without synthesised required_skills the skills channel is absent
    entirely and its 0.3 gets redistributed."""
    client.patch(f"/api/session/{session}/target",
                 json={"jd_text": "", "role_key": "balanced"})
    body = client.post("/api/score", json={"session": session}).json()
    assert "skills" in body["channels"]
    assert body["weights_used"]["skills"] == 0.3


def test_the_response_says_the_target_was_synthesised(client, session):
    client.patch(f"/api/session/{session}/target",
                 json={"jd_text": "", "role_key": "balanced"})
    target = client.post("/api/score", json={"session": session}).json()["target"]
    assert target["synthesised"] is True
    assert target["jd_text"]            # shown verbatim in the UI


def test_a_real_jd_is_not_replaced_by_the_role(client, session):
    """A role selected alongside a pasted posting must not override it."""
    jd = "Backend engineer. Go, Kubernetes, gRPC, Postgres."
    client.patch(f"/api/session/{session}/target",
                 json={"jd_text": jd, "role_key": "balanced"})
    body = client.post("/api/score", json={"session": session}).json()
    assert body.get("target") is None   # nothing was synthesised


def test_the_batch_endpoint_also_accepts_a_role(client, session):
    """Both the headline and the per-line deltas were 400ing."""
    doc = client.get(f"/api/session/{session}").json()["document"]
    bullet = next(b for s in doc["sections"] for e in s.get("entries", [])
                  for b in e.get("bullets", []))
    client.patch(f"/api/session/{session}/target",
                 json={"jd_text": "", "role_key": "balanced"})
    response = client.post("/api/score/batch", json={
        "session": session,
        "variants": [{"key": "k",
                      "ops": [{"op": "disable", "bullet_id": bullet["id"]}]}]})
    assert response.status_code == 200
    assert "delta" in response.json()["variants"][0]


def test_with_neither_a_jd_nor_a_role_it_still_refuses(client, session):
    response = client.post("/api/score", json={"session": session})
    assert response.status_code == 400
    assert "role" in response.json()["detail"]
