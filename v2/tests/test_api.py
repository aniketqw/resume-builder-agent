"""The API must not be able to invent a number.

Every assertion about a score here compares the endpoint against calling the
engine directly. If these pass, the browser is showing real fused scores; if
they drift, the whole premise of the UI is gone.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rba.resume.attribution import attribute, rescore
from rba.resume.document import ResumeDoc
from rba_api.app import create_app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
JD = ("AI Engineer. Python, LangGraph, RAG, retrieval, evaluation, Docker, "
      "embeddings, vector search, tool-calling, structured output.")
RESUME = (FIXTURES / "aniket_bits.txt").read_bytes()


@pytest.fixture
def client():
    with TestClient(create_app(testing=True)) as c:
        yield c


@pytest.fixture
def session(client):
    """An uploaded document with a target set. Returns (id, ResumeDoc)."""
    response = client.post("/api/upload",
                           files={"file": ("r.txt", RESUME, "text/plain")})
    assert response.status_code == 200
    payload = response.json()
    client.patch(f"/api/session/{payload['session_id']}/target",
                 json={"jd_text": JD, "role_key": "ai_llm_agent_engineer"})
    return payload["session_id"], ResumeDoc.from_dict(payload["document"])


# ------------------------------------------------------------------ meta --
def test_health_reports_the_model(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["model"] == "BAAI/bge-small-en-v1.5"
    assert "dense" in body


def test_scoring_meta_publishes_the_real_weights(client):
    from rba.ats.fusion import CHANNEL_WEIGHTS

    body = client.get("/api/meta/scoring").json()
    assert body["weights"] == dict(CHANNEL_WEIGHTS)
    assert body["delta_basis"] == "raw"
    assert body["max_tokens"] == 512
    # The page must be able to explain truncation; it changes what a delta means.
    assert "truncation" in body["notes"]


def test_roles_come_from_roles_yaml(client):
    from rba.scoring import load_roles

    body = client.get("/api/meta/roles").json()
    assert {r["key"] for r in body["roles"]} == set(load_roles())


# -------------------------------------------------------------- document --
def test_upload_parses_and_opens_a_session(client):
    body = client.post("/api/upload",
                       files={"file": ("r.txt", RESUME, "text/plain")}).json()
    assert body["rev"] == 1
    assert len(ResumeDoc.from_dict(body["document"]).all_bullets()) > 0


def test_an_empty_upload_is_rejected(client):
    r = client.post("/api/upload", files={"file": ("r.txt", b"", "text/plain")})
    assert r.status_code == 400


def test_paste_works_as_well_as_upload(client):
    r = client.post("/api/parse", json={"text": RESUME.decode(), "name": "x"})
    assert r.status_code == 200
    assert r.json()["document"]["sections"]


def test_patching_a_document_bumps_the_revision(client, session):
    sid, doc = session
    target = doc.enabled()[0]
    r = client.patch(f"/api/session/{sid}/document",
                     json={"ops": [{"op": "disable", "bullet_id": target.id}],
                           "if_match": 1})
    assert r.status_code == 200
    assert r.json()["rev"] == 2


def test_a_stale_revision_is_refused(client, session):
    """Two tabs editing one session must not silently clobber each other."""
    sid, doc = session
    target = doc.enabled()[0]
    r = client.patch(f"/api/session/{sid}/document",
                     json={"ops": [{"op": "disable", "bullet_id": target.id}],
                           "if_match": 99})
    assert r.status_code == 409


def test_an_op_against_an_unknown_bullet_is_refused(client, session):
    sid, _ = session
    r = client.patch(f"/api/session/{sid}/document",
                     json={"ops": [{"op": "disable", "bullet_id": "nope"}]})
    assert r.status_code == 422


def test_an_unknown_session_is_404(client):
    assert client.get("/api/session/s_nope").status_code == 404


# --------------------------------------------------------------- scoring --
def test_score_equals_rescore_exactly(client, session):
    """The load-bearing assertion of the entire API."""
    sid, doc = session
    body = client.post("/api/score", json={"session": sid}).json()
    expected, _channels = rescore(doc, JD)
    assert body["raw"] == expected
    assert body["score_100"] == round(expected * 100, 2)
    assert body["delta_basis"] == "raw"


def test_score_works_without_a_session(client):
    """The stateless path keeps the endpoint a pure function of its input."""
    doc = ResumeDoc.from_dict(
        client.post("/api/upload",
                    files={"file": ("r.txt", RESUME, "text/plain")}
                    ).json()["document"])
    body = client.post("/api/score", json={
        "document": doc.to_dict(), "target": {"jd_text": JD}}).json()
    assert body["raw"] == rescore(doc, JD)[0]


def test_scoring_without_a_job_description_is_refused(client):
    r = client.post("/api/upload", files={"file": ("r.txt", RESUME, "text/plain")})
    sid = r.json()["session_id"]
    assert client.post("/api/score", json={"session": sid}).status_code == 400


def test_every_batch_delta_matches_an_independent_rescore(client, session):
    sid, doc = session
    targets = doc.enabled()[:5]
    variants = [{"key": b.id, "ops": [{"op": "disable", "bullet_id": b.id}]}
                for b in targets]
    body = client.post("/api/score/batch",
                       json={"session": sid, "variants": variants}).json()

    base_raw = body["base"]["raw"]
    assert base_raw == rescore(doc, JD)[0]

    by_key = {v["key"]: v for v in body["variants"]}
    for bullet in targets:
        bullet.enabled = False
        expected, _ = rescore(doc, JD)
        bullet.enabled = True
        assert by_key[bullet.id]["delta"] == round((expected - base_raw) * 100, 2)


def test_batch_reports_how_much_work_it_avoided(client, session):
    sid, doc = session
    variants = [{"key": b.id, "ops": [{"op": "disable", "bullet_id": b.id}]}
                for b in doc.enabled()]
    body = client.post("/api/score/batch",
                       json={"session": sid, "variants": variants}).json()
    stats = body["stats"]
    assert stats["candidates"] == len(variants) + 1
    # The dedup must actually be doing something on a real resume, or the
    # batch endpoint's performance claim is false.
    assert stats["unique_prefixes"] < stats["candidates"]


def test_too_many_variants_is_refused(client, session):
    sid, doc = session
    target = doc.enabled()[0]
    variants = [{"key": f"k{i}", "ops": [{"op": "disable", "bullet_id": target.id}]}
                for i in range(300)]
    r = client.post("/api/score/batch", json={"session": sid, "variants": variants})
    assert r.status_code == 413


def test_attribution_matches_the_engine(client, session):
    sid, doc = session
    body = client.post("/api/attribution", json={"session": sid}).json()
    expected = attribute(doc, JD)
    assert body["raw_total"] == expected.raw_total
    got = [(i["bullet_id"], i["delta"]) for i in body["impacts"]]
    assert got == [(i.bullet_id, i.delta) for i in expected.impacts]


def test_attribution_is_cached_until_the_document_moves(client, session):
    sid, doc = session
    first = client.post("/api/attribution", json={"session": sid}).json()
    second = client.post("/api/attribution", json={"session": sid}).json()
    assert first["signature"] == second["signature"]

    target = doc.enabled()[0]
    client.patch(f"/api/session/{sid}/document",
                 json={"ops": [{"op": "disable", "bullet_id": target.id}]})
    third = client.post("/api/attribution", json={"session": sid}).json()
    assert third["signature"] != first["signature"]


# ---------------------------------------------------------------- render --
def test_html_preview_renders_both_styles(client, session):
    sid, _ = session
    for style in ("bits", "clean"):
        body = client.post("/api/render/html",
                           json={"session": sid, "style": style}).json()
        assert body["style"] == style
        assert len(body["html"]) > 100


def test_pdf_is_a_pdf_and_reports_its_page_count(client, session):
    sid, _ = session
    r = client.post("/api/render/pdf", json={"session": sid, "style": "clean"})
    assert r.status_code == 200
    assert r.content[:4] == b"%PDF"
    assert int(r.headers["X-RBA-Pages"]) >= 1


def test_export_bundles_both_styles(client, session):
    import io
    import zipfile

    sid, _ = session
    r = client.post("/api/export", json={"session": sid})
    assert r.status_code == 200
    names = set(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    assert {"resume-bits.pdf", "resume-clean.pdf", "resume.txt",
            "resume.json", "README.txt"} <= names


def test_rendering_a_session_with_no_document_is_a_conflict(client):
    sid = client.post("/api/session").json()["session_id"]
    assert client.post("/api/render/html", json={"session": sid}).status_code == 409
