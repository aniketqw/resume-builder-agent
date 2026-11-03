"""The import endpoint, driven against a fake pipeline.

No network, no credentials: `build_from_github` is replaced, so what is under
test is the HTTP surface, the job/SSE plumbing and -- most importantly -- that
generated lines arrive switched OFF.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rba.resume.document import Bullet, Entry, ResumeDoc, Section
from rba.resume.merge import adopted, candidates
from rba.resume.pipeline import ImportResult
from rba_api.app import create_app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
RESUME = (FIXTURES / "aniket_bits.txt").read_bytes()


def fake_generated() -> ResumeDoc:
    entry = Entry(title="cool-repo", subtitle="Python",
                  bullets=[Bullet(text="Built a thing with 12 tests.",
                                  source_repo="cool-repo", verified=True),
                           Bullet(text="Added CI on every push.",
                                  source_repo="cool-repo", verified=True)])
    return ResumeDoc(name="octocat", sections=[Section(title="PROJECTS",
                                                       entries=[entry])])


def fake_result(stats=None) -> ImportResult:
    return ImportResult(
        doc=fake_generated(),
        stats=stats or {"repos_found": 9, "public": 7, "selected": 2,
                        "summarised": 2, "errors": [], "claims_total": 4,
                        "claims_verified": 4, "claims_rejected": 0},
        facts={"cool-repo": object()},
        summaries={"cool-repo": object()},
        repos=[_repo("cool-repo"), _repo("skipped-repo")],
        selected=[_repo("cool-repo")],
        excluded=[_repo("skipped-repo", "fork with no commits of yours")],
    )


def _repo(name, reason=""):
    from rba.state import RepoRef
    return RepoRef(name=name, full_name=f"octocat/{name}",
                   selection_reason=reason)


@pytest.fixture
def client(monkeypatch):
    async def fake_build(username, **kwargs):
        progress = kwargs.get("progress")
        if progress:
            progress("Checking credentials…", 0.02)
            progress("Summarising projects…", 0.55)
            progress("Done", 1.0)
        return fake_result()

    monkeypatch.setattr("rba_api.routes.github.build_from_github", fake_build)
    with TestClient(create_app(testing=True)) as c:
        yield c


@pytest.fixture
def session(client):
    return client.post("/api/upload",
                       files={"file": ("r.txt", RESUME, "text/plain")}
                       ).json()["session_id"]


def _run(client, session, **body):
    payload = {"username": "octocat", **body}
    job_id = client.post(f"/api/session/{session}/github",
                         json=payload).json()["job_id"]
    # Draining the SSE stream also waits for the task to finish.
    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        events = list(stream.iter_lines())
    return job_id, events


def test_import_adds_lines_switched_off(client, session):
    """The core rule: your resume is the baseline and is never rewritten."""
    before = client.get(f"/api/session/{session}").json()["document"]
    doc_before = ResumeDoc.from_dict(before)
    enabled_before = len(doc_before.enabled())

    _run(client, session)

    after = ResumeDoc.from_dict(
        client.get(f"/api/session/{session}").json()["document"])
    assert len(candidates(after)) == 2         # two lines were added
    assert adopted(after) == []                # and none of them are on
    assert len(after.enabled()) == enabled_before


def test_the_score_does_not_move_until_a_line_is_accepted(client, session):
    jd = "Software Engineer. Python, Docker, testing."
    client.patch(f"/api/session/{session}/target", json={"jd_text": jd})
    before = client.post("/api/score", json={"session": session}).json()["raw"]
    _run(client, session)
    after = client.post("/api/score", json={"session": session}).json()["raw"]
    assert after == before


def test_accepting_a_line_moves_the_score(client, session):
    jd = "Software Engineer. Python, Docker, testing, CI."
    client.patch(f"/api/session/{session}/target", json={"jd_text": jd})
    _run(client, session)
    doc = ResumeDoc.from_dict(
        client.get(f"/api/session/{session}").json()["document"])
    before = client.post("/api/score", json={"session": session}).json()["raw"]
    new_line = candidates(doc)[0]
    client.patch(f"/api/session/{session}/document",
                 json={"ops": [{"op": "enable", "bullet_id": new_line.id}]})
    after = client.post("/api/score", json={"session": session}).json()["raw"]
    assert after != before


def test_progress_streams_then_a_done_event(client, session):
    _job, events = _run(client, session)
    text = "\n".join(events)
    assert "event: progress" in text
    assert "event: done" in text
    assert "Summarising projects" in text


def test_the_done_event_carries_the_new_document(client, session):
    _job, events = _run(client, session)
    payloads = [json.loads(line[6:]) for line in events if line.startswith("data: ")]
    done = payloads[-1]
    assert done["rev"] >= 2
    assert done["added"] == 2
    assert done["document"]["sections"]


def test_discarded_pipeline_state_is_kept(client, session):
    """facts/summaries/repos/excluded used to be dropped on the floor."""
    _run(client, session)
    done = json.loads([line for line in _run(client, session)[1]
                       if line.startswith("data: ")][-1][6:])
    assert done["excluded"] == [{"name": "skipped-repo",
                                 "reason": "fork with no commits of yours"}]


def test_polling_works_as_well_as_streaming(client, session):
    job_id, _ = _run(client, session)
    body = client.get(f"/api/jobs/{job_id}").json()
    assert body["state"] == "done"
    assert body["progress"] == 1.0


def test_a_blank_username_is_refused(client, session):
    assert client.post(f"/api/session/{session}/github",
                       json={"username": "  "}).status_code == 400


def test_an_unknown_session_is_404(client):
    assert client.post("/api/session/s_nope/github",
                       json={"username": "octocat"}).status_code == 404


def test_a_failing_import_reports_why_without_leaking_the_key(client, session,
                                                              monkeypatch):
    from fakes import fake_secret

    token = fake_secret("gsk_")

    async def boom(username, **kwargs):
        raise RuntimeError(f"groq: 401 invalid_api_key for {token}")

    monkeypatch.setattr("rba_api.routes.github.build_from_github", boom)
    _job, events = _run(client, session)
    text = "\n".join(events)
    assert "event: error" in text
    assert token not in text              # redacted
    assert "revoked" in text or "rejected" in text


def test_partial_failures_are_explained(client, session, monkeypatch):
    async def partial(username, **kwargs):
        return fake_result({"summarised": 0, "errors": ["summarize[a]: 401 invalid_api_key"]})

    monkeypatch.setattr("rba_api.routes.github.build_from_github", partial)
    _job, events = _run(client, session)
    done = json.loads([line for line in events if line.startswith("data: ")][-1][6:])
    assert done["failure"] is not None
    assert done["failure"]["level"] == "error"


# --- turning provider errors into sentences ---------------------------------

@pytest.mark.parametrize("raw, expect", [
    ("RuntimeError: groq 401 invalid_api_key", "revoked"),
    ("RuntimeError: GitHub returned an error: Validation Failed\n"
     "The listed users and repositories cannot be searched", "spelling"),
    ("RuntimeError: 404 not found", "spelling"),
    ("RuntimeError: secondary rate limit exceeded", "60 requests"),
])
def test_provider_errors_become_actionable_sentences(raw, expect):
    from rba_api.routes.github import _explain

    assert expect in _explain(RuntimeError(raw))


def test_an_unrecognised_error_is_still_redacted():
    from fakes import fake_secret
    from rba_api.routes.github import _explain

    token = fake_secret("ghp_")
    message = _explain(RuntimeError(f"something odd happened using {token}"))
    assert token not in message
    assert "[redacted]" in message
