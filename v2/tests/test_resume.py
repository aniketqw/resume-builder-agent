"""Document model and per-bullet attribution.

The central test is `test_delta_equals_an_independent_rescore`: whatever the UI
shows next to a bullet must be what actually happens when the user removes it.
If that drifts, the feature is lying.
"""
import time

import pytest

from rba.resume import Bullet, ResumeDoc, Section, attribute, rescore
from rba.resume.document import from_summaries

JD = ("Backend Engineer. Required: Python, Docker, Kubernetes, REST API, SQL. "
      "Preferred: AWS, Kafka, Terraform. Build and operate microservices.")
REQUIRED = ["Python", "Docker", "Kubernetes", "REST API", "SQL"]
PREFERRED = ["AWS", "Kafka", "Terraform"]


def make_doc(n: int = 5) -> ResumeDoc:
    texts = [
        "Built Python FastAPI microservices serving internal teams",
        "Containerised services with Docker and deployed on Kubernetes",
        "Designed REST APIs consumed by three downstream teams",
        "Wrote the PostgreSQL schema and migration tooling",
        "Ran incident response and wrote postmortems",
        "Automated infrastructure provisioning with Terraform",
        "Migrated a batch job to a streaming Kafka consumer",
        "Documented runbooks for the operations team",
    ][:n]
    return ResumeDoc(
        name="Jane Doe", headline="Backend Engineer",
        contact="jane@example.com | github.com/jane",
        sections=[Section("payments-api",
                          [Bullet(t, "payments-api") for t in texts])])


# -- document -------------------------------------------------------------
def test_bullet_ids_are_content_derived_and_stable():
    """Regenerating the same bullet from the same evidence keeps its id, so a
    stored selection survives a re-run."""
    first = Bullet("Built a thing", "repo-a")
    again = Bullet("Built a thing", "repo-a")
    assert first.id == again.id
    assert Bullet("Built a thing", "repo-b").id != first.id


def test_disabled_bullets_leave_the_scored_text():
    doc = make_doc(3)
    target = doc.all_bullets()[1]
    assert "Kubernetes" in doc.as_text()
    doc.toggle(target.id, False)
    assert "Kubernetes" not in doc.as_text()
    assert len(doc.enabled()) == 2


def test_toggle_returns_the_new_state_and_rejects_unknown_ids():
    doc = make_doc(2)
    bullet_id = doc.all_bullets()[0].id
    assert doc.toggle(bullet_id) is False
    assert doc.toggle(bullet_id) is True
    with pytest.raises(KeyError):
        doc.toggle("nope")


def test_an_emptied_section_does_not_pad_the_text():
    doc = make_doc(2)
    for bullet in doc.all_bullets():
        bullet.enabled = False
    assert "payments-api" not in doc.as_text()


def test_skills_come_only_from_enabled_bullets():
    doc = make_doc(2)
    assert "kubernetes" in doc.skills()
    doc.toggle(doc.all_bullets()[1].id, False)
    assert "kubernetes" not in doc.skills()


def test_round_trips_through_json():
    doc = make_doc(4)
    doc.toggle(doc.all_bullets()[2].id, False)
    restored = ResumeDoc.from_json(doc.to_json())
    assert restored.as_text() == doc.as_text()
    assert [b.enabled for b in restored.all_bullets()] == \
           [b.enabled for b in doc.all_bullets()]


def test_markdown_omits_disabled_bullets():
    doc = make_doc(3)
    doc.toggle(doc.all_bullets()[0].id, False)
    markdown = doc.as_markdown()
    assert "FastAPI" not in markdown
    assert markdown.startswith("# Jane Doe")


# -- clubbing summaries ---------------------------------------------------
def test_rejected_claims_never_enter_the_document():
    from rba.state import Claim, ClaimStatus, Evidence, RepoSummary
    summary = RepoSummary(repo="demo", claims=[
        Claim(id="a", repo="demo", text="Ships 34 tests",
              evidence=[Evidence(path="README.md", quote="34 tests")]),
        Claim(id="b", repo="demo", text="Reduced latency by 40%",
              status=ClaimStatus.NEEDS_MEASURE, measure_hint="time it"),
        Claim(id="c", repo="demo", text="Invented", status=ClaimStatus.REJECTED),
    ])
    doc = from_summaries({"demo": summary})
    texts = [b.text for b in doc.all_bullets()]
    assert "Invented" not in texts
    assert len(texts) == 2
    needs_measure = doc.all_bullets()[1]
    assert not needs_measure.verified and needs_measure.measure_hint == "time it"


def test_evidence_survives_into_a_checkable_permalink():
    """A generated bullet must carry enough to prove itself: not just the
    quote, but the file, commit and line range it was read at. Before this,
    `from_summaries` kept claim.evidence[0].quote and silently dropped
    claim.evidence[0].path, so there was nothing to link back to."""
    from rba.state import Claim, Evidence, RepoFacts, RepoSummary

    summary = RepoSummary(repo="octocat/demo", claims=[
        Claim(id="a", repo="octocat/demo", text="Ships 34 tests",
              evidence=[Evidence(path="tests/test_x.py", quote="34 tests",
                                 repo="octocat/demo", sha="deadbeef",
                                 line_start=10, line_end=12)]),
    ])
    facts = {"octocat/demo": RepoFacts(repo="octocat/demo", head_sha="deadbeef")}
    doc = from_summaries({"octocat/demo": summary}, facts=facts)
    bullet = doc.all_bullets()[0]
    assert bullet.evidence_path == "tests/test_x.py"
    assert bullet.evidence_sha == "deadbeef"
    assert (bullet.evidence_line_start, bullet.evidence_line_end) == (10, 12)
    assert bullet.permalink == (
        "https://github.com/octocat/demo/blob/deadbeef/tests/test_x.py#L10-L12")

    # Round-trips through the session/document JSON boundary unchanged.
    restored = Bullet.from_dict(bullet.to_dict())
    assert restored.evidence_path == bullet.evidence_path
    assert restored.evidence_sha == bullet.evidence_sha
    assert restored.permalink == bullet.permalink


def test_permalink_is_empty_without_a_repo_file_and_commit():
    """A bullet parsed from the user's own resume has no repo evidence at
    all, and a claim cited to a commit message (not a file) cannot become a
    blob link -- both must render as "nothing to click", never a broken URL."""
    yours = Bullet(text="Led a team of 4", origin="yours")
    assert yours.permalink == ""

    from rba.state import Claim, Evidence, RepoFacts, RepoSummary
    summary = RepoSummary(repo="octocat/demo", claims=[
        Claim(id="a", repo="octocat/demo", text="Shipped a release",
              evidence=[Evidence(path="commit:deadbeef", quote="v1.0.0")]),
    ])
    facts = {"octocat/demo": RepoFacts(repo="octocat/demo", head_sha="deadbeef")}
    doc = from_summaries({"octocat/demo": summary}, facts=facts)
    assert doc.all_bullets()[0].permalink == ""


def test_an_unverified_claim_never_gets_a_permalink():
    """Found by running a real import: a `needs_measure` claim's evidence
    path is whatever the model wrote and was never validated -- `verify_claim`
    returns before resolving it. A live run produced the permalink
    `.../blob/<sha>/files`, where "files" was the evidence bundle's own JSON
    key, not a real path in the repo. `from_summaries` must only trust a
    claim's path/sha/line-range when it actually reached VERIFIED."""
    from rba.state import Claim, ClaimStatus, Evidence, RepoFacts, RepoSummary

    summary = RepoSummary(repo="octocat/demo", claims=[
        Claim(id="a", repo="octocat/demo", text="Contains 21 files",
              status=ClaimStatus.NEEDS_MEASURE, measure_hint="count the files",
              evidence=[Evidence(path="files", quote="21 files")]),
    ])
    facts = {"octocat/demo": RepoFacts(repo="octocat/demo", head_sha="deadbeef")}
    doc = from_summaries({"octocat/demo": summary}, facts=facts)
    bullet = doc.all_bullets()[0]
    assert not bullet.verified
    assert bullet.evidence_path == "" and bullet.evidence_sha == ""
    assert bullet.permalink == ""
    # The quote itself still reaches the document -- only the unvalidated
    # path/sha/permalink are withheld.
    assert bullet.evidence == "21 files"


# -- attribution ----------------------------------------------------------
def test_delta_equals_an_independent_rescore():
    """The number shown beside a bullet must be what removing it really costs."""
    doc = make_doc(5)
    att = attribute(doc, JD, required_skills=REQUIRED, preferred_skills=PREFERRED)
    for impact in att.impacts:
        doc.toggle(impact.bullet_id, False)
        raw, _ = rescore(doc, JD, required_skills=REQUIRED,
                         preferred_skills=PREFERRED)
        doc.toggle(impact.bullet_id, True)
        assert (att.raw_total - raw) * 100 == pytest.approx(impact.delta, abs=1e-3)


def test_every_enabled_bullet_gets_an_impact():
    doc = make_doc(5)
    doc.toggle(doc.all_bullets()[0].id, False)
    att = attribute(doc, JD, required_skills=REQUIRED)
    assert len(att.impacts) == 4
    assert att.by_id(doc.all_bullets()[0].id) is None


def test_harmful_bullets_are_identified():
    """A grounded scorer can say 'remove this and your score goes up'. One
    measured example: a backend JD scored 0.66 higher without an incident
    response bullet."""
    doc = make_doc(8)
    att = attribute(doc, JD, required_skills=REQUIRED, preferred_skills=PREFERRED)
    assert all(i.delta < 0 for i in att.harmful())
    assert all(i.helps for i in att.ranked()[:1])


def test_relevant_bullets_outrank_irrelevant_ones():
    doc = ResumeDoc(headline="Engineer", sections=[Section("x", [
        Bullet("Deployed services on Kubernetes with Docker", "x"),
        Bullet("Organised the office social committee", "x"),
    ])])
    att = attribute(doc, JD, required_skills=REQUIRED)
    ranked = att.ranked()
    assert "Kubernetes" in ranked[0].text


def test_impact_label_shows_a_sign():
    doc = make_doc(3)
    att = attribute(doc, JD, required_skills=REQUIRED)
    assert all(i.label[0] in "+-" for i in att.impacts)


def test_an_empty_document_does_not_crash():
    att = attribute(ResumeDoc(), JD, required_skills=REQUIRED)
    assert att.impacts == []


@pytest.mark.slow
def test_a_single_toggle_rescore_is_fast_enough_for_a_ui():
    doc = make_doc(8)
    rescore(doc, JD, required_skills=REQUIRED)        # warm the model
    start = time.perf_counter()
    rescore(doc, JD, required_skills=REQUIRED)
    assert (time.perf_counter() - start) < 1.0
