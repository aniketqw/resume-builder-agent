"""`build_evidence` and the code-structure block it adds.

There was no dedicated test file for `rba.nodes.summarize` before this --
these tests cover the pre-existing bundle shape as well as what this revamp
added, so a future change to either has something to catch it.
"""
from __future__ import annotations

import json

from fakes import FakeGroq, chat_response
from rba.cache import Cache
from rba.config import Config
from rba.groq_client import GroqClient
from rba.limiter import Limiter
from rba.nodes.summarize import build_evidence, summarize_one
from rba.repobrain import index
from rba.state import RepoFacts


def _facts(**kw) -> RepoFacts:
    base = {"repo": "octocat/demo", "readme": "# Demo\nDoes a thing.",
           "readme_path": "README.md", "tree": ["README.md", "src/app.py"]}
    base.update(kw)
    return RepoFacts(**base)


SOURCE = (
    "def _resolve(session, body):\n"
    "    return session.doc, body.target\n"
    "\n"
    "async def score_one(req):\n"
    "    doc, target = _resolve(req.session, req.body)\n"
    "    return score_many([doc], target)\n"
)


def test_build_evidence_without_a_brain_matches_the_old_shape():
    """No source was fetched (or nothing parsed) -> no `code` key at all,
    not an empty one. The bundle must be identical to before this revamp."""
    facts = _facts()
    bundle = json.loads(build_evidence(facts))
    assert "code" not in bundle
    assert bundle["readme"]["text"].startswith("# Demo")
    assert bundle["files"] == ["README.md", "src/app.py"]


def test_build_evidence_adds_citeable_structure_from_the_brain():
    facts = _facts(source_files={"src/app.py": SOURCE})
    brain = index(facts.source_files)
    bundle = json.loads(build_evidence(facts, brain=brain))
    code = bundle["code"]
    assert "src/app.py" in code["defs"]
    names = {d["name"] for d in code["defs"]["src/app.py"]}
    assert names == {"_resolve", "score_one"}
    assert any(d["lines"] for d in code["defs"]["src/app.py"])
    assert any("_resolve" in c for c in code["calls"])
    # The excerpt must be real text sliced from the actual source, not a
    # paraphrase -- otherwise the model would be quoting a quote.
    assert any("_resolve" in e["text"] or "score_many" in e["text"]
              for e in code["excerpts"])


def test_code_structure_is_dropped_before_the_file_list_under_budget_pressure():
    facts = _facts(source_files={"src/app.py": SOURCE})
    brain = index(facts.source_files)
    without_code = len(build_evidence(facts, max_chars=100_000))       # no brain
    with_code = len(build_evidence(facts, max_chars=100_000, brain=brain))
    assert with_code > without_code, "the fixture must actually add bytes"
    # A budget that fits everything except the code block.
    budget = without_code
    bundle = json.loads(build_evidence(facts, max_chars=budget, brain=brain))
    assert "code" not in bundle
    assert "files" in bundle  # files survive one tier longer than code


def test_a_route_registration_reaches_the_bundle():
    source = (
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n\n"
        "@router.post(\"/api/score\")\n"
        "async def score_one(req):\n"
        "    return 1\n"
    )
    facts = _facts(source_files={"src/routes.py": source})
    brain = index(facts.source_files)
    bundle = json.loads(build_evidence(facts, brain=brain))
    assert bundle["code"]["routes"] == ["POST /api/score -> score_one"]


async def test_summarize_one_threads_the_brain_into_the_prompt():
    """The evidence text handed to the model (and later checked by the
    verifier) must contain the code structure, not just the README."""
    facts = _facts(source_files={"src/app.py": SOURCE}, head_sha="deadbeef")
    brain = index(facts.source_files)
    cfg = Config.load()
    fake = FakeGroq([chat_response(
        '{"one_line":"x","problem":"","approach":"","tech":[],'
        '"claims":[],"not_claimable_reason":""}')])
    async with GroqClient(config=cfg, limiter=Limiter(cfg), cache=Cache(enabled=False),
                         api_key="test-key", transport=fake.transport()) as client:
        _summary, evidence = await summarize_one(client, facts, brain=brain)
    assert "_resolve" in evidence
    sent_user_message = fake.requests[0]["messages"][-1]["content"]
    assert "_resolve" in sent_user_message
