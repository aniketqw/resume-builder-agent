"""Offline tests for the Hugging Face bundle and deployment planner.

The Space is public and this repository is not, so the property that matters
most is negative: things that must never be published, aren't.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_space import (
    ALLOWLIST,
    MANIFEST,
    build,
    validate_card,
)
from deploy_space import (
    REPO_ID_PATTERN,
    plan_changes,
    scrub,
    validate_manifest,
)

CARD = """---
title: Resume Builder Agent
emoji: 📄
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
---
body
"""


# -- the allowlist is the safety property ---------------------------------
def test_allowlist_is_small_and_explicit():
    """Anything not named here cannot reach a public Space."""
    assert set(ALLOWLIST) == {"README.md", "index.html", "style.css"}


def test_nothing_sensitive_is_reachable_through_the_allowlist():
    forbidden = (".env", "out/", "fixtures/", "tests/", "evals/", "src/",
                 "config.yaml", ".cache")
    for source in ALLOWLIST.values():
        assert not any(source.startswith(f) or f in source for f in forbidden), source


def test_build_stages_only_allowlisted_files(tmp_path):
    staged = tmp_path / "stage"
    manifest = build(ROOT, staged)
    assert set(manifest["files"]) == set(ALLOWLIST)
    on_disk = {p.name for p in staged.rglob("*") if p.is_file()}
    assert on_disk == set(ALLOWLIST) | {MANIFEST}


def test_build_refuses_a_non_empty_staging_directory(tmp_path):
    staged = tmp_path / "stage"
    staged.mkdir()
    (staged / "stray.txt").write_text("x")
    with pytest.raises(ValueError, match="new or empty"):
        build(ROOT, staged)


def test_build_refuses_to_stage_inside_the_repository():
    with pytest.raises(ValueError, match="outside the source repository"):
        build(ROOT, ROOT / "space" / "stage")


# -- Space card -----------------------------------------------------------
def test_valid_card_passes():
    validate_card(CARD)


@pytest.mark.parametrize("broken,match", [
    (CARD.replace("sdk: static", "sdk: gradio"), "sdk: static"),
    (CARD.replace("app_file: index.html", "app_file: app.py"), "app_file"),
    (CARD.replace("colorFrom: green", "colorFrom: chartreuse"), "colour"),
    ("no frontmatter at all", "frontmatter"),
])
def test_bad_cards_are_rejected(broken, match):
    with pytest.raises(ValueError, match=match):
        validate_card(broken)


def test_the_real_card_is_valid():
    validate_card((ROOT / "space" / "README.md").read_text())


# -- manifest -------------------------------------------------------------
def good_manifest(**files):
    return {"managed_by": "aniketqw/resume-builder-agent",
            "files": files or {"index.html": "a" * 64}}


def test_manifest_from_another_project_is_rejected():
    with pytest.raises(ValueError, match="does not belong"):
        validate_manifest({"managed_by": "someone/else", "files": {}})


@pytest.mark.parametrize("path", ["../escape.html", "/etc/passwd", "a/../../b"])
def test_path_traversal_in_a_manifest_is_rejected(path):
    with pytest.raises(ValueError, match="unsafe path"):
        validate_manifest(good_manifest(**{path: "a" * 64}))


def test_a_bad_hash_is_rejected():
    with pytest.raises(ValueError, match="invalid hash"):
        validate_manifest(good_manifest(**{"index.html": "not-a-sha"}))


# -- change planning ------------------------------------------------------
def test_first_deploy_uploads_everything():
    current = good_manifest(**{"index.html": "a" * 64, "style.css": "b" * 64})
    changed, deleted = plan_changes(None, current, set())
    assert changed == ["index.html", "style.css"]
    assert deleted == []


def test_unchanged_files_are_not_reuploaded():
    manifest = good_manifest(**{"index.html": "a" * 64, "style.css": "b" * 64})
    changed, deleted = plan_changes(manifest, manifest,
                                    {"index.html", "style.css"})
    assert changed == [] and deleted == []


def test_only_changed_files_are_uploaded():
    previous = good_manifest(**{"index.html": "a" * 64, "style.css": "b" * 64})
    current = good_manifest(**{"index.html": "c" * 64, "style.css": "b" * 64})
    changed, _ = plan_changes(previous, current, {"index.html", "style.css"})
    assert changed == ["index.html"]


def test_deletions_are_limited_to_files_we_previously_owned():
    """A Space may hold files from elsewhere; we must never remove those."""
    previous = good_manifest(**{"index.html": "a" * 64, "old.html": "b" * 64})
    current = good_manifest(**{"index.html": "a" * 64})
    remote = {"index.html", "old.html", "somebody-elses.txt"}
    changed, deleted = plan_changes(previous, current, remote)
    assert deleted == ["old.html"]
    assert "somebody-elses.txt" not in deleted


def test_a_file_missing_remotely_is_reuploaded_even_if_unchanged():
    manifest = good_manifest(**{"index.html": "a" * 64})
    changed, _ = plan_changes(manifest, manifest, set())
    assert changed == ["index.html"]


# -- repo id pin ----------------------------------------------------------
@pytest.mark.parametrize("repo_id", [
    "someoneelse/resume-builder-agent", "Phoenix21/x/y", "../../etc", "",
])
def test_deployment_target_is_pinned_to_phoenix21(repo_id):
    assert not REPO_ID_PATTERN.fullmatch(repo_id)


def test_the_intended_target_is_allowed():
    assert REPO_ID_PATTERN.fullmatch("Phoenix21/resume-builder-agent")


# -- log hygiene ----------------------------------------------------------
def test_token_shaped_text_is_scrubbed():
    text = "failed with hf_" + "A1b2C3d4E5f6G7h8I9j0" + " while uploading"
    assert "hf_A1b2" not in scrub(text)
    assert "[redacted]" in scrub(text)


def test_env_secrets_are_scrubbed(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "a-very-secret-value")
    assert "a-very-secret-value" not in scrub("boom: a-very-secret-value")


# -- trace fidelity -------------------------------------------------------
def test_merge_accumulates_errors_instead_of_overwriting():
    """Regression: `state |= update` replaced the previous `errors` list, so the
    run trace reported "no bullets dropped" on a run that dropped five. A trace
    that quietly loses failures is worse than no trace."""
    from build_sample import merge

    state = {"errors": ["summarize: boom"]}
    state = merge(state, {"resumes": {}, "errors": ["dropped bullet — invented 9,999"]})
    state = merge(state, {"plan": [], "errors": []})
    assert state["errors"] == ["summarize: boom", "dropped bullet — invented 9,999"]


def test_merge_still_replaces_non_error_keys():
    from build_sample import merge

    state = merge({"claims": [1, 2]}, {"claims": [3]})
    assert state["claims"] == [3]


# -- the published sample -------------------------------------------------
def test_sample_meta_records_the_repositories_in_the_report():
    meta_path = ROOT / "space" / "assets" / "sample_meta.json"
    if not meta_path.is_file():
        pytest.skip("sample not generated yet")
    meta = json.loads(meta_path.read_text())
    assert meta["repos_in_report"], "the sample must record its repositories"
    assert meta["public_repos_considered"] >= len(meta["repos_in_report"])
