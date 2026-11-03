"""Publish the staged bundle in one commit, with selective deletion.

Adapted from the deployment used by `lecturelens-research`. The safety
properties that matter:

* the target repo id is **pinned** to a Phoenix21 Space, and the token must
  authenticate as that account;
* it refuses to touch a Space holding files it does not own, so it can never
  overwrite an unrelated app;
* deletions are limited to files the *previous manifest* owned;
* provider errors are re-raised without their HTTP bodies, and any token-shaped
  text is scrubbed, so nothing leaks into Actions logs.

Unlike the lecturelens original this never calls `add_space_secret`. The agent
needs a GitHub token to read repositories, and the account it was built for has
private ones — a server-side credential here would expose them to every
visitor. There is nothing to configure because the Space runs nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_space import MANIFEST, OWNER, sha256

REPO_ID_PATTERN = re.compile(r"Phoenix21/[A-Za-z0-9][A-Za-z0-9._-]*")
TOKEN_SHAPED = re.compile(r"\b(?:hf|gsk|lsv2_pt|ghp|gho)[_-][A-Za-z0-9_-]+")


def scrub(text: str) -> str:
    for key in ("HF_TOKEN", "GROQ_API_KEY", "GITHUB_PERSONAL_ACCESS_TOKEN"):
        value = os.environ.get(key)
        if value:
            text = text.replace(value, "[redacted]")
    return TOKEN_SHAPED.sub("[redacted]", text)


def api_step(label, call):
    """Run one Hub call, reporting failures without their response bodies."""
    print(label + "...", flush=True)
    try:
        return call()
    except Exception as exc:
        from huggingface_hub.errors import RepositoryNotFoundError
        if isinstance(exc, RepositoryNotFoundError):
            raise
        status = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f"HTTP {status}" if status else type(exc).__name__
        if status == 402:
            detail += (": the Hub refused on billing grounds. Static Spaces are "
                       "free; Gradio and Docker Spaces need a paid plan.")
        elif isinstance(exc, ValueError):
            detail += ": " + scrub(str(exc))[:500].replace("\n", " ")
        raise ValueError(
            f"Hugging Face failed while {label.lower()} ({detail}). Check the "
            f"token's Space permissions and account eligibility.") from None


def validate_manifest(value: dict) -> dict:
    if (not isinstance(value, dict) or value.get("managed_by") != OWNER
            or not isinstance(value.get("files"), dict)):
        raise ValueError("the Space manifest does not belong to this deployment")
    for path, digest in value["files"].items():
        parts = PurePosixPath(path).parts
        if (not isinstance(path, str) or PurePosixPath(path).is_absolute()
                or ".." in parts):
            raise ValueError(f"unsafe path in deployment manifest: {path}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid hash in deployment manifest for {path}")
    return value


def plan_changes(previous, current, remote_files):
    """Which files to upload, and which of *ours* to remove."""
    validate_manifest(current)
    if previous is not None:
        validate_manifest(previous)
    old = previous["files"] if previous else {}
    changed = [p for p, digest in current["files"].items()
               if old.get(p) != digest or p not in remote_files]
    deleted = [p for p in old if p not in current["files"] and p in remote_files]
    return sorted(changed), sorted(deleted)


def deploy(folder: str, repo_id: str) -> int:
    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationDelete,
        HfApi,
        hf_hub_download,
    )
    from huggingface_hub.errors import RepositoryNotFoundError

    folder = Path(folder).resolve()
    current = validate_manifest(json.loads((folder / MANIFEST).read_text()))
    for path, digest in current["files"].items():
        if sha256(folder / path) != digest:
            raise ValueError(f"staged file changed after validation: {path}")
    if not REPO_ID_PATTERN.fullmatch(repo_id):
        raise ValueError("this deployment targets only a Space owned by Phoenix21")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("configure the HF_TOKEN encrypted GitHub Actions secret")
    api = HfApi(token=token)
    identity = api_step("Authenticating the Hugging Face account", api.whoami)
    if identity.get("name", "").lower() != "phoenix21":
        raise ValueError("HF_TOKEN must authenticate as Phoenix21")

    exists = True
    try:
        info = api_step("Reading the target Space", lambda: api.space_info(repo_id))
        if info.sdk != "static":
            raise ValueError(f"the target Space is '{info.sdk}', not static; "
                             f"it will not be overwritten")
    except RepositoryNotFoundError:
        exists = False

    remote_files = set(api.list_repo_files(repo_id, repo_type="space")) if exists else set()
    previous = None
    if MANIFEST in remote_files:
        path = hf_hub_download(repo_id=repo_id, repo_type="space",
                               filename=MANIFEST, token=token)
        previous = validate_manifest(json.loads(Path(path).read_text()))
    elif remote_files - {"README.md", ".gitattributes", "index.html", "style.css"}:
        raise ValueError("the existing Space holds unmanaged files; refusing to "
                         "replace another app")

    if not exists:
        api_step("Creating the public static Space",
                 lambda: api.create_repo(repo_id, repo_type="space",
                                         space_sdk="static", private=False))

    changed, deleted = plan_changes(previous, current, remote_files)
    if not changed and not deleted and previous == current:
        print("No files changed; the Space does not need a rebuild.")
        _export_commit(api, repo_id)
        return 0

    operations = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(folder / p))
                  for p in changed]
    operations += [CommitOperationDelete(path_in_repo=p) for p in deleted]
    operations.append(CommitOperationAdd(path_in_repo=MANIFEST,
                                         path_or_fileobj=str(folder / MANIFEST)))
    source = os.environ.get("SOURCE_COMMIT", "manual")[:12]
    commit = api_step("Publishing the bundle", lambda: api.create_commit(
        repo_id=repo_id, repo_type="space", operations=operations,
        commit_message=f"Deploy from GitHub {source}"))

    print(f"Published {len(changed)} changed file(s), removed {len(deleted)}.")
    print(f"Space:  https://huggingface.co/spaces/{repo_id}")
    print(f"Commit: {commit.oid}")
    _write_github_env("HF_DEPLOY_COMMIT", commit.oid)
    return 0


def _export_commit(api, repo_id: str) -> None:
    try:
        _write_github_env("HF_DEPLOY_COMMIT", api.space_info(repo_id).sha)
    except Exception:
        pass


def _write_github_env(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_ENV")
    if path and value:
        with open(path, "a") as handle:
            handle.write(f"{key}={value}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--repo-id", default="Phoenix21/resume-builder-agent")
    args = ap.parse_args()
    try:
        return deploy(args.folder, args.repo_id)
    except ValueError as exc:
        raise SystemExit(scrub(str(exc))) from None
    except Exception as exc:
        raise SystemExit(
            f"Hugging Face deployment failed ({type(exc).__name__}). Check token "
            f"permissions, account eligibility, Space ownership and network access."
        ) from None


if __name__ == "__main__":
    sys.exit(main())
