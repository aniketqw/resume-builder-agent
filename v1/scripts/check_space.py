"""Verify the Space actually serves the page we just deployed.

"The upload succeeded" is not the same as "the Space works". This waits until
the Hub reports the Space RUNNING **at the commit we deployed** — not at some
earlier state that happened to be healthy — then fetches the page over HTTP and
asserts it contains the content we expect. Build logs are printed on failure,
which is the only way to diagnose a Space from CI.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

FAILED_STAGES = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "PAUSED",
                 "DELETING", "STOPPED"}
HEALTH = Path("space-health.json")


def print_logs(api, repo_id: str, build: bool) -> None:
    kind = "build" if build else "run"
    try:
        logs = api.fetch_space_logs(repo_id=repo_id, build=build)
        text = logs if isinstance(logs, str) else "".join(logs)
        print(f"--- last {kind} logs ---", flush=True)
        print("\n".join(text.splitlines()[-60:]), flush=True)
    except Exception as exc:
        print(f"(could not fetch {kind} logs: {type(exc).__name__})", flush=True)


def check(repo_id: str, expect: list[str], timeout: int = 600) -> int:
    import httpx
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    deployed = os.environ.get("HF_DEPLOY_COMMIT", "")
    deadline = time.monotonic() + timeout
    stage = "UNKNOWN"

    while time.monotonic() < deadline:
        info = api.space_info(repo_id)
        runtime = info.runtime
        stage = getattr(runtime, "stage", None) or "UNKNOWN"
        at_commit = (not deployed) or info.sha == deployed
        print(f"stage={stage} sha={str(info.sha)[:8]} "
              f"{'(deployed)' if at_commit else '(waiting for our commit)'}",
              flush=True)

        if stage in FAILED_STAGES:
            print_logs(api, repo_id, build=stage == "BUILD_ERROR")
            HEALTH.write_text(json.dumps(
                {"repo_id": repo_id, "stage": stage, "healthy": False}, indent=2) + "\n")
            raise RuntimeError(f"the Space did not start: {stage}")

        if stage == "RUNNING" and at_commit:
            host = f"https://{repo_id.replace('/', '-').lower()}.static.hf.space"
            try:
                response = httpx.get(host, timeout=30, follow_redirects=True)
                body = response.text
            except Exception as exc:
                print(f"page not reachable yet ({type(exc).__name__}); retrying",
                      flush=True)
                time.sleep(10)
                continue
            missing = [token for token in expect if token not in body]
            healthy = response.status_code == 200 and not missing
            report = {"repo_id": repo_id, "stage": stage, "url": host,
                      "status_code": response.status_code,
                      "bytes": len(body), "missing_content": missing,
                      "healthy": healthy, "commit": info.sha}
            HEALTH.write_text(json.dumps(report, indent=2) + "\n")
            if not healthy:
                print(f"page served {response.status_code} but is missing: {missing}",
                      flush=True)
                raise RuntimeError("the Space is running but served unexpected content")
            print(f"OK: {host} served {len(body):,} bytes containing all "
                  f"{len(expect)} expected markers", flush=True)
            return 0
        time.sleep(10)

    HEALTH.write_text(json.dumps(
        {"repo_id": repo_id, "stage": stage, "healthy": False}, indent=2) + "\n")
    raise RuntimeError(f"timed out waiting for the Space; last stage: {stage}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default="Phoenix21/resume-builder-agent")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--expect", action="append", default=None,
                    help="Substring that must appear in the served page")
    args = ap.parse_args()
    expect = args.expect or ["Resume Builder Agent", "Run trace", "claims verified"]
    try:
        return check(args.repo_id, expect, args.timeout)
    except Exception as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    sys.exit(main())
