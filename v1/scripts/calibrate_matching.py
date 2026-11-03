"""Measure whether project-to-repository matching can be trusted automatically.

The answer, on a real account, is **no** -- and the thresholds in
`rba.resume.matching` exist because of what this prints, not the other way
round. Run it against your own account before trusting any auto-match:

    python scripts/calibrate_matching.py --user aniketqw
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import httpx

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from rba.resume.ingest import parse_bits
from rba.resume.matching import match_projects
from rba.state import RepoRef

# Established by reading the repositories, not by running the matcher.
GROUND_TRUTH = {
    "Live Debugging Partner": "live-debugging-partner",
    "Sandboxed Code-Execution Agent": "sandbox-agent",
    "CodeAct Fine-tuning with Execution Feedback": "CodeActImplementation",
    "VisionDev": "vision_dev_project",
    # Deliberately absent: these have no public repository, and the matcher
    # must NOT confidently attach them to anything.
    "Multi-University Bounty Platform": None,
    "Resume Builder Agent": None,
}


def load_env() -> None:
    env = pathlib.Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def fetch_repos(user: str, token: str) -> list[RepoRef]:
    out: list[RepoRef] = []
    for page in (1, 2, 3):
        resp = httpx.get(
            f"https://api.github.com/users/{user}/repos?per_page=100&page={page}",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=30)
        data = resp.json()
        if not isinstance(data, list) or not data:
            break
        out += [RepoRef(name=r["name"], description=r.get("description") or "",
                        language=r.get("language") or "")
                for r in data if not r.get("fork")]
    return out


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=os.environ.get("GITHUB_USERNAME", "aniketqw"))
    ap.add_argument("--fixtures", nargs="*",
                    default=["aniket_agents_clean", "aniket_bits",
                             "aniket_agents_bits"])
    args = ap.parse_args()

    repos = fetch_repos(args.user, os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", ""))
    print(f"{len(repos)} non-fork repositories for {args.user}\n")

    root = pathlib.Path(__file__).resolve().parent.parent / "fixtures"
    rows = []
    for name in args.fixtures:
        doc = parse_bits((root / f"{name}.txt").read_text())
        for match in match_projects(doc, repos, min_confidence=0.0):
            truth = GROUND_TRUTH.get(match.project_title, "unknown")
            rows.append((match, truth))

    print(f"{'verdict':<10}{'conf':<8}{'margin':<9}project -> repo")
    for match, truth in rows:
        if truth == "unknown":
            verdict = "-"
        elif truth is None:
            verdict = "FALSE" if match.strong else "ok(ask)"
        else:
            verdict = "TRUE" if match.repo == truth else "WRONG"
        print(f"{verdict:<10}{match.confidence:<8.3f}{match.margin:<9.3f}"
              f"{match.project_title[:36]} -> {match.repo or '(none)'}")

    correct = [m for m, t in rows if t not in (None, "unknown") and m.repo == t]
    absent = [m for m, t in rows if t is None]
    print()
    if correct:
        print(f"true matches       confidence {min(m.confidence for m in correct):.3f}"
              f"-{max(m.confidence for m in correct):.3f}   margin "
              f"{min(m.margin for m in correct):.3f}-{max(m.margin for m in correct):.3f}")
    if absent:
        print(f"no repo exists     confidence {min(m.confidence for m in absent):.3f}"
              f"-{max(m.confidence for m in absent):.3f}   margin "
              f"{min(m.margin for m in absent):.3f}-{max(m.margin for m in absent):.3f}")
    false_accepts = [m for m, t in rows if t is None and m.strong]
    missed = [m for m, t in rows if t not in (None, "unknown")
              and m.repo == t and not m.strong]
    print(f"\nauto-accepted wrongly: {len(false_accepts)}  "
          f"(this must be 0 -- a wrong repository verifies real claims "
          f"against unrelated evidence)")
    print(f"true matches sent for confirmation: {len(missed)}  "
          f"(acceptable: asking costs a click, a wrong match costs the résumé)")
    return 1 if false_accepts else 0


if __name__ == "__main__":
    raise SystemExit(main())
