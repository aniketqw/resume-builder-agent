"""Fail the build if the published sample names a repository it should not.

The sample ships on a *public* Space, and the account it is generated from has
private repositories. `build_sample.py` already filters those out before
anything is fetched, so this is the second gate: it proves the artefact on disk
only names repositories from the recorded public allowlist.

The check is deliberately **positive** — every repository the report names must
appear in `sample_meta.json` — rather than a denylist of private names. A
denylist would mean committing the private names to a public repository, which
is the very thing being avoided, and it would miss any name nobody thought of.

With a GitHub token present it additionally cross-checks against the live list
of private repositories, which is the strongest form of the check but needs
credentials CI does not have.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "space" / "assets" / "sample_report.md"
META = ROOT / "space" / "assets" / "sample_meta.json"

# Repo names appear in two structured places in a report: evidence-index
# headings, and the leading `repo` of a measure/refused bullet.
HEADING = re.compile(r"(?m)^###\s+([A-Za-z0-9][\w.\-]{1,70})\s*$")
BULLET_REPO = re.compile(r"(?m)^-\s+`([A-Za-z0-9][\w.\-]{1,70})`\s+—")
PLAN_REPO = re.compile(r"repo:\s*`([A-Za-z0-9][\w.\-]{1,70})`")

# Headings that are section titles, not repositories.
SECTION_WORDS = {
    "summary", "experience", "projects", "skills", "education", "measure",
    "refused", "evidence", "plan", "comparison", "security", "run",
}


def named_repos(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in (HEADING, BULLET_REPO, PLAN_REPO):
        for m in pattern.finditer(text):
            name = m.group(1)
            if name.lower() in SECTION_WORDS:
                continue
            # Numbered plan headings like "1. Add a diagram" are not repos.
            if name[0].isdigit():
                continue
            found.add(name)
    return found


def live_private_repos() -> set[str] | None:
    """Private repo names from gh, when credentials allow. None if unavailable."""
    if not os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") and not os.environ.get("GH_TOKEN"):
        try:
            subprocess.run(["gh", "auth", "status"], capture_output=True, check=True, timeout=20)
        except Exception:
            return None
    try:
        out = subprocess.run(
            ["gh", "repo", "list", "--limit", "300", "--visibility", "private",
             "--json", "name", "--jq", ".[].name"],
            capture_output=True, text=True, check=True, timeout=60).stdout
    except Exception:
        return None
    return {line.strip() for line in out.splitlines() if line.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--meta", default=str(META))
    args = ap.parse_args()

    report_path, meta_path = Path(args.report), Path(args.meta)
    if not report_path.is_file() or not meta_path.is_file():
        print(f"missing sample artefacts: {report_path} / {meta_path}")
        print("run: python scripts/build_sample.py")
        return 2

    text = report_path.read_text()
    meta = json.loads(meta_path.read_text())
    allowed = set(meta.get("repos_in_report") or [])
    if not allowed:
        print("sample_meta.json records no repositories")
        return 2

    named = named_repos(text)
    unexpected = sorted(named - allowed)

    print(f"sample names {len(named)} repositories; allowlist has {len(allowed)}")
    if unexpected:
        print("FAIL: names a repository outside the recorded public allowlist:")
        for name in unexpected:
            print(f"  - {name}")
        return 1

    private = live_private_repos()
    if private is None:
        print("note: no GitHub credentials, skipped the live private-repo cross-check")
    else:
        # Compare against names in *repository positions*, not free text. The
        # account has a private repo literally called "python", and the word
        # appears throughout any report about Python projects; a substring
        # search reports that as a leak every time.
        leaked = sorted(private & named)
        if leaked:
            print(f"FAIL: {len(leaked)} private repository name(s) present in the sample:")
            for name in leaked:
                print(f"  - {name}")
            return 1
        print(f"cross-checked against {len(private)} live private repositories: clean")

    print("PASS: the sample names only public repositories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
