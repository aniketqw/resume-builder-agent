"""Stage an allowlisted Space bundle — never the repository.

The Space is public; this repository is not. So the bundle is built from an
explicit **allowlist** rather than by excluding things: anything not named here
cannot reach the Hub, which is the safe direction for that mistake. Tests,
evaluation fixtures, `.env`, cached state and the private-inclusive `out/`
report are all absent by construction.

Every staged text file is additionally scanned for credential-shaped literals,
behind the repository's own `scripts/scan_secrets.sh`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWNER = "aniketqw/resume-builder-agent"
MANIFEST = ".rba-manifest.json"

# The complete set of files that may reach the Space. Add deliberately.
ALLOWLIST = {
    "README.md": "space/README.md",
    "index.html": "space/index.html",
    "style.css": "space/style.css",
}

CARD_COLORS = {"red", "yellow", "green", "blue", "indigo", "purple", "pink", "gray"}
CREDENTIAL = re.compile(r"\b(?:hf|gsk|lsv2_pt|ghp|gho|sk-ant)[_-][A-Za-z0-9_-]{16,}")
TEXT_SUFFIXES = {".md", ".html", ".css", ".js", ".json", ".txt", ".py"}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_card(text: str) -> None:
    """A Space card must declare valid frontmatter or the build fails on the Hub."""
    parts = text.split("---", 2)
    if len(parts) < 3 or parts[0].strip():
        raise ValueError("space/README.md must begin with YAML frontmatter")
    front = parts[1]
    if not re.search(r"(?m)^sdk:\s*static\s*$", front):
        raise ValueError("space/README.md must declare 'sdk: static'")
    if not re.search(r"(?m)^app_file:\s*index\.html\s*$", front):
        raise ValueError("space/README.md must declare 'app_file: index.html'")
    for field in ("colorFrom", "colorTo"):
        found = re.search(rf"(?m)^{field}:\s*(\w+)\s*$", front)
        if found and found.group(1) not in CARD_COLORS:
            raise ValueError(f"unsupported Hugging Face card colour for {field}: "
                             f"{found.group(1)}")


def build(repo_root: Path, output: Path) -> dict:
    repo_root, output = Path(repo_root).resolve(), Path(output).resolve()
    if output == repo_root or output.is_relative_to(repo_root):
        raise ValueError("choose a staging directory outside the source repository")
    if output.exists() and any(output.iterdir()):
        raise ValueError("staging directory must be new or empty")

    sources = {name: repo_root / rel for name, rel in ALLOWLIST.items()}
    missing = [rel for name, rel in ALLOWLIST.items() if not (repo_root / rel).is_file()]
    if missing:
        raise ValueError("missing Space files: " + ", ".join(sorted(missing))
                         + " — run scripts/build_sample.py then scripts/build_site.py")

    validate_card(sources["README.md"].read_text(encoding="utf-8"))

    for name, source in sources.items():
        if source.is_symlink():
            raise ValueError(f"symlinks are not allowed in the bundle: {name}")
        if source.suffix in TEXT_SUFFIXES:
            hit = CREDENTIAL.search(source.read_text(encoding="utf-8"))
            if hit:
                raise ValueError(f"credential-shaped literal in {name}: "
                                 f"{hit.group(0)[:8]}…")

    output.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    manifest = {"managed_by": OWNER,
                "files": {name: sha256(output / name) for name in sorted(sources)}}
    (output / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n",
                                   encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    manifest = build(ROOT, Path(args.output))
    print(f"staged {len(manifest['files'])} files: {', '.join(sorted(manifest['files']))}")
    print("source, tests, evaluation fixtures and .env are excluded by construction")
    return 0


if __name__ == "__main__":
    sys.exit(main())
