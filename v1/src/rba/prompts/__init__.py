"""Versioned prompt files.

Prompts are data, not code: the file name carries the version, and that version
is part of the response-cache key, so changing a prompt invalidates exactly the
cached responses it affects.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

PROMPT_DIR = Path(__file__).parent


@lru_cache(maxsize=32)
def load(name: str) -> tuple[str, str, int]:
    """Return (system, user_template, version) for a prompt file."""
    path = PROMPT_DIR / f"{name}.md"
    text = path.read_text()
    version_match = re.search(r"<!--\s*version:\s*(\d+)\s*-->", text)
    version = int(version_match.group(1)) if version_match else 1
    parts = re.split(r"^#\s+(system|user)\s*$", text, flags=re.M)
    sections: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i].strip()] = parts[i + 1].strip()
    if "system" not in sections or "user" not in sections:
        raise ValueError(f"{path} must contain '# system' and '# user' sections")
    return sections["system"], sections["user"], version


def render(name: str, **kwargs) -> tuple[list[dict[str, str]], int]:
    """Build the message list for a prompt, plus its version."""
    system, user, version = load(name)
    return ([{"role": "system", "content": system},
             {"role": "user", "content": user.format(**kwargs)}], version)
