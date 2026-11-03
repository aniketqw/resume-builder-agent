"""Flag credentials and private data committed into the scanned repositories.

This runs over the *user's* repos, not this project. It is often the most
immediately useful thing a first run produces: a committed ``.env`` or
``token.txt`` is a live exposure, and it is easy to forget one is there.

Detection is in two parts. Path rules catch files that should never be in a
repository at all. Content rules catch credentials pasted into files that are
otherwise fine. Only bounded content already fetched is scanned — no repo is
downloaded to search it, and no matched secret value is ever logged or stored.
"""
from __future__ import annotations

import logging
import re

from ..state import AgentState, SafetyFinding

log = logging.getLogger(__name__)

# Files whose presence in a repository is itself the finding.
PATH_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"(^|/)\.env(\.|$)(?!sample|example|template)"), "env-file",
     "An .env file is committed. These normally hold live credentials."),
    (re.compile(r"(^|/)(token|tokens|secret|secrets|credentials?|password)s?\.(txt|json|ya?ml)$", re.I),
     "credential-file", "A file named for credentials is committed."),
    (re.compile(r"(^|/)cookies?\.(txt|json)$", re.I), "cookie-file",
     "Browser cookies are committed. These are session credentials."),
    (re.compile(r"(^|/)id_(rsa|dsa|ecdsa|ed25519)$"), "private-key",
     "An SSH private key is committed."),
    (re.compile(r"\.(pem|pfx|p12|keystore|jks)$", re.I), "private-key",
     "A certificate or key store is committed."),
    (re.compile(r"(^|/)(serviceaccount|service-account|gcp-key|firebase-adminsdk)[^/]*\.json$", re.I),
     "cloud-credential", "A cloud service-account key is committed."),
    (re.compile(r"(^|/)\.netrc$|(^|/)\.pypirc$|(^|/)\.npmrc$"), "credential-file",
     "A tool config that commonly stores auth tokens is committed."),
    (re.compile(r"(^|/)(Default|Profile \d+)/(Cookies|Login Data|Web Data)$"),
     "browser-profile", "A browser profile directory is committed."),
]

# Credential shapes. Each pattern is specific enough to avoid matching prose.
CONTENT_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bgsk_[A-Za-z0-9]{40,}"), "groq-key", "high"),
    (re.compile(r"\blsv2_(pt|sk)_[A-Za-z0-9]{20,}"), "langsmith-key", "high"),
    # Anthropic first: `sk-ant-...` also satisfies the broader OpenAI shape,
    # so the more specific pattern has to win.
    (re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{30,}"), "anthropic-key", "high"),
    (re.compile(r"\bsk-(?!ant-)(proj-)?[A-Za-z0-9_\-]{30,}"), "openai-key", "high"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"), "github-token", "high"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "aws-access-key", "high"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"), "google-api-key", "high"),
    (re.compile(r"\bxox[baprs]-[0-9A-Za-z\-]{10,}"), "slack-token", "high"),
    (re.compile(r"-----BEGIN (RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----"), "private-key", "high"),
    (re.compile(r"\bey[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
     "jwt", "medium"),
    (re.compile(r"(?i)\b(api[_\-]?key|secret|password|passwd|token)\s*[:=]\s*"
                r"['\"][^'\"\s]{12,}['\"]"), "hardcoded-secret", "medium"),
]

# Placeholders people write in examples. Not findings.
PLACEHOLDER = re.compile(
    r"(?i)(your[_\-]?|my[_\-]?|example|sample|dummy|placeholder|changeme|xxxx+|<[^>]+>|"
    r"\*\*\*\*|insert[_\-]?|f+a+k+e+|test[_\-]?key|abc123|0{8,}|replace[_\-]?me)")


def scan_paths(repo: str, paths: list[str]) -> list[SafetyFinding]:
    out = []
    for p in paths:
        for pattern, kind, detail in PATH_RULES:
            if pattern.search(p):
                out.append(SafetyFinding(repo=repo, path=p, kind=kind,
                                         detail=detail, severity="high"))
                break
    return out


def scan_content(repo: str, path: str, text: str) -> list[SafetyFinding]:
    out = []
    seen: set[str] = set()
    for pattern, kind, severity in CONTENT_RULES:
        for m in pattern.finditer(text or ""):
            hit = m.group(0)
            if PLACEHOLDER.search(hit):
                continue
            if kind in seen:
                break
            seen.add(kind)
            line = text.count("\n", 0, m.start()) + 1
            out.append(SafetyFinding(
                repo=repo, path=f"{path}:{line}", kind=kind,
                # Record the shape and location, never the value itself.
                detail=f"Looks like a {kind.replace('-', ' ')} "
                       f"({_redact(hit)}). Rotate it and purge it from history.",
                severity=severity))
            break
    return out


def _redact(secret: str) -> str:
    """Show only enough to locate the string, never enough to use it."""
    if len(secret) <= 10:
        return secret[:3] + "…"
    return f"{secret[:6]}…{secret[-2:]} [{len(secret)} chars]"


async def safety_scan(state: AgentState) -> AgentState:
    findings: list[SafetyFinding] = []
    for name, facts in (state.get("facts") or {}).items():
        findings += scan_paths(name, facts.tree)
        if facts.readme:
            findings += scan_content(name, facts.readme_path or "README.md", facts.readme)
        for fname, content in facts.key_files.items():
            findings += scan_content(name, fname, content)

    if findings:
        high = sum(1 for f in findings if f.severity == "high")
        log.warning("safety scan: %d findings (%d high) across %d repos",
                    len(findings), high, len({f.repo for f in findings}))
    return {"safety": findings}
