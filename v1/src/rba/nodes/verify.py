"""Claim verification. Code, not a prompt.

A prompt asking a model to "only state supported facts" is a request. This
module is the enforcement. Every claim is decomposed mechanically and checked
against the exact evidence text the model was shown:

* **Numbers** are the highest-risk invention and get the strictest rule. Every
  numeric token in a claim must appear in the evidence. A claim saying "reduced
  latency by 40%" when the evidence contains no 40 is rejected outright, and
  reappears in the plan as a `[MEASURE: ...]` task.
* **Technologies** are checked against the summary's declared ``tech`` list,
  not against arbitrary words in claim prose. Scanning prose for
  "technology-shaped" tokens rejected 39 of 48 real claims in a live run over
  words like "fallback." and "scripts." — the punctuation made them look like
  package names. Unsupported entries are dropped from ``tech``; they do not
  invalidate the claim that happened to mention them.
* **Quotes** the model attributes to the evidence must actually be in it.
  A fabricated quote invalidates the claim it supports.

Rejected claims are not deleted. They become measurement tasks, which is the
useful form of "we could not prove this".
"""
from __future__ import annotations

import logging
import re

from ..state import AgentState, Claim, ClaimStatus

log = logging.getLogger(__name__)

# Numbers that carry no factual weight: years, and version strings. Checking
# these produces false rejections without catching any real invention.
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_DOTTED = re.compile(r"\b(v?)(\d+(?:\.\d+)+)\b")

# Words that introduce a measurement. A decimal after one of these is a metric,
# never a version: "accuracy from 0.81 to 0.92" is exactly what this module
# exists to police.
_MEASURE_LEAD = {
    "from", "to", "of", "at", "by", "is", "was", "reached", "scored", "rate",
    "accuracy", "precision", "recall", "f1", "auc", "score", "latency", "than",
    "under", "over", "above", "below", "around", "about", "approximately",
}


def _is_version(prefix: str, number: str, preceding_word: str) -> bool:
    """Does this dotted number look like a version rather than a measurement?"""
    if prefix:                       # explicit "v2.1"
        return True
    if preceding_word.lower() in _MEASURE_LEAD:
        return False                 # "from 0.81", "of 49.8"
    if number.count(".") >= 2:       # 3.1.5 is a version in any context
        return True
    # "Python 3.11", "Spring Boot 3.1" — a capitalised name right before it.
    return bool(preceding_word) and preceding_word[:1].isupper()


def mask_versions(text: str) -> str:
    """Blank out version strings so they are not mistaken for metrics."""
    out = []
    last = 0
    for m in _DOTTED.finditer(text):
        before = text[:m.start()].rstrip()
        preceding_word = re.split(r"[\s(\[]", before)[-1] if before else ""
        out.append(text[last:m.start()])
        out.append(" " if _is_version(m.group(1), m.group(2), preceding_word)
                   else m.group(0))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


# Comparative language that turns a sentence into a performance claim — the kind
# a resume reader believes and an interviewer probes.
#
# Deliberately narrow. An earlier version included bare nouns like "performance",
# "speed" and "saving", which flagged the perfectly ordinary claim "MongoDB is
# used for saving and searching documents" as an unmeasured effect. Only
# comparatives and explicit metric names belong here.
_IMPACT = re.compile(
    r"(?i)(\b(reduc\w+|improv\w+|increas\w+|decreas\w+|accelerat\w+|optimis\w+|"
    r"optimiz\w+|outperform\w*|speed(?:s|ed)?\s+up|scal\w+\s+to|"
    r"faster|slower|cheaper|higher|lower|better\s+than|more\s+than|less\s+than)\b"
    r"|%|\bpercent\b"
    r"|\b(accurac\w+|precision|recall|f1|auc|latenc\w+|throughput|uptime|"
    r"p9[59]|qps|rps)\b)")


_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "using", "used", "use",
    "repository", "repo", "project", "code", "author", "implements", "implemented",
    "provides", "includes", "contains", "built", "build", "supports", "based",
    "which", "into", "each", "across", "within", "their", "there", "when",
    "library", "framework", "application", "app", "via", "run", "runs",
}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9.+#/_-]+", " ", text.lower())


def numbers_in(text: str) -> list[str]:
    """Numeric tokens worth verifying, with versions and years excluded."""
    masked = _YEAR.sub(" ", mask_versions(text))
    out = []
    for m in _NUMBER.finditer(masked):
        token = m.group(0).rstrip(".,")
        if token and token not in out:
            out.append(token)
    return out


def _number_supported(token: str, evidence: str) -> bool:
    """A number counts as supported if it appears in the evidence."""
    bare = token.replace(",", "")
    for candidate in {token, bare, bare.rstrip("0").rstrip(".") if "." in bare else bare}:
        if candidate and re.search(rf"(?<![\d.]){re.escape(candidate)}(?![\d])", evidence):
            return True
    return False


def is_impact_claim(text: str) -> bool:
    """Does this claim assert a measured effect? Those need the strictest proof."""
    return bool(_IMPACT.search(text)) and bool(numbers_in(text))


def quote_supported(quote: str, evidence: str, min_overlap: float = 0.6) -> bool:
    """Is this quote really in the evidence?

    Exact containment first. Models reformat whitespace, so a token-overlap
    fallback avoids rejecting an otherwise honest quote over a line break.
    """
    q = normalise(quote).split()
    if not q:
        return False
    if normalise(quote).strip() in normalise(evidence):
        return True
    ev_tokens = set(normalise(evidence).split())
    hits = sum(1 for t in q if t in ev_tokens)
    return hits / len(q) >= min_overlap


def quote_relevant(claim_text: str, quote: str, path: str = "",
                   min_shared: int = 1) -> bool:
    """Does this quote support *this* claim, or merely exist?

    Presence alone is not support. A live run produced the claim "uses the
    tcell library for terminal UI" attributed to the quote
    "github.com/goccy/go-json" — a real line from go.mod, and entirely
    unrelated. Requiring shared content words catches that without punishing
    honest paraphrase, since any real citation repeats at least one
    distinctive term.
    """
    claim_words = {w for w in normalise(claim_text).split()
                   if len(w) >= 3 and w not in _STOPWORDS}
    quote_words = {w for w in normalise(f"{quote} {path}").split() if len(w) >= 3}
    if not claim_words:
        return True
    # Dotted paths ("github.com/gin-gonic/gin") carry their parts as terms too.
    for w in list(quote_words):
        quote_words.update(re.split(r"[./_-]", w))
    shared = {w for w in claim_words
              if w in quote_words or any(w in q for q in quote_words if len(q) > 4)}
    return len(shared) >= min_shared


def verify_claim(claim: Claim, evidence: str) -> Claim:
    """Decide a single claim's status. Pure function, fully testable."""
    unsupported_numbers = [n for n in numbers_in(claim.text)
                           if not _number_supported(n, evidence)]
    if unsupported_numbers:
        claim.status = ClaimStatus.NEEDS_MEASURE
        claim.reason = ("contains number(s) absent from the evidence: "
                        + ", ".join(unsupported_numbers))
        claim.measure_hint = _measure_hint(claim.text)
        return claim

    # A performance claim with no quantity is an opinion, not a claim.
    if _IMPACT.search(claim.text) and not numbers_in(claim.text):
        claim.status = ClaimStatus.NEEDS_MEASURE
        claim.reason = "asserts an effect without a measured quantity"
        claim.measure_hint = _measure_hint(claim.text)
        return claim

    if not claim.evidence:
        claim.status = ClaimStatus.REJECTED
        claim.reason = "no evidence supplied"
        return claim

    present = [e for e in claim.evidence if quote_supported(e.quote, evidence)]
    if not present:
        claim.status = ClaimStatus.REJECTED
        claim.reason = "every supporting quote is absent from the evidence"
        return claim

    good = [e for e in present if quote_relevant(claim.text, e.quote, e.path)]
    if not good:
        claim.status = ClaimStatus.REJECTED
        claim.reason = ("supporting quote is real but does not mention anything "
                        "in the claim")
        return claim
    claim.evidence = good

    claim.status = ClaimStatus.VERIFIED
    claim.reason = ""
    return claim


def filter_tech(tech: list[str], evidence: str) -> tuple[list[str], list[str]]:
    """Split declared technologies into (supported, unsupported).

    This is the right place for the technology check: the model states these
    explicitly, so there is no guessing about which words were meant as
    technology names.
    """
    ev = normalise(evidence)
    ev_tokens = set(ev.split())
    keep, drop = [], []
    for name in tech:
        norm = normalise(name).strip()
        if not norm:
            continue
        parts = norm.split()
        supported = (
            norm in ev
            or all(p in ev_tokens for p in parts)
            # "TransNet V2" in the claim vs "transnet" in a filename.
            or any(len(p) > 3 and p in ev for p in parts)
        )
        (keep if supported else drop).append(name)
    return keep, drop


def _measure_hint(text: str) -> str:
    low = text.lower()
    if "test" in low or "coverage" in low:
        return "pytest --cov -q  # record the real test count and coverage"
    if "latenc" in low or "speed" in low or "faster" in low or "throughput" in low:
        return "time the operation before and after; record both numbers"
    if "accurac" in low or "f1" in low or "precision" in low or "recall" in low:
        return "run the evaluation set and record the metric"
    if "user" in low or "request" in low or "traffic" in low:
        return "pull the real count from logs or analytics"
    return "measure this and record the command and result"


async def verify_claims(state: AgentState) -> AgentState:
    summaries = state.get("summaries") or {}
    evidence_texts = state.get("evidence_texts") or {}
    verified: list[Claim] = []

    dropped_tech: list[str] = []
    for name, summary in summaries.items():
        evidence = evidence_texts.get(name, "")
        checked = [verify_claim(c, evidence) for c in summary.claims]
        summary.claims = checked
        summary.tech, unsupported = filter_tech(summary.tech, evidence)
        dropped_tech += [f"{name}:{t}" for t in unsupported]
        verified.extend(checked)

    if dropped_tech:
        log.info("dropped %d unsupported technology names: %s",
                 len(dropped_tech), ", ".join(dropped_tech[:8]))

    counts = {s: sum(1 for c in verified if c.status is s) for s in ClaimStatus}
    total = len(verified)
    if total:
        log.info("verified %d/%d claims (%d need measurement, %d rejected)",
                 counts[ClaimStatus.VERIFIED], total,
                 counts[ClaimStatus.NEEDS_MEASURE], counts[ClaimStatus.REJECTED])
    return {"claims": verified, "summaries": summaries}
