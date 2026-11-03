"""BM25 lexical matching.

The dense channel is better at meaning; this one is better at *literals*. A
384-dimension embedding happily blurs "AWS" into "cloud" and "Kubernetes" into
"containers", which is exactly wrong when a job description says it requires
AWS. BM25 keeps certification codes, tool names and version strings sharp —
the TalentCLEF 2026 overview notes top systems paired dense encoders with BM25
for precisely this reason.

BM25 also solves the problem that made the old rubric arbitrary. Keyword
weights used to be hand-assigned (``langgraph: 3.0``, ``python: 1.0``). BM25's
IDF term derives them instead: a term appearing in one job description out of
many scores high, a term appearing in all of them scores low. The weighting
comes from the corpus rather than from my opinion.
"""
from __future__ import annotations

import math
import re

# Keep tokens that carry signal: c++, c#, .net, ci/cd, node.js, 3.11.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#./_-]*")

# Deliberately short. An aggressive stop list removes "r" and "go", which are
# languages, and BM25's IDF already discounts common words far better than a
# hand-written list does.
_STOP = frozenset("""
a an the and or of to in for on with at by from as is are be been was were
this that these those it its we our you your they their he she his her
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase tokens, keeping internal punctuation but not trailing.

    "node.js" and "ci/cd" must survive intact, but "aws." at the end of a
    sentence has to become "aws" or it will never match "AWS" elsewhere.
    """
    out: list[str] = []
    for raw in _TOKEN.findall((text or "").lower()):
        token = raw.rstrip("./-_")
        if token and token not in _STOP:
            out.append(token)
    return out


def idf(term: str, corpus_tokens: list[list[str]]) -> float:
    """Robertson/Sparck-Jones IDF with the usual +0.5 smoothing.

    Floored at zero: a term present in more than half the corpus can otherwise
    go negative, which would let a very common word *subtract* from a score.
    """
    n = len(corpus_tokens)
    if n == 0:
        return 0.0
    df = sum(1 for doc in corpus_tokens if term in doc)
    return max(0.0, math.log(1.0 + (n - df + 0.5) / (df + 0.5)))


def bm25_score(query_tokens: list[str], doc_tokens: list[str],
               corpus_tokens: list[list[str]], k1: float = 1.5,
               b: float = 0.75) -> float:
    """Unnormalised BM25 of one document against one query."""
    if not query_tokens or not doc_tokens:
        return 0.0
    avgdl = (sum(len(d) for d in corpus_tokens) / len(corpus_tokens)
             if corpus_tokens else len(doc_tokens))
    if avgdl <= 0:
        return 0.0
    doc_len = len(doc_tokens)
    counts: dict[str, int] = {}
    for token in doc_tokens:
        counts[token] = counts.get(token, 0) + 1

    total = 0.0
    for term in set(query_tokens):
        freq = counts.get(term, 0)
        if not freq:
            continue
        denominator = freq + k1 * (1 - b + b * doc_len / avgdl)
        total += idf(term, corpus_tokens) * (freq * (k1 + 1)) / denominator
    return total


def bm25_similarity(resume_text: str, jd_text: str,
                    corpus: list[str] | None = None) -> float:
    """BM25 of the resume against the job description, normalised to [0, 1].

    Normalisation is theoretical, not observed: the score is divided by the
    ceiling the resume would reach if it contained every query term at
    saturating frequency. Observed min-max is undefined for a single document,
    which is the case this tool is actually used in.
    """
    query = tokenize(jd_text)
    doc = tokenize(resume_text)
    if not query or not doc:
        return 0.0

    corpus_tokens = [tokenize(c) for c in corpus] if corpus else [doc, query]
    raw = bm25_score(query, doc, corpus_tokens)

    # Ceiling = the job description scored against itself. That is the best any
    # document could do lexically, and it keeps the ratio interpretable: 1.0
    # means "contains the job description's vocabulary as densely as the job
    # description does".
    #
    # An earlier version used a theoretical ceiling assuming every query term
    # present at saturating frequency. That included filler like "we", "need"
    # and "engineer", so a near-perfect resume scored 0.12 and the channel was
    # effectively dead weight in the fusion.
    ceiling = bm25_score(query, query, corpus_tokens)
    if ceiling <= 0:
        return 0.0
    return max(0.0, min(1.0, raw / ceiling))


def matched_terms(resume_text: str, jd_text: str) -> tuple[list[str], list[str]]:
    """Which job-description terms the resume does and does not contain.

    This is the auditable part: "14 of 18 required terms present" is a claim a
    human can check in seconds, unlike "38 of 44 points".
    """
    doc = set(tokenize(resume_text))
    query = tokenize(jd_text)
    seen: set[str] = set()
    present: list[str] = []
    missing: list[str] = []
    for term in query:
        if term in seen or len(term) < 2:
            continue
        seen.add(term)
        (present if term in doc else missing).append(term)
    return present, missing
