"""Opt-in scoring fixes, found by benchmarking the default scorer.

NOTHING in the existing code imports this module. `score.py`, `batch.py`,
`fusion.py` and `attribution.py` are untouched, so default behaviour (and every
existing test) is exactly as before. To try a fix, build a `Fixes` and use
`ImprovedScorer`, whose `rescore(doc, jd)` has the same shape as
`rba.resume.attribution.rescore` and can be dropped in where that is called.

Each fix targets one problem the public-dataset benchmark exposed
(see benchmarks/LITERATURE_SURVEY.txt):

  P1 skills channel hurt agreement    -> `llm_skills`
       JD skills (required/preferred) come from an LLM, cached per JD; the
       resume side is a deterministic text match, so a skill only counts when
       it is evidenced in the resume. One LLM call per JD, none per resume.
  P2 irrelevant bullets raised score  -> `topk_pooling`
       Dense similarity = mean of the best-k chunk similarities instead of a
       length-weighted average of all chunks, so adding off-topic text cannot
       lift the score by diluting or padding.
  P3 keyword stuffing beat real text  -> `stuffing_guard`
       Lines that are mostly JD vocabulary with almost no sentence structure
       are not scored.
  P4 gibberish input scored high      -> `validity_guard` + `capped_lexical`
       Inputs whose text fragments into far too many word-pieces are refused
       ("cannot score"), and BM25 gives never-seen words the weight of a
       moderately rare word instead of the maximum.
  P5 tiers not separated              -> `requirement_match`
       Each JD requirement line is matched to its best resume line; the score
       is how well the requirements are covered.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..resume.chunking import document_chunks
from ..resume.document import ResumeDoc
from ..resume.selection import copy_doc
from .batch import DenseCache, _ensure_chunks_embedded
from .corpus import background_corpus
from .dense import embed, tokenizer
from .lexical import tokenize
from .skills import weighted_jaccard

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

BASE_WEIGHTS = {"dense": 0.5, "lexical": 0.2, "skills": 0.3}


@dataclass(frozen=True)
class Fixes:
    validity_guard: bool = False
    capped_lexical: bool = False
    topk_pooling: bool = False
    stuffing_guard: bool = False
    llm_skills: bool = False
    requirement_match: bool = False
    topk: int = 3
    # Channel weights; None = BASE_WEIGHTS (plus an equal-share "req" channel
    # when requirement_match is on). Fitted values live in the benchmark results.
    weights: dict | None = None

    @classmethod
    def all(cls, **kw) -> Fixes:
        return cls(validity_guard=True, capped_lexical=True, topk_pooling=True,
                   stuffing_guard=True, llm_skills=True, requirement_match=True, **kw)


@dataclass(frozen=True)
class ImprovedScore:
    raw: float
    channels: dict
    valid: bool = True
    notes: tuple = ()


# --------------------------------------------------------------------------
# P4: validity guard + capped lexical
# --------------------------------------------------------------------------

# Normal English fragments into ~1.2-1.6 word-pieces per word; invented words
# ("flibberwock snazzled") fragment into 3+. Threshold chosen on the netsol
# invalid-vs-valid split (see benchmarks/RESULTS notes), not guessed.
MAX_PIECES_PER_WORD = 2.2


def pieces_per_word(text: str, sample_words: int = 150) -> float | None:
    tok = tokenizer()
    if tok is None:
        return None
    words = re.findall(r"[A-Za-z]{2,}", text)[:sample_words]
    if len(words) < 5:
        return None
    ids = tok.encode(" ".join(words), add_special_tokens=False).ids
    return len(ids) / len(words)


def check_valid(resume_text: str, jd_text: str) -> tuple[bool, str]:
    # A "job description" that is really the resume (or vice versa) would score
    # as a perfect match. Found in netsol's invalid_* files, where the two
    # fields hold the same text.
    a, b = set(tokenize(resume_text)), set(tokenize(jd_text))
    if a and b and len(a & b) / len(a | b) > 0.8:
        return False, "resume and job description are the same text"
    for label, text in (("job description", jd_text), ("resume", resume_text)):
        ratio = pieces_per_word(text)
        if ratio is not None and ratio > MAX_PIECES_PER_WORD:
            return False, f"{label} does not look like natural language"
        if len(text.split()) < 15:
            return False, f"{label} is too short to score"
    return True, ""


class _Background:
    """Document frequencies of the background corpus, computed once."""
    _lock = threading.Lock()
    df: dict[str, int] | None = None
    n = 0
    avgdl = 1.0

    @classmethod
    def load(cls):
        with cls._lock:
            if cls.df is None:
                docs = [set(tokenize(t)) for t in background_corpus()]
                lens = [len(tokenize(t)) for t in background_corpus()]
                cls.n = len(docs)
                cls.avgdl = (sum(lens) / len(lens)) if lens else 1.0
                df: dict[str, int] = {}
                for d in docs:
                    for tkn in d:
                        df[tkn] = df.get(tkn, 0) + 1
                cls.df = df
        return cls


def _idf_capped(term: str, bg) -> float:
    # Unseen terms get the weight of a term in ~2% of documents, not the
    # maximum: a made-up word is not evidence of a rare, valuable skill.
    floor = max(1.0, 0.02 * bg.n)
    df = max(bg.df.get(term, 0), floor)
    return max(0.0, math.log(1.0 + (bg.n - df + 0.5) / (df + 0.5)))


def bm25_capped(resume_text: str, jd_text: str, k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 of resume vs JD normalised by the JD scored against itself, with
    the IDF of unseen words capped (see `_idf_capped`)."""
    bg = _Background.load()
    query, doc = tokenize(jd_text), tokenize(resume_text)
    if not query or not doc or bg.n == 0:
        return 0.0

    def score(q, d):
        counts: dict[str, int] = {}
        for tkn in d:
            counts[tkn] = counts.get(tkn, 0) + 1
        total = 0.0
        for term in set(q):
            f = counts.get(term, 0)
            if f:
                total += _idf_capped(term, bg) * (f * (k1 + 1)) / (
                    f + k1 * (1 - b + b * len(d) / bg.avgdl))
        return total

    ceiling = score(query, query)
    return 0.0 if ceiling <= 0 else max(0.0, min(1.0, score(query, doc) / ceiling))


# --------------------------------------------------------------------------
# P3: stuffing guard
# --------------------------------------------------------------------------

_FUNCTION = frozenset("""the a an and or of to in for with on by as is was were are
be been at from that this which who it its their our we i he she they into over
through during while using used per across within""".split())


def looks_stuffed(text: str, jd_terms: set[str]) -> bool:
    """A long run of mostly job-description vocabulary with almost no
    function words: not a sentence, so not evidence. (A genuine skills list
    can trip this; the trade-off is measured in the benchmark.)"""
    words = text.split()
    if len(words) < 8:
        return False
    func = sum(1 for w in words if w.lower().strip(".,;:") in _FUNCTION) / len(words)
    toks = tokenize(text)
    if not toks:
        return True
    density = sum(1 for t in toks if t in jd_terms) / len(toks)
    return func < 0.08 and density >= 0.6


def drop_stuffed(doc: ResumeDoc, jd_text: str) -> tuple[ResumeDoc, int]:
    jd_terms = set(tokenize(jd_text))
    work = copy_doc(doc)
    removed = 0
    for bullet in work.all_bullets():
        if bullet.enabled and looks_stuffed(bullet.text, jd_terms):
            bullet.enabled = False
            removed += 1
    return work, removed


# --------------------------------------------------------------------------
# P2: top-k dense pooling      P5: requirement matching
# --------------------------------------------------------------------------

_VEC_CACHE: dict[str, np.ndarray] = {}
_VEC_LOCK = threading.Lock()


def _embed_cached(texts: list[str]) -> list[np.ndarray] | None:
    missing = [t for t in dict.fromkeys(texts) if t not in _VEC_CACHE]
    if missing:
        vectors = embed(missing)
        if vectors is None:
            return None
        with _VEC_LOCK:
            if len(_VEC_CACHE) > 200_000:
                _VEC_CACHE.clear()
            for t, v in zip(missing, vectors, strict=True):
                _VEC_CACHE[t] = v
    return [_VEC_CACHE[t] for t in texts]


def _jd_vec(jd_text: str) -> np.ndarray | None:
    from ..resume.attribution import _jd_array
    return _jd_array(jd_text)


def dense_topk(doc: ResumeDoc, jd_text: str, k: int = 3,
               cache: DenseCache | None = None) -> float | None:
    chunks = tuple(document_chunks(doc))
    jd = _jd_vec(jd_text)
    if not chunks or jd is None:
        return None
    vectors = _ensure_chunks_embedded([chunks], cache)
    if vectors is None:
        return None
    sims = sorted((max(0.0, float(np.dot(vectors[cid], jd))) for cid, _ in chunks),
                  reverse=True)
    return float(np.mean(sims[:k]))


def jd_requirements(jd_text: str, limit: int = 30) -> list[str]:
    """Requirement-sized pieces of a JD: its lines/sentences of 4-60 words."""
    out: list[str] = []
    for line in re.split(r"[\n\r]+|(?<=[.;!?])\s+", jd_text):
        line = line.strip(" -*•\t")
        n = len(line.split())
        if 4 <= n <= 60 and line not in out:
            out.append(line)
    return out[:limit]


def requirement_match(doc: ResumeDoc, jd_text: str) -> float | None:
    """Mean over JD requirements of the best cosine to any resume line."""
    reqs = jd_requirements(jd_text)
    lines = [b.text for b in doc.enabled()]
    if doc.summary:
        lines.append(doc.summary)
    if not reqs or not lines:
        return None
    vecs = _embed_cached(reqs + lines)
    if vecs is None:
        return None
    r = np.stack(vecs[:len(reqs)])
    s = np.stack(vecs[len(reqs):])
    best = np.clip(r @ s.T, 0.0, None).max(axis=1)
    return float(best.mean())


# --------------------------------------------------------------------------
# P1: LLM-extracted JD skills, deterministic evidence match on the resume
# --------------------------------------------------------------------------

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
# Cloud model first (as requested), local fallback. The cloud model is flaky,
# so after repeated failures it is skipped for the rest of the process.
LLM_MODELS = tuple(os.environ.get("RBA_SKILL_MODELS",
                                  "glm-5.3:cloud,llama3.1:latest").split(","))
CACHE_PATH = Path(os.environ.get("RBA_SKILLS_CACHE",
                                 Path.home() / ".cache" / "rba" / "jd_skills.json"))
_PROMPT = ("Extract the skills from this job description. Reply with ONLY a JSON "
           'object {"required":[...],"preferred":[...]}. List concrete hard skills '
           "only: tools, languages, frameworks, certifications, licences, domain "
           "techniques. No soft skills, no sentences, at most 15 per list.\n\n"
           "JOB DESCRIPTION:\n")
_cache_lock = threading.Lock()
_cloud_failures = 0


def _key(jd: str) -> str:
    return hashlib.sha256(" ".join(jd.split()).encode()).hexdigest()[:20]


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return {}


def _parse_skill_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    clean = lambda xs: [s.strip() for s in xs if isinstance(s, str) and 1 < len(s.strip()) < 60][:15]
    return {"required": clean(data.get("required", [])),
            "preferred": clean(data.get("preferred", []))}


def llm_jd_skills(jd_text: str, *, timeout: float = 90.0) -> dict | None:
    """{"required": [...], "preferred": [...]} for a JD, cached on disk by
    content hash. None if every model failed (the skills channel then drops
    out rather than scoring zero)."""
    global _cloud_failures
    key = _key(jd_text)
    with _cache_lock:
        cached = _load_cache().get(key)
    if cached is not None:
        return cached or None
    for model in LLM_MODELS:
        if model.endswith(":cloud") and _cloud_failures >= 3:
            continue
        body = {"model": model, "stream": False,
                "messages": [{"role": "user", "content": _PROMPT + jd_text[:3500]}]}
        try:
            req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                parsed = _parse_skill_json(json.load(resp)["message"]["content"])
        except Exception as exc:
            log.warning("skill extraction via %s failed: %s", model, exc)
            if model.endswith(":cloud"):
                _cloud_failures += 1
            continue
        if parsed is not None:
            if model.endswith(":cloud"):
                _cloud_failures = 0
            parsed["model"] = model
            with _cache_lock:
                cache = _load_cache()
                cache[key] = parsed
                CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                CACHE_PATH.write_text(json.dumps(cache))
            return parsed
    return None


_CACHE_MEM: dict | None = None


def cached_jd_skills(jd_text: str) -> dict | None:
    """Cache lookup only: never calls a model. For scoring thousands of pairs
    after `llm_jd_skills` has already been run once per JD."""
    global _CACHE_MEM
    if _CACHE_MEM is None:
        _CACHE_MEM = _load_cache()
    return _CACHE_MEM.get(_key(jd_text)) or None


def skills_from_llm(resume_text: str, jd_text: str, *, extract=llm_jd_skills) -> float | None:
    """Coverage of the JD's LLM-extracted skills that the resume evidences.

    The resume is never sent to a model: a skill counts only if its words
    appear in the resume text.
    """
    found = extract(jd_text)
    if not found:
        return None
    required, preferred = found.get("required", []), found.get("preferred", [])
    if not required and not preferred:
        return None
    text = " ".join(resume_text.lower().split())

    def present(skill: str) -> bool:
        return re.search(r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"s?(?![a-z0-9])",
                         text) is not None

    have = [s for s in (*required, *preferred) if present(s)]
    return weighted_jaccard(have, required=required, preferred=preferred).score


# --------------------------------------------------------------------------
# putting it together
# --------------------------------------------------------------------------

def compute_features(doc: ResumeDoc, jd_text: str, fixes: Fixes,
                     cache: DenseCache | None = None,
                     skills_fn=llm_jd_skills) -> tuple[dict, ResumeDoc, tuple]:
    """Every channel value the enabled fixes can produce, plus the (possibly
    stuffing-filtered) document they were computed on."""
    notes: list[str] = []
    work = doc
    if fixes.stuffing_guard:
        work, removed = drop_stuffed(doc, jd_text)
        if removed:
            notes.append(f"ignored {removed} keyword-stuffed line(s)")
    text = work.as_text()
    feats: dict[str, float | None] = {}
    if fixes.topk_pooling:
        feats["dense"] = dense_topk(work, jd_text, fixes.topk, cache)
    feats["lexical"] = (bm25_capped(text, jd_text) if fixes.capped_lexical else None)
    if fixes.llm_skills:
        feats["skills"] = skills_from_llm(text, jd_text, extract=skills_fn)
    if fixes.requirement_match:
        feats["req"] = requirement_match(work, jd_text)
    return feats, work, tuple(notes)


def fuse(feats: dict, weights: dict | None = None) -> tuple[float, dict]:
    """Convex combination of whichever channels are present (None dropped,
    weight redistributed -- same rule as `fusion.convex_fuse`)."""
    w = dict(weights or BASE_WEIGHTS)
    if weights is None and feats.get("req") is not None:
        w["req"] = 0.3
    used = {k: v for k, v in feats.items() if v is not None and w.get(k, 0) > 0}
    total = sum(w[k] for k in used)
    if total <= 0:
        return 0.0, {}
    return sum(w[k] * used[k] for k in used) / total, used


class ImprovedScorer:
    """Drop-in alternative to `attribution.rescore` with the fixes enabled.

        scorer = ImprovedScorer(Fixes.all())
        raw, channels = scorer.rescore(doc, jd_text)

    Channels a fix does not replace fall back to the default scorer's value, so
    enabling one fix changes exactly one thing.
    """

    def __init__(self, fixes: Fixes, cache: DenseCache | None = None,
                 skills_fn=llm_jd_skills):
        self.fixes, self.cache, self.skills_fn = fixes, cache, skills_fn

    def score(self, doc: ResumeDoc, jd_text: str, *, required_skills=None,
              preferred_skills=None) -> ImprovedScore:
        from ..resume.attribution import rescore
        f = self.fixes
        if f.validity_guard:
            ok, why = check_valid(doc.as_text(), jd_text)
            if not ok:
                return ImprovedScore(0.0, {}, valid=False, notes=(why,))
        feats, work, notes = compute_features(doc, jd_text, f, self.cache, self.skills_fn)
        # Any channel the enabled fixes did not replace comes from the
        # default scorer, evaluated on the same (filtered) document.
        if any(feats.get(k) is None for k in ("dense", "lexical", "skills")):
            base_raw, base = rescore(work, jd_text, required_skills=required_skills,
                                     preferred_skills=preferred_skills, cache=self.cache)
            for k in ("dense", "lexical", "skills"):
                if feats.get(k) is None and not (k == "skills" and f.llm_skills):
                    feats[k] = base.get(k)
        raw, used = fuse(feats, f.weights)
        return ImprovedScore(raw, used, True, notes)

    def rescore(self, doc: ResumeDoc, jd_text: str, **kw) -> tuple[float, dict]:
        s = self.score(doc, jd_text, **kw)
        return s.raw, s.channels
