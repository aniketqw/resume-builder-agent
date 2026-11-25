"""Opt-in scoring fixes, found by benchmarking the default scorer.

NOTHING in the existing code imports this module. `score.py`, `batch.py`,
`fusion.py` and `attribution.py` are untouched, so default behaviour (and every
existing test) is exactly as before. To try a fix, build a `Fixes` and use
`ImprovedScorer`, whose `rescore(doc, jd)` has the same shape as
`rba.resume.attribution.rescore` and can be dropped in where that is called.

Each fix targets one problem the public-dataset benchmark exposed
(see benchmarks/LITERATURE_SURVEY.txt). Two ideas that were built and
measured but did not help (top-k dense pooling, requirement-by-requirement
matching) were removed; see section 9 of the survey.

  P1 skills channel hurt agreement    -> `llm_skills`
       JD skills (required/preferred) come from an LLM, cached per JD; the
       resume side is a deterministic text match, so a skill only counts when
       it is evidenced in the resume. One LLM call per JD, none per resume.
  P3 keyword stuffing beat real text  -> `stuffing_guard`
       Lines that are mostly JD vocabulary with almost no sentence structure
       are not scored.
  P4 gibberish input scored high      -> `validity_guard` + `capped_lexical`
       Inputs whose text fragments into far too many word-pieces are refused
       ("cannot score"), and BM25 gives never-seen words the weight of a
       moderately rare word instead of the maximum.
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..resume.document import ResumeDoc
from ..resume.selection import copy_doc
from .batch import DenseCache
from .corpus import background_corpus
from .dense import tokenizer
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
    stuffing_guard: bool = False
    llm_skills: bool = False
    # Channel weights; None = BASE_WEIGHTS.
    weights: dict | None = None

    @classmethod
    def recommended(cls, **kw) -> Fixes:
        """The fixes that measurably helped: P1, P3, P4."""
        return cls(validity_guard=True, capped_lexical=True,
                   stuffing_guard=True, llm_skills=True, **kw)


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
    feats["lexical"] = (bm25_capped(text, jd_text) if fixes.capped_lexical else None)
    if fixes.llm_skills:
        feats["skills"] = skills_from_llm(text, jd_text, extract=skills_fn)
    return feats, work, tuple(notes)


def fuse(feats: dict, weights: dict | None = None) -> tuple[float, dict]:
    """Convex combination of whichever channels are present (None dropped,
    weight redistributed -- same rule as `fusion.convex_fuse`)."""
    w = dict(weights or BASE_WEIGHTS)
    used = {k: v for k, v in feats.items() if v is not None and w.get(k, 0) > 0}
    total = sum(w[k] for k in used)
    if total <= 0:
        return 0.0, {}
    return sum(w[k] * used[k] for k in used) / total, used


class ImprovedScorer:
    """Drop-in alternative to `attribution.rescore` with the fixes enabled.

        scorer = ImprovedScorer(Fixes.recommended())
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


# --------------------------------------------------------------------------
# calibration: raw fused score -> expected label score (monotone, no deps)
# --------------------------------------------------------------------------

def fit_isotonic(scores, targets) -> dict:
    """Pool-adjacent-violators fit of a non-decreasing map score -> target.
    Returns {"x": [...], "y": [...]} for `calibrate`."""
    order = np.argsort(scores)
    x = np.asarray(scores, float)[order]
    blocks = [[float(v), 1.0, float(x[i]), float(x[i])]       # mean, weight, xlo, xhi
              for i, v in enumerate(np.asarray(targets, float)[order])]
    out: list[list[float]] = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            m2, w2, lo2, hi2 = out.pop()
            m1, w1, lo1, hi1 = out.pop()
            out.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, lo1, hi2])
    xs, ys = [], []
    for m, _, lo, hi in out:
        xs += [lo, hi]; ys += [m, m]
    return {"x": xs, "y": ys}


def calibrate(raw: float, table: dict) -> float:
    """Map a raw fused score through a `fit_isotonic` table (clamped at the ends)."""
    return float(np.interp(raw, table["x"], table["y"]))
