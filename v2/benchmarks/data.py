"""Loaders that turn each public dataset into the same flat `Pair` record.

Only the resume and JD *text* and the label are kept. netsol's files also hold
the candidates' real names/contact details; those are never read into a Pair.

Dataset quirks that matter (found by inspecting them, not assumed):

* ``0xnbk/resume-ats-score-v1-en`` has ONE column, ``text``: resume and JD
  joined by the literal marker `` SEP `` (exactly one per row).
* ~48% of its resumes also appear in ``cnamuangtoun/...``, so the two are not
  independent. `Pair.in_0xnbk` flags the overlap instead of hiding it.
* netsol's `datasets` loader fails (ragged JSON); the raw files are read.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

TIERS = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}


@dataclass(frozen=True)
class Pair:
    id: str
    dataset: str
    resume: str
    jd: str
    tier: int                      # 0 No Fit, 1 Potential, 2 Good
    score: float | None = None     # continuous label, where the dataset has one
    in_0xnbk: bool = False         # overlap flag for cnamuangtoun rows
    group: str = ""                # JD identity, used to find same-JD resumes


def _jd_key(jd: str) -> str:
    return " ".join(jd.split())[:200]


def load_0xnbk() -> list[Pair]:
    from datasets import load_dataset
    ds = load_dataset("0xnbk/resume-ats-score-v1-en")
    pairs: list[Pair] = []
    for split in ds.values():
        for i, row in enumerate(split):
            resume, sep, jd = row["text"].partition(" SEP ")
            if not sep:
                continue
            pairs.append(Pair(id=f"0xnbk-{len(pairs)}", dataset="0xnbk",
                              resume=resume.strip(), jd=jd.strip(),
                              tier=TIERS[row["original_label"]],
                              score=float(row["ats_score"]),
                              group=_jd_key(jd)))
    return pairs


def load_cnamuangtoun(reference: list[Pair] | None = None) -> list[Pair]:
    from datasets import load_dataset
    ds = load_dataset("cnamuangtoun/resume-job-description-fit")
    seen = {p.resume[:200] for p in (reference or [])}
    pairs: list[Pair] = []
    for split in ds.values():
        for row in split:
            r, j = row["resume_text"].strip(), row["job_description_text"].strip()
            pairs.append(Pair(id=f"cnam-{len(pairs)}", dataset="cnamuangtoun",
                              resume=r, jd=j, tier=TIERS[row["label"]],
                              in_0xnbk=r[:200] in seen, group=_jd_key(j)))
    return pairs


# netsol file prefix -> (ordinal quality, human label). Gibberish/invalid
# inputs are deliberately broken; "match" should outrank "mismatch" should
# outrank invalid.
def _netsol_tier(name: str) -> int | None:
    if name.startswith("match_"):
        return 2
    if name.startswith("mismatch_"):
        return 1
    if name.startswith("invalid"):
        return 0
    return None


def load_netsol() -> list[Pair]:
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download("netsol/resume-score-details",
                                  repo_type="dataset",
                                  allow_patterns=["*.json"]))
    pairs: list[Pair] = []
    for path in sorted(root.glob("*.json")):
        tier = _netsol_tier(path.name)
        if tier is None:
            continue
        payload = json.loads(path.read_text())
        inp = payload.get("input") or {}
        resume, jd = inp.get("resume"), inp.get("job_description")
        if not isinstance(resume, str) or not isinstance(jd, str):
            continue
        agg = ((payload.get("output") or {}).get("scores") or {}
               ).get("aggregated_scores") or {}
        macro = agg.get("macro_scores")
        pairs.append(Pair(id=f"netsol-{path.stem}", dataset="netsol",
                          resume=resume.strip(), jd=jd.strip(), tier=tier,
                          score=float(macro) if macro is not None else None,
                          group=_jd_key(jd)))
    return pairs


def subsample(pairs: list[Pair], n: int | None, seed: int = 0) -> list[Pair]:
    """Deterministic stratified-by-tier subsample (all pairs when n is None)."""
    if not n or n >= len(pairs):
        return list(pairs)
    import random
    rng = random.Random(seed)
    by_tier: dict[int, list[Pair]] = {}
    for p in pairs:
        by_tier.setdefault(p.tier, []).append(p)
    out: list[Pair] = []
    for tier, items in sorted(by_tier.items()):
        k = max(1, round(n * len(items) / len(pairs)))
        out += rng.sample(items, min(k, len(items)))
    return out
