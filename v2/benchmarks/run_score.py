"""Part A: how well does v2's score agree with each dataset's labels?

    uv run python -m benchmarks.run_score --dataset 0xnbk --n 300
    uv run python -m benchmarks.run_score --dataset all

Writes benchmarks/results/scores_<dataset>.csv (one row per pair with every
channel) and metrics_<dataset>.json. Scoring is deterministic, so a re-run
reproduces the CSV exactly.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from . import adapter, data, metrics

RESULTS = Path(__file__).parent / "results"
FIELDS = ["id", "dataset", "tier", "label_score", "in_0xnbk", "dense",
          "lexical", "skills", "n_jd_skills", "n_bullets", "tfidf", "raw_dense"]


def tfidf_scores(pairs: list[data.Pair]) -> list[float]:
    """Baseline: TF-IDF cosine of resume vs JD, vocabulary fit on the set."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2)
    vec.fit([p.resume for p in pairs] + [p.jd for p in pairs])
    a, b = vec.transform([p.resume for p in pairs]), vec.transform([p.jd for p in pairs])
    return [float(a[i].multiply(b[i]).sum()) for i in range(len(pairs))]


def score_pairs(pairs: list[data.Pair], baselines: bool = True,
                log_every: int = 100) -> list[dict]:
    from rba.ats.dense import cosine_similarity
    tfidf = tfidf_scores(pairs) if baselines else [None] * len(pairs)
    rows, t0 = [], time.time()
    for i, p in enumerate(pairs):
        doc = adapter.to_doc(p.resume)
        scored = adapter.score_docs([doc], p.jd)[0]
        ch = scored.channels
        rows.append({
            "id": p.id, "dataset": p.dataset, "tier": p.tier,
            "label_score": p.score, "in_0xnbk": int(p.in_0xnbk),
            "dense": ch.get("dense"), "lexical": ch.get("lexical"),
            "skills": ch.get("skills"),
            "n_jd_skills": len(adapter.jd_skills(p.jd)),
            "n_bullets": len(doc.all_bullets()), "tfidf": tfidf[i],
            "raw_dense": cosine_similarity(p.resume, p.jd) if baselines else None,
        })
        if log_every and (i + 1) % log_every == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(pairs)}  {rate:.1f} pairs/s", flush=True)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def _col(rows, key):
    return np.array([np.nan if r[key] in (None, "") else float(r[key]) for r in rows])


def evaluate(rows: list[dict]) -> dict:
    """All Part A metrics for one dataset's scored rows."""
    tiers = np.array([int(r["tier"]) for r in rows])
    label = _col(rows, "label_score")
    dense, lex, skl = _col(rows, "dense"), _col(rows, "lexical"), _col(rows, "skills")
    from rba.ats.fusion import ChannelScores, convex_fuse
    variants: dict[str, np.ndarray] = {
        "lexical (BM25)": lex, "dense": dense,
        "dense+lexical": np.array([
            convex_fuse(ChannelScores(dense=None if np.isnan(d) else d,
                                      lexical=l)).raw for d, l in zip(dense, lex)]),
        "full (3 channels)": np.array([
            convex_fuse(ChannelScores(dense=None if np.isnan(d) else d, lexical=l,
                                      skills=None if np.isnan(s) else s)).raw
            for d, l, s in zip(dense, lex, skl)]),
        "baseline: TF-IDF cosine": _col(rows, "tfidf"),
        "baseline: raw bge cosine (512 tok)": _col(rows, "raw_dense"),
    }
    out: dict = {"n": len(rows), "tier_counts": np.bincount(tiers, minlength=3).tolist(),
                 "skills_channel_coverage": float(np.mean(~np.isnan(skl))), "variants": {}}
    for name, s in variants.items():
        ok = np.isfinite(s)
        if ok.sum() < 10:
            continue
        s_, t_ = s[ok], tiers[ok]
        entry: dict = {"n": int(ok.sum()),
                       "tier_means": [float(s_[t_ == k].mean()) if (t_ == k).any() else None
                                      for k in range(3)]}
        has_label = np.isfinite(label[ok])
        if has_label.sum() >= 10:
            entry["spearman"] = metrics.bootstrap_ci(metrics.spearman, s_[has_label], label[ok][has_label], n_boot=500)
            entry["kendall"] = metrics.kendall(s_[has_label], label[ok][has_label])
        for hi, lo, nm in ((2, 0, "auc_good_vs_no"), (2, 1, "auc_good_vs_potential"),
                           (1, 0, "auc_potential_vs_no")):
            if ((t_ == hi).any() and (t_ == lo).any()):
                entry[nm] = metrics.bootstrap_ci(
                    lambda a, b, hi=hi, lo=lo: metrics.pairwise_auc(a, b, hi, lo),
                    s_, t_, n_boot=500)
        if len(set(t_.tolist())) == 3:
            entry["holdout"] = metrics.macro_f1_holdout(s_, t_)
        out["variants"][name] = entry
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="0xnbk",
                    choices=["0xnbk", "cnamuangtoun", "netsol", "screening", "all"])
    ap.add_argument("--n-jds", type=int, default=None, help="keep all pairs of this many JDs")
    ap.add_argument("--n", type=int, default=None, help="stratified subsample size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args(argv)

    base = data.load_0xnbk() if args.dataset in ("0xnbk", "cnamuangtoun", "all") else []
    loaders = {"0xnbk": lambda: base, "cnamuangtoun": lambda: data.load_cnamuangtoun(base),
               "netsol": data.load_netsol, "screening": data.load_screening}
    names = ["0xnbk", "cnamuangtoun", "netsol"] if args.dataset == "all" else [args.dataset]
    for name in names:
        pairs = loaders[name]()
        if args.n_jds:
            pairs = data.jd_subset(pairs, args.n_jds)
        pairs = data.subsample(pairs, args.n, args.seed)
        print(f"== {name}: scoring {len(pairs)} pairs", flush=True)
        rows = score_pairs(pairs, baselines=not args.no_baselines)
        write_csv(rows, RESULTS / f"scores_{name}.csv")
        (RESULTS / f"metrics_{name}.json").write_text(
            json.dumps(evaluate(rows), indent=1, default=float))
        print(f"   wrote results for {name}")


if __name__ == "__main__":
    main()
