"""Stage 2: evaluate every fix against the baseline on the same pairs.

    RBA_SKILLS_CACHE=benchmarks/results/jd_skills_cache.json \
        uv run python -m benchmarks.analyze_improved

Joins the baseline scores (scores_<ds>.csv), the fix features
(improved_features_<ds>.csv) and the LLM-skills cache, builds one score per
variant per pair, and reports Part A metrics plus a per-job ranking metric.
Weights for the combined scorer are fitted on 0xnbk only and tested on pairs
from the other datasets whose JD never appeared in 0xnbk.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from rba.ats import improved

from . import adapter, data, metrics

RES = Path(__file__).parent / "results"
DATASETS = ["0xnbk", "cnamuangtoun", "netsol"]


def _f(x):
    return None if x in ("", None) else float(x)


def load_table(ds: str, groups: dict, texts: dict) -> list[dict]:
    base = {r["id"]: r for r in csv.DictReader((RES / f"scores_{ds}.csv").open())}
    out = []
    for r in csv.DictReader((RES / f"improved_features_{ds}.csv").open()):
        b = base[r["id"]]
        row = {"id": r["id"], "ds": ds, "tier": int(b["tier"]), "label": _f(b["label_score"]),
               "in_0xnbk": int(b["in_0xnbk"]), "group": groups[r["id"]],
               "dense": _f(b["dense"]), "lex": _f(b["lexical"]), "skills_old": _f(b["skills"]),
               "dense_topk": _f(r["dense_topk"]), "lex_capped": _f(r["lex_capped"]),
               "req": _f(r["req"]), "valid": int(r["valid"]), "n_stuffed": int(r["n_stuffed"])}
        resume, jd = texts[r["id"]]
        row["skills_llm"] = improved.skills_from_llm(resume, jd, extract=improved.cached_jd_skills)
        out.append(row)
    return out


def variants(row: dict, w_fit: dict | None) -> dict[str, float]:
    g = improved.fuse
    v = {}
    v["V0 baseline (3-ch, shortcut skills)"] = g({"dense": row["dense"], "lexical": row["lex"], "skills": row["skills_old"]})[0]
    v["V1 +P1 LLM skills"] = g({"dense": row["dense"], "lexical": row["lex"], "skills": row["skills_llm"]})[0]
    v["V2 +P2 top-k dense"] = g({"dense": row["dense_topk"], "lexical": row["lex"], "skills": row["skills_old"]})[0]
    v["V3 +P3 stuffing guard"] = v["V0 baseline (3-ch, shortcut skills)"]   # see n_stuffed
    v["V4 +P4 validity + capped BM25"] = (g({"dense": row["dense"], "lexical": row["lex_capped"],
                                            "skills": row["skills_old"]})[0] if row["valid"] else 0.0)
    v["V5 +P5 requirement match"] = g({"dense": row["dense"], "lexical": row["lex"],
                                       "skills": row["skills_old"], "req": row["req"]})[0]
    allf = {"dense": row["dense_topk"], "lexical": row["lex_capped"],
            "skills": row["skills_llm"], "req": row["req"]}
    v["V6 all fixes (default weights)"] = g(allf)[0] if row["valid"] else 0.0
    if w_fit:
        v["V7 all fixes (fitted weights)"] = g(allf, w_fit)[0] if row["valid"] else 0.0
    # single channels, for the "does this channel carry signal" question
    for name, key in (("ch: dense top-k", "dense_topk"), ("ch: requirement match", "req"),
                      ("ch: capped BM25", "lex_capped"), ("ch: LLM skills", "skills_llm"),
                      ("ch: old dense (mean)", "dense"), ("ch: old BM25", "lex"),
                      ("ch: old skills", "skills_old")):
        if row[key] is not None:
            v[name] = row[key]
    return v


def fit_weights(rows: list[dict]) -> dict:
    keys = ["dense_topk", "lex_capped", "skills_llm", "req"]
    names = ["dense", "lexical", "skills", "req"]
    ok = [r for r in rows if r["valid"]]
    X = np.array([[0.0 if r[k] is None else r[k] for k in keys] for r in ok])
    mean = X.mean(axis=0)
    # standardise columns so weights are comparable, fit tier with non-negative LS
    Z = (X - mean) / (X.std(axis=0) + 1e-9)
    y = np.array([r["tier"] for r in ok], float)
    y = (y - y.mean()) / y.std()
    coef, _ = nnls(Z, y)
    w = coef / (X.std(axis=0) + 1e-9)            # back to raw-score units
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
    return {n: float(x) for n, x in zip(names, w)}


def per_jd_auc(rows, scores):
    """Mean over JDs of AUC(Good vs No Fit) *within* that JD: the ATS question
    'rank this JD's applicants'."""
    by: dict[str, list] = {}
    for r, s in zip(rows, scores):
        by.setdefault(r["group"], []).append((r["tier"], s))
    vals = []
    for items in by.values():
        t = np.array([i[0] for i in items]); s = np.array([i[1] for i in items])
        if (t == 2).any() and (t == 0).any():
            vals.append(metrics.pairwise_auc(s, t, 2, 0))
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)


def evaluate(rows: list[dict], names: list[str], table: dict, ds: str) -> dict:
    tiers = np.array([r["tier"] for r in rows])
    label = np.array([np.nan if r["label"] is None else r["label"] for r in rows])
    out = {}
    for name in names:
        s = np.array([table[i].get(name, np.nan) for i in range(len(rows))], float)
        ok = np.isfinite(s)
        if ok.sum() < 30:
            continue
        s_, t_ = s[ok], tiers[ok]
        e = {"n": int(ok.sum())}
        if np.isfinite(label[ok]).sum() > 30:
            lab = label[ok]; m = np.isfinite(lab)
            e["spearman"] = metrics.bootstrap_ci(metrics.spearman, s_[m], lab[m], n_boot=300)
        for hi, lo, nm in ((2, 0, "auc_good_vs_no"), (2, 1, "auc_good_vs_pot"), (1, 0, "auc_pot_vs_no")):
            if (t_ == hi).any() and (t_ == lo).any():
                e[nm] = metrics.bootstrap_ci(lambda a, b, hi=hi, lo=lo: metrics.pairwise_auc(a, b, hi, lo),
                                             s_, t_, n_boot=300)
        if len(set(t_.tolist())) == 3:
            e["holdout_f1"] = metrics.macro_f1_holdout(s_, t_)["macro_f1"]
        sub = [r for r, o in zip(rows, ok) if o]
        e["per_jd_auc"] = per_jd_auc(sub, s_)
        out[name] = e
    return out


def main() -> None:
    base0 = data.load_0xnbk()
    allp = base0 + data.load_cnamuangtoun(base0) + data.load_netsol()
    groups = {p.id: p.group for p in allp}
    texts = {p.id: (p.resume, p.jd) for p in allp}
    tables = {ds: load_table(ds, groups, texts) for ds in DATASETS
              if (RES / f"improved_features_{ds}.csv").exists()}
    train = tables.get("0xnbk", [])
    w_fit = fit_weights(train) if train else None
    print("fitted weights (on 0xnbk):", w_fit)
    nbk_groups = {r["group"] for r in train}
    result = {"fitted_weights": w_fit, "datasets": {}}
    for ds, rows in tables.items():
        for r in rows:
            r["_v"] = variants(r, w_fit)
        names = list(dict.fromkeys(k for r in rows for k in r["_v"]))
        table = [r["_v"] for r in rows]
        res = {"n": len(rows), "n_stuffed_pairs": sum(1 for r in rows if r["n_stuffed"]),
               "invalid_rejected": sum(1 for r in rows if not r["valid"]),
               "llm_skills_coverage": float(np.mean([r["skills_llm"] is not None for r in rows])),
               "all_pairs": evaluate(rows, names, table, ds)}
        if ds != "0xnbk":
            held = [(r, t) for r, t in zip(rows, table)
                    if not r["in_0xnbk"] and r["group"] not in nbk_groups]
            if len(held) >= 100:
                res["held_out_n"] = len(held)
                res["held_out"] = evaluate([h[0] for h in held], names, [h[1] for h in held], ds)
        result["datasets"][ds] = res
    (RES / "improved_metrics.json").write_text(json.dumps(result, indent=1, default=float))
    (RES / "fitted_weights.json").write_text(json.dumps(w_fit, indent=1))
    for ds, res in result["datasets"].items():
        print(f"\n== {ds}  n={res['n']} stuffed_pairs={res['n_stuffed_pairs']} "
              f"rejected_invalid={res['invalid_rejected']} llm_skill_cov={res['llm_skills_coverage']:.2f}")
        for name, e in res["all_pairs"].items():
            f = lambda x: f"{x[0]:.3f}" if x else "  -  "
            jd = e["per_jd_auc"]
            print(f"  {name:38s} spear={f(e.get('spearman'))} aucGN={f(e.get('auc_good_vs_no'))} "
                  f"aucGP={f(e.get('auc_good_vs_pot'))} aucPN={f(e.get('auc_pot_vs_no'))} "
                  f"perJD={jd[0] if jd[0] is None else round(jd[0], 3)}")


if __name__ == "__main__":
    main()
