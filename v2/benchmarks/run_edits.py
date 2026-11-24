"""Part B: does adding/removing bullets move v2's score the right way?

    uv run python -m benchmarks.run_edits --dataset cnamuangtoun --n 100

Public datasets have no before/after edit labels, so each experiment is a
controlled, paired intervention on a real resume+JD pair, with the dataset's
own labels/grouping supplying "relevant" vs "irrelevant" content:

  E1 dose-response   add k relevant (same-JD Good-Fit resume) vs k length-
                     matched irrelevant (Good-Fit resume of a different JD)
  E2 gap closing     Potential-Fit resume + bullets of a Good-Fit resume for
                     the same JD: share of the score gap to that Good-Fit
                     resume recovered
  E3 pruning         remove k bullets the scorer calls harmful vs random vs
                     most-helpful
  E3b planted noise  plant 3 relevant / 3 irrelevant bullets, run exact
                     leave-one-out; can the scorer tell them from originals?
  E4 budget          greedy vs random vs document-order selection at B bullets
  E5 gaming          keyword-stuffed pseudo-bullet vs a genuine relevant one
  E6 fidelity        v2's `attribute()` vs independently re-scored removals

All scores are v2's product score (`score_many`'s fused raw x 100).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import adapter, data, edits, metrics

RESULTS = Path(__file__).parent / "results"
KS = [0, 1, 2, 3, 4, 5]


def S(doc_list, jd):
    """Product score (x100) for each doc."""
    return [s.raw * 100 for s in adapter.score_docs(doc_list, jd)]


def S2(doc_list, jd):
    """Sensitivity: dense+lexical only (skills channel excluded)."""
    return [adapter.fuse(s.channels, "dense+lexical") * 100
            for s in adapter.score_docs(doc_list, jd)]


class Pools:
    def __init__(self, pairs):
        self.by_group_good = defaultdict(list)     # group -> [Pair tier 2]
        self.good = []
        for p in pairs:
            if p.tier == 2:
                self.by_group_good[p.group].append(p)
                self.good.append(p)
        self._seg = {}

    def bullets(self, p):
        if p.id not in self._seg:
            self._seg[p.id] = adapter.segment(p.resume)
        return self._seg[p.id]

    def same_jd(self, target):
        out = []
        for g in self.by_group_good.get(target.group, []):
            if g.id != target.id and g.resume[:200] != target.resume[:200]:
                out += self.bullets(g)
        return out

    def other_jd(self, target, rng, n=400):
        others = [g for g in rng.sample(self.good, min(len(self.good), 60))
                  if g.group != target.group]
        out = []
        for g in others:
            out += self.bullets(g)
        return out[:n * 5]


def e1_e5(targets, pools, rng, log):
    """E1 dose-response (+ E5 gaming control) on tier 0/1 targets."""
    curves = {"relevant": defaultdict(list), "irrelevant": defaultdict(list)}
    curves2 = {"relevant": defaultdict(list), "irrelevant": defaultdict(list)}
    gaming = {"genuine": [], "stuffed": [], "irrelevant": []}
    used = 0
    for i, t in enumerate(targets):
        rel_pool, irr_pool = pools.same_jd(t), pools.other_jd(t, rng)
        if len(rel_pool) < 5 or len(irr_pool) < 5:
            continue
        used += 1
        base = adapter.to_doc(t.resume)
        rel = rng.sample(rel_pool, 5)
        irr = [edits.nearest_length(irr_pool, len(r.split()), rng) for r in rel]
        docs = [base]
        keys = [("base", 0)]
        for k in KS[1:]:
            docs.append(edits.add_bullets(base, rel[:k])); keys.append(("relevant", k))
            docs.append(edits.add_bullets(base, irr[:k])); keys.append(("irrelevant", k))
        docs.append(edits.add_bullets(base, [edits.stuffed_bullet(t.jd)])); keys.append(("stuffed", 1))
        for fn, store in ((S, curves), (S2, curves2)):
            sc = fn(docs, t.jd)
            for (kind, k), v in zip(keys, sc):
                if kind in store:
                    store[kind][k].append(v - sc[0])
                elif fn is S:
                    gaming["stuffed"].append(v - sc[0])
        # k=1 genuine/irrelevant already in curves; mirror for the gaming table
        gaming["genuine"].append(curves["relevant"][1][-1])
        gaming["irrelevant"].append(curves["irrelevant"][1][-1])
        if (i + 1) % 20 == 0:
            log(f"  E1 {i + 1}/{len(targets)} (eligible {used})")
    return used, curves, curves2, gaming


def e2(targets, pools, rng, log):
    """Gap closing on Potential-Fit targets."""
    fracs, deltas, crossed, gaps = [], [], [], []
    for t in targets:
        goods = [g for g in pools.by_group_good.get(t.group, [])
                 if g.resume[:200] != t.resume[:200]]
        if not goods:
            continue
        ref = rng.choice(goods)
        pool = pools.bullets(ref)
        if len(pool) < 5:
            continue
        base = adapter.to_doc(t.resume)
        after = edits.add_bullets(base, rng.sample(pool, 5))
        s_base, s_after, s_ref = S([base, after, adapter.to_doc(ref.resume)], t.jd)
        gap = s_ref - s_base
        deltas.append(s_after - s_base)
        if gap > 1.0:
            fracs.append((s_after - s_base) / gap); gaps.append(gap)
            crossed.append(float(s_after >= (s_base + s_ref) / 2))
    log(f"  E2 eligible {len(deltas)}, gap>1pt {len(fracs)}")
    return {"deltas": deltas, "closure": fracs, "gaps": gaps, "crossed_midpoint": crossed}


def loo(doc, jd):
    """(base, deltas[i] = base - score without bullet i) for enabled bullets."""
    n = len(doc.all_bullets())
    docs = [doc] + [edits.disable_indices(doc, [i]) for i in range(n)]
    sc = S(docs, jd)
    return sc[0], [sc[0] - v for v in sc[1:]]


def e3_e3b_e6(resumes, pools, rng, log, n_fid=12):
    prune = {k: {"harmful": [], "random": [], "helpful": []} for k in (1, 3, 5)}
    n_with_harm, n_total, harm_share = 0, 0, []
    planted = {"relevant": [], "irrelevant": []}
    planted_top3 = {"relevant": [], "irrelevant": []}
    fid, sum_vs_joint = [], []
    for i, t in enumerate(resumes):
        doc = adapter.to_doc(t.resume)
        n = len(doc.all_bullets())
        if n < 8:
            continue
        base, d = loo(doc, t.jd)
        n_total += 1
        harm = [j for j in sorted(range(n), key=lambda j: d[j]) if d[j] < 0]
        help_ = sorted(range(n), key=lambda j: -d[j])
        harm_share.append(len(harm) / n)
        if harm:
            n_with_harm += 1
            for k in (1, 3, 5):
                ke = min(k, len(harm))
                variants = [edits.disable_indices(doc, harm[:ke]),
                            edits.disable_indices(doc, help_[:ke])]
                rnd = [edits.disable_indices(doc, rng.sample(range(n), ke)) for _ in range(3)]
                sc = S(variants + rnd, t.jd)
                prune[k]["harmful"].append(sc[0] - base)
                prune[k]["helpful"].append(sc[1] - base)
                prune[k]["random"].append(float(np.mean(sc[2:])) - base)
            if len(harm) >= 3:
                joint = S([edits.disable_indices(doc, harm[:3])], t.jd)[0] - base
                sum_vs_joint.append((-sum(d[j] for j in harm[:3]), joint))
        # E3b planted noise
        rel_pool, irr_pool = pools.same_jd(t), pools.other_jd(t, rng)
        for kind, pool in (("relevant", rel_pool), ("irrelevant", irr_pool)):
            if len(pool) < 3:
                continue
            extra = rng.sample(pool, 3)
            pdoc = edits.add_bullets(doc, extra)
            _, pd_ = loo(pdoc, t.jd)
            flags = [False] * n + [True] * 3
            planted[kind].append(edits.auc_planted(pd_, flags, kind == "relevant"))
            order = sorted(range(n + 3), key=lambda j: (-pd_[j] if kind == "relevant" else pd_[j]))
            planted_top3[kind].append(sum(1 for j in order[:3] if j >= n) / 3)
        # E6 fidelity: product `attribute()` vs independent re-scoring
        if len(fid) < n_fid:
            from rba.resume.attribution import attribute
            att = attribute(doc, t.jd, required_skills=adapter.jd_skills(t.jd),
                            preferred_skills=[], cache=adapter.CACHE)
            theirs = [imp.delta for imp in att.impacts]
            fid.append(float(np.max(np.abs(np.array(theirs) - np.array(d)))))
        if (i + 1) % 10 == 0:
            log(f"  E3 {i + 1}/{len(resumes)} (usable {n_total})")
    return {"prune": prune, "n_total": n_total, "n_with_harm": n_with_harm,
            "harm_share": harm_share, "planted_auc": planted,
            "planted_precision_at3": planted_top3,
            "fidelity_max_abs_diff": fid, "sum_vs_joint": sum_vs_joint}


def e4(resumes, rng, log, budgets=(6, 10, 15)):
    """Greedy vs random vs document-order selection at fixed bullet budgets.

    Bullets are toggled in place (entry grouping fixed) so each candidate
    changes exactly one chunk and the embedding cache stays effective.
    """
    out = {b: {"greedy": [], "random": [], "order": [], "all": []} for b in budgets}
    B = max(budgets)
    for i, t in enumerate(resumes):
        doc = adapter.to_doc(t.resume)
        n = len(doc.all_bullets())
        if n < B + 4:
            continue

        def only(sel):
            return edits.disable_indices(doc, [j for j in range(n) if j not in set(sel)])

        def gains(selected, cands):
            return S([only(selected + [c]) for c in cands], t.jd)

        order = edits.greedy_order(gains, n, B)
        for b in budgets:
            sc = S([only(order[:b]), only(list(range(b)))]
                   + [only(rng.sample(range(n), b)) for _ in range(5)] + [doc], t.jd)
            out[b]["greedy"].append(sc[0]); out[b]["order"].append(sc[1])
            out[b]["random"].append(float(np.mean(sc[2:7]))); out[b]["all"].append(sc[7])
        log(f"  E4 {i + 1}/{len(resumes)}")
    return out


def summarise(res: dict) -> dict:
    summ: dict = {}
    for name in ("curves", "curves2"):
        summ[name] = {kind: {k: metrics.paired_summary(v) for k, v in by_k.items()}
                      for kind, by_k in res[name].items()}
    summ["gaming"] = {k: metrics.paired_summary(v) for k, v in res["gaming"].items()}
    e2d = res["e2"]
    summ["e2"] = {"delta": metrics.paired_summary(e2d["deltas"]),
                  "closure_median": float(np.median(e2d["closure"])) if e2d["closure"] else None,
                  "closure_mean": float(np.mean(e2d["closure"])) if e2d["closure"] else None,
                  "crossed_midpoint": float(np.mean(e2d["crossed_midpoint"])) if e2d["crossed_midpoint"] else None,
                  "n_gap": len(e2d["closure"])}
    e3 = res["e3"]
    summ["e3"] = {"n_total": e3["n_total"], "n_with_harm": e3["n_with_harm"],
                  "harm_share_mean": float(np.mean(e3["harm_share"])) if e3["harm_share"] else None,
                  "prune": {k: {s: metrics.paired_summary(v) for s, v in d.items()}
                            for k, d in e3["prune"].items()},
                  "planted_auc": {k: (float(np.nanmean(v)) if v else None) for k, v in e3["planted_auc"].items()},
                  "planted_precision_at3": {k: (float(np.mean(v)) if v else None) for k, v in e3["planted_precision_at3"].items()},
                  "fidelity_max_abs_diff": max(e3["fidelity_max_abs_diff"], default=None),
                  "sum_vs_joint_mean": [float(np.mean([a for a, _ in e3["sum_vs_joint"]])),
                                        float(np.mean([b for _, b in e3["sum_vs_joint"]]))]
                  if e3["sum_vs_joint"] else None}
    summ["e4"] = {b: {k: (float(np.mean(v)) if v else None) for k, v in d.items()} | {"n": len(d["greedy"])}
                  for b, d in res["e4"].items()}
    return summ


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cnamuangtoun", choices=["0xnbk", "cnamuangtoun", "netsol"])
    ap.add_argument("--n", type=int, default=100, help="targets per experiment")
    ap.add_argument("--n-budget", type=int, default=20, help="resumes for E4 (slow)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    if args.dataset == "0xnbk":
        pairs = data.load_0xnbk()
    elif args.dataset == "cnamuangtoun":
        pairs = data.load_cnamuangtoun()
    else:
        pairs = data.load_netsol()
    pools = Pools(pairs)
    low = [p for p in pairs if p.tier == 0]
    mid = [p for p in pairs if p.tier == 1]
    rng.shuffle(low); rng.shuffle(mid)
    # Only targets whose JD has same-JD Good-Fit resumes can get "relevant" bullets.
    elig = lambda ps: [p for p in ps if pools.same_jd(p)]
    t_low, t_mid = elig(low)[:args.n], elig(mid)[:args.n]
    log(f"{args.dataset}: {len(pairs)} pairs; eligible No-Fit {len(t_low)}, Potential {len(t_mid)}")

    res: dict = {"dataset": args.dataset, "seed": args.seed,
                 "n_pairs": len(pairs), "n_targets": {"tier0": len(t_low), "tier1": len(t_mid)}}
    for label, targets in (("tier0", t_low), ("tier1", t_mid)):
        used, c1, c2, g = e1_e5(targets, pools, rng, log)
        res.setdefault("by_tier", {})[label] = {"eligible": used, "curves": c1, "curves2": c2, "gaming": g}
    # Pool tier0+tier1 for the headline curves
    res["curves"] = {k: defaultdict(list) for k in ("relevant", "irrelevant")}
    res["curves2"] = {k: defaultdict(list) for k in ("relevant", "irrelevant")}
    res["gaming"] = {k: [] for k in ("genuine", "stuffed", "irrelevant")}
    for tier in res["by_tier"].values():
        for kind in res["curves"]:
            for k in KS[1:]:
                res["curves"][kind][k] += tier["curves"][kind][k]
                res["curves2"][kind][k] += tier["curves2"][kind][k]
        for kind in res["gaming"]:
            res["gaming"][kind] += tier["gaming"][kind]
    res["e2"] = e2(t_mid, pools, rng, log)
    prune_set = rng.sample(pairs, min(args.n, len(pairs)))
    res["e3"] = e3_e3b_e6(prune_set, pools, rng, log)
    res["e4"] = e4(prune_set[:args.n_budget * 2], rng, log) if args.n_budget else {}
    summ = summarise({**res, "curves": {k: dict(v) for k, v in res["curves"].items()},
                      "curves2": {k: dict(v) for k, v in res["curves2"].items()}})
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"edits_{args.dataset}.json").write_text(json.dumps(
        {"summary": summ, "n_targets": res["n_targets"], "n_pairs": res["n_pairs"],
         "eligible": {k: v["eligible"] for k, v in res["by_tier"].items()}}, indent=1, default=float))
    (RESULTS / f"edits_{args.dataset}_raw.json").write_text(json.dumps(
        {"curves": {k: {str(i): v for i, v in d.items()} for k, d in res["curves"].items()},
         "gaming": res["gaming"], "e2": res["e2"], "e3": res["e3"], "e4": res["e4"]},
        default=float))
    log("done")


if __name__ == "__main__":
    main()
