"""Build charts (PNG) and tables from benchmarks/results/*.json.

    uv run python -m benchmarks.report

Writes results/charts/*.png and results/tables.json (consumed by the written
REPORT.md / the published page). Missing result files are skipped.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RES = Path(__file__).parent / "results"
CH = RES / "charts"
DATASETS = ["0xnbk", "cnamuangtoun", "netsol"]
TIER_NAMES = ["No Fit", "Potential", "Good"]
COL = {"relevant": "#1f77b4", "irrelevant": "#d62728", "greedy": "#2ca02c",
       "random": "#7f7f7f", "order": "#ff7f0e", "all": "#9467bd"}


def _load(name):
    p = RES / name
    return json.loads(p.read_text()) if p.exists() else None


def score_boxplots(scores_by_ds):
    fig, axes = plt.subplots(1, len(scores_by_ds), figsize=(4.2 * len(scores_by_ds), 3.8), squeeze=False)
    for ax, (ds, rows) in zip(axes[0], scores_by_ds.items()):
        groups = [[float(r["fused"]) for r in rows if int(r["tier"]) == t] for t in range(3)]
        ax.boxplot(groups, showfliers=False, tick_labels=TIER_NAMES)
        ax.set_title(f"{ds}  (n={len(rows)})")
        ax.set_ylabel("v2 fused score (0-1)")
    fig.suptitle("Score distribution by labelled fit tier")
    fig.tight_layout(); fig.savefig(CH / "score_by_tier.png", dpi=140); plt.close(fig)


def variant_bars(metrics_by_ds):
    metric_keys = [("spearman", "Spearman vs label score"), ("auc_good_vs_no", "AUC Good vs No Fit"),
                   ("auc_good_vs_potential", "AUC Good vs Potential")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (key, title) in zip(axes, metric_keys):
        names = None
        width = 0.8 / max(1, len(metrics_by_ds))
        for j, (ds, m) in enumerate(metrics_by_ds.items()):
            vs = {n: v for n, v in m["variants"].items() if key in v}
            if not vs:
                continue
            names = names or list(vs)
            vals = [vs[n][key][0] if isinstance(vs[n][key], list) else vs[n][key] for n in vs]
            ax.bar(np.arange(len(vs)) + j * width, vals, width, label=ds)
        if names:
            ax.set_xticks(np.arange(len(names)) + 0.4 - width / 2)
            ax.set_xticklabels([n.replace("baseline: ", "bl: ") for n in names], rotation=35, ha="right", fontsize=8)
        ax.set_title(title); ax.axhline(0.5 if key != "spearman" else 0, color="k", lw=0.6)
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(CH / "variant_metrics.png", dpi=140); plt.close(fig)


def dose_response(edits_by_ds):
    fig, axes = plt.subplots(1, len(edits_by_ds), figsize=(5 * len(edits_by_ds), 4), squeeze=False, sharey=True)
    for ax, (ds, e) in zip(axes[0], edits_by_ds.items()):
        for kind in ("relevant", "irrelevant"):
            c = e["summary"]["curves"][kind]
            ks = sorted(int(k) for k in c)
            m = [c[str(k)]["mean"] for k in ks]
            lo = [c[str(k)]["ci_lo"] for k in ks]; hi = [c[str(k)]["ci_hi"] for k in ks]
            ax.plot([0] + ks, [0] + m, "-o", color=COL[kind], label=f"{kind} bullets")
            ax.fill_between(ks, lo, hi, color=COL[kind], alpha=0.18)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(ds); ax.set_xlabel("bullets added (k)"); ax.set_ylabel("score change (points /100)")
        ax.legend(fontsize=8)
    fig.suptitle("Dose-response: relevant vs length-matched irrelevant bullets (95% CI)")
    fig.tight_layout(); fig.savefig(CH / "dose_response.png", dpi=140); plt.close(fig)


def pruning(edits_by_ds):
    fig, axes = plt.subplots(1, len(edits_by_ds), figsize=(5 * len(edits_by_ds), 4), squeeze=False, sharey=True)
    for ax, (ds, e) in zip(axes[0], edits_by_ds.items()):
        p = e["summary"]["e3"]["prune"]
        for j, strat in enumerate(("harmful", "random", "helpful")):
            ax.bar(np.arange(3) + j * 0.27, [p[k][strat].get("mean", 0) for k in ("1", "3", "5")],
                   0.27, label=f"remove {strat}", color=["#2ca02c", "#7f7f7f", "#d62728"][j])
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(np.arange(3) + 0.27); ax.set_xticklabels(["k=1", "k=3", "k=5"])
        ax.set_title(ds); ax.set_ylabel("score change (points)"); ax.legend(fontsize=8)
    fig.suptitle("Removing k bullets: scorer-flagged harmful vs random vs most-helpful")
    fig.tight_layout(); fig.savefig(CH / "pruning.png", dpi=140); plt.close(fig)


def budget(edits_by_ds):
    fig, axes = plt.subplots(1, len(edits_by_ds), figsize=(5 * len(edits_by_ds), 4), squeeze=False, sharey=True)
    for ax, (ds, e) in zip(axes[0], edits_by_ds.items()):
        b = e["summary"]["e4"]
        if not b:
            continue
        bs = sorted(int(k) for k in b)
        for strat in ("greedy", "random", "order"):
            ax.plot(bs, [b[str(k)][strat] for k in bs], "-o", color=COL[strat], label=strat)
        ax.axhline(b[str(bs[0])]["all"], color=COL["all"], ls="--", label="all bullets")
        ax.set_title(f"{ds} (n={b[str(bs[0])]['n']})"); ax.set_xlabel("bullet budget B")
        ax.set_ylabel("score /100"); ax.legend(fontsize=8)
    fig.suptitle("Score at a fixed bullet budget")
    fig.tight_layout(); fig.savefig(CH / "budget.png", dpi=140); plt.close(fig)


def main() -> None:
    CH.mkdir(parents=True, exist_ok=True)
    from rba.ats.fusion import ChannelScores, convex_fuse  # noqa: F401
    scores_by_ds, metrics_by_ds, edits_by_ds = {}, {}, {}
    for ds in DATASETS:
        p = RES / f"scores_{ds}.csv"
        if p.exists():
            rows = list(csv.DictReader(p.open()))
            for r in rows:
                f = lambda k: float(r[k]) if r[k] not in ("", None) else None
                r["fused"] = convex_fuse(ChannelScores(dense=f("dense"), lexical=f("lexical"),
                                                       skills=f("skills"))).raw
            scores_by_ds[ds] = rows
        m = _load(f"metrics_{ds}.json")
        if m:
            metrics_by_ds[ds] = m
        e = _load(f"edits_{ds}.json")
        if e:
            edits_by_ds[ds] = e
    if scores_by_ds:
        score_boxplots(scores_by_ds)
    if metrics_by_ds:
        variant_bars(metrics_by_ds)
    if edits_by_ds:
        dose_response(edits_by_ds); pruning(edits_by_ds); budget(edits_by_ds)
    (RES / "tables.json").write_text(json.dumps(
        {"metrics": metrics_by_ds, "edits": {k: v["summary"] for k, v in edits_by_ds.items()}},
        indent=1, default=float))
    print("charts ->", CH)


if __name__ == "__main__":
    main()
