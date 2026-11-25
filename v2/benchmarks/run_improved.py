"""Stage 1: compute the fix features (embedding/lexical ones) per pair.

    uv run python -m benchmarks.run_improved --dataset cnamuangtoun

Evaluates on the pairs of `--n-jds` deterministic job descriptions per dataset
(the LLM-skills fix needs one slow LLM call per JD, so the set is bounded).
Writes results/improved_features_<dataset>.csv. LLM skills are joined later in
`analyze_improved.py` from the JD-skills cache, so this stage needs no LLM.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from rba.ats import improved

from . import adapter, data

RESULTS = Path(__file__).parent / "results"
FIELDS = ["id", "valid", "valid_reason", "lex_capped", "n_stuffed", "ppw_resume", "ppw_jd"]
FIX = improved.Fixes(capped_lexical=True)


def load(dataset: str, n_jds: int):
    if dataset == "0xnbk":
        pairs = data.load_0xnbk()
    elif dataset == "cnamuangtoun":
        pairs = data.load_cnamuangtoun()
    else:
        pairs = data.load_netsol()
    return data.jd_subset(pairs, n_jds)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["0xnbk", "cnamuangtoun", "netsol"])
    ap.add_argument("--n-jds", type=int, default=60)
    args = ap.parse_args(argv)
    pairs = load(args.dataset, args.n_jds)
    print(f"{args.dataset}: {len(pairs)} pairs over {len({p.group for p in pairs})} JDs", flush=True)
    rows, t0 = [], time.time()
    for i, p in enumerate(pairs):
        doc = adapter.to_doc(p.resume)
        ok, why = improved.check_valid(doc.as_text(), p.jd)
        _, removed = improved.drop_stuffed(doc, p.jd)
        feats, _, _ = improved.compute_features(doc, p.jd, FIX, adapter.CACHE)
        rows.append({"id": p.id, "valid": int(ok), "valid_reason": why,
                     "lex_capped": feats.get("lexical"), "n_stuffed": removed,
                     "ppw_resume": improved.pieces_per_word(p.resume),
                     "ppw_jd": improved.pieces_per_word(p.jd)})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pairs)}  {(i + 1) / (time.time() - t0):.1f} pairs/s", flush=True)
    with (RESULTS / f"improved_features_{args.dataset}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader(); w.writerows(rows)
    print("done")


if __name__ == "__main__":
    main()
