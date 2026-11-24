"""Unit tests for the benchmark helpers. No network, no embedding model."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import adapter, edits, metrics  # noqa: E402
from benchmarks.data import _netsol_tier, subsample, Pair  # noqa: E402

TEXT = ("Built a data pipeline in Python and SQL. Reduced latency by 40 percent. "
        "Led a team of five engineers.\nMentored interns on Kubernetes deployment. "
        "Wrote design docs for the platform. Ran weekly incident reviews.")


def test_segment_and_grouping():
    items = adapter.segment(TEXT)
    assert len(items) == 6 and all(len(i.split()) >= 3 for i in items)
    doc = adapter.to_doc(TEXT, entry_size=4)
    assert [len(e.bullets) for e in doc.sections[0].entries] == [4, 2]
    assert len(doc.all_bullets()) == 6


def test_add_bullets_does_not_mutate_and_lands_in_last_entry():
    doc = adapter.to_doc(TEXT, entry_size=4)
    new = edits.add_bullets(doc, ["Shipped an extra thing today."])
    assert len(doc.all_bullets()) == 6 and len(new.all_bullets()) == 7
    assert new.sections[0].entries[-1].bullets[-1].text.startswith("Shipped")


def test_disable_indices_is_index_addressed_with_duplicate_text():
    doc = adapter.to_doc("Same sentence repeated here. Same sentence repeated here. "
                         "Another distinct sentence follows.")
    out = edits.disable_indices(doc, [1])
    assert [b.enabled for b in out.all_bullets()] == [True, False, True]
    assert all(b.enabled for b in doc.all_bullets())


def test_stuffed_bullet_uses_jd_terms_only():
    jd = "Kubernetes Kubernetes Terraform engineer engineer engineer needed urgently"
    s = edits.stuffed_bullet(jd, n_terms=3).split()
    assert s[0] == "engineer" and set(s) <= set(jd.lower().split())


def test_nearest_length_prefers_closest():
    pool = ["a b c", "a b c d e f g h", "a b c d e"]
    assert edits.nearest_length(pool, 5, random.Random(0)) == "a b c d e"


def test_greedy_order_picks_best_each_step():
    value = {0: 1.0, 1: 5.0, 2: 3.0}
    fn = lambda sel, cands: [sum(value[j] for j in sel) + value[c] for c in cands]
    assert edits.greedy_order(fn, 3, 2) == [1, 2]


def test_auc_planted():
    assert edits.auc_planted([5, 4, 1, 0], [True, True, False, False], True) == 1.0
    assert edits.auc_planted([0, 1, 4, 5], [True, True, False, False], False) == 1.0
    assert edits.auc_planted([1, 1], [True, False], True) == 0.5


def test_metrics_perfect_and_inverted():
    x = np.arange(100.0)
    assert metrics.spearman(x, x) == pytest.approx(1.0)
    tiers = np.repeat([0, 1, 2], 40)
    s = np.arange(120.0)
    assert metrics.pairwise_auc(s, tiers, 2, 0) == 1.0
    assert metrics.pairwise_auc(-s, tiers, 2, 0) == 0.0
    out = metrics.macro_f1_holdout(s, tiers)
    assert out["macro_f1"] > 0.9 and out["majority_baseline"] < 0.5


def test_paired_summary_detects_positive_shift():
    d = np.random.default_rng(0).normal(1.0, 0.5, 200)
    r = metrics.paired_summary(d)
    assert r["ci_lo"] > 0.8 and r["win_rate"] > 0.9 and r["wilcoxon_p"] < 1e-6


def test_subsample_is_deterministic_and_stratified():
    pairs = [Pair(str(i), "x", "r", "j", i % 3) for i in range(300)]
    a, b = subsample(pairs, 30, seed=1), subsample(pairs, 30, seed=1)
    assert [p.id for p in a] == [p.id for p in b]
    assert {p.tier for p in a} == {0, 1, 2}


def test_netsol_tiers():
    assert _netsol_tier("match_3.json") == 2
    assert _netsol_tier("mismatch_3.json") == 1
    assert _netsol_tier("invalid_resume_match_1.json") == 0
    assert _netsol_tier("empty_additional_info_0.json") is None
