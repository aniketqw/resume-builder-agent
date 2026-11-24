"""Statistics for the benchmark. Pure numpy/scipy/sklearn, no v2 imports,
so it is unit-testable without the embedding model."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score


def bootstrap_ci(fn, *arrays, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """(estimate, lo, hi) of `fn(*arrays)` with a paired bootstrap."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    est = float(fn(*arrays))
    if n < 3:
        return est, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(*[a[idx] for a in arrays])
        except ValueError:      # e.g. a resample missing a class
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return est, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


def spearman(x, y) -> float:
    return float(stats.spearmanr(x, y)[0])


def kendall(x, y) -> float:
    return float(stats.kendalltau(x, y)[0])


def auc(scores, positive) -> float:
    """P(score of a random positive > score of a random negative)."""
    positive = np.asarray(positive).astype(int)
    if positive.min() == positive.max():
        raise ValueError("need both classes")
    return float(roc_auc_score(positive, scores))


def pairwise_auc(scores, tiers, hi: int, lo: int):
    """AUC of tier `hi` vs tier `lo`, ignoring other tiers."""
    scores, tiers = np.asarray(scores), np.asarray(tiers)
    keep = (tiers == hi) | (tiers == lo)
    return auc(scores[keep], tiers[keep] == hi)


def fit_thresholds(scores, tiers) -> tuple[float, float]:
    """Two cut-points (No|Potential, Potential|Good) maximising macro-F1."""
    scores, tiers = np.asarray(scores), np.asarray(tiers)
    grid = np.quantile(scores, np.linspace(0.02, 0.98, 49))
    best, best_f = (grid[0], grid[-1]), -1.0
    for i, a in enumerate(grid):
        for b in grid[i + 1:]:
            pred = np.digitize(scores, [a, b])
            f = f1_score(tiers, pred, average="macro", labels=[0, 1, 2])
            if f > best_f:
                best, best_f = (float(a), float(b)), f
    return best


def predict_tier(scores, cuts: tuple[float, float]):
    return np.digitize(np.asarray(scores), list(cuts))


def macro_f1_holdout(scores, tiers, seed: int = 0) -> dict:
    """Thresholds fitted on a random half, macro-F1 reported on the other."""
    scores, tiers = np.asarray(scores), np.asarray(tiers)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(scores))
    half = len(idx) // 2
    tr, te = idx[:half], idx[half:]
    cuts = fit_thresholds(scores[tr], tiers[tr])
    pred = predict_tier(scores[te], cuts)
    return {"cuts": cuts,
            "macro_f1": float(f1_score(tiers[te], pred, average="macro",
                                       labels=[0, 1, 2])),
            "accuracy": float(np.mean(pred == tiers[te])),
            "majority_baseline": float(np.bincount(tiers[te], minlength=3).max()
                                       / len(te))}


def paired_summary(deltas, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean paired delta with CI, win-rate and Wilcoxon p."""
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"n": 0}
    mean, lo, hi = bootstrap_ci(np.mean, d, n_boot=n_boot, seed=seed)
    nz = d[d != 0]
    p = float(stats.wilcoxon(nz)[1]) if len(nz) >= 10 else float("nan")
    return {"n": int(len(d)), "mean": mean, "ci_lo": lo, "ci_hi": hi,
            "win_rate": float(np.mean(d > 0)), "loss_rate": float(np.mean(d < 0)),
            "wilcoxon_p": p}
