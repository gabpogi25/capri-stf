"""
forensics/eval/metrics.py

Evaluation metrics for the forensic experiments. Every classifier table
reports N, class balance, coverage and confidence intervals, and every
classifier is compared with the majority-class baseline.

Conventions: y = 1 means "generated" (positive), score higher = more likely
generated, abstained items carry abstain = True and are excluded from
accuracy metrics but counted in coverage.
"""

from __future__ import annotations
from typing import Callable

import numpy as np
from scipy import stats


# ----------------------------------------------------------------- AUC (DeLong)

def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _delong_components(y: np.ndarray, s: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    positives, negatives = s[y == 1], s[y == 0]
    m, n = len(positives), len(negatives)
    tx, ty, tz = _midrank(positives), _midrank(negatives), _midrank(np.r_[positives, negatives])
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    return float(auc), v01, v10


def auc_delong(y, s, alpha: float = 0.05) -> dict:
    """AUC with DeLong standard error and normal confidence interval."""
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError("AUC needs both classes")
    auc, v01, v10 = _delong_components(y, s)
    variance = np.var(v01, ddof=1) / len(v01) + np.var(v10, ddof=1) / len(v10)
    se = float(np.sqrt(variance))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"auc": auc, "se": se, "ci_lo": max(0.0, auc - z * se), "ci_hi": min(1.0, auc + z * se)}


def delong_paired_test(y, s_a, s_b) -> dict:
    """Paired DeLong test of AUC(a) - AUC(b) on the same items."""
    y = np.asarray(y, dtype=int)
    auc_a, v01a, v10a = _delong_components(y, np.asarray(s_a, float))
    auc_b, v01b, v10b = _delong_components(y, np.asarray(s_b, float))
    s01 = np.cov(np.vstack([v01a, v01b]))
    s10 = np.cov(np.vstack([v10a, v10b]))
    cov = s01 / len(v01a) + s10 / len(v10a)
    variance = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (auc_a - auc_b) / np.sqrt(variance) if variance > 0 else 0.0
    return {"auc_a": auc_a, "auc_b": auc_b, "difference": auc_a - auc_b,
            "z": float(z), "p_value": float(2 * stats.norm.sf(abs(z)))}


# ----------------------------------------------------------------- operating points

def tpr_at_fpr(y, s, fpr: float) -> dict:
    """Threshold set on NEGATIVES (authentic) only so that FPR <= fpr; returns TPR there."""
    y, s = np.asarray(y, dtype=int), np.asarray(s, dtype=float)
    negatives = np.sort(s[y == 0])
    k = int(np.floor((1 - fpr) * len(negatives)))
    threshold = negatives[min(k, len(negatives) - 1)]
    return {"threshold": float(threshold),
            "tpr": float(np.mean(s[y == 1] > threshold)),
            "fpr": float(np.mean(s[y == 0] > threshold))}


def confusion(y, prediction) -> dict:
    y, p = np.asarray(y, dtype=int), np.asarray(prediction, dtype=int)
    return {"tp": int(np.sum((p == 1) & (y == 1))), "fp": int(np.sum((p == 1) & (y == 0))),
            "tn": int(np.sum((p == 0) & (y == 0))), "fn": int(np.sum((p == 0) & (y == 1)))}


def classification_report(y, prediction) -> dict:
    """Confusion matrix, balanced accuracy, MCC, F1, with the majority-class baseline beside it."""
    y, p = np.asarray(y, dtype=int), np.asarray(prediction, dtype=int)

    def summarize(pred):
        c = confusion(y, pred)
        tpr = c["tp"] / max(c["tp"] + c["fn"], 1)
        tnr = c["tn"] / max(c["tn"] + c["fp"], 1)
        precision = c["tp"] / max(c["tp"] + c["fp"], 1)
        f1 = 2 * precision * tpr / max(precision + tpr, 1e-300)
        num = c["tp"] * c["tn"] - c["fp"] * c["fn"]
        den = np.sqrt(float((c["tp"] + c["fp"]) * (c["tp"] + c["fn"]) * (c["tn"] + c["fp"]) * (c["tn"] + c["fn"])))
        return {**c, "balanced_accuracy": 0.5 * (tpr + tnr), "mcc": float(num / den) if den > 0 else 0.0,
                "f1": float(f1), "precision": float(precision), "recall": float(tpr)}

    majority = int(np.mean(y) >= 0.5)
    return {"n": int(len(y)), "positive_fraction": float(np.mean(y)),
            "model": summarize(p), "majority_baseline": summarize(np.full_like(y, majority))}


# ----------------------------------------------------------------- abstention, compression, video

def risk_coverage(y, s, abstain, threshold: float) -> dict:
    """Coverage = fraction answered; risk = error rate among answered items."""
    y, s, abstain = np.asarray(y, int), np.asarray(s, float), np.asarray(abstain, bool)
    answered = ~abstain
    if not answered.any():
        return {"coverage": 0.0, "risk": float("nan"), "n_answered": 0}
    errors = (s[answered] > threshold).astype(int) != y[answered]
    return {"coverage": float(answered.mean()), "risk": float(errors.mean()), "n_answered": int(answered.sum())}


def delta_fpr(threshold: float, authentic_raw, authentic_compressed) -> float:
    """FPR(compressed) - FPR(raw) at a threshold fixed on raw authentic data."""
    return float(np.mean(np.asarray(authentic_compressed) > threshold)
                 - np.mean(np.asarray(authentic_raw) > threshold))


def flip_rate(decisions) -> float:
    d = np.asarray(decisions)
    return float(np.mean(d[1:] != d[:-1])) if d.size > 1 else 0.0


def segment_f1(predicted: list[tuple[float, float]], truth: list[tuple[float, float]], tol: float = 1.0) -> float:
    """F1 of onset localization: a predicted interval matches a true one if their onsets differ by <= tol."""
    matched_truth = set()
    tp = 0
    for p_start, _ in predicted:
        for j, (t_start, _) in enumerate(truth):
            if j not in matched_truth and abs(p_start - t_start) <= tol:
                matched_truth.add(j)
                tp += 1
                break
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(truth) if truth else 0.0
    return float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def reliability(y, probability, bins: int = 10, min_n: int = 50) -> dict:
    """Reliability table; bins with fewer than min_n items are listed but excluded from ECE."""
    y, p = np.asarray(y, float), np.asarray(probability, float)
    edges = np.linspace(0, 1, bins + 1)
    rows, omitted = [], []
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        n = int(mask.sum())
        if n == 0:
            continue
        row = {"bin_lo": edges[i], "bin_hi": edges[i + 1], "n": n,
               "mean_probability": float(p[mask].mean()), "accuracy": float(y[mask].mean())}
        (rows if n >= min_n else omitted).append(row)
    total = sum(r["n"] for r in rows)
    ece = sum(r["n"] * abs(r["mean_probability"] - r["accuracy"]) for r in rows) / total if total else float("nan")
    return {"rows": rows, "omitted_bins": omitted, "ece": float(ece)}


# ----------------------------------------------------------------- resampling

def cluster_bootstrap(metric: Callable, y, s, cluster_ids, n_boot: int = 2000, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """Bootstrap CI resampling whole clusters (source scene / video)."""
    y, s, clusters = np.asarray(y), np.asarray(s), np.asarray(cluster_ids)
    unique = np.unique(clusters)
    index = {c: np.flatnonzero(clusters == c) for c in unique}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[c] for c in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(metric(y[idx], s[idx]))
    return {"estimate": float(metric(y, s)),
            "ci_lo": float(np.percentile(values, 100 * alpha / 2)),
            "ci_hi": float(np.percentile(values, 100 * (1 - alpha / 2))),
            "n_valid_resamples": len(values)}


def paired_bootstrap_diff(metric: Callable, y, s_a, s_b, cluster_ids, n_boot: int = 2000,
                          seed: int = 0, alpha: float = 0.05) -> dict:
    """CI of metric(a) - metric(b) on the same items, resampling clusters."""
    y, a, b, clusters = map(np.asarray, (y, s_a, s_b, cluster_ids))
    unique = np.unique(clusters)
    index = {c: np.flatnonzero(clusters == c) for c in unique}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([index[c] for c in rng.choice(unique, size=len(unique), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"difference": float(metric(y, a) - metric(y, b)), "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (multiple generators)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
