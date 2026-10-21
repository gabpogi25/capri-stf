"""
experiments/eval/selection_by_information.py

selection-by-information study (docs/prereg/evaluation_plan.md): D-optimal tuple selection against
max coverage. Descriptive only; no hypothesis and no decision depend on it.

Data: the pencil drift experiment drift model (experiments/math_lab/lab.gen_drift: 12 generic points,
two measurements A, B in independent random frames, bounded vs excessive drift).
Selection uses measurement A only, in its Tyler frame, among the kappa_J-feasible
six-tuples (kappa_J below the calibrated median threshold of the pipeline):
  maxcov   algorithms/selection/coverage.select_tuples_max_coverage (stops at full
           coverage; usually 2-3 tuples)
  dopt_k   greedy maximisation of log det(eps I + sum_t G_t^T G_t), G_t the (3, 4n)
           gradient of log|J| of tuple t (Lemma 3.1), eps = 1e-9; k = 6 tuples.
           log det(eps I + .) is monotone submodular in the tuple set, so greedy has
           the (1 - 1/e) guarantee on its increase over the empty set (cited).
  random_k k = 6 random feasible tuples.
Downstream score: median over the selected tuples of max_k |d log|J_k|| / delta
(lab.b_drift_median_tuples restricted to the selection). Reported: AUC and TPR at 5%
FPR per selector, coverage, number of tuples, log det. Seeds 38,000,000 + scene.
Output: selection_by_information_scenes.csv, selection_by_information_summary.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "math_lab"))

from experiment_utils import save_results  # noqa: E402
from algorithms.conditioning.scores import batch_invariant_noise_gain, log_invariant_gradients  # noqa: E402
from algorithms.selection.coverage import candidate_tuples, select_tuples_max_coverage  # noqa: E402
from algorithms.utils import whiten_homogeneous_points  # noqa: E402
from pipeline.main_pipeline import calibrated_conditioning_threshold  # noqa: E402
import lab  # noqa: E402

SEED, N_SCENES, K, EPS = 38_000_000, 1000, 6, 1e-9


def gradients(points: np.ndarray, t: tuple) -> np.ndarray:
    G = np.zeros((3, 4 * points.shape[1]))
    local = log_invariant_gradients(points, t)
    for m, p in enumerate(t):
        G[:, 4 * p:4 * p + 4] = local[:, 4 * m:4 * m + 4]
    return G


def logdet(M: np.ndarray) -> float:
    return float(np.linalg.slogdet(EPS * np.eye(len(M)) + M)[1])


def dopt(points: np.ndarray, feasible: list[tuple]) -> list[tuple]:
    info = np.zeros((4 * points.shape[1],) * 2)
    blocks = {t: gradients(points, t) for t in feasible}
    chosen = []
    for _ in range(min(K, len(feasible))):
        best = max((t for t in feasible if t not in chosen), key=lambda t: logdet(info + blocks[t].T @ blocks[t]))
        chosen.append(best)
        info = info + blocks[best].T @ blocks[best]
    return chosen


def downstream(sample: dict, tuples: list[tuple]) -> float:
    if not tuples:
        return float("nan")
    T = np.array(tuples)
    d = np.max(np.abs(lab.log_abs_J_batch(sample["A"], T) - lab.log_abs_J_batch(sample["B"], T)), axis=1)
    return float(np.median(d)) / sample["delta"]


def scene(index: int, threshold: float) -> dict:
    rng = np.random.default_rng(SEED + index)
    sample, label = lab.gen_drift(rng)
    W = whiten_homogeneous_points(sample["A"])
    candidates = candidate_tuples(12, rng)                 # all 924 six-tuples for n = 12
    kappa = batch_invariant_noise_gain(W, np.array(candidates, dtype=np.int64))
    feasible = [t for t, k in zip(candidates, kappa) if np.isfinite(k) and k < threshold]
    selections = {"maxcov": select_tuples_max_coverage(W, None, threshold, 50, rng, batch_score_fn=batch_invariant_noise_gain)[0],
                  "dopt_k": dopt(W, feasible),
                  "random_k": [feasible[i] for i in rng.choice(len(feasible), min(K, len(feasible)), replace=False)]}
    row = {"Scene": index, "Seed": SEED + index, "Label_Excessive": label, "N_feasible": len(feasible)}
    for name, tuples in selections.items():
        info = sum((gradients(W, t).T @ gradients(W, t) for t in tuples), np.zeros((48, 48)))
        row.update({f"{name}_score": downstream(sample, tuples), f"{name}_n": len(tuples),
                    f"{name}_coverage": len({p for t in tuples for p in t}) / 12, f"{name}_logdet": logdet(info)})
    return row


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import rankdata
    y = df["Label_Excessive"].to_numpy()
    rows = []
    for name in ("maxcov", "dopt_k", "random_k"):
        s = df[f"{name}_score"].to_numpy()
        ok = np.isfinite(s)
        r = rankdata(s[ok])
        n1, n0 = y[ok].sum(), ok.sum() - y[ok].sum()
        rows.append({"Selector": name, "AUC": (r[y[ok] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0),
                     "TPR_at_5pct": float(np.mean(s[ok][y[ok] == 1] > np.quantile(s[ok][y[ok] == 0], 0.95))),
                     "Mean_n_tuples": df[f"{name}_n"].mean(), "Mean_coverage": df[f"{name}_coverage"].mean(),
                     "Mean_logdet": df[f"{name}_logdet"].mean(), "N_scored": int(ok.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    threshold = calibrated_conditioning_threshold(metric="kappa")
    df = pd.DataFrame([scene(i, threshold) for i in range(N_SCENES)])
    out = summarize(df)
    print(out.to_string(index=False))
    save_results(df, "selection_by_information_scenes.csv", SEED, "selection-by-information study: tuple selection, D-optimal vs max coverage (descriptive)")
    save_results(out, "selection_by_information_summary.csv", SEED, "selection-by-information study: summary (descriptive)")


if __name__ == "__main__":
    main()
