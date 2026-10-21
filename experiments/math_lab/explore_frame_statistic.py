"""
experiments/math_lab/explore_frame_statistic.py

EXPLORATORY (not confirmatory; plan item B1). Compares the frame-coordinate
statistic (3(n - 5) cross-ratios, algorithms/consistency/frame.py) with the full pencil
family on the frame-free drift benchmark of run_pencil_drift.py (lab.gen_drift), and times
both for larger n. Seeds from the exploration band (90.2e6). Any claim about the frame
statistic's power must come from a pre-registered run.

Run: python experiments/math_lab/explore_frame_statistic.py [--runtime-only]
Output: frame_statistic_exploration.csv (per scene), frame_statistic_exploration_summary.csv,
frame_statistic_runtime.csv.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "math_lab"))

from experiment_utils import save_results  # noqa: E402
import lab  # noqa: E402
from algorithms.consistency.frame import choose_frame, frame_discrepancies  # noqa: E402
from algorithms.consistency.pencil import pencil_discrepancies, pencil_drift_score  # noqa: E402

SEED = 90_200_000
N_SCENES = 1000
N_BOOT = 1000


def scores(sample: dict) -> dict:
    A, B, delta = sample["A"], sample["B"], sample["delta"]
    D = frame_discrepancies(A, B, choose_frame(A, B))
    D = D[np.isfinite(D)]
    return {"pencil_q10": pencil_drift_score(A, B, delta),
            "frame_q10": float(np.quantile(D, 0.1)) / delta,
            "frame_q50": float(np.quantile(D, 0.5)) / delta}


def runtime(rng) -> pd.DataFrame:
    rows = []
    for n in (12, 30, 100, 300):
        P = rng.normal(size=(4, n))
        Q = P + 1e-3 * rng.normal(size=(4, n))
        for name, fn in (("pencil_full_or_50k", lambda: pencil_discrepancies(P, Q)),
                         ("frame_given", lambda: frame_discrepancies(P, Q, (0, 1, 2, 3, 4))),
                         ("frame_with_choice", lambda: frame_discrepancies(P, Q))):
            times = []
            for _ in range(5):
                t0 = time.perf_counter()
                out = fn()
                times.append(time.perf_counter() - t0)
            rows.append({"N_Points": n, "Method": name, "Values": len(out), "Median_ms": 1e3 * float(np.median(times))})
    return pd.DataFrame(rows)


def main() -> None:
    if "--runtime-only" in sys.argv:
        save_runtime()
        return
    rng = np.random.default_rng(SEED)
    rows = []
    for scene in range(N_SCENES):
        sample, label = lab.gen_drift(rng)
        rows.append({"Scene": scene, "Label_Excessive": label, "Delta": sample["delta"], **scores(sample)})
    df = pd.DataFrame(rows)
    save_results(df, "frame_statistic_exploration.csv", SEED, "EXPLORATORY frame-coordinate statistic vs pencil family (drift benchmark)")
    y = df["Label_Excessive"].to_numpy()
    boot = np.random.default_rng(SEED + 1)
    idx = [boot.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    summary = []
    for name in ("pencil_q10", "frame_q10", "frame_q50"):
        s = df[name].to_numpy()
        ref = df["pencil_q10"].to_numpy()
        d = [lab.auc(y[i], s[i]) - lab.auc(y[i], ref[i]) for i in idx]
        summary.append({"Method": name, "N": len(y), "AUC": lab.auc(y, s), "TPR_at_5pct_FPR": lab.tpr_at_fpr(y, s),
                        "dAUC_vs_pencil_Lo": np.percentile(d, 2.5), "dAUC_vs_pencil_Hi": np.percentile(d, 97.5)})
        print(f"  {name:12s} AUC {summary[-1]['AUC']:.4f}  TPR@5% {summary[-1]['TPR_at_5pct_FPR']:.3f}")
    save_results(pd.DataFrame(summary), "frame_statistic_exploration_summary.csv", SEED,
                 "EXPLORATORY frame-coordinate statistic: AUC, TPR at 5% FPR, paired bootstrap vs pencil_q10")
    save_runtime()


def save_runtime() -> None:
    rt = runtime(np.random.default_rng(SEED + 2))
    print(rt.to_string(index=False), flush=True)
    save_results(rt, "frame_statistic_runtime.csv", SEED + 2, "EXPLORATORY runtime: frame statistic vs pencil family")


if __name__ == "__main__":
    main()
