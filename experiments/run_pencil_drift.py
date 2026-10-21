"""
experiments/run_pencil_drift.py

The pencil drift test against the alternatives (docs/MATHEMATICS.md, Section 4).

Data model (frame-free, as for real reconstructions): 12 generic points;
two measurements, each point moved within a ball of radius r in the unit
chart; each measurement then put in an independent random projective frame
with random per-point scale and sign. Label 1 = excessive, r = f * delta with
f ~ U(1.5, 5); label 0 = bounded, r = delta; delta log-uniform in [0.003, 0.1].
This is the 'drift' track of experiments/math_lab with a fresh seed.

The rho / tau_exact test cannot be computed here: rho needs both
measurements in one coordinate chart, which frame-free data do not have
(docs/MATHEMATICS.md, Section 2, Interpretation). drift-admission experiment reports it in the
shared-chart model, where it rejects no excessive drift.

Methods (score: larger = more drift, all divided by delta):
  pencil_q10        Q_0.1 of |d log|CR|| over all 41,580 pencil cross-ratios
  registration_rms  RMS sine residual of a DLT projective fit in Tyler frames
  median_tuple_dJ   median over the 924 6-tuples of max_k |d log J_k|
  single_tuple_dJ   max_k |d log J_k| of one 6-tuple (one certification unit)

Output: pencil_drift.csv (per scene) and pencil_drift_summary.csv
(AUC, TPR at 5% FPR, bootstrap 95% CIs, paired differences vs registration).
Seed 42016.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "math_lab"))

from experiment_utils import get_experiment_seed, save_results  # noqa: E402
import lab  # noqa: E402
from algorithms.consistency.pencil import pencil_drift_score  # noqa: E402

EXPERIMENT_ID = 16
SEED = get_experiment_seed(EXPERIMENT_ID)
N_SCENES = 2000
N_BOOT = 1000

METHODS = {
    "pencil_q10": lambda s: pencil_drift_score(s["A"], s["B"], s["delta"]),
    "registration_rms": lab.b_drift_registration,
    "median_tuple_dJ": lab.b_drift_median_tuples,
    "single_tuple_dJ": lab.b_drift_single_tuple,
}


def main() -> None:
    print("=" * 60)
    print("EXP 6b: PENCIL DRIFT TEST (frame-free)")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    rows = []
    for scene in range(N_SCENES):
        sample, label = lab.gen_drift(rng)
        row = {"Scene": scene, "Label_Excessive": label, "Delta": sample["delta"], "Seed": SEED}
        row.update({name: fn(sample) for name, fn in METHODS.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    save_results(df, "pencil_drift.csv", SEED, "pencil drift experiment: pencil drift test, frame-free")

    y = df["Label_Excessive"].to_numpy()
    boot_rng = np.random.default_rng(SEED + 1)
    idx = [boot_rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    summary = []
    for name in METHODS:
        s = df[name].to_numpy()
        auc_b = [lab.auc(y[i], s[i]) for i in idx]
        tpr_b = [lab.tpr_at_fpr(y[i], s[i]) for i in idx]
        ref = df["registration_rms"].to_numpy()
        d_auc = [lab.auc(y[i], s[i]) - lab.auc(y[i], ref[i]) for i in idx]
        d_tpr = [lab.tpr_at_fpr(y[i], s[i]) - lab.tpr_at_fpr(y[i], ref[i]) for i in idx]
        summary.append({
            "Method": name, "N": len(y), "AUC": lab.auc(y, s),
            "AUC_CI_Lo": np.percentile(auc_b, 2.5), "AUC_CI_Hi": np.percentile(auc_b, 97.5),
            "TPR_at_5pct_FPR": lab.tpr_at_fpr(y, s),
            "TPR_CI_Lo": np.percentile(tpr_b, 2.5), "TPR_CI_Hi": np.percentile(tpr_b, 97.5),
            "dAUC_vs_registration_Lo": np.percentile(d_auc, 2.5), "dAUC_vs_registration_Hi": np.percentile(d_auc, 97.5),
            "dTPR_vs_registration_Lo": np.percentile(d_tpr, 2.5), "dTPR_vs_registration_Hi": np.percentile(d_tpr, 97.5),
            "Seed": SEED,
        })
        print(f"  {name:18s} AUC {summary[-1]['AUC']:.4f}  TPR@5%FPR {summary[-1]['TPR_at_5pct_FPR']:.3f}")
    save_results(pd.DataFrame(summary), "pencil_drift_summary.csv", SEED,
                 "pencil drift experiment: summary with bootstrap CIs (paired differences vs registration)")


if __name__ == "__main__":
    main()
