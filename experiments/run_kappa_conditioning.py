"""
experiments/run_kappa_conditioning.py

Which conditioning score predicts the real noise error of the
invariants? (docs/MATHEMATICS.md, Section 5)

Configurations: generic and near-degenerate six-tuples (4 or 5 nearly
coplanar points, 3 nearly collinear; offset log-uniform in [1e-3, 0.3]) in a
random projective frame, unit columns. Target: median over 300 draws of
max_k |log J_k(P + E) - log J_k(P)| with E ~ N(0, 1e-3 I) in the given
coordinates (the 'cond' track of experiments/math_lab, fresh seed).

Scores: kappa_J (exact gradient), log L(S), -log sigma_4.
Output: kappa_conditioning.csv (per tuple) and _summary.csv
(Spearman correlation with the target, AUC for target > 0.05, bootstrap CIs),
plus the first-order regime flag sigma * kappa_J < 0.1.
Seed 42017.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "math_lab"))

from experiment_utils import get_experiment_seed, save_results  # noqa: E402
import lab  # noqa: E402
from algorithms.conditioning.scores import (  # noqa: E402
    compute_conditioning_score, compute_invariant_noise_gain, compute_singular_value_conditioning,
)

EXPERIMENT_ID = 17
SEED = get_experiment_seed(EXPERIMENT_ID)
N_TUPLES = 1500
N_BOOT = 1000
SIGMA = 1e-3


def main() -> None:
    print("=" * 60)
    print("EXP 4b: CONDITIONING SCORES VS REAL NOISE ERROR")
    print("=" * 60)
    rng = np.random.default_rng(SEED)
    rows = []
    t = tuple(range(6))
    for k in range(N_TUPLES):
        sample, target = lab.gen_cond(rng, sigma=SIGMA)
        P = sample["P"]
        kappa = compute_invariant_noise_gain(P, t)
        rows.append({"Tuple": k, "Target_Median_Max_dlogJ": target, "Kappa_J": kappa,
                     "Log_L_S": float(np.log(compute_conditioning_score(P, t))),
                     "Neg_Log_Sigma4": float(-np.log(compute_singular_value_conditioning(P, t))),
                     "First_Order_Regime": bool(SIGMA * kappa < 0.1), "Seed": SEED})
    df = pd.DataFrame(rows)
    save_results(df, "kappa_conditioning.csv", SEED, "kappa_J conditioning experiment: conditioning scores vs noise error")

    target = df["Target_Median_Max_dlogJ"].to_numpy()
    positive = (target > 0.05).astype(int)
    boot_rng = np.random.default_rng(SEED + 1)
    idx = [boot_rng.integers(0, len(target), len(target)) for _ in range(N_BOOT)]
    summary = []
    for name in ("Kappa_J", "Log_L_S", "Neg_Log_Sigma4"):
        s = df[name].to_numpy()
        rho_b = [stats.spearmanr(s[i], target[i]).statistic for i in idx]
        summary.append({"Score": name, "N": len(s), "Spearman": stats.spearmanr(s, target).statistic,
                        "Spearman_CI_Lo": np.percentile(rho_b, 2.5), "Spearman_CI_Hi": np.percentile(rho_b, 97.5),
                        "AUC_target_gt_0.05": lab.auc(positive, s), "Positive_Fraction": positive.mean(),
                        "Seed": SEED})
        print(f"  {name:15s} Spearman {summary[-1]['Spearman']:.4f}  AUC {summary[-1]['AUC_target_gt_0.05']:.4f}")
    regime = df["First_Order_Regime"].to_numpy()
    ratio = target[regime] / (SIGMA * df["Kappa_J"].to_numpy()[regime])
    print(f"  first-order regime (sigma*kappa < 0.1): {regime.mean():.2%} of tuples; "
          f"target/(sigma kappa) in [{ratio.min():.3f}, {ratio.max():.3f}] (Prop. 5.1 band 0.674-1.383)")
    save_results(pd.DataFrame(summary), "kappa_conditioning_summary.csv", SEED,
                 "kappa_J conditioning experiment: summary with bootstrap CIs")


if __name__ == "__main__":
    main()
