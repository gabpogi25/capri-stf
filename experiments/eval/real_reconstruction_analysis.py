"""
experiments/eval/real_reconstruction_analysis.py

Decision rule of docs/prereg/real_reconstruction.md, implemented verbatim.

Input: experiments/results/real_world_baselines_batch{1,2}.csv (run_real_world_baselines.py).
Output: real_reconstruction_decision.csv (primary comparison and decision) and
real_reconstruction_summary.csv (per pipeline, descriptive), both via save_results.

Primary: per unit, D = mean over seeds of mAA_at_10(superglue_ransac_capri) minus the
same for superglue_ransac. Mean of D over units with a 95% percentile bootstrap CI over
units (10,000 resamples, seed 86,500,000).
    benefit supported   CI lower end > 0
    harm                CI upper end < 0
    no difference shown otherwise; "equivalent within 0.01" if the CI lies inside
                        [-0.01, 0.01], else "inconclusive"
The decision is made once, on the full pre-registered set. With --interim BATCH the same
numbers are computed on the scenes run so far and written to *_interim_batchB_*.csv with
no decision.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_utils import RESULTS_DIR, save_results  # noqa: E402

TREATMENT = "superglue_ransac_capri"
CONTROL = "superglue_ransac"
PRIMARY = "mAA_at_10"
SECONDARY = ("N_Registered", "Median_Pose_Error_deg", "Acc_at_5", "Acc_at_10", "Acc_at_20",
             "ATE_RMSE_Sim3_GT_Units")
BOOTSTRAP_SEED = 86_500_000
N_BOOTSTRAP = 10_000
EQUIVALENCE_MARGIN = 0.01


def paired_differences(df: pd.DataFrame, metric: str) -> pd.Series:
    """D per unit: seed-mean of the treatment minus seed-mean of the control."""
    runs = df.groupby(["Dataset", "Scene", "Pipeline"]).size().unstack("Pipeline")
    assert TREATMENT in runs and CONTROL in runs, "both pipelines must be present"
    assert runs[[TREATMENT, CONTROL]].notna().all().all(), "every unit needs runs of both pipelines"
    per_unit = df.groupby(["Dataset", "Scene", "Pipeline"])[metric].mean().unstack("Pipeline")
    # inf - inf (a median error that is infinite in both pipelines) is undefined: NaN, dropped
    # by the caller for descriptive metrics; mAA_at_10 is always finite
    return per_unit[TREATMENT] - per_unit[CONTROL]


def bootstrap_ci(d: np.ndarray, seed: int = BOOTSTRAP_SEED, n: int = N_BOOTSTRAP) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean, resampling units."""
    assert d.ndim == 1 and d.size >= 2
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n, d.size))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def decide(low: float, high: float) -> str:
    """Decision rule of the pre-registration."""
    if low > 0:
        return "benefit supported"
    if high < 0:
        return "harm"
    if -EQUIVALENCE_MARGIN <= low and high <= EQUIVALENCE_MARGIN:
        return "no difference shown: equivalent within 0.01 mAA"
    return "no difference shown: inconclusive"


def interim_label(batch: int) -> str:
    """Interim rows carry no decision: the decision rule applies to the full set only."""
    return f"interim after batch {batch}: no decision (pre-registered analysis on the full set decides)"


def finite_mean(values: pd.Series) -> float:
    """Mean that treats infinite medians (most pairs unregistered) as NaN for display only."""
    return float(values.replace([np.inf, -np.inf], np.nan).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+",
                        default=[RESULTS_DIR / "real_world_baselines_batch1.csv",
                                 RESULTS_DIR / "real_world_baselines_batch2.csv"],
                        help="result files of the batches (the full set: batch 1 and batch 2)")
    parser.add_argument("--prefix", default="real_reconstruction")
    parser.add_argument("--interim", type=int, default=None, metavar="BATCH",
                        help="interim analysis after this batch: same numbers, no decision")
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(path, comment="#") for path in args.input], ignore_index=True)
    keys = ["Dataset", "Scene", "Pipeline", "Seed"]
    assert not df.duplicated(keys).any(), "a (scene, pipeline, seed) row appears in two batch files"
    if args.interim is not None:
        args.prefix = f"{args.prefix}_interim_batch{args.interim}"
    rows = []
    for metric in (PRIMARY,) + SECONDARY:
        d = paired_differences(df, metric).replace([np.inf, -np.inf], np.nan).dropna()
        low, high = bootstrap_ci(d.to_numpy())
        wilcoxon_p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        rows.append({
            "Metric": metric, "Role": "primary" if metric == PRIMARY else "descriptive",
            "N_Units": int(d.size), "Mean_D": float(d.mean()), "CI_Low": low, "CI_High": high,
            "Wilcoxon_p": wilcoxon_p, "Units_D_Positive": int((d > 0).sum()),
            "Units_D_Negative": int((d < 0).sum()),
            "Decision": (interim_label(args.interim) if args.interim is not None
                         else decide(low, high)) if metric == PRIMARY else "",
        })
    seeds = sorted(df["Seed"].unique())
    save_results(pd.DataFrame(rows), f"{args.prefix}_decision.csv", BOOTSTRAP_SEED,
                 f"Real reconstruction decision ({TREATMENT} vs {CONTROL}), seeds {seeds}")

    summary = df.groupby("Pipeline").agg(
        N_Runs=(PRIMARY, "size"), N_Failures=("Failure", lambda s: int((s.fillna("") != "").sum())),
        mAA_at_10_Mean=(PRIMARY, "mean"), Registered_Fraction=("N_Registered", "sum"),
        Median_Pose_Error_Mean=("Median_Pose_Error_deg", finite_mean),
    ).reset_index()
    total_images = df.groupby("Pipeline")["N_Images"].sum().to_numpy()
    summary["Registered_Fraction"] = summary["Registered_Fraction"] / total_images
    save_results(summary, f"{args.prefix}_summary.csv", BOOTSTRAP_SEED,
                 "Real reconstruction, per-pipeline summary (descriptive)")


if __name__ == "__main__":
    main()
