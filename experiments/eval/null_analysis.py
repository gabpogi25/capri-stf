"""
experiments/eval/null_analysis.py

Analysis of null calibration (docs/prereg/evaluation_plan.md): sizes of the analytic tests,
the tuple-level Mahalanobis Delta_J test with the kappa gate, and conformal /
Mondrian-conformal p-values under exchangeability and under shift.
Output: null_calibration_sizes.csv, null_calibration_decisions.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from experiment_utils import RESULTS_DIR, save_results  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "eval"))
from mv_window import out_name  # noqa: E402
from forensics.stats.empirical_null import EmpiricalNull  # noqa: E402

N_POINTS = 40
DF1, DF_ALL = 3 * N_POINTS - 15, 9 * N_POINTS - 51
CONFORMAL = ["pencil_q10", "asinh_q10", "acat", "gauge_q90", "registration_rms", "sampson_mean", "glrt_all"]
ALPHAS = (0.01, 0.05, 0.10)


def cp(k: int, n: int) -> tuple[float, float]:
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def size_row(label: str, p: np.ndarray, **meta) -> dict:
    p = p[np.isfinite(p)]
    row = {"Test": label, **meta, "N": len(p), "KS_D": float(stats.kstest(p, "uniform").statistic) if len(p) else np.nan}
    for a in ALPHAS:
        k = int(np.sum(p <= a))
        row[f"Size_{a}"], (row[f"CI_lo_{a}"], row[f"CI_hi_{a}"]) = k / max(len(p), 1), cp(k, len(p))
    return row


def analytic(w: pd.DataFrame) -> list[dict]:
    rows = []
    for sigma, d in w[(w.Kind == "rigid") & ~w.abstain.astype(bool)].groupby("Sigma"):
        rows.append(size_row("f_ratio (sigma-free)", d.f_ratio_p.to_numpy(float), Sigma=sigma))
        rows.append(size_row("split_glrt chi2_{3n-15} (known sigma)", stats.chi2.sf(d.split_glrt.to_numpy(float), DF1), Sigma=sigma))
        rows.append(size_row("glrt_all chi2_{9n-51} (known sigma)", stats.chi2.sf(d.glrt_all.to_numpy(float), DF_ALL), Sigma=sigma))
        rows.append(size_row("split_glrt with 0.5 sigma (misspecified control)", stats.chi2.sf(4 * d.split_glrt.to_numpy(float), DF1), Sigma=sigma))
    return rows


def gate(t: pd.DataFrame) -> tuple[list[dict], dict]:
    """kappa_max on the calibration half (pooled over sigma): the largest of 10 log-spaced values from the 50th
    to the 99th percentile of calibration kappa whose calibration size(0.05) with sigma_hat is in [0.04, 0.06]."""
    cal, ev = t[t.Half == "calibration"], t[t.Half == "evaluation"]
    grid = np.exp(np.linspace(*np.log(np.quantile(cal.Kappa, [0.5, 0.99])), 10))
    valid = [k for k in grid if 0.04 <= np.mean(cal.p_hat[cal.Kappa <= k] <= 0.05) <= 0.06]
    ungated_cal = float(np.mean(cal.p_hat <= 0.05))
    kmax = float(max(valid)) if valid else float("nan")
    rows = []
    for sigma, e in ev.groupby("Sigma"):
        for name in ("oracle", "hat", "half"):
            rows.append(size_row(f"tuple Delta_J chi2_3 ({name}), ungated", e[f"p_{name}"].to_numpy(float), Sigma=sigma))
            if valid:
                kept = e[e.Kappa <= kmax]
                rows.append(size_row(f"tuple Delta_J chi2_3 ({name}), gated", kept[f"p_{name}"].to_numpy(float), Sigma=sigma,
                                     Retention=len(kept) / len(e)))
        if valid:
            kept = e[e.Kappa <= kmax]
            deciles = pd.qcut(kept.Kappa, 10, labels=False, duplicates="drop")
            for q, g in kept.groupby(deciles):
                rows.append(size_row("tuple Delta_J (hat), gated, kappa decile", g.p_hat.to_numpy(float), Sigma=sigma, Decile=int(q)))
    return rows, {"kappa_max": kmax, "ungated_calibration_size_hat": ungated_cal, "n_valid_grid": len(valid)}


def conformal(w: pd.DataFrame) -> list[dict]:
    w = w[~w.abstain.astype(bool)]
    rows = []
    rigid = w[w.Kind == "rigid"]
    cal05 = rigid[(rigid.Sigma == 0.5) & (rigid.Half == "calibration")]
    pooled_cal = rigid[rigid.Half == "calibration"]
    edges = np.quantile(pooled_cal.sigma_hat_px, [0.25, 0.5, 0.75])
    targets = [(f"rigid sigma={s}", rigid[(rigid.Sigma == s) & (rigid.Half == "evaluation")]) for s in sorted(rigid.Sigma.unique())]
    targets.append(("shift_gaussian sigma=0.5", w[w.Kind == "shift_gaussian"]))
    for stat in CONFORMAL:
        null = EmpiricalNull.fit(cal05[stat].to_numpy(float))
        bins = np.digitize(pooled_cal.sigma_hat_px, edges)
        mondrian = {b: EmpiricalNull.fit(pooled_cal[stat].to_numpy(float)[bins == b]) for b in range(4)}
        for name, data in targets:
            rows.append(size_row(f"conformal (cal sigma=0.5) {stat}", null.p_value(data[stat].to_numpy(float)), Target=name))
            b = np.digitize(data.sigma_hat_px, edges)
            p = np.array([mondrian[k].p_value(v)[0] for k, v in zip(b, data[stat].to_numpy(float))])
            rows.append(size_row(f"Mondrian conformal (sigma_hat quartiles) {stat}", p, Target=name))
    return rows


def main() -> None:
    w = pd.read_csv(RESULTS_DIR / out_name("null_calibration_windows.csv"), comment="#")
    t = pd.read_csv(RESULTS_DIR / out_name("null_calibration_tuples.csv"), comment="#")
    n_errors = int(t["Error"].notna().sum()) if "Error" in t else 0
    t = t[t.Kappa.notna() & t.p_hat.notna()] if "Kappa" in t else t
    gate_rows, info = gate(t)
    sizes = pd.DataFrame(analytic(w) + gate_rows + conformal(w))
    dec = []
    add = lambda item, value, rule: dec.append({"Item": item, "Value": value, "Rule": rule})   # noqa: E731
    add("tuple rows with a numerical failure (left out, logged in null_calibration_tuples.csv)", n_errors, "recorded deviation (null calibration)")
    add("kappa_max (calibration half)", info["kappa_max"], "largest grid value with calibration size in [4%, 6%]")
    add("ungated calibration size (hat)", info["ungated_calibration_size_hat"], "in [3.5%, 6.5%] -> gate removed (ablation)")
    for label in ("f_ratio (sigma-free)", "split_glrt chi2_{3n-15} (known sigma)", "glrt_all chi2_{9n-51} (known sigma)"):
        s = sizes[sizes.Test == label]
        add(f"{label}: size(0.05) all sigma in [3.5%, 6.5%]", bool(s["Size_0.05"].between(0.035, 0.065).all()), str(s["Size_0.05"].round(4).tolist()))
    mis = sizes[sizes.Test.str.contains("misspecified")]
    add("misspecified-sigma control fires (size >= 10%)", bool((mis["Size_0.05"] >= 0.10).all()), str(mis["Size_0.05"].round(4).tolist()))
    g = sizes[sizes.Test == "tuple Delta_J chi2_3 (hat), gated"]
    if len(g):
        add("tuple test gated (hat): size(0.05) in [4%, 6%] every sigma", bool(g["Size_0.05"].between(0.04, 0.06).all()), str(g["Size_0.05"].round(4).tolist()))
        add("tuple test gated (hat): KS D <= 0.02 every sigma", bool((g.KS_D <= 0.02).all()), str(g.KS_D.round(4).tolist()))
        add("tuple test gated (hat): size(0.01) <= 2%", bool((g["Size_0.01"] <= 0.02).all()), str(g["Size_0.01"].round(4).tolist()))
        add("tuple test gated: retention >= 50%", bool((g.Retention >= 0.5).all()), str(g.Retention.round(3).tolist()))
        dd = sizes[sizes.Test.str.contains("decile")]
        add("tuple test gated: every kappa decile in [3.5%, 6.5%]", bool(dd["Size_0.05"].between(0.035, 0.065).all()), f"{int((~dd['Size_0.05'].between(0.035, 0.065)).sum())} of {len(dd)} outside")
    half = sizes[sizes.Test == "tuple Delta_J chi2_3 (half), ungated"]
    add("tuple misspecified-sigma control fires (size >= 10%)", bool((half["Size_0.05"] >= 0.10).all()), str(half["Size_0.05"].round(4).tolist()))
    c = sizes[sizes.Test.str.startswith("conformal") & (sizes.Target == "rigid sigma=0.5")]
    add("conformal, exchangeable target: size(0.05) in [3.5%, 6.5%] (each statistic)", bool(c["Size_0.05"].between(0.035, 0.065).all()), str(c["Size_0.05"].round(4).tolist()))
    m = sizes[sizes.Test.str.startswith("Mondrian") & sizes.Target.str.startswith("rigid")]
    add("Mondrian conformal: size(0.05) in [3.5%, 6.5%] at every sigma (each statistic)", bool(m["Size_0.05"].between(0.035, 0.065).all()), f"{int((~m['Size_0.05'].between(0.035, 0.065)).sum())} of {len(m)} outside")
    dec = pd.DataFrame(dec)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(dec.to_string(index=False))
    save_results(sizes, out_name("null_calibration_sizes.csv"), 33_000_000, "null calibration: null sizes (analytic, tuple Mahalanobis, conformal)")
    save_results(dec, out_name("null_calibration_decisions.csv"), 33_000_000, "null calibration: pre-registered decisions")


if __name__ == "__main__":
    main()
