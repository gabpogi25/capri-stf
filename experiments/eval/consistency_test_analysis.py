"""
experiments/eval/consistency_test_analysis.py

Analysis of consistency_test_windows.csv. Every rule is copied from docs/prereg/consistency_test.md;
nothing is tuned here.

Common rules. Thresholds: the 95th percentile of the 400 'cal' windows (S0, sigma 0.5 px) for each
statistic. Abstained windows and NaN statistics enter intention-to-treat with the median 'cal'
value (never imputed from the label). Size = rejection rate; the band is [3.5%, 6.5%].
Statistics: pencil_q10, std_native_q10, std_tyler_q10, acat, sampson_mean, split_glrt, frame_q50.

H1 (geometry transfer). std_tyler_q10 transfers to scene S1 if its size on 'shift' lies in the band.
H1b (replication). The same rule for acat (standardised in native frames with the fixed m0 of
  mv_window, Cauchy combination), the statistic whose null transferred in null_calibration.
H2 (noise misspecification). Plug-in noise sigma_hat = f * sigma, f in {0.5, 0.75, 1.5, 2}. With the
  threshold c set at f = 1 on 'cal', the test at f rejects when the statistic exceeds c * f for the
  sigma-linear statistics (pencil_q10, std_tyler_q10) and c * f^2 for the sigma-quadratic ones
  (split_glrt, sampson_mean); split_glrt is also tested against its chi^2_{3n-15} quantile times
  f^2. Size on 'test'. Supported if, at f = 0.75 and at f = 1.5, |size - 0.05| of pencil_q10 is
  smaller than that of split_glrt (empirical threshold).
H3 (combination). Conformal p-values of pencil_q10 and sampson_mean against 'cal'; Cauchy
  combination (forensics/stats/combine.py); reject when the combined p <= 0.05. Requires size on
  'test' in the band. Power on jitter/drift/warp/swap against the better single statistic at its
  own threshold. Supported if, on every type, the paired-bootstrap 95% CI (2,000 resamples) of
  power(combined) - max(power single) has lower end > -0.02, and on at least one type > 0.
Descriptive: AUC against 'test' per statistic and type (paired bootstrap CI against pencil_q10);
  sizes at sigma 0.25 and 1 px with f = 1: thresholds scaled by sigma/0.5 for the raw statistics that
  are linear in sigma (pencil_q10, std_native_q10, std_tyler_q10, frame_q50), by (sigma/0.5)^2 for
  sampson_mean, and not at all for split_glrt (already divided by the generating sigma^2);
  frame_q50 on fresh seeds.
Output: consistency_test_sizes.csv, consistency_test_misspecification.csv, consistency_test_power.csv,
consistency_test_auc.csv, consistency_test_decisions.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiment_utils import RESULTS_DIR, save_results  # noqa: E402
from forensics.stats.combine import cauchy_combination  # noqa: E402
import mv_window as mw  # noqa: E402

STATS = ["pencil_q10", "std_native_q10", "std_tyler_q10", "acat", "sampson_mean", "split_glrt", "frame_q50"]
LINEAR, QUADRATIC = ("pencil_q10", "std_tyler_q10"), ("split_glrt", "sampson_mean")
TYPES = ["jitter", "drift", "warp", "swap"]
BAND, ALPHA, N_BOOT, SEED = (0.035, 0.065), 0.05, 2000, 81_900_000
N_POINTS = 40


def read() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / mw.out_name("consistency_test_windows.csv"), comment="#")


def itt(df: pd.DataFrame, stat: str, fill: float) -> np.ndarray:
    values = df[stat].to_numpy(float)
    return np.where(df["abstain"].astype(bool).to_numpy() | ~np.isfinite(values), fill, values)


def auc(neg: np.ndarray, pos: np.ndarray) -> float:
    r = stats.rankdata(np.r_[neg, pos])
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def conformal_p(cal: np.ndarray, x: np.ndarray) -> np.ndarray:
    """p = (1 + #{cal >= x}) / (n_cal + 1)."""
    cal = np.sort(cal)
    return (1 + len(cal) - np.searchsorted(cal, x, side="left")) / (len(cal) + 1)


def main() -> None:
    df = read()
    cal = df[df["Role"] == "cal"]
    fill = {s: float(np.nanmedian(cal[s])) for s in STATS}
    calv = {s: itt(cal, s, fill[s]) for s in STATS}
    thr = {s: float(np.quantile(calv[s], 0.95)) for s in STATS}
    role = lambda r: df[df["Role"] == r]                                            # noqa: E731
    size_rows, decisions = [], []
    for r in ("test", "sig025", "sig1", "shift"):
        sub = role(r)
        for s in STATS:
            v = itt(sub, s, fill[s])
            scale = 1.0
            if r in ("sig025", "sig1"):                                             # f = 1: sigma_hat = sigma
                ratio = sub["Sigma"].iloc[0] / 0.5
                scale = (1.0 if s == "split_glrt" else ratio ** 2 if s == "sampson_mean"
                         else np.nan if s == "acat" else ratio)
            size_rows.append({"Role": r, "Statistic": s, "N": len(v), "Coverage": float(1 - sub["abstain"].mean()),
                              "Size": float(np.mean(v > thr[s] * scale)) if np.isfinite(scale) else np.nan})
    sizes = pd.DataFrame(size_rows)
    sz = lambda r, s: float(sizes[(sizes["Role"] == r) & (sizes["Statistic"] == s)]["Size"].iloc[0])   # noqa: E731
    h1 = BAND[0] <= sz("shift", "std_tyler_q10") <= BAND[1]
    decisions.append({"Item": "H1 size of std_tyler_q10 on scene S1", "Value": sz("shift", "std_tyler_q10"),
                      "Rule": "in [0.035, 0.065]", "Supported": h1})
    h1b = BAND[0] <= sz("shift", "acat") <= BAND[1]
    decisions.append({"Item": "H1b size of acat (native frames, fixed m0) on scene S1", "Value": sz("shift", "acat"),
                      "Rule": "in [0.035, 0.065]", "Supported": h1b})
    shift = role("shift")
    chi_shift = stats.chi2.ppf(1 - ALPHA, 3 * 60 - 15)
    decisions.append({"Item": "descriptive: split_glrt on S1 against chi^2_{3n-15} with n = 60",
                      "Value": float(np.mean(itt(shift, "split_glrt", fill["split_glrt"]) > chi_shift)),
                      "Rule": "none (descriptive)", "Supported": ""})

    test = role("test")
    chi = stats.chi2.ppf(1 - ALPHA, 3 * N_POINTS - 15)
    mis_rows = []
    for f in (0.5, 0.75, 1.0, 1.5, 2.0):
        for s in ("pencil_q10", "std_tyler_q10", "split_glrt", "sampson_mean"):
            power = f if s in LINEAR else f ** 2
            mis_rows.append({"f": f, "Statistic": s, "Threshold": "empirical",
                             "Size": float(np.mean(itt(test, s, fill[s]) > thr[s] * power))})
        mis_rows.append({"f": f, "Statistic": "split_glrt", "Threshold": "chi2",
                         "Size": float(np.mean(itt(test, "split_glrt", fill["split_glrt"]) > chi * f ** 2))})
    mis = pd.DataFrame(mis_rows)
    ms = lambda f, s, t="empirical": float(mis[(mis["f"] == f) & (mis["Statistic"] == s) & (mis["Threshold"] == t)]["Size"].iloc[0])  # noqa: E731
    h2 = all(abs(ms(f, "pencil_q10") - ALPHA) < abs(ms(f, "split_glrt") - ALPHA) for f in (0.75, 1.5))
    decisions.append({"Item": "H2 |size - 0.05| pencil_q10 < split_glrt at f = 0.75 and 1.5",
                      "Value": "; ".join(f"f={f}: {ms(f, 'pencil_q10'):.3f} vs {ms(f, 'split_glrt'):.3f}" for f in (0.75, 1.5)),
                      "Rule": "both f", "Supported": h2})

    def combined_p(sub):
        pp = conformal_p(calv["pencil_q10"], itt(sub, "pencil_q10", fill["pencil_q10"]))
        ps = conformal_p(calv["sampson_mean"], itt(sub, "sampson_mean", fill["sampson_mean"]))
        return np.array([cauchy_combination(np.array([a, b])) for a, b in zip(pp, ps)])
    size_comb = float(np.mean(combined_p(test) <= ALPHA))
    decisions.append({"Item": "H3 size of the combination on test", "Value": size_comb, "Rule": "in [0.035, 0.065]",
                      "Supported": BAND[0] <= size_comb <= BAND[1]})
    rng = np.random.default_rng(SEED)
    pow_rows, h3_ok, h3_gain = [], BAND[0] <= size_comb <= BAND[1], False
    for t in TYPES:
        sub = role(t)
        rej = {"combined": combined_p(sub) <= ALPHA}
        for s in ("pencil_q10", "sampson_mean"):
            rej[s] = itt(sub, s, fill[s]) > thr[s]
        best = max(("pencil_q10", "sampson_mean"), key=lambda s: rej[s].mean())
        diff = [rej["combined"][i].mean() - rej[best][i].mean()
                for i in (rng.integers(0, len(sub), len(sub)) for _ in range(N_BOOT))]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        h3_ok &= lo > -0.02
        h3_gain |= lo > 0
        pow_rows.append({"Type": t, "Power_combined": rej["combined"].mean(), "Power_pencil": rej["pencil_q10"].mean(),
                         "Power_sampson": rej["sampson_mean"].mean(), "Best_single": best, "Diff_CI_Lo": lo, "Diff_CI_Hi": hi})
    decisions.append({"Item": "H3 combination power", "Value": "see consistency_test_power.csv",
                      "Rule": "every type CI lo > -0.02 and one type CI lo > 0", "Supported": bool(h3_ok and h3_gain)})

    auc_rows = []
    neg = {s: itt(test, s, fill[s]) for s in STATS}
    for t in TYPES:
        sub = role(t)
        pos = {s: itt(sub, s, fill[s]) for s in STATS}
        idx = [(rng.integers(0, len(test), len(test)), rng.integers(0, len(sub), len(sub))) for _ in range(N_BOOT)]
        for s in STATS:
            d = [auc(neg[s][i], pos[s][j]) - auc(neg["pencil_q10"][i], pos["pencil_q10"][j]) for i, j in idx]
            auc_rows.append({"Type": t, "Statistic": s, "AUC": auc(neg[s], pos[s]),
                             "dAUC_vs_pencil_Lo": np.percentile(d, 2.5), "dAUC_vs_pencil_Hi": np.percentile(d, 97.5)})

    for name, frame in (("sizes", sizes), ("misspecification", mis), ("power", pd.DataFrame(pow_rows)),
                        ("auc", pd.DataFrame(auc_rows)), ("decisions", pd.DataFrame(decisions))):
        save_results(frame, mw.out_name(f"consistency_test_{name}.csv"), SEED, f"consistency_test analysis: {name}")
    print(pd.DataFrame(decisions).to_string(index=False))


if __name__ == "__main__":
    main()
