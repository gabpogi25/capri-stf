"""
experiments/eval/clip_analysis.py

Analysis of clip aggregation (docs/prereg/evaluation_plan.md): clip-level aggregation, K-sweep,
view split, U-statistic, CUSUM and the e-process, from clip_windows.csv.

Per-window p-values: conformal from the 400 calibration windows (EmpiricalNull) for
pencil_q10, glrt_all and sampson_mean; analytic for f_ratio_p. Abstained windows get
p = 1 (no evidence; never imputed from the label). Clip aggregators over the 8
windows: naive Fisher, Cauchy combination (CCT), harmonic mean p (raw, compared with
alpha), Bonferroni minimum, block Fisher, e-value product (kappa = 0.5, reject at
>= 1/alpha). alpha = 0.05.
Output: clip_aggregation.csv, clip_decisions.csv.
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
from forensics.stats.combine import bonferroni_min, cauchy_combination, e_process_alarm, e_product, harmonic_mean_p  # noqa: E402
from forensics.stats.empirical_null import EmpiricalNull  # noqa: E402
from forensics.stats.fisher import block_fisher, fisher_combine  # noqa: E402

ALPHA, ARL_TARGET, CUSUM_K = 0.05, 50.0, 0.5
STATS = ("pencil_q10", "glrt_all", "sampson_mean", "f_ratio")
AGGREGATORS = {"fisher_naive": lambda p: fisher_combine(p)[1], "cct": cauchy_combination, "hmp_raw": harmonic_mean_p,
               "bonferroni": bonferroni_min, "block_fisher": lambda p: block_fisher(p)[1]}
K_SWEEP = (100, 300, 1000, 3000, 10_000, 50_000)


def window_p(df: pd.DataFrame, cal: pd.DataFrame, stat: str) -> np.ndarray:
    covered = ~df.abstain.astype(bool).to_numpy()
    values = df[stat if stat != "f_ratio" else "f_ratio_p"].to_numpy(float)
    if stat == "f_ratio":
        p = values
    else:
        null = EmpiricalNull.fit(cal[stat][~cal.abstain.astype(bool)].to_numpy(float))
        p = null.p_value(np.where(np.isfinite(values), values, -np.inf))
    return np.where(covered & np.isfinite(p), p, 1.0)


def clip_table(df: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stat in STATS:
        clips = df[df.Role != "calibration"].copy()
        clips["p"] = window_p(clips, cal, stat)
        for (role, clip), g in clips.groupby(["Role", "Clip"]):
            p = g.sort_values("Window")["p"].to_numpy()
            row = {"Statistic": stat, "Role": role, "Clip": clip}
            row.update({name: fn(p) <= ALPHA for name, fn in AGGREGATORS.items()})
            row["e_product"] = e_product(p) >= 1 / ALPHA
            row["e_alarm"] = e_process_alarm(p, ALPHA)
            rows.append(row)
    return pd.DataFrame(rows)


def cusum_alarms(z: np.ndarray, h: float) -> list[int]:
    alarms, s = [], 0.0
    for t, v in enumerate(z):
        s = max(0.0, s + v - CUSUM_K)
        if s > h:
            alarms.append(t)
            s = 0.0
    return alarms


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / out_name("clip_windows.csv"), comment="#")
    cal = df[df.Role == "calibration"]
    table = clip_table(df, cal)
    names = list(AGGREGATORS) + ["e_product"]
    agg = table.groupby(["Statistic", "Role"])[names].mean().reset_index()
    dec = []
    add = lambda item, value, rule: dec.append({"Item": item, "Value": value, "Rule": rule})   # noqa: E731
    for stat in STATS:
        a = agg[agg.Statistic == stat].set_index("Role")
        eligible = [n for n in AGGREGATORS if 0.035 <= a.loc["rigid", n] <= 0.065]
        add(f"{stat}: eligible aggregators (clip size in [3.5%, 6.5%])", ", ".join(eligible) or "none", "")
        if eligible:
            best = max(eligible, key=lambda n: min(a.loc[f, n] for f in ("jitter", "drift", "warp")))
            add(f"{stat}: V6 winner (highest min power jitter/drift/warp)", f"{best} ({min(a.loc[f, best] for f in ('jitter', 'drift', 'warp')):.3f})", "")
        add(f"{stat}: e-product size (valid iff 0 < size <= 6.5%)", float(a.loc["rigid", "e_product"]), "anytime-valid")
    # K-sweep (window level, drift clips at d*, thresholds from calibration windows at the same K)
    f3 = df[df.Role == "drift"]
    powers = {K: float(np.mean(f3[f"pencil_q10_K{K}"] > cal[f"pencil_q10_K{K}"].quantile(0.95))) for K in K_SWEEP}
    k_star = min(K for K in K_SWEEP if powers[K] >= powers[50_000] - 0.01)
    add("K-sweep window power (drift at d*)", str({k: round(v, 3) for k, v in powers.items()}), "")
    add("K* (smallest K within 0.01 of K = 50,000)", k_star, "")
    # view split and U-statistic (window level, AUC null vs families)
    def auc(neg, pos):
        neg, pos = neg[np.isfinite(neg)], pos[np.isfinite(pos)]
        r = stats.rankdata(np.r_[neg, pos])
        return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    sub = df[df.pencil_q10_contiguous.notna()] if "pencil_q10_contiguous" in df else df.iloc[:0]
    for f in ("jitter", "drift", "warp"):
        neg, pos = sub[sub.Role == "rigid"], sub[sub.Role == f]
        add(f"split AUC {f}: interleaved vs contiguous", f"{auc(neg.pencil_q10.to_numpy(float), pos.pencil_q10.to_numpy(float)):.4f} vs "
            f"{auc(neg.pencil_q10_contiguous.to_numpy(float), pos.pencil_q10_contiguous.to_numpy(float)):.4f}", "descriptive")
    u = df[df.pencil_q10_U.notna()] if "pencil_q10_U" in df else df.iloc[:0]
    add("window-0 AUC drift: interleaved vs U-statistic", f"{auc(u[u.Role == 'rigid'].pencil_q10.to_numpy(float), u[u.Role == 'drift'].pencil_q10.to_numpy(float)):.4f} vs "
        f"{auc(u[u.Role == 'rigid'].pencil_q10_U.to_numpy(float), u[u.Role == 'drift'].pencil_q10_U.to_numpy(float)):.4f}", "descriptive")
    # CUSUM on z = Phi^-1(1 - p) of the pencil statistic
    clips = df[df.Role.isin(["rigid", "change"])].copy()
    clips["z"] = stats.norm.isf(np.clip(window_p(clips, cal, "pencil_q10"), 1e-12, 1 - 1e-12))
    null = clips[clips.Role == "rigid"].sort_values(["Clip", "Window"])
    first, second = null[null.Clip < 150].z.to_numpy(), null[null.Clip >= 150].z.to_numpy()
    grid = np.arange(1.0, 12.01, 0.25)
    arl = {h: len(first) / max(len(cusum_alarms(first, h)), 1) for h in grid}
    h = float(min([g for g in grid if arl[g] >= ARL_TARGET], default=grid[-1]))
    arl_eval = len(second) / max(len(cusum_alarms(second, h)), 1) if len(second) else float("nan")
    add("CUSUM h (ARL0 target 50 on null clips 0-149)", h, "")
    add("CUSUM ARL0 on null clips 150-299", arl_eval, "within +/-30% of 50")
    delays, e_delays = [], []
    for _, g in clips[clips.Role == "change"].groupby("Clip"):
        z = g.sort_values("Window").z.to_numpy()
        hits = [t for t in cusum_alarms(z, h) if t >= 4]
        delays.append(hits[0] - 4 if hits else np.inf)
        p = stats.norm.sf(z)
        alarm = e_process_alarm(p[4:], ALPHA)
        e_delays.append(alarm if alarm is not None else np.inf)
    add("CUSUM median delay after change (windows; inf = not within clip)", float(np.median(delays)), "<= 2")
    add("CUSUM detected within clip", float(np.mean(np.isfinite(delays))), "descriptive")
    add("e-process median delay (from window 4)", float(np.median(e_delays)), "descriptive")
    nul = table[(table.Statistic == "pencil_q10") & (table.Role == "rigid")]
    add("e-process false-alarm rate on null clips (any alarm in 8 windows)", float(nul.e_alarm.notna().mean()), "<= 0.05 (anytime-valid)")
    dec = pd.DataFrame(dec)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    print(agg.to_string(index=False))
    print(dec.to_string(index=False))
    save_results(agg, out_name("clip_aggregation.csv"), 34_000_000, "clip aggregation: clip-level rejection rates per aggregator")
    save_results(dec, out_name("clip_decisions.csv"), 34_000_000, "clip aggregation: pre-registered decisions and descriptives")


if __name__ == "__main__":
    main()
