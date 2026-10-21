"""
experiments/eval/detector_benchmark.py

Analysis of injection grid (kill switch, dose-response, injection harness) and detector benchmark (classical-test benchmark) from injection_windows.csv. Every rule below is
copied from docs/prereg/evaluation_plan.md; nothing is tuned here.

Thresholds: per sigma and statistic, the 95th percentile of the covered rigid
calibration windows. Abstained windows enter intention-to-treat (ITT) with the
median calibration score (never imputed from the label; rule R4); covered
results are reported next to them. Size = ITT rejection rate on rigid test windows;
power and AUC (against the rigid test windows of the same sigma) per cell.
Output: detector_benchmark_cells.csv, detector_benchmark_decisions.csv.
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
from forensics.eval.metrics import delong_paired_test  # noqa: E402

PRIMARY = "pencil_q10"
STATS = ["pencil_q10", "pencil_q50", "pencil_q90", "asinh_q10", "acat", "gauge_q90", "gauge_med", "deltaj_med",
         "registration_rms", "split_glrt", "glrt_all", "f_ratio", "sampson_mean", "magsac_outlier"]
GLRT = ("split_glrt", "glrt_all")
REALIZABLE = ("jitter", "drift", "warp", "swap")


def itt(df: pd.DataFrame, stat: str, fill: float) -> np.ndarray:
    values = df[stat].to_numpy(float)
    return np.where(df["abstain"].astype(bool).to_numpy() | ~np.isfinite(values), fill, values)


def auc(neg: np.ndarray, pos: np.ndarray) -> float:
    r = stats.rankdata(np.r_[neg, pos])
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def jonckheere(groups: list[np.ndarray]) -> float:
    """One-sided (increasing) Jonckheere-Terpstra p-value, normal approximation."""
    S = sum(np.sum(groups[j][None, :] > groups[i][:, None]) + 0.5 * np.sum(groups[j][None, :] == groups[i][:, None])
            for i in range(len(groups)) for j in range(i + 1, len(groups)))
    n = np.array([len(g) for g in groups])
    N = n.sum()
    mean = (N ** 2 - np.sum(n ** 2)) / 4
    var = (N ** 2 * (2 * N + 3) - np.sum(n ** 2 * (2 * n + 3))) / 72
    return float(stats.norm.sf((S - mean) / np.sqrt(var)))


def cells(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, store = [], {}
    for sigma, d in df.groupby("Sigma"):
        cal = d[(d.Family == "rigid") & (d.Role == "calibration")]
        test0 = d[(d.Family == "rigid") & (d.Role == "test")]
        for stat in STATS:
            covered = cal[stat][~cal.abstain.astype(bool)].astype(float)
            fill, threshold = float(np.nanmedian(covered)), float(np.nanquantile(covered, 0.95))
            neg = itt(test0, stat, fill)
            store[(sigma, stat, "rigid", 0.0)] = neg
            for (family, dose), c in d[d.Role == "test"].groupby(["Family", "Dose"]):
                pos = itt(c, stat, fill)
                store[(sigma, stat, family, dose)] = pos
                cov = ~c.abstain.astype(bool)
                rows.append({"Sigma": sigma, "Statistic": stat, "Family": family, "Dose": dose, "N": len(c),
                             "Coverage": float(cov.mean()), "Reject_ITT": float(np.mean(pos > threshold)),
                             "Reject_covered": float(np.mean(c[stat][cov].astype(float) > threshold)) if cov.any() else np.nan,
                             "AUC_vs_rigid": auc(neg, pos) if family != "rigid" else np.nan,
                             "Induced_px": float(c.Induced_px.mean()), "Threshold": threshold,
                             "MAGSAC_removed_moved": (float(c.magsac_removed_moved.mean()) if family != "rigid" and c.magsac_removed_moved.notna().any()
                                                     else np.nan)})
    return pd.DataFrame(rows), store


def size_ok(cell: pd.DataFrame, sigma: float, stat: str) -> bool:
    size = cell[(cell.Sigma == sigma) & (cell.Statistic == stat) & (cell.Family == "rigid")]["Reject_ITT"].iloc[0]
    return 0.035 <= size <= 0.065


def d_star(cell: pd.DataFrame, family: str) -> float:
    c = cell[(cell.Sigma == 0.5) & (cell.Statistic == PRIMARY) & (cell.Family == family)]
    hits = c[c.Reject_ITT >= 0.5]
    return float(hits.Dose.min()) if len(hits) else float(c.Dose.max())


def decisions(cell: pd.DataFrame, store: dict) -> pd.DataFrame:
    out = []
    add = lambda item, value, rule: out.append({"Item": item, "Value": value, "Rule": rule})   # noqa: E731
    jt, rho = {}, {}
    for stat in STATS:
        for family in REALIZABLE + ("global_projective",):
            levels = [0.0] + sorted(cell[(cell.Sigma == 0.5) & (cell.Family == family)].Dose.unique())
            groups = [store[(0.5, stat, "rigid" if dose == 0 else family, dose)] for dose in levels]
            jt[(stat, family)] = jonckheere(groups)
            rho[(stat, family)] = stats.spearmanr(levels, [np.median(g) for g in groups])[0]
    power = lambda stat, fam, dose, sigma=0.5: float(cell[(cell.Sigma == sigma) & (cell.Statistic == stat) & (cell.Family == fam)  # noqa: E731
                                                         & (cell.Dose == dose)]["Reject_ITT"].iloc[0])
    p = PRIMARY
    for sigma in sorted(cell.Sigma.unique()):
        add(f"size_rigid_test sigma={sigma} ({p})", float(cell[(cell.Sigma == sigma) & (cell.Statistic == p) & (cell.Family == "rigid")]["Reject_ITT"].iloc[0]), "[0.035, 0.065]")
    kill = sum(jt[(p, f)] < 1e-3 for f in ("jitter", "drift", "warp")) >= 2 and max(power(p, f, 8.0) for f in ("jitter", "drift", "warp")) >= 0.8
    for f in ("jitter", "drift", "warp"):
        add(f"JT p {f} ({p})", jt[(p, f)], "< 1e-3 on >= 2 of jitter/drift/warp")
        add(f"power 8px sigma=0.5 {f} ({p})", power(p, f, 8.0), ">= 0.8 on >= 1")
    add("KILL SWITCH continuation (pencil responds)", bool(kill), "False -> stop")
    full = []
    for f in REALIZABLE:
        top = 8.0 if f != "swap" else 0.3
        pw = power(p, f, top, 1.0) if f != "swap" else power(p, f, top)
        ok = jt[(p, f)] < 1e-3 and rho[(p, f)] >= 0.9 and pw >= 0.8
        full.append(ok)
        add(f"full pass {f}: JT, Spearman, power ({'8px sigma=1' if f != 'swap' else 'swap 0.3'})",
            f"{jt[(p, f)]:.2e}, {rho[(p, f)]:.3f}, {pw:.3f} -> {ok}", "JT < 1e-3, rho >= 0.9, power >= 0.8")
    f7 = cell[(cell.Statistic == p) & (cell.Family == "planar")]["Coverage"].iloc[0]
    add("planar abstention rate", 1 - f7, ">= 0.9")
    for stat in STATS:
        add(f"global_projective control max power ({stat})", float(cell[(cell.Sigma == 0.5) & (cell.Statistic == stat) & (cell.Family == "global_projective")]["Reject_ITT"].max()), "<= 0.10")
    add("FULL PASS (pencil)", bool(all(full) and 1 - f7 >= 0.9 and all(size_ok(cell, s, p) for s in cell.Sigma.unique())), "all of the above")
    # ---- detector benchmark / classical-test benchmark at d* of each family
    star = {f: d_star(cell, f) for f in REALIZABLE}
    a = lambda stat, f, dose: float(cell[(cell.Sigma == 0.5) & (cell.Statistic == stat) & (cell.Family == f) & (cell.Dose == dose)]["AUC_vs_rigid"].iloc[0])  # noqa: E731
    g_best = lambda f, dose: max(a(s, f, dose) for s in GLRT)   # noqa: E731
    rule1 = all(a(p, f, d) < g_best(f, d) - 0.05 for f in REALIZABLE for d in cell[(cell.Family == f) & (cell.Sigma == 0.5)].Dose.unique())
    rule2 = any(a(p, f, star[f]) >= g_best(f, star[f]) for f in REALIZABLE)
    valid = [s for s in STATS if size_ok(cell, 0.5, s)]
    best_all = lambda f: max(a(s, f, star[f]) for s in (valid or STATS))   # noqa: E731
    add("statistics with size in band at sigma = 0.5 (eligible for 'best')", ", ".join(valid) or "none (all used)", "R3")
    rule3 = all(a("sampson_mean", f, star[f]) >= best_all(f) - 0.02 for f in REALIZABLE)
    decision = all(max(a("sampson_mean", f, star[f]), g_best(f, star[f])) >= a(p, f, star[f]) - 0.01 for f in REALIZABLE)
    for f in REALIZABLE:
        add(f"d* {f}", star[f], "smallest dose with pencil power >= 0.5 (else largest)")
        for s in STATS:
            neg, pos = store[(0.5, s, "rigid", 0.0)], store[(0.5, s, f, star[f])]
            test = delong_paired_test(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[store[(0.5, p, "rigid", 0.0)], store[(0.5, p, f, star[f])]], np.r_[neg, pos])
            add(f"AUC at d* {f}: {s}", f"{a(s, f, star[f]):.4f} (pencil - this: {test['difference']:+.4f}, DeLong p {test['p_value']:.2e}; size ok {size_ok(cell, 0.5, s)})", "descriptive")
    add("classical-test benchmark rule 1: pencil below GLRT by > 0.05 everywhere -> GLRT becomes S_alg", bool(rule1), "")
    add("classical-test benchmark rule 2: pencil >= GLRT on >= 1 family at d* -> keep both", bool(rule2), "")
    add("classical-test benchmark rule 3: Sampson within 0.02 of best on every family -> drop 'invariants add value'", bool(rule3), "")
    add("PLAN STEP 9 DECISION: Sampson or GLRT matches/beats pencil on every family (methods-paper framing)", bool(decision), "")
    return pd.DataFrame(out)


def main() -> None:
    df = pd.read_csv(RESULTS_DIR / out_name("injection_windows.csv"), comment="#")
    cell, store = cells(df)
    dec = decisions(cell, store)
    pd.set_option("display.width", 250, "display.max_colwidth", 140)
    print(dec.to_string(index=False))
    save_results(cell, out_name("detector_benchmark_cells.csv"), 32_000_000, "injection grid/detector benchmark: size, power, AUC per cell and statistic")
    save_results(dec, out_name("detector_benchmark_decisions.csv"), 32_000_000, "injection grid/detector benchmark: pre-registered decisions (kill switch, injection harness, classical-test benchmark)")


if __name__ == "__main__":
    main()
