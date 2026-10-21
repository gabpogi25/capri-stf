"""
experiments/eval/real_pencil_analysis.py

Decision rules of docs/prereg/real_pencil.md, implemented verbatim.

Input: real_pencil_windows_batch{1,2}.csv (experiments/run_real_pencil.py). Windows of
calibration scenes (Role cal) fix every threshold; windows of test scenes (Role test) give
sizes, powers and AUCs. Rows with Injection "none" are the null.

Statistics (larger = more inconsistent):
  pencil_q10, acat, sampson_mean   threshold: 95th percentile of the cal null windows
  combination                      Cauchy combination of the conformal p-values of pencil_q10
                                   and sampson_mean against the cal null; rejects at p <= 0.05
  split_glrt                       split sum of squares / sigma_hat^2, sigma_hat^2 = median of
                                   sigma_hat_px^2 over the cal null windows; rejects above the
                                   chi^2_{3n-15} 95% quantile (the classical plug-in test)
Intention to treat: an abstained window or a NaN value gets the median cal null value of that
statistic (never a value from the label); coverage is reported.

Decisions (on the full set; --interim B gives the same numbers with no decision):
  S   size of each statistic on the test null windows in [3.5%, 6.5%] ("size valid").
      Rule R3: power and AUC are reported only for statistics with a valid size.
  C   combination: supported if its size is valid and, on every injection family, the
      scene-cluster bootstrap 95% CI of power(combination) - max(power of pencil_q10,
      power of sampson_mean) has lower end > -0.02, and on at least one family > 0.
  T   transfer (descriptive): thresholds from the cal scenes of one family, size on the test
      scenes of the same family and of the other families.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "experiments")]

from experiment_utils import RESULTS_DIR, save_results  # noqa: E402
from forensics.stats.combine import cauchy_combination  # noqa: E402

N_POINTS = 30
ALPHA, BAND = 0.05, (0.035, 0.065)
SINGLE = ("pencil_q10", "acat", "sampson_mean")
STATS = SINGLE + ("combination", "split_glrt")
BOOTSTRAP_SEED, N_BOOTSTRAP = 87_500_000, 2_000
MARGIN = 0.02


def itt(df: pd.DataFrame, stat: str, fill: float) -> np.ndarray:
    values = df[stat].to_numpy(float)
    return np.where(df["abstain"].astype(bool).to_numpy() | ~np.isfinite(values), fill, values)


def conformal_p(cal: np.ndarray, x: np.ndarray) -> np.ndarray:
    """p = (1 + #{cal >= x}) / (n_cal + 1)."""
    cal = np.sort(cal)
    return (1 + len(cal) - np.searchsorted(cal, x, side="left")) / (len(cal) + 1)


def auc(neg: np.ndarray, pos: np.ndarray) -> float:
    r = stats.rankdata(np.r_[neg, pos])
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Calibration:
    """Thresholds and fills from the cal null windows; `reject` and `score` for any rows."""

    def __init__(self, cal_null: pd.DataFrame):
        assert len(cal_null) >= 20, "too few calibration windows"
        ok = ~cal_null["abstain"].astype(bool)
        self.fill = {s: float(np.nanmedian(cal_null.loc[ok, s])) for s in SINGLE}
        self.cal = {s: itt(cal_null, s, self.fill[s]) for s in SINGLE}
        self.threshold = {s: float(np.quantile(self.cal[s], 1 - ALPHA)) for s in SINGLE}
        self.sigma2 = float(np.nanmedian(cal_null.loc[ok, "sigma_hat_px"] ** 2))
        self.glrt_fill = float(np.nanmedian(cal_null.loc[ok, "split_glrt"])) / self.sigma2
        self.glrt_threshold = float(stats.chi2.ppf(1 - ALPHA, 3 * N_POINTS - 15))

    def score(self, df: pd.DataFrame, stat: str) -> np.ndarray:
        """Larger = more inconsistent; the combination is scored as -log10 p."""
        if stat in SINGLE:
            return itt(df, stat, self.fill[stat])
        if stat == "split_glrt":
            return np.where(df["abstain"].astype(bool) | ~np.isfinite(df["split_glrt"]),
                            self.glrt_fill, df["split_glrt"].to_numpy(float) / self.sigma2)
        pp = conformal_p(self.cal["pencil_q10"], itt(df, "pencil_q10", self.fill["pencil_q10"]))
        ps = conformal_p(self.cal["sampson_mean"], itt(df, "sampson_mean", self.fill["sampson_mean"]))
        return -np.log10([cauchy_combination(np.array([a, b])) for a, b in zip(pp, ps)])

    def reject(self, df: pd.DataFrame, stat: str) -> np.ndarray:
        score = self.score(df, stat)
        if stat in SINGLE:
            return score > self.threshold[stat]
        if stat == "split_glrt":
            return score > self.glrt_threshold
        return score >= -np.log10(ALPHA)


def cluster_bootstrap(df: pd.DataFrame, value, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% percentile CI of value(rows) resampling test scenes with replacement."""
    scenes = df["Unit"].unique()
    groups = {u: df[df["Unit"] == u] for u in scenes}
    rng = np.random.default_rng(seed)
    draws = [value(pd.concat([groups[u] for u in rng.choice(scenes, len(scenes))])) for _ in range(N_BOOTSTRAP)]
    return tuple(float(x) for x in np.nanpercentile(draws, [2.5, 97.5]))


def analyse(df: pd.DataFrame, interim: int | None) -> tuple[list[dict], list[dict]]:
    df = df.assign(Unit=df["Dataset"] + "/" + df["Scene"])
    null, test = df[df["Injection"] == "none"], df[df["Role"] == "test"]
    calib = Calibration(null[null["Role"] == "cal"])
    test_null = test[test["Injection"] == "none"]
    families = [f for f in df["Injection"].unique() if f != "none"]
    rows, decisions = [], []
    size = {s: float(calib.reject(test_null, s).mean()) for s in STATS}
    for s in STATS:
        valid = BAND[0] <= size[s] <= BAND[1]
        rows.append({"Statistic": s, "Injection": "none", "Size": size[s], "Size_Valid": valid,
                     "N_Test_Windows": len(test_null), "N_Test_Scenes": test_null["Unit"].nunique(),
                     "Coverage": float(1 - test_null["abstain"].astype(bool).mean())})
        for fam in families:
            pos = test[test["Injection"] == fam]
            power = float(calib.reject(pos, s).mean()) if valid else float("nan")
            area = auc(calib.score(test_null, s), calib.score(pos, s)) if valid else float("nan")
            rows.append({"Statistic": s, "Injection": fam, "Power": power, "AUC": area, "Size_Valid": valid,
                         "N_Test_Windows": len(pos), "N_Test_Scenes": pos["Unit"].nunique()})
    for s in STATS:
        decisions.append({"Item": f"S size of {s} on test scenes in [3.5%, 6.5%]", "Value": size[s],
                          "Decision": _label(interim, BAND[0] <= size[s] <= BAND[1], "size valid", "size fails")})

    combo_ok, lows = BAND[0] <= size["combination"] <= BAND[1], []
    for fam in families:
        pos = test[test["Injection"] == fam]

        def gain(rows_: pd.DataFrame) -> float:
            best = max(calib.reject(rows_, "pencil_q10").mean(), calib.reject(rows_, "sampson_mean").mean())
            return float(calib.reject(rows_, "combination").mean() - best)

        low, high = cluster_bootstrap(pos, gain)
        lows.append(low)
        decisions.append({"Item": f"C gain of the combination over the better single statistic, {fam}",
                          "Value": gain(pos), "CI_Low": low, "CI_High": high, "Decision": ""})
    supported = combo_ok and all(l > -MARGIN for l in lows) and any(l > 0 for l in lows)
    decisions.append({"Item": "C combination keeps its size and is at least as powerful on every family",
                      "Value": size["combination"], "Decision": _label(interim, supported, "supported", "not supported")})

    for source in sorted(null["Family"].unique()):
        cal_f = null[(null["Role"] == "cal") & (null["Family"] == source)]
        if len(cal_f) < 20:
            continue
        calib_f = Calibration(cal_f)
        for target in sorted(test_null["Family"].unique()):
            target_rows = test_null[test_null["Family"] == target]
            for s in ("pencil_q10", "acat", "sampson_mean"):
                rows.append({"Statistic": s, "Injection": "none", "Transfer_From": source, "Transfer_To": target,
                             "Size": float(calib_f.reject(target_rows, s).mean()),
                             "N_Test_Windows": len(target_rows)})
    return rows, decisions


def _label(interim: int | None, ok: bool, yes: str, no: str) -> str:
    if interim is not None:
        return f"interim after batch {interim}: no decision"
    return yes if ok else no


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+",
                        default=[RESULTS_DIR / "real_pencil_windows_batch1.csv",
                                 RESULTS_DIR / "real_pencil_windows_batch2.csv"])
    parser.add_argument("--interim", type=int, default=None, metavar="BATCH")
    args = parser.parse_args()
    df = pd.concat([pd.read_csv(p, comment="#") for p in args.input], ignore_index=True)
    assert not df.duplicated(["Dataset", "Scene", "Window", "Injection"]).any()
    rows, decisions = analyse(df, args.interim)
    prefix = "real_pencil" + (f"_interim_batch{args.interim}" if args.interim is not None else "")
    save_results(pd.DataFrame(rows), f"{prefix}_summary.csv", BOOTSTRAP_SEED, "Real-data pencil test: sizes, powers, AUCs")
    save_results(pd.DataFrame(decisions), f"{prefix}_decisions.csv", BOOTSTRAP_SEED, "Real-data pencil test: decisions")


if __name__ == "__main__":
    main()
