"""
experiments/eval/score_audit.py

score audit (docs/prereg/evaluation_plan.md): validity audit
of the result files. Checks per (result file, score):
  1. static label path: an AST scan of every experiment/algorithm source for a
     scoring function that reads a label name;
  2. floor fraction F_floor = P(|s| <= 1e3 eps_mach) per class (non-negative scores
     on unit-normalized quantities only; otherwise not applicable);
  3. distinct-value ratio U = #unique / #items;
  4. permutation test of |AUC - 0.5| with B = 1000 label permutations within the
     file, Phipson-Smyth p = (1 + #{perm >= obs}) / (B + 1);
  5. abstention leak: |AUC(non-finite score, label) - 0.5|.
Verdict: RETAINED iff no label path, F_floor <= 0.5 in at least one class, U >= 0.05,
p < 0.01 and leak <= 0.10; RETRACTED (the score is not valid evidence) if check 1
or 2 fails; else INCONCLUSIVE.
Check 6 (construction swap) is not run here (noise-model check is the swap for the drift test).
Self-test (must fire, else the audit is broken): rho from ONE point set is flagged
by check 2; a planted scorer that reads the label is flagged by check 1; a planted
abstention equal to the label is flagged by check 5.
Output: score_audit.csv, score_audit_selftest.csv. Seed get_experiment_seed(21).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from experiment_utils import get_experiment_seed, save_results  # noqa: E402

SEED = get_experiment_seed(21)
N_PERM = 1000
EPS_FLOOR = 1e3 * np.finfo(float).eps
LABEL_NAMES = {"label", "labels", "y", "y_true", "degenerate", "label_bad", "tuple_bad", "correct",
               "contains_corrupted", "label_excessive", "is_fake", "is_generated"}
RES = ROOT / "experiments" / "results"
# Manual review of static hits (2026-10-12): evaluation or calibration code that must read labels.
# A hit not listed here counts as a label path.
REVIEWED_NOT_SCORERS = {
    ("experiments/run_degeneracy_matched.py", "score_summary"): "evaluation: AUC/TPR from stored scores",
    ("experiments/run_trust_calibrated.py", "fit_score_maps"): "isotonic calibration fitted on the calibration split",
    ("experiments/run_trust_calibrated.py", "score_statistics"): "evaluation: ECE/Brier/AUC of stored predictions",
    ("experiments/run_trust_calibrated.py", "<lambda>"): "lambdas inside score_statistics (evaluation)",
}
# (file, label column, cluster column, row filter, [(score, non-negative unit-scale?)])
TARGETS = [
    ("pencil_drift.csv", "Label_Excessive", "Scene", None,
     [("pencil_q10", True), ("registration_rms", True), ("median_tuple_dJ", True), ("single_tuple_dJ", True)]),
    ("degeneracy_matched_samples.csv", "Degenerate", "Trial", ("Split", "test"),
     [("m8_capri_coplanarity", False), ("m8_capri_kappa", True), ("m8_capri_sigma", True),
      ("m8_degensac_h_inliers", False), ("m8_homography_ls6", False)]),
    ("trust_calibrated_items.csv", "Y_tol0.1", "scene_seed", ("split", "test"), [("R0", True), ("K", True)]),
    ("logistic_fusion_dataset.csv", "label_bad", "item_seed", ("split", "test"),
     [("pencil_q10", True), ("rho_indep", True), ("kappa_J", True)]),
    ("true_synthetic_baselines.csv", "Degenerate", "Trial", None,
     [("capri_ls", True), ("capri_kappa", True), ("capri_sigma", True), ("degensac_h_inliers", False)]),
    ("syzygy_residual.csv", "Contains_Corrupted", "Trial", None, [("rho", True), ("rho_full", True)]),
    ("ablation.csv", "Tuple_Bad", "Trial", None, [("Rho", True), ("LS", True), ("Kappa", True), ("Pencil_Q10", True)]),
    ("signal_combination.csv", "Correct", "Sample", None, [("LS", True), ("Rho", True)]),
]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def label_reading_scorers(source: str) -> list[str]:
    """Names of functions whose name contains 'score' and that read a label name."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.Lambda)) and "score" in getattr(node, "name", "score").lower():
            for inner in ast.walk(node):
                name = (inner.id if isinstance(inner, ast.Name) else inner.attr if isinstance(inner, ast.Attribute)
                        else inner.value if isinstance(inner, ast.Constant) and isinstance(inner.value, str) else None)
                if isinstance(name, str) and name.lower() in LABEL_NAMES and not (isinstance(inner, ast.Name)
                                                                                 and isinstance(inner.ctx, ast.Store)):
                    hits.append(getattr(node, "name", "<lambda>"))
                    break
    return hits


def static_scan() -> dict[str, list[str]]:
    files = list((ROOT / "experiments").glob("run_*.py")) + list((ROOT / "algorithms").rglob("*.py")) + \
        list((ROOT / "forensics").rglob("*.py")) + list((ROOT / "experiments" / "forensics").glob("*.py"))
    return {str(f.relative_to(ROOT)): hits for f in files if (hits := label_reading_scorers(f.read_text()))}


def checks(y: np.ndarray, s: np.ndarray, unit_scale: bool, rng: np.random.Generator) -> dict:
    finite = np.isfinite(s)
    y_f, s_f = y[finite], s[finite]
    floor = ({f"F_floor_class{c}": float(np.mean(np.abs(s_f[y_f == c]) <= EPS_FLOOR)) for c in (0, 1)}
             if unit_scale else {"F_floor_class0": np.nan, "F_floor_class1": np.nan})
    observed = abs(auc(y_f, s_f) - 0.5)
    perms = np.array([abs(auc(rng.permutation(y_f), s_f) - 0.5) for _ in range(N_PERM)])
    leak = abs(auc(y, (~finite).astype(float)) - 0.5) if (~finite).any() else 0.0
    return {"N": int(len(y)), "N_finite": int(finite.sum()), "AUC": auc(y_f, s_f), **floor,
            "U": float(len(np.unique(s_f)) / max(len(s_f), 1)),
            "Perm_p": float((1 + np.sum(perms >= observed)) / (N_PERM + 1)), "Abstain_leak": float(leak)}


def verdict(row: dict, label_path: bool) -> str:
    floor_fail = unit = not np.isnan(row["F_floor_class0"])
    if unit:
        floor_fail = min(row["F_floor_class0"], row["F_floor_class1"]) > 0.5
    if label_path or floor_fail:
        return "RETRACTED" if label_path else "DEGENERATE (RETRACTED)"
    ok = row["U"] >= 0.05 and row["Perm_p"] < 0.01 and row["Abstain_leak"] <= 0.10
    return "RETAINED" if ok else "INCONCLUSIVE"


def self_test(rng: np.random.Generator) -> pd.DataFrame:
    from algorithms.invariants.brackets import compute_all_brackets_for_tuple
    from algorithms.syzygy.residual import compute_syzygy_residual
    rows = []
    rho, y = [], []
    for k in range(400):                    # rho of brackets of ONE point set (Cor. 2.6) vs a shuffled label
        P = np.vstack([rng.uniform(-1, 1, (3, 6)), np.ones((1, 6))])
        rho.append(compute_syzygy_residual(compute_all_brackets_for_tuple(P / np.linalg.norm(P, axis=0), (0, 1, 2, 3, 4, 5))))
        y.append(k % 2)
    r = checks(np.array(y), np.array(rho, dtype=float), True, rng)
    rows.append({"Check": "rho_one_set_flagged_by_floor", **r, "Fired": verdict(r, False).startswith("DEGENERATE")})
    leaking = "def score_sample(row):\n    return row['label'] + 0.1\n"
    rows.append({"Check": "planted_label_reader_flagged", "Fired": bool(label_reading_scorers(leaking))})
    yy = rng.integers(0, 2, 500)
    ss = np.where(yy == 1, np.nan, rng.uniform(size=500))
    r = checks(yy, ss, False, rng)
    rows.append({"Check": "planted_abstention_leak_flagged", **r, "Fired": r["Abstain_leak"] > 0.10})
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    selftest = self_test(rng)
    print(selftest[["Check", "Fired"]].to_string(index=False))
    assert selftest["Fired"].all(), "audit self-test did not fire: the audit is broken"
    scan = static_scan()
    unreviewed = {f: [h for h in hits if (f, h) not in REVIEWED_NOT_SCORERS] for f, hits in scan.items()}
    unreviewed = {f: h for f, h in unreviewed.items() if h}
    print("static hits:", scan or "none", "| unreviewed (label paths):", unreviewed or "none")
    rows = []
    for name, label, cluster, row_filter, scores in TARGETS:
        df = pd.read_csv(RES / name, comment="#")
        if row_filter:
            df = df[df[row_filter[0]] == row_filter[1]]
        y = df[label].astype(float).to_numpy()
        y = (y > 0).astype(int)
        for score, unit_scale in scores:
            r = checks(y, pd.to_numeric(df[score], errors="coerce").to_numpy(dtype=float), unit_scale, rng)
            rows.append({"File": name, "Label": label, "Score": score, **r,
                         "Static_label_path": bool(unreviewed), "Verdict": verdict(r, bool(unreviewed))})
    out = pd.DataFrame(rows)
    print(out[["File", "Score", "AUC", "F_floor_class0", "F_floor_class1", "U", "Perm_p", "Abstain_leak", "Verdict"]]
          .to_string(index=False))
    save_results(selftest, "score_audit_selftest.csv", SEED, "score audit: audit self-test (must fire)")
    save_results(out, "score_audit.csv", SEED, f"score audit: audit checks 1-5; static hits {scan or 'none'}; unreviewed {unreviewed or 'none'}")


if __name__ == "__main__":
    main()
