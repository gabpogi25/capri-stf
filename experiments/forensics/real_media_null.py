"""
experiments/forensics/real_media_null.py

the real-media null gate: null calibration on REAL authentic media.

  1. Fit empirical nulls on the authentic items of split train_null
     (image statistic per image or frame; Delta_J statistic per window).
  2. On the authentic items of split val, compute p-values against those nulls
     and the rejection rate at the nominal level (default 0.05).
  3. PASS if the rate lies in the pre-registered band (default [3.5%, 6.5%]).
     Also reported: KS distance of the held-out p-values to U(0,1) and the
     chi^2 rejection rate (diagnostic; it is expected to be miscalibrated).
  4. Writes the nulls (forensic_nulls_<media>.json) used by the held-out detection gates.
     (kappa_max is written only for the legacy Delta_J statistic.)

Usage:
  python experiments/forensics/real_media_null.py --manifest data/manifests/images.csv --media image
"""

import argparse

import numpy as np
from scipy import stats

from common import RESULTS_DIR, require_manifest, save_json
from forensics.config import load_config
from forensics.io.manifests import load_manifest
from forensics.pipeline_video import ForensicNulls
from forensics.stats.empirical_null import EmpiricalNull, rejection_rate, uniformity_ks
from media_statistics import cached_table


MIN_NULL_VALUES = 20   # EmpiricalNull.fit refuses fewer authentic values


def flatten(table, column):
    import json
    return np.asarray([v for cell in table[column] for v in json.loads(cell)], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--media", choices=["image", "video"], required=True)
    parser.add_argument("--seed", type=int, default=42301)
    parser.add_argument("--tag", default="", help="suffix for every output name (another study, e.g. _ffpp)")
    args = parser.parse_args()
    manifest_path = require_manifest(args.manifest, "the real-media null gate")
    config = load_config()
    gates = config["gates"]
    manifest = load_manifest(manifest_path)
    table = cached_table(manifest_path, manifest, config, args.media)
    table = table[table["label"] == "real"]
    train, val = table[table["split"] == "train_null"], table[table["split"] == "val"]
    if len(train) == 0 or len(val) == 0:
        raise SystemExit("the real-media null gate needs authentic items in both train_null and val splits.")

    result = {"media": args.media, "nominal_alpha": gates["real_media_null_nominal"], "band": gates["real_media_null_band"],
              "n_train_items": int(len(train)), "n_val_items": int(len(val))}
    nulls = ForensicNulls()
    if args.media == "image":
        nulls.frame_null = EmpiricalNull.fit(train["statistic"].dropna())
        val_stat = val["statistic"].dropna().to_numpy()
        p = nulls.frame_null.p_value(val_stat)
        result.update({"coverage_val": float(val["statistic"].notna().mean()), "n_val_tested": int(len(val_stat))})
        dof = val.loc[val["statistic"].notna(), "n_constraints"].to_numpy()
        p_chi2 = stats.chi2.sf(val_stat * dof, dof)
    else:
        nulls.frame_null = EmpiricalNull.fit(flatten(train, "frame_statistics"))
        train_windows = flatten(train, "window_statistics")
        # too few authentic windows (e.g. talking-head video without camera motion): the pencil
        # branch is empty, no window null is fitted, and gate A uses the frame statistics
        nulls.window_null = EmpiricalNull.fit(train_windows) if train_windows.size >= MIN_NULL_VALUES else None
        kappas = flatten(train, "kappa_values")       # empty with the pencil statistic (no kappa gate)
        nulls.kappa_max = float(np.quantile(kappas, config["video"]["kappa_quantile"])) if kappas.size else None
        if nulls.window_null is not None:
            p, statistic = nulls.window_null.p_value(flatten(val, "window_statistics")), "window"
        else:
            p, statistic = nulls.frame_null.p_value(flatten(val, "frame_statistics")), "frame"
        p_chi2 = np.array([])
        result.update({"gate_a_statistic": statistic, "n_train_windows": int(train_windows.size),
                       "n_val_windows": int(flatten(val, "window_statistics").size),
                       "n_val_tested": int(len(p)), "kappa_max": nulls.kappa_max})
    rate = rejection_rate(p, gates["real_media_null_nominal"])
    ks, ks_p = uniformity_ks(p)
    lo, hi = gates["real_media_null_band"]
    result.update({"rejection_rate_empirical_null": rate, "ks_to_uniform": ks, "ks_p": ks_p,
                   "rejection_rate_chi2_diagnostic": float(np.mean(p_chi2 <= gates["real_media_null_nominal"])) if p_chi2.size else None,
                   "passed": bool(lo <= rate <= hi)})
    used = nulls.window_null if (args.media == "video" and nulls.window_null is not None) else nulls.frame_null
    n_null = len(used.reference)
    result["n_null_values"] = int(n_null)
    result["min_attainable_p"] = 1.0 / (n_null + 1)
    if n_null < 20 / gates["real_media_null_nominal"]:
        print(f"WARNING: null has {n_null} values; p-values below {1 / (n_null + 1):.3g} are unattainable and "
              f"the rejection rate at alpha={gates['real_media_null_nominal']} is coarse. Use >= {int(20 / gates['real_media_null_nominal'])}.")
    nulls.save(RESULTS_DIR / f"forensic_nulls_{args.media}{args.tag}.json")
    save_json(result, f"real_media_null_{args.media}{args.tag}.json", args.seed, config, [manifest_path])
    print(result)
    if not result["passed"]:
        raise SystemExit(f"GATE A FAILED: rejection rate {rate:.3f} outside [{lo}, {hi}]")


if __name__ == "__main__":
    main()
