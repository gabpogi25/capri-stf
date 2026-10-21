"""
experiments/forensics/heldout_detection.py

the held-out image gate (images) and the held-out video gate (videos): detection on HELD-OUT generators.

Per generator in the test split (against all authentic test items):
  * AUC with DeLong CI and cluster-bootstrap CI by source;
  * TPR at 5% FPR, with the threshold fixed on authentic VALIDATION items only
    (reject when score >= threshold: empirical-null p-values are discrete, with
    smallest value 1 / (n_null + 1), so ties at the threshold are common);
  * coverage (fraction not abstained) for each class; abstained items are
    excluded from AUC and reported, never imputed;
  * class balance and N in every row.
Plus the metadata-only shortcut audit (the held-out detection gates requirement: AUC <= 0.55).

Scores:
  image: empirical-null p-value of the per-image statistic, score = -log10 p
  video: block-Fisher combination of window p-values (Delta_J) and frame
         p-values against the real-media null gate nulls, score = -log10 p

PASS: at least one held-out generator with AUC >= the pre-registered minimum,
its CI excluding 0.5, and metadata AUC <= the pre-registered maximum.

Usage:
  python experiments/forensics/heldout_detection.py --manifest data/manifests/images.csv --media image
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import RESULTS_DIR, require_manifest, save_csv, save_json
from forensics.config import load_config
from forensics.eval.audit_metadata import metadata_audit
from forensics.eval.metrics import auc_delong, cluster_bootstrap, holm_correction
from forensics.io.manifests import load_manifest
from forensics.pipeline_video import ForensicNulls
from forensics.stats.fisher import block_fisher
from media_statistics import cached_table


def scores_for(table: pd.DataFrame, nulls: ForensicNulls, media: str) -> np.ndarray:
    if media == "image":
        stat = table["statistic"].to_numpy(dtype=float)
        p = np.full(len(stat), np.nan)
        ok = np.isfinite(stat)
        p[ok] = nulls.frame_null.p_value(stat[ok])
        return -np.log10(p)
    out = []
    for _, row in table.iterrows():
        p_values = [float(x) for x in nulls.window_null.p_value(np.asarray(json.loads(row["window_statistics"]), float))] \
            if nulls.window_null is not None and json.loads(row["window_statistics"]) else []
        frames = json.loads(row["frame_statistics"])
        p_values += [float(x) for x in nulls.frame_null.p_value(np.asarray(frames, float))] if frames else []
        out.append(-np.log10(block_fisher(np.asarray(p_values))[1]) if p_values else np.nan)
    return np.asarray(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--media", choices=["image", "video"], required=True)
    parser.add_argument("--seed", type=int, default=42302)
    parser.add_argument("--tag", default="", help="suffix for every input and output name (another study, e.g. _ffpp)")
    args = parser.parse_args()
    manifest_path = require_manifest(args.manifest, "the held-out detection gates")
    nulls_path = RESULTS_DIR / f"forensic_nulls_{args.media}{args.tag}.json"
    if not nulls_path.exists():
        raise SystemExit(f"Run the real-media null gate first ({nulls_path} missing).")
    config, gates = load_config(), load_config()["gates"]
    nulls = ForensicNulls.load(nulls_path)
    manifest = load_manifest(manifest_path)
    table = cached_table(manifest_path, manifest, config, args.media)
    table["score"] = scores_for(table, nulls, args.media)
    val_authentic = table[(table["split"] == "val") & (table["label"] == "real")]["score"].dropna()
    threshold = float(np.quantile(val_authentic, 0.95)) if len(val_authentic) else np.nan
    test = table[table["split"] == "test"]
    authentic = test[test["label"] == "real"]
    rows = []
    for generator, fakes in test[test["label"] == "generated"].groupby("generator"):
        both = pd.concat([authentic, fakes])
        answered = both[both["score"].notna()]
        y, s = answered["y"].to_numpy(), answered["score"].to_numpy()
        delong = auc_delong(y, s)
        boot = cluster_bootstrap(lambda yy, ss: roc_auc_score(yy, ss), y, s, answered["source"].to_numpy(),
                                 seed=args.seed)
        rows.append({"Generator": generator, "N_Authentic": len(authentic), "N_Generated": len(fakes),
                     "Coverage_Authentic": float(authentic["score"].notna().mean()),
                     "Coverage_Generated": float(fakes["score"].notna().mean()),
                     "Positive_Fraction_Answered": float(y.mean()), "AUC": delong["auc"],
                     "AUC_DeLong_Lo": delong["ci_lo"], "AUC_DeLong_Hi": delong["ci_hi"],
                     "AUC_Cluster_Lo": boot["ci_lo"], "AUC_Cluster_Hi": boot["ci_hi"],
                     "DeLong_p_vs_0.5": float(2 * min(_normal_sf((delong["auc"] - 0.5) / max(delong["se"], 1e-12)), 0.5)),
                     "Threshold_From_Val_Authentic": threshold,
                     "TPR_at_5pct_FPR": float(np.mean(fakes["score"].dropna() >= threshold)),
                     "FPR_Test_Authentic": float(np.mean(authentic["score"].dropna() >= threshold))})
    per_generator = pd.DataFrame(rows)
    if len(per_generator):
        per_generator["p_Holm"] = holm_correction(per_generator["DeLong_p_vs_0.5"].tolist())
    audit = metadata_audit(manifest[manifest["split"] == "test"].reset_index(drop=True))
    min_auc = gates["heldout_image_min_auc"] if args.media == "image" else gates["heldout_video_min_auc"]
    qualifying = per_generator[(per_generator["AUC"] >= min_auc) & (per_generator["AUC_Cluster_Lo"] > 0.5)] \
        if len(per_generator) else per_generator
    passed = bool(len(qualifying) > 0 and audit["metadata_auc"] <= gates["heldout_max_metadata_auc"])
    save_csv(per_generator, f"heldout_detection_{args.media}{args.tag}_per_generator.csv",
             args.seed, f"held-out detection ({args.media}): per-generator detection", config, [manifest_path])
    save_json({"media": args.media, "passed": passed, "metadata_audit": audit, "min_auc": min_auc,
               "qualifying_generators": qualifying["Generator"].tolist() if len(qualifying) else []},
              f"heldout_detection_{args.media}{args.tag}.json", args.seed, config, [manifest_path])
    print(per_generator.to_string(index=False))
    print({"passed": passed, **audit})
    if not passed:
        raise SystemExit("GATE FAILED (see the JSON for the reason). Per the plan, stop the forensic direction "
                         "here and report the negative result.")


def _normal_sf(x: float) -> float:
    from scipy import stats
    return float(stats.norm.sf(x))


if __name__ == "__main__":
    main()
