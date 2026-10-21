"""
experiments/forensics/compression_sweep.py

degradation sweep: robustness of the single-image branch to post-processing.
Authentic and generated TEST images are re-encoded (JPEG quality 95/85/75/60)
and down-scaled (0.75, 0.5) AFTER generation; for each condition we report

  * FPR: rejection rate of authentic images at the real-media null gate level, and
    Delta-FPR = FPR(processed) - FPR(original) (processing must not create
    false alarms);
  * flip rate: fraction of items whose decision changes relative to the
    original file;
  * AUC (authentic vs generated) and coverage.

The empirical null is the one written by real_media_null.py (forensic_nulls_image.json);
it is NOT refitted per condition, because in deployment the processing history
is unknown. Exits without output if the manifest or the null is missing.

Usage:
  python experiments/forensics/compression_sweep.py --manifest data/manifests/images.csv
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from common import RESULTS_DIR, require_manifest, save_csv
from forensics.config import load_config
from forensics.eval.metrics import auc_delong, delta_fpr
from forensics.io.manifests import load_manifest
from forensics.pipeline_image import analyze_gray, image_statistic
from forensics.pipeline_video import ForensicNulls

CONDITIONS = [("original", None, 1.0), ("jpeg95", 95, 1.0), ("jpeg85", 85, 1.0), ("jpeg75", 75, 1.0),
              ("jpeg60", 60, 1.0), ("resize0.75", None, 0.75), ("resize0.5", None, 0.5),
              ("resize0.75_jpeg75", 75, 0.75)]


def process(gray: np.ndarray, quality: int | None, scale: float) -> np.ndarray:
    import cv2

    out = gray
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if quality is not None:
        ok, buffer = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
        out = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    return out


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42004)
    args = parser.parse_args()
    manifest_path = require_manifest(args.manifest, "degradation sweep")
    null_path = RESULTS_DIR / "forensic_nulls_image.json"
    if not null_path.exists():
        raise SystemExit(f"degradation sweep needs the real-media null gate null {null_path}; run real_media_null.py --media image first.")
    config = load_config()
    alpha = config["gates"]["real_media_null_nominal"]
    null = ForensicNulls.load(null_path).frame_null
    manifest = load_manifest(manifest_path)
    items = manifest[manifest["split"] == args.split]  # keep manifest index: same seeds as the gates
    if items.empty:
        raise SystemExit(f"No items in split {args.split!r}.")

    rows = []
    for k, item in items.iterrows():
        gray = cv2.imread(item["abs_path"], cv2.IMREAD_GRAYSCALE)
        for name, quality, scale in CONDITIONS:
            evidence = analyze_gray(process(gray, quality, scale), config, np.random.default_rng(int(config["global"]["base_seed"]) + k))
            abstain = evidence.decision == "insufficient"
            stat = np.nan if abstain else image_statistic(np.array([c.z for c in evidence.constraints]))
            p = np.nan if abstain else float(null.p_value(stat)[0])
            rows.append({"path": item["path"], "source": item["source"], "generator": item["generator"],
                         "y": int(item["y"]), "condition": name, "abstain": abstain, "statistic": stat,
                         "p_value": p, "reject": bool(p <= alpha) if not abstain else np.nan})
    per_item = pd.DataFrame(rows)
    save_csv(per_item, "compression_sweep_per_item.csv", args.seed, "degradation sweep per-item statistics under post-processing",
             config, [manifest_path, null_path])

    original = per_item[per_item.condition == "original"].set_index("path")
    summary = []
    for name, _, _ in CONDITIONS:
        cond = per_item[per_item.condition == name].set_index("path")
        both = cond.index[cond["abstain"].eq(False) & original.loc[cond.index, "abstain"].eq(False)]
        authentic = cond[(cond.y == 0) & ~cond.abstain]
        tested = cond[~cond.abstain]
        threshold = -np.log10(alpha)
        orig_auth = original[(original.y == 0) & ~original.abstain]
        score = lambda frame: -np.log10(frame["p_value"].clip(lower=1e-300)).to_numpy()  # noqa: E731
        row = {"condition": name, "n_items": len(cond), "coverage": float((~cond.abstain).mean()),
               "n_authentic_tested": len(authentic), "n_generated_tested": int((tested.y == 1).sum()),
               "fpr": float(np.mean(authentic["p_value"] <= alpha)) if len(authentic) else np.nan,
               "delta_fpr": delta_fpr(threshold, score(orig_auth), score(authentic))
               if len(authentic) and len(orig_auth) else np.nan,
               "flip_rate": float(np.mean(original.loc[both, "reject"].astype(bool).to_numpy()
                                          != cond.loc[both, "reject"].astype(bool).to_numpy())) if len(both) else np.nan}
        if tested.y.nunique() == 2:
            result = auc_delong(tested["y"].to_numpy(), score(tested))
            row.update({"auc": result["auc"], "auc_ci_low": result["ci_lo"], "auc_ci_high": result["ci_hi"]})
        summary.append(row)
    summary = pd.DataFrame(summary)
    print(summary.to_string(index=False))
    save_csv(summary, "compression_sweep.csv", args.seed, "degradation sweep robustness to JPEG and resizing", config,
             [manifest_path, null_path])


if __name__ == "__main__":
    main()
