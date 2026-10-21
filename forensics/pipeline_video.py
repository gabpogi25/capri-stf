"""
forensics/pipeline_video.py

Video branch.

    decode (native resolution) -> shots -> windows of V = 6 sampled views
    -> KLT tracks + rigid segmentation -> parallax gate
    -> projective reconstructions from subsets A = {1,3,5}, B = {2,4,6}
    -> pencil drift statistic Q_0.1 of |d log|CR|| between the two
       reconstructions (docs/MATHEMATICS.md, Section 4)
    + per-frame single-image branch on sampled frames
    -> block-Fisher clip statistic, CUSUM onset, temporal TV.

Statistics become p-values only through EMPIRICAL NULLS fitted on authentic
clips (ForensicNulls, written by experiments/forensics/real_media_null.py). Without
them the clip decision is reported as uncalibrated (reasons list) and must not
be used as a result.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from forensics.io.records import ClipEvidence
from forensics.pipeline_image import analyze_gray, image_statistic
from forensics.stats.cusum import cusum, temporal_total_variation
from forensics.stats.empirical_null import EmpiricalNull
from forensics.stats.fisher import block_fisher
from forensics.video.decode import decode, split_shots
from forensics.video.multiview import projective_reconstruction
from algorithms.consistency.pencil import pencil_discrepancies
from forensics.video.tracking import parallax_ok, rigid_inliers, track_window


@dataclass
class ForensicNulls:
    frame_null: EmpiricalNull | None = None
    window_null: EmpiricalNull | None = None
    kappa_max: float | None = None

    def save(self, path: Path | str) -> None:
        payload = {"frame_null": None if self.frame_null is None else self.frame_null.reference.tolist(),
                   "window_null": None if self.window_null is None else self.window_null.reference.tolist(),
                   "kappa_max": self.kappa_max}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "ForensicNulls":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        to_null = lambda v: None if v is None else EmpiricalNull(np.asarray(v, dtype=float))  # noqa: E731
        return cls(to_null(payload["frame_null"]), to_null(payload["window_null"]), payload.get("kappa_max"))


def window_statistic(frames: list[np.ndarray], config: dict, seed: int, kappa_max: float | None = None) -> dict:
    """
    Pencil drift statistic for one window of V views: the two subset
    reconstructions are two measurements of the same tracks in unrelated
    projective frames, so the statistic is Q_p of the pencil cross-ratio
    discrepancies (p = video.pencil_quantile). The low quantile already
    discounts ill-conditioned pencils (Prop. 4.3), so no sigma_4 / kappa gate is
    applied; kappa_max is accepted for backward compatibility and ignored.
    Returns {"used": bool, "reason"/"statistic", ...}.
    """
    cfg = config["video"]
    height, width = frames[0].shape[:2]
    tracks, _ = track_window(frames, cfg["tracker_grid"], cfg["track_conf_min"])
    if len(tracks) < cfg["min_tracks_per_window"]:
        return {"used": False, "reason": f"too few tracks ({len(tracks)})"}
    rigid = rigid_inliers(tracks, cfg["ransac_threshold_px"], cfg["ransac_confidence"],
                          cfg["rigid_inlier_frac"], seed)
    outlier_fraction = 1.0 - rigid.mean()
    tracks = tracks[rigid]
    if len(tracks) < cfg["min_tracks_per_window"]:
        return {"used": False, "reason": f"too few rigid tracks ({len(tracks)})", "outlier_fraction": outlier_fraction}
    ok, parallax = parallax_ok(tracks, width, height, cfg["parallax_min_disp_frac"],
                               cfg["parallax_min_f_over_h"], cfg["ransac_threshold_px"], seed)
    if not ok:
        return {"used": False, "reason": "parallax gate", **parallax}
    centre, scale = np.array([width / 2.0, height / 2.0]), max(width, height) / 2.0
    normalized = (tracks - centre) / scale
    try:
        rec_a = projective_reconstruction([normalized[:, i] for i in cfg["subset_a"]], seed)
        rec_b = projective_reconstruction([normalized[:, i] for i in cfg["subset_b"]], seed)
    except (RuntimeError, np.linalg.LinAlgError) as error:
        return {"used": False, "reason": f"reconstruction failed: {error}"}
    D = pencil_discrepancies(rec_a["X"], rec_b["X"], max_pencils=cfg.get("pencils_per_window", 50_000), seed=seed)
    D = D[np.isfinite(D)]
    if D.size == 0:
        return {"used": False, "reason": "no finite pencil cross-ratio"}
    return {"used": True, "statistic": float(np.quantile(D, cfg.get("pencil_quantile", 0.1))),
            "n_pencils": int(D.size), "rms_a": rec_a["rms"], "rms_b": rec_b["rms"],
            "kappa_values": [], "outlier_fraction": outlier_fraction, **parallax}


def analyze_video(path: Path | str, config: dict, seed: int, nulls: ForensicNulls | None = None,
                  frame_stride: int = 5, alpha: float = 0.05) -> ClipEvidence:
    cfg_ingest, cfg = config["ingest"], config["video"]
    nulls = nulls or ForensicNulls()
    clip = decode(path, cfg_ingest["video_sample_fps"], cfg_ingest["max_frames_per_clip"])
    shots = split_shots(clip, cfg_ingest["shot_threshold"], cfg_ingest["min_shot_len_frames"])
    rng = np.random.default_rng(seed)
    reasons: list[str] = []
    if nulls.frame_null is None or nulls.window_null is None:
        reasons.append("uncalibrated: no empirical null supplied; decision is not a result")

    window_records, window_p = [], []
    span = (cfg["window_views"] - 1) * cfg["view_spacing"] + 1
    for start, end in shots:
        for w0 in range(start, end - span + 1, span):
            frames = [clip.frames[w0 + k * cfg["view_spacing"]] for k in range(cfg["window_views"])]
            record = window_statistic(frames, config, int(rng.integers(2 ** 31)), nulls.kappa_max)
            window_records.append(record)
            if record["used"] and nulls.window_null is not None:
                window_p.append(float(nulls.window_null.p_value(record["statistic"])[0]))

    frame_p = []
    for start, end in shots:
        for i in range(start, end, frame_stride):
            gray = clip.frames[i] if clip.frames[i].ndim == 2 else clip.frames[i].mean(axis=2).astype(np.uint8)
            evidence = analyze_gray(gray, config, rng)
            if evidence.decision == "insufficient":
                continue
            z = np.array([c.z for c in evidence.constraints])
            if nulls.frame_null is not None:
                frame_p.append(float(nulls.frame_null.p_value(image_statistic(z))[0]))
            else:
                frame_p.append(evidence.p_value_chi2)

    used = [r for r in window_records if r["used"]]
    all_p = np.asarray(window_p + frame_p, dtype=float)
    if not shots:
        reasons.append("no shot of sufficient length")
    if not used:
        reasons.append("no window passed the gates")
    if all_p.size == 0:
        statistic, p_value, n_eff, decision = float("nan"), float("nan"), 0.0, "insufficient"
    else:
        statistic, p_value, n_eff = block_fisher(all_p)
        decision = "inconsistent" if p_value <= alpha else "consistent"
    u = stats.norm.isf(np.clip(np.asarray(frame_p, dtype=float), 1e-300, 1.0)) if frame_p else np.zeros(0)
    _, alarm = cusum(u, cfg["cusum_k"], cfg["cusum_h"]) if u.size else (None, None)
    return ClipEvidence(
        source=str(path), n_frames_sampled=len(clip.frames), n_windows_total=len(window_records),
        n_windows_used=len(used), frame_p_values=frame_p,
        window_statistics=[r["statistic"] for r in used], clip_statistic=statistic,
        clip_p_value_chi2=p_value, effective_sample_size=n_eff, cusum_alarm_index=alarm,
        temporal_tv=temporal_total_variation(u), decision=decision,
        reasons=reasons + sorted({r["reason"] for r in window_records if not r["used"]}),
    )
