"""
pipeline/main_pipeline.py

End-to-end CAPRI pipeline: input images -> certified 3D reconstruction.

Stages:
  1. Feature detection (SIFT via OpenCV)
  2. Feature matching across image pairs
  3. Fundamental matrix estimation + projective triangulation
  4. CAPRI reliability layer on triangulated 3D points
     a. Six-point tuple formation from 3D points
     b. Bracket computation + degeneracy classification
     c. Conditioning score filtering
     d. Six-point invariant evaluation (J1, J2, J3)
     e. Multi-view consistency residual
     f. Bounded-drift threshold application
     g. Shape prior completion (sparse-view cases)
     h. Confidence fusion scoring
  5. Reliability certificate assembly
  6. PLY export + visualization

The reliability layer runs POST-triangulation on actual 3D points, not on
2D keypoints padded with z=0. This is mathematically required because the
six-point invariants are defined for points in P^3 (not coplanar points).

Run:
    python pipeline/main_pipeline.py --image_dir path/to/photos --output_dir outputs/run1
"""

from __future__ import annotations
import argparse
import json
import sys
import time
import numpy as np
from pathlib import Path
from itertools import combinations

# Allow `python pipeline/main_pipeline.py ...` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


from functools import lru_cache

from algorithms.invariants.six_point import evaluate_six_point_invariants
from algorithms.utils import whiten_homogeneous_points
from algorithms.conditioning.scores import compute_conditioning_score
from algorithms.degeneracy.stratification import (
    classify_stratum_by_rank,
    DegeneracyStratum,
)
from ml_models.confidence_fusion.features import (
    build_fusion_features_neural,
    load_fusion_model_neural,
    predict_fusion_probability,
)
from algorithms.selection.greedy import select_tuples_greedy
from algorithms.selection.coverage import select_tuples_max_coverage
from algorithms.conditioning.scores import batch_invariant_noise_gain, compute_invariant_noise_gain
from algorithms.consistency.pencil import pencil_discrepancies, pencil_point_scores
from algorithms.utils import embed_affine_to_homogeneous
from algorithms.backend import backend_name
from reconstruction.triangulation import (
    estimate_fundamental_matrix_eight_point,
    estimate_projection_matrices_from_fundamental,
    triangulate_point_cloud,
)
from reconstruction.certification import (
    TupleCertificate,
    assemble_point_certificates,
    export_certificates_to_dict,
)
from reconstruction.visualize import visualize_reconstruction


WEIGHTS_DIR = Path(__file__).parent.parent / "ml_models" / "weights"


def load_images(image_dir: Path) -> list[tuple[str, np.ndarray]]:
    """
    Load all .jpg, .jpeg, and .png images from a directory.

    Returns:
        List of (filename, image_array) tuples.
    """
    assert CV2_AVAILABLE, "opencv-python required. Run: pip install opencv-python"
    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix in extensions
    )
    assert len(image_paths) >= 2, (
        f"Need at least 2 images, found {len(image_paths)} in {image_dir}"
    )
    images = []
    for p in image_paths:
        img = cv2.imread(str(p))
        assert img is not None, f"Failed to load image: {p}"
        images.append((p.name, img))
    print(f"Loaded {len(images)} images from {image_dir}")
    return images


def detect_sift_keypoints(
    image_bgr: np.ndarray,
    n_features: int = 500
) -> tuple[list, np.ndarray]:
    """
    Detect SIFT keypoints and compute descriptors for one image.

    Args:
        image_bgr: BGR image array from cv2.
        n_features: Maximum keypoints to detect.

    Returns:
        (keypoints, descriptors) list of cv2.KeyPoint and (N, 128) array.
    """
    assert CV2_AVAILABLE
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_keypoints(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_threshold: float = 0.75
) -> list[tuple[int, int, float]]:
    """
    Match SIFT descriptors between two images using Lowe's ratio test.

    Args:
        desc1: (N, 128) descriptor array for image 1.
        desc2: (M, 128) descriptor array for image 2.
        ratio_threshold: Lowe ratio test threshold.

    Returns:
        List of (idx1, idx2, confidence) for accepted matches.
    """
    assert CV2_AVAILABLE
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)
    good_matches = []
    for match_pair in raw_matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < ratio_threshold * n.distance:
            confidence = float(1.0 - m.distance / (n.distance + 1e-9))
            good_matches.append((m.queryIdx, m.trainIdx, confidence))
    return good_matches


CONDITIONING_METRICS = {
    "kappa": compute_invariant_noise_gain,   # noise gain of log J1-J3 (docs/MATHEMATICS.md, Section 5)
    "L": compute_conditioning_score,         # L(S) = 8 / delta_min^2 (upper proxy, Prop. 5.2)
}


@lru_cache(maxsize=8)
def calibrated_conditioning_threshold(quantile: float = 0.5, n_points: int = 12,
                                      n_samples: int = 2000, seed: int = 9001, metric: str = "kappa") -> float:
    """
    Default conditioning threshold: the `quantile` of the chosen score over
    random six-tuples of generic synthetic configurations, processed exactly
    like a reconstruction (whitened frame). With quantile 0.5 about half of
    generic tuples pass (the "calibrated" operating point of ablation study). For L(S)
    the fixed value 1000 accepts under 1% of generic tuples and is available
    only as an explicit --conditioning_threshold choice.
    """
    from synthetic_data.generator import generate_generic_configuration

    score_fn = CONDITIONING_METRICS[metric]
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_samples):
        points = whiten_homogeneous_points(generate_generic_configuration(n_points, rng=rng))
        scores.append(score_fn(points, tuple(int(i) for i in rng.choice(n_points, 6, replace=False))))
    return float(np.quantile(np.asarray(scores), quantile))


def run_capri_reliability_layer(
    points_3d_homogeneous: np.ndarray,
    descriptor_confidences: np.ndarray,
    conditioning_threshold: float | None = None,
    rho_threshold: float = 0.05,
    deformation_bound: float = 0.02,
    max_tuples: int = 200,
    seed: int = 42,
    ml_model=None,
    device: str = "cpu",
    whiten: bool = True,
    stratum_tolerance: float = 1e-3,
    conditioning_metric: str = "kappa",
) -> tuple[list[TupleCertificate], list[tuple]]:
    """
    Run the CAPRI reliability layer on triangulated 3D points.

    Data flow: 2D matches -> RANSAC fundamental
    matrix -> projective cameras P1 = [I | 0], P2 from F -> DLT triangulation
    of every inlier match -> these (4, N) homogeneous points are the input
    here. CAPRI removes MATCHES: a match survives only if its 3D point belongs
    to at least one accepted tuple.

    Tuple selection: with conditioning_metric="kappa" (default), greedy
    maximum coverage over the tuples with kappa_J below the threshold, which
    carries the (1 - 1/e) guarantee (algorithms/selection/coverage.py); with
    "L", the penalised greedy on L(S) (algorithms/selection/greedy.py).

    Per selected tuple the decision is:
        1. conditioning score below conditioning_threshold (re-checked here so
           the function is safe on its own);
        2. reject the CRITICAL stratum (rank classifier: largest coplanar subset,
           tolerance stratum_tolerance in the whitened frame; stratification experiment) and tuples
           whose invariants are undefined.

    There is no consistency gate inside one reconstruction: all 15 brackets
    come from one triangulated point set, so every Plucker relation holds and
    rho(S) = 0 up to rounding (docs/MATHEMATICS.md, Corollary 2.6), so a
    rho <= max(rho_threshold, tau_exact) step would never reject anything;
    rho_threshold and deformation_bound are accepted for interface
    compatibility and ignored. Consistency is tested ACROSS reconstructions
    instead (cross_pair_pencil_consistency, Section 4).

    If ml_model (the neural Confidence Fusion network) is given, its probability
    must be >= 0.5 as well. It is off by default because the neural model predicts
    the positive class for every validation sample.

    Frame: with whiten=True (default) all scores are computed in the whitened
    frame (algorithms.utils.whiten_homogeneous_points). J1-J3 and rho are
    unaffected by this linear change of frame; L(S) and the stratum tolerances
    become independent of the arbitrary frame of the projective reconstruction.
    conditioning_threshold=None uses calibrated_conditioning_threshold(metric=...).
    The certificate field conditioning_score holds the score of the active metric.

    Returns:
        (all_certificates, accepted_tuple_indices)
    """
    n_points = points_3d_homogeneous.shape[1]
    if n_points < 6:
        return [], []
    if conditioning_metric not in CONDITIONING_METRICS:
        raise ValueError(f"conditioning_metric must be one of {sorted(CONDITIONING_METRICS)}")
    if ml_model is not None and conditioning_metric != "L":
        raise ValueError("the neural fusion model was trained on L(S) features; use conditioning_metric='L'")
    if conditioning_threshold is None:
        conditioning_threshold = calibrated_conditioning_threshold(metric=conditioning_metric)
    if whiten:
        points_3d_homogeneous = whiten_homogeneous_points(points_3d_homogeneous)
    score_fn = CONDITIONING_METRICS[conditioning_metric]

    rng = np.random.default_rng(seed)
    if conditioning_metric == "kappa":
        selected_tuples, _ = select_tuples_max_coverage(
            points_3d_homogeneous, score_fn, conditioning_threshold, max_tuples, rng=rng,
            batch_score_fn=batch_invariant_noise_gain)
    else:
        selected_tuples = select_tuples_greedy(
            points_3d_homogeneous,
            conditioning_threshold=conditioning_threshold,
            max_tuples=max_tuples,
            rng=rng,
        )
    all_certs: list[TupleCertificate] = []
    accepted_tuples: list[tuple] = []

    for t in selected_tuples:
        t = tuple(int(i) for i in t)
        stratum = classify_stratum_by_rank(points_3d_homogeneous, t, tolerance=stratum_tolerance)
        l_s = score_fn(points_3d_homogeneous, t)
        invariants = evaluate_six_point_invariants(points_3d_homogeneous, t)
        desc_conf = float(np.mean(descriptor_confidences[list(t)]))

        if (
            stratum == DegeneracyStratum.CRITICAL
            or invariants is None
            or not np.isfinite(l_s)
            or l_s >= conditioning_threshold
        ):
            all_certs.append(TupleCertificate(
                tuple_indices=t,
                conditioning_score=l_s,
                consistency_residual=float("nan"),
                drift_threshold=float("nan"),
                stratum=stratum.value,
                accepted=False,
                descriptor_confidence=desc_conf,
            ))
            continue

        accepted = True
        if ml_model is not None:
            features = build_fusion_features_neural(points_3d_homogeneous, t, conditioning_threshold)
            accepted = predict_fusion_probability(ml_model, features, device) >= 0.5

        all_certs.append(TupleCertificate(
            tuple_indices=t,
            conditioning_score=l_s,
            consistency_residual=float("nan"),      # not defined within one reconstruction (Cor. 2.6)
            drift_threshold=float("nan"),
            stratum=stratum.value,
            accepted=bool(accepted),
            descriptor_confidence=desc_conf,
        ))
        if accepted:
            accepted_tuples.append(t)

    return all_certs, accepted_tuples


def cross_pair_pencil_consistency(pair_reconstructions: dict, min_common: int = 8,
                                  flag_quantile: float = 0.9) -> list[dict]:
    """
    Consistency ACROSS reconstructions (docs/MATHEMATICS.md, Section 4).

    Two image pairs that share an image i reconstruct the keypoints of image i
    that both matched, each in its own projective frame. Those are two
    measurements of the same points, so the pencil drift statistic compares
    them: Q_0.1 of |d log|CR|| over the pencils of the common points. A wrong
    match in either pair changes that point's cross-ratios; its per-point
    median discrepancy (pencil_point_scores, a heuristic) is reported, and
    points above the flag_quantile of that pair-of-pairs are listed.

    Reported only (diagnostics.json): no threshold has been calibrated on real
    images, so the scores do not change acceptance.
    """
    reports = []
    keys = sorted(pair_reconstructions)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            shared = set(keys[a]) & set(keys[b])
            for image in shared:
                X1, kp1 = pair_reconstructions[keys[a]]
                X2, kp2 = pair_reconstructions[keys[b]]
                common, i1, i2 = np.intersect1d(kp1[image], kp2[image], return_indices=True)
                if len(common) < min_common:
                    continue
                A, B = X1[:, i1], X2[:, i2]
                D = pencil_discrepancies(A, B)
                D = D[np.isfinite(D)]
                if D.size == 0:
                    continue
                point_scores = pencil_point_scores(A, B)
                cut = np.nanquantile(point_scores, flag_quantile)
                reports.append({
                    "pairs": [list(keys[a]), list(keys[b])], "shared_image": int(image),
                    "n_common_points": int(len(common)), "pencil_q10": float(np.quantile(D, 0.1)),
                    "pencil_median": float(np.median(D)),
                    "flagged_keypoints": [int(k) for k, sc in zip(common, point_scores) if sc > cut],
                })
    return reports


def run_pipeline(
    image_dir: Path,
    output_dir: Path,
    conditioning_threshold: float | None = None,
    conditioning_metric: str = "kappa",
    rho_threshold: float = 0.05,
    deformation_bound: float = 0.02,
    n_sift_features: int = 300,
    max_tuples_per_pair: int = 150,
    seed: int = 42,
    interactive_viz: bool = False,
    use_fusion_model: bool = False,
) -> None:
    """
    Full end-to-end pipeline from image directory to certified reconstruction.

    Args:
        image_dir: Directory containing input photographs.
        output_dir: Directory for outputs (PLY, certificates, diagnostics).
        conditioning_threshold: rejection threshold for the conditioning score (None: calibrated default).
        conditioning_metric: "kappa" (kappa_J, default) or "L" (L(S)).
        rho_threshold: Consistency residual acceptance threshold.
        deformation_bound: Assumed scene deformation magnitude.
        n_sift_features: SIFT features per image.
        max_tuples_per_pair: Greedy selection cap per image pair.
        seed: Global random seed.
        interactive_viz: Open interactive viewer after reconstruction.
    """
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    print("=" * 60)
    print("CAPRI — Calibration-Free 3D Reconstruction Pipeline")
    print("=" * 60)

    device = "cpu"
    ml_model = None
    if use_fusion_model:
        ml_model = load_fusion_model_neural(device)
        print("[ML] Loaded neural Confidence Fusion model (replaces the rho check).")
    print(f"[Backend] {backend_name()}")
    if use_fusion_model and conditioning_metric != "L":
        print("[ML] The neural fusion model uses L(S) features: switching conditioning_metric to 'L'.")
        conditioning_metric = "L"
    if conditioning_threshold is None:
        conditioning_threshold = calibrated_conditioning_threshold(metric=conditioning_metric)
        print(f"[Conditioning] calibrated {conditioning_metric} threshold {conditioning_threshold:.1f} "
              "(median over generic tuples, whitened frame)")

    # Stage 1: Load images
    images = load_images(image_dir)
    n_images = len(images)

    # Stage 2: Detect features for each image
    print(f"\nDetecting SIFT features (max {n_sift_features} per image)...")
    all_keypoints = []
    all_descriptors = []
    for name, img in images:
        kps, descs = detect_sift_keypoints(img, n_sift_features)
        all_keypoints.append(kps)
        all_descriptors.append(descs)
        print(f"  {name}: {len(kps)} keypoints")

    # Stage 3: Process image pairs — triangulate FIRST, then certify
    all_points_3d_list: list[np.ndarray] = []
    all_trust_scores_list: list[np.ndarray] = []
    all_certificates_list: list[TupleCertificate] = []
    pair_results = []
    pair_reconstructions: dict[tuple[int, int], tuple[np.ndarray, dict[int, np.ndarray]]] = {}

    image_pairs = list(combinations(range(n_images), 2))
    print(f"\nProcessing {len(image_pairs)} image pairs...")

    for i, j in image_pairs:
        name_i, img_i = images[i]
        name_j, img_j = images[j]

        if all_descriptors[i] is None or all_descriptors[j] is None:
            continue

        # Match features
        matches = match_keypoints(all_descriptors[i], all_descriptors[j])
        if len(matches) < 8:
            print(f"  Pair ({name_i}, {name_j}): insufficient matches ({len(matches)}).")
            continue

        # Build 2D point arrays
        idx1 = [m[0] for m in matches]
        idx2 = [m[1] for m in matches]
        confidences = np.array([m[2] for m in matches])

        pts1_2d = np.array(
            [[all_keypoints[i][k].pt[0], all_keypoints[i][k].pt[1]] for k in idx1]
        )
        pts2_2d = np.array(
            [[all_keypoints[j][k].pt[0], all_keypoints[j][k].pt[1]] for k in idx2]
        )

        # Stage 3: Triangulate FIRST (before CAPRI layer)
        try:
            # Use RANSAC to robustly estimate F and filter raw outliers
            F, mask = cv2.findFundamentalMat(pts1_2d, pts2_2d, cv2.FM_RANSAC, 3.0, 0.99)
            if F is None or F.shape != (3, 3):
                print(f"  Pair ({name_i}, {name_j}): RANSAC failed to find stable F.")
                continue
                
            inliers = mask.ravel() == 1
            keypoints_i, keypoints_j = np.asarray(idx1)[inliers], np.asarray(idx2)[inliers]
            pts1_2d = pts1_2d[inliers]
            pts2_2d = pts2_2d[inliers]
            confidences = confidences[inliers]
            
            if len(pts1_2d) < 8:
                continue

            P1, P2 = estimate_projection_matrices_from_fundamental(
                F, pts1_2d, pts2_2d
            )
            proj_mats = np.stack([P1, P2], axis=0)
            pts_3d = triangulate_point_cloud(
                [pts1_2d, pts2_2d], proj_mats
            )
        except Exception as e:
            print(f"  Pair ({name_i}, {name_j}): triangulation failed: {e}")
            continue

        if pts_3d.shape[1] < 6:
            print(f"  Pair ({name_i}, {name_j}): too few 3D points ({pts_3d.shape[1]}).")
            continue
        if pts_3d.shape[1] == len(keypoints_i):
            pair_reconstructions[(i, j)] = (pts_3d, {i: keypoints_i, j: keypoints_j})

        # Stage 4: Run CAPRI reliability layer on 3D points
        certs, accepted_tuples = run_capri_reliability_layer(
            pts_3d,
            confidences[:pts_3d.shape[1]],
            conditioning_threshold=conditioning_threshold,
            conditioning_metric=conditioning_metric,
            rho_threshold=rho_threshold,
            deformation_bound=deformation_bound,
            max_tuples=max_tuples_per_pair,
            seed=seed,
            ml_model=ml_model,
            device=device,
        )

        n_accepted = len(accepted_tuples)
        n_total = len(certs)
        print(
            f"  Pair ({name_i}, {name_j}): "
            f"{len(matches)} matches, "
            f"{pts_3d.shape[1]} 3D points, "
            f"{n_total} tuples evaluated, "
            f"{n_accepted} accepted."
        )

        # Stage 5: Assemble point certificates
        point_certs = assemble_point_certificates(
            pts_3d, certs, conditioning_threshold
        )
        trust_scores = np.array([c.trust_score for c in point_certs])
        all_points_3d_list.append(pts_3d)
        all_trust_scores_list.append(trust_scores)
        all_certificates_list.extend(certs)

        pair_results.append({
            "pair": (name_i, name_j),
            "n_matches": len(matches),
            "n_tuples_evaluated": n_total,
            "n_tuples_accepted": n_accepted,
            "n_points_3d": pts_3d.shape[1],
            "mean_trust_score": float(np.mean(trust_scores)),
        })

    if not all_points_3d_list:
        print("\nNo valid reconstructions produced.")
        return

    # Each image pair is triangulated in its OWN projective frame
    # (P1 = [I | 0], P2 from F). Clouds from different pairs are related by
    # unknown 4x4 homographies, so concatenating them is meaningless. Each
    # pair is exported separately; reconstruction.ply is the pair with the
    # most certified (trust > 0) points.
    from reconstruction.visualize import save_ply
    pairs_dir = output_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    certified_counts = []
    for k, (pts_3d, trust) in enumerate(zip(all_points_3d_list, all_trust_scores_list)):
        stem = f"pair_{k:03d}"
        save_ply(pts_3d, trust, pairs_dir / f"{stem}.ply")
        np.save(pairs_dir / f"{stem}_points3d.npy", pts_3d)
        np.save(pairs_dir / f"{stem}_trust.npy", trust)
        pair_results[k]["files_stem"] = f"pairs/{stem}"
        certified_counts.append(int(np.sum(trust > 0.0)))

    best = int(np.argmax(certified_counts))
    best_points = all_points_3d_list[best]
    best_trust = all_trust_scores_list[best]
    save_ply(best_points, best_trust, output_dir / "reconstruction.ply")
    np.save(output_dir / "points3d.npy", best_points)
    np.save(output_dir / "trust_scores.npy", best_trust)

    from dataclasses import asdict
    with open(output_dir / "certificates.json", "w") as f:
        json.dump([asdict(c) for c in all_certificates_list], f, indent=2, default=float)

    all_trust = np.concatenate(all_trust_scores_list)
    cross_pair = cross_pair_pencil_consistency(pair_reconstructions)
    if cross_pair:
        q10 = [r["pencil_q10"] for r in cross_pair]
        print(f"[Consistency] {len(cross_pair)} pair-of-pairs comparisons: pencil Q_0.1 "
              f"median {np.median(q10):.3e} (range {min(q10):.3e} - {max(q10):.3e})")
    diagnostics = {
        "seed": seed,
        "backend": backend_name(),
        "fusion_model": "neural" if use_fusion_model else "none",
        "conditioning_metric": conditioning_metric,
        "conditioning_threshold": conditioning_threshold,
        "deformation_bound": deformation_bound,
        "n_images": n_images,
        "n_pairs_processed": len(pair_results),
        "primary_pair_index": best,
        "n_points_primary": int(best_points.shape[1]),
        "n_points_all_pairs": int(all_trust.size),
        "mean_trust_score_all_pairs": float(np.mean(all_trust)),
        "certified_fraction_all_pairs": float(np.mean(all_trust > 0.0)),
        "pair_results": pair_results,
        "cross_pair_pencil_consistency": cross_pair,
    }
    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)

    t_elapsed = time.time() - t_start
    print(f"\nReconstruction complete in {t_elapsed:.1f}s")
    print(f"  Pairs reconstructed: {len(pair_results)} (primary: pair {best})")
    print(f"  Points in primary reconstruction: {best_points.shape[1]}")
    print(f"  Certified fraction over all pairs: {float(np.mean(all_trust > 0.0)) * 100:.1f}%")
    print(f"  Saved to: {output_dir}")

    if interactive_viz:
        visualize_reconstruction(best_points, best_trust, output_ply_path=None, interactive=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CAPRI end-to-end 3D reconstruction pipeline."
    )
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/reconstruction")
    parser.add_argument("--conditioning_metric", choices=["kappa", "L"], default="kappa",
                        help="kappa: noise gain kappa_J (default); L: conditioning score L(S)")
    parser.add_argument("--conditioning_threshold", type=float, default=None,
                        help="threshold on the chosen score; default: calibrated (median over generic tuples). "
                             "The fixed L(S) operating point is 1000.")
    parser.add_argument("--rho_threshold", type=float, default=0.05,
                        help="ignored: rho is identically 0 within one reconstruction (Corollary 2.6)")
    parser.add_argument("--deformation_bound", type=float, default=0.02,
                        help="ignored (accepted for interface compatibility)")
    parser.add_argument("--n_sift_features", type=int, default=300)
    parser.add_argument("--max_tuples_per_pair", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--use_fusion_model", action="store_true",
                        help="Use the neural Confidence Fusion network instead of the rho check.")
    args = parser.parse_args()

    run_pipeline(
        image_dir=Path(args.image_dir),
        output_dir=Path(args.output_dir),
        conditioning_threshold=args.conditioning_threshold,
        conditioning_metric=args.conditioning_metric,
        rho_threshold=args.rho_threshold,
        deformation_bound=args.deformation_bound,
        n_sift_features=args.n_sift_features,
        max_tuples_per_pair=args.max_tuples_per_pair,
        seed=args.seed,
        interactive_viz=args.interactive,
        use_fusion_model=args.use_fusion_model,
    )