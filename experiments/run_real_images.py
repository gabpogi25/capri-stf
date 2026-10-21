"""
experiments/run_real_images.py

Real-Image Baseline Benchmarking (with DUSt3R)

Compares CAPRI's degeneracy filtering on real-world image pairs from the
Image Matching Challenge 2025 (IMC 2025) Kaggle dataset.

Pipeline Branches:
  A) Sparse 2D Branch (LightGlue, SuperGlue, BruteForce)
     - Extract SuperPoint features
     - Match 2D keypoints
     - Estimate geometry (RANSAC F-matrix)
     - Triangulate to 3D -> Certify with CAPRI

  B) Dense 3D Branch (DUSt3R)
     - Direct dense 3D pointmap inference via ViT
     - Extract high-confidence 3D points
     - Estimate relative pose via Kabsch (Procrustes)
     - Certify 3D points directly with CAPRI

Parameters:
  - Seed: 13042
  - Dataset: IMC 2025 (train split)
  - Max pairs per scene: 10
  - Max scenes: 20

NOTE: Requires DUSt3R weights (run scripts/download_dust3r_weights.py)
"""

import os
import sys
import time
import csv
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from itertools import combinations
import cv2

# Fix OpenMP duplicate library crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import (
    get_experiment_seed, make_rng, save_results, log_hardware_info,
    random_six_tuple, RESULTS_DIR,
)
from algorithms.invariants.brackets import compute_all_brackets_for_tuple
from algorithms.syzygy.residual import compute_syzygy_residual
from algorithms.conditioning.scores import compute_conditioning_score
from algorithms.degeneracy.stratification import classify_degeneracy_stratum, DegeneracyStratum

EXPERIMENT_ID = 13
SEED = get_experiment_seed(EXPERIMENT_ID)  # 13042
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real_images"
SUPERGLUE_DIR = Path(__file__).resolve().parent.parent / "SuperGluePretrainedNetwork"
DUST3R_DIR = Path(__file__).resolve().parent.parent / "dust3r"

MAX_SCENES = 20
MAX_PAIRS_PER_SCENE = 10
MAX_KEYPOINTS = 1024


# ========================================================================
# Dataset Loading
# ========================================================================
def find_imc_dataset() -> Path:
    ref_file = DATA_DIR / "imc2025_path.txt"
    if ref_file.exists():
        cached_path = Path(ref_file.read_text().strip())
        if cached_path.exists():
            return cached_path

    if (DATA_DIR / "train_labels.csv").exists():
        return DATA_DIR

    subfolder = DATA_DIR / "image-matching-challenge-2025"
    if (subfolder / "train_labels.csv").exists():
        return subfolder

    return None


def load_ground_truth(dataset_path: Path) -> dict:
    labels_file = dataset_path / "train_labels.csv"
    if not labels_file.exists():
        return {}

    gt = {}
    with open(labels_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row.get("dataset", "")
            sc = row.get("scene", "")
            im = row.get("image", "")
            r_str = row.get("rotation_matrix", "")
            t_str = row.get("translation_vector", "")
            if not r_str or not t_str:
                continue

            try:
                R = np.array([float(x) for x in r_str.split(';')]).reshape(3, 3)
                t = np.array([float(x) for x in t_str.split(';')]).reshape(3, 1)
                gt[(ds, sc, im)] = {"R": R, "t": t}
            except (ValueError, IndexError):
                continue
    return gt


def find_image_pairs_imc(dataset_path: Path, gt: dict) -> list:
    scene_images = {}
    for (ds, scene, img), pose in gt.items():
        key = (ds, scene)
        if key not in scene_images:
            scene_images[key] = []
        scene_images[key].append((img, pose))

    pairs = []
    scene_count = 0

    for (ds, scene), images in sorted(scene_images.items()):
        if scene_count >= MAX_SCENES:
            break
        images.sort(key=lambda x: x[0])
        if len(images) < 2:
            continue

        scene_count += 1
        pair_count = 0
        for i in range(len(images) - 1):
            if pair_count >= MAX_PAIRS_PER_SCENE:
                break

            img0_name, pose0 = images[i]
            img1_name, pose1 = images[i + 1]

            img0_path = dataset_path / "train" / ds / img0_name
            img1_path = dataset_path / "train" / ds / img1_name

            if not img0_path.exists() or not img1_path.exists():
                continue

            R_rel = pose1["R"] @ pose0["R"].T
            t_rel = pose1["t"] - R_rel @ pose0["t"]

            pairs.append({
                "dataset": ds, "scene": scene,
                "img0_name": img0_name, "img1_name": img1_name,
                "img0_path": str(img0_path), "img1_path": str(img1_path),
                "R_gt": R_rel, "t_gt": t_rel,
            })
            pair_count += 1
    return pairs


# ========================================================================
# Feature Extraction (Sparse Branch)
# ========================================================================
def extract_superpoint_features(image: np.ndarray, device: str = "cpu") -> dict:
    from lightglue import SuperPoint
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    img_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
    with torch.no_grad():
        feats = extractor.extract(img_tensor)

    return {
        "keypoints": feats["keypoints"],
        "descriptors": feats["descriptors"],
        "scores": feats.get("keypoint_scores", None),
    }


def extract_opencv_sift_features(image: np.ndarray, device: str = "cpu") -> dict:
    """Fallback feature extractor using OpenCV SIFT (no pycolmap needed)."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=MAX_KEYPOINTS)
    kps, descs = sift.detectAndCompute(image, None)

    if kps is None or len(kps) == 0:
        # Return empty tensors
        return {
            "keypoints": torch.zeros(1, 0, 2),
            "descriptors": torch.zeros(1, 0, 128),
            "scores": torch.zeros(1, 0),
        }

    coords = np.array([kp.pt for kp in kps], dtype=np.float32)
    scores_np = np.array([kp.response for kp in kps], dtype=np.float32)

    # Pad descriptors from 128-dim SIFT to 256-dim to match SuperPoint format
    if descs.shape[1] < 256:
        pad = np.zeros((descs.shape[0], 256 - descs.shape[1]), dtype=np.float32)
        descs = np.concatenate([descs, pad], axis=1)

    return {
        "keypoints": torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device),
        "descriptors": torch.tensor(descs, dtype=torch.float32).unsqueeze(0).to(device),
        "scores": torch.tensor(scores_np, dtype=torch.float32).unsqueeze(0).to(device),
    }


# Choose the best available feature extractor
_SUPERPOINT_AVAILABLE = None
def extract_features(image: np.ndarray, device: str = "cpu") -> dict:
    """Try SuperPoint first; fall back to OpenCV SIFT if pycolmap is blocked."""
    global _SUPERPOINT_AVAILABLE
    if _SUPERPOINT_AVAILABLE is None:
        try:
            from lightglue import SuperPoint
            # Quick smoke test to see if pycolmap backend loads
            _ = SuperPoint(max_num_keypoints=10)
            _SUPERPOINT_AVAILABLE = True
        except Exception:
            _SUPERPOINT_AVAILABLE = False
            print("    [INFO] SuperPoint unavailable (pycolmap blocked), using OpenCV SIFT fallback.")

    if _SUPERPOINT_AVAILABLE:
        return extract_superpoint_features(image, device)
    else:
        return extract_opencv_sift_features(image, device)


# ========================================================================
# Matchers (Sparse Branch)
# ========================================================================
def match_lightglue(feats0: dict, feats1: dict, device: str = "cpu") -> dict:
    from lightglue import LightGlue
    matcher = LightGlue(features="superpoint").eval().to(device)

    t_start = time.perf_counter()
    with torch.no_grad():
        result = matcher({"image0": feats0, "image1": feats1})
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    matches = result["matches0"][0].cpu().numpy()
    valid = matches > -1
    return {
        "mkpts0": feats0["keypoints"][0][valid].cpu().numpy(),
        "mkpts1": feats1["keypoints"][0][matches[valid]].cpu().numpy(),
        "n_matches": int(valid.sum()),
        "time_ms": elapsed_ms,
    }


def match_superglue(feats0: dict, feats1: dict, device: str = "cpu") -> dict:
    if str(SUPERGLUE_DIR) not in sys.path:
        sys.path.insert(0, str(SUPERGLUE_DIR))
    from models.superglue import SuperGlue

    config = {
        "descriptor_dim": 256, "weights": "outdoor",
        "keypoint_encoder": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn_iterations": 20, "match_threshold": 0.2,
    }
    matcher = SuperGlue(config).eval().to(device)
    weights_path = SUPERGLUE_DIR / "models" / "weights" / "superglue_outdoor.pth"
    matcher.load_state_dict(torch.load(str(weights_path), map_location=device, weights_only=True))

    n0, n1 = feats0["keypoints"].shape[1], feats1["keypoints"].shape[1]
    data = {
        "keypoints0": feats0["keypoints"].to(device),
        "keypoints1": feats1["keypoints"].to(device),
        "descriptors0": feats0["descriptors"].permute(0, 2, 1).to(device),
        "descriptors1": feats1["descriptors"].permute(0, 2, 1).to(device),
        "scores0": feats0["scores"].to(device) if feats0["scores"] is not None else torch.ones(1, n0).to(device),
        "scores1": feats1["scores"].to(device) if feats1["scores"] is not None else torch.ones(1, n1).to(device),
    }

    t_start = time.perf_counter()
    with torch.no_grad():
        result = matcher(data)
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    matches = result["matches0"][0].cpu().numpy()
    valid = matches > -1
    return {
        "mkpts0": feats0["keypoints"][0][valid].cpu().numpy(),
        "mkpts1": feats1["keypoints"][0][matches[valid]].cpu().numpy(),
        "n_matches": int(valid.sum()),
        "time_ms": elapsed_ms,
    }


def match_bruteforce(feats0: dict, feats1: dict) -> dict:
    desc0 = feats0["descriptors"][0].cpu().numpy()
    desc1 = feats1["descriptors"][0].cpu().numpy()
    kp0 = feats0["keypoints"][0].cpu().numpy()
    kp1 = feats1["keypoints"][0].cpu().numpy()

    t_start = time.perf_counter()
    norms0 = np.sum(desc0 ** 2, axis=1, keepdims=True)
    norms1 = np.sum(desc1 ** 2, axis=1, keepdims=True)
    dists_sq = np.maximum(norms0 + norms1.T - 2 * (desc0 @ desc1.T), 0)

    matches = []
    for i in range(len(desc0)):
        sorted_idx = np.argsort(dists_sq[i])
        if len(sorted_idx) >= 2:
            d1, d2 = np.sqrt(dists_sq[i, sorted_idx[0]]), np.sqrt(dists_sq[i, sorted_idx[1]])
            if d2 > 1e-10 and d1 / d2 < 0.75:
                matches.append((i, sorted_idx[0]))
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    if matches:
        idx0, idx1 = np.array([m[0] for m in matches]), np.array([m[1] for m in matches])
        return {"mkpts0": kp0[idx0], "mkpts1": kp1[idx1], "n_matches": len(matches), "time_ms": elapsed_ms}
    return {"mkpts0": np.zeros((0, 2)), "mkpts1": np.zeros((0, 2)), "n_matches": 0, "time_ms": elapsed_ms}


# ========================================================================
# DUSt3R (Dense Branch)
# ========================================================================
def load_dust3r_model(device: str):
    """Load the DUSt3R ViT-Large model."""
    if str(DUST3R_DIR) not in sys.path:
        sys.path.insert(0, str(DUST3R_DIR))
    try:
        from dust3r.model import AsymmetricCroCo3DStereo
    except ImportError as e:
        print(f"    DUSt3R import failed: {e}")
        return None

    weights_path = DUST3R_DIR / "checkpoints" / "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    if not weights_path.exists():
        print("    DUSt3R weights not found. Run: python scripts/download_dust3r_weights.py")
        return None

    try:
        model = AsymmetricCroCo3DStereo.from_pretrained(str(weights_path)).to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"    DUSt3R model load failed: {e}")
        return None


def run_dust3r(model, img0_path: str, img1_path: str, device: str) -> dict:
    """
    Run DUSt3R to get dense 3D pointmaps and estimated relative pose.

    DUSt3R API:
      1. load_images() -> list of dicts with {'img': tensor, 'true_shape': ...}
      2. make_pairs()  -> list of (view1, view2) tuples
      3. inference()   -> dict with 'pred1', 'pred2', 'view1', 'view2'
         - pred1['pts3d']: (B, H, W, 3) -- 3D points in camera 1 frame
         - pred2['pts3d_in_other_view']: (B, H, W, 3) -- view2's points in camera 1 frame
         - pred1['conf']: (B, H, W) -- confidence per pixel
    """
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs

    t_start = time.perf_counter()

    # Step 1: Load and preprocess images (RGB, resize to 512, normalize)
    imgs = load_images([img0_path, img1_path], size=512, verbose=False)

    # Step 2: Create pairs (for 2 images, this gives 1 pair)
    pairs = make_pairs(imgs, scene_graph='complete', prefilter=None, symmetrize=True)

    # Step 3: Run inference through the ViT
    output = inference(pairs, model, device, batch_size=1, verbose=False)

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Step 4: Extract 3D pointmaps
    # pred1['pts3d'] contains 3D points from view1 in view1's coordinate frame
    # pred2['pts3d_in_other_view'] contains 3D points from view2 in view1's frame
    pred1 = output['pred1']
    pred2 = output['pred2']

    # Get the pointmaps (B, H, W, 3)
    pts3d_v1 = pred1['pts3d'][0].cpu().numpy()       # (H, W, 3) -- view1's 3D pts
    conf1 = pred1['conf'][0].cpu().numpy()            # (H, W)

    # View2's points expressed in view1's frame
    if 'pts3d_in_other_view' in pred2:
        pts3d_v2_in_v1 = pred2['pts3d_in_other_view'][0].cpu().numpy()  # (H, W, 3)
    elif 'pts3d' in pred2:
        pts3d_v2_in_v1 = pred2['pts3d'][0].cpu().numpy()
    else:
        return {"pts_3d": np.zeros((3, 0)), "R": np.eye(3), "t": np.zeros((3, 1)),
                "time_ms": elapsed_ms, "accepted": False}

    conf2 = pred2['conf'][0].cpu().numpy() if 'conf' in pred2 else np.ones_like(conf1)

    # Step 5: Flatten and select top-confidence 3D points
    v1_flat = pts3d_v1.reshape(-1, 3)
    v2_flat = pts3d_v2_in_v1.reshape(-1, 3)
    c1_flat = conf1.flatten()
    c2_flat = conf2.flatten()

    # Combined confidence (both views should be confident for a good point)
    combined_conf = np.minimum(c1_flat, c2_flat)

    # Filter out NaN/Inf points
    valid = (np.isfinite(v1_flat).all(axis=1) & np.isfinite(v2_flat).all(axis=1)
             & np.isfinite(combined_conf))
    v1_valid = v1_flat[valid]
    v2_valid = v2_flat[valid]
    conf_valid = combined_conf[valid]

    top_k = min(1000, len(conf_valid))
    if top_k < 6:
        return {"pts_3d": np.zeros((3, 0)), "R": np.eye(3), "t": np.zeros((3, 1)),
                "time_ms": elapsed_ms, "accepted": False}

    top_indices = np.argsort(conf_valid)[-top_k:]

    pts_3d_high_conf = v1_valid[top_indices]  # Use View1's 3D points for CAPRI

    # Step 6: Estimate relative pose via Kabsch algorithm
    # We have corresponding 3D points in two coordinate frames:
    #   src = view2's own 3D points (need view2's pts in view2's frame... but DUSt3R
    #   directly gives us view2 in view1 frame). For relative pose estimation,
    #   we use the actual geometry from the pointmaps.
    # Since both pointmaps are in view1's frame, we estimate the rigid transform
    # from the view2 pointmap to view1 pointmap.
    src_pts = v2_valid[top_indices]  # View2's points (in view1 frame)
    dst_pts = v1_valid[top_indices]  # View1's points (in view1 frame)

    # Kabsch algorithm
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)
    src_centered = src_pts - src_mean
    dst_centered = dst_pts - dst_mean

    H = src_centered.T @ dst_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection correction
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = dst_mean - R @ src_mean

    return {
        "pts_3d": pts_3d_high_conf.T,  # (3, N) for CAPRI
        "R": R,
        "t": t.reshape(3, 1),
        "time_ms": elapsed_ms,
        "accepted": True,
    }


# ========================================================================
# Geometry Estimation & CAPRI Certification
# ========================================================================
def estimate_pose_ransac(mkpts0: np.ndarray, mkpts1: np.ndarray, K: np.ndarray) -> dict:
    if len(mkpts0) < 8:
        return {"R": None, "t": None, "inliers": 0, "accepted": False, "time_ms": 0}

    t_start = time.perf_counter()
    F, mask_f = cv2.findFundamentalMat(
        mkpts0.astype(np.float64), mkpts1.astype(np.float64),
        cv2.FM_RANSAC, 1.0, 0.999)

    if F is None or F.shape != (3, 3):
        return {"R": None, "t": None, "inliers": 0, "accepted": False,
                "time_ms": (time.perf_counter() - t_start) * 1000}

    E = K.T @ F @ K
    n_inliers, R, t, _ = cv2.recoverPose(
        E, mkpts0.astype(np.float64), mkpts1.astype(np.float64), K)

    return {
        "R": R, "t": t, "F": F, "inliers": n_inliers,
        "inlier_ratio": int(mask_f.sum()) / len(mkpts0) if mask_f is not None else 0,
        "accepted": n_inliers >= 5,
        "time_ms": (time.perf_counter() - t_start) * 1000,
    }


def triangulate_2d_matches(mkpts0, mkpts1, K, R, t) -> np.ndarray:
    """Triangulate 2D matches into 3D points using camera projection matrices."""
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t.reshape(3, 1)])
    pts_4d = cv2.triangulatePoints(
        P0.astype(np.float64), P1.astype(np.float64),
        mkpts0.T.astype(np.float64), mkpts1.T.astype(np.float64))

    w = pts_4d[3, :]
    valid = np.abs(w) > 1e-10
    pts_3d = np.zeros((3, len(mkpts0)))
    pts_3d[:, valid] = pts_4d[:3, valid] / w[valid]
    return pts_3d


def run_capri_certification(pts_3d: np.ndarray, rng: np.random.Generator) -> dict:
    """
    Run CAPRI algebraic certification on a 3D point cloud.

    Samples 6-point tuples, computes bracket invariants, syzygy residuals,
    conditioning scores, and classifies degeneracy strata. A majority vote
    decides whether the geometry is CAPRI-accepted.
    """
    n = pts_3d.shape[1]
    if n < 6:
        return {"capri_accepted": False, "accept_ratio": 0.0,
                "n_tuples_tested": 0, "n_good_3d": n}

    n_tuples = min(50, n * (n - 1) // 2)
    n_accepted = 0
    strata_counts = {}

    for i in range(n_tuples):
        t_idx = random_six_tuple(n, make_rng(SEED + i * 7 + 31337))
        try:
            brackets = compute_all_brackets_for_tuple(pts_3d, t_idx)
            rho = compute_syzygy_residual(brackets)
            ls = compute_conditioning_score(pts_3d, t_idx)
            stratum = classify_degeneracy_stratum(pts_3d, t_idx)

            stratum_name = stratum.name
            strata_counts[stratum_name] = strata_counts.get(stratum_name, 0) + 1

            if (stratum not in {DegeneracyStratum.CRITICAL, DegeneracyStratum.COMPOUND}
                    and ls < 1000.0 and rho < 0.05):
                n_accepted += 1
        except Exception:
            pass  # Computation failure -> treat as degenerate

    accept_ratio = n_accepted / n_tuples if n_tuples > 0 else 0.0

    return {
        "capri_accepted": accept_ratio > 0.5,  # Majority vote
        "accept_ratio": accept_ratio,
        "n_tuples_tested": n_tuples,
        "n_good_3d": n,
        "strata_distribution": strata_counts,
    }


# ========================================================================
# Metrics
# ========================================================================
def compute_pose_error(R_est, t_est, R_gt, t_gt) -> dict:
    """Compute rotation and translation angular error in degrees."""
    if R_est is None or t_est is None:
        return {"rot_err_deg": 180.0, "trans_err_deg": 180.0, "max_err_deg": 180.0}

    R_err = R_est @ R_gt.T
    cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    rot_err = np.degrees(np.arccos(cos_angle))

    t_est_f, t_gt_f = t_est.flatten(), t_gt.flatten()
    n_est, n_gt = np.linalg.norm(t_est_f), np.linalg.norm(t_gt_f)
    if n_est < 1e-10 or n_gt < 1e-10:
        trans_err = 180.0
    else:
        trans_err = np.degrees(np.arccos(np.clip(
            np.dot(t_est_f / n_est, t_gt_f / n_gt), -1.0, 1.0)))

    return {
        "rot_err_deg": float(rot_err),
        "trans_err_deg": float(trans_err),
        "max_err_deg": float(max(rot_err, trans_err)),
    }


def compute_auc(errors: np.ndarray, thresholds=None) -> dict:
    """
    Accuracy at each threshold: the fraction of pairs with error <= threshold.
    This is a single point of the accuracy curve, not an area, so the keys are
    Acc@<threshold> despite the function name.
    """
    if thresholds is None:
        thresholds = [5, 10, 20]
    return {f"Acc@{thr}": float(np.mean(errors <= thr)) for thr in thresholds}


# ========================================================================
# Main Experiment Loop
# ========================================================================
def run_experiment():
    print("=" * 60)
    print("EXP 13: REAL-IMAGE BASELINE BENCHMARKING (WITH DUSt3R)")
    print("=" * 60)

    hw_info = log_hardware_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # ---- Locate dataset ----
    dataset_path = find_imc_dataset()
    if not dataset_path:
        print("  ERROR: IMC 2025 dataset not found.")
        print(f"  Expected at: {DATA_DIR}")
        print("  Run: python scripts/download_imc_dataset.py")
        return

    print(f"  Dataset: {dataset_path}")

    # ---- Load ground truth ----
    gt = load_ground_truth(dataset_path)
    if not gt:
        print("  ERROR: No ground-truth poses found.")
        return
    print(f"  Ground-truth poses loaded: {len(gt)}")

    # ---- Generate image pairs ----
    pairs = find_image_pairs_imc(dataset_path, gt)
    if not pairs:
        print("  ERROR: No valid image pairs found.")
        return

    n_scenes = len(set((p["dataset"], p["scene"]) for p in pairs))
    print(f"  Image pairs: {len(pairs)} across {n_scenes} scenes")

    # ---- Check available matchers ----
    available_matchers = {}

    try:
        from lightglue import LightGlue, SuperPoint
        available_matchers["LightGlue"] = match_lightglue
        print("  LightGlue: available")
    except (ImportError, RuntimeError, OSError) as e:
        print(f"  LightGlue: NOT available ({e})")

    if (SUPERGLUE_DIR.exists()
            and (SUPERGLUE_DIR / "models" / "weights" / "superglue_outdoor.pth").exists()):
        available_matchers["SuperGlue"] = match_superglue
        print("  SuperGlue: available")
    else:
        print("  SuperGlue: NOT available")

    available_matchers["BruteForce+Ratio"] = match_bruteforce
    print("  BruteForce+Ratio: available")

    # ---- Load DUSt3R ----
    dust3r_model = None
    try:
        dust3r_model = load_dust3r_model(device)
        if dust3r_model:
            print("  DUSt3R: available (ViT-Large)")
        else:
            print("  DUSt3R: NOT available")
    except Exception as e:
        print(f"  DUSt3R: NOT available ({e})")

    # ---- Run benchmark ----
    results = []

    for pair_idx, pair in enumerate(pairs):
        scene_label = f"{pair['dataset']}/{pair['scene']}"
        print(f"\n  [{pair_idx + 1}/{len(pairs)}] {scene_label}: "
              f"{pair['img0_name']} <-> {pair['img1_name']}")

        # Load images
        img0 = cv2.imread(pair["img0_path"], cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(pair["img1_path"], cv2.IMREAD_GRAYSCALE)

        if img0 is None or img1 is None:
            print("    Skipping: could not load images")
            continue

        # Approximate camera intrinsics
        h, w = img0.shape[:2]
        focal = max(h, w) * 1.2
        K = np.array([[focal, 0, w / 2],
                       [0, focal, h / 2],
                       [0, 0, 1]], dtype=np.float64)

        # =============================================================
        # SPARSE BRANCH: SuperPoint + Matchers + RANSAC + CAPRI
        # =============================================================
        try:
            feats0 = extract_features(img0, device)
            feats1 = extract_features(img1, device)
            n_kp0 = feats0["keypoints"].shape[1]
            n_kp1 = feats1["keypoints"].shape[1]
        except Exception as e:
            print(f"    Feature extraction error: {e}")
            continue

        for m_name, m_func in available_matchers.items():
            print(f"    {m_name}...", end="")

            try:
                if m_name == "BruteForce+Ratio":
                    m_res = m_func(feats0, feats1)
                else:
                    m_res = m_func(feats0, feats1, device)
            except Exception as e:
                print(f" error: {e}")
                continue

            # Estimate pose via RANSAC
            p_res = estimate_pose_ransac(m_res["mkpts0"], m_res["mkpts1"], K)

            # Pose error vs ground truth
            p_err = compute_pose_error(p_res["R"], p_res["t"],
                                       pair["R_gt"], pair["t_gt"])

            # CAPRI certification
            capri = {"capri_accepted": False, "accept_ratio": 0.0,
                     "n_tuples_tested": 0}
            if p_res["R"] is not None and m_res["n_matches"] >= 6:
                pts_3d = triangulate_2d_matches(
                    m_res["mkpts0"], m_res["mkpts1"], K, p_res["R"], p_res["t"])
                # Filter: positive depth and not too far
                valid = (pts_3d[2, :] > 0) & (np.linalg.norm(pts_3d, axis=0) < 1000)
                capri = run_capri_certification(pts_3d[:, valid], make_rng(SEED))

            print(f" matches={m_res['n_matches']}, "
                  f"err={p_err['max_err_deg']:.1f}deg, "
                  f"CAPRI={'PASS' if capri.get('capri_accepted') else 'REJECT'}")

            results.append({
                "Dataset": pair["dataset"],
                "Scene": pair["scene"],
                "Pair_Idx": pair_idx,
                "Img0": pair["img0_name"],
                "Img1": pair["img1_name"],
                "Matcher": m_name,
                "N_Matches": m_res["n_matches"],
                "Match_Time_ms": m_res["time_ms"],
                "RANSAC_Accepted": p_res["accepted"],
                "RANSAC_Inliers": p_res["inliers"],
                "RANSAC_Time_ms": p_res["time_ms"],
                "Rot_Error_deg": p_err["rot_err_deg"],
                "Trans_Error_deg": p_err["trans_err_deg"],
                "Max_Error_deg": p_err["max_err_deg"],
                "CAPRI_Accepted": capri.get("capri_accepted", False),
                "CAPRI_Accept_Ratio": capri.get("accept_ratio", 0.0),
                "CAPRI_Tuples_Tested": capri.get("n_tuples_tested", 0),
                "Total_Time_ms": m_res["time_ms"] + p_res["time_ms"],
                "Seed": SEED,
            })

        # =============================================================
        # DENSE BRANCH: DUSt3R -> Direct 3D -> CAPRI
        # =============================================================
        if dust3r_model:
            print(f"    DUSt3R...", end="")
            try:
                d_res = run_dust3r(dust3r_model, pair["img0_path"],
                                   pair["img1_path"], device)

                if d_res["accepted"] and d_res["pts_3d"].shape[1] >= 6:
                    p_err = compute_pose_error(d_res["R"], d_res["t"],
                                               pair["R_gt"], pair["t_gt"])
                    capri = run_capri_certification(d_res["pts_3d"], make_rng(SEED))

                    print(f" dense_pts={d_res['pts_3d'].shape[1]}, "
                          f"err={p_err['max_err_deg']:.1f}deg, "
                          f"CAPRI={'PASS' if capri.get('capri_accepted') else 'REJECT'}")
                else:
                    p_err = {"rot_err_deg": 180.0, "trans_err_deg": 180.0, "max_err_deg": 180.0}
                    capri = {"capri_accepted": False, "accept_ratio": 0.0, "n_tuples_tested": 0}
                    print(f" FAILED (too few valid points)")

                results.append({
                    "Dataset": pair["dataset"],
                    "Scene": pair["scene"],
                    "Pair_Idx": pair_idx,
                    "Img0": pair["img0_name"],
                    "Img1": pair["img1_name"],
                    "Matcher": "DUSt3R",
                    "N_Matches": d_res["pts_3d"].shape[1],
                    "Match_Time_ms": d_res["time_ms"],
                    "RANSAC_Accepted": True,  # DUSt3R always produces a pose
                    "RANSAC_Inliers": d_res["pts_3d"].shape[1],
                    "RANSAC_Time_ms": 0,
                    "Rot_Error_deg": p_err["rot_err_deg"],
                    "Trans_Error_deg": p_err["trans_err_deg"],
                    "Max_Error_deg": p_err["max_err_deg"],
                    "CAPRI_Accepted": capri.get("capri_accepted", False),
                    "CAPRI_Accept_Ratio": capri.get("accept_ratio", 0.0),
                    "CAPRI_Tuples_Tested": capri.get("n_tuples_tested", 0),
                    "Total_Time_ms": d_res["time_ms"],
                    "Seed": SEED,
                })

            except Exception as e:
                print(f" error: {e}")

    if not results:
        print("\n  No results generated.")
        return

    df = pd.DataFrame(results)

    # ================================================================
    # RESULTS SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for matcher in df["Matcher"].unique():
        sub = df[df["Matcher"] == matcher]
        auc = compute_auc(sub["Max_Error_deg"].values)
        print(f"\n  {matcher} ({len(sub)} pairs):")
        print(f"    Median total time:    {sub['Total_Time_ms'].median():.1f} ms")
        for thr in [5, 10, 20]:
            print(f"    Acc@{thr}:               {auc[f'Acc@{thr}']:.3f}")
        print(f"    CAPRI accept rate:    {sub['CAPRI_Accepted'].mean():.3f}")

    # ---- CAPRI Filtering Analysis ----
    print("\n" + "-" * 40)
    print("CAPRI FILTERING ANALYSIS")
    print("-" * 40)
    print("Does CAPRI rejection correlate with high pose error?")

    for matcher in df["Matcher"].unique():
        sub = df[df["Matcher"] == matcher]
        accepted = sub[sub["CAPRI_Accepted"] == True]
        rejected = sub[sub["CAPRI_Accepted"] == False]

        if len(accepted) > 0 and len(rejected) > 0:
            err_acc = accepted["Max_Error_deg"].median()
            err_rej = rejected["Max_Error_deg"].median()
            print(f"\n  {matcher}:")
            print(f"    CAPRI-accepted: median err = {err_acc:.1f} deg ({len(accepted)} pairs)")
            print(f"    CAPRI-rejected: median err = {err_rej:.1f} deg ({len(rejected)} pairs)")
            if err_rej > err_acc:
                print(f"    -> CAPRI successfully filters bad geometry!")
            else:
                print(f"    -> No clear benefit (may need threshold tuning)")
        elif len(accepted) == len(sub):
            print(f"\n  {matcher}: All pairs accepted by CAPRI")
        else:
            print(f"\n  {matcher}: All pairs rejected by CAPRI")

    # ---- AUC improvement from CAPRI filtering ----
    print("\n" + "-" * 40)
    print("AUC IMPROVEMENT FROM CAPRI FILTERING")
    print("-" * 40)

    for matcher in df["Matcher"].unique():
        sub = df[df["Matcher"] == matcher]
        accepted = sub[sub["CAPRI_Accepted"] == True]

        if 0 < len(accepted) < len(sub):
            auc_all = compute_auc(sub["Max_Error_deg"].values)
            auc_filt = compute_auc(accepted["Max_Error_deg"].values)

            print(f"\n  {matcher}:")
            for thr in [5, 10, 20]:
                key = f"Acc@{thr}"
                delta = auc_filt[key] - auc_all[key]
                print(f"    {key}: {auc_all[key]:.3f} -> {auc_filt[key]:.3f} "
                      f"({'+'if delta >= 0 else ''}{delta:.3f})")

    # ---- Save ----
    save_results(df, "real_images.csv", SEED,
                 "real-image experiment: Real-Image Baseline Benchmarking (with DUSt3R)")
    print(f"\n  Results saved to real_images.csv")


if __name__ == "__main__":
    run_experiment()
