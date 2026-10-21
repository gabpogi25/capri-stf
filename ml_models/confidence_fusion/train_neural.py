"""
ml_models/confidence_fusion/train_neural.py

Confidence fusion classifier (neural):
  - Labels come from simulated reconstruction error (angular pose recovery)
    rather than from the input features themselves.
  - The third feature is bracket_signal_strength, a geometric quantity (log of
    minimum bracket magnitude), not the label-derived `desc_conf` of the label-leak model.
  - So the labels are not circular, unlike the label-leak model (perfect F1/AUC by label leak).

Run:
    python ml_models/confidence_fusion/train_neural.py --epochs 30 --seed 42 --n_samples 5000
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm
import sys
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import f1_score, roc_auc_score
from ml_models.confidence_fusion.model import ConfidenceFusionClassifier
from synthetic_data.generator import (
    generate_generic_configuration,
    generate_near_degenerate_configuration,
    generate_mismatched_configuration,
)
from algorithms.conditioning.scores import compute_conditioning_score
from algorithms.invariants.brackets import compute_all_brackets_for_tuple, compute_bracket
from algorithms.syzygy.residual import compute_syzygy_residual
from itertools import combinations

WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "ml_models" / "weights"


def _sample_tuple(n_points: int, rng: np.random.Generator) -> tuple:
    return tuple(int(x) for x in rng.choice(n_points, size=6, replace=False))


from ml_models.confidence_fusion.features import compute_bracket_signal_strength as _compute_bracket_signal_strength


def _simulate_reconstruction_label(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple,
    rng: np.random.Generator,
    noise_sigma: float = 0.1,
    angular_threshold_deg: float = 15.0,
) -> int:
    """
    Generate a ground-truth label based on simulated reconstruction quality.
    
    Procedure:
      1. Generate a random camera pair (two projection matrices).
      2. Project the 6 tuple points into both cameras (with noise).
      3. Estimate the fundamental matrix from the noisy 2D correspondences.
      4. Recover the relative pose from the fundamental matrix.
      5. Compare recovered pose to ground truth.
      6. Label = 1 if angular error < threshold, else 0.
    
    This creates labels INDEPENDENT of the input features.
    """
    pts = points_homogeneous[:, list(tuple_indices)].copy()  # (4, 6)
    pts[2, :] += 3.0  # Move points in front of cameras (z in [2, 4])
    
    # Generate 4 extra 3D points by perturbing the existing ones
    # This guarantees they represent valid 3D geometry and satisfy epipolar constraints
    extra_pts = pts[:, :4].copy()
    extra_pts[:3, :] += rng.standard_normal((3, 4)) * 0.1
    all_pts_3d = np.hstack([pts, extra_pts])  # (4, 10)
    
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    
    # Random rotation via axis-angle
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis) + 1e-10
    angle = rng.uniform(0.1, 0.8)
    K_skew = np.array([[0, -axis[2], axis[1]],
                        [axis[2], 0, -axis[0]],
                        [-axis[1], axis[0], 0]])
    R_gt = np.eye(3) + np.sin(angle) * K_skew + (1 - np.cos(angle)) * (K_skew @ K_skew)
    t_gt = rng.standard_normal(3)
    t_gt = t_gt / (np.linalg.norm(t_gt) + 1e-10)
    
    P2 = K @ np.hstack([R_gt, t_gt.reshape(3, 1)])
    
    proj1 = P1 @ all_pts_3d
    proj2 = P2 @ all_pts_3d
    
    if np.any(proj1[2, :] <= 0) or np.any(proj2[2, :] <= 0):
        return 0
    
    pts2d_1 = (proj1[:2, :] / proj1[2:, :]).T
    pts2d_2 = (proj2[:2, :] / proj2[2:, :]).T
    
    pts2d_1_noisy = pts2d_1 + rng.standard_normal(pts2d_1.shape) * noise_sigma
    pts2d_2_noisy = pts2d_2 + rng.standard_normal(pts2d_2.shape) * noise_sigma
    
    try:
        F, mask = cv2.findFundamentalMat(
            pts2d_1_noisy.astype(np.float64), pts2d_2_noisy.astype(np.float64), cv2.FM_8POINT)
        if F is None or F.shape != (3, 3):
            return 0
        
        E = K.T @ F @ K
        n_inliers, R_est, t_est, _ = cv2.recoverPose(
            E, pts2d_1_noisy.astype(np.float64), pts2d_2_noisy.astype(np.float64), K)
        
        if R_est is None or t_est is None:
            return 0
        
        R_err = R_est @ R_gt.T
        cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rot_err_deg = np.degrees(np.arccos(cos_angle))
        
        t_est_flat = t_est.flatten()
        t_gt_norm = np.linalg.norm(t_gt)
        if t_gt_norm > 1e-10:
            t_gt_unit = t_gt / t_gt_norm
            cos_t = np.clip(np.dot(t_est_flat, t_gt_unit), -1.0, 1.0)
            trans_err_deg = np.degrees(np.arccos(abs(cos_t)))
        else:
            trans_err_deg = 0.0
        
        max_err = max(rot_err_deg, trans_err_deg)
        return 1 if max_err < angular_threshold_deg else 0
        
    except Exception:
        return 0


def build_fusion_dataset_neural(
    n_samples: int,
    n_points: int,
    seed: int,
    conditioning_threshold: float = 1000.0,
    noise_sigma: float = 0.1,
    angular_threshold_deg: float = 15.0,
) -> tuple:
    # Build (features, labels) with reconstruction-error-based labels.
    rng = np.random.default_rng(seed)
    features_list = []
    labels_list = []
    
    generators = [
        ("generic", lambda: generate_generic_configuration(n_points, scale=1.0, rng=rng)),
        ("near_degen", lambda: generate_near_degenerate_configuration(
            n_points, coplanar_fraction=rng.uniform(0.3, 0.8), rng=rng)),
        ("mismatched", lambda: generate_mismatched_configuration(n_points, rng=rng)[1]),
    ]
    
    with tqdm(total=n_samples, desc="Building dataset") as pbar:
        attempts = 0
        max_attempts = n_samples * 10
        while len(features_list) < n_samples and attempts < max_attempts:
            attempts += 1
            gen_name, gen_func = generators[rng.integers(len(generators))]
            try:
                pts = gen_func()
            except Exception:
                continue
            
            t = _sample_tuple(n_points, rng)
            
            try:
                l_s = compute_conditioning_score(pts, t)
                brackets = compute_all_brackets_for_tuple(pts, t)
                rho = compute_syzygy_residual(brackets)
                bracket_strength = _compute_bracket_signal_strength(pts, t)
            except Exception:
                continue
            
            label = _simulate_reconstruction_label(
                pts, t, rng, noise_sigma=noise_sigma,
                angular_threshold_deg=angular_threshold_deg)
            
            cond_norm = min(1.0, l_s / conditioning_threshold)
            features_list.append([cond_norm, rho, bracket_strength])
            labels_list.append(float(label))
            pbar.update(1)
    
    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    
    print(f"Dataset: {len(features)} samples, "
          f"{labels.sum():.0f} positive ({labels.mean()*100:.1f}%), "
          f"{len(labels) - labels.sum():.0f} negative ({(1-labels.mean())*100:.1f}%)")
    
    return features, labels


class FusionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features)
        self.labels = torch.tensor(labels).unsqueeze(1)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_confidence_fusion_neural(
    n_samples=10000, n_points=12, epochs=30, batch_size=256,
    lr=1e-3, seed=42, hidden_dim=64, conditioning_threshold=1000.0,
    noise_sigma=0.1, angular_threshold_deg=15.0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    features, labels = build_fusion_dataset_neural(
        n_samples, n_points, seed, conditioning_threshold, noise_sigma, angular_threshold_deg)
    dataset = FusionDataset(features, labels)
    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = ConfidenceFusionClassifier(hidden_dim=hidden_dim).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for feat, lab in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            feat, lab = feat.to(device), lab.to(device)
            pred = model(feat)
            loss = criterion(pred, lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, all_preds, all_labs = [], [], []
        with torch.no_grad():
            for feat, lab in val_loader:
                feat, lab = feat.to(device), lab.to(device)
                pred = model(feat)
                val_losses.append(criterion(pred, lab).item())
                all_preds.extend(pred.cpu().numpy().flatten())
                all_labs.extend(lab.cpu().numpy().flatten())

        val_loss = float(np.mean(val_losses))
        train_loss = float(np.mean(train_losses))
        all_preds = np.array(all_preds)
        all_labs = np.array(all_labs)
        all_preds_binary = (all_preds >= 0.5).astype(float)
        
        try:
            val_f1 = f1_score(all_labs, all_preds_binary, zero_division=0)
            val_auc = roc_auc_score(all_labs, all_preds)
        except ValueError:
            val_f1, val_auc = 0.0, 0.5

        history.append({"Epoch": epoch, "Train_BCE": train_loss, "Val_BCE": val_loss,
                        "Val_F1": float(val_f1), "Val_AUC": float(val_auc)})
        print(f"Epoch {epoch:3d} | train BCE: {train_loss:.5f} | "
              f"val BCE: {val_loss:.5f} | val F1: {val_f1:.3f} | val AUC: {val_auc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_DIR / "confidence_fusion_neural_best.pt")

    print(f"\nTraining complete. Best val BCE: {best_val_loss:.5f}")
    
    RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "experiments" / "results"
    import pandas as pd
    pd.DataFrame(history).to_csv(RESULTS_DIR / "neural_fusion_network.csv", index=False)

    # Final evaluation
    model.load_state_dict(torch.load(WEIGHTS_DIR / "confidence_fusion_neural_best.pt", weights_only=True))
    model.eval()
    eval_data = []
    with torch.no_grad():
        for feat, lab in val_loader:
            feat, lab = feat.to(device), lab.to(device)
            pred = model(feat)
            for i in range(len(pred)):
                eval_data.append({
                    "Sample_Idx": len(eval_data),
                    "Probability": float(pred[i].item()),
                    "True_Label": int(lab[i].item())
                })
    df_eval = pd.DataFrame(eval_data)
    df_eval.to_csv(RESULTS_DIR / "neural_fusion_eval.csv", index=False)
    
    preds_binary = (df_eval["Probability"] >= 0.5).astype(int)
    tp = ((preds_binary == 1) & (df_eval["True_Label"] == 1)).sum()
    fp = ((preds_binary == 1) & (df_eval["True_Label"] == 0)).sum()
    fn = ((preds_binary == 0) & (df_eval["True_Label"] == 1)).sum()
    tn = ((preds_binary == 0) & (df_eval["True_Label"] == 0)).sum()
    total = tp + fp + fn + tn
    print(f"\nConfusion Matrix (n={total}):")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    print(f"  Accuracy: {(tp+tn)/total:.4f}")
    f1_final = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0
    print(f"  F1: {f1_final:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--n_points", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--conditioning_threshold", type=float, default=1000.0)
    parser.add_argument("--noise_sigma", type=float, default=0.1)
    parser.add_argument("--angular_threshold", type=float, default=15.0)
    args = parser.parse_args()

    train_confidence_fusion_neural(
        n_samples=args.n_samples, n_points=args.n_points,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, seed=args.seed, hidden_dim=args.hidden_dim,
        conditioning_threshold=args.conditioning_threshold,
        noise_sigma=args.noise_sigma, angular_threshold_deg=args.angular_threshold,
    )
