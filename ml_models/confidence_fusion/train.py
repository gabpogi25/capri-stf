"""
Not for reported results (label leak). The third feature desc_conf is sampled
from U(0.7, 1.0) for positives and U(0, 0.6) for negatives, so it encodes the
label and the resulting scores are meaningless. This script shows the label
leak; use train_neural.py. The script refuses to run unless --allow-label-leak is
given.

ml_models/confidence_fusion/train.py

Training script for the confidence fusion classifier.

Generates training samples from synthetic configurations by computing:
  - L(S): conditioning score for each tuple
  - rho: multi-view consistency residual (label-leak model: self-consistency via perturbation)
  - descriptor_confidence: simulated as high for clean, low for mismatched
  - label: 1 for valid (generic, small residual), 0 for invalid

Run:
    python ml_models/confidence_fusion/train.py --epochs 30 --seed 42
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from ml_models.confidence_fusion.model import ConfidenceFusionClassifier
from synthetic_data.generator import (
    generate_generic_configuration,
    generate_near_degenerate_configuration,
    generate_mismatched_configuration,
)
from algorithms.conditioning.scores import compute_conditioning_score
from algorithms.invariants.brackets import compute_all_brackets_for_tuple
from algorithms.syzygy.residual import compute_syzygy_residual

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"


def _sample_tuple(n_points: int, rng: np.random.Generator) -> tuple:
    return tuple(int(x) for x in rng.choice(n_points, size=6, replace=False))


def _compute_self_consistency_rho(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple,
    rng: np.random.Generator,
    perturbation_eps: float = 1e-4
) -> float:
    brackets = compute_all_brackets_for_tuple(points_homogeneous, tuple_indices)
    return compute_syzygy_residual(brackets)


def build_fusion_dataset(
    n_samples: int,
    n_points: int,
    seed: int,
    conditioning_threshold: float = 1000.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (features, labels) arrays for confidence fusion training.

    Each sample is one tuple from one synthetic configuration.
    Labels: 1 = valid (generic + low residual), 0 = invalid.

    Returns:
        features: (N, 3) float32 array
        labels:   (N,)   float32 array of 0/1
    """
    rng = np.random.default_rng(seed)
    features_list = []
    labels_list = []

    n_per_class = n_samples // 2

    # Positive samples: generic configurations, well-conditioned tuples
    with tqdm(total=n_per_class, desc="Generating positive samples") as pbar:
        while len(features_list) < n_per_class:
            pts = generate_generic_configuration(n_points, scale=1.0, rng=rng)
            t = _sample_tuple(n_points, rng)
            l_s = compute_conditioning_score(pts, t)
            if l_s >= conditioning_threshold:
                continue
            rho = _compute_self_consistency_rho(pts, t, rng)
            if rho >= 99.0:
                continue
            desc_conf = float(rng.uniform(0.7, 1.0))
            cond_norm = min(1.0, l_s / conditioning_threshold)
            features_list.append([cond_norm, rho, desc_conf])
            labels_list.append(1.0)
            pbar.update(1)

    # Negative samples: mismatched or near-degenerate
    with tqdm(total=n_per_class, desc="Generating negative samples") as pbar:
        while len(features_list) < 2 * n_per_class:
            if rng.random() < 0.5:
                pts_orig, pts, _ = generate_mismatched_configuration(n_points, rng=rng)
            else:
                pts = generate_near_degenerate_configuration(
                    n_points, coplanar_fraction=rng.uniform(0.5, 0.9), rng=rng
                )
            t = _sample_tuple(n_points, rng)
            l_s = compute_conditioning_score(pts, t)
            rho = _compute_self_consistency_rho(pts, t, rng)
            desc_conf = float(rng.uniform(0.0, 0.6))
            cond_norm = min(1.0, l_s / max(conditioning_threshold, 1e-9))
            features_list.append([cond_norm, rho, desc_conf])
            labels_list.append(0.0)
            pbar.update(1)

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    return features, labels


class FusionDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.tensor(features)
        self.labels = torch.tensor(labels).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def train_confidence_fusion(
    n_samples: int = 20000,
    n_points: int = 12,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    hidden_dim: int = 64,
    conditioning_threshold: float = 1000.0,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    features, labels = build_fusion_dataset(
        n_samples, n_points, seed, conditioning_threshold
    )
    dataset = FusionDataset(features, labels)
    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

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
        val_losses = []
        all_preds = []
        all_labs = []
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

        history.append({"Epoch": epoch, "Train_BCE": train_loss, "Val_BCE": val_loss, "Val_F1": float(val_f1), "Val_AUC": float(val_auc)})
        print(
            f"Epoch {epoch:3d} | "
            f"train BCE: {train_loss:.5f} | "
            f"val BCE: {val_loss:.5f} | "
            f"val F1: {val_f1:.3f} | "
            f"val AUC: {val_auc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_DIR / "confidence_fusion_best.pt")

    print(f"Training complete. Best val BCE: {best_val_loss:.5f}")
    print(f"Weights saved to {WEIGHTS_DIR / 'confidence_fusion_best.pt'}")
    
    RESULTS_DIR = Path(__file__).parent.parent.parent / "experiments" / "results"
    import pandas as pd
    pd.DataFrame(history).to_csv(RESULTS_DIR / "label_leak_fusion_network.csv", index=False)

    print("Running evaluation pass on validation set to save per-sample metrics...")
    model.load_state_dict(torch.load(WEIGHTS_DIR / "confidence_fusion_best.pt", weights_only=True))
    model.eval()
    eval_data = []
    
    with torch.no_grad():
        for feat, lab in val_loader:
            feat, lab = feat.to(device), lab.to(device)
            pred = model(feat)
            pred_np = pred.cpu().numpy().flatten()
            lab_np = lab.cpu().numpy().flatten()
            
            for i in range(len(pred_np)):
                eval_data.append({
                    "Sample_Idx": len(eval_data),
                    "Probability": float(pred_np[i]),
                    "True_Label": int(lab_np[i])
                })
                
    pd.DataFrame(eval_data).to_csv(RESULTS_DIR / "label_leak_fusion_eval.csv", index=False)
    print("Evaluation data saved.")


if __name__ == "__main__":
    if "--allow-label-leak" not in sys.argv:
        sys.exit("train.py (label-leak model) has a label leak; use train_neural.py. Pass --allow-label-leak to run it anyway.")
    sys.argv.remove("--allow-label-leak")
    parser = argparse.ArgumentParser(description="Train CAPRI confidence fusion classifier.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--n_points", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--conditioning_threshold", type=float, default=1000.0)
    args = parser.parse_args()

    train_confidence_fusion(
        n_samples=args.n_samples,
        n_points=args.n_points,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        conditioning_threshold=args.conditioning_threshold,
    )