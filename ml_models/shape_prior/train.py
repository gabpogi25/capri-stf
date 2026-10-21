"""
ml_models/shape_prior/train.py

Training script for the shape prior network.

Generates training pairs from synthetic sparse-view configurations:
  - Input:  partial six-point configuration (some points zeroed out)
  - Target: complete six-point configuration

The network is trained to complete missing points consistent with
observed geometry. After training, the model is validated by checking
that completions pass bracket conditioning and consistency checks.

Run:
    python ml_models/shape_prior/train.py --epochs 50 --seed 42
"""

from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from tqdm import tqdm

from ml_models.shape_prior.model import ShapePriorNetwork
from synthetic_data.generator import generate_generic_configuration

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"


class SparseViewDataset(Dataset):
    """
    Dataset of (partial_config, full_config) pairs for shape prior training.

    Each sample is a six-point homogeneous configuration in P^3.
    The partial input has 1 or 2 points zeroed out to simulate missing views.
    """

    def __init__(self, n_samples: int, n_points: int, seed: int) -> None:
        self.n_samples = n_samples
        self.n_points = n_points
        rng = np.random.default_rng(seed)

        self.full_configs: list[np.ndarray] = []
        self.partial_configs: list[np.ndarray] = []

        for _ in range(n_samples):
            pts = generate_generic_configuration(n_points, scale=1.0, rng=rng)
            # Select 6 points for the tuple
            idx = rng.choice(n_points, size=6, replace=False)
            full = pts[:, idx].T.flatten()  # (24,)

            # Zero out 1 or 2 points to simulate missing correspondences
            partial = full.copy()
            n_missing = int(rng.integers(1, 3))
            missing_idx = rng.choice(6, size=n_missing, replace=False)
            for m in missing_idx:
                partial[m*4:(m+1)*4] = 0.0

            self.full_configs.append(full.astype(np.float32))
            self.partial_configs.append(partial.astype(np.float32))

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.partial_configs[idx]),
            torch.tensor(self.full_configs[idx]),
        )


def train_shape_prior(
    n_samples: int = 50000,
    n_points: int = 12,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    hidden_dim: int = 128,
) -> None:
    """
    Train the shape prior network and save weights to ml_models/weights/.

    Args:
        n_samples: Total training samples.
        n_points: Points per synthetic configuration.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Adam learning rate.
        seed: Random seed for data generation and training.
        hidden_dim: Network hidden dimension.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    dataset = SparseViewDataset(n_samples=n_samples, n_points=n_points, seed=seed)
    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = ShapePriorNetwork(hidden_dim=hidden_dim).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for partial, full in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            partial, full = partial.to(device), full.to(device)
            pred = model(partial)
            loss = criterion(pred, full)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for partial, full in val_loader:
                partial, full = partial.to(device), full.to(device)
                pred = model(partial)
                val_losses.append(criterion(pred, full).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"Epoch": epoch, "Train_Loss": train_loss, "Val_Loss": val_loss})
        print(f"Epoch {epoch:3d} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_DIR / "shape_prior_best.pt")

    print(f"Training complete. Best val MSE: {best_val_loss:.6f}")
    print(f"Weights saved to {WEIGHTS_DIR / 'shape_prior_best.pt'}")
    
    RESULTS_DIR = Path(__file__).parent.parent.parent / "experiments" / "results"
    import pandas as pd
    pd.DataFrame(history).to_csv(RESULTS_DIR / "shape_prior.csv", index=False)

    print("Running evaluation pass on validation set to save per-sample metrics...")
    model.load_state_dict(torch.load(WEIGHTS_DIR / "shape_prior_best.pt", weights_only=True))
    model.eval()
    
    eval_hist_data = []
    eval_scatter_data = []
    
    with torch.no_grad():
        for partial, full in val_loader:
            partial, full = partial.to(device), full.to(device)
            pred = model(partial)
            mse_per_sample = torch.mean((pred - full)**2, dim=1).cpu().numpy()
            pred_np = pred.cpu().numpy()
            full_np = full.cpu().numpy()
            
            for i in range(len(mse_per_sample)):
                sample_idx = len(eval_hist_data)
                eval_hist_data.append({"Sample_Idx": sample_idx, "MSE_Error": float(mse_per_sample[i])})
                for j in range(24):
                    eval_scatter_data.append({
                        "Sample_Idx": sample_idx,
                        "Coord_Idx": j,
                        "True_Val": float(full_np[i, j]),
                        "Pred_Val": float(pred_np[i, j])
                    })

    pd.DataFrame(eval_hist_data).to_csv(RESULTS_DIR / "shape_prior_eval_hist.csv", index=False)
    pd.DataFrame(eval_scatter_data).to_csv(RESULTS_DIR / "shape_prior_eval_scatter.csv", index=False)
    print("Evaluation data saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CAPRI shape prior network.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--n_points", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=128)
    args = parser.parse_args()

    train_shape_prior(
        n_samples=args.n_samples,
        n_points=args.n_points,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
    )