"""
ml_models/confidence_fusion/features.py

Single source of truth for the neural Confidence Fusion input features.

Training (train_neural.py) and inference (pipeline/main_pipeline.py) both call
build_fusion_features_neural(), so the network always sees the features it was
trained on (the same conditioning feature min(1, L / threshold) in both).

Feature vector (3 values, all derived from the six 3D points of one tuple):
    cond_norm        = min(1, L(S) / conditioning_threshold)
    rho              = Plucker residual rho(S) of the 15 brackets
    bracket_strength = clip((log10(max(delta_min, 1e-20)) + 20) / 20, 0, 1)

The label-leak model (train.py, weights confidence_fusion_best.pt) uses a third
feature, desc_conf, drawn from U(0.7, 1.0) for positives and U(0, 0.6) for
negatives. That feature encodes the label, so the label-leak model's near-perfect scores are a
label leak. The label-leak model is not used for any reported result; only the neural model is loaded.

Note on the neural model's measured performance: on its own validation split the neural model
predicts the positive class for every sample (see
experiments/run_fusion_metrics.py). It is therefore OFF by default in the
pipeline and must be enabled explicitly.

This module imports torch only inside load_fusion_model_neural(), so the rest of
CAPRI runs without PyTorch installed.
"""

from __future__ import annotations
from itertools import combinations
from pathlib import Path
import numpy as np

from algorithms.conditioning.scores import compute_conditioning_score
from algorithms.invariants.brackets import compute_all_brackets_for_tuple, compute_bracket
from algorithms.syzygy.residual import compute_syzygy_residual

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
NEURAL_WEIGHTS_FILENAME = "confidence_fusion_neural_best.pt"


def compute_bracket_signal_strength(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, ...],
) -> float:
    """log10 of the smallest normalized bracket, mapped to [0, 1]."""
    min_bracket = min(
        abs(compute_bracket(points_homogeneous, quad)) for quad in combinations(tuple_indices, 4)
    )
    log_val = np.log10(max(min_bracket, 1e-20))
    return float(np.clip((log_val + 20.0) / 20.0, 0.0, 1.0))


def build_fusion_features_neural(
    points_homogeneous: np.ndarray,
    tuple_indices: tuple[int, ...],
    conditioning_threshold: float = 1000.0,
) -> list[float]:
    """Return [cond_norm, rho, bracket_strength] for one six-point tuple."""
    l_s = compute_conditioning_score(points_homogeneous, tuple_indices)
    rho = compute_syzygy_residual(compute_all_brackets_for_tuple(points_homogeneous, tuple_indices))
    cond_norm = min(1.0, l_s / conditioning_threshold) if np.isfinite(l_s) else 1.0
    return [float(cond_norm), float(rho), compute_bracket_signal_strength(points_homogeneous, tuple_indices)]


def load_fusion_model_neural(device: str = "cpu"):
    """Load the neural Confidence Fusion network. Requires PyTorch."""
    import torch  # noqa: PLC0415 (optional dependency)
    from ml_models.confidence_fusion.model import ConfidenceFusionClassifier  # noqa: PLC0415

    weights_path = WEIGHTS_DIR / NEURAL_WEIGHTS_FILENAME
    if not weights_path.exists():
        raise FileNotFoundError(
            f"{weights_path} not found. Train it with: python ml_models/confidence_fusion/train_neural.py"
        )
    model = ConfidenceFusionClassifier(hidden_dim=64, use_invariant_features=False).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict_fusion_probability(model, features: list[float], device: str = "cpu") -> float:
    """Probability that a tuple is valid according to the neural network."""
    import torch  # noqa: PLC0415

    with torch.no_grad():
        tensor = torch.tensor([features], dtype=torch.float32, device=device)
        return float(model(tensor).item())
