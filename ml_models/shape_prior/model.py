"""
ml_models/shape_prior/model.py

Shape prior network for sparse-view projective configuration completion.

Justification for ML approach:
    When fewer than 6 well-conditioned correspondences are available,
    the six-point invariant system is underdetermined. A closed-form
    algebraic completion does not exist without additional constraints.
    A learned category-level prior encodes plausible geometric structure
    for a known object class, used only to select among configurations
    consistent with observed invariants. It is subordinate to algebraic
    evidence: any completion that fails bracket conditioning or the
    consistency check is rejected regardless of prior confidence.

Architecture:
    3-layer MLP with hidden dimension 128. Input is the observed partial
    six-point configuration (concatenated homogeneous coordinates of the
    available points, zero-padded for missing points). Output is the
    completed configuration. Parameters < 50,000 (target from GOAL.md).

Reference: GOAL.md Section 10 (ML Model Scale Targets).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class ShapePriorNetwork(nn.Module):
    """
    MLP-based shape completion network for partial six-point configurations.

    Input:  (batch, 24) - six homogeneous 4-vectors concatenated.
            Missing points are represented as zero vectors.
    Output: (batch, 24) - completed six-point configuration.
    """

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        input_dim = 6 * 4   # six homogeneous 4-vectors
        output_dim = 6 * 4

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, partial_config: torch.Tensor) -> torch.Tensor:
        """
        Args:
            partial_config: (batch, 24) tensor. Missing points encoded as zeros.

        Returns:
            (batch, 24) completed configuration tensor.
        """
        assert partial_config.shape[-1] == 24, (
            f"Expected 24-dim input (6 points x 4 coords), got {partial_config.shape[-1]}"
        )
        return self.network(partial_config)

    def predict_numpy(self, partial_config_np: np.ndarray) -> np.ndarray:
        """
        Numpy-interface wrapper for inference.

        Args:
            partial_config_np: (24,) or (batch, 24) array.

        Returns:
            (24,) or (batch, 24) completed configuration.
        """
        squeeze = partial_config_np.ndim == 1
        if squeeze:
            partial_config_np = partial_config_np[np.newaxis, :]
        tensor = torch.tensor(partial_config_np, dtype=torch.float32)
        with torch.no_grad():
            out = self.forward(tensor).numpy()
        return out[0] if squeeze else out

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)