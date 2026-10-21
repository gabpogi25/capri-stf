"""
ml_models/confidence_fusion/model.py

Confidence fusion classifier for six-point tuple acceptance.

Justification for ML approach:
    The algebraic conditioning score L(S) captures bracket-level numerical
    stability, and the multi-view consistency residual rho captures
    projective agreement across measurements. However, feature matchers
    (SIFT, SuperGlue) also produce descriptor-level confidence scores that
    carry complementary information about appearance repeatability and
    matching quality. These two signal types are not redundant: a tuple may
    be algebraically well-conditioned yet visually ambiguous, or vice versa.
    A small learned classifier combines these signals into a single
    acceptance probability while treating algebraic conditioning as a hard
    floor: no tuple rejected by the conditioning threshold is passed to
    this classifier.

Architecture:
    2-layer MLP, hidden dimension 64.
    Input (standard mode): [L(S) normalized, rho, descriptor_confidence] = 3-dim.
    Input (invariant mode): [L(S) normalized, rho, descriptor_confidence,
                             J1, J2, J3] = 6-dim.
    Output: Scalar acceptance probability in [0, 1].
    Parameters: < 10,000 in standard mode, < 12,000 in invariant mode.

    Invariant mode enables end-to-end training through the differentiable
    bracket/invariant computation (algorithms/invariants/six_point_torch.py),
    allowing the classifier to learn directly from the projective invariant
    gradients rather than relying solely on summary statistics.

Reference: GOAL.md Section 10 (ML Model Scale Targets).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class ConfidenceFusionClassifier(nn.Module):
    """
    Binary classifier combining algebraic and descriptor-based confidence.

    Input features (standard 3-feature mode):
        - conditioning_score_normalized: L(S) / L_threshold, clipped to [0, 1].
          Values near 0 indicate excellent conditioning; near 1 indicates marginal.
        - consistency_residual: rho = max-norm invariant disagreement.
        - descriptor_confidence: Matcher-provided confidence in [0, 1].

    Input features (invariant 6-feature mode):
        - All 3 features above, PLUS:
        - j1, j2, j3: Raw six-point projective invariant values.
        This mode enables gradient flow from the invariant computation
        through to the classifier loss, supporting end-to-end training.

    Output:
        Scalar in [0, 1]: estimated probability that the tuple is a valid,
        stable six-point projective correspondence.
    """

    def __init__(self, hidden_dim: int = 64, use_invariant_features: bool = False) -> None:
        super().__init__()
        self.use_invariant_features = use_invariant_features
        input_dim = 6 if use_invariant_features else 3

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,
        invariant_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (batch, 3) tensor with columns
                      [conditioning_normalized, rho, descriptor_confidence].
            invariant_features: Optional (batch, 3) tensor with columns
                                [J1, J2, J3]. Required if use_invariant_features=True.
                                Gradients from the invariant computation flow
                                through this input to the classifier loss.

        Returns:
            (batch, 1) acceptance probability tensor.
        """
        if self.use_invariant_features:
            assert invariant_features is not None, (
                "invariant_features required when use_invariant_features=True"
            )
            assert invariant_features.shape[-1] == 3, (
                f"Expected 3 invariant features, got {invariant_features.shape[-1]}"
            )
            assert features.shape[-1] == 3, (
                f"Expected 3 base features, got {features.shape[-1]}"
            )
            combined = torch.cat([features, invariant_features], dim=-1)
            return self.network(combined)
        else:
            assert features.shape[-1] == 3, (
                f"Expected 3 input features, got {features.shape[-1]}"
            )
            return self.network(features)

    def predict_numpy(
        self,
        conditioning_score: float,
        consistency_residual: float,
        descriptor_confidence: float,
        conditioning_threshold: float = 1000.0
    ) -> float:
        """
        Numpy-interface wrapper for single-tuple inference (standard mode).

        Args:
            conditioning_score: L(S) for this tuple.
            consistency_residual: rho for this tuple.
            descriptor_confidence: Matcher confidence in [0, 1].
            conditioning_threshold: L_threshold for normalization.

        Returns:
            Acceptance probability in [0, 1].
        """
        conditioning_normalized = min(
            1.0, conditioning_score / max(conditioning_threshold, 1e-9)
        )
        features = np.array(
            [[conditioning_normalized, consistency_residual, descriptor_confidence]],
            dtype=np.float32
        )
        tensor = torch.tensor(features)
        with torch.no_grad():
            prob = self.forward(tensor).item()
        return float(prob)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
