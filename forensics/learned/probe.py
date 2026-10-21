"""
forensics/learned/probe.py

Learned baseline branch (F2): frozen image-encoder
features + L2-regularized logistic regression, trained on the TRAINING
generators only. This branch is separate from the CAPRI mathematics; it is a
baseline and a fusion partner, and its benefit must be shown by ablation (E07).

Requires PyTorch and open_clip (`pip install torch open_clip_torch`); verify
the chosen weights' licence before use. Nothing here runs without them.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def load_encoder(model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
    try:
        import open_clip
        import torch  # noqa: F401
    except ImportError as error:
        raise ImportError("The learned probe needs torch and open_clip_torch installed.") from error
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess


def image_features(paths: list[str | Path], model, preprocess, device: str = "cpu", batch_size: int = 32) -> np.ndarray:
    import torch
    from PIL import Image

    features = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[start:start + batch_size]])
            encoded = model.encode_image(batch.to(device)).float()
            features.append(torch.nn.functional.normalize(encoded, dim=-1).cpu().numpy())
    return np.concatenate(features) if features else np.zeros((0, 0))


def train_probe(features: np.ndarray, labels: np.ndarray, groups: np.ndarray | None = None, seed: int = 0):
    """Logistic regression with C chosen by 5-fold CV on the training split."""
    search = GridSearchCV(LogisticRegression(max_iter=5000, random_state=seed),
                          {"C": [0.01, 0.1, 1.0, 10.0]}, cv=5, scoring="roc_auc")
    search.fit(features, labels, groups=groups)
    return search.best_estimator_
