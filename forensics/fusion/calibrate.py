"""
forensics/fusion/calibrate.py

Fusion of the algebraic branch with an optional learned branch, probability
calibration and abstention.

Rules:
  * the fusion model is trained on the CALIBRATION split, whose generators are
    disjoint from both train and test;
  * isotonic calibration only with >= 1,000 calibration samples per class,
    otherwise Platt scaling;
  * if the algebraic branch abstains, the item is marked geometry-insufficient
    and only the learned branch (if any) decides; S_alg is never imputed;
  * fusion is kept only if it beats both single branches by a paired
    bootstrap (experiment E07); otherwise it is dropped and that is reported.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class Calibrator:
    method: str
    model: object

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray, min_per_class_isotonic: int = 1000) -> "Calibrator":
        scores, labels = np.asarray(scores, float), np.asarray(labels, int)
        if min(np.sum(labels == 0), np.sum(labels == 1)) >= min_per_class_isotonic:
            model = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
            return cls("isotonic", model)
        model = LogisticRegression().fit(scores.reshape(-1, 1), labels)
        return cls("platt", model)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, float)
        if self.method == "isotonic":
            return self.model.predict(scores)
        return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]


def fuse(algebraic_features: np.ndarray, abstain: np.ndarray, learned_score: np.ndarray | None,
         labels: np.ndarray) -> LogisticRegression:
    """
    Logistic-regression stacker on [algebraic features (0 where abstained), abstain flag, learned score].
    Train on the calibration split only.
    """
    X = _design(algebraic_features, abstain, learned_score)
    return LogisticRegression(max_iter=2000).fit(X, labels)


def fused_scores(model: LogisticRegression, algebraic_features: np.ndarray, abstain: np.ndarray,
                 learned_score: np.ndarray | None) -> np.ndarray:
    return model.predict_proba(_design(algebraic_features, abstain, learned_score))[:, 1]


def _design(algebraic_features, abstain, learned_score) -> np.ndarray:
    algebraic = np.where(np.asarray(abstain, bool)[:, None], 0.0, np.atleast_2d(np.asarray(algebraic_features, float)))
    columns = [algebraic, np.asarray(abstain, float)[:, None]]
    if learned_score is not None:
        columns.append(np.asarray(learned_score, float)[:, None])
    return np.hstack(columns)
