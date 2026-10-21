"""
experiments/run_fusion_metrics.py

neural Confidence Fusion evaluation report.

Input: experiments/results/neural_fusion_eval.csv, written by
ml_models/confidence_fusion/train_neural.py (validation split = 10% of the
5,000 generated samples, i.e. 500 samples; seed 42).

Outputs:
    fusion_metrics.csv           class balance, confusion matrix, precision,
                                      recall, F1, balanced accuracy, MCC, ROC AUC,
                                      average precision, and the same metrics for
                                      the majority-class baseline
    fusion_threshold_sweep.csv   the same metrics at thresholds 0.00..1.00

This script reads an existing evaluation file and needs no PyTorch.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_utils import RESULTS_DIR, save_results

EVAL_FILE = RESULTS_DIR / "neural_fusion_eval.csv"
TRAINING_SEED = 42


def binary_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "Balanced_Accuracy": balanced_accuracy_score(labels, predictions),
        "MCC": matthews_corrcoef(labels, predictions),
    }


def run_experiment() -> pd.DataFrame:
    df = pd.read_csv(EVAL_FILE, comment="#")
    labels = df["True_Label"].to_numpy(dtype=int)
    probabilities = df["Probability"].to_numpy(dtype=float)
    positive_fraction = float(labels.mean())
    majority_class = int(positive_fraction >= 0.5)

    model = binary_metrics(labels, (probabilities >= 0.5).astype(int))
    majority = binary_metrics(labels, np.full_like(labels, majority_class))
    rows = [
        {"Classifier": "Neural Confidence Fusion (threshold 0.5)", **model,
         "ROC_AUC": roc_auc_score(labels, probabilities),
         "Average_Precision": average_precision_score(labels, probabilities)},
        {"Classifier": f"Majority class (always {majority_class})", **majority,
         "ROC_AUC": 0.5, "Average_Precision": positive_fraction},
    ]
    summary = pd.DataFrame(rows)
    summary.insert(1, "N", len(labels))
    summary.insert(2, "Positive_Fraction", positive_fraction)
    summary.insert(3, "Probability_Min", float(probabilities.min()))
    summary.insert(4, "Probability_Max", float(probabilities.max()))

    sweep = pd.DataFrame([
        {"Threshold": threshold, **binary_metrics(labels, (probabilities >= threshold).astype(int))}
        for threshold in np.round(np.linspace(0.0, 1.0, 101), 2)
    ])

    print(summary.T.to_string(header=False))
    best = sweep.loc[sweep["MCC"].idxmax()]
    print(f"\nBest MCC over thresholds: {best['MCC']:.3f} at threshold {best['Threshold']:.2f}")
    save_results(summary, "fusion_metrics.csv", TRAINING_SEED,
                 "fusion experiment: neural Confidence Fusion metrics vs majority-class baseline")
    save_results(sweep, "fusion_threshold_sweep.csv", TRAINING_SEED,
                 "fusion experiment: neural Confidence Fusion threshold sweep")
    return summary


if __name__ == "__main__":
    run_experiment()
