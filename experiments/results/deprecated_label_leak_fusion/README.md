# Deprecated: first Confidence Fusion model (label leak)

These files come from `ml_models/confidence_fusion/train.py` (label-leak model). Its third
input feature, `desc_conf`, was drawn from U(0.7, 1.0) for valid samples and
U(0, 0.6) for invalid ones, so the feature encoded the label. The near-perfect
AUC (0.99999) and F1 in these files measure that leak, not the method.

The weights of this model were moved to
`ml_models/weights/deprecated/confidence_fusion_label_leak.pt`. No script
loads them any more (the synthetic baselines and the COLMAP automator used to).

Use the neural fusion (`neural_fusion_eval.csv`, `neural_fusion_network.csv`) and
the report written by `experiments/run_fusion_metrics.py`.
