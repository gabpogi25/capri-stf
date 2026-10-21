"""Known-answer tests for forensics.eval.metrics and forensics.stats."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from forensics.eval import metrics
from forensics.io.manifests import ManifestError, load_manifest, write_manifest
from forensics.stats.cusum import calibrate_h, cusum
from forensics.stats.empirical_null import EmpiricalNull, rejection_rate
from forensics.stats.fisher import block_fisher, effective_sample_size, fisher_combine
from forensics.stats.nulls import chi2_statistic


def test_auc_delong_separable_and_random():
    y = np.r_[np.zeros(50), np.ones(50)]
    assert metrics.auc_delong(y, np.r_[np.zeros(50), np.ones(50)])["auc"] == 1.0
    rng = np.random.default_rng(0)
    s = rng.standard_normal(100)
    result = metrics.auc_delong(y, s)
    assert abs(result["auc"] - roc_auc_score(y, s)) < 1e-12
    assert result["ci_lo"] < 0.5 + 0.2 and result["ci_hi"] > 0.5 - 0.2


def test_tpr_at_fpr_uses_negatives_only():
    y = np.r_[np.zeros(100), np.ones(100)]
    s = np.r_[np.arange(100), np.arange(100) + 1000]
    result = metrics.tpr_at_fpr(y, s, 0.05)
    assert result["fpr"] <= 0.05 and result["tpr"] == 1.0


def test_classification_report_majority_baseline():
    y = np.r_[np.zeros(5), np.ones(95)]
    report = metrics.classification_report(y, np.ones(100))
    assert report["model"]["mcc"] == 0.0
    assert report["model"]["balanced_accuracy"] == 0.5
    assert report["majority_baseline"]["f1"] == report["model"]["f1"]


def test_reliability_omits_small_bins():
    rng = np.random.default_rng(1)
    p = np.r_[rng.uniform(0, 0.1, 500), [0.95]]
    y = (rng.uniform(size=501) < p).astype(int)
    result = metrics.reliability(y, p, bins=10, min_n=50)
    assert len(result["omitted_bins"]) == 1 and result["rows"][0]["n"] == 500


def test_flip_rate_segment_f1_holm():
    assert metrics.flip_rate([0, 0, 1, 1, 0]) == 0.5
    assert metrics.segment_f1([(1.0, 2.0)], [(1.5, 3.0)], tol=1.0) == 1.0
    assert metrics.holm_correction([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_cluster_and_paired_bootstrap():
    rng = np.random.default_rng(2)
    y = np.tile([0, 1], 100)
    good = y + rng.normal(0, 0.5, 200)
    bad = rng.normal(0, 1, 200)
    clusters = np.repeat(np.arange(40), 5)
    auc = lambda yy, ss: roc_auc_score(yy, ss)  # noqa: E731
    assert metrics.cluster_bootstrap(auc, y, good, clusters, n_boot=300)["ci_lo"] > 0.5
    assert metrics.paired_bootstrap_diff(auc, y, good, bad, clusters, n_boot=300)["excludes_zero"]


def test_chi2_statistic_matches_definition():
    T, dof, p = chi2_statistic(np.array([1.0, 2.0]), np.diag([1.0, 4.0]))
    assert T == pytest.approx(2.0) and dof == 2 and p == pytest.approx(np.exp(-1.0))


def test_fisher_and_effective_sample_size():
    stat, p = fisher_combine(np.full(10, 0.5))
    assert stat == pytest.approx(-20 * np.log(0.5)) and 0.5 < p < 1.0
    rng = np.random.default_rng(3)
    assert effective_sample_size(rng.standard_normal(500)) > 400
    correlated = np.repeat(rng.standard_normal(50), 10)
    assert effective_sample_size(correlated) < 200
    _, p_block, n_eff = block_fisher(rng.uniform(size=200))
    assert 0 < p_block <= 1 and n_eff <= 200


def test_empirical_null_is_calibrated_on_its_own_distribution():
    rng = np.random.default_rng(4)
    null = EmpiricalNull.fit(rng.standard_normal(5000))
    assert abs(rejection_rate(null.p_value(rng.standard_normal(20000)), 0.05) - 0.05) < 0.01


def test_cusum_alarm_and_calibration():
    _, alarm = cusum(np.r_[np.zeros(50), np.full(20, 2.0)], k=0.5, h=5.0)
    assert alarm is not None and 50 <= alarm < 60
    rng = np.random.default_rng(5)
    nulls = [rng.standard_normal(600) for _ in range(5)]
    assert calibrate_h(nulls, 0.5, target_arl=300.0) >= 3.0


def test_manifest_validation(tmp_path):
    files = []
    for i in range(4):
        path = tmp_path / f"img{i}.bin"
        path.write_bytes(bytes([i]) * 10)
        files.append(path)
    rows = [{"path": str(files[0]), "label": "real", "source": "a", "generator": "none", "split": "test"},
            {"path": str(files[1]), "label": "generated", "source": "b", "generator": "g1", "split": "test"}]
    manifest_path = tmp_path / "m.csv"
    write_manifest(rows, manifest_path)
    assert list(load_manifest(manifest_path)["y"]) == [0, 1]
    leaking = rows + [{"path": str(files[2]), "label": "real", "source": "a", "generator": "none", "split": "val"}]
    write_manifest(leaking, manifest_path)
    with pytest.raises(ManifestError, match="more than one split"):
        load_manifest(manifest_path)
    write_manifest(rows, manifest_path)
    files[1].write_bytes(b"tampered")
    with pytest.raises(ManifestError, match="checksum"):
        load_manifest(manifest_path)
