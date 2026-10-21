"""Real-data pencil test: scene split, window search and the analysis rules (synthetic inputs)."""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiments"), str(ROOT / "experiments" / "eval")]

import real_pencil_analysis as rpa  # noqa: E402
import run_real_pencil as rp  # noqa: E402


def test_split_is_fixed_by_its_seed_and_halves_the_scenes() -> None:
    batches = pd.DataFrame({"Dataset": ["pt_a", "pt_a", "imc2023_b", "ETs", "c", "d"],
                            "Scene": list("uvwxyz"), "N_Images": 10, "Batch": [1, 2, 1, 2, 1, 2],
                            "Included": [True] * 5 + [False]})
    a, b = rp.draw_split(batches), rp.draw_split(batches)
    assert a["Role"].tolist() == b["Role"].tolist()
    assert len(a) == 5 and (a["Role"] == "cal").sum() == 2
    assert a.set_index("Scene").loc["u", "Family"] == "pt" and a.set_index("Scene").loc["x", "Family"] == "ETs"


def test_find_windows_needs_six_views_sharing_enough_points() -> None:
    rng = np.random.default_rng(0)
    views = {i: {"name": f"im{i:02d}", "obs": {p: rng.normal(size=2) for p in range(60)}} for i in range(8)}
    views[7]["obs"] = {p: rng.normal(size=2) for p in range(10)}  # sees too few points
    windows = rp.find_windows({"views": views, "points": {}}, np.random.default_rng(1), 5)
    assert 1 <= len(windows) <= 5
    for order, points in windows:
        assert len(order) == rp.N_VIEWS and len(set(order)) == rp.N_VIEWS and 7 not in order
        assert len(points) == rp.N_POINTS and len(set(points.tolist())) == rp.N_POINTS
        assert [views[i]["name"] for i in order] == sorted(views[i]["name"] for i in order)
    assert len({frozenset(o) for o, _ in windows}) == len(windows)  # distinct view sets


def _windows(n_per_scene: int = 60, shift: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for unit, role in [("a", "cal"), ("b", "cal"), ("c", "test"), ("d", "test"), ("e", "test")]:
        for k in range(n_per_scene):
            for injection in ("none", "jitter", "drift"):
                bump = 0.0 if injection == "none" else shift
                rows.append({"Dataset": "pt_x", "Scene": unit, "Family": "pt", "Role": role, "Window": k,
                             "Injection": injection, "abstain": False,
                             "pencil_q10": rng.exponential() + bump, "acat": rng.exponential() + bump,
                             "sampson_mean": rng.exponential() + bump,
                             "split_glrt": rng.chisquare(3 * rpa.N_POINTS - 15) * (1 + bump), "sigma_hat_px": 1.0})
    return pd.DataFrame(rows)


def test_analysis_sizes_near_nominal_and_interim_never_decides() -> None:
    rows, decisions = rpa.analyse(_windows(shift=5.0), interim=1)
    sizes = {r["Statistic"]: r["Size"] for r in rows if r["Injection"] == "none" and "Transfer_From" not in r}
    for stat in rpa.SINGLE:
        assert 0.0 <= sizes[stat] <= 0.15
    assert all("no decision" in d["Decision"] for d in decisions if d["Decision"])
    powers = [r["Power"] for r in rows if r.get("Injection") == "jitter" and r["Size_Valid"]]
    assert all(p > 0.5 for p in powers)


def test_abstentions_get_the_calibration_median() -> None:
    df = _windows()
    df.loc[(df["Role"] == "test") & (df["Injection"] == "jitter"), "abstain"] = True
    rows, _ = rpa.analyse(df, interim=1)
    jitter = [r for r in rows if r.get("Injection") == "jitter" and r["Statistic"] == "pencil_q10"]
    assert jitter and (np.isnan(jitter[0]["Power"]) or jitter[0]["Power"] < 0.1)


def test_conformal_p_matches_definition() -> None:
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    assert rpa.conformal_p(cal, np.array([0.5, 2.0, 5.0])).tolist() == pytest.approx([1.0, 0.8, 0.2])
