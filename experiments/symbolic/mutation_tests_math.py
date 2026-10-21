"""
experiments/symbolic/mutation_tests_math.py

Shows that the checks can fail: each mutation below breaks the code under test or a
constant of a statement, runs the affected check, and must make it FAIL. A mutation that
passes is a check without power, and is reported as such (exit code 1). Also runs a
reconstruction of the first version of U1 (never committed), which failed in quick mode (see
docs/MATHEMATICS.md, Section 4b), and a mutated copy of the Macaulay2 check (if M2 is installed).

Run: python experiments/symbolic/mutation_tests_math.py   (output recorded in
mutation_tests_math_output.txt)
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

import verify_math_exact as ex  # noqa: E402
import verify_math_numerical as nu  # noqa: E402
import verify_pencil as vp  # noqa: E402

SEED = 20261020
OUTCOMES: list[tuple[str, bool]] = []


def caught(check, *args) -> bool:
    vp.RESULTS.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        check(np.random.default_rng(SEED), *args)
    return any(not ok for _, ok in vp.RESULTS)


def mutate(name: str, module, attr: str, value, check, *args) -> None:
    original = getattr(module, attr)
    setattr(module, attr, value)
    try:
        hit = caught(check, *args)
    finally:
        setattr(module, attr, original)
    OUTCOMES.append((name, hit))
    print(f"  [{'caught' if hit else 'NOT CAUGHT'}] {name}")


def bad_stabiliser(P, n):
    X = sp.Matrix(4, 4, sp.symbols("x0:16"))
    eqs = [e for k in range(n) for e in X * sp.Matrix([P[r][k] for r in range(4)])]
    return len(sp.Matrix([[sp.diff(e, v) for v in list(X)] for e in eqs]).nullspace())


def first_version_u1(rng, draws: int = 60) -> None:
    """A reconstruction of the first U1 (never committed): fixed thresholds that ignored Monte-Carlo error."""
    sigma = 1e-7
    q = nu.geometry(rng, "cube")
    G = nu.log_abs_pencil_gradients(q)
    norms = np.linalg.norm(G, axis=1)
    err, m2 = 0.0, 0.0
    for k in range(draws):
        u, v = rng.normal(0, sigma, q.shape), rng.normal(0, sigma, q.shape)
        D = np.abs(nu.log_abs_pencil_cross_ratios(q + u) - nu.log_abs_pencil_cross_ratios(q + v))
        m2 += (D / (np.sqrt(2) * sigma * norms)) ** 2 / draws
        if k < 5:
            err = max(err, np.max(np.abs(D - np.abs(G @ (u - v).T.ravel())) / (np.sqrt(2) * sigma * norms)))
    vp.report("first version: max first-order error < 1e-3", err < 1e-3)
    vp.report("first version: 95th pct |E z^2 - 1| < 0.25", np.quantile(np.abs(m2 - 1), 0.95) < 0.25)


def main() -> None:
    print(nu.provenance() + f"; seed {SEED}")
    print("Exact checks (verify_math_exact.py)")
    orig = ex.FRAME_TUPLES
    mutate("W3: a frame tuple duplicated", ex, "FRAME_TUPLES", (orig[0], orig[0], orig[2]), ex.w3_rank, (8, 12))
    mutate("W2: a frame pairing swapped", ex, "FRAME_TUPLES", ((0, 1, 2, 4, 3), orig[1], orig[2]), ex.w2_general_frame, 8, 2)
    rq = ex.rank_q
    mutate("W3: rank under-counted by one", ex, "rank_q", lambda rows: rq(rows) - 1 if len(rows) > 3 else rq(rows),
           ex.w3_rank, (8,))
    mutate("W4: mu dropped from the stabiliser equations", ex, "stabiliser_dimension", bad_stabiliser, ex.w4_stabiliser)

    print("Numerical checks (verify_math_numerical.py, quick sizes)")
    orig_src = (HERE / "verify_math_numerical.py").read_text()
    src = orig_src.replace("np.sqrt(2) * sigma * norms", "sigma * norms").replace("m0 = sigma / np.sqrt(2)", "m0 = sigma")
    if orig_src.count("np.sqrt(2) * sigma * norms") == 0 or orig_src.count("m0 = sigma / np.sqrt(2)") == 0:
        raise SystemExit("U1 mutation did not apply: the source text it edits has changed")
    scope = {"__file__": str(HERE / "verify_math_numerical.py"), "__name__": "mutant"}
    exec(compile(src, "mutant", "exec"), scope)                                     # noqa: S102 (local, own source)
    hit = caught(scope["u1_standardised"], 200)
    OUTCOMES.append(("U1: the sqrt(2) of the standardisation dropped", hit))
    print(f"  [{'caught' if hit else 'NOT CAUGHT'}] U1: the sqrt(2) of the standardisation dropped")
    jb = nu.j_bounds
    mutate("U3: the Theorem 5.3 bound divided by 2", nu, "j_bounds", lambda Q, d: (jb(Q, d)[0] / 2, jb(Q, d)[1]), nu.u3_log_j_bound, 4)
    mutate("U4: 32/pi^2 halved", nu, "UPPER_DET", nu.UPPER_DET / 2, nu.u4_small_ball, 200_000)
    mutate("U5: wrong set of unused brackets", nu, "UNUSED", {(0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 3, 4), (1, 2, 3, 5)},
           nu.u5_boundary_rates)
    st = nu.standardised_threshold
    mutate("U2: standardised bound too small by a factor of 2", nu, "standardised_threshold",
           lambda T, k, a: st(T, k, 2 * a), nu.u2_size_bound, 400)

    print("Macaulay2 (verify_frame_m2.m2 with the target of CR_12(3,4;5,k) changed from z/w to y/w)")
    import shutil
    import subprocess
    import tempfile
    if shutil.which("M2"):
        src = (HERE / "verify_frame_m2.m2").read_text().replace("frameCheck(1,2,3,4,5,6, z, w)", "frameCheck(1,2,3,4,5,6, y, w)")
        assert "frameCheck(1,2,3,4,5,6, y, w)" in src
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mutant.m2"
            path.write_text(src)
            out = subprocess.run(["M2", "--script", str(path)], capture_output=True, text=True, timeout=300).stdout
        hit = "M2 FRAME CHECK FAILED" in out
        OUTCOMES.append(("Macaulay2: wrong target", hit))
        print(f"  [{'caught' if hit else 'NOT CAUGHT'}] Macaulay2: wrong target")
    else:                                       # a skipped mutation must not count as caught
        OUTCOMES.append(("Macaulay2: wrong target (M2 not installed, not run)", False))
        print("  [NOT RUN] Macaulay2: wrong target (M2 not installed)")

    print("Recorded failure: a reconstruction of the first version of U1 (quick mode)")
    hit = caught(first_version_u1)
    print(f"  [{'fails, as recorded' if hit else 'PASSES (record is wrong)'}] reconstructed first version of U1")
    OUTCOMES.append(("reconstructed first version of U1 fails in quick mode", hit))

    missed = [name for name, hit in OUTCOMES if not hit]
    print("ALL MUTATIONS CAUGHT" if not missed else f"{len(missed)} NOT CAUGHT: {missed}")
    raise SystemExit(1 if missed else 0)


if __name__ == "__main__":
    main()
