"""
experiments/symbolic/verify_math_exact.py

Exact checks of the results in docs/MATHEMATICS.md, Section 3b.
Each check prints PASS or FAIL and has a failure control that must be rejected.

  W1  Prop 3.5  frame identities: with p1..p5 the standard frame and p_k = (x, y, z, w),
          CR_12(3,4;5,k) = z/w, CR_13(2,4;5,k) = y/w, CR_23(1,4;5,k) = x/w (SymPy, symbolic)
  W2  Prop 3.5  in any frame, the three frame cross-ratios of p_k equal the ratios of the
          frame coordinates of p_k (exact over Q, random integer configurations)
  W3  Thm 3.7   the 3(n-5) frame cross-ratios have Jacobian rank 3n - 15 over Q for n in
          {6..10, 12, 15, 20, 25, 30}; rows from the full family do not raise it; the full
          family itself has rank 3n - 15 for n = 6, 7, 8 (exact Fraction elimination)
  W4  Lemma 3.6 the stabiliser of a configuration with p1..p5 in general position has
          dimension 1 (Lie algebra, exact nullspace over Q)
  W5  Remark 3.9 the 45 pencil cross-ratios of six points as rational functions of the frame
          coordinates of point 6, checked against exact evaluation

Statuses: W1, W2, W4, W5 exact identities or exact linear algebra at the stated points;
W3 exact rank over Q at random integer points (it corroborates a proved statement).

Run: python experiments/symbolic/verify_math_exact.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_pencil import (RESULTS, all_pencil_tuples, exact_gradient_rows, pencil_cr,  # noqa: E402
                           random_int_points, report)

SEED = 20261018
FRAME_TUPLES = ((0, 1, 2, 3, 4), (0, 2, 1, 3, 4), (1, 2, 0, 3, 4))   # (i, j, c, d, e) of CR_ij(c, d; e, k), 0-based


def frame_tuples(n: int) -> list[tuple[int, ...]]:
    """The 3(n - 5) frame cross-ratios CR_12(3,4;5,k), CR_13(2,4;5,k), CR_23(1,4;5,k), k = 6..n (0-based)."""
    return [t + (k,) for k in range(5, n) for t in FRAME_TUPLES]


def rank_q(rows: list[list[Fraction]]) -> int:
    """Exact rank over Q by Gaussian elimination on Fractions."""
    M = [list(r) for r in rows]
    rank, n_cols = 0, len(M[0]) if M else 0
    for col in range(n_cols):
        pivot = next((r for r in range(rank, len(M)) if M[r][col] != 0), None)
        if pivot is None:
            continue
        M[rank], M[pivot] = M[pivot], M[rank]
        for r in range(rank + 1, len(M)):
            if M[r][col] != 0:
                factor = M[r][col] / M[rank][col]
                M[r] = [a - factor * b for a, b in zip(M[r], M[rank])]
        rank += 1
    return rank


def w1_frame_identities() -> None:
    print("W1  Prop 3.5: frame identities CR_12(3,4;5,k) = z/w, CR_13(2,4;5,k) = y/w, CR_23(1,4;5,k) = x/w (symbolic)")
    x, y, z, w = sp.symbols("x y z w")
    P = {1: (1, 0, 0, 0), 2: (0, 1, 0, 0), 3: (0, 0, 1, 0), 4: (0, 0, 0, 1), 5: (1, 1, 1, 1), 6: (x, y, z, w)}

    def br(*idx):
        return sp.Matrix([P[i] for i in idx]).T.det()

    def cr(i, j, c, d, e, f):
        return sp.simplify(br(i, j, c, e) * br(i, j, d, f) / (br(i, j, c, f) * br(i, j, d, e)))

    got = (cr(1, 2, 3, 4, 5, 6), cr(1, 3, 2, 4, 5, 6), cr(2, 3, 1, 4, 5, 6))
    report("the three identities hold as rational functions", [sp.simplify(g - t) for g, t in zip(got, (z / w, y / w, x / w))] == [0, 0, 0],
           f"{got}")
    wrong = cr(1, 2, 3, 5, 4, 6)
    report("control: CR_12(3,5;4,k) is not z/w", sp.simplify(wrong - z / w) != 0, f"{wrong}")
    signs = (br(2, 3, 4, 6), br(1, 3, 4, 6), br(1, 2, 4, 6), br(1, 2, 3, 6))
    report("brackets in the standard frame: [234k] = -x, [134k] = y, [124k] = -z, [123k] = w",
           [sp.expand(a - b) for a, b in zip(signs, (-x, y, -z, w))] == [0, 0, 0, 0], f"{signs}")


def frame_coordinates(P: list[list[int]], n: int) -> list[tuple[Fraction, Fraction, Fraction]]:
    """Exact (x/w, y/w, z/w) of p_6..p_n in the projective frame p1..p5 (A p_i ~ e_i, A p5 ~ (1,1,1,1))."""
    B = sp.Matrix([[P[r][c] for c in range(4)] for r in range(4)])
    coeff = B.LUsolve(sp.Matrix([P[r][4] for r in range(4)]))
    A = (B * sp.diag(*coeff)).inv()
    out = []
    for k in range(5, n):
        v = A * sp.Matrix([P[r][k] for r in range(4)])
        out.append(tuple(Fraction(int(sp.fraction(v[i] / v[3])[0]), int(sp.fraction(v[i] / v[3])[1])) for i in range(3)))
    return out


def w2_general_frame(rng, n=9, trials=5) -> None:
    print("W2  Prop 3.5: in a general frame, the frame cross-ratios are the frame-coordinate ratios (exact over Q)")
    ok, ok_control = True, True
    for _ in range(trials):
        P = random_int_points(rng, n, 1000)
        coords = frame_coordinates(P, n)
        for k, (cx, cy, cz) in zip(range(5, n), coords):
            crs = [pencil_cr(P, *t, k) for t in FRAME_TUPLES]          # (z/w, y/w, x/w)
            ok &= crs == [cz, cy, cx]
            ok_control &= pencil_cr(P, 0, 1, 2, 4, 3, k) != cz         # wrong pairing
    report(f"{trials} configurations of {n} points: all {3 * (n - 5)} frame cross-ratios match", ok)
    report("control: a wrong pairing CR_12(3,5;4,k) does not match z/w", ok_control)


def w3_rank(rng, n_values=(6, 7, 8, 9, 10, 12, 15, 20, 25, 30)) -> None:
    print("W3  Thm 3.7: the frame cross-ratios have Jacobian rank 3n - 15 over Q (exact); full family for n <= 8")
    for n in n_values:
        P = random_int_points(rng, n, 1000)
        rows = exact_gradient_rows(P, n, frame_tuples(n))
        r = rank_q(rows)
        all_t = list(all_pencil_tuples(n))
        extra_t = [all_t[k] for k in rng.choice(len(all_t), 12, replace=False)]
        r_extra = rank_q(rows + exact_gradient_rows(P, n, extra_t))
        one_line = [t for t in all_t if t[:2] == (0, 1)][:len(rows)]                # same count, one line
        r_line = rank_q(exact_gradient_rows(P, n, one_line))
        report(f"n = {n}: frame rows rank {r} = 3n - 15 = {3 * n - 15}; with 12 full-family rows {r_extra}; "
               f"control, {len(one_line)} pencils through one line: rank {r_line} = n - 5 < 3n - 15",
               r == 3 * n - 15 and r_extra == r and r_line == n - 5 and r_line < r)
        if n <= 8:
            r_full = rank_q(exact_gradient_rows(P, n, all_t))
            report(f"n = {n}: the full family ({len(all_t)} pencils) has rank {r_full} = 3n - 15", r_full == 3 * n - 15)


def stabiliser_dimension(P: list[list[int]], n: int) -> int:
    """dim of {(X, mu): X p_k + mu_k p_k = 0 for all k} (Lie algebra of the stabiliser in GL(4) x (R*)^n)."""
    X = sp.Matrix(4, 4, sp.symbols("x0:16"))
    mu = sp.symbols(f"m0:{n}")
    eqs = []
    for k in range(n):
        p = sp.Matrix([P[r][k] for r in range(4)])
        eqs += list(X * p + mu[k] * p)
    M = sp.Matrix([[sp.diff(e, v) for v in list(X) + list(mu)] for e in eqs])
    return len(M.nullspace())


def w4_stabiliser(rng) -> None:
    print("W4  Lemma 3.6: the stabiliser of a configuration with p1..p5 in general position has dimension 1")
    dims = [stabiliser_dimension(random_int_points(rng, n, 1000), n) for n in (5, 6, 8)]
    report(f"random integer configurations, n = 5, 6, 8: dimensions {dims}", dims == [1, 1, 1])
    P = random_int_points(rng, 7, 1000)
    plane = [[P[r][c] for c in range(3)] for r in range(4)]                    # p1, p2, p3
    for k in range(4, 7):                                                      # p5.. on the plane of p1, p2, p3
        a, b, c = (int(v) for v in rng.integers(1, 50, 3))
        for r in range(4):
            P[r][k] = a * plane[r][0] + b * plane[r][1] + c * plane[r][2]
    d = stabiliser_dimension(P, 7)
    report(f"control: with p5..p7 on the plane of p1, p2, p3 the dimension is {d} > 1", d > 1)


def w5_six_point_formulas(rng) -> None:
    print("W5  Remark 3.9: the 45 pencil cross-ratios of six points as functions of the frame coordinates of point 6")
    x, y, z, w = sp.symbols("x y z w")
    pts = [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 1, 1, 1), (x, y, z, w)]

    def br(idx):
        return sp.Matrix([pts[i] for i in idx]).T.det()

    formulas = {}
    for (i, j, c, d, e, f) in all_pencil_tuples(6):
        formulas[(i, j, c, d, e, f)] = sp.factor(br((i, j, c, e)) * br((i, j, d, f)) / (br((i, j, c, f)) * br((i, j, d, e))))
    ok, ok_control = True, True
    for _ in range(5):
        vals = [int(v) for v in rng.integers(-10 ** 4, 10 ** 4, 4)]
        P = [[pts[c][r] if c < 5 else vals[r] for c in range(6)] for r in range(4)]
        sub = dict(zip((x, y, z, w), vals))
        for t, expr in formulas.items():
            exact = pencil_cr(P, *t)
            ok &= sp.Rational(exact.numerator, exact.denominator) == expr.subs(sub)
        wrong = 0                                       # control: the reciprocal pairing convention must disagree
        for (i, j, c, d, e, f), expr in formulas.items():
            exact = pencil_cr(P, i, j, d, c, e, f)
            wrong += sp.Rational(exact.numerator, exact.denominator) != expr.subs(sub)
        ok_control &= wrong == len(formulas)
    report(f"{len(formulas)} closed forms agree with exact evaluation at 5 random points", ok and len(formulas) == 45)
    report("control: all 45 formulas with c and d swapped (the reciprocal) disagree with exact evaluation", ok_control)
    distinct = sorted({str(e) for e in formulas.values()}, key=lambda t: (len(t), t))
    print(f"      distinct closed forms ({len(distinct)}): " + "; ".join(distinct))


def main() -> None:
    from verify_math_numerical import provenance
    print(provenance() + f"; seed {SEED}")
    rng = np.random.default_rng(SEED)
    w1_frame_identities()
    w2_general_frame(rng)
    w3_rank(rng)
    w4_stabiliser(rng)
    w5_six_point_formulas(rng)
    failed = [name for name, ok in RESULTS if not ok]
    print("ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED: {failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
