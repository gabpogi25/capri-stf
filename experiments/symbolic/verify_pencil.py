"""
experiments/symbolic/verify_pencil.py

Checks for every computational claim in docs/MATHEMATICS.md, Sections 2-6.
Each check prints PASS or FAIL. Each has a failure control: a deliberately
wrong input that must be rejected, which shows that the check can fail.

  V1  Lemma 2.1   gauge invariance of all pencil cross-ratios (exact rational arithmetic)
  V2  Prop. 2.3   J1, J2, J3 are pencil cross-ratios (randomized exact identity test)
  V3  Thm. 2.5    Grassmann-Plucker relation  <=>  CR(c,d;e,f) + CR(c,e;d,f) = 1 (randomized exact test)
  V4  Thm. 3.2    generic rank of the pencil family = 3n - 15; orbit tangent dimension 3n + ... (exact mod p)
  V5  Prop. 4.3   low quantile ~ p * sqrt(pi/2) * harmonic mean (numerical)
  V6  Prop. 5.1   0.674 <= median max|dlogJ| / (sigma kappa_J) <= 1.383 to first order (numerical)
  V7  Thm. 6.1    tau_exact is never exceeded under adversarial search (numerical; no counterexample found)
  V8  Prop. 6.3   Tyler whitening is equivariant up to an orthogonal map (numerical)

Statuses (see docs/MATHEMATICS.md): V1, V4 exact computer-assisted; V2, V3 randomized
exact identity tests (Schwartz-Zippel); V5-V8 numerical evidence only.

Run: python experiments/symbolic/verify_pencil.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from algorithms.consistency.pencil import log_abs_pencil_cross_ratios, log_abs_pencil_gradients, pencil_index  # noqa: E402
from algorithms.conditioning.scores import compute_conditioning_score, compute_invariant_noise_gain  # noqa: E402
from algorithms.utils import whiten_homogeneous_points  # noqa: E402

RESULTS: list[tuple[str, bool]] = []
PRIME = 2 ** 61 - 1


def report(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------- exact helpers
def det4(M):
    """Exact determinant of a 4x4 matrix of ints/Fractions (cofactor expansion)."""
    def det3(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    total = 0
    for c in range(4):
        minor = [[M[r][k] for k in range(4) if k != c] for r in range(1, 4)]
        total += (-1) ** c * M[0][c] * det3(minor)
    return total


def bracket(P, cols):
    """Exact bracket of columns `cols` (ordered) of P given as list of 4 rows."""
    return det4([[P[r][c] for c in cols] for r in range(4)])


def pencil_cr(P, i, j, c, d, e, f):
    """Exact CR_ij(c, d; e, f) = [ijce][ijdf] / ([ijcf][ijde])."""
    return Fraction(bracket(P, (i, j, c, e)) * bracket(P, (i, j, d, f)),
                    bracket(P, (i, j, c, f)) * bracket(P, (i, j, d, e)))


def random_int_points(rng, n, bound=10 ** 6):
    return [[int(v) for v in rng.integers(-bound, bound, n)] for _ in range(4)]


def all_pencil_tuples(n):
    for i, j in combinations(range(n), 2):
        rest = [k for k in range(n) if k not in (i, j)]
        for x in combinations(rest, 4):
            for (a1, a2), (b1, b2) in (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))):
                yield i, j, x[a1], x[a2], x[b1], x[b2]


# ---------------------------------------------------------------- V1
def v1_gauge_invariance(rng, n=8):
    print("V1  Lemma 2.1: every pencil cross-ratio is invariant under p_k -> lambda_k A p_k (exact)")
    P = random_int_points(rng, n, 50)
    A = [[int(v) for v in rng.integers(-9, 10, 4)] for _ in range(4)]
    while det4(A) == 0:
        A = [[int(v) for v in rng.integers(-9, 10, 4)] for _ in range(4)]
    lam = [int(v) * int(rng.choice([-1, 1])) for v in rng.integers(1, 9, n)]
    Q = [[sum(A[r][k] * P[k][c] for k in range(4)) * lam[c] for c in range(n)] for r in range(4)]
    tuples = list(all_pencil_tuples(n))
    ok = all(pencil_cr(P, *t) == pencil_cr(Q, *t) for t in tuples)
    report(f"all {len(tuples)} pencil cross-ratios of {n} points unchanged by a gauge", ok)
    # failure control: an unbalanced ratio [ijce][ijdf]/([ijcf][ijdc]) (e appears once on top, never below)
    i, j, c, d, e, f = tuples[0]
    unbalanced = lambda X: Fraction(bracket(X, (i, j, c, e)), bracket(X, (i, j, c, f)))  # noqa: E731
    report("control: an unbalanced bracket ratio [ijce]/[ijcf] CHANGES under the gauge", unbalanced(P) != unbalanced(Q))


# ---------------------------------------------------------------- V2, V3
def v2_j_are_pencil_ratios(rng, trials=30):
    print("V2  Prop. 2.3: J1, J2, J3 are pencil cross-ratios (randomized exact identity test)")
    ok = True
    for _ in range(trials):
        P = random_int_points(rng, 6)
        B = lambda *q: bracket(P, q)  # noqa: E731
        J1 = Fraction(B(0, 1, 2, 3) * B(0, 1, 4, 5), B(0, 1, 2, 4) * B(0, 1, 3, 5))
        J2 = Fraction(B(0, 2, 3, 4) * B(1, 2, 3, 5), B(0, 2, 3, 5) * B(1, 2, 3, 4))
        J3 = Fraction(B(0, 1, 4, 5) * B(2, 3, 4, 5), B(0, 2, 4, 5) * B(1, 3, 4, 5))
        ok &= J1 == pencil_cr(P, 0, 1, 2, 5, 3, 4)      # line p1p2:  CR_12(3,6;4,5)
        ok &= J2 == pencil_cr(P, 2, 3, 0, 1, 4, 5)      # line p3p4:  CR_34(1,2;5,6)
        ok &= J3 == pencil_cr(P, 4, 5, 0, 3, 1, 2)      # line p5p6:  CR_56(1,4;2,3)
    report(f"J1 = CR_12(3,6;4,5), J2 = CR_34(1,2;5,6), J3 = CR_56(1,4;2,3) at {trials} random integer points", ok)
    P = random_int_points(rng, 6)
    J1 = Fraction(bracket(P, (0, 1, 2, 3)) * bracket(P, (0, 1, 4, 5)), bracket(P, (0, 1, 2, 4)) * bracket(P, (0, 1, 3, 5)))
    report("control: J1 differs from a wrong pencil ratio CR_12(3,4;5,6)", J1 != pencil_cr(P, 0, 1, 2, 3, 4, 5))


def v3_plucker_is_cross_ratio_relation(rng, trials=30):
    print("V3  Thm. 2.5: GP relation for {a,b}  <=>  CR_ab(c,d;e,f) + CR_ab(c,e;d,f) = 1 (randomized exact)")
    ok_gp, ok_cr = True, True
    for _ in range(trials):
        P = random_int_points(rng, 6)
        B = lambda *q: bracket(P, q)  # noqa: E731
        for a, b in combinations(range(6), 2):
            c, d, e, f = [x for x in range(6) if x not in (a, b)]
            gp = B(a, b, c, d) * B(a, b, e, f) - B(a, b, c, e) * B(a, b, d, f) + B(a, b, c, f) * B(a, b, d, e)
            ok_gp &= gp == 0
            ok_cr &= pencil_cr(P, a, b, c, d, e, f) + pencil_cr(P, a, b, c, e, d, f) == 1
    report(f"all 15 GP relations vanish at {trials} random integer configurations", ok_gp)
    report("CR_ab(c,d;e,f) + CR_ab(c,e;d,f) = 1 for all 15 pencils", ok_cr)
    P = random_int_points(rng, 6)
    report("control: CR(c,d;e,f) + CR(c,f;d,e) != 1 (a wrong pairing)",
           pencil_cr(P, 0, 1, 2, 3, 4, 5) + pencil_cr(P, 0, 1, 2, 5, 3, 4) != 1)


# ---------------------------------------------------------------- V4
def rank_mod_p(rows, p=PRIME):
    rows = [[x % p for x in r] for r in rows]
    rank, ncols = 0, len(rows[0])
    for col in range(ncols):
        pivot = next((r for r in range(rank, len(rows)) if rows[r][col]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        inv = pow(rows[rank][col], p - 2, p)
        rows[rank] = [x * inv % p for x in rows[rank]]
        for r in range(len(rows)):
            if r != rank and rows[r][col]:
                factor = rows[r][col]
                rows[r] = [(x - factor * y) % p for x, y in zip(rows[r], rows[rank])]
        rank += 1
    return rank


def frac_mod_p(x: Fraction, p=PRIME):
    assert x.denominator % p != 0
    return x.numerator % p * pow(x.denominator % p, p - 2, p) % p


def exact_gradient_rows(P, n, tuples):
    """Rows of the exact gradient of log|CR_t| (point-major 4n coordinates), entries as Fractions."""
    inverse_cache = {}

    def inv_rows(quad):
        # row m of M^{-1} = cofactor column m / det: M^{-1}[m][r] = C[r][m] / det
        if quad not in inverse_cache:
            M = [[P[r][c] for c in quad] for r in range(4)]
            D = det4(M)
            rows = []
            for m in range(4):
                row = []
                for r in range(4):
                    minor = [[M[rr][cc] for cc in range(4) if cc != m] for rr in range(4) if rr != r]
                    cof = (-1) ** (r + m) * (minor[0][0] * (minor[1][1] * minor[2][2] - minor[1][2] * minor[2][1])
                                             - minor[0][1] * (minor[1][0] * minor[2][2] - minor[1][2] * minor[2][0])
                                             + minor[0][2] * (minor[1][0] * minor[2][1] - minor[1][1] * minor[2][0]))
                    row.append(Fraction(cof, D))
                rows.append(row)
            inverse_cache[quad] = rows
        return inverse_cache[quad]

    out = []
    for (i, j, c, d, e, f) in tuples:
        g = [Fraction(0)] * (4 * n)
        for quad, s in (((i, j, c, e), 1), ((i, j, d, f), 1), ((i, j, c, f), -1), ((i, j, d, e), -1)):
            rows = inv_rows(quad)
            for m, point in enumerate(quad):
                for r in range(4):
                    g[4 * point + r] += s * rows[m][r]
        out.append(g)
    return out


def orbit_tangent(P, n):
    """Tangent vectors of the gauge orbit: E_rs p_k for all k (16) and per-point scalings (n)."""
    vecs = []
    for r in range(4):
        for s in range(4):
            v = [0] * (4 * n)
            for k in range(n):
                v[4 * k + r] = P[s][k]
            vecs.append(v)
    for k in range(n):
        v = [0] * (4 * n)
        for r in range(4):
            v[4 * k + r] = P[r][k]
        vecs.append(v)
    return vecs


def v4_rank(rng, n=12, n_rows=240):
    print(f"V4  Thm. 3.2: the pencil family has generic rank 3n - 15 = {3 * n - 15} (exact, mod p = 2^61 - 1)")
    P = random_int_points(rng, n, 1000)
    tuples = list(all_pencil_tuples(n))
    pick = [tuples[k] for k in rng.choice(len(tuples), min(n_rows, len(tuples)), replace=False)]
    rows = exact_gradient_rows(P, n, pick)
    rows_p = [[frac_mod_p(x) for x in r] for r in rows]
    T = orbit_tangent(P, n)
    tangent_rank = rank_mod_p([list(v) for v in T])
    report(f"orbit tangent has rank >= {tangent_rank} (expected 16 + n - 1 = {16 + n - 1})", tangent_rank == 16 + n - 1)
    orth = all(sum(a * b for a, b in zip(g, t)) % PRIME == 0 for g in rows_p for t in T)
    report("sanity check: every pencil gradient is orthogonal to the orbit tangent mod p", orth)
    r = rank_mod_p(rows_p)
    report(f"rank of {len(pick)} pencil gradients mod p = {r}; with the tangent bound this gives rank_Q = {3 * n - 15}",
           r == 3 * n - 15 and tangent_rank == 16 + n - 1)
    extra = [int(v) for v in rng.integers(1, 10 ** 6, 4 * n)]
    report(f"control: adding a random row raises the rank to {3 * n - 14}", rank_mod_p(rows_p + [extra]) == 3 * n - 14)
    if n > 6:
        j_tuples = [(0, 1, 2, 5, 3, 4), (2, 3, 0, 1, 4, 5), (4, 5, 0, 3, 1, 2)]      # J1, J2, J3 (V2)
        j_rows = [[frac_mod_p(x) for x in r] for r in exact_gradient_rows(P, n, j_tuples)]
        report("control: J1, J2, J3 of one 6-tuple have rank 3 (< 3n - 15)", rank_mod_p(j_rows) == 3)
    pencil_rows = [[frac_mod_p(x) for x in r] for r in exact_gradient_rows(P, n, tuples[:3])]
    report("the three pairings of one 4-set on one line have rank 1 (functions of one cross-ratio)",
           rank_mod_p(pencil_rows) == 1)


# ---------------------------------------------------------------- V5
def v5_quantile_harmonic(rng, n=12):
    print("V5  Prop. 4.3: Gaussian mean-field quantile x_p ~ p sqrt(pi/2) H as p -> 0 (numerical)")
    from scipy import optimize, stats

    P = rng.standard_normal((4, n))
    G = log_abs_pencil_gradients(P)
    sigma = 1e-3 * np.linalg.norm(G, axis=1)                    # w ~ N(0, 1e-6 I): sigma_t = 1e-3 |g_t|
    H = 1.0 / np.mean(1.0 / sigma)
    def mean_field_quantile(p):
        F = lambda x: np.mean(2 * stats.norm.cdf(x / sigma) - 1) - p  # noqa: E731
        return optimize.brentq(F, 1e-300, 10 * sigma.max(), xtol=1e-300, rtol=1e-15, maxiter=2000)

    ps = (0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
    ratios = [mean_field_quantile(p) / (p * np.sqrt(np.pi / 2) * H) for p in ps]
    report(f"(a) lower bound x_p >= p sqrt(pi/2) H at p = {ps}: min ratio {min(ratios):.12f}",
           min(ratios) >= 1 - 1e-9)
    report(f"(b) ratio -> 1 as p -> 0: {np.round(ratios, 6).tolist()}", abs(ratios[-1] - 1) < 1e-3)
    xp = mean_field_quantile(0.1)
    if xp <= sigma.min():
        report(f"(c) x_0.1 <= min sigma_t, so ratio <= e^(1/2): ratio {ratios[0]:.3f}", ratios[0] <= np.exp(0.5))
    else:
        print(f"      (c) not applicable: x_0.1 > min sigma_t; ratio at p = 0.1 is {ratios[0]:.3f}")
    # realized quantile over the noise draws vs mean-field (dependence across t)
    draws = []
    for _ in range(200):
        w = 1e-3 * rng.standard_normal(4 * n)
        draws.append(np.quantile(np.abs(G @ w), 0.1))
    xp = mean_field_quantile(0.1)
    spread = np.std(np.log(draws))
    report(f"realized Q_0.1 / mean-field x_0.1: median {np.median(draws) / xp:.3f}, sd of log {spread:.3f}",
           0.7 < np.median(draws) / xp < 1.3)
    # the harmonic mean ignores ill-conditioned invariants (exact bound), RMS does not
    worst = np.argsort(sigma)[-len(sigma) // 100:]              # the 1% largest scales
    w_S = np.sum(1 / sigma[worst]) / np.sum(1 / sigma)
    s2 = sigma.copy()
    s2[worst] *= 100.0
    H2, rms, rms2 = 1 / np.mean(1 / s2), np.sqrt(np.mean(sigma ** 2)), np.sqrt(np.mean(s2 ** 2))
    report(f"inflating the largest 1% of scales 100x: H moves {H2 / H:.5f}x <= bound 1/(1 - w_S) = {1 / (1 - w_S):.5f}; "
           f"RMS moves {rms2 / rms:.1f}x", 1 <= H2 / H <= 1 / (1 - w_S) + 1e-12)
    report("control: the bound is violated if the inflated set is taken from the SMALLEST scales",
           (lambda idx: (lambda s3: 1 / np.mean(1 / s3) / H)(np.where(np.isin(np.arange(len(sigma)), idx), sigma * 100, sigma)))(
               np.argsort(sigma)[:len(sigma) // 100]) > 1 / (1 - w_S))


# ---------------------------------------------------------------- V6
def v6_kappa_bounds(rng, n_configs=150, gain=1e-3, draws=2000):
    print("V6  Prop. 5.1: 0.674 <= median max|dlogJ| / (sigma kappa_J) <= 1.383 in the first-order regime (numerical)")
    from scipy import stats

    sys.path.insert(0, str(ROOT / "experiments" / "math_lab"))
    import lab

    # The proposition is first order: it needs sigma * kappa_J << 1. Each configuration is
    # therefore tested at sigma = gain / kappa_J. (With a fixed sigma = 1e-5, near-degenerate
    # tuples have sigma * kappa_J ~ 1 and the ratio falls to 0.32; docs/MATHEMATICS.md
    # records this as the non-linear limit of kappa_J.)
    ratios, l_spread, not_in_j = [], [], 0
    j_quads = {q for pair in lab.J_QUADS for half in pair for q in half}
    for _ in range(n_configs):
        P = lab._cond_config(rng)
        t = tuple(range(6))
        kappa = compute_invariant_noise_gain(P, t)
        sigma = gain / kappa
        base = lab.log_abs_J_batch(P, np.arange(6)[None])[0]
        noisy = np.array([lab.log_abs_J_batch(P + sigma * rng.standard_normal((4, 6)), np.arange(6)[None])[0]
                          for _ in range(draws)])
        med = np.median(np.max(np.abs(noisy - base), axis=1))
        ratios.append(med / (sigma * kappa))
        l_spread.append(med / (sigma * np.sqrt(compute_conditioning_score(P, t))))
        dets = {q: abs(np.linalg.det(P[:, list(q)])) for q in combinations(range(6), 4)}
        not_in_j += min(dets, key=dets.get) not in j_quads
    lo, hi = stats.norm.ppf(0.75), stats.norm.isf(1 / 12)
    # Sampling s.e. of a median of `draws` values is 0.5 / (sqrt(draws) f(median)). At the lower end the
    # density of |N(0,1)| at its median is 2 phi(lo); at the upper end 2 phi(hi) is used (conservative).
    slack_lo = 4 * 0.5 / (np.sqrt(draws) * 2 * stats.norm.pdf(lo))
    slack_hi = 4 * 0.5 / (np.sqrt(draws) * 2 * stats.norm.pdf(hi))
    ratios = np.array(ratios)
    report(f"ratio range [{ratios.min():.3f}, {ratios.max():.3f}] within [{lo:.3f} - {slack_lo:.3f}, "
           f"{hi:.3f} + {slack_hi:.3f}] (4 Monte Carlo s.e. each side)",
           ratios.min() > lo - slack_lo and ratios.max() < hi + slack_hi)
    l_spread = np.array(l_spread)
    report(f"control: the same ratio with sqrt(L(S)) in place of kappa spans {l_spread.max() / l_spread.min():.0f}x "
           f"(kappa: {ratios.max() / ratios.min():.2f}x)", l_spread.max() / l_spread.min() > 5 * ratios.max() / ratios.min())
    print(f"      note: in {not_in_j}/{n_configs} configurations the smallest bracket (which sets L(S)) "
          "is not one of the 11 brackets used by J1-J3")


# ---------------------------------------------------------------- V7
def v7_tau_exact(rng, trials=25, restarts=3):
    print("V7  Thm. 6.1: rho never exceeds tau_exact under adversarial perturbations (numerical)")
    from scipy import optimize

    from algorithms.syzygy.residual import compute_exact_admissibility_bound, compute_syzygy_residual

    quads = list(combinations(range(6), 4))
    worst, worst_pb = 0.0, 0.0
    for _ in range(trials):
        Q = rng.standard_normal((4, 6))
        Q /= np.linalg.norm(Q, axis=0)                       # true unit representatives
        delta = float(np.exp(rng.uniform(np.log(1e-3), np.log(0.2))))
        tau = compute_exact_admissibility_bound(delta)
        pattern = rng.integers(0, 2, len(quads))              # which measurement each bracket comes from

        def neg_rho(z):
            U = z.reshape(2, 4, 6)
            U = delta * U / np.maximum(1.0, np.linalg.norm(U, axis=1, keepdims=True))   # |u_k| <= delta
            meas = (Q + U[0], Q + U[1])
            brackets = {q: float(np.linalg.det(meas[s_][:, list(q)])) for q, s_ in zip(quads, pattern)}
            return -compute_syzygy_residual(brackets)

        def neg_rho_per_bracket(z):
            # strongest adversary allowed by the hypothesis: every bracket uses its own representatives
            U = z.reshape(len(quads), 4, 4)
            U = delta * U / np.maximum(1.0, np.linalg.norm(U, axis=1, keepdims=True))
            brackets = {q: float(np.linalg.det(Q[:, list(q)] + U[k])) for k, q in enumerate(quads)}
            return -compute_syzygy_residual(brackets)

        best = max(-optimize.minimize(neg_rho, rng.standard_normal(48), method="Powell",
                                      options={"maxiter": 3000, "xtol": 1e-6}).fun for _ in range(restarts))
        best_pb = -optimize.minimize(neg_rho_per_bracket, rng.standard_normal(16 * len(quads)), method="Powell",
                                     options={"maxiter": 4000, "xtol": 1e-6}).fun
        worst = max(worst, best / tau)
        worst_pb = max(worst_pb, best_pb / tau)
    report(f"two shared measurements: max rho / tau_exact over {trials} searches = {worst:.4f} (<= 1)", worst <= 1.0)
    report(f"independent representatives per bracket: max rho / tau_exact = {worst_pb:.4f} (<= 1)", worst_pb <= 1.0)
    print("      (no failure control: the proof is analytic; a search can only fail to find a counterexample)")


# ---------------------------------------------------------------- V8
def v8_tyler(rng, n=20):
    print("V8  Prop. 6.3: Tyler whitening of X and of A X agree up to an orthogonal map (numerical)")
    X = rng.standard_normal((4, n))
    A = rng.standard_normal((4, 4)) @ np.diag([1, 3, 0.2, 7])
    Wx, Wa = whiten_homogeneous_points(X), whiten_homogeneous_points(A @ X * rng.uniform(0.5, 2, n))

    # column signs are arbitrary (projective points): align by trying the per-column sign that best matches
    def residual(U, V):
        best = np.inf
        for _ in range(3):
            s = np.ones(U.shape[1])
            for _ in range(5):
                u, _, vt = np.linalg.svd((V * s) @ U.T)
                R = u @ vt
                s = np.sign(np.sum((R @ U) * V, axis=0))
            best = min(best, np.max(np.abs(R @ U - V * s)))
        return best

    r = residual(Wx, Wa)
    report(f"Tyler: max residual after the best orthogonal alignment = {r:.2e}", r < 1e-6)
    C = lambda Y: np.linalg.inv(np.linalg.cholesky(Y @ Y.T / Y.shape[1]))  # noqa: E731
    plain = lambda Y: (lambda Z: Z / np.linalg.norm(Z, axis=0))(C(Y / np.linalg.norm(Y, axis=0)) @ (Y / np.linalg.norm(Y, axis=0)))  # noqa: E731
    r_plain = residual(plain(X), plain(A @ X * rng.uniform(0.5, 2, n)))
    report(f"control: covariance whitening of unit columns is NOT equivariant (residual {r_plain:.2e})", r_plain > 1e-3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip the slow numerical checks V6 and V7")
    args = parser.parse_args()
    rng = np.random.default_rng(20261011)
    v1_gauge_invariance(rng)
    v2_j_are_pencil_ratios(rng)
    v3_plucker_is_cross_ratio_relation(rng)
    v4_rank(rng)
    v4_rank(rng, n=6, n_rows=45)
    v4_rank(rng, n=7, n_rows=315)
    v4_rank(rng, n=8, n_rows=240)
    v5_quantile_harmonic(rng)
    if not args.quick:
        v6_kappa_bounds(rng)
        v7_tau_exact(rng)
    v8_tyler(rng)
    failed = [name for name, ok in RESULTS if not ok]
    print("ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED: {failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
