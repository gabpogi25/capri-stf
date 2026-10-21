"""
experiments/symbolic/verify_math_numerical.py

Numerical checks of the results in docs/MATHEMATICS.md, Sections 4b-5c.
The statements are proved there; these checks test the code and the hypotheses
and can fail. Each prints PASS or FAIL and has a failure control. Status: numerical.

  U1  Prop 4.4    standardised discrepancies: first-order agreement, equal per-pencil scale under
          isotropic noise (not under per-point heteroscedastic noise), the implemented function
  U2  Prop 4.5    the first-order size bounds hold; how conservative they are
  U3  Thm 5.3     the exact bound on |Delta log|J_k|| is never exceeded under adversarial search
  U4  Prop 5.4    P(|det| < d) for uniform unit columns lies between the proved bounds; tails of
          kappa_J and L(S)
  U5  Prop 5.5    near a boundary configuration kappa_J ~ 1/d for the 11 brackets of J1-J3 and
          stays bounded for the other 4
  U6  Remark 5.6  P(all brackets of m points < d) ~ d^(m-3)   (numerical only; no proof claimed)

Thresholds marked (*) were set after an exploration run on other seeds (90.1e6 band). The
thresholds of the revised checks (U1 ratio [8, 12], dispersion 0.5, gamma; U2 comonotone
[0.04, 0.0515]; U3 2.5 delta; U4 bound / 20; U5 1.01 and 0.01) were set after the first recorded
run (seed 20261019) and the in-project readings; the recorded run uses the fresh seed 20261021.
Run: python experiments/symbolic/verify_math_numerical.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithms.conditioning.scores import compute_invariant_noise_gain  # noqa: E402
from algorithms.consistency.pencil import log_abs_pencil_cross_ratios, log_abs_pencil_gradients  # noqa: E402
from algorithms.consistency.variants import sampled_pencils, standardized_discrepancies  # noqa: E402
from verify_pencil import RESULTS, report  # noqa: E402

SEED = 20261021                                       # fresh seed for the run after the readings
J_BRACKETS = (((0, 1, 2, 3), (0, 1, 4, 5), (0, 1, 2, 4), (0, 1, 3, 5)),      # J1 = [1234][1256]/([1235][1246])
              ((0, 2, 3, 4), (1, 2, 3, 5), (0, 2, 3, 5), (1, 2, 3, 4)),      # J2 = [1345][2346]/([1346][2345])
              ((0, 1, 4, 5), (2, 3, 4, 5), (0, 2, 4, 5), (1, 3, 4, 5)))      # J3 = [1256][3456]/([1356][2456])
UNUSED = {(0, 1, 2, 5), (0, 1, 3, 4), (0, 3, 4, 5), (1, 2, 4, 5)}             # [1236], [1245], [1456], [2356]
UPPER_DET = 32 / np.pi ** 2                                                   # P(|det| < d) <= (32/pi^2) d


def provenance() -> str:
    """One header line: the code commit and library versions that produced the output."""
    import platform
    import subprocess
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parent).stdout.strip() or "unknown"
    except OSError:
        commit = "unknown"
    import numpy, scipy, sympy
    return (f"# commit {commit}; Python {platform.python_version()}, numpy {numpy.__version__}, "
            f"scipy {scipy.__version__}, sympy {sympy.__version__}")


def geometry(rng, kind: str, n: int = 12) -> np.ndarray:
    """Unit representatives of n points: 'cube', 'flat' (thin slab) or 'far' (small, distant cloud)."""
    if kind == "cube":
        X = rng.uniform(-1, 1, (3, n)) + np.array([[0], [0], [4]])
    elif kind == "flat":
        X = rng.normal(0, 1, (3, n)) * np.array([[1], [1], [0.15]]) + np.array([[0], [0], [4]])
    else:
        X = rng.normal(0, 0.3, (3, n)) + np.array([[0], [0], [10]])
    P = np.vstack([X, np.ones((1, n))])
    return P / np.linalg.norm(P, axis=0)


def mean_field_quantile(scales: np.ndarray, p: float) -> float:
    """x_p with (1/T) sum_t (2 Phi(x / sigma_t) - 1) = p (Proposition 4.3)."""
    return optimize.brentq(lambda x: np.mean(2 * stats.norm.cdf(x / scales) - 1) - p, 1e-15, 1e15)


def first_order_ratio(rng, q, G, norms, sigma, draws=3) -> tuple[float, float, float]:
    """max_t |D_t - |g_t . w|| / (sqrt(2) sigma |g_t|) at sigma and sigma/10 with common random numbers,
    and the same error with the gradients of another scene (control)."""
    e1 = e2 = c1 = c2 = 0.0
    G_other = log_abs_pencil_gradients(geometry(rng, "cube"))
    for _ in range(draws):
        u0, v0 = rng.normal(size=q.shape), rng.normal(size=q.shape)
        for s_, store in ((sigma, 1), (sigma / 10, 2)):
            u, v = s_ * u0, s_ * v0
            D = np.abs(log_abs_pencil_cross_ratios(q + u) - log_abs_pencil_cross_ratios(q + v))
            w = (u - v).T.ravel()                                                  # point-major 4n coordinates
            err = np.max(np.abs(D - np.abs(G @ w)) / (np.sqrt(2) * s_ * norms))
            ctrl = np.max(np.abs(D - np.abs(G_other @ w)) / (np.sqrt(2) * s_ * norms))
            if store == 1:
                e1, c1 = max(e1, err), max(c1, ctrl)
            else:
                e2, c2 = max(e2, err), max(c2, ctrl)
    return e1, e2, c1 / c2


def random_frame(rng, P: np.ndarray) -> np.ndarray:
    """P in a random projective frame with random per-point scale and sign."""
    return (rng.normal(size=(4, 4)) @ P) * rng.uniform(0.5, 2.0, P.shape[1]) * rng.choice([-1, 1], P.shape[1])


def u1_standardised(rng, draws: int) -> None:
    print("U1  Prop 4.4: standardised discrepancies are marginally pivotal to first order (isotropic Gaussian noise)")
    sigma = 1e-7
    hetero = np.sqrt(np.where(np.arange(12) % 2 == 0, 3.0, 1 / 3))[None, :]       # per-point variance 3 or 1/3 (not i.i.d.)
    pencils = sampled_pencils(12, 20_000, 1)
    m0 = sigma / np.sqrt(2)                                                       # z ~ |N(0,1)| under the model
    tol = 2.5 * np.sqrt(2 / draws)                                                # Monte-Carlo sd of a mean of chi^2_1
    mf_raw, real_std = {}, {}
    for kind in ("cube", "flat", "far"):
        q = geometry(rng, kind)
        G = log_abs_pencil_gradients(q)
        norms = np.linalg.norm(G, axis=1)
        e1, e2, r_ctrl = first_order_ratio(rng, q, G, norms, sigma)
        report(f"{kind}: D_t - |g_t . w| is second order: with common noise, max rel. error {e1:.2e} at sigma = 1e-7 and "
               f"{e2:.2e} at 1e-8 (ratio {e1 / e2:.1f} in [8, 12]); control, gradients of another scene: ratio "
               f"{r_ctrl:.2f} not in [8, 12]", 8 <= e1 / e2 <= 12 and not 8 <= r_ctrl <= 12)
        print(f"      numerical only: {kind}: smallest gradient norm on the family, min |g_t| = {norms.min():.3g} "
              f"(the hypothesis |g_t| >= gamma > 0 of Proposition 4.4; reported, not a test)")
        m2_iso, m2_aniso, m2_impl, m2_frames, stds = 0.0, 0.0, 0.0, 0.0, []
        for _ in range(draws):
            u, v = rng.normal(0, sigma, q.shape), rng.normal(0, sigma, q.shape)
            z = np.abs(log_abs_pencil_cross_ratios(q + u) - log_abs_pencil_cross_ratios(q + v)) / (np.sqrt(2) * sigma * norms)
            m2_iso += z ** 2 / draws
            stds.append(np.quantile(z, 0.1))
            m2_impl += standardized_discrepancies(q + u, q + v, pencils, m0) ** 2 / draws
            m2_frames += standardized_discrepancies(random_frame(rng, q + u), random_frame(rng, q + v), pencils, m0) ** 2 / draws
            ua, va = rng.normal(0, sigma, q.shape) * hetero, rng.normal(0, sigma, q.shape) * hetero
            za = np.abs(log_abs_pencil_cross_ratios(q + ua) - log_abs_pencil_cross_ratios(q + va)) / (np.sqrt(2) * sigma * norms)
            m2_aniso += za ** 2 / draws
        spread = lambda m: np.quantile(np.abs(m - 1), 0.95)                       # noqa: E731
        disp = lambda m: np.quantile(np.abs(m / np.median(m) - 1), 0.95)          # noqa: E731  (level-free)
        report(f"{kind}: per-pencil E[z^2] = 1 with the gradient at q (95th pct |E z^2 - 1| = {spread(m2_iso):.3f} < {tol:.3f}; "
               f"dispersion around the median {disp(m2_iso):.3f}); control, noise variance 3 or 1/3 by point (not i.i.d.): "
               f"dispersion around the median {disp(m2_aniso):.3f} > 0.5 (unequal per-pencil scales, not a level shift)",
               spread(m2_iso) < tol and disp(m2_iso) < tol and disp(m2_aniso) > 0.5)
        report(f"{kind}: the implemented standardized_discrepancies (gradients at the measured points, m0 = sigma/sqrt 2) "
               f"has per-pencil E[z^2] = 1 ({spread(m2_impl):.3f} < {tol:.3f}); control, each measurement in its own "
               f"random projective frame: {spread(m2_frames):.3f} > 0.5", spread(m2_impl) < tol and spread(m2_frames) > 0.5)
        mf_raw[kind], real_std[kind] = mean_field_quantile(norms, 0.1), np.median(stds)
    r_raw = max(mf_raw.values()) / min(mf_raw.values())
    report(f"mean-field Q_0.1 of the raw statistic varies across geometries (computed ratio {r_raw:.1f} > 5); "
           f"for the standardised statistic it is {stats.halfnorm.ppf(0.1):.4f} in every geometry by the proof (not computed)",
           r_raw > 5)
    print(f"      numerical only: realised median Q_0.1 of z per geometry "
          f"{', '.join(f'{k} {v:.3f}' for k, v in real_std.items())} (dependence between pencils is not covered by Prop 4.4)")


def size_bounds(sig: np.ndarray, p: float, alpha: float) -> float:
    """Smallest c with min(union bound over the k smallest scales, Markov bound) <= alpha."""
    T = len(sig)
    k = int(np.ceil((T - 1) * p)) + 1
    small = np.sort(sig)[:k]

    def bound(c):
        return min(2 * stats.norm.sf(c / small).sum(), 2 * stats.norm.sf(c / sig).sum() / (T - k + 1))
    return optimize.brentq(lambda c: bound(c) - alpha, 1e-12, 1e8)


def standardised_threshold(T: int, k: int, alpha: float) -> float:
    """Smallest c with 2 T Phibar(c) / (T - k + 1) <= alpha (Proposition 4.5, standardised)."""
    return stats.norm.isf(alpha * (T - k + 1) / (2 * T))


def u2_size_bound(rng, draws: int) -> None:
    print("U2  Prop 4.5: first-order size bounds for Q_0.1 hold under the Gaussian surrogate")
    p = 0.1
    for kind in ("cube", "flat", "far"):
        G = log_abs_pencil_gradients(geometry(rng, kind))
        sig = np.linalg.norm(G, axis=1)
        Gs = G / sig[:, None]
        Z, Zs = [], []
        for _ in range(draws // 200):
            W = rng.normal(0, 1, (200, G.shape[1]))
            Z.append(np.quantile(np.abs(W @ G.T), p, axis=1))
            Zs.append(np.quantile(np.abs(W @ Gs.T), p, axis=1))
        Z, Zs = np.concatenate(Z), np.concatenate(Zs)
        T = len(sig)
        k = int(np.ceil((T - 1) * p)) + 1
        ok = True
        lines = []
        for alpha in (0.05, 0.01):
            tol = 3 * np.sqrt(alpha * (1 - alpha) / len(Z))
            c = size_bounds(sig, p, alpha)
            cs = standardised_threshold(T, k, alpha)
            ok &= np.mean(Z > c) <= alpha + tol and np.mean(Zs > cs) <= alpha + tol
            lines.append(f"alpha {alpha}: raw size {np.mean(Z > c):.3f} at c = {c / np.median(Z):.1f} x median, "
                         f"standardised size {np.mean(Zs > cs):.3f} at c = {cs:.3f}")
        mean_prob = lambda c: np.mean(2 * stats.norm.sf(c / sig))                       # noqa: E731
        c_ind = optimize.brentq(lambda c: stats.binom.sf(T - k, T, mean_prob(c)) - 0.05, 1e-12, 1e8)
        report(f"{kind}: " + "; ".join(lines), ok)
        report(f"{kind}: control, a binomial bound at the mean tail probability (as if pencils were independent and "
               f"equally scaled) fails (size {np.mean(Z > c_ind):.3f} > 0.2)",
               np.mean(Z > c_ind) > 0.2)
    # comonotone limit (all pencils move together): the size at the standardised threshold must be at most alpha
    # (plus Monte-Carlo error) and not far below it. Doubling alpha is caught; dropping the factor T/(T - k + 1)
    # is not (it gives size alpha, still valid in this limit)
    T, k = 41_580, int(np.ceil((41_580 - 1) * p)) + 1
    Zc = np.abs(rng.normal(size=200_000))                                        # Q_p of T equal values = |N(0,1)|
    cs = standardised_threshold(T, k, 0.05)
    size_c = np.mean(Zc > cs)
    hi = 0.05 + 3 * np.sqrt(0.05 * 0.95 / len(Zc))
    report(f"comonotone limit (Z = |N(0,1)|, {len(Zc):,} draws): size at the standardised threshold {size_c:.4f} in "
           f"[0.04, {hi:.4f}] (closed form 2 Phibar(c) = {2 * stats.norm.sf(cs):.4f}; valid and at most 20% conservative "
           f"in this limit; a missing factor T/(T - k + 1) would not be detected)", 0.04 <= size_c <= hi)


def log_j(P: np.ndarray) -> np.ndarray:
    d = lambda q: np.log(abs(np.linalg.det(P[:, list(q)])))                             # noqa: E731
    return np.array([d(a) + d(b) - d(c) - d(e) for a, b, c, e in J_BRACKETS])


def j_bounds(Q: np.ndarray, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """(the Theorem 5.3 bound: sum over the four brackets; control: the largest single-bracket term)."""
    eps = (1 + delta) ** 4 - 1
    terms = [[-np.log(1 - eps / abs(np.linalg.det(Q[:, list(q)]))) for q in k] for k in J_BRACKETS]
    return np.array([sum(t) for t in terms]), np.array([max(t) for t in terms])


def u3_log_j_bound(rng, configs: int) -> None:
    print("U3  Thm 5.3: |Delta log|J_k|| <= sum of -log(1 - eps/|b|) over its four brackets (adversarial search)")
    worst, worst_ctrl, worst_big = 0.0, 0.0, 0.0
    for trial in range(configs):
        Q = rng.normal(size=(4, 6))
        if trial % 2:
            Q[3] *= rng.uniform(0.01, 0.3)                                              # towards degeneracy
        Q /= np.linalg.norm(Q, axis=0)
        eps_max = min(abs(np.linalg.det(Q[:, list(q)])) for k in J_BRACKETS for q in k)
        for frac in (0.02, 0.2, 0.8):
            delta = (1 + frac * eps_max) ** 0.25 - 1                                    # eps = frac * min bracket
            bound, single = j_bounds(Q, delta)

            def clip(x):
                U = x.reshape(4, 6)
                return U * np.minimum(1.0, delta / np.maximum(np.linalg.norm(U, axis=0), 1e-300))  # |u_k| <= delta

            def neg(x, b=bound):
                return -np.max(np.abs(log_j(Q + clip(x)) - log_j(Q)) / b)
            starts = [rng.normal(size=24) * delta for _ in range(3)]
            for k in range(3):                                                  # steepest first-order ascent of each log|J_k|
                eps_fd = 1e-7
                grad = np.array([(log_j(Q + eps_fd * np.eye(24)[m].reshape(4, 6))[k] - log_j(Q)[k]) / eps_fd
                                 for m in range(24)])
                starts += [grad * delta, -grad * delta]
                for sgn in (1, -1):                                             # control: |u_k| <= 2.5 delta
                    U = sgn * grad.reshape(4, 6)
                    U = U * 2.5 * delta / np.maximum(np.linalg.norm(U, axis=0), 1e-300)
                    with np.errstate(all="ignore"):
                        worst_big = max(worst_big, np.nan_to_num(np.max(np.abs(log_j(Q + U) - log_j(Q)) / bound), nan=np.inf))
            for x0 in starts:
                res = optimize.minimize(neg, x0, method="Powell", options={"maxiter": 4000})
                change = np.abs(log_j(Q + clip(res.x)) - log_j(Q))
                worst = max(worst, np.max(change / bound))
                worst_ctrl = max(worst_ctrl, np.max(change / single))          # at the same solution
    report(f"{configs} configurations x 3 noise levels: largest |Delta log J| / bound = {worst:.3f} <= 1", worst <= 1)
    report(f"control: perturbations of size 2.5 delta (hypothesis broken) exceed the bound for delta "
           f"(ratio {worst_big:.2f} > 1)", worst_big > 1)
    print(f"      numerical only: largest |Delta log J| / (largest single-bracket term) = {worst_ctrl:.2f}")


def u4_small_ball(rng, n_mats: int) -> None:
    print("U4  Prop 5.4: (2/pi)(d sqrt(1-d^2) + asin d) <= P(|det| < d) <= (32/pi^2) d for uniform unit columns")
    X = rng.normal(size=(n_mats, 4, 4))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    dets = np.abs(np.linalg.det(X))
    ok = True
    parts = []
    for d in (1e-3, 1e-2, 1e-1):
        emp, se = np.mean(dets < d), np.sqrt(np.mean(dets < d) / n_mats)
        lo, up = (2 / np.pi) * (d * np.sqrt(1 - d * d) + np.arcsin(d)), UPPER_DET * d
        ok &= lo - 3 * se <= emp <= up + 3 * se
        parts.append(f"d = {d:g}: {emp:.5f} within 3 s.e. of [{lo:.5f}, {up:.5f}]")
    report(f"{n_mats:,} matrices: " + "; ".join(parts), ok)
    x = X[:, :, 0]                                                                 # uniform points of S^3
    laws = [("|x_1| < 0.5", np.mean(np.abs(x[:, 0]) < 0.5), (2 / np.pi) * (0.5 * np.sqrt(0.75) + np.arcsin(0.5))),
            ("|(x_1, x_2)| < 0.5", np.mean(np.hypot(x[:, 0], x[:, 1]) < 0.5), 0.25),
            ("|(x_1, x_2, x_3)| < 0.9", np.mean(np.linalg.norm(x[:, :3], axis=1) < 0.9),
             (2 / np.pi) * (np.arcsin(0.9) - 0.9 * np.sqrt(1 - 0.81)))]
    ok = all(abs(e - t) < 4 * np.sqrt(t * (1 - t) / n_mats) for _, e, t in laws)
    report("projection laws: " + "; ".join(f"P({name}) = {e:.5f} vs {t:.5f}" for name, e, t in laws), ok)
    Y = X[:, :, :3]
    inv_vol = 1 / np.sqrt(np.abs(np.linalg.det(np.transpose(Y, (0, 2, 1)) @ Y)))
    print(f"      numerical only: mean of 1/vol3 = {np.mean(inv_vol):.3f} vs 8/pi = {8 / np.pi:.3f} "
          f"(1/vol3 has infinite variance; no tolerance is claimed)")
    Xc = rng.normal(size=(n_mats // 4, 4, 4))
    Xc[:, 3, :] *= 0.05
    Xc /= np.linalg.norm(Xc, axis=1, keepdims=True)
    emp_c = np.mean(np.abs(np.linalg.det(Xc)) < 1e-2)
    report(f"control ({len(Xc):,} matrices): columns concentrated near a hyperplane exceed the upper bound "
           f"({emp_c:.3f} > {UPPER_DET * 1e-2:.3f})",
           emp_c > UPPER_DET * 1e-2)
    kappas, lmins = [], []
    for _ in range(n_mats // 50):
        P = rng.normal(size=(4, 6))
        P /= np.linalg.norm(P, axis=0)
        kappas.append(compute_invariant_noise_gain(P, (0, 1, 2, 3, 4, 5)))
        lmins.append(min(abs(np.linalg.det(P[:, list(q)])) for q in combinations(range(6), 4)))
    kappas, lmins = np.array(kappas), np.array(lmins)
    ok, parts = True, []
    for K in (1e2, 1e3, 1e4):
        emp, bound = np.mean(kappas > K), 11 * UPPER_DET * 8 / K
        ok &= emp <= bound
        parts.append(f"K = {K:g}: P(kappa_J > K) = {emp:.4f} <= {bound:.4f}, K P = {K * emp:.1f}")
    report(f"{len(kappas):,} six-point configurations: " + "; ".join(parts), ok)
    report("control: the tail bound divided by 20 is violated", any(np.mean(kappas > K) > 11 * UPPER_DET * 8 / K / 20
                                                                    for K in (1e2, 1e3, 1e4)))
    ok, parts = True, []
    for ell in (1e4, 1e6):
        d = np.sqrt(8 / ell)
        emp, lo, up = np.mean(lmins < d), (2 / np.pi) * (d * np.sqrt(1 - d * d) + np.arcsin(d)), 15 * UPPER_DET * d
        ok &= lo <= emp <= up
        parts.append(f"l = {ell:g}: P(L(S) > l) = {emp:.4f} in [{lo:.4f}, {up:.4f}]")
    report("; ".join(parts), ok)


def u5_boundary_rates(rng) -> None:
    print("U5  Prop 5.5: near a configuration with one zero bracket, kappa_J ~ 1/d iff the bracket is used by J1-J3")
    ok_used, ok_unused, ok_control = True, True, True
    for quad in combinations(range(6), 4):
        P = rng.normal(size=(4, 6))
        P /= np.linalg.norm(P, axis=0)
        a, b, c, e = quad
        P[:, e] = rng.normal(size=3) @ P[:, [a, b, c]].T                                # e on the plane of a, b, c
        P[:, e] /= np.linalg.norm(P[:, e])
        V = rng.normal(size=4)
        kap = []
        for d in (1e-5, 1e-6, 1e-7):
            Pd = P.copy()
            Pd[:, e] = P[:, e] + d * V
            kap.append(compute_invariant_noise_gain(Pd, (0, 1, 2, 3, 4, 5)))
        kd = np.array(kap) * np.array([1e-5, 1e-6, 1e-7])
        if quad in UNUSED:
            ok_unused &= kap[2] / kap[0] < 1.01
        else:
            ok_used &= abs(kd[2] / kd[1] - 1) < 0.01 and kd[2] > 1e-6
            ok_control &= not kap[2] / kap[0] < 1.01                     # the 'bounded' test must fail here
    report("the 11 used brackets: kappa_J * d converges to a positive constant", ok_used)
    report("the 4 unused brackets: kappa_J stays bounded", ok_unused)
    report("control: the bounded criterion fails for every used bracket", ok_control)


def u6_codimension(rng, n_samples: int) -> None:
    print("U6  Remark 5.6: P(all brackets of m of 6 uniform unit points < d) ~ d^(m-3)  (numerical)")
    X = rng.normal(size=(n_samples, 4, 6))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    quads = list(combinations(range(6), 4))
    B = np.abs(np.stack([np.linalg.det(X[:, :, list(q)]) for q in quads], 1))
    ds = np.array([0.02, 0.05, 0.1])
    slopes = []
    for m in (4, 5, 6):
        mx = B[:, [k for k, q in enumerate(quads) if max(q) < m]].max(1)
        P = np.array([np.mean(mx < d) for d in ds])
        slopes.append(np.polyfit(np.log(ds), np.log(np.maximum(P, 1e-12)), 1)[0])
    report(f"fitted slopes {', '.join(f'{s:.2f}' for s in slopes)} for m = 4, 5, 6 (each within 0.2 of m - 3) (*)",
           all(abs(s - (m - 3)) < 0.2 for s, m in zip(slopes, (4, 5, 6))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller samples (a fast run of the same checks)")
    args = parser.parse_args()
    print(provenance() + f"; seed {SEED}" + ("; quick" if args.quick else ""))
    rng = np.random.default_rng(SEED)
    q = args.quick
    u1_standardised(rng, 200 if q else 400)
    u2_size_bound(rng, 400 if q else 2000)
    u3_log_j_bound(rng, 4 if q else 40)
    u4_small_ball(rng, 200_000 if q else 2_000_000)
    u5_boundary_rates(rng)
    u6_codimension(rng, 300_000 if q else 3_000_000)
    failed = [name for name, ok in RESULTS if not ok]
    print("ALL CHECKS PASSED" if not failed else f"{len(failed)} CHECK(S) FAILED: {failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
