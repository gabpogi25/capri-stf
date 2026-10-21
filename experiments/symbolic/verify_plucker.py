"""
experiments/symbolic/verify_plucker.py

Exact symbolic checks (sympy) of the algebraic facts used by the study; the
Macaulay2 script verify_plucker.m2 next to this file does the same.

  1. Sigma_1, Sigma_2, Sigma_3 (the relations behind rho) vanish identically
     as polynomials in the 24 coordinates of six points in P^3.
  2. All 15 three-term Grassmann-Plucker relations, one per pair {a, b}:
        [abcd][abef] - [abce][abdf] + [abcf][abde] = 0, with c<d<e<f the
        remaining indices, vanish identically.
  3. Blindness: the bracket vector b* with [1234] = [1456] = 1
     and all other brackets 0 satisfies Sigma_1 = Sigma_2 = Sigma_3 = 0 but
     violates the {1,4} relation.
  4. Stratum codimensions: m points of P^3 on a common plane <=> the 4 x m
     coordinate matrix has rank <= 3 <=> its 4 x 4 minors vanish. The
     codimension of that determinantal variety is (4-3)(m-3) = m - 3, checked
     here as the rank of the Jacobian of the minors at a random point of the
     variety (exact rational arithmetic).

Run: python experiments/symbolic/verify_plucker.py
"""

from itertools import combinations
import random

import sympy as sp

P = sp.Matrix(4, 6, lambda r, c: sp.Symbol(f"p{c + 1}_{r + 1}"))


def B(i, j, k, l):
    return P[:, [i - 1, j - 1, k - 1, l - 1]].det()


def check(name, expr):
    ok = sp.expand(expr) == 0
    print(f"  {name}: {'0 (identity holds)' if ok else 'NONZERO'}")
    return ok


def main() -> None:
    results = []
    print("1. Sigma relations")
    results.append(check("Sigma_1 = [1234][1256] - [1235][1246] + [1236][1245]",
                         B(1, 2, 3, 4) * B(1, 2, 5, 6) - B(1, 2, 3, 5) * B(1, 2, 4, 6) + B(1, 2, 3, 6) * B(1, 2, 4, 5)))
    results.append(check("Sigma_2 = [1345][2346] - [1346][2345] - [1234][3456]",
                         B(1, 3, 4, 5) * B(2, 3, 4, 6) - B(1, 3, 4, 6) * B(2, 3, 4, 5) - B(1, 2, 3, 4) * B(3, 4, 5, 6)))
    results.append(check("Sigma_3 = [1256][3456] - [1356][2456] + [1456][2356]",
                         B(1, 2, 5, 6) * B(3, 4, 5, 6) - B(1, 3, 5, 6) * B(2, 4, 5, 6) + B(1, 4, 5, 6) * B(2, 3, 5, 6)))

    print("2. The 15 Grassmann-Plucker relations")
    relations = {}
    for a, b in combinations(range(1, 7), 2):
        c, d, e, f = [x for x in range(1, 7) if x not in (a, b)]
        # brackets with sorted indices; sign of the sort permutation is absorbed by det
        rel = B(a, b, c, d) * B(a, b, e, f) - B(a, b, c, e) * B(a, b, d, f) + B(a, b, c, f) * B(a, b, d, e)
        relations[(a, b)] = (a, b, c, d, e, f)
        results.append(check(f"{{{a},{b}}}", rel))

    print("3. Blindness of Sigma_1..3: b* = e_1234 + e_1456")
    bstar = {q: 0 for q in combinations(range(1, 7), 4)}
    bstar[(1, 2, 3, 4)] = bstar[(1, 4, 5, 6)] = 1

    def val(*idx):
        s = sorted(idx)
        sign = sp.combinatorics.Permutation([s.index(x) for x in idx]).signature()
        return sign * bstar[tuple(s)]

    s1 = val(1, 2, 3, 4) * val(1, 2, 5, 6) - val(1, 2, 3, 5) * val(1, 2, 4, 6) + val(1, 2, 3, 6) * val(1, 2, 4, 5)
    s2 = val(1, 3, 4, 5) * val(2, 3, 4, 6) - val(1, 3, 4, 6) * val(2, 3, 4, 5) - val(1, 2, 3, 4) * val(3, 4, 5, 6)
    s3 = val(1, 2, 5, 6) * val(3, 4, 5, 6) - val(1, 3, 5, 6) * val(2, 4, 5, 6) + val(1, 4, 5, 6) * val(2, 3, 5, 6)
    a, b, c, d, e, f = relations[(1, 4)]
    r14 = val(a, b, c, d) * val(a, b, e, f) - val(a, b, c, e) * val(a, b, d, f) + val(a, b, c, f) * val(a, b, d, e)
    print(f"  Sigma_1, Sigma_2, Sigma_3 at b*: {s1}, {s2}, {s3};  {{1,4}} relation at b*: {r14}")
    results.append(s1 == s2 == s3 == 0 and r14 != 0)

    print("4. Codimension of 'm points coplanar' (rank <= 3 of the 4 x m matrix)")
    rng = random.Random(0)
    for m, name in ((4, "Boundary"), (5, "Compound"), (6, "Critical")):
        X = sp.Matrix(4, m, lambda r, c: sp.Symbol(f"x{c}_{r}"))
        minors = [X[:, list(cols)].det() for cols in combinations(range(m), 4)]
        J = sp.Matrix([[sp.diff(g, v) for v in X] for g in minors])
        # random rational point on the variety: m points on the plane x_3 = q0 x_0 + q1 x_1 + q2 x_2
        q = [sp.Rational(rng.randint(-9, 9), rng.randint(1, 9)) for _ in range(3)]
        point = {}
        for c in range(m):
            coords = [sp.Rational(rng.randint(-9, 9), rng.randint(1, 9)) for _ in range(3)]
            coords.append(sum(qi * ci for qi, ci in zip(q, coords)))
            point.update({X[r, c]: coords[r] for r in range(4)})
        codim = J.subs(point).rank()
        print(f"  m={m} ({name}): codimension {codim} (expected {m - 3})")
        results.append(codim == m - 3)

    print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    import sympy.combinatorics  # noqa: F401
    main()
