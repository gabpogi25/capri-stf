# The mathematics of CAPRI: pencils, invariants, consistency and conditioning

Dated 2026-10-18. Sections 3b, 4b, 4c, 5b, 5c and 11 extend the first set of results (2026-10-11),
whose statements and proofs are unchanged. All results were read in the project by readers told
to find errors; this is not peer review (Section 11).
This document states every mathematical result the study relies on, with a proof or the reason
there is none. It covers the bracket theory (brackets, Plücker relations, the residual rho, the
bound tau_exact) and the pencil theory built on it: pencil cross-ratios, the drift test, and
kappa_J. Each computational claim names the check that tests it: `verify_pencil.py` (V1-V8),
`verify_plucker.py`, `verify_math_exact.py` (W1-W5), `verify_math_numerical.py` (U1-U6),
`verify_frame_m2.m2` (Macaulay2), all in `experiments/symbolic/`, and the Lean development in
`formal/`. Every number is copied from the recorded run of its check (`*_output.txt`; seeds
20261011, 20261018, 20261021, and 20261020 for the mutation harness); numbers marked *derived*
are computed from printed values.

## Status labels

| Label | Meaning |
|---|---|
| **proved** | A complete written proof, no computation. |
| **computer-assisted** | A written argument in which finitely many facts are decided by a named program in exact arithmetic. "Randomized" means a Schwartz-Zippel identity test with failure probability stated. |
| **cited** | A published result. Its hypotheses are stated; it has not been re-proved here, and the source was not read in full in this project (see `LITERATURE_SEARCH.md`). |
| **numerical** | Evidence from floating-point experiments. Never used as a step in a proof. |

## Novelty

Nothing in this document is claimed to be new. Pencil cross-ratios, brackets
and the Grassmann-Plücker relations are classical. Searches of the web
(summaries only) found no earlier statement of the drift test of Section 4,
of Proposition 4.3, or of Theorem 2.5 in this form. OpenAlex, arXiv and
publisher full texts could not be opened, and the projective-shape statistics
literature (Mardia-Patrangenaru 2005, Kent-Mardia 2012) was not read. The frame coordinates of Section 3b are classical (projective frames;
Mardia and Patrangenaru 2005 per a search summary, not read; addendum of 2026-10-18). The tail
arguments of Sections 4c and 5c are elementary.

---

## 1. Setting

**Points.** A configuration is n >= 6 vectors p_1, ..., p_n in R^4 \ {0},
each representing a point of P^3. P denotes the 4 x n matrix with these
columns.

**Brackets.** [i j k l] = det(p_i, p_j, p_k, p_l). Swapping two indices
changes the sign.

**Gauge.** The group G = GL(4) x (R^*)^n acts by p_k -> lambda_k A p_k. Two
measurements of the same scene in different projective frames, with
arbitrary representatives, differ by an element of G. Under the gauge,

    [i j k l] -> det(A) lambda_i lambda_j lambda_k lambda_l [i j k l].      (1.1)

**Six-point invariants.** For a 6-tuple labelled 1..6:
J1 = [1234][1256] / ([1235][1246]), J2 = [1345][2346] / ([1346][2345]),
J3 = [1256][3456] / ([1356][2456]).

**Unit chart.** For noise models, each point has a true unit representative
q_k (|q_k| = 1). A measurement is a_k = q_k + u_k, possibly followed by an
unknown gauge.

---

## 2. Pencil geometry

**Definition 2.0 (pencil cross-ratio).** For i != j and distinct indices
c, d, e, f outside {i, j}:

    CR_ij(c, d; e, f) = [i j c e][i j d f] / ( [i j c f][i j d e] ),

defined when the denominator is nonzero.

**Lemma 2.1 (gauge invariance).** CR_ij(c, d; e, f) is unchanged by every
gauge (A, lambda), including its sign. *Status: proved.* Check: V1, exact
over Q for all 1,260 pencil cross-ratios of 8 random integer points. Failure
control: the unbalanced ratio [ijce]/[ijcf] changes.

*Proof.* By (1.1), each bracket gains det(A) times the product of its
lambdas. The numerator and denominator each contain two brackets, so det(A)^2
cancels. Each index appears equally often above and below the line: i and j
twice each, and c, d, e, f once each ([ijce] and [ijcf] for c; [ijdf] and
[ijde] for d; [ijce] and [ijde] for e; [ijdf] and [ijcf] for f). So every
lambda_k cancels, sign included. ∎

**Lemma 2.2 (quotient by a line).** Let p_i, p_j be linearly independent.
There is a linear map pi : R^4 -> R^2 with kernel span(p_i, p_j), and a
constant kappa != 0, such that for all x, y in R^4

    det(p_i, p_j, x, y) = kappa * det_2( pi(x), pi(y) ).

Consequently

    CR_ij(c, d; e, f) = det(c', e') det(d', f') / ( det(c', f') det(d', e') ),   with x' = pi(p_x).

This is the classical cross-ratio of the four points c', d', e', f' of
P^1. Geometrically, P^1 here is the pencil of planes through the line
p_i p_j, and x' stands for the plane span(p_i, p_j, p_x). *Status: proved.*

*Proof.* Extend p_i, p_j to a basis (p_i, p_j, b_3, b_4) and write
x = alpha p_i + beta p_j + x_3 b_3 + x_4 b_4, and similarly for y. The
determinant is multilinear and vanishes when a column repeats, so

    det(p_i, p_j, x, y) = (x_3 y_4 - x_4 y_3) det(p_i, p_j, b_3, b_4).

Take pi(x) = (x_3, x_4) and kappa = det(p_i, p_j, b_3, b_4) != 0. In the
cross-ratio, kappa^2 cancels. ∎

**Proposition 2.3 (J1, J2, J3 are pencil cross-ratios).** In the tuple
labels 1..6:

    J1 = CR_12(3,6; 4,5),    J2 = CR_34(1,2; 5,6),    J3 = CR_56(1,4; 2,3).

*Status: proved.* Check: V2, a randomized exact test at 30 random integer
configurations. After clearing denominators the identities are polynomial
identities of total degree 16, with coordinates drawn from 2*10^6 integers. A
false identity therefore survives one trial with probability at most
16/(2*10^6) = 8*10^-6 (Schwartz-Zippel), and all 30 trials with probability
below 1.3*10^-153. Failure control:
CR_12(3,4;5,6) != J1.

*Proof.*
- CR_12(3,6;4,5) = [1234][1265] / ([1235][1264]). Since [1265] = -[1256] and [1264] = -[1246], this equals [1234][1256] / ([1235][1246]) = J1.
- For J2: CR_34(1,2;5,6) = [3415][3426] / ([3416][3425]). In each bracket, moving the third index to the front is a 3-cycle, an even permutation. This gives [1345][2346] / ([1346][2345]) = J2.
- For J3: CR_56(1,4;2,3) = [5612][5643] / ([5613][5642]). Here [56xy] = [xy56], [4356] = -[3456] and [4256] = -[2456]. The two minus signs cancel, giving [1256][3456] / ([1356][2456]) = J3. ∎

**Proposition 2.4 (the pencil family and its redundancy).** For a line (ij)
and a 4-set {c, d, e, f}, the three pairings give

    lambda_1 = CR_ij(c,d; e,f),   lambda_2 = CR_ij(c,e; d,f),   lambda_3 = CR_ij(c,f; d,e),

with lambda_2 = 1 - lambda_1 and lambda_3 = (lambda_1 - 1) / lambda_1. All
three are functions of one number (assuming all brackets of the pencil are
nonzero, so that lambda_1 != 0). There are C(n,2) C(n-2,4) pairs (line,
4-set) (66 x 210 = 13,860 for n = 12) and three values for each (41,580). Every log|CR| is a
signed sum of four of the C(n,4) log-brackets (495 for n = 12).
*Status: proved* (given Theorem 2.5). Check: V4, where the three pairings of
one pencil have gradient rank 1, exactly.

*Proof.* Theorem 2.5 gives lambda_1 + lambda_2 = 1. By the definition and
antisymmetry,

    lambda_2 / lambda_1 = -[ijcd][ijef] / ([ijce][ijdf]) = -lambda_3,

so lambda_3 = -lambda_2 / lambda_1 = (lambda_1 - 1) / lambda_1. The counts are
immediate. ∎

**Theorem 2.5 (each Grassmann-Plücker relation is a cross-ratio identity).**
For distinct a, b and the remaining c, d, e, f of a 6-tuple, define the
Grassmann-Plücker expression

    GP_ab = [abcd][abef] - [abce][abdf] + [abcf][abde].

Whenever [abcf][abde] != 0,

    GP_ab = [abcf][abde] * ( 1 - CR_ab(c,d; e,f) - CR_ab(c,e; d,f) ).        (2.1)

Moreover GP_ab = 0 identically, so CR_ab(c,d;e,f) + CR_ab(c,e;d,f) = 1.
Away from the locus where [abcf][abde] = 0, the 15 Plücker relations of six
points are therefore exactly the 15 instances of the cross-ratio identity
lambda + (1 - lambda) = 1, one on each pencil.
The relations Sigma_1, Sigma_2, Sigma_3 used for J1-J3 are three of them:
Sigma_1 = GP_12, Sigma_2 = -GP_34, Sigma_3 = GP_56 (as coded in
`algorithms/syzygy/residual.py`).
*Status: proved.* Check: V3, a randomized exact test of both statements for
all 15 pencils. Failure control: the wrong pairing sum
CR(c,d;e,f) + CR(c,f;d,e) is not equal to 1.

*Proof.* Divide GP_ab by [abcf][abde]:
- The middle term gives -[abce][abdf] / ([abcf][abde]) = -CR_ab(c,d;e,f).
- For the first term, CR_ab(c,e;d,f) = [abcd][abef] / ([abcf][abed]) = -[abcd][abef] / ([abcf][abde]), so the first term equals -CR_ab(c,e;d,f).
- The last term gives 1.

This proves (2.1).

For the vanishing: if p_a and p_b are linearly dependent, every bracket
[ab..] is 0 and GP_ab = 0 trivially. Otherwise, by Lemma 2.2,
GP_ab = kappa^2 * E(c', d', e', f'), where

    E(c,d,e,f) = det(c,d) det(e,f) - det(c,e) det(d,f) + det(c,f) det(d,e),   c, d, e, f in R^2.

E is 4-linear. Swapping c and d turns E into

    -det(c,d)det(e,f) - det(d,e)det(c,f) + det(d,f)det(c,e),

which is -E term by term. The swaps d <-> e and e <-> f are checked the same
way (an independent review checked all six transpositions symbolically; E
expands to 0 identically), so E is alternating. An alternating 4-linear form on the 2-dimensional
space R^2 is zero (Lambda^4 of R^2 = 0). ∎

**Corollary 2.6.** If all brackets are computed
from one set of vectors, every GP_ab vanishes, so rho = max_k |Sigma_k| = 0
whatever the vectors are. rho can be nonzero only when the brackets of one
pencil come from different measurements. In that case, by (2.1),
|Sigma| = |[abcf][abde]| * |1 - lambda_1 - lambda_2|. So rho measures the
violation of lambda_1 + lambda_2 = 1 inside a pencil, weighted by a product
of brackets. The weight shrinks near degenerate pencils. *Status: proved*
(immediate from Theorem 2.5).

**Interpretation for the study.** The Plücker relations and the pencil
cross-ratios are two uses of the same object, the pencil of planes through a
line:
- **consistency within one measurement:** the Plücker relations, always
  satisfied by Corollary 2.6;
- **comparison between two measurements:** the change in each cross-ratio,
  which is informative (Section 3).

---

## 3. Completeness of the pencil family

**Lemma 3.1 (gradient of a log-bracket).** For an invertible 4x4 matrix M,
d log|det M| = tr(M^{-1} dM). The gradient with respect to column m of M is
row m of M^{-1}. *Status: proved* (Jacobi's formula). The code
`log_abs_pencil_gradients` uses it; it is checked against finite differences
(relative error 4e-7).

**Theorem 3.2 (generic rank).** Let Phi(P) = (log|CR_t(P)|) over all t be
the pencil map on R^{4n} (where defined). For n = 12, its Jacobian has rank
3n - 15 = 21 on a dense Zariski-open set of configurations.
*Status: computer-assisted (exact, mod p).* Check: V4.

*Proof.*

*Upper bound.* Invariance (Lemma 2.1) makes Phi constant along gauge orbits.
So the Jacobian annihilates the orbit tangent space, which is spanned by
E_rs P (r, s = 1..4) and by the per-point scalings e_k (x) p_k. At a random
integer configuration these 16 + n vectors have rank 16 + n - 1 = 27 modulo
p = 2^61 - 1, hence rank at least 27 over Q. (The overall scaling appears in
both families, which accounts for the -1.) Rank is lower semicontinuous, so
the tangent rank is at least 27 on a Zariski-open set. There the Jacobian
rank is at most 4n - 27 = 21.

*Lower bound.* At the same integer point, 240 exact gradient rows (rational
by Lemma 3.1) have rank 21 modulo p. Rank modulo p never exceeds rank over
Q, so the rank is at least 21 at that point, and therefore on a Zariski-open
set.

The two open sets are nonempty, so they meet in a dense open set. Rank mod p
is at most rank over Q because reduction from the p-integral rationals to F_p
is a ring homomorphism; the program asserts that no denominator is divisible
by p. As a sanity check (mod p only), V4 also confirms that every computed
gradient is orthogonal to the orbit tangent. Failure controls: a random extra row raises
the rank to 22; J1, J2, J3 of one 6-tuple have rank 3. ∎

*Remark.* The same computation (V4) gives:
- n = 6: rank 3 from all 45 pencil cross-ratios (the three invariants J1-J3), tangent rank 21;
- n = 7: rank 6 from all 315, tangent rank 22;
- n = 8: rank 9 from 240, tangent rank 23.

An independent rerun by the reviewer (different prime and points) agreed. For
general n the upper bound holds by the same reasoning; the lower bound was
computed only for n = 6, 7, 8 and 12.

**Corollary 3.3 (local separation, n = 12).** Let P be a configuration of 12
points in the dense open set of Theorem 3.2. There is a neighbourhood U of P such that, for Q in
U, Phi(Q) = Phi(P) if and only if Q = (A, lambda) . P for some gauge near the
identity. In words: near a generic configuration the pencil cross-ratios
detect every change that is not a change of frame or of representatives.
*Status: computer-assisted* (it rests on Theorem 3.2).

*Proof.* Near P the rank is at least 21 (lower semicontinuity) and at most 21
(the tangent rank stays 27 on an open set and the gradients annihilate the
tangent). So it is constant, equal to 21, near P. By the constant rank
theorem, the level set of Phi through P is, near P, a submanifold of
dimension 48 - 21 = 27. The orbit map (A, lambda) -> (A, lambda) . P has
differential of rank 27 near the identity, so it is a submersion onto that
27-dimensional level set, and its image is open there. ∎

*Contrast with rho.* Corollary 2.6 says the Plücker expressions vanish on
every configuration. Corollary 3.3 says that, locally, the pencil differences vanish only on the
gauge orbit. This is the precise sense in which the cross-ratio
comparison can detect deformation and rho cannot.

### 3b. Frame cross-ratios and the rank for every n

Let U_n be the set of configurations P = (p_1, ..., p_n), n >= 6, in which every bracket
[ijkl] is nonzero. U_n is open, dense and invariant under the gauge group G.

**Proposition 3.5 (frame cross-ratios).** Let p_1..p_5 be the standard frame
e_1, e_2, e_3, e_4, (1,1,1,1) and p_k = (x, y, z, w). Then

    CR_12(3,4;5,k) = z/w,    CR_13(2,4;5,k) = y/w,    CR_23(1,4;5,k) = x/w.

*Status: proved.* Checks: W1 (SymPy, symbolic), `verify_frame_m2.m2` (Macaulay2, an independent
system), W2 (exact over Q, in general frames). Control: the pairing CR_12(3,5;4,k) gives
(w - z)/w, not z/w.

*Proof.* By Lemma 2.2 with line (1,2) and pi(x) = (x_3, x_4): p_3 -> (1,0), p_4 -> (0,1),
p_5 -> (1,1), p_k -> (z, w). With CR = det(c',e') det(d',f') / (det(c',f') det(d',e')) and
(c,d,e,f) = (3,4,5,k): det((1,0),(1,1)) = 1, det((0,1),(z,w)) = -z, det((1,0),(z,w)) = w,
det((0,1),(1,1)) = -1, so CR = z/w. For line (1,3) take pi(x) = (x_2, x_4), which sends
p_2, p_4, p_5, p_k to (1,0), (0,1), (1,1), (y, w); the same computation gives y/w. For line
(2,3) take pi(x) = (x_1, x_4), which gives x/w. The constant kappa of Lemma 2.2 cancels in each
ratio. ∎

**Lemma 3.6 (stabiliser).** For P in U_n, the Lie algebra
{(X, mu) in gl(4) x R^n : X p_k + mu_k p_k = 0 for all k} is one-dimensional, spanned by
(I, -1). *Status: proved.* Check: W4 (exact nullspace, n = 5, 6, 8: dimension 1). Control: with
p_5..p_7 on the plane of p_1, p_2, p_3 the dimension is 2.

*Proof.* p_1..p_4 is a basis (their bracket is nonzero); in it X p_i = -mu_i p_i for i <= 4, so
X is diagonal with entries -mu_1..-mu_4. Write p_5 = sum c_i p_i; by Cramer's rule c_i is the
bracket with p_5 in place of p_i divided by [1234], so c_i != 0 on U_n. X p_5 = -mu_5 p_5 gives
mu_i c_i = mu_5 c_i, so mu_i = mu_5 for i <= 4 and X = -mu_5 I. Then X p_k = -mu_5 p_k forces
mu_k = mu_5 for every k. ∎

**Theorem 3.7 (rank for every n).** For every n >= 6 and every P in U_n, the Jacobian of the
pencil map Phi = (log|CR_t|)_t over all pencils has rank exactly 3n - 15. *Status: proved* (this
covers every n; Theorem 3.2 was computer-assisted for n = 6, 7, 8, 12 at generic
points). Checks: W3, exact rank over Q at random integer points: the 3(n - 5) frame rows have
rank 3n - 15 for n in {6, ..., 10, 12, 15, 20, 25, 30}, 12 further rows of the full family do
not raise it, and the full family itself has rank 3n - 15 for n = 6, 7, 8 (45, 315 and 1,260
rows). Control: the same number of pencils, all through one line, has rank n - 5.

*Proof.* Upper bound: Phi is constant on gauge orbits (Lemma 2.1), so DPhi(P) vanishes on the
image of the differential at the identity of the orbit map g -> g.P, which is
(X, mu) -> (X p_k + mu_k p_k)_k. Its kernel is one-dimensional (Lemma 3.6), so its image has
dimension 16 + n - 1 = n + 15, and rank DPhi(P) <= 4n - (n + 15) = 3n - 15.

Lower bound: take g in G with g.p_i = e_i (i <= 4) and g.p_5 = (1,1,1,1) (p_1..p_5 is a frame,
and G contains the per-point scalings). U_n is G-invariant, Phi(g.Q) = Phi(Q) on U_n, and
Q -> g.Q is an invertible linear map, so rank DPhi(P) = rank DPhi(g.P). At P_0 = g.P, consider
the affine slice on which p_1..p_5 are fixed and p_6..p_n vary; the Jacobian of Phi restricted
to it is a submatrix of DPhi(P_0). By Proposition 3.5, on that slice the components
log|CR_12(3,4;5,k)|, log|CR_13(2,4;5,k)|, log|CR_23(1,4;5,k)| equal log|z_k/w_k|, log|y_k/w_k|,
log|x_k/w_k|. Their Jacobian is block diagonal, one block per k, with rows
(0, 0, 1/z, -1/w), (0, 1/y, 0, -1/w), (1/x, 0, 0, -1/w), of rank 3 because
[234k] = -x, [134k] = y, [124k] = -z, [123k] = w at P_0 (W1, symbolic), all nonzero.
So rank DPhi(P) >= 3(n - 5). ∎

**Corollary 3.8 (separation).** (a) *Local, every n:* near P in U_n, Phi(Q) = Phi(P) if and only
if Q = h.P for a gauge h near the identity (the proof of Corollary 3.3 applies, with constant
rank 3n - 15 on all of U_n and Lemma 3.6). (b) *Global, signed:* for P, Q in U_n, the 3(n - 5)
signed frame cross-ratios agree if and only if Q is in G.P. *Status: proved.* With absolute
values (log|CR|, as the statistic uses), only (a) is claimed.

*Proof of (b).* Signed cross-ratios are gauge-invariant (Lemma 2.1). Map both configurations to
the standard frame; the signed cross-ratios fix x_k/w_k, y_k/w_k, z_k/w_k for every k >= 6
(w_k = [123k] != 0), so the two normalised configurations agree up to the scale of each point,
which is a gauge. ∎

**Remark 3.9 (six points in closed form).** In the frame of Proposition 3.5 all 45 pencil cross-ratios of six points are
rational functions of the frame coordinates (x, y, z, w) of point 6; W5 lists them and checks all
45 against exact evaluation at 5 random points. Control: with c and d swapped (the reciprocal
convention) the formulas disagree with exact evaluation.

Theorem 3.7 and Corollary 3.8 supersede the computed cases of Theorem 3.2 and Corollary 3.3.

---

## 4. The pencil drift test

**Model.** True unit representatives q_k. Two measurements a_k = q_k + u_k
and b_k = q_k + v_k, each followed by an unknown gauge. The perturbations
scale with a radius r: u = r u~ and v = r v~, where the law of (u~, v~) does
not depend on r. In the benchmark, each u~_k has a uniform direction and a
magnitude uniform on [0, 1]. The question is whether r <= delta (bounded) or
r > delta (excessive).

**Test statistic.** With D_t = | log|CR_t(A)| - log|CR_t(B)| | over all t,

    S = Q_p( {D_t} ) / delta,    p = 0.1,

where Q_p is the empirical p-quantile. Implemented as
`algorithms/consistency/pencil.py::pencil_drift_score`.

**Lemma 4.0 (two properties of the quantile).** For vectors x, y in R^T and
s >= 0:
- Q_p(s x) = s Q_p(x);
- |Q_p(x) - Q_p(y)| <= max_t |x_t - y_t|.

*Status: proved.* Order statistics are positively homogeneous, and the k-th
order statistic is 1-Lipschitz in the sup norm: if |x_t - y_t| <= h for all t,
then at least k entries of y are at most x_(k) + h, and conversely. The
interpolated quantile (np.quantile) is a fixed convex combination of two order
statistics, so it inherits both properties.

**Theorem 4.1 (first-order scale family).** Let K be a compact set of
configurations on which every bracket used is bounded away from 0, and
assume |u~_k|, |v~_k| <= B almost surely (true in the benchmark, with B = 1). Write
g_t for the gradient of log|CR_t| at q (Lemma 3.1) and w = u~ - v~. Then,
uniformly for q in K,

    D_t = r | g_t . w | + O(r^2),      S = (r / delta) Z + O(r^2 / delta),

where Z = Q_p( { |g_t . w| } ). The variable Z does not depend on r, on
delta, on the two frames or on the representatives. Z does depend on q and on the chart in which the noise is added. In the limit
delta -> 0 with f fixed, the decision "S > c" has size P(Z > c) if Z has no atom at c, and
power P(Z > c / f) when r = f delta if Z has no atom at c / f. Neither depends on delta,
which is why one threshold can be used for every delta. *Status: proved.*
An independent numerical test (reviewer) found Q_p(D)/r -> Z with relative
error 2.7e-2, 2.8e-3, 5.4e-4 and 1.1e-4 at r = 0.1, 0.01, 1e-3 and 1e-4:
linear in r, and unaffected by a random gauge.

*Proof.*
1. By Lemma 2.1, D_t = | log|CR_t(q + r u~)| - log|CR_t(q + r v~)| |; the gauges have no effect.
2. On K, log|CR_t| is smooth with bounded second derivatives, so each term is log|CR_t(q)| + r g_t . u~ + O(r^2), with the same for v~. Subtracting gives D_t = r |g_t . w| + O(r^2).
3. Lemma 4.0 transfers this to the quantile: Q_p(D) = r Q_p(|g . w|) + O(r^2). ∎

**Remark 4.2 (the worst-case bound for rho is not usable).** The worst-case
admissibility bound tau_exact (Theorem 6.1) is proved but loose. In the drift-admission experiment the
median rho/tau_exact is 0.003 for bounded drift and 0.01 for excessive drift,
so a worst-case threshold rejects nothing. This concerns rho, not S: it does not show that no worst-case bound for S
exists, and none is derived here. The threshold c for S is calibrated from
the law of Z, by simulation or from authentic data. *Status: numerical
remark.*

**Proposition 4.3 (the low quantile follows the harmonic mean).** Fix q and
take the Gaussian surrogate w ~ N(0, s^2 I). Here D_t stands for the
first-order discrepancy |g_t . w|. Then g_t . w = sigma_t Z_t with
sigma_t = s |g_t| > 0 (assumed) and Z_t ~ N(0, 1); the Z_t are dependent
across t. Define
the mean-field distribution and quantile

    F(x) = E[ (1/T) #{ t : D_t <= x } ] = (1/T) sum_t ( 2 Phi(x / sigma_t) - 1 ),    x_p = F^{-1}(p),

and the harmonic mean H = T / sum_t (1 / sigma_t). Then:

- (a) x_p >= p sqrt(pi/2) H for every p in (0, 1).
- (b) x_p / ( p sqrt(pi/2) H ) -> 1 as p -> 0.
- (c) If x_p <= min_t sigma_t, then x_p <= e^{1/2} p sqrt(pi/2) H.
- (d) Raising the scales of any set S of invariants with w_S < 1 multiplies H
  by a factor in [1, 1/(1 - w_S)], where w_S = sum_{t in S} sigma_t^{-1} / sum_t sigma_t^{-1}.
  Ill-conditioned invariants (large sigma_t, small w_S) barely move H; they
  dominate the RMS.

*Status: proved* for the Gaussian surrogate. Checks: V5, numerical.
- (a) holds to 1e-11 for p from 0.1 down to 1e-7.
- (b) the ratio goes 1.1453, 1.0197, 1.0007, 1.000007, then 1.0000 at p = 1e-5 to 1e-7. (At p = 0.1, (c) does not apply: x_0.1 > min sigma_t.)
- (d) the bound is respected while the RMS moves 99.8x. Failure control: the bound fails if the smallest scales are inflated.
- The realised Q_0.1 over noise draws has median 1.020 x_0.1 and log-sd 0.214. That is numerical only, because the D_t are dependent and no law of large numbers is claimed.

*Proof.* Let phi(y) = 2 Phi(y) - 1 for y >= 0. Then phi(0) = 0 and
phi'(y) = sqrt(2/pi) e^{-y^2/2}.
- (a) phi' <= sqrt(2/pi), so phi(y) <= sqrt(2/pi) y. Hence F(x) <= sqrt(2/pi) x (1/T) sum_t 1/sigma_t = sqrt(2/pi) x / H. F is continuous and strictly increasing with F(0) = 0, so p = F(x_p) <= sqrt(2/pi) x_p / H.
- (b) e^{-t^2/2} >= 1 - t^2/2 gives phi(y) >= sqrt(2/pi)(y - y^3/6). So p = F(x_p) >= sqrt(2/pi)( x_p / H - x_p^3 m_3 / 6 ), with m_3 = (1/T) sum_t sigma_t^{-3}. Hence the ratio x_p / (p sqrt(pi/2) H) is at most 1 / (1 - x_p^2 m_3 H / 6). As p -> 0, x_p -> 0, so this tends to 1; with (a) the ratio tends to 1.
- (c) On [0, 1], phi' >= sqrt(2/pi) e^{-1/2}, so phi(y) >= sqrt(2/pi) e^{-1/2} y. If every x_p / sigma_t <= 1, then p >= sqrt(2/pi) e^{-1/2} x_p / H.
- (d) With sigma'_t >= sigma_t on S and equal elsewhere, 1/sigma'_t <= 1/sigma_t gives H' >= H. Dropping S from the sum gives 1/H' >= (1/T) sum_{t not in S} 1/sigma_t = (1 - w_S)/H. ∎

*Remark (non-Gaussian perturbations; sketch, not a full proof).* If g_t . w has a density f_t that is
continuous and positive at 0, then P(|g_t . w| <= x) = 2 f_t(0) x + o(x).
The same argument gives x_p ~ p / (2 mean_t f_t(0)), a harmonic-type mean of
the scales 1/(2 f_t(0)). The Gaussian case has 1/(2 f_t(0)) = sqrt(pi/2) sigma_t.

**Why not the mean or RMS.** Let one bracket of CR_t tend to 0 along a path of
finite length, while its other three brackets stay away from 0. Then
|log|CR_t|| -> infinity, so |g_t| cannot stay bounded along the path (a
function with bounded gradient changes by at most that bound times the path
length). Pencil gradients therefore blow up near coplanar quadruples. The rate
is about 1/|bracket|: row m of M^{-1} has norm vol_3(other three columns) /
|det M| (verified numerically to 1e-6 over 20,000 matrices; the reviewer
measured max_t |g_t| growing like 1/|det|). Moment statistics inherit that
blow-up; by (d) the harmonic mean does not. *Status: proved* (the blow-up);
the rate is *numerical*.

**Why p = 0.1.** By (a)-(c), the quantile tracks H closely only while x_p is
small compared with most sigma_t. That favours small p. Very small p uses few
invariants and is noisy. p = 0.1 sits in between. On simulations, p = 0.05,
0.1 and 0.25 give the same AUC to within 0.002, and the median (p = 0.5) is
worse (candidate d3 notes; *numerical*). p was fixed before the hidden test.

**Empirical performance** (*numerical*, `experiments/math_lab/RESULTS.md`).
Hidden test, n = 1000:
- AUC 0.985 and TPR at 5% FPR of 0.944, against projective registration at 0.972 and 0.865.
- Paired bootstrap 95% intervals for the gain: AUC +0.009 to +0.018; TPR +0.045 to +0.129.
- On Gaussian instead of cube-shaped point clouds: AUC 0.993, TPR 0.969.

### 4b. Standardised discrepancies

**Proposition 4.4.** Let q range over a compact set K of configurations of unit vectors q_k
(|q_k| = 1) on which every bracket used is bounded away from 0 and |g_t(q)| >= gamma > 0 for every
pencil t used. Let a_k = q_k + r u~_k and b_k = q_k + r v~_k with u~_k, v~_k i.i.d. N(0, I_4)
relative to the unit representatives (the unit chart), each
measurement possibly followed by per-point scalings and an orthogonal map (which leave the
unit-chart gradient norms unchanged), and let
m_0 = mu r with mu > 0 fixed. Then, as r -> 0, uniformly in q in K and for every pencil t,

    z_t = D_t / (m_0 (|g_t(A)| + |g_t(B)|))  converges in distribution to  |N(0,1)| / (sqrt(2) mu),

where the gradients are taken at the measured unit representatives in the unit chart. The limit
law, and hence the mean-field p-quantile Phi^{-1}((1 + p)/2) / (sqrt(2) mu) of Proposition 4.3,
does not depend on q or t. The raw D_t / r has the limit |g_t(q) . w|, w = u~ - v~, with scales
sqrt(2) |g_t(q)| that depend on q and t; its mean-field quantile follows the harmonic mean of
those scales (Proposition 4.3). *Status: proved* (marginal laws only; isotropic Gaussian noise in
the chart where the gradients are taken).

*Proof.* w ~ N(0, 2 I_{4n}), so g_t(q) . w ~ N(0, 2 |g_t(q)|^2) and
L_t = |g_t(q) . w| / (2 mu |g_t(q)|) has the law |N(0,1)| / (sqrt(2) mu) for every q and t. Fix
beta > 0 and work on E_beta = {|u~|, |v~| <= beta}; P(E_beta) does not depend on q. For r beta
small enough the perturbed configurations stay in a neighbourhood of K on which the brackets
remain bounded away from 0 and |g_t| >= gamma/2. On E_beta, the expansion of Theorem 4.1 gives
D_t = r |g_t(q) . w| + O(r^2) and smoothness of the gradient gives
|g_t(A)| + |g_t(B)| = 2 |g_t(q)| + O(r), so |z_t - L_t| <= C_beta r with C_beta depending only on
beta, gamma, mu, K and the (finite) set of pencils. Hence, for every eta > 0 and every
r <= r_0(beta) (the condition "r beta small enough"),
sup_q P(|z_t - L_t| > eta) <= P(not E_beta) + 1{C_beta r > eta}. So
limsup_{r -> 0} sup_q P(|z_t - L_t| > eta) <= P(not E_beta) for every beta, and letting
beta -> infinity gives 0. With F the continuous distribution function of the limit,
sup_q sup_x |P(z_t <= x) - F(x)| <= sup_q P(|z_t - L_t| > eta) + sup_x (F(x + eta) - F(x - eta)),
which tends to 0 as r -> 0 and then eta -> 0: the convergence is uniform in q. ∎

*The lower bound on |g_t| is a real hypothesis:* the gradient of a pencil cross-ratio can vanish
while its four brackets stay away from zero (an in-project reading found such a configuration),
and there z_t is not half-normal. It is part of K's definition here, and must be checked for a
sampled subfamily.

**Not claimed.** (i) The joint law of the z_t, and therefore the law of Q_p(z) or of any
combination, is not geometry-free; U1 reports realised medians as numerical only. (ii) m_0 must
match the noise level (m_0 = mu r): the statistic is not noise-free. (iii) **The evaluated
implementation is not covered.** In `experiments/eval/mv_window.py` (a) each reconstruction is in
its own projective frame, where |g_t| is not gauge-invariant (U1: a random projective frame per
measurement breaks the unit scale), (b) the noise is pixel noise propagated through shared
cameras and bundle adjustment, correlated across points rather than i.i.d. in a chart, (c) m_0 is a
median calibrated on rigid windows, and (d) the per-pencil p-values 2 Phibar(z_t) are combined by
the Cauchy combination, not by a quantile. Proposition 4.4 therefore does not explain why the
false-alarm rate of `acat` at nominal 5% stayed in or near the band under a change of scene
geometry (2.5% in the null calibration, `experiments/results/null_calibration_sizes.csv`; 4.5%
on fresh seeds in the pre-registered replication, `experiments/results/consistency_test_sizes.csv`).
That result is numerical. The same replication found that standardising in Tyler's canonical
frame does **not** transfer (97.5%).

Checks (U1, numerical; recorded run `verify_math_numerical_output.txt`, seed 20261021;
400 draws per geometry for the scale checks, 3 common draws for the first-order check). With
common random numbers the largest first-order relative error falls
by a factor of 10.0, 9.5 and 10.0 (cube, flat, far) when sigma falls from 1e-7 to 1e-8 (criterion
[8, 12]); with the gradients of another scene the factor is 1.00 (control). The smallest gradient
norm on the sampled families is 0.301, 0.566, 2.67 (reported, not a test: the hypothesis
|g_t| >= gamma cannot fail detectably on generic configurations). With the gradient at q, the 95th
percentile over pencils of |E z^2 - 1| is 0.124, 0.137 and 0.115, and the dispersion around the
median 0.128, 0.158 and 0.119; both must be below the Monte-Carlo tolerance 0.177 (2.5
sqrt(2/400), derived). Control: noise of variance 3 or 1/3
by point (not i.i.d.) gives unequal per-pencil scales, dispersion 1.247, 0.717 and 0.777
(criterion > 0.5); weaker heteroscedasticity was not tried, and the flat and far margins are
small. The implemented `standardized_discrepancies`, with gradients at the measured points and
m_0 = sigma/sqrt 2, gives 0.123, 0.136 and 0.116; with each measurement in its own random
projective frame, 0.863-0.877 (control). The raw statistic's mean-field Q_0.1 varies by a factor
of 9.8 across the three geometries (the standardised one is 0.1257 in each by the proof; not
computed). Realised medians of Q_0.1(z): 0.130, 0.137, 0.135 (numerical; dependence not covered).

*Recorded failure.* The first version of U1 compared the largest first-order error over all
pencils with a fixed threshold and the per-pencil scales with thresholds that ignored
Monte-Carlo error; it failed in quick mode. The statement was not at fault. That version was
never committed; a reconstruction of it in `mutation_tests_math.py` fails.

### 4c. A size bound for the low quantile under any dependence

**Proposition 4.5.** Let X_1..X_T be random variables with any joint law, put
s_t(c) = P(|X_t| > c), let Z be the np.quantile (linear interpolation) of {|X_t|} at level p,
and k = ceil((T - 1) p) + 1. For every c > 0 and every set R of k indices,

    P(Z > c) <= min( sum_{t in R} s_t(c),   sum_t s_t(c) / (T - k + 1) ).

*Status: proved* (any marginal laws, any dependence).

*Proof.* The interpolated quantile at 0-based position h = (T - 1) p is at most the order
statistic at position ceil(h), the k-th smallest value. If it exceeds c, then (i) every set of k
values has a member above c, in particular R, which gives the union bound; and (ii) at least
T - k + 1 values exceed c, so Markov's inequality on N_c = #{t : |X_t| > c} gives the second
term. ∎

*Instances.* (a) Gaussian surrogate (Proposition 4.3): X_t = g_t . w, w ~ N(0, s^2 I), gives
s_t(c) = 2 Phibar(c / sigma_t), sigma_t = s |g_t|. (b) Standardised (Proposition 4.4 in the
limit, scale tau = 1/(sqrt(2) mu)): s_t(c) = 2 Phibar(c / tau) for every t, so
P(Z > c) <= 2 T Phibar(c / tau) / (T - k + 1), which does not depend on the geometry. (c) The
bounded perturbations of Theorem 4.1: s_t is the tail of |g_t . w| under that law, to first order.

*Sharpness.* Among bounds valid for every dependence, (b) is close to the best possible: if all
pencils move together, Z = tau |N(0,1)| and P(Z > c) = 2 Phibar(c / tau), which the bound exceeds
only by the factor T / (T - k + 1), about 1/(1 - p). For real pencil families it is very
conservative (numerical, U2).

Checks (U2, numerical; three geometries, 2,000 draws). Observed size 0.000 at the bounded
thresholds for alpha = 0.05 and 0.01: 11.3-12.7 times the median of Z for the raw statistic, and
the absolute thresholds c = 2.005 and 2.612 for the standardised one (about 15 and 20 times its
realised median, U1; derived). These checks have no power against moderate errors of
the bound: with the threshold more than ten times the median, a bound too large, or too small by
a few times, would also give size 0 (reasoning; not computed). Control: a binomial bound at the
mean tail probability (as if pencils were independent and equally scaled) fails (size
0.516-0.526 at nominal 0.05). Comonotone limit: if all pencils move together, Z = |N(0,1)| and
the bound exceeds the true size only by T/(T - k + 1); with 200,000 draws of |N(0,1)| the size at
the standardised threshold is 0.0453 against the closed form 2 Phibar(c) = 0.0450 (criterion
[0.04, 0.0515], the upper end 0.05 plus 3 standard errors). This checks that the threshold is
valid and at most 20% conservative in this limit, not Proposition 4.5 on simulated pencils. A bound
too small by a factor of 2 (alpha doubled) is caught (mutation record); a missing factor
T/(T - k + 1) is not (it gives size 0.050, still valid in this limit; derived).

---

## 5. Conditioning: kappa_J

**Definition.** For a 6-tuple with unit representatives,
kappa_J = max_k ||grad log|J_k| ||, with the gradient taken over all 24
coordinates. Implemented as
`algorithms/conditioning/scores.py::compute_invariant_noise_gain` (exact,
Lemma 3.1). It agrees with the finite-difference `forensics` version to 1e-9
relative.

**Proposition 5.1 (first-order noise law).** Let E ~ N(0, sigma^2 I) be added
to the 24 coordinates, and let Y_k = g_k . vec(E) be the first-order change
of log|J_k|, with g_k its gradient. Let M = max_k |Y_k|. Then

    Phi^{-1}(3/4) sigma kappa_J  <=  median(M)  <=  Phi^{-1}(11/12) sigma kappa_J,

that is, 0.674 sigma kappa_J <= median(M) <= 1.383 sigma kappa_J.
*Status: proved* (first order; the hypothesis is sigma kappa_J << 1). Check: V6,
numerical. At sigma = 10^-3 / kappa_J, over 150 configurations with 2000 draws
each, the full non-linear ratio ranges over [0.627, 1.072]. That is inside the
band up to 4 Monte-Carlo standard errors (0.07 at the low end). Failure
control: with sqrt(L(S)) in place of kappa_J, the ratio spreads over 394x,
against 1.71x for kappa_J. (Šidák's inequality would sharpen the upper
constant to 1.264; *cited*, not used.)

*Recorded failure.* A first version of V6 used one fixed sigma = 10^-5. For
near-degenerate tuples kappa_J reaches about 10^5, so sigma kappa_J was about
1, outside the first-order hypothesis, and the measured ratio fell to 0.32.
That run failed the check, as it should. It shows the limit of kappa_J: once
noise is no longer small relative to 1/kappa_J, the error grows more slowly
than linearly, because log|J| saturates, and kappa_J over-predicts it.

*Proof.*
- Lower bound: let k* attain kappa_J. Then Y_k* ~ N(0, sigma^2 kappa_J^2), and M >= |Y_k*|, so median(M) >= median|Y_k*| = Phi^{-1}(3/4) sigma kappa_J.
- Upper bound: each |g_k| <= kappa_J, so by the union bound P(M > x) <= sum_k P(|Y_k| > x) <= 6 (1 - Phi(x / (sigma kappa_J))). At x = Phi^{-1}(11/12) sigma kappa_J the right side is 1/2, so the median is at most x. ∎

*Consequence.* Across configurations, kappa_J fixes the noise amplification
up to a factor of at most 1.383/0.674 = 2.05. Real configurations range over
orders of magnitude, which is why kappa_J ranks them almost perfectly
(Spearman 0.995 against Monte-Carlo error in the benchmark; *numerical*).

**Proposition 5.2 (L(S) is only an upper proxy).** For unit representatives,

    kappa_J <= 8 / delta_J <= 8 / delta_min = 2 sqrt(2 L(S)),

where delta_J is the smallest of the 11 brackets used by J1-J3. There is no
bound in the other direction: L(S) can be infinite while kappa_J is finite.
*Status: proved.*

*Proof.*
- Row m of M^{-1} is orthogonal to the other three columns and has inner product 1 with column m. So its norm is 1 / dist(column m, span of the others) = vol_3(other three columns) / |det M|. By Hadamard, vol_3 <= 1 for unit columns, so each row has norm at most 1/|det M|, and ||M^{-1}||_F <= 2/|det M|.
- grad log|J_k| is a signed sum of four such gradients, so its norm is at most 8/delta_J. (Numerically, the largest kappa_J delta_J / 8 over 20,000 configurations, half pushed towards degeneracy, was 0.36.)
- For the converse, the brackets [1236], [1245], [1456] and [2356] (1-based) are not used by J1-J3. Make one of them zero by placing those four points on a plane, while the 11 used brackets stay nonzero. Then L(S) is infinite and kappa_J is finite. ∎

In V6 the smallest bracket is not one of J's 11 in 49 of 150 configurations
from the benchmark generator. This is one reason L(S) ranks
noise amplification worse (Spearman 0.83).

### 5b. A bound for log J beyond first order

**Theorem 5.3.** Let q_1..q_6 be unit vectors (the true unit representatives) and a_1..a_6 any
vectors with |a_i - q_i| <= delta. Put eps = (1 + delta)^4 - 1. For each invariant J_k with
brackets b_1..b_4 (computed from q): if eps < min_i |b_i|, then

    |log|J_k(a)| - log|J_k(q)|| <= sum_{i=1}^{4} -log(1 - eps / |b_i|),

and if moreover eps < delta_J, the smallest of the 11 brackets used by J1-J3, this is at most
-4 log(1 - eps / delta_J). *Status: proved.*

*Proof.* As in Theorem 6.1, multilinear expansion and Hadamard's inequality (|q_i| = 1,
|a_i - q_i| <= delta) give |b_i(a) - b_i(q)| <= eps; the a_i need not be unit vectors. If
eps < |b_i(q)|, then |b_i(a)| / |b_i(q)| lies in [1 - x, 1 + x] with x = eps/|b_i|, so
|log|b_i(a)| - log|b_i(q)|| <= max(-log(1 - x), log(1 + x)) = -log(1 - x). log|J_k| is a signed
sum of its four log-brackets; the triangle inequality gives the first bound and monotonicity in
|b_i| the second. ∎

*Remark.* For small delta the bound is about 16 delta / delta_J, the same order as the
first-order bound kappa_J sqrt(6) delta <= 8 sqrt(6) delta / delta_J (Proposition 5.2). It needs
no first-order hypothesis but requires eps < delta_J; beyond that the bound no longer excludes a
vanishing bracket, which is where log|J| saturates (the regime recorded under Proposition 5.1).

Check (U3, numerical): an adversarial search (Powell from 3 random starts and from the
steepest first-order ascent and descent of each log|J_k|; 40 configurations, half pushed towards
degeneracy; eps at 2%, 20% and 80% of the smallest of the 11 used brackets) finds no violation:
the largest |Delta log J| / bound is 0.676. Control: a fixed perturbation, each column of size
2.5 delta along the steepest first-order direction (hypothesis broken), exceeds the bound for
delta (ratio 2.71). Power: since the largest ratio found is 0.676, a bound too small by a factor
below 1/0.676 = 1.48 (derived) would not be detected. The mutation run uses the quick size
(4 configurations), where the search finds smaller ratios; it catches a bound halved, but with a
small margin, and the 1.48 applies to the full run only.
Numerical only: the largest ratio to the single largest bracket term is 1.20.

### 5c. Conditioning near the degeneracy strata

**Proposition 5.4 (small ball).** Let the columns of a 4x4 matrix M be i.i.d. uniform on S^3.
For 0 < d <= 1,

    (2/pi)(d sqrt(1 - d^2) + arcsin d) <= P(|det M| < d) <= (32/pi^2) d,

and P(|det M| < d)/d -> 32/pi^2 as d -> 0. Consequently, for six i.i.d. uniform unit
representatives, P(L(S) > l) = Theta(l^{-1/2}) and P(kappa_J > M) <= 2816/(pi^2 M) for M > 0.
*Status: proved.*

*Three facts about a uniform point x of S^3.* Write x = g/|g| with g ~ N(0, I_4). For m < 4,
x_1^2 + ... + x_m^2 = G_a / (G_a + G_b) with independent G_a = g_1^2 + ... + g_m^2 ~ Gamma(m/2) and
G_b ~ Gamma((4 - m)/2) (common scale); the change of variables (G_a, G_b) -> (S, U = G_a/S),
S = G_a + G_b, factorises the joint density, so U ~ Beta(m/2, (4 - m)/2). Hence
(i) m = 1: x_1 has density (2/pi) sqrt(1 - t^2) on [-1, 1];
(ii) m = 2: x_1^2 + x_2^2 is uniform on [0, 1], so rho = |(x_1, x_2)| has density 2 rho;
(iii) m = 3: rho = |(x_1, x_2, x_3)| has density (4/pi) rho^2 / sqrt(1 - rho^2).
(The Beta(1/2, 3/2) and Beta(3/2, 1/2) densities have normaliser B = pi/2.)

*Proof.* Write det M = <N, x_4> with N the generalised cross product of x_1, x_2, x_3,
|N| = vol_3(x_1, x_2, x_3) <= 1. Given x_1..x_3, t = <N/|N|, x_4> has the density of fact (i),
at most 2/pi, so P(|det| < d | x_1..x_3) <= (4/pi) d / |N| and P(|det| < d) <= (4/pi) d E[1/vol_3].
vol_3 = vol_2(x_1, x_2) dist(x_3, span(x_1, x_2)). Given (x_1, x_2), by rotation invariance the
distance has the law of rho in fact (ii), independently of (x_1, x_2): E[1/dist] = 2.
vol_2(x_1, x_2) = sqrt(1 - <x_1, x_2>^2) and <x_1, x_2> has the law of fact (i):
E[1/vol_2] = (2/pi) * 2 = 4/pi. Hence E[1/vol_3] = 8/pi and the upper bound (32/pi^2) d. For the
lower bound, |N| <= 1 gives P(|det| < d) >= P(|t| < d), the stated integral of fact (i). The limit
follows from P(|t| < x)/x -> 4/pi and dominated convergence (E[1/vol_3] < infinity). For
L(S) = 8/delta_min^2: P(L > l) = P(delta_min < d), d = sqrt(8/l), lies between P(|[1234]| < d)
and 15 times the upper bound. For kappa_J <= 8/delta_J (Proposition 5.2):
P(kappa_J > M) <= P(delta_J < 8/M) <= 11 (32/pi^2)(8/M). ∎

Checks (U4, numerical; 2,000,000 matrices for the determinant and the projection laws, 500,000
for the hyperplane control, 40,000 six-point configurations for the kappa_J and L(S) tails).
P(|det| < d) = 0.00323, 0.03203, 0.28729 at d = 10^-3, 10^-2, 10^-1, against upper bounds
0.00324, 0.03242, 0.32423 and lower bounds 0.00127, 0.01273, 0.12711; the criterion is "within 3
standard errors of the bounds". At d = 10^-3 the Monte-Carlo standard error is about 4.0e-5
(sqrt(p/N), derived), larger than the distance to the upper bound: the bound is asymptotically
exact as d -> 0, so the estimate is expected to sit at it, and this row cannot separate the bound
from a slightly smaller one. Projection laws: P(|x_1| < 0.5) = 0.60865 (theory 0.60900),
P(|(x_1, x_2)| < 0.5) = 0.24981 (0.25000), P(|(x_1, x_2, x_3)| < 0.9) = 0.46303 (0.46312). Mean
of 1/vol_3: 2.546 against 8/pi = 2.546 (no tolerance; infinite variance). K P(kappa_J > K) =
26.8, 29.4, 29.0 at K = 10^2, 10^3, 10^4, below 2816/pi^2 = 285.3 (derived; the bound exceeds 1
at K = 10^2). At K = 10^4 the standard error of K P is about 2.7 (derived). The bound divided by
20 is violated (control); the check therefore detects a bound too small by more than about
285.3/29.4 = 9.7 (derived, up to that sampling error), not a smaller error. P(L(S) > l) =
0.1213 at l = 10^6, inside [0.0036, 0.1376]. Control: columns concentrated near a hyperplane
exceed the upper bound (0.291 against 0.032).

**Proposition 5.5 (rate near the boundary stratum).** Let P_0 be a six-point configuration of
unit vectors with exactly one zero bracket b_0 = [I] (I a set of four indices), and P_d the unit
normalisation of P_0 + d V, where the derivative of d -> b_0(P_d) at d = 0 is nonzero. If I is
one of the 11 brackets used by J1-J3, then
c/d <= kappa_J(P_d) <= C/d for small d > 0; if I is one of [1236], [1245], [1456], [2356],
kappa_J(P_d) stays bounded. *Status: proved.*

*Proof.* b_0(P_d) is smooth in d, vanishes at d = 0 and has nonzero derivative there, so
c' d <= |b_0(P_d)| <= C'' d for small d; the other 14 brackets stay bounded away from 0. If b_0
is used, delta_J = |b_0(P_d)| for small d, and kappa_J <= 8/delta_J <= 8/(c' d)
(Proposition 5.2). If it is unused, delta_J stays bounded away from 0 and so does
kappa_J <= 8/delta_J. Lower bound (used case): b_0 belongs to some J_k. Each J_k has two axis
points, which appear in all four of its brackets, and four further points, which appear in
exactly two. Take x a non-axis point of I. The gradient of log|J_k| with respect to x is
+-row_x(M_{b_0}^{-1}) +- row_x(M_{b_1}^{-1}), where b_1 is the other bracket of J_k containing x.
The first row has norm vol_3(other three columns of b_0) / |b_0| >= c''' / (C'' d), because those
three columns are independent at P_0 (they lie in a nonzero bracket). The second stays bounded
because b_1(P_0) != 0. So |grad log|J_k|| >= c/d for small d. ∎

Check (U5, numerical): each of the 15 brackets made zero in turn (d = 1e-5, 1e-6, 1e-7):
kappa_J d converges to a positive constant for the 11 used brackets, kappa_J stays bounded for
the other 4, and the bounded criterion fails for every used bracket (control).

**Remark 5.6 (codimension and frequency).** For six i.i.d. uniform unit points, the probability
that all brackets of a given m-subset are below d appears to decay like d^(m-3), the codimension
in Proposition 6.2. U6 fits slopes 0.94, 1.93 and 2.96 for m = 4, 5, 6
(d = 0.02, 0.05, 0.1; 3,000,000 samples). *Status: numerical* (only m = 4 follows from
Proposition 5.4). Near-coplanar quadruples are therefore far more common than near-coplanar
quintuples or sextuples.

---

## 6. Further results on brackets, with proofs

**Theorem 6.1 (tau_exact).** Suppose every representative used to compute a
bracket is within distance delta of the true unit representative, possibly
from different measurements for different brackets. Then

    rho = max_k |Sigma_k| <= tau_exact(delta) = 3 eps (2 + eps),    eps = (1 + delta)^4 - 1.

The hypothesis concerns distance to the *unit true* representatives; the
measured representatives need not be unit. Renormalising a measured vector
can push it slightly beyond delta.

*Status: proved.* Check: V7, numerical, over 25 adversarial searches. With two
shared measurements the largest rho / tau_exact was 0.215. With independent
representatives for every bracket (the full hypothesis) it was 0.421. There is
no failure control, because a search can only fail to find a counterexample.

*Proof.*
1. **Each bracket moves by at most eps.** Expand det(q_1 + u_1, ..., q_4 + u_4) by multilinearity over the subsets S of replaced columns. By Hadamard, the term for S has absolute value at most prod_{k in S} |u_k| * prod_{k not in S} |q_k| <= delta^{|S|}. Summing over the nonempty S gives |det(q + u) - det(q)| <= (1 + delta)^4 - 1 = eps.
2. **Each product of two brackets moves by at most eps(2 + eps).** True brackets have absolute value at most 1 (Hadamard), and measured ones at most 1 + eps. So |x y - x'y'| <= |x||y - y'| + |y'||x - x'| <= eps + (1 + eps) eps = eps(2 + eps).
3. **Each Sigma_k has three such products and is 0 at the truth** (Theorem 2.5). So |Sigma_k| <= 3 eps(2 + eps). ∎

**Proposition 6.2 (Plücker relations and strata).** All 15 relations hold
identically (`verify_plucker.py`, symbolic). Sigma_1-3 are blind at
b* = e_1234 + e_1456 (checked exactly in `verify_plucker.py`). The stratum "m points
coplanar" has codimension m - 3. *Status:* the identities and b* are *computer-assisted* (exact).
For the codimension, the stratum is the image of
(plane, m points on it), which has 3 + 2m parameters. Its generic fibre is
finite, because four points in general position on a plane determine the
plane, so the stratum has dimension 3 + 2m and codimension 3m - (3 + 2m) =
m - 3 (*proved*). The Jacobian rank m - 3 of the 4x4 minors at a random
rational point (`verify_plucker.py`) is a consistency check: it gives only
codimension >= m - 3.

**Proposition 6.3 (Tyler whitening is equivariant).** Let x_i be unit
columns and let C solve Tyler's equation

    C = (4/n) sum_i x_i x_i^T / (x_i^T C^{-1} x_i),

normalised by trace. Let y_i = A x_i / |A x_i| with A invertible. Then
C' = A C A^T / c solves the equation for the y_i, for some c > 0. The
whitened sets C^{-1/2} x_i and C'^{-1/2} y_i, renormalised, differ by one
orthogonal map. Hence any statistic that is invariant under orthogonal maps,
L(S) included, is the same in every projective frame.
*Status: proved* given the cited uniqueness. Tyler (1987) and Kent and Tyler
(1988) give existence and uniqueness up to scale under a **strict** condition:
every proper subspace of dimension k contains fewer than k n / 4 of the points
(*cited, not read*; verify the exact form against the source).

The non-strict version is false. Counterexample (reviewer): X = (e1, e2, e3, e4)
with n = 4 meets "at most k n / 4" with equality, and every diagonal C solves
the equation. Convergence of the fixed-point iteration is assumed. Because
representatives carry arbitrary signs, the relation holds up to per-point
signs, which is harmless for |det|-based statistics.

Check: V8, numerical, residual 3.1e-12. Failure control: plain covariance
whitening of unit columns leaves a residual of 0.53.

*Proof.*
- Substitution: y^T C'^{-1} y = c x^T C^{-1} x / |Ax|^2 and y y^T = A x x^T A^T / |Ax|^2. So the right-hand side for the y_i is A [ (4/n) sum_i x_i x_i^T / (x_i^T C^{-1} x_i) ] A^T / c = A C A^T / c = C'.
- Let R = C'^{-1/2} A C^{1/2}. Then R R^T = C'^{-1/2} A C A^T C'^{-1/2} = c I, so R / sqrt(c) is orthogonal.
- C'^{-1/2} y_i is proportional to C'^{-1/2} A x_i = R (C^{-1/2} x_i), so after renormalisation the two whitened sets differ by the orthogonal map R / sqrt(c). ∎

---

## 7. Mismatch detection across views (classical, used as the engine)

**Model.** A track is observed in V views, x_v = pi(P_v X) + e_v with
e_v ~ N(0, sigma^2 I_2). Under H1 exactly one view v* is foreign. Let
SSE(all) be the least-squares triangulation residual over all views, and
SSE(-v) the same without view v.

**Proposition 7.1 (one-bad-view GLRT).** Assume sigma is known and V >= 3, so
that deleting any one view leaves a design of rank 3. The generalised
likelihood ratio statistic is

    Lambda = [ SSE(all) - min_v SSE(-v) ] / sigma^2.

Linearise around the true point with known cameras. Under H0, each
[SSE(all) - SSE(-v)] / sigma^2 is chi-square with 2 degrees of freedom (two
observations removed from a linear model with 3 parameters). Hence

    P_H0(Lambda > c) <= V e^{-c/2}.

*Status:* the distributional fact is the classical nested linear-model
result (*cited*: standard regression theory, Cochran's theorem). The union
bound is *proved*. The implemented method (candidate m1) estimates cameras
robustly (LMedS F, RANSAC + IRLS resection, Cauchy-weighted bundle
adjustment). Treating them as known is an *assumption*. Empirical hidden-test
AUC is 0.9996 (*numerical*). The reviewer's simulation agreed:
- V = 3 and V = 5 match chi-square(2).
- At V = 2 the statistic is chi-square(1), because the hypothesis V >= 3 fails; the bound still holds.

This part uses no CAPRI invariants.

*Proof of the bound.* P(max_v Q_v > c) <= sum_v P(Q_v > c), and a chi-square
variable with 2 degrees of freedom satisfies P(Q > c) = e^{-c/2}. ∎

---

## 8. Non-rigid localisation (conditional; not re-derived here)

Candidate n4 (`experiments/math_lab/candidates/n4/NOTES.md`) combines a
posterior over projective charts, under a uniform-cube point prior, with a
matched-subspace Bayes factor over spatial k-nearest-neighbour clusters. Its
derivation is the candidate's and has *not* been re-derived or checked in
this document.

The underlying fact is a geometric observation reported by n1, n2 and n4: a
15-dof projective fit absorbs most of a coherent cluster displacement, and
the detectable remainder lies in a small subspace orthogonal to the fit's
tangent space.

Scope: the method depends on the cluster prior. On non-clustered motion it
falls below RANSAC (AUC 0.55 against 0.68). Treat it as future work.

---

## 9. What this document does not claim

- That any result is new (see the Novelty section).
- That the drift test controls its size in the worst case. Calibration is
  empirical (Remark 4.2).
- That the realised quantile equals the mean-field quantile. That is
  numerical (V5).
- Any performance on real images. Every empirical number is synthetic.
- Any finite-noise size guarantee for the drift test. Proposition 4.5 bounds the size at first
  order under a stated marginal law and any dependence, and is too conservative to set a
  threshold. Proposition 4.4 is marginal and does not cover the evaluated statistic.
- (Theorem 3.2 beyond n = 6, 7, 8, 12 was listed here; Theorem 3.7 now proves it for every n.)

---

## 10. Computational implementation

This section states how the code evaluates the quantities above, and the two
facts the implementation relies on.

**Exact family.** `pencil_index(n)` lists the C(n,4) sorted quadruples and,
for each pencil value, the four quadruples of its brackets. The full family
therefore costs C(n,4) determinants plus four table lookups per value. For
n = 12 that is 495 determinants for 41,580 values. log|det| is computed on
unit columns, and per-point scalings cancel in every cross-ratio (Lemma 2.1).
An exactly zero bracket gives a non-finite value. Such pencils carry no
information, and the drift statistic drops them before taking the quantile.

**Lemma 10.1 (uniform sampling; proved).** Let (i, j, c, d, e, f) be uniform
over ordered 6-tuples of distinct points, and let X = |log|CR_ij(c,d;e,f)(A)| -
log|CR_ij(c,d;e,f)(B)||. Then X is distributed as D_t for t uniform over the
family of Section 2: one value per line, 4-set and pairing.

*Proof.* The unordered line {i, j} is uniform, and the 4-set {c, d, e, f} is
uniform given the line. Fix both. The 24 orderings of the 4-set give the six
values λ, 1/λ, 1 - λ, 1/(1 - λ), (λ - 1)/λ and λ/(λ - 1), four orderings each.
Since |log|1/μ|| = |log|μ||, the discrepancy is the same for μ and 1/μ. So the
six values fall into the three classes {λ, 1/λ}, {1 - λ, 1/(1 - λ)} and
{(λ - 1)/λ, λ/(λ - 1)}, each reached by 8 of the 24 orderings. The family's
three pairings take exactly one value from each class (Section 2). ∎

`sample_pencil_quads` draws the 6-tuples by rejection: it draws six indices
uniformly and redraws any row with a repeated index. Accepted rows are uniform
over ordered 6-tuples of distinct points, and the work is O(K) for K pencils.

**Proposition 10.2 (accuracy of the sampled quantile; cited).** Let F be the
distribution function of D_t over the family, and F_K the empirical
distribution of K independent draws (Lemma 10.1). By the
Dvoretzky-Kiefer-Wolfowitz inequality with Massart's constant,
P(sup |F_K - F| > ε) <= 2 exp(-2 K ε²). On the complementary event the
sample quantile lies between the family quantiles at levels p - ε and p + ε.
With K = 50,000 (`DEFAULT_MAX_PENCILS`) and ε = 0.01, the failure probability
is 2 e^{-10}, about 9.1e-5. So the sampled Q_0.1 lies between the family's
Q_0.09 and Q_0.11 except with probability below 1e-4. The code uses the full
family whenever it has at most K members, which covers every n <= 12.

**kappa_J.** By Lemma 3.1, the gradient of log|det M| with respect to column m
is row m of M^{-1}. The gradient of log|J_k| is therefore a signed sum of rows
of the inverses of its four bracket matrices. kappa_J is the largest Euclidean
norm of the three gradients, evaluated on unit representatives. The C++ kernel
(`cpp_core/src/pencil.cpp`) computes the 15 inverses of a tuple with Eigen
fixed-size 4x4 routines. It handles T tuples per call with the GIL released. A
tuple with a bracket of unit columns below `SINGULAR_BRACKET_TOL = 1e-12` in
absolute value gets kappa_J = inf. The C++ kernel and the NumPy reference
apply the same rule, so they agree to 1e-9 relative and give inf on the same
tuples (`tests/test_pencil.py`, pencil runtime benchmark).

**Selection.** The vectorized greedy step computes the gains of all feasible
tuples at once from a boolean membership matrix. It takes the first maximum,
with rows ordered by increasing score. This is the plain greedy rule with ties
broken towards the smaller score, so the (1 - 1/e) guarantee is unchanged.
pencil runtime benchmark checks that its output equals the per-tuple implementation.

**Measured cost** (pencil runtime benchmark, `pencil_runtime.csv`; one CPU, median of
5 runs; the per-tuple rows are single runs):

| Computation | Reference | Optimised |
|---|---|---|
| kappa_J, 10,000 tuples (n = 50) | 2.0 s (per-tuple loop) | 115 ms (NumPy batch), 4.0 ms (C++) |
| pencil statistic, n = 12 (exact, 41,580 values) | 2.2 ms (NumPy) | 1.6 ms (C++) |
| pencil statistic, n = 300 (50,000 sampled) | 136 ms (NumPy) | 17 ms (C++) |
| max-coverage selection, n = 30, 10,000 candidates | 2.1 s (per-tuple scoring) | 34 ms (batch) |

In every row the implementations give the same output (column
`Outputs_Agree`). Above n = 12 the cost of the pencil statistic does not
depend on n.

**Locking.** The mathematics is fixed as stated here. `tests/test_math_lock.py` runs the exact checks V1-V4 of `verify_pencil.py` (the
pencil identities, the Plücker factorisation, the rank 3n - 15 for n = 6 and 12) and W1-W5 of
`verify_math_exact.py` on small n (frame identities, general frames, rank, stabiliser, the
45 closed forms). A code change that breaks them fails the tests.

---

## 11. Checks, mutation record, Lean and readings of the results

### Mutation record (the checks can fail)

`experiments/symbolic/mutation_tests_math.py` breaks the code under test or a constant of a
statement and reruns the affected check, which must fail (recorded output
`mutation_tests_math_output.txt`). All ten were caught: a duplicated frame tuple and an
under-counted rank (W3), a swapped pairing (W2), mu dropped from the stabiliser equations (W4),
the sqrt(2) of the standardisation dropped (U1), the Theorem 5.3 bound halved (U3), 32/pi^2 halved (U4),
the wrong set of unused brackets (U5), the standardised bound too small by a factor of 2 (U2),
and, in `verify_frame_m2.m2`, the target of CR_12(3,4;5,k) changed from z/w to y/w (Macaulay2).
A reconstruction of the first version of U1 fails, as recorded. The numerical mutations run at
the quick sizes. If Macaulay2 is not installed the harness reports that mutation as not run and
fails.

### Lean

`formal/` (Lean 4 and Mathlib v4.35.0) proves, with no `sorry`: Lemma 2.1, Theorem 2.5,
Proposition 3.5, Lemma 3.6 (Cramer step included) and Theorem 3.7 for every n >= 5 (the
Jacobian and orbit argument); Corollary 3.8 (b), and (a) as orbit membership near P; Lemma 3.1
for brackets and Lemma 4.0; Theorem 4.1 (the uniform Taylor bound, the transfer to S and the
size and power in the limit); Proposition 4.4 in full (the Taylor step with gradients at the
measured unit representatives, the Gaussian assembly and the gauge clause); Proposition 4.5;
Propositions 5.1 and 5.2 (with an explicit unit configuration for the converse); Theorem 5.3;
Proposition 5.5 (c/d <= kappa_J <= C/d, and boundedness for an unused bracket); Theorem 6.1;
and Proposition 5.4 in full: facts (i) and (ii) (derived in Lean from volumes of cones in R^4,
Fubini over R^1 x R^3 and R^2 x R^2, not from the Gamma/Beta route above), the conditioning on
x_1..x_3, E[1/vol_3] = 8/pi, both bounds, the limit 32/pi^2 and both tails for six independent
uniform points. `bash formal/check.sh` builds it and checks twenty-three false variants, which
Lean must reject (recorded in `formal/check_output.txt`). Not formalized: fact (iii) of
Proposition 5.4 (used only by the numerical check of the projection laws); that the gauge in
Corollary 3.8 (a) can be taken near the identity; Lemma 2.2, Propositions
2.3-2.4, Corollary 2.6, Theorem 3.2 and Corollary 3.3 (superseded), Proposition 4.3, Remarks
3.9, 4.2 and 5.6, Propositions 6.2-6.3 and 7.1, and Section 10. The formalization found one
imprecision: the power statement of Theorem 4.1 needs Z to have no atom at c / f, not only at c
(corrected above). Details: `formal/README.md`.

### In-project readings of the results (not peer review)

Four readings in this project, each told to find errors; none edited the repository. They are
not outside review. Every finding below was fixed before the recorded runs.

*Reading 1 (Sections 3b, 5b, 5c).*
- Proposition 5.5 had an inequality in the wrong direction and omitted |b_0| <= C d.
- Theorem 5.3 needed "any vectors" (the a_i need not be unit vectors), and its second inequality
  needs eps < delta_J; the remark said a bracket "can vanish" where the bound no longer excludes it.
- Proposition 5.4 relied on facts labelled cited; they are now derived from Beta laws in the text.
- Checks: the U3 control was computed at the wrong place (a product of maxima over different
  invariants); the mutation record had no committed code; the controls of W3 (row drop), W5
  (shift by 1) and U5 could not fail; E[1/vol_3] in U4 has infinite variance; W5's output order
  was not deterministic.

*Reading 2 (Sections 4b, 4c).*
- Proposition 4.4 lacked the hypothesis |g_t| >= gamma > 0 (the gradient can vanish with all
  brackets nonzero); the Gaussian case needs a truncation argument (it was stated for bounded
  noise); its title claimed more than marginal laws.
- Proposition 4.5 was stated for Gaussian marginals only; it holds for any marginal law, which
  instance (c) needs.
- Checks: U1 did not test the implemented statistic; the U2 size check was nearly vacuous; the
  recorded failure had no committed code; a PASS line claimed more than it checked.

*Reading 3 (the fixes of readings 1 and 2).*
- A summary in this file said the first two readings found no error in the mathematics. That was
  wrong.
- The uniformity in q of Proposition 4.4 was asserted, not proved; the proof now bounds sup_q.
- Several fixes of the second round were themselves inadequate: the W5 control (one reciprocal
  formula had to disagree) now requires all 45 to disagree; the U1 control (errors with another
  scene's gradients) became a ratio criterion its control must fail; the U1 anisotropic control
  was mostly a change of level and became per-point heteroscedastic noise judged by dispersion;
  the Macaulay2 mutation, run by hand, moved into the harness; thresholds set after a first run
  meet a fresh seed (20261021).
- The U2 raw-bound check has no power against moderate errors; the U3 control and its power
  limit were described inaccurately; the README overstated the result; a cited 2.5% lacked its
  source file.

*Reading 4 (the fixes of reading 3; no must-fix finding).* It confirmed the uniformity proof
of Proposition 4.4 and every copied number. Fixed after it: Proposition 4.4 now states that the
q_k are unit vectors (the reading showed numerically that the standardisation fails without it)
and the proof states r <= r_0(beta) and the limsup step; the U2 comonotone check is described as a
check of the threshold with its actual criterion; the U3 mutation margin at the quick
size is stated; U4 states its sample sizes and its 3-standard-error criterion (its PASS line now
says so); the gamma check is reported, not called a test; the mutation harness fails when
Macaulay2 is missing; derived numbers are labelled; the Python outputs of the later results and the Lean
check output start with the commit and tool versions (the Macaulay2 output and the earlier
outputs do not); the seeds are listed in the pre-registrations.

*Reading 5 (the merge and the fixes of reading 4; no must-fix finding).* It confirmed the fixes,
every number of Sections 3b-5c and 11 against the outputs, the cross-references and the Lean
paragraph. Fixed after it: the U2 comonotone wording (it cannot detect a missing factor
T/(T - k + 1)) and the description of its mutation (alpha doubled); the scope of the provenance
statement; the remaining references to the draft and its item labels in the scripts and
documents; the commit to tag; the wording "statements and proofs unchanged"; the lists of what
Lean does not cover.
