-- experiments/symbolic/verify_plucker.m2
--
-- Macaulay2 version of verify_plucker.py (same checks, same numbering):
--   1. Sigma_1..Sigma_3 (the relations behind rho) vanish identically.
--   2. All 15 three-term Grassmann-Plucker relations vanish identically.
--   3. Blindness of Sigma_1..3: b* = e_1234 + e_1456 satisfies Sigma_1..3 but not the {1,4} relation.
--   4. Codimension of "m points of P^3 coplanar" = codim of the ideal of 4x4
--      minors of the 4 x m coordinate matrix (rank <= 3): expected m - 3.
--
-- The plane variables are eliminated by using the determinantal (minors)
-- ideal directly, in a ring of exactly the 4m coordinates involved. Putting
-- the plane coefficients a_1..a_4 into the ring and reporting 24 - dim(S/I)
-- in a 28-variable ring, with only m of the 6 points constrained, would not
-- give the stratum codimension.
--
-- Run: M2 --script experiments/symbolic/verify_plucker.m2

R = QQ[p_(1,1)..p_(6,4)];
pt = i -> matrix{{p_(i,1)}, {p_(i,2)}, {p_(i,3)}, {p_(i,4)}};
B = (i, j, k, l) -> det(pt(i) | pt(j) | pt(k) | pt(l));

-- 1. Sigma relations
Sigma1 = B(1,2,3,4)*B(1,2,5,6) - B(1,2,3,5)*B(1,2,4,6) + B(1,2,3,6)*B(1,2,4,5);
Sigma2 = B(1,3,4,5)*B(2,3,4,6) - B(1,3,4,6)*B(2,3,4,5) - B(1,2,3,4)*B(3,4,5,6);
Sigma3 = B(1,2,5,6)*B(3,4,5,6) - B(1,3,5,6)*B(2,4,5,6) + B(1,4,5,6)*B(2,3,5,6);
print("1. Sigma_1, Sigma_2, Sigma_3 vanish: " | toString({Sigma1 == 0, Sigma2 == 0, Sigma3 == 0}));

-- 2. All 15 relations {a,b}: [abcd][abef] - [abce][abdf] + [abcf][abde]
ok15 = apply(subsets(toList(1..6), 2), ab -> (
    a := ab#0; b := ab#1;
    r := select(toList(1..6), x -> x != a and x != b);
    c := r#0; d := r#1; e := r#2; f := r#3;
    B(a,b,c,d)*B(a,b,e,f) - B(a,b,c,e)*B(a,b,d,f) + B(a,b,c,f)*B(a,b,d,e) == 0));
print("2. All 15 Grassmann-Plucker relations vanish: " | toString(all(ok15, identity)));

-- 3. b* = e_1234 + e_1456 in bracket coordinates (signed by permutation parity)
bstar = (i, j, k, l) -> (
    s := sort {i, j, k, l};
    sgn := (-1)^(#select(subsets({i,j,k,l} / (x -> position(s, y -> y == x)), 2), pr -> pr#0 > pr#1));
    if s == {1,2,3,4} or s == {1,4,5,6} then sgn else 0);
-- note: subsets keeps list order, so the count above is the number of inversions
s1 = bstar(1,2,3,4)*bstar(1,2,5,6) - bstar(1,2,3,5)*bstar(1,2,4,6) + bstar(1,2,3,6)*bstar(1,2,4,5);
s2 = bstar(1,3,4,5)*bstar(2,3,4,6) - bstar(1,3,4,6)*bstar(2,3,4,5) - bstar(1,2,3,4)*bstar(3,4,5,6);
s3 = bstar(1,2,5,6)*bstar(3,4,5,6) - bstar(1,3,5,6)*bstar(2,4,5,6) + bstar(1,4,5,6)*bstar(2,3,5,6);
r14 = bstar(1,4,2,3)*bstar(1,4,5,6) - bstar(1,4,2,5)*bstar(1,4,3,6) + bstar(1,4,2,6)*bstar(1,4,3,5);
print("3. Sigma_1..3 at b*: " | toString({s1, s2, s3}) | "; {1,4} relation at b*: " | toString(r14));

-- 4. Stratum codimensions
scan({(4, "Boundary"), (5, "Compound"), (6, "Critical")}, mn -> (
    m := mn#0;
    S := QQ[x_(1,1)..x_(4,m)];
    X := transpose genericMatrix(S, x_(1,1), m, 4);
    I := minors(4, X);
    print("4. m=" | toString(m) | " (" | mn#1 | "): codimension " | toString(codim I) | ", expected " | toString(m - 3))));
