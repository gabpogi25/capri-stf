-- experiments/symbolic/verify_frame_m2.m2
-- Item A1, checked in Macaulay2 (independently of SymPy, verify_math_exact.py W1):
-- with p1..p5 the standard projective frame and p6 = (x,y,z,w),
--   CR_12(3,4;5,6) = z/w,  CR_13(2,4;5,6) = y/w,  CR_23(1,4;5,6) = x/w,
-- where CR_ij(c,d;e,f) = [ijce][ijdf] / ([ijcf][ijde]).
-- Each identity is checked by clearing denominators; a wrong pairing must fail.
R = QQ[x,y,z,w];
P = {matrix{{1},{0},{0},{0}}, matrix{{0},{1},{0},{0}}, matrix{{0},{0},{1},{0}},
     matrix{{0},{0},{0},{1}}, matrix{{1},{1},{1},{1}}, matrix{{x},{y},{z},{w}}};
br = (a,b,c,d) -> det(P#(a-1) | P#(b-1) | P#(c-1) | P#(d-1));
-- CR_ij(c,d;e,f) = num/den; target t = tn/td; identity <=> num*td - den*tn == 0
frameCheck = (i,j,c,d,e,f,tn,td) -> (
    num := br(i,j,c,e) * br(i,j,d,f);
    den := br(i,j,c,f) * br(i,j,d,e);
    num * td - den * tn == 0 );
r1 = frameCheck(1,2,3,4,5,6, z, w);
r2 = frameCheck(1,3,2,4,5,6, y, w);
r3 = frameCheck(2,3,1,4,5,6, x, w);
ctrl = frameCheck(1,2,3,5,4,6, z, w);
print("CR_12(3,4;5,6) = z/w: " | toString r1);
print("CR_13(2,4;5,6) = y/w: " | toString r2);
print("CR_23(1,4;5,6) = x/w: " | toString r3);
print("control, CR_12(3,5;4,6) = z/w: " | toString ctrl | " (must be false)");
if r1 and r2 and r3 and not ctrl then print "ALL M2 FRAME CHECKS PASSED" else print "M2 FRAME CHECK FAILED";
exit 0
