from add_suffix import add_H_suffix_blocks
from fractions import Fraction
from wick.index import Idx
from wick.operator import FOperator, Tensor
from wick.expression import Term, Expression, AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, two_p, one_p, ep11, E1, E2, P1, EPS1, PE1, ketE1, ketE2, ketP1, ketP1E1, commute
from convenience_extra import P_dexit1, EPS_dexit1, PB1

i = Idx(0, "occ")
a = Idx(0, "vir")
j = Idx(1, "occ")
b = Idx(1, "vir")
k = Idx(2, "occ")
c = Idx(2, "vir")
l = Idx(3, "occ")
d = Idx(3, "vir")

# Define excitation Ansatz
T1 = E1("T1", ["occ"], ["vir"])
T2 = E2("T2", ["occ"], ["vir"])
S1 = P1("S1", ["nm"])
U11 = EPS1("U11", ["nm"], ["occ"], ["vir"])
T = T1 + T2 + S1 + U11

# Define deexcitation operators
L1 = E1("L1", ["vir"], ["occ"])
L2 = E2("L2", ["vir"], ["occ"])
LS1 = P_dexit1("LS1old", ["nm"])
LU11 = EPS_dexit1("LU11old", ["nm"], ["vir"], ["occ"])
L = L1 + L2 + LS1 + LU11

# Define hamiltonian with bosons                                                         
H1 = one_e("F", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True, compress=True)
Hp = two_p("w") + one_p("G")
#Hep = ep11("g", ["occ", "vir"], ["nm"], norder=True)
# g[p,q,x] is the tensor for p^+ q b_x (annihilation)
# g_boscre[p,q,x] is the tensor for p^+ q b_x^+
Hep = ep11("g", ["occ", "vir"], ["nm"], norder=True, name2="g_boscre")
H = H1 + H2 + Hp + Hep

simplify = True

# Construct Hbar
HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
Hbar = H + HT + Fraction('1/2')*HTT
Hbar += Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT

S0 = Hbar
E0 = apply_wick(S0)                                                                           
E0.resolve()                                                                                

# vo hole block (Operator with rank -1, therefore T^3 max)
L_expr = Expression([Term(1, [], [Tensor([a], "")], [FOperator(a, True)], [])])
R_expr = Expression([Term(1, [], [Tensor([i], "")], [FOperator(i, False)], [])])
expect = L_expr * (H - E0) * R_expr 
PT = commute(expect, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
PTTTTT = commute(PTTTT, T)
mid = expect + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out, simplify=simplify)
print("P_{vo hole} = ")
print(add_H_suffix_blocks(final._print_einsum('mom1_h_vo')))
print('***',flush=True)

# vv hole block (Operator with rank -2, therefore T^4 max)
L_expr = Expression([Term(1, [], [Tensor([a], "")], [FOperator(a, True)], [])])
R_expr = Expression([Term(1, [], [Tensor([b], "")], [FOperator(b, False)], [])])
expect = L_expr * (H - E0) * R_expr 
PT = commute(expect, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
PTTTTT = commute(PTTTT, T)
mid = expect + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT + Fraction('1/24')*PTTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out, simplify=simplify)
print("P_{vv hole} = ")
print(add_H_suffix_blocks(final._print_einsum('mom1_h_vv')))
print('***',flush=True)

# oo hole block (Operator with rank -2, therefore T^4 max)
L_expr = Expression([Term(1, [], [Tensor([i], "")], [FOperator(i, True)], [])])
R_expr = Expression([Term(1, [], [Tensor([j], "")], [FOperator(j, False)], [])])
expect = L_expr * (H - E0) * R_expr 
PT = commute(expect, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
PTTTTT = commute(PTTTT, T)
mid = expect + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT + Fraction('1/24')*PTTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out, simplify=simplify)
print("P_{oo hole} = ")
print(add_H_suffix_blocks(final._print_einsum('mom1_h_oo')))
print('***',flush=True)

# ov hole block (Operator with rank -3, therefore T^5 max)
L_expr = Expression([Term(1, [], [Tensor([i], "")], [FOperator(i, True)], [])])
R_expr = Expression([Term(1, [], [Tensor([a], "")], [FOperator(a, False)], [])])
expect = L_expr * (H - E0) * R_expr 
PT = commute(expect, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
PTTTTT = commute(PTTTT, T)
mid = expect + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT + Fraction('1/24')*PTTTT + Fraction('1/120')*PTTTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out, simplify=simplify)
print("P_{ov hole} = ")
print(add_H_suffix_blocks(final._print_einsum('mom1_h_ov')))
print('***',flush=True)
