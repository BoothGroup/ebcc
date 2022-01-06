from add_suffix import add_H_suffix_blocks
from fractions import Fraction
from wick.index import Idx
from wick.operator import FOperator, Tensor
from wick.expression import Term, Expression, AExpression
from wick.wick import apply_wick
from wick.convenience import E1, E2, commute, one_e, two_e

i = Idx(0, "occ")
a = Idx(0, "vir")
j = Idx(1, "occ")
b = Idx(1, "vir")
k = Idx(2, "occ")
c = Idx(2, "vir")
l = Idx(3, "occ")
d = Idx(3, "vir")

T1 = E1("T1", ["occ"], ["vir"])
T2 = E2("T2", ["occ"], ["vir"])
T = T1 + T2

L1 = E1("L1", ["vir"], ["occ"])
L2 = E2("L2", ["vir"], ["occ"])
L = L1 + L2

# Define hamiltonian
H1 = one_e("F", ["occ", "vir"], norder=True)
#H2 = two_e("I", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True, compress=True)
H = H1 + H2

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
