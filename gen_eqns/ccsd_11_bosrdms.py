import sys
sys.path.append('../')
from add_suffix import add_H_suffix_blocks
from fractions import Fraction
from wick.index import Idx
from wick.operator import FOperator, Tensor, BOperator
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
d = Idx(4, "vir")
I = Idx(5, "nm", fermion=False)
J = Idx(6, "nm", fermion=False)

# Define excitation Ansatz
T1 = E1("T1", ["occ"], ["vir"])
T2 = E2("T2", ["occ"], ["vir"])
S1 = P1("S1", ["nm"])
U11 = EPS1("U11", ["nm"], ["occ"], ["vir"])
T = T1 + T2 + S1 + U11

# Define deexcitation operators
L1 = E1("L1", ["vir"], ["occ"])
L2 = E2("L2", ["vir"], ["occ"])
LS1 = P_dexit1("LS1", ["nm"])
LU11 = EPS_dexit1("LU11", ["nm"], ["vir"], ["occ"])
L = L1 + L2 + LS1 + LU11

# Bosonic RDMs <b^+ b>
operators = [BOperator(I, True), BOperator(J, False)]
pbb = Expression([Term(1, [], [Tensor([I, J], "")], operators, [])])
PT = commute(pbb, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
mid = pbb + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
print("P_{bb} = ")
print(final._print_einsum('dm1_bb'))

# Check same as other way around
operators = [BOperator(I, False), BOperator(J, True)]
pbb = Expression([Term(1, [], [Tensor([I, J], "")], operators, [])])
PT = commute(pbb, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
mid = pbb + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
print("P_{bb}_permute = ")
print(final._print_einsum('dm1_bb_perm'))

# Coupling RDM <b^+  o^+ o>
operators = [BOperator(I, True), FOperator(i, True), FOperator(j, False)]
p = Expression([Term(1, [], [Tensor([I, i, j], "")], operators, [])])
PT = commute(p, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = p + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT + Fraction('1/24')*PTTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{b_oo} = ")
print(final._print_einsum('dm1_b_oo'))

# Coupling RDM <b^+  v^+ v>
operators = [BOperator(I, True), FOperator(a, True), FOperator(b, False)]
p = Expression([Term(1, [], [Tensor([I, a, b], "")], operators, [])])
PT = commute(p, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = p + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT + Fraction('1/24')*PTTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{b_vv} = ")
print(final._print_einsum('dm1_b_vv'))

# Coupling RDM <b^+  v^+ o>
operators = [BOperator(I, True), FOperator(a, True), FOperator(i, False)]
p = Expression([Term(1, [], [Tensor([I, a, i], "")], operators, [])])
PT = commute(p, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = p + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT + Fraction('1/24')*PTTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{b_vo} = ")
print(final._print_einsum('dm1_b_vo'))

# Coupling RDM <b^+  o^+ v>
operators = [BOperator(I, True), FOperator(i, True), FOperator(a, False)]
p = Expression([Term(1, [], [Tensor([I, i, a], "")], operators, [])])
PT = commute(p, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = p + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT + Fraction('1/24')*PTTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{b_ov} = ")
print(final._print_einsum('dm1_b_ov'))

# Coupling RDM <b  o^+ o>
operators = [BOperator(I, False), FOperator(i, True), FOperator(j, False)]
p = Expression([Term(1, [], [Tensor([I, i, j], "")], operators, [])])
PT = commute(p, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = p + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT + Fraction('1/24')*PTTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{boo} = ")
print(final._print_einsum('dm1_boo'))

# Coupling RDM <b  v^+ v>
operators = [BOperator(I, False), FOperator(a, True), FOperator(b, False)]
p = Expression([Term(1, [], [Tensor([I, a, b], "")], operators, [])])
PT = commute(p, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = p + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT + Fraction('1/24')*PTTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{bvv} = ")
print(final._print_einsum('dm1_bvv'))

# Coupling RDM <b  v^+ o>
operators = [BOperator(I, False), FOperator(a, True), FOperator(i, False)]
p = Expression([Term(1, [], [Tensor([I, a, i], "")], operators, [])])
PT = commute(p, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = p + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT + Fraction('1/24')*PTTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{bvo} = ")
print(final._print_einsum('dm1_bvo'))

# Coupling RDM <b  o^+ v>
operators = [BOperator(I, False), FOperator(i, True), FOperator(a, False)]
p = Expression([Term(1, [], [Tensor([I, i, a], "")], operators, [])])
PT = commute(p, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = p + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT + Fraction('1/24')*PTTTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{bov} = ")
print(final._print_einsum('dm1_bov'))
