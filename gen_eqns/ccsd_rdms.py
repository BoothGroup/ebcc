from add_suffix import add_H_suffix_blocks
from fractions import Fraction
from wick.index import Idx
from wick.operator import FOperator, Tensor
from wick.expression import Term, Expression, AExpression
from wick.wick import apply_wick
from wick.convenience import E1, E2, commute

i = Idx(0, "occ")
a = Idx(0, "vir")
j = Idx(1, "occ")
b = Idx(1, "vir")
k = Idx(2, "occ")
c = Idx(2, "vir")
l = Idx(3, "occ")
d = Idx(4, "vir")

T1 = E1("T1", ["occ"], ["vir"])
T2 = E2("T2", ["occ"], ["vir"])
T = T1 + T2

L1 = E1("L1", ["vir"], ["occ"])
L2 = E2("L2", ["vir"], ["occ"])
L = L1 + L2

# ov block
operators = [FOperator(a, True), FOperator(i, False)]
pvo = Expression([Term(1, [], [Tensor([i, a], "")], operators, [])])

PT = commute(pvo, T)
PTT = commute(PT, T)
mid = pvo + PT + Fraction('1/2')*PTT
full = L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
print("P_{ov} = ")
print(final._print_einsum('dm1_ov'))

# vv block
operators = [FOperator(a, True), FOperator(b, False)]
pvv = Expression([Term(1, [], [Tensor([b, a], "")], operators, [])])

PT = commute(pvv, T)
PTT = commute(PT, T)
mid = pvv + PT + Fraction('1/2')*PTT
full = L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{vv} = ")
print(final._print_einsum('dm1_vv'))

# oo block
operators = [FOperator(j, False), FOperator(i, True)]
poo = Expression([Term(-1, [], [Tensor([j, i], "")], operators, [])])

PT = commute(poo, T)
PTT = commute(PT, T)
mid = poo + PT + Fraction('1/2')*PTT
full = L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{oo} = ")
print(final._print_einsum('dm1_oo'))

# vo block
operators = [FOperator(i, True), FOperator(a, False)]
pvo = Expression([Term(1, [], [Tensor([a, i], "")], operators, [])])

PT = commute(pvo, T)
PTT = commute(PT, T)
mid = pvo + PT + Fraction('1/2')*PTT
full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
final.sort_tensors()
print("P_{vo} = ")
print(final._print_einsum('dm1_vo'))

# oooo block of 2RDM
operators = [FOperator(i, True), FOperator(j, True), FOperator(k, False), FOperator(l, False)]
poooo = Expression([Term(1, [], [Tensor([i, j, k, l], "")], operators, [])])

PT = commute(poooo, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = poooo + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{oooo} = ")
print(final._print_einsum('dm2_oooo'))

# vvvv block of 2RDM
operators = [FOperator(a, True), FOperator(b, True), FOperator(c, False), FOperator(d, False)]
pvvvv = Expression([Term(1, [], [Tensor([a, b, c, d], "")], operators, [])])

PT = commute(pvvvv, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = pvvvv + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{vvvv} = ")
print(final._print_einsum('dm2_vvvv'))

# ovvv block of 2RDM
operators = [FOperator(i, True), FOperator(a, True), FOperator(b, False), FOperator(c, False)]
povvv = Expression([Term(1, [], [Tensor([i, a, b, c], "")], operators, [])])

PT = commute(povvv, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = povvv + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{ovvv} = ")
print(final._print_einsum('dm2_ovvv'))

# vovv block of 2RDM
operators = [FOperator(a, True), FOperator(i, True), FOperator(b, False), FOperator(c, False)]
pvovv = Expression([Term(1, [], [Tensor([a, i, b, c], "")], operators, [])])

PT = commute(pvovv, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = pvovv + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{vovv} = ")
print(final._print_einsum('dm2_vovv'))

# vvov block of 2RDM
operators = [FOperator(a, True), FOperator(b, True), FOperator(i, False), FOperator(c, False)]
pvvov = Expression([Term(1, [], [Tensor([a, b, i, c], "")], operators, [])])

PT = commute(pvvov, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = pvvov + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{vvov} = ")
print(final._print_einsum('dm2_vvov'))

# vvvo block of 2RDM
operators = [FOperator(a, True), FOperator(b, True), FOperator(c, False), FOperator(i, False)]
pvvvo = Expression([Term(1, [], [Tensor([a, b, c, i], "")], operators, [])])

PT = commute(pvvvo, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = pvvvo + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{vvvo} = ")
print(final._print_einsum('dm2_vvvo'))

# vooo block of 2RDM
operators = [FOperator(a, True), FOperator(i, True), FOperator(j, False), FOperator(k, False)]
pvooo = Expression([Term(1, [], [Tensor([a, i, j, k], "")], operators, [])])

PT = commute(pvooo, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = pvooo + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{vooo} = ")
print(final._print_einsum('dm2_vooo'))


# ovoo block of 2RDM
operators = [FOperator(i, True), FOperator(a, True), FOperator(j, False), FOperator(k, False)]
povoo = Expression([Term(1, [], [Tensor([i, a, j, k], "")], operators, [])])

PT = commute(povoo, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = povoo + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{ovoo} = ")
print(final._print_einsum('dm2_ovoo'))

# oovo block of 2RDM
operators = [FOperator(i, True), FOperator(j, True), FOperator(a, False), FOperator(k, False)]
poovo = Expression([Term(1, [], [Tensor([i, j, a, k], "")], operators, [])])

PT = commute(poovo, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = poovo + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{oovo} = ")
print(final._print_einsum('dm2_oovo'))

# ooov block of 2RDM
operators = [FOperator(i, True), FOperator(j, True), FOperator(k, False), FOperator(a, False)]
pooov = Expression([Term(1, [], [Tensor([i, j, k, a], "")], operators, [])])

PT = commute(pooov, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = pooov + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{ooov} = ")
print(final._print_einsum('dm2_ooov'))

# vvoo block of 2RDM
operators = [FOperator(a, True), FOperator(b, True), FOperator(i, False), FOperator(j, False)]
pvvoo = Expression([Term(1, [], [Tensor([a, b, i, j], "")], operators, [])])

PT = commute(pvvoo, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = pvvoo + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{vvoo} = ")
print(final._print_einsum('dm2_vvoo'))

# oovv block of 2RDM
operators = [FOperator(i, True), FOperator(j, True), FOperator(a, False), FOperator(b, False)]
poovv = Expression([Term(1, [], [Tensor([i, j, a, b], "")], operators, [])])

PT = commute(poovv, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = poovv + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{oovv} = ")
print(final._print_einsum('dm2_oovv'))

# vovo block of 2RDM
operators = [FOperator(a, True), FOperator(i, True), FOperator(b, False), FOperator(j, False)]
pvovo = Expression([Term(1, [], [Tensor([a, i, b, j], "")], operators, [])])

PT = commute(pvovo, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = pvovo + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{vovo} = ")
print(final._print_einsum('dm2_vovo'))

# voov block of 2RDM
operators = [FOperator(a, True), FOperator(i, True), FOperator(j, False), FOperator(b, False)]
pvoov = Expression([Term(1, [], [Tensor([a, i, j, b], "")], operators, [])])

PT = commute(pvoov, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = pvoov + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{voov} = ")
print(final._print_einsum('dm2_voov'))

# ovov block of 2RDM
operators = [FOperator(i, True), FOperator(a, True), FOperator(j, False), FOperator(b, False)]
povov = Expression([Term(1, [], [Tensor([i, a, j, b], "")], operators, [])])

PT = commute(povov, T)
PTT = commute(PT, T)
PTTT = commute(PTT, T)
PTTTT = commute(PTTT, T)
mid = povov + PT + Fraction('1/2')*PTT + Fraction('1/6')*PTTT
mid += Fraction('1/24')*PTTTT

full = mid + L*mid
out = apply_wick(full)
out.resolve()
final = AExpression(Ex=out)
#final = final.get_connected()
final.sort_tensors()
print("P_{ovov} = ")
print(final._print_einsum('dm2_ovov'))
