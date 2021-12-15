from fractions import Fraction
from add_suffix import add_H_suffix_blocks
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, two_p, one_p, ep11, E1, E2, P1, EPS1, braE1, braE2, braP1, braP1E1, commute

# Define hamiltonian
H1 = one_e("F", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)
Hp = two_p("w") + one_p("G")
Hep = ep11("g", ["occ", "vir"], ["nm"], norder=True)
H = H1 + H2 + Hp + Hep

# Define projection spaces
bra_singles = braE1("occ", "vir")
bra_doubles = braE2("occ", "vir", "occ", "vir")
bra_1b = braP1("nm") 
bra_1b1e = braP1E1("nm", "occ", "vir")

# Define Ansatz
T1 = E1("T1old", ["occ"], ["vir"])
T2 = E2("T2old", ["occ"], ["vir"])
S1 = P1("S1old", ["nm"])
U11 = EPS1("U11old", ["nm"], ["occ"], ["vir"])
T = T1 + T2 + S1 + U11

# Construct Hbar
HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
Hbar = H + HT + Fraction('1/2')*HTT
Hbar += Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT

simplify = True 

# Compute energy expression
print('Energy expression: ')
out = apply_wick(Hbar)
out.resolve()
final = AExpression(Ex=out, simplify=simplify)
print(add_H_suffix_blocks(final._print_einsum('E')))
print('*******',flush=True)

# Projection onto singles
S = bra_singles * Hbar
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out, simplify=simplify)
print(add_H_suffix_blocks(final._print_einsum('T1')))
print('*******',flush=True)

# Projection onto doubles
S = bra_doubles * Hbar
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out, simplify=simplify)
print(add_H_suffix_blocks(final._print_einsum('T2')))
print('*******',flush=True)

# Projection onto single boson creation
S = bra_1b * Hbar
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out, simplify=simplify)
print(add_H_suffix_blocks(final._print_einsum('S1')))
print('*******',flush=True)

# Projection onto single boson creation + single ferm excitation
S = bra_1b1e * Hbar
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out, simplify=simplify)
print(add_H_suffix_blocks(final._print_einsum('U11')))
print('*******',flush=True)
