from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E1, E2, braE1, braE2, commute

# Define hamiltonian
H1 = one_e("F", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)
H = H1 + H2

# Define left projection spaces
bra_singles = braE1("occ", "vir")
bra_doubles = braE2("occ", "vir", "occ", "vir")

# Define Ansatz
T1 = E1("T1old", ["occ"], ["vir"])
T2 = E2("T2old", ["occ"], ["vir"])
T = T1 + T2

# Construct Hbar
HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
Hbar = H + HT + Fraction('1/2')*HTT
Hbar += Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT

# Compute energy expression
print('Energy expression: ')
out = apply_wick(Hbar)
out.resolve()
final = AExpression(Ex=out)
print(final._print_einsum('E'))
print('*******')

# Projection onto singles
S = bra_singles * Hbar
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final._print_einsum('T1'))
print('*******')

# Projection onto doubles
S = bra_doubles * Hbar
out = apply_wick(S)
out.resolve()
final = AExpression(Ex=out)
print(final._print_einsum('T2'))
