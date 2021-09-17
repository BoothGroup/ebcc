from add_suffix import add_H_suffix_blocks
from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, E1, E2, PE1, ketE1, ketE2, commute

# Define hamiltonian
H1 = one_e("F", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)
H = H1 + H2

# Define excitation Ansatz
T1 = E1("T1old", ["occ"], ["vir"])
T2 = E2("T2old", ["occ"], ["vir"])
T = T1 + T2

# Define deexcitation operators
L1 = E1("L1old", ["vir"], ["occ"])
L2 = E2("L2old", ["vir"], ["occ"])
L = L1 + L2

ket_doubles = ketE2("occ", "vir", "occ", "vir")
ket_singles = ketE1("occ", "vir")

# Construct Hbar
HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
Hbar = H + HT + Fraction('1/2')*HTT
Hbar += Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT

# ***** L1 residuals *****
# <0|Hbar|singles> (not proportional to lambda)
S = Hbar*ket_singles
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out)
ex = ex.get_connected()  # Does this fundamentally change the expression?
ex.sort_tensors()
#print(ex._print_einsum('L1'),flush=True)
#print("",flush=True)
print(add_H_suffix_blocks(ex._print_einsum('L1')),flush=True)
print("",flush=True)

# <0|(L Hbar)_c|singles> (Connected pieces proportional to Lambda)
S1 = L*S
out1 = apply_wick(S1)
out1.resolve()
ex1 = AExpression(Ex=out1)
ex1 = ex1.get_connected()
ex1.sort_tensors()
if False: 
    # It looks as though some of the terms in ex1 will sum to zero,
    # and can be identified and removed.
    # Subtract those terms that sum to zero
    # NOTE: If we don't remove these terms, then remember that the *update*
    # to the L1 will have to be modified slightly, to make sure that things
    # that involve the fock matrix contracted more than linearly with L1 will
    # have to include the full fock matrix.
    # See p. 371 of Shavitt and Bartlett for more details.  
    S2 = L*ket_singles*Hbar
    out2 = apply_wick(S2)
    out2.resolve()
    ex2 = AExpression(Ex=out2)
    ex2 = ex2.get_connected()
    ex2.sort_tensors()
    #print('Terms to remove: ')
    #print(ex2._print_einsum('ToRemove'),flush=True)
    ex1 = ex1 - ex2
#print(ex1._print_einsum('L1'),flush=True)
#print("****",flush=True)
print(add_H_suffix_blocks(ex1._print_einsum('L1')),flush=True)
print("****",flush=True)

# ***** L2 residuals *****
# <0|Hbar|doubles> (not proportional to lambda)
S = Hbar*ket_doubles
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out)
ex = ex.get_connected()
ex.sort_tensors()
print(add_H_suffix_blocks(ex._print_einsum('L2')))
print("****",flush=True)

# <0|L Hbar|doubles> (Connected pieces proportional to Lambda)
S = L*S
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out)
ex = ex.get_connected()
ex.sort_tensors()
print(add_H_suffix_blocks(ex._print_einsum('L2')))
print("",flush=True)

# Disonnected pieces proportional to Lambda
# Don't apply get_connected here.
# Form projector onto singles space
P1 = PE1("occ", "vir")
#S = (H + HT)*P1*L*ket
S = Hbar*P1*L*ket_doubles
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out)
ex.sort_tensors()
print(add_H_suffix_blocks(ex._print_einsum('L2')),flush=True)
