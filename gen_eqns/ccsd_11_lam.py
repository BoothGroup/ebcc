import sys
sys.path.append('../')
from add_suffix import add_H_suffix_blocks
from fractions import Fraction
from wick.expression import AExpression
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, two_p, one_p, ep11, E1, E2, P1, EPS1, PE1, ketE1, ketE2, ketP1, ketP1E1, commute
from convenience_extra import P_dexit1, EPS_dexit1, PB1

# Define hamiltonian with bosons
H1 = one_e("F", ["occ", "vir"], norder=True)
H2 = two_e("I", ["occ", "vir"], norder=True)
Hp = two_p("w") + one_p("G")
Hep = ep11("g", ["occ", "vir"], ["nm"], norder=True)
H = H1 + H2 + Hp + Hep

# Define excitation Ansatz
T1 = E1("T1old", ["occ"], ["vir"])
T2 = E2("T2old", ["occ"], ["vir"])
S1 = P1("S1old", ["nm"])
U11 = EPS1("U11old", ["nm"], ["occ"], ["vir"])
T = T1 + T2 + S1 + U11

# Define deexcitation operators
L1 = E1("L1old", ["vir"], ["occ"])
L2 = E2("L2old", ["vir"], ["occ"])
LS1 = P_dexit1("LS1old", ["nm"])
LU11 = EPS_dexit1("LU11old", ["nm"], ["vir"], ["occ"])
L = L1 + L2 + LS1 + LU11

# Construct ket projection spaces
ket_singles = ketE1("occ", "vir")
ket_doubles = ketE2("occ", "vir", "occ", "vir")
ket_1b = ketP1("nm")
ket_1b1e = ketP1E1("nm", "occ", "vir")

# Construct Hbar
HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
Hbar_4t = H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT
Hbar_3t = H + HT + Fraction('1/2')*HTT + Fraction('1/6')*HTTT
Hbar_2t = H + HT + Fraction('1/2')*HTT
Hbar_1t = H + HT
Hbar_0t = H

simplify = True

# ***** L1 residuals *****
print("Computing lambda residuals for 1 fermion deexcitation space...")
print("Computing connected piece not proportional to Lambda...")
# <0|Hbar|singles> (not proportional to lambda)
S = Hbar_1t*ket_singles
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out, simplify=simplify)
ex = ex.get_connected()  # Does this fundamentally change the expression?
ex.sort_tensors()
#print(ex._print_einsum('L1'),flush=True)
#print("",flush=True)
print(add_H_suffix_blocks(ex._print_einsum('L1')),flush=True)
print("",flush=True)

# <0|(L Hbar)_c|singles> (Connected pieces proportional to Lambda)
print("Computing connected piece proportional to Lambda...")
S1 = L*Hbar_3t*ket_singles
out1 = apply_wick(S1)
out1.resolve()
ex1 = AExpression(Ex=out1, simplify=simplify)
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
    S2 = L*ket_singles*Hbar_3t
    out2 = apply_wick(S2)
    out2.resolve()
    ex2 = AExpression(Ex=out2, simplify=simplify)
    ex2 = ex2.get_connected()
    ex2.sort_tensors()
    #print('Terms to remove: ')
    #print(ex2._print_einsum('ToRemove'),flush=True)
    ex1 = ex1 - ex2
#print(ex1._print_einsum('L1'),flush=True)
#print("****",flush=True)
print(add_H_suffix_blocks(ex1._print_einsum('L1')),flush=True)
print('*****')

# ***** L2 residuals *****
print("Computing lambda residuals for 2 fermion deexcitation space...")
print("Computing connected piece not proportional to Lambda...")
# <0|Hbar|doubles> (not proportional to lambda)
S = Hbar_0t*ket_doubles
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out, simplify=simplify)
ex = ex.get_connected()
ex.sort_tensors()
print(add_H_suffix_blocks(ex._print_einsum('L2')))
print("",flush=True)

# <0|L Hbar|doubles> (Connected pieces proportional to Lambda)
print("Computing connected piece proportional to Lambda...")
S = L*Hbar_2t*ket_doubles
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out, simplify=simplify)
ex = ex.get_connected()
ex.sort_tensors()
print(add_H_suffix_blocks(ex._print_einsum('L2')))
print("",flush=True)

print("Computing disconnected piece projecting onto single fermionic excitations...")
# Disonnected pieces proportional to Lambda
# Don't apply get_connected here.
# Form projector onto singles space
P1 = PE1("occ", "vir")
S = Hbar_1t * P1 * L * ket_doubles
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out, simplify=simplify)
ex.sort_tensors()
print(add_H_suffix_blocks(ex._print_einsum('L2')),flush=True)
print("****")

# ***** B1 residuals *****
print("Computing lambda residuals for 1 boson deexcitation space...")
print("Computing connected piece not proportional to Lambda...")
# <0|Hbar|1b> (not proportional to lambda)
S = Hbar_1t * ket_1b
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out, simplify=simplify)
ex = ex.get_connected()
ex.sort_tensors()
print(add_H_suffix_blocks(ex._print_einsum('LS1')))
print("",flush=True)

# <0|L Hbar|1b> (Connected pieces proportional to Lambda)
print("Computing connected piece proportional to Lambda...")
S = L * Hbar_3t * ket_1b
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out, simplify=simplify)
ex = ex.get_connected()
ex.sort_tensors()
print(add_H_suffix_blocks(ex._print_einsum('LS1')))
print("*****",flush=True)

# ***** Boson-1FermExcit residuals *****
print("Computing lambda residuals for 1 boson + 1 fermion deexcitation space...")
print("Computing connected piece not proportional to Lambda...")
# <0|Hbar|EPS1> (not proportional to lambda)
S = Hbar_0t*ket_1b1e
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out, simplify=simplify)
ex = ex.get_connected()
ex.sort_tensors()
print(add_H_suffix_blocks(ex._print_einsum('LU11')))
print("",flush=True)

# <0|L Hbar|EPS1> (Connected pieces proportional to Lambda)
print("Computing connected piece proportional to Lambda...")
S = L*Hbar_2t*ket_1b1e
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out, simplify=simplify)
ex = ex.get_connected()
ex.sort_tensors()
print(add_H_suffix_blocks(ex._print_einsum('LU11')))
print("",flush=True)

# Disonnected pieces proportional to Lambda
# Don't apply get_connected here.
# Form projector onto singles space
print("Computing disconnected piece projecting onto single fermionic excitations...")
P1 = PE1("occ", "vir")
S = Hbar_1t*P1*L*ket_1b1e
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out, simplify=simplify)
ex.sort_tensors()
print(add_H_suffix_blocks(ex._print_einsum('LU11')),flush=True)
print("",flush=True)

# Also require projector onto 1 boson space (from L1 deexcitations)
print("Computing disconnected piece projecting onto single bosonic excitations...")
P1b = PB1("nm")
S = Hbar_1t*P1*L*ket_1b1e
out = apply_wick(S)
out.resolve()
ex = AExpression(Ex=out, simplify=simplify)
ex.sort_tensors()
print(add_H_suffix_blocks(ex._print_einsum('LU11')),flush=True)
print("",flush=True)

print("****")
