from add_suffix import add_H_suffix_blocks
from fractions import Fraction
from wick.index import Idx
from wick.operator import FOperator, Tensor
from wick.expression import Term, Expression, AExpression, ATerm
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, two_p, one_p, ep11, E1, E2, P1, EPS1, PE1, ketE1, ketE2, ketP1, ketP1E1, commute, braEip1, braEip2, braEea1, braEea2, Eea1, Eea2, Eip1, Eip2
from wick.convenience import ketEea1, ketEea2, ketEip1, ketEip2, braP1Eea1, ketP1Eea1, braP1Eip1, ketP1Eip1, EP1ea1, EP1ip1, braE1, braE2, braP1, braP1E1
from convenience_extra import P_dexit1, EPS_dexit1, PB1
import time

bosons = True 
if bosons:
    T1 = E1("T1", ["occ"], ["vir"])
    T2 = E2("T2", ["occ"], ["vir"])
    S1 = P1("S1", ["nm"])
    U11 = EPS1("U11", ["nm"], ["occ"], ["vir"])
    T = T1 + T2 + S1 + U11

    L1 = E1("L1", ["vir"], ["occ"])
    L2 = E2("L2", ["vir"], ["occ"])
    LS1 = P_dexit1("LS1old", ["nm"])
    LU11 = EPS_dexit1("LU11old", ["nm"], ["vir"], ["occ"])
    L = L1 + L2 + LS1 + LU11
else:
    T1 = E1("T1", ["occ"], ["vir"])
    T2 = E2("T2", ["occ"], ["vir"])
    T = T1 + T2

    L1 = E1("L1", ["vir"], ["occ"])
    L2 = E2("L2", ["vir"], ["occ"])
    L = L1 + L2

# Define hamiltonian
if bosons:
    H1 = one_e("F", ["occ", "vir"], norder=True)
    H2 = two_e("I", ["occ", "vir"], norder=True, compress=True)
    Hp = two_p("w") + one_p("G")
    #Hep = ep11("g", ["occ", "vir"], ["nm"], norder=True)
    # g[p,q,x] is the tensor for p^+ q b_x (annihilation)
    # g_boscre[p,q,x] is the tensor for p^+ q b_x^+
    Hep = ep11("g", ["occ", "vir"], ["nm"], norder=True, name2="g_boscre")
    H = H1 + H2 + Hp + Hep
else:
    H1 = one_e("F", ["occ", "vir"], norder=True)
    H2 = two_e("I", ["occ", "vir"], norder=True, compress=True)
    H = H1 + H2

fracs = [1.0, Fraction('1/2'), Fraction('1/6'), Fraction('1/24'), Fraction('1/120'), Fraction('1/720')]
HT = commute(H,T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
HTTTTT = commute(HTTTT, T)

simplify = True
gen_h = True 
moms = 'sp'   # 'sp' or 'dd'
ip_mom = False 
include_ref_proj = False
if include_ref_proj:
    assert(moms == 'dd')

# Generate ket state first 
tic_ket = time.perf_counter()
if moms == 'sp':
# First the occupied index
    inds = [Idx(0, "occ"), Idx(0, "vir")]
    for op_space in range(2):
        # First loop for occupied perturbation, second iteration for virtual

        R_expr = Expression([Term(1, [], [Tensor([inds[op_space]], "")], [FOperator(inds[op_space], not ip_mom)], [])])
        # Should never need to be more 
        RT = commute(R_expr, T)
        RTT = commute(RT, T)
        RTTT = commute(RTT, T)
        RTTTT = commute(RTTT, T)
        R_renom = R_expr + RT + fracs[1]*RTT + fracs[2]*RTTT

        # Space of left-hand projector
        spaces = []
        if ip_mom:
            spaces.append(braEip1("occ"))
            spaces.append(braEip2("occ", "occ", "vir"))
            if bosons:
                # Include bra projector onto 1h+1b
                spaces.append(braP1Eip1("nm", "occ"))
        else:
            spaces.append(braEea1("vir"))
            spaces.append(braEea2("occ", "vir", "vir"))
            if bosons:
                # Include bra projector onto 1p+1b, i.e. "nm", "vir"
                spaces.append(braP1Eea1("nm", "vir"))

        if ip_mom:
            if op_space == 0:
                print('Occupied Ket state for IP moment (final index is perturbation index)')
            else:
                print('Virtual Ket state for IP moment (final index is perturbation index)')
        else:
            if op_space == 0:
                print('Occupied Ket state for EA moment (final index is perturbation index)')
            else:
                print('Virtual Ket state for EA moment (final index is perturbation index)')
        
        for i, proj_term in enumerate(spaces):
            
            ket_term = proj_term * R_renom
            out = apply_wick(ket_term)
            out.resolve()
            final = AExpression(Ex=out, simplify=simplify)
            print(add_H_suffix_blocks(final._print_einsum('R0_{}'.format(i))), flush=True)
        print('')

elif moms == 'dd':
    for op_space in ['oo', 'ov', 'vo', 'vv']:
        if op_space == 'oo':
            R_expr = Expression([Term(1, [], [Tensor([Idx(0, "occ"), Idx(1, "occ")], "")], [FOperator(Idx(0, "occ"), True), FOperator(Idx(1, "occ"), False)], [])])
            print('Occupied-Occupied density perturbation for Ket state (final indices are perturbation indices?)')
        elif op_space == 'ov':
            R_expr = Expression([Term(1, [], [Tensor([Idx(0, "occ"), Idx(0, "vir")], "")], [FOperator(Idx(0, "occ"), True), FOperator(Idx(0, "vir"), False)], [])])
            print('Occupied-Virtual density perturbation for Ket state (final indices are perturbation indices?)')
        elif op_space == 'vo':
            R_expr = Expression([Term(1, [], [Tensor([Idx(0, "vir"), Idx(0, "occ")], "")], [FOperator(Idx(0, "vir"), True), FOperator(Idx(0, "occ"), False)], [])])
            print('Virtual-Occupied density perturbation for Ket state (final indices are perturbation indices?)')
        elif op_space == 'vv':
            R_expr = Expression([Term(1, [], [Tensor([Idx(0, "vir"), Idx(1, "vir")], "")], [FOperator(Idx(0, "vir"), True), FOperator(Idx(1, "vir"), False)], [])])
            print('Virtual-Virtual density perturbation for Ket state (final indices are perturbation indices?)')

        # Should never need to be more 
        RT = commute(R_expr, T)
        RTT = commute(RT, T)
        RTTT = commute(RTT, T)
        RTTTT = commute(RTTT, T)
        R_renom = R_expr + RT + fracs[1]*RTT + fracs[2]*RTTT

        # Space of left-hand projector
        spaces = []
        spaces.append(braE1("occ", "vir"))
        spaces.append(braE2("occ", "vir", "occ", "vir"))
        if bosons:
            spaces.append(braP1("nm"))
            spaces.append(braP1E1("nm", "occ", "vir"))

        if include_ref_proj:
            # Projection onto reference state
            ket_term = R_renom
            out = apply_wick(ket_term)
            out.resolve()
            final = AExpression(Ex=out, simplify=simplify)
            print(add_H_suffix_blocks(final._print_einsum('R0_ref')), flush=True)

        for i, proj_term in enumerate(spaces):
            ket_term = proj_term * R_renom
            out = apply_wick(ket_term)
            out.resolve()
            final = AExpression(Ex=out, simplify=simplify)
            print(add_H_suffix_blocks(final._print_einsum('R0_{}'.format(i))), flush=True)
        print('')
else:
    raise NotImplementedError
toc_ket = time.perf_counter()
print('Time to generate all ket states: ',toc_ket-tic_ket,flush=True)

# Construct Hbar (independent of the perturbation or projector). Would need to be increased for other ansatz
Hbar = H + HT + fracs[1]*HTT + fracs[2]*HTTT + fracs[3]*HTTTT + fracs[4]*HTTTTT
S0 = Hbar
E0 = apply_wick(S0)                                                                           
E0.resolve()                                                                                
if gen_h:
# Now, we want to find {\bar H}-E acting on an arbitrary R1 and R2
# Create arbitrary R in the right space
    spaces = [] # Spaces for the projector defining the space of the effective hamiltonian
    excitations = [] # Excitation classes on the RHS
    if moms == 'sp' and ip_mom:
        spaces.append(braEip1("occ"))
        spaces.append(braEip2("occ","occ","vir"))
        if bosons:
            spaces.append(braP1Eip1("nm", "occ"))

        # The space of the arbitrary function we are applying the hamiltonian to
        excitations.append(Eip1("R1", ["occ"]))
        excitations.append(Eip2("R2", ["occ"], ["vir"]))
        if bosons:
            excitations.append(EP1ip1("R3", ["nm"], ["occ"]))

    elif moms == 'sp' and not ip_mom:
        spaces.append(braEea1("vir"))
        spaces.append(braEea2("occ","vir","vir"))
        if bosons:
            spaces.append(braP1Eea1("nm", "vir"))

        excitations.append(Eea1("R1", ["vir"]))
        excitations.append(Eea2("R2", ["occ"], ["vir"]))
        if bosons:
            excitations.append(EP1ea1("R3", ["nm"], ["vir"]))
    elif moms == 'dd':
        spaces.append(braE1("occ", "vir"))
        spaces.append(braE2("occ", "vir", "occ", "vir"))
        if bosons:
            spaces.append(braP1("nm"))
            spaces.append(braP1E1("nm", "occ", "vir"))

        excitations.append(E1("R1", ["occ"], ["vir"]))
        excitations.append(E2("R2", ["occ"], ["vir"]))
        if bosons:
            excitations.append(P1("R3", ["nm"]))
            excitations.append(EPS1("R4", ["nm"], ["occ"], ["vir"]))
    else:
        raise NotImplementedError

    print('')
    print('Generating action of H-E on a single perturbation...')
    if include_ref_proj:
        # include projection onto HF state
        # Don't need to include excitations from reference state, as these will always produce zero
        comb_terms = []
        for j, R_term in enumerate(excitations):
            S = (Hbar - E0) * R_term
            out = apply_wick(S)
            out.resolve()
            comb_terms.append(AExpression(Ex=out, simplify=simplify))
        allterms = comb_terms[0].terms
        for k in range(1,len(comb_terms)):
            allterms = allterms + comb_terms[k].terms
        simp_terms = AExpression(terms=allterms, simplify=simplify)
        print(add_H_suffix_blocks(simp_terms._print_einsum('S_ref')), flush=True)
    
    for i, proj_term in enumerate(spaces):
        comb_terms = []
        for j, R_term in enumerate(excitations):
            S = proj_term * (Hbar - E0) * R_term
            print('Applying wick theorem for space {}/{} and R-term {}/{}...'.format(i+1,len(spaces),j+1,len(excitations)),flush=True)
            tic_wick = time.perf_counter()
            out = apply_wick(S)
            toc_wick = time.perf_counter()
            print('Wick theorem applied in {} sec'.format(toc_wick-tic_wick),flush=True)
            out.resolve()
            tic_simp = time.perf_counter()
            comb_terms.append(AExpression(Ex=out, simplify=simplify))
            toc_simp = time.perf_counter()
            print('Simplifying terms took {} sec'.format(toc_simp-tic_simp),flush=True)
        allterms = comb_terms[0].terms
        for k in range(1,len(comb_terms)):
            allterms = allterms + comb_terms[k].terms
        simp_terms = AExpression(terms=allterms, simplify=simplify)
        print(add_H_suffix_blocks(simp_terms._print_einsum('S_{}'.format(i))),flush=True)
    print('')

# Now to find the bra with which to do the dot product
print('Finding bra expression...(first index is operator index)',flush=True)
if moms == 'sp':
    inds = [Idx(0, "occ"), Idx(0, "vir")]
    for op_space in range(2):
        # Occupied operator in first loop, Virtual in second
        L_expr = Expression([Term(1, [], [Tensor([inds[op_space]], "")], [FOperator(inds[op_space], ip_mom)], [])])
        # Should never need to be more 
        L_opT = commute(L_expr, T)
        L_opTT = commute(L_opT, T)
        L_opTTT = commute(L_opTT, T)
        L_opTTTT = commute(L_opTTT, T)
        L_renom = L_expr + L_opT + fracs[1]*L_opTT + fracs[2]*L_opTTT + fracs[3]*L_opTTTT

        # Space of right-hand projector
        spaces = []
        if ip_mom:
            spaces.append(ketEip1("occ"))
            spaces.append(ketEip2("occ", "occ", "vir"))
            if bosons:
                # RH projector onto 1h+1b
                spaces.append(ketP1Eip1("nm", "occ"))
        else:
            spaces.append(ketEea1("vir"))
            spaces.append(ketEea2("occ", "vir", "vir"))
            if bosons:
                spaces.append(ketP1Eea1("nm", "occ"))
        
        if ip_mom:
            if op_space == 0:
                print('Occupied Bra state for IP moment (first index is perturbation index)')
            else:
                print('Virtual Bra state for IP moment (first index is perturbation index)')
        else:
            if op_space == 0:
                print('Occupied Bra state for EA moment (first index is perturbation index)')
            else:
                print('Virtual Bra state for EA moment (first index is perturbation index)')

        for i, proj_term in enumerate(spaces):
            bra_term = (L * L_renom + L_renom) * proj_term
            out = apply_wick(bra_term)
            out.resolve()
            final = AExpression(Ex=out)
            print(add_H_suffix_blocks(final._print_einsum('E_bra_{}'.format(i))),flush=True)
        print('')
elif moms == 'dd':
    for op_space in ['oo', 'ov', 'vo', 'vv']:
        if op_space == 'oo':
            L_expr = Expression([Term(1, [], [Tensor([Idx(0, "occ"), Idx(1, "occ")], "")], [FOperator(Idx(0, "occ"), True), FOperator(Idx(1, "occ"), False)], [])])
            print('Occupied-Occupied density perturbation for Bra state (first indices are perturbation indices?)')
        elif op_space == 'ov':
            L_expr = Expression([Term(1, [], [Tensor([Idx(0, "occ"), Idx(0, "vir")], "")], [FOperator(Idx(0, "occ"), True), FOperator(Idx(0, "vir"), False)], [])])
            print('Occupied-Virtual density perturbation for Bra state (first indices are perturbation indices?)')
        elif op_space == 'vo':
            L_expr = Expression([Term(1, [], [Tensor([Idx(0, "vir"), Idx(0, "occ")], "")], [FOperator(Idx(0, "vir"), True), FOperator(Idx(0, "occ"), False)], [])])
            print('Virtual-Occupied density perturbation for Bra state (first indices are perturbation indices?)')
        elif op_space == 'vv':
            L_expr = Expression([Term(1, [], [Tensor([Idx(0, "vir"), Idx(1, "vir")], "")], [FOperator(Idx(0, "vir"), True), FOperator(Idx(1, "vir"), False)], [])])
            print('Virtual-Virtual density perturbation for Bra state (first indices are perturbation indices?)')
        # Should never need to be more 
        L_opT = commute(L_expr, T)
        L_opTT = commute(L_opT, T)
        L_opTTT = commute(L_opTT, T)
        L_opTTTT = commute(L_opTTT, T)
        L_renom = L_expr + L_opT + fracs[1]*L_opTT + fracs[2]*L_opTTT + fracs[3]*L_opTTTT

        # Space of right-hand projector
        spaces = []
        spaces.append(ketE1("occ", "vir"))
        spaces.append(ketE2("occ", "vir", "occ", "vir"))
        if bosons:
            spaces.append(ketP1("nm"))
            spaces.append(ketP1E1("nm", "occ", "vir"))

        if include_ref_proj:
            bra_term = (L * L_renom + L_renom)
            out = apply_wick(bra_term)
            out.resolve()
            final = AExpression(Ex=out)
            print(add_H_suffix_blocks(final._print_einsum('E_bra_ref')), flush=True)
        
        for i, proj_term in enumerate(spaces):
            bra_term = (L * L_renom + L_renom) * proj_term
            out = apply_wick(bra_term)
            out.resolve()
            final = AExpression(Ex=out)
            print(add_H_suffix_blocks(final._print_einsum('E_bra_{}'.format(i))),flush=True)
        print('')
