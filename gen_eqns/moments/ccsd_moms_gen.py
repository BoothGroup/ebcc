from add_suffix import add_H_suffix_blocks
from fractions import Fraction
from wick.index import Idx
from wick.operator import FOperator, Tensor
from wick.expression import Term, Expression, AExpression, ATerm
from wick.wick import apply_wick
from wick.convenience import one_e, two_e, two_p, one_p, ep11, E1, E2, P1, EPS1, PE1, ketE1, ketE2, ketP1, ketP1E1, commute
from convenience_extra import P_dexit1, EPS_dexit1, PB1

def mom_expression(ov_string, T, L_terms, L_term_rank, H_terms, H_term_rank, simplify=True, ip_mom=True):

    if len(ov_string) == 4:
        print('Finding dd expression for ov string: ',ov_string)
        output_string = 'mom1_dd_'+ov_string
        pert_rank = 0

        occ_inds = [Idx(i, "occ") for i in range(4)]
        vir_inds = [Idx(a, "vir") for a in range(4)]

        i = 0
        a = 0
        # Left density perturbation 
        if ov_string[:2] == 'oo':
            L_expr = Expression([Term(1, [], [Tensor([occ_inds[i], occ_inds[i+1]], "")], [FOperator(occ_inds[i], True), FOperator(occ_inds[i+1], False)], [])])
            i += 2
        elif ov_string[:2] == 'vv':
            L_expr = Expression([Term(1, [], [Tensor([vir_inds[a], vir_inds[a+1]], "")], [FOperator(vir_inds[a], True), FOperator(vir_inds[a+1], False)], [])])
            a += 2
        elif ov_string[:2] == 'ov':
            L_expr = Expression([Term(1, [], [Tensor([occ_inds[i], vir_inds[a]], "")], [FOperator(occ_inds[i], True), FOperator(vir_inds[a], False)], [])])
            a += 1
            i += 1
            pert_rank -= 1
        elif ov_string[:2] == 'vo':
            L_expr = Expression([Term(1, [], [Tensor([vir_inds[a], occ_inds[i]], "")], [FOperator(vir_inds[a], True), FOperator(occ_inds[i], False)], [])])
            a += 1
            i += 1
            pert_rank += 1

        # Right density perturbation
        if ov_string[2:] == 'oo':
            R_expr = Expression([Term(1, [], [Tensor([occ_inds[i], occ_inds[i+1]], "")], [FOperator(occ_inds[i], True), FOperator(occ_inds[i+1], False)], [])])
            i += 2
        elif ov_string[2:] == 'vv':
            R_expr = Expression([Term(1, [], [Tensor([vir_inds[a], vir_inds[a+1]], "")], [FOperator(vir_inds[a], True), FOperator(vir_inds[a+1], False)], [])])
            a += 2
        elif ov_string[2:] == 'ov':
            R_expr = Expression([Term(1, [], [Tensor([occ_inds[i], vir_inds[a]], "")], [FOperator(occ_inds[i], True), FOperator(vir_inds[a], False)], [])])
            a += 1
            i += 1
            pert_rank -= 1
        elif ov_string[2:] == 'vo':
            R_expr = Expression([Term(1, [], [Tensor([vir_inds[a], occ_inds[i]], "")], [FOperator(vir_inds[a], True), FOperator(occ_inds[i], False)], [])])
            a += 1
            i += 1
            pert_rank += 1

    elif len(ov_string) == 2:
        print('Finding sp moment expression for ov string: ',ov_string)
        pert_rank = 0
        if ov_string == 'ov':
            pert_rank = -1
        elif ov_string == 'vo':
            pert_rank = 1
        if not ip_mom:
            pert_rank *= -1
            output_string = 'mom1_p_'+ov_string
        else:
            output_string = 'mom1_h_'+ov_string

        occ_inds = [Idx(i, "occ") for i in range(2)]
        vir_inds = [Idx(a, "vir") for a in range(2)]

        i = 0
        a = 0
        # Left SP perturbation 
        if ov_string[0] == 'o':
            L_expr = Expression([Term(1, [], [Tensor([occ_inds[i]], "")], [FOperator(occ_inds[i], ip_mom)], [])])
            i += 1
        elif ov_string[0] == 'v':
            L_expr = Expression([Term(1, [], [Tensor([vir_inds[a]], "")], [FOperator(vir_inds[a], ip_mom)], [])])
            a += 1
        else:
            raise NotImplementedError

        # Right SP perturbation
        if ov_string[1] == 'o':
            R_expr = Expression([Term(1, [], [Tensor([occ_inds[i]], "")], [FOperator(occ_inds[i], not ip_mom)], [])])
            i += 1
        elif ov_string[1] == 'v':
            R_expr = Expression([Term(1, [], [Tensor([vir_inds[a]], "")], [FOperator(vir_inds[a], not ip_mom)], [])])
            a += 1
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    print('Rank of perturbation: ',pert_rank)

    largest_tpow = 0
    final = []
    fracs = [1.0, Fraction('1/2'), Fraction('1/6'), Fraction('1/24'), Fraction('1/120'), Fraction('1/720')]
    for i, lterm in enumerate(L_terms):
        ltermrank_f, ltermrank_b = L_term_rank[i]

        #print('For L item {}, rank change is: {}'.format(i,ltermrank_f))

        for j, hterm in enumerate(H_terms):
            htermrank_f, htermrank_b = H_term_rank[j]
            #print('For H item {}, rank change is: {}'.format(j,htermrank_f))

            rank_f = pert_rank + htermrank_f + ltermrank_f
            rank_b = htermrank_b + ltermrank_b
            # We need the number of T amplitudes so that both rank_f and rank_b are zero or positive
            # We assume that the minimum excitation is 1 in T (for both bosons and fermions)
            #print('Overall number of fermionic deexcitations: {}'.format(rank_f))
            if rank_f <= 0:
                ncommute_f = abs(rank_f)
            else:
                ncommute_f = 0
            if rank_b <= 0:
                ncommute_b = abs(rank_b)
            else:
                ncommute_b = 0
            commute_tot = ncommute_f + ncommute_b
            if commute_tot > largest_tpow:
                largest_tpow = commute_tot

            expr = [L_expr * hterm * R_expr]
            for t_ord in range(commute_tot):
                expr.append(commute(expr[t_ord], T))

            term = expr[0]
            for t_ord in range(commute_tot):
                term += fracs[t_ord]*expr[t_ord+1]

            term = lterm * term
            out = apply_wick(term)
            out.resolve()
            final.append(AExpression(Ex=out, simplify=simplify))
            print('Simplified expression for Lambda term {} and H term {}.'.format(i,j), flush=True)

    allterms = final[0].terms
    for i in range(1,len(final)):
        allterms = allterms + final[i].terms
    simp_terms = AExpression(terms=allterms, simplify=simplify)
    print(add_H_suffix_blocks(simp_terms._print_einsum(output_string)), flush=True)

    print('All terms computed and simplified separately. Max order of T was: {}'.format(largest_tpow), flush=True)
    return

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
    L_terms = [1.0, L1, L2, LS1, LU11]
    L_term_rank = [(0,0), (-1,0), (-2,0), (0,-1), (-1,-1)]
else:
    T1 = E1("T1", ["occ"], ["vir"])
    T2 = E2("T2", ["occ"], ["vir"])
    T = T1 + T2

    L1 = E1("L1", ["vir"], ["occ"])
    L2 = E2("L2", ["vir"], ["occ"])
    L_terms = [1.0, L1, L2]
    L_term_rank = [(0,0), (-1,0), (-2,0)]

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

simplify = True 

# Construct Hbar
HT = commute(H, T)
HTT = commute(HT, T)
Hbar = H + HT + Fraction('1/2')*HTT

S0 = Hbar
E0 = apply_wick(S0)                                                                           
E0.resolve()                                                                                

if bosons:
    H_terms = [-1.*E0, H1, H2, Hp, Hep]
    # H_term_rank is the maximum (fermionic, bosonic) de-excitations in each part
    H_term_rank = [(0,0), (-1,0), (-2,0), (0,-1), (-1,-1)]
else:
    H_terms = [-1.*E0, H1, H2]
    # H_term_rank is the maximum (fermionic, bosonic) de-excitations in each part
    H_term_rank = [(0,0), (-1,0), (-2,0)]

#ov_string = 'vvvo'
for ov_string in ['vo', 'vv', 'oo']:
    mom_expression(ov_string, T, L_terms, L_term_rank, H_terms, H_term_rank, simplify=simplify, ip_mom=True)
