"""
Generate the CCSD-S-1-1 code.
"""

import sys

from qwick.index import Idx
from qwick.operator import Tensor as WTensor
from qwick.expression import AExpression
from qwick.convenience import PE1
from albert.qc._wick import *
from albert.qc.index import Index
from albert.tensor import Tensor
from albert.canon import canonicalise_indices

from ebcc.codegen.bootstrap_common import *

# Get the spin case
spin = sys.argv[1]

# Set up the code generators
code_generators = {
    "einsum": EinsumCodeGen(
        stdout=open(f"{spin[0].upper()}CCSD_S_1_1.py", "w"),
        name_generator=name_generators[spin],
        spin=spin,
    ),
}

# Write the preamble
for codegen in code_generators.values():
    codegen.preamble()

# Set up wick
rank = ("SD", "S", "S")
occs = [Idx(n, "occ") for n in range(6)]
virs = [Idx(n, "vir") for n in range(6)]
nms = [Idx(n, "nm", fermion=False) for n in range(6)]
index_dict = {
    **{k: o for k, o in zip(default_indices["o"][:len(occs)], occs)},
    **{k: v for k, v in zip(default_indices["v"][:len(virs)], virs)},
    **{k: n for k, n in zip(default_indices["b"][:len(nms)], nms)},
}

# Set up common wick objects
with Stopwatch("Common"):
    bra1e, bra2e, bra1b, bra1b1e = get_bra_spaces(rank=rank, occs=occs, virs=virs, nms=nms)
    ket1e, ket2e, ket1b, ket1b1e = get_ket_spaces(rank=rank, occs=occs, virs=virs, nms=nms)
    wick_h = get_hamiltonian(rank=rank)
    wick_t = get_excitation_ansatz(rank=rank, occs=occs, virs=virs, nms=nms)
    wick_l = get_deexcitation_ansatz(rank=rank, occs=occs, virs=virs, nms=nms)
    wick_hbars = construct_hbar(wick_h, wick_t, max_commutator=4)
    wick_hbar = wick_hbars[-1]

with Stopwatch("Energy"):
    # Get the energy contractions in wick format
    out = apply_wick(wick_hbar)
    out.resolve()
    expr = AExpression(Ex=out, simplify=True)

    # Get the energy in albert format
    expr = import_from_wick(expr)
    expr = spin_integrate(expr, spin)
    output = tuple(Tensor(name="e_cc") for _ in range(len(expr)))
    output, expr = optimise(output, expr, spin, strategy="exhaust")
    returns = (Tensor(name="e_cc"),)

    # Generate the energy code
    for codegen in code_generators.values():
        codegen(
            "energy",
            returns,
            output,
            expr,
        )

with Stopwatch("T amplitudes"):
    # Get the T1 contractions in wick format
    out = bra1e * wick_hbar
    out = apply_wick(out)
    out.resolve()
    expr_t1 = AExpression(Ex=out, simplify=True)

    # Get the T2 contractions in wick format
    out = bra2e * wick_hbar
    out = apply_wick(out)
    out.resolve()
    expr_t2 = AExpression(Ex=out, simplify=True)

    # Get the S1 contractions in wick format
    out = bra1b * wick_hbar
    out = apply_wick(out)
    out.resolve()
    expr_s1 = AExpression(Ex=out, simplify=True)

    # Get the U11 contractions in wick format
    out = bra1b1e * wick_hbar
    out = apply_wick(out)
    out.resolve()
    expr_u11 = AExpression(Ex=out, simplify=True)

    # Get the T amplitudes in albert format
    expr = []
    output = []
    returns = []
    terms = (expr_t1, expr_t2)
    for n in range(2):
        for index_spins in get_amplitude_spins(n + 1, spin):
            indices = default_indices["o"][: n + 1] + default_indices["v"][: n + 1]
            expr_n = import_from_wick(terms[n], index_spins=index_spins)
            expr_n = spin_integrate(expr_n, spin)
            output_n = get_t_amplitude_outputs(expr_n, f"t{n+1}new")
            returns_n = (Tensor(*indices, name=f"t{n+1}new"),)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)

    # Get the S amplitudes in albert format
    terms = (expr_s1,)
    for n in range(1):
        indices = default_indices["b"][: n + 1]
        expr_n = import_from_wick(terms[n])
        expr_n = spin_integrate(expr_n, spin)
        output_n = get_t_amplitude_outputs(expr_n, f"s{n+1}new")
        returns_n = (Tensor(*indices, name=f"s{n+1}new"),)
        expr.extend(expr_n)
        output.extend(output_n)
        returns.extend(returns_n)

    # Get the U amplitudes in albert format
    terms = ((expr_u11,),)
    for nb in range(1):
        for nf in range(1):
            for index_spins in get_amplitude_spins(nf + 1, spin):
                indices = default_indices["b"][: nb + 1] + default_indices["o"][: nf + 1] + default_indices["v"][: nf + 1]
                expr_n = import_from_wick(terms[nb][nf], index_spins=index_spins)
                expr_n = spin_integrate(expr_n, spin)
                output_n = get_t_amplitude_outputs(expr_n, f"u{nf+1}{nb+1}new")
                returns_n = (Tensor(*indices, name=f"u{nf+1}{nb+1}new"),)
                expr.extend(expr_n)
                output.extend(output_n)
                returns.extend(returns_n)

    # Generate the T amplitude code
    output, expr = optimise(output, expr, spin, strategy="trav")
    for name, codegen in code_generators.items():
        if name == "einsum":
            preamble = get_boson_einsum_preamble(spin)
            if spin == "uhf":
                preamble += "t1new = Namespace()\n"
                preamble += "t2new = Namespace()\n"
                preamble += "u11new = Namespace()"
            kwargs = {
                "preamble": preamble,
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "update_amps",
            returns,
            output,
            expr,
            **kwargs,
        )

with Stopwatch("L amplitudes"):
    # Get the L1 contractions in wick format: ⟨0|H̄|1e⟩
    out = wick_hbars[1] * ket1e
    out = apply_wick(out)
    out.resolve()
    expr1 = AExpression(Ex=out)
    expr1 = expr1.get_connected()
    expr1.sort_tensors()

    # Get the L1 contractions in wick format: ⟨0|[L, H̄]|1e⟩
    out = wick_l * wick_hbars[3] * ket1e
    out = apply_wick(out)
    out.resolve()
    expr2 = AExpression(Ex=out)
    expr2 = expr2.get_connected()
    expr2.sort_tensors()
    expr_l1 = expr1 + expr2

    # Get the L2 contractions in wick format: ⟨0|H̄|2e⟩
    out = wick_hbars[0] * ket2e
    out = apply_wick(out)
    out.resolve()
    expr1 = AExpression(Ex=out)
    expr1 = expr1.get_connected()
    expr1.sort_tensors()

    # Get the L2 contractions in wick format: ⟨0|L H̄|2e⟩
    out = wick_l * wick_hbars[2] * ket2e
    out = apply_wick(out)
    out.resolve()
    expr2 = AExpression(Ex=out)
    expr2 = expr2.get_connected()
    expr2.sort_tensors()

    # Get the L2 contractions in wick format: disconnected parts
    p1 = PE1("occ", "vir")
    out = wick_hbars[1] * p1 * wick_l * ket2e
    out = apply_wick(out)
    out.resolve()
    expr3 = AExpression(Ex=out)
    expr3.sort_tensors()
    expr_l2 = expr1 + expr2 + expr3

    # Get the LS1 contractions in wick format: ⟨0|H̄|1b⟩:
    out = wick_hbars[1] * ket1b
    out = apply_wick(out)
    out.resolve()
    expr1 = AExpression(Ex=out)
    expr1 = expr1.get_connected()
    expr1.sort_tensors()

    # Get the LS1 contractions in wick format: ⟨0|L H̄|1b⟩:
    out = wick_l * wick_hbars[3] * ket1b
    out = apply_wick(out)
    out.resolve()
    expr2 = AExpression(Ex=out)
    expr2 = expr2.get_connected()
    expr2.sort_tensors()
    expr_ls1 = expr1 + expr2

    # Get the LU11 contractions in wick format: ⟨0|H̄|1b1e⟩:
    out = wick_hbars[0] * ket1b1e
    out = apply_wick(out)
    out.resolve()
    expr1 = AExpression(Ex=out)
    expr1 = expr1.get_connected()
    expr1.sort_tensors()

    # Get the LU11 contractions in wick format: ⟨0|L H̄|1b1e⟩:
    out = wick_l * wick_hbars[2] * ket1b1e
    out = apply_wick(out)
    out.resolve()
    expr2 = AExpression(Ex=out)
    expr2 = expr2.get_connected()
    expr2.sort_tensors()

    # Get the LU11 contractions in wick format: disconnected parts, fermion projection
    p1 = PE1("occ", "vir")
    out = wick_hbars[1] * p1 * wick_l * ket1b1e
    out = apply_wick(out)
    out.resolve()
    expr3 = AExpression(Ex=out)
    expr3.sort_tensors()

    # Get the LU11 contractions in wick format: disconnected parts, boson projection
    p1b = PB1("nm")
    out = wick_hbars[1] * p1b * wick_l * ket1b1e
    out = apply_wick(out)
    out.resolve()
    expr4 = AExpression(Ex=out)
    expr4.sort_tensors()
    expr_lu11 = expr1 + expr2 + expr3 + expr4

    # Get the L amplitudes in albert format
    expr = []
    output = []
    returns = []
    terms = (expr_l1, expr_l2)
    for n in range(2):
        for index_spins in get_amplitude_spins(n + 1, spin):
            indices = default_indices["v"][: n + 1] + default_indices["o"][: n + 1]
            expr_n = import_from_wick(terms[n], index_spins=index_spins)
            expr_n = spin_integrate(expr_n, spin)
            output_n = get_l_amplitude_outputs(expr_n, f"l{n+1}new")
            returns_n = (Tensor(*indices, name=f"l{n+1}new"),)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)

    # Get the LS amplitudes in albert format
    terms = (expr_ls1,)
    for n in range(1):
        indices = default_indices["b"][: n + 1]
        expr_n = import_from_wick(terms[n])
        expr_n = spin_integrate(expr_n, spin)
        output_n = get_l_amplitude_outputs(expr_n, f"ls{n+1}new")
        returns_n = (Tensor(*indices, name=f"ls{n+1}new"),)
        expr.extend(expr_n)
        output.extend(output_n)
        returns.extend(returns_n)

    # Get the LU amplitudes in albert format
    terms = ((expr_lu11,),)
    for nb in range(1):
        for nf in range(1):
            for index_spins in get_amplitude_spins(nf + 1, spin):
                indices = default_indices["b"][: nb + 1] + default_indices["v"][: nf + 1] + default_indices["o"][: nf + 1]
                expr_n = import_from_wick(terms[nb][nf], index_spins=index_spins)
                expr_n = spin_integrate(expr_n, spin)
                output_n = get_l_amplitude_outputs(expr_n, f"lu{nf+1}{nb+1}new")
                returns_n = (Tensor(*indices, name=f"lu{nf+1}{nb+1}new"),)
                expr.extend(expr_n)
                output.extend(output_n)
                returns.extend(returns_n)

    # Generate the L amplitude code
    output, expr = optimise(output, expr, spin, strategy="trav")
    for name, codegen in code_generators.items():
        if name == "einsum":
            preamble = get_boson_einsum_preamble(spin)
            if spin == "uhf":
                preamble += "l1new = Namespace()\n"
                preamble += "l2new = Namespace()\n"
                preamble += "ls1new = Namespace()\n"
                preamble += "lu11new = Namespace()"
            kwargs = {
                "preamble": preamble,
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "update_lams",
            returns,
            output,
            expr,
            **kwargs,
        )

with Stopwatch("1RDM"):
    # Get the 1RDM contractions in wick format
    terms = {}
    for sectors, indices in [("oo", "ij"), ("ov", "ia"), ("vo", "ai"), ("vv", "ab")]:
        i = index_dict[indices[0]]
        j = index_dict[indices[1]]
        ops = [FOperator(j, True), FOperator(i, False)]
        p = Expression([Term(1, [], [WTensor([i, j], "")], ops, [])])
        out = bch(p, wick_t, max_commutator=2)[-1]
        out = out + wick_l * out
        out = apply_wick(out)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms[sectors, indices] = expr

    # Get the 1RDM in albert format
    expr = []
    output = []
    returns = []
    for sectors, indices in [("oo", "ij"), ("ov", "ia"), ("vo", "ai"), ("vv", "ab")]:
        for index_spins in get_density_spins(1, spin, indices):
            # wick canonicalisation is buggy:
            expr_n = [import_from_wick(t) for t in str(terms[sectors, indices]).split("\n")]
            expr_n = [canonicalise_indices(e, "ijklmn", "abcdef", ["x", "y", "z", "b0", "b1", "b2"]) for e in expr_n]
            if index_spins:
                expr_n = [e.map_indices({Index(i, space=o): Index(i, space=o, spin=index_spins[i]) for i, o in zip(indices, sectors)}) for e in expr_n]
            expr_n = sum([spin_integrate(e, spin) for e in expr_n], tuple())
            if spin == "uhf":
                expr_n = tuple(e * 0.5 for e in expr_n)
            expr_n = tuple(e for e in expr_n if not (isinstance(e, (int, float)) and e == 0))
            output_n = get_density_outputs(expr_n, f"d", indices)
            if index_spins:
                returns_n = (Tensor(*[Index(i, space=o, spin=index_spins[i]) for i, o in zip(indices, sectors)], name=f"d"),)
            else:
                returns_n = (Tensor(*[Index(i, space=o) for i, o in zip(indices, sectors)], name=f"d"),)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)
    output, expr = optimise(output, expr, spin, strategy="exhaust")

    # Generate the 1RDM code
    for name, codegen in code_generators.items():
        if name == "einsum":
            kwargs = {
                "preamble": get_density_einsum_preamble(1, spin),
                "postamble": get_density_einsum_postamble(1, spin),
            }
        else:
            kwargs = {}
        codegen(
            "make_rdm1_f",
            returns,
            output,
            expr,
            **kwargs,
        )

with Stopwatch("2RDM"):
    # Get the 2RDM contractions in wick format
    terms = {}
    for sectors, indices in [
        ("oooo", "ijkl"),
        ("ooov", "ijka"),
        ("oovo", "ijak"),
        ("ovoo", "iajk"),
        ("vooo", "aijk"),
        ("oovv", "ijab"),
        ("ovov", "iajb"),
        ("ovvo", "iabj"),
        ("voov", "aijb"),
        ("vovo", "aibj"),
        ("vvoo", "abij"),
        ("ovvv", "iabc"),
        ("vovv", "aibc"),
        ("vvov", "abic"),
        ("vvvo", "abci"),
        ("vvvv", "abcd"),
    ]:
        i = index_dict[indices[0]]
        j = index_dict[indices[1]]
        k = index_dict[indices[2]]
        l = index_dict[indices[3]]
        ops = [FOperator(l, True), FOperator(k, True), FOperator(i, False), FOperator(j, False)]
        p = Expression([Term(1, [], [WTensor([i, j, k, l], "")], ops, [])])
        out = bch(p, wick_t, max_commutator=4)[-1]
        out = out + wick_l * out
        out = apply_wick(out)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms[sectors, indices] = expr

    # Get the 2RDM in albert format
    expr = []
    output = []
    returns = []
    for sectors, indices in [
        ("oooo", "ijkl"),
        ("ooov", "ijka"),
        ("oovo", "ijak"),
        ("ovoo", "iajk"),
        ("vooo", "aijk"),
        ("oovv", "ijab"),
        ("ovov", "iajb"),
        ("ovvo", "iabj"),
        ("voov", "aijb"),
        ("vovo", "aibj"),
        ("vvoo", "abij"),
        ("ovvv", "iabc"),
        ("vovv", "aibc"),
        ("vvov", "abic"),
        ("vvvo", "abci"),
        ("vvvv", "abcd"),
    ]:
        for index_spins in get_density_spins(2, spin, indices):
            # wick canonicalisation is buggy:
            expr_n = [import_from_wick(t) for t in str(terms[sectors, indices]).split("\n")]
            expr_n = [canonicalise_indices(e, "ijklmn", "abcdef", ["x", "y", "z", "b0", "b1", "b2"]) for e in expr_n]
            if index_spins:
                expr_n = [e.map_indices({Index(i, space=o): Index(i, space=o, spin=index_spins[i]) for i, o in zip(indices, sectors)}) for e in expr_n]
            expr_n = sum([spin_integrate(e, spin) for e in expr_n], tuple())
            expr_n = tuple(e for e in expr_n if not (isinstance(e, (int, float)) and e == 0))
            output_n = get_density_outputs(expr_n, f"Γ", indices)
            if index_spins:
                returns_n = (Tensor(*[Index(i, space=o, spin=index_spins[i]) for i, o in zip(indices, sectors)], name=f"d"),)
            else:
                returns_n = (Tensor(*[Index(i, space=o) for i, o in zip(indices, sectors)], name=f"d"),)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)
    output, expr = optimise(output, expr, spin, strategy="trav")

    # Generate the 2RDM code
    for name, codegen in code_generators.items():
        if name == "einsum":
            kwargs = {
                "preamble": get_density_einsum_preamble(2, spin),
                "postamble": get_density_einsum_postamble(2, spin),
            }
        else:
            kwargs = {}
        codegen(
            "make_rdm2_f",
            returns,
            output,
            expr,
            **kwargs,
        )

with Stopwatch("BDM"):
    # Get the BDM contractions in wick format
    terms = {}
    for cre, index in [(True, "x"), (False, "x")]:
        x = index_dict[index]
        ops = [BOperator(x, cre)]
        p = Expression([Term(1, [], [WTensor([x], "")], ops, [])])
        out = bch(p, wick_t, max_commutator=3)[-1]
        out = out + wick_l * out
        out = apply_wick(out)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms[cre, index] = expr

    # Get the BDM in albert format
    expr = []
    output = []
    returns = []
    for cre, index in [(True, "x"), (False, "x")]:
        expr_n = import_from_wick(terms[cre, index])
        expr_n = spin_integrate(expr_n, spin)
        output_n = get_density_outputs(expr_n, f"dm_{'cre' if cre else 'des'}", (index,))
        returns_n = (Tensor(Index(index, space="b"), name=f"dm_{'cre' if cre else 'des'}"),)
        expr.extend(expr_n)
        output.extend(output_n)
        returns.extend(returns_n)
    output, expr = optimise(output, expr, spin, strategy="exhaust")

    # Generate the BDM code
    for name, codegen in code_generators.items():
        kwargs = {}
        codegen(
            "make_sing_b_dm",
            returns,
            output,
            expr,
            **kwargs,
        )

with Stopwatch("B1RDM"):
    # Get the B1RDM contractions in wick format
    x = index_dict["x"]
    y = index_dict["y"]
    ops = [BOperator(x, True), BOperator(y, False)]
    p = Expression([Term(1, [], [WTensor([x, y], "")], ops, [])])
    out = bch(p, wick_t, max_commutator=3)[-1]
    out = out + wick_l * out
    out = apply_wick(out)
    out.resolve()
    term = AExpression(Ex=out, simplify=True)

    # Get the B1RDM in albert format
    expr = []
    output = []
    returns = []
    expr_n = import_from_wick(term)
    expr_n = spin_integrate(expr_n, spin)
    output_n = get_density_outputs(expr_n, f"rdm1_b", ("x", "y"))
    returns_n = (Tensor(Index("x", space="b"), Index("y", space="b"), name=f"rdm1_b"),)
    expr.extend(expr_n)
    output.extend(output_n)
    returns.extend(returns_n)
    output, expr = optimise(output, expr, spin, strategy="exhaust")

    # Generate the B1RDM code
    for name, codegen in code_generators.items():
        kwargs = {}
        codegen(
            "make_rdm1_b",
            returns,
            output,
            expr,
            **kwargs,
        )

with Stopwatch("EBRDM"):
    # Get the EBRDM contractions in wick format
    terms = {}
    x = index_dict["x"]
    for sectors, indices in [("boo", "xij"), ("bov", "xia"), ("bvo", "xai"), ("bvv", "xab")]:
        for cre in (False, True):
            i = index_dict[indices[1]]
            j = index_dict[indices[2]]
            ops = [BOperator(x, cre), FOperator(j, True), FOperator(i, False)]
            p = Expression([Term(1, [], [WTensor([x, i, j], "")], ops, [])])
            out = bch(p, wick_t, max_commutator=2)[-1]
            out = out + wick_l * out
            out = apply_wick(out)
            out.resolve()
            expr = AExpression(Ex=out, simplify=True)
            terms[sectors, indices, cre] = expr

    # Get the EBRDM in albert format
    expr = []
    output = []
    returns = []
    for sectors, indices in [("boo", "xij"), ("bov", "xia"), ("bvo", "xai"), ("bvv", "xab")]:
        for cre in (False, True):
            for index_spins in get_density_spins(1, spin, indices):
                # wick canonicalisation is buggy:
                expr_n = [import_from_wick(t) for t in str(terms[sectors, indices, cre]).split("\n")]
                expr_n = [canonicalise_indices(e, "ijklmn", "abcdef", "xyz") for e in expr_n]
                if index_spins:
                    expr_n = [e.map_indices({Index(i, space=o): Index(i, space=o, spin=index_spins.get(i)) for i, o in zip(indices, sectors)}) for e in expr_n]
                expr_n = sum([spin_integrate(e, spin) for e in expr_n], tuple())
                if spin == "uhf":
                    expr_n = tuple(e * 0.5 for e in expr_n)
                expr_n = tuple(e for e in expr_n if not (isinstance(e, (int, float)) and e == 0))
                output_n = get_density_outputs(expr_n, f"rdm_eb_{'cre' if cre else 'des'}", indices)
                if index_spins:
                    returns_n = (Tensor(*[Index(i, space=o, spin=index_spins.get(i)) for i, o in zip(indices, sectors)], name=f"d"),)
                else:
                    returns_n = (Tensor(*[Index(i, space=o) for i, o in zip(indices, sectors)], name=f"d"),)
                expr.extend(expr_n)
                output.extend(output_n)
                returns.extend(returns_n)
    output, expr = optimise(output, expr, spin, strategy="exhaust")

    # Generate the EBRDM code
    for name, codegen in code_generators.items():
        if name == "einsum":
            preamble = get_density_einsum_preamble(1, spin)
            preamble = preamble.replace("rdm1 = Namespace()", "rdm_eb_cre = Namespace()\nrdm_eb_des = Namespace()")
            postamble = get_density_einsum_postamble(1, spin, spaces=["boo", "bov", "bvo", "bvv"])
            postamble = "\n".join([postamble.replace("rdm1", f"rdm_eb_{'cre' if cre else 'des'}") for cre in (True, False)])
            kwargs = {
                "preamble": preamble,
                "postamble": postamble,
            }
        else:
            kwargs = {}
        codegen(
            "make_eb_coup_rdm",
            returns,
            output,
            expr,
            **kwargs,
        )
