"""
Generate the CCSDT code.
"""

import sys

import pdaggerq
from albert.qc._pdaggerq import remove_reference_energy, remove_reference_energy_eom
from albert.qc.spin import ghf_to_uhf, ghf_to_rhf
from albert.qc import ghf, uhf, rhf
from albert.tensor import Tensor
from albert.index import Index
from albert.code._ebcc import EBCCCodeGenerator
from albert.misc import Stopwatch
from albert.opt.tools import _tensor_info

from ebcc.codegen.bootstrap_common import get_energy, get_amplitudes, get_rdm1, get_rdm2, get_eom

# Get the spin case
spin = sys.argv[1]

# Set up the code generators
code_generators = {
    "einsum": EBCCCodeGenerator(
        stdout=open(f"{spin[0].upper()}CCSDT.py", "w"),
    ),
}

# Write the preamble
for codegen in code_generators.values():
    codegen.preamble()

# Set up pdaggerq
pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

with Stopwatch("Energy"):
    # Get the energy contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["1"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    terms = pq.fully_contracted_strings()
    terms = remove_reference_energy(terms)

    # Get the energy in albert format
    output_expr, returns = get_energy(terms, spin)

    # Generate the energy code
    for codegen in code_generators.values():
        codegen(
            "energy",
            returns,
            output_expr,
        )

with Stopwatch("T amplitudes"):
    # Get the T1 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e1(i,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_t1 = pq.fully_contracted_strings()

    # Get the T2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e2(i,j,b,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_t2 = pq.fully_contracted_strings()

    # Get the T3 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e3(i,j,k,c,b,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_t3 = pq.fully_contracted_strings()

    # Get the T amplitudes in albert format
    output_expr, returns = get_amplitudes([terms_t1, terms_t2, terms_t3], spin, strategy="greedy")

    # Generate the T amplitude code
    for name, codegen in code_generators.items():
        codegen(
            "update_amps",
            returns,
            output_expr,
            as_dict=True,
        )

with Stopwatch("L amplitudes"):
    # Get the L1 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["1"]])
    pq.set_right_operators([["1"]])
    pq.add_st_operator(1.0, ["f", "e1(a,i)"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v", "e1(a,i)"], ["t1", "t2", "t3"])
    pq.set_left_operators([["l1"], ["l2"], ["l3"]])
    pq.add_st_operator(1.0, ["f", "e1(a,i)"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v", "e1(a,i)"], ["t1", "t2", "t3"])
    pq.add_st_operator(-1.0, ["e1(a,i)", "f"], ["t1", "t2", "t3"])
    pq.add_st_operator(-1.0, ["e1(a,i)", "v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_l1 = pq.fully_contracted_strings()

    # Get the L2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["1"]])
    pq.set_right_operators([["1"]])
    pq.add_st_operator(1.0, ["f", "e2(a,b,j,i)"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v", "e2(a,b,j,i)"], ["t1", "t2", "t3"])
    pq.set_left_operators([["l1"], ["l2"], ["l3"]])
    pq.add_st_operator(1.0, ["f", "e2(a,b,j,i)"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v", "e2(a,b,j,i)"], ["t1", "t2", "t3"])
    pq.add_st_operator(-1.0, ["e2(a,b,j,i)", "f"], ["t1", "t2", "t3"])
    pq.add_st_operator(-1.0, ["e2(a,b,j,i)", "v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_l2 = pq.fully_contracted_strings()

    # Get the L3 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["1"]])
    pq.set_right_operators([["1"]])
    pq.add_st_operator(1.0, ["f", "e3(a,b,c,k,j,i)"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v", "e3(a,b,c,k,j,i)"], ["t1", "t2", "t3"])
    pq.set_left_operators([["l1"], ["l2"], ["l3"]])
    pq.add_st_operator(1.0, ["f", "e3(a,b,c,k,j,i)"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v", "e3(a,b,c,k,j,i)"], ["t1", "t2", "t3"])
    pq.add_st_operator(-1.0, ["e3(a,b,c,k,j,i)", "f"], ["t1", "t2", "t3"])
    pq.add_st_operator(-1.0, ["e3(a,b,c,k,j,i)", "v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_l3 = pq.fully_contracted_strings()

    # Get the L amplitudes in albert format
    output_expr, returns = get_amplitudes([terms_l1, terms_l2, terms_l3], spin, strategy="greedy", which="l")

    # Generate the L amplitude code
    for name, codegen in code_generators.items():
        codegen(
            "update_lams",
            returns,
            output_expr,
            as_dict=True,
        )

with Stopwatch("1RDM"):
    # Get the 1RDM contractions in pdaggerq format
    terms = {}
    for sectors, indices in [("oo", "ij"), ("ov", "ia"), ("vo", "ai"), ("vv", "ab")]:
        pq.clear()
        pq.set_left_operators([["1"], ["l1"], ["l2"], ["l3"]])
        pq.add_st_operator(1.0, [f"e1({','.join(indices)})"], ["t1", "t2", "t3"])
        pq.simplify()
        terms[sectors, indices] = pq.fully_contracted_strings()

    # Get the 1RDM in albert format
    output_expr, returns, deltas, deltas_sources = get_rdm1(terms, spin)

    # Generate the 1RDM code
    for name, codegen in code_generators.items():
        def preamble():
            done = set()
            for delta, delta_source in zip(deltas, deltas_sources):
                if delta in done:
                    continue
                shape_source_index = 0 if delta.external_indices[0].space == "o" else 1
                codegen.tensor_declaration(
                    delta,
                    is_identity=True,
                    shape_source=delta_source,
                    shape_source_index=shape_source_index,
                )
                codegen._tensor_declared.add(_tensor_info(delta))
                done.add(delta)

        def postamble():
            if name != "einsum":
                raise NotImplementedError  # FIXME remove packing
            if spin != "uhf":
                codegen.write("rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])")
            else:
                codegen.write("rdm1.aa = np.block([[rdm1.aa.oo, rdm1.aa.ov], [rdm1.aa.vo, rdm1.aa.vv]])")
                codegen.write("rdm1.bb = np.block([[rdm1.bb.oo, rdm1.bb.ov], [rdm1.bb.vo, rdm1.bb.vv]])")

        codegen(
            "make_rdm1_f",
            returns,
            output_expr,
            preamble=preamble,
            postamble=postamble,
        )

with Stopwatch("2RDM"):
    # Get the 2RDM contractions in pdaggerq format
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
        pq.clear()
        pq.set_left_operators([["1"], ["l1"], ["l2"], ["l3"]])
        pq.add_st_operator(
            1.0, [f"e2({indices[0]},{indices[1]},{indices[3]},{indices[2]})"], ["t1", "t2", "t3"]
        )
        pq.simplify()
        terms[sectors, indices] = pq.fully_contracted_strings()

    # Get the 2RDM in albert format
    output_expr, returns, deltas, deltas_sources = get_rdm2(terms, spin, strategy="trav" if spin == "uhf" else "exhaust")

    # Generate the 2RDM code
    for name, codegen in code_generators.items():
        def preamble():
            done = set()
            for delta, delta_source in zip(deltas, deltas_sources):
                if delta in done:
                    continue
                shape_source_index = 0 if delta.external_indices[0].space == "o" else 1
                codegen.tensor_declaration(
                    delta,
                    is_identity=True,
                    shape_source=delta_source,
                    shape_source_index=shape_source_index,
                )
                codegen._tensor_declared.add(_tensor_info(delta))
                done.add(delta)

        def postamble():
            if name != "einsum":
                raise NotImplementedError  # FIXME remove packing, handle transpose
            if spin != "uhf":
                codegen.write("rdm2 = pack_2e(rdm2.oooo, rdm2.ooov, rdm2.oovo, rdm2.ovoo, rdm2.vooo, rdm2.oovv, rdm2.ovov, rdm2.ovvo, rdm2.voov, rdm2.vovo, rdm2.vvoo, rdm2.ovvv, rdm2.vovv, rdm2.vvov, rdm2.vvvo, rdm2.vvvv).transpose((0, 2, 1, 3))")
            else:
                codegen.write("rdm2.aaaa = pack_2e(rdm2.aaaa.oooo, rdm2.aaaa.ooov, rdm2.aaaa.oovo, rdm2.aaaa.ovoo, rdm2.aaaa.vooo, rdm2.aaaa.oovv, rdm2.aaaa.ovov, rdm2.aaaa.ovvo, rdm2.aaaa.voov, rdm2.aaaa.vovo, rdm2.aaaa.vvoo, rdm2.aaaa.ovvv, rdm2.aaaa.vovv, rdm2.aaaa.vvov, rdm2.aaaa.vvvo, rdm2.aaaa.vvvv).transpose((0, 2, 1, 3))")
                codegen.write("rdm2.aabb = pack_2e(rdm2.abab.oooo, rdm2.abab.ooov, rdm2.abab.oovo, rdm2.abab.ovoo, rdm2.abab.vooo, rdm2.abab.oovv, rdm2.abab.ovov, rdm2.abab.ovvo, rdm2.abab.voov, rdm2.abab.vovo, rdm2.abab.vvoo, rdm2.abab.ovvv, rdm2.abab.vovv, rdm2.abab.vvov, rdm2.abab.vvvo, rdm2.abab.vvvv).transpose((0, 2, 1, 3))")
                codegen.write("rdm2.bbbb = pack_2e(rdm2.bbbb.oooo, rdm2.bbbb.ooov, rdm2.bbbb.oovo, rdm2.bbbb.ovoo, rdm2.bbbb.vooo, rdm2.bbbb.oovv, rdm2.bbbb.ovov, rdm2.bbbb.ovvo, rdm2.bbbb.voov, rdm2.bbbb.vovo, rdm2.bbbb.vvoo, rdm2.bbbb.ovvv, rdm2.bbbb.vovv, rdm2.bbbb.vvov, rdm2.bbbb.vvvo, rdm2.bbbb.vvvv).transpose((0, 2, 1, 3))")
                codegen.write("del rdm2.abab")

        codegen(
            "make_rdm2_f",
            returns,
            output_expr,
            preamble=preamble,
            postamble=postamble,
        )

with Stopwatch("IP-EOM"):
    # Get the R1 contractions in pdaggerq format
    pq.clear()
    pq.set_right_operators_type("IP")
    pq.set_left_operators([["a*(i)"]])
    pq.set_right_operators([["r1"], ["r2"], ["r3"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_r1 = pq.fully_contracted_strings()
    terms_r1 = remove_reference_energy_eom(terms_r1)

    # Get the R2 contractions in pdaggerq format
    pq.clear()
    pq.set_right_operators_type("IP")
    pq.set_left_operators([["a*(i)", "a*(j)", "a(a)"]])
    pq.set_right_operators([["r1"], ["r2"], ["r3"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_r2 = pq.fully_contracted_strings()
    terms_r2 = remove_reference_energy_eom(terms_r2)

    # Get the R3 contractions in pdaggerq format
    pq.clear()
    pq.set_right_operators_type("IP")
    pq.set_left_operators([["a*(i)", "a*(j)", "a*(k)", "a(b)", "a(a)"]])
    pq.set_right_operators([["r1"], ["r2"], ["r3"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_r3 = pq.fully_contracted_strings()
    terms_r3 = remove_reference_energy_eom(terms_r3)

    # Get the R amplitudes in albert format
    output_expr_nr, returns_nr, output_expr_r, returns_r = get_eom([terms_r1, terms_r2, terms_r3], spin, strategy="greedy", which="ip")

    # Generate the R amplitude intermediates code
    for name, codegen in code_generators.items():
        codegen(
            "hbar_matvec_ip_intermediates",
            returns_nr,
            output_expr_nr,
            as_dict=True,
        )

    # Generate the R amplitude code
    for name, codegen in code_generators.items():
        codegen(
            "hbar_matvec_ip",
            returns_r,
            output_expr_r,
            as_dict=True,
        )

with Stopwatch("EA-EOM"):
    # Get the R1 contractions in pdaggerq format
    pq.clear()
    pq.set_right_operators_type("EA")
    pq.set_left_operators([["a(a)"]])
    pq.set_right_operators([["r1"], ["r2"], ["r3"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_r1 = pq.fully_contracted_strings()
    terms_r1 = remove_reference_energy_eom(terms_r1)

    # Get the R2 contractions in pdaggerq format
    pq.clear()
    pq.set_right_operators_type("EA")
    pq.set_left_operators([["a*(i)", "a(b)", "a(a)"]])
    pq.set_right_operators([["r1"], ["r2"], ["r3"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_r2 = pq.fully_contracted_strings()
    terms_r2 = remove_reference_energy_eom(terms_r2)

    # Get the R3 contractions in pdaggerq format
    pq.clear()
    pq.set_right_operators_type("EA")
    pq.set_left_operators([["a*(i)", "a*(j)", "a(c)", "a(b)", "a(a)"]])
    pq.set_right_operators([["r1"], ["r2"], ["r3"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_r3 = pq.fully_contracted_strings()
    terms_r3 = remove_reference_energy_eom(terms_r3)

    # Get the R amplitudes in albert format
    output_expr_nr, returns_nr, output_expr_r, returns_r = get_eom([terms_r1, terms_r2, terms_r3], spin, strategy="greedy", which="ea")

    # Generate the R amplitude intermediates code
    for name, codegen in code_generators.items():
        codegen(
            "hbar_matvec_ea_intermediates",
            returns_nr,
            output_expr_nr,
            as_dict=True,
        )

    # Generate the R amplitude code
    for name, codegen in code_generators.items():
        codegen(
            "hbar_matvec_ea",
            returns_r,
            output_expr_r,
            as_dict=True,
        )

if spin == "ghf":  # FIXME
    with Stopwatch("EE-EOM"):
        # Get the R1 contractions in pdaggerq format
        pq.clear()
        pq.set_right_operators_type("EE")
        pq.set_left_operators([["e1(i,a)"]])
        pq.set_right_operators([["r1"], ["r2"], ["r3"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
        pq.simplify()
        terms_r1 = pq.fully_contracted_strings()
        terms_r1 = remove_reference_energy_eom(terms_r1)

        # Get the R2 contractions in pdaggerq format
        pq.clear()
        pq.set_right_operators_type("EE")
        pq.set_left_operators([["e2(i,j,b,a)"]])
        pq.set_right_operators([["r1"], ["r2"], ["r3"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
        pq.simplify()
        terms_r2 = pq.fully_contracted_strings()
        terms_r2 = remove_reference_energy_eom(terms_r2)

        # Get the R3 contractions in pdaggerq format
        pq.clear()
        pq.set_right_operators_type("EE")
        pq.set_left_operators([["e3(i,j,k,c,b,a)"]])
        pq.set_right_operators([["r1"], ["r2"], ["r3"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
        pq.simplify()
        terms_r3 = pq.fully_contracted_strings()
        terms_r3 = remove_reference_energy_eom(terms_r3)

        # Get the R amplitudes in albert format
        returns_nr, output_expr_nr, returns_r, output_expr_r = get_eom([terms_r1, terms_r2, terms_r3], spin, strategy="greedy", which="ee")

        # Generate the R amplitude intermediates code
        for name, codegen in code_generators.items():
            codegen(
                "hbar_matvec_ee_intermediates",
                returns_nr,
                output_expr_nr,
                as_dict=True,
            )

        # Generate the R amplitude code
        for name, codegen in code_generators.items():
            def postamble():
                if spin == "uhf":
                    r2_abab = Tensor(
                        Index("i", space="o", spin="a"),
                        Index("j", space="v", spin="b"),
                        Index("a", space="o", spin="a"),
                        Index("b", space="v", spin="b"),
                        name="r2new",
                    )
                    r2_baba = Tensor(
                        Index("i", space="o", spin="b"),
                        Index("j", space="v", spin="a"),
                        Index("a", space="o", spin="b"),
                        Index("b", space="v", spin="a"),
                        name="r2new",
                    )
                    codegen.tensor_expression(r2_baba, r2_abab, is_return=True)

            codegen(
                "hbar_matvec_ee",
                returns_r,
                output_expr_r,
                postamble=postamble,
                as_dict=True,
            )

with Stopwatch("L-IP-EOM"):
    # Get the L1 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators_type("IP")
    pq.set_left_operators([["l1"], ["l2"], ["l3"]])
    pq.set_right_operators([["a(i)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_r1 = pq.fully_contracted_strings()
    terms_r1 = remove_reference_energy_eom(terms_r1)

    # Get the L2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators_type("IP")
    pq.set_right_operators([["a*(a)", "a(j)", "a(i)"]])
    pq.set_left_operators([["l1"], ["l2"], ["l3"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_r2 = pq.fully_contracted_strings()
    terms_r2 = remove_reference_energy_eom(terms_r2)

    # Get the L3 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators_type("IP")
    pq.set_right_operators([["a*(a)", "a*(b)", "a(k)", "a(j)", "a(i)"]])
    pq.set_left_operators([["l1"], ["l2"], ["l3"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_r3 = pq.fully_contracted_strings()
    terms_r3 = remove_reference_energy_eom(terms_r3)

    # Get the L amplitudes in albert format
    returns_nr, output_expr_nr, returns_r, output_expr_r = get_eom([terms_r1, terms_r2, terms_r3], spin, strategy="greedy", which="ip")

    # Generate the L amplitude intermediates code
    for name, codegen in code_generators.items():
        codegen(
            "hbar_lmatvec_ip_intermediates",
            returns_nr,
            output_expr_nr,
            as_dict=True,
        )

    # Generate the L amplitude code
    for name, codegen in code_generators.items():
        codegen(
            "hbar_lmatvec_ip",
            returns_r,
            output_expr_r,
            as_dict=True,
        )

with Stopwatch("L-EA-EOM"):
    # Get the L1 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators_type("EA")
    pq.set_left_operators([["l1"], ["l2"], ["l3"]])
    pq.set_right_operators([["a*(a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_r1 = pq.fully_contracted_strings()
    terms_r1 = remove_reference_energy_eom(terms_r1)

    # Get the L2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators_type("EA")
    pq.set_right_operators([["a*(a)", "a*(b)", "a(i)"]])
    pq.set_left_operators([["l1"], ["l2"], ["l3"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_r2 = pq.fully_contracted_strings()
    terms_r2 = remove_reference_energy_eom(terms_r2)

    # Get the L3 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators_type("EA")
    pq.set_right_operators([["a*(a)", "a*(b)", "a*(c)", "a(j)", "a(i)"]])
    pq.set_left_operators([["l1"], ["l2"], ["l3"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
    pq.simplify()
    terms_r3 = pq.fully_contracted_strings()
    terms_r3 = remove_reference_energy_eom(terms_r3)

    # Get the L amplitudes in albert format
    returns_nr, output_expr_nr, returns_r, output_expr_r = get_eom([terms_r1, terms_r2, terms_r3], spin, strategy="greedy", which="ea")

    # Generate the L amplitude intermediates code
    for name, codegen in code_generators.items():
        codegen(
            "hbar_lmatvec_ea_intermediates",
            returns_nr,
            output_expr_nr,
            as_dict=True,
        )

    # Generate the L amplitude code
    for name, codegen in code_generators.items():
        codegen(
            "hbar_lmatvec_ea",
            returns_r,
            output_expr_r,
            as_dict=True,
        )

if spin == "ghf":  # FIXME
    with Stopwatch("L-EE-EOM"):
        # Get the L1 contractions in pdaggerq format
        pq.clear()
        pq.set_left_operators_type("EE")
        pq.set_left_operators([["l1"], ["l2"], ["l3"]])
        pq.set_right_operators([["e1(a,i)"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
        pq.simplify()
        terms_r1 = pq.fully_contracted_strings()
        terms_r1 = remove_reference_energy_eom(terms_r1)

        # Get the L2 contractions in pdaggerq format
        pq.clear()
        pq.set_left_operators_type("EE")
        pq.set_left_operators([["l1"], ["l2"], ["l3"]])
        pq.set_right_operators([["e2(a,b,j,i)"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
        pq.simplify()
        terms_r2 = pq.fully_contracted_strings()
        terms_r2 = remove_reference_energy_eom(terms_r2)

        # Get the L3 contractions in pdaggerq format
        pq.clear()
        pq.set_left_operators_type("EE")
        pq.set_left_operators([["l1"], ["l2"], ["l3"]])
        pq.set_right_operators([["e3(a,b,c,k,j,i)"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
        pq.simplify()
        terms_r3 = pq.fully_contracted_strings()
        terms_r3 = remove_reference_energy_eom(terms_r3)

        # Get the L amplitudes in albert format
        returns_nr, output_expr_nr, returns_r, output_expr_r = get_eom([terms_r1, terms_r2, terms_r3], spin, strategy="greedy", which="ee")

        # Generate the L amplitude intermediates code
        for name, codegen in code_generators.items():
            codegen(
                "hbar_lmatvec_ee_intermediates",
                returns_nr,
                output_expr_nr,
                as_dict=True,
            )

        # Generate the L amplitude code
        for name, codegen in code_generators.items():
            def postamble():
                if spin == "uhf":
                    r2_abab = Tensor(
                        Index("i", space="o", spin="a"),
                        Index("j", space="v", spin="b"),
                        Index("a", space="o", spin="a"),
                        Index("b", space="v", spin="b"),
                        name="r2new",
                    )
                    r2_baba = Tensor(
                        Index("i", space="o", spin="b"),
                        Index("j", space="v", spin="a"),
                        Index("a", space="o", spin="b"),
                        Index("b", space="v", spin="a"),
                        name="r2new",
                    )
                    codegen.tensor_expression(r2_baba, r2_abab, is_return=True)

            codegen(
                "hbar_lmatvec_ee",
                returns_r,
                output_expr_r,
                postamble=postamble,
                as_dict=True,
            )

for codegen in code_generators.values():
    codegen.postamble()
    codegen.stdout.close()
