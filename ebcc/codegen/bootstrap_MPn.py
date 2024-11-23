"""
Generate the MPn code.
"""

import itertools
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
from ebcc.codegen import bootstrap_hugenholtz as hugenholtz

# Get the spin case
spin = sys.argv[1]
order = int(sys.argv[2])

# Set up the code generators
code_generators = {
    "einsum": EBCCCodeGenerator(
        stdout=open(f"{spin[0].upper()}MP{order}.py", "w"),
    ),
}

# Write the preamble
for codegen in code_generators.values():
    codegen.preamble()

with Stopwatch("Energy"):
    # Get the energy contractions in pdaggerq format
    terms = sum([
        [hugenholtz.get_pdaggerq(graph) for graph in hugenholtz.get_hugenholtz_diagrams(n)]
        for n in range(2, order+1)
    ], [])

    # Get the energy in albert format
    output_expr, returns = get_energy(terms, spin)

    # Generate the energy code
    for codegen in code_generators.values():
        codegen(
            "energy",
            returns,
            output_expr,
        )

if order == 2:
    pq = pdaggerq.pq_helper("fermi")
    pq.set_print_level(0)

    with Stopwatch("1RDM"):
        # Get the 1RDM contractions in pdaggerq format
        terms = {
            ("oo", "ij"): [["+1.0", "d(i,j)"], ["-0.5", "t2(b,a,k,i)", "l2(k,j,b,a)"]],
            ("vv", "ab"): [["+0.5", "t2(b,c,i,j)", "l2(i,j,a,c)"]]
        }

        # Get the 1RDM in albert format
        output_expr, returns, deltas, deltas_sources = get_rdm1(terms, spin)

        # Generate the 1RDM code
        for name, codegen in code_generators.items():
            def preamble():
                done = set()
                for delta, delta_source in zip(deltas, deltas_sources):
                    if delta in done:
                        continue
                    shape_source_index = 0 if delta.external_indices[0].space == "o" else -1
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
                    codegen.write("rdm1.ov = np.zeros((t1.shape[0], t1.shape[1]))")
                    codegen.write("rdm1.vo = np.zeros((t1.shape[1], t1.shape[0]))")
                    codegen.write("rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])")
                else:
                    codegen.write("rdm1.aa.ov = np.zeros((t1.aa.shape[0], t1.aa.shape[1]))")
                    codegen.write("rdm1.aa.vo = np.zeros((t1.aa.shape[1], t1.aa.shape[0]))")
                    codegen.write("rdm1.bb.ov = np.zeros((t1.bb.shape[0], t1.bb.shape[1]))")
                    codegen.write("rdm1.bb.vo = np.zeros((t1.bb.shape[1], t1.bb.shape[0]))")
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
        terms = {
            ("oovv", "ijab"): [["t2(a,b,i,j)"]],
            ("vvoo", "abij"): [["l2(i,j,a,b)"]],
        }

        # Get the 2RDM in albert format
        output_expr, returns, deltas, deltas_sources = get_rdm2(terms, spin, strategy="trav" if spin == "uhf" else "exhaust")

        # Generate the 2RDM code
        for name, codegen in code_generators.items():
            def preamble():
                done = set()
                for delta, delta_source in zip(deltas, deltas_sources):
                    if delta in done:
                        continue
                    shape_source_index = 0 if delta.external_indices[0].space == "o" else -1
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
                    for key in itertools.product("ov", repeat=4):
                        if tuple(key) in {("o", "o", "v", "v") , ("v", "v", "o", "o")}:
                            continue
                        i, j, k, l = ["ov".index(k) for k in key]
                        codegen.write(f"rdm2.{''.join(key)} = np.zeros((t1.shape[{i}], t1.shape[{j}], t1.shape[{k}], t1.shape[{l}]))")
                    codegen.write("rdm2 = pack_2e(rdm2.oooo, rdm2.ooov, rdm2.oovo, rdm2.ovoo, rdm2.vooo, rdm2.oovv, rdm2.ovov, rdm2.ovvo, rdm2.voov, rdm2.vovo, rdm2.vvoo, rdm2.ovvv, rdm2.vovv, rdm2.vvov, rdm2.vvvo, rdm2.vvvv).transpose((0, 2, 1, 3))")
                else:
                    for s in ("aaaa", "aabb", "bbbb"):
                        for key in itertools.product("ov", repeat=4):
                            if tuple(key) in {("o", "o", "v", "v") , ("v", "v", "o", "o")}:
                                continue
                            i, j, k, l = ["ov".index(k) for k in key]
                            si, sj, sk, sl = s
                            codegen.write(f"rdm2.{si+sj+sk+sl}.{''.join(key)} = np.zeros((t1.{si}.shape[{i}], t1.{sj}.shape[{j}], t1.{sk}.shape[{k}], t1.{sl}.shape[{l}]))")
                    codegen.write("rdm2.aaaa = pack_2e(rdm2.aaaa.oooo, rdm2.aaaa.ooov, rdm2.aaaa.oovo, rdm2.aaaa.ovoo, rdm2.aaaa.vooo, rdm2.aaaa.oovv, rdm2.aaaa.ovov, rdm2.aaaa.ovvo, rdm2.aaaa.voov, rdm2.aaaa.vovo, rdm2.aaaa.vvoo, rdm2.aaaa.ovvv, rdm2.aaaa.vovv, rdm2.aaaa.vvov, rdm2.aaaa.vvvo, rdm2.aaaa.vvvv).transpose((0, 2, 1, 3))")
                    codegen.write("rdm2.aabb = pack_2e(rdm2.abab.oooo, rdm2.abab.ooov, rdm2.abab.oovo, rdm2.abab.ovoo, rdm2.abab.vooo, rdm2.abab.oovv, rdm2.abab.ovov, rdm2.abab.ovvo, rdm2.abab.voov, rdm2.abab.vovo, rdm2.abab.vvoo, rdm2.abab.ovvv, rdm2.abab.vovv, rdm2.abab.vvov, rdm2.abab.vvvo, rdm2.abab.vvvv).transpose((0, 2, 1, 3))")
                    codegen.write("rdm2.bbbb = pack_2e(rdm2.bbbb.oooo, rdm2.bbbb.ooov, rdm2.bbbb.oovo, rdm2.bbbb.ovoo, rdm2.bbbb.vooo, rdm2.bbbb.oovv, rdm2.bbbb.ovov, rdm2.bbbb.ovvo, rdm2.bbbb.voov, rdm2.bbbb.vovo, rdm2.bbbb.vvoo, rdm2.bbbb.ovvv, rdm2.bbbb.vovv, rdm2.bbbb.vvov, rdm2.bbbb.vvvo, rdm2.bbbb.vvvv).transpose((0, 2, 1, 3))")
                    codegen.write("del rdm2.abab")
                codegen.write("rdm1 = make_rdm1_f(t2=t2, l2=l2)")
                if spin == "uhf":
                    codegen.write("delta = Namespace(")
                    codegen.write("    aa=np.diag(np.concatenate([np.ones(t2.aaaa.shape[0]), np.zeros(t2.aaaa.shape[-1])])),")
                    codegen.write("    bb=np.diag(np.concatenate([np.ones(t2.bbbb.shape[0]), np.zeros(t2.bbbb.shape[-1])])),")
                    codegen.write(")")
                    codegen.write("rdm1.aa -= delta.aa")
                    codegen.write("rdm1.bb -= delta.bb")
                    codegen.write("rdm2.aaaa += einsum(delta.aa, (0, 1), rdm1.aa, (3, 2), (0, 1, 2, 3))")
                    codegen.write("rdm2.aaaa += einsum(rdm1.aa, (1, 0), delta.aa, (2, 3), (0, 1, 2, 3))")
                    codegen.write("rdm2.aaaa -= einsum(delta.aa, (0, 3), rdm1.aa, (2, 1), (0, 1, 2, 3))")
                    codegen.write("rdm2.aaaa -= einsum(rdm1.aa, (0, 3), delta.aa, (1, 2), (0, 1, 2, 3))")
                    codegen.write("rdm2.aaaa += einsum(delta.aa, (0, 1), delta.aa, (2, 3), (0, 1, 2, 3))")
                    codegen.write("rdm2.aaaa -= einsum(delta.aa, (0, 3), delta.aa, (1, 2), (0, 1, 2, 3))")
                    codegen.write("rdm2.bbbb += einsum(delta.bb, (0, 1), rdm1.bb, (3, 2), (0, 1, 2, 3))")
                    codegen.write("rdm2.bbbb += einsum(rdm1.bb, (1, 0), delta.bb, (2, 3), (0, 1, 2, 3))")
                    codegen.write("rdm2.bbbb -= einsum(delta.bb, (0, 3), rdm1.bb, (2, 1), (0, 1, 2, 3))")
                    codegen.write("rdm2.bbbb -= einsum(rdm1.bb, (0, 3), delta.bb, (1, 2), (0, 1, 2, 3))")
                    codegen.write("rdm2.bbbb += einsum(delta.bb, (0, 1), delta.bb, (2, 3), (0, 1, 2, 3))")
                    codegen.write("rdm2.bbbb -= einsum(delta.bb, (0, 3), delta.bb, (1, 2), (0, 1, 2, 3))")
                    codegen.write("rdm2.aabb += einsum(delta.aa, (0, 1), rdm1.bb, (3, 2), (0, 1, 2, 3))")
                    codegen.write("rdm2.aabb += einsum(rdm1.aa, (1, 0), delta.bb, (2, 3), (0, 1, 2, 3))")
                    codegen.write("rdm2.aabb += einsum(delta.aa, (0, 1), delta.bb, (2, 3), (0, 1, 2, 3))")
                elif spin == "ghf":
                    codegen.write("delta = np.diag(np.concatenate([np.ones(t2.shape[0]), np.zeros(t2.shape[-1])]))")
                    codegen.write("rdm1 -= delta")
                    codegen.write("rdm2 += einsum(delta, (0, 1), rdm1, (3, 2), (0, 1, 2, 3))")
                    codegen.write("rdm2 += einsum(rdm1, (1, 0), delta, (2, 3), (0, 1, 2, 3))")
                    codegen.write("rdm2 -= einsum(delta, (0, 3), rdm1, (2, 1), (0, 1, 2, 3))")
                    codegen.write("rdm2 -= einsum(rdm1, (0, 3), delta, (1, 2), (0, 1, 2, 3))")
                    codegen.write("rdm2 += einsum(delta, (0, 1), delta, (2, 3), (0, 1, 2, 3))")
                    codegen.write("rdm2 -= einsum(delta, (0, 3), delta, (1, 2), (0, 1, 2, 3))")
                elif spin == "rhf":
                    codegen.write("delta = np.diag(np.concatenate([np.ones(t2.shape[0]), np.zeros(t2.shape[-1])]))")
                    codegen.write("rdm1 -= delta * 2")
                    codegen.write("rdm2 += einsum(delta, (0, 1), rdm1, (3, 2), (0, 1, 2, 3)) * 2")
                    codegen.write("rdm2 += einsum(rdm1, (1, 0), delta, (2, 3), (0, 1, 2, 3)) * 2")
                    codegen.write("rdm2 -= einsum(delta, (0, 3), rdm1, (2, 1), (0, 1, 2, 3))")
                    codegen.write("rdm2 -= einsum(rdm1, (0, 3), delta, (1, 2), (0, 1, 2, 3))")
                    codegen.write("rdm2 += einsum(delta, (0, 1), delta, (2, 3), (0, 1, 2, 3)) * 4")
                    codegen.write("rdm2 -= einsum(delta, (0, 3), delta, (1, 2), (0, 1, 2, 3)) * 2")

            codegen(
                "make_rdm2_f",
                returns,
                output_expr,
                preamble=preamble,
                postamble=postamble,
            )

    with Stopwatch("IP-EOM"):
        # Get the R1 contractions in pdaggerq format
        terms_r1 = [
            ["-1.0", "f(j,i)", "r1(j)"],
            ["+0.5", "<k,j||a,i>", "r2(a,k,j)"],
            ["-0.25", "<k,j||a,b>", "t2(a,b,i,j)", "r1(k)"],
            ["-0.25", "<i,j||a,b>", "t2(a,b,k,j)", "r1(k)"],
        ]

        # Get the R2 contractions in pdaggerq format
        terms_r2 = [
            ["-1.0", "<k,a||i,j>", "r1(k)"],
            ["+1.0", "f(a,b)", "r2(b,i,j)"],
            ["-1.0", "P(i,j)", "f(k,j)", "r2(a,i,k)"],
        ]

        # Get the R amplitudes in albert format
        output_expr_nr, returns_nr, output_expr_r, returns_r = get_eom([terms_r1, terms_r2], spin, strategy="exhaust", which="ip")

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
        terms_r1 = [
            ["+1.0", "f(a,b)", "r1(b)"],
            ["-0.5", "<i,a||b,c>", "r2(b,c,i)"],
            ["-0.25", "<j,i||b,c>", "t2(b,a,j,i)", "r1(c)"],
            ["-0.25", "<j,i||b,a>", "t2(b,c,j,i)", "r1(c)"],
        ]

        # Get the R2 contractions in pdaggerq format
        terms_r2 = [
            ["-1.0", "f(j,i)", "r2(a,b,j)"],
            ["+1.0", "P(a,b)", "f(a,c)", "r2(c,b,i)"],
            ["+1.0", "<a,b||c,i>", "r1(c)"],
        ]

        # Get the R amplitudes in albert format
        output_expr_nr, returns_nr, output_expr_r, returns_r = get_eom([terms_r1, terms_r2], spin, strategy="exhaust", which="ea")

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
            terms_r1 = [
                ["-1.0", "f(j,i)", "r1(a,j)"],
                ["+1.0", "f(a,b)", "r1(b,i)"],
                ["+1.0", "<j,a||b,i>", "r1(b,j)"],
                ["-0.5", "<k,j||b,i>", "r2(b,a,k,j)"],
                ["-0.5", "<j,a||b,c>", "r2(b,c,i,j)"],
                ["-0.5", "<k,j||b,c>", "r1(a,k)", "t2(b,c,i,j)"],
                ["-0.5", "<k,j||b,c>", "r1(c,i)", "t2(b,a,k,j)"],
                ["+1.0", "<k,j||b,c>", "r1(c,k)", "t2(b,a,i,j)"],
            ]

            # Get the R2 contractions in pdaggerq format
            terms_r2 = [
                ["-1.0", "P(i,j)", "f(k,j)", "r2(a,b,i,k)"],
                ["+1.0", "P(a,b)", "f(a,c)", "r2(c,b,i,j)"],
                ["+1.0", "P(a,b)", "<k,a||i,j>", "r1(b,k)"],
                ["+1.0", "P(i,j)", "<a,b||c,j>", "r1(c,i)"],
            ]

            # Get the R amplitudes in albert format
            output_expr_nr, returns_nr, output_expr_r, returns_r = get_eom([terms_r1, terms_r2], spin, strategy="trav", which="ee")

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

for codegen in code_generators.values():
    codegen.postamble()
    codegen.stdout.close()
