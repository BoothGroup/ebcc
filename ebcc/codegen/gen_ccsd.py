"""Script to generate equations for the CCSD model.
"""

import re
import warnings
import sympy
import drudge
from fractions import Fraction
from qwick.wick import apply_wick
from qwick.index import Idx
from qwick.operator import FOperator
from qwick.expression import *
from qwick.convenience import *
from qwick import codegen
from ebcc.codegen import common, wick

from dummy_spark import SparkContext
ctx = SparkContext()
dr = drudge.Drudge(ctx)

warnings.simplefilter("ignore", UserWarning)

# Spin setting:
spin = "ghf"  # {"ghf", "rhf", "uhf"}

# Indices
occs = i, j, k, l = [Idx(n, "occ") for n in range(4)]
virs = a, b, c, d = [Idx(n, "vir") for n in range(4)]

# Tensors
H, _ = wick.get_hamiltonian(rank=(2, 0, 0))
bra = bra1, bra2 = wick.get_bra_spaces(rank=(2, 0, 0), occs=occs, virs=virs)
ket = ket1, ket2 = wick.get_ket_spaces(rank=(2, 0, 0), occs=occs, virs=virs)
braip = bra1ip, bra2ip = wick.get_bra_ip_spaces(rank=(2, 0, 0), occs=occs, virs=virs)
braea = bra1ea, bra2ea = wick.get_bra_ea_spaces(rank=(2, 0, 0), occs=occs, virs=virs)
ketip = ket1ip, ket2ip = wick.get_ket_ip_spaces(rank=(2, 0, 0), occs=occs, virs=virs)
ketea = ket1ea, ket2ea = wick.get_ket_ea_spaces(rank=(2, 0, 0), occs=occs, virs=virs)
rip = r1ip, r2ip = wick.get_r_ip_spaces(rank=(2, 0, 0), occs=occs, virs=virs)
rea = r1ea, r2ea = wick.get_r_ea_spaces(rank=(2, 0, 0), occs=occs, virs=virs)
T, _ = wick.get_excitation_ansatz(rank=(2, 0, 0), occs=occs, virs=virs)
L, _ = wick.get_deexcitation_ansatz(rank=(2, 0, 0), occs=occs, virs=virs)
Hbars = wick.construct_hbar(H, T, max_commutator=5)
Hbar = Hbars[-2]

# Printer
reorder_axes = {
        # TODO remove:
        "l1new": (1, 0),
        "l2new": (2, 3, 0, 1),
}
if spin == "rhf":
    reorder_axes["v"] = (0, 2, 1, 3)
    for x in common.ov_2e:
        reorder_axes["rdm2_f_%s" % x] = (0, 2, 1, 3)
printer = codegen.EinsumPrinter(
        occupancy_tags={
            "v": "{base}.{tags}",
            "f": "{base}.{tags}",
            "delta": "delta_{tags}",
        },
        reorder_axes=reorder_axes,
        remove_spacing=True,
        garbage_collection=True,
        base_indent=1,
        einsum="lib.einsum",
        zeros="np.zeros",
        dtype="np.float64",
)
sizes = {"nocc": sympy.Symbol("N"), "nvir": sympy.Symbol("N")*5}

# Get prefix and spin transformation function according to setting:
if spin == "rhf":
    transform_spin = lambda terms, indices, **kwargs: codegen.ghf_to_rhf(terms, indices, **kwargs)
    prefix = ""
elif spin == "uhf":
    transform_spin = lambda terms, indices, **kwargs: codegen.ghf_to_uhf(terms, indices, **kwargs)
    prefix = "u"
elif spin == "ghf":
    transform_spin = lambda terms, indices, **kwargs: terms
    prefix = "g"

# Declare particle types:
particles = {
        "f": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "v": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "t1": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "t2": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "l1": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "l2": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "r1": ((codegen.FERMION, 0),),
        "r2": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0)),
        "t1new": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "t2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "l1new": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "l2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "r1new": ((codegen.FERMION, 0),),
        "r2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0)),
        "delta": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        **{"r1_%s" % x: ((codegen.FERMION, 0), (codegen.FERMION, 0),) for x in ["o", "v"]},
        **{"r2_%s" % x: ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)) for x in ["o", "v"]},
        **{"rdm1_f_%s" % x: ((codegen.FERMION, 0), (codegen.FERMION, 0)) for x in common.ov_1e},
        **{"rdm2_f_%s" % x: ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)) for x in common.ov_2e},
        "h11": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "h22": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 2), (codegen.FERMION, 3), (codegen.FERMION, 2)),  # FIXME?
}

# Timer:
timer = common.Stopwatch()

with common.FilePrinter("%sccsd" % prefix) as file_printer:
    # Get energy expression:
    with common.FunctionPrinter(
            file_printer,
            "energy",
            ["f", "v", "nocc", "nvir", "t1", "t2"],
            ["e_cc"],
            timer=timer,
    ) as function_printer:
        out = apply_wick(Hbar)
        out.resolve()
        expr = AExpression(Ex=out)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="e_cc")
        terms = transform_spin(terms, indices)
        terms = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms.latex(), comment="CCSD energy")
        terms = codegen.optimize([terms], sizes=sizes, optimize="exhaust", verify=True, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="CCSD energy")

    # Get amplitudes function:
    with common.FunctionPrinter(
            file_printer,
            "update_amps",
            ["f", "v", "nocc", "nvir", "t1", "t2"],
            ["t1new", "t2new"],
            timer=timer,
    ) as function_printer:
        # T1 residuals:
        S = bra1 * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="t1new")
        terms = transform_spin(terms, indices, project_onto=[(codegen.ALPHA, codegen.ALPHA)])
        terms_t1 = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms_t1.latex(), comment="T1 amplitude")

        # T2 residuals:
        S = bra2 * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="t2new")
        terms = transform_spin(terms, indices, project_onto=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)])
        terms_t2 = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms_t2.latex(), comment="T2 amplitude")

        terms = codegen.optimize([terms_t1, terms_t2], sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="T1 and T2 amplitudes")

    # Get lambda amplitudes function:
    with common.FunctionPrinter(
            file_printer,
            "update_lams",
            ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
            ["l1new", "l2new"],
            timer=timer,
    ) as function_printer:
        # L1 residuals <0|Hbar|singles> (not proportional to lambda):
        S = Hbar * ket1
        out = apply_wick(S)
        out.resolve()
        expr1 = AExpression(Ex=out)
        expr1 = expr1.get_connected()
        expr1.sort_tensors()

        # L1 residuals <0|(L Hbar)_c|singles> (connected pieces proportional to Lambda):
        S = L * Hbar * ket1
        out = apply_wick(S)
        out.resolve()
        expr2 = AExpression(Ex=out)
        expr2 = expr2.get_connected()
        expr2.sort_tensors()
        terms, indices = codegen.wick_to_sympy(expr1+expr2, particles, return_value="l1new")
        terms = transform_spin(terms, indices, project_onto=[(codegen.ALPHA, codegen.ALPHA)])
        terms_l1 = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms_l1.latex(), comment="L1 amplitude")

        # L2 residuals <0|Hbar|doubles> (not proportional to lambda):
        S = Hbar * ket2
        out = apply_wick(S)
        out.resolve()
        expr1 = AExpression(Ex=out)
        expr1 = expr1.get_connected()
        expr1.sort_tensors()

        # L2 residuals <0|L Hbar|doubles> (connected pieces proportional to lambda):
        S = L * Hbar * ket2
        out = apply_wick(S)
        out.resolve()
        expr2 = AExpression(Ex=out)
        expr2 = expr2.get_connected()
        expr2.sort_tensors()

        # L2 residuals (disonnected pieces proportional to lambda):
        P1 = PE1("occ", "vir")
        S = Hbar * P1 * L * ket2
        out = apply_wick(S)
        out.resolve()
        expr3 = AExpression(Ex=out)
        expr3.sort_tensors()
        terms, indices = codegen.wick_to_sympy(expr1+expr2+expr3, particles, return_value="l2new")
        terms = transform_spin(terms, indices, project_onto=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)])
        terms_l2 = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms_l2.latex(), comment="L2 amplitude")

        terms = codegen.optimize([terms_l1, terms_l2], sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="L1 and L2 amplitudes")

    # Get 1RDM expressions:
    with common.FunctionPrinter(
            file_printer,
            "make_rdm1_f",
            ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
            ["rdm1_f"],
            timer=timer,
    ) as function_printer:
        function_printer.write_python("    delta_oo = np.eye(nocc)")
        function_printer.write_python("    delta_vv = np.eye(nvir)\n")

        def case(i, j, return_value, comment=None):
            ops = [FOperator(j, True), FOperator(i, False)]
            P = Expression([Term(1, [], [Tensor([i, j], "")], ops, [])])
            mid = wick.bch(P, T, max_commutator=2)[-1]
            full = mid + L * mid
            out = apply_wick(full)
            out.resolve()
            expr = AExpression(Ex=out)
            terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
            terms = transform_spin(terms, indices)
            terms = codegen.sympy_to_drudge(terms, indices, dr=dr)
            function_printer.write_latex(terms.latex(), comment=comment)
            return terms

        # Blocks:
        terms = [
            case(i, j, "rdm1_f_oo", comment="oo block"),
            case(i, a, "rdm1_f_ov", comment="ov block"),
            case(a, i, "rdm1_f_vo", comment="vo block"),
            case(a, b, "rdm1_f_vv", comment="vv block"),
        ]

        terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="1RDM")
        function_printer.write_python("    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])\n")

    # Get 2RDM expressions:
    with common.FunctionPrinter(
            file_printer,
            "make_rdm2_f",
            ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
            ["rdm2_f"],
            timer=timer,
    ) as function_printer:
        function_printer.write_python("    delta_oo = np.eye(nocc)")
        function_printer.write_python("    delta_vv = np.eye(nvir)\n")

        def case(i, j, k, l, return_value, comment=None):
            ops = [FOperator(i, True), FOperator(j, True), FOperator(l, False), FOperator(k, False)]
            P = Expression([Term(1, [], [Tensor([i, j, k, l], "")], ops, [])])
            mid = wick.bch(P, T, max_commutator=4)[-1]
            full = mid + L * mid
            out = apply_wick(full)
            out.resolve()
            expr = AExpression(Ex=out)
            terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
            terms = transform_spin(terms, indices)
            terms = codegen.sympy_to_drudge(terms, indices, dr=dr)
            function_printer.write_latex(terms.latex(), comment=comment)
            return terms

        # Blocks:  NOTE: transposed is applied
        terms = [
            case(i, j, k, l, "rdm2_f_oooo", comment="oooo block"),
            case(i, j, k, a, "rdm2_f_ooov", comment="ooov block"),
            case(i, j, a, k, "rdm2_f_ovoo", comment="oovo block"),
            case(i, a, j, k, "rdm2_f_oovo", comment="ovoo block"),
            case(a, i, j, k, "rdm2_f_vooo", comment="vooo block"),
            case(i, j, a, b, "rdm2_f_ovov", comment="oovv block"),
            case(i, a, j, b, "rdm2_f_oovv", comment="ovov block"),
            case(i, a, b, j, "rdm2_f_ovvo", comment="ovvo block"),
            case(a, i, j, b, "rdm2_f_voov", comment="voov block"),
            case(a, i, b, j, "rdm2_f_vvoo", comment="vovo block"),
            case(a, b, i, j, "rdm2_f_vovo", comment="vvoo block"),
            case(i, a, b, c, "rdm2_f_ovvv", comment="ovvv block"),
            case(a, i, b, c, "rdm2_f_vvov", comment="vovv block"),
            case(a, b, i, c, "rdm2_f_vovv", comment="vvov block"),
            case(a, b, c, i, "rdm2_f_vvvo", comment="vvvo block"),
            case(a, b, c, d, "rdm2_f_vvvv", comment="vvvv block"),
        ]

        terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="2RDM")
        function_printer.write_python("    rdm2_f = common.pack_2e(%s)\n" % ", ".join(["rdm2_f_%s" % x for x in common.ov_2e]))

    # Get IP and EA moment expressions:
    for ket, ket_name in [(True, "bra"), (False, "ket")]:
        for ip, ip_name in [(True, "ip"), (False, "ea")]:
            with common.FunctionPrinter(
                    file_printer,
                    "make_%s_mom_%ss" % (ip_name, ket_name),
                    ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
                    ["r1", "r2"],
                    timer=timer,
            ) as function_printer:
                spaces = (braip if ip else braea) if ket else (ketip if ip else ketea)

                function_printer.write_python("    delta_oo = np.eye(nocc)")
                function_printer.write_python("    delta_vv = np.eye(nvir)\n")

                all_terms = []
                for space_no, space in enumerate(spaces):
                    for ind, ind_name in zip([i, a], ["o", "v"]):
                        return_value = "r%d_%s" % (space_no+1, ind_name)

                        ops = [FOperator(ind, (not ip) if ket else ip)]
                        R = Expression([Term(1, [], [Tensor([ind], "")], ops, [])])
                        mid = wick.bch(R, T, max_commutator=4)[-1]

                        if ket:
                            full = space * mid
                        else:
                            full = (L * mid + mid) * space

                        out = apply_wick(full)
                        out.resolve()
                        expr = AExpression(Ex=out)

                        if len(expr.terms) == 0:
                            # Bit of a hack to make sure the term definition remains  TODO as a function
                            term = Term(0, [], [t for t in out.terms[0].tensors if t.name == ""], [], [])
                            expr = AExpression(terms=[ATerm(term)], simplify=False)

                        terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
                        if all(all(f == 0 for f in term.rhs) for term in terms):
                            name = terms[0].lhs.base.name
                            shape = []
                            for index in terms[0].lhs.external_indices:
                                shape.append("nocc" if index.space is codegen.OCCUPIED else "nvir" if index.space is codegen.VIRTUAL else "nbos")
                            function_printer.write_python("    %s = %s((%s), dtype=%s)" % (name, printer._zeros, ", ".join(shape), printer._dtype))
                        else:
                            terms = transform_spin(terms, indices)
                            terms = codegen.sympy_to_drudge(terms, indices, dr=dr)
                            function_printer.write_latex(terms.latex())
                            all_terms.append(terms)

                all_terms = codegen.optimize(all_terms, sizes=sizes, optimize="exhaust", verify=False, interm_fmt="x{}")
                function_printer.write_python(printer.doprint(all_terms)+"\n")
                if ket:
                    function_printer.write_python(
                            "    r1 = np.concatenate([r1_o, r1_v], axis=1)\n"
                            "    r2 = np.concatenate([r2_o, r2_v], axis=3).swapaxes(1, 2)\n"
                    )
                else:
                    function_printer.write_python(
                            "    r1 = np.concatenate([r1_o, r1_v], axis=0)\n"
                            "    r2 = np.concatenate([r2_o, r2_v], axis=0).swapaxes(1, 2)\n"
                    )

    # Get the diagonal of the IP and EA EOM hamiltonians:
    for ip, ip_name in [(True, "ip"), (False, "ea")]:
        with common.FunctionPrinter(
                file_printer,
                "hbar_diag_%s" % ip_name,
                ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
                ["r1", "r2"],
                timer=timer,
        ) as function_printer:
            function_printer.write_python("    delta_oo = np.eye(nocc)")
            function_printer.write_python("    delta_vv = np.eye(nvir)\n")

            E0 = apply_wick(Hbar)
            E0.resolve()
            bras = braip if ip else braea
            kets = ketip if ip else ketea

            def _subs_indices(tensor, subs):
                if not isinstance(tensor, codegen.Tensor):
                    return tensor
                indices = tuple(subs.get(i, i) for i in tensor.indices)
                return tensor.copy(indices=indices)

            all_terms = []
            for space_no, (bra, ket) in enumerate(zip(bras, kets)):
                return_value = "h%d%d" % ((space_no + 1,) * 2)
                diag_return_value = "r%d" % (space_no + 1)

                full = bra * (Hbars[-1] - E0) * ket
                full *= Fraction(1, 2)  # FIXME find where I lost this factor!!
                out = apply_wick(full)
                out.resolve()
                expr = AExpression(Ex=out, simplify=True)

                terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
                terms = transform_spin(terms, indices)
                terms = codegen.sympy_to_drudge(terms, indices, dr=dr)
                function_printer.write_latex(terms.latex())
                einsums = printer.doprint([terms])

                # Convert the einsums to diagonal of the hamiltonian,
                # this is actually kind of difficult algebraically
                # because most tools assume external indices only
                # appear once:
                lines = []
                for line in einsums.split("\n"):
                    if printer._einsum in line:
                        subscript = line.split("\"")[1]
                        inp, out = subscript.split("->")
                        subs = dict(zip(out[len(out)//2:], out[:len(out)//2]))
                        for key, val in subs.items():
                            out = out.replace(key, val)
                        new_subscript = inp + "->" + out[:len(out)//2]
                        einsum = line.replace(return_value, diag_return_value, 1)
                        einsum = einsum.replace(subscript, new_subscript)
                        lines.append(einsum)
                    elif printer._zeros in line:
                        shape = line.replace("(", ")").split(")")[2]
                        new_shape = shape.split(",")
                        new_shape = ", ".join(new_shape[:len(new_shape)//2])
                        zeros = line.replace(return_value, diag_return_value, 1)
                        zeros = zeros.replace(shape, new_shape)
                        lines.append(zeros)
                    else:
                        lines.append(line)

                lines = "\n".join(lines)
                function_printer.write_python(lines+"\n")

            if not ip:
                # r2 for the EA will be calculated as iab, transpose to abi
                function_printer.write_python("    r2 = r2.transpose(1, 2, 0)\n")

    # Get IP and EA EOM hamiltonian-vector product expressions:
    for ip, ip_name in [(True, "ip"), (False, "ea")]:
        with common.FunctionPrinter(
                file_printer,
                "hbar_matvec_%s" % ip_name,
                ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2", "r1", "r2"],
                ["r1new", "r2new"],
                timer=timer,
        ) as function_printer:
            E0 = apply_wick(Hbar)
            E0.resolve()
            spaces = braip if ip else braea
            excitations = rip if ip else rea

            all_terms = []
            for space_no, space in enumerate(spaces):
                return_value = "r%dnew" % (space_no + 1)
                comb_terms = []
                for excit_no, excit in enumerate(excitations):
                    full = space * (Hbars[-1] - E0) * excit
                    full *= Fraction(1, 2)  # FIXME find where I lost this factor!!
                    out = apply_wick(full)
                    out.resolve()
                    expr = AExpression(Ex=out, simplify=True)
                    comb_terms.append(expr)

                terms = comb_terms[0].terms
                for t in comb_terms[1:]:
                    terms += t.terms
                expr = AExpression(terms=terms, simplify=True)

                terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
                terms = transform_spin(terms, indices)
                terms = codegen.sympy_to_drudge(terms, indices, dr=dr)
                function_printer.write_latex(terms.latex())
                all_terms.append(terms)

            all_terms = codegen.optimize(all_terms, sizes=sizes, optimize="exhaust", verify=False, interm_fmt="x{}")
            function_printer.write_python(printer.doprint(all_terms)+"\n")
            if not ip:
                # r2 for the EA will be calculated as iab, transpose to abi
                function_printer.write_python("    r2new = r2new.transpose(1, 2, 0)\n")

