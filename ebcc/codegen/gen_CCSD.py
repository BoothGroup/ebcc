"""Script to generate equations for the CCSD model.
"""

import warnings
import sympy
import drudge
from fractions import Fraction
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

# Rank of fermion, boson, coupling operators:
rank = ("SD", "", "")

# Spin setting:
spin = "ghf"  # {"ghf", "rhf", "uhf"}

# Indices
occs = i, j, k, l = [Idx(n, "occ") for n in range(4)]
virs = a, b, c, d = [Idx(n, "vir") for n in range(4)]
sizes = common.get_sizes(200, 500, 0, spin)

# Tensors
H, _ = wick.get_hamiltonian(rank=rank)
bra = bra1, bra2 = wick.get_bra_spaces(rank=rank, occs=occs, virs=virs)
ket = ket1, ket2 = wick.get_ket_spaces(rank=rank, occs=occs, virs=virs)
braip = bra1ip, bra2ip = wick.get_bra_ip_spaces(rank=rank, occs=occs, virs=virs)
braea = bra1ea, bra2ea = wick.get_bra_ea_spaces(rank=rank, occs=occs, virs=virs)
ketip = ket1ip, ket2ip = wick.get_ket_ip_spaces(rank=rank, occs=occs, virs=virs)
ketea = ket1ea, ket2ea = wick.get_ket_ea_spaces(rank=rank, occs=occs, virs=virs)
rip = r1ip, r2ip = wick.get_r_ip_spaces(rank=rank, occs=occs, virs=virs)
rea = r1ea, r2ea = wick.get_r_ea_spaces(rank=rank, occs=occs, virs=virs)
T, _ = wick.get_excitation_ansatz(rank=rank, occs=occs, virs=virs)
L, _ = wick.get_deexcitation_ansatz(rank=rank, occs=occs, virs=virs)
Hbars = wick.construct_hbar(H, T, max_commutator=5)
Hbar = Hbars[-2]

# Printer
printer = common.get_printer(spin)
FunctionPrinter = common.get_function_printer(spin)

# Get prefix and spin transformation function according to setting:
transform_spin, prefix = common.get_transformation_function(spin)

# Declare particle types:
particles = common.particles

# Timer:
timer = common.Stopwatch()

with common.FilePrinter("%sCCSD" % prefix.upper()) as file_printer:
    ## Get energy expression:
    #with FunctionPrinter(
    #        file_printer,
    #        "energy",
    #        ["f", "v", "nocc", "nvir", "t1", "t2"],
    #        ["e_cc"],
    #        return_dict=False,
    #        timer=timer,
    #) as function_printer:
    #    out = wick.apply_wick(Hbar)
    #    out.resolve()
    #    expr = AExpression(Ex=out)
    #    terms, indices = codegen.wick_to_sympy(expr, particles, return_value="e_cc")
    #    terms = transform_spin(terms, indices)
    #    terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

    #    terms = codegen.spin_integrate._flatten(terms)
    #    terms = codegen.optimize(terms, sizes=sizes, optimize="exhaust", verify=True, interm_fmt="x{}")
    #    function_printer.write_python(printer.doprint(terms)+"\n", comment="energy")

    ## Get amplitudes function:
    #with FunctionPrinter(
    #        file_printer,
    #        "update_amps",
    #        ["f", "v", "nocc", "nvir", "t1", "t2"],
    #        ["t1new", "t2new"],
    #        spin_cases={
    #            "t1new": ["aa", "bb"],
    #            "t2new": ["abab", "baba", "aaaa", "bbbb"],
    #        },
    #        timer=timer,
    #) as function_printer:
    #    # T1 residuals:
    #    S = bra1 * Hbar
    #    out = wick.apply_wick(S)
    #    out.resolve()
    #    expr = AExpression(Ex=out)
    #    terms, indices = codegen.wick_to_sympy(expr, particles, return_value="t1new")
    #    terms = transform_spin(terms, indices, project_rhf=[(codegen.ALPHA, codegen.ALPHA)])
    #    terms_t1 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

    #    # T2 residuals:
    #    S = bra2 * Hbar
    #    out = wick.apply_wick(S)
    #    out.resolve()
    #    expr = AExpression(Ex=out)
    #    terms, indices = codegen.wick_to_sympy(expr, particles, return_value="t2new")
    #    terms = transform_spin(terms, indices, project_rhf=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)])
    #    terms_t2 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

    #    terms = codegen.spin_integrate._flatten([terms_t1, terms_t2])
    #    terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
    #    function_printer.write_python(printer.doprint(terms)+"\n", comment="T1 and T2 amplitudes")

    ## Get lambda amplitudes function:
    #with FunctionPrinter(
    #        file_printer,
    #        "update_lams",
    #        ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
    #        ["l1new", "l2new"],
    #        spin_cases={
    #            "l1new": ["aa", "bb"],
    #            "l2new": ["abab", "baba", "aaaa", "bbbb"],
    #        },
    #        timer=timer,
    #) as function_printer:
    #    # L1 residuals <0|Hbar|singles> (not proportional to lambda):
    #    S = Hbar * ket1
    #    out = wick.apply_wick(S)
    #    out.resolve()
    #    expr1 = AExpression(Ex=out)
    #    expr1 = expr1.get_connected()
    #    expr1.sort_tensors()

    #    # L1 residuals <0|(L Hbar)_c|singles> (connected pieces proportional to Lambda):
    #    S = L * Hbar * ket1
    #    out = wick.apply_wick(S)
    #    out.resolve()
    #    expr2 = AExpression(Ex=out)
    #    expr2 = expr2.get_connected()
    #    expr2.sort_tensors()
    #    terms, indices = codegen.wick_to_sympy(expr1+expr2, particles, return_value="l1new")
    #    terms = transform_spin(terms, indices, project_rhf=[(codegen.ALPHA, codegen.ALPHA)])
    #    terms_l1 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

    #    # L2 residuals <0|Hbar|doubles> (not proportional to lambda):
    #    S = Hbar * ket2
    #    out = wick.apply_wick(S)
    #    out.resolve()
    #    expr1 = AExpression(Ex=out)
    #    expr1 = expr1.get_connected()
    #    expr1.sort_tensors()

    #    # L2 residuals <0|L Hbar|doubles> (connected pieces proportional to lambda):
    #    S = L * Hbar * ket2
    #    out = wick.apply_wick(S)
    #    out.resolve()
    #    expr2 = AExpression(Ex=out)
    #    expr2 = expr2.get_connected()
    #    expr2.sort_tensors()

    #    # L2 residuals (disonnected pieces proportional to lambda):
    #    P1 = PE1("occ", "vir")
    #    S = Hbar * P1 * L * ket2
    #    out = wick.apply_wick(S)
    #    out.resolve()
    #    expr3 = AExpression(Ex=out)
    #    expr3.sort_tensors()
    #    terms, indices = codegen.wick_to_sympy(expr1+expr2+expr3, particles, return_value="l2new")
    #    terms = transform_spin(terms, indices, project_rhf=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)])
    #    terms_l2 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

    #    terms = codegen.spin_integrate._flatten([terms_l1, terms_l2])
    #    terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
    #    function_printer.write_python(printer.doprint(terms)+"\n", comment="L1 and L2 amplitudes")

    ## Get 1RDM expressions:
    #with FunctionPrinter(
    #        file_printer,
    #        "make_rdm1_f",
    #        ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
    #        ["rdm1_f"],
    #        spin_cases={
    #            "rdm1_f": ["aa", "bb"],
    #        },
    #        return_dict=False,
    #        timer=timer,
    #) as function_printer:
    #    if spin != "uhf":
    #        function_printer.write_python(
    #                "    delta_oo = np.eye(nocc)\n"
    #                "    delta_vv = np.eye(nvir)\n"
    #        )
    #    else:
    #        function_printer.write_python(
    #                "    delta_oo = SimpleNamespace()\n"
    #                "    delta_oo.aa = np.eye(nocc[0])\n"
    #                "    delta_oo.bb = np.eye(nocc[1])\n"
    #                "    delta_vv = SimpleNamespace()\n"
    #                "    delta_vv.aa = np.eye(nvir[0])\n"
    #                "    delta_vv.bb = np.eye(nvir[1])\n"
    #        )

    #    def case(i, j, return_value, comment=None):
    #        ops = [FOperator(j, True), FOperator(i, False)]
    #        P = Expression([Term(1, [], [Tensor([i, j], "")], ops, [])])
    #        mid = wick.bch(P, T, max_commutator=2)[-1]
    #        full = mid + L * mid
    #        out = wick.apply_wick(full)
    #        out.resolve()
    #        expr = AExpression(Ex=out)
    #        terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
    #        terms = transform_spin(terms, indices)
    #        terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]
    #        return terms

    #    # Blocks:
    #    terms = [
    #        case(i, j, "rdm1_f_oo", comment="oo block"),
    #        case(i, a, "rdm1_f_ov", comment="ov block"),
    #        case(a, i, "rdm1_f_vo", comment="vo block"),
    #        case(a, b, "rdm1_f_vv", comment="vv block"),
    #    ]
    #    terms = codegen.spin_integrate._flatten(terms)

    #    terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
    #    function_printer.write_python(printer.doprint(terms)+"\n", comment="1RDM")

    #    if spin != "uhf":
    #        function_printer.write_python("    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])\n")
    #    else:
    #        function_printer.write_python(
    #            "    rdm1_f_aa = np.block([[rdm1_f_oo_aa, rdm1_f_ov_aa], [rdm1_f_vo_aa, rdm1_f_vv_aa]])\n"
    #            "    rdm1_f_bb = np.block([[rdm1_f_oo_bb, rdm1_f_ov_bb], [rdm1_f_vo_bb, rdm1_f_vv_bb]])\n"
    #        )

    ## Get 2RDM expressions:
    #with FunctionPrinter(
    #        file_printer,
    #        "make_rdm2_f",
    #        ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
    #        ["rdm2_f"],
    #        spin_cases={
    #            "rdm2_f": ["aaaa", "abab", "baba", "bbbb"],
    #        },
    #        return_dict=False,
    #        timer=timer,
    #) as function_printer:
    #    if spin != "uhf":
    #        function_printer.write_python(
    #                "    delta_oo = np.eye(nocc)\n"
    #                "    delta_vv = np.eye(nvir)\n"
    #        )
    #    else:
    #        function_printer.write_python(
    #                "    delta_oo = SimpleNamespace()\n"
    #                "    delta_oo.aa = np.eye(nocc[0])\n"
    #                "    delta_oo.bb = np.eye(nocc[1])\n"
    #                "    delta_vv = SimpleNamespace()\n"
    #                "    delta_vv.aa = np.eye(nvir[0])\n"
    #                "    delta_vv.bb = np.eye(nvir[1])\n"
    #        )

    #    def case(i, j, k, l, return_value, comment=None):
    #        ops = [FOperator(i, True), FOperator(j, True), FOperator(l, False), FOperator(k, False)]
    #        P = Expression([Term(1, [], [Tensor([i, j, k, l], "")], ops, [])])
    #        mid = wick.bch(P, T, max_commutator=4)[-1]
    #        full = mid + L * mid
    #        out = wick.apply_wick(full)
    #        out.resolve()
    #        expr = AExpression(Ex=out)
    #        expr.sort_tensors()
    #        terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
    #        terms = transform_spin(terms, indices)
    #        terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]
    #        return terms

    #    # FIXME for ghf
    #    if spin != "ghf":
    #        transpose = lambda name: name[:-3] + name[-2] + name[-3] + name[-1]
    #    else:
    #        transpose = lambda name: name

    #    # Blocks:  NOTE: transposed is applied
    #    terms = [
    #        case(i, j, k, l, transpose("rdm2_f_oooo"), comment="oooo block"),
    #        case(i, j, k, a, transpose("rdm2_f_ooov"), comment="ooov block"),
    #        case(i, j, a, k, transpose("rdm2_f_oovo"), comment="oovo block"),
    #        case(i, a, j, k, transpose("rdm2_f_ovoo"), comment="ovoo block"),
    #        case(a, i, j, k, transpose("rdm2_f_vooo"), comment="vooo block"),
    #        case(i, j, a, b, transpose("rdm2_f_oovv"), comment="oovv block"),
    #        case(i, a, j, b, transpose("rdm2_f_ovov"), comment="ovov block"),
    #        case(i, a, b, j, transpose("rdm2_f_ovvo"), comment="ovvo block"),
    #        case(a, i, j, b, transpose("rdm2_f_voov"), comment="voov block"),
    #        case(a, i, b, j, transpose("rdm2_f_vovo"), comment="vovo block"),
    #        case(a, b, i, j, transpose("rdm2_f_vvoo"), comment="vvoo block"),
    #        case(i, a, b, c, transpose("rdm2_f_ovvv"), comment="ovvv block"),
    #        case(a, i, b, c, transpose("rdm2_f_vovv"), comment="vovv block"),
    #        case(a, b, i, c, transpose("rdm2_f_vvov"), comment="vvov block"),
    #        case(a, b, c, i, transpose("rdm2_f_vvvo"), comment="vvvo block"),
    #        case(a, b, c, d, transpose("rdm2_f_vvvv"), comment="vvvv block"),
    #    ]
    #    terms = codegen.spin_integrate._flatten(terms)

    #    terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
    #    function_printer.write_python(printer.doprint(terms)+"\n", comment="2RDM")

    #    if spin != "uhf":
    #        function_printer.write_python("    rdm2_f = common.pack_2e(%s)\n" % ", ".join(["rdm2_f_%s" % x for x in common.ov_2e]))
    #    else:
    #        function_printer.write_python(""
    #                + "    rdm2_f_aaaa = common.pack_2e(%s)\n" % ", ".join(["rdm2_f_%s_aaaa" % x for x in common.ov_2e])
    #                + "    rdm2_f_aabb = common.pack_2e(%s)\n" % ", ".join(["rdm2_f_%s_aabb" % x for x in common.ov_2e])
    #                + "    rdm2_f_bbaa = common.pack_2e(%s)\n" % ", ".join(["rdm2_f_%s_bbaa" % x for x in common.ov_2e])
    #                + "    rdm2_f_bbbb = common.pack_2e(%s)\n" % ", ".join(["rdm2_f_%s_bbbb" % x for x in common.ov_2e])
    #        )

    #    # TODO fix
    #    if spin == "ghf":
    #        function_printer.write_python("    rdm2_f = rdm2_f.transpose(0, 2, 1, 3)\n")

    if spin == "ghf":
        # Get IP and EA moment expressions:
        for ket, ket_name in [(True, "ket"), (False, "bra")]:
            for ip, ip_name in [(True, "ip"), (False, "ea")]:
                with FunctionPrinter(
                        file_printer,
                        "make_%s_mom_%ss" % (ip_name, ket_name),
                        ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
                        ["%s1" % ket_name, "%s2" % ket_name],
                        spin_cases = {
                            ("%s1" % ket_name): ("a", "b"),
                            ("%s2" % ket_name): ("aaa", "aba", "bab", "bbb"),
                        },
                        return_dict=False,
                        timer=timer,
                ) as function_printer:
                    spaces = (braip if ip else braea) if ket else (ketip if ip else ketea)

                    function_printer.write_python(
                            "    delta_oo = np.eye(nocc)\n"
                            "    delta_vv = np.eye(nvir)\n"
                    )

                    all_terms = []
                    for space_no, space in enumerate(spaces):
                        for ind, ind_name in zip([i, a], ["o", "v"]):
                            return_value = "%s%d_%s" % (ket_name, space_no+1, ind_name)

                            ops = [FOperator(ind, (not ip) if ket else ip)]
                            R = Expression([Term(1, [], [Tensor([ind], "")], ops, [])])
                            mid = wick.bch(R, T, max_commutator=4)[-1]

                            if ket:
                                full = space * mid
                            else:
                                full = (L * mid + mid) * space

                            if spin != "ghf":
                                full *= Fraction(1, 2)  # FIXME find where I lost this factor!!

                            out = wick.apply_wick(full)
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
                                terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]
                                all_terms.append(terms)

                    all_terms = codegen.spin_integrate._flatten(all_terms)
                    all_terms = codegen.optimize(all_terms, sizes=sizes, optimize="exhaust", verify=False, interm_fmt="x{}")
                    function_printer.write_python(printer.doprint(all_terms)+"\n")

                    if spin != "uhf":
                        function_printer.write_python(""
                                + "    %s1 = np.concatenate([%s1_o, %s1_v], axis=%d)\n" % ((ket_name,) * 3 + ((1 if ket else 0),))
                                + "    %s2 = np.concatenate([%s2_o, %s2_v], axis=%d)\n" % ((ket_name,) * 3 + ((3 if ket else 0),))
                        )
                    else:
                        function_printer.write_python(""
                                + "    %s1_a = np.concatenate([%s1_o_a, %s1_v_a], axis=%d)\n" % ((ket_name,) * 3 + ((1 if ket else 0),))
                                + "    %s1_b = np.concatenate([%s1_o_b, %s1_v_b], axis=%d)\n" % ((ket_name,) * 3 + ((1 if ket else 0),))
                                + "    %s2_aaa = np.concatenate([%s2_o_aaa, %s2_v_aaa], axis=%d)\n" % ((ket_name,) * 3 + ((3 if ket else 0),))
                                + "    %s2_aba = np.concatenate([%s2_o_aba, %s2_v_aba], axis=%d)\n" % ((ket_name,) * 3 + ((3 if ket else 0),))
                                + "    %s2_bab = np.concatenate([%s2_o_bab, %s2_v_bab], axis=%d)\n" % ((ket_name,) * 3 + ((3 if ket else 0),))
                                + "    %s2_bbb = np.concatenate([%s2_o_bbb, %s2_v_bbb], axis=%d)\n" % ((ket_name,) * 3 + ((3 if ket else 0),))
                        )

                    if not ip:
                        if ket:
                            function_printer.write_python("    ket2 = ket2.transpose(1, 0, 2, 3)\n")
                        else:
                            function_printer.write_python("    bra2 = bra2.transpose(0, 2, 1, 3)\n")

        # Get the diagonal of the IP and EA EOM hamiltonians:
        for ip, ip_name in [(True, "ip"), (False, "ea")]:
            with FunctionPrinter(
                    file_printer,
                    "hbar_diag_%s" % ip_name,
                    ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
                    ["r1", "r2"],
                    return_dict=False,
                    timer=timer,
            ) as function_printer:
                function_printer.write_python(
                        "    delta_oo = np.eye(nocc)\n"
                        "    delta_vv = np.eye(nvir)\n"
                )

                E0 = wick.apply_wick(Hbars[-1])
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
                    out = wick.apply_wick(full)
                    out.resolve()
                    expr = AExpression(Ex=out, simplify=True)

                    terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
                    terms = transform_spin(terms, indices)
                    terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]
                    terms = codegen.spin_integrate._flatten([terms])
                    einsums = printer.doprint(terms)

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

                function_printer.write_python("    r2 = r2.transpose(1, 0, 2)\n")

        # Get IP and EA EOM hamiltonian-vector product expressions:
        for ip, ip_name in [(True, "ip"), (False, "ea")]:
            with FunctionPrinter(
                    file_printer,
                    "hbar_matvec_%s" % ip_name,
                    ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2", "r1", "r2"],
                    ["r1new", "r2new"],
                    return_dict=False,
                    timer=timer,
            ) as function_printer:
                E0 = wick.apply_wick(Hbars[-1])
                E0.resolve()
                spaces = braip if ip else braea
                excitations = rip if ip else rea
                excitation = Expression(sum([ex.terms for ex in excitations], []))

                all_terms = []
                for space_no, space in enumerate(spaces):
                    return_value = "r%dnew" % (space_no + 1)

                    full = space * (Hbars[-1] - E0) * excitation
                    out = wick.apply_wick(full)
                    out.resolve()
                    expr = AExpression(Ex=out, simplify=True)

                    terms, indices = codegen.wick_to_sympy(
                            expr,
                            particles,
                            return_value=return_value,
                    )
                    terms = transform_spin(
                            terms,
                            indices,
                    )
                    terms = [codegen.sympy_to_drudge(
                            group,
                            indices,
                            dr=dr,
                            restricted=spin!="uhf",
                    ) for group in terms]
                    all_terms.append(terms)

                all_terms = codegen.spin_integrate._flatten(all_terms)
                all_terms = codegen.optimize(all_terms, sizes=sizes, optimize="exhaust", verify=False, interm_fmt="x{}")
                function_printer.write_python(printer.doprint(all_terms)+"\n")

                function_printer.write_python("    r2new = r2new.transpose(1, 0, 2)\n")

        # Get the IP and EA EOM Hamiltonians
        for ip, ip_name in [(True, "ip"), (False, "ea")]:
            with FunctionPrinter(
                    file_printer,
                    "hbar_%s" % ip_name,
                    ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
                    ["h11", "h12", "h21", "h22"],
                    return_dict=True,
                    timer=timer,
            ) as function_printer:
                function_printer.write_python(
                        "    delta_oo = np.eye(nocc)\n"
                        "    delta_vv = np.eye(nvir)\n"
                )

                E0 = wick.apply_wick(Hbars[-1])
                E0.resolve()
                bra_spaces = braip if ip else braea
                ket_spaces = ketip if ip else ketea

                all_terms = []
                for bra_no, bra_space in enumerate(bra_spaces):
                    for ket_no, ket_space in enumerate(ket_spaces):
                        return_value = "h%d%d" % (bra_no+1, ket_no+1)

                        full = bra_space * (Hbars[-1] - E0) * ket_space
                        out = wick.apply_wick(full)
                        out.resolve()
                        expr = AExpression(Ex=out, simplify=True)

                        terms, indices = codegen.wick_to_sympy(
                                expr,
                                particles,
                                return_value=return_value,
                        )
                        terms = transform_spin(terms, indices)
                        terms = [codegen.sympy_to_drudge(
                                group,
                                indices,
                                dr=dr,
                                restricted=spin!="uhf",
                        ) for group in terms]
                        all_terms.append(terms)

                all_terms = codegen.spin_integrate._flatten(all_terms)
                all_terms = codegen.optimize(all_terms, sizes=sizes, optimize="exhaust", verify=False, interm_fmt="x{}")
                function_printer.write_python(printer.doprint(all_terms)+"\n")
