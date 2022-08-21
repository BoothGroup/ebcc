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
from ebcc.util import pack_2e, wick

from dummy_spark import SparkContext
ctx = SparkContext()
dr = drudge.Drudge(ctx)

warnings.simplefilter("ignore", UserWarning)

# Rank of fermion, boson, coupling operators:
rank = ("SD", "", "")

# Spin setting:
spin = "uhf"  # {"ghf", "rhf", "uhf"}

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
ree = r1ee, r2ee = wick.get_r_ee_spaces(rank=rank, occs=occs, virs=virs)
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
    #        function_printer.write_python("    rdm2_f = pack_2e(%s)\n" % ", ".join(["rdm2_f_%s" % x for x in common.ov_2e]))
    #    else:
    #        function_printer.write_python(""
    #                + "    rdm2_f_aaaa = pack_2e(%s)\n" % ", ".join(["rdm2_f_%s_aaaa" % x for x in common.ov_2e])
    #                + "    rdm2_f_aabb = pack_2e(%s)\n" % ", ".join(["rdm2_f_%s_aabb" % x for x in common.ov_2e])
    #                + "    rdm2_f_bbaa = pack_2e(%s)\n" % ", ".join(["rdm2_f_%s_bbaa" % x for x in common.ov_2e])
    #                + "    rdm2_f_bbbb = pack_2e(%s)\n" % ", ".join(["rdm2_f_%s_bbbb" % x for x in common.ov_2e])
    #        )

    #    # TODO fix
    #    if spin == "ghf":
    #        function_printer.write_python("    rdm2_f = rdm2_f.transpose(0, 2, 1, 3)\n")

    if spin == "ghf" or spin == "uhf":
        ## Get IP and EA moment expressions:
        #for is_ket, ket_name in [(True, "ket"), (False, "bra")]:
        #    for ip, ip_name in [(True, "ip"), (False, "ea")]:
        #        with FunctionPrinter(
        #                file_printer,
        #                "make_%s_mom_%ss" % (ip_name, ket_name),
        #                ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
        #                ["%s1" % ket_name, "%s2" % ket_name],
        #                spin_cases = {
        #                    ("%s1" % ket_name): ("aa", "bb"),
        #                    ("%s2" % ket_name): ("aaaa", "abab", "baba", "bbbb"),
        #                },
        #                return_dict=False,
        #                timer=timer,
        #        ) as function_printer:
        #            spaces = (braip if ip else braea) if is_ket else (ketip if ip else ketea)

        #            if spin != "uhf":
        #                function_printer.write_python(
        #                        "    delta_oo = np.eye(nocc)\n"
        #                        "    delta_vv = np.eye(nvir)\n"
        #                )
        #            else:
        #                function_printer.write_python(
        #                        "    delta_oo = SimpleNamespace()\n"
        #                        "    delta_oo.aa = np.eye(nocc[0])\n"
        #                        "    delta_oo.bb = np.eye(nocc[1])\n"
        #                        "    delta_vv = SimpleNamespace()\n"
        #                        "    delta_vv.aa = np.eye(nvir[0])\n"
        #                        "    delta_vv.bb = np.eye(nvir[1])\n"
        #                )

        #            all_terms = []
        #            for space_no, space in enumerate(spaces):
        #                for ind, ind_name in zip([i, a], ["o", "v"]):
        #                    return_value = "%s%d_%s" % (ket_name, space_no+1, ind_name)

        #                    ops = [FOperator(ind, (not ip) if is_ket else ip)]
        #                    R = Expression([Term(1, [], [Tensor([ind], "")], ops, [])])
        #                    mid = wick.bch(R, T, max_commutator=4)[-1]

        #                    if is_ket:
        #                        full = space * mid
        #                    else:
        #                        full = (L * mid + mid) * space

        #                    out = wick.apply_wick(full)
        #                    out.resolve()
        #                    expr = AExpression(Ex=out)

        #                    if len(expr.terms) == 0:
        #                        # Bit of a hack to make sure the term definition remains  TODO as a function
        #                        term = Term(0, [], [t for t in out.terms[0].tensors if t.name == ""], [], [])
        #                        expr = AExpression(terms=[ATerm(term)], simplify=False)

        #                    terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
        #                    if all(all(f == 0 for f in term.rhs) for term in terms):
        #                        if spin != "uhf":
        #                            name = terms[0].lhs.base.name
        #                            shape = []
        #                            for index in terms[0].lhs.external_indices:
        #                                shape.append("nocc" if index.space is codegen.OCCUPIED else "nvir" if index.space is codegen.VIRTUAL else "nbos")
        #                            function_printer.write_python("    %s = %s((%s), dtype=%s)" % (name, printer._zeros, ", ".join(shape), printer._dtype))
        #                        else:
        #                            spins = [[(0, 0), (1, 1)], [(0, 0, 0, 0), (0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)]][space_no]
        #                            for sps in spins:
        #                                name = terms[0].lhs.base.name + ("_%s" % "".join(["ab"[sp] for sp in sps]))
        #                                shape = []
        #                                for index, sp in zip(terms[0].lhs.external_indices, sps):
        #                                    shape.append(("nocc[%d]" % sp) if index.space is codegen.OCCUPIED else ("nvir[%d]" % sp) if index.space is codegen.VIRTUAL else "nbos")
        #                                function_printer.write_python("    %s = %s((%s), dtype=%s)" % (name, printer._zeros, ", ".join(shape), printer._dtype))
        #                    else:
        #                        terms = transform_spin(terms, indices)
        #                        terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]
        #                        all_terms.append(terms)

        #            all_terms = codegen.spin_integrate._flatten(all_terms)
        #            all_terms = codegen.optimize(all_terms, sizes=sizes, optimize="exhaust", verify=False, interm_fmt="x{}")
        #            function_printer.write_python(printer.doprint(all_terms)+"\n")

        #            if spin != "uhf":
        #                function_printer.write_python(""
        #                        + "    %s1 = np.concatenate([%s1_o, %s1_v], axis=%d)\n" % ((ket_name,) * 3 + ((1 if is_ket else 0),))
        #                        + "    %s2 = np.concatenate([%s2_o, %s2_v], axis=%d)\n" % ((ket_name,) * 3 + ((3 if is_ket else 0),))
        #                )
        #            else:
        #                function_printer.write_python(""
        #                        + "    %s1_aa = np.concatenate([%s1_o_aa, %s1_v_aa], axis=%d)\n" % ((ket_name,) * 3 + ((1 if is_ket else 0),))
        #                        + "    %s1_bb = np.concatenate([%s1_o_bb, %s1_v_bb], axis=%d)\n" % ((ket_name,) * 3 + ((1 if is_ket else 0),))
        #                        + "    %s2_aaaa = np.concatenate([%s2_o_aaaa, %s2_v_aaaa], axis=%d)\n" % ((ket_name,) * 3 + ((3 if is_ket else 0),))
        #                        + "    %s2_abab = np.concatenate([%s2_o_abab, %s2_v_abab], axis=%d)\n" % ((ket_name,) * 3 + ((3 if is_ket else 0),))
        #                        + "    %s2_baba = np.concatenate([%s2_o_baba, %s2_v_baba], axis=%d)\n" % ((ket_name,) * 3 + ((3 if is_ket else 0),))
        #                        + "    %s2_bbbb = np.concatenate([%s2_o_bbbb, %s2_v_bbbb], axis=%d)\n" % ((ket_name,) * 3 + ((3 if is_ket else 0),))
        #                )

        #            if spin == "uhf" and not ip:
        #                # FIXME why?
        #                function_printer.write_python(""
        #                        + "    %s2_aaaa *= -1\n" % ket_name
        #                        + "    %s2_abab *= -1\n" % ket_name
        #                        + "    %s2_baba *= -1\n" % ket_name
        #                        + "    %s2_bbbb *= -1\n" % ket_name
        #                )

        ## Get the diagonal of the IP and EA EOM hamiltonians:
        #for ip, ip_name in [(True, "ip"), (False, "ea")]:
        #    with FunctionPrinter(
        #            file_printer,
        #            "hbar_diag_%s" % ip_name,
        #            ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
        #            ["r1", "r2"],
        #            return_dict=False,
        #            timer=timer,
        #    ) as function_printer:
        #        if spin != "uhf":
        #            function_printer.write_python(
        #                    "    delta_oo = np.eye(nocc)\n"
        #                    "    delta_vv = np.eye(nvir)\n"
        #            )
        #        else:
        #            function_printer.write_python(
        #                    "    delta_oo = SimpleNamespace()\n"
        #                    "    delta_oo.aa = np.eye(nocc[0])\n"
        #                    "    delta_oo.bb = np.eye(nocc[1])\n"
        #                    "    delta_vv = SimpleNamespace()\n"
        #                    "    delta_vv.aa = np.eye(nvir[0])\n"
        #                    "    delta_vv.bb = np.eye(nvir[1])\n"
        #            )

        #        E0 = wick.apply_wick(Hbars[-1])
        #        E0.resolve()
        #        bras = braip if ip else braea
        #        kets = ketip if ip else ketea

        #        def _subs_indices(tensor, subs):
        #            if not isinstance(tensor, codegen.Tensor):
        #                return tensor
        #            indices = tuple(subs.get(i, i) for i in tensor.indices)
        #            return tensor.copy(indices=indices)

        #        all_terms = []
        #        for space_no, (bra_space, ket_space) in enumerate(zip(bras, kets)):
        #            return_value = "h%d%d" % ((space_no + 1,) * 2)
        #            diag_return_value = "r%d" % (space_no + 1)

        #            full = bra_space * (Hbars[-1] - E0) * ket_space
        #            out = wick.apply_wick(full)
        #            out.resolve()
        #            expr = AExpression(Ex=out, simplify=True)

        #            terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
        #            terms = transform_spin(terms, indices)
        #            terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]
        #            terms = codegen.spin_integrate._flatten([terms])
        #            einsums = printer.doprint(terms)

        #            # Convert the einsums to diagonal of the hamiltonian,
        #            # this is actually kind of difficult algebraically
        #            # because most tools assume external indices only
        #            # appear once:
        #            lines = []
        #            for line in einsums.split("\n"):
        #                # Ignore parts that aren't diagonal w.r.t spin
        #                if line.strip().startswith("r2"):
        #                    sa = line.strip()[3:6]
        #                    sb = line.strip()[6:9]
        #                    if sa != sb:
        #                        continue
        #                if printer._einsum in line:
        #                    subscript = line.split("\"")[1]
        #                    inp, out = subscript.split("->")
        #                    subs = dict(zip(out[len(out)//2:], out[:len(out)//2]))
        #                    for key, val in subs.items():
        #                        out = out.replace(key, val)
        #                    new_subscript = inp + "->" + out[:len(out)//2]
        #                    einsum = line.replace(return_value, diag_return_value, 1)
        #                    einsum = einsum.replace(subscript, new_subscript)
        #                    lines.append(einsum)
        #                elif printer._zeros in line:
        #                    shape = line.replace("(", ")").split(")")[2]
        #                    new_shape = shape.split(",")
        #                    new_shape = ", ".join(new_shape[:len(new_shape)//2])
        #                    zeros = line.replace(return_value, diag_return_value, 1)
        #                    zeros = zeros.replace(shape, new_shape)
        #                    lines.append(zeros)
        #                else:
        #                    lines.append(line)

        #            lines = "\n".join(lines)
        #            function_printer.write_python(lines+"\n")

        #        if spin == "uhf":
        #            function_printer.write_python(
        #                    "    r1 = SimpleNamespace(a=r1_aa, b=r1_bb)\n"
        #                    "    r2 = SimpleNamespace(aaa=r2_aaaaaa, aba=r2_abaaba, bab=r2_babbab, bbb=r2_bbbbbb)\n"
        #            )

        #        if spin == "uhf" and not ip:
        #            # FIXME why?
        #            function_printer.write_python(
        #                    "    r2.aaa *= -1\n"
        #                    "    r2.aba *= -1\n"
        #                    "    r2.bab *= -1\n"
        #                    "    r2.bbb *= -1\n"
        #            )

        ## Get IP and EA EOM hamiltonian-vector product expressions:
        #for ip, ip_name in [(True, "ip"), (False, "ea")]:
        #    with FunctionPrinter(
        #            file_printer,
        #            "hbar_matvec_%s" % ip_name,
        #            ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2", "r1", "r2"],
        #            ["r1new", "r2new"],
        #            return_dict=False,
        #            timer=timer,
        #    ) as function_printer:
        #        E0 = wick.apply_wick(Hbars[-1])
        #        E0.resolve()
        #        spaces = braip if ip else braea
        #        excitations = rip if ip else rea
        #        excitation = Expression(sum([ex.terms for ex in excitations], []))

        #        all_terms = []
        #        for space_no, space in enumerate(spaces):
        #            return_value = "r%dnew" % (space_no + 1)

        #            full = space * (Hbars[-1] - E0) * excitation
        #            out = wick.apply_wick(full)
        #            out.resolve()
        #            expr = AExpression(Ex=out, simplify=True)

        #            terms, indices = codegen.wick_to_sympy(
        #                    expr,
        #                    particles,
        #                    return_value=return_value,
        #            )
        #            terms = transform_spin(
        #                    terms,
        #                    indices,
        #            )
        #            terms = [codegen.sympy_to_drudge(
        #                    group,
        #                    indices,
        #                    dr=dr,
        #                    restricted=spin!="uhf",
        #            ) for group in terms]
        #            all_terms.append(terms)

        #        all_terms = codegen.spin_integrate._flatten(all_terms)
        #        all_terms = codegen.optimize(all_terms, sizes=sizes, optimize="exhaust", verify=False, interm_fmt="x{}")
        #        function_printer.write_python(printer.doprint(all_terms)+"\n")

        #        if spin == "uhf":
        #            function_printer.write_python(
        #                    "    r1new = SimpleNamespace(a=r1new_a, b=r1new_b)\n"
        #                    "    r2new = SimpleNamespace(aaa=r2new_aaa, aba=r2new_aba, bab=r2new_bab, bbb=r2new_bbb)\n"
        #            )

        #        if spin == "uhf" and not ip:
        #            # FIXME why?
        #            function_printer.write_python(
        #                    "    r2new.aaa *= -1\n"
        #                    "    r2new.aba *= -1\n"
        #                    "    r2new.bab *= -1\n"
        #                    "    r2new.bbb *= -1\n"
        #            )

        ## Get the IP and EA EOM Hamiltonians
        #for ip, ip_name in [(True, "ip"), (False, "ea")]:
        #    with FunctionPrinter(
        #            file_printer,
        #            "hbar_%s" % ip_name,
        #            ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
        #            ["h11", "h12", "h21", "h22"],
        #            return_dict=True,
        #            timer=timer,
        #    ) as function_printer:
        #        function_printer.write_python(
        #                "    delta_oo = np.eye(nocc)\n"
        #                "    delta_vv = np.eye(nvir)\n"
        #        )

        #        E0 = wick.apply_wick(Hbars[-1])
        #        E0.resolve()
        #        bra_spaces = braip if ip else braea
        #        ket_spaces = ketip if ip else ketea

        #        all_terms = []
        #        for bra_no, bra_space in enumerate(bra_spaces):
        #            for ket_no, ket_space in enumerate(ket_spaces):
        #                return_value = "h%d%d" % (bra_no+1, ket_no+1)

        #                full = bra_space * (Hbars[-1] - E0) * ket_space
        #                out = wick.apply_wick(full)
        #                out.resolve()
        #                expr = AExpression(Ex=out, simplify=True)

        #                terms, indices = codegen.wick_to_sympy(
        #                        expr,
        #                        particles,
        #                        return_value=return_value,
        #                )
        #                terms = transform_spin(terms, indices)
        #                terms = [codegen.sympy_to_drudge(
        #                        group,
        #                        indices,
        #                        dr=dr,
        #                        restricted=spin!="uhf",
        #                ) for group in terms]
        #                all_terms.append(terms)

        #        all_terms = codegen.spin_integrate._flatten(all_terms)
        #        all_terms = codegen.optimize(all_terms, sizes=sizes, optimize="greedy", verify=False, interm_fmt="x{}")
        #        function_printer.write_python(printer.doprint(all_terms)+"\n")

        # Get EE moment expressions:
        for is_ket, ket_name in [(True, "ket"), (False, "bra")]:
            with FunctionPrinter(
                    file_printer,
                    "make_ee_mom_%ss" % ket_name,
                    ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
                    ["%see1" % ket_name, "%see2" % ket_name],
                    return_dict=False,
                    timer=timer,
            ) as function_printer:
                spaces = bra if is_ket else ket

                if spin != "uhf":
                    function_printer.write_python(
                            "    delta_oo = np.eye(nocc)\n"
                            "    delta_vv = np.eye(nvir)\n"
                    )
                else:
                    function_printer.write_python(
                            "    delta_oo = SimpleNamespace()\n"
                            "    delta_oo.aa = np.eye(nocc[0])\n"
                            "    delta_oo.bb = np.eye(nocc[1])\n"
                            "    delta_vv = SimpleNamespace()\n"
                            "    delta_vv.aa = np.eye(nvir[0])\n"
                            "    delta_vv.bb = np.eye(nvir[1])\n"
                    )

                all_terms = []
                for space_no, space in enumerate(spaces):
                    for inds, inds_name in zip([(k, l), (k, c), (c, k), (c, d)], ["oo", "ov", "vo", "vv"]):
                        return_value = "%see%d_%s" % (ket_name, space_no+1, inds_name)

                        ops = [FOperator(inds[0], True), FOperator(inds[1], False)]
                        tens = [Tensor(inds, "")]
                        R = Expression([Term(1, [], tens, ops, [])])
                        mid = wick.bch(R, T, max_commutator=4)[-1]

                        if is_ket:
                            full = space * mid
                        else:
                            full = (L * mid + mid) * space

                        if spin != "ghf":
                            full *= Fraction(1, 2)  # FIXME find where I lost this factor!!

                        empty_tensors = [t for t in full.terms[0].tensors if t.name == ""]
                        out = wick.apply_wick(full)
                        out.resolve()
                        expr = AExpression(Ex=out)

                        if len(expr.terms):
                            terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value, skip_symmetry={"%see%s%s" % (xx, yy, zz) for xx in ("ket", "bra") for yy in ("1", "2") for zz in ("", "_oo", "_ov", "_vo", "vv")})
                            terms = transform_spin(terms, indices)
                            if len(terms):
                                terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf", skip_symmetry={"%see%s%s" % (xx, yy, zz) for xx in ("ket", "bra") for yy in ("1", "2") for zz in ("", "_oo", "_ov", "_vo", "vv")}) for group in terms]
                                all_terms.append(terms)

                all_terms = codegen.spin_integrate._flatten(all_terms)
                all_terms = codegen.optimize(all_terms, sizes=sizes, optimize="exhaust", verify=False, interm_fmt="x{}")
                function_printer.write_python(printer.doprint(all_terms)+"\n")

                # Check if any tensors haven't been initialised
                defined = set()
                for term in all_terms:
                    name = term.base.name + "_" + "".join([{"α": "a", "β": "b"}[ind[0].name[1]] for ind in term.exts])
                    defined.add(name)
                for space_no, space in enumerate(spaces):
                    for inds_name in ["oo", "ov", "vo", "vv"]:
                        spins = [
                            [(0, 0, 0, 0), (0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)],
                            [(0,0,0,0,0,0), (0,0,0,1,0,1), (0,0,1,0,1,0), (0,0,1,1,1,1), (1,1,0,0,0,0), (1,1,0,1,0,1), (1,1,1,0,1,0), (1,1,1,1,1,1)]
                        ][space_no]
                        for sps in spins:
                            if is_ket:
                                sps = sps[2:] + sps[:2]
                            spin_name = "".join(["ab"[sp] for sp in sps])
                            name = "%see%d_%s_%s" % (ket_name, space_no+1, inds_name, spin_name)
                            if name not in defined:
                                shape = [index[1].size.name for index in term.exts]
                                if space_no == 0:
                                    perm = inds_name + "ov" if not is_ket else "ov" + inds_name
                                else:
                                    perm = inds_name + "oovv" if not is_ket else "oovv" + inds_name
                                shape = ["%s[%d]" % ({"o": "nocc", "v": "nvir", "b": "nbos"}[pe], sp) for pe, sp in zip(perm, sps)]
                                function_printer.write_python("    %s = %s((%s), dtype=%s)" % (name, printer._zeros, ", ".join(shape), printer._dtype))

                function_printer.write_python("")

                if spin != "uhf":
                    function_printer.write_python(""
                            + "    {name}ee1 = np.concatenate([np.concatenate([{name}ee1_oo, {name}ee1_ov], axis={ax1}), np.concatenate([{name}ee1_vo, {name}ee1_vv], axis={ax1})], axis={ax2})\n".format(name=ket_name, ax1=-1 if is_ket else 1, ax2=-2 if is_ket else 0)
                            + "    {name}ee2 = np.concatenate([np.concatenate([{name}ee2_oo, {name}ee2_ov], axis={ax1}), np.concatenate([{name}ee2_vo, {name}ee2_vv], axis={ax1})], axis={ax2})\n".format(name=ket_name, ax1=-1 if is_ket else 1, ax2=-2 if is_ket else 0)
                    )
                else:
                    part = ""
                    for sp in ("aaaa", "abab", "baba", "bbbb"):
                        part += "    {name}ee1_{sp} = np.concatenate([np.concatenate([{name}ee1_oo_{sp}, {name}ee1_ov_{sp}], axis={ax1}), np.concatenate([{name}ee1_vo_{sp}, {name}ee1_vv_{sp}], axis={ax1})], axis={ax2})\n".format(name=ket_name, sp=sp, ax1=-1 if is_ket else 1, ax2=-2 if is_ket else 0)
                    for sp1 in ("aa", "bb"):
                        for sp2 in ("aaaa", "abab", "baba", "bbbb"):
                            part += "    {name}ee2_{sp1}{sp2} = np.concatenate([np.concatenate([{name}ee2_oo_{sp1}{sp2}, {name}ee2_ov_{sp1}{sp2}], axis={ax1}), np.concatenate([{name}ee2_vo_{sp1}{sp2}, {name}ee2_vv_{sp1}{sp2}], axis={ax1})], axis={ax2})\n".format(name=ket_name, sp1=sp2 if is_ket else sp1, sp2=sp1 if is_ket else sp2, ax1=-1 if is_ket else 1, ax2=-2 if is_ket else 0)
                    function_printer.write_python(part)
                    function_printer.write_python(""
                            + "    {name}ee1 = SimpleNamespace(aaaa={name}ee1_aaaa, abab={name}ee1_abab, baba={name}ee1_baba, bbbb={name}ee1_bbbb)\n".format(name=ket_name)
                            + "    {name}ee2 = SimpleNamespace({a1}aaaa{a2}={name}ee2_{a1}aaaa{a2}, {a1}abab{a2}={name}ee2_{a1}abab{a2}, {a1}baba{a2}={name}ee2_{a1}baba{a2}, {a1}bbbb{a2}={name}ee2_{a1}bbbb{a2}, {b1}aaaa{b2}={name}ee2_{b1}aaaa{b2}, {b1}abab{b2}={name}ee2_{b1}abab{b2}, {b1}baba{b2}={name}ee2_{b1}baba{b2}, {b1}bbbb{b2}={name}ee2_{b1}bbbb{b2})\n".format(name=ket_name, a1="" if is_ket else "aa", a2="aa" if is_ket else "", b1="" if is_ket else "bb", b2="bb" if is_ket else "")
                    )

        ## Get the diagonal of the EE EOM hamiltonians:
        #with FunctionPrinter(
        #        file_printer,
        #        "hbar_diag_ee",
        #        ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
        #        ["ree1", "ree2"],
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

        #    E0 = wick.apply_wick(Hbars[-1])
        #    E0.resolve()
        #    bras = bra
        #    kets = ket

        #    def _subs_indices(tensor, subs):
        #        if not isinstance(tensor, codegen.Tensor):
        #            return tensor
        #        indices = tuple(subs.get(i, i) for i in tensor.indices)
        #        return tensor.copy(indices=indices)

        #    all_terms = []
        #    for space_no, (bra_space, ket_space) in enumerate(zip(bras, kets)):
        #        return_value = "hee%d%d" % ((space_no + 1,) * 2)
        #        diag_return_value = "ree%d" % (space_no + 1)

        #        full = bra_space * (Hbars[-1] - E0) * ket_space
        #        out = wick.apply_wick(full)
        #        out.resolve()
        #        expr = AExpression(Ex=out, simplify=True)

        #        terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
        #        terms = transform_spin(terms, indices)
        #        terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]
        #        terms = codegen.spin_integrate._flatten([terms])
        #        einsums = printer.doprint(terms)

        #        # Convert the einsums to diagonal of the hamiltonian,
        #        # this is actually kind of difficult algebraically
        #        # because most tools assume external indices only
        #        # appear once:
        #        lines = []
        #        for line in einsums.split("\n"):
        #            if printer._einsum in line:
        #                subscript = line.split("\"")[1]
        #                inp, out = subscript.split("->")
        #                subs = dict(zip(out[len(out)//2:], out[:len(out)//2]))
        #                for key, val in subs.items():
        #                    out = out.replace(key, val)
        #                new_subscript = inp + "->" + out[:len(out)//2]
        #                einsum = line.replace(return_value, diag_return_value, 1)
        #                einsum = einsum.replace(subscript, new_subscript)
        #                lines.append(einsum)
        #            elif printer._zeros in line:
        #                shape = line.replace("(", ")").split(")")[2]
        #                new_shape = shape.split(",")
        #                new_shape = ", ".join(new_shape[:len(new_shape)//2])
        #                zeros = line.replace(return_value, diag_return_value, 1)
        #                zeros = zeros.replace(shape, new_shape)
        #                lines.append(zeros)
        #            else:
        #                lines.append(line)

        #        lines = "\n".join(lines)
        #        function_printer.write_python(lines+"\n")

        #    if spin == "uhf":
        #        function_printer.write_python(""
        #                + "    ree1 = SimpleNamespace(%s)\n" % ", ".join(["%s=ree1_%s%s" % ((a+b,) * 3) for a in "ab" for b in "ab"])
        #                + "    ree2 = SimpleNamespace(%s)\n" % ", ".join(["%s=ree2_%s%s" % ((a+b+c+d,) * 3) for a in "ab" for b in "ab" for c in "ab" for d in "ab"])
        #        )

        ## Get EE EOM hamiltonian-vector product expressions:
        #with FunctionPrinter(
        #        file_printer,
        #        "hbar_matvec_ee",
        #        ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2", "r1", "r2"],
        #        ["ree1new", "ree2new"],
        #        return_dict=False,
        #        timer=timer,
        #) as function_printer:
        #    E0 = wick.apply_wick(Hbars[-1])
        #    E0.resolve()
        #    spaces = bra
        #    excitations = ree
        #    excitation = Expression(sum([ex.terms for ex in excitations], []))

        #    all_terms = []
        #    for space_no, space in enumerate(spaces):
        #        return_value = "ree%dnew" % (space_no + 1)

        #        full = space * (Hbars[-1] - E0) * excitation
        #        out = wick.apply_wick(full)
        #        out.resolve()
        #        expr = AExpression(Ex=out, simplify=True)

        #        terms, indices = codegen.wick_to_sympy(
        #                expr,
        #                particles,
        #                return_value=return_value,
        #        )
        #        terms = transform_spin(
        #                terms,
        #                indices,
        #        )
        #        terms = [codegen.sympy_to_drudge(
        #                group,
        #                indices,
        #                dr=dr,
        #                restricted=spin!="uhf",
        #        ) for group in terms]
        #        all_terms.append(terms)

        #    all_terms = codegen.spin_integrate._flatten(all_terms)
        #    all_terms = codegen.optimize(all_terms, sizes=sizes, optimize="greedy", verify=False, interm_fmt="x{}")
        #    function_printer.write_python(printer.doprint(all_terms)+"\n")

        #    if spin == "uhf":
        #        function_printer.write_python(""
        #                + "    ree1new = SimpleNamespace(%s)\n" % ", ".join(["%s=ree1new_%s%s" % ((a+b,) * 3) for a in "ab" for b in "ab"])
        #                + "    ree2new = SimpleNamespace(%s)\n" % ", ".join(["%s=ree2new_%s%s" % ((a+b+c+d,) * 3) for a in "ab" for b in "ab" for c in "ab" for d in "ab"])
        #        )

        ## Get the EE EOM Hamiltonians
        #with FunctionPrinter(
        #        file_printer,
        #        "hbar_ee",
        #        ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
        #        ["hee11", "hee12", "hee21", "hee22"],
        #        return_dict=True,
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

        #    E0 = wick.apply_wick(Hbars[-1])
        #    E0.resolve()
        #    bra_spaces = bra
        #    ket_spaces = ket

        #    all_terms = []
        #    for bra_no, bra_space in enumerate(bra_spaces):
        #        for ket_no, ket_space in enumerate(ket_spaces):
        #            return_value = "hee%d%d" % (bra_no+1, ket_no+1)

        #            full = bra_space * (Hbars[-1] - E0) * ket_space
        #            out = wick.apply_wick(full)
        #            out.resolve()
        #            expr = AExpression(Ex=out, simplify=True)

        #            terms, indices = codegen.wick_to_sympy(
        #                    expr,
        #                    particles,
        #                    return_value=return_value,
        #            )
        #            terms = transform_spin(terms, indices)
        #            terms = [codegen.sympy_to_drudge(
        #                    group,
        #                    indices,
        #                    dr=dr,
        #                    restricted=spin!="uhf",
        #            ) for group in terms]
        #            all_terms.append(terms)

        #    all_terms = codegen.spin_integrate._flatten(all_terms)
        #    all_terms = codegen.optimize(all_terms, sizes=sizes, optimize="greedy", verify=False, interm_fmt="x{}")
        #    function_printer.write_python(printer.doprint(all_terms)+"\n")
