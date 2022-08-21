"""Script to generate equations for the CCSD-11 model.
"""

import sympy
import drudge
import warnings
from fractions import Fraction
from qwick.expression import AExpression
from qwick.wick import apply_wick
from qwick.convenience import *
from qwick import codegen
from ebcc.util import pack_2e, wick
from ebcc.codegen.convenience_extra import *

from pyspark import SparkContext
ctx = SparkContext("local[4]")
#from dummy_spark import SparkContext
#ctx = SparkContext()
dr = drudge.Drudge(ctx)

warnings.simplefilter("ignore", UserWarning)

# Rank of fermion, boson, coupling operators:
rank = ("SD", "SD", "SD")

# Spin setting:
spin = "uhf"  # {"ghf", "rhf", "uhf"}

# Indices
occs = i, j, k, l = [Idx(n, "occ") for n in range(4)]
virs = a, b, c, d = [Idx(n, "vir") for n in range(4)]
nms = w, x, y, z = [Idx(n, "nm", fermion=False) for n in range(4)]
sizes = common.get_sizes(200, 500, 100, spin)

# Tensors
H, _ = wick.get_hamiltonian(rank=rank)
bra = bra1, bra2, bra1b, bra2b, bra1b1e, bra2b1e = wick.get_bra_spaces(rank=rank, occs=occs, virs=virs, nms=nms)
ket = ket1, ket2, ket1b, ket2b, ket1b1e, ket2b1e = wick.get_ket_spaces(rank=rank, occs=occs, virs=virs, nms=nms)
T, _ = wick.get_excitation_ansatz(rank=rank, occs=occs, virs=virs, nms=nms)
L, _ = wick.get_deexcitation_ansatz(rank=rank, occs=occs, virs=virs, nms=nms)
Hbars = wick.construct_hbar(H, T, max_commutator=4)
Hbar = Hbars[-1]

# Printer
printer = common.get_printer(spin)
FunctionPrinter = common.get_function_printer(spin, has_bosons=True)

# Get prefix and spin transformation function according to setting:
transform_spin, prefix = common.get_transformation_function(spin)

# Declare particle types:
particles = common.particles

# Timer:
timer = common.Stopwatch()

with common.FilePrinter("%sCCSD_SD_1_2" % prefix.upper()) as file_printer:
    # Get energy expression:
    with FunctionPrinter(
            file_printer,
            "energy",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "s2", "u11", "u12"],
            ["e_cc"],
            init_gc=False,
            return_dict=False,
            timer=timer,
    ) as function_printer:
        out = apply_wick(Hbar)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="e_cc")
        terms = transform_spin(terms, indices)
        terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        terms = codegen.spin_integrate._flatten(terms)
        terms = codegen.optimize(terms, sizes=sizes, optimize="exhaust", verify=True, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="Energy")

    # Get amplitudes function:
    with FunctionPrinter(
            file_printer,
            "update_amps",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "s2", "u11", "u12"],
            ["t1new", "t2new", "s1new", "s2new", "u11new", "u12new"],
            spin_cases={
                "t1new": ["aa", "bb"],
                "t2new": ["abab", "baba", "aaaa", "bbbb"],
                "u11new": ["aa", "bb"],
                "u12new": ["aa", "bb"],
            },
            timer=timer,
    ) as function_printer:
        # T1 residuals:
        S = bra1 * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="t1new")
        terms = transform_spin(terms, indices, project_onto=[(codegen.ALPHA, codegen.ALPHA)])
        terms_t1 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        # T2 residuals:
        S = bra2 * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="t2new")
        terms = transform_spin(terms, indices, project_onto=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)])
        terms_t2 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        # S1 residuals:
        S = bra1b * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="s1new")
        terms = transform_spin(terms, indices, project_onto=[(None,)])
        terms_s1 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        # S2 residuals:
        S = bra2b * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="s2new")
        terms = transform_spin(terms, indices, project_onto=[(None, None)])
        terms_s2 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        # U11 residuals:
        S = bra1b1e * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="u11new")
        terms = transform_spin(terms, indices, project_onto=[(None, codegen.ALPHA, codegen.ALPHA)])
        terms_u11 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        # U12 residuals:
        S = bra2b1e * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="u12new")
        terms = transform_spin(terms, indices, project_onto=[(None, None, codegen.ALPHA, codegen.ALPHA)])
        terms_u12 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        terms = codegen.spin_integrate._flatten([terms_t1, terms_t2, terms_s1, terms_s2, terms_u11, terms_u12])
        terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="T1, T2, S1, S2, U11 and U12 amplitudes")

    # Get lambda amplitudes function:
    with FunctionPrinter(
            file_printer,
            "update_lams",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "s2", "u11", "u12", "l1", "l2", "ls1", "ls2", "lu11", "lu12"],
            ["l1new", "l2new", "ls1new", "ls2new", "lu11new", "lu12new"],
            spin_cases={
                "l1new": ["aa", "bb"],
                "l2new": ["abab", "baba", "aaaa", "bbbb"],
                "lu11new": ["aa", "bb"],
                "lu12new": ["aa", "bb"],
            },
            timer=timer,
    ) as function_printer:
        # L1 residuals <0|Hbar|singles> (not proportional to lambda):
        S = Hbars[1]*ket1
        out = apply_wick(S)
        out.resolve()
        expr1 = AExpression(Ex=out)
        expr1 = expr1.get_connected()
        expr1.sort_tensors()

        # L1 residuals <0|(L Hbar)_c|singles> (connected pieces proportional to Lambda):
        S = L * Hbars[3] * ket1
        out = apply_wick(S)
        out.resolve()
        expr2 = AExpression(Ex=out)
        expr2 = expr2.get_connected()
        expr2.sort_tensors()
        terms, indices = codegen.wick_to_sympy(expr1+expr2, particles, return_value="l1new")
        terms = transform_spin(terms, indices, project_onto=[(codegen.ALPHA, codegen.ALPHA)])
        terms_l1 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        # L2 residuals <0|Hbar|doubles> (not proportional to lambda):
        S = Hbars[0] * ket2
        out = apply_wick(S)
        out.resolve()
        expr1 = AExpression(Ex=out)
        expr1 = expr1.get_connected()
        expr1.sort_tensors()

        # L2 residuals <0|L Hbar|doubles> (connected pieces proportional to lambda):
        S = L * Hbars[2] * ket2
        out = apply_wick(S)
        out.resolve()
        expr2 = AExpression(Ex=out)
        expr2 = expr2.get_connected()
        expr2.sort_tensors()

        # L2 residuals (disonnected pieces proportional to lambda):
        P1 = PE1("occ", "vir")
        S = Hbars[1] * P1 * L * ket2
        out = apply_wick(S)
        out.resolve()
        expr3 = AExpression(Ex=out)
        expr3.sort_tensors()
        terms, indices = codegen.wick_to_sympy(expr1+expr2+expr3, particles, return_value="l2new")
        terms = transform_spin(terms, indices, project_onto=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)])
        terms_l2 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        # B1 residuals <0|Hbar|1b> (not proportional to lambda):
        S = Hbars[1] * ket1b
        out = apply_wick(S)
        out.resolve()
        expr1 = AExpression(Ex=out)
        expr1 = expr1.get_connected()
        expr1.sort_tensors()

        # B1 residuals <0|L Hbar|1b> (connected pieces proportional to lambda)
        S = L * Hbars[3] * ket1b
        out = apply_wick(S)
        out.resolve()
        expr2 = AExpression(Ex=out)
        expr2 = expr2.get_connected()
        expr2.sort_tensors()
        terms, indices = codegen.wick_to_sympy(expr1+expr2, particles, return_value="ls1new")
        terms = transform_spin(terms, indices, project_onto=[(None,)])
        terms_ls1 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        # B2 residuals <0|Hbar|2b> (not proportional to lambda):
        S = Hbar * ket2b  # FIXME truncate rank
        out = apply_wick(S)
        out.resolve()
        expr1 = AExpression(Ex=out)
        expr1 = expr1.get_connected()
        expr1.sort_tensors()

        # B2 residuals <0|L Hbar|2b> (connected pieces proportional to lambda)
        S = L * Hbar * ket2b  # FIXME truncate rank
        out = apply_wick(S)
        out.resolve()
        expr2 = AExpression(Ex=out)
        expr2 = expr2.get_connected()
        expr2.sort_tensors()

        # B2 residuals (disonnected pieces proportional to lambda):
        P1 = PB1("nm")
        S = Hbar * P1 * L * ket2b  # FIXME truncate rank
        out = apply_wick(S)
        out.resolve()
        expr3 = AExpression(Ex=out)
        expr3.sort_tensors()
        terms, indices = codegen.wick_to_sympy(expr1+expr2+expr3, particles, return_value="ls2new")
        terms = transform_spin(terms, indices, project_onto=[(None, None)])
        terms_ls2 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        # BE1 residuals <0|Hbar|EPS1> (not proportional to lambda)
        S = Hbars[0] * ket1b1e
        out = apply_wick(S)
        out.resolve()
        expr1 = AExpression(Ex=out)
        expr1 = expr1.get_connected()
        expr1.sort_tensors()

        # BE1 residuals <0|L Hbar|EPS1> (connected pieces proportional to lambda)
        S = L * Hbars[0] * ket1b1e
        out = apply_wick(S)
        out.resolve()
        expr2 = AExpression(Ex=out)
        expr2 = expr2.get_connected()
        expr2.sort_tensors()

        # BE1 residuals (disconnected pieces proportional to lambda)
        P1 = PE1("occ", "vir")
        S = Hbars[1] * P1 * L * ket1b1e
        out = apply_wick(S)
        out.resolve()
        expr3 = AExpression(Ex=out)
        expr3.sort_tensors()

        # BE1 residuals (disconnected pieces proportional to lambda, projection onto boson singles)
        P1b = PB1("nm")
        S = Hbars[1] * P1 * L * ket1b1e
        out = apply_wick(S)
        out.resolve()
        expr4 = AExpression(Ex=out)
        expr4.sort_tensors()
        terms, indices = codegen.wick_to_sympy(expr1+expr2+expr3+expr4, particles, return_value="lu11new")
        terms = transform_spin(terms, indices, project_onto=[(None, codegen.ALPHA, codegen.ALPHA)])
        terms_lu11 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        # TODO: check the following, and truncate Hbar

        # BE2 residuals <0|Hbar|EPS1> (not proportional to lambda)
        S = Hbar * ket2b1e
        out = apply_wick(S)
        out.resolve()
        expr1 = AExpression(Ex=out)
        expr1 = expr1.get_connected()
        expr1.sort_tensors()

        # BE2 residuals <0|L Hbar|EPS1> (connected pieces proportional to lambda)
        S = L * Hbar * ket2b1e
        out = apply_wick(S)
        out.resolve()
        expr2 = AExpression(Ex=out)
        expr2 = expr2.get_connected()
        expr2.sort_tensors()

        # BE2 residuals (disconnected pieces proportional to lambda)
        P2 = PE2("occ", "vir")
        S = Hbar * P2 * L * ket2b1e
        out = apply_wick(S)
        out.resolve()
        expr3 = AExpression(Ex=out)
        expr3.sort_tensors()

        # BE2 residuals (disconnected pieces proportional to lambda, projection onto boson singles)
        P2b = PB2("nm")
        S = Hbar * P2 * L * ket2b1e
        out = apply_wick(S)
        out.resolve()
        expr4 = AExpression(Ex=out)
        expr4.sort_tensors()
        terms, indices = codegen.wick_to_sympy(expr1+expr2+expr3+expr4, particles, return_value="lu12new")
        terms = transform_spin(terms, indices, project_onto=[(None, None, codegen.ALPHA, codegen.ALPHA)])
        terms_lu12 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        terms = codegen.spin_integrate._flatten([terms_l1, terms_l2, terms_ls1, terms_ls2, terms_lu11, terms_lu12])
        terms = codegen.optimize(terms, sizes=sizes, optimize="opt", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="L1, L2, LS1, LS2, LU11 and LU12 amplitudes")

    # Get 1RDM expressions:
    with FunctionPrinter(
            file_printer,
            "make_rdm1_f",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "s2", "u11", "u12", "l1", "l2", "ls1", "ls2", "lu11", "lu12"],
            ["rdm1_f"],
            spin_cases={
                "rdm1_f": ["aa", "bb"],
            },
            return_dict=False,
            timer=timer,
    ) as function_printer:
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

        def case(i, j, return_value, comment=None):
            ops = [FOperator(j, True), FOperator(i, False)]
            P = Expression([Term(1, [], [Tensor([i, j], "")], ops, [])])
            mid = wick.bch(P, T, max_commutator=3)[-1]  # FIXME more ranks?
            full = mid + L * mid
            out = apply_wick(full)
            out.resolve()
            expr = AExpression(Ex=out)
            terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
            terms = transform_spin(terms, indices)
            terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]
            return terms

        # Blocks:
        terms = [
            case(i, j, "rdm1_f_oo", comment="oo block"),
            case(i, a, "rdm1_f_ov", comment="ov block"),
            case(a, i, "rdm1_f_vo", comment="vo block"),
            case(a, b, "rdm1_f_vv", comment="vv block"),
        ]
        terms = codegen.spin_integrate._flatten(terms)

        terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="1RDM")

        if spin != "uhf":
            function_printer.write_python("    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])\n")
        else:
            function_printer.write_python(
                "    rdm1_f_aa = np.block([[rdm1_f_oo_aa, rdm1_f_ov_aa], [rdm1_f_vo_aa, rdm1_f_vv_aa]])\n"
                "    rdm1_f_bb = np.block([[rdm1_f_oo_bb, rdm1_f_ov_bb], [rdm1_f_vo_bb, rdm1_f_vv_bb]])\n"
            )

    # Get 2RDM expressions:
    with FunctionPrinter(
            file_printer,
            "make_rdm2_f",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "s2", "u11", "u12", "l1", "l2", "ls1", "ls2", "lu11", "lu12"],
            ["rdm2_f"],
            spin_cases={
                "rdm2_f": ["aaaa", "abab", "baba", "bbbb"],
            },
            return_dict=False,
            timer=timer,
    ) as function_printer:
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

        def case(i, j, k, l, return_value, comment=None):
            ops = [FOperator(l, True), FOperator(k, True), FOperator(i, False), FOperator(j, False)]
            P = Expression([Term(1, [], [Tensor([i, j, k, l], "")], ops, [])])
            mid = wick.bch(P, T, max_commutator=4)[-1]  # FIXME more ranks?
            full = mid + L * mid
            out = apply_wick(full)
            out.resolve()
            expr = AExpression(Ex=out)
            terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
            terms = transform_spin(terms, indices)
            terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]
            return terms

        # Blocks:  NOTE: transposed
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
        terms = codegen.spin_integrate._flatten(terms)

        terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="2RDM")

        if spin != "uhf":
            function_printer.write_python("    rdm2_f = pack_2e(%s)\n" % ", ".join(["rdm2_f_%s" % x for x in common.ov_2e]))
        else:
            function_printer.write_python(""
                    + "    rdm2_f_aaaa = pack_2e(%s)\n" % ", ".join(["rdm2_f_%s_aaaa" % x for x in common.ov_2e])
                    + "    rdm2_f_abab = pack_2e(%s)\n" % ", ".join(["rdm2_f_%s_abab" % x for x in common.ov_2e])
                    + "    rdm2_f_baba = pack_2e(%s)\n" % ", ".join(["rdm2_f_%s_baba" % x for x in common.ov_2e])
                    + "    rdm2_f_bbbb = pack_2e(%s)\n" % ", ".join(["rdm2_f_%s_bbbb" % x for x in common.ov_2e])
            )

    # Get single boson DM expressions:
    with FunctionPrinter(
            file_printer,
            "make_sing_b_dm",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "s2", "u11", "u12", "l1", "l2", "ls1", "ls2", "lu11", "lu12"],
            ["dm_b"],
            return_dict=False,
            timer=timer,
    ) as function_printer:
        def case(i, cre, return_value, comment=None):
            ops = [BOperator(i, cre)]
            P = Expression([Term(1, [], [Tensor([i], "")], ops, [])])
            mid = wick.bch(P, T, max_commutator=4)[-1]  # FIXME more ranks?
            full = mid + L * mid
            out = apply_wick(full)
            out.resolve()
            expr = AExpression(Ex=out)
            terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
            terms = transform_spin(terms, indices)
            terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]
            return terms

        terms = [
            case(x, True, "dm_b_cre"),
            case(y, False, "dm_b_des"),
        ]
        terms = codegen.spin_integrate._flatten(terms)

        terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="Single boson DM")
        function_printer.write_python(
                "    dm_b = np.array([dm_b_cre, dm_b_des])\n"
        )

    # Get boson 1RDM expressions:
    with FunctionPrinter(
            file_printer,
            "make_rdm1_b",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "s2", "u11", "u12", "l1", "l2", "ls1", "ls2", "lu11", "lu12"],
            ["rdm1_b"],
            return_dict=False,
            timer=timer,
    ) as function_printer:
        ops = [BOperator(x, True), BOperator(y, False)]
        P = Expression([Term(1, [], [Tensor([x, y], "")], ops, [])])
        mid = wick.bch(P, T, max_commutator=4)[-1]  # FIXME more ranks?
        full = mid + L * mid
        out = apply_wick(full)
        out.resolve()
        expr = AExpression(Ex=out)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="rdm1_b")
        terms = transform_spin(terms, indices)
        terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        terms = codegen.spin_integrate._flatten(terms)
        terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="Boson 1RDM")

    # Get boson-fermion coupling RDM expressions:
    with FunctionPrinter(
            file_printer,
            "make_eb_coup_rdm",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "s2", "u11", "u12", "l1", "l2", "ls1", "ls2", "lu11", "lu12"],
            ["rdm_eb"],
            spin_cases={
                "rdm_eb": ["aa", "bb"],
            },
            return_dict=False,
            timer=timer,
    ) as function_printer:
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

        def case(bos_cre, i, j, return_value, comment=None):
            ops = [BOperator(x, bos_cre), FOperator(i, True), FOperator(j, False)]
            P = Expression([Term(1, [], [Tensor([x, i, j], "")], ops, [])])
            mid = wick.bch(P, T, max_commutator=4)[-1]  # FIXME more ranks?
            full = mid + L * mid
            out = apply_wick(full)
            out.resolve()
            expr = AExpression(Ex=out)
            terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
            terms = transform_spin(terms, indices)
            terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]
            return terms

        terms = [
            case(True, i, j, "rdm_eb_cre_oo"),
            case(True, i, a, "rdm_eb_cre_ov"),
            case(True, a, i, "rdm_eb_cre_vo"),
            case(True, a, b, "rdm_eb_cre_vv"),
            case(False, i, j, "rdm_eb_des_oo"),
            case(False, i, a, "rdm_eb_des_ov"),
            case(False, a, i, "rdm_eb_des_vo"),
            case(False, a, b, "rdm_eb_des_vv"),
        ]
        terms = codegen.spin_integrate._flatten(terms)

        terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="Boson-fermion coupling RDM")

        if spin != "uhf":
            function_printer.write_python(
                    "    rdm_eb = np.array([\n"
                    "            np.block([[rdm_eb_cre_oo, rdm_eb_cre_ov], [rdm_eb_cre_vo, rdm_eb_cre_vv]]),\n"
                    "            np.block([[rdm_eb_des_oo, rdm_eb_des_ov], [rdm_eb_des_vo, rdm_eb_des_vv]]),\n"
                    "    ])\n"
            )
        else:
            function_printer.write_python(
                    "    rdm_eb_aa = np.array([\n"
                    "            np.block([[rdm_eb_cre_oo_aa, rdm_eb_cre_ov_aa], [rdm_eb_cre_vo_aa, rdm_eb_cre_vv_aa]]),\n"
                    "            np.block([[rdm_eb_des_oo_aa, rdm_eb_des_ov_aa], [rdm_eb_des_vo_aa, rdm_eb_des_vv_aa]]),\n"
                    "    ])\n"
                    "    rdm_eb_bb = np.array([\n"
                    "            np.block([[rdm_eb_cre_oo_bb, rdm_eb_cre_ov_bb], [rdm_eb_cre_vo_bb, rdm_eb_cre_vv_bb]]),\n"
                    "            np.block([[rdm_eb_des_oo_bb, rdm_eb_des_ov_bb], [rdm_eb_des_vo_bb, rdm_eb_des_vv_bb]]),\n"
                    "    ])\n"
            )
