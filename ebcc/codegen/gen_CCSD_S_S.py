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
from ebcc.codegen import common, wick
from ebcc.codegen.convenience_extra import *

from dummy_spark import SparkContext
ctx = SparkContext()
dr = drudge.Drudge(ctx)

warnings.simplefilter("ignore", UserWarning)

# Rank of fermion, boson, coupling operators:
rank = ("SD", "S", "S")

# Spin setting:
spin = "rhf"  # {"ghf", "rhf", "uhf"}

# Indices
occs = i, j, k, l = [Idx(n, "occ") for n in range(4)]
virs = a, b, c, d = [Idx(n, "vir") for n in range(4)]
nms = x, y = [Idx(n, "nm") for n in range(2)]
sizes = {"nocc": sympy.Symbol("N")*2, "nvir": sympy.Symbol("N")*4, "nbos": sympy.Symbol("N")}

# Tensors
H, _ = wick.get_hamiltonian(rank=rank)
bra = bra1, bra2 = wick.get_bra_spaces(rank=rank, occs=occs, virs=virs, nms=nms)
ket = ket1, ket2 = wick.get_ket_spaces(rank=rank, occs=occs, virs=virs, nms=nms)
braip = bra1ip, bra2ip = wick.get_bra_ip_spaces(rank=rank, occs=occs, virs=virs, nms=nms)
braea = bra1ea, bra2ea = wick.get_bra_ea_spaces(rank=rank, occs=occs, virs=virs, nms=nms)
ketip = ket1ip, ket2ip = wick.get_ket_ip_spaces(rank=rank, occs=occs, virs=virs, nms=nms)
ketea = ket1ea, ket2ea = wick.get_ket_ea_spaces(rank=rank, occs=occs, virs=virs, nms=nms)
rip = r1ip, r2ip = wick.get_r_ip_spaces(rank=rank, occs=occs, virs=virs, nms=nms)
rea = r1ea, r2ea = wick.get_r_ea_spaces(rank=rank, occs=occs, virs=virs, nms=nms)
T, _ = wick.get_excitation_ansatz(rank=rank, occs=occs, virs=virs, nms=nms)
L, _ = wick.get_deexcitation_ansatz(rank=rank, occs=occs, virs=virs, nms=nms)
Hbars = wick.construct_hbar(H, T, max_commutator=5)
Hbar = Hbars[-2]

# Printer
printer = common.get_printer(spin)

# Get prefix and spin transformation function according to setting:
transform_spin, prefix = common.get_transformation_function(spin)

# Declare particle types:
particles = common.particles

# Timer:
timer = common.Stopwatch()

# FunctionPrinter with some extra bits:
class FunctionPrinter(common.FunctionPrinter):
    def __init__(self, *args, init_gc=True, remove_w_diagonal=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_gc = init_gc
        self.remove_w_diagonal = remove_w_diagonal

    def __enter__(self):
        # Initialise python function
        self = super().__enter__()
        if self.remove_w_diagonal:
            self.write_python(
                    "    # Remove diagonal from omega:\n"
                    "    w = w - np.diag(np.diag(w))\n"
            )
        if self.init_gc:
            self.write_python(
                    "    # Get boson coupling creation array:\n"
                    "    gc = SimpleNamespace(\n"
                    "        boo=g.boo.transpose(0, 2, 1),\n"
                    "        bov=g.bvo.transpose(0, 2, 1),\n"
                    "        bvo=g.bov.transpose(0, 2, 1),\n"
                    "        bvv=g.bvv.transpose(0, 2, 1),\n"
                    "    )\n"
            )
        return self

with common.FilePrinter("%sCC%s" % (prefix.upper(), "_".join(rank).rstrip("_"))) as file_printer:
    # Get energy expression:
    with FunctionPrinter(
            file_printer,
            "energy",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "u11"],
            ["e_cc"],
            init_gc=False,
            return_dict=False,
            timer=timer,
    ) as function_printer:
        out = apply_wick(Hbar)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="e_cc")
        terms = codegen.ghf_to_rhf(terms, indices)
        terms = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms.latex(), comment="Energy")
        terms = codegen.optimize([terms], sizes=sizes, optimize="exhaust", verify=True, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="Energy")

    # Get amplitudes function:
    with FunctionPrinter(
            file_printer,
            "update_amps",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "u11"],
            ["t1new", "t2new", "s1new", "u11new"],
            timer=timer,
    ) as function_printer:
        # T1 residuals:
        S = bra1 * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="t1new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(codegen.ALPHA, codegen.ALPHA)])
        terms_t1 = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms_t1.latex(), comment="T1 amplitude")

        # T2 residuals:
        S = bra2 * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="t2new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)])
        terms_t2 = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms_t2.latex(), comment="T2 amplitude")

        # S1 residuals:
        S = bra1b * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="s1new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(None,)])
        terms_s1 = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms_s1.latex(), comment="S1 amplitude")

        # U11 residuals:
        S = bra1b1e * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="u11new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(None, codegen.ALPHA, codegen.ALPHA)])
        terms_u11 = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms_u11.latex(), comment="U11 amplitude")

        terms = codegen.optimize([terms_t1, terms_t2, terms_s1, terms_u11], sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="T1, T2, S1 and U11 amplitudes")

    # Get lambda amplitudes function:
    with FunctionPrinter(
            file_printer,
            "update_lams",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "u11", "l1", "l2", "ls1", "lu11"],
            ["l1new", "l2new", "ls1new", "lu11new"],
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
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(codegen.ALPHA, codegen.ALPHA)])
        terms_l1 = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms_l1.latex(), comment="L1 amplitude")

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
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)])
        terms_l2 = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms_l2.latex(), comment="L2 amplitude")

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
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(None,)])
        terms_ls1 = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms_ls1.latex(), comment="LS1 amplitude")

        # BE1 residuals <0|Hbar|EPS1> (not proportional to lambda)
        S = Hbars[0] * ket1b1e
        out = apply_wick(S)
        out.resolve()
        expr1 = AExpression(Ex=out)
        expr1 = expr1.get_connected()
        expr1.sort_tensors()

        # BE1 residuals <0|L Hbar|EPS1> (connected pieces proportional to lambda)
        S = L * Hbars[2] * ket1b1e
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
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(None, codegen.ALPHA, codegen.ALPHA)])
        terms_lu11 = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms_lu11.latex(), comment="LU11 amplitude")

        terms = codegen.optimize([terms_l1, terms_l2, terms_ls1, terms_lu11], sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="L1, L2, LS1 and LU11 amplitudes")

    # Get 1RDM expressions:
    with FunctionPrinter(
            file_printer,
            "make_rdm1_f",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "u11", "l1", "l2", "ls1", "lu11"],
            ["rdm1_f"],
            return_dict=False,
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
            terms = codegen.ghf_to_rhf(terms, indices)
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
    with FunctionPrinter(
            file_printer,
            "make_rdm2_f",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "u11", "l1", "l2", "ls1", "lu11"],
            ["rdm2_f"],
            return_dict=False,
            timer=timer,
    ) as function_printer:
        function_printer.write_python("    delta_oo = np.eye(nocc)")
        function_printer.write_python("    delta_vv = np.eye(nvir)\n")

        def case(i, j, k, l, return_value, comment=None):
            ops = [FOperator(l, True), FOperator(k, True), FOperator(i, False), FOperator(j, False)]
            P = Expression([Term(1, [], [Tensor([i, j, k, l], "")], ops, [])])
            mid = wick.bch(P, T, max_commutator=4)[-1]
            full = mid + L * mid
            out = apply_wick(full)
            out.resolve()
            expr = AExpression(Ex=out)
            terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
            terms = codegen.ghf_to_rhf(terms, indices)
            terms = codegen.sympy_to_drudge(terms, indices, dr=dr)
            function_printer.write_latex(terms.latex(), comment=comment)
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

        terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="2RDM")
        function_printer.write_python("    rdm2_f = common.pack_2e(%s)\n" % ", ".join(["rdm2_f_%s" % x for x in common.ov_2e]))

    # Get single boson DM expressions:
    with FunctionPrinter(
            file_printer,
            "make_sing_b_dm",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "u11", "l1", "l2", "ls1", "lu11"],
            ["dm_b"],
            return_dict=False,
            timer=timer,
    ) as function_printer:
        def case(i, cre, return_value, comment=None):
            ops = [BOperator(i, cre)]
            P = Expression([Term(1, [], [Tensor([i], "")], ops, [])])
            mid = wick.bch(P, T, max_commutator=3)[-1]
            full = mid + L * mid
            out = apply_wick(full)
            out.resolve()
            expr = AExpression(Ex=out)
            terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
            terms = codegen.ghf_to_rhf(terms, indices)
            terms = codegen.sympy_to_drudge(terms, indices, dr=dr)
            function_printer.write_latex(terms.latex(), comment=comment)
            return terms

        terms = [
            case(x, True, "dm_b_cre"),
            case(y, False, "dm_b_des"),
        ]

        terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="Single boson DM")
        function_printer.write_python(
                "    dm_b = np.array([dm_b_cre, dm_b_des])\n"
        )

    # Get boson 1RDM expressions:
    with FunctionPrinter(
            file_printer,
            "make_rdm1_b",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "u11", "l1", "l2", "ls1", "lu11"],
            ["rdm1_b"],
            return_dict=False,
            timer=timer,
    ) as function_printer:
        ops = [BOperator(x, True), BOperator(y, False)]
        P = Expression([Term(1, [], [Tensor([x, y], "")], ops, [])])
        mid = wick.bch(P, T, max_commutator=3)[-1]
        full = mid + L * mid
        out = apply_wick(full)
        out.resolve()
        expr = AExpression(Ex=out)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="rdm1_b")
        terms = codegen.ghf_to_rhf(terms, indices)
        terms = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms.latex(), comment="Boson 1RDM")

        terms = codegen.optimize([terms], sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="Boson 1RDM")

    # Get boson-fermion coupling RDM expressions:
    with FunctionPrinter(
            file_printer,
            "make_eb_coup_rdm",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "u11", "l1", "l2", "ls1", "lu11"],
            ["rdm_eb"],
            return_dict=False,
            timer=timer,
    ) as function_printer:
        function_printer.write_python("    delta_oo = np.eye(nocc)")
        function_printer.write_python("    delta_vv = np.eye(nvir)\n")

        def case(bos_cre, i, j, return_value, comment=None):
            ops = [BOperator(x, bos_cre), FOperator(i, True), FOperator(j, False)]
            P = Expression([Term(1, [], [Tensor([x, i, j], "")], ops, [])])
            mid = wick.bch(P, T, max_commutator=4)[-1]
            full = mid + L * mid
            out = apply_wick(full)
            out.resolve()
            expr = AExpression(Ex=out)
            terms, indices = codegen.wick_to_sympy(expr, particles, return_value=return_value)
            terms = codegen.ghf_to_rhf(terms, indices)
            terms = codegen.sympy_to_drudge(terms, indices, dr=dr)
            function_printer.write_latex(terms.latex(), comment=comment)
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

        terms = codegen.optimize(terms, sizes=sizes, optimize="trav", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="Boson-fermion coupling RDM")
        function_printer.write_python(
                "    rdm_eb = np.array([\n"
                "            np.block([[rdm_eb_cre_oo, rdm_eb_cre_ov], [rdm_eb_cre_vo, rdm_eb_cre_vv]]),\n"
                "            np.block([[rdm_eb_des_oo, rdm_eb_des_ov], [rdm_eb_des_vo, rdm_eb_des_vv]]),\n"
                "    ])\n"
        )