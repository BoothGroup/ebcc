"""Script to generate equations for the CCSD-11 model.
"""

import sympy
import drudge
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

H, _ = wick.get_hamiltonian(rank=(2, 1, 1))
bra1, bra2, bra1b, bra1b1e = wick.get_bra_spaces(rank=(2, 1, 1))
ket1, ket2, ket1b, ket1b1e = wick.get_ket_spaces(rank=(2, 1, 1))
T, _ = wick.get_excitation_ansatz(rank=(2, 1, 1))
L, _ = wick.get_deexcitation_ansatz(rank=(2, 1, 1))
Hbars = wick.construct_hbar(H, T)
Hbar = Hbars[-1]

# Printer
printer = codegen.EinsumPrinter(
        occupancy_tags={
            "v": "{base}.{tags}",
            "f": "{base}.{tags}",
            "g": "{base}.{tags}",
            "gc": "{base}.{tags}",
            "delta": "delta_{tags}",
        },
        reorder_axes={
            "v": (0, 2, 1, 3),
            "t1": (1, 0),
            "t2": (2, 3, 0, 1),
            "l1": (1, 0),
            "l2": (2, 3, 0, 1),
            "u11": (0, 2, 1),
            "lu11": (0, 2, 1),
            "t1new": (1, 0),
            "t2new": (2, 3, 0, 1),
            "l1new": (1, 0),
            "l2new": (2, 3, 0, 1),
            "u11new": (0, 2, 1),
            "lu11new": (0, 2, 1),
            **{"rdm2_f_%s" % x: (0, 2, 1, 3) for x in common.ov_2e},
        },
        remove_spacing=True,
        garbage_collection=True,
        base_indent=1,
        einsum="lib.einsum",
        zeros="np.zeros",
        dtype="np.float64",
)
sizes = {"nocc": sympy.Symbol("N")*2, "nvir": sympy.Symbol("N")*4, "nbos": sympy.Symbol("N")}

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

# Declare particle types:
particles = {
        "f": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "v": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "G": ((codegen.SCALAR_BOSON, 0),),
        "w": ((codegen.SCALAR_BOSON, 0), (codegen.SCALAR_BOSON, 1)),
        "g": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "gc": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "t1": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "t2": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "l1": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "l2": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "s1": ((codegen.SCALAR_BOSON, 0),),
        "ls1": ((codegen.SCALAR_BOSON, 0),),
        "u11": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "lu11": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "t1new": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "t2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "l1new": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "l2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "s1new": ((codegen.SCALAR_BOSON, 0),),
        "ls1new": ((codegen.SCALAR_BOSON, 0),),
        "u11new": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "lu11new": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "delta": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        **{"rdm1_f_%s" % x: ((codegen.FERMION, 0), (codegen.FERMION, 0)) for x in common.ov_1e},
        **{"rdm2_f_%s" % x: ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)) for x in common.ov_2e},
        "rdm1_b": ((codegen.BOSON, 0), (codegen.BOSON, 0)),
        "dm_b_cre": ((codegen.BOSON, 0),),
        "dm_b_des": ((codegen.BOSON, 0),),
        "dm_b": ((codegen.BOSON, 0),),
        **{"rdm_eb_%s_%s" % (x, y): ((codegen.BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)) for y in common.ov_1e for x in ("cre", "des")},
}

with common.FilePrinter("ccsd_1_1") as file_printer:
    # Get energy expression:
    with FunctionPrinter(
            file_printer,
            "energy",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "u11"],
            ["e_cc"],
            init_gc=False,
            timer=timer,
    ) as function_printer:
        out = apply_wick(Hbar)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="e_cc")
        terms = codegen.ghf_to_rhf(terms, indices)
        terms = codegen.sympy_to_drudge(terms, indices, dr=dr)
        function_printer.write_latex(terms.latex(), comment="CCSD-11 energy")
        terms = codegen.optimize([terms], sizes=sizes, optimize="exhaust", verify=True, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="CCSD-11 energy")

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
            timer=timer,
    ) as function_printer:
        i, a = Idx(0, "occ"), Idx(0, "vir")
        j, b = Idx(1, "occ"), Idx(1, "vir")

        function_printer.write_python("    delta_oo = np.eye(nocc)")
        function_printer.write_python("    delta_vv = np.eye(nvir)\n")

        def case(i, j, return_value, comment=None):
            ops = [FOperator(j, True), FOperator(i, False)]
            P = Expression([Term(1, [], [Tensor([i, j], "")], ops, [])])
            PT = commute(P, T)
            PTT = commute(PT, T)
            mid = P + PT + Fraction("1/2") * PTT
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
            timer=timer,
    ) as function_printer:
        i, a = Idx(0, "occ"), Idx(0, "vir")
        j, b = Idx(1, "occ"), Idx(1, "vir")
        k, c = Idx(2, "occ"), Idx(2, "vir")
        l, d = Idx(3, "occ"), Idx(3, "vir")

        function_printer.write_python("    delta_oo = np.eye(nocc)")
        function_printer.write_python("    delta_vv = np.eye(nvir)\n")

        def case(i, j, k, l, return_value, comment=None):
            ops = [FOperator(l, True), FOperator(k, True), FOperator(i, False), FOperator(j, False)]
            P = Expression([Term(1, [], [Tensor([i, j, k, l], "")], ops, [])])
            PT = commute(P, T)
            PTT = commute(PT, T)
            PTTT = commute(PTT, T)
            PTTTT = commute(PTTT, T)
            mid = P + PT + Fraction("1/2")*PTT + Fraction("1/6")*PTTT
            mid += Fraction("1/24")*PTTTT
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
            timer=timer,
    ) as function_printer:
        I = Idx(0, "nm", fermion=False)
        J = Idx(1, "nm", fermion=False)

        def case(i, cre, return_value, comment=None):
            ops = [BOperator(i, cre)]
            P = Expression([Term(1, [], [Tensor([i], "")], ops, [])])
            PT = commute(P, T)
            PTT = commute(PT, T)
            PTTT = commute(PTT, T)
            mid = P + PT + Fraction("1/2")*PTT + Fraction("1/6")*PTTT
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
            case(I, True, "dm_b_cre"),
            case(J, False, "dm_b_des"),
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
            timer=timer,
    ) as function_printer:
        I = Idx(0, "nm", fermion=False)
        J = Idx(1, "nm", fermion=False)

        ops = [BOperator(I, True), BOperator(J, False)]
        P = Expression([Term(1, [], [Tensor([I, J], "")], ops, [])])
        PT = commute(P, T)
        PTT = commute(PT, T)
        PTTT = commute(PTT, T)
        mid = P + PT + Fraction("1/2")*PTT + Fraction("1/6")*PTTT
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
            timer=timer,
    ) as function_printer:
        I = Idx(0, "nm", fermion=False)
        i = Idx(1, "occ")
        a = Idx(1, "vir")
        j = Idx(2, "occ")
        b = Idx(2, "vir")

        function_printer.write_python("    delta_oo = np.eye(nocc)")
        function_printer.write_python("    delta_vv = np.eye(nvir)\n")

        def case(bos_cre, i, j, return_value, comment=None):
            ops = [BOperator(I, bos_cre), FOperator(i, True), FOperator(j, False)]
            P = Expression([Term(1, [], [Tensor([I, i, j], "")], ops, [])])
            PT = commute(P, T)
            PTT = commute(PT, T)
            PTTT = commute(PTT, T)
            PTTTT = commute(PTTT, T)
            mid = P + PT + Fraction("1/2")*PTT + Fraction("1/6")*PTTT + Fraction("1/24")*PTTT
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
