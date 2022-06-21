"""Script to generate equations for the CCSD model.
"""

import sympy
import drudge
from fractions import Fraction
from qwick.wick import apply_wick
from qwick.index import Idx
from qwick.operator import FOperator
from qwick.expression import *
from qwick.convenience import *
from qwick import codegen
from ebcc.codegen import common

# Define hamiltonian
H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("v", ["occ", "vir"], norder=True)
H = H1 + H2

# Define left projection spaces
bra1 = braE1("occ", "vir")
bra2 = braE2("occ", "vir", "occ", "vir")

# Define right projection spaces
ket1 = ketE1("occ", "vir")
ket2 = ketE2("occ", "vir", "occ", "vir")

# Define ansatz
T1 = E1("t1", ["occ"], ["vir"])
T2 = E2("t2", ["occ"], ["vir"])
T = T1 + T2

# Define deexcitation operators
L1 = E1("l1", ["vir"], ["occ"])
L2 = E2("l2", ["vir"], ["occ"])
L = L1 + L2

# Construct Hbar
HT = commute(H, T)
HTT = commute(HT, T)
HTTT = commute(HTT, T)
HTTTT = commute(HTTT, T)
Hbar = H + HT + Fraction('1/2')*HTT
Hbar += Fraction('1/6')*HTTT + Fraction('1/24')*HTTTT

# Printer
printer = codegen.EinsumPrinter(
        occupancy_tags={
            "v": "{base}.{tags}",
            "f": "{base}.{tags}",
            "delta": "delta_{tags}",
        },
        reorder_axes={
            "v": (0, 2, 1, 3),
            "t1": (1, 0),
            "t2": (2, 3, 0, 1),
            "l1": (1, 0),
            "l2": (2, 3, 0, 1),
            "t1new": (1, 0),
            "t2new": (2, 3, 0, 1),
            "l1new": (1, 0),
            "l2new": (2, 3, 0, 1),
        },
        remove_spacing=True,
        garbage_collection=True,
        base_indent=1,
        einsum="lib.einsum",
        zeros="np.zeros",
        dtype="np.float64",
)

# Declare particle types:
particles = {
        "f": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "v": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "t1": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "t2": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "l1": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "l2": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "t1new": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "t2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "l1new": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "l2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "delta": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        **{"rdm1_%s" % x: ((codegen.FERMION, 0), (codegen.FERMION, 0)) for x in ["oo", "ov", "vo", "vv"]},
        **{"rdm2_%s" % x: ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1))
            for x in ["oooo", "ooov", "oovo", "ovoo", "vooo", "oovv", "ovov", "ovvo", "vovo", "vvoo", "ovvv", "vovv", "vvov", "vvvo", "vvvv"]},
}

with common.FilePrinter("ccsd") as file_printer:
    # Get energy expression:
    with common.FunctionPrinter(
            file_printer,
            "energy",
            ["f", "v", "nocc", "nvir", "t1", "t2"],
            ["e_cc"],
    ) as function_printer:
        out = apply_wick(Hbar)
        out.resolve()
        expr = AExpression(Ex=out)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="e_cc")
        terms = codegen.ghf_to_rhf(terms, indices)
        terms = codegen.sympy_to_drudge(terms, indices)
        function_printer.write_python(printer.doprint([terms])+"\n", comment="CCSD energy")
        function_printer.write_latex(terms.latex(), comment="CCSD energy")

    # Get amplitudes function:
    with common.FunctionPrinter(
            file_printer,
            "update_amps",
            ["f", "v", "nocc", "nvir", "t1", "t2"],
            ["t1new", "t2new"],
    ) as function_printer:
        # T1 residuals:
        S = bra1 * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="t1new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(codegen.ALPHA, codegen.ALPHA)])
        terms = codegen.sympy_to_drudge(terms, indices)
        function_printer.write_python(printer.doprint([terms])+"\n", comment="T1 amplitude")
        function_printer.write_latex(terms.latex(), comment="T1 amplitude")

        # T2 residuals:
        S = bra2 * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="t2new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)])
        terms = codegen.sympy_to_drudge(terms, indices)
        function_printer.write_python(printer.doprint([terms])+"\n", comment="T2 amplitude")
        function_printer.write_latex(terms.latex(), comment="T2 amplitude")

    # Get lambda amplitudes function:
    with common.FunctionPrinter(
            file_printer,
            "update_lams",
            ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
            ["l1new", "l2new"],
    ) as function_printer:
        # L1 residuals <0|Hbar|singles> (not proportional to lambda):
        S = Hbar*ket1
        out = apply_wick(S)
        out.resolve()
        expr1 = AExpression(Ex=out)
        expr1 = expr1.get_connected()
        expr1.sort_tensors()

        # L1 residuals <0|(L Hbar)_c|singles> (connected pieces proportional to Lambda):
        S1 = L*S
        out = apply_wick(S1)
        out.resolve()
        expr2 = AExpression(Ex=out)
        expr2 = expr2.get_connected()
        expr2.sort_tensors()
        terms, indices = codegen.wick_to_sympy(expr1+expr2, particles, return_value="l1new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(codegen.ALPHA, codegen.ALPHA)])
        terms = codegen.sympy_to_drudge(terms, indices)
        function_printer.write_python(printer.doprint([terms])+"\n", comment="L1 amplitude")
        function_printer.write_latex(terms.latex(), comment="L1 amplitude")

        # L2 residuals <0|Hbar|doubles> (not proportional to lambda):
        S = Hbar*ket2
        out = apply_wick(S)
        out.resolve()
        expr1 = AExpression(Ex=out)
        expr1 = expr1.get_connected()
        expr1.sort_tensors()

        # L2 residuals <0|L Hbar|doubles> (connected pieces proportional to lambda):
        S = L*S
        out = apply_wick(S)
        out.resolve()
        expr2 = AExpression(Ex=out)
        expr2 = expr2.get_connected()
        expr2.sort_tensors()

        # L2 residuals (disonnected pieces proportional to lambda):
        P1 = PE1("occ", "vir")
        S = Hbar*P1*L*ket2
        out = apply_wick(S)
        out.resolve()
        expr3 = AExpression(Ex=out)
        expr3.sort_tensors()
        terms, indices = codegen.wick_to_sympy(expr1+expr2+expr3, particles, return_value="l2new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)])
        terms = codegen.sympy_to_drudge(terms, indices)
        function_printer.write_python(printer.doprint([terms])+"\n", comment="L2 amplitude")
        function_printer.write_latex(terms.latex(), comment="L2 amplitude")

    # Get 1RDM expressions:
    with common.FunctionPrinter(
            file_printer,
            "make_rdm1",
            ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
            ["rdm1"],
    ) as function_printer:
        i, a = Idx(0, "occ"), Idx(0, "vir")
        j, b = Idx(1, "occ"), Idx(1, "vir")

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
            terms = codegen.sympy_to_drudge(terms, indices)
            function_printer.write_python(printer.doprint([terms])+"\n", comment=comment)
            function_printer.write_latex(terms.latex(), comment=comment)

        # Blocks:
        case(i, j, "rdm1_oo", comment="oo block")
        case(i, a, "rdm1_ov", comment="ov block")
        case(a, i, "rdm1_vo", comment="vo block")
        case(a, b, "rdm1_vv", comment="vv block")

        function_printer.write_python("    rdm1 = np.block([[rdm1_oo, rdm1_ov], [rdm1_vo, rdm1_vv]])\n")

    # Get 2RDM expressions:
    with common.FunctionPrinter(
            file_printer,
            "make_rdm2",
            ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
            ["rdm2"],
    ) as function_printer:
        i, a = Idx(0, "occ"), Idx(0, "vir")
        j, b = Idx(1, "occ"), Idx(1, "vir")
        k, c = Idx(2, "occ"), Idx(2, "vir")
        l, d = Idx(3, "occ"), Idx(3, "vir")

        function_printer.write_python("    delta_oo = np.eye(nocc)")
        function_printer.write_python("    delta_vv = np.eye(nvir)\n")

        def case(i, j, k, l, return_value, comment=None):
            ops = [FOperator(i, True), FOperator(j, True), FOperator(k, False), FOperator(l, False)]
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
            terms = codegen.sympy_to_drudge(terms, indices)
            function_printer.write_python(printer.doprint([terms])+"\n", comment=comment)
            function_printer.write_latex(terms.latex(), comment=comment)

        # Blocks:
        case(i, j, k, l, "rdm2_oooo", comment="oooo block")
        case(i, j, k, a, "rdm2_ooov", comment="ooov block")
        case(i, j, a, k, "rdm2_oovo", comment="oovo block")
        case(i, a, j, k, "rdm2_ovoo", comment="ovoo block")
        case(a, i, j, k, "rdm2_vooo", comment="vooo block")
        case(i, j, a, b, "rdm2_oovv", comment="oovv block")
        case(i, a, j, b, "rdm2_ovov", comment="ovov block")
        case(i, a, b, j, "rdm2_ovvo", comment="ovvo block")
        case(a, i, b, j, "rdm2_vovo", comment="vovo block")
        case(a, b, i, j, "rdm2_vvoo", comment="vvoo block")
        case(i, a, b, c, "rdm2_ovvv", comment="ovvv block")
        case(a, i, b, c, "rdm2_vovv", comment="vovv block")
        case(a, b, i, c, "rdm2_vvov", comment="vvov block")
        case(a, b, c, i, "rdm2_vvvo", comment="vvvo block")
        case(a, b, c, d, "rdm2_vvvv", comment="vvvv block")

        function_printer.write_python(
                "    rdm2 = np.block([\n"
                "            [[[rdm2_oooo, rdm2_ooov], [rdm2_oovo, rdm2_oovv]],\n"
                "             [[rdm2_ovoo, rdm2_ovov], [rdm2_ovvo, rdm2_ovvv]]],\n"
                "            [[[rdm2_vooo, rdm2_voov], [rdm2_vovo, rdm2_vovv]],\n"
                "             [[rdm2_vvoo, rdm2_vvov], [rdm2_vvvo, rdm2_vvvv]]],\n"
                "    ])\n"
        )
