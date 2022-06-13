"""Script to generate equations for the CCSD model.
"""

import sympy
import drudge
from fractions import Fraction
from qwick.expression import AExpression
from qwick.wick import apply_wick
from qwick.convenience import one_e, two_e, E1, E2, PE1, braE1, braE2, ketE1, ketE2, commute
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
