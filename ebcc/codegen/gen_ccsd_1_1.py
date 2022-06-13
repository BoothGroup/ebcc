"""Script to generate equations for the CCSD-11 model.
"""

import sympy
import drudge
from fractions import Fraction
from qwick.expression import AExpression
from qwick.wick import apply_wick
from qwick.convenience import *
from qwick import codegen
from ebcc.codegen import common

# Define hamiltonian
H1 = one_e("f", ["occ", "vir"], norder=True)
H2 = two_e("v", ["occ", "vir"], norder=True)
Hp = one_p("G") + two_p("w")
Hep = ep11("g", ["occ", "vir"], ["nm"], norder=True, name2="gc")
H = H1 + H2 + Hp + Hep

# Define left projection spaces
bra1 = braE1("occ", "vir")
bra2 = braE2("occ", "vir", "occ", "vir")
bra1b = braP1("nm")
bra1b1e = braP1E1("nm", "occ", "vir")

# Define ansatz
T1 = E1("t1", ["occ"], ["vir"])
T2 = E2("t2", ["occ"], ["vir"])
S1 = P1("s1", ["nm"])
U11 = EPS1("u11", ["nm"], ["occ"], ["vir"])
T = T1 + T2 + S1 + U11

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
            "g": "{base}.{tags}",
            "gc": "{base}.{tags}",
        },
        reorder_axes={
            "v": (0, 2, 1, 3),
            "t1": (1, 0),
            "t2": (2, 3, 0, 1),
            "l1": (1, 0),
            "l2": (2, 3, 0, 1),
            "u11": (0, 2, 1),  # FIXME?
            "t1new": (1, 0),
            "t2new": (2, 3, 0, 1),
            "l1new": (1, 0),
            "l2new": (2, 3, 0, 1),
            "u11new": (0, 2, 1),  # FIXME?
        },
        remove_spacing=True,
        garbage_collection=True,
        base_indent=1,
        einsum="lib.einsum",
        zeros="np.zeros",
        dtype="np.float64",
)

class FunctionPrinter(common.FunctionPrinter):
    def __init__(self, *args, init_gc=True, remove_w_diagonal=True, **kwargs):
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
        "u11": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
        "t1new": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "t2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "l1new": ((codegen.FERMION, 0), (codegen.FERMION, 0)),
        "l2new": ((codegen.FERMION, 0), (codegen.FERMION, 1), (codegen.FERMION, 0), (codegen.FERMION, 1)),
        "s1new": ((codegen.SCALAR_BOSON, 0),),
        "u11new": ((codegen.SCALAR_BOSON, 0), (codegen.FERMION, 1), (codegen.FERMION, 1)),
}

with common.FilePrinter("ccsd_1_1") as file_printer:
    # Get energy expression:
    with FunctionPrinter(
            file_printer,
            "energy",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "u11"],
            ["e_cc"],
            init_gc=False,
    ) as function_printer:
        out = apply_wick(Hbar)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="e_cc")
        terms = codegen.ghf_to_rhf(terms, indices)
        terms = codegen.sympy_to_drudge(terms, indices)
        function_printer.write_python(printer.doprint([terms])+"\n", comment="CCSD-11 energy")
        function_printer.write_latex(terms.latex(), comment="CCSD-11 energy")

    # Get amplitudes function:
    with FunctionPrinter(
            file_printer,
            "update_amps",
            ["f", "v", "w", "g", "G", "nocc", "nvir", "nbos", "t1", "t2", "s1", "u11"],
            ["t1new", "t2new", "s1new", "u11new"],
    ) as function_printer:
        # T1 residuals:
        S = bra1 * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="t1new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(codegen.ALPHA, codegen.ALPHA)])
        terms = codegen.sympy_to_drudge(terms, indices)
        function_printer.write_python(printer.doprint([terms])+"\n", comment="T1 amplitude")
        function_printer.write_latex(terms.latex(), comment="T1 amplitude")

        # T2 residuals:
        S = bra2 * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="t2new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)])
        terms = codegen.sympy_to_drudge(terms, indices)
        function_printer.write_python(printer.doprint([terms])+"\n", comment="T2 amplitude")
        function_printer.write_latex(terms.latex(), comment="T2 amplitude")

        # S1 residuals:
        S = bra1b * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="s1new")
        terms = codegen.ghf_to_rhf(terms, indices)
        terms = codegen.sympy_to_drudge(terms, indices)
        function_printer.write_python(printer.doprint([terms])+"\n", comment="S1 amplitude")
        function_printer.write_latex(terms.latex(), comment="S1 amplitude")

        # U11 residuals:
        S = bra1b1e * Hbar
        out = apply_wick(S)
        out.resolve()
        expr = AExpression(Ex=out, simplify=True)
        terms, indices = codegen.wick_to_sympy(expr, particles, return_value="u11new")
        terms = codegen.ghf_to_rhf(terms, indices, project_onto=[(None, codegen.ALPHA, codegen.ALPHA)])
        terms = codegen.sympy_to_drudge(terms, indices)
        function_printer.write_python(printer.doprint([terms])+"\n", comment="U11 amplitude")
        function_printer.write_latex(terms.latex(), comment="U11 amplitude")
