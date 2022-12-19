"""Script to generate equations for the CCSDT model.

This uses wicked instead of qwick.
"""

import os
import sys
import json
import pybind11
import qwick
import drudge
from qwick.index import Idx
from qwick import codegen
from ebcc.codegen import common
from wicked_utils import symmetry, antisymmetrise, wicked_indices

from dummy_spark import SparkContext
ctx = SparkContext()
#from pyspark import SparkContext
#ctx = SparkContext("local[8]")
dr = drudge.Drudge(ctx)

# Spin setting:
spin = "rhf"  # {"ghf", "rhf", "uhf"}

# Indices
occs = i, j, k, l = [Idx(n, "occ") for n in range(4)]
virs = a, b, c, d = [Idx(n, "vir") for n in range(4)]
sizes = common.get_sizes(20, 50, 0, spin)

# Printer
printer = common.get_printer(spin)
FunctionPrinter = common.get_function_printer(spin)

# Get prefix and spin transformation function according to setting:
transform_spin, prefix = common.get_transformation_function(spin)

# Declare particle types:
particles = common.particles

# Timer:
timer = common.Stopwatch()

# Hack to get around stupid pybind11 namespace clash rules:
script = """
import json
import wicked as w
from wicked_utils import wicked_indices, compile

w.reset_space()
w.add_space("o", "fermion", "occupied", wicked_indices["o"])
w.add_space("v", "fermion", "unoccupied", wicked_indices["v"])

T = w.op("t1", ["v+ o"]) + w.op("t2", ["v+ v+ o o"]) + w.op("t3", ["v+ v+ v+ o o o"])
H = w.utils.gen_op("f", 1, "ov", "ov") + w.utils.gen_op("v", 2, "ov", "ov")
Hbar = w.bch_series(H, T, 6)

wt = w.WickTheorem()

expr = wt.contract(w.rational(1), Hbar, 0, 6)
mbeq = expr.to_manybody_equation("res")

strings = {key: tuple(compile(e) for e in val) for key, val in mbeq.items()}

with open("tmp.json", "w") as f:
    json.dump(strings, f)
"""
with open("tmp.py", "w") as f:
    f.write(script)
os.system("%s tmp.py" % sys.executable)
with open("tmp.json", "r") as f:
    strings = json.load(f)
os.system("rm tmp.json")
os.system("rm tmp.py")

groups = {
        "f": symmetry(1),
        "v": symmetry(2),
        "t1": symmetry(1),
        "t2": symmetry(2),
        "t3": symmetry(3),
        "t1new": symmetry(1),
        "t2new": symmetry(2),
        "t3new": symmetry(3),
}

with common.FilePrinter("%sCCSDT" % prefix.upper()) as file_printer:
    file_printer.python_file.write("from ebcc.util import symmetrise\n\n")

    # Get energy expression:
    with FunctionPrinter(
            file_printer,
            "energy",
            ["f", "v", "nocc", "nvir", "t1", "t2", "t3"],
            ["e_cc"],
            return_dict=False,
            timer=timer,
    ) as function_printer:
        terms, indices = codegen.wicked_to_sympy(
                strings["|"],
                wicked_indices,
                groups,
                particles,
                return_value="e_cc",
        )
        terms = transform_spin(terms, indices)
        terms = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]
        terms = codegen.spin_integrate._flatten(terms)
        terms = codegen.optimize(terms, sizes=sizes, optimize="exhaust", verify=True, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="energy")

    # Get amplitudes function:
    with FunctionPrinter(
            file_printer,
            "update_amps",
            ["f", "v", "nocc", "nvir", "t1", "t2", "t3"],
            ["t1new", "t2new", "t3new"],
            spin_cases={
                "t1new": ["aa", "bb"],
                "t2new": ["abab", "baba", "aaaa", "bbbb"],
                "t3new": [x+x for x in ("aaa", "aab", "aba", "baa", "abb", "bab", "bba", "bbb")],
            },
            timer=timer,
    ) as function_printer:
        # T1 residuals:
        terms, indices = codegen.wicked_to_sympy(
                strings["o|v"],
                wicked_indices,
                groups,
                particles,
                return_value="t1new",
        )
        terms = antisymmetrise(terms, symmetry(1))
        terms = transform_spin(
                terms,
                indices,
                project_rhf=[(codegen.ALPHA, codegen.ALPHA)],
        )
        terms_t1 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        # T2 residuals:
        terms, indices = codegen.wicked_to_sympy(
                strings["oo|vv"],
                wicked_indices,
                groups,
                particles,
                return_value="t2new",
        )
        terms = antisymmetrise(terms, symmetry(2))
        terms = transform_spin(
                terms,
                indices,
                project_rhf=[(codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.BETA)],
        )
        terms_t2 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        terms = codegen.spin_integrate._flatten([terms_t1, terms_t2])
        #terms = codegen.optimize(terms, sizes=sizes, optimize="exhaust", verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="T1 T2 amplitudes")

        # T3 residuals:
        terms, indices = codegen.wicked_to_sympy(
                strings["ooo|vvv"],
                wicked_indices,
                groups,
                particles,
                return_value="t3new",
        )
        terms = antisymmetrise(terms, symmetry(3))
        terms = transform_spin(
                terms,
                indices,
                project_rhf=[
                    #((ai, aj, ak) * 2)
                    #for ai in (codegen.ALPHA, codegen.BETA)
                    #for aj in (codegen.ALPHA, codegen.BETA)
                    #for ak in (codegen.ALPHA, codegen.BETA)
                    #if not (ai == aj == ak)
                    (codegen.ALPHA, codegen.BETA, codegen.ALPHA, codegen.ALPHA, codegen.BETA, codegen.ALPHA),
                    #(codegen.BETA, codegen.ALPHA, codegen.BETA, codegen.BETA, codegen.ALPHA, codegen.BETA),
                ],
        )
        terms_t3 = [codegen.sympy_to_drudge(group, indices, dr=dr, restricted=spin!="uhf") for group in terms]

        terms = codegen.spin_integrate._flatten([terms_t3])
        #terms = codegen.optimize(terms, sizes=sizes, optimize="greedy", drop_cutoff=2, verify=False, interm_fmt="x{}")
        function_printer.write_python(printer.doprint(terms)+"\n", comment="T3 amplitudes")

        #for n in range(1, 4):
        #    function_printer.write_python(
        #            "    t%dnew = symmetrise(\"%s%s\", t%dnew, \"%s\", apply_factor=False)"
        #            % (n, "i"*n, "a"*n, n, "-"*n*2),
        #    )
        #function_printer.write_python("")
