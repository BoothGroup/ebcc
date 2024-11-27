"""
Generate the DCD code.
"""

import sys

import pdaggerq
from albert.qc._pdaggerq import remove_reference_energy, remove_reference_energy_eom
from albert.qc.spin import ghf_to_uhf, ghf_to_rhf
from albert.qc import ghf, uhf, rhf
from albert.tensor import Tensor
from albert.index import Index
from albert.code._ebcc import EBCCCodeGenerator
from albert.misc import Stopwatch
from albert.opt.tools import _tensor_info

from ebcc.codegen.bootstrap_common import get_energy, get_amplitudes, get_rdm1, get_rdm2, get_eom

# Get the spin case
spin = sys.argv[1]

# Set up the code generators
code_generators = {
    "einsum": EBCCCodeGenerator(
        stdout=open(f"{spin[0].upper()}DCD.py", "w"),
    ),
}

# Write the preamble
for codegen in code_generators.values():
    codegen.preamble()

# Set up pdaggerq
pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

with Stopwatch("Energy"):
    # Get the energy contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["1"]])
    pq.add_st_operator(1.0, ["f"], ["t2"])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms = pq.strings()
    terms = remove_reference_energy(terms)

    # Get the energy in albert format
    output_expr, returns = get_energy(terms, spin)

    # Generate the energy code
    for codegen in code_generators.values():
        codegen(
            "energy",
            returns,
            output_expr,
        )

with Stopwatch("T amplitudes"):
    # Get the T2 contractions in pdaggerq format
    # 10.1063/1.4944087
    terms = [
        ["-1.00", "P(i,j)", "f(k,j)", "t2(a,b,i,k)"],
        ["+1.00", "P(a,b)", "f(a,c)", "t2(c,b,i,j)"],
        ["+1.00", "<i,j||a,b>"],
        ["+0.50", "<i,j||k,l>", "t2(a,b,k,l)"],
        ["+0.50", "<c,d||a,b>", "t2(c,d,i,j)"],
        ["+1.00", "P(i,j)", "P(a,b)", "<c,j||k,b>", "t2(a,c,i,k)"],
        ["-0.25", "P(i,j)", "<c,d||k,l>", "t2(d,c,i,k)", "t2(a,b,l,j)"],
        ["-0.25", "P(a,b)", "<c,d||k,l>", "t2(a,c,l,k)", "t2(d,b,i,j)"],
        ["+0.50", "P(i,j)", "P(a,b)", "<c,d|k,l>", "t2(a,c,i,k)", "t2(b,d,j,l)"],
    ]

    # Get the T amplitudes in albert format
    output_expr, returns = get_amplitudes([terms], spin, orders=[2])

    # Generate the T amplitude code
    for name, codegen in code_generators.items():
        codegen(
            "update_amps",
            returns,
            output_expr,
            as_dict=True,
        )

for codegen in code_generators.values():
    codegen.postamble()
    codegen.stdout.close()
