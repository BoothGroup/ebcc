"""
Generate the DCD code.
"""

import sys

import pdaggerq
from albert.qc._pdaggerq import import_from_pdaggerq
from albert.qc.index import Index
from albert.tensor import Tensor

from ebcc.codegen.bootstrap_common import *

# Get the spin case
spin = sys.argv[1]

# Set up the code generators
code_generators = {
    "einsum": EinsumCodeGen(
        stdout=open(f"{spin[0].upper()}DCD.py", "w"),
        name_generator=name_generators[spin],
        spin=spin,
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
    terms = pq.fully_contracted_strings()
    terms = remove_hf_energy(terms)

    # Get the energy in albert format
    expr = import_from_pdaggerq(terms)
    expr = spin_integrate(expr, spin)
    output = tuple(Tensor(name="e_cc") for _ in range(len(expr)))
    output, expr = optimise(output, expr, spin, strategy="exhaust")
    returns = (Tensor(name="e_cc"),)

    # Generate the energy code
    for codegen in code_generators.values():
        codegen(
            "energy",
            returns,
            output,
            expr,
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
    expr = []
    output = []
    returns = []
    for index_spins in get_amplitude_spins(2, spin):
        indices = default_indices["o"][:2] + default_indices["v"][:2]
        indices = tuple(Index(i, index_spins[i]) for i in indices)
        expr_n = import_from_pdaggerq(terms, index_spins=index_spins)
        expr_n = spin_integrate(expr_n, spin)
        output_n = get_t_amplitude_outputs(expr_n, f"t2new")
        returns_n = (Tensor(*indices, name=f"t2new"),)
        expr.extend(expr_n)
        output.extend(output_n)
        returns.extend(returns_n)
    output, expr = optimise(output, expr, spin, strategy="exhaust")

    # Generate the T amplitude code
    for name, codegen in code_generators.items():
        if name == "einsum":
            kwargs = {
                "preamble": "t1new = Namespace()\nt2new = Namespace()" if spin == "uhf" else None,
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "update_amps",
            returns,
            output,
            expr,
            **kwargs,
        )

for codegen in code_generators.values():
    codegen.postamble()
    codegen.stdout.close()
