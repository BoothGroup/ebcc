"""
Generate the DF-QCISD code.
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
        stdout=open(f"{spin[0].upper()}DFQCISD.py", "w"),
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
    pq.add_operator_product(1.0, ["f"])
    pq.add_operator_product(1.0, ["v"])
    pq.add_commutator(1.0, ["f"], ["t2"])
    pq.add_commutator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms = pq.fully_contracted_strings()
    terms = remove_hf_energy(terms)

    # Get the energy in albert format
    expr = import_from_pdaggerq(terms)
    expr = spin_integrate(expr, spin)
    expr = tuple(e.apply(get_density_fit()) for e in expr)
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
    # Get the T1 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e1(i,a)"]])
    pq.add_commutator(1.0, ["f"], ["t1"])
    pq.add_commutator(1.0, ["v"], ["t1"])
    pq.add_commutator(1.0, ["f"], ["t2"])
    pq.add_commutator(1.0, ["v"], ["t2"])
    pq.add_double_commutator(1.0, ["f"], ["t1"], ["t2"])
    pq.add_double_commutator(1.0, ["v"], ["t1"], ["t2"])
    pq.simplify()
    terms_t1 = pq.fully_contracted_strings()

    # Get the T2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e2(i,j,b,a)"]])
    pq.add_operator_product(1.0, ["f"])
    pq.add_operator_product(1.0, ["v"])
    pq.add_commutator(1.0, ["f"], ["t1"])
    pq.add_commutator(1.0, ["v"], ["t1"])
    pq.add_commutator(1.0, ["f"], ["t2"])
    pq.add_commutator(1.0, ["v"], ["t2"])
    pq.add_double_commutator(0.5, ["f"], ["t2"], ["t2"])
    pq.add_double_commutator(0.5, ["v"], ["t2"], ["t2"])
    pq.simplify()
    terms_t2 = pq.fully_contracted_strings()

    # Get the T amplitudes in albert format
    terms = [terms_t1, terms_t2]
    expr = []
    output = []
    returns = []
    for n in range(2):
        for index_spins in get_amplitude_spins(n + 1, spin):
            indices = default_indices["o"][: n + 1] + default_indices["v"][: n + 1]
            expr_n = import_from_pdaggerq(terms[n], index_spins=index_spins)
            expr_n = spin_integrate(expr_n, spin)
            expr_n = tuple(e.apply(get_density_fit()) for e in expr_n)
            output_n = get_t_amplitude_outputs(expr_n, f"t{n+1}new")
            returns_n = (Tensor(*tuple(Index(i, index_spins[i]) for i in indices), name=f"t{n+1}new"),)
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
