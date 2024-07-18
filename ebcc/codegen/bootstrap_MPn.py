"""
Generate the MPn code.
"""

import sys

import pdaggerq
from albert.qc._pdaggerq import import_from_pdaggerq
from albert.tensor import Tensor

from ebcc.codegen import hugenholtz
from ebcc.codegen.bootstrap_common import *

# Get the spin case
spin = sys.argv[1]
order = int(sys.argv[2])

# Set up the code generators
code_generators = {
    "einsum": EinsumCodeGen(
        stdout=open(f"{spin[0].upper()}MP{order}.py", "w"),
        name_generator=name_generators[spin],
        spin=spin,
    ),
}

# Write the preamble
for codegen in code_generators.values():
    codegen.preamble()

with Stopwatch("Energy"):
    # Get the energy contractions in pdaggerq format
    terms = sum([
        [hugenholtz.get_pdaggerq(graph) for graph in hugenholtz.get_hugenholtz_diagrams(n)]
        for n in range(2, order+1)
    ], [])

    # Get the energy in albert format
    expr = import_from_pdaggerq(terms)
    expr = spin_integrate(expr, spin)
    output = tuple(Tensor(name="e_mp") for _ in range(len(expr)))
    output, expr = optimise(output, expr, spin, strategy="exhaust")
    returns = (Tensor(name="e_mp"),)

    # Get the denominators for higher orders
    if order > 3:
        if spin != "uhf":
            lines = []
            for n in ([2] if order in (2, 3) else list(range(1, order+1))):
                subscripts = ["ia", "jb", "kc", "ld", "me", "nf"][:n]
                lines += [
                        "    denom%d = 1 / direct_sum(" % n,
                        "           \"%s->%s%s\"," % ("+".join(subscripts), "".join([s[0] for s in subscripts]), "".join([s[1] for s in subscripts])),
                ]
                for subscript in subscripts:
                    lines.append(
                        "           direct_sum(\"%s->%s\", np.diag(f.oo), np.diag(f.vv)),"
                        % ("-".join(list(subscript)), subscript)
                    )
                lines.append("    )")
        else:
            lines = [
                    "    denom3 = Namespace()",
            ]
            for n in ([2] if order in (2, 3) else list(range(1, order+1))):
                spins_list = [list(y) for y in sorted(set("".join(sorted(x)) for x in itertools.product("ab", repeat=n)))]
                for spins in spins_list:
                    subscripts = ["ia", "jb", "kc", "ld", "me", "nf"][:n]
                    lines += [
                            "    denom%d.%s%s = 1 / direct_sum(" % (n, "".join(spins), "".join(spins)),
                            "           \"%s->%s%s\"," % ("+".join(subscripts), "".join([s[0] for s in subscripts]), "".join([s[1] for s in subscripts])),
                    ]
                    for subscript, spin in zip(subscripts, spins):
                        lines.append(
                            "           direct_sum(\"%s->%s\", np.diag(f.%s%s.oo), np.diag(f.%s%s.vv)),"
                            % ("-".join(list(subscript)), subscript, spin, spin, spin, spin)
                        )
                    lines.append("    )")
        function_printer.write_python("\n".join(lines)+"\n")

    # Generate the energy code
    for codegen in code_generators.values():
        codegen(
            "energy",
            returns,
            output,
            expr,
        )
