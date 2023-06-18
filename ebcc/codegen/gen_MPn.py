"""Script to generate equations for the MPn model.
"""

import sys
import itertools
from ebcc.codegen import common, hugenholtz
import qccg
from qccg import index, tensor, read, write
import pdaggerq

# Order
order = int(sys.argv[-1])

# Spin integration mode
spin = sys.argv[-2] if sys.argv[-2] in {"rhf", "uhf", "ghf"} else "ghf"

# Printer setup
FunctionPrinter = common.get_function_printer(spin)
timer = common.Stopwatch()
common.PYTHON_HEADER = common.PYTHON_HEADER.replace(
        "from ebcc.util import pack_2e, einsum, Namespace",
        "from ebcc.util import pack_2e, einsum, direct_sum, Namespace",
)

with common.FilePrinter("%sMP%d" % (spin[0].upper(), order)) as file_printer:
    # Get energy expression:
    with FunctionPrinter(
            file_printer,
            "energy",
            ["f", "v", "nocc", "nvir", "t2"],
            ["e_mp"],
            return_dict=False,
            timer=timer,
    ) as function_printer:
        terms = sum([
                [hugenholtz.get_pdaggerq(graph) for graph in hugenholtz.get_hugenholtz_diagrams(n)]
                for n in range(2, order+1)
        ], [])

        qccg.clear()
        qccg.set_spin(spin)

        expression = read.from_pdaggerq(terms, index_spins={})
        expression = expression.expand_spin_orbitals()
        output = tensor.Scalar("e_mp")

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

        expressions, outputs = qccg.optimisation.optimise_expression_gristmill(
                (expression,),
                (output,),
        )
        einsums = write.write_opt_einsums(
                expressions,
                outputs,
                (output,),
                indent=4,
                einsum_function="einsum",
        )
        function_printer.write_python(einsums+"\n", comment="energy")
