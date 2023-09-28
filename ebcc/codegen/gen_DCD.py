"""Script to generate equations for the DCD model.
"""

import sys
import itertools
from ebcc.codegen import common
import qccg
from qccg import index, tensor, contraction, read, write
import pdaggerq

# Spin integration mode
spin = sys.argv[-1] if sys.argv[-1] in {"rhf", "uhf", "ghf"} else "ghf"

# pdaggerq setup
pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

# Printer setup
FunctionPrinter = common.get_function_printer(spin)
timer = common.Stopwatch()

with common.FilePrinter("%sDCD" % spin[0].upper()) as file_printer:
    # Get energy expression:
    with FunctionPrinter(
            file_printer,
            "energy",
            ["f", "v", "nocc", "nvir", "t2"],
            ["e_cc"],
            return_dict=False,
            timer=timer,
    ) as function_printer:
        pq.clear()
        pq.set_left_operators([["1"]])
        pq.add_st_operator(1.0, ["f"], ["t2"])
        pq.add_st_operator(1.0, ["v"], ["t2"])
        pq.simplify()

        terms = pq.fully_contracted_strings()
        terms = [term for term in terms if set(term) != {'+1.00000000000000', 'f(i,i)'}]
        terms = [term for term in terms if set(term) != {'-0.50000000000000', '<j,i||j,i>'}]

        qccg.clear()
        qccg.set_spin(spin)

        expression = read.from_pdaggerq(terms, index_spins={})
        expression = expression.expand_spin_orbitals()
        output = tensor.Scalar("e_cc")

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

    # Get amplitudes function:
    with FunctionPrinter(
            file_printer,
            "update_amps",
            ["f", "v", "nocc", "nvir", "t2"],
            ["t2new"],
            spin_cases={
                "t2new": [x+x for x in ("aa", "ab", "bb")],
            },
            timer=timer,
    ) as function_printer:
        # T2 residuals:
        terms = [
            ["-1.00", "P(i,j)", "f(k,j)", "t2(a,b,i,k)"],
            ["+1.00", "P(a,b)", "f(a,c)", "t2(c,b,i,j)"],
            ["+1.00", "<i,j||a,b>"],
            ["+0.50", "<i,j||k,l>", "t2(a,b,k,l)"],
            ["+0.50", "<c,d||a,b>", "t2(c,d,i,j)"],
            ["+1.00", "P(i,j)", "P(a,b)", "<c,j||k,b>", "t2(a,c,i,k)"],
            ["-0.50", "P(i,j)", "<c,d||k,l>", "t2(d,c,i,k)", "t2(a,b,l,j)"],
            ["+0.25", "<c,d||k,l>", "t2(c,d,i,j)", "t2(a,b,k,l)"],
            ["-0.50", "P(a,b)", "<c,d||k,l>", "t2(a,c,l,k)", "t2(d,b,i,j)"],
            ["+0.50", "P(i,j)", "P(a,b)", "<c,d||k,l>", "t2(a,c,i,k)", "t2(b,d,j,l)"],
        ]

        if spin == "ghf":
            spins_list = [(None, None, None, None)]
        elif spin == "rhf":
            spins_list = [("a", "b", "a", "b")]
        elif spin == "uhf":
            spins_list = [list(y) for y in sorted(set("".join(sorted(x)) for x in itertools.product("ab", repeat=2)))]

        expressions = []
        outputs = []
        for spins in spins_list:
            qccg.clear()
            qccg.set_spin(spin)

            if spin == "rhf":
                occ = index.index_factory(index.ExternalIndex, "ij", "oo", "rr")
                vir = index.index_factory(index.ExternalIndex, "ab", "vv", "rr")
                output = tensor.FermionicAmplitude("t2new", occ, vir)
                shape = "nocc, nocc, nvir, nvir"
            elif spin == "uhf":
                occ = index.index_factory(index.ExternalIndex, "ij", "oo", spins)
                vir = index.index_factory(index.ExternalIndex, "ab", "vv", spins)
                output = tensor.FermionicAmplitude("t2new", occ, vir)
                shape = "nocc[%d], nocc[%d], nvir[%d], nvir[%d]" % tuple("ab".index(s) for s in (spins+spins))
            elif spin == "ghf":
                occ = index.index_factory(index.ExternalIndex, "ij", "oo", [None, None])
                vir = index.index_factory(index.ExternalIndex, "ab", "vv", [None, None])
                output = tensor.FermionicAmplitude("t2new", occ, vir)
                shape = "nocc, nocc, nvir, nvir"

            index_spins = {index.character: s for index, s in zip(occ+vir, spins+spins)}
            expression = read.from_pdaggerq(terms, index_spins=index_spins)

            expression = expression.expand_spin_orbitals()

            expressions.append(expression)
            outputs.append(output)

        final_outputs = outputs
        # Dummies change, canonicalise_dummies messes up the indices FIXME
        expressions, outputs = qccg.optimisation.optimise_expression_gristmill(
                expressions,
                outputs,
                strat="exhaust",
        )
        einsums = write.write_opt_einsums(
                expressions,
                outputs,
                final_outputs,
                indent=4,
                einsum_function="einsum",
        )
        function_printer.write_python(einsums+"\n", comment="T amplitudes")
