"""Script to generate equations for the QCISD model.
"""

import sys
import itertools
from ebcc.codegen import common
import qccg
from qccg import index, tensor, read, write
import pdaggerq

# Spin integration mode
spin = sys.argv[-1] if sys.argv[-1] in {"rhf", "uhf", "ghf"} else "ghf"

# pdaggerq setup
pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

# Printer setup
FunctionPrinter = common.get_function_printer(spin)
timer = common.Stopwatch()

with common.FilePrinter("%sQCISD" % spin[0].upper()) as file_printer:
    # Get energy expression:
    with FunctionPrinter(
            file_printer,
            "energy",
            ["f", "v", "nocc", "nvir", "t1", "t2"],
            ["e_cc"],
            return_dict=False,
            timer=timer,
    ) as function_printer:
        pq.clear()
        pq.set_left_operators([["1"]])
        for h in ("f", "v"):
            pq.add_operator_product(1.0, [h])
            pq.add_commutator(1.0, [h], ["t2"])
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
            ["f", "v", "nocc", "nvir", "t1", "t2"],
            ["t1new", "t2new"],
            spin_cases={
                "t1new": [x+x for x in ("a", "b")],
                "t2new": [x+x for x in ("aa", "ab", "bb")],
            },
            timer=timer,
    ) as function_printer:
        # T1 residuals:
        pq.clear()
        pq.set_left_operators([["e1(i,a)"]])
        for h in ("f", "v"):
            pq.add_commutator(1.0, [h], ["t1"])
            pq.add_commutator(1.0, [h], ["t2"])
            pq.add_double_commutator(1.0, [h], ["t1"], ["t2"])
        pq.simplify()
        terms_t1 = pq.fully_contracted_strings()

        # T2 residuals:
        pq.clear()
        pq.set_left_operators([["e2(i,j,b,a)"]])
        for h in ("f", "v"):
            pq.add_operator_product(1.0, [h])
            pq.add_commutator(1.0, [h], ["t1"])
            pq.add_commutator(1.0, [h], ["t2"])
            pq.add_double_commutator(0.5, [h], ["t2"], ["t2"])
        pq.simplify()
        terms_t2 = pq.fully_contracted_strings()

        expressions = []
        outputs = []
        for n, terms in enumerate([terms_t1, terms_t2]):
            if spin == "ghf":
                spins_list = [(None,) * (n+1)]
            elif spin == "rhf":
                spins_list = [(["a", "b"] * (n+1))[:n+1]]
            elif spin == "uhf":
                spins_list = [list(y) for y in sorted(set("".join(sorted(x)) for x in itertools.product("ab", repeat=n+1)))]

            for spins in spins_list:
                qccg.clear()
                qccg.set_spin(spin)

                if spin == "rhf":
                    occ = index.index_factory(index.ExternalIndex, ["i", "j"][:n+1], ["o", "o"][:n+1], ["r", "r"][:n+1])
                    vir = index.index_factory(index.ExternalIndex, ["a", "b"][:n+1], ["v", "v"][:n+1], ["r", "r"][:n+1])
                    output = tensor.FermionicAmplitude("t%dnew" % (n+1), occ, vir)
                    shape = ", ".join(["nocc"] * (n+1) + ["nvir"] * (n+1))
                elif spin == "uhf":
                    occ = index.index_factory(index.ExternalIndex, ["i", "j"][:n+1], ["o", "o"][:n+1], spins)
                    vir = index.index_factory(index.ExternalIndex, ["a", "b"][:n+1], ["v", "v"][:n+1], spins)
                    output = tensor.FermionicAmplitude("t%dnew" % (n+1), occ, vir)
                    shape = ", ".join(["nocc[%d]" % "ab".index(s) for s in spins] + ["nvir[%d]" % "ab".index(s) for s in spins])
                elif spin == "ghf":
                    occ = index.index_factory(index.ExternalIndex, ["i", "j"][:n+1], ["o", "o"][:n+1], [None, None][:n+1])
                    vir = index.index_factory(index.ExternalIndex, ["a", "b"][:n+1], ["v", "v"][:n+1], [None, None][:n+1])
                    output = tensor.FermionicAmplitude("t%dnew" % (n+1), occ, vir)
                    shape = ", ".join(["nocc"] * (n+1) + ["nvir"] * (n+1))

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
