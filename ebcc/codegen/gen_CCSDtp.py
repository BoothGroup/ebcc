"""Script to generate equations for the CCSDt' model.

This uses pdaggerq and qccg instead of qwick.
"""

import re
import sys
import itertools
from ebcc.codegen import common
from ebcc.util import generate_spin_combinations
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

with common.FilePrinter("%sCCSDtp" % spin[0].upper()) as file_printer:
    # Get energy expression:
    with FunctionPrinter(
            file_printer,
            "energy",
            ["f", "v", "nocc", "nvir", "t1", "t2", "t3"],
            ["e_cc"],
            return_dict=False,
            timer=timer,
    ) as function_printer:
        pq.clear()
        pq.set_left_operators([["1"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
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

        i = 0
        einsums = list(einsums)
        repl = {}
        while i < len(einsums):
            if einsums[i] == "[":
                j = i+1
                while einsums[j] != "]":
                    j += 1
                repl["".join(einsums[i:j+1])] = "[np.ix_(%s)]" % einsums[i+1:j]
                i = j
            else:
                i += 1
        einsums = "".join(einsums)
        for key, val in repl.items():
            einsums = einsums.replace(key, val)

        function_printer.write_python(einsums+"\n", comment="energy")

    # Get amplitudes function:
    with FunctionPrinter(
            file_printer,
            "update_amps",
            ["f", "v", "space", "t1", "t2", "t3"],
            ["t1new", "t2new", "t3new"],
            spin_cases={
                "t1new": [x+x for x in ("a", "b")],
                "t2new": [x+x for x in ("aa", "ab", "bb")],
                "t3new": [x+x for x in ("aaa", "aba", "bab", "bbb")],
            },
            timer=timer,
    ) as function_printer:
        # T1 residuals:
        pq.clear()
        pq.set_left_operators([["e1(i,a)"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
        pq.simplify()
        terms_t1 = pq.fully_contracted_strings()

        # T2 residuals:
        pq.clear()
        pq.set_left_operators([["e2(i,j,b,a)"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
        pq.simplify()
        terms_t2 = pq.fully_contracted_strings()

        # T3 residuals:
        pq.clear()
        pq.set_left_operators([["e3(i,j,k,c,b,a)"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3"])
        pq.simplify()
        terms_t3 = pq.fully_contracted_strings()

        expressions = []
        outputs = []
        for n, terms in enumerate([terms_t1, terms_t2, terms_t3]):
            if spin == "ghf":
                spins_list = [(None,) * (n+1)]
            elif spin == "rhf":
                spins_list = [(["a", "b"] * (n+1))[:n+1]]
            elif spin == "uhf":
                spins_list = [x[:n+1] for x in generate_spin_combinations(n+1, unique=True)]

            for spins in spins_list:
                qccg.clear()
                qccg.set_spin(spin)

                if spin == "rhf":
                    occ = index.index_factory(index.ExternalIndex, ["i", "j", "k"][:n+1], ["o", "o", "o"][:n+1], ["r", "r", "r"][:n+1])
                    vir = index.index_factory(index.ExternalIndex, ["a", "b", "c"][:n+1], ["v", "v", "v"][:n+1], ["r", "r", "r"][:n+1])
                    output = tensor.FermionicAmplitude("t%dnew" % (n+1), occ, vir)
                    shape = ", ".join(["nocc"] * (n+1) + ["nvir"] * (n+1))
                elif spin == "uhf":
                    occ = index.index_factory(index.ExternalIndex, ["i", "j", "k"][:n+1], ["o", "o", "o"][:n+1], spins)
                    vir = index.index_factory(index.ExternalIndex, ["a", "b", "c"][:n+1], ["v", "v", "v"][:n+1], spins)
                    output = tensor.FermionicAmplitude("t%dnew" % (n+1), occ, vir)
                    shape = ", ".join(["nocc[%d]" % "ab".index(s) for s in spins] + ["nvir[%d]" % "ab".index(s) for s in spins])
                elif spin == "ghf":
                    occ = index.index_factory(index.ExternalIndex, ["i", "j", "k"][:n+1], ["o", "o", "o"][:n+1], [None, None, None][:n+1])
                    vir = index.index_factory(index.ExternalIndex, ["a", "b", "c"][:n+1], ["v", "v", "v"][:n+1], [None, None, None][:n+1])
                    output = tensor.FermionicAmplitude("t%dnew" % (n+1), occ, vir)
                    shape = ", ".join(["nocc"] * (n+1) + ["nvir"] * (n+1))

                index_spins = {index.character: s for index, s in zip(occ+vir, spins+spins)}
                expression = read.from_pdaggerq(terms, index_spins=index_spins)

                expression = expression.expand_spin_orbitals()

                expressions.append(expression)
                outputs.append(output)

        # Make all T3 indices active only
        new_outputs = []
        new_expressions = []
        for j, (output, expression) in enumerate(zip(outputs, expressions)):
            for contraction in expression.contractions:
                # Get all T3 indices
                t3_indices = set()
                if output.symbol.startswith("t3"):
                    t3_indices = t3_indices.union(set(output.indices))
                for tens in contraction.tensors:
                    if tens.symbol.startswith("t3"):
                        t3_indices = t3_indices.union(set(tens.indices))

                # Get the index substitutions
                subs = {
                    i: i.copy(occupancy=i.occupancy.upper())
                    for i in t3_indices
                }

                # Make the substitutions
                new_output = output.substitute_indices(subs)
                new_tensors = []
                for tens in contraction.tensors:
                    new_tensors.append(tens.substitute_indices(subs))
                new_contraction = contraction.__class__([contraction.factor] + new_tensors)

                # Record the new expression
                new_outputs.append(new_output)
                new_expressions.append(expression.__class__([new_contraction], simplify=False))

        outputs = new_outputs
        expressions = new_expressions

        final_outputs = outputs.copy()
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
                add_slices={"t1", "t2", "t1new", "t2new"},
                custom_shapes={
                    "t1new": "(nocc, nvir)",
                    "t2new": "(nocc, nocc, nvir, nvir)",
                    "t3new": "(naocc, naocc, naocc, navir, navir, navir)",
                },
        )

        i = 0
        einsums = list(einsums)
        repl = {}
        while i < len(einsums):
            if einsums[i] == "[":
                j = i+1
                while einsums[j] != "]":
                    j += 1
                part = [x if x == "," else "s"+x for x in einsums[i+1:j]]
                repl["".join(einsums[i:j+1])] = "[np.ix_(%s)]" % "".join(part)
                i = j
            else:
                i += 1
        einsums = "".join(einsums)
        for key, val in repl.items():
            einsums = einsums.replace(key, val)

        function_printer.write_python("    nocc = space.nocc")
        function_printer.write_python("    nvir = space.nvir")
        function_printer.write_python("    naocc = space.naocc")
        function_printer.write_python("    navir = space.navir")
        function_printer.write_python("    so = np.ones((nocc,), dtype=bool)")  # hack to avoid v clash
        function_printer.write_python("    sv = np.ones((nvir,), dtype=bool)")  # hack to avoid v clash
        function_printer.write_python("    sO = space.active[space.occupied]")
        function_printer.write_python("    sV = space.active[space.virtual]")
        function_printer.write_python("")
        function_printer.write_python(einsums+"\n", comment="T amplitudes")
