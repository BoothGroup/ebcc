"""Script to generate equations for the CCSD(T) model.

This uses pdaggerq and qccg instead of qwick.
"""

import itertools
from ebcc.codegen import common
import qccg
from qccg import index, tensor, read, write
from qccg.contraction import insert
import pdaggerq

# TODO N^4 memory

# Spin integration mode
spin = "uhf"

# pdaggerq setup
pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

# Printer setup
FunctionPrinter = common.get_function_printer(spin)
timer = common.Stopwatch()
common.PYTHON_HEADER = common.PYTHON_HEADER.replace(
        "from ebcc.util import pack_2e, einsum, Namespace",
        "from ebcc.util import pack_2e, einsum, direct_sum, Namespace",
)

with common.FilePrinter("%sCCSDxTx" % spin[0].upper()) as file_printer:
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
        pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
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
                "t2new": [x+x for x in ("aa", "ab", "ba", "bb")],
            },
            timer=timer,
    ) as function_printer:
        # T1 residuals:
        pq.clear()
        pq.set_left_operators([["e1(i,a)"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
        pq.simplify()
        terms_t1 = pq.fully_contracted_strings()

        # T2 residuals:
        pq.clear()
        pq.set_left_operators([["e2(i,j,b,a)"]])
        pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
        pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
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
                spins_list = list(itertools.product("ab", repeat=n+1))

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

    # Get perturbative energy expression:
    with FunctionPrinter(
            file_printer,
            "energy_perturbative",
            ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
            ["e_pert"],
            return_dict=False,
            timer=timer,
    ) as function_printer:
        pq.clear()
        pq.set_left_operators([["e3(i,j,k,c,b,a)"]])
        pq.add_commutator(1.0, ["v"], ["t2"])
        pq.simplify()
        terms = pq.fully_contracted_strings()

        if spin == "ghf":
            spins_list = [(None, None, None)]
        elif spin == "rhf":
            spins_list = [("a", "b", "a")]
        elif spin == "uhf":
            spins_list = list(itertools.product("ab", repeat=3))

        expressions = []
        outputs = []
        for spins in spins_list:
            qccg.clear()
            qccg.set_spin(spin)

            if spin == "rhf":
                occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", "rrr")
                vir = index.index_factory(index.ExternalIndex, "abc", "vvv", "rrr")
                output = tensor.FermionicAmplitude("t3", occ, vir)
                shape = ", ".join(["nocc"] * 3 + ["nvir"] * 3)
            elif spin == "uhf":
                occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", spins)
                vir = index.index_factory(index.ExternalIndex, "abc", "vvv", spins)
                output = tensor.FermionicAmplitude("t3", occ, vir)
                shape = ", ".join(["nocc[%d]" % "ab".index(s) for s in spins] + ["nvir[%d]" % "ab".index(s) for s in spins])
            elif spin == "ghf":
                occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", [None, None, None])
                vir = index.index_factory(index.ExternalIndex, "abc", "vvv", [None, None, None])
                output = tensor.FermionicAmplitude("t3", occ, vir)
                shape = ", ".join(["nocc"] * 3 + ["nvir"] * 3)

            index_spins = {index.character: spin for index, spin in zip(occ+vir, spins+spins)}
            expression = read.from_pdaggerq(terms, index_spins=index_spins)
            expression = expression.expand_spin_orbitals()

            expressions.append(expression)
            outputs.append(output)

        expressions_t3, outputs_t3 = expressions, outputs

        #final_outputs = outputs
        #if spin == "rhf":
        #    expressions, outputs = qccg.optimisation.optimise_expression(
        #            expressions,
        #            outputs,
        #    )
        #einsums = write.write_opt_einsums(expressions, outputs, final_outputs, indent=4, einsum_function="einsum")
        #function_printer.write_python(einsums, comment="T3 amplitude")

        # FIXME messy
        if spin != "uhf":
            lines = [
                    "    e_ia = direct_sum(\"i-a->ia\", np.diag(f.oo), np.diag(f.vv))",
                    "    denom3 = 1 / direct_sum(\"ia+jb+kc->ijkabc\", e_ia, e_ia, e_ia)",
            ]
        else:
            lines = [
                    "    denom3 = Namespace()",
            ]
            for spins in spins_list:
                lines += [
                        "    denom3.{spins}{spins} = 1 / direct_sum(".format(spins="".join(spins)),
                        "            \"ia+jb+kc->ijkabc\",",
                        "            direct_sum(\"i-a->ia\", np.diag(f.{s}{s}.oo), np.diag(f.{s}{s}.vv)),".format(s=spins[0]),
                        "            direct_sum(\"i-a->ia\", np.diag(f.{s}{s}.oo), np.diag(f.{s}{s}.vv)),".format(s=spins[1]),
                        "            direct_sum(\"i-a->ia\", np.diag(f.{s}{s}.oo), np.diag(f.{s}{s}.vv)),".format(s=spins[2]),
                        "    )",
                ]
        function_printer.write_python("\n".join(lines)+"\n")

        pq.clear()
        pq.set_left_operators([["l1"], ["l2"]])
        pq.add_commutator(1.0, ["v"], ["t3"])
        pq.simplify()
        terms = pq.fully_contracted_strings()

        output = tensor.Scalar("e_pert")

        expression = read.from_pdaggerq(terms)
        expression = expression.expand_spin_orbitals()

        contractions = []
        for c in expression.contractions:
            tensors = [c.factor]
            for t in c.tensors:
                tensors.append(t)
                if t.symbol == "t3":
                    class TempTensor(tensor.ATensor):
                        @property
                        def perms(self):
                            for p1 in itertools.permutations([0,1,2]):
                                for p2 in itertools.permutations([3,4,5]):
                                    perm = p1 + p2
                                    if all(self.indices[i].spin == self.indices[p].spin for i, p in enumerate(perm)):
                                        yield (perm, 1)
                    denom = TempTensor("denom3", t.indices)
                    tensors.append(denom)
            contractions.append(c.__class__(tensors))
        expression = expression.__class__(contractions)

        expression = insert(expression, expressions_t3, outputs_t3)

        # Dummies change, canonicalise_dummies messes up the indices FIXME
        if spin == "uhf":
            expressions = (expression,)
            outputs = (output,)
        else:
            expressions, outputs = qccg.optimisation.optimise_expression_gristmill(
                    (expression,),
                    (output,),
                    strat="exhaust" if spin == "ghf" else "greedy",  # FIXME
            )
        einsums = write.write_opt_einsums(
                expressions,
                outputs,
                (output,),
                indent=4,
                einsum_function="einsum",
        )
        # FIXME messy
        if spin == "uhf":
            einsums = einsums.replace("t3.", "t3_")
            einsums += "\n    e_pert /= 2"  # FIXME where did I lose this?
        function_printer.write_python(einsums+"\n", comment="energy")
