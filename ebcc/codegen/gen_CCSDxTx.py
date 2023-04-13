"""Script to generate equations for the CCSD(T) model.

This uses pdaggerq and qccg instead of qwick.
"""

import sys
import itertools
from ebcc.codegen import common
from ebcc.util import generate_spin_combinations
import qccg
from qccg import index, tensor, read, write
from qccg.contraction import insert
import pdaggerq

# TODO N^4 memory

# Spin integration mode
spin = sys.argv[-1] if sys.argv[-1] in {"rhf", "uhf", "ghf"} else "ghf"

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
                "t2new": [x+x for x in ("aa", "ab", "bb")],
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
                spins_list = [x[:n+1] for x in generate_spin_combinations(n+1, unique=True)]

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
            spins_list = [x[:3] for x in generate_spin_combinations(3, unique=True)]

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

    ## Get perturbative lambda amplitudes function:
    #with FunctionPrinter(
    #        file_printer,
    #        "update_lams_perturbative",
    #        ["f", "v", "nocc", "nvir", "t1", "t2"],
    #        ["l1new", "l2new"],
    #        spin_cases={
    #            "l1new": [x+x for x in ("a", "b")],
    #            "l2new": [x+x for x in ("aa", "ab", "ba", "bb")],
    #        },
    #        timer=timer,
    #) as function_printer:
    #    # https://aip.scitation.org/doi/pdf/10.1063/1.4994918
    #    if spin == "ghf":
    #        spins_list = [(None,) * (n+1)]
    #    elif spin == "rhf":
    #        spins_list = [(["a", "b"] * (n+1))[:n+1]]
    #    elif spin == "uhf":
    #        spins_list = list(itertools.product("ab", repeat=n+1))

    #    # Connected T3:
    #    expressions = []
    #    outputs = []
    #    for spins in spins_list:
    #        qccg.clear()
    #        qccg.set_spin(spin)

    #        if spin == "rhf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", "rrr")
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", "rrr")
    #            output = tensor.FermionicAmplitude("t3c", occ, vir)
    #            shape = "nocc, nocc, nocc, nvir, nvir, nvir"
    #        elif spin == "uhf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", spins)
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", spins)
    #            output = tensor.FermionicAmplitude("t3c", occ, vir)
    #            shape = "nocc[%d], nocc[%d], nocc[%d], nvir[%d], nvir[%d], nvir[%d]" % (spins + spins)
    #        elif spin == "ghf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", [None, None, None])
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", [None, None, None])
    #            output = tensor.FermionicAmplitude("t3c", occ, vir)
    #            shape = "nocc, nocc, nocc, nvir, nvir, nvir"

    #        index_spins = {index.character: s for index, s in zip(occ+vir, spins+spins)}
    #        # NOTE sign difference in second term in pyscf?
    #        expression = read.from_pdaggerq(["+1.0 t2(a,e,j,k) <b,c||e,i>".split(), "+1.0 t2(b,c,m,i) <j,k||m,a>".split()], index_spins=index_spins)
    #        expression = expression.expand_spin_orbitals()

    #        expressions.append(expression)
    #        outputs.append(output)

    #    # Disconnected T3:
    #    for spins in spins_list:
    #        qccg.clear()
    #        qccg.set_spin(spin)

    #        if spin == "rhf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", "rrr")
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", "rrr")
    #            output = tensor.FermionicAmplitude("t3d", occ, vir)
    #            shape = "nocc, nocc, nocc, nvir, nvir, nvir"
    #        elif spin == "uhf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", spins)
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", spins)
    #            output = tensor.FermionicAmplitude("t3d", occ, vir)
    #            shape = "nocc[%d], nocc[%d], nocc[%d], nvir[%d], nvir[%d], nvir[%d]" % (spins + spins)
    #        elif spin == "ghf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", [None, None, None])
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", [None, None, None])
    #            output = tensor.FermionicAmplitude("t3d", occ, vir)
    #            shape = "nocc, nocc, nocc, nvir, nvir, nvir"

    #        index_spins = {index.character: s for index, s in zip(occ+vir, spins+spins)}
    #        expression = read.from_pdaggerq(["+1.0 t1(a,i) <b,c||j,k>".split(), "-1.0 t2(b,c,j,k) f(i,a)".split()], index_spins=index_spins)
    #        expression = expression.expand_spin_orbitals()

    #        expressions.append(expression)
    #        outputs.append(output)

    #    final_outputs = outputs
    #    expressions, outputs = qccg.optimisation.optimise_expression_gristmill(
    #            expressions,
    #            outputs,
    #    )
    #    einsums = write.write_opt_einsums(
    #            expressions,
    #            outputs,
    #            final_outputs,
    #            indent=4,
    #            einsum_function="einsum",
    #    )
    #    einsums += "\n" + "\n".join([
    #            "    t3c = t3c - t3c.swapaxes(0, 1) - t3c.swapaxes(0, 2)",
    #            "    t3c = t3c - t3c.swapaxes(3, 4) - t3c.swapaxes(3, 5)",
    #            "    t3d = t3d - t3d.swapaxes(0, 1) - t3d.swapaxes(0, 2)",
    #            "    t3d = t3d - t3d.swapaxes(3, 4) - t3d.swapaxes(3, 5)",
    #    ])
    #    function_printer.write_python(einsums+"\n", comment="Connected and disconnected T3")

    #    # FIXME messy
    #    if spin != "uhf":
    #        lines = [
    #                "    e_ia = direct_sum(\"i-a->ia\", np.diag(f.oo), np.diag(f.vv))",
    #                "    denom3 = 1 / direct_sum(\"ia+jb+kc->ijkabc\", e_ia, e_ia, e_ia)",
    #                "    t3c *= denom3",
    #                "    t3d *= denom3",
    #                "    del denom3",
    #        ]
    #    else:
    #        for spins in spins_list:
    #            lines += [
    #                    "    denom3 = 1 / direct_sum(".format(spins="".join(spins)),
    #                    "            \"ia+jb+kc->ijkabc\",",
    #                    "            direct_sum(\"i-a->ia\", np.diag(f.{s}{s}.oo), np.diag(f.{s}{s}.vv)),".format(s=spins[0]),
    #                    "            direct_sum(\"i-a->ia\", np.diag(f.{s}{s}.oo), np.diag(f.{s}{s}.vv)),".format(s=spins[1]),
    #                    "            direct_sum(\"i-a->ia\", np.diag(f.{s}{s}.oo), np.diag(f.{s}{s}.vv)),".format(s=spins[2]),
    #                    "    )",
    #                    "    t3c.{spins}{spins} *= denom3",
    #                    "    t3d.{spins}{spins} *= denom3",
    #                    "    del denom3",
    #            ]
    #    function_printer.write_python("\n".join(lines)+"\n")

    #    # Intermediate:
    #    expressions = []
    #    outputs = []
    #    for spins in spins_list:
    #        qccg.clear()
    #        qccg.set_spin(spin)

    #        if spin == "rhf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", "rrr")
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", "rrr")
    #            output = tensor.FermionicAmplitude("t3x", occ, vir)
    #            shape = "nocc, nocc, nocc, nvir, nvir, nvir"
    #        elif spin == "uhf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", spins)
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", spins)
    #            output = tensor.FermionicAmplitude("t3x", occ, vir)
    #            shape = "nocc[%d], nocc[%d], nocc[%d], nvir[%d], nvir[%d], nvir[%d]" % (spins + spins)
    #        elif spin == "ghf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", [None, None, None])
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", [None, None, None])
    #            output = tensor.FermionicAmplitude("t3x", occ, vir)
    #            shape = "nocc, nocc, nocc, nvir, nvir, nvir"

    #        index_spins = {index.character: s for index, s in zip(occ+vir, spins+spins)}
    #        expression = read.from_pdaggerq(["+2.0 t3c(a,b,c,i,j,k)".split(), "+1.0 t3d(a,b,c,i,j,k)".split()], index_spins=index_spins)
    #        expression = expression.expand_spin_orbitals()

    #        expressions.append(expression)
    #        outputs.append(output)

    #    final_outputs = outputs
    #    expressions, outputs = qccg.optimisation.optimise_expression_gristmill(
    #            expressions,
    #            outputs,
    #    )
    #    einsums = write.write_opt_einsums(
    #            expressions,
    #            outputs,
    #            final_outputs,
    #            indent=4,
    #            einsum_function="einsum",
    #    )
    #    function_printer.write_python(einsums+"\n", comment="Intermediate")

    #    terms_l1 = ["+0.250 t3c(a,b,c,i,j,k) <j,k||b,c>".split()]

    #    terms_l2  = ["+0.50 P(a,b) t3x(a,e,d,i,j,k) <b,k||e,d>".split()]
    #    terms_l2 += ["-0.50 P(i,j) t3x(a,b,c,i,m,n) <m,n||j,c>".split()]
    #    terms_l2 += ["+1.00 f(k,c) t3c(a,b,c,i,j,k)".split()]

    #    expressions = []
    #    outputs = []
    #    for n, terms in enumerate([terms_l1, terms_l2]):
    #        for spins in spins_list:
    #            qccg.clear()
    #            qccg.set_spin(spin)

    #            if spin == "rhf":
    #                occ = index.index_factory(index.ExternalIndex, ["i", "j"][:n+1], ["o", "o"][:n+1], ["r", "r"][:n+1])
    #                vir = index.index_factory(index.ExternalIndex, ["a", "b"][:n+1], ["v", "v"][:n+1], ["r", "r"][:n+1])
    #                output = tensor.FermionicAmplitude("l%dnew" % (n+1), vir, occ)
    #                shape = ", ".join(["nvir"] * (n+1) + ["nocc"] * (n+1))
    #            elif spin == "uhf":
    #                occ = index.index_factory(index.ExternalIndex, ["i", "j"][:n+1], ["o", "o"][:n+1], spins)
    #                vir = index.index_factory(index.ExternalIndex, ["a", "b"][:n+1], ["v", "v"][:n+1], spins)
    #                output = tensor.FermionicAmplitude("l%dnew" % (n+1), vir, occ)
    #                shape = ", ".join(["nvir[%d]" % "ab".index(s) for s in spins] + ["nocc[%d]" % "ab".index(s) for s in spins])
    #            elif spin == "ghf":
    #                occ = index.index_factory(index.ExternalIndex, ["i", "j"][:n+1], ["o", "o"][:n+1], [None, None][:n+1])
    #                vir = index.index_factory(index.ExternalIndex, ["a", "b"][:n+1], ["v", "v"][:n+1], [None, None][:n+1])
    #                output = tensor.FermionicAmplitude("l%dnew" % (n+1), vir, occ)
    #                shape = ", ".join(["nvir"] * (n+1) + ["nocc"] * (n+1))

    #            index_spins = {index.character: s for index, s in zip(vir+occ, spins+spins)}
    #            expression = read.from_pdaggerq(terms, index_spins=index_spins)
    #            expression = expression.expand_spin_orbitals()

    #            expressions.append(expression)
    #            outputs.append(output)

    #    final_outputs = outputs
    #    expressions, outputs = qccg.optimisation.optimise_expression_gristmill(
    #            expressions,
    #            outputs,
    #    )
    #    einsums = write.write_opt_einsums(
    #            expressions,
    #            outputs,
    #            final_outputs,
    #            indent=4,
    #            einsum_function="einsum",
    #    )
    #    function_printer.write_python(einsums+"\n", comment="L amplitudes (perturbative part)")

    ## Get lambda amplitudes function:
    #with FunctionPrinter(
    #        file_printer,
    #        "update_lams",
    #        ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2", "l1pert", "l2pert"],
    #        ["l1new", "l2new"],
    #        spin_cases={
    #            "l1new": [x+x for x in ("a", "b")],
    #            "l2new": [x+x for x in ("aa", "ab", "ba", "bb")],
    #        },
    #        timer=timer,
    #) as function_printer:
    #    if spin == "ghf":
    #        spins_list = [(None,) * (n+1)]
    #    elif spin == "rhf":
    #        spins_list = [(["a", "b"] * (n+1))[:n+1]]
    #    elif spin == "uhf":
    #        spins_list = list(itertools.product("ab", repeat=n+1))

    #    # L1 residuals:
    #    pq.clear()
    #    pq.set_left_operators([["1"]])
    #    pq.set_right_operators([["1"]])
    #    pq.add_st_operator(1.0, ["f", "e1(a,i)"], ["t1", "t2"])
    #    pq.add_st_operator(1.0, ["v", "e1(a,i)"], ["t1", "t2"])
    #    pq.set_left_operators([["l1"], ["l2"]])
    #    pq.add_st_operator( 1.0, ["f", "e1(a,i)"], ["t1", "t2"])
    #    pq.add_st_operator( 1.0, ["v", "e1(a,i)"], ["t1", "t2"])
    #    pq.add_st_operator(-1.0, ["e1(a,i)", "f"], ["t1", "t2"])
    #    pq.add_st_operator(-1.0, ["e1(a,i)", "v"], ["t1", "t2"])
    #    pq.simplify()
    #    terms_l1 = pq.fully_contracted_strings()
    #    terms_l1 += ["+1.0 l1pert(i,a)".split()]

    #    # L2 residuals:
    #    pq.clear()
    #    pq.set_left_operators([["1"]])
    #    pq.set_right_operators([["1"]])
    #    pq.add_st_operator(1.0, ["f", "e2(a,b,j,i)"], ["t1", "t2"])
    #    pq.add_st_operator(1.0, ["v", "e2(a,b,j,i)"], ["t1", "t2"])
    #    pq.set_left_operators([["l1"], ["l2"]])
    #    pq.add_st_operator( 1.0, ["f", "e2(a,b,j,i)"], ["t1", "t2"])
    #    pq.add_st_operator( 1.0, ["v", "e2(a,b,j,i)"], ["t1", "t2"])
    #    pq.add_st_operator(-1.0, ["e2(a,b,j,i)", "f"], ["t1", "t2"])
    #    pq.add_st_operator(-1.0, ["e2(a,b,j,i)", "v"], ["t1", "t2"])
    #    pq.simplify()
    #    terms_l2 = pq.fully_contracted_strings()
    #    terms_l2 += ["+1.0 l2pert(i,j,a,b)".split()]

    #    expressions = []
    #    outputs = []
    #    for n, terms in enumerate([terms_l1, terms_l2]):
    #        for spins in spins_list:
    #            qccg.clear()
    #            qccg.set_spin(spin)

    #            if spin == "rhf":
    #                occ = index.index_factory(index.ExternalIndex, ["i", "j"][:n+1], ["o", "o"][:n+1], ["r", "r"][:n+1])
    #                vir = index.index_factory(index.ExternalIndex, ["a", "b"][:n+1], ["v", "v"][:n+1], ["r", "r"][:n+1])
    #                output = tensor.FermionicAmplitude("l%dnew" % (n+1), vir, occ)
    #                shape = ", ".join(["nvir"] * (n+1) + ["nocc"] * (n+1))
    #            elif spin == "uhf":
    #                occ = index.index_factory(index.ExternalIndex, ["i", "j"][:n+1], ["o", "o"][:n+1], spins)
    #                vir = index.index_factory(index.ExternalIndex, ["a", "b"][:n+1], ["v", "v"][:n+1], spins)
    #                output = tensor.FermionicAmplitude("l%dnew" % (n+1), vir, occ)
    #                shape = ", ".join(["nvir[%d]" % "ab".index(s) for s in spins] + ["nocc[%d]" % "ab".index(s) for s in spins])
    #            elif spin == "ghf":
    #                occ = index.index_factory(index.ExternalIndex, ["i", "j"][:n+1], ["o", "o"][:n+1], [None, None][:n+1])
    #                vir = index.index_factory(index.ExternalIndex, ["a", "b"][:n+1], ["v", "v"][:n+1], [None, None][:n+1])
    #                output = tensor.FermionicAmplitude("l%dnew" % (n+1), vir, occ)
    #                shape = ", ".join(["nvir"] * (n+1) + ["nocc"] * (n+1))

    #            index_spins = {index.character: s for index, s in zip(vir+occ, spins+spins)}
    #            expression = read.from_pdaggerq(terms, index_spins=index_spins)
    #            expression = expression.expand_spin_orbitals()

    #            expressions.append(expression)
    #            outputs.append(output)

    #    final_outputs = outputs
    #    expressions, outputs = qccg.optimisation.optimise_expression_gristmill(
    #            expressions,
    #            outputs,
    #    )
    #    einsums = write.write_opt_einsums(
    #            expressions,
    #            outputs,
    #            final_outputs,
    #            indent=4,
    #            einsum_function="einsum",
    #    )
    #    function_printer.write_python(einsums+"\n", comment="L amplitudes")

    ## Get 1RDM expressions:
    #with FunctionPrinter(
    #        file_printer,
    #        "make_rdm1_f",
    #        ["f", "v", "nocc", "nvir", "t1", "t2", "l1", "l2"],
    #        ["rdm1_f"],
    #        spin_cases={
    #            "rdm1_f": ["aa", "bb"],
    #        },
    #        return_dict=False,
    #        timer=timer,
    #) as function_printer:
    #    if spin != "uhf":
    #        function_printer.write_python(
    #                "    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))\n"
    #        )
    #    else:
    #        function_printer.write_python(
    #                "    delta = Namespace(aa=Namespace(), bb=Namespace())\n"
    #                "    delta.aa = Namespace(oo=np.eye(nocc[0]), vv=np.eye(nvir[0]))\n"
    #                "    delta.bb = Namespace(oo=np.eye(nocc[1]), vv=np.eye(nvir[1]))\n"
    #        )

    #    if spin == "ghf":
    #        spins_list = [(None, None)]
    #    elif spin == "rhf":
    #        spins_list = [("a", "a")]
    #    elif spin == "uhf":
    #        spins_list = [("a", "a"), ("b", "b")]

    #    # Connected T3:
    #    expressions = []
    #    outputs = []
    #    for spins in spins_list:
    #        qccg.clear()
    #        qccg.set_spin(spin)

    #        if spin == "rhf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", "rrr")
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", "rrr")
    #            output = tensor.FermionicAmplitude("t3c", occ, vir)
    #            shape = "nocc, nocc, nocc, nvir, nvir, nvir"
    #        elif spin == "uhf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", spins)
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", spins)
    #            output = tensor.FermionicAmplitude("t3c", occ, vir)
    #            shape = "nocc[%d], nocc[%d], nocc[%d], nvir[%d], nvir[%d], nvir[%d]" % (spins + spins)
    #        elif spin == "ghf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", [None, None, None])
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", [None, None, None])
    #            output = tensor.FermionicAmplitude("t3c", occ, vir)
    #            shape = "nocc, nocc, nocc, nvir, nvir, nvir"

    #        index_spins = {index.character: s for index, s in zip(occ+vir, spins+spins)}
    #        # NOTE sign difference in second term in pyscf?
    #        expression = read.from_pdaggerq(["+1.0 t2(a,e,j,k) <b,c||e,i>".split(), "+1.0 t2(b,c,m,i) <j,k||m,a>".split()], index_spins=index_spins)
    #        expression = expression.expand_spin_orbitals()

    #        expressions.append(expression)
    #        outputs.append(output)

    #    # Disconnected T3:
    #    for spins in spins_list:
    #        qccg.clear()
    #        qccg.set_spin(spin)

    #        if spin == "rhf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", "rrr")
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", "rrr")
    #            output = tensor.FermionicAmplitude("t3d", occ, vir)
    #            shape = "nocc, nocc, nocc, nvir, nvir, nvir"
    #        elif spin == "uhf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", spins)
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", spins)
    #            output = tensor.FermionicAmplitude("t3d", occ, vir)
    #            shape = "nocc[%d], nocc[%d], nocc[%d], nvir[%d], nvir[%d], nvir[%d]" % (spins + spins)
    #        elif spin == "ghf":
    #            occ = index.index_factory(index.ExternalIndex, "ijk", "ooo", [None, None, None])
    #            vir = index.index_factory(index.ExternalIndex, "abc", "vvv", [None, None, None])
    #            output = tensor.FermionicAmplitude("t3d", occ, vir)
    #            shape = "nocc, nocc, nocc, nvir, nvir, nvir"

    #        index_spins = {index.character: s for index, s in zip(occ+vir, spins+spins)}
    #        expression = read.from_pdaggerq(["+1.0 t1(a,i) <b,c||j,k>".split(), "-1.0 t2(b,c,j,k) f(i,a)".split()], index_spins=index_spins)
    #        expression = expression.expand_spin_orbitals()

    #        expressions.append(expression)
    #        outputs.append(output)

    #    final_outputs = outputs
    #    expressions, outputs = qccg.optimisation.optimise_expression_gristmill(
    #            expressions,
    #            outputs,
    #    )
    #    einsums = write.write_opt_einsums(
    #            expressions,
    #            outputs,
    #            final_outputs,
    #            indent=4,
    #            einsum_function="einsum",
    #    )
    #    einsums += "\n" + "\n".join([
    #            "    t3c = t3c - t3c.swapaxes(0, 1) - t3c.swapaxes(0, 2)",
    #            "    t3c = t3c - t3c.swapaxes(3, 4) - t3c.swapaxes(3, 5)",
    #            "    t3d = t3d - t3d.swapaxes(0, 1) - t3d.swapaxes(0, 2)",
    #            "    t3d = t3d - t3d.swapaxes(3, 4) - t3d.swapaxes(3, 5)",
    #    ])
    #    function_printer.write_python(einsums+"\n", comment="Connected and disconnected T3")

    #    # FIXME messy
    #    if spin != "uhf":
    #        lines = [
    #                "    e_ia = direct_sum(\"i-a->ia\", np.diag(f.oo), np.diag(f.vv))",
    #                "    denom3 = 1 / direct_sum(\"ia+jb+kc->ijkabc\", e_ia, e_ia, e_ia)",
    #                "    t3c *= denom3",
    #                "    t3d *= denom3",
    #                "    del denom3",
    #        ]
    #    else:
    #        for spins in spins_list:
    #            lines += [
    #                    "    denom3 = 1 / direct_sum(".format(spins="".join(spins)),
    #                    "            \"ia+jb+kc->ijkabc\",",
    #                    "            direct_sum(\"i-a->ia\", np.diag(f.{s}{s}.oo), np.diag(f.{s}{s}.vv)),".format(s=spins[0]),
    #                    "            direct_sum(\"i-a->ia\", np.diag(f.{s}{s}.oo), np.diag(f.{s}{s}.vv)),".format(s=spins[1]),
    #                    "            direct_sum(\"i-a->ia\", np.diag(f.{s}{s}.oo), np.diag(f.{s}{s}.vv)),".format(s=spins[2]),
    #                    "    )",
    #                    "    t3c.{spins}{spins} *= denom3",
    #                    "    t3d.{spins}{spins} *= denom3",
    #                    "    del denom3",
    #            ]
    #    function_printer.write_python("\n".join(lines)+"\n")

    #    expressions = {} 
    #    for sectors, indices in [("oo", "ij"), ("ov", "ia"), ("vo", "ai"), ("vv", "ab")]:
    #        pq.clear()
    #        pq.set_left_operators([["1"], ["l1"], ["l2"]])
    #        pq.add_st_operator(1.0, ["e1(%s,%s)" % tuple(indices)], ["t1", "t2"])
    #        pq.simplify()
    #        terms = pq.fully_contracted_strings()

    #        for spins in spins_list:
    #            qccg.clear()
    #            qccg.set_spin(spin)

    #            if spin == "rhf":
    #                inds = index.index_factory(index.ExternalIndex, indices, sectors, "rr")
    #                output = tensor.RDM1(inds)
    #                shape = ", ".join(["nocc" if o == "o" else "nvir" for o in sectors])
    #            elif spin == "uhf":
    #                inds = index.index_factory(index.ExternalIndex, indices, sectors, spins)
    #                output = tensor.RDM1(inds)
    #                shape = ", ".join(["nocc[%d]" % "ab".index(s) if o == "o" else "nvir[%d]" % "ab".index(s) for o, s in zip(sectors, spins)])
    #            elif spin == "ghf":
    #                inds = index.index_factory(index.ExternalIndex, indices, sectors, [None]*2)
    #                output = tensor.RDM1(inds)
    #                shape = ", ".join(["nocc" if o == "o" else "nvir" for o in sectors])

    #            index_spins = {index.character: s for index, s in zip(inds, spins+spins)}
    #            expression = read.from_pdaggerq(terms, index_spins=index_spins)
    #            expression = expression.expand_spin_orbitals()

    #            if spin == "rhf":
    #                expression = expression * 2

    #            expressions[output] = expression

    #    for sectors, indices in [("oo", "ij"), ("ov", "ia"), ("vv", "ab")]:
    #        for spins in spins_list:
    #            qccg.clear()
    #            qccg.set_spin(spin)

    #            if spin == "rhf":
    #                inds = index.index_factory(index.ExternalIndex, indices, sectors, "rr")
    #                output = tensor.RDM1(inds)
    #                shape = ", ".join(["nocc" if o == "o" else "nvir" for o in sectors])
    #            elif spin == "uhf":
    #                inds = index.index_factory(index.ExternalIndex, indices, sectors, spins)
    #                output = tensor.RDM1(inds)
    #                shape = ", ".join(["nocc[%d]" % "ab".index(s) if o == "o" else "nvir[%d]" % "ab".index(s) for o, s in zip(sectors, spins)])
    #            elif spin == "ghf":
    #                inds = index.index_factory(index.ExternalIndex, indices, sectors, [None]*2)
    #                output = tensor.RDM1(inds)
    #                shape = ", ".join(["nocc" if o == "o" else "nvir" for o in sectors])

    #            if sectors == "vv":
    #                terms = [
    #                        "+0.08333333333333333 t3c(a,d,c,i,j,k) t3c(b,d,c,i,j,k)".split(),
    #                        "+0.08333333333333333 t3c(a,d,c,i,j,k) t3d(b,d,c,i,j,k)".split(),
    #                ]
    #            elif sectors == "oo":
    #                terms = [
    #                        "+0.08333333333333333 t3c(a,b,c,i,l,k) t3c(a,b,c,j,l,k)".split(),
    #                        "+0.08333333333333333 t3c(a,b,c,i,l,k) t3d(a,b,c,j,l,k)".split(),
    #                ]
    #            elif sectors == "ov":
    #                terms = [
    #                        "+0.25 t3c(a,b,c,i,j,k) t2(b,c,j,k)".split(),
    #                ]

    #            index_spins = {index.character: s for index, s in zip(inds, spins+spins)}
    #            expression = read.from_pdaggerq(terms, index_spins=index_spins)

    #            if sectors in ("oo", "vv"):
    #                expression *= tensor.Delta(inds)

    #            expression = expression.expand_spin_orbitals()

    #            if spin == "rhf":
    #                expression = expression * 2

    #            expressions[output] += expression

    #    outputs, expressions = zip(*expressions.items())

    #    final_outputs = outputs
    #    expressions, outputs = qccg.optimisation.optimise_expression_gristmill(
    #            expressions,
    #            outputs,
    #    )
    #    einsums = write.write_opt_einsums(
    #            expressions,
    #            outputs,
    #            final_outputs,
    #            indent=4,
    #            einsum_function="einsum",
    #            add_occupancies={"f", "v", "rdm1_f", "rdm2_f", "delta"},
    #    )
    #    function_printer.write_python(einsums+"\n", comment="RDM1")

    #    if spin != "uhf":
    #        function_printer.write_python("    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])\n")
    #    else:
    #        function_printer.write_python(
    #            "    rdm1_f_aa = np.block([[rdm1_f_aa_oo, rdm1_f_aa_ov], [rdm1_f_aa_vo, rdm1_f_aa_vv]])\n"
    #            "    rdm1_f_bb = np.block([[rdm1_f_bb_oo, rdm1_f_bb_ov], [rdm1_f_bb_vo, rdm1_f_bb_vv]])\n"
    #        )
