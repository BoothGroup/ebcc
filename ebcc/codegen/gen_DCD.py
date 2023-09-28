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
        i, j = index.index_factory(index.ExternalIndex, "ij", "oo", [None, None])
        a, b = index.index_factory(index.ExternalIndex, "ab", "vv", [None, None])
        k, l = index.index_factory(index.DummyIndex, "kl", "oo", [None, None])
        c, d = index.index_factory(index.DummyIndex, "cd", "vv", [None, None])

        f = lambda *inds: contraction.Contraction([tensor.Fock(inds)])
        eri = lambda *inds: contraction.Contraction([tensor.ERI(inds)])
        t2 = lambda *inds: contraction.Contraction([tensor.FermionicAmplitude("t2", inds[:2], inds[2:])])

        contractions = []

        ## (ai|bj)
        #contractions.append(eri(a, i, b, j))

        ## (ac|bd) t_{ij}^{cd}
        #contractions.append(eri(a, c, b, d) * t2(i, j, c, d))

        ## (ki|lj) t_{kl}^{ab}
        #contractions.append(eri(k, i, l, j) * t2(k, l, a, b))

        ## (2 t_{ik}^{ac} - t_{ik}^{ca}) (kc|ld) (2 t_{lj}^{db} - t_{lj}^{bd})
        #contractions.append( 4.0 * t2(i, k, a, c) * eri(k, c, l, d) * t2(l, j, d, b))
        #contractions.append(-2.0 * t2(i, k, a, c) * eri(k, c, l, d) * t2(l, j, b, d))
        #contractions.append(-2.0 * t2(i, k, c, a) * eri(k, c, l, d) * t2(l, j, d, b))
        #contractions.append( 1.0 * t2(i, k, c, a) * eri(k, c, l, d) * t2(l, j, b, d))

        #def perm(i, j, a, b):
        #    # [f_{ac} - 1/2 (2 t_{kl}^{ad} - t_{kl}^{da}) (ld|kc)] t_{ij}^{cb}
        #    contractions.append( 1.0 * f(a, c) * t2(i, j, c, b))
        #    contractions.append(-1.0 * t2(k, l, a, d) * eri(l, d, k, c) * t2(i, j, c, b))
        #    contractions.append( 0.5 * t2(k, l, d, a) * eri(l, d, k, c) * t2(i, j, c, b))

        #    # - [f_{ki} + 1/2 (2 t_{il}^{cd} - t_{il}^{dc}) (ld|kc)] t_{kj}^{ab}
        #    contractions.append(-1.0 * f(k, i) * t2(k, j, a, b))
        #    contractions.append(-1.0 * t2(i, l, c, d) * eri(l, d, k, c) * t2(k, j, a, b))
        #    contractions.append( 0.5 * t2(i, l, d, c) * eri(l, d, k, c) * t2(k, j, a, b))

        #    # - (ki|ac) T_{kj}^{cb}
        #    contractions.append(-1.0 * eri(k, i, a, c) * t2(k, j, c, b))

        #    # - (ki|bc) T_{kj}^{ac}
        #    contractions.append(-1.0 * eri(k, i, b, c) * t2(k, j, a, c))

        #    # (2 t_{ik}^{ac} - t_{ik}^{ca}) (kc|bj)
        #    contractions.append( 2.0 * t2(i, k, a, c) * eri(k, c, b, j))
        #    contractions.append(-1.0 * t2(i, k, c, a) * eri(k, c, b, j))

        #perm(i, j, a, b)
        #perm(j, i, b, a)

        # <ij||ab>
        contractions.append(eri(i, j, a, b))

        # 1/2 <ij||kl> t_{kl}^{ab}
        contractions.append(0.5 * eri(i, j, k, l) * t2(k, l, a, b))

        # 1/2 <cd||ab> t_{ij}^{cd}
        contractions.append(0.5 * eri(c, d, a, b) * t2(i, j, c, d))

        # P(ij|ab) <cj||kb> t_{ik}^{ac}
        contractions.append(eri(c, j, k, b) * t2(i, k, a, c))
        contractions.append(eri(c, i, k, a) * t2(j, k, b, c))

        # -1/4 P(ij) <cd||kl> t_{ik}^{dc} t_{lj}^{ab}
        contractions.append(-0.25 * eri(c, d, k, l) * t2(i, k, d, c) * t2(l, j, a, b))
        contractions.append(-0.25 * eri(c, d, k, l) * t2(j, k, d, c) * t2(l, i, a, b))

        # -1/4 P(ab) <cd||kl> t_{lk}^{ac} t_{ij}^{db}
        contractions.append(-0.25 * eri(c, d, k, l) * t2(l, k, a, c) * t2(i, j, d, b))
        contractions.append(-0.25 * eri(c, d, k, l) * t2(l, k, b, c) * t2(i, j, d, a))

        # 1/2 P(ij|ab) <cd||kl> t_{ik}^{ac} t_{jl}^{bd}
        contractions.append(0.5 * eri(c, d, k, l) * t2(i, k, a, c) * t2(j, l, b, d))
        contractions.append(0.5 * eri(c, d, k, l) * t2(j, k, b, c) * t2(i, l, a, d))

        outputs = [tensor.FermionicAmplitude("t2new", (i, j), (a, b))]
        expressions = [contraction.Expression(contractions)]

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
