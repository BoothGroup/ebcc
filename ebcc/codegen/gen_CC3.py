"""Script to generate equations for the CC3 model.

This uses pdaggerq and qccg instead of qwick.
"""

import itertools
from ebcc.codegen import common
import qccg
from qccg import index, tensor, read, write
import pdaggerq

# Spin integration mode
spin = "rhf"

# pdaggerq setup
pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

# Printer setup
FunctionPrinter = common.get_function_printer(spin)
timer = common.Stopwatch()

with common.FilePrinter("%sCC3" % spin[0].upper()) as file_printer:
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

        expression = read.from_pdaggerq(terms, {})
        expression = expression.expand_spin_orbitals()
        output = tensor.Scalar("e_cc")

        einsums = write.write_einsum(expression, output, indent=4)
        einsums = "    e_cc = 0.0\n" + einsums
        function_printer.write_python(einsums+"\n", comment="energy")

    # Get amplitudes function:
    with FunctionPrinter(
            file_printer,
            "update_amps",
            ["f", "v", "nocc", "nvir", "t1", "t2", "t3"],
            ["t1new", "t2new", "t3new"],
            spin_cases={
                "t1new": [x+x for x in ("a", "b")],
                "t2new": [x+x for x in ("aa", "ab", "ba", "bb")],
                "t3new": [x+x for x in ("aaa", "aab", "aba", "baa", "abb", "bab", "bba", "bbb")],
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
        pq.add_st_operator(1.0, ["v"], ["t1"])
        pq.add_operator_product(1.0, ["v"])
        pq.add_commutator(1.0, ["v"], ["t2"])
        pq.add_double_commutator(1.0, ["v"], ["t1"], ["t2"])
        pq.add_triple_commutator(0.5, ["v"], ["t1"], ["t1"], ["t2"])
        pq.add_quadruple_commutator(1/6, ["v"], ["t1"], ["t1"], ["t1"], ["t2"])
        pq.simplify()
        terms_t3 = pq.fully_contracted_strings()

        for n, (terms, name) in enumerate([(terms_t1, "T1"), (terms_t2, "T2"), (terms_t3, "T3")]):
            if spin == "ghf":
                spins_list = [(None,) * (n+1)]
            elif spin == "rhf":
                spins_list = [(("a", "b") * (n+1))[:n+1]]
            elif spin == "uhf":
                spins_list = list(itertools.product("ab", repeat=n+1))

            for spins in spins_list:
                qccg.clear()
                qccg.set_spin(spin)

                occ = index.index_factory(index.ExternalIndex, ["i", "j", "k"][:n+1], ["o", "o", "o"][:n+1], spins)
                vir = index.index_factory(index.ExternalIndex, ["a", "b", "c"][:n+1], ["v", "v", "v"][:n+1], spins)

                if spin == "uhf":
                    output = tensor.FermionicAmplitude("t%dnew_%s" % (n+1, "".join(spins+spins)), occ, vir)
                    shape = ", ".join(["nocc[%d]" % "ab".index(s) for s in spins] + ["nvir[%d]" % "ab".index(s) for s in spins])
                else:
                    output = tensor.FermionicAmplitude("t%dnew" % (n+1), occ, vir)
                    shape = ", ".join(["nocc"] * (n+1) + ["nvir"] * (n+1))

                index_spins = {index.character: index.spin for index in occ+vir}
                expression = read.from_pdaggerq(terms, index_spins=index_spins)
                expression = expression.expand_spin_orbitals()

                einsums = write.write_einsum(expression, output, indent=4)
                einsums = "    %s = np.zeros((%s), dtype=np.float64)\n" % (output.symbol, shape) + einsums
                function_printer.write_python(einsums+"\n", comment="%s amplitudes" % name)
