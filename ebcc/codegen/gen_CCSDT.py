"""Script to generate equations for the CCSDT model.

This uses pdaggerq and qccg instead of qwick.
"""

import itertools
from ebcc.codegen import common
import qccg
from qccg import index, tensor, read, write
import pdaggerq

# Spin integration mode
spin = "uhf"

# pdaggerq setup
pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

# Printer setup
FunctionPrinter = common.get_function_printer(spin)
timer = common.Stopwatch()

with common.FilePrinter("%sCCSDT" % spin[0].upper()) as file_printer:
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
        function_printer.write_python(einsums+"\n", comment="energy")

    # Get amplitudes function:
    with FunctionPrinter(
            file_printer,
            "update_amps",
            ["f", "v", "nocc", "nvir", "t1", "t2", "t3"],
            ["t1new", "t2new", "t3new"],
            spin_cases={
                "t1new": [x+x for x in ("a", "b")],
                "t2new": [x+x for x in ("aa", "ab", "bb")],
                "t3new": [x+x for x in ("aaa", "aab", "abb", "bbb")],
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
                spins_list = [list(y) for y in sorted(set("".join(sorted(x)) for x in itertools.product("ab", repeat=n+1)))]

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

    # Get lambda amplitudes function:
    with FunctionPrinter(
            file_printer,
            "update_lams",
            ["f", "v", "nocc", "nvir", "t1", "t2", "t3", "l1", "l2", "l3"],
            ["l1new", "l2new", "l3new"],
            spin_cases={
                "l1new": [x+x for x in ("a", "b")],
                "l2new": [x+x for x in ("aa", "ab", "bb")],
                "l3new": [x+x for x in ("aaa", "aab", "abb", "bbb")],
            },
            timer=timer,
    ) as function_printer:
        # L1 residuals:
        pq.clear()
        pq.set_left_operators([["1"]])
        pq.set_right_operators([["1"]])
        pq.add_st_operator(1.0, ["f", "e1(a,i)"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v", "e1(a,i)"], ["t1", "t2", "t3"])
        pq.set_left_operators([["l1"], ["l2"], ["l3"]])
        pq.add_st_operator( 1.0, ["f", "e1(a,i)"], ["t1", "t2", "t3"])
        pq.add_st_operator( 1.0, ["v", "e1(a,i)"], ["t1", "t2", "t3"])
        pq.add_st_operator(-1.0, ["e1(a,i)", "f"], ["t1", "t2", "t3"])
        pq.add_st_operator(-1.0, ["e1(a,i)", "v"], ["t1", "t2", "t3"])
        pq.simplify()
        terms_l1 = pq.fully_contracted_strings()

        # L2 residuals:
        pq.clear()
        pq.set_left_operators([["1"]])
        pq.set_right_operators([["1"]])
        pq.add_st_operator(1.0, ["f", "e2(a,b,j,i)"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v", "e2(a,b,j,i)"], ["t1", "t2", "t3"])
        pq.set_left_operators([["l1"], ["l2"], ["l3"]])
        pq.add_st_operator( 1.0, ["f", "e2(a,b,j,i)"], ["t1", "t2", "t3"])
        pq.add_st_operator( 1.0, ["v", "e2(a,b,j,i)"], ["t1", "t2", "t3"])
        pq.add_st_operator(-1.0, ["e2(a,b,j,i)", "f"], ["t1", "t2", "t3"])
        pq.add_st_operator(-1.0, ["e2(a,b,j,i)", "v"], ["t1", "t2", "t3"])
        pq.simplify()
        terms_l2 = pq.fully_contracted_strings()

        # L3 residuals:
        pq.clear()
        pq.set_left_operators([["1"]])
        pq.set_right_operators([["1"]])
        pq.add_st_operator(1.0, ["f", "e3(a,b,c,k,j,i)"], ["t1", "t2", "t3"])
        pq.add_st_operator(1.0, ["v", "e3(a,b,c,k,j,i)"], ["t1", "t2", "t3"])
        pq.set_left_operators([["l1"], ["l2"], ["l3"]])
        pq.add_st_operator( 1.0, ["f", "e3(a,b,c,k,j,i)"], ["t1", "t2", "t3"])
        pq.add_st_operator( 1.0, ["v", "e3(a,b,c,k,j,i)"], ["t1", "t2", "t3"])
        pq.add_st_operator(-1.0, ["e3(a,b,c,k,j,i)", "f"], ["t1", "t2", "t3"])
        pq.add_st_operator(-1.0, ["e3(a,b,c,k,j,i)", "v"], ["t1", "t2", "t3"])
        pq.simplify()
        terms_l3 = pq.fully_contracted_strings()

        expressions = []
        outputs = []
        for n, terms in enumerate([terms_l1, terms_l2, terms_l3]):
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
                    occ = index.index_factory(index.ExternalIndex, ["i", "j", "k"][:n+1], ["o", "o", "o"][:n+1], ["r", "r", "r"][:n+1])
                    vir = index.index_factory(index.ExternalIndex, ["a", "b", "c"][:n+1], ["v", "v", "v"][:n+1], ["r", "r", "r"][:n+1])
                    output = tensor.FermionicAmplitude("l%dnew" % (n+1), vir, occ)
                    shape = ", ".join(["nvir"] * (n+1) + ["nocc"] * (n+1))
                elif spin == "uhf":
                    occ = index.index_factory(index.ExternalIndex, ["i", "j", "k"][:n+1], ["o", "o", "o"][:n+1], spins)
                    vir = index.index_factory(index.ExternalIndex, ["a", "b", "c"][:n+1], ["v", "v", "v"][:n+1], spins)
                    output = tensor.FermionicAmplitude("l%dnew" % (n+1), vir, occ)
                    shape = ", ".join(["nvir[%d]" % "ab".index(s) for s in spins] + ["nocc[%d]" % "ab".index(s) for s in spins])
                elif spin == "ghf":
                    occ = index.index_factory(index.ExternalIndex, ["i", "j", "k"][:n+1], ["o", "o", "o"][:n+1], [None, None, None][:n+1])
                    vir = index.index_factory(index.ExternalIndex, ["a", "b", "c"][:n+1], ["v", "v", "v"][:n+1], [None, None, None][:n+1])
                    output = tensor.FermionicAmplitude("l%dnew" % (n+1), vir, occ)
                    shape = ", ".join(["nvir"] * (n+1) + ["nocc"] * (n+1))

                index_spins = {index.character: s for index, s in zip(vir+occ, spins+spins)}
                expression = read.from_pdaggerq(terms, index_spins=index_spins)
                expression = expression.expand_spin_orbitals()

                expressions.append(expression)
                outputs.append(output)

        final_outputs = outputs
        expressions, outputs = qccg.optimisation.optimise_expression_gristmill(
                expressions,
                outputs,
        )
        einsums = write.write_opt_einsums(
                expressions,
                outputs,
                final_outputs,
                indent=4,
                einsum_function="einsum",
        )
        function_printer.write_python(einsums+"\n", comment="L amplitudes")

    # Get 1RDM expressions:
    with FunctionPrinter(
            file_printer,
            "make_rdm1_f",
            ["f", "v", "nocc", "nvir", "t1", "t2", "t3", "l1", "l2", "l3"],
            ["rdm1_f"],
            spin_cases={
                "rdm1_f": ["aa", "bb"],
            },
            return_dict=False,
            timer=timer,
    ) as function_printer:
        if spin != "uhf":
            function_printer.write_python(
                    "    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))\n"
            )
        else:
            function_printer.write_python(
                    "    delta = Namespace(aa=Namespace(), bb=Namespace())\n"
                    "    delta.aa = Namespace(oo=np.eye(nocc[0]), vv=np.eye(nvir[0]))\n"
                    "    delta.bb = Namespace(oo=np.eye(nocc[1]), vv=np.eye(nvir[1]))\n"
            )

        if spin == "ghf":
            spins_list = [(None, None)]
        elif spin == "rhf":
            spins_list = [("a", "a")]
        elif spin == "uhf":
            spins_list = [("a", "a"), ("b", "b")]

        expressions = []
        outputs = []
        terms_rdm1 = []
        for sectors, indices in [("oo", "ij"), ("ov", "ia"), ("vo", "ai"), ("vv", "ab")]:
            pq.clear()
            pq.set_left_operators([["1"], ["l1"], ["l2"], ["l3"]])
            pq.add_st_operator(1.0, ["e1(%s,%s)" % tuple(indices)], ["t1", "t2", "t3"])
            pq.simplify()
            terms = pq.fully_contracted_strings()

            for spins in spins_list:
                qccg.clear()
                qccg.set_spin(spin)

                if spin == "rhf":
                    inds = index.index_factory(index.ExternalIndex, indices, sectors, "rr")
                    output = tensor.RDM1(inds)
                    shape = ", ".join(["nocc" if o == "o" else "nvir" for o in sectors])
                elif spin == "uhf":
                    inds = index.index_factory(index.ExternalIndex, indices, sectors, spins)
                    output = tensor.RDM1(inds)
                    shape = ", ".join(["nocc[%d]" % "ab".index(s) if o == "o" else "nvir[%d]" % "ab".index(s) for o, s in zip(sectors, spins)])
                elif spin == "ghf":
                    inds = index.index_factory(index.ExternalIndex, indices, sectors, [None]*2)
                    output = tensor.RDM1(inds)
                    shape = ", ".join(["nocc" if o == "o" else "nvir" for o in sectors])

                index_spins = {index.character: s for index, s in zip(inds, spins+spins)}
                expression = read.from_pdaggerq(terms, index_spins=index_spins)
                expression = expression.expand_spin_orbitals()

                if spin == "rhf":
                    expression = expression * 2

                expressions.append(expression)
                outputs.append(output)

        final_outputs = outputs
        expressions, outputs = qccg.optimisation.optimise_expression_gristmill(
                expressions,
                outputs,
        )
        einsums = write.write_opt_einsums(
                expressions,
                outputs,
                final_outputs,
                indent=4,
                einsum_function="einsum",
                add_occupancies={"f", "v", "rdm1_f", "rdm2_f", "delta"},
        )
        function_printer.write_python(einsums+"\n", comment="RDM1")

        if spin != "uhf":
            function_printer.write_python("    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])\n")
        else:
            function_printer.write_python(
                "    rdm1_f_aa = np.block([[rdm1_f_aa_oo, rdm1_f_aa_ov], [rdm1_f_aa_vo, rdm1_f_aa_vv]])\n"
                "    rdm1_f_bb = np.block([[rdm1_f_bb_oo, rdm1_f_bb_ov], [rdm1_f_bb_vo, rdm1_f_bb_vv]])\n"
            )

    # Get 2RDM expressions:
    with FunctionPrinter(
            file_printer,
            "make_rdm2_f",
            ["f", "v", "nocc", "nvir", "t1", "t2", "t3", "l1", "l2", "l3"],
            ["rdm2_f"],
            spin_cases={
                "rdm2_f": ["aaaa", "aabb", "bbbb"],
            },
            return_dict=False,
            timer=timer,
    ) as function_printer:
        if spin != "uhf":
            function_printer.write_python(
                    "    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))\n"
            )
        else:
            function_printer.write_python(
                    "    delta = Namespace(aa=Namespace(), bb=Namespace())\n"
                    "    delta.aa = Namespace(oo=np.eye(nocc[0]), vv=np.eye(nvir[0]))\n"
                    "    delta.bb = Namespace(oo=np.eye(nocc[1]), vv=np.eye(nvir[1]))\n"
            )

        if spin == "ghf":
            spins_list = [(None, None, None, None)]
        else:
            spins_list = [("a", "a", "a", "a"), ("a", "b", "a", "b"), ("b", "b", "b", "b")]

        expressions = []
        outputs = []
        terms_rdm1 = []
        for sectors, indices in [
                ("oooo", "ijkl"), ("ooov", "ijka"), ("oovo", "ijak"), ("ovoo", "iajk"),
                ("vooo", "aijk"), ("oovv", "ijab"), ("ovov", "iajb"), ("ovvo", "iabj"),
                ("voov", "aijb"), ("vovo", "aibj"), ("vvoo", "abij"), ("ovvv", "iabc"),
                ("vovv", "aibc"), ("vvov", "abic"), ("vvvo", "abci"), ("vvvv", "abcd"),
        ]:
            pq.clear()
            pq.set_left_operators([["1"], ["l1"], ["l2"], ["l3"]])
            pq.add_st_operator(1.0, ["e2(%s,%s,%s,%s)" % tuple(indices[:2]+indices[2:][::-1])], ["t1", "t2", "t3"])
            pq.simplify()
            terms = pq.fully_contracted_strings()

            for spins in spins_list:
                qccg.clear()
                qccg.set_spin(spin)

                if spin == "rhf":
                    inds = index.index_factory(index.ExternalIndex, indices, sectors, "rrrr")
                    output = tensor.RDM2(inds)
                    shape = ", ".join(["nocc" if o == "o" else "nvir" for o in sectors])
                elif spin == "uhf":
                    inds = index.index_factory(index.ExternalIndex, indices, sectors, spins)
                    output = tensor.RDM2(inds)
                    shape = ", ".join(["nocc[%d]" % "ab".index(s) if o == "o" else "nvir[%d]" % "ab".index(s) for o, s in zip(sectors, spins)])
                elif spin == "ghf":
                    inds = index.index_factory(index.ExternalIndex, indices, sectors, [None]*4)
                    output = tensor.RDM2(inds)
                    shape = ", ".join(["nocc" if o == "o" else "nvir" for o in sectors])

                index_spins = {index.character: s for index, s in zip(inds, spins+spins)}
                expression = read.from_pdaggerq(terms, index_spins=index_spins)
                expression = expression.expand_spin_orbitals()

                expressions.append(expression)
                outputs.append(output)

        final_outputs = outputs
        expressions, outputs = qccg.optimisation.optimise_expression_gristmill(
                expressions,
                outputs,
        )
        einsums = write.write_opt_einsums(
                expressions,
                outputs,
                final_outputs,
                indent=4,
                einsum_function="einsum",
                add_occupancies={"f", "v", "rdm1_f", "rdm2_f", "delta"},
        )

        function_printer.write_python(einsums+"\n", comment="RDM2")

        if spin != "uhf":
            function_printer.write_python("    rdm2_f = pack_2e(%s)\n" % ", ".join(["rdm2_f_%s" % x for x in common.ov_2e]))
        else:
            function_printer.write_python(""
                    + "    rdm2_f_aaaa = pack_2e(%s)\n" % ", ".join(["rdm2_f_aaaa_%s" % x for x in common.ov_2e])
                    + "    rdm2_f_abab = pack_2e(%s)\n" % ", ".join(["rdm2_f_abab_%s" % x for x in common.ov_2e])
                    + "    rdm2_f_bbbb = pack_2e(%s)\n" % ", ".join(["rdm2_f_bbbb_%s" % x for x in common.ov_2e])
            )

        if spin == "rhf":
            function_printer.write_python("    rdm2_f = rdm2_f.swapaxes(1, 2)\n")
        elif spin == "uhf":
            function_printer.write_python(""
                    + "    rdm2_f_aaaa = rdm2_f_aaaa.swapaxes(1, 2)\n"
                    + "    rdm2_f_aabb = rdm2_f_abab.swapaxes(1, 2)\n"
                    + "    rdm2_f_bbbb = rdm2_f_bbbb.swapaxes(1, 2)\n"
            )

        if spin == "ghf":
            function_printer.write_python("    rdm2_f = rdm2_f.swapaxes(1, 2)\n")
