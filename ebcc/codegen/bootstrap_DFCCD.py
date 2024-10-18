"""
Generate the DF-CCD code.
"""

import sys

import pdaggerq
from albert.qc._pdaggerq import import_from_pdaggerq
from albert.tensor import Tensor

from ebcc.codegen.bootstrap_common import *

# Get the spin case
spin = sys.argv[1]
if spin == "ghf":
    raise NotImplementedError

# Set up the code generators
code_generators = {
    "einsum": EinsumCodeGen(
        stdout=open(f"{spin[0].upper()}DFCCD.py", "w"),
        name_generator=name_generators[spin],
        spin=spin,
    ),
}

# Write the preamble
for codegen in code_generators.values():
    codegen.preamble()

# Set up pdaggerq
pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

with Stopwatch("Energy"):
    # Get the energy contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["1"]])
    pq.add_st_operator(1.0, ["f"], ["t2"])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms = pq.fully_contracted_strings()
    terms = remove_hf_energy(terms)

    # Get the energy in albert format
    expr = import_from_pdaggerq(terms)
    expr = spin_integrate(expr, spin)
    expr = tuple(e.apply(get_density_fit()) for e in expr)
    output = tuple(Tensor(name="e_cc") for _ in range(len(expr)))
    output, expr = optimise(output, expr, spin, strategy="exhaust")
    returns = (Tensor(name="e_cc"),)

    # Generate the energy code
    for codegen in code_generators.values():
        codegen(
            "energy",
            returns,
            output,
            expr,
        )

with Stopwatch("T amplitudes"):
    # Get the T2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e2(i,j,b,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t2"])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms = pq.fully_contracted_strings()

    # Get the T amplitudes in albert format
    expr = []
    output = []
    returns = []
    for index_spins in get_amplitude_spins(2, spin):
        indices = default_indices["o"][: 2] + default_indices["v"][: 2]
        expr_n = import_from_pdaggerq(terms, index_spins=index_spins)
        expr_n = spin_integrate(expr_n, spin)
        expr_n = tuple(e.apply(get_density_fit()) for e in expr_n)
        output_n = get_t_amplitude_outputs(expr_n, f"t2new")
        returns_n = (Tensor(*tuple(Index(i, index_spins[i]) for i in indices), name=f"t2new"),)
        expr.extend(expr_n)
        output.extend(output_n)
        returns.extend(returns_n)
    output, expr = optimise(output, expr, spin, strategy="exhaust")

    # Generate the T amplitude code
    for name, codegen in code_generators.items():
        if name == "einsum":
            kwargs = {
                "preamble": "t1new = Namespace()\nt2new = Namespace()" if spin == "uhf" else None,
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "update_amps",
            returns,
            output,
            expr,
            **kwargs,
        )

with Stopwatch("L amplitudes"):
    # Get the L2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["1"]])
    pq.set_right_operators([["1"]])
    pq.add_st_operator(1.0, ["f", "e2(a,b,j,i)"], ["t2"])
    pq.add_st_operator(1.0, ["v", "e2(a,b,j,i)"], ["t2"])
    pq.set_left_operators([["l2"]])
    pq.add_st_operator(1.0, ["f", "e2(a,b,j,i)"], ["t2"])
    pq.add_st_operator(1.0, ["v", "e2(a,b,j,i)"], ["t2"])
    pq.add_st_operator(-1.0, ["e2(a,b,j,i)", "f"], ["t2"])
    pq.add_st_operator(-1.0, ["e2(a,b,j,i)", "v"], ["t2"])
    pq.simplify()
    terms = pq.fully_contracted_strings()

    # Get the L amplitudes in albert format
    expr = []
    output = []
    returns = []
    for index_spins in get_amplitude_spins(2, spin):
        indices = default_indices["v"][: 2] + default_indices["o"][: 2]
        expr_n = import_from_pdaggerq(terms, index_spins=index_spins)
        expr_n = spin_integrate(expr_n, spin)
        expr_n = tuple(e.apply(get_density_fit()) for e in expr_n)
        output_n = get_l_amplitude_outputs(expr_n, f"l2new")
        returns_n = (Tensor(*tuple(Index(i, index_spins[i]) for i in indices), name=f"l2new"),)
        expr.extend(expr_n)
        output.extend(output_n)
        returns.extend(returns_n)
    output, expr = optimise(output, expr, spin, strategy="opt")

    # Generate the L amplitude code
    for name, codegen in code_generators.items():
        if name == "einsum":
            kwargs = {
                "preamble": "l1new = Namespace()\nl2new = Namespace()" if spin == "uhf" else None,
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "update_lams",
            returns,
            output,
            expr,
            **kwargs,
        )

with Stopwatch("1RDM"):
    # Get the 1RDM contractions in pdaggerq format
    terms = {}
    for sectors, indices in [("oo", "ij"), ("ov", "ia"), ("vo", "ai"), ("vv", "ab")]:
        pq.clear()
        pq.set_left_operators([["1"], ["l2"]])
        pq.add_st_operator(1.0, [f"e1({','.join(indices)})"], ["t2"])
        pq.simplify()
        terms[sectors, indices] = pq.fully_contracted_strings()

    # Get the 1RDM in albert format
    expr = []
    output = []
    returns = []
    for sectors, indices in [("oo", "ij"), ("ov", "ia"), ("vo", "ai"), ("vv", "ab")]:
        for index_spins in get_density_spins(1, spin, indices):
            expr_n = import_from_pdaggerq(terms[sectors, indices], index_spins=index_spins)
            if not (isinstance(expr_n, int) and expr_n == 0):
                expr_n = spin_integrate(expr_n, spin)
                if spin == "rhf":
                    expr_n = tuple(e * 2 for e in expr_n)
                expr_n = tuple(e.apply(get_density_fit()) for e in expr_n)
                output_n = get_density_outputs(expr_n, f"d", indices)
                returns_n = (Tensor(*tuple(Index(i, index_spins[i], space=s) for i, s in zip(indices, sectors)), name=f"d"),)
                expr.extend(expr_n)
                output.extend(output_n)
                returns.extend(returns_n)
    output, expr = optimise(output, expr, spin, strategy="exhaust")

    # Generate the 1RDM code
    for name, codegen in code_generators.items():
        if name == "einsum":
            def get_postamble(n, spin, name="rdm{n}"):
                nm = name.format(n=n)
                postamble = ""
                if spin != "uhf":
                    for occ in ("ov", "vo"):
                        shape = ", ".join(f"t2.shape[{'0' if o == 'o' else '-1'}]" for o in occ)
                        postamble += f"{nm}.{occ} = np.zeros(({shape}))\n"
                else:
                    for s in "ab":
                        for occ in ("oo", "vv"):
                            shape = ", ".join(f"t2.{ss}{ss}.shape[{'0' if o == 'o' else '-1'}]" for ss, o in zip(s+s, occ))
                            postamble += f"{nm}.{s}{s}.{occ} = np.zeros(({shape}))\n"
                return postamble + get_density_einsum_postamble(n, spin)
            kwargs = {
                "preamble": get_density_einsum_preamble(1, spin),
                "postamble": get_postamble(1, spin),
            }
        else:
            kwargs = {}
        codegen(
            "make_rdm1_f",
            returns,
            output,
            expr,
            **kwargs,
        )

with Stopwatch("2RDM"):
    # Get the 2RDM contractions in pdaggerq format
    terms = {}
    for sectors, indices in [
        ("oooo", "ijkl"),
        ("ooov", "ijka"),
        ("oovo", "ijak"),
        ("ovoo", "iajk"),
        ("vooo", "aijk"),
        ("oovv", "ijab"),
        ("ovov", "iajb"),
        ("ovvo", "iabj"),
        ("voov", "aijb"),
        ("vovo", "aibj"),
        ("vvoo", "abij"),
        ("ovvv", "iabc"),
        ("vovv", "aibc"),
        ("vvov", "abic"),
        ("vvvo", "abci"),
        ("vvvv", "abcd"),
    ]:
        pq.clear()
        pq.set_left_operators([["1"], ["l2"]])
        pq.add_st_operator(
            1.0, [f"e2({indices[0]},{indices[1]},{indices[3]},{indices[2]})"], ["t2"]
        )
        pq.simplify()
        terms[sectors, indices] = pq.fully_contracted_strings()

    # Get the 2RDM in albert format
    expr = []
    output = []
    returns = []
    for sectors, indices in [
        ("oooo", "ijkl"),
        ("ooov", "ijka"),
        ("oovo", "ijak"),
        ("ovoo", "iajk"),
        ("vooo", "aijk"),
        ("oovv", "ijab"),
        ("ovov", "iajb"),
        ("ovvo", "iabj"),
        ("voov", "aijb"),
        ("vovo", "aibj"),
        ("vvoo", "abij"),
        ("ovvv", "iabc"),
        ("vovv", "aibc"),
        ("vvov", "abic"),
        ("vvvo", "abci"),
        ("vvvv", "abcd"),
    ]:
        for index_spins in get_density_spins(2, spin, indices):
            expr_n = import_from_pdaggerq(terms[sectors, indices], index_spins=index_spins)
            if not (isinstance(expr_n, int) and expr_n == 0):
                expr_n = spin_integrate(expr_n, spin)
                expr_n = tuple(e.apply(get_density_fit()) for e in expr_n)
                output_n = get_density_outputs(expr_n, f"Γ", indices)
                returns_n = (Tensor(*tuple(Index(i, index_spins[i], space=s) for i, s in zip(indices, sectors)), name=f"Γ"),)
                expr.extend(expr_n)
                output.extend(output_n)
                returns.extend(returns_n)
    output, expr = optimise(output, expr, spin, strategy="trav")

    # Generate the 2RDM code
    for name, codegen in code_generators.items():
        if name == "einsum":
            def get_postamble(n, spin, name="rdm{n}"):
                nm = name.format(n=n)
                postamble = ""
                if spin != "uhf":
                    for occ in ("ooov", "oovo", "ovoo", "vooo", "ovvv", "vovv", "vvov", "vvvo"):
                        shape = ", ".join(f"t2.shape[{'0' if o == 'o' else '-1'}]" for o in occ)
                        postamble += f"{nm}.{occ} = np.zeros(({shape}))\n"
                else:
                    for s1 in "ab":
                        for s2 in "ab":
                            for occ in ("ooov", "oovo", "ovoo", "vooo", "ovvv", "vovv", "vvov", "vvvo"):
                                shape = ", ".join(f"t2.{s}{s}{s}{s}.shape[{'0' if o == 'o' else '-1'}]" for o, s in zip(occ, s1+s1+s2+s2))
                                postamble += f"{nm}.{s1}{s1}{s2}{s2}.{occ} = np.zeros(({shape}))\n"
                return postamble + get_density_einsum_postamble(n, spin)
            kwargs = {
                "preamble": get_density_einsum_preamble(2, spin),
                "postamble": get_postamble(2, spin),
            }
        codegen(
            "make_rdm2_f",
            returns,
            output,
            expr,
            **kwargs,
        )

with Stopwatch("IP-EOM"):
    # Get the R1 contractions in pdaggerq format
    pq.clear()
    pq.set_right_operators_type("IP")
    pq.set_left_operators([["a*(i)"]])
    pq.set_right_operators([["r1"], ["r2"]])
    pq.add_st_operator(1.0, ["f"], ["t2"])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms_r1 = pq.fully_contracted_strings()
    terms_r1 = remove_e0_eom(terms_r1)

    # Get the R2 contractions in pdaggerq format
    pq.clear()
    pq.set_right_operators_type("IP")
    pq.set_left_operators([["a*(i)", "a*(j)", "a(a)"]])
    pq.set_right_operators([["r1"], ["r2"]])
    pq.add_st_operator(1.0, ["f"], ["t2"])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms_r2 = pq.fully_contracted_strings()
    terms_r2 = remove_e0_eom(terms_r2)

    # Get the R amplitudes in albert format
    terms = [terms_r1, terms_r2]
    expr = []
    output = []
    returns = []
    for n in range(2):
        for index_spins in get_amplitude_spins(n + 1, spin, which="ip"):
            indices = default_indices["o"][: n + 1] + default_indices["v"][: n]
            expr_n = import_from_pdaggerq(terms[n], index_spins=index_spins)
            expr_n = spin_integrate(expr_n, spin)
            expr_n = tuple(e.apply(get_density_fit()) for e in expr_n)
            output_n = get_t_amplitude_outputs(expr_n, f"r{n+1}new", indices=indices)
            returns_n = (Tensor(*tuple(Index(i, index_spins[i]) for i in indices), name=f"r{n+1}new"),)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)

    (returns_nr, output_nr, expr_nr), (returns_r, output_r, expr_r) = optimise_eom(returns, output, expr, spin, strategy="exhaust")

    # Generate the R amplitude intermediates code
    for name, codegen in code_generators.items():
        if name == "einsum":
            kwargs = {
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "hbar_matvec_ip_intermediates",
            returns_nr,
            output_nr,
            expr_nr,
            **kwargs,
        )

    # Generate the R amplitude code
    for name, codegen in code_generators.items():
        if name == "einsum":
            preamble = "ints = kwargs[\"ints\"]"
            if spin == "uhf":
                preamble += "\nr1new = Namespace()\nr2new = Namespace()"
            kwargs = {
                "preamble": preamble,
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "hbar_matvec_ip",
            returns_r,
            output_r,
            expr_r,
            **kwargs,
        )

with Stopwatch("EA-EOM"):
    # Get the R1 contractions in pdaggerq format
    pq.clear()
    pq.set_right_operators_type("EA")
    pq.set_left_operators([["a(a)"]])
    pq.set_right_operators([["r1"], ["r2"]])
    pq.add_st_operator(1.0, ["f"], ["t2"])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms_r1 = pq.fully_contracted_strings()
    terms_r1 = remove_e0_eom(terms_r1)

    # Get the R2 contractions in pdaggerq format
    pq.clear()
    pq.set_right_operators_type("EA")
    pq.set_left_operators([["a*(i)", "a(b)", "a(a)"]])
    pq.set_right_operators([["r1"], ["r2"]])
    pq.add_st_operator(1.0, ["f"], ["t2"])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms_r2 = pq.fully_contracted_strings()
    terms_r2 = remove_e0_eom(terms_r2)

    # Get the R amplitudes in albert format
    terms = [terms_r1, terms_r2]
    expr = []
    output = []
    returns = []
    for n in range(2):
        for index_spins in get_amplitude_spins(n + 1, spin, which="ea"):
            indices = default_indices["v"][: n + 1] + default_indices["o"][: n]
            expr_n = import_from_pdaggerq(terms[n], index_spins=index_spins)
            expr_n = spin_integrate(expr_n, spin)
            expr_n = tuple(e.apply(get_density_fit()) for e in expr_n)
            output_n = get_t_amplitude_outputs(expr_n, f"r{n+1}new", indices=indices)
            returns_n = (Tensor(*tuple(Index(i, index_spins[i]) for i in indices), name=f"r{n+1}new"),)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)

    (returns_nr, output_nr, expr_nr), (returns_r, output_r, expr_r) = optimise_eom(returns, output, expr, spin, strategy="exhaust")

    # Generate the R amplitude intermediates code
    for name, codegen in code_generators.items():
        if name == "einsum":
            kwargs = {
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "hbar_matvec_ea_intermediates",
            returns_nr,
            output_nr,
            expr_nr,
            **kwargs,
        )

    # Generate the R amplitude code
    for name, codegen in code_generators.items():
        if name == "einsum":
            preamble = "ints = kwargs[\"ints\"]"
            if spin == "uhf":
                preamble += "\nr1new = Namespace()\nr2new = Namespace()"
            kwargs = {
                "preamble": preamble,
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "hbar_matvec_ea",
            returns_r,
            output_r,
            expr_r,
            **kwargs,
        )

if spin == "ghf":  # FIXME
    with Stopwatch("EE-EOM"):
        # Get the R1 contractions in pdaggerq format
        pq.clear()
        pq.set_right_operators_type("EE")
        pq.set_left_operators([["e1(i,a)"]])
        pq.set_right_operators([["r1"], ["r2"]])
        pq.add_st_operator(1.0, ["f"], ["t2"])
        pq.add_st_operator(1.0, ["v"], ["t2"])
        pq.simplify()
        terms_r1 = pq.fully_contracted_strings()
        terms_r1 = remove_e0_eom(terms_r1)

        # Get the R2 contractions in pdaggerq format
        pq.clear()
        pq.set_right_operators_type("EE")
        pq.set_left_operators([["e2(i,j,b,a)"]])
        pq.set_right_operators([["r1"], ["r2"]])
        pq.add_st_operator(1.0, ["f"], ["t2"])
        pq.add_st_operator(1.0, ["v"], ["t2"])
        pq.simplify()
        terms_r2 = pq.fully_contracted_strings()
        terms_r2 = remove_e0_eom(terms_r2)

        # Get the R amplitudes in albert format
        terms = [terms_r1, terms_r2]
        expr = []
        output = []
        returns = []
        for n in range(2):
            for index_spins in get_amplitude_spins(n + 1, spin, which="ee"):
                indices = default_indices["o"][: n + 1] + default_indices["v"][: n + 1]
                expr_n = import_from_pdaggerq(terms[n], index_spins=index_spins)
                expr_n = spin_integrate(expr_n, spin)
                expr_n = tuple(e.apply(get_density_fit()) for e in expr_n)
                output_n = get_t_amplitude_outputs(expr_n, f"r{n+1}new", indices=indices)
                returns_n = (Tensor(*tuple(Index(i, index_spins[i]) for i in indices), name=f"r{n+1}new"),)
                expr.extend(expr_n)
                output.extend(output_n)
                returns.extend(returns_n)

        (returns_nr, output_nr, expr_nr), (returns_r, output_r, expr_r) = optimise_eom(returns, output, expr, spin, strategy="trav")

        # Generate the R amplitude intermediates code
        for name, codegen in code_generators.items():
            if name == "einsum":
                kwargs = {
                    "as_dict": True,
                }
            else:
                kwargs = {}
            codegen(
                "hbar_matvec_ee_intermediates",
                returns_nr,
                output_nr,
                expr_nr,
                **kwargs,
            )

        # Generate the R amplitude code
        for name, codegen in code_generators.items():
            if name == "einsum":
                preamble = "ints = kwargs[\"ints\"]"
                if spin == "uhf":
                    preamble += "\nr1new = Namespace()\nr2new = Namespace()"
                kwargs = {
                    "preamble": preamble,
                    "as_dict": True,
                }
            else:
                kwargs = {}
            codegen(
                "hbar_matvec_ee",
                returns_r,
                output_r,
                expr_r,
                **kwargs,
            )

with Stopwatch("L-IP-EOM"):
    # Get the L1 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators_type("IP")
    pq.set_left_operators([["l1"], ["l2"]])
    pq.set_right_operators([["a(i)"]])
    pq.add_st_operator(1.0, ["f"], ["t2"])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms_r1 = pq.fully_contracted_strings()
    terms_r1 = remove_e0_eom(terms_r1)

    # Get the L2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators_type("IP")
    pq.set_right_operators([["a*(a)", "a(j)", "a(i)"]])
    pq.set_left_operators([["l1"], ["l2"]])
    pq.add_st_operator(1.0, ["f"], ["t2"])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms_r2 = pq.fully_contracted_strings()
    terms_r2 = remove_e0_eom(terms_r2)

    # Get the L amplitudes in albert format
    terms = [terms_r1, terms_r2]
    expr = []
    output = []
    returns = []
    for n in range(2):
        for index_spins in get_amplitude_spins(n + 1, spin, which="ip"):
            indices = default_indices["o"][: n + 1] + default_indices["v"][: n]
            expr_n = import_from_pdaggerq(terms[n], index_spins=index_spins, l_is_lambda=False)
            expr_n = spin_integrate(expr_n, spin)
            expr_n = tuple(e.apply(get_density_fit()) for e in expr_n)
            output_n = get_t_amplitude_outputs(expr_n, f"r{n+1}new", indices=indices)
            returns_n = (Tensor(*tuple(Index(i, index_spins[i]) for i in indices), name=f"r{n+1}new"),)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)

    (returns_nr, output_nr, expr_nr), (returns_r, output_r, expr_r) = optimise_eom(returns, output, expr, spin, strategy="exhaust")

    # Generate the L amplitude intermediates code
    for name, codegen in code_generators.items():
        if name == "einsum":
            kwargs = {
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "hbar_lmatvec_ip_intermediates",
            returns_nr,
            output_nr,
            expr_nr,
            **kwargs,
        )

    # Generate the L amplitude code
    for name, codegen in code_generators.items():
        if name == "einsum":
            preamble = "ints = kwargs[\"ints\"]"
            if spin == "uhf":
                preamble += "\nr1new = Namespace()\nr2new = Namespace()"
            kwargs = {
                "preamble": preamble,
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "hbar_lmatvec_ip",
            returns_r,
            output_r,
            expr_r,
            **kwargs,
        )

with Stopwatch("L-EA-EOM"):
    # Get the L1 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators_type("EA")
    pq.set_left_operators([["l1"], ["l2"]])
    pq.set_right_operators([["a*(a)"]])
    pq.add_st_operator(1.0, ["f"], ["t2"])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms_r1 = pq.fully_contracted_strings()
    terms_r1 = remove_e0_eom(terms_r1)

    # Get the L2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators_type("EA")
    pq.set_right_operators([["a*(a)", "a*(b)", "a(i)"]])
    pq.set_left_operators([["l1"], ["l2"]])
    pq.add_st_operator(1.0, ["f"], ["t2"])
    pq.add_st_operator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms_r2 = pq.fully_contracted_strings()
    terms_r2 = remove_e0_eom(terms_r2)

    # Get the L amplitudes in albert format
    terms = [terms_r1, terms_r2]
    expr = []
    output = []
    returns = []
    for n in range(2):
        for index_spins in get_amplitude_spins(n + 1, spin, which="ea"):
            indices = default_indices["v"][: n + 1] + default_indices["o"][: n]
            expr_n = import_from_pdaggerq(terms[n], index_spins=index_spins, l_is_lambda=False)
            expr_n = spin_integrate(expr_n, spin)
            expr_n = tuple(e.apply(get_density_fit()) for e in expr_n)
            output_n = get_t_amplitude_outputs(expr_n, f"r{n+1}new", indices=indices)
            returns_n = (Tensor(*tuple(Index(i, index_spins[i]) for i in indices), name=f"r{n+1}new"),)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)

    (returns_nr, output_nr, expr_nr), (returns_r, output_r, expr_r) = optimise_eom(returns, output, expr, spin, strategy="exhaust")

    # Generate the L amplitude intermediates code
    for name, codegen in code_generators.items():
        if name == "einsum":
            kwargs = {
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "hbar_lmatvec_ea_intermediates",
            returns_nr,
            output_nr,
            expr_nr,
            **kwargs,
        )

    # Generate the L amplitude code
    for name, codegen in code_generators.items():
        if name == "einsum":
            preamble = "ints = kwargs[\"ints\"]"
            if spin == "uhf":
                preamble += "\nr1new = Namespace()\nr2new = Namespace()"
            kwargs = {
                "preamble": preamble,
                "as_dict": True,
            }
        else:
            kwargs = {}
        codegen(
            "hbar_lmatvec_ea",
            returns_r,
            output_r,
            expr_r,
            **kwargs,
        )

if spin == "ghf":  # FIXME
    with Stopwatch("L-EE-EOM"):
        # Get the L1 contractions in pdaggerq format
        pq.clear()
        pq.set_left_operators_type("EE")
        pq.set_left_operators([["l1"], ["l2"]])
        pq.set_right_operators([["e1(a,i)"]])
        pq.add_st_operator(1.0, ["f"], ["t2"])
        pq.add_st_operator(1.0, ["v"], ["t2"])
        pq.simplify()
        terms_r1 = pq.fully_contracted_strings()
        terms_r1 = remove_e0_eom(terms_r1)

        # Get the L2 contractions in pdaggerq format
        pq.clear()
        pq.set_left_operators_type("EE")
        pq.set_left_operators([["l1"], ["l2"]])
        pq.set_right_operators([["e2(a,b,j,i)"]])
        pq.add_st_operator(1.0, ["f"], ["t2"])
        pq.add_st_operator(1.0, ["v"], ["t2"])
        pq.simplify()
        terms_r2 = pq.fully_contracted_strings()
        terms_r2 = remove_e0_eom(terms_r2)

        # Get the L amplitudes in albert format
        terms = [terms_r1, terms_r2]
        expr = []
        output = []
        returns = []
        for n in range(2):
            for index_spins in get_amplitude_spins(n + 1, spin, which="ee"):
                indices = default_indices["o"][: n + 1] + default_indices["v"][: n + 1]
                expr_n = import_from_pdaggerq(terms[n], index_spins=index_spins, l_is_lambda=False)
                expr_n = spin_integrate(expr_n, spin)
                expr_n = tuple(e.apply(get_density_fit()) for e in expr_n)
                output_n = get_t_amplitude_outputs(expr_n, f"r{n+1}new", indices=indices)
                returns_n = (Tensor(*tuple(Index(i, index_spins[i]) for i in indices), name=f"r{n+1}new"),)
                expr.extend(expr_n)
                output.extend(output_n)
                returns.extend(returns_n)

        (returns_nr, output_nr, expr_nr), (returns_r, output_r, expr_r) = optimise_eom(returns, output, expr, spin, strategy="trav")

        # Generate the L amplitude intermediates code
        for name, codegen in code_generators.items():
            if name == "einsum":
                kwargs = {
                    "as_dict": True,
                }
            else:
                kwargs = {}
            codegen(
                "hbar_lmatvec_ee_intermediates",
                returns_nr,
                output_nr,
                expr_nr,
                **kwargs,
            )

        # Generate the L amplitude code
        for name, codegen in code_generators.items():
            if name == "einsum":
                preamble = "ints = kwargs[\"ints\"]"
                if spin == "uhf":
                    preamble += "\nr1new = Namespace()\nr2new = Namespace()"
                kwargs = {
                    "preamble": preamble,
                    "as_dict": True,
                }
            else:
                kwargs = {}
            codegen(
                "hbar_lmatvec_ee",
                returns_r,
                output_r,
                expr_r,
                **kwargs,
            )

for codegen in code_generators.values():
    codegen.postamble()
    codegen.stdout.close()