"""
Generate the MPn code.
"""

import sys

import pdaggerq
from albert.qc._pdaggerq import import_from_pdaggerq
from albert.tensor import Tensor

from ebcc.codegen import bootstrap_hugenholtz as hugenholtz
from ebcc.codegen.bootstrap_common import *

# Get the spin case
spin = sys.argv[1]
order = int(sys.argv[2])

# Set up the code generators
code_generators = {
    "einsum": EinsumCodeGen(
        stdout=open(f"{spin[0].upper()}MP{order}.py", "w"),
        name_generator=name_generators[spin],
        spin=spin,
    ),
}

# Write the preamble
for codegen in code_generators.values():
    codegen.preamble()

with Stopwatch("Energy"):
    # Get the energy contractions in pdaggerq format
    terms = sum([
        [hugenholtz.get_pdaggerq(graph) for graph in hugenholtz.get_hugenholtz_diagrams(n)]
        for n in range(2, order+1)
    ], [])

    # Get the energy in albert format
    expr = import_from_pdaggerq(terms)
    expr = spin_integrate(expr, spin)
    output = tuple(Tensor(name="e_mp") for _ in range(len(expr)))
    output, expr = optimise(output, expr, spin, strategy="exhaust")
    returns = (Tensor(name="e_mp"),)

    # Get the denominators for higher orders
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

    # Generate the energy code
    for codegen in code_generators.values():
        codegen(
            "energy",
            returns,
            output,
            expr,
        )

if order == 2:
    pq = pdaggerq.pq_helper("fermi")
    pq.set_print_level(0)

    with Stopwatch("1RDM"):
        # Get the 1RDM contractions in pdaggerq format
        terms = {
            ("oo", "ij"): [["+1.0", "d(i,j)"], ["-0.5", "t2(b,a,k,i)", "l2(k,j,b,a)"]],
            ("vv", "ab"): [["+0.5", "t2(b,c,i,j)", "l2(i,j,a,c)"]]
        }

        # Get the 1RDM in albert format
        expr = []
        output = []
        returns = []
        for sectors, indices in [("oo", "ij"), ("ov", "ia"), ("vo", "ai"), ("vv", "ab")]:
            if (sectors, indices) not in terms:
                continue
            for index_spins in get_density_spins(1, spin, indices):
                expr_n = import_from_pdaggerq(terms[sectors, indices], index_spins=index_spins)
                if not (isinstance(expr_n, int) and expr_n == 0):
                    expr_n = spin_integrate(expr_n, spin)
                    if spin == "rhf":
                        expr_n = tuple(e * 2 for e in expr_n)
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
                            for occ in ("ov", "vo"):
                                shape = ", ".join(f"t2.{s}{s}{s}{s}.shape[{'0' if o == 'o' else '-1'}]" for o in occ)
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
        terms = {
            ('oooo', 'ijkl'): [['+1.0', 'P(i,j)', 'd(i,k)', 'd(j,l)']],
            ('oovv', 'ijab'): [['+0.25', 't2(a,b,i,j)']],
            ('vvoo', 'abij'): [['+0.25', 'l2(a,b,i,j)']],
        }

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
            if (sectors, indices) not in terms:
                continue
            for index_spins in get_density_spins(2, spin, indices):
                expr_n = import_from_pdaggerq(terms[sectors, indices], index_spins=index_spins)
                if not (isinstance(expr_n, int) and expr_n == 0):
                    expr_n = spin_integrate(expr_n, spin)
                    output_n = get_density_outputs(expr_n, f"Γ", indices)
                    returns_n = (Tensor(*tuple(Index(i, index_spins[i], space=s) for i, s in zip(indices, sectors)), name=f"Γ"),)
                    expr.extend(expr_n)
                    output.extend(output_n)
                    returns.extend(returns_n)
        output, expr = optimise(output, expr, spin, strategy="exhaust")

        # Generate the 2RDM code
        for name, codegen in code_generators.items():
            if name == "einsum":
                def get_postamble(n, spin, name="rdm{n}"):
                    nm = name.format(n=n)
                    postamble = ""
                    if spin != "uhf":
                        for occ in [k[0] for k, v in terms.items() if v]:
                            shape = ", ".join(f"t2.shape[{'0' if o == 'o' else '-1'}]" for o in occ)
                            postamble += f"{nm}.{occ} = np.zeros(({shape}))\n"
                    else:
                        for s1, s2 in [("a", "a"), ("a", "b"), ("b", "b")]:
                            for occ in [k[0] for k, v in terms.items() if v]:
                                shape = ", ".join(f"t2.{s}{s}{s}{s}.shape[{'0' if o == 'o' else '-1'}]" for o, s in zip(occ, s1+s2+s1+s2))
                                postamble += f"{nm}.{s1}{s2}{s1}{s2}.{occ} = np.zeros(({shape}))\n"
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
        terms_r1 = [
            ["-1.0", "f(j,i)", "r1(j)"],
            ["+0.5", "<k,j||a,i>", "r2(a,k,j)"],
            ["-0.5", "<k,j||a,b>", "r1(k)", "t2(a,b,i,j)"],
        ]

        # Get the R2 contractions in pdaggerq format
        terms_r2 = [
            ["-1.0", "<k,a||i,j>", "r1(k)"],
            ["+1.0", "f(a,b)", "r2(b,i,j)"],
            ["-1.0", "P(i,j)", "f(k,j)", "r2(a,i,k)"],
        ]

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
        terms_r1 = [
            ["+1.0", "f(a,b)", "r1(b)"],
            ["-0.5", "<i,a||b,c>", "r2(b,c,i)"],
            ["-0.5", "<j,i||b,c>", "r1(c)", "t2(b,a,j,i)"],
        ]

        # Get the R2 contractions in pdaggerq format
        terms_r2 = [
            ["-1.0", "f(j,i)", "r2(a,b,j)"],
            ["+1.0", "P(a,b)", "f(a,c)", "r2(c,b,i)"],
            ["+1.0", "<a,b||c,i>", "r1(c)"],
        ]

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

    with Stopwatch("EE-EOM"):
        # Get the R1 contractions in pdaggerq format
        terms_r1 = [
            ["-1.0", "f(j,i)", "r1(a,j)"],  # ph-ph 0th
            ["+1.0", "f(a,b)", "r1(b,i)"],  # ph-ph 0th
            ["+1.00", "<i,j||a,b>", "r1(b,j)"],  # ph-ph 1st
            ["+0.25", "t2(a,c,k,l)", "<k,l||b,c>", "r1(b,i)"],  # ph-ph 2nd
            ["+0.25", "t2(b,c,k,l)", "<k,l||a,c>", "r1(b,i)"],  # ph-ph 2nd
            ["+0.25", "t2(c,d,i,k)", "<j,k||c,d>", "r1(a,j)"],  # ph-ph 2nd
            ["+0.25", "t2(c,d,j,k)", "<i,k||c,d>", "r1(a,j)"],  # ph-ph 2nd
            ["-0.50", "t2(a,c,i,k)", "<j,k||b,c>", "r1(b,j)"],  # ph-ph 2nd
            ["-0.50", "t2(b,c,j,k)", "<i,k||a,c>", "r1(b,j)"],  # ph-ph 2nd
            ["+1.0", "d(a,c)", "<k,l||i,d>", "r2(c,d,k,l)"],  # ph-pphh 1st
            ["-1.0", "d(a,d)", "<k,l||i,c>", "r2(c,d,k,l)"],  # ph-pphh 1st
            ["-1.0", "d(i,k)", "<a,l||c,d>", "r2(c,d,k,l)"],  # ph-pphh 1st
            ["+1.0", "d(i,l)", "<a,k||c,d>", "r2(c,d,k,l)"],  # ph-pphh 1st
        ]

        # Get the R2 contractions in pdaggerq format
        terms_r2 = [
            ["-1.0", "P(i,j)", "f(k,j)", "r2(a,b,i,k)"],  # pphh-pphh 0th
            ["+1.0", "P(a,b)", "f(a,c)", "r2(c,b,i,j)"],  # pphh-pphh 0th
            ["+1.00", "d(a,c)", "<k,b||i,j>", "r1(c,k)"],  # pphh-ph 1st
            ["-1.00", "d(b,c)", "<k,a||i,j>", "r1(c,k)"],  # pphh-ph 1st
            ["-1.00", "d(i,k)", "<a,b||c,j>", "r1(c,k)"],  # pphh-ph 1st
            ["+1.00", "d(j,k)", "<a,b||c,i>", "r1(c,k)"],  # pphh-ph 1st
        ]

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
                    "postamble": "r2new.baba = r2new.abab.transpose(1, 0, 3, 2)" if spin == "uhf" else None,  # FIXME
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

for codegen in code_generators.values():
    codegen.postamble()
    codegen.stdout.close()
