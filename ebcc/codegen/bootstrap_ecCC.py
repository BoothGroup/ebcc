"""
Generate the ecCC code.
"""

import itertools
import sys

import pdaggerq
from albert.qc._pdaggerq import remove_reference_energy, remove_reference_energy_eom, import_from_pdaggerq
from albert.qc.spin import ghf_to_uhf, ghf_to_rhf, get_amplitude_spins
from albert.qc import ghf, uhf, rhf
from albert.tensor import Tensor
from albert.index import Index, from_list
from albert.code._ebcc import EBCCCodeGenerator
from albert.misc import Stopwatch
from albert.opt.tools import _tensor_info, combine_expressions
from albert.opt import optimise
from albert.scalar import Scalar
from albert.misc import ExclusionSet

from ebcc.util import factorial
from ebcc.codegen.bootstrap_common import get_energy, get_amplitudes, get_rdm1, get_rdm2, get_eom, spin_integrate

# Get the spin case
spin = sys.argv[1]

# Set up the code generators
class CodeGenerator(EBCCCodeGenerator):
    _add_spaces = EBCCCodeGenerator._add_spaces
    _add_spaces.add("t1new")
    _add_spaces.add("t2new")

code_generators = {
    "einsum": CodeGenerator(
        stdout=open(f"{spin[0].upper()}ecCC.py", "w"),
    ),
}

# Write the preamble
for codegen in code_generators.values():
    codegen.preamble()

# Set up pdaggerq
pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

with Stopwatch("T amplitudes"):
    # Get the T1 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e1(i,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3", "t4"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3", "t4"])
    pq.simplify()
    terms_t1 = pq.strings()

    # Get the T2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e2(i,j,b,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2", "t3", "t4"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2", "t3", "t4"])
    pq.simplify()
    terms_t2 = pq.strings()

    # Get the T amplitudes in albert format
    output_expr, returns = get_amplitudes([terms_t1, terms_t2], spin, strategy=None)

    # Make the T3 and T4 indices active
    new_expr = []
    new_output = []
    for i, (o, e) in enumerate(output_expr):
        # Get the indices
        indices = set()
        for add in e.expand()._children:
            indices_a = indices.copy()
            def _get_indices(tensor):
                global indices_a
                if tensor.name in {"t3", "t4", "t4a"}:
                    indices_a = indices_a.union(set(tensor.external_indices))
                return tensor

            add = add.apply(_get_indices, Tensor)
            subs = {i: Index(i.name.upper(), spin=i.spin, space=i.space.upper()) for i in indices_a}

            def _substitute(tensor):
                return tensor.map_indices(subs)

            new_output.append(o.map_indices(subs))
            new_expr.append(add.apply(_substitute, Tensor))

    output_expr = list(zip(new_output, new_expr))

    #if spin == "uhf":
    #    # Why do we have to refactor these amplitudes? Is this a deficiency in albert?
    #    def _refactor(tensor):
    #        extra_factor = 1.0
    #        if type(tensor) in (uhf.T1, uhf.T2, uhf.T3, uhf.T4):
    #            spin = tuple(i.spin for i in tensor.external_indices[: tensor.rank // 2])
    #            extra_factor /= factorial(sum(s == "a" for s in spin))
    #            extra_factor /= factorial(sum(s == "b" for s in spin))
    #        return tensor * Scalar(extra_factor)
    #    output_expr = [(o, e.apply(_refactor, Tensor)) for o, e in output_expr]

    # Separate the T3 and T4 dependnent parts
    output_expr_external = []
    output_expr_mixed = []
    output_expr_internal = []
    for output, expr in output_expr:
        for e in expr.expand()._children:
            names = set(tensor.name for tensor in e.search_leaves(Tensor))
            if names.intersection({"t3", "t4", "t4a"}) and names.intersection({"t1", "t2"}):
                output_expr_mixed.append((output, e))
            elif names.intersection({"t3", "t4", "t4a"}):
                output_expr_external.append((output, e))
            else:
                output_expr_internal.append((output, e))

    # Optimise
    output_expr_external = optimise([o for o, e in output_expr_external], [e for o, e in output_expr_external], strategy="exhaust")
    output_expr_external = [(o, e.apply(lambda tensor: tensor.canonicalise(), Tensor)) for o, e in output_expr_external]
    output_expr_external = combine_expressions(output_expr_external)
    return_external = [r for r in returns if any(o.name == r.name for o, e in output_expr_external)]
    output_expr_mixed = optimise([o for o, e in output_expr_mixed], [e for o, e in output_expr_mixed], strategy="exhaust")
    output_expr_mixed = [(o, e.apply(lambda tensor: tensor.canonicalise(), Tensor)) for o, e in output_expr_mixed]
    output_expr_mixed = combine_expressions(output_expr_mixed)
    return_mixed = [r for r in returns if any(o.name == r.name for o, e in output_expr_mixed)]

    # Generate the external T amplitude code
    for name, codegen in code_generators.items():
        codegen(
            "update_amps_external",
            return_external,
            output_expr_external,
            as_dict=True,
        )

    # Generate the mixed T amplitude code
    for name, codegen in code_generators.items():
        codegen(
            "update_amps_mixed",
            return_mixed,
            output_expr_mixed,
            as_dict=True,
        )

with Stopwatch("C->T conversions"):
    # Get the C->T contractions in albert format
    string_t1 = r"c_{i}^{a}"
    string_t2 = r"c_{ij}^{ab} - t_{i}^{a} t_{j}^{b} + t_{i}^{b} t_{j}^{a}"
    string_t3 = r"c_{ijk}^{abc} - t_{i}^{a} t_{jk}^{bc} + t_{i}^{b} t_{jk}^{ac} - t_{i}^{c} t_{jk}^{ab} + t_{j}^{a} t_{ik}^{bc} - t_{j}^{b} t_{ik}^{ac} + t_{j}^{c} t_{ik}^{ab} - t_{k}^{a} t_{ij}^{bc} + t_{k}^{b} t_{ij}^{ac} - t_{k}^{c} t_{ij}^{ab} - t_{i}^{a} t_{j}^{b} t_{k}^{c} + t_{i}^{a} t_{j}^{c} t_{k}^{b} + t_{i}^{b} t_{j}^{a} t_{k}^{c} - t_{i}^{b} t_{j}^{c} t_{k}^{a} - t_{i}^{c} t_{j}^{a} t_{k}^{b} + t_{i}^{c} t_{j}^{b} t_{k}^{a}"
    string_t4 = r"c_{ijkl}^{abcd} - t_{i}^{a} t_{jkl}^{bcd} + t_{i}^{b} t_{jkl}^{acd} - t_{i}^{c} t_{jkl}^{abd} + t_{i}^{d} t_{jkl}^{abc} + t_{j}^{a} t_{ikl}^{bcd} - t_{j}^{b} t_{ikl}^{acd} + t_{j}^{c} t_{ikl}^{abd} - t_{j}^{d} t_{ikl}^{abc} - t_{k}^{a} t_{ijl}^{bcd} + t_{k}^{b} t_{ijl}^{acd} - t_{k}^{c} t_{ijl}^{abd} + t_{k}^{d} t_{ijl}^{abc} + t_{l}^{a} t_{ijk}^{bcd} - t_{l}^{b} t_{ijk}^{acd} + t_{l}^{c} t_{ijk}^{abd} - t_{l}^{d} t_{ijk}^{abc} - t_{ij}^{ab} t_{kl}^{cd} + t_{ij}^{ac} t_{kl}^{bd} - t_{ij}^{ad} t_{kl}^{bc} - t_{ij}^{bc} t_{kl}^{ad} + t_{ij}^{bd} t_{kl}^{ac} - t_{ij}^{cd} t_{kl}^{ab} + t_{ik}^{ab} t_{jl}^{cd} - t_{ik}^{ac} t_{jl}^{bd} + t_{ik}^{ad} t_{jl}^{bc} + t_{ik}^{bc} t_{jl}^{ad} - t_{ik}^{bd} t_{jl}^{ac} + t_{ik}^{cd} t_{jl}^{ab} - t_{il}^{ab} t_{jk}^{cd} + t_{il}^{ac} t_{jk}^{bd} - t_{il}^{ad} t_{jk}^{bc} - t_{il}^{bc} t_{jk}^{ad} + t_{il}^{bd} t_{jk}^{ac} - t_{il}^{cd} t_{jk}^{ab} - t_{i}^{a} t_{j}^{b} t_{kl}^{cd} + t_{i}^{a} t_{j}^{c} t_{kl}^{bd} - t_{i}^{a} t_{j}^{d} t_{kl}^{bc} + t_{i}^{a} t_{k}^{b} t_{jl}^{cd} - t_{i}^{a} t_{k}^{c} t_{jl}^{bd} + t_{i}^{a} t_{k}^{d} t_{jl}^{bc} - t_{i}^{a} t_{l}^{b} t_{jk}^{cd} + t_{i}^{a} t_{l}^{c} t_{jk}^{bd} - t_{i}^{a} t_{l}^{d} t_{jk}^{bc} + t_{i}^{b} t_{j}^{a} t_{kl}^{cd} - t_{i}^{b} t_{j}^{c} t_{kl}^{ad} + t_{i}^{b} t_{j}^{d} t_{kl}^{ac} - t_{i}^{b} t_{k}^{a} t_{jl}^{cd} + t_{i}^{b} t_{k}^{c} t_{jl}^{ad} - t_{i}^{b} t_{k}^{d} t_{jl}^{ac} + t_{i}^{b} t_{l}^{a} t_{jk}^{cd} - t_{i}^{b} t_{l}^{c} t_{jk}^{ad} + t_{i}^{b} t_{l}^{d} t_{jk}^{ac} - t_{i}^{c} t_{j}^{a} t_{kl}^{bd} + t_{i}^{c} t_{j}^{b} t_{kl}^{ad} - t_{i}^{c} t_{j}^{d} t_{kl}^{ab} + t_{i}^{c} t_{k}^{a} t_{jl}^{bd} - t_{i}^{c} t_{k}^{b} t_{jl}^{ad} + t_{i}^{c} t_{k}^{d} t_{jl}^{ab} - t_{i}^{c} t_{l}^{a} t_{jk}^{bd} + t_{i}^{c} t_{l}^{b} t_{jk}^{ad} - t_{i}^{c} t_{l}^{d} t_{jk}^{ab} + t_{i}^{d} t_{j}^{a} t_{kl}^{bc} - t_{i}^{d} t_{j}^{b} t_{kl}^{ac} + t_{i}^{d} t_{j}^{c} t_{kl}^{ab} - t_{i}^{d} t_{k}^{a} t_{jl}^{bc} + t_{i}^{d} t_{k}^{b} t_{jl}^{ac} - t_{i}^{d} t_{k}^{c} t_{jl}^{ab} + t_{i}^{d} t_{l}^{a} t_{jk}^{bc} - t_{i}^{d} t_{l}^{b} t_{jk}^{ac} + t_{i}^{d} t_{l}^{c} t_{jk}^{ab} - t_{j}^{a} t_{k}^{b} t_{il}^{cd} + t_{j}^{a} t_{k}^{c} t_{il}^{bd} - t_{j}^{a} t_{k}^{d} t_{il}^{bc} + t_{j}^{a} t_{l}^{b} t_{ik}^{cd} - t_{j}^{a} t_{l}^{c} t_{ik}^{bd} + t_{j}^{a} t_{l}^{d} t_{ik}^{bc} + t_{j}^{b} t_{k}^{a} t_{il}^{cd} - t_{j}^{b} t_{k}^{c} t_{il}^{ad} + t_{j}^{b} t_{k}^{d} t_{il}^{ac} - t_{j}^{b} t_{l}^{a} t_{ik}^{cd} + t_{j}^{b} t_{l}^{c} t_{ik}^{ad} - t_{j}^{b} t_{l}^{d} t_{ik}^{ac} - t_{j}^{c} t_{k}^{a} t_{il}^{bd} + t_{j}^{c} t_{k}^{b} t_{il}^{ad} - t_{j}^{c} t_{k}^{d} t_{il}^{ab} + t_{j}^{c} t_{l}^{a} t_{ik}^{bd} - t_{j}^{c} t_{l}^{b} t_{ik}^{ad} + t_{j}^{c} t_{l}^{d} t_{ik}^{ab} + t_{j}^{d} t_{k}^{a} t_{il}^{bc} - t_{j}^{d} t_{k}^{b} t_{il}^{ac} + t_{j}^{d} t_{k}^{c} t_{il}^{ab} - t_{j}^{d} t_{l}^{a} t_{ik}^{bc} + t_{j}^{d} t_{l}^{b} t_{ik}^{ac} - t_{j}^{d} t_{l}^{c} t_{ik}^{ab} - t_{k}^{a} t_{l}^{b} t_{ij}^{cd} + t_{k}^{a} t_{l}^{c} t_{ij}^{bd} - t_{k}^{a} t_{l}^{d} t_{ij}^{bc} + t_{k}^{b} t_{l}^{a} t_{ij}^{cd} - t_{k}^{b} t_{l}^{c} t_{ij}^{ad} + t_{k}^{b} t_{l}^{d} t_{ij}^{ac} - t_{k}^{c} t_{l}^{a} t_{ij}^{bd} + t_{k}^{c} t_{l}^{b} t_{ij}^{ad} - t_{k}^{c} t_{l}^{d} t_{ij}^{ab} + t_{k}^{d} t_{l}^{a} t_{ij}^{bc} - t_{k}^{d} t_{l}^{b} t_{ij}^{ac} + t_{k}^{d} t_{l}^{c} t_{ij}^{ab} - t_{i}^{a} t_{j}^{b} t_{k}^{c} t_{l}^{d} + t_{i}^{a} t_{j}^{b} t_{k}^{d} t_{l}^{c} + t_{i}^{a} t_{j}^{c} t_{k}^{b} t_{l}^{d} - t_{i}^{a} t_{j}^{c} t_{k}^{d} t_{l}^{b} - t_{i}^{a} t_{j}^{d} t_{k}^{b} t_{l}^{c} + t_{i}^{a} t_{j}^{d} t_{k}^{c} t_{l}^{b} + t_{i}^{b} t_{j}^{a} t_{k}^{c} t_{l}^{d} - t_{i}^{b} t_{j}^{a} t_{k}^{d} t_{l}^{c} - t_{i}^{b} t_{j}^{c} t_{k}^{a} t_{l}^{d} + t_{i}^{b} t_{j}^{c} t_{k}^{d} t_{l}^{a} + t_{i}^{b} t_{j}^{d} t_{k}^{a} t_{l}^{c} - t_{i}^{b} t_{j}^{d} t_{k}^{c} t_{l}^{a} - t_{i}^{c} t_{j}^{a} t_{k}^{b} t_{l}^{d} + t_{i}^{c} t_{j}^{a} t_{k}^{d} t_{l}^{b} + t_{i}^{c} t_{j}^{b} t_{k}^{a} t_{l}^{d} - t_{i}^{c} t_{j}^{b} t_{k}^{d} t_{l}^{a} - t_{i}^{c} t_{j}^{d} t_{k}^{a} t_{l}^{b} + t_{i}^{c} t_{j}^{d} t_{k}^{b} t_{l}^{a} + t_{i}^{d} t_{j}^{a} t_{k}^{b} t_{l}^{c} - t_{i}^{d} t_{j}^{a} t_{k}^{c} t_{l}^{b} - t_{i}^{d} t_{j}^{b} t_{k}^{a} t_{l}^{c} + t_{i}^{d} t_{j}^{b} t_{k}^{c} t_{l}^{a} + t_{i}^{d} t_{j}^{c} t_{k}^{a} t_{l}^{b} - t_{i}^{d} t_{j}^{c} t_{k}^{b} t_{l}^{a}"
    for n, string in enumerate([string_t1, string_t2, string_t3, string_t4]):
        output = []
        expr = []
        returns = []
        for index_spins in get_amplitude_spins("ijkl"[: n + 1], "abcd"[: n + 1], spin):
            signs = [1] + [1 if s == "+" else -1 for s in string if s in "+-"]
            terms = [s.strip() for s in string.replace("-", "+").split("+")]
            expr_n = Scalar(0.0)
            for sign, term in zip(signs, terms):
                part = Scalar(sign)
                for tensor in term.split():
                    order = sum(1 for c in tensor if c in "ijkl")
                    name = f"{tensor[0]}{order}"
                    occ = [Index(i, space="o", spin=index_spins.get(i, None)) for i in tensor[tensor.index("_") + 2 : tensor.index("_") + 2 + order]]
                    vir = [Index(a, space="v", spin=index_spins.get(a, None)) for a in tensor[tensor.index("^") + 2 : tensor.index("^") + 2 + order]]
                    part *= getattr(ghf, f"T{order}")(*occ, *vir, name=name)
                expr_n += part

            expr_n = spin_integrate(expr_n, spin)
            if spin == "rhf" and (n + 1) == 4 and len([s for s in index_spins.values() if s == "a"]) == 6:
                name = "t4a"
            else:
                name = f"t{n + 1}"
            output_n = [Tensor(*tuple(sorted(e.external_indices, key=lambda i: "ijklabcd".index(i.name))), name=name) for e in expr_n]
            returns_n = (output_n[0],)

            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)

        if spin == "uhf":
            # Why do we have to refactor these amplitudes? Is this a deficiency in albert?
            def _refactor(tensor):
                extra_factor = 1.0
                if type(tensor) in (uhf.T1, uhf.T2, uhf.T3, uhf.T4):
                    spin = tuple(i.spin for i in tensor.external_indices[: tensor.rank // 2])
                    extra_factor /= factorial(sum(s == "a" for s in spin))
                    extra_factor /= factorial(sum(s == "b" for s in spin))
                return tensor * Scalar(extra_factor)
            expr = [e.apply(_refactor, Tensor) for e in expr]

        # Optimise and generate the code separately for each amplitude
        output_expr = optimise(output, expr, strategy="exhaust" if n < 3 else "trav")
        output_expr = [(o, e.apply(lambda tensor: tensor.canonicalise(), Tensor)) for o, e in output_expr]

        for name, codegen in code_generators.items():
            codegen(
                f"convert_c{n + 1}_to_t{n + 1}",
                returns,
                output_expr,
                as_dict=True,
            )

with Stopwatch("T->C conversions"):
    # Get the T amplitudes in albert format
    eT = [[1.0, ["1"]]]
    for n in range(4):
        factor = 1 / factorial(n + 1)
        for ts in itertools.product(["t1", "t2", "t3", "t4"], repeat=n + 1):
            eT.append([factor, list(ts)])

    for n in range(4):
        # Get the T->C contractions in pdaggerq format
        pq.clear()
        pq.set_left_operators([[f"e{n + 1}({','.join('ijkl'[:n + 1]+''.join(reversed('abcd'[:n + 1])))})"]])
        pq.set_right_operators([["1"]])
        for term in eT:
            pq.add_operator_product(term[0], term[1])
        pq.simplify()
        terms = pq.strings()

        ## Get the T->C contractions in albert format
        #output_expr, returns = get_amplitudes([terms], spin, strategy="exhaust" if (n, spin) != (4, "uhf") else "greedy", orders=[n + 1])

        output = []
        expr = []
        returns = []
        for index_spins in get_amplitude_spins("ijkl"[: n + 1], "abcd"[: n + 1], spin):
            expr_n = import_from_pdaggerq(terms, index_spins=index_spins)
            expr_n = spin_integrate(expr_n, spin)
            if spin == "rhf" and (n + 1) == 4 and len([s for s in index_spins.values() if s == "a"]) == 6:
                name = "c4a"
            else:
                name = f"c{n + 1}"
            output_n = [Tensor(*tuple(sorted(e.external_indices, key=lambda i: "ijklabcd".index(i.name))), name=name) for e in expr_n]
            returns_n = (output_n[0],)
            expr.extend(expr_n)
            output.extend(output_n)
            returns.extend(returns_n)

        # Optimise and generate the code separately for each amplitude
        output_expr = optimise(output, expr, strategy="exhaust" if n < 3 else "trav")
        output_expr = [(o, e.apply(lambda tensor: tensor.canonicalise(), Tensor)) for o, e in output_expr]

        for name, codegen in code_generators.items():
            codegen(
                f"convert_t{n + 1}_to_c{n + 1}",
                returns,
                output_expr,
                as_dict=True,
            )

for codegen in code_generators.values():
    codegen.postamble()
    codegen.stdout.close()
