"""
Generate the CCSD(T) code.
"""

import sys
import types
sys.setrecursionlimit(100000)

import pdaggerq
from albert.qc._pdaggerq import import_from_pdaggerq
from albert.tensor import Tensor

from ebcc.codegen.bootstrap_common import *

# Get the spin case
spin = sys.argv[1]

# Set up the code generators
def name_generator_pert_t3(tensor, add_spaces=True):
    name = name_generators[spin](tensor, add_spaces=add_spaces)
    if isinstance(tensor, Tensor) and tensor.name in ("f", "v", "t1", "t2", "l1", "l2"):
        if add_spaces:
            indices = tuple(f"[{i.name}]" if i.name in "ijk" else ":" for i in tensor.indices)
            indices = tuple(f"[{', '.join([':'] * n)}{', ' if n else ''}{i}]" for n, i in enumerate(indices))
            indices = "".join(indices)
            name = f"{name}{indices}"
    return name

# Set up the code generators
file = open(f"{spin[0].upper()}CCSDxTx.py", "w")
code_generators_pert_t3 = {
    "einsum": EinsumCodeGen(
        stdout=file,
        name_generator=name_generator_pert_t3,
        spin=spin,
    ),
}
code_generators = {
    "einsum": EinsumCodeGen(
        stdout=file,
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
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    terms = pq.fully_contracted_strings()
    terms = remove_hf_energy(terms)

    # Get the energy in albert format
    expr = import_from_pdaggerq(terms)
    expr = spin_integrate(expr, spin)
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
    # Get the T1 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e1(i,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    terms_t1 = pq.fully_contracted_strings()

    # Get the T2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e2(i,j,b,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    terms_t2 = pq.fully_contracted_strings()

    # Get the T amplitudes in albert format
    terms = [terms_t1, terms_t2]
    expr = []
    output = []
    returns = []
    for n in range(2):
        for index_spins in get_amplitude_spins(n + 1, spin):
            indices = default_indices["o"][: n + 1] + default_indices["v"][: n + 1]
            expr_n = import_from_pdaggerq(terms[n], index_spins=index_spins)
            expr_n = spin_integrate(expr_n, spin)
            output_n = get_t_amplitude_outputs(expr_n, f"t{n+1}new", indices=indices)
            returns_n = (Tensor(*tuple(Index(i, index_spins[i]) for i in indices), name=f"t{n+1}new"),)
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

with Stopwatch("Perturbative energy"):
    # Get the T3 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e3(i,j,k,c,b,a)"]])
    pq.add_commutator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms = pq.fully_contracted_strings()

    # Get the T amplitudes in albert format
    expr = []
    output = []
    returns = []
    for index_spins in get_amplitude_spins(3, spin):
        indices = default_indices["o"][:3] + default_indices["v"][:3]
        expr_n = import_from_pdaggerq(terms, index_spins=index_spins)
        expr_n = spin_integrate(expr_n, spin)
        output_n = get_t_amplitude_outputs(expr_n, "t3", indices=indices)
        returns_n = (Tensor(*tuple(Index(i, index_spins[i]) for i in indices), name="t3"),)
        expr.extend(expr_n)
        output.extend(output_n)
        returns.extend(returns_n)
    output, expr = optimise(output, expr, spin, strategy="exhaust")

    # Generate the T amplitude code
    for name, codegen in code_generators_pert_t3.items():
        if name == "einsum":
            kwargs = {
                "preamble": "t3 = Namespace()" if spin == "uhf" else None,
            }
            if spin != "uhf":
                kwargs["postamble"] = "\n"
                kwargs["postamble"] += "eo = np.diag(f.oo)\nev = np.diag(f.vv)\n"
                kwargs["postamble"] += "t3 /= direct_sum(\"i,j,k,a,b,c->ijkabc\", eo[[i]], eo[[j]], eo[[k]], -ev, -ev, -ev)"
            else:
                raise NotImplementedError
        else:
            kwargs = {}
        _function_declaration = codegen.function_declaration
        codegen.function_declaration = lambda name, args: _function_declaration(name, tuple(args) + ("f", "i", "j", "k"))
        codegen(
            "_build_t3_ijk",
            returns,
            output,
            expr,
            **kwargs,
        )

    # Get the energy in contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["l1"], ["l2"]])
    pq.add_commutator(1.0, ["v"], ["t3"])
    pq.simplify()
    terms = pq.fully_contracted_strings()

    # Get the energy in albert format
    expr = import_from_pdaggerq(terms)
    expr = spin_integrate(expr, spin)
    output = tuple(Tensor(name="e_pert") for _ in range(len(expr)))
    output, expr = optimise(output, expr, spin, strategy="exhaust")
    returns = (Tensor(name="e_pert"),)

    # Generate the energy code
    for codegen in code_generators_pert_t3.values():
        codegen(
            "_energy_perturbative_ijk",
            returns,
            output,
            expr,
        )

    # Generate the driver code
    for name, codegen in code_generators.items():
        if name == "einsum":
            args = ("f", "v", "t2", "l1", "l2")
            codegen.function_declaration("energy_perturbative", args)
            codegen.indent()
            metadata = codegen.get_metadata()
            parameters_str = "\n".join([f"{arg} : array" for arg in args])
            returns_str = "e_pert : array"
            docstring = f"Code generated by `albert` {metadata['albert_version']} on {metadata['date']}.\n"
            docstring += "\n"
            docstring += "Parameters\n----------\n"
            docstring += parameters_str
            docstring += "\n\n"
            docstring += "Returns\n-------\n"
            docstring += returns_str
            codegen.function_docstring(docstring)
            codegen.blank()
            if spin != "uhf":
                codegen.write("nocc = t2.shape[0]")
                codegen.write("e_pert = 0.0")
                codegen.write("for i in range(nocc):")
                codegen.indent()
                codegen.write("for j in range(nocc):")
                codegen.indent()
                codegen.write("for k in range(nocc):")
                codegen.indent()
                codegen.write("t3 = _build_t3_ijk(f=f, v=v, t2=t2, i=i, j=j, k=k)")
                codegen.write("e_pert += _energy_perturbative_ijk(f=f, v=v, t2=t2, t3=t3, l1=l1, l2=l2, i=i, j=j, k=k)")
            else:
                raise NotImplementedError
            codegen.dedent()
            codegen.dedent()
            codegen.dedent()
            codegen.blank()
            codegen.function_return(("e_pert",))
            codegen.dedent()
            codegen.blank()


for codegen in code_generators.values():
    codegen.postamble()
    codegen.stdout.close()
