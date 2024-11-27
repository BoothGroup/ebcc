"""
Generate the CCSD(T) code.
"""

from numbers import Number
import itertools
import re
import sys
import types

import pdaggerq
from albert.qc._pdaggerq import import_from_pdaggerq
from albert.tensor import Tensor
from albert.algebra import Algebraic
from albert.codegen.einsum import _parse_indices

from ebcc.codegen.bootstrap_common import *

# Get the spin case
spin = sys.argv[1]

# Set up the code generators
code_generators = {
    "einsum": EinsumCodeGen(
        stdout=open(f"{spin[0].upper()}CCSDxTx.py", "w"),
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

# Function for writing T3 slices
def algebraic_expression_t3(self, output, expr, already_declared=False):
    """Write an algebraic expression."""

    assert isinstance(output, Tensor)
    assert isinstance(expr, (Tensor, Algebraic))

    for i, mul_args in enumerate(expr.nested_view()):
        # Separate the factors and tensors
        factors = [arg for arg in mul_args if isinstance(arg, Number)]
        tensors = [arg for arg in mul_args if isinstance(arg, Tensor)]

        # Get the indices
        lhs = [arg.indices for arg in tensors]
        rhs = output.indices
        indices = _parse_indices(lhs, rhs)
        to_slice = [i.name for i in output.indices[:3]]

        # Get the arguments
        args = []
        for tensor, index in zip(tensors, indices):
            name = self.get_name(tensor).lower()
            inds = repr(index)
            if len(tensors) == 2:
                slices = [f"{i.name.lower()}0:{i.name.lower()}1" if i.space == i.space.upper() else ":" for i in tensor.indices]
                name += f"[{', '.join(slices)}]"
            args.append(name)
            args.append(inds)
        args.append(repr(indices[-1]))

        # Get the operator and LHS
        output_name = self.get_name(output)
        if len(tensors) == 1:
            slices = [f"{i.name.lower()}0:{i.name.lower()}1" if i.space == i.space.upper() else ":" for i in output.indices]
            output_name += f"[{', '.join(slices)}]"
        operator = "=" if output_name not in self._declared else "+="
        self._declared.add(output_name)

        # Get the factor
        factor = 1
        for f in factors:
            factor *= f
        if abs(factor - round(factor)) < 1e-12:
            factor = int(round(factor))
        factor = f" * {factor}" if factor != 1 else ""

        # Write the expression
        if len(tensors) > 1:
            if self.einsum_kwargs:
                kwargs = ", " + ", ".join(f"{k}={v}" for k, v in self.einsum_kwargs.items())
            else:
                kwargs = ""
            args = ", ".join(args)
            self.write(f"{output_name} {operator} {self.einsum_func}({args}{kwargs}){factor}")
        else:
            transpose = tuple(indices[0].index(i) for i in indices[1])
            if transpose != tuple(range(len(transpose))):
                targ = self.transpose_func.format(arg=args[0], transpose=transpose)
            else:
                targ = args[0]
            copy = ".copy()" if i == 0 and not already_declared else ""
            self.write(f"{output_name} {operator} {targ}{copy}{factor}")

# Function for writing e_pert slices
def algebraic_expression_e(self, output, expr, already_declared=False):
    """Write an algebraic expression."""

    assert isinstance(output, Tensor)
    assert isinstance(expr, (Tensor, Algebraic))

    for i, mul_args in enumerate(expr.nested_view()):
        # Separate the factors and tensors
        factors = [arg for arg in mul_args if isinstance(arg, Number)]
        tensors = [arg for arg in mul_args if isinstance(arg, Tensor)]

        # Get the indices
        lhs = [arg.indices for arg in tensors]
        rhs = output.indices
        indices = _parse_indices(lhs, rhs)

        # Get the sliced indices
        sliced_chars = None
        sliced_inds = None
        for tensor, index in zip(tensors, indices):
            if tensor.name in self._slice_cache:
                sliced_inds = index
                sliced_chars = self._slice_cache[tensor.name]
                break

        # Get the arguments
        args = []
        for tensor, index in zip(tensors, indices):
            inds = repr(index)
            name = self.get_name(tensor)
            chars = [sliced_chars[sliced_inds.index(i)] if i in sliced_inds else None for i in index]
            if name != "t3":
                name += f"[{', '.join(':' if c is None else f'{c}0:{c}1' for c in chars)}]"
            args.append(name)
            args.append(inds)
        args.append(repr(indices[-1]))

        # Get the operator and LHS
        output_name = self.get_name(output)
        operator = "=" if output_name not in self._declared else "+="
        self._declared.add(output_name)
        chars = [sliced_chars[sliced_inds.index(i)] if i in sliced_inds else None for i in indices[-1]]
        self._slice_cache[self.get_name(output)] = chars

        # Get the factor
        factor = 1
        for f in factors:
            factor *= f
        if abs(factor - round(factor)) < 1e-12:
            factor = int(round(factor))
        factor = f" * {factor}" if factor != 1 else ""

        # Write the expression
        if len(tensors) > 1:
            if self.einsum_kwargs:
                kwargs = ", " + ", ".join(f"{k}={v}" for k, v in self.einsum_kwargs.items())
            else:
                kwargs = ""
            args = ", ".join(args)
            self.write(f"{output_name} {operator} {self.einsum_func}({args}{kwargs}){factor}")
        else:
            transpose = tuple(indices[0].index(i) for i in indices[1])
            if transpose != tuple(range(len(transpose))):
                targ = self.transpose_func.format(arg=args[0], transpose=transpose)
            else:
                targ = args[0]
            copy = ".copy()" if i == 0 and not already_declared else ""
            self.write(f"{output_name} {operator} {targ}{copy}{factor}")

with Stopwatch("Energy"):
    # Get the energy contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["1"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    terms = pq.strings()
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
    terms_t1 = pq.strings()

    # Get the T2 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e2(i,j,b,a)"]])
    pq.add_st_operator(1.0, ["f"], ["t1", "t2"])
    pq.add_st_operator(1.0, ["v"], ["t1", "t2"])
    pq.simplify()
    terms_t2 = pq.strings()

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
            output_n = get_t_amplitude_outputs(expr_n, f"t{n+1}new")
            returns_n = (Tensor(*indices, name=f"t{n+1}new"),)
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

with Stopwatch("Perturbative triples"):
    # Get the T3 contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["e3(i,j,k,c,b,a)"]])
    pq.add_commutator(1.0, ["v"], ["t2"])
    pq.simplify()
    terms = pq.strings()

    # Get the T3 amplitudes in albert format
    expr = []
    output = []
    returns = []
    for index_spins in get_amplitude_spins(3, spin):
        index_spaces = {c: "O" for c in "IJK"}  # Hacky, use the active space indices for the slices
        index_spaces.update({c: "v" for c in "abc"})
        indices = [x.upper() for x in default_indices["o"][:3]] + default_indices["v"][:3]
        expr_n = import_from_pdaggerq(terms, index_spins=index_spins, index_spaces=index_spaces)
        expr_n = expr_n.map_indices({i: Index(i.name.upper(), space="O", spin=i.spin) for i in expr_n.external_indices if i.name in "ijk"})
        expr_n = spin_integrate(expr_n, spin)
        output_n = get_t_amplitude_outputs(expr_n, f"t3", indices=indices)
        output_n = [o.map_indices({i: i.to_space(index_spaces.get(i.name, None)) for i in o.indices}) for o in output_n]
        returns_n = (Tensor(*indices, name=f"t3"),)
        output_n, expr_n = optimise(output_n, expr_n, spin, strategy="exhaust")
        expr.extend(expr_n)
        output.extend(output_n)
        returns.extend(returns_n)
    output, expr = optimise(output, expr, spin, strategy="exhaust")

    # Generate the T3 amplitude code
    for name, codegen in code_generators.items():
        preamble = ""
        if name == "einsum":
            _algebraic_expression = codegen.algebraic_expression
            codegen.algebraic_expression = types.MethodType(algebraic_expression_t3, codegen)
            if spin in ("rhf", "ghf"):
                preamble += "t3 = Namespace()\n"
                preamble += "i0 = kwargs[\"i0\"]\n"
                preamble += "j0 = kwargs[\"j0\"]\n"
                preamble += "k0 = kwargs[\"k0\"]\n"
                preamble += "i1 = kwargs[\"i1\"]\n"
                preamble += "j1 = kwargs[\"j1\"]\n"
                preamble += "k1 = kwargs[\"k1\"]\n"
            else:
                1/0
            codegen._declared = set()
        codegen(
            "_get_t3",
            returns,
            output,
            expr,
            preamble=preamble,
        )
        if name == "einsum":
            codegen.algebraic_expression = _algebraic_expression
            del codegen._declared

    # Get the energy contractions in pdaggerq format
    pq.clear()
    pq.set_left_operators([["l1"], ["l2"]])
    pq.add_commutator(1.0, ["v"], ["t3"])
    pq.simplify()
    terms = pq.strings()

    # Get the energy in albert format
    expr = import_from_pdaggerq(terms)
    expr = spin_integrate(expr, spin)
    output = tuple(Tensor(name="e_pert") for _ in range(len(expr)))
    output, expr = optimise(output, expr, spin, strategy="exhaust")
    returns = (Tensor(name="e_pert"),)

    # Insert the slices
    new_output = []
    new_expr = []

    # Generate the energy code
    for codegen in code_generators.values():
        if name == "einsum":
            _algebraic_expression = codegen.algebraic_expression
            codegen._slice_cache = {"t3": ["i", "j", "k", None, None, None]}
            codegen.algebraic_expression = types.MethodType(algebraic_expression_e, codegen)
            preamble = ""
            if spin in ("rhf", "ghf"):
                preamble += "i0 = kwargs[\"i0\"]\n"
                preamble += "j0 = kwargs[\"j0\"]\n"
                preamble += "k0 = kwargs[\"k0\"]\n"
                preamble += "i1 = kwargs[\"i1\"]\n"
                preamble += "j1 = kwargs[\"j1\"]\n"
                preamble += "k1 = kwargs[\"k1\"]\n"
            else:
                1/0
            codegen._declared = set()
        codegen(
            "_et_block",
            returns,
            output,
            expr,
            preamble=preamble,
        )
        if name == "einsum":
            codegen.algebraic_expression = _algebraic_expression
            del codegen._slice_cache
            del codegen._declared

    for codegen in code_generators.values():
        if name == "einsum":
            # Generate the energy function
            codegen.function_declaration("energy_perturbative", ["f", "l1", "l2", "t2", "v"])
            codegen.indent()

            # Write the function docstring
            metadata = codegen.get_metadata()
            docstring = f"Code generated by `albert` {metadata['albert_version']} on {metadata['date']}.\n"
            docstring += "\n"
            docstring += "Parameters\n----------\n"
            docstring += "f : array\n"
            docstring += "l1 : array\n"
            docstring += "l2 : array\n"
            docstring += "l2 : array\n"
            docstring += "v : array\n"
            docstring += "\n"
            docstring += "Returns\n-------\n"
            docstring += "e_pert : float"
            codegen.function_docstring(docstring)
            codegen.blank()
            code = ""
            code += "ei = f.oo.diagonal()\n"
            code += "ea = f.vv.diagonal()\n"
            code += "eabc = direct_sum(\"a,b,c->abc\", -ea, -ea, -ea)\n"
            code += "block_size = kwargs.get(\"block_size\", 32)\n"
            code += "\n"
            code += "e_pert = 0.0\n"
            code += "for i0 in range(l1.shape[1]):\n"
            code += "    for j0 in range(l1.shape[1]):\n"
            code += "        for k0 in range(0, l1.shape[1], block_size):\n"
            code += "            i1 = i0 + 1\n"
            code += "            j1 = j0 + 1\n"
            code += "            k1 = min(k0 + block_size, l1.shape[1])\n"
            code += "            t3 = _get_t3(l1=l1, l2=l2, t2=t2, v=v, i0=i0, j0=j0, k0=k0, i1=i1, j1=j1, k1=k1)\n"
            for i, j, k in itertools.permutations("ijk"):
                code += f"            if hasattr(t3, \"{i}{j}{k}\"):\n"
                code += f"                t3_block = t3.{i}{j}{k}\n"
                code += f"                t3_block /= direct_sum(\"i,j,k,abc->ijkabc\", ei[{i}0:{i}1], ei[{j}0:{j}1], ei[{k}0:{k}1], eabc)\n"
                code += f"                e_pert += _et_block(l1=l1, l2=l2, t2=t2, t3=t3_block, v=v, i0={i}0, j0={j}0, k0={k}0, i1={i}1, j1={j}1, k1={k}1)\n"
                code += f"                del t3_block\n"
            codegen.write(code)
            codegen.function_return(["e_pert"])
            codegen.dedent()
            codegen.blank()


for codegen in code_generators.values():
    codegen.postamble()
    codegen.stdout.close()
