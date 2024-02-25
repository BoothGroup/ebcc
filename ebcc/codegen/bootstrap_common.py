"""Common functionality for the bootstrap scripts for `ebcc`.
"""

import itertools

from pdaggerq.config import OCC_INDICES, VIRT_INDICES

from albert.tensor import Tensor
from albert.codegen.einsum import EinsumCodeGen as _EinsumCodeGen
from albert.qc.spin import generalised_to_unrestricted, generalised_to_restricted
from albert.qc.uhf import SpinIndex
from albert.optim._gristmill import optimise as _optimise


ov_2e = ["oooo", "ooov", "oovo", "ovoo", "vooo", "oovv", "ovov", "ovvo", "voov", "vovo", "vvoo", "ovvv", "vovv", "vvov", "vvvo", "vvvv"]
ov_1e = ["oo", "ov", "vo", "vv"]


default_indices = {
    "o": OCC_INDICES,
    "v": VIRT_INDICES,
}


default_sectors = {i: k for k, v in default_indices.items() for i in v}


default_sizes = {
    "o": 4,
    "v": 20,
}


class EinsumCodeGen(_EinsumCodeGen):
    """Code generator for the bootstrap scripts for `ebcc`.
    """

    def preamble(self):
        preamble = "from ebcc import numpy as np\n"
        preamble += "from ebcc.util import pack_2e, einsum, Namespace"
        super().preamble(preamble)

    def ignore_argument(self, arg):
        """
        Return `True` if a potential function argument should be
        ignored.
        """
        return "tmp" in arg.name or arg.name == "δ"


def name_generator_rhf(tensor, add_spaces=True):
    """Generate names for the RHF case.
    """
    if tensor.name in ("f", "v", "d", "Γ", "δ"):
        if tensor.name == "d":
            name = "rdm1"
        elif tensor.name == "Γ":
            name = "rdm2"
        elif tensor.name == "δ":
            name = "delta"
        else:
            name = tensor.name
        if add_spaces:
            spaces = [default_sectors[i] for i in tensor.indices]
            return f"{name}.{''.join(spaces)}"
        else:
            return name
    else:
        return tensor.name


def name_generator_uhf(tensor, add_spaces=True):
    """Generate names for the UHF case.
    """
    if tensor.name in ("f", "v", "d", "Γ", "δ"):
        if tensor.name == "d":
            name = "rdm1"
        elif tensor.name == "Γ":
            name = "rdm2"
        elif tensor.name == "δ":
            name = "delta"
        else:
            name = tensor.name
        if add_spaces:
            spins = ["a" if i.spin == "α" else "b" for i in tensor.indices]
            spaces = [default_sectors[i.index] for i in tensor.indices]
            return f"{name}.{''.join(spins)}.{''.join(spaces)}"
        else:
            return name
    elif tensor.name in ("t1", "t2", "t1new", "t2new", "l1", "l2", "l1new", "l2new"):
        if add_spaces:
            spins = ["a" if i.spin == "α" else "b" for i in tensor.indices]
            return f"{tensor.name}.{''.join(spins)}"
        else:
            return tensor.name
    else:
        return tensor.name


name_generators = {
    "rhf": name_generator_rhf,
    "uhf": name_generator_uhf,
    "ghf": name_generator_rhf,
}


def spin_integrate(expr, spin):
    """Perform the spin integration.
    """
    if spin == "rhf":
        return (generalised_to_restricted(expr),)
    elif spin == "uhf":
        return generalised_to_unrestricted(expr)
    else:
        return (expr,)


def remove_hf_energy(terms):
    """Remove the HF energy from the terms.
    """
    terms = [term for term in terms if set(term) != {"+1.00000000000000", "f(i,i)"}]
    terms = [term for term in terms if set(term) != {"-0.50000000000000", "<j,i||j,i>"}]
    return terms


def optimise(outputs, exprs, spin, strategy="greedy", sizes=None):
    """Optimise the expressions.
    """

    if sizes is None:
        sizes = default_sizes

    if spin in ("rhf", "ghf"):
        index_sizes = {}
        for sector, indices in default_indices.items():
            index_sizes.update({index: sizes[sector] for index in indices})
        index_groups = list(default_indices.values())
    else:
        index_sizes = {}
        for sector, indices in default_indices.items():
            for s in ("α", "β"):
                index_sizes.update({SpinIndex(index, s): sizes[sector] for index in indices})
        index_groups = list(default_indices.values())
        index_groups = [
            *[SpinIndex(index, "α") for index in indices],
            *[SpinIndex(index, "β") for index in indices],
        ]

    opt = _optimise(
        *zip(outputs, exprs),
        index_groups=index_groups,
        sizes=index_sizes,
        strategy=strategy,
    )

    return zip(*opt)


def get_t_amplitude_outputs(exprs, name):
    """Get the outputs for the T amplitude code.
    """

    def index_sort(x):
        if not isinstance(x, SpinIndex):
            return (default_indices["o"] + default_indices["v"]).index(x)
        else:
            return (x.index in default_indices["v"], x.spin, x.index)

    return [Tensor(*sorted(e.external_indices, key=index_sort), name=name) for e in exprs]


def get_l_amplitude_outputs(exprs, name):
    """Get the outputs for the L amplitude code.
    """

    def index_sort(x):
        if not isinstance(x, SpinIndex):
            return (default_indices["v"] + default_indices["o"]).index(x)
        else:
            return (x.index in default_indices["o"], x.spin, x.index)

    return [Tensor(*sorted(e.external_indices, key=index_sort), name=name) for e in exprs]


def get_density_outputs(exprs, name, indices):
    """Get the outputs for the density code.
    """
    to_index = lambda i: i if not isinstance(i, SpinIndex) else i.index
    tensors = []
    for expr in exprs:
        external_indices = sorted(expr.external_indices, key=lambda i: indices.index(to_index(i)))
        tensors.append(Tensor(*external_indices, name=name))
    return tensors


def get_amplitude_spins(n, spin):
    """Get the spin cases for the amplitudes.
    """

    if spin == "rhf":
        case = {}
        for i in range(n):
            case[default_indices["o"][i]] = ["α", "β"][i % 2]
            case[default_indices["v"][i]] = ["α", "β"][i % 2]
        cases = [case]
    elif spin == "uhf":
        cases = []
        for spins in sorted(set([set(sorted(x)) for x in itertools.permutations(["α", "β"], n)])):
            case = {}
            for i, s in enumerate(spins):
                case[default_indices["o"][i]] = s
                case[default_indices["v"][i]] = s
            cases.append(case)
    elif spin == "ghf":
        cases = [None]

    return cases


def get_density_spins(n, spin):
    """Get the spin cases for the density.
    """

    assert n in (1, 2)  # hardcoded, TODO

    if spin == "rhf":
        if n == 1:
            cases = [("α", "α")]
        elif n == 2:
            cases = [("α", "α", "α", "α"), ("α", "β", "α", "β"), ("β", "β", "β", "β")]
    elif spin == "uhf":
        if n == 1:
            cases = [("α", "α"), ("β", "β")]
        elif n == 2:
            cases = [("α", "α", "α", "α"), ("α", "β", "α", "β"), ("β", "α", "β", "α"), ("β", "β", "β", "β")]
    elif spin == "ghf":
        cases = [None]

    return cases
