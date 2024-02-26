"""Common functionality for the bootstrap scripts for `ebcc`.
"""

import itertools
import time

from albert.codegen.einsum import EinsumCodeGen as _EinsumCodeGen
from albert.optim._gristmill import optimise as _optimise
from albert.qc.spin import generalised_to_restricted, generalised_to_unrestricted
from albert.qc.uhf import SpinIndex
from albert.tensor import Tensor
from pdaggerq.config import OCC_INDICES, VIRT_INDICES

ov_2e = [
    "oooo",
    "ooov",
    "oovo",
    "ovoo",
    "vooo",
    "oovv",
    "ovov",
    "ovvo",
    "voov",
    "vovo",
    "vvoo",
    "ovvv",
    "vovv",
    "vvov",
    "vvvo",
    "vvvv",
]
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
    """Code generator for the bootstrap scripts for `ebcc`."""

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


class Stopwatch:
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.elapsed = self.end - self.start
        if self.name:
            print(f"{self.name}: {self.elapsed:.2f}s")


def name_generator_rhf(tensor, add_spaces=True):
    """Generate names for the RHF case."""
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
    """Generate names for the UHF case."""
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
    """Perform the spin integration."""
    if spin == "rhf":
        return (generalised_to_restricted(expr),)
    elif spin == "uhf":
        return generalised_to_unrestricted(expr)
    else:
        return (expr,)


def remove_hf_energy(terms):
    """Remove the HF energy from the terms."""
    terms = [term for term in terms if set(term) != {"+1.00000000000000", "f(i,i)"}]
    terms = [term for term in terms if set(term) != {"-0.50000000000000", "<j,i||j,i>"}]
    return terms


def optimise(outputs, exprs, spin, strategy="greedy", sizes=None):
    """Optimise the expressions."""

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
        index_groups = []
        for indices in default_indices.values():
            for s in ("α", "β"):
                index_groups.append([SpinIndex(index, s) for index in indices])

    opt = _optimise(
        *zip(outputs, exprs),
        index_groups=index_groups,
        sizes=index_sizes,
        strategy=strategy,
    )

    return zip(*opt)


def get_t_amplitude_outputs(exprs, name):
    """Get the outputs for the T amplitude code."""

    def index_sort(x):
        if not isinstance(x, SpinIndex):
            return (default_indices["o"] + default_indices["v"]).index(x)
        else:
            return (x.index in default_indices["v"], x.spin, x.index)

    return [Tensor(*sorted(e.external_indices, key=index_sort), name=name) for e in exprs]


def get_l_amplitude_outputs(exprs, name):
    """Get the outputs for the L amplitude code."""

    def index_sort(x):
        if not isinstance(x, SpinIndex):
            return (default_indices["v"] + default_indices["o"]).index(x)
        else:
            return (x.index in default_indices["o"], x.spin, x.index)

    return [Tensor(*sorted(e.external_indices, key=index_sort), name=name) for e in exprs]


def get_density_outputs(exprs, name, indices):
    """Get the outputs for the density code."""
    to_index = lambda i: i if not isinstance(i, SpinIndex) else i.index
    tensors = []
    for expr in exprs:
        external_indices = sorted(expr.external_indices, key=lambda i: indices.index(to_index(i)))
        tensors.append(Tensor(*external_indices, name=name))
    return tensors


def get_amplitude_spins(n, spin):
    """Get the spin cases for the amplitudes."""

    if spin == "rhf":
        case = {}
        for i in range(n):
            case[default_indices["o"][i]] = ["α", "β"][i % 2]
            case[default_indices["v"][i]] = ["α", "β"][i % 2]
        cases = [case]
    elif spin == "uhf":
        cases = []
        for spins in itertools.combinations_with_replacement(["α", "β"], n):
            case = {}
            for i, s in enumerate(spins):
                case[default_indices["o"][i]] = s
                case[default_indices["v"][i]] = s
            cases.append(case)
    elif spin == "ghf":
        cases = [None]

    return cases


def get_density_spins(n, spin, indices):
    """Get the spin cases for the density."""

    assert n in (1, 2)  # hardcoded, TODO

    if spin == "rhf":
        if n == 1:
            cases = [("α", "α")]
        elif n == 2:
            cases = [
                ("α", "α", "α", "α"),
                ("α", "β", "α", "β"),
                ("β", "α", "β", "α"),
                ("β", "β", "β", "β"),
            ]
        cases = [dict(zip(indices, case)) for case in cases]
    elif spin == "uhf":
        if n == 1:
            cases = [("α", "α"), ("β", "β")]
        elif n == 2:
            cases = [("α", "α", "α", "α"), ("α", "β", "α", "β"), ("β", "β", "β", "β")]
        cases = [dict(zip(indices, case)) for case in cases]
    elif spin == "ghf":
        cases = [None]

    return cases


def get_density_einsum_preamble(n, spin):
    """Get the einsum preamble for the density."""
    preamble = "rdm1 = Namespace()"
    if spin == "uhf":
        for spins in itertools.combinations_with_replacement(["α", "β"], n):
            preamble += f"\nrdm1.{''.join(spins)} = Namespace()"
        preamble += "\ndelta = Namespace("
        preamble += "\n    aa=Namespace(oo=np.eye(t1.aa.shape[0]), vv=np.eye(t1.aa.shape[1])),"
        preamble += "\n    bb=Namespace(oo=np.eye(t1.bb.shape[0]), vv=np.eye(t1.bb.shape[1])),"
        preamble += "\n)"
    else:
        preamble += "\ndelta = Namespace("
        preamble += "\n    oo=np.eye(t1.shape[0]),"
        preamble += "\n    vv=np.eye(t1.shape[1]),"
        preamble += "\n)"
    return preamble


def get_density_einsum_postamble(n, spin):
    """Get the einsum postamble for the density."""
    # TODO hardcoded
    if n == 1:
        if spin == "uhf":
            postamble = "rdm1.aa = np.block([[rdm1.aa.oo, rdm1.aa.ov], [rdm1.aa.vo, rdm1.aa.vv]])"
            postamble += (
                "\nrdm1.bb = np.block([[rdm1.bb.oo, rdm1.bb.ov], [rdm1.bb.vo, rdm1.bb.vv]])"
            )
        else:
            postamble = "rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])"
    elif n == 2:
        if spin == "uhf":
            postamble = "rdm2.aaaa = pack_2e(%s)" % ", ".join(f"rdm2.aaaa.{perm}" for perm in ov_2e)
            postamble += "\nrdm2.abab = pack_2e(%s)" % ", ".join(
                f"rdm2.abab.{perm}" for perm in ov_2e
            )
            postamble += "\nrdm2.bbbb = pack_2e(%s)" % ", ".join(
                f"rdm2.bbbb.{perm}" for perm in ov_2e
            )
            postamble += "\nrdm2 = Namespace("
            postamble += "\n    aaaa=rdm2.aaaa.swapaxes(1, 2),"
            postamble += "\n    aabb=rdm2.abab.swapaxes(1, 2),"
            postamble += "\n    bbbb=rdm2.bbbb.swapaxes(1, 2),"
            postamble += "\n)"
        else:
            postamble = "rdm2 = pack_2e(%s)" % ", ".join(f"rdm2.{perm}" for perm in ov_2e)
            postamble += "\nrdm2 = rdm2.swapaxes(1, 2)"
