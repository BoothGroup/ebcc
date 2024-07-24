"""Common functionality for the bootstrap scripts for `ebcc`.
"""

import itertools
import time
from collections import defaultdict

from albert.codegen.einsum import EinsumCodeGen as _EinsumCodeGen
from albert.codegen.base import sort_exprs
from albert.optim._gristmill import optimise as _optimise
from albert.qc.spin import generalised_to_restricted, generalised_to_unrestricted
from albert.qc.index import Index
from albert.qc.rhf import ERI as RERI, CDERI as RCDERI
from albert.qc.uhf import ERI as UERI, CDERI as UCDERI
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
    "b": ["x", "y", "z", "b0", "b1", "b2", "b3"],
    "x": ["P", "Q", "R", "S", "x0", "x1", "x2", "x3", "x4", "x5", "x7"],
}


default_sectors = {i: k for k, v in default_indices.items() for i in v}


default_sizes = {
    "o": 200,
    "v": 1000,
    "b": 10,
    "x": 3000,
}


class EinsumCodeGen(_EinsumCodeGen):
    """Code generator for the bootstrap scripts for `ebcc`."""

    def __init__(
        self,
        einsum_func="einsum",
        einsum_kwargs=None,
        transpose_func="{arg}.transpose({transpose})",
        name_generator=None,
        spin="ghf",
        **kwargs,
    ):
        if einsum_kwargs is None:
            einsum_kwargs = {}
        super().__init__(
            einsum_func=einsum_func,
            einsum_kwargs=einsum_kwargs,
            transpose_func=transpose_func,
            name_generator=name_generator,
            spin=spin,
            **kwargs,
        )

    def preamble(self):
        preamble = "from ebcc import numpy as np\n"
        preamble += "from ebcc.util import pack_2e, einsum, direct_sum, Namespace"
        super().preamble(preamble)

    def ignore_argument(self, arg):
        """
        Return `True` if a potential function argument should be
        ignored.
        """
        return "tmp" in arg.name or arg.name in ("δ", "gc")

    def function_docstring(self, docstring):
        """Write the function docstring."""
        if self.spin in ("rhf", "ghf"):
            array_type = "array"
        else:
            array_type = "Namespace of arrays"
        amplitude_names = [
            "t1", "t2", "t3", "l1", "l2", "l3", "s1", "s2", "ls1", "ls2", "u11", "u12", "lu11", "lu12", "r1", "r2", "r3"
        ]
        descriptions = {
            "f": "Fock matrix.",
            "v": "Electron repulsion integrals.",
            "G": "One-boson Hamiltonian.",
            "w": "Two-boson Hamiltonian.",
            "g": "Electron-boson coupling.",
            "e_cc": "Coupled cluster energy.",
            "e_pert": "Perturbation energy.",
            "rdm1": "One-particle reduced density matrix.",
            "rdm2": "Two-particle reduced density matrix.",
            "rdm1_b": "One-body reduced density matrix.",
            "rdm_eb_cre": "Electron-boson coupling reduced density matrix, creation part.",
            "rdm_eb_des": "Electron-boson coupling reduced density matrix, annihilation part.",
            "dm_cre": "Single boson density matrix, creation part.",
            "dm_des": "Single boson density matrix, annihilation part.",
            **{f"{name}": f"{name.upper()} amplitudes." for name in amplitude_names},
            **{f"{name}new": f"Updated {name.upper()} residuals." for name in amplitude_names},
        }
        types = {
            "f": array_type,
            "v": array_type,
            "G": "array",
            "w": "array",
            "g": array_type,
            "e_cc": "float",
            "e_pert": "float",
            "rdm1": array_type,
            "rdm2": array_type,
            "rdm1_b": "array",
            "rdm_eb_cre": array_type,
            "rdm_eb_des": array_type,
            "dm_cre": "array",
            "dm_des": "array",
            **{f"{name}": array_type if not name.startswith("s") else "array" for name in amplitude_names},
            **{f"{name}new": array_type if not name.startswith("s") else "array" for name in amplitude_names},
        }
        docstring = docstring.split("\n")
        new_docstring = []
        for line in docstring:
            if len(line.split()) and line.split()[0] in descriptions:
                name = line.split()[0]
                line = f"{name} : {types[name]}\n    {descriptions[name]}"
            new_docstring.append(line)
        docstring = "\n".join(new_docstring)
        super().function_docstring(docstring)


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
    if tensor.name in ("f", "v", "d", "Γ", "δ", "g", "gc", "rdm_eb_cre", "rdm_eb_des"):
        if tensor.name == "d":
            name = "rdm1"
        elif tensor.name == "Γ":
            name = "rdm2"
        elif tensor.name == "δ":
            name = "delta"
        else:
            name = tensor.name
        if add_spaces:
            spaces = [i.space for i in tensor.indices]
            return f"{name}.{''.join(spaces)}"
        else:
            return name
    else:
        return tensor.name


def name_generator_uhf(tensor, add_spaces=True):
    """Generate names for the UHF case."""
    if tensor.name in ("f", "v", "d", "Γ", "δ", "g", "gc", "rdm_eb_cre", "rdm_eb_des"):
        if tensor.name == "d":
            name = "rdm1"
        elif tensor.name == "Γ":
            name = "rdm2"
        elif tensor.name == "δ":
            name = "delta"
        else:
            name = tensor.name
        if add_spaces:
            spins = ["a" if i.spin == "α" else "b" for i in tensor.indices if i.spin is not None]
            spaces = [i.space for i in tensor.indices]
            return f"{name}.{''.join(spins)}.{''.join(spaces)}"
        else:
            return name
    elif tensor.name in ("t1", "t2", "t3", "t1new", "t2new", "t3new", "l1", "l2", "l3", "l1new", "l2new", "l3new", "u11", "u11new", "u12", "u12new", "r1", "r2", "r3", "r1new", "r2new", "r3new"):
        if add_spaces:
            spins = ["a" if i.spin == "α" else "b" for i in tensor.indices if i.spin is not None]
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


def remove_disconnected_eom(terms):
    """Remove the EOM terms that are not connected to the R amplitudes."""
    new_terms = []
    for term in terms:
        r = None
        rest = []
        for t in term[1:]:
            if "r" in t:
                r = t
            elif not t.startswith("P("):
                rest.append(t)
        r_inds = set(r.split("(")[1].split(")")[0].split(","))
        rest_inds = set()
        for r in rest:
            if "<" in r:
                r = r.replace("<", "(").replace(">", ")").replace("||", ",")
            rest_inds.update(r.split("(")[1].split(")")[0].split(","))
        if r_inds & rest_inds:
            new_terms.append(term)
    return new_terms


def optimise(outputs, exprs, spin, strategy="greedy", sizes=None):
    """Optimise the expressions."""

    if sizes is None:
        sizes = default_sizes

    index_sizes = {}
    index_groups = []
    for sector, indices in default_indices.items():
        spins = ("α", "β") if spin == "uhf" and sector not in ("b", "x") else (None,)
        for s in spins:
            index_sizes.update({Index(index, space=sector, spin=s): sizes[sector] for index in indices})
            index_groups.append([Index(index, space=sector, spin=s) for index in indices])

    opt = _optimise(
        *zip(outputs, exprs),
        index_groups=index_groups,
        sizes=index_sizes,
        strategy=strategy,
    )

    return zip(*opt)


def get_t_amplitude_outputs(exprs, name, indices=None):
    """Get the outputs for the T amplitude code."""
    if indices is not None:
        key = lambda i: indices.index(i.name)
    else:
        key = lambda i: i
    return [Tensor(*sorted(e.external_indices, key=key), name=name) for e in exprs]


def get_l_amplitude_outputs(exprs, name, indices=None):
    """Get the outputs for the L amplitude code."""
    def key(i):
        if indices is not None:
            return indices.index(i.name)
        return (" bvo".index(i.space), i.spin, i.name)
    return [Tensor(*sorted(e.external_indices, key=key), name=name) for e in exprs]


def get_density_outputs(exprs, name, indices):
    """Get the outputs for the density code."""
    tensors = []
    for expr in exprs:
        external_indices = sorted(expr.external_indices, key=lambda i: indices.index(i.name))
        tensors.append(Tensor(*external_indices, name=name))
    return tensors


def get_amplitude_spins(n, spin, which="t"):
    """Get the spin cases for the amplitudes."""

    if which in ("t", "l", "ee"):
        no = nv = n
    elif which == "ip":
        no = n
        nv = n - 1
    elif which == "ea":
        no = n - 1
        nv = n

    if spin == "rhf":
        case = {}
        for i in range(no):
            case[default_indices["o"][i]] = ["α", "β"][i % 2]
        for i in range(nv):
            case[default_indices["v"][i]] = ["α", "β"][i % 2]
        cases = [case]
    elif spin == "uhf":
        cases = []
        if which in ("ip", "ea"):
            it = itertools.product("αβ", repeat=max(no, nv))
        else:
            it = itertools.combinations_with_replacement("αβ", max(no, nv))
        for spins in it:
            # Canonicalise the spin order
            best = [None, 1e10]
            for s in itertools.permutations(spins):
                penalty = 0
                for i in range(len(s) - 1):
                    penalty += int(s[i] == s[i + 1]) * 2
                if s[0] != min(s):
                    penalty += 1
                if penalty < best[1]:
                    best = [s, penalty]
            spins = best[0]
            case = {}
            for i, s in enumerate(spins):
                if i < no:
                    case[default_indices["o"][i]] = s
                if i < nv:
                    case[default_indices["v"][i]] = s
            cases.append(case)
    elif spin == "ghf":
        case = {}
        for i in range(no):
            case[default_indices["o"][i]] = None
        for i in range(nv):
            case[default_indices["v"][i]] = None
        cases = [case]

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
        if n == 1:
            cases = [(None, None)]
        elif n == 2:
            cases = [(None, None, None, None)]
        cases = [dict(zip(indices, case)) for case in cases]

    return cases


def get_density_einsum_preamble(n, spin, name="rdm{n}"):
    """Get the einsum preamble for the density."""
    name = name.format(n=n)
    preamble = f"{name} = Namespace()"
    if spin == "uhf":
        for spins in itertools.combinations_with_replacement(["a", "b"], n):
            preamble += f"\n{name}.{''.join(spins+spins)} = Namespace()"
        preamble += "\ndelta = Namespace("
        preamble += "\n    aa=Namespace(oo=np.eye(t2.aaaa.shape[0]), vv=np.eye(t2.aaaa.shape[-1])),"
        preamble += "\n    bb=Namespace(oo=np.eye(t2.bbbb.shape[0]), vv=np.eye(t2.bbbb.shape[-1])),"
        preamble += "\n)"
    else:
        preamble += "\ndelta = Namespace("
        preamble += "\n    oo=np.eye(t2.shape[0]),"
        preamble += "\n    vv=np.eye(t2.shape[-1]),"
        preamble += "\n)"
    return preamble


def get_density_einsum_postamble(n, spin, name="rdm{n}", spaces=None):
    """Get the einsum postamble for the density."""
    # TODO hardcoded
    name = name.format(n=n)
    if spaces is None:
        if n == 1:
            spaces = ["oo", "ov", "vo", "vv"]
        elif n == 2:
            spaces = ov_2e
    if n == 1:
        if spin == "uhf":
            postamble = f"{name}.aa = np.block([[{name}.aa.{spaces[0]}, {name}.aa.{spaces[1]}], [{name}.aa.{spaces[2]}, {name}.aa.{spaces[3]}]])"
            postamble += f"\n{name}.bb = np.block([[{name}.bb.{spaces[0]}, {name}.bb.{spaces[1]}], [{name}.bb.{spaces[2]}, {name}.bb.{spaces[3]}]])"
        else:
            postamble = f"{name} = np.block([[{name}.{spaces[0]}, {name}.{spaces[1]}], [{name}.{spaces[2]}, {name}.{spaces[3]}]])"
    elif n == 2:
        if spin == "uhf":
            postamble = f"{name}.aaaa = pack_2e(%s)" % ", ".join(f"{name}.aaaa.{perm}" for perm in spaces)
            postamble += f"\n{name}.abab = pack_2e(%s)" % ", ".join(
                f"{name}.abab.{perm}" for perm in spaces
            )
            postamble += f"\n{name}.bbbb = pack_2e(%s)" % ", ".join(
                f"{name}.bbbb.{perm}" for perm in spaces
            )
            postamble += f"\n{name} = Namespace("
            postamble += f"\n    aaaa={name}.aaaa.swapaxes(1, 2),"
            postamble += f"\n    aabb={name}.abab.swapaxes(1, 2),"
            postamble += f"\n    bbbb={name}.bbbb.swapaxes(1, 2),"
            postamble += f"\n)"
        else:
            postamble = f"{name} = pack_2e(%s)" % ", ".join(f"{name}.{perm}" for perm in spaces)
            postamble += f"\n{name} = {name}.swapaxes(1, 2)"
    return postamble


def get_boson_einsum_preamble(spin):
    """Get the einsum preamble for the density."""
    if spin == "uhf":
        preamble = "gc = Namespace("
        preamble += "\n    aa=Namespace("
        preamble += "\n        boo=g.aa.boo.transpose(0, 2, 1),"
        preamble += "\n        bov=g.aa.bvo.transpose(0, 2, 1),"
        preamble += "\n        bvo=g.aa.bov.transpose(0, 2, 1),"
        preamble += "\n        bvv=g.aa.bvv.transpose(0, 2, 1),"
        preamble += "\n    ),"
        preamble += "\n    bb=Namespace("
        preamble += "\n        boo=g.bb.boo.transpose(0, 2, 1),"
        preamble += "\n        bov=g.bb.bvo.transpose(0, 2, 1),"
        preamble += "\n        bvo=g.bb.bov.transpose(0, 2, 1),"
        preamble += "\n        bvv=g.bb.bvv.transpose(0, 2, 1),"
        preamble += "\n    ),"
        preamble += "\n)"
    else:
        preamble = "gc = Namespace("
        preamble += "\n    boo=g.boo.transpose(0, 2, 1),"
        preamble += "\n    bov=g.bvo.transpose(0, 2, 1),"
        preamble += "\n    bvo=g.bov.transpose(0, 2, 1),"
        preamble += "\n    bvv=g.bvv.transpose(0, 2, 1),"
        preamble += "\n)"
    return preamble


def get_density_fit():
    indices = {"": 0}

    def density_fit(obj):
        """Get the function to apply over an expression to density fit the ERIs."""
        i = indices[""] % len(default_indices["x"])  # TODO this will break for huge ansatzes
        aux_index = Index(f"{default_indices['x'][i]}", space="x")
        indices[""] += 1
        if hasattr(obj, "_symbol") and obj._symbol == RERI:
            left = RCDERI[(aux_index,) + obj.indices[:2]]
            right = RCDERI[(aux_index,) + obj.indices[2:]]
        elif hasattr(obj, "_symbol") and obj._symbol == UERI:
            left = UCDERI[(aux_index,) + obj.indices[:2]]
            right = UCDERI[(aux_index,) + obj.indices[2:]]
        else:
            return obj
        return left * right

    return density_fit
