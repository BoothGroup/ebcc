"""Tools for FCI solvers to get amplitudes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf.ci.cisd import tn_addrs_signs

from ebcc import numpy as np
from ebcc import util
from ebcc.codegen import RecCC, UecCC
from ebcc.backend import _inflate

if TYPE_CHECKING:
    from typing import Iterator, Union

    from numpy import floating, integer
    from numpy.typing import NDArray
    from pyscf.fci import FCI

    from ebcc.cc.rebcc import SpinArrayType
    from ebcc.ham import Space
    from ebcc.util import Namespace

    T = floating


def _tn_addrs_signs(norb: int, nocc: int, order: int) -> tuple[NDArray[integer], NDArray[integer]]:
    """Get the addresses and signs for the given order.

    Args:
        norb: Number of orbitals.
        nocc: Number of occupied orbitals.
        order: Order of the excitation.

    Returns:
        Addresses and signs for the given order.
    """
    addrs, signs = tn_addrs_signs(norb, nocc, order)
    return np.asarray(addrs, dtype=np.int64), np.asarray(signs, dtype=np.int64)


def extract_amplitudes_restricted(
    fci: FCI, space: Space, max_order: int = 4
) -> Namespace[SpinArrayType]:
    """Extract amplitudes from an FCI calculation with restricted symmetry.

    The FCI calculatiion should have been performed in the active space according to the given
    `space`, i.e. using `mo_coeff[:, space.active]`.

    Args:
        fci: PySCF FCI object.
        space: Space containing the frozen, correlated, and active fermionic spaces.
        max_order: Maximum order of the excitation.

    Returns:
        Cluster amplitudes in the active space.
    """
    return _extract_amplitudes_restricted(fci.ci, space, max_order=max_order)


def _extract_amplitudes_restricted(
    ci: NDArray[T], space: Space, max_order: int = 4
) -> Namespace[SpinArrayType]:
    """Extract amplitudes from a CI vector with restricted symmetry.

    Args:
        ci: CI vector.
        space: Space containing the frozen, correlated, and active fermionic spaces.
        max_order: Maximum order of the excitation.

    Returns:
        Cluster amplitudes in the active space.
    """
    if max_order > 4:
        # TODO: Just need a rule to generalise the RHF amplitude spins
        raise NotImplementedError("Only up to 4th order amplitudes are supported.")

    # Get the adresses for each order
    addrs: dict[int, NDArray[integer]] = {}
    signs: dict[int, NDArray[integer]] = {}
    for order in range(1, max_order + 1):
        addrs[order], signs[order] = _tn_addrs_signs(space.nact, space.naocc, order)

    def _get_c(spins: str) -> NDArray[T]:
        """Get the C amplitudes for a given spin configuration."""
        # Find the spins
        nalph = spins.count("a")
        nbeta = spins.count("b")

        # Get the addresses and signs
        addrsi: Union[int, NDArray[integer]] = 0
        addrsj: Union[int, NDArray[integer]] = 0
        signsi: Union[int, NDArray[integer]] = 1
        signsj: Union[int, NDArray[integer]] = 1
        if nalph != 0:
            addrsi = addrs[nalph]
            signsi = signs[nalph]
        if nbeta != 0:
            addrsj = addrs[nbeta]
            signsj = signs[nbeta]
        if nalph != 0 and nbeta != 0:
            addrsi, addrsj = np.ix_(addrsi, addrsj)  # type: ignore
            signsi = signsi[:, None]  # type: ignore
            signsj = signsj[None, :]  # type: ignore

        # Get the amplitudes
        cn = ci[addrsi, addrsj] * signsi * signsj

        # Decompress the axes
        shape = tuple(
            space.size(char) for char in ("O" * nalph + "V" * nalph + "O" * nbeta + "V" * nbeta)
        )
        subscript = "i" * nalph + "a" * nalph + "j" * nbeta + "b" * nbeta
        cn = util.decompress_axes(subscript, cn, shape=shape)

        # Transpose the axes
        subscript_target = ""
        for spin in spins:
            subscript_target += "i" if spin == "a" else "j"
        for spin in spins:
            subscript_target += spin  # a->a and b->b
        perm = util.get_string_permutation(subscript, subscript_target)
        cn = np.transpose(cn, perm)

        # Scale by reference energy
        cn /= ci[0, 0]

        return cn

    # Get the C amplitudes
    c1 = _get_c("b")
    if max_order > 1:
        c2 = _get_c("ab")
    if max_order > 2:
        c3 = _get_c("aba")
    if max_order > 3:
        c4 = _get_c("abab")
        c4a = _get_c("abaa")

    # Transform to T amplitudes
    amps: Namespace[SpinArrayType] = util.Namespace()
    amps.t1 = RecCC.convert_c1_to_t1(c1=c1)["t1"]  # type: ignore
    if max_order > 1:
        amps.t2 = RecCC.convert_c2_to_t2(c2=c2, **dict(amps))["t2"]  # type: ignore
    if max_order > 2:
        amps.t3 = RecCC.convert_c3_to_t3(c3=c3, **dict(amps))["t3"]  # type: ignore
    if max_order > 3:
        t4s = RecCC.convert_c4_to_t4(c4=c4, c4a=c4a, **dict(amps))  # type: ignore
        amps.t4 = t4s["t4"]  # type: ignore
        amps.t4a = t4s["t4a"]  # type: ignore

    return amps


def _pack_vector_restricted(
    amps: Namespace[SpinArrayType], max_order: int = 4
) -> NDArray[T]:
    """Pack the amplitudes into a CI vector with restricted symmetry.

    Args:
        amps: Cluster amplitudes.
        space: Space containing the frozen, correlated, and active fermionic spaces.
        max_order: Maximum order of the excitation.

    Returns:
        CI vector.
    """
    if max_order > 4:
        #TODO: Just to match extract_amplitudes_restricted
        raise NotImplementedError("Only up to 4th order amplitudes are supported.")

    # Get the adresses for each order
    nocc = next(iter(amps.values())).shape[0]
    nvir = next(iter(amps.values())).shape[-1]
    addrs: dict[int, NDArray[integer]] = {}
    signs: dict[int, NDArray[integer]] = {}
    for order in range(1, max_order + 1):
        addrs[order], signs[order] = _tn_addrs_signs(nocc + nvir, nocc, order)

    # TODO fix t4a

    def _pack_vector(spins: str) -> (tuple[NDArray[integer], NDArray[integer]], NDArray[T]):
        """Get the CI vector for a given spin configuration and T amplitude.

        Also returns the indices for the given block of the CI vector.
        """
        # Get the C amplitude
        order = len(spins)
        res = getattr(RecCC, f"convert_t{order}_to_c{order}")(**dict(amps))
        cn = res[f"t{order}new"]  # TODO fix names
        cn = getattr(cn, "o" * order + "v" * order)  # FIXME

        # Find the spins
        nalph = spins.count("a")
        nbeta = spins.count("b")

        # Get the addresses and signs
        addrsi: Union[int, NDArray[integer]] = 0
        addrsj: Union[int, NDArray[integer]] = 0
        signsi: Union[int, NDArray[integer]] = 1
        signsj: Union[int, NDArray[integer]] = 1
        if nalph != 0:
            addrsi = addrs[nalph]
            signsi = signs[nalph]
        if nbeta != 0:
            addrsj = addrs[nbeta]
            signsj = signs[nbeta]
        if nalph != 0 and nbeta != 0:
            addrsi, addrsj = np.ix_(addrsi, addrsj)  # type: ignore
            signsi = signsi[:, None]  # type: ignore
            signsj = signsj[None, :]  # type: ignore

        # Transpose the axes
        subscript = "i" * nalph + "a" * nalph + "j" * nbeta + "b" * nbeta
        subscript_target = ""
        for spin in spins:
            subscript_target += "i" if spin == "a" else "j"
        for spin in spins:
            subscript_target += spin
        perm = util.get_string_permutation(subscript, subscript_target)
        perm = np.argsort(perm)
        cn = np.transpose(cn, perm)

        # Compress the axes
        cn = util.compress_axes(subscript, cn)
        shape = (
            util.get_compressed_size("".join([s for s in subscript if s in "ia"]), i=nocc, a=nvir),
            util.get_compressed_size("".join([s for s in subscript if s in "jb"]), j=nocc, b=nvir),
        )
        if "j" not in subscript and "b" not in subscript:
            shape = shape[:-1]
        if "i" not in subscript and "a" not in subscript:
            shape = shape[1:]
        cn = cn.reshape(shape)

        # Scale by the signs
        cn *= signsi
        cn *= signsj

        return (addrsi, addrsj), cn.ravel()

    # Get the contributions to the CI vector
    indices_i: list[NDArray[integer]] = []
    indices_j: list[NDArray[integer]] = []
    values_ij: list[NDArray[T]] = []
    for order in range(1, max_order + 1):
        for spins in util.generate_spin_combinations(order, unique=True):
            # TODO: antisymmetry treatment?
            ij, c = _pack_vector(spins[: order])
            if isinstance(ij[0], int) and isinstance(ij[1], int):
                indices_i.append(np.array([ij[0]]))
                indices_j.append(np.array([ij[1]]))
            elif isinstance(ij[0], int):
                indices_i.append(np.full(len(ij[1]), ij[0]))
                indices_j.append(ij[1])
            elif isinstance(ij[1], int):
                indices_i.append(ij[0])
                indices_j.append(np.full(len(ij[0]), ij[1]))
            values_ij.append(c)

    # Build the CI vector
    i = np.concatenate(indices_i)
    j = np.concatenate(indices_j)
    val = np.concatenate(values_ij)
    shape = (np.max(i) + 1, np.max(j) + 1)
    ci = _inflate(shape, np.ix_(i, j), val)

    return ci


def extract_amplitudes_unrestricted(
    fci: FCI, space: tuple[Space, Space], max_order: int = 4
) -> Namespace[SpinArrayType]:
    """Extract amplitudes from an FCI calculation with unrestricted symmetry.

    The FCI calculatiion should have been performed in the active space according to the given
    `space`, i.e. using `mo_coeff[:, space.active]`.

    Args:
        fci: PySCF FCI object.
        space: Space containing the frozen, correlated, and active fermionic spaces for each spin
            channel.
        max_order: Maximum order of the excitation.

    Returns:
        Cluster amplitudes in the active space.
    """
    return _extract_amplitudes_unrestricted(fci.ci, space, max_order=max_order)


def _extract_amplitudes_unrestricted(
    ci: NDArray[T], space: tuple[Space, Space], max_order: int = 4
) -> Namespace[SpinArrayType]:
    """Extract amplitudes from a CI vector with unrestricted symmetry.

    Args:
        ci: CI vector.
        space: Space containing the frozen, correlated, and active fermionic spaces for each spin
            channel.
        max_order: Maximum order of the excitation.

    Returns:
        Cluster amplitudes in the active space.
    """
    if max_order > 4:
        # TODO: Just to match RHF
        raise NotImplementedError("Only up to 4th order amplitudes are supported.")

    # Get the adresses for each order
    addrsa: dict[int, NDArray[integer]] = {}
    signsa: dict[int, NDArray[integer]] = {}
    addrsb: dict[int, NDArray[integer]] = {}
    signsb: dict[int, NDArray[integer]] = {}
    for order in range(1, max_order + 1):
        addrsa[order], signsa[order] = _tn_addrs_signs(space[0].nact, space[0].naocc, order)
        addrsb[order], signsb[order] = _tn_addrs_signs(space[1].nact, space[1].naocc, order)

    def _get_c(spins: str) -> NDArray[T]:
        """Get the C amplitudes for a given spin configuration."""
        # Find the spins
        nalph = spins.count("a")
        nbeta = spins.count("b")

        # Get the addresses and signs
        addrsi: Union[int, NDArray[integer]] = 0
        addrsj: Union[int, NDArray[integer]] = 0
        signsi: Union[int, NDArray[integer]] = 1
        signsj: Union[int, NDArray[integer]] = 1
        if nalph != 0:
            addrsi = addrsa[nalph]
            signsi = signsa[nalph]
        if nbeta != 0:
            addrsj = addrsb[nbeta]
            signsj = signsb[nbeta]
        if nalph != 0 and nbeta != 0:
            addrsi, addrsj = np.ix_(addrsi, addrsj)  # type: ignore
            signsi = signsi[:, None]  # type: ignore
            signsj = signsj[None, :]  # type: ignore

        # Get the amplitudes
        cn = fci.ci[addrsi, addrsj] * signsi * signsj

        # Decompress the axes
        shape = tuple(
            space["ab".index(s)].size(char)
            for char, s in zip("O" * nalph + "V" * nalph + "O" * nbeta + "V" * nbeta, spins + spins)
        )
        subscript = "i" * nalph + "a" * nalph + "j" * nbeta + "b" * nbeta
        cn = util.decompress_axes(subscript, cn, shape=shape)

        # Transpose the axes
        subscript_target = ""
        for spin in spins:
            subscript_target += "i" if spin == "a" else "j"
        for spin in spins:
            subscript_target += spin  # a->a and b->b
        perm = util.get_string_permutation(subscript, subscript_target)
        cn = np.transpose(cn, perm)

        # Scale by reference energy
        cn /= fci.ci[0, 0]

        return cn

    def _generator(order: int) -> Iterator[tuple[str, NDArray[T]]]:
        """Generate the key-value pairs for the spin cases."""
        for comb in util.generate_spin_combinations(order, unique=True):
            yield (comb, _get_c(comb[:order]))

    # Get the C amplitudes
    c1 = util.Namespace(**dict(_generator(1)))
    if max_order > 1:
        c2 = util.Namespace(**dict(_generator(2)))
    if max_order > 2:
        c3 = util.Namespace(**dict(_generator(3)))
    if max_order > 3:
        c4 = util.Namespace(**dict(_generator(4)))

    # Transform to T amplitudes
    amps: Namespace[SpinArrayType] = util.Namespace()
    amps.t1 = UecCC.convert_c1_to_t1(c1=c1)["t1"]  # type: ignore
    if max_order > 1:
        amps.t2 = UecCC.convert_c2_to_t2(c2=c2, **dict(amps))["t2"]  # type: ignore
    if max_order > 2:
        amps.t3 = UecCC.convert_c3_to_t3(c3=c3, **dict(amps))["t3"]  # type: ignore
    if max_order > 3:
        amps.t4 = UecCC.convert_c4_to_t4(c4=c4, **dict(amps))["t4"]  # type: ignore

    # FIXME: This is some representational issue to ensure that tCC is correct -- why is it 0.25
    #        and not 0.5, and should we generalise with util.symmetrise?
    amps.t2.aaaa = (amps.t2.aaaa - amps.t2.aaaa.swapaxes(0, 1)) * 0.25  # type: ignore
    amps.t2.bbbb = (amps.t2.bbbb - amps.t2.bbbb.swapaxes(0, 1)) * 0.25  # type: ignore

    return amps
