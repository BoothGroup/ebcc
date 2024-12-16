"""Tools for FCI solvers to get amplitudes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf.ci.cisd import tn_addrs_signs

from ebcc import numpy as np
from ebcc import util
from ebcc.backend import _put
from ebcc.codegen import RecCC, UecCC
from ebcc.core.precision import types

if TYPE_CHECKING:
    from typing import Iterator, Union

    from numpy import floating, integer
    from numpy.typing import NDArray
    from pyscf.fci import FCI

    from ebcc.cc.rebcc import SpinArrayType as RSpinArrayType
    from ebcc.cc.uebcc import SpinArrayType as USpinArrayType
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


def fci_to_amplitudes_restricted(
    fci: FCI, space: Space, max_order: int = 4
) -> Namespace[RSpinArrayType]:
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
    return _ci_vector_to_amplitudes_restricted(fci.ci, space, max_order=max_order)


def _coefficients_to_amplitudes_restricted(
    camps: Namespace[RSpinArrayType], max_order: int = 4
) -> Namespace[RSpinArrayType]:
    """Convert coefficient amplitudes to cluster amplitudes with restricted symmetry.

    Args:
        camps: Coefficient amplitudes.
        space: Space containing the frozen, correlated, and active fermionic spaces.
        max_order: Maximum order of the excitation.

    Returns:
        Cluster amplitudes in the active space.
    """
    amps: Namespace[RSpinArrayType] = util.Namespace()
    amps.t1 = RecCC.convert_c1_to_t1(c1=camps.c1)["t1"]  # type: ignore
    if max_order > 1:
        amps.t2 = RecCC.convert_c2_to_t2(c2=camps.c2, **dict(amps))["t2"]  # type: ignore
    if max_order > 2:
        amps.t3 = RecCC.convert_c3_to_t3(c3=camps.c3, **dict(amps))["t3"]  # type: ignore
    if max_order > 3:
        t4s = RecCC.convert_c4_to_t4(c4=camps.c4, c4a=camps.c4a, **dict(amps))  # type: ignore
        amps.t4 = t4s["t4"]  # type: ignore
        amps.t4a = t4s["t4a"]  # type: ignore

    # FIXME: There should probably be some scaling here as in UHF, but instead the scaling is
    #        done in a hacky way in `_amplitudes_to_ci_vector_restricted`

    return amps


def _ci_vector_to_amplitudes_restricted(
    ci: NDArray[T], space: Space, max_order: int = 4
) -> Namespace[RSpinArrayType]:
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
    camps: Namespace[RSpinArrayType] = util.Namespace()
    camps.c1 = _get_c("b")
    if max_order > 1:
        camps.c2 = _get_c("ab")
    if max_order > 2:
        camps.c3 = _get_c("aba")
    if max_order > 3:
        camps.c4 = _get_c("abab")
        camps.c4a = _get_c("abaa")

    # Transform to T amplitudes
    amps = _coefficients_to_amplitudes_restricted(camps, max_order=max_order)

    return amps


def _amplitudes_to_ci_vector_restricted(
    amps: Namespace[RSpinArrayType], max_order: int = 4, normalise: bool = True
) -> NDArray[T]:
    """Pack the amplitudes into a CI vector with restricted symmetry.

    Args:
        amps: Cluster amplitudes.
        space: Space containing the frozen, correlated, and active fermionic spaces.
        max_order: Maximum order of the excitation.
        normalise: Whether to normalise the CI vector. If the vector is not normalised, the
            reference energy term (`ci[0, 0]`) will be `1.0`.

    Returns:
        CI vector.

    Note:
        This function converts the amplitudes to an unrestricted representation and calls
        `_pack_vector_unrestricted`, and so the restricted T->C conversion routines are never
        called. This makes the spin cases easier to handle, and there is no loss in performance
        since the CI vector needs all spin cases to be included anyway.
    """
    if max_order > 4:
        # TODO: Just to match extract_amplitudes_restricted
        raise NotImplementedError("Only up to 4th order amplitudes are supported.")

    # FIXME: I have literally no idea why this works but it does, if you want well defined
    #        behaviour, you should probably use the unrestricted version

    amps_uhf: Namespace[USpinArrayType] = util.Namespace()
    amps_uhf.t1 = util.Namespace(
        aa=amps.t1,
        bb=amps.t1,
    )
    if max_order > 1:
        t2_aa = util.symmetrise("iiaa", amps.t2, apply_factor=False) / (2**2)
        t2_ab = amps.t2
        amps_uhf.t2 = util.Namespace(
            aaaa=t2_aa,
            abab=t2_ab,
            bbbb=t2_aa,
        )
    if max_order > 2:
        t3_aaa = util.symmetrise("iiiaaa", amps.t3, apply_factor=False) / (6**2 * 2)
        t3_aba = util.symmetrise("ijiaba", amps.t3, apply_factor=False) / (2**2 * 2)
        amps_uhf.t3 = util.Namespace(
            aaaaaa=t3_aaa,
            abaaba=t3_aba,
            babbab=t3_aba,
            bbbbbb=t3_aaa,
        )
    if max_order > 3:
        t4_aaaa = util.symmetrise("iiiiaaaa", amps.t4, apply_factor=False) / (24**2 * 4)
        t4_aaab = util.symmetrise(
            "iiijaaab", amps.t4a.transpose(0, 2, 3, 1, 4, 6, 7, 5), apply_factor=False
        ) / (6**2 * 6)
        t4_baaa = util.symmetrise(
            "jiiibaaa", amps.t4a.transpose(1, 0, 2, 3, 5, 4, 6, 7), apply_factor=False
        ) / (6**2 * 6)
        t4_abab = util.symmetrise("ijijabab", amps.t4, apply_factor=False) / (2**6)
        amps_uhf.t4 = util.Namespace(
            aaaaaaaa=t4_aaaa,
            aaabaaab=t4_aaab,
            abababab=t4_abab,
            abbbabbb=t4_baaa,
            bbbbbbbb=t4_aaaa,
        )

    return _amplitudes_to_ci_vector_unrestricted(amps_uhf, max_order=max_order, normalise=normalise)


def fci_to_amplitudes_unrestricted(
    fci: FCI, space: tuple[Space, Space], max_order: int = 4
) -> Namespace[USpinArrayType]:
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
    return _ci_vector_to_amplitudes_unrestricted(fci.ci, space, max_order=max_order)


def _coefficients_to_amplitudes_unrestricted(
    camps: Namespace[USpinArrayType], max_order: int = 4
) -> Namespace[USpinArrayType]:
    """Convert coefficient amplitudes to cluster amplitudes with unrestricted symmetry.

    Args:
        camps: Coefficient amplitudes.
        space: Space containing the frozen, correlated, and active fermionic spaces for each spin
            channel.
        max_order: Maximum order of the excitation.

    Returns:
        Cluster amplitudes in the active space.
    """
    amps: Namespace[USpinArrayType] = util.Namespace()
    amps.t1 = UecCC.convert_c1_to_t1(c1=camps.c1)["t1"]  # type: ignore
    if max_order > 1:
        amps.t2 = UecCC.convert_c2_to_t2(c2=camps.c2, **dict(amps))["t2"]  # type: ignore
    if max_order > 2:
        amps.t3 = UecCC.convert_c3_to_t3(c3=camps.c3, **dict(amps))["t3"]  # type: ignore
    if max_order > 3:
        amps.t4 = UecCC.convert_c4_to_t4(c4=camps.c4, **dict(amps))["t4"]  # type: ignore

    # FIXME: This is probably a convention issue -- I multiply by the factor in the C->T generated
    #        code, but removing that doesn't seem to work. This works for now.
    for order in range(1, max_order + 1):
        name = f"t{order}"
        for spins in util.generate_spin_combinations(order, unique=True):
            factor_a = util.factorial(spins[:order].count("a"))
            factor_b = util.factorial(spins[:order].count("b"))
            amps[name][spins] /= factor_a * factor_b

    return amps


def _amplitudes_to_coefficients_unrestricted(
    amps: Namespace[USpinArrayType], max_order: int = 4
) -> Namespace[USpinArrayType]:
    """Convert cluster amplitudes to coefficient amplitudes with unrestricted symmetry.

    Args:
        amps: Cluster amplitudes.
        space: Space containing the frozen, correlated, and active fermionic spaces for each spin
            channel.
        max_order: Maximum order of the excitation.

    Returns:
        Coefficient amplitudes.
    """
    camps: Namespace[USpinArrayType] = util.Namespace()
    camps.c1 = UecCC.convert_t1_to_c1(**dict(amps))["c1"]  # type: ignore
    if max_order > 1:
        camps.c2 = UecCC.convert_t2_to_c2(**dict(amps))["c2"]  # type: ignore
    if max_order > 2:
        camps.c3 = UecCC.convert_t3_to_c3(**dict(amps))["c3"]  # type: ignore
    if max_order > 3:
        camps.c4 = UecCC.convert_t4_to_c4(**dict(amps))["c4"]  # type: ignore
    return camps


def _ci_vector_to_amplitudes_unrestricted(
    ci: NDArray[T], space: tuple[Space, Space], max_order: int = 4
) -> Namespace[USpinArrayType]:
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
        cn = ci[addrsi, addrsj] * signsi * signsj

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
        cn /= ci[0, 0]

        return cn

    def _generator(order: int) -> Iterator[tuple[str, NDArray[T]]]:
        """Generate the key-value pairs for the spin cases."""
        for comb in util.generate_spin_combinations(order, unique=True):
            yield (comb, _get_c(comb[:order]))

    # Get the C amplitudes
    camps: Namespace[USpinArrayType] = util.Namespace()
    camps.c1 = util.Namespace(**dict(_generator(1)))
    if max_order > 1:
        camps.c2 = util.Namespace(**dict(_generator(2)))
    if max_order > 2:
        camps.c3 = util.Namespace(**dict(_generator(3)))
    if max_order > 3:
        camps.c4 = util.Namespace(**dict(_generator(4)))

    # Transform to T amplitudes
    amps = _coefficients_to_amplitudes_unrestricted(camps, max_order=max_order)

    return amps


def _amplitudes_to_ci_vector_unrestricted(
    amps: Namespace[USpinArrayType], max_order: int = 4, normalise: bool = True
) -> NDArray[T]:
    """Pack the amplitudes into a CI vector with unrestricted symmetry.

    Args:
        amps: Cluster amplitudes.
        max_order: Maximum order of the excitation.
        normalise: Whether to normalise the CI vector. If the vector is not normalised, the
            reference energy term (`ci[0, 0]`) will be `1.0`.

    Returns:
        CI vector.
    """
    if max_order > 4:
        # TODO: Just to match extract_amplitudes_unrestricted
        raise NotImplementedError("Only up to 4th order amplitudes are supported.")

    # Get the adresses for each order
    nocca, nvira = amps.t1.aa.shape
    noccb, nvirb = amps.t1.bb.shape
    nocc = (nocca, noccb)
    nvir = (nvira, nvirb)
    addrsa: dict[int, NDArray[integer]] = {}
    signsa: dict[int, NDArray[integer]] = {}
    addrsb: dict[int, NDArray[integer]] = {}
    signsb: dict[int, NDArray[integer]] = {}
    for order in range(1, max_order + 1):
        addrsa[order], signsa[order] = _tn_addrs_signs(nocca + nvira, nocca, order)
        addrsb[order], signsb[order] = _tn_addrs_signs(noccb + nvirb, noccb, order)

    def _pack_vector(
        spins: str, cn: NDArray[T]
    ) -> tuple[tuple[Union[int, NDArray[integer]], Union[int, NDArray[integer]]], NDArray[T]]:
        """Get the CI vector for a given spin configuration and T amplitude.

        Also returns the indices for the given block of the CI vector.
        """
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
        shape: tuple[int, ...] = (
            util.get_compressed_size(
                "".join([s for s in subscript if s in "ia"]), i=nocc[0], a=nvir[0]
            ),
            util.get_compressed_size(
                "".join([s for s in subscript if s in "jb"]), j=nocc[1], b=nvir[1]
            ),
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

    # Get the C amplitudes
    camps = _amplitudes_to_coefficients_unrestricted(amps, max_order=max_order)

    # Get the contributions to the CI vector
    indices_i: list[Union[int, NDArray[integer]]] = []
    indices_j: list[Union[int, NDArray[integer]]] = []
    values_ij: list[NDArray[T]] = []
    for order in range(1, max_order + 1):
        for spins in util.generate_spin_combinations(order, unique=True):
            ij, c = _pack_vector(spins[:order], camps[f"c{order}"][spins])
            indices_i.append(ij[0])
            indices_j.append(ij[1])
            values_ij.append(c)

    # Find the shape of the CI vector
    shape = (
        max([np.max(i) if np.asarray(i).size else 0 for i in indices_i]) + 1,
        max([np.max(j) if np.asarray(j).size else 0 for j in indices_j]) + 1,
    )

    # Build the CI vector
    ci = np.zeros(shape, dtype=types[float])
    for i, j, val in zip(indices_i, indices_j, values_ij):
        ci = _put(
            ci,
            (i, j),  # type: ignore
            val,
        )

    # Set to intermediate normalisation
    ci[0, 0] = 1.0

    # Normalise
    if normalise:
        ci /= np.linalg.norm(ci)

    return ci
