"""Tools for FCI solvers to get amplitudes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf.ci.cisd import tn_addrs_signs

from ebcc import numpy as np
from ebcc import util
from ebcc.codegen import RecCC, UecCC

if TYPE_CHECKING:

    from numpy import float64, int64
    from numpy.typing import NDArray
    from pyscf.fci import FCI

    from ebcc.cc.rebcc import SpinArrayType
    from ebcc.ham import Space
    from ebcc.util import Namespace


def _tn_addrs_signs(norb: int, nocc: int, order: int) -> tuple[NDArray[int64], NDArray[int64]]:
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


def extract_amplitudes_restricted(fci: FCI, space: Space) -> Namespace[SpinArrayType]:
    """Extract amplitudes from an FCI calculation with restricted symmetry.

    The FCI calculatiion should have been performed in the active space according to the given
    `space`, i.e. using `mo_coeff[:, space.active]`.

    Args:
        fci: PySCF FCI object.
        space: Space containing the frozen, correlated, and active fermionic spaces.

    Returns:
        Cluster amplitudes in the active space.
    """

    def _shape(chars: str) -> tuple[int, ...]:
        """Get a shape from the space."""
        return tuple(space.size(char) for char in chars)

    # Get the adresses for each order
    addr1, sign1 = _tn_addrs_signs(space.nact, space.naocc, 1)
    addr2, sign2 = _tn_addrs_signs(space.nact, space.naocc, 2)
    addr3, sign3 = _tn_addrs_signs(space.nact, space.naocc, 3)
    addr4, sign4 = _tn_addrs_signs(space.nact, space.naocc, 4)

    # C1 amplitudes
    c1 = fci.ci[0, addr1] * sign1
    c1 = c1.reshape(_shape("OV"))

    # C2 amplitudes
    c2 = fci.ci[np.ix_(addr1, addr1)] * sign1[:, None] * sign1[None, :]
    c2 = c2.reshape(_shape("OVOV"))
    c2 = c2.transpose(0, 2, 1, 3)

    # C3 amplitudes
    c3 = fci.ci[np.ix_(addr2, addr1)] * sign2[:, None] * sign1[None, :]
    c3 = util.decompress_axes("iiaajb", c3, shape=_shape("OOVVOV"))
    c3 = c3.transpose(0, 4, 1, 2, 5, 3)

    # C4 amplitudes
    c4 = fci.ci[np.ix_(addr2, addr2)] * sign2[:, None] * sign2[None, :]
    c4 = util.decompress_axes("iiaajjbb", c4, shape=_shape("OOVVOOVV"))
    c4 = c4.transpose(0, 4, 1, 5, 2, 6, 3, 7)

    # C4a amplitudes
    c4a = fci.ci[np.ix_(addr3, addr1)] * sign3[:, None] * sign1[None, :]
    c4a = util.decompress_axes("iiiaaajb", c4a, shape=_shape("OOOVVVOV"))
    c4a = c4a.transpose(0, 1, 6, 2, 3, 4, 7, 5)

    # Scale by reference energy
    c1 /= fci.ci[0, 0]
    c2 /= fci.ci[0, 0]
    c3 /= fci.ci[0, 0]
    c4 /= fci.ci[0, 0]
    c4a /= fci.ci[0, 0]

    # Transform to T amplitudes
    t1 = RecCC.convert_c1_to_t1(c1=c1)["t1"]  # type: ignore
    t2 = RecCC.convert_c2_to_t2(c2=c2, t1=t1)["t2"]  # type: ignore
    t3 = RecCC.convert_c3_to_t3(c3=c3, t1=t1, t2=t2)["t3"]  # type: ignore
    t4s = RecCC.convert_c4_to_t4(c4=c4, c4a=c4a, t1=t1, t2=t2, t3=t3)  # type: ignore
    t4 = t4s["t4"]  # type: ignore
    t4a = t4s["t4a"]  # type: ignore

    return util.Namespace(t1=t1, t2=t2, t3=t3, t4=t4, t4a=t4a)


def extract_amplitudes_unrestricted(
    fci: FCI, space: tuple[Space, Space]
) -> Namespace[SpinArrayType]:
    """Extract amplitudes from an FCI calculation with unrestricted symmetry.

    The FCI calculatiion should have been performed in the active space according to the given
    `space`, i.e. using `mo_coeff[:, space.active]`.

    Args:
        fci: PySCF FCI object.
        space: Space containing the frozen, correlated, and active fermionic spaces for each spin
            channel.

    Returns:
        Cluster amplitudes in the active space.
    """

    def _shape(comb: str, chars: str) -> tuple[int, ...]:
        """Get a shape from the space."""
        return tuple(space["ab".index(s)].size(char) for s, char in zip(comb, chars))

    # Get the adresses for each order
    addr1a, sign1a = _tn_addrs_signs(space[0].nact, space[0].naocc, 1)
    addr2a, sign2a = _tn_addrs_signs(space[0].nact, space[0].naocc, 2)
    addr3a, sign3a = _tn_addrs_signs(space[0].nact, space[0].naocc, 3)
    addr4a, sign4a = _tn_addrs_signs(space[0].nact, space[0].naocc, 4)
    addr1b, sign1b = _tn_addrs_signs(space[1].nact, space[1].naocc, 1)
    addr2b, sign2b = _tn_addrs_signs(space[1].nact, space[1].naocc, 2)
    addr3b, sign3b = _tn_addrs_signs(space[1].nact, space[1].naocc, 3)
    addr4b, sign4b = _tn_addrs_signs(space[1].nact, space[1].naocc, 4)

    # Amplitude containers
    c1: Namespace[NDArray[float64]] = util.Namespace()
    c2: Namespace[NDArray[float64]] = util.Namespace()
    c3: Namespace[NDArray[float64]] = util.Namespace()
    c4: Namespace[NDArray[float64]] = util.Namespace()

    # C1aa amplitudes
    c1.aa = fci.ci[addr1a, 0] * sign1a
    c1.aa = c1.aa.reshape(_shape("aa", "OV"))

    # C1bb amplitudes
    c1.bb = fci.ci[0, addr1b] * sign1b
    c1.bb = c1.bb.reshape(_shape("bb", "OV"))

    # C2aaaa amplitudes
    c2.aaaa = fci.ci[addr2a, 0] * sign2a
    c2.aaaa = util.decompress_axes("iiaa", c2.aaaa, shape=_shape("aaaa", "OOVV"))

    # C2abab amplitudes
    c2.abab = fci.ci[np.ix_(addr1a, addr1b)] * sign1a[:, None] * sign1b[None, :]
    c2.abab = c2.abab.reshape(_shape("aabb", "OVOV"))
    c2.abab = c2.abab.transpose(0, 2, 1, 3)

    # C2bbbb amplitudes
    c2.bbbb = fci.ci[0, addr2b] * sign2b
    c2.bbbb = util.decompress_axes("iiaa", c2.bbbb, shape=_shape("bbbb", "OOVV"))

    # C3aaaaaa amplitudes
    c3.aaaaaa = fci.ci[addr3a, 0] * sign3a
    c3.aaaaaa = util.decompress_axes("iiiaaa", c3.aaaaaa, shape=_shape("aaaaaa", "OOOVVV"))

    # C3abaaba amplitudes
    c3.abaaba = fci.ci[np.ix_(addr2a, addr1b)] * sign2a[:, None] * sign1b[None, :]
    c3.abaaba = util.decompress_axes("iiaajb", c3.abaaba, shape=_shape("aaaabb", "OOVVOV"))
    c3.abaaba = c3.abaaba.transpose(0, 4, 1, 2, 5, 3)

    # C3babbab amplitudes
    c3.babbab = fci.ci[np.ix_(addr1a, addr2b)] * sign1a[:, None] * sign2b[None, :]
    c3.babbab = util.decompress_axes("iajjbb", c3.babbab, shape=_shape("aabbbb", "OVOOVV"))
    c3.babbab = c3.babbab.transpose(2, 0, 3, 4, 1, 5)

    # C3bbbbbb amplitudes
    c3.bbbbbb = fci.ci[0, addr3b] * sign3b
    c3.bbbbbb = util.decompress_axes("iiiaaa", c3.bbbbbb, shape=_shape("bbbbbb", "OOOVVV"))

    # C4aaaaaaaa amplitudes
    c4.aaaaaaaa = fci.ci[addr4a, 0] * sign4a
    c4.aaaaaaaa = util.decompress_axes(
        "iiiiaaaa", c4.aaaaaaaa, shape=_shape("aaaaaaaa", "OOOOVVVV")
    )

    # C4aabaaaba amplitudes
    c4.aabaaaba = fci.ci[np.ix_(addr3a, addr1b)] * sign3a[:, None] * sign1b[None, :]
    c4.aabaaaba = util.decompress_axes(
        "iiiaaajb", c4.aabaaaba, shape=_shape("aaaaaabb", "OOOVVVOV")
    )
    c4.aabaaaba = c4.aabaaaba.transpose(0, 1, 6, 2, 3, 4, 7, 5)

    # C4abababab amplitudes
    c4.abababab = fci.ci[np.ix_(addr2a, addr2b)] * sign2a[:, None] * sign2b[None, :]
    c4.abababab = util.decompress_axes(
        "iiaajjbb", c4.abababab, shape=_shape("aaaabbbb", "OOVVOOVV")
    )
    c4.abababab = c4.abababab.transpose(0, 4, 1, 5, 2, 6, 3, 7)

    # C4babbbabb amplitudes
    c4.babbbabb = fci.ci[np.ix_(addr1a, addr3b)] * sign1a[:, None] * sign3b[None, :]
    c4.babbbabb = util.decompress_axes(
        "iajjjbbb", c4.babbbabb, shape=_shape("aabbbbbb", "OVOOOVVV")
    )
    c4.babbbabb = c4.babbbabb.transpose(2, 0, 3, 4, 5, 1, 6, 7)

    # C4bbbbbbbb amplitudes
    c4.bbbbbbbb = fci.ci[0, addr4b] * sign4b
    c4.bbbbbbbb = util.decompress_axes(
        "iiiiaaaa", c4.bbbbbbbb, shape=_shape("bbbbbbbb", "OOOOVVVV")
    )

    # Scale by reference energy
    for key in c1.keys():
        c1[key] /= fci.ci[0, 0]
    for key in c2.keys():
        c2[key] /= fci.ci[0, 0]
    for key in c3.keys():
        c3[key] /= fci.ci[0, 0]
    for key in c4.keys():
        c4[key] /= fci.ci[0, 0]

    # Transform to T amplitudes
    t1 = UecCC.convert_c1_to_t1(c1=c1)["t1"]  # type: ignore
    t2 = UecCC.convert_c2_to_t2(c2=c2, t1=t1)["t2"]  # type: ignore
    t3 = UecCC.convert_c3_to_t3(c3=c3, t1=t1, t2=t2)["t3"]  # type: ignore
    t4 = UecCC.convert_c4_to_t4(c4=c4, t1=t1, t2=t2, t3=t3)["t4"]  # type: ignore

    return util.Namespace(t1=t1, t2=t2, t3=t3, t4=t4)
