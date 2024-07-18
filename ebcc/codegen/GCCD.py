"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-07-18T20:05:00.813398
  * python version: 3.10.12 (main, Mar 22 2024, 16:50:05) [GCC 11.4.0]
  * albert version: 0.0.0
  * caller: /home/ollie/git/albert/albert/codegen/einsum.py
  * node: ollie-desktop
  * system: Linux
  * processor: x86_64
  * release: 6.5.0-44-generic
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace


def energy(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:05:00.853940.

    Parameters
    ----------
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    e_cc : float
        Coupled cluster energy.
    """

    e_cc = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 2, 3), (), optimize=True) * 0.25

    return e_cc

def update_amps(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:05:01.637880.

    Parameters
    ----------
    f : array
        Fock matrix.
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    t2new : array
        Updated T2 residuals.
    """

    tmp7 = einsum(t2, (0, 1, 2, 3), v.oovv, (1, 4, 2, 3), (0, 4), optimize=True) * -1
    tmp5 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 3, 4), (2, 4), optimize=True) * -1
    tmp3 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 5, 2), optimize=True)
    tmp9 = v.oooo.transpose((2, 3, 1, 0)).copy() * -2
    tmp9 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (5, 4, 1, 0), optimize=True)
    t2new = einsum(tmp9, (0, 1, 2, 3), t2, (0, 1, 4, 5), (2, 3, 5, 4), optimize=True) * -0.25
    del tmp9
    tmp2 = einsum(t2, (0, 1, 2, 3), f.oo, (4, 1), (4, 0, 2, 3), optimize=True)
    t2new += tmp2.transpose((1, 0, 3, 2))
    t2new += tmp2.transpose((0, 1, 3, 2)) * -1
    del tmp2
    tmp8 = einsum(t2, (0, 1, 2, 3), tmp7, (4, 1), (0, 4, 2, 3), optimize=True)
    del tmp7
    t2new += tmp8.transpose((1, 0, 3, 2)) * -0.5
    t2new += tmp8.transpose((0, 1, 3, 2)) * 0.5
    del tmp8
    tmp0 = einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4), optimize=True)
    t2new += tmp0.transpose((1, 0, 3, 2)) * -1
    t2new += tmp0.transpose((1, 0, 2, 3))
    del tmp0
    tmp1 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5), optimize=True)
    t2new += tmp1.transpose((1, 0, 3, 2)) * -1
    t2new += tmp1.transpose((1, 0, 2, 3))
    t2new += tmp1.transpose((0, 1, 3, 2))
    t2new += tmp1 * -1
    del tmp1
    tmp6 = einsum(tmp5, (0, 1), t2, (2, 3, 4, 1), (2, 3, 4, 0), optimize=True)
    del tmp5
    t2new += tmp6.transpose((0, 1, 3, 2)) * 0.5
    t2new += tmp6 * -0.5
    del tmp6
    tmp4 = einsum(t2, (0, 1, 2, 3), tmp3, (4, 1, 5, 3), (0, 4, 2, 5), optimize=True)
    del tmp3
    t2new += tmp4.transpose((0, 1, 3, 2)) * -1
    t2new += tmp4
    del tmp4
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 5, 3, 2), (1, 0, 5, 4), optimize=True) * -0.5
    t2new += v.oovv.transpose((1, 0, 3, 2))

    return {f"t2new": t2new}

def update_lams(f=None, l2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:05:02.761881.

    Parameters
    ----------
    f : array
        Fock matrix.
    l2 : array
        L2 amplitudes.
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    l2new : array
        Updated L2 residuals.
    """

    tmp13 = einsum(l2, (0, 1, 2, 3), t2, (2, 3, 1, 4), (0, 4), optimize=True) * -1
    tmp11 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 2), (3, 4), optimize=True) * -1
    tmp6 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 0, 2, 3), (1, 4), optimize=True) * -1
    tmp8 = einsum(l2, (0, 1, 2, 3), t2, (3, 4, 0, 1), (2, 4), optimize=True) * -1
    tmp2 = einsum(v.oovv, (0, 1, 2, 3), t2, (1, 4, 3, 5), (4, 0, 5, 2), optimize=True)
    tmp14 = einsum(v.oovv, (0, 1, 2, 3), tmp13, (4, 3), (0, 1, 4, 2), optimize=True) * -1
    del tmp13
    tmp12 = einsum(tmp11, (0, 1), l2, (2, 0, 3, 4), (3, 4, 2, 1), optimize=True)
    del tmp11
    tmp7 = einsum(l2, (0, 1, 2, 3), tmp6, (3, 4), (2, 4, 0, 1), optimize=True)
    del tmp6
    tmp9 = einsum(v.oovv, (0, 1, 2, 3), tmp8, (4, 1), (4, 0, 2, 3), optimize=True) * -1
    del tmp8
    tmp3 = v.ovov.transpose((2, 0, 1, 3)).copy() * -1
    tmp3 += tmp2
    del tmp2
    tmp16 = v.oooo.transpose((2, 3, 1, 0)).copy() * -1
    tmp16 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (1, 0, 5, 4), optimize=True) * 0.5
    l2new = einsum(l2, (0, 1, 2, 3), tmp16, (2, 3, 4, 5), (1, 0, 4, 5), optimize=True) * -0.5
    del tmp16
    tmp15 = tmp12.copy() * 0.5
    del tmp12
    tmp15 += tmp14 * 0.5
    del tmp14
    l2new += tmp15.transpose((3, 2, 1, 0)) * -1
    l2new += tmp15.transpose((2, 3, 1, 0))
    del tmp15
    tmp5 = einsum(l2, (0, 1, 2, 3), f.oo, (4, 3), (4, 2, 0, 1), optimize=True)
    l2new += tmp5.transpose((3, 2, 1, 0))
    l2new += tmp5.transpose((3, 2, 0, 1)) * -1
    del tmp5
    tmp0 = einsum(t2, (0, 1, 2, 3), l2, (2, 3, 4, 5), (4, 5, 0, 1), optimize=True)
    l2new += einsum(v.oovv, (0, 1, 2, 3), tmp0, (4, 5, 0, 1), (2, 3, 4, 5), optimize=True) * 0.25
    del tmp0
    tmp1 = einsum(l2, (0, 1, 2, 3), f.vv, (4, 1), (2, 3, 4, 0), optimize=True)
    l2new += tmp1.transpose((3, 2, 1, 0)) * -1
    l2new += tmp1.transpose((2, 3, 1, 0))
    del tmp1
    tmp10 = tmp7.transpose((0, 1, 3, 2)).copy() * -0.5
    del tmp7
    tmp10 += tmp9.transpose((0, 1, 3, 2)) * -0.5
    del tmp9
    l2new += tmp10.transpose((3, 2, 1, 0)) * -1
    l2new += tmp10.transpose((3, 2, 0, 1))
    del tmp10
    tmp4 = einsum(tmp3, (0, 1, 2, 3), l2, (2, 4, 0, 5), (5, 1, 4, 3), optimize=True)
    del tmp3
    l2new += tmp4.transpose((3, 2, 1, 0))
    l2new += tmp4.transpose((2, 3, 1, 0)) * -1
    l2new += tmp4.transpose((3, 2, 0, 1)) * -1
    l2new += tmp4.transpose((2, 3, 0, 1))
    del tmp4
    l2new += einsum(v.vvvv, (0, 1, 2, 3), l2, (3, 2, 4, 5), (1, 0, 5, 4), optimize=True) * -0.5
    l2new += v.oovv.transpose((3, 2, 1, 0))

    return {f"l2new": l2new}

def make_rdm1_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:05:02.857428.

    Parameters
    ----------
    l2 : array
        L2 amplitudes.
    t2 : array
        T2 amplitudes.

    Returns
    -------
    rdm1 : array
        One-particle reduced density matrix.
    """

    rdm1 = Namespace()
    delta = Namespace(
        oo=np.eye(t2.shape[0]),
        vv=np.eye(t2.shape[-1]),
    )
    rdm1.oo = delta.oo.transpose((1, 0)).copy()
    del delta
    rdm1.oo += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (4, 2), optimize=True) * -0.5
    rdm1.vv = einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4), optimize=True) * 0.5
    rdm1.ov = np.zeros((t2.shape[0], t2.shape[-1]))
    rdm1.vo = np.zeros((t2.shape[-1], t2.shape[0]))
    rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])

    return rdm1

def make_rdm2_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:05:03.861067.

    Parameters
    ----------
    l2 : array
        L2 amplitudes.
    t2 : array
        T2 amplitudes.

    Returns
    -------
    rdm2 : array
        Two-particle reduced density matrix.
    """

    rdm2 = Namespace()
    delta = Namespace(
        oo=np.eye(t2.shape[0]),
        vv=np.eye(t2.shape[-1]),
    )
    tmp4 = einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 0), (1, 4), optimize=True) * -1
    rdm2.vovo = einsum(tmp4, (0, 1), delta.oo, (2, 3), (0, 3, 1, 2), optimize=True) * 0.5
    rdm2.voov = einsum(tmp4, (0, 1), delta.oo, (2, 3), (0, 3, 2, 1), optimize=True) * -0.5
    rdm2.ovvo = einsum(tmp4, (0, 1), delta.oo, (2, 3), (2, 0, 1, 3), optimize=True) * -0.5
    rdm2.ovov = einsum(tmp4, (0, 1), delta.oo, (2, 3), (2, 0, 3, 1), optimize=True) * 0.5
    tmp2 = einsum(t2, (0, 1, 2, 3), l2, (4, 3, 5, 1), (5, 0, 4, 2), optimize=True)
    rdm2.vovo += tmp2.transpose((2, 1, 3, 0)) * -1
    rdm2.voov += tmp2.transpose((2, 1, 0, 3))
    rdm2.ovvo += tmp2.transpose((1, 2, 3, 0))
    rdm2.ovov += tmp2.transpose((1, 2, 0, 3)) * -1
    tmp1 = einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4), optimize=True)
    rdm2.oooo = einsum(delta.oo, (0, 1), tmp1, (2, 3), (0, 3, 1, 2), optimize=True) * -0.5
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (1, 3, 2, 0), optimize=True) * 0.5
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (2, 1, 0, 3), optimize=True) * 0.5
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp1, (2, 3), (3, 1, 2, 0), optimize=True) * -0.5
    tmp5 = einsum(tmp4, (0, 1), t2, (2, 3, 4, 0), (2, 3, 4, 1), optimize=True)
    del tmp4
    rdm2.oovv = tmp5.transpose((0, 1, 3, 2)).copy() * 0.5
    rdm2.oovv += tmp5 * -0.5
    del tmp5
    tmp0 = einsum(t2, (0, 1, 2, 3), l2, (2, 3, 4, 5), (4, 5, 0, 1), optimize=True)
    rdm2.oovv += einsum(t2, (0, 1, 2, 3), tmp0, (0, 1, 4, 5), (5, 4, 3, 2), optimize=True) * 0.25
    rdm2.oooo += tmp0.transpose((3, 2, 1, 0)) * 0.5
    del tmp0
    tmp3 = einsum(t2, (0, 1, 2, 3), tmp2, (1, 4, 3, 5), (4, 0, 5, 2), optimize=True)
    del tmp2
    rdm2.oovv += tmp3.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += tmp3
    del tmp3
    tmp6 = einsum(t2, (0, 1, 2, 3), tmp1, (1, 4), (0, 4, 2, 3), optimize=True)
    del tmp1
    rdm2.oovv += tmp6.transpose((1, 0, 3, 2)) * -0.5
    rdm2.oovv += tmp6.transpose((0, 1, 3, 2)) * 0.5
    del tmp6
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (3, 1, 2, 0), optimize=True)
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (1, 2, 3, 0), optimize=True) * -1
    del delta
    rdm2.oovv += t2.transpose((1, 0, 3, 2))
    rdm2.vvoo = l2.transpose((1, 0, 3, 2)).copy()
    rdm2.vvvv = einsum(t2, (0, 1, 2, 3), l2, (4, 5, 0, 1), (5, 4, 3, 2), optimize=True) * 0.5
    rdm2.ooov = np.zeros((t2.shape[0], t2.shape[0], t2.shape[0], t2.shape[-1]))
    rdm2.oovo = np.zeros((t2.shape[0], t2.shape[0], t2.shape[-1], t2.shape[0]))
    rdm2.ovoo = np.zeros((t2.shape[0], t2.shape[-1], t2.shape[0], t2.shape[0]))
    rdm2.vooo = np.zeros((t2.shape[-1], t2.shape[0], t2.shape[0], t2.shape[0]))
    rdm2.ovvv = np.zeros((t2.shape[0], t2.shape[-1], t2.shape[-1], t2.shape[-1]))
    rdm2.vovv = np.zeros((t2.shape[-1], t2.shape[0], t2.shape[-1], t2.shape[-1]))
    rdm2.vvov = np.zeros((t2.shape[-1], t2.shape[-1], t2.shape[0], t2.shape[-1]))
    rdm2.vvvo = np.zeros((t2.shape[-1], t2.shape[-1], t2.shape[-1], t2.shape[0]))
    rdm2 = pack_2e(rdm2.oooo, rdm2.ooov, rdm2.oovo, rdm2.ovoo, rdm2.vooo, rdm2.oovv, rdm2.ovov, rdm2.ovvo, rdm2.voov, rdm2.vovo, rdm2.vvoo, rdm2.ovvv, rdm2.vovv, rdm2.vvov, rdm2.vvvo, rdm2.vvvv)
    rdm2 = rdm2.swapaxes(1, 2)

    return rdm2

