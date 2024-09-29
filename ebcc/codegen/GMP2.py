"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-09-28T12:18:35.820935
  * python version: 3.10.15 (main, Sep  9 2024, 03:02:45) [GCC 11.4.0]
  * albert version: 0.0.0
  * caller: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/albert/codegen/einsum.py
  * node: fv-az1272-977
  * system: Linux
  * processor: x86_64
  * release: 6.8.0-1014-azure
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, dirsum, Namespace


def energy(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:35.882526.

    Parameters
    ----------
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    e_mp : array
    """

    e_mp = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 2, 3), ()) * 0.25

    return e_mp

def make_rdm1_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:36.007568.

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
    rdm1.vv = einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4)) * 0.5
    rdm1.oo = einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (4, 2)) * -0.5
    rdm1.oo += delta.oo
    del delta
    rdm1.ov = np.zeros((t2.shape[0], t2.shape[-1]))
    rdm1.vo = np.zeros((t2.shape[-1], t2.shape[0]))
    rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])

    return rdm1

def make_rdm2_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:36.031278.

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
    rdm2.vvoo = np.copy(l2)
    rdm2.oovv = np.copy(t2)
    rdm2.oooo = np.zeros((t2.shape[0], t2.shape[0], t2.shape[0], t2.shape[0]))
    rdm2.ooov = np.zeros((t2.shape[0], t2.shape[0], t2.shape[0], t2.shape[-1]))
    rdm2.oovo = np.zeros((t2.shape[0], t2.shape[0], t2.shape[-1], t2.shape[0]))
    rdm2.ovoo = np.zeros((t2.shape[0], t2.shape[-1], t2.shape[0], t2.shape[0]))
    rdm2.vooo = np.zeros((t2.shape[-1], t2.shape[0], t2.shape[0], t2.shape[0]))
    rdm2.ovov = np.zeros((t2.shape[0], t2.shape[-1], t2.shape[0], t2.shape[-1]))
    rdm2.ovvo = np.zeros((t2.shape[0], t2.shape[-1], t2.shape[-1], t2.shape[0]))
    rdm2.voov = np.zeros((t2.shape[-1], t2.shape[0], t2.shape[0], t2.shape[-1]))
    rdm2.vovo = np.zeros((t2.shape[-1], t2.shape[0], t2.shape[-1], t2.shape[0]))
    rdm2.ovvv = np.zeros((t2.shape[0], t2.shape[-1], t2.shape[-1], t2.shape[-1]))
    rdm2.vovv = np.zeros((t2.shape[-1], t2.shape[0], t2.shape[-1], t2.shape[-1]))
    rdm2.vvov = np.zeros((t2.shape[-1], t2.shape[-1], t2.shape[0], t2.shape[-1]))
    rdm2.vvvo = np.zeros((t2.shape[-1], t2.shape[-1], t2.shape[-1], t2.shape[0]))
    rdm2.vvvv = np.zeros((t2.shape[-1], t2.shape[-1], t2.shape[-1], t2.shape[-1]))
    rdm2 = pack_2e(rdm2.oooo, rdm2.ooov, rdm2.oovo, rdm2.ovoo, rdm2.vooo, rdm2.oovv, rdm2.ovov, rdm2.ovvo, rdm2.voov, rdm2.vovo, rdm2.vvoo, rdm2.ovvv, rdm2.vovv, rdm2.vvov, rdm2.vvvo, rdm2.vvvv)
    rdm2 = np.transpose(rdm2, (0, 2, 1, 3))
    rdm1 = make_rdm1_f(t2=t2, l2=l2)
    nocc = t2.shape[0]
    rdm1[np.diag_indices(nocc)] -= 1
    for i in range(nocc):
        rdm2[i, i, :, :] += rdm1.T
        rdm2[:, :, i, i] += rdm1.T
        rdm2[:, i, i, :] -= rdm1.T
        rdm2[i, :, :, i] -= rdm1
    for i in range(nocc):
        for j in range(nocc):
            rdm2[i, i, j, j] += 1
            rdm2[i, j, j, i] -= 1

    return rdm2

def hbar_matvec_ip_intermediates(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:36.961524.

    Parameters
    ----------
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    tmp0 : array
    """

    tmp0 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 2, 3), (4, 0))

    return {f"tmp0": tmp0}

def hbar_matvec_ip(f=None, r1=None, r2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:36.963821.

    Parameters
    ----------
    f : array
        Fock matrix.
    r1 : array
        R1 amplitudes.
    r2 : array
        R2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    r1new : array
        Updated R1 residuals.
    r2new : array
        Updated R2 residuals.
    """

    ints = kwargs["ints"]
    tmp1 = einsum(f.oo, (0, 1), r2, (2, 1, 3), (0, 2, 3))
    tmp0 = np.copy(f.oo) * 4
    tmp0 += ints.tmp0
    tmp0 += np.transpose(ints.tmp0, (1, 0))
    del ints.tmp0
    r2new = einsum(r2, (0, 1, 2), f.vv, (3, 2), (0, 1, 3))
    r2new += einsum(v.ooov, (0, 1, 2, 3), r1, (2,), (0, 1, 3)) * -1
    r2new += tmp1
    r2new += np.transpose(tmp1, (1, 0, 2)) * -1
    del tmp1
    r1new = einsum(r2, (0, 1, 2), v.ovoo, (3, 2, 0, 1), (3,)) * -0.5
    r1new += einsum(r1, (0,), tmp0, (0, 1), (1,)) * -0.25
    del tmp0

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_matvec_ea_intermediates(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:37.824435.

    Parameters
    ----------
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    tmp0 : array
    """

    tmp0 = einsum(v.oovv, (0, 1, 2, 3), t2, (0, 1, 4, 3), (4, 2))

    return {f"tmp0": tmp0}

def hbar_matvec_ea(f=None, r1=None, r2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:37.826872.

    Parameters
    ----------
    f : array
        Fock matrix.
    r1 : array
        R1 amplitudes.
    r2 : array
        R2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    r1new : array
        Updated R1 residuals.
    r2new : array
        Updated R2 residuals.
    """

    ints = kwargs["ints"]
    tmp1 = einsum(r2, (0, 1, 2), f.vv, (3, 1), (2, 3, 0))
    tmp0 = np.copy(f.vv) * -1
    tmp0 += ints.tmp0 * 0.25
    tmp0 += np.transpose(ints.tmp0, (1, 0)) * 0.25
    del ints.tmp0
    r2new = einsum(r1, (0,), v.ovvv, (1, 0, 2, 3), (2, 3, 1)) * -1
    r2new += einsum(r2, (0, 1, 2), f.oo, (3, 2), (0, 1, 3)) * -1
    r2new += np.transpose(tmp1, (1, 2, 0)) * -1
    r2new += np.transpose(tmp1, (2, 1, 0))
    del tmp1
    r1new = einsum(r2, (0, 1, 2), v.ovvv, (2, 3, 0, 1), (3,)) * -0.5
    r1new += einsum(tmp0, (0, 1), r1, (0,), (1,)) * -1
    del tmp0

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_matvec_ee_intermediates(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:39.625983.

    Parameters
    ----------
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    tmp0 : array
    tmp1 : array
    """

    tmp1 = einsum(v.oovv, (0, 1, 2, 3), t2, (0, 1, 4, 3), (4, 2))
    tmp0 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 2, 3), (4, 0))

    return {f"tmp0": tmp0, f"tmp1": tmp1}

def hbar_matvec_ee(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T12:18:39.630771.

    Parameters
    ----------
    f : array
        Fock matrix.
    r1 : array
        R1 amplitudes.
    r2 : array
        R2 amplitudes.
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    r1new : array
        Updated R1 residuals.
    r2new : array
        Updated R2 residuals.
    """

    ints = kwargs["ints"]
    tmp3 = einsum(r1, (0, 1), v.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    tmp4 = einsum(f.vv, (0, 1), r2, (2, 3, 4, 1), (2, 3, 0, 4))
    tmp6 = einsum(f.oo, (0, 1), r2, (2, 1, 3, 4), (0, 2, 3, 4))
    tmp7 = einsum(r1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    tmp5 = np.copy(np.transpose(tmp3, (1, 0, 2, 3))) * -1
    del tmp3
    tmp5 += np.transpose(tmp4, (1, 0, 2, 3)) * -1
    del tmp4
    tmp8 = np.copy(np.transpose(tmp6, (0, 1, 3, 2))) * -1
    del tmp6
    tmp8 += np.transpose(tmp7, (0, 1, 3, 2))
    del tmp7
    tmp0 = einsum(v.oovv, (0, 1, 2, 3), r1, (1, 3), (0, 2))
    tmp1 = np.copy(f.vv)
    tmp1 += np.transpose(ints.tmp1, (1, 0)) * -0.5
    del ints.tmp1
    tmp2 = np.copy(f.oo)
    tmp2 += np.transpose(ints.tmp0, (1, 0)) * 0.5
    del ints.tmp0
    r2new = np.copy(tmp5) * -1
    r2new += np.transpose(tmp5, (0, 1, 3, 2))
    del tmp5
    r2new += tmp8
    r2new += np.transpose(tmp8, (1, 0, 2, 3)) * -1
    del tmp8
    r1new = einsum(v.ovvv, (0, 1, 2, 3), r2, (4, 0, 2, 3), (4, 1)) * -0.5
    r1new += einsum(tmp0, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    del tmp0
    r1new += einsum(v.ovoo, (0, 1, 2, 3), r2, (2, 3, 4, 1), (0, 4)) * -0.5
    r1new += einsum(v.ovov, (0, 1, 2, 3), r1, (2, 1), (0, 3)) * -1
    r1new += einsum(tmp1, (0, 1), r1, (2, 0), (2, 1))
    del tmp1
    r1new += einsum(tmp2, (0, 1), r1, (0, 2), (1, 2)) * -1
    del tmp2

    return {f"r1new": r1new, f"r2new": r2new}

