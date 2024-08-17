"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-08-17T12:50:33.889738
  * python version: 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0]
  * albert version: 0.0.0
  * caller: /home/ollie/git/albert/albert/codegen/einsum.py
  * node: ollie-desktop
  * system: Linux
  * processor: x86_64
  * release: 6.8.0-40-generic
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace


def energy(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T12:50:33.926346.

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
    Code generated by `albert` 0.0.0 on 2024-08-17T12:50:34.001989.

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
    rdm1.vv = einsum(t2, (0, 1, 2, 3), l2, (4, 3, 0, 1), (4, 2)) * 0.5
    rdm1.oo = delta.oo.copy()
    del delta
    rdm1.oo += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (4, 2)) * -0.5
    rdm1.ov = np.zeros((t2.shape[0], t2.shape[-1]))
    rdm1.vo = np.zeros((t2.shape[-1], t2.shape[0]))
    rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])

    return rdm1

def make_rdm2_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T12:50:34.016048.

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
    rdm2.vvoo = l2.copy()
    rdm2.oovv = t2.copy()
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
    rdm2 = rdm2.swapaxes(1, 2)
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
    Code generated by `albert` 0.0.0 on 2024-08-17T12:50:34.562235.

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

    tmp0 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))

    return {f"tmp0": tmp0}

def hbar_matvec_ip(f=None, r1=None, r2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T12:50:34.563471.

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
    tmp1 = einsum(r2, (0, 1, 2), f.oo, (3, 1), (3, 0, 2))
    tmp0 = f.oo.copy() * 4
    tmp0 += ints.tmp0
    tmp0 += ints.tmp0.transpose((1, 0))
    del ints.tmp0
    r2new = einsum(f.vv, (0, 1), r2, (2, 3, 1), (2, 3, 0))
    r2new += einsum(v.ooov, (0, 1, 2, 3), r1, (2,), (0, 1, 3)) * -1
    r2new += tmp1
    r2new += tmp1.transpose((1, 0, 2)) * -1
    del tmp1
    r1new = einsum(r2, (0, 1, 2), v.ovoo, (3, 2, 0, 1), (3,)) * -0.5
    r1new += einsum(r1, (0,), tmp0, (0, 1), (1,)) * -0.25
    del tmp0

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_matvec_ea_intermediates(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T12:50:35.060732.

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

    tmp0 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))

    return {f"tmp0": tmp0}

def hbar_matvec_ea(f=None, r1=None, r2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T12:50:35.062043.

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
    tmp1 = einsum(f.vv, (0, 1), r2, (2, 1, 3), (3, 0, 2))
    tmp0 = f.vv.copy() * -1
    tmp0 += ints.tmp0 * 0.25
    tmp0 += ints.tmp0.transpose((1, 0)) * 0.25
    del ints.tmp0
    r2new = einsum(r1, (0,), v.ovvv, (1, 0, 2, 3), (2, 3, 1)) * -1
    r2new += einsum(f.oo, (0, 1), r2, (2, 3, 1), (2, 3, 0)) * -1
    r2new += tmp1.transpose((1, 2, 0)) * -1
    r2new += tmp1.transpose((2, 1, 0))
    del tmp1
    r1new = einsum(r2, (0, 1, 2), v.ovvv, (2, 3, 0, 1), (3,)) * -0.5
    r1new += einsum(tmp0, (0, 1), r1, (0,), (1,)) * -1
    del tmp0

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_matvec_ee_intermediates(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T12:50:36.102028.

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

    tmp1 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    tmp0 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))

    return {f"tmp0": tmp0, f"tmp1": tmp1}

def hbar_matvec_ee(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T12:50:36.104342.

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
    tmp3 = einsum(v.ooov, (0, 1, 2, 3), r1, (2, 4), (0, 1, 4, 3))
    tmp4 = einsum(f.vv, (0, 1), r2, (2, 3, 4, 1), (2, 3, 0, 4))
    tmp7 = einsum(r1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    tmp6 = einsum(r2, (0, 1, 2, 3), f.oo, (4, 1), (4, 0, 2, 3))
    tmp5 = tmp3.transpose((1, 0, 2, 3)).copy() * -1
    del tmp3
    tmp5 += tmp4.transpose((1, 0, 2, 3)) * -1
    del tmp4
    tmp8 = tmp6.transpose((0, 1, 3, 2)).copy() * -1
    del tmp6
    tmp8 += tmp7.transpose((0, 1, 3, 2))
    del tmp7
    tmp2 = f.oo.copy()
    tmp2 += ints.tmp0.transpose((1, 0)) * 0.5
    del ints.tmp0
    tmp1 = f.vv.copy()
    tmp1 += ints.tmp1.transpose((1, 0)) * -0.5
    del ints.tmp1
    tmp0 = einsum(v.oovv, (0, 1, 2, 3), r1, (1, 3), (0, 2))
    r2new = tmp5.copy() * -1
    r2new += tmp5.transpose((0, 1, 3, 2))
    del tmp5
    r2new += tmp8
    r2new += tmp8.transpose((1, 0, 2, 3)) * -1
    del tmp8
    r1new = einsum(tmp0, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    del tmp0
    r1new += einsum(v.ovvv, (0, 1, 2, 3), r2, (4, 0, 2, 3), (4, 1)) * -0.5
    r1new += einsum(r2, (0, 1, 2, 3), v.ovoo, (4, 3, 0, 1), (4, 2)) * -0.5
    r1new += einsum(r1, (0, 1), v.ovov, (2, 1, 0, 3), (2, 3)) * -1
    r1new += einsum(r1, (0, 1), tmp1, (1, 2), (0, 2))
    del tmp1
    r1new += einsum(r1, (0, 1), tmp2, (0, 2), (2, 1)) * -1
    del tmp2

    return {f"r1new": r1new, f"r2new": r2new}

