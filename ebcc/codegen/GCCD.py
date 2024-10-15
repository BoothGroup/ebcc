"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-09-29T15:23:06.843112
  * python version: 3.10.15 (main, Sep  9 2024, 03:02:45) [GCC 11.4.0]
  * albert version: 0.0.0
  * caller: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/albert/codegen/einsum.py
  * node: fv-az1788-690
  * system: Linux
  * processor: x86_64
  * release: 6.8.0-1014-azure
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, dirsum, Namespace


def energy(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:06.910278.

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

    e_cc = einsum(v.oovv, (0, 1, 2, 3), t2, (0, 1, 2, 3), ()) * 0.25

    return e_cc

def update_amps(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:08.134197.

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

    tmp3 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    tmp1 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    tmp5 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 2, 3), (4, 0))
    tmp4 = einsum(t2, (0, 1, 2, 3), tmp3, (4, 3), (0, 1, 2, 4))
    del tmp3
    tmp8 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 5, 1), (4, 0, 5, 3))
    tmp2 = einsum(tmp1, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp1
    tmp6 = einsum(tmp5, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4))
    del tmp5
    tmp0 = einsum(t2, (0, 1, 2, 3), f.oo, (4, 1), (4, 0, 2, 3))
    tmp9 = np.copy(v.oooo)
    tmp9 += einsum(v.oovv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (0, 1, 4, 5)) * 0.5
    tmp7 = einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new = einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 5, 2, 3), (0, 1, 4, 5)) * 0.5
    t2new += v.oovv
    t2new += np.transpose(tmp0, (0, 1, 3, 2)) * -1
    t2new += np.transpose(tmp0, (1, 0, 3, 2))
    del tmp0
    t2new += tmp2
    t2new += np.transpose(tmp2, (0, 1, 3, 2)) * -1
    del tmp2
    t2new += tmp4 * -0.5
    t2new += np.transpose(tmp4, (0, 1, 3, 2)) * 0.5
    del tmp4
    t2new += np.transpose(tmp6, (0, 1, 3, 2)) * 0.5
    t2new += np.transpose(tmp6, (1, 0, 3, 2)) * -0.5
    del tmp6
    t2new += np.transpose(tmp7, (1, 0, 2, 3))
    t2new += np.transpose(tmp7, (1, 0, 3, 2)) * -1
    del tmp7
    t2new += tmp8 * -1
    t2new += np.transpose(tmp8, (0, 1, 3, 2))
    t2new += np.transpose(tmp8, (1, 0, 2, 3))
    t2new += np.transpose(tmp8, (1, 0, 3, 2)) * -1
    del tmp8
    t2new += einsum(t2, (0, 1, 2, 3), tmp9, (0, 1, 4, 5), (4, 5, 2, 3)) * 0.5
    del tmp9

    return {f"t2new": t2new}

def update_lams(f=None, l2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:09.828872.

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

    tmp12 = einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4))
    tmp10 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    tmp2 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    tmp5 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 2, 3), (4, 0))
    tmp7 = einsum(t2, (0, 1, 2, 3), l2, (2, 3, 4, 1), (4, 0))
    tmp13 = einsum(v.oovv, (0, 1, 2, 3), tmp12, (4, 3), (0, 1, 4, 2)) * -1
    del tmp12
    tmp11 = einsum(tmp10, (0, 1), l2, (2, 0, 3, 4), (3, 4, 2, 1))
    del tmp10
    tmp3 = np.copy(np.transpose(v.ovov, (0, 2, 3, 1))) * -1
    tmp3 += tmp2
    del tmp2
    tmp6 = einsum(l2, (0, 1, 2, 3), tmp5, (3, 4), (2, 4, 0, 1))
    del tmp5
    tmp8 = einsum(tmp7, (0, 1), v.oovv, (2, 1, 3, 4), (0, 2, 3, 4)) * -1
    del tmp7
    tmp14 = np.copy(tmp11) * 0.5
    del tmp11
    tmp14 += tmp13 * 0.5
    del tmp13
    tmp4 = einsum(l2, (0, 1, 2, 3), tmp3, (3, 4, 1, 5), (2, 4, 0, 5))
    del tmp3
    tmp1 = einsum(l2, (0, 1, 2, 3), f.oo, (4, 3), (4, 2, 0, 1))
    tmp15 = einsum(l2, (0, 1, 2, 3), f.vv, (4, 1), (2, 3, 4, 0))
    tmp16 = np.copy(v.oooo) * 2
    tmp16 += einsum(v.oovv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (4, 5, 0, 1))
    tmp0 = einsum(t2, (0, 1, 2, 3), l2, (2, 3, 4, 5), (4, 5, 0, 1))
    tmp9 = np.copy(np.transpose(tmp6, (0, 1, 3, 2))) * -0.5
    del tmp6
    tmp9 += np.transpose(tmp8, (0, 1, 3, 2)) * -0.5
    del tmp8
    l2new = einsum(v.vvvv, (0, 1, 2, 3), l2, (2, 3, 4, 5), (0, 1, 4, 5)) * 0.5
    l2new += np.transpose(v.oovv, (2, 3, 0, 1))
    l2new += einsum(v.oovv, (0, 1, 2, 3), tmp0, (4, 5, 1, 0), (2, 3, 4, 5)) * -0.25
    del tmp0
    l2new += np.transpose(tmp1, (3, 2, 0, 1)) * -1
    l2new += np.transpose(tmp1, (3, 2, 1, 0))
    del tmp1
    l2new += np.transpose(tmp4, (2, 3, 0, 1))
    l2new += np.transpose(tmp4, (3, 2, 0, 1)) * -1
    l2new += np.transpose(tmp4, (2, 3, 1, 0)) * -1
    l2new += np.transpose(tmp4, (3, 2, 1, 0))
    del tmp4
    l2new += np.transpose(tmp9, (3, 2, 0, 1))
    l2new += np.transpose(tmp9, (3, 2, 1, 0)) * -1
    del tmp9
    l2new += np.transpose(tmp14, (2, 3, 1, 0))
    l2new += np.transpose(tmp14, (3, 2, 1, 0)) * -1
    del tmp14
    l2new += np.transpose(tmp15, (2, 3, 1, 0))
    l2new += np.transpose(tmp15, (3, 2, 1, 0)) * -1
    del tmp15
    l2new += einsum(tmp16, (0, 1, 2, 3), l2, (4, 5, 0, 1), (4, 5, 2, 3)) * 0.25
    del tmp16

    return {f"l2new": l2new}

def make_rdm1_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:09.980310.

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
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:11.493025.

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
    tmp2 = einsum(t2, (0, 1, 2, 3), l2, (4, 3, 5, 1), (5, 0, 4, 2))
    tmp4 = einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4))
    tmp1 = einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    tmp3 = einsum(t2, (0, 1, 2, 3), tmp2, (1, 4, 3, 5), (0, 4, 2, 5))
    tmp5 = einsum(tmp4, (0, 1), t2, (2, 3, 4, 0), (2, 3, 4, 1))
    tmp6 = einsum(tmp1, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4))
    tmp0 = einsum(t2, (0, 1, 2, 3), l2, (2, 3, 4, 5), (4, 5, 0, 1))
    rdm2.vvvv = einsum(t2, (0, 1, 2, 3), l2, (4, 5, 0, 1), (4, 5, 2, 3)) * 0.5
    rdm2.vvoo = np.copy(l2)
    rdm2.vovo = einsum(tmp4, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3)) * 0.5
    rdm2.vovo += np.transpose(tmp2, (2, 1, 3, 0)) * -1
    rdm2.voov = einsum(tmp4, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -0.5
    rdm2.voov += np.transpose(tmp2, (2, 1, 0, 3))
    rdm2.ovvo = einsum(tmp4, (0, 1), delta.oo, (2, 3), (2, 0, 1, 3)) * -0.5
    rdm2.ovvo += np.transpose(tmp2, (1, 2, 3, 0))
    rdm2.ovov = einsum(tmp4, (0, 1), delta.oo, (2, 3), (2, 0, 3, 1)) * 0.5
    del tmp4
    rdm2.ovov += np.transpose(tmp2, (1, 2, 0, 3)) * -1
    del tmp2
    rdm2.oovv = einsum(tmp0, (0, 1, 2, 3), t2, (1, 0, 4, 5), (3, 2, 4, 5)) * 0.25
    rdm2.oovv += t2
    rdm2.oovv += tmp3
    rdm2.oovv += np.transpose(tmp3, (0, 1, 3, 2)) * -1
    del tmp3
    rdm2.oovv += tmp5 * -0.5
    rdm2.oovv += np.transpose(tmp5, (0, 1, 3, 2)) * 0.5
    del tmp5
    rdm2.oovv += np.transpose(tmp6, (0, 1, 3, 2)) * 0.5
    rdm2.oovv += np.transpose(tmp6, (1, 0, 3, 2)) * -0.5
    del tmp6
    rdm2.oooo = np.copy(np.transpose(tmp0, (3, 2, 1, 0))) * 0.5
    del tmp0
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (1, 2, 0, 3)) * -0.5
    rdm2.oooo += einsum(tmp1, (0, 1), delta.oo, (2, 3), (2, 1, 0, 3)) * 0.5
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp1, (2, 3), (3, 0, 1, 2)) * 0.5
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp1, (2, 3), (0, 3, 1, 2)) * -0.5
    del tmp1
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (2, 0, 3, 1))
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1
    del delta
    rdm2.ooov = np.zeros((t2.shape[0], t2.shape[0], t2.shape[0], t2.shape[-1]))
    rdm2.oovo = np.zeros((t2.shape[0], t2.shape[0], t2.shape[-1], t2.shape[0]))
    rdm2.ovoo = np.zeros((t2.shape[0], t2.shape[-1], t2.shape[0], t2.shape[0]))
    rdm2.vooo = np.zeros((t2.shape[-1], t2.shape[0], t2.shape[0], t2.shape[0]))
    rdm2.ovvv = np.zeros((t2.shape[0], t2.shape[-1], t2.shape[-1], t2.shape[-1]))
    rdm2.vovv = np.zeros((t2.shape[-1], t2.shape[0], t2.shape[-1], t2.shape[-1]))
    rdm2.vvov = np.zeros((t2.shape[-1], t2.shape[-1], t2.shape[0], t2.shape[-1]))
    rdm2.vvvo = np.zeros((t2.shape[-1], t2.shape[-1], t2.shape[-1], t2.shape[0]))
    rdm2 = pack_2e(rdm2.oooo, rdm2.ooov, rdm2.oovo, rdm2.ovoo, rdm2.vooo, rdm2.oovv, rdm2.ovov, rdm2.ovvo, rdm2.voov, rdm2.vovo, rdm2.vvoo, rdm2.ovvv, rdm2.vovv, rdm2.vvov, rdm2.vvvo, rdm2.vvvv)
    rdm2 = np.transpose(rdm2, (0, 2, 1, 3))

    return rdm2

def hbar_matvec_ip_intermediates(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:15.330483.

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
    tmp0 : array
    tmp11 : array
    tmp12 : array
    tmp13 : array
    tmp2 : array
    tmp3 : array
    tmp5 : array
    tmp6 : array
    tmp8 : array
    tmp9 : array
    """

    tmp13 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    tmp12 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    tmp11 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (4, 5, 0, 1))
    tmp9 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (4, 5, 0, 1))
    tmp8 = einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 2, 3), (0, 4))
    tmp6 = einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    tmp5 = einsum(t2, (0, 1, 2, 3), v.ovoo, (4, 3, 0, 1), (4, 2))
    tmp3 = einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp2 = einsum(f.ov, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    tmp0 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 2, 3), (4, 0))

    return {f"tmp0": tmp0, f"tmp11": tmp11, f"tmp12": tmp12, f"tmp13": tmp13, f"tmp2": tmp2, f"tmp3": tmp3, f"tmp5": tmp5, f"tmp6": tmp6, f"tmp8": tmp8, f"tmp9": tmp9}

def hbar_matvec_ip(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:15.342329.

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
    tmp3 = einsum(r1, (0,), ints.tmp6, (1, 0, 2, 3), (1, 2, 3))
    del ints.tmp6
    tmp2 = einsum(f.oo, (0, 1), r2, (2, 1, 3), (0, 2, 3))
    tmp6 = einsum(ints.tmp13, (0, 1, 2, 3), r2, (4, 1, 3), (0, 4, 2))
    del ints.tmp13
    tmp7 = einsum(v.ovov, (0, 1, 2, 3), r2, (4, 2, 1), (4, 0, 3))
    tmp5 = einsum(ints.tmp0, (0, 1), r2, (2, 1, 3), (0, 2, 3))
    tmp1 = einsum(r2, (0, 1, 2), v.oovv, (0, 1, 3, 2), (3,))
    tmp4 = np.copy(tmp2)
    del tmp2
    tmp4 += tmp3
    del tmp3
    tmp12 = np.copy(f.ov)
    tmp12 += ints.tmp2
    del ints.tmp2
    tmp12 += ints.tmp5 * -0.5
    del ints.tmp5
    tmp12 += ints.tmp8 * -0.5
    del ints.tmp8
    tmp8 = np.copy(tmp5) * -0.5
    del tmp5
    tmp8 += tmp6
    del tmp6
    tmp8 += tmp7
    del tmp7
    tmp10 = np.copy(f.vv)
    tmp10 += np.transpose(ints.tmp12, (1, 0)) * -0.5
    del ints.tmp12
    tmp11 = np.copy(np.transpose(ints.tmp9, (2, 0, 1, 3))) * 0.5
    del ints.tmp9
    tmp11 += np.transpose(v.ovoo, (0, 2, 3, 1))
    tmp11 += ints.tmp3 * -1
    del ints.tmp3
    tmp9 = np.copy(ints.tmp11) * 0.5
    del ints.tmp11
    tmp9 += v.oooo
    tmp0 = np.copy(f.oo) * 2
    tmp0 += np.transpose(ints.tmp0, (1, 0))
    del ints.tmp0
    r2new = einsum(tmp1, (0,), t2, (1, 2, 3, 0), (1, 2, 3)) * 0.5
    del tmp1
    r2new += tmp4
    r2new += np.transpose(tmp4, (1, 0, 2)) * -1
    del tmp4
    r2new += tmp8 * -1
    r2new += np.transpose(tmp8, (1, 0, 2))
    del tmp8
    r2new += einsum(tmp9, (0, 1, 2, 3), r2, (2, 3, 4), (0, 1, 4)) * 0.5
    del tmp9
    r2new += einsum(r2, (0, 1, 2), tmp10, (2, 3), (0, 1, 3))
    del tmp10
    r2new += einsum(r1, (0,), tmp11, (0, 1, 2, 3), (2, 1, 3))
    del tmp11
    r2new += einsum(tmp12, (0, 1), r1, (2,), (2, 0, 1))
    r2new += einsum(r1, (0,), tmp12, (1, 2), (1, 0, 2)) * -1
    del tmp12
    r1new = einsum(r2, (0, 1, 2), v.ovoo, (3, 2, 0, 1), (3,)) * -0.5
    r1new += einsum(r2, (0, 1, 2), f.ov, (1, 2), (0,))
    r1new += einsum(tmp0, (0, 1), r1, (0,), (1,)) * -0.5
    del tmp0

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_matvec_ea_intermediates(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:19.283901.

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
    tmp0 : array
    tmp10 : array
    tmp12 : array
    tmp2 : array
    tmp5 : array
    tmp6 : array
    tmp7 : array
    tmp8 : array
    """

    tmp12 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    tmp10 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 2, 3), (4, 0))
    tmp8 = einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 3), (0, 2, 4, 5))
    tmp7 = einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 2, 3), (0, 4))
    tmp6 = einsum(t2, (0, 1, 2, 3), v.ovoo, (4, 5, 0, 1), (4, 2, 3, 5))
    tmp5 = einsum(t2, (0, 1, 2, 3), v.ovoo, (4, 3, 0, 1), (4, 2))
    tmp2 = einsum(f.ov, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    tmp0 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))

    return {f"tmp0": tmp0, f"tmp10": tmp10, f"tmp12": tmp12, f"tmp2": tmp2, f"tmp5": tmp5, f"tmp6": tmp6, f"tmp7": tmp7, f"tmp8": tmp8}

def hbar_matvec_ea(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:19.294679.

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
    tmp5 = np.copy(f.vv) * 2
    tmp5 += np.transpose(ints.tmp0, (1, 0)) * -1
    tmp6 = einsum(tmp5, (0, 1), r2, (2, 0, 3), (3, 2, 1)) * -0.5
    del tmp5
    tmp4 = einsum(r2, (0, 1, 2), v.ovov, (3, 1, 2, 4), (3, 0, 4))
    tmp3 = einsum(ints.tmp12, (0, 1, 2, 3), r2, (4, 3, 1), (0, 2, 4))
    del ints.tmp12
    tmp1 = einsum(v.oovv, (0, 1, 2, 3), r2, (2, 3, 4), (4, 0, 1))
    tmp10 = np.copy(f.oo)
    tmp10 += np.transpose(ints.tmp10, (1, 0)) * 0.5
    del ints.tmp10
    tmp2 = einsum(r1, (0,), ints.tmp8, (1, 2, 3, 0), (1, 2, 3))
    del ints.tmp8
    tmp11 = np.copy(f.ov)
    tmp11 += ints.tmp2
    del ints.tmp2
    tmp11 += ints.tmp5 * -0.5
    del ints.tmp5
    tmp11 += ints.tmp7 * -0.5
    del ints.tmp7
    tmp9 = einsum(r1, (0,), f.ov, (1, 0), (1,))
    tmp9 += einsum(v.oovv, (0, 1, 2, 3), r2, (2, 3, 1), (0,)) * 0.5
    tmp8 = np.copy(ints.tmp6) * 0.5
    del ints.tmp6
    tmp8 += np.transpose(v.ovvv, (0, 2, 3, 1))
    tmp7 = np.copy(tmp3)
    del tmp3
    tmp7 += tmp4
    del tmp4
    tmp7 += np.transpose(tmp6, (0, 2, 1)) * -1
    del tmp6
    tmp0 = np.copy(f.vv)
    tmp0 += np.transpose(ints.tmp0, (1, 0)) * -0.5
    del ints.tmp0
    r2new = einsum(tmp1, (0, 1, 2), t2, (2, 1, 3, 4), (3, 4, 0)) * -0.25
    del tmp1
    r2new += einsum(r2, (0, 1, 2), v.vvvv, (3, 4, 0, 1), (3, 4, 2)) * 0.5
    r2new += np.transpose(tmp2, (1, 2, 0))
    r2new += np.transpose(tmp2, (2, 1, 0)) * -1
    del tmp2
    r2new += np.transpose(tmp7, (1, 2, 0)) * -1
    r2new += np.transpose(tmp7, (2, 1, 0))
    del tmp7
    r2new += einsum(r1, (0,), tmp8, (1, 2, 3, 0), (3, 2, 1))
    del tmp8
    r2new += einsum(tmp9, (0,), t2, (1, 0, 2, 3), (2, 3, 1))
    del tmp9
    r2new += einsum(tmp10, (0, 1), r2, (2, 3, 0), (2, 3, 1)) * -1
    del tmp10
    r2new += einsum(tmp11, (0, 1), r1, (2,), (2, 1, 0))
    r2new += einsum(tmp11, (0, 1), r1, (2,), (1, 2, 0)) * -1
    del tmp11
    r1new = einsum(v.ovvv, (0, 1, 2, 3), r2, (2, 3, 0), (1,)) * -0.5
    r1new += einsum(f.ov, (0, 1), r2, (2, 1, 0), (2,))
    r1new += einsum(tmp0, (0, 1), r1, (0,), (1,))
    del tmp0

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_matvec_ee_intermediates(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:29.230560.

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
    tmp0 : array
    tmp1 : array
    tmp13 : array
    tmp14 : array
    tmp16 : array
    tmp20 : array
    tmp21 : array
    tmp23 : array
    tmp28 : array
    tmp30 : array
    tmp5 : array
    tmp6 : array
    """

    tmp30 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    tmp28 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (4, 5, 0, 1))
    tmp23 = einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 3), (0, 2, 4, 5))
    tmp21 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (4, 5, 0, 1))
    tmp20 = einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 2, 3), (0, 4))
    tmp16 = einsum(t2, (0, 1, 2, 3), v.ovoo, (4, 5, 0, 1), (4, 2, 3, 5))
    tmp14 = einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    tmp13 = einsum(t2, (0, 1, 2, 3), v.ovoo, (4, 3, 0, 1), (4, 2))
    tmp6 = einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp5 = einsum(f.ov, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    tmp1 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    tmp0 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 2, 3), (4, 0))

    return {f"tmp0": tmp0, f"tmp1": tmp1, f"tmp13": tmp13, f"tmp14": tmp14, f"tmp16": tmp16, f"tmp20": tmp20, f"tmp21": tmp21, f"tmp23": tmp23, f"tmp28": tmp28, f"tmp30": tmp30, f"tmp5": tmp5, f"tmp6": tmp6}

def hbar_matvec_ee(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:29.249522.

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
    tmp19 = einsum(r1, (0, 1), f.ov, (2, 1), (2, 0))
    tmp20 = einsum(v.oovv, (0, 1, 2, 3), r2, (4, 1, 2, 3), (4, 0))
    tmp10 = einsum(r2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    tmp17 = np.copy(f.oo) * 2
    tmp17 += np.transpose(ints.tmp0, (1, 0))
    tmp21 = np.copy(np.transpose(tmp19, (1, 0)))
    del tmp19
    tmp21 += tmp20 * 0.5
    del tmp20
    tmp30 = einsum(v.ooov, (0, 1, 2, 3), r1, (1, 3), (0, 2))
    tmp11 = einsum(tmp10, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4)) * -1
    del tmp10
    tmp9 = einsum(r1, (0, 1), ints.tmp6, (0, 2, 3, 4), (2, 3, 4, 1))
    del ints.tmp6
    tmp6 = einsum(v.ovov, (0, 1, 2, 3), r2, (4, 2, 5, 1), (4, 0, 5, 3))
    tmp4 = einsum(r1, (0, 1), ints.tmp14, (2, 0, 3, 4), (2, 3, 4, 1))
    del ints.tmp14
    tmp7 = einsum(r2, (0, 1, 2, 3), ints.tmp30, (4, 1, 5, 3), (4, 0, 5, 2))
    del ints.tmp30
    tmp5 = einsum(ints.tmp23, (0, 1, 2, 3), r1, (4, 3), (0, 4, 1, 2))
    del ints.tmp23
    tmp14 = einsum(f.vv, (0, 1), r2, (2, 3, 4, 1), (2, 3, 0, 4))
    tmp13 = einsum(v.ooov, (0, 1, 2, 3), r1, (2, 4), (0, 1, 4, 3))
    tmp16 = einsum(ints.tmp16, (0, 1, 2, 3), r1, (4, 3), (0, 4, 1, 2))
    del ints.tmp16
    tmp18 = einsum(tmp17, (0, 1), r2, (2, 0, 3, 4), (2, 1, 3, 4)) * -0.5
    del tmp17
    tmp22 = einsum(tmp21, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4)) * -1
    del tmp21
    tmp27 = einsum(v.ovvv, (0, 1, 2, 3), r1, (0, 3), (1, 2))
    tmp29 = einsum(v.ovvv, (0, 1, 2, 3), r1, (4, 1), (4, 0, 2, 3))
    tmp31 = einsum(tmp30, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4)) * -1
    del tmp30
    tmp24 = einsum(r1, (0, 1), ints.tmp21, (2, 3, 0, 4), (2, 3, 4, 1))
    del ints.tmp21
    tmp25 = einsum(r2, (0, 1, 2, 3), ints.tmp1, (4, 3), (0, 1, 4, 2))
    tmp12 = np.copy(tmp9)
    del tmp9
    tmp12 += tmp11 * 0.5
    del tmp11
    tmp3 = einsum(v.oovv, (0, 1, 2, 3), r2, (4, 5, 2, 3), (4, 5, 0, 1))
    tmp8 = np.copy(tmp4)
    del tmp4
    tmp8 += tmp5
    del tmp5
    tmp8 += tmp6
    del tmp6
    tmp8 += tmp7 * -1
    del tmp7
    tmp15 = np.copy(np.transpose(tmp13, (1, 0, 2, 3))) * -1
    del tmp13
    tmp15 += np.transpose(tmp14, (1, 0, 2, 3)) * -1
    del tmp14
    tmp23 = np.copy(np.transpose(tmp16, (0, 1, 3, 2))) * -0.5
    del tmp16
    tmp23 += np.transpose(tmp18, (1, 0, 3, 2))
    del tmp18
    tmp23 += np.transpose(tmp22, (1, 0, 3, 2))
    del tmp22
    tmp34 = np.copy(f.ov)
    tmp34 += ints.tmp5
    tmp34 += ints.tmp13 * -0.5
    tmp34 += ints.tmp20 * -0.5
    tmp28 = einsum(t2, (0, 1, 2, 3), tmp27, (4, 3), (0, 1, 2, 4)) * -1
    del tmp27
    tmp33 = np.copy(ints.tmp28)
    del ints.tmp28
    tmp33 += v.oooo * 2
    tmp32 = np.copy(np.transpose(tmp29, (0, 1, 3, 2))) * -1
    del tmp29
    tmp32 += np.transpose(tmp31, (0, 1, 3, 2))
    del tmp31
    tmp35 = np.copy(f.ov) * 2
    tmp35 += ints.tmp5 * 2
    del ints.tmp5
    tmp35 += ints.tmp13 * -1
    del ints.tmp13
    tmp35 += ints.tmp20 * -1
    del ints.tmp20
    tmp26 = np.copy(tmp24) * 0.5
    del tmp24
    tmp26 += np.transpose(tmp25, (1, 0, 2, 3)) * -0.5
    del tmp25
    tmp1 = np.copy(f.vv)
    tmp1 += np.transpose(ints.tmp1, (1, 0)) * -0.5
    del ints.tmp1
    tmp2 = np.copy(f.oo)
    tmp2 += np.transpose(ints.tmp0, (1, 0)) * 0.5
    del ints.tmp0
    tmp0 = einsum(v.oovv, (0, 1, 2, 3), r1, (1, 3), (0, 2))
    r2new = einsum(v.vvvv, (0, 1, 2, 3), r2, (4, 5, 2, 3), (4, 5, 0, 1)) * 0.5
    r2new += einsum(tmp3, (0, 1, 2, 3), t2, (3, 2, 4, 5), (1, 0, 4, 5)) * 0.25
    del tmp3
    r2new += tmp8 * -1
    r2new += np.transpose(tmp8, (0, 1, 3, 2))
    r2new += np.transpose(tmp8, (1, 0, 2, 3))
    r2new += np.transpose(tmp8, (1, 0, 3, 2)) * -1
    del tmp8
    r2new += tmp12 * -1
    r2new += np.transpose(tmp12, (0, 1, 3, 2))
    del tmp12
    r2new += tmp15 * -1
    r2new += np.transpose(tmp15, (0, 1, 3, 2))
    del tmp15
    r2new += tmp23
    r2new += np.transpose(tmp23, (1, 0, 2, 3)) * -1
    del tmp23
    r2new += np.transpose(tmp26, (1, 0, 2, 3)) * -1
    r2new += np.transpose(tmp26, (1, 0, 3, 2))
    del tmp26
    r2new += tmp28
    r2new += np.transpose(tmp28, (0, 1, 3, 2)) * -1
    del tmp28
    r2new += tmp32 * -1
    r2new += np.transpose(tmp32, (1, 0, 2, 3))
    del tmp32
    r2new += einsum(r2, (0, 1, 2, 3), tmp33, (4, 5, 0, 1), (4, 5, 2, 3)) * 0.25
    del tmp33
    r2new += einsum(tmp34, (0, 1), r1, (2, 3), (0, 2, 1, 3))
    r2new += einsum(r1, (0, 1), tmp34, (2, 3), (0, 2, 1, 3))
    r2new += einsum(r1, (0, 1), tmp35, (2, 3), (0, 2, 3, 1)) * -0.5
    del tmp35
    r2new += einsum(tmp34, (0, 1), r1, (2, 3), (0, 2, 3, 1)) * -1
    del tmp34
    r1new = einsum(v.ovvv, (0, 1, 2, 3), r2, (4, 0, 2, 3), (4, 1)) * -0.5
    r1new += einsum(t2, (0, 1, 2, 3), tmp0, (1, 3), (0, 2))
    del tmp0
    r1new += einsum(r2, (0, 1, 2, 3), v.ovoo, (4, 3, 0, 1), (4, 2)) * -0.5
    r1new += einsum(r2, (0, 1, 2, 3), f.ov, (1, 3), (0, 2))
    r1new += einsum(v.ovov, (0, 1, 2, 3), r1, (2, 1), (0, 3)) * -1
    r1new += einsum(r1, (0, 1), tmp1, (1, 2), (0, 2))
    del tmp1
    r1new += einsum(tmp2, (0, 1), r1, (0, 2), (1, 2)) * -1
    del tmp2

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_lmatvec_ip_intermediates(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:32.772040.

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
    tmp0 : array
    tmp1 : array
    tmp10 : array
    tmp11 : array
    tmp12 : array
    tmp2 : array
    tmp3 : array
    tmp4 : array
    tmp5 : array
    tmp6 : array
    """

    tmp12 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    tmp11 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    tmp10 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (4, 5, 0, 1))
    tmp6 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (4, 5, 0, 1))
    tmp5 = einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 2, 3), (0, 4))
    tmp4 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 2, 5))
    tmp3 = einsum(t2, (0, 1, 2, 3), v.ovoo, (4, 3, 0, 1), (4, 2))
    tmp2 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 2, 3), (4, 0))
    tmp1 = einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp0 = einsum(f.ov, (0, 1), t2, (2, 0, 3, 1), (2, 3))

    return {f"tmp0": tmp0, f"tmp1": tmp1, f"tmp10": tmp10, f"tmp11": tmp11, f"tmp12": tmp12, f"tmp2": tmp2, f"tmp3": tmp3, f"tmp4": tmp4, f"tmp5": tmp5, f"tmp6": tmp6}

def hbar_lmatvec_ip(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:32.783748.

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
    tmp4 = einsum(ints.tmp2, (0, 1), r2, (2, 0, 3), (1, 2, 3))
    tmp5 = einsum(ints.tmp12, (0, 1, 2, 3), r2, (4, 0, 2), (1, 4, 3))
    del ints.tmp12
    tmp6 = einsum(v.ovov, (0, 1, 2, 3), r2, (4, 2, 1), (4, 0, 3))
    tmp3 = einsum(r2, (0, 1, 2), t2, (0, 1, 3, 2), (3,))
    tmp8 = einsum(f.oo, (0, 1), r2, (2, 1, 3), (0, 2, 3))
    tmp10 = np.copy(f.vv)
    tmp10 += ints.tmp11 * -0.5
    del ints.tmp11
    tmp9 = np.copy(ints.tmp10) * 0.5
    del ints.tmp10
    tmp9 += v.oooo
    tmp7 = einsum(r1, (0,), f.ov, (1, 2), (1, 0, 2))
    tmp7 += tmp4 * -0.5
    del tmp4
    tmp7 += tmp5
    del tmp5
    tmp7 += tmp6
    del tmp6
    tmp1 = np.copy(f.ov)
    tmp1 += ints.tmp0
    del ints.tmp0
    tmp1 += ints.tmp3 * -0.5
    del ints.tmp3
    tmp1 += ints.tmp5 * -0.5
    del ints.tmp5
    tmp2 = np.copy(f.oo) * 2
    tmp2 += ints.tmp2
    del ints.tmp2
    tmp0 = np.copy(ints.tmp1) * 0.5
    del ints.tmp1
    tmp0 += np.transpose(ints.tmp4, (1, 0, 2, 3))
    del ints.tmp4
    tmp0 += np.transpose(ints.tmp6, (2, 0, 1, 3)) * -0.25
    del ints.tmp6
    tmp0 += np.transpose(v.ovoo, (0, 2, 3, 1)) * -0.5
    r2new = einsum(v.oovv, (0, 1, 2, 3), tmp3, (3,), (0, 1, 2)) * 0.5
    del tmp3
    r2new += einsum(r1, (0,), v.ooov, (1, 2, 0, 3), (1, 2, 3)) * -1
    r2new += tmp7 * -1
    r2new += np.transpose(tmp7, (1, 0, 2))
    del tmp7
    r2new += tmp8
    r2new += np.transpose(tmp8, (1, 0, 2)) * -1
    del tmp8
    r2new += einsum(tmp9, (0, 1, 2, 3), r2, (0, 1, 4), (2, 3, 4)) * 0.5
    del tmp9
    r2new += einsum(r2, (0, 1, 2), tmp10, (2, 3), (0, 1, 3))
    del tmp10
    r1new = einsum(tmp0, (0, 1, 2, 3), r2, (2, 1, 3), (0,)) * -1
    del tmp0
    r1new += einsum(r2, (0, 1, 2), tmp1, (1, 2), (0,))
    del tmp1
    r1new += einsum(r1, (0,), tmp2, (0, 1), (1,)) * -0.5
    del tmp2

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_lmatvec_ea_intermediates(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:36.434478.

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
    tmp0 : array
    tmp11 : array
    tmp2 : array
    tmp3 : array
    tmp4 : array
    tmp5 : array
    tmp6 : array
    tmp9 : array
    """

    tmp11 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    tmp9 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 2, 3), (4, 0))
    tmp6 = einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 3), (0, 2, 4, 5))
    tmp5 = einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 2, 3), (0, 4))
    tmp4 = einsum(t2, (0, 1, 2, 3), v.ovoo, (4, 5, 0, 1), (4, 2, 3, 5))
    tmp3 = einsum(t2, (0, 1, 2, 3), v.ovoo, (4, 3, 0, 1), (4, 2))
    tmp2 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    tmp0 = einsum(f.ov, (0, 1), t2, (2, 0, 3, 1), (2, 3))

    return {f"tmp0": tmp0, f"tmp11": tmp11, f"tmp2": tmp2, f"tmp3": tmp3, f"tmp4": tmp4, f"tmp5": tmp5, f"tmp6": tmp6, f"tmp9": tmp9}

def hbar_lmatvec_ea(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:36.445025.

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
    tmp7 = np.copy(f.vv) * 2
    tmp7 += ints.tmp2 * -1
    tmp8 = einsum(r2, (0, 1, 2), tmp7, (1, 3), (2, 0, 3)) * -0.5
    del tmp7
    tmp5 = einsum(r2, (0, 1, 2), ints.tmp11, (2, 3, 1, 4), (3, 4, 0))
    del ints.tmp11
    tmp6 = einsum(r2, (0, 1, 2), v.ovov, (3, 1, 2, 4), (3, 0, 4))
    tmp4 = einsum(r2, (0, 1, 2), t2, (3, 4, 0, 1), (2, 3, 4))
    tmp10 = np.copy(f.oo)
    tmp10 += ints.tmp9 * 0.5
    del ints.tmp9
    tmp0 = einsum(r2, (0, 1, 2), t2, (3, 2, 0, 1), (3,))
    tmp9 = einsum(r1, (0,), f.ov, (1, 2), (1, 2, 0))
    tmp9 += tmp5
    del tmp5
    tmp9 += tmp6
    del tmp6
    tmp9 += np.transpose(tmp8, (0, 2, 1)) * -1
    del tmp8
    tmp1 = np.copy(ints.tmp6)
    del ints.tmp6
    tmp1 += ints.tmp4 * -0.25
    del ints.tmp4
    tmp1 += np.transpose(v.ovvv, (0, 2, 3, 1)) * -0.5
    tmp3 = np.copy(f.vv)
    tmp3 += ints.tmp2 * -0.5
    del ints.tmp2
    tmp2 = np.copy(f.ov)
    tmp2 += ints.tmp0
    del ints.tmp0
    tmp2 += ints.tmp3 * -0.5
    del ints.tmp3
    tmp2 += ints.tmp5 * -0.5
    del ints.tmp5
    r2new = einsum(v.oovv, (0, 1, 2, 3), tmp4, (4, 1, 0), (2, 3, 4)) * -0.25
    del tmp4
    r2new += einsum(v.oovv, (0, 1, 2, 3), tmp0, (1,), (2, 3, 0)) * 0.5
    r2new += einsum(r2, (0, 1, 2), v.vvvv, (3, 4, 0, 1), (3, 4, 2)) * 0.5
    r2new += einsum(r1, (0,), v.ovvv, (1, 0, 2, 3), (2, 3, 1)) * -1
    r2new += np.transpose(tmp9, (1, 2, 0)) * -1
    r2new += np.transpose(tmp9, (2, 1, 0))
    del tmp9
    r2new += einsum(tmp10, (0, 1), r2, (2, 3, 0), (2, 3, 1)) * -1
    del tmp10
    r1new = einsum(tmp0, (0,), f.ov, (0, 1), (1,)) * -0.5
    del tmp0
    r1new += einsum(tmp1, (0, 1, 2, 3), r2, (2, 1, 0), (3,)) * -1
    del tmp1
    r1new += einsum(tmp2, (0, 1), r2, (2, 1, 0), (2,))
    del tmp2
    r1new += einsum(r1, (0,), tmp3, (0, 1), (1,))
    del tmp3

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_lmatvec_ee_intermediates(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:43.616564.

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
    tmp0 : array
    tmp1 : array
    tmp10 : array
    tmp11 : array
    tmp19 : array
    tmp21 : array
    tmp3 : array
    tmp4 : array
    tmp6 : array
    tmp7 : array
    tmp8 : array
    tmp9 : array
    """

    tmp21 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    tmp19 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (4, 5, 0, 1))
    tmp11 = einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 3), (0, 2, 4, 5))
    tmp10 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (4, 5, 0, 1))
    tmp9 = einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 2, 3), (0, 4))
    tmp8 = einsum(t2, (0, 1, 2, 3), v.ovoo, (4, 5, 0, 1), (4, 2, 3, 5))
    tmp7 = einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    tmp6 = einsum(t2, (0, 1, 2, 3), v.ovoo, (4, 3, 0, 1), (4, 2))
    tmp4 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    tmp3 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 2, 3), (4, 0))
    tmp1 = einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp0 = einsum(f.ov, (0, 1), t2, (2, 0, 3, 1), (2, 3))

    return {f"tmp0": tmp0, f"tmp1": tmp1, f"tmp10": tmp10, f"tmp11": tmp11, f"tmp19": tmp19, f"tmp21": tmp21, f"tmp3": tmp3, f"tmp4": tmp4, f"tmp6": tmp6, f"tmp7": tmp7, f"tmp8": tmp8, f"tmp9": tmp9}

def hbar_lmatvec_ee(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T15:23:43.634093.

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
    tmp7 = np.copy(f.oo) * 2
    tmp7 += ints.tmp3
    del ints.tmp3
    tmp1 = einsum(t2, (0, 1, 2, 3), r2, (4, 1, 2, 3), (4, 0))
    tmp13 = einsum(f.vv, (0, 1), r2, (2, 3, 4, 1), (2, 3, 0, 4))
    tmp12 = einsum(v.ooov, (0, 1, 2, 3), r1, (2, 4), (0, 1, 4, 3))
    tmp10 = einsum(v.ovov, (0, 1, 2, 3), r2, (4, 2, 5, 1), (4, 0, 5, 3))
    tmp9 = einsum(ints.tmp21, (0, 1, 2, 3), r2, (4, 0, 5, 2), (1, 4, 3, 5))
    del ints.tmp21
    tmp16 = einsum(tmp7, (0, 1), r2, (2, 0, 3, 4), (2, 1, 3, 4)) * -0.5
    tmp15 = einsum(v.oovv, (0, 1, 2, 3), tmp1, (4, 1), (4, 0, 2, 3)) * -1
    tmp2 = einsum(t2, (0, 1, 2, 3), r2, (0, 1, 4, 3), (4, 2))
    tmp14 = np.copy(np.transpose(tmp12, (1, 0, 2, 3))) * -1
    del tmp12
    tmp14 += np.transpose(tmp13, (1, 0, 2, 3)) * -1
    del tmp13
    tmp19 = einsum(v.ovvv, (0, 1, 2, 3), r1, (4, 1), (4, 0, 2, 3))
    tmp8 = einsum(t2, (0, 1, 2, 3), r2, (4, 5, 2, 3), (4, 5, 0, 1))
    tmp21 = np.copy(ints.tmp19)
    del ints.tmp19
    tmp21 += v.oooo * 2
    tmp11 = einsum(r1, (0, 1), f.ov, (2, 3), (2, 0, 3, 1))
    tmp11 += tmp9
    del tmp9
    tmp11 += tmp10 * -1
    del tmp10
    tmp20 = einsum(ints.tmp4, (0, 1), r2, (2, 3, 4, 0), (2, 3, 1, 4))
    tmp17 = np.copy(np.transpose(tmp15, (0, 1, 3, 2))) * 0.5
    del tmp15
    tmp17 += np.transpose(tmp16, (1, 0, 3, 2))
    del tmp16
    tmp18 = einsum(v.oovv, (0, 1, 2, 3), tmp2, (4, 3), (0, 1, 4, 2)) * -1
    tmp4 = np.copy(np.transpose(ints.tmp1, (1, 2, 0, 3)))
    del ints.tmp1
    tmp4 += np.transpose(ints.tmp7, (0, 2, 1, 3)) * 2
    del ints.tmp7
    tmp4 += ints.tmp10 * -0.5
    del ints.tmp10
    tmp4 += v.ooov * -1
    tmp3 = np.copy(ints.tmp8) * 0.25
    del ints.tmp8
    tmp3 += np.transpose(v.ovvv, (0, 2, 3, 1)) * 0.5
    tmp3 += ints.tmp11 * -1
    del ints.tmp11
    tmp5 = np.copy(f.ov)
    tmp5 += ints.tmp0
    del ints.tmp0
    tmp5 += ints.tmp6 * -0.5
    del ints.tmp6
    tmp5 += ints.tmp9 * -0.5
    del ints.tmp9
    tmp6 = np.copy(f.vv)
    tmp6 += ints.tmp4 * -0.5
    del ints.tmp4
    tmp0 = einsum(t2, (0, 1, 2, 3), r1, (1, 3), (0, 2))
    r2new = einsum(v.vvvv, (0, 1, 2, 3), r2, (4, 5, 2, 3), (4, 5, 0, 1)) * 0.5
    r2new += einsum(v.oovv, (0, 1, 2, 3), tmp8, (4, 5, 1, 0), (5, 4, 2, 3)) * 0.25
    del tmp8
    r2new += tmp11
    r2new += np.transpose(tmp11, (0, 1, 3, 2)) * -1
    r2new += np.transpose(tmp11, (1, 0, 2, 3)) * -1
    r2new += np.transpose(tmp11, (1, 0, 3, 2))
    del tmp11
    r2new += tmp14 * -1
    r2new += np.transpose(tmp14, (0, 1, 3, 2))
    del tmp14
    r2new += tmp17
    r2new += np.transpose(tmp17, (1, 0, 2, 3)) * -1
    del tmp17
    r2new += tmp18 * -0.5
    r2new += np.transpose(tmp18, (0, 1, 3, 2)) * 0.5
    del tmp18
    r2new += np.transpose(tmp19, (0, 1, 3, 2))
    r2new += np.transpose(tmp19, (1, 0, 3, 2)) * -1
    del tmp19
    r2new += np.transpose(tmp20, (1, 0, 2, 3)) * -0.5
    r2new += np.transpose(tmp20, (1, 0, 3, 2)) * 0.5
    del tmp20
    r2new += einsum(r2, (0, 1, 2, 3), tmp21, (0, 1, 4, 5), (4, 5, 2, 3)) * 0.25
    del tmp21
    r1new = einsum(v.oovv, (0, 1, 2, 3), tmp0, (1, 3), (0, 2))
    del tmp0
    r1new += einsum(f.ov, (0, 1), tmp1, (2, 0), (2, 1)) * -0.5
    r1new += einsum(v.ovvv, (0, 1, 2, 3), tmp2, (1, 3), (0, 2)) * 0.5
    del tmp2
    r1new += einsum(v.ooov, (0, 1, 2, 3), tmp1, (2, 1), (0, 3)) * 0.5
    del tmp1
    r1new += einsum(v.ovov, (0, 1, 2, 3), r1, (2, 1), (0, 3)) * -1
    r1new += einsum(r2, (0, 1, 2, 3), tmp3, (1, 2, 3, 4), (0, 4)) * -1
    del tmp3
    r1new += einsum(tmp4, (0, 1, 2, 3), r2, (1, 0, 4, 3), (2, 4)) * -0.5
    del tmp4
    r1new += einsum(r2, (0, 1, 2, 3), tmp5, (1, 3), (0, 2))
    del tmp5
    r1new += einsum(r1, (0, 1), tmp6, (1, 2), (0, 2))
    del tmp6
    r1new += einsum(tmp7, (0, 1), r1, (0, 2), (1, 2)) * -0.5
    del tmp7

    return {f"r1new": r1new, f"r2new": r2new}

