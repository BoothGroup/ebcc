"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-08-17T18:48:06.061718
  * python version: 3.10.12 (main, Jul 29 2024, 16:56:48) [GCC 11.4.0]
  * albert version: 0.0.0
  * caller: /home/ollie/git/albert/albert/codegen/einsum.py
  * node: ollie-desktop
  * system: Linux
  * processor: x86_64
  * release: 6.8.0-40-generic
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, dirsum, Namespace


def energy(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:06.214643.

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

    e_cc = einsum(v.ovov, (0, 1, 2, 3), t2, (0, 2, 1, 3), ()) * 2
    e_cc += einsum(v.ovov, (0, 1, 2, 3), t2, (0, 2, 3, 1), ()) * -1

    return e_cc

def update_amps(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:08.356370.

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

    tmp8 = np.copy(np.transpose(v.ovov, (0, 2, 3, 1)))
    tmp8 += np.transpose(v.ovov, (0, 2, 1, 3)) * -0.5
    tmp5 = np.copy(np.transpose(v.ovov, (0, 2, 3, 1))) * -0.5
    tmp5 += np.transpose(v.ovov, (0, 2, 1, 3))
    tmp2 = np.copy(np.transpose(v.ovov, (0, 2, 3, 1))) * 2
    tmp2 += np.transpose(v.ovov, (0, 2, 1, 3)) * -1
    tmp9 = einsum(tmp8, (0, 1, 2, 3), t2, (4, 0, 2, 3), (4, 1)) * 2
    del tmp8
    tmp6 = einsum(t2, (0, 1, 2, 3), tmp5, (0, 1, 4, 3), (2, 4))
    del tmp5
    tmp3 = einsum(tmp2, (0, 1, 2, 3), t2, (4, 0, 5, 3), (4, 1, 5, 2))
    del tmp2
    tmp0 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    tmp10 = einsum(t2, (0, 1, 2, 3), tmp9, (4, 1), (0, 4, 3, 2))
    del tmp9
    tmp7 = einsum(tmp6, (0, 1), t2, (2, 3, 1, 4), (2, 3, 4, 0)) * 2
    del tmp6
    tmp1 = einsum(t2, (0, 1, 2, 3), f.oo, (4, 1), (4, 0, 2, 3))
    tmp13 = np.copy(np.transpose(v.ovov, (0, 2, 1, 3))) * 2
    tmp13 += v.oovv * -1
    tmp13 += tmp3 * 2
    tmp12 = einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    tmp15 = np.copy(np.transpose(v.ovov, (0, 2, 1, 3))) * -1
    tmp15 += tmp0
    tmp16 = np.copy(v.oovv) * -1
    tmp16 += einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 1, 5), (4, 0, 5, 3))
    tmp14 = np.copy(v.oooo)
    tmp14 += einsum(v.ovov, (0, 1, 2, 3), t2, (4, 5, 1, 3), (0, 4, 2, 5))
    tmp11 = np.copy(np.transpose(tmp7, (1, 0, 2, 3)))
    del tmp7
    tmp11 += np.transpose(tmp10, (0, 1, 3, 2))
    del tmp10
    tmp4 = einsum(tmp3, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 5, 2))
    del tmp3
    t2new = einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new += np.transpose(v.ovov, (0, 2, 1, 3))
    t2new += einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 2, 5)) * -1
    t2new += np.transpose(tmp0, (1, 0, 3, 2)) * -1
    del tmp0
    t2new += einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 2, 5)) * -1
    t2new += einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 5, 3), (0, 4, 1, 5)) * 2
    t2new += np.transpose(tmp1, (0, 1, 3, 2)) * -1
    t2new += np.transpose(tmp1, (1, 0, 2, 3)) * -1
    del tmp1
    t2new += np.transpose(tmp4, (1, 0, 3, 2)) * -1
    t2new += tmp4 * -1
    del tmp4
    t2new += tmp11 * -1
    t2new += np.transpose(tmp11, (1, 0, 3, 2)) * -1
    del tmp11
    t2new += np.transpose(tmp12, (0, 1, 3, 2))
    t2new += np.transpose(tmp12, (1, 0, 2, 3))
    del tmp12
    t2new += einsum(tmp13, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 5, 2))
    del tmp13
    t2new += einsum(tmp14, (0, 1, 2, 3), t2, (0, 2, 4, 5), (1, 3, 4, 5))
    del tmp14
    t2new += einsum(tmp15, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 5, 2))
    del tmp15
    t2new += einsum(tmp16, (0, 1, 2, 3), t2, (4, 1, 3, 5), (0, 4, 5, 2))
    del tmp16

    return {f"t2new": t2new}

def update_lams(f=None, l2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:11.877391.

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

    tmp5 = np.copy(np.transpose(l2, (3, 2, 0, 1))) * -1
    tmp5 += np.transpose(l2, (2, 3, 0, 1)) * 2
    tmp15 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 1, 5), (4, 0, 5, 3))
    tmp10 = einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), (3, 4, 0, 5))
    tmp6 = einsum(tmp5, (0, 1, 2, 3), t2, (4, 0, 5, 2), (4, 1, 5, 3))
    del tmp5
    tmp4 = einsum(t2, (0, 1, 2, 3), l2, (4, 2, 5, 1), (5, 0, 4, 3))
    tmp26 = np.copy(np.transpose(t2, (0, 1, 3, 2))) * -0.5
    tmp26 += t2
    tmp29 = np.copy(np.transpose(t2, (0, 1, 3, 2)))
    tmp29 += t2 * -0.5
    tmp19 = np.copy(np.transpose(t2, (0, 1, 3, 2))) * 2
    tmp19 += t2 * -1
    tmp22 = np.copy(np.transpose(t2, (0, 1, 3, 2))) * -1
    tmp22 += t2 * 2
    tmp16 = np.copy(v.oovv) * -1
    tmp16 += tmp15
    del tmp15
    tmp11 = np.copy(tmp10)
    del tmp10
    tmp11 += np.transpose(l2, (3, 2, 0, 1)) * -1
    tmp11 += np.transpose(l2, (2, 3, 0, 1)) * 2
    tmp8 = np.copy(np.transpose(v.ovov, (0, 2, 3, 1))) * 2
    tmp8 += np.transpose(v.ovov, (0, 2, 1, 3)) * -1
    tmp7 = np.copy(tmp4)
    del tmp4
    tmp7 += np.transpose(tmp6, (1, 0, 3, 2)) * -1
    del tmp6
    tmp27 = einsum(l2, (0, 1, 2, 3), tmp26, (2, 3, 4, 1), (0, 4)) * 2
    del tmp26
    tmp30 = einsum(tmp29, (0, 1, 2, 3), l2, (3, 2, 0, 4), (4, 1)) * 2
    del tmp29
    tmp20 = einsum(v.ovov, (0, 1, 2, 3), tmp19, (0, 2, 3, 4), (1, 4)) * 0.5
    del tmp19
    tmp23 = einsum(v.ovov, (0, 1, 2, 3), tmp22, (2, 4, 3, 1), (0, 4))
    del tmp22
    tmp14 = einsum(f.vv, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    tmp17 = einsum(l2, (0, 1, 2, 3), tmp16, (2, 4, 1, 5), (3, 4, 0, 5))
    del tmp16
    tmp3 = einsum(v.oovv, (0, 1, 2, 3), l2, (4, 3, 5, 1), (5, 0, 4, 2))
    tmp12 = einsum(v.ovov, (0, 1, 2, 3), tmp11, (4, 2, 5, 3), (0, 4, 1, 5))
    del tmp11
    tmp9 = einsum(tmp8, (0, 1, 2, 3), tmp7, (4, 0, 5, 3), (4, 1, 5, 2))
    del tmp8, tmp7
    tmp28 = einsum(v.ovov, (0, 1, 2, 3), tmp27, (4, 1), (0, 2, 3, 4))
    del tmp27
    tmp31 = einsum(v.ovov, (0, 1, 2, 3), tmp30, (4, 2), (0, 4, 3, 1))
    del tmp30
    tmp21 = einsum(l2, (0, 1, 2, 3), tmp20, (4, 1), (3, 2, 0, 4)) * 2
    del tmp20
    tmp24 = einsum(tmp23, (0, 1), l2, (2, 3, 1, 4), (4, 0, 2, 3))
    del tmp23
    tmp1 = einsum(t2, (0, 1, 2, 3), l2, (2, 3, 4, 5), (4, 5, 0, 1))
    tmp18 = np.copy(tmp14)
    del tmp14
    tmp18 += tmp17
    del tmp17
    tmp13 = np.copy(tmp3)
    del tmp3
    tmp13 += tmp9
    del tmp9
    tmp13 += np.transpose(tmp12, (1, 0, 3, 2)) * -1
    del tmp12
    tmp32 = np.copy(np.transpose(tmp28, (1, 0, 3, 2)))
    del tmp28
    tmp32 += np.transpose(tmp31, (1, 0, 3, 2))
    del tmp31
    tmp0 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 5, 1, 3), (4, 5, 0, 2))
    tmp25 = np.copy(np.transpose(tmp21, (1, 0, 2, 3)))
    del tmp21
    tmp25 += np.transpose(tmp24, (0, 1, 3, 2))
    del tmp24
    tmp2 = einsum(l2, (0, 1, 2, 3), f.oo, (4, 3), (4, 2, 0, 1))
    l2new = einsum(l2, (0, 1, 2, 3), v.vvvv, (4, 1, 5, 0), (4, 5, 3, 2))
    l2new += np.transpose(v.ovov, (1, 3, 0, 2))
    l2new += einsum(l2, (0, 1, 2, 3), tmp0, (3, 2, 4, 5), (0, 1, 5, 4))
    del tmp0
    l2new += einsum(v.ovov, (0, 1, 2, 3), tmp1, (4, 5, 0, 2), (1, 3, 4, 5))
    del tmp1
    l2new += einsum(l2, (0, 1, 2, 3), v.oooo, (4, 2, 5, 3), (0, 1, 4, 5))
    l2new += np.transpose(tmp2, (3, 2, 0, 1)) * -1
    l2new += np.transpose(tmp2, (2, 3, 1, 0)) * -1
    del tmp2
    l2new += np.transpose(tmp13, (2, 3, 0, 1)) * -1
    l2new += np.transpose(tmp13, (3, 2, 1, 0)) * -1
    del tmp13
    l2new += np.transpose(tmp18, (3, 2, 0, 1))
    l2new += np.transpose(tmp18, (2, 3, 1, 0))
    del tmp18
    l2new += np.transpose(tmp25, (2, 3, 0, 1)) * -1
    l2new += np.transpose(tmp25, (3, 2, 1, 0)) * -1
    del tmp25
    l2new += np.transpose(tmp32, (3, 2, 0, 1)) * -1
    l2new += np.transpose(tmp32, (2, 3, 1, 0)) * -1
    del tmp32

    return {f"l2new": l2new}

def make_rdm1_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:12.171694.

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
    tmp0 = np.copy(np.transpose(t2, (0, 1, 3, 2)))
    tmp0 += t2 * -0.5
    rdm1.vv = einsum(l2, (0, 1, 2, 3), tmp0, (3, 2, 4, 1), (0, 4)) * 4
    rdm1.oo = np.copy(delta.oo) * 2
    del delta
    rdm1.oo += einsum(l2, (0, 1, 2, 3), tmp0, (2, 4, 1, 0), (4, 3)) * -4
    del tmp0
    rdm1.ov = np.zeros((t2.shape[0], t2.shape[-1]))
    rdm1.vo = np.zeros((t2.shape[-1], t2.shape[0]))
    rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])

    return rdm1

def make_rdm2_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:22.905206.

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
    tmp6 = np.copy(np.transpose(l2, (3, 2, 0, 1))) * -1
    tmp6 += np.transpose(l2, (2, 3, 0, 1))
    tmp3 = np.copy(np.transpose(l2, (3, 2, 0, 1))) * -1
    tmp3 += np.transpose(l2, (2, 3, 0, 1)) * 2
    tmp1 = np.copy(np.transpose(t2, (0, 1, 3, 2))) * -0.5
    tmp1 += t2
    tmp13 = np.copy(np.transpose(l2, (3, 2, 0, 1))) * 2
    tmp13 += np.transpose(l2, (2, 3, 0, 1)) * -1
    tmp7 = einsum(tmp6, (0, 1, 2, 3), t2, (4, 0, 2, 5), (4, 1, 5, 3))
    del tmp6
    tmp4 = einsum(tmp3, (0, 1, 2, 3), t2, (4, 0, 5, 2), (4, 1, 5, 3))
    tmp2 = einsum(tmp1, (0, 1, 2, 3), l2, (2, 3, 0, 4), (4, 1)) * 2
    del tmp1
    tmp14 = einsum(t2, (0, 1, 2, 3), tmp13, (0, 1, 3, 4), (2, 4)) * 0.5
    tmp27 = np.copy(np.transpose(l2, (3, 2, 0, 1)))
    tmp27 += np.transpose(l2, (2, 3, 0, 1)) * -1
    tmp22 = np.copy(np.transpose(t2, (0, 1, 3, 2)))
    tmp22 += t2 * -1
    tmp24 = np.copy(np.transpose(t2, (0, 1, 3, 2))) * 2
    tmp24 += t2 * -1
    tmp8 = einsum(tmp7, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 5, 2))
    tmp5 = einsum(tmp4, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 5, 2))
    tmp20 = einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), (3, 4, 0, 5))
    tmp16 = einsum(tmp2, (0, 1), t2, (2, 0, 3, 4), (2, 1, 4, 3))
    tmp15 = einsum(tmp14, (0, 1), t2, (2, 3, 1, 4), (2, 3, 4, 0)) * 2
    tmp18 = einsum(t2, (0, 1, 2, 3), l2, (4, 2, 5, 1), (5, 0, 4, 3))
    tmp0 = einsum(t2, (0, 1, 2, 3), l2, (2, 3, 4, 5), (4, 5, 0, 1))
    tmp10 = einsum(tmp3, (0, 1, 2, 3), t2, (4, 0, 2, 5), (4, 1, 5, 3))
    del tmp3
    tmp29 = einsum(t2, (0, 1, 2, 3), l2, (4, 5, 1, 0), (4, 5, 3, 2))
    tmp26 = einsum(t2, (0, 1, 2, 3), tmp13, (0, 1, 3, 4), (2, 4))
    del tmp13
    tmp28 = einsum(tmp27, (0, 1, 2, 3), t2, (4, 0, 2, 5), (4, 1, 5, 3))
    del tmp27
    tmp23 = einsum(l2, (0, 1, 2, 3), tmp22, (2, 4, 5, 1), (3, 4, 0, 5))
    del tmp22
    tmp25 = einsum(l2, (0, 1, 2, 3), tmp24, (3, 4, 5, 1), (2, 4, 0, 5))
    del tmp24
    tmp9 = np.copy(np.transpose(tmp5, (1, 0, 3, 2))) * 2
    tmp9 += tmp8
    del tmp8
    tmp21 = einsum(tmp20, (0, 1, 2, 3), t2, (4, 0, 2, 5), (1, 4, 3, 5))
    tmp17 = np.copy(np.transpose(tmp15, (1, 0, 2, 3)))
    del tmp15
    tmp17 += np.transpose(tmp16, (0, 1, 3, 2))
    del tmp16
    tmp19 = einsum(tmp18, (0, 1, 2, 3), t2, (4, 0, 2, 5), (1, 4, 3, 5))
    tmp12 = einsum(t2, (0, 1, 2, 3), tmp0, (1, 0, 4, 5), (4, 5, 3, 2))
    tmp11 = einsum(tmp10, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 5, 2))
    del tmp10
    rdm2.vvvv = np.copy(np.transpose(tmp29, (1, 0, 2, 3))) * -2
    rdm2.vvvv += np.transpose(tmp29, (1, 0, 3, 2)) * 4
    del tmp29
    rdm2.vvoo = np.copy(np.transpose(l2, (0, 1, 3, 2))) * -2
    rdm2.vvoo += l2 * 4
    rdm2.vovo = np.copy(np.transpose(tmp4, (3, 0, 2, 1))) * -2
    rdm2.vovo += np.transpose(tmp28, (3, 0, 2, 1)) * -2
    del tmp28
    rdm2.vovo += einsum(delta.oo, (0, 1), tmp26, (2, 3), (3, 0, 2, 1)) * 4
    rdm2.vovo += np.transpose(tmp20, (2, 1, 3, 0)) * -2
    rdm2.voov = np.copy(np.transpose(tmp25, (2, 1, 0, 3))) * 2
    rdm2.voov += np.transpose(tmp23, (2, 1, 0, 3)) * -2
    rdm2.voov += einsum(delta.oo, (0, 1), tmp26, (2, 3), (3, 0, 1, 2)) * -2
    del tmp26
    rdm2.voov += np.transpose(tmp18, (2, 1, 0, 3)) * -2
    rdm2.voov += np.transpose(tmp4, (3, 0, 1, 2)) * 2
    rdm2.ovvo = np.copy(np.transpose(tmp4, (0, 3, 2, 1))) * 4
    del tmp4
    rdm2.ovvo += np.transpose(tmp7, (0, 3, 2, 1)) * -2
    del tmp7
    rdm2.ovvo += einsum(tmp14, (0, 1), delta.oo, (2, 3), (2, 1, 0, 3)) * -4
    rdm2.ovvo += np.transpose(tmp18, (1, 2, 3, 0)) * -2
    del tmp18
    rdm2.ovov = np.copy(np.transpose(tmp23, (1, 2, 0, 3))) * 2
    del tmp23
    rdm2.ovov += np.transpose(tmp25, (1, 2, 0, 3)) * -2
    del tmp25
    rdm2.ovov += einsum(tmp14, (0, 1), delta.oo, (2, 3), (2, 1, 3, 0)) * 8
    del tmp14
    rdm2.ovov += np.transpose(tmp20, (1, 2, 0, 3)) * -2
    del tmp20
    rdm2.oovv = np.copy(tmp9) * 2
    rdm2.oovv += np.transpose(tmp9, (0, 1, 3, 2)) * -2
    del tmp9
    rdm2.oovv += tmp11 * -4
    rdm2.oovv += np.transpose(tmp11, (0, 1, 3, 2)) * 2
    rdm2.oovv += np.transpose(tmp11, (1, 0, 2, 3)) * 2
    rdm2.oovv += np.transpose(tmp11, (1, 0, 3, 2)) * -4
    del tmp11
    rdm2.oovv += tmp12 * 4
    rdm2.oovv += np.transpose(tmp12, (0, 1, 3, 2)) * -2
    del tmp12
    rdm2.oovv += tmp17 * -4
    rdm2.oovv += np.transpose(tmp17, (0, 1, 3, 2)) * 2
    rdm2.oovv += np.transpose(tmp17, (1, 0, 2, 3)) * 2
    rdm2.oovv += np.transpose(tmp17, (1, 0, 3, 2)) * -4
    del tmp17
    rdm2.oovv += np.transpose(t2, (0, 1, 3, 2)) * -2
    rdm2.oovv += t2 * 4
    rdm2.oovv += tmp19 * 2
    del tmp19
    rdm2.oovv += np.transpose(tmp21, (0, 1, 3, 2)) * 2
    del tmp21
    rdm2.oovv += np.transpose(tmp5, (1, 0, 3, 2)) * 4
    del tmp5
    rdm2.oooo = np.copy(np.transpose(tmp0, (2, 3, 1, 0))) * -2
    rdm2.oooo += np.transpose(tmp0, (3, 2, 1, 0)) * 4
    del tmp0
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp2, (2, 3), (3, 0, 2, 1)) * -4
    rdm2.oooo += einsum(tmp2, (0, 1), delta.oo, (2, 3), (2, 1, 0, 3)) * 2
    rdm2.oooo += einsum(tmp2, (0, 1), delta.oo, (2, 3), (1, 2, 3, 0)) * 2
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp2, (2, 3), (0, 3, 1, 2)) * -4
    del tmp2
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3)) * 4
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (2, 0, 1, 3)) * -2
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
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:28.097116.

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
    tmp14 : array
    tmp15 : array
    tmp17 : array
    tmp18 : array
    tmp2 : array
    tmp20 : array
    tmp21 : array
    tmp22 : array
    tmp23 : array
    tmp24 : array
    tmp26 : array
    tmp27 : array
    tmp28 : array
    tmp29 : array
    tmp30 : array
    tmp32 : array
    tmp38 : array
    tmp39 : array
    tmp4 : array
    tmp6 : array
    tmp8 : array
    """

    tmp39 = einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 2, 4, 3), (0, 4))
    tmp38 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 3, 1), (4, 2))
    tmp32 = einsum(t2, (0, 1, 2, 3), v.ovov, (0, 2, 1, 4), (3, 4))
    tmp30 = einsum(t2, (0, 1, 2, 3), v.ovov, (0, 4, 1, 2), (3, 4))
    tmp29 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 2, 5))
    tmp28 = einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 0, 2), (4, 3))
    tmp27 = einsum(v.ooov, (0, 1, 2, 3), t2, (1, 2, 3, 4), (0, 4))
    tmp26 = einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp24 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 5, 3), (4, 0, 5, 1))
    tmp23 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    tmp22 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 5, 1), (4, 0, 5, 3))
    tmp21 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 1, 5), (4, 0, 5, 3))
    tmp20 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 2, 3, 5), (4, 0, 1, 5))
    tmp18 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 2, 5, 3), (4, 0, 1, 5))
    tmp17 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 2, 5))
    tmp15 = einsum(t2, (0, 1, 2, 3), f.ov, (1, 3), (0, 2))
    tmp14 = einsum(t2, (0, 1, 2, 3), f.ov, (1, 2), (0, 3))
    tmp8 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 5, 1, 3), (4, 5, 0, 2))
    tmp6 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 5, 1, 3), (4, 5, 0, 2))
    tmp4 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 3), (0, 4))
    tmp2 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 2), (0, 4))

    return {f"tmp14": tmp14, f"tmp15": tmp15, f"tmp17": tmp17, f"tmp18": tmp18, f"tmp2": tmp2, f"tmp20": tmp20, f"tmp21": tmp21, f"tmp22": tmp22, f"tmp23": tmp23, f"tmp24": tmp24, f"tmp26": tmp26, f"tmp27": tmp27, f"tmp28": tmp28, f"tmp29": tmp29, f"tmp30": tmp30, f"tmp32": tmp32, f"tmp38": tmp38, f"tmp39": tmp39, f"tmp4": tmp4, f"tmp6": tmp6, f"tmp8": tmp8}

def hbar_matvec_ip(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:28.108116.

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
    tmp7 = np.copy(np.transpose(v.ovov, (0, 2, 3, 1)))
    tmp7 += np.transpose(v.ovov, (0, 2, 1, 3)) * -0.5
    tmp10 = np.copy(f.ov)
    tmp10 += ints.tmp15 * 2
    del ints.tmp15
    tmp10 += ints.tmp27
    del ints.tmp27
    tmp10 += ints.tmp38 * 2
    del ints.tmp38
    tmp10 += ints.tmp14 * -1
    del ints.tmp14
    tmp10 += ints.tmp28 * -2
    del ints.tmp28
    tmp10 += ints.tmp39 * -1
    del ints.tmp39
    tmp5 = np.copy(ints.tmp21)
    del ints.tmp21
    tmp5 += v.oovv * -1
    tmp9 = np.copy(ints.tmp18) * 2
    del ints.tmp18
    tmp9 += np.transpose(ints.tmp26, (1, 2, 0, 3))
    del ints.tmp26
    tmp9 += np.transpose(ints.tmp6, (1, 0, 2, 3))
    del ints.tmp6
    tmp9 += np.transpose(v.ovoo, (0, 2, 3, 1))
    tmp9 += ints.tmp17 * -1
    del ints.tmp17
    tmp9 += ints.tmp20 * -1
    del ints.tmp20
    tmp9 += np.transpose(ints.tmp29, (1, 0, 2, 3)) * -1
    del ints.tmp29
    tmp8 = einsum(tmp7, (0, 1, 2, 3), r2, (0, 1, 3), (2,))
    del tmp7
    tmp6 = np.copy(f.vv)
    tmp6 += np.transpose(ints.tmp30, (1, 0))
    del ints.tmp30
    tmp6 += np.transpose(ints.tmp32, (1, 0)) * -2
    del ints.tmp32
    tmp3 = np.copy(r2) * -0.5
    tmp3 += np.transpose(r2, (1, 0, 2))
    tmp4 = np.copy(ints.tmp24) * 2
    del ints.tmp24
    tmp4 += np.transpose(v.ovov, (0, 2, 1, 3))
    tmp4 += ints.tmp22 * -1
    del ints.tmp22
    tmp4 += ints.tmp23 * -1
    del ints.tmp23
    tmp2 = np.copy(f.oo)
    tmp2 += np.transpose(ints.tmp4, (1, 0)) * 2
    del ints.tmp4
    tmp2 += np.transpose(ints.tmp2, (1, 0)) * -1
    del ints.tmp2
    tmp1 = np.copy(r2) * 2
    tmp1 += np.transpose(r2, (1, 0, 2)) * -1
    tmp0 = np.copy(v.ooov) * -0.5
    tmp0 += np.transpose(v.ovoo, (0, 2, 3, 1))
    r2new = einsum(r2, (0, 1, 2), v.oooo, (3, 0, 4, 1), (3, 4, 2))
    r2new += einsum(ints.tmp8, (0, 1, 2, 3), r2, (3, 2, 4), (1, 0, 4))
    del ints.tmp8
    r2new += einsum(tmp3, (0, 1, 2), tmp4, (3, 1, 4, 2), (3, 0, 4)) * 2
    del tmp3, tmp4
    r2new += einsum(r2, (0, 1, 2), tmp5, (3, 1, 4, 2), (0, 3, 4))
    r2new += einsum(tmp5, (0, 1, 2, 3), r2, (1, 4, 3), (0, 4, 2))
    del tmp5
    r2new += einsum(tmp6, (0, 1), r2, (2, 3, 0), (2, 3, 1))
    del tmp6
    r2new += einsum(tmp8, (0,), t2, (1, 2, 3, 0), (1, 2, 3)) * -2
    del tmp8
    r2new += einsum(r1, (0,), tmp9, (1, 2, 0, 3), (1, 2, 3))
    del tmp9
    r2new += einsum(r2, (0, 1, 2), tmp2, (0, 3), (3, 1, 2)) * -1
    r2new += einsum(tmp2, (0, 1), r2, (2, 0, 3), (2, 1, 3)) * -1
    r2new += einsum(r1, (0,), tmp10, (1, 2), (1, 0, 2)) * -1
    del tmp10
    r1new = einsum(tmp0, (0, 1, 2, 3), r2, (0, 2, 3), (1,)) * 2
    del tmp0
    r1new += einsum(f.ov, (0, 1), tmp1, (0, 2, 1), (2,)) * -1
    del tmp1
    r1new += einsum(r1, (0,), tmp2, (0, 1), (1,)) * -1
    del tmp2

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_matvec_ea_intermediates(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:33.396193.

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
    tmp11 : array
    tmp2 : array
    tmp20 : array
    tmp21 : array
    tmp22 : array
    tmp24 : array
    tmp26 : array
    tmp27 : array
    tmp28 : array
    tmp29 : array
    tmp30 : array
    tmp33 : array
    tmp34 : array
    tmp35 : array
    tmp37 : array
    tmp38 : array
    tmp4 : array
    tmp6 : array
    tmp9 : array
    """

    tmp38 = einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 0, 2), (4, 3))
    tmp37 = einsum(v.ooov, (0, 1, 2, 3), t2, (1, 2, 3, 4), (0, 4))
    tmp35 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 5, 3), (4, 0, 5, 1))
    tmp34 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    tmp33 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 5, 1), (4, 0, 5, 3))
    tmp30 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 1, 5), (4, 5, 2, 3))
    tmp29 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 5, 3), (4, 5, 1, 2))
    tmp28 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 5, 1), (4, 5, 2, 3))
    tmp27 = einsum(t2, (0, 1, 2, 3), f.ov, (1, 3), (0, 2))
    tmp26 = einsum(t2, (0, 1, 2, 3), f.ov, (1, 2), (0, 3))
    tmp24 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 1, 3), (4, 0))
    tmp22 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 3, 1), (4, 0))
    tmp21 = einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 2, 4, 3), (0, 4))
    tmp20 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 3, 1), (4, 2))
    tmp11 = einsum(v.ooov, (0, 1, 2, 3), t2, (2, 1, 4, 5), (0, 5, 4, 3))
    tmp9 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 1, 5), (4, 0, 5, 3))
    tmp6 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 3, 5), (4, 5, 1, 2))
    tmp4 = einsum(t2, (0, 1, 2, 3), v.ovov, (0, 4, 1, 3), (2, 4))
    tmp2 = einsum(t2, (0, 1, 2, 3), v.ovov, (0, 3, 1, 4), (2, 4))

    return {f"tmp11": tmp11, f"tmp2": tmp2, f"tmp20": tmp20, f"tmp21": tmp21, f"tmp22": tmp22, f"tmp24": tmp24, f"tmp26": tmp26, f"tmp27": tmp27, f"tmp28": tmp28, f"tmp29": tmp29, f"tmp30": tmp30, f"tmp33": tmp33, f"tmp34": tmp34, f"tmp35": tmp35, f"tmp37": tmp37, f"tmp38": tmp38, f"tmp4": tmp4, f"tmp6": tmp6, f"tmp9": tmp9}

def hbar_matvec_ea(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:33.407127.

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
    tmp8 = np.copy(np.transpose(v.ovov, (0, 2, 3, 1))) * -1
    tmp8 += np.transpose(v.ovov, (0, 2, 1, 3)) * 2
    tmp10 = np.copy(f.oo)
    tmp10 += np.transpose(ints.tmp24, (1, 0)) * 2
    del ints.tmp24
    tmp10 += np.transpose(ints.tmp22, (1, 0)) * -1
    del ints.tmp22
    tmp5 = np.copy(ints.tmp35) * 2
    del ints.tmp35
    tmp5 += np.transpose(v.ovov, (0, 2, 1, 3))
    tmp5 += ints.tmp33 * -1
    del ints.tmp33
    tmp5 += ints.tmp34 * -1
    del ints.tmp34
    tmp7 = np.copy(ints.tmp11)
    del ints.tmp11
    tmp7 += ints.tmp28 * 2
    del ints.tmp28
    tmp7 += v.ovvv
    tmp7 += np.transpose(ints.tmp29, (0, 1, 3, 2)) * -1
    del ints.tmp29
    tmp7 += ints.tmp30 * -1
    del ints.tmp30
    tmp7 += np.transpose(ints.tmp6, (0, 3, 1, 2)) * -1
    del ints.tmp6
    tmp9 = einsum(f.ov, (0, 1), r1, (1,), (0,))
    tmp9 += einsum(tmp8, (0, 1, 2, 3), r2, (2, 3, 0), (1,)) * -1
    del tmp8
    tmp6 = np.copy(ints.tmp9)
    del ints.tmp9
    tmp6 += v.oovv * -1
    tmp11 = np.copy(f.ov)
    tmp11 += ints.tmp20 * 2
    del ints.tmp20
    tmp11 += ints.tmp27 * 2
    del ints.tmp27
    tmp11 += ints.tmp37
    del ints.tmp37
    tmp11 += ints.tmp21 * -1
    del ints.tmp21
    tmp11 += ints.tmp26 * -1
    del ints.tmp26
    tmp11 += ints.tmp38 * -2
    del ints.tmp38
    tmp3 = einsum(r2, (0, 1, 2), v.ovov, (3, 0, 4, 1), (2, 3, 4))
    tmp4 = np.copy(np.transpose(r2, (2, 0, 1)))
    tmp4 += np.transpose(r2, (2, 1, 0)) * -0.5
    tmp2 = np.copy(f.vv)
    tmp2 += np.transpose(ints.tmp2, (1, 0))
    del ints.tmp2
    tmp2 += np.transpose(ints.tmp4, (1, 0)) * -2
    del ints.tmp4
    tmp1 = np.copy(np.transpose(r2, (2, 0, 1))) * -1
    tmp1 += np.transpose(r2, (2, 1, 0)) * 2
    tmp0 = np.copy(v.ovvv) * 2
    tmp0 += np.transpose(v.ovvv, (0, 2, 1, 3)) * -1
    r2new = einsum(r2, (0, 1, 2), v.vvvv, (3, 0, 4, 1), (3, 4, 2))
    r2new += einsum(t2, (0, 1, 2, 3), tmp3, (4, 1, 0), (3, 2, 4))
    del tmp3
    r2new += einsum(tmp5, (0, 1, 2, 3), tmp4, (1, 3, 4), (2, 4, 0)) * 2
    del tmp5, tmp4
    r2new += einsum(tmp6, (0, 1, 2, 3), r2, (4, 3, 1), (4, 2, 0))
    r2new += einsum(r2, (0, 1, 2), tmp6, (3, 2, 4, 0), (4, 1, 3))
    del tmp6
    r2new += einsum(r1, (0,), tmp7, (1, 2, 3, 0), (2, 3, 1)) * -1
    del tmp7
    r2new += einsum(r2, (0, 1, 2), tmp2, (0, 3), (3, 1, 2))
    r2new += einsum(tmp2, (0, 1), r2, (2, 0, 3), (2, 1, 3))
    r2new += einsum(tmp9, (0,), t2, (1, 0, 2, 3), (2, 3, 1))
    del tmp9
    r2new += einsum(r2, (0, 1, 2), tmp10, (2, 3), (0, 1, 3)) * -1
    del tmp10
    r2new += einsum(tmp11, (0, 1), r1, (2,), (1, 2, 0)) * -1
    del tmp11
    r1new = einsum(tmp0, (0, 1, 2, 3), r2, (1, 2, 0), (3,)) * -1
    del tmp0
    r1new += einsum(tmp1, (0, 1, 2), f.ov, (0, 2), (1,)) * -1
    del tmp1
    r1new += einsum(tmp2, (0, 1), r1, (0,), (1,))
    del tmp2

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_lmatvec_ip_intermediates(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:41.161446.

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
    tmp12 : array
    tmp15 : array
    tmp18 : array
    tmp19 : array
    tmp21 : array
    tmp22 : array
    tmp24 : array
    tmp27 : array
    tmp3 : array
    tmp30 : array
    tmp33 : array
    tmp34 : array
    tmp36 : array
    tmp37 : array
    tmp49 : array
    tmp51 : array
    tmp53 : array
    tmp8 : array
    """

    tmp53 = einsum(t2, (0, 1, 2, 3), v.ovov, (0, 2, 1, 4), (3, 4))
    tmp51 = einsum(t2, (0, 1, 2, 3), v.ovov, (0, 3, 1, 4), (2, 4))
    tmp49 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 5, 1, 3), (4, 5, 0, 2))
    tmp37 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 5, 3), (4, 0, 5, 1))
    tmp36 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 3, 5), (4, 0, 5, 1))
    tmp34 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 5, 1), (4, 0, 5, 3))
    tmp33 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 1, 5), (4, 0, 5, 3))
    tmp30 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 5, 1, 3), (4, 5, 0, 2))
    tmp27 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 1, 3), (4, 2))
    tmp24 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 3, 1), (4, 2))
    tmp22 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 2, 5, 3), (4, 0, 1, 5))
    tmp21 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 2, 3, 5), (4, 0, 1, 5))
    tmp19 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 2, 5))
    tmp18 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 2, 5))
    tmp15 = einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 0, 2), (4, 3))
    tmp12 = einsum(v.ooov, (0, 1, 2, 3), t2, (2, 1, 4, 3), (0, 4))
    tmp10 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 3), (0, 4))
    tmp8 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 2), (0, 4))
    tmp3 = einsum(t2, (0, 1, 2, 3), f.ov, (4, 3), (4, 0, 1, 2))
    tmp1 = einsum(t2, (0, 1, 2, 3), f.ov, (1, 3), (0, 2))
    tmp0 = einsum(t2, (0, 1, 2, 3), f.ov, (1, 2), (0, 3))

    return {f"tmp0": tmp0, f"tmp1": tmp1, f"tmp10": tmp10, f"tmp12": tmp12, f"tmp15": tmp15, f"tmp18": tmp18, f"tmp19": tmp19, f"tmp21": tmp21, f"tmp22": tmp22, f"tmp24": tmp24, f"tmp27": tmp27, f"tmp3": tmp3, f"tmp30": tmp30, f"tmp33": tmp33, f"tmp34": tmp34, f"tmp36": tmp36, f"tmp37": tmp37, f"tmp49": tmp49, f"tmp51": tmp51, f"tmp53": tmp53, f"tmp8": tmp8}

def hbar_lmatvec_ip(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:41.173351.

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
    tmp10 = np.copy(np.transpose(t2, (0, 1, 3, 2)))
    tmp10 += t2 * -0.5
    tmp5 = np.copy(r2) * -0.5
    tmp5 += np.transpose(r2, (1, 0, 2))
    tmp7 = np.copy(ints.tmp33)
    del ints.tmp33
    tmp7 += v.oovv * -1
    tmp9 = np.copy(f.vv)
    tmp9 += ints.tmp51
    del ints.tmp51
    tmp9 += ints.tmp53 * -2
    del ints.tmp53
    tmp8 = np.copy(ints.tmp49)
    del ints.tmp49
    tmp8 += np.transpose(v.oooo, (0, 2, 1, 3))
    tmp6 = np.copy(ints.tmp37) * 2
    del ints.tmp37
    tmp6 += np.transpose(v.ovov, (0, 2, 1, 3))
    tmp6 += ints.tmp34 * -1
    del ints.tmp34
    tmp6 += ints.tmp36 * -1
    del ints.tmp36
    tmp11 = einsum(tmp10, (0, 1, 2, 3), r2, (0, 1, 3), (2,))
    del tmp10
    tmp4 = np.copy(f.oo)
    tmp4 += ints.tmp10 * 2
    del ints.tmp10
    tmp4 += ints.tmp8 * -1
    del ints.tmp8
    tmp1 = np.copy(ints.tmp18)
    tmp1 += np.transpose(ints.tmp22, (0, 2, 1, 3))
    tmp1 += np.transpose(ints.tmp3, (1, 2, 0, 3)) * 0.5
    tmp1 += ints.tmp19 * -0.5
    tmp1 += np.transpose(ints.tmp21, (0, 2, 1, 3)) * -0.5
    tmp1 += ints.tmp30 * -1
    tmp0 = np.copy(ints.tmp18)
    del ints.tmp18
    tmp0 += np.transpose(ints.tmp22, (0, 2, 1, 3)) * 4
    del ints.tmp22
    tmp0 += np.transpose(ints.tmp3, (1, 2, 0, 3)) * 2
    del ints.tmp3
    tmp0 += ints.tmp19 * -2
    del ints.tmp19
    tmp0 += np.transpose(ints.tmp21, (0, 2, 1, 3)) * -2
    del ints.tmp21
    tmp0 += ints.tmp30 * -1
    del ints.tmp30
    tmp0 += np.transpose(v.ooov, (0, 2, 1, 3)) * -1
    tmp0 += np.transpose(v.ovoo, (0, 2, 3, 1)) * 2
    tmp3 = np.copy(r2) * 2
    tmp3 += np.transpose(r2, (1, 0, 2)) * -1
    tmp2 = np.copy(f.ov) * 0.5
    tmp2 += ints.tmp12 * 0.5
    del ints.tmp12
    tmp2 += ints.tmp1
    del ints.tmp1
    tmp2 += ints.tmp24
    del ints.tmp24
    tmp2 += ints.tmp0 * -0.5
    del ints.tmp0
    tmp2 += ints.tmp15 * -1
    del ints.tmp15
    tmp2 += ints.tmp27 * -0.5
    del ints.tmp27
    r2new = einsum(r1, (0,), v.ovoo, (1, 2, 3, 0), (1, 3, 2))
    r2new += einsum(r1, (0,), f.ov, (1, 2), (1, 0, 2)) * -1
    r2new += einsum(tmp5, (0, 1, 2), tmp6, (1, 3, 2, 4), (3, 0, 4)) * 2
    del tmp5, tmp6
    r2new += einsum(r2, (0, 1, 2), tmp7, (1, 3, 2, 4), (0, 3, 4))
    r2new += einsum(r2, (0, 1, 2), tmp7, (0, 3, 2, 4), (3, 1, 4))
    del tmp7
    r2new += einsum(tmp8, (0, 1, 2, 3), r2, (0, 1, 4), (2, 3, 4))
    del tmp8
    r2new += einsum(tmp9, (0, 1), r2, (2, 3, 0), (2, 3, 1))
    del tmp9
    r2new += einsum(v.ovov, (0, 1, 2, 3), tmp11, (3,), (0, 2, 1)) * -2
    del tmp11
    r2new += einsum(r2, (0, 1, 2), tmp4, (0, 3), (3, 1, 2)) * -1
    r2new += einsum(tmp4, (0, 1), r2, (2, 0, 3), (2, 1, 3)) * -1
    r1new = einsum(tmp0, (0, 1, 2, 3), r2, (0, 1, 3), (2,))
    del tmp0
    r1new += einsum(tmp1, (0, 1, 2, 3), r2, (1, 0, 3), (2,)) * -2
    del tmp1
    r1new += einsum(tmp3, (0, 1, 2), tmp2, (0, 2), (1,)) * -2
    del tmp3, tmp2
    r1new += einsum(r1, (0,), tmp4, (0, 1), (1,)) * -1
    del tmp4

    return {f"r1new": r1new, f"r2new": r2new}

def hbar_lmatvec_ea_intermediates(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:49.340062.

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
    tmp11 : array
    tmp13 : array
    tmp16 : array
    tmp19 : array
    tmp22 : array
    tmp25 : array
    tmp28 : array
    tmp30 : array
    tmp33 : array
    tmp35 : array
    tmp37 : array
    tmp39 : array
    tmp41 : array
    tmp42 : array
    tmp44 : array
    tmp45 : array
    tmp9 : array
    """

    tmp45 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 5, 3), (4, 0, 5, 1))
    tmp44 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 3, 5), (4, 0, 5, 1))
    tmp42 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 5, 1), (4, 0, 5, 3))
    tmp41 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 1, 5), (4, 0, 5, 3))
    tmp39 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 1, 3), (4, 0))
    tmp37 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 3, 1), (4, 0))
    tmp35 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 5, 3), (4, 5, 1, 2))
    tmp33 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 3, 5), (4, 5, 1, 2))
    tmp30 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 5, 1), (4, 5, 2, 3))
    tmp28 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 1, 5), (4, 5, 2, 3))
    tmp25 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 1, 3), (4, 2))
    tmp22 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 3, 1), (4, 2))
    tmp19 = einsum(v.ooov, (0, 1, 2, 3), t2, (2, 1, 4, 5), (0, 5, 4, 3))
    tmp16 = einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 0, 2), (4, 3))
    tmp13 = einsum(v.ooov, (0, 1, 2, 3), t2, (2, 1, 4, 3), (0, 4))
    tmp11 = einsum(t2, (0, 1, 2, 3), v.ovov, (0, 4, 1, 3), (2, 4))
    tmp9 = einsum(t2, (0, 1, 2, 3), v.ovov, (0, 4, 1, 2), (3, 4))
    tmp1 = einsum(t2, (0, 1, 2, 3), f.ov, (1, 3), (0, 2))
    tmp0 = einsum(t2, (0, 1, 2, 3), f.ov, (1, 2), (0, 3))

    return {f"tmp0": tmp0, f"tmp1": tmp1, f"tmp11": tmp11, f"tmp13": tmp13, f"tmp16": tmp16, f"tmp19": tmp19, f"tmp22": tmp22, f"tmp25": tmp25, f"tmp28": tmp28, f"tmp30": tmp30, f"tmp33": tmp33, f"tmp35": tmp35, f"tmp37": tmp37, f"tmp39": tmp39, f"tmp41": tmp41, f"tmp42": tmp42, f"tmp44": tmp44, f"tmp45": tmp45, f"tmp9": tmp9}

def hbar_lmatvec_ea(f=None, r1=None, r2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-17T18:48:49.350979.

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
    tmp6 = np.copy(np.transpose(t2, (0, 1, 3, 2)))
    tmp6 += t2 * -0.5
    tmp10 = np.copy(ints.tmp45) * 2
    del ints.tmp45
    tmp10 += np.transpose(v.ovov, (0, 2, 1, 3))
    tmp10 += ints.tmp42 * -1
    del ints.tmp42
    tmp10 += ints.tmp44 * -1
    del ints.tmp44
    tmp5 = np.copy(f.vv)
    tmp5 += ints.tmp9
    del ints.tmp9
    tmp5 += ints.tmp11 * -2
    del ints.tmp11
    tmp9 = np.copy(np.transpose(r2, (2, 0, 1)))
    tmp9 += np.transpose(r2, (2, 1, 0)) * -0.5
    tmp8 = einsum(r2, (0, 1, 2), t2, (3, 4, 0, 1), (2, 3, 4))
    tmp13 = einsum(r2, (0, 1, 2), tmp6, (2, 3, 1, 0), (3,))
    tmp12 = np.copy(f.oo)
    tmp12 += ints.tmp39 * 2
    del ints.tmp39
    tmp12 += ints.tmp37 * -1
    del ints.tmp37
    tmp11 = np.copy(ints.tmp41)
    del ints.tmp41
    tmp11 += v.oovv * -1
    tmp1 = np.copy(ints.tmp19)
    del ints.tmp19
    tmp1 += np.transpose(ints.tmp30, (0, 1, 3, 2)) * 2
    del ints.tmp30
    tmp1 += np.transpose(ints.tmp28, (0, 1, 3, 2)) * -1
    del ints.tmp28
    tmp1 += np.transpose(ints.tmp35, (0, 1, 3, 2)) * -1
    del ints.tmp35
    tmp7 = einsum(r2, (0, 1, 2), tmp6, (2, 3, 1, 0), (3,)) * 2
    del tmp6
    tmp0 = np.copy(np.transpose(r2, (2, 0, 1))) * 2
    tmp0 += np.transpose(r2, (2, 1, 0)) * -1
    tmp3 = np.copy(f.ov) * 0.5
    tmp3 += ints.tmp13 * 0.5
    del ints.tmp13
    tmp3 += ints.tmp1
    del ints.tmp1
    tmp3 += ints.tmp22
    del ints.tmp22
    tmp3 += ints.tmp0 * -0.5
    del ints.tmp0
    tmp3 += ints.tmp16 * -1
    del ints.tmp16
    tmp3 += ints.tmp25 * -0.5
    del ints.tmp25
    tmp4 = np.copy(np.transpose(r2, (2, 0, 1))) * -1
    tmp4 += np.transpose(r2, (2, 1, 0)) * 2
    tmp2 = np.copy(ints.tmp33)
    tmp2 += v.ovvv * 2
    tmp2 += np.transpose(v.ovvv, (0, 2, 3, 1)) * -1
    r2new = einsum(r2, (0, 1, 2), v.vvvv, (3, 0, 4, 1), (3, 4, 2))
    r2new += einsum(f.ov, (0, 1), r1, (2,), (1, 2, 0)) * -1
    r2new += einsum(tmp8, (0, 1, 2), v.ovov, (2, 3, 1, 4), (4, 3, 0))
    del tmp8
    r2new += einsum(r1, (0,), v.ovvv, (1, 2, 3, 0), (2, 3, 1)) * -1
    r2new += einsum(tmp9, (0, 1, 2), tmp10, (0, 3, 1, 4), (4, 2, 3)) * 2
    del tmp9, tmp10
    r2new += einsum(tmp11, (0, 1, 2, 3), r2, (4, 2, 0), (4, 3, 1))
    r2new += einsum(r2, (0, 1, 2), tmp11, (2, 3, 0, 4), (4, 1, 3))
    del tmp11
    r2new += einsum(r2, (0, 1, 2), tmp5, (0, 3), (3, 1, 2))
    r2new += einsum(r2, (0, 1, 2), tmp5, (1, 3), (0, 3, 2))
    r2new += einsum(r2, (0, 1, 2), tmp12, (2, 3), (0, 1, 3)) * -1
    del tmp12
    r2new += einsum(tmp13, (0,), v.ovov, (1, 2, 0, 3), (2, 3, 1)) * -2
    del tmp13
    r1new = einsum(r2, (0, 1, 2), ints.tmp33, (2, 1, 3, 0), (3,)) * 2
    del ints.tmp33
    r1new += einsum(tmp1, (0, 1, 2, 3), tmp0, (0, 1, 2), (3,)) * -1
    del tmp1, tmp0
    r1new += einsum(r2, (0, 1, 2), tmp2, (2, 0, 3, 1), (3,)) * -1
    del tmp2
    r1new += einsum(tmp4, (0, 1, 2), tmp3, (0, 2), (1,)) * -2
    del tmp4, tmp3
    r1new += einsum(tmp5, (0, 1), r1, (0,), (1,))
    del tmp5
    r1new += einsum(f.ov, (0, 1), tmp7, (0,), (1,))
    del tmp7

    return {f"r1new": r1new, f"r2new": r2new}

