"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-08-09T21:27:42.457041
  * python version: 3.10.14 (main, Jul 16 2024, 19:03:10) [GCC 11.4.0]
  * albert version: 0.0.0
  * caller: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/albert/codegen/einsum.py
  * node: fv-az1487-369
  * system: Linux
  * processor: x86_64
  * release: 6.5.0-1025-azure
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace


def energy(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T21:27:42.730032.

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
    e_cc += einsum(t2, (0, 1, 2, 3), v.ovov, (0, 3, 1, 2), ()) * -1

    return e_cc

def update_amps(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T21:27:46.502577.

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

    tmp9 = v.ovov.transpose((0, 2, 3, 1)).copy() * -0.5
    tmp9 += v.ovov.transpose((0, 2, 1, 3))
    tmp6 = v.ovov.transpose((0, 2, 3, 1)).copy()
    tmp6 += v.ovov.transpose((0, 2, 1, 3)) * -0.5
    tmp3 = v.ovov.transpose((0, 2, 3, 1)).copy() * 2
    tmp3 += v.ovov.transpose((0, 2, 1, 3)) * -1
    tmp10 = einsum(t2, (0, 1, 2, 3), tmp9, (1, 4, 3, 2), (0, 4)) * 2
    del tmp9
    tmp7 = einsum(tmp6, (0, 1, 2, 3), t2, (0, 1, 3, 4), (4, 2))
    del tmp6
    tmp4 = einsum(tmp3, (0, 1, 2, 3), t2, (4, 0, 5, 3), (4, 1, 5, 2))
    del tmp3
    tmp11 = einsum(t2, (0, 1, 2, 3), tmp10, (4, 1), (0, 4, 3, 2))
    del tmp10
    tmp8 = einsum(tmp7, (0, 1), t2, (2, 3, 1, 4), (2, 3, 4, 0)) * 2
    del tmp7
    tmp0 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new = tmp0.copy() * -1
    tmp16 = v.oovv.copy() * -1
    tmp16 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    t2new += einsum(tmp16, (0, 1, 2, 3), t2, (4, 1, 3, 5), (0, 4, 5, 2))
    del tmp16
    tmp13 = v.ovov.transpose((0, 2, 1, 3)).copy() * 2
    tmp13 += v.oovv * -1
    tmp13 += tmp4 * 2
    t2new += einsum(t2, (0, 1, 2, 3), tmp13, (4, 1, 5, 3), (4, 0, 5, 2))
    del tmp13
    tmp5 = einsum(t2, (0, 1, 2, 3), tmp4, (4, 1, 5, 2), (0, 4, 3, 5))
    del tmp4
    t2new += tmp5 * -1
    t2new += tmp5.transpose((1, 0, 3, 2)) * -1
    del tmp5
    tmp12 = tmp8.transpose((1, 0, 2, 3)).copy()
    del tmp8
    tmp12 += tmp11.transpose((0, 1, 3, 2))
    del tmp11
    t2new += tmp12.transpose((1, 0, 3, 2)) * -1
    t2new += tmp12 * -1
    del tmp12
    tmp15 = v.ovov.transpose((0, 2, 1, 3)).copy() * -1
    tmp15 += tmp0
    del tmp0
    t2new += einsum(t2, (0, 1, 2, 3), tmp15, (4, 1, 5, 2), (4, 0, 5, 3))
    del tmp15
    tmp2 = einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new += tmp2.transpose((1, 0, 2, 3))
    t2new += tmp2.transpose((0, 1, 3, 2))
    del tmp2
    tmp14 = v.oooo.copy()
    tmp14 += einsum(v.ovov, (0, 1, 2, 3), t2, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new += einsum(tmp14, (0, 1, 2, 3), t2, (0, 2, 4, 5), (1, 3, 4, 5))
    del tmp14
    tmp1 = einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new += tmp1.transpose((1, 0, 2, 3)) * -1
    t2new += tmp1.transpose((0, 1, 3, 2)) * -1
    del tmp1
    t2new += v.ovov.transpose((0, 2, 1, 3))
    t2new += einsum(v.vvvv, (0, 1, 2, 3), t2, (4, 5, 1, 3), (4, 5, 0, 2))
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5)) * -1
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (0, 4, 5, 3)) * -1
    t2new += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5)) * 2

    return {f"t2new": t2new}

def update_lams(f=None, l2=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T21:27:52.705831.

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

    tmp10 = l2.transpose((3, 2, 0, 1)).copy() * 2
    tmp10 += l2.transpose((2, 3, 0, 1)) * -1
    tmp4 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    tmp22 = t2.transpose((0, 1, 3, 2)).copy() * -1
    tmp22 += t2 * 2
    tmp19 = t2.transpose((0, 1, 3, 2)).copy() * 2
    tmp19 += t2 * -1
    tmp15 = einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), (3, 4, 0, 5))
    tmp11 = einsum(tmp10, (0, 1, 2, 3), t2, (4, 0, 5, 3), (4, 1, 5, 2))
    del tmp10
    tmp9 = einsum(t2, (0, 1, 2, 3), l2, (4, 2, 5, 1), (5, 0, 4, 3))
    tmp26 = t2.transpose((0, 1, 3, 2)).copy()
    tmp26 += t2 * -0.5
    tmp29 = t2.transpose((0, 1, 3, 2)).copy() * -0.5
    tmp29 += t2
    tmp5 = v.oovv.copy() * -1
    tmp5 += tmp4
    del tmp4
    tmp23 = einsum(tmp22, (0, 1, 2, 3), v.ovov, (4, 3, 0, 2), (4, 1))
    del tmp22
    tmp20 = einsum(tmp19, (0, 1, 2, 3), v.ovov, (0, 3, 1, 4), (4, 2)) * 0.5
    del tmp19
    tmp16 = tmp15.copy()
    del tmp15
    tmp16 += l2.transpose((3, 2, 0, 1)) * -1
    tmp16 += l2.transpose((2, 3, 0, 1)) * 2
    tmp13 = v.ovov.transpose((0, 2, 3, 1)).copy() * 2
    tmp13 += v.ovov.transpose((0, 2, 1, 3)) * -1
    tmp12 = tmp9.copy()
    del tmp9
    tmp12 += tmp11.transpose((1, 0, 3, 2)) * -1
    del tmp11
    tmp27 = einsum(tmp26, (0, 1, 2, 3), l2, (4, 3, 1, 0), (4, 2)) * 2
    del tmp26
    tmp30 = einsum(l2, (0, 1, 2, 3), tmp29, (2, 4, 0, 1), (3, 4)) * 2
    del tmp29
    tmp6 = einsum(l2, (0, 1, 2, 3), tmp5, (2, 4, 1, 5), (3, 4, 0, 5))
    del tmp5
    tmp3 = einsum(l2, (0, 1, 2, 3), f.vv, (4, 1), (2, 3, 4, 0))
    tmp24 = einsum(tmp23, (0, 1), l2, (2, 3, 1, 4), (4, 0, 2, 3))
    del tmp23
    tmp21 = einsum(l2, (0, 1, 2, 3), tmp20, (4, 1), (3, 2, 0, 4)) * 2
    del tmp20
    tmp17 = einsum(v.ovov, (0, 1, 2, 3), tmp16, (4, 2, 5, 3), (0, 4, 1, 5))
    del tmp16
    tmp8 = einsum(v.oovv, (0, 1, 2, 3), l2, (4, 3, 5, 1), (5, 0, 4, 2))
    tmp14 = einsum(tmp12, (0, 1, 2, 3), tmp13, (1, 4, 5, 3), (0, 4, 2, 5))
    del tmp12, tmp13
    tmp28 = einsum(tmp27, (0, 1), v.ovov, (2, 1, 3, 4), (2, 3, 4, 0))
    del tmp27
    tmp31 = einsum(tmp30, (0, 1), v.ovov, (2, 3, 1, 4), (2, 0, 4, 3))
    del tmp30
    tmp7 = tmp3.copy()
    del tmp3
    tmp7 += tmp6
    del tmp6
    l2new = tmp7.transpose((2, 3, 1, 0)).copy()
    l2new += tmp7.transpose((3, 2, 0, 1))
    del tmp7
    tmp25 = tmp21.transpose((1, 0, 2, 3)).copy()
    del tmp21
    tmp25 += tmp24.transpose((0, 1, 3, 2))
    del tmp24
    l2new += tmp25.transpose((3, 2, 1, 0)) * -1
    l2new += tmp25.transpose((2, 3, 0, 1)) * -1
    del tmp25
    tmp18 = tmp8.copy()
    del tmp8
    tmp18 += tmp14
    del tmp14
    tmp18 += tmp17.transpose((1, 0, 3, 2)) * -1
    del tmp17
    l2new += tmp18.transpose((3, 2, 1, 0)) * -1
    l2new += tmp18.transpose((2, 3, 0, 1)) * -1
    del tmp18
    tmp0 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 5, 1, 3), (4, 5, 0, 2))
    l2new += einsum(l2, (0, 1, 2, 3), tmp0, (3, 2, 4, 5), (0, 1, 5, 4))
    del tmp0
    tmp2 = einsum(f.oo, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new += tmp2.transpose((2, 3, 1, 0)) * -1
    l2new += tmp2.transpose((3, 2, 0, 1)) * -1
    del tmp2
    tmp32 = tmp28.transpose((1, 0, 3, 2)).copy()
    del tmp28
    tmp32 += tmp31.transpose((1, 0, 3, 2))
    del tmp31
    l2new += tmp32.transpose((2, 3, 1, 0)) * -1
    l2new += tmp32.transpose((3, 2, 0, 1)) * -1
    del tmp32
    tmp1 = einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    l2new += einsum(v.ovov, (0, 1, 2, 3), tmp1, (4, 5, 0, 2), (1, 3, 4, 5))
    del tmp1
    l2new += einsum(v.oooo, (0, 1, 2, 3), l2, (4, 5, 1, 3), (4, 5, 0, 2))
    l2new += v.ovov.transpose((1, 3, 0, 2))
    l2new += einsum(l2, (0, 1, 2, 3), v.vvvv, (4, 1, 5, 0), (4, 5, 3, 2))

    return {f"l2new": l2new}

def make_rdm1_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T21:27:53.211849.

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
    tmp1 = t2.transpose((0, 1, 3, 2)).copy() * -0.5
    tmp1 += t2
    rdm1.vv = einsum(tmp1, (0, 1, 2, 3), l2, (4, 3, 0, 1), (4, 2)) * 4
    del tmp1
    tmp0 = t2.transpose((0, 1, 3, 2)).copy()
    tmp0 += t2 * -0.5
    rdm1.oo = einsum(l2, (0, 1, 2, 3), tmp0, (2, 4, 1, 0), (4, 3)) * -4
    del tmp0
    rdm1.oo += delta.oo * 2
    del delta
    rdm1.ov = np.zeros((t2.shape[0], t2.shape[-1]))
    rdm1.vo = np.zeros((t2.shape[-1], t2.shape[0]))
    rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])

    return rdm1

def make_rdm2_f(l2=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T21:28:12.168357.

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
    tmp1 = t2.transpose((0, 1, 3, 2)).copy()
    tmp1 += t2 * -0.5
    tmp12 = l2.transpose((3, 2, 0, 1)).copy() * 2
    tmp12 += l2.transpose((2, 3, 0, 1)) * -1
    tmp3 = l2.transpose((3, 2, 0, 1)).copy() * 2
    tmp3 += l2.transpose((2, 3, 0, 1)) * -1
    tmp6 = l2.transpose((3, 2, 0, 1)).copy()
    tmp6 += l2.transpose((2, 3, 0, 1)) * -1
    tmp2 = einsum(l2, (0, 1, 2, 3), tmp1, (2, 4, 1, 0), (3, 4)) * 2
    del tmp1
    rdm2.oooo = einsum(delta.oo, (0, 1), tmp2, (2, 3), (0, 3, 1, 2)) * -1
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp2, (2, 3), (3, 0, 1, 2))
    rdm2.oooo += einsum(tmp2, (0, 1), delta.oo, (2, 3), (2, 1, 0, 3))
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp2, (2, 3), (3, 0, 2, 1)) * -1
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp2, (2, 3), (0, 3, 1, 2)) * -1
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp2, (2, 3), (3, 0, 2, 1)) * -1
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp2, (2, 3), (0, 3, 1, 2)) * -1
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp2, (2, 3), (3, 0, 2, 1)) * -1
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp2, (2, 3), (0, 3, 1, 2)) * -1
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp2, (2, 3), (3, 0, 1, 2))
    rdm2.oooo += einsum(tmp2, (0, 1), delta.oo, (2, 3), (2, 1, 0, 3))
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp2, (2, 3), (3, 0, 2, 1)) * -1
    tmp13 = einsum(tmp12, (0, 1, 2, 3), t2, (0, 1, 3, 4), (4, 2)) * 0.5
    rdm2.ovvo = einsum(delta.oo, (0, 1), tmp13, (2, 3), (0, 3, 2, 1)) * -2
    rdm2.ovvo += einsum(delta.oo, (0, 1), tmp13, (2, 3), (0, 3, 2, 1)) * -2
    rdm2.ovov = einsum(delta.oo, (0, 1), tmp13, (2, 3), (0, 3, 1, 2)) * 2
    rdm2.ovov += einsum(delta.oo, (0, 1), tmp13, (2, 3), (0, 3, 1, 2)) * 2
    rdm2.ovov += einsum(delta.oo, (0, 1), tmp13, (2, 3), (0, 3, 1, 2)) * 2
    rdm2.ovov += einsum(delta.oo, (0, 1), tmp13, (2, 3), (0, 3, 1, 2)) * 2
    tmp4 = einsum(tmp3, (0, 1, 2, 3), t2, (4, 0, 5, 3), (4, 1, 5, 2))
    del tmp3
    rdm2.vovo = tmp4.transpose((3, 0, 2, 1)).copy() * -1
    rdm2.vovo += tmp4.transpose((3, 0, 2, 1)) * -1
    rdm2.voov = tmp4.transpose((3, 0, 1, 2)).copy()
    rdm2.voov += tmp4.transpose((3, 0, 1, 2))
    rdm2.ovvo += tmp4.transpose((0, 3, 2, 1))
    rdm2.ovvo += tmp4.transpose((0, 3, 2, 1))
    rdm2.ovvo += tmp4.transpose((0, 3, 2, 1))
    rdm2.ovvo += tmp4.transpose((0, 3, 2, 1))
    tmp7 = einsum(t2, (0, 1, 2, 3), tmp6, (1, 4, 5, 2), (0, 4, 3, 5))
    del tmp6
    rdm2.ovvo += tmp7.transpose((0, 3, 2, 1)) * -1
    rdm2.ovvo += tmp7.transpose((0, 3, 2, 1)) * -1
    tmp26 = l2.transpose((3, 2, 0, 1)).copy() * -1
    tmp26 += l2.transpose((2, 3, 0, 1))
    tmp23 = t2.transpose((0, 1, 3, 2)).copy() * 2
    tmp23 += t2 * -1
    tmp21 = t2.transpose((0, 1, 3, 2)).copy()
    tmp21 += t2 * -1
    tmp15 = einsum(t2, (0, 1, 2, 3), tmp2, (1, 4), (0, 4, 3, 2))
    del tmp2
    tmp14 = einsum(tmp13, (0, 1), t2, (2, 3, 1, 4), (2, 3, 4, 0)) * 2
    del tmp13
    tmp17 = einsum(t2, (0, 1, 2, 3), l2, (4, 2, 5, 1), (5, 0, 4, 3))
    rdm2.voov += tmp17.transpose((2, 1, 0, 3)) * -1
    rdm2.voov += tmp17.transpose((2, 1, 0, 3)) * -1
    rdm2.ovvo += tmp17.transpose((1, 2, 3, 0)) * -1
    rdm2.ovvo += tmp17.transpose((1, 2, 3, 0)) * -1
    tmp0 = einsum(l2, (0, 1, 2, 3), t2, (4, 5, 1, 0), (3, 2, 4, 5))
    rdm2.oooo += tmp0.transpose((3, 2, 1, 0))
    rdm2.oooo += tmp0.transpose((2, 3, 1, 0)) * -1
    rdm2.oooo += tmp0.transpose((3, 2, 1, 0))
    rdm2.oooo += tmp0.transpose((3, 2, 1, 0))
    rdm2.oooo += tmp0.transpose((3, 2, 1, 0))
    rdm2.oooo += tmp0.transpose((2, 3, 1, 0)) * -1
    tmp19 = einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), (3, 4, 0, 5))
    rdm2.vovo += tmp19.transpose((2, 1, 3, 0)) * -1
    rdm2.vovo += tmp19.transpose((2, 1, 3, 0)) * -1
    rdm2.ovov += tmp19.transpose((1, 2, 0, 3)) * -1
    rdm2.ovov += tmp19.transpose((1, 2, 0, 3)) * -1
    tmp5 = einsum(tmp4, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 5, 2))
    rdm2.oovv = tmp5.copy() * 2
    rdm2.oovv += tmp5 * 2
    tmp8 = einsum(t2, (0, 1, 2, 3), tmp7, (4, 1, 5, 2), (0, 4, 3, 5))
    del tmp7
    tmp28 = einsum(l2, (0, 1, 2, 3), t2, (3, 2, 4, 5), (0, 1, 5, 4))
    rdm2.vvvv = tmp28.transpose((1, 0, 3, 2)).copy()
    rdm2.vvvv += tmp28.transpose((1, 0, 2, 3)) * -1
    rdm2.vvvv += tmp28.transpose((1, 0, 3, 2))
    rdm2.vvvv += tmp28.transpose((1, 0, 3, 2))
    rdm2.vvvv += tmp28.transpose((1, 0, 3, 2))
    rdm2.vvvv += tmp28.transpose((1, 0, 2, 3)) * -1
    del tmp28
    tmp25 = einsum(tmp12, (0, 1, 2, 3), t2, (0, 1, 3, 4), (4, 2))
    del tmp12
    rdm2.vovo += einsum(delta.oo, (0, 1), tmp25, (2, 3), (3, 0, 2, 1))
    rdm2.vovo += einsum(delta.oo, (0, 1), tmp25, (2, 3), (3, 0, 2, 1))
    rdm2.vovo += einsum(delta.oo, (0, 1), tmp25, (2, 3), (3, 0, 2, 1))
    rdm2.vovo += einsum(delta.oo, (0, 1), tmp25, (2, 3), (3, 0, 2, 1))
    rdm2.voov += einsum(delta.oo, (0, 1), tmp25, (2, 3), (3, 0, 1, 2)) * -1
    rdm2.voov += einsum(delta.oo, (0, 1), tmp25, (2, 3), (3, 0, 1, 2)) * -1
    del tmp25
    tmp27 = einsum(t2, (0, 1, 2, 3), tmp26, (1, 4, 5, 2), (0, 4, 3, 5))
    del tmp26
    rdm2.vovo += tmp27.transpose((3, 0, 2, 1)) * -1
    rdm2.vovo += tmp27.transpose((3, 0, 2, 1)) * -1
    del tmp27
    tmp24 = einsum(l2, (0, 1, 2, 3), tmp23, (3, 4, 5, 1), (2, 4, 0, 5))
    del tmp23
    rdm2.voov += tmp24.transpose((2, 1, 0, 3))
    rdm2.voov += tmp24.transpose((2, 1, 0, 3))
    rdm2.ovov += tmp24.transpose((1, 2, 0, 3)) * -1
    rdm2.ovov += tmp24.transpose((1, 2, 0, 3)) * -1
    del tmp24
    tmp22 = einsum(l2, (0, 1, 2, 3), tmp21, (2, 4, 5, 1), (3, 4, 0, 5))
    del tmp21
    rdm2.voov += tmp22.transpose((2, 1, 0, 3)) * -1
    rdm2.voov += tmp22.transpose((2, 1, 0, 3)) * -1
    rdm2.ovov += tmp22.transpose((1, 2, 0, 3))
    rdm2.ovov += tmp22.transpose((1, 2, 0, 3))
    del tmp22
    tmp16 = tmp14.transpose((1, 0, 2, 3)).copy()
    del tmp14
    tmp16 += tmp15.transpose((0, 1, 3, 2))
    del tmp15
    rdm2.oovv += tmp16.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp16.transpose((1, 0, 2, 3))
    rdm2.oovv += tmp16.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp16 * -1
    rdm2.oovv += tmp16.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp16 * -1
    rdm2.oovv += tmp16.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp16 * -1
    rdm2.oovv += tmp16.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp16.transpose((1, 0, 2, 3))
    rdm2.oovv += tmp16.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp16 * -1
    del tmp16
    tmp18 = einsum(t2, (0, 1, 2, 3), tmp17, (1, 4, 2, 5), (0, 4, 3, 5))
    del tmp17
    rdm2.oovv += tmp18
    rdm2.oovv += tmp18
    del tmp18
    tmp10 = einsum(t2, (0, 1, 2, 3), tmp4, (4, 1, 5, 2), (0, 4, 3, 5))
    del tmp4
    rdm2.oovv += tmp10 * -1
    rdm2.oovv += tmp10.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp10.transpose((1, 0, 2, 3))
    rdm2.oovv += tmp10.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp10 * -1
    rdm2.oovv += tmp10.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp10 * -1
    rdm2.oovv += tmp10.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp10 * -1
    rdm2.oovv += tmp10.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp10.transpose((1, 0, 2, 3))
    rdm2.oovv += tmp10.transpose((1, 0, 3, 2)) * -1
    del tmp10
    tmp11 = einsum(tmp0, (0, 1, 2, 3), t2, (1, 0, 4, 5), (2, 3, 5, 4))
    del tmp0
    rdm2.oovv += tmp11.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += tmp11
    rdm2.oovv += tmp11
    rdm2.oovv += tmp11
    rdm2.oovv += tmp11.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += tmp11
    del tmp11
    tmp20 = einsum(t2, (0, 1, 2, 3), tmp19, (1, 4, 2, 5), (0, 4, 3, 5))
    del tmp19
    rdm2.oovv += tmp20.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp20.transpose((0, 1, 3, 2))
    del tmp20
    tmp9 = tmp5.copy() * 2
    del tmp5
    tmp9 += tmp8.transpose((1, 0, 3, 2))
    del tmp8
    rdm2.oovv += tmp9.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += tmp9
    rdm2.oovv += tmp9.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += tmp9
    del tmp9
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (2, 0, 3, 1))
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (2, 0, 3, 1))
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (2, 0, 3, 1))
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (2, 0, 3, 1))
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1
    del delta
    rdm2.oovv += t2.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += t2
    rdm2.oovv += t2
    rdm2.oovv += t2
    rdm2.oovv += t2.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += t2
    rdm2.vvoo = l2.transpose((0, 1, 3, 2)).copy() * -1
    rdm2.vvoo += l2
    rdm2.vvoo += l2
    rdm2.vvoo += l2
    rdm2.vvoo += l2.transpose((0, 1, 3, 2)) * -1
    rdm2.vvoo += l2
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

