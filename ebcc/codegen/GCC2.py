"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-08-09T22:01:35.226778
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


def energy(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T22:01:35.476815.

    Parameters
    ----------
    f : array
        Fock matrix.
    t1 : array
        T1 amplitudes.
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    e_cc : float
        Coupled cluster energy.
    """

    tmp0 = t2.copy()
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2
    e_cc = einsum(tmp0, (0, 1, 2, 3), v.oovv, (0, 1, 2, 3), ()) * 0.25
    del tmp0
    e_cc += einsum(t1, (0, 1), f.ov, (0, 1), ())

    return e_cc

def update_amps(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T22:01:39.018242.

    Parameters
    ----------
    f : array
        Fock matrix.
    t1 : array
        T1 amplitudes.
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    t1new : array
        Updated T1 residuals.
    t2new : array
        Updated T2 residuals.
    """

    tmp15 = einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp11 = einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp0 = einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp4 = einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    tmp16 = einsum(tmp15, (0, 1, 2, 3), t1, (1, 4), (0, 2, 3, 4)) * -1
    del tmp15
    tmp12 = einsum(t1, (0, 1), tmp11, (2, 3, 4, 1), (2, 0, 3, 4)) * -1
    del tmp11
    tmp23 = v.oooo.copy()
    tmp23 += einsum(t1, (0, 1), tmp0, (2, 3, 4, 1), (4, 3, 0, 2)) * -1
    tmp19 = einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp14 = einsum(t2, (0, 1, 2, 3), tmp4, (1, 4), (4, 0, 2, 3))
    tmp17 = einsum(t1, (0, 1), tmp16, (2, 0, 3, 4), (2, 3, 4, 1)) * -1
    del tmp16
    tmp21 = einsum(v.ovov, (0, 1, 2, 3), t1, (4, 3), (4, 0, 2, 1))
    tmp5 = t2.copy()
    tmp5 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2
    tmp13 = einsum(t1, (0, 1), tmp12, (2, 3, 0, 4), (3, 2, 1, 4)) * -1
    del tmp12
    t2new = tmp13.transpose((0, 1, 3, 2)).copy()
    t2new += tmp13 * -1
    del tmp13
    tmp24 = v.ooov.copy() * -1
    tmp24 += einsum(tmp23, (0, 1, 2, 3), t1, (0, 4), (2, 3, 1, 4)) * -1
    del tmp23
    t2new += einsum(t1, (0, 1), tmp24, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp24
    tmp20 = einsum(t1, (0, 1), tmp19, (0, 2, 3, 4), (2, 3, 1, 4))
    del tmp19
    t2new += tmp20.transpose((0, 1, 3, 2)) * -1
    t2new += tmp20
    del tmp20
    tmp8 = einsum(t2, (0, 1, 2, 3), f.oo, (4, 1), (4, 0, 2, 3))
    t2new += tmp8.transpose((1, 0, 3, 2))
    t2new += tmp8.transpose((0, 1, 3, 2)) * -1
    del tmp8
    tmp18 = tmp14.transpose((0, 1, 3, 2)).copy() * -1
    del tmp14
    tmp18 += tmp17.transpose((0, 1, 3, 2))
    del tmp17
    t2new += tmp18.transpose((1, 0, 2, 3)) * -1
    t2new += tmp18
    del tmp18
    tmp7 = einsum(v.vvvv, (0, 1, 2, 3), t1, (4, 3), (4, 0, 1, 2))
    t2new += einsum(tmp7, (0, 1, 2, 3), t1, (4, 3), (4, 0, 2, 1)) * -1
    del tmp7
    tmp22 = einsum(t1, (0, 1), tmp21, (2, 0, 3, 4), (2, 3, 1, 4))
    del tmp21
    t2new += tmp22.transpose((1, 0, 3, 2))
    t2new += tmp22.transpose((1, 0, 2, 3)) * -1
    t2new += tmp22.transpose((0, 1, 3, 2)) * -1
    t2new += tmp22
    del tmp22
    tmp9 = einsum(t2, (0, 1, 2, 3), f.vv, (4, 3), (0, 1, 4, 2))
    t2new += tmp9.transpose((1, 0, 3, 2)) * -1
    t2new += tmp9.transpose((1, 0, 2, 3))
    del tmp9
    tmp10 = einsum(v.ovvv, (0, 1, 2, 3), t1, (4, 1), (4, 0, 2, 3))
    t2new += tmp10.transpose((1, 0, 3, 2)) * -1
    t2new += tmp10.transpose((0, 1, 3, 2))
    del tmp10
    tmp6 = f.oo.copy()
    tmp6 += tmp4
    del tmp4
    tmp6 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 1), (2, 3))
    tmp6 += einsum(tmp5, (0, 1, 2, 3), v.oovv, (4, 0, 2, 3), (4, 1)) * -0.5
    del tmp5
    t1new = einsum(tmp6, (0, 1), t1, (0, 2), (1, 2)) * -1
    del tmp6
    tmp3 = f.ov.copy()
    tmp3 += einsum(t1, (0, 1), v.oovv, (2, 0, 3, 1), (2, 3))
    t1new += einsum(tmp3, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    del tmp3
    tmp1 = v.ovoo.transpose((0, 2, 3, 1)).copy() * -1
    tmp1 += tmp0.transpose((0, 2, 1, 3)) * -1
    del tmp0
    t1new += einsum(tmp1, (0, 1, 2, 3), t2, (1, 2, 4, 3), (0, 4)) * 0.5
    del tmp1
    tmp2 = t2.copy() * 0.5
    tmp2 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    t1new += einsum(tmp2, (0, 1, 2, 3), v.ovvv, (0, 4, 2, 3), (1, 4))
    del tmp2
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t1new += f.ov
    t1new += einsum(t1, (0, 1), v.ovov, (2, 1, 0, 3), (2, 3)) * -1
    t2new += einsum(t1, (0, 1), v.ooov, (2, 3, 0, 4), (2, 3, 4, 1))
    t2new += v.oovv

    return {f"t1new": t1new, f"t2new": t2new}

def update_lams(f=None, l1=None, l2=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T22:01:48.647912.

    Parameters
    ----------
    f : array
        Fock matrix.
    l1 : array
        L1 amplitudes.
    l2 : array
        L2 amplitudes.
    t1 : array
        T1 amplitudes.
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    l1new : array
        Updated L1 residuals.
    l2new : array
        Updated L2 residuals.
    """

    tmp5 = einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp2 = einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp10 = v.ooov.copy()
    tmp10 += tmp5.transpose((2, 1, 0, 3)) * 0.5
    tmp0 = einsum(l2, (0, 1, 2, 3), t1, (4, 1), (2, 3, 4, 0))
    l2new = einsum(tmp0, (0, 1, 2, 3), v.ovvv, (2, 3, 4, 5), (4, 5, 0, 1))
    l1new = einsum(tmp2, (0, 1, 2, 3), tmp0, (4, 0, 1, 2), (3, 4))
    l1new += einsum(v.ovov, (0, 1, 2, 3), tmp0, (4, 2, 0, 1), (3, 4)) * -1
    tmp7 = einsum(v.oovv, (0, 1, 2, 3), t1, (1, 3), (0, 2))
    tmp24 = einsum(v.ooov, (0, 1, 2, 3), t1, (1, 3), (0, 2))
    tmp18 = einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    tmp36 = einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    tmp22 = einsum(t1, (0, 1), v.ovvv, (0, 2, 3, 1), (2, 3))
    tmp6 = v.ovoo.transpose((0, 2, 3, 1)).copy() * -1
    tmp6 += tmp5.transpose((0, 2, 1, 3)) * -1
    tmp42 = v.ovov.transpose((0, 2, 3, 1)).copy() * -1
    tmp42 += tmp2
    tmp11 = einsum(tmp10, (0, 1, 2, 3), t1, (4, 3), (0, 1, 2, 4))
    del tmp10
    tmp19 = einsum(t2, (0, 1, 2, 3), l2, (2, 3, 4, 1), (4, 0))
    tmp3 = einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    l1new += einsum(tmp3, (0, 1, 2, 3), v.ovoo, (1, 4, 3, 2), (4, 0)) * 0.25
    tmp4 = einsum(t1, (0, 1), tmp0, (2, 3, 4, 1), (2, 3, 0, 4))
    l2new += einsum(tmp4, (0, 1, 2, 3), v.oovv, (3, 2, 4, 5), (4, 5, 1, 0)) * 0.5
    l1new += einsum(tmp4, (0, 1, 2, 3), v.ooov, (2, 3, 1, 4), (4, 0)) * -0.5
    tmp45 = einsum(t1, (0, 1), tmp7, (2, 1), (0, 2))
    tmp31 = einsum(v.ovvv, (0, 1, 2, 3), l1, (1, 4), (4, 0, 2, 3))
    tmp30 = einsum(l2, (0, 1, 2, 3), f.oo, (4, 3), (4, 2, 0, 1))
    tmp39 = einsum(l2, (0, 1, 2, 3), tmp24, (4, 3), (2, 4, 0, 1))
    tmp38 = einsum(v.oovv, (0, 1, 2, 3), tmp18, (4, 1), (4, 0, 2, 3))
    tmp37 = einsum(l2, (0, 1, 2, 3), tmp36, (4, 3), (4, 2, 0, 1))
    del tmp36
    tmp26 = einsum(tmp0, (0, 1, 2, 3), f.ov, (2, 4), (0, 1, 4, 3))
    tmp27 = einsum(tmp22, (0, 1), l2, (2, 0, 3, 4), (3, 4, 2, 1))
    tmp28 = einsum(l1, (0, 1), tmp6, (1, 2, 3, 4), (2, 3, 4, 0))
    tmp43 = einsum(tmp42, (0, 1, 2, 3), l2, (4, 2, 5, 0), (1, 5, 3, 4))
    del tmp42
    tmp41 = einsum(v.ooov, (0, 1, 2, 3), tmp0, (4, 2, 1, 5), (4, 0, 5, 3))
    tmp33 = einsum(f.vv, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    tmp34 = einsum(tmp0, (0, 1, 2, 3), tmp7, (2, 4), (0, 1, 3, 4))
    tmp12 = v.oooo.copy() * -0.5
    tmp12 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (4, 5, 0, 1)) * -0.25
    tmp12 += tmp11.transpose((1, 0, 3, 2)) * -1
    tmp8 = f.ov.copy()
    tmp8 += tmp7
    tmp9 = v.ovov.transpose((0, 2, 1, 3)).copy()
    tmp9 += tmp2.transpose((0, 1, 3, 2)) * -0.5
    del tmp2
    tmp20 = tmp18.copy() * 2
    del tmp18
    tmp20 += tmp19
    del tmp19
    l1new += einsum(tmp20, (0, 1), tmp8, (1, 2), (2, 0)) * -0.5
    l1new += einsum(tmp20, (0, 1), v.ooov, (2, 1, 0, 3), (3, 2)) * 0.5
    tmp15 = tmp3.transpose((1, 0, 3, 2)).copy() * -0.5
    del tmp3
    tmp15 += tmp4.transpose((0, 1, 3, 2))
    del tmp4
    tmp46 = einsum(tmp45, (0, 1), l2, (2, 3, 4, 0), (4, 1, 2, 3))
    del tmp45
    l2new += tmp46.transpose((2, 3, 1, 0))
    l2new += tmp46.transpose((2, 3, 0, 1)) * -1
    del tmp46
    tmp32 = tmp30.transpose((0, 1, 3, 2)).copy() * -1
    del tmp30
    tmp32 += tmp31.transpose((0, 1, 3, 2))
    del tmp31
    l2new += tmp32.transpose((2, 3, 1, 0)) * -1
    l2new += tmp32.transpose((2, 3, 0, 1))
    del tmp32
    tmp47 = einsum(tmp0, (0, 1, 2, 3), tmp5, (0, 4, 2, 5), (1, 4, 3, 5)) * -1
    del tmp5
    l2new += tmp47.transpose((3, 2, 1, 0)) * -1
    l2new += tmp47.transpose((2, 3, 1, 0))
    l2new += tmp47.transpose((3, 2, 0, 1))
    l2new += tmp47.transpose((2, 3, 0, 1)) * -1
    del tmp47
    tmp40 = tmp37.transpose((0, 1, 3, 2)).copy() * -1
    del tmp37
    tmp40 += tmp38.transpose((0, 1, 3, 2)) * -1
    del tmp38
    tmp40 += tmp39.transpose((0, 1, 3, 2))
    del tmp39
    l2new += tmp40.transpose((3, 2, 1, 0))
    l2new += tmp40.transpose((3, 2, 0, 1)) * -1
    del tmp40
    tmp29 = tmp26.copy() * -1
    del tmp26
    tmp29 += tmp27
    del tmp27
    tmp29 += tmp28.transpose((1, 0, 3, 2))
    del tmp28
    l2new += tmp29.transpose((3, 2, 0, 1))
    l2new += tmp29.transpose((2, 3, 0, 1)) * -1
    del tmp29
    tmp44 = einsum(l1, (0, 1), tmp7, (2, 3), (1, 2, 0, 3))
    del tmp7
    tmp44 += tmp41
    del tmp41
    tmp44 += tmp43.transpose((1, 0, 3, 2))
    del tmp43
    l2new += tmp44.transpose((3, 2, 1, 0))
    l2new += tmp44.transpose((2, 3, 1, 0)) * -1
    l2new += tmp44.transpose((3, 2, 0, 1)) * -1
    l2new += tmp44.transpose((2, 3, 0, 1))
    del tmp44
    tmp48 = v.oooo.copy() * 0.5
    tmp48 += tmp11.transpose((2, 3, 1, 0)) * -1
    del tmp11
    l2new += einsum(l2, (0, 1, 2, 3), tmp48, (2, 3, 4, 5), (0, 1, 4, 5))
    del tmp48
    tmp35 = tmp33.transpose((1, 0, 2, 3)).copy() * -1
    del tmp33
    tmp35 += tmp34
    del tmp34
    l2new += tmp35.transpose((3, 2, 0, 1))
    l2new += tmp35.transpose((2, 3, 0, 1)) * -1
    del tmp35
    tmp13 = v.ovoo.transpose((0, 2, 3, 1)).copy() * -0.5
    tmp13 += einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (0, 4, 5, 1)) * -0.25
    tmp13 += einsum(tmp6, (0, 1, 2, 3), t2, (4, 1, 5, 3), (2, 4, 0, 5))
    del tmp6
    tmp13 += einsum(t2, (0, 1, 2, 3), tmp8, (4, 3), (4, 0, 1, 2)) * 0.5
    tmp13 += einsum(t1, (0, 1), tmp9, (2, 3, 1, 4), (3, 2, 0, 4)) * -1
    del tmp9
    tmp13 += einsum(tmp12, (0, 1, 2, 3), t1, (0, 4), (1, 3, 2, 4)) * -1
    del tmp12
    l1new += einsum(l2, (0, 1, 2, 3), tmp13, (4, 2, 3, 1), (0, 4))
    del tmp13
    tmp1 = einsum(l2, (0, 1, 2, 3), t2, (4, 3, 5, 1), (2, 4, 0, 5))
    l1new += einsum(v.ovvv, (0, 1, 2, 3), tmp1, (4, 0, 1, 3), (2, 4)) * -1
    del tmp1
    tmp21 = t1.copy() * -2
    tmp21 += einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3)) * -2
    tmp21 += einsum(t2, (0, 1, 2, 3), tmp0, (1, 0, 4, 3), (4, 2))
    tmp21 += einsum(tmp20, (0, 1), t1, (0, 2), (1, 2))
    del tmp20
    l1new += einsum(tmp21, (0, 1), v.oovv, (2, 0, 3, 1), (3, 2)) * -0.5
    del tmp21
    tmp17 = einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    tmp17 += einsum(t2, (0, 1, 2, 3), l2, (4, 3, 0, 1), (4, 2)) * 0.5
    l1new += einsum(tmp17, (0, 1), v.ovvv, (2, 0, 3, 1), (3, 2))
    del tmp17
    tmp16 = einsum(t2, (0, 1, 2, 3), l1, (3, 4), (4, 0, 1, 2))
    tmp16 += einsum(tmp0, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 2, 5)) * -2
    del tmp0
    tmp16 += einsum(tmp15, (0, 1, 2, 3), t1, (0, 4), (1, 3, 2, 4)) * -1
    del tmp15
    l1new += einsum(tmp16, (0, 1, 2, 3), v.oovv, (1, 2, 4, 3), (4, 0)) * 0.5
    del tmp16
    tmp25 = f.oo.copy()
    tmp25 += tmp24.transpose((1, 0))
    del tmp24
    tmp25 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4)) * 0.5
    tmp25 += einsum(t1, (0, 1), tmp8, (2, 1), (0, 2))
    del tmp8
    l1new += einsum(l1, (0, 1), tmp25, (1, 2), (0, 2)) * -1
    del tmp25
    tmp14 = v.ovvv.copy()
    tmp14 += einsum(t1, (0, 1), v.vvvv, (2, 1, 3, 4), (0, 2, 3, 4)) * -1
    l1new += einsum(tmp14, (0, 1, 2, 3), l2, (2, 3, 4, 0), (1, 4)) * -0.5
    del tmp14
    tmp23 = f.vv.copy()
    tmp23 += tmp22 * -1
    del tmp22
    l1new += einsum(tmp23, (0, 1), l1, (0, 2), (1, 2))
    del tmp23
    l1new += f.ov.transpose((1, 0))
    l1new += einsum(l1, (0, 1), v.ovov, (2, 0, 1, 3), (3, 2)) * -1
    l2new += einsum(l2, (0, 1, 2, 3), v.vvvv, (4, 5, 0, 1), (4, 5, 2, 3)) * 0.5
    l2new += v.oovv.transpose((2, 3, 0, 1))
    l2new += einsum(l1, (0, 1), f.ov, (2, 3), (3, 0, 2, 1))
    l2new += einsum(f.ov, (0, 1), l1, (2, 3), (2, 1, 0, 3)) * -1
    l2new += einsum(f.ov, (0, 1), l1, (2, 3), (1, 2, 3, 0)) * -1
    l2new += einsum(l1, (0, 1), f.ov, (2, 3), (0, 3, 1, 2))

    return {f"l1new": l1new, f"l2new": l2new}

def make_rdm1_f(l1=None, l2=None, t1=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T22:01:49.327918.

    Parameters
    ----------
    l1 : array
        L1 amplitudes.
    l2 : array
        L2 amplitudes.
    t1 : array
        T1 amplitudes.
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
    tmp1 = einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    rdm1.oo = tmp1.transpose((1, 0)).copy() * -1
    tmp0 = einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    rdm1.oo += tmp0.transpose((1, 0)) * -0.5
    tmp3 = tmp1.copy()
    del tmp1
    tmp3 += tmp0 * 0.5
    del tmp0
    rdm1.ov = einsum(tmp3, (0, 1), t1, (0, 2), (1, 2)) * -1
    del tmp3
    tmp2 = einsum(l2, (0, 1, 2, 3), t1, (4, 1), (2, 3, 4, 0))
    rdm1.ov += einsum(t2, (0, 1, 2, 3), tmp2, (1, 0, 4, 3), (4, 2)) * -0.5
    del tmp2
    rdm1.oo += delta.oo
    del delta
    rdm1.ov += t1
    rdm1.ov += einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3))
    rdm1.vo = l1.copy()
    rdm1.vv = einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    rdm1.vv += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4)) * 0.5
    rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])

    return rdm1

def make_rdm2_f(l1=None, l2=None, t1=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T22:02:00.305118.

    Parameters
    ----------
    l1 : array
        L1 amplitudes.
    l2 : array
        L2 amplitudes.
    t1 : array
        T1 amplitudes.
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
    tmp3 = einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    rdm2.oovo = einsum(t1, (0, 1), tmp3, (2, 3), (3, 0, 1, 2))
    rdm2.oovo += einsum(t1, (0, 1), tmp3, (2, 3), (0, 3, 1, 2)) * -1
    rdm2.ooov = einsum(t1, (0, 1), tmp3, (2, 3), (3, 0, 2, 1)) * -1
    rdm2.ooov += einsum(t1, (0, 1), tmp3, (2, 3), (0, 3, 2, 1))
    rdm2.oooo = einsum(delta.oo, (0, 1), tmp3, (2, 3), (0, 3, 1, 2)) * -1
    rdm2.oooo += einsum(tmp3, (0, 1), delta.oo, (2, 3), (1, 2, 3, 0))
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp3, (2, 3), (0, 3, 2, 1))
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp3, (2, 3), (3, 0, 2, 1)) * -1
    tmp1 = einsum(l2, (0, 1, 2, 3), t1, (4, 1), (2, 3, 4, 0))
    rdm2.vooo = tmp1.transpose((3, 2, 1, 0)).copy()
    rdm2.ovoo = tmp1.transpose((2, 3, 1, 0)).copy() * -1
    tmp4 = einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    rdm2.oovo += einsum(t1, (0, 1), tmp4, (2, 3), (3, 0, 1, 2)) * 0.5
    rdm2.oovo += einsum(t1, (0, 1), tmp4, (2, 3), (0, 3, 1, 2)) * -0.5
    rdm2.ooov += einsum(t1, (0, 1), tmp4, (2, 3), (3, 0, 2, 1)) * -0.5
    rdm2.ooov += einsum(t1, (0, 1), tmp4, (2, 3), (0, 3, 2, 1)) * 0.5
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp4, (2, 3), (0, 3, 1, 2)) * -0.5
    rdm2.oooo += einsum(tmp4, (0, 1), delta.oo, (2, 3), (1, 2, 3, 0)) * 0.5
    rdm2.oooo += einsum(tmp4, (0, 1), delta.oo, (2, 3), (2, 1, 0, 3)) * 0.5
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp4, (2, 3), (3, 0, 2, 1)) * -0.5
    tmp21 = einsum(l2, (0, 1, 2, 3), t2, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2.vovo = tmp21.transpose((2, 1, 3, 0)).copy() * -1
    rdm2.voov = tmp21.transpose((2, 1, 0, 3)).copy()
    rdm2.ovvo = tmp21.transpose((1, 2, 3, 0)).copy()
    rdm2.ovov = tmp21.transpose((1, 2, 0, 3)).copy() * -1
    tmp5 = einsum(t2, (0, 1, 2, 3), l1, (3, 4), (4, 0, 1, 2))
    rdm2.oovo += tmp5.transpose((2, 1, 3, 0))
    rdm2.ooov += tmp5.transpose((2, 1, 0, 3)) * -1
    tmp11 = einsum(tmp3, (0, 1), t1, (0, 2), (1, 2))
    tmp10 = einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3))
    tmp6 = einsum(tmp1, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 2, 4, 5)) * -1
    rdm2.oovo += tmp6.transpose((2, 1, 3, 0)) * -1
    rdm2.oovo += tmp6.transpose((1, 2, 3, 0))
    rdm2.ooov += tmp6.transpose((2, 1, 0, 3))
    rdm2.ooov += tmp6.transpose((1, 2, 0, 3)) * -1
    tmp2 = einsum(t1, (0, 1), tmp1, (2, 3, 4, 1), (2, 3, 0, 4))
    rdm2.oooo += tmp2.transpose((2, 3, 1, 0)) * -1
    tmp0 = einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2.oooo += tmp0.transpose((3, 2, 1, 0)) * 0.5
    tmp14 = tmp3.copy() * 2
    tmp14 += tmp4
    tmp25 = einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4))
    tmp30 = einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    rdm2.ovvv = einsum(tmp30, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1
    rdm2.ovvv += einsum(tmp30, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    tmp29 = einsum(t1, (0, 1), tmp1, (2, 0, 3, 4), (2, 3, 4, 1))
    rdm2.vovo += tmp29.transpose((2, 1, 3, 0))
    rdm2.voov += tmp29.transpose((2, 1, 0, 3)) * -1
    rdm2.ovvo += tmp29.transpose((1, 2, 3, 0)) * -1
    rdm2.ovov += tmp29.transpose((1, 2, 0, 3))
    tmp36 = einsum(t1, (0, 1), tmp21, (0, 2, 3, 4), (2, 3, 1, 4))
    rdm2.vovv = tmp36.transpose((1, 0, 3, 2)).copy() * -1
    rdm2.vovv += tmp36.transpose((1, 0, 2, 3))
    tmp8 = einsum(tmp4, (0, 1), t1, (0, 2), (1, 2))
    tmp7 = einsum(tmp1, (0, 1, 2, 3), t2, (1, 0, 4, 3), (2, 4))
    tmp20 = einsum(t1, (0, 1), tmp5, (0, 2, 3, 4), (2, 3, 1, 4))
    del tmp5
    tmp22 = einsum(tmp21, (0, 1, 2, 3), t2, (4, 0, 5, 2), (4, 1, 5, 3))
    del tmp21
    tmp12 = tmp10.copy()
    tmp12 += tmp11 * -1
    del tmp11
    rdm2.ooov += einsum(delta.oo, (0, 1), tmp12, (2, 3), (0, 2, 1, 3))
    rdm2.ooov += einsum(tmp12, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1
    tmp18 = einsum(t1, (0, 1), tmp6, (0, 2, 3, 4), (2, 3, 1, 4))
    del tmp6
    tmp13 = tmp0.transpose((1, 0, 3, 2)).copy() * -0.5
    tmp13 += tmp2.transpose((0, 1, 3, 2))
    rdm2.oovv = einsum(tmp13, (0, 1, 2, 3), t2, (0, 1, 4, 5), (3, 2, 4, 5)) * 0.5
    rdm2.ooov += einsum(tmp13, (0, 1, 2, 3), t1, (0, 4), (3, 2, 1, 4)) * -1
    tmp15 = einsum(t1, (0, 1), tmp14, (0, 2), (2, 1)) * 0.5
    del tmp14
    tmp38 = einsum(t1, (0, 1), l2, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2.vvvv = einsum(tmp38, (0, 1, 2, 3), t1, (0, 4), (1, 2, 4, 3))
    rdm2.vvvo = tmp38.transpose((2, 1, 3, 0)).copy()
    rdm2.vvov = tmp38.transpose((2, 1, 0, 3)).copy() * -1
    del tmp38
    tmp32 = tmp30.copy()
    tmp32 += tmp25 * 0.5
    rdm2.vovv += einsum(t1, (0, 1), tmp32, (2, 3), (2, 0, 3, 1))
    rdm2.vovv += einsum(tmp32, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * -1
    rdm2.vovo += einsum(delta.oo, (0, 1), tmp32, (2, 3), (2, 0, 3, 1))
    rdm2.ovvo += einsum(delta.oo, (0, 1), tmp32, (2, 3), (0, 2, 3, 1)) * -1
    del tmp32
    tmp33 = einsum(tmp1, (0, 1, 2, 3), t2, (1, 0, 4, 5), (2, 3, 4, 5)) * -1
    del tmp1
    rdm2.vovv += tmp33.transpose((1, 0, 3, 2)) * 0.5
    rdm2.ovvv += tmp33.transpose((0, 1, 3, 2)) * -0.5
    del tmp33
    tmp34 = einsum(t1, (0, 1), tmp29, (0, 2, 3, 4), (2, 3, 4, 1)) * -1
    del tmp29
    rdm2.vovv += tmp34.transpose((1, 0, 3, 2))
    rdm2.ovvv += tmp34.transpose((0, 1, 3, 2)) * -1
    del tmp34
    tmp35 = einsum(t2, (0, 1, 2, 3), l1, (4, 1), (0, 4, 2, 3))
    rdm2.vovv += tmp35.transpose((1, 0, 3, 2))
    rdm2.ovvv += tmp35.transpose((0, 1, 3, 2)) * -1
    del tmp35
    tmp37 = tmp36.copy()
    del tmp36
    tmp37 += einsum(tmp25, (0, 1), t1, (2, 3), (2, 0, 3, 1)) * -0.5
    rdm2.ovvv += tmp37.transpose((0, 1, 3, 2))
    rdm2.ovvv += tmp37 * -1
    del tmp37
    tmp31 = tmp30.copy() * 2
    del tmp30
    tmp31 += tmp25
    rdm2.voov += einsum(delta.oo, (0, 1), tmp31, (2, 3), (2, 0, 1, 3)) * -0.5
    rdm2.ovov += einsum(delta.oo, (0, 1), tmp31, (2, 3), (0, 2, 1, 3)) * 0.5
    del tmp31
    tmp26 = einsum(tmp25, (0, 1), t2, (2, 3, 4, 0), (2, 3, 4, 1))
    del tmp25
    rdm2.oovv += tmp26.transpose((0, 1, 3, 2)) * 0.5
    rdm2.oovv += tmp26 * -0.5
    del tmp26
    tmp24 = einsum(t2, (0, 1, 2, 3), tmp3, (1, 4), (4, 0, 2, 3))
    del tmp3
    rdm2.oovv += tmp24.transpose((1, 0, 3, 2))
    rdm2.oovv += tmp24.transpose((0, 1, 3, 2)) * -1
    del tmp24
    tmp27 = einsum(tmp4, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4))
    del tmp4
    rdm2.oovv += tmp27.transpose((1, 0, 3, 2)) * -0.5
    rdm2.oovv += tmp27.transpose((0, 1, 3, 2)) * 0.5
    del tmp27
    tmp9 = tmp7.copy()
    tmp9 += tmp8
    del tmp8
    rdm2.oovv += einsum(t1, (0, 1), tmp9, (2, 3), (2, 0, 3, 1)) * -0.5
    rdm2.oovv += einsum(t1, (0, 1), tmp9, (2, 3), (2, 0, 1, 3)) * 0.5
    rdm2.oovv += einsum(t1, (0, 1), tmp9, (2, 3), (0, 2, 3, 1)) * 0.5
    rdm2.oovv += einsum(t1, (0, 1), tmp9, (2, 3), (0, 2, 1, 3)) * -0.5
    rdm2.ooov += einsum(delta.oo, (0, 1), tmp9, (2, 3), (0, 2, 1, 3)) * -0.5
    rdm2.ooov += einsum(tmp9, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * 0.5
    del tmp9
    tmp23 = tmp20.copy()
    del tmp20
    tmp23 += tmp22
    del tmp22
    rdm2.oovv += tmp23.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += tmp23
    del tmp23
    tmp19 = tmp18.copy()
    del tmp18
    tmp19 += einsum(t1, (0, 1), tmp12, (2, 3), (0, 2, 1, 3)) * -1
    del tmp12
    rdm2.oovv += tmp19.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp19.transpose((1, 0, 2, 3))
    rdm2.oovv += tmp19.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp19 * -1
    del tmp19
    tmp28 = einsum(tmp13, (0, 1, 2, 3), t1, (0, 4), (1, 3, 2, 4)) * -2
    del tmp13
    rdm2.oovv += einsum(t1, (0, 1), tmp28, (0, 2, 3, 4), (2, 3, 1, 4)) * 0.5
    del tmp28
    tmp17 = tmp0.transpose((1, 0, 3, 2)).copy() * -1
    del tmp0
    tmp17 += tmp2.transpose((0, 1, 3, 2)) * 2
    del tmp2
    rdm2.oovo += einsum(tmp17, (0, 1, 2, 3), t1, (0, 4), (3, 2, 4, 1)) * 0.5
    del tmp17
    tmp16 = tmp10.copy() * -1
    del tmp10
    tmp16 += tmp7 * 0.5
    del tmp7
    tmp16 += tmp15
    del tmp15
    rdm2.oovo += einsum(delta.oo, (0, 1), tmp16, (2, 3), (0, 2, 3, 1))
    rdm2.oovo += einsum(tmp16, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3)) * -1
    del tmp16
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (2, 0, 3, 1))
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1
    rdm2.ooov += einsum(t1, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1
    rdm2.ooov += einsum(t1, (0, 1), delta.oo, (2, 3), (2, 0, 3, 1))
    rdm2.oovo += einsum(t1, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3))
    rdm2.oovo += einsum(t1, (0, 1), delta.oo, (2, 3), (2, 0, 1, 3)) * -1
    rdm2.ovoo += einsum(delta.oo, (0, 1), l1, (2, 3), (0, 2, 3, 1)) * -1
    rdm2.ovoo += einsum(l1, (0, 1), delta.oo, (2, 3), (2, 0, 3, 1))
    rdm2.vooo += einsum(delta.oo, (0, 1), l1, (2, 3), (2, 0, 3, 1))
    rdm2.vooo += einsum(l1, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1
    del delta
    rdm2.oovv += t2
    rdm2.oovv += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1
    rdm2.oovv += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2.ovov += einsum(l1, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1
    rdm2.ovvo += einsum(l1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    rdm2.voov += einsum(l1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2.vovo += einsum(l1, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * -1
    rdm2.vvoo = l2.copy()
    rdm2.vvvv += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 5), (0, 1, 4, 5)) * 0.5
    rdm2 = pack_2e(rdm2.oooo, rdm2.ooov, rdm2.oovo, rdm2.ovoo, rdm2.vooo, rdm2.oovv, rdm2.ovov, rdm2.ovvo, rdm2.voov, rdm2.vovo, rdm2.vvoo, rdm2.ovvv, rdm2.vovv, rdm2.vvov, rdm2.vvvo, rdm2.vvvv)
    rdm2 = rdm2.swapaxes(1, 2)

    return rdm2

