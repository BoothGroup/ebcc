"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-03-02T15:24:42.798018
  * python version: 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0]
  * albert version: 0.0.0
  * caller: /home/ollie/git/albert/albert/codegen/einsum.py
  * node: ollie-desktop
  * system: Linux
  * processor: x86_64
  * release: 6.5.0-21-generic
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace


def energy(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-03-02T15:24:42.982804.

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

    tmp0 = einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1), optimize=True) * 2
    tmp0 += t2.transpose((1, 0, 3, 2))
    e_cc = einsum(tmp0, (0, 1, 2, 3), v.oovv, (0, 1, 2, 3), (), optimize=True) * 0.25
    del tmp0
    e_cc += einsum(f.ov, (0, 1), t1, (0, 1), (), optimize=True)

    return e_cc

def update_amps(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-03-02T15:24:49.127945.

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

    tmp2 = einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1), optimize=True)
    tmp2 += t2.transpose((1, 0, 3, 2)) * 0.5
    t2new = einsum(tmp2, (0, 1, 2, 3), v.vvvv, (2, 3, 4, 5), (1, 0, 4, 5), optimize=True) * -1
    tmp7 = einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1), optimize=True) * 2
    tmp7 += t2.transpose((1, 0, 3, 2))
    tmp49 = einsum(v.oovv, (0, 1, 2, 3), tmp7, (4, 5, 2, 3), (1, 0, 5, 4), optimize=True)
    tmp49 += v.oooo.transpose((2, 3, 1, 0)) * -2
    t2new += einsum(tmp7, (0, 1, 2, 3), tmp49, (0, 1, 4, 5), (4, 5, 3, 2), optimize=True) * -0.25
    del tmp49
    tmp45 = einsum(v.ovvv, (0, 1, 2, 3), t1, (4, 3), (4, 0, 1, 2), optimize=True)
    tmp47 = einsum(t1, (0, 1), tmp45, (2, 3, 4, 1), (2, 0, 3, 4), optimize=True) * -1
    tmp48 = einsum(tmp47, (0, 1, 2, 3), t1, (2, 4), (1, 0, 4, 3), optimize=True) * -1
    del tmp47
    t2new += tmp48.transpose((0, 1, 3, 2))
    t2new += tmp48 * -1
    del tmp48
    tmp46 = einsum(tmp45, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 5, 2), optimize=True) * -1
    del tmp45
    t2new += tmp46.transpose((1, 0, 3, 2))
    t2new += tmp46.transpose((1, 0, 2, 3)) * -1
    t2new += tmp46.transpose((0, 1, 3, 2)) * -1
    t2new += tmp46
    del tmp46
    tmp44 = einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4), optimize=True)
    t2new += tmp44.transpose((1, 0, 3, 2))
    t2new += tmp44.transpose((0, 1, 3, 2)) * -1
    del tmp44
    tmp42 = einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4), optimize=True)
    tmp43 = tmp42.transpose((1, 0, 2, 3)).copy() * -1
    del tmp42
    tmp41 = einsum(t1, (0, 1), v.ooov, (2, 3, 0, 4), (2, 3, 1, 4), optimize=True)
    tmp43 += tmp41.transpose((1, 0, 2, 3)) * -1
    del tmp41
    t2new += tmp43.transpose((0, 1, 3, 2))
    t2new += tmp43 * -1
    del tmp43
    tmp38 = einsum(t1, (0, 1), t1, (2, 3), (0, 2, 3, 1), optimize=True) * -1
    tmp38 += t2.transpose((1, 0, 3, 2))
    tmp39 = einsum(tmp38, (0, 1, 2, 3), v.ovov, (0, 4, 5, 2), (1, 5, 3, 4), optimize=True)
    del tmp38
    tmp40 = tmp39.copy() * -1
    del tmp39
    tmp36 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 0, 5, 3), (4, 1, 2, 5), optimize=True) * -1
    tmp37 = einsum(t1, (0, 1), tmp36, (2, 0, 3, 4), (2, 3, 1, 4), optimize=True)
    del tmp36
    tmp40 += tmp37
    del tmp37
    t2new += tmp40.transpose((1, 0, 3, 2))
    t2new += tmp40.transpose((1, 0, 2, 3)) * -1
    t2new += tmp40.transpose((0, 1, 3, 2)) * -1
    t2new += tmp40
    del tmp40
    tmp3 = einsum(t1, (0, 1), v.oovv, (2, 0, 3, 1), (2, 3), optimize=True)
    tmp33 = einsum(tmp3, (0, 1), t1, (2, 1), (2, 0), optimize=True)
    tmp34 = einsum(t2, (0, 1, 2, 3), tmp33, (4, 1), (4, 0, 2, 3), optimize=True) * -1
    del tmp33
    tmp35 = tmp34.transpose((0, 1, 3, 2)).copy() * -1
    del tmp34
    tmp6 = einsum(v.ooov, (0, 1, 2, 3), t1, (0, 3), (1, 2), optimize=True) * -1
    tmp32 = einsum(tmp6, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4), optimize=True) * -1
    tmp35 += tmp32.transpose((0, 1, 3, 2))
    del tmp32
    tmp31 = einsum(v.ovvv, (0, 1, 2, 3), t1, (4, 1), (4, 0, 2, 3), optimize=True)
    tmp35 += tmp31.transpose((0, 1, 3, 2)) * -1
    del tmp31
    t2new += tmp35.transpose((1, 0, 2, 3))
    t2new += tmp35 * -1
    del tmp35
    tmp0 = einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4), optimize=True)
    tmp29 = einsum(t2, (0, 1, 2, 3), tmp0, (4, 1, 5, 3), (4, 0, 5, 2), optimize=True)
    tmp30 = einsum(tmp29, (0, 1, 2, 3), t1, (2, 4), (0, 1, 4, 3), optimize=True)
    del tmp29
    t2new += tmp30.transpose((1, 0, 3, 2)) * -1
    t2new += tmp30.transpose((1, 0, 2, 3))
    t2new += tmp30.transpose((0, 1, 3, 2))
    t2new += tmp30 * -1
    del tmp30
    tmp26 = einsum(v.ooov, (0, 1, 2, 3), t1, (4, 3), (4, 0, 1, 2), optimize=True)
    tmp27 = einsum(tmp7, (0, 1, 2, 3), tmp26, (4, 0, 1, 5), (4, 5, 2, 3), optimize=True) * 0.5
    del tmp26
    tmp28 = tmp27.transpose((0, 1, 3, 2)).copy() * -1
    del tmp27
    tmp24 = einsum(t2, (0, 1, 2, 3), v.oovv, (1, 4, 2, 3), (0, 4), optimize=True) * -1
    tmp25 = einsum(tmp24, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4), optimize=True)
    del tmp24
    tmp28 += tmp25.transpose((0, 1, 3, 2)) * -0.5
    del tmp25
    tmp5 = einsum(t1, (0, 1), f.ov, (2, 1), (2, 0), optimize=True)
    tmp23 = einsum(t2, (0, 1, 2, 3), tmp5, (1, 4), (4, 0, 2, 3), optimize=True)
    tmp28 += tmp23.transpose((0, 1, 3, 2))
    del tmp23
    t2new += tmp28.transpose((1, 0, 2, 3))
    t2new += tmp28 * -1
    del tmp28
    tmp19 = einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 5, 2, 3), (0, 1, 4, 5), optimize=True)
    tmp20 = tmp19.transpose((2, 1, 0, 3)).copy() * 0.5
    del tmp19
    tmp18 = einsum(t2, (0, 1, 2, 3), f.ov, (4, 3), (4, 0, 1, 2), optimize=True)
    tmp20 += tmp18.transpose((0, 2, 1, 3)) * -1
    del tmp18
    tmp21 = einsum(tmp20, (0, 1, 2, 3), t1, (0, 4), (1, 2, 3, 4), optimize=True)
    del tmp20
    tmp22 = tmp21.transpose((1, 0, 3, 2)).copy()
    del tmp21
    tmp16 = einsum(v.oovv, (0, 1, 2, 3), t2, (0, 1, 4, 2), (4, 3), optimize=True) * -1
    tmp17 = einsum(t2, (0, 1, 2, 3), tmp16, (4, 3), (0, 1, 2, 4), optimize=True)
    del tmp16
    tmp22 += tmp17 * 0.5
    del tmp17
    t2new += tmp22.transpose((0, 1, 3, 2))
    t2new += tmp22 * -1
    del tmp22
    tmp13 = einsum(t2, (0, 1, 2, 3), tmp3, (4, 3), (0, 1, 4, 2), optimize=True) * -1
    tmp14 = einsum(t1, (0, 1), tmp13, (2, 3, 0, 4), (2, 3, 1, 4), optimize=True)
    del tmp13
    tmp15 = tmp14.copy() * -1
    del tmp14
    tmp11 = einsum(t2, (0, 1, 2, 3), v.oovv, (1, 4, 3, 5), (0, 4, 2, 5), optimize=True)
    tmp12 = einsum(t2, (0, 1, 2, 3), tmp11, (4, 1, 5, 3), (0, 4, 2, 5), optimize=True)
    del tmp11
    tmp15 += tmp12
    del tmp12
    tmp9 = einsum(t1, (0, 1), v.ovvv, (0, 2, 1, 3), (2, 3), optimize=True) * -1
    tmp10 = einsum(tmp9, (0, 1), t2, (2, 3, 4, 1), (2, 3, 4, 0), optimize=True) * -1
    del tmp9
    tmp15 += tmp10
    del tmp10
    t2new += tmp15.transpose((0, 1, 3, 2)) * -1
    t2new += tmp15
    del tmp15
    t2new += v.oovv.transpose((1, 0, 3, 2))
    tmp8 = einsum(tmp7, (0, 1, 2, 3), v.oovv, (0, 4, 2, 3), (4, 1), optimize=True)
    del tmp7
    tmp8 += tmp6 * 2
    del tmp6
    tmp8 += tmp5 * 2
    del tmp5
    tmp8 += f.oo.transpose((1, 0)) * 2
    t1new = einsum(tmp8, (0, 1), t1, (0, 2), (1, 2), optimize=True) * -0.5
    del tmp8
    tmp4 = tmp3.copy()
    del tmp3
    tmp4 += f.ov
    t1new += einsum(t2, (0, 1, 2, 3), tmp4, (0, 2), (1, 3), optimize=True)
    del tmp4
    t1new += einsum(v.ovvv, (0, 1, 2, 3), tmp2, (0, 4, 2, 3), (4, 1), optimize=True)
    del tmp2
    tmp1 = tmp0.transpose((0, 2, 1, 3)).copy() * -1
    del tmp0
    tmp1 += v.ooov.transpose((2, 1, 0, 3))
    t1new += einsum(tmp1, (0, 1, 2, 3), t2, (1, 2, 3, 4), (0, 4), optimize=True) * -0.5
    del tmp1
    t1new += einsum(t1, (0, 1), f.vv, (2, 1), (0, 2), optimize=True)
    t1new += f.ov
    t1new += einsum(v.ovov, (0, 1, 2, 3), t1, (2, 1), (0, 3), optimize=True) * -1

    return {f"t1new": t1new, f"t2new": t2new}

def update_lams(f=None, l1=None, l2=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-03-02T15:24:59.466285.

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

    tmp1 = einsum(l2, (0, 1, 2, 3), t1, (4, 1), (2, 3, 4, 0), optimize=True)
    tmp2 = einsum(t1, (0, 1), tmp1, (2, 3, 4, 1), (2, 3, 0, 4), optimize=True)
    tmp59 = tmp2.transpose((0, 1, 3, 2)).copy() * 2
    tmp0 = einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5), optimize=True)
    tmp59 += tmp0.transpose((1, 0, 3, 2)) * -1
    l2new = einsum(v.oovv, (0, 1, 2, 3), tmp59, (4, 5, 0, 1), (3, 2, 5, 4), optimize=True) * -0.25
    del tmp59
    tmp5 = einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4), optimize=True)
    tmp11 = tmp5.transpose((2, 1, 0, 3)).copy() * 0.5
    tmp11 += v.ooov.transpose((1, 0, 2, 3)) * -1
    tmp58 = einsum(t1, (0, 1), tmp11, (2, 3, 4, 1), (4, 0, 3, 2), optimize=True) * -4
    tmp10 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (0, 1, 4, 5), optimize=True)
    tmp58 += tmp10.transpose((1, 0, 3, 2))
    tmp58 += v.oooo.transpose((2, 3, 1, 0)) * -2
    l2new += einsum(l2, (0, 1, 2, 3), tmp58, (2, 3, 4, 5), (1, 0, 4, 5), optimize=True) * -0.25
    del tmp58
    l2new += einsum(f.ov, (0, 1), l1, (2, 3), (2, 1, 3, 0), optimize=True)
    l2new += einsum(l1, (0, 1), f.ov, (2, 3), (3, 0, 1, 2), optimize=True) * -1
    l2new += einsum(l1, (0, 1), f.ov, (2, 3), (0, 3, 2, 1), optimize=True) * -1
    l2new += einsum(l1, (0, 1), f.ov, (2, 3), (3, 0, 2, 1), optimize=True)
    tmp56 = einsum(v.ovvv, (0, 1, 2, 3), l1, (1, 4), (4, 0, 2, 3), optimize=True)
    tmp57 = tmp56.transpose((0, 1, 3, 2)).copy()
    del tmp56
    tmp55 = einsum(f.oo, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3), optimize=True)
    tmp57 += tmp55.transpose((0, 1, 3, 2)) * -1
    del tmp55
    l2new += tmp57.transpose((2, 3, 1, 0)) * -1
    l2new += tmp57.transpose((2, 3, 0, 1))
    del tmp57
    tmp4 = einsum(v.ovvv, (0, 1, 2, 3), t1, (4, 3), (4, 0, 1, 2), optimize=True)
    tmp52 = tmp4.copy()
    tmp52 += v.ovov.transpose((2, 0, 1, 3)) * -1
    tmp53 = einsum(tmp52, (0, 1, 2, 3), l2, (2, 4, 0, 5), (1, 5, 3, 4), optimize=True)
    del tmp52
    tmp54 = tmp53.transpose((1, 0, 3, 2)).copy()
    del tmp53
    tmp51 = einsum(tmp1, (0, 1, 2, 3), v.ooov, (4, 2, 1, 5), (0, 4, 3, 5), optimize=True)
    tmp54 += tmp51
    del tmp51
    tmp7 = einsum(t1, (0, 1), v.oovv, (2, 0, 3, 1), (2, 3), optimize=True)
    tmp54 += einsum(l1, (0, 1), tmp7, (2, 3), (1, 2, 0, 3), optimize=True)
    l2new += tmp54.transpose((3, 2, 1, 0))
    l2new += tmp54.transpose((2, 3, 1, 0)) * -1
    l2new += tmp54.transpose((3, 2, 0, 1)) * -1
    l2new += tmp54.transpose((2, 3, 0, 1))
    del tmp54
    tmp49 = einsum(tmp7, (0, 1), tmp1, (2, 3, 0, 4), (2, 3, 4, 1), optimize=True)
    tmp50 = tmp49.copy()
    del tmp49
    tmp48 = einsum(f.vv, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2), optimize=True)
    tmp50 += tmp48.transpose((1, 0, 2, 3)) * -1
    del tmp48
    l2new += tmp50.transpose((3, 2, 0, 1))
    l2new += tmp50.transpose((2, 3, 0, 1)) * -1
    del tmp50
    tmp46 = einsum(tmp7, (0, 1), t1, (2, 1), (2, 0), optimize=True)
    tmp47 = einsum(l2, (0, 1, 2, 3), tmp46, (3, 4), (2, 4, 0, 1), optimize=True)
    del tmp46
    l2new += tmp47.transpose((2, 3, 1, 0))
    l2new += tmp47.transpose((2, 3, 0, 1)) * -1
    del tmp47
    tmp6 = tmp5.transpose((0, 2, 1, 3)).copy() * -1
    tmp6 += v.ooov.transpose((2, 1, 0, 3))
    tmp44 = einsum(tmp6, (0, 1, 2, 3), l1, (4, 0), (1, 2, 3, 4), optimize=True)
    tmp45 = tmp44.transpose((1, 0, 3, 2)).copy()
    del tmp44
    tmp41 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 2), (3, 4), optimize=True) * -1
    tmp42 = tmp41.copy() * 0.5
    del tmp41
    tmp24 = einsum(v.ovvv, (0, 1, 2, 3), t1, (0, 3), (1, 2), optimize=True)
    tmp42 += tmp24
    tmp43 = einsum(l2, (0, 1, 2, 3), tmp42, (0, 4), (2, 3, 4, 1), optimize=True)
    del tmp42
    tmp45 += tmp43.transpose((1, 0, 3, 2))
    del tmp43
    tmp17 = einsum(l2, (0, 1, 2, 3), t2, (2, 3, 0, 4), (1, 4), optimize=True)
    tmp40 = einsum(tmp17, (0, 1), v.oovv, (2, 3, 4, 1), (3, 2, 0, 4), optimize=True)
    tmp45 += tmp40 * 0.5
    del tmp40
    tmp39 = einsum(f.ov, (0, 1), tmp1, (2, 3, 0, 4), (2, 3, 1, 4), optimize=True)
    tmp45 += tmp39 * -1
    del tmp39
    l2new += tmp45.transpose((3, 2, 0, 1))
    l2new += tmp45.transpose((2, 3, 0, 1)) * -1
    del tmp45
    tmp27 = einsum(v.oovv, (0, 1, 2, 3), t2, (1, 4, 2, 3), (4, 0), optimize=True) * -1
    tmp36 = tmp27.transpose((1, 0)).copy()
    tmp26 = einsum(v.ooov, (0, 1, 2, 3), t1, (1, 3), (0, 2), optimize=True)
    tmp36 += tmp26 * 2
    tmp37 = einsum(l2, (0, 1, 2, 3), tmp36, (4, 2), (4, 3, 0, 1), optimize=True) * 0.5
    del tmp36
    tmp38 = tmp37.transpose((1, 0, 3, 2)).copy()
    del tmp37
    tmp20 = einsum(l2, (0, 1, 2, 3), t2, (2, 4, 0, 1), (3, 4), optimize=True)
    tmp23 = tmp20.copy() * 0.5
    tmp19 = einsum(t1, (0, 1), l1, (1, 2), (2, 0), optimize=True)
    tmp23 += tmp19
    tmp35 = einsum(tmp23, (0, 1), v.oovv, (1, 2, 3, 4), (0, 2, 3, 4), optimize=True)
    tmp38 += tmp35.transpose((0, 1, 3, 2)) * -1
    del tmp35
    tmp33 = einsum(t1, (0, 1), f.ov, (2, 1), (2, 0), optimize=True)
    tmp34 = einsum(tmp33, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3), optimize=True)
    del tmp33
    tmp38 += tmp34.transpose((0, 1, 3, 2))
    del tmp34
    l2new += tmp38.transpose((3, 2, 1, 0)) * -1
    l2new += tmp38.transpose((3, 2, 0, 1))
    del tmp38
    tmp31 = einsum(tmp1, (0, 1, 2, 3), tmp5, (0, 4, 2, 5), (1, 4, 3, 5), optimize=True) * -1
    del tmp5
    tmp32 = tmp31.copy() * -1
    del tmp31
    tmp29 = einsum(v.oovv, (0, 1, 2, 3), t2, (1, 4, 2, 5), (4, 0, 5, 3), optimize=True) * -1
    tmp30 = einsum(tmp29, (0, 1, 2, 3), l2, (4, 2, 0, 5), (5, 1, 4, 3), optimize=True) * -1
    del tmp29
    tmp32 += tmp30
    del tmp30
    l2new += tmp32.transpose((3, 2, 1, 0))
    l2new += tmp32.transpose((2, 3, 1, 0)) * -1
    l2new += tmp32.transpose((3, 2, 0, 1)) * -1
    l2new += tmp32.transpose((2, 3, 0, 1))
    del tmp32
    l2new += einsum(tmp1, (0, 1, 2, 3), v.ovvv, (2, 3, 4, 5), (4, 5, 0, 1), optimize=True)
    l2new += v.oovv.transpose((3, 2, 1, 0))
    l2new += einsum(l2, (0, 1, 2, 3), v.vvvv, (4, 5, 0, 1), (5, 4, 3, 2), optimize=True) * 0.5
    tmp8 = tmp7.copy()
    del tmp7
    tmp8 += f.ov
    l1new = einsum(tmp23, (0, 1), tmp8, (1, 2), (2, 0), optimize=True) * -1
    tmp28 = einsum(tmp8, (0, 1), t1, (2, 1), (2, 0), optimize=True)
    tmp28 += tmp27 * 0.5
    del tmp27
    tmp28 += tmp26.transpose((1, 0))
    del tmp26
    tmp28 += f.oo.transpose((1, 0))
    l1new += einsum(l1, (0, 1), tmp28, (1, 2), (0, 2), optimize=True) * -1
    del tmp28
    tmp25 = tmp24.copy() * -1
    del tmp24
    tmp25 += f.vv.transpose((1, 0))
    l1new += einsum(l1, (0, 1), tmp25, (0, 2), (2, 1), optimize=True)
    del tmp25
    l1new += einsum(v.ooov, (0, 1, 2, 3), tmp23, (2, 0), (3, 1), optimize=True) * -1
    del tmp23
    tmp21 = tmp20.copy()
    del tmp20
    tmp21 += tmp19 * 2
    del tmp19
    tmp22 = einsum(t1, (0, 1), tmp21, (0, 2), (2, 1), optimize=True) * 0.5
    del tmp21
    tmp22 += einsum(tmp1, (0, 1, 2, 3), t2, (0, 1, 3, 4), (2, 4), optimize=True) * 0.5
    tmp22 += einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3), optimize=True) * -1
    tmp22 += t1 * -1
    l1new += einsum(v.oovv, (0, 1, 2, 3), tmp22, (0, 2), (3, 1), optimize=True) * -1
    del tmp22
    tmp18 = tmp17.copy() * 0.5
    del tmp17
    tmp18 += einsum(t1, (0, 1), l1, (2, 0), (2, 1), optimize=True)
    l1new += einsum(tmp18, (0, 1), v.ovvv, (2, 0, 1, 3), (3, 2), optimize=True) * -1
    del tmp18
    tmp15 = tmp2.transpose((0, 1, 3, 2)).copy()
    tmp15 += tmp0.transpose((1, 0, 3, 2)) * -0.5
    tmp16 = einsum(t1, (0, 1), tmp15, (0, 2, 3, 4), (2, 4, 3, 1), optimize=True) * -0.5
    del tmp15
    tmp16 += einsum(tmp1, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 2, 5), optimize=True) * -1
    tmp16 += einsum(t2, (0, 1, 2, 3), l1, (3, 4), (4, 1, 0, 2), optimize=True) * -0.5
    l1new += einsum(tmp16, (0, 1, 2, 3), v.oovv, (1, 2, 3, 4), (4, 0), optimize=True) * -1
    del tmp16
    tmp14 = einsum(v.vvvv, (0, 1, 2, 3), t1, (4, 1), (4, 0, 3, 2), optimize=True)
    tmp14 += v.ovvv.transpose((0, 1, 3, 2)) * -1
    l1new += einsum(l2, (0, 1, 2, 3), tmp14, (2, 4, 0, 1), (4, 3), optimize=True) * 0.5
    del tmp14
    tmp12 = einsum(tmp11, (0, 1, 2, 3), t1, (4, 3), (1, 0, 4, 2), optimize=True) * -1
    del tmp11
    tmp12 += tmp10.transpose((3, 2, 1, 0)) * -0.25
    del tmp10
    tmp12 += v.oooo.transpose((2, 3, 1, 0)) * 0.5
    tmp13 = einsum(t1, (0, 1), tmp12, (0, 2, 3, 4), (2, 4, 3, 1), optimize=True) * -1
    del tmp12
    tmp9 = tmp4.transpose((0, 1, 3, 2)).copy() * -0.5
    tmp9 += v.ovov.transpose((2, 0, 3, 1))
    tmp13 += einsum(tmp9, (0, 1, 2, 3), t1, (4, 2), (1, 0, 4, 3), optimize=True) * -1
    del tmp9
    tmp13 += einsum(tmp8, (0, 1), t2, (2, 3, 1, 4), (0, 3, 2, 4), optimize=True) * 0.5
    del tmp8
    tmp13 += einsum(t2, (0, 1, 2, 3), tmp6, (4, 0, 5, 2), (5, 1, 4, 3), optimize=True)
    del tmp6
    tmp13 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 5, 2, 3), (4, 1, 0, 5), optimize=True) * 0.25
    tmp13 += v.ooov.transpose((2, 1, 0, 3)) * 0.5
    l1new += einsum(tmp13, (0, 1, 2, 3), l2, (3, 4, 1, 2), (4, 0), optimize=True) * -1
    del tmp13
    l1new += einsum(v.ovov, (0, 1, 2, 3), tmp1, (4, 0, 2, 3), (1, 4), optimize=True) * -1
    l1new += f.ov.transpose((1, 0))
    l1new += einsum(tmp4, (0, 1, 2, 3), tmp1, (4, 0, 1, 2), (3, 4), optimize=True)
    del tmp1, tmp4
    tmp3 = einsum(t2, (0, 1, 2, 3), l2, (4, 3, 1, 5), (5, 0, 4, 2), optimize=True) * -1
    l1new += einsum(tmp3, (0, 1, 2, 3), v.ovvv, (1, 2, 4, 3), (4, 0), optimize=True) * -1
    del tmp3
    l1new += einsum(v.ovov, (0, 1, 2, 3), l1, (1, 2), (3, 0), optimize=True) * -1
    l1new += einsum(v.ooov, (0, 1, 2, 3), tmp2, (4, 2, 1, 0), (3, 4), optimize=True) * 0.5
    del tmp2
    l1new += einsum(v.ooov, (0, 1, 2, 3), tmp0, (4, 2, 0, 1), (3, 4), optimize=True) * -0.25
    del tmp0

    return {f"l1new": l1new, f"l2new": l2new}

def make_rdm1_f(l1=None, l2=None, t1=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-03-02T15:24:59.948892.

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
        oo=np.eye(t1.shape[0]),
        vv=np.eye(t1.shape[1]),
    )
    rdm1.vv = einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 0), (1, 4), optimize=True) * -0.5
    rdm1.vv += einsum(t1, (0, 1), l1, (2, 0), (2, 1), optimize=True)
    rdm1.vo = l1.copy()
    tmp0 = einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4), optimize=True)
    tmp3 = tmp0.copy() * 0.5
    tmp1 = einsum(t1, (0, 1), l1, (1, 2), (2, 0), optimize=True)
    tmp3 += tmp1
    rdm1.ov = einsum(t1, (0, 1), tmp3, (0, 2), (2, 1), optimize=True) * -1
    del tmp3
    rdm1.ov += einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3), optimize=True)
    rdm1.ov += t1
    tmp2 = einsum(l2, (0, 1, 2, 3), t1, (4, 1), (2, 3, 4, 0), optimize=True)
    rdm1.ov += einsum(tmp2, (0, 1, 2, 3), t2, (0, 1, 4, 3), (2, 4), optimize=True) * 0.5
    del tmp2
    rdm1.oo = tmp1.transpose((1, 0)).copy() * -1
    del tmp1
    rdm1.oo += delta.oo.transpose((1, 0))
    del delta
    rdm1.oo += tmp0.transpose((1, 0)) * -0.5
    del tmp0
    rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])

    return rdm1

def make_rdm2_f(l1=None, l2=None, t1=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-03-02T15:25:08.561385.

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
        oo=np.eye(t1.shape[0]),
        vv=np.eye(t1.shape[1]),
    )
    tmp38 = einsum(l2, (0, 1, 2, 3), t1, (3, 4), (2, 0, 1, 4), optimize=True)
    rdm2.vvvv = einsum(tmp38, (0, 1, 2, 3), t1, (0, 4), (1, 2, 4, 3), optimize=True)
    rdm2.vvvv += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 5), (1, 0, 5, 4), optimize=True) * 0.5
    rdm2.vvvo = tmp38.transpose((2, 1, 3, 0)).copy()
    rdm2.vvov = tmp38.transpose((2, 1, 0, 3)).copy() * -1
    del tmp38
    tmp15 = einsum(t2, (0, 1, 2, 3), l2, (3, 4, 5, 1), (5, 0, 4, 2), optimize=True) * -1
    tmp36 = einsum(t1, (0, 1), tmp15, (0, 2, 3, 4), (2, 3, 1, 4), optimize=True)
    rdm2.vovv = tmp36.transpose((1, 0, 3, 2)).copy() * -1
    rdm2.vovv += tmp36.transpose((1, 0, 2, 3))
    tmp18 = einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 0), (1, 4), optimize=True) * -1
    tmp31 = tmp18.copy() * 0.5
    tmp30 = einsum(t1, (0, 1), l1, (2, 0), (2, 1), optimize=True)
    tmp31 += tmp30
    rdm2.vovv += einsum(t1, (0, 1), tmp31, (2, 3), (2, 0, 3, 1), optimize=True)
    rdm2.vovv += einsum(t1, (0, 1), tmp31, (2, 3), (2, 0, 1, 3), optimize=True) * -1
    tmp1 = einsum(l2, (0, 1, 2, 3), t1, (4, 1), (2, 3, 4, 0), optimize=True)
    tmp29 = einsum(tmp1, (0, 1, 2, 3), t1, (1, 4), (0, 2, 3, 4), optimize=True)
    tmp35 = einsum(tmp29, (0, 1, 2, 3), t1, (0, 4), (1, 2, 3, 4), optimize=True) * -1
    rdm2.vovv += tmp35.transpose((1, 0, 3, 2))
    tmp34 = einsum(t2, (0, 1, 2, 3), tmp1, (0, 1, 4, 5), (4, 5, 2, 3), optimize=True)
    rdm2.vovv += tmp34.transpose((1, 0, 3, 2)) * 0.5
    tmp33 = einsum(l1, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4), optimize=True)
    rdm2.vovv += tmp33.transpose((1, 0, 3, 2))
    tmp37 = einsum(t1, (0, 1), tmp18, (2, 3), (0, 2, 1, 3), optimize=True) * -0.5
    tmp37 += tmp36
    del tmp36
    rdm2.ovvv = tmp37.transpose((0, 1, 3, 2)).copy()
    rdm2.ovvv += tmp37 * -1
    del tmp37
    rdm2.ovvv += einsum(tmp30, (0, 1), t1, (2, 3), (2, 0, 1, 3), optimize=True) * -1
    rdm2.ovvv += einsum(t1, (0, 1), tmp30, (2, 3), (0, 2, 1, 3), optimize=True)
    rdm2.ovvv += tmp35.transpose((0, 1, 3, 2)) * -1
    del tmp35
    rdm2.ovvv += tmp34.transpose((0, 1, 3, 2)) * -0.5
    del tmp34
    rdm2.ovvv += tmp33.transpose((0, 1, 3, 2)) * -1
    del tmp33
    rdm2.vvoo = l2.transpose((1, 0, 3, 2)).copy()
    tmp32 = tmp18.copy()
    tmp32 += tmp30 * 2
    del tmp30
    rdm2.vovo = einsum(tmp32, (0, 1), delta.oo, (2, 3), (0, 3, 1, 2), optimize=True) * 0.5
    rdm2.vovo += einsum(t1, (0, 1), l1, (2, 3), (2, 0, 1, 3), optimize=True) * -1
    rdm2.vovo += tmp29.transpose((2, 1, 3, 0))
    rdm2.vovo += tmp15.transpose((2, 1, 3, 0)) * -1
    rdm2.voov = einsum(tmp32, (0, 1), delta.oo, (2, 3), (0, 3, 2, 1), optimize=True) * -0.5
    del tmp32
    rdm2.voov += einsum(t1, (0, 1), l1, (2, 3), (2, 0, 3, 1), optimize=True)
    rdm2.voov += tmp29.transpose((2, 1, 0, 3)) * -1
    rdm2.voov += tmp15.transpose((2, 1, 0, 3))
    rdm2.ovvo = einsum(tmp31, (0, 1), delta.oo, (2, 3), (3, 0, 1, 2), optimize=True) * -1
    rdm2.ovvo += einsum(t1, (0, 1), l1, (2, 3), (0, 2, 1, 3), optimize=True)
    rdm2.ovvo += tmp29.transpose((1, 2, 3, 0)) * -1
    rdm2.ovvo += tmp15.transpose((1, 2, 3, 0))
    rdm2.ovov = einsum(tmp31, (0, 1), delta.oo, (2, 3), (3, 0, 2, 1), optimize=True)
    del tmp31
    rdm2.ovov += einsum(t1, (0, 1), l1, (2, 3), (0, 2, 3, 1), optimize=True) * -1
    rdm2.ovov += tmp29.transpose((1, 2, 0, 3))
    del tmp29
    rdm2.ovov += tmp15.transpose((1, 2, 0, 3)) * -1
    tmp2 = einsum(t1, (0, 1), tmp1, (2, 3, 4, 1), (2, 3, 0, 4), optimize=True)
    tmp12 = tmp2.transpose((0, 1, 3, 2)).copy()
    tmp0 = einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5), optimize=True)
    tmp12 += tmp0.transpose((1, 0, 3, 2)) * -0.5
    tmp28 = einsum(t1, (0, 1), tmp12, (0, 2, 3, 4), (2, 4, 3, 1), optimize=True) * -2
    rdm2.oovv = einsum(tmp28, (0, 1, 2, 3), t1, (0, 4), (1, 2, 4, 3), optimize=True) * 0.5
    del tmp28
    rdm2.oovv += einsum(t2, (0, 1, 2, 3), tmp12, (0, 1, 4, 5), (5, 4, 3, 2), optimize=True) * -0.5
    rdm2.oovv += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1), optimize=True)
    rdm2.oovv += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 3, 1), optimize=True) * -1
    tmp4 = einsum(t1, (0, 1), l1, (1, 2), (2, 0), optimize=True)
    tmp27 = einsum(tmp4, (0, 1), t2, (2, 0, 3, 4), (1, 2, 3, 4), optimize=True)
    rdm2.oovv += tmp27.transpose((1, 0, 3, 2))
    rdm2.oovv += tmp27.transpose((0, 1, 3, 2)) * -1
    del tmp27
    tmp3 = einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4), optimize=True)
    tmp25 = einsum(t1, (0, 1), tmp3, (0, 2), (2, 1), optimize=True)
    tmp26 = tmp25.copy()
    del tmp25
    tmp8 = einsum(tmp1, (0, 1, 2, 3), t2, (0, 1, 4, 3), (2, 4), optimize=True) * -1
    tmp26 += tmp8
    rdm2.oovv += einsum(tmp26, (0, 1), t1, (2, 3), (0, 2, 1, 3), optimize=True) * -0.5
    rdm2.oovv += einsum(t1, (0, 1), tmp26, (2, 3), (2, 0, 1, 3), optimize=True) * 0.5
    rdm2.oovv += einsum(t1, (0, 1), tmp26, (2, 3), (0, 2, 3, 1), optimize=True) * 0.5
    rdm2.oovv += einsum(tmp26, (0, 1), t1, (2, 3), (2, 0, 3, 1), optimize=True) * -0.5
    del tmp26
    tmp22 = einsum(t1, (0, 1), tmp4, (0, 2), (2, 1), optimize=True)
    tmp23 = tmp22.copy() * -1
    del tmp22
    tmp7 = einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3), optimize=True)
    tmp23 += tmp7
    tmp24 = einsum(tmp23, (0, 1), t1, (2, 3), (2, 0, 3, 1), optimize=True) * -1
    del tmp23
    tmp6 = einsum(tmp1, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 2, 4, 5), optimize=True) * -1
    tmp21 = einsum(tmp6, (0, 1, 2, 3), t1, (0, 4), (1, 2, 4, 3), optimize=True)
    tmp24 += tmp21
    del tmp21
    rdm2.oovv += tmp24.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp24.transpose((1, 0, 2, 3))
    rdm2.oovv += tmp24.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp24 * -1
    del tmp24
    tmp20 = einsum(t2, (0, 1, 2, 3), tmp3, (1, 4), (0, 4, 2, 3), optimize=True)
    rdm2.oovv += tmp20.transpose((1, 0, 3, 2)) * -0.5
    rdm2.oovv += tmp20.transpose((0, 1, 3, 2)) * 0.5
    del tmp20
    tmp19 = einsum(tmp18, (0, 1), t2, (2, 3, 4, 0), (2, 3, 4, 1), optimize=True)
    del tmp18
    rdm2.oovv += tmp19.transpose((0, 1, 3, 2)) * 0.5
    rdm2.oovv += tmp19 * -0.5
    del tmp19
    tmp16 = einsum(t2, (0, 1, 2, 3), tmp15, (1, 4, 3, 5), (0, 4, 2, 5), optimize=True)
    del tmp15
    tmp17 = tmp16.copy()
    del tmp16
    tmp5 = einsum(t2, (0, 1, 2, 3), l1, (3, 4), (4, 0, 1, 2), optimize=True)
    tmp14 = einsum(tmp5, (0, 1, 2, 3), t1, (0, 4), (1, 2, 4, 3), optimize=True)
    tmp17 += tmp14
    del tmp14
    rdm2.oovv += tmp17.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += tmp17
    del tmp17
    rdm2.oovv += t2.transpose((1, 0, 3, 2))
    rdm2.vooo = einsum(l1, (0, 1), delta.oo, (2, 3), (0, 3, 2, 1), optimize=True) * -1
    rdm2.vooo += einsum(l1, (0, 1), delta.oo, (2, 3), (0, 3, 1, 2), optimize=True)
    rdm2.vooo += tmp1.transpose((3, 2, 1, 0))
    rdm2.ovoo = einsum(l1, (0, 1), delta.oo, (2, 3), (3, 0, 2, 1), optimize=True)
    rdm2.ovoo += einsum(l1, (0, 1), delta.oo, (2, 3), (3, 0, 1, 2), optimize=True) * -1
    rdm2.ovoo += tmp1.transpose((2, 3, 1, 0)) * -1
    del tmp1
    tmp13 = tmp2.transpose((0, 1, 3, 2)).copy() * 2
    tmp13 += tmp0.transpose((1, 0, 3, 2)) * -1
    rdm2.oovo = einsum(t1, (0, 1), tmp13, (0, 2, 3, 4), (4, 3, 1, 2), optimize=True) * 0.5
    del tmp13
    rdm2.oovo += einsum(tmp4, (0, 1), t1, (2, 3), (1, 2, 3, 0), optimize=True)
    rdm2.oovo += einsum(tmp4, (0, 1), t1, (2, 3), (2, 1, 3, 0), optimize=True) * -1
    tmp9 = tmp3.copy()
    tmp9 += tmp4 * 2
    tmp10 = einsum(t1, (0, 1), tmp9, (0, 2), (2, 1), optimize=True) * 0.5
    del tmp9
    tmp11 = tmp10.copy()
    del tmp10
    tmp11 += tmp8 * 0.5
    del tmp8
    tmp11 += tmp7 * -1
    del tmp7
    rdm2.oovo += einsum(tmp11, (0, 1), delta.oo, (2, 3), (2, 0, 1, 3), optimize=True)
    rdm2.oovo += einsum(tmp11, (0, 1), delta.oo, (2, 3), (0, 3, 1, 2), optimize=True) * -1
    rdm2.oovo += einsum(t1, (0, 1), tmp3, (2, 3), (3, 0, 1, 2), optimize=True) * 0.5
    rdm2.oovo += einsum(tmp3, (0, 1), t1, (2, 3), (2, 1, 3, 0), optimize=True) * -0.5
    rdm2.oovo += tmp6.transpose((2, 1, 3, 0)) * -1
    rdm2.oovo += tmp6.transpose((1, 2, 3, 0))
    rdm2.oovo += einsum(t1, (0, 1), delta.oo, (2, 3), (3, 0, 1, 2), optimize=True) * -1
    rdm2.oovo += einsum(delta.oo, (0, 1), t1, (2, 3), (2, 1, 3, 0), optimize=True)
    rdm2.oovo += tmp5.transpose((2, 1, 3, 0))
    rdm2.ooov = einsum(t1, (0, 1), tmp12, (0, 2, 3, 4), (4, 3, 2, 1), optimize=True) * -1
    del tmp12
    rdm2.ooov += einsum(tmp4, (0, 1), t1, (2, 3), (1, 2, 0, 3), optimize=True) * -1
    rdm2.ooov += einsum(tmp4, (0, 1), t1, (2, 3), (2, 1, 0, 3), optimize=True)
    rdm2.ooov += einsum(tmp11, (0, 1), delta.oo, (2, 3), (2, 0, 3, 1), optimize=True) * -1
    rdm2.ooov += einsum(tmp11, (0, 1), delta.oo, (2, 3), (0, 3, 2, 1), optimize=True)
    del tmp11
    rdm2.ooov += einsum(t1, (0, 1), tmp3, (2, 3), (3, 0, 2, 1), optimize=True) * -0.5
    rdm2.ooov += einsum(tmp3, (0, 1), t1, (2, 3), (2, 1, 0, 3), optimize=True) * 0.5
    rdm2.ooov += tmp6.transpose((2, 1, 0, 3))
    rdm2.ooov += tmp6.transpose((1, 2, 0, 3)) * -1
    del tmp6
    rdm2.ooov += einsum(t1, (0, 1), delta.oo, (2, 3), (3, 0, 2, 1), optimize=True)
    rdm2.ooov += einsum(delta.oo, (0, 1), t1, (2, 3), (2, 1, 0, 3), optimize=True) * -1
    rdm2.ooov += tmp5.transpose((2, 1, 0, 3)) * -1
    del tmp5
    rdm2.oooo = einsum(tmp4, (0, 1), delta.oo, (2, 3), (3, 1, 2, 0), optimize=True) * -1
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp4, (2, 3), (3, 1, 0, 2), optimize=True)
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp4, (2, 3), (1, 3, 2, 0), optimize=True)
    rdm2.oooo += einsum(tmp4, (0, 1), delta.oo, (2, 3), (1, 3, 0, 2), optimize=True) * -1
    del tmp4
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (3, 0, 1, 2), optimize=True) * -1
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (3, 1, 2, 0), optimize=True)
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp3, (2, 3), (0, 3, 1, 2), optimize=True) * -0.5
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp3, (2, 3), (3, 1, 0, 2), optimize=True) * 0.5
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp3, (2, 3), (0, 3, 2, 1), optimize=True) * 0.5
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp3, (2, 3), (3, 1, 2, 0), optimize=True) * -0.5
    del delta, tmp3
    rdm2.oooo += tmp2.transpose((2, 3, 1, 0)) * -1
    del tmp2
    rdm2.oooo += tmp0.transpose((3, 2, 1, 0)) * 0.5
    del tmp0
    rdm2 = pack_2e(rdm2.oooo, rdm2.ooov, rdm2.oovo, rdm2.ovoo, rdm2.vooo, rdm2.oovv, rdm2.ovov, rdm2.ovvo, rdm2.voov, rdm2.vovo, rdm2.vvoo, rdm2.ovvv, rdm2.vovv, rdm2.vvov, rdm2.vvvo, rdm2.vvvv)
    rdm2 = rdm2.swapaxes(1, 2)

    return rdm2

