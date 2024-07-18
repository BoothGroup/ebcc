"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-07-18T20:10:22.664181
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


def energy(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:10:22.833677.

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

    tmp0 = t2.transpose((1, 0, 3, 2)).copy()
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1), optimize=True) * 2
    e_cc = einsum(v.oovv, (0, 1, 2, 3), tmp0, (0, 1, 2, 3), (), optimize=True) * 0.25
    del tmp0
    e_cc += einsum(f.ov, (0, 1), t1, (0, 1), (), optimize=True)

    return e_cc

def update_amps(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:10:27.948736.

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

    tmp3 = einsum(v.oovv, (0, 1, 2, 3), t1, (1, 3), (0, 2), optimize=True)
    tmp21 = einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 5, 2, 3), (0, 1, 4, 5), optimize=True)
    tmp20 = einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4), optimize=True)
    tmp9 = einsum(v.ovvv, (0, 1, 2, 3), t1, (4, 3), (4, 0, 1, 2), optimize=True)
    tmp39 = einsum(v.ooov, (0, 1, 2, 3), t1, (4, 3), (4, 0, 1, 2), optimize=True)
    tmp7 = t2.transpose((1, 0, 3, 2)).copy()
    tmp7 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1), optimize=True) * 2
    tmp37 = einsum(t2, (0, 1, 2, 3), v.oovv, (1, 4, 2, 3), (0, 4), optimize=True) * -1
    tmp5 = einsum(f.ov, (0, 1), t1, (2, 1), (0, 2), optimize=True)
    tmp44 = einsum(tmp3, (0, 1), t1, (2, 1), (2, 0), optimize=True)
    tmp6 = einsum(t1, (0, 1), v.ooov, (0, 2, 3, 1), (2, 3), optimize=True) * -1
    tmp0 = einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4), optimize=True)
    tmp13 = einsum(t2, (0, 1, 2, 3), v.oovv, (1, 4, 3, 5), (0, 4, 2, 5), optimize=True)
    tmp15 = einsum(tmp3, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4), optimize=True) * -1
    tmp11 = einsum(v.ovvv, (0, 1, 2, 3), t1, (0, 2), (1, 3), optimize=True) * -1
    tmp31 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 0, 5, 3), (4, 1, 2, 5), optimize=True) * -1
    tmp33 = t2.transpose((1, 0, 3, 2)).copy()
    tmp33 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 3, 1), optimize=True) * -1
    tmp22 = tmp20.transpose((0, 2, 1, 3)).copy() * -1
    del tmp20
    tmp22 += tmp21.transpose((2, 1, 0, 3)) * 0.5
    del tmp21
    tmp18 = einsum(v.oovv, (0, 1, 2, 3), t2, (0, 1, 4, 2), (4, 3), optimize=True) * -1
    tmp27 = einsum(v.ooov, (0, 1, 2, 3), t1, (2, 4), (0, 1, 4, 3), optimize=True)
    tmp28 = einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4), optimize=True)
    tmp25 = einsum(tmp9, (0, 1, 2, 3), t1, (4, 3), (0, 4, 1, 2), optimize=True) * -1
    tmp40 = einsum(tmp7, (0, 1, 2, 3), tmp39, (4, 0, 1, 5), (4, 5, 2, 3), optimize=True) * 0.5
    del tmp39
    tmp38 = einsum(tmp37, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4), optimize=True)
    del tmp37
    tmp36 = einsum(t2, (0, 1, 2, 3), tmp5, (1, 4), (4, 0, 2, 3), optimize=True)
    tmp45 = einsum(tmp44, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4), optimize=True) * -1
    del tmp44
    tmp42 = einsum(v.ovvv, (0, 1, 2, 3), t1, (4, 1), (4, 0, 2, 3), optimize=True)
    tmp43 = einsum(tmp6, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4), optimize=True) * -1
    tmp47 = einsum(t2, (0, 1, 2, 3), tmp0, (4, 1, 5, 3), (4, 0, 5, 2), optimize=True)
    tmp14 = einsum(tmp13, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 2, 5), optimize=True)
    del tmp13
    tmp16 = einsum(t1, (0, 1), tmp15, (2, 3, 0, 4), (2, 3, 1, 4), optimize=True)
    del tmp15
    tmp12 = einsum(tmp11, (0, 1), t2, (2, 3, 4, 1), (2, 3, 4, 0), optimize=True) * -1
    del tmp11
    tmp32 = einsum(tmp31, (0, 1, 2, 3), t1, (1, 4), (0, 2, 4, 3), optimize=True)
    del tmp31
    tmp34 = einsum(v.ovov, (0, 1, 2, 3), tmp33, (0, 4, 3, 5), (4, 2, 5, 1), optimize=True)
    del tmp33
    tmp23 = einsum(t1, (0, 1), tmp22, (0, 2, 3, 4), (2, 3, 4, 1), optimize=True)
    del tmp22
    tmp19 = einsum(t2, (0, 1, 2, 3), tmp18, (4, 3), (0, 1, 2, 4), optimize=True)
    del tmp18
    tmp29 = tmp27.transpose((1, 0, 2, 3)).copy() * -1
    del tmp27
    tmp29 += tmp28.transpose((1, 0, 2, 3)) * -1
    del tmp28
    t2new = tmp29.transpose((0, 1, 3, 2)).copy()
    t2new += tmp29 * -1
    del tmp29
    tmp26 = einsum(t1, (0, 1), tmp25, (2, 3, 0, 4), (3, 2, 1, 4), optimize=True) * -1
    del tmp25
    t2new += tmp26.transpose((0, 1, 3, 2))
    t2new += tmp26 * -1
    del tmp26
    tmp30 = einsum(t2, (0, 1, 2, 3), f.oo, (4, 1), (4, 0, 2, 3), optimize=True)
    t2new += tmp30.transpose((1, 0, 3, 2))
    t2new += tmp30.transpose((0, 1, 3, 2)) * -1
    del tmp30
    tmp41 = tmp36.transpose((0, 1, 3, 2)).copy()
    del tmp36
    tmp41 += tmp38.transpose((0, 1, 3, 2)) * -0.5
    del tmp38
    tmp41 += tmp40.transpose((0, 1, 3, 2)) * -1
    del tmp40
    t2new += tmp41.transpose((1, 0, 2, 3))
    t2new += tmp41 * -1
    del tmp41
    tmp46 = tmp42.transpose((0, 1, 3, 2)).copy() * -1
    del tmp42
    tmp46 += tmp43.transpose((0, 1, 3, 2))
    del tmp43
    tmp46 += tmp45.transpose((0, 1, 3, 2)) * -1
    del tmp45
    t2new += tmp46.transpose((1, 0, 2, 3))
    t2new += tmp46 * -1
    del tmp46
    tmp10 = einsum(tmp9, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 5, 2), optimize=True) * -1
    del tmp9
    t2new += tmp10.transpose((1, 0, 3, 2))
    t2new += tmp10.transpose((1, 0, 2, 3)) * -1
    t2new += tmp10.transpose((0, 1, 3, 2)) * -1
    t2new += tmp10
    del tmp10
    tmp48 = einsum(t1, (0, 1), tmp47, (2, 3, 0, 4), (2, 3, 1, 4), optimize=True)
    del tmp47
    t2new += tmp48.transpose((1, 0, 3, 2)) * -1
    t2new += tmp48.transpose((1, 0, 2, 3))
    t2new += tmp48.transpose((0, 1, 3, 2))
    t2new += tmp48 * -1
    del tmp48
    tmp17 = tmp12.copy()
    del tmp12
    tmp17 += tmp14
    del tmp14
    tmp17 += tmp16 * -1
    del tmp16
    t2new += tmp17.transpose((0, 1, 3, 2)) * -1
    t2new += tmp17
    del tmp17
    tmp35 = tmp32.copy()
    del tmp32
    tmp35 += tmp34 * -1
    del tmp34
    t2new += tmp35.transpose((1, 0, 3, 2))
    t2new += tmp35.transpose((1, 0, 2, 3)) * -1
    t2new += tmp35.transpose((0, 1, 3, 2)) * -1
    t2new += tmp35
    del tmp35
    tmp24 = tmp19.copy() * 0.5
    del tmp19
    tmp24 += tmp23.transpose((1, 0, 3, 2))
    del tmp23
    t2new += tmp24.transpose((0, 1, 3, 2))
    t2new += tmp24 * -1
    del tmp24
    tmp49 = v.oooo.transpose((2, 3, 1, 0)).copy() * -1
    tmp49 += einsum(v.oovv, (0, 1, 2, 3), tmp7, (4, 5, 2, 3), (1, 0, 5, 4), optimize=True) * 0.5
    t2new += einsum(tmp7, (0, 1, 2, 3), tmp49, (0, 1, 4, 5), (4, 5, 3, 2), optimize=True) * -0.5
    del tmp49
    tmp2 = t2.transpose((1, 0, 3, 2)).copy() * 0.5
    tmp2 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1), optimize=True)
    t2new += einsum(tmp2, (0, 1, 2, 3), v.vvvv, (2, 3, 4, 5), (1, 0, 4, 5), optimize=True) * -1
    t1new = einsum(tmp2, (0, 1, 2, 3), v.ovvv, (0, 4, 2, 3), (1, 4), optimize=True)
    del tmp2
    tmp8 = f.oo.transpose((1, 0)).copy() * 2
    tmp8 += tmp5 * 2
    del tmp5
    tmp8 += tmp6 * 2
    del tmp6
    tmp8 += einsum(v.oovv, (0, 1, 2, 3), tmp7, (0, 4, 2, 3), (1, 4), optimize=True)
    del tmp7
    t1new += einsum(tmp8, (0, 1), t1, (0, 2), (1, 2), optimize=True) * -0.5
    del tmp8
    tmp4 = f.ov.copy()
    tmp4 += tmp3
    del tmp3
    t1new += einsum(t2, (0, 1, 2, 3), tmp4, (0, 2), (1, 3), optimize=True)
    del tmp4
    tmp1 = v.ooov.transpose((2, 1, 0, 3)).copy()
    tmp1 += tmp0.transpose((0, 2, 1, 3)) * -1
    del tmp0
    t1new += einsum(tmp1, (0, 1, 2, 3), t2, (1, 2, 3, 4), (0, 4), optimize=True) * -0.5
    del tmp1
    t1new += f.ov
    t1new += einsum(v.ovov, (0, 1, 2, 3), t1, (2, 1), (0, 3), optimize=True) * -1
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0), optimize=True)
    t2new += v.oovv.transpose((1, 0, 3, 2))

    return {f"t1new": t1new, f"t2new": t2new}

def update_lams(f=None, l1=None, l2=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:10:35.470671.

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

    tmp27 = einsum(v.oovv, (0, 1, 2, 3), t2, (1, 4, 2, 3), (4, 0), optimize=True) * -1
    tmp26 = einsum(t1, (0, 1), v.ooov, (2, 0, 3, 1), (2, 3), optimize=True)
    tmp19 = einsum(l1, (0, 1), t1, (2, 0), (1, 2), optimize=True)
    tmp20 = einsum(l2, (0, 1, 2, 3), t2, (2, 4, 0, 1), (3, 4), optimize=True)
    tmp35 = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 2), (3, 4), optimize=True) * -1
    tmp24 = einsum(v.ovvv, (0, 1, 2, 3), t1, (0, 3), (1, 2), optimize=True)
    tmp5 = einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4), optimize=True)
    tmp4 = einsum(v.ovvv, (0, 1, 2, 3), t1, (4, 3), (4, 0, 1, 2), optimize=True)
    tmp1 = einsum(l2, (0, 1, 2, 3), t1, (4, 1), (2, 3, 4, 0), optimize=True)
    l2new = einsum(v.ovvv, (0, 1, 2, 3), tmp1, (4, 5, 0, 1), (2, 3, 4, 5), optimize=True)
    l1new = einsum(v.ovov, (0, 1, 2, 3), tmp1, (4, 0, 2, 3), (1, 4), optimize=True) * -1
    l1new += einsum(tmp1, (0, 1, 2, 3), tmp4, (1, 2, 3, 4), (4, 0), optimize=True)
    tmp52 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 0, 2, 5), (1, 4, 3, 5), optimize=True) * -1
    tmp43 = einsum(f.ov, (0, 1), t1, (2, 1), (0, 2), optimize=True)
    tmp46 = tmp26.copy() * 2
    tmp46 += tmp27.transpose((1, 0))
    tmp23 = tmp19.copy()
    tmp23 += tmp20 * 0.5
    l1new += einsum(tmp23, (0, 1), v.ooov, (1, 2, 0, 3), (3, 2), optimize=True) * -1
    tmp36 = tmp24.copy()
    tmp36 += tmp35 * 0.5
    del tmp35
    tmp6 = v.ooov.transpose((2, 1, 0, 3)).copy()
    tmp6 += tmp5.transpose((0, 2, 1, 3)) * -1
    tmp17 = einsum(t2, (0, 1, 2, 3), l2, (2, 4, 0, 1), (4, 3), optimize=True)
    tmp7 = einsum(v.oovv, (0, 1, 2, 3), t1, (1, 3), (0, 2), optimize=True)
    tmp30 = v.ovov.transpose((2, 0, 1, 3)).copy() * -1
    tmp30 += tmp4
    tmp0 = einsum(t2, (0, 1, 2, 3), l2, (2, 3, 4, 5), (4, 5, 0, 1), optimize=True)
    l1new += einsum(v.ooov, (0, 1, 2, 3), tmp0, (4, 2, 0, 1), (3, 4), optimize=True) * -0.25
    tmp2 = einsum(tmp1, (0, 1, 2, 3), t1, (4, 3), (0, 1, 4, 2), optimize=True)
    l1new += einsum(tmp2, (0, 1, 2, 3), v.ooov, (3, 2, 1, 4), (4, 0), optimize=True) * 0.5
    tmp11 = v.ooov.transpose((1, 0, 2, 3)).copy() * -1
    tmp11 += tmp5.transpose((2, 1, 0, 3)) * 0.5
    tmp10 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (0, 1, 4, 5), optimize=True)
    tmp53 = einsum(tmp52, (0, 1, 2, 3), l2, (4, 2, 0, 5), (5, 1, 4, 3), optimize=True) * -1
    del tmp52
    tmp54 = einsum(tmp5, (0, 1, 2, 3), tmp1, (0, 4, 2, 5), (4, 1, 5, 3), optimize=True) * -1
    del tmp5
    tmp44 = einsum(l2, (0, 1, 2, 3), tmp43, (4, 3), (4, 2, 0, 1), optimize=True)
    del tmp43
    tmp47 = einsum(l2, (0, 1, 2, 3), tmp46, (4, 2), (4, 3, 0, 1), optimize=True) * 0.5
    del tmp46
    tmp45 = einsum(tmp23, (0, 1), v.oovv, (1, 2, 3, 4), (0, 2, 3, 4), optimize=True)
    tmp40 = einsum(l2, (0, 1, 2, 3), f.oo, (4, 3), (4, 2, 0, 1), optimize=True)
    tmp41 = einsum(v.ovvv, (0, 1, 2, 3), l1, (1, 4), (4, 0, 2, 3), optimize=True)
    tmp37 = einsum(l2, (0, 1, 2, 3), tmp36, (0, 4), (2, 3, 4, 1), optimize=True)
    del tmp36
    tmp38 = einsum(tmp6, (0, 1, 2, 3), l1, (4, 0), (1, 2, 3, 4), optimize=True)
    tmp33 = einsum(f.ov, (0, 1), tmp1, (2, 3, 0, 4), (2, 3, 1, 4), optimize=True)
    tmp34 = einsum(v.oovv, (0, 1, 2, 3), tmp17, (4, 3), (1, 0, 4, 2), optimize=True)
    tmp49 = einsum(f.vv, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2), optimize=True)
    tmp50 = einsum(tmp7, (0, 1), tmp1, (2, 3, 0, 4), (2, 3, 4, 1), optimize=True)
    tmp29 = einsum(tmp1, (0, 1, 2, 3), v.ooov, (4, 2, 1, 5), (0, 4, 3, 5), optimize=True)
    tmp31 = einsum(l2, (0, 1, 2, 3), tmp30, (2, 4, 0, 5), (4, 3, 5, 1), optimize=True)
    del tmp30
    tmp56 = einsum(tmp7, (0, 1), t1, (2, 1), (2, 0), optimize=True)
    tmp21 = tmp19.copy() * 2
    del tmp19
    tmp21 += tmp20
    del tmp20
    tmp15 = tmp0.transpose((1, 0, 3, 2)).copy() * -0.5
    tmp15 += tmp2.transpose((0, 1, 3, 2))
    tmp8 = f.ov.copy()
    tmp8 += tmp7
    l1new += einsum(tmp8, (0, 1), tmp23, (2, 0), (1, 2), optimize=True) * -1
    del tmp23
    tmp12 = v.oooo.transpose((2, 3, 1, 0)).copy() * 0.5
    tmp12 += tmp10.transpose((3, 2, 1, 0)) * -0.25
    tmp12 += einsum(tmp11, (0, 1, 2, 3), t1, (4, 3), (1, 0, 4, 2), optimize=True) * -1
    tmp9 = v.ovov.transpose((2, 0, 3, 1)).copy()
    tmp9 += tmp4.transpose((0, 1, 3, 2)) * -0.5
    del tmp4
    tmp55 = tmp53.copy()
    del tmp53
    tmp55 += tmp54 * -1
    del tmp54
    l2new += tmp55.transpose((3, 2, 1, 0))
    l2new += tmp55.transpose((2, 3, 1, 0)) * -1
    l2new += tmp55.transpose((3, 2, 0, 1)) * -1
    l2new += tmp55.transpose((2, 3, 0, 1))
    del tmp55
    tmp59 = tmp0.transpose((1, 0, 3, 2)).copy() * -1
    del tmp0
    tmp59 += tmp2.transpose((0, 1, 3, 2)) * 2
    del tmp2
    l2new += einsum(tmp59, (0, 1, 2, 3), v.oovv, (2, 3, 4, 5), (5, 4, 1, 0), optimize=True) * -0.25
    del tmp59
    tmp48 = tmp44.transpose((0, 1, 3, 2)).copy()
    del tmp44
    tmp48 += tmp45.transpose((0, 1, 3, 2)) * -1
    del tmp45
    tmp48 += tmp47.transpose((1, 0, 3, 2))
    del tmp47
    l2new += tmp48.transpose((3, 2, 1, 0)) * -1
    l2new += tmp48.transpose((3, 2, 0, 1))
    del tmp48
    tmp42 = tmp40.transpose((0, 1, 3, 2)).copy() * -1
    del tmp40
    tmp42 += tmp41.transpose((0, 1, 3, 2))
    del tmp41
    l2new += tmp42.transpose((2, 3, 1, 0)) * -1
    l2new += tmp42.transpose((2, 3, 0, 1))
    del tmp42
    tmp58 = v.oooo.transpose((2, 3, 1, 0)).copy() * -1
    tmp58 += tmp10.transpose((1, 0, 3, 2)) * 0.5
    del tmp10
    tmp58 += einsum(tmp11, (0, 1, 2, 3), t1, (4, 3), (2, 4, 1, 0), optimize=True) * -2
    del tmp11
    l2new += einsum(l2, (0, 1, 2, 3), tmp58, (2, 3, 4, 5), (1, 0, 4, 5), optimize=True) * -0.5
    del tmp58
    tmp39 = tmp33.copy() * -1
    del tmp33
    tmp39 += tmp34 * 0.5
    del tmp34
    tmp39 += tmp37.transpose((1, 0, 3, 2))
    del tmp37
    tmp39 += tmp38.transpose((1, 0, 3, 2))
    del tmp38
    l2new += tmp39.transpose((3, 2, 0, 1))
    l2new += tmp39.transpose((2, 3, 0, 1)) * -1
    del tmp39
    tmp51 = tmp49.transpose((1, 0, 2, 3)).copy() * -1
    del tmp49
    tmp51 += tmp50
    del tmp50
    l2new += tmp51.transpose((3, 2, 0, 1))
    l2new += tmp51.transpose((2, 3, 0, 1)) * -1
    del tmp51
    tmp32 = einsum(tmp7, (0, 1), l1, (2, 3), (3, 0, 2, 1), optimize=True)
    del tmp7
    tmp32 += tmp29
    del tmp29
    tmp32 += tmp31.transpose((1, 0, 3, 2))
    del tmp31
    l2new += tmp32.transpose((3, 2, 1, 0))
    l2new += tmp32.transpose((2, 3, 1, 0)) * -1
    l2new += tmp32.transpose((3, 2, 0, 1)) * -1
    l2new += tmp32.transpose((2, 3, 0, 1))
    del tmp32
    tmp57 = einsum(tmp56, (0, 1), l2, (2, 3, 4, 0), (4, 1, 2, 3), optimize=True)
    del tmp56
    l2new += tmp57.transpose((2, 3, 1, 0))
    l2new += tmp57.transpose((2, 3, 0, 1)) * -1
    del tmp57
    tmp18 = einsum(t1, (0, 1), l1, (2, 0), (2, 1), optimize=True) * 2
    tmp18 += tmp17
    del tmp17
    l1new += einsum(v.ovvv, (0, 1, 2, 3), tmp18, (1, 2), (3, 0), optimize=True) * -0.5
    del tmp18
    tmp22 = t1.copy() * -1
    tmp22 += einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3), optimize=True) * -1
    tmp22 += einsum(t2, (0, 1, 2, 3), tmp1, (0, 1, 4, 2), (4, 3), optimize=True) * 0.5
    tmp22 += einsum(tmp21, (0, 1), t1, (0, 2), (1, 2), optimize=True) * 0.5
    del tmp21
    l1new += einsum(tmp22, (0, 1), v.oovv, (0, 2, 1, 3), (3, 2), optimize=True) * -1
    del tmp22
    tmp16 = einsum(t2, (0, 1, 2, 3), l1, (3, 4), (4, 1, 0, 2), optimize=True) * -0.5
    tmp16 += einsum(tmp1, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 2, 5), optimize=True) * -1
    del tmp1
    tmp16 += einsum(tmp15, (0, 1, 2, 3), t1, (0, 4), (1, 3, 2, 4), optimize=True) * -0.5
    del tmp15
    l1new += einsum(v.oovv, (0, 1, 2, 3), tmp16, (4, 0, 1, 2), (3, 4), optimize=True) * -1
    del tmp16
    tmp13 = v.ooov.transpose((2, 1, 0, 3)).copy()
    tmp13 += einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (0, 5, 4, 1), optimize=True) * 0.5
    tmp13 += einsum(tmp6, (0, 1, 2, 3), t2, (1, 4, 3, 5), (2, 4, 0, 5), optimize=True) * 2
    del tmp6
    tmp13 += einsum(tmp8, (0, 1), t2, (2, 3, 1, 4), (0, 3, 2, 4), optimize=True)
    tmp13 += einsum(t1, (0, 1), tmp9, (2, 3, 1, 4), (3, 2, 0, 4), optimize=True) * -2
    del tmp9
    tmp13 += einsum(tmp12, (0, 1, 2, 3), t1, (0, 4), (1, 3, 2, 4), optimize=True) * -2
    del tmp12
    l1new += einsum(tmp13, (0, 1, 2, 3), l2, (3, 4, 1, 2), (4, 0), optimize=True) * -0.5
    del tmp13
    tmp3 = einsum(l2, (0, 1, 2, 3), t2, (4, 2, 5, 1), (3, 4, 0, 5), optimize=True) * -1
    l1new += einsum(tmp3, (0, 1, 2, 3), v.ovvv, (1, 2, 4, 3), (4, 0), optimize=True) * -1
    del tmp3
    tmp28 = f.oo.transpose((1, 0)).copy()
    tmp28 += tmp26.transpose((1, 0))
    del tmp26
    tmp28 += tmp27 * 0.5
    del tmp27
    tmp28 += einsum(tmp8, (0, 1), t1, (2, 1), (2, 0), optimize=True)
    del tmp8
    l1new += einsum(tmp28, (0, 1), l1, (2, 0), (2, 1), optimize=True) * -1
    del tmp28
    tmp14 = v.ovvv.transpose((0, 1, 3, 2)).copy() * -1
    tmp14 += einsum(t1, (0, 1), v.vvvv, (2, 1, 3, 4), (0, 2, 4, 3), optimize=True)
    l1new += einsum(tmp14, (0, 1, 2, 3), l2, (2, 3, 0, 4), (1, 4), optimize=True) * 0.5
    del tmp14
    tmp25 = f.vv.transpose((1, 0)).copy()
    tmp25 += tmp24 * -1
    del tmp24
    l1new += einsum(tmp25, (0, 1), l1, (0, 2), (1, 2), optimize=True)
    del tmp25
    l1new += f.ov.transpose((1, 0))
    l1new += einsum(l1, (0, 1), v.ovov, (2, 0, 1, 3), (3, 2), optimize=True) * -1
    l2new += v.oovv.transpose((3, 2, 1, 0))
    l2new += einsum(l2, (0, 1, 2, 3), v.vvvv, (4, 5, 0, 1), (5, 4, 3, 2), optimize=True) * 0.5
    l2new += einsum(l1, (0, 1), f.ov, (2, 3), (3, 0, 2, 1), optimize=True)
    l2new += einsum(f.ov, (0, 1), l1, (2, 3), (2, 1, 0, 3), optimize=True) * -1
    l2new += einsum(f.ov, (0, 1), l1, (2, 3), (1, 2, 3, 0), optimize=True) * -1
    l2new += einsum(l1, (0, 1), f.ov, (2, 3), (0, 3, 1, 2), optimize=True)

    return {f"l1new": l1new, f"l2new": l2new}

def make_rdm1_f(l1=None, l2=None, t1=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:10:35.909741.

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
    tmp1 = einsum(l1, (0, 1), t1, (2, 0), (1, 2), optimize=True)
    rdm1.oo = tmp1.transpose((1, 0)).copy() * -1
    tmp0 = einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4), optimize=True)
    rdm1.oo += tmp0.transpose((1, 0)) * -0.5
    tmp3 = tmp1.copy() * 2
    del tmp1
    tmp3 += tmp0
    del tmp0
    rdm1.ov = einsum(tmp3, (0, 1), t1, (0, 2), (1, 2), optimize=True) * -0.5
    del tmp3
    tmp2 = einsum(l2, (0, 1, 2, 3), t1, (4, 1), (2, 3, 4, 0), optimize=True)
    rdm1.ov += einsum(t2, (0, 1, 2, 3), tmp2, (0, 1, 4, 3), (4, 2), optimize=True) * 0.5
    del tmp2
    rdm1.oo += delta.oo.transpose((1, 0))
    del delta
    rdm1.ov += einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3), optimize=True)
    rdm1.ov += t1
    rdm1.vo = l1.copy()
    rdm1.vv = einsum(t2, (0, 1, 2, 3), l2, (3, 4, 0, 1), (4, 2), optimize=True) * -0.5
    rdm1.vv += einsum(t1, (0, 1), l1, (2, 0), (2, 1), optimize=True)
    rdm1 = np.block([[rdm1.oo, rdm1.ov], [rdm1.vo, rdm1.vv]])

    return rdm1

def make_rdm2_f(l1=None, l2=None, t1=None, t2=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T20:10:43.285135.

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
    tmp3 = einsum(l1, (0, 1), t1, (2, 0), (1, 2), optimize=True)
    rdm2.oovo = einsum(tmp3, (0, 1), t1, (2, 3), (1, 2, 3, 0), optimize=True)
    rdm2.oovo += einsum(tmp3, (0, 1), t1, (2, 3), (2, 1, 3, 0), optimize=True) * -1
    rdm2.ooov = einsum(tmp3, (0, 1), t1, (2, 3), (1, 2, 0, 3), optimize=True) * -1
    rdm2.ooov += einsum(tmp3, (0, 1), t1, (2, 3), (2, 1, 0, 3), optimize=True)
    rdm2.oooo = einsum(tmp3, (0, 1), delta.oo, (2, 3), (3, 1, 2, 0), optimize=True) * -1
    rdm2.oooo += einsum(tmp3, (0, 1), delta.oo, (2, 3), (1, 3, 2, 0), optimize=True)
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp3, (2, 3), (1, 3, 2, 0), optimize=True)
    rdm2.oooo += einsum(tmp3, (0, 1), delta.oo, (2, 3), (1, 3, 0, 2), optimize=True) * -1
    tmp1 = einsum(l2, (0, 1, 2, 3), t1, (4, 1), (2, 3, 4, 0), optimize=True)
    rdm2.vooo = tmp1.transpose((3, 2, 1, 0)).copy()
    rdm2.ovoo = tmp1.transpose((2, 3, 1, 0)).copy() * -1
    tmp4 = einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4), optimize=True)
    rdm2.oovo += einsum(tmp4, (0, 1), t1, (2, 3), (1, 2, 3, 0), optimize=True) * 0.5
    rdm2.oovo += einsum(tmp4, (0, 1), t1, (2, 3), (2, 1, 3, 0), optimize=True) * -0.5
    rdm2.ooov += einsum(tmp4, (0, 1), t1, (2, 3), (1, 2, 0, 3), optimize=True) * -0.5
    rdm2.ooov += einsum(tmp4, (0, 1), t1, (2, 3), (2, 1, 0, 3), optimize=True) * 0.5
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp4, (2, 3), (0, 3, 1, 2), optimize=True) * -0.5
    rdm2.oooo += einsum(tmp4, (0, 1), delta.oo, (2, 3), (1, 3, 2, 0), optimize=True) * 0.5
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp4, (2, 3), (0, 3, 2, 1), optimize=True) * 0.5
    rdm2.oooo += einsum(delta.oo, (0, 1), tmp4, (2, 3), (3, 1, 2, 0), optimize=True) * -0.5
    tmp21 = einsum(t2, (0, 1, 2, 3), l2, (3, 4, 5, 1), (5, 0, 4, 2), optimize=True) * -1
    rdm2.vovo = tmp21.transpose((2, 1, 3, 0)).copy() * -1
    rdm2.voov = tmp21.transpose((2, 1, 0, 3)).copy()
    rdm2.ovvo = tmp21.transpose((1, 2, 3, 0)).copy()
    rdm2.ovov = tmp21.transpose((1, 2, 0, 3)).copy() * -1
    tmp6 = einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3), optimize=True)
    tmp7 = einsum(tmp3, (0, 1), t1, (0, 2), (1, 2), optimize=True)
    tmp9 = einsum(tmp1, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 2, 4, 5), optimize=True) * -1
    rdm2.oovo += tmp9.transpose((2, 1, 3, 0)) * -1
    rdm2.oovo += tmp9.transpose((1, 2, 3, 0))
    rdm2.ooov += tmp9.transpose((2, 1, 0, 3))
    rdm2.ooov += tmp9.transpose((1, 2, 0, 3)) * -1
    tmp5 = einsum(t2, (0, 1, 2, 3), l1, (3, 4), (4, 0, 1, 2), optimize=True)
    rdm2.oovo += tmp5.transpose((2, 1, 3, 0))
    rdm2.ooov += tmp5.transpose((2, 1, 0, 3)) * -1
    tmp0 = einsum(t2, (0, 1, 2, 3), l2, (2, 3, 4, 5), (4, 5, 0, 1), optimize=True)
    rdm2.oooo += tmp0.transpose((3, 2, 1, 0)) * 0.5
    tmp2 = einsum(tmp1, (0, 1, 2, 3), t1, (4, 3), (0, 1, 4, 2), optimize=True)
    rdm2.oooo += tmp2.transpose((2, 3, 1, 0)) * -1
    tmp14 = tmp3.copy() * 2
    tmp14 += tmp4
    tmp29 = einsum(t1, (0, 1), tmp1, (2, 0, 3, 4), (2, 3, 4, 1), optimize=True)
    rdm2.vovo += tmp29.transpose((2, 1, 3, 0))
    rdm2.voov += tmp29.transpose((2, 1, 0, 3)) * -1
    rdm2.ovvo += tmp29.transpose((1, 2, 3, 0)) * -1
    rdm2.ovov += tmp29.transpose((1, 2, 0, 3))
    tmp30 = einsum(t1, (0, 1), l1, (2, 0), (2, 1), optimize=True)
    rdm2.ovvv = einsum(tmp30, (0, 1), t1, (2, 3), (2, 0, 1, 3), optimize=True) * -1
    rdm2.ovvv += einsum(t1, (0, 1), tmp30, (2, 3), (0, 2, 1, 3), optimize=True)
    tmp25 = einsum(t2, (0, 1, 2, 3), l2, (3, 4, 0, 1), (4, 2), optimize=True) * -1
    tmp36 = einsum(tmp21, (0, 1, 2, 3), t1, (0, 4), (1, 2, 4, 3), optimize=True)
    rdm2.vovv = tmp36.transpose((1, 0, 3, 2)).copy() * -1
    rdm2.vovv += tmp36.transpose((1, 0, 2, 3))
    tmp8 = tmp6.copy()
    tmp8 += tmp7 * -1
    del tmp7
    rdm2.ooov += einsum(tmp8, (0, 1), delta.oo, (2, 3), (2, 0, 3, 1), optimize=True)
    rdm2.ooov += einsum(tmp8, (0, 1), delta.oo, (2, 3), (0, 3, 2, 1), optimize=True) * -1
    tmp18 = einsum(t1, (0, 1), tmp9, (0, 2, 3, 4), (2, 3, 1, 4), optimize=True)
    del tmp9
    tmp22 = einsum(t2, (0, 1, 2, 3), tmp21, (1, 4, 3, 5), (0, 4, 2, 5), optimize=True)
    del tmp21
    tmp20 = einsum(tmp5, (0, 1, 2, 3), t1, (0, 4), (1, 2, 4, 3), optimize=True)
    del tmp5
    tmp13 = tmp0.transpose((1, 0, 3, 2)).copy() * -0.5
    tmp13 += tmp2.transpose((0, 1, 3, 2))
    rdm2.oovv = einsum(tmp13, (0, 1, 2, 3), t2, (0, 1, 4, 5), (3, 2, 5, 4), optimize=True) * -0.5
    rdm2.ooov += einsum(tmp13, (0, 1, 2, 3), t1, (0, 4), (3, 2, 1, 4), optimize=True) * -1
    tmp11 = einsum(tmp4, (0, 1), t1, (0, 2), (1, 2), optimize=True)
    tmp10 = einsum(t2, (0, 1, 2, 3), tmp1, (0, 1, 4, 3), (4, 2), optimize=True) * -1
    tmp15 = einsum(tmp14, (0, 1), t1, (0, 2), (1, 2), optimize=True) * 0.5
    del tmp14
    tmp38 = einsum(l2, (0, 1, 2, 3), t1, (3, 4), (2, 0, 1, 4), optimize=True)
    rdm2.vvvv = einsum(tmp38, (0, 1, 2, 3), t1, (0, 4), (1, 2, 4, 3), optimize=True)
    rdm2.vvvo = tmp38.transpose((2, 1, 3, 0)).copy()
    rdm2.vvov = tmp38.transpose((2, 1, 0, 3)).copy() * -1
    del tmp38
    tmp34 = einsum(tmp1, (0, 1, 2, 3), t2, (0, 1, 4, 5), (2, 3, 4, 5), optimize=True)
    del tmp1
    rdm2.vovv += tmp34.transpose((1, 0, 3, 2)) * 0.5
    rdm2.ovvv += tmp34.transpose((0, 1, 3, 2)) * -0.5
    del tmp34
    tmp33 = einsum(t2, (0, 1, 2, 3), l1, (4, 1), (0, 4, 2, 3), optimize=True)
    rdm2.vovv += tmp33.transpose((1, 0, 3, 2))
    rdm2.ovvv += tmp33.transpose((0, 1, 3, 2)) * -1
    del tmp33
    tmp35 = einsum(tmp29, (0, 1, 2, 3), t1, (0, 4), (1, 2, 3, 4), optimize=True) * -1
    del tmp29
    rdm2.vovv += tmp35.transpose((1, 0, 3, 2))
    rdm2.ovvv += tmp35.transpose((0, 1, 3, 2)) * -1
    del tmp35
    tmp32 = tmp30.copy()
    tmp32 += tmp25 * 0.5
    rdm2.vovv += einsum(tmp32, (0, 1), t1, (2, 3), (0, 2, 1, 3), optimize=True)
    rdm2.vovv += einsum(tmp32, (0, 1), t1, (2, 3), (0, 2, 3, 1), optimize=True) * -1
    rdm2.vovo += einsum(tmp32, (0, 1), delta.oo, (2, 3), (0, 3, 1, 2), optimize=True)
    rdm2.ovvo += einsum(tmp32, (0, 1), delta.oo, (2, 3), (3, 0, 1, 2), optimize=True) * -1
    del tmp32
    tmp37 = tmp36.copy()
    del tmp36
    tmp37 += einsum(t1, (0, 1), tmp25, (2, 3), (0, 2, 1, 3), optimize=True) * -0.5
    rdm2.ovvv += tmp37.transpose((0, 1, 3, 2))
    rdm2.ovvv += tmp37 * -1
    del tmp37
    tmp31 = tmp30.copy() * 2
    del tmp30
    tmp31 += tmp25
    rdm2.voov += einsum(delta.oo, (0, 1), tmp31, (2, 3), (2, 1, 0, 3), optimize=True) * -0.5
    rdm2.ovov += einsum(delta.oo, (0, 1), tmp31, (2, 3), (1, 2, 0, 3), optimize=True) * 0.5
    del tmp31
    tmp27 = einsum(tmp4, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4), optimize=True)
    del tmp4
    rdm2.oovv += tmp27.transpose((1, 0, 3, 2)) * -0.5
    rdm2.oovv += tmp27.transpose((0, 1, 3, 2)) * 0.5
    del tmp27
    tmp19 = tmp18.copy()
    del tmp18
    tmp19 += einsum(tmp8, (0, 1), t1, (2, 3), (2, 0, 3, 1), optimize=True) * -1
    del tmp8
    rdm2.oovv += tmp19.transpose((1, 0, 3, 2)) * -1
    rdm2.oovv += tmp19.transpose((1, 0, 2, 3))
    rdm2.oovv += tmp19.transpose((0, 1, 3, 2))
    rdm2.oovv += tmp19 * -1
    del tmp19
    tmp26 = einsum(t2, (0, 1, 2, 3), tmp25, (3, 4), (0, 1, 2, 4), optimize=True)
    del tmp25
    rdm2.oovv += tmp26.transpose((0, 1, 3, 2)) * 0.5
    rdm2.oovv += tmp26 * -0.5
    del tmp26
    tmp23 = tmp20.copy()
    del tmp20
    tmp23 += tmp22
    del tmp22
    rdm2.oovv += tmp23.transpose((0, 1, 3, 2)) * -1
    rdm2.oovv += tmp23
    del tmp23
    tmp28 = einsum(tmp13, (0, 1, 2, 3), t1, (0, 4), (1, 3, 2, 4), optimize=True) * -2
    del tmp13
    rdm2.oovv += einsum(tmp28, (0, 1, 2, 3), t1, (0, 4), (1, 2, 4, 3), optimize=True) * 0.5
    del tmp28
    tmp12 = tmp10.copy()
    tmp12 += tmp11
    del tmp11
    rdm2.oovv += einsum(t1, (0, 1), tmp12, (2, 3), (2, 0, 3, 1), optimize=True) * -0.5
    rdm2.oovv += einsum(tmp12, (0, 1), t1, (2, 3), (0, 2, 3, 1), optimize=True) * 0.5
    rdm2.oovv += einsum(t1, (0, 1), tmp12, (2, 3), (0, 2, 3, 1), optimize=True) * 0.5
    rdm2.oovv += einsum(tmp12, (0, 1), t1, (2, 3), (2, 0, 3, 1), optimize=True) * -0.5
    rdm2.ooov += einsum(delta.oo, (0, 1), tmp12, (2, 3), (0, 2, 1, 3), optimize=True) * -0.5
    rdm2.ooov += einsum(delta.oo, (0, 1), tmp12, (2, 3), (2, 1, 0, 3), optimize=True) * 0.5
    del tmp12
    tmp24 = einsum(tmp3, (0, 1), t2, (2, 0, 3, 4), (1, 2, 3, 4), optimize=True)
    del tmp3
    rdm2.oovv += tmp24.transpose((1, 0, 3, 2))
    rdm2.oovv += tmp24.transpose((0, 1, 3, 2)) * -1
    del tmp24
    tmp17 = tmp0.transpose((1, 0, 3, 2)).copy() * -1
    del tmp0
    tmp17 += tmp2.transpose((0, 1, 3, 2)) * 2
    del tmp2
    rdm2.oovo += einsum(tmp17, (0, 1, 2, 3), t1, (0, 4), (3, 2, 4, 1), optimize=True) * 0.5
    del tmp17
    tmp16 = tmp6.copy() * -1
    del tmp6
    tmp16 += tmp10 * 0.5
    del tmp10
    tmp16 += tmp15
    del tmp15
    rdm2.oovo += einsum(delta.oo, (0, 1), tmp16, (2, 3), (0, 2, 3, 1), optimize=True)
    rdm2.oovo += einsum(delta.oo, (0, 1), tmp16, (2, 3), (2, 1, 3, 0), optimize=True) * -1
    del tmp16
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (3, 1, 2, 0), optimize=True)
    rdm2.oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (1, 2, 3, 0), optimize=True) * -1
    rdm2.ooov += einsum(delta.oo, (0, 1), t1, (2, 3), (2, 1, 0, 3), optimize=True) * -1
    rdm2.ooov += einsum(delta.oo, (0, 1), t1, (2, 3), (1, 2, 0, 3), optimize=True)
    rdm2.oovo += einsum(delta.oo, (0, 1), t1, (2, 3), (2, 1, 3, 0), optimize=True)
    rdm2.oovo += einsum(delta.oo, (0, 1), t1, (2, 3), (1, 2, 3, 0), optimize=True) * -1
    rdm2.ovoo += einsum(delta.oo, (0, 1), l1, (2, 3), (1, 2, 3, 0), optimize=True) * -1
    rdm2.ovoo += einsum(l1, (0, 1), delta.oo, (2, 3), (3, 0, 2, 1), optimize=True)
    rdm2.vooo += einsum(delta.oo, (0, 1), l1, (2, 3), (2, 1, 3, 0), optimize=True)
    rdm2.vooo += einsum(l1, (0, 1), delta.oo, (2, 3), (0, 3, 2, 1), optimize=True) * -1
    del delta
    rdm2.oovv += t2.transpose((1, 0, 3, 2))
    rdm2.oovv += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 3, 1), optimize=True) * -1
    rdm2.oovv += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1), optimize=True)
    rdm2.ovov += einsum(t1, (0, 1), l1, (2, 3), (0, 2, 3, 1), optimize=True) * -1
    rdm2.ovvo += einsum(t1, (0, 1), l1, (2, 3), (0, 2, 1, 3), optimize=True)
    rdm2.voov += einsum(t1, (0, 1), l1, (2, 3), (2, 0, 3, 1), optimize=True)
    rdm2.vovo += einsum(t1, (0, 1), l1, (2, 3), (2, 0, 1, 3), optimize=True) * -1
    rdm2.vvoo = l2.transpose((1, 0, 3, 2)).copy()
    rdm2.vvvv += einsum(t2, (0, 1, 2, 3), l2, (4, 5, 0, 1), (5, 4, 3, 2), optimize=True) * 0.5
    rdm2 = pack_2e(rdm2.oooo, rdm2.ooov, rdm2.oovo, rdm2.ovoo, rdm2.vooo, rdm2.oovv, rdm2.ovov, rdm2.ovvo, rdm2.voov, rdm2.vovo, rdm2.vvoo, rdm2.ovvv, rdm2.vovv, rdm2.vvov, rdm2.vvvo, rdm2.vvvv)
    rdm2 = rdm2.swapaxes(1, 2)

    return rdm2

