"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-08-09T22:10:45.485014
  * python version: 3.10.14 (main, Jul 16 2024, 19:03:10) [GCC 11.4.0]
  * albert version: 0.0.0
  * caller: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/albert/codegen/einsum.py
  * node: fv-az1487-369
  * system: Linux
  * processor: x86_64
  * release: 6.5.0-1025-azure
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, dirsum, Namespace


def energy(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T22:10:46.620368.

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

    tmp3 = einsum(t1, (0, 1), v.xov, (2, 3, 1), (0, 3, 2))
    tmp0 = einsum(t1, (0, 1), v.xov, (2, 0, 1), (2,))
    tmp1 = t2.transpose((0, 1, 3, 2)).copy() * 2
    tmp1 += t2 * -1
    tmp4 = f.ov.copy() * 2
    tmp4 += einsum(tmp3, (0, 1, 2), v.xov, (2, 0, 3), (1, 3)) * -1
    del tmp3
    e_cc = einsum(tmp4, (0, 1), t1, (0, 1), ())
    del tmp4
    tmp2 = einsum(t1, (0, 1), tmp0, (2,), (0, 1, 2))
    del tmp0
    tmp2 += einsum(tmp1, (0, 1, 2, 3), v.xov, (4, 0, 3), (1, 2, 4)) * 0.5
    del tmp1
    e_cc += einsum(v.xov, (0, 1, 2), tmp2, (1, 2, 0), ()) * 2
    del tmp2

    return e_cc

def update_amps(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T22:11:18.171443.

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

    tmp2 = einsum(t1, (0, 1), v.xov, (2, 3, 1), (0, 3, 2))
    tmp0 = einsum(t1, (0, 1), v.xov, (2, 0, 1), (2,))
    tmp29 = einsum(tmp2, (0, 1, 2), v.xov, (2, 3, 4), (0, 3, 1, 4))
    tmp9 = einsum(tmp2, (0, 1, 2), v.xov, (2, 0, 3), (1, 3))
    tmp1 = einsum(v.xov, (0, 1, 2), tmp0, (0,), (1, 2))
    t1new = tmp1.copy() * 2
    tmp31 = einsum(v.xoo, (0, 1, 2), v.xov, (0, 3, 4), (1, 2, 3, 4))
    tmp36 = t2.copy()
    tmp36 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    tmp5 = t2.transpose((0, 1, 3, 2)).copy() * 2
    tmp5 += t2 * -1
    tmp19 = einsum(t2, (0, 1, 2, 3), v.xov, (4, 1, 2), (0, 3, 4))
    t2new = einsum(tmp19, (0, 1, 2), tmp19, (3, 4, 2), (0, 3, 1, 4))
    tmp55 = tmp29.copy() * 2
    tmp55 += tmp29.transpose((0, 2, 1, 3)) * -1
    tmp45 = einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    tmp45 += t2.transpose((0, 1, 3, 2)) * -2
    tmp45 += t2
    tmp18 = einsum(v.xvv, (0, 1, 2), t1, (3, 2), (3, 1, 0))
    t2new += einsum(tmp18, (0, 1, 2), tmp18, (3, 4, 2), (3, 0, 4, 1))
    tmp34 = tmp1.copy() * 2
    tmp34 += tmp9 * -1
    tmp32 = tmp31.transpose((1, 0, 2, 3)).copy() * 2
    tmp32 += tmp31.transpose((1, 2, 0, 3)) * -1
    tmp37 = einsum(v.xov, (0, 1, 2), tmp36, (1, 3, 4, 2), (3, 4, 0))
    del tmp36
    tmp11 = t2.transpose((0, 1, 3, 2)).copy() * 2
    tmp11 += t2 * -1
    tmp13 = einsum(v.xov, (0, 1, 2), tmp5, (1, 3, 4, 2), (3, 4, 0))
    tmp70 = einsum(v.xvv, (0, 1, 2), v.xov, (0, 3, 4), (3, 1, 2, 4))
    tmp54 = einsum(tmp2, (0, 1, 2), tmp19, (3, 4, 2), (0, 3, 1, 4))
    tmp56 = einsum(tmp55, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 2, 4, 5))
    del tmp55
    tmp53 = einsum(tmp31, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 2, 5))
    del tmp31
    tmp6 = einsum(v.xoo, (0, 1, 2), t1, (2, 3), (1, 3, 0))
    tmp46 = einsum(tmp45, (0, 1, 2, 3), v.xov, (4, 0, 3), (1, 2, 4))
    del tmp45
    tmp49 = einsum(v.xvv, (0, 1, 2), tmp0, (0,), (1, 2))
    tmp50 = einsum(v.xov, (0, 1, 2), tmp18, (1, 3, 0), (2, 3))
    tmp59 = einsum(v.xoo, (0, 1, 2), tmp2, (2, 3, 0), (1, 3))
    tmp12 = einsum(v.xoo, (0, 1, 2), tmp0, (0,), (1, 2))
    tmp30 = einsum(tmp29, (0, 1, 2, 3), t2, (4, 2, 3, 5), (0, 4, 1, 5))
    del tmp29
    tmp35 = einsum(tmp34, (0, 1), t2, (2, 3, 1, 4), (0, 2, 3, 4))
    tmp33 = einsum(tmp32, (0, 1, 2, 3), t2, (4, 2, 5, 3), (0, 1, 4, 5))
    del tmp32
    tmp38 = einsum(tmp37, (0, 1, 2), v.xoo, (2, 3, 4), (0, 3, 4, 1))
    del tmp37
    tmp26 = einsum(v.xov, (0, 1, 2), tmp11, (1, 3, 4, 2), (3, 4, 0))
    tmp65 = einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp67 = einsum(v.xoo, (0, 1, 2), tmp2, (3, 4, 0), (3, 1, 2, 4))
    tmp63 = einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    tmp76 = einsum(tmp13, (0, 1, 2), v.xov, (2, 3, 1), (0, 3))
    tmp20 = einsum(t2, (0, 1, 2, 3), v.xov, (4, 1, 3), (0, 2, 4))
    t2new += einsum(tmp20, (0, 1, 2), tmp20, (3, 4, 2), (0, 3, 1, 4)) * 4
    tmp71 = einsum(tmp70, (0, 1, 2, 3), t2, (4, 5, 3, 2), (4, 5, 0, 1))
    del tmp70
    tmp74 = einsum(v.xov, (0, 1, 2), tmp13, (1, 3, 0), (3, 2)) * 0.5
    tmp17 = einsum(v.xov, (0, 1, 2), v.xov, (0, 3, 4), (3, 1, 4, 2))
    t2new += tmp17.transpose((1, 0, 3, 2))
    tmp80 = einsum(tmp2, (0, 1, 2), tmp18, (3, 4, 2), (0, 3, 1, 4))
    tmp24 = einsum(v.xvv, (0, 1, 2), tmp2, (3, 4, 0), (3, 4, 1, 2))
    tmp22 = einsum(v.xvv, (0, 1, 2), v.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp57 = tmp53.copy()
    del tmp53
    tmp57 += tmp54
    del tmp54
    tmp57 += tmp56.transpose((0, 2, 1, 3)) * -1
    del tmp56
    tmp47 = tmp6.copy()
    tmp47 += tmp46
    del tmp46
    tmp51 = tmp49.transpose((1, 0)).copy() * 2
    del tmp49
    tmp51 += tmp50 * -1
    del tmp50
    tmp60 = tmp59.transpose((1, 0)).copy() * -1
    del tmp59
    tmp60 += tmp12.transpose((1, 0)) * 2
    tmp41 = einsum(tmp34, (0, 1), t1, (2, 1), (0, 2))
    del tmp34
    tmp39 = tmp30.transpose((0, 2, 1, 3)).copy()
    del tmp30
    tmp39 += tmp33.transpose((2, 1, 0, 3)) * -1
    del tmp33
    tmp39 += tmp35.transpose((2, 0, 1, 3)) * -1
    del tmp35
    tmp39 += tmp38.transpose((0, 2, 1, 3))
    del tmp38
    tmp27 = v.xov.transpose((1, 2, 0)).copy()
    tmp27 += tmp6 * -1
    tmp27 += tmp26
    del tmp26
    tmp66 = einsum(t1, (0, 1), tmp65, (0, 2, 3, 4), (2, 3, 1, 4))
    del tmp65
    tmp68 = einsum(tmp67, (0, 1, 2, 3), t2, (1, 3, 4, 5), (0, 2, 4, 5))
    del tmp67
    tmp64 = einsum(t2, (0, 1, 2, 3), tmp63, (1, 4), (4, 0, 2, 3))
    del tmp63
    tmp77 = einsum(tmp76, (0, 1), t2, (2, 1, 3, 4), (0, 2, 4, 3)) * 0.5
    del tmp76
    tmp73 = einsum(tmp20, (0, 1, 2), tmp19, (3, 4, 2), (0, 3, 1, 4))
    del tmp19, tmp20
    tmp72 = einsum(t1, (0, 1), tmp71, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp71
    tmp75 = einsum(tmp74, (0, 1), t2, (2, 3, 1, 4), (2, 3, 0, 4))
    del tmp74
    tmp86 = einsum(t1, (0, 1), tmp17, (2, 0, 3, 4), (2, 1, 3, 4))
    del tmp17
    tmp81 = einsum(t1, (0, 1), tmp80, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp80
    tmp79 = einsum(t2, (0, 1, 2, 3), tmp24, (4, 1, 5, 2), (4, 0, 3, 5))
    tmp44 = einsum(t2, (0, 1, 2, 3), tmp22, (4, 1, 5, 3), (0, 4, 2, 5))
    tmp58 = einsum(t1, (0, 1), tmp57, (2, 3, 0, 4), (2, 3, 4, 1))
    del tmp57
    tmp48 = einsum(v.xov, (0, 1, 2), tmp47, (3, 4, 0), (3, 1, 4, 2))
    del tmp47
    tmp52 = einsum(tmp51, (0, 1), t2, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp51
    tmp61 = einsum(t2, (0, 1, 2, 3), tmp60, (1, 4), (4, 0, 3, 2))
    del tmp60
    tmp42 = einsum(tmp41, (0, 1), t2, (2, 0, 3, 4), (1, 2, 4, 3))
    del tmp41
    tmp40 = einsum(t1, (0, 1), tmp39, (2, 0, 3, 4), (2, 3, 4, 1))
    del tmp39
    tmp23 = einsum(t2, (0, 1, 2, 3), tmp22, (4, 1, 5, 2), (0, 4, 3, 5))
    del tmp22
    tmp21 = einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    tmp25 = einsum(t2, (0, 1, 2, 3), tmp24, (4, 1, 5, 3), (4, 0, 2, 5))
    del tmp24
    tmp28 = einsum(tmp27, (0, 1, 2), tmp18, (3, 4, 2), (3, 0, 4, 1))
    del tmp18, tmp27
    tmp3 = v.xoo.transpose((1, 2, 0)).copy()
    tmp3 += tmp2.transpose((1, 0, 2))
    tmp7 = einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 0.5
    tmp7 += t2.transpose((0, 1, 3, 2)) * -1
    tmp7 += t2 * 0.5
    tmp14 = tmp6.copy() * -1
    tmp14 += einsum(t1, (0, 1), tmp0, (2,), (0, 1, 2)) * 2
    tmp14 += tmp13
    del tmp13
    tmp15 = f.ov.copy()
    tmp15 += tmp9 * -1
    tmp69 = tmp64.copy()
    del tmp64
    tmp69 += tmp66
    del tmp66
    tmp69 += tmp68 * -1
    del tmp68
    t2new += tmp69.transpose((1, 0, 2, 3)) * -1
    t2new += tmp69.transpose((0, 1, 3, 2)) * -1
    del tmp69
    tmp84 = einsum(v.xoo, (0, 1, 2), v.xoo, (0, 3, 4), (1, 3, 4, 2))
    tmp84 += einsum(tmp2, (0, 1, 2), tmp2, (3, 4, 2), (1, 3, 4, 0))
    del tmp2
    tmp78 = tmp72.copy()
    del tmp72
    tmp78 += tmp73 * 2
    del tmp73
    tmp78 += tmp75.transpose((1, 0, 3, 2))
    del tmp75
    tmp78 += tmp77.transpose((1, 0, 3, 2))
    del tmp77
    t2new += tmp78.transpose((1, 0, 3, 2)) * -1
    t2new += tmp78 * -1
    del tmp78
    tmp87 = einsum(v.xvv, (0, 1, 2), v.xvv, (0, 3, 4), (1, 3, 4, 2))
    tmp87 += einsum(tmp86, (0, 1, 2, 3), t1, (0, 4), (3, 4, 2, 1))
    del tmp86
    t2new += einsum(tmp87, (0, 1, 2, 3), t2, (4, 5, 2, 0), (4, 5, 1, 3))
    del tmp87
    tmp85 = t2.copy()
    tmp85 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    t2new += einsum(tmp85, (0, 1, 2, 3), tmp84, (0, 4, 1, 5), (4, 5, 3, 2))
    del tmp85, tmp84
    tmp82 = tmp79.copy()
    del tmp79
    tmp82 += tmp81
    del tmp81
    t2new += tmp82.transpose((1, 0, 3, 2)) * -1
    t2new += tmp82 * -1
    del tmp82
    tmp62 = tmp44.copy()
    del tmp44
    tmp62 += tmp48
    del tmp48
    tmp62 += tmp52.transpose((1, 0, 3, 2)) * -1
    del tmp52
    tmp62 += tmp58.transpose((0, 1, 3, 2)) * -1
    del tmp58
    tmp62 += tmp61.transpose((1, 0, 3, 2))
    del tmp61
    t2new += tmp62.transpose((1, 0, 3, 2)) * -1
    t2new += tmp62 * -1
    del tmp62
    tmp43 = tmp21.copy()
    del tmp21
    tmp43 += tmp23 * -1
    del tmp23
    tmp43 += tmp25 * -1
    del tmp25
    tmp43 += tmp28.transpose((0, 1, 3, 2))
    del tmp28
    tmp43 += tmp40.transpose((0, 1, 3, 2))
    del tmp40
    tmp43 += tmp42.transpose((0, 1, 3, 2)) * -1
    del tmp42
    t2new += tmp43.transpose((1, 0, 2, 3))
    t2new += tmp43.transpose((0, 1, 3, 2))
    del tmp43
    tmp83 = einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new += tmp83.transpose((1, 0, 2, 3)) * -1
    t2new += tmp83.transpose((0, 1, 3, 2)) * -1
    del tmp83
    tmp4 = einsum(tmp3, (0, 1, 2), v.xov, (2, 3, 4), (1, 3, 0, 4))
    del tmp3
    t1new += einsum(tmp4, (0, 1, 2, 3), tmp5, (1, 2, 4, 3), (0, 4)) * -1
    del tmp5, tmp4
    tmp10 = f.ov.copy()
    tmp10 += tmp1 * 2
    del tmp1
    tmp10 += tmp9 * -1
    del tmp9
    t1new += einsum(tmp11, (0, 1, 2, 3), tmp10, (0, 3), (1, 2))
    del tmp11, tmp10
    tmp8 = tmp6.copy()
    del tmp6
    tmp8 += einsum(t1, (0, 1), tmp0, (2,), (0, 1, 2)) * -2
    del tmp0
    tmp8 += einsum(v.xov, (0, 1, 2), tmp7, (1, 3, 4, 2), (3, 4, 0)) * 2
    del tmp7
    t1new += einsum(v.xvv, (0, 1, 2), tmp8, (3, 2, 0), (3, 1)) * -1
    del tmp8
    tmp16 = f.oo.copy() * 0.5
    tmp16 += tmp12.transpose((1, 0))
    del tmp12
    tmp16 += einsum(tmp14, (0, 1, 2), v.xov, (2, 3, 1), (3, 0)) * 0.5
    del tmp14
    tmp16 += einsum(tmp15, (0, 1), t1, (2, 1), (0, 2)) * 0.5
    del tmp15
    t1new += einsum(tmp16, (0, 1), t1, (0, 2), (1, 2)) * -2
    del tmp16
    t1new += einsum(t1, (0, 1), f.vv, (2, 1), (0, 2))
    t1new += f.ov

    return {f"t1new": t1new, f"t2new": t2new}

