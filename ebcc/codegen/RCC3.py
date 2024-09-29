"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-09-28T14:12:55.101815
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


def energy(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T14:12:55.807503.

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

    tmp0 = np.copy(np.transpose(v.ovov, (0, 2, 3, 1))) * -0.5
    tmp0 += np.transpose(v.ovov, (0, 2, 1, 3))
    tmp1 = np.copy(f.ov)
    tmp1 += einsum(t1, (0, 1), tmp0, (0, 2, 1, 3), (2, 3))
    del tmp0
    e_cc = einsum(t2, (0, 1, 2, 3), v.ovov, (0, 2, 1, 3), ()) * 2
    e_cc += einsum(t2, (0, 1, 2, 3), v.ovov, (0, 3, 1, 2), ()) * -1
    e_cc += einsum(tmp1, (0, 1), t1, (0, 1), ()) * 2
    del tmp1

    return e_cc

def update_amps(f=None, t1=None, t2=None, t3=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-28T14:13:48.495498.

    Parameters
    ----------
    f : array
        Fock matrix.
    t1 : array
        T1 amplitudes.
    t2 : array
        T2 amplitudes.
    t3 : array
        T3 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    t1new : array
        Updated T1 residuals.
    t2new : array
        Updated T2 residuals.
    t3new : array
        Updated T3 residuals.
    """

    tmp2 = einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    tmp4 = np.copy(np.transpose(v.ovov, (0, 2, 3, 1))) * -1
    tmp4 += np.transpose(v.ovov, (0, 2, 1, 3)) * 2
    tmp97 = einsum(tmp2, (0, 1, 2, 3), t1, (4, 3), (0, 4, 2, 1))
    tmp65 = np.copy(tmp2) * 2
    tmp65 += np.transpose(tmp2, (0, 2, 1, 3)) * -1
    tmp5 = einsum(t1, (0, 1), tmp4, (0, 2, 1, 3), (2, 3))
    tmp25 = einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    tmp25 += np.transpose(t2, (0, 1, 3, 2)) * -2
    tmp25 += t2
    tmp94 = einsum(tmp2, (0, 1, 2, 3), t2, (4, 5, 6, 3), (0, 4, 5, 2, 1, 6))
    tmp98 = einsum(tmp97, (0, 1, 2, 3), t1, (2, 4), (1, 0, 3, 4))
    del tmp97
    tmp73 = einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp107 = einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    tmp47 = einsum(v.ooov, (0, 1, 2, 3), t1, (4, 3), (4, 0, 1, 2))
    tmp18 = einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp82 = np.copy(t2)
    tmp82 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    tmp13 = np.copy(np.transpose(v.ovov, (0, 2, 3, 1))) * -0.5
    tmp13 += np.transpose(v.ovov, (0, 2, 1, 3))
    tmp75 = einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    tmp58 = einsum(v.ovov, (0, 1, 2, 3), t1, (2, 3), (0, 1))
    tmp69 = np.copy(v.ooov) * 2
    tmp69 += np.transpose(v.ooov, (0, 2, 1, 3)) * -1
    tmp66 = einsum(tmp65, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 2, 5))
    del tmp65
    tmp64 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 2, 5))
    tmp54 = einsum(tmp2, (0, 1, 2, 3), t1, (2, 4), (0, 1, 4, 3))
    tmp55 = einsum(t2, (0, 1, 2, 3), tmp4, (1, 4, 3, 5), (0, 4, 2, 5))
    del tmp4
    tmp61 = np.copy(v.ovvv) * 2
    tmp61 += np.transpose(v.ovvv, (0, 2, 3, 1)) * -1
    tmp27 = einsum(t2, (0, 1, 2, 3), tmp5, (4, 2), (0, 1, 4, 3))
    tmp26 = einsum(v.ooov, (0, 1, 2, 3), tmp25, (2, 4, 5, 3), (0, 1, 4, 5))
    del tmp25
    tmp22 = einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp24 = einsum(t2, (0, 1, 2, 3), tmp2, (4, 5, 1, 2), (4, 0, 5, 3))
    tmp23 = einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    tmp116 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 5, 6, 3), (4, 5, 0, 2, 6, 1))
    tmp95 = einsum(tmp94, (0, 1, 2, 3, 4, 5), t1, (4, 6), (0, 1, 2, 3, 6, 5))
    del tmp94
    tmp99 = np.copy(np.transpose(tmp73, (2, 1, 0, 3))) * -1
    tmp99 += tmp98
    del tmp98
    tmp108 = einsum(tmp107, (0, 1, 2, 3, 4, 5), t1, (4, 6), (0, 1, 2, 3, 6, 5))
    del tmp107
    tmp106 = einsum(tmp47, (0, 1, 2, 3), t2, (4, 3, 5, 6), (0, 4, 2, 1, 5, 6))
    tmp109 = einsum(tmp18, (0, 1, 2, 3), t2, (4, 5, 6, 2), (0, 4, 5, 1, 6, 3))
    tmp19 = einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    tmp91 = einsum(t1, (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp89 = einsum(v.oooo, (0, 1, 2, 3), t1, (3, 4), (0, 1, 2, 4))
    tmp83 = einsum(tmp82, (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    tmp41 = np.copy(np.transpose(t3, (0, 2, 1, 3, 5, 4)))
    tmp41 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 4, 1, 5)) * -1
    tmp35 = einsum(t3, (0, 1, 2, 3, 4, 5), v.ovov, (6, 3, 2, 5), (0, 1, 6, 4))
    tmp43 = einsum(tmp13, (0, 1, 2, 3), t2, (4, 0, 3, 2), (4, 1)) * 2
    tmp39 = einsum(tmp13, (0, 1, 2, 3), t2, (0, 1, 2, 4), (4, 3))
    tmp37 = einsum(t1, (0, 1), v.ovov, (2, 1, 0, 3), (2, 3))
    tmp76 = np.copy(f.oo)
    tmp76 += tmp75
    tmp59 = np.copy(f.ov)
    tmp59 += tmp58 * 2
    del tmp58
    tmp70 = einsum(t1, (0, 1), tmp69, (2, 3, 0, 1), (2, 3))
    del tmp69
    tmp67 = np.copy(tmp2)
    tmp67 += tmp64 * -1
    del tmp64
    tmp67 += np.transpose(tmp66, (1, 0, 2, 3))
    del tmp66
    tmp56 = np.copy(tmp54)
    del tmp54
    tmp56 += tmp55 * -1
    tmp62 = einsum(t1, (0, 1), tmp61, (0, 1, 2, 3), (2, 3))
    del tmp61
    tmp51 = einsum(t3, (0, 1, 2, 3, 4, 5), v.ovov, (6, 4, 2, 5), (0, 1, 6, 3))
    tmp28 = np.copy(np.transpose(tmp22, (0, 2, 1, 3))) * -1
    tmp28 += np.transpose(tmp23, (0, 2, 1, 3))
    del tmp23
    tmp28 += np.transpose(tmp24, (0, 2, 1, 3))
    del tmp24
    tmp28 += np.transpose(tmp26, (2, 1, 0, 3))
    del tmp26
    tmp28 += np.transpose(tmp27, (1, 2, 0, 3)) * -1
    del tmp27
    tmp30 = einsum(tmp5, (0, 1), t1, (2, 1), (2, 0))
    tmp7 = np.copy(np.transpose(t2, (0, 1, 3, 2))) * -1
    tmp7 += t2 * 2
    tmp115 = einsum(tmp2, (0, 1, 2, 3), t2, (4, 2, 5, 6), (0, 4, 1, 5, 6, 3))
    tmp117 = einsum(tmp116, (0, 1, 2, 3, 4, 5), t1, (3, 6), (0, 1, 2, 6, 4, 5))
    del tmp116
    tmp124 = einsum(t1, (0, 1), tmp18, (2, 3, 1, 4), (0, 2, 3, 4))
    tmp121 = np.copy(f.oo)
    tmp121 += tmp75
    del tmp75
    tmp96 = einsum(t1, (0, 1), tmp95, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del tmp95
    tmp100 = einsum(t2, (0, 1, 2, 3), tmp99, (4, 5, 1, 6), (0, 4, 5, 3, 2, 6))
    del tmp99
    tmp110 = np.copy(np.transpose(tmp106, (0, 1, 3, 2, 4, 5)))
    del tmp106
    tmp110 += np.transpose(tmp108, (0, 1, 3, 2, 4, 5))
    del tmp108
    tmp110 += np.transpose(tmp109, (0, 1, 3, 2, 4, 5)) * -1
    del tmp109
    tmp119 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    tmp126 = einsum(t2, (0, 1, 2, 3), tmp19, (4, 5, 6, 3), (4, 0, 1, 5, 2, 6))
    tmp92 = einsum(t2, (0, 1, 2, 3), tmp91, (4, 5, 3, 6), (4, 0, 1, 2, 6, 5))
    del tmp91
    tmp90 = einsum(t2, (0, 1, 2, 3), tmp89, (4, 1, 5, 6), (0, 5, 4, 6, 2, 3))
    del tmp89
    tmp112 = einsum(t3, (0, 1, 2, 3, 4, 5), f.ov, (6, 5), (6, 0, 2, 1, 3, 4))
    tmp104 = einsum(tmp47, (0, 1, 2, 3), t1, (3, 4), (0, 2, 1, 4))
    tmp85 = np.copy(v.oooo)
    tmp85 += np.transpose(tmp83, (3, 1, 0, 2))
    tmp79 = np.copy(t2)
    tmp79 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    tmp42 = einsum(tmp41, (0, 1, 2, 3, 4, 5), v.ovvv, (0, 3, 6, 5), (1, 2, 6, 4))
    del tmp41
    tmp36 = einsum(t1, (0, 1), tmp35, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp35
    tmp34 = einsum(t3, (0, 1, 2, 3, 4, 5), tmp2, (6, 0, 2, 5), (6, 1, 3, 4)) * -1
    tmp44 = einsum(tmp43, (0, 1), t2, (2, 1, 3, 4), (2, 0, 4, 3))
    del tmp43
    tmp40 = einsum(tmp39, (0, 1), t2, (2, 3, 1, 4), (2, 3, 4, 0)) * 2
    del tmp39
    tmp33 = einsum(v.ooov, (0, 1, 2, 3), t3, (4, 1, 2, 5, 6, 3), (4, 0, 5, 6))
    tmp38 = einsum(tmp37, (0, 1), t3, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    del tmp37
    tmp16 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 3, 5), (4, 0, 5, 1))
    tmp74 = einsum(t1, (0, 1), tmp73, (0, 2, 3, 4), (2, 3, 1, 4))
    del tmp73
    tmp77 = einsum(tmp76, (0, 1), t2, (2, 0, 3, 4), (2, 1, 4, 3))
    del tmp76
    tmp60 = einsum(t3, (0, 1, 2, 3, 4, 5), tmp59, (2, 5), (0, 1, 3, 4))
    del tmp59
    tmp71 = einsum(tmp70, (0, 1), t2, (2, 1, 3, 4), (2, 0, 4, 3))
    del tmp70
    tmp68 = einsum(tmp67, (0, 1, 2, 3), t1, (2, 4), (0, 1, 4, 3))
    del tmp67
    tmp57 = einsum(tmp56, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 5, 2))
    del tmp56
    tmp63 = einsum(tmp62, (0, 1), t2, (2, 3, 1, 4), (2, 3, 4, 0))
    del tmp62
    tmp52 = einsum(t1, (0, 1), tmp51, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp51
    tmp50 = einsum(t3, (0, 1, 2, 3, 4, 5), tmp2, (6, 2, 1, 5), (6, 0, 3, 4))
    tmp48 = einsum(t2, (0, 1, 2, 3), tmp47, (4, 5, 1, 0), (4, 5, 3, 2))
    del tmp47
    tmp46 = einsum(v.ooov, (0, 1, 2, 3), t3, (2, 4, 1, 5, 6, 3), (4, 0, 5, 6)) * -1
    tmp49 = einsum(t3, (0, 1, 2, 3, 4, 5), v.ovvv, (2, 3, 6, 5), (0, 1, 4, 6))
    tmp29 = einsum(t1, (0, 1), tmp28, (2, 0, 3, 4), (2, 3, 1, 4))
    del tmp28
    tmp17 = einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    tmp31 = einsum(tmp30, (0, 1), t2, (2, 1, 3, 4), (2, 0, 4, 3))
    del tmp30
    tmp21 = einsum(tmp18, (0, 1, 2, 3), tmp7, (1, 4, 2, 5), (4, 0, 5, 3))
    tmp20 = einsum(tmp19, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 5, 2))
    tmp9 = np.copy(v.ovvv)
    tmp9 += np.transpose(v.ovvv, (0, 2, 3, 1)) * -0.5
    tmp12 = np.copy(v.ooov)
    tmp12 += np.transpose(v.ooov, (0, 2, 1, 3)) * -0.5
    tmp11 = np.copy(np.transpose(v.ovov, (0, 2, 3, 1))) * 2
    tmp11 += np.transpose(v.ovov, (0, 2, 1, 3)) * -1
    tmp14 = np.copy(f.ov) * 0.5
    tmp14 += einsum(t1, (0, 1), tmp13, (0, 2, 1, 3), (2, 3))
    del tmp13
    tmp103 = einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    tmp123 = einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 6), (0, 4, 5, 2, 3, 6))
    tmp114 = einsum(t2, (0, 1, 2, 3), tmp22, (4, 5, 1, 6), (4, 0, 5, 2, 3, 6))
    del tmp22
    tmp118 = np.copy(tmp115)
    del tmp115
    tmp118 += tmp117
    del tmp117
    tmp125 = einsum(tmp124, (0, 1, 2, 3), t2, (4, 2, 5, 6), (0, 1, 4, 5, 6, 3))
    del tmp124
    tmp122 = einsum(tmp121, (0, 1), t3, (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 6, 5)) * -1
    tmp101 = np.copy(tmp96)
    del tmp96
    tmp101 += np.transpose(tmp100, (2, 1, 0, 5, 4, 3))
    del tmp100
    tmp111 = einsum(tmp110, (0, 1, 2, 3, 4, 5), t1, (2, 6), (0, 1, 3, 6, 4, 5))
    del tmp110
    tmp120 = einsum(tmp119, (0, 1, 2, 3, 4, 5), t1, (2, 6), (0, 1, 3, 6, 4, 5))
    del tmp119
    tmp127 = einsum(tmp126, (0, 1, 2, 3, 4, 5), t1, (3, 6), (0, 1, 2, 6, 4, 5))
    del tmp126
    tmp93 = np.copy(tmp90)
    del tmp90
    tmp93 += tmp92
    del tmp92
    tmp113 = einsum(tmp112, (0, 1, 2, 3, 4, 5), t1, (0, 6), (1, 2, 3, 6, 4, 5))
    del tmp112
    tmp128 = np.copy(f.vv)
    tmp128 += einsum(f.ov, (0, 1), t1, (0, 2), (1, 2)) * -1
    tmp105 = einsum(t2, (0, 1, 2, 3), tmp104, (4, 1, 5, 6), (4, 0, 5, 6, 2, 3))
    del tmp104
    tmp102 = einsum(t3, (0, 1, 2, 3, 4, 5), f.vv, (6, 5), (0, 2, 1, 6, 3, 4))
    tmp87 = np.copy(v.oovv) * -1
    tmp87 += einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 1, 5), (4, 0, 5, 3))
    tmp86 = np.copy(np.transpose(v.ooov, (0, 2, 1, 3))) * -1
    tmp86 += einsum(tmp85, (0, 1, 2, 3), t1, (0, 4), (2, 1, 3, 4))
    del tmp85
    tmp84 = np.copy(v.oooo)
    tmp84 += np.transpose(tmp83, (3, 1, 2, 0))
    del tmp83
    tmp80 = einsum(tmp79, (0, 1, 2, 3), tmp19, (4, 0, 3, 5), (4, 1, 5, 2))
    del tmp19, tmp79
    tmp45 = np.copy(tmp33)
    del tmp33
    tmp45 += tmp34
    del tmp34
    tmp45 += tmp36
    del tmp36
    tmp45 += tmp38
    del tmp38
    tmp45 += np.transpose(tmp40, (1, 0, 2, 3))
    del tmp40
    tmp45 += np.transpose(tmp42, (0, 1, 3, 2)) * -1
    del tmp42
    tmp45 += np.transpose(tmp44, (0, 1, 3, 2))
    del tmp44
    tmp88 = np.copy(np.transpose(v.ovov, (0, 2, 1, 3))) * -1
    tmp88 += tmp16
    tmp78 = np.copy(tmp74)
    del tmp74
    tmp78 += np.transpose(tmp77, (1, 0, 3, 2))
    del tmp77
    tmp72 = np.copy(np.transpose(tmp57, (1, 0, 3, 2)))
    del tmp57
    tmp72 += tmp60
    del tmp60
    tmp72 += np.transpose(tmp63, (1, 0, 2, 3))
    del tmp63
    tmp72 += tmp68 * -1
    del tmp68
    tmp72 += np.transpose(tmp71, (0, 1, 3, 2)) * -1
    del tmp71
    tmp81 = np.copy(np.transpose(v.ovov, (0, 2, 1, 3))) * 2
    tmp81 += v.oovv * -1
    tmp81 += tmp55 * 2
    del tmp55
    tmp53 = np.copy(tmp46)
    del tmp46
    tmp53 += tmp48 * -1
    del tmp48
    tmp53 += tmp49
    del tmp49
    tmp53 += tmp50
    del tmp50
    tmp53 += tmp52
    del tmp52
    tmp32 = np.copy(tmp17)
    del tmp17
    tmp32 += tmp18
    del tmp18
    tmp32 += tmp20 * -1
    del tmp20
    tmp32 += np.transpose(tmp21, (1, 0, 2, 3))
    del tmp21
    tmp32 += tmp29
    del tmp29
    tmp32 += np.transpose(tmp31, (1, 0, 3, 2)) * -1
    del tmp31
    tmp8 = np.copy(np.transpose(v.ovov, (0, 2, 1, 3))) * 2
    tmp8 += v.oovv * -1
    tmp3 = np.copy(v.ooov) * -0.5
    tmp3 += np.transpose(v.ooov, (0, 2, 1, 3))
    tmp3 += tmp2
    tmp3 += np.transpose(tmp2, (0, 2, 1, 3)) * -0.5
    del tmp2
    tmp10 = np.copy(f.vv)
    tmp10 += einsum(t1, (0, 1), tmp9, (0, 1, 2, 3), (3, 2)) * 2
    del tmp9
    tmp1 = np.copy(v.ovvv) * -0.5
    tmp1 += np.transpose(v.ovvv, (0, 2, 1, 3))
    tmp15 = np.copy(f.oo) * 0.5
    tmp15 += einsum(t2, (0, 1, 2, 3), tmp11, (1, 4, 2, 3), (4, 0)) * 0.5
    del tmp11
    tmp15 += einsum(tmp12, (0, 1, 2, 3), t1, (2, 3), (1, 0))
    del tmp12
    tmp15 += einsum(tmp14, (0, 1), t1, (2, 1), (0, 2))
    del tmp14
    tmp6 = np.copy(f.ov)
    tmp6 += tmp5
    del tmp5
    tmp0 = np.copy(np.transpose(t3, (0, 2, 1, 3, 5, 4))) * -3
    tmp0 += np.transpose(t3, (0, 2, 1, 3, 4, 5))
    tmp0 += t3 * -1
    t3new = np.copy(np.transpose(tmp93, (0, 1, 2, 4, 3, 5)))
    t3new += np.transpose(tmp93, (0, 1, 2, 5, 3, 4)) * -1
    t3new += np.transpose(tmp93, (0, 2, 1, 3, 5, 4)) * -1
    t3new += np.transpose(tmp93, (0, 2, 1, 4, 5, 3))
    t3new += np.transpose(tmp93, (1, 0, 2, 3, 4, 5))
    t3new += np.transpose(tmp93, (1, 0, 2, 5, 4, 3)) * -1
    t3new += np.transpose(tmp93, (2, 0, 1, 3, 4, 5)) * -1
    t3new += np.transpose(tmp93, (2, 0, 1, 5, 4, 3))
    t3new += np.transpose(tmp93, (1, 2, 0, 3, 5, 4))
    t3new += np.transpose(tmp93, (1, 2, 0, 4, 5, 3)) * -1
    t3new += np.transpose(tmp93, (2, 1, 0, 4, 3, 5)) * -1
    t3new += np.transpose(tmp93, (2, 1, 0, 5, 3, 4))
    del tmp93
    t3new += np.transpose(tmp101, (0, 1, 2, 3, 5, 4))
    t3new += np.transpose(tmp101, (0, 1, 2, 4, 5, 3)) * -1
    t3new += np.transpose(tmp101, (0, 2, 1, 3, 4, 5))
    t3new += np.transpose(tmp101, (0, 2, 1, 5, 4, 3)) * -1
    t3new += np.transpose(tmp101, (1, 0, 2, 4, 3, 5)) * -1
    t3new += np.transpose(tmp101, (1, 0, 2, 5, 3, 4))
    t3new += np.transpose(tmp101, (2, 0, 1, 4, 3, 5))
    t3new += np.transpose(tmp101, (2, 0, 1, 5, 3, 4)) * -1
    t3new += np.transpose(tmp101, (1, 2, 0, 3, 4, 5)) * -1
    t3new += np.transpose(tmp101, (1, 2, 0, 5, 4, 3))
    t3new += np.transpose(tmp101, (2, 1, 0, 3, 5, 4)) * -1
    t3new += np.transpose(tmp101, (2, 1, 0, 4, 5, 3))
    del tmp101
    t3new += np.transpose(tmp102, (0, 2, 1, 3, 5, 4)) * -1
    t3new += np.transpose(tmp102, (0, 2, 1, 4, 5, 3))
    del tmp102
    t3new += np.transpose(tmp103, (0, 1, 2, 3, 5, 4))
    t3new += np.transpose(tmp103, (0, 1, 2, 4, 5, 3)) * -1
    t3new += np.transpose(tmp103, (0, 2, 1, 3, 4, 5))
    t3new += np.transpose(tmp103, (0, 2, 1, 5, 4, 3)) * -1
    t3new += np.transpose(tmp103, (1, 0, 2, 4, 3, 5)) * -1
    t3new += np.transpose(tmp103, (1, 0, 2, 5, 3, 4))
    t3new += np.transpose(tmp103, (2, 0, 1, 4, 3, 5))
    t3new += np.transpose(tmp103, (2, 0, 1, 5, 3, 4)) * -1
    t3new += np.transpose(tmp103, (1, 2, 0, 3, 4, 5)) * -1
    t3new += np.transpose(tmp103, (1, 2, 0, 5, 4, 3))
    t3new += np.transpose(tmp103, (2, 1, 0, 3, 5, 4)) * -1
    t3new += np.transpose(tmp103, (2, 1, 0, 4, 5, 3))
    del tmp103
    t3new += tmp105
    t3new += np.transpose(tmp105, (0, 1, 2, 5, 4, 3)) * -1
    t3new += np.transpose(tmp105, (0, 2, 1, 3, 5, 4))
    t3new += np.transpose(tmp105, (0, 2, 1, 4, 5, 3)) * -1
    t3new += np.transpose(tmp105, (1, 0, 2, 4, 3, 5))
    t3new += np.transpose(tmp105, (1, 0, 2, 5, 3, 4)) * -1
    t3new += np.transpose(tmp105, (2, 0, 1, 4, 3, 5)) * -1
    t3new += np.transpose(tmp105, (2, 0, 1, 5, 3, 4))
    t3new += np.transpose(tmp105, (1, 2, 0, 3, 5, 4)) * -1
    t3new += np.transpose(tmp105, (1, 2, 0, 4, 5, 3))
    t3new += np.transpose(tmp105, (2, 1, 0, 3, 4, 5)) * -1
    t3new += np.transpose(tmp105, (2, 1, 0, 5, 4, 3))
    del tmp105
    t3new += tmp111 * -1
    t3new += np.transpose(tmp111, (0, 1, 2, 5, 4, 3))
    t3new += np.transpose(tmp111, (0, 2, 1, 4, 3, 5)) * -1
    t3new += np.transpose(tmp111, (0, 2, 1, 5, 3, 4))
    t3new += np.transpose(tmp111, (1, 0, 2, 3, 5, 4)) * -1
    t3new += np.transpose(tmp111, (1, 0, 2, 4, 5, 3))
    t3new += np.transpose(tmp111, (2, 0, 1, 3, 5, 4))
    t3new += np.transpose(tmp111, (2, 0, 1, 4, 5, 3)) * -1
    t3new += np.transpose(tmp111, (1, 2, 0, 4, 3, 5))
    t3new += np.transpose(tmp111, (1, 2, 0, 5, 3, 4)) * -1
    t3new += np.transpose(tmp111, (2, 1, 0, 3, 4, 5))
    t3new += np.transpose(tmp111, (2, 1, 0, 5, 4, 3)) * -1
    del tmp111
    t3new += np.transpose(tmp113, (0, 2, 1, 3, 5, 4))
    t3new += np.transpose(tmp113, (0, 2, 1, 4, 5, 3)) * -1
    del tmp113
    t3new += np.transpose(tmp114, (0, 1, 2, 4, 3, 5))
    t3new += np.transpose(tmp114, (0, 1, 2, 5, 3, 4)) * -1
    t3new += np.transpose(tmp114, (0, 2, 1, 3, 4, 5))
    t3new += np.transpose(tmp114, (0, 2, 1, 5, 4, 3)) * -1
    t3new += np.transpose(tmp114, (1, 0, 2, 3, 5, 4)) * -1
    t3new += np.transpose(tmp114, (1, 0, 2, 4, 5, 3))
    t3new += np.transpose(tmp114, (2, 0, 1, 3, 5, 4))
    t3new += np.transpose(tmp114, (2, 0, 1, 4, 5, 3)) * -1
    t3new += np.transpose(tmp114, (1, 2, 0, 3, 4, 5)) * -1
    t3new += np.transpose(tmp114, (1, 2, 0, 5, 4, 3))
    t3new += np.transpose(tmp114, (2, 1, 0, 4, 3, 5)) * -1
    t3new += np.transpose(tmp114, (2, 1, 0, 5, 3, 4))
    del tmp114
    t3new += np.transpose(tmp118, (0, 1, 2, 4, 3, 5)) * -1
    t3new += np.transpose(tmp118, (0, 1, 2, 5, 3, 4))
    t3new += np.transpose(tmp118, (0, 2, 1, 3, 5, 4))
    t3new += np.transpose(tmp118, (0, 2, 1, 4, 5, 3)) * -1
    t3new += np.transpose(tmp118, (1, 0, 2, 3, 4, 5)) * -1
    t3new += np.transpose(tmp118, (1, 0, 2, 5, 4, 3))
    t3new += np.transpose(tmp118, (2, 0, 1, 3, 4, 5))
    t3new += np.transpose(tmp118, (2, 0, 1, 5, 4, 3)) * -1
    t3new += np.transpose(tmp118, (1, 2, 0, 3, 5, 4)) * -1
    t3new += np.transpose(tmp118, (1, 2, 0, 4, 5, 3))
    t3new += np.transpose(tmp118, (2, 1, 0, 4, 3, 5))
    t3new += np.transpose(tmp118, (2, 1, 0, 5, 3, 4)) * -1
    del tmp118
    t3new += np.transpose(tmp120, (0, 1, 2, 3, 5, 4))
    t3new += np.transpose(tmp120, (0, 1, 2, 4, 5, 3)) * -1
    t3new += np.transpose(tmp120, (0, 2, 1, 4, 3, 5)) * -1
    t3new += np.transpose(tmp120, (0, 2, 1, 5, 3, 4))
    t3new += np.transpose(tmp120, (1, 0, 2, 3, 4, 5))
    t3new += np.transpose(tmp120, (1, 0, 2, 5, 4, 3)) * -1
    t3new += np.transpose(tmp120, (2, 0, 1, 3, 4, 5)) * -1
    t3new += np.transpose(tmp120, (2, 0, 1, 5, 4, 3))
    t3new += np.transpose(tmp120, (1, 2, 0, 4, 3, 5))
    t3new += np.transpose(tmp120, (1, 2, 0, 5, 3, 4)) * -1
    t3new += np.transpose(tmp120, (2, 1, 0, 3, 5, 4)) * -1
    t3new += np.transpose(tmp120, (2, 1, 0, 4, 5, 3))
    del tmp120
    t3new += np.transpose(tmp122, (2, 1, 0, 4, 5, 3))
    t3new += np.transpose(tmp122, (0, 1, 2, 4, 5, 3)) * -1
    del tmp122
    t3new += tmp123 * -1
    t3new += np.transpose(tmp123, (0, 1, 2, 5, 4, 3))
    t3new += np.transpose(tmp123, (0, 2, 1, 3, 5, 4)) * -1
    t3new += np.transpose(tmp123, (0, 2, 1, 4, 5, 3))
    t3new += np.transpose(tmp123, (1, 0, 2, 4, 3, 5)) * -1
    t3new += np.transpose(tmp123, (1, 0, 2, 5, 3, 4))
    t3new += np.transpose(tmp123, (2, 0, 1, 4, 3, 5))
    t3new += np.transpose(tmp123, (2, 0, 1, 5, 3, 4)) * -1
    t3new += np.transpose(tmp123, (1, 2, 0, 3, 5, 4))
    t3new += np.transpose(tmp123, (1, 2, 0, 4, 5, 3)) * -1
    t3new += np.transpose(tmp123, (2, 1, 0, 3, 4, 5))
    t3new += np.transpose(tmp123, (2, 1, 0, 5, 4, 3)) * -1
    del tmp123
    t3new += np.transpose(tmp125, (0, 1, 2, 3, 5, 4))
    t3new += np.transpose(tmp125, (0, 1, 2, 4, 5, 3)) * -1
    t3new += np.transpose(tmp125, (0, 2, 1, 4, 3, 5)) * -1
    t3new += np.transpose(tmp125, (0, 2, 1, 5, 3, 4))
    t3new += np.transpose(tmp125, (1, 0, 2, 3, 4, 5))
    t3new += np.transpose(tmp125, (1, 0, 2, 5, 4, 3)) * -1
    t3new += np.transpose(tmp125, (2, 0, 1, 3, 4, 5)) * -1
    t3new += np.transpose(tmp125, (2, 0, 1, 5, 4, 3))
    t3new += np.transpose(tmp125, (1, 2, 0, 4, 3, 5))
    t3new += np.transpose(tmp125, (1, 2, 0, 5, 3, 4)) * -1
    t3new += np.transpose(tmp125, (2, 1, 0, 3, 5, 4)) * -1
    t3new += np.transpose(tmp125, (2, 1, 0, 4, 5, 3))
    del tmp125
    t3new += tmp127 * -1
    t3new += np.transpose(tmp127, (0, 1, 2, 5, 4, 3))
    t3new += np.transpose(tmp127, (0, 2, 1, 3, 5, 4)) * -1
    t3new += np.transpose(tmp127, (0, 2, 1, 4, 5, 3))
    t3new += np.transpose(tmp127, (1, 0, 2, 4, 3, 5)) * -1
    t3new += np.transpose(tmp127, (1, 0, 2, 5, 3, 4))
    t3new += np.transpose(tmp127, (2, 0, 1, 4, 3, 5))
    t3new += np.transpose(tmp127, (2, 0, 1, 5, 3, 4)) * -1
    t3new += np.transpose(tmp127, (1, 2, 0, 3, 5, 4))
    t3new += np.transpose(tmp127, (1, 2, 0, 4, 5, 3)) * -1
    t3new += np.transpose(tmp127, (2, 1, 0, 3, 4, 5))
    t3new += np.transpose(tmp127, (2, 1, 0, 5, 4, 3)) * -1
    del tmp127
    t3new += einsum(t3, (0, 1, 2, 3, 4, 5), tmp128, (4, 6), (0, 1, 2, 3, 6, 5))
    del tmp128
    t3new += einsum(t3, (0, 1, 2, 3, 4, 5), tmp121, (1, 6), (0, 6, 2, 3, 4, 5)) * -1
    del tmp121
    t2new = np.copy(np.transpose(v.ovov, (0, 2, 1, 3)))
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5)) * -1
    t2new += tmp16 * -1
    del tmp16
    t2new += einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 3, 5), (0, 4, 5, 2)) * -1
    t2new += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5)) * 2
    t2new += einsum(v.ovoo, (0, 1, 2, 3), t1, (3, 4), (0, 2, 1, 4)) * -1
    t2new += np.transpose(tmp32, (0, 1, 3, 2))
    t2new += np.transpose(tmp32, (1, 0, 2, 3))
    del tmp32
    t2new += tmp45 * -1
    t2new += np.transpose(tmp45, (1, 0, 3, 2)) * -1
    del tmp45
    t2new += np.transpose(tmp53, (0, 1, 3, 2)) * -1
    t2new += np.transpose(tmp53, (1, 0, 2, 3)) * -1
    del tmp53
    t2new += tmp72
    t2new += np.transpose(tmp72, (1, 0, 3, 2))
    del tmp72
    t2new += np.transpose(tmp78, (0, 1, 3, 2)) * -1
    t2new += np.transpose(tmp78, (1, 0, 2, 3)) * -1
    del tmp78
    t2new += np.transpose(tmp80, (0, 1, 3, 2)) * -1
    t2new += np.transpose(tmp80, (1, 0, 2, 3)) * -1
    del tmp80
    t2new += einsum(tmp81, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp81
    t2new += einsum(t2, (0, 1, 2, 3), tmp84, (0, 4, 1, 5), (4, 5, 2, 3))
    del tmp84
    t2new += einsum(tmp82, (0, 1, 2, 3), v.vvvv, (4, 3, 5, 2), (1, 0, 4, 5))
    del tmp82
    t2new += einsum(t1, (0, 1), tmp86, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp86
    t2new += einsum(tmp87, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 2, 5))
    del tmp87
    t2new += einsum(t2, (0, 1, 2, 3), tmp88, (4, 1, 5, 2), (4, 0, 5, 3))
    del tmp88
    t1new = np.copy(f.ov)
    t1new += einsum(t3, (0, 1, 2, 3, 4, 5), v.ovov, (2, 4, 1, 5), (0, 3)) * -0.5
    t1new += einsum(tmp0, (0, 1, 2, 3, 4, 5), v.ovov, (0, 3, 2, 5), (1, 4)) * -0.5
    del tmp0
    t1new += einsum(tmp1, (0, 1, 2, 3), t2, (4, 0, 1, 2), (4, 3)) * 2
    del tmp1
    t1new += einsum(tmp3, (0, 1, 2, 3), t2, (1, 2, 3, 4), (0, 4)) * -2
    del tmp3
    t1new += einsum(tmp6, (0, 1), tmp7, (0, 2, 1, 3), (2, 3))
    del tmp7, tmp6
    t1new += einsum(t1, (0, 1), tmp8, (0, 2, 1, 3), (2, 3))
    del tmp8
    t1new += einsum(t1, (0, 1), tmp10, (1, 2), (0, 2))
    del tmp10
    t1new += einsum(tmp15, (0, 1), t1, (0, 2), (1, 2)) * -2
    del tmp15

    return {f"t1new": t1new, f"t2new": t2new, f"t3new": t3new}

