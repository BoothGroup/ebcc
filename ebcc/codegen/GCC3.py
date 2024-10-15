"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-09-29T17:15:48.888323
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


def energy(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T17:15:49.140059.

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

    tmp0 = np.copy(t2)
    tmp0 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1)) * 2
    e_cc = einsum(t1, (0, 1), f.ov, (0, 1), ())
    e_cc += einsum(tmp0, (0, 1, 2, 3), v.oovv, (0, 1, 2, 3), ()) * 0.25
    del tmp0

    return e_cc

def update_amps(f=None, t1=None, t2=None, t3=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T17:16:20.295753.

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

    tmp0 = einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp60 = einsum(t1, (0, 1), tmp0, (2, 3, 4, 1), (2, 0, 4, 3))
    tmp12 = einsum(t2, (0, 1, 2, 3), f.ov, (4, 3), (4, 0, 1, 2))
    tmp61 = einsum(tmp60, (0, 1, 2, 3), t1, (3, 4), (0, 1, 2, 4))
    del tmp60
    tmp66 = einsum(t2, (0, 1, 2, 3), tmp0, (4, 5, 6, 3), (4, 0, 1, 6, 5, 2))
    tmp79 = einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    tmp18 = einsum(v.ovvv, (0, 1, 2, 3), t1, (4, 3), (4, 0, 1, 2))
    tmp3 = einsum(t1, (0, 1), v.oovv, (2, 0, 3, 1), (2, 3))
    tmp13 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (4, 5, 0, 1))
    tmp14 = einsum(v.oovv, (0, 1, 2, 3), t3, (4, 5, 1, 6, 2, 3), (4, 5, 0, 6))
    tmp62 = np.copy(np.transpose(tmp12, (2, 1, 0, 3))) * -1
    tmp62 += tmp61
    del tmp61
    tmp5 = einsum(t1, (0, 1), f.ov, (2, 1), (2, 0))
    tmp39 = einsum(v.ooov, (0, 1, 2, 3), t1, (4, 3), (4, 0, 1, 2))
    tmp67 = einsum(tmp66, (0, 1, 2, 3, 4, 5), t1, (4, 6), (0, 1, 2, 3, 6, 5))
    del tmp66
    tmp54 = einsum(v.oooo, (0, 1, 2, 3), t1, (3, 4), (0, 1, 2, 4))
    tmp56 = einsum(v.vvvv, (0, 1, 2, 3), t1, (4, 3), (4, 0, 1, 2))
    tmp70 = einsum(t1, (0, 1), f.ov, (0, 2), (2, 1))
    tmp80 = einsum(tmp79, (0, 1, 2, 3, 4, 5), t1, (3, 6), (0, 1, 2, 4, 6, 5))
    del tmp79
    tmp27 = einsum(t1, (0, 1), tmp18, (2, 3, 4, 1), (2, 0, 3, 4)) * -1
    tmp7 = np.copy(t2)
    tmp7 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1)) * 2
    tmp37 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    tmp6 = einsum(t1, (0, 1), v.ooov, (2, 0, 3, 1), (2, 3))
    tmp46 = einsum(tmp3, (0, 1), t1, (2, 1), (2, 0))
    tmp15 = np.copy(np.transpose(tmp12, (0, 2, 1, 3)))
    del tmp12
    tmp15 += np.transpose(tmp13, (2, 1, 0, 3)) * -0.5
    del tmp13
    tmp15 += np.transpose(tmp14, (2, 1, 0, 3)) * 0.5
    del tmp14
    tmp10 = einsum(v.oovv, (0, 1, 2, 3), t2, (0, 1, 4, 3), (4, 2))
    tmp20 = einsum(v.ovvv, (0, 1, 2, 3), t1, (0, 3), (1, 2))
    tmp22 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    tmp24 = einsum(t2, (0, 1, 2, 3), tmp3, (4, 3), (0, 1, 4, 2)) * -1
    tmp29 = einsum(v.ooov, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 2, 5))
    tmp31 = np.copy(t2)
    tmp31 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1
    tmp59 = einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    tmp63 = einsum(t2, (0, 1, 2, 3), tmp62, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3)) * -1
    del tmp62
    tmp73 = np.copy(f.oo)
    tmp73 += tmp5
    tmp77 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 5, 6, 3), (4, 5, 0, 2, 6, 1))
    tmp86 = einsum(t1, (0, 1), tmp39, (2, 0, 3, 4), (2, 3, 4, 1)) * -1
    tmp75 = einsum(v.ovov, (0, 1, 2, 3), t1, (4, 3), (4, 0, 2, 1))
    tmp65 = einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 1, 6), (0, 4, 5, 2, 3, 6))
    tmp68 = einsum(t1, (0, 1), tmp67, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6)) * -1
    del tmp67
    tmp55 = einsum(tmp54, (0, 1, 2, 3), t2, (4, 2, 5, 6), (4, 0, 1, 3, 5, 6)) * -1
    del tmp54
    tmp57 = einsum(t2, (0, 1, 2, 3), tmp56, (4, 5, 6, 3), (4, 0, 1, 2, 5, 6)) * -1
    del tmp56
    tmp84 = einsum(t2, (0, 1, 2, 3), tmp18, (4, 5, 6, 3), (4, 0, 1, 5, 2, 6)) * -1
    tmp71 = np.copy(f.vv)
    tmp71 += tmp70 * -1
    del tmp70
    tmp81 = einsum(tmp80, (0, 1, 2, 3, 4, 5), t1, (2, 6), (0, 1, 3, 4, 6, 5)) * -1
    del tmp80
    tmp82 = einsum(t2, (0, 1, 2, 3), tmp27, (4, 5, 1, 6), (5, 4, 0, 2, 3, 6)) * -1
    tmp40 = einsum(tmp7, (0, 1, 2, 3), tmp39, (4, 0, 1, 5), (4, 5, 2, 3)) * 0.5
    del tmp39
    tmp34 = einsum(t2, (0, 1, 2, 3), tmp5, (1, 4), (4, 0, 2, 3))
    tmp38 = einsum(tmp37, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4))
    del tmp37
    tmp35 = einsum(v.ovoo, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 1), (4, 0, 5, 6))
    tmp36 = einsum(tmp0, (0, 1, 2, 3), t3, (4, 2, 1, 5, 6, 3), (0, 4, 5, 6))
    tmp45 = einsum(tmp6, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4)) * -1
    tmp44 = einsum(v.ovvv, (0, 1, 2, 3), t1, (4, 1), (4, 0, 2, 3))
    tmp47 = einsum(t2, (0, 1, 2, 3), tmp46, (4, 1), (4, 0, 2, 3)) * -1
    del tmp46
    tmp49 = einsum(v.ooov, (0, 1, 2, 3), t1, (2, 4), (0, 1, 4, 3))
    tmp50 = einsum(t2, (0, 1, 2, 3), f.vv, (4, 3), (0, 1, 4, 2))
    tmp9 = einsum(t3, (0, 1, 2, 3, 4, 5), v.ovvv, (2, 6, 4, 5), (0, 1, 3, 6))
    tmp16 = einsum(t1, (0, 1), tmp15, (0, 2, 3, 4), (2, 3, 4, 1))
    del tmp15
    tmp11 = einsum(tmp10, (0, 1), t2, (2, 3, 4, 1), (2, 3, 4, 0))
    del tmp10
    tmp21 = einsum(t2, (0, 1, 2, 3), tmp20, (4, 3), (0, 1, 2, 4)) * -1
    del tmp20
    tmp23 = einsum(tmp22, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp22
    tmp25 = einsum(tmp24, (0, 1, 2, 3), t1, (2, 4), (0, 1, 4, 3))
    del tmp24
    tmp42 = einsum(tmp0, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 2, 5))
    tmp30 = einsum(tmp29, (0, 1, 2, 3), t1, (1, 4), (0, 2, 4, 3))
    del tmp29
    tmp32 = einsum(tmp31, (0, 1, 2, 3), v.ovov, (4, 2, 0, 5), (1, 4, 3, 5))
    del tmp31
    tmp64 = np.copy(np.transpose(tmp59, (0, 1, 2, 3, 5, 4))) * -1
    del tmp59
    tmp64 += np.transpose(tmp63, (0, 1, 2, 3, 5, 4)) * -1
    del tmp63
    tmp74 = einsum(tmp73, (0, 1), t3, (2, 3, 0, 4, 5, 6), (1, 2, 3, 4, 5, 6))
    del tmp73
    tmp78 = einsum(t1, (0, 1), tmp77, (2, 3, 0, 4, 5, 6), (2, 3, 4, 1, 5, 6))
    del tmp77
    tmp87 = einsum(tmp86, (0, 1, 2, 3), t2, (4, 1, 5, 6), (0, 4, 2, 3, 5, 6)) * -1
    del tmp86
    tmp76 = einsum(tmp75, (0, 1, 2, 3), t2, (4, 1, 5, 6), (0, 4, 2, 5, 6, 3))
    del tmp75
    tmp69 = np.copy(np.transpose(tmp65, (0, 2, 1, 4, 3, 5)))
    del tmp65
    tmp69 += np.transpose(tmp68, (0, 2, 1, 4, 3, 5))
    del tmp68
    tmp58 = np.copy(np.transpose(tmp55, (0, 2, 1, 3, 5, 4)))
    del tmp55
    tmp58 += np.transpose(tmp57, (0, 2, 1, 3, 5, 4))
    del tmp57
    tmp85 = einsum(tmp84, (0, 1, 2, 3, 4, 5), t1, (3, 6), (0, 1, 2, 6, 4, 5))
    del tmp84
    tmp72 = einsum(t3, (0, 1, 2, 3, 4, 5), tmp71, (5, 6), (0, 1, 2, 6, 3, 4))
    del tmp71
    tmp83 = np.copy(np.transpose(tmp81, (0, 1, 2, 4, 3, 5))) * -1
    del tmp81
    tmp83 += np.transpose(tmp82, (0, 1, 2, 4, 3, 5)) * -1
    del tmp82
    tmp41 = np.copy(np.transpose(tmp34, (0, 1, 3, 2)))
    del tmp34
    tmp41 += np.transpose(tmp35, (0, 1, 3, 2)) * -0.5
    del tmp35
    tmp41 += np.transpose(tmp36, (0, 1, 3, 2)) * 0.5
    del tmp36
    tmp41 += np.transpose(tmp38, (0, 1, 3, 2)) * -0.5
    del tmp38
    tmp41 += np.transpose(tmp40, (0, 1, 3, 2)) * -1
    del tmp40
    tmp48 = np.copy(np.transpose(tmp44, (0, 1, 3, 2))) * -1
    del tmp44
    tmp48 += np.transpose(tmp45, (0, 1, 3, 2))
    del tmp45
    tmp48 += np.transpose(tmp47, (0, 1, 3, 2)) * -1
    del tmp47
    tmp51 = np.copy(np.transpose(tmp49, (1, 0, 2, 3))) * -1
    del tmp49
    tmp51 += np.transpose(tmp50, (1, 0, 2, 3)) * -1
    del tmp50
    tmp19 = einsum(t2, (0, 1, 2, 3), tmp18, (4, 1, 5, 3), (4, 0, 2, 5)) * -1
    del tmp18
    tmp17 = np.copy(tmp9) * 0.5
    del tmp9
    tmp17 += tmp11 * 0.5
    del tmp11
    tmp17 += np.transpose(tmp16, (1, 0, 3, 2)) * -1
    del tmp16
    tmp26 = np.copy(tmp21)
    del tmp21
    tmp26 += tmp23
    del tmp23
    tmp26 += tmp25 * -1
    del tmp25
    tmp2 = np.copy(t2) * 0.5
    tmp2 += einsum(t1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    tmp43 = einsum(t1, (0, 1), tmp42, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp42
    tmp53 = np.copy(v.oooo)
    tmp53 += einsum(tmp7, (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (4, 5, 1, 0)) * -0.5
    tmp33 = np.copy(tmp30)
    del tmp30
    tmp33 += tmp32 * -1
    del tmp32
    tmp28 = einsum(tmp27, (0, 1, 2, 3), t1, (2, 4), (1, 0, 4, 3)) * -1
    del tmp27
    tmp52 = einsum(t2, (0, 1, 2, 3), f.oo, (4, 1), (4, 0, 2, 3))
    tmp4 = np.copy(f.ov)
    tmp4 += tmp3
    del tmp3
    tmp1 = np.copy(np.transpose(v.ovoo, (0, 2, 3, 1))) * -1
    tmp1 += np.transpose(tmp0, (0, 2, 1, 3)) * -1
    del tmp0
    tmp8 = np.copy(f.oo)
    tmp8 += tmp5
    del tmp5
    tmp8 += tmp6
    del tmp6
    tmp8 += einsum(v.oovv, (0, 1, 2, 3), tmp7, (1, 4, 2, 3), (0, 4)) * -0.5
    t3new = np.copy(tmp58) * -1
    t3new += np.transpose(tmp58, (0, 1, 2, 4, 3, 5))
    t3new += np.transpose(tmp58, (0, 1, 2, 4, 5, 3)) * -1
    t3new += np.transpose(tmp58, (2, 0, 1, 3, 4, 5)) * -1
    t3new += np.transpose(tmp58, (2, 0, 1, 4, 3, 5))
    t3new += np.transpose(tmp58, (2, 0, 1, 4, 5, 3)) * -1
    t3new += np.transpose(tmp58, (2, 1, 0, 3, 4, 5))
    t3new += np.transpose(tmp58, (2, 1, 0, 4, 3, 5)) * -1
    t3new += np.transpose(tmp58, (2, 1, 0, 4, 5, 3))
    del tmp58
    t3new += tmp64 * -1
    t3new += np.transpose(tmp64, (0, 1, 2, 4, 3, 5))
    t3new += np.transpose(tmp64, (0, 1, 2, 4, 5, 3)) * -1
    t3new += np.transpose(tmp64, (0, 2, 1, 3, 4, 5))
    t3new += np.transpose(tmp64, (0, 2, 1, 4, 3, 5)) * -1
    t3new += np.transpose(tmp64, (0, 2, 1, 4, 5, 3))
    t3new += np.transpose(tmp64, (2, 1, 0, 3, 4, 5))
    t3new += np.transpose(tmp64, (2, 1, 0, 4, 3, 5)) * -1
    t3new += np.transpose(tmp64, (2, 1, 0, 4, 5, 3))
    del tmp64
    t3new += tmp69 * -1
    t3new += np.transpose(tmp69, (0, 1, 2, 3, 5, 4))
    t3new += np.transpose(tmp69, (0, 1, 2, 5, 3, 4)) * -1
    t3new += np.transpose(tmp69, (2, 0, 1, 3, 4, 5)) * -1
    t3new += np.transpose(tmp69, (2, 0, 1, 3, 5, 4))
    t3new += np.transpose(tmp69, (2, 0, 1, 5, 3, 4)) * -1
    t3new += np.transpose(tmp69, (2, 1, 0, 3, 4, 5))
    t3new += np.transpose(tmp69, (2, 1, 0, 3, 5, 4)) * -1
    t3new += np.transpose(tmp69, (2, 1, 0, 5, 3, 4))
    del tmp69
    t3new += np.transpose(tmp72, (1, 2, 0, 3, 4, 5))
    t3new += np.transpose(tmp72, (1, 2, 0, 4, 3, 5)) * -1
    t3new += np.transpose(tmp72, (1, 2, 0, 4, 5, 3))
    del tmp72
    t3new += tmp74 * -1
    t3new += np.transpose(tmp74, (2, 0, 1, 3, 4, 5)) * -1
    t3new += np.transpose(tmp74, (2, 1, 0, 3, 4, 5))
    del tmp74
    t3new += np.transpose(tmp76, (0, 1, 2, 4, 3, 5))
    t3new += np.transpose(tmp76, (0, 1, 2, 4, 5, 3)) * -1
    t3new += np.transpose(tmp76, (0, 1, 2, 5, 4, 3))
    t3new += np.transpose(tmp76, (0, 2, 1, 4, 3, 5)) * -1
    t3new += np.transpose(tmp76, (0, 2, 1, 4, 5, 3))
    t3new += np.transpose(tmp76, (0, 2, 1, 5, 4, 3)) * -1
    t3new += np.transpose(tmp76, (1, 0, 2, 4, 3, 5)) * -1
    t3new += np.transpose(tmp76, (1, 0, 2, 4, 5, 3))
    t3new += np.transpose(tmp76, (1, 0, 2, 5, 4, 3)) * -1
    t3new += np.transpose(tmp76, (2, 0, 1, 4, 3, 5))
    t3new += np.transpose(tmp76, (2, 0, 1, 4, 5, 3)) * -1
    t3new += np.transpose(tmp76, (2, 0, 1, 5, 4, 3))
    t3new += np.transpose(tmp76, (1, 2, 0, 4, 3, 5))
    t3new += np.transpose(tmp76, (1, 2, 0, 4, 5, 3)) * -1
    t3new += np.transpose(tmp76, (1, 2, 0, 5, 4, 3))
    t3new += np.transpose(tmp76, (2, 1, 0, 4, 3, 5)) * -1
    t3new += np.transpose(tmp76, (2, 1, 0, 4, 5, 3))
    t3new += np.transpose(tmp76, (2, 1, 0, 5, 4, 3)) * -1
    del tmp76
    t3new += tmp78 * -1
    t3new += np.transpose(tmp78, (0, 1, 2, 3, 5, 4))
    t3new += np.transpose(tmp78, (0, 1, 2, 4, 3, 5))
    t3new += np.transpose(tmp78, (0, 1, 2, 5, 3, 4)) * -1
    t3new += np.transpose(tmp78, (0, 1, 2, 4, 5, 3)) * -1
    t3new += np.transpose(tmp78, (0, 1, 2, 5, 4, 3))
    t3new += np.transpose(tmp78, (0, 2, 1, 3, 4, 5))
    t3new += np.transpose(tmp78, (0, 2, 1, 3, 5, 4)) * -1
    t3new += np.transpose(tmp78, (0, 2, 1, 4, 3, 5)) * -1
    t3new += np.transpose(tmp78, (0, 2, 1, 5, 3, 4))
    t3new += np.transpose(tmp78, (0, 2, 1, 4, 5, 3))
    t3new += np.transpose(tmp78, (0, 2, 1, 5, 4, 3)) * -1
    t3new += np.transpose(tmp78, (2, 1, 0, 3, 4, 5))
    t3new += np.transpose(tmp78, (2, 1, 0, 3, 5, 4)) * -1
    t3new += np.transpose(tmp78, (2, 1, 0, 4, 3, 5)) * -1
    t3new += np.transpose(tmp78, (2, 1, 0, 5, 3, 4))
    t3new += np.transpose(tmp78, (2, 1, 0, 4, 5, 3))
    t3new += np.transpose(tmp78, (2, 1, 0, 5, 4, 3)) * -1
    del tmp78
    t3new += tmp83 * -1
    t3new += np.transpose(tmp83, (0, 1, 2, 3, 5, 4))
    t3new += np.transpose(tmp83, (0, 1, 2, 5, 3, 4)) * -1
    t3new += np.transpose(tmp83, (0, 2, 1, 3, 4, 5))
    t3new += np.transpose(tmp83, (0, 2, 1, 3, 5, 4)) * -1
    t3new += np.transpose(tmp83, (0, 2, 1, 5, 3, 4))
    t3new += np.transpose(tmp83, (2, 1, 0, 3, 4, 5))
    t3new += np.transpose(tmp83, (2, 1, 0, 3, 5, 4)) * -1
    t3new += np.transpose(tmp83, (2, 1, 0, 5, 3, 4))
    del tmp83
    t3new += np.transpose(tmp85, (0, 2, 1, 3, 4, 5))
    t3new += np.transpose(tmp85, (0, 2, 1, 3, 5, 4)) * -1
    t3new += np.transpose(tmp85, (0, 2, 1, 4, 3, 5)) * -1
    t3new += np.transpose(tmp85, (0, 2, 1, 5, 3, 4))
    t3new += np.transpose(tmp85, (0, 2, 1, 4, 5, 3))
    t3new += np.transpose(tmp85, (0, 2, 1, 5, 4, 3)) * -1
    t3new += np.transpose(tmp85, (1, 0, 2, 3, 4, 5))
    t3new += np.transpose(tmp85, (1, 0, 2, 3, 5, 4)) * -1
    t3new += np.transpose(tmp85, (1, 0, 2, 4, 3, 5)) * -1
    t3new += np.transpose(tmp85, (1, 0, 2, 5, 3, 4))
    t3new += np.transpose(tmp85, (1, 0, 2, 4, 5, 3))
    t3new += np.transpose(tmp85, (1, 0, 2, 5, 4, 3)) * -1
    t3new += np.transpose(tmp85, (1, 2, 0, 3, 4, 5)) * -1
    t3new += np.transpose(tmp85, (1, 2, 0, 3, 5, 4))
    t3new += np.transpose(tmp85, (1, 2, 0, 4, 3, 5))
    t3new += np.transpose(tmp85, (1, 2, 0, 5, 3, 4)) * -1
    t3new += np.transpose(tmp85, (1, 2, 0, 4, 5, 3)) * -1
    t3new += np.transpose(tmp85, (1, 2, 0, 5, 4, 3))
    del tmp85
    t3new += np.transpose(tmp87, (0, 1, 2, 3, 5, 4))
    t3new += np.transpose(tmp87, (0, 1, 2, 5, 3, 4)) * -1
    t3new += np.transpose(tmp87, (0, 1, 2, 5, 4, 3))
    t3new += np.transpose(tmp87, (0, 2, 1, 3, 5, 4)) * -1
    t3new += np.transpose(tmp87, (0, 2, 1, 5, 3, 4))
    t3new += np.transpose(tmp87, (0, 2, 1, 5, 4, 3)) * -1
    t3new += np.transpose(tmp87, (1, 0, 2, 3, 5, 4)) * -1
    t3new += np.transpose(tmp87, (1, 0, 2, 5, 3, 4))
    t3new += np.transpose(tmp87, (1, 0, 2, 5, 4, 3)) * -1
    t3new += np.transpose(tmp87, (2, 0, 1, 3, 5, 4))
    t3new += np.transpose(tmp87, (2, 0, 1, 5, 3, 4)) * -1
    t3new += np.transpose(tmp87, (2, 0, 1, 5, 4, 3))
    t3new += np.transpose(tmp87, (1, 2, 0, 3, 5, 4))
    t3new += np.transpose(tmp87, (1, 2, 0, 5, 3, 4)) * -1
    t3new += np.transpose(tmp87, (1, 2, 0, 5, 4, 3))
    t3new += np.transpose(tmp87, (2, 1, 0, 3, 5, 4)) * -1
    t3new += np.transpose(tmp87, (2, 1, 0, 5, 3, 4))
    t3new += np.transpose(tmp87, (2, 1, 0, 5, 4, 3)) * -1
    del tmp87
    t2new = np.copy(v.oovv)
    t2new += tmp17 * -1
    t2new += np.transpose(tmp17, (0, 1, 3, 2))
    del tmp17
    t2new += tmp19
    t2new += np.transpose(tmp19, (0, 1, 3, 2)) * -1
    t2new += np.transpose(tmp19, (1, 0, 2, 3)) * -1
    t2new += np.transpose(tmp19, (1, 0, 3, 2))
    del tmp19
    t2new += tmp26
    t2new += np.transpose(tmp26, (0, 1, 3, 2)) * -1
    del tmp26
    t2new += tmp28 * -1
    t2new += np.transpose(tmp28, (0, 1, 3, 2))
    del tmp28
    t2new += tmp33
    t2new += np.transpose(tmp33, (0, 1, 3, 2)) * -1
    t2new += np.transpose(tmp33, (1, 0, 2, 3)) * -1
    t2new += np.transpose(tmp33, (1, 0, 3, 2))
    del tmp33
    t2new += tmp41 * -1
    t2new += np.transpose(tmp41, (1, 0, 2, 3))
    del tmp41
    t2new += tmp43 * -1
    t2new += np.transpose(tmp43, (0, 1, 3, 2))
    t2new += np.transpose(tmp43, (1, 0, 2, 3))
    t2new += np.transpose(tmp43, (1, 0, 3, 2)) * -1
    del tmp43
    t2new += tmp48 * -1
    t2new += np.transpose(tmp48, (1, 0, 2, 3))
    del tmp48
    t2new += tmp51 * -1
    t2new += np.transpose(tmp51, (0, 1, 3, 2))
    del tmp51
    t2new += np.transpose(tmp52, (0, 1, 3, 2)) * -1
    t2new += np.transpose(tmp52, (1, 0, 3, 2))
    del tmp52
    t2new += einsum(t3, (0, 1, 2, 3, 4, 5), tmp4, (2, 5), (0, 1, 3, 4))
    t2new += einsum(tmp53, (0, 1, 2, 3), tmp7, (0, 1, 4, 5), (2, 3, 5, 4)) * -0.5
    del tmp7, tmp53
    t2new += einsum(tmp2, (0, 1, 2, 3), v.vvvv, (4, 5, 2, 3), (1, 0, 4, 5)) * -1
    t1new = einsum(t1, (0, 1), v.ovov, (2, 1, 0, 3), (2, 3)) * -1
    t1new += f.ov
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t1new += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 0, 1, 5, 2, 3), (4, 5)) * 0.25
    t1new += einsum(tmp1, (0, 1, 2, 3), t2, (1, 2, 4, 3), (0, 4)) * 0.5
    del tmp1
    t1new += einsum(v.ovvv, (0, 1, 2, 3), tmp2, (0, 4, 2, 3), (4, 1))
    del tmp2
    t1new += einsum(tmp4, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    del tmp4
    t1new += einsum(t1, (0, 1), tmp8, (0, 2), (2, 1)) * -1
    del tmp8

    return {f"t1new": t1new, f"t2new": t2new, f"t3new": t3new}

