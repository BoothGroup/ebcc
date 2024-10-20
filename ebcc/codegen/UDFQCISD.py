"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-09-29T16:30:16.752075
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
    Code generated by `albert` 0.0.0 on 2024-09-29T16:30:17.460329.

    Parameters
    ----------
    t2 : Namespace of arrays
        T2 amplitudes.
    v : Namespace of arrays
        Electron repulsion integrals.

    Returns
    -------
    e_cc : float
        Coupled cluster energy.
    """

    tmp1 = einsum(v.baa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (0, 3, 4))
    tmp1 += einsum(v.bbb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (0, 3, 4))
    tmp0 = einsum(v.baa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (0, 3, 4))
    e_cc = einsum(v.baa.xov, (0, 1, 2), tmp0, (0, 1, 2), ())
    del tmp0
    e_cc += einsum(v.bbb.xov, (0, 1, 2), tmp1, (0, 1, 2), ())
    del tmp1

    return e_cc

def update_amps(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T16:30:44.112471.

    Parameters
    ----------
    f : Namespace of arrays
        Fock matrix.
    t1 : Namespace of arrays
        T1 amplitudes.
    t2 : Namespace of arrays
        T2 amplitudes.
    v : Namespace of arrays
        Electron repulsion integrals.

    Returns
    -------
    t1new : Namespace of arrays
        Updated T1 residuals.
    t2new : Namespace of arrays
        Updated T2 residuals.
    """

    t1new = Namespace()
    t2new = Namespace()
    tmp42 = einsum(v.bbb.xov, (0, 1, 2), v.bbb.xov, (0, 3, 4), (1, 3, 2, 4))
    tmp37 = einsum(v.baa.xov, (0, 1, 2), v.baa.xov, (0, 3, 4), (1, 3, 2, 4))
    tmp78 = np.copy(np.transpose(tmp42, (1, 0, 2, 3)))
    tmp78 += np.transpose(tmp42, (1, 0, 3, 2)) * -1
    tmp27 = einsum(v.baa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (0, 3, 4))
    tmp28 = einsum(v.bbb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (0, 3, 4))
    tmp17 = einsum(t2.abab, (0, 1, 2, 3), v.bbb.xov, (4, 1, 3), (4, 0, 2))
    tmp16 = einsum(v.baa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (0, 3, 4))
    tmp38 = np.copy(np.transpose(tmp37, (1, 0, 2, 3))) * -1
    tmp38 += np.transpose(tmp37, (1, 0, 3, 2))
    tmp5 = einsum(v.baa.xov, (0, 1, 2), t1.aa, (1, 2), (0,))
    tmp6 = einsum(v.bbb.xov, (0, 1, 2), t1.bb, (1, 2), (0,))
    tmp71 = einsum(v.bbb.xvv, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp79 = einsum(t2.bbbb, (0, 1, 2, 3), tmp78, (1, 4, 5, 3), (0, 4, 2, 5))
    del tmp78
    tmp22 = einsum(t1.bb, (0, 1), v.bbb.xvv, (2, 3, 1), (2, 0, 3))
    tmp30 = np.copy(tmp27) * 0.5
    tmp30 += tmp28
    tmp55 = np.copy(tmp16) * 2
    tmp55 += tmp17
    tmp1 = einsum(v.baa.xvv, (0, 1, 2), t1.aa, (3, 2), (0, 3, 1))
    tmp39 = einsum(tmp38, (0, 1, 2, 3), t2.aaaa, (4, 0, 5, 2), (4, 1, 5, 3))
    tmp36 = einsum(v.baa.xvv, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp43 = np.copy(np.transpose(tmp42, (1, 0, 2, 3))) * -1
    tmp43 += np.transpose(tmp42, (1, 0, 3, 2))
    tmp20 = np.copy(tmp16) * 2
    tmp20 += tmp17
    tmp7 = np.copy(tmp5)
    tmp7 += tmp6
    tmp3 = einsum(v.bbb.xov, (0, 1, 2), t1.bb, (3, 2), (0, 3, 1))
    tmp10 = einsum(t1.aa, (0, 1), v.baa.xov, (2, 3, 1), (2, 0, 3))
    tmp66 = np.copy(tmp27)
    tmp66 += tmp28 * 2
    tmp80 = np.copy(np.transpose(tmp71, (1, 0, 3, 2)))
    tmp80 += tmp79 * -2
    del tmp79
    tmp86 = einsum(t2.bbbb, (0, 1, 2, 3), tmp42, (4, 5, 3, 2), (0, 1, 5, 4))
    tmp82 = einsum(t2.abab, (0, 1, 2, 3), tmp38, (0, 4, 2, 5), (4, 1, 5, 3))
    del tmp38
    tmp59 = np.copy(v.bbb.xov)
    tmp59 += tmp22
    tmp92 = einsum(v.bbb.xov, (0, 1, 2), tmp30, (0, 1, 3), (2, 3))
    tmp90 = einsum(v.bbb.xoo, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (3, 1, 2, 4))
    tmp23 = einsum(v.bbb.xoo, (0, 1, 2), v.bbb.xov, (0, 3, 4), (1, 2, 3, 4))
    tmp31 = einsum(tmp30, (0, 1, 2), v.bbb.xov, (0, 3, 2), (3, 1)) * 2
    tmp56 = einsum(v.baa.xov, (0, 1, 2), tmp55, (0, 3, 4), (1, 3, 2, 4))
    tmp60 = einsum(v.baa.xov, (0, 1, 2), v.bbb.xov, (0, 3, 4), (1, 3, 2, 4))
    tmp63 = einsum(v.baa.xoo, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 2, 3, 4))
    tmp18 = np.copy(tmp16)
    tmp18 += tmp17 * 0.5
    tmp69 = einsum(v.bbb.xov, (0, 1, 2), tmp55, (0, 3, 4), (3, 1, 4, 2))
    del tmp55
    tmp52 = einsum(v.baa.xoo, (0, 1, 2), v.baa.xoo, (0, 3, 4), (1, 3, 4, 2))
    tmp32 = np.copy(v.baa.xov)
    tmp32 += tmp1
    tmp49 = einsum(t2.aaaa, (0, 1, 2, 3), tmp37, (4, 5, 3, 2), (0, 1, 5, 4))
    tmp2 = einsum(v.baa.xov, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp40 = np.copy(np.transpose(tmp36, (1, 0, 3, 2)))
    tmp40 += tmp39 * -2
    del tmp39
    tmp44 = einsum(tmp43, (0, 1, 2, 3), t2.abab, (4, 0, 5, 3), (4, 1, 5, 2))
    del tmp43
    tmp21 = einsum(tmp20, (0, 1, 2), v.baa.xov, (0, 3, 2), (3, 1)) * 0.5
    tmp8 = einsum(tmp7, (0,), v.bbb.xov, (0, 1, 2), (1, 2))
    tmp4 = einsum(tmp3, (0, 1, 2), v.bbb.xov, (0, 1, 3), (2, 3))
    tmp11 = einsum(v.baa.xov, (0, 1, 2), tmp10, (0, 1, 3), (3, 2))
    tmp12 = einsum(tmp7, (0,), v.baa.xov, (0, 1, 2), (1, 2))
    del tmp7
    tmp93 = einsum(v.bbb.xov, (0, 1, 2), tmp66, (0, 3, 4), (1, 3, 2, 4))
    tmp81 = einsum(t2.bbbb, (0, 1, 2, 3), tmp80, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp80
    tmp77 = einsum(tmp28, (0, 1, 2), tmp27, (0, 3, 4), (3, 1, 4, 2))
    tmp89 = einsum(f.bb.oo, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4))
    tmp88 = einsum(v.bbb.xvv, (0, 1, 2), v.bbb.xvv, (0, 3, 4), (1, 3, 4, 2))
    tmp87 = einsum(tmp86, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 5), (1, 0, 4, 5)) * -1
    del tmp86
    tmp83 = einsum(tmp82, (0, 1, 2, 3), t2.abab, (0, 4, 2, 5), (4, 1, 5, 3))
    del tmp82
    tmp75 = einsum(tmp59, (0, 1, 2), v.bbb.xov, (0, 3, 4), (3, 1, 4, 2))
    tmp94 = np.copy(f.bb.vv) * -0.5
    tmp94 += np.transpose(tmp92, (1, 0))
    tmp85 = einsum(tmp22, (0, 1, 2), v.bbb.xov, (0, 3, 4), (1, 3, 4, 2))
    tmp84 = einsum(t2.bbbb, (0, 1, 2, 3), tmp71, (4, 1, 5, 3), (0, 4, 2, 5))
    tmp91 = einsum(tmp90, (0, 1, 2, 3), t2.bbbb, (1, 3, 4, 5), (2, 0, 4, 5))
    del tmp90
    tmp76 = einsum(t1.bb, (0, 1), tmp23, (2, 0, 3, 4), (2, 3, 1, 4))
    tmp95 = einsum(tmp31, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -1
    tmp73 = einsum(v.bbb.xvv, (0, 1, 2), v.baa.xvv, (0, 3, 4), (3, 4, 1, 2))
    tmp68 = np.copy(np.transpose(tmp36, (1, 0, 3, 2)))
    tmp68 += einsum(t2.aaaa, (0, 1, 2, 3), tmp37, (1, 4, 5, 3), (4, 0, 5, 2)) * 2
    del tmp37
    tmp68 += tmp56 * -1
    tmp72 = einsum(t2.abab, (0, 1, 2, 3), tmp60, (4, 5, 2, 3), (0, 4, 1, 5))
    tmp64 = einsum(t2.abab, (0, 1, 2, 3), tmp63, (4, 0, 5, 1), (4, 5, 2, 3))
    del tmp63
    tmp61 = einsum(v.bbb.xoo, (0, 1, 2), v.baa.xvv, (0, 3, 4), (1, 2, 3, 4))
    tmp61 += einsum(t2.abab, (0, 1, 2, 3), tmp60, (0, 4, 5, 3), (4, 1, 5, 2)) * -1
    del tmp60
    tmp65 = einsum(tmp30, (0, 1, 2), v.bbb.xov, (0, 1, 3), (2, 3)) * 2
    tmp67 = np.copy(f.aa.vv) * -1
    tmp67 += einsum(v.baa.xov, (0, 1, 2), tmp18, (0, 1, 3), (2, 3)) * 2
    tmp62 = einsum(v.bbb.xvv, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp24 = einsum(v.baa.xov, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 3, 4, 2))
    tmp70 = einsum(tmp42, (0, 1, 2, 3), t2.abab, (4, 0, 5, 3), (4, 1, 5, 2)) * -1
    del tmp42
    tmp70 += tmp69
    tmp0 = einsum(v.bbb.xov, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp74 = einsum(v.bbb.xov, (0, 1, 2), tmp30, (0, 3, 2), (3, 1))
    del tmp30
    tmp53 = einsum(tmp52, (0, 1, 2, 3), t2.aaaa, (1, 3, 4, 5), (2, 0, 4, 5))
    del tmp52
    tmp54 = einsum(v.baa.xov, (0, 1, 2), tmp20, (0, 1, 3), (3, 2))
    tmp47 = einsum(tmp1, (0, 1, 2), v.baa.xov, (0, 3, 4), (1, 3, 4, 2))
    tmp33 = einsum(v.baa.xov, (0, 1, 2), tmp32, (0, 3, 4), (1, 3, 2, 4))
    del tmp32
    tmp50 = einsum(tmp49, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 5), (1, 0, 4, 5)) * -1
    del tmp49
    tmp34 = einsum(t1.aa, (0, 1), tmp2, (2, 0, 3, 4), (2, 3, 1, 4))
    tmp48 = einsum(v.baa.xvv, (0, 1, 2), v.baa.xvv, (0, 3, 4), (1, 3, 4, 2))
    tmp41 = einsum(t2.aaaa, (0, 1, 2, 3), tmp40, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp40
    tmp51 = einsum(f.aa.oo, (0, 1), t2.aaaa, (2, 1, 3, 4), (0, 2, 3, 4))
    tmp46 = einsum(t2.aaaa, (0, 1, 2, 3), tmp36, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp36
    tmp35 = einsum(tmp17, (0, 1, 2), tmp16, (0, 3, 4), (3, 1, 4, 2))
    del tmp16, tmp17
    tmp45 = einsum(t2.abab, (0, 1, 2, 3), tmp44, (4, 1, 5, 3), (0, 4, 2, 5)) * -1
    del tmp44
    tmp58 = einsum(tmp21, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2
    tmp57 = np.copy(f.aa.vv) * -0.5
    tmp57 += einsum(v.baa.xov, (0, 1, 2), tmp20, (0, 1, 3), (3, 2)) * 0.5
    del tmp20
    tmp9 = np.copy(f.bb.ov)
    tmp9 += tmp4 * -1
    del tmp4
    tmp9 += tmp8
    del tmp8
    tmp26 = einsum(v.baa.xov, (0, 1, 2), tmp3, (0, 3, 4), (1, 3, 4, 2))
    tmp29 = np.copy(tmp27)
    del tmp27
    tmp29 += tmp28 * 2
    del tmp28
    tmp13 = np.copy(f.aa.ov)
    tmp13 += tmp11 * -1
    del tmp11
    tmp13 += tmp12
    del tmp12
    tmp19 = np.copy(tmp5)
    del tmp5
    tmp19 += tmp6
    del tmp6
    tmp25 = einsum(v.bbb.xov, (0, 1, 2), tmp3, (0, 3, 4), (3, 1, 4, 2))
    del tmp3
    tmp15 = einsum(tmp10, (0, 1, 2), v.baa.xov, (0, 3, 4), (1, 3, 2, 4))
    tmp14 = einsum(tmp10, (0, 1, 2), v.bbb.xov, (0, 3, 4), (1, 2, 3, 4))
    del tmp10
    t2new.bbbb = np.copy(tmp75)
    t2new.bbbb += np.transpose(tmp75, (1, 0, 2, 3)) * -1
    del tmp75
    t2new.bbbb += np.transpose(tmp76, (0, 1, 3, 2))
    t2new.bbbb += tmp77 * 2
    t2new.bbbb += np.transpose(tmp77, (1, 0, 3, 2)) * 2
    t2new.bbbb += np.transpose(tmp81, (1, 0, 3, 2)) * -2
    t2new.bbbb += np.transpose(tmp83, (1, 0, 3, 2))
    t2new.bbbb += np.transpose(tmp76, (1, 0, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp77, (0, 1, 3, 2)) * -2
    t2new.bbbb += np.transpose(tmp77, (1, 0, 2, 3)) * -2
    del tmp77
    t2new.bbbb += np.transpose(tmp81, (0, 1, 3, 2)) * 2
    del tmp81
    t2new.bbbb += np.transpose(tmp83, (1, 0, 2, 3)) * -1
    del tmp83
    t2new.bbbb += np.transpose(tmp84, (1, 0, 2, 3)) * 2
    t2new.bbbb += tmp76 * -1
    t2new.bbbb += tmp84 * -2
    del tmp84
    t2new.bbbb += np.transpose(tmp76, (1, 0, 2, 3))
    del tmp76
    t2new.bbbb += np.transpose(tmp85, (0, 1, 3, 2))
    t2new.bbbb += tmp87 * 2
    del tmp87
    t2new.bbbb += np.transpose(tmp85, (1, 0, 3, 2)) * -1
    del tmp85
    t2new.bbbb += einsum(tmp88, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 3), (4, 5, 0, 2)) * -2
    del tmp88
    t2new.bbbb += np.transpose(tmp89, (1, 0, 3, 2)) * 2
    t2new.bbbb += np.transpose(tmp91, (0, 1, 3, 2)) * -2
    del tmp91
    t2new.bbbb += np.transpose(tmp89, (0, 1, 3, 2)) * -2
    del tmp89
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), tmp92, (3, 4), (0, 1, 2, 4)) * -4
    del tmp92
    t2new.bbbb += tmp93
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), tmp94, (4, 3), (0, 1, 4, 2)) * 4
    del tmp94
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), f.bb.vv, (4, 3), (0, 1, 2, 4)) * 2
    t2new.bbbb += np.transpose(tmp93, (0, 1, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp93, (1, 0, 2, 3)) * -1
    t2new.bbbb += np.transpose(tmp93, (1, 0, 3, 2))
    del tmp93
    t2new.bbbb += np.transpose(tmp95, (0, 1, 3, 2)) * -2
    t2new.bbbb += np.transpose(tmp95, (1, 0, 3, 2)) * 2
    del tmp95
    t2new.abab = einsum(v.baa.xov, (0, 1, 2), tmp59, (0, 3, 4), (1, 3, 2, 4))
    del tmp59
    t2new.abab += einsum(t1.bb, (0, 1), tmp24, (2, 3, 0, 4), (2, 3, 4, 1)) * -1
    t2new.abab += einsum(tmp61, (0, 1, 2, 3), t2.abab, (4, 0, 2, 5), (4, 1, 3, 5)) * -1
    del tmp61
    t2new.abab += einsum(t1.aa, (0, 1), tmp0, (2, 0, 3, 4), (2, 3, 1, 4)) * -1
    t2new.abab += einsum(tmp62, (0, 1, 2, 3), t2.abab, (1, 4, 5, 3), (0, 4, 5, 2)) * -1
    del tmp62
    t2new.abab += einsum(tmp1, (0, 1, 2), v.bbb.xov, (0, 3, 4), (1, 3, 2, 4))
    t2new.abab += tmp64
    del tmp64
    t2new.abab += einsum(f.bb.oo, (0, 1), t2.abab, (2, 1, 3, 4), (2, 0, 3, 4)) * -1
    t2new.abab += einsum(f.aa.oo, (0, 1), t2.abab, (1, 2, 3, 4), (0, 2, 3, 4)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp65, (4, 3), (0, 1, 2, 4)) * -1
    del tmp65
    t2new.abab += einsum(v.baa.xov, (0, 1, 2), tmp66, (0, 3, 4), (1, 3, 2, 4))
    del tmp66
    t2new.abab += einsum(tmp67, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1
    del tmp67
    t2new.abab += einsum(f.bb.vv, (0, 1), t2.abab, (2, 3, 4, 1), (2, 3, 4, 0))
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp68, (0, 4, 2, 5), (4, 1, 5, 3)) * -1
    del tmp68
    t2new.abab += einsum(tmp70, (0, 1, 2, 3), t2.bbbb, (4, 1, 5, 3), (0, 4, 2, 5)) * 2
    del tmp70
    t2new.abab += tmp69
    del tmp69
    t2new.abab += einsum(tmp71, (0, 1, 2, 3), t2.abab, (4, 1, 5, 3), (4, 0, 5, 2)) * -1
    del tmp71
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp72, (4, 0, 5, 1), (4, 5, 2, 3))
    del tmp72
    t2new.abab += einsum(tmp73, (0, 1, 2, 3), t2.abab, (4, 5, 1, 3), (4, 5, 0, 2))
    del tmp73
    t2new.abab += einsum(tmp74, (0, 1), t2.abab, (2, 1, 3, 4), (2, 0, 3, 4)) * -2
    del tmp74
    t2new.abab += einsum(tmp21, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -2
    t2new.aaaa = np.copy(tmp33)
    t2new.aaaa += np.transpose(tmp33, (1, 0, 2, 3)) * -1
    del tmp33
    t2new.aaaa += np.transpose(tmp34, (0, 1, 3, 2))
    t2new.aaaa += tmp35 * 2
    t2new.aaaa += np.transpose(tmp35, (1, 0, 3, 2)) * 2
    t2new.aaaa += np.transpose(tmp41, (1, 0, 3, 2)) * -2
    t2new.aaaa += tmp45
    t2new.aaaa += np.transpose(tmp34, (1, 0, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp35, (0, 1, 3, 2)) * -2
    t2new.aaaa += np.transpose(tmp35, (1, 0, 2, 3)) * -2
    del tmp35
    t2new.aaaa += np.transpose(tmp41, (0, 1, 3, 2)) * 2
    del tmp41
    t2new.aaaa += np.transpose(tmp45, (1, 0, 2, 3)) * -1
    del tmp45
    t2new.aaaa += tmp34 * -1
    t2new.aaaa += np.transpose(tmp46, (1, 0, 2, 3)) * 2
    t2new.aaaa += np.transpose(tmp34, (1, 0, 2, 3))
    del tmp34
    t2new.aaaa += tmp46 * -2
    del tmp46
    t2new.aaaa += np.transpose(tmp47, (0, 1, 3, 2))
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), tmp48, (4, 2, 5, 3), (0, 1, 4, 5)) * -2
    del tmp48
    t2new.aaaa += tmp50 * 2
    del tmp50
    t2new.aaaa += np.transpose(tmp47, (1, 0, 3, 2)) * -1
    del tmp47
    t2new.aaaa += np.transpose(tmp51, (1, 0, 3, 2)) * 2
    t2new.aaaa += np.transpose(tmp53, (0, 1, 3, 2)) * -2
    del tmp53
    t2new.aaaa += np.transpose(tmp51, (0, 1, 3, 2)) * -2
    del tmp51
    t2new.aaaa += einsum(tmp54, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 4, 0)) * -2
    del tmp54
    t2new.aaaa += tmp56
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), tmp57, (4, 3), (0, 1, 4, 2)) * 4
    del tmp57
    t2new.aaaa += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 4, 0)) * 2
    t2new.aaaa += np.transpose(tmp56, (0, 1, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp56, (1, 0, 2, 3)) * -1
    t2new.aaaa += np.transpose(tmp56, (1, 0, 3, 2))
    del tmp56
    t2new.aaaa += np.transpose(tmp58, (0, 1, 3, 2)) * -2
    t2new.aaaa += np.transpose(tmp58, (1, 0, 3, 2)) * 2
    del tmp58
    t1new.bb = einsum(tmp22, (0, 1, 2), v.bbb.xoo, (0, 3, 1), (3, 2)) * -1
    del tmp22
    t1new.bb += einsum(t2.bbbb, (0, 1, 2, 3), tmp23, (4, 0, 1, 3), (4, 2)) * -2
    del tmp23
    t1new.bb += einsum(t2.abab, (0, 1, 2, 3), tmp24, (0, 4, 1, 2), (4, 3)) * -1
    del tmp24
    t1new.bb += einsum(f.bb.oo, (0, 1), t1.bb, (1, 2), (0, 2)) * -1
    t1new.bb += einsum(tmp9, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2
    t1new.bb += einsum(tmp13, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    t1new.bb += einsum(tmp25, (0, 1, 2, 3), t2.bbbb, (2, 1, 4, 3), (0, 4)) * -2
    del tmp25
    t1new.bb += einsum(tmp26, (0, 1, 2, 3), t2.abab, (0, 2, 3, 4), (1, 4)) * -1
    del tmp26
    t1new.bb += einsum(t1.bb, (0, 1), f.bb.vv, (2, 1), (0, 2))
    t1new.bb += einsum(v.bbb.xvv, (0, 1, 2), tmp29, (0, 3, 2), (3, 1))
    del tmp29
    t1new.bb += einsum(v.bbb.xov, (0, 1, 2), tmp19, (0,), (1, 2))
    t1new.bb += einsum(tmp31, (0, 1), t1.bb, (0, 2), (1, 2)) * -1
    del tmp31
    t1new.aa = einsum(t2.abab, (0, 1, 2, 3), tmp0, (4, 0, 1, 3), (4, 2)) * -1
    del tmp0
    t1new.aa += einsum(f.aa.oo, (0, 1), t1.aa, (1, 2), (0, 2)) * -1
    t1new.aa += einsum(tmp1, (0, 1, 2), v.baa.xoo, (0, 3, 1), (3, 2)) * -1
    del tmp1
    t1new.aa += einsum(tmp2, (0, 1, 2, 3), t2.aaaa, (1, 2, 4, 3), (0, 4)) * -2
    del tmp2
    t1new.aa += einsum(tmp9, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    del tmp9
    t1new.aa += einsum(tmp13, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2
    del tmp13
    t1new.aa += einsum(f.aa.vv, (0, 1), t1.aa, (2, 1), (2, 0))
    t1new.aa += einsum(t2.abab, (0, 1, 2, 3), tmp14, (4, 0, 1, 3), (4, 2)) * -1
    del tmp14
    t1new.aa += einsum(t2.aaaa, (0, 1, 2, 3), tmp15, (4, 1, 0, 3), (4, 2)) * -2
    del tmp15
    t1new.aa += einsum(tmp18, (0, 1, 2), v.baa.xvv, (0, 3, 2), (1, 3)) * 2
    del tmp18
    t1new.aa += einsum(v.baa.xov, (0, 1, 2), tmp19, (0,), (1, 2))
    del tmp19
    t1new.aa += einsum(t1.aa, (0, 1), tmp21, (0, 2), (2, 1)) * -2
    del tmp21

    return {f"t1new": t1new, f"t2new": t2new}

