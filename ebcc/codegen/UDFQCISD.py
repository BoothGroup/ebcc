"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-08-09T21:55:35.384445
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


def energy(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T21:55:36.098346.

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

    tmp0 = einsum(v.baa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    e_cc = einsum(v.baa.xov, (0, 1, 2), tmp0, (1, 2, 0), ())
    del tmp0
    tmp1 = einsum(v.baa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    tmp1 += einsum(v.bbb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    e_cc += einsum(tmp1, (0, 1, 2), v.bbb.xov, (2, 0, 1), ())
    del tmp1

    return e_cc

def update_amps(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T21:56:02.936488.

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
    tmp40 = einsum(v.bbb.xov, (0, 1, 2), v.bbb.xov, (0, 3, 4), (3, 1, 4, 2))
    tmp35 = einsum(v.baa.xov, (0, 1, 2), v.baa.xov, (0, 3, 4), (3, 1, 4, 2))
    tmp77 = tmp40.transpose((1, 0, 2, 3)).copy()
    tmp77 += tmp40.transpose((1, 0, 3, 2)) * -1
    tmp26 = einsum(v.baa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    tmp27 = einsum(v.bbb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    tmp16 = einsum(v.baa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    tmp17 = einsum(v.bbb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (3, 4, 0))
    tmp36 = tmp35.transpose((1, 0, 2, 3)).copy() * -1
    tmp36 += tmp35.transpose((1, 0, 3, 2))
    tmp6 = einsum(v.bbb.xov, (0, 1, 2), t1.bb, (1, 2), (0,))
    tmp5 = einsum(v.baa.xov, (0, 1, 2), t1.aa, (1, 2), (0,))
    tmp78 = einsum(t2.bbbb, (0, 1, 2, 3), tmp77, (1, 4, 5, 3), (0, 4, 2, 5))
    del tmp77
    tmp71 = einsum(v.bbb.xvv, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (3, 4, 1, 2))
    t2new.abab = einsum(tmp71, (0, 1, 2, 3), t2.abab, (4, 1, 5, 3), (4, 0, 5, 2)) * -1
    tmp22 = einsum(v.bbb.xvv, (0, 1, 2), t1.bb, (3, 2), (3, 1, 0))
    t1new.bb = einsum(tmp22, (0, 1, 2), v.bbb.xoo, (2, 3, 0), (3, 1)) * -1
    tmp28 = tmp26.copy() * 0.5
    tmp28 += tmp27
    t1new.bb += einsum(tmp28, (0, 1, 2), v.bbb.xvv, (2, 3, 1), (0, 3)) * 2
    tmp53 = tmp16.copy() * 2
    tmp53 += tmp17
    tmp18 = tmp16.copy() * 2
    tmp18 += tmp17
    t1new.aa = einsum(tmp18, (0, 1, 2), v.baa.xvv, (2, 3, 1), (0, 3))
    tmp34 = einsum(v.baa.xvv, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp37 = einsum(t2.aaaa, (0, 1, 2, 3), tmp36, (1, 4, 3, 5), (0, 4, 2, 5))
    tmp41 = tmp40.transpose((1, 0, 2, 3)).copy() * -1
    tmp41 += tmp40.transpose((1, 0, 3, 2))
    tmp1 = einsum(v.baa.xvv, (0, 1, 2), t1.aa, (3, 2), (3, 1, 0))
    t2new.abab += einsum(v.bbb.xov, (0, 1, 2), tmp1, (3, 4, 0), (3, 1, 4, 2))
    t1new.aa += einsum(tmp1, (0, 1, 2), v.baa.xoo, (2, 3, 0), (3, 1)) * -1
    tmp7 = tmp5.copy()
    tmp7 += tmp6
    tmp3 = einsum(v.baa.xov, (0, 1, 2), t1.aa, (3, 2), (3, 1, 0))
    tmp10 = einsum(v.bbb.xov, (0, 1, 2), t1.bb, (3, 2), (3, 1, 0))
    tmp81 = einsum(t2.abab, (0, 1, 2, 3), tmp36, (0, 4, 2, 5), (4, 1, 5, 3))
    del tmp36
    tmp85 = einsum(t2.bbbb, (0, 1, 2, 3), tmp40, (4, 5, 3, 2), (0, 1, 5, 4))
    tmp79 = tmp71.transpose((1, 0, 3, 2)).copy()
    tmp79 += tmp78 * -2
    del tmp78
    tmp58 = v.bbb.xov.transpose((1, 2, 0)).copy()
    tmp58 += tmp22
    t2new.abab += einsum(v.baa.xov, (0, 1, 2), tmp58, (3, 4, 0), (1, 3, 2, 4))
    tmp21 = einsum(v.bbb.xov, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (3, 4, 1, 2))
    t1new.bb += einsum(tmp21, (0, 1, 2, 3), t2.bbbb, (1, 2, 4, 3), (0, 4)) * -2
    tmp65 = tmp26.copy()
    tmp65 += tmp27 * 2
    t2new.abab += einsum(tmp65, (0, 1, 2), v.baa.xov, (2, 3, 4), (3, 0, 4, 1))
    tmp94 = einsum(tmp28, (0, 1, 2), v.bbb.xov, (2, 3, 1), (3, 0)) * 2
    tmp88 = einsum(v.bbb.xoo, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 3, 4, 2))
    tmp91 = einsum(tmp28, (0, 1, 2), v.bbb.xov, (2, 0, 3), (3, 1))
    t2new.bbbb = einsum(t2.bbbb, (0, 1, 2, 3), tmp91, (3, 4), (0, 1, 2, 4)) * -4
    tmp66 = tmp16.copy()
    tmp66 += tmp17 * 0.5
    tmp59 = einsum(v.bbb.xov, (0, 1, 2), v.baa.xov, (0, 3, 4), (3, 1, 4, 2))
    tmp69 = einsum(v.bbb.xov, (0, 1, 2), tmp53, (3, 4, 0), (3, 1, 4, 2))
    t2new.abab += tmp69
    tmp54 = einsum(tmp53, (0, 1, 2), v.baa.xov, (2, 3, 4), (3, 0, 4, 1))
    del tmp53
    t2new.aaaa = tmp54.transpose((1, 0, 3, 2)).copy()
    t2new.aaaa += tmp54.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp54.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp54
    tmp62 = einsum(v.baa.xoo, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 2, 3, 4))
    tmp49 = einsum(v.baa.xoo, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 1, 2, 4))
    tmp56 = einsum(tmp18, (0, 1, 2), v.baa.xov, (2, 3, 1), (3, 0)) * 0.5
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp56, (0, 4), (4, 1, 2, 3)) * -2
    tmp47 = einsum(t2.aaaa, (0, 1, 2, 3), tmp35, (4, 5, 3, 2), (0, 1, 5, 4))
    tmp2 = einsum(v.baa.xoo, (0, 1, 2), v.baa.xov, (0, 3, 4), (1, 2, 3, 4))
    t1new.aa += einsum(t2.aaaa, (0, 1, 2, 3), tmp2, (4, 0, 1, 3), (4, 2)) * -2
    tmp38 = tmp34.transpose((1, 0, 3, 2)).copy()
    tmp38 += tmp37 * -2
    del tmp37
    tmp42 = einsum(tmp41, (0, 1, 2, 3), t2.abab, (4, 0, 5, 3), (4, 1, 5, 2))
    del tmp41
    tmp30 = v.baa.xov.transpose((1, 2, 0)).copy()
    tmp30 += tmp1
    tmp8 = einsum(tmp7, (0,), v.baa.xov, (0, 1, 2), (1, 2))
    tmp4 = einsum(v.baa.xov, (0, 1, 2), tmp3, (1, 3, 0), (3, 2))
    tmp12 = einsum(tmp7, (0,), v.bbb.xov, (0, 1, 2), (1, 2))
    del tmp7
    tmp11 = einsum(v.bbb.xov, (0, 1, 2), tmp10, (1, 3, 0), (3, 2))
    tmp82 = einsum(tmp81, (0, 1, 2, 3), t2.abab, (0, 4, 2, 5), (4, 1, 5, 3))
    del tmp81
    t2new.bbbb += tmp82.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp82
    del tmp82
    tmp87 = einsum(v.bbb.xvv, (0, 1, 2), v.bbb.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), tmp87, (4, 2, 5, 3), (0, 1, 4, 5)) * -2
    del tmp87
    tmp86 = einsum(tmp85, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 5), (1, 0, 4, 5)) * -1
    del tmp85
    t2new.bbbb += tmp86.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp86.transpose((0, 1, 3, 2)) * -1
    del tmp86
    tmp80 = einsum(t2.bbbb, (0, 1, 2, 3), tmp79, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp79
    t2new.bbbb += tmp80.transpose((0, 1, 3, 2)) * 2
    t2new.bbbb += tmp80.transpose((1, 0, 3, 2)) * -2
    del tmp80
    tmp74 = einsum(tmp58, (0, 1, 2), v.bbb.xov, (2, 3, 4), (3, 0, 4, 1))
    del tmp58
    t2new.bbbb += tmp74.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp74
    del tmp74
    tmp76 = einsum(tmp26, (0, 1, 2), tmp27, (3, 4, 2), (0, 3, 1, 4))
    del tmp26, tmp27
    t2new.bbbb += tmp76.transpose((1, 0, 2, 3)) * -2
    t2new.bbbb += tmp76.transpose((0, 1, 3, 2)) * -2
    t2new.bbbb += tmp76.transpose((1, 0, 3, 2)) * 2
    t2new.bbbb += tmp76 * 2
    del tmp76
    tmp75 = einsum(tmp21, (0, 1, 2, 3), t1.bb, (1, 4), (0, 2, 4, 3))
    del tmp21
    t2new.bbbb += tmp75.transpose((1, 0, 2, 3))
    t2new.bbbb += tmp75 * -1
    t2new.bbbb += tmp75.transpose((1, 0, 3, 2)) * -1
    t2new.bbbb += tmp75.transpose((0, 1, 3, 2))
    del tmp75
    tmp83 = einsum(tmp71, (0, 1, 2, 3), t2.bbbb, (4, 1, 5, 3), (4, 0, 5, 2))
    del tmp71
    t2new.bbbb += tmp83 * -2
    t2new.bbbb += tmp83.transpose((1, 0, 2, 3)) * 2
    del tmp83
    tmp92 = einsum(v.bbb.xov, (0, 1, 2), tmp65, (3, 4, 0), (1, 3, 2, 4))
    del tmp65
    t2new.bbbb += tmp92.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp92.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp92.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp92
    del tmp92
    tmp95 = einsum(tmp94, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -1
    del tmp94
    t2new.bbbb += tmp95.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp95.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp95.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp95.transpose((0, 1, 3, 2)) * -1
    del tmp95
    tmp89 = einsum(tmp88, (0, 1, 2, 3), t2.bbbb, (1, 3, 4, 5), (2, 0, 4, 5))
    del tmp88
    t2new.bbbb += tmp89.transpose((0, 1, 3, 2)) * -0.5
    t2new.bbbb += tmp89.transpose((0, 1, 3, 2)) * -0.5
    t2new.bbbb += tmp89.transpose((0, 1, 3, 2)) * -0.5
    t2new.bbbb += tmp89.transpose((0, 1, 3, 2)) * -0.5
    del tmp89
    tmp90 = einsum(f.bb.oo, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new.bbbb += tmp90.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp90.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp90.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp90.transpose((1, 0, 3, 2))
    del tmp90
    tmp84 = einsum(v.bbb.xov, (0, 1, 2), tmp22, (3, 4, 0), (3, 1, 2, 4))
    del tmp22
    t2new.bbbb += tmp84.transpose((1, 0, 3, 2)) * -1
    t2new.bbbb += tmp84.transpose((0, 1, 3, 2))
    del tmp84
    tmp93 = f.bb.vv.copy() * -0.5
    tmp93 += tmp91.transpose((1, 0))
    del tmp91
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), tmp93, (4, 3), (0, 1, 4, 2)) * 4
    del tmp93
    tmp67 = f.aa.vv.copy() * -1
    tmp67 += einsum(v.baa.xov, (0, 1, 2), tmp66, (1, 3, 0), (2, 3)) * 2
    del tmp66
    t2new.abab += einsum(tmp67, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1
    del tmp67
    tmp61 = einsum(v.bbb.xvv, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp61, (4, 0, 5, 3), (4, 1, 2, 5)) * -1
    del tmp61
    tmp60 = einsum(v.baa.xvv, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp60 += einsum(t2.abab, (0, 1, 2, 3), tmp59, (0, 4, 5, 3), (4, 1, 5, 2)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp60, (1, 4, 2, 5), (0, 4, 5, 3)) * -1
    del tmp60
    tmp70 = einsum(tmp40, (0, 1, 2, 3), t2.abab, (4, 0, 5, 3), (4, 1, 5, 2)) * -1
    del tmp40
    tmp70 += tmp69
    del tmp69
    t2new.abab += einsum(tmp70, (0, 1, 2, 3), t2.bbbb, (4, 1, 5, 3), (0, 4, 2, 5)) * 2
    del tmp70
    tmp73 = einsum(t2.abab, (0, 1, 2, 3), tmp59, (4, 5, 2, 3), (0, 4, 1, 5))
    del tmp59
    t2new.abab += einsum(tmp73, (0, 1, 2, 3), t2.abab, (1, 3, 4, 5), (0, 2, 4, 5))
    del tmp73
    tmp68 = tmp34.transpose((1, 0, 3, 2)).copy()
    tmp68 += einsum(tmp35, (0, 1, 2, 3), t2.aaaa, (4, 0, 5, 3), (1, 4, 2, 5)) * 2
    del tmp35
    tmp68 += tmp54 * -1
    del tmp54
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp68, (0, 4, 2, 5), (4, 1, 5, 3)) * -1
    del tmp68
    tmp64 = einsum(tmp28, (0, 1, 2), v.bbb.xov, (2, 0, 3), (1, 3)) * 2
    t2new.abab += einsum(tmp64, (0, 1), t2.abab, (2, 3, 4, 1), (2, 3, 4, 0)) * -1
    del tmp64
    tmp23 = einsum(v.baa.xov, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 3, 4, 2))
    t2new.abab += einsum(tmp23, (0, 1, 2, 3), t1.bb, (2, 4), (0, 1, 3, 4)) * -1
    t1new.bb += einsum(tmp23, (0, 1, 2, 3), t2.abab, (0, 2, 3, 4), (1, 4)) * -1
    del tmp23
    tmp72 = einsum(v.bbb.xvv, (0, 1, 2), v.baa.xvv, (0, 3, 4), (3, 4, 1, 2))
    t2new.abab += einsum(tmp72, (0, 1, 2, 3), t2.abab, (4, 5, 1, 3), (4, 5, 0, 2))
    del tmp72
    tmp63 = einsum(tmp62, (0, 1, 2, 3), t2.abab, (1, 3, 4, 5), (0, 2, 4, 5))
    del tmp62
    t2new.abab += tmp63 * 0.5
    t2new.abab += tmp63 * 0.5
    del tmp63
    tmp0 = einsum(v.bbb.xov, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    t2new.abab += einsum(tmp0, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3)) * -1
    t1new.aa += einsum(t2.abab, (0, 1, 2, 3), tmp0, (4, 0, 1, 3), (4, 2)) * -1
    del tmp0
    tmp29 = einsum(tmp28, (0, 1, 2), v.bbb.xov, (2, 3, 1), (3, 0))
    del tmp28
    t2new.abab += einsum(tmp29, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -2
    t1new.bb += einsum(t1.bb, (0, 1), tmp29, (0, 2), (2, 1)) * -2
    del tmp29
    tmp50 = einsum(tmp49, (0, 1, 2, 3), t2.aaaa, (1, 3, 4, 5), (2, 0, 4, 5))
    del tmp49
    t2new.aaaa += tmp50.transpose((0, 1, 3, 2)) * -0.5
    t2new.aaaa += tmp50.transpose((0, 1, 3, 2)) * -0.5
    t2new.aaaa += tmp50.transpose((0, 1, 3, 2)) * -0.5
    t2new.aaaa += tmp50.transpose((0, 1, 3, 2)) * -0.5
    del tmp50
    tmp45 = einsum(v.baa.xov, (0, 1, 2), tmp1, (3, 4, 0), (3, 1, 2, 4))
    del tmp1
    t2new.aaaa += tmp45.transpose((1, 0, 3, 2)) * -1
    t2new.aaaa += tmp45.transpose((0, 1, 3, 2))
    del tmp45
    tmp44 = einsum(t2.aaaa, (0, 1, 2, 3), tmp34, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp34
    t2new.aaaa += tmp44 * -2
    t2new.aaaa += tmp44.transpose((1, 0, 2, 3)) * 2
    del tmp44
    tmp57 = einsum(t2.aaaa, (0, 1, 2, 3), tmp56, (1, 4), (0, 4, 2, 3)) * -2
    del tmp56
    t2new.aaaa += tmp57.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp57.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp57.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp57.transpose((0, 1, 3, 2)) * -1
    del tmp57
    tmp48 = einsum(tmp47, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 5), (1, 0, 4, 5)) * -1
    del tmp47
    t2new.aaaa += tmp48.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp48.transpose((0, 1, 3, 2)) * -1
    del tmp48
    tmp46 = einsum(v.baa.xvv, (0, 1, 2), v.baa.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), tmp46, (4, 2, 5, 3), (0, 1, 4, 5)) * -2
    del tmp46
    tmp32 = einsum(tmp2, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3))
    del tmp2
    t2new.aaaa += tmp32.transpose((1, 0, 2, 3))
    t2new.aaaa += tmp32 * -1
    t2new.aaaa += tmp32.transpose((1, 0, 3, 2)) * -1
    t2new.aaaa += tmp32.transpose((0, 1, 3, 2))
    del tmp32
    tmp52 = einsum(tmp18, (0, 1, 2), v.baa.xov, (2, 0, 3), (1, 3)) * 0.5
    t2new.aaaa += einsum(tmp52, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 4, 0)) * -4
    del tmp52
    tmp39 = einsum(t2.aaaa, (0, 1, 2, 3), tmp38, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp38
    t2new.aaaa += tmp39.transpose((0, 1, 3, 2)) * 2
    t2new.aaaa += tmp39.transpose((1, 0, 3, 2)) * -2
    del tmp39
    tmp43 = einsum(tmp42, (0, 1, 2, 3), t2.abab, (4, 1, 5, 3), (4, 0, 5, 2)) * -1
    del tmp42
    t2new.aaaa += tmp43.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp43
    del tmp43
    tmp51 = einsum(f.aa.oo, (0, 1), t2.aaaa, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new.aaaa += tmp51.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp51.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp51.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp51.transpose((1, 0, 3, 2))
    del tmp51
    tmp31 = einsum(tmp30, (0, 1, 2), v.baa.xov, (2, 3, 4), (3, 0, 4, 1))
    del tmp30
    t2new.aaaa += tmp31.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp31
    del tmp31
    tmp33 = einsum(tmp17, (0, 1, 2), tmp16, (3, 4, 2), (3, 0, 4, 1))
    del tmp16, tmp17
    t2new.aaaa += tmp33.transpose((1, 0, 2, 3)) * -2
    t2new.aaaa += tmp33.transpose((0, 1, 3, 2)) * -2
    t2new.aaaa += tmp33.transpose((1, 0, 3, 2)) * 2
    t2new.aaaa += tmp33 * 2
    del tmp33
    tmp55 = f.aa.vv.copy() * -1
    tmp55 += einsum(tmp18, (0, 1, 2), v.baa.xov, (2, 0, 3), (1, 3))
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), tmp55, (4, 3), (0, 1, 4, 2)) * 2
    del tmp55
    tmp9 = f.aa.ov.copy()
    tmp9 += tmp4 * -1
    del tmp4
    tmp9 += tmp8
    del tmp8
    t1new.bb += einsum(t2.abab, (0, 1, 2, 3), tmp9, (0, 2), (1, 3))
    t1new.aa += einsum(tmp9, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2
    del tmp9
    tmp19 = tmp5.copy()
    del tmp5
    tmp19 += tmp6
    del tmp6
    t1new.bb += einsum(v.bbb.xov, (0, 1, 2), tmp19, (0,), (1, 2))
    t1new.aa += einsum(v.baa.xov, (0, 1, 2), tmp19, (0,), (1, 2))
    del tmp19
    tmp13 = f.bb.ov.copy()
    tmp13 += tmp11 * -1
    del tmp11
    tmp13 += tmp12
    del tmp12
    t1new.bb += einsum(tmp13, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2
    t1new.aa += einsum(tmp13, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    del tmp13
    tmp25 = einsum(v.bbb.xov, (0, 1, 2), tmp10, (3, 4, 0), (3, 1, 4, 2))
    t1new.bb += einsum(t2.bbbb, (0, 1, 2, 3), tmp25, (4, 1, 0, 3), (4, 2)) * -2
    del tmp25
    tmp24 = einsum(tmp10, (0, 1, 2), v.baa.xov, (2, 3, 4), (3, 0, 1, 4))
    del tmp10
    t1new.bb += einsum(tmp24, (0, 1, 2, 3), t2.abab, (0, 2, 3, 4), (1, 4)) * -1
    del tmp24
    tmp14 = einsum(v.baa.xov, (0, 1, 2), tmp3, (3, 4, 0), (3, 1, 4, 2))
    t1new.aa += einsum(tmp14, (0, 1, 2, 3), t2.aaaa, (2, 1, 4, 3), (0, 4)) * -2
    del tmp14
    tmp15 = einsum(tmp3, (0, 1, 2), v.bbb.xov, (2, 3, 4), (0, 1, 3, 4))
    del tmp3
    t1new.aa += einsum(t2.abab, (0, 1, 2, 3), tmp15, (4, 0, 1, 3), (4, 2)) * -1
    del tmp15
    tmp20 = einsum(v.baa.xov, (0, 1, 2), tmp18, (3, 2, 0), (3, 1))
    del tmp18
    t1new.aa += einsum(tmp20, (0, 1), t1.aa, (1, 2), (0, 2)) * -1
    del tmp20
    t1new.aa += einsum(f.aa.oo, (0, 1), t1.aa, (1, 2), (0, 2)) * -1
    t1new.aa += einsum(f.aa.vv, (0, 1), t1.aa, (2, 1), (2, 0))
    t1new.bb += einsum(t1.bb, (0, 1), f.bb.oo, (2, 0), (2, 1)) * -1
    t1new.bb += einsum(f.bb.vv, (0, 1), t1.bb, (2, 1), (2, 0))
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), f.aa.vv, (4, 3), (0, 1, 2, 4)) * 2
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), f.bb.oo, (4, 1), (0, 4, 2, 3)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), f.aa.oo, (4, 0), (4, 1, 2, 3)) * -1
    t2new.abab += einsum(f.bb.vv, (0, 1), t2.abab, (2, 3, 4, 1), (2, 3, 4, 0))
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), f.bb.vv, (4, 3), (0, 1, 2, 4)) * 2

    return {f"t1new": t1new, f"t2new": t2new}

