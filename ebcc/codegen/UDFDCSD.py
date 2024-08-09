"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-08-09T22:11:19.899459
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
    Code generated by `albert` 0.0.0 on 2024-08-09T22:11:21.706198.

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
    e_cc : float
        Coupled cluster energy.
    """

    tmp0 = einsum(v.bbb.xov, (0, 1, 2), t1.bb, (1, 2), (0,))
    tmp5 = einsum(t1.bb, (0, 1), v.bbb.xov, (2, 3, 1), (0, 3, 2))
    tmp1 = einsum(v.baa.xov, (0, 1, 2), t1.aa, (1, 2), (0,))
    tmp1 += tmp0 * 2
    tmp3 = einsum(t1.aa, (0, 1), v.baa.xov, (2, 3, 1), (0, 3, 2))
    tmp6 = f.bb.ov.copy() * 2
    tmp6 += einsum(tmp5, (0, 1, 2), v.bbb.xov, (2, 0, 3), (1, 3)) * -1
    del tmp5
    e_cc = einsum(t1.bb, (0, 1), tmp6, (0, 1), ()) * 0.5
    del tmp6
    tmp2 = einsum(v.baa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    tmp2 += einsum(v.bbb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (3, 4, 0))
    tmp2 += einsum(tmp1, (0,), t1.aa, (1, 2), (1, 2, 0)) * 0.5
    del tmp1
    e_cc += einsum(v.baa.xov, (0, 1, 2), tmp2, (1, 2, 0), ())
    del tmp2
    tmp7 = einsum(t2.bbbb, (0, 1, 2, 3), v.bbb.xov, (4, 1, 3), (0, 2, 4))
    tmp7 += einsum(t1.bb, (0, 1), tmp0, (2,), (0, 1, 2)) * 0.5
    del tmp0
    e_cc += einsum(v.bbb.xov, (0, 1, 2), tmp7, (1, 2, 0), ())
    del tmp7
    tmp4 = f.aa.ov.copy() * 2
    tmp4 += einsum(tmp3, (0, 1, 2), v.baa.xov, (2, 0, 3), (1, 3)) * -1
    del tmp3
    e_cc += einsum(t1.aa, (0, 1), tmp4, (0, 1), ()) * 0.5
    del tmp4

    return e_cc

def update_amps(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T22:12:46.857900.

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
    tmp21 = einsum(t1.bb, (0, 1), v.bbb.xov, (2, 3, 1), (0, 3, 2))
    tmp6 = einsum(v.bbb.xov, (0, 1, 2), t1.bb, (1, 2), (0,))
    tmp5 = einsum(v.baa.xov, (0, 1, 2), t1.aa, (1, 2), (0,))
    tmp0 = einsum(t1.aa, (0, 1), v.baa.xov, (2, 3, 1), (0, 3, 2))
    tmp134 = einsum(t1.bb, (0, 1), v.bbb.xov, (2, 0, 3), (1, 3, 2))
    tmp130 = einsum(t1.bb, (0, 1), tmp21, (2, 0, 3), (2, 1, 3))
    tmp39 = einsum(v.bbb.xvv, (0, 1, 2), t1.bb, (3, 2), (3, 1, 0))
    t1new.bb = einsum(tmp39, (0, 1, 2), v.bbb.xoo, (2, 3, 0), (3, 1)) * -1
    tmp9 = tmp5.copy()
    tmp9 += tmp6
    tmp22 = einsum(tmp21, (0, 1, 2), v.bbb.xov, (2, 0, 3), (1, 3))
    tmp7 = tmp5.copy()
    del tmp5
    tmp7 += tmp6
    del tmp6
    tmp33 = einsum(v.bbb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    tmp29 = einsum(v.baa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    tmp110 = einsum(t1.bb, (0, 1), v.bbb.xoo, (2, 3, 0), (3, 1, 2))
    tmp70 = einsum(tmp0, (0, 1, 2), t1.aa, (1, 3), (0, 3, 2))
    tmp16 = einsum(v.baa.xvv, (0, 1, 2), t1.aa, (3, 2), (3, 1, 0))
    t1new.aa = einsum(v.baa.xoo, (0, 1, 2), tmp16, (2, 3, 0), (1, 3)) * -1
    tmp72 = einsum(t1.aa, (0, 1), v.baa.xov, (2, 0, 3), (1, 3, 2))
    tmp11 = einsum(v.baa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    tmp42 = einsum(v.baa.xoo, (0, 1, 2), t1.aa, (2, 3), (1, 3, 0))
    tmp3 = einsum(v.bbb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (3, 4, 0))
    tmp13 = einsum(v.baa.xov, (0, 1, 2), tmp0, (1, 3, 0), (3, 2))
    tmp176 = v.bbb.xvv.transpose((1, 2, 0)).copy()
    tmp176 += tmp134.transpose((1, 0, 2)) * -1
    tmp146 = tmp39.copy()
    tmp146 += tmp130 * -1
    t2new.abab = einsum(tmp11, (0, 1, 2), tmp146, (3, 4, 2), (0, 3, 1, 4)) * 2
    tmp32 = einsum(v.bbb.xov, (0, 1, 2), tmp9, (0,), (1, 2))
    t1new.bb += tmp32
    tmp35 = f.bb.ov.copy()
    tmp35 += tmp22 * -1
    tmp118 = tmp39.copy()
    tmp118 += tmp29 * 0.5
    tmp118 += tmp33
    tmp118 += einsum(t1.bb, (0, 1), tmp7, (2,), (0, 1, 2))
    tmp183 = tmp110.copy()
    tmp183 += tmp33 * -1
    tmp171 = v.bbb.xvv.transpose((1, 2, 0)).copy()
    tmp171 += tmp134.transpose((1, 0, 2)) * -1
    tmp111 = tmp110.copy() * -1
    tmp111 += tmp29 * 0.5
    tmp111 += tmp33
    t2new.abab += einsum(tmp11, (0, 1, 2), tmp111, (3, 4, 2), (0, 3, 1, 4)) * 2
    tmp79 = tmp16.copy()
    tmp79 += tmp70 * -1
    t2new.abab += einsum(tmp79, (0, 1, 2), tmp33, (3, 4, 2), (0, 3, 1, 4)) * 2
    tmp81 = v.baa.xvv.transpose((1, 2, 0)).copy()
    tmp81 += tmp72.transpose((1, 0, 2)) * -1
    tmp43 = tmp42.copy() * -1
    tmp43 += tmp11
    tmp43 += tmp3 * 0.5
    t2new.abab += einsum(tmp43, (0, 1, 2), tmp33, (3, 4, 2), (0, 3, 1, 4)) * 2
    tmp88 = tmp42.copy()
    tmp88 += tmp11 * -1
    tmp73 = v.baa.xvv.transpose((1, 2, 0)).copy()
    tmp73 += tmp72.transpose((1, 0, 2)) * -1
    tmp52 = tmp16.copy()
    tmp52 += tmp11
    tmp52 += tmp3 * 0.5
    tmp52 += einsum(t1.aa, (0, 1), tmp7, (2,), (0, 1, 2))
    tmp14 = f.aa.ov.copy()
    tmp14 += tmp13 * -1
    tmp10 = einsum(v.baa.xov, (0, 1, 2), tmp9, (0,), (1, 2))
    del tmp9
    t1new.aa += tmp10
    tmp177 = einsum(tmp21, (0, 1, 2), tmp176, (3, 4, 2), (0, 1, 3, 4))
    del tmp176
    tmp147 = einsum(v.bbb.xov, (0, 1, 2), tmp146, (3, 4, 0), (1, 3, 2, 4))
    tmp163 = einsum(v.bbb.xoo, (0, 1, 2), tmp21, (3, 4, 0), (3, 1, 2, 4))
    tmp23 = einsum(v.bbb.xov, (0, 1, 2), tmp7, (0,), (1, 2))
    tmp167 = f.bb.ov.copy()
    tmp167 += tmp22 * -1
    tmp167 += tmp32
    del tmp32
    tmp192 = einsum(tmp21, (0, 1, 2), tmp21, (3, 4, 2), (0, 3, 1, 4))
    tmp120 = einsum(tmp7, (0,), v.bbb.xvv, (0, 1, 2), (1, 2))
    tmp121 = einsum(tmp35, (0, 1), t1.bb, (0, 2), (2, 1))
    tmp119 = einsum(v.bbb.xov, (0, 1, 2), tmp118, (1, 3, 0), (2, 3))
    del tmp118
    tmp184 = einsum(v.bbb.xov, (0, 1, 2), tmp183, (3, 4, 0), (3, 1, 4, 2))
    del tmp183
    tmp185 = einsum(tmp171, (0, 1, 2), v.bbb.xoo, (2, 3, 4), (3, 4, 0, 1))
    tmp112 = einsum(v.bbb.xov, (0, 1, 2), tmp111, (3, 2, 0), (1, 3))
    tmp40 = einsum(tmp7, (0,), v.bbb.xoo, (0, 1, 2), (1, 2))
    tmp142 = einsum(v.bbb.xov, (0, 1, 2), v.baa.xov, (0, 3, 4), (3, 1, 4, 2))
    tmp80 = einsum(v.baa.xov, (0, 1, 2), tmp79, (3, 4, 0), (1, 3, 2, 4))
    tmp82 = einsum(tmp0, (0, 1, 2), tmp81, (3, 4, 2), (0, 1, 3, 4))
    del tmp81
    tmp44 = einsum(v.baa.xov, (0, 1, 2), tmp43, (3, 2, 0), (1, 3)) * 2
    tmp45 = einsum(v.baa.xoo, (0, 1, 2), tmp7, (0,), (1, 2)) * 2
    tmp98 = einsum(tmp0, (0, 1, 2), tmp0, (3, 4, 2), (3, 0, 4, 1))
    tmp89 = einsum(v.baa.xov, (0, 1, 2), tmp88, (3, 4, 0), (1, 3, 2, 4))
    del tmp88
    tmp90 = einsum(tmp73, (0, 1, 2), v.baa.xoo, (2, 3, 4), (3, 4, 0, 1))
    tmp19 = einsum(v.baa.xov, (0, 1, 2), tmp7, (0,), (1, 2))
    tmp53 = einsum(v.baa.xov, (0, 1, 2), tmp52, (1, 3, 0), (2, 3))
    del tmp52
    tmp54 = einsum(tmp7, (0,), v.baa.xvv, (0, 1, 2), (1, 2))
    tmp55 = einsum(tmp14, (0, 1), t1.aa, (0, 2), (2, 1))
    tmp67 = f.aa.ov.copy()
    tmp67 += tmp13 * -1
    tmp67 += tmp10
    del tmp10
    tmp62 = einsum(tmp0, (0, 1, 2), v.baa.xoo, (2, 3, 4), (0, 3, 4, 1))
    tmp178 = tmp147.transpose((1, 0, 3, 2)).copy() * -1
    tmp178 += tmp177.transpose((0, 1, 3, 2))
    del tmp177
    tmp174 = tmp39.copy() * -1
    tmp174 += tmp130
    tmp165 = einsum(t1.bb, (0, 1), tmp163, (2, 3, 4, 0), (2, 3, 4, 1))
    tmp188 = tmp110.copy()
    tmp188 += tmp29 * -0.5
    tmp200 = einsum(v.bbb.xov, (0, 1, 2), tmp171, (3, 4, 0), (1, 4, 3, 2))
    tmp198 = einsum(tmp21, (0, 1, 2), tmp39, (3, 4, 2), (0, 3, 1, 4))
    tmp24 = f.bb.ov.copy()
    tmp24 += tmp22 * -1
    del tmp22
    tmp24 += tmp23
    del tmp23
    t1new.bb += einsum(tmp24, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2
    t1new.aa += einsum(tmp24, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    tmp37 = einsum(v.bbb.xov, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (3, 4, 1, 2))
    t1new.bb += einsum(t2.bbbb, (0, 1, 2, 3), tmp37, (4, 0, 1, 3), (4, 2)) * -2
    tmp168 = einsum(t1.bb, (0, 1), tmp167, (2, 1), (2, 0))
    del tmp167
    tmp196 = einsum(tmp192, (0, 1, 2, 3), t1.bb, (2, 4), (1, 0, 3, 4))
    tmp131 = v.bbb.xov.transpose((1, 2, 0)).copy()
    tmp131 += tmp39
    tmp131 += tmp130 * -1
    t2new.abab += einsum(v.baa.xov, (0, 1, 2), tmp131, (3, 4, 0), (1, 3, 2, 4))
    tmp190 = tmp110.copy() * 2
    del tmp110
    tmp190 += tmp29 * -1
    tmp155 = einsum(v.bbb.xoo, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (3, 1, 2, 4))
    tmp28 = einsum(tmp21, (0, 1, 2), v.bbb.xov, (2, 3, 4), (0, 3, 1, 4))
    t1new.bb += einsum(tmp28, (0, 1, 2, 3), t2.bbbb, (2, 1, 4, 3), (0, 4)) * -2
    tmp122 = f.bb.vv.copy() * -1
    tmp122 += tmp119
    del tmp119
    tmp122 += tmp120.transpose((1, 0)) * -1
    del tmp120
    tmp122 += tmp121.transpose((1, 0))
    del tmp121
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp122, (3, 4), (0, 1, 2, 4)) * -1
    tmp117 = tmp29.copy()
    tmp117 += tmp33 * 2
    t2new.abab += einsum(v.baa.xov, (0, 1, 2), tmp117, (3, 4, 0), (1, 3, 2, 4))
    tmp186 = tmp184.copy() * -1
    del tmp184
    tmp186 += tmp185.transpose((1, 0, 3, 2)) * -1
    del tmp185
    tmp170 = tmp39.copy()
    tmp170 += tmp130 * -1
    tmp113 = f.bb.oo.copy()
    tmp113 += tmp112
    del tmp112
    tmp113 += tmp40.transpose((1, 0))
    t2new.abab += einsum(tmp113, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1
    tmp161 = einsum(v.bbb.xoo, (0, 1, 2), tmp39, (3, 4, 0), (3, 1, 2, 4))
    tmp203 = einsum(v.bbb.xov, (0, 1, 2), v.bbb.xvv, (0, 3, 4), (1, 3, 4, 2))
    tmp128 = v.baa.xvv.transpose((1, 2, 0)).copy()
    tmp128 += tmp72 * -1
    tmp150 = einsum(v.baa.xov, (0, 1, 2), v.bbb.xvv, (0, 3, 4), (1, 2, 3, 4))
    tmp132 = v.baa.xvv.transpose((1, 2, 0)).copy()
    tmp132 += tmp72 * -1
    del tmp72
    tmp140 = v.bbb.xvv.transpose((1, 2, 0)).copy()
    tmp140 += tmp134 * -1
    tmp143 = einsum(tmp142, (0, 1, 2, 3), t1.bb, (1, 4), (0, 2, 4, 3))
    del tmp142
    tmp135 = v.bbb.xvv.transpose((1, 2, 0)).copy()
    tmp135 += tmp134 * -1
    del tmp134
    tmp25 = einsum(v.baa.xoo, (0, 1, 2), tmp7, (0,), (1, 2))
    tmp83 = tmp80.transpose((1, 0, 3, 2)).copy() * -1
    tmp83 += tmp82.transpose((0, 1, 3, 2))
    del tmp82
    tmp46 = f.aa.oo.copy() * 2
    tmp46 += tmp44
    del tmp44
    tmp46 += tmp45.transpose((1, 0))
    del tmp45
    tmp103 = einsum(tmp98, (0, 1, 2, 3), t1.aa, (2, 4), (1, 0, 3, 4))
    tmp58 = tmp11.copy() * 2
    tmp58 += tmp3
    t2new.abab += einsum(tmp58, (0, 1, 2), v.bbb.xov, (2, 3, 4), (0, 3, 1, 4))
    tmp91 = tmp89.transpose((1, 0, 3, 2)).copy() * -1
    del tmp89
    tmp91 += tmp90.transpose((1, 0, 3, 2)) * -1
    del tmp90
    tmp1 = einsum(v.baa.xov, (0, 1, 2), tmp0, (3, 4, 0), (3, 1, 4, 2))
    t1new.aa += einsum(tmp1, (0, 1, 2, 3), t2.aaaa, (2, 1, 4, 3), (0, 4)) * -2
    tmp105 = einsum(tmp73, (0, 1, 2), v.baa.xov, (2, 3, 4), (3, 1, 0, 4))
    tmp93 = tmp42.copy()
    tmp93 += tmp3 * -0.5
    tmp17 = einsum(v.baa.xov, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    t1new.aa += einsum(tmp17, (0, 1, 2, 3), t2.aaaa, (1, 2, 4, 3), (0, 4)) * -2
    tmp101 = einsum(tmp0, (0, 1, 2), tmp16, (3, 4, 2), (0, 3, 1, 4))
    tmp75 = v.baa.xov.transpose((1, 2, 0)).copy()
    tmp75 += tmp16
    tmp75 += tmp70 * -1
    tmp20 = f.aa.ov.copy()
    tmp20 += tmp13 * -1
    del tmp13
    tmp20 += tmp19
    del tmp19
    t1new.bb += einsum(t2.abab, (0, 1, 2, 3), tmp20, (0, 2), (1, 3))
    t1new.aa += einsum(t2.aaaa, (0, 1, 2, 3), tmp20, (1, 3), (0, 2)) * 2
    tmp60 = einsum(v.baa.xoo, (0, 1, 2), tmp16, (3, 4, 0), (3, 1, 2, 4))
    tmp77 = tmp16.copy() * -1
    tmp77 += tmp70
    tmp56 = f.aa.vv.copy() * -1
    tmp56 += tmp53
    del tmp53
    tmp56 += tmp54.transpose((1, 0)) * -1
    del tmp54
    tmp56 += tmp55.transpose((1, 0))
    del tmp55
    t2new.abab += einsum(tmp56, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1
    tmp48 = einsum(v.baa.xoo, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 1, 2, 4))
    tmp108 = einsum(v.baa.xov, (0, 1, 2), v.baa.xvv, (0, 3, 4), (1, 3, 4, 2))
    tmp95 = tmp42.copy() * 2
    del tmp42
    tmp95 += tmp3 * -1
    tmp68 = einsum(tmp67, (0, 1), t1.aa, (2, 1), (2, 0))
    del tmp67
    t2new.abab += einsum(tmp68, (0, 1), t2.abab, (1, 2, 3, 4), (0, 2, 3, 4)) * -1
    tmp71 = tmp16.copy()
    tmp71 += tmp70 * -1
    del tmp70
    tmp64 = einsum(tmp62, (0, 1, 2, 3), t1.aa, (3, 4), (0, 1, 2, 4))
    tmp30 = t2.bbbb.copy() * 2
    tmp30 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (2, 0, 1, 3)) * -1
    tmp34 = tmp29.copy()
    tmp34 += tmp33 * 2
    tmp34 += einsum(t1.bb, (0, 1), tmp7, (2,), (0, 1, 2))
    tmp4 = t2.aaaa.copy() * 2
    tmp4 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -1
    tmp12 = tmp11.copy()
    tmp12 += tmp3 * 0.5
    tmp12 += einsum(t1.aa, (0, 1), tmp7, (2,), (0, 1, 2)) * 0.5
    tmp179 = einsum(tmp178, (0, 1, 2, 3), t2.bbbb, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp178
    t2new.bbbb = tmp179.transpose((0, 1, 3, 2)).copy() * 2
    t2new.bbbb += tmp179.transpose((1, 0, 3, 2)) * -2
    t2new.bbbb += tmp179 * -2
    del tmp179
    tmp156 = t2.bbbb.copy()
    tmp156 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3)) * 2
    t2new.bbbb += einsum(tmp155, (0, 1, 2, 3), tmp156, (0, 2, 4, 5), (1, 3, 5, 4)) * 0.5
    del tmp156
    tmp195 = einsum(tmp39, (0, 1, 2), tmp39, (3, 4, 2), (3, 0, 4, 1))
    t2new.bbbb += tmp195.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp195
    del tmp195
    tmp175 = einsum(tmp174, (0, 1, 2), tmp29, (3, 4, 2), (0, 3, 1, 4))
    del tmp174
    t2new.bbbb += tmp175.transpose((1, 0, 3, 2)) * -1
    t2new.bbbb += tmp175.transpose((1, 0, 2, 3))
    del tmp175
    tmp166 = einsum(t1.bb, (0, 1), tmp165, (2, 0, 3, 4), (2, 3, 1, 4))
    del tmp165
    t2new.bbbb += tmp166.transpose((0, 1, 3, 2))
    t2new.bbbb += tmp166 * -1
    t2new.bbbb += tmp166.transpose((1, 0, 3, 2)) * -1
    t2new.bbbb += tmp166.transpose((1, 0, 2, 3))
    del tmp166
    tmp189 = einsum(tmp188, (0, 1, 2), tmp29, (3, 4, 2), (0, 3, 1, 4))
    del tmp188
    t2new.bbbb += tmp189.transpose((1, 0, 2, 3))
    t2new.bbbb += tmp189 * -1
    del tmp189
    tmp201 = einsum(v.bbb.xvv, (0, 1, 2), v.bbb.xvv, (0, 3, 4), (3, 1, 2, 4))
    tmp201 += einsum(tmp200, (0, 1, 2, 3), t1.bb, (0, 4), (3, 2, 4, 1))
    del tmp200
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), tmp201, (2, 3, 4, 5), (0, 1, 5, 4)) * 2
    del tmp201
    tmp180 = einsum(tmp29, (0, 1, 2), tmp146, (3, 4, 2), (0, 3, 1, 4))
    del tmp146
    t2new.bbbb += tmp180.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp180.transpose((1, 0, 3, 2))
    del tmp180
    tmp199 = einsum(t1.bb, (0, 1), tmp198, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp198
    t2new.bbbb += tmp199.transpose((1, 0, 2, 3))
    t2new.bbbb += tmp199 * -1
    t2new.bbbb += tmp199.transpose((1, 0, 3, 2)) * -1
    t2new.bbbb += tmp199.transpose((0, 1, 3, 2))
    del tmp199
    tmp123 = einsum(tmp24, (0, 1), t1.bb, (2, 1), (2, 0))
    del tmp24
    t2new.bbbb += einsum(tmp123, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp123, (4, 1), (0, 4, 2, 3)) * -1
    del tmp123
    tmp181 = einsum(t1.bb, (0, 1), tmp37, (2, 0, 3, 4), (2, 3, 1, 4))
    del tmp37
    t2new.bbbb += tmp181.transpose((1, 0, 2, 3))
    t2new.bbbb += tmp181 * -1
    t2new.bbbb += tmp181.transpose((1, 0, 3, 2)) * -1
    t2new.bbbb += tmp181.transpose((0, 1, 3, 2))
    del tmp181
    tmp169 = einsum(tmp168, (0, 1), t2.bbbb, (2, 0, 3, 4), (1, 2, 3, 4)) * -1
    del tmp168
    t2new.bbbb += tmp169.transpose((0, 1, 3, 2))
    t2new.bbbb += tmp169.transpose((0, 1, 3, 2))
    t2new.bbbb += tmp169.transpose((1, 0, 3, 2)) * -1
    del tmp169
    tmp197 = einsum(t1.bb, (0, 1), tmp196, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp196
    t2new.bbbb += tmp197.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp197
    del tmp197
    tmp182 = einsum(tmp33, (0, 1, 2), tmp29, (3, 4, 2), (3, 0, 4, 1))
    del tmp33
    t2new.bbbb += tmp182.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp182
    t2new.bbbb += tmp182.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp182.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp182.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp182.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp182.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp182
    del tmp182
    tmp158 = einsum(t1.bb, (0, 1), t1.bb, (2, 3), (2, 0, 1, 3)) * 2
    tmp158 += t2.bbbb * -1
    t2new.bbbb += einsum(tmp155, (0, 1, 2, 3), tmp158, (0, 2, 4, 5), (1, 3, 5, 4)) * -0.5
    del tmp158
    tmp173 = einsum(v.bbb.xov, (0, 1, 2), tmp131, (3, 4, 0), (1, 3, 2, 4))
    del tmp131
    t2new.bbbb += tmp173.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp173
    del tmp173
    tmp191 = einsum(tmp29, (0, 1, 2), tmp190, (3, 4, 2), (3, 0, 4, 1)) * 0.5
    del tmp190
    t2new.bbbb += tmp191.transpose((1, 0, 3, 2)) * -1
    t2new.bbbb += tmp191.transpose((0, 1, 3, 2))
    del tmp191
    tmp157 = einsum(tmp155, (0, 1, 2, 3), t2.bbbb, (1, 3, 4, 5), (2, 0, 4, 5))
    del tmp155
    t2new.bbbb += tmp157.transpose((0, 1, 3, 2)) * -0.5
    t2new.bbbb += tmp157.transpose((0, 1, 3, 2)) * -0.5
    del tmp157
    tmp202 = einsum(t1.bb, (0, 1), tmp28, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp28
    t2new.bbbb += tmp202 * -1
    t2new.bbbb += tmp202.transpose((1, 0, 2, 3))
    del tmp202
    tmp159 = einsum(t2.bbbb, (0, 1, 2, 3), tmp122, (3, 4), (0, 1, 2, 4)) * -2
    del tmp122
    t2new.bbbb += tmp159.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp159.transpose((1, 0, 3, 2))
    del tmp159
    tmp193 = einsum(t2.bbbb, (0, 1, 2, 3), tmp192, (4, 5, 1, 0), (5, 4, 2, 3))
    del tmp192
    t2new.bbbb += tmp193.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp193.transpose((0, 1, 3, 2)) * -1
    del tmp193
    tmp160 = einsum(v.bbb.xov, (0, 1, 2), tmp117, (3, 4, 0), (1, 3, 2, 4))
    del tmp117
    t2new.bbbb += tmp160.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp160.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp160.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp160
    del tmp160
    tmp187 = einsum(tmp186, (0, 1, 2, 3), t2.bbbb, (4, 1, 5, 3), (0, 4, 2, 5)) * 2
    del tmp186
    t2new.bbbb += tmp187.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp187.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp187.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp187
    del tmp187
    tmp172 = einsum(tmp170, (0, 1, 2), v.bbb.xov, (2, 3, 4), (0, 3, 1, 4)) * -1
    del tmp170
    tmp172 += einsum(tmp21, (0, 1, 2), tmp171, (3, 4, 2), (0, 1, 4, 3))
    del tmp171
    t2new.bbbb += einsum(tmp172, (0, 1, 2, 3), t2.bbbb, (4, 1, 5, 3), (4, 0, 2, 5)) * 2
    del tmp172
    tmp154 = einsum(t2.bbbb, (0, 1, 2, 3), tmp113, (1, 4), (0, 4, 2, 3)) * -1
    del tmp113
    t2new.bbbb += tmp154.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp154.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp154.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp154.transpose((0, 1, 3, 2)) * -1
    del tmp154
    tmp162 = einsum(t1.bb, (0, 1), tmp161, (2, 0, 3, 4), (2, 3, 1, 4))
    del tmp161
    t2new.bbbb += tmp162
    t2new.bbbb += tmp162.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp162.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp162.transpose((1, 0, 3, 2))
    del tmp162
    tmp204 = einsum(tmp203, (0, 1, 2, 3), t2.bbbb, (4, 5, 2, 3), (4, 5, 0, 1)) * -1
    del tmp203
    t2new.bbbb += einsum(t1.bb, (0, 1), tmp204, (2, 3, 0, 4), (2, 3, 1, 4)) * -2
    del tmp204
    tmp194 = einsum(tmp39, (0, 1, 2), v.bbb.xov, (2, 3, 4), (0, 3, 4, 1))
    t2new.bbbb += tmp194.transpose((1, 0, 3, 2)) * -1
    t2new.bbbb += tmp194.transpose((0, 1, 3, 2))
    del tmp194
    tmp164 = einsum(tmp163, (0, 1, 2, 3), t2.bbbb, (1, 3, 4, 5), (0, 2, 4, 5))
    del tmp163
    t2new.bbbb += tmp164.transpose((0, 1, 3, 2))
    t2new.bbbb += tmp164.transpose((0, 1, 3, 2))
    t2new.bbbb += tmp164.transpose((1, 0, 3, 2)) * -1
    t2new.bbbb += tmp164.transpose((1, 0, 3, 2)) * -1
    del tmp164
    tmp139 = einsum(tmp43, (0, 1, 2), v.baa.xov, (2, 3, 4), (3, 0, 4, 1))
    tmp139 += einsum(v.baa.xoo, (0, 1, 2), tmp128, (3, 4, 0), (1, 2, 4, 3)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp139, (0, 4, 2, 5), (4, 1, 5, 3))
    del tmp139
    tmp138 = einsum(tmp0, (0, 1, 2), tmp21, (3, 4, 2), (0, 1, 3, 4))
    tmp151 = einsum(t2.abab, (0, 1, 2, 3), tmp150, (4, 2, 5, 3), (0, 4, 1, 5))
    del tmp150
    t2new.abab += einsum(tmp151, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3)) * -1
    del tmp151
    tmp115 = t2.abab.copy()
    tmp115 += einsum(t1.bb, (0, 1), t1.aa, (2, 3), (2, 0, 3, 1)) * 2
    tmp133 = einsum(v.bbb.xoo, (0, 1, 2), tmp132, (3, 4, 0), (1, 2, 4, 3))
    t2new.abab += einsum(tmp133, (0, 1, 2, 3), t2.abab, (4, 0, 2, 5), (4, 1, 3, 5)) * -1
    del tmp133
    tmp129 = einsum(tmp128, (0, 1, 2), tmp21, (3, 4, 2), (3, 4, 1, 0))
    del tmp128
    t2new.abab += einsum(tmp129, (0, 1, 2, 3), t2.abab, (4, 1, 2, 5), (4, 0, 3, 5)) * -1
    del tmp129
    tmp144 = einsum(tmp143, (0, 1, 2, 3), t1.aa, (0, 4), (1, 4, 3, 2))
    del tmp143
    tmp144 += einsum(tmp140, (0, 1, 2), v.baa.xvv, (2, 3, 4), (3, 4, 1, 0))
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp144, (2, 4, 3, 5), (0, 1, 4, 5))
    del tmp144
    tmp114 = einsum(v.baa.xoo, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 2, 3, 4))
    t2new.abab += einsum(tmp114, (0, 1, 2, 3), t2.abab, (1, 3, 4, 5), (0, 2, 4, 5)) * 0.5
    t2new.abab += einsum(tmp115, (0, 1, 2, 3), tmp114, (0, 4, 1, 5), (4, 5, 2, 3)) * 0.5
    del tmp114, tmp115
    tmp38 = einsum(v.baa.xov, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 3, 4, 2))
    t2new.abab += einsum(t1.bb, (0, 1), tmp38, (2, 3, 0, 4), (2, 3, 4, 1)) * -1
    t1new.bb += einsum(tmp38, (0, 1, 2, 3), t2.abab, (0, 2, 3, 4), (1, 4)) * -1
    del tmp38
    tmp136 = einsum(v.baa.xoo, (0, 1, 2), tmp135, (3, 4, 0), (1, 2, 4, 3))
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp136, (0, 4, 3, 5), (4, 1, 2, 5)) * -1
    del tmp136
    tmp2 = einsum(tmp0, (0, 1, 2), v.bbb.xov, (2, 3, 4), (0, 1, 3, 4))
    t2new.abab += einsum(tmp2, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3)) * -1
    t1new.aa += einsum(tmp2, (0, 1, 2, 3), t2.abab, (1, 2, 4, 3), (0, 4)) * -1
    del tmp2
    tmp148 = tmp147.transpose((1, 0, 2, 3)).copy() * -1
    del tmp147
    tmp148 += einsum(tmp21, (0, 1, 2), tmp135, (3, 4, 2), (0, 1, 4, 3))
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp148, (4, 1, 3, 5), (0, 4, 2, 5)) * -1
    del tmp148
    tmp152 = einsum(v.baa.xoo, (0, 1, 2), tmp39, (3, 4, 0), (1, 2, 3, 4))
    t2new.abab += einsum(tmp152, (0, 1, 2, 3), t1.aa, (0, 4), (1, 2, 4, 3)) * -1
    del tmp152
    tmp149 = einsum(tmp0, (0, 1, 2), tmp135, (3, 4, 2), (0, 1, 4, 3))
    del tmp135
    t2new.abab += einsum(tmp149, (0, 1, 2, 3), t2.abab, (1, 4, 5, 2), (0, 4, 5, 3)) * -1
    del tmp149
    tmp124 = einsum(v.baa.xoo, (0, 1, 2), tmp21, (3, 4, 0), (1, 2, 3, 4))
    tmp145 = tmp80.transpose((1, 0, 2, 3)).copy() * -1
    del tmp80
    tmp145 += einsum(tmp0, (0, 1, 2), tmp132, (3, 4, 2), (0, 1, 4, 3))
    del tmp132
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp145, (4, 0, 2, 5), (4, 1, 5, 3)) * -1
    del tmp145
    tmp18 = einsum(v.bbb.xov, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    t2new.abab += einsum(tmp18, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3)) * -1
    t1new.aa += einsum(tmp18, (0, 1, 2, 3), t2.abab, (1, 2, 4, 3), (0, 4)) * -1
    del tmp18
    tmp153 = einsum(tmp0, (0, 1, 2), tmp39, (3, 4, 2), (0, 1, 3, 4))
    t2new.abab += einsum(tmp153, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3)) * -1
    del tmp153
    tmp125 = t2.abab.copy()
    tmp125 += einsum(t1.bb, (0, 1), t1.aa, (2, 3), (2, 0, 3, 1))
    t2new.abab += einsum(tmp138, (0, 1, 2, 3), tmp125, (1, 3, 4, 5), (0, 2, 4, 5))
    del tmp138
    t2new.abab += einsum(tmp125, (0, 1, 2, 3), tmp124, (4, 0, 5, 1), (4, 5, 2, 3))
    del tmp124
    tmp116 = f.aa.oo.copy()
    tmp116 += einsum(v.baa.xov, (0, 1, 2), tmp43, (3, 2, 0), (1, 3))
    del tmp43
    tmp116 += tmp25.transpose((1, 0))
    t2new.abab += einsum(tmp116, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1
    del tmp116
    tmp126 = einsum(v.bbb.xoo, (0, 1, 2), tmp16, (3, 4, 0), (3, 1, 2, 4))
    t2new.abab += einsum(t1.bb, (0, 1), tmp126, (2, 0, 3, 4), (2, 3, 4, 1)) * -1
    del tmp126
    tmp127 = einsum(tmp0, (0, 1, 2), v.bbb.xoo, (2, 3, 4), (0, 1, 3, 4))
    t2new.abab += einsum(tmp127, (0, 1, 2, 3), tmp125, (1, 2, 4, 5), (0, 3, 4, 5))
    del tmp125, tmp127
    tmp141 = einsum(v.bbb.xov, (0, 1, 2), tmp111, (3, 4, 0), (1, 3, 2, 4))
    del tmp111
    tmp141 += einsum(tmp140, (0, 1, 2), v.bbb.xoo, (2, 3, 4), (3, 4, 1, 0)) * -1
    del tmp140
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp141, (1, 4, 3, 5), (0, 4, 2, 5))
    del tmp141
    tmp137 = v.bbb.xov.transpose((1, 2, 0)).copy()
    tmp137 += tmp39
    del tmp39
    tmp137 += tmp130 * -1
    del tmp130
    t2new.abab += einsum(tmp137, (0, 1, 2), tmp16, (3, 4, 2), (3, 0, 4, 1))
    del tmp137
    tmp84 = einsum(t2.aaaa, (0, 1, 2, 3), tmp83, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp83
    t2new.aaaa = tmp84.transpose((1, 0, 2, 3)).copy() * 2
    t2new.aaaa += tmp84 * -2
    t2new.aaaa += tmp84.transpose((1, 0, 3, 2)) * -2
    del tmp84
    tmp47 = einsum(tmp46, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -0.5
    del tmp46
    t2new.aaaa += tmp47.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp47.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp47.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp47.transpose((0, 1, 3, 2)) * -1
    del tmp47
    tmp97 = einsum(v.baa.xov, (0, 1, 2), tmp16, (3, 4, 0), (3, 1, 2, 4))
    t2new.aaaa += tmp97.transpose((1, 0, 3, 2)) * -1
    t2new.aaaa += tmp97.transpose((0, 1, 3, 2))
    del tmp97
    tmp51 = einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * 2
    tmp51 += t2.aaaa * -1
    t2new.aaaa += einsum(tmp48, (0, 1, 2, 3), tmp51, (0, 2, 4, 5), (1, 3, 5, 4)) * -0.5
    del tmp51
    tmp104 = einsum(tmp103, (0, 1, 2, 3), t1.aa, (2, 4), (0, 1, 4, 3))
    del tmp103
    t2new.aaaa += tmp104.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp104
    del tmp104
    tmp59 = einsum(tmp58, (0, 1, 2), v.baa.xov, (2, 3, 4), (3, 0, 4, 1))
    del tmp58
    t2new.aaaa += tmp59.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp59.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp59.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp59
    del tmp59
    tmp92 = einsum(t2.aaaa, (0, 1, 2, 3), tmp91, (4, 1, 5, 3), (0, 4, 2, 5)) * 2
    del tmp91
    t2new.aaaa += tmp92
    t2new.aaaa += tmp92.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp92.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp92.transpose((1, 0, 3, 2))
    del tmp92
    tmp107 = einsum(t1.aa, (0, 1), tmp1, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp1
    t2new.aaaa += tmp107 * -1
    t2new.aaaa += tmp107.transpose((1, 0, 2, 3))
    del tmp107
    tmp106 = einsum(v.baa.xvv, (0, 1, 2), v.baa.xvv, (0, 3, 4), (3, 1, 2, 4))
    tmp106 += einsum(t1.aa, (0, 1), tmp105, (0, 2, 3, 4), (4, 3, 1, 2))
    del tmp105
    t2new.aaaa += einsum(tmp106, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), (4, 5, 3, 2)) * 2
    del tmp106
    tmp94 = einsum(tmp93, (0, 1, 2), tmp3, (3, 4, 2), (3, 0, 4, 1))
    del tmp93
    t2new.aaaa += tmp94.transpose((0, 1, 3, 2))
    t2new.aaaa += tmp94.transpose((1, 0, 3, 2)) * -1
    del tmp94
    tmp86 = einsum(t1.aa, (0, 1), tmp17, (2, 0, 3, 4), (2, 3, 1, 4))
    del tmp17
    t2new.aaaa += tmp86.transpose((1, 0, 2, 3))
    t2new.aaaa += tmp86 * -1
    t2new.aaaa += tmp86.transpose((1, 0, 3, 2)) * -1
    t2new.aaaa += tmp86.transpose((0, 1, 3, 2))
    del tmp86
    tmp102 = einsum(t1.aa, (0, 1), tmp101, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp101
    t2new.aaaa += tmp102.transpose((1, 0, 2, 3))
    t2new.aaaa += tmp102 * -1
    t2new.aaaa += tmp102.transpose((1, 0, 3, 2)) * -1
    t2new.aaaa += tmp102.transpose((0, 1, 3, 2))
    del tmp102
    tmp76 = einsum(v.baa.xov, (0, 1, 2), tmp75, (3, 4, 0), (1, 3, 2, 4))
    del tmp75
    t2new.aaaa += tmp76.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp76
    del tmp76
    tmp66 = einsum(tmp20, (0, 1), t1.aa, (2, 1), (2, 0))
    del tmp20
    t2new.aaaa += einsum(tmp66, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -1
    del tmp66
    tmp61 = einsum(tmp60, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3))
    del tmp60
    t2new.aaaa += tmp61
    t2new.aaaa += tmp61.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp61.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp61.transpose((1, 0, 3, 2))
    del tmp61
    tmp78 = einsum(tmp77, (0, 1, 2), tmp3, (3, 4, 2), (3, 0, 4, 1))
    del tmp77
    t2new.aaaa += tmp78 * -1
    t2new.aaaa += tmp78.transpose((0, 1, 3, 2))
    del tmp78
    tmp85 = einsum(tmp79, (0, 1, 2), tmp3, (3, 4, 2), (3, 0, 4, 1))
    del tmp79
    t2new.aaaa += tmp85.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp85.transpose((1, 0, 3, 2))
    del tmp85
    tmp57 = einsum(tmp56, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -2
    del tmp56
    t2new.aaaa += tmp57.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp57.transpose((1, 0, 3, 2))
    del tmp57
    tmp50 = einsum(t2.aaaa, (0, 1, 2, 3), tmp48, (4, 0, 5, 1), (5, 4, 2, 3))
    t2new.aaaa += tmp50.transpose((0, 1, 3, 2)) * -0.5
    t2new.aaaa += tmp50.transpose((0, 1, 3, 2)) * -0.5
    del tmp50
    tmp109 = einsum(t2.aaaa, (0, 1, 2, 3), tmp108, (4, 5, 2, 3), (0, 1, 4, 5)) * -1
    del tmp108
    t2new.aaaa += einsum(t1.aa, (0, 1), tmp109, (2, 3, 0, 4), (2, 3, 1, 4)) * -2
    del tmp109
    tmp96 = einsum(tmp95, (0, 1, 2), tmp3, (3, 4, 2), (3, 0, 4, 1)) * 0.5
    del tmp95
    t2new.aaaa += tmp96 * -1
    t2new.aaaa += tmp96.transpose((1, 0, 2, 3))
    del tmp96
    tmp49 = t2.aaaa.copy()
    tmp49 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (2, 0, 3, 1)) * 2
    t2new.aaaa += einsum(tmp48, (0, 1, 2, 3), tmp49, (0, 2, 4, 5), (1, 3, 5, 4)) * 0.5
    del tmp48, tmp49
    tmp99 = einsum(tmp98, (0, 1, 2, 3), t2.aaaa, (3, 2, 4, 5), (1, 0, 4, 5))
    del tmp98
    t2new.aaaa += tmp99.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp99.transpose((0, 1, 3, 2)) * -1
    del tmp99
    tmp69 = einsum(t2.aaaa, (0, 1, 2, 3), tmp68, (4, 1), (0, 4, 2, 3)) * -1
    del tmp68
    t2new.aaaa += tmp69.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp69.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp69.transpose((0, 1, 3, 2)) * -1
    del tmp69
    tmp87 = einsum(tmp3, (0, 1, 2), tmp11, (3, 4, 2), (3, 0, 4, 1))
    del tmp11
    t2new.aaaa += tmp87.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp87
    t2new.aaaa += tmp87.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp87.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp87.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp87.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp87.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp87
    del tmp87
    tmp63 = einsum(tmp62, (0, 1, 2, 3), t2.aaaa, (1, 3, 4, 5), (0, 2, 4, 5))
    del tmp62
    t2new.aaaa += tmp63.transpose((0, 1, 3, 2))
    t2new.aaaa += tmp63.transpose((0, 1, 3, 2))
    t2new.aaaa += tmp63.transpose((1, 0, 3, 2)) * -1
    t2new.aaaa += tmp63.transpose((1, 0, 3, 2)) * -1
    del tmp63
    tmp74 = einsum(v.baa.xov, (0, 1, 2), tmp71, (3, 4, 0), (3, 1, 4, 2)) * -1
    del tmp71
    tmp74 += einsum(tmp0, (0, 1, 2), tmp73, (3, 4, 2), (0, 1, 4, 3))
    del tmp73
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), tmp74, (4, 1, 5, 3), (0, 4, 5, 2)) * 2
    del tmp74
    tmp100 = einsum(tmp16, (0, 1, 2), tmp16, (3, 4, 2), (0, 3, 1, 4))
    del tmp16
    t2new.aaaa += tmp100.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp100
    del tmp100
    tmp65 = einsum(t1.aa, (0, 1), tmp64, (2, 0, 3, 4), (2, 3, 1, 4))
    del tmp64
    t2new.aaaa += tmp65.transpose((0, 1, 3, 2))
    t2new.aaaa += tmp65 * -1
    t2new.aaaa += tmp65.transpose((1, 0, 3, 2)) * -1
    t2new.aaaa += tmp65.transpose((1, 0, 2, 3))
    del tmp65
    tmp41 = f.bb.oo.copy()
    tmp41 += einsum(v.bbb.xoo, (0, 1, 2), tmp21, (2, 3, 0), (3, 1)) * -1
    tmp41 += tmp40.transpose((1, 0))
    del tmp40
    t1new.bb += einsum(t1.bb, (0, 1), tmp41, (0, 2), (2, 1)) * -1
    del tmp41
    tmp31 = tmp29.copy()
    del tmp29
    tmp31 += einsum(tmp30, (0, 1, 2, 3), v.bbb.xov, (4, 0, 2), (1, 3, 4))
    del tmp30
    tmp31 += einsum(t1.bb, (0, 1), tmp7, (2,), (0, 1, 2))
    t1new.bb += einsum(v.bbb.xvv, (0, 1, 2), tmp31, (3, 2, 0), (3, 1))
    del tmp31
    tmp36 = einsum(v.bbb.xov, (0, 1, 2), tmp34, (3, 2, 0), (1, 3))
    del tmp34
    tmp36 += einsum(t1.bb, (0, 1), tmp35, (2, 1), (2, 0))
    del tmp35
    t1new.bb += einsum(t1.bb, (0, 1), tmp36, (0, 2), (2, 1)) * -1
    del tmp36
    tmp27 = einsum(v.baa.xov, (0, 1, 2), tmp21, (3, 4, 0), (1, 3, 4, 2))
    del tmp21
    t1new.bb += einsum(tmp27, (0, 1, 2, 3), t2.abab, (0, 2, 3, 4), (1, 4)) * -1
    del tmp27
    tmp26 = f.aa.oo.copy()
    tmp26 += einsum(tmp0, (0, 1, 2), v.baa.xoo, (2, 3, 0), (1, 3)) * -1
    del tmp0
    tmp26 += tmp25.transpose((1, 0))
    del tmp25
    t1new.aa += einsum(tmp26, (0, 1), t1.aa, (0, 2), (1, 2)) * -1
    del tmp26
    tmp8 = tmp3.copy() * 0.5
    del tmp3
    tmp8 += einsum(v.baa.xov, (0, 1, 2), tmp4, (1, 3, 2, 4), (3, 4, 0)) * 0.5
    del tmp4
    tmp8 += einsum(t1.aa, (0, 1), tmp7, (2,), (0, 1, 2)) * 0.5
    del tmp7
    t1new.aa += einsum(tmp8, (0, 1, 2), v.baa.xvv, (2, 3, 1), (0, 3)) * 2
    del tmp8
    tmp15 = einsum(v.baa.xov, (0, 1, 2), tmp12, (3, 2, 0), (1, 3)) * 2
    del tmp12
    tmp15 += einsum(tmp14, (0, 1), t1.aa, (2, 1), (0, 2))
    del tmp14
    t1new.aa += einsum(tmp15, (0, 1), t1.aa, (0, 2), (1, 2)) * -1
    del tmp15
    t1new.aa += f.aa.ov
    t1new.aa += einsum(f.aa.vv, (0, 1), t1.aa, (2, 1), (2, 0))
    t1new.bb += einsum(f.bb.vv, (0, 1), t1.bb, (2, 1), (2, 0))
    t1new.bb += f.bb.ov

    return {f"t1new": t1new, f"t2new": t2new}

