"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-09-29T17:09:50.861759
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
    Code generated by `albert` 0.0.0 on 2024-09-29T17:09:52.671447.

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

    tmp0 = einsum(v.baa.xov, (0, 1, 2), t1.aa, (1, 2), (0,))
    tmp5 = einsum(t1.aa, (0, 1), v.baa.xov, (2, 3, 1), (2, 0, 3))
    tmp1 = np.copy(tmp0) * 2
    tmp1 += einsum(v.bbb.xov, (0, 1, 2), t1.bb, (1, 2), (0,))
    tmp3 = einsum(v.bbb.xov, (0, 1, 2), t1.bb, (3, 2), (0, 3, 1))
    tmp7 = einsum(v.baa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (0, 3, 4))
    tmp7 += einsum(tmp0, (0,), t1.aa, (1, 2), (0, 1, 2)) * 0.5
    del tmp0
    tmp6 = np.copy(f.aa.ov)
    tmp6 += einsum(v.baa.xov, (0, 1, 2), tmp5, (0, 1, 3), (3, 2)) * -0.5
    del tmp5
    tmp2 = einsum(v.baa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (0, 3, 4))
    tmp2 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbb.xov, (4, 1, 3), (4, 0, 2))
    tmp2 += einsum(tmp1, (0,), t1.bb, (1, 2), (0, 1, 2)) * 0.5
    del tmp1
    tmp4 = np.copy(f.bb.ov)
    tmp4 += einsum(tmp3, (0, 1, 2), v.bbb.xov, (0, 1, 3), (2, 3)) * -0.5
    del tmp3
    e_cc = einsum(tmp2, (0, 1, 2), v.bbb.xov, (0, 1, 2), ())
    del tmp2
    e_cc += einsum(t1.bb, (0, 1), tmp4, (0, 1), ())
    del tmp4
    e_cc += einsum(tmp6, (0, 1), t1.aa, (0, 1), ())
    del tmp6
    e_cc += einsum(v.baa.xov, (0, 1, 2), tmp7, (0, 1, 2), ())
    del tmp7

    return e_cc

def update_amps(f=None, t1=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-09-29T17:11:17.045708.

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
    tmp5 = einsum(v.baa.xov, (0, 1, 2), t1.aa, (1, 2), (0,))
    tmp6 = einsum(v.bbb.xov, (0, 1, 2), t1.bb, (1, 2), (0,))
    tmp19 = einsum(v.bbb.xov, (0, 1, 2), t1.bb, (3, 2), (0, 3, 1))
    tmp0 = einsum(t1.aa, (0, 1), v.baa.xov, (2, 3, 1), (2, 0, 3))
    tmp107 = einsum(t1.bb, (0, 1), v.bbb.xoo, (2, 3, 0), (2, 3, 1))
    tmp33 = einsum(t2.bbbb, (0, 1, 2, 3), v.bbb.xov, (4, 1, 3), (4, 0, 2))
    tmp29 = einsum(v.baa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (0, 3, 4))
    tmp9 = np.copy(tmp5)
    tmp9 += tmp6
    tmp130 = einsum(t1.bb, (0, 1), v.bbb.xov, (2, 0, 3), (2, 1, 3))
    tmp38 = einsum(v.bbb.xvv, (0, 1, 2), t1.bb, (3, 2), (0, 3, 1))
    tmp126 = einsum(t1.bb, (0, 1), tmp19, (2, 3, 0), (2, 3, 1))
    tmp20 = einsum(tmp19, (0, 1, 2), v.bbb.xov, (0, 1, 3), (2, 3))
    tmp7 = np.copy(tmp5)
    del tmp5
    tmp7 += tmp6
    del tmp6
    tmp3 = einsum(v.bbb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (0, 3, 4))
    tmp18 = einsum(v.baa.xvv, (0, 1, 2), t1.aa, (3, 2), (0, 3, 1))
    tmp11 = einsum(v.baa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (0, 3, 4))
    tmp13 = einsum(v.baa.xov, (0, 1, 2), tmp0, (0, 1, 3), (3, 2))
    tmp71 = einsum(v.baa.xov, (0, 1, 2), t1.aa, (1, 3), (0, 3, 2))
    tmp69 = einsum(tmp0, (0, 1, 2), t1.aa, (2, 3), (0, 1, 3))
    tmp42 = einsum(t1.aa, (0, 1), v.baa.xoo, (2, 3, 0), (2, 3, 1))
    tmp108 = np.copy(tmp107) * -1
    tmp108 += tmp29 * 0.5
    tmp108 += tmp33
    tmp32 = einsum(v.bbb.xov, (0, 1, 2), tmp9, (0,), (1, 2))
    tmp168 = np.copy(v.bbb.xvv)
    tmp168 += np.transpose(tmp130, (0, 2, 1)) * -1
    tmp180 = np.copy(tmp107)
    tmp180 += tmp33 * -1
    tmp173 = np.copy(v.bbb.xvv)
    tmp173 += np.transpose(tmp130, (0, 2, 1)) * -1
    tmp143 = np.copy(tmp38)
    tmp143 += tmp126 * -1
    tmp35 = np.copy(f.bb.ov)
    tmp35 += tmp20 * -1
    tmp114 = np.copy(tmp38)
    tmp114 += tmp29 * 0.5
    tmp114 += tmp33
    tmp114 += einsum(tmp7, (0,), t1.bb, (1, 2), (0, 1, 2))
    tmp51 = np.copy(tmp18)
    tmp51 += tmp11
    tmp51 += tmp3 * 0.5
    tmp51 += einsum(tmp7, (0,), t1.aa, (1, 2), (0, 1, 2))
    tmp14 = np.copy(f.aa.ov)
    tmp14 += tmp13 * -1
    tmp10 = einsum(v.baa.xov, (0, 1, 2), tmp9, (0,), (1, 2))
    del tmp9
    tmp80 = np.copy(v.baa.xvv)
    tmp80 += np.transpose(tmp71, (0, 2, 1)) * -1
    tmp78 = np.copy(tmp18)
    tmp78 += tmp69 * -1
    tmp72 = np.copy(v.baa.xvv)
    tmp72 += np.transpose(tmp71, (0, 2, 1)) * -1
    tmp87 = np.copy(tmp42)
    tmp87 += tmp11 * -1
    tmp43 = np.copy(tmp42) * -1
    tmp43 += tmp11
    tmp43 += tmp3 * 0.5
    tmp109 = einsum(v.bbb.xov, (0, 1, 2), tmp108, (0, 3, 2), (1, 3))
    tmp40 = einsum(tmp7, (0,), v.bbb.xoo, (0, 1, 2), (1, 2))
    tmp164 = np.copy(f.bb.ov)
    tmp164 += tmp20 * -1
    tmp164 += tmp32
    tmp182 = einsum(tmp168, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp181 = einsum(tmp180, (0, 1, 2), v.bbb.xov, (0, 3, 4), (1, 3, 2, 4))
    del tmp180
    tmp174 = einsum(tmp173, (0, 1, 2), tmp19, (0, 3, 4), (3, 4, 1, 2))
    del tmp173
    tmp144 = einsum(v.bbb.xov, (0, 1, 2), tmp143, (0, 3, 4), (1, 3, 2, 4))
    tmp189 = einsum(tmp19, (0, 1, 2), tmp19, (0, 3, 4), (1, 3, 2, 4))
    tmp160 = einsum(tmp19, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 3, 4, 2))
    tmp116 = einsum(v.bbb.xvv, (0, 1, 2), tmp7, (0,), (1, 2))
    tmp117 = einsum(t1.bb, (0, 1), tmp35, (0, 2), (1, 2))
    tmp115 = einsum(v.bbb.xov, (0, 1, 2), tmp114, (0, 1, 3), (2, 3))
    del tmp114
    tmp21 = einsum(v.bbb.xov, (0, 1, 2), tmp7, (0,), (1, 2))
    tmp139 = einsum(v.bbb.xov, (0, 1, 2), v.baa.xov, (0, 3, 4), (3, 1, 4, 2))
    tmp23 = einsum(tmp7, (0,), v.baa.xov, (0, 1, 2), (1, 2))
    tmp53 = einsum(v.baa.xvv, (0, 1, 2), tmp7, (0,), (1, 2))
    tmp52 = einsum(v.baa.xov, (0, 1, 2), tmp51, (0, 1, 3), (2, 3))
    del tmp51
    tmp54 = einsum(tmp14, (0, 1), t1.aa, (0, 2), (2, 1))
    tmp95 = einsum(tmp0, (0, 1, 2), tmp0, (0, 3, 4), (1, 3, 2, 4))
    tmp66 = np.copy(f.aa.ov)
    tmp66 += tmp13 * -1
    tmp66 += tmp10
    tmp61 = einsum(v.baa.xoo, (0, 1, 2), tmp0, (0, 3, 4), (3, 1, 2, 4))
    tmp81 = einsum(tmp0, (0, 1, 2), tmp80, (0, 3, 4), (1, 2, 3, 4))
    del tmp80
    tmp79 = einsum(v.baa.xov, (0, 1, 2), tmp78, (0, 3, 4), (1, 3, 2, 4))
    tmp89 = einsum(tmp72, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp88 = einsum(v.baa.xov, (0, 1, 2), tmp87, (0, 3, 4), (1, 3, 2, 4))
    del tmp87
    tmp44 = einsum(v.baa.xov, (0, 1, 2), tmp43, (0, 3, 2), (1, 3))
    tmp25 = einsum(tmp7, (0,), v.baa.xoo, (0, 1, 2), (1, 2))
    tmp110 = np.copy(f.bb.oo)
    tmp110 += tmp109
    del tmp109
    tmp110 += np.transpose(tmp40, (1, 0))
    tmp113 = np.copy(tmp29)
    tmp113 += tmp33 * 2
    tmp195 = einsum(tmp19, (0, 1, 2), tmp38, (0, 3, 4), (1, 3, 2, 4))
    tmp185 = np.copy(tmp107) * 2
    tmp185 += tmp29 * -1
    tmp152 = einsum(v.bbb.xoo, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 3, 4, 2))
    tmp171 = np.copy(tmp38) * -1
    tmp171 += tmp126
    tmp167 = np.copy(tmp38)
    tmp167 += tmp126 * -1
    tmp165 = einsum(t1.bb, (0, 1), tmp164, (2, 1), (2, 0))
    del tmp164
    tmp183 = np.copy(tmp181) * -1
    del tmp181
    tmp183 += np.transpose(tmp182, (1, 0, 3, 2)) * -1
    del tmp182
    tmp175 = np.copy(np.transpose(tmp144, (1, 0, 3, 2))) * -1
    tmp175 += np.transpose(tmp174, (0, 1, 3, 2))
    del tmp174
    tmp197 = einsum(tmp168, (0, 1, 2), v.bbb.xov, (0, 3, 4), (3, 2, 1, 4))
    tmp192 = einsum(t1.bb, (0, 1), tmp189, (2, 3, 0, 4), (3, 2, 4, 1))
    tmp187 = np.copy(tmp107)
    del tmp107
    tmp187 += tmp29 * -0.5
    tmp162 = einsum(t1.bb, (0, 1), tmp160, (2, 3, 4, 0), (2, 3, 4, 1))
    tmp127 = np.copy(v.bbb.xov)
    tmp127 += tmp38
    tmp127 += tmp126 * -1
    tmp118 = np.copy(f.bb.vv) * -1
    tmp118 += tmp115
    del tmp115
    tmp118 += np.transpose(tmp116, (1, 0)) * -1
    del tmp116
    tmp118 += np.transpose(tmp117, (1, 0))
    del tmp117
    tmp200 = einsum(v.bbb.xvv, (0, 1, 2), v.bbb.xov, (0, 3, 4), (3, 1, 2, 4))
    tmp22 = np.copy(f.bb.ov)
    tmp22 += tmp20 * -1
    del tmp20
    tmp22 += tmp21
    del tmp21
    tmp28 = einsum(tmp19, (0, 1, 2), v.bbb.xov, (0, 3, 4), (1, 3, 2, 4))
    tmp39 = einsum(v.bbb.xov, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp158 = einsum(tmp38, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 3, 4, 2))
    tmp131 = np.copy(v.bbb.xvv)
    tmp131 += tmp130 * -1
    tmp124 = np.copy(v.baa.xvv)
    tmp124 += tmp71 * -1
    tmp128 = np.copy(v.baa.xvv)
    tmp128 += tmp71 * -1
    del tmp71
    tmp140 = einsum(t1.bb, (0, 1), tmp139, (2, 0, 3, 4), (2, 3, 1, 4))
    del tmp139
    tmp137 = np.copy(v.bbb.xvv)
    tmp137 += tmp130 * -1
    del tmp130
    tmp147 = einsum(v.bbb.xvv, (0, 1, 2), v.baa.xov, (0, 3, 4), (3, 4, 1, 2))
    tmp76 = np.copy(tmp18) * -1
    tmp76 += tmp69
    tmp1 = einsum(tmp0, (0, 1, 2), v.baa.xov, (0, 3, 4), (1, 3, 2, 4))
    tmp92 = np.copy(tmp42) * 2
    tmp92 += tmp3 * -1
    tmp24 = np.copy(f.aa.ov)
    tmp24 += tmp13 * -1
    del tmp13
    tmp24 += tmp23
    del tmp23
    tmp55 = np.copy(f.aa.vv) * -1
    tmp55 += tmp52
    del tmp52
    tmp55 += np.transpose(tmp53, (1, 0)) * -1
    del tmp53
    tmp55 += np.transpose(tmp54, (1, 0))
    del tmp54
    tmp97 = einsum(tmp95, (0, 1, 2, 3), t1.aa, (2, 4), (1, 0, 3, 4))
    tmp67 = einsum(tmp66, (0, 1), t1.aa, (2, 1), (2, 0))
    del tmp66
    tmp70 = np.copy(tmp18)
    tmp70 += tmp69 * -1
    tmp16 = einsum(v.baa.xov, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp105 = einsum(v.baa.xvv, (0, 1, 2), v.baa.xov, (0, 3, 4), (3, 1, 2, 4))
    tmp63 = einsum(tmp61, (0, 1, 2, 3), t1.aa, (3, 4), (0, 1, 2, 4))
    tmp82 = np.copy(np.transpose(tmp79, (1, 0, 3, 2))) * -1
    tmp82 += np.transpose(tmp81, (0, 1, 3, 2))
    del tmp81
    tmp100 = einsum(tmp0, (0, 1, 2), tmp18, (0, 3, 4), (1, 3, 2, 4))
    tmp74 = np.copy(v.baa.xov)
    tmp74 += tmp18
    tmp74 += tmp69 * -1
    del tmp69
    tmp47 = einsum(v.baa.xoo, (0, 1, 2), v.baa.xoo, (0, 3, 4), (1, 3, 4, 2))
    tmp59 = einsum(v.baa.xoo, (0, 1, 2), tmp18, (0, 3, 4), (3, 1, 2, 4))
    tmp90 = np.copy(np.transpose(tmp88, (1, 0, 3, 2))) * -1
    del tmp88
    tmp90 += np.transpose(tmp89, (1, 0, 3, 2)) * -1
    del tmp89
    tmp45 = np.copy(f.aa.oo)
    tmp45 += tmp44
    del tmp44
    tmp45 += np.transpose(tmp25, (1, 0))
    tmp102 = einsum(v.baa.xov, (0, 1, 2), tmp72, (0, 3, 4), (1, 4, 3, 2))
    tmp57 = np.copy(tmp11) * 2
    tmp57 += tmp3
    tmp30 = np.copy(t2.bbbb)
    tmp30 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -0.5
    tmp34 = np.copy(tmp29)
    tmp34 += tmp33 * 2
    tmp34 += einsum(tmp7, (0,), t1.bb, (1, 2), (0, 1, 2))
    tmp4 = np.copy(t2.aaaa)
    tmp4 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (2, 0, 1, 3)) * -0.5
    tmp12 = np.copy(tmp11)
    tmp12 += tmp3 * 0.5
    tmp12 += einsum(tmp7, (0,), t1.aa, (1, 2), (0, 1, 2)) * 0.5
    tmp151 = einsum(tmp110, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -1
    tmp157 = einsum(v.bbb.xov, (0, 1, 2), tmp113, (0, 3, 4), (1, 3, 2, 4))
    tmp196 = einsum(tmp195, (0, 1, 2, 3), t1.bb, (2, 4), (0, 1, 4, 3))
    del tmp195
    tmp186 = einsum(tmp29, (0, 1, 2), tmp185, (0, 3, 4), (3, 1, 4, 2)) * 0.5
    del tmp185
    tmp154 = einsum(tmp152, (0, 1, 2, 3), t2.bbbb, (1, 3, 4, 5), (2, 0, 4, 5))
    tmp172 = einsum(tmp29, (0, 1, 2), tmp171, (0, 3, 4), (3, 1, 4, 2))
    del tmp171
    tmp169 = einsum(tmp167, (0, 1, 2), v.bbb.xov, (0, 3, 4), (1, 3, 2, 4)) * -1
    del tmp167
    tmp169 += einsum(tmp168, (0, 1, 2), tmp19, (0, 3, 4), (3, 4, 2, 1))
    del tmp168
    tmp161 = einsum(t2.bbbb, (0, 1, 2, 3), tmp160, (4, 0, 5, 1), (4, 5, 2, 3))
    del tmp160
    tmp166 = einsum(t2.bbbb, (0, 1, 2, 3), tmp165, (1, 4), (4, 0, 2, 3)) * -1
    del tmp165
    tmp184 = einsum(tmp183, (0, 1, 2, 3), t2.bbbb, (4, 1, 5, 3), (0, 4, 2, 5)) * 2
    del tmp183
    tmp194 = einsum(tmp38, (0, 1, 2), tmp38, (0, 3, 4), (1, 3, 2, 4))
    tmp176 = einsum(t2.bbbb, (0, 1, 2, 3), tmp175, (4, 1, 5, 3), (4, 0, 5, 2))
    del tmp175
    tmp190 = einsum(t2.bbbb, (0, 1, 2, 3), tmp189, (4, 5, 1, 0), (5, 4, 2, 3))
    del tmp189
    tmp198 = einsum(v.bbb.xvv, (0, 1, 2), v.bbb.xvv, (0, 3, 4), (3, 1, 2, 4))
    tmp198 += einsum(t1.bb, (0, 1), tmp197, (0, 2, 3, 4), (4, 3, 1, 2))
    del tmp197
    tmp193 = einsum(tmp192, (0, 1, 2, 3), t1.bb, (2, 4), (0, 1, 4, 3))
    del tmp192
    tmp155 = einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * 2
    tmp155 += t2.bbbb * -1
    tmp188 = einsum(tmp29, (0, 1, 2), tmp187, (0, 3, 4), (3, 1, 4, 2))
    del tmp187
    tmp191 = einsum(v.bbb.xov, (0, 1, 2), tmp38, (0, 3, 4), (3, 1, 2, 4))
    tmp163 = einsum(tmp162, (0, 1, 2, 3), t1.bb, (1, 4), (0, 2, 4, 3))
    del tmp162
    tmp170 = einsum(v.bbb.xov, (0, 1, 2), tmp127, (0, 3, 4), (1, 3, 2, 4))
    tmp156 = einsum(tmp118, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 4, 1)) * -2
    tmp201 = einsum(tmp200, (0, 1, 2, 3), t2.bbbb, (4, 5, 2, 3), (4, 5, 0, 1)) * -1
    del tmp200
    tmp119 = einsum(t1.bb, (0, 1), tmp22, (2, 1), (0, 2))
    tmp153 = np.copy(t2.bbbb)
    tmp153 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (2, 0, 3, 1)) * 2
    tmp179 = einsum(tmp33, (0, 1, 2), tmp29, (0, 3, 4), (3, 1, 4, 2))
    tmp177 = einsum(tmp29, (0, 1, 2), tmp143, (0, 3, 4), (1, 3, 2, 4))
    tmp199 = einsum(t1.bb, (0, 1), tmp28, (2, 3, 0, 4), (2, 3, 1, 4))
    tmp178 = einsum(tmp39, (0, 1, 2, 3), t1.bb, (1, 4), (0, 2, 4, 3))
    tmp159 = einsum(tmp158, (0, 1, 2, 3), t1.bb, (1, 4), (0, 2, 4, 3))
    del tmp158
    tmp37 = einsum(v.baa.xov, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 3, 4, 2))
    tmp134 = einsum(tmp0, (0, 1, 2), tmp19, (0, 3, 4), (1, 2, 3, 4))
    tmp17 = einsum(v.bbb.xov, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp150 = einsum(tmp0, (0, 1, 2), tmp38, (0, 3, 4), (1, 2, 3, 4))
    tmp146 = einsum(tmp0, (0, 1, 2), tmp131, (0, 3, 4), (1, 2, 4, 3))
    tmp145 = np.copy(np.transpose(tmp144, (1, 0, 2, 3))) * -1
    del tmp144
    tmp145 += einsum(tmp131, (0, 1, 2), tmp19, (0, 3, 4), (3, 4, 2, 1))
    tmp136 = np.copy(tmp42) * -2
    del tmp42
    tmp136 += tmp11 * 2
    tmp136 += tmp3
    tmp149 = einsum(tmp38, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp133 = np.copy(v.bbb.xov)
    tmp133 += tmp38
    tmp133 += tmp126 * -1
    del tmp126
    tmp125 = einsum(tmp124, (0, 1, 2), tmp19, (0, 3, 4), (3, 4, 2, 1))
    tmp111 = einsum(v.bbb.xoo, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp2 = einsum(tmp0, (0, 1, 2), v.bbb.xov, (0, 3, 4), (1, 2, 3, 4))
    tmp112 = np.copy(t2.abab)
    tmp112 += einsum(t1.bb, (0, 1), t1.aa, (2, 3), (2, 0, 3, 1)) * 2
    tmp120 = einsum(v.baa.xoo, (0, 1, 2), tmp19, (0, 3, 4), (1, 2, 3, 4))
    tmp142 = np.copy(np.transpose(tmp79, (1, 0, 2, 3))) * -1
    del tmp79
    tmp142 += einsum(tmp0, (0, 1, 2), tmp128, (0, 3, 4), (1, 2, 4, 3))
    tmp141 = einsum(tmp140, (0, 1, 2, 3), t1.aa, (0, 4), (1, 4, 3, 2))
    del tmp140
    tmp141 += einsum(tmp137, (0, 1, 2), v.baa.xvv, (0, 3, 4), (3, 4, 2, 1))
    tmp132 = einsum(tmp131, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 2, 1))
    del tmp131
    tmp148 = einsum(t2.abab, (0, 1, 2, 3), tmp147, (4, 2, 5, 3), (0, 4, 1, 5))
    del tmp147
    tmp122 = einsum(v.bbb.xoo, (0, 1, 2), tmp18, (0, 3, 4), (3, 1, 2, 4))
    tmp121 = np.copy(t2.abab)
    tmp121 += einsum(t1.bb, (0, 1), t1.aa, (2, 3), (2, 0, 3, 1))
    tmp129 = einsum(v.bbb.xoo, (0, 1, 2), tmp128, (0, 3, 4), (1, 2, 4, 3))
    del tmp128
    tmp123 = einsum(tmp0, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 2, 3, 4))
    tmp138 = einsum(v.bbb.xov, (0, 1, 2), tmp108, (0, 3, 4), (1, 3, 2, 4))
    tmp138 += einsum(v.bbb.xoo, (0, 1, 2), tmp137, (0, 3, 4), (1, 2, 4, 3)) * -1
    del tmp137
    tmp135 = einsum(v.baa.xov, (0, 1, 2), tmp43, (0, 3, 4), (1, 3, 2, 4))
    del tmp43
    tmp135 += einsum(tmp124, (0, 1, 2), v.baa.xoo, (0, 3, 4), (3, 4, 2, 1)) * -1
    del tmp124
    tmp77 = einsum(tmp76, (0, 1, 2), tmp3, (0, 3, 4), (3, 1, 4, 2))
    del tmp76
    tmp104 = einsum(t1.aa, (0, 1), tmp1, (2, 3, 0, 4), (2, 3, 1, 4))
    tmp93 = einsum(tmp3, (0, 1, 2), tmp92, (0, 3, 4), (1, 3, 2, 4)) * 0.5
    del tmp92
    tmp84 = einsum(tmp78, (0, 1, 2), tmp3, (0, 3, 4), (3, 1, 4, 2))
    tmp50 = einsum(t1.aa, (0, 1), t1.aa, (2, 3), (2, 0, 1, 3)) * 2
    tmp50 += t2.aaaa * -1
    tmp86 = einsum(tmp11, (0, 1, 2), tmp3, (0, 3, 4), (1, 3, 2, 4))
    tmp99 = einsum(tmp18, (0, 1, 2), tmp18, (0, 3, 4), (3, 1, 4, 2))
    tmp96 = einsum(tmp95, (0, 1, 2, 3), t2.aaaa, (3, 2, 4, 5), (1, 0, 4, 5))
    del tmp95
    tmp62 = einsum(t2.aaaa, (0, 1, 2, 3), tmp61, (4, 0, 5, 1), (4, 5, 2, 3))
    del tmp61
    tmp65 = einsum(tmp24, (0, 1), t1.aa, (2, 1), (2, 0))
    tmp56 = einsum(tmp55, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -2
    tmp48 = np.copy(t2.aaaa)
    tmp48 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (2, 0, 3, 1)) * 2
    tmp98 = einsum(t1.aa, (0, 1), tmp97, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp97
    tmp94 = einsum(tmp18, (0, 1, 2), v.baa.xov, (0, 3, 4), (1, 3, 4, 2))
    tmp68 = einsum(tmp67, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -1
    tmp73 = einsum(v.baa.xov, (0, 1, 2), tmp70, (0, 3, 4), (3, 1, 4, 2)) * -1
    del tmp70
    tmp73 += einsum(tmp0, (0, 1, 2), tmp72, (0, 3, 4), (1, 2, 4, 3))
    del tmp72
    tmp85 = einsum(t1.aa, (0, 1), tmp16, (2, 0, 3, 4), (2, 3, 1, 4))
    tmp106 = einsum(tmp105, (0, 1, 2, 3), t2.aaaa, (4, 5, 2, 3), (4, 5, 0, 1)) * -1
    del tmp105
    tmp64 = einsum(t1.aa, (0, 1), tmp63, (2, 0, 3, 4), (2, 3, 1, 4))
    del tmp63
    tmp83 = einsum(tmp82, (0, 1, 2, 3), t2.aaaa, (4, 1, 5, 3), (4, 0, 5, 2))
    del tmp82
    tmp101 = einsum(tmp100, (0, 1, 2, 3), t1.aa, (2, 4), (0, 1, 4, 3))
    del tmp100
    tmp75 = einsum(v.baa.xov, (0, 1, 2), tmp74, (0, 3, 4), (1, 3, 2, 4))
    del tmp74
    tmp49 = einsum(t2.aaaa, (0, 1, 2, 3), tmp47, (4, 0, 5, 1), (5, 4, 2, 3))
    tmp60 = einsum(t1.aa, (0, 1), tmp59, (2, 0, 3, 4), (2, 3, 1, 4))
    del tmp59
    tmp91 = einsum(t2.aaaa, (0, 1, 2, 3), tmp90, (4, 1, 5, 3), (0, 4, 2, 5)) * 2
    del tmp90
    tmp46 = einsum(t2.aaaa, (0, 1, 2, 3), tmp45, (1, 4), (0, 4, 2, 3)) * -1
    tmp103 = einsum(v.baa.xvv, (0, 1, 2), v.baa.xvv, (0, 3, 4), (3, 1, 2, 4))
    tmp103 += einsum(t1.aa, (0, 1), tmp102, (0, 2, 3, 4), (4, 3, 1, 2))
    del tmp102
    tmp58 = einsum(v.baa.xov, (0, 1, 2), tmp57, (0, 3, 4), (1, 3, 2, 4))
    tmp31 = np.copy(tmp29)
    del tmp29
    tmp31 += einsum(tmp30, (0, 1, 2, 3), v.bbb.xov, (4, 0, 2), (4, 1, 3)) * 2
    del tmp30
    tmp31 += einsum(tmp7, (0,), t1.bb, (1, 2), (0, 1, 2))
    tmp36 = einsum(v.bbb.xov, (0, 1, 2), tmp34, (0, 3, 2), (1, 3))
    del tmp34
    tmp36 += einsum(t1.bb, (0, 1), tmp35, (2, 1), (2, 0))
    del tmp35
    tmp27 = einsum(v.baa.xov, (0, 1, 2), tmp19, (0, 3, 4), (1, 3, 4, 2))
    tmp41 = np.copy(f.bb.oo)
    tmp41 += einsum(tmp19, (0, 1, 2), v.bbb.xoo, (0, 3, 1), (2, 3)) * -1
    del tmp19
    tmp41 += np.transpose(tmp40, (1, 0))
    del tmp40
    tmp8 = np.copy(tmp3)
    del tmp3
    tmp8 += einsum(v.baa.xov, (0, 1, 2), tmp4, (1, 3, 2, 4), (0, 3, 4)) * 2
    del tmp4
    tmp8 += einsum(tmp7, (0,), t1.aa, (1, 2), (0, 1, 2))
    del tmp7
    tmp26 = np.copy(f.aa.oo)
    tmp26 += einsum(v.baa.xoo, (0, 1, 2), tmp0, (0, 2, 3), (3, 1)) * -1
    del tmp0
    tmp26 += np.transpose(tmp25, (1, 0))
    del tmp25
    tmp15 = einsum(v.baa.xov, (0, 1, 2), tmp12, (0, 3, 2), (1, 3)) * 2
    del tmp12
    tmp15 += einsum(t1.aa, (0, 1), tmp14, (2, 1), (2, 0))
    del tmp14
    t2new.bbbb = np.copy(np.transpose(tmp151, (0, 1, 3, 2))) * -2
    t2new.bbbb += einsum(tmp153, (0, 1, 2, 3), tmp152, (0, 4, 1, 5), (4, 5, 3, 2)) * 0.5
    del tmp153
    t2new.bbbb += np.transpose(tmp154, (0, 1, 3, 2)) * -1
    del tmp154
    t2new.bbbb += np.transpose(tmp151, (1, 0, 3, 2)) * 2
    del tmp151
    t2new.bbbb += einsum(tmp155, (0, 1, 2, 3), tmp152, (0, 4, 1, 5), (4, 5, 3, 2)) * -0.5
    del tmp152, tmp155
    t2new.bbbb += np.transpose(tmp156, (1, 0, 3, 2))
    t2new.bbbb += tmp157
    t2new.bbbb += np.transpose(tmp156, (1, 0, 2, 3)) * -1
    del tmp156
    t2new.bbbb += np.transpose(tmp157, (0, 1, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp159, (1, 0, 3, 2))
    t2new.bbbb += np.transpose(tmp161, (1, 0, 3, 2)) * -2
    t2new.bbbb += np.transpose(tmp163, (1, 0, 2, 3))
    t2new.bbbb += np.transpose(tmp163, (1, 0, 3, 2)) * -1
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), tmp119, (4, 1), (0, 4, 2, 3)) * -1
    t2new.bbbb += np.transpose(tmp159, (1, 0, 2, 3)) * -1
    t2new.bbbb += np.transpose(tmp166, (1, 0, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp159, (0, 1, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp161, (0, 1, 3, 2)) * 2
    del tmp161
    t2new.bbbb += tmp163 * -1
    t2new.bbbb += np.transpose(tmp163, (0, 1, 3, 2))
    del tmp163
    t2new.bbbb += np.transpose(tmp166, (0, 1, 3, 2)) * 2
    del tmp166
    t2new.bbbb += tmp159
    del tmp159
    t2new.bbbb += einsum(tmp169, (0, 1, 2, 3), t2.bbbb, (4, 1, 5, 3), (4, 0, 2, 5)) * 2
    del tmp169
    t2new.bbbb += tmp170
    t2new.bbbb += np.transpose(tmp172, (1, 0, 2, 3))
    t2new.bbbb += tmp176 * -2
    t2new.bbbb += np.transpose(tmp170, (1, 0, 2, 3)) * -1
    del tmp170
    t2new.bbbb += np.transpose(tmp177, (1, 0, 3, 2))
    t2new.bbbb += np.transpose(tmp178, (0, 1, 3, 2))
    t2new.bbbb += tmp179 * 2
    t2new.bbbb += np.transpose(tmp179, (1, 0, 3, 2)) * 2
    t2new.bbbb += tmp184
    t2new.bbbb += tmp186 * -1
    t2new.bbbb += np.transpose(tmp178, (1, 0, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp179, (0, 1, 3, 2)) * -2
    t2new.bbbb += np.transpose(tmp179, (1, 0, 2, 3)) * -2
    del tmp179
    t2new.bbbb += np.transpose(tmp184, (1, 0, 2, 3)) * -1
    t2new.bbbb += np.transpose(tmp186, (1, 0, 2, 3))
    del tmp186
    t2new.bbbb += tmp178 * -1
    t2new.bbbb += np.transpose(tmp184, (0, 1, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp188, (0, 1, 3, 2))
    t2new.bbbb += np.transpose(tmp178, (1, 0, 2, 3))
    del tmp178
    t2new.bbbb += np.transpose(tmp184, (1, 0, 3, 2))
    del tmp184
    t2new.bbbb += np.transpose(tmp188, (1, 0, 3, 2)) * -1
    del tmp188
    t2new.bbbb += np.transpose(tmp190, (0, 1, 3, 2)) * -2
    del tmp190
    t2new.bbbb += np.transpose(tmp191, (0, 1, 3, 2))
    t2new.bbbb += tmp193
    t2new.bbbb += np.transpose(tmp193, (0, 1, 3, 2)) * -1
    del tmp193
    t2new.bbbb += tmp194
    t2new.bbbb += np.transpose(tmp194, (0, 1, 3, 2)) * -1
    del tmp194
    t2new.bbbb += np.transpose(tmp196, (0, 1, 3, 2))
    t2new.bbbb += np.transpose(tmp196, (1, 0, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp191, (1, 0, 3, 2)) * -1
    del tmp191
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), tmp198, (2, 3, 4, 5), (0, 1, 5, 4)) * 2
    del tmp198
    t2new.bbbb += np.transpose(tmp157, (1, 0, 2, 3)) * -1
    t2new.bbbb += np.transpose(tmp157, (1, 0, 3, 2))
    del tmp157
    t2new.bbbb += np.transpose(tmp199, (1, 0, 2, 3))
    t2new.bbbb += np.transpose(tmp176, (1, 0, 3, 2)) * -2
    t2new.bbbb += np.transpose(tmp172, (1, 0, 3, 2)) * -1
    del tmp172
    t2new.bbbb += tmp199 * -1
    del tmp199
    t2new.bbbb += np.transpose(tmp176, (0, 1, 3, 2)) * 2
    del tmp176
    t2new.bbbb += np.transpose(tmp177, (1, 0, 2, 3)) * -1
    del tmp177
    t2new.bbbb += einsum(tmp201, (0, 1, 2, 3), t1.bb, (2, 4), (0, 1, 4, 3)) * -2
    del tmp201
    t2new.bbbb += tmp196 * -1
    t2new.bbbb += np.transpose(tmp196, (1, 0, 2, 3))
    del tmp196
    t2new.abab = einsum(t2.abab, (0, 1, 2, 3), tmp110, (1, 4), (0, 4, 2, 3)) * -1
    del tmp110
    t2new.abab += einsum(tmp112, (0, 1, 2, 3), tmp111, (0, 4, 1, 5), (4, 5, 2, 3)) * 0.5
    del tmp112
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp111, (4, 0, 5, 1), (4, 5, 2, 3)) * 0.5
    del tmp111
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp45, (0, 4), (4, 1, 2, 3)) * -1
    del tmp45
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp55, (2, 4), (0, 1, 4, 3)) * -1
    del tmp55
    t2new.abab += einsum(v.baa.xov, (0, 1, 2), tmp113, (0, 3, 4), (1, 3, 2, 4))
    del tmp113
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp118, (3, 4), (0, 1, 2, 4)) * -1
    del tmp118
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp119, (4, 1), (0, 4, 2, 3)) * -1
    del tmp119
    t2new.abab += einsum(tmp121, (0, 1, 2, 3), tmp120, (4, 0, 5, 1), (4, 5, 2, 3))
    del tmp120
    t2new.abab += einsum(t1.bb, (0, 1), tmp122, (2, 0, 3, 4), (2, 3, 4, 1)) * -1
    del tmp122
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp67, (4, 0), (4, 1, 2, 3)) * -1
    del tmp67
    t2new.abab += einsum(tmp123, (0, 1, 2, 3), tmp121, (1, 2, 4, 5), (0, 3, 4, 5))
    del tmp123
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp125, (4, 1, 2, 5), (0, 4, 5, 3)) * -1
    del tmp125
    t2new.abab += einsum(v.baa.xov, (0, 1, 2), tmp127, (0, 3, 4), (1, 3, 2, 4))
    del tmp127
    t2new.abab += einsum(tmp37, (0, 1, 2, 3), t1.bb, (2, 4), (0, 1, 3, 4)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp129, (1, 4, 2, 5), (0, 4, 5, 3)) * -1
    del tmp129
    t2new.abab += einsum(t1.aa, (0, 1), tmp17, (2, 0, 3, 4), (2, 3, 1, 4)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp132, (0, 4, 3, 5), (4, 1, 2, 5)) * -1
    del tmp132
    t2new.abab += einsum(tmp133, (0, 1, 2), tmp18, (0, 3, 4), (3, 1, 4, 2))
    del tmp133
    t2new.abab += einsum(tmp134, (0, 1, 2, 3), tmp121, (1, 3, 4, 5), (0, 2, 4, 5))
    del tmp121, tmp134
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp135, (0, 4, 2, 5), (4, 1, 5, 3))
    del tmp135
    t2new.abab += einsum(tmp136, (0, 1, 2), tmp33, (0, 3, 4), (1, 3, 2, 4))
    del tmp136
    t2new.abab += einsum(v.bbb.xov, (0, 1, 2), tmp57, (0, 3, 4), (3, 1, 4, 2))
    del tmp57
    t2new.abab += einsum(tmp138, (0, 1, 2, 3), t2.abab, (4, 0, 5, 2), (4, 1, 5, 3))
    del tmp138
    t2new.abab += einsum(tmp108, (0, 1, 2), tmp11, (0, 3, 4), (3, 1, 4, 2)) * 2
    del tmp108
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp141, (2, 4, 3, 5), (0, 1, 4, 5))
    del tmp141
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp142, (4, 0, 2, 5), (4, 1, 5, 3)) * -1
    del tmp142
    t2new.abab += einsum(tmp33, (0, 1, 2), tmp78, (0, 3, 4), (3, 1, 4, 2)) * 2
    del tmp33, tmp78
    t2new.abab += einsum(tmp145, (0, 1, 2, 3), t2.abab, (4, 1, 5, 2), (4, 0, 5, 3)) * -1
    del tmp145
    t2new.abab += einsum(tmp11, (0, 1, 2), tmp143, (0, 3, 4), (1, 3, 2, 4)) * 2
    del tmp11, tmp143
    t2new.abab += einsum(tmp2, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp146, (4, 0, 3, 5), (4, 1, 2, 5)) * -1
    del tmp146
    t2new.abab += einsum(t1.aa, (0, 1), tmp148, (2, 0, 3, 4), (2, 3, 1, 4)) * -1
    del tmp148
    t2new.abab += einsum(tmp149, (0, 1, 2, 3), t1.aa, (0, 4), (1, 2, 4, 3)) * -1
    del tmp149
    t2new.abab += einsum(t1.aa, (0, 1), tmp150, (2, 0, 3, 4), (2, 3, 1, 4)) * -1
    del tmp150
    t2new.aaaa = np.copy(np.transpose(tmp46, (0, 1, 3, 2))) * -2
    t2new.aaaa += einsum(tmp47, (0, 1, 2, 3), tmp48, (0, 2, 4, 5), (1, 3, 5, 4)) * 0.5
    del tmp48
    t2new.aaaa += np.transpose(tmp49, (0, 1, 3, 2)) * -1
    del tmp49
    t2new.aaaa += np.transpose(tmp46, (1, 0, 3, 2)) * 2
    del tmp46
    t2new.aaaa += einsum(tmp50, (0, 1, 2, 3), tmp47, (0, 4, 1, 5), (4, 5, 3, 2)) * -0.5
    del tmp47, tmp50
    t2new.aaaa += np.transpose(tmp56, (1, 0, 3, 2))
    t2new.aaaa += tmp58
    t2new.aaaa += np.transpose(tmp56, (1, 0, 2, 3)) * -1
    del tmp56
    t2new.aaaa += np.transpose(tmp58, (0, 1, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp60, (1, 0, 3, 2))
    t2new.aaaa += np.transpose(tmp62, (1, 0, 3, 2)) * -2
    t2new.aaaa += np.transpose(tmp64, (1, 0, 2, 3))
    t2new.aaaa += np.transpose(tmp64, (1, 0, 3, 2)) * -1
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), tmp65, (4, 1), (0, 4, 2, 3)) * -1
    del tmp65
    t2new.aaaa += np.transpose(tmp60, (1, 0, 2, 3)) * -1
    t2new.aaaa += np.transpose(tmp68, (0, 1, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp60, (0, 1, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp62, (0, 1, 3, 2)) * 2
    del tmp62
    t2new.aaaa += tmp64 * -1
    t2new.aaaa += np.transpose(tmp64, (0, 1, 3, 2))
    del tmp64
    t2new.aaaa += np.transpose(tmp68, (1, 0, 3, 2)) * 2
    del tmp68
    t2new.aaaa += tmp60
    del tmp60
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), tmp73, (4, 1, 5, 3), (0, 4, 5, 2)) * 2
    del tmp73
    t2new.aaaa += tmp75
    t2new.aaaa += np.transpose(tmp77, (0, 1, 3, 2))
    t2new.aaaa += np.transpose(tmp83, (1, 0, 3, 2)) * -2
    t2new.aaaa += np.transpose(tmp75, (1, 0, 2, 3)) * -1
    del tmp75
    t2new.aaaa += np.transpose(tmp84, (1, 0, 3, 2))
    t2new.aaaa += np.transpose(tmp85, (0, 1, 3, 2))
    t2new.aaaa += tmp86 * 2
    t2new.aaaa += np.transpose(tmp86, (1, 0, 3, 2)) * 2
    t2new.aaaa += np.transpose(tmp91, (1, 0, 3, 2))
    t2new.aaaa += np.transpose(tmp93, (1, 0, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp85, (1, 0, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp86, (0, 1, 3, 2)) * -2
    t2new.aaaa += np.transpose(tmp86, (1, 0, 2, 3)) * -2
    del tmp86
    t2new.aaaa += np.transpose(tmp91, (0, 1, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp93, (0, 1, 3, 2))
    t2new.aaaa += tmp85 * -1
    t2new.aaaa += np.transpose(tmp91, (1, 0, 2, 3)) * -1
    t2new.aaaa += np.transpose(tmp93, (1, 0, 2, 3))
    t2new.aaaa += np.transpose(tmp85, (1, 0, 2, 3))
    del tmp85
    t2new.aaaa += tmp91
    del tmp91
    t2new.aaaa += tmp93 * -1
    del tmp93
    t2new.aaaa += np.transpose(tmp94, (0, 1, 3, 2))
    t2new.aaaa += np.transpose(tmp96, (0, 1, 3, 2)) * -2
    del tmp96
    t2new.aaaa += tmp98
    t2new.aaaa += np.transpose(tmp98, (0, 1, 3, 2)) * -1
    del tmp98
    t2new.aaaa += tmp99
    t2new.aaaa += np.transpose(tmp99, (0, 1, 3, 2)) * -1
    del tmp99
    t2new.aaaa += np.transpose(tmp101, (0, 1, 3, 2))
    t2new.aaaa += np.transpose(tmp101, (1, 0, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp94, (1, 0, 3, 2)) * -1
    del tmp94
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), tmp103, (2, 3, 4, 5), (0, 1, 5, 4)) * 2
    del tmp103
    t2new.aaaa += np.transpose(tmp58, (1, 0, 2, 3)) * -1
    t2new.aaaa += np.transpose(tmp58, (1, 0, 3, 2))
    del tmp58
    t2new.aaaa += np.transpose(tmp104, (1, 0, 2, 3))
    t2new.aaaa += tmp83 * -2
    t2new.aaaa += tmp77 * -1
    del tmp77
    t2new.aaaa += tmp104 * -1
    del tmp104
    t2new.aaaa += np.transpose(tmp83, (1, 0, 2, 3)) * 2
    del tmp83
    t2new.aaaa += np.transpose(tmp84, (1, 0, 2, 3)) * -1
    del tmp84
    t2new.aaaa += einsum(tmp106, (0, 1, 2, 3), t1.aa, (2, 4), (0, 1, 4, 3)) * -2
    del tmp106
    t2new.aaaa += tmp101 * -1
    t2new.aaaa += np.transpose(tmp101, (1, 0, 2, 3))
    del tmp101
    t1new.bb = einsum(tmp27, (0, 1, 2, 3), t2.abab, (0, 2, 3, 4), (1, 4)) * -1
    del tmp27
    t1new.bb += einsum(t2.bbbb, (0, 1, 2, 3), tmp28, (4, 1, 0, 3), (4, 2)) * -2
    del tmp28
    t1new.bb += f.bb.ov
    t1new.bb += einsum(t1.bb, (0, 1), f.bb.vv, (2, 1), (0, 2))
    t1new.bb += einsum(tmp31, (0, 1, 2), v.bbb.xvv, (0, 3, 2), (1, 3))
    del tmp31
    t1new.bb += tmp32
    del tmp32
    t1new.bb += einsum(t1.bb, (0, 1), tmp36, (0, 2), (2, 1)) * -1
    del tmp36
    t1new.bb += einsum(t2.abab, (0, 1, 2, 3), tmp37, (0, 4, 1, 2), (4, 3)) * -1
    del tmp37
    t1new.bb += einsum(v.bbb.xoo, (0, 1, 2), tmp38, (0, 2, 3), (1, 3)) * -1
    del tmp38
    t1new.bb += einsum(t2.bbbb, (0, 1, 2, 3), tmp39, (4, 0, 1, 3), (4, 2)) * -2
    del tmp39
    t1new.bb += einsum(tmp24, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    t1new.bb += einsum(tmp22, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2
    t1new.bb += einsum(tmp41, (0, 1), t1.bb, (0, 2), (1, 2)) * -1
    del tmp41
    t1new.aa = einsum(f.aa.vv, (0, 1), t1.aa, (2, 1), (2, 0))
    t1new.aa += f.aa.ov
    t1new.aa += einsum(t2.aaaa, (0, 1, 2, 3), tmp1, (4, 1, 0, 3), (4, 2)) * -2
    del tmp1
    t1new.aa += einsum(tmp2, (0, 1, 2, 3), t2.abab, (1, 2, 4, 3), (0, 4)) * -1
    del tmp2
    t1new.aa += einsum(v.baa.xvv, (0, 1, 2), tmp8, (0, 3, 2), (3, 1))
    del tmp8
    t1new.aa += tmp10
    del tmp10
    t1new.aa += einsum(t1.aa, (0, 1), tmp15, (0, 2), (2, 1)) * -1
    del tmp15
    t1new.aa += einsum(t2.aaaa, (0, 1, 2, 3), tmp16, (4, 0, 1, 3), (4, 2)) * -2
    del tmp16
    t1new.aa += einsum(t2.abab, (0, 1, 2, 3), tmp17, (4, 0, 1, 3), (4, 2)) * -1
    del tmp17
    t1new.aa += einsum(tmp18, (0, 1, 2), v.baa.xoo, (0, 3, 1), (3, 2)) * -1
    del tmp18
    t1new.aa += einsum(tmp22, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    del tmp22
    t1new.aa += einsum(t2.aaaa, (0, 1, 2, 3), tmp24, (1, 3), (0, 2)) * 2
    del tmp24
    t1new.aa += einsum(t1.aa, (0, 1), tmp26, (0, 2), (2, 1)) * -1
    del tmp26

    return {f"t1new": t1new, f"t2new": t2new}

