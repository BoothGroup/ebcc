"""Code generated by `albert` version 0.0.0.

 * date: 2024-12-19T15:11:04.994641
 * python version: 3.10.15 (main, Sep  9 2024, 03:03:06) [GCC 13.2.0]
 * albert version: 0.0.0
 * caller: /opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages/albert/code/einsum.py
 * node: fv-az1676-657
 * system: Linux
 * processor: x86_64
 * release: 6.8.0-1017-azure
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, dirsum, Namespace


def energy(f=None, t1=None, t2=None, v=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        f: 
        t1: 
        t2: 
        v: 

    Returns:
        e_cc: 
    """

    e_cc = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 2, 1, 3), ())
    tmp1 = np.copy(f.bb.ov)
    tmp1 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    tmp0 = np.copy(np.transpose(v.bbbb.ovov, (2, 0, 1, 3)))
    tmp0 += np.transpose(v.bbbb.ovov, (2, 0, 3, 1)) * -1
    tmp1 += einsum(t1.bb, (0, 1), tmp0, (0, 2, 1, 3), (2, 3)) * -0.5
    del tmp0
    e_cc += einsum(tmp1, (0, 1), t1.bb, (0, 1), ())
    del tmp1
    tmp3 = np.copy(f.aa.ov)
    tmp2 = np.copy(np.transpose(v.aaaa.ovov, (2, 0, 1, 3))) * -1
    tmp2 += np.transpose(v.aaaa.ovov, (2, 0, 3, 1))
    tmp3 += einsum(t1.aa, (0, 1), tmp2, (0, 2, 3, 1), (2, 3)) * -0.5
    del tmp2
    e_cc += einsum(tmp3, (0, 1), t1.aa, (0, 1), ())
    del tmp3

    return e_cc

def update_amps(f=None, t1=None, t2=None, v=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        f: 
        t1: 
        t2: 
        v: 

    Returns:
        t1new: 
        t2new: 
    """

    t1new = Namespace()
    t1new.aa = einsum(v.aabb.ooov, (0, 1, 2, 3), t2.abab, (1, 2, 4, 3), (0, 4)) * -1
    tmp0 = einsum(v.aabb.ovov, (0, 1, 2, 3), t1.bb, (2, 3), (0, 1))
    t1new.aa += tmp0
    t1new.aa += f.aa.ov
    t1new.aa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ooov, (4, 1, 0, 3), (4, 2)) * 2
    tmp4 = np.copy(f.bb.ov)
    tmp1 = einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    tmp4 += tmp1
    tmp2 = np.copy(np.transpose(v.bbbb.ovov, (0, 2, 3, 1)))
    tmp2 += np.transpose(v.bbbb.ovov, (0, 2, 1, 3)) * -1
    tmp3 = einsum(t1.bb, (0, 1), tmp2, (0, 2, 1, 3), (2, 3))
    del tmp2
    tmp4 += tmp3 * -1
    del tmp3
    t1new.aa += einsum(t2.abab, (0, 1, 2, 3), tmp4, (1, 3), (0, 2))
    tmp7 = np.copy(f.aa.ov)
    tmp7 += tmp0
    del tmp0
    tmp5 = np.copy(np.transpose(v.aaaa.ovov, (0, 2, 3, 1))) * -1
    tmp5 += np.transpose(v.aaaa.ovov, (0, 2, 1, 3))
    tmp6 = einsum(tmp5, (0, 1, 2, 3), t1.aa, (0, 3), (1, 2))
    del tmp5
    tmp7 += tmp6 * -1
    del tmp6
    t1new.aa += einsum(t2.aaaa, (0, 1, 2, 3), tmp7, (1, 3), (0, 2)) * 2
    tmp8 = np.copy(np.transpose(v.aaaa.ovov, (0, 2, 1, 3)))
    tmp8 += v.aaaa.oovv * -1
    t1new.aa += einsum(tmp8, (0, 1, 2, 3), t1.aa, (0, 2), (1, 3))
    tmp12 = np.copy(f.aa.oo)
    tmp9 = einsum(v.aabb.ooov, (0, 1, 2, 3), t1.bb, (2, 3), (0, 1))
    tmp12 += np.transpose(tmp9, (1, 0))
    tmp10 = np.copy(v.aaaa.ooov) * -1
    tmp10 += np.transpose(v.aaaa.ooov, (0, 2, 1, 3))
    tmp11 = einsum(tmp10, (0, 1, 2, 3), t1.aa, (2, 3), (0, 1))
    del tmp10
    tmp12 += np.transpose(tmp11, (1, 0)) * -1
    t1new.aa += einsum(tmp12, (0, 1), t1.aa, (0, 2), (1, 2)) * -1
    del tmp12
    tmp13 = einsum(v.aabb.ovov, (0, 1, 2, 3), t1.aa, (4, 1), (4, 0, 2, 3))
    t1new.aa += einsum(tmp13, (0, 1, 2, 3), t2.abab, (1, 2, 4, 3), (0, 4)) * -1
    tmp14 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t1.aa, (4, 3), (4, 0, 2, 1))
    t1new.aa += einsum(tmp14, (0, 1, 2, 3), t2.aaaa, (1, 2, 4, 3), (0, 4)) * 2
    tmp15 = np.copy(t2.abab)
    tmp15 += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    t1new.aa += einsum(v.aabb.vvov, (0, 1, 2, 3), tmp15, (4, 2, 1, 3), (4, 0))
    tmp16 = np.copy(t2.aaaa) * 2
    tmp16 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (2, 0, 3, 1))
    t1new.aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), tmp16, (0, 4, 1, 3), (4, 2))
    del tmp16
    tmp17 = einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1
    tmp19 = np.copy(np.transpose(tmp17, (1, 0)))
    tmp18 = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    tmp19 += np.transpose(tmp18, (1, 0)) * 0.5
    tmp19 += einsum(t1.aa, (0, 1), tmp7, (2, 1), (2, 0)) * 0.5
    t1new.aa += einsum(tmp19, (0, 1), t1.aa, (0, 2), (1, 2)) * -2
    del tmp19
    tmp20 = np.copy(f.aa.vv)
    tmp20 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), t1.aa, (0, 3), (1, 2)) * -1
    t1new.aa += einsum(tmp20, (0, 1), t1.aa, (2, 0), (2, 1))
    del tmp20
    t1new.bb = np.copy(tmp1)
    del tmp1
    t1new.bb += f.bb.ov
    t1new.bb += einsum(v.bbbb.ooov, (0, 1, 2, 3), t2.bbbb, (2, 1, 4, 3), (0, 4)) * 2
    t1new.bb += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 1), (4, 3)) * -1
    t1new.bb += einsum(tmp4, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2
    t1new.bb += einsum(t2.abab, (0, 1, 2, 3), tmp7, (0, 2), (1, 3))
    tmp21 = np.copy(np.transpose(v.bbbb.ovov, (0, 2, 1, 3)))
    tmp21 += v.bbbb.oovv * -1
    t1new.bb += einsum(t1.bb, (0, 1), tmp21, (0, 2, 1, 3), (2, 3))
    tmp25 = np.copy(f.bb.oo)
    tmp22 = einsum(v.aabb.ovoo, (0, 1, 2, 3), t1.aa, (0, 1), (2, 3))
    tmp25 += np.transpose(tmp22, (1, 0))
    tmp23 = np.copy(v.bbbb.ooov) * -1
    tmp23 += np.transpose(v.bbbb.ooov, (0, 2, 1, 3))
    tmp24 = einsum(tmp23, (0, 1, 2, 3), t1.bb, (2, 3), (0, 1))
    tmp25 += np.transpose(tmp24, (1, 0)) * -1
    t1new.bb += einsum(t1.bb, (0, 1), tmp25, (0, 2), (2, 1)) * -1
    del tmp25
    tmp26 = einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    t1new.bb += einsum(tmp26, (0, 1, 2, 3), t2.bbbb, (1, 2, 4, 3), (0, 4)) * 2
    tmp27 = einsum(v.aabb.ovov, (0, 1, 2, 3), t1.bb, (4, 3), (0, 4, 2, 1))
    t1new.bb += einsum(t2.abab, (0, 1, 2, 3), tmp27, (0, 4, 1, 2), (4, 3)) * -1
    tmp28 = np.copy(t2.bbbb) * 2
    tmp28 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (2, 0, 3, 1))
    t1new.bb += einsum(v.bbbb.ovvv, (0, 1, 2, 3), tmp28, (0, 4, 3, 1), (4, 2)) * -1
    del tmp28
    t1new.bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), tmp15, (0, 4, 1, 3), (4, 2))
    tmp29 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 4, 1, 3), (4, 2))
    tmp32 = np.copy(np.transpose(tmp29, (1, 0)))
    tmp30 = einsum(v.bbbb.ovov, (0, 1, 2, 3), t2.bbbb, (4, 2, 1, 3), (4, 0))
    tmp32 += np.transpose(tmp30, (1, 0)) * 2
    tmp31 = einsum(tmp4, (0, 1), t1.bb, (2, 1), (2, 0))
    tmp32 += np.transpose(tmp31, (1, 0))
    t1new.bb += einsum(t1.bb, (0, 1), tmp32, (0, 2), (2, 1)) * -1
    del tmp32
    tmp33 = np.copy(f.bb.vv)
    tmp33 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (0, 1, 2, 3), (2, 3))
    t1new.bb += einsum(t1.bb, (0, 1), tmp33, (1, 2), (0, 2))
    del tmp33
    tmp34 = np.copy(f.aa.oo)
    tmp34 += np.transpose(tmp9, (1, 0))
    del tmp9
    tmp34 += np.transpose(tmp17, (1, 0))
    del tmp17
    tmp34 += np.transpose(tmp18, (1, 0)) * 0.5
    del tmp18
    tmp34 += np.transpose(tmp11, (1, 0)) * -1
    del tmp11
    tmp35 = einsum(tmp34, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2
    t2new = Namespace()
    t2new.aaaa = np.copy(np.transpose(tmp35, (0, 1, 3, 2))) * -1
    tmp36 = einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new.aaaa += np.transpose(tmp36, (0, 1, 3, 2))
    t2new.aaaa += einsum(v.aaaa.oooo, (0, 1, 2, 3), t2.aaaa, (3, 1, 4, 5), (0, 2, 4, 5)) * -2
    tmp37 = einsum(v.aaaa.oooo, (0, 1, 2, 3), t1.aa, (3, 4), (0, 1, 2, 4))
    tmp38 = einsum(t1.aa, (0, 1), tmp37, (2, 0, 3, 4), (3, 2, 4, 1))
    del tmp37
    t2new.aaaa += tmp38
    t2new.aaaa += np.transpose(tmp38, (0, 1, 3, 2)) * -1
    del tmp38
    t2new.aaaa += np.transpose(tmp35, (1, 0, 3, 2))
    del tmp35
    tmp45 = np.copy(f.aa.vv) * -1
    tmp39 = einsum(v.aabb.vvov, (0, 1, 2, 3), t1.bb, (2, 3), (0, 1))
    tmp45 += np.transpose(tmp39, (1, 0)) * -1
    del tmp39
    tmp40 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (0, 2, 4, 1), (4, 3)) * -1
    tmp45 += np.transpose(tmp40, (1, 0))
    del tmp40
    tmp41 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 2, 4, 3), (4, 1))
    tmp45 += np.transpose(tmp41, (1, 0)) * 0.5
    del tmp41
    tmp42 = np.copy(v.aaaa.ovvv) * -1
    tmp42 += np.transpose(v.aaaa.ovvv, (0, 2, 1, 3))
    tmp43 = einsum(tmp42, (0, 1, 2, 3), t1.aa, (0, 2), (1, 3))
    del tmp42
    tmp45 += tmp43 * -1
    del tmp43
    tmp44 = einsum(tmp7, (0, 1), t1.aa, (0, 2), (2, 1))
    tmp45 += np.transpose(tmp44, (1, 0))
    del tmp44
    tmp46 = einsum(t2.aaaa, (0, 1, 2, 3), tmp45, (3, 4), (0, 1, 2, 4)) * -2
    t2new.aaaa += np.transpose(tmp46, (1, 0, 3, 2))
    t2new.aaaa += np.transpose(tmp46, (1, 0, 2, 3)) * -1
    del tmp46
    tmp47 = einsum(v.aaaa.ooov, (0, 1, 2, 3), t1.aa, (4, 3), (4, 0, 1, 2))
    tmp48 = einsum(tmp47, (0, 1, 2, 3), t2.aaaa, (3, 2, 4, 5), (0, 1, 4, 5)) * -1
    t2new.aaaa += np.transpose(tmp48, (1, 0, 3, 2)) * -2
    tmp49 = einsum(v.aaaa.ovvv, (0, 1, 2, 3), t1.aa, (4, 3), (4, 0, 1, 2))
    t2new.aaaa += np.transpose(tmp49, (1, 0, 2, 3))
    tmp50 = einsum(t1.aa, (0, 1), tmp47, (2, 3, 4, 0), (2, 4, 3, 1))
    del tmp47
    tmp51 = einsum(tmp50, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3))
    del tmp50
    t2new.aaaa += np.transpose(tmp51, (1, 0, 2, 3))
    t2new.aaaa += np.transpose(tmp51, (1, 0, 3, 2)) * -1
    tmp52 = einsum(tmp7, (0, 1), t1.aa, (2, 1), (2, 0))
    del tmp7
    tmp53 = einsum(tmp52, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -2
    t2new.aaaa += np.transpose(tmp53, (0, 1, 3, 2)) * -1
    tmp54 = einsum(t1.aa, (0, 1), tmp8, (2, 3, 1, 4), (0, 2, 3, 4))
    del tmp8
    tmp55 = einsum(t1.aa, (0, 1), tmp54, (2, 0, 3, 4), (2, 3, 1, 4))
    del tmp54
    t2new.aaaa += np.transpose(tmp55, (1, 0, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp48, (0, 1, 3, 2)) * 2
    del tmp48
    t2new.aaaa += tmp49 * -1
    t2new.aaaa += tmp51 * -1
    t2new.aaaa += np.transpose(tmp51, (0, 1, 3, 2))
    del tmp51
    t2new.aaaa += np.transpose(tmp53, (1, 0, 3, 2))
    del tmp53
    t2new.aaaa += np.transpose(tmp55, (0, 1, 3, 2))
    t2new.aaaa += np.transpose(tmp36, (1, 0, 3, 2)) * -1
    t2new.aaaa += np.transpose(v.aaaa.ovov, (0, 2, 1, 3))
    tmp56 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (4, 2, 5, 3), (4, 0, 5, 1))
    tmp57 = einsum(t2.aaaa, (0, 1, 2, 3), tmp56, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new.aaaa += tmp57
    t2new.aaaa += np.transpose(tmp57, (1, 0, 3, 2))
    tmp61 = np.copy(np.transpose(v.aaaa.ovov, (0, 2, 1, 3)))
    tmp61 += v.aaaa.oovv * -1
    tmp58 = einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    tmp61 += tmp58
    tmp59 = np.copy(v.aaaa.ooov)
    tmp59 += np.transpose(v.aaaa.ovoo, (0, 2, 3, 1)) * -1
    tmp60 = einsum(tmp59, (0, 1, 2, 3), t1.aa, (0, 4), (1, 2, 4, 3))
    del tmp59
    tmp61 += tmp60 * -1
    del tmp60
    tmp62 = einsum(t2.aaaa, (0, 1, 2, 3), tmp61, (4, 1, 5, 3), (0, 4, 2, 5)) * 2
    del tmp61
    t2new.aaaa += np.transpose(tmp62, (1, 0, 3, 2))
    tmp65 = np.copy(np.transpose(v.aabb.ovov, (0, 2, 1, 3)))
    tmp63 = einsum(v.aabb.ooov, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3))
    tmp65 += tmp63 * -1
    tmp64 = einsum(t2.abab, (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    tmp65 += tmp64 * 0.5
    tmp66 = einsum(t2.abab, (0, 1, 2, 3), tmp65, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp65
    t2new.aaaa += np.transpose(tmp66, (1, 0, 3, 2))
    t2new.aaaa += np.transpose(tmp36, (1, 0, 2, 3))
    t2new.aaaa += np.transpose(v.aaaa.ovov, (0, 2, 3, 1)) * -1
    t2new.aaaa += np.transpose(tmp57, (0, 1, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp57, (1, 0, 2, 3)) * -1
    t2new.aaaa += np.transpose(tmp62, (1, 0, 2, 3)) * -1
    t2new.aaaa += np.transpose(tmp66, (1, 0, 2, 3)) * -1
    t2new.aaaa += tmp36 * -1
    del tmp36
    t2new.aaaa += np.transpose(tmp49, (0, 1, 3, 2))
    t2new.aaaa += tmp55 * -1
    t2new.aaaa += np.transpose(tmp49, (1, 0, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp55, (1, 0, 2, 3))
    del tmp55
    t2new.aaaa += np.transpose(tmp57, (0, 1, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp57, (1, 0, 2, 3)) * -1
    t2new.aaaa += np.transpose(tmp62, (0, 1, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp66, (0, 1, 3, 2)) * -1
    t2new.aaaa += tmp57
    t2new.aaaa += np.transpose(tmp57, (1, 0, 3, 2))
    del tmp57
    t2new.aaaa += tmp62
    del tmp62
    t2new.aaaa += tmp66
    del tmp66
    tmp68 = np.copy(v.aaaa.vvvv)
    tmp67 = np.copy(v.aaaa.ovvv)
    tmp67 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 0, 4), (2, 4, 1, 3))
    tmp68 += einsum(tmp67, (0, 1, 2, 3), t1.aa, (0, 4), (3, 4, 2, 1))
    del tmp67
    t2new.aaaa += einsum(tmp68, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 3), (4, 5, 2, 1)) * -2
    del tmp68
    tmp69 = np.copy(v.aaaa.ovvv)
    tmp69 += np.transpose(v.aaaa.ovvv, (0, 2, 1, 3)) * -1
    tmp70 = einsum(tmp69, (0, 1, 2, 3), t1.aa, (4, 2), (4, 0, 1, 3))
    del tmp69
    tmp73 = np.copy(np.transpose(tmp70, (0, 1, 3, 2))) * -1
    del tmp70
    tmp71 = np.copy(tmp14) * -1
    tmp71 += np.transpose(tmp14, (0, 2, 1, 3))
    tmp72 = einsum(tmp71, (0, 1, 2, 3), t1.aa, (2, 4), (0, 1, 4, 3))
    del tmp71
    tmp73 += tmp72 * -1
    tmp74 = einsum(t2.aaaa, (0, 1, 2, 3), tmp73, (4, 1, 5, 3), (0, 4, 2, 5)) * 2
    del tmp73
    t2new.aaaa += np.transpose(tmp74, (0, 1, 3, 2))
    tmp75 = einsum(v.aabb.vvov, (0, 1, 2, 3), t1.aa, (4, 1), (4, 2, 0, 3))
    tmp77 = np.copy(tmp75) * -1
    tmp76 = einsum(tmp13, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3))
    tmp77 += tmp76
    tmp78 = einsum(t2.abab, (0, 1, 2, 3), tmp77, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp77
    t2new.aaaa += np.transpose(tmp78, (0, 1, 3, 2))
    t2new.aaaa += np.transpose(tmp74, (1, 0, 3, 2)) * -1
    tmp79 = np.copy(tmp75)
    tmp79 += tmp76 * -1
    tmp80 = einsum(t2.abab, (0, 1, 2, 3), tmp79, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new.aaaa += np.transpose(tmp80, (1, 0, 3, 2))
    t2new.aaaa += tmp74 * -1
    t2new.aaaa += tmp78 * -1
    del tmp78
    t2new.aaaa += np.transpose(tmp74, (1, 0, 2, 3))
    del tmp74
    t2new.aaaa += np.transpose(tmp80, (1, 0, 2, 3)) * -1
    del tmp80
    tmp81 = einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new.aaaa += einsum(tmp81, (0, 1, 2, 3), t1.aa, (2, 4), (0, 1, 4, 3)) * -2
    del tmp81
    tmp82 = einsum(tmp14, (0, 1, 2, 3), t1.aa, (4, 3), (0, 4, 2, 1))
    del tmp14
    t2new.aaaa += einsum(tmp82, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 5), (1, 0, 4, 5)) * -2
    tmp83 = einsum(t1.aa, (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp84 = einsum(tmp83, (0, 1, 2, 3), t1.aa, (4, 2), (4, 0, 1, 3))
    del tmp83
    tmp87 = np.copy(tmp84)
    del tmp84
    tmp85 = einsum(t1.aa, (0, 1), tmp82, (2, 3, 0, 4), (3, 2, 4, 1))
    del tmp82
    tmp86 = einsum(tmp85, (0, 1, 2, 3), t1.aa, (2, 4), (0, 1, 4, 3))
    del tmp85
    tmp87 += tmp86
    del tmp86
    t2new.aaaa += tmp87
    t2new.aaaa += np.transpose(tmp87, (0, 1, 3, 2)) * -1
    del tmp87
    tmp88 = einsum(tmp49, (0, 1, 2, 3), t1.aa, (4, 2), (4, 0, 1, 3))
    del tmp49
    tmp89 = einsum(tmp88, (0, 1, 2, 3), t1.aa, (2, 4), (0, 1, 4, 3))
    del tmp88
    t2new.aaaa += np.transpose(tmp89, (0, 1, 3, 2))
    t2new.aaaa += np.transpose(tmp89, (1, 0, 3, 2)) * -1
    t2new.aaaa += tmp89 * -1
    t2new.aaaa += np.transpose(tmp89, (1, 0, 2, 3))
    del tmp89
    tmp134 = np.copy(f.bb.oo) * 2
    tmp134 += np.transpose(tmp22, (1, 0)) * 2
    tmp134 += np.transpose(tmp29, (1, 0))
    tmp134 += np.transpose(tmp30, (1, 0)) * 2
    tmp133 = einsum(tmp23, (0, 1, 2, 3), t1.bb, (2, 3), (0, 1)) * 2
    del tmp23
    tmp134 += np.transpose(tmp133, (1, 0)) * -1
    del tmp133
    tmp135 = einsum(t2.bbbb, (0, 1, 2, 3), tmp134, (1, 4), (0, 4, 2, 3)) * -1
    del tmp134
    t2new.bbbb = np.copy(np.transpose(tmp135, (0, 1, 3, 2))) * -1
    tmp136 = einsum(v.bbbb.ooov, (0, 1, 2, 3), t1.bb, (1, 4), (0, 2, 4, 3))
    t2new.bbbb += np.transpose(tmp136, (0, 1, 3, 2))
    t2new.bbbb += einsum(v.bbbb.oooo, (0, 1, 2, 3), t2.bbbb, (1, 3, 4, 5), (0, 2, 4, 5)) * 2
    tmp137 = einsum(t1.bb, (0, 1), v.bbbb.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    tmp138 = einsum(t1.bb, (0, 1), tmp137, (2, 0, 3, 4), (3, 2, 4, 1))
    del tmp137
    t2new.bbbb += tmp138
    t2new.bbbb += np.transpose(tmp138, (0, 1, 3, 2)) * -1
    del tmp138
    t2new.bbbb += np.transpose(tmp135, (1, 0, 3, 2))
    del tmp135
    tmp97 = np.copy(f.bb.vv) * -1
    tmp91 = einsum(v.aabb.ovvv, (0, 1, 2, 3), t1.aa, (0, 1), (2, 3))
    tmp97 += np.transpose(tmp91, (1, 0)) * -1
    del tmp91
    tmp92 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 2, 1, 4), (4, 3))
    tmp97 += np.transpose(tmp92, (1, 0)) * 0.5
    del tmp92
    tmp93 = einsum(v.bbbb.ovov, (0, 1, 2, 3), t2.bbbb, (0, 2, 4, 1), (4, 3)) * -1
    tmp97 += np.transpose(tmp93, (1, 0))
    del tmp93
    tmp94 = np.copy(v.bbbb.ovvv)
    tmp94 += np.transpose(v.bbbb.ovvv, (0, 2, 3, 1)) * -1
    tmp95 = einsum(t1.bb, (0, 1), tmp94, (0, 1, 2, 3), (2, 3))
    del tmp94
    tmp97 += np.transpose(tmp95, (1, 0)) * -1
    del tmp95
    tmp96 = einsum(tmp4, (0, 1), t1.bb, (0, 2), (2, 1))
    del tmp4
    tmp97 += np.transpose(tmp96, (1, 0))
    del tmp96
    tmp139 = einsum(t2.bbbb, (0, 1, 2, 3), tmp97, (3, 4), (0, 1, 2, 4)) * -2
    t2new.bbbb += np.transpose(tmp139, (1, 0, 3, 2))
    t2new.bbbb += np.transpose(tmp139, (1, 0, 2, 3)) * -1
    del tmp139
    tmp140 = einsum(v.bbbb.ovvv, (0, 1, 2, 3), t1.bb, (4, 3), (4, 0, 1, 2))
    t2new.bbbb += np.transpose(tmp140, (1, 0, 2, 3))
    tmp141 = einsum(v.bbbb.ooov, (0, 1, 2, 3), t1.bb, (4, 3), (4, 0, 1, 2))
    tmp142 = einsum(t2.bbbb, (0, 1, 2, 3), tmp141, (4, 5, 0, 1), (4, 5, 2, 3))
    t2new.bbbb += np.transpose(tmp142, (1, 0, 3, 2)) * -2
    tmp143 = einsum(t1.bb, (0, 1), tmp141, (2, 3, 4, 0), (2, 4, 3, 1))
    del tmp141
    tmp144 = einsum(tmp143, (0, 1, 2, 3), t1.bb, (1, 4), (0, 2, 4, 3))
    del tmp143
    t2new.bbbb += np.transpose(tmp144, (1, 0, 2, 3))
    t2new.bbbb += np.transpose(tmp144, (1, 0, 3, 2)) * -1
    tmp145 = einsum(tmp31, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4)) * -2
    t2new.bbbb += np.transpose(tmp145, (0, 1, 3, 2)) * -1
    tmp146 = einsum(t1.bb, (0, 1), tmp21, (2, 3, 1, 4), (0, 2, 3, 4))
    del tmp21
    tmp147 = einsum(tmp146, (0, 1, 2, 3), t1.bb, (1, 4), (0, 2, 4, 3))
    del tmp146
    t2new.bbbb += np.transpose(tmp147, (1, 0, 3, 2)) * -1
    t2new.bbbb += tmp140 * -1
    t2new.bbbb += np.transpose(tmp142, (0, 1, 3, 2)) * 2
    del tmp142
    t2new.bbbb += tmp144 * -1
    t2new.bbbb += np.transpose(tmp144, (0, 1, 3, 2))
    del tmp144
    t2new.bbbb += np.transpose(tmp145, (1, 0, 3, 2))
    del tmp145
    t2new.bbbb += np.transpose(tmp147, (0, 1, 3, 2))
    t2new.bbbb += np.transpose(v.bbbb.ovov, (0, 2, 1, 3))
    t2new.bbbb += np.transpose(tmp136, (1, 0, 3, 2)) * -1
    tmp113 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.bbbb, (4, 2, 5, 3), (0, 4, 1, 5))
    tmp148 = einsum(tmp113, (0, 1, 2, 3), t2.abab, (0, 4, 2, 5), (4, 1, 5, 3))
    t2new.bbbb += tmp148
    t2new.bbbb += np.transpose(tmp148, (1, 0, 3, 2))
    tmp151 = np.copy(np.transpose(v.bbbb.ovov, (0, 2, 1, 3)))
    tmp151 += v.bbbb.oovv * -1
    tmp109 = einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    tmp151 += tmp109
    tmp149 = np.copy(v.bbbb.ooov)
    tmp149 += np.transpose(v.bbbb.ovoo, (0, 2, 3, 1)) * -1
    tmp150 = einsum(tmp149, (0, 1, 2, 3), t1.bb, (0, 4), (1, 2, 4, 3))
    del tmp149
    tmp151 += tmp150 * -1
    del tmp150
    tmp152 = einsum(t2.bbbb, (0, 1, 2, 3), tmp151, (4, 1, 5, 3), (0, 4, 2, 5)) * 2
    del tmp151
    t2new.bbbb += np.transpose(tmp152, (1, 0, 3, 2))
    tmp153 = np.copy(np.transpose(v.aabb.ovov, (0, 2, 1, 3)))
    tmp102 = einsum(t1.bb, (0, 1), v.aabb.ovoo, (2, 3, 4, 0), (2, 4, 3, 1))
    tmp153 += tmp102 * -1
    tmp112 = einsum(t2.abab, (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 2), (4, 1, 5, 3))
    tmp153 += tmp112 * 0.5
    tmp154 = einsum(t2.abab, (0, 1, 2, 3), tmp153, (0, 4, 2, 5), (1, 4, 3, 5))
    del tmp153
    t2new.bbbb += np.transpose(tmp154, (1, 0, 3, 2))
    t2new.bbbb += np.transpose(v.bbbb.ovov, (0, 2, 3, 1)) * -1
    t2new.bbbb += np.transpose(tmp136, (1, 0, 2, 3))
    t2new.bbbb += np.transpose(tmp148, (0, 1, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp148, (1, 0, 2, 3)) * -1
    t2new.bbbb += np.transpose(tmp152, (1, 0, 2, 3)) * -1
    tmp155 = np.copy(np.transpose(v.aabb.ovov, (0, 2, 1, 3))) * 2
    tmp155 += tmp102 * -2
    tmp155 += tmp112
    tmp156 = einsum(t2.abab, (0, 1, 2, 3), tmp155, (0, 4, 2, 5), (1, 4, 3, 5)) * 0.5
    del tmp155
    t2new.bbbb += np.transpose(tmp156, (1, 0, 2, 3)) * -1
    t2new.bbbb += tmp136 * -1
    del tmp136
    t2new.bbbb += np.transpose(tmp140, (0, 1, 3, 2))
    t2new.bbbb += tmp147 * -1
    t2new.bbbb += np.transpose(tmp140, (1, 0, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp147, (1, 0, 2, 3))
    del tmp147
    t2new.bbbb += np.transpose(tmp148, (0, 1, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp148, (1, 0, 2, 3)) * -1
    t2new.bbbb += np.transpose(tmp152, (0, 1, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp154, (0, 1, 3, 2)) * -1
    del tmp154
    t2new.bbbb += tmp148
    t2new.bbbb += np.transpose(tmp148, (1, 0, 3, 2))
    del tmp148
    t2new.bbbb += tmp152
    del tmp152
    t2new.bbbb += tmp156
    del tmp156
    tmp158 = np.copy(v.bbbb.vvvv)
    tmp157 = np.copy(v.bbbb.ovvv)
    tmp157 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 0, 4), (2, 4, 1, 3))
    tmp158 += einsum(t1.bb, (0, 1), tmp157, (0, 2, 3, 4), (4, 1, 3, 2))
    del tmp157
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), tmp158, (2, 4, 5, 3), (0, 1, 5, 4)) * -2
    del tmp158
    tmp121 = np.copy(v.bbbb.ovvv) * -1
    tmp121 += np.transpose(v.bbbb.ovvv, (0, 2, 1, 3))
    tmp159 = einsum(tmp121, (0, 1, 2, 3), t1.bb, (4, 1), (4, 0, 2, 3))
    tmp160 = np.copy(np.transpose(tmp159, (0, 1, 3, 2))) * -1
    del tmp159
    tmp122 = np.copy(tmp26) * -1
    tmp122 += np.transpose(tmp26, (0, 2, 1, 3))
    tmp123 = einsum(t1.bb, (0, 1), tmp122, (2, 3, 0, 4), (2, 3, 1, 4))
    del tmp122
    tmp160 += tmp123 * -1
    tmp161 = einsum(t2.bbbb, (0, 1, 2, 3), tmp160, (4, 1, 5, 3), (0, 4, 2, 5)) * 2
    del tmp160
    t2new.bbbb += np.transpose(tmp161, (0, 1, 3, 2))
    tmp99 = einsum(v.aabb.ovvv, (0, 1, 2, 3), t1.bb, (4, 3), (0, 4, 1, 2))
    tmp162 = np.copy(tmp99) * -1
    tmp98 = einsum(t1.bb, (0, 1), tmp27, (2, 3, 0, 4), (2, 3, 4, 1))
    tmp162 += tmp98
    tmp163 = einsum(t2.abab, (0, 1, 2, 3), tmp162, (0, 4, 2, 5), (1, 4, 3, 5))
    del tmp162
    t2new.bbbb += np.transpose(tmp163, (0, 1, 3, 2))
    t2new.bbbb += np.transpose(tmp161, (1, 0, 3, 2)) * -1
    tmp125 = np.copy(tmp99)
    tmp125 += tmp98 * -1
    tmp164 = einsum(tmp125, (0, 1, 2, 3), t2.abab, (0, 4, 2, 5), (4, 1, 5, 3))
    t2new.bbbb += np.transpose(tmp164, (1, 0, 3, 2))
    t2new.bbbb += tmp161 * -1
    t2new.bbbb += tmp163 * -1
    del tmp163
    t2new.bbbb += np.transpose(tmp161, (1, 0, 2, 3))
    del tmp161
    t2new.bbbb += np.transpose(tmp164, (1, 0, 2, 3)) * -1
    del tmp164
    tmp165 = einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new.bbbb += einsum(tmp165, (0, 1, 2, 3), t1.bb, (2, 4), (0, 1, 4, 3)) * -2
    del tmp165
    tmp166 = einsum(t1.bb, (0, 1), tmp26, (2, 3, 4, 1), (2, 0, 4, 3))
    del tmp26
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), tmp166, (4, 5, 1, 0), (5, 4, 2, 3)) * 2
    tmp167 = einsum(t1.bb, (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp168 = einsum(t1.bb, (0, 1), tmp167, (2, 3, 1, 4), (0, 2, 3, 4))
    del tmp167
    tmp171 = np.copy(tmp168)
    del tmp168
    tmp169 = einsum(t1.bb, (0, 1), tmp166, (2, 3, 0, 4), (3, 2, 4, 1))
    del tmp166
    tmp170 = einsum(tmp169, (0, 1, 2, 3), t1.bb, (2, 4), (0, 1, 4, 3))
    del tmp169
    tmp171 += tmp170
    del tmp170
    t2new.bbbb += tmp171
    t2new.bbbb += np.transpose(tmp171, (0, 1, 3, 2)) * -1
    del tmp171
    tmp172 = einsum(tmp140, (0, 1, 2, 3), t1.bb, (4, 2), (4, 0, 1, 3))
    del tmp140
    tmp173 = einsum(tmp172, (0, 1, 2, 3), t1.bb, (2, 4), (0, 1, 4, 3))
    del tmp172
    t2new.bbbb += np.transpose(tmp173, (0, 1, 3, 2))
    t2new.bbbb += np.transpose(tmp173, (1, 0, 3, 2)) * -1
    t2new.bbbb += tmp173 * -1
    t2new.bbbb += np.transpose(tmp173, (1, 0, 2, 3))
    del tmp173
    tmp90 = np.copy(f.bb.oo)
    tmp90 += np.transpose(tmp22, (1, 0))
    del tmp22
    tmp90 += np.transpose(tmp29, (1, 0)) * 0.5
    del tmp29
    tmp90 += np.transpose(tmp30, (1, 0))
    del tmp30
    tmp90 += np.transpose(tmp24, (1, 0)) * -1
    del tmp24
    t2new.abab = einsum(t2.abab, (0, 1, 2, 3), tmp90, (1, 4), (0, 4, 2, 3)) * -1
    del tmp90
    t2new.abab += einsum(tmp34, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1
    del tmp34
    t2new.abab += einsum(tmp15, (0, 1, 2, 3), v.aabb.oooo, (4, 0, 5, 1), (4, 5, 2, 3))
    t2new.abab += einsum(tmp45, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1
    del tmp45
    t2new.abab += einsum(tmp97, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1
    del tmp97
    t2new.abab += tmp98 * -1
    del tmp98
    t2new.abab += tmp99
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp31, (4, 1), (0, 4, 2, 3)) * -1
    del tmp31
    tmp100 = einsum(v.aabb.ooov, (0, 1, 2, 3), t1.bb, (4, 3), (0, 1, 4, 2))
    t2new.abab += einsum(tmp15, (0, 1, 2, 3), tmp100, (0, 4, 5, 1), (4, 5, 2, 3))
    del tmp100
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp52, (4, 0), (4, 1, 2, 3)) * -1
    del tmp52
    tmp101 = einsum(v.aabb.ovoo, (0, 1, 2, 3), t1.aa, (4, 1), (4, 0, 2, 3))
    t2new.abab += einsum(tmp101, (0, 1, 2, 3), tmp15, (1, 2, 4, 5), (0, 3, 4, 5))
    del tmp101
    t2new.abab += np.transpose(v.aabb.ovov, (0, 2, 1, 3))
    t2new.abab += tmp102 * -1
    tmp104 = np.copy(np.transpose(v.aaaa.ovov, (0, 2, 1, 3)))
    tmp104 += v.aaaa.oovv * -1
    tmp104 += np.transpose(tmp58, (1, 0, 3, 2))
    del tmp58
    tmp104 += np.transpose(tmp56, (1, 0, 3, 2)) * 0.5
    del tmp56
    tmp103 = np.copy(v.aaaa.ooov)
    tmp103 += np.transpose(v.aaaa.ooov, (0, 2, 1, 3)) * -1
    tmp104 += einsum(t1.aa, (0, 1), tmp103, (2, 3, 0, 4), (3, 2, 4, 1))
    del tmp103
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp104, (0, 4, 2, 5), (4, 1, 5, 3))
    del tmp104
    tmp105 = np.copy(np.transpose(v.aabb.ovov, (0, 2, 1, 3))) * 2
    tmp105 += tmp63 * -2
    tmp105 += einsum(v.aabb.ovov, (0, 1, 2, 3), t2.aaaa, (4, 0, 5, 1), (4, 2, 5, 3)) * 2
    tmp105 += tmp64
    del tmp64
    t2new.abab += einsum(tmp105, (0, 1, 2, 3), t2.bbbb, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp105
    t2new.abab += tmp63 * -1
    del tmp63
    t2new.abab += tmp75
    del tmp75
    tmp106 = einsum(v.aabb.vvoo, (0, 1, 2, 3), t1.aa, (4, 1), (4, 2, 3, 0))
    t2new.abab += einsum(t1.bb, (0, 1), tmp106, (2, 3, 0, 4), (2, 3, 4, 1)) * -1
    del tmp106
    tmp107 = np.copy(v.aaaa.ovvv)
    tmp107 += np.transpose(v.aaaa.ovvv, (0, 2, 3, 1)) * -1
    tmp108 = einsum(tmp107, (0, 1, 2, 3), t1.aa, (4, 1), (4, 0, 3, 2)) * -1
    del tmp107
    tmp108 += np.transpose(tmp72, (0, 1, 3, 2))
    del tmp72
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp108, (4, 0, 2, 5), (4, 1, 5, 3))
    del tmp108
    t2new.abab += einsum(tmp79, (0, 1, 2, 3), t2.bbbb, (4, 1, 5, 3), (0, 4, 2, 5)) * 2
    del tmp79
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvoo, (4, 2, 5, 1), (0, 5, 4, 3)) * -1
    tmp111 = np.copy(np.transpose(v.bbbb.ovov, (0, 2, 1, 3)))
    tmp111 += v.bbbb.oovv * -1
    tmp111 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (4, 1, 5, 3)) * 0.5
    tmp111 += np.transpose(tmp109, (1, 0, 3, 2))
    del tmp109
    tmp110 = np.copy(v.bbbb.ooov)
    tmp110 += np.transpose(v.bbbb.ooov, (0, 2, 1, 3)) * -1
    tmp111 += einsum(t1.bb, (0, 1), tmp110, (2, 3, 0, 4), (3, 2, 4, 1))
    del tmp110
    t2new.abab += einsum(tmp111, (0, 1, 2, 3), t2.abab, (4, 0, 5, 2), (4, 1, 5, 3))
    del tmp111
    tmp114 = np.copy(np.transpose(v.aabb.ovov, (0, 2, 1, 3)))
    tmp114 += tmp102 * -1
    del tmp102
    tmp114 += tmp112 * 0.5
    del tmp112
    tmp114 += tmp113
    del tmp113
    t2new.abab += einsum(t2.aaaa, (0, 1, 2, 3), tmp114, (1, 4, 3, 5), (0, 4, 2, 5)) * 2
    del tmp114
    tmp115 = np.copy(v.aabb.oovv)
    tmp115 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 4), (2, 3, 4, 1)) * -1
    t2new.abab += einsum(tmp115, (0, 1, 2, 3), t2.abab, (0, 4, 5, 2), (1, 4, 5, 3)) * -1
    del tmp115
    tmp117 = np.copy(v.aabb.vvvv)
    tmp116 = np.copy(np.transpose(v.aabb.vvov, (2, 0, 1, 3)))
    tmp116 += einsum(v.aabb.ovov, (0, 1, 2, 3), t1.aa, (0, 4), (2, 4, 1, 3)) * -1
    tmp117 += einsum(t1.bb, (0, 1), tmp116, (0, 2, 3, 4), (3, 2, 4, 1)) * -1
    del tmp116
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp117, (2, 4, 3, 5), (0, 1, 4, 5))
    del tmp117
    tmp118 = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovoo, (4, 2, 5, 1), (0, 4, 5, 3))
    t2new.abab += einsum(t1.aa, (0, 1), tmp118, (2, 0, 3, 4), (2, 3, 1, 4))
    del tmp118
    tmp119 = einsum(v.aabb.vvov, (0, 1, 2, 3), t1.bb, (4, 3), (4, 2, 0, 1))
    tmp120 = np.copy(np.transpose(tmp119, (0, 1, 3, 2)))
    tmp120 += einsum(tmp27, (0, 1, 2, 3), t1.aa, (0, 4), (1, 2, 3, 4)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp120, (4, 1, 2, 5), (0, 4, 5, 3)) * -1
    del tmp120
    tmp124 = einsum(t1.bb, (0, 1), tmp121, (2, 3, 1, 4), (0, 2, 3, 4)) * -1
    del tmp121
    tmp124 += np.transpose(tmp123, (0, 1, 3, 2))
    del tmp123
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp124, (4, 1, 3, 5), (0, 4, 2, 5))
    del tmp124
    t2new.abab += einsum(t2.aaaa, (0, 1, 2, 3), tmp125, (1, 4, 3, 5), (0, 4, 2, 5)) * 2
    del tmp125
    tmp126 = einsum(v.aabb.ovvv, (0, 1, 2, 3), t1.aa, (4, 1), (4, 0, 2, 3))
    tmp126 += einsum(t1.bb, (0, 1), tmp13, (2, 3, 0, 4), (2, 3, 4, 1)) * -1
    del tmp13
    t2new.abab += einsum(tmp126, (0, 1, 2, 3), t2.abab, (1, 4, 5, 2), (0, 4, 5, 3)) * -1
    del tmp126
    tmp127 = einsum(v.aabb.ovvv, (0, 1, 2, 3), t2.abab, (4, 5, 1, 3), (4, 0, 5, 2))
    t2new.abab += einsum(tmp127, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3)) * -1
    del tmp127
    t2new.abab += tmp76 * -1
    del tmp76
    tmp128 = einsum(v.aabb.oovv, (0, 1, 2, 3), t1.bb, (4, 3), (0, 1, 4, 2))
    t2new.abab += einsum(tmp128, (0, 1, 2, 3), t1.aa, (1, 4), (0, 2, 4, 3)) * -1
    del tmp128
    tmp129 = einsum(t1.bb, (0, 1), v.aabb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new.abab += einsum(tmp129, (0, 1, 2, 3), t1.aa, (4, 2), (4, 0, 1, 3))
    del tmp129
    tmp130 = einsum(tmp119, (0, 1, 2, 3), t1.aa, (4, 3), (4, 0, 1, 2))
    del tmp119
    t2new.abab += einsum(t1.bb, (0, 1), tmp130, (2, 3, 0, 4), (2, 3, 4, 1)) * -1
    del tmp130
    tmp131 = einsum(tmp27, (0, 1, 2, 3), t1.aa, (4, 3), (4, 0, 1, 2))
    del tmp27
    t2new.abab += einsum(tmp131, (0, 1, 2, 3), tmp15, (1, 3, 4, 5), (0, 2, 4, 5))
    del tmp131, tmp15
    tmp132 = einsum(t1.aa, (0, 1), tmp99, (2, 3, 1, 4), (0, 2, 3, 4))
    del tmp99
    t2new.abab += einsum(t1.aa, (0, 1), tmp132, (2, 0, 3, 4), (2, 3, 1, 4)) * -1
    del tmp132

    return {"t1new": t1new, "t2new": t2new}

