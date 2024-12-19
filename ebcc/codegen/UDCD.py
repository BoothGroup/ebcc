"""Code generated by `albert` version 0.0.0.

 * date: 2024-12-19T15:11:05.378848
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


def energy(t2=None, v=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        t2: 
        v: 

    Returns:
        e_cc: 
    """

    e_cc = einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (0, 2, 1, 3), ())
    e_cc += einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 2, 1, 3), ())

    return e_cc

def update_amps(f=None, t2=None, v=None, **kwargs):
    """Code generated by `albert` 0.0.0.

    Args:
        f: 
        t2: 
        v: 

    Returns:
        t2new: 
    """

    tmp2 = np.copy(f.aa.oo)
    tmp0 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (4, 2, 1, 3), (4, 0))
    tmp2 += tmp0
    tmp1 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (4, 2, 1, 3), (4, 0))
    tmp2 += tmp1 * 0.5
    tmp3 = einsum(t2.aaaa, (0, 1, 2, 3), tmp2, (4, 1), (0, 4, 2, 3)) * -2
    del tmp2
    t2new = Namespace()
    t2new.aaaa = np.copy(np.transpose(tmp3, (0, 1, 3, 2))) * -1
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.oooo, (4, 0, 5, 1), (4, 5, 2, 3)) * 2
    t2new.aaaa += np.transpose(tmp3, (1, 0, 3, 2))
    del tmp3
    tmp6 = np.copy(f.aa.vv) * -1
    tmp4 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (0, 2, 4, 3), (4, 1))
    tmp6 += tmp4
    tmp5 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 2, 4, 3), (4, 1))
    tmp6 += tmp5 * 0.5
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), tmp6, (4, 3), (0, 1, 4, 2)) * 2
    del tmp6
    tmp7 = np.copy(f.aa.vv) * -2
    tmp7 += tmp4 * 2
    tmp7 += tmp5
    t2new.aaaa += einsum(tmp7, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 4, 0)) * -1
    del tmp7
    t2new.aaaa += np.transpose(v.aaaa.ovov, (0, 2, 1, 3))
    tmp8 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (4, 2, 5, 3), (4, 0, 5, 1))
    tmp9 = einsum(tmp8, (0, 1, 2, 3), t2.aaaa, (4, 1, 5, 3), (4, 0, 5, 2))
    t2new.aaaa += tmp9
    t2new.aaaa += np.transpose(tmp9, (1, 0, 3, 2))
    tmp11 = np.copy(np.transpose(v.aaaa.ovov, (0, 2, 1, 3)))
    tmp11 += v.aaaa.oovv * -1
    tmp10 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (4, 2, 5, 3), (4, 0, 5, 1))
    tmp11 += tmp10
    tmp12 = einsum(t2.aaaa, (0, 1, 2, 3), tmp11, (4, 1, 5, 3), (0, 4, 2, 5)) * 2
    del tmp11
    t2new.aaaa += np.transpose(tmp12, (1, 0, 3, 2))
    tmp14 = np.copy(np.transpose(v.aabb.ovov, (0, 2, 1, 3))) * 2
    tmp13 = einsum(v.bbbb.ovov, (0, 1, 2, 3), t2.abab, (4, 2, 5, 3), (4, 0, 5, 1))
    tmp14 += tmp13
    tmp15 = einsum(tmp14, (0, 1, 2, 3), t2.abab, (4, 1, 5, 3), (4, 0, 5, 2)) * 0.5
    del tmp14
    t2new.aaaa += np.transpose(tmp15, (1, 0, 3, 2))
    t2new.aaaa += np.transpose(v.aaaa.ovov, (0, 2, 3, 1)) * -1
    t2new.aaaa += np.transpose(tmp9, (0, 1, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp9, (1, 0, 2, 3)) * -1
    t2new.aaaa += np.transpose(tmp12, (1, 0, 2, 3)) * -1
    t2new.aaaa += np.transpose(tmp15, (1, 0, 2, 3)) * -1
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 2, 5, 3), (0, 1, 4, 5)) * 2
    t2new.aaaa += tmp9
    t2new.aaaa += np.transpose(tmp9, (1, 0, 3, 2))
    t2new.aaaa += tmp12
    t2new.aaaa += tmp15
    del tmp15
    t2new.aaaa += np.transpose(tmp9, (0, 1, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp9, (1, 0, 2, 3)) * -1
    del tmp9
    t2new.aaaa += np.transpose(tmp12, (0, 1, 3, 2)) * -1
    del tmp12
    tmp16 = np.copy(np.transpose(v.aabb.ovov, (0, 2, 1, 3)))
    tmp16 += tmp13 * 0.5
    t2new.aaaa += einsum(t2.abab, (0, 1, 2, 3), tmp16, (4, 1, 5, 3), (0, 4, 5, 2)) * -1
    del tmp16
    tmp32 = np.copy(f.bb.oo) * 2
    tmp17 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 4, 1, 3), (4, 2))
    tmp32 += tmp17
    tmp18 = einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (0, 4)) * -1
    tmp32 += tmp18 * 2
    tmp33 = einsum(tmp32, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4)) * -1
    del tmp32
    t2new.bbbb = np.copy(np.transpose(tmp33, (1, 0, 3, 2))) * -1
    t2new.bbbb += einsum(v.bbbb.oooo, (0, 1, 2, 3), t2.bbbb, (1, 3, 4, 5), (0, 2, 4, 5)) * 2
    t2new.bbbb += np.transpose(tmp33, (0, 1, 3, 2))
    del tmp33
    tmp34 = np.copy(f.bb.vv) * -2
    tmp22 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 2, 1, 4), (4, 3))
    tmp34 += tmp22
    tmp23 = einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 4), (2, 4)) * -1
    tmp34 += tmp23 * 2
    t2new.bbbb += einsum(tmp34, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    del tmp34
    tmp35 = np.copy(f.bb.vv) * -1
    tmp35 += tmp22 * 0.5
    tmp35 += tmp23
    t2new.bbbb += einsum(tmp35, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 4, 0)) * -2
    del tmp35
    t2new.bbbb += np.transpose(v.bbbb.ovov, (0, 2, 1, 3))
    tmp30 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.bbbb, (4, 2, 5, 3), (0, 4, 1, 5))
    tmp36 = einsum(tmp30, (0, 1, 2, 3), t2.abab, (0, 4, 2, 5), (4, 1, 5, 3))
    t2new.bbbb += tmp36
    t2new.bbbb += np.transpose(tmp36, (1, 0, 3, 2))
    tmp37 = np.copy(np.transpose(v.bbbb.ovov, (0, 2, 1, 3)))
    tmp37 += v.bbbb.oovv * -1
    tmp27 = einsum(v.bbbb.ovov, (0, 1, 2, 3), t2.bbbb, (4, 2, 5, 3), (4, 0, 5, 1))
    tmp37 += tmp27
    tmp38 = einsum(t2.bbbb, (0, 1, 2, 3), tmp37, (4, 1, 5, 3), (4, 0, 5, 2)) * 2
    del tmp37
    t2new.bbbb += tmp38
    tmp39 = np.copy(np.transpose(v.aabb.ovov, (0, 2, 1, 3)))
    tmp29 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.abab, (2, 4, 3, 5), (0, 4, 1, 5))
    tmp39 += tmp29 * 0.5
    tmp40 = einsum(t2.abab, (0, 1, 2, 3), tmp39, (0, 4, 2, 5), (4, 1, 5, 3))
    del tmp39
    t2new.bbbb += tmp40
    t2new.bbbb += np.transpose(v.bbbb.ovov, (0, 2, 3, 1)) * -1
    t2new.bbbb += np.transpose(tmp36, (0, 1, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp36, (1, 0, 2, 3)) * -1
    t2new.bbbb += np.transpose(tmp38, (0, 1, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp40, (0, 1, 3, 2)) * -1
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5)) * 2
    t2new.bbbb += tmp36
    t2new.bbbb += np.transpose(tmp36, (1, 0, 3, 2))
    t2new.bbbb += np.transpose(tmp38, (1, 0, 3, 2))
    t2new.bbbb += np.transpose(tmp40, (1, 0, 3, 2))
    t2new.bbbb += np.transpose(tmp36, (0, 1, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp36, (1, 0, 2, 3)) * -1
    del tmp36
    t2new.bbbb += np.transpose(tmp38, (1, 0, 2, 3)) * -1
    del tmp38
    t2new.bbbb += np.transpose(tmp40, (1, 0, 2, 3)) * -1
    del tmp40
    tmp19 = np.copy(f.bb.oo) * 2
    tmp19 += np.transpose(tmp17, (1, 0))
    del tmp17
    tmp19 += np.transpose(tmp18, (1, 0)) * 2
    del tmp18
    t2new.abab = einsum(tmp19, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -0.5
    del tmp19
    t2new.abab += einsum(v.aabb.oooo, (0, 1, 2, 3), t2.abab, (1, 3, 4, 5), (0, 2, 4, 5))
    tmp20 = np.copy(f.aa.oo)
    tmp20 += np.transpose(tmp0, (1, 0))
    del tmp0
    tmp20 += np.transpose(tmp1, (1, 0)) * 0.5
    del tmp1
    t2new.abab += einsum(tmp20, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1
    del tmp20
    tmp21 = np.copy(f.aa.vv) * -1
    tmp21 += np.transpose(tmp4, (1, 0))
    del tmp4
    tmp21 += np.transpose(tmp5, (1, 0)) * 0.5
    del tmp5
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp21, (2, 4), (0, 1, 4, 3)) * -1
    del tmp21
    tmp24 = np.copy(f.bb.vv) * -1
    tmp24 += np.transpose(tmp22, (1, 0)) * 0.5
    del tmp22
    tmp24 += np.transpose(tmp23, (1, 0))
    del tmp23
    t2new.abab += einsum(tmp24, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1
    del tmp24
    t2new.abab += np.transpose(v.aabb.ovov, (0, 2, 1, 3))
    tmp25 = np.copy(np.transpose(v.aaaa.ovov, (0, 2, 1, 3)))
    tmp25 += v.aaaa.oovv * -1
    tmp25 += np.transpose(tmp10, (1, 0, 3, 2))
    del tmp10
    tmp25 += np.transpose(tmp8, (1, 0, 3, 2)) * 0.5
    del tmp8
    t2new.abab += einsum(tmp25, (0, 1, 2, 3), t2.abab, (0, 4, 2, 5), (1, 4, 3, 5))
    del tmp25
    tmp26 = np.copy(np.transpose(v.aabb.ovov, (0, 2, 1, 3))) * 2
    tmp26 += einsum(v.aabb.ovov, (0, 1, 2, 3), t2.aaaa, (4, 0, 5, 1), (4, 2, 5, 3)) * 2
    tmp26 += tmp13
    del tmp13
    t2new.abab += einsum(tmp26, (0, 1, 2, 3), t2.bbbb, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp26
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    tmp28 = np.copy(np.transpose(v.bbbb.ovov, (0, 2, 1, 3)))
    tmp28 += v.bbbb.oovv * -1
    tmp28 += einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 4, 1, 5), (2, 4, 3, 5)) * 0.5
    tmp28 += np.transpose(tmp27, (1, 0, 3, 2))
    del tmp27
    t2new.abab += einsum(tmp28, (0, 1, 2, 3), t2.abab, (4, 0, 5, 2), (4, 1, 5, 3))
    del tmp28
    tmp31 = np.copy(np.transpose(v.aabb.ovov, (0, 2, 1, 3)))
    tmp31 += tmp29 * 0.5
    del tmp29
    tmp31 += tmp30
    del tmp30
    t2new.abab += einsum(tmp31, (0, 1, 2, 3), t2.aaaa, (4, 0, 5, 2), (4, 1, 5, 3)) * 2
    del tmp31
    t2new.abab += einsum(v.aabb.oovv, (0, 1, 2, 3), t2.abab, (1, 4, 5, 3), (0, 4, 5, 2)) * -1
    t2new.abab += einsum(v.aabb.vvoo, (0, 1, 2, 3), t2.abab, (4, 3, 1, 5), (4, 2, 0, 5)) * -1

    return {"t2new": t2new}

