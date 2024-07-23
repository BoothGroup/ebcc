"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-07-18T10:46:51.791009
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


def energy(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T10:46:52.067075.

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

    e_cc = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (0, 2, 1, 3), (), optimize=True)
    e_cc += einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 2, 1, 3), (), optimize=True)
    e_cc += einsum(v.bbbb.ovov, (0, 1, 2, 3), t2.bbbb, (0, 2, 3, 1), (), optimize=True) * -1

    return e_cc

def update_amps(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T10:46:59.520905.

    Parameters
    ----------
    f : Namespace of arrays
        Fock matrix.
    t2 : Namespace of arrays
        T2 amplitudes.
    v : Namespace of arrays
        Electron repulsion integrals.

    Returns
    -------
    t2new : Namespace of arrays
        Updated T2 residuals.
    """

    t1new = Namespace()
    t2new = Namespace()
    tmp31 = einsum(v.bbbb.ovov, (0, 1, 2, 3), t2.bbbb, (4, 0, 5, 1), (4, 2, 5, 3), optimize=True)
    tmp33 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.abab, (0, 4, 1, 5), (2, 4, 3, 5), optimize=True)
    tmp21 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 4, 1, 3), (4, 2), optimize=True)
    tmp22 = einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (1, 2, 4, 3), (0, 4), optimize=True) * -1
    tmp26 = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4), optimize=True)
    tmp27 = einsum(v.bbbb.ovov, (0, 1, 2, 3), t2.bbbb, (0, 2, 4, 1), (4, 3), optimize=True) * -1
    tmp1 = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4), optimize=True)
    tmp0 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (4, 0, 3, 1), (4, 2), optimize=True)
    tmp11 = einsum(v.bbbb.ovov, (0, 1, 2, 3), t2.abab, (4, 2, 5, 3), (4, 0, 5, 1), optimize=True)
    tmp14 = einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 1, 3), (0, 4, 2, 5), optimize=True)
    tmp42 = v.bbbb.ovov.transpose((2, 0, 3, 1)).copy()
    tmp42 += tmp31
    tmp44 = v.aabb.ovov.transpose((0, 2, 1, 3)).copy() * 2
    tmp44 += tmp33
    tmp36 = f.bb.oo.transpose((1, 0)).copy()
    tmp36 += tmp21 * 0.5
    tmp36 += tmp22
    tmp34 = einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (4, 0, 5, 2), optimize=True)
    tmp39 = f.bb.vv.transpose((1, 0)).copy() * -1
    tmp39 += tmp26 * 0.5
    tmp39 += tmp27
    tmp46 = v.bbbb.ovov.transpose((2, 0, 3, 1)).copy()
    tmp46 += v.bbbb.oovv.transpose((1, 0, 3, 2)) * -1
    tmp46 += tmp31
    tmp5 = einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (0, 2, 4, 1), (4, 3), optimize=True) * -1
    tmp6 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 2, 4, 3), (4, 1), optimize=True)
    tmp9 = einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (4, 2, 5, 3), (4, 0, 5, 1), optimize=True)
    tmp2 = f.aa.oo.transpose((1, 0)).copy() * 2
    tmp2 += tmp0 * 2
    tmp2 += tmp1
    tmp17 = v.aabb.ovov.transpose((0, 2, 1, 3)).copy()
    tmp17 += tmp11 * 0.5
    tmp15 = v.aaaa.ovov.transpose((2, 0, 3, 1)).copy()
    tmp15 += tmp14
    tmp12 = v.aabb.ovov.transpose((0, 2, 1, 3)).copy() * 2
    tmp12 += tmp11
    tmp19 = v.aaaa.ovov.transpose((2, 0, 3, 1)).copy()
    tmp19 += v.aaaa.oovv.transpose((1, 0, 3, 2)) * -1
    tmp19 += tmp14
    tmp43 = einsum(t2.bbbb, (0, 1, 2, 3), tmp42, (4, 0, 5, 2), (4, 1, 5, 3), optimize=True) * 2
    del tmp42
    t2new.bbbb = tmp43.transpose((0, 1, 3, 2)).copy() * -1
    t2new.bbbb += tmp43
    del tmp43
    tmp45 = einsum(tmp44, (0, 1, 2, 3), t2.abab, (0, 4, 2, 5), (1, 4, 3, 5), optimize=True) * 0.5
    del tmp44
    t2new.bbbb += tmp45.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp45.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp45.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp45
    del tmp45
    tmp37 = einsum(t2.bbbb, (0, 1, 2, 3), tmp36, (4, 0), (4, 1, 2, 3), optimize=True) * 2
    del tmp36
    t2new.bbbb += tmp37.transpose((0, 1, 3, 2))
    t2new.bbbb += tmp37.transpose((1, 0, 3, 2)) * -1
    del tmp37
    tmp38 = einsum(v.bbbb.oovv, (0, 1, 2, 3), t2.bbbb, (4, 1, 5, 3), (4, 0, 5, 2), optimize=True)
    t2new.bbbb += tmp38.transpose((1, 0, 2, 3)) * 2
    t2new.bbbb += tmp38.transpose((1, 0, 3, 2)) * -2
    del tmp38
    tmp41 = einsum(tmp34, (0, 1, 2, 3), t2.abab, (0, 4, 2, 5), (4, 1, 5, 3), optimize=True)
    t2new.bbbb += tmp41.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp41.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp41.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp41
    t2new.bbbb += tmp41.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp41.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp41.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp41
    del tmp41
    tmp40 = einsum(t2.bbbb, (0, 1, 2, 3), tmp39, (4, 2), (0, 1, 4, 3), optimize=True) * 2
    del tmp39
    t2new.bbbb += tmp40.transpose((1, 0, 3, 2)) * -1
    t2new.bbbb += tmp40.transpose((1, 0, 2, 3))
    del tmp40
    tmp47 = einsum(tmp46, (0, 1, 2, 3), t2.bbbb, (1, 4, 3, 5), (0, 4, 2, 5), optimize=True) * 2
    del tmp46
    t2new.bbbb += tmp47.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp47.transpose((1, 0, 3, 2))
    del tmp47
    tmp25 = f.aa.vv.transpose((1, 0)).copy() * -2
    tmp25 += tmp5.transpose((1, 0)) * 2
    tmp25 += tmp6.transpose((1, 0))
    t2new.abab = einsum(tmp25, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4), optimize=True) * -0.5
    del tmp25
    tmp29 = v.aabb.ovov.transpose((0, 2, 1, 3)).copy()
    tmp29 += einsum(v.aabb.ovov, (0, 1, 2, 3), t2.aaaa, (4, 0, 5, 1), (4, 2, 5, 3), optimize=True)
    tmp29 += tmp11 * 0.5
    del tmp11
    t2new.abab += einsum(t2.bbbb, (0, 1, 2, 3), tmp29, (4, 0, 5, 2), (4, 1, 5, 3), optimize=True) * 2
    del tmp29
    tmp24 = f.aa.oo.transpose((1, 0)).copy()
    tmp24 += tmp0.transpose((1, 0))
    del tmp0
    tmp24 += tmp1.transpose((1, 0)) * 0.5
    del tmp1
    t2new.abab += einsum(tmp24, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4), optimize=True) * -1
    del tmp24
    tmp32 = v.bbbb.ovov.transpose((2, 0, 3, 1)).copy()
    tmp32 += v.bbbb.oovv.transpose((1, 0, 3, 2)) * -1
    tmp32 += einsum(v.aabb.ovov, (0, 1, 2, 3), t2.abab, (0, 4, 1, 5), (2, 4, 3, 5), optimize=True) * 0.5
    tmp32 += tmp31.transpose((1, 0, 3, 2))
    del tmp31
    t2new.abab += einsum(tmp32, (0, 1, 2, 3), t2.abab, (4, 0, 5, 2), (4, 1, 5, 3), optimize=True)
    del tmp32
    tmp23 = f.bb.oo.transpose((1, 0)).copy()
    tmp23 += tmp21.transpose((1, 0)) * 0.5
    del tmp21
    tmp23 += tmp22.transpose((1, 0))
    del tmp22
    t2new.abab += einsum(tmp23, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4), optimize=True) * -1
    del tmp23
    tmp35 = v.aabb.ovov.transpose((0, 2, 1, 3)).copy()
    tmp35 += tmp33 * 0.5
    del tmp33
    tmp35 += tmp34
    del tmp34
    t2new.abab += einsum(tmp35, (0, 1, 2, 3), t2.aaaa, (0, 4, 2, 5), (4, 1, 5, 3), optimize=True) * 2
    del tmp35
    tmp30 = v.aaaa.ovov.transpose((2, 0, 3, 1)).copy()
    tmp30 += tmp14.transpose((1, 0, 3, 2))
    del tmp14
    tmp30 += tmp9.transpose((1, 0, 3, 2)) * 0.5
    t2new.abab += einsum(tmp30, (0, 1, 2, 3), t2.abab, (0, 4, 2, 5), (1, 4, 3, 5), optimize=True)
    del tmp30
    tmp28 = f.bb.vv.transpose((1, 0)).copy() * -1
    tmp28 += tmp26.transpose((1, 0)) * 0.5
    del tmp26
    tmp28 += tmp27.transpose((1, 0))
    del tmp27
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp28, (3, 4), (0, 1, 2, 4), optimize=True) * -1
    del tmp28
    tmp4 = einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.oovv, (4, 1, 5, 3), (0, 4, 2, 5), optimize=True)
    t2new.aaaa = tmp4.transpose((1, 0, 2, 3)).copy() * 2
    t2new.aaaa += tmp4.transpose((1, 0, 3, 2)) * -2
    del tmp4
    tmp8 = f.aa.vv.transpose((1, 0)).copy() * -1
    tmp8 += tmp5
    tmp8 += tmp6 * 0.5
    t2new.aaaa += einsum(tmp8, (0, 1), t2.aaaa, (2, 3, 1, 4), (3, 2, 4, 0), optimize=True) * -2
    del tmp8
    tmp3 = einsum(t2.aaaa, (0, 1, 2, 3), tmp2, (4, 0), (1, 4, 2, 3), optimize=True)
    del tmp2
    t2new.aaaa += tmp3.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp3.transpose((0, 1, 3, 2)) * -1
    del tmp3
    tmp7 = f.aa.vv.transpose((1, 0)).copy() * -2
    tmp7 += tmp5 * 2
    del tmp5
    tmp7 += tmp6
    del tmp6
    t2new.aaaa += einsum(tmp7, (0, 1), t2.aaaa, (2, 3, 1, 4), (3, 2, 0, 4), optimize=True)
    del tmp7
    tmp18 = einsum(tmp17, (0, 1, 2, 3), t2.abab, (4, 1, 5, 3), (4, 0, 5, 2), optimize=True)
    del tmp17
    t2new.aaaa += tmp18
    t2new.aaaa += tmp18.transpose((1, 0, 2, 3)) * -1
    del tmp18
    tmp16 = einsum(t2.aaaa, (0, 1, 2, 3), tmp15, (4, 0, 5, 2), (1, 4, 3, 5), optimize=True) * 2
    del tmp15
    t2new.aaaa += tmp16.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp16.transpose((1, 0, 3, 2))
    del tmp16
    tmp13 = einsum(tmp12, (0, 1, 2, 3), t2.abab, (4, 1, 5, 3), (4, 0, 5, 2), optimize=True) * 0.5
    del tmp12
    t2new.aaaa += tmp13.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp13.transpose((1, 0, 3, 2))
    del tmp13
    tmp20 = einsum(t2.aaaa, (0, 1, 2, 3), tmp19, (4, 0, 5, 2), (1, 4, 3, 5), optimize=True) * 2
    del tmp19
    t2new.aaaa += tmp20.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp20
    del tmp20
    tmp10 = einsum(t2.aaaa, (0, 1, 2, 3), tmp9, (4, 1, 5, 3), (0, 4, 2, 5), optimize=True)
    del tmp9
    t2new.aaaa += tmp10.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp10.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp10.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp10
    t2new.aaaa += tmp10.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp10.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp10.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp10
    del tmp10
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.oooo, (4, 0, 1, 5), (5, 4, 3, 2), optimize=True) * 2
    t2new.aaaa += v.aaaa.ovov.transpose((2, 0, 3, 1))
    t2new.aaaa += v.aaaa.ovov.transpose((2, 0, 1, 3)) * -1
    t2new.aaaa += einsum(v.aaaa.vvvv, (0, 1, 2, 3), t2.aaaa, (4, 5, 3, 1), (5, 4, 2, 0), optimize=True) * -2
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.oooo, (4, 0, 5, 1), (4, 5, 2, 3), optimize=True)
    t2new.abab += einsum(v.aaaa.oovv, (0, 1, 2, 3), t2.abab, (1, 4, 3, 5), (0, 4, 2, 5), optimize=True) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.oovv, (4, 0, 5, 3), (4, 1, 2, 5), optimize=True) * -1
    t2new.abab += v.aabb.ovov.transpose((0, 2, 1, 3))
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5), optimize=True)
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), v.bbaa.oovv, (4, 1, 5, 2), (0, 4, 5, 3), optimize=True) * -1
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.oooo, (4, 1, 5, 0), (5, 4, 3, 2), optimize=True) * -2
    t2new.bbbb += v.bbbb.ovov.transpose((2, 0, 3, 1))
    t2new.bbbb += v.bbbb.ovov.transpose((2, 0, 1, 3)) * -1
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 2, 3, 5), (1, 0, 5, 4), optimize=True) * 2

    return {f"t2new": t2new}
