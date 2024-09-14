"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-08-09T22:08:58.063564
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
    Code generated by `albert` 0.0.0 on 2024-08-09T22:08:58.781780.

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

    tmp0 = einsum(t2.aaaa, (0, 1, 2, 3), v.baa.xov, (4, 1, 3), (0, 2, 4))
    e_cc = einsum(tmp0, (0, 1, 2), v.baa.xov, (2, 0, 1), ())
    del tmp0
    tmp1 = einsum(v.baa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    tmp1 += einsum(v.bbb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    e_cc += einsum(tmp1, (0, 1, 2), v.bbb.xov, (2, 0, 1), ())
    del tmp1

    return e_cc

def update_amps(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T22:09:16.572318.

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
    tmp23 = einsum(v.baa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    tmp24 = einsum(v.bbb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    tmp0 = einsum(t2.aaaa, (0, 1, 2, 3), v.baa.xov, (4, 1, 3), (0, 2, 4))
    tmp1 = einsum(v.bbb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (3, 4, 0))
    t2new.aaaa = einsum(tmp1, (0, 1, 2), tmp1, (3, 4, 2), (3, 0, 4, 1)) * 0.5
    tmp40 = tmp23.copy() * 0.5
    tmp40 += tmp24
    tmp2 = tmp0.copy() * 2
    tmp2 += tmp1
    tmp47 = einsum(v.bbb.xov, (0, 1, 2), tmp40, (1, 3, 0), (3, 2))
    tmp41 = einsum(v.bbb.xov, (0, 1, 2), tmp40, (3, 2, 0), (3, 1))
    del tmp40
    tmp3 = einsum(tmp2, (0, 1, 2), v.baa.xov, (2, 3, 1), (3, 0)) * 0.5
    tmp10 = einsum(v.baa.xov, (0, 1, 2), tmp2, (1, 3, 0), (2, 3)) * 0.5
    del tmp2
    tmp55 = einsum(v.bbb.xvv, (0, 1, 2), v.bbb.xvv, (0, 3, 4), (1, 3, 4, 2))
    tmp48 = f.bb.vv.copy() * -1
    tmp48 += tmp47
    del tmp47
    tmp57 = v.bbb.xov.transpose((1, 2, 0)).copy()
    tmp57 += tmp24
    tmp44 = einsum(v.bbb.xoo, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 3, 4, 2))
    tmp53 = v.bbb.xov.transpose((1, 2, 0)).copy()
    tmp53 += tmp23
    tmp53 += tmp24 * 2
    tmp42 = f.bb.oo.copy()
    tmp42 += tmp41
    del tmp41
    tmp38 = einsum(v.bbb.xvv, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (3, 4, 1, 2))
    t2new.abab = einsum(tmp38, (0, 1, 2, 3), t2.abab, (4, 1, 5, 3), (4, 0, 5, 2)) * -1
    tmp25 = tmp23.copy()
    tmp25 += tmp24 * 2
    tmp34 = einsum(v.baa.xvv, (0, 1, 2), v.bbb.xvv, (0, 3, 4), (1, 2, 3, 4))
    tmp28 = tmp0.copy()
    tmp28 += tmp1 * 0.5
    tmp8 = einsum(v.baa.xoo, (0, 1, 2), v.baa.xvv, (0, 3, 4), (1, 2, 3, 4))
    t2new.abab += einsum(tmp8, (0, 1, 2, 3), t2.abab, (1, 4, 3, 5), (0, 4, 2, 5)) * -1
    tmp16 = v.baa.xov.transpose((1, 2, 0)).copy() * 2
    tmp16 += tmp1
    tmp6 = einsum(v.baa.xoo, (0, 1, 2), v.baa.xoo, (0, 3, 4), (1, 3, 4, 2))
    tmp4 = f.aa.oo.copy()
    tmp4 += tmp3.transpose((1, 0))
    del tmp3
    tmp11 = f.aa.vv.copy() * -1
    tmp11 += tmp10.transpose((1, 0))
    del tmp10
    tmp21 = v.baa.xov.transpose((1, 2, 0)).copy()
    tmp21 += tmp0
    tmp19 = einsum(v.baa.xvv, (0, 1, 2), v.baa.xvv, (0, 3, 4), (3, 1, 2, 4))
    tmp56 = einsum(tmp55, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 3), (4, 5, 2, 0))
    del tmp55
    t2new.bbbb = tmp56.transpose((0, 1, 3, 2)).copy() * -1
    t2new.bbbb += tmp56.transpose((0, 1, 3, 2)) * -1
    del tmp56
    tmp49 = einsum(t2.bbbb, (0, 1, 2, 3), tmp48, (4, 3), (0, 1, 2, 4)) * -2
    del tmp48
    t2new.bbbb += tmp49.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp49.transpose((1, 0, 3, 2))
    del tmp49
    tmp58 = einsum(tmp24, (0, 1, 2), tmp57, (3, 4, 2), (0, 3, 1, 4)) * 2
    del tmp57
    t2new.bbbb += tmp58.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp58
    del tmp58
    tmp45 = einsum(t2.bbbb, (0, 1, 2, 3), tmp44, (4, 0, 5, 1), (5, 4, 2, 3))
    del tmp44
    t2new.bbbb += tmp45.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp45.transpose((0, 1, 3, 2)) * -1
    del tmp45
    tmp60 = v.bbb.xov.transpose((1, 2, 0)).copy() * 2
    tmp60 += tmp23
    t2new.bbbb += einsum(tmp60, (0, 1, 2), tmp23, (3, 4, 2), (3, 0, 1, 4)) * -0.5
    del tmp60
    tmp54 = einsum(v.bbb.xov, (0, 1, 2), tmp53, (3, 4, 0), (3, 1, 4, 2))
    del tmp53
    t2new.bbbb += tmp54.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp54.transpose((1, 0, 3, 2))
    del tmp54
    tmp43 = einsum(tmp42, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4)) * -1
    del tmp42
    t2new.bbbb += tmp43.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp43.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp43.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp43.transpose((0, 1, 3, 2)) * -1
    del tmp43
    tmp59 = v.bbb.xov.transpose((1, 2, 0)).copy()
    tmp59 += tmp23 * 0.5
    t2new.bbbb += einsum(tmp59, (0, 1, 2), tmp23, (3, 4, 2), (3, 0, 4, 1))
    del tmp59
    tmp46 = einsum(t2.bbbb, (0, 1, 2, 3), tmp38, (4, 1, 5, 3), (0, 4, 2, 5))
    del tmp38
    t2new.bbbb += tmp46.transpose((0, 1, 3, 2)) * 2
    t2new.bbbb += tmp46 * -2
    t2new.bbbb += tmp46.transpose((1, 0, 2, 3)) * 2
    t2new.bbbb += tmp46.transpose((1, 0, 3, 2)) * -2
    del tmp46
    tmp52 = einsum(tmp23, (0, 1, 2), tmp24, (3, 4, 2), (0, 3, 1, 4))
    t2new.bbbb += tmp52.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp52.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp52.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp52
    t2new.bbbb += tmp52.transpose((1, 0, 2, 3)) * -1
    t2new.bbbb += tmp52.transpose((0, 1, 3, 2)) * -1
    t2new.bbbb += tmp52.transpose((1, 0, 3, 2))
    t2new.bbbb += tmp52
    del tmp52
    tmp50 = einsum(tmp23, (0, 1, 2), tmp23, (3, 4, 2), (0, 3, 1, 4))
    t2new.bbbb += tmp50.transpose((0, 1, 3, 2)) * -0.5
    t2new.bbbb += tmp50.transpose((1, 0, 3, 2)) * 0.5
    del tmp50
    tmp51 = einsum(tmp24, (0, 1, 2), tmp24, (3, 4, 2), (0, 3, 1, 4))
    t2new.bbbb += tmp51.transpose((0, 1, 3, 2)) * -2
    t2new.bbbb += tmp51 * 2
    del tmp51
    tmp36 = tmp0.copy() * 2
    tmp36 += tmp1
    tmp39 = einsum(v.baa.xvv, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (3, 4, 1, 2))
    t2new.abab += einsum(tmp39, (0, 1, 2, 3), t2.abab, (4, 1, 3, 5), (4, 0, 2, 5)) * -1
    del tmp39
    tmp27 = einsum(v.baa.xoo, (0, 1, 2), v.bbb.xoo, (0, 3, 4), (1, 2, 3, 4))
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp27, (4, 0, 5, 1), (4, 5, 2, 3))
    del tmp27
    tmp26 = f.bb.oo.copy()
    tmp26 += einsum(v.bbb.xov, (0, 1, 2), tmp25, (3, 2, 0), (1, 3)) * 0.5
    t2new.abab += einsum(tmp26, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1
    del tmp26
    tmp33 = v.baa.xov.transpose((1, 2, 0)).copy() * 2
    tmp33 += tmp0 * 2
    tmp33 += tmp1
    t2new.abab += einsum(tmp33, (0, 1, 2), tmp25, (3, 4, 2), (0, 3, 1, 4)) * 0.5
    del tmp33
    tmp35 = einsum(tmp34, (0, 1, 2, 3), t2.abab, (4, 5, 1, 3), (4, 5, 0, 2))
    del tmp34
    t2new.abab += tmp35 * 0.5
    t2new.abab += tmp35 * 0.5
    del tmp35
    tmp31 = einsum(v.baa.xoo, (0, 1, 2), v.bbb.xvv, (0, 3, 4), (1, 2, 3, 4))
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp31, (4, 0, 5, 3), (4, 1, 2, 5)) * -1
    del tmp31
    tmp37 = v.bbb.xov.transpose((1, 2, 0)).copy()
    tmp37 += tmp23 * 0.5
    del tmp23
    tmp37 += tmp24
    del tmp24
    t2new.abab += einsum(tmp37, (0, 1, 2), tmp36, (3, 4, 2), (3, 0, 4, 1))
    del tmp37, tmp36
    tmp32 = f.bb.vv.copy() * -2
    tmp32 += einsum(v.bbb.xov, (0, 1, 2), tmp25, (1, 3, 0), (2, 3))
    del tmp25
    t2new.abab += einsum(tmp32, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -0.5
    del tmp32
    tmp29 = f.aa.oo.copy()
    tmp29 += einsum(v.baa.xov, (0, 1, 2), tmp28, (3, 2, 0), (1, 3))
    t2new.abab += einsum(tmp29, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1
    del tmp29
    tmp30 = f.aa.vv.copy() * -1
    tmp30 += einsum(tmp28, (0, 1, 2), v.baa.xov, (2, 0, 3), (3, 1))
    del tmp28
    t2new.abab += einsum(tmp30, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1
    del tmp30
    tmp9 = einsum(tmp8, (0, 1, 2, 3), t2.aaaa, (4, 1, 5, 3), (4, 0, 5, 2))
    del tmp8
    t2new.aaaa += tmp9.transpose((0, 1, 3, 2)) * 2
    t2new.aaaa += tmp9 * -2
    t2new.aaaa += tmp9.transpose((1, 0, 2, 3)) * 2
    t2new.aaaa += tmp9.transpose((1, 0, 3, 2)) * -2
    del tmp9
    tmp17 = einsum(tmp1, (0, 1, 2), tmp16, (3, 4, 2), (3, 0, 4, 1)) * 0.5
    del tmp16
    t2new.aaaa += tmp17.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp17.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp17.transpose((0, 1, 3, 2)) * -1
    del tmp17
    tmp14 = einsum(tmp0, (0, 1, 2), tmp1, (3, 4, 2), (0, 3, 1, 4))
    t2new.aaaa += tmp14.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp14.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp14.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp14
    t2new.aaaa += tmp14.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp14.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp14.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp14
    del tmp14
    tmp7 = einsum(tmp6, (0, 1, 2, 3), t2.aaaa, (1, 3, 4, 5), (2, 0, 4, 5))
    del tmp6
    t2new.aaaa += tmp7.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp7.transpose((0, 1, 3, 2)) * -1
    del tmp7
    tmp5 = einsum(tmp4, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -1
    del tmp4
    t2new.aaaa += tmp5.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp5.transpose((1, 0, 3, 2))
    t2new.aaaa += tmp5.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp5.transpose((0, 1, 3, 2)) * -1
    del tmp5
    tmp15 = v.baa.xov.transpose((1, 2, 0)).copy()
    tmp15 += tmp0 * 2
    tmp15 += tmp1
    del tmp1
    t2new.aaaa += einsum(v.baa.xov, (0, 1, 2), tmp15, (3, 4, 0), (1, 3, 2, 4))
    del tmp15
    tmp12 = einsum(t2.aaaa, (0, 1, 2, 3), tmp11, (4, 3), (0, 1, 2, 4)) * -2
    del tmp11
    t2new.aaaa += tmp12.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp12.transpose((1, 0, 3, 2))
    del tmp12
    tmp13 = einsum(tmp0, (0, 1, 2), tmp0, (3, 4, 2), (0, 3, 1, 4))
    t2new.aaaa += tmp13.transpose((0, 1, 3, 2)) * -2
    t2new.aaaa += tmp13 * 2
    del tmp13
    tmp22 = einsum(tmp21, (0, 1, 2), tmp0, (3, 4, 2), (0, 3, 1, 4)) * 2
    del tmp21
    t2new.aaaa += tmp22.transpose((1, 0, 2, 3)) * -1
    t2new.aaaa += tmp22.transpose((1, 0, 3, 2))
    del tmp22
    tmp20 = einsum(t2.aaaa, (0, 1, 2, 3), tmp19, (4, 2, 5, 3), (0, 1, 5, 4))
    del tmp19
    t2new.aaaa += tmp20.transpose((0, 1, 3, 2)) * -1
    t2new.aaaa += tmp20.transpose((0, 1, 3, 2)) * -1
    del tmp20
    tmp18 = v.baa.xov.transpose((1, 2, 0)).copy()
    tmp18 += tmp0 * 2
    del tmp0
    t2new.aaaa += einsum(tmp18, (0, 1, 2), v.baa.xov, (2, 3, 4), (3, 0, 1, 4)) * -1
    del tmp18
    t2new.abab += einsum(v.bbb.xov, (0, 1, 2), v.baa.xov, (0, 3, 4), (3, 1, 4, 2))

    return {f"t2new": t2new}

