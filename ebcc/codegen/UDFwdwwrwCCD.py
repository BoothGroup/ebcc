"""Code generated by `albert` version 0.0.0.

 * date: 2024-12-18T11:59:07.730741
 * python version: 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0]
 * albert version: 0.0.0
 * caller: /home/ollie/git/albert/albert/code/einsum.py
 * node: ollie-desktop
 * system: Linux
 * processor: x86_64
 * release: 6.8.0-49-generic
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

    tmp0 = einsum(v.bb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    e_cc = einsum(v.bb.xov, (0, 1, 2), tmp0, (1, 2, 0), ())
    del tmp0
    tmp1 = einsum(t2.aaaa, (0, 1, 2, 3), v.aa.xov, (4, 1, 3), (0, 2, 4))
    tmp1 += einsum(t2.abab, (0, 1, 2, 3), v.bb.xov, (4, 1, 3), (0, 2, 4))
    e_cc += einsum(tmp1, (0, 1, 2), v.aa.xov, (2, 0, 1), ())
    del tmp1

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

    tmp0 = einsum(t2.aaaa, (0, 1, 2, 3), f.aa.oo, (4, 1), (4, 0, 2, 3))
    t2new = Namespace()
    t2new.aaaa = np.copy(np.transpose(tmp0, (1, 0, 3, 2))) * 2
    t2new.aaaa += np.transpose(tmp0, (0, 1, 3, 2)) * -2
    del tmp0
    tmp1 = einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new.aaaa += np.transpose(tmp1, (1, 0, 2, 3)) * 2
    t2new.aaaa += np.transpose(tmp1, (1, 0, 3, 2)) * -2
    del tmp1
    tmp2 = einsum(t2.aaaa, (0, 1, 2, 3), v.aa.xov, (4, 1, 3), (0, 2, 4))
    t2new.aaaa += einsum(tmp2, (0, 1, 2), tmp2, (3, 4, 2), (0, 3, 1, 4)) * 4
    tmp3 = einsum(t2.abab, (0, 1, 2, 3), v.bb.xov, (4, 1, 3), (0, 2, 4))
    t2new.aaaa += einsum(tmp3, (0, 1, 2), tmp3, (3, 4, 2), (3, 0, 4, 1))
    tmp4 = einsum(tmp2, (0, 1, 2), tmp3, (3, 4, 2), (0, 3, 1, 4))
    t2new.aaaa += tmp4 * 2
    t2new.aaaa += np.transpose(tmp4, (1, 0, 3, 2)) * 2
    del tmp4
    tmp5 = np.copy(np.transpose(v.aa.xov, (1, 2, 0))) * 0.5
    tmp5 += tmp2
    tmp5 += tmp3 * 0.5
    t2new.aaaa += einsum(v.aa.xov, (0, 1, 2), tmp5, (3, 4, 0), (3, 1, 4, 2)) * 2
    del tmp5
    tmp6 = np.copy(tmp2)
    del tmp2
    tmp6 += tmp3 * 0.5
    del tmp3
    t2new.aaaa += einsum(tmp6, (0, 1, 2), v.aa.xov, (2, 3, 4), (3, 0, 4, 1)) * 2
    t2new.abab = einsum(f.bb.oo, (0, 1), t2.abab, (2, 1, 3, 4), (2, 0, 3, 4)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), f.aa.oo, (4, 0), (4, 1, 2, 3)) * -1
    t2new.abab += einsum(f.aa.vv, (0, 1), t2.abab, (2, 3, 1, 4), (2, 3, 0, 4))
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), f.bb.vv, (4, 3), (0, 1, 2, 4))
    t2new.abab += einsum(v.aa.xov, (0, 1, 2), v.bb.xov, (0, 3, 4), (1, 3, 2, 4))
    tmp9 = np.copy(np.transpose(v.bb.xov, (1, 2, 0)))
    tmp7 = einsum(v.aa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    tmp9 += tmp7
    tmp8 = einsum(v.bb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    tmp9 += tmp8 * 2
    t2new.abab += einsum(tmp9, (0, 1, 2), tmp6, (3, 4, 2), (3, 0, 4, 1)) * 2
    del tmp6, tmp9
    tmp10 = np.copy(tmp7) * 0.5
    tmp10 += tmp8
    t2new.abab += einsum(v.aa.xov, (0, 1, 2), tmp10, (3, 4, 0), (1, 3, 2, 4)) * 2
    tmp11 = einsum(f.bb.oo, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new.bbbb = np.copy(np.transpose(tmp11, (1, 0, 3, 2))) * 2
    t2new.bbbb += np.transpose(tmp11, (0, 1, 3, 2)) * -2
    del tmp11
    tmp12 = einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new.bbbb += np.transpose(tmp12, (1, 0, 2, 3)) * 2
    t2new.bbbb += np.transpose(tmp12, (1, 0, 3, 2)) * -2
    del tmp12
    t2new.bbbb += einsum(tmp7, (0, 1, 2), tmp7, (3, 4, 2), (3, 0, 4, 1))
    t2new.bbbb += einsum(tmp8, (0, 1, 2), tmp8, (3, 4, 2), (3, 0, 4, 1)) * 4
    tmp13 = einsum(tmp8, (0, 1, 2), tmp7, (3, 4, 2), (3, 0, 4, 1))
    t2new.bbbb += tmp13 * 2
    t2new.bbbb += np.transpose(tmp13, (1, 0, 3, 2)) * 2
    del tmp13
    tmp14 = np.copy(np.transpose(v.bb.xov, (1, 2, 0))) * 0.5
    tmp14 += tmp7 * 0.5
    del tmp7
    tmp14 += tmp8
    del tmp8
    t2new.bbbb += einsum(v.bb.xov, (0, 1, 2), tmp14, (3, 4, 0), (3, 1, 4, 2)) * 2
    del tmp14
    t2new.bbbb += einsum(tmp10, (0, 1, 2), v.bb.xov, (2, 3, 4), (3, 0, 4, 1)) * 2
    del tmp10

    return {"t2new": t2new}

