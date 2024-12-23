"""Code generated by `albert` version 0.0.0.

 * date: 2024-12-17T16:12:07.457835
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

    e_cc = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(v.aaaa.ovov, (0, 1, 2, 3), t2.aaaa, (0, 2, 3, 1), ()) * -1
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 2), ()) * -1

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
    t2new.aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.oooo, (4, 0, 5, 1), (4, 5, 2, 3)) * 2
    tmp1 = einsum(t2.aaaa, (0, 1, 2, 3), f.aa.vv, (4, 3), (0, 1, 4, 2))
    t2new.aaaa += np.transpose(tmp1, (1, 0, 2, 3)) * 2
    t2new.aaaa += np.transpose(tmp1, (1, 0, 3, 2)) * -2
    del tmp1
    t2new.aaaa += np.transpose(v.aaaa.ovov, (0, 2, 1, 3))
    tmp2 = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new.aaaa += np.transpose(tmp2, (1, 0, 3, 2))
    tmp3 = np.copy(np.transpose(v.aaaa.ovov, (0, 2, 1, 3)))
    tmp3 += v.aaaa.oovv * -1
    tmp4 = einsum(tmp3, (0, 1, 2, 3), t2.aaaa, (4, 0, 5, 2), (4, 1, 5, 3)) * 2
    t2new.aaaa += np.transpose(tmp4, (1, 0, 3, 2))
    t2new.aaaa += np.transpose(v.aaaa.ovov, (0, 2, 3, 1)) * -1
    t2new.aaaa += np.transpose(tmp2, (1, 0, 2, 3)) * -1
    t2new.aaaa += np.transpose(tmp4, (1, 0, 2, 3)) * -1
    t2new.aaaa += np.transpose(tmp2, (0, 1, 3, 2)) * -1
    t2new.aaaa += np.transpose(tmp4, (0, 1, 3, 2)) * -1
    t2new.aaaa += tmp2
    del tmp2
    t2new.aaaa += tmp4
    del tmp4
    t2new.aaaa += einsum(v.aaaa.vvvv, (0, 1, 2, 3), t2.aaaa, (4, 5, 1, 3), (4, 5, 0, 2)) * 2
    t2new.abab = einsum(f.bb.oo, (0, 1), t2.abab, (2, 1, 3, 4), (2, 0, 3, 4)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.oooo, (4, 0, 5, 1), (4, 5, 2, 3))
    t2new.abab += einsum(f.aa.oo, (0, 1), t2.abab, (1, 2, 3, 4), (0, 2, 3, 4)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), f.aa.vv, (4, 2), (0, 1, 4, 3))
    t2new.abab += einsum(f.bb.vv, (0, 1), t2.abab, (2, 3, 4, 1), (2, 3, 4, 0))
    t2new.abab += einsum(v.aabb.ovov, (0, 1, 2, 3), t2.bbbb, (4, 2, 5, 3), (0, 4, 1, 5)) * 2
    t2new.abab += np.transpose(v.aabb.ovov, (0, 2, 1, 3))
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp3, (0, 4, 2, 5), (4, 1, 5, 3))
    del tmp3
    t2new.abab += einsum(v.aabb.vvoo, (0, 1, 2, 3), t2.abab, (4, 3, 1, 5), (4, 2, 0, 5)) * -1
    t2new.abab += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ovov, (1, 3, 4, 5), (0, 4, 2, 5)) * 2
    tmp5 = np.copy(np.transpose(v.bbbb.ovov, (0, 2, 1, 3)))
    tmp5 += v.bbbb.oovv * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), tmp5, (1, 4, 3, 5), (0, 4, 2, 5))
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.oovv, (4, 0, 5, 3), (4, 1, 2, 5)) * -1
    t2new.abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    tmp6 = einsum(f.bb.oo, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new.bbbb = np.copy(np.transpose(tmp6, (1, 0, 3, 2))) * 2
    t2new.bbbb += np.transpose(tmp6, (0, 1, 3, 2)) * -2
    del tmp6
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.oooo, (4, 0, 5, 1), (4, 5, 2, 3)) * 2
    tmp7 = einsum(t2.bbbb, (0, 1, 2, 3), f.bb.vv, (4, 3), (0, 1, 4, 2))
    t2new.bbbb += np.transpose(tmp7, (1, 0, 2, 3)) * 2
    t2new.bbbb += np.transpose(tmp7, (1, 0, 3, 2)) * -2
    del tmp7
    tmp8 = einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 4, 3, 5))
    t2new.bbbb += np.transpose(tmp8, (1, 0, 3, 2))
    t2new.bbbb += np.transpose(v.bbbb.ovov, (0, 2, 1, 3))
    tmp9 = einsum(t2.bbbb, (0, 1, 2, 3), tmp5, (1, 4, 3, 5), (0, 4, 2, 5)) * 2
    del tmp5
    t2new.bbbb += np.transpose(tmp9, (1, 0, 3, 2))
    t2new.bbbb += np.transpose(v.bbbb.ovov, (0, 2, 3, 1)) * -1
    t2new.bbbb += np.transpose(tmp8, (1, 0, 2, 3)) * -1
    t2new.bbbb += np.transpose(tmp9, (1, 0, 2, 3)) * -1
    t2new.bbbb += np.transpose(tmp8, (0, 1, 3, 2)) * -1
    t2new.bbbb += np.transpose(tmp9, (0, 1, 3, 2)) * -1
    t2new.bbbb += tmp8
    del tmp8
    t2new.bbbb += tmp9
    del tmp9
    t2new.bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5)) * 2

    return {"t2new": t2new}

