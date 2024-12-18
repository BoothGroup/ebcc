"""Code generated by `albert` version 0.0.0.

 * date: 2024-12-17T16:22:36.741829
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

    e_cc = einsum(v.ovov, (0, 1, 2, 3), t2, (0, 2, 1, 3), ()) * 2
    e_cc += einsum(t2, (0, 1, 2, 3), v.ovov, (0, 3, 1, 2), ()) * -1
    e_cc += einsum(t1, (0, 1), f.ov, (0, 1), ()) * 2

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

    t1new = np.copy(f.ov)
    t1new += einsum(t1, (0, 1), f.oo, (2, 0), (2, 1)) * -1
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    tmp0 = np.copy(v.ovvv) * -0.5
    tmp0 += np.transpose(v.ovvv, (0, 2, 1, 3))
    t1new += einsum(t2, (0, 1, 2, 3), tmp0, (1, 2, 3, 4), (0, 4)) * 2
    del tmp0
    tmp1 = np.copy(v.ooov) * -0.5
    tmp1 += np.transpose(v.ovoo, (0, 2, 3, 1))
    t1new += einsum(tmp1, (0, 1, 2, 3), t2, (0, 2, 3, 4), (1, 4)) * -2
    del tmp1
    tmp2 = np.copy(np.transpose(v.ovov, (0, 2, 1, 3)))
    tmp2 += v.oovv * -0.5
    t1new += einsum(t1, (0, 1), tmp2, (0, 2, 1, 3), (2, 3)) * 2
    del tmp2
    tmp3 = np.copy(np.transpose(t2, (0, 1, 3, 2))) * -0.5
    tmp3 += t2
    t1new += einsum(f.ov, (0, 1), tmp3, (0, 2, 1, 3), (2, 3)) * 2
    del tmp3
    t2new = einsum(v.vvvv, (0, 1, 2, 3), t2, (4, 5, 1, 3), (4, 5, 0, 2))
    t2new += np.transpose(v.ovov, (0, 2, 1, 3))
    t2new += einsum(t2, (0, 1, 2, 3), v.oooo, (4, 1, 5, 0), (4, 5, 3, 2))
    tmp4 = einsum(v.ooov, (0, 1, 2, 3), t1, (1, 4), (0, 2, 4, 3))
    tmp8 = np.copy(tmp4)
    del tmp4
    tmp5 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 5, 2))
    tmp8 += tmp5
    del tmp5
    tmp6 = np.copy(np.transpose(t2, (0, 1, 3, 2))) * -1
    tmp6 += t2 * 2
    tmp7 = einsum(tmp6, (0, 1, 2, 3), v.ovov, (4, 5, 0, 2), (4, 1, 5, 3))
    del tmp6
    tmp8 += np.transpose(tmp7, (1, 0, 3, 2)) * -1
    del tmp7
    t2new += tmp8 * -1
    t2new += np.transpose(tmp8, (1, 0, 3, 2)) * -1
    del tmp8
    tmp9 = einsum(t2, (0, 1, 2, 3), f.oo, (4, 1), (4, 0, 2, 3))
    tmp11 = np.copy(tmp9)
    del tmp9
    tmp10 = einsum(v.oovv, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 5, 2))
    tmp11 += tmp10
    del tmp10
    t2new += np.transpose(tmp11, (0, 1, 3, 2)) * -1
    t2new += np.transpose(tmp11, (1, 0, 2, 3)) * -1
    del tmp11
    tmp12 = einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    tmp14 = np.copy(tmp12)
    del tmp12
    tmp13 = einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    tmp14 += tmp13
    del tmp13
    t2new += np.transpose(tmp14, (0, 1, 3, 2))
    t2new += np.transpose(tmp14, (1, 0, 2, 3))
    del tmp14

    return {"t1new": t1new, "t2new": t2new}

