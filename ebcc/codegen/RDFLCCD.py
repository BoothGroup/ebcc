"""Code generated by `albert` version 0.0.0.

 * date: 2024-12-18T00:18:12.591121
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

    tmp0 = np.copy(np.transpose(t2, (1, 0, 2, 3))) * 2
    tmp0 += np.transpose(t2, (1, 0, 3, 2)) * -1
    tmp1 = einsum(v.xov, (0, 1, 2), tmp0, (1, 3, 4, 2), (3, 4, 0)) * 0.5
    del tmp0
    e_cc = einsum(tmp1, (0, 1, 2), v.xov, (2, 0, 1), ()) * 2
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

    tmp0 = einsum(v.xoo, (0, 1, 2), v.xoo, (0, 3, 4), (1, 3, 4, 2))
    t2new = einsum(tmp0, (0, 1, 2, 3), t2, (1, 3, 4, 5), (2, 0, 4, 5))
    del tmp0
    t2new += einsum(v.xov, (0, 1, 2), v.xov, (0, 3, 4), (3, 1, 4, 2))
    tmp1 = einsum(v.xvv, (0, 1, 2), v.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new += einsum(t2, (0, 1, 2, 3), tmp1, (4, 2, 5, 3), (0, 1, 5, 4))
    del tmp1
    tmp2 = einsum(v.xvv, (0, 1, 2), v.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp3 = einsum(tmp2, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 5, 2))
    tmp7 = np.copy(tmp3)
    del tmp3
    tmp4 = np.copy(np.transpose(t2, (0, 1, 3, 2))) * 2
    tmp4 += t2 * -1
    tmp5 = einsum(v.xov, (0, 1, 2), tmp4, (1, 3, 4, 2), (3, 4, 0)) * 0.5
    del tmp4
    tmp6 = einsum(v.xov, (0, 1, 2), tmp5, (3, 4, 0), (1, 3, 2, 4)) * 2
    del tmp5
    tmp7 += np.transpose(tmp6, (1, 0, 3, 2)) * -1
    del tmp6
    t2new += tmp7 * -1
    t2new += np.transpose(tmp7, (1, 0, 3, 2)) * -1
    del tmp7
    tmp8 = einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    tmp10 = np.copy(tmp8)
    del tmp8
    tmp9 = einsum(t2, (0, 1, 2, 3), tmp2, (4, 1, 5, 2), (0, 4, 3, 5))
    del tmp2
    tmp10 += tmp9
    del tmp9
    t2new += np.transpose(tmp10, (0, 1, 3, 2)) * -1
    t2new += np.transpose(tmp10, (1, 0, 2, 3)) * -1
    del tmp10
    tmp11 = einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new += np.transpose(tmp11, (0, 1, 3, 2))
    t2new += np.transpose(tmp11, (1, 0, 2, 3))
    del tmp11

    return {"t2new": t2new}

