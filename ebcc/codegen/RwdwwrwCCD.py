"""Code generated by `albert` version 0.0.0.

 * date: 2024-12-18T11:34:01.402919
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

    e_cc = einsum(v.ovov, (0, 1, 2, 3), t2, (0, 2, 1, 3), ()) * 2
    e_cc += einsum(t2, (0, 1, 2, 3), v.ovov, (0, 3, 1, 2), ()) * -1

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

    t2new = np.copy(np.transpose(v.ovov, (0, 2, 1, 3)))
    tmp0 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new += np.transpose(tmp0, (1, 0, 3, 2)) * 2
    tmp1 = einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 3, 5), (4, 0, 5, 1))
    t2new += np.transpose(tmp1, (1, 0, 3, 2)) * -1
    tmp2 = einsum(t2, (0, 1, 2, 3), tmp1, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new += tmp2 * -2
    t2new += np.transpose(tmp2, (1, 0, 3, 2)) * -2
    del tmp2
    tmp3 = einsum(t2, (0, 1, 2, 3), f.vv, (4, 3), (0, 1, 4, 2))
    t2new += np.transpose(tmp3, (0, 1, 3, 2))
    t2new += np.transpose(tmp3, (1, 0, 2, 3))
    del tmp3
    tmp4 = einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new += np.transpose(tmp4, (0, 1, 3, 2)) * -1
    t2new += np.transpose(tmp4, (1, 0, 2, 3)) * -1
    del tmp4
    tmp5 = np.copy(np.transpose(v.ovov, (0, 2, 1, 3))) * 0.5
    tmp5 += tmp0
    del tmp0
    t2new += einsum(tmp5, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 5, 2)) * 4
    del tmp5
    tmp6 = np.copy(np.transpose(v.ovov, (0, 2, 1, 3))) * -1
    tmp6 += tmp1
    del tmp1
    t2new += einsum(t2, (0, 1, 2, 3), tmp6, (4, 1, 5, 2), (0, 4, 3, 5))
    del tmp6

    return {"t2new": t2new}

