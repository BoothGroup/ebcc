"""Code generated by `albert` version 0.0.0.

 * date: 2024-12-17T16:11:07.723784
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

    e_cc = einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 2, 3), ()) * 0.25

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

    t2new = einsum(t2, (0, 1, 2, 3), v.oooo, (4, 5, 0, 1), (4, 5, 2, 3)) * 0.5
    t2new += v.oovv
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 5, 2, 3), (0, 1, 4, 5)) * 0.5
    tmp0 = einsum(t2, (0, 1, 2, 3), f.vv, (4, 3), (0, 1, 4, 2))
    t2new += np.transpose(tmp0, (1, 0, 2, 3))
    t2new += np.transpose(tmp0, (1, 0, 3, 2)) * -1
    del tmp0
    tmp1 = einsum(t2, (0, 1, 2, 3), f.oo, (4, 1), (4, 0, 2, 3))
    t2new += np.transpose(tmp1, (0, 1, 3, 2)) * -1
    t2new += np.transpose(tmp1, (1, 0, 3, 2))
    del tmp1
    tmp2 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    t2new += tmp2 * -1
    t2new += np.transpose(tmp2, (0, 1, 3, 2))
    t2new += np.transpose(tmp2, (1, 0, 2, 3))
    t2new += np.transpose(tmp2, (1, 0, 3, 2)) * -1
    del tmp2

    return {"t2new": t2new}
