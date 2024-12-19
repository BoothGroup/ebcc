"""Code generated by `albert` version 0.0.0.

 * date: 2024-12-19T15:11:05.188308
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

    tmp0 = einsum(t2, (0, 1, 2, 3), t2, (4, 5, 2, 3), (4, 1, 0, 5)) * -1
    e_cc = einsum(v.oooo, (0, 1, 2, 3), tmp0, (0, 2, 3, 1), ()) * 0.125
    del tmp0
    tmp1 = np.copy(np.transpose(v.oovv, (1, 0, 3, 2))) * 2
    tmp1 += einsum(v.ovov, (0, 1, 2, 3), t2, (2, 4, 1, 5), (0, 4, 3, 5)) * -8
    tmp1 += einsum(v.vvvv, (0, 1, 2, 3), t2, (4, 5, 3, 2), (5, 4, 1, 0)) * -1
    e_cc += einsum(t2, (0, 1, 2, 3), tmp1, (0, 1, 2, 3), ()) * 0.125
    del tmp1

    return e_cc

