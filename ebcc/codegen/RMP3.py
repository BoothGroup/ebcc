"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-07-18T09:27:17.240842
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
    Code generated by `albert` 0.0.0 on 2024-07-18T09:27:19.628209.

    Parameters
    ----------
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    e_mp : array
    """

    tmp3 = einsum(v.vvvv, (0, 1, 2, 3), t2, (4, 5, 3, 1), (4, 5, 2, 0), optimize=True)
    tmp1 = t2.transpose((1, 0, 2, 3)).copy()
    tmp1 += t2.transpose((1, 0, 3, 2)) * -1
    tmp4 = v.ovov.transpose((2, 0, 3, 1)).copy() * 0.5
    tmp4 += tmp3.transpose((1, 0, 3, 2)) * 0.5
    del tmp3
    tmp8 = einsum(t2, (0, 1, 2, 3), t2, (4, 5, 3, 2), (4, 1, 0, 5), optimize=True)
    tmp7 = v.ovov.transpose((2, 0, 3, 1)).copy() * -1
    tmp7 += v.oovv.transpose((1, 0, 3, 2)) * 2
    tmp0 = v.ovov.transpose((2, 0, 3, 1)).copy() * 2
    tmp0 += v.oovv.transpose((1, 0, 3, 2)) * -1
    tmp2 = einsum(tmp1, (0, 1, 2, 3), t2, (0, 4, 3, 5), (1, 4, 2, 5), optimize=True)
    del tmp1
    e_mp = einsum(tmp2, (0, 1, 2, 3), tmp0, (0, 1, 2, 3), (), optimize=True) * 4
    del tmp2, tmp0
    tmp5 = tmp4.transpose((0, 1, 3, 2)).copy() * -1
    tmp5 += tmp4 * 2
    del tmp4
    e_mp += einsum(tmp5, (0, 1, 2, 3), t2, (0, 1, 2, 3), (), optimize=True) * 2
    del tmp5
    tmp6 = einsum(t2, (0, 1, 2, 3), t2, (4, 1, 2, 5), (0, 4, 3, 5), optimize=True)
    e_mp += einsum(tmp7, (0, 1, 2, 3), tmp6, (0, 1, 2, 3), (), optimize=True) * -2
    del tmp7, tmp6
    tmp9 = tmp8.transpose((3, 1, 2, 0)).copy() * -0.5
    tmp9 += tmp8.transpose((3, 2, 1, 0))
    del tmp8
    e_mp += einsum(tmp9, (0, 1, 2, 3), v.oooo, (0, 1, 3, 2), (), optimize=True) * 2
    del tmp9

    return e_mp
