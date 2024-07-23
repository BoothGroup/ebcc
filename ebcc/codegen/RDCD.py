"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-07-18T10:47:22.922801
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
    Code generated by `albert` 0.0.0 on 2024-07-18T10:47:23.079638.

    Parameters
    ----------
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    e_cc : float
        Coupled cluster energy.
    """

    e_cc = einsum(v.ovov, (0, 1, 2, 3), t2, (0, 2, 1, 3), (), optimize=True) * 2
    e_cc += einsum(v.ovov, (0, 1, 2, 3), t2, (0, 2, 3, 1), (), optimize=True) * -1

    return e_cc

def update_amps(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-07-18T10:47:25.024756.

    Parameters
    ----------
    f : array
        Fock matrix.
    t2 : array
        T2 amplitudes.
    v : array
        Electron repulsion integrals.

    Returns
    -------
    t2new : array
        Updated T2 residuals.
    """

    tmp6 = v.ovov.transpose((2, 0, 1, 3)).copy() * -0.5
    tmp6 += v.ovov.transpose((2, 0, 3, 1))
    tmp0 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5), optimize=True)
    t2new = tmp0.copy() * -1
    tmp9 = einsum(tmp6, (0, 1, 2, 3), t2, (0, 4, 2, 3), (4, 1), optimize=True) * 2
    tmp7 = einsum(t2, (0, 1, 2, 3), tmp6, (0, 1, 4, 3), (2, 4), optimize=True)
    del tmp6
    tmp1 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5), optimize=True)
    t2new += tmp1 * 2
    tmp5 = einsum(t2, (0, 1, 2, 3), tmp0, (4, 1, 5, 3), (0, 4, 2, 5), optimize=True)
    tmp10 = einsum(tmp9, (0, 1), t2, (1, 2, 3, 4), (2, 0, 3, 4), optimize=True) * 0.5
    del tmp9
    tmp8 = einsum(t2, (0, 1, 2, 3), tmp7, (4, 2), (0, 1, 3, 4), optimize=True)
    del tmp7
    tmp2 = einsum(t2, (0, 1, 2, 3), f.vv, (4, 3), (0, 1, 4, 2), optimize=True)
    tmp3 = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (0, 4, 3, 5), optimize=True)
    tmp12 = einsum(t2, (0, 1, 2, 3), f.oo, (4, 1), (4, 0, 2, 3), optimize=True)
    t2new += tmp12.transpose((1, 0, 2, 3)) * -1
    t2new += tmp12.transpose((0, 1, 3, 2)) * -1
    del tmp12
    tmp13 = v.ovov.transpose((2, 0, 3, 1)).copy() * 0.5
    tmp13 += v.oovv.transpose((1, 0, 3, 2)) * -0.25
    tmp13 += tmp1
    del tmp1
    t2new += einsum(tmp13, (0, 1, 2, 3), t2, (1, 4, 3, 5), (0, 4, 2, 5), optimize=True) * 4
    del tmp13
    tmp14 = v.ovov.transpose((2, 0, 3, 1)).copy() * -1
    tmp14 += tmp0
    del tmp0
    t2new += einsum(t2, (0, 1, 2, 3), tmp14, (4, 0, 5, 3), (4, 1, 5, 2), optimize=True)
    del tmp14
    tmp11 = tmp5.copy() * 2
    del tmp5
    tmp11 += tmp8.transpose((1, 0, 2, 3))
    del tmp8
    tmp11 += tmp10.transpose((0, 1, 3, 2))
    del tmp10
    t2new += tmp11.transpose((1, 0, 3, 2)) * -1
    t2new += tmp11 * -1
    del tmp11
    tmp4 = tmp2.copy()
    del tmp2
    tmp4 += tmp3 * -1
    del tmp3
    t2new += tmp4.transpose((1, 0, 2, 3))
    t2new += tmp4.transpose((0, 1, 3, 2))
    del tmp4
    t2new += v.ovov.transpose((2, 0, 3, 1))
    t2new += einsum(v.oooo, (0, 1, 2, 3), t2, (1, 2, 4, 5), (3, 0, 5, 4), optimize=True)
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5), optimize=True) * -1
    t2new += einsum(v.vvvv, (0, 1, 2, 3), t2, (4, 5, 1, 3), (5, 4, 2, 0), optimize=True)

    return {f"t2new": t2new}
