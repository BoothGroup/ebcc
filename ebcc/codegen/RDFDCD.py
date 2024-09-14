"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-08-09T22:08:51.581015
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
    Code generated by `albert` 0.0.0 on 2024-08-09T22:08:52.136786.

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

    tmp0 = t2.transpose((0, 1, 3, 2)).copy() * 2
    tmp0 += t2 * -1
    tmp1 = einsum(v.xov, (0, 1, 2), tmp0, (1, 3, 4, 2), (3, 4, 0)) * 0.5
    del tmp0
    e_cc = einsum(tmp1, (0, 1, 2), v.xov, (2, 0, 1), ()) * 2
    del tmp1

    return e_cc

def update_amps(f=None, t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T22:08:56.535754.

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

    tmp15 = t2.transpose((0, 1, 3, 2)).copy() * 2
    tmp15 += t2 * -1
    tmp7 = t2.transpose((0, 1, 3, 2)).copy() * 2
    tmp7 += t2 * -1
    tmp16 = einsum(v.xov, (0, 1, 2), tmp15, (1, 3, 4, 2), (3, 4, 0))
    del tmp15
    tmp5 = einsum(v.xvv, (0, 1, 2), v.xoo, (0, 3, 4), (3, 4, 1, 2))
    tmp8 = einsum(v.xov, (0, 1, 2), tmp7, (1, 3, 4, 2), (3, 4, 0))
    del tmp7
    tmp17 = einsum(tmp16, (0, 1, 2), v.xov, (2, 0, 3), (3, 1)) * 0.5
    tmp19 = einsum(tmp16, (0, 1, 2), v.xov, (2, 3, 1), (3, 0))
    del tmp16
    tmp3 = einsum(v.xov, (0, 1, 2), t2, (3, 1, 4, 2), (3, 4, 0))
    t2new = einsum(tmp3, (0, 1, 2), tmp3, (3, 4, 2), (0, 3, 1, 4)) * 4
    tmp2 = einsum(v.xov, (0, 1, 2), t2, (3, 1, 2, 4), (3, 4, 0))
    t2new += einsum(tmp2, (0, 1, 2), tmp2, (3, 4, 2), (0, 3, 1, 4))
    tmp6 = einsum(tmp5, (0, 1, 2, 3), t2, (4, 1, 5, 3), (4, 0, 5, 2))
    tmp9 = einsum(tmp8, (0, 1, 2), v.xov, (2, 3, 4), (3, 0, 4, 1))
    del tmp8
    tmp12 = einsum(tmp5, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 5, 2))
    del tmp5
    tmp11 = einsum(t2, (0, 1, 2, 3), f.vv, (4, 3), (0, 1, 4, 2))
    tmp18 = einsum(tmp17, (0, 1), t2, (2, 3, 0, 4), (2, 3, 4, 1))
    del tmp17
    tmp20 = einsum(tmp19, (0, 1), t2, (2, 0, 3, 4), (2, 1, 4, 3)) * 0.5
    del tmp19
    tmp14 = einsum(tmp3, (0, 1, 2), tmp2, (3, 4, 2), (0, 3, 1, 4))
    del tmp3, tmp2
    tmp4 = einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new += tmp4.transpose((1, 0, 2, 3)) * -1
    t2new += tmp4.transpose((0, 1, 3, 2)) * -1
    del tmp4
    tmp1 = einsum(v.xvv, (0, 1, 2), v.xvv, (0, 3, 4), (3, 1, 2, 4))
    t2new += einsum(tmp1, (0, 1, 2, 3), t2, (4, 5, 3, 1), (4, 5, 0, 2))
    del tmp1
    tmp10 = tmp6.copy()
    del tmp6
    tmp10 += tmp9.transpose((1, 0, 3, 2)) * -1
    del tmp9
    t2new += tmp10.transpose((1, 0, 3, 2)) * -1
    t2new += tmp10 * -1
    del tmp10
    tmp0 = einsum(v.xoo, (0, 1, 2), v.xoo, (0, 3, 4), (1, 3, 4, 2))
    t2new += einsum(t2, (0, 1, 2, 3), tmp0, (4, 0, 5, 1), (5, 4, 2, 3))
    del tmp0
    tmp13 = tmp11.copy()
    del tmp11
    tmp13 += tmp12 * -1
    del tmp12
    t2new += tmp13.transpose((1, 0, 2, 3))
    t2new += tmp13.transpose((0, 1, 3, 2))
    del tmp13
    tmp21 = tmp14.copy() * 2
    del tmp14
    tmp21 += tmp18.transpose((1, 0, 2, 3))
    del tmp18
    tmp21 += tmp20.transpose((0, 1, 3, 2))
    del tmp20
    t2new += tmp21.transpose((1, 0, 3, 2)) * -1
    t2new += tmp21 * -1
    del tmp21
    t2new += einsum(v.xov, (0, 1, 2), v.xov, (0, 3, 4), (3, 1, 4, 2))

    return {f"t2new": t2new}

