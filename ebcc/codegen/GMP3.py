"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-08-09T21:27:40.383646
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
    Code generated by `albert` 0.0.0 on 2024-08-09T21:27:40.838811.

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

    tmp0 = einsum(t2, (0, 1, 2, 3), t2, (4, 5, 2, 3), (0, 4, 5, 1))
    e_mp = einsum(v.oooo, (0, 1, 2, 3), tmp0, (3, 0, 1, 2), ()) * -0.125
    del tmp0
    tmp1 = np.copy(v.oovv) * 2
    tmp1 += einsum(v.ovov, (0, 1, 2, 3), t2, (4, 2, 5, 1), (0, 4, 3, 5)) * -8
    tmp1 += einsum(v.vvvv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (4, 5, 0, 1))
    e_mp += einsum(tmp1, (0, 1, 2, 3), t2, (0, 1, 2, 3), ()) * 0.125
    del tmp1

    return e_mp

