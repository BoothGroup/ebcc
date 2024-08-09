"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-08-09T21:27:20.832574
  * python version: 3.10.14 (main, Jul 16 2024, 19:03:10) [GCC 11.4.0]
  * albert version: 0.0.0
  * caller: /opt/hostedtoolcache/Python/3.10.14/x64/lib/python3.10/site-packages/albert/codegen/einsum.py
  * node: fv-az1487-369
  * system: Linux
  * processor: x86_64
  * release: 6.5.0-1025-azure
"""

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace


def energy(t2=None, v=None, **kwargs):
    """
    Code generated by `albert` 0.0.0 on 2024-08-09T21:27:21.094943.

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

    e_mp = einsum(t2, (0, 1, 2, 3), v.ovov, (0, 2, 1, 3), ()) * 2
    e_mp += einsum(t2, (0, 1, 2, 3), v.ovov, (0, 3, 1, 2), ()) * -1

    return e_mp

