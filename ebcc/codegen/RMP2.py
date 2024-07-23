"""
Code generated by `albert`:
https://github.com/obackhouse/albert

  * date: 2024-07-18T09:27:14.960165
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
    Code generated by `albert` 0.0.0 on 2024-07-18T09:27:15.109289.

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

    e_mp = einsum(t2, (0, 1, 2, 3), v.ovov, (0, 2, 1, 3), (), optimize=True) * 2
    e_mp += einsum(t2, (0, 1, 2, 3), v.ovov, (0, 3, 1, 2), (), optimize=True) * -1

    return e_mp
