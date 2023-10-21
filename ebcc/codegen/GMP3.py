# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 2, 3), (0, 4, 5, 1))
    e_mp = 0
    e_mp += einsum(v.oooo, (0, 1, 2, 3), x0, (0, 2, 3, 1), ()) * 0.125
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x1 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (4, 0, 5, 2)) * -8.0
    x1 += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 5, 2, 3), (0, 1, 4, 5))
    e_mp += einsum(t2, (0, 1, 2, 3), x1, (0, 1, 2, 3), ()) * 0.125
    del x1

    return e_mp

