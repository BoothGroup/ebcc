# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x1 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x1 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x2 += einsum(x0, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x2 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2))
    x2 += einsum(t2, (0, 1, 2, 3), x1, (1, 4, 3, 5), (0, 4, 2, 5)) * 4.0
    x2 += einsum(t2, (0, 1, 2, 3), x1, (1, 4, 2, 5), (4, 0, 5, 3)) * -4.0
    e_mp = 0
    e_mp += einsum(t2, (0, 1, 2, 3), x2, (1, 0, 3, 2), ()) * 2.0
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x3 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x3 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x4 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x4 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x4 += einsum(t2, (0, 1, 2, 3), x3, (1, 4, 2, 5), (0, 4, 3, 5)) * -1.0
    e_mp += einsum(t2, (0, 1, 2, 3), x4, (1, 0, 2, 3), ()) * 2.0
    x5 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x5 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 2, 3), (0, 4, 5, 1))
    x6 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x6 += einsum(x5, (0, 1, 2, 3), (3, 1, 2, 0)) * -0.5
    x6 += einsum(x5, (0, 1, 2, 3), (3, 2, 1, 0))
    e_mp += einsum(v.oooo, (0, 1, 2, 3), x6, (0, 1, 2, 3), ()) * 2.0

    return e_mp

