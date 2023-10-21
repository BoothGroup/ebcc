# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x1 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2))
    del x0
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x2 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x3 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(x1, (0, 1, 2, 3), (0, 1, 3, 2))
    x4 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x1
    x4 += einsum(t2, (0, 1, 2, 3), x2, (1, 4, 3, 5), (0, 4, 2, 5)) * -2.0
    del x2
    x4 += einsum(t2, (0, 1, 2, 3), x3, (1, 4, 2, 5), (0, 4, 3, 5)) * -2.0
    del x3
    e_mp = 0
    e_mp += einsum(t2, (0, 1, 2, 3), x4, (0, 1, 3, 2), ()) * 2.0
    del x4
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x5 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(t2, (0, 1, 2, 3), x5, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x5
    e_mp += einsum(t2, (0, 1, 2, 3), x6, (0, 1, 2, 3), ()) * 4.0
    del x6
    x7 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x7 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 2, 3), (0, 4, 5, 1))
    x8 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x8 += einsum(x7, (0, 1, 2, 3), (3, 1, 2, 0)) * -0.5
    x8 += einsum(x7, (0, 1, 2, 3), (3, 2, 1, 0))
    del x7
    e_mp += einsum(v.oooo, (0, 1, 2, 3), x8, (1, 0, 2, 3), ()) * 2.0
    del x8

    return e_mp

