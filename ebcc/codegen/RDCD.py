# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), x0, (0, 1, 3, 2), ()) * 2.0
    del x0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # T amplitudes
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new += einsum(t2, (0, 1, 2, 3), v.oooo, (4, 1, 5, 0), (4, 5, 3, 2))
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (4, 0, 5, 2)) * -1.0
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (0, 4, 3, 5))
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3))
    del x2
    x5 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x3
    x5 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3))
    del x4
    t2new += einsum(x5, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x5, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x5
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(t2, (0, 1, 2, 3), x1, (4, 1, 5, 3), (0, 4, 2, 5))
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x7 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x8 = np.zeros((nvir, nvir), dtype=types[float])
    x8 += einsum(t2, (0, 1, 2, 3), x7, (0, 1, 4, 2), (3, 4))
    del x7
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(x8, (0, 1), t2, (2, 3, 1, 4), (2, 3, 4, 0))
    del x8
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x10 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x11 = np.zeros((nocc, nocc), dtype=types[float])
    x11 += einsum(t2, (0, 1, 2, 3), x10, (1, 4, 3, 2), (0, 4))
    del x10
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(x11, (0, 1), t2, (2, 1, 3, 4), (2, 0, 4, 3))
    del x11
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x6
    x13 += einsum(x9, (0, 1, 2, 3), (1, 0, 2, 3))
    del x9
    x13 += einsum(x12, (0, 1, 2, 3), (0, 1, 3, 2))
    del x12
    t2new += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x13, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x13
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x14 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    del x1
    t2new += einsum(t2, (0, 1, 2, 3), x14, (4, 1, 5, 2), (4, 0, 5, 3))
    del x14
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x15 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.25
    x15 += einsum(x0, (0, 1, 2, 3), (0, 1, 2, 3))
    del x0
    t2new += einsum(t2, (0, 1, 2, 3), x15, (4, 1, 5, 3), (0, 4, 2, 5)) * 4.0
    del x15

    return {"t2new": t2new}

