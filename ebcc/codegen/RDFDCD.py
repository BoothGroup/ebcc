# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, naux=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x1 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x1 += einsum(v.xov, (0, 1, 2), x0, (1, 3, 2, 4), (3, 4, 0))
    del x0
    e_cc = 0
    e_cc += einsum(v.xov, (0, 1, 2), x1, (1, 2, 0), ()) * 2.0
    del x1

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, naux=None, t2=None, **kwargs):
    # T amplitudes
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(v.xov, (0, 1, 2), v.xov, (0, 3, 4), (1, 3, 2, 4))
    x0 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x0 += einsum(v.xvv, (0, 1, 2), v.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new += einsum(t2, (0, 1, 2, 3), x0, (4, 2, 5, 3), (0, 1, 5, 4))
    del x0
    x1 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x1 += einsum(v.xov, (0, 1, 2), t2, (3, 1, 2, 4), (3, 4, 0))
    t2new += einsum(x1, (0, 1, 2), x1, (3, 4, 2), (0, 3, 1, 4))
    x2 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x2 += einsum(v.xov, (0, 1, 2), t2, (3, 1, 4, 2), (3, 4, 0))
    t2new += einsum(x2, (0, 1, 2), x2, (3, 4, 2), (0, 3, 1, 4)) * 4.0
    del x2
    x3 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x3 += einsum(v.xoo, (0, 1, 2), v.xoo, (0, 3, 4), (1, 3, 4, 2))
    t2new += einsum(t2, (0, 1, 2, 3), x3, (4, 0, 5, 1), (5, 4, 2, 3))
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(v.xov, (0, 1, 2), x1, (3, 4, 0), (3, 1, 4, 2))
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(v.xoo, (0, 1, 2), v.xvv, (0, 3, 4), (1, 2, 3, 4))
    x6 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x6 += einsum(v.xov, (0, 1, 2), (1, 2, 0))
    x6 += einsum(x1, (0, 1, 2), (0, 1, 2)) * -1.0
    del x1
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(v.xov, (0, 1, 2), x6, (3, 4, 0), (1, 3, 2, 4)) * 2.0
    del x6
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(x5, (0, 1, 2, 3), (1, 0, 3, 2))
    x8 += einsum(x7, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x7
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(t2, (0, 1, 2, 3), x8, (4, 1, 5, 3), (0, 4, 2, 5))
    del x8
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x10 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x11 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x11 += einsum(v.xov, (0, 1, 2), x10, (1, 3, 2, 4), (3, 4, 0))
    del x10
    x12 = np.zeros((nvir, nvir), dtype=types[float])
    x12 += einsum(v.xov, (0, 1, 2), x11, (1, 3, 0), (2, 3))
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(x12, (0, 1), t2, (2, 3, 0, 4), (2, 3, 4, 1))
    del x12
    x14 = np.zeros((nocc, nocc), dtype=types[float])
    x14 += einsum(v.xov, (0, 1, 2), x11, (3, 2, 0), (1, 3))
    del x11
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(x14, (0, 1), t2, (2, 0, 3, 4), (2, 1, 4, 3))
    del x14
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3))
    del x4
    x16 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    del x9
    x16 += einsum(x13, (0, 1, 2, 3), (1, 0, 2, 3))
    del x13
    x16 += einsum(x15, (0, 1, 2, 3), (0, 1, 3, 2))
    del x15
    t2new += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x16, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x16
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(t2, (0, 1, 2, 3), x5, (4, 1, 5, 2), (0, 4, 3, 5))
    del x5
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3))
    del x17
    x20 += einsum(x18, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x18
    x20 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    del x19
    t2new += einsum(x20, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x20

    return {"t2new": t2new}

