# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 2, 3), ()) * 0.25

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(t1, (0, 1), v.ovov, (2, 1, 0, 3), (2, 3)) * -1.0
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t1new += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 2, 3), (0, 4)) * -0.5
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 5, 2, 3), (0, 1, 4, 5)) * 0.5
    t2new += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x0 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x0 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    t1new += einsum(t2, (0, 1, 2, 3), x0, (4, 0, 1, 3), (4, 2)) * 0.5
    del x0
    x1 = np.zeros((nocc, nvir), dtype=types[float])
    x1 += einsum(f.ov, (0, 1), (0, 1))
    x1 += einsum(t1, (0, 1), v.oovv, (2, 0, 3, 1), (2, 3))
    t1new += einsum(x1, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    del x1
    x2 = np.zeros((nocc, nocc), dtype=types[float])
    x2 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    x3 = np.zeros((nocc, nocc), dtype=types[float])
    x3 += einsum(f.oo, (0, 1), (0, 1))
    x3 += einsum(x2, (0, 1), (1, 0)) * 0.5
    t1new += einsum(t1, (0, 1), x3, (0, 2), (2, 1)) * -1.0
    del x3
    x4 = np.zeros((nvir, nvir), dtype=types[float])
    x4 += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(x4, (0, 1), t2, (2, 3, 4, 1), (2, 3, 4, 0))
    del x4
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(t2, (0, 1, 2, 3), x6, (4, 1, 5, 3), (4, 0, 5, 2))
    del x6
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x5
    x8 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    del x7
    t2new += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x8, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(x2, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4))
    del x2
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(x9, (0, 1, 2, 3), (0, 1, 3, 2))
    del x9
    x12 += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x10
    x12 += einsum(x11, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    del x11
    t2new += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x12, (0, 1, 2, 3), (1, 0, 2, 3))
    del x12
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    t2new += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x13, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new += einsum(x13, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new += einsum(x13, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x13
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(t1, (0, 1), v.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(x14, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x14
    x16 += einsum(x15, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x15
    t2new += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x16, (0, 1, 2, 3), (0, 1, 3, 2))
    del x16
    x17 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x17 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x17 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (4, 5, 0, 1))
    t2new += einsum(t2, (0, 1, 2, 3), x17, (0, 1, 4, 5), (4, 5, 2, 3)) * 0.25
    del x17

    return {"t1new": t1new, "t2new": t2new}

