# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x1 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x1 += einsum(v.xov, (0, 1, 2), x0, (1, 3, 4, 2), (3, 4, 0))
    del x0
    e_cc = 0
    e_cc += einsum(v.xov, (0, 1, 2), x1, (1, 2, 0), ()) * 2.0
    del x1

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    x0 = np.zeros((naux,), dtype=types[float])
    x0 += einsum(t1, (0, 1), v.xov, (2, 0, 1), (2,))
    x1 = np.zeros((nocc, nvir), dtype=types[float])
    x1 += einsum(x0, (0,), v.xov, (0, 1, 2), (1, 2))
    del x0
    t1new += einsum(x1, (0, 1), (0, 1)) * 2.0
    x2 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x2 += einsum(t1, (0, 1), v.xoo, (2, 3, 0), (3, 1, 2))
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x3 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x4 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x4 += einsum(x2, (0, 1, 2), (0, 1, 2))
    x4 += einsum(v.xov, (0, 1, 2), x3, (1, 3, 4, 2), (3, 4, 0)) * -1.0
    t1new += einsum(v.xvv, (0, 1, 2), x4, (3, 2, 0), (3, 1)) * -1.0
    del x4
    x5 = np.zeros((nocc, nocc, naux), dtype=types[float])
    x5 += einsum(t1, (0, 1), v.xov, (2, 3, 1), (0, 3, 2))
    x6 = np.zeros((nocc, nocc, naux), dtype=types[float])
    x6 += einsum(v.xoo, (0, 1, 2), (1, 2, 0))
    x6 += einsum(x5, (0, 1, 2), (1, 0, 2))
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x7 += einsum(v.xov, (0, 1, 2), x6, (3, 4, 0), (4, 1, 3, 2))
    del x6
    t1new += einsum(x3, (0, 1, 2, 3), x7, (4, 0, 1, 3), (4, 2)) * -1.0
    del x7
    x8 = np.zeros((nocc, nvir), dtype=types[float])
    x8 += einsum(f.ov, (0, 1), (0, 1))
    x8 += einsum(x1, (0, 1), (0, 1)) * 2.0
    del x1
    x8 += einsum(v.xov, (0, 1, 2), x5, (1, 3, 0), (3, 2)) * -1.0
    del x5
    t1new += einsum(x8, (0, 1), x3, (0, 2, 3, 1), (2, 3))
    del x3, x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x9 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x10 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x10 += einsum(v.xov, (0, 1, 2), x9, (1, 3, 4, 2), (3, 4, 0))
    del x9
    x11 = np.zeros((nocc, nocc), dtype=types[float])
    x11 += einsum(v.xov, (0, 1, 2), x10, (3, 2, 0), (1, 3))
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum(f.oo, (0, 1), (0, 1)) * 0.5
    x12 += einsum(x11, (0, 1), (0, 1))
    t1new += einsum(t1, (0, 1), x12, (0, 2), (2, 1)) * -2.0
    del x12
    x13 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x13 += einsum(v.xov, (0, 1, 2), t2, (3, 1, 2, 4), (3, 4, 0))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(x13, (0, 1, 2), x13, (3, 4, 2), (0, 3, 1, 4))
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(v.xov, (0, 1, 2), v.xov, (0, 3, 4), (1, 3, 2, 4))
    t2new += einsum(x14, (0, 1, 2, 3), (1, 0, 3, 2))
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(t2, (0, 1, 2, 3), x14, (1, 4, 5, 2), (0, 4, 3, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x15, (4, 1, 5, 2), (0, 4, 5, 3))
    x16 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x16 += einsum(v.xvv, (0, 1, 2), v.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new += einsum(t2, (0, 1, 2, 3), x16, (4, 3, 5, 2), (0, 1, 4, 5))
    del x16
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x19 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x19 += einsum(t1, (0, 1), v.xvv, (2, 3, 1), (0, 3, 2))
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(v.xov, (0, 1, 2), x19, (3, 4, 0), (3, 1, 2, 4))
    del x19
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(v.xoo, (0, 1, 2), v.xvv, (0, 3, 4), (1, 2, 3, 4))
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(t2, (0, 1, 2, 3), x21, (4, 1, 5, 2), (0, 4, 3, 5))
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3))
    del x17
    x23 += einsum(x18, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x18
    x23 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x20
    x23 += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3))
    del x22
    t2new += einsum(x23, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x23
    x24 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x24 += einsum(v.xov, (0, 1, 2), (1, 2, 0))
    x24 += einsum(x13, (0, 1, 2), (0, 1, 2)) * -1.0
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(v.xov, (0, 1, 2), x24, (3, 4, 0), (3, 1, 4, 2)) * 2.0
    del x24
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(x21, (0, 1, 2, 3), (1, 0, 3, 2))
    del x21
    x26 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x15
    x26 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x25
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(t2, (0, 1, 2, 3), x26, (4, 1, 5, 3), (4, 0, 5, 2))
    del x26
    x28 = np.zeros((nvir, nvir), dtype=types[float])
    x28 += einsum(v.xov, (0, 1, 2), x10, (1, 3, 0), (2, 3))
    del x10
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum(x28, (0, 1), t2, (2, 3, 0, 4), (2, 3, 1, 4)) * 2.0
    del x28
    x30 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x30 += einsum(x2, (0, 1, 2), (0, 1, 2))
    del x2
    x30 += einsum(x13, (0, 1, 2), (0, 1, 2))
    del x13
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum(v.xov, (0, 1, 2), x30, (3, 4, 0), (3, 1, 4, 2))
    del x30
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum(x11, (0, 1), t2, (2, 0, 3, 4), (1, 2, 4, 3)) * 2.0
    del x11
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum(x27, (0, 1, 2, 3), (1, 0, 3, 2))
    del x27
    x33 += einsum(x29, (0, 1, 2, 3), (1, 0, 3, 2))
    del x29
    x33 += einsum(x31, (0, 1, 2, 3), (0, 1, 2, 3))
    del x31
    x33 += einsum(x32, (0, 1, 2, 3), (1, 0, 3, 2))
    del x32
    t2new += einsum(x33, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x33, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x33
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x34 += einsum(x14, (0, 1, 2, 3), (1, 0, 2, 3))
    x34 += einsum(x14, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum(t2, (0, 1, 2, 3), x34, (1, 4, 5, 3), (0, 4, 2, 5))
    del x34
    t2new += einsum(t2, (0, 1, 2, 3), x35, (4, 1, 5, 3), (4, 0, 5, 2)) * 4.0
    del x35
    x36 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x36 += einsum(v.xoo, (0, 1, 2), v.xoo, (0, 3, 4), (1, 3, 4, 2))
    x36 += einsum(t2, (0, 1, 2, 3), x14, (4, 5, 3, 2), (4, 0, 5, 1))
    del x14
    t2new += einsum(t2, (0, 1, 2, 3), x36, (0, 4, 1, 5), (4, 5, 3, 2))
    del x36

    return {"t1new": t1new, "t2new": t2new}

