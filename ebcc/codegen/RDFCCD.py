# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, naux=None, t2=None, **kwargs):
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

def update_amps(f=None, v=None, nocc=None, nvir=None, naux=None, t2=None, **kwargs):
    # T amplitudes
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.xov, (0, 1, 2), v.xov, (0, 3, 4), (1, 3, 2, 4))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2))
    x1 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x1 += einsum(v.xoo, (0, 1, 2), v.xoo, (0, 3, 4), (1, 3, 4, 2))
    t2new += einsum(t2, (0, 1, 2, 3), x1, (4, 1, 5, 0), (5, 4, 3, 2))
    del x1
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum(t2, (0, 1, 2, 3), x0, (4, 5, 3, 2), (0, 1, 5, 4))
    t2new += einsum(t2, (0, 1, 2, 3), x2, (4, 5, 1, 0), (5, 4, 2, 3))
    del x2
    x3 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x3 += einsum(v.xov, (0, 1, 2), t2, (3, 1, 2, 4), (3, 4, 0))
    t2new += einsum(x3, (0, 1, 2), x3, (3, 4, 2), (0, 3, 1, 4))
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(t2, (0, 1, 2, 3), x0, (4, 1, 2, 5), (0, 4, 3, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x4, (4, 1, 5, 2), (0, 4, 5, 3))
    x5 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x5 += einsum(v.xvv, (0, 1, 2), v.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new += einsum(t2, (0, 1, 2, 3), x5, (4, 2, 5, 3), (0, 1, 5, 4))
    del x5
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(v.xoo, (0, 1, 2), v.xvv, (0, 3, 4), (1, 2, 3, 4))
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(t2, (0, 1, 2, 3), x8, (4, 1, 5, 2), (0, 4, 3, 5))
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    del x6
    x10 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x7
    x10 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    del x9
    t2new += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x10, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x10
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(v.xov, (0, 1, 2), x3, (3, 4, 0), (3, 1, 4, 2))
    x12 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x12 += einsum(v.xov, (0, 1, 2), (1, 2, 0))
    x12 += einsum(x3, (0, 1, 2), (0, 1, 2)) * -1.0
    del x3
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(v.xov, (0, 1, 2), x12, (3, 4, 0), (1, 3, 2, 4)) * 2.0
    del x12
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(x8, (0, 1, 2, 3), (1, 0, 3, 2))
    del x8
    x14 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x4
    x14 += einsum(x13, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x13
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(t2, (0, 1, 2, 3), x14, (4, 1, 5, 3), (0, 4, 2, 5))
    del x14
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x16 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x17 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x17 += einsum(v.xov, (0, 1, 2), x16, (1, 3, 4, 2), (3, 4, 0))
    del x16
    x18 = np.zeros((nvir, nvir), dtype=types[float])
    x18 += einsum(v.xov, (0, 1, 2), x17, (1, 3, 0), (2, 3))
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(x18, (0, 1), t2, (2, 3, 0, 4), (2, 3, 4, 1)) * 2.0
    del x18
    x20 = np.zeros((nocc, nocc), dtype=types[float])
    x20 += einsum(v.xov, (0, 1, 2), x17, (3, 2, 0), (1, 3))
    del x17
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(x20, (0, 1), t2, (2, 0, 3, 4), (2, 1, 4, 3)) * 2.0
    del x20
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3))
    del x11
    x22 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    del x15
    x22 += einsum(x19, (0, 1, 2, 3), (1, 0, 2, 3))
    del x19
    x22 += einsum(x21, (0, 1, 2, 3), (0, 1, 3, 2))
    del x21
    t2new += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x22
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(x0, (0, 1, 2, 3), (1, 0, 2, 3))
    x23 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    del x0
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum(t2, (0, 1, 2, 3), x23, (1, 4, 5, 3), (0, 4, 2, 5)) * 2.0
    del x23
    t2new += einsum(t2, (0, 1, 2, 3), x24, (4, 1, 5, 3), (0, 4, 2, 5)) * 2.0
    del x24

    return {"t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, naux=None, t2=None, l2=None, **kwargs):
    # L amplitudes
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.xov, (0, 1, 2), v.xov, (0, 3, 4), (1, 3, 2, 4))
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum(x0, (0, 1, 2, 3), (3, 2, 1, 0))
    x1 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x1 += einsum(v.xoo, (0, 1, 2), v.xoo, (0, 3, 4), (1, 3, 4, 2))
    l2new += einsum(l2, (0, 1, 2, 3), x1, (4, 3, 5, 2), (0, 1, 4, 5))
    del x1
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    l2new += einsum(x0, (0, 1, 2, 3), x2, (4, 5, 1, 0), (2, 3, 5, 4))
    del x2
    x3 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x3 += einsum(t2, (0, 1, 2, 3), x0, (4, 5, 2, 3), (0, 1, 4, 5))
    l2new += einsum(l2, (0, 1, 2, 3), x3, (2, 3, 4, 5), (0, 1, 4, 5))
    del x3
    x4 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x4 += einsum(v.xvv, (0, 1, 2), v.xvv, (0, 3, 4), (1, 3, 4, 2))
    l2new += einsum(l2, (0, 1, 2, 3), x4, (4, 0, 5, 1), (4, 5, 3, 2))
    del x4
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(f.oo, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3))
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(f.vv, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(v.xoo, (0, 1, 2), v.xvv, (0, 3, 4), (1, 2, 3, 4))
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(t2, (0, 1, 2, 3), x0, (4, 1, 2, 5), (0, 4, 3, 5))
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(x7, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x9 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3))
    del x8
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(l2, (0, 1, 2, 3), x9, (2, 4, 1, 5), (3, 4, 0, 5))
    del x9
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x11 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum(l2, (0, 1, 2, 3), x11, (2, 4, 1, 0), (3, 4))
    x13 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x13 += einsum(x12, (0, 1), v.xov, (2, 1, 3), (0, 3, 2))
    del x12
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(v.xov, (0, 1, 2), x13, (3, 4, 0), (1, 3, 2, 4)) * 2.0
    del x13
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x15 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x16 = np.zeros((nvir, nvir), dtype=types[float])
    x16 += einsum(l2, (0, 1, 2, 3), x15, (2, 3, 4, 1), (0, 4))
    del x15
    x17 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x17 += einsum(x16, (0, 1), v.xov, (2, 3, 1), (3, 0, 2)) * 2.0
    del x16
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(v.xov, (0, 1, 2), x17, (3, 4, 0), (1, 3, 2, 4))
    del x17
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    del x5
    x19 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x6
    x19 += einsum(x10, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x10
    x19 += einsum(x14, (0, 1, 2, 3), (1, 0, 2, 3))
    del x14
    x19 += einsum(x18, (0, 1, 2, 3), (0, 1, 3, 2))
    del x18
    l2new += einsum(x19, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new += einsum(x19, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    del x19
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 5, 1), (3, 4, 0, 5))
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(x0, (0, 1, 2, 3), x20, (4, 1, 5, 2), (4, 0, 5, 3))
    del x20
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(x0, (0, 1, 2, 3), x11, (0, 4, 5, 3), (1, 4, 2, 5)) * 2.0
    del x0
    x23 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x23 += einsum(v.xov, (0, 1, 2), x11, (1, 3, 4, 2), (3, 4, 0)) * 2.0
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum(v.xov, (0, 1, 2), x23, (3, 4, 0), (1, 3, 2, 4)) * 2.0
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(x7, (0, 1, 2, 3), (1, 0, 3, 2))
    del x7
    x25 += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2))
    del x22
    x25 += einsum(x24, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x24
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(l2, (0, 1, 2, 3), x25, (3, 4, 1, 5), (2, 4, 0, 5))
    del x25
    x27 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x27 += einsum(v.xov, (0, 1, 2), l2, (3, 2, 4, 1), (4, 3, 0))
    x28 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x28 += einsum(v.xov, (0, 1, 2), (1, 2, 0))
    x28 += einsum(x23, (0, 1, 2), (0, 1, 2))
    del x23
    x29 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x29 += einsum(x28, (0, 1, 2), l2, (3, 1, 0, 4), (4, 3, 2))
    del x28
    x30 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x30 += einsum(x27, (0, 1, 2), (0, 1, 2)) * 2.0
    del x27
    x30 += einsum(x29, (0, 1, 2), (0, 1, 2)) * -1.0
    del x29
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum(v.xov, (0, 1, 2), x30, (3, 4, 0), (1, 3, 2, 4))
    del x30
    x32 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x32 += einsum(v.xov, (0, 1, 2), x11, (1, 3, 4, 2), (3, 4, 0))
    del x11
    x33 = np.zeros((nvir, nvir), dtype=types[float])
    x33 += einsum(v.xov, (0, 1, 2), x32, (1, 3, 0), (2, 3))
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x34 += einsum(x33, (0, 1), l2, (2, 1, 3, 4), (4, 3, 2, 0)) * 2.0
    del x33
    x35 = np.zeros((nocc, nocc), dtype=types[float])
    x35 += einsum(v.xov, (0, 1, 2), x32, (3, 2, 0), (1, 3))
    del x32
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 += einsum(x35, (0, 1), l2, (2, 3, 1, 4), (4, 0, 2, 3)) * 2.0
    del x35
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x21
    x37 += einsum(x26, (0, 1, 2, 3), (0, 1, 2, 3))
    del x26
    x37 += einsum(x31, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x31
    x37 += einsum(x34, (0, 1, 2, 3), (1, 0, 2, 3))
    del x34
    x37 += einsum(x36, (0, 1, 2, 3), (0, 1, 3, 2))
    del x36
    l2new += einsum(x37, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    l2new += einsum(x37, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x37

    return {"l2new": l2new}

