# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((naux,), dtype=types[float])
    x0 += einsum(t1, (0, 1), v.xov, (2, 0, 1), (2,))
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x2 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x2 += einsum(x0, (0,), t1, (1, 2), (1, 2, 0))
    del x0
    x2 += einsum(v.xov, (0, 1, 2), x1, (1, 3, 4, 2), (3, 4, 0))
    del x1
    e_cc = 0
    e_cc += einsum(v.xov, (0, 1, 2), x2, (1, 2, 0), ()) * 2.0
    del x2
    x3 = np.zeros((nocc, nocc, naux), dtype=types[float])
    x3 += einsum(t1, (0, 1), v.xov, (2, 3, 1), (0, 3, 2))
    x4 = np.zeros((nocc, nvir), dtype=types[float])
    x4 += einsum(f.ov, (0, 1), (0, 1)) * 2.0
    x4 += einsum(v.xov, (0, 1, 2), x3, (1, 3, 0), (3, 2)) * -1.0
    del x3
    e_cc += einsum(t1, (0, 1), x4, (0, 1), ())
    del x4

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(v.xov, (0, 1, 2), v.xov, (0, 3, 4), (1, 3, 2, 4))
    x0 = np.zeros((naux,), dtype=types[float])
    x0 += einsum(t1, (0, 1), v.xov, (2, 0, 1), (2,))
    x1 = np.zeros((nocc, nvir), dtype=types[float])
    x1 += einsum(x0, (0,), v.xov, (0, 1, 2), (1, 2))
    t1new += einsum(x1, (0, 1), (0, 1)) * 2.0
    x2 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x2 += einsum(t1, (0, 1), v.xoo, (2, 3, 0), (3, 1, 2))
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    x3 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    x3 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x4 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x4 += einsum(x2, (0, 1, 2), (0, 1, 2)) * 0.5
    x4 += einsum(x0, (0,), t1, (1, 2), (1, 2, 0)) * -1.0
    x4 += einsum(v.xov, (0, 1, 2), x3, (1, 3, 4, 2), (3, 4, 0)) * 0.5
    del x3
    t1new += einsum(v.xvv, (0, 1, 2), x4, (3, 2, 0), (3, 1)) * -2.0
    del x4
    x5 = np.zeros((nocc, nocc, naux), dtype=types[float])
    x5 += einsum(t1, (0, 1), v.xov, (2, 3, 1), (0, 3, 2))
    x6 = np.zeros((nocc, nocc, naux), dtype=types[float])
    x6 += einsum(v.xoo, (0, 1, 2), (1, 2, 0))
    x6 += einsum(x5, (0, 1, 2), (1, 0, 2))
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x7 += einsum(v.xov, (0, 1, 2), x6, (3, 4, 0), (3, 4, 1, 2))
    del x6
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x8 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t1new += einsum(x7, (0, 1, 2, 3), x8, (2, 0, 4, 3), (1, 4)) * -1.0
    x9 = np.zeros((nocc, nvir), dtype=types[float])
    x9 += einsum(v.xov, (0, 1, 2), x5, (1, 3, 0), (3, 2))
    x10 = np.zeros((nocc, nvir), dtype=types[float])
    x10 += einsum(f.ov, (0, 1), (0, 1))
    x10 += einsum(x1, (0, 1), (0, 1)) * 2.0
    del x1
    x10 += einsum(x9, (0, 1), (0, 1)) * -1.0
    t1new += einsum(x10, (0, 1), x8, (0, 2, 3, 1), (2, 3))
    del x8, x10
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x11 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x12 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x12 += einsum(x2, (0, 1, 2), (0, 1, 2)) * -1.0
    x12 += einsum(x0, (0,), t1, (1, 2), (1, 2, 0)) * 2.0
    x12 += einsum(v.xov, (0, 1, 2), x11, (1, 3, 4, 2), (3, 4, 0)) * 2.0
    del x11
    x13 = np.zeros((nocc, nvir), dtype=types[float])
    x13 += einsum(f.ov, (0, 1), (0, 1))
    x13 += einsum(x9, (0, 1), (0, 1)) * -1.0
    del x9
    x14 = np.zeros((nocc, nocc), dtype=types[float])
    x14 += einsum(f.oo, (0, 1), (0, 1))
    x14 += einsum(x0, (0,), v.xoo, (0, 1, 2), (1, 2)) * 2.0
    del x0
    x14 += einsum(v.xov, (0, 1, 2), x12, (3, 2, 0), (1, 3))
    del x12
    x14 += einsum(t1, (0, 1), x13, (2, 1), (2, 0))
    del x13
    t1new += einsum(t1, (0, 1), x14, (0, 2), (2, 1)) * -1.0
    del x14
    x15 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x15 += einsum(t1, (0, 1), v.xvv, (2, 3, 1), (0, 3, 2))
    t2new += einsum(x15, (0, 1, 2), x15, (3, 4, 2), (0, 3, 1, 4))
    x16 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x16 += einsum(v.xoo, (0, 1, 2), x5, (3, 4, 0), (3, 1, 2, 4))
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum(t1, (0, 1), x16, (2, 3, 4, 0), (2, 3, 4, 1))
    del x16
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(t1, (0, 1), x17, (2, 0, 3, 4), (2, 3, 1, 4))
    del x17
    x19 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x19 += einsum(v.xov, (0, 1, 2), (1, 2, 0)) * -1.0
    x19 += einsum(x2, (0, 1, 2), (0, 1, 2))
    del x2
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(x15, (0, 1, 2), x19, (3, 4, 2), (3, 0, 4, 1))
    del x19
    x21 = np.zeros((nvir, nvir), dtype=types[float])
    x21 += einsum(f.ov, (0, 1), t1, (0, 2), (1, 2))
    x22 = np.zeros((nvir, nvir), dtype=types[float])
    x22 += einsum(f.vv, (0, 1), (0, 1))
    x22 += einsum(x21, (0, 1), (0, 1)) * -1.0
    del x21
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(x22, (0, 1), t2, (2, 3, 0, 4), (2, 3, 1, 4))
    del x22
    x24 = np.zeros((nocc, nocc), dtype=types[float])
    x24 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x25 = np.zeros((nocc, nocc), dtype=types[float])
    x25 += einsum(f.oo, (0, 1), (0, 1))
    x25 += einsum(x24, (0, 1), (0, 1))
    del x24
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(x25, (0, 1), t2, (2, 0, 3, 4), (1, 2, 4, 3))
    del x25
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(x18, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x18
    x27 += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3))
    del x20
    x27 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x23
    x27 += einsum(x26, (0, 1, 2, 3), (0, 1, 3, 2))
    del x26
    t2new += einsum(x27, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x27, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x27
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum(x15, (0, 1, 2), x5, (3, 4, 2), (3, 0, 4, 1))
    del x15
    x29 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x29 += einsum(x28, (0, 1, 2, 3), (0, 1, 2, 3))
    del x28
    x29 += einsum(x7, (0, 1, 2, 3), (1, 2, 0, 3))
    del x7
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum(t1, (0, 1), x29, (2, 3, 0, 4), (2, 3, 4, 1))
    del x29
    t2new += einsum(x30, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x30, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x30
    x31 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x31 += einsum(v.xoo, (0, 1, 2), v.xoo, (0, 3, 4), (1, 3, 4, 2))
    x31 += einsum(x5, (0, 1, 2), x5, (3, 4, 2), (4, 0, 1, 3))
    del x5
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x32 += einsum(t1, (0, 1), x31, (0, 2, 3, 4), (2, 4, 3, 1))
    del x31
    t2new += einsum(t1, (0, 1), x32, (2, 3, 0, 4), (2, 3, 1, 4))
    del x32

    return {"t1new": t1new, "t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    # L amplitudes
    l1new = np.zeros((nvir, nocc), dtype=types[float])
    l1new += einsum(f.ov, (0, 1), (1, 0))
    x0 = np.zeros((naux,), dtype=types[float])
    x0 += einsum(t1, (0, 1), v.xov, (2, 0, 1), (2,))
    x1 = np.zeros((nocc, nvir), dtype=types[float])
    x1 += einsum(x0, (0,), v.xov, (0, 1, 2), (1, 2))
    x2 = np.zeros((nocc, nocc), dtype=types[float])
    x2 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    l1new += einsum(x1, (0, 1), x2, (2, 0), (1, 2)) * -2.0
    x3 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x3 += einsum(t1, (0, 1), v.xvv, (2, 3, 1), (0, 3, 2))
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x4 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x5 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x5 += einsum(v.xov, (0, 1, 2), (1, 2, 0))
    x5 += einsum(x3, (0, 1, 2), (0, 1, 2))
    x5 += einsum(v.xov, (0, 1, 2), x4, (1, 3, 4, 2), (3, 4, 0)) * 2.0
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * 2.0
    x6 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x7 = np.zeros((nocc, nocc, naux), dtype=types[float])
    x7 += einsum(t1, (0, 1), v.xov, (2, 3, 1), (0, 3, 2))
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x8 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x9 += einsum(x8, (0, 1, 2, 3), (1, 0, 2, 3))
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x10 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x11 = np.zeros((nvir, nvir), dtype=types[float])
    x11 += einsum(l2, (0, 1, 2, 3), x10, (2, 3, 4, 1), (0, 4))
    x12 = np.zeros((nocc, nocc, naux), dtype=types[float])
    x12 += einsum(v.xoo, (0, 1, 2), (1, 2, 0))
    x12 += einsum(x7, (0, 1, 2), (0, 1, 2))
    x13 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x13 += einsum(x5, (0, 1, 2), x6, (0, 3, 4, 1), (3, 4, 2)) * -0.5
    del x5, x6
    x13 += einsum(x7, (0, 1, 2), x9, (0, 3, 1, 4), (3, 4, 2))
    x13 += einsum(v.xoo, (0, 1, 2), x9, (1, 3, 2, 4), (3, 4, 0))
    del x9
    x13 += einsum(x11, (0, 1), v.xov, (2, 3, 1), (3, 0, 2))
    del x11
    x13 += einsum(l1, (0, 1), x12, (1, 2, 3), (2, 0, 3)) * 0.5
    del x12
    l1new += einsum(v.xvv, (0, 1, 2), x13, (3, 2, 0), (1, 3)) * -2.0
    del x13
    x14 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x14 += einsum(v.xoo, (0, 1, 2), x3, (3, 4, 0), (3, 1, 2, 4))
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    x16 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x16 += einsum(v.xov, (0, 1, 2), v.xvv, (0, 3, 4), (1, 2, 3, 4))
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum(t2, (0, 1, 2, 3), x16, (4, 2, 3, 5), (0, 1, 4, 5))
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(v.xov, (0, 1, 2), v.xov, (0, 3, 4), (1, 3, 2, 4))
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum(x18, (0, 1, 2, 3), (3, 2, 1, 0))
    x19 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x19 += einsum(t2, (0, 1, 2, 3), x18, (4, 5, 2, 3), (0, 1, 4, 5))
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x20 += einsum(t1, (0, 1), x19, (2, 3, 4, 0), (2, 3, 4, 1))
    del x19
    x21 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x21 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x17
    x21 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x20
    x22 = np.zeros((nocc, nvir), dtype=types[float])
    x22 += einsum(v.xov, (0, 1, 2), x7, (1, 3, 0), (3, 2))
    x23 = np.zeros((nocc, nvir), dtype=types[float])
    x23 += einsum(x1, (0, 1), (0, 1))
    x23 += einsum(x22, (0, 1), (0, 1)) * -0.5
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x24 += einsum(x23, (0, 1), t2, (2, 3, 1, 4), (2, 3, 0, 4)) * 2.0
    x25 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x25 += einsum(v.xoo, (0, 1, 2), v.xov, (0, 3, 4), (1, 2, 3, 4))
    x26 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x26 += einsum(v.xov, (0, 1, 2), x7, (3, 4, 0), (3, 1, 4, 2))
    x27 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x27 += einsum(x25, (0, 1, 2, 3), (1, 0, 2, 3))
    x27 += einsum(x25, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.5
    x27 += einsum(x26, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x27 += einsum(x26, (0, 1, 2, 3), (0, 2, 1, 3))
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum(x25, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x28 += einsum(x25, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    x28 += einsum(x26, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x28 += einsum(x26, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x29 = np.zeros((nocc, nocc, naux), dtype=types[float])
    x29 += einsum(v.xoo, (0, 1, 2), (1, 2, 0))
    x29 += einsum(x7, (0, 1, 2), (1, 0, 2))
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x30 += einsum(v.xov, (0, 1, 2), x29, (3, 4, 0), (1, 3, 4, 2)) * 0.5
    x31 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x31 += einsum(v.xoo, (0, 1, 2), v.xoo, (0, 3, 4), (1, 3, 4, 2))
    x32 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x32 += einsum(v.xoo, (0, 1, 2), x7, (3, 4, 0), (3, 1, 2, 4))
    x33 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x33 += einsum(x31, (0, 1, 2, 3), (3, 2, 1, 0))
    x33 += einsum(x32, (0, 1, 2, 3), (1, 0, 3, 2))
    x33 += einsum(x32, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum(x14, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x34 += einsum(x15, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x34 += einsum(x15, (0, 1, 2, 3), (2, 0, 1, 3)) * 0.5
    del x15
    x34 += einsum(x21, (0, 1, 2, 3), (0, 2, 1, 3))
    x34 += einsum(x21, (0, 1, 2, 3), (1, 2, 0, 3)) * -2.0
    del x21
    x34 += einsum(x24, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    x34 += einsum(x24, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    del x24
    x34 += einsum(t2, (0, 1, 2, 3), x27, (4, 5, 1, 3), (4, 5, 0, 2))
    del x27
    x34 += einsum(t2, (0, 1, 2, 3), x28, (4, 5, 1, 2), (4, 5, 0, 3)) * 0.5
    x34 += einsum(x30, (0, 1, 2, 3), (2, 1, 0, 3))
    x34 += einsum(t1, (0, 1), x33, (0, 2, 3, 4), (2, 3, 4, 1)) * -0.5
    del x33
    l1new += einsum(l2, (0, 1, 2, 3), x34, (3, 4, 2, 1), (0, 4)) * 2.0
    del x34
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x35 += einsum(x3, (0, 1, 2), x7, (3, 4, 2), (3, 0, 4, 1))
    x36 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x36 += einsum(x7, (0, 1, 2), x7, (3, 4, 2), (0, 3, 1, 4))
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x37 += einsum(t1, (0, 1), x36, (2, 3, 4, 0), (2, 3, 4, 1))
    x38 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x38 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x35
    x38 += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x37
    x39 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x39 += einsum(x31, (0, 1, 2, 3), (3, 2, 1, 0))
    x39 += einsum(x32, (0, 1, 2, 3), (1, 0, 3, 2))
    x39 += einsum(x32, (0, 1, 2, 3), (3, 0, 2, 1)) * -0.5
    x40 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x40 += einsum(x14, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.25
    del x14
    x40 += einsum(x38, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x40 += einsum(x38, (0, 1, 2, 3), (1, 2, 0, 3)) * 0.5
    del x38
    x40 += einsum(x10, (0, 1, 2, 3), x28, (4, 0, 5, 2), (4, 5, 1, 3)) * -0.5
    del x28
    x40 += einsum(x30, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x30
    x40 += einsum(t1, (0, 1), x39, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x39
    l1new += einsum(l2, (0, 1, 2, 3), x40, (2, 4, 3, 1), (0, 4)) * 4.0
    del x40
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x41 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum(l2, (0, 1, 2, 3), x4, (3, 4, 5, 1), (2, 4, 0, 5))
    x42 += einsum(l2, (0, 1, 2, 3), x41, (2, 4, 5, 1), (3, 4, 0, 5)) * 0.5
    x43 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x43 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), (1, 2, 3, 4))
    x44 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x44 += einsum(t2, (0, 1, 2, 3), x8, (4, 1, 5, 2), (4, 5, 0, 3))
    x45 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x45 += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x43
    x45 += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x44
    x46 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x46 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    x47 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x47 += einsum(t1, (0, 1), x46, (2, 0, 3, 4), (2, 3, 4, 1))
    x48 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x48 += einsum(t2, (0, 1, 2, 3), x8, (4, 1, 5, 3), (4, 5, 0, 2))
    x49 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x49 += einsum(x41, (0, 1, 2, 3), x8, (0, 4, 5, 2), (4, 5, 1, 3))
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x50 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3))
    del x47
    x50 += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3))
    del x48
    x50 += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x49
    x51 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x51 += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x51 += einsum(x45, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    del x45
    x51 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3))
    x51 += einsum(x8, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    x51 += einsum(x50, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x51 += einsum(x50, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    del x50
    x52 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x52 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x52 += einsum(x8, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x53 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x53 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x54 = np.zeros((nocc, nocc), dtype=types[float])
    x54 += einsum(l2, (0, 1, 2, 3), x10, (2, 4, 0, 1), (3, 4)) * 2.0
    x55 = np.zeros((nocc, nocc), dtype=types[float])
    x55 += einsum(x2, (0, 1), (0, 1))
    x55 += einsum(x54, (0, 1), (0, 1))
    x56 = np.zeros((nocc, nvir), dtype=types[float])
    x56 += einsum(t1, (0, 1), (0, 1)) * -1.0
    x56 += einsum(x10, (0, 1, 2, 3), x8, (1, 0, 4, 3), (4, 2)) * 2.0
    x56 += einsum(l1, (0, 1), x53, (1, 2, 3, 0), (2, 3)) * -1.0
    x56 += einsum(t1, (0, 1), x55, (0, 2), (2, 1))
    x57 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x57 += einsum(t1, (0, 1), x8, (2, 3, 4, 1), (3, 2, 4, 0))
    l2new += einsum(x18, (0, 1, 2, 3), x57, (4, 5, 0, 1), (3, 2, 5, 4))
    del x18
    x58 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x58 += einsum(x46, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x58 += einsum(x46, (0, 1, 2, 3), (1, 0, 3, 2))
    del x46
    x58 += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3))
    x58 += einsum(x57, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x59 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x59 += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x59 += einsum(x57, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x57
    x60 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x60 += einsum(l1, (0, 1), v.xvv, (2, 3, 0), (1, 3, 2))
    x61 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x61 += einsum(x60, (0, 1, 2), (0, 1, 2)) * -1.0
    x61 += einsum(x55, (0, 1), v.xov, (2, 1, 3), (0, 3, 2))
    x62 = np.zeros((nocc, nocc), dtype=types[float])
    x62 += einsum(l2, (0, 1, 2, 3), x10, (2, 4, 0, 1), (3, 4))
    x63 = np.zeros((nocc, nocc), dtype=types[float])
    x63 += einsum(x2, (0, 1), (0, 1)) * 0.5
    x63 += einsum(x62, (0, 1), (0, 1))
    del x62
    l1new += einsum(f.ov, (0, 1), x63, (2, 0), (1, 2)) * -2.0
    x64 = np.zeros((nocc, nocc, naux), dtype=types[float])
    x64 += einsum(v.xvv, (0, 1, 2), x42, (3, 4, 1, 2), (4, 3, 0)) * -1.0
    del x42
    x64 += einsum(v.xov, (0, 1, 2), x51, (3, 1, 4, 2), (4, 3, 0)) * 0.5
    del x51
    x64 += einsum(x3, (0, 1, 2), x52, (3, 0, 4, 1), (4, 3, 2)) * -0.5
    x64 += einsum(x56, (0, 1), v.xov, (2, 3, 1), (0, 3, 2)) * 0.5
    del x56
    x64 += einsum(v.xoo, (0, 1, 2), x58, (1, 3, 2, 4), (4, 3, 0))
    del x58
    x64 += einsum(x7, (0, 1, 2), x59, (0, 3, 4, 1), (4, 3, 2)) * 0.5
    del x59
    x64 += einsum(t1, (0, 1), x61, (2, 1, 3), (0, 2, 3)) * 0.5
    del x61
    x64 += einsum(x63, (0, 1), v.xoo, (2, 3, 0), (1, 3, 2))
    x64 += einsum(x0, (0,), x54, (1, 2), (2, 1, 0)) * -1.0
    del x54
    l1new += einsum(v.xov, (0, 1, 2), x64, (1, 3, 0), (2, 3)) * 2.0
    del x64
    x65 = np.zeros((nvir, nvir), dtype=types[float])
    x65 += einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    x65 += einsum(l2, (0, 1, 2, 3), x10, (2, 3, 4, 1), (4, 0)) * 2.0
    del x10
    x66 = np.zeros((nocc, nvir), dtype=types[float])
    x66 += einsum(l1, (0, 1), (1, 0))
    x66 += einsum(t1, (0, 1), (0, 1))
    x66 += einsum(x41, (0, 1, 2, 3), x8, (1, 0, 4, 3), (4, 2)) * -1.0
    del x41
    x66 += einsum(l1, (0, 1), x4, (1, 2, 3, 0), (2, 3)) * 2.0
    del x4
    x66 += einsum(t1, (0, 1), x63, (0, 2), (2, 1)) * -2.0
    del x63
    x67 = np.zeros((naux,), dtype=types[float])
    x67 += einsum(x65, (0, 1), v.xvv, (2, 1, 0), (2,))
    del x65
    x67 += einsum(x66, (0, 1), v.xov, (2, 0, 1), (2,))
    del x66
    x67 += einsum(x55, (0, 1), v.xoo, (2, 1, 0), (2,)) * -1.0
    del x55
    l1new += einsum(x67, (0,), v.xov, (0, 1, 2), (2, 1)) * 2.0
    del x67
    x68 = np.zeros((nocc, nocc), dtype=types[float])
    x68 += einsum(x0, (0,), v.xoo, (0, 1, 2), (1, 2))
    x69 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x69 += einsum(t1, (0, 1), v.xoo, (2, 3, 0), (3, 1, 2))
    x69 += einsum(v.xov, (0, 1, 2), x53, (1, 3, 4, 2), (3, 4, 0)) * -1.0
    del x53
    x70 = np.zeros((nocc, nvir), dtype=types[float])
    x70 += einsum(f.ov, (0, 1), (0, 1))
    x70 += einsum(x1, (0, 1), (0, 1)) * 2.0
    x70 += einsum(x22, (0, 1), (0, 1)) * -1.0
    x71 = np.zeros((nocc, nocc), dtype=types[float])
    x71 += einsum(f.oo, (0, 1), (0, 1))
    x71 += einsum(x68, (0, 1), (1, 0)) * 2.0
    x71 += einsum(v.xov, (0, 1, 2), x69, (3, 2, 0), (3, 1)) * -1.0
    del x69
    x71 += einsum(t1, (0, 1), x70, (2, 1), (0, 2))
    del x70
    l1new += einsum(l1, (0, 1), x71, (1, 2), (0, 2)) * -1.0
    del x71
    x72 = np.zeros((nvir, nvir), dtype=types[float])
    x72 += einsum(x0, (0,), v.xvv, (0, 1, 2), (1, 2))
    del x0
    x73 = np.zeros((nvir, nvir), dtype=types[float])
    x73 += einsum(f.vv, (0, 1), (0, 1)) * 0.5
    x73 += einsum(x72, (0, 1), (1, 0))
    l1new += einsum(l1, (0, 1), x73, (0, 2), (2, 1)) * 2.0
    del x73
    x74 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x74 += einsum(v.xvv, (0, 1, 2), v.xvv, (0, 3, 4), (1, 3, 4, 2))
    l2new += einsum(l2, (0, 1, 2, 3), x74, (4, 5, 1, 0), (4, 5, 2, 3))
    del x74
    x75 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum(f.vv, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum(f.ov, (0, 1), x8, (2, 3, 0, 4), (2, 3, 1, 4))
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum(l1, (0, 1), x26, (1, 2, 3, 4), (2, 3, 0, 4))
    x78 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x78 += einsum(x25, (0, 1, 2, 3), x8, (1, 4, 2, 5), (4, 0, 5, 3))
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum(l2, (0, 1, 2, 3), x32, (2, 4, 3, 5), (4, 5, 0, 1))
    del x32
    x80 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x80 += einsum(x26, (0, 1, 2, 3), x8, (0, 4, 1, 5), (4, 2, 5, 3))
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum(v.xvv, (0, 1, 2), x29, (3, 4, 0), (3, 4, 1, 2))
    del x29
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum(l2, (0, 1, 2, 3), x81, (4, 2, 5, 1), (3, 4, 0, 5))
    x83 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x83 += einsum(x2, (0, 1), v.xov, (2, 1, 3), (0, 3, 2))
    del x2
    x84 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x84 += einsum(x60, (0, 1, 2), (0, 1, 2))
    del x60
    x84 += einsum(x83, (0, 1, 2), (0, 1, 2)) * -1.0
    del x83
    x85 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x85 += einsum(v.xov, (0, 1, 2), x84, (3, 4, 0), (1, 3, 2, 4))
    del x84
    x86 = np.zeros((nocc, nocc), dtype=types[float])
    x86 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x87 = np.zeros((nocc, nocc), dtype=types[float])
    x87 += einsum(f.oo, (0, 1), (0, 1))
    x87 += einsum(x86, (0, 1), (1, 0))
    del x86
    x88 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x88 += einsum(x87, (0, 1), l2, (2, 3, 0, 4), (4, 1, 2, 3))
    del x87
    x89 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x89 += einsum(x75, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x75
    x89 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3))
    del x76
    x89 += einsum(x77, (0, 1, 2, 3), (0, 1, 2, 3))
    del x77
    x89 += einsum(x78, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x78
    x89 += einsum(x79, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x79
    x89 += einsum(x80, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x80
    x89 += einsum(x82, (0, 1, 2, 3), (0, 1, 2, 3))
    del x82
    x89 += einsum(x85, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x85
    x89 += einsum(x88, (0, 1, 2, 3), (1, 0, 3, 2))
    del x88
    l2new += einsum(x89, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new += einsum(x89, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    del x89
    x90 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x90 += einsum(l1, (0, 1), x25, (2, 1, 3, 4), (2, 3, 0, 4))
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x91 += einsum(x25, (0, 1, 2, 3), x8, (4, 1, 2, 5), (4, 0, 5, 3))
    x92 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x92 += einsum(x16, (0, 1, 2, 3), x8, (4, 5, 0, 2), (5, 4, 1, 3))
    del x16
    x93 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x93 += einsum(x26, (0, 1, 2, 3), x8, (4, 0, 1, 5), (4, 2, 5, 3))
    x94 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x94 += einsum(v.xov, (0, 1, 2), (1, 2, 0))
    x94 += einsum(x3, (0, 1, 2), (0, 1, 2))
    x95 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x95 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x95 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -0.5
    x96 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x96 += einsum(x94, (0, 1, 2), x95, (0, 3, 4, 1), (3, 4, 2)) * 2.0
    del x94, x95
    x97 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x97 += einsum(v.xov, (0, 1, 2), x96, (3, 4, 0), (1, 3, 2, 4))
    del x96
    x98 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x98 += einsum(l2, (0, 1, 2, 3), x81, (4, 3, 5, 1), (2, 4, 0, 5))
    del x81
    x99 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x99 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x99 += einsum(x8, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x100 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x100 += einsum(x25, (0, 1, 2, 3), x99, (4, 0, 1, 5), (2, 4, 3, 5))
    del x25, x99
    x101 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x101 += einsum(x26, (0, 1, 2, 3), x52, (0, 4, 2, 5), (1, 4, 3, 5))
    del x26, x52
    x102 = np.zeros((nvir, nvir), dtype=types[float])
    x102 += einsum(v.xov, (0, 1, 2), x3, (1, 3, 0), (2, 3))
    del x3
    x103 = np.zeros((nvir, nvir), dtype=types[float])
    x103 += einsum(x72, (0, 1), (1, 0)) * 2.0
    del x72
    x103 += einsum(x102, (0, 1), (0, 1)) * -1.0
    del x102
    x104 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x104 += einsum(x103, (0, 1), l2, (2, 1, 3, 4), (4, 3, 2, 0))
    del x103
    x105 = np.zeros((nocc, nocc), dtype=types[float])
    x105 += einsum(v.xoo, (0, 1, 2), x7, (2, 3, 0), (1, 3))
    del x7
    x106 = np.zeros((nocc, nocc), dtype=types[float])
    x106 += einsum(t1, (0, 1), x23, (2, 1), (0, 2))
    x107 = np.zeros((nocc, nocc), dtype=types[float])
    x107 += einsum(x68, (0, 1), (1, 0))
    del x68
    x107 += einsum(x105, (0, 1), (0, 1)) * -0.5
    del x105
    x107 += einsum(x106, (0, 1), (0, 1))
    del x106
    x108 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x108 += einsum(x107, (0, 1), l2, (2, 3, 0, 4), (4, 1, 2, 3)) * 2.0
    del x107
    x109 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x109 += einsum(x23, (0, 1), x8, (2, 3, 0, 4), (2, 3, 4, 1)) * 2.0
    del x8, x23
    x110 = np.zeros((nocc, nvir), dtype=types[float])
    x110 += einsum(x1, (0, 1), (0, 1)) * 2.0
    del x1
    x110 += einsum(x22, (0, 1), (0, 1)) * -1.0
    del x22
    x111 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x111 += einsum(f.ov, (0, 1), l1, (2, 3), (0, 3, 1, 2)) * -1.0
    x111 += einsum(x90, (0, 1, 2, 3), (0, 1, 2, 3))
    del x90
    x111 += einsum(x91, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x91
    x111 += einsum(x92, (0, 1, 2, 3), (0, 1, 2, 3))
    del x92
    x111 += einsum(x93, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x93
    x111 += einsum(x97, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x97
    x111 += einsum(x98, (0, 1, 2, 3), (0, 1, 2, 3))
    del x98
    x111 += einsum(x100, (0, 1, 2, 3), (1, 0, 3, 2))
    del x100
    x111 += einsum(x101, (0, 1, 2, 3), (1, 0, 3, 2))
    del x101
    x111 += einsum(x104, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x104
    x111 += einsum(x108, (0, 1, 2, 3), (0, 1, 3, 2))
    del x108
    x111 += einsum(x109, (0, 1, 2, 3), (0, 1, 2, 3))
    del x109
    x111 += einsum(l1, (0, 1), x110, (2, 3), (1, 2, 0, 3)) * -1.0
    del x110
    l2new += einsum(x111, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    l2new += einsum(x111, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x111
    x112 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x112 += einsum(x31, (0, 1, 2, 3), (3, 2, 1, 0))
    del x31
    x112 += einsum(x36, (0, 1, 2, 3), (0, 3, 1, 2))
    del x36
    l2new += einsum(l2, (0, 1, 2, 3), x112, (3, 4, 2, 5), (0, 1, 4, 5))
    del x112

    return {"l1new": l1new, "l2new": l2new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))

    # RDM1
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=types[float])
    rdm1_f_oo += einsum(delta.oo, (0, 1), (0, 1)) * 2.0
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=types[float])
    rdm1_f_ov += einsum(t1, (0, 1), (0, 1)) * 2.0
    rdm1_f_vo = np.zeros((nvir, nocc), dtype=types[float])
    rdm1_f_vo += einsum(l1, (0, 1), (0, 1)) * 2.0
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=types[float])
    rdm1_f_vv += einsum(l1, (0, 1), t1, (1, 2), (0, 2)) * 2.0
    x0 = np.zeros((nocc, nocc), dtype=types[float])
    x0 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    rdm1_f_oo += einsum(x0, (0, 1), (1, 0)) * -2.0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm1_f_oo += einsum(l2, (0, 1, 2, 3), x1, (3, 4, 0, 1), (4, 2)) * -2.0
    del x1
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x2 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x3 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x3 += einsum(x2, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x2
    rdm1_f_ov += einsum(t2, (0, 1, 2, 3), x3, (0, 1, 4, 2), (4, 3)) * -2.0
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x4 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    rdm1_f_ov += einsum(l1, (0, 1), x4, (1, 2, 3, 0), (2, 3)) * 4.0
    del x4
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x5 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x6 = np.zeros((nocc, nocc), dtype=types[float])
    x6 += einsum(x0, (0, 1), (0, 1)) * 0.5
    del x0
    x6 += einsum(l2, (0, 1, 2, 3), x5, (2, 4, 0, 1), (3, 4))
    del x5
    rdm1_f_ov += einsum(t1, (0, 1), x6, (0, 2), (2, 1)) * -4.0
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.5
    x7 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm1_f_vv += einsum(t2, (0, 1, 2, 3), x7, (0, 1, 4, 3), (4, 2)) * 4.0
    del x7

    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])

    return rdm1_f

def make_rdm2_f(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))

    # RDM2
    rdm2_f_oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_ovoo += einsum(delta.oo, (0, 1), l1, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovoo += einsum(delta.oo, (0, 1), l1, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_ovoo += einsum(delta.oo, (0, 1), l1, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovoo += einsum(delta.oo, (0, 1), l1, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovoo += einsum(delta.oo, (0, 1), l1, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovoo += einsum(delta.oo, (0, 1), l1, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_vooo += einsum(delta.oo, (0, 1), l1, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_vooo += einsum(delta.oo, (0, 1), l1, (2, 3), (2, 0, 3, 1))
    rdm2_f_vooo += einsum(delta.oo, (0, 1), l1, (2, 3), (2, 0, 3, 1))
    rdm2_f_vooo += einsum(delta.oo, (0, 1), l1, (2, 3), (2, 0, 3, 1))
    rdm2_f_vooo += einsum(delta.oo, (0, 1), l1, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_vooo += einsum(delta.oo, (0, 1), l1, (2, 3), (2, 0, 3, 1))
    rdm2_f_oovv = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovov = np.zeros((nocc, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_ovov += einsum(l1, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_ovov += einsum(l1, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_ovvo += einsum(l1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    rdm2_f_ovvo += einsum(l1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    rdm2_f_ovvo += einsum(l1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    rdm2_f_ovvo += einsum(l1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_voov += einsum(l1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2_f_voov += einsum(l1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2_f_voov += einsum(l1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2_f_voov += einsum(l1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_vovo += einsum(l1, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_vovo += einsum(l1, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 = np.zeros((nocc, nocc), dtype=types[float])
    x0 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x0, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x0, (2, 3), (3, 0, 1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x0, (2, 3), (0, 3, 2, 1))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x0, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x0, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x0, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x0, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x0, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x0, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x0, (2, 3), (3, 0, 1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x0, (2, 3), (0, 3, 2, 1))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x0, (2, 3), (3, 0, 2, 1)) * -1.0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x2 = np.zeros((nocc, nocc), dtype=types[float])
    x2 += einsum(l2, (0, 1, 2, 3), x1, (2, 4, 0, 1), (3, 4))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x2, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x2, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x2, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x2, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x2, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x2, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x2, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x2, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x2, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x2, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x2, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x2, (2, 3), (3, 0, 2, 1)) * -2.0
    x3 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x3 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_oooo += einsum(x3, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_oooo += einsum(x3, (0, 1, 2, 3), (3, 2, 1, 0))
    x4 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x4 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm2_f_ovoo += einsum(x4, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm2_f_ovoo += einsum(x4, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_ovoo += einsum(x4, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_ovoo += einsum(x4, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_ovoo += einsum(x4, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm2_f_ovoo += einsum(x4, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_vooo += einsum(x4, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x4, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_vooo += einsum(x4, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x4, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x4, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x4, (0, 1, 2, 3), (3, 2, 1, 0))
    x5 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x5 += einsum(t1, (0, 1), x4, (2, 3, 4, 1), (3, 2, 4, 0))
    rdm2_f_oooo += einsum(x5, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_oooo += einsum(x5, (0, 1, 2, 3), (3, 2, 1, 0))
    x6 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x6 += einsum(x3, (0, 1, 2, 3), (1, 0, 3, 2))
    del x3
    x6 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    del x5
    rdm2_f_oooo += einsum(x6, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_oooo += einsum(x6, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm2_f_oooo += einsum(x6, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_oooo += einsum(x6, (0, 1, 2, 3), (2, 3, 0, 1))
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x7 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov += einsum(x7, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_ooov += einsum(x7, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo += einsum(x7, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_oovo += einsum(x7, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x8 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x8 += einsum(x4, (0, 1, 2, 3), (1, 0, 2, 3))
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 += einsum(t2, (0, 1, 2, 3), x8, (1, 4, 5, 2), (4, 5, 0, 3))
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3))
    x10 += einsum(x4, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum(t2, (0, 1, 2, 3), x10, (1, 4, 5, 3), (4, 5, 0, 2)) * 2.0
    del x10
    x12 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x12 += einsum(t1, (0, 1), x6, (0, 2, 3, 4), (2, 3, 4, 1))
    rdm2_f_ooov += einsum(x12, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_ooov += einsum(x12, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_oovo += einsum(x12, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_oovo += einsum(x12, (0, 1, 2, 3), (1, 2, 3, 0))
    x13 = np.zeros((nocc, nocc), dtype=types[float])
    x13 += einsum(l2, (0, 1, 2, 3), x1, (2, 4, 0, 1), (3, 4)) * 2.0
    del x1
    x14 = np.zeros((nocc, nocc), dtype=types[float])
    x14 += einsum(x0, (0, 1), (0, 1))
    x14 += einsum(x13, (0, 1), (0, 1))
    rdm2_f_ooov += einsum(t1, (0, 1), x14, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_ooov += einsum(t1, (0, 1), x14, (2, 3), (3, 0, 2, 1)) * -1.0
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum(delta.oo, (0, 1), t1, (2, 3), (0, 1, 2, 3))
    x15 += einsum(x7, (0, 1, 2, 3), (1, 0, 2, 3))
    x15 += einsum(x9, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x9
    x15 += einsum(x11, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x11
    x15 += einsum(x12, (0, 1, 2, 3), (2, 0, 1, 3))
    x15 += einsum(t1, (0, 1), x14, (2, 3), (0, 2, 3, 1))
    rdm2_f_ooov += einsum(x15, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_ooov += einsum(x15, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(x15, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_ooov += einsum(x15, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_oovo += einsum(x15, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(x15, (0, 1, 2, 3), (2, 0, 3, 1))
    rdm2_f_oovo += einsum(x15, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(x15, (0, 1, 2, 3), (2, 0, 3, 1))
    del x15
    x16 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x16 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x16 += einsum(x4, (0, 1, 2, 3), (1, 0, 2, 3))
    x17 = np.zeros((nocc, nvir), dtype=types[float])
    x17 += einsum(t2, (0, 1, 2, 3), x16, (0, 1, 4, 3), (4, 2)) * 2.0
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x18 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x19 = np.zeros((nocc, nvir), dtype=types[float])
    x19 += einsum(l1, (0, 1), x18, (1, 2, 3, 0), (2, 3))
    x20 = np.zeros((nocc, nvir), dtype=types[float])
    x20 += einsum(t1, (0, 1), x14, (0, 2), (2, 1))
    x21 = np.zeros((nocc, nvir), dtype=types[float])
    x21 += einsum(x17, (0, 1), (0, 1))
    x21 += einsum(x19, (0, 1), (0, 1)) * -1.0
    x21 += einsum(x20, (0, 1), (0, 1))
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x21, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x21, (2, 3), (2, 0, 1, 3))
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x21, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x21, (2, 3), (2, 0, 1, 3))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x21, (2, 3), (0, 2, 3, 1))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x21, (2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x21, (2, 3), (0, 2, 3, 1))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x21, (2, 3), (2, 0, 3, 1)) * -1.0
    del x21
    x22 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x22 += einsum(t2, (0, 1, 2, 3), x4, (1, 4, 5, 2), (4, 5, 0, 3))
    rdm2_f_ooov += einsum(x22, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_ooov += einsum(x22, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_oovo += einsum(x22, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_oovo += einsum(x22, (0, 1, 2, 3), (2, 1, 3, 0))
    del x22
    x23 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x23 += einsum(t2, (0, 1, 2, 3), x4, (4, 1, 5, 2), (4, 5, 0, 3))
    rdm2_f_ooov += einsum(x23, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_ooov += einsum(x23, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_oovo += einsum(x23, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_oovo += einsum(x23, (0, 1, 2, 3), (1, 2, 3, 0))
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x24 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x24 += einsum(x4, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x25 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x25 += einsum(t2, (0, 1, 2, 3), x24, (4, 1, 5, 3), (4, 5, 0, 2))
    del x24
    rdm2_f_ooov += einsum(x25, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ooov += einsum(x25, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x25
    x26 = np.zeros((nocc, nvir), dtype=types[float])
    x26 += einsum(t1, (0, 1), (0, 1)) * -1.0
    x26 += einsum(x17, (0, 1), (0, 1))
    del x17
    x26 += einsum(x19, (0, 1), (0, 1)) * -1.0
    del x19
    x26 += einsum(x20, (0, 1), (0, 1))
    del x20
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x26, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x26, (2, 3), (0, 2, 1, 3)) * -1.0
    del x26
    x27 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x27 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x27 += einsum(x4, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum(t2, (0, 1, 2, 3), x27, (1, 4, 5, 3), (4, 5, 0, 2))
    del x27
    rdm2_f_oovo += einsum(x28, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_oovo += einsum(x28, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x28
    x29 = np.zeros((nocc, nocc), dtype=types[float])
    x29 += einsum(delta.oo, (0, 1), (0, 1)) * -1.0
    x29 += einsum(x0, (0, 1), (0, 1))
    x29 += einsum(x13, (0, 1), (0, 1))
    rdm2_f_oovo += einsum(t1, (0, 1), x29, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oovo += einsum(t1, (0, 1), x29, (2, 3), (0, 3, 1, 2)) * -1.0
    del x29
    x30 = np.zeros((nocc, nvir), dtype=types[float])
    x30 += einsum(t2, (0, 1, 2, 3), x16, (0, 1, 4, 3), (4, 2))
    x31 = np.zeros((nocc, nvir), dtype=types[float])
    x31 += einsum(l1, (0, 1), x18, (1, 2, 3, 0), (2, 3)) * 0.5
    x32 = np.zeros((nocc, nvir), dtype=types[float])
    x32 += einsum(t1, (0, 1), x14, (0, 2), (2, 1)) * 0.5
    del x14
    x33 = np.zeros((nocc, nvir), dtype=types[float])
    x33 += einsum(x30, (0, 1), (0, 1))
    x33 += einsum(x31, (0, 1), (0, 1)) * -1.0
    x33 += einsum(x32, (0, 1), (0, 1))
    del x32
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x33, (2, 3), (2, 0, 3, 1)) * -2.0
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x33, (2, 3), (2, 0, 3, 1)) * -2.0
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x34 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x34 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2_f_oovv += einsum(x34, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x34, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3))
    del x34
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum(x0, (0, 1), t2, (2, 0, 3, 4), (1, 2, 3, 4))
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.5
    x36 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum(t2, (0, 1, 2, 3), x36, (1, 4, 2, 5), (4, 0, 5, 3)) * 2.0
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum(t2, (0, 1, 2, 3), x37, (1, 4, 3, 5), (4, 0, 5, 2))
    del x37
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x39 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -0.5
    x40 = np.zeros((nvir, nvir), dtype=types[float])
    x40 += einsum(t2, (0, 1, 2, 3), x39, (0, 1, 4, 2), (4, 3)) * 2.0
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum(x40, (0, 1), t2, (2, 3, 0, 4), (2, 3, 1, 4))
    x42 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x42 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3))
    x42 += einsum(x4, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x43 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x43 += einsum(t2, (0, 1, 2, 3), x42, (4, 1, 5, 2), (4, 5, 0, 3))
    x44 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x44 += einsum(t2, (0, 1, 2, 3), x16, (4, 1, 5, 3), (4, 5, 0, 2)) * 2.0
    del x16
    x45 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x45 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    x45 += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x43
    x45 += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x44
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum(t1, (0, 1), x45, (0, 2, 3, 4), (2, 3, 4, 1))
    del x45
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x47 += einsum(x13, (0, 1), t2, (2, 0, 3, 4), (1, 2, 4, 3))
    del x13
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    x48 += einsum(x38, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x48 += einsum(x41, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x48 += einsum(x46, (0, 1, 2, 3), (0, 1, 3, 2))
    del x46
    x48 += einsum(x47, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x48 += einsum(t1, (0, 1), x33, (2, 3), (0, 2, 1, 3)) * -2.0
    del x33
    rdm2_f_oovv += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x48, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x48, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x48, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_oovv += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x48, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x48, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x48, (0, 1, 2, 3), (1, 0, 3, 2))
    del x48
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum(t2, (0, 1, 2, 3), x36, (1, 4, 3, 5), (4, 0, 5, 2))
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum(t2, (0, 1, 2, 3), x49, (1, 4, 3, 5), (4, 0, 5, 2)) * 4.0
    del x49
    x51 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x51 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x51 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x52 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x52 += einsum(t2, (0, 1, 2, 3), x51, (1, 4, 2, 5), (4, 0, 5, 3))
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x53 += einsum(t2, (0, 1, 2, 3), x52, (1, 4, 2, 5), (4, 0, 5, 3)) * -1.0
    del x52
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum(t2, (0, 1, 2, 3), x6, (0, 1, 4, 5), (4, 5, 2, 3))
    del x6
    rdm2_f_oovv += einsum(x54, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_oovv += einsum(x54, (0, 1, 2, 3), (1, 0, 3, 2))
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum(t1, (0, 1), x12, (0, 2, 3, 4), (3, 2, 4, 1))
    del x12
    rdm2_f_oovv += einsum(x55, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x55, (0, 1, 2, 3), (0, 1, 3, 2))
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum(x50, (0, 1, 2, 3), (0, 1, 2, 3))
    del x50
    x56 += einsum(x53, (0, 1, 2, 3), (1, 0, 3, 2))
    del x53
    x56 += einsum(x54, (0, 1, 2, 3), (1, 0, 3, 2))
    del x54
    x56 += einsum(x55, (0, 1, 2, 3), (1, 0, 2, 3))
    del x55
    rdm2_f_oovv += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x56, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x56, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x56
    x57 = np.zeros((nocc, nvir), dtype=types[float])
    x57 += einsum(t1, (0, 1), x0, (0, 2), (2, 1))
    del x0
    rdm2_f_oovv += einsum(t1, (0, 1), x57, (2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_oovv += einsum(t1, (0, 1), x57, (2, 3), (2, 0, 3, 1)) * -1.0
    x58 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x58 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 1, 5), (2, 4, 0, 5))
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum(t2, (0, 1, 2, 3), x58, (1, 4, 2, 5), (4, 0, 5, 3))
    del x58
    rdm2_f_oovv += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3))
    del x59
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), (3, 4, 0, 5))
    rdm2_f_ovov += einsum(x60, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x60, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x60, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x60, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum(t2, (0, 1, 2, 3), x60, (1, 4, 2, 5), (4, 0, 5, 3))
    rdm2_f_oovv += einsum(x61, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x61, (0, 1, 2, 3), (0, 1, 3, 2))
    del x61
    x62 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x62 += einsum(t2, (0, 1, 2, 3), x4, (4, 1, 5, 3), (4, 5, 0, 2))
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x63 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x64 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x64 += einsum(x4, (0, 1, 2, 3), x63, (0, 4, 3, 5), (4, 1, 2, 5))
    x65 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x65 += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3))
    del x62
    x65 += einsum(x64, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x64
    x66 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x66 += einsum(t1, (0, 1), x65, (0, 2, 3, 4), (2, 3, 4, 1))
    del x65
    x67 = np.zeros((nocc, nvir), dtype=types[float])
    x67 += einsum(t1, (0, 1), x2, (0, 2), (2, 1))
    del x2
    x68 = np.zeros((nocc, nvir), dtype=types[float])
    x68 += einsum(x30, (0, 1), (0, 1))
    del x30
    x68 += einsum(x31, (0, 1), (0, 1)) * -1.0
    del x31
    x68 += einsum(x67, (0, 1), (0, 1))
    del x67
    x69 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x69 += einsum(x38, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x38
    x69 += einsum(x41, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x41
    x69 += einsum(x66, (0, 1, 2, 3), (0, 1, 3, 2))
    del x66
    x69 += einsum(x47, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x47
    x69 += einsum(t1, (0, 1), x68, (2, 3), (0, 2, 1, 3)) * -2.0
    del x68
    rdm2_f_oovv += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x69, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_oovv += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x69, (0, 1, 2, 3), (1, 0, 3, 2))
    del x69
    x70 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x70 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    del x7
    x70 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x23
    x71 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x71 += einsum(t1, (0, 1), x70, (0, 2, 3, 4), (2, 3, 4, 1))
    del x70
    x72 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x72 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    del x35
    x72 += einsum(x71, (0, 1, 2, 3), (0, 1, 3, 2))
    del x71
    rdm2_f_oovv += einsum(x72, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x72, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x72, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x72, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x72
    x73 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum(t2, (0, 1, 2, 3), x36, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    del x36
    x74 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x74 += einsum(t2, (0, 1, 2, 3), x73, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    del x73
    rdm2_f_oovv += einsum(x74, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x74, (0, 1, 2, 3), (0, 1, 2, 3))
    del x74
    x75 = np.zeros((nocc, nvir), dtype=types[float])
    x75 += einsum(t1, (0, 1), (0, 1)) * -1.0
    x75 += einsum(x57, (0, 1), (0, 1))
    del x57
    rdm2_f_oovv += einsum(t1, (0, 1), x75, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_oovv += einsum(t1, (0, 1), x75, (2, 3), (0, 2, 1, 3)) * -1.0
    del x75
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x76 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum(l2, (0, 1, 2, 3), x76, (2, 4, 5, 1), (4, 3, 5, 0))
    rdm2_f_ovov += einsum(x77, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_ovov += einsum(x77, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    del x77
    x78 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x78 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x78 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum(l2, (0, 1, 2, 3), x78, (3, 4, 5, 1), (4, 2, 5, 0)) * 2.0
    del x78
    rdm2_f_ovov += einsum(x79, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_ovov += einsum(x79, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_voov += einsum(x79, (0, 1, 2, 3), (3, 0, 1, 2))
    rdm2_f_voov += einsum(x79, (0, 1, 2, 3), (3, 0, 1, 2))
    del x79
    x80 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x80 += einsum(t1, (0, 1), x8, (2, 0, 3, 4), (2, 3, 4, 1))
    del x8
    rdm2_f_ovov += einsum(x80, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x80, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x80, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x80, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x80
    x81 = np.zeros((nvir, nvir), dtype=types[float])
    x81 += einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    x82 = np.zeros((nvir, nvir), dtype=types[float])
    x82 += einsum(t2, (0, 1, 2, 3), x39, (0, 1, 4, 2), (4, 3))
    del x39
    x83 = np.zeros((nvir, nvir), dtype=types[float])
    x83 += einsum(x81, (0, 1), (0, 1)) * 0.5
    x83 += einsum(x82, (0, 1), (0, 1))
    del x82
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x83, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x83, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x83, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x83, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv += einsum(t1, (0, 1), x83, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_ovvv += einsum(t1, (0, 1), x83, (2, 3), (0, 2, 1, 3)) * 2.0
    del x83
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum(t1, (0, 1), x4, (0, 2, 3, 4), (2, 3, 4, 1))
    rdm2_f_ovov += einsum(x84, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x84, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x84, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x84, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x84
    x85 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x85 += einsum(t2, (0, 1, 2, 3), x51, (1, 4, 5, 2), (4, 0, 5, 3))
    del x51
    rdm2_f_ovvo += einsum(x85, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x85, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x85, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vovo += einsum(x85, (0, 1, 2, 3), (2, 1, 3, 0))
    del x85
    x86 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x86 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * 2.0
    x86 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x87 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x87 += einsum(t2, (0, 1, 2, 3), x86, (1, 4, 5, 3), (4, 0, 5, 2))
    del x86
    rdm2_f_ovvo += einsum(x87, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_ovvo += einsum(x87, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_vovo += einsum(x87, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x87, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x87
    x88 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x88 += einsum(t1, (0, 1), x42, (2, 0, 3, 4), (2, 3, 4, 1))
    del x42
    rdm2_f_ovvo += einsum(x88, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x88, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x88, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x88, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x88
    x89 = np.zeros((nvir, nvir), dtype=types[float])
    x89 += einsum(x81, (0, 1), (0, 1))
    del x81
    x89 += einsum(x40, (0, 1), (0, 1))
    del x40
    rdm2_f_ovvo += einsum(delta.oo, (0, 1), x89, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_ovvo += einsum(delta.oo, (0, 1), x89, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_voov += einsum(delta.oo, (0, 1), x89, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_voov += einsum(delta.oo, (0, 1), x89, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x89, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x89, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x89, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x89, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv += einsum(t1, (0, 1), x89, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovv += einsum(t1, (0, 1), x89, (2, 3), (2, 0, 3, 1))
    x90 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x90 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 5, 1), (3, 4, 0, 5))
    rdm2_f_ovvo += einsum(x90, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x90, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x90, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x90, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x91 += einsum(t1, (0, 1), x4, (2, 0, 3, 4), (2, 3, 4, 1))
    rdm2_f_ovvo += einsum(x91, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x91, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x91, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x91, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x92 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x92 += einsum(l2, (0, 1, 2, 3), x63, (3, 4, 1, 5), (4, 2, 5, 0))
    del x63
    rdm2_f_ovvo += einsum(x92, (0, 1, 2, 3), (0, 3, 2, 1))
    rdm2_f_ovvo += einsum(x92, (0, 1, 2, 3), (0, 3, 2, 1))
    rdm2_f_voov += einsum(x92, (0, 1, 2, 3), (3, 0, 1, 2))
    rdm2_f_voov += einsum(x92, (0, 1, 2, 3), (3, 0, 1, 2))
    del x92
    x93 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x93 += einsum(l2, (0, 1, 2, 3), x76, (2, 4, 1, 5), (4, 3, 5, 0))
    del x76
    rdm2_f_voov += einsum(x93, (0, 1, 2, 3), (3, 0, 1, 2)) * -1.0
    rdm2_f_voov += einsum(x93, (0, 1, 2, 3), (3, 0, 1, 2)) * -1.0
    del x93
    x94 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x94 += einsum(l1, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_ovvv += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vovv += einsum(x94, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vovv += einsum(x94, (0, 1, 2, 3), (1, 0, 3, 2))
    x95 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x95 += einsum(t2, (0, 1, 2, 3), x4, (0, 1, 4, 5), (4, 5, 2, 3))
    del x4
    rdm2_f_ovvv += einsum(x95, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_ovvv += einsum(x95, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x95, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x95, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x96 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x96 += einsum(l2, (0, 1, 2, 3), x18, (3, 4, 5, 1), (4, 2, 5, 0))
    x97 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x97 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x97 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x98 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x98 += einsum(l2, (0, 1, 2, 3), x97, (2, 4, 5, 1), (4, 3, 5, 0))
    del x97
    x99 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x99 += einsum(x91, (0, 1, 2, 3), (0, 1, 2, 3))
    x99 += einsum(x96, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x99 += einsum(x98, (0, 1, 2, 3), (1, 0, 3, 2))
    del x98
    x100 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x100 += einsum(t1, (0, 1), x99, (0, 2, 3, 4), (2, 3, 4, 1))
    del x99
    x101 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x101 += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3))
    del x94
    x101 += einsum(x95, (0, 1, 2, 3), (0, 1, 2, 3))
    del x95
    x101 += einsum(x100, (0, 1, 2, 3), (0, 1, 3, 2))
    del x100
    x101 += einsum(t1, (0, 1), x89, (2, 3), (0, 2, 1, 3))
    del x89
    rdm2_f_ovvv += einsum(x101, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x101, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_ovvv += einsum(x101, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x101, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x101, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x101, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vovv += einsum(x101, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x101, (0, 1, 2, 3), (1, 0, 3, 2))
    del x101
    x102 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x102 += einsum(t1, (0, 1), x60, (0, 2, 3, 4), (2, 3, 1, 4))
    del x60
    rdm2_f_ovvv += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_ovvv += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x102, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x102, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x102
    x103 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x103 += einsum(x90, (0, 1, 2, 3), (0, 1, 2, 3))
    x103 += einsum(x91, (0, 1, 2, 3), (0, 1, 2, 3))
    x103 += einsum(x96, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x96
    x104 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x104 += einsum(t1, (0, 1), x103, (0, 2, 3, 4), (2, 3, 4, 1))
    del x103
    rdm2_f_ovvv += einsum(x104, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_ovvv += einsum(x104, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x104
    x105 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x105 += einsum(l2, (0, 1, 2, 3), x18, (3, 4, 5, 1), (4, 2, 5, 0)) * 0.5
    del x18
    x106 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x106 += einsum(x90, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x90
    x106 += einsum(x91, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x91
    x106 += einsum(x105, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x105
    x107 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x107 += einsum(t1, (0, 1), x106, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    del x106
    rdm2_f_vovv += einsum(x107, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x107, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x107
    x108 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x108 += einsum(t1, (0, 1), l2, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov += einsum(x108, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvov += einsum(x108, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vvov += einsum(x108, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvov += einsum(x108, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvov += einsum(x108, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvov += einsum(x108, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo += einsum(x108, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_vvvo += einsum(x108, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vvvo += einsum(x108, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vvvo += einsum(x108, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vvvo += einsum(x108, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_vvvo += einsum(x108, (0, 1, 2, 3), (2, 1, 3, 0))
    x109 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x109 += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 5), (0, 1, 4, 5))
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv += einsum(x109, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vvvv += einsum(x109, (0, 1, 2, 3), (1, 0, 3, 2))
    x110 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x110 += einsum(t1, (0, 1), x108, (0, 2, 3, 4), (3, 2, 4, 1))
    del x108
    rdm2_f_vvvv += einsum(x110, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vvvv += einsum(x110, (0, 1, 2, 3), (1, 0, 3, 2))
    x111 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x111 += einsum(x109, (0, 1, 2, 3), (1, 0, 3, 2))
    del x109
    x111 += einsum(x110, (0, 1, 2, 3), (1, 0, 3, 2))
    del x110
    rdm2_f_vvvv += einsum(x111, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vvvv += einsum(x111, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvvv += einsum(x111, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vvvv += einsum(x111, (0, 1, 2, 3), (0, 1, 2, 3))
    del x111

    rdm2_f = pack_2e(rdm2_f_oooo, rdm2_f_ooov, rdm2_f_oovo, rdm2_f_ovoo, rdm2_f_vooo, rdm2_f_oovv, rdm2_f_ovov, rdm2_f_ovvo, rdm2_f_voov, rdm2_f_vovo, rdm2_f_vvoo, rdm2_f_ovvv, rdm2_f_vovv, rdm2_f_vvov, rdm2_f_vvvo, rdm2_f_vvvv)

    rdm2_f = rdm2_f.swapaxes(1, 2)

    return rdm2_f

