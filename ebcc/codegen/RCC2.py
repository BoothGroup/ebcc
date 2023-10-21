# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(f.ov, (0, 1), t1, (0, 1), ()) * 2.0
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x1 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    e_cc += einsum(x0, (0, 1, 2, 3), x1, (0, 1, 3, 2), ()) * 2.0
    del x0, x1

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (3, 2, 4, 1)) * -1.0
    x0 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x0 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    t1new += einsum(t2, (0, 1, 2, 3), x0, (1, 2, 3, 4), (0, 4)) * 2.0
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x2 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x2 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x2 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x2 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    t1new += einsum(t2, (0, 1, 2, 3), x2, (4, 1, 0, 2), (4, 3)) * -1.0
    del x2
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x3 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x4 = np.zeros((nocc, nvir), dtype=types[float])
    x4 += einsum(t1, (0, 1), x3, (0, 2, 3, 1), (2, 3)) * 2.0
    x5 = np.zeros((nocc, nvir), dtype=types[float])
    x5 += einsum(f.ov, (0, 1), (0, 1))
    x5 += einsum(x4, (0, 1), (0, 1))
    del x4
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t1new += einsum(x5, (0, 1), x6, (0, 2, 3, 1), (2, 3)) * 2.0
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x7 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t1new += einsum(t1, (0, 1), x7, (0, 2, 1, 3), (2, 3)) * 2.0
    del x7
    x8 = np.zeros((nvir, nvir), dtype=types[float])
    x8 += einsum(f.vv, (0, 1), (0, 1)) * 0.5
    x8 += einsum(t1, (0, 1), x0, (0, 2, 1, 3), (2, 3))
    del x0
    t1new += einsum(t1, (0, 1), x8, (1, 2), (0, 2)) * 2.0
    del x8
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x9 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x10 = np.zeros((nocc, nocc), dtype=types[float])
    x10 += einsum(f.oo, (0, 1), (0, 1))
    x10 += einsum(t2, (0, 1, 2, 3), x3, (1, 4, 2, 3), (4, 0)) * 2.0
    del x3
    x10 += einsum(t1, (0, 1), x9, (2, 3, 0, 1), (3, 2)) * 2.0
    del x9
    x10 += einsum(t1, (0, 1), x5, (2, 1), (2, 0))
    del x5
    t1new += einsum(t1, (0, 1), x10, (0, 2), (2, 1)) * -1.0
    del x10
    x11 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x11 += einsum(t1, (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new += einsum(t1, (0, 1), x11, (2, 3, 1, 4), (0, 2, 3, 4))
    del x11
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x14 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x14 += einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x16 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x16 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum(t1, (0, 1), x16, (2, 3, 4, 0), (2, 4, 3, 1))
    del x16
    x18 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x18 += einsum(x14, (0, 1, 2, 3), (0, 1, 2, 3))
    del x14
    x18 += einsum(x15, (0, 1, 2, 3), (2, 0, 1, 3))
    del x15
    x18 += einsum(x17, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x17
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(t1, (0, 1), x18, (0, 2, 3, 4), (2, 3, 4, 1))
    del x18
    x20 = np.zeros((nocc, nocc), dtype=types[float])
    x20 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x21 = np.zeros((nocc, nocc), dtype=types[float])
    x21 += einsum(f.oo, (0, 1), (0, 1))
    x21 += einsum(x20, (0, 1), (0, 1))
    del x20
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(x21, (0, 1), t2, (2, 0, 3, 4), (1, 2, 4, 3))
    del x21
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x12
    x23 += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x23 += einsum(x19, (0, 1, 2, 3), (0, 1, 3, 2))
    del x19
    x23 += einsum(x22, (0, 1, 2, 3), (0, 1, 3, 2))
    del x22
    t2new += einsum(x23, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x23
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x24 += einsum(x13, (0, 1, 2, 3), (1, 0, 2, 3))
    del x13
    x25 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x25 += einsum(t1, (0, 1), x24, (2, 3, 1, 4), (2, 3, 0, 4))
    del x24
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(t1, (0, 1), x25, (0, 2, 3, 4), (3, 2, 4, 1))
    del x25
    t2new += einsum(x26, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x26, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x26
    x27 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x27 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x27 += einsum(t1, (0, 1), x1, (2, 3, 4, 1), (3, 0, 2, 4))
    del x1
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x28 += einsum(t1, (0, 1), x27, (0, 2, 3, 4), (3, 2, 4, 1))
    del x27
    t2new += einsum(t1, (0, 1), x28, (2, 3, 0, 4), (2, 3, 1, 4))
    del x28

    return {"t1new": t1new, "t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    # L amplitudes
    l1new = np.zeros((nvir, nocc), dtype=types[float])
    l1new += einsum(f.ov, (0, 1), (1, 0))
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum(l2, (0, 1, 2, 3), v.vvvv, (4, 1, 5, 0), (4, 5, 3, 2))
    l2new += einsum(v.ovov, (0, 1, 2, 3), (1, 3, 0, 2))
    l2new += einsum(l2, (0, 1, 2, 3), v.oooo, (4, 2, 5, 3), (0, 1, 4, 5))
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x0 += einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x3 += einsum(t1, (0, 1), x2, (2, 3, 4, 1), (2, 0, 3, 4))
    x4 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x4 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x5 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x6 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x6 += einsum(t1, (0, 1), x5, (2, 3, 4, 1), (0, 2, 3, 4))
    l2new += einsum(l2, (0, 1, 2, 3), x6, (2, 3, 4, 5), (0, 1, 4, 5))
    x7 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x7 += einsum(x4, (0, 1, 2, 3), (1, 0, 3, 2))
    del x4
    x7 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    del x6
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x8 += einsum(t1, (0, 1), x7, (2, 3, 0, 4), (2, 3, 4, 1)) * 2.0
    del x7
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x1
    x9 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x3
    x9 += einsum(x8, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x8
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x10 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x11 = np.zeros((nocc, nvir), dtype=types[float])
    x11 += einsum(t1, (0, 1), x10, (0, 2, 3, 1), (2, 3))
    x12 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x12 += einsum(x11, (0, 1), t2, (2, 3, 1, 4), (2, 3, 0, 4)) * 2.0
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x13 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x13 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x13 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x13 += einsum(x5, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x14 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x15 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x16 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x16 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x17 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x17 += einsum(v.oooo, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x17 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x17 += einsum(x16, (0, 1, 2, 3), (2, 1, 0, 3))
    x17 += einsum(x16, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.5
    x18 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x18 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -2.0
    x18 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x18 += einsum(x0, (0, 1, 2, 3), (1, 2, 0, 3))
    x18 += einsum(x0, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    del x0
    x18 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x18 += einsum(x9, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    del x9
    x18 += einsum(x12, (0, 1, 2, 3), (1, 0, 2, 3))
    x18 += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x12
    x18 += einsum(x13, (0, 1, 2, 3), x14, (1, 4, 3, 5), (0, 4, 2, 5)) * -2.0
    del x13
    x18 += einsum(t1, (0, 1), x15, (2, 3, 1, 4), (0, 3, 2, 4)) * -1.0
    del x15
    x18 += einsum(t1, (0, 1), x17, (0, 2, 3, 4), (3, 2, 4, 1)) * 2.0
    del x17
    l1new += einsum(l2, (0, 1, 2, 3), x18, (2, 3, 4, 1), (0, 4))
    del x18
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x19 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x20 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x21 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x21 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x21 += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3))
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(l2, (0, 1, 2, 3), x14, (3, 4, 1, 5), (2, 4, 0, 5)) * 2.0
    x22 += einsum(l2, (0, 1, 2, 3), x19, (2, 4, 1, 5), (3, 4, 0, 5))
    x22 += einsum(t1, (0, 1), x21, (2, 0, 3, 4), (2, 3, 1, 4)) * 2.0
    l1new += einsum(v.ovvv, (0, 1, 2, 3), x22, (4, 0, 3, 2), (1, 4)) * -1.0
    del x22
    x23 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x23 += einsum(t1, (0, 1), l2, (2, 3, 4, 0), (4, 2, 3, 1))
    x24 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x24 += einsum(v.vvvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x24 += einsum(v.vvvv, (0, 1, 2, 3), (0, 1, 2, 3))
    l1new += einsum(x23, (0, 1, 2, 3), x24, (1, 4, 2, 3), (4, 0)) * 2.0
    del x23, x24
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.5
    x25 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x26 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x27 += einsum(x25, (0, 1, 2, 3), x26, (0, 4, 2, 5), (1, 4, 3, 5)) * -2.0
    del x25, x26
    l1new += einsum(v.ovvv, (0, 1, 2, 3), x27, (4, 0, 3, 1), (2, 4)) * -1.0
    del x27
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), (1, 2, 3, 4))
    x29 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x29 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 1, 0), (3, 2, 4, 5))
    x30 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x30 += einsum(t1, (0, 1), x20, (2, 3, 4, 1), (3, 2, 4, 0))
    l2new += einsum(v.ovov, (0, 1, 2, 3), x30, (4, 5, 0, 2), (3, 1, 5, 4))
    x31 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x31 += einsum(x29, (0, 1, 2, 3), (1, 0, 3, 2))
    x31 += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3))
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x32 += einsum(t1, (0, 1), x31, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x31
    x33 = np.zeros((nocc, nocc), dtype=types[float])
    x33 += einsum(l2, (0, 1, 2, 3), x14, (2, 4, 0, 1), (3, 4)) * 2.0
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum(x28, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x34 += einsum(x28, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    del x28
    x34 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x34 += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x34 += einsum(x32, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x34 += einsum(x32, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x32
    x34 += einsum(x19, (0, 1, 2, 3), x21, (4, 0, 5, 3), (4, 1, 5, 2)) * -1.0
    x34 += einsum(t1, (0, 1), x33, (2, 3), (2, 0, 3, 1)) * -1.0
    l1new += einsum(v.ovov, (0, 1, 2, 3), x34, (4, 0, 2, 1), (3, 4)) * 2.0
    del x34
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x35 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x35 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x35 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    x35 += einsum(x5, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x36 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x36 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x36 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x36 += einsum(x5, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x37 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x38 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x38 += einsum(x16, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x38 += einsum(x16, (0, 1, 2, 3), (0, 3, 2, 1))
    x39 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x39 += einsum(t2, (0, 1, 2, 3), x35, (4, 1, 5, 3), (4, 0, 5, 2))
    del x35
    x39 += einsum(t2, (0, 1, 2, 3), x36, (4, 1, 5, 2), (4, 0, 5, 3)) * 0.5
    del x36
    x39 += einsum(t1, (0, 1), x37, (2, 3, 1, 4), (0, 3, 2, 4)) * -0.5
    del x37
    x39 += einsum(t1, (0, 1), x38, (2, 0, 3, 4), (2, 3, 4, 1))
    del x38
    l1new += einsum(l2, (0, 1, 2, 3), x39, (3, 2, 4, 1), (0, 4)) * 2.0
    del x39
    x40 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x40 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x40 += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x41 += einsum(t1, (0, 1), x40, (2, 0, 3, 4), (2, 3, 4, 1)) * -0.5
    l1new += einsum(v.ovvv, (0, 1, 2, 3), x41, (4, 0, 3, 1), (2, 4)) * 2.0
    del x41
    x42 = np.zeros((nocc, nocc), dtype=types[float])
    x42 += einsum(l2, (0, 1, 2, 3), x14, (2, 4, 0, 1), (3, 4))
    x43 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x43 += einsum(t2, (0, 1, 2, 3), x21, (4, 1, 5, 3), (4, 0, 5, 2)) * 2.0
    del x21
    x43 += einsum(t2, (0, 1, 2, 3), x40, (4, 1, 5, 2), (4, 0, 5, 3))
    x43 += einsum(t1, (0, 1), x42, (2, 3), (2, 0, 3, 1)) * 2.0
    del x42
    l1new += einsum(v.ovov, (0, 1, 2, 3), x43, (4, 0, 2, 3), (1, 4))
    del x43
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x44 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x45 = np.zeros((nvir, nvir), dtype=types[float])
    x45 += einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    x45 += einsum(l2, (0, 1, 2, 3), x44, (2, 3, 1, 4), (0, 4)) * 2.0
    x46 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x46 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x46 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    l1new += einsum(x45, (0, 1), x46, (2, 1, 0, 3), (3, 2)) * 2.0
    del x45
    x47 = np.zeros((nocc, nocc), dtype=types[float])
    x47 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    l1new += einsum(x11, (0, 1), x47, (2, 0), (1, 2)) * -2.0
    x48 = np.zeros((nocc, nocc), dtype=types[float])
    x48 += einsum(x47, (0, 1), (0, 1))
    x48 += einsum(x33, (0, 1), (0, 1))
    del x33
    l1new += einsum(f.ov, (0, 1), x48, (2, 0), (1, 2)) * -1.0
    x49 = np.zeros((nocc, nvir), dtype=types[float])
    x49 += einsum(t1, (0, 1), (0, 1)) * -1.0
    x49 += einsum(x14, (0, 1, 2, 3), x20, (0, 1, 4, 2), (4, 3)) * 2.0
    del x14
    x49 += einsum(l1, (0, 1), x19, (1, 2, 3, 0), (2, 3)) * -1.0
    del x19
    x49 += einsum(t1, (0, 1), x48, (0, 2), (2, 1))
    l1new += einsum(x49, (0, 1), x10, (0, 2, 3, 1), (3, 2)) * -2.0
    del x49
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x50 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3))
    x50 += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    l1new += einsum(v.oovv, (0, 1, 2, 3), x50, (4, 0, 1, 3), (2, 4)) * -2.0
    del x50
    x51 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x51 += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x51 += einsum(x30, (0, 1, 2, 3), (0, 1, 3, 2))
    del x30
    l1new += einsum(v.ooov, (0, 1, 2, 3), x51, (4, 0, 1, 2), (3, 4)) * 2.0
    del x51
    x52 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x52 += einsum(x29, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x52 += einsum(x29, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x29
    l1new += einsum(v.ooov, (0, 1, 2, 3), x52, (1, 4, 2, 0), (3, 4))
    del x52
    x53 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x53 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x53 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    l1new += einsum(x48, (0, 1), x53, (0, 2, 1, 3), (3, 2)) * -1.0
    del x48, x53
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x54 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    l1new += einsum(l1, (0, 1), x54, (1, 2, 0, 3), (3, 2)) * 2.0
    del x54
    x55 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x55 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x55 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x56 = np.zeros((nvir, nvir), dtype=types[float])
    x56 += einsum(f.vv, (0, 1), (0, 1)) * 0.5
    x56 += einsum(t1, (0, 1), x55, (0, 2, 1, 3), (3, 2))
    l1new += einsum(l1, (0, 1), x56, (0, 2), (2, 1)) * 2.0
    del x56
    x57 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x57 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x57 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -0.5
    x58 = np.zeros((nocc, nvir), dtype=types[float])
    x58 += einsum(t1, (0, 1), x10, (0, 2, 3, 1), (2, 3)) * 2.0
    del x10
    x59 = np.zeros((nocc, nvir), dtype=types[float])
    x59 += einsum(f.ov, (0, 1), (0, 1))
    x59 += einsum(x58, (0, 1), (0, 1))
    x60 = np.zeros((nocc, nocc), dtype=types[float])
    x60 += einsum(f.oo, (0, 1), (0, 1))
    x60 += einsum(v.ovov, (0, 1, 2, 3), x44, (2, 4, 1, 3), (4, 0)) * 2.0
    del x44
    x60 += einsum(t1, (0, 1), x57, (2, 3, 0, 1), (3, 2)) * 2.0
    x60 += einsum(t1, (0, 1), x59, (2, 1), (0, 2))
    del x59
    l1new += einsum(l1, (0, 1), x60, (1, 2), (0, 2)) * -1.0
    del x60
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum(l1, (0, 1), v.ooov, (2, 1, 3, 4), (2, 3, 0, 4))
    x62 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x62 += einsum(v.ooov, (0, 1, 2, 3), x20, (4, 1, 2, 5), (4, 0, 5, 3))
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum(v.ovvv, (0, 1, 2, 3), x20, (4, 5, 0, 3), (5, 4, 1, 2))
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum(x20, (0, 1, 2, 3), x5, (1, 2, 4, 5), (0, 4, 3, 5))
    x65 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x65 += einsum(t1, (0, 1), x46, (2, 1, 3, 4), (0, 2, 3, 4)) * 2.0
    del x46
    x66 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x66 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * 2.0
    x66 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x66 += einsum(x65, (0, 1, 2, 3), (0, 1, 3, 2))
    del x65
    x67 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x67 += einsum(l2, (0, 1, 2, 3), x66, (3, 4, 5, 1), (2, 4, 0, 5))
    del x66
    x68 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x68 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x69 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x69 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x69 += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3))
    del x68
    x70 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x70 += einsum(l2, (0, 1, 2, 3), x69, (2, 4, 5, 1), (3, 4, 0, 5))
    del x69
    x71 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x71 += einsum(v.ooov, (0, 1, 2, 3), x40, (4, 0, 1, 5), (2, 4, 3, 5))
    del x40
    x72 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x72 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x72 += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x73 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum(x5, (0, 1, 2, 3), x72, (0, 4, 2, 5), (1, 4, 3, 5))
    del x72
    x74 = np.zeros((nvir, nvir), dtype=types[float])
    x74 += einsum(t1, (0, 1), x55, (0, 2, 1, 3), (2, 3)) * 2.0
    del x55
    x75 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum(x74, (0, 1), l2, (2, 1, 3, 4), (4, 3, 2, 0))
    del x74
    x76 = np.zeros((nocc, nocc), dtype=types[float])
    x76 += einsum(t1, (0, 1), x57, (2, 3, 0, 1), (2, 3))
    del x57
    x77 = np.zeros((nocc, nocc), dtype=types[float])
    x77 += einsum(t1, (0, 1), x11, (2, 1), (0, 2))
    x78 = np.zeros((nocc, nocc), dtype=types[float])
    x78 += einsum(x76, (0, 1), (1, 0))
    del x76
    x78 += einsum(x77, (0, 1), (0, 1))
    del x77
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum(x78, (0, 1), l2, (2, 3, 0, 4), (4, 1, 2, 3)) * 2.0
    del x78
    x80 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x80 += einsum(x11, (0, 1), x20, (2, 3, 0, 4), (2, 3, 4, 1)) * 2.0
    del x11
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum(f.ov, (0, 1), l1, (2, 3), (0, 3, 1, 2)) * -1.0
    x81 += einsum(x61, (0, 1, 2, 3), (0, 1, 2, 3))
    del x61
    x81 += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x62
    x81 += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3))
    del x63
    x81 += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x64
    x81 += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x67
    x81 += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3))
    del x70
    x81 += einsum(x71, (0, 1, 2, 3), (1, 0, 3, 2))
    del x71
    x81 += einsum(x73, (0, 1, 2, 3), (1, 0, 3, 2))
    del x73
    x81 += einsum(x75, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x75
    x81 += einsum(x79, (0, 1, 2, 3), (0, 1, 3, 2))
    del x79
    x81 += einsum(x80, (0, 1, 2, 3), (0, 1, 2, 3))
    del x80
    x81 += einsum(l1, (0, 1), x58, (2, 3), (1, 2, 0, 3)) * -1.0
    del x58
    l2new += einsum(x81, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    l2new += einsum(x81, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x81
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum(f.vv, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x83 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x83 += einsum(l1, (0, 1), v.ovvv, (2, 3, 4, 0), (1, 2, 3, 4))
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum(f.ov, (0, 1), x20, (2, 3, 0, 4), (2, 3, 1, 4))
    x85 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x85 += einsum(l1, (0, 1), x5, (1, 2, 3, 4), (2, 3, 0, 4))
    x86 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x86 += einsum(x47, (0, 1), v.ovov, (2, 3, 1, 4), (0, 2, 3, 4))
    del x47
    x87 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x87 += einsum(l2, (0, 1, 2, 3), x16, (2, 4, 3, 5), (4, 5, 0, 1))
    del x16
    x88 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x88 += einsum(v.ooov, (0, 1, 2, 3), x20, (1, 4, 2, 5), (4, 0, 5, 3))
    x89 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x89 += einsum(x20, (0, 1, 2, 3), x5, (0, 2, 4, 5), (1, 4, 3, 5))
    del x5, x20
    x90 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x90 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x90 += einsum(x2, (0, 1, 2, 3), (0, 1, 3, 2))
    del x2
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x91 += einsum(l2, (0, 1, 2, 3), x90, (2, 4, 1, 5), (3, 4, 0, 5))
    del x90
    x92 = np.zeros((nocc, nocc), dtype=types[float])
    x92 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x93 = np.zeros((nocc, nocc), dtype=types[float])
    x93 += einsum(f.oo, (0, 1), (0, 1))
    x93 += einsum(x92, (0, 1), (1, 0))
    del x92
    x94 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x94 += einsum(x93, (0, 1), l2, (2, 3, 0, 4), (4, 1, 2, 3))
    del x93
    x95 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x95 += einsum(x82, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x82
    x95 += einsum(x83, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x83
    x95 += einsum(x84, (0, 1, 2, 3), (0, 1, 2, 3))
    del x84
    x95 += einsum(x85, (0, 1, 2, 3), (0, 1, 2, 3))
    del x85
    x95 += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3))
    del x86
    x95 += einsum(x87, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x87
    x95 += einsum(x88, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x88
    x95 += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x89
    x95 += einsum(x91, (0, 1, 2, 3), (0, 1, 2, 3))
    del x91
    x95 += einsum(x94, (0, 1, 2, 3), (1, 0, 3, 2))
    del x94
    l2new += einsum(x95, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new += einsum(x95, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    del x95

    return {"l1new": l1new, "l2new": l2new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
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
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm1_f_oo += einsum(l2, (0, 1, 2, 3), x1, (2, 4, 0, 1), (4, 3)) * -2.0
    del x1
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x2 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x3 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x3 += einsum(x2, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x2
    rdm1_f_ov += einsum(t2, (0, 1, 2, 3), x3, (1, 0, 4, 2), (4, 3)) * -2.0
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
    x7 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x7 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -0.5
    rdm1_f_vv += einsum(t2, (0, 1, 2, 3), x7, (0, 1, 4, 2), (4, 3)) * 4.0
    del x7

    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])

    return rdm1_f

def make_rdm2_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
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
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x0 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 1, 0), (3, 2, 4, 5))
    rdm2_f_oooo += einsum(x0, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_oooo += einsum(x0, (0, 1, 2, 3), (3, 2, 1, 0))
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm2_f_ovoo += einsum(x1, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm2_f_ovoo += einsum(x1, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_ovoo += einsum(x1, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_ovoo += einsum(x1, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_ovoo += einsum(x1, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm2_f_ovoo += einsum(x1, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_vooo += einsum(x1, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x1, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_vooo += einsum(x1, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x1, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x1, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x1, (0, 1, 2, 3), (3, 2, 1, 0))
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum(t1, (0, 1), x1, (2, 3, 4, 1), (2, 3, 0, 4))
    rdm2_f_oooo += einsum(x2, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_oooo += einsum(x2, (0, 1, 2, 3), (3, 2, 1, 0))
    x3 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x3 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2))
    x3 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oooo += einsum(x3, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_oooo += einsum(x3, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm2_f_oooo += einsum(x3, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_oooo += einsum(x3, (0, 1, 2, 3), (2, 3, 0, 1))
    x4 = np.zeros((nocc, nocc), dtype=types[float])
    x4 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (3, 0, 1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (0, 3, 2, 1))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (3, 0, 1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (0, 3, 2, 1))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (3, 0, 2, 1)) * -1.0
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x5 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x6 = np.zeros((nocc, nocc), dtype=types[float])
    x6 += einsum(l2, (0, 1, 2, 3), x5, (2, 4, 0, 1), (3, 4))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x6, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x6, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x6, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x6, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x6, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x6, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x6, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x6, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x6, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x6, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x6, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x6, (2, 3), (3, 0, 2, 1)) * -2.0
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x7 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    x7 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x8 = np.zeros((nocc, nvir), dtype=types[float])
    x8 += einsum(t2, (0, 1, 2, 3), x7, (1, 0, 4, 3), (4, 2)) * 2.0
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x9 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x10 = np.zeros((nocc, nvir), dtype=types[float])
    x10 += einsum(l1, (0, 1), x9, (1, 2, 3, 0), (2, 3))
    x11 = np.zeros((nocc, nocc), dtype=types[float])
    x11 += einsum(l2, (0, 1, 2, 3), x5, (2, 4, 0, 1), (3, 4)) * 2.0
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum(x4, (0, 1), (0, 1))
    x12 += einsum(x11, (0, 1), (0, 1))
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov += einsum(t1, (0, 1), x12, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_ooov += einsum(t1, (0, 1), x12, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo += einsum(t1, (0, 1), x12, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oovo += einsum(t1, (0, 1), x12, (2, 3), (0, 3, 1, 2)) * -1.0
    x13 = np.zeros((nocc, nvir), dtype=types[float])
    x13 += einsum(t1, (0, 1), x12, (0, 2), (2, 1))
    x14 = np.zeros((nocc, nvir), dtype=types[float])
    x14 += einsum(x8, (0, 1), (0, 1))
    x14 += einsum(x10, (0, 1), (0, 1)) * -1.0
    x14 += einsum(x13, (0, 1), (0, 1))
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x14, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x14, (2, 3), (2, 0, 1, 3))
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x14, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x14, (2, 3), (2, 0, 1, 3))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x14, (2, 3), (0, 2, 3, 1))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x14, (2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x14, (2, 3), (0, 2, 3, 1))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x14, (2, 3), (2, 0, 3, 1)) * -1.0
    del x14
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_ooov += einsum(x15, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_ooov += einsum(x15, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_oovo += einsum(x15, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_oovo += einsum(x15, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    x16 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x16 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    x16 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum(t2, (0, 1, 2, 3), x16, (4, 1, 5, 2), (4, 5, 0, 3))
    x18 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x18 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x18 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3))
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x19 += einsum(t2, (0, 1, 2, 3), x18, (4, 1, 5, 3), (4, 5, 0, 2)) * 2.0
    del x18
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x20 += einsum(t1, (0, 1), x3, (0, 2, 3, 4), (2, 3, 4, 1))
    del x3
    rdm2_f_ooov += einsum(x20, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_ooov += einsum(x20, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_oovo += einsum(x20, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_oovo += einsum(x20, (0, 1, 2, 3), (1, 2, 3, 0))
    x21 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x21 += einsum(delta.oo, (0, 1), t1, (2, 3), (0, 1, 2, 3))
    x21 += einsum(x15, (0, 1, 2, 3), (1, 0, 2, 3))
    x21 += einsum(x17, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x21 += einsum(x19, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x21 += einsum(x20, (0, 1, 2, 3), (2, 0, 1, 3))
    x21 += einsum(t1, (0, 1), x12, (2, 3), (0, 2, 3, 1))
    rdm2_f_ooov += einsum(x21, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_ooov += einsum(x21, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(x21, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_ooov += einsum(x21, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_oovo += einsum(x21, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(x21, (0, 1, 2, 3), (2, 0, 3, 1))
    rdm2_f_oovo += einsum(x21, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(x21, (0, 1, 2, 3), (2, 0, 3, 1))
    del x21
    x22 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x22 += einsum(t2, (0, 1, 2, 3), x1, (1, 4, 5, 2), (4, 5, 0, 3))
    rdm2_f_ooov += einsum(x22, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_ooov += einsum(x22, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_oovo += einsum(x22, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_oovo += einsum(x22, (0, 1, 2, 3), (2, 1, 3, 0))
    del x22
    x23 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x23 += einsum(t2, (0, 1, 2, 3), x1, (4, 1, 5, 2), (4, 5, 0, 3))
    rdm2_f_ooov += einsum(x23, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_ooov += einsum(x23, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_oovo += einsum(x23, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_oovo += einsum(x23, (0, 1, 2, 3), (1, 2, 3, 0))
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x24 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x24 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x25 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x25 += einsum(t2, (0, 1, 2, 3), x24, (4, 1, 5, 3), (4, 5, 0, 2))
    del x24
    rdm2_f_ooov += einsum(x25, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ooov += einsum(x25, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x25
    x26 = np.zeros((nocc, nvir), dtype=types[float])
    x26 += einsum(t1, (0, 1), (0, 1)) * -1.0
    x26 += einsum(x8, (0, 1), (0, 1))
    del x8
    x26 += einsum(x10, (0, 1), (0, 1)) * -1.0
    del x10
    x26 += einsum(x13, (0, 1), (0, 1))
    del x13
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x26, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x26, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x26, (2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x26, (2, 3), (2, 0, 3, 1)) * -1.0
    del x26
    x27 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x27 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x27 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum(t2, (0, 1, 2, 3), x27, (1, 4, 5, 3), (4, 5, 0, 2))
    del x27
    rdm2_f_oovo += einsum(x28, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_oovo += einsum(x28, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x28
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x29 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2_f_oovv += einsum(x29, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x29, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3))
    del x29
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum(x4, (0, 1), t2, (2, 0, 3, 4), (1, 2, 3, 4))
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.5
    x31 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum(t2, (0, 1, 2, 3), x31, (1, 4, 2, 5), (4, 0, 5, 3)) * 2.0
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum(t2, (0, 1, 2, 3), x32, (1, 4, 3, 5), (4, 0, 5, 2))
    del x32
    x34 = np.zeros((nvir, nvir), dtype=types[float])
    x34 += einsum(t2, (0, 1, 2, 3), x31, (0, 1, 4, 3), (4, 2)) * 2.0
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum(x34, (0, 1), t2, (2, 3, 0, 4), (2, 3, 1, 4))
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x36 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    x36 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x17
    x36 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x19
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum(t1, (0, 1), x36, (0, 2, 3, 4), (2, 3, 4, 1))
    del x36
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum(x11, (0, 1), t2, (2, 0, 3, 4), (1, 2, 4, 3))
    del x11
    x39 = np.zeros((nocc, nvir), dtype=types[float])
    x39 += einsum(t2, (0, 1, 2, 3), x7, (1, 0, 4, 3), (4, 2))
    del x7
    x40 = np.zeros((nocc, nvir), dtype=types[float])
    x40 += einsum(l1, (0, 1), x9, (1, 2, 3, 0), (2, 3)) * 0.5
    x41 = np.zeros((nocc, nvir), dtype=types[float])
    x41 += einsum(t1, (0, 1), x12, (0, 2), (2, 1)) * 0.5
    del x12
    x42 = np.zeros((nocc, nvir), dtype=types[float])
    x42 += einsum(x39, (0, 1), (0, 1))
    x42 += einsum(x40, (0, 1), (0, 1)) * -1.0
    x42 += einsum(x41, (0, 1), (0, 1))
    del x41
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3))
    x43 += einsum(x33, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x43 += einsum(x35, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x43 += einsum(x37, (0, 1, 2, 3), (0, 1, 3, 2))
    del x37
    x43 += einsum(x38, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x43 += einsum(t1, (0, 1), x42, (2, 3), (0, 2, 1, 3)) * -2.0
    del x42
    rdm2_f_oovv += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x43, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x43, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x43, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_oovv += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x43, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x43, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x43, (0, 1, 2, 3), (1, 0, 3, 2))
    del x43
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum(t2, (0, 1, 2, 3), x0, (0, 1, 4, 5), (4, 5, 2, 3))
    del x0
    rdm2_f_oovv += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3))
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum(t2, (0, 1, 2, 3), x2, (1, 0, 4, 5), (4, 5, 3, 2))
    del x2
    rdm2_f_oovv += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum(t2, (0, 1, 2, 3), x31, (1, 4, 3, 5), (4, 0, 5, 2))
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x47 += einsum(t2, (0, 1, 2, 3), x46, (1, 4, 3, 5), (4, 0, 5, 2)) * 4.0
    del x46
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x48 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum(t2, (0, 1, 2, 3), x48, (1, 4, 2, 5), (4, 0, 5, 3))
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum(t2, (0, 1, 2, 3), x49, (1, 4, 2, 5), (4, 0, 5, 3)) * -1.0
    del x49
    x51 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x51 += einsum(t1, (0, 1), x20, (0, 2, 3, 4), (3, 2, 4, 1))
    del x20
    rdm2_f_oovv += einsum(x51, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x51, (0, 1, 2, 3), (0, 1, 3, 2))
    x52 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x52 += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3))
    del x44
    x52 += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    del x45
    x52 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3))
    del x47
    x52 += einsum(x50, (0, 1, 2, 3), (1, 0, 3, 2))
    del x50
    x52 += einsum(x51, (0, 1, 2, 3), (0, 1, 3, 2))
    del x51
    rdm2_f_oovv += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x52, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x52, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x52
    x53 = np.zeros((nocc, nvir), dtype=types[float])
    x53 += einsum(t1, (0, 1), x4, (0, 2), (2, 1))
    del x4
    rdm2_f_oovv += einsum(t1, (0, 1), x53, (2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_oovv += einsum(t1, (0, 1), x53, (2, 3), (2, 0, 3, 1)) * -1.0
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 1, 5), (2, 4, 0, 5))
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum(t2, (0, 1, 2, 3), x54, (1, 4, 2, 5), (4, 0, 5, 3))
    del x54
    rdm2_f_oovv += einsum(x55, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x55, (0, 1, 2, 3), (0, 1, 2, 3))
    del x55
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), (3, 4, 0, 5))
    rdm2_f_ovov += einsum(x56, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x56, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x56, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x56, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum(t2, (0, 1, 2, 3), x56, (1, 4, 2, 5), (4, 0, 5, 3))
    rdm2_f_oovv += einsum(x57, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x57, (0, 1, 2, 3), (0, 1, 3, 2))
    del x57
    x58 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x58 += einsum(t2, (0, 1, 2, 3), x1, (4, 1, 5, 3), (4, 5, 0, 2))
    x59 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x59 += einsum(x1, (0, 1, 2, 3), x9, (0, 4, 5, 3), (4, 1, 2, 5))
    x60 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x60 += einsum(x58, (0, 1, 2, 3), (0, 1, 2, 3))
    del x58
    x60 += einsum(x59, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x59
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum(t1, (0, 1), x60, (0, 2, 3, 4), (2, 3, 4, 1))
    del x60
    x62 = np.zeros((nocc, nvir), dtype=types[float])
    x62 += einsum(t1, (0, 1), x6, (0, 2), (2, 1))
    del x6
    x63 = np.zeros((nocc, nvir), dtype=types[float])
    x63 += einsum(x39, (0, 1), (0, 1))
    del x39
    x63 += einsum(x40, (0, 1), (0, 1)) * -1.0
    del x40
    x63 += einsum(x62, (0, 1), (0, 1))
    del x62
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum(x33, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x33
    x64 += einsum(x35, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x35
    x64 += einsum(x61, (0, 1, 2, 3), (0, 1, 3, 2))
    del x61
    x64 += einsum(x38, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x38
    x64 += einsum(t1, (0, 1), x63, (2, 3), (0, 2, 1, 3)) * -2.0
    del x63
    rdm2_f_oovv += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x64, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_oovv += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x64, (0, 1, 2, 3), (1, 0, 3, 2))
    del x64
    x65 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x65 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    del x15
    x65 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x23
    x66 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x66 += einsum(t1, (0, 1), x65, (0, 2, 3, 4), (2, 3, 4, 1))
    del x65
    x67 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x67 += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3))
    del x30
    x67 += einsum(x66, (0, 1, 2, 3), (0, 1, 3, 2))
    del x66
    rdm2_f_oovv += einsum(x67, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x67, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x67, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x67, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x67
    x68 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x68 += einsum(t2, (0, 1, 2, 3), x31, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    x69 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x69 += einsum(t2, (0, 1, 2, 3), x68, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    del x68
    rdm2_f_oovv += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3))
    del x69
    x70 = np.zeros((nocc, nvir), dtype=types[float])
    x70 += einsum(t1, (0, 1), (0, 1)) * -1.0
    x70 += einsum(x53, (0, 1), (0, 1))
    del x53
    rdm2_f_oovv += einsum(t1, (0, 1), x70, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_oovv += einsum(t1, (0, 1), x70, (2, 3), (0, 2, 1, 3)) * -1.0
    del x70
    x71 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x71 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x71 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x72 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x72 += einsum(l2, (0, 1, 2, 3), x71, (2, 4, 1, 5), (4, 3, 5, 0))
    rdm2_f_ovov += einsum(x72, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_ovov += einsum(x72, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    del x72
    x73 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum(l2, (0, 1, 2, 3), x5, (3, 4, 1, 5), (2, 4, 0, 5)) * 2.0
    del x5
    rdm2_f_ovov += einsum(x73, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x73, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x73, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_voov += einsum(x73, (0, 1, 2, 3), (2, 1, 0, 3))
    del x73
    x74 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x74 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x74 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3))
    x75 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum(t1, (0, 1), x74, (2, 0, 3, 4), (2, 3, 4, 1))
    del x74
    rdm2_f_ovov += einsum(x75, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x75, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x75, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x75, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x75
    x76 = np.zeros((nvir, nvir), dtype=types[float])
    x76 += einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    x77 = np.zeros((nvir, nvir), dtype=types[float])
    x77 += einsum(t2, (0, 1, 2, 3), x31, (0, 1, 4, 3), (4, 2))
    del x31
    x78 = np.zeros((nvir, nvir), dtype=types[float])
    x78 += einsum(x76, (0, 1), (0, 1)) * 0.5
    x78 += einsum(x77, (0, 1), (0, 1))
    del x77
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x78, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x78, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x78, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x78, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv += einsum(t1, (0, 1), x78, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_ovvv += einsum(t1, (0, 1), x78, (2, 3), (0, 2, 1, 3)) * 2.0
    del x78
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum(t1, (0, 1), x1, (0, 2, 3, 4), (2, 3, 4, 1))
    rdm2_f_ovov += einsum(x79, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x79, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x79, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x79, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x79
    x80 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x80 += einsum(t2, (0, 1, 2, 3), x48, (1, 4, 5, 2), (4, 0, 5, 3))
    del x48
    rdm2_f_ovvo += einsum(x80, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x80, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x80, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vovo += einsum(x80, (0, 1, 2, 3), (2, 1, 3, 0))
    del x80
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * 2.0
    x81 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum(t2, (0, 1, 2, 3), x81, (1, 4, 5, 3), (4, 0, 5, 2))
    del x81
    rdm2_f_ovvo += einsum(x82, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_ovvo += einsum(x82, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_vovo += einsum(x82, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x82, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x82
    x83 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x83 += einsum(t1, (0, 1), x16, (2, 0, 3, 4), (2, 3, 4, 1))
    del x16
    rdm2_f_ovvo += einsum(x83, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x83, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x83, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x83, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x83
    x84 = np.zeros((nvir, nvir), dtype=types[float])
    x84 += einsum(x76, (0, 1), (0, 1))
    del x76
    x84 += einsum(x34, (0, 1), (0, 1))
    del x34
    rdm2_f_ovvo += einsum(delta.oo, (0, 1), x84, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_ovvo += einsum(delta.oo, (0, 1), x84, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_voov += einsum(delta.oo, (0, 1), x84, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_voov += einsum(delta.oo, (0, 1), x84, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x84, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x84, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x84, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x84, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv += einsum(t1, (0, 1), x84, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovv += einsum(t1, (0, 1), x84, (2, 3), (2, 0, 3, 1))
    x85 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x85 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 5, 1), (3, 4, 0, 5))
    rdm2_f_ovvo += einsum(x85, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x85, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x85, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x85, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x86 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x86 += einsum(t1, (0, 1), x1, (2, 0, 3, 4), (2, 3, 4, 1))
    rdm2_f_ovvo += einsum(x86, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x86, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x86, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x86, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x87 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x87 += einsum(l2, (0, 1, 2, 3), x9, (3, 4, 5, 1), (4, 2, 5, 0))
    del x9
    rdm2_f_ovvo += einsum(x87, (0, 1, 2, 3), (0, 3, 2, 1))
    rdm2_f_ovvo += einsum(x87, (0, 1, 2, 3), (0, 3, 2, 1))
    rdm2_f_voov += einsum(x87, (0, 1, 2, 3), (3, 0, 1, 2))
    rdm2_f_voov += einsum(x87, (0, 1, 2, 3), (3, 0, 1, 2))
    del x87
    x88 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x88 += einsum(l2, (0, 1, 2, 3), x71, (2, 4, 5, 1), (4, 3, 5, 0))
    del x71
    rdm2_f_voov += einsum(x88, (0, 1, 2, 3), (3, 0, 1, 2)) * -1.0
    rdm2_f_voov += einsum(x88, (0, 1, 2, 3), (3, 0, 1, 2)) * -1.0
    del x88
    x89 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x89 += einsum(l1, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_ovvv += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vovv += einsum(x89, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vovv += einsum(x89, (0, 1, 2, 3), (1, 0, 3, 2))
    x90 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x90 += einsum(t2, (0, 1, 2, 3), x1, (0, 1, 4, 5), (4, 5, 2, 3))
    del x1
    rdm2_f_ovvv += einsum(x90, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_ovvv += einsum(x90, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x90, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x90, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x91 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x91 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x92 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x92 += einsum(l2, (0, 1, 2, 3), x91, (3, 4, 1, 5), (4, 2, 5, 0))
    x93 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x93 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x93 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x94 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x94 += einsum(l2, (0, 1, 2, 3), x93, (2, 4, 1, 5), (4, 3, 5, 0))
    del x93
    x95 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x95 += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3))
    x95 += einsum(x92, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x95 += einsum(x94, (0, 1, 2, 3), (1, 0, 3, 2))
    del x94
    x96 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x96 += einsum(t1, (0, 1), x95, (0, 2, 3, 4), (2, 3, 4, 1))
    del x95
    x97 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x97 += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3))
    del x89
    x97 += einsum(x90, (0, 1, 2, 3), (0, 1, 2, 3))
    del x90
    x97 += einsum(x96, (0, 1, 2, 3), (0, 1, 3, 2))
    del x96
    x97 += einsum(t1, (0, 1), x84, (2, 3), (0, 2, 1, 3))
    del x84
    rdm2_f_ovvv += einsum(x97, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x97, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_ovvv += einsum(x97, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x97, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x97, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x97, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vovv += einsum(x97, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x97, (0, 1, 2, 3), (1, 0, 3, 2))
    del x97
    x98 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x98 += einsum(t1, (0, 1), x56, (0, 2, 3, 4), (2, 3, 1, 4))
    del x56
    rdm2_f_ovvv += einsum(x98, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_ovvv += einsum(x98, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x98, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x98, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x98
    x99 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x99 += einsum(x85, (0, 1, 2, 3), (0, 1, 2, 3))
    x99 += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3))
    x99 += einsum(x92, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x92
    x100 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x100 += einsum(t1, (0, 1), x99, (0, 2, 3, 4), (2, 3, 4, 1))
    del x99
    rdm2_f_ovvv += einsum(x100, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_ovvv += einsum(x100, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x100
    x101 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x101 += einsum(l2, (0, 1, 2, 3), x91, (3, 4, 1, 5), (4, 2, 5, 0)) * 0.5
    del x91
    x102 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x102 += einsum(x85, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x85
    x102 += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x86
    x102 += einsum(x101, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x101
    x103 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x103 += einsum(t1, (0, 1), x102, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    del x102
    rdm2_f_vovv += einsum(x103, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x103, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x103
    x104 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x104 += einsum(t1, (0, 1), l2, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov += einsum(x104, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvov += einsum(x104, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vvov += einsum(x104, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvov += einsum(x104, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvov += einsum(x104, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvov += einsum(x104, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo += einsum(x104, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_vvvo += einsum(x104, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vvvo += einsum(x104, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vvvo += einsum(x104, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vvvo += einsum(x104, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_vvvo += einsum(x104, (0, 1, 2, 3), (2, 1, 3, 0))
    x105 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x105 += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 5), (0, 1, 4, 5))
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv += einsum(x105, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vvvv += einsum(x105, (0, 1, 2, 3), (1, 0, 3, 2))
    x106 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x106 += einsum(t1, (0, 1), x104, (0, 2, 3, 4), (3, 2, 4, 1))
    del x104
    rdm2_f_vvvv += einsum(x106, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vvvv += einsum(x106, (0, 1, 2, 3), (1, 0, 3, 2))
    x107 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x107 += einsum(x105, (0, 1, 2, 3), (1, 0, 3, 2))
    del x105
    x107 += einsum(x106, (0, 1, 2, 3), (1, 0, 3, 2))
    del x106
    rdm2_f_vvvv += einsum(x107, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vvvv += einsum(x107, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvvv += einsum(x107, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vvvv += einsum(x107, (0, 1, 2, 3), (0, 1, 2, 3))
    del x107

    rdm2_f = pack_2e(rdm2_f_oooo, rdm2_f_ooov, rdm2_f_oovo, rdm2_f_ovoo, rdm2_f_vooo, rdm2_f_oovv, rdm2_f_ovov, rdm2_f_ovvo, rdm2_f_voov, rdm2_f_vovo, rdm2_f_vvoo, rdm2_f_ovvv, rdm2_f_vovv, rdm2_f_vvov, rdm2_f_vvvo, rdm2_f_vvvv)

    rdm2_f = rdm2_f.swapaxes(1, 2)

    return rdm2_f

