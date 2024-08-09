# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(f.ov, (0, 1), t1, (0, 1), ())
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    e_cc += einsum(v.oovv, (0, 1, 2, 3), x0, (0, 1, 2, 3), ()) * 0.25
    del x0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t1new += einsum(t1, (0, 1), v.ovov, (2, 1, 0, 3), (2, 3)) * -1.0
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(t1, (0, 1), v.ooov, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x0 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x1 += einsum(x0, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new += einsum(t2, (0, 1, 2, 3), x1, (4, 0, 1, 3), (4, 2)) * 0.5
    del x1
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x2 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    t1new += einsum(v.ovvv, (0, 1, 2, 3), x2, (0, 4, 2, 3), (4, 1)) * 0.5
    x3 = np.zeros((nocc, nvir), dtype=types[float])
    x3 += einsum(f.ov, (0, 1), (0, 1))
    x3 += einsum(t1, (0, 1), v.oovv, (2, 0, 3, 1), (2, 3))
    t1new += einsum(x3, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    del x3
    x4 = np.zeros((nocc, nocc), dtype=types[float])
    x4 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x5 = np.zeros((nocc, nocc), dtype=types[float])
    x5 += einsum(f.oo, (0, 1), (0, 1))
    x5 += einsum(x4, (0, 1), (0, 1))
    x5 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 1), (2, 3))
    x5 += einsum(v.oovv, (0, 1, 2, 3), x2, (1, 4, 2, 3), (0, 4)) * -0.5
    del x2
    t1new += einsum(t1, (0, 1), x5, (0, 2), (2, 1)) * -1.0
    del x5
    x6 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x6 += einsum(t1, (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new += einsum(t1, (0, 1), x6, (2, 3, 4, 1), (0, 2, 4, 3)) * -1.0
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new += einsum(x7, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new += einsum(x7, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x7
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x9 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x9 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum(t1, (0, 1), x9, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x9
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(t1, (0, 1), x10, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x10
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum(f.oo, (0, 1), (0, 1))
    x12 += einsum(x4, (0, 1), (0, 1))
    del x4
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(x12, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x12
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(x8, (0, 1, 2, 3), (0, 1, 3, 2))
    del x8
    x14 += einsum(x11, (0, 1, 2, 3), (0, 1, 3, 2))
    del x11
    x14 += einsum(x13, (0, 1, 2, 3), (1, 0, 3, 2))
    del x13
    t2new += einsum(x14, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x14, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x14
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum(t1, (0, 1), x16, (2, 3, 4, 1), (2, 0, 3, 4)) * -1.0
    del x16
    x18 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x18 += einsum(x15, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x15
    x18 += einsum(x17, (0, 1, 2, 3), (2, 1, 0, 3))
    del x17
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(t1, (0, 1), x18, (0, 2, 3, 4), (2, 3, 1, 4))
    del x18
    t2new += einsum(x19, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new += einsum(x19, (0, 1, 2, 3), (1, 0, 3, 2))
    del x19
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x20 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(t1, (0, 1), x20, (2, 0, 3, 4), (2, 3, 1, 4))
    del x20
    t2new += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x21, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x21, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new += einsum(x21, (0, 1, 2, 3), (1, 0, 3, 2))
    del x21
    x22 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x22 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x22 += einsum(t1, (0, 1), x0, (2, 3, 4, 1), (4, 3, 0, 2)) * -1.0
    del x0
    x23 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x23 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x23 += einsum(t1, (0, 1), x22, (0, 2, 3, 4), (3, 4, 2, 1))
    del x22
    t2new += einsum(t1, (0, 1), x23, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x23

    return {"t1new": t1new, "t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    # L amplitudes
    l1new = np.zeros((nvir, nocc), dtype=types[float])
    l1new += einsum(f.ov, (0, 1), (1, 0))
    l1new += einsum(l1, (0, 1), v.ovov, (2, 0, 1, 3), (3, 2)) * -1.0
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum(v.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new += einsum(l2, (0, 1, 2, 3), v.vvvv, (4, 5, 0, 1), (4, 5, 2, 3)) * 0.5
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x0 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    l1new += einsum(v.ooov, (0, 1, 2, 3), x0, (4, 2, 0, 1), (3, 4)) * -0.25
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    l1new += einsum(v.ovov, (0, 1, 2, 3), x1, (4, 2, 0, 1), (3, 4)) * -1.0
    l2new += einsum(v.ovvv, (0, 1, 2, 3), x1, (4, 5, 0, 1), (2, 3, 4, 5))
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum(t1, (0, 1), x1, (2, 3, 4, 1), (3, 2, 4, 0))
    l1new += einsum(v.ooov, (0, 1, 2, 3), x2, (4, 2, 0, 1), (3, 4)) * -0.5
    l2new += einsum(v.oovv, (0, 1, 2, 3), x2, (4, 5, 1, 0), (2, 3, 5, 4)) * 0.5
    x3 = np.zeros((nocc, nvir), dtype=types[float])
    x3 += einsum(t1, (0, 1), v.oovv, (2, 0, 3, 1), (2, 3))
    x4 = np.zeros((nocc, nocc), dtype=types[float])
    x4 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    l1new += einsum(x3, (0, 1), x4, (2, 0), (1, 2)) * -1.0
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 5, 1), (2, 4, 0, 5))
    l1new += einsum(v.ovvv, (0, 1, 2, 3), x5, (4, 0, 1, 3), (2, 4)) * -1.0
    del x5
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    l1new += einsum(x1, (0, 1, 2, 3), x6, (1, 2, 3, 4), (4, 0))
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x7 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x8 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x8 += einsum(x7, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x9 = np.zeros((nocc, nvir), dtype=types[float])
    x9 += einsum(f.ov, (0, 1), (0, 1))
    x9 += einsum(x3, (0, 1), (0, 1))
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x10 += einsum(x6, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x11 += einsum(x7, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.5
    x12 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x12 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x12 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (4, 5, 0, 1)) * -1.0
    x12 += einsum(t1, (0, 1), x11, (2, 3, 4, 1), (3, 2, 0, 4)) * -4.0
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x13 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x13 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 5, 2, 3), (4, 0, 1, 5)) * -0.5
    x13 += einsum(t2, (0, 1, 2, 3), x8, (4, 1, 5, 3), (5, 0, 4, 2)) * 2.0
    x13 += einsum(x9, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    x13 += einsum(t1, (0, 1), x10, (2, 3, 1, 4), (3, 2, 0, 4)) * -2.0
    del x10
    x13 += einsum(t1, (0, 1), x12, (0, 2, 3, 4), (2, 4, 3, 1)) * -0.5
    del x12
    l1new += einsum(l2, (0, 1, 2, 3), x13, (4, 2, 3, 1), (0, 4)) * 0.5
    del x13
    x14 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x14 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x14 += einsum(t1, (0, 1), v.vvvv, (2, 1, 3, 4), (0, 2, 3, 4)) * -1.0
    l1new += einsum(l2, (0, 1, 2, 3), x14, (3, 4, 0, 1), (4, 2)) * -0.5
    del x14
    x15 = np.zeros((nocc, nocc), dtype=types[float])
    x15 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    x16 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x16 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x0
    x16 += einsum(x2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x2
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), (1, 2, 3, 4)) * 2.0
    x17 += einsum(t1, (0, 1), x15, (2, 3), (2, 0, 3, 1)) * 2.0
    x17 += einsum(t2, (0, 1, 2, 3), x1, (4, 1, 5, 3), (4, 0, 5, 2)) * -4.0
    x17 += einsum(t1, (0, 1), x16, (0, 2, 3, 4), (2, 4, 3, 1)) * -1.0
    del x16
    l1new += einsum(v.oovv, (0, 1, 2, 3), x17, (4, 0, 1, 3), (2, 4)) * 0.25
    del x17
    x18 = np.zeros((nvir, nvir), dtype=types[float])
    x18 += einsum(l1, (0, 1), t1, (1, 2), (0, 2)) * 2.0
    x18 += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4))
    l1new += einsum(x18, (0, 1), v.ovvv, (2, 0, 3, 1), (3, 2)) * 0.5
    del x18
    x19 = np.zeros((nocc, nocc), dtype=types[float])
    x19 += einsum(x4, (0, 1), (0, 1))
    x19 += einsum(x15, (0, 1), (0, 1)) * 0.5
    l1new += einsum(x19, (0, 1), v.ooov, (2, 1, 0, 3), (3, 2))
    x20 = np.zeros((nocc, nvir), dtype=types[float])
    x20 += einsum(t1, (0, 1), (0, 1)) * -1.0
    x20 += einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3)) * -1.0
    x20 += einsum(t2, (0, 1, 2, 3), x1, (1, 0, 4, 3), (4, 2)) * 0.5
    x20 += einsum(t1, (0, 1), x19, (0, 2), (2, 1))
    del x19
    l1new += einsum(x20, (0, 1), v.oovv, (2, 0, 3, 1), (3, 2)) * -1.0
    del x20
    x21 = np.zeros((nvir, nvir), dtype=types[float])
    x21 += einsum(t1, (0, 1), v.ovvv, (0, 2, 3, 1), (2, 3))
    x22 = np.zeros((nvir, nvir), dtype=types[float])
    x22 += einsum(f.vv, (0, 1), (0, 1))
    x22 += einsum(x21, (0, 1), (0, 1)) * -1.0
    l1new += einsum(l1, (0, 1), x22, (0, 2), (2, 1))
    del x22
    x23 = np.zeros((nocc, nocc), dtype=types[float])
    x23 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 1), (2, 3))
    x24 = np.zeros((nocc, nocc), dtype=types[float])
    x24 += einsum(f.oo, (0, 1), (0, 1))
    x24 += einsum(x23, (0, 1), (1, 0))
    x24 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4)) * 0.5
    x24 += einsum(t1, (0, 1), x9, (2, 1), (0, 2))
    del x9
    l1new += einsum(l1, (0, 1), x24, (1, 2), (0, 2)) * -1.0
    del x24
    x25 = np.zeros((nocc, nocc), dtype=types[float])
    x25 += einsum(x4, (0, 1), (0, 1)) * 2.0
    x25 += einsum(x15, (0, 1), (0, 1))
    del x15
    l1new += einsum(f.ov, (0, 1), x25, (2, 0), (1, 2)) * -0.5
    del x25
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(f.vv, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(f.ov, (0, 1), x1, (2, 3, 0, 4), (2, 3, 1, 4))
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum(x21, (0, 1), l2, (2, 0, 3, 4), (3, 4, 2, 1))
    del x21
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum(x3, (0, 1), x1, (2, 3, 0, 4), (2, 3, 4, 1))
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum(l1, (0, 1), x8, (1, 2, 3, 4), (2, 3, 4, 0))
    del x8
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum(x26, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x26
    x31 += einsum(x27, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x27
    x31 += einsum(x28, (0, 1, 2, 3), (0, 1, 2, 3))
    del x28
    x31 += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3))
    del x29
    x31 += einsum(x30, (0, 1, 2, 3), (1, 0, 3, 2))
    del x30
    l2new += einsum(x31, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    l2new += einsum(x31, (0, 1, 2, 3), (3, 2, 0, 1))
    del x31
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum(f.oo, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3))
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum(l1, (0, 1), v.ovvv, (2, 0, 3, 4), (1, 2, 3, 4))
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x34 += einsum(x32, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x32
    x34 += einsum(x33, (0, 1, 2, 3), (0, 1, 3, 2))
    del x33
    l2new += einsum(x34, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new += einsum(x34, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    del x34
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum(v.ooov, (0, 1, 2, 3), x1, (4, 2, 1, 5), (4, 0, 5, 3))
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 += einsum(x1, (0, 1, 2, 3), x7, (0, 4, 2, 5), (1, 4, 3, 5)) * -1.0
    del x1, x7
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x37 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    del x6
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum(l2, (0, 1, 2, 3), x37, (3, 4, 1, 5), (4, 2, 5, 0))
    del x37
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum(f.ov, (0, 1), l1, (2, 3), (0, 3, 1, 2))
    x39 += einsum(l1, (0, 1), x3, (2, 3), (1, 2, 0, 3))
    x39 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    del x35
    x39 += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x36
    x39 += einsum(x38, (0, 1, 2, 3), (1, 0, 3, 2))
    del x38
    l2new += einsum(x39, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new += einsum(x39, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new += einsum(x39, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new += einsum(x39, (0, 1, 2, 3), (3, 2, 1, 0))
    del x39
    x40 = np.zeros((nocc, nocc), dtype=types[float])
    x40 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum(x40, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3))
    del x40
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum(x4, (0, 1), v.oovv, (2, 1, 3, 4), (0, 2, 3, 4))
    del x4
    x43 = np.zeros((nocc, nocc), dtype=types[float])
    x43 += einsum(t1, (0, 1), x3, (2, 1), (0, 2))
    del x3
    x44 = np.zeros((nocc, nocc), dtype=types[float])
    x44 += einsum(x23, (0, 1), (0, 1))
    del x23
    x44 += einsum(x43, (0, 1), (1, 0))
    del x43
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum(x44, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3)) * -1.0
    del x44
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum(x41, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x41
    x46 += einsum(x42, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x42
    x46 += einsum(x45, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x45
    l2new += einsum(x46, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new += einsum(x46, (0, 1, 2, 3), (3, 2, 1, 0))
    del x46
    x47 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x47 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x47 += einsum(t1, (0, 1), x11, (2, 3, 4, 1), (4, 0, 3, 2)) * -1.0
    del x11
    l2new += einsum(l2, (0, 1, 2, 3), x47, (2, 3, 4, 5), (0, 1, 4, 5))
    del x47

    return {"l1new": l1new, "l2new": l2new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))

    # RDM1
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=types[float])
    rdm1_f_oo += einsum(delta.oo, (0, 1), (0, 1))
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=types[float])
    rdm1_f_ov += einsum(t1, (0, 1), (0, 1))
    rdm1_f_ov += einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3))
    rdm1_f_vo = np.zeros((nvir, nocc), dtype=types[float])
    rdm1_f_vo += einsum(l1, (0, 1), (0, 1))
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=types[float])
    rdm1_f_vv += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4)) * 0.5
    rdm1_f_vv += einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    x0 = np.zeros((nocc, nocc), dtype=types[float])
    x0 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    rdm1_f_oo += einsum(x0, (0, 1), (1, 0)) * -1.0
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    rdm1_f_oo += einsum(x1, (0, 1), (1, 0)) * -0.5
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x2 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm1_f_ov += einsum(t2, (0, 1, 2, 3), x2, (0, 1, 4, 3), (4, 2)) * 0.5
    del x2
    x3 = np.zeros((nocc, nocc), dtype=types[float])
    x3 += einsum(x0, (0, 1), (0, 1)) * 2.0
    del x0
    x3 += einsum(x1, (0, 1), (0, 1))
    del x1
    rdm1_f_ov += einsum(t1, (0, 1), x3, (0, 2), (2, 1)) * -0.5
    del x3

    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])

    return rdm1_f

def make_rdm2_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))

    # RDM2
    rdm2_f_oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_ovoo += einsum(delta.oo, (0, 1), l1, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovoo += einsum(delta.oo, (0, 1), l1, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_vooo += einsum(delta.oo, (0, 1), l1, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_vooo += einsum(delta.oo, (0, 1), l1, (2, 3), (2, 0, 3, 1))
    rdm2_f_oovv = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovv += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovov = np.zeros((nocc, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_ovov += einsum(l1, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_ovvo += einsum(l1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_voov += einsum(l1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_vovo += einsum(l1, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 5), (0, 1, 4, 5)) * 0.5
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x0 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_oooo += einsum(x0, (0, 1, 2, 3), (3, 2, 1, 0)) * 0.5
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm2_f_ovoo += einsum(x1, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_vooo += einsum(x1, (0, 1, 2, 3), (3, 2, 1, 0))
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum(t1, (0, 1), x1, (2, 3, 4, 1), (2, 3, 0, 4))
    rdm2_f_oooo += einsum(x2, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    x3 = np.zeros((nocc, nocc), dtype=types[float])
    x3 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x3, (2, 3), (0, 3, 1, 2)) * -0.5
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x3, (2, 3), (0, 3, 2, 1)) * 0.5
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x3, (2, 3), (3, 0, 1, 2)) * 0.5
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x3, (2, 3), (3, 0, 2, 1)) * -0.5
    x4 = np.zeros((nocc, nocc), dtype=types[float])
    x4 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (3, 0, 1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (0, 3, 2, 1))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (3, 0, 2, 1)) * -1.0
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x5 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov += einsum(x5, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo += einsum(x5, (0, 1, 2, 3), (2, 1, 3, 0))
    x6 = np.zeros((nocc, nvir), dtype=types[float])
    x6 += einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3))
    x7 = np.zeros((nocc, nvir), dtype=types[float])
    x7 += einsum(t2, (0, 1, 2, 3), x1, (0, 1, 4, 3), (4, 2)) * -1.0
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum(x4, (0, 1), (0, 1))
    x8 += einsum(x3, (0, 1), (0, 1)) * 0.5
    x9 = np.zeros((nocc, nvir), dtype=types[float])
    x9 += einsum(t1, (0, 1), x8, (0, 2), (2, 1))
    x10 = np.zeros((nocc, nvir), dtype=types[float])
    x10 += einsum(x6, (0, 1), (0, 1)) * -1.0
    del x6
    x10 += einsum(x7, (0, 1), (0, 1)) * 0.5
    del x7
    x10 += einsum(x9, (0, 1), (0, 1))
    del x9
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x10, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x10, (2, 3), (2, 0, 1, 3))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x10, (2, 3), (0, 2, 3, 1))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x10, (2, 3), (2, 0, 3, 1)) * -1.0
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum(t2, (0, 1, 2, 3), x1, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x12 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x12 += einsum(delta.oo, (0, 1), t1, (2, 3), (0, 1, 2, 3))
    x12 += einsum(x11, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x12 += einsum(t1, (0, 1), x8, (2, 3), (0, 2, 3, 1))
    del x8
    rdm2_f_ooov += einsum(x12, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_ooov += einsum(x12, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_oovo += einsum(x12, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(x12, (0, 1, 2, 3), (2, 0, 3, 1))
    del x12
    x13 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x13 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x0
    x13 += einsum(x2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x2
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), x13, (0, 1, 4, 5), (5, 4, 2, 3)) * 0.25
    x14 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x14 += einsum(t1, (0, 1), x13, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x13
    rdm2_f_ooov += einsum(x14, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_oovo += einsum(x14, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_oovv += einsum(t1, (0, 1), x14, (0, 2, 3, 4), (3, 2, 4, 1))
    del x14
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(t1, (0, 1), x11, (0, 2, 3, 4), (2, 3, 1, 4))
    del x11
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    del x15
    x16 += einsum(t1, (0, 1), x10, (2, 3), (0, 2, 1, 3))
    del x10
    rdm2_f_oovv += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x16, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x16, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_oovv += einsum(x16, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x16
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum(x4, (0, 1), t2, (2, 0, 3, 4), (1, 2, 3, 4))
    del x4
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(x3, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4))
    del x3
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(x17, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x17
    x19 += einsum(x18, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    del x18
    rdm2_f_oovv += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x19, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x19
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(t1, (0, 1), x5, (0, 2, 3, 4), (2, 3, 1, 4))
    del x5
    x21 = np.zeros((nvir, nvir), dtype=types[float])
    x21 += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4))
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(x21, (0, 1), t2, (2, 3, 4, 0), (2, 3, 4, 1))
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_ovov += einsum(x23, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovvo += einsum(x23, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_voov += einsum(x23, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_vovo += einsum(x23, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum(t2, (0, 1, 2, 3), x23, (1, 4, 3, 5), (0, 4, 2, 5))
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3))
    del x20
    x25 += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x22
    x25 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    del x24
    rdm2_f_oovv += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x25, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x25
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(t1, (0, 1), x1, (2, 0, 3, 4), (2, 3, 4, 1))
    rdm2_f_ovov += einsum(x26, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_ovvo += einsum(x26, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x26, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x26, (0, 1, 2, 3), (2, 1, 3, 0))
    x27 = np.zeros((nvir, nvir), dtype=types[float])
    x27 += einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    x28 = np.zeros((nvir, nvir), dtype=types[float])
    x28 += einsum(x27, (0, 1), (0, 1)) * 2.0
    x28 += einsum(x21, (0, 1), (0, 1))
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x28, (2, 3), (0, 2, 1, 3)) * 0.5
    rdm2_f_ovvo += einsum(delta.oo, (0, 1), x28, (2, 3), (0, 2, 3, 1)) * -0.5
    rdm2_f_voov += einsum(delta.oo, (0, 1), x28, (2, 3), (2, 0, 1, 3)) * -0.5
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x28, (2, 3), (2, 0, 3, 1)) * 0.5
    del x28
    x29 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x29 += einsum(l1, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv += einsum(x29, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv += einsum(x29, (0, 1, 2, 3), (1, 0, 3, 2))
    del x29
    x30 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x30 += einsum(t2, (0, 1, 2, 3), x1, (0, 1, 4, 5), (4, 5, 2, 3))
    del x1
    rdm2_f_ovvv += einsum(x30, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    rdm2_f_vovv += einsum(x30, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    del x30
    x31 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x31 += einsum(t1, (0, 1), x26, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    del x26
    rdm2_f_ovvv += einsum(x31, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x31, (0, 1, 2, 3), (1, 0, 3, 2))
    del x31
    x32 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x32 += einsum(t1, (0, 1), x23, (0, 2, 3, 4), (2, 3, 1, 4))
    del x23
    x33 = np.zeros((nvir, nvir), dtype=types[float])
    x33 += einsum(x27, (0, 1), (0, 1))
    del x27
    x33 += einsum(x21, (0, 1), (0, 1)) * 0.5
    del x21
    x34 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x34 += einsum(x32, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x32
    x34 += einsum(t1, (0, 1), x33, (2, 3), (0, 2, 1, 3))
    del x33
    rdm2_f_ovvv += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x34, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x34, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x34, (0, 1, 2, 3), (1, 0, 3, 2))
    del x34
    x35 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x35 += einsum(t1, (0, 1), l2, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov += einsum(x35, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo += einsum(x35, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vvvv += einsum(t1, (0, 1), x35, (0, 2, 3, 4), (2, 3, 1, 4))
    del x35

    rdm2_f = pack_2e(rdm2_f_oooo, rdm2_f_ooov, rdm2_f_oovo, rdm2_f_ovoo, rdm2_f_vooo, rdm2_f_oovv, rdm2_f_ovov, rdm2_f_ovvo, rdm2_f_voov, rdm2_f_vovo, rdm2_f_vvoo, rdm2_f_ovvv, rdm2_f_vovv, rdm2_f_vvov, rdm2_f_vvvo, rdm2_f_vvvv)

    rdm2_f = rdm2_f.swapaxes(1, 2)

    return rdm2_f

