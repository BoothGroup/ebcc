# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), x0, (0, 1, 2, 3), ()) * 2.0
    del x0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5)) * 2.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (0, 4, 5, 3)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5)) * -1.0
    x0 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    t1new += einsum(t2, (0, 1, 2, 3), x0, (1, 3, 2, 4), (0, 4)) * 2.0
    del x0
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x2 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x2 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x2 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x2 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    del x1
    t1new += einsum(t2, (0, 1, 2, 3), x2, (4, 1, 0, 2), (4, 3)) * -1.0
    del x2
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x3 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x4 = np.zeros((nocc, nvir), dtype=types[float])
    x4 += einsum(f.ov, (0, 1), (0, 1))
    x4 += einsum(t1, (0, 1), x3, (0, 2, 1, 3), (2, 3)) * 2.0
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x5 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t1new += einsum(x4, (0, 1), x5, (0, 2, 3, 1), (2, 3))
    del x4, x5
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x6 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t1new += einsum(t1, (0, 1), x6, (0, 2, 1, 3), (2, 3)) * 2.0
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x7 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum(f.oo, (0, 1), (0, 1)) * 0.5
    x8 += einsum(t2, (0, 1, 2, 3), x7, (1, 4, 2, 3), (4, 0))
    t1new += einsum(t1, (0, 1), x8, (0, 2), (2, 1)) * -2.0
    del x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(x10, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x10
    x13 += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3))
    del x11
    x13 += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3))
    del x12
    t2new += einsum(x13, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new += einsum(x13, (0, 1, 2, 3), (1, 0, 2, 3))
    del x13
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(t2, (0, 1, 2, 3), x3, (1, 4, 2, 5), (0, 4, 3, 5))
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(t2, (0, 1, 2, 3), x15, (4, 1, 5, 3), (0, 4, 2, 5)) * 2.0
    del x15
    x17 = np.zeros((nvir, nvir), dtype=types[float])
    x17 += einsum(t2, (0, 1, 2, 3), x7, (0, 1, 4, 2), (3, 4))
    del x7
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(x17, (0, 1), t2, (2, 3, 1, 4), (2, 3, 4, 0)) * 2.0
    del x17
    x19 = np.zeros((nocc, nocc), dtype=types[float])
    x19 += einsum(t2, (0, 1, 2, 3), x3, (1, 4, 3, 2), (0, 4))
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(x19, (0, 1), t2, (2, 1, 3, 4), (2, 0, 4, 3)) * 2.0
    del x19
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(x14, (0, 1, 2, 3), (0, 1, 2, 3))
    del x14
    x21 += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3))
    del x16
    x21 += einsum(x18, (0, 1, 2, 3), (1, 0, 2, 3))
    del x18
    x21 += einsum(x20, (0, 1, 2, 3), (0, 1, 3, 2))
    del x20
    t2new += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x21, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x21
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x22 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x22 += einsum(t2, (0, 1, 2, 3), x3, (1, 4, 3, 5), (0, 4, 2, 5)) * 4.0
    del x3
    t2new += einsum(t2, (0, 1, 2, 3), x22, (4, 1, 5, 3), (4, 0, 5, 2))
    del x22
    x23 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x23 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x23 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (4, 0, 1, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x23, (0, 4, 5, 1), (5, 4, 3, 2))
    del x23
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x24 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    del x9
    t2new += einsum(t2, (0, 1, 2, 3), x24, (4, 1, 5, 2), (4, 0, 5, 3))
    del x24
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x25 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x25, (4, 1, 5, 2), (4, 0, 3, 5))
    del x25

    return {"t1new": t1new, "t2new": t2new}

