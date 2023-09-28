# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), x0, (0, 1, 3, 2), ()) * 2.0
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x1 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x1 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x2 = np.zeros((nocc, nvir), dtype=np.float64)
    x2 += einsum(f.ov, (0, 1), (0, 1))
    x2 += einsum(t1, (0, 1), x1, (0, 2, 1, 3), (2, 3))
    del x1
    e_cc += einsum(t1, (0, 1), x2, (0, 1), ()) * 2.0
    del x2

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=np.float64)
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2new += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    t2new += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (3, 2, 4, 1)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new += einsum(t2, (0, 1, 2, 3), v.oooo, (4, 1, 5, 0), (4, 5, 3, 2))
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    x1 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x1 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x1 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    t1new += einsum(x0, (0, 1, 2, 3), x1, (0, 3, 2, 4), (1, 4)) * 2.0
    del x1
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x2 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x3 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x3 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x3 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x3 += einsum(x2, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    t1new += einsum(t2, (0, 1, 2, 3), x3, (4, 1, 0, 2), (4, 3)) * -1.0
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x4 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x4 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x5 = np.zeros((nocc, nvir), dtype=np.float64)
    x5 += einsum(f.ov, (0, 1), (0, 1))
    x5 += einsum(t1, (0, 1), x4, (0, 2, 1, 3), (2, 3)) * 2.0
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t1new += einsum(x5, (0, 1), x6, (0, 2, 1, 3), (2, 3))
    del x5, x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x7 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x7 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t1new += einsum(t1, (0, 1), x7, (0, 2, 1, 3), (2, 3)) * 2.0
    del x7
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x8 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x8 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x9 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x9 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x10 = np.zeros((nocc, nocc), dtype=np.float64)
    x10 += einsum(f.oo, (0, 1), (0, 1))
    x10 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x10 += einsum(x0, (0, 1, 2, 3), x8, (0, 4, 3, 2), (4, 1)) * 2.0
    x10 += einsum(t1, (0, 1), x9, (0, 2, 3, 1), (3, 2)) * 2.0
    del x9
    t1new += einsum(t1, (0, 1), x10, (0, 2), (2, 1)) * -1.0
    del x10
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x11 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x12 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x13 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x13 += einsum(t1, (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new += einsum(t1, (0, 1), x13, (2, 3, 1, 4), (0, 2, 3, 4))
    del x13
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x14 += einsum(t2, (0, 1, 2, 3), x12, (4, 1, 5, 3), (0, 4, 2, 5))
    x15 = np.zeros((nvir, nvir), dtype=np.float64)
    x15 += einsum(t2, (0, 1, 2, 3), x8, (0, 1, 4, 2), (3, 4))
    del x8
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x16 += einsum(x15, (0, 1), t2, (2, 3, 1, 4), (2, 3, 0, 4))
    del x15
    x17 = np.zeros((nocc, nocc), dtype=np.float64)
    x17 += einsum(t2, (0, 1, 2, 3), x4, (1, 4, 3, 2), (0, 4))
    del x4
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x18 += einsum(x17, (0, 1), t2, (2, 1, 3, 4), (0, 2, 4, 3))
    del x17
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x19 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x20 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x20 += einsum(x19, (0, 1, 2, 3), (1, 0, 2, 3))
    x21 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x21 += einsum(t1, (0, 1), x20, (2, 3, 1, 4), (2, 3, 0, 4))
    del x20
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x22 += einsum(t1, (0, 1), x21, (0, 2, 3, 4), (3, 2, 4, 1))
    del x21
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x23 += einsum(x14, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x14
    x23 += einsum(x16, (0, 1, 2, 3), (1, 0, 3, 2))
    del x16
    x23 += einsum(x18, (0, 1, 2, 3), (1, 0, 3, 2))
    del x18
    x23 += einsum(x22, (0, 1, 2, 3), (0, 1, 3, 2))
    del x22
    t2new += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x23, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x23
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x24 += einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x25 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x26 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x26 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x27 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x27 += einsum(t1, (0, 1), x26, (2, 3, 4, 0), (2, 4, 3, 1))
    del x26
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x28 += einsum(t1, (0, 1), x27, (2, 0, 3, 4), (2, 3, 1, 4))
    del x27
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x29 += einsum(v.oovv, (0, 1, 2, 3), x0, (1, 4, 5, 3), (0, 4, 2, 5))
    del x0
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x30 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    del x24
    x30 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x25
    x30 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x19
    x30 += einsum(x28, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x28
    x30 += einsum(x29, (0, 1, 2, 3), (1, 0, 3, 2))
    del x29
    t2new += einsum(x30, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x30, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x30
    x31 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x31 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x31 += einsum(t1, (0, 1), x2, (2, 3, 4, 1), (3, 0, 2, 4))
    del x2
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x32 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x32 += einsum(t1, (0, 1), x31, (0, 2, 3, 4), (3, 2, 4, 1))
    del x31
    t2new += einsum(t1, (0, 1), x32, (2, 3, 0, 4), (2, 3, 1, 4))
    del x32
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x33 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x33 += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3))
    del x12
    t2new += einsum(t2, (0, 1, 2, 3), x33, (4, 1, 5, 2), (4, 0, 5, 3))
    del x33
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x34 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x34 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x34 += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x11
    t2new += einsum(t2, (0, 1, 2, 3), x34, (4, 1, 5, 3), (4, 0, 5, 2))
    del x34

    return {"t1new": t1new, "t2new": t2new}

