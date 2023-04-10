# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(f.ov, (0, 1), t1, (0, 1), ())
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2))
    x0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    e_cc += einsum(v.oovv, (0, 1, 2, 3), x0, (0, 1, 2, 3), ()) * 0.25

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=np.float64)
    t1new += einsum(t1, (0, 1), f.vv, (2, 1), (0, 2))
    t1new += einsum(t1, (0, 1), v.ovov, (2, 1, 0, 3), (2, 3)) * -1.0
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2new += einsum(v.ooov, (0, 1, 2, 3), t1, (2, 4), (1, 0, 3, 4)) * -1.0
    t2new += einsum(v.oovv, (0, 1, 2, 3), (1, 0, 3, 2))
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x0 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x1 += einsum(v.ooov, (0, 1, 2, 3), (2, 1, 0, 3))
    x1 += einsum(x0, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new += einsum(x1, (0, 1, 2, 3), t2, (1, 2, 3, 4), (0, 4)) * -0.5
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x2 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2))
    x2 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    t1new += einsum(x2, (0, 1, 2, 3), v.ovvv, (0, 4, 2, 3), (1, 4)) * 0.5
    x3 = np.zeros((nocc, nvir), dtype=np.float64)
    x3 += einsum(f.ov, (0, 1), (0, 1))
    x3 += einsum(v.oovv, (0, 1, 2, 3), t1, (1, 3), (0, 2))
    t1new += einsum(x3, (0, 1), t2, (0, 2, 1, 3), (2, 3))
    x4 = np.zeros((nocc, nocc), dtype=np.float64)
    x4 += einsum(t1, (0, 1), f.ov, (2, 1), (2, 0))
    x5 = np.zeros((nocc, nocc), dtype=np.float64)
    x5 += einsum(f.oo, (0, 1), (1, 0))
    x5 += einsum(x4, (0, 1), (0, 1))
    x5 += einsum(t1, (0, 1), v.ooov, (0, 2, 3, 1), (2, 3)) * -1.0
    x5 += einsum(v.oovv, (0, 1, 2, 3), x2, (0, 4, 2, 3), (1, 4)) * 0.5
    t1new += einsum(x5, (0, 1), t1, (0, 2), (1, 2)) * -1.0
    x6 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x6 += einsum(v.vvvv, (0, 1, 2, 3), t1, (4, 2), (4, 0, 1, 3)) * -1.0
    t2new += einsum(x6, (0, 1, 2, 3), t1, (4, 3), (4, 0, 2, 1)) * -1.0
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x7 += einsum(v.ovvv, (0, 1, 2, 3), t1, (4, 1), (4, 0, 2, 3))
    x8 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x8 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x9 += einsum(x8, (0, 1, 2, 3), t1, (1, 4), (0, 2, 3, 4)) * -1.0
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x10 += einsum(x9, (0, 1, 2, 3), t1, (1, 4), (0, 2, 3, 4)) * -1.0
    x11 = np.zeros((nocc, nocc), dtype=np.float64)
    x11 += einsum(f.oo, (0, 1), (1, 0))
    x11 += einsum(x4, (0, 1), (0, 1))
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x12 += einsum(x11, (0, 1), t2, (0, 2, 3, 4), (2, 1, 3, 4))
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x13 += einsum(x7, (0, 1, 2, 3), (0, 1, 3, 2))
    x13 += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2))
    x13 += einsum(x12, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x13, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x14 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x14 += einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x15 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x16 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x16 += einsum(t1, (0, 1), x15, (2, 3, 4, 1), (2, 0, 3, 4)) * -1.0
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x17 += einsum(x14, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x17 += einsum(x16, (0, 1, 2, 3), (2, 1, 0, 3))
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x18 += einsum(t1, (0, 1), x17, (0, 2, 3, 4), (2, 3, 1, 4))
    t2new += einsum(x18, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new += einsum(x18, (0, 1, 2, 3), (1, 0, 3, 2))
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x19 += einsum(t1, (0, 1), v.ovov, (2, 1, 3, 4), (0, 3, 2, 4))
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x20 += einsum(x19, (0, 1, 2, 3), t1, (1, 4), (0, 2, 4, 3))
    t2new += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x20, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new += einsum(x20, (0, 1, 2, 3), (1, 0, 3, 2))
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x21 += einsum(t2, (0, 1, 2, 3), f.vv, (4, 3), (0, 1, 4, 2))
    t2new += einsum(x21, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new += einsum(x21, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x22 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x22 += einsum(v.oooo, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    x22 += einsum(x0, (0, 1, 2, 3), t1, (4, 3), (2, 1, 4, 0)) * -1.0
    x23 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x23 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3))
    x23 += einsum(x22, (0, 1, 2, 3), t1, (0, 4), (2, 3, 1, 4)) * -1.0
    t2new += einsum(x23, (0, 1, 2, 3), t1, (2, 4), (0, 1, 4, 3))

    return {"t1new": t1new, "t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    # L amplitudes
    l1new = np.zeros((nvir, nocc), dtype=np.float64)
    l1new += einsum(l1, (0, 1), v.ovov, (2, 0, 1, 3), (3, 2)) * -1.0
    l1new += einsum(f.ov, (0, 1), (1, 0))
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=np.float64)
    l2new += einsum(v.vvvv, (0, 1, 2, 3), l2, (2, 3, 4, 5), (1, 0, 5, 4)) * 0.5
    l2new += einsum(v.oovv, (0, 1, 2, 3), (3, 2, 1, 0))
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 5, 1), (3, 4, 0, 5)) * -1.0
    l1new += einsum(x0, (0, 1, 2, 3), v.ovvv, (1, 2, 4, 3), (4, 0)) * -1.0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x1 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x2 += einsum(l2, (0, 1, 2, 3), t1, (4, 1), (2, 3, 4, 0))
    l1new += einsum(x1, (0, 1, 2, 3), x2, (4, 0, 1, 2), (3, 4))
    l1new += einsum(x2, (0, 1, 2, 3), v.ovov, (1, 4, 2, 3), (4, 0)) * -1.0
    l2new += einsum(v.ovvv, (0, 1, 2, 3), x2, (4, 5, 0, 1), (2, 3, 4, 5))
    x3 = np.zeros((nocc, nvir), dtype=np.float64)
    x3 += einsum(v.oovv, (0, 1, 2, 3), t1, (1, 3), (0, 2))
    x4 = np.zeros((nocc, nocc), dtype=np.float64)
    x4 += einsum(t1, (0, 1), l1, (1, 2), (2, 0))
    l1new += einsum(x4, (0, 1), x3, (1, 2), (2, 0)) * -1.0
    x5 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x5 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    l1new += einsum(x5, (0, 1, 2, 3), v.ooov, (2, 3, 1, 4), (4, 0)) * -0.25
    x6 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x6 += einsum(x2, (0, 1, 2, 3), t1, (4, 3), (1, 0, 2, 4))
    l1new += einsum(x6, (0, 1, 2, 3), v.ooov, (3, 2, 1, 4), (4, 0)) * 0.5
    l2new += einsum(v.oovv, (0, 1, 2, 3), x6, (4, 5, 0, 1), (2, 3, 5, 4)) * -0.5
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x7 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x8 += einsum(v.ooov, (0, 1, 2, 3), (2, 1, 0, 3))
    x8 += einsum(x7, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x9 = np.zeros((nocc, nvir), dtype=np.float64)
    x9 += einsum(f.ov, (0, 1), (0, 1))
    x9 += einsum(x3, (0, 1), (0, 1))
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x10 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1))
    x10 += einsum(x1, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x11 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x11 += einsum(x7, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.5
    x12 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x12 += einsum(v.oooo, (0, 1, 2, 3), (2, 3, 1, 0)) * 2.0
    x12 += einsum(v.oovv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (1, 0, 5, 4)) * -1.0
    x12 += einsum(x11, (0, 1, 2, 3), t1, (4, 3), (1, 0, 4, 2)) * -4.0
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x13 += einsum(v.ooov, (0, 1, 2, 3), (2, 1, 0, 3))
    x13 += einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 5, 2, 3), (0, 5, 4, 1)) * 0.5
    x13 += einsum(x8, (0, 1, 2, 3), t2, (1, 4, 3, 5), (2, 4, 0, 5)) * 2.0
    x13 += einsum(t2, (0, 1, 2, 3), x9, (4, 2), (4, 1, 0, 3))
    x13 += einsum(x10, (0, 1, 2, 3), t1, (4, 2), (1, 0, 4, 3)) * -2.0
    x13 += einsum(x12, (0, 1, 2, 3), t1, (0, 4), (1, 3, 2, 4)) * -0.5
    l1new += einsum(x13, (0, 1, 2, 3), l2, (3, 4, 1, 2), (4, 0)) * -0.5
    x14 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x14 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x14 += einsum(v.vvvv, (0, 1, 2, 3), t1, (4, 1), (4, 0, 3, 2))
    l1new += einsum(l2, (0, 1, 2, 3), x14, (2, 4, 0, 1), (4, 3)) * 0.5
    x15 = np.zeros((nocc, nocc), dtype=np.float64)
    x15 += einsum(l2, (0, 1, 2, 3), t2, (2, 4, 0, 1), (3, 4))
    x16 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x16 += einsum(x5, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x16 += einsum(x6, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x17 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), (1, 3, 2, 4)) * -2.0
    x17 += einsum(t1, (0, 1), x15, (2, 3), (2, 0, 3, 1)) * 2.0
    x17 += einsum(x2, (0, 1, 2, 3), t2, (4, 1, 5, 3), (0, 4, 2, 5)) * -4.0
    x17 += einsum(x16, (0, 1, 2, 3), t1, (0, 4), (1, 3, 2, 4)) * -1.0
    l1new += einsum(v.oovv, (0, 1, 2, 3), x17, (4, 0, 1, 2), (3, 4)) * -0.25
    x18 = np.zeros((nvir, nvir), dtype=np.float64)
    x18 += einsum(t1, (0, 1), l1, (2, 0), (2, 1))
    x18 += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 0, 4), (1, 4)) * 0.5
    l1new += einsum(x18, (0, 1), v.ovvv, (2, 0, 1, 3), (3, 2)) * -1.0
    x19 = np.zeros((nocc, nocc), dtype=np.float64)
    x19 += einsum(x4, (0, 1), (0, 1))
    x19 += einsum(x15, (0, 1), (0, 1)) * 0.5
    l1new += einsum(x19, (0, 1), f.ov, (1, 2), (2, 0)) * -1.0
    x20 = np.zeros((nocc, nvir), dtype=np.float64)
    x20 += einsum(t1, (0, 1), (0, 1)) * -1.0
    x20 += einsum(t2, (0, 1, 2, 3), l1, (3, 1), (0, 2)) * -1.0
    x20 += einsum(t2, (0, 1, 2, 3), x2, (0, 1, 4, 2), (4, 3)) * 0.5
    x20 += einsum(x19, (0, 1), t1, (0, 2), (1, 2))
    l1new += einsum(x20, (0, 1), v.oovv, (0, 2, 1, 3), (3, 2)) * -1.0
    x21 = np.zeros((nocc, nocc), dtype=np.float64)
    x21 += einsum(x4, (0, 1), (0, 1)) * 2.0
    x21 += einsum(x15, (0, 1), (0, 1))
    l1new += einsum(v.ooov, (0, 1, 2, 3), x21, (2, 0), (3, 1)) * -0.5
    x22 = np.zeros((nvir, nvir), dtype=np.float64)
    x22 += einsum(t1, (0, 1), v.ovvv, (0, 2, 3, 1), (2, 3))
    x23 = np.zeros((nvir, nvir), dtype=np.float64)
    x23 += einsum(f.vv, (0, 1), (1, 0))
    x23 += einsum(x22, (0, 1), (0, 1)) * -1.0
    l1new += einsum(l1, (0, 1), x23, (0, 2), (2, 1))
    x24 = np.zeros((nocc, nocc), dtype=np.float64)
    x24 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 1), (2, 3))
    x25 = np.zeros((nocc, nocc), dtype=np.float64)
    x25 += einsum(f.oo, (0, 1), (1, 0))
    x25 += einsum(x24, (0, 1), (1, 0))
    x25 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 0, 2, 3), (1, 4)) * -0.5
    x25 += einsum(t1, (0, 1), x9, (2, 1), (0, 2))
    l1new += einsum(l1, (0, 1), x25, (1, 2), (0, 2)) * -1.0
    x26 = np.zeros((nocc, nocc), dtype=np.float64)
    x26 += einsum(t1, (0, 1), f.ov, (2, 1), (2, 0))
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x27 += einsum(x26, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3))
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x28 += einsum(x4, (0, 1), v.oovv, (2, 1, 3, 4), (0, 2, 3, 4))
    x29 = np.zeros((nocc, nocc), dtype=np.float64)
    x29 += einsum(t1, (0, 1), x3, (2, 1), (0, 2))
    x30 = np.zeros((nocc, nocc), dtype=np.float64)
    x30 += einsum(x24, (0, 1), (0, 1))
    x30 += einsum(x29, (0, 1), (1, 0))
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x31 += einsum(x30, (0, 1), l2, (2, 3, 1, 4), (0, 4, 2, 3))
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x32 += einsum(x27, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x32 += einsum(x28, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x32 += einsum(x31, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    l2new += einsum(x32, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new += einsum(x32, (0, 1, 2, 3), (3, 2, 1, 0))
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x33 += einsum(x2, (0, 1, 2, 3), v.ooov, (4, 2, 1, 5), (0, 4, 3, 5))
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x34 += einsum(x2, (0, 1, 2, 3), x7, (0, 4, 2, 5), (1, 4, 3, 5)) * -1.0
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x35 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x35 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x36 += einsum(x35, (0, 1, 2, 3), l2, (2, 4, 0, 5), (1, 5, 3, 4))
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x37 += einsum(f.ov, (0, 1), l1, (2, 3), (0, 3, 1, 2))
    x37 += einsum(x3, (0, 1), l1, (2, 3), (3, 0, 2, 1))
    x37 += einsum(x33, (0, 1, 2, 3), (0, 1, 2, 3))
    x37 += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x37 += einsum(x36, (0, 1, 2, 3), (1, 0, 3, 2))
    l2new += einsum(x37, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new += einsum(x37, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new += einsum(x37, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new += einsum(x37, (0, 1, 2, 3), (3, 2, 1, 0))
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x38 += einsum(l2, (0, 1, 2, 3), f.vv, (4, 1), (2, 3, 4, 0))
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x39 += einsum(f.ov, (0, 1), x2, (2, 3, 0, 4), (2, 3, 1, 4))
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x40 += einsum(x22, (0, 1), l2, (2, 0, 3, 4), (3, 4, 2, 1))
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x41 += einsum(x3, (0, 1), x2, (2, 3, 0, 4), (2, 3, 4, 1))
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x42 += einsum(l1, (0, 1), x8, (1, 2, 3, 4), (2, 3, 4, 0))
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x43 += einsum(x38, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x43 += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x43 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3))
    x43 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3))
    x43 += einsum(x42, (0, 1, 2, 3), (1, 0, 3, 2))
    l2new += einsum(x43, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    l2new += einsum(x43, (0, 1, 2, 3), (3, 2, 0, 1))
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x44 += einsum(f.oo, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3))
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x45 += einsum(v.ovvv, (0, 1, 2, 3), l1, (1, 4), (4, 0, 2, 3))
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x46 += einsum(x44, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x46 += einsum(x45, (0, 1, 2, 3), (0, 1, 3, 2))
    l2new += einsum(x46, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new += einsum(x46, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    x47 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x47 += einsum(v.oooo, (0, 1, 2, 3), (2, 3, 1, 0)) * -0.5
    x47 += einsum(x11, (0, 1, 2, 3), t1, (4, 3), (2, 4, 1, 0)) * -1.0
    l2new += einsum(l2, (0, 1, 2, 3), x47, (2, 3, 4, 5), (1, 0, 4, 5)) * -1.0

    return {"l1new": l1new, "l2new": l2new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))

    # RDM1
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=np.float64)
    rdm1_f_oo += einsum(delta.oo, (0, 1), (1, 0))
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=np.float64)
    rdm1_f_ov += einsum(t2, (0, 1, 2, 3), l1, (3, 1), (0, 2))
    rdm1_f_ov += einsum(t1, (0, 1), (0, 1))
    rdm1_f_vo = np.zeros((nvir, nocc), dtype=np.float64)
    rdm1_f_vo += einsum(l1, (0, 1), (0, 1))
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=np.float64)
    rdm1_f_vv += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 0), (1, 4)) * -0.5
    rdm1_f_vv += einsum(t1, (0, 1), l1, (2, 0), (2, 1))
    x0 = np.zeros((nocc, nocc), dtype=np.float64)
    x0 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    rdm1_f_oo += einsum(x0, (0, 1), (1, 0)) * -0.5
    x1 = np.zeros((nocc, nocc), dtype=np.float64)
    x1 += einsum(t1, (0, 1), l1, (1, 2), (2, 0))
    rdm1_f_oo += einsum(x1, (0, 1), (1, 0)) * -1.0
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x2 += einsum(l2, (0, 1, 2, 3), t1, (4, 1), (2, 3, 4, 0))
    rdm1_f_ov += einsum(x2, (0, 1, 2, 3), t2, (0, 1, 4, 3), (2, 4)) * 0.5
    x3 = np.zeros((nocc, nocc), dtype=np.float64)
    x3 += einsum(x1, (0, 1), (0, 1))
    x3 += einsum(x0, (0, 1), (0, 1)) * 0.5
    rdm1_f_ov += einsum(t1, (0, 1), x3, (0, 2), (2, 1)) * -1.0

    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])

    return rdm1_f

def make_rdm2_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))

    # RDM2
    rdm2_f_oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (1, 3, 0, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=np.float64)
    rdm2_f_ovoo += einsum(l1, (0, 1), delta.oo, (2, 3), (3, 0, 2, 1))
    rdm2_f_ovoo += einsum(l1, (0, 1), delta.oo, (2, 3), (3, 0, 1, 2)) * -1.0
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=np.float64)
    rdm2_f_vooo += einsum(l1, (0, 1), delta.oo, (2, 3), (0, 3, 2, 1)) * -1.0
    rdm2_f_vooo += einsum(l1, (0, 1), delta.oo, (2, 3), (0, 3, 1, 2))
    rdm2_f_oovv = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_oovv += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovv += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovov = np.zeros((nocc, nvir, nocc, nvir), dtype=np.float64)
    rdm2_f_ovov += einsum(t1, (0, 1), l1, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=np.float64)
    rdm2_f_ovvo += einsum(t1, (0, 1), l1, (2, 3), (0, 2, 1, 3))
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=np.float64)
    rdm2_f_voov += einsum(t1, (0, 1), l1, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=np.float64)
    rdm2_f_vovo += einsum(t1, (0, 1), l1, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=np.float64)
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=np.float64)
    rdm2_f_vvvv += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 5), (1, 0, 5, 4)) * 0.5
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x0 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_oooo += einsum(x0, (0, 1, 2, 3), (3, 2, 1, 0)) * 0.5
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x1 += einsum(l2, (0, 1, 2, 3), t1, (4, 1), (2, 3, 4, 0))
    rdm2_f_ovoo += einsum(x1, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_vooo += einsum(x1, (0, 1, 2, 3), (3, 2, 1, 0))
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x2 += einsum(t1, (0, 1), x1, (2, 3, 4, 1), (2, 3, 0, 4))
    rdm2_f_oooo += einsum(x2, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    x3 = np.zeros((nocc, nocc), dtype=np.float64)
    x3 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    rdm2_f_oooo += einsum(x3, (0, 1), delta.oo, (2, 3), (2, 1, 3, 0)) * -0.5
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x3, (2, 3), (0, 3, 2, 1)) * 0.5
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x3, (2, 3), (3, 1, 0, 2)) * 0.5
    rdm2_f_oooo += einsum(x3, (0, 1), delta.oo, (2, 3), (1, 3, 0, 2)) * -0.5
    x4 = np.zeros((nocc, nocc), dtype=np.float64)
    x4 += einsum(t1, (0, 1), l1, (1, 2), (2, 0))
    rdm2_f_oooo += einsum(x4, (0, 1), delta.oo, (2, 3), (3, 1, 2, 0)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (3, 1, 0, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x4, (2, 3), (1, 3, 2, 0))
    rdm2_f_oooo += einsum(x4, (0, 1), delta.oo, (2, 3), (1, 3, 0, 2)) * -1.0
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x5 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    rdm2_f_ooov += einsum(x5, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=np.float64)
    rdm2_f_oovo += einsum(x5, (0, 1, 2, 3), (2, 1, 3, 0))
    x6 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x6 += einsum(t2, (0, 1, 2, 3), x1, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x7 = np.zeros((nocc, nocc), dtype=np.float64)
    x7 += einsum(x4, (0, 1), (0, 1))
    x7 += einsum(x3, (0, 1), (0, 1)) * 0.5
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x8 += einsum(t1, (0, 1), delta.oo, (2, 3), (3, 2, 0, 1))
    x8 += einsum(x6, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x8 += einsum(t1, (0, 1), x7, (2, 3), (0, 2, 3, 1))
    rdm2_f_ooov += einsum(x8, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_ooov += einsum(x8, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_oovo += einsum(x8, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(x8, (0, 1, 2, 3), (2, 0, 3, 1))
    x9 = np.zeros((nocc, nvir), dtype=np.float64)
    x9 += einsum(t2, (0, 1, 2, 3), l1, (3, 1), (0, 2))
    x10 = np.zeros((nocc, nvir), dtype=np.float64)
    x10 += einsum(x1, (0, 1, 2, 3), t2, (0, 1, 4, 3), (2, 4)) * -1.0
    x11 = np.zeros((nocc, nvir), dtype=np.float64)
    x11 += einsum(x7, (0, 1), t1, (0, 2), (1, 2))
    x12 = np.zeros((nocc, nvir), dtype=np.float64)
    x12 += einsum(x9, (0, 1), (0, 1)) * -1.0
    x12 += einsum(x10, (0, 1), (0, 1)) * 0.5
    x12 += einsum(x11, (0, 1), (0, 1))
    rdm2_f_ooov += einsum(x12, (0, 1), delta.oo, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x12, (2, 3), (2, 0, 1, 3))
    rdm2_f_oovo += einsum(x12, (0, 1), delta.oo, (2, 3), (3, 0, 1, 2))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x12, (2, 3), (2, 0, 3, 1)) * -1.0
    x13 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x13 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x13 += einsum(x2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    rdm2_f_oovv += einsum(x13, (0, 1, 2, 3), t2, (0, 1, 4, 5), (3, 2, 5, 4)) * -0.25
    x14 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x14 += einsum(x13, (0, 1, 2, 3), t1, (0, 4), (1, 2, 3, 4)) * 0.5
    rdm2_f_ooov += einsum(x14, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_oovo += einsum(x14, (0, 1, 2, 3), (2, 1, 3, 0))
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x15 += einsum(t1, (0, 1), x5, (0, 2, 3, 4), (2, 3, 1, 4))
    x16 = np.zeros((nvir, nvir), dtype=np.float64)
    x16 += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 0), (1, 4)) * -1.0
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x17 += einsum(t2, (0, 1, 2, 3), x16, (3, 4), (0, 1, 2, 4))
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x18 += einsum(t2, (0, 1, 2, 3), l2, (3, 4, 5, 1), (5, 0, 4, 2)) * -1.0
    rdm2_f_ovov += einsum(x18, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovvo += einsum(x18, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_voov += einsum(x18, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_vovo += einsum(x18, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x19 += einsum(t2, (0, 1, 2, 3), x18, (1, 4, 3, 5), (4, 0, 5, 2))
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x20 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    x20 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x20 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x20, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x21 += einsum(x4, (0, 1), t2, (2, 0, 3, 4), (1, 2, 3, 4))
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x22 += einsum(x3, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4))
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x23 += einsum(x21, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x23 += einsum(x22, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    rdm2_f_oovv += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x24 += einsum(t1, (0, 1), x6, (0, 2, 3, 4), (2, 3, 1, 4))
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x25 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    x25 += einsum(x12, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    rdm2_f_oovv += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x25, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x25, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_oovv += einsum(x25, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x26 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x26 += einsum(x13, (0, 1, 2, 3), t1, (0, 4), (1, 3, 2, 4)) * -1.0
    rdm2_f_oovv += einsum(x26, (0, 1, 2, 3), t1, (0, 4), (1, 2, 4, 3)) * 0.5
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x27 += einsum(x1, (0, 1, 2, 3), t1, (1, 4), (0, 2, 3, 4))
    rdm2_f_ovov += einsum(x27, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_ovvo += einsum(x27, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x27, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x27, (0, 1, 2, 3), (2, 1, 3, 0))
    x28 = np.zeros((nvir, nvir), dtype=np.float64)
    x28 += einsum(t1, (0, 1), l1, (2, 0), (2, 1))
    x29 = np.zeros((nvir, nvir), dtype=np.float64)
    x29 += einsum(x28, (0, 1), (0, 1)) * 2.0
    x29 += einsum(x16, (0, 1), (0, 1))
    rdm2_f_ovov += einsum(x29, (0, 1), delta.oo, (2, 3), (3, 0, 2, 1)) * 0.5
    rdm2_f_ovvo += einsum(x29, (0, 1), delta.oo, (2, 3), (3, 0, 1, 2)) * -0.5
    rdm2_f_voov += einsum(x29, (0, 1), delta.oo, (2, 3), (0, 3, 2, 1)) * -0.5
    rdm2_f_vovo += einsum(x29, (0, 1), delta.oo, (2, 3), (0, 3, 1, 2)) * 0.5
    x30 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x30 += einsum(t2, (0, 1, 2, 3), l1, (4, 1), (0, 4, 2, 3))
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    rdm2_f_ovvv += einsum(x30, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=np.float64)
    rdm2_f_vovv += einsum(x30, (0, 1, 2, 3), (1, 0, 3, 2))
    x31 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x31 += einsum(x1, (0, 1, 2, 3), t2, (0, 1, 4, 5), (2, 3, 4, 5))
    rdm2_f_ovvv += einsum(x31, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    rdm2_f_vovv += einsum(x31, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    x32 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x32 += einsum(t1, (0, 1), x27, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    rdm2_f_ovvv += einsum(x32, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x32, (0, 1, 2, 3), (1, 0, 3, 2))
    x33 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x33 += einsum(x18, (0, 1, 2, 3), t1, (0, 4), (1, 2, 4, 3))
    x34 = np.zeros((nvir, nvir), dtype=np.float64)
    x34 += einsum(x28, (0, 1), (0, 1))
    x34 += einsum(x16, (0, 1), (0, 1)) * 0.5
    x35 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x35 += einsum(x33, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x35 += einsum(t1, (0, 1), x34, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovvv += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x35, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x35, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x35, (0, 1, 2, 3), (1, 0, 3, 2))
    x36 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x36 += einsum(l2, (0, 1, 2, 3), t1, (3, 4), (2, 0, 1, 4))
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=np.float64)
    rdm2_f_vvov += einsum(x36, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=np.float64)
    rdm2_f_vvvo += einsum(x36, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vvvv += einsum(x36, (0, 1, 2, 3), t1, (0, 4), (1, 2, 4, 3))

    rdm2_f = pack_2e(rdm2_f_oooo, rdm2_f_ooov, rdm2_f_oovo, rdm2_f_ovoo, rdm2_f_vooo, rdm2_f_oovv, rdm2_f_ovov, rdm2_f_ovvo, rdm2_f_voov, rdm2_f_vovo, rdm2_f_vvoo, rdm2_f_ovvv, rdm2_f_vovv, rdm2_f_vvov, rdm2_f_vvvo, rdm2_f_vvvv)

    rdm2_f = rdm2_f.swapaxes(1, 2)

    return rdm2_f

