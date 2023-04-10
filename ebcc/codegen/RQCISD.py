# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 1, 3)) * -0.5
    x0 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1))
    e_cc = 0
    e_cc += einsum(x0, (0, 1, 2, 3), t2, (0, 1, 2, 3), ()) * 2.0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=np.float64)
    t1new += einsum(t1, (0, 1), f.vv, (2, 1), (0, 2))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2new += einsum(t2, (0, 1, 2, 3), v.ovov, (1, 2, 4, 5), (0, 4, 3, 5)) * -1.0
    t2new += einsum(v.oovv, (0, 1, 2, 3), t2, (1, 4, 5, 3), (0, 4, 5, 2)) * -1.0
    t2new += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1))
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 2, 3, 5), (0, 1, 4, 5))
    x0 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x0 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 3, 2))
    x0 += einsum(v.ovvv, (0, 1, 2, 3), (0, 3, 1, 2)) * -0.5
    t1new += einsum(x0, (0, 1, 2, 3), t2, (4, 0, 2, 1), (4, 3)) * 2.0
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x1 += einsum(v.ovov, (0, 1, 2, 3), t1, (4, 3), (4, 0, 2, 1))
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x2 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3)) * 3.0
    x2 += einsum(v.ooov, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    x2 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x2 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3)) * 3.0
    t1new += einsum(t2, (0, 1, 2, 3), x2, (4, 0, 1, 3), (4, 2)) * -0.5
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x3 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3))
    x3 += einsum(v.ooov, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    x3 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x3 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3))
    t1new += einsum(x3, (0, 1, 2, 3), t2, (1, 2, 3, 4), (0, 4)) * 0.5
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x4 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 1, 3)) * -0.5
    x4 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1))
    x5 = np.zeros((nocc, nvir), dtype=np.float64)
    x5 += einsum(f.ov, (0, 1), (0, 1))
    x5 += einsum(t1, (0, 1), x4, (0, 2, 1, 3), (2, 3)) * 2.0
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    t1new += einsum(x6, (0, 1, 2, 3), x5, (1, 2), (0, 3)) * 2.0
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x7 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1))
    x7 += einsum(v.oovv, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    t1new += einsum(t1, (0, 1), x7, (0, 2, 1, 3), (2, 3)) * 2.0
    t2new += einsum(t2, (0, 1, 2, 3), x7, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x8 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 1, 3))
    x8 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1)) * -0.5
    x9 = np.zeros((nocc, nocc), dtype=np.float64)
    x9 += einsum(f.oo, (0, 1), (1, 0))
    x9 += einsum(t2, (0, 1, 2, 3), x8, (1, 4, 2, 3), (4, 0)) * 2.0
    t1new += einsum(x9, (0, 1), t1, (0, 2), (1, 2)) * -1.0
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x10 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new += einsum(x10, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x10, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x11 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new += einsum(x11, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new += einsum(x11, (0, 1, 2, 3), (1, 0, 2, 3))
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x12 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x12 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x13 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1))
    x13 += einsum(t2, (0, 1, 2, 3), v.ovov, (1, 4, 5, 3), (0, 5, 2, 4)) * -1.0
    x13 += einsum(x12, (0, 1, 2, 3), v.ovov, (1, 3, 4, 5), (0, 4, 2, 5))
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x14 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x14 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x14 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new += einsum(x13, (0, 1, 2, 3), x14, (4, 1, 5, 3), (0, 4, 2, 5))
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x15 += einsum(v.oovv, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x15 += einsum(t2, (0, 1, 2, 3), v.ovov, (1, 4, 5, 2), (0, 5, 3, 4))
    t2new += einsum(t2, (0, 1, 2, 3), x15, (4, 0, 5, 2), (4, 1, 5, 3))
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x16 += einsum(v.oovv, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x16 += einsum(t2, (0, 1, 2, 3), v.ovov, (0, 4, 5, 3), (1, 5, 2, 4))
    t2new += einsum(t2, (0, 1, 2, 3), x16, (4, 1, 5, 2), (0, 4, 5, 3))
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x17 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x17 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1))
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x18 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 1, 3))
    x18 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1)) * -0.3333333333333333
    x19 = np.zeros((nvir, nvir), dtype=np.float64)
    x19 += einsum(f.vv, (0, 1), (1, 0)) * 2.0
    x19 += einsum(t2, (0, 1, 2, 3), x17, (0, 1, 4, 3), (2, 4)) * -1.0
    x19 += einsum(t2, (0, 1, 2, 3), x18, (0, 1, 4, 2), (3, 4)) * -3.0
    t2new += einsum(x19, (0, 1), t2, (2, 3, 4, 1), (2, 3, 4, 0)) * 0.5
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x20 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x20 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1)) * 3.0
    x21 = np.zeros((nvir, nvir), dtype=np.float64)
    x21 += einsum(f.vv, (0, 1), (1, 0)) * 2.0
    x21 += einsum(x20, (0, 1, 2, 3), t2, (0, 1, 4, 3), (4, 2)) * -1.0
    x21 += einsum(t2, (0, 1, 2, 3), x17, (0, 1, 4, 2), (3, 4))
    t2new += einsum(t2, (0, 1, 2, 3), x21, (4, 2), (0, 1, 4, 3)) * 0.5
    x22 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x22 += einsum(v.oooo, (0, 1, 2, 3), (2, 3, 1, 0))
    x22 += einsum(v.ovov, (0, 1, 2, 3), t2, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x22, (0, 4, 1, 5), (4, 5, 2, 3))
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x23 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 1, 3))
    x23 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    x24 = np.zeros((nocc, nocc), dtype=np.float64)
    x24 += einsum(f.oo, (0, 1), (1, 0))
    x24 += einsum(v.ovov, (0, 1, 2, 3), t2, (0, 4, 1, 3), (4, 2))
    x24 += einsum(t2, (0, 1, 2, 3), x23, (1, 4, 3, 2), (0, 4)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), x24, (4, 1), (0, 4, 2, 3)) * -1.0
    x25 = np.zeros((nocc, nocc), dtype=np.float64)
    x25 += einsum(f.oo, (0, 1), (1, 0))
    x25 += einsum(t2, (0, 1, 2, 3), x4, (1, 4, 3, 2), (0, 4)) * 2.0
    t2new += einsum(x25, (0, 1), t2, (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0

    return {"t1new": t1new, "t2new": t2new}

