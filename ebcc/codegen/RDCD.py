# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), x0, (0, 1, 2, 3), ()) * 2.0
    del x0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=np.float64)
    t1new += einsum(v.ovov, (0, 1, 2, 3), (0, 1)) * 2.0
    t1new += einsum(v.ovov, (0, 1, 2, 3), (0, 3)) * -1.0
    x0 = np.zeros((nocc, nvir), dtype=np.float64)
    x0 += einsum(t2, (0, 1, 2, 3), (0, 2))
    x0 += einsum(t2, (0, 1, 2, 3), (0, 3)) * -0.5
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x1 += einsum(x0, (0, 1), v.ovov, (2, 1, 3, 4), (0, 2, 3, 4)) * 4.0
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x2 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x2 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x1
    t1new += einsum(t2, (0, 1, 2, 3), x2, (4, 0, 1, 2), (4, 3))
    del x2
    x3 = np.zeros((nocc, nvir), dtype=np.float64)
    x3 += einsum(t2, (0, 1, 2, 3), (0, 2)) * 2.0
    x3 += einsum(t2, (0, 1, 2, 3), (0, 3)) * -1.0
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x4 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x4 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x5 = np.zeros((nocc, nvir), dtype=np.float64)
    x5 += einsum(v.ovov, (0, 1, 2, 3), (0, 1))
    x5 += einsum(v.oovv, (0, 1, 2, 3), (0, 2)) * -0.5
    x5 += einsum(x3, (0, 1), x4, (0, 2, 1, 3), (2, 3)) * 0.5
    del x3, x4
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t1new += einsum(x5, (0, 1), x6, (0, 2, 1, 3), (2, 3)) * 2.0
    del x5, x6
    x7 = np.zeros((nocc, nocc, nvir), dtype=np.float64)
    x7 += einsum(t2, (0, 1, 2, 3), (0, 1, 2))
    x7 += einsum(t2, (0, 1, 2, 3), (0, 1, 3)) * -0.5
    t1new += einsum(x7, (0, 1, 2), v.ovov, (3, 4, 0, 2), (1, 4)) * -2.0
    x8 = np.zeros((nocc, nocc, nvir), dtype=np.float64)
    x8 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3))
    x8 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 5)) * -1.0
    x8 += einsum(x7, (0, 1, 2), v.ovov, (3, 4, 0, 2), (1, 3, 4)) * 2.0
    del x7
    x9 = np.zeros((nocc, nvir, nvir), dtype=np.float64)
    x9 += einsum(t2, (0, 1, 2, 3), (0, 2, 3)) * -1.0
    x9 += einsum(t2, (0, 1, 2, 3), (0, 3, 2)) * 2.0
    t1new += einsum(x8, (0, 1, 2), x9, (1, 3, 2), (0, 3)) * -1.0
    del x8, x9
    x10 = np.zeros((nvir, nvir, nvir), dtype=np.float64)
    x10 += einsum(v.vvvv, (0, 1, 1, 2), (0, 1, 2)) * -0.5
    x10 += einsum(v.vvvv, (0, 1, 1, 2), (0, 1, 2))
    t1new += einsum(x10, (0, 1, 2), t2, (3, 4, 2, 0), (3, 1)) * 2.0
    del x10
    x11 = np.zeros((nocc, nocc, nvir), dtype=np.float64)
    x11 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2)) * -1.0
    x11 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 5))
    x12 = np.zeros((nocc, nvir, nvir), dtype=np.float64)
    x12 += einsum(t2, (0, 1, 2, 3), (0, 2, 3))
    x12 += einsum(t2, (0, 1, 2, 3), (0, 3, 2)) * -0.5
    t1new += einsum(x11, (0, 1, 2), x12, (1, 3, 2), (0, 3)) * 2.0
    del x11, x12
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x13 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x13 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t1new += einsum(x0, (0, 1), x13, (0, 2, 1, 3), (2, 3)) * 2.0
    del x13
    x14 = np.zeros((nocc, nocc, nvir), dtype=np.float64)
    x14 += einsum(t2, (0, 1, 2, 3), (0, 1, 2)) * -1.0
    x14 += einsum(t2, (0, 1, 2, 3), (0, 1, 3)) * 2.0
    t1new += einsum(x14, (0, 1, 2), v.oovv, (3, 0, 4, 2), (1, 4)) * -1.0
    del x14
    x15 = np.zeros((nocc, nocc, nocc), dtype=np.float64)
    x15 += einsum(t2, (0, 1, 2, 3), v.ovov, (1, 2, 4, 3), (0, 1, 4))
    x16 = np.zeros((nocc, nocc, nocc), dtype=np.float64)
    x16 += einsum(v.oooo, (0, 1, 1, 2), (0, 1, 2))
    x16 += einsum(v.oooo, (0, 1, 1, 2), (0, 1, 2)) * -0.5
    x16 += einsum(x15, (0, 1, 2), (1, 0, 2))
    x16 += einsum(x15, (0, 1, 2), (2, 0, 1)) * -0.5
    del x15
    t1new += einsum(x16, (0, 1, 2), t2, (0, 2, 3, 4), (1, 3)) * 2.0
    del x16
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x17 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x17 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x18 = np.zeros((nvir,), dtype=np.float64)
    x18 += einsum(t2, (0, 1, 2, 3), x17, (0, 1, 3, 4), (4,)) * 2.0
    x19 = np.zeros((nocc, nvir, nvir), dtype=np.float64)
    x19 += einsum(t2, (0, 1, 2, 3), (0, 2, 3)) * -0.5
    x19 += einsum(t2, (0, 1, 2, 3), (0, 3, 2))
    t1new += einsum(x18, (0,), x19, (1, 0, 2), (1, 2)) * -2.0
    del x18, x19
    x20 = np.zeros((nocc, nocc), dtype=np.float64)
    x20 += einsum(t2, (0, 1, 2, 3), x17, (1, 4, 2, 3), (0, 4)) * 2.0
    t1new += einsum(x0, (0, 1), x20, (2, 0), (2, 1)) * -2.0
    del x0, x20
    x21 = np.zeros((nocc,), dtype=np.float64)
    x21 += einsum(t2, (0, 1, 2, 3), x17, (0, 4, 3, 2), (4,))
    del x17
    x22 = np.zeros((nocc, nocc, nvir), dtype=np.float64)
    x22 += einsum(t2, (0, 1, 2, 3), (0, 1, 2)) * 2.0
    x22 += einsum(t2, (0, 1, 2, 3), (0, 1, 3)) * -1.0
    t1new += einsum(x21, (0,), x22, (1, 0, 2), (1, 2)) * -2.0
    del x21, x22

    return {"t2new": t2new}

