# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 1, 3))
    x0 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1)) * -0.5
    e_cc = 0
    e_cc += einsum(x0, (0, 1, 2, 3), t2, (1, 0, 2, 3), ()) * 2.0
    x1 = np.zeros((nocc, nvir), dtype=np.float64)
    x1 += einsum(f.ov, (0, 1), (0, 1))
    x1 += einsum(x0, (0, 1, 2, 3), t1, (0, 3), (1, 2))
    e_cc += einsum(t1, (0, 1), x1, (0, 1), ()) * 2.0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=np.float64)
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t1new += einsum(t1, (0, 1), f.vv, (2, 1), (0, 2))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2new += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (3, 2, 4, 1)) * -1.0
    t2new += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1))
    t2new += einsum(t2, (0, 1, 2, 3), v.ovov, (1, 3, 4, 5), (4, 0, 5, 2))
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    x1 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x1 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 3, 2))
    x1 += einsum(v.ovvv, (0, 1, 2, 3), (0, 3, 1, 2)) * -0.5
    t1new += einsum(x0, (0, 1, 2, 3), x1, (1, 3, 2, 4), (0, 4)) * 2.0
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x2 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x3 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.3333333333333333
    x3 += einsum(v.ooov, (0, 1, 2, 3), (1, 2, 0, 3))
    x3 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3))
    x3 += einsum(x2, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.3333333333333333
    t1new += einsum(t2, (0, 1, 2, 3), x3, (4, 1, 0, 3), (4, 2)) * -1.5
    x4 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x4 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3))
    x4 += einsum(v.ooov, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    x4 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x4 += einsum(x2, (0, 1, 2, 3), (0, 2, 1, 3))
    t1new += einsum(t2, (0, 1, 2, 3), x4, (4, 1, 0, 2), (4, 3)) * -0.5
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x5 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 1, 3))
    x5 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1)) * -0.5
    x6 = np.zeros((nocc, nvir), dtype=np.float64)
    x6 += einsum(f.ov, (0, 1), (0, 1))
    x6 += einsum(t1, (0, 1), x5, (0, 2, 3, 1), (2, 3)) * 2.0
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x7 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x7 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    t1new += einsum(x7, (0, 1, 2, 3), x6, (1, 2), (0, 3)) * 2.0
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x8 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1))
    x8 += einsum(v.oovv, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    t1new += einsum(t1, (0, 1), x8, (0, 2, 1, 3), (2, 3)) * 2.0
    x9 = np.zeros((nocc, nocc), dtype=np.float64)
    x9 += einsum(t1, (0, 1), f.ov, (2, 1), (2, 0))
    x10 = np.zeros((nocc, nocc), dtype=np.float64)
    x10 += einsum(x0, (0, 1, 2, 3), x5, (1, 4, 2, 3), (0, 4)) * 2.0
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x11 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3))
    x11 += einsum(v.ooov, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.5
    x12 = np.zeros((nocc, nocc), dtype=np.float64)
    x12 += einsum(t1, (0, 1), x11, (2, 3, 0, 1), (2, 3)) * 2.0
    x13 = np.zeros((nocc, nocc), dtype=np.float64)
    x13 += einsum(f.oo, (0, 1), (1, 0))
    x13 += einsum(x9, (0, 1), (0, 1))
    x13 += einsum(x10, (0, 1), (1, 0))
    x13 += einsum(x12, (0, 1), (1, 0))
    t1new += einsum(t1, (0, 1), x13, (0, 2), (2, 1)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), x13, (0, 4), (4, 1, 2, 3)) * -1.0
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x14 += einsum(v.ovov, (0, 1, 2, 3), t2, (4, 0, 1, 5), (4, 2, 5, 3))
    t2new += einsum(x14, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x15 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x15 += einsum(t1, (0, 1), v.vvvv, (2, 1, 3, 4), (0, 3, 4, 2))
    t2new += einsum(t1, (0, 1), x15, (2, 3, 1, 4), (0, 2, 3, 4))
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x16 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x17 += einsum(x16, (0, 1, 2, 3), t2, (4, 1, 5, 2), (0, 4, 5, 3))
    t2new += einsum(x17, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new += einsum(x17, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x18 += einsum(x16, (0, 1, 2, 3), t2, (4, 1, 2, 5), (0, 4, 5, 3))
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x19 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x20 += einsum(x0, (0, 1, 2, 3), v.ooov, (4, 5, 1, 2), (4, 5, 0, 3))
    x21 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x21 += einsum(x19, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x21 += einsum(x20, (0, 1, 2, 3), (2, 1, 0, 3))
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x22 += einsum(t1, (0, 1), x21, (2, 0, 3, 4), (2, 3, 1, 4))
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x23 += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3))
    x23 += einsum(x18, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x23 += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x23, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3))
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x24 += einsum(x2, (0, 1, 2, 3), t2, (4, 1, 3, 5), (0, 4, 2, 5))
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x25 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1))
    x25 += einsum(x16, (0, 1, 2, 3), (1, 0, 2, 3))
    x26 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x26 += einsum(t1, (0, 1), x25, (2, 3, 1, 4), (0, 2, 3, 4))
    x27 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x27 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x27 += einsum(x26, (0, 1, 2, 3), (0, 2, 1, 3))
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x28 += einsum(t1, (0, 1), x27, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new += einsum(x28, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x28, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x29 += einsum(t2, (0, 1, 2, 3), x14, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new += einsum(x29, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x30 += einsum(t2, (0, 1, 2, 3), x2, (4, 1, 5, 3), (4, 0, 5, 2))
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x31 += einsum(t1, (0, 1), x30, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new += einsum(x31, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x31, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x32 += einsum(v.ooov, (0, 1, 2, 3), t2, (4, 2, 5, 3), (4, 0, 1, 5))
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x33 += einsum(t1, (0, 1), x32, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new += einsum(x33, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new += einsum(x33, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x34 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x34 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1))
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x35 += einsum(x5, (0, 1, 2, 3), t2, (4, 0, 5, 3), (4, 1, 5, 2)) * 2.0
    x36 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x36 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x36 += einsum(v.ovvv, (0, 1, 2, 3), (0, 3, 1, 2))
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x37 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x37 += einsum(v.ooov, (0, 1, 2, 3), (2, 1, 0, 3))
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x38 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1))
    x38 += einsum(v.oovv, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x38 += einsum(x34, (0, 1, 2, 3), x0, (4, 0, 3, 5), (4, 1, 5, 2))
    x38 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    x38 += einsum(t1, (0, 1), x36, (2, 3, 1, 4), (0, 2, 4, 3)) * -1.0
    x38 += einsum(t1, (0, 1), x37, (2, 3, 0, 4), (3, 2, 1, 4)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), x38, (4, 0, 5, 2), (4, 1, 5, 3))
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x39 += einsum(t1, (0, 1), v.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x40 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x41 += einsum(v.ovov, (0, 1, 2, 3), x0, (4, 0, 3, 5), (4, 2, 5, 1))
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x42 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1)) * 2.0
    x42 += einsum(v.oovv, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x42 += einsum(x39, (0, 1, 2, 3), (1, 0, 2, 3))
    x42 += einsum(x40, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x42 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    x42 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(t2, (0, 1, 2, 3), x42, (4, 1, 5, 3), (0, 4, 2, 5))
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x43 += einsum(v.oovv, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x43 += einsum(x39, (0, 1, 2, 3), (1, 0, 2, 3))
    x43 += einsum(x40, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x43 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x43, (0, 1, 2, 3), t2, (1, 4, 5, 3), (0, 4, 5, 2))
    x44 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x44 += einsum(v.ooov, (0, 1, 2, 3), (1, 2, 0, 3))
    x44 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3))
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x45 += einsum(v.oovv, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x45 += einsum(x40, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x45 += einsum(t1, (0, 1), x44, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new += einsum(t2, (0, 1, 2, 3), x45, (4, 1, 5, 2), (0, 4, 5, 3))
    x46 = np.zeros((nvir, nvir, nvir, nvir), dtype=np.float64)
    x46 += einsum(t1, (0, 1), v.ovvv, (0, 2, 3, 4), (1, 2, 3, 4))
    x47 = np.zeros((nvir, nvir, nvir, nvir), dtype=np.float64)
    x47 += einsum(v.vvvv, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    x47 += einsum(x46, (0, 1, 2, 3), (1, 0, 3, 2))
    x47 += einsum(x46, (0, 1, 2, 3), (3, 2, 0, 1))
    t2new += einsum(x47, (0, 1, 2, 3), t2, (4, 5, 3, 0), (4, 5, 2, 1)) * -1.0
    x48 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x48 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x49 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x49 += einsum(v.oooo, (0, 1, 2, 3), (2, 3, 1, 0))
    x49 += einsum(x48, (0, 1, 2, 3), (3, 0, 2, 1))
    x49 += einsum(x48, (0, 1, 2, 3), (2, 1, 3, 0))
    x49 += einsum(x0, (0, 1, 2, 3), v.ovov, (4, 3, 5, 2), (5, 0, 4, 1))
    t2new += einsum(t2, (0, 1, 2, 3), x49, (0, 4, 1, 5), (4, 5, 2, 3))
    x50 = np.zeros((nvir, nvir), dtype=np.float64)
    x50 += einsum(t1, (0, 1), f.ov, (0, 2), (2, 1))
    x51 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x51 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    x51 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x51 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x52 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x52 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 4.0
    x52 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x52 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x53 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x53 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x53 += einsum(v.ovvv, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    x54 = np.zeros((nvir, nvir), dtype=np.float64)
    x54 += einsum(x53, (0, 1, 2, 3), t1, (0, 1), (2, 3)) * 0.5
    x55 = np.zeros((nvir, nvir), dtype=np.float64)
    x55 += einsum(f.vv, (0, 1), (1, 0)) * -0.5
    x55 += einsum(x50, (0, 1), (0, 1)) * 0.5
    x55 += einsum(v.ovov, (0, 1, 2, 3), x51, (0, 2, 3, 4), (1, 4)) * -0.25
    x55 += einsum(x52, (0, 1, 2, 3), v.ovov, (0, 2, 1, 4), (4, 3)) * 0.25
    x55 += einsum(x54, (0, 1), (1, 0)) * -1.0
    t2new += einsum(x55, (0, 1), t2, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x56 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 0.6666666666666666
    x56 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.3333333333333333
    x56 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x57 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 4.0
    x57 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x57 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x58 = np.zeros((nvir, nvir), dtype=np.float64)
    x58 += einsum(f.vv, (0, 1), (1, 0)) * -0.5
    x58 += einsum(x50, (0, 1), (0, 1)) * 0.5
    x58 += einsum(v.ovov, (0, 1, 2, 3), x56, (0, 2, 3, 4), (1, 4)) * -0.75
    x58 += einsum(v.ovov, (0, 1, 2, 3), x57, (0, 2, 1, 4), (3, 4)) * 0.25
    x58 += einsum(x54, (0, 1), (1, 0)) * -1.0
    t2new += einsum(x58, (0, 1), t2, (2, 3, 0, 4), (2, 3, 1, 4)) * -2.0
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x59 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    x59 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x59 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2))
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x60 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x60 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    x61 = np.zeros((nocc, nocc), dtype=np.float64)
    x61 += einsum(f.oo, (0, 1), (1, 0))
    x61 += einsum(x9, (0, 1), (0, 1))
    x61 += einsum(v.ovov, (0, 1, 2, 3), x59, (0, 4, 3, 1), (2, 4)) * -1.0
    x61 += einsum(x12, (0, 1), (1, 0))
    x61 += einsum(v.ovov, (0, 1, 2, 3), x60, (4, 0, 3, 1), (2, 4))
    t2new += einsum(x61, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    x62 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x62 += einsum(v.oooo, (0, 1, 2, 3), (2, 3, 1, 0))
    x62 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (5, 1, 0, 4))
    x62 += einsum(t1, (0, 1), x2, (2, 3, 4, 1), (3, 0, 2, 4))
    x63 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x63 += einsum(v.ooov, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    x63 += einsum(t1, (0, 1), x62, (0, 2, 3, 4), (3, 2, 4, 1))
    t2new += einsum(t1, (0, 1), x63, (2, 3, 0, 4), (2, 3, 1, 4))
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x64 += einsum(v.ovov, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    x64 += einsum(x14, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x64, (0, 1, 2, 3), t2, (4, 1, 3, 5), (4, 0, 5, 2))

    return {"t1new": t1new, "t2new": t2new}

def energy_perturbative(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    e_ia = direct_sum("i-a->ia", np.diag(f.oo), np.diag(f.vv))
    denom3 = 1 / direct_sum("ia+jb+kc->ijkabc", e_ia, e_ia, e_ia)

    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x0 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (0, 3, 5, 6), v.ovvv, (0, 4, 3, 7), (2, 1, 5, 4, 6, 7))
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x1 += einsum(x0, (0, 1, 2, 3, 4, 5), t2, (1, 0, 2, 5), (1, 0, 3, 4))
    e_pert = 0
    e_pert += einsum(x1, (0, 1, 2, 3), l2, (2, 3, 0, 1), ())
    e_pert += einsum(x1, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * -1.0
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x2 += einsum(x0, (0, 1, 2, 3, 4, 5), t2, (1, 0, 3, 4), (1, 0, 2, 5))
    e_pert += einsum(x2, (0, 1, 2, 3), l2, (2, 3, 0, 1), ())
    e_pert += einsum(x2, (0, 1, 2, 3), l2, (3, 2, 0, 1), ()) * -1.0
    e_pert += einsum(x2, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * -2.0
    x3 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x3 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (0, 5, 3, 6), v.ovvv, (0, 4, 3, 7), (2, 1, 5, 4, 6, 7))
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x4 += einsum(x3, (0, 1, 2, 3, 4, 5), t2, (1, 0, 5, 2), (1, 0, 3, 4))
    e_pert += einsum(x4, (0, 1, 2, 3), l2, (2, 3, 0, 1), ())
    e_pert += einsum(x4, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * -1.0
    x5 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x5 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (0, 3, 4, 6), t2, (1, 0, 3, 7), (1, 2, 4, 5, 7, 6))
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x6 += einsum(x5, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 3, 2, 4), (0, 1, 3, 5))
    e_pert += einsum(x6, (0, 1, 2, 3), l2, (2, 3, 0, 1), ())
    e_pert += einsum(x6, (0, 1, 2, 3), l2, (3, 2, 0, 1), ()) * -1.0
    e_pert += einsum(x6, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * -2.0
    x7 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x7 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (0, 4, 3, 6), t2, (1, 0, 3, 7), (1, 2, 4, 5, 7, 6))
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x8 += einsum(x7, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 2, 3, 4), (0, 1, 3, 5))
    e_pert += einsum(x8, (0, 1, 2, 3), l2, (2, 3, 0, 1), ())
    e_pert += einsum(x8, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * -1.0
    x9 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x9 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 3, 5), denom3, (1, 0, 6, 3, 4, 7), (0, 6, 4, 7, 2, 5))
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x10 += einsum(v.ovvv, (0, 1, 2, 3), x9, (4, 0, 2, 1, 3, 5), (4, 0, 1, 5))
    e_pert += einsum(x10, (0, 1, 2, 3), l2, (2, 3, 0, 1), ())
    e_pert += einsum(x10, (0, 1, 2, 3), l2, (3, 2, 0, 1), ()) * -1.0
    e_pert += einsum(x10, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * -2.0
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x11 += einsum(x3, (0, 1, 2, 3, 4, 5), t2, (0, 1, 2, 5), (0, 1, 3, 4))
    e_pert += einsum(x11, (0, 1, 2, 3), l2, (2, 3, 1, 0), ())
    e_pert += einsum(x11, (0, 1, 2, 3), l2, (3, 2, 1, 0), ()) * -1.0
    e_pert += einsum(x11, (0, 1, 2, 3), l2, (2, 3, 0, 1), ()) * -2.0
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x12 += einsum(x0, (0, 1, 2, 3, 4, 5), t2, (0, 1, 5, 2), (0, 1, 3, 4))
    e_pert += einsum(l2, (0, 1, 2, 3), x12, (3, 2, 0, 1), ())
    e_pert += einsum(l2, (0, 1, 2, 3), x12, (3, 2, 1, 0), ()) * -1.0
    e_pert += einsum(l2, (0, 1, 2, 3), x12, (2, 3, 0, 1), ()) * -2.0
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x13 += einsum(x0, (0, 1, 2, 3, 4, 5), t2, (0, 1, 4, 3), (0, 1, 2, 5))
    e_pert += einsum(l2, (0, 1, 2, 3), x13, (3, 2, 0, 1), ())
    e_pert += einsum(l2, (0, 1, 2, 3), x13, (2, 3, 0, 1), ()) * -1.0
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x14 += einsum(x5, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 2, 3, 4), (0, 1, 3, 5))
    e_pert += einsum(x14, (0, 1, 2, 3), l2, (2, 3, 1, 0), ())
    e_pert += einsum(x14, (0, 1, 2, 3), l2, (3, 2, 1, 0), ()) * -1.0
    e_pert += einsum(x14, (0, 1, 2, 3), l2, (2, 3, 0, 1), ()) * -2.0
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x15 += einsum(v.ovvv, (0, 1, 2, 3), x9, (4, 0, 1, 2, 3, 5), (4, 0, 2, 5))
    e_pert += einsum(l2, (0, 1, 2, 3), x15, (3, 2, 0, 1), ())
    e_pert += einsum(l2, (0, 1, 2, 3), x15, (2, 3, 0, 1), ()) * -1.0
    x16 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x16 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (0, 4, 3, 6), t2, (0, 1, 3, 7), (1, 2, 4, 5, 7, 6))
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x17 += einsum(v.ovvv, (0, 1, 2, 3), x16, (4, 0, 2, 1, 3, 5), (4, 0, 1, 5))
    e_pert += einsum(x17, (0, 1, 2, 3), l2, (2, 3, 0, 1), ())
    e_pert += einsum(x17, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * -1.0
    x18 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x18 += einsum(t2, (0, 1, 2, 3), v.ovvv, (0, 3, 4, 5), denom3, (0, 1, 6, 3, 4, 7), (1, 6, 4, 7, 2, 5))
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x19 += einsum(v.ovvv, (0, 1, 2, 3), x18, (4, 0, 2, 1, 3, 5), (4, 0, 1, 5))
    e_pert += einsum(x19, (0, 1, 2, 3), l2, (2, 3, 0, 1), ())
    e_pert += einsum(x19, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * -1.0
    x20 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x20 += einsum(t2, (0, 1, 2, 3), v.ovvv, (0, 4, 3, 5), denom3, (0, 1, 6, 3, 4, 7), (1, 6, 4, 7, 2, 5))
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x21 += einsum(x20, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 2, 3, 4), (0, 1, 3, 5))
    e_pert += einsum(x21, (0, 1, 2, 3), l2, (2, 3, 0, 1), ())
    e_pert += einsum(x21, (0, 1, 2, 3), l2, (3, 2, 0, 1), ()) * -1.0
    e_pert += einsum(x21, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * -2.0
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x22 += einsum(v.ovvv, (0, 1, 2, 3), x16, (4, 0, 1, 2, 3, 5), (4, 0, 2, 5))
    e_pert += einsum(x22, (0, 1, 2, 3), l2, (2, 3, 1, 0), ())
    e_pert += einsum(x22, (0, 1, 2, 3), l2, (3, 2, 1, 0), ()) * -1.0
    e_pert += einsum(x22, (0, 1, 2, 3), l2, (2, 3, 0, 1), ()) * -2.0
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x23 += einsum(v.ovvv, (0, 1, 2, 3), x18, (4, 0, 1, 2, 3, 5), (4, 0, 2, 5))
    e_pert += einsum(x23, (0, 1, 2, 3), l2, (2, 3, 1, 0), ())
    e_pert += einsum(x23, (0, 1, 2, 3), l2, (2, 3, 0, 1), ()) * -1.0
    x24 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x24 += einsum(v.ovvv, (0, 1, 2, 3), denom3, (0, 4, 5, 1, 3, 6), v.ovvv, (0, 1, 6, 7), (5, 4, 6, 3, 7, 2))
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x25 += einsum(t2, (0, 1, 2, 3), x24, (1, 0, 2, 4, 5, 3), (0, 1, 4, 5))
    e_pert += einsum(x25, (0, 1, 2, 3), l2, (3, 2, 0, 1), ())
    e_pert += einsum(x25, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * 3.0
    e_pert += einsum(x25, (0, 1, 2, 3), l2, (2, 3, 0, 1), ()) * -2.0
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x26 += einsum(x7, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 3, 2, 4), (0, 1, 3, 5))
    e_pert += einsum(l2, (0, 1, 2, 3), x26, (2, 3, 1, 0), ())
    e_pert += einsum(l2, (0, 1, 2, 3), x26, (3, 2, 0, 1), ()) * 4.0
    e_pert += einsum(l2, (0, 1, 2, 3), x26, (2, 3, 0, 1), ()) * -3.0
    x27 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x27 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 3, 4, 5), denom3, (1, 0, 6, 3, 4, 7), (0, 6, 4, 7, 2, 5))
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x28 += einsum(v.ovvv, (0, 1, 2, 3), x27, (4, 0, 2, 1, 3, 5), (4, 0, 1, 5))
    e_pert += einsum(x28, (0, 1, 2, 3), l2, (3, 2, 0, 1), ())
    e_pert += einsum(x28, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * 4.0
    e_pert += einsum(x28, (0, 1, 2, 3), l2, (2, 3, 0, 1), ()) * -3.0
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x29 += einsum(t2, (0, 1, 2, 3), x24, (0, 1, 3, 4, 5, 2), (0, 1, 4, 5))
    e_pert += einsum(x29, (0, 1, 2, 3), l2, (3, 2, 1, 0), ())
    e_pert += einsum(x29, (0, 1, 2, 3), l2, (2, 3, 0, 1), ()) * 3.0
    e_pert += einsum(x29, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * -2.0
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x30 += einsum(v.ovvv, (0, 1, 2, 3), x27, (4, 0, 1, 2, 3, 5), (4, 0, 2, 5))
    e_pert += einsum(x30, (0, 1, 2, 3), l2, (3, 2, 1, 0), ())
    e_pert += einsum(x30, (0, 1, 2, 3), l2, (2, 3, 0, 1), ()) * 4.0
    e_pert += einsum(x30, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * -3.0
    x31 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x31 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (0, 3, 4, 6), t2, (0, 1, 3, 7), (1, 2, 4, 5, 7, 6))
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x32 += einsum(x31, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 3, 2, 4), (0, 1, 3, 5))
    e_pert += einsum(x32, (0, 1, 2, 3), l2, (3, 2, 0, 1), ())
    e_pert += einsum(x32, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * 2.0
    e_pert += einsum(x32, (0, 1, 2, 3), l2, (2, 3, 0, 1), ()) * -1.0
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x33 += einsum(x20, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 3, 2, 4), (0, 1, 3, 5))
    e_pert += einsum(x33, (0, 1, 2, 3), l2, (3, 2, 0, 1), ())
    e_pert += einsum(x33, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * 2.0
    e_pert += einsum(x33, (0, 1, 2, 3), l2, (2, 3, 0, 1), ()) * -1.0
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x34 += einsum(x31, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 2, 3, 4), (0, 1, 3, 5))
    e_pert += einsum(x34, (0, 1, 2, 3), l2, (3, 2, 1, 0), ())
    e_pert += einsum(x34, (0, 1, 2, 3), l2, (2, 3, 0, 1), ()) * 2.0
    e_pert += einsum(x34, (0, 1, 2, 3), l2, (2, 3, 1, 0), ()) * -1.0
    x35 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x35 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (2, 3, 5, 6), v.ovvv, (1, 5, 3, 7), (1, 2, 0, 4, 7, 6))
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x36 += einsum(l2, (0, 1, 2, 3), x35, (3, 4, 2, 0, 5, 1), (2, 4, 0, 5))
    e_pert += einsum(x36, (0, 1, 2, 3), t2, (0, 1, 2, 3), ()) * 4.0
    x37 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x37 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (2, 5, 3, 6), v.ovvv, (1, 5, 3, 7), (1, 2, 0, 4, 7, 6))
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x38 += einsum(l2, (0, 1, 2, 3), x37, (3, 4, 2, 0, 5, 1), (2, 4, 0, 5))
    e_pert += einsum(x38, (0, 1, 2, 3), t2, (0, 1, 2, 3), ()) * -1.0
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x39 += einsum(x35, (0, 1, 2, 3, 4, 5), l2, (3, 5, 0, 2), (2, 1, 3, 4))
    e_pert += einsum(x39, (0, 1, 2, 3), t2, (0, 1, 2, 3), ()) * -3.0
    x40 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x40 += einsum(v.ovvv, (0, 1, 2, 3), denom3, (4, 0, 5, 2, 6, 1), v.ovov, (4, 2, 5, 1), (5, 4, 0, 6, 3))
    x41 = np.zeros((nocc, nvir), dtype=np.float64)
    x41 += einsum(t2, (0, 1, 2, 3), x40, (1, 0, 4, 2, 3), (4, 2))
    e_pert += einsum(x41, (0, 1), l1, (1, 0), ())
    x42 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x42 += einsum(v.ovov, (0, 1, 2, 3), t2, (0, 2, 3, 4), denom3, (0, 5, 2, 3, 1, 6), (5, 1, 6, 4))
    x43 = np.zeros((nocc, nvir), dtype=np.float64)
    x43 += einsum(v.ovvv, (0, 1, 2, 3), x42, (0, 1, 2, 3), (0, 2))
    e_pert += einsum(l1, (0, 1), x43, (1, 0), ())
    x44 = np.zeros((nocc, nvir), dtype=np.float64)
    x44 += einsum(t2, (0, 1, 2, 3), x40, (0, 1, 4, 3, 2), (4, 3))
    e_pert += einsum(x44, (0, 1), l1, (1, 0), ())
    x45 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x45 += einsum(v.ovov, (0, 1, 2, 3), t2, (0, 2, 4, 1), denom3, (0, 5, 2, 1, 3, 6), (5, 3, 6, 4))
    x46 = np.zeros((nocc, nvir), dtype=np.float64)
    x46 += einsum(v.ovvv, (0, 1, 2, 3), x45, (0, 1, 2, 3), (0, 2))
    e_pert += einsum(l1, (0, 1), x46, (1, 0), ())
    x47 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x47 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovov, (0, 3, 2, 5), v.ovvv, (0, 4, 3, 6), (2, 1, 5, 4, 6))
    x48 = np.zeros((nocc, nvir), dtype=np.float64)
    x48 += einsum(t2, (0, 1, 2, 3), x47, (1, 0, 2, 4, 3), (0, 4))
    e_pert += einsum(l1, (0, 1), x48, (1, 0), ()) * 2.0
    x49 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x49 += einsum(v.ovov, (0, 1, 2, 3), denom3, (0, 4, 2, 1, 5, 3), v.ovvv, (0, 3, 6, 5), (2, 4, 5, 1, 6))
    x50 = np.zeros((nocc, nvir), dtype=np.float64)
    x50 += einsum(t2, (0, 1, 2, 3), x49, (1, 0, 4, 2, 3), (0, 4))
    e_pert += einsum(x50, (0, 1), l1, (1, 0), ()) * 2.0
    x51 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x51 += einsum(v.ovvv, (0, 1, 2, 3), denom3, (0, 4, 5, 2, 6, 1), v.ovov, (0, 2, 4, 1), (4, 5, 6, 3))
    x52 = np.zeros((nocc, nvir), dtype=np.float64)
    x52 += einsum(x51, (0, 1, 2, 3), t2, (1, 0, 3, 2), (1, 2))
    e_pert += einsum(l1, (0, 1), x52, (1, 0), ()) * 2.0
    x53 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x53 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovov, (0, 3, 2, 5), v.ovvv, (0, 4, 5, 6), (2, 1, 3, 4, 6))
    x54 = np.zeros((nocc, nvir), dtype=np.float64)
    x54 += einsum(x53, (0, 1, 2, 3, 4), t2, (1, 0, 4, 2), (1, 3))
    e_pert += einsum(x54, (0, 1), l1, (1, 0), ()) * 2.0
    x55 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x55 += einsum(v.ovov, (0, 1, 2, 3), denom3, (0, 4, 2, 1, 5, 3), v.ovvv, (0, 1, 6, 5), (2, 4, 5, 3, 6))
    x56 = np.zeros((nocc, nvir), dtype=np.float64)
    x56 += einsum(x55, (0, 1, 2, 3, 4), t2, (0, 1, 3, 4), (1, 2))
    e_pert += einsum(l1, (0, 1), x56, (1, 0), ()) * 2.0
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x57 += einsum(v.ovvv, (0, 1, 2, 3), denom3, (0, 4, 5, 2, 6, 1), v.ovov, (0, 1, 4, 2), (4, 5, 6, 3))
    x58 = np.zeros((nocc, nvir), dtype=np.float64)
    x58 += einsum(x57, (0, 1, 2, 3), t2, (0, 1, 3, 2), (1, 2))
    e_pert += einsum(l1, (0, 1), x58, (1, 0), ()) * 2.0
    x59 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x59 += einsum(v.ovov, (0, 1, 2, 3), t2, (0, 2, 1, 4), denom3, (0, 5, 2, 1, 3, 6), (5, 3, 6, 4))
    x60 = np.zeros((nocc, nvir), dtype=np.float64)
    x60 += einsum(v.ovvv, (0, 1, 2, 3), x59, (0, 2, 1, 3), (0, 1))
    e_pert += einsum(l1, (0, 1), x60, (1, 0), ()) * 4.0
    x61 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x61 += einsum(v.ovov, (0, 1, 2, 3), t2, (0, 2, 4, 3), denom3, (0, 5, 2, 3, 1, 6), (5, 1, 6, 4))
    x62 = np.zeros((nocc, nvir), dtype=np.float64)
    x62 += einsum(v.ovvv, (0, 1, 2, 3), x61, (0, 2, 1, 3), (0, 1))
    e_pert += einsum(l1, (0, 1), x62, (1, 0), ()) * 4.0
    x63 = np.zeros((nocc, nvir), dtype=np.float64)
    x63 += einsum(x57, (0, 1, 2, 3), t2, (1, 0, 2, 3), (1, 2))
    e_pert += einsum(x63, (0, 1), l1, (1, 0), ()) * 6.0
    x64 = np.zeros((nocc, nvir), dtype=np.float64)
    x64 += einsum(x55, (0, 1, 2, 3, 4), t2, (1, 0, 4, 3), (1, 2))
    e_pert += einsum(l1, (0, 1), x64, (1, 0), ()) * 6.0
    x65 = np.zeros((nocc, nvir), dtype=np.float64)
    x65 += einsum(v.ovvv, (0, 1, 2, 3), x59, (0, 1, 2, 3), (0, 2))
    e_pert += einsum(x65, (0, 1), l1, (1, 0), ()) * -1.0
    x66 = np.zeros((nocc, nvir), dtype=np.float64)
    x66 += einsum(t2, (0, 1, 2, 3), x40, (1, 0, 4, 3, 2), (4, 3))
    e_pert += einsum(x66, (0, 1), l1, (1, 0), ()) * -1.0
    x67 = np.zeros((nocc, nvir), dtype=np.float64)
    x67 += einsum(x51, (0, 1, 2, 3), t2, (1, 0, 2, 3), (1, 2))
    e_pert += einsum(x67, (0, 1), l1, (1, 0), ()) * -2.0
    x68 = np.zeros((nocc, nvir), dtype=np.float64)
    x68 += einsum(x49, (0, 1, 2, 3, 4), t2, (1, 0, 4, 3), (1, 2))
    e_pert += einsum(x68, (0, 1), l1, (1, 0), ()) * -2.0
    x69 = np.zeros((nocc, nvir), dtype=np.float64)
    x69 += einsum(t2, (0, 1, 2, 3), x49, (0, 1, 4, 2, 3), (1, 4))
    e_pert += einsum(x69, (0, 1), l1, (1, 0), ()) * -2.0
    x70 = np.zeros((nocc, nvir), dtype=np.float64)
    x70 += einsum(x51, (0, 1, 2, 3), t2, (0, 1, 3, 2), (1, 2))
    e_pert += einsum(x70, (0, 1), l1, (1, 0), ()) * -2.0
    x71 = np.zeros((nocc, nvir), dtype=np.float64)
    x71 += einsum(v.ovvv, (0, 1, 2, 3), x42, (0, 2, 1, 3), (0, 1))
    e_pert += einsum(x71, (0, 1), l1, (1, 0), ()) * -2.0
    x72 = np.zeros((nocc, nvir), dtype=np.float64)
    x72 += einsum(v.ovvv, (0, 1, 2, 3), x45, (0, 2, 1, 3), (0, 1))
    e_pert += einsum(x72, (0, 1), l1, (1, 0), ()) * -2.0
    x73 = np.zeros((nocc, nvir), dtype=np.float64)
    x73 += einsum(x55, (0, 1, 2, 3, 4), t2, (1, 0, 3, 4), (1, 2))
    e_pert += einsum(x73, (0, 1), l1, (1, 0), ()) * -4.0
    x74 = np.zeros((nocc, nvir), dtype=np.float64)
    x74 += einsum(t2, (0, 1, 2, 3), x53, (1, 0, 2, 4, 3), (0, 4))
    e_pert += einsum(x74, (0, 1), l1, (1, 0), ()) * -4.0
    x75 = np.zeros((nocc, nvir), dtype=np.float64)
    x75 += einsum(x57, (0, 1, 2, 3), t2, (1, 0, 3, 2), (1, 2))
    e_pert += einsum(l1, (0, 1), x75, (1, 0), ()) * -4.0
    x76 = np.zeros((nocc, nvir), dtype=np.float64)
    x76 += einsum(t2, (0, 1, 2, 3), x47, (1, 0, 3, 4, 2), (0, 4))
    e_pert += einsum(x76, (0, 1), l1, (1, 0), ()) * -4.0
    x77 = np.zeros((nocc, nvir), dtype=np.float64)
    x77 += einsum(t2, (0, 1, 2, 3), x40, (0, 1, 4, 2, 3), (4, 2))
    e_pert += einsum(x77, (0, 1), l1, (1, 0), ()) * -3.0
    x78 = np.zeros((nocc, nvir), dtype=np.float64)
    x78 += einsum(v.ovvv, (0, 1, 2, 3), x61, (0, 1, 2, 3), (0, 2))
    e_pert += einsum(l1, (0, 1), x78, (1, 0), ()) * -3.0
    x79 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x79 += einsum(t2, (0, 1, 2, 3), denom3, (4, 5, 0, 2, 6, 3), l2, (2, 3, 5, 7), (5, 0, 4, 7, 1, 6))
    x80 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x80 += einsum(v.ooov, (0, 1, 2, 3), x79, (4, 0, 2, 1, 5, 3), (4, 2, 5, 3))
    e_pert += einsum(x80, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * 0.5
    x81 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x81 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (2, 6, 5, 3), l2, (3, 5, 1, 7), (1, 2, 0, 7, 6, 4))
    x82 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x82 += einsum(v.ooov, (0, 1, 2, 3), x81, (4, 2, 0, 1, 5, 3), (4, 0, 5, 3))
    e_pert += einsum(x82, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * 0.5
    x83 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x83 += einsum(t2, (0, 1, 2, 3), denom3, (4, 5, 1, 2, 6, 3), l2, (2, 3, 5, 7), (5, 1, 4, 7, 0, 6))
    x84 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x84 += einsum(v.ooov, (0, 1, 2, 3), x83, (4, 2, 0, 1, 5, 3), (4, 0, 5, 3))
    e_pert += einsum(x84, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * 0.5
    x85 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x85 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (6, 2, 5, 3), l2, (3, 5, 1, 7), (1, 2, 0, 7, 6, 4))
    x86 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x86 += einsum(v.ooov, (0, 1, 2, 3), x85, (4, 0, 2, 1, 5, 3), (4, 2, 5, 3))
    e_pert += einsum(x86, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * 0.5
    x87 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x87 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), denom3, (6, 3, 4, 0, 7, 1), (3, 4, 6, 2, 5, 7))
    x88 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x88 += einsum(v.ooov, (0, 1, 2, 3), x87, (4, 2, 0, 1, 5, 3), (4, 0, 5, 3))
    e_pert += einsum(x88, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * 0.5
    x89 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x89 += einsum(l2, (0, 1, 2, 3), denom3, (4, 3, 5, 0, 6, 1), t2, (5, 7, 1, 0), (3, 5, 4, 2, 7, 6))
    x90 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x90 += einsum(v.ooov, (0, 1, 2, 3), x89, (4, 0, 2, 1, 5, 3), (4, 2, 5, 3))
    e_pert += einsum(x90, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * 0.5
    x91 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x91 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), denom3, (6, 3, 5, 0, 7, 1), (3, 5, 6, 2, 4, 7))
    x92 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x92 += einsum(v.ooov, (0, 1, 2, 3), x91, (4, 0, 2, 1, 5, 3), (4, 2, 5, 3))
    e_pert += einsum(x92, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * 0.5
    x93 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x93 += einsum(l2, (0, 1, 2, 3), denom3, (4, 3, 5, 0, 6, 1), t2, (7, 5, 1, 0), (3, 5, 4, 2, 7, 6))
    x94 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x94 += einsum(v.ooov, (0, 1, 2, 3), x93, (4, 2, 0, 1, 5, 3), (4, 0, 5, 3))
    e_pert += einsum(x94, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * 0.5
    x95 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x95 += einsum(v.ooov, (0, 1, 2, 3), x81, (4, 0, 2, 1, 5, 3), (4, 2, 5, 3))
    e_pert += einsum(x95, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -0.5
    x96 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x96 += einsum(v.ooov, (0, 1, 2, 3), x85, (4, 2, 0, 1, 5, 3), (4, 0, 5, 3))
    e_pert += einsum(x96, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -0.5
    x97 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x97 += einsum(v.ooov, (0, 1, 2, 3), x87, (4, 0, 2, 1, 5, 3), (4, 2, 5, 3))
    e_pert += einsum(x97, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -0.5
    x98 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x98 += einsum(v.ooov, (0, 1, 2, 3), x91, (4, 2, 0, 1, 5, 3), (4, 0, 5, 3))
    e_pert += einsum(x98, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -0.5
    x99 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x99 += einsum(v.ooov, (0, 1, 2, 3), x79, (4, 2, 0, 1, 5, 3), (4, 0, 5, 3))
    e_pert += einsum(x99, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -1.5
    x100 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x100 += einsum(v.ooov, (0, 1, 2, 3), x83, (4, 0, 2, 1, 5, 3), (4, 2, 5, 3))
    e_pert += einsum(x100, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -1.5
    x101 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x101 += einsum(v.ooov, (0, 1, 2, 3), x89, (4, 2, 0, 1, 5, 3), (4, 0, 5, 3))
    e_pert += einsum(x101, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -1.5
    x102 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x102 += einsum(v.ooov, (0, 1, 2, 3), x93, (4, 0, 2, 1, 5, 3), (4, 2, 5, 3))
    e_pert += einsum(x102, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -1.5
    x103 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x103 += einsum(v.ovov, (0, 1, 2, 3), v.ooov, (0, 4, 5, 1), denom3, (0, 5, 2, 1, 6, 3), (5, 2, 4, 3, 6))
    x104 = np.zeros((nocc, nvir), dtype=np.float64)
    x104 += einsum(t2, (0, 1, 2, 3), x103, (4, 0, 1, 2, 3), (4, 3))
    e_pert += einsum(x104, (0, 1), l1, (1, 0), ())
    x105 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x105 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (0, 6, 5, 3), v.ovov, (0, 5, 1, 3), (1, 2, 6, 4))
    x106 = np.zeros((nocc, nvir), dtype=np.float64)
    x106 += einsum(x105, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), (1, 3))
    e_pert += einsum(x106, (0, 1), l1, (1, 0), ())
    x107 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x107 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ooov, (6, 1, 0, 3), v.ovov, (0, 5, 2, 3), (1, 2, 6, 5, 4))
    x108 = np.zeros((nocc, nvir), dtype=np.float64)
    x108 += einsum(t2, (0, 1, 2, 3), x107, (4, 1, 0, 3, 2), (4, 2))
    e_pert += einsum(l1, (0, 1), x108, (1, 0), ())
    x109 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x109 += einsum(v.ovov, (0, 1, 2, 3), v.ooov, (4, 5, 0, 1), denom3, (0, 5, 2, 1, 6, 3), (5, 2, 4, 3, 6))
    x110 = np.zeros((nocc, nvir), dtype=np.float64)
    x110 += einsum(t2, (0, 1, 2, 3), x109, (4, 1, 0, 2, 3), (4, 3))
    e_pert += einsum(l1, (0, 1), x110, (1, 0), ())
    x111 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x111 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ooov, (0, 6, 1, 3), v.ovov, (0, 5, 2, 3), (1, 2, 6, 5, 4))
    x112 = np.zeros((nocc, nvir), dtype=np.float64)
    x112 += einsum(t2, (0, 1, 2, 3), x111, (4, 1, 0, 2, 3), (4, 3))
    e_pert += einsum(l1, (0, 1), x112, (1, 0), ())
    x113 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x113 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (6, 0, 5, 3), v.ovov, (0, 5, 1, 3), (1, 2, 6, 4))
    x114 = np.zeros((nocc, nvir), dtype=np.float64)
    x114 += einsum(x113, (0, 1, 2, 3), v.ooov, (0, 2, 1, 3), (1, 3))
    e_pert += einsum(l1, (0, 1), x114, (1, 0), ())
    x115 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x115 += einsum(v.ovov, (0, 1, 2, 3), denom3, (0, 4, 2, 1, 3, 5), v.ooov, (0, 6, 2, 1), (4, 6, 3, 5))
    x116 = np.zeros((nocc, nvir), dtype=np.float64)
    x116 += einsum(x115, (0, 1, 2, 3), t2, (0, 1, 3, 2), (0, 3))
    e_pert += einsum(l1, (0, 1), x116, (1, 0), ()) * 2.0
    x117 = np.zeros((nocc, nvir), dtype=np.float64)
    x117 += einsum(x115, (0, 1, 2, 3), t2, (1, 0, 2, 3), (0, 3))
    e_pert += einsum(l1, (0, 1), x117, (1, 0), ()) * 2.0
    x118 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x118 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ooov, (0, 6, 2, 3), v.ovov, (0, 4, 2, 3), (1, 6, 4, 5))
    x119 = np.zeros((nocc, nvir), dtype=np.float64)
    x119 += einsum(x118, (0, 1, 2, 3), t2, (0, 1, 2, 3), (0, 3))
    e_pert += einsum(l1, (0, 1), x119, (1, 0), ()) * 4.0
    x120 = np.zeros((nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x120 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovov, (0, 3, 2, 5), t2, (1, 6, 5, 3), (1, 2, 0, 6, 4))
    x121 = np.zeros((nocc, nvir), dtype=np.float64)
    x121 += einsum(x120, (0, 1, 2, 3, 4), v.ooov, (2, 3, 1, 4), (0, 4))
    e_pert += einsum(l1, (0, 1), x121, (1, 0), ()) * 4.0
    x122 = np.zeros((nocc, nvir), dtype=np.float64)
    x122 += einsum(t2, (0, 1, 2, 3), x109, (4, 0, 1, 3, 2), (4, 2))
    e_pert += einsum(x122, (0, 1), l1, (1, 0), ()) * 3.0
    x123 = np.zeros((nocc, nvir), dtype=np.float64)
    x123 += einsum(t2, (0, 1, 2, 3), x111, (4, 0, 1, 3, 2), (4, 2))
    e_pert += einsum(x123, (0, 1), l1, (1, 0), ()) * 3.0
    x124 = np.zeros((nocc, nvir), dtype=np.float64)
    x124 += einsum(t2, (0, 1, 2, 3), x107, (4, 0, 1, 2, 3), (4, 3))
    e_pert += einsum(x124, (0, 1), l1, (1, 0), ()) * 3.0
    x125 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x125 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (0, 6, 5, 3), v.ovov, (0, 3, 1, 5), (1, 2, 6, 4))
    x126 = np.zeros((nocc, nvir), dtype=np.float64)
    x126 += einsum(x125, (0, 1, 2, 3), v.ooov, (0, 2, 1, 3), (1, 3))
    e_pert += einsum(x126, (0, 1), l1, (1, 0), ()) * 3.0
    x127 = np.zeros((nocc, nvir), dtype=np.float64)
    x127 += einsum(t2, (0, 1, 2, 3), x103, (4, 1, 0, 3, 2), (4, 2))
    e_pert += einsum(l1, (0, 1), x127, (1, 0), ()) * 3.0
    x128 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x128 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (6, 0, 5, 3), v.ovov, (0, 3, 1, 5), (1, 2, 6, 4))
    x129 = np.zeros((nocc, nvir), dtype=np.float64)
    x129 += einsum(x128, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), (1, 3))
    e_pert += einsum(x129, (0, 1), l1, (1, 0), ()) * 3.0
    x130 = np.zeros((nocc, nvir), dtype=np.float64)
    x130 += einsum(t2, (0, 1, 2, 3), x107, (4, 0, 1, 3, 2), (4, 2))
    e_pert += einsum(x130, (0, 1), l1, (1, 0), ()) * -1.0
    x131 = np.zeros((nocc, nvir), dtype=np.float64)
    x131 += einsum(t2, (0, 1, 2, 3), x103, (4, 0, 1, 3, 2), (4, 2))
    e_pert += einsum(l1, (0, 1), x131, (1, 0), ()) * -1.0
    x132 = np.zeros((nocc, nvir), dtype=np.float64)
    x132 += einsum(t2, (0, 1, 2, 3), x111, (4, 0, 1, 2, 3), (4, 3))
    e_pert += einsum(x132, (0, 1), l1, (1, 0), ()) * -1.0
    x133 = np.zeros((nocc, nvir), dtype=np.float64)
    x133 += einsum(x125, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), (1, 3))
    e_pert += einsum(x133, (0, 1), l1, (1, 0), ()) * -1.0
    x134 = np.zeros((nocc, nvir), dtype=np.float64)
    x134 += einsum(t2, (0, 1, 2, 3), x111, (4, 1, 0, 3, 2), (4, 2))
    e_pert += einsum(l1, (0, 1), x134, (1, 0), ()) * -1.0
    x135 = np.zeros((nocc, nvir), dtype=np.float64)
    x135 += einsum(t2, (0, 1, 2, 3), x107, (4, 1, 0, 2, 3), (4, 3))
    e_pert += einsum(l1, (0, 1), x135, (1, 0), ()) * -1.0
    x136 = np.zeros((nocc, nvir), dtype=np.float64)
    x136 += einsum(t2, (0, 1, 2, 3), x103, (4, 1, 0, 2, 3), (4, 3))
    e_pert += einsum(l1, (0, 1), x136, (1, 0), ()) * -1.0
    x137 = np.zeros((nocc, nvir), dtype=np.float64)
    x137 += einsum(x113, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), (1, 3))
    e_pert += einsum(x137, (0, 1), l1, (1, 0), ()) * -1.0
    x138 = np.zeros((nocc, nvir), dtype=np.float64)
    x138 += einsum(x115, (0, 1, 2, 3), t2, (0, 1, 2, 3), (0, 3))
    e_pert += einsum(x138, (0, 1), l1, (1, 0), ()) * -2.0
    x139 = np.zeros((nocc, nvir), dtype=np.float64)
    x139 += einsum(x120, (0, 1, 2, 3, 4), v.ooov, (1, 3, 2, 4), (0, 4))
    e_pert += einsum(x139, (0, 1), l1, (1, 0), ()) * -2.0
    x140 = np.zeros((nocc, nvir), dtype=np.float64)
    x140 += einsum(x118, (0, 1, 2, 3), t2, (1, 0, 2, 3), (0, 3))
    e_pert += einsum(x140, (0, 1), l1, (1, 0), ()) * -2.0
    x141 = np.zeros((nocc, nvir), dtype=np.float64)
    x141 += einsum(t2, (0, 1, 2, 3), x109, (4, 1, 0, 3, 2), (4, 2))
    e_pert += einsum(l1, (0, 1), x141, (1, 0), ()) * -3.0
    x142 = np.zeros((nocc, nvir), dtype=np.float64)
    x142 += einsum(x128, (0, 1, 2, 3), v.ooov, (0, 2, 1, 3), (1, 3))
    e_pert += einsum(l1, (0, 1), x142, (1, 0), ()) * -3.0
    x143 = np.zeros((nocc, nvir), dtype=np.float64)
    x143 += einsum(x118, (0, 1, 2, 3), t2, (0, 1, 3, 2), (0, 3))
    e_pert += einsum(x143, (0, 1), l1, (1, 0), ()) * -6.0
    x144 = np.zeros((nocc, nvir), dtype=np.float64)
    x144 += einsum(t2, (0, 1, 2, 3), x109, (4, 0, 1, 2, 3), (4, 3))
    e_pert += einsum(x144, (0, 1), l1, (1, 0), ()) * -5.0
    x145 = np.zeros((nocc, nvir), dtype=np.float64)
    x145 += einsum(x105, (0, 1, 2, 3), v.ooov, (0, 2, 1, 3), (1, 3))
    e_pert += einsum(x145, (0, 1), l1, (1, 0), ()) * -5.0
    x146 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x146 += einsum(t2, (0, 1, 2, 3), denom3, (4, 5, 0, 6, 3, 2), l2, (2, 7, 0, 5), (5, 4, 1, 3, 6, 7))
    x147 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x147 += einsum(v.ovvv, (0, 1, 2, 3), x146, (4, 0, 5, 2, 1, 3), (4, 0, 5, 1))
    e_pert += einsum(x147, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ())
    x148 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x148 += einsum(l2, (0, 1, 2, 3), t2, (3, 4, 5, 0), denom3, (6, 2, 3, 7, 5, 0), (2, 6, 4, 5, 7, 1))
    x149 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x149 += einsum(x148, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 4, 3, 5), (0, 1, 2, 4))
    e_pert += einsum(x149, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ())
    x150 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x150 += einsum(t2, (0, 1, 2, 3), denom3, (4, 5, 1, 6, 3, 2), l2, (2, 7, 1, 5), (5, 4, 0, 3, 6, 7))
    x151 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x151 += einsum(x150, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 3, 4, 5), (0, 1, 2, 4))
    e_pert += einsum(x151, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ())
    x152 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x152 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 5, 0), denom3, (6, 2, 3, 7, 5, 0), (2, 6, 4, 5, 7, 1))
    x153 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x153 += einsum(x152, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 3, 4, 5), (0, 1, 2, 4))
    e_pert += einsum(x153, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ())
    x154 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x154 += einsum(l2, (0, 1, 2, 3), t2, (3, 4, 5, 1), denom3, (6, 2, 3, 7, 5, 1), (2, 6, 4, 5, 7, 0))
    x155 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x155 += einsum(x154, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 3, 4, 5), (0, 1, 2, 4))
    e_pert += einsum(x155, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ())
    x156 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x156 += einsum(t2, (0, 1, 2, 3), denom3, (4, 5, 1, 6, 3, 2), l2, (7, 2, 1, 5), (5, 4, 0, 3, 6, 7))
    x157 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x157 += einsum(x156, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 4, 3, 5), (0, 1, 2, 4))
    e_pert += einsum(x157, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ())
    x158 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x158 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 5, 1), denom3, (6, 2, 3, 7, 5, 1), (2, 6, 4, 5, 7, 0))
    x159 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x159 += einsum(x158, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 4, 3, 5), (0, 1, 2, 4))
    e_pert += einsum(x159, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ())
    x160 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x160 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (2, 6, 4, 5), l2, (5, 7, 2, 1), (1, 0, 6, 4, 3, 7))
    x161 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x161 += einsum(v.ovvv, (0, 1, 2, 3), x160, (4, 0, 5, 1, 2, 3), (4, 0, 5, 2))
    e_pert += einsum(x161, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * 2.0
    x162 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x162 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (6, 2, 4, 5), l2, (5, 7, 2, 1), (1, 0, 6, 4, 3, 7))
    x163 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x163 += einsum(x162, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 4, 3, 5), (0, 1, 2, 4))
    e_pert += einsum(x163, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * 2.0
    x164 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x164 += einsum(t2, (0, 1, 2, 3), l2, (2, 4, 5, 1), denom3, (6, 5, 1, 7, 3, 2), (5, 6, 0, 3, 7, 4))
    x165 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x165 += einsum(x164, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 4, 3, 5), (0, 1, 2, 4))
    e_pert += einsum(x165, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * 2.0
    x166 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x166 += einsum(t2, (0, 1, 2, 3), l2, (2, 4, 5, 0), denom3, (6, 5, 0, 7, 3, 2), (5, 6, 1, 3, 7, 4))
    x167 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x167 += einsum(v.ovvv, (0, 1, 2, 3), x166, (4, 0, 5, 1, 2, 3), (4, 0, 5, 2))
    e_pert += einsum(x167, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * 3.0
    x168 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x168 += einsum(v.ovvv, (0, 1, 2, 3), x160, (4, 0, 5, 2, 1, 3), (4, 0, 5, 1))
    e_pert += einsum(x168, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -1.0
    x169 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x169 += einsum(v.ovvv, (0, 1, 2, 3), x166, (4, 0, 5, 2, 1, 3), (4, 0, 5, 1))
    e_pert += einsum(x169, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -1.0
    x170 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x170 += einsum(x148, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 3, 4, 5), (0, 1, 2, 4))
    e_pert += einsum(x170, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -1.0
    x171 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x171 += einsum(x150, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 4, 3, 5), (0, 1, 2, 4))
    e_pert += einsum(x171, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -1.0
    x172 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x172 += einsum(x164, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 3, 4, 5), (0, 1, 2, 4))
    e_pert += einsum(x172, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -1.0
    x173 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x173 += einsum(x152, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 4, 3, 5), (0, 1, 2, 4))
    e_pert += einsum(x173, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -1.0
    x174 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x174 += einsum(t2, (0, 1, 2, 3), l2, (4, 2, 5, 0), denom3, (6, 5, 0, 7, 3, 2), (5, 6, 1, 3, 7, 4))
    x175 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x175 += einsum(x174, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 3, 4, 5), (0, 1, 2, 4))
    e_pert += einsum(x175, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -1.0
    x176 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x176 += einsum(x158, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 3, 4, 5), (0, 1, 2, 4))
    e_pert += einsum(x176, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -1.0
    x177 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x177 += einsum(x162, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 3, 4, 5), (0, 1, 2, 4))
    e_pert += einsum(x177, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -2.0
    x178 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x178 += einsum(v.ovvv, (0, 1, 2, 3), x146, (4, 0, 5, 1, 2, 3), (4, 0, 5, 2))
    e_pert += einsum(x178, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -4.0
    x179 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x179 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (0, 5, 3, 6), t2, (0, 7, 4, 3), (2, 1, 7, 4, 5, 6))
    x180 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x180 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x180 += einsum(l2, (0, 1, 2, 3), (3, 2, 1, 0))
    x181 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x181 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.5
    x181 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x181 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    x182 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x182 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.3333333333333333
    x182 += einsum(t2, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.6666666666666666
    x182 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2))
    x183 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x183 += einsum(x181, (0, 1, 2, 3), x180, (4, 0, 3, 5), (0, 4, 1, 3, 5, 2)) * 2.0
    x183 += einsum(x180, (0, 1, 2, 3), x182, (4, 1, 5, 3), (1, 0, 4, 3, 2, 5)) * 3.0
    x184 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x184 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.3333333333333333
    x184 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.6666666666666666
    x184 += einsum(t2, (0, 1, 2, 3), (1, 0, 2, 3)) * 1.3333333333333333
    x184 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x185 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x185 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x185 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.25
    x185 += einsum(t2, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.75
    x185 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2))
    x186 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x186 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x186 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2))
    x187 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x187 += einsum(x184, (0, 1, 2, 3), l2, (3, 4, 5, 1), (1, 5, 0, 3, 4, 2)) * 3.0
    x187 += einsum(x185, (0, 1, 2, 3), l2, (3, 4, 1, 5), (1, 5, 0, 3, 4, 2)) * 4.0
    x187 += einsum(l2, (0, 1, 2, 3), x186, (2, 4, 5, 1), (2, 3, 4, 1, 0, 5))
    x187 += einsum(l2, (0, 1, 2, 3), x186, (4, 3, 5, 1), (3, 2, 4, 1, 0, 5))
    x188 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x188 += einsum(denom3, (0, 1, 2, 3, 4, 5), x183, (0, 6, 1, 3, 4, 7), (1, 2, 6, 4, 5, 7))
    x188 += einsum(denom3, (0, 1, 2, 3, 4, 5), x187, (0, 2, 6, 3, 7, 5), (6, 1, 2, 7, 4, 5))
    x189 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x189 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x189 += einsum(t2, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x189 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2))
    x190 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x190 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x190 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x191 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x191 += einsum(x189, (0, 1, 2, 3), x180, (1, 4, 3, 5), (1, 4, 0, 3, 5, 2))
    x191 += einsum(x190, (0, 1, 2, 3), x180, (0, 4, 5, 3), (0, 4, 1, 3, 5, 2))
    x192 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x192 += einsum(x191, (0, 1, 2, 3, 4, 5), denom3, (0, 2, 6, 3, 4, 7), (2, 6, 1, 4, 7, 5))
    x193 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x193 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x193 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    x194 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x194 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x194 += einsum(t2, (0, 1, 2, 3), (1, 0, 2, 3))
    x195 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x195 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * 2.0
    x195 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    x195 += einsum(l2, (0, 1, 2, 3), (3, 2, 1, 0))
    x196 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x196 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x196 += einsum(l2, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    x196 += einsum(l2, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x197 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x197 += einsum(x193, (0, 1, 2, 3), x194, (0, 4, 2, 5), (0, 1, 4, 2, 3, 5)) * -1.0
    x197 += einsum(x195, (0, 1, 2, 3), t2, (1, 4, 5, 2), (1, 0, 4, 2, 3, 5))
    x197 += einsum(x196, (0, 1, 2, 3), t2, (4, 1, 5, 3), (1, 0, 4, 3, 2, 5))
    x198 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x198 += einsum(denom3, (0, 1, 2, 3, 4, 5), x197, (0, 1, 6, 3, 7, 4), (1, 2, 6, 4, 5, 7))
    x199 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x199 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.6
    x199 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.19999999999999998
    x199 += einsum(t2, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.6
    x199 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2))
    x200 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x200 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.3333333333333333
    x200 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x200 += einsum(t2, (0, 1, 2, 3), (1, 0, 2, 3)) * 1.666666666666667
    x200 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x201 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x201 += einsum(v.ooov, (0, 1, 2, 3), x199, (4, 2, 5, 3), (2, 4, 1, 0, 3, 5)) * 1.6666666666666667
    x201 += einsum(v.ooov, (0, 1, 2, 3), x200, (4, 1, 5, 3), (1, 4, 0, 2, 3, 5))
    x202 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x202 += einsum(denom3, (0, 1, 2, 3, 4, 5), x201, (0, 6, 7, 1, 3, 4), (1, 2, 6, 7, 4, 5)) * 3.0
    x203 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x203 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x203 += einsum(v.ooov, (0, 1, 2, 3), (2, 1, 0, 3))
    x204 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x204 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x204 += einsum(t2, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x205 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x205 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x205 += einsum(v.ooov, (0, 1, 2, 3), (2, 1, 0, 3)) * 3.0
    x206 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x206 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3))
    x206 += einsum(v.ooov, (0, 1, 2, 3), (2, 1, 0, 3)) * -0.3333333333333333
    x207 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x207 += einsum(x204, (0, 1, 2, 3), x203, (4, 5, 1, 2), (1, 0, 5, 4, 2, 3)) * 0.3333333333333333
    x207 += einsum(x205, (0, 1, 2, 3), t2, (2, 4, 5, 3), (2, 4, 1, 0, 3, 5)) * 0.3333333333333333
    x207 += einsum(x206, (0, 1, 2, 3), t2, (4, 2, 5, 3), (2, 4, 1, 0, 3, 5))
    x208 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x208 += einsum(x207, (0, 1, 2, 3, 4, 5), denom3, (0, 3, 6, 4, 5, 7), (3, 6, 1, 2, 5, 7)) * 3.0
    x209 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x209 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x209 += einsum(l2, (0, 1, 2, 3), (2, 3, 1, 0)) * -0.25
    x209 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.25
    x209 += einsum(l2, (0, 1, 2, 3), (3, 2, 1, 0))
    x210 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x210 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x210 += einsum(l2, (0, 1, 2, 3), (2, 3, 1, 0)) * -0.3333333333333333
    x210 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.3333333333333333
    x210 += einsum(l2, (0, 1, 2, 3), (3, 2, 1, 0))
    x211 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x211 += einsum(v.ovvv, (0, 1, 2, 3), x209, (4, 0, 5, 1), (0, 4, 1, 5, 3, 2))
    x211 += einsum(x210, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 3), (1, 0, 3, 2, 5, 4)) * -0.75
    x212 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x212 += einsum(denom3, (0, 1, 2, 3, 4, 5), x211, (0, 6, 3, 4, 7, 5), (1, 2, 6, 4, 5, 7)) * 1.3333333333333333
    x213 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x213 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x213 += einsum(l2, (0, 1, 2, 3), (2, 3, 1, 0)) * -0.5
    x213 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.5
    x213 += einsum(l2, (0, 1, 2, 3), (3, 2, 1, 0))
    x214 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x214 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x214 += einsum(l2, (0, 1, 2, 3), (2, 3, 1, 0)) * 3.0
    x214 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * 3.0
    x214 += einsum(l2, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x215 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x215 += einsum(v.ovvv, (0, 1, 2, 3), x213, (4, 0, 5, 1), (0, 4, 1, 5, 3, 2))
    x215 += einsum(x214, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 3), (1, 0, 3, 2, 5, 4)) * 0.5
    x216 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x216 += einsum(denom3, (0, 1, 2, 3, 4, 5), x215, (0, 6, 3, 4, 7, 5), (1, 2, 6, 4, 5, 7)) * 2.0
    x217 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x217 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x217 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.5
    x217 += einsum(l2, (0, 1, 2, 3), (3, 2, 1, 0)) * 0.5
    x218 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x218 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (2, 5, 6, 3), x204, (1, 7, 5, 3), (1, 2, 0, 7, 4, 6))
    x219 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x219 += einsum(v.ovvv, (0, 1, 2, 3), l2, (1, 2, 4, 5), (4, 5, 0, 1, 2, 3))
    x219 += einsum(v.ovvv, (0, 1, 2, 3), l2, (2, 1, 4, 5), (5, 4, 0, 2, 1, 3))
    x220 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x220 += einsum(denom3, (0, 1, 2, 3, 4, 5), x219, (6, 1, 2, 3, 5, 7), (6, 1, 2, 0, 7, 4))
    x221 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x221 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x221 += einsum(t2, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.75
    x221 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2))
    x222 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x222 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.6666666666666666
    x222 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x223 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x223 += einsum(x221, (0, 1, 2, 3), v.ovvv, (1, 3, 4, 5), (1, 0, 3, 2, 5, 4)) * 2.0
    x223 += einsum(x222, (0, 1, 2, 3), v.ovvv, (0, 4, 5, 3), (0, 1, 3, 2, 5, 4)) * -1.5
    x224 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x224 += einsum(x223, (0, 1, 2, 3, 4, 5), denom3, (0, 6, 7, 2, 3, 5), (6, 7, 1, 3, 5, 4))
    x225 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x225 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x225 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x226 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x226 += einsum(v.ovvv, (0, 1, 2, 3), l2, (1, 2, 0, 4), (0, 4, 1, 2, 3))
    x226 += einsum(v.ovvv, (0, 1, 2, 3), l2, (2, 1, 4, 0), (0, 4, 2, 1, 3))
    x227 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x227 += einsum(denom3, (0, 1, 2, 3, 4, 5), x226, (0, 6, 3, 5, 7), (6, 1, 2, 7, 4))
    x228 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x228 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x228 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x229 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x229 += einsum(v.ovvv, (0, 1, 2, 3), l2, (2, 1, 0, 4), (0, 4, 2, 1, 3))
    x229 += einsum(v.ovvv, (0, 1, 2, 3), l2, (1, 2, 4, 0), (0, 4, 1, 2, 3))
    x230 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x230 += einsum(denom3, (0, 1, 2, 3, 4, 5), x229, (0, 6, 3, 5, 7), (6, 1, 2, 7, 4))
    x231 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x231 += einsum(denom3, (0, 1, 2, 3, 4, 5), x219, (1, 6, 2, 3, 5, 7), (1, 6, 2, 0, 7, 4))
    x232 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x232 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 3, 2))
    x232 += einsum(v.ovvv, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    x233 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x233 += einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 2, 1), (0, 4, 2, 1, 3))
    x233 += einsum(x232, (0, 1, 2, 3), t2, (0, 4, 3, 1), (0, 4, 3, 1, 2)) * -1.0
    x234 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x234 += einsum(denom3, (0, 1, 2, 3, 4, 5), x233, (0, 6, 5, 3, 7), (1, 2, 6, 4, 7))
    x235 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x235 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * 4.0
    x235 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -3.0
    x235 += einsum(l2, (0, 1, 2, 3), (3, 2, 1, 0))
    x236 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x236 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (2, 3, 5, 6), t2, (1, 7, 5, 3), (1, 2, 0, 7, 4, 6))
    x237 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x237 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3)) * 3.0
    x237 += einsum(v.ooov, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x238 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x238 += einsum(x194, (0, 1, 2, 3), l2, (3, 2, 4, 5), (5, 4, 1, 0, 3, 2)) * -1.0
    x238 += einsum(l2, (0, 1, 2, 3), x194, (4, 5, 0, 1), (2, 3, 5, 4, 0, 1)) * -1.0
    x239 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x239 += einsum(x238, (0, 1, 2, 3, 4, 5), denom3, (6, 0, 3, 4, 7, 5), (0, 3, 6, 1, 2, 7))
    x240 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x240 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x240 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x241 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x241 += einsum(t2, (0, 1, 2, 3), l2, (3, 2, 1, 4), (1, 4, 0, 3, 2))
    x241 += einsum(t2, (0, 1, 2, 3), l2, (2, 3, 4, 1), (1, 4, 0, 2, 3))
    x241 += einsum(x240, (0, 1, 2, 3), x193, (0, 4, 2, 3), (0, 4, 1, 2, 3)) * -1.0
    x242 = np.zeros((nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x242 += einsum(x241, (0, 1, 2, 3, 4), denom3, (0, 5, 6, 3, 7, 4), (5, 6, 1, 2, 7))
    x243 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x243 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x243 += einsum(v.ovvv, (0, 1, 2, 3), (0, 3, 2, 1))
    x244 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x244 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (0, 6, 5, 3), x243, (0, 3, 7, 5), (1, 2, 6, 4, 7)) * 2.0
    x245 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x245 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x245 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x246 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x246 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (2, 3, 5, 6), t2, (7, 1, 5, 3), (1, 2, 0, 7, 4, 6))
    x247 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x247 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3))
    x247 += einsum(v.ooov, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x248 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x248 += einsum(t2, (0, 1, 2, 3), l2, (2, 3, 4, 5), (4, 5, 0, 1, 2, 3))
    x248 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 1, 0), (3, 2, 4, 5, 0, 1))
    x249 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x249 += einsum(denom3, (0, 1, 2, 3, 4, 5), x248, (6, 1, 7, 2, 3, 5), (1, 2, 0, 6, 7, 4))
    x250 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x250 += einsum(v.ooov, (0, 1, 2, 3), (1, 0, 2, 3)) * 1.6666666666666667
    x250 += einsum(v.ooov, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x251 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x251 += einsum(denom3, (0, 1, 2, 3, 4, 5), x248, (6, 1, 2, 7, 3, 5), (1, 2, 0, 6, 7, 4))
    x252 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x252 += einsum(x179, (0, 1, 2, 3, 4, 5), l2, (5, 3, 1, 0), (0, 1, 2, 4)) * -2.0
    x252 += einsum(x188, (0, 1, 2, 3, 4, 5), v.ovvv, (1, 4, 5, 3), (1, 0, 2, 4)) * 2.0
    x252 += einsum(v.ovvv, (0, 1, 2, 3), x192, (4, 0, 5, 1, 3, 2), (0, 4, 5, 3)) * -2.0
    x252 += einsum(v.ovvv, (0, 1, 2, 3), x198, (4, 0, 5, 1, 3, 2), (0, 5, 4, 3)) * -2.0
    x252 += einsum(x180, (0, 1, 2, 3), x202, (4, 0, 5, 1, 3, 2), (0, 4, 5, 2)) * -1.0
    x252 += einsum(x208, (0, 1, 2, 3, 4, 5), x180, (1, 3, 4, 5), (1, 0, 2, 5))
    x252 += einsum(x212, (0, 1, 2, 3, 4, 5), t2, (0, 1, 5, 3), (0, 1, 2, 4)) * 3.0
    x252 += einsum(x216, (0, 1, 2, 3, 4, 5), t2, (0, 1, 3, 5), (0, 1, 2, 4)) * -1.0
    x252 += einsum(x217, (0, 1, 2, 3), x218, (1, 4, 0, 5, 2, 3), (0, 4, 5, 2)) * -4.0
    x252 += einsum(x220, (0, 1, 2, 3, 4, 5), x182, (3, 1, 5, 4), (3, 2, 0, 5)) * 6.0
    x252 += einsum(x224, (0, 1, 2, 3, 4, 5), l2, (3, 5, 1, 0), (0, 1, 2, 4)) * 4.0
    x252 += einsum(x225, (0, 1, 2, 3), x227, (4, 0, 1, 2, 3), (0, 1, 4, 3)) * 4.0
    x252 += einsum(x230, (0, 1, 2, 3, 4), x228, (1, 2, 3, 4), (1, 2, 0, 4))
    x252 += einsum(x231, (0, 1, 2, 3, 4, 5), x225, (0, 3, 5, 4), (3, 2, 1, 5)) * -4.0
    x252 += einsum(x234, (0, 1, 2, 3, 4), l2, (3, 4, 1, 0), (0, 1, 2, 3)) * -4.0
    x252 += einsum(x235, (0, 1, 2, 3), x236, (1, 4, 0, 5, 2, 3), (0, 4, 5, 2)) * 2.0
    x252 += einsum(x237, (0, 1, 2, 3), x239, (4, 0, 2, 1, 5, 3), (2, 5, 4, 3))
    x252 += einsum(v.ooov, (0, 1, 2, 3), x242, (1, 2, 0, 4, 3), (1, 2, 4, 3)) * 2.0
    x252 += einsum(x244, (0, 1, 2, 3, 4), l2, (4, 3, 1, 0), (0, 1, 2, 3)) * 2.0
    x252 += einsum(x246, (0, 1, 2, 3, 4, 5), x245, (2, 0, 4, 5), (2, 1, 3, 4)) * 2.0
    x252 += einsum(x249, (0, 1, 2, 3, 4, 5), x247, (1, 3, 2, 5), (2, 4, 0, 5))
    x252 += einsum(x251, (0, 1, 2, 3, 4, 5), x250, (1, 3, 2, 5), (2, 4, 0, 5)) * -3.0
    e_pert += einsum(x252, (0, 1, 2, 3), v.ooov, (1, 2, 0, 3), ()) * -0.5
    x253 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x253 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (0, 6, 5, 3), x180, (7, 0, 3, 4), (7, 6, 1, 2, 4, 5))
    x254 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x254 += einsum(x253, (0, 1, 2, 3, 4, 5), v.ooov, (1, 2, 3, 4), (2, 3, 0, 5)) * 4.0
    x255 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x255 += einsum(t2, (0, 1, 2, 3), x180, (4, 0, 2, 5), (0, 4, 1, 5, 2, 3))
    x255 += einsum(t2, (0, 1, 2, 3), x180, (0, 4, 3, 5), (0, 4, 1, 5, 3, 2))
    x256 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x256 += einsum(x255, (0, 1, 2, 3, 4, 5), denom3, (0, 6, 7, 4, 3, 5), (1, 2, 6, 7, 3, 5))
    x257 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x257 += einsum(x256, (0, 1, 2, 3, 4, 5), v.ooov, (1, 2, 3, 4), (2, 3, 0, 5)) * 2.0
    x258 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x258 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (0, 6, 3, 5), x180, (7, 0, 4, 3), (7, 6, 1, 2, 4, 5))
    x259 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x259 += einsum(x258, (0, 1, 2, 3, 4, 5), v.ooov, (1, 2, 3, 4), (2, 3, 0, 5)) * 2.0
    x260 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x260 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (6, 0, 5, 3), x180, (7, 0, 4, 3), (7, 6, 1, 2, 4, 5))
    x261 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x261 += einsum(x260, (0, 1, 2, 3, 4, 5), v.ooov, (1, 2, 3, 4), (2, 3, 0, 5)) * 2.0
    x262 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x262 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x262 += einsum(l2, (0, 1, 2, 3), (2, 3, 1, 0))
    x262 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x262 += einsum(l2, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x263 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x263 += einsum(x228, (0, 1, 2, 3), x262, (4, 0, 5, 3), (0, 4, 1, 3, 5, 2))
    x263 += einsum(t2, (0, 1, 2, 3), x180, (4, 1, 2, 5), (1, 4, 0, 2, 5, 3))
    x263 += einsum(t2, (0, 1, 2, 3), x180, (4, 1, 5, 3), (1, 4, 0, 3, 5, 2))
    x264 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x264 += einsum(denom3, (0, 1, 2, 3, 4, 5), x263, (0, 6, 1, 3, 4, 7), (1, 2, 6, 4, 5, 7))
    x265 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x265 += einsum(x180, (0, 1, 2, 3), x182, (4, 0, 5, 2), (0, 1, 4, 2, 3, 5)) * 3.0
    x265 += einsum(x225, (0, 1, 2, 3), x180, (0, 4, 5, 3), (0, 4, 1, 3, 5, 2)) * -2.0
    x266 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x266 += einsum(denom3, (0, 1, 2, 3, 4, 5), x265, (0, 6, 1, 3, 4, 7), (1, 2, 6, 4, 5, 7))
    x267 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x267 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x267 += einsum(l2, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    x267 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    x267 += einsum(l2, (0, 1, 2, 3), (3, 2, 1, 0))
    x268 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x268 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.3333333333333333
    x268 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x269 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x269 += einsum(x204, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (1, 0, 4, 5, 2, 3)) * -1.0
    x269 += einsum(x268, (0, 1, 2, 3), v.ooov, (4, 5, 1, 2), (1, 0, 5, 4, 3, 2)) * 3.0
    x270 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x270 += einsum(x269, (0, 1, 2, 3, 4, 5), denom3, (0, 3, 6, 5, 4, 7), (3, 6, 1, 2, 4, 7))
    x271 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x271 += einsum(v.ooov, (0, 1, 2, 3), t2, (0, 4, 3, 5), (0, 4, 1, 2, 3, 5))
    x271 += einsum(v.ooov, (0, 1, 2, 3), t2, (2, 4, 5, 3), (2, 4, 1, 0, 3, 5))
    x272 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x272 += einsum(x271, (0, 1, 2, 3, 4, 5), denom3, (0, 3, 6, 4, 5, 7), (3, 6, 1, 2, 5, 7))
    x273 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x273 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (0, 5, 6, 3), x262, (7, 0, 4, 3), (1, 2, 7, 4, 5, 6)) * -1.0
    x274 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x274 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x274 += einsum(t2, (0, 1, 2, 3), (1, 0, 2, 3))
    x275 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x275 += einsum(x274, (0, 1, 2, 3), v.ovvv, (4, 2, 5, 3), (1, 0, 4, 2, 3, 5))
    x275 += einsum(x204, (0, 1, 2, 3), v.ovvv, (4, 3, 5, 2), (1, 0, 4, 2, 3, 5)) * 0.5
    x276 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x276 += einsum(v.ooov, (0, 1, 2, 3), t2, (4, 0, 3, 5), (4, 0, 1, 2, 3, 5))
    x276 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 0, 2), (1, 0, 5, 4, 2, 3)) * 0.3333333333333333
    x277 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x277 += einsum(x276, (0, 1, 2, 3, 4, 5), denom3, (1, 3, 6, 4, 5, 7), (0, 2, 3, 6, 5, 7))
    x278 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x278 += einsum(denom3, (0, 1, 2, 3, 4, 5), x275, (6, 0, 1, 5, 3, 7), (1, 2, 6, 0, 7, 4)) * -1.3333333333333333
    x278 += einsum(x277, (0, 1, 2, 3, 4, 5), (2, 3, 0, 1, 4, 5))
    x279 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x279 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x279 += einsum(t2, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x280 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x280 += einsum(x204, (0, 1, 2, 3), v.ovvv, (4, 3, 5, 2), (1, 0, 4, 2, 3, 5)) * -1.0
    x280 += einsum(x279, (0, 1, 2, 3), v.ovvv, (4, 2, 5, 3), (1, 0, 4, 2, 3, 5))
    x281 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x281 += einsum(v.ooov, (0, 1, 2, 3), t2, (4, 0, 3, 5), (4, 0, 1, 2, 3, 5))
    x281 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 0, 2), (1, 0, 5, 4, 2, 3)) * 5.0
    x282 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x282 += einsum(denom3, (0, 1, 2, 3, 4, 5), x281, (6, 0, 7, 1, 3, 4), (6, 7, 1, 2, 4, 5))
    x283 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x283 += einsum(x280, (0, 1, 2, 3, 4, 5), denom3, (1, 2, 6, 4, 7, 3), (2, 6, 0, 1, 5, 7)) * 2.0
    x283 += einsum(x282, (0, 1, 2, 3, 4, 5), (2, 3, 0, 1, 4, 5))
    x284 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x284 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.5
    x284 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x285 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x285 += einsum(x189, (0, 1, 2, 3), v.ovvv, (1, 3, 4, 5), (1, 0, 3, 2, 5, 4))
    x285 += einsum(x284, (0, 1, 2, 3), v.ovvv, (0, 4, 5, 3), (0, 1, 3, 2, 5, 4))
    x286 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x286 += einsum(x285, (0, 1, 2, 3, 4, 5), denom3, (0, 6, 7, 2, 3, 5), (6, 7, 1, 3, 5, 4))
    x287 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x287 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x287 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x288 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x288 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x288 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 3.0
    x289 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x289 += einsum(x180, (0, 1, 2, 3), denom3, (1, 4, 5, 2, 3, 6), v.ovvv, (1, 2, 6, 7), (4, 5, 0, 3, 6, 7))
    x290 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x290 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x290 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x291 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x291 += einsum(denom3, (0, 1, 2, 3, 4, 5), x180, (6, 0, 4, 3), v.ovvv, (0, 3, 5, 7), (1, 2, 6, 4, 5, 7))
    x292 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x292 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.75
    x292 += einsum(v.ovvv, (0, 1, 2, 3), (0, 3, 2, 1))
    x293 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x293 += einsum(v.ovvv, (0, 1, 2, 3), t2, (4, 0, 2, 1), (0, 4, 2, 1, 3))
    x293 += einsum(t2, (0, 1, 2, 3), x292, (0, 3, 4, 2), (0, 1, 2, 3, 4)) * 2.0
    x294 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x294 += einsum(denom3, (0, 1, 2, 3, 4, 5), x293, (0, 6, 5, 3, 7), (1, 2, 6, 4, 7)) * 0.6666666666666666
    x295 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x295 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (6, 1, 5, 3), v.ovvv, (2, 5, 3, 7), (2, 0, 6, 1, 7, 4)) * -0.6666666666666666
    x295 += einsum(x277, (0, 1, 2, 3, 4, 5), (2, 3, 0, 1, 4, 5))
    x296 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x296 += einsum(denom3, (0, 1, 2, 3, 4, 5), t2, (1, 6, 5, 3), v.ovvv, (2, 5, 3, 7), (2, 0, 6, 1, 7, 4)) * 2.0
    x296 += einsum(x282, (0, 1, 2, 3, 4, 5), (2, 3, 0, 1, 4, 5))
    x297 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x297 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x297 += einsum(v.ovvv, (0, 1, 2, 3), (0, 3, 2, 1)) * 2.0
    x298 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x298 += einsum(v.ovvv, (0, 1, 2, 3), t2, (0, 4, 5, 1), (0, 4, 1, 5, 3, 2))
    x298 += einsum(t2, (0, 1, 2, 3), x297, (0, 4, 5, 2), (0, 1, 2, 3, 5, 4)) * -1.0
    x299 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x299 += einsum(denom3, (0, 1, 2, 3, 4, 5), x298, (0, 6, 3, 4, 7, 5), (1, 2, 6, 4, 5, 7)) * 0.5
    x300 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x300 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x300 += einsum(t2, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x300 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2)) * 1.5
    x301 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x301 += einsum(x300, (0, 1, 2, 3), l2, (2, 3, 4, 1), (1, 4, 0, 2, 3)) * 2.0
    x301 += einsum(x182, (0, 1, 2, 3), l2, (3, 2, 1, 4), (1, 4, 0, 3, 2)) * 3.0
    x302 = np.zeros((nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x302 += einsum(x301, (0, 1, 2, 3, 4), denom3, (0, 5, 6, 3, 7, 4), (5, 6, 1, 2, 7))
    x303 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x303 += einsum(x254, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x303 += einsum(x254, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    x303 += einsum(x257, (0, 1, 2, 3), (0, 1, 2, 3))
    x303 += einsum(x257, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    x303 += einsum(x259, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x303 += einsum(x259, (0, 1, 2, 3), (1, 0, 2, 3)) * 3.0
    x303 += einsum(x261, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x303 += einsum(x261, (0, 1, 2, 3), (1, 0, 2, 3))
    x303 += einsum(v.ovvv, (0, 1, 2, 3), x264, (4, 0, 5, 3, 1, 2), (0, 4, 5, 1)) * 2.0
    x303 += einsum(v.ovvv, (0, 1, 2, 3), x266, (4, 0, 5, 1, 3, 2), (0, 4, 5, 3)) * -2.0
    x303 += einsum(x270, (0, 1, 2, 3, 4, 5), x267, (1, 3, 5, 4), (1, 0, 2, 5)) * -1.0
    x303 += einsum(x214, (0, 1, 2, 3), x272, (4, 0, 5, 1, 3, 2), (0, 4, 5, 2)) * -1.0
    x303 += einsum(x273, (0, 1, 2, 3, 4, 5), x240, (0, 1, 5, 3), (0, 1, 2, 4))
    x303 += einsum(x189, (0, 1, 2, 3), x220, (4, 1, 5, 0, 3, 2), (0, 5, 4, 2)) * 2.0
    x303 += einsum(x278, (0, 1, 2, 3, 4, 5), l2, (5, 4, 1, 3), (1, 0, 2, 5)) * -3.0
    x303 += einsum(x283, (0, 1, 2, 3, 4, 5), l2, (5, 4, 3, 1), (1, 0, 2, 5))
    x303 += einsum(x286, (0, 1, 2, 3, 4, 5), l2, (3, 5, 1, 0), (0, 1, 2, 4)) * 4.0
    x303 += einsum(x227, (0, 1, 2, 3, 4), x190, (1, 2, 3, 4), (1, 2, 0, 4)) * -2.0
    x303 += einsum(x230, (0, 1, 2, 3, 4), x287, (1, 2, 3, 4), (1, 2, 0, 4))
    x303 += einsum(x231, (0, 1, 2, 3, 4, 5), x225, (0, 3, 4, 5), (3, 2, 1, 5)) * 4.0
    x303 += einsum(x288, (0, 1, 2, 3), x289, (0, 1, 4, 3, 5, 2), (0, 1, 4, 5))
    x303 += einsum(x291, (0, 1, 2, 3, 4, 5), x290, (0, 1, 5, 3), (0, 1, 2, 4)) * -2.0
    x303 += einsum(x294, (0, 1, 2, 3, 4), l2, (3, 4, 1, 0), (0, 1, 2, 3)) * -6.0
    x303 += einsum(x295, (0, 1, 2, 3, 4, 5), l2, (4, 5, 3, 1), (1, 0, 2, 5)) * -3.0
    x303 += einsum(x296, (0, 1, 2, 3, 4, 5), l2, (4, 5, 1, 3), (1, 0, 2, 5))
    x303 += einsum(x299, (0, 1, 2, 3, 4, 5), l2, (5, 3, 1, 0), (0, 1, 2, 4)) * 4.0
    x303 += einsum(v.ooov, (0, 1, 2, 3), x302, (1, 2, 0, 4, 3), (1, 2, 4, 3)) * 2.0
    e_pert += einsum(x303, (0, 1, 2, 3), v.ooov, (0, 2, 1, 3), ()) * 0.5
    x304 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x304 += einsum(v.ovvv, (0, 1, 2, 3), denom3, (0, 4, 5, 2, 6, 1), v.ovvv, (0, 1, 2, 7), (5, 4, 6, 3, 7))
    x305 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x305 += einsum(denom3, (0, 1, 2, 3, 4, 5), v.ovvv, (0, 5, 6, 3), x243, (0, 3, 7, 5), (1, 2, 4, 7, 6)) * 2.0
    x306 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x306 += einsum(l2, (0, 1, 2, 3), x37, (2, 4, 3, 0, 5, 1), (4, 3, 0, 5))
    x306 += einsum(l2, (0, 1, 2, 3), x304, (3, 2, 1, 0, 4), (2, 3, 1, 4))
    x306 += einsum(x35, (0, 1, 2, 3, 4, 5), l2, (5, 3, 0, 2), (1, 2, 3, 4))
    x306 += einsum(l2, (0, 1, 2, 3), x305, (2, 3, 0, 1, 4), (2, 3, 0, 4)) * -1.0
    e_pert += einsum(x186, (0, 1, 2, 3), x306, (1, 0, 2, 3), ())
    x307 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x307 += einsum(v.ovvv, (0, 1, 2, 3), denom3, (0, 4, 5, 2, 6, 1), v.ovvv, (0, 2, 1, 7), (5, 4, 6, 3, 7))
    x308 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x308 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * 0.5
    x308 += einsum(l2, (0, 1, 2, 3), (3, 2, 1, 0))
    x309 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x309 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x309 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.5
    x310 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x310 += einsum(x307, (0, 1, 2, 3, 4), l2, (3, 2, 1, 0), (0, 1, 2, 4)) * 0.5
    x310 += einsum(x308, (0, 1, 2, 3), x37, (0, 4, 1, 3, 5, 2), (1, 4, 3, 5))
    x310 += einsum(x35, (0, 1, 2, 3, 4, 5), x309, (2, 0, 3, 5), (2, 1, 3, 4)) * -1.0
    e_pert += einsum(x310, (0, 1, 2, 3), t2, (1, 0, 3, 2), ()) * -2.0
    x311 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x311 += einsum(l2, (0, 1, 2, 3), x307, (3, 2, 0, 1, 4), (2, 3, 0, 4))
    x311 += einsum(x36, (0, 1, 2, 3), (1, 0, 2, 3))
    x312 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x312 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x312 += einsum(t2, (0, 1, 2, 3), (1, 0, 3, 2))
    e_pert += einsum(x312, (0, 1, 2, 3), x311, (0, 1, 2, 3), ()) * -2.0
    x313 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x313 += einsum(l2, (0, 1, 2, 3), x304, (3, 2, 0, 1, 4), (2, 3, 0, 4))
    x314 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x314 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x314 += einsum(l2, (0, 1, 2, 3), (2, 3, 1, 0))
    x315 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x315 += einsum(x313, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x315 += einsum(x37, (0, 1, 2, 3, 4, 5), x245, (2, 1, 3, 4), (0, 2, 3, 5)) * -1.0
    x315 += einsum(x314, (0, 1, 2, 3), x35, (4, 0, 1, 2, 3, 5), (4, 1, 2, 5)) * -1.0
    e_pert += einsum(x315, (0, 1, 2, 3), t2, (1, 0, 3, 2), ())
    x316 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x316 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x316 += einsum(l2, (0, 1, 2, 3), (2, 3, 1, 0))
    x316 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * 2.0
    x317 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x317 += einsum(x313, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.5
    x317 += einsum(x39, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    x317 += einsum(x316, (0, 1, 2, 3), x37, (4, 1, 0, 2, 3, 5), (4, 0, 2, 5)) * -0.5
    e_pert += einsum(x317, (0, 1, 2, 3), t2, (0, 1, 2, 3), ()) * 2.0

    return e_pert

