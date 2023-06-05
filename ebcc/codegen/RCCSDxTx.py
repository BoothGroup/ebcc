# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(f.ov, (0, 1), t1, (0, 1), ()) * 2.0
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x1 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    e_cc += einsum(x0, (0, 1, 2, 3), x1, (0, 1, 3, 2), ()) * 2.0
    del x0, x1

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=np.float64)
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2new += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (4, 0, 5, 2)) * 2.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (4, 0, 3, 5)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (4, 0, 5, 2)) * -1.0
    t2new += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (3, 2, 4, 1)) * -1.0
    x0 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x0 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    t1new += einsum(t2, (0, 1, 2, 3), x0, (1, 3, 2, 4), (0, 4)) * 2.0
    del x0
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x1 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x2 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x2 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x2 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x2 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new += einsum(t2, (0, 1, 2, 3), x2, (4, 0, 1, 2), (4, 3)) * -1.0
    del x2
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x3 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x3 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x4 = np.zeros((nocc, nvir), dtype=np.float64)
    x4 += einsum(t1, (0, 1), x3, (0, 2, 3, 1), (2, 3)) * 2.0
    x5 = np.zeros((nocc, nvir), dtype=np.float64)
    x5 += einsum(f.ov, (0, 1), (0, 1))
    x5 += einsum(x4, (0, 1), (0, 1))
    del x4
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t1new += einsum(x5, (0, 1), x6, (0, 2, 3, 1), (2, 3))
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x7 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x7 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t1new += einsum(t1, (0, 1), x7, (0, 2, 1, 3), (2, 3)) * 2.0
    del x7
    x8 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x8 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x8 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x9 = np.zeros((nvir, nvir), dtype=np.float64)
    x9 += einsum(f.vv, (0, 1), (0, 1)) * 0.5
    x9 += einsum(t1, (0, 1), x8, (0, 1, 2, 3), (3, 2))
    del x8
    t1new += einsum(t1, (0, 1), x9, (1, 2), (0, 2)) * 2.0
    del x9
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x10 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x10 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x11 = np.zeros((nocc, nocc), dtype=np.float64)
    x11 += einsum(t1, (0, 1), x5, (2, 1), (2, 0))
    del x5
    x12 = np.zeros((nocc, nocc), dtype=np.float64)
    x12 += einsum(f.oo, (0, 1), (0, 1))
    x12 += einsum(t2, (0, 1, 2, 3), x3, (1, 4, 2, 3), (4, 0)) * 2.0
    x12 += einsum(t1, (0, 1), x10, (2, 3, 0, 1), (3, 2)) * 2.0
    x12 += einsum(x11, (0, 1), (0, 1))
    t1new += einsum(t1, (0, 1), x12, (0, 2), (2, 1)) * -1.0
    del x12
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x13 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x14 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x15 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x16 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x16 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x17 += einsum(t2, (0, 1, 2, 3), x16, (4, 5, 1, 0), (4, 5, 3, 2))
    del x16
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x18 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x19 += einsum(t2, (0, 1, 2, 3), x18, (4, 1, 5, 3), (4, 0, 2, 5))
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x20 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x20 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x21 += einsum(x15, (0, 1, 2, 3), x20, (1, 4, 5, 2), (4, 0, 5, 3)) * 2.0
    del x20
    x22 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x22 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x23 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x23 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x24 += einsum(t2, (0, 1, 2, 3), x1, (4, 5, 1, 2), (4, 0, 5, 3))
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x25 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 0.5
    x25 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x25 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x26 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x26 += einsum(v.ooov, (0, 1, 2, 3), x25, (2, 4, 5, 3), (0, 1, 4, 5)) * 2.0
    del x25
    x27 = np.zeros((nocc, nvir), dtype=np.float64)
    x27 += einsum(t1, (0, 1), x3, (0, 2, 3, 1), (2, 3))
    x28 = np.zeros((nocc, nvir), dtype=np.float64)
    x28 += einsum(f.ov, (0, 1), (0, 1)) * 0.5
    x28 += einsum(x27, (0, 1), (0, 1))
    del x27
    x29 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x29 += einsum(x28, (0, 1), t2, (2, 3, 1, 4), (0, 2, 3, 4)) * 2.0
    del x28
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x30 += einsum(x22, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    del x22
    x30 += einsum(x23, (0, 1, 2, 3), (2, 0, 1, 3))
    del x23
    x30 += einsum(x24, (0, 1, 2, 3), (2, 0, 1, 3))
    del x24
    x30 += einsum(x26, (0, 1, 2, 3), (1, 2, 0, 3))
    del x26
    x30 += einsum(x29, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x29
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x31 += einsum(t1, (0, 1), x30, (0, 2, 3, 4), (2, 3, 4, 1))
    del x30
    x32 = np.zeros((nocc, nocc), dtype=np.float64)
    x32 += einsum(f.oo, (0, 1), (0, 1))
    x32 += einsum(x11, (0, 1), (1, 0))
    del x11
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x33 += einsum(x32, (0, 1), t2, (2, 1, 3, 4), (0, 2, 4, 3))
    del x32
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x34 += einsum(x14, (0, 1, 2, 3), (0, 1, 2, 3))
    del x14
    x34 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    del x15
    x34 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3))
    del x17
    x34 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x19
    x34 += einsum(x21, (0, 1, 2, 3), (1, 0, 2, 3))
    del x21
    x34 += einsum(x31, (0, 1, 2, 3), (0, 1, 3, 2))
    del x31
    x34 += einsum(x33, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x33
    t2new += einsum(x34, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new += einsum(x34, (0, 1, 2, 3), (1, 0, 2, 3))
    del x34
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x35 += einsum(t2, (0, 1, 2, 3), x18, (4, 1, 5, 2), (4, 0, 3, 5))
    del x18
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x36 += einsum(t1, (0, 1), x1, (2, 3, 0, 4), (2, 3, 1, 4))
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x37 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x37 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x38 += einsum(t2, (0, 1, 2, 3), x37, (1, 4, 3, 5), (4, 0, 5, 2)) * 0.5
    del x37
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x39 += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x36
    x39 += einsum(x38, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x38
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x40 += einsum(t2, (0, 1, 2, 3), x39, (4, 1, 5, 2), (4, 0, 5, 3)) * 2.0
    del x39
    x41 = np.zeros((nvir, nvir), dtype=np.float64)
    x41 += einsum(t2, (0, 1, 2, 3), x3, (0, 1, 3, 4), (4, 2)) * 2.0
    x42 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x42 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x42 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x43 = np.zeros((nvir, nvir), dtype=np.float64)
    x43 += einsum(t1, (0, 1), x42, (0, 1, 2, 3), (2, 3))
    del x42
    x44 = np.zeros((nvir, nvir), dtype=np.float64)
    x44 += einsum(x41, (0, 1), (0, 1))
    del x41
    x44 += einsum(x43, (0, 1), (1, 0)) * -1.0
    del x43
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x45 += einsum(x44, (0, 1), t2, (2, 3, 0, 4), (2, 3, 1, 4))
    del x44
    x46 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x46 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 2), (0, 4, 5, 3))
    x47 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x47 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x47 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3))
    x48 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x48 += einsum(t2, (0, 1, 2, 3), x47, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    del x47
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x49 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x49 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    t2new += einsum(v.vvvv, (0, 1, 2, 3), x49, (4, 5, 3, 1), (5, 4, 0, 2))
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x50 += einsum(v.ovvv, (0, 1, 2, 3), x49, (4, 5, 3, 1), (0, 4, 5, 2))
    x51 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x51 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    del x1
    x51 += einsum(x46, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x46
    x51 += einsum(x48, (0, 1, 2, 3), (0, 2, 1, 3))
    del x48
    x51 += einsum(x50, (0, 1, 2, 3), (2, 1, 0, 3))
    del x50
    x52 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x52 += einsum(t1, (0, 1), x51, (2, 3, 0, 4), (2, 3, 4, 1))
    del x51
    x53 = np.zeros((nocc, nocc), dtype=np.float64)
    x53 += einsum(t2, (0, 1, 2, 3), x3, (1, 4, 2, 3), (4, 0))
    del x3
    x54 = np.zeros((nocc, nocc), dtype=np.float64)
    x54 += einsum(t1, (0, 1), x10, (2, 3, 0, 1), (2, 3))
    del x10
    x55 = np.zeros((nocc, nocc), dtype=np.float64)
    x55 += einsum(x53, (0, 1), (0, 1))
    del x53
    x55 += einsum(x54, (0, 1), (1, 0))
    del x54
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x56 += einsum(x55, (0, 1), t2, (2, 0, 3, 4), (1, 2, 4, 3)) * 2.0
    del x55
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x57 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    del x35
    x57 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x40
    x57 += einsum(x45, (0, 1, 2, 3), (1, 0, 3, 2))
    del x45
    x57 += einsum(x52, (0, 1, 2, 3), (0, 1, 3, 2))
    del x52
    x57 += einsum(x56, (0, 1, 2, 3), (1, 0, 3, 2))
    del x56
    t2new += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x57, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x57
    x58 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x58 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x58 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x59 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x59 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x59 += einsum(t2, (0, 1, 2, 3), x58, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x58
    t2new += einsum(t2, (0, 1, 2, 3), x59, (4, 1, 5, 3), (0, 4, 2, 5)) * 2.0
    del x59
    x60 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x60 += einsum(v.ovov, (0, 1, 2, 3), x49, (4, 5, 1, 3), (4, 5, 0, 2))
    del x49
    x61 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x61 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x61 += einsum(x60, (0, 1, 2, 3), (3, 1, 0, 2))
    del x60
    t2new += einsum(t2, (0, 1, 2, 3), x61, (0, 4, 5, 1), (5, 4, 3, 2))
    x62 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x62 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x62 += einsum(t1, (0, 1), x61, (0, 2, 3, 4), (3, 2, 4, 1))
    del x61
    t2new += einsum(t1, (0, 1), x62, (2, 3, 0, 4), (2, 3, 1, 4))
    del x62
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x63 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x63 += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3))
    del x13
    t2new += einsum(t2, (0, 1, 2, 3), x63, (4, 1, 5, 2), (4, 0, 5, 3))
    del x63
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x64 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x64 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x64, (4, 1, 5, 2), (0, 4, 5, 3))
    del x64

    return {"t1new": t1new, "t2new": t2new}

def energy_perturbative(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    e_ia = direct_sum("i-a->ia", np.diag(f.oo), np.diag(f.vv))
    denom3 = 1 / direct_sum("ia+jb+kc->ijkabc", e_ia, e_ia, e_ia)

    # energy
    x0 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x0 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 5, 1), denom3, (6, 4, 3, 7, 0, 1), (4, 6, 2, 0, 7, 5))
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x1 += einsum(v.ovvv, (0, 1, 2, 3), x0, (4, 0, 5, 1, 2, 3), (4, 0, 5, 2))
    e_pert = 0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x1, (0, 2, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x1, (2, 1, 0, 3), ()) * -8.0
    del x1
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x2 += einsum(v.ovvv, (0, 1, 2, 3), x0, (4, 0, 5, 3, 1, 2), (4, 0, 5, 1))
    del x0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x2, (2, 1, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x2, (0, 2, 1, 3), ()) * -8.0
    del x2
    x3 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x3 += einsum(l2, (0, 1, 2, 3), v.ovvv, (4, 1, 5, 0), denom3, (6, 3, 4, 0, 7, 1), (3, 4, 6, 2, 7, 5))
    x4 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x4 += einsum(t2, (0, 1, 2, 3), x3, (1, 4, 0, 5, 3, 2), (0, 4, 5, 3))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x4, (1, 2, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x4, (2, 0, 1, 3), ()) * -2.0
    del x4
    x5 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x5 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 1, 5), denom3, (6, 4, 3, 7, 0, 1), (4, 6, 2, 0, 7, 5))
    x6 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x6 += einsum(v.ovvv, (0, 1, 2, 3), x5, (4, 0, 5, 2, 1, 3), (4, 0, 5, 1))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x6, (0, 2, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x6, (2, 1, 0, 3), ()) * -2.0
    del x6
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x7 += einsum(v.ovvv, (0, 1, 2, 3), x5, (4, 0, 5, 1, 2, 3), (4, 0, 5, 2))
    del x5
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x7, (2, 1, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x7, (0, 2, 1, 3), ()) * -2.0
    del x7
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x8 += einsum(t2, (0, 1, 2, 3), x3, (1, 4, 0, 5, 2, 3), (0, 4, 5, 2))
    del x3
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x8, (2, 0, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x8, (1, 2, 0, 3), ()) * -2.0
    del x8
    x9 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x9 += einsum(l2, (0, 1, 2, 3), v.ovvv, (2, 1, 4, 5), denom3, (2, 6, 7, 1, 0, 4), (7, 6, 3, 0, 4, 5))
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x10 += einsum(t2, (0, 1, 2, 3), x9, (0, 1, 4, 2, 5, 3), (0, 1, 4, 5))
    del x9
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x10, (1, 2, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x10, (2, 1, 0, 3), ()) * -2.0
    del x10
    x11 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x11 += einsum(l2, (0, 1, 2, 3), v.ovvv, (3, 1, 4, 0), denom3, (3, 5, 6, 0, 7, 1), (6, 5, 2, 7, 4))
    x12 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x12 += einsum(t2, (0, 1, 2, 3), x11, (0, 1, 4, 2, 3), (0, 1, 4, 2))
    del x11
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x12, (1, 2, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x12, (2, 1, 0, 3), ()) * -8.0
    del x12
    x13 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x13 += einsum(l2, (0, 1, 2, 3), v.ovvv, (2, 4, 5, 1), denom3, (2, 6, 7, 1, 0, 4), (7, 6, 3, 0, 4, 5))
    x14 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x14 += einsum(t2, (0, 1, 2, 3), x13, (1, 0, 4, 2, 5, 3), (0, 1, 4, 5))
    del x13
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x14, (2, 1, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x14, (1, 2, 0, 3), ()) * -2.0
    del x14
    x15 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x15 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 5, 1), denom3, (6, 4, 2, 7, 0, 1), (4, 6, 3, 0, 7, 5))
    x16 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x16 += einsum(v.ovvv, (0, 1, 2, 3), x15, (4, 0, 5, 3, 1, 2), (4, 0, 5, 1))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x16, (0, 2, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x16, (2, 1, 0, 3), ()) * -2.0
    del x16
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x17 += einsum(v.ovvv, (0, 1, 2, 3), x15, (4, 0, 5, 1, 2, 3), (4, 0, 5, 2))
    del x15
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x17, (2, 1, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x17, (0, 2, 1, 3), ()) * -2.0
    del x17
    x18 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x18 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), denom3, (6, 4, 2, 7, 0, 1), (4, 6, 3, 0, 7, 5))
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x19 += einsum(v.ovvv, (0, 1, 2, 3), x18, (4, 0, 5, 1, 2, 3), (4, 0, 5, 2))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x19, (0, 2, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x19, (2, 1, 0, 3), ()) * -2.0
    del x19
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x20 += einsum(v.ovvv, (0, 1, 2, 3), x18, (4, 0, 5, 3, 1, 2), (4, 0, 5, 1))
    del x18
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x20, (2, 1, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x20, (0, 2, 1, 3), ()) * -8.0
    del x20
    x21 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x21 += einsum(l2, (0, 1, 2, 3), v.ovvv, (3, 4, 5, 1), denom3, (3, 6, 7, 1, 0, 4), (7, 6, 2, 0, 4, 5))
    x22 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x22 += einsum(t2, (0, 1, 2, 3), x21, (0, 1, 4, 2, 5, 3), (0, 1, 4, 5))
    del x21
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x22, (1, 2, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x22, (2, 1, 0, 3), ()) * -2.0
    del x22
    x23 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x23 += einsum(l2, (0, 1, 2, 3), v.ovvv, (3, 1, 4, 5), denom3, (3, 6, 7, 1, 0, 4), (7, 6, 2, 0, 4, 5))
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x24 += einsum(t2, (0, 1, 2, 3), x23, (1, 0, 4, 2, 5, 3), (0, 1, 4, 5))
    del x23
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x24, (2, 1, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x24, (1, 2, 0, 3), ()) * -8.0
    del x24
    x25 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x25 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 5, 1), denom3, (6, 2, 3, 7, 5, 1), (2, 6, 4, 5, 7, 0))
    x26 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x26 += einsum(v.ovvv, (0, 1, 2, 3), x25, (4, 0, 5, 1, 2, 3), (4, 0, 5, 2))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x26, (1, 2, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x26, (2, 1, 0, 3), ()) * -8.0
    del x26
    x27 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x27 += einsum(v.ovvv, (0, 1, 2, 3), x25, (4, 0, 5, 2, 1, 3), (4, 0, 5, 1))
    del x25
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x27, (2, 1, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x27, (1, 2, 0, 3), ()) * -8.0
    del x27
    x28 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x28 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 1, 5), denom3, (6, 2, 3, 7, 5, 1), (2, 6, 4, 5, 7, 0))
    x29 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x29 += einsum(v.ovvv, (0, 1, 2, 3), x28, (4, 0, 5, 2, 1, 3), (4, 0, 5, 1))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x29, (1, 2, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x29, (2, 1, 0, 3), ()) * -2.0
    del x29
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x30 += einsum(v.ovvv, (0, 1, 2, 3), x28, (4, 0, 5, 1, 2, 3), (4, 0, 5, 2))
    del x28
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x30, (2, 1, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x30, (1, 2, 0, 3), ()) * -2.0
    del x30
    x31 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x31 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 2, 5, 3), denom3, (6, 0, 4, 2, 7, 3), (0, 4, 6, 1, 7, 5))
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x32 += einsum(l2, (0, 1, 2, 3), x31, (2, 4, 3, 5, 0, 1), (3, 4, 5, 0))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x32, (1, 2, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x32, (2, 0, 1, 3), ()) * -2.0
    del x32
    x33 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x33 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 3, 5, 2), denom3, (6, 0, 4, 3, 7, 2), (0, 4, 6, 1, 7, 5))
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x34 += einsum(l2, (0, 1, 2, 3), x33, (2, 4, 3, 5, 0, 1), (3, 4, 5, 0))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x34, (2, 0, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x34, (1, 2, 0, 3), ()) * -2.0
    del x34
    x35 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x35 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 2, 4, 5), denom3, (1, 6, 7, 2, 3, 4), (7, 6, 0, 3, 4, 5))
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x36 += einsum(l2, (0, 1, 2, 3), x35, (2, 3, 4, 1, 5, 0), (3, 2, 4, 5))
    del x35
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x36, (0, 2, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x36, (2, 0, 1, 3), ()) * -2.0
    del x36
    x37 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x37 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 3), denom3, (1, 6, 7, 3, 2, 4), (7, 6, 0, 2, 4, 5))
    x38 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x38 += einsum(l2, (0, 1, 2, 3), x37, (2, 3, 4, 1, 5, 0), (3, 2, 4, 5))
    del x37
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x38, (0, 2, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x38, (2, 0, 1, 3), ()) * -2.0
    del x38
    x39 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x39 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 3, 4, 2), denom3, (1, 5, 6, 3, 7, 2), (6, 5, 0, 7, 4))
    x40 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x40 += einsum(l2, (0, 1, 2, 3), x39, (3, 2, 4, 0, 1), (2, 3, 4, 0))
    del x39
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x40, (0, 2, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x40, (2, 0, 1, 3), ()) * -8.0
    del x40
    x41 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x41 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 5, 1), denom3, (6, 3, 2, 7, 5, 1), (3, 6, 4, 5, 7, 0))
    x42 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x42 += einsum(v.ovvv, (0, 1, 2, 3), x41, (4, 0, 5, 2, 1, 3), (4, 0, 5, 1))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x42, (1, 2, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x42, (2, 1, 0, 3), ()) * -2.0
    del x42
    x43 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x43 += einsum(v.ovvv, (0, 1, 2, 3), x41, (4, 0, 5, 1, 2, 3), (4, 0, 5, 2))
    del x41
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x43, (2, 1, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x43, (1, 2, 0, 3), ()) * -2.0
    del x43
    x44 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x44 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), denom3, (6, 3, 2, 7, 5, 1), (3, 6, 4, 5, 7, 0))
    x45 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x45 += einsum(v.ovvv, (0, 1, 2, 3), x44, (4, 0, 5, 1, 2, 3), (4, 0, 5, 2))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x45, (1, 2, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x45, (2, 1, 0, 3), ()) * -2.0
    del x45
    x46 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x46 += einsum(v.ovvv, (0, 1, 2, 3), x44, (4, 0, 5, 2, 1, 3), (4, 0, 5, 1))
    del x44
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x46, (2, 1, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x46, (1, 2, 0, 3), ()) * -8.0
    del x46
    x47 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x47 += einsum(l2, (0, 1, 2, 3), x31, (3, 4, 2, 5, 0, 1), (2, 4, 5, 0))
    del x31
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x47, (2, 0, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x47, (1, 2, 0, 3), ()) * -2.0
    del x47
    x48 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x48 += einsum(l2, (0, 1, 2, 3), x33, (3, 4, 2, 5, 0, 1), (2, 4, 5, 0))
    del x33
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x48, (1, 2, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x48, (2, 0, 1, 3), ()) * -8.0
    del x48
    x49 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x49 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 2), denom3, (1, 6, 7, 2, 3, 4), (7, 6, 0, 3, 4, 5))
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x50 += einsum(l2, (0, 1, 2, 3), x49, (2, 3, 4, 0, 5, 1), (2, 3, 4, 5))
    del x49
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x50, (2, 0, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x50, (0, 2, 1, 3), ()) * -2.0
    del x50
    x51 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x51 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 3, 4, 5), denom3, (1, 6, 7, 3, 2, 4), (7, 6, 0, 2, 4, 5))
    x52 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x52 += einsum(l2, (0, 1, 2, 3), x51, (2, 3, 4, 0, 5, 1), (2, 3, 4, 5))
    del x51
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x52, (2, 0, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x52, (0, 2, 1, 3), ()) * -8.0
    del x52
    x53 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x53 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 2, 4, 3), denom3, (1, 5, 6, 2, 7, 3), (6, 5, 0, 7, 4))
    x54 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x54 += einsum(l2, (0, 1, 2, 3), x53, (2, 3, 4, 0, 1), (2, 3, 4, 0))
    del x53
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x54, (2, 0, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x54, (0, 2, 1, 3), ()) * -2.0
    del x54
    x55 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x55 += einsum(l2, (0, 1, 2, 3), v.ovvv, (4, 1, 5, 0), denom3, (6, 2, 4, 0, 7, 1), (2, 4, 6, 3, 7, 5))
    x56 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x56 += einsum(t2, (0, 1, 2, 3), x55, (1, 4, 0, 5, 3, 2), (0, 4, 5, 3))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x56, (2, 0, 1, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x56, (1, 2, 0, 3), ()) * -2.0
    del x56
    x57 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x57 += einsum(t2, (0, 1, 2, 3), x55, (1, 4, 0, 5, 2, 3), (0, 4, 5, 2))
    del x55
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x57, (1, 2, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x57, (2, 0, 1, 3), ()) * -8.0
    del x57
    x58 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x58 += einsum(l2, (0, 1, 2, 3), v.ovvv, (2, 1, 4, 0), denom3, (2, 5, 6, 0, 7, 1), (6, 5, 3, 7, 4))
    x59 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x59 += einsum(t2, (0, 1, 2, 3), x58, (1, 0, 4, 2, 3), (0, 1, 4, 2))
    del x58
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x59, (2, 1, 0, 3), ()) * 4.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x59, (1, 2, 0, 3), ()) * -2.0
    del x59
    x60 = np.zeros((nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x60 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 0, 1), denom3, (2, 5, 6, 1, 7, 0), (6, 5, 3, 4, 7))
    x61 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x61 += einsum(v.ooov, (0, 1, 2, 3), x60, (0, 2, 1, 4, 3), (0, 2, 4, 3))
    del x60
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x61, (2, 0, 1, 3), ()) * 2.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x61, (1, 2, 0, 3), ()) * -4.0
    del x61
    x62 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x62 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 1, 5), denom3, (3, 6, 7, 1, 0, 5), (7, 6, 2, 4, 0, 5))
    x63 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x63 += einsum(v.ooov, (0, 1, 2, 3), x62, (0, 2, 4, 1, 3, 5), (0, 2, 4, 5))
    del x62
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x63, (1, 2, 0, 3), ()) * 2.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x63, (2, 1, 0, 3), ()) * -4.0
    del x63
    x64 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x64 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 1, 0), denom3, (6, 3, 4, 0, 7, 1), (3, 4, 6, 2, 5, 7))
    x65 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x65 += einsum(v.ooov, (0, 1, 2, 3), x64, (1, 4, 2, 5, 0, 3), (4, 2, 5, 3))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x65, (2, 1, 0, 3), ()) * 2.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x65, (0, 2, 1, 3), ()) * -4.0
    del x65
    x66 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x66 += einsum(v.ooov, (0, 1, 2, 3), x64, (2, 4, 0, 5, 1, 3), (4, 0, 5, 3))
    del x64
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x66, (0, 2, 1, 3), ()) * 2.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x66, (2, 1, 0, 3), ()) * -4.0
    del x66
    x67 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x67 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 2), denom3, (1, 5, 6, 2, 3, 7), (5, 6, 0, 4, 3, 7))
    x68 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x68 += einsum(l2, (0, 1, 2, 3), x67, (4, 3, 5, 2, 1, 0), (3, 4, 5, 0))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x68, (1, 2, 0, 3), ()) * 2.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x68, (2, 0, 1, 3), ()) * -4.0
    del x68
    x69 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x69 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 1, 2), denom3, (1, 4, 6, 2, 3, 7), (4, 6, 0, 5, 3, 7))
    x70 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x70 += einsum(l2, (0, 1, 2, 3), x69, (4, 3, 5, 2, 1, 0), (3, 4, 5, 0))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x70, (2, 0, 1, 3), ()) * 2.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x70, (1, 2, 0, 3), ()) * -4.0
    del x70
    x71 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x71 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), denom3, (6, 3, 4, 0, 7, 1), (3, 4, 6, 2, 5, 7))
    x72 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x72 += einsum(v.ooov, (0, 1, 2, 3), x71, (2, 4, 0, 5, 1, 3), (4, 0, 5, 3))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x72, (2, 1, 0, 3), ()) * 2.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x72, (0, 2, 1, 3), ()) * -4.0
    del x72
    x73 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x73 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), denom3, (1, 5, 6, 3, 2, 7), (5, 6, 0, 4, 2, 7))
    x74 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x74 += einsum(l2, (0, 1, 2, 3), x73, (4, 3, 5, 2, 1, 0), (3, 4, 5, 0))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x74, (2, 0, 1, 3), ()) * 2.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x74, (1, 2, 0, 3), ()) * -4.0
    del x74
    x75 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x75 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 5, 1), denom3, (2, 6, 7, 1, 0, 5), (7, 6, 3, 4, 0, 5))
    x76 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x76 += einsum(v.ooov, (0, 1, 2, 3), x75, (0, 2, 4, 1, 3, 5), (0, 2, 4, 5))
    del x75
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x76, (1, 2, 0, 3), ()) * 2.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x76, (2, 1, 0, 3), ()) * -4.0
    del x76
    x77 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x77 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), denom3, (2, 6, 7, 1, 0, 5), (7, 6, 3, 4, 0, 5))
    x78 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x78 += einsum(v.ooov, (0, 1, 2, 3), x77, (2, 0, 4, 1, 3, 5), (0, 2, 4, 5))
    del x77
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x78, (2, 1, 0, 3), ()) * 2.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x78, (1, 2, 0, 3), ()) * -4.0
    del x78
    x79 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x79 += einsum(l2, (0, 1, 2, 3), x69, (4, 2, 5, 3, 1, 0), (2, 4, 5, 0))
    del x69
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x79, (1, 2, 0, 3), ()) * 2.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x79, (2, 0, 1, 3), ()) * -4.0
    del x79
    x80 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x80 += einsum(l2, (0, 1, 2, 3), x73, (4, 2, 5, 3, 1, 0), (2, 4, 5, 0))
    del x73
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x80, (1, 2, 0, 3), ()) * 2.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x80, (2, 0, 1, 3), ()) * -4.0
    del x80
    x81 = np.zeros((nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x81 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 0), denom3, (2, 5, 6, 1, 7, 0), (6, 5, 3, 4, 7))
    x82 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x82 += einsum(v.ooov, (0, 1, 2, 3), x81, (0, 2, 1, 4, 3), (0, 2, 4, 3))
    del x81
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x82, (1, 2, 0, 3), ()) * 8.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x82, (2, 0, 1, 3), ()) * -4.0
    del x82
    x83 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x83 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 5, 1), denom3, (3, 6, 7, 1, 0, 5), (7, 6, 2, 4, 0, 5))
    x84 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x84 += einsum(v.ooov, (0, 1, 2, 3), x83, (2, 0, 4, 1, 3, 5), (0, 2, 4, 5))
    del x83
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x84, (2, 1, 0, 3), ()) * 8.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x84, (1, 2, 0, 3), ()) * -4.0
    del x84
    x85 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x85 += einsum(v.ooov, (0, 1, 2, 3), x71, (1, 4, 2, 5, 0, 3), (4, 2, 5, 3))
    del x71
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x85, (0, 2, 1, 3), ()) * 8.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x85, (2, 1, 0, 3), ()) * -4.0
    del x85
    x86 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x86 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 1, 3), denom3, (1, 4, 6, 3, 2, 7), (4, 6, 0, 5, 2, 7))
    x87 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x87 += einsum(l2, (0, 1, 2, 3), x86, (4, 3, 5, 2, 1, 0), (3, 4, 5, 0))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x87, (1, 2, 0, 3), ()) * 8.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x87, (2, 0, 1, 3), ()) * -4.0
    del x87
    x88 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x88 += einsum(l2, (0, 1, 2, 3), x67, (4, 2, 5, 3, 1, 0), (2, 4, 5, 0))
    del x67
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x88, (2, 0, 1, 3), ()) * 8.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x88, (1, 2, 0, 3), ()) * -4.0
    del x88
    x89 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x89 += einsum(l2, (0, 1, 2, 3), x86, (4, 2, 5, 3, 1, 0), (2, 4, 5, 0))
    del x86
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x89, (2, 0, 1, 3), ()) * 8.0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x89, (1, 2, 0, 3), ()) * -4.0
    del x89
    x90 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x90 += einsum(v.ovov, (0, 1, 2, 3), v.ovvv, (2, 4, 5, 3), denom3, (2, 6, 0, 3, 4, 1), (0, 6, 1, 4, 5))
    x91 = np.zeros((nocc, nvir), dtype=np.float64)
    x91 += einsum(t2, (0, 1, 2, 3), x90, (1, 0, 2, 4, 3), (0, 4))
    e_pert += einsum(l1, (0, 1), x91, (1, 0), ()) * 2.0
    del x91
    x92 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x92 += einsum(v.ovov, (0, 1, 2, 3), v.ovvv, (2, 1, 4, 5), denom3, (2, 6, 0, 3, 4, 1), (0, 6, 4, 3, 5))
    x93 = np.zeros((nocc, nvir), dtype=np.float64)
    x93 += einsum(t2, (0, 1, 2, 3), x92, (1, 0, 4, 2, 3), (0, 4))
    e_pert += einsum(l1, (0, 1), x93, (1, 0), ()) * 2.0
    del x93
    x94 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x94 += einsum(v.ovov, (0, 1, 2, 3), v.ovvv, (2, 1, 4, 3), denom3, (2, 0, 5, 1, 6, 3), (0, 5, 6, 4))
    x95 = np.zeros((nocc, nvir), dtype=np.float64)
    x95 += einsum(t2, (0, 1, 2, 3), x94, (1, 0, 3, 2), (0, 3))
    e_pert += einsum(l1, (0, 1), x95, (1, 0), ()) * 2.0
    del x95
    x96 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x96 += einsum(v.ovov, (0, 1, 2, 3), v.ovvv, (2, 4, 5, 1), denom3, (2, 6, 0, 3, 4, 1), (0, 6, 3, 4, 5))
    x97 = np.zeros((nocc, nvir), dtype=np.float64)
    x97 += einsum(t2, (0, 1, 2, 3), x96, (1, 0, 3, 4, 2), (0, 4))
    e_pert += einsum(l1, (0, 1), x97, (1, 0), ()) * 2.0
    del x97
    x98 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x98 += einsum(v.ovov, (0, 1, 2, 3), v.ovvv, (4, 1, 5, 3), denom3, (2, 4, 0, 3, 6, 1), (0, 2, 4, 6, 5))
    x99 = np.zeros((nocc, nvir), dtype=np.float64)
    x99 += einsum(t2, (0, 1, 2, 3), x98, (0, 1, 4, 3, 2), (4, 3))
    e_pert += einsum(l1, (0, 1), x99, (1, 0), ()) * 2.0
    del x99
    x100 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x100 += einsum(t2, (0, 1, 2, 3), v.ovov, (0, 4, 1, 2), denom3, (0, 5, 1, 2, 4, 6), (5, 4, 6, 3))
    x101 = np.zeros((nocc, nvir), dtype=np.float64)
    x101 += einsum(v.ovvv, (0, 1, 2, 3), x100, (0, 1, 2, 3), (0, 2))
    e_pert += einsum(l1, (0, 1), x101, (1, 0), ()) * 2.0
    del x101
    x102 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x102 += einsum(v.ovov, (0, 1, 2, 3), v.ovvv, (2, 3, 4, 1), denom3, (2, 0, 5, 3, 6, 1), (0, 5, 6, 4))
    x103 = np.zeros((nocc, nvir), dtype=np.float64)
    x103 += einsum(t2, (0, 1, 2, 3), x102, (1, 0, 2, 3), (0, 2))
    e_pert += einsum(l1, (0, 1), x103, (1, 0), ()) * 8.0
    del x103
    x104 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x104 += einsum(v.ovov, (0, 1, 2, 3), v.ovvv, (2, 3, 4, 5), denom3, (2, 6, 0, 3, 4, 1), (0, 6, 4, 1, 5))
    x105 = np.zeros((nocc, nvir), dtype=np.float64)
    x105 += einsum(t2, (0, 1, 2, 3), x104, (1, 0, 4, 3, 2), (0, 4))
    e_pert += einsum(l1, (0, 1), x105, (1, 0), ()) * 8.0
    del x105
    x106 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x106 += einsum(t2, (0, 1, 2, 3), v.ovov, (0, 2, 1, 4), denom3, (0, 5, 1, 2, 4, 6), (5, 4, 6, 3))
    x107 = np.zeros((nocc, nvir), dtype=np.float64)
    x107 += einsum(v.ovvv, (0, 1, 2, 3), x106, (0, 2, 1, 3), (0, 1))
    e_pert += einsum(l1, (0, 1), x107, (1, 0), ()) * 8.0
    del x107
    x108 = np.zeros((nocc, nvir), dtype=np.float64)
    x108 += einsum(t2, (0, 1, 2, 3), x94, (1, 0, 2, 3), (0, 2))
    del x94
    e_pert += einsum(l1, (0, 1), x108, (1, 0), ()) * -4.0
    del x108
    x109 = np.zeros((nocc, nvir), dtype=np.float64)
    x109 += einsum(t2, (0, 1, 2, 3), x104, (1, 0, 4, 2, 3), (0, 4))
    del x104
    e_pert += einsum(l1, (0, 1), x109, (1, 0), ()) * -4.0
    del x109
    x110 = np.zeros((nocc, nvir), dtype=np.float64)
    x110 += einsum(t2, (0, 1, 2, 3), x96, (1, 0, 2, 4, 3), (0, 4))
    del x96
    e_pert += einsum(l1, (0, 1), x110, (1, 0), ()) * -4.0
    del x110
    x111 = np.zeros((nocc, nvir), dtype=np.float64)
    x111 += einsum(t2, (0, 1, 2, 3), x102, (1, 0, 3, 2), (0, 3))
    del x102
    e_pert += einsum(l1, (0, 1), x111, (1, 0), ()) * -4.0
    del x111
    x112 = np.zeros((nocc, nvir), dtype=np.float64)
    x112 += einsum(t2, (0, 1, 2, 3), x90, (1, 0, 3, 4, 2), (0, 4))
    del x90
    e_pert += einsum(l1, (0, 1), x112, (1, 0), ()) * -4.0
    del x112
    x113 = np.zeros((nocc, nvir), dtype=np.float64)
    x113 += einsum(t2, (0, 1, 2, 3), x92, (1, 0, 4, 3, 2), (0, 4))
    del x92
    e_pert += einsum(l1, (0, 1), x113, (1, 0), ()) * -4.0
    del x113
    x114 = np.zeros((nocc, nvir), dtype=np.float64)
    x114 += einsum(t2, (0, 1, 2, 3), x98, (0, 1, 4, 2, 3), (4, 2))
    del x98
    e_pert += einsum(l1, (0, 1), x114, (1, 0), ()) * -4.0
    del x114
    x115 = np.zeros((nocc, nvir), dtype=np.float64)
    x115 += einsum(v.ovvv, (0, 1, 2, 3), x100, (0, 2, 1, 3), (0, 1))
    del x100
    e_pert += einsum(l1, (0, 1), x115, (1, 0), ()) * -4.0
    del x115
    x116 = np.zeros((nocc, nvir), dtype=np.float64)
    x116 += einsum(v.ovvv, (0, 1, 2, 3), x106, (0, 1, 2, 3), (0, 2))
    del x106
    e_pert += einsum(l1, (0, 1), x116, (1, 0), ()) * -4.0
    del x116
    x117 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x117 += einsum(v.ooov, (0, 1, 2, 3), v.ovov, (1, 3, 2, 4), denom3, (1, 5, 2, 3, 4, 6), (5, 0, 4, 6))
    x118 = np.zeros((nocc, nvir), dtype=np.float64)
    x118 += einsum(t2, (0, 1, 2, 3), x117, (0, 1, 3, 2), (0, 2))
    e_pert += einsum(l1, (0, 1), x118, (1, 0), ()) * 4.0
    del x118
    x119 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x119 += einsum(v.ooov, (0, 1, 2, 3), v.ovov, (1, 4, 2, 3), denom3, (1, 5, 2, 3, 4, 6), (5, 0, 4, 6))
    x120 = np.zeros((nocc, nvir), dtype=np.float64)
    x120 += einsum(t2, (0, 1, 2, 3), x119, (0, 1, 2, 3), (0, 3))
    e_pert += einsum(l1, (0, 1), x120, (1, 0), ()) * 4.0
    del x120
    x121 = np.zeros((nocc, nocc, nocc, nocc, nvir), dtype=np.float64)
    x121 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), denom3, (5, 0, 4, 2, 6, 3), (0, 4, 5, 1, 6))
    x122 = np.zeros((nocc, nvir), dtype=np.float64)
    x122 += einsum(v.ooov, (0, 1, 2, 3), x121, (4, 2, 0, 1, 3), (4, 3))
    e_pert += einsum(l1, (0, 1), x122, (1, 0), ()) * 4.0
    del x122
    x123 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x123 += einsum(v.ooov, (0, 1, 2, 3), v.ovov, (4, 5, 2, 3), denom3, (2, 0, 4, 3, 6, 5), (0, 4, 1, 5, 6))
    x124 = np.zeros((nocc, nvir), dtype=np.float64)
    x124 += einsum(t2, (0, 1, 2, 3), x123, (4, 0, 1, 3, 2), (4, 2))
    e_pert += einsum(l1, (0, 1), x124, (1, 0), ()) * 4.0
    del x124
    x125 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x125 += einsum(v.ooov, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), denom3, (1, 2, 4, 3, 6, 5), (2, 4, 0, 5, 6))
    x126 = np.zeros((nocc, nvir), dtype=np.float64)
    x126 += einsum(t2, (0, 1, 2, 3), x125, (4, 0, 1, 3, 2), (4, 2))
    e_pert += einsum(l1, (0, 1), x126, (1, 0), ()) * 4.0
    del x126
    x127 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x127 += einsum(v.ooov, (0, 1, 2, 3), v.ovov, (4, 3, 2, 5), denom3, (2, 0, 4, 3, 6, 5), (0, 4, 1, 5, 6))
    x128 = np.zeros((nocc, nvir), dtype=np.float64)
    x128 += einsum(t2, (0, 1, 2, 3), x127, (4, 0, 1, 2, 3), (4, 3))
    e_pert += einsum(l1, (0, 1), x128, (1, 0), ()) * 4.0
    del x128
    x129 = np.zeros((nocc, nocc, nocc, nvir, nvir), dtype=np.float64)
    x129 += einsum(v.ooov, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), denom3, (1, 2, 4, 3, 6, 5), (2, 4, 0, 5, 6))
    x130 = np.zeros((nocc, nvir), dtype=np.float64)
    x130 += einsum(t2, (0, 1, 2, 3), x129, (4, 0, 1, 2, 3), (4, 3))
    e_pert += einsum(l1, (0, 1), x130, (1, 0), ()) * 4.0
    del x130
    x131 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x131 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 3), denom3, (1, 4, 5, 3, 6, 2), (4, 5, 0, 6))
    x132 = np.zeros((nocc, nvir), dtype=np.float64)
    x132 += einsum(v.ooov, (0, 1, 2, 3), x131, (2, 0, 1, 3), (0, 3))
    e_pert += einsum(l1, (0, 1), x132, (1, 0), ()) * 4.0
    del x132
    x133 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x133 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 2), denom3, (1, 4, 5, 3, 6, 2), (4, 5, 0, 6))
    x134 = np.zeros((nocc, nvir), dtype=np.float64)
    x134 += einsum(v.ooov, (0, 1, 2, 3), x133, (0, 2, 1, 3), (2, 3))
    e_pert += einsum(l1, (0, 1), x134, (1, 0), ()) * 4.0
    del x134
    x135 = np.zeros((nocc, nvir), dtype=np.float64)
    x135 += einsum(t2, (0, 1, 2, 3), x117, (0, 1, 2, 3), (0, 3))
    del x117
    e_pert += einsum(l1, (0, 1), x135, (1, 0), ()) * -2.0
    del x135
    x136 = np.zeros((nocc, nvir), dtype=np.float64)
    x136 += einsum(v.ooov, (0, 1, 2, 3), x121, (4, 0, 2, 1, 3), (4, 3))
    del x121
    e_pert += einsum(l1, (0, 1), x136, (1, 0), ()) * -2.0
    del x136
    x137 = np.zeros((nocc, nvir), dtype=np.float64)
    x137 += einsum(t2, (0, 1, 2, 3), x127, (4, 0, 1, 3, 2), (4, 2))
    del x127
    e_pert += einsum(l1, (0, 1), x137, (1, 0), ()) * -2.0
    del x137
    x138 = np.zeros((nocc, nvir), dtype=np.float64)
    x138 += einsum(t2, (0, 1, 2, 3), x129, (4, 0, 1, 3, 2), (4, 2))
    del x129
    e_pert += einsum(l1, (0, 1), x138, (1, 0), ()) * -2.0
    del x138
    x139 = np.zeros((nocc, nvir), dtype=np.float64)
    x139 += einsum(t2, (0, 1, 2, 3), x125, (4, 0, 1, 2, 3), (4, 3))
    del x125
    e_pert += einsum(l1, (0, 1), x139, (1, 0), ()) * -2.0
    del x139
    x140 = np.zeros((nocc, nvir), dtype=np.float64)
    x140 += einsum(v.ooov, (0, 1, 2, 3), x133, (2, 0, 1, 3), (0, 3))
    del x133
    e_pert += einsum(l1, (0, 1), x140, (1, 0), ()) * -2.0
    del x140
    x141 = np.zeros((nocc, nvir), dtype=np.float64)
    x141 += einsum(t2, (0, 1, 2, 3), x119, (0, 1, 3, 2), (0, 2))
    del x119
    e_pert += einsum(l1, (0, 1), x141, (1, 0), ()) * -8.0
    del x141
    x142 = np.zeros((nocc, nvir), dtype=np.float64)
    x142 += einsum(t2, (0, 1, 2, 3), x123, (4, 0, 1, 2, 3), (4, 3))
    del x123
    e_pert += einsum(l1, (0, 1), x142, (1, 0), ()) * -8.0
    del x142
    x143 = np.zeros((nocc, nvir), dtype=np.float64)
    x143 += einsum(v.ooov, (0, 1, 2, 3), x131, (0, 2, 1, 3), (2, 3))
    del x131
    e_pert += einsum(l1, (0, 1), x143, (1, 0), ()) * -8.0
    del x143
    x144 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x144 += einsum(v.ovvv, (0, 1, 2, 3), v.ovvv, (0, 4, 5, 3), denom3, (0, 6, 7, 3, 1, 4), (7, 6, 1, 4, 2, 5))
    x145 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x145 += einsum(t2, (0, 1, 2, 3), x144, (0, 1, 4, 2, 3, 5), (0, 1, 4, 5))
    del x144
    x146 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x146 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x146 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x147 = np.zeros((nocc, nvir, nvir, nvir, nvir, nvir), dtype=np.float64)
    x147 += einsum(v.ovvv, (0, 1, 2, 3), v.ovvv, (0, 4, 1, 5), (0, 1, 5, 2, 3, 4))
    x147 += einsum(v.ovvv, (0, 1, 2, 3), x146, (0, 1, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    del x146
    x148 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x148 += einsum(denom3, (0, 1, 2, 3, 4, 5), x147, (0, 3, 6, 4, 7, 5), (1, 2, 4, 5, 6, 7))
    del x147
    x149 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x149 += einsum(t2, (0, 1, 2, 3), x148, (1, 0, 3, 4, 2, 5), (0, 1, 4, 5))
    del x148
    x150 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x150 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x150 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x151 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x151 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x151 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x152 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x152 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x152 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x153 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x153 += einsum(v.ovvv, (0, 1, 2, 3), x151, (4, 0, 1, 5), (0, 4, 1, 5, 2, 3))
    x153 += einsum(v.ovvv, (0, 1, 2, 3), x152, (4, 0, 2, 5), (0, 4, 2, 5, 3, 1)) * 0.5
    del x152
    x154 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x154 += einsum(denom3, (0, 1, 2, 3, 4, 5), x153, (0, 1, 3, 6, 7, 4), (1, 2, 4, 5, 6, 7)) * 2.0
    x155 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x155 += einsum(x145, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x155 += einsum(x145, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x145
    x155 += einsum(x149, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x155 += einsum(x149, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x149
    x155 += einsum(x150, (0, 1, 2, 3), x154, (4, 0, 1, 3, 2, 5), (4, 0, 3, 5)) * -2.0
    del x154
    e_pert += einsum(l2, (0, 1, 2, 3), x155, (3, 2, 0, 1), ()) * -2.0
    del x155
    x156 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x156 += einsum(denom3, (0, 1, 2, 3, 4, 5), x153, (0, 1, 3, 6, 7, 4), (1, 2, 4, 5, 6, 7))
    del x153
    x157 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x157 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x157 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x158 = np.zeros((nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x158 += einsum(x151, (0, 1, 2, 3), x157, (1, 2, 4, 5), denom3, (1, 0, 6, 2, 4, 7), (0, 6, 4, 7, 3, 5)) * -2.0
    del x151, x157
    x159 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x159 += einsum(v.ovvv, (0, 1, 2, 3), x156, (4, 0, 2, 1, 3, 5), (4, 0, 1, 5)) * 2.0
    del x156
    x159 += einsum(v.ovvv, (0, 1, 2, 3), x158, (4, 0, 1, 2, 3, 5), (4, 0, 2, 5))
    del x158
    e_pert += einsum(l2, (0, 1, 2, 3), x159, (3, 2, 1, 0), ()) * -2.0
    del x159
    x160 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x160 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x160 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x161 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x161 += einsum(v.ovvv, (0, 1, 2, 3), v.ovvv, (4, 3, 5, 1), denom3, (6, 0, 4, 1, 7, 3), (0, 4, 6, 7, 2, 5))
    x162 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x162 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x162 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x163 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x163 += einsum(v.ovvv, (0, 1, 2, 3), x162, (4, 3, 5, 1), denom3, (6, 0, 4, 1, 7, 3), (0, 4, 6, 7, 2, 5)) * 2.0
    del x162
    x164 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x164 += einsum(l2, (0, 1, 2, 3), x161, (2, 4, 3, 0, 5, 1), (3, 4, 0, 5))
    del x161
    x164 += einsum(l2, (0, 1, 2, 3), x163, (3, 4, 2, 0, 5, 1), (2, 4, 0, 5)) * -1.0
    del x163
    e_pert += einsum(x160, (0, 1, 2, 3), x164, (0, 1, 3, 2), ()) * -2.0
    del x160, x164
    x165 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x165 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x165 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x166 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x166 += einsum(v.ovvv, (0, 1, 2, 3), v.ovvv, (4, 1, 5, 3), denom3, (6, 0, 4, 1, 7, 3), (0, 4, 6, 7, 2, 5))
    x167 = np.zeros((nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x167 += einsum(v.ovvv, (0, 1, 2, 3), x150, (0, 3, 4, 1), denom3, (0, 5, 6, 1, 7, 3), (5, 6, 7, 4, 2)) * 2.0
    del x150
    x168 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x168 += einsum(l2, (0, 1, 2, 3), x166, (2, 4, 3, 0, 5, 1), (4, 3, 0, 5))
    del x166
    x168 += einsum(l2, (0, 1, 2, 3), x167, (2, 3, 0, 1, 4), (2, 3, 0, 4)) * -1.0
    del x167
    e_pert += einsum(x165, (0, 1, 2, 3), x168, (1, 0, 3, 2), ()) * -4.0
    del x165, x168

    return e_pert

