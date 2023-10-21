# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), x0, (0, 1, 3, 2), ()) * 2.0
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x1 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x2 = np.zeros((nocc, nvir), dtype=types[float])
    x2 += einsum(f.ov, (0, 1), (0, 1))
    x2 += einsum(t1, (0, 1), x1, (0, 2, 1, 3), (2, 3))
    del x1
    e_cc += einsum(t1, (0, 1), x2, (0, 1), ()) * 2.0
    del x2

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (3, 2, 4, 1)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5)) * 2.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (0, 4, 5, 3)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5)) * -1.0
    t2new += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x1 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x1 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x1 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t1new += einsum(x0, (0, 1, 2, 3), x1, (0, 1, 4, 3, 2, 5), (4, 5)) * -0.25
    del x1
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x2 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.3333333333333333
    t1new += einsum(x2, (0, 1, 2, 3), t3, (4, 1, 0, 5, 2, 3), (4, 5)) * 1.5
    del x2
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x3 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    t2new += einsum(v.vvvv, (0, 1, 2, 3), x3, (4, 5, 3, 1), (5, 4, 0, 2))
    x4 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x4 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x4 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    t1new += einsum(x3, (0, 1, 2, 3), x4, (0, 2, 3, 4), (1, 4)) * 2.0
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x5 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x6 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x6 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x6 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x6 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x6 += einsum(x5, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    t1new += einsum(t2, (0, 1, 2, 3), x6, (4, 0, 1, 3), (4, 2)) * -1.0
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x7 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x8 = np.zeros((nocc, nvir), dtype=types[float])
    x8 += einsum(t1, (0, 1), x7, (0, 2, 1, 3), (2, 3)) * 2.0
    x9 = np.zeros((nocc, nvir), dtype=types[float])
    x9 += einsum(f.ov, (0, 1), (0, 1))
    x9 += einsum(x8, (0, 1), (0, 1))
    del x8
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x10 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new += einsum(x9, (0, 1), x10, (0, 2, 1, 3), (2, 3)) * 2.0
    del x10
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x11 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t1new += einsum(t1, (0, 1), x11, (0, 2, 1, 3), (2, 3)) * 2.0
    del x11
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x13 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x13 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x14 = np.zeros((nocc, nocc), dtype=types[float])
    x14 += einsum(f.oo, (0, 1), (0, 1))
    x14 += einsum(x12, (0, 1), (0, 1))
    x14 += einsum(x3, (0, 1, 2, 3), x7, (0, 4, 2, 3), (4, 1)) * 2.0
    x14 += einsum(t1, (0, 1), x13, (2, 3, 0, 1), (3, 2)) * 2.0
    t1new += einsum(t1, (0, 1), x14, (0, 2), (2, 1)) * -1.0
    del x14
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x18 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x18 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(t2, (0, 1, 2, 3), x18, (4, 5, 1, 0), (4, 5, 3, 2))
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(t2, (0, 1, 2, 3), x17, (4, 1, 2, 5), (4, 0, 3, 5))
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(x5, (0, 1, 2, 3), t3, (4, 2, 1, 5, 6, 3), (0, 4, 5, 6))
    x22 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x22 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x22 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(x22, (0, 1, 2, 3), t3, (4, 5, 0, 2, 6, 1), (4, 5, 6, 3)) * 0.5
    del x22
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x24 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x24 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(x24, (0, 1, 2, 3), t3, (2, 4, 0, 3, 5, 6), (4, 1, 6, 5)) * 0.5
    del x24
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(t1, (0, 1), x4, (2, 3, 1, 4), (0, 2, 3, 4)) * 2.0
    del x4
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(t2, (0, 1, 2, 3), x26, (4, 1, 3, 5), (0, 4, 2, 5))
    del x26
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x29 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x29 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x30 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 5, 2, 6, 1, 3), (4, 5, 0, 6))
    x31 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x31 += einsum(t2, (0, 1, 2, 3), x5, (4, 5, 1, 2), (4, 0, 5, 3))
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 0.5
    x32 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x32 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x33 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x33 += einsum(v.ooov, (0, 1, 2, 3), x32, (2, 4, 5, 3), (0, 1, 4, 5)) * 2.0
    del x32
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum(x9, (0, 1), t2, (2, 3, 1, 4), (2, 3, 0, 4))
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x35 += einsum(x28, (0, 1, 2, 3), (2, 0, 1, 3))
    x35 += einsum(x29, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    del x29
    x35 += einsum(x30, (0, 1, 2, 3), (2, 0, 1, 3))
    del x30
    x35 += einsum(x31, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    del x31
    x35 += einsum(x33, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x33
    x35 += einsum(x34, (0, 1, 2, 3), (2, 1, 0, 3))
    del x34
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 += einsum(t1, (0, 1), x35, (0, 2, 3, 4), (2, 3, 1, 4))
    del x35
    x37 = np.zeros((nocc, nocc), dtype=types[float])
    x37 += einsum(t1, (0, 1), x9, (2, 1), (0, 2)) * 0.5
    del x9
    x38 = np.zeros((nocc, nocc), dtype=types[float])
    x38 += einsum(f.oo, (0, 1), (0, 1)) * 0.5
    x38 += einsum(x37, (0, 1), (0, 1))
    del x37
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum(x38, (0, 1), t2, (2, 1, 3, 4), (2, 0, 4, 3)) * 2.0
    del x38
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x16
    x40 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x40 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x19
    x40 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3))
    del x20
    x40 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3))
    del x21
    x40 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x23
    x40 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x25
    x40 += einsum(x27, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x27
    x40 += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3))
    del x36
    x40 += einsum(x39, (0, 1, 2, 3), (1, 0, 3, 2))
    del x39
    t2new += einsum(x40, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x40, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x40
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 1, 2, 5, 6, 3), (4, 0, 5, 6))
    x42 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x42 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x42 += einsum(x5, (0, 1, 2, 3), (0, 2, 1, 3))
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum(x42, (0, 1, 2, 3), t3, (1, 4, 2, 3, 5, 6), (4, 0, 6, 5)) * 0.5
    del x42
    x44 = np.zeros((nocc, nvir), dtype=types[float])
    x44 += einsum(t1, (0, 1), x7, (0, 2, 1, 3), (2, 3))
    x45 = np.zeros((nocc, nvir), dtype=types[float])
    x45 += einsum(f.ov, (0, 1), (0, 1)) * 0.5
    x45 += einsum(x44, (0, 1), (0, 1))
    del x44
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum(x45, (0, 1), t3, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 2.0
    del x45
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x47 += einsum(t1, (0, 1), x5, (2, 3, 0, 4), (2, 3, 1, 4))
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x48 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum(t2, (0, 1, 2, 3), x48, (1, 4, 3, 5), (0, 4, 2, 5)) * 0.5
    del x48
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x47
    x50 += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x49
    x51 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x51 += einsum(t2, (0, 1, 2, 3), x50, (4, 1, 5, 2), (0, 4, 3, 5)) * 2.0
    del x50
    x52 = np.zeros((nvir, nvir), dtype=types[float])
    x52 += einsum(t2, (0, 1, 2, 3), x7, (0, 1, 4, 3), (2, 4)) * 2.0
    x53 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x53 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x53 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x54 = np.zeros((nvir, nvir), dtype=types[float])
    x54 += einsum(t1, (0, 1), x53, (0, 2, 1, 3), (2, 3))
    del x53
    x55 = np.zeros((nvir, nvir), dtype=types[float])
    x55 += einsum(x52, (0, 1), (1, 0))
    del x52
    x55 += einsum(x54, (0, 1), (0, 1)) * -1.0
    del x54
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum(x55, (0, 1), t2, (2, 3, 0, 4), (2, 3, 4, 1))
    del x55
    x57 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x57 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 2), (0, 4, 5, 3))
    x58 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x58 += einsum(x0, (0, 1, 2, 3), t3, (4, 5, 0, 2, 6, 3), (4, 5, 1, 6)) * 0.5
    del x0
    x59 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x59 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    x59 += einsum(x5, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x60 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x60 += einsum(t2, (0, 1, 2, 3), x59, (4, 1, 5, 3), (0, 4, 5, 2)) * 2.0
    del x59
    x61 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x61 += einsum(v.ovvv, (0, 1, 2, 3), x3, (4, 5, 3, 1), (0, 4, 5, 2))
    x62 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x62 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    x62 += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x57
    x62 += einsum(x58, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x58
    x62 += einsum(x60, (0, 1, 2, 3), (1, 0, 2, 3))
    del x60
    x62 += einsum(x61, (0, 1, 2, 3), (2, 1, 0, 3))
    del x61
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum(t1, (0, 1), x62, (2, 3, 0, 4), (2, 3, 1, 4))
    del x62
    x64 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x64 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x64 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 1, 4, 5)) * -1.0
    x65 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x65 += einsum(v.ovvv, (0, 1, 2, 3), x64, (0, 4, 5, 1, 6, 3), (4, 5, 2, 6))
    del x64
    x66 = np.zeros((nocc, nocc), dtype=types[float])
    x66 += einsum(t2, (0, 1, 2, 3), x7, (1, 4, 3, 2), (0, 4))
    x67 = np.zeros((nocc, nocc), dtype=types[float])
    x67 += einsum(t1, (0, 1), x13, (2, 3, 0, 1), (2, 3))
    del x13
    x68 = np.zeros((nocc, nocc), dtype=types[float])
    x68 += einsum(x66, (0, 1), (1, 0))
    del x66
    x68 += einsum(x67, (0, 1), (1, 0))
    del x67
    x69 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x69 += einsum(x68, (0, 1), t2, (2, 0, 3, 4), (2, 1, 4, 3)) * 2.0
    del x68
    x70 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x70 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3))
    del x41
    x70 += einsum(x43, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x43
    x70 += einsum(x46, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x46
    x70 += einsum(x51, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x51
    x70 += einsum(x56, (0, 1, 2, 3), (1, 0, 2, 3))
    del x56
    x70 += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3))
    del x63
    x70 += einsum(x65, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x65
    x70 += einsum(x69, (0, 1, 2, 3), (0, 1, 3, 2))
    del x69
    t2new += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x70, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x70
    x71 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x71 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x71 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.25
    x71 += einsum(t2, (0, 1, 2, 3), x7, (1, 4, 3, 5), (0, 4, 2, 5))
    del x7
    t2new += einsum(t2, (0, 1, 2, 3), x71, (4, 1, 5, 3), (4, 0, 5, 2)) * 4.0
    del x71
    x72 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x72 += einsum(v.ovov, (0, 1, 2, 3), x3, (4, 5, 1, 3), (4, 5, 0, 2))
    del x3
    x73 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x73 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x73 += einsum(x72, (0, 1, 2, 3), (3, 1, 0, 2))
    del x72
    t2new += einsum(t2, (0, 1, 2, 3), x73, (0, 4, 5, 1), (5, 4, 3, 2))
    x74 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x74 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x74 += einsum(t1, (0, 1), x73, (0, 2, 3, 4), (3, 2, 4, 1))
    del x73
    t2new += einsum(t1, (0, 1), x74, (2, 3, 0, 4), (2, 3, 1, 4))
    del x74
    x75 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x75 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    del x15
    t2new += einsum(t2, (0, 1, 2, 3), x75, (4, 1, 5, 2), (4, 0, 5, 3))
    del x75
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x76 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x76, (4, 1, 5, 2), (4, 0, 3, 5))
    del x76
    x77 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x77 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 6), (0, 4, 5, 2, 3, 6))
    x78 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x78 += einsum(t2, (0, 1, 2, 3), x18, (4, 5, 1, 6), (4, 0, 5, 6, 2, 3))
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x80 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x80 += einsum(t2, (0, 1, 2, 3), x79, (4, 5, 6, 3), (4, 0, 1, 5, 2, 6))
    del x79
    x81 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum(x78, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x78
    x81 += einsum(x80, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x80
    x82 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x82 += einsum(t1, (0, 1), x81, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x81
    x83 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x83 += einsum(x77, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x77
    x83 += einsum(x82, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x82
    t3new = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x83
    x84 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum(t2, (0, 1, 2, 3), x18, (4, 5, 6, 1), (4, 0, 6, 5, 2, 3))
    del x18
    x85 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x85 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    x86 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x86 += einsum(t1, (0, 1), x85, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x85
    x87 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x87 += einsum(t2, (0, 1, 2, 3), x17, (4, 5, 3, 6), (4, 0, 1, 5, 2, 6))
    x88 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x88 += einsum(x84, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x84
    x88 += einsum(x86, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x86
    x88 += einsum(x87, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x87
    x89 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x89 += einsum(t1, (0, 1), x88, (2, 3, 0, 4, 5, 6), (2, 3, 4, 1, 5, 6))
    del x88
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x89
    x90 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x90 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    x91 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x91 += einsum(t2, (0, 1, 2, 3), x5, (4, 5, 6, 3), (4, 0, 1, 6, 5, 2))
    x92 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x92 += einsum(t1, (0, 1), x91, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x91
    x93 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x93 += einsum(t1, (0, 1), x92, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x92
    x94 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x94 += einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    x95 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x95 += einsum(t1, (0, 1), x5, (2, 3, 4, 1), (2, 0, 4, 3))
    x96 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x96 += einsum(t1, (0, 1), x95, (2, 3, 0, 4), (3, 2, 4, 1))
    del x95
    x97 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x97 += einsum(x94, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x94
    x97 += einsum(x96, (0, 1, 2, 3), (0, 1, 2, 3))
    del x96
    x98 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x98 += einsum(t2, (0, 1, 2, 3), x97, (4, 5, 1, 6), (0, 4, 5, 3, 2, 6))
    del x97
    x99 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x99 += einsum(x90, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x90
    x99 += einsum(x93, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x93
    x99 += einsum(x98, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    del x98
    t3new += einsum(x99, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x99, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x99, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x99, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x99, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x99, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x99, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x99, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x99, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x99, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x99, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x99, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x99
    x100 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x100 += einsum(t2, (0, 1, 2, 3), x5, (4, 5, 1, 6), (4, 0, 5, 2, 3, 6))
    del x5
    x101 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x101 += einsum(t1, (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x102 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x102 += einsum(t2, (0, 1, 2, 3), x101, (4, 5, 3, 6), (4, 0, 1, 2, 6, 5))
    del x101
    x103 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x103 += einsum(t2, (0, 1, 2, 3), v.oooo, (4, 5, 6, 1), (0, 4, 5, 6, 2, 3))
    x104 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x104 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 6, 3), (0, 1, 4, 6, 2, 5))
    x105 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x105 += einsum(x103, (0, 1, 2, 3, 4, 5), (0, 2, 3, 1, 4, 5)) * -1.0
    del x103
    x105 += einsum(x104, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x104
    x106 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x106 += einsum(t1, (0, 1), x105, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x105
    x107 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x107 += einsum(x100, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x100
    x107 += einsum(x102, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x102
    x107 += einsum(x106, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x106
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x107
    x108 = np.zeros((nvir, nvir), dtype=types[float])
    x108 += einsum(f.ov, (0, 1), t1, (0, 2), (1, 2))
    x109 = np.zeros((nvir, nvir), dtype=types[float])
    x109 += einsum(f.vv, (0, 1), (0, 1))
    x109 += einsum(x108, (0, 1), (0, 1)) * -1.0
    del x108
    t3new += einsum(x109, (0, 1), t3, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    x110 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x110 += einsum(x109, (0, 1), t3, (2, 3, 4, 0, 5, 6), (2, 4, 3, 6, 5, 1))
    del x109
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    del x110
    x111 = np.zeros((nocc, nocc), dtype=types[float])
    x111 += einsum(f.oo, (0, 1), (0, 1))
    x111 += einsum(x12, (0, 1), (0, 1))
    del x12
    t3new += einsum(x111, (0, 1), t3, (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -1.0
    x112 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x112 += einsum(x111, (0, 1), t3, (2, 3, 0, 4, 5, 6), (2, 3, 1, 6, 4, 5))
    del x111
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    del x112
    x113 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x113 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 5, 6, 3), (0, 1, 4, 5, 2, 6))
    x114 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x114 += einsum(t1, (0, 1), x113, (2, 3, 0, 4, 5, 6), (2, 3, 4, 1, 5, 6))
    del x113
    x115 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x115 += einsum(t1, (0, 1), x17, (2, 3, 1, 4), (0, 2, 3, 4))
    del x17
    x116 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x116 += einsum(t2, (0, 1, 2, 3), x115, (4, 5, 1, 6), (4, 5, 0, 2, 3, 6))
    del x115
    x117 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x117 += einsum(x114, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x114
    x117 += einsum(x116, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x116
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x117
    x118 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x118 += einsum(t2, (0, 1, 2, 3), x28, (4, 5, 1, 6), (4, 0, 5, 2, 3, 6))
    del x28
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x118

    return {"t1new": t1new, "t2new": t2new, "t3new": t3new}

