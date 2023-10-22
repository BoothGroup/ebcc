# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(f.ov, (0, 1), t1, (0, 1), ()) * 2.0
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x1 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    e_cc += einsum(x0, (0, 1, 2, 3), x1, (0, 1, 2, 3), ()) * 2.0
    del x0, x1

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (4, 0, 5, 2)) * 2.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (4, 0, 3, 5)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (4, 0, 5, 2)) * -1.0
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x1 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x1 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x1 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t1new += einsum(x0, (0, 1, 2, 3), x1, (1, 4, 0, 2, 3, 5), (4, 5)) * -0.25
    del x1
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.3333333333333333
    x2 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t1new += einsum(x2, (0, 1, 2, 3), t3, (4, 0, 1, 5, 2, 3), (4, 5)) * 1.5
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x3 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
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
    x8 += einsum(t1, (0, 1), x7, (0, 2, 1, 3), (2, 3))
    x9 = np.zeros((nocc, nvir), dtype=types[float])
    x9 += einsum(f.ov, (0, 1), (0, 1)) * 0.5
    x9 += einsum(x8, (0, 1), (0, 1))
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x10 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t1new += einsum(x9, (0, 1), x10, (0, 2, 3, 1), (2, 3)) * 2.0
    del x10
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x11 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t1new += einsum(t1, (0, 1), x11, (0, 2, 1, 3), (2, 3)) * 2.0
    del x11
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x13 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x14 = np.zeros((nocc, nocc), dtype=types[float])
    x14 += einsum(x13, (0, 1, 2, 3), x3, (0, 4, 3, 2), (4, 1)) * 2.0
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x15 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x16 = np.zeros((nocc, nocc), dtype=types[float])
    x16 += einsum(t1, (0, 1), x15, (2, 3, 0, 1), (2, 3)) * 2.0
    x17 = np.zeros((nocc, nocc), dtype=types[float])
    x17 += einsum(f.oo, (0, 1), (0, 1))
    x17 += einsum(x12, (0, 1), (0, 1))
    x17 += einsum(x14, (0, 1), (1, 0))
    x17 += einsum(x16, (0, 1), (1, 0))
    del x16
    t1new += einsum(t1, (0, 1), x17, (0, 2), (2, 1)) * -1.0
    t3new = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    t3new += einsum(x17, (0, 1), t3, (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -1.0
    del x17
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new += einsum(x18, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x19 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x19 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x19, (4, 5, 0, 1), (5, 4, 3, 2))
    x20 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x20 += einsum(t1, (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new += einsum(t1, (0, 1), x20, (2, 3, 1, 4), (0, 2, 3, 4))
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x24 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x24 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(t2, (0, 1, 2, 3), x24, (4, 5, 0, 1), (4, 5, 2, 3))
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(t2, (0, 1, 2, 3), x23, (4, 1, 2, 5), (4, 0, 3, 5))
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(x5, (0, 1, 2, 3), t3, (4, 2, 1, 5, 6, 3), (0, 4, 5, 6))
    x28 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x28 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x28 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum(x28, (0, 1, 2, 3), t3, (4, 5, 0, 1, 6, 2), (4, 5, 6, 3)) * 0.5
    del x28
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x30 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x30 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum(x30, (0, 1, 2, 3), t3, (0, 4, 2, 5, 6, 3), (4, 1, 5, 6)) * 0.5
    del x30
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum(t1, (0, 1), x4, (2, 3, 1, 4), (0, 2, 3, 4)) * 2.0
    del x4
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum(t2, (0, 1, 2, 3), x32, (4, 1, 3, 5), (0, 4, 2, 5))
    del x32
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x35 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x36 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 5, 2, 6, 1, 3), (4, 5, 0, 6))
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x37 += einsum(t2, (0, 1, 2, 3), x5, (4, 5, 1, 2), (4, 0, 5, 3))
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 0.5
    x38 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x38 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x39 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x39 += einsum(v.ooov, (0, 1, 2, 3), x38, (2, 4, 5, 3), (0, 1, 4, 5)) * 2.0
    x40 = np.zeros((nocc, nvir), dtype=types[float])
    x40 += einsum(t1, (0, 1), x7, (0, 2, 1, 3), (2, 3)) * 2.0
    x41 = np.zeros((nocc, nvir), dtype=types[float])
    x41 += einsum(f.ov, (0, 1), (0, 1))
    x41 += einsum(x40, (0, 1), (0, 1))
    del x40
    x42 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x42 += einsum(x41, (0, 1), t2, (2, 3, 1, 4), (2, 3, 0, 4))
    x43 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x43 += einsum(x34, (0, 1, 2, 3), (2, 0, 1, 3))
    x43 += einsum(x35, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    del x35
    x43 += einsum(x36, (0, 1, 2, 3), (2, 0, 1, 3))
    x43 += einsum(x37, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x43 += einsum(x39, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x39
    x43 += einsum(x42, (0, 1, 2, 3), (2, 1, 0, 3))
    del x42
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum(t1, (0, 1), x43, (0, 2, 3, 4), (2, 3, 1, 4))
    del x43
    x45 = np.zeros((nocc, nocc), dtype=types[float])
    x45 += einsum(t1, (0, 1), x41, (2, 1), (0, 2)) * 0.5
    del x41
    x46 = np.zeros((nocc, nocc), dtype=types[float])
    x46 += einsum(f.oo, (0, 1), (0, 1)) * 0.5
    x46 += einsum(x45, (0, 1), (0, 1))
    del x45
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x47 += einsum(x46, (0, 1), t2, (2, 1, 3, 4), (2, 0, 4, 3)) * 2.0
    del x46
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x22
    x48 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x48 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x25
    x48 += einsum(x26, (0, 1, 2, 3), (0, 1, 2, 3))
    del x26
    x48 += einsum(x27, (0, 1, 2, 3), (0, 1, 2, 3))
    del x27
    x48 += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x29
    x48 += einsum(x31, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x31
    x48 += einsum(x33, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x33
    x48 += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3))
    del x44
    x48 += einsum(x47, (0, 1, 2, 3), (1, 0, 3, 2))
    del x47
    t2new += einsum(x48, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x48, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x48
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 1, 2, 5, 6, 3), (4, 0, 5, 6))
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x50 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x50 += einsum(x5, (0, 1, 2, 3), (0, 2, 1, 3))
    x51 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x51 += einsum(x50, (0, 1, 2, 3), t3, (2, 4, 1, 5, 6, 3), (4, 0, 5, 6)) * 0.5
    del x50
    x52 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x52 += einsum(x9, (0, 1), t3, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 2.0
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x53 += einsum(t1, (0, 1), x5, (2, 3, 0, 4), (2, 3, 1, 4))
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x54 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum(t2, (0, 1, 2, 3), x54, (1, 4, 3, 5), (0, 4, 2, 5)) * 0.5
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum(x53, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x56 += einsum(x55, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x55
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum(t2, (0, 1, 2, 3), x56, (4, 1, 5, 2), (0, 4, 3, 5)) * 2.0
    del x56
    x58 = np.zeros((nvir, nvir), dtype=types[float])
    x58 += einsum(t2, (0, 1, 2, 3), x7, (0, 1, 2, 4), (3, 4)) * 2.0
    x59 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x59 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x59 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x60 = np.zeros((nvir, nvir), dtype=types[float])
    x60 += einsum(t1, (0, 1), x59, (0, 1, 2, 3), (2, 3))
    del x59
    x61 = np.zeros((nvir, nvir), dtype=types[float])
    x61 += einsum(x58, (0, 1), (1, 0))
    del x58
    x61 += einsum(x60, (0, 1), (1, 0)) * -1.0
    x62 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x62 += einsum(x61, (0, 1), t2, (2, 3, 0, 4), (2, 3, 4, 1))
    del x61
    x63 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x63 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 2), (0, 4, 5, 3))
    x64 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x64 += einsum(x0, (0, 1, 2, 3), t3, (4, 5, 0, 3, 6, 2), (4, 5, 1, 6)) * 0.5
    x65 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x65 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x65 += einsum(x5, (0, 1, 2, 3), (0, 2, 1, 3))
    x66 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x66 += einsum(t2, (0, 1, 2, 3), x65, (4, 5, 1, 3), (0, 4, 5, 2)) * 2.0
    del x65
    x67 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x67 += einsum(v.ovvv, (0, 1, 2, 3), x3, (4, 5, 3, 1), (0, 4, 5, 2))
    x68 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x68 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    x68 += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x68 += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x64
    x68 += einsum(x66, (0, 1, 2, 3), (1, 0, 2, 3))
    x68 += einsum(x67, (0, 1, 2, 3), (2, 1, 0, 3))
    del x67
    x69 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x69 += einsum(t1, (0, 1), x68, (2, 3, 0, 4), (2, 3, 1, 4))
    del x68
    x70 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x70 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x70 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 1, 4, 5)) * -1.0
    x71 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x71 += einsum(v.ovvv, (0, 1, 2, 3), x70, (0, 4, 5, 1, 6, 3), (4, 5, 2, 6))
    del x70
    x72 = np.zeros((nocc, nocc), dtype=types[float])
    x72 += einsum(t2, (0, 1, 2, 3), x13, (1, 4, 2, 3), (0, 4))
    x73 = np.zeros((nocc, nocc), dtype=types[float])
    x73 += einsum(t1, (0, 1), x15, (2, 3, 0, 1), (2, 3))
    del x15
    x74 = np.zeros((nocc, nocc), dtype=types[float])
    x74 += einsum(x72, (0, 1), (1, 0))
    del x72
    x74 += einsum(x73, (0, 1), (1, 0))
    x75 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum(x74, (0, 1), t2, (2, 0, 3, 4), (2, 1, 4, 3)) * 2.0
    del x74
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3))
    del x49
    x76 += einsum(x51, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x51
    x76 += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x52
    x76 += einsum(x57, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x57
    x76 += einsum(x62, (0, 1, 2, 3), (1, 0, 2, 3))
    del x62
    x76 += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3))
    del x69
    x76 += einsum(x71, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x71
    x76 += einsum(x75, (0, 1, 2, 3), (0, 1, 3, 2))
    del x75
    t2new += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x76, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x76
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x77 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x77 += einsum(t2, (0, 1, 2, 3), x7, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    t2new += einsum(t2, (0, 1, 2, 3), x77, (4, 1, 5, 3), (0, 4, 2, 5)) * 2.0
    del x77
    x78 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x78 += einsum(t1, (0, 1), x5, (2, 3, 4, 1), (2, 0, 4, 3))
    x79 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x79 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x79 += einsum(x78, (0, 1, 2, 3), (3, 1, 2, 0))
    t2new += einsum(t2, (0, 1, 2, 3), x79, (1, 4, 0, 5), (4, 5, 3, 2))
    del x79
    x80 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x80 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x80 += einsum(x19, (0, 1, 2, 3), (3, 1, 0, 2))
    x80 += einsum(x78, (0, 1, 2, 3), (3, 1, 0, 2))
    x81 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x81 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x81 += einsum(t1, (0, 1), x80, (0, 2, 3, 4), (3, 2, 4, 1))
    del x80
    t2new += einsum(t1, (0, 1), x81, (2, 3, 0, 4), (2, 3, 1, 4))
    del x81
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x82 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3))
    del x21
    t2new += einsum(t2, (0, 1, 2, 3), x82, (4, 1, 5, 2), (4, 0, 5, 3))
    del x82
    x83 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x83 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x84 += einsum(x83, (0, 1, 2, 3), (0, 1, 2, 3))
    del x83
    t2new += einsum(t2, (0, 1, 2, 3), x84, (4, 1, 5, 2), (0, 4, 5, 3))
    x85 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x85 += einsum(v.oooo, (0, 1, 2, 3), t3, (3, 4, 1, 5, 6, 7), (4, 0, 2, 7, 5, 6))
    x86 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x86 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x86 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 0.99999999999999
    x87 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x87 += einsum(v.ovov, (0, 1, 2, 3), x86, (2, 4, 5, 1), (4, 0, 5, 3))
    x88 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x88 += einsum(x87, (0, 1, 2, 3), t3, (4, 1, 5, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 1.00000000000001
    x89 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x89 += einsum(x85, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -0.5
    del x85
    x89 += einsum(x88, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    del x88
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    del x89
    x90 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x90 += einsum(v.oooo, (0, 1, 2, 3), t3, (4, 3, 1, 5, 6, 7), (4, 0, 2, 5, 7, 6))
    x91 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x91 += einsum(x24, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 7), (0, 4, 1, 5, 7, 6))
    x92 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x92 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x93 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x93 += einsum(x92, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (0, 4, 5, 6, 7, 2))
    x94 = np.zeros((nocc, nocc), dtype=types[float])
    x94 += einsum(f.oo, (0, 1), (0, 1))
    x94 += einsum(x12, (0, 1), (0, 1))
    del x12
    x94 += einsum(x14, (0, 1), (1, 0))
    del x14
    x95 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x95 += einsum(x94, (0, 1), t3, (2, 3, 0, 4, 5, 6), (2, 3, 1, 6, 4, 5))
    del x94
    x96 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x96 += einsum(x90, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x90
    x96 += einsum(x91, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x91
    x96 += einsum(x93, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x93
    x96 += einsum(x95, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    del x95
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x96
    x97 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x97 += einsum(v.ovov, (0, 1, 2, 3), x86, (2, 4, 5, 1), (4, 0, 5, 3)) * 1.00000000000001
    x98 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x98 += einsum(x97, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    x99 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x99 += einsum(t2, (0, 1, 2, 3), v.oooo, (4, 5, 6, 1), (0, 4, 5, 6, 2, 3))
    x100 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x100 += einsum(t2, (0, 1, 2, 3), x24, (4, 5, 1, 6), (4, 0, 5, 6, 2, 3))
    x101 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x101 += einsum(x99, (0, 1, 2, 3, 4, 5), (0, 2, 3, 1, 4, 5)) * -1.0
    del x99
    x101 += einsum(x100, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x100
    x102 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x102 += einsum(t1, (0, 1), x101, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x101
    x103 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x103 += einsum(x98, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x98
    x103 += einsum(x102, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x102
    t3new += einsum(x103, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x103, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x103, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x103, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x103
    x104 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x104 += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    x105 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x105 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x105 += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3))
    del x63
    x106 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x106 += einsum(t2, (0, 1, 2, 3), x105, (4, 5, 1, 6), (0, 4, 5, 3, 2, 6))
    del x105
    x107 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x107 += einsum(x104, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x104
    x107 += einsum(x106, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x106
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x107
    x108 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x108 += einsum(v.vvvv, (0, 1, 2, 3), t3, (4, 5, 6, 7, 3, 1), (4, 6, 5, 7, 0, 2))
    x109 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x109 += einsum(x19, (0, 1, 2, 3), t3, (3, 4, 2, 5, 6, 7), (1, 0, 4, 5, 7, 6))
    x110 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x110 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 5, 6, 1, 7, 3), (4, 6, 5, 0, 2, 7))
    x111 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x111 += einsum(t2, (0, 1, 2, 3), x110, (4, 5, 6, 0, 1, 7), (4, 5, 6, 2, 3, 7))
    x112 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x112 += einsum(x78, (0, 1, 2, 3), t3, (2, 4, 3, 5, 6, 7), (1, 0, 4, 7, 5, 6))
    del x78
    x113 = np.zeros((nvir, nvir), dtype=types[float])
    x113 += einsum(f.ov, (0, 1), t1, (0, 2), (1, 2))
    x114 = np.zeros((nvir, nvir), dtype=types[float])
    x114 += einsum(x3, (0, 1, 2, 3), x7, (0, 1, 2, 4), (3, 4)) * 2.0
    del x7
    x115 = np.zeros((nvir, nvir), dtype=types[float])
    x115 += einsum(f.vv, (0, 1), (0, 1)) * -1.0
    x115 += einsum(x113, (0, 1), (0, 1))
    x115 += einsum(x114, (0, 1), (1, 0))
    x116 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x116 += einsum(x115, (0, 1), t3, (2, 3, 4, 0, 5, 6), (2, 4, 3, 6, 5, 1))
    del x115
    x117 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x117 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 1, 5, 6, 7, 3), (4, 5, 0, 2, 6, 7))
    x118 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x118 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 5, 6, 7, 3, 1), (4, 6, 5, 0, 7, 2))
    x119 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x119 += einsum(t1, (0, 1), x110, (2, 3, 4, 0, 5, 6), (3, 2, 4, 5, 1, 6))
    del x110
    x120 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x120 += einsum(x117, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x117
    x120 += einsum(x118, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x118
    x120 += einsum(x119, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.5
    del x119
    x121 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x121 += einsum(t1, (0, 1), x120, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x120
    x122 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x122 += einsum(x108, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x108
    x122 += einsum(x109, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.5
    del x109
    x122 += einsum(x111, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.5
    del x111
    x122 += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.5
    del x112
    x122 += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    del x116
    x122 += einsum(x121, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x121
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    del x122
    x123 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x123 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 3), (0, 2, 4, 5))
    x124 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x124 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x124 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x125 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x125 += einsum(v.ovvv, (0, 1, 2, 3), x124, (0, 4, 1, 5), (4, 2, 3, 5))
    x126 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x126 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x126 += einsum(x123, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x123
    x126 += einsum(x125, (0, 1, 2, 3), (0, 3, 2, 1))
    del x125
    x127 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x127 += einsum(t2, (0, 1, 2, 3), x126, (4, 5, 2, 6), (0, 1, 4, 3, 5, 6))
    del x126
    x128 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x128 += einsum(t2, (0, 1, 2, 3), x5, (4, 0, 1, 5), (4, 2, 3, 5))
    x129 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x129 += einsum(t1, (0, 1), x53, (2, 0, 3, 4), (2, 1, 3, 4))
    del x53
    x130 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x130 += einsum(x128, (0, 1, 2, 3), (0, 1, 2, 3))
    del x128
    x130 += einsum(x129, (0, 1, 2, 3), (0, 1, 2, 3))
    del x129
    x131 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x131 += einsum(t2, (0, 1, 2, 3), x130, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6))
    del x130
    x132 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x132 += einsum(x127, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    del x127
    x132 += einsum(x131, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    del x131
    t3new += einsum(x132, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x132, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x132, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x132, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x132, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x132, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x132, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x132, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x132, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x132, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x132, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x132, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x132
    x133 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x133 += einsum(t2, (0, 1, 2, 3), x54, (1, 4, 3, 5), (0, 4, 2, 5))
    del x54
    x134 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x134 += einsum(v.ovov, (0, 1, 2, 3), x86, (2, 4, 5, 3), (4, 0, 5, 1))
    del x86
    x135 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x135 += einsum(x133, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x133
    x135 += einsum(x134, (0, 1, 2, 3), (0, 1, 2, 3))
    del x134
    x136 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x136 += einsum(x135, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 1.00000000000001
    del x135
    x137 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x137 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x137 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x138 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x138 += einsum(x137, (0, 1, 2, 3), t3, (0, 4, 1, 5, 6, 2), (4, 5, 6, 3))
    del x137
    x139 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x139 += einsum(t2, (0, 1, 2, 3), x138, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -0.5
    del x138
    x140 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x140 += einsum(x0, (0, 1, 2, 3), t3, (4, 5, 0, 3, 6, 2), (4, 5, 1, 6))
    del x0
    x141 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x141 += einsum(t2, (0, 1, 2, 3), x140, (4, 5, 1, 6), (0, 4, 5, 3, 2, 6)) * -0.5
    del x140
    x142 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x142 += einsum(x136, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x136
    x142 += einsum(x139, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    del x139
    x142 += einsum(x141, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x141
    t3new += einsum(x142, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x142, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x142, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x142, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x142, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x142, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x142, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x142, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x142
    x143 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x143 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 0, 2, 5, 6, 1), (4, 5, 6, 3))
    x144 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x144 += einsum(t2, (0, 1, 2, 3), x143, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    del x143
    x145 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x145 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 2, 5, 1, 6, 3), (4, 5, 0, 6))
    x146 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x146 += einsum(t2, (0, 1, 2, 3), x145, (4, 5, 1, 6), (0, 4, 5, 2, 3, 6))
    del x145
    x147 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x147 += einsum(x144, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.5
    del x144
    x147 += einsum(x146, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -0.5
    del x146
    t3new += einsum(x147, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x147, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x147, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x147, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x147
    x148 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x148 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 0, 2, 5, 6, 3), (4, 5, 6, 1))
    x149 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x149 += einsum(t2, (0, 1, 2, 3), x148, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    del x148
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 0.5
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -0.5
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -0.5
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 0.5
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    del x149
    x150 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x150 += einsum(x2, (0, 1, 2, 3), t3, (4, 0, 1, 5, 2, 6), (4, 6, 5, 3))
    del x2
    x151 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x151 += einsum(t2, (0, 1, 2, 3), x150, (4, 5, 6, 2), (0, 1, 4, 3, 6, 5)) * 1.5
    del x150
    t3new += einsum(x151, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x151, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    del x151
    x152 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x152 += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (4, 5, 0, 6, 7, 2))
    x153 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x153 += einsum(v.ovov, (0, 1, 2, 3), x3, (4, 5, 1, 3), (4, 5, 0, 2))
    x154 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x154 += einsum(x153, (0, 1, 2, 3), t3, (4, 3, 2, 5, 6, 7), (4, 1, 0, 7, 5, 6))
    x155 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x155 += einsum(x73, (0, 1), t3, (2, 3, 1, 4, 5, 6), (2, 3, 0, 6, 4, 5)) * 2.0
    del x73
    x156 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x156 += einsum(x152, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x152
    x156 += einsum(x154, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    del x154
    x156 += einsum(x155, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x155
    t3new += einsum(x156, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x156, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    del x156
    x157 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x157 += einsum(t2, (0, 1, 2, 3), x36, (4, 5, 1, 6), (0, 4, 5, 2, 3, 6))
    del x36
    t3new += einsum(x157, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x157, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x157, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x157, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x157, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x157, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x157, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x157, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    del x157
    x158 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x158 += einsum(x13, (0, 1, 2, 3), t3, (4, 0, 5, 2, 3, 6), (4, 5, 1, 6))
    del x13
    x159 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x159 += einsum(t2, (0, 1, 2, 3), x158, (4, 5, 1, 6), (0, 5, 4, 3, 2, 6)) * 2.0
    del x158
    t3new += einsum(x159, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x159, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    del x159
    x160 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x160 += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 1, 5, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    x161 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x161 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x161 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x162 = np.zeros((nvir, nvir), dtype=types[float])
    x162 += einsum(t1, (0, 1), x161, (0, 1, 2, 3), (2, 3)) * 2.0
    del x161
    x163 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x163 += einsum(x162, (0, 1), t3, (2, 3, 4, 1, 5, 6), (2, 4, 3, 6, 5, 0))
    del x162
    x164 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x164 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 5, 6, 7, 3, 1), (4, 6, 5, 0, 2, 7))
    x165 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x165 += einsum(x3, (0, 1, 2, 3), x164, (4, 5, 6, 1, 0, 7), (4, 5, 6, 2, 3, 7))
    del x3, x164
    x166 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x166 += einsum(x160, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x160
    x166 += einsum(x163, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    del x163
    x166 += einsum(x165, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x165
    t3new += einsum(x166, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x166, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    del x166
    x167 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x167 += einsum(x87, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (4, 5, 0, 7, 6, 2)) * 1.00000000000001
    del x87
    t3new += einsum(x167, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x167, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x167
    x168 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x168 += einsum(x92, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 6, 7, 2))
    x169 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x169 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 5, 6, 1, 7, 3), (4, 6, 5, 0, 7, 2))
    x170 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x170 += einsum(t1, (0, 1), x169, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x169
    x171 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x171 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 1, 2), (0, 4, 5, 3))
    x172 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x172 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x172 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x173 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x173 += einsum(t2, (0, 1, 2, 3), x172, (4, 5, 1, 3), (0, 4, 5, 2))
    del x172
    x174 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x174 += einsum(x171, (0, 1, 2, 3), (0, 2, 1, 3))
    del x171
    x174 += einsum(x173, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x173
    x175 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x175 += einsum(t2, (0, 1, 2, 3), x174, (4, 1, 5, 6), (0, 4, 5, 3, 2, 6))
    del x174
    x176 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x176 += einsum(x34, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x176 += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3))
    del x37
    x177 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x177 += einsum(t2, (0, 1, 2, 3), x176, (4, 5, 1, 6), (0, 4, 5, 3, 2, 6))
    del x176
    x178 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x178 += einsum(x168, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x168
    x178 += einsum(x170, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -0.5
    del x170
    x178 += einsum(x175, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x175
    x178 += einsum(x177, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    del x177
    t3new += einsum(x178, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x178, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x178, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x178, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    del x178
    x179 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x179 += einsum(t1, (0, 1), x23, (2, 3, 1, 4), (0, 2, 3, 4))
    x180 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x180 += einsum(t2, (0, 1, 2, 3), x179, (4, 5, 1, 6), (4, 5, 0, 2, 3, 6))
    del x179
    x181 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x181 += einsum(t2, (0, 1, 2, 3), x5, (4, 1, 5, 2), (4, 0, 5, 3))
    x182 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x182 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    x182 += einsum(x181, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x181
    x182 += einsum(x66, (0, 1, 2, 3), (1, 0, 2, 3))
    del x66
    x183 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x183 += einsum(t2, (0, 1, 2, 3), x182, (4, 5, 1, 6), (0, 4, 5, 3, 2, 6))
    del x182
    x184 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x184 += einsum(x180, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x180
    x184 += einsum(x183, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    del x183
    t3new += einsum(x184, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x184, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x184, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x184, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    del x184
    x185 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x185 += einsum(x24, (0, 1, 2, 3), t3, (3, 4, 2, 5, 6, 7), (0, 4, 1, 7, 5, 6))
    x186 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x186 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (4, 5, 0, 2, 6, 7))
    x187 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x187 += einsum(t1, (0, 1), x186, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x186
    x188 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x188 += einsum(x9, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4)) * 2.0
    x189 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x189 += einsum(t1, (0, 1), x153, (2, 3, 4, 0), (3, 2, 4, 1))
    del x153
    x190 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x190 += einsum(x188, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x188
    x190 += einsum(x189, (0, 1, 2, 3), (1, 0, 2, 3))
    del x189
    x191 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x191 += einsum(t2, (0, 1, 2, 3), x190, (4, 5, 1, 6), (0, 4, 5, 3, 2, 6))
    del x190
    x192 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x192 += einsum(x185, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.5
    del x185
    x192 += einsum(x187, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x187
    x192 += einsum(x191, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x191
    t3new += einsum(x192, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x192, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x192, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x192, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    del x192
    x193 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x193 += einsum(t2, (0, 1, 2, 3), x20, (4, 5, 3, 6), (4, 0, 1, 2, 6, 5))
    del x20
    x194 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x194 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    x195 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x195 += einsum(v.ovov, (0, 1, 2, 3), x124, (2, 4, 3, 5), (4, 0, 5, 1))
    del x124
    x196 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x196 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x196 += einsum(x194, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x196 += einsum(x195, (0, 1, 2, 3), (0, 1, 2, 3))
    del x195
    x197 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x197 += einsum(t2, (0, 1, 2, 3), x196, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6))
    del x196
    x198 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x198 += einsum(t1, (0, 1), x197, (2, 3, 4, 0, 5, 6), (3, 2, 4, 1, 5, 6))
    del x197
    x199 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x199 += einsum(x193, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x193
    x199 += einsum(x198, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x198
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x199
    x200 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x200 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x201 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x201 += einsum(t2, (0, 1, 2, 3), x200, (4, 5, 1, 6), (4, 5, 0, 2, 3, 6))
    del x200
    x202 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x202 += einsum(t2, (0, 1, 2, 3), x84, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6))
    del x84
    x203 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x203 += einsum(t1, (0, 1), x202, (2, 3, 4, 0, 5, 6), (3, 2, 4, 1, 5, 6)) * -1.0
    del x202
    x204 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x204 += einsum(x201, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x201
    x204 += einsum(x203, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x203
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x204
    x205 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x205 += einsum(x23, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 2), (0, 4, 5, 6, 7, 3))
    t3new += einsum(x205, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x205, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x205, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x205, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x205, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x205, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x205, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    t3new += einsum(x205, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    del x205
    x206 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x206 += einsum(x92, (0, 1, 2, 3), t3, (4, 1, 5, 6, 7, 3), (0, 4, 5, 6, 7, 2))
    t3new += einsum(x206, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x206, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    del x206
    x207 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x207 += einsum(v.vvvv, (0, 1, 2, 3), t3, (4, 5, 6, 1, 7, 3), (4, 6, 5, 7, 0, 2))
    x208 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x208 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 5, 6, 7, 1, 3), (4, 6, 5, 0, 7, 2))
    x209 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x209 += einsum(t1, (0, 1), x208, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x208
    x210 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x210 += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -0.5
    del x207
    x210 += einsum(x209, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x209
    t3new += einsum(x210, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x210, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    del x210
    x211 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x211 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 2), (0, 3, 4, 5))
    x212 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x212 += einsum(t2, (0, 1, 2, 3), x211, (4, 5, 3, 6), (0, 1, 4, 2, 5, 6))
    del x211
    x213 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x213 += einsum(t2, (0, 1, 2, 3), x92, (4, 5, 6, 3), (4, 0, 1, 5, 2, 6))
    x214 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x214 += einsum(t1, (0, 1), x213, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x213
    x215 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x215 += einsum(x212, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x212
    x215 += einsum(x214, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x214
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x215
    x216 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x216 += einsum(t2, (0, 1, 2, 3), x23, (4, 5, 3, 6), (4, 0, 1, 5, 2, 6))
    x217 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x217 += einsum(t1, (0, 1), x216, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x216
    x218 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x218 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 0, 1, 5), (4, 2, 3, 5))
    x219 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x219 += einsum(t1, (0, 1), v.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    x220 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x220 += einsum(t1, (0, 1), x219, (2, 0, 3, 4), (2, 1, 3, 4))
    x221 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x221 += einsum(x218, (0, 1, 2, 3), (0, 1, 2, 3))
    del x218
    x221 += einsum(x220, (0, 1, 2, 3), (0, 1, 2, 3))
    del x220
    x222 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x222 += einsum(t2, (0, 1, 2, 3), x221, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6))
    del x221
    x223 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x223 += einsum(x217, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x217
    x223 += einsum(x222, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    del x222
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x223
    x224 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x224 += einsum(x24, (0, 1, 2, 3), t3, (4, 3, 2, 5, 6, 7), (0, 4, 1, 5, 7, 6))
    t3new += einsum(x224, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4))
    t3new += einsum(x224, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -1.0
    del x224
    x225 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x225 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 5, 2, 6, 7, 3), (4, 5, 0, 1, 6, 7))
    x226 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x226 += einsum(t1, (0, 1), x225, (2, 3, 0, 4, 5, 6), (2, 3, 4, 1, 5, 6))
    del x225
    t3new += einsum(x226, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x226, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x226, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x226, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x226, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x226, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x226, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x226, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x226
    x227 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x227 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (4, 5, 0, 2, 6, 7))
    x228 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x228 += einsum(t1, (0, 1), x227, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x227
    t3new += einsum(x228, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x228, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    del x228
    x229 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x229 += einsum(t1, (0, 1), x24, (2, 3, 0, 4), (2, 3, 4, 1))
    x230 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x230 += einsum(t2, (0, 1, 2, 3), x229, (4, 5, 1, 6), (4, 0, 5, 6, 2, 3))
    del x229
    t3new += einsum(x230, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x230, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x230, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new += einsum(x230, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    del x230
    x231 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x231 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 5, 2, 6, 7, 3), (4, 5, 0, 6, 7, 1))
    t3new += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x231, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new += einsum(x231, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    del x231
    x232 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x232 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 0.499999999999995
    x232 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x232 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x233 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x233 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x233 += einsum(x18, (0, 1, 2, 3), (1, 0, 3, 2))
    del x18
    x233 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x233 += einsum(x194, (0, 1, 2, 3), (1, 0, 3, 2)) * 1.00000000000001
    del x194
    x233 += einsum(v.ovov, (0, 1, 2, 3), x232, (2, 4, 5, 3), (0, 4, 1, 5)) * 2.00000000000002
    del x232
    x234 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x234 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x234 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    x234 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    x234 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x233, (0, 1, 2, 3), x234, (0, 4, 5, 2, 6, 7), (4, 1, 5, 7, 3, 6))
    del x233, x234
    x235 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x235 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x235 += einsum(x219, (0, 1, 2, 3), (1, 0, 3, 2))
    del x219
    x235 += einsum(x92, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x235 += einsum(x97, (0, 1, 2, 3), (1, 0, 3, 2))
    del x97
    t3new += einsum(x235, (0, 1, 2, 3), t3, (4, 0, 5, 6, 2, 7), (4, 1, 5, 6, 3, 7))
    del x235
    x236 = np.zeros((nvir, nvir), dtype=types[float])
    x236 += einsum(f.vv, (0, 1), (0, 1)) * -1.0
    x236 += einsum(x113, (0, 1), (0, 1))
    del x113
    x236 += einsum(x114, (0, 1), (1, 0))
    del x114
    x236 += einsum(x60, (0, 1), (1, 0)) * -1.0
    del x60
    t3new += einsum(x236, (0, 1), t3, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -1.0
    del x236
    x237 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x237 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x237 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    x238 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x238 += einsum(x237, (0, 1, 2, 3), x38, (1, 4, 5, 3), (4, 0, 2, 5))
    x239 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x239 += einsum(t2, (0, 1, 2, 3), x237, (4, 5, 1, 3), (0, 4, 5, 2)) * 0.5
    x240 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x240 += einsum(t2, (0, 1, 2, 3), x237, (4, 5, 1, 2), (0, 4, 5, 3)) * 0.5
    del x237
    x241 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x241 += einsum(x9, (0, 1), t2, (2, 3, 1, 4), (2, 3, 0, 4))
    del x9
    x242 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x242 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x242 += einsum(x92, (0, 1, 2, 3), (0, 1, 3, 2))
    del x92
    x243 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x243 += einsum(t1, (0, 1), x242, (2, 3, 1, 4), (0, 2, 3, 4)) * 0.5
    del x242
    x244 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x244 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x244 += einsum(x24, (0, 1, 2, 3), (2, 1, 0, 3))
    x244 += einsum(x19, (0, 1, 2, 3), (3, 1, 0, 2))
    x245 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x245 += einsum(t1, (0, 1), x244, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x244
    x246 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x246 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -0.5
    x246 += einsum(x5, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.5
    x246 += einsum(x238, (0, 1, 2, 3), (0, 2, 1, 3))
    del x238
    x246 += einsum(x239, (0, 1, 2, 3), (0, 2, 1, 3))
    del x239
    x246 += einsum(x240, (0, 1, 2, 3), (1, 2, 0, 3))
    del x240
    x246 += einsum(x241, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x241
    x246 += einsum(x243, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x243
    x246 += einsum(x245, (0, 1, 2, 3), (0, 2, 1, 3))
    del x245
    x247 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x247 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x247 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    t3new += einsum(x246, (0, 1, 2, 3), x247, (1, 4, 5, 6), (4, 0, 2, 5, 3, 6)) * -2.0
    x248 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x248 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x248 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t3new += einsum(x246, (0, 1, 2, 3), x248, (1, 4, 5, 6), (2, 0, 4, 5, 3, 6)) * -2.0
    del x246
    x249 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x249 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x249 += einsum(x24, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x250 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x250 += einsum(t1, (0, 1), x249, (0, 2, 3, 4), (2, 3, 4, 1))
    del x249
    x251 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x251 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x251 += einsum(x250, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x250
    x252 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x252 += einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    x253 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x253 += einsum(t1, (0, 1), x19, (2, 3, 4, 0), (2, 3, 4, 1))
    del x19
    x254 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x254 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    x254 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x254 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x255 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x255 += einsum(v.ooov, (0, 1, 2, 3), x254, (1, 4, 5, 3), (0, 2, 4, 5))
    del x254
    x256 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x256 += einsum(x38, (0, 1, 2, 3), x5, (4, 0, 5, 3), (4, 5, 1, 2)) * 2.0
    del x38
    x257 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x257 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x257 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x258 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x258 += einsum(v.ooov, (0, 1, 2, 3), x257, (2, 4, 5, 3), (0, 1, 4, 5)) * 2.0
    del x257
    x259 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x259 += einsum(x247, (0, 1, 2, 3), x5, (4, 5, 0, 3), (4, 5, 1, 2))
    del x247
    x260 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x260 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x260 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x260 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x261 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x261 += einsum(t1, (0, 1), x260, (2, 3, 1, 4), (0, 2, 3, 4))
    del x260
    x262 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x262 += einsum(x8, (0, 1), t2, (2, 3, 1, 4), (2, 3, 0, 4)) * 2.0
    del x8
    x263 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x263 += einsum(x253, (0, 1, 2, 3), (0, 2, 1, 3))
    x263 += einsum(x255, (0, 1, 2, 3), (2, 1, 0, 3))
    x263 += einsum(x256, (0, 1, 2, 3), (0, 1, 2, 3))
    x263 += einsum(x258, (0, 1, 2, 3), (2, 1, 0, 3))
    x263 += einsum(x259, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x259
    x263 += einsum(x261, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x261
    x263 += einsum(x262, (0, 1, 2, 3), (1, 2, 0, 3))
    x264 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x264 += einsum(x251, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x264 += einsum(x251, (0, 1, 2, 3), (0, 2, 1, 3))
    del x251
    x264 += einsum(x252, (0, 1, 2, 3), (0, 1, 2, 3))
    x264 += einsum(x252, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x264 += einsum(x263, (0, 1, 2, 3), (1, 0, 2, 3))
    x264 += einsum(x263, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x263
    t3new += einsum(t2, (0, 1, 2, 3), x264, (1, 4, 5, 6), (4, 0, 5, 6, 2, 3)) * -1.0
    del x264
    x265 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x265 += einsum(t1, (0, 1), v.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    x266 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x266 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x266 += einsum(x265, (0, 1, 2, 3), (1, 0, 2, 3))
    del x265
    x266 += einsum(x252, (0, 1, 2, 3), (1, 0, 2, 3))
    del x252
    x267 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x267 += einsum(x34, (0, 1, 2, 3), (0, 2, 1, 3))
    del x34
    x267 += einsum(x258, (0, 1, 2, 3), (2, 1, 0, 3))
    del x258
    x268 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x268 += einsum(x248, (0, 1, 2, 3), x5, (4, 5, 0, 2), (4, 5, 1, 3))
    del x5, x248
    x269 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x269 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x269 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3))
    del x23
    x270 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x270 += einsum(t1, (0, 1), x269, (2, 3, 1, 4), (0, 2, 3, 4))
    del x269
    x271 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x271 += einsum(x253, (0, 1, 2, 3), (0, 1, 2, 3))
    del x253
    x271 += einsum(x255, (0, 1, 2, 3), (2, 0, 1, 3))
    del x255
    x271 += einsum(x256, (0, 1, 2, 3), (0, 2, 1, 3))
    del x256
    x271 += einsum(x268, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x268
    x271 += einsum(x262, (0, 1, 2, 3), (1, 0, 2, 3))
    del x262
    x271 += einsum(x270, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x270
    x272 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x272 += einsum(t1, (0, 1), x24, (2, 3, 4, 0), (2, 4, 3, 1))
    del x24
    x273 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x273 += einsum(x266, (0, 1, 2, 3), (0, 2, 1, 3))
    x273 += einsum(x266, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    del x266
    x273 += einsum(x267, (0, 1, 2, 3), (0, 1, 2, 3))
    x273 += einsum(x267, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x267
    x273 += einsum(x271, (0, 1, 2, 3), (0, 1, 2, 3))
    x273 += einsum(x271, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x271
    x273 += einsum(x272, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x273 += einsum(x272, (0, 1, 2, 3), (1, 0, 2, 3))
    del x272
    t3new += einsum(t2, (0, 1, 2, 3), x273, (4, 5, 1, 6), (5, 0, 4, 3, 2, 6)) * -1.0
    del x273

    return {"t1new": t1new, "t2new": t2new, "t3new": t3new}

def update_lams(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, l1=None, l2=None, l3=None, **kwargs):
    # L amplitudes
    l1new = np.zeros((nvir, nocc), dtype=types[float])
    l1new += einsum(f.ov, (0, 1), (1, 0))
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum(l2, (0, 1, 2, 3), v.vvvv, (4, 1, 5, 0), (4, 5, 3, 2))
    l2new += einsum(v.ovov, (0, 1, 2, 3), (1, 3, 0, 2))
    l2new += einsum(l2, (0, 1, 2, 3), v.oooo, (4, 2, 5, 3), (0, 1, 4, 5))
    l3new = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc), dtype=types[float])
    l3new += einsum(l1, (0, 1), v.ovov, (2, 3, 4, 5), (0, 3, 5, 1, 2, 4))
    l3new += einsum(l1, (0, 1), v.ovov, (2, 3, 4, 5), (5, 3, 0, 1, 2, 4)) * -1.0
    l3new += einsum(l1, (0, 1), v.ovov, (2, 3, 4, 5), (5, 0, 3, 2, 1, 4)) * -1.0
    l3new += einsum(l1, (0, 1), v.ovov, (2, 3, 4, 5), (3, 0, 5, 2, 1, 4))
    l3new += einsum(l1, (0, 1), v.ovov, (2, 3, 4, 5), (0, 5, 3, 2, 4, 1)) * -1.0
    l3new += einsum(l1, (0, 1), v.ovov, (2, 3, 4, 5), (3, 5, 0, 2, 4, 1))
    x0 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 5, 2, 6, 7, 3), (4, 5, 0, 1, 6, 7))
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x1 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x2 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum(x1, (0, 1, 2, 3), t3, (4, 5, 0, 6, 3, 7), (4, 5, 1, 2, 7, 6)) * 0.25
    x3 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (4, 5, 0, 2, 6, 7))
    x4 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 2, 5, 6, 3, 7), (4, 5, 0, 1, 6, 7))
    x5 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x5 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x6 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(t2, (0, 1, 2, 3), x5, (4, 5, 1, 6), (4, 0, 5, 6, 2, 3))
    x7 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(t2, (0, 1, 2, 3), x5, (4, 5, 6, 1), (4, 0, 5, 6, 2, 3))
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x8 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x9 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(x8, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x10 += einsum(x8, (0, 1, 2, 3), (0, 2, 1, 3))
    x11 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(x10, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (4, 5, 0, 2, 7, 6)) * 0.25
    x12 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(x8, (0, 1, 2, 3), t3, (4, 5, 2, 6, 7, 3), (0, 4, 5, 1, 6, 7))
    x13 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(x8, (0, 1, 2, 3), t3, (4, 1, 5, 6, 3, 7), (0, 4, 5, 2, 6, 7))
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x14 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(t2, (0, 1, 2, 3), x14, (1, 4, 3, 5), (0, 4, 2, 5))
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x16 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum(t2, (0, 1, 2, 3), x16, (1, 4, 2, 5), (0, 4, 3, 5)) * 0.5
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    x18 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3))
    x19 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(t2, (0, 1, 2, 3), x18, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6)) * 2.0
    x20 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x20 += einsum(t1, (0, 1), x8, (2, 3, 4, 1), (0, 2, 3, 4))
    x21 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(t2, (0, 1, 2, 3), x20, (4, 5, 1, 6), (5, 4, 0, 6, 2, 3))
    x22 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x22 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x22 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(t1, (0, 1), x22, (2, 3, 1, 4), (0, 2, 3, 4)) * -1.0
    x24 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x24 += einsum(t2, (0, 1, 2, 3), x8, (4, 5, 6, 3), (4, 0, 1, 6, 5, 2))
    x25 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x25 += einsum(x24, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    x25 += einsum(x24, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    x26 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * 1.25
    x26 += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -0.25
    x26 += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -0.25
    x26 += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 1.25
    del x0
    x26 += einsum(x2, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    x26 += einsum(x2, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    del x2
    x26 += einsum(x3, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -0.25
    x26 += einsum(x3, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * 0.25
    x26 += einsum(x3, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 0.25
    x26 += einsum(x3, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.25
    del x3
    x26 += einsum(x4, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -0.25
    x26 += einsum(x4, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 0.25
    del x4
    x26 += einsum(x6, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 0.5
    x26 += einsum(x6, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.5
    x26 += einsum(x7, (0, 1, 2, 3, 4, 5), (0, 3, 1, 2, 4, 5)) * -0.5
    x26 += einsum(x7, (0, 1, 2, 3, 4, 5), (0, 3, 1, 2, 5, 4)) * 0.5
    x26 += einsum(x9, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 1.25
    x26 += einsum(x9, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -0.25
    x26 += einsum(x9, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -0.25
    x26 += einsum(x9, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 1.25
    del x9
    x26 += einsum(x11, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    x26 += einsum(x11, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    del x11
    x26 += einsum(x12, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -0.25
    x26 += einsum(x12, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 0.25
    x26 += einsum(x12, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 0.25
    x26 += einsum(x12, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.25
    del x12
    x26 += einsum(x13, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -0.25
    x26 += einsum(x13, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.25
    del x13
    x26 += einsum(x19, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    x26 += einsum(x19, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * 0.5
    del x19
    x26 += einsum(x21, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.5
    x26 += einsum(x21, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.5
    x26 += einsum(x21, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -0.5
    x26 += einsum(x21, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 0.5
    x26 += einsum(t2, (0, 1, 2, 3), x23, (4, 5, 3, 6), (4, 0, 1, 5, 2, 6)) * -1.0
    del x23
    x26 += einsum(t1, (0, 1), x25, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 6, 1)) * -1.0
    del x25
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x26, (3, 4, 5, 6, 1, 2), (0, 6)) * -1.0
    del x26
    x27 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(f.ov, (0, 1), t3, (2, 3, 4, 5, 6, 1), (0, 2, 4, 3, 5, 6))
    x28 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum(f.ov, (0, 1), t3, (2, 3, 4, 5, 1, 6), (0, 2, 4, 3, 5, 6))
    x29 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum(x1, (0, 1, 2, 3), t3, (4, 5, 0, 6, 7, 3), (4, 5, 1, 2, 6, 7)) * 1.00000000000004
    x30 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 5, 6, 1, 7, 3), (4, 6, 5, 0, 7, 2))
    x31 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum(t2, (0, 1, 2, 3), x18, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6)) * 2.00000000000008
    del x18
    x32 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum(x30, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x32 += einsum(x31, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    del x31
    x33 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum(x10, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (4, 5, 0, 2, 6, 7)) * 1.00000000000004
    x34 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 5, 6, 1, 7, 3), (4, 6, 5, 0, 2, 7))
    x35 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum(t1, (0, 1), x34, (2, 3, 4, 0, 5, 6), (3, 2, 4, 5, 1, 6))
    del x34
    x36 = np.zeros((nocc, nvir), dtype=types[float])
    x36 += einsum(t1, (0, 1), x14, (0, 2, 1, 3), (2, 3))
    x37 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum(x36, (0, 1), t3, (2, 3, 4, 5, 1, 6), (2, 4, 3, 0, 5, 6)) * 0.6666666666666666
    x38 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum(x36, (0, 1), t3, (2, 3, 4, 1, 5, 6), (2, 4, 3, 0, 6, 5)) * 0.6666666666666666
    x39 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x39 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x39 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x39 += einsum(x8, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x39 += einsum(x8, (0, 1, 2, 3), (2, 0, 1, 3))
    x40 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x40 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x40 += einsum(x8, (0, 1, 2, 3), (0, 2, 1, 3))
    x41 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x41 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x41 += einsum(x8, (0, 1, 2, 3), (2, 0, 1, 3))
    x42 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum(x27, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -0.3333333333333333
    x42 += einsum(x27, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 2.333333333333333
    x42 += einsum(x27, (0, 1, 2, 3, 4, 5), (2, 3, 0, 1, 4, 5)) * 0.3333333333333333
    x42 += einsum(x27, (0, 1, 2, 3, 4, 5), (2, 3, 0, 1, 5, 4)) * -0.3333333333333333
    x42 += einsum(x28, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 0.3333333333333333
    x42 += einsum(x28, (0, 1, 2, 3, 4, 5), (1, 3, 0, 2, 5, 4)) * -0.3333333333333333
    del x28
    x42 += einsum(x29, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    x42 += einsum(x29, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x29
    x42 += einsum(x32, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    x42 += einsum(x32, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x32
    x42 += einsum(x33, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    x42 += einsum(x33, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x33
    x42 += einsum(x35, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    x42 += einsum(x35, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 5, 4))
    x42 += einsum(x37, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    x42 += einsum(x37, (0, 1, 2, 3, 4, 5), (1, 2, 3, 0, 4, 5)) * -1.0
    del x37
    x42 += einsum(x38, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    x42 += einsum(x38, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * 6.999999999999999
    x42 += einsum(x38, (0, 1, 2, 3, 4, 5), (0, 2, 3, 1, 4, 5))
    x42 += einsum(x38, (0, 1, 2, 3, 4, 5), (0, 2, 3, 1, 5, 4)) * -1.0
    del x38
    x42 += einsum(x39, (0, 1, 2, 3), t3, (4, 5, 0, 6, 3, 7), (4, 5, 2, 1, 6, 7)) * -1.00000000000004
    del x39
    x42 += einsum(x40, (0, 1, 2, 3), t3, (4, 2, 5, 6, 3, 7), (4, 5, 1, 0, 7, 6)) * 1.00000000000004
    x42 += einsum(x41, (0, 1, 2, 3), t3, (4, 0, 5, 3, 6, 7), (4, 5, 2, 1, 6, 7)) * -2.00000000000008
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x42, (3, 5, 6, 4, 1, 2), (0, 6)) * -0.24999999999999
    del x42
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x43 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum(t2, (0, 1, 2, 3), x43, (1, 4, 3, 5), (0, 4, 2, 5)) * 1.5000000000000002
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x45 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 3.0000000000000004
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum(t2, (0, 1, 2, 3), x45, (1, 4, 2, 5), (0, 4, 3, 5)) * 0.5
    del x45
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x47 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x47 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -2.0
    x47 += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x44
    x47 += einsum(x46, (0, 1, 2, 3), (0, 1, 2, 3))
    del x46
    x48 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum(t2, (0, 1, 2, 3), x47, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6)) * 6.00000000000024
    del x47
    x49 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x49 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    x50 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x50 += einsum(x49, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -0.5
    x50 += einsum(x49, (0, 1, 2, 3, 4, 5), (0, 1, 3, 4, 2, 5))
    x51 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x51 += einsum(t1, (0, 1), x50, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * 12.00000000000048
    del x50
    x52 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x52 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 5, 6, 7, 1, 3), (4, 6, 5, 0, 7, 2))
    x53 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x53 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 5, 6, 7, 3, 1), (4, 6, 5, 0, 7, 2))
    x54 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x54 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x54 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum(t1, (0, 1), x54, (2, 3, 1, 4), (0, 2, 3, 4))
    x56 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum(t2, (0, 1, 2, 3), x55, (4, 5, 2, 6), (0, 1, 4, 5, 3, 6)) * 12.00000000000048
    del x55
    x57 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x57 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 5, 6, 7, 3, 1), (4, 6, 5, 0, 2, 7))
    x58 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x58 += einsum(t1, (0, 1), x57, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    x59 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum(t1, (0, 1), x57, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    x60 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x60 += einsum(x24, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x60 += einsum(x24, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -0.5
    x61 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum(t1, (0, 1), x60, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6)) * 12.00000000000048
    del x60
    x62 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x62 += einsum(x48, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    x62 += einsum(x48, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x48
    x62 += einsum(x51, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    x62 += einsum(x51, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 5, 4)) * -1.0
    del x51
    x62 += einsum(x52, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * 2.0
    x62 += einsum(x52, (0, 1, 2, 3, 4, 5), (2, 1, 3, 0, 4, 5)) * -2.0
    x62 += einsum(x30, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    x62 += einsum(x30, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    x62 += einsum(x30, (0, 1, 2, 3, 4, 5), (2, 0, 3, 1, 4, 5)) * -1.0
    x62 += einsum(x30, (0, 1, 2, 3, 4, 5), (2, 1, 3, 0, 4, 5))
    x62 += einsum(x53, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -8.0
    x62 += einsum(x53, (0, 1, 2, 3, 4, 5), (2, 1, 3, 0, 4, 5)) * 2.0
    del x53
    x62 += einsum(x56, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    x62 += einsum(x56, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x56
    x62 += einsum(x58, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * 8.0
    x62 += einsum(x58, (0, 1, 2, 3, 4, 5), (2, 1, 3, 0, 5, 4)) * -2.0
    del x58
    x62 += einsum(x35, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    x62 += einsum(x35, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 5, 4)) * -1.0
    x62 += einsum(x35, (0, 1, 2, 3, 4, 5), (2, 0, 3, 1, 5, 4))
    x62 += einsum(x35, (0, 1, 2, 3, 4, 5), (2, 1, 3, 0, 5, 4)) * -1.0
    x62 += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -2.0
    x62 += einsum(x59, (0, 1, 2, 3, 4, 5), (2, 1, 3, 0, 5, 4)) * 2.0
    x62 += einsum(x61, (0, 1, 2, 3, 4, 5), (1, 2, 3, 0, 5, 4))
    x62 += einsum(x61, (0, 1, 2, 3, 4, 5), (2, 1, 3, 0, 5, 4)) * -1.0
    del x61
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x62, (5, 3, 6, 4, 2, 1), (0, 6)) * 0.08333333333333
    del x62
    x63 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum(t2, (0, 1, 2, 3), v.oooo, (4, 5, 6, 1), (0, 4, 5, 6, 2, 3))
    x64 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum(x36, (0, 1), t3, (2, 3, 4, 1, 5, 6), (2, 4, 3, 0, 6, 5)) * 0.11111111111110666
    x65 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x65 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x66 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x66 += einsum(t2, (0, 1, 2, 3), x65, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    x67 = np.zeros((nocc, nvir), dtype=types[float])
    x67 += einsum(t1, (0, 1), x14, (0, 2, 1, 3), (2, 3)) * 2.0
    x68 = np.zeros((nocc, nvir), dtype=types[float])
    x68 += einsum(f.ov, (0, 1), (0, 1))
    x68 += einsum(x67, (0, 1), (0, 1))
    l3new += einsum(x68, (0, 1), l2, (2, 3, 4, 5), (1, 2, 3, 0, 4, 5))
    l3new += einsum(x68, (0, 1), l2, (2, 3, 4, 5), (2, 3, 1, 4, 5, 0))
    l3new += einsum(x68, (0, 1), l2, (2, 3, 4, 5), (2, 3, 1, 0, 5, 4)) * -1.0
    l3new += einsum(x68, (0, 1), l2, (2, 3, 4, 5), (1, 2, 3, 5, 4, 0)) * -1.0
    x69 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x69 += einsum(x63, (0, 1, 2, 3, 4, 5), (1, 2, 3, 0, 4, 5)) * 0.3333333333333333
    x69 += einsum(x63, (0, 1, 2, 3, 4, 5), (1, 2, 3, 0, 5, 4)) * -1.0
    x69 += einsum(x63, (0, 1, 2, 3, 4, 5), (1, 3, 2, 0, 4, 5)) * -0.3333333333333333
    x69 += einsum(x63, (0, 1, 2, 3, 4, 5), (1, 3, 2, 0, 5, 4)) * 0.3333333333333333
    x69 += einsum(x27, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -0.05555555555555333
    x69 += einsum(x27, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 0.05555555555555333
    del x27
    x69 += einsum(x7, (0, 1, 2, 3, 4, 5), (2, 3, 0, 1, 4, 5)) * 0.3333333333333333
    x69 += einsum(x7, (0, 1, 2, 3, 4, 5), (2, 3, 0, 1, 5, 4)) * -1.0
    x69 += einsum(x6, (0, 1, 2, 3, 4, 5), (3, 2, 0, 1, 4, 5)) * -0.3333333333333333
    x69 += einsum(x6, (0, 1, 2, 3, 4, 5), (3, 2, 0, 1, 5, 4)) * 0.3333333333333333
    x69 += einsum(x64, (0, 1, 2, 3, 4, 5), (3, 1, 0, 2, 4, 5)) * -1.0
    x69 += einsum(x64, (0, 1, 2, 3, 4, 5), (3, 1, 0, 2, 5, 4))
    del x64
    x69 += einsum(x66, (0, 1, 2, 3, 4, 5), (3, 0, 1, 2, 4, 5)) * 0.3333333333333333
    x69 += einsum(x66, (0, 1, 2, 3, 4, 5), (3, 0, 1, 2, 5, 4)) * -1.0
    x69 += einsum(x66, (0, 1, 2, 3, 4, 5), (3, 1, 0, 2, 4, 5)) * -0.3333333333333333
    x69 += einsum(x66, (0, 1, 2, 3, 4, 5), (3, 1, 0, 2, 5, 4)) * 0.3333333333333333
    x69 += einsum(x68, (0, 1), t3, (2, 3, 4, 5, 1, 6), (0, 2, 4, 3, 5, 6)) * 0.05555555555555333
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x69, (6, 5, 4, 3, 1, 2), (0, 6)) * 1.5
    del x69
    x70 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x70 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    x71 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x71 += einsum(t2, (0, 1, 2, 3), x70, (4, 5, 6, 3), (0, 1, 4, 5, 2, 6))
    x72 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x72 += einsum(t1, (0, 1), x49, (2, 3, 4, 5, 0, 6), (2, 3, 5, 4, 1, 6))
    x73 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum(t1, (0, 1), x24, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    x74 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x74 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x74 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    x74 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    x75 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum(x71, (0, 1, 2, 3, 4, 5), (3, 1, 0, 2, 4, 5)) * 2.0
    x75 += einsum(x66, (0, 1, 2, 3, 4, 5), (3, 2, 1, 0, 5, 4)) * 2.0
    del x66
    x75 += einsum(x59, (0, 1, 2, 3, 4, 5), (3, 0, 1, 2, 5, 4)) * -1.9999999999999198
    del x59
    x75 += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -2.0
    x75 += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 2.0
    x75 += einsum(x73, (0, 1, 2, 3, 4, 5), (3, 1, 2, 0, 5, 4)) * -2.0
    x75 += einsum(x73, (0, 1, 2, 3, 4, 5), (3, 2, 1, 0, 5, 4)) * 2.0
    x75 += einsum(x40, (0, 1, 2, 3), x74, (2, 4, 5, 3, 6, 7), (1, 5, 4, 0, 7, 6))
    del x74
    x75 += einsum(x41, (0, 1, 2, 3), t3, (4, 0, 5, 6, 3, 7), (2, 4, 5, 1, 6, 7)) * -1.0
    del x41
    x75 += einsum(x68, (0, 1), t3, (2, 3, 4, 5, 1, 6), (0, 2, 4, 3, 5, 6)) * 0.99999999999996
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x75, (6, 3, 5, 4, 0, 2), (1, 6)) * -0.25
    del x75
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x77 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x78 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x78 += einsum(v.ovov, (0, 1, 2, 3), x77, (2, 4, 3, 5), (0, 4, 1, 5)) * 0.5
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x79 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x79 += einsum(x78, (0, 1, 2, 3), (1, 0, 3, 2))
    del x78
    x80 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x80 += einsum(t2, (0, 1, 2, 3), x79, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6))
    del x79
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x81 += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x82 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum(t2, (0, 1, 2, 3), x81, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6)) * 2.0
    del x81
    x83 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x83 += einsum(t1, (0, 1), x49, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 1, 6))
    x84 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum(x80, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    x84 += einsum(x80, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x80
    x84 += einsum(x82, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    x84 += einsum(x82, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * 0.5
    del x82
    x84 += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * 2.0
    x84 += einsum(x72, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 5, 4)) * -1.0
    del x72
    x84 += einsum(x83, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    x84 += einsum(x83, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 5, 4))
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x84, (3, 4, 6, 5, 2, 1), (0, 6)) * -1.0
    del x84
    x85 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x85 += einsum(x5, (0, 1, 2, 3), (0, 2, 1, 3))
    x85 += einsum(x5, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    x86 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x86 += einsum(t2, (0, 1, 2, 3), x85, (4, 1, 5, 6), (0, 4, 5, 6, 3, 2)) * 6.0
    del x85
    x87 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x87 += einsum(x65, (0, 1, 2, 3), (1, 0, 3, 2))
    x87 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3))
    l2new += einsum(l2, (0, 1, 2, 3), x87, (3, 2, 4, 5), (0, 1, 5, 4))
    x88 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x88 += einsum(t2, (0, 1, 2, 3), x87, (4, 5, 1, 6), (0, 4, 5, 6, 3, 2)) * 3.0
    x89 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x89 += einsum(x63, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -3.0
    x89 += einsum(x63, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4))
    x89 += einsum(x63, (0, 1, 2, 3, 4, 5), (3, 0, 2, 1, 4, 5)) * 3.0
    x89 += einsum(x63, (0, 1, 2, 3, 4, 5), (3, 0, 2, 1, 5, 4)) * -1.0
    x89 += einsum(x86, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 5, 4)) * -1.0
    x89 += einsum(x86, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * 0.3333333333333333
    del x86
    x89 += einsum(x88, (0, 1, 2, 3, 4, 5), (2, 0, 3, 1, 5, 4)) * -1.0
    x89 += einsum(x88, (0, 1, 2, 3, 4, 5), (2, 0, 3, 1, 4, 5)) * 0.3333333333333333
    x89 += einsum(x88, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 5, 4))
    x89 += einsum(x88, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -0.3333333333333333
    del x88
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x89, (3, 4, 6, 5, 1, 2), (0, 6)) * -0.25
    del x89
    x90 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x90 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x91 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x91 += einsum(t2, (0, 1, 2, 3), x90, (4, 5, 6, 3), (4, 0, 1, 5, 2, 6))
    x92 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x92 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x92 += einsum(x5, (0, 1, 2, 3), (2, 1, 0, 3))
    x92 += einsum(x65, (0, 1, 2, 3), (3, 1, 0, 2))
    x93 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x93 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x93 += einsum(x90, (0, 1, 2, 3), (0, 1, 3, 2))
    x93 += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x94 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x94 += einsum(x63, (0, 1, 2, 3, 4, 5), (3, 0, 2, 1, 5, 4)) * -1.0000000000000402
    del x63
    x94 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 5, 6, 2), (0, 1, 4, 5, 6, 3)) * 1.0000000000000402
    x94 += einsum(x6, (0, 1, 2, 3, 4, 5), (2, 1, 3, 0, 5, 4)) * -1.0000000000000402
    del x6
    x94 += einsum(x52, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x52
    x94 += einsum(x91, (0, 1, 2, 3, 4, 5), (2, 1, 3, 0, 5, 4)) * 1.0000000000000402
    x94 += einsum(x40, (0, 1, 2, 3), t3, (4, 5, 2, 6, 7, 3), (4, 5, 1, 0, 6, 7)) * 0.5000000000000201
    x94 += einsum(t2, (0, 1, 2, 3), x92, (1, 4, 5, 6), (4, 0, 6, 5, 2, 3)) * 1.0000000000000402
    del x92
    x94 += einsum(t2, (0, 1, 2, 3), x93, (4, 5, 6, 3), (0, 1, 5, 4, 6, 2)) * -1.0000000000000402
    del x93
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x94, (5, 3, 6, 4, 0, 2), (1, 6)) * 0.49999999999997996
    del x94
    x95 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x95 += einsum(t1, (0, 1), x24, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x24
    x96 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x96 += einsum(v.ovov, (0, 1, 2, 3), x77, (2, 4, 3, 5), (0, 4, 1, 5))
    x97 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x97 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x97 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x97 += einsum(x96, (0, 1, 2, 3), (1, 0, 3, 2))
    x98 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x98 += einsum(x8, (0, 1, 2, 3), t3, (4, 5, 2, 6, 3, 7), (4, 5, 0, 1, 6, 7))
    x98 += einsum(x95, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    x98 += einsum(x7, (0, 1, 2, 3, 4, 5), (1, 3, 0, 2, 4, 5))
    x98 += einsum(x7, (0, 1, 2, 3, 4, 5), (1, 3, 0, 2, 5, 4)) * -1.0
    del x7
    x98 += einsum(x21, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5))
    x98 += einsum(x21, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -1.0
    del x21
    x98 += einsum(t2, (0, 1, 2, 3), x97, (4, 5, 6, 3), (0, 1, 4, 5, 2, 6)) * -2.0
    del x97
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x98, (3, 4, 5, 6, 0, 2), (1, 6)) * 0.5
    del x98
    x99 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x99 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x100 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x100 += einsum(t2, (0, 1, 2, 3), x99, (4, 5, 3, 6), (4, 0, 1, 5, 2, 6))
    x101 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x101 += einsum(x71, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.25
    del x71
    x101 += einsum(x100, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * 0.5
    x101 += einsum(x100, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -0.5
    x101 += einsum(x91, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    x101 += einsum(x91, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 0.5
    del x91
    x101 += einsum(x73, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    x101 += einsum(x73, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -0.5
    del x73
    x101 += einsum(x95, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -0.5
    x101 += einsum(x95, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 0.5
    del x95
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x101, (5, 4, 3, 6, 2, 1), (0, 6)) * 2.0
    del x101
    x102 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x102 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x102 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x103 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x103 += einsum(x57, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    x103 += einsum(x57, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    x104 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x104 += einsum(x30, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x104 += einsum(x30, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    del x30
    x104 += einsum(x35, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    x104 += einsum(x35, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4))
    del x35
    x104 += einsum(x102, (0, 1, 2, 3), t3, (4, 5, 6, 7, 1, 2), (4, 6, 5, 0, 7, 3)) * -2.0
    del x102
    x104 += einsum(t1, (0, 1), x103, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 6, 1)) * -2.0
    del x103
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x104, (5, 4, 3, 6, 2, 1), (0, 6)) * 0.08333333333333
    del x104
    x105 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x105 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x105 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x105 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    del x15
    x105 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3))
    del x17
    x106 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x106 += einsum(x49, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    x106 += einsum(x49, (0, 1, 2, 3, 4, 5), (0, 1, 3, 4, 2, 5))
    del x49
    x107 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x107 += einsum(t2, (0, 1, 2, 3), x105, (4, 5, 6, 3), (0, 1, 5, 4, 2, 6)) * 2.0
    del x105
    x107 += einsum(t1, (0, 1), x106, (2, 3, 4, 5, 0, 6), (2, 3, 5, 4, 6, 1)) * -2.0
    del x106
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x107, (4, 3, 6, 5, 1, 2), (0, 6)) * -0.5
    del x107
    x108 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x108 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (4, 5, 0, 2, 6, 7)) * 0.5
    x108 += einsum(x83, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x83
    x108 += einsum(x100, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x100
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x108, (3, 4, 5, 6, 0, 2), (1, 6))
    del x108
    x109 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x109 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 6, 5, 7, 1, 2), (4, 6, 0, 7))
    x110 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x110 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 4, 3, 2, 7, 0), (5, 6, 1, 7))
    x111 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x111 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 6, 5, 0, 7, 2), (4, 6, 1, 7))
    x112 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x112 += einsum(t1, (0, 1), l3, (2, 3, 1, 4, 5, 6), (4, 6, 5, 0, 2, 3))
    l3new += einsum(v.ooov, (0, 1, 2, 3), x112, (4, 5, 1, 0, 6, 7), (6, 3, 7, 4, 2, 5))
    x113 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x113 += einsum(t2, (0, 1, 2, 3), x112, (0, 4, 1, 5, 6, 3), (4, 5, 6, 2))
    x114 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x114 += einsum(t2, (0, 1, 2, 3), x112, (0, 4, 1, 5, 2, 6), (4, 5, 6, 3))
    x115 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x115 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    x115 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 5.0
    x115 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    x115 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    x115 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    x115 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    x116 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x116 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x116 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    x116 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    x117 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x117 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x117 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x118 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x118 += einsum(t1, (0, 1), l3, (2, 1, 3, 4, 5, 6), (4, 6, 5, 0, 2, 3))
    l3new += einsum(x8, (0, 1, 2, 3), x118, (4, 5, 0, 1, 6, 7), (7, 3, 6, 5, 2, 4))
    l3new += einsum(x68, (0, 1), x118, (2, 3, 4, 0, 5, 6), (6, 1, 5, 3, 4, 2)) * -1.0
    x119 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x119 += einsum(x118, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x119 += einsum(x118, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    x120 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x120 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x120 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x121 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x121 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x121 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x122 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x122 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x123 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x123 += einsum(t2, (0, 1, 2, 3), l3, (4, 2, 3, 1, 5, 6), (6, 5, 0, 4))
    x124 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x124 += einsum(t2, (0, 1, 2, 3), l3, (4, 3, 2, 1, 5, 6), (6, 5, 0, 4))
    x125 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x125 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.3333333333333333
    x125 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x126 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x126 += einsum(x125, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (6, 5, 1, 4)) * 0.75
    x127 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x127 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3))
    x127 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x127 += einsum(x123, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.75
    x127 += einsum(x123, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.25
    x127 += einsum(x124, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    x127 += einsum(x124, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.25
    x127 += einsum(x126, (0, 1, 2, 3), (1, 0, 2, 3))
    x127 += einsum(x117, (0, 1, 2, 3), l3, (2, 4, 3, 0, 5, 6), (5, 6, 1, 4)) * -0.25
    x128 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x128 += einsum(x109, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x128 += einsum(x110, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x128 += einsum(x111, (0, 1, 2, 3), (0, 1, 2, 3))
    x128 += einsum(x113, (0, 1, 2, 3), (0, 1, 3, 2)) * 4.0
    x128 += einsum(x114, (0, 1, 2, 3), (0, 1, 3, 2)) * 4.0
    x128 += einsum(l3, (0, 1, 2, 3, 4, 5), x115, (5, 6, 4, 2, 7, 1), (3, 6, 0, 7))
    x128 += einsum(l3, (0, 1, 2, 3, 4, 5), x116, (3, 6, 5, 1, 7, 2), (4, 6, 0, 7))
    x128 += einsum(x117, (0, 1, 2, 3), x119, (0, 1, 4, 5, 6, 3), (4, 5, 2, 6)) * -4.0
    del x119
    x128 += einsum(l2, (0, 1, 2, 3), x120, (3, 4, 1, 5), (2, 4, 0, 5)) * 8.0
    x128 += einsum(l2, (0, 1, 2, 3), x121, (2, 4, 1, 5), (3, 4, 0, 5)) * 4.0
    x128 += einsum(t1, (0, 1), x127, (0, 2, 3, 4), (2, 3, 1, 4)) * 8.0
    del x127
    l1new += einsum(v.ovvv, (0, 1, 2, 3), x128, (4, 0, 3, 2), (1, 4)) * -0.25
    del x128
    x129 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x129 += einsum(t1, (0, 1), l2, (2, 3, 4, 0), (4, 2, 3, 1))
    x130 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x130 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x130 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x131 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x131 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    x131 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1)) * 0.3333333333333333
    x131 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 3, 5, 0, 1, 2)) * 0.3333333333333333
    x132 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x132 += einsum(x129, (0, 1, 2, 3), (0, 1, 2, 3))
    x132 += einsum(x130, (0, 1, 2, 3), l3, (4, 5, 2, 1, 6, 0), (6, 4, 5, 3)) * -0.125
    x132 += einsum(t2, (0, 1, 2, 3), x131, (0, 4, 1, 5, 2, 6), (4, 5, 6, 3)) * 0.75
    del x131
    l1new += einsum(v.vvvv, (0, 1, 2, 3), x132, (4, 1, 2, 3), (0, 4)) * 2.0
    del x132
    x133 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x133 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    x133 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1))
    x133 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 3, 5, 0, 1, 2))
    x134 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x134 += einsum(x129, (0, 1, 2, 3), (0, 1, 2, 3))
    del x129
    x134 += einsum(x130, (0, 1, 2, 3), l3, (4, 5, 2, 1, 6, 0), (6, 4, 5, 3)) * -0.75
    x134 += einsum(t2, (0, 1, 2, 3), x133, (0, 4, 1, 5, 2, 6), (4, 5, 6, 3)) * 0.5
    del x133
    l1new += einsum(v.vvvv, (0, 1, 2, 3), x134, (4, 2, 1, 3), (0, 4)) * -1.0
    del x134
    x135 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x135 += einsum(l2, (0, 1, 2, 3), t3, (4, 5, 3, 6, 0, 1), (2, 4, 5, 6))
    x136 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x136 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), (1, 2, 3, 4))
    x137 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x137 += einsum(t1, (0, 1), x122, (2, 3, 4, 1), (3, 2, 4, 0))
    l2new += einsum(v.ovov, (0, 1, 2, 3), x137, (4, 5, 0, 2), (3, 1, 5, 4))
    x138 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x138 += einsum(t1, (0, 1), x137, (0, 2, 3, 4), (2, 4, 3, 1))
    x139 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x139 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 4, 7, 0, 1, 2), (3, 5, 6, 7))
    x140 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x140 += einsum(t1, (0, 1), x139, (0, 2, 3, 4), (2, 4, 3, 1))
    x141 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x141 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    x141 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    x142 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x142 += einsum(t1, (0, 1), x141, (2, 3, 4, 1, 5, 6), (0, 2, 3, 4, 5, 6))
    l3new += einsum(x8, (0, 1, 2, 3), x142, (2, 4, 5, 0, 6, 7), (6, 3, 7, 5, 1, 4)) * -1.0
    x143 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x143 += einsum(t1, (0, 1), x142, (2, 3, 4, 5, 1, 6), (0, 4, 3, 5, 2, 6)) * -1.0
    del x142
    x144 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x144 += einsum(t2, (0, 1, 2, 3), x143, (4, 5, 0, 1, 6, 3), (5, 4, 6, 2)) * -0.25
    del x143
    x145 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x145 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 1, 2))
    x145 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x146 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x146 += einsum(t2, (0, 1, 2, 3), x145, (1, 4, 0, 5, 2, 6), (4, 3, 5, 6))
    del x145
    x147 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x147 += einsum(t2, (0, 1, 2, 3), x146, (4, 5, 3, 2), (0, 1, 4, 5)) * 0.25
    del x146
    x148 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x148 += einsum(x140, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.124999999999995
    del x140
    x148 += einsum(x144, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x144
    x148 += einsum(x147, (0, 1, 2, 3), (2, 1, 0, 3))
    del x147
    x149 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x149 += einsum(t2, (0, 1, 2, 3), l3, (4, 5, 2, 0, 1, 6), (6, 4, 5, 3))
    x150 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x150 += einsum(t2, (0, 1, 2, 3), x149, (4, 2, 3, 5), (4, 0, 1, 5))
    x151 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x151 += einsum(t1, (0, 1), x118, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    x152 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x152 += einsum(t2, (0, 1, 2, 3), x151, (1, 4, 0, 5, 6, 3), (4, 5, 6, 2))
    x153 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x153 += einsum(x150, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x150
    x153 += einsum(x152, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.25
    del x152
    x154 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x154 += einsum(x130, (0, 1, 2, 3), x151, (0, 1, 4, 5, 6, 2), (4, 5, 6, 3)) * 0.375
    x155 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x155 += einsum(x117, (0, 1, 2, 3), l3, (4, 5, 2, 1, 6, 0), (6, 4, 5, 3))
    x156 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x156 += einsum(t2, (0, 1, 2, 3), x155, (4, 3, 2, 5), (0, 1, 4, 5)) * -0.375
    x157 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x157 += einsum(x154, (0, 1, 2, 3), (0, 1, 2, 3))
    del x154
    x157 += einsum(x156, (0, 1, 2, 3), (2, 1, 0, 3))
    del x156
    x158 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x158 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 0.2
    x158 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x158 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.2
    x158 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -0.2
    x158 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.2
    x158 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -0.2
    x158 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -0.2
    x158 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    x159 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x159 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x159 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 0, 2))
    x160 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x160 += einsum(t1, (0, 1), x159, (2, 3, 4, 1, 5, 6), (0, 2, 3, 4, 5, 6))
    x161 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x161 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x161 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    x161 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    x162 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x162 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -0.3333333333333333
    x162 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2)) * 0.3333333333333333
    x162 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2))
    x162 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2)) * -0.3333333333333333
    x163 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x163 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x163 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2)) * -1.0
    x163 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2)) * -1.0
    x163 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2)) * 3.0
    x164 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x164 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    x164 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    x164 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    x165 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x165 += einsum(t2, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 1), (5, 6, 0, 4))
    x166 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x166 += einsum(x125, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (6, 5, 1, 4)) * 0.1875
    x167 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x167 += einsum(t2, (0, 1, 2, 3), l3, (4, 3, 2, 5, 6, 1), (5, 6, 0, 4))
    x168 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x168 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * 1.3333333333333333
    x168 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2)) * -1.0
    x169 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x169 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x169 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x169 += einsum(x165, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.0625
    x169 += einsum(x165, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.75
    x169 += einsum(x166, (0, 1, 2, 3), (1, 0, 2, 3))
    x169 += einsum(x166, (0, 1, 2, 3), (0, 1, 2, 3)) * -3.0
    del x166
    x169 += einsum(x167, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.0625
    x169 += einsum(x167, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.25
    x169 += einsum(x117, (0, 1, 2, 3), x168, (0, 4, 5, 2, 3, 6), (4, 5, 1, 6)) * 0.1875
    x170 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x170 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 7, 4, 1, 0, 2), (5, 3, 6, 7))
    x171 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x171 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 7, 3, 2, 1, 0), (5, 4, 6, 7))
    x172 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x172 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 7, 3, 2, 0, 1), (5, 4, 6, 7))
    x173 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x173 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 7, 3, 1, 2, 0), (5, 4, 6, 7))
    x174 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x174 += einsum(t1, (0, 1), x167, (2, 3, 4, 1), (2, 3, 0, 4))
    x175 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x175 += einsum(x172, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.16666666666666
    x175 += einsum(x173, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.16666666666666
    x175 += einsum(x174, (0, 1, 2, 3), (0, 1, 2, 3))
    del x174
    x176 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x176 += einsum(t1, (0, 1), x165, (2, 3, 4, 1), (2, 3, 0, 4))
    x177 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x177 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x177 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    x178 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x178 += einsum(l3, (0, 1, 2, 3, 4, 5), x177, (4, 6, 7, 0, 2, 1), (3, 5, 6, 7)) * 0.16666666666666
    x179 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x179 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x179 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x180 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x180 += einsum(x179, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (5, 6, 1, 4)) * 0.3333333333333333
    x180 += einsum(x130, (0, 1, 2, 3), l3, (3, 4, 2, 5, 6, 0), (6, 5, 1, 4)) * 0.3333333333333333
    x181 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x181 += einsum(x170, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.16666666666666
    x181 += einsum(x171, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.16666666666662
    x181 += einsum(x171, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.16666666666666
    x181 += einsum(x175, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x181 += einsum(x175, (0, 1, 2, 3), (1, 0, 2, 3))
    x181 += einsum(x176, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x181 += einsum(x176, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x181 += einsum(x178, (0, 1, 2, 3), (1, 0, 2, 3))
    x181 += einsum(t1, (0, 1), x180, (2, 3, 4, 1), (2, 3, 0, 4)) * 3.0
    del x180
    x182 = np.zeros((nocc, nocc), dtype=types[float])
    x182 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 6, 5, 0, 1, 2), (4, 6))
    x183 = np.zeros((nocc, nocc), dtype=types[float])
    x183 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 4, 3, 1, 2, 0), (5, 6))
    x184 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x184 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    x184 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1))
    x185 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x185 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x185 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    x186 = np.zeros((nocc, nocc), dtype=types[float])
    x186 += einsum(x184, (0, 1, 2, 3, 4, 5), x185, (1, 6, 0, 3, 4, 5), (2, 6)) * 0.041666666666665
    x187 = np.zeros((nocc, nocc), dtype=types[float])
    x187 += einsum(t3, (0, 1, 2, 3, 4, 5), x184, (1, 2, 6, 5, 4, 3), (0, 6)) * 0.041666666666665
    x188 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x188 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x188 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.14285714285714288
    x189 = np.zeros((nocc, nocc), dtype=types[float])
    x189 += einsum(l3, (0, 1, 2, 3, 4, 5), x188, (3, 6, 4, 0, 2, 1), (5, 6)) * 0.291666666666655
    x190 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x190 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x190 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x191 = np.zeros((nocc, nocc), dtype=types[float])
    x191 += einsum(l2, (0, 1, 2, 3), x190, (3, 4, 0, 1), (2, 4))
    x192 = np.zeros((nocc, nocc), dtype=types[float])
    x192 += einsum(x182, (0, 1), (0, 1)) * 0.124999999999995
    x192 += einsum(x183, (0, 1), (0, 1)) * -0.041666666666665
    x192 += einsum(x186, (0, 1), (0, 1)) * -1.0
    del x186
    x192 += einsum(x187, (0, 1), (1, 0))
    del x187
    x192 += einsum(x189, (0, 1), (0, 1))
    del x189
    x192 += einsum(x191, (0, 1), (0, 1))
    x193 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x193 += einsum(x135, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.75
    x193 += einsum(t3, (0, 1, 2, 3, 4, 5), x112, (1, 2, 6, 7, 4, 5), (6, 0, 7, 3)) * -0.125
    x193 += einsum(t3, (0, 1, 2, 3, 4, 5), x118, (1, 2, 6, 7, 5, 4), (6, 0, 7, 3)) * -0.125
    x193 += einsum(x136, (0, 1, 2, 3), (0, 1, 2, 3))
    x193 += einsum(x136, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    del x136
    x193 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x193 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3))
    x193 += einsum(x138, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x193 += einsum(x138, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x138
    x193 += einsum(x148, (0, 1, 2, 3), (0, 1, 2, 3))
    x193 += einsum(x148, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x148
    x193 += einsum(x153, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x193 += einsum(x153, (0, 1, 2, 3), (0, 2, 1, 3)) * 3.0
    del x153
    x193 += einsum(x157, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x193 += einsum(x157, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.3333333333333333
    del x157
    x193 += einsum(x112, (0, 1, 2, 3, 4, 5), x158, (0, 6, 2, 4, 7, 5), (1, 6, 3, 7)) * 0.625
    x193 += einsum(x160, (0, 1, 2, 3, 4, 5), x161, (1, 6, 2, 5, 4, 7), (3, 6, 0, 7)) * 0.125
    x193 += einsum(t2, (0, 1, 2, 3), x162, (1, 4, 5, 2, 3, 6), (4, 5, 0, 6)) * 0.75
    del x162
    x193 += einsum(t2, (0, 1, 2, 3), x163, (1, 4, 5, 3, 2, 6), (4, 5, 0, 6)) * 0.25
    del x163
    x193 += einsum(l2, (0, 1, 2, 3), x164, (2, 4, 5, 1, 6, 0), (3, 4, 5, 6)) * -0.125
    del x164
    x193 += einsum(x120, (0, 1, 2, 3), x169, (4, 0, 5, 2), (4, 1, 5, 3)) * -2.0
    del x169
    x193 += einsum(t1, (0, 1), x181, (2, 0, 3, 4), (2, 4, 3, 1)) * -0.25
    del x181
    x193 += einsum(t1, (0, 1), x192, (2, 3), (2, 0, 3, 1)) * 2.0
    l1new += einsum(v.ovov, (0, 1, 2, 3), x193, (4, 0, 2, 1), (3, 4)) * -2.0
    del x193
    x194 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x194 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 5, 3, 7, 2, 1), (4, 6, 0, 7))
    x195 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x195 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 3, 5, 7, 2, 0), (4, 6, 1, 7))
    x196 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x196 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x196 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    x197 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x197 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.5
    x197 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x198 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x198 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * 8.0
    x198 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 6, 5, 2, 7, 1), (4, 6, 0, 7))
    x198 += einsum(x111, (0, 1, 2, 3), (0, 1, 2, 3))
    del x111
    x198 += einsum(x194, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x198 += einsum(x195, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x198 += einsum(l3, (0, 1, 2, 3, 4, 5), x158, (5, 6, 4, 2, 7, 1), (3, 6, 0, 7)) * 5.0
    del x158
    x198 += einsum(x159, (0, 1, 2, 3, 4, 5), x196, (1, 6, 0, 3, 4, 7), (2, 6, 5, 7))
    x198 += einsum(x120, (0, 1, 2, 3), x197, (0, 4, 2, 5), (4, 1, 5, 3)) * 16.0
    l1new += einsum(v.ovvv, (0, 1, 2, 3), x198, (4, 0, 3, 1), (2, 4)) * 0.25
    del x198
    x199 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x199 += einsum(x117, (0, 1, 2, 3), l3, (2, 4, 3, 0, 5, 6), (6, 5, 1, 4)) * 0.5
    x200 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x200 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3))
    x200 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    x200 += einsum(x123, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x200 += einsum(x123, (0, 1, 2, 3), (1, 0, 2, 3)) * 1.5
    x200 += einsum(x124, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x200 += einsum(x124, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x200 += einsum(x125, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (5, 6, 1, 4)) * 1.5
    x200 += einsum(x199, (0, 1, 2, 3), (0, 1, 2, 3))
    x201 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x201 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    x201 += einsum(x113, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x113
    x201 += einsum(x130, (0, 1, 2, 3), x160, (4, 1, 0, 5, 2, 6), (5, 4, 6, 3)) * -1.0
    del x160
    x201 += einsum(x190, (0, 1, 2, 3), x118, (4, 1, 0, 5, 2, 6), (4, 5, 6, 3)) * 4.0
    x201 += einsum(t1, (0, 1), x200, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    del x200
    l1new += einsum(v.ovvv, (0, 1, 2, 3), x201, (4, 0, 3, 1), (2, 4)) * 0.5
    del x201
    x202 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x202 += einsum(l2, (0, 1, 2, 3), t3, (4, 3, 5, 6, 1, 0), (2, 4, 5, 6))
    x203 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x203 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    l2new += einsum(v.ovov, (0, 1, 2, 3), x203, (4, 5, 0, 2), (1, 3, 4, 5))
    x204 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x204 += einsum(t1, (0, 1), x203, (2, 0, 3, 4), (2, 3, 4, 1))
    x205 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x205 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x205 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    x205 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    x205 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    x206 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x206 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    x206 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.6666666666666666
    x206 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    x207 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x207 += einsum(x124, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.75
    del x124
    x207 += einsum(x126, (0, 1, 2, 3), (1, 0, 2, 3))
    x207 += einsum(x126, (0, 1, 2, 3), (0, 1, 2, 3)) * -3.0
    x207 += einsum(x167, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.25
    x207 += einsum(x167, (0, 1, 2, 3), (1, 0, 2, 3))
    x208 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x208 += einsum(t2, (0, 1, 2, 3), l3, (3, 4, 2, 1, 5, 6), (6, 5, 0, 4))
    x209 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x209 += einsum(x208, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    x209 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x209 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3))
    x209 += einsum(x165, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.0625
    x209 += einsum(x165, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.75
    x209 += einsum(t2, (0, 1, 2, 3), x168, (1, 4, 5, 2, 3, 6), (4, 5, 0, 6)) * -0.1875
    del x168
    x210 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x210 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * 0.4444444444444444
    x210 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 0, 2))
    x211 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x211 += einsum(x208, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x211 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x211 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3))
    x211 += einsum(x165, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.375
    x211 += einsum(x165, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    x211 += einsum(t2, (0, 1, 2, 3), x210, (1, 4, 5, 2, 3, 6), (5, 4, 0, 6)) * 1.125
    del x210
    x212 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x212 += einsum(x117, (0, 1, 2, 3), l3, (2, 4, 3, 0, 5, 6), (6, 5, 1, 4))
    x213 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x213 += einsum(x125, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (5, 6, 1, 4)) * 3.0
    x213 += einsum(x212, (0, 1, 2, 3), (0, 1, 2, 3))
    x214 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x214 += einsum(x170, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.16666666666666
    x214 += einsum(x171, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.16666666666666
    x214 += einsum(x171, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.16666666666662
    x214 += einsum(x175, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x214 += einsum(x175, (0, 1, 2, 3), (1, 0, 2, 3))
    del x175
    x214 += einsum(x176, (0, 1, 2, 3), (0, 1, 2, 3))
    x214 += einsum(x176, (0, 1, 2, 3), (1, 0, 2, 3)) * -3.0
    x214 += einsum(x178, (0, 1, 2, 3), (1, 0, 2, 3))
    del x178
    x214 += einsum(t1, (0, 1), x213, (2, 3, 4, 1), (2, 3, 0, 4))
    del x213
    x215 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x215 += einsum(x135, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x135
    x215 += einsum(t3, (0, 1, 2, 3, 4, 5), x118, (6, 2, 1, 7, 3, 5), (6, 0, 7, 4)) * 4.0
    x215 += einsum(t3, (0, 1, 2, 3, 4, 5), x118, (2, 0, 6, 7, 5, 3), (6, 1, 7, 4)) * 2.0
    x215 += einsum(x202, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    x215 += einsum(x202, (0, 1, 2, 3), (0, 2, 1, 3)) * -4.0
    del x202
    x215 += einsum(x204, (0, 1, 2, 3), (0, 1, 2, 3)) * 16.0
    x215 += einsum(x204, (0, 1, 2, 3), (0, 2, 1, 3)) * -8.0
    del x204
    x215 += einsum(x112, (0, 1, 2, 3, 4, 5), x115, (0, 6, 2, 4, 7, 5), (1, 6, 3, 7)) * 2.0
    del x115
    x215 += einsum(x112, (0, 1, 2, 3, 4, 5), x205, (1, 6, 0, 5, 7, 4), (2, 6, 3, 7)) * 2.0
    del x205
    x215 += einsum(l2, (0, 1, 2, 3), x206, (2, 4, 5, 1, 6, 0), (3, 4, 5, 6)) * -6.0
    del x206
    x215 += einsum(x117, (0, 1, 2, 3), x207, (4, 0, 5, 2), (4, 1, 5, 3)) * 4.0
    del x207
    x215 += einsum(t2, (0, 1, 2, 3), x209, (4, 1, 5, 3), (4, 0, 5, 2)) * 16.0
    del x209
    x215 += einsum(t2, (0, 1, 2, 3), x211, (4, 1, 5, 2), (4, 0, 5, 3)) * -8.0
    del x211
    x215 += einsum(t1, (0, 1), x214, (2, 0, 3, 4), (2, 4, 3, 1)) * -4.0
    del x214
    x215 += einsum(t1, (0, 1), x192, (2, 3), (2, 0, 3, 1)) * 16.0
    del x192
    l1new += einsum(v.ovov, (0, 1, 2, 3), x215, (4, 0, 2, 3), (1, 4)) * 0.125
    del x215
    x216 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x216 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 5, 2, 6, 1, 3), (4, 5, 0, 6))
    x217 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x217 += einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    x218 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x218 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x219 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x219 += einsum(t1, (0, 1), x90, (2, 3, 4, 1), (2, 0, 3, 4))
    x220 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x220 += einsum(t1, (0, 1), x87, (2, 3, 0, 4), (2, 3, 4, 1))
    del x87
    x221 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x221 += einsum(x218, (0, 1, 2, 3), (0, 1, 2, 3))
    x221 += einsum(x219, (0, 1, 2, 3), (0, 1, 2, 3))
    x221 += einsum(x220, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x220
    x222 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x222 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 2, 5, 6, 3, 1), (4, 5, 0, 6))
    x223 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x223 += einsum(x36, (0, 1), t2, (2, 3, 1, 4), (2, 3, 0, 4))
    x224 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x224 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.5
    x224 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 1.5
    x224 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    x225 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x225 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x225 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x225 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x225 += einsum(x8, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x226 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x226 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x226 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x227 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x227 += einsum(v.oooo, (0, 1, 2, 3), (0, 2, 3, 1))
    x227 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x227 += einsum(x5, (0, 1, 2, 3), (2, 0, 3, 1))
    x227 += einsum(x5, (0, 1, 2, 3), (3, 0, 2, 1)) * -0.5
    x228 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x228 += einsum(x216, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    x228 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x228 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * 0.5
    x228 += einsum(x217, (0, 1, 2, 3), (1, 2, 0, 3)) * 0.5
    x228 += einsum(x217, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x228 += einsum(x221, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x228 += einsum(x221, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    del x221
    x228 += einsum(x222, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    x228 += einsum(x222, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.25
    del x222
    x228 += einsum(x223, (0, 1, 2, 3), (1, 0, 2, 3))
    x228 += einsum(x223, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x228 += einsum(v.ovov, (0, 1, 2, 3), x224, (2, 4, 5, 3, 1, 6), (4, 5, 0, 6)) * -0.25
    del x224
    x228 += einsum(x120, (0, 1, 2, 3), x225, (4, 5, 0, 2), (4, 1, 5, 3)) * -1.0
    x228 += einsum(t1, (0, 1), x226, (2, 3, 1, 4), (0, 3, 2, 4)) * -0.5
    del x226
    x228 += einsum(t1, (0, 1), x227, (0, 2, 3, 4), (2, 4, 3, 1))
    del x227
    l1new += einsum(l2, (0, 1, 2, 3), x228, (2, 3, 4, 1), (0, 4)) * 2.0
    del x228
    x229 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x229 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -0.5
    x229 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.5
    x229 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    x230 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x230 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x230 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x230 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3))
    x230 += einsum(x8, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x231 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x231 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x231 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x232 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x232 += einsum(x5, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x232 += einsum(x5, (0, 1, 2, 3), (0, 3, 2, 1)) * 2.0
    x233 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x233 += einsum(x216, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.5
    x233 += einsum(v.ovov, (0, 1, 2, 3), x229, (2, 4, 5, 3, 1, 6), (4, 5, 0, 6)) * -0.5
    del x229
    x233 += einsum(t2, (0, 1, 2, 3), x225, (4, 5, 1, 3), (4, 0, 5, 2)) * -1.0
    del x225
    x233 += einsum(t2, (0, 1, 2, 3), x230, (4, 5, 1, 2), (4, 0, 5, 3)) * -2.0
    del x230
    x233 += einsum(t1, (0, 1), x231, (2, 3, 1, 4), (0, 3, 2, 4)) * 2.0
    del x231
    x233 += einsum(t1, (0, 1), x232, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x232
    l1new += einsum(l2, (0, 1, 2, 3), x233, (3, 2, 4, 1), (0, 4)) * -1.0
    del x233
    x234 = np.zeros((nvir, nvir), dtype=types[float])
    x234 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 4, 5, 0, 6, 2), (1, 6))
    x235 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x235 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -0.14285714285714288
    x235 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 0.14285714285714288
    x235 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    x235 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 0.14285714285714288
    x235 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -0.14285714285714288
    x235 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -0.14285714285714288
    x236 = np.zeros((nvir, nvir), dtype=types[float])
    x236 += einsum(x234, (0, 1), (0, 1)) * 3.0
    x236 += einsum(l3, (0, 1, 2, 3, 4, 5), x235, (3, 5, 4, 6, 1, 2), (0, 6)) * 6.999999999999999
    x236 += einsum(l3, (0, 1, 2, 3, 4, 5), x116, (3, 4, 5, 6, 1, 2), (0, 6)) * -1.0
    x237 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x237 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x237 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    l1new += einsum(x236, (0, 1), x237, (2, 3, 0, 1), (3, 2)) * 0.16666666666666
    del x236, x237
    x238 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x238 += einsum(x130, (0, 1, 2, 3), x159, (0, 4, 5, 2, 3, 6), (1, 4, 5, 6)) * 0.5
    x239 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x239 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x239 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.3333333333333333
    x240 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x240 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3))
    x240 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    x240 += einsum(x238, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    x240 += einsum(x179, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (5, 6, 1, 4)) * 0.5
    x240 += einsum(x239, (0, 1, 2, 3), l3, (4, 2, 3, 0, 5, 6), (5, 6, 1, 4)) * 1.5
    l1new += einsum(v.oovv, (0, 1, 2, 3), x240, (0, 4, 1, 3), (2, 4))
    del x240
    x241 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x241 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x241 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.999999999999999
    x241 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    x241 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    x241 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    x241 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    x242 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x242 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    x242 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x242 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    x243 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x243 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    x243 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 6.0
    x243 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    x243 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    x243 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x244 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x244 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3))
    x244 += einsum(x239, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 0), (5, 6, 1, 4)) * 0.75
    x245 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x245 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3))
    x245 += einsum(x126, (0, 1, 2, 3), (1, 0, 2, 3))
    x245 += einsum(x126, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x126
    x245 += einsum(x238, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x238
    x246 = np.zeros((nocc, nocc), dtype=types[float])
    x246 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    l1new += einsum(x246, (0, 1), x36, (1, 2), (2, 0)) * -2.0
    x247 = np.zeros((nocc, nocc), dtype=types[float])
    x247 += einsum(x246, (0, 1), (0, 1))
    x247 += einsum(x182, (0, 1), (0, 1)) * 0.24999999999999
    x247 += einsum(x183, (0, 1), (0, 1)) * -0.08333333333333
    x247 += einsum(x184, (0, 1, 2, 3, 4, 5), x185, (1, 6, 0, 3, 4, 5), (2, 6)) * -0.08333333333333
    x247 += einsum(t3, (0, 1, 2, 3, 4, 5), x184, (1, 2, 6, 5, 4, 3), (6, 0)) * 0.08333333333333
    x247 += einsum(l3, (0, 1, 2, 3, 4, 5), x188, (3, 6, 4, 0, 2, 1), (5, 6)) * 0.58333333333331
    x247 += einsum(l2, (0, 1, 2, 3), x190, (2, 4, 1, 0), (3, 4)) * 2.0
    x248 = np.zeros((nocc, nvir), dtype=types[float])
    x248 += einsum(t1, (0, 1), (0, 1)) * -4.0
    x248 += einsum(l2, (0, 1, 2, 3), t3, (4, 2, 3, 5, 1, 0), (4, 5)) * 2.0
    x248 += einsum(t3, (0, 1, 2, 3, 4, 5), x118, (2, 0, 1, 6, 5, 3), (6, 4)) * 0.99999999999996
    x248 += einsum(x112, (0, 1, 2, 3, 4, 5), x241, (0, 1, 2, 5, 6, 4), (3, 6)) * 0.33333333333332
    del x241
    x248 += einsum(x112, (0, 1, 2, 3, 4, 5), x242, (0, 2, 1, 4, 6, 5), (3, 6)) * -0.33333333333332
    del x242
    x248 += einsum(l2, (0, 1, 2, 3), x243, (3, 4, 2, 1, 5, 0), (4, 5)) * -1.0
    del x243
    x248 += einsum(t2, (0, 1, 2, 3), x244, (0, 1, 4, 2), (4, 3)) * 8.0
    del x244
    x248 += einsum(t2, (0, 1, 2, 3), x245, (0, 1, 4, 3), (4, 2)) * -4.0
    del x245
    x248 += einsum(l1, (0, 1), x121, (1, 2, 3, 0), (2, 3)) * -4.0
    del x121
    x248 += einsum(t1, (0, 1), x247, (0, 2), (2, 1)) * 4.0
    del x247
    x249 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x249 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * 2.0
    x249 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l1new += einsum(x248, (0, 1), x249, (0, 2, 3, 1), (3, 2)) * -0.25
    del x248, x249
    x250 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x250 += einsum(x172, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.041666666666665
    x250 += einsum(x173, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.041666666666665
    x251 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x251 += einsum(x130, (0, 1, 2, 3), x159, (0, 4, 5, 2, 3, 6), (4, 5, 1, 6)) * -1.0
    del x159
    x251 += einsum(x179, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (5, 6, 1, 4))
    del x179
    x251 += einsum(x239, (0, 1, 2, 3), l3, (4, 2, 3, 0, 5, 6), (5, 6, 1, 4)) * 3.0
    del x239
    x252 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x252 += einsum(x170, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.041666666666665
    x252 += einsum(x203, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x252 += einsum(x203, (0, 1, 2, 3), (1, 0, 3, 2))
    del x203
    x252 += einsum(x137, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x252 += einsum(x137, (0, 1, 2, 3), (0, 3, 2, 1))
    del x137
    x252 += einsum(x171, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.291666666666655
    x252 += einsum(x171, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.041666666666665
    x252 += einsum(x250, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x252 += einsum(x250, (0, 1, 2, 3), (1, 0, 2, 3))
    del x250
    x252 += einsum(l3, (0, 1, 2, 3, 4, 5), x177, (4, 6, 7, 0, 2, 1), (5, 3, 6, 7)) * 0.041666666666665
    x252 += einsum(t1, (0, 1), x251, (2, 3, 4, 1), (2, 0, 4, 3)) * -0.25
    del x251
    l1new += einsum(v.ooov, (0, 1, 2, 3), x252, (4, 1, 2, 0), (3, 4)) * 2.0
    del x252
    x253 = np.zeros((nvir, nvir), dtype=types[float])
    x253 += einsum(l1, (0, 1), t1, (1, 2), (0, 2)) * 0.5
    x253 += einsum(l2, (0, 1, 2, 3), x120, (2, 3, 4, 1), (0, 4))
    x254 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x254 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x254 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    l1new += einsum(x253, (0, 1), x254, (2, 1, 0, 3), (3, 2)) * 4.0
    del x253, x254
    x255 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x255 += einsum(x172, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.14285714285714288
    del x172
    x255 += einsum(x173, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.14285714285714288
    del x173
    x256 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x256 += einsum(x170, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.14285714285714288
    del x170
    x256 += einsum(x171, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.14285714285714288
    x256 += einsum(x171, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x256 += einsum(x139, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.4285714285714286
    x256 += einsum(x139, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.4285714285714286
    del x139
    x256 += einsum(x255, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x256 += einsum(x255, (0, 1, 2, 3), (1, 0, 2, 3))
    del x255
    x256 += einsum(l3, (0, 1, 2, 3, 4, 5), x177, (4, 6, 7, 0, 2, 1), (5, 3, 6, 7)) * 0.14285714285714288
    del x177
    l1new += einsum(v.ooov, (0, 1, 2, 3), x256, (4, 1, 0, 2), (3, 4)) * -0.58333333333331
    del x256
    x257 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x257 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * 0.3333333333333333
    x257 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2)) * -0.3333333333333333
    x257 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1))
    x257 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1)) * -0.3333333333333333
    x258 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x258 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    x258 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    x258 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1)) * -1.0
    x258 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1)) * 3.0
    x259 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x259 += einsum(t2, (0, 1, 2, 3), x257, (1, 4, 5, 6, 3, 2), (5, 4, 0, 6))
    del x257
    x259 += einsum(t2, (0, 1, 2, 3), x258, (1, 4, 5, 6, 2, 3), (5, 4, 0, 6)) * 0.3333333333333333
    del x258
    x260 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x260 += einsum(t1, (0, 1), x259, (2, 3, 4, 1), (2, 3, 0, 4)) * 3.0
    del x259
    l1new += einsum(v.ooov, (0, 1, 2, 3), x260, (4, 0, 2, 1), (3, 4)) * 0.5
    del x260
    x261 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x261 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 3.0
    x261 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4))
    x261 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    x262 = np.zeros((nocc, nocc), dtype=types[float])
    x262 += einsum(x246, (0, 1), (0, 1)) * 0.5
    x262 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 3, 5, 1, 2, 0), (4, 6)) * -0.041666666666665
    x262 += einsum(l3, (0, 1, 2, 3, 4, 5), x261, (3, 5, 6, 0, 2, 1), (4, 6)) * 0.041666666666665
    del x261
    x262 += einsum(x191, (0, 1), (0, 1))
    del x191
    x263 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x263 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x263 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    l1new += einsum(x262, (0, 1), x263, (0, 2, 1, 3), (3, 2)) * -2.0
    del x262
    x264 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x264 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x264 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    x264 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 6.999999999999999
    x264 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    x265 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x265 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x265 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    x266 = np.zeros((nocc, nocc), dtype=types[float])
    x266 += einsum(l3, (0, 1, 2, 3, 4, 5), x264, (4, 6, 3, 0, 2, 1), (5, 6))
    del x264
    x266 += einsum(l3, (0, 1, 2, 3, 4, 5), x265, (4, 6, 3, 0, 1, 2), (5, 6)) * -1.0
    x267 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x267 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x267 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    l1new += einsum(x266, (0, 1), x267, (0, 2, 1, 3), (3, 2)) * -0.16666666666666
    del x266, x267
    x268 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x268 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x268 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    l1new += einsum(l1, (0, 1), x268, (1, 2, 0, 3), (3, 2)) * 2.0
    del x268
    x269 = np.zeros((nocc, nocc), dtype=types[float])
    x269 += einsum(x246, (0, 1), (0, 1)) * 1.714285714285783
    x269 += einsum(x182, (0, 1), (0, 1)) * 0.4285714285714286
    x269 += einsum(x183, (0, 1), (0, 1)) * -0.14285714285714288
    x269 += einsum(x184, (0, 1, 2, 3, 4, 5), x185, (1, 6, 0, 3, 4, 5), (2, 6)) * -0.14285714285714288
    x269 += einsum(t3, (0, 1, 2, 3, 4, 5), x184, (1, 2, 6, 5, 4, 3), (6, 0)) * 0.14285714285714288
    x269 += einsum(l3, (0, 1, 2, 3, 4, 5), x188, (3, 6, 4, 0, 2, 1), (5, 6))
    x269 += einsum(l2, (0, 1, 2, 3), x190, (2, 4, 1, 0), (3, 4)) * 3.428571428571566
    l1new += einsum(f.ov, (0, 1), x269, (2, 0), (1, 2)) * -0.58333333333331
    del x269
    x270 = np.zeros((nvir, nvir), dtype=types[float])
    x270 += einsum(f.vv, (0, 1), (0, 1)) * 0.5
    x270 += einsum(t1, (0, 1), x54, (0, 1, 2, 3), (3, 2))
    l1new += einsum(l1, (0, 1), x270, (0, 2), (2, 1)) * 2.0
    del x270
    x271 = np.zeros((nocc, nocc), dtype=types[float])
    x271 += einsum(v.ovov, (0, 1, 2, 3), x190, (2, 4, 1, 3), (0, 4)) * 2.0
    x272 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x272 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x272 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -0.5
    x273 = np.zeros((nocc, nocc), dtype=types[float])
    x273 += einsum(t1, (0, 1), x272, (2, 3, 0, 1), (2, 3)) * 2.0
    del x272
    x274 = np.zeros((nocc, nocc), dtype=types[float])
    x274 += einsum(t1, (0, 1), x68, (2, 1), (0, 2))
    x275 = np.zeros((nocc, nocc), dtype=types[float])
    x275 += einsum(f.oo, (0, 1), (0, 1))
    x275 += einsum(x271, (0, 1), (1, 0))
    x275 += einsum(x273, (0, 1), (1, 0))
    x275 += einsum(x274, (0, 1), (0, 1))
    del x274
    l1new += einsum(l1, (0, 1), x275, (1, 2), (0, 2)) * -1.0
    l3new += einsum(x275, (0, 1), l3, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -1.0
    del x275
    x276 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x276 += einsum(f.vv, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x277 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x277 += einsum(l1, (0, 1), v.ovvv, (2, 3, 4, 0), (1, 2, 3, 4))
    x278 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x278 += einsum(l1, (0, 1), x8, (1, 2, 3, 4), (2, 3, 0, 4))
    x279 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x279 += einsum(l2, (0, 1, 2, 3), x5, (2, 4, 3, 5), (4, 5, 0, 1))
    x280 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x280 += einsum(x217, (0, 1, 2, 3), l3, (4, 5, 3, 6, 2, 1), (0, 6, 4, 5))
    x281 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x281 += einsum(x122, (0, 1, 2, 3), x8, (0, 2, 4, 5), (1, 4, 3, 5))
    x282 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x282 += einsum(v.ovvv, (0, 1, 2, 3), x123, (4, 5, 0, 3), (4, 5, 1, 2))
    x283 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x283 += einsum(t1, (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x284 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x284 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x284 += einsum(x8, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x285 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x285 += einsum(t2, (0, 1, 2, 3), x284, (0, 4, 1, 5), (4, 2, 3, 5))
    del x284
    x286 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x286 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x286 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x287 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x287 += einsum(x286, (0, 1, 2, 3), t3, (4, 1, 0, 5, 6, 2), (4, 5, 6, 3)) * 0.49999999999998
    del x286
    x288 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x288 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x288 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.3333333333333333
    x289 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x289 += einsum(x288, (0, 1, 2, 3), t3, (4, 1, 0, 5, 2, 6), (4, 6, 5, 3)) * 1.49999999999994
    del x288
    x290 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x290 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x290 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x290 += einsum(x283, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x290 += einsum(x283, (0, 1, 2, 3), (0, 3, 2, 1))
    x290 += einsum(x285, (0, 1, 2, 3), (0, 1, 2, 3))
    x290 += einsum(x285, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x285
    x290 += einsum(x287, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x287
    x290 += einsum(x289, (0, 1, 2, 3), (0, 1, 2, 3))
    del x289
    x291 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x291 += einsum(x290, (0, 1, 2, 3), l3, (2, 4, 1, 0, 5, 6), (6, 5, 4, 3)) * 0.5
    del x290
    x292 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x292 += einsum(x16, (0, 1, 2, 3), t3, (4, 1, 0, 5, 6, 2), (4, 5, 6, 3)) * 0.24999999999999
    x293 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x293 += einsum(v.ovvv, (0, 1, 2, 3), x120, (0, 4, 1, 5), (4, 2, 3, 5)) * 2.0
    x294 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x294 += einsum(v.ovvv, (0, 1, 2, 3), x117, (0, 4, 3, 5), (4, 1, 2, 5))
    x295 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x295 += einsum(x292, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x292
    x295 += einsum(x293, (0, 1, 2, 3), (0, 2, 3, 1))
    del x293
    x295 += einsum(x294, (0, 1, 2, 3), (0, 2, 3, 1))
    del x294
    x296 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x296 += einsum(x295, (0, 1, 2, 3), l3, (1, 4, 2, 0, 5, 6), (6, 5, 4, 3))
    del x295
    x297 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x297 += einsum(t1, (0, 1), v.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    x298 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x298 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x298 += einsum(x297, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    del x297
    x299 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x299 += einsum(t1, (0, 1), x65, (2, 3, 0, 4), (3, 2, 4, 1))
    x300 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x300 += einsum(x218, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x300 += einsum(x299, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x299
    x300 += einsum(x223, (0, 1, 2, 3), (1, 0, 2, 3))
    del x223
    x301 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x301 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -0.5
    x301 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.5
    x301 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    x301 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    x302 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x302 += einsum(v.ovov, (0, 1, 2, 3), x301, (2, 4, 5, 3, 1, 6), (0, 4, 5, 6)) * 0.49999999999998
    del x301
    x303 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x303 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x303 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x304 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x304 += einsum(t2, (0, 1, 2, 3), x303, (4, 5, 1, 3), (0, 4, 5, 2))
    del x303
    x305 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x305 += einsum(t2, (0, 1, 2, 3), x1, (4, 5, 1, 2), (0, 4, 5, 3))
    x306 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x306 += einsum(x216, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.49999999999998
    x306 += einsum(x298, (0, 1, 2, 3), (0, 2, 1, 3))
    x306 += einsum(x298, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    del x298
    x306 += einsum(x300, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x306 += einsum(x300, (0, 1, 2, 3), (1, 0, 2, 3))
    del x300
    x306 += einsum(x302, (0, 1, 2, 3), (1, 2, 0, 3))
    del x302
    x306 += einsum(x304, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x304
    x306 += einsum(x305, (0, 1, 2, 3), (0, 2, 1, 3))
    del x305
    x307 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x307 += einsum(x306, (0, 1, 2, 3), l3, (4, 5, 3, 0, 6, 1), (6, 2, 4, 5))
    del x306
    x308 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x308 += einsum(t1, (0, 1), x20, (2, 3, 4, 0), (2, 3, 4, 1))
    x309 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x309 += einsum(x219, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x219
    x309 += einsum(x308, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x308
    x310 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x310 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x310 += einsum(x8, (0, 1, 2, 3), (0, 2, 1, 3))
    x311 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x311 += einsum(t2, (0, 1, 2, 3), x310, (4, 5, 1, 3), (0, 4, 5, 2)) * 2.0
    del x310
    x312 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x312 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3))
    x312 += einsum(x8, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x313 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x313 += einsum(t2, (0, 1, 2, 3), x312, (4, 5, 1, 2), (0, 4, 5, 3))
    del x312
    x314 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x314 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x314 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x315 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x315 += einsum(t1, (0, 1), x314, (2, 3, 1, 4), (0, 2, 3, 4))
    del x314
    x316 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x316 += einsum(x5, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x316 += einsum(x5, (0, 1, 2, 3), (0, 3, 2, 1))
    x317 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x317 += einsum(t1, (0, 1), x316, (2, 3, 4, 0), (2, 3, 4, 1))
    del x316
    x318 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x318 += einsum(x309, (0, 1, 2, 3), (0, 1, 2, 3))
    x318 += einsum(x309, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x309
    x318 += einsum(x311, (0, 1, 2, 3), (1, 0, 2, 3))
    del x311
    x318 += einsum(x313, (0, 1, 2, 3), (1, 0, 2, 3))
    del x313
    x318 += einsum(x315, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x315
    x318 += einsum(x317, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x317
    x319 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x319 += einsum(x318, (0, 1, 2, 3), l3, (4, 5, 3, 0, 6, 1), (6, 2, 4, 5))
    del x318
    x320 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x320 += einsum(t2, (0, 1, 2, 3), x14, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x14
    x321 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x321 += einsum(t2, (0, 1, 2, 3), x16, (1, 4, 2, 5), (0, 4, 3, 5))
    del x16
    x322 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x322 += einsum(t1, (0, 1), x1, (0, 2, 3, 4), (2, 3, 1, 4)) * 0.5
    del x1
    x323 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x323 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x323 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x323 += einsum(x320, (0, 1, 2, 3), (0, 1, 2, 3))
    del x320
    x323 += einsum(x321, (0, 1, 2, 3), (0, 1, 2, 3))
    del x321
    x323 += einsum(x322, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x322
    x324 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x324 += einsum(x323, (0, 1, 2, 3), x112, (0, 4, 5, 1, 2, 6), (4, 5, 6, 3))
    del x323
    x325 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x325 += einsum(t1, (0, 1), x22, (2, 1, 3, 4), (0, 2, 3, 4))
    del x22
    x326 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x326 += einsum(t1, (0, 1), x10, (2, 3, 0, 4), (2, 3, 1, 4)) * 0.5
    del x10
    x327 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x327 += einsum(x325, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x325
    x327 += einsum(x326, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x326
    x328 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x328 += einsum(x327, (0, 1, 2, 3), x112, (0, 4, 5, 1, 2, 6), (4, 5, 6, 3))
    del x327
    x329 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x329 += einsum(x130, (0, 1, 2, 3), l3, (4, 5, 2, 1, 6, 0), (6, 4, 5, 3))
    x330 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x330 += einsum(x149, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x330 += einsum(x329, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x329
    x331 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x331 += einsum(v.ovvv, (0, 1, 2, 3), x330, (4, 5, 3, 1), (0, 4, 2, 5)) * 0.5
    del x330
    x332 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x332 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), (3, 4, 0, 5))
    x333 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x333 += einsum(x130, (0, 1, 2, 3), x118, (0, 1, 4, 5, 6, 2), (4, 5, 6, 3))
    del x130
    x334 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x334 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -0.5
    x334 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2))
    x335 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x335 += einsum(t2, (0, 1, 2, 3), x334, (1, 4, 5, 2, 3, 6), (0, 4, 5, 6)) * 2.0
    del x334
    x336 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x336 += einsum(x208, (0, 1, 2, 3), (1, 0, 2, 3))
    del x208
    x336 += einsum(x335, (0, 1, 2, 3), (2, 1, 0, 3))
    del x335
    x337 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x337 += einsum(t1, (0, 1), x336, (0, 2, 3, 4), (2, 3, 1, 4))
    x338 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x338 += einsum(x332, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x332
    x338 += einsum(x109, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.99999999999996
    del x109
    x338 += einsum(x110, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.99999999999996
    del x110
    x338 += einsum(x114, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x114
    x338 += einsum(x333, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x333
    x338 += einsum(x337, (0, 1, 2, 3), (0, 1, 3, 2))
    del x337
    x339 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x339 += einsum(v.ovov, (0, 1, 2, 3), x338, (4, 2, 5, 1), (0, 4, 3, 5)) * 0.5
    del x338
    x340 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x340 += einsum(x155, (0, 1, 2, 3), x40, (0, 4, 5, 3), (4, 5, 1, 2)) * -0.5
    x341 = np.zeros((nvir, nvir), dtype=types[float])
    x341 += einsum(l3, (0, 1, 2, 3, 4, 5), x235, (3, 5, 4, 6, 1, 2), (0, 6)) * 2.333333333333333
    del x235
    x342 = np.zeros((nvir, nvir), dtype=types[float])
    x342 += einsum(l3, (0, 1, 2, 3, 4, 5), x116, (3, 4, 5, 6, 1, 2), (0, 6)) * 0.3333333333333333
    del x116
    x343 = np.zeros((nvir, nvir), dtype=types[float])
    x343 += einsum(l2, (0, 1, 2, 3), x120, (2, 3, 4, 1), (0, 4)) * 8.00000000000032
    del x120
    x344 = np.zeros((nvir, nvir), dtype=types[float])
    x344 += einsum(x234, (0, 1), (0, 1))
    del x234
    x344 += einsum(x341, (0, 1), (0, 1))
    del x341
    x344 += einsum(x342, (0, 1), (0, 1)) * -1.0
    del x342
    x344 += einsum(x343, (0, 1), (0, 1))
    del x343
    x345 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x345 += einsum(x344, (0, 1), v.ovov, (2, 1, 3, 4), (2, 3, 4, 0)) * 0.24999999999999
    del x344
    x346 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x346 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x346 += einsum(x90, (0, 1, 2, 3), (0, 1, 3, 2))
    x347 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x347 += einsum(l2, (0, 1, 2, 3), x346, (2, 4, 1, 5), (3, 4, 0, 5))
    del x346
    x348 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x348 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x348 += einsum(x123, (0, 1, 2, 3), (0, 1, 2, 3))
    del x123
    x348 += einsum(x199, (0, 1, 2, 3), (1, 0, 2, 3))
    x349 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x349 += einsum(v.ooov, (0, 1, 2, 3), x348, (1, 4, 2, 5), (0, 4, 3, 5))
    del x348
    x350 = np.zeros((nocc, nocc), dtype=types[float])
    x350 += einsum(x184, (0, 1, 2, 3, 4, 5), x185, (1, 6, 0, 3, 4, 5), (2, 6)) * 0.3333333333333333
    del x185
    x351 = np.zeros((nocc, nocc), dtype=types[float])
    x351 += einsum(t3, (0, 1, 2, 3, 4, 5), x184, (1, 2, 6, 5, 4, 3), (0, 6)) * 0.3333333333333333
    del x184
    x352 = np.zeros((nocc, nocc), dtype=types[float])
    x352 += einsum(l3, (0, 1, 2, 3, 4, 5), x188, (3, 6, 4, 0, 2, 1), (5, 6)) * 2.333333333333333
    del x188
    x353 = np.zeros((nocc, nocc), dtype=types[float])
    x353 += einsum(l2, (0, 1, 2, 3), x190, (3, 4, 0, 1), (2, 4)) * 8.00000000000032
    x354 = np.zeros((nocc, nocc), dtype=types[float])
    x354 += einsum(x246, (0, 1), (0, 1)) * 4.00000000000016
    del x246
    x354 += einsum(x182, (0, 1), (0, 1))
    del x182
    x354 += einsum(x183, (0, 1), (0, 1)) * -0.3333333333333333
    del x183
    x354 += einsum(x350, (0, 1), (0, 1)) * -1.0
    del x350
    x354 += einsum(x351, (0, 1), (1, 0))
    del x351
    x354 += einsum(x352, (0, 1), (0, 1))
    del x352
    x354 += einsum(x353, (0, 1), (0, 1))
    del x353
    x355 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x355 += einsum(x354, (0, 1), v.ovov, (2, 3, 1, 4), (2, 0, 4, 3)) * 0.24999999999999
    del x354
    x356 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x356 += einsum(t1, (0, 1), x212, (2, 3, 4, 1), (0, 2, 3, 4)) * -1.0
    x357 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x357 += einsum(v.ovov, (0, 1, 2, 3), x356, (0, 4, 5, 2), (4, 5, 3, 1)) * 0.5
    del x356
    x358 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x358 += einsum(x36, (0, 1), x212, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    x359 = np.zeros((nocc, nocc), dtype=types[float])
    x359 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x360 = np.zeros((nocc, nocc), dtype=types[float])
    x360 += einsum(f.oo, (0, 1), (0, 1))
    x360 += einsum(x359, (0, 1), (1, 0))
    del x359
    x361 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x361 += einsum(x360, (0, 1), l2, (2, 3, 0, 4), (4, 1, 2, 3))
    x362 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x362 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3))
    x362 += einsum(x165, (0, 1, 2, 3), (0, 1, 2, 3))
    x363 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x363 += einsum(f.ov, (0, 1), x362, (2, 3, 0, 4), (2, 3, 4, 1))
    x364 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x364 += einsum(x276, (0, 1, 2, 3), (0, 1, 2, 3))
    del x276
    x364 += einsum(x277, (0, 1, 2, 3), (0, 1, 2, 3))
    del x277
    x364 += einsum(x278, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x278
    x364 += einsum(x279, (0, 1, 2, 3), (0, 1, 2, 3))
    del x279
    x364 += einsum(x280, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x280
    x364 += einsum(x281, (0, 1, 2, 3), (0, 1, 2, 3))
    del x281
    x364 += einsum(x282, (0, 1, 2, 3), (0, 1, 2, 3))
    del x282
    x364 += einsum(x291, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x291
    x364 += einsum(x296, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x296
    x364 += einsum(x307, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x307
    x364 += einsum(x319, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x319
    x364 += einsum(x324, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x324
    x364 += einsum(x328, (0, 1, 2, 3), (0, 1, 2, 3))
    del x328
    x364 += einsum(x331, (0, 1, 2, 3), (1, 0, 3, 2))
    del x331
    x364 += einsum(x339, (0, 1, 2, 3), (1, 0, 3, 2))
    del x339
    x364 += einsum(x340, (0, 1, 2, 3), (0, 1, 2, 3))
    del x340
    x364 += einsum(x345, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x345
    x364 += einsum(x347, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x347
    x364 += einsum(x349, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x349
    x364 += einsum(x355, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x355
    x364 += einsum(x357, (0, 1, 2, 3), (0, 1, 3, 2))
    del x357
    x364 += einsum(x358, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x358
    x364 += einsum(x361, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x361
    x364 += einsum(x363, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x363
    l2new += einsum(x364, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new += einsum(x364, (0, 1, 2, 3), (2, 3, 1, 0))
    del x364
    x365 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x365 += einsum(l1, (0, 1), v.ooov, (2, 1, 3, 4), (2, 3, 0, 4))
    x366 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x366 += einsum(l2, (0, 1, 2, 3), x99, (2, 4, 5, 1), (3, 4, 0, 5))
    x367 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x367 += einsum(x122, (0, 1, 2, 3), x8, (1, 2, 4, 5), (0, 4, 3, 5))
    x368 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x368 += einsum(v.ooov, (0, 1, 2, 3), x165, (4, 1, 2, 5), (4, 0, 5, 3))
    x369 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x369 += einsum(v.ooov, (0, 1, 2, 3), x149, (1, 4, 5, 3), (0, 2, 4, 5))
    del x149
    x370 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x370 += einsum(x99, (0, 1, 2, 3), x118, (0, 4, 5, 1, 3, 6), (4, 5, 6, 2))
    x371 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x371 += einsum(l3, (0, 1, 2, 3, 4, 5), x57, (5, 3, 4, 6, 7, 2), (6, 7, 0, 1))
    del x57
    x372 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x372 += einsum(t2, (0, 1, 2, 3), l3, (4, 5, 2, 6, 1, 0), (6, 4, 5, 3))
    x373 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x373 += einsum(x372, (0, 1, 2, 3), x8, (0, 4, 5, 3), (5, 4, 1, 2))
    x374 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x374 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 0, 1, 5), (4, 2, 3, 5))
    x375 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x375 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 2), (0, 3, 4, 5))
    x376 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x376 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 0, 2, 5, 6, 3), (4, 5, 6, 1))
    x377 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x377 += einsum(t2, (0, 1, 2, 3), x8, (4, 0, 1, 5), (4, 2, 3, 5))
    x378 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x378 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    x378 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x379 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x379 += einsum(v.ovov, (0, 1, 2, 3), x378, (0, 2, 4, 5, 3, 6), (4, 1, 5, 6)) * 0.49999999999998
    x380 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x380 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x380 += einsum(x283, (0, 1, 2, 3), (0, 3, 2, 1))
    del x283
    x380 += einsum(x374, (0, 1, 2, 3), (0, 1, 2, 3))
    del x374
    x380 += einsum(x375, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    del x375
    x380 += einsum(x376, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.99999999999996
    del x376
    x380 += einsum(x377, (0, 1, 2, 3), (0, 2, 1, 3))
    del x377
    x380 += einsum(x379, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    del x379
    x381 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x381 += einsum(x380, (0, 1, 2, 3), l3, (4, 2, 1, 5, 6, 0), (5, 6, 4, 3))
    del x380
    x382 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x382 += einsum(v.ovov, (0, 1, 2, 3), x378, (2, 4, 5, 3, 1, 6), (0, 4, 5, 6)) * 0.49999999999998
    del x378
    x383 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x383 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x383 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x383 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x383 += einsum(x8, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x384 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x384 += einsum(t2, (0, 1, 2, 3), x383, (4, 1, 5, 3), (0, 4, 5, 2))
    del x383
    x385 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x385 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x385 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3))
    x386 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x386 += einsum(t2, (0, 1, 2, 3), x385, (4, 1, 5, 2), (0, 4, 5, 3))
    x387 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x387 += einsum(x67, (0, 1), t2, (2, 3, 1, 4), (2, 3, 0, 4))
    x388 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x388 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x388 += einsum(x5, (0, 1, 2, 3), (2, 1, 3, 0))
    x388 += einsum(x65, (0, 1, 2, 3), (3, 1, 2, 0))
    x389 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x389 += einsum(t1, (0, 1), x388, (0, 2, 3, 4), (2, 3, 4, 1))
    del x388
    x390 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x390 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x390 += einsum(x8, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x390 += einsum(x218, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x218
    x390 += einsum(x216, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.99999999999996
    del x216
    x390 += einsum(x382, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x382
    x390 += einsum(x384, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x384
    x390 += einsum(x386, (0, 1, 2, 3), (0, 1, 2, 3))
    del x386
    x390 += einsum(x387, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x387
    x390 += einsum(x389, (0, 1, 2, 3), (0, 2, 1, 3))
    del x389
    x391 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x391 += einsum(x390, (0, 1, 2, 3), l3, (4, 5, 3, 6, 1, 0), (6, 2, 4, 5))
    del x390
    x392 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x392 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 2, 4, 5), (0, 3, 4, 5))
    x393 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x393 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x393 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x394 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x394 += einsum(t2, (0, 1, 2, 3), x393, (1, 3, 4, 5), (0, 2, 4, 5)) * 0.5
    x395 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x395 += einsum(x392, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    del x392
    x395 += einsum(x394, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x394
    x396 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x396 += einsum(x395, (0, 1, 2, 3), l3, (4, 3, 1, 5, 6, 0), (5, 6, 4, 2)) * 2.0
    del x395
    x397 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x397 += einsum(t2, (0, 1, 2, 3), x385, (4, 5, 1, 2), (0, 4, 5, 3))
    del x385
    x398 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x398 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x398 += einsum(x90, (0, 1, 2, 3), (1, 0, 3, 2))
    x399 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x399 += einsum(t1, (0, 1), x398, (2, 3, 1, 4), (0, 2, 3, 4))
    del x398
    x400 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x400 += einsum(t1, (0, 1), x40, (2, 3, 4, 1), (0, 2, 3, 4))
    del x40
    x401 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x401 += einsum(t1, (0, 1), x400, (2, 3, 4, 0), (2, 4, 3, 1))
    del x400
    x402 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x402 += einsum(x397, (0, 1, 2, 3), (1, 0, 2, 3))
    del x397
    x402 += einsum(x399, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x399
    x402 += einsum(x401, (0, 1, 2, 3), (0, 2, 1, 3))
    del x401
    x403 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x403 += einsum(x402, (0, 1, 2, 3), l3, (4, 5, 3, 6, 1, 0), (6, 2, 4, 5))
    del x402
    x404 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x404 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x405 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x405 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    x406 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x406 += einsum(t2, (0, 1, 2, 3), x43, (1, 4, 3, 5), (0, 4, 2, 5))
    del x43
    x407 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x407 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x407 += einsum(x404, (0, 1, 2, 3), (0, 1, 2, 3))
    del x404
    x407 += einsum(x405, (0, 1, 2, 3), (0, 1, 2, 3))
    del x405
    x407 += einsum(x406, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x406
    x408 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x408 += einsum(x407, (0, 1, 2, 3), x118, (4, 0, 5, 1, 6, 2), (4, 5, 6, 3))
    del x407
    x409 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x409 += einsum(t2, (0, 1, 2, 3), l3, (4, 5, 3, 1, 6, 0), (6, 4, 5, 2))
    x410 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x410 += einsum(t2, (0, 1, 2, 3), l3, (4, 2, 5, 1, 6, 0), (6, 4, 5, 3))
    x411 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x411 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x411 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2)) * -1.0
    x411 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2)) * 0.5
    x411 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2)) * 2.0
    x412 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x412 += einsum(t2, (0, 1, 2, 3), x411, (0, 4, 1, 2, 5, 6), (4, 3, 5, 6))
    del x411
    x413 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x413 += einsum(x409, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    del x409
    x413 += einsum(x410, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x413 += einsum(x410, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x413 += einsum(x412, (0, 1, 2, 3), (0, 2, 3, 1))
    del x412
    x414 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x414 += einsum(v.ovvv, (0, 1, 2, 3), x413, (4, 5, 2, 3), (0, 4, 1, 5))
    del x413
    x415 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x415 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 5.0
    x415 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    x415 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    x415 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x415 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    x415 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 5.0
    x416 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x416 += einsum(l3, (0, 1, 2, 3, 4, 5), x415, (5, 6, 4, 2, 7, 1), (3, 6, 0, 7))
    del x415
    x417 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x417 += einsum(l3, (0, 1, 2, 3, 4, 5), x161, (5, 6, 3, 0, 2, 7), (4, 6, 1, 7))
    del x161
    x418 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x418 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x418 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 1, 2)) * -1.0
    x419 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x419 += einsum(t3, (0, 1, 2, 3, 4, 5), x418, (1, 2, 6, 5, 7, 4), (0, 6, 3, 7))
    del x418
    x420 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x420 += einsum(l3, (0, 1, 2, 3, 4, 5), x265, (5, 6, 4, 2, 1, 7), (3, 6, 0, 7))
    del x265
    x421 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x421 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x421 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    x422 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x422 += einsum(l3, (0, 1, 2, 3, 4, 5), x421, (3, 6, 5, 2, 1, 7), (4, 6, 0, 7))
    del x421
    x423 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x423 += einsum(x118, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.5
    x423 += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -0.5
    x423 += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    x423 += einsum(x112, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -0.25
    x423 += einsum(x112, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * 0.25
    x424 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x424 += einsum(t2, (0, 1, 2, 3), x423, (4, 0, 1, 5, 6, 2), (4, 5, 3, 6)) * 8.00000000000032
    del x423
    x425 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x425 += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    x425 += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    x426 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x426 += einsum(t2, (0, 1, 2, 3), x425, (0, 1, 4, 5, 6, 3), (4, 5, 2, 6)) * 2.00000000000008
    del x425
    x427 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x427 += einsum(x197, (0, 1, 2, 3), x77, (0, 4, 2, 5), (4, 1, 5, 3)) * 8.00000000000032
    del x197
    x428 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x428 += einsum(x125, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (6, 5, 1, 4))
    del x125
    x429 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x429 += einsum(x117, (0, 1, 2, 3), l3, (2, 4, 3, 0, 5, 6), (6, 5, 1, 4)) * 0.3333333333333333
    x430 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x430 += einsum(x165, (0, 1, 2, 3), (0, 1, 2, 3))
    x430 += einsum(x165, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.3333333333333333
    x430 += einsum(x167, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.3333333333333333
    x430 += einsum(x167, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.3333333333333333
    x430 += einsum(x428, (0, 1, 2, 3), (1, 0, 2, 3))
    x430 += einsum(x429, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x429
    x431 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x431 += einsum(t1, (0, 1), x430, (2, 0, 3, 4), (2, 3, 1, 4)) * 6.00000000000024
    del x430
    x432 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x432 += einsum(x194, (0, 1, 2, 3), (0, 1, 2, 3))
    del x194
    x432 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * 4.00000000000016
    x432 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -8.00000000000032
    x432 += einsum(x416, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x416
    x432 += einsum(x417, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x417
    x432 += einsum(x419, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x419
    x432 += einsum(x420, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x420
    x432 += einsum(x422, (0, 1, 2, 3), (0, 1, 2, 3))
    del x422
    x432 += einsum(x424, (0, 1, 2, 3), (0, 1, 3, 2))
    del x424
    x432 += einsum(x426, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x426
    x432 += einsum(x427, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x427
    x432 += einsum(x431, (0, 1, 2, 3), (0, 1, 3, 2))
    del x431
    x433 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x433 += einsum(v.ovov, (0, 1, 2, 3), x432, (4, 2, 5, 3), (0, 4, 1, 5)) * 0.24999999999999
    del x432
    x434 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x434 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 5, 1), (3, 4, 0, 5))
    x435 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x435 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 6, 5, 1, 7, 2), (4, 6, 0, 7))
    x436 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x436 += einsum(t2, (0, 1, 2, 3), x118, (4, 0, 1, 5, 6, 2), (4, 5, 6, 3))
    x437 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x437 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    x437 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    x437 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x438 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x438 += einsum(l3, (0, 1, 2, 3, 4, 5), x437, (4, 6, 5, 1, 7, 2), (3, 6, 0, 7))
    del x437
    x439 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x439 += einsum(l3, (0, 1, 2, 3, 4, 5), x196, (5, 6, 3, 0, 2, 7), (4, 6, 1, 7))
    del x196
    x440 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x440 += einsum(x117, (0, 1, 2, 3), x112, (0, 1, 4, 5, 2, 6), (4, 5, 6, 3)) * 2.00000000000008
    x441 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x441 += einsum(l2, (0, 1, 2, 3), x77, (3, 4, 1, 5), (2, 4, 0, 5)) * 4.00000000000016
    x442 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x442 += einsum(t1, (0, 1), x336, (2, 0, 3, 4), (2, 3, 1, 4)) * 2.00000000000008
    del x336
    x443 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x443 += einsum(x434, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.00000000000016
    del x434
    x443 += einsum(x435, (0, 1, 2, 3), (0, 1, 2, 3))
    del x435
    x443 += einsum(x195, (0, 1, 2, 3), (0, 1, 2, 3))
    del x195
    x443 += einsum(x436, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.00000000000016
    del x436
    x443 += einsum(x438, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x438
    x443 += einsum(x439, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x439
    x443 += einsum(x440, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x440
    x443 += einsum(x441, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x441
    x443 += einsum(x442, (0, 1, 2, 3), (0, 1, 3, 2))
    del x442
    x444 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x444 += einsum(v.ovov, (0, 1, 2, 3), x443, (4, 2, 5, 1), (0, 4, 3, 5)) * 0.24999999999999
    del x443
    x445 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x445 += einsum(x217, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x445 += einsum(x217, (0, 1, 2, 3), (0, 2, 1, 3))
    del x217
    x446 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x446 += einsum(x445, (0, 1, 2, 3), l3, (4, 5, 3, 1, 6, 2), (6, 0, 4, 5)) * 0.5
    del x445
    x447 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x447 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x447 += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3))
    x448 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x448 += einsum(x447, (0, 1, 2, 3), x112, (4, 0, 5, 1, 6, 2), (4, 5, 6, 3))
    del x447
    x449 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x449 += einsum(x117, (0, 1, 2, 3), l3, (4, 5, 2, 1, 6, 0), (6, 4, 5, 3)) * 0.5
    del x117
    x450 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x450 += einsum(x372, (0, 1, 2, 3), (0, 1, 2, 3))
    x450 += einsum(x449, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x449
    x451 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x451 += einsum(v.ovvv, (0, 1, 2, 3), x450, (4, 5, 3, 1), (0, 4, 2, 5))
    del x450
    x452 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x452 += einsum(t1, (0, 1), x8, (2, 0, 3, 4), (2, 3, 1, 4))
    x453 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x453 += einsum(x90, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x453 += einsum(x452, (0, 1, 2, 3), (0, 1, 2, 3))
    del x452
    x454 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x454 += einsum(x453, (0, 1, 2, 3), x112, (4, 0, 5, 1, 6, 2), (4, 5, 6, 3))
    del x453
    x455 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x455 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3))
    x455 += einsum(x199, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x199
    x456 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x456 += einsum(v.ovvv, (0, 1, 2, 3), x455, (4, 5, 0, 3), (4, 5, 1, 2))
    del x455
    x457 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x457 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x457 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * 2.0
    x458 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x458 += einsum(t1, (0, 1), x457, (2, 1, 3, 4), (0, 2, 3, 4))
    del x457
    x459 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x459 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x459 += einsum(x458, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x458
    x460 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x460 += einsum(l2, (0, 1, 2, 3), x459, (3, 4, 5, 1), (2, 4, 0, 5))
    del x459
    x461 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x461 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x461 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x462 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x462 += einsum(x212, (0, 1, 2, 3), x461, (0, 4, 2, 5), (1, 4, 3, 5)) * -0.5
    del x461
    x463 = np.zeros((nvir, nvir), dtype=types[float])
    x463 += einsum(v.ovov, (0, 1, 2, 3), x190, (0, 2, 4, 1), (3, 4))
    x464 = np.zeros((nvir, nvir), dtype=types[float])
    x464 += einsum(t1, (0, 1), x393, (0, 1, 2, 3), (2, 3)) * 0.5
    x465 = np.zeros((nvir, nvir), dtype=types[float])
    x465 += einsum(x463, (0, 1), (0, 1))
    del x463
    x465 += einsum(x464, (0, 1), (1, 0)) * -1.0
    del x464
    x466 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x466 += einsum(x465, (0, 1), l2, (2, 1, 3, 4), (4, 3, 2, 0)) * 2.0
    del x465
    x467 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x467 += einsum(x171, (0, 1, 2, 3), (0, 1, 2, 3))
    del x171
    x467 += einsum(x176, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0000000000000804
    del x176
    x468 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x468 += einsum(v.ovov, (0, 1, 2, 3), x467, (4, 5, 0, 2), (4, 5, 3, 1)) * 0.49999999999997996
    del x467
    x469 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x469 += einsum(x122, (0, 1, 2, 3), x263, (1, 4, 2, 5), (0, 4, 3, 5))
    del x263
    x470 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x470 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x470 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1))
    x470 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1)) * -3.0
    x471 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x471 += einsum(t2, (0, 1, 2, 3), x470, (1, 4, 5, 6, 2, 3), (0, 4, 5, 6))
    del x470
    x472 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x472 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * 0.3333333333333333
    x472 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1))
    x472 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1)) * -0.3333333333333333
    x473 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x473 += einsum(t2, (0, 1, 2, 3), x472, (1, 4, 5, 6, 3, 2), (0, 4, 5, 6)) * 3.0
    del x472
    x474 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x474 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x474 += einsum(x471, (0, 1, 2, 3), (2, 1, 0, 3))
    del x471
    x474 += einsum(x473, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x473
    x475 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x475 += einsum(v.ooov, (0, 1, 2, 3), x474, (4, 0, 1, 5), (2, 4, 3, 5)) * 0.5
    del x474
    x476 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x476 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x476 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x477 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x477 += einsum(x476, (0, 1, 2, 3), x8, (0, 4, 2, 5), (4, 1, 5, 3))
    del x476
    x478 = np.zeros((nocc, nocc), dtype=types[float])
    x478 += einsum(t1, (0, 1), x36, (2, 1), (0, 2)) * 2.0
    x479 = np.zeros((nocc, nocc), dtype=types[float])
    x479 += einsum(x271, (0, 1), (1, 0))
    del x271
    x479 += einsum(x273, (0, 1), (1, 0))
    del x273
    x479 += einsum(x478, (0, 1), (0, 1))
    del x478
    x480 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x480 += einsum(x479, (0, 1), l2, (2, 3, 0, 4), (4, 1, 2, 3))
    x481 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x481 += einsum(x67, (0, 1), x362, (2, 3, 0, 4), (2, 3, 4, 1))
    del x67, x362
    x482 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x482 += einsum(f.ov, (0, 1), x212, (2, 3, 0, 4), (2, 3, 4, 1)) * -0.5
    x483 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x483 += einsum(f.ov, (0, 1), l1, (2, 3), (0, 3, 1, 2))
    x483 += einsum(x365, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x365
    x483 += einsum(x366, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x366
    x483 += einsum(x367, (0, 1, 2, 3), (0, 1, 2, 3))
    del x367
    x483 += einsum(x368, (0, 1, 2, 3), (0, 1, 2, 3))
    del x368
    x483 += einsum(x369, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x369
    x483 += einsum(x370, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x370
    x483 += einsum(x371, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.49999999999997996
    del x371
    x483 += einsum(x373, (0, 1, 2, 3), (0, 1, 2, 3))
    del x373
    x483 += einsum(x381, (0, 1, 2, 3), (0, 1, 2, 3))
    del x381
    x483 += einsum(x391, (0, 1, 2, 3), (0, 1, 2, 3))
    del x391
    x483 += einsum(x396, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x396
    x483 += einsum(x403, (0, 1, 2, 3), (0, 1, 2, 3))
    del x403
    x483 += einsum(x408, (0, 1, 2, 3), (0, 1, 2, 3))
    del x408
    x483 += einsum(x414, (0, 1, 2, 3), (1, 0, 3, 2))
    del x414
    x483 += einsum(x433, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x433
    x483 += einsum(x444, (0, 1, 2, 3), (1, 0, 3, 2))
    del x444
    x483 += einsum(x446, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x446
    x483 += einsum(x448, (0, 1, 2, 3), (0, 1, 2, 3))
    del x448
    x483 += einsum(x451, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x451
    x483 += einsum(x454, (0, 1, 2, 3), (0, 1, 2, 3))
    del x454
    x483 += einsum(x456, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x456
    x483 += einsum(x460, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x460
    x483 += einsum(x462, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x462
    x483 += einsum(x466, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x466
    x483 += einsum(x468, (0, 1, 2, 3), (0, 1, 3, 2))
    del x468
    x483 += einsum(x469, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x469
    x483 += einsum(x475, (0, 1, 2, 3), (1, 0, 3, 2))
    del x475
    x483 += einsum(x477, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x477
    x483 += einsum(x480, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x480
    x483 += einsum(x481, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x481
    x483 += einsum(x482, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x482
    x483 += einsum(l1, (0, 1), x36, (2, 3), (1, 2, 0, 3)) * 2.0
    l2new += einsum(x483, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new += einsum(x483, (0, 1, 2, 3), (3, 2, 1, 0))
    del x483
    x484 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x484 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x484 += einsum(x65, (0, 1, 2, 3), (1, 3, 0, 2))
    x484 += einsum(x20, (0, 1, 2, 3), (0, 2, 1, 3))
    x485 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x485 += einsum(x484, (0, 1, 2, 3), l3, (4, 5, 6, 0, 7, 2), (7, 1, 3, 4, 6, 5)) * 0.5
    del x484
    l3new += einsum(x485, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2)) * -1.0
    l3new += einsum(x485, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 0, 2))
    del x485
    x486 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x486 += einsum(x20, (0, 1, 2, 3), l3, (4, 5, 6, 7, 1, 0), (7, 2, 3, 4, 6, 5))
    del x20
    x487 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x487 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x487 += einsum(x5, (0, 1, 2, 3), (2, 1, 0, 3))
    x487 += einsum(x65, (0, 1, 2, 3), (1, 3, 0, 2))
    del x65
    x488 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x488 += einsum(x487, (0, 1, 2, 3), l3, (4, 5, 6, 0, 2, 7), (7, 1, 3, 4, 6, 5))
    del x487
    x489 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x489 += einsum(x360, (0, 1), l3, (2, 3, 4, 0, 5, 6), (6, 5, 1, 2, 4, 3))
    del x360
    x490 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x490 += einsum(x486, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x486
    x490 += einsum(x488, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x488
    x490 += einsum(x489, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    del x489
    l3new += einsum(x490, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    l3new += einsum(x490, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 2, 0)) * -1.0
    del x490
    x491 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x491 += einsum(l2, (0, 1, 2, 3), v.ooov, (4, 3, 5, 6), (2, 4, 5, 0, 1, 6))
    l3new += einsum(x491, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2)) * -1.0
    l3new += einsum(x491, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 1, 2))
    l3new += einsum(x491, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    l3new += einsum(x491, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1))
    l3new += einsum(x491, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2)) * -1.0
    l3new += einsum(x491, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    l3new += einsum(x491, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 0, 1))
    l3new += einsum(x491, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 0, 1)) * -1.0
    l3new += einsum(x491, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 2, 0))
    l3new += einsum(x491, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * -1.0
    l3new += einsum(x491, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 1, 0))
    l3new += einsum(x491, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0)) * -1.0
    del x491
    x492 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x492 += einsum(f.vv, (0, 1), l3, (2, 3, 1, 4, 5, 6), (4, 6, 5, 0, 2, 3))
    x493 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x493 += einsum(f.ov, (0, 1), x112, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    x494 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x494 += einsum(v.vvvv, (0, 1, 2, 3), l3, (4, 3, 1, 5, 6, 7), (5, 7, 6, 4, 0, 2))
    x495 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x495 += einsum(v.ovvv, (0, 1, 2, 3), x112, (4, 5, 6, 0, 7, 3), (4, 5, 6, 7, 1, 2))
    x496 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x496 += einsum(t2, (0, 1, 2, 3), l3, (4, 3, 2, 5, 6, 7), (5, 7, 6, 0, 1, 4))
    x497 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x497 += einsum(x496, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x496
    x497 += einsum(x151, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x151
    x498 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x498 += einsum(v.ovov, (0, 1, 2, 3), x497, (4, 5, 6, 2, 0, 7), (4, 5, 6, 1, 3, 7))
    del x497
    x499 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x499 += einsum(x492, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x492
    x499 += einsum(x493, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x493
    x499 += einsum(x494, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x494
    x499 += einsum(x495, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x495
    x499 += einsum(x498, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    del x498
    l3new += einsum(x499, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    l3new += einsum(x499, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1)) * -1.0
    del x499
    x500 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x500 += einsum(l2, (0, 1, 2, 3), v.ovvv, (4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    x501 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x501 += einsum(l2, (0, 1, 2, 3), x8, (3, 4, 5, 6), (2, 4, 5, 0, 1, 6))
    x502 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x502 += einsum(v.ovov, (0, 1, 2, 3), x122, (4, 5, 2, 6), (4, 5, 0, 6, 1, 3))
    del x122
    x503 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x503 += einsum(x500, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x500
    x503 += einsum(x501, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x501
    x503 += einsum(x502, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x502
    l3new += einsum(x503, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 1, 2)) * -1.0
    l3new += einsum(x503, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2))
    l3new += einsum(x503, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1)) * -1.0
    l3new += einsum(x503, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1))
    l3new += einsum(x503, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2))
    l3new += einsum(x503, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2)) * -1.0
    l3new += einsum(x503, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 0, 1)) * -1.0
    l3new += einsum(x503, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 0, 1))
    l3new += einsum(x503, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0))
    l3new += einsum(x503, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * -1.0
    l3new += einsum(x503, (0, 1, 2, 3, 4, 5), (3, 5, 4, 2, 1, 0))
    l3new += einsum(x503, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0)) * -1.0
    del x503
    x504 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x504 += einsum(v.vvvv, (0, 1, 2, 3), l3, (1, 4, 3, 5, 6, 7), (5, 7, 6, 4, 0, 2))
    x505 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x505 += einsum(t2, (0, 1, 2, 3), l3, (2, 4, 3, 5, 6, 7), (5, 7, 6, 0, 1, 4))
    x506 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x506 += einsum(t1, (0, 1), x112, (2, 3, 4, 5, 1, 6), (2, 3, 4, 0, 5, 6))
    x507 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x507 += einsum(x505, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x505
    x507 += einsum(x506, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x506
    x508 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x508 += einsum(v.ovov, (0, 1, 2, 3), x507, (4, 5, 6, 2, 0, 7), (4, 5, 6, 1, 3, 7)) * 0.5
    del x507
    x509 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x509 += einsum(x504, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.5
    del x504
    x509 += einsum(x508, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    del x508
    l3new += einsum(x509, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0)) * -1.0
    l3new += einsum(x509, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0))
    del x509
    x510 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x510 += einsum(v.ovov, (0, 1, 2, 3), x212, (4, 5, 2, 6), (0, 4, 5, 3, 1, 6)) * -0.5
    del x212
    l3new += einsum(x510, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0))
    l3new += einsum(x510, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 2, 0)) * -1.0
    l3new += einsum(x510, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0)) * -1.0
    l3new += einsum(x510, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 1, 0))
    l3new += einsum(x510, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 1, 2))
    l3new += einsum(x510, (0, 1, 2, 3, 4, 5), (4, 3, 5, 0, 1, 2)) * -1.0
    l3new += einsum(x510, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1)) * -1.0
    l3new += einsum(x510, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    del x510
    x511 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x511 += einsum(v.ovov, (0, 1, 2, 3), x165, (4, 5, 2, 6), (4, 5, 0, 6, 1, 3))
    del x165
    l3new += einsum(x511, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 1, 2)) * -1.0
    l3new += einsum(x511, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2))
    l3new += einsum(x511, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1)) * -0.5
    l3new += einsum(x511, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1)) * 0.5
    l3new += einsum(x511, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2))
    l3new += einsum(x511, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2)) * -1.0
    l3new += einsum(x511, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 0, 1)) * -1.0
    l3new += einsum(x511, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 0, 1))
    l3new += einsum(x511, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0)) * 0.5
    l3new += einsum(x511, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * -0.5
    l3new += einsum(x511, (0, 1, 2, 3, 4, 5), (3, 5, 4, 2, 1, 0))
    l3new += einsum(x511, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0)) * -1.0
    del x511
    x512 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x512 += einsum(v.ooov, (0, 1, 2, 3), x112, (4, 5, 1, 2, 6, 7), (4, 5, 0, 6, 7, 3))
    x513 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x513 += einsum(v.ovvv, (0, 1, 2, 3), x118, (4, 5, 6, 0, 3, 7), (5, 4, 6, 7, 1, 2))
    x514 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x514 += einsum(x8, (0, 1, 2, 3), x112, (4, 5, 0, 1, 6, 7), (4, 5, 2, 6, 7, 3))
    x515 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x515 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x515 += einsum(x90, (0, 1, 2, 3), (0, 1, 3, 2))
    del x90
    x515 += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.00000000000001
    del x70
    l3new += einsum(x515, (0, 1, 2, 3), l3, (4, 2, 5, 6, 0, 7), (4, 3, 5, 6, 1, 7)) * -1.0
    x516 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x516 += einsum(x515, (0, 1, 2, 3), l3, (4, 5, 2, 6, 0, 7), (7, 6, 1, 4, 5, 3))
    x517 = np.zeros((nvir, nvir), dtype=types[float])
    x517 += einsum(v.ovov, (0, 1, 2, 3), x190, (0, 2, 4, 1), (3, 4)) * 2.0
    del x190
    x518 = np.zeros((nvir, nvir), dtype=types[float])
    x518 += einsum(t1, (0, 1), x393, (0, 1, 2, 3), (2, 3))
    del x393
    x519 = np.zeros((nvir, nvir), dtype=types[float])
    x519 += einsum(x517, (0, 1), (0, 1))
    del x517
    x519 += einsum(x518, (0, 1), (1, 0)) * -1.0
    del x518
    x520 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x520 += einsum(x519, (0, 1), l3, (2, 3, 1, 4, 5, 6), (6, 4, 5, 2, 3, 0))
    del x519
    x521 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x521 += einsum(x36, (0, 1), x112, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x36
    x522 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x522 += einsum(v.ovov, (0, 1, 2, 3), x428, (4, 5, 2, 6), (0, 5, 4, 3, 1, 6)) * 1.5
    del x428
    x523 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x523 += einsum(x512, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x512
    x523 += einsum(x513, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x513
    x523 += einsum(x514, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x514
    x523 += einsum(x516, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    del x516
    x523 += einsum(x520, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    del x520
    x523 += einsum(x521, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x521
    x523 += einsum(x522, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    del x522
    l3new += einsum(x523, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1)) * -1.0
    l3new += einsum(x523, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1))
    del x523
    x524 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x524 += einsum(v.ovov, (0, 1, 2, 3), x155, (4, 5, 6, 1), (0, 2, 4, 3, 5, 6)) * -0.5
    del x155
    l3new += einsum(x524, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 0, 1)) * -1.0
    l3new += einsum(x524, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 0, 1))
    l3new += einsum(x524, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0))
    l3new += einsum(x524, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 2, 0)) * -1.0
    l3new += einsum(x524, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1)) * -1.0
    l3new += einsum(x524, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    l3new += einsum(x524, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 0, 2))
    l3new += einsum(x524, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2)) * -1.0
    del x524
    x525 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x525 += einsum(v.ovov, (0, 1, 2, 3), x410, (4, 5, 6, 3), (4, 0, 2, 6, 5, 1))
    del x410
    l3new += einsum(x525, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 1, 2)) * 0.5
    l3new += einsum(x525, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2)) * -0.5
    l3new += einsum(x525, (0, 1, 2, 3, 4, 5), (3, 5, 4, 2, 1, 0)) * -0.5
    l3new += einsum(x525, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0)) * 0.5
    del x525
    x526 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x526 += einsum(v.ovov, (0, 1, 2, 3), x167, (4, 5, 2, 6), (4, 5, 0, 6, 1, 3))
    del x167
    l3new += einsum(x526, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1)) * 0.5
    l3new += einsum(x526, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1)) * -0.5
    l3new += einsum(x526, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0)) * -0.5
    l3new += einsum(x526, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * 0.5
    del x526
    x527 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x527 += einsum(x5, (0, 1, 2, 3), l3, (4, 5, 6, 7, 2, 0), (7, 1, 3, 4, 6, 5))
    x528 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x528 += einsum(v.ooov, (0, 1, 2, 3), x118, (4, 1, 5, 2, 6, 7), (4, 5, 0, 6, 7, 3))
    x529 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x529 += einsum(x8, (0, 1, 2, 3), x118, (0, 4, 5, 1, 6, 7), (4, 5, 2, 7, 6, 3))
    x530 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x530 += einsum(x515, (0, 1, 2, 3), l3, (4, 2, 5, 0, 6, 7), (7, 6, 1, 4, 5, 3))
    x531 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x531 += einsum(x479, (0, 1), l3, (2, 3, 4, 0, 5, 6), (6, 5, 1, 2, 4, 3))
    del x479
    x532 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x532 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -0.5
    x532 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1))
    x533 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x533 += einsum(t2, (0, 1, 2, 3), x532, (0, 4, 1, 5, 6, 2), (4, 3, 5, 6)) * 2.0
    del x532
    x534 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x534 += einsum(v.ovov, (0, 1, 2, 3), x533, (4, 1, 5, 6), (0, 2, 4, 3, 6, 5))
    del x533
    x535 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x535 += einsum(x527, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x527
    x535 += einsum(x528, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x528
    x535 += einsum(x529, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x529
    x535 += einsum(x530, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x530
    x535 += einsum(x531, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x531
    x535 += einsum(x534, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    del x534
    l3new += einsum(x535, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 1, 2))
    l3new += einsum(x535, (0, 1, 2, 3, 4, 5), (3, 5, 4, 2, 1, 0)) * -1.0
    del x535
    x536 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x536 += einsum(v.ovov, (0, 1, 2, 3), x372, (4, 5, 6, 3), (4, 0, 2, 5, 6, 1))
    del x372
    l3new += einsum(x536, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1)) * -1.0
    l3new += einsum(x536, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1))
    l3new += einsum(x536, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2))
    l3new += einsum(x536, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2)) * -1.0
    l3new += einsum(x536, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 0, 1)) * -1.0
    l3new += einsum(x536, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 0, 1))
    l3new += einsum(x536, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0))
    l3new += einsum(x536, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * -1.0
    del x536
    x537 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x537 += einsum(v.ooov, (0, 1, 2, 3), x112, (4, 0, 5, 1, 6, 7), (4, 5, 2, 6, 7, 3))
    x538 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x538 += einsum(x8, (0, 1, 2, 3), x112, (4, 0, 5, 2, 6, 7), (4, 5, 1, 6, 7, 3))
    x539 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x539 += einsum(v.ovov, (0, 1, 2, 3), x77, (2, 4, 3, 5), (0, 4, 1, 5)) * 1.00000000000001
    x540 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x540 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x540 += einsum(x99, (0, 1, 2, 3), (0, 1, 3, 2))
    x540 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.00000000000001
    x540 += einsum(x539, (0, 1, 2, 3), (1, 0, 3, 2))
    del x539
    x541 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x541 += einsum(x540, (0, 1, 2, 3), l3, (4, 5, 2, 6, 7, 0), (6, 7, 1, 4, 5, 3))
    del x540
    x542 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x542 += einsum(x537, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x537
    x542 += einsum(x538, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x538
    x542 += einsum(x541, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x541
    l3new += einsum(x542, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2)) * -1.0
    l3new += einsum(x542, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 1, 2))
    l3new += einsum(x542, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    l3new += einsum(x542, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2)) * -1.0
    l3new += einsum(x542, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    l3new += einsum(x542, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 0, 1))
    l3new += einsum(x542, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 0, 1)) * -1.0
    l3new += einsum(x542, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 2, 0))
    l3new += einsum(x542, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 1, 0))
    l3new += einsum(x542, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0)) * -1.0
    del x542
    x543 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x543 += einsum(v.ooov, (0, 1, 2, 3), x112, (4, 1, 5, 2, 6, 7), (4, 5, 0, 6, 7, 3))
    x544 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x544 += einsum(x8, (0, 1, 2, 3), x112, (4, 0, 5, 1, 6, 7), (4, 5, 2, 6, 7, 3))
    del x8
    x545 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x545 += einsum(x515, (0, 1, 2, 3), l3, (4, 5, 2, 6, 7, 0), (6, 7, 1, 4, 5, 3))
    del x515
    x546 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x546 += einsum(x543, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x543
    x546 += einsum(x544, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x544
    x546 += einsum(x545, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x545
    l3new += einsum(x546, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2))
    l3new += einsum(x546, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 1, 2)) * -1.0
    l3new += einsum(x546, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 1, 0)) * -1.0
    l3new += einsum(x546, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0))
    del x546
    x547 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x547 += einsum(v.ovvv, (0, 1, 2, 3), x112, (4, 5, 6, 0, 3, 7), (5, 4, 6, 7, 1, 2))
    del x112
    l3new += einsum(x547, (0, 1, 2, 3, 4, 5), (4, 3, 5, 0, 2, 1)) * -1.0
    l3new += einsum(x547, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 2, 1))
    del x547
    x548 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x548 += einsum(x5, (0, 1, 2, 3), l3, (4, 5, 6, 0, 7, 2), (7, 1, 3, 4, 6, 5))
    del x5
    l3new += einsum(x548, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 0, 2)) * -1.0
    l3new += einsum(x548, (0, 1, 2, 3, 4, 5), (3, 5, 4, 2, 0, 1))
    del x548
    x549 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x549 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.99999999999999
    x549 += einsum(x99, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.99999999999999
    del x99
    x549 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x76
    x549 += einsum(x96, (0, 1, 2, 3), (1, 0, 3, 2))
    del x96
    l3new += einsum(x549, (0, 1, 2, 3), x141, (4, 5, 0, 2, 6, 7), (6, 3, 7, 5, 1, 4)) * 1.00000000000001
    del x141, x549
    x550 = np.zeros((nvir, nvir), dtype=types[float])
    x550 += einsum(f.vv, (0, 1), (0, 1))
    x550 += einsum(v.ovov, (0, 1, 2, 3), x77, (0, 2, 1, 4), (4, 3)) * -1.0
    del x77
    x550 += einsum(t1, (0, 1), x54, (0, 1, 2, 3), (3, 2)) * 2.0
    del x54
    l3new += einsum(x550, (0, 1), l3, (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6))
    del x550
    x551 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x551 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x551 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l3new += einsum(x551, (0, 1, 2, 3), x118, (4, 5, 0, 1, 6, 7), (7, 3, 6, 5, 2, 4)) * -1.0
    del x118, x551
    x552 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x552 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x552 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    l3new += einsum(x68, (0, 1), x552, (2, 3, 4, 5), (5, 1, 4, 3, 0, 2)) * -1.0
    del x68, x552

    return {"l1new": l1new, "l2new": l2new, "l3new": l3new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, l1=None, l2=None, l3=None, **kwargs):
    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))

    # RDM1
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=types[float])
    rdm1_f_oo += einsum(delta.oo, (0, 1), (0, 1)) * 2.0
    rdm1_f_oo += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 5, 3, 1, 0, 2), (6, 4)) * 0.16666666666666
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=types[float])
    rdm1_f_ov += einsum(t1, (0, 1), (0, 1)) * 2.0
    rdm1_f_vo = np.zeros((nvir, nocc), dtype=types[float])
    rdm1_f_vo += einsum(l1, (0, 1), (0, 1)) * 2.0
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=types[float])
    rdm1_f_vv += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 4, 5, 0, 6, 2), (1, 6)) * 0.49999999999998
    rdm1_f_vv += einsum(l1, (0, 1), t1, (1, 2), (0, 2)) * 2.0
    x0 = np.zeros((nocc, nocc), dtype=types[float])
    x0 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    rdm1_f_oo += einsum(x0, (0, 1), (1, 0)) * -2.0
    x1 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x1 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -0.14285714285714288
    x1 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 0.14285714285714288
    x1 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    x1 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -0.14285714285714288
    rdm1_f_oo += einsum(l3, (0, 1, 2, 3, 4, 5), x1, (4, 6, 3, 0, 2, 1), (6, 5)) * -1.16666666666662
    del x1
    x2 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x2 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 3.0
    x2 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    x2 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    rdm1_f_oo += einsum(l3, (0, 1, 2, 3, 4, 5), x2, (3, 5, 6, 0, 2, 1), (6, 4)) * -0.16666666666666
    del x2
    x3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x3 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x3 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    rdm1_f_oo += einsum(l3, (0, 1, 2, 3, 4, 5), x3, (3, 6, 4, 0, 1, 2), (6, 5)) * 0.16666666666666
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x4 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm1_f_oo += einsum(l2, (0, 1, 2, 3), x4, (2, 4, 0, 1), (4, 3)) * -2.0
    del x4
    x5 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(t1, (0, 1), l3, (2, 1, 3, 4, 5, 6), (4, 6, 5, 0, 2, 3))
    rdm1_f_ov += einsum(t3, (0, 1, 2, 3, 4, 5), x5, (0, 2, 1, 6, 3, 5), (6, 4)) * -0.49999999999998
    del x5
    x6 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(t1, (0, 1), l3, (2, 3, 1, 4, 5, 6), (4, 6, 5, 0, 2, 3))
    x7 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x7 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x7 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.999999999999999
    x7 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    x7 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    x7 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    x7 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    rdm1_f_ov += einsum(x6, (0, 1, 2, 3, 4, 5), x7, (0, 1, 2, 5, 6, 4), (3, 6)) * -0.16666666666666
    del x7
    x8 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x8 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x8 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    x8 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    rdm1_f_ov += einsum(x6, (0, 1, 2, 3, 4, 5), x8, (0, 2, 1, 5, 4, 6), (3, 6)) * 0.16666666666666
    del x6, x8
    x9 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x9 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x9 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 6.0
    x9 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    x9 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    rdm1_f_ov += einsum(l2, (0, 1, 2, 3), x9, (2, 4, 3, 1, 0, 5), (4, 5)) * 0.5
    del x9
    x10 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x10 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x10 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    rdm1_f_ov += einsum(l2, (0, 1, 2, 3), x10, (3, 4, 2, 1, 0, 5), (4, 5)) * -0.5
    del x10
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.3333333333333333
    x12 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x13 += einsum(x12, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (1, 6, 5, 4)) * 0.75
    del x12
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x14 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.3333333333333333
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x15 += einsum(x11, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x11
    x15 += einsum(x13, (0, 1, 2, 3), (2, 1, 0, 3))
    x15 += einsum(x13, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x13
    x15 += einsum(x14, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 0), (5, 6, 1, 4)) * 1.5
    del x14
    rdm1_f_ov += einsum(t2, (0, 1, 2, 3), x15, (0, 1, 4, 2), (4, 3)) * -2.0
    del x15
    x16 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x16 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 2, 1)) * -1.0
    x16 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x16 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 0, 2))
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum(t2, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 1), (5, 6, 0, 4)) * -1.0
    x17 += einsum(t2, (0, 1, 2, 3), x16, (1, 4, 5, 3, 2, 6), (4, 5, 0, 6))
    del x16
    rdm1_f_ov += einsum(t2, (0, 1, 2, 3), x17, (0, 1, 4, 3), (4, 2)) * -1.0
    del x17
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x18 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm1_f_ov += einsum(l1, (0, 1), x18, (1, 2, 0, 3), (2, 3)) * 4.0
    del x18
    x19 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x19 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x19 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    x19 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 6.999999999999999
    x19 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    x20 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x20 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x20 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    x20 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 3.0
    x21 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x21 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x21 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1))
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x22 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x23 = np.zeros((nocc, nocc), dtype=types[float])
    x23 += einsum(x0, (0, 1), (0, 1))
    del x0
    x23 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 5, 4, 1, 0, 2), (3, 6)) * 0.08333333333333
    x23 += einsum(l3, (0, 1, 2, 3, 4, 5), x19, (4, 6, 5, 2, 0, 1), (3, 6)) * 0.08333333333333
    del x19
    x23 += einsum(l3, (0, 1, 2, 3, 4, 5), x20, (3, 6, 5, 1, 0, 2), (4, 6)) * 0.08333333333333
    del x20
    x23 += einsum(t3, (0, 1, 2, 3, 4, 5), x21, (2, 6, 1, 5, 4, 3), (6, 0)) * -0.08333333333333
    del x21
    x23 += einsum(l2, (0, 1, 2, 3), x22, (2, 4, 1, 0), (3, 4)) * 2.0
    del x22
    rdm1_f_ov += einsum(t1, (0, 1), x23, (0, 2), (2, 1)) * -2.0
    del x23
    x24 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x24 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.14285714285714288
    x24 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -0.14285714285714288
    x24 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    x24 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -0.14285714285714288
    x24 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 0.14285714285714288
    x24 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -0.14285714285714288
    rdm1_f_vv += einsum(l3, (0, 1, 2, 3, 4, 5), x24, (5, 3, 4, 1, 2, 6), (0, 6)) * 1.16666666666662
    del x24
    x25 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x25 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x25 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    x25 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    rdm1_f_vv += einsum(l3, (0, 1, 2, 3, 4, 5), x25, (5, 4, 3, 1, 6, 2), (0, 6)) * -0.16666666666666
    del x25
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.5
    x26 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm1_f_vv += einsum(t2, (0, 1, 2, 3), x26, (0, 1, 2, 4), (4, 3)) * 4.0
    del x26

    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])

    return rdm1_f

def make_rdm2_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, l1=None, l2=None, l3=None, **kwargs):
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
    x0 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 7, 4, 1, 2, 0), (3, 5, 6, 7))
    rdm2_f_oooo += einsum(x0, (0, 1, 2, 3), (3, 2, 1, 0)) * -0.16666666666667
    rdm2_f_oooo += einsum(x0, (0, 1, 2, 3), (3, 2, 1, 0)) * -0.16666666666667
    del x0
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 2, 1))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 2, 1))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 2, 1)) * -1.0
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 7, 5, 1, 0, 2), (3, 4, 6, 7))
    x3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x3 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x3 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    x4 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x4 += einsum(l3, (0, 1, 2, 3, 4, 5), x3, (5, 6, 7, 1, 0, 2), (3, 4, 6, 7)) * 0.16666666666667
    x5 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x5 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.16666666666667
    x5 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x4
    rdm2_f_oooo += einsum(x5, (0, 1, 2, 3), (3, 2, 0, 1))
    rdm2_f_oooo += einsum(x5, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    rdm2_f_oooo += einsum(x5, (0, 1, 2, 3), (3, 2, 0, 1))
    rdm2_f_oooo += einsum(x5, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x5
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x7 += einsum(x6, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 0), (5, 6, 1, 4))
    x8 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x8 += einsum(t1, (0, 1), x7, (2, 3, 4, 1), (0, 2, 3, 4)) * -0.5
    rdm2_f_oooo += einsum(x8, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(x8, (0, 1, 2, 3), (3, 0, 1, 2))
    rdm2_f_oooo += einsum(x8, (0, 1, 2, 3), (0, 3, 2, 1))
    rdm2_f_oooo += einsum(x8, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_oooo += einsum(x8, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(x8, (0, 1, 2, 3), (3, 0, 1, 2))
    rdm2_f_oooo += einsum(x8, (0, 1, 2, 3), (0, 3, 2, 1))
    rdm2_f_oooo += einsum(x8, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    del x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.3333333333333333
    x9 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum(x9, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (6, 5, 1, 4))
    x11 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x11 += einsum(t1, (0, 1), x10, (2, 3, 4, 1), (0, 3, 2, 4)) * 1.5
    rdm2_f_oooo += einsum(x11, (0, 1, 2, 3), (0, 3, 1, 2))
    rdm2_f_oooo += einsum(x11, (0, 1, 2, 3), (3, 0, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(x11, (0, 1, 2, 3), (0, 3, 1, 2))
    rdm2_f_oooo += einsum(x11, (0, 1, 2, 3), (3, 0, 1, 2)) * -1.0
    del x11
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 5, 4, 1, 0, 2), (3, 6))
    x13 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x13 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x13 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.999999999999999
    x13 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    x13 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    x14 = np.zeros((nocc, nocc), dtype=types[float])
    x14 += einsum(l3, (0, 1, 2, 3, 4, 5), x13, (3, 6, 4, 1, 2, 0), (5, 6)) * 0.041666666666665
    x15 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x15 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x15 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    x15 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 3.0
    x16 = np.zeros((nocc, nocc), dtype=types[float])
    x16 += einsum(l3, (0, 1, 2, 3, 4, 5), x15, (5, 6, 3, 0, 2, 1), (4, 6)) * 0.041666666666665
    x17 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x17 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x17 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 3, 5, 0, 2, 1))
    x18 = np.zeros((nocc, nocc), dtype=types[float])
    x18 += einsum(t3, (0, 1, 2, 3, 4, 5), x17, (1, 2, 6, 5, 4, 3), (0, 6)) * 0.041666666666665
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x19 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x20 = np.zeros((nocc, nocc), dtype=types[float])
    x20 += einsum(l2, (0, 1, 2, 3), x19, (3, 4, 0, 1), (2, 4))
    x21 = np.zeros((nocc, nocc), dtype=types[float])
    x21 += einsum(x12, (0, 1), (0, 1)) * 0.041666666666665
    x21 += einsum(x14, (0, 1), (0, 1))
    del x14
    x21 += einsum(x16, (0, 1), (0, 1))
    del x16
    x21 += einsum(x18, (0, 1), (1, 0)) * -1.0
    del x18
    x21 += einsum(x20, (0, 1), (0, 1))
    del x20
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x21, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x21, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x21, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x21, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x21, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x21, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x21, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x21, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x21, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x21, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x21, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x21, (2, 3), (3, 0, 2, 1)) * -2.0
    x22 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x22 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_oooo += einsum(x22, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_oooo += einsum(x22, (0, 1, 2, 3), (3, 2, 1, 0))
    x23 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x23 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm2_f_ovoo += einsum(x23, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_ovoo += einsum(x23, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_vooo += einsum(x23, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x23, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    x24 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x24 += einsum(t1, (0, 1), x23, (2, 3, 4, 1), (2, 3, 0, 4))
    rdm2_f_oooo += einsum(x24, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_oooo += einsum(x24, (0, 1, 2, 3), (3, 2, 1, 0))
    x25 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x25 += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2))
    x25 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oooo += einsum(x25, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_oooo += einsum(x25, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm2_f_oooo += einsum(x25, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_oooo += einsum(x25, (0, 1, 2, 3), (2, 3, 0, 1))
    x26 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x26 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -0.3333333333333333
    x26 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 0.3333333333333333
    x26 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    x27 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x27 += einsum(l3, (0, 1, 2, 3, 4, 5), x26, (4, 6, 7, 1, 0, 2), (5, 3, 6, 7)) * 0.50000000000001
    del x26
    rdm2_f_oooo += einsum(x27, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_oooo += einsum(x27, (0, 1, 2, 3), (3, 2, 1, 0))
    del x27
    x28 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x28 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 7, 5, 0, 1, 2), (3, 4, 6, 7))
    x29 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x29 += einsum(t2, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 1), (5, 6, 0, 4))
    rdm2_f_ovoo += einsum(x29, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_ovoo += einsum(x29, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_vooo += einsum(x29, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x29, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    x30 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x30 += einsum(t1, (0, 1), x29, (2, 3, 4, 1), (2, 3, 0, 4))
    x31 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x31 += einsum(x28, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.50000000000001
    x31 += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oooo += einsum(x31, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm2_f_oooo += einsum(x31, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_oooo += einsum(x31, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm2_f_oooo += einsum(x31, (0, 1, 2, 3), (3, 2, 1, 0))
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x32 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x33 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x33 += einsum(x32, (0, 1, 2, 3), l3, (2, 4, 3, 5, 6, 0), (5, 6, 1, 4))
    x34 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x34 += einsum(t1, (0, 1), x33, (2, 3, 4, 1), (0, 2, 3, 4)) * -0.5
    rdm2_f_oooo += einsum(x34, (0, 1, 2, 3), (3, 0, 1, 2))
    rdm2_f_oooo += einsum(x34, (0, 1, 2, 3), (0, 3, 2, 1))
    rdm2_f_oooo += einsum(x34, (0, 1, 2, 3), (3, 0, 1, 2))
    rdm2_f_oooo += einsum(x34, (0, 1, 2, 3), (0, 3, 2, 1))
    x35 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum(t1, (0, 1), l3, (2, 1, 3, 4, 5, 6), (4, 6, 5, 0, 2, 3))
    x36 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x36 += einsum(t1, (0, 1), x35, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x37 += einsum(t2, (0, 1, 2, 3), x36, (0, 4, 1, 5, 6, 2), (4, 5, 6, 3))
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov += einsum(x37, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_ooov += einsum(x37, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_ooov += einsum(x37, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_ooov += einsum(x37, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo += einsum(x37, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_oovo += einsum(x37, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_oovo += einsum(x37, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_oovo += einsum(x37, (0, 1, 2, 3), (2, 1, 3, 0))
    x38 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x38 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_ooov += einsum(x38, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_ooov += einsum(x38, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_oovo += einsum(x38, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_oovo += einsum(x38, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    x39 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x39 += einsum(l2, (0, 1, 2, 3), t3, (4, 3, 5, 0, 6, 1), (2, 4, 5, 6))
    x40 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x40 += einsum(t3, (0, 1, 2, 3, 4, 5), x35, (0, 2, 6, 7, 3, 5), (6, 7, 1, 4))
    x41 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x41 += einsum(t2, (0, 1, 2, 3), x29, (1, 4, 5, 2), (4, 0, 5, 3))
    x42 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum(t1, (0, 1), l3, (2, 3, 1, 4, 5, 6), (4, 6, 5, 0, 2, 3))
    x43 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x43 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    x43 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x43 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    x43 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 5.0
    x43 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    x43 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    x44 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x44 += einsum(x42, (0, 1, 2, 3, 4, 5), x43, (2, 6, 0, 4, 7, 5), (1, 3, 6, 7)) * 0.25
    del x43
    x45 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x45 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x45 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    x45 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    x46 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x46 += einsum(x42, (0, 1, 2, 3, 4, 5), x45, (0, 6, 1, 5, 7, 4), (2, 3, 6, 7)) * 0.25
    x47 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x47 += einsum(t2, (0, 1, 2, 3), l3, (4, 5, 3, 1, 0, 6), (6, 4, 5, 2))
    x48 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x48 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x48 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 0, 2))
    x48 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 0, 2)) * 0.5
    x48 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2)) * -0.5
    x49 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x49 += einsum(t2, (0, 1, 2, 3), x48, (1, 4, 0, 5, 6, 2), (4, 3, 5, 6))
    del x48
    x50 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x50 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x50 += einsum(x49, (0, 1, 2, 3), (0, 3, 2, 1))
    del x49
    x51 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x51 += einsum(t2, (0, 1, 2, 3), x50, (4, 3, 2, 5), (0, 1, 4, 5)) * 0.5
    del x50
    x52 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x52 += einsum(t2, (0, 1, 2, 3), l3, (4, 3, 2, 5, 1, 6), (5, 6, 0, 4))
    x53 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x53 += einsum(t2, (0, 1, 2, 3), l3, (4, 3, 2, 5, 6, 1), (5, 6, 0, 4))
    x54 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x54 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    x54 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1))
    x55 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x55 += einsum(t2, (0, 1, 2, 3), x54, (1, 4, 5, 3, 6, 2), (0, 4, 5, 6))
    del x54
    x56 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x56 += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    del x52
    x56 += einsum(x53, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x56 += einsum(x53, (0, 1, 2, 3), (1, 0, 2, 3))
    x56 += einsum(x55, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x55
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    x57 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x57 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x58 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x58 += einsum(x56, (0, 1, 2, 3), x57, (1, 4, 5, 3), (0, 2, 4, 5)) * 0.5
    del x57
    x59 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x59 += einsum(x6, (0, 1, 2, 3), l3, (2, 4, 3, 5, 6, 0), (5, 6, 1, 4))
    x60 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x60 += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x60 += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3))
    x61 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x61 += einsum(t2, (0, 1, 2, 3), x60, (1, 4, 5, 3), (0, 4, 5, 2)) * 0.5
    del x60
    x62 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x62 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x62 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3))
    x63 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x63 += einsum(t2, (0, 1, 2, 3), x62, (1, 4, 5, 2), (0, 4, 5, 3))
    del x62
    x64 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x64 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x64 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x65 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x65 += einsum(t2, (0, 1, 2, 3), x64, (1, 4, 5, 3), (0, 4, 5, 2)) * 2.0
    del x64
    x66 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x66 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x66 += einsum(x29, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x67 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x67 += einsum(t1, (0, 1), x66, (2, 3, 4, 1), (0, 2, 3, 4))
    del x66
    x68 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x68 += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2))
    x68 += einsum(x67, (0, 1, 2, 3), (2, 1, 3, 0))
    del x67
    x69 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x69 += einsum(t1, (0, 1), x68, (0, 2, 3, 4), (2, 3, 4, 1))
    del x68
    x70 = np.zeros((nocc, nocc), dtype=types[float])
    x70 += einsum(l3, (0, 1, 2, 3, 4, 5), x13, (3, 6, 4, 1, 2, 0), (5, 6)) * 0.08333333333333
    del x13
    x71 = np.zeros((nocc, nocc), dtype=types[float])
    x71 += einsum(l3, (0, 1, 2, 3, 4, 5), x15, (5, 6, 3, 0, 2, 1), (4, 6)) * 0.08333333333333
    del x15
    x72 = np.zeros((nocc, nocc), dtype=types[float])
    x72 += einsum(t3, (0, 1, 2, 3, 4, 5), x17, (1, 2, 6, 5, 4, 3), (0, 6)) * 0.08333333333333
    del x17
    x73 = np.zeros((nocc, nocc), dtype=types[float])
    x73 += einsum(l2, (0, 1, 2, 3), x19, (3, 4, 0, 1), (2, 4)) * 2.0
    x74 = np.zeros((nocc, nocc), dtype=types[float])
    x74 += einsum(x1, (0, 1), (0, 1))
    x74 += einsum(x12, (0, 1), (0, 1)) * 0.08333333333333
    x74 += einsum(x70, (0, 1), (0, 1))
    x74 += einsum(x71, (0, 1), (0, 1))
    x74 += einsum(x72, (0, 1), (1, 0)) * -1.0
    x74 += einsum(x73, (0, 1), (0, 1))
    rdm2_f_ooov += einsum(t1, (0, 1), x74, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_ooov += einsum(t1, (0, 1), x74, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_oovo += einsum(t1, (0, 1), x74, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oovo += einsum(t1, (0, 1), x74, (2, 3), (0, 3, 1, 2)) * -1.0
    x75 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x75 += einsum(delta.oo, (0, 1), t1, (2, 3), (0, 1, 2, 3))
    x75 += einsum(x38, (0, 1, 2, 3), (1, 0, 2, 3))
    x75 += einsum(x39, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    x75 += einsum(x40, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.25
    x75 += einsum(x41, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x75 += einsum(x44, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x75 += einsum(x46, (0, 1, 2, 3), (1, 0, 2, 3))
    x75 += einsum(x51, (0, 1, 2, 3), (1, 2, 0, 3))
    x75 += einsum(x58, (0, 1, 2, 3), (2, 0, 1, 3))
    del x58
    x75 += einsum(x61, (0, 1, 2, 3), (0, 1, 2, 3))
    x75 += einsum(x63, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x63
    x75 += einsum(x65, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x65
    x75 += einsum(x69, (0, 1, 2, 3), (2, 0, 1, 3))
    del x69
    x75 += einsum(t1, (0, 1), x74, (2, 3), (0, 2, 3, 1))
    rdm2_f_ooov += einsum(x75, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_ooov += einsum(x75, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(x75, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_ooov += einsum(x75, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_oovo += einsum(x75, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(x75, (0, 1, 2, 3), (2, 0, 3, 1))
    rdm2_f_oovo += einsum(x75, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(x75, (0, 1, 2, 3), (2, 0, 3, 1))
    del x75
    x76 = np.zeros((nocc, nvir), dtype=types[float])
    x76 += einsum(t3, (0, 1, 2, 3, 4, 5), x35, (2, 0, 1, 6, 5, 3), (6, 4))
    x77 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x77 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x77 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    x77 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 6.999999999999999
    x77 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    x77 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    x77 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    x78 = np.zeros((nocc, nvir), dtype=types[float])
    x78 += einsum(x42, (0, 1, 2, 3, 4, 5), x77, (0, 1, 2, 5, 4, 6), (3, 6)) * 0.08333333333333
    x79 = np.zeros((nocc, nvir), dtype=types[float])
    x79 += einsum(x42, (0, 1, 2, 3, 4, 5), x45, (2, 0, 1, 6, 5, 4), (3, 6)) * 0.08333333333333
    x80 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x80 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    x80 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x80 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    x80 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * 6.0
    x81 = np.zeros((nocc, nvir), dtype=types[float])
    x81 += einsum(l2, (0, 1, 2, 3), x80, (2, 3, 4, 1, 0, 5), (4, 5)) * 0.25
    x82 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x82 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 0.5
    x82 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x83 = np.zeros((nocc, nvir), dtype=types[float])
    x83 += einsum(l2, (0, 1, 2, 3), x82, (3, 4, 2, 0, 5, 1), (4, 5)) * 0.5
    x84 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x84 += einsum(x9, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (6, 5, 1, 4)) * 0.375
    x85 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x85 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x85 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 1, 2)) * -1.0
    x86 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x86 += einsum(x6, (0, 1, 2, 3), x85, (0, 4, 5, 3, 2, 6), (1, 4, 5, 6)) * 0.25
    del x85
    x87 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x87 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x87 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3))
    x87 += einsum(x84, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x87 += einsum(x84, (0, 1, 2, 3), (0, 1, 2, 3))
    del x84
    x87 += einsum(x86, (0, 1, 2, 3), (1, 2, 0, 3))
    x88 = np.zeros((nocc, nvir), dtype=types[float])
    x88 += einsum(t2, (0, 1, 2, 3), x87, (0, 1, 4, 3), (4, 2)) * 2.0
    x89 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x89 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x89 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.3333333333333333
    x90 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x90 += einsum(x89, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 0), (5, 6, 1, 4)) * 3.0
    x91 = np.zeros((nocc, nvir), dtype=types[float])
    x91 += einsum(t2, (0, 1, 2, 3), x90, (0, 1, 4, 2), (4, 3)) * 0.5
    x92 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x92 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x92 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x93 = np.zeros((nocc, nvir), dtype=types[float])
    x93 += einsum(l1, (0, 1), x92, (1, 2, 3, 0), (2, 3))
    x94 = np.zeros((nocc, nvir), dtype=types[float])
    x94 += einsum(t1, (0, 1), x74, (0, 2), (2, 1))
    x95 = np.zeros((nocc, nvir), dtype=types[float])
    x95 += einsum(x76, (0, 1), (0, 1)) * 0.24999999999999
    x95 += einsum(x78, (0, 1), (0, 1))
    x95 += einsum(x79, (0, 1), (0, 1)) * -1.0
    x95 += einsum(x81, (0, 1), (0, 1)) * -1.0
    x95 += einsum(x83, (0, 1), (0, 1))
    x95 += einsum(x88, (0, 1), (0, 1))
    x95 += einsum(x91, (0, 1), (0, 1))
    x95 += einsum(x93, (0, 1), (0, 1)) * -1.0
    x95 += einsum(x94, (0, 1), (0, 1))
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x95, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x95, (2, 3), (2, 0, 1, 3))
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x95, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x95, (2, 3), (2, 0, 1, 3))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x95, (2, 3), (0, 2, 3, 1))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x95, (2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x95, (2, 3), (0, 2, 3, 1))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x95, (2, 3), (2, 0, 3, 1)) * -1.0
    del x95
    x96 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x96 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.5
    x96 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x97 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x97 += einsum(x96, (0, 1, 2, 3), t3, (4, 0, 5, 3, 2, 6), (4, 5, 1, 6)) * 2.0
    rdm2_f_ooov += einsum(x97, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_ooov += einsum(x97, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_oovo += einsum(x97, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_oovo += einsum(x97, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x97
    x98 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x98 += einsum(t1, (0, 1), x42, (2, 3, 4, 5, 1, 6), (3, 2, 4, 5, 0, 6))
    x99 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x99 += einsum(x98, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    x99 += einsum(x36, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    x99 += einsum(x36, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    x99 += einsum(x36, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    x100 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x100 += einsum(t2, (0, 1, 2, 3), x99, (1, 4, 0, 5, 6, 2), (4, 5, 6, 3)) * 0.5
    del x99
    rdm2_f_ooov += einsum(x100, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_ooov += einsum(x100, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_oovo += einsum(x100, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_oovo += einsum(x100, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x100
    x101 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x101 += einsum(l3, (0, 1, 2, 3, 4, 5), x3, (5, 6, 7, 1, 0, 2), (3, 4, 6, 7))
    x102 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x102 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3))
    del x2
    x102 += einsum(x101, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x101
    x103 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x103 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -0.3333333333333333
    x103 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    x104 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x104 += einsum(l3, (0, 1, 2, 3, 4, 5), x103, (4, 6, 7, 0, 1, 2), (3, 5, 6, 7)) * 3.0
    del x103
    x105 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x105 += einsum(l3, (0, 1, 2, 3, 4, 5), x3, (4, 6, 7, 1, 0, 2), (5, 3, 6, 7))
    x106 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x106 += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3))
    x106 += einsum(x102, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x106 += einsum(x104, (0, 1, 2, 3), (1, 0, 2, 3))
    x106 += einsum(x105, (0, 1, 2, 3), (1, 0, 2, 3))
    del x105
    x107 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x107 += einsum(t1, (0, 1), x106, (2, 0, 3, 4), (2, 3, 4, 1)) * 0.16666666666667
    del x106
    rdm2_f_ooov += einsum(x107, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_ooov += einsum(x107, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_oovo += einsum(x107, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_oovo += einsum(x107, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x107
    x108 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x108 += einsum(l2, (0, 1, 2, 3), t3, (4, 5, 3, 6, 0, 1), (2, 4, 5, 6))
    rdm2_f_ooov += einsum(x108, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_ooov += einsum(x108, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_oovo += einsum(x108, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_oovo += einsum(x108, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    x109 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x109 += einsum(t3, (0, 1, 2, 3, 4, 5), x42, (0, 2, 6, 7, 3, 4), (6, 7, 1, 5))
    rdm2_f_ooov += einsum(x109, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.5
    rdm2_f_ooov += einsum(x109, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.5
    rdm2_f_oovo += einsum(x109, (0, 1, 2, 3), (1, 2, 3, 0)) * 0.5
    rdm2_f_oovo += einsum(x109, (0, 1, 2, 3), (1, 2, 3, 0)) * 0.5
    x110 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x110 += einsum(t3, (0, 1, 2, 3, 4, 5), x35, (2, 6, 1, 7, 5, 3), (6, 7, 0, 4))
    rdm2_f_ooov += einsum(x110, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.5
    rdm2_f_ooov += einsum(x110, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.5
    rdm2_f_oovo += einsum(x110, (0, 1, 2, 3), (1, 2, 3, 0)) * 0.5
    rdm2_f_oovo += einsum(x110, (0, 1, 2, 3), (1, 2, 3, 0)) * 0.5
    x111 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x111 += einsum(t3, (0, 1, 2, 3, 4, 5), x42, (0, 2, 6, 7, 5, 3), (6, 7, 1, 4))
    rdm2_f_ooov += einsum(x111, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.25
    rdm2_f_ooov += einsum(x111, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.25
    rdm2_f_oovo += einsum(x111, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.25
    rdm2_f_oovo += einsum(x111, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.25
    x112 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x112 += einsum(t3, (0, 1, 2, 3, 4, 5), x35, (2, 1, 6, 7, 5, 4), (6, 7, 0, 3))
    rdm2_f_ooov += einsum(x112, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.25
    rdm2_f_ooov += einsum(x112, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.25
    del x112
    x113 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x113 += einsum(t2, (0, 1, 2, 3), x47, (4, 2, 3, 5), (4, 0, 1, 5))
    rdm2_f_ooov += einsum(x113, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ooov += einsum(x113, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_oovo += einsum(x113, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_oovo += einsum(x113, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x114 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x114 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    x114 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x114 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 4.0
    x115 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x115 += einsum(x114, (0, 1, 2, 3, 4, 5), x42, (1, 6, 0, 7, 3, 4), (6, 7, 2, 5)) * 0.25
    del x114
    rdm2_f_ooov += einsum(x115, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ooov += einsum(x115, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x115
    x116 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x116 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    x116 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x117 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x117 += einsum(x116, (0, 1, 2, 3, 4, 5), x35, (2, 0, 6, 7, 3, 5), (6, 7, 1, 4)) * 0.25
    del x116
    rdm2_f_ooov += einsum(x117, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_ooov += einsum(x117, (0, 1, 2, 3), (1, 2, 0, 3))
    del x117
    x118 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x118 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    x118 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x119 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x119 += einsum(l2, (0, 1, 2, 3), x118, (3, 4, 5, 0, 1, 6), (2, 4, 5, 6)) * 0.5
    del x118
    rdm2_f_ooov += einsum(x119, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ooov += einsum(x119, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x119
    x120 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x120 += einsum(x36, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    x120 += einsum(x36, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    x121 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x121 += einsum(t2, (0, 1, 2, 3), x120, (1, 0, 4, 5, 6, 2), (4, 5, 6, 3)) * 0.5
    rdm2_f_ooov += einsum(x121, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_ooov += einsum(x121, (0, 1, 2, 3), (2, 1, 0, 3))
    del x121
    x122 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x122 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x122 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x123 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x123 += einsum(x122, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (6, 5, 1, 4)) * 0.5
    rdm2_f_vooo += einsum(x123, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    rdm2_f_vooo += einsum(x123, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x124 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x124 += einsum(x32, (0, 1, 2, 3), l3, (2, 4, 3, 5, 6, 0), (5, 6, 1, 4)) * 0.5
    x125 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x125 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x125 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x125 += einsum(x53, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x125 += einsum(x53, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x125 += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x125 += einsum(x29, (0, 1, 2, 3), (1, 0, 2, 3)) * 1.5
    x125 += einsum(x123, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x125 += einsum(x124, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x126 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x126 += einsum(t2, (0, 1, 2, 3), x125, (4, 1, 5, 3), (0, 4, 5, 2))
    del x125
    rdm2_f_ooov += einsum(x126, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(x126, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_oovo += einsum(x126, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(x126, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    del x126
    x127 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x127 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 2, 1)) * -1.0
    x127 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x128 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x128 += einsum(t2, (0, 1, 2, 3), x127, (0, 1, 4, 5, 2, 6), (4, 3, 5, 6))
    x129 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x129 += einsum(t2, (0, 1, 2, 3), x128, (4, 5, 2, 3), (0, 1, 4, 5)) * -0.5
    rdm2_f_ooov += einsum(x129, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_ooov += einsum(x129, (0, 1, 2, 3), (1, 0, 2, 3))
    del x129
    x130 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x130 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x130 += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x130 += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3))
    x131 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x131 += einsum(t2, (0, 1, 2, 3), x130, (1, 4, 5, 2), (0, 4, 5, 3)) * 0.5
    rdm2_f_ooov += einsum(x131, (0, 1, 2, 3), (2, 0, 1, 3))
    rdm2_f_ooov += einsum(x131, (0, 1, 2, 3), (2, 0, 1, 3))
    del x131
    x132 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x132 += einsum(t2, (0, 1, 2, 3), l3, (2, 4, 3, 1, 5, 6), (6, 5, 0, 4))
    x133 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x133 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x133 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2)) * -0.5
    x134 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x134 += einsum(t2, (0, 1, 2, 3), x133, (1, 4, 5, 6, 3, 2), (0, 4, 5, 6))
    x135 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x135 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x135 += einsum(x132, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    x135 += einsum(x134, (0, 1, 2, 3), (1, 2, 0, 3))
    del x134
    x136 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x136 += einsum(t2, (0, 1, 2, 3), x135, (4, 1, 5, 2), (0, 4, 5, 3))
    del x135
    rdm2_f_ooov += einsum(x136, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(x136, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_oovo += einsum(x136, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(x136, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    del x136
    x137 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x137 += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2))
    x137 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    x137 += einsum(x31, (0, 1, 2, 3), (0, 1, 2, 3))
    x137 += einsum(x31, (0, 1, 2, 3), (1, 0, 3, 2))
    del x31
    x137 += einsum(x34, (0, 1, 2, 3), (1, 2, 3, 0))
    x137 += einsum(x34, (0, 1, 2, 3), (2, 1, 0, 3))
    del x34
    x138 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x138 += einsum(t1, (0, 1), x137, (0, 2, 3, 4), (2, 3, 4, 1))
    del x137
    rdm2_f_ooov += einsum(x138, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_ooov += einsum(x138, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_oovo += einsum(x138, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_oovo += einsum(x138, (0, 1, 2, 3), (1, 2, 3, 0))
    del x138
    x139 = np.zeros((nocc, nvir), dtype=types[float])
    x139 += einsum(t1, (0, 1), (0, 1)) * -1.0
    x139 += einsum(x76, (0, 1), (0, 1)) * 0.24999999999999
    x139 += einsum(x78, (0, 1), (0, 1))
    del x78
    x139 += einsum(x79, (0, 1), (0, 1)) * -1.0
    del x79
    x139 += einsum(x81, (0, 1), (0, 1)) * -1.0
    del x81
    x139 += einsum(x83, (0, 1), (0, 1))
    del x83
    x139 += einsum(x88, (0, 1), (0, 1))
    del x88
    x139 += einsum(x91, (0, 1), (0, 1))
    del x91
    x139 += einsum(x93, (0, 1), (0, 1)) * -1.0
    del x93
    x139 += einsum(x94, (0, 1), (0, 1))
    del x94
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x139, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x139, (2, 3), (0, 2, 1, 3)) * -1.0
    del x139
    x140 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x140 += einsum(t3, (0, 1, 2, 3, 4, 5), x35, (2, 1, 6, 7, 5, 3), (6, 7, 0, 4))
    rdm2_f_oovo += einsum(x140, (0, 1, 2, 3), (2, 1, 3, 0)) * 0.25
    rdm2_f_oovo += einsum(x140, (0, 1, 2, 3), (2, 1, 3, 0)) * 0.25
    x141 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x141 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    x141 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x141 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 4.0
    x142 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x142 += einsum(x141, (0, 1, 2, 3, 4, 5), x42, (1, 6, 0, 7, 4, 3), (6, 7, 2, 5)) * 0.25
    del x141
    rdm2_f_oovo += einsum(x142, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_oovo += einsum(x142, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x142
    x143 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x143 += einsum(x35, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x143 += einsum(x35, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    x144 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x144 += einsum(t3, (0, 1, 2, 3, 4, 5), x143, (1, 2, 6, 7, 4, 5), (0, 6, 7, 3)) * 0.25
    rdm2_f_oovo += einsum(x144, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(x144, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    del x144
    x145 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x145 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    x145 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x146 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x146 += einsum(l2, (0, 1, 2, 3), x145, (2, 4, 5, 0, 1, 6), (3, 4, 5, 6)) * 0.5
    rdm2_f_oovo += einsum(x146, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_oovo += einsum(x146, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x146
    x147 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x147 += einsum(x36, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x147 += einsum(x36, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    x148 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x148 += einsum(t2, (0, 1, 2, 3), x147, (1, 0, 4, 5, 6, 2), (4, 5, 6, 3)) * 0.5
    del x147
    rdm2_f_oovo += einsum(x148, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_oovo += einsum(x148, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    del x148
    x149 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x149 += einsum(t2, (0, 1, 2, 3), x128, (4, 5, 3, 2), (0, 1, 4, 5)) * -0.5
    rdm2_f_oovo += einsum(x149, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_oovo += einsum(x149, (0, 1, 2, 3), (1, 0, 3, 2))
    x150 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x150 += einsum(x6, (0, 1, 2, 3), l3, (2, 4, 3, 5, 6, 0), (5, 6, 1, 4)) * 0.5
    rdm2_f_ovoo += einsum(x150, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    rdm2_f_ovoo += einsum(x150, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x150, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    rdm2_f_vooo += einsum(x150, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x151 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x151 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x151 += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3))
    x151 += einsum(x150, (0, 1, 2, 3), (1, 0, 2, 3))
    del x150
    x152 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x152 += einsum(t2, (0, 1, 2, 3), x151, (1, 4, 5, 2), (0, 4, 5, 3))
    rdm2_f_oovo += einsum(x152, (0, 1, 2, 3), (0, 2, 3, 1))
    rdm2_f_oovo += einsum(x152, (0, 1, 2, 3), (0, 2, 3, 1))
    del x152
    x153 = np.zeros((nocc, nvir), dtype=types[float])
    x153 += einsum(x42, (0, 1, 2, 3, 4, 5), x77, (0, 1, 2, 5, 4, 6), (3, 6))
    x154 = np.zeros((nocc, nvir), dtype=types[float])
    x154 += einsum(x42, (0, 1, 2, 3, 4, 5), x45, (2, 0, 1, 6, 5, 4), (3, 6))
    x155 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x155 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    x155 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x155 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    x155 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * 6.000000000000001
    x156 = np.zeros((nocc, nvir), dtype=types[float])
    x156 += einsum(l2, (0, 1, 2, 3), x155, (2, 3, 4, 1, 0, 5), (4, 5)) * 3.00000000000012
    del x155
    x157 = np.zeros((nocc, nvir), dtype=types[float])
    x157 += einsum(l2, (0, 1, 2, 3), x82, (3, 4, 2, 0, 5, 1), (4, 5)) * 6.00000000000024
    x158 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x158 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.33333333333333326
    x158 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x159 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x159 += einsum(x158, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (6, 5, 1, 4)) * 0.37500000000000006
    del x158
    x160 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x160 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x160 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3))
    x160 += einsum(x159, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x160 += einsum(x159, (0, 1, 2, 3), (0, 1, 2, 3))
    del x159
    x160 += einsum(x86, (0, 1, 2, 3), (1, 2, 0, 3))
    del x86
    x161 = np.zeros((nocc, nvir), dtype=types[float])
    x161 += einsum(t2, (0, 1, 2, 3), x160, (1, 0, 4, 2), (4, 3)) * 24.00000000000096
    del x160
    x162 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x162 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x162 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.33333333333333326
    x163 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x163 += einsum(x162, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 0), (5, 6, 1, 4)) * 3.0000000000000004
    del x162
    x164 = np.zeros((nocc, nvir), dtype=types[float])
    x164 += einsum(t2, (0, 1, 2, 3), x163, (0, 1, 4, 2), (4, 3)) * 6.00000000000024
    del x163
    x165 = np.zeros((nocc, nvir), dtype=types[float])
    x165 += einsum(l1, (0, 1), x92, (1, 2, 3, 0), (2, 3)) * 12.00000000000048
    x166 = np.zeros((nocc, nvir), dtype=types[float])
    x166 += einsum(t1, (0, 1), x74, (0, 2), (2, 1)) * 12.00000000000048
    x167 = np.zeros((nocc, nvir), dtype=types[float])
    x167 += einsum(t1, (0, 1), (0, 1)) * -12.00000000000048
    x167 += einsum(x76, (0, 1), (0, 1)) * 3.0
    x167 += einsum(x153, (0, 1), (0, 1))
    del x153
    x167 += einsum(x154, (0, 1), (0, 1)) * -1.0
    del x154
    x167 += einsum(x156, (0, 1), (0, 1)) * -1.0
    del x156
    x167 += einsum(x157, (0, 1), (0, 1))
    del x157
    x167 += einsum(x161, (0, 1), (0, 1))
    del x161
    x167 += einsum(x164, (0, 1), (0, 1))
    del x164
    x167 += einsum(x165, (0, 1), (0, 1)) * -1.0
    del x165
    x167 += einsum(x166, (0, 1), (0, 1))
    del x166
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x167, (2, 3), (2, 0, 3, 1)) * -0.08333333333333
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x167, (2, 3), (2, 0, 3, 1)) * -0.08333333333333
    del x167
    x168 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x168 += einsum(x32, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 0), (5, 6, 1, 4)) * 0.5
    x169 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x169 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x169 += einsum(x168, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_ovoo += einsum(x169, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm2_f_ovoo += einsum(x169, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_ovoo += einsum(x169, (0, 1, 2, 3), (2, 3, 0, 1))
    rdm2_f_ovoo += einsum(x169, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_vooo += einsum(x169, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x169, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_vooo += einsum(x169, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vooo += einsum(x169, (0, 1, 2, 3), (3, 2, 1, 0))
    del x169
    x170 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x170 += einsum(x9, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (6, 5, 1, 4)) * 1.5
    rdm2_f_ovoo += einsum(x170, (0, 1, 2, 3), (2, 3, 1, 0))
    rdm2_f_ovoo += einsum(x170, (0, 1, 2, 3), (2, 3, 1, 0))
    del x170
    x171 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x171 += einsum(t3, (0, 1, 2, 3, 4, 5), x36, (1, 0, 2, 6, 7, 4), (6, 7, 3, 5))
    rdm2_f_oovv += einsum(x171, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.16666666666667
    rdm2_f_oovv += einsum(x171, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.16666666666667
    del x171
    x172 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x172 += einsum(t3, (0, 1, 2, 3, 4, 5), x36, (0, 1, 2, 6, 7, 4), (6, 7, 3, 5))
    rdm2_f_oovv += einsum(x172, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.16666666666667
    rdm2_f_oovv += einsum(x172, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.16666666666667
    del x172
    x173 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x173 += einsum(x1, (0, 1), t2, (2, 0, 3, 4), (1, 2, 3, 4))
    x174 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x174 += einsum(t2, (0, 1, 2, 3), l3, (4, 3, 2, 5, 6, 7), (5, 7, 6, 0, 1, 4))
    x175 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x175 += einsum(t3, (0, 1, 2, 3, 4, 5), x174, (0, 1, 2, 6, 7, 3), (6, 7, 5, 4))
    x176 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x176 += einsum(t2, (0, 1, 2, 3), x35, (4, 1, 0, 5, 6, 3), (4, 5, 6, 2))
    rdm2_f_ovvo += einsum(x176, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x176, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x176, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x176, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x177 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x177 += einsum(t2, (0, 1, 2, 3), x176, (1, 4, 3, 5), (4, 0, 2, 5))
    x178 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x178 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x178 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    x179 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x179 += einsum(x174, (0, 1, 2, 3, 4, 5), x178, (1, 2, 0, 6, 5, 7), (3, 4, 6, 7)) * 0.08333333333333
    x180 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x180 += einsum(x122, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (5, 6, 1, 4)) * 0.5
    del x122
    x181 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x181 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x181 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3))
    x181 += einsum(x168, (0, 1, 2, 3), (0, 1, 2, 3))
    x181 += einsum(x180, (0, 1, 2, 3), (1, 0, 2, 3))
    del x180
    x182 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x182 += einsum(x181, (0, 1, 2, 3), t3, (4, 0, 1, 5, 6, 3), (4, 2, 5, 6)) * 0.5
    del x181
    x183 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x183 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (5, 6, 3, 1, 7, 2), (4, 6, 0, 7))
    x184 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x184 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 3, 5, 7, 2, 1), (4, 6, 0, 7))
    rdm2_f_vovo += einsum(x184, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.25
    rdm2_f_vovo += einsum(x184, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.25
    x185 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x185 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 0.2
    x185 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x185 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.2
    x185 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -0.2
    x185 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.2
    x185 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -0.2
    x185 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -0.2
    x185 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    x186 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x186 += einsum(l3, (0, 1, 2, 3, 4, 5), x185, (5, 6, 4, 2, 7, 1), (3, 6, 0, 7)) * 1.25
    del x185
    x187 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x187 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x187 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 0, 2))
    x188 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x188 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x188 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    x189 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x189 += einsum(x187, (0, 1, 2, 3, 4, 5), x188, (1, 6, 0, 5, 4, 7), (2, 6, 3, 7)) * 0.25
    del x187
    x190 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x190 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x190 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    x191 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x191 += einsum(l3, (0, 1, 2, 3, 4, 5), x190, (3, 5, 6, 0, 2, 7), (4, 6, 1, 7)) * 0.25
    del x190
    x192 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x192 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    x192 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * 2.0
    x193 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x193 += einsum(t2, (0, 1, 2, 3), x192, (1, 4, 2, 5), (0, 4, 3, 5))
    x194 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x194 += einsum(x183, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    x194 += einsum(x184, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x184
    x194 += einsum(x186, (0, 1, 2, 3), (0, 1, 2, 3))
    del x186
    x194 += einsum(x189, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x189
    x194 += einsum(x191, (0, 1, 2, 3), (0, 1, 2, 3))
    del x191
    x194 += einsum(x193, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x193
    x195 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x195 += einsum(t2, (0, 1, 2, 3), x194, (1, 4, 3, 5), (0, 4, 2, 5))
    del x194
    x196 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x196 += einsum(x7, (0, 1, 2, 3), t3, (4, 1, 0, 5, 6, 3), (4, 2, 5, 6)) * -0.25
    x197 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x197 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 6, 5, 0, 7, 2), (4, 6, 1, 7))
    rdm2_f_ovov += einsum(x197, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.25
    rdm2_f_ovov += einsum(x197, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.25
    rdm2_f_ovvo += einsum(x197, (0, 1, 2, 3), (1, 2, 3, 0)) * 0.25
    rdm2_f_ovvo += einsum(x197, (0, 1, 2, 3), (1, 2, 3, 0)) * 0.25
    rdm2_f_voov += einsum(x197, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.25
    rdm2_f_voov += einsum(x197, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.25
    rdm2_f_vovo += einsum(x197, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.25
    rdm2_f_vovo += einsum(x197, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.25
    x198 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x198 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    x198 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 5.0
    x198 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    x198 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    x198 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    x198 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    x199 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x199 += einsum(l3, (0, 1, 2, 3, 4, 5), x198, (5, 6, 4, 2, 7, 1), (3, 6, 0, 7))
    x200 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x200 += einsum(l3, (0, 1, 2, 3, 4, 5), x45, (5, 6, 3, 1, 7, 2), (4, 6, 0, 7))
    x201 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x201 += einsum(x197, (0, 1, 2, 3), (0, 1, 2, 3))
    x201 += einsum(x199, (0, 1, 2, 3), (0, 1, 2, 3))
    del x199
    x201 += einsum(x200, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x200
    x202 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x202 += einsum(t2, (0, 1, 2, 3), x201, (1, 4, 2, 5), (0, 4, 3, 5)) * 0.25
    del x201
    x203 = np.zeros((nvir, nvir), dtype=types[float])
    x203 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 4, 5, 0, 6, 2), (1, 6))
    x204 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x204 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x204 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    x204 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    x204 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    x204 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    x204 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 6.999999999999999
    x205 = np.zeros((nvir, nvir), dtype=types[float])
    x205 += einsum(l3, (0, 1, 2, 3, 4, 5), x204, (4, 3, 5, 1, 6, 2), (0, 6)) * 0.08333333333333
    x206 = np.zeros((nvir, nvir), dtype=types[float])
    x206 += einsum(l3, (0, 1, 2, 3, 4, 5), x45, (4, 5, 3, 6, 1, 2), (0, 6)) * 0.08333333333333
    x207 = np.zeros((nvir, nvir), dtype=types[float])
    x207 += einsum(t2, (0, 1, 2, 3), x96, (0, 1, 2, 4), (3, 4)) * 2.0
    x208 = np.zeros((nvir, nvir), dtype=types[float])
    x208 += einsum(x203, (0, 1), (0, 1)) * 0.24999999999999
    x208 += einsum(x205, (0, 1), (0, 1))
    x208 += einsum(x206, (0, 1), (0, 1)) * -1.0
    x208 += einsum(x207, (0, 1), (1, 0))
    x209 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x209 += einsum(x208, (0, 1), t2, (2, 3, 0, 4), (2, 3, 4, 1))
    del x208
    x210 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x210 += einsum(t1, (0, 1), x56, (0, 2, 3, 4), (2, 3, 1, 4))
    del x56
    x211 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x211 += einsum(x210, (0, 1, 2, 3), x32, (0, 4, 5, 3), (4, 1, 5, 2)) * 0.5
    del x210
    x212 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x212 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x212 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x213 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x213 += einsum(t2, (0, 1, 2, 3), x212, (4, 1, 5, 2), (0, 4, 5, 3))
    del x212
    x214 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x214 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x214 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3))
    x215 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x215 += einsum(t2, (0, 1, 2, 3), x214, (4, 1, 5, 3), (0, 4, 5, 2)) * 2.0
    del x214
    x216 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x216 += einsum(t1, (0, 1), x7, (2, 3, 4, 1), (0, 2, 3, 4)) * -1.0
    x217 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x217 += einsum(t1, (0, 1), x216, (2, 3, 0, 4), (3, 2, 4, 1)) * 0.5
    del x216
    x218 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x218 += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3))
    x218 += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x39
    x218 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.25
    del x40
    x218 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x41
    x218 += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x44
    x218 += einsum(x46, (0, 1, 2, 3), (0, 1, 2, 3))
    del x46
    x218 += einsum(x51, (0, 1, 2, 3), (2, 1, 0, 3))
    del x51
    x218 += einsum(x61, (0, 1, 2, 3), (1, 0, 2, 3))
    del x61
    x218 += einsum(x213, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x213
    x218 += einsum(x215, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x215
    x218 += einsum(x217, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x217
    x219 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x219 += einsum(t1, (0, 1), x218, (0, 2, 3, 4), (2, 3, 1, 4))
    del x218
    x220 = np.zeros((nocc, nocc), dtype=types[float])
    x220 += einsum(x12, (0, 1), (0, 1)) * 0.08333333333333
    del x12
    x220 += einsum(x70, (0, 1), (0, 1))
    del x70
    x220 += einsum(x71, (0, 1), (0, 1))
    del x71
    x220 += einsum(x72, (0, 1), (1, 0)) * -1.0
    del x72
    x220 += einsum(x73, (0, 1), (0, 1))
    del x73
    x221 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x221 += einsum(x220, (0, 1), t2, (2, 0, 3, 4), (2, 1, 4, 3))
    del x220
    x222 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x222 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    x222 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 1, 2))
    x223 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x223 += einsum(t2, (0, 1, 2, 3), x222, (1, 4, 5, 6, 2, 3), (4, 5, 0, 6))
    x224 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x224 += einsum(t1, (0, 1), x223, (2, 3, 4, 1), (2, 3, 4, 0)) * -1.0
    del x223
    x225 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x225 += einsum(t2, (0, 1, 2, 3), x224, (0, 1, 4, 5), (5, 4, 3, 2)) * 0.5
    del x224
    x226 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x226 += einsum(t1, (0, 1), x10, (2, 3, 4, 1), (0, 3, 2, 4)) * 3.0
    x227 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x227 += einsum(t2, (0, 1, 2, 3), x226, (4, 0, 1, 5), (4, 5, 2, 3)) * 0.25
    del x226
    x228 = np.zeros((nocc, nvir), dtype=types[float])
    x228 += einsum(x42, (0, 1, 2, 3, 4, 5), x77, (0, 1, 2, 5, 4, 6), (3, 6)) * 0.041666666666665
    del x77
    x229 = np.zeros((nocc, nvir), dtype=types[float])
    x229 += einsum(x42, (0, 1, 2, 3, 4, 5), x45, (2, 0, 1, 6, 5, 4), (3, 6)) * 0.041666666666665
    x230 = np.zeros((nocc, nvir), dtype=types[float])
    x230 += einsum(l2, (0, 1, 2, 3), x80, (2, 3, 4, 1, 0, 5), (4, 5)) * 0.125
    del x80
    x231 = np.zeros((nocc, nvir), dtype=types[float])
    x231 += einsum(l2, (0, 1, 2, 3), x82, (3, 4, 2, 0, 5, 1), (4, 5)) * 0.25
    del x82
    x232 = np.zeros((nocc, nvir), dtype=types[float])
    x232 += einsum(t2, (0, 1, 2, 3), x87, (0, 1, 4, 3), (4, 2))
    del x87
    x233 = np.zeros((nocc, nvir), dtype=types[float])
    x233 += einsum(t2, (0, 1, 2, 3), x90, (0, 1, 4, 2), (4, 3)) * 0.25
    del x90
    x234 = np.zeros((nocc, nvir), dtype=types[float])
    x234 += einsum(l1, (0, 1), x92, (1, 2, 3, 0), (2, 3)) * 0.5
    x235 = np.zeros((nocc, nvir), dtype=types[float])
    x235 += einsum(t1, (0, 1), x74, (0, 2), (2, 1)) * 0.5
    del x74
    x236 = np.zeros((nocc, nvir), dtype=types[float])
    x236 += einsum(x76, (0, 1), (0, 1)) * 0.124999999999995
    x236 += einsum(x228, (0, 1), (0, 1))
    x236 += einsum(x229, (0, 1), (0, 1)) * -1.0
    x236 += einsum(x230, (0, 1), (0, 1)) * -1.0
    x236 += einsum(x231, (0, 1), (0, 1))
    x236 += einsum(x232, (0, 1), (0, 1))
    x236 += einsum(x233, (0, 1), (0, 1))
    x236 += einsum(x234, (0, 1), (0, 1)) * -1.0
    x236 += einsum(x235, (0, 1), (0, 1))
    del x235
    x237 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x237 += einsum(x173, (0, 1, 2, 3), (0, 1, 2, 3))
    x237 += einsum(x175, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.08333333333333
    del x175
    x237 += einsum(x177, (0, 1, 2, 3), (0, 1, 2, 3))
    x237 += einsum(x179, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x179
    x237 += einsum(x182, (0, 1, 2, 3), (1, 0, 2, 3))
    del x182
    x237 += einsum(x195, (0, 1, 2, 3), (0, 1, 2, 3))
    x237 += einsum(x196, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x196
    x237 += einsum(x202, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x202
    x237 += einsum(x209, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x237 += einsum(x211, (0, 1, 2, 3), (0, 1, 3, 2))
    del x211
    x237 += einsum(x219, (0, 1, 2, 3), (0, 1, 2, 3))
    del x219
    x237 += einsum(x221, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x237 += einsum(x225, (0, 1, 2, 3), (0, 1, 3, 2))
    del x225
    x237 += einsum(x227, (0, 1, 2, 3), (0, 1, 2, 3))
    del x227
    x237 += einsum(t1, (0, 1), x236, (2, 3), (0, 2, 1, 3)) * -2.0
    del x236
    rdm2_f_oovv += einsum(x237, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x237, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x237, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x237, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_oovv += einsum(x237, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x237, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x237, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x237, (0, 1, 2, 3), (1, 0, 3, 2))
    del x237
    x238 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x238 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x238 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    x239 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x239 += einsum(x174, (0, 1, 2, 3, 4, 5), x238, (1, 2, 0, 6, 7, 5), (3, 4, 6, 7)) * 0.08333333333333
    x240 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x240 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 5.000000000000001
    x240 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x241 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x241 += einsum(x240, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 0), (1, 5, 6, 4)) * 0.16666666666666666
    del x240
    x242 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x242 += einsum(x9, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (5, 6, 1, 4)) * 0.5
    x243 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x243 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x243 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.3333333333333333
    x243 += einsum(x241, (0, 1, 2, 3), (1, 2, 0, 3))
    del x241
    x243 += einsum(x242, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x242
    x244 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x244 += einsum(x243, (0, 1, 2, 3), t3, (4, 0, 1, 5, 3, 6), (2, 4, 6, 5)) * 1.5
    del x243
    x245 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x245 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * 2.0
    x245 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 1, 2)) * -1.0
    x246 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x246 += einsum(x6, (0, 1, 2, 3), x245, (0, 4, 5, 3, 2, 6), (4, 5, 1, 6))
    del x245
    x247 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x247 += einsum(x246, (0, 1, 2, 3), t3, (4, 1, 0, 5, 3, 6), (2, 4, 6, 5)) * 0.25
    del x246
    x248 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x248 += einsum(x6, (0, 1, 2, 3), x42, (1, 4, 0, 5, 2, 6), (4, 5, 6, 3))
    x249 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x249 += einsum(x248, (0, 1, 2, 3), x32, (0, 4, 5, 2), (1, 4, 3, 5)) * -1.0
    x250 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x250 += einsum(x42, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x250 += einsum(x42, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    x251 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x251 += einsum(t2, (0, 1, 2, 3), x250, (0, 1, 4, 5, 3, 6), (4, 5, 6, 2))
    x252 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x252 += einsum(t2, (0, 1, 2, 3), x251, (1, 4, 2, 5), (4, 0, 5, 3)) * -0.5
    x253 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x253 += einsum(x239, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x239
    x253 += einsum(x244, (0, 1, 2, 3), (0, 1, 3, 2))
    del x244
    x253 += einsum(x247, (0, 1, 2, 3), (0, 1, 3, 2))
    del x247
    x253 += einsum(x249, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x249
    x253 += einsum(x252, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x252
    rdm2_f_oovv += einsum(x253, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x253, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x253, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x253, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x253
    x254 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x254 += einsum(l1, (0, 1), t3, (2, 3, 1, 4, 5, 0), (2, 3, 4, 5))
    x255 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x255 += einsum(t2, (0, 1, 2, 3), x24, (1, 0, 4, 5), (5, 4, 2, 3))
    rdm2_f_oovv += einsum(x255, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x255, (0, 1, 2, 3), (0, 1, 2, 3))
    x256 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x256 += einsum(t2, (0, 1, 2, 3), l3, (2, 4, 3, 5, 6, 7), (5, 7, 6, 0, 1, 4))
    x257 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x257 += einsum(t3, (0, 1, 2, 3, 4, 5), x256, (2, 0, 1, 6, 7, 4), (6, 7, 5, 3))
    del x256
    x258 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x258 += einsum(t2, (0, 1, 2, 3), l3, (4, 2, 5, 6, 0, 1), (6, 4, 5, 3))
    x259 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x259 += einsum(x258, (0, 1, 2, 3), t3, (4, 0, 5, 2, 6, 1), (4, 5, 3, 6))
    x260 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x260 += einsum(x89, (0, 1, 2, 3), l3, (4, 5, 2, 6, 0, 1), (6, 4, 5, 3)) * 6.0
    del x89
    x261 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x261 += einsum(x258, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x261 += einsum(x258, (0, 1, 2, 3), (0, 2, 1, 3)) * -2.0
    x261 += einsum(x260, (0, 1, 2, 3), (0, 1, 2, 3))
    del x260
    x261 += einsum(x128, (0, 1, 2, 3), (0, 2, 3, 1))
    x262 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x262 += einsum(x261, (0, 1, 2, 3), t3, (4, 0, 5, 2, 1, 6), (4, 5, 3, 6)) * 0.25
    del x261
    x263 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x263 += einsum(t2, (0, 1, 2, 3), l3, (4, 5, 3, 6, 0, 1), (6, 4, 5, 2))
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov += einsum(x263, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvov += einsum(x263, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo += einsum(x263, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vvvo += einsum(x263, (0, 1, 2, 3), (2, 1, 3, 0))
    x264 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x264 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 2, 1)) * 1.5
    x264 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.5
    x264 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1))
    x265 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x265 += einsum(t2, (0, 1, 2, 3), x264, (0, 1, 4, 5, 2, 6), (4, 5, 6, 3)) * 0.6666666666666666
    del x264
    x266 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x266 += einsum(x263, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.6666666666666666
    x266 += einsum(x265, (0, 1, 2, 3), (0, 1, 2, 3))
    del x265
    x267 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x267 += einsum(x266, (0, 1, 2, 3), t3, (4, 0, 5, 1, 2, 6), (4, 5, 3, 6)) * 0.75
    del x266
    x268 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x268 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 2, 1)) * 0.5
    x268 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -0.5
    x268 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1))
    x269 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x269 += einsum(t2, (0, 1, 2, 3), x268, (0, 1, 4, 5, 2, 6), (4, 5, 6, 3)) * 2.0
    del x268
    x270 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x270 += einsum(x263, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x270 += einsum(x269, (0, 1, 2, 3), (0, 1, 2, 3))
    del x269
    x271 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x271 += einsum(x270, (0, 1, 2, 3), t3, (4, 0, 5, 2, 6, 1), (4, 5, 3, 6)) * 0.25
    x272 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x272 += einsum(t2, (0, 1, 2, 3), x96, (1, 4, 3, 5), (0, 4, 2, 5))
    x273 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x273 += einsum(t2, (0, 1, 2, 3), x272, (4, 1, 5, 3), (4, 0, 5, 2)) * 4.0
    del x272
    x274 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x274 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x274 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x275 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x275 += einsum(t2, (0, 1, 2, 3), x274, (1, 4, 2, 5), (4, 0, 5, 3))
    x276 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x276 += einsum(t2, (0, 1, 2, 3), x275, (1, 4, 2, 5), (4, 0, 5, 3)) * -1.0
    del x275
    x277 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x277 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 4, 7, 2, 1, 0), (5, 3, 6, 7))
    x278 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x278 += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2))
    x278 += einsum(x277, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.24999999999999
    del x277
    x279 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x279 += einsum(t2, (0, 1, 2, 3), x278, (0, 1, 4, 5), (4, 5, 2, 3))
    del x278
    x280 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x280 += einsum(t2, (0, 1, 2, 3), x98, (1, 4, 0, 5, 6, 2), (4, 6, 5, 3))
    del x98
    x281 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x281 += einsum(x96, (0, 1, 2, 3), t3, (4, 0, 5, 3, 2, 6), (4, 5, 1, 6))
    x282 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x282 += einsum(t1, (0, 1), x25, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x25
    x283 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x283 += einsum(x280, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x280
    x283 += einsum(x281, (0, 1, 2, 3), (2, 1, 0, 3))
    del x281
    x283 += einsum(x282, (0, 1, 2, 3), (0, 2, 1, 3))
    del x282
    x284 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x284 += einsum(t1, (0, 1), x283, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    del x283
    x285 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x285 += einsum(x254, (0, 1, 2, 3), (0, 1, 2, 3))
    x285 += einsum(x255, (0, 1, 2, 3), (0, 1, 2, 3))
    del x255
    x285 += einsum(x257, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.24999999999999
    del x257
    x285 += einsum(x259, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x259
    x285 += einsum(x262, (0, 1, 2, 3), (1, 0, 2, 3))
    del x262
    x285 += einsum(x267, (0, 1, 2, 3), (1, 0, 2, 3))
    del x267
    x285 += einsum(x271, (0, 1, 2, 3), (1, 0, 2, 3))
    del x271
    x285 += einsum(x273, (0, 1, 2, 3), (0, 1, 2, 3))
    del x273
    x285 += einsum(x276, (0, 1, 2, 3), (1, 0, 3, 2))
    del x276
    x285 += einsum(x279, (0, 1, 2, 3), (1, 0, 3, 2))
    del x279
    x285 += einsum(x284, (0, 1, 2, 3), (0, 1, 3, 2))
    del x284
    rdm2_f_oovv += einsum(x285, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x285, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x285, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x285, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x285
    x286 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x286 += einsum(t3, (0, 1, 2, 3, 4, 5), x174, (0, 1, 2, 6, 7, 4), (6, 7, 3, 5))
    x287 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x287 += einsum(x42, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x287 += einsum(x42, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    x287 += einsum(x42, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * 2.0
    x288 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x288 += einsum(t2, (0, 1, 2, 3), x287, (0, 1, 4, 5, 6, 3), (4, 5, 6, 2)) * 0.5
    x289 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x289 += einsum(x288, (0, 1, 2, 3), x6, (0, 4, 5, 2), (1, 4, 3, 5))
    del x288
    x290 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x290 += einsum(t1, (0, 1), x10, (2, 3, 4, 1), (0, 3, 2, 4))
    x291 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x291 += einsum(t1, (0, 1), x290, (2, 3, 0, 4), (3, 2, 4, 1)) * 3.0
    del x290
    x292 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x292 += einsum(t1, (0, 1), x291, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x291
    x293 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x293 += einsum(x286, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.08333333333333
    del x286
    x293 += einsum(x289, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x289
    x293 += einsum(x292, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x292
    rdm2_f_oovv += einsum(x293, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x293, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x293, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x293, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x293
    x294 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x294 += einsum(t3, (0, 1, 2, 3, 4, 5), x36, (0, 1, 2, 6, 7, 3), (6, 7, 5, 4))
    x295 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x295 += einsum(x178, (0, 1, 2, 3, 4, 5), x36, (1, 0, 2, 6, 7, 4), (6, 7, 3, 5)) * 0.16666666666667
    x296 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x296 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x296 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1)) * -0.5
    x297 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x297 += einsum(x296, (0, 1, 2, 3, 4, 5), x3, (0, 6, 7, 5, 4, 3), (1, 2, 6, 7))
    del x3, x296
    x298 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x298 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * 0.5
    x298 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 3, 5, 0, 2, 1))
    x299 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x299 += einsum(t3, (0, 1, 2, 3, 4, 5), x298, (6, 7, 2, 4, 5, 3), (6, 7, 0, 1))
    del x298
    x300 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x300 += einsum(x297, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x297
    x300 += einsum(x299, (0, 1, 2, 3), (1, 0, 2, 3))
    del x299
    x301 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x301 += einsum(t2, (0, 1, 2, 3), x300, (0, 1, 4, 5), (4, 5, 2, 3)) * 0.16666666666666
    del x300
    x302 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x302 += einsum(x36, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x302 += einsum(x36, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -0.5
    x302 += einsum(x36, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * 0.5
    x303 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x303 += einsum(t2, (0, 1, 2, 3), x302, (0, 4, 1, 5, 6, 3), (4, 5, 6, 2)) * 5.99999999999988
    del x302
    x304 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x304 += einsum(t1, (0, 1), x102, (2, 0, 3, 4), (2, 3, 4, 1))
    del x102
    x305 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x305 += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3)) * -5.99999999999988
    x305 += einsum(x303, (0, 1, 2, 3), (0, 1, 2, 3))
    del x303
    x305 += einsum(x304, (0, 1, 2, 3), (0, 1, 2, 3))
    del x304
    x306 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x306 += einsum(t1, (0, 1), x305, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.16666666666667
    del x305
    x307 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x307 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x307 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * -1.0
    x307 += einsum(x294, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.16666666666667
    del x294
    x307 += einsum(x295, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x295
    x307 += einsum(x301, (0, 1, 2, 3), (1, 0, 3, 2))
    del x301
    x307 += einsum(x306, (0, 1, 2, 3), (1, 0, 2, 3))
    del x306
    rdm2_f_oovv += einsum(x307, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x307, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x307, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x307, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x307
    x308 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x308 += einsum(t2, (0, 1, 2, 3), x251, (1, 4, 3, 5), (4, 0, 5, 2)) * -1.0
    rdm2_f_oovv += einsum(x308, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x308, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    rdm2_f_oovv += einsum(x308, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x308, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    rdm2_f_oovv += einsum(x308, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x308, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    rdm2_f_oovv += einsum(x308, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x308, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    del x308
    x309 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x309 += einsum(t2, (0, 1, 2, 3), x42, (1, 4, 0, 5, 6, 2), (4, 5, 6, 3))
    x310 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x310 += einsum(x309, (0, 1, 2, 3), x32, (0, 4, 5, 2), (4, 1, 5, 3))
    rdm2_f_oovv += einsum(x310, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_oovv += einsum(x310, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    rdm2_f_oovv += einsum(x310, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x310, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    rdm2_f_oovv += einsum(x310, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_oovv += einsum(x310, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    rdm2_f_oovv += einsum(x310, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x310, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x310
    x311 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x311 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x311 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 0, 2)) * 0.3333333333333333
    x312 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x312 += einsum(t1, (0, 1), x311, (2, 3, 4, 1, 5, 6), (0, 2, 3, 4, 5, 6)) * 3.0
    del x311
    x313 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x313 += einsum(t1, (0, 1), x312, (2, 3, 4, 5, 1, 6), (0, 4, 3, 5, 2, 6)) * 0.3333333333333333
    del x312
    x314 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x314 += einsum(t3, (0, 1, 2, 3, 4, 5), x313, (6, 0, 2, 1, 7, 4), (6, 7, 5, 3)) * 0.50000000000001
    del x313
    rdm2_f_oovv += einsum(x314, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_oovv += einsum(x314, (0, 1, 2, 3), (1, 0, 2, 3))
    del x314
    x315 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x315 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x315 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    x316 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x316 += einsum(l1, (0, 1), x315, (2, 3, 1, 4, 5, 0), (2, 3, 4, 5))
    rdm2_f_oovv += einsum(x316, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x316, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x316
    x317 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x317 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x317 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    x318 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x318 += einsum(l3, (0, 1, 2, 3, 4, 5), x317, (4, 6, 7, 1, 2, 0), (3, 5, 6, 7))
    x319 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x319 += einsum(x318, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x318
    x319 += einsum(x104, (0, 1, 2, 3), (1, 0, 2, 3))
    del x104
    x320 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x320 += einsum(t1, (0, 1), x319, (0, 2, 3, 4), (2, 3, 4, 1))
    del x319
    x321 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x321 += einsum(t1, (0, 1), x320, (0, 2, 3, 4), (2, 3, 1, 4)) * 0.16666666666667
    del x320
    rdm2_f_oovv += einsum(x321, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x321, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x321
    x322 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x322 += einsum(t2, (0, 1, 2, 3), x42, (4, 0, 1, 5, 2, 6), (4, 5, 6, 3))
    x323 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x323 += einsum(t2, (0, 1, 2, 3), x322, (1, 4, 2, 5), (4, 0, 3, 5))
    del x322
    rdm2_f_oovv += einsum(x323, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x323, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x323
    x324 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x324 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 1, 5), (2, 4, 0, 5))
    x325 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x325 += einsum(t2, (0, 1, 2, 3), x324, (1, 4, 2, 5), (4, 0, 5, 3))
    del x324
    rdm2_f_oovv += einsum(x325, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x325, (0, 1, 2, 3), (0, 1, 2, 3))
    del x325
    x326 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x326 += einsum(t1, (0, 1), x24, (0, 2, 3, 4), (2, 4, 3, 1))
    del x24
    x327 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x327 += einsum(t1, (0, 1), x326, (0, 2, 3, 4), (2, 3, 1, 4))
    del x326
    rdm2_f_oovv += einsum(x327, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x327, (0, 1, 2, 3), (0, 1, 2, 3))
    del x327
    x328 = np.zeros((nocc, nvir), dtype=types[float])
    x328 += einsum(t1, (0, 1), x1, (0, 2), (2, 1))
    del x1
    rdm2_f_oovv += einsum(t1, (0, 1), x328, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_oovv += einsum(t1, (0, 1), x328, (2, 3), (0, 2, 1, 3)) * -1.0
    x329 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x329 += einsum(t2, (0, 1, 2, 3), x28, (0, 1, 4, 5), (4, 5, 2, 3))
    x330 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x330 += einsum(x258, (0, 1, 2, 3), t3, (4, 5, 0, 2, 6, 1), (4, 5, 3, 6))
    x331 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x331 += einsum(t2, (0, 1, 2, 3), l3, (4, 2, 3, 1, 5, 6), (6, 5, 0, 4))
    x332 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x332 += einsum(t1, (0, 1), x331, (2, 3, 4, 1), (2, 3, 0, 4))
    x333 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x333 += einsum(t2, (0, 1, 2, 3), x332, (1, 0, 4, 5), (4, 5, 3, 2))
    del x332
    x334 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x334 += einsum(x270, (0, 1, 2, 3), t3, (4, 5, 0, 1, 6, 2), (4, 5, 3, 6)) * 0.25
    x335 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x335 += einsum(x174, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x174
    x335 += einsum(x36, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 1.0000000000000602
    del x36
    x336 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x336 += einsum(t3, (0, 1, 2, 3, 4, 5), x335, (0, 2, 1, 6, 7, 3), (6, 7, 5, 4)) * 0.49999999999997996
    del x335
    x337 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x337 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x337 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x337 += einsum(x168, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x168
    x337 += einsum(x123, (0, 1, 2, 3), (1, 0, 2, 3))
    del x123
    x338 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x338 += einsum(x337, (0, 1, 2, 3), t3, (0, 4, 1, 3, 5, 6), (2, 4, 6, 5)) * 0.5
    del x337
    x339 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x339 += einsum(x7, (0, 1, 2, 3), t3, (0, 4, 1, 5, 6, 3), (4, 2, 5, 6)) * -0.25
    del x7
    x340 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x340 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 3, 5, 7, 0, 2), (4, 6, 1, 7))
    x341 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x341 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x341 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -0.25
    x341 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.25
    x342 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x342 += einsum(l3, (0, 1, 2, 3, 4, 5), x341, (4, 6, 5, 1, 7, 2), (6, 3, 7, 0))
    del x341
    rdm2_f_ovvo += einsum(x342, (0, 1, 2, 3), (0, 3, 2, 1))
    rdm2_f_ovvo += einsum(x342, (0, 1, 2, 3), (0, 3, 2, 1))
    x343 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x343 += einsum(l3, (0, 1, 2, 3, 4, 5), x188, (3, 6, 5, 0, 2, 7), (4, 6, 1, 7)) * 0.25
    del x188
    x344 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x344 += einsum(x183, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    x344 += einsum(x340, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x340
    x344 += einsum(x342, (0, 1, 2, 3), (1, 0, 3, 2))
    del x342
    x344 += einsum(x343, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x343
    x345 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x345 += einsum(t2, (0, 1, 2, 3), x344, (1, 4, 2, 5), (4, 0, 5, 3))
    del x344
    x346 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x346 += einsum(t2, (0, 1, 2, 3), x143, (0, 1, 4, 5, 6, 2), (4, 5, 3, 6))
    x347 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x347 += einsum(t2, (0, 1, 2, 3), x346, (1, 4, 5, 2), (4, 0, 5, 3)) * -0.5
    del x346
    x348 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x348 += einsum(t2, (0, 1, 2, 3), x23, (4, 1, 5, 3), (4, 5, 0, 2))
    x349 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x349 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    x349 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    x349 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x350 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x350 += einsum(x349, (0, 1, 2, 3, 4, 5), x42, (2, 6, 0, 7, 5, 3), (1, 6, 7, 4)) * 0.5
    x351 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x351 += einsum(x35, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    x351 += einsum(x35, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    x352 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x352 += einsum(t3, (0, 1, 2, 3, 4, 5), x351, (1, 2, 6, 7, 4, 5), (6, 7, 0, 3)) * 0.5
    del x351
    x353 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x353 += einsum(l2, (0, 1, 2, 3), x145, (2, 4, 5, 0, 1, 6), (3, 4, 5, 6))
    del x145
    x354 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x354 += einsum(t2, (0, 1, 2, 3), x133, (1, 4, 5, 6, 3, 2), (0, 4, 5, 6)) * 2.0
    del x133
    x355 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x355 += einsum(x132, (0, 1, 2, 3), (1, 0, 2, 3))
    del x132
    x355 += einsum(x354, (0, 1, 2, 3), (1, 2, 0, 3))
    del x354
    x356 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x356 += einsum(t2, (0, 1, 2, 3), x355, (4, 1, 5, 2), (4, 5, 0, 3))
    del x355
    x357 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x357 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x357 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x358 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x358 += einsum(x23, (0, 1, 2, 3), x357, (0, 4, 3, 5), (4, 1, 2, 5)) * 2.0
    x359 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x359 += einsum(x28, (0, 1, 2, 3), (0, 1, 2, 3))
    del x28
    x359 += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.99999999999996
    del x30
    x360 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x360 += einsum(t1, (0, 1), x359, (2, 0, 3, 4), (2, 3, 4, 1)) * 1.00000000000002
    del x359
    x361 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x361 += einsum(x348, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x348
    x361 += einsum(x140, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x140
    x361 += einsum(x113, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x113
    x361 += einsum(x111, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x111
    x361 += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x37
    x361 += einsum(x350, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x350
    x361 += einsum(x352, (0, 1, 2, 3), (0, 1, 2, 3))
    del x352
    x361 += einsum(x353, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x353
    x361 += einsum(x356, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x356
    x361 += einsum(x358, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x358
    x361 += einsum(x360, (0, 1, 2, 3), (0, 1, 2, 3))
    del x360
    x362 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x362 += einsum(t1, (0, 1), x361, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x361
    x363 = np.zeros((nocc, nvir), dtype=types[float])
    x363 += einsum(t1, (0, 1), x21, (0, 2), (2, 1))
    del x21
    x364 = np.zeros((nocc, nvir), dtype=types[float])
    x364 += einsum(x76, (0, 1), (0, 1)) * 0.124999999999995
    del x76
    x364 += einsum(x228, (0, 1), (0, 1))
    del x228
    x364 += einsum(x229, (0, 1), (0, 1)) * -1.0
    del x229
    x364 += einsum(x230, (0, 1), (0, 1)) * -1.0
    del x230
    x364 += einsum(x231, (0, 1), (0, 1))
    del x231
    x364 += einsum(x232, (0, 1), (0, 1))
    del x232
    x364 += einsum(x233, (0, 1), (0, 1))
    del x233
    x364 += einsum(x234, (0, 1), (0, 1)) * -1.0
    del x234
    x364 += einsum(x363, (0, 1), (0, 1))
    del x363
    x365 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x365 += einsum(x254, (0, 1, 2, 3), (0, 1, 2, 3))
    del x254
    x365 += einsum(x329, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.49999999999997996
    del x329
    x365 += einsum(x330, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x330
    x365 += einsum(x333, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x333
    x365 += einsum(x334, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x334
    x365 += einsum(x336, (0, 1, 2, 3), (0, 1, 2, 3))
    del x336
    x365 += einsum(x338, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x338
    x365 += einsum(x195, (0, 1, 2, 3), (0, 1, 2, 3))
    del x195
    x365 += einsum(x339, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x339
    x365 += einsum(x345, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x345
    x365 += einsum(x209, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x209
    x365 += einsum(x347, (0, 1, 2, 3), (0, 1, 3, 2))
    del x347
    x365 += einsum(x362, (0, 1, 2, 3), (0, 1, 3, 2))
    del x362
    x365 += einsum(x221, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x221
    x365 += einsum(t1, (0, 1), x364, (2, 3), (0, 2, 1, 3)) * -2.0
    del x364
    rdm2_f_oovv += einsum(x365, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x365, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_oovv += einsum(x365, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x365, (0, 1, 2, 3), (1, 0, 3, 2))
    del x365
    x366 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x366 += einsum(x29, (0, 1, 2, 3), t3, (4, 1, 0, 5, 6, 3), (2, 4, 5, 6))
    x367 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x367 += einsum(x263, (0, 1, 2, 3), t3, (4, 5, 0, 6, 2, 1), (4, 5, 3, 6))
    x368 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x368 += einsum(x128, (0, 1, 2, 3), t3, (4, 5, 0, 6, 2, 3), (4, 5, 6, 1)) * -0.5
    x369 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x369 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3))
    x369 += einsum(x124, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x124
    x370 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x370 += einsum(x369, (0, 1, 2, 3), t3, (4, 0, 1, 5, 6, 3), (2, 4, 5, 6))
    del x369
    x371 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x371 += einsum(t2, (0, 1, 2, 3), x250, (0, 1, 4, 5, 3, 6), (4, 5, 6, 2)) * 0.5
    x372 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x372 += einsum(x176, (0, 1, 2, 3), (0, 1, 2, 3))
    x372 += einsum(x371, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x371
    x373 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x373 += einsum(t2, (0, 1, 2, 3), x372, (1, 4, 2, 5), (4, 0, 5, 3))
    del x372
    x374 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x374 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (5, 6, 3, 2, 1, 7), (4, 6, 0, 7))
    rdm2_f_ovov += einsum(x374, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.5
    rdm2_f_ovov += einsum(x374, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.5
    rdm2_f_vovo += einsum(x374, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.5
    rdm2_f_vovo += einsum(x374, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.5
    x375 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x375 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 4, 5, 0, 7, 2), (3, 6, 1, 7))
    rdm2_f_ovov += einsum(x375, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.5
    rdm2_f_ovov += einsum(x375, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.5
    rdm2_f_vovo += einsum(x375, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.5
    rdm2_f_vovo += einsum(x375, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.5
    x376 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x376 += einsum(x374, (0, 1, 2, 3), (0, 1, 2, 3))
    x376 += einsum(x375, (0, 1, 2, 3), (0, 1, 2, 3))
    x377 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x377 += einsum(t2, (0, 1, 2, 3), x376, (1, 4, 2, 5), (4, 0, 5, 3)) * 0.5
    del x376
    x378 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x378 += einsum(x9, (0, 1, 2, 3), l3, (4, 2, 3, 5, 0, 6), (6, 5, 1, 4)) * 3.0
    del x9
    x379 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x379 += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3))
    x379 += einsum(x29, (0, 1, 2, 3), (1, 0, 2, 3)) * -3.0
    x379 += einsum(x53, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x379 += einsum(x53, (0, 1, 2, 3), (1, 0, 2, 3))
    del x53
    x379 += einsum(x378, (0, 1, 2, 3), (1, 0, 2, 3))
    x379 += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x380 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x380 += einsum(t1, (0, 1), x379, (0, 2, 3, 4), (2, 3, 4, 1))
    del x379
    x381 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x381 += einsum(t2, (0, 1, 2, 3), x380, (1, 4, 3, 5), (4, 0, 5, 2)) * 0.5
    del x380
    x382 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x382 += einsum(t2, (0, 1, 2, 3), x23, (4, 1, 5, 2), (4, 5, 0, 3))
    x383 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x383 += einsum(t2, (0, 1, 2, 3), x120, (0, 1, 4, 5, 6, 2), (4, 5, 6, 3)) * 0.5
    del x120
    x384 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x384 += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x29
    x384 += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3))
    del x59
    x385 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x385 += einsum(t2, (0, 1, 2, 3), x384, (1, 4, 5, 2), (4, 5, 0, 3)) * 0.5
    del x384
    x386 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x386 += einsum(t1, (0, 1), x33, (2, 3, 4, 1), (0, 2, 3, 4)) * -1.0
    x387 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x387 += einsum(t1, (0, 1), x386, (2, 3, 0, 4), (3, 2, 4, 1)) * 0.5
    x388 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x388 += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x38
    x388 += einsum(x108, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x108
    x388 += einsum(x382, (0, 1, 2, 3), (0, 1, 2, 3))
    del x382
    x388 += einsum(x110, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x110
    x388 += einsum(x109, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x109
    x388 += einsum(x383, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x383
    x388 += einsum(x149, (0, 1, 2, 3), (2, 1, 0, 3))
    del x149
    x388 += einsum(x385, (0, 1, 2, 3), (0, 2, 1, 3))
    del x385
    x388 += einsum(x387, (0, 1, 2, 3), (0, 1, 2, 3))
    del x387
    x389 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x389 += einsum(t1, (0, 1), x388, (0, 2, 3, 4), (2, 3, 4, 1))
    del x388
    x390 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x390 += einsum(t2, (0, 1, 2, 3), x386, (4, 0, 1, 5), (4, 5, 3, 2)) * 0.5
    del x386
    x391 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x391 += einsum(x173, (0, 1, 2, 3), (0, 1, 2, 3))
    del x173
    x391 += einsum(x366, (0, 1, 2, 3), (0, 1, 2, 3))
    del x366
    x391 += einsum(x367, (0, 1, 2, 3), (0, 1, 2, 3))
    del x367
    x391 += einsum(x177, (0, 1, 2, 3), (0, 1, 2, 3))
    del x177
    x391 += einsum(x368, (0, 1, 2, 3), (0, 1, 3, 2))
    del x368
    x391 += einsum(x370, (0, 1, 2, 3), (0, 1, 2, 3))
    del x370
    x391 += einsum(x373, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x373
    x391 += einsum(x377, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x377
    x391 += einsum(x381, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x381
    x391 += einsum(x389, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x389
    x391 += einsum(x390, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x390
    rdm2_f_oovv += einsum(x391, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x391, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x391, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x391, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x391
    x392 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x392 += einsum(t2, (0, 1, 2, 3), x251, (1, 4, 3, 5), (4, 0, 5, 2)) * -0.5
    del x251
    rdm2_f_oovv += einsum(x392, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x392, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    rdm2_f_oovv += einsum(x392, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x392, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x392
    x393 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x393 += einsum(t2, (0, 1, 2, 3), x309, (1, 4, 3, 5), (4, 0, 2, 5))
    del x309
    rdm2_f_oovv += einsum(x393, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    rdm2_f_oovv += einsum(x393, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_oovv += einsum(x393, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    rdm2_f_oovv += einsum(x393, (0, 1, 2, 3), (1, 0, 2, 3))
    del x393
    x394 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x394 += einsum(t2, (0, 1, 2, 3), x287, (0, 1, 4, 5, 6, 3), (4, 5, 6, 2)) * 0.25
    del x287
    x395 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x395 += einsum(t2, (0, 1, 2, 3), x192, (1, 4, 3, 5), (0, 4, 2, 5))
    del x192
    x396 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x396 += einsum(x394, (0, 1, 2, 3), (0, 1, 2, 3))
    del x394
    x396 += einsum(x395, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x395
    x397 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x397 += einsum(t2, (0, 1, 2, 3), x396, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    del x396
    rdm2_f_oovv += einsum(x397, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x397, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x397
    x398 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x398 += einsum(t2, (0, 1, 2, 3), x248, (1, 4, 3, 5), (4, 0, 5, 2)) * -1.0
    del x248
    rdm2_f_oovv += einsum(x398, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x398, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x398
    x399 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x399 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), (3, 4, 0, 5))
    rdm2_f_ovov += einsum(x399, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x399, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x399, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x399, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x400 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x400 += einsum(t2, (0, 1, 2, 3), x42, (0, 4, 1, 5, 2, 6), (4, 5, 6, 3))
    rdm2_f_ovov += einsum(x400, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x400, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x400, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x400, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x401 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x401 += einsum(x399, (0, 1, 2, 3), (0, 1, 2, 3))
    x401 += einsum(x400, (0, 1, 2, 3), (0, 1, 2, 3))
    x402 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x402 += einsum(t2, (0, 1, 2, 3), x401, (1, 4, 2, 5), (4, 0, 5, 3))
    del x401
    rdm2_f_oovv += einsum(x402, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_oovv += einsum(x402, (0, 1, 2, 3), (1, 0, 2, 3))
    del x402
    x403 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x403 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x403 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    x404 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x404 += einsum(x22, (0, 1, 2, 3), x403, (0, 1, 4, 5), (2, 3, 4, 5))
    del x22, x403
    rdm2_f_oovv += einsum(x404, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_oovv += einsum(x404, (0, 1, 2, 3), (1, 0, 3, 2))
    del x404
    x405 = np.zeros((nocc, nvir), dtype=types[float])
    x405 += einsum(t1, (0, 1), (0, 1))
    x405 += einsum(x328, (0, 1), (0, 1)) * -1.0
    del x328
    rdm2_f_oovv += einsum(t1, (0, 1), x405, (2, 3), (2, 0, 3, 1))
    rdm2_f_oovv += einsum(t1, (0, 1), x405, (2, 3), (2, 0, 3, 1))
    del x405
    x406 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x406 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 5, 4, 1, 7, 2), (3, 6, 0, 7))
    rdm2_f_ovov += einsum(x406, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.25
    rdm2_f_ovov += einsum(x406, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.25
    rdm2_f_voov += einsum(x406, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.25
    rdm2_f_voov += einsum(x406, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.25
    del x406
    x407 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x407 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.2
    x407 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.2
    x407 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -0.2
    x407 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    x408 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x408 += einsum(l3, (0, 1, 2, 3, 4, 5), x407, (4, 6, 5, 1, 7, 2), (6, 3, 7, 0)) * 1.25
    del x407
    rdm2_f_ovov += einsum(x408, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_ovov += einsum(x408, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    del x408
    x409 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x409 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    x409 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1))
    x410 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x410 += einsum(t3, (0, 1, 2, 3, 4, 5), x409, (2, 1, 6, 5, 7, 3), (6, 0, 7, 4)) * 0.25
    del x409
    rdm2_f_ovov += einsum(x410, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_ovov += einsum(x410, (0, 1, 2, 3), (1, 2, 0, 3))
    del x410
    x411 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x411 += einsum(l3, (0, 1, 2, 3, 4, 5), x317, (5, 6, 3, 1, 7, 2), (4, 6, 0, 7)) * 0.25
    rdm2_f_ovov += einsum(x411, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x411, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovvo += einsum(x411, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_ovvo += einsum(x411, (0, 1, 2, 3), (1, 2, 3, 0))
    del x411
    x412 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x412 += einsum(t2, (0, 1, 2, 3), x250, (0, 1, 4, 5, 6, 2), (4, 5, 6, 3)) * 0.5
    rdm2_f_ovov += einsum(x412, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x412, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x412, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x412, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x412
    x413 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x413 += einsum(x42, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    x413 += einsum(x42, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    x414 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x414 += einsum(t2, (0, 1, 2, 3), x413, (4, 0, 1, 5, 6, 3), (4, 5, 6, 2))
    del x413
    rdm2_f_ovov += einsum(x414, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_ovov += einsum(x414, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_ovvo += einsum(x414, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x414, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    del x414
    x415 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x415 += einsum(l2, (0, 1, 2, 3), x32, (2, 4, 5, 1), (3, 4, 0, 5))
    rdm2_f_ovov += einsum(x415, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x415, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x415
    x416 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x416 += einsum(l2, (0, 1, 2, 3), x19, (3, 4, 5, 1), (2, 4, 0, 5)) * 2.0
    del x19
    rdm2_f_ovov += einsum(x416, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x416, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x416, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_voov += einsum(x416, (0, 1, 2, 3), (2, 1, 0, 3))
    del x416
    x417 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x417 += einsum(x32, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 0), (5, 6, 1, 4))
    x418 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x418 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x418 += einsum(x417, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x417
    x419 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x419 += einsum(x418, (0, 1, 2, 3), (0, 1, 2, 3))
    x419 += einsum(x418, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x419 += einsum(x378, (0, 1, 2, 3), (1, 0, 2, 3))
    x420 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x420 += einsum(t1, (0, 1), x419, (2, 0, 3, 4), (2, 3, 4, 1)) * 0.5
    del x419
    rdm2_f_ovov += einsum(x420, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_ovov += einsum(x420, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_ovvo += einsum(x420, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x420, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x420, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x420, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x420, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vovo += einsum(x420, (0, 1, 2, 3), (2, 1, 3, 0))
    del x420
    x421 = np.zeros((nvir, nvir), dtype=types[float])
    x421 += einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    x422 = np.zeros((nvir, nvir), dtype=types[float])
    x422 += einsum(x421, (0, 1), (0, 1))
    x422 += einsum(x203, (0, 1), (0, 1)) * 0.24999999999999
    x422 += einsum(x205, (0, 1), (0, 1))
    del x205
    x422 += einsum(x206, (0, 1), (0, 1)) * -1.0
    del x206
    x422 += einsum(x207, (0, 1), (1, 0))
    del x207
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x422, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x422, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x422, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x422, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovvo += einsum(delta.oo, (0, 1), x422, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_ovvo += einsum(delta.oo, (0, 1), x422, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_voov += einsum(delta.oo, (0, 1), x422, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_voov += einsum(delta.oo, (0, 1), x422, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x422, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x422, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x422, (2, 3), (2, 0, 3, 1))
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x422, (2, 3), (2, 0, 3, 1))
    x423 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x423 += einsum(t2, (0, 1, 2, 3), x143, (0, 1, 4, 5, 2, 6), (4, 5, 3, 6)) * 0.5
    rdm2_f_ovov += einsum(x423, (0, 1, 2, 3), (1, 3, 0, 2)) * -1.0
    rdm2_f_ovov += einsum(x423, (0, 1, 2, 3), (1, 3, 0, 2)) * -1.0
    rdm2_f_vovo += einsum(x423, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    rdm2_f_vovo += einsum(x423, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    del x423
    x424 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x424 += einsum(t1, (0, 1), x151, (0, 2, 3, 4), (2, 3, 1, 4))
    rdm2_f_ovov += einsum(x424, (0, 1, 2, 3), (1, 3, 0, 2)) * -1.0
    rdm2_f_ovov += einsum(x424, (0, 1, 2, 3), (1, 3, 0, 2)) * -1.0
    del x424
    x425 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x425 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 3, 5, 1, 7, 2), (4, 6, 0, 7))
    rdm2_f_ovvo += einsum(x425, (0, 1, 2, 3), (1, 2, 3, 0)) * 0.25
    rdm2_f_ovvo += einsum(x425, (0, 1, 2, 3), (1, 2, 3, 0)) * 0.25
    rdm2_f_vovo += einsum(x425, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.25
    rdm2_f_vovo += einsum(x425, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.25
    del x425
    x426 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x426 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x426 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    x426 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    x426 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 5.0
    x427 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x427 += einsum(l3, (0, 1, 2, 3, 4, 5), x426, (4, 6, 5, 1, 7, 2), (6, 3, 7, 0)) * 0.25
    del x426
    rdm2_f_ovvo += einsum(x427, (0, 1, 2, 3), (0, 3, 2, 1))
    rdm2_f_ovvo += einsum(x427, (0, 1, 2, 3), (0, 3, 2, 1))
    rdm2_f_voov += einsum(x427, (0, 1, 2, 3), (3, 0, 1, 2))
    rdm2_f_voov += einsum(x427, (0, 1, 2, 3), (3, 0, 1, 2))
    del x427
    x428 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x428 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x428 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    x429 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x429 += einsum(l3, (0, 1, 2, 3, 4, 5), x428, (5, 6, 4, 2, 1, 7), (6, 3, 7, 0)) * 0.25
    del x428
    rdm2_f_ovvo += einsum(x429, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    rdm2_f_ovvo += einsum(x429, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    del x429
    x430 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x430 += einsum(x42, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    x430 += einsum(x42, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    x431 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x431 += einsum(t2, (0, 1, 2, 3), x430, (0, 1, 4, 5, 6, 2), (4, 5, 6, 3)) * 0.5
    del x430
    rdm2_f_ovvo += einsum(x431, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x431, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x431, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x431, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x431
    x432 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x432 += einsum(t2, (0, 1, 2, 3), x274, (1, 4, 5, 2), (4, 0, 5, 3))
    rdm2_f_ovvo += einsum(x432, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x432, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x432, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vovo += einsum(x432, (0, 1, 2, 3), (2, 1, 3, 0))
    del x432
    x433 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x433 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * 2.0
    x433 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x434 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x434 += einsum(t2, (0, 1, 2, 3), x433, (1, 4, 5, 3), (4, 0, 5, 2))
    del x433
    rdm2_f_ovvo += einsum(x434, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_ovvo += einsum(x434, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_vovo += einsum(x434, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x434, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x434
    x435 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x435 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 3, 5, 0, 7, 2), (4, 6, 1, 7))
    rdm2_f_ovvo += einsum(x435, (0, 1, 2, 3), (1, 2, 3, 0)) * -0.25
    rdm2_f_ovvo += einsum(x435, (0, 1, 2, 3), (1, 2, 3, 0)) * -0.25
    rdm2_f_voov += einsum(x435, (0, 1, 2, 3), (2, 1, 0, 3)) * -0.25
    rdm2_f_voov += einsum(x435, (0, 1, 2, 3), (2, 1, 0, 3)) * -0.25
    x436 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x436 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 6, 5, 1, 7, 2), (4, 6, 0, 7))
    rdm2_f_ovvo += einsum(x436, (0, 1, 2, 3), (1, 2, 3, 0)) * -0.25
    rdm2_f_ovvo += einsum(x436, (0, 1, 2, 3), (1, 2, 3, 0)) * -0.25
    rdm2_f_voov += einsum(x436, (0, 1, 2, 3), (2, 1, 0, 3)) * -0.25
    rdm2_f_voov += einsum(x436, (0, 1, 2, 3), (2, 1, 0, 3)) * -0.25
    del x436
    x437 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x437 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 5, 1), (3, 4, 0, 5))
    rdm2_f_ovvo += einsum(x437, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x437, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x437, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x437, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x438 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x438 += einsum(t3, (0, 1, 2, 3, 4, 5), x127, (1, 2, 6, 5, 4, 7), (0, 6, 3, 7)) * 0.25
    rdm2_f_ovvo += einsum(x438, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    rdm2_f_ovvo += einsum(x438, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    rdm2_f_voov += einsum(x438, (0, 1, 2, 3), (3, 0, 1, 2)) * -1.0
    rdm2_f_voov += einsum(x438, (0, 1, 2, 3), (3, 0, 1, 2)) * -1.0
    del x438
    x439 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x439 += einsum(t2, (0, 1, 2, 3), x250, (0, 1, 4, 5, 2, 6), (4, 5, 6, 3)) * 0.5
    rdm2_f_ovvo += einsum(x439, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x439, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x439, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x439, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x439
    x440 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x440 += einsum(l2, (0, 1, 2, 3), x92, (3, 4, 5, 1), (2, 4, 0, 5))
    del x92
    rdm2_f_ovvo += einsum(x440, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_ovvo += einsum(x440, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_voov += einsum(x440, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_voov += einsum(x440, (0, 1, 2, 3), (2, 1, 0, 3))
    del x440
    x441 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x441 += einsum(t1, (0, 1), x130, (2, 0, 3, 4), (2, 3, 1, 4)) * 0.5
    rdm2_f_ovvo += einsum(x441, (0, 1, 2, 3), (1, 3, 2, 0)) * -1.0
    rdm2_f_ovvo += einsum(x441, (0, 1, 2, 3), (1, 3, 2, 0)) * -1.0
    del x441
    x442 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x442 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x442 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1)) * -1.0
    x443 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x443 += einsum(t3, (0, 1, 2, 3, 4, 5), x442, (2, 1, 6, 5, 7, 3), (6, 0, 7, 4)) * 0.25
    del x442
    rdm2_f_voov += einsum(x443, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_voov += einsum(x443, (0, 1, 2, 3), (2, 1, 0, 3))
    del x443
    x444 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x444 += einsum(l3, (0, 1, 2, 3, 4, 5), x317, (5, 6, 3, 2, 7, 1), (4, 6, 0, 7)) * 0.25
    del x317
    rdm2_f_voov += einsum(x444, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x444, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x444
    x445 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x445 += einsum(x42, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -0.5
    x445 += einsum(x42, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    x446 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x446 += einsum(t2, (0, 1, 2, 3), x445, (4, 0, 1, 5, 6, 2), (4, 5, 6, 3)) * 2.0
    del x445
    rdm2_f_voov += einsum(x446, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x446, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x446, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vovo += einsum(x446, (0, 1, 2, 3), (2, 1, 3, 0))
    del x446
    x447 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x447 += einsum(l2, (0, 1, 2, 3), x6, (2, 4, 5, 1), (3, 4, 0, 5))
    rdm2_f_voov += einsum(x447, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x447, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x447
    x448 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x448 += einsum(l3, (0, 1, 2, 3, 4, 5), x349, (4, 6, 5, 1, 7, 2), (6, 3, 7, 0)) * 0.25
    rdm2_f_voov += einsum(x448, (0, 1, 2, 3), (3, 0, 1, 2))
    rdm2_f_voov += einsum(x448, (0, 1, 2, 3), (3, 0, 1, 2))
    del x448
    x449 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x449 += einsum(t1, (0, 1), x151, (2, 0, 3, 4), (2, 3, 1, 4))
    del x151
    rdm2_f_voov += einsum(x449, (0, 1, 2, 3), (3, 1, 0, 2)) * -1.0
    rdm2_f_voov += einsum(x449, (0, 1, 2, 3), (3, 1, 0, 2)) * -1.0
    del x449
    x450 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x450 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * 5.0
    x450 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1)) * -1.0
    x451 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x451 += einsum(t3, (0, 1, 2, 3, 4, 5), x450, (2, 6, 1, 5, 7, 4), (6, 0, 7, 3)) * 0.25
    del x450
    rdm2_f_vovo += einsum(x451, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x451, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x451
    x452 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x452 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x452 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    x452 += einsum(t3, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    x453 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x453 += einsum(l3, (0, 1, 2, 3, 4, 5), x452, (4, 6, 5, 1, 7, 2), (6, 3, 7, 0)) * 0.25
    del x452
    rdm2_f_vovo += einsum(x453, (0, 1, 2, 3), (3, 0, 2, 1))
    rdm2_f_vovo += einsum(x453, (0, 1, 2, 3), (3, 0, 2, 1))
    del x453
    x454 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x454 += einsum(l3, (0, 1, 2, 3, 4, 5), x315, (5, 6, 4, 2, 1, 7), (3, 6, 0, 7)) * 0.25
    del x315
    rdm2_f_vovo += einsum(x454, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x454, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x454
    x455 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x455 += einsum(t1, (0, 1), x130, (0, 2, 3, 4), (2, 3, 1, 4)) * 0.5
    rdm2_f_vovo += einsum(x455, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    rdm2_f_vovo += einsum(x455, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    del x455
    x456 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x456 += einsum(t3, (0, 1, 2, 3, 4, 5), x42, (2, 0, 1, 6, 7, 4), (6, 7, 5, 3))
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv += einsum(x456, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.50000000000001
    rdm2_f_ovvv += einsum(x456, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.50000000000001
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv += einsum(x456, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.50000000000001
    rdm2_f_vovv += einsum(x456, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.50000000000001
    del x456
    x457 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x457 += einsum(t3, (0, 1, 2, 3, 4, 5), x35, (0, 1, 2, 6, 4, 7), (6, 7, 3, 5))
    rdm2_f_ovvv += einsum(x457, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.16666666666667
    rdm2_f_ovvv += einsum(x457, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.16666666666667
    rdm2_f_vovv += einsum(x457, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.16666666666667
    rdm2_f_vovv += einsum(x457, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.16666666666667
    del x457
    x458 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x458 += einsum(l1, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_ovvv += einsum(x458, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x458, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vovv += einsum(x458, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vovv += einsum(x458, (0, 1, 2, 3), (1, 0, 3, 2))
    x459 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x459 += einsum(t2, (0, 1, 2, 3), x263, (1, 3, 4, 5), (0, 4, 2, 5))
    x460 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x460 += einsum(t3, (0, 1, 2, 3, 4, 5), x35, (0, 1, 2, 6, 3, 7), (6, 7, 5, 4))
    x461 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x461 += einsum(x238, (0, 1, 2, 3, 4, 5), x35, (0, 1, 2, 6, 7, 4), (6, 3, 5, 7)) * 0.16666666666667
    x462 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x462 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    x462 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x463 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x463 += einsum(x462, (0, 1, 2, 3), t3, (4, 1, 0, 5, 6, 3), (4, 2, 5, 6)) * 0.5
    del x462
    x464 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x464 += einsum(l3, (0, 1, 2, 3, 4, 5), x198, (5, 6, 4, 2, 7, 1), (3, 6, 0, 7)) * 0.25
    del x198
    x465 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x465 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    x465 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x465 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    x466 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x466 += einsum(l3, (0, 1, 2, 3, 4, 5), x465, (5, 6, 3, 2, 7, 1), (6, 4, 7, 0)) * 0.25
    del x465
    x467 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x467 += einsum(x42, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x467 += einsum(x35, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    x467 += einsum(x35, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 0.5
    x467 += einsum(x35, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -0.5
    x468 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x468 += einsum(t2, (0, 1, 2, 3), x467, (4, 1, 0, 5, 6, 2), (4, 5, 6, 3))
    del x467
    x469 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x469 += einsum(l2, (0, 1, 2, 3), x357, (3, 4, 1, 5), (4, 2, 5, 0))
    x470 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x470 += einsum(l2, (0, 1, 2, 3), x32, (2, 4, 1, 5), (3, 4, 0, 5))
    x471 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x471 += einsum(t1, (0, 1), x418, (2, 0, 3, 4), (2, 3, 4, 1)) * 0.5
    del x418
    x472 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x472 += einsum(x197, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.25
    del x197
    x472 += einsum(x176, (0, 1, 2, 3), (0, 1, 2, 3))
    x472 += einsum(x464, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x464
    x472 += einsum(x466, (0, 1, 2, 3), (1, 0, 3, 2))
    del x466
    x472 += einsum(x468, (0, 1, 2, 3), (0, 1, 2, 3))
    del x468
    x472 += einsum(x469, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x469
    x472 += einsum(x470, (0, 1, 2, 3), (0, 1, 2, 3))
    del x470
    x472 += einsum(x471, (0, 1, 2, 3), (0, 1, 2, 3))
    del x471
    x473 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x473 += einsum(t1, (0, 1), x472, (0, 2, 3, 4), (2, 3, 4, 1))
    del x472
    x474 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x474 += einsum(t2, (0, 1, 2, 3), x222, (1, 4, 5, 6, 2, 3), (4, 5, 0, 6)) * 0.5
    del x222
    x475 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x475 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x475 += einsum(x474, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x474
    x476 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x476 += einsum(t2, (0, 1, 2, 3), x475, (1, 0, 4, 5), (4, 5, 2, 3))
    del x475
    x477 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x477 += einsum(t2, (0, 1, 2, 3), x378, (1, 0, 4, 5), (4, 5, 2, 3)) * 0.25
    del x378
    x478 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x478 += einsum(x458, (0, 1, 2, 3), (0, 1, 2, 3))
    del x458
    x478 += einsum(x459, (0, 1, 2, 3), (0, 1, 2, 3))
    del x459
    x478 += einsum(x460, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.16666666666667
    del x460
    x478 += einsum(x461, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    del x461
    x478 += einsum(x463, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x463
    x478 += einsum(x473, (0, 1, 2, 3), (0, 1, 3, 2))
    del x473
    x478 += einsum(x476, (0, 1, 2, 3), (0, 1, 3, 2))
    del x476
    x478 += einsum(x477, (0, 1, 2, 3), (0, 1, 2, 3))
    del x477
    x478 += einsum(t1, (0, 1), x422, (2, 3), (0, 2, 1, 3))
    del x422
    rdm2_f_ovvv += einsum(x478, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x478, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_ovvv += einsum(x478, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x478, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x478, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x478, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vovv += einsum(x478, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x478, (0, 1, 2, 3), (1, 0, 3, 2))
    del x478
    x479 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x479 += einsum(t2, (0, 1, 2, 3), x128, (1, 4, 5, 3), (0, 2, 5, 4)) * -1.0
    rdm2_f_ovvv += einsum(x479, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_ovvv += einsum(x479, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    rdm2_f_ovvv += einsum(x479, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_ovvv += einsum(x479, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    rdm2_f_vovv += einsum(x479, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_vovv += einsum(x479, (0, 1, 2, 3), (2, 0, 3, 1)) * 0.5
    rdm2_f_vovv += einsum(x479, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_vovv += einsum(x479, (0, 1, 2, 3), (2, 0, 3, 1)) * 0.5
    del x479
    x480 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x480 += einsum(x258, (0, 1, 2, 3), x6, (0, 4, 2, 5), (4, 1, 3, 5))
    rdm2_f_ovvv += einsum(x480, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_ovvv += einsum(x480, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    rdm2_f_ovvv += einsum(x480, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_ovvv += einsum(x480, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    rdm2_f_vovv += einsum(x480, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x480, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    rdm2_f_vovv += einsum(x480, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x480, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x480
    x481 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x481 += einsum(x238, (0, 1, 2, 3, 4, 5), x35, (0, 2, 1, 6, 7, 5), (6, 3, 4, 7)) * 0.16666666666667
    rdm2_f_ovvv += einsum(x481, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_ovvv += einsum(x481, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    del x481
    x482 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x482 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.3333333333333333
    x482 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x483 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x483 += einsum(x482, (0, 1, 2, 3), t3, (4, 1, 0, 5, 3, 6), (4, 2, 6, 5)) * 1.5
    del x482
    rdm2_f_ovvv += einsum(x483, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_ovvv += einsum(x483, (0, 1, 2, 3), (0, 1, 3, 2))
    del x483
    x484 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x484 += einsum(t2, (0, 1, 2, 3), l3, (4, 3, 5, 6, 0, 1), (6, 4, 5, 2))
    x485 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x485 += einsum(t2, (0, 1, 2, 3), l3, (4, 3, 5, 0, 6, 1), (6, 4, 5, 2))
    x486 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x486 += einsum(x484, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x486 += einsum(x485, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x486 += einsum(x485, (0, 1, 2, 3), (0, 2, 1, 3))
    x487 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x487 += einsum(x32, (0, 1, 2, 3), x486, (0, 4, 3, 5), (1, 4, 5, 2)) * 0.5
    del x486
    rdm2_f_ovvv += einsum(x487, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x487, (0, 1, 2, 3), (0, 1, 2, 3))
    del x487
    x488 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x488 += einsum(t2, (0, 1, 2, 3), x270, (1, 4, 2, 5), (0, 4, 5, 3)) * 0.5
    del x270
    rdm2_f_ovvv += einsum(x488, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_ovvv += einsum(x488, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x488
    x489 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x489 += einsum(x6, (0, 1, 2, 3), l3, (4, 5, 2, 6, 0, 1), (6, 4, 5, 3))
    x490 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x490 += einsum(t2, (0, 1, 2, 3), x489, (1, 4, 3, 5), (0, 4, 5, 2)) * -1.0
    rdm2_f_ovvv += einsum(x490, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_ovvv += einsum(x490, (0, 1, 2, 3), (0, 1, 3, 2))
    del x490
    x491 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x491 += einsum(t1, (0, 1), x10, (2, 0, 3, 4), (2, 3, 1, 4)) * 3.0
    x492 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x492 += einsum(t1, (0, 1), x491, (0, 2, 3, 4), (2, 4, 3, 1)) * 0.5
    del x491
    rdm2_f_ovvv += einsum(x492, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_ovvv += einsum(x492, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x492
    x493 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x493 += einsum(l2, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 1), (4, 0, 5, 6))
    rdm2_f_ovvv += einsum(x493, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x493, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vovv += einsum(x493, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vovv += einsum(x493, (0, 1, 2, 3), (1, 0, 3, 2))
    del x493
    x494 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x494 += einsum(t3, (0, 1, 2, 3, 4, 5), x42, (2, 0, 1, 6, 5, 7), (6, 7, 3, 4))
    del x42
    rdm2_f_ovvv += einsum(x494, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.50000000000001
    rdm2_f_ovvv += einsum(x494, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.50000000000001
    rdm2_f_vovv += einsum(x494, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.50000000000001
    rdm2_f_vovv += einsum(x494, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.50000000000001
    del x494
    x495 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x495 += einsum(t3, (0, 1, 2, 3, 4, 5), x35, (0, 2, 1, 6, 7, 5), (6, 7, 3, 4))
    rdm2_f_ovvv += einsum(x495, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.50000000000001
    rdm2_f_ovvv += einsum(x495, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.50000000000001
    rdm2_f_vovv += einsum(x495, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.50000000000001
    rdm2_f_vovv += einsum(x495, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.50000000000001
    del x495
    x496 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x496 += einsum(x274, (0, 1, 2, 3), t3, (0, 4, 1, 5, 6, 3), (4, 2, 5, 6)) * 0.5
    del x274
    rdm2_f_ovvv += einsum(x496, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_ovvv += einsum(x496, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x496, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x496, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x496
    x497 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x497 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 2, 1))
    x497 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    x497 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1))
    x497 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2)) * -1.0
    x498 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x498 += einsum(t2, (0, 1, 2, 3), x497, (0, 1, 4, 5, 2, 6), (4, 5, 6, 3))
    del x497
    x499 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x499 += einsum(x263, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x499 += einsum(x263, (0, 1, 2, 3), (0, 2, 1, 3))
    x499 += einsum(x498, (0, 1, 2, 3), (0, 1, 2, 3))
    del x498
    x500 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x500 += einsum(t2, (0, 1, 2, 3), x499, (1, 4, 3, 5), (0, 4, 5, 2))
    del x499
    rdm2_f_ovvv += einsum(x500, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_ovvv += einsum(x500, (0, 1, 2, 3), (0, 1, 3, 2))
    del x500
    x501 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x501 += einsum(t2, (0, 1, 2, 3), x127, (0, 1, 4, 5, 2, 6), (4, 3, 5, 6)) * 0.5
    del x127
    rdm2_f_vvov += einsum(x501, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vvov += einsum(x501, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vvvo += einsum(x501, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_vvvo += einsum(x501, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    x502 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x502 += einsum(x263, (0, 1, 2, 3), (0, 1, 2, 3))
    x502 += einsum(x501, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    del x501
    x503 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x503 += einsum(t2, (0, 1, 2, 3), x502, (1, 4, 2, 5), (0, 4, 5, 3))
    rdm2_f_ovvv += einsum(x503, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_ovvv += einsum(x503, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x503
    x504 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x504 += einsum(x263, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x504 += einsum(x128, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    x505 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x505 += einsum(t2, (0, 1, 2, 3), x504, (1, 2, 4, 5), (0, 4, 5, 3)) * 0.5
    del x504
    rdm2_f_ovvv += einsum(x505, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_ovvv += einsum(x505, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x505
    x506 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x506 += einsum(l3, (0, 1, 2, 3, 4, 5), x349, (4, 6, 5, 1, 7, 2), (6, 3, 7, 0)) * 0.125
    del x349
    x507 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x507 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 2, 1))
    x507 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    x508 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x508 += einsum(t3, (0, 1, 2, 3, 4, 5), x507, (1, 2, 6, 4, 5, 7), (6, 0, 7, 3)) * 0.125
    x509 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x509 += einsum(t2, (0, 1, 2, 3), x250, (0, 1, 4, 5, 3, 6), (4, 5, 6, 2)) * 0.25
    del x250
    x510 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x510 += einsum(l2, (0, 1, 2, 3), x357, (3, 4, 1, 5), (4, 2, 5, 0)) * 0.5
    x511 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x511 += einsum(t1, (0, 1), x130, (2, 0, 3, 4), (2, 3, 1, 4)) * 0.25
    del x130
    x512 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x512 += einsum(x437, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x437
    x512 += einsum(x435, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.125
    del x435
    x512 += einsum(x183, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.125
    del x183
    x512 += einsum(x176, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x176
    x512 += einsum(x506, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x506
    x512 += einsum(x508, (0, 1, 2, 3), (0, 1, 2, 3))
    del x508
    x512 += einsum(x509, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x509
    x512 += einsum(x510, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x510
    x512 += einsum(x511, (0, 1, 2, 3), (0, 1, 3, 2))
    del x511
    x513 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x513 += einsum(t1, (0, 1), x512, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    del x512
    rdm2_f_ovvv += einsum(x513, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_ovvv += einsum(x513, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x513, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x513, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x513
    x514 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x514 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    del x23
    x514 += einsum(x331, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x331
    x515 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x515 += einsum(t2, (0, 1, 2, 3), x514, (0, 1, 4, 5), (4, 5, 3, 2))
    del x514
    rdm2_f_ovvv += einsum(x515, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_ovvv += einsum(x515, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x515, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x515, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x515
    x516 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x516 += einsum(t2, (0, 1, 2, 3), x33, (0, 1, 4, 5), (4, 3, 2, 5)) * -0.5
    del x33
    rdm2_f_ovvv += einsum(x516, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    rdm2_f_ovvv += einsum(x516, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    rdm2_f_vovv += einsum(x516, (0, 1, 2, 3), (3, 0, 1, 2)) * -1.0
    rdm2_f_vovv += einsum(x516, (0, 1, 2, 3), (3, 0, 1, 2)) * -1.0
    del x516
    x517 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x517 += einsum(t2, (0, 1, 2, 3), x143, (0, 1, 4, 5, 6, 2), (4, 5, 3, 6)) * 0.5
    del x143
    x518 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x518 += einsum(x399, (0, 1, 2, 3), (0, 1, 2, 3))
    del x399
    x518 += einsum(x374, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x374
    x518 += einsum(x375, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x375
    x518 += einsum(x400, (0, 1, 2, 3), (0, 1, 2, 3))
    del x400
    x518 += einsum(x517, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x517
    x519 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x519 += einsum(t1, (0, 1), x518, (0, 2, 3, 4), (2, 3, 4, 1))
    del x518
    rdm2_f_ovvv += einsum(x519, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_ovvv += einsum(x519, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x519, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x519, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x519
    x520 = np.zeros((nvir, nvir), dtype=types[float])
    x520 += einsum(l3, (0, 1, 2, 3, 4, 5), x204, (4, 3, 5, 1, 6, 2), (0, 6)) * 0.041666666666665
    del x204
    x521 = np.zeros((nvir, nvir), dtype=types[float])
    x521 += einsum(l3, (0, 1, 2, 3, 4, 5), x45, (4, 5, 3, 6, 1, 2), (0, 6)) * 0.041666666666665
    del x45
    x522 = np.zeros((nvir, nvir), dtype=types[float])
    x522 += einsum(t2, (0, 1, 2, 3), x96, (0, 1, 2, 4), (3, 4))
    del x96
    x523 = np.zeros((nvir, nvir), dtype=types[float])
    x523 += einsum(x421, (0, 1), (0, 1)) * 0.5
    del x421
    x523 += einsum(x203, (0, 1), (0, 1)) * 0.124999999999995
    del x203
    x523 += einsum(x520, (0, 1), (0, 1))
    del x520
    x523 += einsum(x521, (0, 1), (0, 1)) * -1.0
    del x521
    x523 += einsum(x522, (0, 1), (1, 0))
    del x522
    rdm2_f_ovvv += einsum(t1, (0, 1), x523, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_ovvv += einsum(t1, (0, 1), x523, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_vovv += einsum(t1, (0, 1), x523, (2, 3), (2, 0, 3, 1)) * 2.0
    rdm2_f_vovv += einsum(t1, (0, 1), x523, (2, 3), (2, 0, 3, 1)) * 2.0
    del x523
    x524 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x524 += einsum(x178, (0, 1, 2, 3, 4, 5), x35, (0, 2, 1, 6, 7, 5), (6, 7, 3, 4)) * 0.16666666666667
    del x35, x178
    rdm2_f_vovv += einsum(x524, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x524, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x524
    x525 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x525 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    x525 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * 3.0
    x526 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x526 += einsum(x525, (0, 1, 2, 3), t3, (4, 1, 0, 5, 3, 6), (4, 2, 6, 5)) * 0.5
    del x525
    rdm2_f_vovv += einsum(x526, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x526, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x526
    x527 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x527 += einsum(x484, (0, 1, 2, 3), (0, 1, 2, 3))
    del x484
    x527 += einsum(x485, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x527 += einsum(x485, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x528 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x528 += einsum(x32, (0, 1, 2, 3), x527, (0, 4, 3, 5), (1, 4, 5, 2))
    del x527
    rdm2_f_vovv += einsum(x528, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x528, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x528
    x529 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x529 += einsum(x489, (0, 1, 2, 3), x6, (0, 4, 2, 5), (4, 1, 3, 5)) * -1.0
    del x6, x489
    rdm2_f_vovv += einsum(x529, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vovv += einsum(x529, (0, 1, 2, 3), (1, 0, 3, 2))
    del x529
    x530 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x530 += einsum(t2, (0, 1, 2, 3), x128, (1, 4, 5, 2), (0, 3, 5, 4)) * -0.5
    rdm2_f_vovv += einsum(x530, (0, 1, 2, 3), (2, 0, 1, 3))
    rdm2_f_vovv += einsum(x530, (0, 1, 2, 3), (2, 0, 1, 3))
    del x530
    x531 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x531 += einsum(t1, (0, 1), x10, (2, 0, 3, 4), (2, 3, 1, 4))
    del x10
    x532 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x532 += einsum(t1, (0, 1), x531, (0, 2, 3, 4), (2, 4, 3, 1)) * 1.5
    del x531
    rdm2_f_vovv += einsum(x532, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vovv += einsum(x532, (0, 1, 2, 3), (1, 0, 3, 2))
    del x532
    x533 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x533 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 2, 1))
    x533 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1)) * -1.0
    x533 += einsum(l3, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2)) * 4.0
    x534 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x534 += einsum(t2, (0, 1, 2, 3), x533, (0, 1, 4, 2, 5, 6), (4, 5, 6, 3))
    del x533
    x535 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x535 += einsum(l3, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 1, 2))
    x535 += einsum(l3, (0, 1, 2, 3, 4, 5), (3, 5, 4, 0, 2, 1))
    x536 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x536 += einsum(t2, (0, 1, 2, 3), x535, (1, 4, 0, 5, 6, 3), (4, 5, 6, 2)) * 2.0
    del x535
    x537 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x537 += einsum(x485, (0, 1, 2, 3), (0, 1, 2, 3))
    x537 += einsum(x485, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x537 += einsum(x534, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x534
    x537 += einsum(x536, (0, 1, 2, 3), (0, 1, 2, 3))
    del x536
    x538 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x538 += einsum(t2, (0, 1, 2, 3), x537, (1, 3, 4, 5), (0, 4, 5, 2)) * 0.5
    del x537
    rdm2_f_vovv += einsum(x538, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_vovv += einsum(x538, (0, 1, 2, 3), (1, 0, 2, 3))
    del x538
    x539 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x539 += einsum(t2, (0, 1, 2, 3), x507, (0, 1, 4, 5, 2, 6), (4, 5, 6, 3))
    x540 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x540 += einsum(x47, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    del x47
    x540 += einsum(x539, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x539
    x541 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x541 += einsum(t2, (0, 1, 2, 3), x540, (1, 2, 4, 5), (0, 4, 5, 3)) * 0.5
    del x540
    rdm2_f_vovv += einsum(x541, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vovv += einsum(x541, (0, 1, 2, 3), (1, 0, 3, 2))
    del x541
    x542 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x542 += einsum(t2, (0, 1, 2, 3), x502, (1, 2, 4, 5), (0, 4, 5, 3))
    del x502
    rdm2_f_vovv += einsum(x542, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x542, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x542
    x543 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x543 += einsum(t1, (0, 1), l2, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_vvov += einsum(x543, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvov += einsum(x543, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvvo += einsum(x543, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vvvo += einsum(x543, (0, 1, 2, 3), (2, 1, 3, 0))
    x544 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x544 += einsum(x543, (0, 1, 2, 3), (0, 1, 2, 3))
    x544 += einsum(x485, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x485
    rdm2_f_vvov += einsum(x544, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvov += einsum(x544, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vvov += einsum(x544, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_vvov += einsum(x544, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vvvo += einsum(x544, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_vvvo += einsum(x544, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vvvo += einsum(x544, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_vvvo += einsum(x544, (0, 1, 2, 3), (2, 1, 3, 0))
    del x544
    x545 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x545 += einsum(x357, (0, 1, 2, 3), l3, (4, 2, 5, 1, 0, 6), (6, 3, 4, 5))
    del x357
    rdm2_f_vvov += einsum(x545, (0, 1, 2, 3), (3, 2, 0, 1))
    rdm2_f_vvov += einsum(x545, (0, 1, 2, 3), (3, 2, 0, 1))
    rdm2_f_vvvo += einsum(x545, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    rdm2_f_vvvo += einsum(x545, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x545
    x546 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x546 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 4, 5, 6, 1, 7), (0, 2, 6, 7))
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv += einsum(x546, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.50000000000001
    rdm2_f_vvvv += einsum(x546, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.50000000000001
    del x546
    x547 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x547 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (5, 3, 4, 6, 2, 7), (0, 1, 6, 7))
    rdm2_f_vvvv += einsum(x547, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.16666666666667
    rdm2_f_vvvv += einsum(x547, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.16666666666667
    del x547
    x548 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x548 += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 5), (0, 1, 4, 5))
    rdm2_f_vvvv += einsum(x548, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vvvv += einsum(x548, (0, 1, 2, 3), (1, 0, 3, 2))
    x549 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x549 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (5, 3, 4, 2, 6, 7), (0, 1, 7, 6))
    x550 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x550 += einsum(l3, (0, 1, 2, 3, 4, 5), x238, (3, 5, 4, 6, 2, 7), (6, 7, 0, 1)) * 0.16666666666667
    x551 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x551 += einsum(x32, (0, 1, 2, 3), l3, (4, 5, 2, 6, 0, 1), (6, 4, 5, 3))
    del x32
    x552 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x552 += einsum(t2, (0, 1, 2, 3), x507, (0, 1, 4, 5, 2, 6), (4, 5, 6, 3)) * 0.5
    del x507
    x553 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x553 += einsum(x543, (0, 1, 2, 3), (0, 1, 2, 3))
    x553 += einsum(x551, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x551
    x553 += einsum(x552, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x552
    x554 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x554 += einsum(t1, (0, 1), x553, (0, 2, 3, 4), (2, 3, 4, 1))
    del x553
    x555 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x555 += einsum(x548, (0, 1, 2, 3), (1, 0, 3, 2))
    del x548
    x555 += einsum(x549, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.16666666666667
    del x549
    x555 += einsum(x550, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x550
    x555 += einsum(x554, (0, 1, 2, 3), (1, 0, 2, 3))
    del x554
    rdm2_f_vvvv += einsum(x555, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vvvv += einsum(x555, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvvv += einsum(x555, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vvvv += einsum(x555, (0, 1, 2, 3), (0, 1, 2, 3))
    del x555
    x556 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x556 += einsum(t1, (0, 1), x258, (0, 2, 3, 4), (2, 3, 1, 4))
    del x258
    rdm2_f_vvvv += einsum(x556, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvvv += einsum(x556, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vvvv += einsum(x556, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvvv += einsum(x556, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x556
    x557 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x557 += einsum(l3, (0, 1, 2, 3, 4, 5), x238, (3, 5, 4, 6, 7, 2), (6, 7, 0, 1)) * 0.16666666666667
    del x238
    rdm2_f_vvvv += einsum(x557, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_vvvv += einsum(x557, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    del x557
    x558 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x558 += einsum(t1, (0, 1), x543, (0, 2, 3, 4), (3, 2, 4, 1))
    del x543
    rdm2_f_vvvv += einsum(x558, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vvvv += einsum(x558, (0, 1, 2, 3), (1, 0, 3, 2))
    del x558
    x559 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x559 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 4, 5, 6, 7, 2), (0, 1, 6, 7))
    x560 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x560 += einsum(t1, (0, 1), x263, (0, 2, 3, 4), (2, 3, 1, 4))
    del x263
    x561 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x561 += einsum(x559, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.50000000000001
    del x559
    x561 += einsum(x560, (0, 1, 2, 3), (0, 1, 2, 3))
    del x560
    rdm2_f_vvvv += einsum(x561, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvvv += einsum(x561, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vvvv += einsum(x561, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvvv += einsum(x561, (0, 1, 2, 3), (1, 0, 3, 2))
    del x561
    x562 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x562 += einsum(t1, (0, 1), x128, (0, 2, 3, 4), (1, 3, 4, 2)) * -0.5
    del x128
    rdm2_f_vvvv += einsum(x562, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_vvvv += einsum(x562, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_vvvv += einsum(x562, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_vvvv += einsum(x562, (0, 1, 2, 3), (2, 1, 0, 3))
    del x562

    rdm2_f = pack_2e(rdm2_f_oooo, rdm2_f_ooov, rdm2_f_oovo, rdm2_f_ovoo, rdm2_f_vooo, rdm2_f_oovv, rdm2_f_ovov, rdm2_f_ovvo, rdm2_f_voov, rdm2_f_vovo, rdm2_f_vvoo, rdm2_f_ovvv, rdm2_f_vovv, rdm2_f_vvov, rdm2_f_vvvo, rdm2_f_vvvv)

    rdm2_f = rdm2_f.swapaxes(1, 2)

    return rdm2_f

