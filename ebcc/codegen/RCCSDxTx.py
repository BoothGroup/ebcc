# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), x0, (0, 1, 2, 3), ()) * 2.0
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x1 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x2 = np.zeros((nocc, nvir), dtype=types[float])
    x2 += einsum(f.ov, (0, 1), (0, 1))
    x2 += einsum(t1, (0, 1), x1, (0, 2, 3, 1), (2, 3))
    del x1
    e_cc += einsum(t1, (0, 1), x2, (0, 1), ()) * 2.0
    del x2

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5)) * 2.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (0, 4, 5, 3)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5)) * -1.0
    t2new += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (3, 2, 4, 1)) * -1.0
    x0 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x0 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    t1new += einsum(t2, (0, 1, 2, 3), x0, (1, 2, 3, 4), (0, 4)) * 2.0
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x2 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x2 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x2 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x2 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new += einsum(t2, (0, 1, 2, 3), x2, (4, 0, 1, 2), (4, 3)) * -1.0
    del x2
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x3 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x4 = np.zeros((nocc, nvir), dtype=types[float])
    x4 += einsum(t1, (0, 1), x3, (0, 2, 3, 1), (2, 3)) * 2.0
    x5 = np.zeros((nocc, nvir), dtype=types[float])
    x5 += einsum(f.ov, (0, 1), (0, 1))
    x5 += einsum(x4, (0, 1), (0, 1))
    del x4
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x6 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t1new += einsum(x5, (0, 1), x6, (0, 2, 3, 1), (2, 3))
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x7 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t1new += einsum(t1, (0, 1), x7, (0, 2, 1, 3), (2, 3)) * 2.0
    del x7
    x8 = np.zeros((nvir, nvir), dtype=types[float])
    x8 += einsum(f.vv, (0, 1), (0, 1))
    x8 += einsum(t1, (0, 1), x0, (0, 2, 1, 3), (2, 3)) * 2.0
    del x0
    t1new += einsum(t1, (0, 1), x8, (1, 2), (0, 2))
    del x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x9 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x10 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x11 = np.zeros((nocc, nocc), dtype=types[float])
    x11 += einsum(t1, (0, 1), x5, (2, 1), (2, 0))
    del x5
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum(f.oo, (0, 1), (0, 1))
    x12 += einsum(t2, (0, 1, 2, 3), x9, (1, 4, 3, 2), (4, 0)) * 2.0
    x12 += einsum(t1, (0, 1), x10, (2, 3, 0, 1), (3, 2)) * 2.0
    x12 += einsum(x11, (0, 1), (0, 1))
    t1new += einsum(t1, (0, 1), x12, (0, 2), (2, 1)) * -1.0
    del x12
    x13 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x13 += einsum(t1, (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new += einsum(t1, (0, 1), x13, (2, 3, 1, 4), (0, 2, 3, 4))
    del x13
    x14 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x14 += einsum(t1, (0, 1), x1, (2, 3, 4, 1), (2, 0, 4, 3))
    t2new += einsum(t2, (0, 1, 2, 3), x14, (4, 5, 0, 1), (5, 4, 3, 2))
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum(t2, (0, 1, 2, 3), x16, (4, 1, 5, 2), (4, 0, 3, 5))
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(t1, (0, 1), x1, (2, 3, 0, 4), (2, 3, 1, 4))
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * 2.0
    x19 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(t2, (0, 1, 2, 3), x19, (1, 4, 5, 3), (4, 0, 5, 2)) * 0.5
    del x19
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(x18, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x18
    x21 += einsum(x20, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x20
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(t2, (0, 1, 2, 3), x21, (4, 1, 5, 2), (4, 0, 5, 3)) * 2.0
    del x21
    x23 = np.zeros((nvir, nvir), dtype=types[float])
    x23 += einsum(t2, (0, 1, 2, 3), x3, (0, 1, 4, 2), (4, 3)) * 2.0
    x24 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x24 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x24 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x25 = np.zeros((nvir, nvir), dtype=types[float])
    x25 += einsum(t1, (0, 1), x24, (0, 2, 1, 3), (2, 3))
    del x24
    x26 = np.zeros((nvir, nvir), dtype=types[float])
    x26 += einsum(x23, (0, 1), (0, 1))
    del x23
    x26 += einsum(x25, (0, 1), (0, 1)) * -1.0
    del x25
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(x26, (0, 1), t2, (2, 3, 0, 4), (2, 3, 1, 4))
    del x26
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 2), (0, 4, 5, 3))
    x29 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x29 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x29 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3))
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x30 += einsum(t2, (0, 1, 2, 3), x29, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    del x29
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x31 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x32 += einsum(v.ovvv, (0, 1, 2, 3), x31, (4, 5, 3, 1), (0, 4, 5, 2))
    x33 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x33 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    x33 += einsum(x28, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x28
    x33 += einsum(x30, (0, 1, 2, 3), (0, 2, 1, 3))
    del x30
    x33 += einsum(x32, (0, 1, 2, 3), (2, 1, 0, 3))
    del x32
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x34 += einsum(t1, (0, 1), x33, (2, 3, 0, 4), (2, 3, 4, 1))
    del x33
    x35 = np.zeros((nocc, nocc), dtype=types[float])
    x35 += einsum(t2, (0, 1, 2, 3), x9, (1, 4, 3, 2), (4, 0))
    del x9
    x36 = np.zeros((nocc, nocc), dtype=types[float])
    x36 += einsum(t1, (0, 1), x10, (2, 3, 0, 1), (2, 3))
    del x10
    x37 = np.zeros((nocc, nocc), dtype=types[float])
    x37 += einsum(x35, (0, 1), (0, 1))
    del x35
    x37 += einsum(x36, (0, 1), (1, 0))
    del x36
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum(x37, (0, 1), t2, (2, 0, 3, 4), (1, 2, 4, 3)) * 2.0
    del x37
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3))
    del x17
    x39 += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x22
    x39 += einsum(x27, (0, 1, 2, 3), (1, 0, 3, 2))
    del x27
    x39 += einsum(x34, (0, 1, 2, 3), (0, 1, 3, 2))
    del x34
    x39 += einsum(x38, (0, 1, 2, 3), (1, 0, 3, 2))
    del x38
    t2new += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x39, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x39
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x42 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x42 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum(t2, (0, 1, 2, 3), x42, (4, 5, 0, 1), (4, 5, 2, 3))
    del x42
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum(t2, (0, 1, 2, 3), x16, (4, 1, 5, 3), (4, 0, 2, 5))
    del x16
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x45 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum(x41, (0, 1, 2, 3), x45, (1, 4, 5, 2), (4, 0, 5, 3)) * 2.0
    del x45
    x47 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x47 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x48 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x48 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x49 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x49 += einsum(t2, (0, 1, 2, 3), x1, (4, 5, 1, 2), (4, 0, 5, 3))
    del x1
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 0.5
    x50 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x50 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x51 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x51 += einsum(v.ooov, (0, 1, 2, 3), x50, (2, 4, 5, 3), (0, 1, 4, 5)) * 2.0
    del x50
    x52 = np.zeros((nocc, nvir), dtype=types[float])
    x52 += einsum(t1, (0, 1), x3, (0, 2, 3, 1), (2, 3))
    x53 = np.zeros((nocc, nvir), dtype=types[float])
    x53 += einsum(f.ov, (0, 1), (0, 1)) * 0.5
    x53 += einsum(x52, (0, 1), (0, 1))
    del x52
    x54 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x54 += einsum(x53, (0, 1), t2, (2, 3, 1, 4), (0, 2, 3, 4)) * 2.0
    del x53
    x55 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x55 += einsum(x47, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    del x47
    x55 += einsum(x48, (0, 1, 2, 3), (2, 0, 1, 3))
    del x48
    x55 += einsum(x49, (0, 1, 2, 3), (2, 0, 1, 3))
    del x49
    x55 += einsum(x51, (0, 1, 2, 3), (1, 2, 0, 3))
    del x51
    x55 += einsum(x54, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x54
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum(t1, (0, 1), x55, (0, 2, 3, 4), (2, 3, 4, 1))
    del x55
    x57 = np.zeros((nocc, nocc), dtype=types[float])
    x57 += einsum(f.oo, (0, 1), (0, 1))
    x57 += einsum(x11, (0, 1), (1, 0))
    del x11
    x58 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x58 += einsum(x57, (0, 1), t2, (2, 1, 3, 4), (0, 2, 4, 3))
    del x57
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3))
    del x40
    x59 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3))
    del x41
    x59 += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3))
    del x43
    x59 += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x44
    x59 += einsum(x46, (0, 1, 2, 3), (1, 0, 2, 3))
    del x46
    x59 += einsum(x56, (0, 1, 2, 3), (0, 1, 3, 2))
    del x56
    x59 += einsum(x58, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x58
    t2new += einsum(x59, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3))
    del x59
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x60 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x60 += einsum(t2, (0, 1, 2, 3), x3, (1, 4, 5, 3), (0, 4, 2, 5)) * 4.0
    del x3
    t2new += einsum(t2, (0, 1, 2, 3), x60, (4, 1, 5, 3), (4, 0, 5, 2))
    del x60
    x61 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x61 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x61 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (4, 0, 1, 5))
    t2new += einsum(x31, (0, 1, 2, 3), x61, (0, 4, 5, 1), (5, 4, 3, 2))
    del x31, x61
    x62 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x62 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x62 += einsum(t1, (0, 1), x14, (2, 3, 0, 4), (3, 2, 4, 1))
    del x14
    t2new += einsum(t1, (0, 1), x62, (2, 3, 0, 4), (2, 3, 1, 4))
    del x62
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x63 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    del x15
    t2new += einsum(t2, (0, 1, 2, 3), x63, (4, 1, 5, 2), (4, 0, 5, 3))
    del x63
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x64 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x64, (4, 1, 5, 2), (4, 0, 3, 5))
    del x64

    return {"t1new": t1new, "t2new": t2new}

def energy_perturbative(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    # T3 amplitude
    x0 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    t3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x0
    x1 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x1 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 6), (0, 4, 5, 2, 3, 6))
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x1
    e_ia = direct_sum("i-a->ia", np.diag(f.oo), np.diag(f.vv))
    t3 /= direct_sum("ia+jb+kc->ijkabc", e_ia, e_ia, e_ia)

    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 5, 0, 6, 1, 3), (4, 5, 6, 2))
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 0, 5, 6, 1, 3), (4, 5, 6, 2))
    x2 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x2 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 0.16666666666666666
    x2 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x2 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -0.16666666666666666
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(x0, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x3 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x3 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x1
    x3 += einsum(v.ovvv, (0, 1, 2, 3), x2, (0, 4, 5, 1, 6, 3), (4, 5, 6, 2)) * 6.0
    del x2
    e_pert = 0
    e_pert += einsum(l2, (0, 1, 2, 3), x3, (2, 3, 0, 1), ()) * 0.5
    del x3
    x4 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x4 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x4 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * 3.0
    x5 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x5 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    x5 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x6 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x6 += einsum(l2, (0, 1, 2, 3), x5, (2, 4, 5, 1, 0, 6), (3, 4, 5, 6)) * -1.0
    del x5
    e_pert += einsum(x4, (0, 1, 2, 3), x6, (1, 2, 0, 3), ()) * -0.5
    del x4, x6
    x7 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x7 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 1.5
    x7 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x7 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.5
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(x0, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.6666666666666666
    del x0
    x8 += einsum(v.ovvv, (0, 1, 2, 3), x7, (0, 4, 5, 1, 6, 3), (4, 5, 6, 2)) * 0.6666666666666666
    del x7
    e_pert += einsum(l2, (0, 1, 2, 3), x8, (2, 3, 1, 0), ()) * -1.5
    del x8
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 += einsum(l2, (0, 1, 2, 3), t3, (4, 2, 5, 6, 0, 1), (3, 4, 5, 6))
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x10 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x11 += einsum(x9, (0, 1, 2, 3), (0, 2, 1, 3))
    del x9
    x11 += einsum(x10, (0, 1, 2, 3), t3, (4, 5, 0, 6, 2, 3), (1, 4, 5, 6)) * -1.0
    del x10
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x11, (1, 0, 2, 3), ()) * -1.0
    del x11
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x12 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -0.3333333333333333
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x13 += einsum(x12, (0, 1, 2, 3), t3, (4, 5, 0, 6, 2, 3), (1, 4, 5, 6))
    del x12
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x13, (1, 2, 0, 3), ()) * -3.0
    del x13
    x14 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x14 += einsum(l1, (0, 1), v.ovov, (2, 3, 4, 5), (1, 2, 4, 0, 5, 3)) * -1.0
    x14 += einsum(l1, (0, 1), v.ovov, (2, 3, 4, 5), (1, 2, 4, 0, 3, 5))
    x15 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x15 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    x15 += einsum(t3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    e_pert += einsum(x14, (0, 1, 2, 3, 4, 5), x15, (2, 0, 1, 5, 4, 3), ()) * -0.5
    del x14, x15
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.3333333333333333
    x16 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x17 = np.zeros((nocc, nvir), dtype=types[float])
    x17 += einsum(x16, (0, 1, 2, 3), t3, (4, 0, 1, 5, 2, 3), (4, 5))
    del x16
    e_pert += einsum(l1, (0, 1), x17, (1, 0), ()) * 3.0
    del x17

    return e_pert

