# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
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

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (4, 0, 5, 2)) * -1.0
    t2new += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (3, 2, 4, 1)) * -1.0
    t2new += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new += einsum(t2, (0, 1, 2, 3), v.oooo, (4, 0, 5, 1), (4, 5, 2, 3))
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    t2new += einsum(v.vvvv, (0, 1, 2, 3), x0, (4, 5, 3, 1), (5, 4, 0, 2))
    x1 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x1 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x1 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    t1new += einsum(x0, (0, 1, 2, 3), x1, (0, 2, 3, 4), (1, 4)) * 2.0
    del x1
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x2 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x3 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x3 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x3 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x3 += einsum(x2, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new += einsum(t2, (0, 1, 2, 3), x3, (4, 0, 1, 2), (4, 3)) * -1.0
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x4 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x5 = np.zeros((nocc, nvir), dtype=types[float])
    x5 += einsum(t1, (0, 1), x4, (0, 2, 1, 3), (2, 3)) * 2.0
    x6 = np.zeros((nocc, nvir), dtype=types[float])
    x6 += einsum(f.ov, (0, 1), (0, 1))
    x6 += einsum(x5, (0, 1), (0, 1))
    del x5
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x7 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new += einsum(x6, (0, 1), x7, (0, 2, 1, 3), (2, 3)) * 2.0
    del x7
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x8 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t1new += einsum(t1, (0, 1), x8, (0, 2, 1, 3), (2, 3)) * 2.0
    del x8
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x9 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x10 = np.zeros((nocc, nocc), dtype=types[float])
    x10 += einsum(t1, (0, 1), x9, (2, 3, 0, 1), (2, 3)) * 2.0
    del x9
    x11 = np.zeros((nocc, nocc), dtype=types[float])
    x11 += einsum(f.oo, (0, 1), (0, 1))
    x11 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x11 += einsum(x0, (0, 1, 2, 3), x4, (0, 4, 2, 3), (4, 1)) * 2.0
    x11 += einsum(x10, (0, 1), (1, 0))
    t1new += einsum(t1, (0, 1), x11, (0, 2), (2, 1)) * -1.0
    del x11
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new += einsum(x12, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x14 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x14 += einsum(t1, (0, 1), x2, (2, 3, 4, 1), (2, 0, 4, 3))
    t2new += einsum(t2, (0, 1, 2, 3), x14, (4, 5, 0, 1), (5, 4, 3, 2))
    del x14
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (0, 4, 3, 5))
    x18 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x18 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(t2, (0, 1, 2, 3), x18, (4, 5, 0, 1), (4, 5, 2, 3))
    del x18
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(t2, (0, 1, 2, 3), x16, (4, 1, 2, 5), (4, 0, 3, 5))
    x21 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x21 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x21 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(t1, (0, 1), x21, (2, 1, 3, 4), (2, 0, 3, 4)) * 2.0
    del x21
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(t2, (0, 1, 2, 3), x22, (1, 4, 5, 3), (4, 0, 5, 2))
    del x22
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x24 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x25 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x25 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x26 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x26 += einsum(t2, (0, 1, 2, 3), x2, (4, 5, 1, 2), (4, 0, 5, 3))
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 0.5
    x27 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x27 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum(v.ooov, (0, 1, 2, 3), x27, (2, 4, 5, 3), (4, 0, 1, 5)) * 2.0
    del x27
    x29 = np.zeros((nocc, nvir), dtype=types[float])
    x29 += einsum(t1, (0, 1), x4, (0, 2, 1, 3), (2, 3))
    x30 = np.zeros((nocc, nvir), dtype=types[float])
    x30 += einsum(f.ov, (0, 1), (0, 1)) * 0.5
    x30 += einsum(x29, (0, 1), (0, 1))
    del x29
    x31 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x31 += einsum(x30, (0, 1), t2, (2, 3, 1, 4), (0, 2, 3, 4)) * 2.0
    del x30
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x32 += einsum(x24, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    del x24
    x32 += einsum(x25, (0, 1, 2, 3), (2, 0, 1, 3))
    del x25
    x32 += einsum(x26, (0, 1, 2, 3), (2, 0, 1, 3))
    del x26
    x32 += einsum(x28, (0, 1, 2, 3), (2, 0, 1, 3))
    del x28
    x32 += einsum(x31, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x31
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum(t1, (0, 1), x32, (0, 2, 3, 4), (2, 3, 4, 1))
    del x32
    x34 = np.zeros((nocc, nocc), dtype=types[float])
    x34 += einsum(t1, (0, 1), x6, (2, 1), (2, 0)) * 0.5
    del x6
    x35 = np.zeros((nocc, nocc), dtype=types[float])
    x35 += einsum(f.oo, (0, 1), (0, 1)) * 0.5
    x35 += einsum(x34, (0, 1), (1, 0))
    del x34
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 += einsum(x35, (0, 1), t2, (2, 1, 3, 4), (0, 2, 4, 3)) * 2.0
    del x35
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x15
    x37 += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x37 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3))
    del x17
    x37 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x19
    x37 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3))
    del x20
    x37 += einsum(x23, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x23
    x37 += einsum(x33, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x33
    x37 += einsum(x36, (0, 1, 2, 3), (0, 1, 3, 2))
    del x36
    t2new += einsum(x37, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x37, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x37
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum(t2, (0, 1, 2, 3), x38, (4, 1, 5, 2), (4, 0, 3, 5))
    del x38
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x40 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x41 = np.zeros((nvir, nvir), dtype=types[float])
    x41 += einsum(t2, (0, 1, 2, 3), x40, (0, 1, 4, 2), (4, 3))
    del x40
    x42 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x42 += einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x42 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x43 = np.zeros((nvir, nvir), dtype=types[float])
    x43 += einsum(t1, (0, 1), x42, (0, 1, 2, 3), (2, 3))
    del x42
    x44 = np.zeros((nvir, nvir), dtype=types[float])
    x44 += einsum(x41, (0, 1), (0, 1))
    del x41
    x44 += einsum(x43, (0, 1), (1, 0)) * -1.0
    del x43
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum(x44, (0, 1), t2, (2, 3, 0, 4), (2, 3, 1, 4))
    del x44
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x46 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * -0.5
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x47 += einsum(v.ovov, (0, 1, 2, 3), x46, (2, 4, 3, 5), (4, 0, 5, 1))
    del x46
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum(t2, (0, 1, 2, 3), x47, (4, 1, 5, 2), (4, 0, 5, 3)) * 2.0
    del x47
    x49 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x49 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 2), (0, 4, 5, 3))
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x50 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x51 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x51 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x51 += einsum(x2, (0, 1, 2, 3), (0, 2, 1, 3))
    del x2
    x52 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x52 += einsum(t2, (0, 1, 2, 3), x51, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    del x51
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x53 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x53 += einsum(x16, (0, 1, 2, 3), (1, 0, 2, 3))
    del x16
    x54 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x54 += einsum(t1, (0, 1), x53, (2, 3, 1, 4), (2, 3, 0, 4))
    del x53
    x55 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x55 += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x49
    x55 += einsum(x50, (0, 1, 2, 3), (0, 1, 2, 3))
    del x50
    x55 += einsum(x52, (0, 1, 2, 3), (0, 2, 1, 3))
    del x52
    x55 += einsum(x54, (0, 1, 2, 3), (2, 1, 0, 3))
    del x54
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum(t1, (0, 1), x55, (2, 3, 0, 4), (2, 3, 4, 1))
    del x55
    x57 = np.zeros((nocc, nocc), dtype=types[float])
    x57 += einsum(t2, (0, 1, 2, 3), x4, (1, 4, 3, 2), (4, 0))
    del x4
    x58 = np.zeros((nocc, nocc), dtype=types[float])
    x58 += einsum(x57, (0, 1), (0, 1))
    del x57
    x58 += einsum(x10, (0, 1), (1, 0))
    del x10
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum(x58, (0, 1), t2, (2, 0, 3, 4), (1, 2, 4, 3))
    del x58
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3))
    del x39
    x60 += einsum(x45, (0, 1, 2, 3), (1, 0, 3, 2))
    del x45
    x60 += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3))
    del x48
    x60 += einsum(x56, (0, 1, 2, 3), (0, 1, 3, 2))
    del x56
    x60 += einsum(x59, (0, 1, 2, 3), (1, 0, 3, 2))
    del x59
    t2new += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x60, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x60
    x61 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x61 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x61 += einsum(v.ovov, (0, 1, 2, 3), x0, (4, 5, 3, 1), (0, 5, 4, 2))
    del x0
    x62 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x62 += einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x62 += einsum(t1, (0, 1), x61, (0, 2, 3, 4), (3, 2, 4, 1))
    del x61
    t2new += einsum(t1, (0, 1), x62, (2, 3, 0, 4), (2, 3, 1, 4))
    del x62
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x63 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x63 += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x12
    t2new += einsum(t2, (0, 1, 2, 3), x63, (4, 1, 5, 3), (0, 4, 2, 5)) * 2.0
    del x63
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x64 += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3))
    del x13
    t2new += einsum(t2, (0, 1, 2, 3), x64, (4, 1, 5, 2), (4, 0, 5, 3))
    del x64

    return {"t1new": t1new, "t2new": t2new}

