# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(f.ov, (0, 1), t1, (0, 1), ())
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    e_cc += einsum(v.oovv, (0, 1, 2, 3), x0, (0, 1, 2, 3), ()) * 0.25
    del x0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t1new += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 0, 1, 5, 2, 3), (4, 5)) * 0.25
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t1new += einsum(t1, (0, 1), v.ovov, (2, 1, 0, 3), (2, 3)) * -1.0
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x0 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x1 += einsum(x0, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new += einsum(t2, (0, 1, 2, 3), x1, (4, 1, 0, 3), (4, 2)) * -0.5
    del x1
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x2 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    t1new += einsum(v.ovvv, (0, 1, 2, 3), x2, (0, 4, 2, 3), (4, 1)) * 0.5
    x3 = np.zeros((nocc, nvir), dtype=types[float])
    x3 += einsum(t1, (0, 1), v.oovv, (2, 0, 3, 1), (2, 3))
    x4 = np.zeros((nocc, nvir), dtype=types[float])
    x4 += einsum(f.ov, (0, 1), (0, 1))
    x4 += einsum(x3, (0, 1), (0, 1))
    del x3
    t1new += einsum(x4, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    t2new += einsum(x4, (0, 1), t3, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    x5 = np.zeros((nocc, nocc), dtype=types[float])
    x5 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x6 = np.zeros((nocc, nocc), dtype=types[float])
    x6 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 1), (2, 3))
    x7 = np.zeros((nocc, nocc), dtype=types[float])
    x7 += einsum(v.oovv, (0, 1, 2, 3), x2, (1, 4, 2, 3), (4, 0)) * -1.0
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum(f.oo, (0, 1), (0, 1)) * 2.0
    x8 += einsum(x5, (0, 1), (0, 1)) * 2.0
    x8 += einsum(x6, (0, 1), (0, 1)) * 2.0
    x8 += einsum(x7, (0, 1), (1, 0))
    t1new += einsum(t1, (0, 1), x8, (0, 2), (2, 1)) * -0.5
    del x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 0, 1, 5, 6, 3), (4, 2, 5, 6))
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(x0, (0, 1, 2, 3), t3, (4, 2, 1, 5, 6, 3), (0, 4, 5, 6))
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum(t1, (0, 1), x4, (2, 1), (2, 0))
    x13 = np.zeros((nocc, nocc), dtype=types[float])
    x13 += einsum(f.oo, (0, 1), (0, 1))
    x13 += einsum(x12, (0, 1), (1, 0))
    del x12
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(x13, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    del x13
    x15 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x15 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(x15, (0, 1, 2, 3), x2, (1, 2, 4, 5), (0, 3, 4, 5)) * 0.5
    x17 = np.zeros((nocc, nocc), dtype=types[float])
    x17 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    x18 = np.zeros((nocc, nocc), dtype=types[float])
    x18 += einsum(x6, (0, 1), (0, 1))
    x18 += einsum(x17, (0, 1), (1, 0)) * 0.5
    del x17
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(x18, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x18
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(x9, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x9
    x20 += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    del x10
    x20 += einsum(x11, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    del x11
    x20 += einsum(x14, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x14
    x20 += einsum(x16, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x16
    x20 += einsum(x19, (0, 1, 2, 3), (0, 1, 3, 2))
    del x19
    t2new += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3))
    del x20
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(t1, (0, 1), v.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(x21, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x21
    x23 += einsum(x22, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x22
    t2new += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x23, (0, 1, 2, 3), (0, 1, 3, 2))
    del x23
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 5, 0, 6, 2, 3), (4, 5, 6, 1))
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(t2, (0, 1, 2, 3), x25, (4, 1, 5, 3), (0, 4, 2, 5))
    x27 = np.zeros((nvir, nvir), dtype=types[float])
    x27 += einsum(t1, (0, 1), v.ovvv, (0, 2, 3, 1), (2, 3))
    x28 = np.zeros((nvir, nvir), dtype=types[float])
    x28 += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    x29 = np.zeros((nvir, nvir), dtype=types[float])
    x29 += einsum(x27, (0, 1), (0, 1))
    x29 += einsum(x28, (0, 1), (0, 1)) * 0.5
    del x28
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum(x29, (0, 1), t2, (2, 3, 4, 1), (2, 3, 4, 0)) * -1.0
    del x29
    x31 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x31 += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 5, 1, 6, 2, 3), (4, 5, 0, 6))
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x32 += einsum(v.ovvv, (0, 1, 2, 3), x2, (4, 5, 2, 3), (0, 4, 5, 1)) * 0.5
    x33 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x33 += einsum(x4, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum(x31, (0, 1, 2, 3), (2, 1, 0, 3)) * -0.5
    x34 += einsum(x32, (0, 1, 2, 3), (0, 2, 1, 3))
    del x32
    x34 += einsum(x33, (0, 1, 2, 3), (2, 1, 0, 3))
    del x33
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum(t1, (0, 1), x34, (0, 2, 3, 4), (2, 3, 4, 1))
    del x34
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x24
    x36 += einsum(x26, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x26
    x36 += einsum(x30, (0, 1, 2, 3), (1, 0, 2, 3))
    del x30
    x36 += einsum(x35, (0, 1, 2, 3), (1, 0, 3, 2))
    del x35
    t2new += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x36, (0, 1, 2, 3), (0, 1, 3, 2))
    del x36
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum(t2, (0, 1, 2, 3), x38, (4, 1, 5, 3), (4, 0, 2, 5)) * -1.0
    x40 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x40 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x41 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x41 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x42 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x42 += einsum(t2, (0, 1, 2, 3), x0, (4, 1, 5, 3), (4, 0, 5, 2))
    x43 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x43 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3))
    x43 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3))
    x43 += einsum(x42, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum(t1, (0, 1), x43, (2, 0, 3, 4), (2, 3, 4, 1))
    del x43
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x37
    x45 += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3))
    del x39
    x45 += einsum(x44, (0, 1, 2, 3), (0, 1, 3, 2))
    del x44
    t2new += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x45, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x45, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new += einsum(x45, (0, 1, 2, 3), (1, 0, 3, 2))
    del x45
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x46 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    t2new += einsum(v.vvvv, (0, 1, 2, 3), x46, (4, 5, 2, 3), (5, 4, 0, 1)) * -1.0
    del x46
    x47 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x47 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x47 += einsum(v.oovv, (0, 1, 2, 3), x2, (4, 5, 2, 3), (0, 1, 5, 4)) * -1.0
    t2new += einsum(x2, (0, 1, 2, 3), x47, (0, 1, 4, 5), (4, 5, 3, 2)) * -0.25
    del x47
    x48 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x48 += einsum(x15, (0, 1, 2, 3), t3, (4, 1, 2, 5, 6, 7), (0, 4, 3, 5, 6, 7))
    t3new = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    t3new += einsum(x48, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 0.5
    t3new += einsum(x48, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.5
    t3new += einsum(x48, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -0.5
    t3new += einsum(x48, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 0.5
    t3new += einsum(x48, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 0.5
    t3new += einsum(x48, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -0.5
    del x48
    x49 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x49 += einsum(v.ovov, (0, 1, 2, 3), t3, (4, 5, 2, 6, 7, 1), (4, 5, 0, 6, 7, 3))
    x50 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x50 += einsum(v.ooov, (0, 1, 2, 3), x2, (0, 1, 4, 5), (2, 3, 4, 5))
    x51 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x51 += einsum(t2, (0, 1, 2, 3), x50, (4, 3, 5, 6), (0, 1, 4, 2, 6, 5)) * 0.5
    del x50
    x52 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x52 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * 2.0
    x52 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x53 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x53 += einsum(v.ovvv, (0, 1, 2, 3), x52, (4, 5, 2, 3), (0, 4, 5, 1))
    x54 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x54 += einsum(t2, (0, 1, 2, 3), x53, (1, 4, 5, 6), (0, 5, 4, 2, 3, 6)) * 0.5
    del x53
    x55 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x55 += einsum(x49, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x49
    x55 += einsum(x51, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    del x51
    x55 += einsum(x54, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    del x54
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x55
    x56 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x56 += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 0, 1, 5, 6, 3), (4, 5, 6, 2))
    x57 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x57 += einsum(x4, (0, 1), t2, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x4
    x58 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x58 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x58 += einsum(x56, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    del x56
    x58 += einsum(x57, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x57
    x59 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x59 += einsum(t2, (0, 1, 2, 3), x58, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6)) * -1.0
    del x58
    x60 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (4, 5, 0, 2, 6, 7))
    x61 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x61 += einsum(v.oovv, (0, 1, 2, 3), x52, (4, 5, 2, 3), (0, 1, 4, 5))
    del x52
    x62 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x62 += einsum(t2, (0, 1, 2, 3), x61, (4, 1, 5, 6), (0, 6, 5, 4, 2, 3)) * -0.5
    x63 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum(x60, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x60
    x63 += einsum(x62, (0, 1, 2, 3, 4, 5), (2, 1, 3, 0, 5, 4)) * -1.0
    del x62
    x64 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x64 += einsum(t1, (0, 1), x63, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 6, 1))
    del x63
    x65 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x65 += einsum(x59, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    del x59
    x65 += einsum(x64, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    del x64
    t3new += einsum(x65, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x65, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x65, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x65, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x65, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x65, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x65, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x65, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x65, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x65
    x66 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x66 += einsum(x6, (0, 1), t3, (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * -1.0
    del x6
    x67 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x67 += einsum(x61, (0, 1, 2, 3), t3, (4, 1, 0, 5, 6, 7), (4, 3, 2, 5, 6, 7)) * -0.25
    del x61
    x68 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x68 += einsum(x66, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x66
    x68 += einsum(x67, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    del x67
    t3new += einsum(x68, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x68, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x68, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    del x68
    x69 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x69 += einsum(t2, (0, 1, 2, 3), x41, (4, 1, 5, 6), (0, 4, 5, 2, 3, 6))
    del x41
    x70 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x70 += einsum(x40, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x40
    x70 += einsum(x42, (0, 1, 2, 3), (0, 1, 2, 3))
    del x42
    x71 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x71 += einsum(t2, (0, 1, 2, 3), x70, (4, 5, 1, 6), (0, 4, 5, 2, 3, 6)) * -1.0
    del x70
    x72 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x72 += einsum(x69, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x69
    x72 += einsum(x71, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    del x71
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    del x72
    x73 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x73 += einsum(t1, (0, 1), x15, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x15
    x74 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x74 += einsum(t2, (0, 1, 2, 3), x73, (4, 1, 5, 6), (4, 0, 5, 6, 2, 3)) * -1.0
    del x73
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x74
    x75 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 5, 6, 7, 2, 3), (4, 5, 6, 0, 7, 1))
    x76 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x76 += einsum(t1, (0, 1), x75, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x75
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.5
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -0.5
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -0.5
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 0.5
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 0.5
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -0.5
    del x76
    x77 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x77 += einsum(x38, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 6, 7, 2)) * -1.0
    x78 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x78 += einsum(x0, (0, 1, 2, 3), x2, (1, 2, 4, 5), (0, 3, 4, 5))
    del x0
    x79 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x79 += einsum(t2, (0, 1, 2, 3), x78, (4, 3, 5, 6), (0, 1, 4, 2, 6, 5)) * 0.5
    del x78
    x80 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x80 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x80 += einsum(x31, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x31
    x81 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x81 += einsum(t2, (0, 1, 2, 3), x80, (4, 5, 1, 6), (0, 4, 5, 2, 3, 6)) * -1.0
    del x80
    x82 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x82 += einsum(x77, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    del x77
    x82 += einsum(x79, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    del x79
    x82 += einsum(x81, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x81
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x82
    x83 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x83 += einsum(x27, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -1.0
    del x27
    x84 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x84 += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 5, 6, 7, 2, 3), (4, 5, 6, 0, 1, 7))
    x85 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x85 += einsum(x2, (0, 1, 2, 3), x84, (4, 5, 6, 0, 1, 7), (4, 5, 6, 2, 3, 7)) * 0.25
    del x84
    x86 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x86 += einsum(x83, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x83
    x86 += einsum(x85, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x85
    t3new += einsum(x86, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x86, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x86, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    del x86
    x87 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x87 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 3), (0, 2, 4, 5))
    x88 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x88 += einsum(t2, (0, 1, 2, 3), x87, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    del x87
    x89 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x89 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x89 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3))
    del x25
    x90 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x90 += einsum(t2, (0, 1, 2, 3), x89, (4, 5, 6, 3), (0, 1, 4, 5, 2, 6)) * -1.0
    del x89
    x91 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x91 += einsum(t1, (0, 1), x90, (2, 3, 4, 0, 5, 6), (3, 2, 4, 5, 6, 1)) * -1.0
    del x90
    x92 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x92 += einsum(x88, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x88
    x92 += einsum(x91, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    del x91
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    del x92
    x93 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x93 += einsum(t2, (0, 1, 2, 3), x38, (4, 5, 6, 3), (4, 0, 1, 5, 2, 6)) * -1.0
    del x38
    x94 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x94 += einsum(t1, (0, 1), x93, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x93
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    del x94
    x95 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x95 += einsum(v.vvvv, (0, 1, 2, 3), t3, (4, 5, 6, 7, 2, 3), (4, 5, 6, 7, 0, 1))
    x96 = np.zeros((nvir, nvir), dtype=types[float])
    x96 += einsum(f.ov, (0, 1), t1, (0, 2), (1, 2))
    x97 = np.zeros((nvir, nvir), dtype=types[float])
    x97 += einsum(v.oovv, (0, 1, 2, 3), x2, (0, 1, 3, 4), (4, 2)) * -1.0
    del x2
    x98 = np.zeros((nvir, nvir), dtype=types[float])
    x98 += einsum(f.vv, (0, 1), (0, 1)) * -2.0
    x98 += einsum(x96, (0, 1), (0, 1)) * 2.0
    del x96
    x98 += einsum(x97, (0, 1), (1, 0))
    del x97
    x99 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x99 += einsum(x98, (0, 1), t3, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * 0.5
    del x98
    x100 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x100 += einsum(x95, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -0.5
    del x95
    x100 += einsum(x99, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    del x99
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x100
    x101 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x101 += einsum(t1, (0, 1), v.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    x102 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x102 += einsum(t2, (0, 1, 2, 3), x101, (4, 5, 1, 6), (0, 4, 5, 6, 2, 3)) * -1.0
    del x101
    x103 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x103 += einsum(t1, (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x104 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x104 += einsum(t2, (0, 1, 2, 3), x103, (4, 5, 6, 3), (4, 0, 1, 2, 5, 6)) * -1.0
    del x103
    x105 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x105 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x105 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * -0.99999999999999
    x106 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x106 += einsum(v.oovv, (0, 1, 2, 3), x105, (1, 4, 3, 5), (0, 4, 2, 5))
    del x105
    x107 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x107 += einsum(x106, (0, 1, 2, 3), t3, (4, 5, 0, 6, 7, 2), (4, 5, 1, 6, 7, 3)) * 1.00000000000001
    del x106
    x108 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x108 += einsum(x102, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    del x102
    x108 += einsum(x104, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    del x104
    x108 += einsum(x107, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x107
    t3new += einsum(x108, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x108, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x108, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x108, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x108, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x108, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x108, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x108, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x108, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x108
    x109 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x109 += einsum(v.oooo, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 7), (4, 0, 1, 5, 6, 7))
    x110 = np.zeros((nocc, nocc), dtype=types[float])
    x110 += einsum(f.oo, (0, 1), (0, 1)) * 2.0
    x110 += einsum(x5, (0, 1), (0, 1)) * 2.0
    del x5
    x110 += einsum(x7, (0, 1), (1, 0))
    del x7
    x111 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x111 += einsum(x110, (0, 1), t3, (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * 0.5
    del x110
    x112 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x112 += einsum(x109, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.5
    del x109
    x112 += einsum(x111, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    del x111
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    del x112

    return {"t1new": t1new, "t2new": t2new, "t3new": t3new}

def update_lams(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, l1=None, l2=None, l3=None, **kwargs):
    # L amplitudes
    l1new = np.zeros((nvir, nocc), dtype=types[float])
    l1new += einsum(l2, (0, 1, 2, 3), v.ovvv, (3, 4, 0, 1), (4, 2)) * -0.5
    l1new += einsum(f.ov, (0, 1), (1, 0))
    l1new += einsum(l1, (0, 1), v.ovov, (2, 0, 1, 3), (3, 2)) * -1.0
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum(v.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new += einsum(l2, (0, 1, 2, 3), v.vvvv, (4, 5, 0, 1), (4, 5, 2, 3)) * 0.5
    x0 = np.zeros((nocc, nocc), dtype=types[float])
    x0 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    x1 = np.zeros((nocc, nvir), dtype=types[float])
    x1 += einsum(t1, (0, 1), v.oovv, (2, 0, 3, 1), (2, 3))
    l1new += einsum(x0, (0, 1), x1, (1, 2), (2, 0)) * -1.0
    l3new = np.zeros((nvir, nvir, nvir, nocc, nocc, nocc), dtype=types[float])
    l3new += einsum(x1, (0, 1), l2, (2, 3, 4, 5), (2, 3, 1, 4, 5, 0))
    l3new += einsum(x1, (0, 1), l2, (2, 3, 4, 5), (2, 1, 3, 4, 5, 0)) * -1.0
    l3new += einsum(x1, (0, 1), l2, (2, 3, 4, 5), (1, 2, 3, 4, 5, 0))
    l3new += einsum(x1, (0, 1), l2, (2, 3, 4, 5), (2, 3, 1, 4, 0, 5)) * -1.0
    l3new += einsum(x1, (0, 1), l2, (2, 3, 4, 5), (2, 1, 3, 4, 0, 5))
    l3new += einsum(x1, (0, 1), l2, (2, 3, 4, 5), (1, 2, 3, 4, 0, 5)) * -1.0
    l3new += einsum(x1, (0, 1), l2, (2, 3, 4, 5), (2, 3, 1, 0, 4, 5))
    l3new += einsum(x1, (0, 1), l2, (2, 3, 4, 5), (2, 1, 3, 0, 4, 5)) * -1.0
    l3new += einsum(x1, (0, 1), l2, (2, 3, 4, 5), (1, 2, 3, 0, 4, 5))
    x2 = np.zeros((nvir, nvir), dtype=types[float])
    x2 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 4, 5, 6, 1, 2), (0, 6))
    l1new += einsum(x2, (0, 1), v.ovvv, (2, 0, 3, 1), (3, 2)) * 0.08333333333333
    x3 = np.zeros((nocc, nocc), dtype=types[float])
    x3 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 4, 5, 0, 1, 2), (3, 6))
    l1new += einsum(x3, (0, 1), v.ooov, (2, 1, 0, 3), (3, 2)) * 0.08333333333333
    x4 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x4 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x5 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x5 += einsum(x4, (0, 1, 2, 3), (2, 1, 0, 3))
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x8 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    x8 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    x9 = np.zeros((nocc, nvir), dtype=types[float])
    x9 += einsum(f.ov, (0, 1), (0, 1))
    x9 += einsum(x1, (0, 1), (0, 1))
    x10 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x10 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (0, 1, 4, 5))
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x11 += einsum(x4, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.5
    x12 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x12 += einsum(t1, (0, 1), x11, (2, 3, 4, 1), (0, 2, 3, 4)) * 4.0
    del x11
    x13 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x13 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x13 += einsum(x10, (0, 1, 2, 3), (3, 2, 1, 0))
    x13 += einsum(x12, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x14 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x14 += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 5, 6, 7, 2, 3), (4, 5, 6, 0, 1, 7))
    l2new += einsum(l3, (0, 1, 2, 3, 4, 5), x14, (3, 4, 5, 6, 7, 2), (0, 1, 6, 7)) * 0.08333333333333
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x15 += einsum(x4, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x16 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x16 += einsum(x14, (0, 1, 2, 3, 4, 5), (0, 1, 4, 3, 2, 5)) * 0.16666666666666
    del x14
    x16 += einsum(t2, (0, 1, 2, 3), x15, (4, 5, 6, 3), (0, 1, 6, 5, 4, 2))
    x17 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 5, 6, 7, 2, 3), (0, 4, 5, 6, 7, 1)) * 0.66666666666664
    x17 += einsum(x5, (0, 1, 2, 3), t3, (4, 5, 0, 6, 7, 3), (1, 4, 2, 5, 6, 7)) * -2.0
    x17 += einsum(t2, (0, 1, 2, 3), x8, (4, 5, 6, 3), (5, 0, 4, 1, 2, 6)) * -4.0
    del x8
    x17 += einsum(x9, (0, 1), t3, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6)) * -0.66666666666664
    x17 += einsum(t2, (0, 1, 2, 3), x13, (1, 4, 5, 6), (4, 0, 6, 5, 2, 3)) * -1.0
    x17 += einsum(t1, (0, 1), x16, (2, 3, 4, 0, 5, 6), (4, 3, 5, 2, 6, 1)) * 4.0
    del x16
    l1new += einsum(l3, (0, 1, 2, 3, 4, 5), x17, (6, 3, 4, 5, 2, 1), (0, 6)) * -0.125
    del x17
    x18 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x18 += einsum(t2, (0, 1, 2, 3), l3, (4, 5, 3, 6, 0, 1), (6, 4, 5, 2))
    l2new += einsum(x18, (0, 1, 2, 3), x5, (4, 5, 0, 3), (1, 2, 5, 4)) * -0.5
    x19 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x19 += einsum(t1, (0, 1), l2, (2, 3, 4, 0), (4, 2, 3, 1)) * 2.0
    x19 += einsum(x18, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l1new += einsum(v.vvvv, (0, 1, 2, 3), x19, (4, 3, 2, 1), (0, 4)) * -0.25
    del x19
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x20 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 5, 2, 3), (0, 1, 4, 5))
    x21 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x21 += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 5, 1, 6, 2, 3), (4, 5, 0, 6))
    x22 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x22 += einsum(t2, (0, 1, 2, 3), x15, (4, 1, 5, 3), (0, 4, 5, 2)) * 2.0
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x23 += einsum(x6, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x24 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x24 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x24 += einsum(x10, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x24 += einsum(x12, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x25 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x25 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x25 += einsum(x20, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.5
    x25 += einsum(x21, (0, 1, 2, 3), (2, 1, 0, 3)) * -0.5
    x25 += einsum(x22, (0, 1, 2, 3), (2, 0, 1, 3))
    x25 += einsum(x9, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    x25 += einsum(t1, (0, 1), x23, (2, 3, 1, 4), (3, 2, 0, 4)) * -2.0
    del x23
    x25 += einsum(t1, (0, 1), x24, (0, 2, 3, 4), (2, 4, 3, 1)) * -0.5
    del x24
    l1new += einsum(l2, (0, 1, 2, 3), x25, (4, 2, 3, 1), (0, 4)) * 0.5
    del x25
    x26 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(t1, (0, 1), l3, (2, 3, 1, 4, 5, 6), (4, 5, 6, 0, 2, 3))
    x27 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x27 += einsum(t1, (0, 1), x26, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x29 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x29 += einsum(t2, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 1), (5, 6, 0, 4))
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x30 += einsum(x28, (0, 1, 2, 3), (1, 0, 2, 3))
    x30 += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    l2new += einsum(v.ovvv, (0, 1, 2, 3), x30, (4, 5, 0, 1), (2, 3, 4, 5)) * -1.0
    x31 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x31 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    x32 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x32 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 7, 5, 0, 1, 2), (3, 4, 6, 7))
    x33 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x33 += einsum(x28, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x33 += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3))
    x34 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x34 += einsum(x31, (0, 1, 2, 3), (1, 0, 3, 2)) * -3.00000000000012
    x34 += einsum(x32, (0, 1, 2, 3), (0, 1, 3, 2))
    x34 += einsum(t1, (0, 1), x33, (2, 3, 4, 1), (2, 3, 0, 4)) * -6.00000000000024
    x35 = np.zeros((nocc, nocc), dtype=types[float])
    x35 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    x36 = np.zeros((nocc, nocc), dtype=types[float])
    x36 += einsum(x35, (0, 1), (0, 1))
    x36 += einsum(x3, (0, 1), (0, 1)) * 0.16666666666666
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x37 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), (1, 2, 3, 4))
    x37 += einsum(l2, (0, 1, 2, 3), t3, (4, 5, 3, 6, 0, 1), (2, 4, 5, 6)) * 0.5
    x37 += einsum(t3, (0, 1, 2, 3, 4, 5), x26, (6, 1, 2, 7, 5, 4), (6, 0, 7, 3)) * -0.5
    x37 += einsum(t2, (0, 1, 2, 3), x18, (4, 3, 2, 5), (4, 0, 1, 5)) * -0.25
    x37 += einsum(t2, (0, 1, 2, 3), x27, (4, 0, 1, 5, 6, 3), (4, 6, 5, 2)) * -0.5
    x37 += einsum(t2, (0, 1, 2, 3), x30, (4, 1, 5, 3), (4, 0, 5, 2)) * 2.0
    x37 += einsum(t1, (0, 1), x34, (0, 2, 3, 4), (2, 4, 3, 1)) * -0.16666666666666
    del x34
    x37 += einsum(t1, (0, 1), x36, (2, 3), (2, 0, 3, 1))
    del x36
    l1new += einsum(v.oovv, (0, 1, 2, 3), x37, (4, 0, 1, 3), (2, 4)) * 0.5
    del x37
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum(t2, (0, 1, 2, 3), x26, (4, 0, 1, 5, 3, 6), (4, 5, 6, 2))
    x39 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x39 += einsum(x28, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x39 += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    l1new += einsum(v.ovov, (0, 1, 2, 3), x39, (0, 4, 2, 3), (1, 4))
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x40 += einsum(t1, (0, 1), x39, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    l1new += einsum(v.ovvv, (0, 1, 2, 3), x40, (4, 0, 1, 3), (2, 4))
    del x40
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 5, 1), (2, 4, 0, 5))
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 4, 5, 7, 1, 2), (3, 6, 0, 7))
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3))
    x43 += einsum(x42, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    l1new += einsum(v.ovvv, (0, 1, 2, 3), x43, (4, 0, 1, 3), (2, 4)) * -1.0
    del x43
    x44 = np.zeros((nocc, nocc), dtype=types[float])
    x44 += einsum(x0, (0, 1), (0, 1)) * 12.00000000000048
    x44 += einsum(x35, (0, 1), (0, 1)) * 6.00000000000024
    x44 += einsum(x3, (0, 1), (0, 1))
    x45 = np.zeros((nocc, nvir), dtype=types[float])
    x45 += einsum(t1, (0, 1), (0, 1)) * -1.0
    x45 += einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3)) * -1.0
    x45 += einsum(l2, (0, 1, 2, 3), t3, (4, 2, 3, 5, 0, 1), (4, 5)) * -0.25
    x45 += einsum(t3, (0, 1, 2, 3, 4, 5), x26, (2, 0, 1, 6, 5, 4), (6, 3)) * -0.08333333333333
    x45 += einsum(t2, (0, 1, 2, 3), x39, (0, 1, 4, 3), (4, 2)) * -0.5
    x45 += einsum(t1, (0, 1), x44, (0, 2), (2, 1)) * 0.08333333333333
    l1new += einsum(x45, (0, 1), v.oovv, (2, 0, 3, 1), (3, 2)) * -1.0
    del x45
    x46 = np.zeros((nvir, nvir), dtype=types[float])
    x46 += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4))
    x47 = np.zeros((nvir, nvir), dtype=types[float])
    x47 += einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    x47 += einsum(x46, (0, 1), (0, 1)) * 0.5
    l1new += einsum(x47, (0, 1), v.ovvv, (2, 0, 3, 1), (3, 2))
    del x47
    x48 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x48 += einsum(x31, (0, 1, 2, 3), (1, 0, 3, 2)) * 3.00000000000012
    x48 += einsum(x32, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    l1new += einsum(v.ooov, (0, 1, 2, 3), x48, (2, 4, 0, 1), (3, 4)) * 0.08333333333333
    del x48
    x49 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x49 += einsum(t1, (0, 1), x33, (2, 3, 4, 1), (3, 2, 4, 0))
    l1new += einsum(v.ooov, (0, 1, 2, 3), x49, (4, 2, 0, 1), (3, 4)) * -0.5
    del x49
    x50 = np.zeros((nocc, nocc), dtype=types[float])
    x50 += einsum(x0, (0, 1), (0, 1)) * 2.0
    x50 += einsum(x35, (0, 1), (0, 1))
    l1new += einsum(x50, (0, 1), v.ooov, (2, 1, 0, 3), (3, 2)) * 0.5
    del x50
    x51 = np.zeros((nvir, nvir), dtype=types[float])
    x51 += einsum(t1, (0, 1), v.ovvv, (0, 2, 3, 1), (2, 3))
    x52 = np.zeros((nvir, nvir), dtype=types[float])
    x52 += einsum(f.vv, (0, 1), (0, 1))
    x52 += einsum(x51, (0, 1), (0, 1)) * -1.0
    l1new += einsum(l1, (0, 1), x52, (0, 2), (2, 1))
    del x52
    x53 = np.zeros((nocc, nocc), dtype=types[float])
    x53 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 1), (2, 3))
    x54 = np.zeros((nocc, nocc), dtype=types[float])
    x54 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    x55 = np.zeros((nocc, nocc), dtype=types[float])
    x55 += einsum(f.oo, (0, 1), (0, 1)) * 2.0
    x55 += einsum(x53, (0, 1), (1, 0)) * 2.0
    x55 += einsum(x54, (0, 1), (0, 1))
    x55 += einsum(t1, (0, 1), x9, (2, 1), (0, 2)) * 2.0
    del x9
    l1new += einsum(l1, (0, 1), x55, (1, 2), (0, 2)) * -0.5
    del x55
    x56 = np.zeros((nocc, nocc), dtype=types[float])
    x56 += einsum(x0, (0, 1), (0, 1)) * 2.0
    del x0
    x56 += einsum(x35, (0, 1), (0, 1))
    del x35
    x56 += einsum(x3, (0, 1), (0, 1)) * 0.16666666666666
    del x3
    l1new += einsum(f.ov, (0, 1), x56, (2, 0), (1, 2)) * -0.5
    del x56
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum(f.vv, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x58 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x58 += einsum(t1, (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x59 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x59 += einsum(t2, (0, 1, 2, 3), v.ovvv, (1, 4, 5, 3), (0, 2, 4, 5))
    x60 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x60 += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 0, 1, 5, 6, 3), (4, 5, 6, 2))
    x61 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x61 += einsum(t2, (0, 1, 2, 3), x5, (0, 1, 4, 5), (4, 2, 3, 5)) * 0.5
    del x5
    x62 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x62 += einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x62 += einsum(x58, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x58
    x62 += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x59
    x62 += einsum(x60, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.49999999999998
    del x60
    x62 += einsum(x61, (0, 1, 2, 3), (0, 2, 1, 3))
    del x61
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum(x62, (0, 1, 2, 3), l3, (4, 1, 2, 5, 6, 0), (5, 6, 4, 3)) * -0.5
    del x62
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x65 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x65 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x65 += einsum(x64, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    del x64
    x65 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    x66 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x66 += einsum(x65, (0, 1, 2, 3), x26, (4, 5, 0, 1, 2, 6), (4, 5, 6, 3))
    del x65
    x67 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x67 += einsum(t1, (0, 1), x4, (2, 3, 0, 4), (2, 3, 1, 4))
    x68 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x68 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    x68 += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x67
    x69 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x69 += einsum(x68, (0, 1, 2, 3), x26, (4, 5, 0, 1, 2, 6), (4, 5, 6, 3))
    del x68
    x70 = np.zeros((nvir, nvir), dtype=types[float])
    x70 += einsum(x46, (0, 1), (0, 1))
    del x46
    x70 += einsum(x2, (0, 1), (0, 1)) * 0.16666666666666
    del x2
    x71 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x71 += einsum(x70, (0, 1), v.oovv, (2, 3, 4, 1), (2, 3, 0, 4)) * -0.5
    del x70
    x72 = np.zeros((nvir, nvir), dtype=types[float])
    x72 += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    x73 = np.zeros((nvir, nvir), dtype=types[float])
    x73 += einsum(x51, (0, 1), (0, 1))
    del x51
    x73 += einsum(x72, (0, 1), (0, 1)) * 0.5
    del x72
    x74 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x74 += einsum(x73, (0, 1), l2, (2, 0, 3, 4), (3, 4, 2, 1)) * -1.0
    x75 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum(l1, (0, 1), x15, (1, 2, 3, 4), (2, 3, 0, 4))
    del x15
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum(f.ov, (0, 1), x39, (2, 3, 0, 4), (2, 3, 4, 1))
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum(x1, (0, 1), x39, (2, 3, 0, 4), (2, 3, 4, 1))
    del x39
    x78 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x78 += einsum(x57, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x57
    x78 += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3))
    del x63
    x78 += einsum(x66, (0, 1, 2, 3), (0, 1, 2, 3))
    del x66
    x78 += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3))
    del x69
    x78 += einsum(x71, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x71
    x78 += einsum(x74, (0, 1, 2, 3), (1, 0, 2, 3))
    del x74
    x78 += einsum(x75, (0, 1, 2, 3), (1, 0, 2, 3))
    del x75
    x78 += einsum(x76, (0, 1, 2, 3), (1, 0, 3, 2))
    del x76
    x78 += einsum(x77, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x77
    l2new += einsum(x78, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    l2new += einsum(x78, (0, 1, 2, 3), (3, 2, 0, 1))
    del x78
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum(x28, (0, 1, 2, 3), x4, (0, 4, 2, 5), (1, 4, 3, 5)) * -1.0
    x80 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x80 += einsum(v.ovvv, (0, 1, 2, 3), x18, (4, 5, 1, 3), (4, 0, 5, 2))
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum(t1, (0, 1), x29, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x41
    x82 += einsum(x42, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.49999999999998
    del x42
    x82 += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3))
    del x38
    x82 += einsum(x81, (0, 1, 2, 3), (0, 1, 2, 3))
    del x81
    x83 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x83 += einsum(v.oovv, (0, 1, 2, 3), x82, (4, 1, 5, 3), (4, 0, 5, 2)) * 0.5
    del x82
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x84 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    x85 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x85 += einsum(l2, (0, 1, 2, 3), x84, (3, 4, 1, 5), (2, 4, 0, 5))
    del x84
    x86 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x86 += einsum(v.ooov, (0, 1, 2, 3), x30, (4, 2, 1, 5), (0, 4, 3, 5)) * -1.0
    del x30
    x87 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x87 += einsum(f.ov, (0, 1), l1, (2, 3), (0, 3, 1, 2))
    x87 += einsum(l1, (0, 1), x1, (2, 3), (1, 2, 0, 3))
    x87 += einsum(x79, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x79
    x87 += einsum(x80, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x80
    x87 += einsum(x83, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x83
    x87 += einsum(x85, (0, 1, 2, 3), (0, 1, 2, 3))
    del x85
    x87 += einsum(x86, (0, 1, 2, 3), (1, 0, 3, 2))
    del x86
    l2new += einsum(x87, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new += einsum(x87, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new += einsum(x87, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new += einsum(x87, (0, 1, 2, 3), (3, 2, 1, 0))
    del x87
    x88 = np.zeros((nocc, nocc), dtype=types[float])
    x88 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x89 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x89 += einsum(x88, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3))
    x90 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x90 += einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x91 += einsum(x90, (0, 1, 2, 3), l3, (4, 5, 3, 6, 2, 1), (0, 6, 4, 5))
    del x90
    x92 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x92 += einsum(x1, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    x93 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x93 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x93 += einsum(x6, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    x94 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x94 += einsum(t1, (0, 1), x93, (2, 3, 4, 1), (0, 2, 3, 4)) * 2.0
    del x93
    x95 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x95 += einsum(t1, (0, 1), x13, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x13
    x96 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x96 += einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x96 += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    del x20
    x96 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.49999999999998
    del x21
    x96 += einsum(x92, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x92
    x96 += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3))
    del x22
    x96 += einsum(x94, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    del x94
    x96 += einsum(x95, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x95
    x97 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x97 += einsum(x96, (0, 1, 2, 3), l3, (4, 5, 3, 6, 0, 1), (6, 2, 4, 5)) * -0.5
    del x96
    x98 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x98 += einsum(x44, (0, 1), v.oovv, (2, 1, 3, 4), (0, 2, 3, 4)) * -0.08333333333333
    del x44
    x99 = np.zeros((nocc, nocc), dtype=types[float])
    x99 += einsum(t1, (0, 1), x1, (2, 1), (0, 2))
    x100 = np.zeros((nocc, nocc), dtype=types[float])
    x100 += einsum(x53, (0, 1), (0, 1))
    del x53
    x100 += einsum(x54, (0, 1), (1, 0)) * 0.5
    del x54
    x100 += einsum(x99, (0, 1), (1, 0))
    del x99
    x101 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x101 += einsum(x100, (0, 1), l2, (2, 3, 4, 1), (4, 0, 2, 3)) * -1.0
    x102 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x102 += einsum(x89, (0, 1, 2, 3), (0, 1, 3, 2))
    del x89
    x102 += einsum(x91, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    del x91
    x102 += einsum(x97, (0, 1, 2, 3), (0, 1, 2, 3))
    del x97
    x102 += einsum(x98, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x98
    x102 += einsum(x101, (0, 1, 2, 3), (0, 1, 3, 2))
    del x101
    l2new += einsum(x102, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new += einsum(x102, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x102
    x103 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x103 += einsum(f.oo, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3))
    x104 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x104 += einsum(l1, (0, 1), v.ovvv, (2, 0, 3, 4), (1, 2, 3, 4))
    x105 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x105 += einsum(x103, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x103
    x105 += einsum(x104, (0, 1, 2, 3), (0, 1, 3, 2))
    del x104
    l2new += einsum(x105, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new += einsum(x105, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    del x105
    x106 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x106 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x106 += einsum(x10, (0, 1, 2, 3), (1, 0, 3, 2))
    del x10
    x106 += einsum(x12, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    del x12
    l2new += einsum(l2, (0, 1, 2, 3), x106, (2, 3, 4, 5), (0, 1, 4, 5)) * 0.25
    x107 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x107 += einsum(x31, (0, 1, 2, 3), (1, 0, 3, 2))
    del x31
    x107 += einsum(x32, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.33333333333332
    del x32
    x107 += einsum(t1, (0, 1), x33, (2, 3, 4, 1), (3, 2, 4, 0)) * 2.0
    del x33
    l2new += einsum(v.oovv, (0, 1, 2, 3), x107, (4, 5, 0, 1), (2, 3, 5, 4)) * -0.25
    del x107
    x108 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x108 += einsum(x106, (0, 1, 2, 3), l3, (4, 5, 6, 7, 0, 1), (7, 2, 3, 4, 5, 6)) * -0.25
    del x106
    x109 = np.zeros((nocc, nocc), dtype=types[float])
    x109 += einsum(f.oo, (0, 1), (0, 1))
    x109 += einsum(x88, (0, 1), (1, 0))
    del x88
    x110 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x110 += einsum(x109, (0, 1), l3, (2, 3, 4, 5, 6, 0), (5, 6, 1, 2, 3, 4))
    del x109
    x111 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x111 += einsum(x108, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x108
    x111 += einsum(x110, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3))
    del x110
    l3new += einsum(x111, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2)) * -1.0
    l3new += einsum(x111, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2))
    l3new += einsum(x111, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * -1.0
    del x111
    x112 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x112 += einsum(x100, (0, 1), l3, (2, 3, 4, 5, 6, 1), (5, 6, 0, 2, 3, 4))
    del x100
    l3new += einsum(x112, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    l3new += einsum(x112, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -1.0
    l3new += einsum(x112, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0))
    del x112
    x113 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x113 += einsum(v.ooov, (0, 1, 2, 3), x26, (4, 2, 5, 1, 6, 7), (4, 5, 0, 6, 7, 3)) * -1.0
    x114 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x114 += einsum(x4, (0, 1, 2, 3), x26, (0, 4, 5, 2, 6, 7), (5, 4, 1, 6, 7, 3)) * -1.0
    x115 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x115 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x115 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    del x6
    x115 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.00000000000001
    del x7
    x116 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x116 += einsum(x115, (0, 1, 2, 3), l3, (4, 5, 2, 6, 7, 0), (6, 7, 1, 4, 5, 3))
    del x115
    x117 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x117 += einsum(x113, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x113
    x117 += einsum(x114, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x114
    x117 += einsum(x116, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x116
    l3new += einsum(x117, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2))
    l3new += einsum(x117, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2)) * -1.0
    l3new += einsum(x117, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2)) * -1.0
    l3new += einsum(x117, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0)) * -1.0
    l3new += einsum(x117, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0))
    l3new += einsum(x117, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0))
    l3new += einsum(x117, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 1, 0))
    l3new += einsum(x117, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0)) * -1.0
    l3new += einsum(x117, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0)) * -1.0
    del x117
    x118 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x118 += einsum(f.vv, (0, 1), l3, (2, 3, 1, 4, 5, 6), (4, 5, 6, 0, 2, 3))
    x119 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x119 += einsum(f.ov, (0, 1), x26, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    x120 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x120 += einsum(v.vvvv, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 7), (5, 6, 7, 4, 0, 1))
    x121 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x121 += einsum(v.ovvv, (0, 1, 2, 3), x26, (4, 5, 6, 0, 7, 1), (4, 5, 6, 7, 2, 3)) * -1.0
    x122 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x122 += einsum(t2, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 7), (5, 6, 7, 0, 1, 4))
    x123 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x123 += einsum(x122, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x122
    x123 += einsum(x27, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    del x27
    x124 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x124 += einsum(v.oovv, (0, 1, 2, 3), x123, (4, 5, 6, 0, 1, 7), (4, 5, 6, 7, 2, 3)) * 0.25
    del x123
    x125 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x125 += einsum(x118, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x118
    x125 += einsum(x119, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x119
    x125 += einsum(x120, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -0.5
    del x120
    x125 += einsum(x121, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x121
    x125 += einsum(x124, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x124
    l3new += einsum(x125, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0))
    l3new += einsum(x125, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0))
    l3new += einsum(x125, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * -1.0
    del x125
    x126 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x126 += einsum(x1, (0, 1), x26, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 6, 1))
    del x1, x26
    x127 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x127 += einsum(x73, (0, 1), l3, (2, 3, 0, 4, 5, 6), (4, 5, 6, 2, 3, 1))
    del x73
    x128 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x128 += einsum(x126, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x126
    x128 += einsum(x127, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    del x127
    l3new += einsum(x128, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0))
    l3new += einsum(x128, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * -1.0
    l3new += einsum(x128, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -1.0
    del x128
    x129 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x129 += einsum(l2, (0, 1, 2, 3), v.ovvv, (4, 1, 5, 6), (2, 3, 4, 0, 5, 6))
    x130 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x130 += einsum(v.oovv, (0, 1, 2, 3), x29, (4, 5, 1, 6), (5, 4, 0, 6, 2, 3))
    del x29
    x131 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x131 += einsum(x129, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x129
    x131 += einsum(x130, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -0.5
    del x130
    l3new += einsum(x131, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2))
    l3new += einsum(x131, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    l3new += einsum(x131, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 0, 2)) * -1.0
    l3new += einsum(x131, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0)) * -1.0
    l3new += einsum(x131, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -1.0
    l3new += einsum(x131, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0))
    l3new += einsum(x131, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 1, 0))
    l3new += einsum(x131, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0))
    l3new += einsum(x131, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0)) * -1.0
    del x131
    x132 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x132 += einsum(l2, (0, 1, 2, 3), v.ooov, (4, 5, 3, 6), (2, 4, 5, 0, 1, 6))
    x133 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x133 += einsum(v.oovv, (0, 1, 2, 3), x18, (4, 5, 6, 3), (4, 0, 1, 6, 5, 2))
    del x18
    x134 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x134 += einsum(x132, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    del x132
    x134 += einsum(x133, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 0.5
    del x133
    l3new += einsum(x134, (0, 1, 2, 3, 4, 5), (4, 3, 5, 0, 1, 2))
    l3new += einsum(x134, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2)) * -1.0
    l3new += einsum(x134, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 1, 2)) * -1.0
    l3new += einsum(x134, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2)) * -1.0
    l3new += einsum(x134, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2))
    l3new += einsum(x134, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    l3new += einsum(x134, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0))
    l3new += einsum(x134, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * -1.0
    l3new += einsum(x134, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -1.0
    del x134
    x135 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x135 += einsum(f.ov, (0, 1), l2, (2, 3, 4, 5), (0, 4, 5, 1, 2, 3))
    x135 += einsum(l1, (0, 1), v.oovv, (2, 3, 4, 5), (1, 2, 3, 0, 4, 5))
    l3new += einsum(x135, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2))
    l3new += einsum(x135, (0, 1, 2, 3, 4, 5), (4, 3, 5, 0, 1, 2)) * -1.0
    l3new += einsum(x135, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2))
    l3new += einsum(x135, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2)) * -1.0
    l3new += einsum(x135, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2))
    l3new += einsum(x135, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2)) * -1.0
    l3new += einsum(x135, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0))
    l3new += einsum(x135, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0)) * -1.0
    l3new += einsum(x135, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0))
    del x135
    x136 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x136 += einsum(l2, (0, 1, 2, 3), x4, (3, 4, 5, 6), (2, 5, 4, 0, 1, 6)) * -1.0
    del x4
    l3new += einsum(x136, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1)) * -1.0
    l3new += einsum(x136, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1)) * -1.0
    l3new += einsum(x136, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1))
    l3new += einsum(x136, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 0, 1))
    l3new += einsum(x136, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 0, 1))
    l3new += einsum(x136, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 0, 1)) * -1.0
    l3new += einsum(x136, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 1, 0)) * -1.0
    l3new += einsum(x136, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0)) * -1.0
    l3new += einsum(x136, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0))
    del x136
    x137 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x137 += einsum(v.oovv, (0, 1, 2, 3), x28, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3))
    del x28
    l3new += einsum(x137, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 0, 2))
    l3new += einsum(x137, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2)) * -1.0
    l3new += einsum(x137, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2)) * -1.0
    l3new += einsum(x137, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 2, 0)) * -1.0
    l3new += einsum(x137, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0))
    l3new += einsum(x137, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0))
    l3new += einsum(x137, (0, 1, 2, 3, 4, 5), (3, 5, 4, 2, 1, 0))
    l3new += einsum(x137, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0)) * -1.0
    l3new += einsum(x137, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0)) * -1.0
    del x137

    return {"l1new": l1new, "l2new": l2new, "l3new": l3new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, l1=None, l2=None, l3=None, **kwargs):
    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))

    # RDM1
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=types[float])
    rdm1_f_oo += einsum(delta.oo, (0, 1), (0, 1))
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=types[float])
    rdm1_f_ov += einsum(t1, (0, 1), (0, 1))
    rdm1_f_ov += einsum(l2, (0, 1, 2, 3), t3, (4, 2, 3, 5, 0, 1), (4, 5)) * 0.25
    rdm1_f_ov += einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3))
    rdm1_f_vo = np.zeros((nvir, nocc), dtype=types[float])
    rdm1_f_vo += einsum(l1, (0, 1), (0, 1))
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=types[float])
    rdm1_f_vv += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 4, 5, 6, 1, 2), (0, 6)) * 0.08333333333333
    rdm1_f_vv += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4)) * 0.5
    rdm1_f_vv += einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    x0 = np.zeros((nocc, nocc), dtype=types[float])
    x0 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    rdm1_f_oo += einsum(x0, (0, 1), (1, 0)) * -0.5
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    rdm1_f_oo += einsum(x1, (0, 1), (1, 0)) * -1.0
    x2 = np.zeros((nocc, nocc), dtype=types[float])
    x2 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 4, 5, 0, 1, 2), (3, 6))
    rdm1_f_oo += einsum(x2, (0, 1), (1, 0)) * -0.08333333333333
    x3 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(t1, (0, 1), l3, (2, 3, 1, 4, 5, 6), (4, 5, 6, 0, 2, 3))
    rdm1_f_ov += einsum(t3, (0, 1, 2, 3, 4, 5), x3, (0, 1, 2, 6, 5, 4), (6, 3)) * 0.08333333333333
    del x3
    x4 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x4 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x4 += einsum(t2, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 1), (5, 6, 0, 4)) * 0.5
    rdm1_f_ov += einsum(t2, (0, 1, 2, 3), x4, (0, 1, 4, 3), (4, 2)) * 0.5
    del x4
    x5 = np.zeros((nocc, nocc), dtype=types[float])
    x5 += einsum(x1, (0, 1), (0, 1))
    del x1
    x5 += einsum(x0, (0, 1), (0, 1)) * 0.5
    del x0
    x5 += einsum(x2, (0, 1), (0, 1)) * 0.08333333333333
    del x2
    rdm1_f_ov += einsum(t1, (0, 1), x5, (0, 2), (2, 1)) * -1.0
    del x5

    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])

    return rdm1_f

def make_rdm2_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, l1=None, l2=None, l3=None, **kwargs):
    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))

    # RDM2
    rdm2_f_oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_ovoo += einsum(delta.oo, (0, 1), l1, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovoo += einsum(delta.oo, (0, 1), l1, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_vooo += einsum(delta.oo, (0, 1), l1, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_vooo += einsum(delta.oo, (0, 1), l1, (2, 3), (2, 0, 3, 1))
    rdm2_f_oovv = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(l1, (0, 1), t3, (2, 3, 1, 4, 5, 0), (2, 3, 4, 5))
    rdm2_f_oovv += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovv += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2_f_ovov = np.zeros((nocc, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_ovov += einsum(l1, (0, 1), t1, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_ovvo += einsum(l1, (0, 1), t1, (2, 3), (2, 0, 3, 1))
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_voov += einsum(l1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_vovo += einsum(l1, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 5), (0, 1, 4, 5)) * 0.5
    rdm2_f_vvvv += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 4, 5, 6, 7, 2), (0, 1, 6, 7)) * 0.16666666666667
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x0 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 7, 5, 0, 1, 2), (3, 4, 6, 7))
    rdm2_f_oooo += einsum(x0, (0, 1, 2, 3), (2, 3, 1, 0)) * -0.16666666666667
    x1 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x1 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_oooo += einsum(x1, (0, 1, 2, 3), (3, 2, 1, 0)) * 0.5
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x2 += einsum(t1, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm2_f_ovoo += einsum(x2, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_vooo += einsum(x2, (0, 1, 2, 3), (3, 2, 1, 0))
    x3 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x3 += einsum(t1, (0, 1), x2, (2, 3, 4, 1), (2, 3, 0, 4))
    rdm2_f_oooo += einsum(x3, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    x4 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x4 += einsum(t2, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 1), (5, 6, 0, 4))
    rdm2_f_ovoo += einsum(x4, (0, 1, 2, 3), (2, 3, 1, 0)) * -0.5
    rdm2_f_vooo += einsum(x4, (0, 1, 2, 3), (3, 2, 1, 0)) * 0.5
    x5 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x5 += einsum(t1, (0, 1), x4, (2, 3, 4, 1), (2, 3, 0, 4))
    rdm2_f_oooo += einsum(x5, (0, 1, 2, 3), (2, 3, 1, 0)) * -0.5
    rdm2_f_oooo += einsum(x5, (0, 1, 2, 3), (3, 2, 1, 0)) * 0.5
    x6 = np.zeros((nocc, nocc), dtype=types[float])
    x6 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    x7 = np.zeros((nocc, nocc), dtype=types[float])
    x7 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 4, 5, 0, 1, 2), (3, 6))
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum(x6, (0, 1), (0, 1))
    x8 += einsum(x7, (0, 1), (0, 1)) * 0.16666666666666
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x8, (2, 3), (0, 3, 1, 2)) * -0.5
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x8, (2, 3), (0, 3, 2, 1)) * 0.5
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x8, (2, 3), (3, 0, 1, 2)) * 0.5
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x8, (2, 3), (3, 0, 2, 1)) * -0.5
    del x8
    x9 = np.zeros((nocc, nocc), dtype=types[float])
    x9 += einsum(l1, (0, 1), t1, (2, 0), (1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x9, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x9, (2, 3), (3, 0, 1, 2))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x9, (2, 3), (0, 3, 2, 1))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x9, (2, 3), (3, 0, 2, 1)) * -1.0
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov += einsum(x10, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo += einsum(x10, (0, 1, 2, 3), (2, 1, 3, 0))
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum(l2, (0, 1, 2, 3), t3, (4, 5, 3, 6, 0, 1), (2, 4, 5, 6))
    rdm2_f_ooov += einsum(x11, (0, 1, 2, 3), (1, 2, 0, 3)) * 0.5
    rdm2_f_oovo += einsum(x11, (0, 1, 2, 3), (1, 2, 3, 0)) * -0.5
    x12 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x12 += einsum(t2, (0, 1, 2, 3), l3, (4, 5, 3, 6, 0, 1), (6, 4, 5, 2))
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov += einsum(x12, (0, 1, 2, 3), (1, 2, 0, 3)) * 0.5
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo += einsum(x12, (0, 1, 2, 3), (1, 2, 3, 0)) * -0.5
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x13 += einsum(t2, (0, 1, 2, 3), x12, (4, 3, 2, 5), (4, 0, 1, 5)) * -1.0
    rdm2_f_ooov += einsum(x13, (0, 1, 2, 3), (1, 2, 0, 3)) * 0.25
    rdm2_f_oovo += einsum(x13, (0, 1, 2, 3), (1, 2, 3, 0)) * -0.25
    x14 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(t1, (0, 1), l3, (2, 3, 1, 4, 5, 6), (4, 5, 6, 0, 2, 3))
    x15 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum(t1, (0, 1), x14, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6))
    x16 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x16 += einsum(t2, (0, 1, 2, 3), x15, (4, 1, 0, 5, 6, 3), (4, 6, 5, 2))
    rdm2_f_ooov += einsum(x16, (0, 1, 2, 3), (1, 2, 0, 3)) * 0.5
    rdm2_f_oovo += einsum(x16, (0, 1, 2, 3), (1, 2, 3, 0)) * -0.5
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum(t3, (0, 1, 2, 3, 4, 5), x14, (6, 1, 2, 7, 4, 5), (6, 7, 0, 3))
    x18 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x18 += einsum(t1, (0, 1), x5, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x19 += einsum(x2, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x19 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x20 += einsum(t2, (0, 1, 2, 3), x19, (1, 4, 5, 3), (4, 5, 0, 2))
    x21 = np.zeros((nocc, nocc), dtype=types[float])
    x21 += einsum(x9, (0, 1), (0, 1)) * 12.00000000000048
    x21 += einsum(x6, (0, 1), (0, 1)) * 6.00000000000024
    x21 += einsum(x7, (0, 1), (0, 1))
    x22 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x22 += einsum(delta.oo, (0, 1), t1, (2, 3), (0, 1, 2, 3))
    x22 += einsum(x17, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.25
    x22 += einsum(x18, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    del x18
    x22 += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x20
    x22 += einsum(t1, (0, 1), x21, (2, 3), (0, 2, 3, 1)) * 0.08333333333333
    rdm2_f_ooov += einsum(x22, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_ooov += einsum(x22, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_oovo += einsum(x22, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovo += einsum(x22, (0, 1, 2, 3), (2, 0, 3, 1))
    del x22
    x23 = np.zeros((nocc, nvir), dtype=types[float])
    x23 += einsum(l1, (0, 1), t2, (2, 1, 3, 0), (2, 3))
    x24 = np.zeros((nocc, nvir), dtype=types[float])
    x24 += einsum(l2, (0, 1, 2, 3), t3, (4, 2, 3, 5, 0, 1), (4, 5))
    x25 = np.zeros((nocc, nvir), dtype=types[float])
    x25 += einsum(t3, (0, 1, 2, 3, 4, 5), x14, (0, 1, 2, 6, 4, 5), (6, 3))
    x26 = np.zeros((nocc, nvir), dtype=types[float])
    x26 += einsum(t2, (0, 1, 2, 3), x19, (0, 1, 4, 3), (4, 2)) * -0.5
    x27 = np.zeros((nocc, nvir), dtype=types[float])
    x27 += einsum(t1, (0, 1), x21, (0, 2), (2, 1)) * 0.08333333333333
    del x21
    x28 = np.zeros((nocc, nvir), dtype=types[float])
    x28 += einsum(x23, (0, 1), (0, 1)) * -1.0
    del x23
    x28 += einsum(x24, (0, 1), (0, 1)) * -0.25
    del x24
    x28 += einsum(x25, (0, 1), (0, 1)) * 0.08333333333333
    del x25
    x28 += einsum(x26, (0, 1), (0, 1))
    del x26
    x28 += einsum(x27, (0, 1), (0, 1))
    del x27
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x28, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_ooov += einsum(delta.oo, (0, 1), x28, (2, 3), (2, 0, 1, 3))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x28, (2, 3), (0, 2, 3, 1))
    rdm2_f_oovo += einsum(delta.oo, (0, 1), x28, (2, 3), (2, 0, 3, 1)) * -1.0
    x29 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x29 += einsum(x1, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x29 += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x29 += einsum(x0, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.33333333333334
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x30 += einsum(t1, (0, 1), x29, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x29
    rdm2_f_ooov += einsum(x30, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_oovo += einsum(x30, (0, 1, 2, 3), (2, 1, 3, 0))
    del x30
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (6, 4, 5, 7, 1, 2), (3, 6, 0, 7))
    rdm2_f_ovov += einsum(x31, (0, 1, 2, 3), (1, 2, 0, 3)) * -0.25
    rdm2_f_ovvo += einsum(x31, (0, 1, 2, 3), (1, 2, 3, 0)) * 0.25
    rdm2_f_voov += einsum(x31, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.25
    rdm2_f_vovo += einsum(x31, (0, 1, 2, 3), (2, 1, 3, 0)) * -0.25
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum(t2, (0, 1, 2, 3), x31, (1, 4, 3, 5), (0, 4, 2, 5))
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum(t2, (0, 1, 2, 3), x14, (4, 1, 0, 5, 6, 3), (4, 5, 6, 2))
    rdm2_f_ovov += einsum(x33, (0, 1, 2, 3), (1, 2, 0, 3)) * 0.5
    rdm2_f_ovvo += einsum(x33, (0, 1, 2, 3), (1, 2, 3, 0)) * -0.5
    rdm2_f_voov += einsum(x33, (0, 1, 2, 3), (2, 1, 0, 3)) * -0.5
    rdm2_f_vovo += einsum(x33, (0, 1, 2, 3), (2, 1, 3, 0)) * 0.5
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x34 += einsum(t1, (0, 1), x4, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum(x33, (0, 1, 2, 3), (0, 1, 2, 3))
    x35 += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3))
    del x34
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 += einsum(t2, (0, 1, 2, 3), x35, (1, 4, 3, 5), (4, 0, 5, 2)) * 0.5
    del x35
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x37 += einsum(t2, (0, 1, 2, 3), x2, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x38 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x38 += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3))
    del x37
    x38 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x17
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum(t1, (0, 1), x38, (0, 2, 3, 4), (2, 3, 1, 4))
    del x38
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum(x32, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.25
    del x32
    x40 += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3))
    del x36
    x40 += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3))
    del x39
    x40 += einsum(t1, (0, 1), x28, (2, 3), (0, 2, 1, 3))
    del x28
    rdm2_f_oovv += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x40, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x40, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_oovv += einsum(x40, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x40
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum(x7, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4))
    del x7
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum(x4, (0, 1, 2, 3), t3, (4, 0, 1, 5, 6, 3), (2, 4, 5, 6))
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.08333333333333
    del x41
    x43 += einsum(x42, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x42
    rdm2_f_oovv += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x43, (0, 1, 2, 3), (1, 0, 2, 3))
    del x43
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_ovov += einsum(x44, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovvo += einsum(x44, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_voov += einsum(x44, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_vovo += einsum(x44, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum(t2, (0, 1, 2, 3), x44, (1, 4, 3, 5), (4, 0, 5, 2))
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum(x12, (0, 1, 2, 3), t3, (4, 5, 0, 6, 1, 2), (4, 5, 3, 6))
    x47 = np.zeros((nvir, nvir), dtype=types[float])
    x47 += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4))
    x48 = np.zeros((nvir, nvir), dtype=types[float])
    x48 += einsum(l3, (0, 1, 2, 3, 4, 5), t3, (3, 4, 5, 6, 1, 2), (0, 6))
    x49 = np.zeros((nvir, nvir), dtype=types[float])
    x49 += einsum(x47, (0, 1), (0, 1))
    x49 += einsum(x48, (0, 1), (0, 1)) * 0.16666666666666
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum(x49, (0, 1), t2, (2, 3, 4, 0), (2, 3, 1, 4)) * -0.5
    del x49
    x51 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x51 += einsum(x10, (0, 1, 2, 3), (0, 2, 1, 3))
    del x10
    x51 += einsum(x11, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    del x11
    x51 += einsum(x13, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.25
    del x13
    x51 += einsum(x16, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    del x16
    x52 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x52 += einsum(t1, (0, 1), x51, (0, 2, 3, 4), (2, 3, 1, 4))
    del x51
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x53 += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    del x45
    x53 += einsum(x46, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.25
    del x46
    x53 += einsum(x50, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x50
    x53 += einsum(x52, (0, 1, 2, 3), (1, 0, 2, 3))
    del x52
    rdm2_f_oovv += einsum(x53, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x53, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x53
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum(x9, (0, 1), t2, (2, 0, 3, 4), (1, 2, 3, 4))
    del x9
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum(x2, (0, 1, 2, 3), t3, (4, 1, 0, 5, 6, 3), (2, 4, 5, 6))
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum(x6, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4))
    del x6
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 3, 1)) * 2.0
    x57 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x58 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x58 += einsum(x5, (0, 1, 2, 3), x57, (0, 1, 4, 5), (2, 3, 4, 5)) * 0.25
    del x5, x57
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum(x54, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x54
    x59 += einsum(x55, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    del x55
    x59 += einsum(x56, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    del x56
    x59 += einsum(x58, (0, 1, 2, 3), (0, 1, 3, 2))
    del x58
    rdm2_f_oovv += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x59
    x60 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x60 += einsum(t2, (0, 1, 2, 3), l3, (4, 2, 3, 5, 6, 7), (5, 6, 7, 0, 1, 4))
    x60 += einsum(x15, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.00000000000012
    del x15
    rdm2_f_oovv += einsum(t3, (0, 1, 2, 3, 4, 5), x60, (0, 1, 2, 6, 7, 5), (6, 7, 3, 4)) * 0.08333333333333
    del x60
    x61 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x61 += einsum(x1, (0, 1, 2, 3), (1, 0, 3, 2)) * -3.00000000000012
    x61 += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.00000000000024
    x61 += einsum(x0, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), x61, (0, 1, 4, 5), (5, 4, 2, 3)) * 0.08333333333333
    del x61
    x62 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x62 += einsum(x1, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.99999999999994
    del x1
    x62 += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2)) * 5.99999999999988
    del x3
    x62 += einsum(x0, (0, 1, 2, 3), (0, 1, 3, 2))
    del x0
    x63 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x63 += einsum(t1, (0, 1), x62, (0, 2, 3, 4), (2, 4, 3, 1)) * -0.16666666666667
    del x62
    rdm2_f_oovv += einsum(t1, (0, 1), x63, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    del x63
    x64 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x64 += einsum(x2, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x2
    x64 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3))
    del x4
    x65 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x65 += einsum(t1, (0, 1), x64, (0, 2, 3, 4), (2, 3, 1, 4)) * 0.5
    del x64
    rdm2_f_ovov += einsum(x65, (0, 1, 2, 3), (1, 3, 0, 2)) * -1.0
    rdm2_f_ovvo += einsum(x65, (0, 1, 2, 3), (1, 3, 2, 0))
    rdm2_f_voov += einsum(x65, (0, 1, 2, 3), (3, 1, 0, 2))
    rdm2_f_vovo += einsum(x65, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    del x65
    x66 = np.zeros((nvir, nvir), dtype=types[float])
    x66 += einsum(l1, (0, 1), t1, (1, 2), (0, 2))
    x67 = np.zeros((nvir, nvir), dtype=types[float])
    x67 += einsum(x66, (0, 1), (0, 1)) * 12.00000000000048
    x67 += einsum(x47, (0, 1), (0, 1)) * 6.00000000000024
    x67 += einsum(x48, (0, 1), (0, 1))
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x67, (2, 3), (0, 2, 1, 3)) * 0.08333333333333
    rdm2_f_voov += einsum(delta.oo, (0, 1), x67, (2, 3), (2, 0, 1, 3)) * -0.08333333333333
    del x67
    x68 = np.zeros((nvir, nvir), dtype=types[float])
    x68 += einsum(x66, (0, 1), (0, 1)) * 2.0
    x68 += einsum(x47, (0, 1), (0, 1))
    x68 += einsum(x48, (0, 1), (0, 1)) * 0.16666666666666
    rdm2_f_ovvo += einsum(delta.oo, (0, 1), x68, (2, 3), (0, 2, 3, 1)) * -0.5
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x68, (2, 3), (2, 0, 3, 1)) * 0.5
    del x68
    x69 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x69 += einsum(l2, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 1), (4, 0, 5, 6))
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv += einsum(x69, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv += einsum(x69, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    del x69
    x70 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x70 += einsum(t3, (0, 1, 2, 3, 4, 5), x14, (0, 1, 2, 6, 7, 5), (6, 7, 3, 4)) * -1.0
    del x14
    rdm2_f_ovvv += einsum(x70, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.16666666666667
    rdm2_f_vovv += einsum(x70, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.16666666666667
    del x70
    x71 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x71 += einsum(l1, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_ovvv += einsum(x71, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x71, (0, 1, 2, 3), (1, 0, 3, 2))
    del x71
    x72 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x72 += einsum(t2, (0, 1, 2, 3), x12, (1, 4, 3, 5), (0, 4, 5, 2)) * -1.0
    x73 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3))
    del x44
    x73 += einsum(x31, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x31
    x73 += einsum(x33, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x33
    x74 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x74 += einsum(t1, (0, 1), x73, (0, 2, 3, 4), (2, 1, 3, 4))
    del x73
    x75 = np.zeros((nvir, nvir), dtype=types[float])
    x75 += einsum(x66, (0, 1), (0, 1))
    del x66
    x75 += einsum(x47, (0, 1), (0, 1)) * 0.5
    del x47
    x75 += einsum(x48, (0, 1), (0, 1)) * 0.08333333333333
    del x48
    x76 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x76 += einsum(x72, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x72
    x76 += einsum(x74, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x74
    x76 += einsum(t1, (0, 1), x75, (2, 3), (0, 2, 1, 3))
    del x75
    rdm2_f_ovvv += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_ovvv += einsum(x76, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x76, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vovv += einsum(x76, (0, 1, 2, 3), (1, 0, 3, 2))
    del x76
    x77 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x77 += einsum(t2, (0, 1, 2, 3), x19, (0, 1, 4, 5), (4, 5, 2, 3)) * 0.5
    rdm2_f_ovvv += einsum(x77, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vovv += einsum(x77, (0, 1, 2, 3), (1, 0, 3, 2))
    del x77
    x78 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x78 += einsum(t1, (0, 1), x19, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    rdm2_f_ovvv += einsum(t1, (0, 1), x78, (0, 2, 3, 4), (2, 3, 1, 4))
    del x78
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum(t1, (0, 1), x19, (0, 2, 3, 4), (2, 3, 4, 1)) * -2.0
    del x19
    rdm2_f_vovv += einsum(t1, (0, 1), x79, (0, 2, 3, 4), (3, 2, 4, 1)) * 0.5
    del x79
    x80 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x80 += einsum(t1, (0, 1), l2, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_vvov += einsum(x80, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_vvvo += einsum(x80, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_vvvv += einsum(t1, (0, 1), x80, (0, 2, 3, 4), (2, 3, 1, 4))
    del x80
    x81 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x81 += einsum(t1, (0, 1), x12, (0, 2, 3, 4), (3, 2, 1, 4)) * -1.0
    del x12
    rdm2_f_vvvv += einsum(x81, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    rdm2_f_vvvv += einsum(x81, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    del x81

    rdm2_f = pack_2e(rdm2_f_oooo, rdm2_f_ooov, rdm2_f_oovo, rdm2_f_ovoo, rdm2_f_vooo, rdm2_f_oovv, rdm2_f_ovov, rdm2_f_ovvo, rdm2_f_voov, rdm2_f_vovo, rdm2_f_vvoo, rdm2_f_ovvv, rdm2_f_vovv, rdm2_f_vvov, rdm2_f_vvvo, rdm2_f_vvvv)

    rdm2_f = rdm2_f.swapaxes(1, 2)

    return rdm2_f

