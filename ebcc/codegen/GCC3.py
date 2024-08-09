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
    t1new += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 0, 1, 5, 2, 3), (4, 5)) * 0.25
    t1new += einsum(t1, (0, 1), v.ovov, (2, 1, 0, 3), (2, 3)) * -1.0
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t1new += einsum(f.ov, (0, 1), (0, 1))
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
    t1new += einsum(v.ovvv, (0, 1, 2, 3), x2, (0, 4, 3, 2), (4, 1)) * -0.5
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
    x7 += einsum(f.oo, (0, 1), (0, 1)) * 2.0
    x7 += einsum(x5, (0, 1), (0, 1)) * 2.0
    x7 += einsum(x6, (0, 1), (0, 1)) * 2.0
    x7 += einsum(v.oovv, (0, 1, 2, 3), x2, (1, 4, 2, 3), (0, 4)) * -1.0
    t1new += einsum(t1, (0, 1), x7, (0, 2), (2, 1)) * -0.5
    del x7
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 5, 0, 6, 2, 3), (4, 5, 6, 1))
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(t2, (0, 1, 2, 3), x9, (4, 1, 5, 3), (4, 0, 5, 2))
    del x9
    x11 = np.zeros((nvir, nvir), dtype=types[float])
    x11 += einsum(t1, (0, 1), v.ovvv, (0, 2, 3, 1), (2, 3))
    x12 = np.zeros((nvir, nvir), dtype=types[float])
    x12 += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    x13 = np.zeros((nvir, nvir), dtype=types[float])
    x13 += einsum(x11, (0, 1), (0, 1))
    del x11
    x13 += einsum(x12, (0, 1), (0, 1)) * 0.5
    del x12
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(x13, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    del x13
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 5, 1, 6, 2, 3), (4, 5, 0, 6))
    x16 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x16 += einsum(v.ovvv, (0, 1, 2, 3), x2, (4, 5, 2, 3), (0, 4, 5, 1)) * 0.5
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum(x4, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    x18 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x18 += einsum(x15, (0, 1, 2, 3), (2, 1, 0, 3)) * -0.5
    del x15
    x18 += einsum(x16, (0, 1, 2, 3), (0, 2, 1, 3))
    del x16
    x18 += einsum(x17, (0, 1, 2, 3), (2, 1, 0, 3))
    del x17
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(t1, (0, 1), x18, (0, 2, 3, 4), (2, 3, 4, 1))
    del x18
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x8
    x20 += einsum(x10, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x10
    x20 += einsum(x14, (0, 1, 2, 3), (1, 0, 3, 2))
    del x14
    x20 += einsum(x19, (0, 1, 2, 3), (1, 0, 3, 2))
    del x19
    t2new += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x20, (0, 1, 2, 3), (0, 1, 3, 2))
    del x20
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(t2, (0, 1, 2, 3), x22, (4, 1, 5, 3), (4, 0, 2, 5)) * -1.0
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x24 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x25 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x25 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x26 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x26 += einsum(t2, (0, 1, 2, 3), x0, (4, 1, 5, 3), (4, 0, 5, 2))
    x27 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x27 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    x27 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3))
    del x25
    x27 += einsum(x26, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x26
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum(t1, (0, 1), x27, (2, 0, 3, 4), (2, 3, 4, 1))
    del x27
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x21
    x29 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    del x23
    x29 += einsum(x28, (0, 1, 2, 3), (0, 1, 3, 2))
    del x28
    t2new += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x29, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x29, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new += einsum(x29, (0, 1, 2, 3), (1, 0, 3, 2))
    del x29
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum(t1, (0, 1), v.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum(x30, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x30
    x32 += einsum(x31, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x31
    t2new += einsum(x32, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x32, (0, 1, 2, 3), (0, 1, 3, 2))
    del x32
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x34 += einsum(v.ooov, (0, 1, 2, 3), t3, (4, 0, 1, 5, 6, 3), (4, 2, 5, 6))
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum(x0, (0, 1, 2, 3), t3, (4, 2, 1, 5, 6, 3), (0, 4, 5, 6))
    x36 = np.zeros((nocc, nocc), dtype=types[float])
    x36 += einsum(t1, (0, 1), x4, (2, 1), (0, 2))
    del x4
    x37 = np.zeros((nocc, nocc), dtype=types[float])
    x37 += einsum(f.oo, (0, 1), (0, 1))
    x37 += einsum(x36, (0, 1), (0, 1))
    del x36
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum(x37, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4)) * -1.0
    del x37
    x39 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x39 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum(x2, (0, 1, 2, 3), x39, (4, 0, 1, 5), (4, 5, 2, 3)) * 0.5
    x41 = np.zeros((nocc, nocc), dtype=types[float])
    x41 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    x42 = np.zeros((nocc, nocc), dtype=types[float])
    x42 += einsum(x6, (0, 1), (0, 1))
    del x6
    x42 += einsum(x41, (0, 1), (1, 0)) * 0.5
    del x41
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum(x42, (0, 1), t2, (2, 0, 3, 4), (1, 2, 3, 4)) * -1.0
    del x42
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum(x33, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x33
    x44 += einsum(x34, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    del x34
    x44 += einsum(x35, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    del x35
    x44 += einsum(x38, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x38
    x44 += einsum(x40, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x40
    x44 += einsum(x43, (0, 1, 2, 3), (1, 0, 3, 2))
    del x43
    t2new += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x44, (0, 1, 2, 3), (1, 0, 2, 3))
    del x44
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x45 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    t2new += einsum(v.vvvv, (0, 1, 2, 3), x45, (4, 5, 2, 3), (5, 4, 0, 1)) * -1.0
    del x45
    x46 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x46 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x46 += einsum(v.oovv, (0, 1, 2, 3), x2, (4, 5, 2, 3), (0, 1, 5, 4)) * -0.5
    t2new += einsum(x2, (0, 1, 2, 3), x46, (0, 1, 4, 5), (4, 5, 3, 2)) * -0.5
    del x2, x46
    x47 = np.zeros((nvir, nvir), dtype=types[float])
    x47 += einsum(f.ov, (0, 1), t1, (0, 2), (1, 2))
    x48 = np.zeros((nvir, nvir), dtype=types[float])
    x48 += einsum(f.vv, (0, 1), (0, 1))
    x48 += einsum(x47, (0, 1), (0, 1)) * -1.0
    del x47
    x49 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x49 += einsum(x48, (0, 1), t3, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x48
    t3new = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    t3new += einsum(x49, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x49, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x49, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x49
    x50 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x50 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 3, 5, 6), (0, 1, 4, 2, 5, 6))
    x51 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x51 += einsum(f.ov, (0, 1), t2, (2, 3, 4, 1), (0, 2, 3, 4))
    x52 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x52 += einsum(t1, (0, 1), x0, (2, 3, 4, 1), (2, 0, 4, 3))
    x53 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x53 += einsum(t1, (0, 1), x52, (2, 3, 4, 0), (2, 3, 4, 1))
    del x52
    x54 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x54 += einsum(x51, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x51
    x54 += einsum(x53, (0, 1, 2, 3), (0, 1, 2, 3))
    del x53
    x55 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x55 += einsum(t2, (0, 1, 2, 3), x54, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3)) * -1.0
    del x54
    x56 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x56 += einsum(x50, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x50
    x56 += einsum(x55, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x55
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x56
    x57 = np.zeros((nocc, nocc), dtype=types[float])
    x57 += einsum(f.oo, (0, 1), (0, 1))
    x57 += einsum(x5, (0, 1), (0, 1))
    del x5
    x58 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x58 += einsum(x57, (0, 1), t3, (2, 3, 0, 4, 5, 6), (1, 2, 3, 4, 5, 6))
    del x57
    t3new += einsum(x58, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x58, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x58, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    del x58
    x59 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x59 += einsum(t2, (0, 1, 2, 3), x24, (4, 1, 5, 6), (4, 0, 5, 2, 3, 6))
    del x24
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x59
    x60 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 6, 3), (0, 1, 4, 6, 2, 5))
    x61 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x61 += einsum(t1, (0, 1), x60, (2, 3, 0, 4, 5, 6), (2, 3, 4, 1, 5, 6))
    del x60
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x61
    x62 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x62 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 1, 6), (0, 4, 5, 2, 3, 6))
    x63 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x63 += einsum(t2, (0, 1, 2, 3), x0, (4, 5, 6, 3), (4, 0, 1, 6, 5, 2))
    del x0
    x64 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum(t1, (0, 1), x63, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x63
    x65 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x65 += einsum(t1, (0, 1), x64, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6)) * -1.0
    del x64
    x66 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x66 += einsum(x62, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    del x62
    x66 += einsum(x65, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    del x65
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x66
    x67 = np.zeros((nocc, nocc, nocc, nocc, nocc, nvir), dtype=types[float])
    x67 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    x68 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x68 += einsum(t1, (0, 1), x67, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x67
    x69 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x69 += einsum(t1, (0, 1), x68, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 1, 6)) * -1.0
    del x68
    x70 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x70 += einsum(t1, (0, 1), x22, (2, 3, 4, 1), (2, 0, 3, 4)) * -1.0
    x71 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x71 += einsum(t2, (0, 1, 2, 3), x70, (4, 5, 1, 6), (5, 4, 0, 2, 3, 6)) * -1.0
    del x70
    x72 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x72 += einsum(x69, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x69
    x72 += einsum(x71, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x71
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x72
    x73 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum(t2, (0, 1, 2, 3), x22, (4, 5, 6, 3), (4, 0, 1, 5, 2, 6)) * -1.0
    del x22
    x74 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x74 += einsum(t1, (0, 1), x73, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x73
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    del x74
    x75 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x75 += einsum(t1, (0, 1), v.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    x76 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x76 += einsum(t2, (0, 1, 2, 3), x75, (4, 5, 1, 6), (0, 4, 5, 6, 2, 3)) * -1.0
    del x75
    x77 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x77 += einsum(t1, (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x78 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x78 += einsum(t2, (0, 1, 2, 3), x77, (4, 5, 6, 3), (4, 0, 1, 2, 5, 6)) * -1.0
    del x77
    x79 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x79 += einsum(x76, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    del x76
    x79 += einsum(x78, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    del x78
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x79
    x80 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x80 += einsum(t1, (0, 1), x39, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x39
    x81 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x81 += einsum(t2, (0, 1, 2, 3), x80, (4, 1, 5, 6), (4, 0, 5, 6, 2, 3)) * -1.0
    del x80
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x81

    return {"t1new": t1new, "t2new": t2new, "t3new": t3new}

