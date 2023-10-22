# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((naux,), dtype=types[float])
    x0 += einsum(t1, (0, 1), v.xov, (2, 0, 1), (2,))
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x2 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x2 += einsum(x0, (0,), t1, (1, 2), (1, 2, 0))
    del x0
    x2 += einsum(v.xov, (0, 1, 2), x1, (1, 3, 4, 2), (3, 4, 0))
    del x1
    e_cc = 0
    e_cc += einsum(v.xov, (0, 1, 2), x2, (1, 2, 0), ()) * 2.0
    del x2
    x3 = np.zeros((nocc, nocc, naux), dtype=types[float])
    x3 += einsum(t1, (0, 1), v.xov, (2, 3, 1), (0, 3, 2))
    x4 = np.zeros((nocc, nvir), dtype=types[float])
    x4 += einsum(f.ov, (0, 1), (0, 1))
    x4 += einsum(v.xov, (0, 1, 2), x3, (1, 3, 0), (3, 2)) * -0.5
    del x3
    e_cc += einsum(t1, (0, 1), x4, (0, 1), ()) * 2.0
    del x4

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    x0 = np.zeros((naux,), dtype=types[float])
    x0 += einsum(t1, (0, 1), v.xov, (2, 0, 1), (2,))
    x1 = np.zeros((nocc, nvir), dtype=types[float])
    x1 += einsum(x0, (0,), v.xov, (0, 1, 2), (1, 2))
    t1new += einsum(x1, (0, 1), (0, 1)) * 2.0
    x2 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x2 += einsum(t1, (0, 1), v.xoo, (2, 3, 0), (3, 1, 2))
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    x3 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    x3 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x4 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x4 += einsum(x2, (0, 1, 2), (0, 1, 2))
    x4 += einsum(x0, (0,), t1, (1, 2), (1, 2, 0)) * -2.0
    x4 += einsum(v.xov, (0, 1, 2), x3, (1, 3, 4, 2), (3, 4, 0))
    del x3
    t1new += einsum(v.xvv, (0, 1, 2), x4, (3, 2, 0), (3, 1)) * -1.0
    del x4
    x5 = np.zeros((nocc, nocc, naux), dtype=types[float])
    x5 += einsum(t1, (0, 1), v.xov, (2, 3, 1), (0, 3, 2))
    x6 = np.zeros((nocc, nocc, naux), dtype=types[float])
    x6 += einsum(v.xoo, (0, 1, 2), (1, 2, 0))
    x6 += einsum(x5, (0, 1, 2), (1, 0, 2))
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x7 += einsum(v.xov, (0, 1, 2), x6, (3, 4, 0), (4, 1, 3, 2))
    del x6
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x8 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t1new += einsum(x7, (0, 1, 2, 3), x8, (1, 2, 3, 4), (0, 4)) * -1.0
    del x7
    x9 = np.zeros((nocc, nvir), dtype=types[float])
    x9 += einsum(v.xov, (0, 1, 2), x5, (1, 3, 0), (3, 2))
    x10 = np.zeros((nocc, nvir), dtype=types[float])
    x10 += einsum(f.ov, (0, 1), (0, 1))
    x10 += einsum(x1, (0, 1), (0, 1)) * 2.0
    x10 += einsum(x9, (0, 1), (0, 1)) * -1.0
    t1new += einsum(x10, (0, 1), x8, (0, 2, 1, 3), (2, 3))
    x11 = np.zeros((nocc, nocc), dtype=types[float])
    x11 += einsum(x0, (0,), v.xoo, (0, 1, 2), (1, 2))
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x12 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x13 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x13 += einsum(v.xov, (0, 1, 2), x12, (1, 3, 4, 2), (3, 4, 0)) * 2.0
    x14 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x14 += einsum(x2, (0, 1, 2), (0, 1, 2)) * -1.0
    x14 += einsum(x0, (0,), t1, (1, 2), (1, 2, 0)) * 2.0
    x14 += einsum(x13, (0, 1, 2), (0, 1, 2))
    x15 = np.zeros((nocc, nvir), dtype=types[float])
    x15 += einsum(f.ov, (0, 1), (0, 1))
    x15 += einsum(x9, (0, 1), (0, 1)) * -1.0
    x16 = np.zeros((nocc, nocc), dtype=types[float])
    x16 += einsum(f.oo, (0, 1), (0, 1))
    x16 += einsum(x11, (0, 1), (1, 0)) * 2.0
    x16 += einsum(v.xov, (0, 1, 2), x14, (3, 2, 0), (1, 3))
    del x14
    x16 += einsum(t1, (0, 1), x15, (2, 1), (2, 0))
    del x15
    t1new += einsum(t1, (0, 1), x16, (0, 2), (2, 1)) * -1.0
    del x16
    x17 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x17 += einsum(v.xov, (0, 1, 2), t2, (3, 1, 2, 4), (3, 4, 0))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(x17, (0, 1, 2), x17, (3, 4, 2), (0, 3, 1, 4))
    x18 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x18 += einsum(v.xov, (0, 1, 2), t2, (3, 1, 4, 2), (3, 4, 0))
    t2new += einsum(x18, (0, 1, 2), x18, (3, 4, 2), (0, 3, 1, 4)) * 4.0
    del x18
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(v.xov, (0, 1, 2), v.xov, (0, 3, 4), (1, 3, 2, 4))
    t2new += einsum(x19, (0, 1, 2, 3), (1, 0, 3, 2))
    x20 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x20 += einsum(t2, (0, 1, 2, 3), x19, (4, 5, 3, 2), (0, 1, 5, 4))
    del x19
    x21 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x21 += einsum(t1, (0, 1), x20, (2, 3, 4, 0), (2, 3, 4, 1))
    del x20
    t2new += einsum(t1, (0, 1), x21, (2, 3, 0, 4), (2, 3, 1, 4))
    del x21
    x22 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x22 += einsum(v.xvv, (0, 1, 2), v.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new += einsum(t2, (0, 1, 2, 3), x22, (4, 2, 5, 3), (0, 1, 5, 4))
    del x22
    x23 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x23 += einsum(t1, (0, 1), v.xvv, (2, 3, 1), (0, 3, 2))
    t2new += einsum(x23, (0, 1, 2), x23, (3, 4, 2), (0, 3, 1, 4))
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum(v.xvv, (0, 1, 2), x5, (3, 4, 0), (3, 4, 1, 2))
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(t2, (0, 1, 2, 3), x24, (4, 1, 5, 2), (4, 0, 3, 5))
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(v.xoo, (0, 1, 2), v.xvv, (0, 3, 4), (1, 2, 3, 4))
    x27 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x27 += einsum(v.xov, (0, 1, 2), (1, 2, 0))
    x27 += einsum(x17, (0, 1, 2), (0, 1, 2)) * -1.0
    del x17
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum(v.xov, (0, 1, 2), x27, (3, 4, 0), (3, 1, 4, 2)) * 2.0
    del x27
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum(x26, (0, 1, 2, 3), (1, 0, 3, 2))
    x29 += einsum(x28, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x28
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum(t2, (0, 1, 2, 3), x29, (4, 1, 5, 3), (4, 0, 5, 2))
    del x29
    x31 = np.zeros((nvir, nvir), dtype=types[float])
    x31 += einsum(x0, (0,), v.xvv, (0, 1, 2), (1, 2))
    del x0
    x32 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x32 += einsum(v.xov, (0, 1, 2), x12, (1, 3, 4, 2), (3, 4, 0))
    del x12
    x33 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x33 += einsum(x23, (0, 1, 2), (0, 1, 2))
    x33 += einsum(x32, (0, 1, 2), (0, 1, 2))
    del x32
    x34 = np.zeros((nvir, nvir), dtype=types[float])
    x34 += einsum(v.xov, (0, 1, 2), x33, (1, 3, 0), (3, 2))
    del x33
    x35 = np.zeros((nvir, nvir), dtype=types[float])
    x35 += einsum(x31, (0, 1), (1, 0)) * -2.0
    del x31
    x35 += einsum(x34, (0, 1), (1, 0))
    del x34
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 += einsum(x35, (0, 1), t2, (2, 3, 0, 4), (2, 3, 1, 4))
    del x35
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x37 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    x38 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x38 += einsum(v.xov, (0, 1, 2), x37, (1, 3, 4, 2), (3, 4, 0))
    x39 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x39 += einsum(x2, (0, 1, 2), (0, 1, 2))
    x39 += einsum(x38, (0, 1, 2), (0, 1, 2))
    del x38
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum(v.xov, (0, 1, 2), x39, (3, 4, 0), (3, 1, 4, 2))
    del x39
    x41 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x41 += einsum(v.xoo, (0, 1, 2), v.xov, (0, 3, 4), (1, 2, 3, 4))
    x42 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x42 += einsum(t2, (0, 1, 2, 3), x41, (4, 1, 5, 2), (0, 4, 5, 3))
    x43 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x43 += einsum(v.xov, (0, 1, 2), v.xvv, (0, 3, 4), (1, 2, 3, 4))
    x44 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x44 += einsum(t2, (0, 1, 2, 3), x43, (4, 2, 5, 3), (0, 1, 4, 5))
    del x43
    x45 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x45 += einsum(x23, (0, 1, 2), x5, (3, 4, 2), (3, 0, 4, 1))
    x46 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x46 += einsum(v.xov, (0, 1, 2), x5, (3, 4, 0), (3, 1, 4, 2))
    x47 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x47 += einsum(t2, (0, 1, 2, 3), x46, (4, 1, 5, 2), (4, 0, 5, 3))
    x48 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x48 += einsum(x46, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x48 += einsum(x46, (0, 1, 2, 3), (0, 2, 1, 3))
    x49 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x49 += einsum(t2, (0, 1, 2, 3), x48, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    del x48
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x50 += einsum(x42, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x42
    x50 += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3))
    del x44
    x50 += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    del x45
    x50 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x47
    x50 += einsum(x49, (0, 1, 2, 3), (0, 2, 1, 3))
    del x49
    x51 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x51 += einsum(t1, (0, 1), x50, (2, 3, 0, 4), (2, 3, 4, 1))
    del x50
    x52 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x52 += einsum(v.xov, (0, 1, 2), x8, (1, 3, 2, 4), (3, 4, 0)) * 0.5
    del x8
    x53 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x53 += einsum(x2, (0, 1, 2), (0, 1, 2))
    x53 += einsum(x52, (0, 1, 2), (0, 1, 2)) * -1.0
    del x52
    x54 = np.zeros((nocc, nocc), dtype=types[float])
    x54 += einsum(v.xov, (0, 1, 2), x53, (3, 2, 0), (3, 1))
    del x53
    x55 = np.zeros((nocc, nocc), dtype=types[float])
    x55 += einsum(x11, (0, 1), (1, 0)) * 2.0
    del x11
    x55 += einsum(x54, (0, 1), (1, 0)) * -1.0
    del x54
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum(x55, (0, 1), t2, (2, 0, 3, 4), (1, 2, 4, 3))
    del x55
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3))
    del x25
    x57 += einsum(x30, (0, 1, 2, 3), (1, 0, 3, 2))
    del x30
    x57 += einsum(x36, (0, 1, 2, 3), (1, 0, 3, 2))
    del x36
    x57 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3))
    del x40
    x57 += einsum(x51, (0, 1, 2, 3), (0, 1, 3, 2))
    del x51
    x57 += einsum(x56, (0, 1, 2, 3), (1, 0, 3, 2))
    del x56
    t2new += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x57, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x57
    x58 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x58 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum(t2, (0, 1, 2, 3), x26, (4, 1, 5, 2), (0, 4, 3, 5))
    del x26
    x60 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x60 += einsum(v.xoo, (0, 1, 2), x5, (3, 4, 0), (3, 1, 2, 4))
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum(t2, (0, 1, 2, 3), x60, (4, 1, 5, 0), (4, 5, 3, 2))
    x62 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x62 += einsum(t2, (0, 1, 2, 3), x24, (4, 1, 5, 3), (4, 0, 2, 5))
    del x24
    x63 = np.zeros((nocc, nvir, naux), dtype=types[float])
    x63 += einsum(v.xov, (0, 1, 2), (1, 2, 0))
    x63 += einsum(x2, (0, 1, 2), (0, 1, 2)) * -1.0
    del x2
    x63 += einsum(x13, (0, 1, 2), (0, 1, 2))
    del x13
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum(x23, (0, 1, 2), x63, (3, 4, 2), (0, 3, 1, 4))
    del x23, x63
    x65 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x65 += einsum(t2, (0, 1, 2, 3), x41, (4, 5, 1, 2), (0, 5, 4, 3))
    x66 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x66 += einsum(t1, (0, 1), x60, (2, 3, 4, 0), (2, 3, 4, 1))
    del x60
    x67 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x67 += einsum(t2, (0, 1, 2, 3), x46, (4, 5, 1, 2), (4, 0, 5, 3))
    del x46
    x68 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x68 += einsum(x41, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x68 += einsum(x41, (0, 1, 2, 3), (2, 1, 0, 3)) * 2.0
    del x41
    x69 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x69 += einsum(t2, (0, 1, 2, 3), x68, (1, 4, 5, 3), (4, 5, 0, 2))
    del x68
    x70 = np.zeros((nocc, nvir), dtype=types[float])
    x70 += einsum(f.ov, (0, 1), (0, 1)) * 0.5
    x70 += einsum(x1, (0, 1), (0, 1))
    del x1
    x70 += einsum(x9, (0, 1), (0, 1)) * -0.5
    del x9
    x71 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x71 += einsum(x70, (0, 1), t2, (2, 3, 1, 4), (0, 2, 3, 4)) * 2.0
    del x70
    x72 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x72 += einsum(x65, (0, 1, 2, 3), (1, 0, 2, 3))
    del x65
    x72 += einsum(x66, (0, 1, 2, 3), (1, 0, 2, 3))
    del x66
    x72 += einsum(x67, (0, 1, 2, 3), (2, 0, 1, 3))
    del x67
    x72 += einsum(x69, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x69
    x72 += einsum(x71, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x71
    x73 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum(t1, (0, 1), x72, (0, 2, 3, 4), (2, 3, 4, 1))
    del x72
    x74 = np.zeros((nocc, nocc), dtype=types[float])
    x74 += einsum(t1, (0, 1), x10, (2, 1), (2, 0)) * 0.5
    del x10
    x75 = np.zeros((nocc, nocc), dtype=types[float])
    x75 += einsum(f.oo, (0, 1), (0, 1)) * 0.5
    x75 += einsum(x74, (0, 1), (1, 0))
    del x74
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum(x75, (0, 1), t2, (2, 1, 3, 4), (0, 2, 4, 3)) * 2.0
    del x75
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum(x58, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x58
    x77 += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3))
    del x59
    x77 += einsum(x61, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x61
    x77 += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3))
    del x62
    x77 += einsum(x64, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x64
    x77 += einsum(x73, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x73
    x77 += einsum(x76, (0, 1, 2, 3), (0, 1, 3, 2))
    del x76
    t2new += einsum(x77, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x77, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x77
    x78 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x78 += einsum(v.xoo, (0, 1, 2), v.xoo, (0, 3, 4), (1, 3, 4, 2))
    x78 += einsum(x5, (0, 1, 2), x5, (3, 4, 2), (4, 0, 1, 3))
    del x5
    t2new += einsum(x37, (0, 1, 2, 3), x78, (0, 4, 1, 5), (4, 5, 3, 2))
    del x37, x78

    return {"t1new": t1new, "t2new": t2new}

