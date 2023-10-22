# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x0 += einsum(v.aaa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    e_cc = 0
    e_cc += einsum(v.aaa.xov, (0, 1, 2), x0, (1, 2, 0), ())
    del x0
    x1 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x1 += einsum(v.aaa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    x1 += einsum(v.abb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    e_cc += einsum(v.abb.xov, (0, 1, 2), x1, (1, 2, 0), ())
    del x1

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, **kwargs):
    t1new = Namespace()
    t2new = Namespace()

    # T amplitudes
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    t1new_aa += einsum(f.aa.vv, (0, 1), t1.aa, (2, 1), (2, 0))
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    t1new_bb += einsum(f.bb.vv, (0, 1), t1.bb, (2, 1), (2, 0))
    x0 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x0 += einsum(t1.aa, (0, 1), v.aaa.xoo, (2, 3, 0), (3, 1, 2))
    x1 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x1 += einsum(v.aaa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    x2 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x2 += einsum(v.abb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (3, 4, 0))
    x3 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x3 += einsum(x0, (0, 1, 2), (0, 1, 2)) * -0.5
    x3 += einsum(x1, (0, 1, 2), (0, 1, 2))
    x3 += einsum(x2, (0, 1, 2), (0, 1, 2)) * 0.5
    t1new_aa += einsum(v.aaa.xvv, (0, 1, 2), x3, (3, 2, 0), (3, 1)) * 2.0
    del x3
    x4 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x4 += einsum(t1.aa, (0, 1), v.aaa.xov, (2, 3, 1), (0, 3, 2))
    x5 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x5 += einsum(v.aaa.xoo, (0, 1, 2), (1, 2, 0))
    x5 += einsum(x4, (0, 1, 2), (0, 1, 2))
    x6 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x6 += einsum(v.abb.xov, (0, 1, 2), x5, (3, 4, 0), (4, 3, 1, 2))
    del x5
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x6, (0, 4, 1, 3), (4, 2)) * -1.0
    del x6
    x7 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x7 += einsum(v.aaa.xoo, (0, 1, 2), (1, 2, 0))
    x7 += einsum(x4, (0, 1, 2), (1, 0, 2))
    x8 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum(v.aaa.xov, (0, 1, 2), x7, (3, 4, 0), (4, 1, 3, 2))
    del x7
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), x8, (4, 0, 1, 3), (4, 2)) * 2.0
    del x8
    x9 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x9 += einsum(t1.bb, (0, 1), v.abb.xov, (2, 3, 1), (0, 3, 2))
    x10 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x10 += einsum(v.abb.xov, (0, 1, 2), x9, (1, 3, 0), (3, 2))
    x11 = np.zeros((naux,), dtype=types[float])
    x11 += einsum(t1.aa, (0, 1), v.aaa.xov, (2, 0, 1), (2,))
    x12 = np.zeros((naux,), dtype=types[float])
    x12 += einsum(t1.bb, (0, 1), v.abb.xov, (2, 0, 1), (2,))
    x13 = np.zeros((naux,), dtype=types[float])
    x13 += einsum(x11, (0,), (0,))
    del x11
    x13 += einsum(x12, (0,), (0,))
    del x12
    x14 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x14 += einsum(x13, (0,), v.abb.xov, (0, 1, 2), (1, 2))
    t1new_bb += einsum(x14, (0, 1), (0, 1))
    x15 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x15 += einsum(f.bb.ov, (0, 1), (0, 1))
    x15 += einsum(x10, (0, 1), (0, 1)) * -1.0
    del x10
    x15 += einsum(x14, (0, 1), (0, 1))
    del x14
    t1new_aa += einsum(x15, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    t1new_bb += einsum(x15, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    del x15
    x16 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x16 += einsum(v.aaa.xov, (0, 1, 2), x4, (1, 3, 0), (3, 2))
    del x4
    x17 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x17 += einsum(x13, (0,), v.aaa.xov, (0, 1, 2), (1, 2))
    del x13
    t1new_aa += einsum(x17, (0, 1), (0, 1))
    x18 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x18 += einsum(f.aa.ov, (0, 1), (0, 1))
    x18 += einsum(x16, (0, 1), (0, 1)) * -1.0
    del x16
    x18 += einsum(x17, (0, 1), (0, 1))
    del x17
    t1new_aa += einsum(x18, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_bb += einsum(x18, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    del x18
    x19 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x19 += einsum(x1, (0, 1, 2), (0, 1, 2))
    x19 += einsum(x2, (0, 1, 2), (0, 1, 2)) * 0.5
    x20 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x20 += einsum(f.aa.oo, (0, 1), (0, 1))
    x20 += einsum(v.aaa.xov, (0, 1, 2), x19, (3, 2, 0), (1, 3)) * 2.0
    t1new_aa += einsum(t1.aa, (0, 1), x20, (0, 2), (2, 1)) * -1.0
    del x20
    x21 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x21 += einsum(t1.bb, (0, 1), v.abb.xoo, (2, 3, 0), (3, 1, 2))
    x22 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x22 += einsum(v.aaa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    x23 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x23 += einsum(v.abb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum(x1, (0, 1, 2), x23, (3, 4, 2), (0, 3, 1, 4)) * 4.0
    x24 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x24 += einsum(x21, (0, 1, 2), (0, 1, 2)) * -0.5
    x24 += einsum(x22, (0, 1, 2), (0, 1, 2)) * 0.5
    x24 += einsum(x23, (0, 1, 2), (0, 1, 2))
    t1new_bb += einsum(v.abb.xvv, (0, 1, 2), x24, (3, 2, 0), (3, 1)) * 2.0
    x25 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x25 += einsum(v.abb.xoo, (0, 1, 2), (1, 2, 0))
    x25 += einsum(x9, (0, 1, 2), (0, 1, 2))
    x26 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x26 += einsum(v.aaa.xov, (0, 1, 2), x25, (3, 4, 0), (1, 4, 3, 2))
    del x25
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x26, (0, 1, 4, 2), (4, 3)) * -1.0
    del x26
    x27 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x27 += einsum(v.abb.xoo, (0, 1, 2), (1, 2, 0))
    x27 += einsum(x9, (0, 1, 2), (1, 0, 2))
    del x9
    x28 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x28 += einsum(v.abb.xov, (0, 1, 2), x27, (3, 4, 0), (4, 1, 3, 2))
    del x27
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), x28, (4, 0, 1, 3), (4, 2)) * 2.0
    del x28
    x29 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x29 += einsum(x22, (0, 1, 2), (0, 1, 2))
    x29 += einsum(x23, (0, 1, 2), (0, 1, 2)) * 2.0
    x30 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x30 += einsum(v.abb.xov, (0, 1, 2), x29, (3, 2, 0), (1, 3))
    x31 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x31 += einsum(f.bb.oo, (0, 1), (0, 1))
    x31 += einsum(x30, (0, 1), (0, 1))
    t1new_bb += einsum(t1.bb, (0, 1), x31, (0, 2), (2, 1)) * -1.0
    t2new_abab += einsum(x31, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x31
    x32 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x32 += einsum(v.aaa.xvv, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x32, (4, 2, 5, 3), (0, 1, 4, 5)) * -2.0
    del x32
    x33 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x33 += einsum(v.aaa.xov, (0, 1, 2), v.aaa.xov, (0, 3, 4), (1, 3, 2, 4))
    x34 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x34 += einsum(x33, (0, 1, 2, 3), (1, 0, 2, 3))
    x34 += einsum(x33, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x35 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x35 += einsum(t2.aaaa, (0, 1, 2, 3), x34, (1, 4, 3, 5), (0, 4, 2, 5))
    x36 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x36 += einsum(t2.aaaa, (0, 1, 2, 3), x35, (4, 1, 5, 3), (0, 4, 2, 5)) * -4.0
    del x35
    x37 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x37 += einsum(v.abb.xov, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 3, 2, 4))
    x38 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x38 += einsum(x37, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x38 += einsum(x37, (0, 1, 2, 3), (1, 0, 3, 2))
    x39 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x39 += einsum(t2.abab, (0, 1, 2, 3), x38, (1, 4, 5, 3), (0, 4, 2, 5))
    del x38
    x40 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x40 += einsum(t2.abab, (0, 1, 2, 3), x39, (4, 1, 5, 3), (0, 4, 2, 5)) * -1.0
    del x39
    x41 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x41 += einsum(v.aaa.xov, (0, 1, 2), x19, (1, 3, 0), (2, 3))
    x42 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x42 += einsum(x41, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -4.0
    x43 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x43 += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x36
    x43 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x40
    x43 += einsum(x42, (0, 1, 2, 3), (1, 0, 2, 3))
    del x42
    t2new_aaaa += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x43, (0, 1, 2, 3), (0, 1, 3, 2))
    del x43
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x44 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x45 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x45 += einsum(x44, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x44
    x45 += einsum(x33, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x45, (0, 1, 2, 3), (0, 1, 3, 2))
    del x45
    x46 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x46 += einsum(f.aa.oo, (0, 1), t2.aaaa, (2, 1, 3, 4), (0, 2, 3, 4))
    x47 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x47 += einsum(v.aaa.xov, (0, 1, 2), x19, (3, 2, 0), (1, 3))
    del x19
    x48 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x48 += einsum(x47, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -4.0
    x49 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x49 += einsum(x46, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x46
    x49 += einsum(x48, (0, 1, 2, 3), (0, 1, 3, 2))
    del x48
    t2new_aaaa += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x49, (0, 1, 2, 3), (1, 0, 2, 3))
    del x49
    x50 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x50 += einsum(t1.aa, (0, 1), v.aaa.xvv, (2, 3, 1), (0, 3, 2))
    x51 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x51 += einsum(v.aaa.xov, (0, 1, 2), x50, (3, 4, 0), (3, 1, 2, 4))
    x52 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x52 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 2, 3, 4))
    x53 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x53 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x53 += einsum(x2, (0, 1, 2), (0, 1, 2))
    x54 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x54 += einsum(v.aaa.xov, (0, 1, 2), x53, (3, 4, 0), (1, 3, 2, 4))
    del x53
    x55 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x55 += einsum(x52, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x55 += einsum(x54, (0, 1, 2, 3), (0, 1, 2, 3))
    del x54
    x56 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x56 += einsum(t2.aaaa, (0, 1, 2, 3), x55, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x55
    x57 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x57 += einsum(x0, (0, 1, 2), (0, 1, 2))
    x57 += einsum(x2, (0, 1, 2), (0, 1, 2)) * -1.0
    del x2
    x58 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x58 += einsum(v.aaa.xov, (0, 1, 2), x57, (3, 4, 0), (1, 3, 2, 4))
    del x57
    x59 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x59 += einsum(x51, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x51
    x59 += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3))
    del x56
    x59 += einsum(x58, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x58
    t2new_aaaa += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x59, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_aaaa += einsum(x59, (0, 1, 2, 3), (1, 0, 3, 2))
    del x59
    x60 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x60 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xoo, (0, 3, 4), (1, 3, 4, 2))
    x60 += einsum(t2.aaaa, (0, 1, 2, 3), x33, (4, 5, 2, 3), (5, 0, 4, 1))
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x60, (0, 4, 1, 5), (4, 5, 2, 3)) * -2.0
    del x60
    x61 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x61 += einsum(v.aaa.xvv, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x61, (4, 2, 5, 3), (0, 1, 4, 5))
    del x61
    x62 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x62 += einsum(v.abb.xoo, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 2, 3, 4))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x62, (4, 1, 5, 2), (0, 4, 5, 3)) * -1.0
    del x62
    x63 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x63 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    x64 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x64 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0))
    x64 += einsum(x22, (0, 1, 2), (0, 1, 2))
    x64 += einsum(x23, (0, 1, 2), (0, 1, 2)) * 2.0
    x65 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x65 += einsum(x63, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    x65 += einsum(t2.bbbb, (0, 1, 2, 3), x37, (1, 4, 5, 3), (4, 0, 5, 2)) * -1.0
    x65 += einsum(v.abb.xov, (0, 1, 2), x64, (3, 4, 0), (1, 3, 2, 4)) * 0.5
    del x64
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x65, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x65
    x66 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x66 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x66 += einsum(x1, (0, 1, 2), (0, 1, 2)) * 2.0
    x67 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x67 += einsum(x52, (0, 1, 2, 3), (1, 0, 3, 2))
    del x52
    x67 += einsum(t2.aaaa, (0, 1, 2, 3), x33, (1, 4, 5, 3), (4, 0, 5, 2)) * 2.0
    del x33
    x67 += einsum(v.aaa.xov, (0, 1, 2), x66, (3, 4, 0), (1, 3, 2, 4)) * -1.0
    del x66
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x67, (0, 4, 2, 5), (4, 1, 5, 3)) * -1.0
    del x67
    x68 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x68 += einsum(v.aaa.xov, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 3, 2, 4))
    x69 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x69 += einsum(v.aaa.xoo, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    x69 += einsum(t2.abab, (0, 1, 2, 3), x68, (4, 1, 2, 5), (4, 0, 5, 3)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x69, (0, 4, 3, 5), (4, 1, 2, 5)) * -1.0
    del x69
    x70 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x70 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x70 += einsum(x0, (0, 1, 2), (0, 1, 2)) * -1.0
    del x0
    x70 += einsum(x50, (0, 1, 2), (0, 1, 2))
    del x50
    x70 += einsum(x1, (0, 1, 2), (0, 1, 2)) * 2.0
    del x1
    t2new_abab += einsum(v.abb.xov, (0, 1, 2), x70, (3, 4, 0), (3, 1, 4, 2))
    del x70
    x71 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x71 += einsum(t1.bb, (0, 1), v.abb.xvv, (2, 3, 1), (0, 3, 2))
    x72 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x72 += einsum(x21, (0, 1, 2), (0, 1, 2)) * -1.0
    del x21
    x72 += einsum(x71, (0, 1, 2), (0, 1, 2))
    x72 += einsum(x23, (0, 1, 2), (0, 1, 2)) * 2.0
    t2new_abab += einsum(v.aaa.xov, (0, 1, 2), x72, (3, 4, 0), (1, 3, 2, 4))
    del x72
    x73 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x73 += einsum(f.bb.vv, (0, 1), (0, 1)) * -0.5
    x73 += einsum(v.abb.xov, (0, 1, 2), x29, (1, 3, 0), (2, 3)) * 0.5
    t2new_abab += einsum(x73, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x73
    x74 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x74 += einsum(f.aa.vv, (0, 1), (0, 1)) * -0.5
    x74 += einsum(x41, (0, 1), (0, 1))
    del x41
    t2new_abab += einsum(x74, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -2.0
    del x74
    x75 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x75 += einsum(v.aaa.xoo, (0, 1, 2), v.abb.xoo, (0, 3, 4), (1, 2, 3, 4))
    x75 += einsum(t2.abab, (0, 1, 2, 3), x68, (4, 5, 2, 3), (4, 0, 5, 1))
    del x68
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x75, (0, 4, 1, 5), (4, 5, 2, 3))
    del x75
    x76 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x76 += einsum(f.aa.oo, (0, 1), (0, 1)) * 0.5
    x76 += einsum(x47, (0, 1), (0, 1))
    del x47
    t2new_abab += einsum(x76, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -2.0
    del x76
    x77 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x77 += einsum(v.abb.xvv, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x77, (4, 3, 5, 2), (0, 1, 4, 5)) * 2.0
    del x77
    x78 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x78 += einsum(v.abb.xov, (0, 1, 2), x71, (3, 4, 0), (3, 1, 2, 4))
    del x71
    x79 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x79 += einsum(t2.bbbb, (0, 1, 2, 3), x63, (4, 1, 5, 3), (0, 4, 2, 5))
    del x63
    x80 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x80 += einsum(x22, (0, 1, 2), x23, (3, 4, 2), (0, 3, 1, 4))
    del x23, x22
    x81 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x81 += einsum(v.abb.xov, (0, 1, 2), x24, (3, 4, 0), (1, 3, 2, 4)) * 2.0
    del x24
    x82 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x82 += einsum(x78, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x78
    x82 += einsum(x79, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x79
    x82 += einsum(x80, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x80
    x82 += einsum(x81, (0, 1, 2, 3), (1, 0, 3, 2))
    del x81
    t2new_bbbb += einsum(x82, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x82, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x82, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb += einsum(x82, (0, 1, 2, 3), (1, 0, 3, 2))
    del x82
    x83 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x83 += einsum(f.bb.oo, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4))
    x84 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x84 += einsum(x30, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x30
    x85 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x85 += einsum(x83, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x83
    x85 += einsum(x84, (0, 1, 2, 3), (0, 1, 3, 2))
    del x84
    t2new_bbbb += einsum(x85, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x85, (0, 1, 2, 3), (1, 0, 2, 3))
    del x85
    x86 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x86 += einsum(x37, (0, 1, 2, 3), (1, 0, 2, 3))
    x86 += einsum(x37, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x87 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x87 += einsum(t2.bbbb, (0, 1, 2, 3), x86, (1, 4, 3, 5), (0, 4, 2, 5))
    del x86
    x88 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x88 += einsum(t2.bbbb, (0, 1, 2, 3), x87, (4, 1, 5, 3), (0, 4, 2, 5)) * -4.0
    del x87
    x89 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x89 += einsum(v.abb.xov, (0, 1, 2), x29, (1, 3, 0), (2, 3))
    del x29
    x90 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x90 += einsum(x89, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x89
    x91 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x91 += einsum(x88, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x88
    x91 += einsum(x90, (0, 1, 2, 3), (1, 0, 2, 3))
    del x90
    t2new_bbbb += einsum(x91, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x91, (0, 1, 2, 3), (0, 1, 3, 2))
    del x91
    x92 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x92 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x93 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x93 += einsum(t2.abab, (0, 1, 2, 3), x34, (0, 4, 2, 5), (4, 1, 5, 3))
    del x34
    x94 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x94 += einsum(t2.abab, (0, 1, 2, 3), x93, (0, 4, 2, 5), (1, 4, 3, 5)) * -1.0
    del x93
    x95 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x95 += einsum(x92, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x92
    x95 += einsum(x37, (0, 1, 2, 3), (1, 0, 2, 3))
    x95 += einsum(x94, (0, 1, 2, 3), (1, 0, 2, 3))
    del x94
    t2new_bbbb += einsum(x95, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x95, (0, 1, 2, 3), (0, 1, 3, 2))
    del x95
    x96 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x96 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xoo, (0, 3, 4), (1, 3, 4, 2))
    x96 += einsum(t2.bbbb, (0, 1, 2, 3), x37, (4, 5, 2, 3), (5, 0, 4, 1))
    del x37
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x96, (0, 4, 1, 5), (4, 5, 2, 3)) * -2.0
    del x96

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t1new": t1new, "t2new": t2new}

