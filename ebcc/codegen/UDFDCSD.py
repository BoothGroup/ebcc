# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((naux,), dtype=types[float])
    x0 += einsum(t1.bb, (0, 1), v.abb.xov, (2, 0, 1), (2,))
    x1 = np.zeros((naux,), dtype=types[float])
    x1 += einsum(t1.aa, (0, 1), v.aaa.xov, (2, 0, 1), (2,))
    x1 += einsum(x0, (0,), (0,)) * 2.0
    x2 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x2 += einsum(v.aaa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    x2 += einsum(v.abb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (3, 4, 0))
    x2 += einsum(x1, (0,), t1.aa, (1, 2), (1, 2, 0)) * 0.5
    del x1
    e_cc = 0
    e_cc += einsum(v.aaa.xov, (0, 1, 2), x2, (1, 2, 0), ())
    del x2
    x3 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x3 += einsum(t1.bb, (0, 1), v.abb.xov, (2, 3, 1), (0, 3, 2))
    x4 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x4 += einsum(f.bb.ov, (0, 1), (0, 1))
    x4 += einsum(v.abb.xov, (0, 1, 2), x3, (1, 3, 0), (3, 2)) * -0.5
    del x3
    e_cc += einsum(t1.bb, (0, 1), x4, (0, 1), ())
    del x4
    x5 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x5 += einsum(t1.aa, (0, 1), v.aaa.xov, (2, 3, 1), (0, 3, 2))
    x6 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum(f.aa.ov, (0, 1), (0, 1))
    x6 += einsum(v.aaa.xov, (0, 1, 2), x5, (1, 3, 0), (3, 2)) * -0.5
    del x5
    e_cc += einsum(t1.aa, (0, 1), x6, (0, 1), ())
    del x6
    x7 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x7 += einsum(v.abb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    x7 += einsum(x0, (0,), t1.bb, (1, 2), (1, 2, 0)) * 0.5
    del x0
    e_cc += einsum(v.abb.xov, (0, 1, 2), x7, (1, 2, 0), ())
    del x7

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, **kwargs):
    t1new = Namespace()
    t2new = Namespace()

    # T amplitudes
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    t1new_aa += einsum(f.aa.vv, (0, 1), t1.aa, (2, 1), (2, 0))
    t1new_aa += einsum(f.aa.ov, (0, 1), (0, 1))
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    t1new_bb += einsum(f.bb.vv, (0, 1), t1.bb, (2, 1), (2, 0))
    t1new_bb += einsum(f.bb.ov, (0, 1), (0, 1))
    x0 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x0 += einsum(t1.aa, (0, 1), v.aaa.xoo, (2, 3, 0), (3, 1, 2))
    x1 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x1 += einsum(v.abb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (3, 4, 0))
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x2 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -0.5
    x3 = np.zeros((naux,), dtype=types[float])
    x3 += einsum(t1.aa, (0, 1), v.aaa.xov, (2, 0, 1), (2,))
    x4 = np.zeros((naux,), dtype=types[float])
    x4 += einsum(t1.bb, (0, 1), v.abb.xov, (2, 0, 1), (2,))
    x5 = np.zeros((naux,), dtype=types[float])
    x5 += einsum(x3, (0,), (0,))
    del x3
    x5 += einsum(x4, (0,), (0,))
    del x4
    x6 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x6 += einsum(x0, (0, 1, 2), (0, 1, 2)) * -1.0
    x6 += einsum(x1, (0, 1, 2), (0, 1, 2))
    x6 += einsum(v.aaa.xov, (0, 1, 2), x2, (1, 3, 2, 4), (3, 4, 0)) * 2.0
    del x2
    x6 += einsum(x5, (0,), t1.aa, (1, 2), (1, 2, 0))
    t1new_aa += einsum(v.aaa.xvv, (0, 1, 2), x6, (3, 2, 0), (3, 1))
    del x6
    x7 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x7 += einsum(t1.aa, (0, 1), v.aaa.xov, (2, 3, 1), (0, 3, 2))
    x8 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x8 += einsum(v.aaa.xoo, (0, 1, 2), (1, 2, 0))
    x8 += einsum(x7, (0, 1, 2), (0, 1, 2))
    x9 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x9 += einsum(v.abb.xov, (0, 1, 2), x8, (3, 4, 0), (4, 3, 1, 2))
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x9, (0, 4, 1, 3), (4, 2)) * -1.0
    del x9
    x10 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x10 += einsum(v.aaa.xoo, (0, 1, 2), (1, 2, 0))
    x10 += einsum(x7, (0, 1, 2), (1, 0, 2))
    x11 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x11 += einsum(v.aaa.xov, (0, 1, 2), x10, (3, 4, 0), (1, 3, 4, 2))
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), x11, (0, 1, 4, 3), (4, 2)) * 2.0
    x12 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x12 += einsum(t1.bb, (0, 1), v.abb.xov, (2, 3, 1), (0, 3, 2))
    x13 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x13 += einsum(v.abb.xov, (0, 1, 2), x12, (1, 3, 0), (3, 2))
    x14 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x14 += einsum(x5, (0,), v.abb.xov, (0, 1, 2), (1, 2))
    t1new_bb += einsum(x14, (0, 1), (0, 1))
    x15 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x15 += einsum(f.bb.ov, (0, 1), (0, 1))
    x15 += einsum(x13, (0, 1), (0, 1)) * -1.0
    x15 += einsum(x14, (0, 1), (0, 1))
    del x14
    t1new_aa += einsum(x15, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    t1new_bb += einsum(x15, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    x16 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x16 += einsum(v.aaa.xov, (0, 1, 2), x7, (1, 3, 0), (3, 2))
    x17 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x17 += einsum(x5, (0,), v.aaa.xov, (0, 1, 2), (1, 2))
    t1new_aa += einsum(x17, (0, 1), (0, 1))
    x18 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x18 += einsum(f.aa.ov, (0, 1), (0, 1))
    x18 += einsum(x16, (0, 1), (0, 1)) * -1.0
    x18 += einsum(x17, (0, 1), (0, 1))
    del x17
    t1new_aa += einsum(x18, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_bb += einsum(x18, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    x19 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x19 += einsum(v.aaa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    x20 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x20 += einsum(x0, (0, 1, 2), (0, 1, 2)) * -1.0
    x20 += einsum(x19, (0, 1, 2), (0, 1, 2)) * 2.0
    x20 += einsum(x1, (0, 1, 2), (0, 1, 2))
    x20 += einsum(x5, (0,), t1.aa, (1, 2), (1, 2, 0))
    x21 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x21 += einsum(x5, (0,), v.aaa.xoo, (0, 1, 2), (1, 2))
    x22 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x22 += einsum(f.aa.ov, (0, 1), (0, 1))
    x22 += einsum(x16, (0, 1), (0, 1)) * -1.0
    del x16
    x23 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x23 += einsum(f.aa.oo, (0, 1), (0, 1))
    x23 += einsum(v.aaa.xov, (0, 1, 2), x20, (3, 2, 0), (1, 3))
    del x20
    x23 += einsum(x21, (0, 1), (1, 0))
    x23 += einsum(t1.aa, (0, 1), x22, (2, 1), (2, 0))
    t1new_aa += einsum(t1.aa, (0, 1), x23, (0, 2), (2, 1)) * -1.0
    del x23
    x24 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x24 += einsum(t1.bb, (0, 1), v.abb.xoo, (2, 3, 0), (3, 1, 2))
    x25 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x25 += einsum(v.aaa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    x26 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x26 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x26 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -0.5
    x27 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x27 += einsum(x24, (0, 1, 2), (0, 1, 2)) * -1.0
    x27 += einsum(x25, (0, 1, 2), (0, 1, 2))
    x27 += einsum(v.abb.xov, (0, 1, 2), x26, (1, 3, 2, 4), (3, 4, 0)) * 2.0
    x27 += einsum(x5, (0,), t1.bb, (1, 2), (1, 2, 0))
    t1new_bb += einsum(v.abb.xvv, (0, 1, 2), x27, (3, 2, 0), (3, 1))
    del x27
    x28 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x28 += einsum(v.abb.xoo, (0, 1, 2), (1, 2, 0))
    x28 += einsum(x12, (0, 1, 2), (1, 0, 2))
    x29 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x29 += einsum(v.abb.xov, (0, 1, 2), x28, (3, 4, 0), (4, 1, 3, 2))
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), x29, (4, 0, 1, 3), (4, 2)) * 2.0
    del x29
    x30 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x30 += einsum(v.abb.xoo, (0, 1, 2), (1, 2, 0))
    x30 += einsum(x12, (0, 1, 2), (0, 1, 2))
    x31 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x31 += einsum(v.aaa.xov, (0, 1, 2), x30, (3, 4, 0), (1, 4, 3, 2))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x31, (0, 1, 4, 2), (4, 3)) * -1.0
    del x31
    x32 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x32 += einsum(v.abb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    x33 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x33 += einsum(x24, (0, 1, 2), (0, 1, 2)) * -1.0
    x33 += einsum(x25, (0, 1, 2), (0, 1, 2))
    x33 += einsum(x32, (0, 1, 2), (0, 1, 2)) * 2.0
    x33 += einsum(x5, (0,), t1.bb, (1, 2), (1, 2, 0))
    x34 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x34 += einsum(x5, (0,), v.abb.xoo, (0, 1, 2), (1, 2))
    x35 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x35 += einsum(f.bb.ov, (0, 1), (0, 1))
    x35 += einsum(x13, (0, 1), (0, 1)) * -1.0
    del x13
    x36 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x36 += einsum(t1.bb, (0, 1), x35, (2, 1), (0, 2))
    x37 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x37 += einsum(f.bb.oo, (0, 1), (0, 1))
    x37 += einsum(v.abb.xov, (0, 1, 2), x33, (3, 2, 0), (1, 3))
    del x33
    x37 += einsum(x34, (0, 1), (1, 0))
    x37 += einsum(x36, (0, 1), (1, 0))
    t1new_bb += einsum(t1.bb, (0, 1), x37, (0, 2), (2, 1)) * -1.0
    del x37
    x38 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x38 += einsum(v.aaa.xov, (0, 1, 2), v.aaa.xov, (0, 3, 4), (1, 3, 2, 4))
    x39 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x39 += einsum(t2.aaaa, (0, 1, 2, 3), x38, (4, 5, 2, 3), (0, 1, 5, 4)) * -1.0
    x40 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x40 += einsum(t1.aa, (0, 1), x39, (2, 3, 4, 0), (2, 3, 4, 1))
    del x39
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum(t1.aa, (0, 1), x40, (2, 3, 0, 4), (2, 3, 1, 4)) * 2.0
    del x40
    x41 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x41 += einsum(v.aaa.xvv, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x41, (4, 2, 5, 3), (0, 1, 4, 5)) * -2.0
    del x41
    x42 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x42 += einsum(t1.aa, (0, 1), v.aaa.xvv, (2, 3, 1), (0, 3, 2))
    x43 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x43 += einsum(x42, (0, 1, 2), x42, (3, 4, 2), (0, 3, 1, 4))
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x44 += einsum(x19, (0, 1, 2), x19, (3, 4, 2), (0, 3, 1, 4))
    x45 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x45 += einsum(x1, (0, 1, 2), x1, (3, 4, 2), (0, 3, 1, 4))
    x46 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x46 += einsum(x42, (0, 1, 2), (0, 1, 2))
    x46 += einsum(x19, (0, 1, 2), (0, 1, 2))
    x46 += einsum(x1, (0, 1, 2), (0, 1, 2)) * 0.5
    x47 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x47 += einsum(v.aaa.xov, (0, 1, 2), x46, (1, 3, 0), (2, 3))
    del x46
    x48 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x48 += einsum(x5, (0,), v.aaa.xvv, (0, 1, 2), (1, 2))
    x49 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x49 += einsum(x47, (0, 1), (0, 1))
    del x47
    x49 += einsum(x48, (0, 1), (1, 0)) * -1.0
    x50 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x50 += einsum(x49, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x49
    x51 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x51 += einsum(v.aaa.xov, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 2, 3, 4))
    x52 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x52 += einsum(t2.aaaa, (0, 1, 2, 3), x51, (4, 3, 5, 2), (0, 1, 4, 5)) * -1.0
    del x51
    x53 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x53 += einsum(x18, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    x54 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x54 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xoo, (0, 3, 4), (1, 3, 4, 2))
    x55 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x55 += einsum(x7, (0, 1, 2), x7, (3, 4, 2), (0, 3, 1, 4))
    x56 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x56 += einsum(x54, (0, 1, 2, 3), (3, 2, 1, 0))
    x56 += einsum(x55, (0, 1, 2, 3), (2, 3, 1, 0))
    x57 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x57 += einsum(t1.aa, (0, 1), x56, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x56
    x58 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x58 += einsum(x52, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x52
    x58 += einsum(x53, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x53
    x58 += einsum(x57, (0, 1, 2, 3), (0, 2, 1, 3))
    del x57
    x59 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x59 += einsum(t1.aa, (0, 1), x58, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    del x58
    x60 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x60 += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3))
    del x43
    x60 += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x44
    x60 += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    del x45
    x60 += einsum(x50, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x50
    x60 += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3))
    del x59
    t2new_aaaa += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x60, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x60
    x61 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x61 += einsum(v.aaa.xoo, (0, 1, 2), x7, (3, 4, 0), (3, 1, 2, 4))
    x62 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x62 += einsum(t2.aaaa, (0, 1, 2, 3), x61, (4, 0, 5, 1), (4, 5, 2, 3))
    x63 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x63 += einsum(t1.aa, (0, 1), x18, (2, 1), (0, 2))
    del x18
    x64 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x64 += einsum(f.aa.oo, (0, 1), (0, 1))
    x64 += einsum(x63, (0, 1), (0, 1))
    del x63
    x65 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x65 += einsum(x64, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x64
    x66 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x66 += einsum(x0, (0, 1, 2), (0, 1, 2)) * -1.0
    x66 += einsum(x19, (0, 1, 2), (0, 1, 2))
    x66 += einsum(x1, (0, 1, 2), (0, 1, 2)) * 0.5
    x67 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x67 += einsum(v.aaa.xov, (0, 1, 2), x66, (3, 2, 0), (1, 3))
    del x66
    x68 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x68 += einsum(x67, (0, 1), (0, 1))
    del x67
    x68 += einsum(x21, (0, 1), (1, 0))
    del x21
    x69 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x69 += einsum(x68, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x68
    x70 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x70 += einsum(x62, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x62
    x70 += einsum(x65, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x65
    x70 += einsum(x69, (0, 1, 2, 3), (0, 1, 3, 2))
    del x69
    t2new_aaaa += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x70, (0, 1, 2, 3), (1, 0, 2, 3))
    del x70
    x71 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x71 += einsum(v.aaa.xov, (0, 1, 2), x1, (3, 4, 0), (3, 1, 4, 2))
    x72 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x72 += einsum(v.aaa.xvv, (0, 1, 2), x7, (3, 4, 0), (3, 4, 1, 2))
    x73 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x73 += einsum(t2.aaaa, (0, 1, 2, 3), x72, (4, 1, 5, 3), (4, 0, 2, 5))
    del x72
    x74 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x74 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0)) * 0.5
    x74 += einsum(x0, (0, 1, 2), (0, 1, 2)) * -0.5
    x74 += einsum(x19, (0, 1, 2), (0, 1, 2))
    x74 += einsum(x1, (0, 1, 2), (0, 1, 2)) * 0.5
    x75 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x75 += einsum(x42, (0, 1, 2), x74, (3, 4, 2), (0, 3, 1, 4)) * 2.0
    del x74
    x76 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x76 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 2, 3, 4))
    x77 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x77 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x77 += einsum(x1, (0, 1, 2), (0, 1, 2))
    x78 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x78 += einsum(v.aaa.xov, (0, 1, 2), x77, (3, 4, 0), (1, 3, 2, 4))
    del x77
    x79 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x79 += einsum(x76, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x76
    x79 += einsum(x78, (0, 1, 2, 3), (0, 1, 2, 3))
    del x78
    x80 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x80 += einsum(t2.aaaa, (0, 1, 2, 3), x79, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x79
    x81 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x81 += einsum(t1.aa, (0, 1), x61, (2, 3, 4, 0), (2, 3, 4, 1))
    del x61
    x82 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x82 += einsum(v.aaa.xoo, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 2, 3, 4))
    x83 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x83 += einsum(t2.abab, (0, 1, 2, 3), x82, (4, 5, 1, 3), (0, 5, 4, 2))
    del x82
    x84 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x84 += einsum(x42, (0, 1, 2), x7, (3, 4, 2), (3, 0, 4, 1))
    x85 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x85 += einsum(v.abb.xov, (0, 1, 2), x7, (3, 4, 0), (3, 4, 1, 2))
    x86 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x86 += einsum(t2.abab, (0, 1, 2, 3), x85, (4, 5, 1, 3), (4, 0, 5, 2))
    del x85
    x87 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x87 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xov, (0, 3, 4), (1, 2, 3, 4))
    x88 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x88 += einsum(x87, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x88 += einsum(x87, (0, 1, 2, 3), (2, 1, 0, 3))
    del x87
    x89 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x89 += einsum(t2.aaaa, (0, 1, 2, 3), x88, (1, 4, 5, 3), (0, 4, 5, 2)) * 2.0
    del x88
    x90 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x90 += einsum(v.aaa.xov, (0, 1, 2), x7, (3, 4, 0), (3, 1, 4, 2))
    del x7
    x91 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x91 += einsum(x90, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x91 += einsum(x90, (0, 1, 2, 3), (0, 2, 1, 3))
    del x90
    x92 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x92 += einsum(t2.aaaa, (0, 1, 2, 3), x91, (4, 1, 5, 3), (0, 4, 5, 2)) * 2.0
    del x91
    x93 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x93 += einsum(x81, (0, 1, 2, 3), (0, 2, 1, 3))
    del x81
    x93 += einsum(x83, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x83
    x93 += einsum(x84, (0, 1, 2, 3), (0, 1, 2, 3))
    del x84
    x93 += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3))
    del x86
    x93 += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x89
    x93 += einsum(x92, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x92
    x93 += einsum(x11, (0, 1, 2, 3), (2, 0, 1, 3))
    del x11
    x94 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x94 += einsum(t1.aa, (0, 1), x93, (2, 3, 0, 4), (2, 3, 1, 4))
    del x93
    x95 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x95 += einsum(x71, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x71
    x95 += einsum(x73, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x73
    x95 += einsum(x75, (0, 1, 2, 3), (0, 1, 3, 2))
    del x75
    x95 += einsum(x80, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x80
    x95 += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3))
    del x94
    t2new_aaaa += einsum(x95, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x95, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa += einsum(x95, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x95, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x95
    x96 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x96 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x97 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x97 += einsum(x96, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x96
    x97 += einsum(x38, (0, 1, 2, 3), (1, 0, 2, 3))
    del x38
    t2new_aaaa += einsum(x97, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x97, (0, 1, 2, 3), (0, 1, 3, 2))
    del x97
    x98 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x98 += einsum(x54, (0, 1, 2, 3), (3, 2, 1, 0))
    del x54
    x98 += einsum(x55, (0, 1, 2, 3), (2, 1, 3, 0))
    del x55
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x98, (0, 4, 1, 5), (4, 5, 2, 3)) * -2.0
    del x98
    x99 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x99 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x99 += einsum(x42, (0, 1, 2), (0, 1, 2))
    x99 += einsum(x19, (0, 1, 2), (0, 1, 2)) * 2.0
    x99 += einsum(x1, (0, 1, 2), (0, 1, 2))
    x100 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x100 += einsum(t1.bb, (0, 1), v.abb.xvv, (2, 3, 1), (0, 3, 2))
    x101 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x101 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0)) * 0.5
    x101 += einsum(x100, (0, 1, 2), (0, 1, 2)) * 0.5
    x101 += einsum(x25, (0, 1, 2), (0, 1, 2)) * 0.5
    x101 += einsum(x32, (0, 1, 2), (0, 1, 2))
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum(x101, (0, 1, 2), x99, (3, 4, 2), (3, 0, 4, 1)) * 2.0
    del x101, x99
    x102 = np.zeros((nvir[1], nvir[1], naux), dtype=types[float])
    x102 += einsum(t1.bb, (0, 1), v.abb.xov, (2, 0, 3), (1, 3, 2))
    x103 = np.zeros((nvir[1], nvir[1], naux), dtype=types[float])
    x103 += einsum(v.abb.xvv, (0, 1, 2), (1, 2, 0))
    x103 += einsum(x102, (0, 1, 2), (0, 1, 2)) * -1.0
    del x102
    x104 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x104 += einsum(v.abb.xov, (0, 1, 2), x30, (3, 4, 0), (4, 3, 1, 2))
    x105 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x105 += einsum(x103, (0, 1, 2), x30, (3, 4, 2), (4, 3, 1, 0))
    x105 += einsum(t1.bb, (0, 1), x104, (0, 2, 3, 4), (3, 2, 4, 1))
    del x104
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x105, (1, 4, 3, 5), (0, 4, 2, 5)) * -1.0
    del x105
    x106 = np.zeros((nvir[0], nvir[0], naux), dtype=types[float])
    x106 += einsum(t1.aa, (0, 1), v.aaa.xov, (2, 0, 3), (1, 3, 2))
    x107 = np.zeros((nvir[0], nvir[0], naux), dtype=types[float])
    x107 += einsum(v.aaa.xvv, (0, 1, 2), (1, 2, 0))
    x107 += einsum(x106, (0, 1, 2), (0, 1, 2)) * -1.0
    del x106
    x108 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x108 += einsum(v.aaa.xov, (0, 1, 2), x8, (3, 4, 0), (4, 3, 1, 2))
    x109 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x109 += einsum(x107, (0, 1, 2), x8, (3, 4, 2), (4, 3, 1, 0))
    x109 += einsum(t1.aa, (0, 1), x108, (0, 2, 3, 4), (3, 2, 4, 1))
    del x108
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x109, (0, 4, 2, 5), (4, 1, 5, 3)) * -1.0
    del x109
    x110 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x110 += einsum(x107, (0, 1, 2), x30, (3, 4, 2), (4, 3, 1, 0))
    del x107
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x110, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    del x110
    x111 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x111 += einsum(x103, (0, 1, 2), x8, (3, 4, 2), (4, 3, 1, 0))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x111, (0, 4, 3, 5), (4, 1, 2, 5)) * -1.0
    del x111
    x112 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x112 += einsum(v.aaa.xov, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    x113 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x113 += einsum(t1.aa, (0, 1), x112, (0, 2, 3, 4), (2, 1, 4, 3))
    del x112
    x113 += einsum(v.aaa.xvv, (0, 1, 2), x103, (3, 4, 0), (1, 2, 4, 3)) * -1.0
    del x103
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x113, (2, 4, 3, 5), (0, 1, 4, 5)) * -1.0
    del x113
    x114 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x114 += einsum(x30, (0, 1, 2), x8, (3, 4, 2), (4, 3, 1, 0))
    x115 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x115 += einsum(t2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    x115 += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    t2new_abab += einsum(x114, (0, 1, 2, 3), x115, (0, 2, 4, 5), (1, 3, 4, 5))
    del x114, x115
    x116 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x116 += einsum(x42, (0, 1, 2), (0, 1, 2))
    x116 += einsum(x19, (0, 1, 2), (0, 1, 2))
    x116 += einsum(x1, (0, 1, 2), (0, 1, 2)) * 0.5
    x116 += einsum(x5, (0,), t1.aa, (1, 2), (1, 2, 0))
    x117 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x117 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x117 += einsum(v.aaa.xov, (0, 1, 2), x116, (1, 3, 0), (2, 3))
    del x116
    x117 += einsum(x48, (0, 1), (1, 0)) * -1.0
    del x48
    x117 += einsum(t1.aa, (0, 1), x22, (0, 2), (2, 1))
    t2new_abab += einsum(x117, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    del x117
    x118 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x118 += einsum(x100, (0, 1, 2), (0, 1, 2))
    x118 += einsum(x25, (0, 1, 2), (0, 1, 2)) * 0.5
    x118 += einsum(x32, (0, 1, 2), (0, 1, 2))
    x118 += einsum(x5, (0,), t1.bb, (1, 2), (1, 2, 0))
    x119 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x119 += einsum(x5, (0,), v.abb.xvv, (0, 1, 2), (1, 2))
    x120 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x120 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x120 += einsum(v.abb.xov, (0, 1, 2), x118, (1, 3, 0), (2, 3))
    del x118
    x120 += einsum(x119, (0, 1), (1, 0)) * -1.0
    x120 += einsum(t1.bb, (0, 1), x35, (0, 2), (2, 1))
    del x35
    t2new_abab += einsum(x120, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x120
    x121 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x121 += einsum(x24, (0, 1, 2), (0, 1, 2)) * -1.0
    x121 += einsum(x25, (0, 1, 2), (0, 1, 2)) * 0.5
    x121 += einsum(x32, (0, 1, 2), (0, 1, 2))
    x121 += einsum(x5, (0,), t1.bb, (1, 2), (1, 2, 0))
    x122 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x122 += einsum(f.bb.oo, (0, 1), (0, 1))
    x122 += einsum(v.abb.xov, (0, 1, 2), x121, (3, 2, 0), (1, 3))
    del x121
    x122 += einsum(x34, (0, 1), (1, 0))
    x122 += einsum(x36, (0, 1), (1, 0))
    del x36
    t2new_abab += einsum(x122, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x122
    x123 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x123 += einsum(x0, (0, 1, 2), (0, 1, 2)) * -1.0
    del x0
    x123 += einsum(x19, (0, 1, 2), (0, 1, 2))
    del x19
    x123 += einsum(x1, (0, 1, 2), (0, 1, 2)) * 0.5
    del x1
    x123 += einsum(x5, (0,), t1.aa, (1, 2), (1, 2, 0))
    x124 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x124 += einsum(f.aa.oo, (0, 1), (0, 1)) * 2.0
    x124 += einsum(v.aaa.xov, (0, 1, 2), x123, (3, 2, 0), (1, 3)) * 2.0
    del x123
    x124 += einsum(x5, (0,), v.aaa.xoo, (0, 1, 2), (1, 2)) * 2.0
    del x5
    x124 += einsum(t1.aa, (0, 1), x22, (2, 1), (2, 0)) * 2.0
    del x22
    t2new_abab += einsum(x124, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -0.5
    del x124
    x125 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x125 += einsum(v.aaa.xov, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 3, 2, 4))
    x126 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x126 += einsum(t2.abab, (0, 1, 2, 3), x125, (4, 5, 2, 3), (0, 4, 1, 5))
    del x125
    x127 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x127 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x127 += einsum(x42, (0, 1, 2), (0, 1, 2))
    del x42
    x128 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x128 += einsum(v.aaa.xov, (0, 1, 2), x28, (3, 4, 0), (1, 4, 3, 2))
    del x28
    x129 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x129 += einsum(t1.aa, (0, 1), x126, (2, 0, 3, 4), (2, 4, 3, 1)) * -1.0
    del x126
    x129 += einsum(x127, (0, 1, 2), x30, (3, 4, 2), (0, 4, 3, 1))
    del x30, x127
    x129 += einsum(t2.aaaa, (0, 1, 2, 3), x128, (1, 4, 5, 3), (0, 5, 4, 2)) * 2.0
    del x128
    t2new_abab += einsum(t1.bb, (0, 1), x129, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x129
    x130 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x130 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0))
    x130 += einsum(x100, (0, 1, 2), (0, 1, 2))
    x131 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x131 += einsum(v.abb.xov, (0, 1, 2), x10, (3, 4, 0), (4, 3, 1, 2))
    del x10
    x132 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x132 += einsum(x130, (0, 1, 2), x8, (3, 4, 2), (4, 3, 0, 1)) * 0.5
    del x8, x130
    x132 += einsum(t2.bbbb, (0, 1, 2, 3), x131, (4, 5, 1, 3), (5, 4, 0, 2))
    del x131
    t2new_abab += einsum(t1.aa, (0, 1), x132, (0, 2, 3, 4), (2, 3, 1, 4)) * -2.0
    del x132
    x133 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x133 += einsum(v.abb.xov, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 3, 2, 4))
    x134 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x134 += einsum(t2.bbbb, (0, 1, 2, 3), x133, (4, 5, 2, 3), (0, 1, 5, 4)) * -1.0
    x135 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x135 += einsum(t1.bb, (0, 1), x134, (2, 3, 4, 0), (2, 3, 4, 1))
    del x134
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb += einsum(t1.bb, (0, 1), x135, (2, 3, 0, 4), (2, 3, 1, 4)) * 2.0
    del x135
    x136 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x136 += einsum(v.abb.xvv, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x136, (4, 3, 5, 2), (0, 1, 4, 5)) * 2.0
    del x136
    x137 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x137 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x138 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x138 += einsum(x25, (0, 1, 2), x25, (3, 4, 2), (0, 3, 1, 4))
    x139 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x139 += einsum(x137, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x137
    x139 += einsum(x133, (0, 1, 2, 3), (1, 0, 2, 3))
    del x133
    x139 += einsum(x138, (0, 1, 2, 3), (1, 0, 2, 3))
    del x138
    t2new_bbbb += einsum(x139, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x139, (0, 1, 2, 3), (0, 1, 3, 2))
    del x139
    x140 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x140 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    x141 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x141 += einsum(t2.bbbb, (0, 1, 2, 3), x140, (4, 1, 5, 3), (0, 4, 2, 5))
    del x140
    x142 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x142 += einsum(v.abb.xvv, (0, 1, 2), x12, (3, 4, 0), (3, 4, 1, 2))
    x143 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x143 += einsum(t2.bbbb, (0, 1, 2, 3), x142, (4, 1, 5, 3), (4, 0, 2, 5))
    del x142
    x144 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x144 += einsum(x25, (0, 1, 2), x32, (3, 4, 2), (0, 3, 1, 4))
    x145 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x145 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0))
    x145 += einsum(x24, (0, 1, 2), (0, 1, 2)) * -1.0
    x145 += einsum(x25, (0, 1, 2), (0, 1, 2))
    x145 += einsum(x32, (0, 1, 2), (0, 1, 2)) * 2.0
    x146 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x146 += einsum(x100, (0, 1, 2), x145, (3, 4, 2), (3, 0, 4, 1))
    del x145
    x147 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x147 += einsum(v.abb.xov, (0, 1, 2), x26, (1, 3, 2, 4), (3, 4, 0))
    del x26
    x148 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x148 += einsum(x24, (0, 1, 2), (0, 1, 2)) * -0.5
    x148 += einsum(x25, (0, 1, 2), (0, 1, 2)) * 0.5
    x148 += einsum(x147, (0, 1, 2), (0, 1, 2))
    del x147
    x149 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x149 += einsum(v.abb.xov, (0, 1, 2), x148, (3, 4, 0), (3, 1, 4, 2)) * 2.0
    del x148
    x150 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x150 += einsum(v.abb.xoo, (0, 1, 2), v.aaa.xov, (0, 3, 4), (3, 1, 2, 4))
    x151 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x151 += einsum(t2.abab, (0, 1, 2, 3), x150, (0, 4, 5, 2), (1, 5, 4, 3))
    del x150
    x152 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x152 += einsum(v.abb.xoo, (0, 1, 2), x12, (3, 4, 0), (3, 1, 2, 4))
    x153 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x153 += einsum(t1.bb, (0, 1), x152, (2, 3, 4, 0), (2, 3, 4, 1))
    x154 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x154 += einsum(x100, (0, 1, 2), x12, (3, 4, 2), (3, 0, 4, 1))
    x155 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x155 += einsum(v.aaa.xov, (0, 1, 2), x12, (3, 4, 0), (1, 3, 4, 2))
    x156 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x156 += einsum(t2.abab, (0, 1, 2, 3), x155, (0, 4, 5, 2), (4, 1, 5, 3))
    del x155
    x157 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x157 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 2, 3, 4))
    x158 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x158 += einsum(x157, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x158 += einsum(x157, (0, 1, 2, 3), (2, 1, 0, 3))
    del x157
    x159 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x159 += einsum(t2.bbbb, (0, 1, 2, 3), x158, (1, 4, 5, 3), (4, 5, 0, 2)) * 2.0
    del x158
    x160 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x160 += einsum(v.abb.xov, (0, 1, 2), x12, (3, 4, 0), (3, 1, 4, 2))
    x161 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x161 += einsum(x160, (0, 1, 2, 3), (0, 1, 2, 3))
    x161 += einsum(x160, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x160
    x162 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x162 += einsum(t2.bbbb, (0, 1, 2, 3), x161, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    del x161
    x163 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x163 += einsum(x151, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x151
    x163 += einsum(x153, (0, 1, 2, 3), (0, 2, 1, 3))
    del x153
    x163 += einsum(x154, (0, 1, 2, 3), (0, 2, 1, 3))
    del x154
    x163 += einsum(x156, (0, 1, 2, 3), (0, 2, 1, 3))
    del x156
    x163 += einsum(x159, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x159
    x163 += einsum(x162, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x162
    x164 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x164 += einsum(t1.bb, (0, 1), x163, (2, 0, 3, 4), (2, 3, 4, 1))
    del x163
    x165 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x165 += einsum(x141, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x141
    x165 += einsum(x143, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x143
    x165 += einsum(x144, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x144
    x165 += einsum(x146, (0, 1, 2, 3), (1, 0, 2, 3))
    del x146
    x165 += einsum(x149, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x149
    x165 += einsum(x164, (0, 1, 2, 3), (0, 1, 3, 2))
    del x164
    t2new_bbbb += einsum(x165, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x165, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb += einsum(x165, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb += einsum(x165, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x165
    x166 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x166 += einsum(t2.bbbb, (0, 1, 2, 3), x152, (4, 0, 5, 1), (4, 5, 2, 3))
    del x152
    x167 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x167 += einsum(t1.bb, (0, 1), x15, (2, 1), (0, 2))
    x168 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x168 += einsum(f.bb.oo, (0, 1), (0, 1))
    x168 += einsum(x167, (0, 1), (0, 1))
    del x167
    x169 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x169 += einsum(x168, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4)) * -2.0
    del x168
    x170 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x170 += einsum(x24, (0, 1, 2), (0, 1, 2)) * -1.0
    del x24
    x170 += einsum(x25, (0, 1, 2), (0, 1, 2)) * 0.5
    x170 += einsum(x32, (0, 1, 2), (0, 1, 2))
    x171 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x171 += einsum(v.abb.xov, (0, 1, 2), x170, (3, 2, 0), (3, 1))
    del x170
    x172 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x172 += einsum(x171, (0, 1), (1, 0))
    del x171
    x172 += einsum(x34, (0, 1), (1, 0))
    del x34
    x173 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x173 += einsum(x172, (0, 1), t2.bbbb, (2, 0, 3, 4), (1, 2, 3, 4)) * -2.0
    del x172
    x174 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x174 += einsum(x166, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x166
    x174 += einsum(x169, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x169
    x174 += einsum(x173, (0, 1, 2, 3), (1, 0, 3, 2))
    del x173
    t2new_bbbb += einsum(x174, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x174, (0, 1, 2, 3), (1, 0, 2, 3))
    del x174
    x175 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x175 += einsum(x100, (0, 1, 2), x100, (3, 4, 2), (0, 3, 1, 4))
    x176 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x176 += einsum(x32, (0, 1, 2), x32, (3, 4, 2), (0, 3, 1, 4))
    x177 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x177 += einsum(x100, (0, 1, 2), (0, 1, 2))
    del x100
    x177 += einsum(x25, (0, 1, 2), (0, 1, 2)) * 0.5
    del x25
    x177 += einsum(x32, (0, 1, 2), (0, 1, 2))
    del x32
    x178 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x178 += einsum(v.abb.xov, (0, 1, 2), x177, (1, 3, 0), (3, 2))
    del x177
    x179 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x179 += einsum(x178, (0, 1), (1, 0))
    del x178
    x179 += einsum(x119, (0, 1), (1, 0)) * -1.0
    del x119
    x180 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x180 += einsum(x179, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 1, 4)) * -2.0
    del x179
    x181 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x181 += einsum(v.abb.xov, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    x182 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x182 += einsum(t2.bbbb, (0, 1, 2, 3), x181, (4, 3, 5, 2), (0, 1, 4, 5)) * -1.0
    del x181
    x183 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x183 += einsum(x15, (0, 1), t2.bbbb, (2, 3, 4, 1), (0, 2, 3, 4)) * -1.0
    del x15
    x184 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x184 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xoo, (0, 3, 4), (1, 3, 4, 2))
    x185 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x185 += einsum(x12, (0, 1, 2), x12, (3, 4, 2), (0, 3, 1, 4))
    del x12
    x186 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x186 += einsum(x184, (0, 1, 2, 3), (3, 2, 1, 0))
    x186 += einsum(x185, (0, 1, 2, 3), (2, 3, 1, 0))
    x187 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x187 += einsum(t1.bb, (0, 1), x186, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x186
    x188 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x188 += einsum(x182, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x182
    x188 += einsum(x183, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x183
    x188 += einsum(x187, (0, 1, 2, 3), (0, 2, 1, 3))
    del x187
    x189 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x189 += einsum(t1.bb, (0, 1), x188, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    del x188
    x190 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x190 += einsum(x175, (0, 1, 2, 3), (0, 1, 2, 3))
    del x175
    x190 += einsum(x176, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x176
    x190 += einsum(x180, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x180
    x190 += einsum(x189, (0, 1, 2, 3), (1, 0, 3, 2))
    del x189
    t2new_bbbb += einsum(x190, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x190, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x190
    x191 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x191 += einsum(x184, (0, 1, 2, 3), (3, 2, 1, 0))
    del x184
    x191 += einsum(x185, (0, 1, 2, 3), (2, 1, 3, 0))
    del x185
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x191, (0, 4, 1, 5), (4, 5, 2, 3)) * -2.0
    del x191

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t1new": t1new, "t2new": t2new}

