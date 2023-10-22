# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((naux,), dtype=types[float])
    x0 += einsum(t1.aa, (0, 1), v.aaa.xov, (2, 0, 1), (2,))
    x1 = np.zeros((naux,), dtype=types[float])
    x1 += einsum(x0, (0,), (0,))
    x1 += einsum(t1.bb, (0, 1), v.abb.xov, (2, 0, 1), (2,)) * 0.5
    x2 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x2 += einsum(v.aaa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    x2 += einsum(v.abb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    x2 += einsum(x1, (0,), t1.bb, (1, 2), (1, 2, 0))
    del x1
    e_cc = 0
    e_cc += einsum(v.abb.xov, (0, 1, 2), x2, (1, 2, 0), ())
    del x2
    x3 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x3 += einsum(t1.bb, (0, 1), v.abb.xov, (2, 3, 1), (0, 3, 2))
    x4 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x4 += einsum(f.bb.ov, (0, 1), (0, 1)) * 2.0
    x4 += einsum(v.abb.xov, (0, 1, 2), x3, (1, 3, 0), (3, 2)) * -1.0
    del x3
    e_cc += einsum(t1.bb, (0, 1), x4, (0, 1), ()) * 0.5
    del x4
    x5 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x5 += einsum(t1.aa, (0, 1), v.aaa.xov, (2, 3, 1), (0, 3, 2))
    x6 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum(f.aa.ov, (0, 1), (0, 1)) * 2.0
    x6 += einsum(v.aaa.xov, (0, 1, 2), x5, (1, 3, 0), (3, 2)) * -1.0
    del x5
    e_cc += einsum(t1.aa, (0, 1), x6, (0, 1), ()) * 0.5
    del x6
    x7 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x7 += einsum(v.aaa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    x7 += einsum(x0, (0,), t1.aa, (1, 2), (1, 2, 0)) * 0.5
    del x0
    e_cc += einsum(v.aaa.xov, (0, 1, 2), x7, (1, 2, 0), ())
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
    x8 += einsum(x7, (0, 1, 2), (1, 0, 2))
    x9 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x9 += einsum(v.aaa.xov, (0, 1, 2), x8, (3, 4, 0), (1, 3, 4, 2))
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), x9, (0, 1, 4, 3), (4, 2)) * 2.0
    x10 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x10 += einsum(v.aaa.xoo, (0, 1, 2), (1, 2, 0))
    x10 += einsum(x7, (0, 1, 2), (0, 1, 2))
    x11 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x11 += einsum(v.abb.xov, (0, 1, 2), x10, (3, 4, 0), (4, 3, 1, 2))
    del x10
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x11, (0, 4, 1, 3), (4, 2)) * -1.0
    del x11
    x12 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x12 += einsum(v.aaa.xov, (0, 1, 2), x7, (1, 3, 0), (3, 2))
    x13 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x13 += einsum(x5, (0,), v.aaa.xov, (0, 1, 2), (1, 2))
    t1new_aa += einsum(x13, (0, 1), (0, 1))
    x14 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x14 += einsum(f.aa.ov, (0, 1), (0, 1))
    x14 += einsum(x12, (0, 1), (0, 1)) * -1.0
    x14 += einsum(x13, (0, 1), (0, 1))
    del x13
    t1new_aa += einsum(x14, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_bb += einsum(x14, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    del x14
    x15 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x15 += einsum(t1.bb, (0, 1), v.abb.xov, (2, 3, 1), (0, 3, 2))
    x16 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x16 += einsum(v.abb.xov, (0, 1, 2), x15, (1, 3, 0), (3, 2))
    x17 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x17 += einsum(x5, (0,), v.abb.xov, (0, 1, 2), (1, 2))
    t1new_bb += einsum(x17, (0, 1), (0, 1))
    x18 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x18 += einsum(f.bb.ov, (0, 1), (0, 1))
    x18 += einsum(x16, (0, 1), (0, 1)) * -1.0
    x18 += einsum(x17, (0, 1), (0, 1))
    del x17
    t1new_aa += einsum(x18, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    t1new_bb += einsum(x18, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    del x18
    x19 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x19 += einsum(x0, (0, 1, 2), (0, 1, 2)) * -1.0
    x19 += einsum(v.aaa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0)) * 2.0
    x19 += einsum(x1, (0, 1, 2), (0, 1, 2))
    del x1
    x19 += einsum(x5, (0,), t1.aa, (1, 2), (1, 2, 0))
    x20 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x20 += einsum(f.aa.ov, (0, 1), (0, 1))
    x20 += einsum(x12, (0, 1), (0, 1)) * -1.0
    del x12
    x21 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x21 += einsum(f.aa.oo, (0, 1), (0, 1)) * 0.5
    x21 += einsum(v.aaa.xov, (0, 1, 2), x19, (3, 2, 0), (1, 3)) * 0.5
    del x19
    x21 += einsum(x5, (0,), v.aaa.xoo, (0, 1, 2), (1, 2)) * 0.5
    x21 += einsum(t1.aa, (0, 1), x20, (2, 1), (2, 0)) * 0.5
    del x20
    t1new_aa += einsum(t1.aa, (0, 1), x21, (0, 2), (2, 1)) * -2.0
    del x21
    x22 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x22 += einsum(t1.bb, (0, 1), v.abb.xoo, (2, 3, 0), (3, 1, 2))
    x23 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x23 += einsum(v.aaa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    x24 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x24 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x24 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -0.5
    x25 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x25 += einsum(x22, (0, 1, 2), (0, 1, 2)) * -1.0
    x25 += einsum(x23, (0, 1, 2), (0, 1, 2))
    x25 += einsum(v.abb.xov, (0, 1, 2), x24, (1, 3, 2, 4), (3, 4, 0)) * 2.0
    del x24
    x25 += einsum(x5, (0,), t1.bb, (1, 2), (1, 2, 0))
    t1new_bb += einsum(v.abb.xvv, (0, 1, 2), x25, (3, 2, 0), (3, 1))
    del x25
    x26 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x26 += einsum(v.abb.xoo, (0, 1, 2), (1, 2, 0))
    x26 += einsum(x15, (0, 1, 2), (0, 1, 2))
    x27 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x27 += einsum(v.aaa.xov, (0, 1, 2), x26, (3, 4, 0), (1, 4, 3, 2))
    del x26
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x27, (0, 1, 4, 2), (4, 3)) * -1.0
    del x27
    x28 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x28 += einsum(v.abb.xoo, (0, 1, 2), (1, 2, 0))
    x28 += einsum(x15, (0, 1, 2), (1, 0, 2))
    x29 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x29 += einsum(v.abb.xov, (0, 1, 2), x28, (3, 4, 0), (1, 3, 4, 2))
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), x29, (0, 1, 4, 3), (4, 2)) * 2.0
    x30 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x30 += einsum(x22, (0, 1, 2), (0, 1, 2)) * -1.0
    x30 += einsum(x23, (0, 1, 2), (0, 1, 2))
    del x23
    x30 += einsum(v.abb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0)) * 2.0
    x30 += einsum(x5, (0,), t1.bb, (1, 2), (1, 2, 0))
    x31 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x31 += einsum(f.bb.ov, (0, 1), (0, 1))
    x31 += einsum(x16, (0, 1), (0, 1)) * -1.0
    del x16
    x32 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x32 += einsum(f.bb.oo, (0, 1), (0, 1))
    x32 += einsum(v.abb.xov, (0, 1, 2), x30, (3, 2, 0), (1, 3))
    del x30
    x32 += einsum(x5, (0,), v.abb.xoo, (0, 1, 2), (1, 2))
    del x5
    x32 += einsum(t1.bb, (0, 1), x31, (2, 1), (2, 0))
    del x31
    t1new_bb += einsum(t1.bb, (0, 1), x32, (0, 2), (2, 1)) * -1.0
    del x32
    x33 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x33 += einsum(f.aa.ov, (0, 1), t1.aa, (2, 1), (0, 2))
    x34 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x34 += einsum(f.aa.oo, (0, 1), (0, 1))
    x34 += einsum(x33, (0, 1), (0, 1))
    del x33
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum(x34, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    x35 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x35 += einsum(x34, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x34
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum(x35, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_aaaa += einsum(x35, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x35
    x36 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x36 += einsum(t1.aa, (0, 1), v.aaa.xvv, (2, 3, 1), (0, 3, 2))
    x37 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x37 += einsum(x36, (0, 1, 2), x36, (3, 4, 2), (0, 3, 1, 4))
    x38 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x38 += einsum(f.aa.ov, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4))
    x39 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x39 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xoo, (0, 3, 4), (1, 3, 4, 2))
    x40 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x40 += einsum(x7, (0, 1, 2), x7, (3, 4, 2), (0, 3, 1, 4))
    x41 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x41 += einsum(x39, (0, 1, 2, 3), (3, 2, 1, 0))
    del x39
    x41 += einsum(x40, (0, 1, 2, 3), (2, 3, 1, 0))
    del x40
    x42 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x42 += einsum(t1.aa, (0, 1), x41, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x41
    x43 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x43 += einsum(x38, (0, 1, 2, 3), (0, 2, 1, 3))
    del x38
    x43 += einsum(x42, (0, 1, 2, 3), (0, 2, 1, 3))
    del x42
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x44 += einsum(t1.aa, (0, 1), x43, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    del x43
    x45 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x45 += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3))
    del x37
    x45 += einsum(x44, (0, 1, 2, 3), (1, 0, 2, 3))
    del x44
    t2new_aaaa += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x45, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x45
    x46 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x46 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0)) * -1.0
    x46 += einsum(x0, (0, 1, 2), (0, 1, 2))
    del x0
    x47 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x47 += einsum(x36, (0, 1, 2), x46, (3, 4, 2), (0, 3, 1, 4))
    del x46
    x48 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x48 += einsum(v.aaa.xoo, (0, 1, 2), x7, (3, 4, 0), (3, 1, 2, 4))
    x49 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x49 += einsum(t1.aa, (0, 1), x48, (2, 3, 4, 0), (2, 3, 4, 1))
    del x48
    x50 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x50 += einsum(x36, (0, 1, 2), x7, (3, 4, 2), (3, 0, 4, 1))
    del x7
    x51 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x51 += einsum(x49, (0, 1, 2, 3), (0, 2, 1, 3))
    del x49
    x51 += einsum(x50, (0, 1, 2, 3), (0, 1, 2, 3))
    del x50
    x51 += einsum(x9, (0, 1, 2, 3), (2, 0, 1, 3))
    del x9
    x52 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x52 += einsum(t1.aa, (0, 1), x51, (2, 3, 0, 4), (2, 3, 1, 4))
    del x51
    x53 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x53 += einsum(x47, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x47
    x53 += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3))
    del x52
    t2new_aaaa += einsum(x53, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x53, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa += einsum(x53, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x53, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x53
    x54 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x54 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x55 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x55 += einsum(v.aaa.xov, (0, 1, 2), v.aaa.xov, (0, 3, 4), (1, 3, 2, 4))
    x56 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x56 += einsum(x54, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x54
    x56 += einsum(x55, (0, 1, 2, 3), (1, 0, 2, 3))
    del x55
    t2new_aaaa += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x56, (0, 1, 2, 3), (0, 1, 3, 2))
    del x56
    x57 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x57 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x57 += einsum(x36, (0, 1, 2), (0, 1, 2))
    del x36
    x57 += einsum(t1.aa, (0, 1), x8, (0, 2, 3), (2, 1, 3)) * -1.0
    del x8
    x58 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x58 += einsum(t1.bb, (0, 1), v.abb.xvv, (2, 3, 1), (0, 3, 2))
    x59 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x59 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0))
    x59 += einsum(x58, (0, 1, 2), (0, 1, 2))
    x59 += einsum(t1.bb, (0, 1), x28, (0, 2, 3), (2, 1, 3)) * -1.0
    del x28
    t2new_abab += einsum(x57, (0, 1, 2), x59, (3, 4, 2), (0, 3, 1, 4))
    del x57, x59
    x60 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x60 += einsum(f.bb.vv, (0, 1), (0, 1))
    x60 += einsum(f.bb.ov, (0, 1), t1.bb, (0, 2), (1, 2)) * -1.0
    t2new_abab += einsum(x60, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1))
    del x60
    x61 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x61 += einsum(f.aa.vv, (0, 1), (0, 1))
    x61 += einsum(f.aa.ov, (0, 1), t1.aa, (0, 2), (1, 2)) * -1.0
    t2new_abab += einsum(x61, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4))
    del x61
    x62 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x62 += einsum(f.bb.ov, (0, 1), t1.bb, (2, 1), (0, 2))
    x63 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x63 += einsum(f.bb.oo, (0, 1), (0, 1))
    x63 += einsum(x62, (0, 1), (0, 1))
    del x62
    t2new_abab += einsum(x63, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    x64 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x64 += einsum(x63, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x63
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb += einsum(x64, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_bbbb += einsum(x64, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x64
    x65 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x65 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0)) * -1.0
    x65 += einsum(x22, (0, 1, 2), (0, 1, 2))
    del x22
    x66 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x66 += einsum(x58, (0, 1, 2), x65, (3, 4, 2), (0, 3, 1, 4))
    del x65
    x67 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x67 += einsum(v.abb.xoo, (0, 1, 2), x15, (3, 4, 0), (3, 1, 2, 4))
    x68 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x68 += einsum(t1.bb, (0, 1), x67, (2, 3, 4, 0), (2, 3, 4, 1))
    del x67
    x69 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x69 += einsum(x15, (0, 1, 2), x58, (3, 4, 2), (0, 3, 1, 4))
    x70 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x70 += einsum(x68, (0, 1, 2, 3), (0, 2, 1, 3))
    del x68
    x70 += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3))
    del x69
    x70 += einsum(x29, (0, 1, 2, 3), (2, 0, 1, 3))
    del x29
    x71 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x71 += einsum(t1.bb, (0, 1), x70, (2, 3, 0, 4), (2, 3, 1, 4))
    del x70
    x72 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x72 += einsum(x66, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x66
    x72 += einsum(x71, (0, 1, 2, 3), (0, 1, 2, 3))
    del x71
    t2new_bbbb += einsum(x72, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x72, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb += einsum(x72, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb += einsum(x72, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x72
    x73 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x73 += einsum(x58, (0, 1, 2), x58, (3, 4, 2), (0, 3, 1, 4))
    del x58
    x74 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x74 += einsum(f.bb.ov, (0, 1), t2.bbbb, (2, 3, 4, 1), (0, 2, 3, 4))
    x75 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x75 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xoo, (0, 3, 4), (1, 3, 4, 2))
    x76 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x76 += einsum(x15, (0, 1, 2), x15, (3, 4, 2), (0, 3, 1, 4))
    del x15
    x77 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x77 += einsum(x75, (0, 1, 2, 3), (3, 2, 1, 0))
    del x75
    x77 += einsum(x76, (0, 1, 2, 3), (2, 3, 1, 0))
    del x76
    x78 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x78 += einsum(t1.bb, (0, 1), x77, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.5
    del x77
    x79 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x79 += einsum(x74, (0, 1, 2, 3), (0, 2, 1, 3))
    del x74
    x79 += einsum(x78, (0, 1, 2, 3), (0, 2, 1, 3))
    del x78
    x80 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x80 += einsum(t1.bb, (0, 1), x79, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    del x79
    x81 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x81 += einsum(x73, (0, 1, 2, 3), (0, 1, 2, 3))
    del x73
    x81 += einsum(x80, (0, 1, 2, 3), (1, 0, 2, 3))
    del x80
    t2new_bbbb += einsum(x81, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x81, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x81
    x82 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x82 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x83 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x83 += einsum(v.abb.xov, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 3, 2, 4))
    x84 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x84 += einsum(x82, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x82
    x84 += einsum(x83, (0, 1, 2, 3), (1, 0, 2, 3))
    del x83
    t2new_bbbb += einsum(x84, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x84, (0, 1, 2, 3), (0, 1, 3, 2))
    del x84

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t1new": t1new, "t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    l1new = Namespace()
    l2new = Namespace()

    # L amplitudes
    l1new_aa = np.zeros((nvir[0], nocc[0]), dtype=types[float])
    l1new_aa += einsum(f.aa.ov, (0, 1), (1, 0))
    l1new_bb = np.zeros((nvir[1], nocc[1]), dtype=types[float])
    l1new_bb += einsum(f.bb.ov, (0, 1), (1, 0))
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x0 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    x1 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x1 += einsum(t1.bb, (0, 1), l2.abab, (2, 1, 3, 4), (3, 4, 0, 2))
    x2 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x2 += einsum(t1.aa, (0, 1), v.aaa.xov, (2, 3, 1), (0, 3, 2))
    x3 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x3 += einsum(t1.bb, (0, 1), v.abb.xov, (2, 3, 1), (0, 3, 2))
    x4 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x4 += einsum(t1.aa, (0, 1), v.aaa.xvv, (2, 3, 1), (0, 3, 2))
    x5 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x5 += einsum(v.aaa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    x6 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x6 += einsum(v.abb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (3, 4, 0))
    x7 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x7 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x7 += einsum(x4, (0, 1, 2), (0, 1, 2))
    x7 += einsum(x5, (0, 1, 2), (0, 1, 2)) * 2.0
    x7 += einsum(x6, (0, 1, 2), (0, 1, 2))
    x8 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x8 += einsum(t1.bb, (0, 1), v.abb.xvv, (2, 3, 1), (0, 3, 2))
    x9 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x9 += einsum(v.aaa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    x10 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x10 += einsum(v.abb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    x11 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x11 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0))
    x11 += einsum(x8, (0, 1, 2), (0, 1, 2))
    x11 += einsum(x9, (0, 1, 2), (0, 1, 2))
    x11 += einsum(x10, (0, 1, 2), (0, 1, 2)) * 2.0
    x12 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4))
    x13 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x13 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    x14 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x14 += einsum(x12, (0, 1), (0, 1))
    x14 += einsum(x13, (0, 1), (0, 1)) * 0.5
    x15 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x15 += einsum(v.aaa.xoo, (0, 1, 2), (1, 2, 0))
    x15 += einsum(x2, (0, 1, 2), (0, 1, 2))
    x16 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x16 += einsum(v.aaa.xoo, (0, 1, 2), x0, (3, 1, 2, 4), (3, 4, 0)) * -1.0
    x16 += einsum(v.abb.xoo, (0, 1, 2), x1, (3, 1, 2, 4), (3, 4, 0)) * -0.5
    x16 += einsum(x2, (0, 1, 2), x0, (0, 3, 1, 4), (3, 4, 2))
    x16 += einsum(x3, (0, 1, 2), x1, (3, 0, 1, 4), (3, 4, 2)) * -0.5
    x16 += einsum(x7, (0, 1, 2), l2.aaaa, (3, 1, 4, 0), (4, 3, 2))
    x16 += einsum(x11, (0, 1, 2), l2.abab, (3, 1, 4, 0), (4, 3, 2)) * 0.5
    x16 += einsum(x14, (0, 1), v.aaa.xov, (2, 3, 1), (3, 0, 2)) * -1.0
    del x14
    x16 += einsum(l1.aa, (0, 1), x15, (1, 2, 3), (2, 0, 3)) * -0.5
    l1new_aa += einsum(v.aaa.xvv, (0, 1, 2), x16, (3, 2, 0), (1, 3)) * 2.0
    del x16
    x17 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x17 += einsum(v.aaa.xov, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    x18 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x18 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xov, (0, 3, 4), (1, 2, 3, 4))
    x19 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x19 += einsum(v.aaa.xov, (0, 1, 2), x2, (3, 4, 0), (3, 1, 4, 2))
    x20 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x20 += einsum(x18, (0, 1, 2, 3), (1, 0, 2, 3))
    x20 += einsum(x18, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x20 += einsum(x19, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x20 += einsum(x19, (0, 1, 2, 3), (2, 0, 1, 3))
    x21 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x21 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0))
    x21 += einsum(x8, (0, 1, 2), (0, 1, 2))
    x22 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x22 += einsum(v.abb.xoo, (0, 1, 2), (1, 2, 0))
    x22 += einsum(x3, (0, 1, 2), (0, 1, 2))
    x23 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x23 += einsum(v.aaa.xov, (0, 1, 2), x22, (3, 4, 0), (1, 4, 3, 2))
    x24 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x24 += einsum(v.aaa.xoo, (0, 1, 2), (1, 2, 0))
    x24 += einsum(x2, (0, 1, 2), (1, 0, 2))
    x25 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x25 += einsum(v.abb.xov, (0, 1, 2), x24, (3, 4, 0), (3, 4, 1, 2))
    l2new_abab = np.zeros((nvir[0], nvir[1], nocc[0], nocc[1]), dtype=types[float])
    l2new_abab += einsum(x1, (0, 1, 2, 3), x25, (4, 0, 2, 5), (3, 5, 4, 1))
    l2new_abab += einsum(l1.aa, (0, 1), x25, (2, 1, 3, 4), (0, 4, 2, 3)) * -1.0
    x26 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x26 += einsum(v.aaa.xov, (0, 1, 2), x2, (1, 3, 0), (3, 2))
    x27 = np.zeros((naux,), dtype=types[float])
    x27 += einsum(t1.aa, (0, 1), v.aaa.xov, (2, 0, 1), (2,))
    x28 = np.zeros((naux,), dtype=types[float])
    x28 += einsum(t1.bb, (0, 1), v.abb.xov, (2, 0, 1), (2,))
    x29 = np.zeros((naux,), dtype=types[float])
    x29 += einsum(x27, (0,), (0,))
    del x27
    x29 += einsum(x28, (0,), (0,))
    del x28
    x30 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x30 += einsum(x29, (0,), v.aaa.xov, (0, 1, 2), (1, 2))
    x31 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x31 += einsum(f.aa.ov, (0, 1), (0, 1))
    x31 += einsum(x26, (0, 1), (0, 1)) * -1.0
    x31 += einsum(x30, (0, 1), (0, 1))
    l2new_abab += einsum(l1.bb, (0, 1), x31, (2, 3), (3, 0, 2, 1))
    x32 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x32 += einsum(v.aaa.xov, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 3, 2, 4))
    x33 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x33 += einsum(t2.abab, (0, 1, 2, 3), x32, (4, 5, 2, 3), (0, 4, 1, 5))
    x34 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x34 += einsum(x33, (0, 1, 2, 3), (0, 1, 3, 2))
    x34 += einsum(x22, (0, 1, 2), x24, (3, 4, 2), (4, 3, 1, 0))
    x35 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x35 += einsum(t2.abab, (0, 1, 2, 3), x17, (4, 2, 3, 5), (4, 0, 1, 5))
    x35 += einsum(t2.abab, (0, 1, 2, 3), x20, (0, 4, 5, 2), (5, 4, 1, 3)) * -1.0
    del x20
    x35 += einsum(x15, (0, 1, 2), x21, (3, 4, 2), (1, 0, 3, 4))
    x35 += einsum(t2.abab, (0, 1, 2, 3), x23, (4, 1, 5, 2), (4, 0, 5, 3)) * -1.0
    del x23
    x35 += einsum(t2.bbbb, (0, 1, 2, 3), x25, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    x35 += einsum(x31, (0, 1), t2.abab, (2, 3, 1, 4), (0, 2, 3, 4))
    x35 += einsum(t1.bb, (0, 1), x34, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x34
    l1new_aa += einsum(l2.abab, (0, 1, 2, 3), x35, (4, 2, 3, 1), (0, 4)) * -1.0
    del x35
    x36 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x36 += einsum(v.aaa.xov, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 2, 3, 4))
    x37 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x37 += einsum(x18, (0, 1, 2, 3), (1, 0, 2, 3))
    x37 += einsum(x18, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    x37 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x37 += einsum(x19, (0, 1, 2, 3), (0, 2, 1, 3))
    x38 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x38 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xoo, (0, 3, 4), (1, 3, 4, 2))
    x39 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x39 += einsum(v.aaa.xov, (0, 1, 2), v.aaa.xov, (0, 3, 4), (1, 3, 2, 4))
    x40 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x40 += einsum(x2, (0, 1, 2), x2, (3, 4, 2), (0, 3, 1, 4))
    x41 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x41 += einsum(v.aaa.xoo, (0, 1, 2), x2, (3, 4, 0), (3, 1, 2, 4))
    x42 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x42 += einsum(x38, (0, 1, 2, 3), (3, 2, 1, 0))
    x42 += einsum(t2.aaaa, (0, 1, 2, 3), x39, (4, 5, 3, 2), (5, 4, 0, 1)) * -1.0
    x42 += einsum(x40, (0, 1, 2, 3), (2, 3, 1, 0))
    x42 += einsum(x41, (0, 1, 2, 3), (1, 3, 0, 2))
    x42 += einsum(x41, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    x43 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x43 += einsum(v.aaa.xoo, (0, 1, 2), x4, (3, 4, 0), (1, 2, 3, 4)) * -1.0
    x43 += einsum(t2.aaaa, (0, 1, 2, 3), x36, (4, 2, 5, 3), (4, 0, 1, 5)) * -1.0
    x43 += einsum(x2, (0, 1, 2), x4, (3, 4, 2), (1, 3, 0, 4))
    x43 += einsum(t2.aaaa, (0, 1, 2, 3), x37, (4, 1, 5, 3), (5, 0, 4, 2)) * -2.0
    del x37
    x43 += einsum(t2.abab, (0, 1, 2, 3), x25, (4, 5, 1, 3), (4, 0, 5, 2))
    del x25
    x43 += einsum(v.aaa.xov, (0, 1, 2), x15, (3, 4, 0), (4, 1, 3, 2))
    x43 += einsum(x31, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4))
    x43 += einsum(t1.aa, (0, 1), x42, (0, 2, 3, 4), (2, 4, 3, 1)) * -1.0
    del x42
    l1new_aa += einsum(l2.aaaa, (0, 1, 2, 3), x43, (4, 2, 3, 1), (0, 4)) * 2.0
    del x43
    x44 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x44 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), (2, 4, 1, 5))
    x45 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x45 += einsum(t1.aa, (0, 1), l2.abab, (1, 2, 3, 4), (3, 0, 4, 2))
    l2new_abab += einsum(x17, (0, 1, 2, 3), x45, (4, 0, 5, 2), (1, 3, 4, 5)) * -1.0
    del x17
    l2new_abab += einsum(x31, (0, 1), x45, (2, 0, 3, 4), (1, 4, 2, 3)) * -1.0
    x46 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x46 += einsum(t1.aa, (0, 1), x0, (2, 3, 4, 1), (3, 2, 4, 0))
    l2new_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    l2new_aaaa += einsum(x39, (0, 1, 2, 3), x46, (4, 5, 0, 1), (3, 2, 5, 4)) * 2.0
    x47 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x47 += einsum(t1.aa, (0, 1), x1, (2, 3, 4, 1), (2, 0, 3, 4))
    l2new_abab += einsum(x32, (0, 1, 2, 3), x47, (4, 0, 5, 1), (2, 3, 4, 5))
    del x32
    x48 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x48 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), (2, 3, 4, 5))
    x49 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x49 += einsum(t2.aaaa, (0, 1, 2, 3), x0, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x50 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x50 += einsum(t2.abab, (0, 1, 2, 3), x45, (4, 5, 1, 3), (4, 5, 0, 2))
    x51 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x51 += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x49
    x51 += einsum(x50, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x50
    x52 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x52 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 0), (1, 2, 3, 4))
    x52 += einsum(x0, (0, 1, 2, 3), (1, 0, 2, 3))
    x52 += einsum(t1.aa, (0, 1), x48, (2, 0, 3, 4), (2, 3, 4, 1))
    x52 += einsum(x51, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x52 += einsum(x51, (0, 1, 2, 3), (0, 2, 1, 3))
    del x51
    x53 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x53 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 5, 1), (2, 4, 0, 5))
    x53 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 4, 0, 5)) * 0.25
    x54 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x54 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), (2, 4, 3, 5))
    x55 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x55 += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), (1, 2, 3, 4))
    x55 += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    x55 += einsum(t2.abab, (0, 1, 2, 3), x0, (4, 0, 5, 2), (4, 5, 1, 3)) * -2.0
    x55 += einsum(t2.bbbb, (0, 1, 2, 3), x45, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    x55 += einsum(t2.abab, (0, 1, 2, 3), x1, (4, 1, 5, 2), (4, 0, 5, 3)) * -1.0
    x55 += einsum(t1.bb, (0, 1), x54, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    x56 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x56 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 0), (2, 3))
    x57 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x57 += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), (2, 3))
    x58 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x58 += einsum(t2.aaaa, (0, 1, 2, 3), x0, (0, 1, 4, 3), (4, 2)) * -1.0
    x59 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x59 += einsum(t2.abab, (0, 1, 2, 3), x45, (0, 4, 1, 3), (4, 2))
    x60 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x60 += einsum(l1.aa, (0, 1), t1.aa, (2, 0), (1, 2))
    l1new_aa += einsum(x30, (0, 1), x60, (2, 0), (1, 2)) * -1.0
    x61 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x61 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    x62 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x62 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    x63 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x63 += einsum(x60, (0, 1), (0, 1))
    x63 += einsum(x61, (0, 1), (0, 1)) * 2.0
    x63 += einsum(x62, (0, 1), (0, 1))
    l1new_aa += einsum(f.aa.ov, (0, 1), x63, (2, 0), (1, 2)) * -1.0
    x64 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x64 += einsum(t1.aa, (0, 1), x63, (0, 2), (2, 1))
    x65 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x65 += einsum(t1.aa, (0, 1), (0, 1)) * -1.0
    x65 += einsum(x56, (0, 1), (0, 1)) * -2.0
    x65 += einsum(x57, (0, 1), (0, 1)) * -1.0
    x65 += einsum(x58, (0, 1), (0, 1)) * 2.0
    x65 += einsum(x59, (0, 1), (0, 1))
    x65 += einsum(x64, (0, 1), (0, 1))
    x66 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x66 += einsum(x48, (0, 1, 2, 3), (1, 0, 3, 2))
    del x48
    x66 += einsum(x46, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x67 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x67 += einsum(x54, (0, 1, 2, 3), (0, 1, 2, 3))
    x67 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3))
    x68 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x68 += einsum(l1.aa, (0, 1), v.aaa.xvv, (2, 3, 0), (1, 3, 2))
    x69 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x69 += einsum(x68, (0, 1, 2), (0, 1, 2))
    x69 += einsum(x63, (0, 1), v.aaa.xov, (2, 1, 3), (0, 3, 2)) * -1.0
    del x63
    x70 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x70 += einsum(x60, (0, 1), (0, 1)) * 0.5
    x70 += einsum(x61, (0, 1), (0, 1))
    x70 += einsum(x62, (0, 1), (0, 1)) * 0.5
    x71 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x71 += einsum(x61, (0, 1), (0, 1))
    x71 += einsum(x62, (0, 1), (0, 1)) * 0.5
    x72 = np.zeros((nocc[0], nocc[0], naux), dtype=types[float])
    x72 += einsum(x4, (0, 1, 2), x0, (3, 0, 4, 1), (4, 3, 2)) * -2.0
    x72 += einsum(v.abb.xvv, (0, 1, 2), x44, (3, 4, 2, 1), (4, 3, 0))
    del x44
    x72 += einsum(x8, (0, 1, 2), x45, (3, 4, 0, 1), (4, 3, 2))
    x72 += einsum(x2, (0, 1, 2), x46, (0, 3, 1, 4), (4, 3, 2)) * -2.0
    del x46
    x72 += einsum(x3, (0, 1, 2), x47, (3, 4, 0, 1), (4, 3, 2)) * -1.0
    x72 += einsum(v.aaa.xov, (0, 1, 2), x52, (3, 1, 4, 2), (4, 3, 0)) * 2.0
    del x52
    x72 += einsum(v.aaa.xvv, (0, 1, 2), x53, (3, 4, 1, 2), (4, 3, 0)) * 4.0
    del x53
    x72 += einsum(v.abb.xov, (0, 1, 2), x55, (3, 4, 1, 2), (4, 3, 0))
    del x55
    x72 += einsum(x65, (0, 1), v.aaa.xov, (2, 3, 1), (0, 3, 2)) * -1.0
    del x65
    x72 += einsum(v.aaa.xoo, (0, 1, 2), x66, (1, 3, 2, 4), (4, 3, 0)) * -2.0
    del x66
    x72 += einsum(v.abb.xoo, (0, 1, 2), x67, (3, 4, 1, 2), (4, 3, 0)) * -1.0
    del x67
    x72 += einsum(t1.aa, (0, 1), x69, (2, 1, 3), (0, 2, 3))
    del x69
    x72 += einsum(x70, (0, 1), v.aaa.xoo, (2, 3, 0), (1, 3, 2)) * -2.0
    del x70
    x72 += einsum(x29, (0,), x71, (1, 2), (2, 1, 0)) * 2.0
    del x71
    l1new_aa += einsum(v.aaa.xov, (0, 1, 2), x72, (1, 3, 0), (2, 3)) * -1.0
    del x72
    x73 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x73 += einsum(l1.aa, (0, 1), t1.aa, (1, 2), (0, 2))
    x74 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x74 += einsum(x73, (0, 1), (0, 1))
    del x73
    x74 += einsum(x12, (0, 1), (1, 0)) * 2.0
    del x12
    x74 += einsum(x13, (0, 1), (1, 0))
    del x13
    x75 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x75 += einsum(l1.bb, (0, 1), t1.bb, (1, 2), (0, 2))
    x76 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x76 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    x77 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x77 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4))
    x78 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x78 += einsum(x75, (0, 1), (0, 1)) * 0.5
    del x75
    x78 += einsum(x76, (0, 1), (0, 1)) * 0.5
    x78 += einsum(x77, (0, 1), (0, 1))
    x79 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x79 += einsum(l1.aa, (0, 1), (1, 0)) * -1.0
    x79 += einsum(t1.aa, (0, 1), (0, 1)) * -1.0
    x79 += einsum(x56, (0, 1), (0, 1)) * -2.0
    del x56
    x79 += einsum(x57, (0, 1), (0, 1)) * -1.0
    del x57
    x79 += einsum(x58, (0, 1), (0, 1)) * 2.0
    del x58
    x79 += einsum(x59, (0, 1), (0, 1))
    del x59
    x79 += einsum(x64, (0, 1), (0, 1))
    del x64
    x80 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x80 += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), (2, 3))
    x81 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x81 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 0), (2, 3))
    x82 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x82 += einsum(t2.abab, (0, 1, 2, 3), x1, (0, 1, 4, 2), (4, 3))
    x83 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x83 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    x84 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x84 += einsum(t2.bbbb, (0, 1, 2, 3), x83, (0, 1, 4, 3), (4, 2)) * -1.0
    x85 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x85 += einsum(l1.bb, (0, 1), t1.bb, (2, 0), (1, 2))
    x86 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x86 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    x87 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x87 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    x88 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x88 += einsum(x85, (0, 1), (0, 1)) * 0.5
    x88 += einsum(x86, (0, 1), (0, 1)) * 0.5
    x88 += einsum(x87, (0, 1), (0, 1))
    x89 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x89 += einsum(t1.bb, (0, 1), x88, (0, 2), (2, 1)) * 2.0
    x90 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x90 += einsum(l1.bb, (0, 1), (1, 0)) * -1.0
    x90 += einsum(t1.bb, (0, 1), (0, 1)) * -1.0
    x90 += einsum(x80, (0, 1), (0, 1)) * -1.0
    x90 += einsum(x81, (0, 1), (0, 1)) * -2.0
    x90 += einsum(x82, (0, 1), (0, 1))
    x90 += einsum(x84, (0, 1), (0, 1)) * 2.0
    x90 += einsum(x89, (0, 1), (0, 1))
    x91 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x91 += einsum(x60, (0, 1), (0, 1)) * 0.5
    x91 += einsum(x61, (0, 1), (0, 1))
    del x61
    x91 += einsum(x62, (0, 1), (1, 0)) * 0.5
    del x62
    x92 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x92 += einsum(x85, (0, 1), (0, 1))
    x92 += einsum(x86, (0, 1), (1, 0))
    x92 += einsum(x87, (0, 1), (0, 1)) * 2.0
    x93 = np.zeros((naux,), dtype=types[float])
    x93 += einsum(x74, (0, 1), v.aaa.xvv, (2, 0, 1), (2,))
    x93 += einsum(x78, (0, 1), v.abb.xvv, (2, 1, 0), (2,)) * 2.0
    x93 += einsum(x79, (0, 1), v.aaa.xov, (2, 0, 1), (2,)) * -1.0
    x93 += einsum(x90, (0, 1), v.abb.xov, (2, 0, 1), (2,)) * -1.0
    x93 += einsum(x91, (0, 1), v.aaa.xoo, (2, 1, 0), (2,)) * -2.0
    x93 += einsum(x92, (0, 1), v.abb.xoo, (2, 1, 0), (2,)) * -1.0
    l1new_aa += einsum(x93, (0,), v.aaa.xov, (0, 1, 2), (2, 1))
    del x93
    x94 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x94 += einsum(x29, (0,), v.aaa.xvv, (0, 1, 2), (1, 2))
    x95 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x95 += einsum(f.aa.vv, (0, 1), (0, 1))
    x95 += einsum(x94, (0, 1), (1, 0))
    l1new_aa += einsum(l1.aa, (0, 1), x95, (0, 2), (2, 1))
    del x95
    x96 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x96 += einsum(t1.aa, (0, 1), v.aaa.xoo, (2, 3, 0), (3, 1, 2)) * -1.0
    x96 += einsum(x5, (0, 1, 2), (0, 1, 2)) * 2.0
    del x5
    x96 += einsum(x6, (0, 1, 2), (0, 1, 2))
    del x6
    x97 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x97 += einsum(f.aa.oo, (0, 1), (0, 1)) * 0.5
    x97 += einsum(v.aaa.xov, (0, 1, 2), x96, (3, 2, 0), (3, 1)) * 0.5
    del x96
    x97 += einsum(x29, (0,), v.aaa.xoo, (0, 1, 2), (1, 2)) * 0.5
    x97 += einsum(t1.aa, (0, 1), x31, (2, 1), (0, 2)) * 0.5
    l1new_aa += einsum(l1.aa, (0, 1), x97, (1, 2), (0, 2)) * -2.0
    del x97
    x98 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x98 += einsum(x76, (0, 1), (0, 1))
    del x76
    x98 += einsum(x77, (0, 1), (0, 1)) * 2.0
    del x77
    x99 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x99 += einsum(v.aaa.xoo, (0, 1, 2), x45, (1, 2, 3, 4), (3, 4, 0)) * -0.5
    x99 += einsum(v.abb.xoo, (0, 1, 2), x83, (1, 3, 2, 4), (3, 4, 0))
    x99 += einsum(x2, (0, 1, 2), x45, (0, 1, 3, 4), (3, 4, 2)) * -0.5
    x99 += einsum(x3, (0, 1, 2), x83, (3, 0, 1, 4), (3, 4, 2)) * -1.0
    x99 += einsum(x7, (0, 1, 2), l2.abab, (1, 3, 0, 4), (4, 3, 2)) * 0.5
    del x7
    x99 += einsum(x11, (0, 1, 2), l2.bbbb, (3, 1, 4, 0), (4, 3, 2))
    del x11
    x99 += einsum(x98, (0, 1), v.abb.xov, (2, 3, 1), (3, 0, 2)) * -0.5
    del x98
    x99 += einsum(l1.bb, (0, 1), x22, (1, 2, 3), (2, 0, 3)) * -0.5
    l1new_bb += einsum(v.abb.xvv, (0, 1, 2), x99, (3, 2, 0), (1, 3)) * 2.0
    del x99
    x100 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x100 += einsum(v.abb.xov, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 3, 4, 2))
    l2new_abab += einsum(x1, (0, 1, 2, 3), x100, (2, 4, 3, 5), (4, 5, 0, 1)) * -1.0
    x101 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x101 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 2, 3, 4))
    x102 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x102 += einsum(v.abb.xov, (0, 1, 2), x3, (3, 4, 0), (3, 1, 4, 2))
    x103 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x103 += einsum(x101, (0, 1, 2, 3), (1, 0, 2, 3))
    x103 += einsum(x101, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x103 += einsum(x102, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x103 += einsum(x102, (0, 1, 2, 3), (2, 0, 1, 3))
    x104 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x104 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x104 += einsum(x4, (0, 1, 2), (0, 1, 2))
    x105 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x105 += einsum(v.abb.xoo, (0, 1, 2), (1, 2, 0))
    x105 += einsum(x3, (0, 1, 2), (1, 0, 2))
    x106 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x106 += einsum(v.aaa.xov, (0, 1, 2), x105, (3, 4, 0), (1, 3, 4, 2))
    l2new_abab += einsum(x106, (0, 1, 2, 3), x45, (4, 0, 2, 5), (3, 5, 4, 1))
    l2new_abab += einsum(l1.bb, (0, 1), x106, (2, 3, 1, 4), (4, 0, 2, 3)) * -1.0
    x107 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x107 += einsum(v.abb.xov, (0, 1, 2), x15, (3, 4, 0), (4, 3, 1, 2))
    x108 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x108 += einsum(v.abb.xov, (0, 1, 2), x3, (1, 3, 0), (3, 2))
    x109 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x109 += einsum(x29, (0,), v.abb.xov, (0, 1, 2), (1, 2))
    l1new_bb += einsum(x109, (0, 1), x85, (2, 0), (1, 2)) * -1.0
    x110 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x110 += einsum(f.bb.ov, (0, 1), (0, 1))
    x110 += einsum(x108, (0, 1), (0, 1)) * -1.0
    x110 += einsum(x109, (0, 1), (0, 1))
    l2new_abab += einsum(x110, (0, 1), x1, (2, 3, 0, 4), (4, 1, 2, 3)) * -1.0
    l2new_abab += einsum(l1.aa, (0, 1), x110, (2, 3), (0, 3, 1, 2))
    x111 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x111 += einsum(x33, (0, 1, 2, 3), (1, 0, 2, 3))
    del x33
    x111 += einsum(x105, (0, 1, 2), x15, (3, 4, 2), (4, 3, 1, 0))
    del x15
    x112 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x112 += einsum(t2.abab, (0, 1, 2, 3), x100, (4, 2, 5, 3), (0, 4, 1, 5))
    del x100
    x112 += einsum(t2.abab, (0, 1, 2, 3), x103, (1, 4, 5, 3), (0, 5, 4, 2)) * -1.0
    del x103
    x112 += einsum(x104, (0, 1, 2), x22, (3, 4, 2), (0, 4, 3, 1))
    x112 += einsum(t2.aaaa, (0, 1, 2, 3), x106, (1, 4, 5, 3), (0, 4, 5, 2)) * 2.0
    x112 += einsum(t2.abab, (0, 1, 2, 3), x107, (0, 4, 5, 3), (4, 5, 1, 2)) * -1.0
    del x107
    x112 += einsum(x110, (0, 1), t2.abab, (2, 3, 4, 1), (2, 0, 3, 4))
    x112 += einsum(t1.aa, (0, 1), x111, (0, 2, 3, 4), (2, 4, 3, 1)) * -1.0
    del x111
    l1new_bb += einsum(l2.abab, (0, 1, 2, 3), x112, (2, 4, 3, 0), (1, 4)) * -1.0
    del x112
    x113 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x113 += einsum(v.abb.xov, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    x114 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x114 += einsum(x101, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x114 += einsum(x101, (0, 1, 2, 3), (1, 2, 0, 3))
    x114 += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3))
    x114 += einsum(x102, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x115 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x115 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xoo, (0, 3, 4), (1, 3, 4, 2))
    x116 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x116 += einsum(v.abb.xov, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 3, 2, 4))
    x117 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x117 += einsum(x3, (0, 1, 2), x3, (3, 4, 2), (0, 3, 1, 4))
    x118 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x118 += einsum(v.abb.xoo, (0, 1, 2), x3, (3, 4, 0), (3, 1, 2, 4))
    x119 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x119 += einsum(x115, (0, 1, 2, 3), (3, 2, 1, 0))
    x119 += einsum(t2.bbbb, (0, 1, 2, 3), x116, (4, 5, 2, 3), (5, 4, 0, 1))
    x119 += einsum(x117, (0, 1, 2, 3), (2, 3, 1, 0))
    x119 += einsum(x118, (0, 1, 2, 3), (1, 3, 0, 2))
    x119 += einsum(x118, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    x120 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x120 += einsum(v.abb.xoo, (0, 1, 2), x8, (3, 4, 0), (1, 2, 3, 4)) * -1.0
    x120 += einsum(t2.bbbb, (0, 1, 2, 3), x113, (4, 2, 3, 5), (4, 0, 1, 5)) * -1.0
    x120 += einsum(x3, (0, 1, 2), x8, (3, 4, 2), (1, 3, 0, 4))
    x120 += einsum(t2.bbbb, (0, 1, 2, 3), x114, (4, 5, 1, 3), (5, 0, 4, 2)) * -2.0
    del x114
    x120 += einsum(t2.abab, (0, 1, 2, 3), x106, (0, 4, 5, 2), (4, 1, 5, 3))
    del x106
    x120 += einsum(v.abb.xov, (0, 1, 2), x22, (3, 4, 0), (4, 1, 3, 2))
    del x22
    x120 += einsum(x110, (0, 1), t2.bbbb, (2, 3, 4, 1), (0, 2, 3, 4))
    x120 += einsum(t1.bb, (0, 1), x119, (0, 2, 3, 4), (2, 4, 3, 1)) * -1.0
    del x119
    l1new_bb += einsum(l2.bbbb, (0, 1, 2, 3), x120, (4, 2, 3, 1), (0, 4)) * 2.0
    del x120
    x121 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x121 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), (3, 4, 0, 5))
    x122 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x122 += einsum(t1.bb, (0, 1), x83, (2, 3, 4, 1), (2, 3, 0, 4))
    l2new_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    l2new_bbbb += einsum(x116, (0, 1, 2, 3), x122, (4, 5, 0, 1), (3, 2, 5, 4)) * 2.0
    x123 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x123 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), (2, 3, 4, 5))
    x124 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x124 += einsum(t2.abab, (0, 1, 2, 3), x1, (0, 4, 5, 2), (4, 5, 1, 3))
    x125 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x125 += einsum(t2.bbbb, (0, 1, 2, 3), x83, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x126 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x126 += einsum(x124, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x124
    x126 += einsum(x125, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x125
    x127 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x127 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 0), (1, 2, 3, 4))
    x127 += einsum(x83, (0, 1, 2, 3), (1, 0, 2, 3))
    x127 += einsum(t1.bb, (0, 1), x123, (2, 0, 3, 4), (2, 3, 4, 1))
    x127 += einsum(x126, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x127 += einsum(x126, (0, 1, 2, 3), (0, 2, 1, 3))
    del x126
    x128 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x128 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), (3, 4, 1, 5))
    x128 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (2, 4, 0, 5)) * 4.0
    x129 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x129 += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), (2, 1, 3, 4))
    x129 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    x129 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (1, 4, 5, 3), (0, 4, 5, 2)) * 2.0
    x129 += einsum(t2.abab, (0, 1, 2, 3), x45, (0, 4, 5, 3), (4, 5, 1, 2)) * -1.0
    x129 += einsum(t1.aa, (0, 1), x54, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    x129 += einsum(t2.abab, (0, 1, 2, 3), x83, (4, 1, 5, 3), (0, 4, 5, 2)) * -2.0
    x130 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x130 += einsum(t1.bb, (0, 1), (0, 1)) * -1.0
    x130 += einsum(x80, (0, 1), (0, 1)) * -1.0
    del x80
    x130 += einsum(x81, (0, 1), (0, 1)) * -2.0
    del x81
    x130 += einsum(x82, (0, 1), (0, 1))
    del x82
    x130 += einsum(x84, (0, 1), (0, 1)) * 2.0
    del x84
    x130 += einsum(x89, (0, 1), (0, 1))
    del x89
    x131 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x131 += einsum(x54, (0, 1, 2, 3), (0, 1, 2, 3))
    del x54
    x131 += einsum(x47, (0, 1, 2, 3), (1, 0, 2, 3))
    x132 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x132 += einsum(x123, (0, 1, 2, 3), (1, 0, 3, 2))
    del x123
    x132 += einsum(x122, (0, 1, 2, 3), (2, 1, 0, 3))
    x133 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x133 += einsum(l1.bb, (0, 1), v.abb.xvv, (2, 3, 0), (1, 3, 2))
    x134 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x134 += einsum(x133, (0, 1, 2), (0, 1, 2))
    x134 += einsum(x88, (0, 1), v.abb.xov, (2, 1, 3), (0, 3, 2)) * -2.0
    del x88
    x135 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x135 += einsum(x85, (0, 1), (0, 1))
    x135 += einsum(x86, (0, 1), (0, 1))
    x135 += einsum(x87, (0, 1), (0, 1)) * 2.0
    l1new_bb += einsum(f.bb.ov, (0, 1), x135, (2, 0), (1, 2)) * -1.0
    x136 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x136 += einsum(x86, (0, 1), (0, 1)) * 0.5
    del x86
    x136 += einsum(x87, (0, 1), (0, 1))
    del x87
    x137 = np.zeros((nocc[1], nocc[1], naux), dtype=types[float])
    x137 += einsum(v.aaa.xvv, (0, 1, 2), x121, (3, 4, 1, 2), (4, 3, 0)) * 0.25
    del x121
    x137 += einsum(x4, (0, 1, 2), x1, (0, 3, 4, 1), (4, 3, 2)) * 0.25
    x137 += einsum(x8, (0, 1, 2), x83, (3, 0, 4, 1), (4, 3, 2)) * -0.5
    x137 += einsum(x2, (0, 1, 2), x47, (0, 1, 3, 4), (4, 3, 2)) * -0.25
    del x47
    x137 += einsum(x3, (0, 1, 2), x122, (0, 3, 1, 4), (4, 3, 2)) * -0.5
    del x122
    x137 += einsum(v.abb.xov, (0, 1, 2), x127, (3, 1, 4, 2), (4, 3, 0)) * 0.5
    del x127
    x137 += einsum(v.abb.xvv, (0, 1, 2), x128, (3, 4, 1, 2), (4, 3, 0)) * 0.25
    del x128
    x137 += einsum(v.aaa.xov, (0, 1, 2), x129, (1, 3, 4, 2), (4, 3, 0)) * 0.25
    del x129
    x137 += einsum(x130, (0, 1), v.abb.xov, (2, 3, 1), (0, 3, 2)) * -0.25
    del x130
    x137 += einsum(v.aaa.xoo, (0, 1, 2), x131, (1, 2, 3, 4), (4, 3, 0)) * -0.25
    del x131
    x137 += einsum(v.abb.xoo, (0, 1, 2), x132, (1, 3, 2, 4), (4, 3, 0)) * -0.5
    del x132
    x137 += einsum(t1.bb, (0, 1), x134, (2, 1, 3), (0, 2, 3)) * 0.25
    del x134
    x137 += einsum(x135, (0, 1), v.abb.xoo, (2, 3, 0), (1, 3, 2)) * -0.25
    del x135
    x137 += einsum(x29, (0,), x136, (1, 2), (2, 1, 0)) * 0.5
    del x136
    l1new_bb += einsum(v.abb.xov, (0, 1, 2), x137, (1, 3, 0), (2, 3)) * -4.0
    del x137
    x138 = np.zeros((naux,), dtype=types[float])
    x138 += einsum(x74, (0, 1), v.aaa.xvv, (2, 0, 1), (2,)) * 0.5
    del x74
    x138 += einsum(x78, (0, 1), v.abb.xvv, (2, 1, 0), (2,))
    del x78
    x138 += einsum(x79, (0, 1), v.aaa.xov, (2, 0, 1), (2,)) * -0.5
    del x79
    x138 += einsum(x90, (0, 1), v.abb.xov, (2, 0, 1), (2,)) * -0.5
    del x90
    x138 += einsum(x91, (0, 1), v.aaa.xoo, (2, 1, 0), (2,)) * -1.0
    del x91
    x138 += einsum(x92, (0, 1), v.abb.xoo, (2, 1, 0), (2,)) * -0.5
    del x92
    l1new_bb += einsum(x138, (0,), v.abb.xov, (0, 1, 2), (2, 1)) * 2.0
    del x138
    x139 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x139 += einsum(x29, (0,), v.abb.xvv, (0, 1, 2), (1, 2))
    x140 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x140 += einsum(f.bb.vv, (0, 1), (0, 1))
    x140 += einsum(x139, (0, 1), (1, 0))
    l1new_bb += einsum(l1.bb, (0, 1), x140, (0, 2), (2, 1))
    del x140
    x141 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x141 += einsum(t1.bb, (0, 1), v.abb.xoo, (2, 3, 0), (3, 1, 2)) * -0.5
    x141 += einsum(x9, (0, 1, 2), (0, 1, 2)) * 0.5
    del x9
    x141 += einsum(x10, (0, 1, 2), (0, 1, 2))
    del x10
    x142 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x142 += einsum(x29, (0,), v.abb.xoo, (0, 1, 2), (1, 2))
    x143 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x143 += einsum(t1.bb, (0, 1), x110, (2, 1), (2, 0))
    del x110
    x144 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x144 += einsum(f.bb.oo, (0, 1), (0, 1))
    x144 += einsum(v.abb.xov, (0, 1, 2), x141, (3, 2, 0), (3, 1)) * 2.0
    del x141
    x144 += einsum(x142, (0, 1), (1, 0))
    x144 += einsum(x143, (0, 1), (1, 0))
    l1new_bb += einsum(l1.bb, (0, 1), x144, (1, 2), (0, 2)) * -1.0
    del x144
    x145 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x145 += einsum(v.aaa.xvv, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 3, 4, 2))
    l2new_aaaa += einsum(l2.aaaa, (0, 1, 2, 3), x145, (4, 5, 0, 1), (4, 5, 2, 3)) * -2.0
    del x145
    x146 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x146 += einsum(v.abb.xoo, (0, 1, 2), v.aaa.xov, (0, 3, 4), (3, 1, 2, 4))
    l2new_abab += einsum(x146, (0, 1, 2, 3), x83, (4, 1, 2, 5), (3, 5, 0, 4)) * -2.0
    x147 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x147 += einsum(x1, (0, 1, 2, 3), x146, (4, 1, 2, 5), (0, 4, 3, 5))
    del x146
    x148 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x148 += einsum(v.aaa.xov, (0, 1, 2), x3, (3, 4, 0), (1, 3, 4, 2))
    l2new_abab += einsum(x148, (0, 1, 2, 3), x83, (4, 1, 2, 5), (3, 5, 0, 4)) * -2.0
    x149 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x149 += einsum(x1, (0, 1, 2, 3), x148, (4, 1, 2, 5), (0, 4, 3, 5))
    del x148
    x150 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x150 += einsum(x104, (0, 1, 2), l2.aaaa, (3, 1, 4, 0), (4, 3, 2))
    x151 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x151 += einsum(x21, (0, 1, 2), l2.abab, (3, 1, 4, 0), (4, 3, 2)) * 0.5
    x152 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x152 += einsum(x150, (0, 1, 2), (0, 1, 2))
    x152 += einsum(x151, (0, 1, 2), (0, 1, 2))
    x153 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x153 += einsum(v.aaa.xov, (0, 1, 2), x152, (3, 4, 0), (3, 1, 4, 2)) * 2.0
    del x152
    x154 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x154 += einsum(v.aaa.xvv, (0, 1, 2), x24, (3, 4, 0), (3, 4, 1, 2))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x154, (4, 2, 5, 0), (5, 1, 4, 3)) * -1.0
    x155 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x155 += einsum(l2.aaaa, (0, 1, 2, 3), x154, (4, 3, 5, 1), (4, 2, 5, 0)) * 2.0
    del x154
    x156 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x156 += einsum(x60, (0, 1), v.aaa.xov, (2, 1, 3), (0, 3, 2))
    del x60
    x157 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x157 += einsum(x68, (0, 1, 2), (0, 1, 2))
    x157 += einsum(x156, (0, 1, 2), (0, 1, 2)) * -1.0
    x158 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x158 += einsum(v.aaa.xov, (0, 1, 2), x157, (3, 4, 0), (3, 1, 4, 2))
    del x157
    x159 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x159 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    x159 += einsum(x19, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l2new_abab += einsum(x159, (0, 1, 2, 3), x45, (0, 2, 4, 5), (3, 5, 1, 4)) * -1.0
    x160 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x160 += einsum(x0, (0, 1, 2, 3), x159, (0, 4, 2, 5), (4, 1, 5, 3)) * 2.0
    del x159
    x161 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x161 += einsum(x18, (0, 1, 2, 3), (1, 0, 2, 3))
    x161 += einsum(x18, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    x162 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x162 += einsum(x0, (0, 1, 2, 3), x161, (0, 2, 4, 5), (4, 1, 5, 3)) * 2.0
    del x161
    x163 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x163 += einsum(x18, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x163 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    del x19
    x164 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x164 += einsum(l1.aa, (0, 1), x163, (1, 2, 3, 4), (2, 3, 4, 0))
    del x163
    x165 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x165 += einsum(x26, (0, 1), (0, 1)) * -1.0
    del x26
    x165 += einsum(x30, (0, 1), (0, 1))
    del x30
    x166 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x166 += einsum(f.aa.ov, (0, 1), l1.aa, (2, 3), (0, 3, 1, 2))
    x166 += einsum(x147, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x147
    x166 += einsum(x149, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x149
    x166 += einsum(x153, (0, 1, 2, 3), (0, 1, 2, 3))
    del x153
    x166 += einsum(x155, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x155
    x166 += einsum(x158, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x158
    x166 += einsum(x160, (0, 1, 2, 3), (1, 0, 3, 2))
    del x160
    x166 += einsum(x162, (0, 1, 2, 3), (1, 0, 3, 2))
    del x162
    x166 += einsum(x164, (0, 1, 2, 3), (0, 1, 3, 2))
    del x164
    x166 += einsum(l1.aa, (0, 1), x165, (2, 3), (1, 2, 0, 3))
    l2new_aaaa += einsum(x166, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_aaaa += einsum(x166, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_aaaa += einsum(x166, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new_aaaa += einsum(x166, (0, 1, 2, 3), (3, 2, 1, 0))
    del x166
    x167 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x167 += einsum(f.aa.vv, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    x168 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x168 += einsum(f.aa.ov, (0, 1), x0, (2, 3, 0, 4), (2, 3, 1, 4))
    x169 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x169 += einsum(x0, (0, 1, 2, 3), x36, (2, 4, 3, 5), (1, 0, 4, 5))
    del x36
    x170 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x170 += einsum(v.aaa.xov, (0, 1, 2), x4, (1, 3, 0), (2, 3))
    del x4
    x171 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x171 += einsum(x170, (0, 1), (0, 1)) * -1.0
    x171 += einsum(x94, (0, 1), (1, 0))
    x172 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x172 += einsum(x171, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2)) * -2.0
    del x171
    x173 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x173 += einsum(x165, (0, 1), x0, (2, 3, 0, 4), (2, 3, 1, 4)) * 2.0
    x174 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x174 += einsum(x167, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x167
    x174 += einsum(x39, (0, 1, 2, 3), (1, 0, 2, 3))
    del x39
    x174 += einsum(x168, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x168
    x174 += einsum(x169, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x169
    x174 += einsum(x172, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x172
    x174 += einsum(x173, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x173
    l2new_aaaa += einsum(x174, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    l2new_aaaa += einsum(x174, (0, 1, 2, 3), (3, 2, 0, 1))
    del x174
    x175 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x175 += einsum(f.aa.ov, (0, 1), t1.aa, (2, 1), (0, 2))
    x176 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x176 += einsum(x175, (0, 1), l2.aaaa, (2, 3, 4, 1), (0, 4, 2, 3))
    del x175
    x177 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x177 += einsum(l2.aaaa, (0, 1, 2, 3), x41, (3, 4, 2, 5), (4, 5, 0, 1)) * -1.0
    del x41
    x178 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x178 += einsum(v.aaa.xoo, (0, 1, 2), x2, (2, 3, 0), (1, 3))
    x179 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x179 += einsum(x29, (0,), v.aaa.xoo, (0, 1, 2), (1, 2))
    del x29
    x180 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x180 += einsum(t1.aa, (0, 1), x165, (2, 1), (2, 0))
    del x165
    x181 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x181 += einsum(x178, (0, 1), (0, 1)) * -1.0
    x181 += einsum(x179, (0, 1), (1, 0))
    x181 += einsum(x180, (0, 1), (1, 0))
    del x180
    x182 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x182 += einsum(x181, (0, 1), l2.aaaa, (2, 3, 4, 0), (1, 4, 2, 3)) * -2.0
    del x181
    x183 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x183 += einsum(x176, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x176
    x183 += einsum(x177, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x177
    x183 += einsum(x182, (0, 1, 2, 3), (1, 0, 3, 2))
    del x182
    l2new_aaaa += einsum(x183, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new_aaaa += einsum(x183, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x183
    x184 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x184 += einsum(f.aa.oo, (0, 1), l2.aaaa, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_aaaa += einsum(x184, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    l2new_aaaa += einsum(x184, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    del x184
    x185 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x185 += einsum(x38, (0, 1, 2, 3), (3, 2, 1, 0))
    del x38
    x185 += einsum(x40, (0, 1, 2, 3), (0, 3, 1, 2))
    del x40
    l2new_aaaa += einsum(l2.aaaa, (0, 1, 2, 3), x185, (3, 4, 2, 5), (0, 1, 4, 5)) * 2.0
    del x185
    x186 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x186 += einsum(v.aaa.xvv, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x186, (4, 0, 1, 5), (4, 5, 2, 3))
    del x186
    x187 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x187 += einsum(v.abb.xov, (0, 1, 2), x2, (3, 4, 0), (3, 4, 1, 2))
    del x2
    l2new_abab += einsum(x0, (0, 1, 2, 3), x187, (1, 2, 4, 5), (3, 5, 0, 4)) * -2.0
    x188 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x188 += einsum(v.aaa.xoo, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 2, 3, 4))
    l2new_abab += einsum(x0, (0, 1, 2, 3), x188, (1, 2, 4, 5), (3, 5, 0, 4)) * -2.0
    del x0
    x189 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x189 += einsum(x85, (0, 1), v.abb.xov, (2, 1, 3), (0, 3, 2))
    del x85
    x190 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x190 += einsum(x104, (0, 1, 2), l2.abab, (1, 3, 0, 4), (4, 3, 2))
    del x104
    x191 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x191 += einsum(x21, (0, 1, 2), l2.bbbb, (3, 1, 4, 0), (4, 3, 2)) * 2.0
    del x21
    x192 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x192 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0))
    x192 += einsum(x133, (0, 1, 2), (0, 1, 2))
    x192 += einsum(x189, (0, 1, 2), (0, 1, 2)) * -1.0
    x192 += einsum(x190, (0, 1, 2), (0, 1, 2))
    x192 += einsum(x191, (0, 1, 2), (0, 1, 2))
    l2new_abab += einsum(v.aaa.xov, (0, 1, 2), x192, (3, 4, 0), (2, 4, 1, 3))
    del x192
    x193 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x193 += einsum(x68, (0, 1, 2), (0, 1, 2)) * 0.5
    del x68
    x193 += einsum(x156, (0, 1, 2), (0, 1, 2)) * -0.5
    del x156
    x193 += einsum(x150, (0, 1, 2), (0, 1, 2))
    del x150
    x193 += einsum(x151, (0, 1, 2), (0, 1, 2))
    del x151
    l2new_abab += einsum(v.abb.xov, (0, 1, 2), x193, (3, 4, 0), (4, 2, 3, 1)) * 2.0
    del x193
    x194 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x194 += einsum(v.aaa.xvv, (0, 1, 2), x105, (3, 4, 0), (4, 3, 1, 2))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x194, (3, 4, 0, 5), (5, 1, 2, 4)) * -1.0
    del x194
    x195 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x195 += einsum(v.abb.xvv, (0, 1, 2), x105, (3, 4, 0), (3, 4, 1, 2))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x195, (4, 3, 5, 1), (0, 5, 2, 4)) * -1.0
    x196 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x196 += einsum(v.abb.xvv, (0, 1, 2), x24, (3, 4, 0), (4, 3, 1, 2))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x196, (2, 4, 1, 5), (0, 5, 4, 3)) * -1.0
    del x196
    x197 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x197 += einsum(x105, (0, 1, 2), x24, (3, 4, 2), (4, 3, 1, 0))
    del x24, x105
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x197, (2, 4, 3, 5), (0, 1, 4, 5))
    del x197
    x198 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x198 += einsum(f.aa.vv, (0, 1), (0, 1))
    x198 += einsum(x170, (0, 1), (1, 0)) * -1.0
    del x170
    x198 += einsum(x94, (0, 1), (1, 0))
    del x94
    l2new_abab += einsum(x198, (0, 1), l2.abab, (0, 2, 3, 4), (1, 2, 3, 4))
    del x198
    x199 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x199 += einsum(v.abb.xov, (0, 1, 2), x8, (1, 3, 0), (2, 3))
    del x8
    x200 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x200 += einsum(f.bb.vv, (0, 1), (0, 1))
    x200 += einsum(x199, (0, 1), (1, 0)) * -1.0
    x200 += einsum(x139, (0, 1), (1, 0))
    l2new_abab += einsum(x200, (0, 1), l2.abab, (2, 0, 3, 4), (2, 1, 3, 4))
    del x200
    x201 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x201 += einsum(x18, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x201 += einsum(x18, (0, 1, 2, 3), (1, 2, 0, 3))
    del x18
    l2new_abab += einsum(x201, (0, 1, 2, 3), x45, (0, 2, 4, 5), (3, 5, 1, 4)) * -1.0
    del x201
    x202 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x202 += einsum(x101, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x202 += einsum(x101, (0, 1, 2, 3), (1, 2, 0, 3))
    l2new_abab += einsum(x1, (0, 1, 2, 3), x202, (1, 4, 2, 5), (3, 5, 0, 4)) * -1.0
    x203 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x203 += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3))
    x203 += einsum(x102, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l2new_abab += einsum(x1, (0, 1, 2, 3), x203, (1, 4, 2, 5), (3, 5, 0, 4)) * -1.0
    del x1
    x204 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x204 += einsum(v.abb.xoo, (0, 1, 2), x3, (2, 3, 0), (1, 3))
    del x3
    x205 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x205 += einsum(f.bb.oo, (0, 1), (0, 1))
    x205 += einsum(x204, (0, 1), (0, 1)) * -1.0
    x205 += einsum(x142, (0, 1), (1, 0))
    x205 += einsum(x143, (0, 1), (1, 0))
    del x143
    l2new_abab += einsum(x205, (0, 1), l2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x205
    x206 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x206 += einsum(f.aa.oo, (0, 1), (0, 1))
    x206 += einsum(x178, (0, 1), (0, 1)) * -1.0
    del x178
    x206 += einsum(x179, (0, 1), (1, 0))
    del x179
    x206 += einsum(t1.aa, (0, 1), x31, (2, 1), (0, 2))
    del x31
    l2new_abab += einsum(x206, (0, 1), l2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    del x206
    x207 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x207 += einsum(v.abb.xvv, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 3, 4, 2))
    l2new_bbbb += einsum(l2.bbbb, (0, 1, 2, 3), x207, (4, 5, 0, 1), (4, 5, 2, 3)) * -2.0
    del x207
    x208 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x208 += einsum(x188, (0, 1, 2, 3), x45, (0, 1, 4, 5), (4, 2, 5, 3))
    del x188
    x209 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x209 += einsum(x187, (0, 1, 2, 3), x45, (0, 1, 4, 5), (4, 2, 5, 3))
    del x45, x187
    x210 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x210 += einsum(x190, (0, 1, 2), (0, 1, 2))
    del x190
    x210 += einsum(x191, (0, 1, 2), (0, 1, 2))
    del x191
    x211 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x211 += einsum(v.abb.xov, (0, 1, 2), x210, (3, 4, 0), (3, 1, 4, 2))
    del x210
    x212 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x212 += einsum(l2.bbbb, (0, 1, 2, 3), x195, (4, 3, 5, 1), (4, 2, 5, 0)) * 2.0
    del x195
    x213 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x213 += einsum(x133, (0, 1, 2), (0, 1, 2))
    del x133
    x213 += einsum(x189, (0, 1, 2), (0, 1, 2)) * -1.0
    del x189
    x214 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x214 += einsum(v.abb.xov, (0, 1, 2), x213, (3, 4, 0), (3, 1, 4, 2))
    del x213
    x215 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x215 += einsum(x203, (0, 1, 2, 3), x83, (0, 4, 2, 5), (1, 4, 3, 5)) * 2.0
    del x203
    x216 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x216 += einsum(x202, (0, 1, 2, 3), x83, (0, 4, 2, 5), (1, 4, 3, 5)) * 2.0
    del x202
    x217 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x217 += einsum(x101, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x101
    x217 += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3))
    del x102
    x218 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x218 += einsum(l1.bb, (0, 1), x217, (1, 2, 3, 4), (2, 3, 4, 0))
    del x217
    x219 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x219 += einsum(x108, (0, 1), (0, 1)) * -1.0
    del x108
    x219 += einsum(x109, (0, 1), (0, 1))
    del x109
    x220 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x220 += einsum(f.bb.ov, (0, 1), l1.bb, (2, 3), (0, 3, 1, 2))
    x220 += einsum(x208, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x208
    x220 += einsum(x209, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x209
    x220 += einsum(x211, (0, 1, 2, 3), (0, 1, 2, 3))
    del x211
    x220 += einsum(x212, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x212
    x220 += einsum(x214, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x214
    x220 += einsum(x215, (0, 1, 2, 3), (1, 0, 3, 2))
    del x215
    x220 += einsum(x216, (0, 1, 2, 3), (1, 0, 3, 2))
    del x216
    x220 += einsum(x218, (0, 1, 2, 3), (0, 1, 3, 2))
    del x218
    x220 += einsum(l1.bb, (0, 1), x219, (2, 3), (1, 2, 0, 3))
    l2new_bbbb += einsum(x220, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_bbbb += einsum(x220, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_bbbb += einsum(x220, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new_bbbb += einsum(x220, (0, 1, 2, 3), (3, 2, 1, 0))
    del x220
    x221 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x221 += einsum(f.bb.vv, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    x222 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x222 += einsum(f.bb.ov, (0, 1), x83, (2, 3, 0, 4), (2, 3, 1, 4))
    x223 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x223 += einsum(x113, (0, 1, 2, 3), x83, (4, 5, 0, 2), (5, 4, 1, 3))
    del x113
    x224 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x224 += einsum(x199, (0, 1), (0, 1)) * -1.0
    del x199
    x224 += einsum(x139, (0, 1), (1, 0))
    del x139
    x225 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x225 += einsum(x224, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2)) * -2.0
    del x224
    x226 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x226 += einsum(x219, (0, 1), x83, (2, 3, 0, 4), (2, 3, 1, 4)) * 2.0
    del x83
    x227 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x227 += einsum(x221, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x221
    x227 += einsum(x116, (0, 1, 2, 3), (1, 0, 2, 3))
    del x116
    x227 += einsum(x222, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x222
    x227 += einsum(x223, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x223
    x227 += einsum(x225, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x225
    x227 += einsum(x226, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x226
    l2new_bbbb += einsum(x227, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    l2new_bbbb += einsum(x227, (0, 1, 2, 3), (3, 2, 0, 1))
    del x227
    x228 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x228 += einsum(f.bb.ov, (0, 1), t1.bb, (2, 1), (0, 2))
    x229 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x229 += einsum(x228, (0, 1), l2.bbbb, (2, 3, 4, 1), (0, 4, 2, 3))
    del x228
    x230 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x230 += einsum(l2.bbbb, (0, 1, 2, 3), x118, (3, 4, 2, 5), (4, 5, 0, 1)) * -1.0
    del x118
    x231 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x231 += einsum(t1.bb, (0, 1), x219, (2, 1), (2, 0))
    del x219
    x232 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x232 += einsum(x204, (0, 1), (0, 1)) * -1.0
    del x204
    x232 += einsum(x142, (0, 1), (1, 0))
    del x142
    x232 += einsum(x231, (0, 1), (1, 0))
    del x231
    x233 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x233 += einsum(x232, (0, 1), l2.bbbb, (2, 3, 4, 0), (1, 4, 2, 3)) * -2.0
    del x232
    x234 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x234 += einsum(x229, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x229
    x234 += einsum(x230, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x230
    x234 += einsum(x233, (0, 1, 2, 3), (1, 0, 3, 2))
    del x233
    l2new_bbbb += einsum(x234, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new_bbbb += einsum(x234, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x234
    x235 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x235 += einsum(f.bb.oo, (0, 1), l2.bbbb, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_bbbb += einsum(x235, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    l2new_bbbb += einsum(x235, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    del x235
    x236 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x236 += einsum(x115, (0, 1, 2, 3), (3, 2, 1, 0))
    del x115
    x236 += einsum(x117, (0, 1, 2, 3), (0, 3, 1, 2))
    del x117
    l2new_bbbb += einsum(l2.bbbb, (0, 1, 2, 3), x236, (3, 4, 2, 5), (0, 1, 4, 5)) * 2.0
    del x236

    l1new.aa = l1new_aa
    l1new.bb = l1new_bb
    l2new.aaaa = l2new_aaaa
    l2new.abab = l2new_abab
    l2new.bbbb = l2new_bbbb

    return {"l1new": l1new, "l2new": l2new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    rdm1_f = Namespace()

    delta = Namespace(aa=Namespace(), bb=Namespace())
    delta.aa = Namespace(oo=np.eye(nocc[0]), vv=np.eye(nvir[0]))
    delta.bb = Namespace(oo=np.eye(nocc[1]), vv=np.eye(nvir[1]))

    # RDM1
    rdm1_f_aa_oo = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    rdm1_f_aa_oo += einsum(delta.aa.oo, (0, 1), (0, 1))
    rdm1_f_bb_oo = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    rdm1_f_bb_oo += einsum(delta.bb.oo, (0, 1), (0, 1))
    rdm1_f_aa_ov = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    rdm1_f_aa_ov += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), (2, 3))
    rdm1_f_aa_ov += einsum(t1.aa, (0, 1), (0, 1))
    rdm1_f_aa_ov += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 0), (2, 3)) * 2.0
    rdm1_f_bb_ov = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    rdm1_f_bb_ov += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 0), (2, 3)) * 2.0
    rdm1_f_bb_ov += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), (2, 3))
    rdm1_f_bb_ov += einsum(t1.bb, (0, 1), (0, 1))
    rdm1_f_aa_vo = np.zeros((nvir[0], nocc[0]), dtype=types[float])
    rdm1_f_aa_vo += einsum(l1.aa, (0, 1), (0, 1))
    rdm1_f_bb_vo = np.zeros((nvir[1], nocc[1]), dtype=types[float])
    rdm1_f_bb_vo += einsum(l1.bb, (0, 1), (0, 1))
    rdm1_f_aa_vv = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    rdm1_f_aa_vv += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4)) * 2.0
    rdm1_f_aa_vv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    rdm1_f_aa_vv += einsum(l1.aa, (0, 1), t1.aa, (1, 2), (0, 2))
    rdm1_f_bb_vv = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    rdm1_f_bb_vv += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4)) * 2.0
    rdm1_f_bb_vv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    rdm1_f_bb_vv += einsum(l1.bb, (0, 1), t1.bb, (1, 2), (0, 2))
    x0 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x0 += einsum(l1.aa, (0, 1), t1.aa, (2, 0), (1, 2))
    rdm1_f_aa_oo += einsum(x0, (0, 1), (1, 0)) * -1.0
    x1 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x1 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    rdm1_f_aa_oo += einsum(x1, (0, 1), (1, 0)) * -1.0
    x2 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x2 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    rdm1_f_aa_oo += einsum(x2, (0, 1), (1, 0)) * -2.0
    x3 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x3 += einsum(l1.bb, (0, 1), t1.bb, (2, 0), (1, 2))
    rdm1_f_bb_oo += einsum(x3, (0, 1), (1, 0)) * -1.0
    x4 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x4 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    rdm1_f_bb_oo += einsum(x4, (0, 1), (1, 0)) * -2.0
    x5 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x5 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    rdm1_f_bb_oo += einsum(x5, (0, 1), (1, 0)) * -1.0
    x6 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x6 += einsum(t1.aa, (0, 1), l2.abab, (1, 2, 3, 4), (3, 0, 4, 2))
    rdm1_f_aa_ov += einsum(t2.abab, (0, 1, 2, 3), x6, (0, 4, 1, 3), (4, 2)) * -1.0
    del x6
    x7 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x7 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm1_f_aa_ov += einsum(t2.aaaa, (0, 1, 2, 3), x7, (1, 0, 4, 3), (4, 2)) * -2.0
    del x7
    x8 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x8 += einsum(x0, (0, 1), (0, 1)) * 0.5
    del x0
    x8 += einsum(x2, (0, 1), (0, 1))
    del x2
    x8 += einsum(x1, (0, 1), (0, 1)) * 0.5
    del x1
    rdm1_f_aa_ov += einsum(t1.aa, (0, 1), x8, (0, 2), (2, 1)) * -2.0
    del x8
    x9 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x9 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm1_f_bb_ov += einsum(t2.bbbb, (0, 1, 2, 3), x9, (1, 0, 4, 3), (4, 2)) * -2.0
    del x9
    x10 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x10 += einsum(t1.bb, (0, 1), l2.abab, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm1_f_bb_ov += einsum(t2.abab, (0, 1, 2, 3), x10, (0, 1, 4, 2), (4, 3)) * -1.0
    del x10
    x11 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x11 += einsum(x3, (0, 1), (0, 1))
    del x3
    x11 += einsum(x5, (0, 1), (0, 1))
    del x5
    x11 += einsum(x4, (0, 1), (0, 1)) * 2.0
    del x4
    rdm1_f_bb_ov += einsum(t1.bb, (0, 1), x11, (0, 2), (2, 1)) * -1.0
    del x11

    rdm1_f_aa = np.block([[rdm1_f_aa_oo, rdm1_f_aa_ov], [rdm1_f_aa_vo, rdm1_f_aa_vv]])
    rdm1_f_bb = np.block([[rdm1_f_bb_oo, rdm1_f_bb_ov], [rdm1_f_bb_vo, rdm1_f_bb_vv]])

    rdm1_f.aa = rdm1_f_aa
    rdm1_f.bb = rdm1_f_bb

    return rdm1_f

def make_rdm2_f(f=None, v=None, nocc=None, nvir=None, naux=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    rdm2_f = Namespace()

    delta = Namespace(aa=Namespace(), bb=Namespace())
    delta.aa = Namespace(oo=np.eye(nocc[0]), vv=np.eye(nvir[0]))
    delta.bb = Namespace(oo=np.eye(nocc[1]), vv=np.eye(nvir[1]))

    # RDM2
    rdm2_f_aaaa_oooo = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), delta.aa.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), delta.aa.oo, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_bbbb_oooo = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), delta.bb.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), delta.bb.oo, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_ovoo = np.zeros((nocc[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_ovoo += einsum(delta.aa.oo, (0, 1), l1.aa, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_ovoo += einsum(delta.aa.oo, (0, 1), l1.aa, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_abab_ovoo = np.zeros((nocc[0], nvir[1], nocc[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_ovoo += einsum(delta.aa.oo, (0, 1), l1.bb, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_ovoo = np.zeros((nocc[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_ovoo += einsum(delta.bb.oo, (0, 1), l1.bb, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_ovoo += einsum(delta.bb.oo, (0, 1), l1.bb, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_vooo = np.zeros((nvir[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_vooo += einsum(delta.aa.oo, (0, 1), l1.aa, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_aaaa_vooo += einsum(delta.aa.oo, (0, 1), l1.aa, (2, 3), (2, 0, 3, 1))
    rdm2_f_abab_vooo = np.zeros((nvir[0], nocc[1], nocc[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_vooo += einsum(delta.bb.oo, (0, 1), l1.aa, (2, 3), (2, 0, 3, 1))
    rdm2_f_bbbb_vooo = np.zeros((nvir[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_vooo += einsum(delta.bb.oo, (0, 1), l1.bb, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_bbbb_vooo += einsum(delta.bb.oo, (0, 1), l1.bb, (2, 3), (2, 0, 3, 1))
    rdm2_f_aaaa_oovv = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_oovv += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_aaaa_oovv += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_oovv += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3))
    rdm2_f_abab_oovv = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_oovv = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_oovv += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_aaaa_ovov = np.zeros((nocc[0], nvir[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_ovov += einsum(l1.aa, (0, 1), t1.aa, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_bbbb_ovov = np.zeros((nocc[1], nvir[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_ovov += einsum(l1.bb, (0, 1), t1.bb, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_aaaa_ovvo = np.zeros((nocc[0], nvir[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_ovvo += einsum(l1.aa, (0, 1), t1.aa, (2, 3), (2, 0, 3, 1))
    rdm2_f_abab_ovvo = np.zeros((nocc[0], nvir[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_ovvo += einsum(l1.bb, (0, 1), t1.aa, (2, 3), (2, 0, 3, 1))
    rdm2_f_bbbb_ovvo = np.zeros((nocc[1], nvir[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_ovvo += einsum(l1.bb, (0, 1), t1.bb, (2, 3), (2, 0, 3, 1))
    rdm2_f_aaaa_voov = np.zeros((nvir[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_voov += einsum(l1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3))
    rdm2_f_abab_voov = np.zeros((nvir[0], nocc[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_voov += einsum(l1.aa, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_voov = np.zeros((nvir[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_voov += einsum(l1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_vovo = np.zeros((nvir[0], nocc[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_vovo += einsum(l1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_bbbb_vovo = np.zeros((nvir[1], nocc[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_vovo += einsum(l1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_vvoo = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_vvoo += einsum(l2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_abab_vvoo = np.zeros((nvir[0], nvir[1], nocc[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_vvoo += einsum(l2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_vvoo = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_vvoo += einsum(l2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_abab_ovvv = np.zeros((nocc[0], nvir[1], nvir[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_ovvv += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_abab_vovv = np.zeros((nvir[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_vovv += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 3, 4), (0, 2, 3, 4))
    rdm2_f_abab_vvvo = np.zeros((nvir[0], nvir[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_vvvo += einsum(t1.aa, (0, 1), l2.abab, (2, 3, 0, 4), (2, 3, 1, 4))
    rdm2_f_aaaa_vvvv = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_vvvv += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 5), (0, 1, 4, 5)) * 2.0
    rdm2_f_abab_vvvv = np.zeros((nvir[0], nvir[1], nvir[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_vvvv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), (0, 1, 4, 5))
    rdm2_f_bbbb_vvvv = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_vvvv += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 5), (0, 1, 4, 5)) * 2.0
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x0 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_aaaa_oooo += einsum(x0, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm2_f_aaaa_ovoo += einsum(x1, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0
    rdm2_f_aaaa_vooo += einsum(x1, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x2 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x2 += einsum(t1.aa, (0, 1), x1, (2, 3, 4, 1), (2, 3, 0, 4))
    rdm2_f_aaaa_oooo += einsum(x2, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0
    x3 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x3 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    x4 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    x5 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x5 += einsum(x3, (0, 1), (0, 1))
    x5 += einsum(x4, (0, 1), (0, 1)) * 0.5
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x5, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x5, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x5, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x5, (2, 3), (3, 0, 2, 1)) * -2.0
    x6 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x6 += einsum(l1.aa, (0, 1), t1.aa, (2, 0), (1, 2))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x6, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x6, (2, 3), (3, 0, 1, 2))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x6, (2, 3), (0, 3, 2, 1))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x6, (2, 3), (3, 0, 2, 1)) * -1.0
    x7 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x7 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), (2, 4, 3, 5))
    rdm2_f_abab_oooo = np.zeros((nocc[0], nocc[1], nocc[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_oooo += einsum(x7, (0, 1, 2, 3), (1, 3, 0, 2))
    x8 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x8 += einsum(t1.bb, (0, 1), l2.abab, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm2_f_abab_vooo += einsum(x8, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    rdm2_f_abab_vovo = np.zeros((nvir[0], nocc[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_vovo += einsum(t1.aa, (0, 1), x8, (0, 2, 3, 4), (4, 3, 1, 2)) * -1.0
    rdm2_f_abab_vovv += einsum(t2.abab, (0, 1, 2, 3), x8, (0, 1, 4, 5), (5, 4, 2, 3)) * -1.0
    x9 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x9 += einsum(t1.aa, (0, 1), x8, (2, 3, 4, 1), (2, 0, 3, 4))
    rdm2_f_abab_oooo += einsum(x9, (0, 1, 2, 3), (1, 3, 0, 2))
    x10 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x10 += einsum(l1.bb, (0, 1), t1.bb, (2, 0), (1, 2))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (3, 0, 1, 2))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (0, 3, 2, 1))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (3, 0, 2, 1)) * -1.0
    x11 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x11 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    x12 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x12 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    x13 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x13 += einsum(delta.bb.oo, (0, 1), (0, 1)) * -1.0
    x13 += einsum(x10, (0, 1), (0, 1))
    x13 += einsum(x11, (0, 1), (0, 1))
    x13 += einsum(x12, (0, 1), (0, 1)) * 2.0
    rdm2_f_abab_oooo += einsum(delta.aa.oo, (0, 1), x13, (2, 3), (0, 3, 1, 2)) * -1.0
    del x13
    x14 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x14 += einsum(x6, (0, 1), (0, 1))
    x14 += einsum(x3, (0, 1), (0, 1)) * 2.0
    x14 += einsum(x4, (0, 1), (0, 1))
    rdm2_f_abab_oooo += einsum(delta.bb.oo, (0, 1), x14, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_abab_oovv += einsum(x14, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    x15 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x15 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_bbbb_oooo += einsum(x15, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x16 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x16 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm2_f_bbbb_ovoo += einsum(x16, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0
    rdm2_f_bbbb_vooo += einsum(x16, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x17 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x17 += einsum(t1.bb, (0, 1), x16, (2, 3, 4, 1), (2, 3, 0, 4))
    rdm2_f_bbbb_oooo += einsum(x17, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0
    x18 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x18 += einsum(x11, (0, 1), (0, 1))
    x18 += einsum(x12, (0, 1), (0, 1)) * 2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x18, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x18, (2, 3), (0, 3, 2, 1))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x18, (2, 3), (3, 0, 1, 2))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x18, (2, 3), (3, 0, 2, 1)) * -1.0
    x19 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x19 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_aaaa_ooov = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_ooov += einsum(x19, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_aaaa_oovo = np.zeros((nocc[0], nocc[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_oovo += einsum(x19, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x20 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x20 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 0), (2, 3))
    x21 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x21 += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), (2, 3))
    x22 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x22 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (0, 1, 4, 3), (4, 2)) * -1.0
    x23 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x23 += einsum(t1.aa, (0, 1), l2.abab, (1, 2, 3, 4), (3, 0, 4, 2))
    rdm2_f_abab_ovoo += einsum(x23, (0, 1, 2, 3), (1, 3, 0, 2)) * -1.0
    rdm2_f_abab_ovov = np.zeros((nocc[0], nvir[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_ovov += einsum(t1.bb, (0, 1), x23, (2, 3, 0, 4), (3, 4, 2, 1)) * -1.0
    rdm2_f_abab_ovvv += einsum(t2.abab, (0, 1, 2, 3), x23, (0, 4, 1, 5), (4, 5, 2, 3)) * -1.0
    x24 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x24 += einsum(t2.abab, (0, 1, 2, 3), x23, (0, 4, 1, 3), (4, 2))
    x25 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x25 += einsum(t1.aa, (0, 1), x14, (0, 2), (2, 1)) * 0.5
    x26 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x26 += einsum(x20, (0, 1), (0, 1)) * -1.0
    x26 += einsum(x21, (0, 1), (0, 1)) * -0.5
    x26 += einsum(x22, (0, 1), (0, 1))
    x26 += einsum(x24, (0, 1), (0, 1)) * 0.5
    x26 += einsum(x25, (0, 1), (0, 1))
    del x25
    rdm2_f_aaaa_ooov += einsum(delta.aa.oo, (0, 1), x26, (2, 3), (0, 2, 1, 3)) * -2.0
    rdm2_f_aaaa_ooov += einsum(delta.aa.oo, (0, 1), x26, (2, 3), (2, 0, 1, 3)) * 2.0
    rdm2_f_aaaa_oovo += einsum(delta.aa.oo, (0, 1), x26, (2, 3), (0, 2, 3, 1)) * 2.0
    rdm2_f_aaaa_oovo += einsum(delta.aa.oo, (0, 1), x26, (2, 3), (2, 0, 3, 1)) * -2.0
    x27 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x27 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x28 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x28 += einsum(t2.abab, (0, 1, 2, 3), x23, (4, 5, 1, 3), (4, 5, 0, 2))
    x29 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x29 += einsum(delta.aa.oo, (0, 1), t1.aa, (2, 3), (0, 1, 2, 3))
    x29 += einsum(x27, (0, 1, 2, 3), (1, 0, 2, 3)) * -4.0
    x29 += einsum(x28, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x29 += einsum(t1.aa, (0, 1), x14, (2, 3), (0, 2, 3, 1))
    rdm2_f_aaaa_ooov += einsum(x29, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_ooov += einsum(x29, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_aaaa_oovo += einsum(x29, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_oovo += einsum(x29, (0, 1, 2, 3), (2, 0, 3, 1))
    del x29
    x30 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x30 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2))
    del x0
    x30 += einsum(x2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x2
    rdm2_f_aaaa_oovv += einsum(t2.aaaa, (0, 1, 2, 3), x30, (0, 1, 4, 5), (5, 4, 2, 3)) * -2.0
    x31 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x31 += einsum(t1.aa, (0, 1), x30, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    rdm2_f_aaaa_ooov += einsum(x31, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_aaaa_oovo += einsum(x31, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x31
    x32 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x32 += einsum(t2.abab, (0, 1, 2, 3), x8, (4, 1, 5, 2), (4, 0, 5, 3))
    rdm2_f_abab_ooov = np.zeros((nocc[0], nocc[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_ooov += einsum(x32, (0, 1, 2, 3), (1, 2, 0, 3))
    x33 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x33 += einsum(t2.bbbb, (0, 1, 2, 3), x23, (4, 5, 1, 3), (4, 5, 0, 2))
    rdm2_f_abab_ooov += einsum(x33, (0, 1, 2, 3), (1, 2, 0, 3)) * -2.0
    x34 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x34 += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), (1, 2, 3, 4))
    rdm2_f_abab_ooov += einsum(x34, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    x35 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x35 += einsum(t2.abab, (0, 1, 2, 3), x1, (4, 0, 5, 2), (4, 5, 1, 3)) * -1.0
    rdm2_f_abab_ooov += einsum(x35, (0, 1, 2, 3), (1, 2, 0, 3)) * -2.0
    x36 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x36 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    del x7
    x36 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    del x9
    rdm2_f_abab_ooov += einsum(t1.bb, (0, 1), x36, (2, 3, 0, 4), (3, 4, 2, 1))
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x36, (0, 4, 1, 5), (4, 5, 2, 3))
    x37 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x37 += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), (2, 3))
    x38 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x38 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 0), (2, 3))
    x39 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x39 += einsum(t2.abab, (0, 1, 2, 3), x8, (0, 1, 4, 2), (4, 3))
    x40 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x40 += einsum(t2.bbbb, (0, 1, 2, 3), x16, (0, 1, 4, 3), (4, 2)) * -1.0
    x41 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x41 += einsum(x10, (0, 1), (0, 1)) * 0.5
    x41 += einsum(x11, (0, 1), (0, 1)) * 0.5
    x41 += einsum(x12, (0, 1), (0, 1))
    x42 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x42 += einsum(t1.bb, (0, 1), x41, (0, 2), (2, 1)) * 2.0
    x43 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x43 += einsum(t1.bb, (0, 1), (0, 1)) * -1.0
    x43 += einsum(x37, (0, 1), (0, 1)) * -1.0
    x43 += einsum(x38, (0, 1), (0, 1)) * -2.0
    x43 += einsum(x39, (0, 1), (0, 1))
    x43 += einsum(x40, (0, 1), (0, 1)) * 2.0
    x43 += einsum(x42, (0, 1), (0, 1))
    rdm2_f_abab_ooov += einsum(delta.aa.oo, (0, 1), x43, (2, 3), (0, 2, 1, 3)) * -1.0
    del x43
    x44 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x44 += einsum(x6, (0, 1), (0, 1)) * 0.5
    x44 += einsum(x3, (0, 1), (0, 1))
    del x3
    x44 += einsum(x4, (0, 1), (0, 1)) * 0.5
    del x4
    rdm2_f_abab_ooov += einsum(t1.bb, (0, 1), x44, (2, 3), (3, 0, 2, 1)) * -2.0
    del x44
    x45 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x45 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_bbbb_ooov = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_ooov += einsum(x45, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_bbbb_oovo = np.zeros((nocc[1], nocc[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_oovo += einsum(x45, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x46 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x46 += einsum(x37, (0, 1), (0, 1)) * -1.0
    x46 += einsum(x38, (0, 1), (0, 1)) * -2.0
    x46 += einsum(x39, (0, 1), (0, 1))
    x46 += einsum(x40, (0, 1), (0, 1)) * 2.0
    x46 += einsum(x42, (0, 1), (0, 1))
    del x42
    rdm2_f_bbbb_ooov += einsum(delta.bb.oo, (0, 1), x46, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_bbbb_ooov += einsum(delta.bb.oo, (0, 1), x46, (2, 3), (2, 0, 1, 3))
    rdm2_f_bbbb_oovo += einsum(delta.bb.oo, (0, 1), x46, (2, 3), (0, 2, 3, 1))
    rdm2_f_bbbb_oovo += einsum(delta.bb.oo, (0, 1), x46, (2, 3), (2, 0, 3, 1)) * -1.0
    x47 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x47 += einsum(t2.abab, (0, 1, 2, 3), x8, (0, 4, 5, 2), (4, 5, 1, 3))
    x48 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x48 += einsum(t2.bbbb, (0, 1, 2, 3), x16, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x49 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x49 += einsum(delta.bb.oo, (0, 1), t1.bb, (2, 3), (0, 1, 2, 3))
    x49 += einsum(x47, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x49 += einsum(x48, (0, 1, 2, 3), (1, 0, 2, 3)) * -4.0
    x49 += einsum(t1.bb, (0, 1), x41, (2, 3), (0, 2, 3, 1)) * 2.0
    rdm2_f_bbbb_ooov += einsum(x49, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_ooov += einsum(x49, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_bbbb_oovo += einsum(x49, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_bbbb_oovo += einsum(x49, (0, 1, 2, 3), (2, 0, 3, 1))
    del x49
    x50 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x50 += einsum(x15, (0, 1, 2, 3), (1, 0, 3, 2))
    del x15
    x50 += einsum(x17, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x17
    rdm2_f_bbbb_oovv += einsum(t2.bbbb, (0, 1, 2, 3), x50, (0, 1, 4, 5), (5, 4, 2, 3)) * -2.0
    x51 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x51 += einsum(t1.bb, (0, 1), x50, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    rdm2_f_bbbb_ooov += einsum(x51, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_bbbb_oovo += einsum(x51, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x51
    x52 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x52 += einsum(t2.abab, (0, 1, 2, 3), x23, (0, 4, 5, 3), (4, 5, 1, 2))
    rdm2_f_abab_oovo = np.zeros((nocc[0], nocc[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_oovo += einsum(x52, (0, 1, 2, 3), (0, 2, 3, 1))
    x53 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x53 += einsum(t2.aaaa, (0, 1, 2, 3), x8, (1, 4, 5, 3), (0, 4, 5, 2))
    rdm2_f_abab_oovo += einsum(x53, (0, 1, 2, 3), (0, 2, 3, 1)) * -2.0
    x54 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x54 += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), (2, 1, 3, 4))
    rdm2_f_abab_oovo += einsum(x54, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x55 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x55 += einsum(t2.abab, (0, 1, 2, 3), x16, (4, 1, 5, 3), (0, 4, 5, 2)) * -1.0
    rdm2_f_abab_oovo += einsum(x55, (0, 1, 2, 3), (0, 2, 3, 1)) * -2.0
    x56 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x56 += einsum(t1.aa, (0, 1), x36, (0, 2, 3, 4), (2, 3, 4, 1))
    del x36
    rdm2_f_abab_oovo += einsum(x56, (0, 1, 2, 3), (0, 2, 3, 1))
    x57 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x57 += einsum(t1.aa, (0, 1), x14, (0, 2), (2, 1))
    del x14
    x58 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x58 += einsum(t1.aa, (0, 1), (0, 1)) * -1.0
    x58 += einsum(x20, (0, 1), (0, 1)) * -2.0
    del x20
    x58 += einsum(x21, (0, 1), (0, 1)) * -1.0
    del x21
    x58 += einsum(x22, (0, 1), (0, 1)) * 2.0
    del x22
    x58 += einsum(x24, (0, 1), (0, 1))
    del x24
    x58 += einsum(x57, (0, 1), (0, 1))
    del x57
    rdm2_f_abab_oovo += einsum(delta.bb.oo, (0, 1), x58, (2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_abab_oovv += einsum(t1.bb, (0, 1), x58, (2, 3), (2, 0, 3, 1)) * -1.0
    del x58
    x59 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x59 += einsum(x10, (0, 1), (0, 1))
    x59 += einsum(x11, (0, 1), (0, 1))
    del x11
    x59 += einsum(x12, (0, 1), (0, 1)) * 2.0
    del x12
    rdm2_f_abab_oovo += einsum(t1.aa, (0, 1), x59, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_abab_oovv += einsum(x59, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x59
    x60 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x60 += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 2, 5, 0), (4, 3, 5, 1))
    rdm2_f_abab_ovvo += einsum(x60, (0, 1, 2, 3), (0, 3, 2, 1)) * 2.0
    x61 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x61 += einsum(t2.abab, (0, 1, 2, 3), x60, (4, 1, 5, 3), (4, 0, 5, 2))
    x62 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x62 += einsum(x27, (0, 1, 2, 3), (0, 1, 2, 3))
    del x27
    x62 += einsum(x28, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x28
    x63 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x63 += einsum(t1.aa, (0, 1), x62, (0, 2, 3, 4), (2, 3, 1, 4)) * 4.0
    del x62
    x64 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x64 += einsum(x61, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x61
    x64 += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3))
    del x63
    x64 += einsum(t1.aa, (0, 1), x26, (2, 3), (0, 2, 1, 3)) * 2.0
    del x26
    rdm2_f_aaaa_oovv += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x64, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_aaaa_oovv += einsum(x64, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_aaaa_oovv += einsum(x64, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x64
    x65 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x65 += einsum(t1.aa, (0, 1), x19, (0, 2, 3, 4), (2, 3, 1, 4))
    del x19
    x66 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x66 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_aaaa_ovov += einsum(x66, (0, 1, 2, 3), (1, 2, 0, 3)) * -4.0
    rdm2_f_aaaa_ovvo += einsum(x66, (0, 1, 2, 3), (1, 2, 3, 0)) * 4.0
    rdm2_f_aaaa_voov += einsum(x66, (0, 1, 2, 3), (2, 1, 0, 3)) * 4.0
    rdm2_f_aaaa_vovo += einsum(x66, (0, 1, 2, 3), (2, 1, 3, 0)) * -4.0
    x67 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x67 += einsum(t2.aaaa, (0, 1, 2, 3), x66, (1, 4, 3, 5), (0, 4, 2, 5))
    x68 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x68 += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (4, 2, 5, 0))
    rdm2_f_abab_ovvo += einsum(x68, (0, 1, 2, 3), (0, 3, 2, 1)) * 2.0
    x69 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x69 += einsum(t2.abab, (0, 1, 2, 3), x68, (4, 1, 5, 3), (4, 0, 5, 2))
    x70 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x70 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4))
    x71 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x71 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    x72 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x72 += einsum(x70, (0, 1), (0, 1))
    x72 += einsum(x71, (0, 1), (0, 1)) * 0.5
    rdm2_f_abab_oovv += einsum(x72, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -2.0
    x73 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x73 += einsum(x72, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -4.0
    del x72
    x74 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x74 += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x65
    x74 += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3)) * 8.0
    del x67
    x74 += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x69
    x74 += einsum(x73, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x73
    rdm2_f_aaaa_oovv += einsum(x74, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_aaaa_oovv += einsum(x74, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x74
    x75 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x75 += einsum(x6, (0, 1), t2.aaaa, (2, 0, 3, 4), (1, 2, 3, 4))
    del x6
    x76 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x76 += einsum(x5, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -4.0
    del x5
    x77 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x77 += einsum(x75, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x75
    x77 += einsum(x76, (0, 1, 2, 3), (0, 1, 3, 2))
    del x76
    rdm2_f_aaaa_oovv += einsum(x77, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x77, (0, 1, 2, 3), (1, 0, 2, 3))
    del x77
    x78 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x78 += einsum(t1.aa, (0, 1), x30, (0, 2, 3, 4), (2, 4, 3, 1))
    del x30
    rdm2_f_aaaa_oovv += einsum(t1.aa, (0, 1), x78, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    del x78
    x79 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x79 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), (3, 4, 0, 5))
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x79, (1, 4, 2, 5), (0, 4, 5, 3))
    rdm2_f_abab_vovo += einsum(x79, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_abab_vovv += einsum(t1.bb, (0, 1), x79, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    del x79
    x80 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x80 += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3))
    x80 += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_abab_oovv += einsum(t2.bbbb, (0, 1, 2, 3), x80, (4, 1, 5, 3), (4, 0, 5, 2)) * 4.0
    del x80
    x81 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x81 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_aaaa_ovov += einsum(x81, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_aaaa_ovvo += einsum(x81, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_aaaa_voov += einsum(x81, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_aaaa_vovo += einsum(x81, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x82 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x82 += einsum(x66, (0, 1, 2, 3), (0, 1, 2, 3))
    del x66
    x82 += einsum(x81, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x81
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x82, (0, 4, 2, 5), (4, 1, 5, 3)) * 4.0
    x83 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x83 += einsum(x54, (0, 1, 2, 3), (0, 1, 2, 3))
    del x54
    x83 += einsum(x53, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x53
    x83 += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x52
    x83 += einsum(x55, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x55
    x83 += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x56
    rdm2_f_abab_oovv += einsum(t1.bb, (0, 1), x83, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x83
    x84 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x84 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    x85 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x85 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4))
    x86 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x86 += einsum(x84, (0, 1), (0, 1)) * 0.5
    x86 += einsum(x85, (0, 1), (0, 1))
    rdm2_f_abab_oovv += einsum(x86, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x86
    x87 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x87 += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3))
    del x34
    x87 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x35
    x87 += einsum(x33, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x33
    x87 += einsum(x32, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x32
    rdm2_f_abab_oovv += einsum(t1.aa, (0, 1), x87, (0, 2, 3, 4), (2, 3, 1, 4)) * -1.0
    del x87
    x88 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x88 += einsum(x37, (0, 1), (0, 1)) * -0.5
    del x37
    x88 += einsum(x38, (0, 1), (0, 1)) * -1.0
    del x38
    x88 += einsum(x39, (0, 1), (0, 1)) * 0.5
    del x39
    x88 += einsum(x40, (0, 1), (0, 1))
    del x40
    x88 += einsum(t1.bb, (0, 1), x41, (0, 2), (2, 1))
    del x41
    rdm2_f_abab_oovv += einsum(t1.aa, (0, 1), x88, (2, 3), (0, 2, 1, 3)) * -2.0
    del x88
    x89 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x89 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), (3, 4, 1, 5))
    rdm2_f_bbbb_ovov += einsum(x89, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_bbbb_ovvo += einsum(x89, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_bbbb_voov += einsum(x89, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_bbbb_vovo += einsum(x89, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x90 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x90 += einsum(t2.bbbb, (0, 1, 2, 3), x89, (1, 4, 3, 5), (4, 0, 5, 2))
    x91 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x91 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3))
    del x47
    x91 += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x48
    x92 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x92 += einsum(t1.bb, (0, 1), x91, (0, 2, 3, 4), (2, 3, 1, 4))
    del x91
    x93 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x93 += einsum(x90, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x90
    x93 += einsum(x92, (0, 1, 2, 3), (0, 1, 2, 3))
    del x92
    x93 += einsum(t1.bb, (0, 1), x46, (2, 3), (0, 2, 1, 3))
    del x46
    rdm2_f_bbbb_oovv += einsum(x93, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_bbbb_oovv += einsum(x93, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_bbbb_oovv += einsum(x93, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_bbbb_oovv += einsum(x93, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x93
    x94 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x94 += einsum(x10, (0, 1), t2.bbbb, (2, 0, 3, 4), (1, 2, 3, 4))
    del x10
    x95 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x95 += einsum(x18, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x18
    x96 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x96 += einsum(x94, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x94
    x96 += einsum(x95, (0, 1, 2, 3), (0, 1, 3, 2))
    del x95
    rdm2_f_bbbb_oovv += einsum(x96, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_bbbb_oovv += einsum(x96, (0, 1, 2, 3), (1, 0, 2, 3))
    del x96
    x97 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x97 += einsum(t1.bb, (0, 1), x45, (0, 2, 3, 4), (2, 3, 1, 4))
    del x45
    x98 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x98 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_bbbb_ovov += einsum(x98, (0, 1, 2, 3), (1, 2, 0, 3)) * -4.0
    rdm2_f_bbbb_ovvo += einsum(x98, (0, 1, 2, 3), (1, 2, 3, 0)) * 4.0
    rdm2_f_bbbb_voov += einsum(x98, (0, 1, 2, 3), (2, 1, 0, 3)) * 4.0
    rdm2_f_bbbb_vovo += einsum(x98, (0, 1, 2, 3), (2, 1, 3, 0)) * -4.0
    x99 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x99 += einsum(t2.bbbb, (0, 1, 2, 3), x98, (1, 4, 3, 5), (4, 0, 5, 2))
    x100 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x100 += einsum(x84, (0, 1), (0, 1))
    x100 += einsum(x85, (0, 1), (0, 1)) * 2.0
    x101 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x101 += einsum(x100, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x100
    x102 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x102 += einsum(x97, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x97
    x102 += einsum(x99, (0, 1, 2, 3), (0, 1, 2, 3)) * 8.0
    del x99
    x102 += einsum(x101, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x101
    rdm2_f_bbbb_oovv += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_oovv += einsum(x102, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x102
    x103 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x103 += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 1, 5), (2, 4, 0, 5))
    rdm2_f_abab_voov += einsum(x103, (0, 1, 2, 3), (2, 1, 0, 3)) * 2.0
    x104 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x104 += einsum(t2.abab, (0, 1, 2, 3), x103, (0, 4, 2, 5), (4, 1, 5, 3))
    x105 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x105 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    x105 += einsum(x104, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x104
    rdm2_f_bbbb_oovv += einsum(x105, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_bbbb_oovv += einsum(x105, (0, 1, 2, 3), (0, 1, 2, 3))
    del x105
    x106 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x106 += einsum(t1.bb, (0, 1), x50, (0, 2, 3, 4), (2, 4, 3, 1))
    del x50
    rdm2_f_bbbb_oovv += einsum(t1.bb, (0, 1), x106, (0, 2, 3, 4), (2, 3, 4, 1)) * -2.0
    del x106
    x107 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x107 += einsum(t1.aa, (0, 1), x1, (2, 0, 3, 4), (2, 3, 4, 1))
    rdm2_f_aaaa_ovov += einsum(x107, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_aaaa_ovvo += einsum(x107, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    rdm2_f_aaaa_voov += einsum(x107, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_aaaa_vovo += einsum(x107, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x108 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x108 += einsum(l1.aa, (0, 1), t1.aa, (1, 2), (0, 2))
    x109 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x109 += einsum(x108, (0, 1), (0, 1))
    x109 += einsum(x70, (0, 1), (0, 1)) * 2.0
    x109 += einsum(x71, (0, 1), (0, 1))
    rdm2_f_aaaa_ovov += einsum(delta.aa.oo, (0, 1), x109, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_voov += einsum(delta.aa.oo, (0, 1), x109, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_abab_vovo += einsum(delta.bb.oo, (0, 1), x109, (2, 3), (2, 0, 3, 1))
    rdm2_f_abab_vovv += einsum(t1.bb, (0, 1), x109, (2, 3), (2, 0, 3, 1))
    x110 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x110 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), (2, 4, 1, 5))
    rdm2_f_abab_ovov += einsum(x110, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_abab_ovvv += einsum(t1.aa, (0, 1), x110, (0, 2, 3, 4), (2, 3, 1, 4)) * -1.0
    del x110
    x111 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x111 += einsum(l1.bb, (0, 1), t1.bb, (1, 2), (0, 2))
    x112 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x112 += einsum(x111, (0, 1), (0, 1)) * 0.5
    x112 += einsum(x84, (0, 1), (0, 1)) * 0.5
    x112 += einsum(x85, (0, 1), (0, 1))
    rdm2_f_abab_ovov += einsum(delta.aa.oo, (0, 1), x112, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_bbbb_ovvo += einsum(delta.bb.oo, (0, 1), x112, (2, 3), (0, 2, 3, 1)) * -2.0
    rdm2_f_bbbb_vovo += einsum(delta.bb.oo, (0, 1), x112, (2, 3), (2, 0, 3, 1)) * 2.0
    rdm2_f_abab_ovvv += einsum(t1.aa, (0, 1), x112, (2, 3), (0, 2, 1, 3)) * 2.0
    del x112
    x113 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x113 += einsum(t1.bb, (0, 1), x16, (2, 0, 3, 4), (2, 3, 4, 1))
    rdm2_f_bbbb_ovov += einsum(x113, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_bbbb_ovvo += einsum(x113, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    rdm2_f_bbbb_voov += einsum(x113, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_bbbb_vovo += einsum(x113, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x114 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x114 += einsum(x111, (0, 1), (0, 1))
    del x111
    x114 += einsum(x84, (0, 1), (0, 1))
    del x84
    x114 += einsum(x85, (0, 1), (0, 1)) * 2.0
    del x85
    rdm2_f_bbbb_ovov += einsum(delta.bb.oo, (0, 1), x114, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_voov += einsum(delta.bb.oo, (0, 1), x114, (2, 3), (2, 0, 1, 3)) * -1.0
    x115 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x115 += einsum(x108, (0, 1), (0, 1)) * 0.5
    del x108
    x115 += einsum(x70, (0, 1), (0, 1))
    del x70
    x115 += einsum(x71, (0, 1), (0, 1)) * 0.5
    del x71
    rdm2_f_aaaa_ovvo += einsum(delta.aa.oo, (0, 1), x115, (2, 3), (0, 2, 3, 1)) * -2.0
    rdm2_f_aaaa_vovo += einsum(delta.aa.oo, (0, 1), x115, (2, 3), (2, 0, 3, 1)) * 2.0
    del x115
    x116 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x116 += einsum(t1.aa, (0, 1), x23, (0, 2, 3, 4), (2, 3, 1, 4))
    del x23
    rdm2_f_abab_ovvo += einsum(x116, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    x117 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x117 += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_abab_voov += einsum(x117, (0, 1, 2, 3), (2, 1, 0, 3)) * 2.0
    x118 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x118 += einsum(t1.bb, (0, 1), x8, (2, 0, 3, 4), (2, 3, 4, 1))
    del x8
    rdm2_f_abab_voov += einsum(x118, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x119 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x119 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_aaaa_ovvv = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_ovvv += einsum(x119, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_aaaa_vovv = np.zeros((nvir[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_vovv += einsum(x119, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x119
    x120 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x120 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (0, 1, 4, 5), (4, 5, 2, 3))
    del x1
    rdm2_f_aaaa_ovvv += einsum(x120, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_aaaa_vovv += einsum(x120, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x120
    x121 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x121 += einsum(t1.aa, (0, 1), x107, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    del x107
    rdm2_f_aaaa_ovvv += einsum(x121, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_aaaa_vovv += einsum(x121, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x121
    x122 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x122 += einsum(t1.aa, (0, 1), x82, (0, 2, 3, 4), (2, 1, 3, 4)) * 4.0
    del x82
    x123 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x123 += einsum(x122, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x122
    x123 += einsum(t1.aa, (0, 1), x109, (2, 3), (0, 2, 1, 3))
    del x109
    rdm2_f_aaaa_ovvv += einsum(x123, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_aaaa_ovvv += einsum(x123, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_aaaa_vovv += einsum(x123, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_aaaa_vovv += einsum(x123, (0, 1, 2, 3), (1, 0, 3, 2))
    del x123
    x124 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x124 += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x60
    x124 += einsum(x116, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x116
    x124 += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x68
    rdm2_f_abab_ovvv += einsum(t1.bb, (0, 1), x124, (2, 0, 3, 4), (2, 4, 3, 1))
    del x124
    x125 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x125 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_bbbb_ovvv = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_ovvv += einsum(x125, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_bbbb_vovv = np.zeros((nvir[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_vovv += einsum(x125, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x125
    x126 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x126 += einsum(t2.bbbb, (0, 1, 2, 3), x16, (0, 1, 4, 5), (4, 5, 2, 3))
    del x16
    rdm2_f_bbbb_ovvv += einsum(x126, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_bbbb_vovv += einsum(x126, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x126
    x127 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x127 += einsum(t1.bb, (0, 1), x113, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    del x113
    rdm2_f_bbbb_ovvv += einsum(x127, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_bbbb_vovv += einsum(x127, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x127
    x128 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x128 += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3))
    del x89
    x128 += einsum(x98, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x98
    x129 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x129 += einsum(t1.bb, (0, 1), x128, (0, 2, 3, 4), (2, 1, 3, 4))
    del x128
    x130 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x130 += einsum(x129, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x129
    x130 += einsum(t1.bb, (0, 1), x114, (2, 3), (0, 2, 1, 3))
    del x114
    rdm2_f_bbbb_ovvv += einsum(x130, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_ovvv += einsum(x130, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_bbbb_vovv += einsum(x130, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_bbbb_vovv += einsum(x130, (0, 1, 2, 3), (1, 0, 3, 2))
    del x130
    x131 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x131 += einsum(x103, (0, 1, 2, 3), (0, 1, 2, 3))
    del x103
    x131 += einsum(x117, (0, 1, 2, 3), (0, 1, 2, 3))
    del x117
    x131 += einsum(x118, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x118
    rdm2_f_abab_vovv += einsum(t1.aa, (0, 1), x131, (0, 2, 3, 4), (3, 2, 1, 4)) * 2.0
    del x131
    x132 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x132 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_aaaa_vvov = np.zeros((nvir[0], nvir[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_vvov += einsum(x132, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_aaaa_vvvo = np.zeros((nvir[0], nvir[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_vvvo += einsum(x132, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    rdm2_f_aaaa_vvvv += einsum(t1.aa, (0, 1), x132, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    del x132
    x133 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x133 += einsum(t1.bb, (0, 1), l2.abab, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_abab_vvov = np.zeros((nvir[0], nvir[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_vvov += einsum(x133, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_abab_vvvv += einsum(t1.aa, (0, 1), x133, (0, 2, 3, 4), (2, 3, 1, 4))
    del x133
    x134 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x134 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_bbbb_vvov = np.zeros((nvir[1], nvir[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_vvov += einsum(x134, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_bbbb_vvvo = np.zeros((nvir[1], nvir[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_vvvo += einsum(x134, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    rdm2_f_bbbb_vvvv += einsum(t1.bb, (0, 1), x134, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    del x134

    rdm2_f_aaaa = pack_2e(rdm2_f_aaaa_oooo, rdm2_f_aaaa_ooov, rdm2_f_aaaa_oovo, rdm2_f_aaaa_ovoo, rdm2_f_aaaa_vooo, rdm2_f_aaaa_oovv, rdm2_f_aaaa_ovov, rdm2_f_aaaa_ovvo, rdm2_f_aaaa_voov, rdm2_f_aaaa_vovo, rdm2_f_aaaa_vvoo, rdm2_f_aaaa_ovvv, rdm2_f_aaaa_vovv, rdm2_f_aaaa_vvov, rdm2_f_aaaa_vvvo, rdm2_f_aaaa_vvvv)
    rdm2_f_abab = pack_2e(rdm2_f_abab_oooo, rdm2_f_abab_ooov, rdm2_f_abab_oovo, rdm2_f_abab_ovoo, rdm2_f_abab_vooo, rdm2_f_abab_oovv, rdm2_f_abab_ovov, rdm2_f_abab_ovvo, rdm2_f_abab_voov, rdm2_f_abab_vovo, rdm2_f_abab_vvoo, rdm2_f_abab_ovvv, rdm2_f_abab_vovv, rdm2_f_abab_vvov, rdm2_f_abab_vvvo, rdm2_f_abab_vvvv)
    rdm2_f_bbbb = pack_2e(rdm2_f_bbbb_oooo, rdm2_f_bbbb_ooov, rdm2_f_bbbb_oovo, rdm2_f_bbbb_ovoo, rdm2_f_bbbb_vooo, rdm2_f_bbbb_oovv, rdm2_f_bbbb_ovov, rdm2_f_bbbb_ovvo, rdm2_f_bbbb_voov, rdm2_f_bbbb_vovo, rdm2_f_bbbb_vvoo, rdm2_f_bbbb_ovvv, rdm2_f_bbbb_vovv, rdm2_f_bbbb_vvov, rdm2_f_bbbb_vvvo, rdm2_f_bbbb_vvvv)

    rdm2_f_aaaa = rdm2_f_aaaa.swapaxes(1, 2)
    rdm2_f_aabb = rdm2_f_abab.swapaxes(1, 2)
    rdm2_f_bbbb = rdm2_f_bbbb.swapaxes(1, 2)

    rdm2_f.aaaa = rdm2_f_aaaa
    rdm2_f.aabb = rdm2_f_aabb
    rdm2_f.bbbb = rdm2_f_bbbb

    return rdm2_f

