# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, naux=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x0 += einsum(v.abb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    e_cc = 0
    e_cc += einsum(v.abb.xov, (0, 1, 2), x0, (1, 2, 0), ())
    del x0
    x1 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x1 += einsum(v.aaa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    x1 += einsum(v.abb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (3, 4, 0))
    e_cc += einsum(v.aaa.xov, (0, 1, 2), x1, (1, 2, 0), ())
    del x1

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, naux=None, t2=None, **kwargs):
    t2new = Namespace()

    # T amplitudes
    x0 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x0 += einsum(v.aaa.xvv, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x0, (4, 3, 5, 2), (0, 1, 4, 5)) * 2.0
    del x0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x1 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xoo, (0, 3, 4), (1, 3, 4, 2))
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x1, (4, 1, 5, 0), (5, 4, 2, 3)) * -2.0
    del x1
    x2 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x2 += einsum(v.aaa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x3 += einsum(x2, (0, 1, 2), x2, (3, 4, 2), (0, 3, 1, 4))
    x4 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x4 += einsum(v.abb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (3, 4, 0))
    x5 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x5 += einsum(x4, (0, 1, 2), x4, (3, 4, 2), (0, 3, 1, 4))
    x6 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x6 += einsum(x2, (0, 1, 2), (0, 1, 2))
    x6 += einsum(x4, (0, 1, 2), (0, 1, 2)) * 0.5
    x7 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x7 += einsum(v.aaa.xov, (0, 1, 2), x6, (1, 3, 0), (3, 2))
    x8 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x8 += einsum(x7, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 4, 0)) * -2.0
    x9 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x9 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x3
    x9 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    del x5
    x9 += einsum(x8, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x8
    t2new_aaaa += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x9, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x9
    x10 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x10 += einsum(f.aa.oo, (0, 1), t2.aaaa, (2, 1, 3, 4), (0, 2, 3, 4))
    x11 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x11 += einsum(v.aaa.xov, (0, 1, 2), x6, (3, 2, 0), (3, 1))
    x12 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum(x11, (0, 1), t2.aaaa, (2, 1, 3, 4), (0, 2, 3, 4)) * -2.0
    del x11
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x13 += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x10
    x13 += einsum(x12, (0, 1, 2, 3), (1, 0, 3, 2))
    del x12
    t2new_aaaa += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x13, (0, 1, 2, 3), (1, 0, 2, 3))
    del x13
    x14 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x14 += einsum(v.aaa.xov, (0, 1, 2), x4, (3, 4, 0), (3, 1, 4, 2))
    x15 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x15 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 2, 3, 4))
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x15, (4, 0, 5, 2), (4, 1, 5, 3)) * -1.0
    x16 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x16 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x16 += einsum(x4, (0, 1, 2), (0, 1, 2))
    x17 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x17 += einsum(v.aaa.xov, (0, 1, 2), x16, (3, 4, 0), (3, 1, 4, 2))
    del x16
    x18 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x18 += einsum(x15, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x15
    x18 += einsum(x17, (0, 1, 2, 3), (1, 0, 3, 2))
    del x17
    x19 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x19 += einsum(t2.aaaa, (0, 1, 2, 3), x18, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    del x18
    x20 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x20 += einsum(x14, (0, 1, 2, 3), (0, 1, 2, 3))
    del x14
    x20 += einsum(x19, (0, 1, 2, 3), (1, 0, 3, 2))
    del x19
    t2new_aaaa += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x20, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_aaaa += einsum(x20, (0, 1, 2, 3), (1, 0, 3, 2))
    del x20
    x21 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x21 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x22 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x22 += einsum(v.aaa.xov, (0, 1, 2), v.aaa.xov, (0, 3, 4), (1, 3, 2, 4))
    x23 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x23 += einsum(x21, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x21
    x23 += einsum(x22, (0, 1, 2, 3), (1, 0, 2, 3))
    del x22
    t2new_aaaa += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x23, (0, 1, 2, 3), (0, 1, 3, 2))
    del x23
    x24 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x24 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x24, (4, 1, 5, 3), (0, 4, 2, 5)) * -1.0
    x25 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x25 += einsum(v.aaa.xoo, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x25, (4, 0, 5, 3), (4, 1, 2, 5)) * -1.0
    del x25
    x26 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x26 += einsum(v.aaa.xoo, (0, 1, 2), v.abb.xoo, (0, 3, 4), (1, 2, 3, 4))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x26, (4, 0, 5, 1), (4, 5, 2, 3))
    del x26
    x27 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x27 += einsum(v.aaa.xvv, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x27, (4, 2, 5, 3), (0, 1, 4, 5))
    del x27
    x28 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x28 += einsum(v.abb.xoo, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 2, 3, 4))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x28, (4, 1, 5, 2), (0, 4, 5, 3)) * -1.0
    del x28
    x29 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x29 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x29 += einsum(x2, (0, 1, 2), (0, 1, 2)) * 2.0
    del x2
    x29 += einsum(x4, (0, 1, 2), (0, 1, 2))
    del x4
    x30 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x30 += einsum(v.aaa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    x31 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x31 += einsum(v.abb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    x32 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x32 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0))
    x32 += einsum(x30, (0, 1, 2), (0, 1, 2))
    x32 += einsum(x31, (0, 1, 2), (0, 1, 2)) * 2.0
    t2new_abab += einsum(x29, (0, 1, 2), x32, (3, 4, 2), (0, 3, 1, 4))
    del x29, x32
    x33 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x33 += einsum(x30, (0, 1, 2), (0, 1, 2))
    x33 += einsum(x31, (0, 1, 2), (0, 1, 2)) * 2.0
    x34 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x34 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x34 += einsum(v.abb.xov, (0, 1, 2), x33, (1, 3, 0), (2, 3)) * 0.5
    t2new_abab += einsum(x34, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x34
    x35 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x35 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x35 += einsum(x7, (0, 1), (1, 0))
    del x7
    t2new_abab += einsum(x35, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    del x35
    x36 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x36 += einsum(f.aa.oo, (0, 1), (0, 1)) * 2.0
    x36 += einsum(v.aaa.xov, (0, 1, 2), x6, (3, 2, 0), (1, 3)) * 2.0
    del x6
    t2new_abab += einsum(x36, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -0.5
    del x36
    x37 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x37 += einsum(f.bb.oo, (0, 1), (0, 1))
    x37 += einsum(v.abb.xov, (0, 1, 2), x33, (3, 2, 0), (1, 3)) * 0.5
    t2new_abab += einsum(x37, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x37
    x38 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x38 += einsum(v.abb.xvv, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x38, (4, 2, 5, 3), (0, 1, 4, 5)) * -2.0
    del x38
    x39 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x39 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xoo, (0, 3, 4), (1, 3, 4, 2))
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x39, (4, 0, 5, 1), (5, 4, 2, 3)) * 2.0
    del x39
    x40 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x40 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x41 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x41 += einsum(v.abb.xov, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 3, 2, 4))
    x42 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x42 += einsum(x30, (0, 1, 2), x30, (3, 4, 2), (0, 3, 1, 4))
    x43 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x43 += einsum(x40, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x40
    x43 += einsum(x41, (0, 1, 2, 3), (1, 0, 2, 3))
    x43 += einsum(x42, (0, 1, 2, 3), (1, 0, 2, 3))
    del x42
    t2new_bbbb += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x43, (0, 1, 2, 3), (0, 1, 3, 2))
    del x43
    x44 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x44 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0))
    x44 += einsum(x31, (0, 1, 2), (0, 1, 2)) * 2.0
    x45 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x45 += einsum(x30, (0, 1, 2), x44, (3, 4, 2), (0, 3, 1, 4))
    del x30, x44
    x46 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x46 += einsum(x24, (0, 1, 2, 3), (1, 0, 3, 2))
    del x24
    x46 += einsum(x41, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x41
    x47 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x47 += einsum(t2.bbbb, (0, 1, 2, 3), x46, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x46
    x48 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x48 += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    del x45
    x48 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x47
    t2new_bbbb += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x48, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x48, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb += einsum(x48, (0, 1, 2, 3), (1, 0, 3, 2))
    del x48
    x49 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x49 += einsum(x31, (0, 1, 2), x31, (3, 4, 2), (0, 3, 1, 4))
    del x31
    x50 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x50 += einsum(v.abb.xov, (0, 1, 2), x33, (1, 3, 0), (3, 2))
    x51 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x51 += einsum(x50, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 4, 0)) * -1.0
    del x50
    x52 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x52 += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3)) * -4.0
    del x49
    x52 += einsum(x51, (0, 1, 2, 3), (1, 0, 2, 3))
    del x51
    t2new_bbbb += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x52, (0, 1, 2, 3), (0, 1, 3, 2))
    del x52
    x53 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x53 += einsum(f.bb.oo, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4))
    x54 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x54 += einsum(v.abb.xov, (0, 1, 2), x33, (3, 2, 0), (3, 1))
    del x33
    x55 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x55 += einsum(x54, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    del x54
    x56 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x56 += einsum(x53, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x53
    x56 += einsum(x55, (0, 1, 2, 3), (0, 1, 3, 2))
    del x55
    t2new_bbbb += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x56, (0, 1, 2, 3), (1, 0, 2, 3))
    del x56

    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t2new": t2new}

