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
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x0, (4, 2, 5, 3), (0, 1, 4, 5)) * -2.0
    del x0
    x1 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x1 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum(v.aaa.xov, (0, 1, 2), v.aaa.xov, (0, 3, 4), (1, 3, 2, 4))
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x3 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x1
    x3 += einsum(x2, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2))
    del x3
    x4 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x4 += einsum(f.aa.oo, (0, 1), t2.aaaa, (2, 1, 3, 4), (0, 2, 3, 4))
    x5 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x5 += einsum(v.aaa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    x6 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x6 += einsum(v.abb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (3, 4, 0))
    x7 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x7 += einsum(x5, (0, 1, 2), (0, 1, 2))
    x7 += einsum(x6, (0, 1, 2), (0, 1, 2)) * 0.5
    x8 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x8 += einsum(v.aaa.xov, (0, 1, 2), x7, (3, 2, 0), (3, 1))
    x9 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x9 += einsum(x8, (0, 1), t2.aaaa, (2, 1, 3, 4), (0, 2, 3, 4)) * -4.0
    del x8
    x10 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x10 += einsum(x4, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x4
    x10 += einsum(x9, (0, 1, 2, 3), (1, 0, 3, 2))
    del x9
    t2new_aaaa += einsum(x10, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x10, (0, 1, 2, 3), (1, 0, 2, 3))
    del x10
    x11 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x11 += einsum(x2, (0, 1, 2, 3), (1, 0, 2, 3))
    x11 += einsum(x2, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x12 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum(t2.aaaa, (0, 1, 2, 3), x11, (1, 4, 3, 5), (4, 0, 5, 2))
    del x11
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x13 += einsum(t2.aaaa, (0, 1, 2, 3), x12, (1, 4, 3, 5), (4, 0, 5, 2)) * -4.0
    del x12
    x14 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x14 += einsum(v.abb.xov, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 3, 2, 4))
    x15 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x15 += einsum(x14, (0, 1, 2, 3), (1, 0, 2, 3))
    x15 += einsum(x14, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x16 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x16 += einsum(t2.abab, (0, 1, 2, 3), x15, (1, 4, 3, 5), (0, 4, 2, 5))
    x17 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x17 += einsum(t2.abab, (0, 1, 2, 3), x16, (4, 1, 5, 3), (4, 0, 5, 2)) * -1.0
    del x16
    x18 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x18 += einsum(v.aaa.xov, (0, 1, 2), x7, (1, 3, 0), (3, 2))
    x19 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x19 += einsum(x18, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4)) * -4.0
    del x18
    x20 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x20 += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x13
    x20 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x17
    x20 += einsum(x19, (0, 1, 2, 3), (1, 0, 3, 2))
    del x19
    t2new_aaaa += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x20, (0, 1, 2, 3), (0, 1, 3, 2))
    del x20
    x21 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x21 += einsum(v.aaa.xov, (0, 1, 2), x6, (3, 4, 0), (3, 1, 4, 2))
    x22 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x22 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 2, 3, 4))
    x23 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x23 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x23 += einsum(x6, (0, 1, 2), (0, 1, 2))
    x24 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x24 += einsum(v.aaa.xov, (0, 1, 2), x23, (3, 4, 0), (3, 1, 4, 2))
    del x23
    x25 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x25 += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x25 += einsum(x24, (0, 1, 2, 3), (1, 0, 3, 2))
    del x24
    x26 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x26 += einsum(t2.aaaa, (0, 1, 2, 3), x25, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    del x25
    x27 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x27 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3))
    del x21
    x27 += einsum(x26, (0, 1, 2, 3), (1, 0, 3, 2))
    del x26
    t2new_aaaa += einsum(x27, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x27, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x27, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_aaaa += einsum(x27, (0, 1, 2, 3), (1, 0, 3, 2))
    del x27
    x28 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x28 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xoo, (0, 3, 4), (1, 3, 4, 2))
    x28 += einsum(t2.aaaa, (0, 1, 2, 3), x2, (4, 5, 2, 3), (5, 0, 4, 1))
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x28, (0, 4, 1, 5), (4, 5, 2, 3)) * -2.0
    del x28
    x29 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x29 += einsum(v.aaa.xoo, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x29, (4, 0, 5, 3), (4, 1, 2, 5)) * -1.0
    del x29
    x30 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x30 += einsum(v.aaa.xvv, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x30, (4, 2, 5, 3), (0, 1, 4, 5))
    del x30
    x31 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x31 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x31 += einsum(x5, (0, 1, 2), (0, 1, 2)) * 2.0
    del x5
    x31 += einsum(x6, (0, 1, 2), (0, 1, 2))
    del x6
    x32 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x32 += einsum(v.aaa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    x33 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x33 += einsum(v.abb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    x34 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x34 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0))
    x34 += einsum(x32, (0, 1, 2), (0, 1, 2))
    x34 += einsum(x33, (0, 1, 2), (0, 1, 2)) * 2.0
    t2new_abab += einsum(x31, (0, 1, 2), x34, (3, 4, 2), (0, 3, 1, 4))
    del x31, x34
    x35 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x35 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    x36 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x36 += einsum(x35, (0, 1, 2, 3), (1, 0, 3, 2))
    x36 += einsum(t2.bbbb, (0, 1, 2, 3), x14, (1, 4, 5, 3), (4, 0, 5, 2)) * 2.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x36, (1, 4, 3, 5), (0, 4, 2, 5)) * -1.0
    del x36
    x37 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x37 += einsum(v.aaa.xov, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 3, 2, 4))
    x38 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x38 += einsum(v.abb.xoo, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 2, 3, 4))
    x38 += einsum(t2.abab, (0, 1, 2, 3), x37, (0, 4, 5, 3), (4, 1, 5, 2)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x38, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    del x38
    x39 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x39 += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2))
    del x22
    x39 += einsum(t2.aaaa, (0, 1, 2, 3), x2, (4, 1, 3, 5), (4, 0, 5, 2)) * 2.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x39, (0, 4, 2, 5), (4, 1, 5, 3)) * -1.0
    del x39
    x40 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x40 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x40 += einsum(v.aaa.xov, (0, 1, 2), x7, (1, 3, 0), (2, 3)) * 2.0
    t2new_abab += einsum(x40, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    del x40
    x41 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x41 += einsum(x32, (0, 1, 2), (0, 1, 2))
    x41 += einsum(x33, (0, 1, 2), (0, 1, 2)) * 2.0
    x42 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x42 += einsum(v.abb.xov, (0, 1, 2), x41, (1, 3, 0), (3, 2))
    x43 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x43 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x43 += einsum(x42, (0, 1), (1, 0))
    t2new_abab += einsum(x43, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x43
    x44 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x44 += einsum(v.aaa.xoo, (0, 1, 2), v.abb.xoo, (0, 3, 4), (1, 2, 3, 4))
    x44 += einsum(t2.abab, (0, 1, 2, 3), x37, (4, 5, 2, 3), (4, 0, 5, 1))
    del x37
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x44, (0, 4, 1, 5), (4, 5, 2, 3))
    del x44
    x45 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x45 += einsum(f.aa.oo, (0, 1), (0, 1))
    x45 += einsum(v.aaa.xov, (0, 1, 2), x7, (3, 2, 0), (1, 3)) * 2.0
    del x7
    t2new_abab += einsum(x45, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    del x45
    x46 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x46 += einsum(v.abb.xov, (0, 1, 2), x41, (3, 2, 0), (3, 1))
    del x41
    x47 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x47 += einsum(f.bb.oo, (0, 1), (0, 1))
    x47 += einsum(x46, (0, 1), (1, 0))
    t2new_abab += einsum(x47, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x47
    x48 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x48 += einsum(v.abb.xvv, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 3, 4, 2))
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x48, (4, 2, 5, 3), (0, 1, 4, 5)) * -2.0
    del x48
    x49 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x49 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0))
    x49 += einsum(x33, (0, 1, 2), (0, 1, 2)) * 2.0
    del x33
    x50 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x50 += einsum(x32, (0, 1, 2), x49, (3, 4, 2), (0, 3, 1, 4))
    del x32, x49
    x51 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x51 += einsum(x35, (0, 1, 2, 3), (1, 0, 3, 2))
    del x35
    x51 += einsum(x14, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x52 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x52 += einsum(t2.bbbb, (0, 1, 2, 3), x51, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x51
    x53 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x53 += einsum(x50, (0, 1, 2, 3), (0, 1, 2, 3))
    del x50
    x53 += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x52
    t2new_bbbb += einsum(x53, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x53, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x53, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb += einsum(x53, (0, 1, 2, 3), (1, 0, 3, 2))
    del x53
    x54 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x54 += einsum(f.bb.oo, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4))
    x55 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x55 += einsum(x46, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x46
    x56 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x56 += einsum(x54, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x54
    x56 += einsum(x55, (0, 1, 2, 3), (0, 1, 3, 2))
    del x55
    t2new_bbbb += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x56, (0, 1, 2, 3), (1, 0, 2, 3))
    del x56
    x57 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x57 += einsum(t2.bbbb, (0, 1, 2, 3), x15, (1, 4, 3, 5), (4, 0, 5, 2))
    del x15
    x58 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x58 += einsum(t2.bbbb, (0, 1, 2, 3), x57, (1, 4, 3, 5), (0, 4, 2, 5)) * -4.0
    del x57
    x59 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x59 += einsum(x42, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 4, 0)) * -2.0
    del x42
    x60 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x60 += einsum(x58, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x58
    x60 += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3))
    del x59
    t2new_bbbb += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x60, (0, 1, 2, 3), (0, 1, 3, 2))
    del x60
    x61 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x61 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x62 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x62 += einsum(x2, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x62 += einsum(x2, (0, 1, 2, 3), (1, 0, 3, 2))
    del x2
    x63 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x63 += einsum(t2.abab, (0, 1, 2, 3), x62, (0, 4, 5, 2), (4, 1, 5, 3))
    del x62
    x64 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x64 += einsum(t2.abab, (0, 1, 2, 3), x63, (0, 4, 2, 5), (4, 1, 5, 3)) * -1.0
    del x63
    x65 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x65 += einsum(x61, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x61
    x65 += einsum(x14, (0, 1, 2, 3), (1, 0, 2, 3))
    x65 += einsum(x64, (0, 1, 2, 3), (0, 1, 3, 2))
    del x64
    t2new_bbbb += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x65, (0, 1, 2, 3), (0, 1, 3, 2))
    del x65
    x66 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x66 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xoo, (0, 3, 4), (1, 3, 4, 2))
    x66 += einsum(t2.bbbb, (0, 1, 2, 3), x14, (4, 5, 2, 3), (5, 0, 4, 1))
    del x14
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x66, (0, 4, 1, 5), (4, 5, 2, 3)) * -2.0
    del x66

    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, naux=None, t2=None, l2=None, **kwargs):
    l2new = Namespace()

    # L amplitudes
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x0 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), (2, 3, 4, 5))
    x1 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x1 += einsum(v.aaa.xov, (0, 1, 2), v.aaa.xov, (0, 3, 4), (1, 3, 2, 4))
    l2new_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    l2new_aaaa += einsum(x0, (0, 1, 2, 3), x1, (2, 3, 4, 5), (5, 4, 1, 0)) * 2.0
    del x0
    x2 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum(v.aaa.xvv, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 3, 4, 2))
    l2new_aaaa += einsum(l2.aaaa, (0, 1, 2, 3), x2, (4, 1, 5, 0), (5, 4, 2, 3)) * -2.0
    del x2
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x3 += einsum(f.aa.vv, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    x4 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x4 += einsum(v.aaa.xov, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (3, 4, 0))
    x5 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x5 += einsum(v.abb.xov, (0, 1, 2), t2.abab, (3, 1, 4, 2), (3, 4, 0))
    x6 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x6 += einsum(x4, (0, 1, 2), (0, 1, 2))
    x6 += einsum(x5, (0, 1, 2), (0, 1, 2)) * 0.5
    x7 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x7 += einsum(v.aaa.xov, (0, 1, 2), x6, (1, 3, 0), (2, 3))
    x8 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x8 += einsum(x7, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2)) * -4.0
    del x7
    x9 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x9 += einsum(x3, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x3
    x9 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x9 += einsum(x8, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x8
    l2new_aaaa += einsum(x9, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_aaaa += einsum(x9, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    del x9
    x10 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x10 += einsum(f.aa.oo, (0, 1), l2.aaaa, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_aaaa += einsum(x10, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    l2new_aaaa += einsum(x10, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    del x10
    x11 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x11 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 4, 0, 5))
    x12 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum(x1, (0, 1, 2, 3), x11, (4, 1, 5, 2), (4, 0, 5, 3))
    del x11
    x13 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x13 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x13 += einsum(x4, (0, 1, 2), (0, 1, 2)) * 2.0
    del x4
    x13 += einsum(x5, (0, 1, 2), (0, 1, 2))
    del x5
    x14 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x14 += einsum(x13, (0, 1, 2), l2.aaaa, (3, 1, 4, 0), (4, 3, 2))
    x15 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x15 += einsum(v.aaa.xov, (0, 1, 2), t2.abab, (1, 3, 2, 4), (3, 4, 0))
    x16 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x16 += einsum(v.abb.xov, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (3, 4, 0))
    x17 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x17 += einsum(v.abb.xov, (0, 1, 2), (1, 2, 0))
    x17 += einsum(x15, (0, 1, 2), (0, 1, 2))
    x17 += einsum(x16, (0, 1, 2), (0, 1, 2)) * 2.0
    x18 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x18 += einsum(x17, (0, 1, 2), l2.abab, (3, 1, 4, 0), (4, 3, 2)) * 0.5
    x19 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x19 += einsum(x14, (0, 1, 2), (0, 1, 2))
    del x14
    x19 += einsum(x18, (0, 1, 2), (0, 1, 2))
    del x18
    x20 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x20 += einsum(v.aaa.xov, (0, 1, 2), x19, (3, 4, 0), (1, 3, 2, 4)) * 2.0
    del x19
    x21 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x21 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 2, 3, 4))
    x22 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x22 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (4, 1, 3, 5), (0, 4, 2, 5))
    x23 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x23 += einsum(x21, (0, 1, 2, 3), (1, 0, 3, 2))
    x23 += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x24 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x24 += einsum(l2.aaaa, (0, 1, 2, 3), x23, (3, 4, 1, 5), (4, 2, 5, 0)) * 2.0
    del x23
    x25 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x25 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    x26 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x26 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    x27 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x27 += einsum(x25, (0, 1), (0, 1))
    del x25
    x27 += einsum(x26, (0, 1), (0, 1)) * 0.5
    del x26
    x28 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x28 += einsum(x27, (0, 1), v.aaa.xov, (2, 1, 3), (0, 3, 2))
    x29 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x29 += einsum(v.aaa.xov, (0, 1, 2), x28, (3, 4, 0), (1, 3, 2, 4)) * -2.0
    del x28
    x30 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x30 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4))
    x31 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x31 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    x32 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x32 += einsum(x30, (0, 1), (0, 1))
    del x30
    x32 += einsum(x31, (0, 1), (0, 1)) * 0.5
    del x31
    x33 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x33 += einsum(x32, (0, 1), v.aaa.xov, (2, 3, 1), (3, 0, 2)) * 2.0
    del x32
    x34 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x34 += einsum(v.aaa.xov, (0, 1, 2), x33, (3, 4, 0), (1, 3, 2, 4)) * -1.0
    x35 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x35 += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x12
    x35 += einsum(x20, (0, 1, 2, 3), (1, 0, 3, 2))
    del x20
    x35 += einsum(x24, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x24
    x35 += einsum(x29, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x29
    x35 += einsum(x34, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x34
    l2new_aaaa += einsum(x35, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_aaaa += einsum(x35, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_aaaa += einsum(x35, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new_aaaa += einsum(x35, (0, 1, 2, 3), (3, 2, 1, 0))
    del x35
    x36 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x36 += einsum(v.aaa.xov, (0, 1, 2), x6, (3, 2, 0), (1, 3))
    x37 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x37 += einsum(x36, (0, 1), l2.aaaa, (2, 3, 4, 1), (0, 4, 2, 3)) * -4.0
    del x36
    l2new_aaaa += einsum(x37, (0, 1, 2, 3), (2, 3, 1, 0))
    l2new_aaaa += einsum(x37, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    del x37
    x38 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x38 += einsum(v.aaa.xoo, (0, 1, 2), v.aaa.xoo, (0, 3, 4), (1, 3, 4, 2))
    x38 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (4, 5, 2, 3), (0, 1, 5, 4))
    l2new_aaaa += einsum(l2.aaaa, (0, 1, 2, 3), x38, (3, 2, 4, 5), (0, 1, 5, 4)) * -2.0
    del x38
    x39 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x39 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    l2new_abab = np.zeros((nvir[0], nvir[1], nocc[0], nocc[1]), dtype=types[float])
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x39, (4, 3, 5, 1), (0, 5, 2, 4)) * -1.0
    x40 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x40 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), (2, 4, 3, 5))
    x41 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x41 += einsum(v.aaa.xov, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 3, 2, 4))
    l2new_abab += einsum(x40, (0, 1, 2, 3), x41, (1, 3, 4, 5), (4, 5, 0, 2))
    del x40
    x42 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x42 += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (4, 2, 5, 0))
    l2new_abab += einsum(x1, (0, 1, 2, 3), x42, (1, 4, 2, 5), (3, 5, 0, 4)) * -2.0
    del x1, x42
    x43 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x43 += einsum(v.aaa.xvv, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x43, (4, 0, 5, 1), (4, 5, 2, 3))
    del x43
    x44 = np.zeros((nocc[0], nvir[0], naux), dtype=types[float])
    x44 += einsum(v.aaa.xov, (0, 1, 2), (1, 2, 0))
    x44 += einsum(x13, (0, 1, 2), l2.aaaa, (3, 1, 4, 0), (4, 3, 2)) * 2.0
    x44 += einsum(x17, (0, 1, 2), l2.abab, (3, 1, 4, 0), (4, 3, 2))
    x44 += einsum(x33, (0, 1, 2), (0, 1, 2)) * -1.0
    del x33
    x44 += einsum(x27, (0, 1), v.aaa.xov, (2, 1, 3), (0, 3, 2)) * -2.0
    del x27
    l2new_abab += einsum(v.abb.xov, (0, 1, 2), x44, (3, 4, 0), (4, 2, 3, 1))
    del x44
    x45 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x45 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    x46 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x46 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4))
    x47 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x47 += einsum(x45, (0, 1), (0, 1))
    del x45
    x47 += einsum(x46, (0, 1), (0, 1)) * 2.0
    del x46
    x48 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x48 += einsum(x47, (0, 1), v.abb.xov, (2, 3, 1), (3, 0, 2)) * 0.5
    del x47
    x49 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x49 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    x50 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x50 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    x51 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x51 += einsum(x49, (0, 1), (0, 1))
    del x49
    x51 += einsum(x50, (0, 1), (0, 1)) * 2.0
    del x50
    x52 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x52 += einsum(x13, (0, 1, 2), l2.abab, (1, 3, 0, 4), (4, 3, 2)) * 0.5
    x52 += einsum(x17, (0, 1, 2), l2.bbbb, (3, 1, 4, 0), (4, 3, 2))
    x52 += einsum(x48, (0, 1, 2), (0, 1, 2)) * -1.0
    x52 += einsum(x51, (0, 1), v.abb.xov, (2, 1, 3), (0, 3, 2)) * -0.5
    l2new_abab += einsum(v.aaa.xov, (0, 1, 2), x52, (3, 4, 0), (2, 4, 1, 3)) * 2.0
    del x52
    x53 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x53 += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 1, 5), (2, 4, 0, 5))
    x53 += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (2, 4, 0, 5))
    x54 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x54 += einsum(v.abb.xov, (0, 1, 2), v.abb.xov, (0, 3, 4), (1, 3, 2, 4))
    l2new_abab += einsum(x53, (0, 1, 2, 3), x54, (1, 4, 5, 3), (2, 5, 0, 4)) * -2.0
    del x53
    x55 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x55 += einsum(v.abb.xoo, (0, 1, 2), v.aaa.xvv, (0, 3, 4), (1, 2, 3, 4))
    x55 += einsum(t2.abab, (0, 1, 2, 3), x41, (0, 4, 5, 3), (1, 4, 2, 5)) * -1.0
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x55, (3, 4, 0, 5), (5, 1, 2, 4)) * -1.0
    del x55
    x56 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x56 += einsum(x21, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    del x21
    x56 += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3))
    del x22
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x56, (2, 4, 0, 5), (5, 1, 4, 3)) * -2.0
    del x56
    x57 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x57 += einsum(v.aaa.xoo, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 2, 3, 4))
    x57 += einsum(t2.abab, (0, 1, 2, 3), x41, (4, 1, 2, 5), (0, 4, 3, 5)) * -1.0
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x57, (2, 4, 1, 5), (0, 5, 4, 3)) * -1.0
    del x57
    x58 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x58 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x58 += einsum(v.aaa.xov, (0, 1, 2), x6, (1, 3, 0), (3, 2)) * 2.0
    l2new_abab += einsum(x58, (0, 1), l2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    del x58
    x59 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x59 += einsum(x15, (0, 1, 2), (0, 1, 2))
    del x15
    x59 += einsum(x16, (0, 1, 2), (0, 1, 2)) * 2.0
    del x16
    x60 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x60 += einsum(f.bb.vv, (0, 1), (0, 1)) * -0.5
    x60 += einsum(v.abb.xov, (0, 1, 2), x59, (1, 3, 0), (3, 2)) * 0.5
    l2new_abab += einsum(x60, (0, 1), l2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x60
    x61 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x61 += einsum(v.aaa.xoo, (0, 1, 2), v.abb.xoo, (0, 3, 4), (1, 2, 3, 4))
    x61 += einsum(t2.abab, (0, 1, 2, 3), x41, (4, 5, 2, 3), (0, 4, 1, 5))
    del x41
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x61, (2, 4, 3, 5), (0, 1, 4, 5))
    del x61
    x62 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x62 += einsum(f.aa.oo, (0, 1), (0, 1))
    x62 += einsum(v.aaa.xov, (0, 1, 2), x6, (3, 2, 0), (3, 1)) * 2.0
    del x6
    l2new_abab += einsum(x62, (0, 1), l2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    del x62
    x63 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x63 += einsum(v.abb.xov, (0, 1, 2), x59, (3, 2, 0), (1, 3))
    x64 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x64 += einsum(f.bb.oo, (0, 1), (0, 1))
    x64 += einsum(x63, (0, 1), (1, 0))
    l2new_abab += einsum(x64, (0, 1), l2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x64
    x65 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x65 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), (2, 3, 4, 5))
    l2new_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    l2new_bbbb += einsum(x54, (0, 1, 2, 3), x65, (4, 5, 0, 1), (3, 2, 5, 4)) * 2.0
    del x65
    x66 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x66 += einsum(v.abb.xvv, (0, 1, 2), v.abb.xvv, (0, 3, 4), (1, 3, 4, 2))
    l2new_bbbb += einsum(l2.bbbb, (0, 1, 2, 3), x66, (4, 1, 5, 0), (5, 4, 2, 3)) * -2.0
    del x66
    x67 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x67 += einsum(l2.bbbb, (0, 1, 2, 3), x39, (4, 3, 5, 1), (2, 4, 0, 5))
    del x39
    x68 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x68 += einsum(x13, (0, 1, 2), l2.abab, (1, 3, 0, 4), (4, 3, 2))
    del x13
    x69 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x69 += einsum(x17, (0, 1, 2), l2.bbbb, (3, 1, 4, 0), (4, 3, 2)) * 2.0
    del x17
    x70 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x70 += einsum(x68, (0, 1, 2), (0, 1, 2))
    del x68
    x70 += einsum(x69, (0, 1, 2), (0, 1, 2))
    del x69
    x71 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x71 += einsum(v.abb.xov, (0, 1, 2), x70, (3, 4, 0), (1, 3, 2, 4))
    del x70
    x72 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x72 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), (3, 4, 1, 5))
    x73 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x73 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (2, 4, 0, 5))
    x74 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x74 += einsum(x72, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x72
    x74 += einsum(x73, (0, 1, 2, 3), (0, 1, 2, 3))
    del x73
    x75 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x75 += einsum(x54, (0, 1, 2, 3), x74, (4, 0, 5, 3), (4, 1, 5, 2)) * 4.0
    del x74
    x76 = np.zeros((nocc[1], nvir[1], naux), dtype=types[float])
    x76 += einsum(x51, (0, 1), v.abb.xov, (2, 1, 3), (0, 3, 2))
    del x51
    x77 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x77 += einsum(v.abb.xov, (0, 1, 2), x76, (3, 4, 0), (1, 3, 2, 4)) * -1.0
    del x76
    x78 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x78 += einsum(v.abb.xov, (0, 1, 2), x48, (3, 4, 0), (1, 3, 2, 4)) * -2.0
    del x48
    x79 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x79 += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x67
    x79 += einsum(x71, (0, 1, 2, 3), (1, 0, 3, 2))
    del x71
    x79 += einsum(x75, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x75
    x79 += einsum(x77, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x77
    x79 += einsum(x78, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x78
    l2new_bbbb += einsum(x79, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_bbbb += einsum(x79, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_bbbb += einsum(x79, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new_bbbb += einsum(x79, (0, 1, 2, 3), (3, 2, 1, 0))
    del x79
    x80 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x80 += einsum(f.bb.vv, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    x81 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x81 += einsum(v.abb.xov, (0, 1, 2), x59, (1, 3, 0), (2, 3))
    del x59
    x82 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x82 += einsum(x81, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 2, 0)) * -2.0
    del x81
    x83 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x83 += einsum(x80, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x80
    x83 += einsum(x54, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x83 += einsum(x82, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x82
    l2new_bbbb += einsum(x83, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_bbbb += einsum(x83, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    del x83
    x84 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x84 += einsum(x63, (0, 1), l2.bbbb, (2, 3, 4, 1), (4, 0, 2, 3)) * -2.0
    del x63
    l2new_bbbb += einsum(x84, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_bbbb += einsum(x84, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    del x84
    x85 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x85 += einsum(f.bb.oo, (0, 1), l2.bbbb, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_bbbb += einsum(x85, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    l2new_bbbb += einsum(x85, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    del x85
    x86 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x86 += einsum(v.abb.xoo, (0, 1, 2), v.abb.xoo, (0, 3, 4), (1, 3, 4, 2))
    x86 += einsum(t2.bbbb, (0, 1, 2, 3), x54, (4, 5, 2, 3), (0, 1, 5, 4))
    del x54
    l2new_bbbb += einsum(l2.bbbb, (0, 1, 2, 3), x86, (3, 2, 4, 5), (0, 1, 5, 4)) * -2.0
    del x86

    l2new.aaaa = l2new_aaaa
    l2new.abab = l2new_abab
    l2new.bbbb = l2new_bbbb

    return {"l2new": l2new}

