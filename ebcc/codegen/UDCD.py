# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    t2new = Namespace()

    # T amplitudes
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 2, 5, 3), (0, 1, 4, 5)) * 2.0
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.oooo, (4, 0, 5, 1), (4, 5, 2, 3)) * 2.0
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvoo, (4, 2, 5, 1), (0, 5, 4, 3)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.oovv, (4, 0, 5, 3), (4, 1, 2, 5)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.oooo, (4, 0, 5, 1), (4, 5, 2, 3))
    t2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.oooo, (4, 1, 5, 0), (4, 5, 2, 3)) * -2.0
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5)) * 2.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x0 += einsum(f.aa.oo, (0, 1), t2.aaaa, (2, 1, 3, 4), (0, 2, 3, 4))
    x1 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x1 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x2 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x2 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x3 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x3 += einsum(x1, (0, 1), (0, 1))
    x3 += einsum(x2, (0, 1), (0, 1)) * 0.5
    x4 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x4 += einsum(x3, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x3
    x5 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x5 += einsum(x0, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x0
    x5 += einsum(x4, (0, 1, 2, 3), (0, 1, 3, 2))
    del x4
    t2new_aaaa += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x5, (0, 1, 2, 3), (1, 0, 2, 3))
    del x5
    x6 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x6 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x7 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x7 += einsum(t2.aaaa, (0, 1, 2, 3), x6, (4, 1, 5, 3), (4, 0, 5, 2))
    x8 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x8 += einsum(t2.abab, (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x9 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x9 += einsum(t2.abab, (0, 1, 2, 3), x8, (4, 1, 5, 3), (4, 0, 5, 2))
    del x8
    x10 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x10 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 4, 1, 3), (2, 4))
    x11 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x11 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    x12 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum(x10, (0, 1), (0, 1))
    x12 += einsum(x11, (0, 1), (0, 1)) * 0.5
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x13 += einsum(x12, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 4, 0)) * -2.0
    del x12
    x14 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x14 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x7
    x14 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    del x9
    x14 += einsum(x13, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x13
    t2new_aaaa += einsum(x14, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x14, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x14
    x15 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x15 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x16 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x16 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x16 += einsum(x15, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x15
    t2new_aaaa += einsum(x16, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3))
    del x16
    x17 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x17 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x18 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x18 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x18 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x18 += einsum(x17, (0, 1, 2, 3), (1, 0, 3, 2))
    x19 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x19 += einsum(t2.aaaa, (0, 1, 2, 3), x18, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x18
    x20 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x20 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3))
    del x17
    x20 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    del x19
    t2new_aaaa += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x20, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_aaaa += einsum(x20, (0, 1, 2, 3), (1, 0, 3, 2))
    del x20
    x21 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x21 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ovov, (1, 3, 4, 5), (0, 4, 2, 5))
    t2new_abab += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x22 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x22 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x23 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x23 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x23 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x23 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (4, 1, 5, 3)) * 0.5
    x23 += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x23, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x23
    x24 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x24 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x24 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x24 += einsum(x6, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x6
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x24, (0, 4, 2, 5), (4, 1, 5, 3))
    del x24
    x25 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x25 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x25 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3))
    del x21
    t2new_abab += einsum(t2.bbbb, (0, 1, 2, 3), x25, (4, 1, 5, 3), (4, 0, 5, 2)) * 4.0
    del x25
    x26 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x26 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    x27 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x27 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x28 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x28 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x28 += einsum(x26, (0, 1), (1, 0)) * 0.5
    x28 += einsum(x27, (0, 1), (1, 0))
    t2new_abab += einsum(x28, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x28
    x29 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x29 += einsum(f.aa.vv, (0, 1), (0, 1)) * -2.0
    x29 += einsum(x10, (0, 1), (1, 0)) * 2.0
    del x10
    x29 += einsum(x11, (0, 1), (1, 0))
    del x11
    t2new_abab += einsum(x29, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -0.5
    del x29
    x30 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x30 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x31 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x31 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x32 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x32 += einsum(f.bb.oo, (0, 1), (0, 1)) * 2.0
    x32 += einsum(x30, (0, 1), (1, 0))
    x32 += einsum(x31, (0, 1), (1, 0)) * 2.0
    t2new_abab += einsum(x32, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -0.5
    del x32
    x33 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x33 += einsum(f.aa.oo, (0, 1), (0, 1))
    x33 += einsum(x1, (0, 1), (1, 0))
    del x1
    x33 += einsum(x2, (0, 1), (1, 0)) * 0.5
    del x2
    t2new_abab += einsum(x33, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    del x33
    x34 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x34 += einsum(f.bb.oo, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4))
    x35 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x35 += einsum(x30, (0, 1), (0, 1))
    del x30
    x35 += einsum(x31, (0, 1), (0, 1)) * 2.0
    del x31
    x36 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x36 += einsum(x35, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4)) * -1.0
    del x35
    x37 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x37 += einsum(x34, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x34
    x37 += einsum(x36, (0, 1, 2, 3), (1, 0, 3, 2))
    del x36
    t2new_bbbb += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x37, (0, 1, 2, 3), (1, 0, 2, 3))
    del x37
    x38 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x38 += einsum(t2.bbbb, (0, 1, 2, 3), x22, (4, 1, 5, 3), (4, 0, 5, 2))
    del x22
    x39 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x39 += einsum(x26, (0, 1), (0, 1))
    del x26
    x39 += einsum(x27, (0, 1), (0, 1)) * 2.0
    del x27
    x40 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x40 += einsum(x39, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    del x39
    x41 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x41 += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3)) * -4.0
    del x38
    x41 += einsum(x40, (0, 1, 2, 3), (1, 0, 3, 2))
    del x40
    t2new_bbbb += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x41, (0, 1, 2, 3), (0, 1, 3, 2))
    del x41
    x42 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x42 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (4, 0, 5, 2))
    x43 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x43 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x43 += einsum(x42, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x42
    x44 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x44 += einsum(t2.abab, (0, 1, 2, 3), x43, (0, 4, 2, 5), (4, 1, 5, 3))
    del x43
    x45 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x45 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x45 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x46 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x46 += einsum(t2.bbbb, (0, 1, 2, 3), x45, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    del x45
    x47 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x47 += einsum(x44, (0, 1, 2, 3), (1, 0, 3, 2))
    del x44
    x47 += einsum(x46, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x46
    t2new_bbbb += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x47, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x47, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb += einsum(x47, (0, 1, 2, 3), (1, 0, 3, 2))
    del x47
    x48 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x48 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x49 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x49 += einsum(t2.abab, (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 2), (4, 1, 5, 3))
    x50 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x50 += einsum(t2.abab, (0, 1, 2, 3), x49, (0, 4, 2, 5), (4, 1, 5, 3))
    del x49
    x51 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x51 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x51 += einsum(x48, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x48
    x51 += einsum(x50, (0, 1, 2, 3), (1, 0, 3, 2))
    del x50
    t2new_bbbb += einsum(x51, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x51, (0, 1, 2, 3), (0, 1, 2, 3))
    del x51

    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t2new": t2new}

