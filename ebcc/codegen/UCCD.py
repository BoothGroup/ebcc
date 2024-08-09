# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 2), ()) * -1.0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    t2new = Namespace()

    # T amplitudes
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.oovv, (4, 0, 5, 3), (4, 1, 2, 5)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    x0 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x0 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ovov, (1, 3, 4, 5), (4, 5, 0, 2))
    x1 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x1 += einsum(x0, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x0
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum(t2.abab, (0, 1, 2, 3), x1, (1, 3, 4, 5), (0, 4, 2, 5))
    del x1
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x3 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x3 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x4 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x4 += einsum(t2.aaaa, (0, 1, 2, 3), x3, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x3
    x5 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x5 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3))
    del x2
    x5 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x4
    t2new_aaaa += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x5, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x5, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_aaaa += einsum(x5, (0, 1, 2, 3), (1, 0, 3, 2))
    del x5
    x6 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x6 += einsum(f.aa.oo, (0, 1), t2.aaaa, (2, 1, 3, 4), (0, 2, 3, 4))
    x7 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x7 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x8 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x8 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x9 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x9 += einsum(x7, (0, 1), (0, 1))
    x9 += einsum(x8, (0, 1), (0, 1)) * 2.0
    x10 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x10 += einsum(x9, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x9
    x11 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x11 += einsum(x6, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x6
    x11 += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2))
    del x10
    t2new_aaaa += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x11, (0, 1, 2, 3), (1, 0, 2, 3))
    del x11
    x12 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x12 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x13 += einsum(t2.aaaa, (0, 1, 2, 3), x12, (1, 4, 3, 5), (4, 0, 5, 2))
    x14 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x14 += einsum(t2.aaaa, (0, 1, 2, 3), x13, (1, 4, 3, 5), (4, 0, 5, 2)) * -4.0
    del x13
    x15 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x15 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    x16 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x16 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 4, 1, 3), (2, 4))
    x17 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x17 += einsum(x15, (0, 1), (0, 1))
    x17 += einsum(x16, (0, 1), (0, 1)) * 2.0
    x18 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x18 += einsum(x17, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4)) * -2.0
    del x17
    x19 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x19 += einsum(x14, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x14
    x19 += einsum(x18, (0, 1, 2, 3), (1, 0, 3, 2))
    del x18
    t2new_aaaa += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x19, (0, 1, 2, 3), (0, 1, 3, 2))
    del x19
    x20 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x20 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x21 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x21 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x21 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x22 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x22 += einsum(t2.abab, (0, 1, 2, 3), x21, (1, 4, 3, 5), (4, 5, 0, 2))
    x23 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x23 += einsum(t2.abab, (0, 1, 2, 3), x22, (1, 3, 4, 5), (4, 0, 5, 2)) * -1.0
    del x22
    x24 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x24 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x24 += einsum(x20, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x20
    x24 += einsum(x23, (0, 1, 2, 3), (1, 0, 3, 2))
    del x23
    t2new_aaaa += einsum(x24, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    del x24
    x25 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x25 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x25 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 5, 3), (4, 0, 1, 5))
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x25, (0, 4, 5, 1), (5, 4, 2, 3)) * -2.0
    del x25
    x26 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x26 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 2, 4, 5))
    t2new_abab += einsum(x26, (0, 1, 2, 3), (2, 0, 3, 1)) * 2.0
    x27 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x27 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x27 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x28 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x28 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x28 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x28 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (4, 0, 5, 2))
    x28 += einsum(t2.aaaa, (0, 1, 2, 3), x27, (1, 4, 5, 3), (4, 0, 5, 2)) * -2.0
    del x27
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x28, (0, 4, 2, 5), (4, 1, 5, 3))
    del x28
    x29 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x29 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x29 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x29 += einsum(t2.bbbb, (0, 1, 2, 3), x21, (1, 4, 5, 3), (4, 0, 5, 2)) * -1.0
    del x21
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x29, (1, 4, 3, 5), (0, 4, 2, 5)) * -2.0
    del x29
    x30 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x30 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x30 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 5, 3), (5, 1, 4, 2)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x30, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    del x30
    x31 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x31 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1)) * 0.5
    x31 += einsum(x26, (0, 1, 2, 3), (0, 1, 2, 3))
    del x26
    t2new_abab += einsum(t2.aaaa, (0, 1, 2, 3), x31, (4, 5, 1, 3), (0, 4, 2, 5)) * 4.0
    del x31
    x32 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x32 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x32 += einsum(x15, (0, 1), (1, 0))
    del x15
    x32 += einsum(x16, (0, 1), (1, 0)) * 2.0
    del x16
    t2new_abab += einsum(x32, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    del x32
    x33 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x33 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 4, 1, 3), (2, 4))
    x34 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x34 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    x35 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x35 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x35 += einsum(x33, (0, 1), (1, 0)) * 2.0
    x35 += einsum(x34, (0, 1), (1, 0))
    t2new_abab += einsum(x35, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x35
    x36 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x36 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x36 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (5, 1, 4, 0))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x36, (1, 4, 0, 5), (5, 4, 2, 3))
    del x36
    x37 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x37 += einsum(f.aa.oo, (0, 1), (0, 1))
    x37 += einsum(x7, (0, 1), (1, 0))
    del x7
    x37 += einsum(x8, (0, 1), (1, 0)) * 2.0
    del x8
    t2new_abab += einsum(x37, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    del x37
    x38 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x38 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x39 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x39 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x40 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x40 += einsum(f.bb.oo, (0, 1), (0, 1))
    x40 += einsum(x38, (0, 1), (1, 0)) * 2.0
    x40 += einsum(x39, (0, 1), (1, 0))
    t2new_abab += einsum(x40, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x40
    x41 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x41 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 4, 3, 5))
    x42 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x42 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x42 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x42 += einsum(x41, (0, 1, 2, 3), (1, 0, 3, 2))
    x43 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x43 += einsum(t2.bbbb, (0, 1, 2, 3), x42, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x42
    x44 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x44 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3))
    del x41
    x44 += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3))
    del x43
    t2new_bbbb += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x44, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x44, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb += einsum(x44, (0, 1, 2, 3), (1, 0, 3, 2))
    del x44
    x45 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x45 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x46 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x46 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x46 += einsum(x45, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x45
    t2new_bbbb += einsum(x46, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x46, (0, 1, 2, 3), (0, 1, 2, 3))
    del x46
    x47 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x47 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x47 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x48 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x48 += einsum(t2.bbbb, (0, 1, 2, 3), x47, (1, 4, 5, 3), (0, 4, 2, 5))
    del x47
    x49 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x49 += einsum(t2.bbbb, (0, 1, 2, 3), x48, (4, 1, 5, 3), (0, 4, 2, 5)) * -4.0
    del x48
    x50 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x50 += einsum(t2.abab, (0, 1, 2, 3), x12, (0, 4, 2, 5), (1, 3, 4, 5))
    del x12
    x51 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x51 += einsum(t2.abab, (0, 1, 2, 3), x50, (4, 5, 0, 2), (4, 1, 5, 3)) * -1.0
    del x50
    x52 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x52 += einsum(x33, (0, 1), (0, 1))
    del x33
    x52 += einsum(x34, (0, 1), (0, 1)) * 0.5
    del x34
    x53 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x53 += einsum(x52, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 4, 0)) * -4.0
    del x52
    x54 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x54 += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x49
    x54 += einsum(x51, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x51
    x54 += einsum(x53, (0, 1, 2, 3), (1, 0, 2, 3))
    del x53
    t2new_bbbb += einsum(x54, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x54, (0, 1, 2, 3), (0, 1, 3, 2))
    del x54
    x55 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x55 += einsum(f.bb.oo, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4))
    x56 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x56 += einsum(x38, (0, 1), (0, 1))
    del x38
    x56 += einsum(x39, (0, 1), (0, 1)) * 0.5
    del x39
    x57 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x57 += einsum(x56, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4)) * -4.0
    del x56
    x58 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x58 += einsum(x55, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x55
    x58 += einsum(x57, (0, 1, 2, 3), (0, 1, 3, 2))
    del x57
    t2new_bbbb += einsum(x58, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x58, (0, 1, 2, 3), (1, 0, 2, 3))
    del x58
    x59 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x59 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x59 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 5, 2), (4, 0, 5, 1)) * -1.0
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x59, (0, 4, 1, 5), (4, 5, 2, 3)) * 2.0
    del x59

    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, t2=None, l2=None, **kwargs):
    l2new = Namespace()

    # L amplitudes
    l2new_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    l2new_aaaa += einsum(l2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 1, 5, 0), (4, 5, 2, 3)) * -2.0
    l2new_abab = np.zeros((nvir[0], nvir[1], nocc[0], nocc[1]), dtype=types[float])
    l2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (1, 3, 0, 2))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 0, 5, 1), (4, 5, 2, 3))
    l2new_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    l2new_bbbb += einsum(l2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 1, 5, 0), (4, 5, 2, 3)) * -2.0
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x0 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), (2, 3, 4, 5))
    l2new_aaaa += einsum(v.aaaa.ovov, (0, 1, 2, 3), x0, (4, 5, 0, 2), (3, 1, 4, 5)) * -2.0
    del x0
    x1 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x1 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x3 += einsum(t2.aaaa, (0, 1, 2, 3), x2, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    x4 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x4 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x4 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x4 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    del x1
    x4 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x3
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x4, (2, 4, 0, 5), (5, 1, 4, 3))
    x5 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x5 += einsum(l2.aaaa, (0, 1, 2, 3), x4, (3, 4, 1, 5), (4, 2, 5, 0)) * 2.0
    del x4
    x6 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 2, 4, 5))
    x7 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x7 += einsum(t2.abab, (0, 1, 2, 3), x2, (0, 4, 2, 5), (1, 3, 4, 5))
    del x2
    x8 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x8 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x6
    x8 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x7
    l2new_abab += einsum(l2.bbbb, (0, 1, 2, 3), x8, (3, 1, 4, 5), (5, 0, 4, 2)) * 2.0
    x9 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x9 += einsum(l2.abab, (0, 1, 2, 3), x8, (3, 1, 4, 5), (4, 2, 5, 0))
    del x8
    x10 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x10 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    x11 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x11 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4))
    x12 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum(x10, (0, 1), (0, 1))
    x12 += einsum(x11, (0, 1), (0, 1)) * 2.0
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x13 += einsum(x12, (0, 1), v.aaaa.ovov, (2, 1, 3, 4), (2, 3, 0, 4))
    del x12
    x14 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x14 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    x15 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x15 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    x16 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x16 += einsum(x14, (0, 1), (0, 1))
    x16 += einsum(x15, (0, 1), (0, 1)) * 2.0
    x17 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x17 += einsum(x16, (0, 1), v.aaaa.ovov, (2, 3, 1, 4), (0, 2, 4, 3))
    del x16
    x18 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x18 += einsum(x5, (0, 1, 2, 3), (1, 0, 3, 2))
    del x5
    x18 += einsum(x9, (0, 1, 2, 3), (1, 0, 3, 2))
    del x9
    x18 += einsum(x13, (0, 1, 2, 3), (1, 0, 2, 3))
    del x13
    x18 += einsum(x17, (0, 1, 2, 3), (0, 1, 3, 2))
    del x17
    l2new_aaaa += einsum(x18, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_aaaa += einsum(x18, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_aaaa += einsum(x18, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new_aaaa += einsum(x18, (0, 1, 2, 3), (3, 2, 1, 0))
    del x18
    x19 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x19 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x20 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x20 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 1, 3), (0, 4))
    x21 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x21 += einsum(x19, (0, 1), (0, 1))
    x21 += einsum(x20, (0, 1), (0, 1)) * 2.0
    x22 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x22 += einsum(x21, (0, 1), l2.aaaa, (2, 3, 4, 0), (1, 4, 2, 3)) * -2.0
    del x21
    l2new_aaaa += einsum(x22, (0, 1, 2, 3), (2, 3, 1, 0))
    l2new_aaaa += einsum(x22, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    del x22
    x23 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x23 += einsum(f.aa.vv, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    x24 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x24 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    x25 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x25 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x26 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x26 += einsum(x24, (0, 1), (0, 1))
    x26 += einsum(x25, (0, 1), (0, 1)) * 2.0
    x27 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x27 += einsum(x26, (0, 1), l2.aaaa, (2, 0, 3, 4), (3, 4, 1, 2)) * -2.0
    del x26
    x28 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x28 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x28 += einsum(x23, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x23
    x28 += einsum(x27, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x27
    l2new_aaaa += einsum(x28, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new_aaaa += einsum(x28, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    del x28
    x29 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x29 += einsum(f.aa.oo, (0, 1), l2.aaaa, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_aaaa += einsum(x29, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    l2new_aaaa += einsum(x29, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    del x29
    x30 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x30 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x30 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 5, 2), (0, 4, 5, 1)) * -1.0
    l2new_aaaa += einsum(l2.aaaa, (0, 1, 2, 3), x30, (3, 4, 5, 2), (0, 1, 5, 4)) * 2.0
    del x30
    x31 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x31 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), (3, 5, 2, 4))
    l2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), x31, (4, 2, 5, 0), (1, 3, 5, 4))
    del x31
    x32 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x32 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 4, 3, 5))
    x33 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x33 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x33 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x34 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x34 += einsum(t2.bbbb, (0, 1, 2, 3), x33, (1, 4, 5, 3), (0, 4, 2, 5)) * 2.0
    del x33
    x35 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x35 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x35 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x35 += einsum(x32, (0, 1, 2, 3), (0, 1, 2, 3))
    del x32
    x35 += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x34
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x35, (3, 4, 1, 5), (0, 5, 2, 4))
    x36 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x36 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ovov, (1, 3, 4, 5), (4, 5, 0, 2))
    x37 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x37 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x37 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x38 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x38 += einsum(t2.abab, (0, 1, 2, 3), x37, (1, 4, 3, 5), (4, 5, 0, 2))
    del x37
    x39 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x39 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x39 += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x36
    x39 += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x38
    l2new_abab += einsum(l2.aaaa, (0, 1, 2, 3), x39, (4, 5, 3, 1), (0, 5, 2, 4)) * 2.0
    x40 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x40 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x40 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 5, 3), (1, 5, 2, 4)) * -1.0
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x40, (3, 4, 0, 5), (5, 1, 2, 4)) * -1.0
    del x40
    x41 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x41 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x41 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 5), (3, 5, 0, 4)) * -1.0
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x41, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    del x41
    x42 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x42 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x42 += einsum(x24, (0, 1), (0, 1))
    del x24
    x42 += einsum(x25, (0, 1), (0, 1)) * 2.0
    del x25
    l2new_abab += einsum(x42, (0, 1), l2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    del x42
    x43 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x43 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x44 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x44 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    x45 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x45 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x45 += einsum(x43, (0, 1), (0, 1)) * 2.0
    x45 += einsum(x44, (0, 1), (0, 1))
    l2new_abab += einsum(x45, (0, 1), l2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x45
    x46 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x46 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x46 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (1, 5, 0, 4))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x46, (3, 4, 2, 5), (0, 1, 5, 4))
    del x46
    x47 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x47 += einsum(x10, (0, 1), (0, 1)) * 0.5
    del x10
    x47 += einsum(x11, (0, 1), (0, 1))
    del x11
    l2new_abab += einsum(x47, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (0, 4, 2, 3)) * -2.0
    del x47
    x48 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x48 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4))
    x49 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x49 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    x50 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x50 += einsum(x48, (0, 1), (0, 1)) * 2.0
    x50 += einsum(x49, (0, 1), (0, 1))
    l2new_abab += einsum(x50, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (3, 0, 2, 4)) * -1.0
    del x50
    x51 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x51 += einsum(f.aa.oo, (0, 1), (0, 1))
    x51 += einsum(x19, (0, 1), (0, 1))
    del x19
    x51 += einsum(x20, (0, 1), (0, 1)) * 2.0
    del x20
    l2new_abab += einsum(x51, (0, 1), l2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    del x51
    x52 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x52 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 1, 3), (0, 4))
    x53 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x53 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x54 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x54 += einsum(f.bb.oo, (0, 1), (0, 1))
    x54 += einsum(x52, (0, 1), (0, 1)) * 2.0
    x54 += einsum(x53, (0, 1), (0, 1))
    l2new_abab += einsum(x54, (0, 1), l2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x54
    x55 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x55 += einsum(x14, (0, 1), (0, 1)) * 0.5
    del x14
    x55 += einsum(x15, (0, 1), (0, 1))
    del x15
    l2new_abab += einsum(x55, (0, 1), v.aabb.ovov, (1, 2, 3, 4), (2, 4, 0, 3)) * -2.0
    del x55
    x56 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x56 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    x57 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x57 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    x58 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x58 += einsum(x56, (0, 1), (0, 1)) * 2.0
    x58 += einsum(x57, (0, 1), (0, 1))
    l2new_abab += einsum(x58, (0, 1), v.aabb.ovov, (2, 3, 1, 4), (3, 4, 2, 0)) * -1.0
    del x58
    x59 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x59 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), (2, 3, 4, 5))
    l2new_bbbb += einsum(v.bbbb.ovov, (0, 1, 2, 3), x59, (4, 5, 0, 2), (3, 1, 4, 5)) * -2.0
    del x59
    x60 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x60 += einsum(l2.bbbb, (0, 1, 2, 3), x35, (3, 4, 1, 5), (2, 4, 0, 5)) * 2.0
    del x35
    x61 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x61 += einsum(l2.abab, (0, 1, 2, 3), x39, (4, 5, 2, 0), (4, 3, 5, 1))
    del x39
    x62 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x62 += einsum(x48, (0, 1), (0, 1))
    del x48
    x62 += einsum(x49, (0, 1), (0, 1)) * 0.5
    del x49
    x63 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x63 += einsum(x62, (0, 1), v.bbbb.ovov, (2, 1, 3, 4), (2, 3, 0, 4)) * 2.0
    del x62
    x64 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x64 += einsum(x56, (0, 1), (0, 1))
    del x56
    x64 += einsum(x57, (0, 1), (0, 1)) * 0.5
    del x57
    x65 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x65 += einsum(x64, (0, 1), v.bbbb.ovov, (2, 3, 1, 4), (0, 2, 4, 3)) * 2.0
    del x64
    x66 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x66 += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3))
    del x60
    x66 += einsum(x61, (0, 1, 2, 3), (1, 0, 3, 2))
    del x61
    x66 += einsum(x63, (0, 1, 2, 3), (1, 0, 2, 3))
    del x63
    x66 += einsum(x65, (0, 1, 2, 3), (0, 1, 3, 2))
    del x65
    l2new_bbbb += einsum(x66, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_bbbb += einsum(x66, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_bbbb += einsum(x66, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new_bbbb += einsum(x66, (0, 1, 2, 3), (3, 2, 1, 0))
    del x66
    x67 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x67 += einsum(f.bb.vv, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    x68 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x68 += einsum(x43, (0, 1), (0, 1))
    del x43
    x68 += einsum(x44, (0, 1), (0, 1)) * 0.5
    del x44
    x69 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x69 += einsum(x68, (0, 1), l2.bbbb, (2, 0, 3, 4), (3, 4, 2, 1)) * -4.0
    del x68
    x70 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x70 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x70 += einsum(x67, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x67
    x70 += einsum(x69, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x69
    l2new_bbbb += einsum(x70, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new_bbbb += einsum(x70, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    del x70
    x71 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x71 += einsum(x52, (0, 1), (0, 1))
    del x52
    x71 += einsum(x53, (0, 1), (0, 1)) * 0.5
    del x53
    x72 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x72 += einsum(x71, (0, 1), l2.bbbb, (2, 3, 4, 0), (4, 1, 2, 3)) * -4.0
    del x71
    l2new_bbbb += einsum(x72, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_bbbb += einsum(x72, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    del x72
    x73 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x73 += einsum(f.bb.oo, (0, 1), l2.bbbb, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_bbbb += einsum(x73, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    l2new_bbbb += einsum(x73, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    del x73
    x74 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x74 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x74 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 5, 2), (0, 4, 1, 5)) * -1.0
    l2new_bbbb += einsum(l2.bbbb, (0, 1, 2, 3), x74, (3, 4, 2, 5), (0, 1, 4, 5)) * -2.0
    del x74

    l2new.aaaa = l2new_aaaa
    l2new.abab = l2new_abab
    l2new.bbbb = l2new_bbbb

    return {"l2new": l2new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, t2=None, l2=None, **kwargs):
    rdm1_f = Namespace()

    delta = Namespace(aa=Namespace(), bb=Namespace())
    delta.aa = Namespace(oo=np.eye(nocc[0]), vv=np.eye(nvir[0]))
    delta.bb = Namespace(oo=np.eye(nocc[1]), vv=np.eye(nvir[1]))

    # RDM1
    rdm1_f_aa_oo = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    rdm1_f_aa_oo += einsum(delta.aa.oo, (0, 1), (0, 1))
    rdm1_f_aa_oo += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (4, 2)) * -1.0
    rdm1_f_aa_oo += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (4, 2)) * -2.0
    rdm1_f_bb_oo = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    rdm1_f_bb_oo += einsum(delta.bb.oo, (0, 1), (0, 1))
    rdm1_f_bb_oo += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (4, 2)) * -2.0
    rdm1_f_bb_oo += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (4, 3)) * -1.0
    rdm1_f_aa_vv = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    rdm1_f_aa_vv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    rdm1_f_aa_vv += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4)) * 2.0
    rdm1_f_bb_vv = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    rdm1_f_bb_vv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    rdm1_f_bb_vv += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4)) * 2.0
    rdm1_f_aa_ov = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    rdm1_f_bb_ov = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    rdm1_f_aa_vo = np.zeros((nvir[0], nocc[0]), dtype=types[float])
    rdm1_f_bb_vo = np.zeros((nvir[1], nocc[1]), dtype=types[float])

    rdm1_f_aa = np.block([[rdm1_f_aa_oo, rdm1_f_aa_ov], [rdm1_f_aa_vo, rdm1_f_aa_vv]])
    rdm1_f_bb = np.block([[rdm1_f_bb_oo, rdm1_f_bb_ov], [rdm1_f_bb_vo, rdm1_f_bb_vv]])

    rdm1_f.aa = rdm1_f_aa
    rdm1_f.bb = rdm1_f_bb

    return rdm1_f

def make_rdm2_f(f=None, v=None, nocc=None, nvir=None, t2=None, l2=None, **kwargs):
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
    rdm2_f_aaaa_oovv = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_oovv += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_abab_oovv = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_oovv = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_oovv += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_abab_ovov = np.zeros((nocc[0], nvir[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_ovov += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), (4, 1, 2, 5)) * -1.0
    rdm2_f_aaaa_vvoo = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_vvoo += einsum(l2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_abab_vvoo = np.zeros((nvir[0], nvir[1], nocc[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_vvoo += einsum(l2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_vvoo = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_vvoo += einsum(l2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_aaaa_vvvv = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_vvvv += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 5), (0, 1, 4, 5)) * 2.0
    rdm2_f_abab_vvvv = np.zeros((nvir[0], nvir[1], nvir[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_vvvv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), (0, 1, 4, 5))
    rdm2_f_bbbb_vvvv = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_vvvv += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 5), (0, 1, 4, 5)) * 2.0
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x0 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_aaaa_oooo += einsum(x0, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    rdm2_f_aaaa_oovv += einsum(t2.aaaa, (0, 1, 2, 3), x0, (0, 1, 4, 5), (5, 4, 2, 3)) * -2.0
    del x0
    x1 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x1 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    x2 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x2 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    x3 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x3 += einsum(x1, (0, 1), (0, 1))
    x3 += einsum(x2, (0, 1), (0, 1)) * 2.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x3, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x3, (2, 3), (0, 3, 2, 1))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x3, (2, 3), (3, 0, 1, 2))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x3, (2, 3), (3, 0, 2, 1)) * -1.0
    x4 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), (3, 5, 2, 4))
    rdm2_f_abab_oooo = np.zeros((nocc[0], nocc[1], nocc[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_oooo += einsum(x4, (0, 1, 2, 3), (3, 1, 2, 0))
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x4, (1, 4, 0, 5), (5, 4, 2, 3))
    del x4
    x5 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x5 += einsum(delta.aa.oo, (0, 1), (0, 1)) * -1.0
    x5 += einsum(x1, (0, 1), (0, 1))
    x5 += einsum(x2, (0, 1), (0, 1)) * 2.0
    rdm2_f_abab_oooo += einsum(delta.bb.oo, (0, 1), x5, (2, 3), (3, 0, 2, 1)) * -1.0
    del x5
    x6 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x6 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    x7 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x7 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    x8 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x8 += einsum(x6, (0, 1), (0, 1)) * 2.0
    x8 += einsum(x7, (0, 1), (0, 1))
    rdm2_f_abab_oooo += einsum(delta.aa.oo, (0, 1), x8, (2, 3), (0, 3, 1, 2)) * -1.0
    del x8
    x9 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x9 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_bbbb_oooo += einsum(x9, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    rdm2_f_bbbb_oovv += einsum(t2.bbbb, (0, 1, 2, 3), x9, (0, 1, 4, 5), (5, 4, 2, 3)) * -2.0
    del x9
    x10 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x10 += einsum(x6, (0, 1), (0, 1))
    del x6
    x10 += einsum(x7, (0, 1), (0, 1)) * 0.5
    del x7
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_abab_oovv += einsum(x10, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    x11 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x11 += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 0, 4, 5))
    rdm2_f_abab_ovvo = np.zeros((nocc[0], nvir[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_ovvo += einsum(x11, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x12 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum(t2.abab, (0, 1, 2, 3), x11, (1, 3, 4, 5), (4, 0, 5, 2))
    del x11
    rdm2_f_aaaa_oovv += einsum(x12, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    rdm2_f_aaaa_oovv += einsum(x12, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x12
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x13 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_aaaa_ovov = np.zeros((nocc[0], nvir[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_ovov += einsum(x13, (0, 1, 2, 3), (1, 2, 0, 3)) * -4.0
    rdm2_f_aaaa_ovvo = np.zeros((nocc[0], nvir[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_ovvo += einsum(x13, (0, 1, 2, 3), (1, 2, 3, 0)) * 4.0
    rdm2_f_aaaa_voov = np.zeros((nvir[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_voov += einsum(x13, (0, 1, 2, 3), (2, 1, 0, 3)) * 4.0
    rdm2_f_aaaa_vovo = np.zeros((nvir[0], nocc[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_vovo += einsum(x13, (0, 1, 2, 3), (2, 1, 3, 0)) * -4.0
    x14 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x14 += einsum(t2.aaaa, (0, 1, 2, 3), x13, (1, 4, 3, 5), (0, 4, 2, 5))
    del x13
    x15 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x15 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    x16 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x16 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4))
    x17 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x17 += einsum(x15, (0, 1), (0, 1))
    x17 += einsum(x16, (0, 1), (0, 1)) * 2.0
    x18 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x18 += einsum(x17, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x17
    x19 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x19 += einsum(x14, (0, 1, 2, 3), (0, 1, 2, 3)) * -8.0
    del x14
    x19 += einsum(x18, (0, 1, 2, 3), (1, 0, 2, 3))
    del x18
    rdm2_f_aaaa_oovv += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x19, (0, 1, 2, 3), (0, 1, 3, 2))
    del x19
    x20 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x20 += einsum(x3, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x3
    rdm2_f_aaaa_oovv += einsum(x20, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x20, (0, 1, 2, 3), (1, 0, 3, 2))
    del x20
    x21 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x21 += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 2, 5, 0), (3, 1, 4, 5))
    rdm2_f_abab_ovvo += einsum(x21, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x22 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x22 += einsum(t2.abab, (0, 1, 2, 3), x21, (1, 3, 4, 5), (0, 4, 2, 5))
    del x21
    rdm2_f_aaaa_oovv += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_aaaa_oovv += einsum(x22, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_aaaa_oovv += einsum(x22, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    rdm2_f_aaaa_oovv += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x22
    x23 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x23 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), (3, 4, 0, 5))
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x23, (1, 4, 2, 5), (0, 4, 5, 3))
    rdm2_f_abab_vovo = np.zeros((nvir[0], nocc[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_vovo += einsum(x23, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x23
    x24 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x24 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_bbbb_ovov = np.zeros((nocc[1], nvir[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_ovov += einsum(x24, (0, 1, 2, 3), (1, 2, 0, 3)) * -4.0
    rdm2_f_bbbb_ovvo = np.zeros((nocc[1], nvir[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_ovvo += einsum(x24, (0, 1, 2, 3), (1, 2, 3, 0)) * 4.0
    rdm2_f_bbbb_voov = np.zeros((nvir[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_voov += einsum(x24, (0, 1, 2, 3), (2, 1, 0, 3)) * 4.0
    rdm2_f_bbbb_vovo = np.zeros((nvir[1], nocc[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_vovo += einsum(x24, (0, 1, 2, 3), (2, 1, 3, 0)) * -4.0
    x25 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x25 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), (3, 4, 1, 5))
    rdm2_f_bbbb_ovov += einsum(x25, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_bbbb_ovvo += einsum(x25, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_bbbb_voov += einsum(x25, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_bbbb_vovo += einsum(x25, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x26 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x26 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    x26 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x25
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x26, (1, 4, 3, 5), (0, 4, 2, 5)) * 4.0
    del x26
    x27 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x27 += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (4, 5, 2, 0))
    rdm2_f_abab_voov = np.zeros((nvir[0], nocc[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_voov += einsum(x27, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x28 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x28 += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 1, 5), (4, 5, 2, 0))
    rdm2_f_abab_voov += einsum(x28, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x29 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x29 += einsum(x27, (0, 1, 2, 3), (0, 1, 2, 3))
    x29 += einsum(x28, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_abab_oovv += einsum(t2.aaaa, (0, 1, 2, 3), x29, (4, 5, 1, 3), (0, 4, 2, 5)) * 4.0
    del x29
    x30 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x30 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4))
    x31 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x31 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    x32 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x32 += einsum(x30, (0, 1), (0, 1))
    x32 += einsum(x31, (0, 1), (0, 1)) * 0.5
    rdm2_f_abab_oovv += einsum(x32, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    rdm2_f_abab_ovov += einsum(delta.aa.oo, (0, 1), x32, (2, 3), (0, 2, 1, 3)) * 2.0
    x33 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x33 += einsum(x15, (0, 1), (0, 1)) * 0.5
    del x15
    x33 += einsum(x16, (0, 1), (0, 1))
    del x16
    rdm2_f_abab_oovv += einsum(x33, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -2.0
    rdm2_f_aaaa_ovov += einsum(delta.aa.oo, (0, 1), x33, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_aaaa_ovvo += einsum(delta.aa.oo, (0, 1), x33, (2, 3), (0, 2, 3, 1)) * -2.0
    rdm2_f_aaaa_voov += einsum(delta.aa.oo, (0, 1), x33, (2, 3), (2, 0, 1, 3)) * -2.0
    rdm2_f_aaaa_vovo += einsum(delta.aa.oo, (0, 1), x33, (2, 3), (2, 0, 3, 1)) * 2.0
    rdm2_f_abab_vovo += einsum(delta.bb.oo, (0, 1), x33, (2, 3), (2, 0, 3, 1)) * 2.0
    del x33
    x34 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x34 += einsum(x1, (0, 1), (0, 1)) * 0.5
    del x1
    x34 += einsum(x2, (0, 1), (0, 1))
    del x2
    rdm2_f_abab_oovv += einsum(x34, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -2.0
    del x34
    x35 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x35 += einsum(t2.bbbb, (0, 1, 2, 3), x24, (1, 4, 3, 5), (4, 0, 5, 2))
    del x24
    x36 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x36 += einsum(t2.abab, (0, 1, 2, 3), x28, (4, 5, 0, 2), (4, 1, 5, 3))
    del x28
    x37 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x37 += einsum(x32, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 1, 4)) * -4.0
    del x32
    x38 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x38 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3)) * 8.0
    del x35
    x38 += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x36
    x38 += einsum(x37, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x37
    rdm2_f_bbbb_oovv += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_oovv += einsum(x38, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x38
    x39 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x39 += einsum(t2.abab, (0, 1, 2, 3), x27, (4, 5, 0, 2), (4, 1, 5, 3))
    del x27
    rdm2_f_bbbb_oovv += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_bbbb_oovv += einsum(x39, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_bbbb_oovv += einsum(x39, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    rdm2_f_bbbb_oovv += einsum(x39, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x39
    x40 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x40 += einsum(x10, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -4.0
    del x10
    rdm2_f_bbbb_oovv += einsum(x40, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_bbbb_oovv += einsum(x40, (0, 1, 2, 3), (1, 0, 3, 2))
    del x40
    x41 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x41 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_aaaa_ovov += einsum(x41, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_aaaa_ovvo += einsum(x41, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_aaaa_voov += einsum(x41, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_aaaa_vovo += einsum(x41, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x41
    x42 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x42 += einsum(x30, (0, 1), (0, 1)) * 2.0
    del x30
    x42 += einsum(x31, (0, 1), (0, 1))
    del x31
    rdm2_f_bbbb_ovov += einsum(delta.bb.oo, (0, 1), x42, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_ovvo += einsum(delta.bb.oo, (0, 1), x42, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_bbbb_voov += einsum(delta.bb.oo, (0, 1), x42, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_bbbb_vovo += einsum(delta.bb.oo, (0, 1), x42, (2, 3), (2, 0, 3, 1))
    del x42
    rdm2_f_aaaa_ooov = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_abab_ooov = np.zeros((nocc[0], nocc[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_ooov = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_aaaa_oovo = np.zeros((nocc[0], nocc[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_abab_oovo = np.zeros((nocc[0], nocc[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_oovo = np.zeros((nocc[1], nocc[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_aaaa_ovoo = np.zeros((nocc[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_abab_ovoo = np.zeros((nocc[0], nvir[1], nocc[0], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_ovoo = np.zeros((nocc[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_aaaa_vooo = np.zeros((nvir[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_abab_vooo = np.zeros((nvir[0], nocc[1], nocc[0], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_vooo = np.zeros((nvir[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_aaaa_ovvv = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_abab_ovvv = np.zeros((nocc[0], nvir[1], nvir[0], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_ovvv = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_aaaa_vovv = np.zeros((nvir[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_abab_vovv = np.zeros((nvir[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_vovv = np.zeros((nvir[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_aaaa_vvov = np.zeros((nvir[0], nvir[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_abab_vvov = np.zeros((nvir[0], nvir[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_vvov = np.zeros((nvir[1], nvir[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_aaaa_vvvo = np.zeros((nvir[0], nvir[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_abab_vvvo = np.zeros((nvir[0], nvir[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_vvvo = np.zeros((nvir[1], nvir[1], nvir[1], nocc[1]), dtype=types[float])

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

