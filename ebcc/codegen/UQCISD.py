# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 2), ()) * -1.0
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 2, 1, 3), ())

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    t1new = Namespace()
    t2new = Namespace()

    # T amplitudes
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvov, (4, 2, 1, 3), (0, 4))
    t1new_aa += einsum(f.aa.vv, (0, 1), t1.aa, (2, 1), (2, 0))
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (1, 3, 4, 2), (0, 4)) * 2.0
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    t1new_bb += einsum(f.bb.vv, (0, 1), t1.bb, (2, 1), (2, 0))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovvv, (0, 2, 4, 3), (1, 4))
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (1, 2, 4, 3), (0, 4)) * -2.0
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum(t1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (2, 0, 3, 4))
    t2new_abab += einsum(t1.aa, (0, 1), v.aabb.ooov, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_abab += einsum(t1.aa, (0, 1), v.aabb.vvov, (2, 1, 3, 4), (0, 3, 2, 4))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvoo, (4, 2, 5, 1), (0, 5, 4, 3)) * -1.0
    t2new_abab += einsum(t1.bb, (0, 1), v.aabb.ovoo, (2, 3, 4, 0), (2, 4, 3, 1)) * -1.0
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    x0 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x0 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    t1new_aa += einsum(x0, (0, 1), (0, 1))
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x1 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), x1, (4, 0, 1, 3), (4, 2)) * 2.0
    del x1
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x2 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x2 += einsum(t1.aa, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (3, 4, 2, 0))
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x2, (1, 3, 0, 4), (4, 2)) * -1.0
    del x2
    x3 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x3 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(x3, (0, 1), (0, 1))
    x4 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x4 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x4 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x5 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x5 += einsum(t1.bb, (0, 1), x4, (0, 2, 3, 1), (2, 3))
    x6 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x6 += einsum(f.bb.ov, (0, 1), (0, 1))
    x6 += einsum(x3, (0, 1), (0, 1))
    del x3
    x6 += einsum(x5, (0, 1), (0, 1)) * -1.0
    del x5
    t1new_aa += einsum(x6, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    t1new_bb += einsum(x6, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    del x6
    x7 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x7 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x7 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x8 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum(t1.aa, (0, 1), x7, (0, 2, 3, 1), (2, 3))
    x9 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x9 += einsum(f.aa.ov, (0, 1), (0, 1))
    x9 += einsum(x0, (0, 1), (0, 1))
    del x0
    x9 += einsum(x8, (0, 1), (0, 1)) * -1.0
    del x8
    t1new_aa += einsum(x9, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_bb += einsum(x9, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    del x9
    x10 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x10 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x10 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_aa += einsum(t1.aa, (0, 1), x10, (0, 2, 1, 3), (2, 3)) * -1.0
    x11 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x11 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x12 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x12 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x13 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x13 += einsum(f.aa.oo, (0, 1), (0, 1)) * 0.5
    x13 += einsum(x11, (0, 1), (1, 0)) * 0.5
    x13 += einsum(x12, (0, 1), (1, 0))
    t1new_aa += einsum(t1.aa, (0, 1), x13, (0, 2), (2, 1)) * -2.0
    t2new_abab += einsum(x13, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -2.0
    del x13
    x14 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x14 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x14 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), x14, (4, 1, 0, 3), (4, 2)) * -2.0
    del x14
    x15 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x15 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x15 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (4, 0, 2, 3))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x15, (1, 4, 0, 2), (4, 3)) * -1.0
    del x15
    x16 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x16 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x16 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_bb += einsum(t1.bb, (0, 1), x16, (0, 2, 1, 3), (2, 3)) * -1.0
    del x16
    x17 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x17 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x18 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x18 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x19 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x19 += einsum(f.bb.oo, (0, 1), (0, 1))
    x19 += einsum(x17, (0, 1), (1, 0)) * 2.0
    x19 += einsum(x18, (0, 1), (1, 0))
    t1new_bb += einsum(t1.bb, (0, 1), x19, (0, 2), (2, 1)) * -1.0
    t2new_abab += einsum(x19, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x19
    x20 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x20 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x21 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x21 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x22 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x22 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ovov, (1, 3, 4, 5), (4, 5, 0, 2))
    x23 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x23 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x23 += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x22
    x24 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x24 += einsum(t2.abab, (0, 1, 2, 3), x23, (1, 3, 4, 5), (0, 4, 2, 5))
    del x23
    x25 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x25 += einsum(t2.aaaa, (0, 1, 2, 3), x10, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x10
    x26 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x26 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x20
    x26 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x21
    x26 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    del x24
    x26 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x25
    t2new_aaaa += einsum(x26, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x26, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x26, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_aaaa += einsum(x26, (0, 1, 2, 3), (1, 0, 3, 2))
    del x26
    x27 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x27 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x28 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x28 += einsum(t2.abab, (0, 1, 2, 3), x4, (1, 4, 5, 3), (4, 5, 0, 2))
    x29 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x29 += einsum(t2.abab, (0, 1, 2, 3), x28, (1, 3, 4, 5), (0, 4, 2, 5)) * -1.0
    del x28
    x30 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x30 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x30 += einsum(x27, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x27
    x30 += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3))
    del x29
    t2new_aaaa += einsum(x30, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3))
    del x30
    x31 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x31 += einsum(f.aa.oo, (0, 1), t2.aaaa, (2, 1, 3, 4), (0, 2, 3, 4))
    x32 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x32 += einsum(x11, (0, 1), (0, 1))
    del x11
    x32 += einsum(x12, (0, 1), (0, 1)) * 2.0
    del x12
    x33 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x33 += einsum(x32, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x32
    x34 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x34 += einsum(x31, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x31
    x34 += einsum(x33, (0, 1, 2, 3), (0, 1, 3, 2))
    del x33
    t2new_aaaa += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x34, (0, 1, 2, 3), (1, 0, 2, 3))
    del x34
    x35 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x35 += einsum(t2.aaaa, (0, 1, 2, 3), x7, (1, 4, 5, 3), (0, 4, 2, 5))
    x36 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x36 += einsum(t2.aaaa, (0, 1, 2, 3), x35, (4, 1, 5, 3), (0, 4, 2, 5)) * -4.0
    del x35
    x37 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x37 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    x38 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x38 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x39 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x39 += einsum(x37, (0, 1), (0, 1))
    x39 += einsum(x38, (0, 1), (0, 1)) * 2.0
    x40 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x40 += einsum(x39, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 4, 0)) * -2.0
    del x39
    x41 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x41 += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x36
    x41 += einsum(x40, (0, 1, 2, 3), (1, 0, 2, 3))
    del x40
    t2new_aaaa += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x41, (0, 1, 2, 3), (0, 1, 3, 2))
    del x41
    x42 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x42 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x42 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 5, 2), (4, 0, 5, 1)) * -1.0
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x42, (0, 4, 1, 5), (4, 5, 2, 3)) * 2.0
    del x42
    x43 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x43 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 2, 4, 5))
    t2new_abab += einsum(x43, (0, 1, 2, 3), (2, 0, 3, 1)) * 2.0
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x44 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x44 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x45 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x45 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x45 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x45 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (4, 0, 5, 2))
    x45 += einsum(t2.aaaa, (0, 1, 2, 3), x44, (1, 4, 3, 5), (4, 0, 5, 2)) * -2.0
    del x44
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x45, (0, 4, 2, 5), (4, 1, 5, 3))
    del x45
    x46 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x46 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x46 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x47 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x47 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x47 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x47 += einsum(t2.bbbb, (0, 1, 2, 3), x46, (1, 4, 5, 3), (4, 0, 5, 2)) * -2.0
    del x46
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x47, (1, 4, 3, 5), (0, 4, 2, 5)) * -1.0
    del x47
    x48 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x48 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x48 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 5), (5, 3, 4, 0)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x48, (3, 4, 0, 5), (5, 1, 2, 4)) * -1.0
    del x48
    x49 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x49 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1)) * 0.5
    x49 += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3))
    del x43
    t2new_abab += einsum(t2.aaaa, (0, 1, 2, 3), x49, (4, 5, 1, 3), (0, 4, 2, 5)) * 4.0
    del x49
    x50 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x50 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 4, 1, 3), (2, 4))
    x51 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x51 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    x52 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x52 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x52 += einsum(x50, (0, 1), (1, 0)) * 2.0
    x52 += einsum(x51, (0, 1), (1, 0))
    t2new_abab += einsum(x52, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x52
    x53 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x53 += einsum(f.aa.vv, (0, 1), (0, 1)) * -0.5
    x53 += einsum(x37, (0, 1), (1, 0)) * 0.5
    del x37
    x53 += einsum(x38, (0, 1), (1, 0))
    del x38
    t2new_abab += einsum(x53, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -2.0
    del x53
    x54 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x54 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x54 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (5, 1, 4, 0))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x54, (1, 4, 0, 5), (5, 4, 2, 3))
    del x54
    x55 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x55 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x56 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x56 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x57 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x57 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 4, 3, 5))
    x58 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x58 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x58 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x58 += einsum(x57, (0, 1, 2, 3), (1, 0, 3, 2))
    x59 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x59 += einsum(t2.bbbb, (0, 1, 2, 3), x58, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x58
    x60 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x60 += einsum(x55, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x55
    x60 += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x56
    x60 += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3))
    del x57
    x60 += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3))
    del x59
    t2new_bbbb += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x60, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x60, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb += einsum(x60, (0, 1, 2, 3), (1, 0, 3, 2))
    del x60
    x61 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x61 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x62 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x62 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x62 += einsum(x61, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x61
    t2new_bbbb += einsum(x62, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3))
    del x62
    x63 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x63 += einsum(f.bb.oo, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4))
    x64 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x64 += einsum(x17, (0, 1), (0, 1))
    del x17
    x64 += einsum(x18, (0, 1), (0, 1)) * 0.5
    del x18
    x65 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x65 += einsum(x64, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4)) * -4.0
    del x64
    x66 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x66 += einsum(x63, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x63
    x66 += einsum(x65, (0, 1, 2, 3), (0, 1, 3, 2))
    del x65
    t2new_bbbb += einsum(x66, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x66, (0, 1, 2, 3), (1, 0, 2, 3))
    del x66
    x67 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x67 += einsum(t2.bbbb, (0, 1, 2, 3), x4, (1, 4, 5, 3), (4, 0, 5, 2))
    del x4
    x68 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x68 += einsum(t2.bbbb, (0, 1, 2, 3), x67, (1, 4, 3, 5), (0, 4, 2, 5)) * -4.0
    del x67
    x69 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x69 += einsum(t2.abab, (0, 1, 2, 3), x7, (0, 4, 5, 2), (1, 3, 4, 5))
    del x7
    x70 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x70 += einsum(t2.abab, (0, 1, 2, 3), x69, (4, 5, 0, 2), (1, 4, 3, 5)) * -1.0
    del x69
    x71 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x71 += einsum(x50, (0, 1), (0, 1))
    del x50
    x71 += einsum(x51, (0, 1), (0, 1)) * 0.5
    del x51
    x72 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x72 += einsum(x71, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 4, 0)) * -4.0
    del x71
    x73 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x73 += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x68
    x73 += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x70
    x73 += einsum(x72, (0, 1, 2, 3), (1, 0, 2, 3))
    del x72
    t2new_bbbb += einsum(x73, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x73, (0, 1, 2, 3), (0, 1, 3, 2))
    del x73
    x74 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x74 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x74 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 5, 2), (4, 0, 1, 5)) * -1.0
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x74, (0, 4, 5, 1), (5, 4, 2, 3)) * -2.0
    del x74

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t1new": t1new, "t2new": t2new}

