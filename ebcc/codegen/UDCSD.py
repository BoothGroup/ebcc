# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 2), ()) * -1.0
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x0 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x0 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x1 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum(f.aa.ov, (0, 1), (0, 1)) * 2.0
    x1 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3)) * 2.0
    x1 += einsum(t1.aa, (0, 1), x0, (0, 2, 3, 1), (2, 3)) * -1.0
    del x0
    e_cc += einsum(t1.aa, (0, 1), x1, (0, 1), ()) * 0.5
    del x1
    x2 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x2 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x2 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x3 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x3 += einsum(f.bb.ov, (0, 1), (0, 1)) * 2.0
    x3 += einsum(t1.bb, (0, 1), x2, (0, 2, 1, 3), (2, 3)) * -1.0
    del x2
    e_cc += einsum(t1.bb, (0, 1), x3, (0, 1), ()) * 0.5
    del x3

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    t1new = Namespace()
    t2new = Namespace()

    # T amplitudes
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    t1new_aa += einsum(f.aa.ov, (0, 1), (0, 1))
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    t1new_bb += einsum(f.bb.ov, (0, 1), (0, 1))
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    x0 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x0 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    t1new_aa += einsum(x0, (0, 1), (0, 1))
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x2 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x2 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x2 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), x2, (4, 0, 1, 3), (4, 2)) * 2.0
    del x2
    x3 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x3 += einsum(t1.aa, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (0, 2, 3, 4))
    x4 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x4 += einsum(v.aabb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x4 += einsum(x3, (0, 1, 2, 3), (1, 0, 2, 3))
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x4, (0, 4, 1, 3), (4, 2)) * -1.0
    del x4
    x5 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x5 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x5 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x5, (0, 4, 1, 3), (4, 2)) * 2.0
    del x5
    x6 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x6 += einsum(t2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    x6 += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    t1new_aa += einsum(v.aabb.vvov, (0, 1, 2, 3), x6, (4, 2, 1, 3), (4, 0))
    t1new_bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), x6, (0, 4, 1, 3), (4, 2))
    del x6
    x7 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x7 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x7 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x8 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum(t1.aa, (0, 1), x7, (0, 2, 3, 1), (2, 3))
    del x7
    x9 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x9 += einsum(f.aa.ov, (0, 1), (0, 1))
    x9 += einsum(x0, (0, 1), (0, 1))
    del x0
    x9 += einsum(x8, (0, 1), (0, 1)) * -1.0
    del x8
    t1new_aa += einsum(x9, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_bb += einsum(x9, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    x10 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x10 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(x10, (0, 1), (0, 1))
    x11 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x11 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x11 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x12 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x12 += einsum(t1.bb, (0, 1), x11, (0, 2, 1, 3), (2, 3))
    del x11
    x13 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x13 += einsum(f.bb.ov, (0, 1), (0, 1))
    x13 += einsum(x10, (0, 1), (0, 1))
    del x10
    x13 += einsum(x12, (0, 1), (0, 1)) * -1.0
    del x12
    t1new_aa += einsum(x13, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    t1new_bb += einsum(x13, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    x14 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x14 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x14 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_aa += einsum(t1.aa, (0, 1), x14, (0, 2, 1, 3), (2, 3)) * -1.0
    del x14
    x15 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x15 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3))
    x16 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x16 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x17 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x17 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x18 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x18 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x18 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x19 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x19 += einsum(t1.aa, (0, 1), x18, (2, 3, 0, 1), (2, 3))
    x20 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x20 += einsum(t1.aa, (0, 1), x9, (2, 1), (0, 2))
    x21 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x21 += einsum(f.aa.oo, (0, 1), (0, 1))
    x21 += einsum(x15, (0, 1), (1, 0))
    x21 += einsum(x16, (0, 1), (1, 0)) * 2.0
    x21 += einsum(x17, (0, 1), (1, 0))
    x21 += einsum(x19, (0, 1), (1, 0)) * -1.0
    x21 += einsum(x20, (0, 1), (1, 0))
    t1new_aa += einsum(t1.aa, (0, 1), x21, (0, 2), (2, 1)) * -1.0
    del x21
    x22 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x22 += einsum(f.aa.vv, (0, 1), (0, 1))
    x22 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (0, 2, 3, 1), (2, 3)) * -1.0
    t1new_aa += einsum(t1.aa, (0, 1), x22, (1, 2), (0, 2))
    del x22
    x23 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x23 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (2, 0, 4, 3))
    x24 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x24 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (0, 2, 3, 1))
    x24 += einsum(x23, (0, 1, 2, 3), (0, 2, 1, 3))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x24, (0, 1, 4, 2), (4, 3)) * -1.0
    x25 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x25 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x26 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x26 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x26 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), x26, (4, 0, 1, 3), (4, 2)) * 2.0
    del x26
    x27 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x27 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x27 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_bb += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x27, (0, 4, 1, 3), (4, 2)) * 2.0
    del x27
    x28 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x28 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x28 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_bb += einsum(t1.bb, (0, 1), x28, (0, 2, 1, 3), (2, 3)) * -1.0
    x29 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x29 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3))
    x30 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x30 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x31 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x31 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x32 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x32 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x32 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x33 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x33 += einsum(t1.bb, (0, 1), x32, (2, 3, 0, 1), (2, 3))
    del x32
    x34 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x34 += einsum(t1.bb, (0, 1), x13, (2, 1), (0, 2))
    x35 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x35 += einsum(f.bb.oo, (0, 1), (0, 1))
    x35 += einsum(x29, (0, 1), (1, 0))
    x35 += einsum(x30, (0, 1), (1, 0))
    x35 += einsum(x31, (0, 1), (1, 0)) * 2.0
    x35 += einsum(x33, (0, 1), (1, 0)) * -1.0
    x35 += einsum(x34, (0, 1), (1, 0))
    t1new_bb += einsum(t1.bb, (0, 1), x35, (0, 2), (2, 1)) * -1.0
    del x35
    x36 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x36 += einsum(f.bb.vv, (0, 1), (0, 1))
    x36 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (0, 2, 3, 1), (2, 3)) * -1.0
    t1new_bb += einsum(t1.bb, (0, 1), x36, (1, 2), (0, 2))
    del x36
    x37 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x37 += einsum(t1.aa, (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x38 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x38 += einsum(t1.aa, (0, 1), x37, (2, 3, 1, 4), (0, 2, 3, 4))
    del x37
    x39 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x39 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x40 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x40 += einsum(t2.aaaa, (0, 1, 2, 3), x39, (4, 1, 5, 3), (4, 0, 5, 2))
    x41 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x41 += einsum(t2.abab, (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x42 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x42 += einsum(t2.abab, (0, 1, 2, 3), x41, (4, 1, 5, 3), (4, 0, 5, 2))
    del x41
    x43 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x43 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 1), (2, 3))
    x44 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x44 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x45 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x45 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    x46 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x46 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x46 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x47 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x47 += einsum(t1.aa, (0, 1), x46, (0, 1, 2, 3), (2, 3))
    x48 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x48 += einsum(x43, (0, 1), (1, 0)) * -1.0
    x48 += einsum(x44, (0, 1), (1, 0))
    x48 += einsum(x45, (0, 1), (1, 0)) * 0.5
    x48 += einsum(x47, (0, 1), (1, 0)) * -1.0
    del x47
    x49 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x49 += einsum(x48, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x48
    x50 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x50 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -1.0
    x51 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x51 += einsum(x9, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4)) * -2.0
    x52 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x52 += einsum(t1.aa, (0, 1), x1, (2, 3, 4, 1), (2, 0, 4, 3))
    x53 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x53 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x53 += einsum(x52, (0, 1, 2, 3), (3, 1, 2, 0))
    del x52
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x53, (0, 4, 1, 5), (4, 5, 2, 3)) * 2.0
    x54 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x54 += einsum(t1.aa, (0, 1), x53, (0, 2, 3, 4), (2, 3, 4, 1))
    del x53
    x55 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x55 += einsum(x50, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    del x50
    x55 += einsum(x51, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x51
    x55 += einsum(x54, (0, 1, 2, 3), (1, 0, 2, 3))
    del x54
    x56 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x56 += einsum(t1.aa, (0, 1), x55, (0, 2, 3, 4), (2, 3, 1, 4))
    del x55
    x57 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x57 += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3))
    del x38
    x57 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x40
    x57 += einsum(x42, (0, 1, 2, 3), (0, 1, 2, 3))
    del x42
    x57 += einsum(x49, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x49
    x57 += einsum(x56, (0, 1, 2, 3), (1, 0, 2, 3))
    del x56
    t2new_aaaa += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x57, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x57
    x58 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x58 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x59 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x59 += einsum(t2.aaaa, (0, 1, 2, 3), x58, (4, 5, 0, 1), (4, 5, 2, 3))
    x60 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x60 += einsum(f.aa.oo, (0, 1), (0, 1))
    x60 += einsum(x20, (0, 1), (0, 1))
    del x20
    x61 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x61 += einsum(x60, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x60
    x62 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x62 += einsum(x15, (0, 1), (1, 0))
    x62 += einsum(x16, (0, 1), (1, 0))
    x62 += einsum(x17, (0, 1), (1, 0)) * 0.5
    x62 += einsum(x19, (0, 1), (1, 0)) * -1.0
    del x19
    x63 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x63 += einsum(x62, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x62
    x64 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x64 += einsum(x59, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x59
    x64 += einsum(x61, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x61
    x64 += einsum(x63, (0, 1, 2, 3), (0, 1, 3, 2))
    del x63
    t2new_aaaa += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x64, (0, 1, 2, 3), (1, 0, 2, 3))
    del x64
    x65 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x65 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x66 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x66 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x67 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x67 += einsum(t1.aa, (0, 1), v.aabb.vvov, (2, 1, 3, 4), (0, 3, 2, 4))
    t2new_abab += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3))
    x68 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x68 += einsum(t2.abab, (0, 1, 2, 3), x67, (4, 1, 5, 3), (4, 0, 2, 5))
    x69 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x69 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x69 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x69 += einsum(x66, (0, 1, 2, 3), (1, 0, 3, 2))
    x70 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x70 += einsum(t2.aaaa, (0, 1, 2, 3), x69, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x69
    x71 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x71 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x71 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x72 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x72 += einsum(t1.aa, (0, 1), x71, (2, 3, 1, 4), (0, 2, 3, 4))
    del x71
    x73 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x73 += einsum(t2.aaaa, (0, 1, 2, 3), x72, (4, 1, 3, 5), (0, 4, 2, 5)) * -2.0
    del x72
    x74 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x74 += einsum(t1.aa, (0, 1), x58, (2, 3, 4, 0), (2, 4, 3, 1))
    del x58
    x75 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x75 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x76 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x76 += einsum(t2.abab, (0, 1, 2, 3), x3, (4, 5, 1, 3), (4, 0, 5, 2))
    x77 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x77 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x77 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x78 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x78 += einsum(t2.aaaa, (0, 1, 2, 3), x77, (1, 4, 5, 3), (0, 4, 5, 2)) * 2.0
    del x77
    x79 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x79 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    x79 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x80 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x80 += einsum(t2.aaaa, (0, 1, 2, 3), x79, (4, 5, 1, 3), (0, 4, 5, 2)) * 2.0
    del x79
    x81 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x81 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x81 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x81 += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3))
    x82 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x82 += einsum(t1.aa, (0, 1), x81, (2, 3, 1, 4), (0, 2, 3, 4))
    del x81
    x83 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x83 += einsum(x74, (0, 1, 2, 3), (0, 2, 1, 3))
    del x74
    x83 += einsum(x75, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x75
    x83 += einsum(x76, (0, 1, 2, 3), (0, 2, 1, 3))
    del x76
    x83 += einsum(x78, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x78
    x83 += einsum(x80, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x80
    x83 += einsum(x82, (0, 1, 2, 3), (0, 2, 1, 3))
    del x82
    x84 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x84 += einsum(t1.aa, (0, 1), x83, (2, 0, 3, 4), (2, 3, 1, 4))
    del x83
    x85 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x85 += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3))
    del x65
    x85 += einsum(x66, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x66
    x85 += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3))
    del x68
    x85 += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x70
    x85 += einsum(x73, (0, 1, 2, 3), (1, 0, 2, 3))
    del x73
    x85 += einsum(x84, (0, 1, 2, 3), (0, 1, 2, 3))
    del x84
    t2new_aaaa += einsum(x85, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x85, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa += einsum(x85, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x85, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x85
    x86 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x86 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new_aaaa += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x86, (0, 1, 2, 3), (1, 0, 2, 3))
    del x86
    x87 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x87 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x88 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x88 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x88 += einsum(x87, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x87
    t2new_aaaa += einsum(x88, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x88, (0, 1, 2, 3), (0, 1, 2, 3))
    del x88
    x89 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x89 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x90 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x90 += einsum(t1.aa, (0, 1), x89, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    del x89
    x90 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x90 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    t2new_aaaa += einsum(t1.aa, (0, 1), x90, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x90
    x91 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x91 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x92 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x92 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x92 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x93 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x93 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x93 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x93 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3))
    x93 += einsum(x25, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x94 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x94 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x94 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x94 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (4, 1, 5, 3)) * 0.5
    x94 += einsum(x91, (0, 1, 2, 3), (1, 0, 3, 2))
    x94 += einsum(t1.bb, (0, 1), x92, (2, 1, 3, 4), (2, 0, 4, 3)) * -0.5
    x94 += einsum(t1.bb, (0, 1), x93, (2, 3, 0, 4), (3, 2, 4, 1)) * -0.5
    del x93
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x94, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x94
    x95 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x95 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x95 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x95 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    x95 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x1
    x96 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x96 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x96 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x96 += einsum(x39, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x39
    x96 += einsum(t1.aa, (0, 1), x46, (2, 1, 3, 4), (2, 0, 4, 3)) * -1.0
    x96 += einsum(t1.aa, (0, 1), x95, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x95
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x96, (0, 4, 2, 5), (4, 1, 5, 3))
    del x96
    x97 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x97 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x97 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -0.5
    x98 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x98 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x98 += einsum(t1.aa, (0, 1), v.aabb.ooov, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    x98 += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3))
    del x67
    x98 += einsum(v.aabb.ovov, (0, 1, 2, 3), x97, (0, 4, 1, 5), (4, 2, 5, 3)) * 2.0
    t2new_abab += einsum(t2.bbbb, (0, 1, 2, 3), x98, (4, 1, 5, 3), (4, 0, 5, 2)) * 2.0
    del x98
    x99 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x99 += einsum(t1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (2, 0, 3, 4))
    x100 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x100 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x100 += einsum(x99, (0, 1, 2, 3), (0, 1, 2, 3))
    x100 += einsum(t1.bb, (0, 1), x24, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x24
    t2new_abab += einsum(x100, (0, 1, 2, 3), x97, (0, 4, 2, 5), (4, 1, 5, 3)) * 2.0
    del x97, x100
    x101 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x101 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 4, 1), (0, 4, 2, 3))
    x102 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x102 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (0, 2, 3, 1))
    x102 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x103 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x103 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x103 += einsum(x101, (0, 1, 2, 3), (1, 0, 3, 2))
    x103 += einsum(t1.aa, (0, 1), x102, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x103, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    del x103
    x104 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x104 += einsum(v.aabb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x104 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    del x3
    x105 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x105 += einsum(v.aabb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x105 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (2, 0, 3, 4))
    x105 += einsum(t1.bb, (0, 1), x104, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x104
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x105, (0, 4, 3, 5), (4, 1, 2, 5)) * -1.0
    del x105
    x106 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x106 += einsum(v.aabb.vvvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x106 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 2, 3, 4), (2, 1, 3, 4))
    x106 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 4), (2, 3, 4, 1))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x106, (2, 4, 3, 5), (0, 1, 4, 5)) * -1.0
    del x106
    x107 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x107 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 1, 2, 3), (2, 3))
    x108 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x108 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    x109 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x109 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x110 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x110 += einsum(t1.bb, (0, 1), x92, (0, 1, 2, 3), (2, 3))
    x111 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x111 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x111 += einsum(x107, (0, 1), (1, 0)) * -1.0
    x111 += einsum(x108, (0, 1), (1, 0)) * 0.5
    x111 += einsum(x109, (0, 1), (1, 0))
    x111 += einsum(x110, (0, 1), (1, 0)) * -1.0
    x111 += einsum(t1.bb, (0, 1), x13, (0, 2), (2, 1))
    t2new_abab += einsum(x111, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x111
    x112 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x112 += einsum(f.aa.vv, (0, 1), (0, 1)) * -2.0
    x112 += einsum(x43, (0, 1), (1, 0)) * -2.0
    del x43
    x112 += einsum(x44, (0, 1), (1, 0)) * 2.0
    del x44
    x112 += einsum(x45, (0, 1), (1, 0))
    del x45
    x112 += einsum(t1.aa, (0, 1), x46, (0, 1, 2, 3), (3, 2)) * -2.0
    del x46
    x112 += einsum(t1.aa, (0, 1), x9, (0, 2), (2, 1)) * 2.0
    t2new_abab += einsum(x112, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -0.5
    del x112
    x113 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x113 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 4, 1), (2, 3, 0, 4))
    x114 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x114 += einsum(v.aabb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x114 += einsum(x113, (0, 1, 2, 3), (1, 0, 3, 2))
    x114 += einsum(t1.aa, (0, 1), x102, (2, 3, 4, 1), (2, 0, 4, 3))
    del x102
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x114, (0, 4, 1, 5), (4, 5, 2, 3))
    del x114
    x115 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x115 += einsum(f.bb.oo, (0, 1), (0, 1))
    x115 += einsum(x29, (0, 1), (1, 0))
    x115 += einsum(x30, (0, 1), (1, 0)) * 0.5
    x115 += einsum(x31, (0, 1), (1, 0))
    x115 += einsum(x33, (0, 1), (1, 0)) * -1.0
    x115 += einsum(x34, (0, 1), (1, 0))
    t2new_abab += einsum(x115, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x115
    x116 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x116 += einsum(f.aa.oo, (0, 1), (0, 1)) * 2.0
    x116 += einsum(x15, (0, 1), (1, 0)) * 2.0
    del x15
    x116 += einsum(x16, (0, 1), (1, 0)) * 2.0
    del x16
    x116 += einsum(x17, (0, 1), (1, 0))
    del x17
    x116 += einsum(t1.aa, (0, 1), x18, (2, 3, 0, 1), (3, 2)) * -2.0
    del x18
    x116 += einsum(t1.aa, (0, 1), x9, (2, 1), (2, 0)) * 2.0
    del x9
    t2new_abab += einsum(x116, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -0.5
    del x116
    x117 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x117 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x117 += einsum(x101, (0, 1, 2, 3), (0, 1, 3, 2))
    del x101
    x118 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x118 += einsum(v.aabb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x118 += einsum(x113, (0, 1, 2, 3), (1, 0, 2, 3))
    del x113
    x118 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (4, 0, 1, 5))
    x119 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x119 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (0, 2, 3, 1))
    x119 += einsum(x23, (0, 1, 2, 3), (0, 2, 1, 3))
    x119 += einsum(t1.aa, (0, 1), x117, (2, 3, 1, 4), (0, 3, 2, 4))
    del x117
    x119 += einsum(t1.aa, (0, 1), x118, (0, 2, 3, 4), (2, 4, 3, 1)) * -1.0
    del x118
    t2new_abab += einsum(t1.bb, (0, 1), x119, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x119
    x120 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x120 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x120 += einsum(t1.aa, (0, 1), v.aabb.vvvv, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_abab += einsum(t1.bb, (0, 1), x120, (2, 3, 1, 4), (2, 0, 3, 4))
    del x120
    x121 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x121 += einsum(v.aabb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x121 += einsum(t1.bb, (0, 1), v.aabb.oovv, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab += einsum(t1.aa, (0, 1), x121, (0, 2, 3, 4), (2, 3, 1, 4)) * -1.0
    del x121
    x122 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x122 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x123 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x123 += einsum(t2.abab, (0, 1, 2, 3), x99, (0, 4, 2, 5), (4, 1, 3, 5))
    del x99
    x124 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x124 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x124 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -0.5
    x125 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x125 += einsum(x124, (0, 1, 2, 3), x28, (0, 4, 2, 5), (4, 1, 5, 3)) * 2.0
    del x28, x124
    x126 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x126 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (4, 0, 5, 2))
    x127 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x127 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x127 += einsum(x126, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x126
    x128 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x128 += einsum(t2.abab, (0, 1, 2, 3), x127, (0, 4, 2, 5), (1, 4, 3, 5))
    del x127
    x129 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x129 += einsum(t1.bb, (0, 1), x92, (2, 1, 3, 4), (0, 2, 3, 4))
    del x92
    x130 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x130 += einsum(t2.bbbb, (0, 1, 2, 3), x129, (4, 1, 5, 3), (0, 4, 2, 5)) * -2.0
    del x129
    x131 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x131 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 5), (1, 4, 5, 3))
    x132 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x132 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x133 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x133 += einsum(t1.bb, (0, 1), x132, (2, 3, 4, 0), (2, 4, 3, 1))
    x134 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x134 += einsum(t1.bb, (0, 1), x122, (2, 3, 1, 4), (0, 2, 3, 4))
    x135 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x135 += einsum(t2.abab, (0, 1, 2, 3), x23, (0, 4, 5, 2), (4, 1, 5, 3))
    del x23
    x136 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x136 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x136 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x137 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x137 += einsum(t2.bbbb, (0, 1, 2, 3), x136, (4, 5, 1, 3), (0, 4, 5, 2)) * 2.0
    del x136
    x138 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x138 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x138 += einsum(x25, (0, 1, 2, 3), (0, 2, 1, 3))
    x139 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x139 += einsum(t2.bbbb, (0, 1, 2, 3), x138, (4, 1, 5, 3), (0, 4, 5, 2)) * 2.0
    del x138
    x140 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x140 += einsum(x131, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x131
    x140 += einsum(x133, (0, 1, 2, 3), (0, 2, 1, 3))
    del x133
    x140 += einsum(x134, (0, 1, 2, 3), (0, 2, 1, 3))
    del x134
    x140 += einsum(x135, (0, 1, 2, 3), (0, 2, 1, 3))
    del x135
    x140 += einsum(x137, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x137
    x140 += einsum(x139, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x139
    x141 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x141 += einsum(t1.bb, (0, 1), x140, (2, 0, 3, 4), (2, 3, 1, 4))
    del x140
    x142 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x142 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3))
    del x122
    x142 += einsum(x123, (0, 1, 2, 3), (0, 1, 2, 3))
    del x123
    x142 += einsum(x125, (0, 1, 2, 3), (1, 0, 3, 2))
    del x125
    x142 += einsum(x128, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x128
    x142 += einsum(x130, (0, 1, 2, 3), (1, 0, 2, 3))
    del x130
    x142 += einsum(x141, (0, 1, 2, 3), (0, 1, 2, 3))
    del x141
    t2new_bbbb += einsum(x142, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x142, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb += einsum(x142, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb += einsum(x142, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x142
    x143 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x143 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x144 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x144 += einsum(t2.abab, (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 2), (4, 1, 5, 3))
    x145 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x145 += einsum(t2.abab, (0, 1, 2, 3), x144, (0, 4, 2, 5), (4, 1, 5, 3))
    del x144
    x146 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x146 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x146 += einsum(x143, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x143
    x146 += einsum(x145, (0, 1, 2, 3), (1, 0, 3, 2))
    del x145
    t2new_bbbb += einsum(x146, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x146, (0, 1, 2, 3), (0, 1, 2, 3))
    del x146
    x147 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x147 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x148 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x148 += einsum(t2.bbbb, (0, 1, 2, 3), x132, (4, 5, 0, 1), (4, 5, 2, 3))
    del x132
    x149 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x149 += einsum(f.bb.oo, (0, 1), (0, 1))
    x149 += einsum(x34, (0, 1), (0, 1))
    del x34
    x150 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x150 += einsum(x149, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x149
    x151 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x151 += einsum(x29, (0, 1), (1, 0))
    del x29
    x151 += einsum(x30, (0, 1), (1, 0)) * 0.5
    del x30
    x151 += einsum(x31, (0, 1), (1, 0))
    del x31
    x151 += einsum(x33, (0, 1), (1, 0)) * -1.0
    del x33
    x152 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x152 += einsum(x151, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x151
    x153 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x153 += einsum(x147, (0, 1, 2, 3), (0, 1, 3, 2))
    del x147
    x153 += einsum(x148, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x148
    x153 += einsum(x150, (0, 1, 2, 3), (1, 0, 3, 2))
    del x150
    x153 += einsum(x152, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x152
    t2new_bbbb += einsum(x153, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x153, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x153
    x154 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x154 += einsum(t1.bb, (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x155 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x155 += einsum(t1.bb, (0, 1), x154, (2, 3, 1, 4), (0, 2, 3, 4))
    del x154
    x156 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x156 += einsum(t2.bbbb, (0, 1, 2, 3), x91, (4, 1, 5, 3), (0, 4, 2, 5))
    del x91
    x157 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x157 += einsum(x107, (0, 1), (1, 0)) * -1.0
    del x107
    x157 += einsum(x108, (0, 1), (1, 0)) * 0.5
    del x108
    x157 += einsum(x109, (0, 1), (1, 0))
    del x109
    x157 += einsum(x110, (0, 1), (1, 0)) * -1.0
    del x110
    x158 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x158 += einsum(x157, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x157
    x159 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x159 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -1.0
    x160 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x160 += einsum(x13, (0, 1), t2.bbbb, (2, 3, 4, 1), (0, 2, 3, 4)) * -2.0
    del x13
    x161 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x161 += einsum(t1.bb, (0, 1), x25, (2, 3, 4, 1), (2, 0, 4, 3))
    del x25
    x162 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x162 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x162 += einsum(x161, (0, 1, 2, 3), (3, 1, 2, 0))
    del x161
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x162, (0, 4, 1, 5), (4, 5, 2, 3)) * 2.0
    x163 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x163 += einsum(t1.bb, (0, 1), x162, (0, 2, 3, 4), (2, 3, 4, 1))
    del x162
    x164 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x164 += einsum(x159, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    del x159
    x164 += einsum(x160, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x160
    x164 += einsum(x163, (0, 1, 2, 3), (1, 0, 2, 3))
    del x163
    x165 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x165 += einsum(t1.bb, (0, 1), x164, (0, 2, 3, 4), (2, 3, 1, 4))
    del x164
    x166 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x166 += einsum(x155, (0, 1, 2, 3), (0, 1, 2, 3))
    del x155
    x166 += einsum(x156, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x156
    x166 += einsum(x158, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x158
    x166 += einsum(x165, (0, 1, 2, 3), (1, 0, 2, 3))
    del x165
    t2new_bbbb += einsum(x166, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x166, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x166
    x167 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x167 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x168 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x168 += einsum(t1.bb, (0, 1), x167, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x167
    x168 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x168 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * 0.5
    t2new_bbbb += einsum(t1.bb, (0, 1), x168, (2, 3, 0, 4), (2, 3, 1, 4)) * 2.0
    del x168

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t1new": t1new, "t2new": t2new}

