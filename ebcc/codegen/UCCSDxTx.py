# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 2, 1, 3), ())
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x0 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x0 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x1 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum(f.aa.ov, (0, 1), (0, 1))
    x1 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    x1 += einsum(t1.aa, (0, 1), x0, (0, 2, 3, 1), (2, 3)) * -0.5
    del x0
    e_cc += einsum(t1.aa, (0, 1), x1, (0, 1), ())
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
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    t1new_bb += einsum(f.bb.ov, (0, 1), (0, 1))
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    t1new_aa += einsum(f.aa.ov, (0, 1), (0, 1))
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    x0 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x0 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(x0, (0, 1), (0, 1))
    x1 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x1 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x2 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x2 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x2 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), x2, (4, 0, 1, 3), (4, 2)) * 2.0
    del x2
    x3 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x3 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (2, 0, 4, 3))
    x4 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x4 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (0, 2, 3, 1))
    x4 += einsum(x3, (0, 1, 2, 3), (0, 2, 1, 3))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x4, (0, 1, 4, 2), (4, 3)) * -1.0
    x5 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x5 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x5 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_bb += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x5, (0, 4, 3, 1), (4, 2)) * -2.0
    del x5
    x6 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x6 += einsum(t2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    x6 += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    t1new_bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), x6, (0, 4, 1, 3), (4, 2))
    t1new_aa += einsum(v.aabb.vvov, (0, 1, 2, 3), x6, (4, 2, 1, 3), (4, 0))
    x7 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x7 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x7 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x8 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x8 += einsum(t1.bb, (0, 1), x7, (0, 2, 1, 3), (2, 3))
    x9 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x9 += einsum(f.bb.ov, (0, 1), (0, 1))
    x9 += einsum(x0, (0, 1), (0, 1))
    del x0
    x9 += einsum(x8, (0, 1), (0, 1)) * -1.0
    del x8
    t1new_bb += einsum(x9, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_aa += einsum(x9, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    x10 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x10 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    t1new_aa += einsum(x10, (0, 1), (0, 1))
    x11 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x11 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x11 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x12 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x12 += einsum(t1.aa, (0, 1), x11, (0, 2, 3, 1), (2, 3))
    x13 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x13 += einsum(f.aa.ov, (0, 1), (0, 1))
    x13 += einsum(x10, (0, 1), (0, 1))
    del x10
    x13 += einsum(x12, (0, 1), (0, 1)) * -1.0
    del x12
    t1new_bb += einsum(x13, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    t1new_aa += einsum(x13, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    x14 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x14 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x14 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_bb += einsum(t1.bb, (0, 1), x14, (0, 2, 1, 3), (2, 3)) * -1.0
    x15 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x15 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3))
    x16 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x16 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x17 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x17 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x18 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x18 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x18 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x19 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x19 += einsum(t1.bb, (0, 1), x18, (2, 3, 0, 1), (2, 3))
    x20 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x20 += einsum(t1.bb, (0, 1), x9, (2, 1), (0, 2))
    x21 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x21 += einsum(f.bb.oo, (0, 1), (0, 1))
    x21 += einsum(x15, (0, 1), (1, 0))
    x21 += einsum(x16, (0, 1), (1, 0))
    x21 += einsum(x17, (0, 1), (1, 0)) * 2.0
    x21 += einsum(x19, (0, 1), (1, 0)) * -1.0
    x21 += einsum(x20, (0, 1), (1, 0))
    t1new_bb += einsum(t1.bb, (0, 1), x21, (0, 2), (2, 1)) * -1.0
    t2new_abab += einsum(x21, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x21
    x22 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x22 += einsum(f.bb.vv, (0, 1), (0, 1))
    x22 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(t1.bb, (0, 1), x22, (1, 2), (0, 2))
    del x22
    x23 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x23 += einsum(t1.aa, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (0, 2, 3, 4))
    x24 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x24 += einsum(v.aabb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x24 += einsum(x23, (0, 1, 2, 3), (1, 0, 2, 3))
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x24, (0, 4, 1, 3), (4, 2)) * -1.0
    del x24
    x25 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x25 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x26 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x26 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x26 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), x26, (4, 1, 0, 3), (4, 2)) * -2.0
    del x26
    x27 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x27 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x27 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x27, (0, 4, 1, 3), (4, 2)) * 2.0
    del x27
    x28 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x28 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x28 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_aa += einsum(t1.aa, (0, 1), x28, (0, 2, 1, 3), (2, 3)) * -1.0
    del x28
    x29 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x29 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3))
    x30 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x30 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x31 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x31 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x32 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x32 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x32 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x33 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x33 += einsum(t1.aa, (0, 1), x32, (2, 3, 0, 1), (2, 3))
    del x32
    x34 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x34 += einsum(t1.aa, (0, 1), x13, (2, 1), (0, 2))
    x35 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x35 += einsum(f.aa.oo, (0, 1), (0, 1))
    x35 += einsum(x29, (0, 1), (1, 0))
    x35 += einsum(x30, (0, 1), (1, 0)) * 2.0
    x35 += einsum(x31, (0, 1), (1, 0))
    x35 += einsum(x33, (0, 1), (1, 0)) * -1.0
    x35 += einsum(x34, (0, 1), (1, 0))
    t1new_aa += einsum(t1.aa, (0, 1), x35, (0, 2), (2, 1)) * -1.0
    t2new_abab += einsum(x35, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    del x35
    x36 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x36 += einsum(f.aa.vv, (0, 1), (0, 1))
    x36 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (0, 2, 3, 1), (2, 3)) * -1.0
    t1new_aa += einsum(t1.aa, (0, 1), x36, (1, 2), (0, 2))
    del x36
    x37 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x37 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x38 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x38 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x39 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x39 += einsum(t1.aa, (0, 1), v.aabb.vvov, (2, 1, 3, 4), (0, 3, 2, 4))
    t2new_abab += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3))
    x40 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x40 += einsum(t2.abab, (0, 1, 2, 3), x39, (4, 1, 5, 3), (4, 0, 2, 5))
    x41 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x41 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x41 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x41 += einsum(x38, (0, 1, 2, 3), (1, 0, 3, 2))
    x42 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x42 += einsum(t2.aaaa, (0, 1, 2, 3), x41, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x41
    x43 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x43 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x43 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x44 += einsum(t1.aa, (0, 1), x43, (2, 1, 3, 4), (0, 2, 3, 4))
    del x43
    x45 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x45 += einsum(t2.aaaa, (0, 1, 2, 3), x44, (4, 1, 5, 3), (0, 4, 2, 5)) * -2.0
    del x44
    x46 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x46 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x47 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x47 += einsum(t1.aa, (0, 1), x46, (2, 3, 4, 0), (2, 4, 3, 1))
    x48 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x48 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x49 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x49 += einsum(t2.abab, (0, 1, 2, 3), x23, (4, 5, 1, 3), (4, 0, 5, 2))
    del x23
    x50 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x50 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x50 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x51 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x51 += einsum(t2.aaaa, (0, 1, 2, 3), x50, (1, 4, 5, 3), (0, 4, 5, 2)) * 2.0
    del x50
    x52 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x52 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x52 += einsum(x25, (0, 1, 2, 3), (0, 2, 1, 3))
    x53 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x53 += einsum(t2.aaaa, (0, 1, 2, 3), x52, (4, 1, 5, 3), (0, 4, 5, 2)) * 2.0
    del x52
    x54 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x54 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x54 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x54 += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3))
    x55 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x55 += einsum(t1.aa, (0, 1), x54, (2, 3, 1, 4), (0, 2, 3, 4))
    del x54
    x56 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x56 += einsum(x47, (0, 1, 2, 3), (0, 2, 1, 3))
    del x47
    x56 += einsum(x48, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x48
    x56 += einsum(x49, (0, 1, 2, 3), (0, 2, 1, 3))
    del x49
    x56 += einsum(x51, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x51
    x56 += einsum(x53, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x53
    x56 += einsum(x55, (0, 1, 2, 3), (0, 2, 1, 3))
    del x55
    x57 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x57 += einsum(t1.aa, (0, 1), x56, (2, 0, 3, 4), (2, 3, 1, 4))
    del x56
    x58 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x58 += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3))
    del x37
    x58 += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x38
    x58 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3))
    del x40
    x58 += einsum(x42, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x42
    x58 += einsum(x45, (0, 1, 2, 3), (1, 0, 2, 3))
    del x45
    x58 += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3))
    del x57
    t2new_aaaa += einsum(x58, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x58, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa += einsum(x58, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x58, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x58
    x59 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x59 += einsum(t1.aa, (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x60 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x60 += einsum(t1.aa, (0, 1), x59, (2, 3, 1, 4), (0, 2, 3, 4))
    del x59
    x61 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x61 += einsum(t2.aaaa, (0, 1, 2, 3), x11, (1, 4, 5, 3), (4, 0, 5, 2))
    x62 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x62 += einsum(t2.aaaa, (0, 1, 2, 3), x61, (1, 4, 3, 5), (0, 4, 2, 5)) * -4.0
    del x61
    x63 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x63 += einsum(t2.abab, (0, 1, 2, 3), x7, (1, 4, 3, 5), (0, 4, 2, 5))
    del x7
    x64 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x64 += einsum(t2.abab, (0, 1, 2, 3), x63, (4, 1, 5, 3), (0, 4, 2, 5)) * -1.0
    del x63
    x65 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x65 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 1), (2, 3))
    x66 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x66 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 4, 1, 3), (2, 4))
    x67 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x67 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    x68 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x68 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x68 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x69 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x69 += einsum(t1.aa, (0, 1), x68, (0, 2, 1, 3), (2, 3))
    del x68
    x70 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x70 += einsum(x65, (0, 1), (1, 0)) * -1.0
    x70 += einsum(x66, (0, 1), (1, 0)) * 2.0
    x70 += einsum(x67, (0, 1), (1, 0))
    x70 += einsum(x69, (0, 1), (0, 1)) * -1.0
    x71 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x71 += einsum(x70, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x70
    x72 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x72 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x73 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x73 += einsum(x13, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4)) * -2.0
    x74 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x74 += einsum(t1.aa, (0, 1), x25, (2, 3, 4, 1), (2, 0, 4, 3))
    del x25
    x75 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x75 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x75 += einsum(x74, (0, 1, 2, 3), (3, 1, 2, 0))
    x76 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x76 += einsum(t1.aa, (0, 1), x75, (0, 2, 3, 4), (2, 3, 4, 1))
    del x75
    x77 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x77 += einsum(x72, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    del x72
    x77 += einsum(x73, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x73
    x77 += einsum(x76, (0, 1, 2, 3), (1, 0, 2, 3))
    del x76
    x78 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x78 += einsum(t1.aa, (0, 1), x77, (0, 2, 3, 4), (2, 3, 1, 4))
    del x77
    x79 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x79 += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3))
    del x60
    x79 += einsum(x62, (0, 1, 2, 3), (1, 0, 3, 2))
    del x62
    x79 += einsum(x64, (0, 1, 2, 3), (1, 0, 3, 2))
    del x64
    x79 += einsum(x71, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x71
    x79 += einsum(x78, (0, 1, 2, 3), (1, 0, 2, 3))
    del x78
    t2new_aaaa += einsum(x79, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x79, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x79
    x80 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x80 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x81 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x81 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x81 += einsum(x80, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x80
    t2new_aaaa += einsum(x81, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x81, (0, 1, 2, 3), (0, 1, 2, 3))
    del x81
    x82 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x82 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x83 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x83 += einsum(t2.aaaa, (0, 1, 2, 3), x46, (4, 5, 0, 1), (4, 5, 2, 3))
    del x46
    x84 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x84 += einsum(f.aa.oo, (0, 1), (0, 1))
    x84 += einsum(x34, (0, 1), (0, 1))
    del x34
    x85 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x85 += einsum(x84, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x84
    x86 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x86 += einsum(x29, (0, 1), (1, 0))
    del x29
    x86 += einsum(x30, (0, 1), (1, 0)) * 2.0
    del x30
    x86 += einsum(x31, (0, 1), (1, 0))
    del x31
    x86 += einsum(x33, (0, 1), (1, 0)) * -1.0
    del x33
    x87 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x87 += einsum(x86, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x86
    x88 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x88 += einsum(x82, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x82
    x88 += einsum(x83, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x83
    x88 += einsum(x85, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x85
    x88 += einsum(x87, (0, 1, 2, 3), (0, 1, 3, 2))
    del x87
    t2new_aaaa += einsum(x88, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x88, (0, 1, 2, 3), (1, 0, 2, 3))
    del x88
    x89 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x89 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x90 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x90 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x90 += einsum(x89, (0, 1, 2, 3), (3, 1, 2, 0))
    x90 += einsum(x74, (0, 1, 2, 3), (3, 1, 2, 0))
    del x74
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x90, (0, 4, 1, 5), (4, 5, 2, 3)) * 2.0
    del x90
    x91 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x91 += einsum(t1.aa, (0, 1), x89, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    del x89
    x91 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x91 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    t2new_aaaa += einsum(t1.aa, (0, 1), x91, (2, 3, 0, 4), (2, 3, 1, 4))
    del x91
    x92 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x92 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x92 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x93 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x93 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x93 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -0.5
    x94 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x94 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x94 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x95 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x95 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x95 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x95 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (4, 1, 5, 3))
    x95 += einsum(x92, (0, 1, 2, 3), x93, (0, 4, 3, 5), (1, 4, 2, 5)) * -2.0
    x95 += einsum(t1.bb, (0, 1), x94, (2, 1, 3, 4), (2, 0, 4, 3)) * -1.0
    x95 += einsum(t1.bb, (0, 1), x18, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x18
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x95, (1, 4, 3, 5), (0, 4, 2, 5))
    del x95
    x96 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x96 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x96 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -0.5
    x97 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x97 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x97 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x98 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x98 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x98 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x99 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x99 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x99 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x99 += einsum(x11, (0, 1, 2, 3), x96, (0, 4, 2, 5), (1, 4, 3, 5)) * -2.0
    x99 += einsum(t1.aa, (0, 1), x97, (2, 3, 1, 4), (2, 0, 3, 4)) * -1.0
    del x97
    x99 += einsum(t1.aa, (0, 1), x98, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x98
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x99, (0, 4, 2, 5), (4, 1, 5, 3)) * -1.0
    del x99
    x100 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x100 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x100 += einsum(t1.aa, (0, 1), v.aabb.ooov, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    x100 += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3))
    del x39
    x100 += einsum(v.aabb.ovov, (0, 1, 2, 3), x96, (0, 4, 1, 5), (4, 2, 5, 3)) * 2.0
    t2new_abab += einsum(t2.bbbb, (0, 1, 2, 3), x100, (4, 1, 5, 3), (4, 0, 5, 2)) * 2.0
    del x100
    x101 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x101 += einsum(v.aabb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x101 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 4), (2, 3, 4, 1))
    x101 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    x101 += einsum(v.aabb.ovov, (0, 1, 2, 3), x6, (4, 2, 1, 5), (0, 4, 3, 5))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x101, (0, 4, 3, 5), (4, 1, 2, 5))
    del x101
    x102 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x102 += einsum(t1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (2, 0, 3, 4))
    x103 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x103 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x103 += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3))
    x103 += einsum(t1.bb, (0, 1), x4, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x4
    t2new_abab += einsum(x103, (0, 1, 2, 3), x96, (0, 4, 2, 5), (4, 1, 5, 3)) * 2.0
    del x96, x103
    x104 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x104 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 4, 1), (0, 4, 2, 3))
    x105 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x105 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (0, 2, 3, 1))
    x105 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    x106 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x106 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x106 += einsum(x104, (0, 1, 2, 3), (1, 0, 3, 2))
    x106 += einsum(t1.aa, (0, 1), x105, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    del x105
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x106, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    del x106
    x107 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x107 += einsum(v.aabb.vvvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x107 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 2, 3, 4), (2, 1, 3, 4))
    x107 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 4), (2, 3, 4, 1))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x107, (2, 4, 3, 5), (0, 1, 4, 5)) * -1.0
    del x107
    x108 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x108 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 4, 1), (2, 3, 0, 4))
    x109 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x109 += einsum(v.aabb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x109 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (2, 1, 3, 4), (2, 0, 3, 4))
    x109 += einsum(x108, (0, 1, 2, 3), (1, 0, 3, 2))
    x109 += einsum(v.aabb.ovov, (0, 1, 2, 3), x6, (4, 5, 1, 3), (0, 4, 2, 5))
    del x6
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x109, (0, 4, 1, 5), (4, 5, 2, 3))
    del x109
    x110 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x110 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 1, 2, 3), (2, 3))
    x111 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x111 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    x112 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x112 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x113 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x113 += einsum(t1.bb, (0, 1), x94, (0, 1, 2, 3), (2, 3))
    del x94
    x114 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x114 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x114 += einsum(x110, (0, 1), (1, 0)) * -1.0
    x114 += einsum(x111, (0, 1), (1, 0))
    x114 += einsum(x112, (0, 1), (1, 0)) * 2.0
    x114 += einsum(x113, (0, 1), (1, 0)) * -1.0
    x114 += einsum(t1.bb, (0, 1), x9, (0, 2), (2, 1))
    t2new_abab += einsum(x114, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x114
    x115 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x115 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x115 += einsum(x65, (0, 1), (1, 0)) * -1.0
    del x65
    x115 += einsum(x66, (0, 1), (1, 0)) * 2.0
    del x66
    x115 += einsum(x67, (0, 1), (1, 0))
    del x67
    x115 += einsum(x69, (0, 1), (0, 1)) * -1.0
    del x69
    x115 += einsum(t1.aa, (0, 1), x13, (0, 2), (2, 1))
    del x13
    t2new_abab += einsum(x115, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    del x115
    x116 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x116 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x116 += einsum(x104, (0, 1, 2, 3), (0, 1, 3, 2))
    del x104
    x117 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x117 += einsum(v.aabb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x117 += einsum(x108, (0, 1, 2, 3), (1, 0, 2, 3))
    del x108
    x117 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (4, 0, 1, 5))
    x118 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x118 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (0, 2, 3, 1))
    x118 += einsum(x3, (0, 1, 2, 3), (0, 2, 1, 3))
    x118 += einsum(t1.aa, (0, 1), x116, (2, 3, 1, 4), (0, 3, 2, 4))
    del x116
    x118 += einsum(t1.aa, (0, 1), x117, (0, 2, 3, 4), (2, 4, 3, 1)) * -1.0
    del x117
    t2new_abab += einsum(t1.bb, (0, 1), x118, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x118
    x119 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x119 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x119 += einsum(t1.aa, (0, 1), v.aabb.vvvv, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_abab += einsum(t1.bb, (0, 1), x119, (2, 3, 1, 4), (2, 0, 3, 4))
    del x119
    x120 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x120 += einsum(v.aabb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x120 += einsum(t1.bb, (0, 1), v.aabb.oovv, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab += einsum(t1.aa, (0, 1), x120, (0, 2, 3, 4), (2, 3, 1, 4)) * -1.0
    del x120
    x121 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x121 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x122 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x122 += einsum(t2.abab, (0, 1, 2, 3), x102, (0, 4, 2, 5), (4, 1, 3, 5))
    del x102
    x123 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x123 += einsum(x14, (0, 1, 2, 3), x93, (0, 4, 2, 5), (1, 4, 3, 5)) * 2.0
    del x14, x93
    x124 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x124 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (4, 0, 5, 2))
    x125 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x125 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x125 += einsum(x124, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x124
    x126 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x126 += einsum(t2.abab, (0, 1, 2, 3), x125, (0, 4, 2, 5), (1, 4, 3, 5))
    del x125
    x127 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x127 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x127 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x128 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x128 += einsum(t1.bb, (0, 1), x127, (2, 3, 1, 4), (0, 2, 3, 4))
    del x127
    x129 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x129 += einsum(t2.bbbb, (0, 1, 2, 3), x128, (4, 1, 3, 5), (0, 4, 2, 5)) * -2.0
    del x128
    x130 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x130 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 5), (1, 4, 5, 3))
    x131 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x131 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x132 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x132 += einsum(t1.bb, (0, 1), x131, (2, 3, 4, 0), (2, 4, 3, 1))
    x133 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x133 += einsum(t1.bb, (0, 1), x121, (2, 3, 1, 4), (0, 2, 3, 4))
    x134 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x134 += einsum(t2.abab, (0, 1, 2, 3), x3, (0, 4, 5, 2), (4, 1, 5, 3))
    del x3
    x135 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x135 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x135 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x136 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x136 += einsum(t2.bbbb, (0, 1, 2, 3), x135, (1, 4, 5, 3), (0, 4, 5, 2)) * 2.0
    del x135
    x137 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x137 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x137 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3))
    x138 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x138 += einsum(t2.bbbb, (0, 1, 2, 3), x137, (4, 1, 5, 3), (0, 4, 5, 2)) * 2.0
    del x137
    x139 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x139 += einsum(x130, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x130
    x139 += einsum(x132, (0, 1, 2, 3), (0, 2, 1, 3))
    del x132
    x139 += einsum(x133, (0, 1, 2, 3), (0, 2, 1, 3))
    del x133
    x139 += einsum(x134, (0, 1, 2, 3), (0, 2, 1, 3))
    del x134
    x139 += einsum(x136, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x136
    x139 += einsum(x138, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x138
    x140 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x140 += einsum(t1.bb, (0, 1), x139, (2, 0, 3, 4), (2, 3, 1, 4))
    del x139
    x141 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x141 += einsum(x121, (0, 1, 2, 3), (0, 1, 2, 3))
    del x121
    x141 += einsum(x122, (0, 1, 2, 3), (0, 1, 2, 3))
    del x122
    x141 += einsum(x123, (0, 1, 2, 3), (1, 0, 3, 2))
    del x123
    x141 += einsum(x126, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x126
    x141 += einsum(x129, (0, 1, 2, 3), (1, 0, 2, 3))
    del x129
    x141 += einsum(x140, (0, 1, 2, 3), (0, 1, 2, 3))
    del x140
    t2new_bbbb += einsum(x141, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x141, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb += einsum(x141, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb += einsum(x141, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x141
    x142 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x142 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x143 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x143 += einsum(t2.bbbb, (0, 1, 2, 3), x131, (4, 5, 0, 1), (4, 5, 2, 3))
    del x131
    x144 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x144 += einsum(f.bb.oo, (0, 1), (0, 1))
    x144 += einsum(x20, (0, 1), (0, 1))
    del x20
    x145 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x145 += einsum(x144, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x144
    x146 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x146 += einsum(x15, (0, 1), (1, 0))
    del x15
    x146 += einsum(x16, (0, 1), (1, 0))
    del x16
    x146 += einsum(x17, (0, 1), (1, 0)) * 2.0
    del x17
    x146 += einsum(x19, (0, 1), (1, 0)) * -1.0
    del x19
    x147 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x147 += einsum(x146, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x146
    x148 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x148 += einsum(x142, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x142
    x148 += einsum(x143, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x143
    x148 += einsum(x145, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x145
    x148 += einsum(x147, (0, 1, 2, 3), (0, 1, 3, 2))
    del x147
    t2new_bbbb += einsum(x148, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x148, (0, 1, 2, 3), (1, 0, 2, 3))
    del x148
    x149 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x149 += einsum(t1.bb, (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x150 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x150 += einsum(t1.bb, (0, 1), x149, (2, 3, 1, 4), (0, 2, 3, 4))
    del x149
    x151 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x151 += einsum(t2.bbbb, (0, 1, 2, 3), x92, (1, 4, 5, 3), (0, 4, 2, 5))
    del x92
    x152 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x152 += einsum(t2.bbbb, (0, 1, 2, 3), x151, (4, 1, 5, 3), (0, 4, 2, 5)) * -4.0
    del x151
    x153 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x153 += einsum(x110, (0, 1), (1, 0)) * -1.0
    del x110
    x153 += einsum(x111, (0, 1), (1, 0))
    del x111
    x153 += einsum(x112, (0, 1), (1, 0)) * 2.0
    del x112
    x153 += einsum(x113, (0, 1), (1, 0)) * -1.0
    del x113
    x154 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x154 += einsum(x153, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x153
    x155 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x155 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x156 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x156 += einsum(x9, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4)) * -2.0
    del x9
    x157 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x157 += einsum(t1.bb, (0, 1), x1, (2, 3, 4, 1), (2, 0, 4, 3))
    del x1
    x158 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x158 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x158 += einsum(x157, (0, 1, 2, 3), (3, 1, 2, 0))
    x159 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x159 += einsum(t1.bb, (0, 1), x158, (0, 2, 3, 4), (2, 3, 4, 1))
    del x158
    x160 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x160 += einsum(x155, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    del x155
    x160 += einsum(x156, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x156
    x160 += einsum(x159, (0, 1, 2, 3), (1, 0, 2, 3))
    del x159
    x161 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x161 += einsum(t1.bb, (0, 1), x160, (0, 2, 3, 4), (2, 3, 1, 4))
    del x160
    x162 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x162 += einsum(x150, (0, 1, 2, 3), (0, 1, 2, 3))
    del x150
    x162 += einsum(x152, (0, 1, 2, 3), (1, 0, 3, 2))
    del x152
    x162 += einsum(x154, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x154
    x162 += einsum(x161, (0, 1, 2, 3), (1, 0, 2, 3))
    del x161
    t2new_bbbb += einsum(x162, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x162, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x162
    x163 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x163 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x164 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x164 += einsum(t2.abab, (0, 1, 2, 3), x11, (0, 4, 5, 2), (4, 1, 5, 3))
    del x11
    x165 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x165 += einsum(t2.abab, (0, 1, 2, 3), x164, (0, 4, 2, 5), (1, 4, 3, 5)) * -1.0
    del x164
    x166 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x166 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x166 += einsum(x163, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x163
    x166 += einsum(x165, (0, 1, 2, 3), (0, 1, 2, 3))
    del x165
    t2new_bbbb += einsum(x166, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x166, (0, 1, 2, 3), (0, 1, 2, 3))
    del x166
    x167 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x167 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x168 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x168 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x168 += einsum(x167, (0, 1, 2, 3), (3, 1, 0, 2))
    x168 += einsum(x157, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x157
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x168, (0, 4, 5, 1), (5, 4, 2, 3)) * -2.0
    del x168
    x169 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x169 += einsum(t1.bb, (0, 1), x167, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    del x167
    x169 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x169 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    t2new_bbbb += einsum(t1.bb, (0, 1), x169, (2, 3, 0, 4), (2, 3, 1, 4))
    del x169

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t1new": t1new, "t2new": t2new}

def energy_perturbative(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    # T3 amplitude
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x0 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ooov, (4, 1, 5, 6), (0, 4, 5, 2, 3, 6))
    t3_aaaaaa = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3_aaaaaa += einsum(x0, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x1 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3_aaaaaa += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x1
    x2 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x2 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ooov, (4, 0, 5, 6), (4, 1, 5, 2, 3, 6))
    x3 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x3 += einsum(t2.abab, (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 5, 6), (0, 4, 5, 2, 3, 6))
    x4 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x4 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvov, (4, 2, 5, 6), (0, 1, 5, 4, 3, 6))
    x5 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x5 += einsum(t2.abab, (0, 1, 2, 3), v.bbbb.ovvv, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    x6 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x6 += einsum(x2, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x2
    x6 += einsum(x3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x3
    x6 += einsum(x4, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x4
    x6 += einsum(x5, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x5
    t3_babbab = np.zeros((nocc[1], nocc[0], nocc[1], nvir[1], nvir[0], nvir[1]), dtype=types[float])
    t3_babbab += einsum(x6, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3_babbab += einsum(x6, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3_babbab += einsum(x6, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3_babbab += einsum(x6, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x6
    x7 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x7 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovvv, (4, 5, 6, 3), (4, 0, 1, 5, 2, 6))
    t3_babbab += einsum(x7, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3_babbab += einsum(x7, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x7
    x8 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x8 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovoo, (4, 5, 6, 1), (4, 0, 6, 5, 2, 3))
    t3_babbab += einsum(x8, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3_babbab += einsum(x8, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x8
    x9 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x9 += einsum(t2.abab, (0, 1, 2, 3), v.aaaa.ooov, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3))
    x10 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x10 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovoo, (4, 5, 6, 1), (0, 4, 6, 2, 5, 3))
    x11 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x11 += einsum(t2.abab, (0, 1, 2, 3), v.aaaa.ovvv, (4, 5, 6, 2), (0, 4, 1, 5, 6, 3))
    x12 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x12 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovvv, (4, 5, 6, 3), (0, 4, 1, 2, 5, 6))
    x13 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x13 += einsum(x9, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x9
    x13 += einsum(x10, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x10
    x13 += einsum(x11, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x11
    x13 += einsum(x12, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x12
    t3_abaaba = np.zeros((nocc[0], nocc[1], nocc[0], nvir[0], nvir[1], nvir[0]), dtype=types[float])
    t3_abaaba += einsum(x13, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3_abaaba += einsum(x13, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3_abaaba += einsum(x13, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3_abaaba += einsum(x13, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x13
    x14 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x14 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ooov, (4, 1, 5, 6), (0, 4, 5, 2, 3, 6))
    t3_abaaba += einsum(x14, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3_abaaba += einsum(x14, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x14
    x15 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x15 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.vvov, (4, 3, 5, 6), (0, 1, 5, 2, 4, 6))
    t3_abaaba += einsum(x15, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3_abaaba += einsum(x15, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x15
    x16 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x16 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 5, 6), (0, 4, 5, 2, 3, 6))
    t3_bbbbbb = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3_bbbbbb += einsum(x16, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x16
    x17 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x17 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3_bbbbbb += einsum(x17, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x17
    t3_aaaaaa /= direct_sum(
            "ia+jb+kc->ijkabc",
            direct_sum("i-a->ia", np.diag(f.aa.oo), np.diag(f.aa.vv)),
            direct_sum("i-a->ia", np.diag(f.aa.oo), np.diag(f.aa.vv)),
            direct_sum("i-a->ia", np.diag(f.aa.oo), np.diag(f.aa.vv)),
    )
    t3_babbab /= direct_sum(
            "ia+jb+kc->ijkabc",
            direct_sum("i-a->ia", np.diag(f.bb.oo), np.diag(f.bb.vv)),
            direct_sum("i-a->ia", np.diag(f.aa.oo), np.diag(f.aa.vv)),
            direct_sum("i-a->ia", np.diag(f.bb.oo), np.diag(f.bb.vv)),
    )
    t3_abaaba /= direct_sum(
            "ia+jb+kc->ijkabc",
            direct_sum("i-a->ia", np.diag(f.aa.oo), np.diag(f.aa.vv)),
            direct_sum("i-a->ia", np.diag(f.bb.oo), np.diag(f.bb.vv)),
            direct_sum("i-a->ia", np.diag(f.aa.oo), np.diag(f.aa.vv)),
    )
    t3_bbbbbb /= direct_sum(
            "ia+jb+kc->ijkabc",
            direct_sum("i-a->ia", np.diag(f.bb.oo), np.diag(f.bb.vv)),
            direct_sum("i-a->ia", np.diag(f.bb.oo), np.diag(f.bb.vv)),
            direct_sum("i-a->ia", np.diag(f.bb.oo), np.diag(f.bb.vv)),
    )

    # energy
    x0 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x0 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3_abaaba, (0, 4, 2, 1, 5, 3), (4, 5))
    e_pert = 0
    e_pert += einsum(l1.bb, (0, 1), x0, (1, 0), ())
    del x0
    x1 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3_abaaba, (4, 2, 0, 5, 3, 1), (4, 5))
    e_pert += einsum(l1.aa, (0, 1), x1, (1, 0), ()) * 2.0
    del x1
    x2 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x2 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3_bbbbbb, (4, 0, 2, 5, 1, 3), (4, 5))
    e_pert += einsum(l1.bb, (0, 1), x2, (1, 0), ()) * 3.0
    del x2
    x3 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x3 += einsum(l2.abab, (0, 1, 2, 3), t3_babbab, (4, 2, 5, 6, 0, 1), (3, 4, 5, 6))
    e_pert += einsum(v.bbbb.ooov, (0, 1, 2, 3), x3, (0, 2, 1, 3), ()) * -2.0
    del x3
    x4 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x4 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3_aaaaaa, (4, 0, 2, 5, 1, 3), (4, 5))
    e_pert += einsum(l1.aa, (0, 1), x4, (1, 0), ()) * 3.0
    del x4
    x5 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x5 += einsum(l2.bbbb, (0, 1, 2, 3), t3_bbbbbb, (4, 5, 3, 6, 0, 1), (2, 4, 5, 6))
    e_pert += einsum(v.bbbb.ooov, (0, 1, 2, 3), x5, (0, 1, 2, 3), ()) * 6.0
    del x5
    x6 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3_babbab, (0, 4, 2, 1, 5, 3), (4, 5))
    e_pert += einsum(l1.aa, (0, 1), x6, (1, 0), ())
    del x6
    x7 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x7 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3_babbab, (4, 0, 2, 5, 1, 3), (4, 5))
    e_pert += einsum(l1.bb, (0, 1), x7, (1, 0), ()) * 2.0
    del x7
    x8 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x8 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), t3_abaaba, (4, 5, 0, 1, 6, 3), (4, 5, 2, 6))
    x8 += einsum(v.aabb.ovvv, (0, 1, 2, 3), t3_abaaba, (4, 5, 0, 6, 3, 1), (4, 5, 6, 2)) * -1.0
    x8 += einsum(v.aabb.vvov, (0, 1, 2, 3), t3_babbab, (4, 5, 2, 6, 1, 3), (5, 4, 0, 6)) * -1.0
    x8 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), t3_babbab, (4, 5, 0, 1, 6, 3), (5, 4, 6, 2))
    e_pert += einsum(l2.abab, (0, 1, 2, 3), x8, (2, 3, 0, 1), ()) * -2.0
    del x8
    x9 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x9 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), t3_aaaaaa, (4, 5, 0, 6, 3, 1), (4, 5, 6, 2)) * -3.0
    x9 += einsum(v.aabb.vvov, (0, 1, 2, 3), t3_abaaba, (4, 2, 5, 6, 3, 1), (4, 5, 6, 0)) * -1.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), x9, (3, 2, 0, 1), ()) * 2.0
    del x9
    x10 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x10 += einsum(v.aabb.ovvv, (0, 1, 2, 3), t3_babbab, (4, 0, 5, 6, 1, 3), (4, 5, 6, 2)) * -1.0
    x10 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), t3_bbbbbb, (4, 5, 0, 6, 3, 1), (4, 5, 6, 2)) * -3.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), x10, (3, 2, 0, 1), ()) * 2.0
    del x10
    x11 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x11 += einsum(l2.abab, (0, 1, 2, 3), t3_abaaba, (4, 5, 2, 6, 1, 0), (4, 3, 5, 6))
    x11 += einsum(l2.bbbb, (0, 1, 2, 3), t3_babbab, (4, 5, 3, 0, 6, 1), (5, 2, 4, 6))
    e_pert += einsum(v.aabb.ovoo, (0, 1, 2, 3), x11, (0, 3, 2, 1), ()) * -2.0
    del x11
    x12 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x12 += einsum(l2.aaaa, (0, 1, 2, 3), t3_aaaaaa, (4, 5, 3, 6, 0, 1), (2, 4, 5, 6)) * 3.0
    x12 += einsum(l2.abab, (0, 1, 2, 3), t3_abaaba, (4, 3, 5, 6, 1, 0), (2, 4, 5, 6))
    e_pert += einsum(v.aaaa.ooov, (0, 1, 2, 3), x12, (1, 0, 2, 3), ()) * 2.0
    del x12
    x13 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x13 += einsum(l2.aaaa, (0, 1, 2, 3), t3_abaaba, (4, 5, 3, 0, 6, 1), (2, 4, 5, 6))
    x13 += einsum(l2.abab, (0, 1, 2, 3), t3_babbab, (4, 5, 3, 6, 0, 1), (2, 5, 4, 6))
    e_pert += einsum(v.aabb.ooov, (0, 1, 2, 3), x13, (1, 0, 2, 3), ()) * -2.0
    del x13
    e_pert /= 2

    return e_pert

