# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 2), ()) * -1.0
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 2, 1, 3), ())
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x0 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x0 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x1 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum(f.aa.ov, (0, 1), (0, 1))
    x1 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    x1 += einsum(t1.aa, (0, 1), x0, (0, 2, 1, 3), (2, 3)) * -0.5
    del x0
    e_cc += einsum(t1.aa, (0, 1), x1, (0, 1), ())
    del x1
    x2 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x2 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x2 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x3 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x3 += einsum(f.bb.ov, (0, 1), (0, 1))
    x3 += einsum(t1.bb, (0, 1), x2, (0, 2, 1, 3), (2, 3)) * -0.5
    del x2
    e_cc += einsum(t1.bb, (0, 1), x3, (0, 1), ())
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
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum(f.aa.vv, (0, 1), t2.abab, (2, 3, 1, 4), (2, 3, 0, 4))
    t2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_abab += einsum(f.bb.vv, (0, 1), t2.abab, (2, 3, 4, 1), (2, 3, 4, 0))
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
    x3 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x3 += einsum(t1.aa, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (3, 4, 0, 2))
    x4 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x4 += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2))
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x4, (1, 3, 0, 4), (4, 2)) * -1.0
    del x4
    x5 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x5 += einsum(t2.abab, (0, 1, 2, 3), (1, 3, 0, 2))
    x5 += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (2, 3, 0, 1))
    t1new_aa += einsum(v.aabb.vvov, (0, 1, 2, 3), x5, (2, 3, 4, 1), (4, 0))
    t1new_bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), x5, (4, 3, 0, 1), (4, 2))
    del x5
    x6 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x6 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x6 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x6, (0, 4, 3, 1), (4, 2)) * -2.0
    del x6
    x7 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x7 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(x7, (0, 1), (0, 1))
    x8 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x8 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x8 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x9 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x9 += einsum(t1.bb, (0, 1), x8, (0, 2, 1, 3), (2, 3))
    del x8
    x10 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x10 += einsum(f.bb.ov, (0, 1), (0, 1))
    x10 += einsum(x7, (0, 1), (0, 1))
    del x7
    x10 += einsum(x9, (0, 1), (0, 1)) * -1.0
    del x9
    t1new_aa += einsum(x10, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    t1new_bb += einsum(x10, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    x11 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x11 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x11 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x12 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x12 += einsum(t1.aa, (0, 1), x11, (0, 2, 1, 3), (2, 3))
    del x11
    x13 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x13 += einsum(f.aa.ov, (0, 1), (0, 1))
    x13 += einsum(x0, (0, 1), (0, 1))
    del x0
    x13 += einsum(x12, (0, 1), (0, 1)) * -1.0
    del x12
    t1new_aa += einsum(x13, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_bb += einsum(x13, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    x14 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x14 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x14 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_aa += einsum(t1.aa, (0, 1), x14, (0, 2, 1, 3), (2, 3)) * -1.0
    del x14
    x15 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x15 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x15 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x16 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x16 += einsum(f.aa.oo, (0, 1), (0, 1))
    x16 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3))
    x16 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (4, 0))
    x16 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 1, 3), (4, 0)) * 2.0
    x16 += einsum(t1.aa, (0, 1), x15, (0, 2, 3, 1), (3, 2)) * -1.0
    del x15
    x16 += einsum(t1.aa, (0, 1), x13, (2, 1), (2, 0))
    del x13
    t1new_aa += einsum(t1.aa, (0, 1), x16, (0, 2), (2, 1)) * -1.0
    del x16
    x17 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x17 += einsum(f.aa.vv, (0, 1), (0, 1))
    x17 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_aa += einsum(t1.aa, (0, 1), x17, (1, 2), (0, 2))
    del x17
    x18 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x18 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x19 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x19 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x19 += einsum(x18, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), x19, (4, 0, 1, 3), (4, 2)) * 2.0
    del x19
    x20 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x20 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x20 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (4, 0, 2, 3))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x20, (1, 4, 0, 2), (4, 3)) * -1.0
    del x20
    x21 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x21 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x21 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_bb += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x21, (0, 4, 3, 1), (4, 2)) * -2.0
    del x21
    x22 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x22 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x22 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_bb += einsum(t1.bb, (0, 1), x22, (0, 2, 1, 3), (2, 3)) * -1.0
    del x22
    x23 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x23 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x23 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x24 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x24 += einsum(f.bb.oo, (0, 1), (0, 1)) * 0.5
    x24 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3)) * 0.5
    x24 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (4, 0)) * -1.0
    x24 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (4, 1)) * 0.5
    x24 += einsum(t1.bb, (0, 1), x23, (2, 3, 0, 1), (3, 2)) * -0.5
    del x23
    x24 += einsum(t1.bb, (0, 1), x10, (2, 1), (2, 0)) * 0.5
    del x10
    t1new_bb += einsum(t1.bb, (0, 1), x24, (0, 2), (2, 1)) * -2.0
    del x24
    x25 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x25 += einsum(f.bb.vv, (0, 1), (0, 1))
    x25 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(t1.bb, (0, 1), x25, (1, 2), (0, 2))
    del x25
    x26 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x26 += einsum(t1.aa, (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x27 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x27 += einsum(t1.aa, (0, 1), x26, (2, 3, 1, 4), (0, 2, 3, 4))
    del x26
    x28 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x28 += einsum(f.aa.ov, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4))
    x29 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x29 += einsum(t1.aa, (0, 1), x1, (2, 3, 4, 1), (2, 0, 4, 3))
    del x1
    x30 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x30 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x30 += einsum(x29, (0, 1, 2, 3), (3, 1, 2, 0))
    del x29
    x31 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x31 += einsum(t1.aa, (0, 1), x30, (0, 2, 3, 4), (2, 3, 4, 1))
    del x30
    x32 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x32 += einsum(x28, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    del x28
    x32 += einsum(x31, (0, 1, 2, 3), (1, 0, 2, 3))
    del x31
    x33 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x33 += einsum(t1.aa, (0, 1), x32, (0, 2, 3, 4), (2, 3, 1, 4))
    del x32
    x34 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x34 += einsum(x27, (0, 1, 2, 3), (0, 1, 2, 3))
    del x27
    x34 += einsum(x33, (0, 1, 2, 3), (1, 0, 2, 3))
    del x33
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x34, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x34
    x35 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x35 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x36 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x36 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x37 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x37 += einsum(t1.aa, (0, 1), x36, (2, 3, 4, 0), (2, 4, 3, 1))
    del x36
    x38 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x38 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x38 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x38 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    x39 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x39 += einsum(t1.aa, (0, 1), x38, (2, 3, 1, 4), (0, 2, 3, 4))
    del x38
    x40 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x40 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x40 += einsum(x37, (0, 1, 2, 3), (0, 2, 1, 3))
    del x37
    x40 += einsum(x39, (0, 1, 2, 3), (0, 2, 1, 3))
    del x39
    x41 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x41 += einsum(t1.aa, (0, 1), x40, (2, 0, 3, 4), (2, 3, 1, 4))
    del x40
    x42 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x42 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    del x35
    x42 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3))
    del x41
    t2new_aaaa += einsum(x42, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x42, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa += einsum(x42, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x42, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x42
    x43 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x43 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x44 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x44 += einsum(x43, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x43
    t2new_aaaa += einsum(x44, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3))
    del x44
    x45 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x45 += einsum(f.aa.ov, (0, 1), t1.aa, (2, 1), (0, 2))
    x46 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x46 += einsum(f.aa.oo, (0, 1), (0, 1))
    x46 += einsum(x45, (0, 1), (0, 1))
    del x45
    t2new_abab += einsum(x46, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    x47 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x47 += einsum(x46, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x46
    t2new_aaaa += einsum(x47, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_aaaa += einsum(x47, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x47
    x48 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x48 += einsum(t1.aa, (0, 1), v.aabb.vvov, (2, 1, 3, 4), (3, 4, 0, 2))
    t2new_abab += einsum(x48, (0, 1, 2, 3), (2, 0, 3, 1))
    x49 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x49 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x49 += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3))
    del x48
    x50 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x50 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x50 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    x51 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x51 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x51 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (2, 1, 3, 4), (3, 4, 2, 0))
    x51 += einsum(t1.bb, (0, 1), x50, (2, 1, 3, 4), (0, 2, 4, 3))
    del x50
    x52 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x52 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x52 += einsum(f.bb.ov, (0, 1), t2.abab, (2, 3, 4, 1), (0, 3, 2, 4))
    x52 += einsum(t1.aa, (0, 1), v.aabb.vvoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x52 += einsum(t1.bb, (0, 1), x49, (2, 1, 3, 4), (2, 0, 3, 4))
    del x49
    x52 += einsum(t1.aa, (0, 1), x51, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x51
    t2new_abab += einsum(t1.bb, (0, 1), x52, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    del x52
    x53 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x53 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (2, 3, 0, 1))
    x53 += einsum(t1.aa, (0, 1), v.aabb.vvvv, (2, 1, 3, 4), (3, 4, 0, 2))
    t2new_abab += einsum(t1.bb, (0, 1), x53, (1, 2, 3, 4), (3, 0, 4, 2))
    del x53
    x54 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x54 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x54 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (3, 4, 0, 2))
    x55 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x55 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x55 += einsum(f.aa.ov, (0, 1), t2.abab, (2, 3, 1, 4), (3, 4, 0, 2))
    x55 += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2))
    del x3
    x55 += einsum(t1.bb, (0, 1), x54, (1, 2, 3, 4), (0, 2, 4, 3))
    del x54
    t2new_abab += einsum(t1.aa, (0, 1), x55, (2, 3, 0, 4), (4, 2, 1, 3)) * -1.0
    del x55
    x56 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x56 += einsum(f.bb.ov, (0, 1), t1.bb, (2, 1), (0, 2))
    x57 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x57 += einsum(f.bb.oo, (0, 1), (0, 1))
    x57 += einsum(x56, (0, 1), (0, 1))
    del x56
    t2new_abab += einsum(x57, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    x58 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x58 += einsum(t1.bb, (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x59 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x59 += einsum(t1.bb, (0, 1), x58, (2, 3, 1, 4), (0, 2, 3, 4))
    del x58
    x60 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x60 += einsum(f.bb.ov, (0, 1), t2.bbbb, (2, 3, 4, 1), (0, 2, 3, 4))
    x61 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x61 += einsum(t1.bb, (0, 1), x18, (2, 3, 4, 1), (2, 0, 4, 3))
    del x18
    x62 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x62 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x62 += einsum(x61, (0, 1, 2, 3), (3, 1, 2, 0))
    del x61
    x63 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x63 += einsum(t1.bb, (0, 1), x62, (0, 2, 3, 4), (2, 3, 4, 1))
    del x62
    x64 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x64 += einsum(x60, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    del x60
    x64 += einsum(x63, (0, 1, 2, 3), (1, 0, 2, 3))
    del x63
    x65 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x65 += einsum(t1.bb, (0, 1), x64, (0, 2, 3, 4), (2, 3, 1, 4))
    del x64
    x66 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x66 += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3))
    del x59
    x66 += einsum(x65, (0, 1, 2, 3), (1, 0, 2, 3))
    del x65
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb += einsum(x66, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x66, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x66
    x67 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x67 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x68 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x68 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x69 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x69 += einsum(t1.bb, (0, 1), x68, (2, 3, 4, 0), (2, 4, 3, 1))
    del x68
    x70 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x70 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x70 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x70 += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3))
    x71 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x71 += einsum(t1.bb, (0, 1), x70, (2, 3, 1, 4), (0, 2, 3, 4))
    del x70
    x72 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x72 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x72 += einsum(x69, (0, 1, 2, 3), (0, 2, 1, 3))
    del x69
    x72 += einsum(x71, (0, 1, 2, 3), (0, 2, 1, 3))
    del x71
    x73 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x73 += einsum(t1.bb, (0, 1), x72, (2, 0, 3, 4), (2, 3, 1, 4))
    del x72
    x74 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x74 += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3))
    del x67
    x74 += einsum(x73, (0, 1, 2, 3), (0, 1, 2, 3))
    del x73
    t2new_bbbb += einsum(x74, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x74, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb += einsum(x74, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb += einsum(x74, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x74
    x75 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x75 += einsum(x57, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x57
    t2new_bbbb += einsum(x75, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_bbbb += einsum(x75, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x75
    x76 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x76 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x77 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x77 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x77 += einsum(x76, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x76
    t2new_bbbb += einsum(x77, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x77, (0, 1, 2, 3), (0, 1, 2, 3))
    del x77

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t1new": t1new, "t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    l1new = Namespace()
    l2new = Namespace()

    # L amplitudes
    l1new_aa = np.zeros((nvir[0], nocc[0]), dtype=types[float])
    l1new_aa += einsum(f.aa.ov, (0, 1), (1, 0))
    l1new_bb = np.zeros((nvir[1], nocc[1]), dtype=types[float])
    l1new_bb += einsum(f.bb.ov, (0, 1), (1, 0))
    l2new_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    l2new_aaaa += einsum(l2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 0, 5, 1), (4, 5, 2, 3)) * 2.0
    l2new_abab = np.zeros((nvir[0], nvir[1], nocc[0], nocc[1]), dtype=types[float])
    l2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (1, 3, 0, 2))
    l2new_abab += einsum(l1.aa, (0, 1), v.aabb.vvov, (2, 0, 3, 4), (2, 4, 1, 3))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 0, 5, 1), (4, 5, 2, 3))
    l2new_abab += einsum(l1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 0), (3, 4, 2, 1))
    l2new_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    l2new_bbbb += einsum(l2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 0, 5, 1), (4, 5, 2, 3)) * 2.0
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x0 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), (2, 3, 4, 5))
    l1new_aa += einsum(v.aaaa.ooov, (0, 1, 2, 3), x0, (4, 1, 2, 0), (3, 4)) * 2.0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    l1new_aa += einsum(v.aaaa.oovv, (0, 1, 2, 3), x1, (4, 0, 1, 3), (2, 4)) * -2.0
    l2new_abab += einsum(v.aabb.ooov, (0, 1, 2, 3), x1, (4, 0, 1, 5), (5, 3, 4, 2)) * -2.0
    x2 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x2 += einsum(t1.aa, (0, 1), x1, (2, 3, 4, 1), (2, 3, 0, 4))
    l1new_aa += einsum(v.aaaa.ooov, (0, 1, 2, 3), x2, (4, 1, 0, 2), (3, 4)) * -2.0
    l2new_aaaa += einsum(v.aaaa.ovov, (0, 1, 2, 3), x2, (4, 5, 0, 2), (3, 1, 5, 4)) * 2.0
    x3 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x3 += einsum(t1.bb, (0, 1), l2.abab, (2, 1, 3, 4), (4, 0, 3, 2))
    l1new_aa += einsum(v.aabb.vvoo, (0, 1, 2, 3), x3, (2, 3, 4, 1), (0, 4)) * -1.0
    l2new_abab += einsum(v.aabb.vvov, (0, 1, 2, 3), x3, (4, 2, 5, 1), (0, 3, 5, 4)) * -1.0
    x4 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x4 += einsum(t1.bb, (0, 1), v.aabb.vvvv, (2, 3, 4, 1), (0, 4, 2, 3))
    l1new_aa += einsum(l2.abab, (0, 1, 2, 3), x4, (3, 1, 4, 0), (4, 2))
    del x4
    x5 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x5 += einsum(t1.aa, (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    l1new_aa += einsum(l2.aaaa, (0, 1, 2, 3), x5, (3, 4, 0, 1), (4, 2)) * 2.0
    del x5
    x6 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x6 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    l1new_aa += einsum(x1, (0, 1, 2, 3), x6, (1, 2, 4, 3), (4, 0)) * 2.0
    x7 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x7 += einsum(t1.aa, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (3, 4, 0, 2))
    l2new_abab += einsum(x1, (0, 1, 2, 3), x7, (4, 5, 1, 2), (3, 5, 0, 4)) * -2.0
    x8 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x9 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x9 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x9 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x9 += einsum(x8, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x9 += einsum(x8, (0, 1, 2, 3), (2, 0, 1, 3))
    x10 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x10 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x10 += einsum(x7, (0, 1, 2, 3), (0, 1, 3, 2))
    x11 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x11 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (0, 4, 2, 3))
    x12 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x12 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x12 += einsum(x11, (0, 1, 2, 3), (1, 0, 2, 3))
    x13 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x13 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    x14 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x14 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x14 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x15 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x15 += einsum(t1.aa, (0, 1), x14, (0, 2, 1, 3), (2, 3))
    x16 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x16 += einsum(f.aa.ov, (0, 1), (0, 1))
    x16 += einsum(x13, (0, 1), (0, 1))
    x16 += einsum(x15, (0, 1), (0, 1)) * -1.0
    l2new_abab += einsum(l1.bb, (0, 1), x16, (2, 3), (3, 0, 2, 1))
    x17 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x17 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (3, 4, 0, 2))
    x18 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x18 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x18 += einsum(x17, (0, 1, 2, 3), (1, 0, 3, 2))
    x19 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x19 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x20 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x20 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (1, 5, 0, 4))
    x21 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x21 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x21 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    l2new_abab += einsum(x21, (0, 1, 2, 3), x3, (4, 0, 2, 5), (5, 1, 3, 4))
    l2new_abab += einsum(l1.aa, (0, 1), x21, (2, 3, 1, 4), (0, 3, 4, 2)) * -1.0
    x22 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x22 += einsum(t1.bb, (0, 1), x21, (2, 1, 3, 4), (0, 2, 3, 4))
    x23 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x23 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x23 += einsum(x19, (0, 1, 2, 3), (1, 0, 3, 2))
    x23 += einsum(x20, (0, 1, 2, 3), (1, 0, 3, 2))
    del x20
    x23 += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2))
    del x22
    x24 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x24 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x24 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    x24 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovvv, (4, 2, 5, 3), (1, 5, 0, 4))
    x24 += einsum(t2.abab, (0, 1, 2, 3), x9, (0, 4, 5, 2), (1, 3, 4, 5)) * -1.0
    del x9
    x24 += einsum(t2.bbbb, (0, 1, 2, 3), x10, (1, 3, 4, 5), (0, 2, 5, 4)) * 2.0
    x24 += einsum(t2.abab, (0, 1, 2, 3), x12, (1, 4, 5, 2), (4, 3, 0, 5)) * -1.0
    x24 += einsum(x16, (0, 1), t2.abab, (2, 3, 1, 4), (3, 4, 2, 0))
    x24 += einsum(t1.bb, (0, 1), x18, (1, 2, 3, 4), (0, 2, 4, 3))
    del x18
    x24 += einsum(t1.bb, (0, 1), x23, (0, 2, 3, 4), (2, 1, 4, 3)) * -1.0
    l1new_aa += einsum(l2.abab, (0, 1, 2, 3), x24, (3, 1, 2, 4), (0, 4)) * -1.0
    del x24
    x25 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x25 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x25 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x25 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3))
    x25 += einsum(x8, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x26 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x26 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x26 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x26 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    del x6
    x27 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x27 += einsum(t1.aa, (0, 1), x8, (2, 3, 4, 1), (0, 2, 3, 4))
    x28 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x28 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x29 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x29 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x29 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 5, 2), (4, 0, 5, 1)) * -1.0
    x29 += einsum(x27, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x29 += einsum(x28, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    x29 += einsum(x28, (0, 1, 2, 3), (3, 0, 2, 1))
    x30 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x30 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x30 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 3), (4, 0, 1, 5)) * -0.5
    x30 += einsum(t2.aaaa, (0, 1, 2, 3), x25, (4, 5, 1, 3), (5, 0, 4, 2)) * -1.0
    del x25
    x30 += einsum(t2.abab, (0, 1, 2, 3), x21, (1, 3, 4, 5), (5, 0, 4, 2)) * 0.5
    del x21
    x30 += einsum(x16, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4)) * 0.5
    x30 += einsum(t1.aa, (0, 1), x26, (2, 3, 1, 4), (3, 2, 0, 4)) * 0.5
    del x26
    x30 += einsum(t1.aa, (0, 1), x29, (0, 2, 3, 4), (3, 4, 2, 1)) * 0.5
    del x29
    l1new_aa += einsum(l2.aaaa, (0, 1, 2, 3), x30, (4, 3, 2, 1), (0, 4)) * -4.0
    del x30
    x31 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x31 += einsum(l2.abab, (0, 1, 2, 3), (3, 1, 2, 0))
    x31 += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (4, 5, 2, 0)) * 2.0
    x31 += einsum(t1.bb, (0, 1), x3, (0, 2, 3, 4), (2, 1, 3, 4)) * -1.0
    x31 += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 1, 5), (4, 5, 2, 0)) * 2.0
    l1new_aa += einsum(v.aabb.vvov, (0, 1, 2, 3), x31, (2, 3, 4, 1), (0, 4))
    del x31
    x32 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x32 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x32 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x33 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x33 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 4, 0, 5))
    x33 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 5, 1), (2, 4, 0, 5)) * 4.0
    l1new_aa += einsum(x32, (0, 1, 2, 3), x33, (4, 0, 3, 2), (1, 4))
    del x33
    x34 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x34 += einsum(t1.aa, (0, 1), l2.abab, (1, 2, 3, 4), (4, 2, 3, 0))
    l1new_bb += einsum(v.aabb.oovv, (0, 1, 2, 3), x34, (4, 3, 0, 1), (2, 4)) * -1.0
    l2new_abab += einsum(v.aabb.ovvv, (0, 1, 2, 3), x34, (4, 3, 5, 0), (1, 2, 5, 4)) * -1.0
    l2new_abab += einsum(x16, (0, 1), x34, (2, 3, 4, 0), (1, 3, 4, 2)) * -1.0
    x35 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x35 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), (3, 5, 2, 4))
    x36 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x36 += einsum(t1.bb, (0, 1), x34, (2, 1, 3, 4), (2, 0, 3, 4))
    l2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), x36, (4, 2, 5, 0), (1, 3, 5, 4))
    x37 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x37 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    x37 += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3))
    x38 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x38 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    x39 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x39 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    x40 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x40 += einsum(x38, (0, 1), (0, 1))
    x40 += einsum(x39, (0, 1), (0, 1)) * 2.0
    x41 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x41 += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), (3, 4, 1, 2))
    x41 += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3))
    x41 += einsum(t2.bbbb, (0, 1, 2, 3), x34, (1, 3, 4, 5), (0, 2, 4, 5)) * 2.0
    x41 += einsum(t2.abab, (0, 1, 2, 3), x3, (1, 4, 5, 2), (4, 3, 5, 0)) * -1.0
    x41 += einsum(t2.abab, (0, 1, 2, 3), x1, (4, 0, 5, 2), (1, 3, 4, 5)) * -2.0
    x41 += einsum(t1.bb, (0, 1), x37, (0, 2, 3, 4), (2, 1, 3, 4)) * -1.0
    x41 += einsum(t1.bb, (0, 1), x40, (2, 3), (0, 1, 2, 3))
    l1new_aa += einsum(v.aabb.ovov, (0, 1, 2, 3), x41, (2, 3, 4, 0), (1, 4)) * -1.0
    del x41
    x42 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x42 += einsum(l2.aaaa, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x42 += einsum(t1.aa, (0, 1), x1, (2, 0, 3, 4), (2, 3, 4, 1))
    l1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x42, (4, 0, 3, 1), (2, 4)) * -2.0
    del x42
    x43 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x43 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), (1, 5, 2, 4))
    x43 += einsum(t1.bb, (0, 1), x34, (0, 2, 3, 4), (1, 2, 3, 4))
    l1new_aa += einsum(v.aabb.ovvv, (0, 1, 2, 3), x43, (3, 2, 4, 0), (1, 4)) * -1.0
    del x43
    x44 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x44 += einsum(t2.abab, (0, 1, 2, 3), x34, (1, 3, 4, 5), (4, 5, 0, 2))
    x45 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x45 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x46 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x46 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2))
    del x0
    x46 += einsum(x2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x2
    x47 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x47 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x47 += einsum(x44, (0, 1, 2, 3), (2, 0, 1, 3)) * 0.5
    x47 += einsum(x45, (0, 1, 2, 3), (2, 0, 1, 3)) * 2.0
    x47 += einsum(t1.aa, (0, 1), x46, (0, 2, 3, 4), (4, 2, 3, 1))
    del x46
    x47 += einsum(t1.aa, (0, 1), x40, (2, 3), (0, 2, 3, 1)) * 0.5
    l1new_aa += einsum(v.aaaa.ovov, (0, 1, 2, 3), x47, (2, 4, 0, 3), (1, 4)) * -2.0
    del x47
    x48 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x48 += einsum(l1.aa, (0, 1), t1.aa, (1, 2), (0, 2))
    x49 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x49 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    x50 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x50 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4))
    x51 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x51 += einsum(x48, (0, 1), (0, 1))
    x51 += einsum(x49, (0, 1), (0, 1))
    x51 += einsum(x50, (0, 1), (0, 1)) * 2.0
    x52 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x52 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x52 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    l1new_aa += einsum(x51, (0, 1), x52, (2, 1, 0, 3), (3, 2)) * -1.0
    del x51, x52
    x53 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x53 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 0), (1, 2, 3, 4))
    x53 += einsum(x44, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    del x44
    x53 += einsum(x45, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    del x45
    x53 += einsum(t1.aa, (0, 1), x40, (2, 3), (2, 0, 3, 1)) * 0.5
    del x40
    l1new_aa += einsum(v.aaaa.ovov, (0, 1, 2, 3), x53, (4, 2, 0, 1), (3, 4)) * 2.0
    del x53
    x54 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x54 += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), (2, 3))
    x55 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x55 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 0), (2, 3))
    x56 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x56 += einsum(t2.abab, (0, 1, 2, 3), x34, (1, 3, 0, 4), (4, 2))
    x57 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x57 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (0, 1, 4, 3), (4, 2)) * -1.0
    x58 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x58 += einsum(l1.aa, (0, 1), t1.aa, (2, 0), (1, 2))
    l2new_abab += einsum(x58, (0, 1), v.aabb.ovov, (1, 2, 3, 4), (2, 4, 0, 3)) * -1.0
    x59 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x59 += einsum(x58, (0, 1), (0, 1)) * 0.5
    x59 += einsum(x38, (0, 1), (0, 1)) * 0.5
    x59 += einsum(x39, (0, 1), (0, 1))
    l1new_aa += einsum(f.aa.ov, (0, 1), x59, (2, 0), (1, 2)) * -2.0
    x60 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x60 += einsum(t1.aa, (0, 1), x59, (0, 2), (2, 1)) * 2.0
    del x59
    x61 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x61 += einsum(t1.aa, (0, 1), (0, 1)) * -1.0
    x61 += einsum(x54, (0, 1), (0, 1)) * -1.0
    x61 += einsum(x55, (0, 1), (0, 1)) * -2.0
    x61 += einsum(x56, (0, 1), (0, 1))
    x61 += einsum(x57, (0, 1), (0, 1)) * 2.0
    x61 += einsum(x60, (0, 1), (0, 1))
    l1new_aa += einsum(x61, (0, 1), x14, (0, 2, 1, 3), (3, 2))
    del x14, x61
    x62 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x62 += einsum(l1.bb, (0, 1), t1.bb, (1, 2), (0, 2))
    x63 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x63 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4))
    x64 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x64 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    x65 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x65 += einsum(x62, (0, 1), (0, 1)) * 0.5
    x65 += einsum(x63, (0, 1), (0, 1))
    x65 += einsum(x64, (0, 1), (0, 1)) * 0.5
    l1new_aa += einsum(x65, (0, 1), v.aabb.ovvv, (2, 3, 1, 0), (3, 2)) * 2.0
    del x65
    x66 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x66 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 0), (2, 3))
    x67 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x67 += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), (2, 3))
    x68 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x68 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    l1new_bb += einsum(v.bbbb.oovv, (0, 1, 2, 3), x68, (4, 1, 0, 3), (2, 4)) * -2.0
    l2new_abab += einsum(v.aabb.ovoo, (0, 1, 2, 3), x68, (4, 2, 3, 5), (1, 5, 0, 4)) * -2.0
    l2new_abab += einsum(x11, (0, 1, 2, 3), x68, (4, 0, 1, 5), (3, 5, 2, 4)) * -2.0
    x69 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x69 += einsum(t2.bbbb, (0, 1, 2, 3), x68, (0, 1, 4, 3), (4, 2)) * -1.0
    x70 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x70 += einsum(t2.abab, (0, 1, 2, 3), x3, (1, 4, 0, 2), (4, 3))
    x71 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x71 += einsum(l1.bb, (0, 1), t1.bb, (2, 0), (1, 2))
    l2new_abab += einsum(x71, (0, 1), v.aabb.ovov, (2, 3, 1, 4), (3, 4, 2, 0)) * -1.0
    x72 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x72 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    x73 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x73 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    x74 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x74 += einsum(x71, (0, 1), (0, 1))
    x74 += einsum(x72, (0, 1), (0, 1)) * 2.0
    x74 += einsum(x73, (0, 1), (0, 1))
    l1new_aa += einsum(x74, (0, 1), v.aabb.ovoo, (2, 3, 0, 1), (3, 2)) * -1.0
    l1new_bb += einsum(f.bb.ov, (0, 1), x74, (2, 0), (1, 2)) * -1.0
    x75 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x75 += einsum(l1.bb, (0, 1), (1, 0)) * -0.5
    x75 += einsum(t1.bb, (0, 1), (0, 1)) * -0.5
    x75 += einsum(x66, (0, 1), (0, 1)) * -1.0
    x75 += einsum(x67, (0, 1), (0, 1)) * -0.5
    x75 += einsum(x69, (0, 1), (0, 1))
    x75 += einsum(x70, (0, 1), (0, 1)) * 0.5
    x75 += einsum(t1.bb, (0, 1), x74, (0, 2), (2, 1)) * 0.5
    l1new_aa += einsum(x75, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (3, 2)) * -2.0
    del x75
    x76 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x76 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    x76 += einsum(x36, (0, 1, 2, 3), (1, 0, 2, 3))
    l1new_aa += einsum(v.aabb.ovoo, (0, 1, 2, 3), x76, (2, 3, 4, 0), (1, 4))
    del x76
    x77 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x77 += einsum(x58, (0, 1), (0, 1))
    x77 += einsum(x38, (0, 1), (0, 1))
    del x38
    x77 += einsum(x39, (0, 1), (0, 1)) * 2.0
    del x39
    l1new_bb += einsum(x77, (0, 1), v.aabb.ooov, (0, 1, 2, 3), (3, 2)) * -1.0
    x78 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x78 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x78 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    l1new_aa += einsum(x77, (0, 1), x78, (1, 0, 2, 3), (3, 2)) * -1.0
    del x77
    x79 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x79 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x79 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l1new_aa += einsum(l1.aa, (0, 1), x79, (1, 2, 0, 3), (3, 2)) * -1.0
    del x79
    x80 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x80 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 1), (2, 3))
    x81 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x81 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x81 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x82 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x82 += einsum(t1.aa, (0, 1), x81, (0, 2, 1, 3), (2, 3))
    del x81
    x83 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x83 += einsum(f.aa.vv, (0, 1), (0, 1))
    x83 += einsum(x80, (0, 1), (1, 0))
    x83 += einsum(x82, (0, 1), (1, 0)) * -1.0
    l1new_aa += einsum(l1.aa, (0, 1), x83, (0, 2), (2, 1))
    l2new_abab += einsum(x83, (0, 1), l2.abab, (0, 2, 3, 4), (1, 2, 3, 4))
    del x83
    x84 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x84 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3))
    x85 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x85 += einsum(t1.aa, (0, 1), x78, (0, 2, 3, 1), (2, 3))
    del x78
    x86 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x86 += einsum(t1.aa, (0, 1), x16, (2, 1), (0, 2))
    del x16
    x87 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x87 += einsum(f.aa.oo, (0, 1), (0, 1))
    x87 += einsum(x84, (0, 1), (1, 0))
    x87 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x87 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 1, 3), (0, 4)) * 2.0
    x87 += einsum(x85, (0, 1), (0, 1)) * -1.0
    x87 += einsum(x86, (0, 1), (0, 1))
    l1new_aa += einsum(l1.aa, (0, 1), x87, (1, 2), (0, 2)) * -1.0
    del x87
    x88 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x88 += einsum(x13, (0, 1), (0, 1))
    del x13
    x88 += einsum(x15, (0, 1), (0, 1)) * -1.0
    del x15
    l1new_aa += einsum(x58, (0, 1), x88, (1, 2), (2, 0)) * -1.0
    x89 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x89 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    l1new_bb += einsum(x68, (0, 1, 2, 3), x89, (1, 2, 4, 3), (4, 0)) * 2.0
    x90 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x90 += einsum(t1.aa, (0, 1), v.aabb.vvvv, (2, 1, 3, 4), (3, 4, 0, 2))
    l1new_bb += einsum(l2.abab, (0, 1, 2, 3), x90, (4, 1, 2, 0), (4, 3))
    del x90
    x91 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x91 += einsum(t1.bb, (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    l1new_bb += einsum(l2.bbbb, (0, 1, 2, 3), x91, (3, 4, 0, 1), (4, 2)) * 2.0
    del x91
    x92 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x92 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x93 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x93 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x93 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x93 += einsum(x92, (0, 1, 2, 3), (1, 0, 2, 3))
    x93 += einsum(x92, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x94 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x94 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    x95 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x95 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x95 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x96 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x96 += einsum(t1.bb, (0, 1), x95, (0, 2, 1, 3), (2, 3))
    x97 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x97 += einsum(f.bb.ov, (0, 1), (0, 1))
    x97 += einsum(x94, (0, 1), (0, 1))
    x97 += einsum(x96, (0, 1), (0, 1)) * -1.0
    l2new_abab += einsum(x97, (0, 1), x3, (2, 0, 3, 4), (4, 1, 3, 2)) * -1.0
    l2new_abab += einsum(l1.aa, (0, 1), x97, (2, 3), (0, 3, 1, 2))
    x98 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x98 += einsum(t1.aa, (0, 1), v.aabb.vvov, (2, 1, 3, 4), (3, 4, 0, 2))
    x99 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x99 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x99 += einsum(x98, (0, 1, 2, 3), (0, 1, 2, 3))
    del x98
    l2new_abab += einsum(l2.aaaa, (0, 1, 2, 3), x99, (4, 5, 3, 1), (0, 5, 2, 4)) * 2.0
    x100 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x100 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x100 += einsum(t1.aa, (0, 1), v.aabb.vvoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x100 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvov, (4, 2, 5, 3), (1, 5, 0, 4))
    x100 += einsum(t2.abab, (0, 1, 2, 3), x93, (4, 5, 1, 3), (5, 4, 0, 2)) * -1.0
    del x93
    x100 += einsum(t2.aaaa, (0, 1, 2, 3), x12, (4, 5, 1, 3), (5, 4, 0, 2)) * 2.0
    del x12
    x100 += einsum(t2.abab, (0, 1, 2, 3), x10, (4, 3, 0, 5), (1, 4, 5, 2)) * -1.0
    x100 += einsum(x97, (0, 1), t2.abab, (2, 3, 4, 1), (3, 0, 2, 4))
    x100 += einsum(t1.bb, (0, 1), x99, (2, 1, 3, 4), (0, 2, 3, 4))
    x100 += einsum(t1.aa, (0, 1), x23, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x23
    l1new_bb += einsum(l2.abab, (0, 1, 2, 3), x100, (3, 4, 2, 0), (1, 4)) * -1.0
    del x100
    x101 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x101 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x101 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x101 += einsum(x92, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x101 += einsum(x92, (0, 1, 2, 3), (0, 2, 1, 3))
    x102 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x102 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x102 += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3))
    l2new_abab += einsum(x102, (0, 1, 2, 3), x34, (0, 4, 5, 2), (3, 4, 5, 1))
    l2new_abab += einsum(l1.bb, (0, 1), x102, (1, 2, 3, 4), (4, 0, 3, 2)) * -1.0
    x103 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x103 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x103 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x103 += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3))
    del x89
    x104 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x104 += einsum(t1.bb, (0, 1), x92, (2, 3, 4, 1), (0, 2, 3, 4))
    x105 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x105 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x106 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x106 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x106 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 5, 2), (4, 0, 5, 1)) * -1.0
    x106 += einsum(x104, (0, 1, 2, 3), (3, 1, 2, 0))
    x106 += einsum(x105, (0, 1, 2, 3), (2, 1, 3, 0))
    x106 += einsum(x105, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x107 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x107 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x107 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (4, 2, 5, 3), (4, 0, 1, 5)) * -1.0
    x107 += einsum(t2.bbbb, (0, 1, 2, 3), x101, (4, 1, 5, 3), (5, 0, 4, 2)) * -2.0
    del x101
    x107 += einsum(t2.abab, (0, 1, 2, 3), x102, (4, 5, 0, 2), (5, 1, 4, 3))
    del x102
    x107 += einsum(x97, (0, 1), t2.bbbb, (2, 3, 4, 1), (0, 2, 3, 4))
    x107 += einsum(t1.bb, (0, 1), x103, (2, 3, 1, 4), (3, 2, 0, 4))
    del x103
    x107 += einsum(t1.bb, (0, 1), x106, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    del x106
    l1new_bb += einsum(l2.bbbb, (0, 1, 2, 3), x107, (4, 2, 3, 1), (0, 4)) * 2.0
    del x107
    x108 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x108 += einsum(l2.abab, (0, 1, 2, 3), (3, 1, 2, 0)) * 0.5
    x108 += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 0, 4, 5))
    x108 += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 2, 5, 0), (3, 1, 4, 5))
    x108 += einsum(t1.aa, (0, 1), x34, (2, 3, 0, 4), (2, 3, 4, 1)) * -0.5
    l1new_bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), x108, (4, 3, 0, 1), (2, 4)) * 2.0
    del x108
    x109 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x109 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x109 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x110 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x110 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (2, 4, 0, 5))
    x110 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), (3, 4, 1, 5)) * 0.25
    l1new_bb += einsum(x109, (0, 1, 2, 3), x110, (4, 0, 2, 1), (3, 4)) * 4.0
    del x110
    x111 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x111 += einsum(x72, (0, 1), (0, 1))
    del x72
    x111 += einsum(x73, (0, 1), (0, 1)) * 0.5
    del x73
    x112 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x112 += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), (1, 3, 2, 4))
    x112 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    x112 += einsum(t2.abab, (0, 1, 2, 3), x68, (4, 1, 5, 3), (4, 5, 0, 2)) * -2.0
    x112 += einsum(t2.aaaa, (0, 1, 2, 3), x3, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    x112 += einsum(t2.abab, (0, 1, 2, 3), x34, (4, 3, 0, 5), (4, 1, 5, 2)) * -1.0
    x112 += einsum(t1.aa, (0, 1), x37, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x37
    x112 += einsum(t1.aa, (0, 1), x111, (2, 3), (2, 3, 0, 1)) * 2.0
    l1new_bb += einsum(v.aabb.ovov, (0, 1, 2, 3), x112, (4, 2, 0, 1), (3, 4)) * -1.0
    del x112
    x113 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x113 += einsum(l2.bbbb, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x113 += einsum(t1.bb, (0, 1), x68, (2, 0, 3, 4), (2, 3, 4, 1))
    l1new_bb += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x113, (4, 0, 3, 1), (2, 4)) * -2.0
    del x113
    x114 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x114 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), (3, 4, 0, 5))
    x114 += einsum(t1.aa, (0, 1), x3, (2, 3, 0, 4), (2, 3, 1, 4))
    l1new_bb += einsum(v.aabb.vvov, (0, 1, 2, 3), x114, (4, 2, 1, 0), (3, 4)) * -1.0
    del x114
    x115 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x115 += einsum(t2.bbbb, (0, 1, 2, 3), x68, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x116 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x116 += einsum(t1.bb, (0, 1), x68, (2, 3, 4, 1), (2, 3, 0, 4))
    l2new_bbbb += einsum(v.bbbb.ovov, (0, 1, 2, 3), x116, (4, 5, 0, 2), (3, 1, 5, 4)) * 2.0
    x117 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x117 += einsum(t2.abab, (0, 1, 2, 3), x3, (4, 5, 0, 2), (4, 5, 1, 3))
    x118 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x118 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 0), (1, 2, 3, 4))
    x118 += einsum(x68, (0, 1, 2, 3), (1, 0, 2, 3))
    x118 += einsum(x115, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x118 += einsum(t1.bb, (0, 1), x116, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    x118 += einsum(x117, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x118 += einsum(t1.bb, (0, 1), x111, (2, 3), (2, 0, 3, 1))
    l1new_bb += einsum(v.bbbb.ovov, (0, 1, 2, 3), x118, (4, 0, 2, 1), (3, 4)) * -2.0
    del x118
    x119 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x119 += einsum(x62, (0, 1), (0, 1))
    del x62
    x119 += einsum(x63, (0, 1), (0, 1)) * 2.0
    del x63
    x119 += einsum(x64, (0, 1), (0, 1))
    del x64
    l1new_bb += einsum(x119, (0, 1), x109, (2, 1, 0, 3), (3, 2)) * -1.0
    del x119
    x120 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x120 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), (2, 3, 4, 5))
    x121 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x121 += einsum(x115, (0, 1, 2, 3), (0, 2, 1, 3)) * 4.0
    del x115
    x121 += einsum(t1.bb, (0, 1), x120, (2, 0, 3, 4), (2, 4, 3, 1)) * -2.0
    x121 += einsum(x117, (0, 1, 2, 3), (0, 2, 1, 3))
    del x117
    x121 += einsum(t1.bb, (0, 1), x111, (2, 3), (2, 0, 3, 1)) * 2.0
    del x111
    l1new_bb += einsum(v.bbbb.ovov, (0, 1, 2, 3), x121, (4, 0, 2, 3), (1, 4))
    del x121
    x122 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x122 += einsum(t1.bb, (0, 1), (0, 1)) * -1.0
    x122 += einsum(x66, (0, 1), (0, 1)) * -2.0
    del x66
    x122 += einsum(x67, (0, 1), (0, 1)) * -1.0
    del x67
    x122 += einsum(x69, (0, 1), (0, 1)) * 2.0
    del x69
    x122 += einsum(x70, (0, 1), (0, 1))
    del x70
    x122 += einsum(t1.bb, (0, 1), x74, (0, 2), (2, 1))
    l1new_bb += einsum(x122, (0, 1), x95, (0, 2, 1, 3), (3, 2))
    del x95, x122
    x123 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x123 += einsum(x48, (0, 1), (0, 1)) * 0.5
    del x48
    x123 += einsum(x49, (0, 1), (0, 1)) * 0.5
    del x49
    x123 += einsum(x50, (0, 1), (0, 1))
    del x50
    l1new_bb += einsum(x123, (0, 1), v.aabb.vvov, (0, 1, 2, 3), (3, 2)) * 2.0
    del x123
    x124 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x124 += einsum(l1.aa, (0, 1), (1, 0)) * -1.0
    x124 += einsum(t1.aa, (0, 1), (0, 1)) * -1.0
    x124 += einsum(x54, (0, 1), (0, 1)) * -1.0
    del x54
    x124 += einsum(x55, (0, 1), (0, 1)) * -2.0
    del x55
    x124 += einsum(x56, (0, 1), (0, 1))
    del x56
    x124 += einsum(x57, (0, 1), (0, 1)) * 2.0
    del x57
    x124 += einsum(x60, (0, 1), (0, 1))
    del x60
    l1new_bb += einsum(x124, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (3, 2)) * -1.0
    del x124
    x125 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x125 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    del x35
    x125 += einsum(x36, (0, 1, 2, 3), (0, 1, 3, 2))
    del x36
    l1new_bb += einsum(v.aabb.ooov, (0, 1, 2, 3), x125, (4, 2, 1, 0), (3, 4))
    del x125
    x126 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x126 += einsum(x120, (0, 1, 2, 3), (1, 0, 3, 2))
    del x120
    x126 += einsum(x116, (0, 1, 2, 3), (2, 1, 0, 3))
    del x116
    l1new_bb += einsum(v.bbbb.ooov, (0, 1, 2, 3), x126, (0, 4, 1, 2), (3, 4)) * 2.0
    del x126
    x127 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x127 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x127 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    l1new_bb += einsum(x74, (0, 1), x127, (0, 2, 1, 3), (3, 2)) * -1.0
    del x74
    l2new_abab += einsum(x127, (0, 1, 2, 3), x3, (0, 2, 4, 5), (5, 3, 4, 1)) * -1.0
    x128 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x128 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x128 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l1new_bb += einsum(l1.bb, (0, 1), x128, (1, 2, 0, 3), (3, 2)) * -1.0
    del x128
    x129 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x129 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 1, 2, 3), (2, 3))
    x130 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x130 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x130 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x131 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x131 += einsum(t1.bb, (0, 1), x130, (0, 2, 1, 3), (2, 3))
    x132 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x132 += einsum(f.bb.vv, (0, 1), (0, 1))
    x132 += einsum(x129, (0, 1), (1, 0))
    x132 += einsum(x131, (0, 1), (1, 0)) * -1.0
    del x131
    l1new_bb += einsum(l1.bb, (0, 1), x132, (0, 2), (2, 1))
    l2new_abab += einsum(x132, (0, 1), l2.abab, (2, 0, 3, 4), (2, 1, 3, 4))
    del x132
    x133 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x133 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3))
    x134 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x134 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x134 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x135 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x135 += einsum(t1.bb, (0, 1), x134, (2, 3, 0, 1), (2, 3))
    del x134
    x136 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x136 += einsum(t1.bb, (0, 1), x97, (2, 1), (0, 2))
    del x97
    x137 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x137 += einsum(f.bb.oo, (0, 1), (0, 1))
    x137 += einsum(x133, (0, 1), (1, 0))
    x137 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 1, 3), (0, 4)) * 2.0
    x137 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x137 += einsum(x135, (0, 1), (1, 0)) * -1.0
    x137 += einsum(x136, (0, 1), (0, 1))
    l1new_bb += einsum(l1.bb, (0, 1), x137, (1, 2), (0, 2)) * -1.0
    del x137
    x138 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x138 += einsum(x94, (0, 1), (0, 1))
    del x94
    x138 += einsum(x96, (0, 1), (0, 1)) * -1.0
    del x96
    l1new_bb += einsum(x138, (0, 1), x71, (2, 0), (1, 2)) * -1.0
    x139 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x139 += einsum(f.aa.ov, (0, 1), t1.aa, (2, 1), (0, 2))
    x140 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x140 += einsum(x139, (0, 1), l2.aaaa, (2, 3, 4, 1), (0, 4, 2, 3))
    del x139
    x141 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x141 += einsum(l2.aaaa, (0, 1, 2, 3), x28, (3, 4, 2, 5), (4, 5, 0, 1)) * -1.0
    del x28
    x142 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x142 += einsum(t1.aa, (0, 1), x88, (2, 1), (0, 2))
    x143 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x143 += einsum(x84, (0, 1), (1, 0))
    x143 += einsum(x85, (0, 1), (0, 1)) * -1.0
    x143 += einsum(x142, (0, 1), (0, 1))
    del x142
    x144 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x144 += einsum(x143, (0, 1), l2.aaaa, (2, 3, 4, 0), (1, 4, 2, 3)) * -2.0
    del x143
    x145 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x145 += einsum(x140, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x140
    x145 += einsum(x141, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x141
    x145 += einsum(x144, (0, 1, 2, 3), (1, 0, 3, 2))
    del x144
    l2new_aaaa += einsum(x145, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new_aaaa += einsum(x145, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x145
    x146 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x146 += einsum(f.aa.vv, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    x147 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x147 += einsum(f.aa.ov, (0, 1), x1, (2, 3, 0, 4), (2, 3, 1, 4))
    x148 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x148 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x1, (4, 5, 0, 3), (5, 4, 1, 2))
    x149 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x149 += einsum(x80, (0, 1), (1, 0))
    del x80
    x149 += einsum(x82, (0, 1), (0, 1)) * -1.0
    del x82
    x150 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x150 += einsum(x149, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2)) * -2.0
    del x149
    x151 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x151 += einsum(x88, (0, 1), x1, (2, 3, 0, 4), (2, 3, 4, 1)) * 2.0
    x152 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x152 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x152 += einsum(x146, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x146
    x152 += einsum(x147, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x147
    x152 += einsum(x148, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x148
    x152 += einsum(x150, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x150
    x152 += einsum(x151, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x151
    l2new_aaaa += einsum(x152, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_aaaa += einsum(x152, (0, 1, 2, 3), (2, 3, 0, 1))
    del x152
    x153 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x153 += einsum(l1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 0), (1, 2, 3, 4))
    x154 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x154 += einsum(x58, (0, 1), v.aaaa.ovov, (2, 3, 1, 4), (0, 2, 3, 4))
    del x58
    x155 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x155 += einsum(v.aabb.ovoo, (0, 1, 2, 3), x3, (2, 3, 4, 5), (4, 0, 5, 1))
    x156 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x156 += einsum(x11, (0, 1, 2, 3), x3, (0, 1, 4, 5), (4, 2, 5, 3))
    del x11
    x157 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x157 += einsum(t1.aa, (0, 1), x32, (2, 3, 1, 4), (0, 2, 3, 4))
    x158 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x158 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x158 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x158 += einsum(x157, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x157
    x159 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x159 += einsum(l2.aaaa, (0, 1, 2, 3), x158, (3, 4, 5, 1), (4, 2, 5, 0)) * 2.0
    del x158
    x160 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x160 += einsum(t1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (0, 4, 2, 3))
    x161 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x161 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x161 += einsum(x160, (0, 1, 2, 3), (0, 1, 2, 3))
    del x160
    l2new_abab += einsum(l2.bbbb, (0, 1, 2, 3), x161, (3, 1, 4, 5), (5, 0, 4, 2)) * 2.0
    x162 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x162 += einsum(l2.abab, (0, 1, 2, 3), x161, (3, 1, 4, 5), (4, 2, 5, 0))
    del x161
    x163 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x163 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3))
    x163 += einsum(x8, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l2new_abab += einsum(x163, (0, 1, 2, 3), x34, (4, 5, 0, 2), (3, 5, 1, 4)) * -1.0
    x164 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x164 += einsum(x1, (0, 1, 2, 3), x163, (0, 4, 2, 5), (4, 1, 5, 3)) * 2.0
    del x163
    x165 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x165 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x165 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l2new_abab += einsum(x165, (0, 1, 2, 3), x34, (4, 5, 0, 1), (3, 5, 2, 4)) * -1.0
    x166 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x166 += einsum(x1, (0, 1, 2, 3), x165, (0, 2, 4, 5), (4, 1, 5, 3)) * 2.0
    del x1, x165
    x167 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x167 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x167 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3))
    del x8
    x168 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x168 += einsum(l1.aa, (0, 1), x167, (1, 2, 3, 4), (2, 3, 4, 0))
    del x167
    x169 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x169 += einsum(f.aa.ov, (0, 1), l1.aa, (2, 3), (0, 3, 1, 2))
    x169 += einsum(x153, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x153
    x169 += einsum(x154, (0, 1, 2, 3), (0, 1, 2, 3))
    del x154
    x169 += einsum(x155, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x155
    x169 += einsum(x156, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x156
    x169 += einsum(x159, (0, 1, 2, 3), (1, 0, 3, 2))
    del x159
    x169 += einsum(x162, (0, 1, 2, 3), (1, 0, 3, 2))
    del x162
    x169 += einsum(x164, (0, 1, 2, 3), (1, 0, 3, 2))
    del x164
    x169 += einsum(x166, (0, 1, 2, 3), (1, 0, 3, 2))
    del x166
    x169 += einsum(x168, (0, 1, 2, 3), (0, 1, 3, 2))
    del x168
    x169 += einsum(l1.aa, (0, 1), x88, (2, 3), (1, 2, 0, 3))
    del x88
    l2new_aaaa += einsum(x169, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_aaaa += einsum(x169, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_aaaa += einsum(x169, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new_aaaa += einsum(x169, (0, 1, 2, 3), (3, 2, 1, 0))
    del x169
    x170 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x170 += einsum(f.aa.oo, (0, 1), l2.aaaa, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_aaaa += einsum(x170, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    l2new_aaaa += einsum(x170, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    del x170
    x171 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x171 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x171 += einsum(x27, (0, 1, 2, 3), (0, 2, 3, 1))
    del x27
    l2new_aaaa += einsum(l2.aaaa, (0, 1, 2, 3), x171, (3, 4, 5, 2), (0, 1, 5, 4)) * 2.0
    del x171
    x172 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x172 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x172 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x172 += einsum(t1.bb, (0, 1), x130, (2, 3, 1, 4), (0, 2, 4, 3)) * -1.0
    del x130
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x172, (3, 4, 1, 5), (0, 5, 2, 4)) * -1.0
    del x172
    x173 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x173 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x173 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x173 += einsum(t1.aa, (0, 1), x32, (2, 1, 3, 4), (0, 2, 4, 3)) * -1.0
    del x32
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x173, (2, 4, 0, 5), (5, 1, 4, 3)) * -1.0
    del x173
    x174 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x174 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x174 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x174, (3, 4, 0, 5), (5, 1, 2, 4)) * -1.0
    del x174
    x175 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x175 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x175 += einsum(x17, (0, 1, 2, 3), (1, 0, 2, 3))
    del x17
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x175, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    del x175
    x176 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x176 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x176 += einsum(x19, (0, 1, 2, 3), (1, 0, 2, 3))
    del x19
    x176 += einsum(t1.bb, (0, 1), x10, (2, 1, 3, 4), (0, 2, 4, 3))
    del x10
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x176, (3, 4, 2, 5), (0, 1, 5, 4))
    del x176
    x177 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x177 += einsum(x92, (0, 1, 2, 3), (0, 1, 2, 3))
    x177 += einsum(x92, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l2new_abab += einsum(x177, (0, 1, 2, 3), x3, (0, 2, 4, 5), (5, 3, 4, 1)) * -1.0
    del x3, x177
    x178 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x178 += einsum(f.aa.oo, (0, 1), (0, 1))
    x178 += einsum(x84, (0, 1), (1, 0))
    del x84
    x178 += einsum(x85, (0, 1), (0, 1)) * -1.0
    del x85
    x178 += einsum(x86, (0, 1), (0, 1))
    del x86
    l2new_abab += einsum(x178, (0, 1), l2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    del x178
    x179 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x179 += einsum(f.bb.oo, (0, 1), (0, 1))
    x179 += einsum(x133, (0, 1), (1, 0))
    x179 += einsum(x135, (0, 1), (1, 0)) * -1.0
    x179 += einsum(x136, (0, 1), (0, 1))
    del x136
    l2new_abab += einsum(x179, (0, 1), l2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x179
    x180 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x180 += einsum(f.bb.ov, (0, 1), t1.bb, (2, 1), (0, 2))
    x181 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x181 += einsum(x180, (0, 1), l2.bbbb, (2, 3, 4, 1), (0, 4, 2, 3))
    del x180
    x182 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x182 += einsum(l2.bbbb, (0, 1, 2, 3), x105, (2, 3, 4, 5), (4, 5, 0, 1))
    del x105
    x183 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x183 += einsum(t1.bb, (0, 1), x138, (2, 1), (2, 0))
    x184 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x184 += einsum(x133, (0, 1), (1, 0))
    del x133
    x184 += einsum(x135, (0, 1), (1, 0)) * -1.0
    del x135
    x184 += einsum(x183, (0, 1), (1, 0))
    del x183
    x185 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x185 += einsum(x184, (0, 1), l2.bbbb, (2, 3, 4, 0), (1, 4, 2, 3)) * -2.0
    del x184
    x186 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x186 += einsum(x181, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x181
    x186 += einsum(x182, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x182
    x186 += einsum(x185, (0, 1, 2, 3), (1, 0, 3, 2))
    del x185
    l2new_bbbb += einsum(x186, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new_bbbb += einsum(x186, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x186
    x187 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x187 += einsum(f.bb.vv, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    x188 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x188 += einsum(f.bb.ov, (0, 1), x68, (2, 3, 0, 4), (2, 3, 1, 4))
    x189 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x189 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x68, (4, 5, 0, 3), (5, 4, 1, 2))
    x190 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x190 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x190 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x191 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x191 += einsum(t1.bb, (0, 1), x190, (0, 1, 2, 3), (2, 3))
    del x190
    x192 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x192 += einsum(x129, (0, 1), (1, 0))
    del x129
    x192 += einsum(x191, (0, 1), (1, 0)) * -1.0
    del x191
    x193 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x193 += einsum(x192, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2)) * -2.0
    del x192
    x194 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x194 += einsum(x138, (0, 1), x68, (2, 3, 0, 4), (2, 3, 1, 4)) * 2.0
    x195 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x195 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x195 += einsum(x187, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x187
    x195 += einsum(x188, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x188
    x195 += einsum(x189, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x189
    x195 += einsum(x193, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x193
    x195 += einsum(x194, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x194
    l2new_bbbb += einsum(x195, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_bbbb += einsum(x195, (0, 1, 2, 3), (2, 3, 0, 1))
    del x195
    x196 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x196 += einsum(l1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 0), (1, 2, 3, 4))
    x197 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x197 += einsum(x71, (0, 1), v.bbbb.ovov, (2, 3, 1, 4), (0, 2, 3, 4))
    del x71
    x198 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x198 += einsum(v.aabb.ooov, (0, 1, 2, 3), x34, (4, 5, 0, 1), (4, 2, 5, 3))
    x199 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x199 += einsum(x34, (0, 1, 2, 3), x7, (4, 5, 2, 3), (0, 4, 1, 5))
    del x7, x34
    x200 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x200 += einsum(t1.bb, (0, 1), x109, (2, 1, 3, 4), (0, 2, 3, 4))
    del x109
    x201 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x201 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x201 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x201 += einsum(x200, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x200
    x202 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x202 += einsum(l2.bbbb, (0, 1, 2, 3), x201, (3, 4, 5, 1), (4, 2, 5, 0)) * 2.0
    del x201
    x203 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x203 += einsum(l2.abab, (0, 1, 2, 3), x99, (4, 5, 2, 0), (3, 4, 1, 5))
    del x99
    x204 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x204 += einsum(x92, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x204 += einsum(x92, (0, 1, 2, 3), (0, 2, 1, 3))
    x205 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x205 += einsum(x204, (0, 1, 2, 3), x68, (0, 4, 1, 5), (2, 4, 3, 5)) * 2.0
    del x204
    x206 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x206 += einsum(x127, (0, 1, 2, 3), x68, (0, 4, 2, 5), (1, 4, 3, 5)) * 2.0
    del x68, x127
    x207 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x207 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x207 += einsum(x92, (0, 1, 2, 3), (0, 1, 2, 3))
    del x92
    x208 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x208 += einsum(l1.bb, (0, 1), x207, (1, 2, 3, 4), (2, 3, 4, 0))
    del x207
    x209 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x209 += einsum(f.bb.ov, (0, 1), l1.bb, (2, 3), (0, 3, 1, 2))
    x209 += einsum(x196, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x196
    x209 += einsum(x197, (0, 1, 2, 3), (0, 1, 2, 3))
    del x197
    x209 += einsum(x198, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x198
    x209 += einsum(x199, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x199
    x209 += einsum(x202, (0, 1, 2, 3), (1, 0, 3, 2))
    del x202
    x209 += einsum(x203, (0, 1, 2, 3), (0, 1, 2, 3))
    del x203
    x209 += einsum(x205, (0, 1, 2, 3), (1, 0, 3, 2))
    del x205
    x209 += einsum(x206, (0, 1, 2, 3), (1, 0, 3, 2))
    del x206
    x209 += einsum(x208, (0, 1, 2, 3), (0, 1, 3, 2))
    del x208
    x209 += einsum(l1.bb, (0, 1), x138, (2, 3), (1, 2, 0, 3))
    del x138
    l2new_bbbb += einsum(x209, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_bbbb += einsum(x209, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_bbbb += einsum(x209, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new_bbbb += einsum(x209, (0, 1, 2, 3), (3, 2, 1, 0))
    del x209
    x210 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x210 += einsum(f.bb.oo, (0, 1), l2.bbbb, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_bbbb += einsum(x210, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    l2new_bbbb += einsum(x210, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    del x210
    x211 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x211 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x211 += einsum(x104, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    del x104
    l2new_bbbb += einsum(l2.bbbb, (0, 1, 2, 3), x211, (3, 4, 2, 5), (0, 1, 4, 5)) * -2.0
    del x211

    l1new.aa = l1new_aa
    l1new.bb = l1new_bb
    l2new.aaaa = l2new_aaaa
    l2new.abab = l2new_abab
    l2new.bbbb = l2new_bbbb

    return {"l1new": l1new, "l2new": l2new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
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
    rdm1_f_aa_ov += einsum(t1.aa, (0, 1), (0, 1))
    rdm1_f_aa_ov += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), (2, 3))
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
    rdm1_f_aa_vv += einsum(l1.aa, (0, 1), t1.aa, (1, 2), (0, 2))
    rdm1_f_aa_vv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    rdm1_f_aa_vv += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4)) * 2.0
    rdm1_f_bb_vv = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    rdm1_f_bb_vv += einsum(l1.bb, (0, 1), t1.bb, (1, 2), (0, 2))
    rdm1_f_bb_vv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    rdm1_f_bb_vv += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4)) * 2.0
    x0 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x0 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    rdm1_f_aa_oo += einsum(x0, (0, 1), (1, 0)) * -2.0
    x1 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x1 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    rdm1_f_aa_oo += einsum(x1, (0, 1), (1, 0)) * -1.0
    x2 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x2 += einsum(l1.aa, (0, 1), t1.aa, (2, 0), (1, 2))
    rdm1_f_aa_oo += einsum(x2, (0, 1), (1, 0)) * -1.0
    x3 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x3 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    rdm1_f_bb_oo += einsum(x3, (0, 1), (1, 0)) * -2.0
    x4 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x4 += einsum(l1.bb, (0, 1), t1.bb, (2, 0), (1, 2))
    rdm1_f_bb_oo += einsum(x4, (0, 1), (1, 0)) * -1.0
    x5 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x5 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    rdm1_f_bb_oo += einsum(x5, (0, 1), (1, 0)) * -1.0
    x6 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x6 += einsum(t1.aa, (0, 1), l2.abab, (1, 2, 3, 4), (4, 2, 3, 0))
    rdm1_f_aa_ov += einsum(t2.abab, (0, 1, 2, 3), x6, (1, 3, 0, 4), (4, 2)) * -1.0
    del x6
    x7 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x7 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm1_f_aa_ov += einsum(t2.aaaa, (0, 1, 2, 3), x7, (1, 0, 4, 3), (4, 2)) * -2.0
    del x7
    x8 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x8 += einsum(x2, (0, 1), (0, 1)) * 0.5
    del x2
    x8 += einsum(x1, (0, 1), (0, 1)) * 0.5
    del x1
    x8 += einsum(x0, (0, 1), (0, 1))
    del x0
    rdm1_f_aa_ov += einsum(t1.aa, (0, 1), x8, (0, 2), (2, 1)) * -2.0
    del x8
    x9 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x9 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm1_f_bb_ov += einsum(t2.bbbb, (0, 1, 2, 3), x9, (0, 1, 4, 3), (4, 2)) * 2.0
    del x9
    x10 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x10 += einsum(t1.bb, (0, 1), l2.abab, (2, 1, 3, 4), (4, 0, 3, 2))
    rdm1_f_bb_ov += einsum(t2.abab, (0, 1, 2, 3), x10, (1, 4, 0, 2), (4, 3)) * -1.0
    del x10
    x11 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x11 += einsum(x4, (0, 1), (0, 1))
    del x4
    x11 += einsum(x3, (0, 1), (0, 1)) * 2.0
    del x3
    x11 += einsum(x5, (0, 1), (0, 1))
    del x5
    rdm1_f_bb_ov += einsum(t1.bb, (0, 1), x11, (0, 2), (2, 1)) * -1.0
    del x11

    rdm1_f_aa = np.block([[rdm1_f_aa_oo, rdm1_f_aa_ov], [rdm1_f_aa_vo, rdm1_f_aa_vv]])
    rdm1_f_bb = np.block([[rdm1_f_bb_oo, rdm1_f_bb_ov], [rdm1_f_bb_vo, rdm1_f_bb_vv]])

    rdm1_f.aa = rdm1_f_aa
    rdm1_f.bb = rdm1_f_bb

    return rdm1_f

def make_rdm2_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
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
    rdm2_f_abab_oovv = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_oovv = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_oovv += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_bbbb_oovv += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_bbbb_oovv += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
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
    rdm2_f_abab_vvov = np.zeros((nvir[0], nvir[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_vvov += einsum(t1.bb, (0, 1), l2.abab, (2, 3, 4, 0), (2, 3, 4, 1))
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
    x3 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    x4 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    x5 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x5 += einsum(x3, (0, 1), (0, 1))
    x5 += einsum(x4, (0, 1), (0, 1)) * 2.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x5, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x5, (2, 3), (0, 3, 2, 1))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x5, (2, 3), (3, 0, 1, 2))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x5, (2, 3), (3, 0, 2, 1)) * -1.0
    x6 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x6 += einsum(l1.aa, (0, 1), t1.aa, (2, 0), (1, 2))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x6, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x6, (2, 3), (3, 0, 1, 2))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x6, (2, 3), (0, 3, 2, 1))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x6, (2, 3), (3, 0, 2, 1)) * -1.0
    x7 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x7 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), (3, 5, 2, 4))
    rdm2_f_abab_oooo = np.zeros((nocc[0], nocc[1], nocc[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_oooo += einsum(x7, (0, 1, 2, 3), (3, 1, 2, 0))
    x8 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x8 += einsum(t1.aa, (0, 1), l2.abab, (1, 2, 3, 4), (4, 2, 3, 0))
    rdm2_f_abab_ovoo += einsum(x8, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    rdm2_f_abab_ovov = np.zeros((nocc[0], nvir[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_ovov += einsum(t1.bb, (0, 1), x8, (0, 2, 3, 4), (4, 2, 3, 1)) * -1.0
    rdm2_f_abab_ovvv += einsum(t2.abab, (0, 1, 2, 3), x8, (1, 4, 0, 5), (5, 4, 2, 3)) * -1.0
    x9 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x9 += einsum(t1.bb, (0, 1), x8, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_abab_oooo += einsum(x9, (0, 1, 2, 3), (3, 1, 2, 0))
    x10 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x10 += einsum(delta.aa.oo, (0, 1), (0, 1)) * -1.0
    x10 += einsum(x6, (0, 1), (0, 1))
    x10 += einsum(x3, (0, 1), (0, 1))
    x10 += einsum(x4, (0, 1), (0, 1)) * 2.0
    rdm2_f_abab_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (3, 0, 2, 1)) * -1.0
    del x10
    x11 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x11 += einsum(l1.bb, (0, 1), t1.bb, (2, 0), (1, 2))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x11, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x11, (2, 3), (3, 0, 1, 2))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x11, (2, 3), (0, 3, 2, 1))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x11, (2, 3), (3, 0, 2, 1)) * -1.0
    x12 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x12 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    x13 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x13 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    x14 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x14 += einsum(x11, (0, 1), (0, 1)) * 0.5
    x14 += einsum(x12, (0, 1), (0, 1))
    x14 += einsum(x13, (0, 1), (0, 1)) * 0.5
    rdm2_f_abab_oooo += einsum(delta.aa.oo, (0, 1), x14, (2, 3), (0, 3, 1, 2)) * -2.0
    del x14
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
    x18 += einsum(x12, (0, 1), (0, 1))
    x18 += einsum(x13, (0, 1), (0, 1)) * 0.5
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x18, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x18, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x18, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x18, (2, 3), (3, 0, 2, 1)) * -2.0
    x19 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x19 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_aaaa_ooov = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_ooov += einsum(x19, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_aaaa_oovo = np.zeros((nocc[0], nocc[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_oovo += einsum(x19, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x20 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x20 += einsum(t2.abab, (0, 1, 2, 3), x8, (1, 3, 4, 5), (4, 5, 0, 2))
    x21 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x21 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x22 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x22 += einsum(x6, (0, 1), (0, 1)) * 0.5
    x22 += einsum(x3, (0, 1), (0, 1)) * 0.5
    del x3
    x22 += einsum(x4, (0, 1), (0, 1))
    del x4
    rdm2_f_abab_ooov = np.zeros((nocc[0], nocc[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_ooov += einsum(t1.bb, (0, 1), x22, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_abab_oovv += einsum(x22, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -2.0
    x23 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x23 += einsum(delta.aa.oo, (0, 1), t1.aa, (2, 3), (0, 1, 2, 3))
    x23 += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x23 += einsum(x21, (0, 1, 2, 3), (1, 0, 2, 3)) * -4.0
    x23 += einsum(t1.aa, (0, 1), x22, (2, 3), (0, 2, 3, 1)) * 2.0
    rdm2_f_aaaa_ooov += einsum(x23, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_ooov += einsum(x23, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_aaaa_oovo += einsum(x23, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_oovo += einsum(x23, (0, 1, 2, 3), (2, 0, 3, 1))
    del x23
    x24 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x24 += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), (2, 3))
    x25 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x25 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 0), (2, 3))
    x26 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x26 += einsum(t2.abab, (0, 1, 2, 3), x8, (1, 3, 0, 4), (4, 2))
    x27 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x27 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (0, 1, 4, 3), (4, 2)) * -1.0
    x28 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x28 += einsum(t1.aa, (0, 1), x22, (0, 2), (2, 1)) * 2.0
    x29 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x29 += einsum(x24, (0, 1), (0, 1)) * -1.0
    x29 += einsum(x25, (0, 1), (0, 1)) * -2.0
    x29 += einsum(x26, (0, 1), (0, 1))
    x29 += einsum(x27, (0, 1), (0, 1)) * 2.0
    x29 += einsum(x28, (0, 1), (0, 1))
    del x28
    rdm2_f_aaaa_ooov += einsum(delta.aa.oo, (0, 1), x29, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_aaaa_ooov += einsum(delta.aa.oo, (0, 1), x29, (2, 3), (2, 0, 1, 3))
    rdm2_f_aaaa_oovo += einsum(delta.aa.oo, (0, 1), x29, (2, 3), (0, 2, 3, 1))
    rdm2_f_aaaa_oovo += einsum(delta.aa.oo, (0, 1), x29, (2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_abab_oovo = np.zeros((nocc[0], nocc[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_oovo += einsum(delta.bb.oo, (0, 1), x29, (2, 3), (2, 0, 3, 1)) * -1.0
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
    x32 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x32 += einsum(t1.bb, (0, 1), l2.abab, (2, 1, 3, 4), (4, 0, 3, 2))
    rdm2_f_abab_vooo += einsum(x32, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    rdm2_f_abab_vovo = np.zeros((nvir[0], nocc[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_vovo += einsum(t1.aa, (0, 1), x32, (2, 3, 0, 4), (4, 3, 1, 2)) * -1.0
    rdm2_f_abab_vovv += einsum(t2.abab, (0, 1, 2, 3), x32, (1, 4, 0, 5), (5, 4, 2, 3)) * -1.0
    x33 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x33 += einsum(t2.abab, (0, 1, 2, 3), x32, (1, 4, 5, 2), (4, 3, 5, 0))
    rdm2_f_abab_ooov += einsum(x33, (0, 1, 2, 3), (3, 0, 2, 1))
    x34 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x34 += einsum(t2.bbbb, (0, 1, 2, 3), x8, (1, 3, 4, 5), (0, 2, 4, 5))
    rdm2_f_abab_ooov += einsum(x34, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    x35 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x35 += einsum(t2.abab, (0, 1, 2, 3), x1, (4, 0, 5, 2), (1, 3, 4, 5)) * -1.0
    rdm2_f_abab_ooov += einsum(x35, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    x36 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x36 += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), (3, 4, 1, 2))
    rdm2_f_abab_ooov += einsum(x36, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    x37 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x37 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    del x7
    x37 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    del x9
    rdm2_f_abab_ooov += einsum(t1.bb, (0, 1), x37, (0, 2, 3, 4), (4, 2, 3, 1))
    rdm2_f_abab_oovo += einsum(t1.aa, (0, 1), x37, (2, 3, 0, 4), (4, 3, 1, 2))
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x37, (1, 4, 0, 5), (5, 4, 2, 3))
    x38 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x38 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 0), (2, 3))
    x39 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x39 += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), (2, 3))
    x40 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x40 += einsum(t2.bbbb, (0, 1, 2, 3), x16, (0, 1, 4, 3), (4, 2)) * -1.0
    x41 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x41 += einsum(t2.abab, (0, 1, 2, 3), x32, (1, 4, 0, 2), (4, 3))
    x42 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x42 += einsum(x11, (0, 1), (0, 1))
    x42 += einsum(x12, (0, 1), (0, 1)) * 2.0
    x42 += einsum(x13, (0, 1), (0, 1))
    rdm2_f_abab_oovv += einsum(x42, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    x43 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x43 += einsum(t1.bb, (0, 1), (0, 1)) * -1.0
    x43 += einsum(x38, (0, 1), (0, 1)) * -2.0
    x43 += einsum(x39, (0, 1), (0, 1)) * -1.0
    x43 += einsum(x40, (0, 1), (0, 1)) * 2.0
    x43 += einsum(x41, (0, 1), (0, 1))
    x43 += einsum(t1.bb, (0, 1), x42, (0, 2), (2, 1))
    rdm2_f_abab_ooov += einsum(delta.aa.oo, (0, 1), x43, (2, 3), (0, 2, 1, 3)) * -1.0
    del x43
    x44 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x44 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_bbbb_ooov = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_ooov += einsum(x44, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_bbbb_oovo = np.zeros((nocc[1], nocc[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_oovo += einsum(x44, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x45 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x45 += einsum(t1.bb, (0, 1), x42, (0, 2), (2, 1)) * 0.5
    x46 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x46 += einsum(x38, (0, 1), (0, 1)) * -1.0
    x46 += einsum(x39, (0, 1), (0, 1)) * -0.5
    x46 += einsum(x40, (0, 1), (0, 1))
    x46 += einsum(x41, (0, 1), (0, 1)) * 0.5
    x46 += einsum(x45, (0, 1), (0, 1))
    rdm2_f_bbbb_ooov += einsum(delta.bb.oo, (0, 1), x46, (2, 3), (0, 2, 1, 3)) * -2.0
    rdm2_f_bbbb_ooov += einsum(delta.bb.oo, (0, 1), x46, (2, 3), (2, 0, 1, 3)) * 2.0
    rdm2_f_bbbb_oovo += einsum(delta.bb.oo, (0, 1), x46, (2, 3), (0, 2, 3, 1)) * 2.0
    rdm2_f_bbbb_oovo += einsum(delta.bb.oo, (0, 1), x46, (2, 3), (2, 0, 3, 1)) * -2.0
    x47 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x47 += einsum(t2.bbbb, (0, 1, 2, 3), x16, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x48 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x48 += einsum(t2.abab, (0, 1, 2, 3), x32, (4, 5, 0, 2), (4, 5, 1, 3))
    x49 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x49 += einsum(delta.bb.oo, (0, 1), t1.bb, (2, 3), (0, 1, 2, 3))
    x49 += einsum(x47, (0, 1, 2, 3), (1, 0, 2, 3)) * -4.0
    x49 += einsum(x48, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x49 += einsum(t1.bb, (0, 1), x42, (2, 3), (0, 2, 3, 1))
    del x42
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
    x52 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x52 += einsum(t2.abab, (0, 1, 2, 3), x16, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    rdm2_f_abab_oovo += einsum(x52, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    x53 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x53 += einsum(t2.abab, (0, 1, 2, 3), x8, (4, 3, 0, 5), (4, 1, 5, 2))
    rdm2_f_abab_oovo += einsum(x53, (0, 1, 2, 3), (2, 1, 3, 0))
    x54 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x54 += einsum(t2.aaaa, (0, 1, 2, 3), x32, (4, 5, 1, 3), (4, 5, 0, 2))
    rdm2_f_abab_oovo += einsum(x54, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    x55 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x55 += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), (1, 3, 2, 4))
    rdm2_f_abab_oovo += einsum(x55, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x56 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x56 += einsum(delta.bb.oo, (0, 1), (0, 1)) * -0.5
    x56 += einsum(x11, (0, 1), (0, 1)) * 0.5
    x56 += einsum(x12, (0, 1), (0, 1))
    del x12
    x56 += einsum(x13, (0, 1), (0, 1)) * 0.5
    del x13
    rdm2_f_abab_oovo += einsum(t1.aa, (0, 1), x56, (2, 3), (0, 3, 1, 2)) * -2.0
    del x56
    x57 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x57 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_aaaa_ovov += einsum(x57, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_aaaa_ovvo += einsum(x57, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_aaaa_voov += einsum(x57, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_aaaa_vovo += einsum(x57, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x58 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x58 += einsum(t2.aaaa, (0, 1, 2, 3), x57, (1, 4, 3, 5), (4, 0, 5, 2))
    x59 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x59 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3))
    del x20
    x59 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x21
    x60 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x60 += einsum(t1.aa, (0, 1), x59, (0, 2, 3, 4), (2, 3, 1, 4))
    del x59
    x61 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x61 += einsum(x58, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x58
    x61 += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3))
    del x60
    x61 += einsum(t1.aa, (0, 1), x29, (2, 3), (0, 2, 1, 3))
    del x29
    rdm2_f_aaaa_oovv += einsum(x61, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x61, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_aaaa_oovv += einsum(x61, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_aaaa_oovv += einsum(x61, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x61
    x62 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x62 += einsum(x6, (0, 1), t2.aaaa, (2, 0, 3, 4), (1, 2, 3, 4))
    del x6
    x63 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x63 += einsum(x5, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x5
    x64 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x64 += einsum(x62, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x62
    x64 += einsum(x63, (0, 1, 2, 3), (0, 1, 3, 2))
    del x63
    rdm2_f_aaaa_oovv += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x64, (0, 1, 2, 3), (1, 0, 2, 3))
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
    x68 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x68 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    x69 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x69 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4))
    x70 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x70 += einsum(x68, (0, 1), (0, 1))
    x70 += einsum(x69, (0, 1), (0, 1)) * 2.0
    rdm2_f_abab_oovv += einsum(x70, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    x71 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x71 += einsum(x70, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x70
    x72 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x72 += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x65
    x72 += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3)) * 8.0
    del x67
    x72 += einsum(x71, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x71
    rdm2_f_aaaa_oovv += einsum(x72, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_aaaa_oovv += einsum(x72, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x72
    x73 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x73 += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 0, 4, 5))
    rdm2_f_abab_ovvo += einsum(x73, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x74 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x74 += einsum(t2.abab, (0, 1, 2, 3), x73, (1, 3, 4, 5), (4, 0, 5, 2))
    x75 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x75 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3))
    x75 += einsum(x74, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x74
    rdm2_f_aaaa_oovv += einsum(x75, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x75, (0, 1, 2, 3), (0, 1, 2, 3))
    del x75
    x76 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x76 += einsum(t1.aa, (0, 1), x30, (0, 2, 3, 4), (2, 4, 3, 1))
    del x30
    rdm2_f_aaaa_oovv += einsum(t1.aa, (0, 1), x76, (0, 2, 3, 4), (2, 3, 4, 1)) * -2.0
    del x76
    x77 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x77 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), (3, 4, 0, 5))
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x77, (1, 4, 2, 5), (0, 4, 5, 3))
    rdm2_f_abab_vovo += einsum(x77, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_abab_vovv += einsum(t1.bb, (0, 1), x77, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    del x77
    x78 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x78 += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 2, 5, 0), (3, 1, 4, 5))
    rdm2_f_abab_ovvo += einsum(x78, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x79 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x79 += einsum(x73, (0, 1, 2, 3), (0, 1, 2, 3))
    x79 += einsum(x78, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_abab_oovv += einsum(t2.bbbb, (0, 1, 2, 3), x79, (1, 3, 4, 5), (4, 0, 5, 2)) * 4.0
    del x79
    x80 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x80 += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3))
    del x57
    x80 += einsum(x66, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x66
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x80, (0, 4, 2, 5), (4, 1, 5, 3))
    x81 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x81 += einsum(x55, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x55
    x81 += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3))
    del x52
    x81 += einsum(x54, (0, 1, 2, 3), (0, 1, 2, 3))
    del x54
    x81 += einsum(x53, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x53
    x81 += einsum(t1.aa, (0, 1), x37, (2, 3, 0, 4), (2, 3, 4, 1)) * -0.5
    del x37
    rdm2_f_abab_oovv += einsum(t1.bb, (0, 1), x81, (0, 2, 3, 4), (3, 2, 4, 1)) * -2.0
    del x81
    x82 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x82 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4))
    x83 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x83 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    x84 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x84 += einsum(x82, (0, 1), (0, 1))
    x84 += einsum(x83, (0, 1), (0, 1)) * 0.5
    rdm2_f_abab_oovv += einsum(x84, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    x85 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x85 += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3))
    del x36
    x85 += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x34
    x85 += einsum(x33, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x33
    x85 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x35
    rdm2_f_abab_oovv += einsum(t1.aa, (0, 1), x85, (2, 3, 0, 4), (4, 2, 1, 3)) * -1.0
    del x85
    x86 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x86 += einsum(t1.bb, (0, 1), (0, 1)) * -0.5
    x86 += einsum(x38, (0, 1), (0, 1)) * -1.0
    del x38
    x86 += einsum(x39, (0, 1), (0, 1)) * -0.5
    del x39
    x86 += einsum(x40, (0, 1), (0, 1))
    del x40
    x86 += einsum(x41, (0, 1), (0, 1)) * 0.5
    del x41
    x86 += einsum(x45, (0, 1), (0, 1))
    del x45
    rdm2_f_abab_oovv += einsum(t1.aa, (0, 1), x86, (2, 3), (0, 2, 1, 3)) * -2.0
    del x86
    x87 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x87 += einsum(x24, (0, 1), (0, 1)) * -0.5
    del x24
    x87 += einsum(x25, (0, 1), (0, 1)) * -1.0
    del x25
    x87 += einsum(x26, (0, 1), (0, 1)) * 0.5
    del x26
    x87 += einsum(x27, (0, 1), (0, 1))
    del x27
    x87 += einsum(t1.aa, (0, 1), x22, (0, 2), (2, 1))
    del x22
    rdm2_f_abab_oovv += einsum(t1.bb, (0, 1), x87, (2, 3), (2, 0, 3, 1)) * -2.0
    del x87
    x88 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x88 += einsum(t1.bb, (0, 1), x44, (0, 2, 3, 4), (2, 3, 1, 4))
    del x44
    x89 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x89 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_bbbb_ovov += einsum(x89, (0, 1, 2, 3), (1, 2, 0, 3)) * -4.0
    rdm2_f_bbbb_ovvo += einsum(x89, (0, 1, 2, 3), (1, 2, 3, 0)) * 4.0
    rdm2_f_bbbb_voov += einsum(x89, (0, 1, 2, 3), (2, 1, 0, 3)) * 4.0
    rdm2_f_bbbb_vovo += einsum(x89, (0, 1, 2, 3), (2, 1, 3, 0)) * -4.0
    x90 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x90 += einsum(t2.bbbb, (0, 1, 2, 3), x89, (1, 4, 3, 5), (4, 0, 5, 2))
    x91 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x91 += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 1, 5), (4, 5, 2, 0))
    rdm2_f_abab_voov += einsum(x91, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x92 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x92 += einsum(t2.abab, (0, 1, 2, 3), x91, (4, 5, 0, 2), (1, 4, 3, 5))
    x93 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x93 += einsum(x84, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 4, 1)) * -4.0
    del x84
    x94 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x94 += einsum(x88, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x88
    x94 += einsum(x90, (0, 1, 2, 3), (0, 1, 2, 3)) * 8.0
    del x90
    x94 += einsum(x92, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x92
    x94 += einsum(x93, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x93
    rdm2_f_bbbb_oovv += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_oovv += einsum(x94, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x94
    x95 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x95 += einsum(x11, (0, 1), t2.bbbb, (2, 0, 3, 4), (1, 2, 3, 4))
    del x11
    x96 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x96 += einsum(x18, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -4.0
    del x18
    x97 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x97 += einsum(x95, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x95
    x97 += einsum(x96, (0, 1, 2, 3), (0, 1, 3, 2))
    del x96
    rdm2_f_bbbb_oovv += einsum(x97, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_bbbb_oovv += einsum(x97, (0, 1, 2, 3), (1, 0, 2, 3))
    del x97
    x98 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x98 += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (4, 5, 2, 0))
    rdm2_f_abab_voov += einsum(x98, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x99 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x99 += einsum(t2.abab, (0, 1, 2, 3), x98, (4, 5, 0, 2), (4, 1, 5, 3))
    x100 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x100 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3))
    del x47
    x100 += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x48
    x101 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x101 += einsum(t1.bb, (0, 1), x100, (0, 2, 3, 4), (2, 3, 1, 4)) * 4.0
    del x100
    x102 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x102 += einsum(x99, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x99
    x102 += einsum(x101, (0, 1, 2, 3), (0, 1, 2, 3))
    del x101
    x102 += einsum(t1.bb, (0, 1), x46, (2, 3), (0, 2, 1, 3)) * 2.0
    del x46
    rdm2_f_bbbb_oovv += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_bbbb_oovv += einsum(x102, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_bbbb_oovv += einsum(x102, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_bbbb_oovv += einsum(x102, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x102
    x103 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x103 += einsum(t1.bb, (0, 1), x50, (0, 2, 3, 4), (2, 4, 3, 1))
    del x50
    rdm2_f_bbbb_oovv += einsum(t1.bb, (0, 1), x103, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    del x103
    x104 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x104 += einsum(t1.aa, (0, 1), x1, (2, 0, 3, 4), (2, 3, 4, 1))
    rdm2_f_aaaa_ovov += einsum(x104, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_aaaa_ovvo += einsum(x104, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    rdm2_f_aaaa_voov += einsum(x104, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_aaaa_vovo += einsum(x104, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x105 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x105 += einsum(l1.aa, (0, 1), t1.aa, (1, 2), (0, 2))
    x106 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x106 += einsum(x105, (0, 1), (0, 1))
    x106 += einsum(x68, (0, 1), (0, 1))
    x106 += einsum(x69, (0, 1), (0, 1)) * 2.0
    rdm2_f_aaaa_ovov += einsum(delta.aa.oo, (0, 1), x106, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_voov += einsum(delta.aa.oo, (0, 1), x106, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_abab_vovo += einsum(delta.bb.oo, (0, 1), x106, (2, 3), (2, 0, 3, 1))
    x107 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x107 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), (1, 5, 2, 4))
    rdm2_f_abab_ovov += einsum(x107, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_abab_ovvv += einsum(t1.aa, (0, 1), x107, (2, 3, 0, 4), (4, 2, 1, 3)) * -1.0
    del x107
    x108 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x108 += einsum(l1.bb, (0, 1), t1.bb, (1, 2), (0, 2))
    x109 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x109 += einsum(x108, (0, 1), (0, 1))
    x109 += einsum(x82, (0, 1), (0, 1)) * 2.0
    x109 += einsum(x83, (0, 1), (0, 1))
    rdm2_f_abab_ovov += einsum(delta.aa.oo, (0, 1), x109, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_ovvo += einsum(delta.bb.oo, (0, 1), x109, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_bbbb_vovo += einsum(delta.bb.oo, (0, 1), x109, (2, 3), (2, 0, 3, 1))
    rdm2_f_abab_ovvv += einsum(t1.aa, (0, 1), x109, (2, 3), (0, 2, 1, 3))
    x110 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x110 += einsum(t1.bb, (0, 1), x16, (2, 0, 3, 4), (2, 3, 4, 1))
    rdm2_f_bbbb_ovov += einsum(x110, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_bbbb_ovvo += einsum(x110, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    rdm2_f_bbbb_voov += einsum(x110, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_bbbb_vovo += einsum(x110, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x111 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x111 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), (3, 4, 1, 5))
    rdm2_f_bbbb_ovov += einsum(x111, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_bbbb_ovvo += einsum(x111, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_bbbb_voov += einsum(x111, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_bbbb_vovo += einsum(x111, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x112 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x112 += einsum(x108, (0, 1), (0, 1)) * 0.5
    del x108
    x112 += einsum(x82, (0, 1), (0, 1))
    del x82
    x112 += einsum(x83, (0, 1), (0, 1)) * 0.5
    del x83
    rdm2_f_bbbb_ovov += einsum(delta.bb.oo, (0, 1), x112, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_bbbb_voov += einsum(delta.bb.oo, (0, 1), x112, (2, 3), (2, 0, 1, 3)) * -2.0
    del x112
    x113 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x113 += einsum(x105, (0, 1), (0, 1)) * 0.5
    del x105
    x113 += einsum(x68, (0, 1), (0, 1)) * 0.5
    del x68
    x113 += einsum(x69, (0, 1), (0, 1))
    del x69
    rdm2_f_aaaa_ovvo += einsum(delta.aa.oo, (0, 1), x113, (2, 3), (0, 2, 3, 1)) * -2.0
    rdm2_f_aaaa_vovo += einsum(delta.aa.oo, (0, 1), x113, (2, 3), (2, 0, 3, 1)) * 2.0
    rdm2_f_abab_vovv += einsum(t1.bb, (0, 1), x113, (2, 3), (2, 0, 3, 1)) * 2.0
    del x113
    x114 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x114 += einsum(t1.aa, (0, 1), x8, (2, 3, 0, 4), (2, 3, 4, 1))
    del x8
    rdm2_f_abab_ovvo += einsum(x114, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x115 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x115 += einsum(t1.bb, (0, 1), x32, (0, 2, 3, 4), (2, 1, 3, 4))
    del x32
    rdm2_f_abab_voov += einsum(x115, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    x116 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x116 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_aaaa_ovvv = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_ovvv += einsum(x116, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_aaaa_vovv = np.zeros((nvir[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_vovv += einsum(x116, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x116
    x117 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x117 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (0, 1, 4, 5), (4, 5, 2, 3))
    del x1
    rdm2_f_aaaa_ovvv += einsum(x117, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_aaaa_vovv += einsum(x117, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x117
    x118 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x118 += einsum(t1.aa, (0, 1), x104, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    del x104
    rdm2_f_aaaa_ovvv += einsum(x118, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_aaaa_vovv += einsum(x118, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x118
    x119 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x119 += einsum(t1.aa, (0, 1), x80, (0, 2, 3, 4), (2, 1, 3, 4))
    del x80
    x120 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x120 += einsum(x119, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x119
    x120 += einsum(t1.aa, (0, 1), x106, (2, 3), (0, 2, 1, 3))
    del x106
    rdm2_f_aaaa_ovvv += einsum(x120, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_aaaa_ovvv += einsum(x120, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_aaaa_vovv += einsum(x120, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_aaaa_vovv += einsum(x120, (0, 1, 2, 3), (1, 0, 3, 2))
    del x120
    x121 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x121 += einsum(x73, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x73
    x121 += einsum(x78, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x78
    x121 += einsum(x114, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x114
    rdm2_f_abab_ovvv += einsum(t1.bb, (0, 1), x121, (0, 2, 3, 4), (3, 2, 4, 1))
    del x121
    x122 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x122 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_bbbb_ovvv = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_ovvv += einsum(x122, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_bbbb_vovv = np.zeros((nvir[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_vovv += einsum(x122, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x122
    x123 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x123 += einsum(t2.bbbb, (0, 1, 2, 3), x16, (0, 1, 4, 5), (4, 5, 2, 3))
    del x16
    rdm2_f_bbbb_ovvv += einsum(x123, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_bbbb_vovv += einsum(x123, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x123
    x124 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x124 += einsum(t1.bb, (0, 1), x110, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    del x110
    rdm2_f_bbbb_ovvv += einsum(x124, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_bbbb_vovv += einsum(x124, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x124
    x125 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x125 += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3))
    del x89
    x125 += einsum(x111, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x111
    x126 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x126 += einsum(t1.bb, (0, 1), x125, (0, 2, 3, 4), (2, 1, 3, 4)) * 4.0
    del x125
    x127 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x127 += einsum(x126, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x126
    x127 += einsum(t1.bb, (0, 1), x109, (2, 3), (0, 2, 1, 3))
    del x109
    rdm2_f_bbbb_ovvv += einsum(x127, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_ovvv += einsum(x127, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_bbbb_vovv += einsum(x127, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_bbbb_vovv += einsum(x127, (0, 1, 2, 3), (1, 0, 3, 2))
    del x127
    x128 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x128 += einsum(x98, (0, 1, 2, 3), (0, 1, 2, 3))
    del x98
    x128 += einsum(x115, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x115
    x128 += einsum(x91, (0, 1, 2, 3), (0, 1, 2, 3))
    del x91
    rdm2_f_abab_vovv += einsum(t1.aa, (0, 1), x128, (2, 3, 0, 4), (4, 2, 1, 3)) * 2.0
    del x128
    x129 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x129 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_aaaa_vvov = np.zeros((nvir[0], nvir[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_vvov += einsum(x129, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_aaaa_vvvo = np.zeros((nvir[0], nvir[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_vvvo += einsum(x129, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    rdm2_f_aaaa_vvvv += einsum(t1.aa, (0, 1), x129, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    del x129
    x130 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x130 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_bbbb_vvov = np.zeros((nvir[1], nvir[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_vvov += einsum(x130, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_bbbb_vvvo = np.zeros((nvir[1], nvir[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_vvvo += einsum(x130, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    rdm2_f_bbbb_vvvv += einsum(t1.bb, (0, 1), x130, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    del x130
    x131 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x131 += einsum(t1.aa, (0, 1), l2.abab, (2, 3, 0, 4), (4, 3, 2, 1))
    rdm2_f_abab_vvvo = np.zeros((nvir[0], nvir[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_vvvo += einsum(x131, (0, 1, 2, 3), (2, 1, 3, 0))
    rdm2_f_abab_vvvv += einsum(t1.bb, (0, 1), x131, (0, 2, 3, 4), (3, 2, 4, 1))
    del x131

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

