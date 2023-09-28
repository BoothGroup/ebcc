# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x0 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x1 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x1 += einsum(f.bb.ov, (0, 1), (0, 1)) * 2.0
    x1 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3)) * 2.0
    x1 += einsum(t1.bb, (0, 1), x0, (0, 2, 3, 1), (2, 3)) * -1.0
    del x0
    e_cc += einsum(t1.bb, (0, 1), x1, (0, 1), ()) * 0.5
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x3 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x3 += einsum(f.aa.ov, (0, 1), (0, 1)) * 2.0
    x3 += einsum(t1.aa, (0, 1), x2, (0, 2, 3, 1), (2, 3)) * -1.0
    del x2
    e_cc += einsum(t1.aa, (0, 1), x3, (0, 1), ()) * 0.5
    del x3

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    t1new = Namespace()
    t2new = Namespace()

    # T amplitudes
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    t1new_aa += einsum(f.aa.ov, (0, 1), (0, 1))
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    t1new_bb += einsum(f.bb.ov, (0, 1), (0, 1))
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.oooo, (4, 0, 5, 1), (4, 5, 2, 3)) * 2.0
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.oovv, (4, 0, 5, 3), (4, 1, 2, 5)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvoo, (4, 2, 5, 1), (0, 5, 4, 3)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.oooo, (4, 0, 5, 1), (4, 5, 2, 3))
    t2new_abab += einsum(t1.aa, (0, 1), v.aabb.vvov, (2, 1, 3, 4), (0, 3, 2, 4))
    t2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5)) * 2.0
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.oooo, (4, 1, 5, 0), (4, 5, 2, 3)) * -2.0
    x0 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x0 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    t1new_aa += einsum(x0, (0, 1), (0, 1))
    x1 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1 += einsum(v.aabb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x1 += einsum(t1.aa, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (2, 0, 3, 4))
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x1, (0, 4, 1, 3), (4, 2)) * -1.0
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x2 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x3 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x3 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x3 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), x3, (4, 0, 1, 3), (4, 2)) * 2.0
    del x3
    x4 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x4 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x4 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x4, (0, 4, 3, 1), (4, 2)) * -2.0
    del x4
    x5 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x5 += einsum(t2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    x5 += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    t1new_aa += einsum(v.aabb.vvov, (0, 1, 2, 3), x5, (4, 2, 1, 3), (4, 0))
    t1new_bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), x5, (0, 4, 1, 3), (4, 2))
    t2new_abab += einsum(v.aabb.vvvv, (0, 1, 2, 3), x5, (4, 5, 1, 3), (4, 5, 0, 2))
    del x5
    x6 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x6 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(x6, (0, 1), (0, 1))
    x7 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x7 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x7 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x8 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x8 += einsum(t1.bb, (0, 1), x7, (0, 2, 3, 1), (2, 3))
    del x7
    x9 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x9 += einsum(f.bb.ov, (0, 1), (0, 1))
    x9 += einsum(x6, (0, 1), (0, 1))
    del x6
    x9 += einsum(x8, (0, 1), (0, 1)) * -1.0
    del x8
    t1new_aa += einsum(x9, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    t1new_bb += einsum(x9, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    x10 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x10 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x10 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x11 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x11 += einsum(t1.aa, (0, 1), x10, (0, 2, 3, 1), (2, 3))
    del x10
    x12 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x12 += einsum(f.aa.ov, (0, 1), (0, 1))
    x12 += einsum(x0, (0, 1), (0, 1))
    del x0
    x12 += einsum(x11, (0, 1), (0, 1)) * -1.0
    del x11
    t1new_aa += einsum(x12, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_bb += einsum(x12, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x13 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x13 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_aa += einsum(t1.aa, (0, 1), x13, (0, 2, 1, 3), (2, 3)) * -1.0
    del x13
    x14 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x14 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x15 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x15 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x16 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x16 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x16 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x17 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x17 += einsum(f.aa.oo, (0, 1), (0, 1))
    x17 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3))
    x17 += einsum(x14, (0, 1), (1, 0)) * 2.0
    x17 += einsum(x15, (0, 1), (1, 0))
    x17 += einsum(t1.aa, (0, 1), x16, (0, 2, 3, 1), (3, 2)) * -1.0
    del x16
    x17 += einsum(t1.aa, (0, 1), x12, (2, 1), (2, 0))
    del x12
    t1new_aa += einsum(t1.aa, (0, 1), x17, (0, 2), (2, 1)) * -1.0
    del x17
    x18 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x18 += einsum(f.aa.vv, (0, 1), (0, 1))
    x18 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_aa += einsum(t1.aa, (0, 1), x18, (1, 2), (0, 2))
    del x18
    x19 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x19 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x20 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x20 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x20 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), x20, (4, 0, 1, 3), (4, 2)) * 2.0
    del x20
    x21 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x21 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (2, 0, 4, 3))
    x22 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x22 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (0, 2, 3, 1))
    x22 += einsum(x21, (0, 1, 2, 3), (0, 2, 1, 3))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x22, (0, 1, 4, 2), (4, 3)) * -1.0
    del x22
    x23 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x23 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x23 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_bb += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x23, (0, 4, 1, 3), (4, 2)) * 2.0
    del x23
    x24 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x24 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x24 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_bb += einsum(t1.bb, (0, 1), x24, (0, 2, 1, 3), (2, 3)) * -1.0
    x25 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x25 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x26 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x26 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 1, 3), (0, 4))
    x27 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x27 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x27 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x28 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x28 += einsum(f.bb.oo, (0, 1), (0, 1))
    x28 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3))
    x28 += einsum(x25, (0, 1), (1, 0))
    x28 += einsum(x26, (0, 1), (1, 0)) * 2.0
    x28 += einsum(t1.bb, (0, 1), x27, (2, 3, 0, 1), (3, 2)) * -1.0
    del x27
    x28 += einsum(t1.bb, (0, 1), x9, (2, 1), (2, 0))
    del x9
    t1new_bb += einsum(t1.bb, (0, 1), x28, (0, 2), (2, 1)) * -1.0
    del x28
    x29 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x29 += einsum(f.bb.vv, (0, 1), (0, 1))
    x29 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (0, 2, 3, 1), (2, 3)) * -1.0
    t1new_bb += einsum(t1.bb, (0, 1), x29, (1, 2), (0, 2))
    del x29
    x30 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x30 += einsum(t1.aa, (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x31 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x31 += einsum(t1.aa, (0, 1), x30, (2, 3, 1, 4), (0, 2, 3, 4))
    del x30
    x32 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x32 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x33 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x33 += einsum(t2.aaaa, (0, 1, 2, 3), x32, (4, 1, 5, 3), (0, 4, 2, 5))
    x34 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x34 += einsum(t2.abab, (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x35 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x35 += einsum(t2.abab, (0, 1, 2, 3), x34, (4, 1, 5, 3), (0, 4, 2, 5))
    del x34
    x36 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x36 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x37 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x37 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    x38 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x38 += einsum(x36, (0, 1), (0, 1))
    x38 += einsum(x37, (0, 1), (0, 1)) * 0.5
    x39 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x39 += einsum(x38, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 4, 0)) * -2.0
    del x38
    x40 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x40 += einsum(t1.aa, (0, 1), x2, (2, 3, 4, 1), (2, 0, 4, 3))
    del x2
    x41 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x41 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x41 += einsum(x40, (0, 1, 2, 3), (3, 1, 0, 2))
    del x40
    x42 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x42 += einsum(t1.aa, (0, 1), x41, (0, 2, 3, 4), (2, 3, 4, 1))
    del x41
    x43 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x43 += einsum(t1.aa, (0, 1), x42, (2, 3, 0, 4), (3, 2, 1, 4))
    del x42
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x44 += einsum(x31, (0, 1, 2, 3), (0, 1, 2, 3))
    del x31
    x44 += einsum(x33, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x33
    x44 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    del x35
    x44 += einsum(x39, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x39
    x44 += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3))
    del x43
    t2new_aaaa += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x44, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x44
    x45 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x45 += einsum(f.aa.oo, (0, 1), t2.aaaa, (2, 1, 3, 4), (0, 2, 3, 4))
    x46 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x46 += einsum(x14, (0, 1), (0, 1))
    x46 += einsum(x15, (0, 1), (0, 1)) * 0.5
    x47 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x47 += einsum(x46, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x46
    x48 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x48 += einsum(x45, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x45
    x48 += einsum(x47, (0, 1, 2, 3), (0, 1, 3, 2))
    del x47
    t2new_aaaa += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x48, (0, 1, 2, 3), (1, 0, 2, 3))
    del x48
    x49 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x49 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x50 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x50 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x51 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x51 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x51 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x51 += einsum(x50, (0, 1, 2, 3), (1, 0, 3, 2))
    x52 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x52 += einsum(t2.aaaa, (0, 1, 2, 3), x51, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x51
    x53 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x53 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x54 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x54 += einsum(t1.aa, (0, 1), x53, (2, 3, 4, 0), (2, 4, 3, 1))
    del x53
    x55 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x55 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x55 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x55 += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3))
    x56 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x56 += einsum(t1.aa, (0, 1), x55, (2, 3, 1, 4), (0, 2, 3, 4))
    del x55
    x57 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x57 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x57 += einsum(x54, (0, 1, 2, 3), (0, 2, 1, 3))
    del x54
    x57 += einsum(x56, (0, 1, 2, 3), (0, 2, 1, 3))
    del x56
    x58 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x58 += einsum(t1.aa, (0, 1), x57, (2, 0, 3, 4), (2, 3, 1, 4))
    del x57
    x59 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x59 += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3))
    del x49
    x59 += einsum(x50, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x59 += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x52
    x59 += einsum(x58, (0, 1, 2, 3), (0, 1, 2, 3))
    del x58
    t2new_aaaa += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x59, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x59, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x59
    x60 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x60 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x61 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x61 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x61 += einsum(x60, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x60
    t2new_aaaa += einsum(x61, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x61, (0, 1, 2, 3), (0, 1, 2, 3))
    del x61
    x62 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x62 += einsum(t1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (2, 0, 3, 4))
    t2new_abab += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3))
    x63 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x63 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (4, 0, 5, 2))
    t2new_abab += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x64 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x64 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x64 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x64 += einsum(x32, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x32
    x64 += einsum(x50, (0, 1, 2, 3), (1, 0, 3, 2))
    del x50
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x64, (0, 4, 2, 5), (4, 1, 5, 3))
    del x64
    x65 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x65 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x66 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x66 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x66 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x66 += einsum(x65, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x66, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x66
    x67 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x67 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x67 += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab += einsum(t2.aaaa, (0, 1, 2, 3), x67, (1, 4, 3, 5), (0, 4, 2, 5)) * 4.0
    del x67
    x68 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x68 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    x69 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x69 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 4, 1, 3), (2, 4))
    x70 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x70 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x70 += einsum(x68, (0, 1), (1, 0)) * 0.5
    x70 += einsum(x69, (0, 1), (1, 0))
    t2new_abab += einsum(x70, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x70
    x71 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x71 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x71 += einsum(x36, (0, 1), (1, 0))
    del x36
    x71 += einsum(x37, (0, 1), (1, 0)) * 0.5
    del x37
    t2new_abab += einsum(x71, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    del x71
    x72 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x72 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x72 += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3))
    del x62
    x73 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x73 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (0, 2, 3, 1))
    x73 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3))
    x74 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x74 += einsum(v.aabb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x74 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 4, 1), (2, 3, 4, 0))
    x74 += einsum(t1.aa, (0, 1), x73, (2, 3, 4, 1), (0, 2, 4, 3))
    del x73
    x75 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x75 += einsum(v.aabb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x75 += einsum(t1.bb, (0, 1), v.aabb.oovv, (2, 3, 4, 1), (2, 3, 0, 4))
    x75 += einsum(t1.aa, (0, 1), x72, (2, 3, 1, 4), (2, 0, 3, 4))
    del x72
    x75 += einsum(t1.bb, (0, 1), x74, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x74
    t2new_abab += einsum(t1.aa, (0, 1), x75, (0, 2, 3, 4), (2, 3, 1, 4)) * -1.0
    del x75
    x76 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x76 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x76 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 4, 1), (0, 4, 2, 3))
    x77 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x77 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (0, 2, 3, 1))
    x77 += einsum(x21, (0, 1, 2, 3), (0, 2, 1, 3))
    del x21
    x77 += einsum(t1.aa, (0, 1), x76, (2, 3, 1, 4), (0, 3, 2, 4))
    del x76
    t2new_abab += einsum(t1.bb, (0, 1), x77, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x77
    x78 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x78 += einsum(f.bb.oo, (0, 1), (0, 1)) * 2.0
    x78 += einsum(x25, (0, 1), (1, 0))
    x78 += einsum(x26, (0, 1), (1, 0)) * 2.0
    t2new_abab += einsum(x78, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -0.5
    del x78
    x79 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x79 += einsum(f.aa.oo, (0, 1), (0, 1))
    x79 += einsum(x14, (0, 1), (1, 0))
    del x14
    x79 += einsum(x15, (0, 1), (1, 0)) * 0.5
    del x15
    t2new_abab += einsum(x79, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    del x79
    x80 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x80 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x81 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x81 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x81 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -1.0
    x82 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x82 += einsum(x24, (0, 1, 2, 3), x81, (0, 4, 2, 5), (1, 4, 3, 5))
    del x24, x81
    x83 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x83 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x83 += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x63
    x84 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x84 += einsum(t2.abab, (0, 1, 2, 3), x83, (0, 4, 2, 5), (1, 4, 3, 5))
    del x83
    x85 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x85 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x86 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x86 += einsum(t1.bb, (0, 1), x85, (2, 3, 4, 0), (2, 4, 3, 1))
    del x85
    x87 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x87 += einsum(t1.bb, (0, 1), x80, (2, 3, 1, 4), (0, 2, 3, 4))
    x88 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x88 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x88 += einsum(x86, (0, 1, 2, 3), (0, 2, 1, 3))
    del x86
    x88 += einsum(x87, (0, 1, 2, 3), (0, 2, 1, 3))
    del x87
    x89 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x89 += einsum(t1.bb, (0, 1), x88, (2, 0, 3, 4), (2, 3, 1, 4))
    del x88
    x90 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x90 += einsum(x80, (0, 1, 2, 3), (0, 1, 2, 3))
    del x80
    x90 += einsum(x82, (0, 1, 2, 3), (1, 0, 3, 2))
    del x82
    x90 += einsum(x84, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x84
    x90 += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3))
    del x89
    t2new_bbbb += einsum(x90, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x90, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb += einsum(x90, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb += einsum(x90, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x90
    x91 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x91 += einsum(f.bb.oo, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4))
    x92 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x92 += einsum(x25, (0, 1), (0, 1))
    del x25
    x92 += einsum(x26, (0, 1), (0, 1)) * 2.0
    del x26
    x93 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x93 += einsum(x92, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    del x92
    x94 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x94 += einsum(x91, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x91
    x94 += einsum(x93, (0, 1, 2, 3), (0, 1, 3, 2))
    del x93
    t2new_bbbb += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x94, (0, 1, 2, 3), (1, 0, 2, 3))
    del x94
    x95 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x95 += einsum(t1.bb, (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x96 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x96 += einsum(t1.bb, (0, 1), x95, (2, 3, 1, 4), (0, 2, 3, 4))
    del x95
    x97 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x97 += einsum(t2.bbbb, (0, 1, 2, 3), x65, (4, 1, 5, 3), (0, 4, 2, 5))
    del x65
    x98 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x98 += einsum(x68, (0, 1), (0, 1))
    del x68
    x98 += einsum(x69, (0, 1), (0, 1)) * 2.0
    del x69
    x99 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x99 += einsum(x98, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 4, 0)) * -1.0
    del x98
    x100 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x100 += einsum(t1.bb, (0, 1), x19, (2, 3, 4, 1), (2, 0, 4, 3))
    del x19
    x101 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x101 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x101 += einsum(x100, (0, 1, 2, 3), (3, 1, 0, 2))
    del x100
    x102 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x102 += einsum(t1.bb, (0, 1), x101, (0, 2, 3, 4), (2, 3, 4, 1))
    del x101
    x103 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x103 += einsum(t1.bb, (0, 1), x102, (2, 3, 0, 4), (3, 2, 1, 4))
    del x102
    x104 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x104 += einsum(x96, (0, 1, 2, 3), (0, 1, 2, 3))
    del x96
    x104 += einsum(x97, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x97
    x104 += einsum(x99, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x99
    x104 += einsum(x103, (0, 1, 2, 3), (0, 1, 2, 3))
    del x103
    t2new_bbbb += einsum(x104, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x104, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x104
    x105 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x105 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x106 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x106 += einsum(t2.abab, (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 2), (4, 1, 5, 3))
    x107 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x107 += einsum(t2.abab, (0, 1, 2, 3), x106, (0, 4, 2, 5), (4, 1, 5, 3))
    del x106
    x108 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x108 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x108 += einsum(x105, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x105
    x108 += einsum(x107, (0, 1, 2, 3), (1, 0, 3, 2))
    del x107
    t2new_bbbb += einsum(x108, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x108, (0, 1, 2, 3), (0, 1, 2, 3))
    del x108

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t1new": t1new, "t2new": t2new}

