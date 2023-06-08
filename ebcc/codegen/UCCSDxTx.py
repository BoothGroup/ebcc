# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 2), ()) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x0 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x1 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x1 += einsum(f.bb.ov, (0, 1), (0, 1))
    x1 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    x1 += einsum(t1.bb, (0, 1), x0, (0, 2, 1, 3), (2, 3)) * -0.5
    del x0
    e_cc += einsum(t1.bb, (0, 1), x1, (0, 1), ())
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x3 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x3 += einsum(f.aa.ov, (0, 1), (0, 1)) * 2.0
    x3 += einsum(t1.aa, (0, 1), x2, (0, 2, 1, 3), (2, 3)) * -1.0
    del x2
    e_cc += einsum(t1.aa, (0, 1), x3, (0, 1), ()) * 0.5
    del x3

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    t1new = Namespace()
    t2new = Namespace()

    # T amplitudes
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    t1new_bb += einsum(f.bb.ov, (0, 1), (0, 1))
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    t1new_aa += einsum(f.aa.ov, (0, 1), (0, 1))
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    t2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    x0 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x0 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(x0, (0, 1), (0, 1))
    x1 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x2 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x2 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), x2, (4, 0, 1, 3), (4, 2)) * 2.0
    del x2
    x3 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x3 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (0, 4, 2, 3))
    x4 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x4 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x4 += einsum(x3, (0, 1, 2, 3), (1, 0, 2, 3))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x4, (1, 4, 0, 2), (4, 3)) * -1.0
    x5 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x5 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x5 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_bb += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x5, (0, 4, 3, 1), (4, 2)) * -2.0
    del x5
    x6 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x6 += einsum(t2.abab, (0, 1, 2, 3), (1, 3, 0, 2))
    x6 += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (2, 3, 0, 1))
    t1new_bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), x6, (4, 3, 0, 1), (4, 2))
    t1new_aa += einsum(v.aabb.vvov, (0, 1, 2, 3), x6, (2, 3, 4, 1), (4, 0))
    x7 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x7 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    t1new_aa += einsum(x7, (0, 1), (0, 1))
    x8 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x8 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x8 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x9 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x9 += einsum(t1.aa, (0, 1), x8, (0, 2, 1, 3), (2, 3))
    x10 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x10 += einsum(f.aa.ov, (0, 1), (0, 1))
    x10 += einsum(x7, (0, 1), (0, 1))
    del x7
    x10 += einsum(x9, (0, 1), (0, 1)) * -1.0
    del x9
    t1new_bb += einsum(x10, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    t1new_aa += einsum(x10, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    x11 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x11 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x11 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x12 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x12 += einsum(t1.bb, (0, 1), x11, (0, 2, 1, 3), (2, 3))
    x13 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x13 += einsum(f.bb.ov, (0, 1), (0, 1))
    x13 += einsum(x0, (0, 1), (0, 1))
    del x0
    x13 += einsum(x12, (0, 1), (0, 1)) * -1.0
    del x12
    t1new_bb += einsum(x13, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_aa += einsum(x13, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    x14 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x14 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x14 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_bb += einsum(t1.bb, (0, 1), x14, (0, 2, 1, 3), (2, 3)) * -1.0
    del x14
    x15 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x15 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3))
    x16 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x16 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x17 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x17 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x18 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x18 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x18 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x19 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x19 += einsum(t1.bb, (0, 1), x18, (2, 3, 0, 1), (2, 3))
    x20 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x20 += einsum(t1.bb, (0, 1), x13, (2, 1), (0, 2))
    x21 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x21 += einsum(f.bb.oo, (0, 1), (0, 1))
    x21 += einsum(x15, (0, 1), (1, 0))
    x21 += einsum(x16, (0, 1), (1, 0)) * 2.0
    x21 += einsum(x17, (0, 1), (1, 0))
    x21 += einsum(x19, (0, 1), (1, 0)) * -1.0
    x21 += einsum(x20, (0, 1), (1, 0))
    t1new_bb += einsum(t1.bb, (0, 1), x21, (0, 2), (2, 1)) * -1.0
    t2new_abab += einsum(x21, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x21
    x22 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x22 += einsum(f.bb.vv, (0, 1), (0, 1))
    x22 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(t1.bb, (0, 1), x22, (1, 2), (0, 2))
    del x22
    x23 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x23 += einsum(t1.aa, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (3, 4, 0, 2))
    x24 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x24 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x24 += einsum(x23, (0, 1, 2, 3), (0, 1, 3, 2))
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x24, (1, 3, 0, 4), (4, 2)) * -1.0
    del x24
    x25 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x25 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x26 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x26 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x26 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), x26, (4, 1, 0, 3), (4, 2)) * -2.0
    del x26
    x27 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x27 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x27 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x27, (0, 4, 3, 1), (4, 2)) * -2.0
    del x27
    x28 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x28 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x28 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_aa += einsum(t1.aa, (0, 1), x28, (0, 2, 1, 3), (2, 3)) * -1.0
    x29 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x29 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3))
    x30 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x30 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x31 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x31 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x32 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x32 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x32 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x33 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x33 += einsum(t1.aa, (0, 1), x32, (0, 2, 3, 1), (2, 3))
    del x32
    x34 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x34 += einsum(t1.aa, (0, 1), x10, (2, 1), (0, 2))
    x35 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x35 += einsum(f.aa.oo, (0, 1), (0, 1))
    x35 += einsum(x29, (0, 1), (1, 0))
    x35 += einsum(x30, (0, 1), (1, 0))
    x35 += einsum(x31, (0, 1), (1, 0)) * 2.0
    x35 += einsum(x33, (0, 1), (1, 0)) * -1.0
    x35 += einsum(x34, (0, 1), (1, 0))
    t1new_aa += einsum(t1.aa, (0, 1), x35, (0, 2), (2, 1)) * -1.0
    t2new_abab += einsum(x35, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    del x35
    x36 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x36 += einsum(f.aa.vv, (0, 1), (0, 1))
    x36 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_aa += einsum(t1.aa, (0, 1), x36, (1, 2), (0, 2))
    del x36
    x37 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x37 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x38 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x38 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x39 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x39 += einsum(t2.aaaa, (0, 1, 2, 3), x38, (4, 5, 1, 0), (4, 5, 2, 3)) * -1.0
    x40 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x40 += einsum(f.aa.oo, (0, 1), (0, 1))
    x40 += einsum(x34, (0, 1), (0, 1))
    del x34
    x41 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x41 += einsum(x40, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x40
    x42 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x42 += einsum(x29, (0, 1), (1, 0))
    del x29
    x42 += einsum(x30, (0, 1), (1, 0))
    del x30
    x42 += einsum(x31, (0, 1), (1, 0)) * 2.0
    del x31
    x42 += einsum(x33, (0, 1), (1, 0)) * -1.0
    del x33
    x43 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x43 += einsum(x42, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x42
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x44 += einsum(x37, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x37
    x44 += einsum(x39, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x39
    x44 += einsum(x41, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x41
    x44 += einsum(x43, (0, 1, 2, 3), (0, 1, 3, 2))
    del x43
    t2new_aaaa += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x44, (0, 1, 2, 3), (1, 0, 2, 3))
    del x44
    x45 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x45 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x46 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x46 += einsum(t1.aa, (0, 1), v.aabb.vvov, (2, 1, 3, 4), (3, 4, 0, 2))
    t2new_abab += einsum(x46, (0, 1, 2, 3), (2, 0, 3, 1))
    x47 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x47 += einsum(t2.abab, (0, 1, 2, 3), x46, (1, 3, 4, 5), (4, 0, 2, 5))
    x48 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x48 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x48 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -0.5
    x49 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x49 += einsum(x28, (0, 1, 2, 3), x48, (0, 4, 2, 5), (1, 4, 3, 5)) * 2.0
    del x28
    x50 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x50 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ovov, (1, 3, 4, 5), (4, 5, 0, 2))
    x51 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x51 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x51 += einsum(x50, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x50
    x52 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x52 += einsum(t2.abab, (0, 1, 2, 3), x51, (1, 3, 4, 5), (0, 4, 2, 5))
    del x51
    x53 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x53 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x53 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x54 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x54 += einsum(t1.aa, (0, 1), x53, (2, 3, 1, 4), (0, 2, 3, 4))
    del x53
    x55 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x55 += einsum(t2.aaaa, (0, 1, 2, 3), x54, (4, 1, 3, 5), (0, 4, 2, 5)) * -2.0
    del x54
    x56 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x56 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x57 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x57 += einsum(t1.aa, (0, 1), x38, (2, 3, 4, 0), (2, 4, 3, 1))
    del x38
    x58 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x58 += einsum(t1.aa, (0, 1), x45, (2, 3, 1, 4), (0, 2, 3, 4))
    x59 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x59 += einsum(t2.abab, (0, 1, 2, 3), x23, (1, 3, 4, 5), (4, 0, 5, 2))
    del x23
    x60 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x60 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x60 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x61 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x61 += einsum(t2.aaaa, (0, 1, 2, 3), x60, (1, 4, 5, 3), (0, 4, 5, 2)) * 2.0
    x62 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x62 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x62 += einsum(x25, (0, 1, 2, 3), (0, 2, 1, 3))
    x63 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x63 += einsum(t2.aaaa, (0, 1, 2, 3), x62, (4, 1, 5, 3), (0, 4, 5, 2)) * 2.0
    del x62
    x64 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x64 += einsum(x56, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x56
    x64 += einsum(x57, (0, 1, 2, 3), (0, 2, 1, 3))
    del x57
    x64 += einsum(x58, (0, 1, 2, 3), (0, 2, 1, 3))
    del x58
    x64 += einsum(x59, (0, 1, 2, 3), (0, 2, 1, 3))
    del x59
    x64 += einsum(x61, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x61
    x64 += einsum(x63, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x63
    x65 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x65 += einsum(t1.aa, (0, 1), x64, (2, 0, 3, 4), (2, 3, 1, 4))
    del x64
    x66 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x66 += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    del x45
    x66 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3))
    del x47
    x66 += einsum(x49, (0, 1, 2, 3), (1, 0, 3, 2))
    del x49
    x66 += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x52
    x66 += einsum(x55, (0, 1, 2, 3), (1, 0, 2, 3))
    del x55
    x66 += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3))
    del x65
    t2new_aaaa += einsum(x66, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x66, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa += einsum(x66, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x66, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x66
    x67 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x67 += einsum(t1.aa, (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x68 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x68 += einsum(t1.aa, (0, 1), x67, (2, 3, 1, 4), (0, 2, 3, 4))
    del x67
    x69 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x69 += einsum(t2.aaaa, (0, 1, 2, 3), x8, (1, 4, 3, 5), (4, 0, 5, 2))
    x70 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x70 += einsum(t2.aaaa, (0, 1, 2, 3), x69, (1, 4, 3, 5), (0, 4, 2, 5)) * -4.0
    del x69
    x71 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x71 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 1), (2, 3))
    x72 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x72 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    x73 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x73 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 4, 1, 3), (2, 4))
    x74 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x74 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x74 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x75 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x75 += einsum(t1.aa, (0, 1), x74, (0, 1, 2, 3), (2, 3))
    del x74
    x76 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x76 += einsum(x71, (0, 1), (1, 0)) * -1.0
    x76 += einsum(x72, (0, 1), (1, 0))
    x76 += einsum(x73, (0, 1), (1, 0)) * 2.0
    x76 += einsum(x75, (0, 1), (1, 0)) * -1.0
    x77 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x77 += einsum(x76, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x76
    x78 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x78 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x79 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x79 += einsum(x10, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4)) * -2.0
    x80 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x80 += einsum(t1.aa, (0, 1), x25, (2, 3, 4, 1), (2, 0, 4, 3))
    del x25
    x81 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x81 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x81 += einsum(x80, (0, 1, 2, 3), (3, 1, 2, 0))
    x82 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x82 += einsum(t1.aa, (0, 1), x81, (0, 2, 3, 4), (2, 3, 4, 1))
    del x81
    x83 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x83 += einsum(x78, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    del x78
    x83 += einsum(x79, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x79
    x83 += einsum(x82, (0, 1, 2, 3), (1, 0, 2, 3))
    del x82
    x84 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x84 += einsum(t1.aa, (0, 1), x83, (0, 2, 3, 4), (2, 3, 1, 4))
    del x83
    x85 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x85 += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3))
    del x68
    x85 += einsum(x70, (0, 1, 2, 3), (1, 0, 3, 2))
    del x70
    x85 += einsum(x77, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x77
    x85 += einsum(x84, (0, 1, 2, 3), (1, 0, 2, 3))
    del x84
    t2new_aaaa += einsum(x85, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x85, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x85
    x86 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x86 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x87 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x87 += einsum(t2.abab, (0, 1, 2, 3), x11, (1, 4, 3, 5), (4, 5, 0, 2))
    del x11
    x88 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x88 += einsum(t2.abab, (0, 1, 2, 3), x87, (1, 3, 4, 5), (0, 4, 2, 5)) * -1.0
    del x87
    x89 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x89 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x89 += einsum(x86, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x86
    x89 += einsum(x88, (0, 1, 2, 3), (0, 1, 2, 3))
    del x88
    t2new_aaaa += einsum(x89, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3))
    del x89
    x90 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x90 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x91 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x91 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x91 += einsum(x90, (0, 1, 2, 3), (3, 1, 0, 2))
    x91 += einsum(x80, (0, 1, 2, 3), (3, 1, 0, 2))
    del x80
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x91, (0, 4, 5, 1), (5, 4, 2, 3)) * -2.0
    del x91
    x92 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x92 += einsum(t1.aa, (0, 1), x90, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x90
    x92 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x92 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * 0.5
    t2new_aaaa += einsum(t1.aa, (0, 1), x92, (2, 3, 0, 4), (2, 3, 1, 4)) * 2.0
    del x92
    x93 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x93 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 4, 3, 5))
    x94 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x94 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x94 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x95 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x95 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x95 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -0.5
    x96 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x96 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x96 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x97 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x97 += einsum(t1.bb, (0, 1), x96, (2, 3, 1, 4), (0, 2, 3, 4))
    x98 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x98 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x98 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x98 += einsum(x93, (0, 1, 2, 3), (1, 0, 3, 2))
    x98 += einsum(x94, (0, 1, 2, 3), x95, (0, 4, 3, 5), (1, 4, 2, 5)) * -2.0
    del x95
    x98 += einsum(x97, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x98 += einsum(t1.bb, (0, 1), x18, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x18
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x98, (1, 4, 3, 5), (0, 4, 2, 5))
    del x98
    x99 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x99 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x99 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x100 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x100 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x100 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x101 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x101 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x101 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x101 += einsum(x48, (0, 1, 2, 3), x99, (0, 4, 2, 5), (4, 1, 5, 3)) * -2.0
    del x99
    x101 += einsum(t1.aa, (0, 1), x100, (2, 1, 3, 4), (2, 0, 4, 3)) * -1.0
    del x100
    x101 += einsum(t1.aa, (0, 1), x60, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    del x60
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x101, (0, 4, 2, 5), (4, 1, 5, 3)) * -1.0
    del x101
    x102 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x102 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x102 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 4), (4, 1, 2, 3))
    x102 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (3, 4, 2, 0)) * -1.0
    x102 += einsum(v.aabb.ovov, (0, 1, 2, 3), x6, (2, 4, 5, 1), (3, 4, 0, 5))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x102, (3, 4, 0, 5), (5, 1, 2, 4))
    del x102
    x103 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x103 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x103 += einsum(t1.aa, (0, 1), v.aabb.ooov, (2, 0, 3, 4), (3, 4, 2, 1)) * -1.0
    x103 += einsum(x46, (0, 1, 2, 3), (0, 1, 2, 3))
    x103 += einsum(v.aabb.ovov, (0, 1, 2, 3), x48, (0, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    del x48
    t2new_abab += einsum(t2.bbbb, (0, 1, 2, 3), x103, (1, 3, 4, 5), (4, 0, 5, 2)) * 2.0
    del x103
    x104 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x104 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x104 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -1.0
    x105 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x105 += einsum(t1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (0, 4, 2, 3))
    x106 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x106 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x106 += einsum(x105, (0, 1, 2, 3), (0, 1, 2, 3))
    x106 += einsum(t1.bb, (0, 1), x4, (0, 2, 3, 4), (2, 1, 3, 4)) * -1.0
    del x4
    t2new_abab += einsum(x104, (0, 1, 2, 3), x106, (4, 5, 0, 2), (1, 4, 3, 5))
    del x104, x106
    x107 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x107 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x107 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    x108 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x108 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x108 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 4, 1), (4, 0, 2, 3))
    x108 += einsum(t1.aa, (0, 1), x107, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x107
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x108, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    del x108
    x109 = np.zeros((nvir[1], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    x109 += einsum(v.aabb.vvvv, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x109 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 4), (4, 1, 2, 3))
    x109 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 2, 3, 4), (3, 4, 2, 1))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x109, (3, 4, 2, 5), (0, 1, 5, 4)) * -1.0
    del x109
    x110 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x110 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 4, 1), (0, 4, 2, 3))
    x111 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x111 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x111 += einsum(x110, (0, 1, 2, 3), (1, 0, 3, 2))
    x111 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (2, 1, 3, 4), (3, 4, 2, 0))
    x111 += einsum(v.aabb.ovov, (0, 1, 2, 3), x6, (4, 3, 5, 1), (2, 4, 0, 5))
    del x6
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x111, (1, 4, 0, 5), (5, 4, 2, 3))
    del x111
    x112 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x112 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 1, 2, 3), (2, 3))
    x113 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x113 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 4, 1, 3), (2, 4))
    x114 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x114 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    x115 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x115 += einsum(t1.bb, (0, 1), x96, (0, 2, 1, 3), (2, 3))
    del x96
    x116 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x116 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x116 += einsum(x112, (0, 1), (1, 0)) * -1.0
    x116 += einsum(x113, (0, 1), (1, 0)) * 2.0
    x116 += einsum(x114, (0, 1), (1, 0))
    x116 += einsum(x115, (0, 1), (0, 1)) * -1.0
    x116 += einsum(t1.bb, (0, 1), x13, (0, 2), (2, 1))
    t2new_abab += einsum(x116, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x116
    x117 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x117 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x117 += einsum(x71, (0, 1), (1, 0)) * -1.0
    del x71
    x117 += einsum(x72, (0, 1), (1, 0))
    del x72
    x117 += einsum(x73, (0, 1), (1, 0)) * 2.0
    del x73
    x117 += einsum(x75, (0, 1), (1, 0)) * -1.0
    del x75
    x117 += einsum(t1.aa, (0, 1), x10, (0, 2), (2, 1))
    del x10
    t2new_abab += einsum(x117, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    del x117
    x118 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x118 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x118 += einsum(x46, (0, 1, 2, 3), (0, 1, 2, 3))
    del x46
    x119 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x119 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x119 += einsum(x110, (0, 1, 2, 3), (0, 1, 3, 2))
    del x110
    x119 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (1, 5, 4, 0))
    x120 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x120 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x120 += einsum(t1.aa, (0, 1), v.aabb.vvoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x120 += einsum(t1.bb, (0, 1), x118, (2, 1, 3, 4), (2, 0, 3, 4))
    del x118
    x120 += einsum(t1.aa, (0, 1), x119, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x119
    t2new_abab += einsum(t1.bb, (0, 1), x120, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    del x120
    x121 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x121 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (2, 3, 0, 1))
    x121 += einsum(t1.aa, (0, 1), v.aabb.vvvv, (2, 1, 3, 4), (3, 4, 0, 2))
    t2new_abab += einsum(t1.bb, (0, 1), x121, (1, 2, 3, 4), (3, 0, 4, 2))
    del x121
    x122 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x122 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x122 += einsum(t1.bb, (0, 1), v.aabb.oovv, (2, 3, 4, 1), (0, 4, 2, 3))
    t2new_abab += einsum(t1.aa, (0, 1), x122, (2, 3, 0, 4), (4, 2, 1, 3)) * -1.0
    del x122
    x123 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x123 += einsum(t1.bb, (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x124 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x124 += einsum(t1.bb, (0, 1), x123, (2, 3, 1, 4), (0, 2, 3, 4))
    del x123
    x125 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x125 += einsum(t2.bbbb, (0, 1, 2, 3), x94, (1, 4, 5, 3), (0, 4, 2, 5))
    del x94
    x126 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x126 += einsum(t2.bbbb, (0, 1, 2, 3), x125, (4, 1, 5, 3), (0, 4, 2, 5)) * -4.0
    del x125
    x127 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x127 += einsum(t2.abab, (0, 1, 2, 3), x8, (0, 4, 2, 5), (1, 3, 4, 5))
    del x8
    x128 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x128 += einsum(t2.abab, (0, 1, 2, 3), x127, (4, 5, 0, 2), (1, 4, 3, 5)) * -1.0
    del x127
    x129 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x129 += einsum(x112, (0, 1), (1, 0)) * -1.0
    del x112
    x129 += einsum(x113, (0, 1), (1, 0)) * 2.0
    del x113
    x129 += einsum(x114, (0, 1), (1, 0))
    del x114
    x129 += einsum(x115, (0, 1), (0, 1)) * -1.0
    del x115
    x130 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x130 += einsum(x129, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x129
    x131 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x131 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x132 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x132 += einsum(x13, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4)) * -2.0
    del x13
    x133 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x133 += einsum(t1.bb, (0, 1), x1, (2, 3, 4, 1), (2, 0, 4, 3))
    x134 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x134 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x134 += einsum(x133, (0, 1, 2, 3), (3, 1, 2, 0))
    del x133
    x135 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x135 += einsum(t1.bb, (0, 1), x134, (0, 2, 3, 4), (2, 3, 4, 1))
    del x134
    x136 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x136 += einsum(x131, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    del x131
    x136 += einsum(x132, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x132
    x136 += einsum(x135, (0, 1, 2, 3), (1, 0, 2, 3))
    del x135
    x137 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x137 += einsum(t1.bb, (0, 1), x136, (0, 2, 3, 4), (2, 3, 1, 4))
    del x136
    x138 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x138 += einsum(x124, (0, 1, 2, 3), (0, 1, 2, 3))
    del x124
    x138 += einsum(x126, (0, 1, 2, 3), (0, 1, 2, 3))
    del x126
    x138 += einsum(x128, (0, 1, 2, 3), (0, 1, 2, 3))
    del x128
    x138 += einsum(x130, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x130
    x138 += einsum(x137, (0, 1, 2, 3), (1, 0, 2, 3))
    del x137
    t2new_bbbb += einsum(x138, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x138, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x138
    x139 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x139 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x140 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x140 += einsum(t2.abab, (0, 1, 2, 3), x105, (4, 5, 0, 2), (4, 1, 3, 5))
    del x105
    x141 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x141 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x141 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x141 += einsum(x93, (0, 1, 2, 3), (1, 0, 3, 2))
    x142 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x142 += einsum(t2.bbbb, (0, 1, 2, 3), x141, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x141
    x143 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x143 += einsum(t2.bbbb, (0, 1, 2, 3), x97, (4, 1, 3, 5), (0, 4, 2, 5)) * -2.0
    del x97
    x144 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x144 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x145 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x145 += einsum(t1.bb, (0, 1), x144, (2, 3, 4, 0), (2, 4, 3, 1))
    x146 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x146 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 5), (1, 4, 5, 3))
    x147 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x147 += einsum(t2.abab, (0, 1, 2, 3), x3, (4, 5, 0, 2), (4, 1, 5, 3))
    del x3
    x148 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x148 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x148 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x149 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x149 += einsum(t2.bbbb, (0, 1, 2, 3), x148, (1, 4, 5, 3), (0, 4, 5, 2)) * 2.0
    del x148
    x150 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x150 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x150 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3))
    del x1
    x151 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x151 += einsum(t2.bbbb, (0, 1, 2, 3), x150, (4, 1, 5, 3), (0, 4, 5, 2)) * 2.0
    del x150
    x152 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x152 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x152 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x152 += einsum(x139, (0, 1, 2, 3), (0, 1, 2, 3))
    x153 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x153 += einsum(t1.bb, (0, 1), x152, (2, 3, 1, 4), (0, 2, 3, 4))
    del x152
    x154 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x154 += einsum(x145, (0, 1, 2, 3), (0, 2, 1, 3))
    del x145
    x154 += einsum(x146, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x146
    x154 += einsum(x147, (0, 1, 2, 3), (0, 2, 1, 3))
    del x147
    x154 += einsum(x149, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x149
    x154 += einsum(x151, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x151
    x154 += einsum(x153, (0, 1, 2, 3), (0, 2, 1, 3))
    del x153
    x155 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x155 += einsum(t1.bb, (0, 1), x154, (2, 0, 3, 4), (2, 3, 1, 4))
    del x154
    x156 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x156 += einsum(x139, (0, 1, 2, 3), (0, 1, 2, 3))
    del x139
    x156 += einsum(x93, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x93
    x156 += einsum(x140, (0, 1, 2, 3), (0, 1, 2, 3))
    del x140
    x156 += einsum(x142, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x142
    x156 += einsum(x143, (0, 1, 2, 3), (1, 0, 2, 3))
    del x143
    x156 += einsum(x155, (0, 1, 2, 3), (0, 1, 2, 3))
    del x155
    t2new_bbbb += einsum(x156, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x156, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb += einsum(x156, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb += einsum(x156, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x156
    x157 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x157 += einsum(t2.bbbb, (0, 1, 2, 3), x144, (4, 5, 0, 1), (4, 5, 2, 3))
    del x144
    x158 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x158 += einsum(f.bb.oo, (0, 1), (0, 1))
    x158 += einsum(x20, (0, 1), (0, 1))
    del x20
    x159 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x159 += einsum(x158, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x158
    x160 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x160 += einsum(x15, (0, 1), (1, 0))
    del x15
    x160 += einsum(x16, (0, 1), (1, 0)) * 2.0
    del x16
    x160 += einsum(x17, (0, 1), (1, 0))
    del x17
    x160 += einsum(x19, (0, 1), (1, 0)) * -1.0
    del x19
    x161 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x161 += einsum(x160, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x160
    x162 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x162 += einsum(x157, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x157
    x162 += einsum(x159, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x159
    x162 += einsum(x161, (0, 1, 2, 3), (0, 1, 3, 2))
    del x161
    t2new_bbbb += einsum(x162, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x162, (0, 1, 2, 3), (1, 0, 2, 3))
    del x162
    x163 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x163 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x164 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x164 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x164 += einsum(x163, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x163
    t2new_bbbb += einsum(x164, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x164, (0, 1, 2, 3), (0, 1, 2, 3))
    del x164
    x165 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x165 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new_bbbb += einsum(x165, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x165, (0, 1, 2, 3), (1, 0, 2, 3))
    del x165
    x166 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x166 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x166 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    x167 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x167 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x167 += einsum(v.bbbb.ovov, (0, 1, 2, 3), x166, (4, 5, 3, 1), (0, 5, 2, 4))
    del x166
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x167, (0, 4, 1, 5), (4, 5, 2, 3)) * 2.0
    del x167
    x168 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x168 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x169 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x169 += einsum(t1.bb, (0, 1), x168, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    del x168
    x169 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x169 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    t2new_bbbb += einsum(t1.bb, (0, 1), x169, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x169

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t1new": t1new, "t2new": t2new}

def energy_perturbative(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    denom3 = Namespace()
    denom3.aaaaaa = 1 / direct_sum(
            "ia+jb+kc->ijkabc",
            direct_sum("i-a->ia", np.diag(f.aa.oo), np.diag(f.aa.vv)),
            direct_sum("i-a->ia", np.diag(f.aa.oo), np.diag(f.aa.vv)),
            direct_sum("i-a->ia", np.diag(f.aa.oo), np.diag(f.aa.vv)),
    )
    denom3.babbab = 1 / direct_sum(
            "ia+jb+kc->ijkabc",
            direct_sum("i-a->ia", np.diag(f.bb.oo), np.diag(f.bb.vv)),
            direct_sum("i-a->ia", np.diag(f.aa.oo), np.diag(f.aa.vv)),
            direct_sum("i-a->ia", np.diag(f.bb.oo), np.diag(f.bb.vv)),
    )
    denom3.abaaba = 1 / direct_sum(
            "ia+jb+kc->ijkabc",
            direct_sum("i-a->ia", np.diag(f.aa.oo), np.diag(f.aa.vv)),
            direct_sum("i-a->ia", np.diag(f.bb.oo), np.diag(f.bb.vv)),
            direct_sum("i-a->ia", np.diag(f.aa.oo), np.diag(f.aa.vv)),
    )
    denom3.bbbbbb = 1 / direct_sum(
            "ia+jb+kc->ijkabc",
            direct_sum("i-a->ia", np.diag(f.bb.oo), np.diag(f.bb.vv)),
            direct_sum("i-a->ia", np.diag(f.bb.oo), np.diag(f.bb.vv)),
            direct_sum("i-a->ia", np.diag(f.bb.oo), np.diag(f.bb.vv)),
    )

    # energy
    e_pert = 0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), v.bbbb.ooov, (4, 2, 5, 6), v.bbbb.ovov, (4, 3, 5, 6), denom3.babbab, (4, 1, 5, 3, 0, 6), ()) * -1.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), v.bbbb.ooov, (4, 2, 5, 6), v.bbbb.ovov, (4, 6, 5, 3), denom3.babbab, (4, 1, 5, 6, 0, 3), ())
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), v.bbbb.ooov, (4, 2, 5, 6), v.bbbb.ovov, (5, 3, 4, 6), denom3.babbab, (5, 1, 4, 3, 0, 6), ())
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), v.bbbb.ooov, (4, 2, 5, 6), v.bbbb.ovov, (5, 6, 4, 3), denom3.babbab, (5, 1, 4, 6, 0, 3), ()) * -1.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), v.aabb.ooov, (1, 2, 5, 6), v.bbbb.ovov, (3, 4, 5, 6), denom3.babbab, (3, 1, 5, 4, 0, 6), ()) * -1.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), v.aabb.ooov, (1, 2, 5, 6), v.bbbb.ovov, (3, 6, 5, 4), denom3.babbab, (3, 1, 5, 6, 0, 4), ())
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), v.aabb.ooov, (1, 2, 5, 6), v.bbbb.ovov, (5, 4, 3, 6), denom3.babbab, (5, 1, 3, 4, 0, 6), ())
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 4, 5), v.aaaa.ooov, (1, 2, 6, 0), v.aabb.ovov, (6, 4, 3, 5), denom3.abaaba, (1, 3, 6, 0, 5, 4), ()) * 2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), v.aaaa.ooov, (1, 2, 5, 6), v.aabb.ovov, (5, 6, 3, 4), denom3.abaaba, (1, 3, 5, 0, 4, 6), ()) * -2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), v.aabb.ooov, (1, 2, 5, 6), v.bbbb.ovov, (5, 6, 3, 4), denom3.babbab, (5, 1, 3, 6, 0, 4), ()) * -1.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 4, 5), v.aaaa.ooov, (6, 2, 1, 0), v.aabb.ovov, (6, 4, 3, 5), denom3.abaaba, (1, 3, 6, 0, 5, 4), ()) * -2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), v.aaaa.ooov, (5, 2, 1, 6), v.aabb.ovov, (5, 6, 3, 4), denom3.abaaba, (1, 3, 5, 0, 4, 6), ()) * 2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 3, 4), v.aabb.ovoo, (5, 0, 6, 2), v.aabb.ovov, (5, 3, 6, 4), denom3.abaaba, (1, 6, 5, 0, 4, 3), ()) * 2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), v.aabb.ovoo, (4, 5, 6, 2), v.aabb.ovov, (4, 5, 6, 3), denom3.abaaba, (1, 6, 4, 0, 3, 5), ()) * -2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 4, 5), v.aabb.ovoo, (1, 0, 6, 3), v.aabb.ovov, (2, 4, 6, 5), denom3.abaaba, (1, 6, 2, 0, 5, 4), ()) * -2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), v.aabb.ovoo, (1, 5, 6, 3), v.aabb.ovov, (2, 5, 6, 4), denom3.abaaba, (1, 6, 2, 0, 4, 5), ()) * 2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), v.bbbb.ovov, (2, 4, 5, 6), v.bbbb.ovvv, (5, 4, 6, 3), denom3.babbab, (2, 1, 5, 4, 0, 6), ()) * -1.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), v.bbbb.ovov, (2, 4, 5, 6), v.bbbb.ovvv, (5, 6, 4, 3), denom3.babbab, (2, 1, 5, 4, 0, 6), ())
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 3, 4), v.aabb.ovov, (5, 3, 2, 6), v.aabb.ovvv, (5, 0, 6, 4), denom3.abaaba, (1, 2, 5, 0, 6, 3), ()) * -2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), v.aabb.ovov, (4, 5, 2, 6), v.aabb.ovvv, (4, 5, 6, 3), denom3.abaaba, (1, 2, 4, 0, 6, 5), ()) * 2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), v.bbbb.ovov, (4, 5, 2, 6), v.bbbb.ovvv, (4, 5, 6, 3), denom3.babbab, (4, 1, 2, 5, 0, 6), ())
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), v.bbbb.ovov, (4, 5, 2, 6), v.bbbb.ovvv, (4, 6, 5, 3), denom3.babbab, (4, 1, 2, 5, 0, 6), ()) * -1.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 3, 4), v.aabb.ovov, (5, 6, 2, 4), v.aaaa.ovvv, (5, 0, 6, 3), denom3.abaaba, (1, 2, 5, 0, 4, 6), ()) * -2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 3, 4), v.aabb.ovov, (5, 6, 2, 4), v.aaaa.ovvv, (5, 6, 0, 3), denom3.abaaba, (1, 2, 5, 0, 4, 6), ()) * 2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 4, 5), v.aabb.ovov, (2, 4, 3, 6), v.aabb.ovvv, (1, 0, 6, 5), denom3.abaaba, (1, 3, 2, 0, 6, 4), ()) * 2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), v.aabb.ovov, (2, 5, 3, 6), v.aabb.ovvv, (1, 5, 6, 4), denom3.abaaba, (1, 3, 2, 0, 6, 5), ()) * -2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 4, 5), v.aabb.ovov, (2, 6, 3, 5), v.aaaa.ovvv, (1, 0, 6, 4), denom3.abaaba, (1, 3, 2, 0, 5, 6), ()) * 2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 4, 5), v.aabb.ovov, (2, 6, 3, 5), v.aaaa.ovvv, (1, 6, 0, 4), denom3.abaaba, (1, 3, 2, 0, 5, 6), ()) * -2.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 3, 4), v.bbbb.ovov, (2, 4, 5, 6), v.aabb.vvov, (0, 3, 5, 6), denom3.babbab, (2, 1, 5, 4, 0, 6), ())
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 3, 4), v.bbbb.ovov, (2, 5, 6, 4), v.aabb.vvov, (0, 3, 6, 5), denom3.babbab, (2, 1, 6, 5, 0, 4), ()) * -1.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 3, 4), v.bbbb.ovov, (5, 4, 2, 6), v.aabb.vvov, (0, 3, 5, 6), denom3.babbab, (5, 1, 2, 4, 0, 6), ()) * -1.0
    e_pert += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 3, 4), v.bbbb.ovov, (5, 6, 2, 4), v.aabb.vvov, (0, 3, 5, 6), denom3.babbab, (5, 1, 2, 6, 0, 4), ())
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 0, 3), v.aaaa.ooov, (4, 2, 5, 6), v.aaaa.ovov, (4, 3, 5, 6), denom3.aaaaaa, (1, 4, 5, 0, 3, 6), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 0, 3), v.aabb.ooov, (4, 2, 5, 6), v.aabb.ovov, (4, 3, 5, 6), denom3.abaaba, (1, 5, 4, 0, 6, 3), ()) * -4.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 3, 4), v.aaaa.ooov, (5, 2, 6, 0), v.aaaa.ovov, (5, 3, 6, 4), denom3.aaaaaa, (1, 5, 6, 0, 3, 4), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 0, 3), v.aaaa.ooov, (4, 2, 5, 6), v.aaaa.ovov, (4, 6, 5, 3), denom3.aaaaaa, (1, 4, 5, 0, 6, 3), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 0, 3), v.aaaa.ooov, (4, 2, 5, 6), v.aaaa.ovov, (5, 3, 4, 6), denom3.aaaaaa, (1, 5, 4, 0, 3, 6), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 3, 4), v.aaaa.ooov, (5, 2, 6, 0), v.aaaa.ovov, (6, 3, 5, 4), denom3.aaaaaa, (1, 6, 5, 0, 3, 4), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 0, 3), v.aaaa.ooov, (4, 2, 5, 6), v.aaaa.ovov, (5, 6, 4, 3), denom3.aaaaaa, (1, 5, 4, 0, 6, 3), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 0, 4), v.aaaa.ooov, (1, 3, 5, 6), v.aaaa.ovov, (2, 4, 5, 6), denom3.aaaaaa, (1, 2, 5, 0, 4, 6), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 0, 4), v.aabb.ooov, (1, 3, 5, 6), v.aabb.ovov, (2, 4, 5, 6), denom3.abaaba, (1, 5, 2, 0, 6, 4), ()) * 4.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), v.aaaa.ooov, (1, 3, 6, 0), v.aaaa.ovov, (2, 4, 6, 5), denom3.aaaaaa, (1, 2, 6, 0, 4, 5), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 0, 4), v.aaaa.ooov, (1, 3, 5, 6), v.aaaa.ovov, (2, 6, 5, 4), denom3.aaaaaa, (1, 2, 5, 0, 6, 4), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 0, 4), v.aaaa.ooov, (1, 3, 5, 6), v.aaaa.ovov, (5, 4, 2, 6), denom3.aaaaaa, (1, 5, 2, 0, 4, 6), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), v.aaaa.ooov, (1, 3, 6, 0), v.aaaa.ovov, (6, 4, 2, 5), denom3.aaaaaa, (1, 6, 2, 0, 4, 5), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 0, 4), v.aaaa.ooov, (1, 3, 5, 6), v.aaaa.ovov, (5, 6, 2, 4), denom3.aaaaaa, (1, 5, 2, 0, 6, 4), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 0, 4), v.aaaa.ooov, (5, 3, 1, 6), v.aaaa.ovov, (2, 4, 5, 6), denom3.aaaaaa, (1, 2, 5, 0, 4, 6), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), v.aaaa.ooov, (6, 3, 1, 0), v.aaaa.ovov, (2, 4, 6, 5), denom3.aaaaaa, (1, 2, 6, 0, 4, 5), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 0, 4), v.aaaa.ooov, (5, 3, 1, 6), v.aaaa.ovov, (2, 6, 5, 4), denom3.aaaaaa, (1, 2, 5, 0, 6, 4), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 0, 4), v.aaaa.ooov, (5, 3, 1, 6), v.aaaa.ovov, (5, 4, 2, 6), denom3.aaaaaa, (1, 5, 2, 0, 4, 6), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), v.aaaa.ooov, (6, 3, 1, 0), v.aaaa.ovov, (6, 4, 2, 5), denom3.aaaaaa, (1, 6, 2, 0, 4, 5), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 0, 4), v.aaaa.ooov, (5, 3, 1, 6), v.aaaa.ovov, (5, 6, 2, 4), denom3.aaaaaa, (1, 5, 2, 0, 6, 4), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.bbbb, (2, 3, 4, 5), v.aabb.ovoo, (1, 0, 6, 3), v.bbbb.ovov, (2, 4, 6, 5), denom3.babbab, (2, 1, 6, 4, 0, 5), ()) * -2.0
    e_pert += einsum(l1.aa, (0, 1), t2.bbbb, (2, 3, 4, 5), v.aabb.ovoo, (1, 0, 6, 3), v.bbbb.ovov, (6, 4, 2, 5), denom3.babbab, (6, 1, 2, 4, 0, 5), ()) * 2.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 3, 4), v.aaaa.ovov, (2, 3, 5, 6), v.aaaa.ovvv, (5, 0, 6, 4), denom3.aaaaaa, (1, 2, 5, 0, 3, 6), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 3, 4), v.aaaa.ovov, (2, 3, 5, 6), v.aaaa.ovvv, (5, 6, 0, 4), denom3.aaaaaa, (1, 2, 5, 0, 3, 6), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 0, 3), v.aaaa.ovov, (2, 4, 5, 6), v.aaaa.ovvv, (5, 4, 6, 3), denom3.aaaaaa, (1, 2, 5, 0, 4, 6), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 0, 3), v.aaaa.ovov, (2, 4, 5, 6), v.aaaa.ovvv, (5, 6, 4, 3), denom3.aaaaaa, (1, 2, 5, 0, 4, 6), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 3, 4), v.aaaa.ovov, (2, 5, 6, 3), v.aaaa.ovvv, (6, 0, 5, 4), denom3.aaaaaa, (1, 2, 6, 0, 5, 3), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 3, 4), v.aaaa.ovov, (2, 5, 6, 3), v.aaaa.ovvv, (6, 5, 0, 4), denom3.aaaaaa, (1, 2, 6, 0, 5, 3), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 3, 4), v.aaaa.ovov, (5, 3, 2, 6), v.aaaa.ovvv, (5, 0, 6, 4), denom3.aaaaaa, (1, 5, 2, 0, 3, 6), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 3, 4), v.aaaa.ovov, (5, 3, 2, 6), v.aaaa.ovvv, (5, 6, 0, 4), denom3.aaaaaa, (1, 5, 2, 0, 3, 6), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 0, 3), v.aaaa.ovov, (4, 5, 2, 6), v.aaaa.ovvv, (4, 5, 6, 3), denom3.aaaaaa, (1, 4, 2, 0, 5, 6), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 0, 3), v.aaaa.ovov, (4, 5, 2, 6), v.aaaa.ovvv, (4, 6, 5, 3), denom3.aaaaaa, (1, 4, 2, 0, 5, 6), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 3, 4), v.aaaa.ovov, (5, 6, 2, 3), v.aaaa.ovvv, (5, 0, 6, 4), denom3.aaaaaa, (1, 5, 2, 0, 6, 3), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 3, 4), v.aaaa.ovov, (5, 6, 2, 3), v.aaaa.ovvv, (5, 6, 0, 4), denom3.aaaaaa, (1, 5, 2, 0, 6, 3), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), v.aaaa.ovov, (2, 4, 3, 6), v.aaaa.ovvv, (1, 0, 6, 5), denom3.aaaaaa, (1, 2, 3, 0, 4, 6), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.bbbb, (2, 3, 4, 5), v.bbbb.ovov, (2, 4, 3, 6), v.aabb.ovvv, (1, 0, 6, 5), denom3.babbab, (2, 1, 3, 4, 0, 6), ()) * 2.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), v.aaaa.ovov, (2, 4, 3, 6), v.aaaa.ovvv, (1, 6, 0, 5), denom3.aaaaaa, (1, 2, 3, 0, 4, 6), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 0, 4), v.aaaa.ovov, (2, 5, 3, 6), v.aaaa.ovvv, (1, 5, 6, 4), denom3.aaaaaa, (1, 2, 3, 0, 5, 6), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 0, 4), v.aaaa.ovov, (2, 5, 3, 6), v.aaaa.ovvv, (1, 6, 5, 4), denom3.aaaaaa, (1, 2, 3, 0, 5, 6), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), v.aaaa.ovov, (2, 6, 3, 4), v.aaaa.ovvv, (1, 0, 6, 5), denom3.aaaaaa, (1, 2, 3, 0, 6, 4), ()) * -6.0
    e_pert += einsum(l1.aa, (0, 1), t2.bbbb, (2, 3, 4, 5), v.bbbb.ovov, (2, 6, 3, 4), v.aabb.ovvv, (1, 0, 6, 5), denom3.babbab, (2, 1, 3, 6, 0, 4), ()) * -2.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), v.aaaa.ovov, (2, 6, 3, 4), v.aaaa.ovvv, (1, 6, 0, 5), denom3.aaaaaa, (1, 2, 3, 0, 6, 4), ()) * 6.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 3, 4), v.aabb.ovov, (2, 3, 5, 6), v.aabb.vvov, (0, 4, 5, 6), denom3.abaaba, (1, 5, 2, 0, 6, 3), ()) * -4.0
    e_pert += einsum(l1.aa, (0, 1), t2.aaaa, (1, 2, 0, 3), v.aabb.ovov, (2, 4, 5, 6), v.aabb.vvov, (4, 3, 5, 6), denom3.abaaba, (1, 5, 2, 0, 6, 4), ()) * 4.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), v.bbbb.ooov, (1, 3, 5, 6), v.aabb.ovov, (2, 4, 5, 6), denom3.babbab, (1, 2, 5, 0, 4, 6), ()) * -2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 5), v.bbbb.ooov, (1, 3, 6, 0), v.aabb.ovov, (2, 4, 6, 5), denom3.babbab, (1, 2, 6, 0, 4, 5), ()) * 2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), v.aabb.ooov, (4, 2, 5, 6), v.aabb.ovov, (4, 3, 5, 6), denom3.babbab, (1, 4, 5, 0, 3, 6), ()) * -2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 4), v.aabb.ooov, (5, 2, 6, 0), v.aabb.ovov, (5, 3, 6, 4), denom3.babbab, (1, 5, 6, 0, 3, 4), ()) * 2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), v.aaaa.ooov, (4, 2, 5, 6), v.aaaa.ovov, (4, 3, 5, 6), denom3.abaaba, (4, 1, 5, 3, 0, 6), ()) * -1.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), v.aaaa.ooov, (4, 2, 5, 6), v.aaaa.ovov, (4, 6, 5, 3), denom3.abaaba, (4, 1, 5, 6, 0, 3), ())
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), v.aaaa.ooov, (4, 2, 5, 6), v.aaaa.ovov, (5, 3, 4, 6), denom3.abaaba, (5, 1, 4, 3, 0, 6), ())
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), v.aaaa.ooov, (4, 2, 5, 6), v.aaaa.ovov, (5, 6, 4, 3), denom3.abaaba, (5, 1, 4, 6, 0, 3), ()) * -1.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), v.bbbb.ooov, (5, 3, 1, 6), v.aabb.ovov, (2, 4, 5, 6), denom3.babbab, (1, 2, 5, 0, 4, 6), ()) * 2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 5), v.bbbb.ooov, (6, 3, 1, 0), v.aabb.ovov, (2, 4, 6, 5), denom3.babbab, (1, 2, 6, 0, 4, 5), ()) * -2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), v.aabb.ooov, (5, 2, 1, 6), v.aabb.ovov, (5, 4, 3, 6), denom3.babbab, (1, 5, 3, 0, 4, 6), ()) * 2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 5), v.aabb.ooov, (6, 2, 1, 0), v.aabb.ovov, (6, 4, 3, 5), denom3.babbab, (1, 6, 3, 0, 4, 5), ()) * -2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), v.aabb.ovoo, (5, 6, 1, 3), v.aaaa.ovov, (2, 4, 5, 6), denom3.abaaba, (2, 1, 5, 4, 0, 6), ()) * -1.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), v.aabb.ovoo, (5, 6, 1, 3), v.aaaa.ovov, (2, 6, 5, 4), denom3.abaaba, (2, 1, 5, 6, 0, 4), ())
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), v.aabb.ovoo, (5, 6, 1, 3), v.aaaa.ovov, (5, 4, 2, 6), denom3.abaaba, (5, 1, 2, 4, 0, 6), ())
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), v.aabb.ovoo, (5, 6, 1, 3), v.aaaa.ovov, (5, 6, 2, 4), denom3.abaaba, (5, 1, 2, 6, 0, 4), ()) * -1.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 5), v.aabb.ovov, (2, 4, 3, 6), v.bbbb.ovvv, (1, 0, 6, 5), denom3.babbab, (1, 2, 3, 0, 4, 6), ()) * 2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 5), v.aabb.ovov, (2, 4, 3, 6), v.bbbb.ovvv, (1, 6, 0, 5), denom3.babbab, (1, 2, 3, 0, 4, 6), ()) * -2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 4), v.aabb.ovov, (2, 3, 5, 6), v.bbbb.ovvv, (5, 0, 6, 4), denom3.babbab, (1, 2, 5, 0, 3, 6), ()) * -2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 4), v.aabb.ovov, (2, 3, 5, 6), v.bbbb.ovvv, (5, 6, 0, 4), denom3.babbab, (1, 2, 5, 0, 3, 6), ()) * 2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 4), v.aaaa.ovov, (2, 3, 5, 6), v.aabb.ovvv, (5, 6, 0, 4), denom3.abaaba, (2, 1, 5, 3, 0, 6), ())
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), v.aaaa.ovov, (2, 4, 5, 6), v.aaaa.ovvv, (5, 4, 6, 3), denom3.abaaba, (2, 1, 5, 4, 0, 6), ()) * -1.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), v.aaaa.ovov, (2, 4, 5, 6), v.aaaa.ovvv, (5, 6, 4, 3), denom3.abaaba, (2, 1, 5, 4, 0, 6), ())
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 4), v.aaaa.ovov, (2, 5, 6, 3), v.aabb.ovvv, (6, 5, 0, 4), denom3.abaaba, (2, 1, 6, 5, 0, 3), ()) * -1.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 4), v.aaaa.ovov, (5, 3, 2, 6), v.aabb.ovvv, (5, 6, 0, 4), denom3.abaaba, (5, 1, 2, 3, 0, 6), ()) * -1.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), v.aaaa.ovov, (4, 5, 2, 6), v.aaaa.ovvv, (4, 5, 6, 3), denom3.abaaba, (4, 1, 2, 5, 0, 6), ())
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), v.aaaa.ovov, (4, 5, 2, 6), v.aaaa.ovvv, (4, 6, 5, 3), denom3.abaaba, (4, 1, 2, 5, 0, 6), ()) * -1.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 4), v.aaaa.ovov, (5, 6, 2, 3), v.aabb.ovvv, (5, 6, 0, 4), denom3.abaaba, (5, 1, 2, 6, 0, 3), ())
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), v.aabb.ovov, (2, 4, 5, 6), v.aabb.vvov, (4, 3, 5, 6), denom3.babbab, (1, 2, 5, 0, 4, 6), ()) * 2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), v.aabb.ovov, (2, 5, 3, 6), v.aabb.vvov, (5, 4, 1, 6), denom3.babbab, (1, 2, 3, 0, 5, 6), ()) * -2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 4), v.aabb.ovov, (2, 5, 6, 4), v.aabb.vvov, (5, 3, 6, 0), denom3.babbab, (1, 2, 6, 0, 5, 4), ()) * -2.0
    e_pert += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 5), v.aabb.ovov, (2, 6, 3, 5), v.aabb.vvov, (6, 4, 1, 0), denom3.babbab, (1, 2, 3, 0, 6, 5), ()) * 2.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 0, 3), v.bbbb.ooov, (4, 2, 5, 6), v.bbbb.ovov, (4, 3, 5, 6), denom3.bbbbbb, (1, 4, 5, 0, 3, 6), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 3, 4), v.bbbb.ooov, (5, 2, 6, 0), v.bbbb.ovov, (5, 3, 6, 4), denom3.bbbbbb, (1, 5, 6, 0, 3, 4), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 0, 3), v.bbbb.ooov, (4, 2, 5, 6), v.bbbb.ovov, (4, 6, 5, 3), denom3.bbbbbb, (1, 4, 5, 0, 6, 3), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 0, 3), v.bbbb.ooov, (4, 2, 5, 6), v.bbbb.ovov, (5, 3, 4, 6), denom3.bbbbbb, (1, 5, 4, 0, 3, 6), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 3, 4), v.bbbb.ooov, (5, 2, 6, 0), v.bbbb.ovov, (6, 3, 5, 4), denom3.bbbbbb, (1, 6, 5, 0, 3, 4), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 0, 3), v.bbbb.ooov, (4, 2, 5, 6), v.bbbb.ovov, (5, 6, 4, 3), denom3.bbbbbb, (1, 5, 4, 0, 6, 3), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 0, 4), v.bbbb.ooov, (1, 3, 5, 6), v.bbbb.ovov, (2, 4, 5, 6), denom3.bbbbbb, (1, 2, 5, 0, 4, 6), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), v.bbbb.ooov, (1, 3, 6, 0), v.bbbb.ovov, (2, 4, 6, 5), denom3.bbbbbb, (1, 2, 6, 0, 4, 5), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 0, 4), v.bbbb.ooov, (1, 3, 5, 6), v.bbbb.ovov, (2, 6, 5, 4), denom3.bbbbbb, (1, 2, 5, 0, 6, 4), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 0, 4), v.bbbb.ooov, (1, 3, 5, 6), v.bbbb.ovov, (5, 4, 2, 6), denom3.bbbbbb, (1, 5, 2, 0, 4, 6), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), v.bbbb.ooov, (1, 3, 6, 0), v.bbbb.ovov, (6, 4, 2, 5), denom3.bbbbbb, (1, 6, 2, 0, 4, 5), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 0, 4), v.bbbb.ooov, (1, 3, 5, 6), v.bbbb.ovov, (5, 6, 2, 4), denom3.bbbbbb, (1, 5, 2, 0, 6, 4), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 0, 4), v.bbbb.ooov, (5, 3, 1, 6), v.bbbb.ovov, (2, 4, 5, 6), denom3.bbbbbb, (1, 2, 5, 0, 4, 6), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), v.bbbb.ooov, (6, 3, 1, 0), v.bbbb.ovov, (2, 4, 6, 5), denom3.bbbbbb, (1, 2, 6, 0, 4, 5), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.aaaa, (2, 3, 4, 5), v.aabb.ooov, (6, 3, 1, 0), v.aaaa.ovov, (2, 4, 6, 5), denom3.abaaba, (2, 1, 6, 4, 0, 5), ()) * -2.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 0, 4), v.bbbb.ooov, (5, 3, 1, 6), v.bbbb.ovov, (2, 6, 5, 4), denom3.bbbbbb, (1, 2, 5, 0, 6, 4), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 0, 4), v.bbbb.ooov, (5, 3, 1, 6), v.bbbb.ovov, (5, 4, 2, 6), denom3.bbbbbb, (1, 5, 2, 0, 4, 6), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), v.bbbb.ooov, (6, 3, 1, 0), v.bbbb.ovov, (6, 4, 2, 5), denom3.bbbbbb, (1, 6, 2, 0, 4, 5), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.aaaa, (2, 3, 4, 5), v.aabb.ooov, (6, 3, 1, 0), v.aaaa.ovov, (6, 4, 2, 5), denom3.abaaba, (6, 1, 2, 4, 0, 5), ()) * 2.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 0, 4), v.bbbb.ooov, (5, 3, 1, 6), v.bbbb.ovov, (5, 6, 2, 4), denom3.bbbbbb, (1, 5, 2, 0, 6, 4), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 0, 3), v.aabb.ovoo, (4, 5, 6, 2), v.aabb.ovov, (4, 5, 6, 3), denom3.babbab, (1, 4, 6, 0, 5, 3), ()) * -4.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 0, 4), v.aabb.ovoo, (5, 6, 1, 3), v.aabb.ovov, (5, 6, 2, 4), denom3.babbab, (1, 5, 2, 0, 6, 4), ()) * 4.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 3, 4), v.bbbb.ovov, (2, 3, 5, 6), v.bbbb.ovvv, (5, 0, 6, 4), denom3.bbbbbb, (1, 2, 5, 0, 3, 6), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 3, 4), v.bbbb.ovov, (2, 3, 5, 6), v.bbbb.ovvv, (5, 6, 0, 4), denom3.bbbbbb, (1, 2, 5, 0, 3, 6), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 0, 3), v.bbbb.ovov, (2, 4, 5, 6), v.bbbb.ovvv, (5, 4, 6, 3), denom3.bbbbbb, (1, 2, 5, 0, 4, 6), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 0, 3), v.bbbb.ovov, (2, 4, 5, 6), v.bbbb.ovvv, (5, 6, 4, 3), denom3.bbbbbb, (1, 2, 5, 0, 4, 6), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 3, 4), v.bbbb.ovov, (2, 5, 6, 3), v.bbbb.ovvv, (6, 0, 5, 4), denom3.bbbbbb, (1, 2, 6, 0, 5, 3), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 3, 4), v.bbbb.ovov, (2, 5, 6, 3), v.bbbb.ovvv, (6, 5, 0, 4), denom3.bbbbbb, (1, 2, 6, 0, 5, 3), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 3, 4), v.bbbb.ovov, (5, 3, 2, 6), v.bbbb.ovvv, (5, 0, 6, 4), denom3.bbbbbb, (1, 5, 2, 0, 3, 6), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 3, 4), v.bbbb.ovov, (5, 3, 2, 6), v.bbbb.ovvv, (5, 6, 0, 4), denom3.bbbbbb, (1, 5, 2, 0, 3, 6), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 0, 3), v.aabb.ovov, (4, 5, 2, 6), v.aabb.ovvv, (4, 5, 6, 3), denom3.babbab, (1, 4, 2, 0, 5, 6), ()) * 4.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 0, 3), v.bbbb.ovov, (4, 5, 2, 6), v.bbbb.ovvv, (4, 5, 6, 3), denom3.bbbbbb, (1, 4, 2, 0, 5, 6), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 0, 3), v.bbbb.ovov, (4, 5, 2, 6), v.bbbb.ovvv, (4, 6, 5, 3), denom3.bbbbbb, (1, 4, 2, 0, 5, 6), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 3, 4), v.bbbb.ovov, (5, 6, 2, 3), v.bbbb.ovvv, (5, 0, 6, 4), denom3.bbbbbb, (1, 5, 2, 0, 6, 3), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 3, 4), v.aabb.ovov, (5, 6, 2, 3), v.aabb.ovvv, (5, 6, 0, 4), denom3.babbab, (1, 5, 2, 0, 6, 3), ()) * -4.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (1, 2, 3, 4), v.bbbb.ovov, (5, 6, 2, 3), v.bbbb.ovvv, (5, 6, 0, 4), denom3.bbbbbb, (1, 5, 2, 0, 6, 3), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), v.bbbb.ovov, (2, 4, 3, 6), v.bbbb.ovvv, (1, 0, 6, 5), denom3.bbbbbb, (1, 2, 3, 0, 4, 6), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), v.bbbb.ovov, (2, 4, 3, 6), v.bbbb.ovvv, (1, 6, 0, 5), denom3.bbbbbb, (1, 2, 3, 0, 4, 6), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 0, 4), v.bbbb.ovov, (2, 5, 3, 6), v.bbbb.ovvv, (1, 5, 6, 4), denom3.bbbbbb, (1, 2, 3, 0, 5, 6), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 0, 4), v.bbbb.ovov, (2, 5, 3, 6), v.bbbb.ovvv, (1, 6, 5, 4), denom3.bbbbbb, (1, 2, 3, 0, 5, 6), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), v.bbbb.ovov, (2, 6, 3, 4), v.bbbb.ovvv, (1, 0, 6, 5), denom3.bbbbbb, (1, 2, 3, 0, 6, 4), ()) * -6.0
    e_pert += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), v.bbbb.ovov, (2, 6, 3, 4), v.bbbb.ovvv, (1, 6, 0, 5), denom3.bbbbbb, (1, 2, 3, 0, 6, 4), ()) * 6.0
    e_pert += einsum(l1.bb, (0, 1), t2.aaaa, (2, 3, 4, 5), v.aaaa.ovov, (2, 4, 3, 6), v.aabb.vvov, (6, 5, 1, 0), denom3.abaaba, (2, 1, 3, 4, 0, 6), ()) * 2.0
    e_pert += einsum(l1.bb, (0, 1), t2.aaaa, (2, 3, 4, 5), v.aaaa.ovov, (2, 6, 3, 4), v.aabb.vvov, (6, 5, 1, 0), denom3.abaaba, (2, 1, 3, 6, 0, 4), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), v.bbbb.ooov, (3, 5, 6, 7), v.bbbb.ooov, (5, 4, 6, 7), denom3.babbab, (5, 2, 6, 1, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.bbbb.ooov, (3, 6, 7, 5), v.bbbb.ooov, (6, 4, 7, 1), denom3.babbab, (6, 2, 7, 1, 0, 5), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), v.bbbb.ooov, (3, 5, 6, 7), v.bbbb.ooov, (6, 4, 5, 7), denom3.babbab, (5, 2, 6, 1, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.bbbb.ooov, (3, 6, 7, 5), v.bbbb.ooov, (7, 4, 6, 1), denom3.babbab, (6, 2, 7, 1, 0, 5), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), v.aabb.ooov, (2, 4, 6, 7), v.bbbb.ooov, (3, 5, 6, 7), denom3.babbab, (3, 4, 6, 1, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ooov, (2, 4, 7, 6), v.bbbb.ooov, (3, 5, 7, 1), denom3.babbab, (3, 4, 7, 1, 0, 6), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), v.aabb.ooov, (2, 4, 6, 7), v.bbbb.ooov, (3, 5, 6, 7), denom3.babbab, (5, 2, 6, 1, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ooov, (2, 4, 7, 1), v.bbbb.ooov, (3, 5, 7, 6), denom3.babbab, (5, 2, 7, 1, 0, 6), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), v.aabb.ooov, (2, 4, 6, 7), v.bbbb.ooov, (3, 6, 5, 7), denom3.babbab, (6, 2, 5, 1, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ooov, (2, 4, 7, 1), v.bbbb.ooov, (3, 7, 5, 6), denom3.babbab, (7, 2, 5, 1, 0, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), v.aabb.ooov, (2, 5, 6, 7), v.aabb.ooov, (5, 4, 6, 7), denom3.babbab, (3, 5, 6, 1, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), v.aabb.ooov, (2, 6, 7, 5), v.aabb.ooov, (6, 4, 7, 1), denom3.babbab, (3, 6, 7, 1, 0, 5), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), v.aaaa.ooov, (2, 5, 6, 7), v.aaaa.ooov, (5, 4, 6, 7), denom3.abaaba, (5, 3, 6, 0, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aaaa.ooov, (2, 6, 7, 5), v.aaaa.ooov, (6, 4, 7, 0), denom3.abaaba, (6, 3, 7, 0, 1, 5), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), v.aaaa.ooov, (2, 5, 6, 7), v.aaaa.ooov, (6, 4, 5, 7), denom3.abaaba, (5, 3, 6, 0, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aaaa.ooov, (2, 6, 7, 5), v.aaaa.ooov, (7, 4, 6, 0), denom3.abaaba, (6, 3, 7, 0, 1, 5), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), v.aabb.ooov, (2, 4, 6, 7), v.bbbb.ooov, (6, 5, 3, 7), denom3.babbab, (3, 4, 6, 1, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ooov, (2, 4, 7, 6), v.bbbb.ooov, (7, 5, 3, 1), denom3.babbab, (3, 4, 7, 1, 0, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), v.aabb.ooov, (2, 6, 5, 7), v.aabb.ooov, (6, 4, 3, 7), denom3.babbab, (3, 6, 5, 1, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ooov, (2, 7, 5, 6), v.aabb.ooov, (7, 4, 3, 1), denom3.babbab, (3, 7, 5, 1, 0, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aaaa.ooov, (2, 4, 7, 6), v.aabb.ovoo, (7, 0, 3, 5), denom3.abaaba, (4, 3, 7, 0, 1, 6), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), v.aaaa.ooov, (2, 4, 6, 7), v.aabb.ovoo, (6, 7, 3, 5), denom3.abaaba, (2, 5, 6, 0, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), v.aaaa.ooov, (2, 4, 6, 7), v.aabb.ovoo, (6, 7, 3, 5), denom3.abaaba, (4, 3, 6, 0, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aaaa.ooov, (2, 4, 7, 0), v.aabb.ovoo, (7, 6, 3, 5), denom3.abaaba, (2, 5, 7, 0, 1, 6), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aaaa.ooov, (2, 7, 4, 6), v.aabb.ovoo, (7, 0, 3, 5), denom3.abaaba, (7, 3, 4, 0, 1, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), v.aaaa.ooov, (2, 6, 4, 7), v.aabb.ovoo, (6, 7, 3, 5), denom3.abaaba, (6, 3, 4, 0, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), v.aaaa.ooov, (6, 4, 2, 7), v.aabb.ovoo, (6, 7, 3, 5), denom3.abaaba, (2, 5, 6, 0, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aaaa.ooov, (7, 4, 2, 0), v.aabb.ovoo, (7, 6, 3, 5), denom3.abaaba, (2, 5, 7, 0, 1, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.bbbb.ooov, (3, 4, 6, 7), v.bbbb.ovvv, (6, 1, 7, 5), denom3.babbab, (4, 2, 6, 1, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.bbbb.ooov, (3, 4, 6, 7), v.bbbb.ovvv, (6, 5, 1, 7), denom3.babbab, (3, 2, 6, 5, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.bbbb.ooov, (3, 4, 6, 7), v.bbbb.ovvv, (6, 7, 1, 5), denom3.babbab, (3, 2, 6, 7, 0, 5), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.bbbb.ooov, (3, 4, 6, 7), v.bbbb.ovvv, (6, 7, 1, 5), denom3.babbab, (4, 2, 6, 1, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.bbbb.ooov, (3, 6, 4, 7), v.bbbb.ovvv, (6, 1, 7, 5), denom3.babbab, (6, 2, 4, 1, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.bbbb.ooov, (3, 6, 4, 7), v.bbbb.ovvv, (6, 7, 1, 5), denom3.babbab, (6, 2, 4, 1, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.bbbb.ooov, (6, 4, 3, 7), v.bbbb.ovvv, (6, 5, 1, 7), denom3.babbab, (3, 2, 6, 5, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.bbbb.ooov, (6, 4, 3, 7), v.bbbb.ovvv, (6, 7, 1, 5), denom3.babbab, (3, 2, 6, 7, 0, 5), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ooov, (2, 4, 5, 7), v.bbbb.ovvv, (3, 1, 7, 6), denom3.babbab, (3, 4, 5, 1, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ooov, (2, 4, 5, 7), v.bbbb.ovvv, (3, 7, 1, 6), denom3.babbab, (3, 4, 5, 1, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aaaa.ooov, (2, 4, 7, 5), v.aabb.ovvv, (7, 0, 1, 6), denom3.abaaba, (4, 3, 7, 0, 1, 5), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aaaa.ooov, (2, 4, 6, 7), v.aaaa.ovvv, (6, 0, 7, 5), denom3.abaaba, (4, 3, 6, 0, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), v.aabb.ooov, (2, 4, 6, 7), v.bbbb.ovvv, (6, 1, 7, 5), denom3.babbab, (3, 4, 6, 1, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), v.aabb.ooov, (2, 4, 6, 7), v.bbbb.ovvv, (6, 5, 1, 7), denom3.babbab, (3, 2, 6, 5, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aaaa.ooov, (2, 4, 6, 7), v.aaaa.ovvv, (6, 5, 0, 7), denom3.abaaba, (2, 3, 6, 5, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aaaa.ooov, (2, 4, 7, 0), v.aabb.ovvv, (7, 5, 1, 6), denom3.abaaba, (2, 3, 7, 0, 6, 5), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), v.aaaa.ooov, (2, 4, 6, 7), v.aabb.ovvv, (6, 7, 1, 5), denom3.abaaba, (2, 3, 6, 0, 5, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), v.aabb.ooov, (2, 4, 6, 7), v.bbbb.ovvv, (6, 7, 1, 5), denom3.babbab, (3, 2, 6, 7, 0, 5), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), v.aabb.ooov, (2, 4, 6, 7), v.bbbb.ovvv, (6, 7, 1, 5), denom3.babbab, (3, 4, 6, 1, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), v.aaaa.ooov, (2, 4, 6, 7), v.aabb.ovvv, (6, 7, 1, 5), denom3.abaaba, (4, 3, 6, 0, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aaaa.ooov, (2, 4, 6, 7), v.aaaa.ovvv, (6, 7, 0, 5), denom3.abaaba, (2, 3, 6, 7, 1, 5), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aaaa.ooov, (2, 4, 6, 7), v.aaaa.ovvv, (6, 7, 0, 5), denom3.abaaba, (4, 3, 6, 0, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aaaa.ooov, (2, 7, 4, 5), v.aabb.ovvv, (7, 0, 1, 6), denom3.abaaba, (7, 3, 4, 0, 1, 5), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aaaa.ooov, (2, 6, 4, 7), v.aaaa.ovvv, (6, 0, 7, 5), denom3.abaaba, (6, 3, 4, 0, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), v.aaaa.ooov, (2, 6, 4, 7), v.aabb.ovvv, (6, 7, 1, 5), denom3.abaaba, (6, 3, 4, 0, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aaaa.ooov, (2, 6, 4, 7), v.aaaa.ovvv, (6, 7, 0, 5), denom3.abaaba, (6, 3, 4, 0, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ooov, (2, 4, 3, 7), v.bbbb.ovvv, (5, 6, 1, 7), denom3.babbab, (3, 2, 5, 6, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ooov, (2, 4, 3, 7), v.bbbb.ovvv, (5, 7, 1, 6), denom3.babbab, (3, 2, 5, 7, 0, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aaaa.ooov, (6, 4, 2, 7), v.aaaa.ovvv, (6, 5, 0, 7), denom3.abaaba, (2, 3, 6, 5, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aaaa.ooov, (7, 4, 2, 0), v.aabb.ovvv, (7, 5, 1, 6), denom3.abaaba, (2, 3, 7, 0, 6, 5), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), v.aaaa.ooov, (6, 4, 2, 7), v.aabb.ovvv, (6, 7, 1, 5), denom3.abaaba, (2, 3, 6, 0, 5, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aaaa.ooov, (6, 4, 2, 7), v.aaaa.ovvv, (6, 7, 0, 5), denom3.abaaba, (2, 3, 6, 7, 1, 5), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), v.aabb.ovoo, (6, 0, 7, 4), v.aabb.ovoo, (6, 5, 3, 7), denom3.abaaba, (2, 7, 6, 0, 1, 5), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), v.aabb.ovoo, (5, 6, 3, 7), v.aabb.ovoo, (5, 6, 7, 4), denom3.abaaba, (2, 7, 5, 0, 1, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aabb.ovoo, (2, 0, 7, 5), v.aabb.ovoo, (4, 6, 3, 7), denom3.abaaba, (2, 7, 4, 0, 1, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), v.aabb.ovoo, (2, 6, 7, 5), v.aabb.ovoo, (4, 6, 3, 7), denom3.abaaba, (2, 7, 4, 0, 1, 6), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.aabb.ovoo, (7, 0, 3, 4), v.aabb.ovvv, (7, 5, 1, 6), denom3.abaaba, (2, 3, 7, 0, 6, 5), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.aabb.ovoo, (7, 5, 3, 4), v.aabb.ovvv, (7, 0, 1, 6), denom3.abaaba, (2, 4, 7, 0, 1, 5), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), v.aabb.ovoo, (6, 7, 3, 4), v.aaaa.ovvv, (6, 0, 7, 5), denom3.abaaba, (2, 4, 6, 0, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), v.aabb.ovoo, (6, 7, 3, 4), v.aaaa.ovvv, (6, 5, 0, 7), denom3.abaaba, (2, 3, 6, 5, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.aabb.ovoo, (6, 7, 3, 4), v.aabb.ovvv, (6, 7, 1, 5), denom3.abaaba, (2, 3, 6, 0, 5, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.aabb.ovoo, (6, 7, 3, 4), v.aabb.ovvv, (6, 7, 1, 5), denom3.abaaba, (2, 4, 6, 0, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), v.aabb.ovoo, (6, 7, 3, 4), v.aaaa.ovvv, (6, 7, 0, 5), denom3.abaaba, (2, 3, 6, 7, 1, 5), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), v.aabb.ovoo, (6, 7, 3, 4), v.aaaa.ovvv, (6, 7, 0, 5), denom3.abaaba, (2, 4, 6, 0, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), v.aabb.ovoo, (2, 0, 3, 5), v.aabb.ovvv, (4, 6, 1, 7), denom3.abaaba, (2, 3, 4, 0, 7, 6), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aabb.ovoo, (2, 7, 3, 5), v.aaaa.ovvv, (4, 6, 0, 7), denom3.abaaba, (2, 3, 4, 6, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ovoo, (2, 7, 3, 5), v.aabb.ovvv, (4, 7, 1, 6), denom3.abaaba, (2, 3, 4, 0, 6, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aabb.ovoo, (2, 7, 3, 5), v.aaaa.ovvv, (4, 7, 0, 6), denom3.abaaba, (2, 3, 4, 7, 1, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), v.aabb.ovoo, (4, 6, 3, 5), v.aabb.ovvv, (2, 0, 1, 7), denom3.abaaba, (2, 5, 4, 0, 1, 6), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aabb.ovoo, (4, 7, 3, 5), v.aaaa.ovvv, (2, 0, 7, 6), denom3.abaaba, (2, 5, 4, 0, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ovoo, (4, 7, 3, 5), v.aabb.ovvv, (2, 7, 1, 6), denom3.abaaba, (2, 5, 4, 0, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aabb.ovoo, (4, 7, 3, 5), v.aaaa.ovvv, (2, 7, 0, 6), denom3.abaaba, (2, 5, 4, 0, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.bbbb.ovvv, (3, 6, 7, 5), v.bbbb.ovvv, (4, 6, 1, 7), denom3.babbab, (3, 2, 4, 6, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.bbbb.ovvv, (3, 6, 7, 5), v.bbbb.ovvv, (4, 7, 1, 6), denom3.babbab, (3, 2, 4, 7, 0, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), v.aabb.ovvv, (6, 0, 7, 5), v.aabb.ovvv, (6, 4, 1, 7), denom3.abaaba, (2, 3, 6, 0, 7, 4), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), v.aaaa.ovvv, (6, 0, 7, 4), v.aabb.ovvv, (6, 7, 1, 5), denom3.abaaba, (2, 3, 6, 0, 5, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), v.aaaa.ovvv, (6, 4, 0, 7), v.aabb.ovvv, (6, 7, 1, 5), denom3.abaaba, (2, 3, 6, 4, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), v.aabb.ovvv, (5, 6, 1, 7), v.aabb.ovvv, (5, 6, 7, 4), denom3.abaaba, (2, 3, 5, 0, 7, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), v.bbbb.ovvv, (5, 6, 1, 7), v.bbbb.ovvv, (5, 6, 7, 4), denom3.babbab, (3, 2, 5, 6, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), v.aaaa.ovvv, (5, 6, 0, 7), v.aaaa.ovvv, (5, 6, 7, 4), denom3.abaaba, (2, 3, 5, 6, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), v.bbbb.ovvv, (5, 6, 1, 7), v.bbbb.ovvv, (5, 7, 6, 4), denom3.babbab, (3, 2, 5, 6, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), v.aaaa.ovvv, (5, 6, 0, 7), v.aaaa.ovvv, (5, 7, 6, 4), denom3.abaaba, (2, 3, 5, 6, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), v.aaaa.ovvv, (6, 7, 0, 4), v.aabb.ovvv, (6, 7, 1, 5), denom3.abaaba, (2, 3, 6, 0, 5, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), v.aaaa.ovvv, (6, 7, 0, 4), v.aabb.ovvv, (6, 7, 1, 5), denom3.abaaba, (2, 3, 6, 7, 1, 4), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aabb.ovvv, (2, 0, 7, 6), v.aabb.ovvv, (4, 5, 1, 7), denom3.abaaba, (2, 3, 4, 0, 7, 5), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aaaa.ovvv, (2, 0, 7, 5), v.aabb.ovvv, (4, 7, 1, 6), denom3.abaaba, (2, 3, 4, 0, 6, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), v.aabb.ovvv, (2, 6, 7, 5), v.aabb.ovvv, (4, 6, 1, 7), denom3.abaaba, (2, 3, 4, 0, 7, 6), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aaaa.ovvv, (2, 6, 7, 5), v.aaaa.ovvv, (4, 6, 0, 7), denom3.abaaba, (2, 3, 4, 6, 1, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aaaa.ovvv, (2, 6, 7, 5), v.aaaa.ovvv, (4, 7, 0, 6), denom3.abaaba, (2, 3, 4, 7, 1, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aabb.ovvv, (2, 7, 1, 6), v.aaaa.ovvv, (4, 5, 0, 7), denom3.abaaba, (2, 3, 4, 5, 1, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aaaa.ovvv, (2, 7, 0, 5), v.aabb.ovvv, (4, 7, 1, 6), denom3.abaaba, (2, 3, 4, 0, 6, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aabb.ovvv, (2, 7, 1, 6), v.aaaa.ovvv, (4, 7, 0, 5), denom3.abaaba, (2, 3, 4, 7, 1, 5), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), v.bbbb.ooov, (3, 4, 6, 7), v.aabb.vvov, (0, 5, 6, 7), denom3.babbab, (3, 2, 6, 1, 5, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.bbbb.ooov, (3, 4, 7, 1), v.aabb.vvov, (0, 5, 7, 6), denom3.babbab, (3, 2, 7, 1, 5, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), v.bbbb.ooov, (3, 4, 6, 7), v.aabb.vvov, (0, 5, 6, 7), denom3.babbab, (4, 2, 6, 1, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.bbbb.ooov, (3, 4, 7, 6), v.aabb.vvov, (0, 5, 7, 1), denom3.babbab, (4, 2, 7, 1, 0, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), v.bbbb.ooov, (3, 6, 4, 7), v.aabb.vvov, (0, 5, 6, 7), denom3.babbab, (6, 2, 4, 1, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.bbbb.ooov, (3, 7, 4, 6), v.aabb.vvov, (0, 5, 7, 1), denom3.babbab, (7, 2, 4, 1, 0, 6), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), v.bbbb.ooov, (6, 4, 3, 7), v.aabb.vvov, (0, 5, 6, 7), denom3.babbab, (3, 2, 6, 1, 5, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.bbbb.ooov, (7, 4, 3, 1), v.aabb.vvov, (0, 5, 7, 6), denom3.babbab, (3, 2, 7, 1, 5, 6), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aabb.ooov, (2, 4, 6, 7), v.aabb.vvov, (0, 5, 6, 7), denom3.babbab, (3, 2, 6, 1, 5, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), v.aabb.ooov, (2, 4, 6, 7), v.aabb.vvov, (0, 5, 6, 7), denom3.babbab, (3, 4, 6, 1, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aabb.ooov, (2, 4, 7, 1), v.aabb.vvov, (0, 5, 7, 6), denom3.babbab, (3, 2, 7, 1, 5, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aabb.ooov, (2, 4, 7, 6), v.aabb.vvov, (0, 5, 7, 1), denom3.babbab, (3, 4, 7, 1, 0, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aabb.ooov, (2, 4, 3, 7), v.aabb.vvov, (0, 6, 5, 7), denom3.babbab, (3, 2, 5, 1, 6, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), v.aabb.ooov, (2, 4, 3, 1), v.aabb.vvov, (0, 6, 5, 7), denom3.babbab, (3, 2, 5, 1, 6, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aabb.ooov, (2, 4, 5, 7), v.aabb.vvov, (0, 6, 3, 7), denom3.babbab, (3, 4, 5, 1, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), v.aabb.ooov, (2, 4, 5, 7), v.aabb.vvov, (0, 6, 3, 1), denom3.babbab, (3, 4, 5, 1, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.bbbb.ovvv, (3, 1, 7, 6), v.aabb.vvov, (0, 5, 4, 7), denom3.babbab, (3, 2, 4, 1, 5, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.bbbb.ovvv, (3, 7, 1, 6), v.aabb.vvov, (0, 5, 4, 7), denom3.babbab, (3, 2, 4, 1, 5, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), v.bbbb.ovvv, (6, 1, 7, 5), v.aabb.vvov, (0, 4, 6, 7), denom3.babbab, (3, 2, 6, 1, 4, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), v.bbbb.ovvv, (6, 5, 1, 7), v.aabb.vvov, (0, 4, 6, 7), denom3.babbab, (3, 2, 6, 5, 0, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), v.bbbb.ovvv, (6, 7, 1, 5), v.aabb.vvov, (0, 4, 6, 7), denom3.babbab, (3, 2, 6, 1, 4, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), v.bbbb.ovvv, (6, 7, 1, 5), v.aabb.vvov, (0, 4, 6, 7), denom3.babbab, (3, 2, 6, 7, 0, 5), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.bbbb.ovvv, (4, 6, 1, 7), v.aabb.vvov, (0, 5, 3, 7), denom3.babbab, (3, 2, 4, 6, 0, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.bbbb.ovvv, (4, 7, 1, 6), v.aabb.vvov, (0, 5, 3, 7), denom3.babbab, (3, 2, 4, 7, 0, 6), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), v.aabb.vvov, (0, 5, 6, 7), v.aabb.vvov, (5, 4, 6, 7), denom3.babbab, (3, 2, 6, 1, 5, 7), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), v.aabb.vvov, (0, 6, 4, 7), v.aabb.vvov, (6, 5, 3, 7), denom3.babbab, (3, 2, 4, 1, 6, 7), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), v.aabb.vvov, (0, 6, 7, 5), v.aabb.vvov, (6, 4, 7, 1), denom3.babbab, (3, 2, 7, 1, 6, 5), ()) * -2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.aabb.vvov, (0, 7, 4, 6), v.aabb.vvov, (7, 5, 3, 1), denom3.babbab, (3, 2, 4, 1, 7, 6), ()) * 2.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (2, 4, 7, 6), v.aabb.ooov, (7, 5, 3, 1), denom3.abaaba, (4, 3, 7, 0, 1, 6), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (2, 7, 4, 6), v.aabb.ooov, (7, 5, 3, 1), denom3.abaaba, (7, 3, 4, 0, 1, 6), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aabb.ooov, (6, 4, 7, 1), v.aabb.ovoo, (6, 5, 3, 7), denom3.abaaba, (2, 7, 6, 0, 1, 5), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (3, 4, 1, 5), v.aabb.ooov, (2, 6, 7, 5), v.aabb.ovoo, (6, 0, 7, 4), denom3.babbab, (3, 6, 7, 1, 0, 5), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aabb.ooov, (2, 5, 7, 1), v.aabb.ovoo, (4, 6, 3, 7), denom3.abaaba, (2, 7, 4, 0, 1, 6), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 6), v.aabb.ooov, (2, 7, 4, 6), v.aabb.ovoo, (7, 0, 3, 5), denom3.babbab, (3, 7, 4, 1, 0, 6), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 6), v.bbbb.ooov, (3, 4, 7, 6), v.aabb.ovoo, (2, 0, 7, 5), denom3.babbab, (4, 2, 7, 1, 0, 6), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 6), v.bbbb.ooov, (3, 7, 4, 6), v.aabb.ovoo, (2, 0, 7, 5), denom3.babbab, (7, 2, 4, 1, 0, 6), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aabb.ooov, (6, 4, 3, 7), v.aabb.ovvv, (6, 5, 1, 7), denom3.abaaba, (2, 3, 6, 0, 7, 5), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aabb.ooov, (7, 4, 3, 1), v.aaaa.ovvv, (7, 5, 0, 6), denom3.abaaba, (2, 3, 7, 5, 1, 6), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (3, 4, 1, 5), v.aabb.ooov, (2, 6, 4, 7), v.aabb.ovvv, (6, 0, 7, 5), denom3.babbab, (3, 6, 4, 1, 0, 7), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (3, 4, 5, 6), v.aabb.ooov, (2, 7, 4, 5), v.aabb.ovvv, (7, 0, 1, 6), denom3.babbab, (3, 7, 4, 1, 0, 5), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aabb.ooov, (2, 5, 3, 7), v.aabb.ovvv, (4, 6, 1, 7), denom3.abaaba, (2, 3, 4, 0, 7, 6), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), v.aabb.ooov, (2, 5, 3, 1), v.aaaa.ovvv, (4, 6, 0, 7), denom3.abaaba, (2, 3, 4, 6, 1, 7), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 6), v.bbbb.ooov, (3, 4, 5, 7), v.aabb.ovvv, (2, 0, 7, 6), denom3.babbab, (4, 2, 5, 1, 0, 7), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), v.bbbb.ooov, (3, 4, 5, 6), v.aabb.ovvv, (2, 0, 1, 7), denom3.babbab, (4, 2, 5, 1, 0, 6), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (3, 4, 5, 6), v.aabb.ovoo, (2, 0, 7, 4), v.bbbb.ovvv, (7, 5, 1, 6), denom3.babbab, (3, 2, 7, 5, 0, 6), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), v.aabb.ovoo, (2, 0, 3, 5), v.bbbb.ovvv, (4, 6, 1, 7), denom3.babbab, (3, 2, 4, 6, 0, 7), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (3, 4, 5, 6), v.aabb.ovvv, (2, 0, 7, 6), v.bbbb.ovvv, (4, 5, 1, 7), denom3.babbab, (3, 2, 4, 5, 0, 7), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (3, 4, 5, 6), v.aabb.ovvv, (2, 0, 7, 6), v.bbbb.ovvv, (4, 7, 1, 5), denom3.babbab, (3, 2, 4, 7, 0, 5), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), v.aaaa.ooov, (2, 4, 5, 6), v.aabb.vvov, (0, 7, 3, 1), denom3.abaaba, (4, 3, 5, 0, 1, 6), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (2, 4, 5, 7), v.aabb.vvov, (7, 6, 3, 1), denom3.abaaba, (4, 3, 5, 0, 1, 7), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aabb.ovoo, (4, 5, 3, 7), v.aabb.vvov, (0, 6, 7, 1), denom3.abaaba, (2, 7, 4, 0, 1, 5), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aabb.ovoo, (4, 6, 3, 7), v.aabb.vvov, (6, 5, 7, 1), denom3.abaaba, (2, 7, 4, 0, 1, 6), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (3, 4, 1, 5), v.aabb.ovoo, (2, 6, 7, 4), v.aabb.vvov, (0, 6, 7, 5), denom3.babbab, (3, 2, 7, 1, 6, 5), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 6), v.aabb.ovoo, (2, 7, 3, 5), v.aabb.vvov, (0, 7, 4, 6), denom3.babbab, (3, 2, 4, 1, 7, 6), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aabb.ovvv, (4, 5, 1, 7), v.aabb.vvov, (0, 6, 3, 7), denom3.abaaba, (2, 3, 4, 0, 7, 5), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aabb.ovvv, (4, 6, 1, 7), v.aabb.vvov, (6, 5, 3, 7), denom3.abaaba, (2, 3, 4, 0, 7, 6), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aaaa.ovvv, (4, 5, 0, 7), v.aabb.vvov, (7, 6, 3, 1), denom3.abaaba, (2, 3, 4, 5, 1, 7), ()) * -4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aaaa.ovvv, (4, 7, 0, 5), v.aabb.vvov, (7, 6, 3, 1), denom3.abaaba, (2, 3, 4, 7, 1, 5), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (3, 4, 1, 5), v.aabb.ovvv, (2, 6, 7, 5), v.aabb.vvov, (0, 6, 4, 7), denom3.babbab, (3, 2, 4, 1, 6, 7), ()) * 4.0
    e_pert += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (3, 4, 5, 6), v.aabb.ovvv, (2, 7, 1, 6), v.aabb.vvov, (0, 7, 4, 5), denom3.babbab, (3, 2, 4, 1, 7, 5), ()) * -4.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aaaa.ooov, (2, 4, 7, 1), v.aabb.ooov, (3, 7, 5, 6), denom3.abaaba, (2, 5, 7, 0, 6, 1), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 1, 6), v.aaaa.ooov, (2, 4, 7, 0), v.aabb.ooov, (3, 7, 5, 6), denom3.abaaba, (2, 5, 7, 0, 6, 1), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ooov, (3, 7, 5, 6), v.aaaa.ooov, (7, 4, 2, 1), denom3.abaaba, (2, 5, 7, 0, 6, 1), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 1, 6), v.aabb.ooov, (3, 7, 5, 6), v.aaaa.ooov, (7, 4, 2, 0), denom3.abaaba, (2, 5, 7, 0, 6, 1), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (2, 4, 1, 5), v.aabb.ooov, (3, 6, 7, 5), v.aabb.ovoo, (6, 0, 7, 4), denom3.abaaba, (2, 7, 6, 0, 5, 1), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.aabb.ooov, (3, 6, 7, 5), v.aabb.ovoo, (6, 1, 7, 4), denom3.abaaba, (2, 7, 6, 0, 5, 1), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 0), v.bbbb.ooov, (2, 5, 7, 1), v.aabb.ovoo, (4, 6, 3, 7), denom3.babbab, (2, 4, 7, 0, 6, 1), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.bbbb.ooov, (2, 5, 7, 0), v.aabb.ovoo, (4, 6, 3, 7), denom3.babbab, (2, 4, 7, 0, 6, 1), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 1, 6), v.aabb.ooov, (3, 4, 7, 6), v.aabb.ovoo, (2, 0, 7, 5), denom3.abaaba, (2, 7, 4, 0, 6, 1), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ooov, (3, 4, 7, 6), v.aabb.ovoo, (2, 1, 7, 5), denom3.abaaba, (2, 7, 4, 0, 6, 1), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 2, 5, 0), v.aabb.ooov, (6, 4, 7, 1), v.aabb.ovoo, (6, 5, 3, 7), denom3.babbab, (2, 6, 7, 0, 5, 1), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 2, 5, 1), v.aabb.ooov, (6, 4, 7, 0), v.aabb.ovoo, (6, 5, 3, 7), denom3.babbab, (2, 6, 7, 0, 5, 1), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 0), v.bbbb.ooov, (7, 5, 2, 1), v.aabb.ovoo, (4, 6, 3, 7), denom3.babbab, (2, 4, 7, 0, 6, 1), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.bbbb.ooov, (7, 5, 2, 0), v.aabb.ovoo, (4, 6, 3, 7), denom3.babbab, (2, 4, 7, 0, 6, 1), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 0), v.aabb.ooov, (7, 4, 2, 1), v.aabb.ovoo, (7, 6, 3, 5), denom3.babbab, (2, 7, 5, 0, 6, 1), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aabb.ooov, (7, 4, 2, 0), v.aabb.ovoo, (7, 6, 3, 5), denom3.babbab, (2, 7, 5, 0, 6, 1), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (2, 4, 1, 5), v.aabb.ooov, (3, 6, 4, 7), v.aabb.ovvv, (6, 0, 7, 5), denom3.abaaba, (2, 4, 6, 0, 7, 1), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.aabb.ooov, (3, 7, 4, 6), v.aaaa.ovvv, (7, 0, 1, 5), denom3.abaaba, (2, 4, 7, 0, 6, 1), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.aabb.ooov, (3, 6, 4, 7), v.aabb.ovvv, (6, 1, 7, 5), denom3.abaaba, (2, 4, 6, 0, 7, 1), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.aabb.ooov, (3, 7, 4, 6), v.aaaa.ovvv, (7, 1, 0, 5), denom3.abaaba, (2, 4, 7, 0, 6, 1), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 0), v.bbbb.ooov, (2, 5, 3, 7), v.aabb.ovvv, (4, 6, 1, 7), denom3.babbab, (2, 4, 3, 0, 6, 7), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), v.bbbb.ooov, (2, 5, 3, 0), v.aabb.ovvv, (4, 6, 1, 7), denom3.babbab, (2, 4, 3, 0, 6, 7), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 1, 6), v.aabb.ooov, (3, 4, 5, 7), v.aabb.ovvv, (2, 0, 7, 6), denom3.abaaba, (2, 5, 4, 0, 7, 1), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), v.aabb.ooov, (3, 4, 5, 7), v.aaaa.ovvv, (2, 0, 1, 6), denom3.abaaba, (2, 5, 4, 0, 7, 1), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aabb.ooov, (3, 4, 5, 7), v.aabb.ovvv, (2, 1, 7, 6), denom3.abaaba, (2, 5, 4, 0, 7, 1), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), v.aabb.ooov, (3, 4, 5, 7), v.aaaa.ovvv, (2, 1, 0, 6), denom3.abaaba, (2, 5, 4, 0, 7, 1), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 0), v.bbbb.ooov, (3, 5, 2, 7), v.aabb.ovvv, (4, 6, 1, 7), denom3.babbab, (2, 4, 3, 0, 6, 7), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), v.bbbb.ooov, (3, 5, 2, 0), v.aabb.ovvv, (4, 6, 1, 7), denom3.babbab, (2, 4, 3, 0, 6, 7), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 2, 5, 0), v.aabb.ooov, (6, 4, 3, 7), v.aabb.ovvv, (6, 5, 1, 7), denom3.babbab, (2, 6, 3, 0, 5, 7), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 2, 5, 6), v.aabb.ooov, (7, 4, 3, 0), v.aabb.ovvv, (7, 5, 1, 6), denom3.babbab, (2, 7, 3, 0, 5, 6), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 0), v.aabb.ooov, (6, 4, 2, 7), v.aabb.ovvv, (6, 5, 1, 7), denom3.babbab, (2, 6, 3, 0, 5, 7), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aabb.ooov, (7, 4, 2, 0), v.aabb.ovvv, (7, 5, 1, 6), denom3.babbab, (2, 7, 3, 0, 5, 6), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), v.aabb.ovoo, (4, 6, 3, 5), v.bbbb.ovvv, (2, 0, 1, 7), denom3.babbab, (2, 4, 5, 0, 6, 1), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), v.aabb.ovoo, (4, 6, 3, 5), v.bbbb.ovvv, (2, 1, 0, 7), denom3.babbab, (2, 4, 5, 0, 6, 1), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 2, 5, 6), v.aabb.ovoo, (4, 5, 3, 7), v.bbbb.ovvv, (7, 0, 1, 6), denom3.babbab, (2, 4, 7, 0, 5, 1), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 2, 5, 6), v.aabb.ovoo, (4, 5, 3, 7), v.bbbb.ovvv, (7, 1, 0, 6), denom3.babbab, (2, 4, 7, 0, 5, 1), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.bbbb.ovvv, (2, 0, 7, 6), v.aabb.ovvv, (4, 5, 1, 7), denom3.babbab, (2, 4, 3, 0, 5, 7), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.bbbb.ovvv, (2, 7, 0, 6), v.aabb.ovvv, (4, 5, 1, 7), denom3.babbab, (2, 4, 3, 0, 5, 7), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 2, 5, 6), v.bbbb.ovvv, (3, 0, 7, 6), v.aabb.ovvv, (4, 5, 1, 7), denom3.babbab, (2, 4, 3, 0, 5, 7), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 2, 5, 6), v.bbbb.ovvv, (3, 7, 0, 6), v.aabb.ovvv, (4, 5, 1, 7), denom3.babbab, (2, 4, 3, 0, 5, 7), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), v.aaaa.ooov, (2, 4, 3, 0), v.aabb.vvov, (1, 6, 5, 7), denom3.abaaba, (2, 5, 3, 0, 7, 6), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aaaa.ooov, (2, 4, 3, 7), v.aabb.vvov, (1, 7, 5, 6), denom3.abaaba, (2, 5, 3, 0, 6, 7), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), v.aaaa.ooov, (3, 4, 2, 0), v.aabb.vvov, (1, 6, 5, 7), denom3.abaaba, (2, 5, 3, 0, 7, 6), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (4, 5, 0, 6), v.aaaa.ooov, (3, 4, 2, 7), v.aabb.vvov, (1, 7, 5, 6), denom3.abaaba, (2, 5, 3, 0, 6, 7), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.aabb.ovoo, (3, 0, 7, 4), v.aabb.vvov, (1, 5, 7, 6), denom3.abaaba, (2, 7, 3, 0, 6, 5), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.aabb.ovoo, (3, 6, 7, 4), v.aabb.vvov, (1, 6, 7, 5), denom3.abaaba, (2, 7, 3, 0, 5, 6), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 5, 6), v.aabb.ovoo, (2, 0, 7, 4), v.aabb.vvov, (1, 5, 7, 6), denom3.abaaba, (2, 7, 3, 0, 6, 5), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 0, 5), v.aabb.ovoo, (2, 6, 7, 4), v.aabb.vvov, (1, 6, 7, 5), denom3.abaaba, (2, 7, 3, 0, 5, 6), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 2, 5, 0), v.aabb.ovoo, (4, 6, 3, 7), v.aabb.vvov, (6, 5, 7, 1), denom3.babbab, (2, 4, 7, 0, 6, 1), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 2, 5, 1), v.aabb.ovoo, (4, 6, 3, 7), v.aabb.vvov, (6, 5, 7, 0), denom3.babbab, (2, 4, 7, 0, 6, 1), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 0), v.aabb.ovoo, (4, 7, 3, 5), v.aabb.vvov, (7, 6, 2, 1), denom3.babbab, (2, 4, 5, 0, 7, 1), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 5, 6, 1), v.aabb.ovoo, (4, 7, 3, 5), v.aabb.vvov, (7, 6, 2, 0), denom3.babbab, (2, 4, 5, 0, 7, 1), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.aabb.ovvv, (3, 0, 7, 6), v.aabb.vvov, (1, 5, 4, 7), denom3.abaaba, (2, 4, 3, 0, 7, 5), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.aaaa.ovvv, (3, 0, 7, 5), v.aabb.vvov, (1, 7, 4, 6), denom3.abaaba, (2, 4, 3, 0, 6, 7), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), v.aabb.ovvv, (3, 6, 7, 5), v.aabb.vvov, (1, 6, 4, 7), denom3.abaaba, (2, 4, 3, 0, 7, 6), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (2, 4, 5, 6), v.aaaa.ovvv, (3, 7, 0, 5), v.aabb.vvov, (1, 7, 4, 6), denom3.abaaba, (2, 4, 3, 0, 6, 7), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 5, 6), v.aabb.ovvv, (2, 0, 7, 6), v.aabb.vvov, (1, 5, 4, 7), denom3.abaaba, (2, 4, 3, 0, 7, 5), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 5, 6), v.aaaa.ovvv, (2, 0, 7, 5), v.aabb.vvov, (1, 7, 4, 6), denom3.abaaba, (2, 4, 3, 0, 6, 7), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 0, 5), v.aabb.ovvv, (2, 6, 7, 5), v.aabb.vvov, (1, 6, 4, 7), denom3.abaaba, (2, 4, 3, 0, 7, 6), ()) * -2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 5, 6), v.aaaa.ovvv, (2, 7, 0, 5), v.aabb.vvov, (1, 7, 4, 6), denom3.abaaba, (2, 4, 3, 0, 6, 7), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 2, 5, 0), v.aabb.ovvv, (4, 6, 1, 7), v.aabb.vvov, (6, 5, 3, 7), denom3.babbab, (2, 4, 3, 0, 6, 7), ()) * 2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 0), v.aabb.ovvv, (4, 6, 1, 7), v.aabb.vvov, (6, 5, 2, 7), denom3.babbab, (2, 4, 3, 0, 6, 7), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 2, 5, 6), v.aabb.ovvv, (4, 7, 1, 6), v.aabb.vvov, (7, 5, 3, 0), denom3.babbab, (2, 4, 3, 0, 7, 6), ()) * -2.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 6), v.aabb.ovvv, (4, 7, 1, 6), v.aabb.vvov, (7, 5, 2, 0), denom3.babbab, (2, 4, 3, 0, 7, 6), ()) * 2.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 1), v.aaaa.ooov, (3, 5, 6, 7), v.aaaa.ooov, (5, 4, 6, 7), denom3.aaaaaa, (2, 5, 6, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 1), v.bbbb.ooov, (3, 5, 6, 7), v.bbbb.ooov, (5, 4, 6, 7), denom3.bbbbbb, (2, 5, 6, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 1), v.aabb.ooov, (3, 5, 6, 7), v.aabb.ooov, (5, 4, 6, 7), denom3.abaaba, (2, 6, 5, 0, 7, 1), ()) * 4.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aaaa.ooov, (3, 6, 7, 5), v.aaaa.ooov, (6, 4, 7, 1), denom3.aaaaaa, (2, 6, 7, 0, 1, 5), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.bbbb.ooov, (3, 6, 7, 5), v.bbbb.ooov, (6, 4, 7, 1), denom3.bbbbbb, (2, 6, 7, 0, 1, 5), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 1, 5), v.aaaa.ooov, (3, 6, 7, 5), v.aaaa.ooov, (6, 4, 7, 0), denom3.aaaaaa, (2, 6, 7, 0, 1, 5), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 1, 5), v.bbbb.ooov, (3, 6, 7, 5), v.bbbb.ooov, (6, 4, 7, 0), denom3.bbbbbb, (2, 6, 7, 0, 1, 5), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 1), v.aaaa.ooov, (3, 5, 6, 7), v.aaaa.ooov, (6, 4, 5, 7), denom3.aaaaaa, (2, 5, 6, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 1), v.bbbb.ooov, (3, 5, 6, 7), v.bbbb.ooov, (6, 4, 5, 7), denom3.bbbbbb, (2, 5, 6, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aaaa.ooov, (3, 6, 7, 5), v.aaaa.ooov, (7, 4, 6, 1), denom3.aaaaaa, (2, 6, 7, 0, 1, 5), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.bbbb.ooov, (3, 6, 7, 5), v.bbbb.ooov, (7, 4, 6, 1), denom3.bbbbbb, (2, 6, 7, 0, 1, 5), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 1, 5), v.aaaa.ooov, (3, 6, 7, 5), v.aaaa.ooov, (7, 4, 6, 0), denom3.aaaaaa, (2, 6, 7, 0, 1, 5), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 1, 5), v.bbbb.ooov, (3, 6, 7, 5), v.bbbb.ooov, (7, 4, 6, 0), denom3.bbbbbb, (2, 6, 7, 0, 1, 5), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), v.aaaa.ooov, (2, 5, 6, 7), v.aaaa.ooov, (3, 4, 6, 7), denom3.aaaaaa, (2, 4, 6, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), v.bbbb.ooov, (2, 5, 6, 7), v.bbbb.ooov, (3, 4, 6, 7), denom3.bbbbbb, (2, 4, 6, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), v.aabb.ooov, (2, 5, 6, 7), v.aabb.ooov, (3, 4, 6, 7), denom3.abaaba, (2, 6, 4, 0, 7, 1), ()) * -4.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (2, 5, 7, 1), v.aaaa.ooov, (3, 4, 7, 6), denom3.aaaaaa, (2, 4, 7, 0, 1, 6), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 6), v.bbbb.ooov, (2, 5, 7, 1), v.bbbb.ooov, (3, 4, 7, 6), denom3.bbbbbb, (2, 4, 7, 0, 1, 6), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 1, 6), v.aaaa.ooov, (2, 5, 7, 0), v.aaaa.ooov, (3, 4, 7, 6), denom3.aaaaaa, (2, 4, 7, 0, 1, 6), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 6), v.bbbb.ooov, (2, 5, 7, 0), v.bbbb.ooov, (3, 4, 7, 6), denom3.bbbbbb, (2, 4, 7, 0, 1, 6), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), v.aaaa.ooov, (2, 5, 6, 7), v.aaaa.ooov, (3, 6, 4, 7), denom3.aaaaaa, (2, 6, 4, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), v.bbbb.ooov, (2, 5, 6, 7), v.bbbb.ooov, (3, 6, 4, 7), denom3.bbbbbb, (2, 6, 4, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (2, 5, 7, 1), v.aaaa.ooov, (3, 7, 4, 6), denom3.aaaaaa, (2, 7, 4, 0, 1, 6), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 6), v.bbbb.ooov, (2, 5, 7, 1), v.bbbb.ooov, (3, 7, 4, 6), denom3.bbbbbb, (2, 7, 4, 0, 1, 6), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 1, 6), v.aaaa.ooov, (2, 5, 7, 0), v.aaaa.ooov, (3, 7, 4, 6), denom3.aaaaaa, (2, 7, 4, 0, 1, 6), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 6), v.bbbb.ooov, (2, 5, 7, 0), v.bbbb.ooov, (3, 7, 4, 6), denom3.bbbbbb, (2, 7, 4, 0, 1, 6), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), v.aaaa.ooov, (3, 4, 6, 7), v.aaaa.ooov, (6, 5, 2, 7), denom3.aaaaaa, (2, 4, 6, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), v.bbbb.ooov, (3, 4, 6, 7), v.bbbb.ooov, (6, 5, 2, 7), denom3.bbbbbb, (2, 4, 6, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (3, 4, 7, 6), v.aaaa.ooov, (7, 5, 2, 1), denom3.aaaaaa, (2, 4, 7, 0, 1, 6), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 6), v.bbbb.ooov, (3, 4, 7, 6), v.bbbb.ooov, (7, 5, 2, 1), denom3.bbbbbb, (2, 4, 7, 0, 1, 6), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 1, 6), v.aaaa.ooov, (3, 4, 7, 6), v.aaaa.ooov, (7, 5, 2, 0), denom3.aaaaaa, (2, 4, 7, 0, 1, 6), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 6), v.bbbb.ooov, (3, 4, 7, 6), v.bbbb.ooov, (7, 5, 2, 0), denom3.bbbbbb, (2, 4, 7, 0, 1, 6), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), v.aaaa.ooov, (3, 6, 4, 7), v.aaaa.ooov, (6, 5, 2, 7), denom3.aaaaaa, (2, 6, 4, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), v.bbbb.ooov, (3, 6, 4, 7), v.bbbb.ooov, (6, 5, 2, 7), denom3.bbbbbb, (2, 6, 4, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (3, 7, 4, 6), v.aaaa.ooov, (7, 5, 2, 1), denom3.aaaaaa, (2, 7, 4, 0, 1, 6), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 6), v.bbbb.ooov, (3, 7, 4, 6), v.bbbb.ooov, (7, 5, 2, 1), denom3.bbbbbb, (2, 7, 4, 0, 1, 6), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 1, 6), v.aaaa.ooov, (3, 7, 4, 6), v.aaaa.ooov, (7, 5, 2, 0), denom3.aaaaaa, (2, 7, 4, 0, 1, 6), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 6), v.bbbb.ooov, (3, 7, 4, 6), v.bbbb.ooov, (7, 5, 2, 0), denom3.bbbbbb, (2, 7, 4, 0, 1, 6), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 1, 5), v.aaaa.ooov, (3, 4, 6, 7), v.aaaa.ovvv, (6, 0, 7, 5), denom3.aaaaaa, (2, 4, 6, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 1, 5), v.bbbb.ooov, (3, 4, 6, 7), v.bbbb.ovvv, (6, 0, 7, 5), denom3.bbbbbb, (2, 4, 6, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aaaa.ooov, (3, 4, 7, 5), v.aaaa.ovvv, (7, 0, 1, 6), denom3.aaaaaa, (2, 4, 7, 0, 1, 5), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 5, 6), v.bbbb.ooov, (3, 4, 7, 5), v.bbbb.ovvv, (7, 0, 1, 6), denom3.bbbbbb, (2, 4, 7, 0, 1, 5), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aaaa.ooov, (3, 4, 6, 7), v.aaaa.ovvv, (6, 1, 7, 5), denom3.aaaaaa, (2, 4, 6, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.bbbb.ooov, (3, 4, 6, 7), v.bbbb.ovvv, (6, 1, 7, 5), denom3.bbbbbb, (2, 4, 6, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aaaa.ooov, (3, 4, 7, 5), v.aaaa.ovvv, (7, 1, 0, 6), denom3.aaaaaa, (2, 4, 7, 0, 1, 5), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 5, 6), v.bbbb.ooov, (3, 4, 7, 5), v.bbbb.ovvv, (7, 1, 0, 6), denom3.bbbbbb, (2, 4, 7, 0, 1, 5), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aaaa.ooov, (3, 4, 6, 7), v.aaaa.ovvv, (6, 5, 1, 7), denom3.aaaaaa, (2, 3, 6, 0, 5, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.bbbb.ooov, (3, 4, 6, 7), v.bbbb.ovvv, (6, 5, 1, 7), denom3.bbbbbb, (2, 3, 6, 0, 5, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aaaa.ooov, (3, 4, 7, 0), v.aaaa.ovvv, (7, 5, 1, 6), denom3.aaaaaa, (2, 3, 7, 0, 5, 6), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 5, 6), v.bbbb.ooov, (3, 4, 7, 0), v.bbbb.ovvv, (7, 5, 1, 6), denom3.bbbbbb, (2, 3, 7, 0, 5, 6), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aaaa.ooov, (3, 4, 6, 7), v.aaaa.ovvv, (6, 7, 1, 5), denom3.aaaaaa, (2, 3, 6, 0, 7, 5), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.bbbb.ooov, (3, 4, 6, 7), v.bbbb.ovvv, (6, 7, 1, 5), denom3.bbbbbb, (2, 3, 6, 0, 7, 5), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aaaa.ooov, (3, 4, 6, 7), v.aaaa.ovvv, (6, 7, 1, 5), denom3.aaaaaa, (2, 4, 6, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.bbbb.ooov, (3, 4, 6, 7), v.bbbb.ovvv, (6, 7, 1, 5), denom3.bbbbbb, (2, 4, 6, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 1, 5), v.aaaa.ooov, (3, 4, 6, 7), v.aaaa.ovvv, (6, 7, 0, 5), denom3.aaaaaa, (2, 4, 6, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 1, 5), v.bbbb.ooov, (3, 4, 6, 7), v.bbbb.ovvv, (6, 7, 0, 5), denom3.bbbbbb, (2, 4, 6, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 1, 5), v.aaaa.ooov, (3, 6, 4, 7), v.aaaa.ovvv, (6, 0, 7, 5), denom3.aaaaaa, (2, 6, 4, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 1, 5), v.bbbb.ooov, (3, 6, 4, 7), v.bbbb.ovvv, (6, 0, 7, 5), denom3.bbbbbb, (2, 6, 4, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aaaa.ooov, (3, 7, 4, 5), v.aaaa.ovvv, (7, 0, 1, 6), denom3.aaaaaa, (2, 7, 4, 0, 1, 5), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 5, 6), v.bbbb.ooov, (3, 7, 4, 5), v.bbbb.ovvv, (7, 0, 1, 6), denom3.bbbbbb, (2, 7, 4, 0, 1, 5), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aaaa.ooov, (3, 6, 4, 7), v.aaaa.ovvv, (6, 1, 7, 5), denom3.aaaaaa, (2, 6, 4, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.bbbb.ooov, (3, 6, 4, 7), v.bbbb.ovvv, (6, 1, 7, 5), denom3.bbbbbb, (2, 6, 4, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aaaa.ooov, (3, 7, 4, 5), v.aaaa.ovvv, (7, 1, 0, 6), denom3.aaaaaa, (2, 7, 4, 0, 1, 5), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 5, 6), v.bbbb.ooov, (3, 7, 4, 5), v.bbbb.ovvv, (7, 1, 0, 6), denom3.bbbbbb, (2, 7, 4, 0, 1, 5), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aaaa.ooov, (3, 6, 4, 7), v.aaaa.ovvv, (6, 7, 1, 5), denom3.aaaaaa, (2, 6, 4, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.bbbb.ooov, (3, 6, 4, 7), v.bbbb.ovvv, (6, 7, 1, 5), denom3.bbbbbb, (2, 6, 4, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 1, 5), v.aaaa.ooov, (3, 6, 4, 7), v.aaaa.ovvv, (6, 7, 0, 5), denom3.aaaaaa, (2, 6, 4, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 1, 5), v.bbbb.ooov, (3, 6, 4, 7), v.bbbb.ovvv, (6, 7, 0, 5), denom3.bbbbbb, (2, 6, 4, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aaaa.ooov, (6, 4, 3, 7), v.aaaa.ovvv, (6, 5, 1, 7), denom3.aaaaaa, (2, 3, 6, 0, 5, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.bbbb.ooov, (6, 4, 3, 7), v.bbbb.ovvv, (6, 5, 1, 7), denom3.bbbbbb, (2, 3, 6, 0, 5, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aaaa.ooov, (7, 4, 3, 0), v.aaaa.ovvv, (7, 5, 1, 6), denom3.aaaaaa, (2, 3, 7, 0, 5, 6), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 5, 6), v.bbbb.ooov, (7, 4, 3, 0), v.bbbb.ovvv, (7, 5, 1, 6), denom3.bbbbbb, (2, 3, 7, 0, 5, 6), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aaaa.ooov, (6, 4, 3, 7), v.aaaa.ovvv, (6, 7, 1, 5), denom3.aaaaaa, (2, 3, 6, 0, 7, 5), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.bbbb.ooov, (6, 4, 3, 7), v.bbbb.ovvv, (6, 7, 1, 5), denom3.bbbbbb, (2, 3, 6, 0, 7, 5), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 0, 5), v.aaaa.ooov, (2, 4, 6, 7), v.aaaa.ovvv, (6, 5, 1, 7), denom3.aaaaaa, (2, 3, 6, 0, 5, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 0, 5), v.bbbb.ooov, (2, 4, 6, 7), v.bbbb.ovvv, (6, 5, 1, 7), denom3.bbbbbb, (2, 3, 6, 0, 5, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 5, 6), v.aaaa.ooov, (2, 4, 7, 0), v.aaaa.ovvv, (7, 5, 1, 6), denom3.aaaaaa, (2, 3, 7, 0, 5, 6), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 5, 6), v.bbbb.ooov, (2, 4, 7, 0), v.bbbb.ovvv, (7, 5, 1, 6), denom3.bbbbbb, (2, 3, 7, 0, 5, 6), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 0, 5), v.aaaa.ooov, (2, 4, 6, 7), v.aaaa.ovvv, (6, 7, 1, 5), denom3.aaaaaa, (2, 3, 6, 0, 7, 5), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 0, 5), v.bbbb.ooov, (2, 4, 6, 7), v.bbbb.ovvv, (6, 7, 1, 5), denom3.bbbbbb, (2, 3, 6, 0, 7, 5), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 0, 5), v.aaaa.ooov, (6, 4, 2, 7), v.aaaa.ovvv, (6, 5, 1, 7), denom3.aaaaaa, (2, 3, 6, 0, 5, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 0, 5), v.bbbb.ooov, (6, 4, 2, 7), v.bbbb.ovvv, (6, 5, 1, 7), denom3.bbbbbb, (2, 3, 6, 0, 5, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 5, 6), v.aaaa.ooov, (7, 4, 2, 0), v.aaaa.ovvv, (7, 5, 1, 6), denom3.aaaaaa, (2, 3, 7, 0, 5, 6), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 5, 6), v.bbbb.ooov, (7, 4, 2, 0), v.bbbb.ovvv, (7, 5, 1, 6), denom3.bbbbbb, (2, 3, 7, 0, 5, 6), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 0, 5), v.aaaa.ooov, (6, 4, 2, 7), v.aaaa.ovvv, (6, 7, 1, 5), denom3.aaaaaa, (2, 3, 6, 0, 7, 5), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 0, 5), v.bbbb.ooov, (6, 4, 2, 7), v.bbbb.ovvv, (6, 7, 1, 5), denom3.bbbbbb, (2, 3, 6, 0, 7, 5), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (2, 5, 3, 7), v.aaaa.ovvv, (4, 6, 1, 7), denom3.aaaaaa, (2, 3, 4, 0, 6, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 6), v.bbbb.ooov, (2, 5, 3, 7), v.bbbb.ovvv, (4, 6, 1, 7), denom3.bbbbbb, (2, 3, 4, 0, 6, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), v.aaaa.ooov, (2, 5, 3, 0), v.aaaa.ovvv, (4, 6, 1, 7), denom3.aaaaaa, (2, 3, 4, 0, 6, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), v.bbbb.ooov, (2, 5, 3, 0), v.bbbb.ovvv, (4, 6, 1, 7), denom3.bbbbbb, (2, 3, 4, 0, 6, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (2, 5, 3, 7), v.aaaa.ovvv, (4, 7, 1, 6), denom3.aaaaaa, (2, 3, 4, 0, 7, 6), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 6), v.bbbb.ooov, (2, 5, 3, 7), v.bbbb.ovvv, (4, 7, 1, 6), denom3.bbbbbb, (2, 3, 4, 0, 7, 6), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 1, 6), v.aaaa.ooov, (3, 4, 5, 7), v.aaaa.ovvv, (2, 0, 7, 6), denom3.aaaaaa, (2, 4, 5, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 6), v.bbbb.ooov, (3, 4, 5, 7), v.bbbb.ovvv, (2, 0, 7, 6), denom3.bbbbbb, (2, 4, 5, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), v.aaaa.ooov, (3, 4, 5, 6), v.aaaa.ovvv, (2, 0, 1, 7), denom3.aaaaaa, (2, 4, 5, 0, 1, 6), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), v.bbbb.ooov, (3, 4, 5, 6), v.bbbb.ovvv, (2, 0, 1, 7), denom3.bbbbbb, (2, 4, 5, 0, 1, 6), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (3, 4, 5, 7), v.aaaa.ovvv, (2, 1, 7, 6), denom3.aaaaaa, (2, 4, 5, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 6), v.bbbb.ooov, (3, 4, 5, 7), v.bbbb.ovvv, (2, 1, 7, 6), denom3.bbbbbb, (2, 4, 5, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), v.aaaa.ooov, (3, 4, 5, 6), v.aaaa.ovvv, (2, 1, 0, 7), denom3.aaaaaa, (2, 4, 5, 0, 1, 6), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), v.bbbb.ooov, (3, 4, 5, 6), v.bbbb.ovvv, (2, 1, 0, 7), denom3.bbbbbb, (2, 4, 5, 0, 1, 6), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (3, 4, 5, 7), v.aaaa.ovvv, (2, 7, 1, 6), denom3.aaaaaa, (2, 4, 5, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 6), v.bbbb.ooov, (3, 4, 5, 7), v.bbbb.ovvv, (2, 7, 1, 6), denom3.bbbbbb, (2, 4, 5, 0, 1, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 1, 6), v.aaaa.ooov, (3, 4, 5, 7), v.aaaa.ovvv, (2, 7, 0, 6), denom3.aaaaaa, (2, 4, 5, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 1, 6), v.bbbb.ooov, (3, 4, 5, 7), v.bbbb.ovvv, (2, 7, 0, 6), denom3.bbbbbb, (2, 4, 5, 0, 1, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (3, 5, 2, 7), v.aaaa.ovvv, (4, 6, 1, 7), denom3.aaaaaa, (2, 3, 4, 0, 6, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 6), v.bbbb.ooov, (3, 5, 2, 7), v.bbbb.ovvv, (4, 6, 1, 7), denom3.bbbbbb, (2, 3, 4, 0, 6, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), v.aaaa.ooov, (3, 5, 2, 0), v.aaaa.ovvv, (4, 6, 1, 7), denom3.aaaaaa, (2, 3, 4, 0, 6, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), v.bbbb.ooov, (3, 5, 2, 0), v.bbbb.ovvv, (4, 6, 1, 7), denom3.bbbbbb, (2, 3, 4, 0, 6, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 6), v.aaaa.ooov, (3, 5, 2, 7), v.aaaa.ovvv, (4, 7, 1, 6), denom3.aaaaaa, (2, 3, 4, 0, 7, 6), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 6), v.bbbb.ooov, (3, 5, 2, 7), v.bbbb.ovvv, (4, 7, 1, 6), denom3.bbbbbb, (2, 3, 4, 0, 7, 6), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 1), v.aabb.ovoo, (5, 6, 3, 7), v.aabb.ovoo, (5, 6, 7, 4), denom3.babbab, (2, 5, 7, 0, 6, 1), ()) * 4.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), v.aabb.ovoo, (6, 7, 2, 5), v.aabb.ovoo, (6, 7, 3, 4), denom3.babbab, (2, 6, 4, 0, 7, 1), ()) * -4.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.aabb.ovoo, (6, 7, 3, 4), v.aabb.ovvv, (6, 7, 1, 5), denom3.babbab, (2, 6, 3, 0, 7, 5), ()) * -4.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.aabb.ovoo, (6, 7, 3, 4), v.aabb.ovvv, (6, 7, 1, 5), denom3.babbab, (2, 6, 4, 0, 7, 1), ()) * -4.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 1, 5), v.aabb.ovoo, (6, 7, 3, 4), v.aabb.ovvv, (6, 7, 0, 5), denom3.babbab, (2, 6, 4, 0, 7, 1), ()) * 4.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 0, 5), v.aabb.ovoo, (6, 7, 2, 4), v.aabb.ovvv, (6, 7, 1, 5), denom3.babbab, (2, 6, 3, 0, 7, 5), ()) * 4.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aaaa.ovvv, (3, 0, 7, 6), v.aaaa.ovvv, (4, 5, 1, 7), denom3.aaaaaa, (2, 3, 4, 0, 5, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 5, 6), v.bbbb.ovvv, (3, 0, 7, 6), v.bbbb.ovvv, (4, 5, 1, 7), denom3.bbbbbb, (2, 3, 4, 0, 5, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aaaa.ovvv, (3, 0, 7, 6), v.aaaa.ovvv, (4, 7, 1, 5), denom3.aaaaaa, (2, 3, 4, 0, 7, 5), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 5, 6), v.bbbb.ovvv, (3, 0, 7, 6), v.bbbb.ovvv, (4, 7, 1, 5), denom3.bbbbbb, (2, 3, 4, 0, 7, 5), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aaaa.ovvv, (3, 6, 7, 5), v.aaaa.ovvv, (4, 6, 1, 7), denom3.aaaaaa, (2, 3, 4, 0, 6, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.bbbb.ovvv, (3, 6, 7, 5), v.bbbb.ovvv, (4, 6, 1, 7), denom3.bbbbbb, (2, 3, 4, 0, 6, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aaaa.ovvv, (3, 6, 7, 5), v.aaaa.ovvv, (4, 7, 1, 6), denom3.aaaaaa, (2, 3, 4, 0, 7, 6), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 0, 5), v.bbbb.ovvv, (3, 6, 7, 5), v.bbbb.ovvv, (4, 7, 1, 6), denom3.bbbbbb, (2, 3, 4, 0, 7, 6), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aaaa.ovvv, (3, 7, 0, 6), v.aaaa.ovvv, (4, 5, 1, 7), denom3.aaaaaa, (2, 3, 4, 0, 5, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 5, 6), v.bbbb.ovvv, (3, 7, 0, 6), v.bbbb.ovvv, (4, 5, 1, 7), denom3.bbbbbb, (2, 3, 4, 0, 5, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 5, 6), v.aaaa.ovvv, (3, 7, 0, 6), v.aaaa.ovvv, (4, 7, 1, 5), denom3.aaaaaa, (2, 3, 4, 0, 7, 5), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 4, 5, 6), v.bbbb.ovvv, (3, 7, 0, 6), v.bbbb.ovvv, (4, 7, 1, 5), denom3.bbbbbb, (2, 3, 4, 0, 7, 5), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 5), v.aaaa.ovvv, (6, 0, 7, 5), v.aaaa.ovvv, (6, 4, 1, 7), denom3.aaaaaa, (2, 3, 6, 0, 4, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 5), v.bbbb.ovvv, (6, 0, 7, 5), v.bbbb.ovvv, (6, 4, 1, 7), denom3.bbbbbb, (2, 3, 6, 0, 4, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 5), v.aaaa.ovvv, (6, 0, 7, 5), v.aaaa.ovvv, (6, 7, 1, 4), denom3.aaaaaa, (2, 3, 6, 0, 7, 4), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 5), v.bbbb.ovvv, (6, 0, 7, 5), v.bbbb.ovvv, (6, 7, 1, 4), denom3.bbbbbb, (2, 3, 6, 0, 7, 4), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 5), v.aaaa.ovvv, (6, 4, 1, 7), v.aaaa.ovvv, (6, 7, 0, 5), denom3.aaaaaa, (2, 3, 6, 0, 4, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 5), v.bbbb.ovvv, (6, 4, 1, 7), v.bbbb.ovvv, (6, 7, 0, 5), denom3.bbbbbb, (2, 3, 6, 0, 4, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 0, 4), v.aaaa.ovvv, (5, 6, 1, 7), v.aaaa.ovvv, (5, 6, 7, 4), denom3.aaaaaa, (2, 3, 5, 0, 6, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 0, 4), v.bbbb.ovvv, (5, 6, 1, 7), v.bbbb.ovvv, (5, 6, 7, 4), denom3.bbbbbb, (2, 3, 5, 0, 6, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 0, 4), v.aabb.ovvv, (5, 6, 1, 7), v.aabb.ovvv, (5, 6, 7, 4), denom3.babbab, (2, 5, 3, 0, 6, 7), ()) * 4.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 0, 4), v.aaaa.ovvv, (5, 6, 1, 7), v.aaaa.ovvv, (5, 7, 6, 4), denom3.aaaaaa, (2, 3, 5, 0, 6, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 0, 4), v.bbbb.ovvv, (5, 6, 1, 7), v.bbbb.ovvv, (5, 7, 6, 4), denom3.bbbbbb, (2, 3, 5, 0, 6, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 5), v.aaaa.ovvv, (6, 7, 0, 5), v.aaaa.ovvv, (6, 7, 1, 4), denom3.aaaaaa, (2, 3, 6, 0, 7, 4), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 5), v.bbbb.ovvv, (6, 7, 0, 5), v.bbbb.ovvv, (6, 7, 1, 4), denom3.bbbbbb, (2, 3, 6, 0, 7, 4), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 5), v.aabb.ovvv, (6, 7, 0, 5), v.aabb.ovvv, (6, 7, 1, 4), denom3.babbab, (2, 6, 3, 0, 7, 4), ()) * -4.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 5, 6), v.aaaa.ovvv, (2, 0, 7, 6), v.aaaa.ovvv, (4, 5, 1, 7), denom3.aaaaaa, (2, 3, 4, 0, 5, 7), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 5, 6), v.bbbb.ovvv, (2, 0, 7, 6), v.bbbb.ovvv, (4, 5, 1, 7), denom3.bbbbbb, (2, 3, 4, 0, 5, 7), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 5, 6), v.aaaa.ovvv, (2, 0, 7, 6), v.aaaa.ovvv, (4, 7, 1, 5), denom3.aaaaaa, (2, 3, 4, 0, 7, 5), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 5, 6), v.bbbb.ovvv, (2, 0, 7, 6), v.bbbb.ovvv, (4, 7, 1, 5), denom3.bbbbbb, (2, 3, 4, 0, 7, 5), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 0, 5), v.aaaa.ovvv, (2, 6, 7, 5), v.aaaa.ovvv, (4, 6, 1, 7), denom3.aaaaaa, (2, 3, 4, 0, 6, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 0, 5), v.bbbb.ovvv, (2, 6, 7, 5), v.bbbb.ovvv, (4, 6, 1, 7), denom3.bbbbbb, (2, 3, 4, 0, 6, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 0, 5), v.aaaa.ovvv, (2, 6, 7, 5), v.aaaa.ovvv, (4, 7, 1, 6), denom3.aaaaaa, (2, 3, 4, 0, 7, 6), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 0, 5), v.bbbb.ovvv, (2, 6, 7, 5), v.bbbb.ovvv, (4, 7, 1, 6), denom3.bbbbbb, (2, 3, 4, 0, 7, 6), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 5, 6), v.aaaa.ovvv, (2, 7, 0, 6), v.aaaa.ovvv, (4, 5, 1, 7), denom3.aaaaaa, (2, 3, 4, 0, 5, 7), ()) * 12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 5, 6), v.bbbb.ovvv, (2, 7, 0, 6), v.bbbb.ovvv, (4, 5, 1, 7), denom3.bbbbbb, (2, 3, 4, 0, 5, 7), ()) * 12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 5, 6), v.aaaa.ovvv, (2, 7, 0, 6), v.aaaa.ovvv, (4, 7, 1, 5), denom3.aaaaaa, (2, 3, 4, 0, 7, 5), ()) * -12.0
    e_pert += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (3, 4, 5, 6), v.bbbb.ovvv, (2, 7, 0, 6), v.bbbb.ovvv, (4, 7, 1, 5), denom3.bbbbbb, (2, 3, 4, 0, 7, 5), ()) * -12.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 1, 5), v.aabb.ooov, (3, 4, 6, 7), v.aabb.vvov, (0, 5, 6, 7), denom3.abaaba, (2, 6, 4, 0, 7, 1), ()) * 4.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aabb.ooov, (3, 4, 6, 7), v.aabb.vvov, (1, 5, 6, 7), denom3.abaaba, (2, 6, 3, 0, 7, 5), ()) * -4.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 4, 0, 5), v.aabb.ooov, (3, 4, 6, 7), v.aabb.vvov, (1, 5, 6, 7), denom3.abaaba, (2, 6, 4, 0, 7, 1), ()) * -4.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (3, 4, 0, 5), v.aabb.ooov, (2, 4, 6, 7), v.aabb.vvov, (1, 5, 6, 7), denom3.abaaba, (2, 6, 3, 0, 7, 5), ()) * 4.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 5), v.aabb.vvov, (0, 5, 6, 7), v.aabb.vvov, (1, 4, 6, 7), denom3.abaaba, (2, 6, 3, 0, 7, 4), ()) * -4.0
    e_pert += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 0, 4), v.aabb.vvov, (1, 5, 6, 7), v.aabb.vvov, (5, 4, 6, 7), denom3.abaaba, (2, 6, 3, 0, 7, 5), ()) * 4.0
    e_pert /= 2

    return e_pert

