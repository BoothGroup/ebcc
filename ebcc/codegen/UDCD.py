# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 2), ()) * -1.0
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    t2new = Namespace()

    # T amplitudes
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    t1new_aa += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 1))
    t1new_aa += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 1))
    t1new_aa += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 3)) * -1.0
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 2, 5, 3), (0, 4)) * 2.0
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvoo, (4, 2, 5, 1), (0, 4)) * -1.0
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 5)) * -1.0
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 3), (0, 4))
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 2, 3, 5), (0, 4)) * 2.0
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), v.aabb.oovv, (0, 4, 5, 3), (1, 5)) * -1.0
    t1new_bb += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 1))
    t1new_bb += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 3)) * -1.0
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 3), (1, 5))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 5)) * -1.0
    x0 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x0 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 2))
    x0 += einsum(t2.abab, (0, 1, 2, 3), (0, 2)) * 0.5
    x1 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1 += einsum(x0, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (0, 2, 3, 4)) * 2.0
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x1, (4, 0, 1, 3), (4, 2)) * -1.0
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x2 += einsum(x0, (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3)) * 2.0
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), x2, (4, 1, 0, 3), (4, 2)) * -2.0
    del x2
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x3 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x3 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x4 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x4 += einsum(x0, (0, 1), x3, (0, 2, 3, 1), (2, 3)) * 2.0
    x5 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x5 += einsum(t2.abab, (0, 1, 2, 3), (1, 3))
    x5 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 2)) * 2.0
    x6 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x6 += einsum(x5, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    x7 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x7 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 1))
    x7 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 1))
    x7 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 2)) * -1.0
    x7 += einsum(x4, (0, 1), (0, 1)) * -1.0
    del x4
    x7 += einsum(x6, (0, 1), (0, 1))
    del x6
    t1new_aa += einsum(x7, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_bb += einsum(x7, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    del x7
    x8 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x8 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x8 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x9 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x9 += einsum(x5, (0, 1), x8, (0, 2, 1, 3), (2, 3))
    x10 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x10 += einsum(x0, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3)) * 2.0
    del x0
    t1new_bb += einsum(x10, (0, 1), (0, 1))
    x11 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x11 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    x11 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 1))
    x11 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 2)) * -1.0
    x11 += einsum(x9, (0, 1), (0, 1)) * -1.0
    del x9
    x11 += einsum(x10, (0, 1), (0, 1))
    del x10
    t1new_aa += einsum(x11, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    t1new_bb += einsum(x11, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    del x11
    x12 = np.zeros((nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x12 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2))
    x12 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3)) * -1.0
    x12 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 5)) * -1.0
    x12 += einsum(t2.aaaa, (0, 1, 2, 3), x3, (1, 4, 5, 3), (0, 4, 5)) * 2.0
    del x3
    t1new_aa += einsum(x12, (0, 1, 2), t2.aaaa, (3, 1, 4, 2), (0, 4)) * 2.0
    del x12
    x13 = np.zeros((nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x13 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 3)) * -1.0
    x13 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ovov, (1, 3, 4, 5), (0, 4, 5)) * -2.0
    x13 += einsum(t2.abab, (0, 1, 2, 3), x8, (1, 4, 3, 5), (0, 4, 5))
    del x8
    t1new_aa += einsum(x13, (0, 1, 2), t2.abab, (3, 1, 4, 2), (0, 4))
    del x13
    x14 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x14 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 2)) * 2.0
    x14 += einsum(t2.abab, (0, 1, 2, 3), (0, 2))
    x15 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x15 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x15 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_aa += einsum(x14, (0, 1), x15, (0, 2, 1, 3), (2, 3)) * -1.0
    del x15
    x16 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x16 += einsum(t2.abab, (0, 1, 2, 3), (1, 3)) * 0.5
    x16 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 2))
    t1new_aa += einsum(x16, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3)) * 2.0
    del x16
    x17 = np.zeros((nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x17 += einsum(v.aabb.oovv, (0, 1, 2, 3), (0, 1, 2))
    x17 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 5), (4, 0, 5)) * -1.0
    t1new_aa += einsum(x17, (0, 1, 2), t2.abab, (0, 3, 4, 2), (1, 4)) * -1.0
    del x17
    x18 = np.zeros((nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x18 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 2, 3))
    x18 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 1, 3)) * -1.0
    t1new_aa += einsum(x18, (0, 1, 2), t2.aaaa, (3, 0, 4, 1), (3, 2)) * 2.0
    del x18
    x19 = np.zeros((nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x19 += einsum(v.aabb.oooo, (0, 1, 2, 3), (0, 1, 2))
    x19 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (4, 0, 5))
    t1new_aa += einsum(x19, (0, 1, 2), t2.abab, (0, 2, 3, 4), (1, 3))
    del x19
    x20 = np.zeros((nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x20 += einsum(v.aaaa.oooo, (0, 1, 1, 2), (0, 1, 2))
    x20 += einsum(t2.aaaa, (0, 0, 1, 2), v.aaaa.ovov, (3, 2, 4, 1), (3, 0, 4)) * -1.0
    t1new_aa += einsum(x20, (0, 1, 2), t2.aaaa, (2, 0, 3, 4), (1, 3)) * -2.0
    del x20
    x21 = np.zeros((nvir[1],), dtype=np.float64)
    x21 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (4,))
    x22 = np.zeros((nvir[1],), dtype=np.float64)
    x22 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 2, 1, 4), (4,)) * -1.0
    x23 = np.zeros((nvir[1],), dtype=np.float64)
    x23 += einsum(x21, (0,), (0,)) * -0.5
    del x21
    x23 += einsum(x22, (0,), (0,))
    del x22
    t1new_aa += einsum(x23, (0,), t2.abab, (1, 2, 3, 0), (1, 3)) * 2.0
    t1new_bb += einsum(x23, (0,), t2.bbbb, (1, 2, 3, 0), (1, 3)) * 4.0
    del x23
    x24 = np.zeros((nvir[0],), dtype=np.float64)
    x24 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 4, 1, 2), (4,))
    x25 = np.zeros((nvir[0],), dtype=np.float64)
    x25 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (4,)) * -1.0
    x26 = np.zeros((nvir[0],), dtype=np.float64)
    x26 += einsum(x24, (0,), (0,)) * 2.0
    x26 += einsum(x25, (0,), (0,))
    t1new_aa += einsum(x26, (0,), t2.aaaa, (1, 2, 3, 0), (1, 3)) * 2.0
    del x26
    x27 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x27 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x27 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4)) * 0.5
    t1new_aa += einsum(x14, (0, 1), x27, (2, 0), (2, 1)) * -2.0
    del x14, x27
    x28 = np.zeros((nocc[1],), dtype=np.float64)
    x28 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (4,))
    x29 = np.zeros((nocc[1],), dtype=np.float64)
    x29 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 0, 2), (4,)) * -1.0
    x30 = np.zeros((nocc[1],), dtype=np.float64)
    x30 += einsum(x28, (0,), (0,)) * -0.5
    del x28
    x30 += einsum(x29, (0,), (0,))
    del x29
    t1new_aa += einsum(x30, (0,), t2.abab, (1, 0, 2, 3), (1, 2)) * 2.0
    t1new_bb += einsum(x30, (0,), t2.bbbb, (1, 0, 2, 3), (1, 2)) * 4.0
    del x30
    x31 = np.zeros((nocc[0],), dtype=np.float64)
    x31 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 0, 2), (4,)) * -1.0
    x32 = np.zeros((nocc[0],), dtype=np.float64)
    x32 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (4,)) * -1.0
    x33 = np.zeros((nocc[0],), dtype=np.float64)
    x33 += einsum(x31, (0,), (0,)) * 2.0
    x33 += einsum(x32, (0,), (0,))
    t1new_aa += einsum(x33, (0,), t2.aaaa, (1, 0, 2, 3), (1, 2)) * 2.0
    del x33
    x34 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x34 += einsum(x5, (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), x34, (4, 1, 0, 3), (4, 2)) * -2.0
    del x34
    x35 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x35 += einsum(x5, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (2, 0, 4, 3))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x35, (0, 4, 1, 2), (4, 3)) * -1.0
    del x35
    x36 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x36 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x36 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x37 = np.zeros((nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x37 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3)) * 0.5
    x37 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2)) * -0.5
    x37 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 4, 5)) * 0.5
    x37 += einsum(t2.bbbb, (0, 1, 2, 3), x36, (1, 4, 3, 5), (0, 4, 5))
    del x36
    t1new_bb += einsum(x37, (0, 1, 2), t2.bbbb, (3, 1, 4, 2), (0, 4)) * -4.0
    del x37
    x38 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x38 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x38 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x39 = np.zeros((nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x39 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1)) * -1.0
    x39 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (4, 0, 5)) * -2.0
    x39 += einsum(t2.abab, (0, 1, 2, 3), x38, (0, 4, 5, 2), (4, 1, 5)) * -1.0
    del x38
    t1new_bb += einsum(x39, (0, 1, 2), t2.abab, (0, 3, 2, 4), (1, 4))
    del x39
    x40 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x40 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x40 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_bb += einsum(x5, (0, 1), x40, (0, 2, 1, 3), (2, 3)) * -1.0
    del x40
    x41 = np.zeros((nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x41 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0))
    x41 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 5, 3), (5, 1, 4)) * -1.0
    t1new_bb += einsum(x41, (0, 1, 2), t2.abab, (3, 0, 2, 4), (1, 4)) * -1.0
    del x41
    x42 = np.zeros((nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x42 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 2, 3))
    x42 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 1, 3)) * -1.0
    t1new_bb += einsum(x42, (0, 1, 2), t2.bbbb, (3, 0, 1, 4), (3, 2)) * -2.0
    del x42
    x43 = np.zeros((nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x43 += einsum(v.aabb.oooo, (0, 1, 2, 3), (0, 2, 3))
    x43 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (4, 5, 1))
    t1new_bb += einsum(x43, (0, 1, 2), t2.abab, (0, 1, 3, 4), (2, 4))
    del x43
    x44 = np.zeros((nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x44 += einsum(v.bbbb.oooo, (0, 1, 2, 2), (0, 1, 2))
    x44 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (4, 0, 1)) * -1.0
    t1new_bb += einsum(x44, (0, 1, 2), t2.bbbb, (2, 0, 3, 4), (1, 3)) * -2.0
    del x44
    x45 = np.zeros((nvir[0],), dtype=np.float64)
    x45 += einsum(x24, (0,), (0,))
    del x24
    x45 += einsum(x25, (0,), (0,)) * 0.5
    del x25
    t1new_bb += einsum(x45, (0,), t2.abab, (1, 2, 0, 3), (2, 3)) * 2.0
    del x45
    x46 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x46 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x46 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 1, 3), (0, 4)) * 2.0
    t1new_bb += einsum(x46, (0, 1), x5, (1, 2), (0, 2)) * -1.0
    del x5, x46
    x47 = np.zeros((nocc[0],), dtype=np.float64)
    x47 += einsum(x31, (0,), (0,))
    del x31
    x47 += einsum(x32, (0,), (0,)) * 0.5
    del x32
    t1new_bb += einsum(x47, (0,), t2.abab, (0, 1, 2, 3), (1, 3)) * 2.0
    del x47

    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t2new": t2new}

