# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 2, 1, 3), ())
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
    x3 += einsum(f.bb.ov, (0, 1), (0, 1)) * 2.0
    x3 += einsum(t1.bb, (0, 1), x2, (0, 2, 1, 3), (2, 3)) * -1.0
    del x2
    e_cc += einsum(t1.bb, (0, 1), x3, (0, 1), ()) * 0.5
    del x3

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    t1new = Namespace()
    t2new = Namespace()
    t3new = Namespace()

    # T amplitudes
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    t1new_bb += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb, (4, 0, 2, 5, 3, 1), (4, 5)) * -3.0
    t1new_bb += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.abaaba, (0, 4, 2, 1, 5, 3), (4, 5))
    t1new_bb += einsum(f.bb.ov, (0, 1), (0, 1))
    t1new_bb += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 0, 2, 5, 1, 3), (4, 5)) * 2.0
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    t1new_aa += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.babbab, (0, 4, 2, 1, 5, 3), (4, 5))
    t1new_aa += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 2, 0, 5, 3, 1), (4, 5)) * 2.0
    t1new_aa += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa, (4, 0, 2, 5, 3, 1), (4, 5)) * -3.0
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
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), x2, (4, 1, 0, 3), (4, 2)) * -2.0
    t2new_abab += einsum(x2, (0, 1, 2, 3), t3.babbab, (2, 4, 1, 5, 6, 3), (4, 0, 6, 5)) * -2.0
    del x2
    x3 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x3 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (0, 4, 2, 3))
    x4 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x4 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x4 += einsum(x3, (0, 1, 2, 3), (1, 0, 2, 3))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x4, (1, 4, 0, 2), (4, 3)) * -1.0
    t2new_abab += einsum(x4, (0, 1, 2, 3), t3.abaaba, (4, 0, 2, 5, 6, 3), (4, 1, 5, 6)) * -2.0
    x5 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x5 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x5 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_bb += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x5, (0, 4, 3, 1), (4, 2)) * -2.0
    del x5
    x6 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum(t2.abab, (0, 1, 2, 3), (1, 3, 0, 2))
    x6 += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (2, 3, 0, 1))
    t1new_bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), x6, (4, 3, 0, 1), (4, 2))
    t1new_aa += einsum(v.aabb.vvov, (0, 1, 2, 3), x6, (2, 3, 4, 1), (4, 0))
    x7 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x7 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    t1new_aa += einsum(x7, (0, 1), (0, 1))
    x8 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x8 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x8 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x9 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x9 += einsum(t1.aa, (0, 1), x8, (0, 2, 1, 3), (2, 3))
    x10 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x10 += einsum(f.aa.ov, (0, 1), (0, 1))
    x10 += einsum(x7, (0, 1), (0, 1))
    del x7
    x10 += einsum(x9, (0, 1), (0, 1)) * -1.0
    del x9
    t1new_bb += einsum(x10, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    t1new_aa += einsum(x10, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    t2new_aaaa += einsum(x10, (0, 1), t3.aaaaaa, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    t2new_abab += einsum(x10, (0, 1), t3.abaaba, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 2.0
    t2new_bbbb += einsum(x10, (0, 1), t3.babbab, (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    x11 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x11 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x11 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x12 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x12 += einsum(t1.bb, (0, 1), x11, (0, 2, 1, 3), (2, 3))
    x13 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x13 += einsum(f.bb.ov, (0, 1), (0, 1))
    x13 += einsum(x0, (0, 1), (0, 1))
    del x0
    x13 += einsum(x12, (0, 1), (0, 1)) * -1.0
    del x12
    t1new_bb += einsum(x13, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_aa += einsum(x13, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    t2new_aaaa += einsum(x13, (0, 1), t3.abaaba, (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    t2new_abab += einsum(x13, (0, 1), t3.babbab, (2, 3, 0, 4, 5, 1), (3, 2, 5, 4)) * 2.0
    t2new_bbbb += einsum(x13, (0, 1), t3.bbbbbb, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    x14 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x14 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x14 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_bb += einsum(t1.bb, (0, 1), x14, (0, 2, 1, 3), (2, 3)) * -1.0
    del x14
    x15 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x15 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3))
    x16 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x16 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x17 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x17 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x18 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x18 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x18 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x19 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x19 += einsum(f.bb.oo, (0, 1), (0, 1)) * 0.5
    x19 += einsum(x15, (0, 1), (1, 0)) * 0.5
    x19 += einsum(x16, (0, 1), (1, 0))
    x19 += einsum(x17, (0, 1), (1, 0)) * 0.5
    x19 += einsum(t1.bb, (0, 1), x18, (2, 3, 0, 1), (3, 2)) * -0.5
    x19 += einsum(t1.bb, (0, 1), x13, (2, 1), (2, 0)) * 0.5
    t1new_bb += einsum(t1.bb, (0, 1), x19, (0, 2), (2, 1)) * -2.0
    del x19
    x20 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x20 += einsum(f.bb.vv, (0, 1), (0, 1))
    x20 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(t1.bb, (0, 1), x20, (1, 2), (0, 2))
    del x20
    x21 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x21 += einsum(t1.aa, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (3, 4, 0, 2))
    x22 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x22 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x22 += einsum(x21, (0, 1, 2, 3), (0, 1, 3, 2))
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x22, (1, 3, 0, 4), (4, 2)) * -1.0
    t2new_abab += einsum(x22, (0, 1, 2, 3), t3.babbab, (4, 2, 0, 5, 6, 1), (3, 4, 6, 5)) * -2.0
    x23 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x23 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x24 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x24 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x24 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), x24, (4, 1, 0, 3), (4, 2)) * -2.0
    t2new_abab += einsum(x24, (0, 1, 2, 3), t3.abaaba, (2, 4, 1, 5, 6, 3), (0, 4, 5, 6)) * -2.0
    del x24
    x25 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x25 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x25 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x25, (0, 4, 3, 1), (4, 2)) * -2.0
    del x25
    x26 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x26 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x26 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_aa += einsum(t1.aa, (0, 1), x26, (0, 2, 1, 3), (2, 3)) * -1.0
    x27 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x27 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3))
    x28 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x28 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x29 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x29 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x30 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x30 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x30 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x31 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x31 += einsum(t1.aa, (0, 1), x30, (2, 3, 0, 1), (2, 3))
    x32 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x32 += einsum(t1.aa, (0, 1), x10, (2, 1), (0, 2))
    x33 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x33 += einsum(f.aa.oo, (0, 1), (0, 1))
    x33 += einsum(x27, (0, 1), (1, 0))
    x33 += einsum(x28, (0, 1), (1, 0))
    x33 += einsum(x29, (0, 1), (1, 0)) * 2.0
    x33 += einsum(x31, (0, 1), (1, 0)) * -1.0
    x33 += einsum(x32, (0, 1), (1, 0))
    t1new_aa += einsum(t1.aa, (0, 1), x33, (0, 2), (2, 1)) * -1.0
    t2new_abab += einsum(x33, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    del x33
    x34 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x34 += einsum(f.aa.vv, (0, 1), (0, 1))
    x34 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_aa += einsum(t1.aa, (0, 1), x34, (1, 2), (0, 2))
    del x34
    x35 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x35 += einsum(v.aabb.ooov, (0, 1, 2, 3), t3.abaaba, (4, 2, 1, 5, 3, 6), (4, 0, 5, 6))
    x36 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x36 += einsum(v.aaaa.ooov, (0, 1, 2, 3), t3.aaaaaa, (4, 1, 2, 5, 6, 3), (4, 0, 5, 6))
    x37 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x37 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x38 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x38 += einsum(t2.aaaa, (0, 1, 2, 3), x37, (4, 5, 1, 0), (4, 5, 2, 3)) * -1.0
    x39 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x39 += einsum(x21, (0, 1, 2, 3), t3.abaaba, (4, 0, 3, 5, 1, 6), (2, 4, 5, 6))
    x40 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x40 += einsum(x23, (0, 1, 2, 3), t3.aaaaaa, (4, 2, 1, 5, 6, 3), (0, 4, 5, 6))
    x41 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x41 += einsum(f.aa.oo, (0, 1), (0, 1))
    x41 += einsum(x32, (0, 1), (0, 1))
    del x32
    x42 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x42 += einsum(x41, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x41
    x43 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x43 += einsum(x27, (0, 1), (1, 0))
    del x27
    x43 += einsum(x28, (0, 1), (1, 0))
    del x28
    x43 += einsum(x29, (0, 1), (1, 0)) * 2.0
    del x29
    x43 += einsum(x31, (0, 1), (1, 0)) * -1.0
    del x31
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x44 += einsum(x43, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x43
    x45 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x45 += einsum(x35, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x35
    x45 += einsum(x36, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.0
    del x36
    x45 += einsum(x38, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x38
    x45 += einsum(x39, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x39
    x45 += einsum(x40, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.0
    del x40
    x45 += einsum(x42, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x42
    x45 += einsum(x44, (0, 1, 2, 3), (0, 1, 3, 2))
    del x44
    t2new_aaaa += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x45, (0, 1, 2, 3), (1, 0, 2, 3))
    del x45
    x46 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x46 += einsum(t1.aa, (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x47 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x47 += einsum(t1.aa, (0, 1), x46, (2, 3, 1, 4), (0, 2, 3, 4))
    x48 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x48 += einsum(v.aabb.vvov, (0, 1, 2, 3), t3.abaaba, (4, 2, 5, 6, 3, 1), (4, 5, 6, 0))
    x49 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x49 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 1, 3), (4, 5, 6, 2))
    x50 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x50 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x50 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x51 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x51 += einsum(t2.aaaa, (0, 1, 2, 3), x50, (1, 4, 5, 3), (0, 4, 2, 5))
    del x50
    x52 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x52 += einsum(t2.aaaa, (0, 1, 2, 3), x51, (4, 1, 5, 3), (0, 4, 2, 5)) * -4.0
    del x51
    x53 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x53 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 1), (2, 3))
    x54 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x54 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    x55 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x55 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 4, 1, 3), (2, 4))
    x56 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x56 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x56 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x57 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x57 += einsum(t1.aa, (0, 1), x56, (0, 1, 2, 3), (2, 3))
    x58 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x58 += einsum(x53, (0, 1), (1, 0)) * -1.0
    x58 += einsum(x54, (0, 1), (1, 0))
    x58 += einsum(x55, (0, 1), (1, 0)) * 2.0
    x58 += einsum(x57, (0, 1), (1, 0)) * -1.0
    x59 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x59 += einsum(x58, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x58
    x60 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x60 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x61 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x61 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 2, 5, 6, 3, 1), (4, 5, 0, 6))
    x62 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x62 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 2, 6, 3, 1), (4, 5, 0, 6)) * -1.0
    x63 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x63 += einsum(x10, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4)) * -2.0
    x64 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x64 += einsum(t1.aa, (0, 1), x23, (2, 3, 4, 1), (2, 0, 4, 3))
    x65 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x65 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x65 += einsum(x64, (0, 1, 2, 3), (3, 1, 2, 0))
    x66 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x66 += einsum(t1.aa, (0, 1), x65, (0, 2, 3, 4), (2, 3, 4, 1))
    del x65
    x67 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x67 += einsum(x60, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    del x60
    x67 += einsum(x61, (0, 1, 2, 3), (2, 1, 0, 3)) * 2.0
    del x61
    x67 += einsum(x62, (0, 1, 2, 3), (2, 1, 0, 3)) * 6.0
    del x62
    x67 += einsum(x63, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x63
    x67 += einsum(x66, (0, 1, 2, 3), (1, 0, 2, 3))
    del x66
    x68 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x68 += einsum(t1.aa, (0, 1), x67, (0, 2, 3, 4), (2, 3, 1, 4))
    del x67
    x69 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x69 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3))
    del x47
    x69 += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x48
    x69 += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3)) * -6.0
    del x49
    x69 += einsum(x52, (0, 1, 2, 3), (1, 0, 3, 2))
    del x52
    x69 += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x59
    x69 += einsum(x68, (0, 1, 2, 3), (1, 0, 2, 3))
    del x68
    t2new_aaaa += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x69, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x69
    x70 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x70 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x71 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x71 += einsum(t1.aa, (0, 1), v.aabb.vvov, (2, 1, 3, 4), (3, 4, 0, 2))
    x72 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x72 += einsum(t2.abab, (0, 1, 2, 3), x71, (1, 3, 4, 5), (4, 0, 2, 5))
    x73 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x73 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x73 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -0.5
    x74 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x74 += einsum(x26, (0, 1, 2, 3), x73, (0, 4, 2, 5), (1, 4, 3, 5)) * 2.0
    del x26
    x75 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x75 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ovov, (1, 3, 4, 5), (4, 5, 0, 2))
    x76 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x76 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x76 += einsum(x75, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x75
    x77 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x77 += einsum(t2.abab, (0, 1, 2, 3), x76, (1, 3, 4, 5), (0, 4, 2, 5))
    del x76
    x78 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x78 += einsum(t1.aa, (0, 1), x56, (2, 1, 3, 4), (0, 2, 3, 4))
    del x56
    x79 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x79 += einsum(t2.aaaa, (0, 1, 2, 3), x78, (4, 1, 5, 3), (0, 4, 2, 5)) * -2.0
    x80 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x80 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x81 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x81 += einsum(t1.aa, (0, 1), x37, (2, 3, 4, 0), (2, 4, 3, 1))
    x82 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x82 += einsum(t1.aa, (0, 1), x70, (2, 3, 1, 4), (0, 2, 3, 4))
    x83 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x83 += einsum(t2.abab, (0, 1, 2, 3), x21, (1, 3, 4, 5), (4, 0, 5, 2))
    x84 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x84 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x84 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x85 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x85 += einsum(t2.aaaa, (0, 1, 2, 3), x84, (4, 5, 1, 3), (0, 4, 5, 2)) * 2.0
    del x84
    x86 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x86 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x86 += einsum(x23, (0, 1, 2, 3), (0, 2, 1, 3))
    x87 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x87 += einsum(t2.aaaa, (0, 1, 2, 3), x86, (4, 1, 5, 3), (0, 4, 5, 2)) * 2.0
    del x86
    x88 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x88 += einsum(x80, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x80
    x88 += einsum(x81, (0, 1, 2, 3), (0, 2, 1, 3))
    del x81
    x88 += einsum(x82, (0, 1, 2, 3), (0, 2, 1, 3))
    x88 += einsum(x83, (0, 1, 2, 3), (0, 2, 1, 3))
    del x83
    x88 += einsum(x85, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x85
    x88 += einsum(x87, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x87
    x89 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x89 += einsum(t1.aa, (0, 1), x88, (2, 0, 3, 4), (2, 3, 1, 4))
    del x88
    x90 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x90 += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3))
    x90 += einsum(x72, (0, 1, 2, 3), (0, 1, 2, 3))
    del x72
    x90 += einsum(x74, (0, 1, 2, 3), (1, 0, 3, 2))
    del x74
    x90 += einsum(x77, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x77
    x90 += einsum(x79, (0, 1, 2, 3), (1, 0, 2, 3))
    del x79
    x90 += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3))
    del x89
    t2new_aaaa += einsum(x90, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x90, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa += einsum(x90, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x90, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x90
    x91 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x91 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x92 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x92 += einsum(t2.abab, (0, 1, 2, 3), x11, (1, 4, 3, 5), (4, 5, 0, 2))
    x93 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x93 += einsum(t2.abab, (0, 1, 2, 3), x92, (1, 3, 4, 5), (0, 4, 2, 5)) * -1.0
    del x92
    x94 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x94 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x94 += einsum(x91, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x91
    x94 += einsum(x93, (0, 1, 2, 3), (0, 1, 2, 3))
    del x93
    t2new_aaaa += einsum(x94, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3))
    del x94
    x95 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x95 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new_aaaa += einsum(x95, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x95, (0, 1, 2, 3), (1, 0, 2, 3))
    del x95
    x96 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x96 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x97 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x97 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x97 += einsum(x96, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    x97 += einsum(x64, (0, 1, 2, 3), (2, 1, 3, 0))
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x97, (0, 4, 1, 5), (4, 5, 2, 3)) * -2.0
    del x97
    x98 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x98 += einsum(t1.aa, (0, 1), x96, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    del x96
    x98 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x98 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    t2new_aaaa += einsum(t1.aa, (0, 1), x98, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x98
    x99 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x99 += einsum(t1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (0, 4, 2, 3))
    t2new_abab += einsum(x99, (0, 1, 2, 3), (2, 0, 3, 1))
    x100 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x100 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x100 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 0, 4), (2, 1, 3, 4))
    t2new_abab += einsum(x100, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 2, 6, 3), (4, 5, 1, 6)) * 2.0
    del x100
    x101 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x101 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 4), (1, 4, 2, 3))
    x102 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x102 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (2, 3, 0, 1))
    x102 += einsum(x101, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_abab += einsum(x102, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 6, 0, 3), (4, 5, 6, 1)) * 2.0
    del x102
    x103 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x103 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 2, 3, 4), (3, 4, 1, 2))
    x104 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x104 += einsum(v.aabb.vvov, (0, 1, 2, 3), (2, 3, 0, 1))
    x104 += einsum(x103, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_abab += einsum(x104, (0, 1, 2, 3), t3.babbab, (4, 5, 0, 6, 2, 1), (5, 4, 3, 6)) * 2.0
    del x104
    x105 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x105 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x105 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 0, 4), (2, 1, 3, 4))
    t2new_abab += einsum(x105, (0, 1, 2, 3), t3.babbab, (4, 5, 0, 2, 6, 3), (5, 4, 6, 1)) * 2.0
    del x105
    x106 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x106 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x106 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x106 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (4, 0, 5, 2))
    x106 += einsum(x73, (0, 1, 2, 3), x8, (0, 4, 2, 5), (4, 1, 5, 3)) * -2.0
    del x73
    x106 += einsum(x78, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x106 += einsum(t1.aa, (0, 1), x30, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x30
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x106, (0, 4, 2, 5), (4, 1, 5, 3))
    del x106
    x107 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x107 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x107 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -0.5
    x108 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x108 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x108 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x109 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x109 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x109 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x110 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x110 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x110 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x110 += einsum(x107, (0, 1, 2, 3), x11, (0, 4, 5, 2), (4, 1, 5, 3)) * -1.0
    del x11
    x110 += einsum(t1.bb, (0, 1), x108, (2, 1, 3, 4), (2, 0, 4, 3)) * -0.5
    del x108
    x110 += einsum(t1.bb, (0, 1), x109, (0, 2, 3, 4), (3, 2, 4, 1)) * -0.5
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x110, (1, 4, 3, 5), (0, 4, 2, 5)) * -2.0
    del x110
    x111 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x111 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1)) * 0.5
    x111 += einsum(t1.bb, (0, 1), v.aabb.ovoo, (2, 3, 4, 0), (4, 1, 2, 3)) * -0.5
    x111 += einsum(x99, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x111 += einsum(v.aabb.ovov, (0, 1, 2, 3), x107, (2, 4, 3, 5), (4, 5, 0, 1))
    t2new_abab += einsum(t2.aaaa, (0, 1, 2, 3), x111, (4, 5, 1, 3), (0, 4, 2, 5)) * 4.0
    del x111
    x112 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x112 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 4), (1, 4, 2, 3))
    x113 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x113 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (3, 4, 0, 2))
    x114 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x114 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x114 += einsum(x112, (0, 1, 2, 3), (1, 0, 3, 2))
    x114 += einsum(x113, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x114 += einsum(v.aabb.ovov, (0, 1, 2, 3), x6, (2, 4, 5, 1), (3, 4, 0, 5))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x114, (3, 4, 0, 5), (5, 1, 2, 4))
    del x114
    x115 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x115 += einsum(t1.aa, (0, 1), x22, (2, 3, 0, 4), (2, 3, 4, 1))
    x116 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x116 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x116 += einsum(x71, (0, 1, 2, 3), (0, 1, 2, 3))
    x116 += einsum(x115, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x115
    t2new_abab += einsum(x107, (0, 1, 2, 3), x116, (0, 2, 4, 5), (4, 1, 5, 3)) * 2.0
    del x107
    x117 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x117 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 4, 1), (0, 4, 2, 3))
    x118 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x118 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x118 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    x119 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x119 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x119 += einsum(x117, (0, 1, 2, 3), (1, 0, 3, 2))
    x119 += einsum(t1.aa, (0, 1), x118, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x119, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    del x119
    x120 = np.zeros((nvir[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x120 += einsum(v.aabb.vvvv, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x120 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 4), (4, 1, 2, 3))
    x120 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 2, 3, 4), (3, 4, 2, 1))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x120, (3, 4, 2, 5), (0, 1, 5, 4)) * -1.0
    del x120
    x121 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x121 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 4, 1), (0, 4, 2, 3))
    x122 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x122 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x123 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x123 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x123 += einsum(x121, (0, 1, 2, 3), (1, 0, 3, 2))
    x123 += einsum(x122, (0, 1, 2, 3), (1, 0, 3, 2))
    x123 += einsum(v.aabb.ovov, (0, 1, 2, 3), x6, (4, 3, 5, 1), (2, 4, 0, 5))
    del x6
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x123, (1, 4, 0, 5), (5, 4, 2, 3))
    del x123
    x124 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x124 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x124 += einsum(x53, (0, 1), (1, 0)) * -1.0
    del x53
    x124 += einsum(x54, (0, 1), (1, 0))
    del x54
    x124 += einsum(x55, (0, 1), (1, 0)) * 2.0
    del x55
    x124 += einsum(x57, (0, 1), (1, 0)) * -1.0
    del x57
    x124 += einsum(t1.aa, (0, 1), x10, (0, 2), (2, 1))
    del x10
    t2new_abab += einsum(x124, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    del x124
    x125 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x125 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 1, 2, 3), (2, 3))
    x126 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x126 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 4, 1, 3), (2, 4))
    x127 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x127 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    x128 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x128 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x128 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x129 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x129 += einsum(t1.bb, (0, 1), x128, (0, 2, 1, 3), (2, 3))
    del x128
    x130 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x130 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x130 += einsum(x125, (0, 1), (1, 0)) * -1.0
    x130 += einsum(x126, (0, 1), (1, 0)) * 2.0
    x130 += einsum(x127, (0, 1), (1, 0))
    x130 += einsum(x129, (0, 1), (0, 1)) * -1.0
    x130 += einsum(t1.bb, (0, 1), x13, (0, 2), (2, 1))
    t2new_abab += einsum(x130, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x130
    x131 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x131 += einsum(t1.bb, (0, 1), x18, (2, 3, 0, 1), (2, 3))
    del x18
    x132 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x132 += einsum(t1.bb, (0, 1), x13, (2, 1), (0, 2))
    x133 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x133 += einsum(f.bb.oo, (0, 1), (0, 1))
    x133 += einsum(x15, (0, 1), (1, 0))
    x133 += einsum(x16, (0, 1), (1, 0)) * 2.0
    x133 += einsum(x17, (0, 1), (1, 0))
    x133 += einsum(x131, (0, 1), (1, 0)) * -1.0
    x133 += einsum(x132, (0, 1), (1, 0))
    t2new_abab += einsum(x133, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x133
    x134 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x134 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x134 += einsum(x113, (0, 1, 2, 3), (1, 0, 2, 3))
    x135 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x135 += einsum(t1.bb, (0, 1), x134, (1, 2, 3, 4), (0, 2, 3, 4))
    del x134
    x136 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x136 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x136 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3))
    x136 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (5, 1, 0, 4))
    x137 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x137 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x137 += einsum(x21, (0, 1, 2, 3), (0, 1, 3, 2))
    x137 += einsum(x135, (0, 1, 2, 3), (0, 1, 3, 2))
    x137 += einsum(t1.bb, (0, 1), x136, (0, 2, 3, 4), (2, 1, 4, 3)) * -1.0
    del x136
    t2new_abab += einsum(t1.aa, (0, 1), x137, (2, 3, 0, 4), (4, 2, 1, 3)) * -1.0
    del x137
    x138 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x138 += einsum(t1.bb, (0, 1), v.aabb.vvvv, (2, 3, 4, 1), (0, 4, 2, 3))
    x139 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x139 += einsum(v.aabb.vvov, (0, 1, 2, 3), (2, 3, 0, 1))
    x139 += einsum(x138, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_abab += einsum(t1.aa, (0, 1), x139, (2, 3, 1, 4), (0, 2, 4, 3))
    del x139
    x140 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x140 += einsum(t1.aa, (0, 1), v.aabb.vvoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x141 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x141 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x141 += einsum(x140, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_abab += einsum(t1.bb, (0, 1), x141, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    del x141
    x142 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x142 += einsum(t1.bb, (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x143 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x143 += einsum(t1.bb, (0, 1), x142, (2, 3, 1, 4), (0, 2, 3, 4))
    x144 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x144 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 0, 6, 1, 3), (4, 5, 6, 2))
    x145 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x145 += einsum(v.aabb.ovvv, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 1, 3), (4, 5, 6, 2))
    x146 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x146 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x146 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x147 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x147 += einsum(t2.bbbb, (0, 1, 2, 3), x146, (1, 4, 5, 3), (4, 0, 5, 2))
    del x146
    x148 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x148 += einsum(t2.bbbb, (0, 1, 2, 3), x147, (1, 4, 3, 5), (4, 0, 5, 2)) * -4.0
    del x147
    x149 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x149 += einsum(t2.abab, (0, 1, 2, 3), x8, (0, 4, 2, 5), (1, 3, 4, 5))
    del x8
    x150 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x150 += einsum(t2.abab, (0, 1, 2, 3), x149, (4, 5, 0, 2), (4, 1, 5, 3)) * -1.0
    del x149
    x151 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x151 += einsum(x125, (0, 1), (1, 0)) * -1.0
    del x125
    x151 += einsum(x126, (0, 1), (1, 0)) * 2.0
    del x126
    x151 += einsum(x127, (0, 1), (1, 0))
    del x127
    x151 += einsum(x129, (0, 1), (0, 1)) * -1.0
    del x129
    x152 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x152 += einsum(x151, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 1, 4)) * -2.0
    del x151
    x153 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x153 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x154 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x154 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 2, 6, 3, 1), (4, 5, 0, 6)) * -1.0
    x155 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x155 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 1, 3), (4, 5, 2, 6))
    x156 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x156 += einsum(x13, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4)) * -2.0
    del x13
    x157 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x157 += einsum(t1.bb, (0, 1), x1, (2, 3, 4, 1), (2, 0, 4, 3))
    x158 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x158 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x158 += einsum(x157, (0, 1, 2, 3), (3, 1, 2, 0))
    x159 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x159 += einsum(t1.bb, (0, 1), x158, (0, 2, 3, 4), (2, 3, 4, 1))
    del x158
    x160 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x160 += einsum(x153, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    del x153
    x160 += einsum(x154, (0, 1, 2, 3), (2, 1, 0, 3)) * 6.0
    del x154
    x160 += einsum(x155, (0, 1, 2, 3), (2, 1, 0, 3)) * 2.0
    del x155
    x160 += einsum(x156, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x156
    x160 += einsum(x159, (0, 1, 2, 3), (1, 0, 2, 3))
    del x159
    x161 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x161 += einsum(t1.bb, (0, 1), x160, (0, 2, 3, 4), (2, 3, 4, 1))
    del x160
    x162 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x162 += einsum(x143, (0, 1, 2, 3), (0, 1, 2, 3))
    del x143
    x162 += einsum(x144, (0, 1, 2, 3), (0, 1, 2, 3)) * -6.0
    del x144
    x162 += einsum(x145, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x145
    x162 += einsum(x148, (0, 1, 2, 3), (0, 1, 2, 3))
    del x148
    x162 += einsum(x150, (0, 1, 2, 3), (0, 1, 2, 3))
    del x150
    x162 += einsum(x152, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x152
    x162 += einsum(x161, (0, 1, 2, 3), (1, 0, 3, 2))
    del x161
    t2new_bbbb += einsum(x162, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x162, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x162
    x163 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x163 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x164 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x164 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 4, 3, 5))
    x165 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x165 += einsum(t2.abab, (0, 1, 2, 3), x99, (4, 5, 0, 2), (4, 1, 3, 5))
    x166 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x166 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x166 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x166 += einsum(x164, (0, 1, 2, 3), (1, 0, 3, 2))
    x167 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x167 += einsum(t2.bbbb, (0, 1, 2, 3), x166, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    del x166
    x168 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x168 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x168 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x169 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x169 += einsum(t1.bb, (0, 1), x168, (2, 1, 3, 4), (2, 0, 3, 4))
    del x168
    x170 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x170 += einsum(t2.bbbb, (0, 1, 2, 3), x169, (1, 4, 5, 3), (4, 0, 5, 2)) * -2.0
    x171 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x171 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x172 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x172 += einsum(t1.bb, (0, 1), x171, (2, 3, 4, 0), (2, 4, 3, 1))
    x173 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x173 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 5), (1, 4, 5, 3))
    x174 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x174 += einsum(t2.abab, (0, 1, 2, 3), x3, (4, 5, 0, 2), (4, 1, 5, 3))
    x175 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x175 += einsum(t2.bbbb, (0, 1, 2, 3), x109, (1, 4, 5, 3), (0, 4, 5, 2)) * 2.0
    del x109
    x176 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x176 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x176 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3))
    x177 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x177 += einsum(t2.bbbb, (0, 1, 2, 3), x176, (4, 1, 5, 3), (4, 5, 0, 2)) * 2.0
    del x176
    x178 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x178 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x178 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x178 += einsum(x163, (0, 1, 2, 3), (0, 1, 2, 3))
    x179 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x179 += einsum(t1.bb, (0, 1), x178, (2, 3, 1, 4), (2, 3, 0, 4))
    del x178
    x180 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x180 += einsum(x172, (0, 1, 2, 3), (0, 2, 1, 3))
    del x172
    x180 += einsum(x173, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x173
    x180 += einsum(x174, (0, 1, 2, 3), (0, 2, 1, 3))
    del x174
    x180 += einsum(x175, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x175
    x180 += einsum(x177, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x177
    x180 += einsum(x179, (0, 1, 2, 3), (2, 1, 0, 3))
    x181 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x181 += einsum(t1.bb, (0, 1), x180, (2, 0, 3, 4), (2, 3, 4, 1))
    del x180
    x182 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x182 += einsum(x163, (0, 1, 2, 3), (0, 1, 2, 3))
    x182 += einsum(x164, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x164
    x182 += einsum(x165, (0, 1, 2, 3), (0, 1, 2, 3))
    del x165
    x182 += einsum(x167, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x167
    x182 += einsum(x170, (0, 1, 2, 3), (0, 1, 3, 2))
    del x170
    x182 += einsum(x181, (0, 1, 2, 3), (0, 1, 3, 2))
    del x181
    t2new_bbbb += einsum(x182, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x182, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb += einsum(x182, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb += einsum(x182, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x182
    x183 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x183 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x184 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x184 += einsum(v.bbbb.ooov, (0, 1, 2, 3), t3.bbbbbb, (4, 1, 2, 5, 6, 3), (4, 0, 5, 6))
    x185 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x185 += einsum(t2.bbbb, (0, 1, 2, 3), x171, (4, 5, 0, 1), (4, 5, 2, 3))
    x186 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x186 += einsum(v.aabb.ovoo, (0, 1, 2, 3), t3.babbab, (4, 0, 3, 5, 1, 6), (4, 2, 5, 6))
    x187 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x187 += einsum(x1, (0, 1, 2, 3), t3.bbbbbb, (4, 1, 2, 5, 6, 3), (0, 4, 5, 6)) * -1.0
    x188 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x188 += einsum(x3, (0, 1, 2, 3), t3.babbab, (4, 2, 1, 5, 3, 6), (0, 4, 5, 6))
    x189 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x189 += einsum(f.bb.oo, (0, 1), (0, 1))
    x189 += einsum(x132, (0, 1), (0, 1))
    del x132
    x190 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x190 += einsum(x189, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4)) * -2.0
    del x189
    x191 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x191 += einsum(x15, (0, 1), (1, 0))
    del x15
    x191 += einsum(x16, (0, 1), (1, 0)) * 2.0
    del x16
    x191 += einsum(x17, (0, 1), (1, 0))
    del x17
    x191 += einsum(x131, (0, 1), (1, 0)) * -1.0
    del x131
    x192 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x192 += einsum(x191, (0, 1), t2.bbbb, (2, 0, 3, 4), (1, 2, 3, 4)) * -2.0
    del x191
    x193 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x193 += einsum(x183, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x183
    x193 += einsum(x184, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.0
    del x184
    x193 += einsum(x185, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x185
    x193 += einsum(x186, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x186
    x193 += einsum(x187, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.0
    del x187
    x193 += einsum(x188, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x188
    x193 += einsum(x190, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x190
    x193 += einsum(x192, (0, 1, 2, 3), (1, 0, 3, 2))
    del x192
    t2new_bbbb += einsum(x193, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x193, (0, 1, 2, 3), (1, 0, 2, 3))
    del x193
    x194 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x194 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x195 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x195 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x195 += einsum(x194, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x194
    t2new_bbbb += einsum(x195, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x195, (0, 1, 2, 3), (0, 1, 2, 3))
    del x195
    x196 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x196 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x196 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    x197 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x197 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x197 += einsum(v.bbbb.ovov, (0, 1, 2, 3), x196, (4, 5, 1, 3), (0, 5, 2, 4))
    del x196
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x197, (0, 4, 1, 5), (4, 5, 2, 3)) * -2.0
    del x197
    x198 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x198 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x199 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x199 += einsum(t1.bb, (0, 1), x198, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    del x198
    x199 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x199 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    t2new_bbbb += einsum(t1.bb, (0, 1), x199, (2, 3, 0, 4), (2, 3, 1, 4))
    del x199
    x200 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x200 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    x201 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x201 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    x202 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x202 += einsum(t1.aa, (0, 1), x201, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x201
    x203 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x203 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x203 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x204 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x204 += einsum(t2.aaaa, (0, 1, 2, 3), x203, (4, 5, 3, 6), (4, 5, 0, 1, 6, 2)) * -1.0
    x205 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x205 += einsum(x202, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x202
    x205 += einsum(x204, (0, 1, 2, 3, 4, 5), (3, 2, 1, 0, 5, 4))
    del x204
    x206 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x206 += einsum(t1.aa, (0, 1), x205, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x205
    x207 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x207 += einsum(x200, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    del x200
    x207 += einsum(x206, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    del x206
    t3new_aaaaaa = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new_aaaaaa += einsum(x207, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x207
    x208 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x208 += einsum(f.aa.ov, (0, 1), t1.aa, (0, 2), (1, 2))
    x209 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x209 += einsum(f.aa.vv, (0, 1), (0, 1))
    x209 += einsum(x208, (0, 1), (0, 1)) * -1.0
    del x208
    t3new_babbab = np.zeros((nocc[1], nocc[0], nocc[1], nvir[1], nvir[0], nvir[1]), dtype=types[float])
    t3new_babbab += einsum(x209, (0, 1), t3.babbab, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * 2.0
    x210 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x210 += einsum(x209, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6)) * 6.0
    del x209
    t3new_aaaaaa += einsum(x210, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new_aaaaaa += einsum(x210, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x210, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x210
    x211 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x211 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.oooo, (4, 5, 6, 1), (0, 4, 5, 6, 2, 3))
    x212 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x212 += einsum(t2.aaaa, (0, 1, 2, 3), x64, (4, 5, 1, 6), (5, 4, 0, 6, 2, 3))
    x213 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x213 += einsum(x37, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x213 += einsum(x37, (0, 1, 2, 3), (0, 2, 3, 1))
    x214 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x214 += einsum(t2.aaaa, (0, 1, 2, 3), x213, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3)) * -1.0
    del x213
    x215 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x215 += einsum(x211, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    del x211
    x215 += einsum(x212, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x212
    x215 += einsum(x214, (0, 1, 2, 3, 4, 5), (0, 3, 2, 1, 5, 4)) * -1.0
    del x214
    x216 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x216 += einsum(t1.aa, (0, 1), x215, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x215
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new_aaaaaa += einsum(x216, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    del x216
    x217 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x217 += einsum(t2.aaaa, (0, 1, 2, 3), x46, (4, 5, 3, 6), (4, 0, 1, 2, 6, 5))
    x218 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x218 += einsum(t2.aaaa, (0, 1, 2, 3), x23, (4, 5, 6, 3), (4, 0, 1, 6, 5, 2))
    x219 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x219 += einsum(t1.aa, (0, 1), x218, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x218
    x220 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x220 += einsum(t2.aaaa, (0, 1, 2, 3), x78, (4, 5, 6, 3), (0, 1, 4, 5, 2, 6))
    x221 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x221 += einsum(x219, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    del x219
    x221 += einsum(x220, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    del x220
    x222 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x222 += einsum(t1.aa, (0, 1), x221, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x221
    x223 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x223 += einsum(x217, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    del x217
    x223 += einsum(x222, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    del x222
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new_aaaaaa += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x223
    x224 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x224 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ooov, (4, 1, 5, 6), (0, 4, 5, 2, 3, 6))
    x225 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x225 += einsum(t2.aaaa, (0, 1, 2, 3), x82, (4, 5, 1, 6), (4, 5, 0, 2, 3, 6))
    del x82
    x226 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x226 += einsum(t1.aa, (0, 1), x203, (2, 3, 1, 4), (2, 3, 0, 4))
    del x203
    x227 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x227 += einsum(t2.aaaa, (0, 1, 2, 3), x226, (4, 1, 5, 6), (5, 4, 0, 6, 2, 3)) * -2.0
    del x226
    x228 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x228 += einsum(x224, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    del x224
    x228 += einsum(x225, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    del x225
    x228 += einsum(x227, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    del x227
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x228, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x228
    x229 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x229 += einsum(f.aa.ov, (0, 1), t1.aa, (2, 1), (0, 2))
    x230 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x230 += einsum(f.aa.oo, (0, 1), (0, 1))
    x230 += einsum(x229, (0, 1), (0, 1))
    del x229
    t3new_babbab += einsum(x230, (0, 1), t3.babbab, (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -2.0
    x231 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x231 += einsum(x230, (0, 1), t3.aaaaaa, (2, 3, 0, 4, 5, 6), (1, 2, 3, 4, 5, 6)) * 6.0
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    del x231
    x232 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x232 += einsum(f.aa.ov, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4))
    x233 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x233 += einsum(t2.aaaa, (0, 1, 2, 3), x232, (1, 4, 5, 6), (4, 5, 0, 6, 2, 3))
    del x232
    t3new_aaaaaa += einsum(x233, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x233, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x233, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x233, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x233, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x233, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x233, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x233, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x233, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    del x233
    x234 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x234 += einsum(t2.abab, (0, 1, 2, 3), v.bbbb.ovvv, (4, 5, 6, 3), (1, 4, 5, 6, 0, 2))
    x235 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x235 += einsum(t2.abab, (0, 1, 2, 3), x142, (4, 5, 3, 6), (4, 1, 6, 5, 0, 2))
    x236 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x236 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    x237 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x237 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x237 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x237 += einsum(x236, (0, 1, 2, 3), (1, 0, 2, 3))
    del x236
    x238 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x238 += einsum(t2.abab, (0, 1, 2, 3), x237, (4, 5, 6, 3), (4, 5, 1, 6, 0, 2))
    del x237
    x239 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x239 += einsum(t1.bb, (0, 1), x1, (2, 0, 3, 4), (2, 3, 1, 4))
    x240 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x240 += einsum(x239, (0, 1, 2, 3), (0, 1, 3, 2))
    del x239
    x240 += einsum(x169, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x241 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x241 += einsum(t2.abab, (0, 1, 2, 3), x240, (4, 5, 3, 6), (4, 5, 1, 6, 0, 2))
    del x240
    x242 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x242 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x242 += einsum(x157, (0, 1, 2, 3), (3, 1, 0, 2))
    x242 += einsum(x171, (0, 1, 2, 3), (2, 1, 0, 3))
    x242 += einsum(x171, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    x243 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x243 += einsum(t2.abab, (0, 1, 2, 3), x242, (1, 4, 5, 6), (4, 5, 6, 3, 0, 2))
    del x242
    x244 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x244 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 2, 3, 4), (3, 4, 1, 2))
    x245 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x245 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x245 += einsum(x244, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x244
    x246 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x246 += einsum(t2.abab, (0, 1, 2, 3), x245, (4, 5, 2, 6), (4, 5, 1, 3, 0, 6))
    del x245
    x247 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x247 += einsum(t1.aa, (0, 1), x3, (2, 3, 0, 4), (2, 3, 1, 4))
    del x3
    x248 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x248 += einsum(x117, (0, 1, 2, 3), (0, 1, 3, 2))
    x248 += einsum(x247, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x247
    x249 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x249 += einsum(t2.abab, (0, 1, 2, 3), x248, (4, 5, 2, 6), (4, 5, 1, 3, 0, 6))
    del x248
    x250 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x250 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x250 += einsum(x122, (0, 1, 2, 3), (1, 0, 3, 2))
    del x122
    x251 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x251 += einsum(t2.abab, (0, 1, 2, 3), x250, (4, 5, 0, 6), (4, 5, 1, 3, 6, 2))
    x252 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x252 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x252 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3))
    x253 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x253 += einsum(t1.bb, (0, 1), x252, (2, 1, 3, 4), (2, 0, 3, 4))
    del x252
    x254 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x254 += einsum(t2.abab, (0, 1, 2, 3), x253, (4, 5, 6, 0), (5, 4, 1, 3, 6, 2))
    x255 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x255 += einsum(x238, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    del x238
    x255 += einsum(x241, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x241
    x255 += einsum(x243, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    del x243
    x255 += einsum(x246, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    del x246
    x255 += einsum(x249, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x249
    x255 += einsum(x251, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    del x251
    x255 += einsum(x254, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x254
    x256 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x256 += einsum(t1.bb, (0, 1), x255, (2, 0, 3, 4, 5, 6), (2, 3, 4, 1, 5, 6))
    del x255
    x257 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x257 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x257 += einsum(x179, (0, 1, 2, 3), (2, 1, 0, 3))
    del x179
    x258 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x258 += einsum(t2.abab, (0, 1, 2, 3), x257, (4, 1, 5, 6), (4, 5, 6, 3, 0, 2))
    del x257
    x259 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x259 += einsum(f.aa.ov, (0, 1), t2.abab, (0, 2, 3, 4), (2, 4, 1, 3))
    x260 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x260 += einsum(v.aabb.vvov, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x260 += einsum(x259, (0, 1, 2, 3), (0, 1, 2, 3))
    x260 += einsum(x103, (0, 1, 2, 3), (0, 1, 3, 2))
    del x103
    x261 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x261 += einsum(t2.abab, (0, 1, 2, 3), x260, (4, 5, 2, 6), (4, 1, 5, 3, 0, 6))
    del x260
    x262 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x262 += einsum(t1.aa, (0, 1), x99, (2, 3, 0, 4), (2, 3, 1, 4))
    x263 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x263 += einsum(x138, (0, 1, 2, 3), (0, 1, 3, 2))
    x263 += einsum(x262, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x262
    x264 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x264 += einsum(t2.abab, (0, 1, 2, 3), x263, (4, 5, 2, 6), (4, 1, 5, 3, 0, 6))
    del x263
    x265 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x265 += einsum(t2.abab, (0, 1, 2, 3), x22, (4, 5, 0, 6), (1, 4, 3, 5, 6, 2))
    del x22
    x266 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x266 += einsum(t2.abab, (0, 1, 2, 3), x135, (4, 5, 6, 0), (4, 1, 5, 3, 6, 2))
    del x135
    x267 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x267 += einsum(x234, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x234
    x267 += einsum(x235, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x235
    x267 += einsum(x256, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x256
    x267 += einsum(x258, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x258
    x267 += einsum(x261, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x261
    x267 += einsum(x264, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x264
    x267 += einsum(x265, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x265
    x267 += einsum(x266, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x266
    t3new_babbab += einsum(x267, (0, 1, 2, 3, 4, 5), (0, 4, 1, 2, 5, 3)) * -1.0
    t3new_babbab += einsum(x267, (0, 1, 2, 3, 4, 5), (0, 4, 1, 3, 5, 2))
    t3new_babbab += einsum(x267, (0, 1, 2, 3, 4, 5), (1, 4, 0, 2, 5, 3))
    t3new_babbab += einsum(x267, (0, 1, 2, 3, 4, 5), (1, 4, 0, 3, 5, 2)) * -1.0
    del x267
    x268 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x268 += einsum(f.bb.ov, (0, 1), t2.abab, (2, 3, 4, 1), (0, 3, 2, 4))
    x269 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x269 += einsum(t1.aa, (0, 1), x250, (2, 3, 0, 4), (2, 3, 4, 1))
    del x250
    x270 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x270 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x270 += einsum(x268, (0, 1, 2, 3), (0, 1, 2, 3))
    del x268
    x270 += einsum(x140, (0, 1, 2, 3), (1, 0, 2, 3))
    del x140
    x270 += einsum(x269, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x269
    x271 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x271 += einsum(t2.bbbb, (0, 1, 2, 3), x270, (1, 4, 5, 6), (4, 0, 2, 3, 5, 6)) * -2.0
    del x270
    x272 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x272 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x272 += einsum(x71, (0, 1, 2, 3), (0, 1, 2, 3))
    x273 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x273 += einsum(t1.bb, (0, 1), x272, (2, 1, 3, 4), (2, 0, 3, 4))
    del x272
    x274 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x274 += einsum(t1.aa, (0, 1), x253, (2, 3, 4, 0), (3, 2, 4, 1))
    del x253
    x275 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x275 += einsum(x273, (0, 1, 2, 3), (1, 0, 2, 3))
    del x273
    x275 += einsum(x274, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x274
    x276 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x276 += einsum(t2.bbbb, (0, 1, 2, 3), x275, (4, 1, 5, 6), (4, 0, 2, 3, 5, 6)) * -2.0
    del x275
    x277 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x277 += einsum(f.bb.ov, (0, 1), t1.bb, (2, 1), (0, 2))
    x278 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x278 += einsum(f.bb.oo, (0, 1), (0, 1))
    x278 += einsum(x277, (0, 1), (0, 1))
    del x277
    t3new_abaaba = np.zeros((nocc[0], nocc[1], nocc[0], nvir[0], nvir[1], nvir[0]), dtype=types[float])
    t3new_abaaba += einsum(x278, (0, 1), t3.abaaba, (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -2.0
    x279 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x279 += einsum(x278, (0, 1), t3.babbab, (2, 3, 0, 4, 5, 6), (1, 2, 4, 6, 3, 5)) * -2.0
    x280 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x280 += einsum(x271, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x271
    x280 += einsum(x276, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x276
    x280 += einsum(x279, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x279
    t3new_babbab += einsum(x280, (0, 1, 2, 3, 4, 5), (0, 4, 1, 2, 5, 3))
    t3new_babbab += einsum(x280, (0, 1, 2, 3, 4, 5), (1, 4, 0, 2, 5, 3)) * -1.0
    del x280
    x281 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x281 += einsum(f.bb.vv, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (2, 4, 0, 5, 3, 6))
    x282 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x282 += einsum(f.bb.ov, (0, 1), t2.abab, (2, 0, 3, 4), (1, 4, 2, 3))
    x283 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x283 += einsum(t1.aa, (0, 1), v.aabb.vvvv, (2, 1, 3, 4), (3, 4, 0, 2))
    x284 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x284 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x284 += einsum(x113, (0, 1, 2, 3), (1, 0, 3, 2))
    x285 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x285 += einsum(t1.aa, (0, 1), x284, (2, 3, 0, 4), (2, 3, 4, 1))
    del x284
    x286 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x286 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x286 += einsum(x282, (0, 1, 2, 3), (0, 1, 2, 3))
    x286 += einsum(x283, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x286 += einsum(x285, (0, 1, 2, 3), (1, 0, 2, 3))
    del x285
    x287 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x287 += einsum(t2.bbbb, (0, 1, 2, 3), x286, (3, 4, 5, 6), (0, 1, 4, 2, 5, 6)) * -2.0
    del x286
    x288 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x288 += einsum(f.bb.ov, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (0, 2, 4, 5, 3, 6))
    x289 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x289 += einsum(t2.bbbb, (0, 1, 2, 3), x116, (4, 3, 5, 6), (0, 1, 4, 2, 5, 6)) * -1.0
    del x116
    x290 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x290 += einsum(x288, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    del x288
    x290 += einsum(x289, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    del x289
    x291 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x291 += einsum(t1.bb, (0, 1), x290, (0, 2, 3, 4, 5, 6), (2, 3, 4, 1, 5, 6)) * 2.0
    del x290
    x292 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x292 += einsum(x281, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    del x281
    x292 += einsum(x287, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x287
    x292 += einsum(x291, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x291
    t3new_babbab += einsum(x292, (0, 1, 2, 3, 4, 5), (0, 4, 1, 2, 5, 3)) * -1.0
    t3new_babbab += einsum(x292, (0, 1, 2, 3, 4, 5), (0, 4, 1, 3, 5, 2))
    del x292
    x293 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x293 += einsum(t2.abab, (0, 1, 2, 3), v.aaaa.ovvv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x294 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x294 += einsum(t2.abab, (0, 1, 2, 3), x46, (4, 5, 2, 6), (1, 3, 4, 0, 6, 5))
    del x46
    x295 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x295 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    x296 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x296 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x296 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x296 += einsum(x295, (0, 1, 2, 3), (1, 0, 2, 3))
    del x295
    x297 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x297 += einsum(t2.abab, (0, 1, 2, 3), x296, (4, 5, 6, 2), (1, 3, 4, 5, 0, 6))
    del x296
    x298 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x298 += einsum(t1.aa, (0, 1), x23, (2, 0, 3, 4), (2, 3, 1, 4))
    del x23
    x299 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x299 += einsum(x298, (0, 1, 2, 3), (0, 1, 3, 2))
    del x298
    x299 += einsum(x78, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x78
    x300 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x300 += einsum(t2.abab, (0, 1, 2, 3), x299, (4, 5, 2, 6), (1, 3, 4, 5, 0, 6))
    del x299
    x301 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x301 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x301 += einsum(x64, (0, 1, 2, 3), (3, 1, 0, 2))
    del x64
    x301 += einsum(x37, (0, 1, 2, 3), (2, 1, 0, 3))
    x301 += einsum(x37, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    del x37
    x302 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x302 += einsum(t2.abab, (0, 1, 2, 3), x301, (0, 4, 5, 6), (1, 3, 4, 5, 6, 2))
    del x301
    x303 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x303 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x303 += einsum(x112, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x112
    x304 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x304 += einsum(t2.abab, (0, 1, 2, 3), x303, (3, 4, 5, 6), (1, 4, 5, 6, 0, 2))
    del x303
    x305 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x305 += einsum(t1.bb, (0, 1), x21, (0, 2, 3, 4), (1, 2, 3, 4))
    del x21
    x306 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x306 += einsum(x113, (0, 1, 2, 3), (1, 0, 2, 3))
    del x113
    x306 += einsum(x305, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x305
    x307 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x307 += einsum(t2.abab, (0, 1, 2, 3), x306, (3, 4, 5, 6), (1, 4, 5, 6, 0, 2))
    del x306
    x308 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x308 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x308 += einsum(x121, (0, 1, 2, 3), (1, 0, 3, 2))
    del x121
    x309 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x309 += einsum(t2.abab, (0, 1, 2, 3), x308, (1, 4, 5, 6), (4, 3, 5, 6, 0, 2))
    x310 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x310 += einsum(t1.aa, (0, 1), x118, (2, 3, 4, 1), (2, 3, 0, 4))
    del x118
    x311 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x311 += einsum(t2.abab, (0, 1, 2, 3), x310, (4, 1, 5, 6), (4, 3, 5, 6, 0, 2))
    x312 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x312 += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 1, 4, 3, 2, 5)) * -1.0
    del x297
    x312 += einsum(x300, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x300
    x312 += einsum(x302, (0, 1, 2, 3, 4, 5), (0, 1, 3, 4, 2, 5))
    del x302
    x312 += einsum(x304, (0, 1, 2, 3, 4, 5), (0, 1, 4, 3, 2, 5))
    del x304
    x312 += einsum(x307, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x307
    x312 += einsum(x309, (0, 1, 2, 3, 4, 5), (0, 1, 4, 3, 2, 5)) * -1.0
    del x309
    x312 += einsum(x311, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x311
    x313 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x313 += einsum(t1.aa, (0, 1), x312, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 6, 1))
    del x312
    x314 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x314 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x314 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x314 += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3))
    del x70
    x315 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x315 += einsum(t1.aa, (0, 1), x314, (2, 3, 1, 4), (2, 3, 0, 4))
    del x314
    x316 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x316 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x316 += einsum(x315, (0, 1, 2, 3), (2, 1, 0, 3))
    del x315
    x317 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x317 += einsum(t2.abab, (0, 1, 2, 3), x316, (4, 0, 5, 6), (1, 3, 4, 5, 6, 2))
    del x316
    x318 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x318 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x318 += einsum(x282, (0, 1, 2, 3), (0, 1, 2, 3))
    del x282
    x318 += einsum(x101, (0, 1, 2, 3), (1, 0, 2, 3))
    del x101
    x319 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x319 += einsum(t2.abab, (0, 1, 2, 3), x318, (3, 4, 5, 6), (1, 4, 5, 0, 6, 2))
    del x318
    x320 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x320 += einsum(t1.bb, (0, 1), x71, (0, 2, 3, 4), (1, 2, 3, 4))
    del x71
    x321 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x321 += einsum(x283, (0, 1, 2, 3), (1, 0, 2, 3))
    del x283
    x321 += einsum(x320, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x320
    x322 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x322 += einsum(t2.abab, (0, 1, 2, 3), x321, (3, 4, 5, 6), (1, 4, 5, 0, 6, 2))
    del x321
    x323 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x323 += einsum(t2.abab, (0, 1, 2, 3), x4, (1, 4, 5, 6), (4, 3, 0, 5, 2, 6))
    x324 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x324 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x324 += einsum(x117, (0, 1, 2, 3), (0, 1, 3, 2))
    x325 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x325 += einsum(t1.aa, (0, 1), x324, (2, 3, 1, 4), (2, 3, 0, 4))
    del x324
    x326 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x326 += einsum(t2.abab, (0, 1, 2, 3), x325, (4, 1, 5, 6), (4, 3, 5, 0, 6, 2))
    del x325
    x327 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x327 += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x293
    x327 += einsum(x294, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x294
    x327 += einsum(x313, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x313
    x327 += einsum(x317, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x317
    x327 += einsum(x319, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x319
    x327 += einsum(x322, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x322
    x327 += einsum(x323, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x323
    x327 += einsum(x326, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x326
    t3new_abaaba += einsum(x327, (0, 1, 2, 3, 4, 5), (2, 0, 3, 4, 1, 5)) * -1.0
    t3new_abaaba += einsum(x327, (0, 1, 2, 3, 4, 5), (2, 0, 3, 5, 1, 4))
    t3new_abaaba += einsum(x327, (0, 1, 2, 3, 4, 5), (3, 0, 2, 4, 1, 5))
    t3new_abaaba += einsum(x327, (0, 1, 2, 3, 4, 5), (3, 0, 2, 5, 1, 4)) * -1.0
    del x327
    x328 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x328 += einsum(t1.bb, (0, 1), v.aabb.oovv, (2, 3, 4, 1), (0, 4, 2, 3))
    x329 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x329 += einsum(f.aa.ov, (0, 1), t2.abab, (2, 3, 1, 4), (3, 4, 0, 2))
    x330 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x330 += einsum(t1.bb, (0, 1), x308, (0, 2, 3, 4), (2, 1, 3, 4))
    del x308
    x331 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x331 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x331 += einsum(x328, (0, 1, 2, 3), (0, 1, 3, 2))
    del x328
    x331 += einsum(x329, (0, 1, 2, 3), (0, 1, 2, 3))
    del x329
    x331 += einsum(x330, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x330
    x332 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x332 += einsum(t2.aaaa, (0, 1, 2, 3), x331, (4, 5, 1, 6), (4, 5, 6, 0, 2, 3)) * -2.0
    del x331
    x333 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x333 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x333 += einsum(x99, (0, 1, 2, 3), (0, 1, 2, 3))
    x334 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x334 += einsum(t1.aa, (0, 1), x333, (2, 3, 4, 1), (2, 3, 4, 0))
    del x333
    x335 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x335 += einsum(t1.bb, (0, 1), x310, (2, 0, 3, 4), (2, 1, 3, 4))
    del x310
    x336 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x336 += einsum(x334, (0, 1, 2, 3), (0, 1, 3, 2))
    del x334
    x336 += einsum(x335, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x335
    x337 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x337 += einsum(t2.aaaa, (0, 1, 2, 3), x336, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3)) * -2.0
    del x336
    x338 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x338 += einsum(x230, (0, 1), t3.abaaba, (2, 3, 0, 4, 5, 6), (3, 5, 1, 2, 4, 6)) * -2.0
    del x230
    x339 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x339 += einsum(x332, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x332
    x339 += einsum(x337, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x337
    x339 += einsum(x338, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x338
    t3new_abaaba += einsum(x339, (0, 1, 2, 3, 4, 5), (2, 0, 3, 4, 1, 5))
    t3new_abaaba += einsum(x339, (0, 1, 2, 3, 4, 5), (3, 0, 2, 4, 1, 5)) * -1.0
    del x339
    x340 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x340 += einsum(f.aa.vv, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (3, 6, 2, 4, 0, 5))
    x341 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x341 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x341 += einsum(x117, (0, 1, 2, 3), (1, 0, 3, 2))
    del x117
    x342 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x342 += einsum(t1.bb, (0, 1), x341, (0, 2, 3, 4), (2, 1, 3, 4))
    del x341
    x343 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x343 += einsum(v.aabb.vvov, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x343 += einsum(x259, (0, 1, 2, 3), (0, 1, 2, 3))
    del x259
    x343 += einsum(x138, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x138
    x343 += einsum(x342, (0, 1, 2, 3), (0, 1, 3, 2))
    del x342
    x344 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x344 += einsum(t2.aaaa, (0, 1, 2, 3), x343, (4, 5, 3, 6), (4, 5, 0, 1, 6, 2)) * -2.0
    del x343
    x345 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x345 += einsum(f.aa.ov, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (3, 6, 0, 2, 4, 5))
    x346 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x346 += einsum(t1.bb, (0, 1), x4, (0, 2, 3, 4), (2, 1, 3, 4))
    del x4
    x347 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x347 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x347 += einsum(x99, (0, 1, 2, 3), (0, 1, 2, 3))
    del x99
    x347 += einsum(x346, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x346
    x348 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x348 += einsum(t2.aaaa, (0, 1, 2, 3), x347, (4, 5, 6, 3), (4, 5, 6, 0, 1, 2)) * -1.0
    del x347
    x349 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x349 += einsum(x345, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x345
    x349 += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x348
    x350 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x350 += einsum(t1.aa, (0, 1), x349, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x349
    x351 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x351 += einsum(x340, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -2.0
    del x340
    x351 += einsum(x344, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x344
    x351 += einsum(x350, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x350
    t3new_abaaba += einsum(x351, (0, 1, 2, 3, 4, 5), (3, 0, 2, 4, 1, 5))
    t3new_abaaba += einsum(x351, (0, 1, 2, 3, 4, 5), (3, 0, 2, 5, 1, 4)) * -1.0
    del x351
    x352 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x352 += einsum(f.bb.ov, (0, 1), t1.bb, (0, 2), (1, 2))
    x353 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x353 += einsum(f.bb.vv, (0, 1), (0, 1))
    x353 += einsum(x352, (0, 1), (0, 1)) * -1.0
    del x352
    t3new_abaaba += einsum(x353, (0, 1), t3.abaaba, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * 2.0
    x354 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x354 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    x355 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x355 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    x356 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x356 += einsum(t1.bb, (0, 1), x355, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x355
    x357 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x357 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x357 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x358 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x358 += einsum(t2.bbbb, (0, 1, 2, 3), x357, (4, 5, 3, 6), (4, 5, 0, 1, 6, 2)) * -1.0
    x359 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x359 += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x356
    x359 += einsum(x358, (0, 1, 2, 3, 4, 5), (3, 2, 1, 0, 5, 4))
    del x358
    x360 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x360 += einsum(t1.bb, (0, 1), x359, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x359
    x361 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x361 += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    del x354
    x361 += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    del x360
    t3new_bbbbbb = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new_bbbbbb += einsum(x361, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x361
    x362 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x362 += einsum(x353, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6)) * 6.0
    del x353
    t3new_bbbbbb += einsum(x362, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new_bbbbbb += einsum(x362, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x362, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x362
    x363 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x363 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.oooo, (4, 5, 6, 1), (0, 4, 5, 6, 2, 3))
    x364 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x364 += einsum(t2.bbbb, (0, 1, 2, 3), x157, (4, 5, 1, 6), (5, 4, 0, 6, 2, 3))
    del x157
    x365 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x365 += einsum(x171, (0, 1, 2, 3), (0, 2, 1, 3))
    x365 += einsum(x171, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    del x171
    x366 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x366 += einsum(t2.bbbb, (0, 1, 2, 3), x365, (4, 1, 5, 6), (4, 5, 6, 0, 2, 3)) * -1.0
    del x365
    x367 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x367 += einsum(x363, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    del x363
    x367 += einsum(x364, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x364
    x367 += einsum(x366, (0, 1, 2, 3, 4, 5), (0, 3, 2, 1, 5, 4)) * -1.0
    del x366
    x368 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x368 += einsum(t1.bb, (0, 1), x367, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x367
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new_bbbbbb += einsum(x368, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    del x368
    x369 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x369 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 5, 6), (0, 4, 5, 2, 3, 6))
    x370 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x370 += einsum(t1.bb, (0, 1), x163, (2, 3, 1, 4), (0, 2, 3, 4))
    del x163
    x371 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x371 += einsum(t2.bbbb, (0, 1, 2, 3), x370, (4, 5, 1, 6), (4, 5, 0, 2, 3, 6))
    del x370
    x372 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x372 += einsum(t1.bb, (0, 1), x357, (2, 3, 1, 4), (2, 3, 0, 4))
    del x357
    x373 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x373 += einsum(t2.bbbb, (0, 1, 2, 3), x372, (4, 1, 5, 6), (5, 4, 0, 6, 2, 3)) * -2.0
    del x372
    x374 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x374 += einsum(x369, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    del x369
    x374 += einsum(x371, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    del x371
    x374 += einsum(x373, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    del x373
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x374, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x374
    x375 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x375 += einsum(x278, (0, 1), t3.bbbbbb, (2, 3, 0, 4, 5, 6), (1, 2, 3, 4, 5, 6)) * 6.0
    del x278
    t3new_bbbbbb += einsum(x375, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x375, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x375, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    del x375
    x376 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x376 += einsum(t2.bbbb, (0, 1, 2, 3), x142, (4, 5, 3, 6), (4, 0, 1, 2, 6, 5))
    del x142
    x377 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x377 += einsum(t2.bbbb, (0, 1, 2, 3), x1, (4, 5, 6, 3), (4, 0, 1, 6, 5, 2))
    del x1
    x378 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x378 += einsum(t1.bb, (0, 1), x377, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x377
    x379 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x379 += einsum(t2.bbbb, (0, 1, 2, 3), x169, (4, 5, 6, 3), (5, 4, 0, 1, 6, 2))
    del x169
    x380 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x380 += einsum(x378, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    del x378
    x380 += einsum(x379, (0, 1, 2, 3, 4, 5), (0, 3, 2, 1, 5, 4)) * -1.0
    del x379
    x381 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x381 += einsum(t1.bb, (0, 1), x380, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x380
    x382 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x382 += einsum(x376, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    del x376
    x382 += einsum(x381, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    del x381
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new_bbbbbb += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x382
    x383 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x383 += einsum(f.bb.ov, (0, 1), t2.bbbb, (2, 3, 4, 1), (0, 2, 3, 4))
    x384 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x384 += einsum(t2.bbbb, (0, 1, 2, 3), x383, (1, 4, 5, 6), (5, 4, 0, 6, 2, 3)) * -1.0
    del x383
    t3new_bbbbbb += einsum(x384, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x384, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x384, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x384, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x384, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x384, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x384, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x384, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x384, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    del x384

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb
    t3new.aaaaaa = t3new_aaaaaa
    t3new.abaaba = t3new_abaaba
    t3new.babbab = t3new_babbab
    t3new.bbbbbb = t3new_bbbbbb

    return {"t1new": t1new, "t2new": t2new, "t3new": t3new}

