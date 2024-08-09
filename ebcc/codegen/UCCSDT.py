# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 2), ()) * -1.0
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 2, 1, 3), ())
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x0 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x0 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x1 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x1 += einsum(f.bb.ov, (0, 1), (0, 1))
    x1 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    x1 += einsum(t1.bb, (0, 1), x0, (0, 2, 1, 3), (2, 3)) * -0.5
    del x0
    e_cc += einsum(t1.bb, (0, 1), x1, (0, 1), ())
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x3 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x3 += einsum(f.aa.ov, (0, 1), (0, 1))
    x3 += einsum(t1.aa, (0, 1), x2, (0, 2, 1, 3), (2, 3)) * -0.5
    del x2
    e_cc += einsum(t1.aa, (0, 1), x3, (0, 1), ())
    del x3

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    t1new = Namespace()
    t2new = Namespace()
    t3new = Namespace()

    # T amplitudes
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    t1new_bb += einsum(f.bb.ov, (0, 1), (0, 1))
    t1new_bb += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 0, 2, 5, 1, 3), (4, 5)) * 2.0
    t1new_bb += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb, (4, 0, 2, 5, 1, 3), (4, 5)) * 3.0
    t1new_bb += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.abaaba, (0, 4, 2, 1, 5, 3), (4, 5))
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    t1new_aa += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 2, 0, 5, 3, 1), (4, 5)) * 2.0
    t1new_aa += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa, (4, 0, 2, 5, 1, 3), (4, 5)) * 3.0
    t1new_aa += einsum(f.aa.ov, (0, 1), (0, 1))
    t1new_aa += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.babbab, (0, 4, 2, 3, 5, 1), (4, 5)) * -1.0
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 2, 5, 3), (0, 1, 4, 5)) * 2.0
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
    t2new_abab += einsum(x2, (0, 1, 2, 3), t3.babbab, (1, 4, 2, 5, 6, 3), (4, 0, 6, 5)) * 2.0
    del x2
    x3 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x3 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (0, 4, 2, 3))
    x4 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x4 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x4 += einsum(x3, (0, 1, 2, 3), (1, 0, 2, 3))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x4, (1, 4, 0, 2), (4, 3)) * -1.0
    t2new_abab += einsum(x4, (0, 1, 2, 3), t3.abaaba, (4, 0, 2, 5, 6, 3), (4, 1, 5, 6)) * -2.0
    x5 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x5 += einsum(t2.abab, (0, 1, 2, 3), (1, 3, 0, 2))
    x5 += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (2, 3, 0, 1))
    t1new_bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), x5, (4, 3, 0, 1), (4, 2))
    t1new_aa += einsum(v.aabb.vvov, (0, 1, 2, 3), x5, (2, 3, 4, 1), (4, 0))
    x6 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x6 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x6 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_bb += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x6, (0, 4, 1, 3), (4, 2)) * 2.0
    x7 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x7 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x7 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x8 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x8 += einsum(t1.bb, (0, 1), x7, (0, 2, 1, 3), (2, 3))
    x9 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x9 += einsum(f.bb.ov, (0, 1), (0, 1))
    x9 += einsum(x0, (0, 1), (0, 1))
    x9 += einsum(x8, (0, 1), (0, 1)) * -1.0
    del x8
    t1new_bb += einsum(x9, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_aa += einsum(x9, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    t2new_aaaa += einsum(x9, (0, 1), t3.abaaba, (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    t2new_abab += einsum(x9, (0, 1), t3.babbab, (2, 3, 0, 4, 5, 1), (3, 2, 5, 4)) * 2.0
    t2new_bbbb += einsum(x9, (0, 1), t3.bbbbbb, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    x10 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x10 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    t1new_aa += einsum(x10, (0, 1), (0, 1))
    x11 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x11 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x11 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x12 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x12 += einsum(t1.aa, (0, 1), x11, (0, 2, 1, 3), (2, 3))
    x13 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x13 += einsum(f.aa.ov, (0, 1), (0, 1))
    x13 += einsum(x10, (0, 1), (0, 1))
    del x10
    x13 += einsum(x12, (0, 1), (0, 1)) * -1.0
    del x12
    t1new_bb += einsum(x13, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    t1new_aa += einsum(x13, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    t2new_aaaa += einsum(x13, (0, 1), t3.aaaaaa, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    t2new_abab += einsum(x13, (0, 1), t3.abaaba, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 2.0
    t2new_bbbb += einsum(x13, (0, 1), t3.babbab, (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    x14 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x14 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x14 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_bb += einsum(t1.bb, (0, 1), x14, (0, 2, 1, 3), (2, 3)) * -1.0
    x15 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x15 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3))
    x16 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x16 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 1, 3), (0, 4))
    x17 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x17 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x18 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x18 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x18 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x19 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x19 += einsum(t1.bb, (0, 1), x18, (2, 3, 0, 1), (2, 3))
    x20 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x20 += einsum(t1.bb, (0, 1), x9, (2, 1), (2, 0))
    x21 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x21 += einsum(f.bb.oo, (0, 1), (0, 1))
    x21 += einsum(x15, (0, 1), (1, 0))
    x21 += einsum(x16, (0, 1), (1, 0)) * 2.0
    x21 += einsum(x17, (0, 1), (1, 0))
    x21 += einsum(x19, (0, 1), (1, 0)) * -1.0
    x21 += einsum(x20, (0, 1), (0, 1))
    t1new_bb += einsum(t1.bb, (0, 1), x21, (0, 2), (2, 1)) * -1.0
    t2new_abab += einsum(x21, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    t3new_abaaba = np.zeros((nocc[0], nocc[1], nocc[0], nvir[0], nvir[1], nvir[0]), dtype=types[float])
    t3new_abaaba += einsum(x21, (0, 1), t3.abaaba, (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -2.0
    del x21
    x22 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x22 += einsum(f.bb.vv, (0, 1), (0, 1))
    x22 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (0, 2, 3, 1), (2, 3)) * -1.0
    t1new_bb += einsum(t1.bb, (0, 1), x22, (1, 2), (0, 2))
    del x22
    x23 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x23 += einsum(t1.aa, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (3, 4, 0, 2))
    x24 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x24 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x24 += einsum(x23, (0, 1, 2, 3), (0, 1, 3, 2))
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x24, (1, 3, 0, 4), (4, 2)) * -1.0
    t2new_abab += einsum(x24, (0, 1, 2, 3), t3.babbab, (4, 2, 0, 5, 6, 1), (3, 4, 6, 5)) * -2.0
    x25 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x25 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x26 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x26 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x26 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), x26, (4, 0, 1, 3), (4, 2)) * 2.0
    t2new_abab += einsum(x26, (0, 1, 2, 3), t3.abaaba, (1, 4, 2, 5, 6, 3), (0, 4, 5, 6)) * 2.0
    del x26
    x27 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x27 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x27 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x27, (0, 4, 3, 1), (4, 2)) * -2.0
    del x27
    x28 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x28 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x28 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_aa += einsum(t1.aa, (0, 1), x28, (0, 2, 1, 3), (2, 3)) * -1.0
    x29 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x29 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3))
    x30 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x30 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x31 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x31 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x32 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x32 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x32 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x33 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x33 += einsum(t1.aa, (0, 1), x32, (2, 3, 0, 1), (2, 3))
    x34 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x34 += einsum(t1.aa, (0, 1), x13, (2, 1), (2, 0))
    x35 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x35 += einsum(f.aa.oo, (0, 1), (0, 1))
    x35 += einsum(x29, (0, 1), (1, 0))
    x35 += einsum(x30, (0, 1), (1, 0))
    x35 += einsum(x31, (0, 1), (1, 0)) * 2.0
    x35 += einsum(x33, (0, 1), (1, 0)) * -1.0
    x35 += einsum(x34, (0, 1), (0, 1))
    t1new_aa += einsum(t1.aa, (0, 1), x35, (0, 2), (2, 1)) * -1.0
    t2new_abab += einsum(x35, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    t3new_babbab = np.zeros((nocc[1], nocc[0], nocc[1], nvir[1], nvir[0], nvir[1]), dtype=types[float])
    t3new_babbab += einsum(x35, (0, 1), t3.babbab, (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -2.0
    del x35
    x36 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x36 += einsum(f.aa.vv, (0, 1), (0, 1))
    x36 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_aa += einsum(t1.aa, (0, 1), x36, (1, 2), (0, 2))
    del x36
    x37 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x37 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x38 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x38 += einsum(t1.aa, (0, 1), v.aabb.vvov, (2, 1, 3, 4), (3, 4, 0, 2))
    t2new_abab += einsum(x38, (0, 1, 2, 3), (2, 0, 3, 1))
    x39 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x39 += einsum(t2.abab, (0, 1, 2, 3), x38, (1, 3, 4, 5), (4, 0, 2, 5))
    x40 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x40 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x40 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -0.5
    x41 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x41 += einsum(x28, (0, 1, 2, 3), x40, (0, 4, 2, 5), (1, 4, 3, 5)) * 2.0
    x42 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x42 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ovov, (1, 3, 4, 5), (4, 5, 0, 2))
    x43 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x43 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x43 += einsum(x42, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x44 += einsum(t2.abab, (0, 1, 2, 3), x43, (1, 3, 4, 5), (4, 0, 5, 2))
    del x43
    x45 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x45 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x45 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x46 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x46 += einsum(t1.aa, (0, 1), x45, (2, 1, 3, 4), (2, 0, 3, 4))
    x47 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x47 += einsum(t2.aaaa, (0, 1, 2, 3), x46, (1, 4, 5, 3), (0, 4, 2, 5)) * -2.0
    x48 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x48 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x49 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x49 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x50 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x50 += einsum(t1.aa, (0, 1), x49, (2, 3, 4, 0), (2, 4, 3, 1))
    x51 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x51 += einsum(t1.aa, (0, 1), x37, (2, 3, 1, 4), (0, 2, 3, 4))
    x52 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x52 += einsum(t2.abab, (0, 1, 2, 3), x23, (1, 3, 4, 5), (4, 0, 5, 2))
    x53 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x53 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x53 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x54 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x54 += einsum(t2.aaaa, (0, 1, 2, 3), x53, (4, 5, 1, 3), (0, 4, 5, 2)) * 2.0
    del x53
    x55 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x55 += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x55 += einsum(x25, (0, 1, 2, 3), (0, 2, 1, 3))
    x56 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x56 += einsum(t2.aaaa, (0, 1, 2, 3), x55, (4, 1, 5, 3), (0, 4, 5, 2)) * 2.0
    x57 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x57 += einsum(x48, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x57 += einsum(x50, (0, 1, 2, 3), (0, 2, 1, 3))
    x57 += einsum(x51, (0, 1, 2, 3), (0, 2, 1, 3))
    del x51
    x57 += einsum(x52, (0, 1, 2, 3), (0, 2, 1, 3))
    x57 += einsum(x54, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x54
    x57 += einsum(x56, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    x58 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x58 += einsum(t1.aa, (0, 1), x57, (2, 0, 3, 4), (2, 3, 4, 1))
    del x57
    x59 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x59 += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3))
    x59 += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3))
    del x39
    x59 += einsum(x41, (0, 1, 2, 3), (1, 0, 3, 2))
    del x41
    x59 += einsum(x44, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x44
    x59 += einsum(x47, (0, 1, 2, 3), (1, 0, 2, 3))
    del x47
    x59 += einsum(x58, (0, 1, 2, 3), (0, 1, 3, 2))
    del x58
    t2new_aaaa += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x59, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x59, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x59
    x60 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x60 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x61 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x61 += einsum(t2.abab, (0, 1, 2, 3), x7, (1, 4, 3, 5), (4, 5, 0, 2))
    x62 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x62 += einsum(t2.abab, (0, 1, 2, 3), x61, (1, 3, 4, 5), (4, 0, 5, 2)) * -1.0
    x63 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x63 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x63 += einsum(x60, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x60
    x63 += einsum(x62, (0, 1, 2, 3), (1, 0, 3, 2))
    del x62
    t2new_aaaa += einsum(x63, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3))
    del x63
    x64 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x64 += einsum(t1.aa, (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x65 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x65 += einsum(t1.aa, (0, 1), x64, (2, 3, 1, 4), (0, 2, 3, 4))
    x66 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x66 += einsum(v.aabb.vvov, (0, 1, 2, 3), t3.abaaba, (4, 2, 5, 6, 3, 1), (4, 5, 6, 0))
    x67 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x67 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 3, 1), (4, 5, 6, 2)) * -1.0
    x68 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x68 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x68 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x69 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x69 += einsum(t2.aaaa, (0, 1, 2, 3), x68, (1, 4, 5, 3), (0, 4, 2, 5))
    x70 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x70 += einsum(t2.aaaa, (0, 1, 2, 3), x69, (4, 1, 5, 3), (0, 4, 2, 5)) * -4.0
    del x69
    x71 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x71 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 1), (2, 3))
    x72 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x72 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    x73 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x73 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x74 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x74 += einsum(t1.aa, (0, 1), x45, (0, 1, 2, 3), (2, 3))
    del x45
    x75 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x75 += einsum(x71, (0, 1), (1, 0)) * -1.0
    x75 += einsum(x72, (0, 1), (1, 0))
    x75 += einsum(x73, (0, 1), (1, 0)) * 2.0
    x75 += einsum(x74, (0, 1), (1, 0)) * -1.0
    x76 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x76 += einsum(x75, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x75
    x77 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x77 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x78 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x78 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 2, 5, 6, 3, 1), (4, 5, 0, 6))
    x79 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x79 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 2, 6, 3, 1), (4, 5, 0, 6)) * -1.0
    x80 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x80 += einsum(x13, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4)) * -2.0
    x81 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x81 += einsum(t1.aa, (0, 1), x25, (2, 3, 4, 1), (2, 0, 4, 3))
    x82 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x82 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x82 += einsum(x81, (0, 1, 2, 3), (3, 1, 2, 0))
    x83 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x83 += einsum(t1.aa, (0, 1), x82, (0, 2, 3, 4), (2, 3, 4, 1))
    del x82
    x84 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x84 += einsum(x77, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    x84 += einsum(x78, (0, 1, 2, 3), (2, 1, 0, 3)) * 2.0
    x84 += einsum(x79, (0, 1, 2, 3), (2, 1, 0, 3)) * 6.0
    x84 += einsum(x80, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x84 += einsum(x83, (0, 1, 2, 3), (1, 0, 2, 3))
    del x83
    x85 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x85 += einsum(t1.aa, (0, 1), x84, (0, 2, 3, 4), (2, 3, 4, 1))
    del x84
    x86 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x86 += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3))
    del x65
    x86 += einsum(x66, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x66
    x86 += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3)) * -6.0
    del x67
    x86 += einsum(x70, (0, 1, 2, 3), (1, 0, 3, 2))
    del x70
    x86 += einsum(x76, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x76
    x86 += einsum(x85, (0, 1, 2, 3), (1, 0, 3, 2))
    del x85
    t2new_aaaa += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x86, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x86
    x87 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x87 += einsum(v.aabb.ooov, (0, 1, 2, 3), t3.abaaba, (4, 2, 1, 5, 3, 6), (4, 0, 5, 6))
    x88 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x88 += einsum(v.aaaa.ooov, (0, 1, 2, 3), t3.aaaaaa, (4, 1, 2, 5, 6, 3), (4, 0, 5, 6))
    x89 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x89 += einsum(t2.aaaa, (0, 1, 2, 3), x49, (4, 5, 0, 1), (4, 5, 2, 3))
    x90 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x90 += einsum(x23, (0, 1, 2, 3), t3.abaaba, (4, 0, 3, 5, 1, 6), (2, 4, 5, 6))
    x91 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x91 += einsum(x25, (0, 1, 2, 3), t3.aaaaaa, (4, 1, 2, 5, 6, 3), (0, 4, 5, 6)) * -1.0
    x92 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x92 += einsum(f.aa.oo, (0, 1), (0, 1))
    x92 += einsum(x34, (0, 1), (1, 0))
    x93 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x93 += einsum(x92, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x92
    x94 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x94 += einsum(x29, (0, 1), (1, 0))
    x94 += einsum(x30, (0, 1), (1, 0))
    x94 += einsum(x31, (0, 1), (1, 0)) * 2.0
    x94 += einsum(x33, (0, 1), (1, 0)) * -1.0
    x95 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x95 += einsum(x94, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    del x94
    x96 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x96 += einsum(x87, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x87
    x96 += einsum(x88, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.0
    del x88
    x96 += einsum(x89, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x89
    x96 += einsum(x90, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x90
    x96 += einsum(x91, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.0
    del x91
    x96 += einsum(x93, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x93
    x96 += einsum(x95, (0, 1, 2, 3), (0, 1, 3, 2))
    del x95
    t2new_aaaa += einsum(x96, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x96, (0, 1, 2, 3), (1, 0, 2, 3))
    del x96
    x97 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x97 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new_aaaa += einsum(x97, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x97, (0, 1, 2, 3), (1, 0, 2, 3))
    del x97
    x98 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x98 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x98 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3))
    x99 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x99 += einsum(v.aaaa.ovov, (0, 1, 2, 3), x98, (4, 5, 1, 3), (0, 2, 4, 5))
    x100 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x100 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x100 += einsum(x99, (0, 1, 2, 3), (1, 3, 0, 2))
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x100, (0, 4, 1, 5), (4, 5, 2, 3)) * 2.0
    del x100
    x101 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x101 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x102 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x102 += einsum(t1.aa, (0, 1), x101, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    x102 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x102 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    t2new_aaaa += einsum(t1.aa, (0, 1), x102, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x102
    x103 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x103 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 4), (1, 4, 2, 3))
    x104 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x104 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (2, 3, 0, 1))
    x104 += einsum(x103, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_abab += einsum(x104, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 6, 0, 3), (4, 5, 6, 1)) * 2.0
    x105 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x105 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 2, 3, 4), (3, 4, 1, 2))
    x106 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x106 += einsum(v.aabb.vvov, (0, 1, 2, 3), (2, 3, 0, 1))
    x106 += einsum(x105, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_abab += einsum(x106, (0, 1, 2, 3), t3.babbab, (4, 5, 0, 6, 2, 1), (5, 4, 3, 6)) * 2.0
    x107 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x107 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 0, 4), (2, 1, 3, 4))
    x108 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x108 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x108 += einsum(x107, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab += einsum(x108, (0, 1, 2, 3), t3.babbab, (4, 5, 0, 2, 6, 3), (5, 4, 6, 1)) * 2.0
    del x108
    x109 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x109 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 0, 4), (2, 1, 3, 4))
    x110 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x110 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x110 += einsum(x109, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab += einsum(x110, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 2, 6, 3), (4, 5, 1, 6)) * 2.0
    del x110
    x111 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x111 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 4, 3, 5))
    x112 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x112 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x112 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x113 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x113 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x113 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -0.5
    x114 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x114 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x114 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x115 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x115 += einsum(t1.bb, (0, 1), x114, (2, 3, 1, 4), (2, 0, 3, 4))
    x116 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x116 += einsum(t1.bb, (0, 1), x18, (2, 3, 0, 4), (2, 3, 4, 1))
    x117 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x117 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x117 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x117 += einsum(x111, (0, 1, 2, 3), (1, 0, 3, 2))
    x117 += einsum(x112, (0, 1, 2, 3), x113, (0, 4, 3, 5), (1, 4, 2, 5)) * -2.0
    x117 += einsum(x115, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x117 += einsum(x116, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x117, (1, 4, 3, 5), (0, 4, 2, 5))
    del x117
    x118 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x118 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x118 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x119 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x119 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x119 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x120 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x120 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x120 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x120 += einsum(x11, (0, 1, 2, 3), x40, (0, 4, 3, 5), (1, 4, 2, 5)) * -2.0
    x120 += einsum(t1.aa, (0, 1), x118, (2, 1, 3, 4), (2, 0, 4, 3)) * -1.0
    x120 += einsum(t1.aa, (0, 1), x119, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    del x119
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x120, (0, 4, 2, 5), (4, 1, 5, 3)) * -1.0
    del x120
    x121 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x121 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 4), (1, 4, 2, 3))
    x122 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x122 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (3, 4, 0, 2))
    x123 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x123 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x123 += einsum(x121, (0, 1, 2, 3), (1, 0, 3, 2))
    x123 += einsum(x122, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x123 += einsum(v.aabb.ovov, (0, 1, 2, 3), x5, (2, 4, 5, 1), (3, 4, 0, 5))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x123, (3, 4, 0, 5), (5, 1, 2, 4))
    del x123
    x124 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x124 += einsum(t1.aa, (0, 1), v.aabb.ooov, (2, 0, 3, 4), (3, 4, 2, 1))
    x125 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x125 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x125 += einsum(x124, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x125 += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3))
    x125 += einsum(v.aabb.ovov, (0, 1, 2, 3), x40, (0, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    t2new_abab += einsum(t2.bbbb, (0, 1, 2, 3), x125, (1, 3, 4, 5), (4, 0, 5, 2)) * 2.0
    del x125
    x126 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x126 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x126 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -1.0
    x127 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x127 += einsum(t1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (0, 4, 2, 3))
    x128 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x128 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x128 += einsum(x127, (0, 1, 2, 3), (0, 1, 2, 3))
    x128 += einsum(t1.bb, (0, 1), x4, (0, 2, 3, 4), (2, 1, 3, 4)) * -1.0
    t2new_abab += einsum(x126, (0, 1, 2, 3), x128, (4, 5, 0, 2), (1, 4, 3, 5))
    del x126, x128
    x129 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x129 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 4, 1), (0, 4, 2, 3))
    x130 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x130 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x130 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    x131 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x131 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x131 += einsum(x129, (0, 1, 2, 3), (1, 0, 3, 2))
    x131 += einsum(t1.aa, (0, 1), x130, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x131, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    del x131
    x132 = np.zeros((nvir[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x132 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 4), (1, 4, 2, 3))
    x133 = np.zeros((nvir[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x133 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 2, 3, 4), (3, 4, 1, 2))
    x134 = np.zeros((nvir[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x134 += einsum(v.aabb.vvvv, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x134 += einsum(x132, (0, 1, 2, 3), (1, 0, 3, 2))
    x134 += einsum(x133, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x134, (3, 4, 2, 5), (0, 1, 5, 4)) * -1.0
    del x134
    x135 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x135 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 4, 1), (0, 4, 2, 3))
    x136 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x136 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x137 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x137 += einsum(v.aabb.ovov, (0, 1, 2, 3), x5, (4, 3, 5, 1), (4, 2, 5, 0))
    x138 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x138 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x138 += einsum(x135, (0, 1, 2, 3), (1, 0, 3, 2))
    x138 += einsum(x136, (0, 1, 2, 3), (1, 0, 3, 2))
    x138 += einsum(x137, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x138, (1, 4, 0, 5), (5, 4, 2, 3))
    del x138
    x139 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x139 += einsum(t1.aa, (0, 1), x13, (0, 2), (2, 1))
    x140 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x140 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x140 += einsum(x71, (0, 1), (1, 0)) * -1.0
    x140 += einsum(x72, (0, 1), (1, 0))
    x140 += einsum(x73, (0, 1), (1, 0)) * 2.0
    x140 += einsum(x74, (0, 1), (1, 0)) * -1.0
    del x74
    x140 += einsum(x139, (0, 1), (0, 1))
    t2new_abab += einsum(x140, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    t3new_babbab += einsum(x140, (0, 1), t3.babbab, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -2.0
    del x140
    x141 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x141 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 1, 2, 3), (2, 3))
    x142 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x142 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 4, 1, 3), (2, 4))
    x143 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x143 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    x144 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x144 += einsum(t1.bb, (0, 1), x114, (0, 2, 1, 3), (2, 3))
    del x114
    x145 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x145 += einsum(t1.bb, (0, 1), x9, (0, 2), (2, 1))
    x146 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x146 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x146 += einsum(x141, (0, 1), (1, 0)) * -1.0
    x146 += einsum(x142, (0, 1), (1, 0)) * 2.0
    x146 += einsum(x143, (0, 1), (1, 0))
    x146 += einsum(x144, (0, 1), (0, 1)) * -1.0
    x146 += einsum(x145, (0, 1), (0, 1))
    t2new_abab += einsum(x146, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    t3new_abaaba += einsum(x146, (0, 1), t3.abaaba, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -2.0
    del x146
    x147 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x147 += einsum(t1.aa, (0, 1), v.aabb.vvoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x148 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x148 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x148 += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3))
    x149 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x149 += einsum(t1.bb, (0, 1), x148, (2, 1, 3, 4), (2, 0, 3, 4))
    del x148
    x150 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x150 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (1, 5, 0, 4))
    x151 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x151 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x151 += einsum(x135, (0, 1, 2, 3), (0, 1, 3, 2))
    x151 += einsum(x150, (0, 1, 2, 3), (0, 1, 3, 2))
    x152 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x152 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x152 += einsum(x147, (0, 1, 2, 3), (1, 0, 2, 3))
    x152 += einsum(x149, (0, 1, 2, 3), (0, 1, 2, 3))
    x152 += einsum(t1.aa, (0, 1), x151, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    del x151
    t2new_abab += einsum(t1.bb, (0, 1), x152, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    del x152
    x153 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x153 += einsum(t1.aa, (0, 1), v.aabb.vvvv, (2, 1, 3, 4), (3, 4, 0, 2))
    x154 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x154 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (2, 3, 0, 1))
    x154 += einsum(x153, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_abab += einsum(t1.bb, (0, 1), x154, (1, 2, 3, 4), (3, 0, 4, 2))
    del x154
    x155 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x155 += einsum(t1.bb, (0, 1), v.aabb.oovv, (2, 3, 4, 1), (0, 4, 2, 3))
    x156 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x156 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x156 += einsum(x155, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_abab += einsum(t1.aa, (0, 1), x156, (2, 3, 0, 4), (4, 2, 1, 3)) * -1.0
    del x156
    x157 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x157 += einsum(t1.bb, (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x158 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x158 += einsum(t1.bb, (0, 1), x157, (2, 3, 1, 4), (0, 2, 3, 4))
    x159 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x159 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 0, 6, 1, 3), (4, 5, 6, 2))
    x160 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x160 += einsum(v.aabb.ovvv, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 1, 3), (4, 5, 6, 2))
    x161 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x161 += einsum(t2.bbbb, (0, 1, 2, 3), x7, (1, 4, 3, 5), (4, 0, 5, 2))
    x162 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x162 += einsum(t2.bbbb, (0, 1, 2, 3), x161, (1, 4, 3, 5), (4, 0, 5, 2)) * -4.0
    del x161
    x163 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x163 += einsum(t2.abab, (0, 1, 2, 3), x11, (0, 4, 2, 5), (1, 3, 4, 5))
    x164 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x164 += einsum(t2.abab, (0, 1, 2, 3), x163, (4, 5, 0, 2), (4, 1, 5, 3)) * -1.0
    x165 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x165 += einsum(x141, (0, 1), (1, 0)) * -1.0
    x165 += einsum(x142, (0, 1), (1, 0)) * 2.0
    x165 += einsum(x143, (0, 1), (1, 0))
    x165 += einsum(x144, (0, 1), (0, 1)) * -1.0
    del x144
    x166 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x166 += einsum(x165, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 1, 4)) * -2.0
    del x165
    x167 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x167 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x168 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x168 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 2, 6, 3, 1), (4, 5, 0, 6)) * -1.0
    x169 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x169 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 1, 3), (4, 5, 2, 6))
    x170 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x170 += einsum(x9, (0, 1), t2.bbbb, (2, 3, 4, 1), (0, 2, 3, 4)) * -2.0
    x171 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x171 += einsum(t1.bb, (0, 1), x1, (2, 3, 4, 1), (2, 0, 4, 3))
    x172 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x172 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x172 += einsum(x171, (0, 1, 2, 3), (3, 1, 2, 0))
    x173 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x173 += einsum(t1.bb, (0, 1), x172, (0, 2, 3, 4), (2, 3, 4, 1))
    del x172
    x174 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x174 += einsum(x167, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    x174 += einsum(x168, (0, 1, 2, 3), (2, 1, 0, 3)) * 6.0
    x174 += einsum(x169, (0, 1, 2, 3), (2, 1, 0, 3)) * 2.0
    x174 += einsum(x170, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x174 += einsum(x173, (0, 1, 2, 3), (1, 0, 2, 3))
    del x173
    x175 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x175 += einsum(t1.bb, (0, 1), x174, (0, 2, 3, 4), (2, 3, 4, 1))
    del x174
    x176 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x176 += einsum(x158, (0, 1, 2, 3), (0, 1, 2, 3))
    del x158
    x176 += einsum(x159, (0, 1, 2, 3), (0, 1, 2, 3)) * -6.0
    del x159
    x176 += einsum(x160, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x160
    x176 += einsum(x162, (0, 1, 2, 3), (0, 1, 2, 3))
    del x162
    x176 += einsum(x164, (0, 1, 2, 3), (0, 1, 2, 3))
    del x164
    x176 += einsum(x166, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x166
    x176 += einsum(x175, (0, 1, 2, 3), (1, 0, 3, 2))
    del x175
    t2new_bbbb += einsum(x176, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x176, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x176
    x177 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x177 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x178 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x178 += einsum(t2.abab, (0, 1, 2, 3), x127, (4, 5, 0, 2), (4, 1, 3, 5))
    x179 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x179 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x179 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x179 += einsum(x111, (0, 1, 2, 3), (1, 0, 3, 2))
    x180 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x180 += einsum(t2.bbbb, (0, 1, 2, 3), x179, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    del x179
    x181 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x181 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x181 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x182 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x182 += einsum(t1.bb, (0, 1), x181, (2, 1, 3, 4), (2, 0, 3, 4))
    del x181
    x183 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x183 += einsum(t2.bbbb, (0, 1, 2, 3), x182, (1, 4, 5, 3), (4, 0, 5, 2)) * -2.0
    x184 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x184 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x185 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x185 += einsum(t1.bb, (0, 1), x184, (2, 3, 4, 0), (2, 4, 3, 1))
    x186 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x186 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 5), (1, 4, 5, 3))
    x187 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x187 += einsum(t2.abab, (0, 1, 2, 3), x3, (4, 5, 0, 2), (4, 1, 5, 3))
    x188 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x188 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x188 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x189 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x189 += einsum(t2.bbbb, (0, 1, 2, 3), x188, (1, 4, 5, 3), (4, 5, 0, 2)) * 2.0
    x190 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x190 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x190 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3))
    x191 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x191 += einsum(t2.bbbb, (0, 1, 2, 3), x190, (4, 1, 5, 3), (4, 5, 0, 2)) * 2.0
    x192 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x192 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x192 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x192 += einsum(x177, (0, 1, 2, 3), (0, 1, 2, 3))
    x193 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x193 += einsum(t1.bb, (0, 1), x192, (2, 3, 1, 4), (2, 3, 0, 4))
    x194 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x194 += einsum(x185, (0, 1, 2, 3), (0, 2, 1, 3))
    x194 += einsum(x186, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x194 += einsum(x187, (0, 1, 2, 3), (0, 2, 1, 3))
    x194 += einsum(x189, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x189
    x194 += einsum(x191, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x191
    x194 += einsum(x193, (0, 1, 2, 3), (2, 1, 0, 3))
    x195 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x195 += einsum(t1.bb, (0, 1), x194, (2, 0, 3, 4), (2, 3, 4, 1))
    del x194
    x196 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x196 += einsum(x177, (0, 1, 2, 3), (0, 1, 2, 3))
    x196 += einsum(x111, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x196 += einsum(x178, (0, 1, 2, 3), (0, 1, 2, 3))
    del x178
    x196 += einsum(x180, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x180
    x196 += einsum(x183, (0, 1, 2, 3), (0, 1, 3, 2))
    del x183
    x196 += einsum(x195, (0, 1, 2, 3), (0, 1, 3, 2))
    del x195
    t2new_bbbb += einsum(x196, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x196, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb += einsum(x196, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb += einsum(x196, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x196
    x197 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x197 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x198 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x198 += einsum(v.bbbb.ooov, (0, 1, 2, 3), t3.bbbbbb, (4, 1, 2, 5, 6, 3), (4, 0, 5, 6))
    x199 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x199 += einsum(t2.bbbb, (0, 1, 2, 3), x184, (4, 5, 0, 1), (4, 5, 2, 3))
    x200 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x200 += einsum(v.aabb.ovoo, (0, 1, 2, 3), t3.babbab, (4, 0, 3, 5, 1, 6), (4, 2, 5, 6))
    x201 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x201 += einsum(x1, (0, 1, 2, 3), t3.bbbbbb, (4, 1, 2, 5, 6, 3), (0, 4, 5, 6)) * -1.0
    x202 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x202 += einsum(x3, (0, 1, 2, 3), t3.babbab, (4, 2, 1, 5, 3, 6), (0, 4, 5, 6))
    x203 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x203 += einsum(f.bb.oo, (0, 1), (0, 1))
    x203 += einsum(x20, (0, 1), (1, 0))
    x204 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x204 += einsum(x203, (0, 1), t2.bbbb, (2, 1, 3, 4), (0, 2, 3, 4)) * -2.0
    del x203
    x205 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x205 += einsum(x15, (0, 1), (1, 0))
    x205 += einsum(x16, (0, 1), (1, 0)) * 2.0
    x205 += einsum(x17, (0, 1), (1, 0))
    x205 += einsum(x19, (0, 1), (1, 0)) * -1.0
    x206 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x206 += einsum(x205, (0, 1), t2.bbbb, (2, 0, 3, 4), (1, 2, 3, 4)) * -2.0
    del x205
    x207 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x207 += einsum(x197, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x197
    x207 += einsum(x198, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.0
    del x198
    x207 += einsum(x199, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x199
    x207 += einsum(x200, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x200
    x207 += einsum(x201, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.0
    del x201
    x207 += einsum(x202, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x202
    x207 += einsum(x204, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x204
    x207 += einsum(x206, (0, 1, 2, 3), (1, 0, 3, 2))
    del x206
    t2new_bbbb += einsum(x207, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x207, (0, 1, 2, 3), (1, 0, 2, 3))
    del x207
    x208 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x208 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x209 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x209 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x209 += einsum(x208, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x208
    t2new_bbbb += einsum(x209, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x209, (0, 1, 2, 3), (0, 1, 2, 3))
    del x209
    x210 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x210 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x211 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x211 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x211 += einsum(x210, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    x211 += einsum(x171, (0, 1, 2, 3), (2, 1, 3, 0))
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x211, (0, 4, 1, 5), (4, 5, 2, 3)) * -2.0
    del x211
    x212 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x212 += einsum(t1.bb, (0, 1), x210, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    x212 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x212 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    t2new_bbbb += einsum(t1.bb, (0, 1), x212, (2, 3, 0, 4), (2, 3, 1, 4))
    del x212
    x213 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x213 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1))
    x214 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x214 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ooov, (4, 0, 1, 5), (4, 2, 3, 5))
    x215 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x215 += einsum(t2.aaaa, (0, 1, 2, 3), x214, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    x216 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x216 += einsum(t2.aaaa, (0, 1, 2, 3), x77, (4, 5, 1, 6), (4, 5, 0, 2, 3, 6))
    x217 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x217 += einsum(x28, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 7, 2), (4, 5, 1, 6, 7, 3)) * 6.0
    x218 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x218 += einsum(x213, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    del x213
    x218 += einsum(x215, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -4.0
    del x215
    x218 += einsum(x216, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -4.0
    del x216
    x218 += einsum(x217, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x217
    t3new_aaaaaa = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    t3new_aaaaaa += einsum(x218, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x218, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_aaaaaa += einsum(x218, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x218, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new_aaaaaa += einsum(x218, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x218, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new_aaaaaa += einsum(x218, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new_aaaaaa += einsum(x218, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x218, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x218
    x219 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x219 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvov, (4, 5, 1, 3), (0, 2, 4, 5))
    x220 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x220 += einsum(t2.aaaa, (0, 1, 2, 3), x118, (1, 3, 4, 5), (0, 2, 4, 5)) * 2.0
    x221 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x221 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x221 += einsum(x219, (0, 1, 2, 3), (0, 1, 3, 2))
    x221 += einsum(x220, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x220
    x222 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x222 += einsum(t2.aaaa, (0, 1, 2, 3), x221, (4, 5, 3, 6), (0, 1, 4, 2, 5, 6)) * -2.0
    del x221
    x223 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x223 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    x224 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x224 += einsum(t1.aa, (0, 1), x223, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x223
    x225 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x225 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x226 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x226 += einsum(t2.aaaa, (0, 1, 2, 3), x68, (1, 4, 5, 3), (0, 4, 2, 5)) * 2.0
    x227 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x227 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x227 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x227 += einsum(x225, (0, 1, 2, 3), (0, 1, 2, 3))
    x227 += einsum(x226, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x228 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x228 += einsum(t2.aaaa, (0, 1, 2, 3), x227, (4, 5, 6, 3), (0, 1, 4, 5, 2, 6)) * -1.0
    del x227
    x229 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x229 += einsum(x224, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x224
    x229 += einsum(x228, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x228
    x230 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x230 += einsum(t1.aa, (0, 1), x229, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x229
    x231 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x231 += einsum(x222, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    del x222
    x231 += einsum(x230, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    del x230
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new_aaaaaa += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x231
    x232 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x232 += einsum(v.aaaa.oooo, (0, 1, 2, 3), t3.aaaaaa, (4, 3, 1, 5, 6, 7), (4, 0, 2, 5, 6, 7)) * -1.0
    x233 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x233 += einsum(f.aa.oo, (0, 1), (0, 1))
    x233 += einsum(x30, (0, 1), (1, 0))
    x233 += einsum(x31, (0, 1), (1, 0)) * 2.0
    x233 += einsum(x34, (0, 1), (0, 1))
    x234 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x234 += einsum(x233, (0, 1), t3.aaaaaa, (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * 6.0
    del x233
    x235 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x235 += einsum(x232, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    del x232
    x235 += einsum(x234, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3))
    del x234
    t3new_aaaaaa += einsum(x235, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x235, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x235, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x235
    x236 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x236 += einsum(v.aaaa.vvvv, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 6, 7, 3, 1), (4, 5, 6, 7, 0, 2)) * -1.0
    x237 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x237 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x237 += einsum(x72, (0, 1), (1, 0))
    x237 += einsum(x73, (0, 1), (1, 0)) * 2.0
    x237 += einsum(x139, (0, 1), (0, 1))
    x238 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x238 += einsum(x237, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * 6.0
    del x237
    x239 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x239 += einsum(x236, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    del x236
    x239 += einsum(x238, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    del x238
    t3new_aaaaaa += einsum(x239, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new_aaaaaa += einsum(x239, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x239, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x239
    x240 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x240 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 2, 0, 5, 3, 6), (4, 5, 6, 1))
    x241 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x241 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa, (4, 0, 2, 5, 6, 3), (4, 5, 6, 1))
    x242 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x242 += einsum(x240, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x242 += einsum(x241, (0, 1, 2, 3), (0, 2, 1, 3)) * -3.0
    x243 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x243 += einsum(t2.aaaa, (0, 1, 2, 3), x242, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6)) * -4.0
    del x242
    x244 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x244 += einsum(v.aabb.ooov, (0, 1, 2, 3), t3.abaaba, (4, 2, 5, 6, 3, 7), (4, 5, 0, 1, 6, 7))
    x245 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x245 += einsum(x32, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 2, 6, 7, 3), (4, 5, 0, 1, 6, 7)) * 3.0
    x246 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x246 += einsum(x244, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x244
    x246 += einsum(x245, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x245
    x247 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x247 += einsum(t1.aa, (0, 1), x246, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x246
    x248 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x248 += einsum(x243, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    del x243
    x248 += einsum(x247, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    del x247
    t3new_aaaaaa += einsum(x248, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new_aaaaaa += einsum(x248, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x248, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new_aaaaaa += einsum(x248, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x248, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new_aaaaaa += einsum(x248, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x248, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x248, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    t3new_aaaaaa += einsum(x248, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    del x248
    x249 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x249 += einsum(x101, (0, 1, 2, 3), (1, 0, 3, 2))
    x249 += einsum(x81, (0, 1, 2, 3), (0, 1, 2, 3))
    x250 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x250 += einsum(x249, (0, 1, 2, 3), t3.aaaaaa, (4, 2, 3, 5, 6, 7), (4, 0, 1, 5, 6, 7)) * -6.0
    del x249
    x251 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x251 += einsum(x29, (0, 1), (1, 0))
    del x29
    x251 += einsum(x33, (0, 1), (1, 0)) * -1.0
    del x33
    x252 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x252 += einsum(x251, (0, 1), t3.aaaaaa, (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * 6.0
    x253 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x253 += einsum(x250, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x250
    x253 += einsum(x252, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    del x252
    t3new_aaaaaa += einsum(x253, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new_aaaaaa += einsum(x253, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x253, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    del x253
    x254 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x254 += einsum(t1.aa, (0, 1), x118, (0, 1, 2, 3), (2, 3))
    x255 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x255 += einsum(x71, (0, 1), (1, 0))
    del x71
    x255 += einsum(x254, (0, 1), (1, 0)) * -1.0
    del x254
    x256 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x256 += einsum(x255, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * 6.0
    x257 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x257 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 6, 7, 3, 1), (4, 5, 6, 0, 2, 7)) * -1.0
    x258 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x258 += einsum(x98, (0, 1, 2, 3), x257, (4, 5, 6, 0, 1, 7), (4, 5, 6, 2, 3, 7)) * 6.0
    del x98, x257
    x259 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x259 += einsum(x256, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    del x256
    x259 += einsum(x258, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x258
    t3new_aaaaaa += einsum(x259, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new_aaaaaa += einsum(x259, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x259, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    del x259
    x260 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x260 += einsum(x38, (0, 1, 2, 3), t3.abaaba, (4, 0, 5, 6, 1, 7), (2, 4, 5, 6, 7, 3))
    x261 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x261 += einsum(t2.aaaa, (0, 1, 2, 3), x25, (4, 0, 1, 5), (4, 2, 3, 5))
    x262 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x262 += einsum(t2.aaaa, (0, 1, 2, 3), x261, (4, 5, 6, 3), (4, 0, 1, 5, 6, 2)) * -1.0
    x263 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x263 += einsum(x46, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 7, 3), (4, 5, 1, 6, 7, 2)) * -6.0
    x264 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x264 += einsum(x78, (0, 1, 2, 3), (0, 1, 2, 3))
    x264 += einsum(x79, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x265 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x265 += einsum(t2.aaaa, (0, 1, 2, 3), x264, (4, 5, 1, 6), (0, 4, 5, 2, 3, 6)) * -4.0
    del x264
    x266 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x266 += einsum(x260, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    del x260
    x266 += einsum(x262, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -4.0
    del x262
    x266 += einsum(x263, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5))
    del x263
    x266 += einsum(x265, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x265
    t3new_aaaaaa += einsum(x266, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new_aaaaaa += einsum(x266, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x266, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new_aaaaaa += einsum(x266, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5))
    t3new_aaaaaa += einsum(x266, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x266, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new_aaaaaa += einsum(x266, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x266, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new_aaaaaa += einsum(x266, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    del x266
    x267 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x267 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0000000000000204
    x267 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -1.0
    x268 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x268 += einsum(x267, (0, 1, 2, 3), x68, (0, 4, 5, 2), (4, 1, 5, 3)) * 0.9999999999999901
    del x267
    x269 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x269 += einsum(x225, (0, 1, 2, 3), (0, 1, 2, 3))
    x269 += einsum(x268, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x268
    x270 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x270 += einsum(x269, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 6.0000000000000595
    del x269
    x271 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x271 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x271 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -0.499999999999995
    x272 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x272 += einsum(v.aabb.ovov, (0, 1, 2, 3), x271, (0, 4, 1, 5), (2, 3, 4, 5))
    x273 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x273 += einsum(x272, (0, 1, 2, 3), t3.abaaba, (4, 0, 5, 6, 1, 7), (4, 5, 2, 6, 7, 3)) * 4.00000000000004
    del x272
    x274 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x274 += einsum(x270, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x270
    x274 += einsum(x273, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    del x273
    t3new_aaaaaa += einsum(x274, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new_aaaaaa += einsum(x274, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x274, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new_aaaaaa += einsum(x274, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5))
    t3new_aaaaaa += einsum(x274, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x274, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3))
    t3new_aaaaaa += einsum(x274, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x274, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    t3new_aaaaaa += einsum(x274, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    del x274
    x275 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x275 += einsum(t2.aaaa, (0, 1, 2, 3), x64, (4, 5, 3, 6), (4, 0, 1, 2, 6, 5))
    x276 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x276 += einsum(t2.aaaa, (0, 1, 2, 3), x25, (4, 5, 6, 3), (4, 0, 1, 6, 5, 2))
    x277 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x277 += einsum(t1.aa, (0, 1), x276, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x276
    x278 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x278 += einsum(t2.aaaa, (0, 1, 2, 3), x46, (4, 5, 6, 3), (0, 1, 5, 4, 2, 6))
    x279 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x279 += einsum(x277, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    del x277
    x279 += einsum(x278, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    del x278
    x280 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x280 += einsum(t1.aa, (0, 1), x279, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x279
    x281 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x281 += einsum(x275, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    del x275
    x281 += einsum(x280, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    del x280
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new_aaaaaa += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x281
    x282 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x282 += einsum(x49, (0, 1, 2, 3), t3.aaaaaa, (4, 2, 3, 5, 6, 7), (0, 4, 1, 5, 6, 7))
    t3new_aaaaaa += einsum(x282, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x282, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x282, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x282, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x282, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x282, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -6.0
    del x282
    x283 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x283 += einsum(x61, (0, 1, 2, 3), t3.abaaba, (4, 0, 5, 6, 1, 7), (4, 5, 2, 6, 7, 3)) * -2.00000000000002
    t3new_aaaaaa += einsum(x283, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    t3new_aaaaaa += einsum(x283, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x283, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    t3new_aaaaaa += einsum(x283, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new_aaaaaa += einsum(x283, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_aaaaaa += einsum(x283, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x283, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new_aaaaaa += einsum(x283, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new_aaaaaa += einsum(x283, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x283
    x284 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x284 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 6, 7, 1, 3), (4, 5, 6, 0, 7, 2))
    x285 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x285 += einsum(t1.aa, (0, 1), x284, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x284
    t3new_aaaaaa += einsum(x285, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 6.0
    t3new_aaaaaa += einsum(x285, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x285, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -6.0
    t3new_aaaaaa += einsum(x285, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x285, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x285, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x285
    x286 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x286 += einsum(t1.aa, (0, 1), v.aaaa.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x287 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x287 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x288 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x288 += einsum(x286, (0, 1, 2, 3), (0, 2, 1, 3))
    x288 += einsum(x48, (0, 1, 2, 3), (0, 2, 1, 3))
    x288 += einsum(x287, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x289 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x289 += einsum(t1.aa, (0, 1), x81, (2, 3, 0, 4), (3, 2, 4, 1))
    del x81
    x290 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x290 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x290 += einsum(x37, (0, 1, 2, 3), (1, 0, 2, 3))
    x291 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x291 += einsum(t1.aa, (0, 1), x290, (2, 3, 1, 4), (2, 3, 0, 4))
    x292 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x292 += einsum(v.aaaa.ooov, (0, 1, 2, 3), x40, (1, 4, 3, 5), (0, 2, 4, 5)) * 2.0
    x293 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x293 += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3))
    x293 += einsum(x289, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x293 += einsum(x56, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x56
    x293 += einsum(x291, (0, 1, 2, 3), (2, 1, 0, 3))
    del x291
    x293 += einsum(x292, (0, 1, 2, 3), (2, 0, 1, 3))
    del x292
    x294 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x294 += einsum(x101, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x101
    x294 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x294 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 2, 1, 3))
    x295 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x295 += einsum(t1.aa, (0, 1), x294, (2, 3, 0, 4), (2, 3, 4, 1))
    x296 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x296 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x296 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x296 += einsum(x288, (0, 1, 2, 3), (0, 1, 2, 3))
    x296 += einsum(x288, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x288
    x296 += einsum(x293, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x296 += einsum(x293, (0, 1, 2, 3), (1, 0, 2, 3))
    del x293
    x296 += einsum(x50, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x296 += einsum(x50, (0, 1, 2, 3), (1, 0, 2, 3))
    x296 += einsum(x80, (0, 1, 2, 3), (2, 1, 0, 3))
    x296 += einsum(x295, (0, 1, 2, 3), (1, 0, 2, 3))
    x297 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x297 += einsum(t2.aaaa, (0, 1, 2, 3), x296, (4, 5, 1, 6), (0, 4, 5, 2, 3, 6)) * -2.0
    del x296
    t3new_aaaaaa += einsum(x297, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x297, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new_aaaaaa += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x297
    x298 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x298 += einsum(x286, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    del x286
    x298 += einsum(x48, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x298 += einsum(x287, (0, 1, 2, 3), (0, 2, 1, 3))
    del x287
    x299 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x299 += einsum(t2.aaaa, (0, 1, 2, 3), x55, (4, 1, 5, 3), (0, 4, 5, 2))
    x300 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x300 += einsum(t1.aa, (0, 1), x290, (2, 3, 1, 4), (2, 3, 0, 4)) * 0.5
    del x290
    x301 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x301 += einsum(v.aaaa.ooov, (0, 1, 2, 3), x40, (1, 4, 3, 5), (0, 2, 4, 5))
    x302 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x302 += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x302 += einsum(x289, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x289
    x302 += einsum(x299, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x299
    x302 += einsum(x300, (0, 1, 2, 3), (2, 1, 0, 3))
    del x300
    x302 += einsum(x301, (0, 1, 2, 3), (2, 0, 1, 3))
    del x301
    x303 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x303 += einsum(x13, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4)) * -1.0
    x304 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x304 += einsum(t1.aa, (0, 1), x294, (2, 3, 0, 4), (2, 3, 4, 1)) * 0.5
    del x294
    x305 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x305 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x305 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * 0.5
    x305 += einsum(x298, (0, 1, 2, 3), (0, 1, 2, 3))
    x305 += einsum(x298, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x298
    x305 += einsum(x302, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x305 += einsum(x302, (0, 1, 2, 3), (1, 0, 2, 3))
    del x302
    x305 += einsum(x50, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x305 += einsum(x50, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    x305 += einsum(x303, (0, 1, 2, 3), (2, 1, 0, 3))
    del x303
    x305 += einsum(x304, (0, 1, 2, 3), (1, 0, 2, 3))
    del x304
    x306 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x306 += einsum(t2.aaaa, (0, 1, 2, 3), x305, (4, 5, 1, 6), (0, 4, 5, 2, 3, 6)) * -4.0
    del x305
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    del x306
    x307 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x307 += einsum(t2.abab, (0, 1, 2, 3), v.bbbb.ovvv, (4, 5, 6, 3), (1, 4, 5, 6, 0, 2))
    x308 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x308 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 7, 1), (5, 2, 7, 3, 4, 6))
    x309 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x309 += einsum(x127, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 6, 7, 3), (0, 5, 7, 1, 4, 6))
    x310 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x310 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.00000000000002
    x310 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -1.0
    x311 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x311 += einsum(x310, (0, 1, 2, 3), x7, (0, 4, 2, 5), (4, 1, 5, 3))
    del x310
    x312 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x312 += einsum(x111, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.00000000000001
    x312 += einsum(x311, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x311
    x313 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x313 += einsum(x312, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (0, 4, 2, 6, 5, 7)) * 2.0
    del x312
    x314 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x314 += einsum(t2.abab, (0, 1, 2, 3), x11, (0, 4, 2, 5), (1, 3, 4, 5)) * 1.00000000000001
    x315 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x315 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x315 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -0.499999999999995
    x316 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x316 += einsum(v.aabb.ovov, (0, 1, 2, 3), x315, (2, 4, 3, 5), (4, 5, 0, 1)) * 2.00000000000002
    x317 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x317 += einsum(x314, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x317 += einsum(x316, (0, 1, 2, 3), (0, 1, 2, 3))
    del x316
    x318 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x318 += einsum(x317, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 6, 7, 3), (5, 0, 7, 1, 4, 6)) * 2.0
    del x317
    x319 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x319 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), (0, 2, 3, 5, 1, 4))
    x319 += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (0, 3, 5, 1, 2, 4)) * -0.5
    x320 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x320 += einsum(x14, (0, 1, 2, 3), x319, (0, 4, 2, 5, 6, 7), (1, 4, 3, 5, 6, 7)) * 2.0
    x321 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x321 += einsum(x182, (0, 1, 2, 3), x319, (0, 4, 3, 5, 6, 7), (1, 4, 2, 5, 6, 7)) * -2.0
    del x319
    x322 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x322 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovoo, (0, 4, 5, 1), (5, 3, 2, 4))
    x323 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x323 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovvv, (0, 4, 5, 3), (1, 5, 2, 4))
    x324 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x324 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 0, 2, 5, 6, 3), (4, 5, 6, 1))
    x325 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x325 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.abaaba, (0, 4, 2, 5, 6, 1), (4, 6, 5, 3)) * -1.0
    x326 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x326 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x326 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x326 += einsum(x109, (0, 1, 2, 3), (0, 2, 3, 1))
    x326 += einsum(x109, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    x327 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x327 += einsum(t2.abab, (0, 1, 2, 3), x326, (0, 4, 2, 5), (1, 3, 4, 5))
    del x326
    x328 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x328 += einsum(x13, (0, 1), t2.abab, (0, 2, 3, 4), (2, 4, 1, 3))
    x329 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x329 += einsum(v.aabb.vvov, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x329 += einsum(x105, (0, 1, 2, 3), (0, 1, 2, 3))
    x329 += einsum(x322, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x322
    x329 += einsum(x323, (0, 1, 2, 3), (0, 1, 2, 3))
    del x323
    x329 += einsum(x324, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x329 += einsum(x325, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x329 += einsum(x327, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x327
    x329 += einsum(x328, (0, 1, 2, 3), (0, 1, 3, 2))
    del x328
    x330 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x330 += einsum(t2.abab, (0, 1, 2, 3), x329, (4, 5, 6, 2), (4, 1, 5, 3, 0, 6))
    del x329
    x331 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x331 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovvv, (0, 2, 4, 5), (1, 3, 4, 5))
    x332 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x332 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x332 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x333 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x333 += einsum(t2.bbbb, (0, 1, 2, 3), x332, (1, 3, 4, 5), (0, 4, 5, 2)) * 2.0
    del x332
    x334 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x334 += einsum(t1.bb, (0, 1), x1, (2, 0, 3, 4), (2, 3, 1, 4))
    x335 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x335 += einsum(t2.bbbb, (0, 1, 2, 3), x7, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    x336 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x336 += einsum(x334, (0, 1, 2, 3), (0, 1, 2, 3))
    del x334
    x336 += einsum(x111, (0, 1, 2, 3), (0, 1, 2, 3))
    x336 += einsum(x335, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x337 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x337 += einsum(t1.bb, (0, 1), x336, (2, 0, 3, 4), (2, 3, 4, 1))
    del x336
    x338 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x338 += einsum(x157, (0, 1, 2, 3), (0, 2, 1, 3))
    x338 += einsum(x331, (0, 1, 2, 3), (0, 3, 2, 1))
    x338 += einsum(x333, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x333
    x338 += einsum(x337, (0, 1, 2, 3), (0, 2, 1, 3))
    del x337
    x339 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x339 += einsum(t2.abab, (0, 1, 2, 3), x338, (4, 3, 5, 6), (4, 1, 5, 6, 0, 2))
    del x338
    x340 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x340 += einsum(t2.abab, (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 6, 3), (1, 4, 5, 6, 0, 2))
    x341 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x341 += einsum(t1.bb, (0, 1), x340, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x340
    x342 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x342 += einsum(v.aabb.ovoo, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 7, 1), (5, 2, 3, 7, 4, 6))
    x343 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x343 += einsum(x188, (0, 1, 2, 3), t3.babbab, (4, 5, 0, 6, 7, 3), (1, 2, 4, 6, 5, 7)) * 2.0
    del x188
    x344 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x344 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 2, 3, 4), (3, 4, 1, 2))
    x345 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x345 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 5, 3), (1, 5, 2, 4))
    x346 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x346 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x346 += einsum(x344, (0, 1, 2, 3), (1, 0, 3, 2))
    x346 += einsum(x345, (0, 1, 2, 3), (0, 1, 3, 2))
    del x345
    x347 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x347 += einsum(t2.abab, (0, 1, 2, 3), x346, (4, 5, 2, 6), (4, 5, 1, 3, 0, 6))
    del x346
    x348 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x348 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x348 += einsum(x136, (0, 1, 2, 3), (1, 0, 3, 2))
    x348 += einsum(x150, (0, 1, 2, 3), (0, 1, 3, 2))
    del x150
    x349 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x349 += einsum(t2.abab, (0, 1, 2, 3), x348, (4, 5, 0, 6), (4, 5, 1, 3, 6, 2))
    del x348
    x350 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x350 += einsum(t1.aa, (0, 1), x3, (2, 3, 0, 4), (2, 3, 1, 4))
    x351 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x351 += einsum(x129, (0, 1, 2, 3), (0, 1, 3, 2))
    x351 += einsum(x350, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x350
    x352 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x352 += einsum(t2.abab, (0, 1, 2, 3), x351, (4, 5, 2, 6), (4, 5, 1, 3, 0, 6))
    del x351
    x353 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x353 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x353 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x354 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x354 += einsum(t1.bb, (0, 1), x353, (2, 1, 3, 4), (2, 0, 3, 4))
    x355 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x355 += einsum(t2.abab, (0, 1, 2, 3), x354, (4, 5, 6, 0), (5, 4, 1, 3, 6, 2))
    del x354
    x356 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x356 += einsum(x341, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    del x341
    x356 += einsum(x342, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    del x342
    x356 += einsum(x343, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    del x343
    x356 += einsum(x347, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    del x347
    x356 += einsum(x349, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    del x349
    x356 += einsum(x352, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x352
    x356 += einsum(x355, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x355
    x357 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x357 += einsum(t1.bb, (0, 1), x356, (2, 0, 3, 4, 5, 6), (2, 3, 4, 1, 5, 6))
    del x356
    x358 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x358 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovoo, (4, 2, 5, 1), (5, 3, 0, 4))
    x359 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x359 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovvv, (4, 2, 5, 3), (1, 5, 0, 4))
    x360 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x360 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 5, 2, 6, 1, 3), (4, 6, 5, 0))
    x361 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x361 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 3, 6, 1), (5, 6, 4, 0)) * -1.0
    x362 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x362 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x362 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x362 += einsum(x25, (0, 1, 2, 3), (1, 0, 2, 3))
    x362 += einsum(x25, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x363 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x363 += einsum(t2.abab, (0, 1, 2, 3), x362, (4, 5, 0, 2), (1, 3, 4, 5))
    del x362
    x364 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x364 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x364 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x364 += einsum(x358, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x358
    x364 += einsum(x359, (0, 1, 2, 3), (0, 1, 2, 3))
    del x359
    x364 += einsum(x360, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x364 += einsum(x361, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x364 += einsum(x363, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x363
    x365 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x365 += einsum(t2.abab, (0, 1, 2, 3), x364, (4, 5, 6, 0), (4, 1, 5, 3, 6, 2))
    del x364
    x366 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x366 += einsum(t1.bb, (0, 1), v.aabb.vvvv, (2, 3, 4, 1), (0, 4, 2, 3))
    x367 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x367 += einsum(t1.aa, (0, 1), x127, (2, 3, 0, 4), (2, 3, 1, 4))
    x368 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x368 += einsum(t2.abab, (0, 1, 2, 3), x3, (4, 1, 0, 5), (4, 3, 2, 5))
    x369 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x369 += einsum(x366, (0, 1, 2, 3), (0, 1, 3, 2))
    del x366
    x369 += einsum(x367, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x367
    x369 += einsum(x368, (0, 1, 2, 3), (0, 1, 3, 2))
    del x368
    x370 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x370 += einsum(t2.abab, (0, 1, 2, 3), x369, (4, 5, 2, 6), (4, 1, 5, 3, 0, 6))
    del x369
    x371 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x371 += einsum(t2.abab, (0, 1, 2, 3), x3, (4, 1, 5, 2), (4, 3, 0, 5))
    x372 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x372 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x372 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3))
    x373 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x373 += einsum(t1.bb, (0, 1), x372, (1, 2, 3, 4), (0, 2, 3, 4))
    del x372
    x374 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x374 += einsum(x371, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x371
    x374 += einsum(x373, (0, 1, 2, 3), (0, 1, 3, 2))
    del x373
    x375 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x375 += einsum(t2.abab, (0, 1, 2, 3), x374, (4, 5, 0, 6), (4, 1, 5, 3, 6, 2))
    del x374
    x376 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x376 += einsum(v.aabb.vvov, (0, 1, 2, 3), (2, 3, 0, 1))
    x376 += einsum(x105, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x105
    x377 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x377 += einsum(t2.bbbb, (0, 1, 2, 3), x376, (1, 3, 4, 5), (0, 2, 4, 5))
    del x376
    x378 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x378 += einsum(t2.abab, (0, 1, 2, 3), x377, (4, 5, 6, 2), (4, 1, 5, 3, 0, 6)) * 2.0
    del x377
    x379 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x379 += einsum(t2.bbbb, (0, 1, 2, 3), x353, (1, 3, 4, 5), (0, 2, 4, 5))
    del x353
    x380 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x380 += einsum(t2.abab, (0, 1, 2, 3), x379, (4, 5, 6, 0), (4, 1, 5, 3, 6, 2)) * 2.0
    del x379
    x381 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x381 += einsum(x307, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x307
    x381 += einsum(x308, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    del x308
    x381 += einsum(x309, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    del x309
    x381 += einsum(x313, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x313
    x381 += einsum(x318, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x318
    x381 += einsum(x320, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x320
    x381 += einsum(x321, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x321
    x381 += einsum(x330, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x330
    x381 += einsum(x339, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x339
    x381 += einsum(x357, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x357
    x381 += einsum(x365, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x365
    x381 += einsum(x370, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x370
    x381 += einsum(x375, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x375
    x381 += einsum(x378, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x378
    x381 += einsum(x380, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x380
    t3new_babbab += einsum(x381, (0, 1, 2, 3, 4, 5), (0, 4, 1, 2, 5, 3)) * -1.0
    t3new_babbab += einsum(x381, (0, 1, 2, 3, 4, 5), (0, 4, 1, 3, 5, 2))
    t3new_babbab += einsum(x381, (0, 1, 2, 3, 4, 5), (1, 4, 0, 2, 5, 3))
    t3new_babbab += einsum(x381, (0, 1, 2, 3, 4, 5), (1, 4, 0, 3, 5, 2)) * -1.0
    del x381
    x382 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x382 += einsum(t2.bbbb, (0, 1, 2, 3), x1, (4, 0, 1, 5), (4, 2, 3, 5))
    x383 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x383 += einsum(t2.abab, (0, 1, 2, 3), x382, (4, 5, 6, 3), (4, 1, 5, 6, 0, 2)) * -1.0
    x384 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x384 += einsum(t2.abab, (0, 1, 2, 3), (1, 3, 0, 2))
    x384 += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (2, 3, 0, 1)) * 0.99999999999999
    x385 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x385 += einsum(v.aabb.ovov, (0, 1, 2, 3), x384, (4, 3, 0, 5), (2, 4, 1, 5)) * 1.00000000000001
    x386 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x386 += einsum(x129, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x386 += einsum(x385, (0, 1, 2, 3), (1, 0, 2, 3))
    del x385
    x387 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x387 += einsum(x386, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 2, 7), (0, 4, 6, 7, 5, 3)) * -2.0
    del x386
    x388 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x388 += einsum(x135, (0, 1, 2, 3), (0, 1, 3, 2))
    x388 += einsum(x137, (0, 1, 2, 3), (0, 1, 3, 2))
    x389 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x389 += einsum(x388, (0, 1, 2, 3), t3.babbab, (4, 2, 1, 5, 6, 7), (0, 4, 5, 7, 3, 6)) * -2.0
    del x388
    x390 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x390 += einsum(t1.aa, (0, 1), v.aabb.oooo, (2, 0, 3, 4), (3, 4, 2, 1))
    x391 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x391 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.babbab, (4, 5, 2, 3, 6, 1), (4, 0, 5, 6)) * -1.0
    x392 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x392 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 3, 1), (5, 2, 4, 6))
    x393 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x393 += einsum(t2.abab, (0, 1, 2, 3), x18, (4, 5, 1, 3), (4, 5, 0, 2))
    x394 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x394 += einsum(t2.abab, (0, 1, 2, 3), x24, (4, 3, 0, 5), (4, 1, 5, 2))
    x395 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x395 += einsum(t2.abab, (0, 1, 2, 3), x106, (4, 3, 2, 5), (4, 1, 0, 5))
    x396 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x396 += einsum(x9, (0, 1), t2.abab, (2, 3, 4, 1), (0, 3, 2, 4))
    x397 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x397 += einsum(v.aabb.ovoo, (0, 1, 2, 3), x40, (0, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    x398 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x398 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x398 += einsum(x390, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x390
    x398 += einsum(x147, (0, 1, 2, 3), (1, 0, 2, 3))
    x398 += einsum(x391, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x398 += einsum(x392, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x398 += einsum(x393, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x393
    x398 += einsum(x394, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x394
    x398 += einsum(x395, (0, 1, 2, 3), (0, 1, 2, 3))
    del x395
    x398 += einsum(x396, (0, 1, 2, 3), (0, 1, 2, 3))
    del x396
    x398 += einsum(x397, (0, 1, 2, 3), (1, 0, 2, 3))
    del x397
    x399 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x399 += einsum(t2.bbbb, (0, 1, 2, 3), x398, (1, 4, 5, 6), (4, 0, 2, 3, 5, 6)) * -2.0
    del x398
    x400 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x400 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x400 += einsum(x344, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x401 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x401 += einsum(x400, (0, 1, 2, 3), t3.babbab, (4, 5, 0, 6, 2, 7), (1, 4, 6, 7, 5, 3)) * -2.0
    del x400
    x402 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x402 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 0, 5), (4, 2, 3, 5)) * -1.0
    x403 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x403 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb, (4, 0, 2, 5, 6, 3), (4, 5, 6, 1))
    x404 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x404 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 0, 2, 5, 1, 6), (4, 5, 6, 3))
    x405 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x405 += einsum(x402, (0, 1, 2, 3), (0, 2, 1, 3))
    x405 += einsum(x403, (0, 1, 2, 3), (0, 2, 1, 3)) * -3.0
    x405 += einsum(x404, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x406 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x406 += einsum(t2.abab, (0, 1, 2, 3), x405, (4, 5, 6, 3), (4, 1, 5, 6, 0, 2)) * 2.0
    del x405
    x407 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x407 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x407 += einsum(x136, (0, 1, 2, 3), (1, 0, 3, 2))
    x408 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x408 += einsum(x407, (0, 1, 2, 3), t3.babbab, (4, 2, 0, 5, 6, 7), (1, 4, 5, 7, 3, 6)) * -2.0
    del x407
    x409 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x409 += einsum(f.bb.oo, (0, 1), (0, 1))
    x409 += einsum(x16, (0, 1), (0, 1)) * 2.0
    x409 += einsum(x17, (0, 1), (0, 1))
    x409 += einsum(x20, (0, 1), (1, 0))
    x410 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x410 += einsum(x409, (0, 1), t3.babbab, (2, 3, 1, 4, 5, 6), (0, 2, 4, 6, 3, 5)) * -2.0
    del x409
    x411 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x411 += einsum(t1.aa, (0, 1), x135, (2, 3, 4, 0), (2, 3, 4, 1))
    x412 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x412 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    x412 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x413 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x413 += einsum(t2.abab, (0, 1, 2, 3), x412, (4, 5, 1, 3), (4, 5, 0, 2))
    x414 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x414 += einsum(x3, (0, 1, 2, 3), x40, (2, 4, 3, 5), (0, 1, 4, 5)) * 2.0
    del x40
    x415 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x415 += einsum(x411, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x411
    x415 += einsum(x413, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x413
    x415 += einsum(x149, (0, 1, 2, 3), (1, 0, 2, 3))
    del x149
    x415 += einsum(x414, (0, 1, 2, 3), (0, 1, 2, 3))
    del x414
    x416 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x416 += einsum(t2.bbbb, (0, 1, 2, 3), x415, (4, 1, 5, 6), (4, 0, 2, 3, 5, 6)) * -2.0
    del x415
    x417 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x417 += einsum(x15, (0, 1), (1, 0))
    del x15
    x417 += einsum(x19, (0, 1), (1, 0)) * -1.0
    del x19
    x418 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x418 += einsum(x417, (0, 1), t3.babbab, (2, 3, 0, 4, 5, 6), (1, 2, 4, 6, 3, 5)) * -2.0
    x419 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x419 += einsum(x383, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * 2.0
    del x383
    x419 += einsum(x387, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x387
    x419 += einsum(x389, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x389
    x419 += einsum(x399, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x399
    x419 += einsum(x401, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x401
    x419 += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x406
    x419 += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x408
    x419 += einsum(x410, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x410
    x419 += einsum(x416, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x416
    x419 += einsum(x418, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x418
    t3new_babbab += einsum(x419, (0, 1, 2, 3, 4, 5), (0, 4, 1, 2, 5, 3)) * -1.0
    t3new_babbab += einsum(x419, (0, 1, 2, 3, 4, 5), (1, 4, 0, 2, 5, 3))
    del x419
    x420 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x420 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 7, 1, 3), (4, 6, 2, 7, 5, 0))
    x421 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x421 += einsum(t2.abab, (0, 1, 2, 3), x420, (4, 5, 1, 6, 7, 0), (4, 5, 3, 6, 7, 2))
    del x420
    x422 = np.zeros((nvir[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x422 += einsum(v.aabb.vvvv, (0, 1, 2, 3), (2, 3, 0, 1))
    x422 += einsum(x133, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x133
    x423 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x423 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), (0, 2, 3, 5, 1, 4))
    x423 += einsum(t1.aa, (0, 1), t2.bbbb, (2, 3, 4, 5), (2, 3, 4, 5, 0, 1))
    x424 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x424 += einsum(x422, (0, 1, 2, 3), x423, (4, 5, 0, 6, 7, 2), (4, 5, 6, 1, 7, 3)) * 2.0
    del x422
    x425 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x425 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x425 += einsum(x122, (0, 1, 2, 3), (1, 0, 3, 2))
    x426 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x426 += einsum(x425, (0, 1, 2, 3), t3.babbab, (4, 2, 5, 6, 7, 0), (4, 5, 1, 6, 3, 7)) * -2.0
    del x425
    x427 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x427 += einsum(t1.aa, (0, 1), v.aabb.oovv, (2, 0, 3, 4), (3, 4, 2, 1))
    x428 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x428 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ovvv, (1, 3, 4, 5), (4, 5, 0, 2))
    x429 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x429 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 2, 0, 5, 6, 1), (6, 3, 4, 5))
    x430 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x430 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x430 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x431 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x431 += einsum(t2.abab, (0, 1, 2, 3), x430, (1, 4, 3, 5), (4, 5, 0, 2))
    x432 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x432 += einsum(t2.abab, (0, 1, 2, 3), x24, (1, 4, 0, 5), (4, 3, 5, 2))
    del x24
    x433 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x433 += einsum(t2.abab, (0, 1, 2, 3), x106, (1, 4, 2, 5), (4, 3, 0, 5))
    x434 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x434 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (2, 3, 0, 1))
    x434 += einsum(x427, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x427
    x434 += einsum(x428, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x428
    x434 += einsum(x429, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    x434 += einsum(x431, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x431
    x434 += einsum(x432, (0, 1, 2, 3), (0, 1, 2, 3))
    del x432
    x434 += einsum(x433, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x433
    x435 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x435 += einsum(t2.bbbb, (0, 1, 2, 3), x434, (3, 4, 5, 6), (0, 1, 4, 2, 5, 6)) * -2.0
    del x434
    x436 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x436 += einsum(v.aabb.ovov, (0, 1, 2, 3), x384, (2, 4, 5, 1), (3, 4, 0, 5))
    x437 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x437 += einsum(x121, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.99999999999999
    x437 += einsum(x436, (0, 1, 2, 3), (1, 0, 2, 3))
    x438 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x438 += einsum(x437, (0, 1, 2, 3), t3.babbab, (4, 2, 5, 6, 7, 1), (4, 5, 0, 6, 3, 7)) * -2.00000000000002
    del x437
    x439 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x439 += einsum(t1.bb, (0, 1), x430, (0, 2, 1, 3), (2, 3))
    x440 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x440 += einsum(x141, (0, 1), (1, 0))
    del x141
    x440 += einsum(x439, (0, 1), (0, 1)) * -1.0
    del x439
    x441 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x441 += einsum(x440, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 0), (2, 4, 1, 5, 3, 6)) * -2.0
    x442 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x442 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 3, 7, 1), (4, 6, 0, 2, 5, 7)) * -1.0
    x443 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x443 += einsum(x106, (0, 1, 2, 3), x423, (4, 5, 1, 6, 7, 2), (0, 4, 5, 6, 7, 3))
    del x106, x423
    x444 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x444 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x444 += einsum(x124, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x444 += einsum(x42, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x42
    x445 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x445 += einsum(t2.bbbb, (0, 1, 2, 3), x444, (4, 3, 5, 6), (4, 0, 1, 2, 5, 6)) * -1.0
    del x444
    x446 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x446 += einsum(f.bb.ov, (0, 1), (0, 1))
    x446 += einsum(x0, (0, 1), (0, 1))
    del x0
    x447 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x447 += einsum(x446, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (0, 2, 4, 5, 3, 6)) * -1.0
    del x446
    x448 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x448 += einsum(x442, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    del x442
    x448 += einsum(x443, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    del x443
    x448 += einsum(x445, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    del x445
    x448 += einsum(x447, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    del x447
    x449 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x449 += einsum(t1.bb, (0, 1), x448, (0, 2, 3, 4, 5, 6), (2, 3, 4, 1, 5, 6)) * 2.0
    del x448
    x450 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x450 += einsum(f.bb.vv, (0, 1), (0, 1))
    x450 += einsum(x143, (0, 1), (1, 0)) * -1.0
    x451 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x451 += einsum(x450, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 0), (2, 4, 1, 5, 3, 6)) * -2.0
    del x450
    x452 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x452 += einsum(x167, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x452 += einsum(x169, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x453 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x453 += einsum(t2.abab, (0, 1, 2, 3), x452, (4, 5, 1, 6), (4, 5, 6, 3, 0, 2)) * 2.0
    del x452
    x454 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x454 += einsum(x421, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    del x421
    x454 += einsum(x424, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    del x424
    x454 += einsum(x426, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x426
    x454 += einsum(x435, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x435
    x454 += einsum(x438, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    del x438
    x454 += einsum(x441, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x441
    x454 += einsum(x449, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x449
    x454 += einsum(x451, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    del x451
    x454 += einsum(x453, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x453
    t3new_babbab += einsum(x454, (0, 1, 2, 3, 4, 5), (0, 4, 1, 2, 5, 3)) * -1.0
    t3new_babbab += einsum(x454, (0, 1, 2, 3, 4, 5), (0, 4, 1, 3, 5, 2))
    del x454
    x455 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x455 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.babbab, (0, 4, 2, 5, 6, 3), (5, 1, 4, 6))
    x456 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x456 += einsum(t2.bbbb, (0, 1, 2, 3), x455, (4, 3, 5, 6), (0, 1, 2, 4, 5, 6))
    x457 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x457 += einsum(t2.abab, (0, 1, 2, 3), x168, (4, 5, 1, 6), (5, 4, 3, 6, 0, 2)) * -1.0
    x458 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x458 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 0, 1), (2, 3))
    x459 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x459 += einsum(t1.bb, (0, 1), x458, (0, 2), (1, 2))
    del x458
    x460 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x460 += einsum(v.bbbb.ovov, (0, 1, 2, 3), x6, (0, 2, 3, 4), (4, 1)) * 2.0
    del x6
    x461 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x461 += einsum(x459, (0, 1), (0, 1))
    del x459
    x461 += einsum(x460, (0, 1), (0, 1)) * -1.0
    del x460
    x462 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x462 += einsum(x461, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (2, 4, 0, 5, 3, 6)) * -2.0
    del x461
    x463 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x463 += einsum(t2.bbbb, (0, 1, 2, 3), x61, (4, 3, 5, 6), (4, 0, 1, 2, 5, 6))
    x464 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x464 += einsum(t1.bb, (0, 1), x463, (0, 2, 3, 4, 5, 6), (3, 2, 4, 1, 5, 6)) * 2.0
    del x463
    x465 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x465 += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 4.0
    del x456
    x465 += einsum(x457, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -6.0
    del x457
    x465 += einsum(x462, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    del x462
    x465 += einsum(x464, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x464
    t3new_babbab += einsum(x465, (0, 1, 2, 3, 4, 5), (1, 4, 0, 2, 5, 3))
    t3new_babbab += einsum(x465, (0, 1, 2, 3, 4, 5), (1, 4, 0, 3, 5, 2)) * -1.0
    del x465
    x466 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x466 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x466 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x466 += einsum(x225, (0, 1, 2, 3), (1, 0, 3, 2)) * 1.00000000000001
    x466 += einsum(x11, (0, 1, 2, 3), x271, (0, 4, 2, 5), (1, 4, 3, 5)) * -2.00000000000002
    x466 += einsum(x46, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x466 += einsum(t1.aa, (0, 1), x32, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    t3new_babbab += einsum(x466, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 2, 7), (4, 1, 5, 6, 3, 7)) * 2.0
    del x466
    x467 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x467 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x467 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -0.4999999999999949
    x468 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x468 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x468 += einsum(x124, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x124
    x468 += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3))
    x468 += einsum(t2.abab, (0, 1, 2, 3), x7, (1, 4, 3, 5), (4, 5, 0, 2)) * -1.00000000000001
    x468 += einsum(v.aabb.ovov, (0, 1, 2, 3), x467, (0, 4, 1, 5), (2, 3, 4, 5)) * 2.0000000000000204
    del x467
    t3new_babbab += einsum(x468, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 0, 6, 7, 1), (4, 2, 5, 6, 3, 7)) * 6.0
    del x468
    x469 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x469 += einsum(v.bbbb.vvvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x469 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 4, 1, 5), (5, 2, 4, 3))
    x469 += einsum(t1.bb, (0, 1), x107, (0, 2, 3, 4), (4, 1, 3, 2))
    t3new_babbab += einsum(x469, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 0, 7, 2), (4, 5, 6, 1, 7, 3)) * -2.0
    del x469
    x470 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x470 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x470 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    x471 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x471 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x471 += einsum(x184, (0, 1, 2, 3), (3, 0, 2, 1))
    x471 += einsum(x184, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x471 += einsum(v.bbbb.ovov, (0, 1, 2, 3), x470, (4, 5, 3, 1), (0, 5, 2, 4))
    t3new_babbab += einsum(x471, (0, 1, 2, 3), t3.babbab, (0, 4, 2, 5, 6, 7), (1, 4, 3, 5, 6, 7)) * 2.0
    del x471
    x472 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x472 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x472 += einsum(x185, (0, 1, 2, 3), (1, 0, 2, 3))
    x473 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x473 += einsum(t2.bbbb, (0, 1, 2, 3), x18, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    x474 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x474 += einsum(t2.bbbb, (0, 1, 2, 3), x190, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    x475 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x475 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x475 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3))
    x476 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x476 += einsum(t1.bb, (0, 1), x475, (2, 3, 4, 1), (2, 3, 4, 0))
    del x475
    x477 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x477 += einsum(t1.bb, (0, 1), x476, (2, 0, 3, 4), (4, 2, 3, 1))
    x478 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x478 += einsum(x186, (0, 1, 2, 3), (0, 2, 1, 3))
    x478 += einsum(x187, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x478 += einsum(x473, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x473
    x478 += einsum(x474, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x474
    x478 += einsum(x193, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x193
    x478 += einsum(x477, (0, 1, 2, 3), (0, 2, 1, 3))
    del x477
    x479 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x479 += einsum(x210, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    x479 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x479 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 2, 1, 3))
    x480 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x480 += einsum(t1.bb, (0, 1), x479, (2, 3, 0, 4), (2, 3, 4, 1))
    x481 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x481 += einsum(x472, (0, 1, 2, 3), (0, 1, 2, 3))
    x481 += einsum(x472, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x472
    x481 += einsum(x478, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x481 += einsum(x478, (0, 1, 2, 3), (1, 2, 0, 3))
    del x478
    x481 += einsum(x170, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x481 += einsum(x480, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    t3new_babbab += einsum(t2.abab, (0, 1, 2, 3), x481, (1, 4, 5, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    del x481
    x482 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x482 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x482 += einsum(x185, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    x483 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x483 += einsum(t2.bbbb, (0, 1, 2, 3), x18, (4, 5, 1, 3), (4, 5, 0, 2))
    del x18
    x484 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x484 += einsum(t2.bbbb, (0, 1, 2, 3), x190, (4, 5, 1, 3), (4, 5, 0, 2))
    del x190
    x485 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x485 += einsum(t1.bb, (0, 1), x192, (2, 3, 1, 4), (2, 3, 0, 4)) * 0.5
    del x192
    x486 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x486 += einsum(t1.bb, (0, 1), x476, (2, 0, 3, 4), (4, 2, 3, 1)) * 0.5
    del x476
    x487 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x487 += einsum(x186, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x487 += einsum(x187, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x487 += einsum(x483, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x483
    x487 += einsum(x484, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x484
    x487 += einsum(x485, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x485
    x487 += einsum(x486, (0, 1, 2, 3), (0, 2, 1, 3))
    del x486
    x488 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x488 += einsum(x482, (0, 1, 2, 3), (0, 1, 2, 3))
    x488 += einsum(x482, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x482
    x488 += einsum(x487, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x488 += einsum(x487, (0, 1, 2, 3), (1, 2, 0, 3))
    del x487
    x488 += einsum(x9, (0, 1), t2.bbbb, (2, 3, 4, 1), (0, 2, 3, 4)) * -1.0
    x488 += einsum(t1.bb, (0, 1), x479, (2, 3, 0, 4), (4, 3, 2, 1)) * -0.5
    del x479
    t3new_babbab += einsum(t2.abab, (0, 1, 2, 3), x488, (1, 4, 5, 6), (5, 0, 4, 3, 2, 6)) * 2.0
    del x488
    x489 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x489 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 7, 3, 1), (5, 2, 4, 6, 0, 7))
    x490 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x490 += einsum(t2.abab, (0, 1, 2, 3), x489, (4, 1, 5, 6, 0, 7), (4, 3, 5, 6, 2, 7))
    del x489
    x491 = np.zeros((nvir[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x491 += einsum(v.aabb.vvvv, (0, 1, 2, 3), (2, 3, 0, 1))
    x491 += einsum(x132, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x132
    x492 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x492 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), (1, 4, 0, 2, 3, 5))
    x492 += einsum(t1.bb, (0, 1), t2.aaaa, (2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x493 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x493 += einsum(x491, (0, 1, 2, 3), x492, (4, 0, 5, 6, 2, 7), (4, 1, 5, 6, 7, 3)) * 2.0
    del x491
    x494 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x494 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x494 += einsum(x129, (0, 1, 2, 3), (1, 0, 3, 2))
    del x129
    x495 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x495 += einsum(x494, (0, 1, 2, 3), t3.abaaba, (4, 0, 5, 6, 7, 2), (1, 7, 4, 5, 6, 3)) * -2.0
    del x494
    x496 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x496 += einsum(t1.bb, (0, 1), v.aabb.vvoo, (2, 3, 4, 0), (4, 1, 2, 3))
    x497 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x497 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.vvov, (4, 5, 1, 3), (0, 2, 4, 5))
    x498 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x498 += einsum(t2.abab, (0, 1, 2, 3), x118, (0, 2, 4, 5), (1, 3, 4, 5))
    del x118
    x499 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x499 += einsum(t2.abab, (0, 1, 2, 3), x4, (1, 4, 0, 5), (4, 3, 5, 2))
    x500 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x500 += einsum(t2.abab, (0, 1, 2, 3), x104, (3, 4, 0, 5), (1, 4, 5, 2))
    x501 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x501 += einsum(v.aabb.vvov, (0, 1, 2, 3), (2, 3, 0, 1))
    x501 += einsum(x496, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x496
    x501 += einsum(x497, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x497
    x501 += einsum(x324, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x324
    x501 += einsum(x325, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x325
    x501 += einsum(x498, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x498
    x501 += einsum(x499, (0, 1, 2, 3), (0, 1, 2, 3))
    del x499
    x501 += einsum(x500, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x500
    x502 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x502 += einsum(t2.aaaa, (0, 1, 2, 3), x501, (4, 5, 3, 6), (4, 5, 0, 1, 2, 6)) * -2.0
    del x501
    x503 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x503 += einsum(v.aabb.ovov, (0, 1, 2, 3), x384, (4, 3, 0, 5), (2, 4, 1, 5))
    del x384
    x504 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x504 += einsum(x344, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.99999999999999
    del x344
    x504 += einsum(x503, (0, 1, 2, 3), (0, 1, 3, 2))
    del x503
    x505 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x505 += einsum(x504, (0, 1, 2, 3), t3.abaaba, (4, 0, 5, 6, 7, 3), (1, 7, 4, 5, 6, 2)) * -2.00000000000002
    del x504
    x506 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x506 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x506 += einsum(x72, (0, 1), (0, 1))
    del x72
    x506 += einsum(x73, (0, 1), (0, 1)) * 2.0
    del x73
    x506 += einsum(x139, (0, 1), (1, 0))
    del x139
    x507 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x507 += einsum(x506, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (3, 6, 2, 4, 5, 0)) * -2.0
    del x506
    x508 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x508 += einsum(x255, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 0), (3, 6, 2, 4, 5, 1)) * -2.0
    del x255
    x509 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x509 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 1, 7, 3), (5, 7, 4, 6, 0, 2))
    x510 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x510 += einsum(x104, (0, 1, 2, 3), x492, (4, 0, 5, 6, 3, 7), (4, 1, 2, 5, 6, 7))
    del x492
    x511 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x511 += einsum(t1.bb, (0, 1), v.aabb.ovoo, (2, 3, 4, 0), (4, 1, 2, 3))
    x512 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x512 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 2, 4, 5))
    x513 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x513 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x513 += einsum(x511, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x513 += einsum(x512, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x512
    x513 += einsum(x163, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x163
    x514 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x514 += einsum(t2.aaaa, (0, 1, 2, 3), x513, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2)) * -1.0
    del x513
    x515 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x515 += einsum(x509, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x509
    x515 += einsum(x510, (0, 1, 2, 3, 4, 5), (0, 1, 4, 3, 2, 5))
    del x510
    x515 += einsum(x514, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x514
    x516 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x516 += einsum(t1.aa, (0, 1), x515, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x515
    x517 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x517 += einsum(x77, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x77
    x517 += einsum(x78, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x78
    x517 += einsum(x79, (0, 1, 2, 3), (0, 1, 2, 3)) * -3.0
    del x79
    x518 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x518 += einsum(t2.abab, (0, 1, 2, 3), x517, (4, 5, 0, 6), (1, 3, 4, 5, 6, 2)) * 2.0
    del x517
    x519 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x519 += einsum(x490, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -2.0
    del x490
    x519 += einsum(x493, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x493
    x519 += einsum(x495, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x495
    x519 += einsum(x502, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x502
    x519 += einsum(x505, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x505
    x519 += einsum(x507, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x507
    x519 += einsum(x508, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x508
    x519 += einsum(x516, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x516
    x519 += einsum(x518, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x518
    t3new_abaaba += einsum(x519, (0, 1, 2, 3, 4, 5), (3, 0, 2, 4, 1, 5))
    t3new_abaaba += einsum(x519, (0, 1, 2, 3, 4, 5), (3, 0, 2, 5, 1, 4)) * -1.0
    del x519
    x520 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x520 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 5, 2, 6, 7, 3), (4, 6, 5, 0, 7, 1))
    x521 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x521 += einsum(t2.abab, (0, 1, 2, 3), x64, (4, 5, 2, 6), (1, 3, 4, 0, 6, 5))
    del x64
    x522 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x522 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ooov, (4, 0, 5, 3), (1, 5, 4, 2))
    x523 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x523 += einsum(t2.abab, (0, 1, 2, 3), x522, (4, 1, 5, 6), (4, 3, 0, 5, 6, 2))
    del x522
    x524 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x524 += einsum(x38, (0, 1, 2, 3), t3.babbab, (4, 5, 0, 6, 7, 1), (4, 6, 2, 5, 7, 3))
    x525 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x525 += einsum(t2.abab, (0, 1, 2, 3), x23, (4, 3, 5, 0), (1, 4, 5, 2))
    x526 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x526 += einsum(t2.abab, (0, 1, 2, 3), x525, (4, 1, 5, 6), (4, 3, 5, 0, 6, 2))
    del x525
    x527 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x527 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.00000000000002
    x527 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -1.0
    x528 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x528 += einsum(x527, (0, 1, 2, 3), x68, (0, 4, 5, 2), (4, 1, 5, 3))
    del x68, x527
    x529 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x529 += einsum(x225, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.00000000000001
    x529 += einsum(x528, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x528
    x530 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x530 += einsum(x529, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 7, 3), (5, 7, 4, 0, 6, 2)) * 2.0
    del x529
    x531 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x531 += einsum(v.aabb.ovov, (0, 1, 2, 3), x271, (0, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    del x271
    x532 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x532 += einsum(x61, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x61
    x532 += einsum(x531, (0, 1, 2, 3), (0, 1, 2, 3))
    del x531
    x533 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x533 += einsum(x532, (0, 1, 2, 3), t3.babbab, (4, 5, 0, 6, 7, 1), (4, 6, 2, 5, 3, 7)) * 2.00000000000002
    del x532
    x534 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x534 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), (1, 4, 0, 2, 3, 5))
    x534 += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (3, 5, 0, 2, 4, 1)) * -0.5
    x535 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x535 += einsum(x28, (0, 1, 2, 3), x534, (4, 5, 0, 6, 2, 7), (4, 5, 1, 6, 3, 7)) * 2.0
    del x28, x534
    x536 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x536 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), (1, 4, 0, 2, 3, 5)) * 2.0
    x536 += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (3, 5, 0, 2, 4, 1)) * -1.0
    x537 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x537 += einsum(x46, (0, 1, 2, 3), x536, (4, 5, 0, 6, 3, 7), (4, 5, 1, 6, 2, 7)) * -1.0
    del x46, x536
    x538 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x538 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ooov, (4, 0, 1, 5), (3, 5, 4, 2))
    x539 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x539 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvov, (4, 2, 1, 5), (3, 5, 0, 4))
    x540 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x540 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x540 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x540 += einsum(x107, (0, 1, 2, 3), (0, 2, 3, 1))
    x540 += einsum(x107, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    del x107
    x541 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x541 += einsum(t2.abab, (0, 1, 2, 3), x540, (1, 4, 3, 5), (4, 5, 0, 2))
    del x540
    x542 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x542 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (2, 3, 0, 1))
    x542 += einsum(x103, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x543 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x543 += einsum(t2.aaaa, (0, 1, 2, 3), x542, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    del x542
    x544 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x544 += einsum(x9, (0, 1), t2.abab, (2, 0, 3, 4), (1, 4, 2, 3))
    del x9
    x545 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x545 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x545 += einsum(x103, (0, 1, 2, 3), (1, 0, 2, 3))
    del x103
    x545 += einsum(x538, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x538
    x545 += einsum(x539, (0, 1, 2, 3), (1, 0, 2, 3))
    del x539
    x545 += einsum(x455, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x455
    x545 += einsum(x429, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x429
    x545 += einsum(x541, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x541
    x545 += einsum(x543, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x543
    x545 += einsum(x544, (0, 1, 2, 3), (0, 1, 2, 3))
    del x544
    x546 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x546 += einsum(t2.abab, (0, 1, 2, 3), x545, (3, 4, 5, 6), (1, 4, 5, 0, 6, 2))
    del x545
    x547 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x547 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x547 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x548 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x548 += einsum(t2.aaaa, (0, 1, 2, 3), x547, (1, 3, 4, 5), (0, 2, 4, 5)) * 2.0
    del x547
    x549 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x549 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    x550 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x550 += einsum(x549, (0, 1, 2, 3), (1, 0, 2, 3))
    del x549
    x550 += einsum(x225, (0, 1, 2, 3), (0, 1, 2, 3))
    del x225
    x550 += einsum(x226, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x226
    x551 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x551 += einsum(t1.aa, (0, 1), x550, (2, 0, 3, 4), (2, 3, 4, 1))
    del x550
    x552 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x552 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x552 += einsum(x219, (0, 1, 2, 3), (0, 1, 3, 2))
    del x219
    x552 += einsum(x548, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x548
    x552 += einsum(x551, (0, 1, 2, 3), (0, 3, 1, 2))
    del x551
    x553 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x553 += einsum(t2.abab, (0, 1, 2, 3), x552, (4, 5, 6, 2), (1, 3, 4, 0, 5, 6))
    del x552
    x554 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x554 += einsum(v.aabb.ooov, (0, 1, 2, 3), t3.babbab, (4, 5, 2, 6, 7, 3), (4, 6, 5, 0, 1, 7))
    x555 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x555 += einsum(t2.abab, (0, 1, 2, 3), x25, (4, 5, 6, 2), (1, 3, 4, 0, 6, 5))
    x556 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x556 += einsum(t1.aa, (0, 1), x555, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x555
    x557 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x557 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x557 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x558 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x558 += einsum(x557, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 7, 3), (5, 7, 4, 1, 2, 6)) * 2.0
    del x557
    x559 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x559 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 5), (3, 5, 0, 4))
    x560 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x560 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x560 += einsum(x121, (0, 1, 2, 3), (1, 0, 3, 2))
    x560 += einsum(x559, (0, 1, 2, 3), (1, 0, 3, 2))
    del x559
    x561 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x561 += einsum(t2.abab, (0, 1, 2, 3), x560, (3, 4, 5, 6), (1, 4, 5, 6, 0, 2))
    del x560
    x562 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x562 += einsum(x136, (0, 1, 2, 3), (1, 0, 2, 3))
    x562 += einsum(x137, (0, 1, 2, 3), (1, 0, 2, 3))
    del x137
    x563 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x563 += einsum(t2.abab, (0, 1, 2, 3), x562, (1, 4, 5, 6), (4, 3, 5, 6, 0, 2))
    x564 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x564 += einsum(t1.bb, (0, 1), x23, (0, 2, 3, 4), (1, 2, 3, 4))
    x565 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x565 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3))
    x565 += einsum(x564, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x564
    x566 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x566 += einsum(t2.abab, (0, 1, 2, 3), x565, (3, 4, 5, 6), (1, 4, 5, 6, 0, 2))
    del x565
    x567 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x567 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x567 += einsum(x135, (0, 1, 2, 3), (1, 0, 3, 2))
    del x135
    x568 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x568 += einsum(t2.abab, (0, 1, 2, 3), x567, (1, 4, 5, 6), (4, 3, 5, 6, 0, 2))
    x569 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x569 += einsum(x554, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    del x554
    x569 += einsum(x556, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x556
    x569 += einsum(x558, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x558
    x569 += einsum(x561, (0, 1, 2, 3, 4, 5), (0, 1, 4, 3, 2, 5)) * -1.0
    del x561
    x569 += einsum(x563, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x563
    x569 += einsum(x566, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x566
    x569 += einsum(x568, (0, 1, 2, 3, 4, 5), (0, 1, 4, 3, 2, 5)) * -1.0
    del x568
    x570 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x570 += einsum(t1.aa, (0, 1), x569, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 6, 1))
    del x569
    x571 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x571 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x571 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x571 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    x571 += einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x572 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x572 += einsum(t2.abab, (0, 1, 2, 3), x571, (4, 5, 1, 3), (4, 5, 0, 2))
    del x571
    x573 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x573 += einsum(t2.aaaa, (0, 1, 2, 3), x130, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    del x130
    x574 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x574 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x574 += einsum(x3, (0, 1, 2, 3), (1, 0, 2, 3))
    del x3
    x574 += einsum(x391, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x391
    x574 += einsum(x392, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x392
    x574 += einsum(x572, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x572
    x574 += einsum(x573, (0, 1, 2, 3), (1, 0, 2, 3))
    del x573
    x575 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x575 += einsum(t2.abab, (0, 1, 2, 3), x574, (1, 4, 5, 6), (4, 3, 5, 0, 6, 2))
    del x574
    x576 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x576 += einsum(t1.bb, (0, 1), x38, (0, 2, 3, 4), (1, 2, 3, 4))
    del x38
    x577 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x577 += einsum(t2.abab, (0, 1, 2, 3), x23, (1, 4, 5, 0), (3, 4, 5, 2))
    x578 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x578 += einsum(x153, (0, 1, 2, 3), (1, 0, 2, 3))
    del x153
    x578 += einsum(x576, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x576
    x578 += einsum(x577, (0, 1, 2, 3), (1, 0, 2, 3))
    del x577
    x579 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x579 += einsum(t2.abab, (0, 1, 2, 3), x578, (3, 4, 5, 6), (1, 4, 5, 0, 6, 2))
    del x578
    x580 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x580 += einsum(v.aabb.vvov, (0, 1, 2, 3), x5, (4, 3, 5, 1), (4, 2, 5, 0))
    del x5
    x581 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x581 += einsum(x147, (0, 1, 2, 3), (1, 0, 2, 3))
    del x147
    x581 += einsum(x580, (0, 1, 2, 3), (1, 0, 2, 3))
    del x580
    x582 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x582 += einsum(t2.abab, (0, 1, 2, 3), x581, (1, 4, 5, 6), (4, 3, 5, 0, 6, 2))
    del x581
    x583 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x583 += einsum(x520, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    del x520
    x583 += einsum(x521, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x521
    x583 += einsum(x523, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x523
    x583 += einsum(x524, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    del x524
    x583 += einsum(x526, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x526
    x583 += einsum(x530, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x530
    x583 += einsum(x533, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x533
    x583 += einsum(x535, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x535
    x583 += einsum(x537, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x537
    x583 += einsum(x546, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x546
    x583 += einsum(x553, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x553
    x583 += einsum(x570, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x570
    x583 += einsum(x575, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x575
    x583 += einsum(x579, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x579
    x583 += einsum(x582, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x582
    t3new_abaaba += einsum(x583, (0, 1, 2, 3, 4, 5), (2, 0, 3, 4, 1, 5)) * -1.0
    t3new_abaaba += einsum(x583, (0, 1, 2, 3, 4, 5), (2, 0, 3, 5, 1, 4))
    t3new_abaaba += einsum(x583, (0, 1, 2, 3, 4, 5), (3, 0, 2, 4, 1, 5))
    t3new_abaaba += einsum(x583, (0, 1, 2, 3, 4, 5), (3, 0, 2, 5, 1, 4)) * -1.0
    del x583
    x584 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x584 += einsum(t2.abab, (0, 1, 2, 3), x261, (4, 5, 6, 2), (1, 3, 4, 0, 5, 6)) * -1.0
    del x261
    x585 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x585 += einsum(x122, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.99999999999999
    del x122
    x585 += einsum(x436, (0, 1, 2, 3), (0, 1, 3, 2))
    del x436
    x586 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x586 += einsum(x585, (0, 1, 2, 3), t3.abaaba, (4, 5, 3, 6, 0, 7), (5, 1, 4, 2, 6, 7)) * -2.00000000000002
    del x585
    x587 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x587 += einsum(x562, (0, 1, 2, 3), t3.abaaba, (4, 0, 3, 5, 6, 7), (1, 6, 4, 2, 5, 7)) * -2.0
    del x562
    x588 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x588 += einsum(t1.bb, (0, 1), v.aabb.oooo, (2, 3, 4, 0), (4, 1, 2, 3))
    x589 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x589 += einsum(t2.abab, (0, 1, 2, 3), x4, (1, 4, 5, 2), (4, 3, 5, 0))
    del x4
    x590 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x590 += einsum(t2.abab, (0, 1, 2, 3), x32, (4, 5, 0, 2), (1, 3, 4, 5))
    x591 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x591 += einsum(t2.abab, (0, 1, 2, 3), x104, (3, 4, 5, 2), (1, 4, 5, 0))
    del x104
    x592 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x592 += einsum(x13, (0, 1), t2.abab, (2, 3, 1, 4), (3, 4, 0, 2))
    del x13
    x593 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x593 += einsum(v.aabb.ooov, (0, 1, 2, 3), x113, (2, 4, 3, 5), (4, 5, 0, 1)) * 2.0
    x594 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x594 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x594 += einsum(x588, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x588
    x594 += einsum(x155, (0, 1, 2, 3), (0, 1, 3, 2))
    del x155
    x594 += einsum(x360, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x360
    x594 += einsum(x361, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x361
    x594 += einsum(x589, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x589
    x594 += einsum(x590, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x590
    x594 += einsum(x591, (0, 1, 2, 3), (0, 1, 2, 3))
    del x591
    x594 += einsum(x592, (0, 1, 2, 3), (0, 1, 2, 3))
    del x592
    x594 += einsum(x593, (0, 1, 2, 3), (0, 1, 3, 2))
    del x593
    x595 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x595 += einsum(t2.aaaa, (0, 1, 2, 3), x594, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3)) * -2.0
    del x594
    x596 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x596 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x596 += einsum(x121, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x121
    x597 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x597 += einsum(x596, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 6, 0, 7), (5, 1, 4, 3, 6, 7)) * -2.0
    del x596
    x598 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x598 += einsum(x214, (0, 1, 2, 3), (0, 2, 1, 3))
    del x214
    x598 += einsum(x240, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x240
    x598 += einsum(x241, (0, 1, 2, 3), (0, 2, 1, 3)) * -3.0
    del x241
    x599 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x599 += einsum(t2.abab, (0, 1, 2, 3), x598, (4, 5, 6, 2), (1, 3, 4, 0, 5, 6)) * 2.0
    del x598
    x600 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x600 += einsum(x567, (0, 1, 2, 3), t3.abaaba, (4, 0, 2, 5, 6, 7), (1, 6, 4, 3, 5, 7)) * -2.0
    del x567
    x601 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x601 += einsum(f.aa.oo, (0, 1), (0, 1))
    x601 += einsum(x30, (0, 1), (0, 1))
    del x30
    x601 += einsum(x31, (0, 1), (0, 1)) * 2.0
    del x31
    x601 += einsum(x34, (0, 1), (1, 0))
    del x34
    x602 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x602 += einsum(x601, (0, 1), t3.abaaba, (2, 3, 1, 4, 5, 6), (3, 5, 2, 0, 4, 6)) * -2.0
    del x601
    x603 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x603 += einsum(t1.bb, (0, 1), x136, (2, 0, 3, 4), (2, 1, 3, 4))
    del x136
    x604 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x604 += einsum(t2.abab, (0, 1, 2, 3), x55, (4, 0, 5, 2), (1, 3, 4, 5))
    x605 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x605 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x605 += einsum(x127, (0, 1, 2, 3), (0, 1, 2, 3))
    x606 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x606 += einsum(t1.aa, (0, 1), x605, (2, 3, 4, 1), (2, 3, 4, 0))
    del x605
    x607 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x607 += einsum(x113, (0, 1, 2, 3), x23, (0, 2, 4, 5), (1, 3, 4, 5)) * 2.0
    del x23
    x608 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x608 += einsum(x603, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x603
    x608 += einsum(x604, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x604
    x608 += einsum(x606, (0, 1, 2, 3), (0, 1, 3, 2))
    del x606
    x608 += einsum(x607, (0, 1, 2, 3), (0, 1, 2, 3))
    del x607
    x609 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x609 += einsum(t2.aaaa, (0, 1, 2, 3), x608, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3)) * -2.0
    del x608
    x610 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x610 += einsum(x251, (0, 1), t3.abaaba, (2, 3, 0, 4, 5, 6), (3, 5, 2, 1, 4, 6)) * -2.0
    del x251
    x611 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x611 += einsum(x584, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    del x584
    x611 += einsum(x586, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x586
    x611 += einsum(x587, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x587
    x611 += einsum(x595, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x595
    x611 += einsum(x597, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x597
    x611 += einsum(x599, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x599
    x611 += einsum(x600, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x600
    x611 += einsum(x602, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x602
    x611 += einsum(x609, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x609
    x611 += einsum(x610, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x610
    t3new_abaaba += einsum(x611, (0, 1, 2, 3, 4, 5), (2, 0, 3, 4, 1, 5)) * -1.0
    t3new_abaaba += einsum(x611, (0, 1, 2, 3, 4, 5), (3, 0, 2, 4, 1, 5))
    del x611
    x612 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x612 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x612 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x612 += einsum(x111, (0, 1, 2, 3), (1, 0, 3, 2)) * 1.00000000000001
    x612 += einsum(x112, (0, 1, 2, 3), x315, (0, 4, 3, 5), (1, 4, 2, 5)) * -2.00000000000002
    del x112
    x612 += einsum(x115, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x115
    x612 += einsum(x116, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x116
    t3new_abaaba += einsum(x612, (0, 1, 2, 3), t3.abaaba, (4, 0, 5, 6, 2, 7), (4, 1, 5, 6, 3, 7)) * 2.0
    del x612
    x613 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x613 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x613 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -0.4999999999999949
    x614 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x614 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x614 += einsum(x511, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x511
    x614 += einsum(x127, (0, 1, 2, 3), (0, 1, 2, 3))
    x614 += einsum(x314, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x314
    x614 += einsum(v.aabb.ovov, (0, 1, 2, 3), x613, (2, 4, 3, 5), (4, 5, 0, 1)) * 2.0000000000000204
    del x613
    t3new_abaaba += einsum(x614, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 2, 6, 7, 3), (4, 0, 5, 6, 1, 7)) * 6.0
    del x614
    x615 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x615 += einsum(v.aaaa.vvvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x615 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 4, 1, 5), (5, 2, 3, 4)) * -1.0
    x615 += einsum(t1.aa, (0, 1), x109, (0, 2, 3, 4), (4, 1, 2, 3)) * -1.0
    del x109
    t3new_abaaba += einsum(x615, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 0, 7, 3), (4, 5, 6, 2, 7, 1)) * -2.0
    del x615
    x616 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x616 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x616 += einsum(x49, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    x616 += einsum(x49, (0, 1, 2, 3), (2, 1, 3, 0))
    del x49
    x616 += einsum(x99, (0, 1, 2, 3), (1, 3, 0, 2))
    del x99
    t3new_abaaba += einsum(x616, (0, 1, 2, 3), t3.abaaba, (0, 4, 2, 5, 6, 7), (1, 4, 3, 5, 6, 7)) * 2.0
    del x616
    x617 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x617 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x617 += einsum(x50, (0, 1, 2, 3), (1, 0, 2, 3))
    del x50
    x618 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x618 += einsum(t2.aaaa, (0, 1, 2, 3), x32, (4, 5, 1, 3), (0, 4, 5, 2)) * 2.0
    del x32
    x619 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x619 += einsum(t2.aaaa, (0, 1, 2, 3), x55, (4, 5, 1, 3), (0, 4, 5, 2)) * 2.0
    del x55
    x620 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x620 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x620 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x620 += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3))
    del x37
    x621 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x621 += einsum(t1.aa, (0, 1), x620, (2, 3, 1, 4), (2, 3, 0, 4))
    del x620
    x622 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x622 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x622 += einsum(x25, (0, 1, 2, 3), (0, 2, 1, 3))
    del x25
    x623 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x623 += einsum(t1.aa, (0, 1), x622, (2, 3, 4, 1), (2, 3, 4, 0))
    del x622
    x624 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x624 += einsum(t1.aa, (0, 1), x623, (2, 0, 3, 4), (4, 2, 3, 1))
    del x623
    x625 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x625 += einsum(x48, (0, 1, 2, 3), (0, 2, 1, 3))
    del x48
    x625 += einsum(x52, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x52
    x625 += einsum(x618, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x618
    x625 += einsum(x619, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x619
    x625 += einsum(x621, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x621
    x625 += einsum(x624, (0, 1, 2, 3), (0, 2, 1, 3))
    del x624
    x626 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x626 += einsum(x617, (0, 1, 2, 3), (0, 1, 2, 3))
    x626 += einsum(x617, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x617
    x626 += einsum(x625, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x626 += einsum(x625, (0, 1, 2, 3), (1, 2, 0, 3))
    del x625
    x626 += einsum(x80, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x80
    x626 += einsum(x295, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x295
    x627 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x627 += einsum(t2.abab, (0, 1, 2, 3), x626, (0, 4, 5, 6), (1, 3, 4, 5, 6, 2))
    del x626
    t3new_abaaba += einsum(x627, (0, 1, 2, 3, 4, 5), (3, 0, 2, 4, 1, 5)) * -1.0
    t3new_abaaba += einsum(x627, (0, 1, 2, 3, 4, 5), (3, 0, 2, 5, 1, 4))
    del x627
    x628 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x628 += einsum(t2.bbbb, (0, 1, 2, 3), x430, (1, 4, 3, 5), (0, 4, 5, 2)) * 2.0
    del x430
    x629 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x629 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x629 += einsum(x331, (0, 1, 2, 3), (0, 1, 3, 2))
    del x331
    x629 += einsum(x628, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    del x628
    x630 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x630 += einsum(t2.bbbb, (0, 1, 2, 3), x629, (4, 5, 3, 6), (4, 0, 1, 5, 6, 2)) * -2.0
    del x629
    x631 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x631 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    x632 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x632 += einsum(t1.bb, (0, 1), x631, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x631
    x633 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x633 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x633 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x633 += einsum(x111, (0, 1, 2, 3), (0, 1, 2, 3))
    x633 += einsum(x335, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x335
    x634 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x634 += einsum(t2.bbbb, (0, 1, 2, 3), x633, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2)) * -1.0
    del x633
    x635 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x635 += einsum(x632, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x632
    x635 += einsum(x634, (0, 1, 2, 3, 4, 5), (3, 2, 1, 0, 5, 4)) * -1.0
    del x634
    x636 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x636 += einsum(t1.bb, (0, 1), x635, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x635
    x637 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x637 += einsum(x630, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x630
    x637 += einsum(x636, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    del x636
    t3new_bbbbbb = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new_bbbbbb += einsum(x637, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x637
    x638 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x638 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 1, 7), (4, 5, 2, 6, 7, 3))
    x639 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x639 += einsum(t2.bbbb, (0, 1, 2, 3), x402, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    del x402
    x640 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x640 += einsum(t2.bbbb, (0, 1, 2, 3), x167, (4, 5, 1, 6), (4, 5, 0, 2, 3, 6))
    del x167
    x641 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x641 += einsum(x14, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 0, 6, 7, 2), (4, 5, 1, 6, 7, 3)) * 6.0
    del x14
    x642 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x642 += einsum(x638, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    del x638
    x642 += einsum(x639, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -4.0
    del x639
    x642 += einsum(x640, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -4.0
    del x640
    x642 += einsum(x641, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x641
    t3new_bbbbbb += einsum(x642, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x642, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_bbbbbb += einsum(x642, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x642, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new_bbbbbb += einsum(x642, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x642, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new_bbbbbb += einsum(x642, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new_bbbbbb += einsum(x642, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x642, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x642
    x643 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x643 += einsum(x127, (0, 1, 2, 3), t3.babbab, (4, 2, 5, 6, 3, 7), (0, 4, 5, 6, 7, 1))
    del x127
    x644 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x644 += einsum(t2.bbbb, (0, 1, 2, 3), x382, (4, 5, 6, 3), (4, 0, 1, 5, 6, 2)) * -1.0
    del x382
    x645 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x645 += einsum(x182, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 0, 6, 7, 3), (4, 5, 1, 6, 7, 2)) * -6.0
    x646 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x646 += einsum(x168, (0, 1, 2, 3), (0, 1, 2, 3))
    del x168
    x646 += einsum(x169, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.3333333333333333
    del x169
    x647 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x647 += einsum(t2.bbbb, (0, 1, 2, 3), x646, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3)) * -12.0
    del x646
    x648 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x648 += einsum(x643, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    del x643
    x648 += einsum(x644, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -4.0
    del x644
    x648 += einsum(x645, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5))
    del x645
    x648 += einsum(x647, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    del x647
    t3new_bbbbbb += einsum(x648, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new_bbbbbb += einsum(x648, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x648, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new_bbbbbb += einsum(x648, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5))
    t3new_bbbbbb += einsum(x648, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x648, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new_bbbbbb += einsum(x648, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x648, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new_bbbbbb += einsum(x648, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    del x648
    x649 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x649 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 6, 7, 1, 3), (4, 5, 6, 0, 7, 2))
    x650 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x650 += einsum(t1.bb, (0, 1), x649, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x649
    t3new_bbbbbb += einsum(x650, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 6.0
    t3new_bbbbbb += einsum(x650, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x650, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -6.0
    t3new_bbbbbb += einsum(x650, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x650, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x650, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x650
    x651 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x651 += einsum(x440, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * 6.0
    del x440
    x652 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x652 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 6, 7, 3, 1), (4, 5, 6, 0, 2, 7)) * -1.0
    x653 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x653 += einsum(x470, (0, 1, 2, 3), x652, (4, 5, 6, 0, 1, 7), (4, 5, 6, 2, 3, 7)) * 6.0
    del x470, x652
    x654 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x654 += einsum(x651, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    del x651
    x654 += einsum(x653, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x653
    t3new_bbbbbb += einsum(x654, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new_bbbbbb += einsum(x654, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x654, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    del x654
    x655 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x655 += einsum(t2.bbbb, (0, 1, 2, 3), x157, (4, 5, 3, 6), (4, 0, 1, 2, 6, 5))
    del x157
    x656 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x656 += einsum(t2.bbbb, (0, 1, 2, 3), x1, (4, 5, 6, 3), (4, 0, 1, 6, 5, 2))
    del x1
    x657 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x657 += einsum(t1.bb, (0, 1), x656, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x656
    x658 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x658 += einsum(t2.bbbb, (0, 1, 2, 3), x182, (4, 5, 6, 3), (5, 4, 0, 1, 6, 2))
    del x182
    x659 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x659 += einsum(x657, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    del x657
    x659 += einsum(x658, (0, 1, 2, 3, 4, 5), (0, 3, 2, 1, 5, 4)) * -1.0
    del x658
    x660 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x660 += einsum(t1.bb, (0, 1), x659, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x659
    x661 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x661 += einsum(x655, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    del x655
    x661 += einsum(x660, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    del x660
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new_bbbbbb += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x661
    x662 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x662 += einsum(v.bbbb.oooo, (0, 1, 2, 3), t3.bbbbbb, (4, 3, 1, 5, 6, 7), (4, 0, 2, 5, 6, 7)) * -1.0
    x663 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x663 += einsum(f.bb.oo, (0, 1), (0, 1))
    x663 += einsum(x16, (0, 1), (1, 0)) * 2.0
    del x16
    x663 += einsum(x17, (0, 1), (1, 0))
    del x17
    x663 += einsum(x20, (0, 1), (0, 1))
    del x20
    x664 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x664 += einsum(x663, (0, 1), t3.bbbbbb, (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * 6.0
    del x663
    x665 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x665 += einsum(x662, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    del x662
    x665 += einsum(x664, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3))
    del x664
    t3new_bbbbbb += einsum(x665, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x665, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x665, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x665
    x666 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x666 += einsum(v.bbbb.vvvv, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 6, 7, 3, 1), (4, 5, 6, 7, 0, 2)) * -1.0
    x667 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x667 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x667 += einsum(x142, (0, 1), (1, 0)) * 2.0
    del x142
    x667 += einsum(x143, (0, 1), (1, 0))
    del x143
    x667 += einsum(x145, (0, 1), (0, 1))
    del x145
    x668 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x668 += einsum(x667, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * 6.0
    del x667
    x669 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x669 += einsum(x666, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    del x666
    x669 += einsum(x668, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    del x668
    t3new_bbbbbb += einsum(x669, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new_bbbbbb += einsum(x669, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x669, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x669
    x670 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x670 += einsum(x210, (0, 1, 2, 3), (1, 0, 3, 2))
    del x210
    x670 += einsum(x171, (0, 1, 2, 3), (0, 1, 2, 3))
    x671 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x671 += einsum(x670, (0, 1, 2, 3), t3.bbbbbb, (4, 2, 3, 5, 6, 7), (4, 0, 1, 5, 6, 7)) * -6.0
    del x670
    x672 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x672 += einsum(x417, (0, 1), t3.bbbbbb, (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * 6.0
    del x417
    x673 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x673 += einsum(x671, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x671
    x673 += einsum(x672, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    del x672
    t3new_bbbbbb += einsum(x673, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new_bbbbbb += einsum(x673, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x673, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    del x673
    x674 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x674 += einsum(x403, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x403
    x674 += einsum(x404, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.3333333333333333
    del x404
    x675 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x675 += einsum(t2.bbbb, (0, 1, 2, 3), x674, (4, 5, 6, 3), (4, 0, 1, 5, 6, 2)) * -12.0
    del x674
    x676 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x676 += einsum(v.aabb.ovoo, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 1, 7), (4, 5, 2, 3, 6, 7))
    x677 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x677 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x677 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x678 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x678 += einsum(x677, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 0, 6, 7, 3), (4, 5, 1, 2, 6, 7))
    del x677
    x679 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x679 += einsum(x676, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -0.3333333333333333
    del x676
    x679 += einsum(x678, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x678
    x680 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x680 += einsum(t1.bb, (0, 1), x679, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 6, 1)) * 6.0
    del x679
    x681 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x681 += einsum(x675, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x675
    x681 += einsum(x680, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    del x680
    t3new_bbbbbb += einsum(x681, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new_bbbbbb += einsum(x681, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x681, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new_bbbbbb += einsum(x681, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x681, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new_bbbbbb += einsum(x681, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_bbbbbb += einsum(x681, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x681, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    t3new_bbbbbb += einsum(x681, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    del x681
    x682 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x682 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0000000000000204
    x682 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -1.0
    x683 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x683 += einsum(x682, (0, 1, 2, 3), x7, (0, 4, 2, 5), (4, 1, 5, 3)) * 0.4999999999999949
    del x7, x682
    x684 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x684 += einsum(x111, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.49999999999999983
    del x111
    x684 += einsum(x683, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x683
    x685 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x685 += einsum(x684, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 12.000000000000123
    del x684
    x686 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x686 += einsum(t2.abab, (0, 1, 2, 3), x11, (0, 4, 2, 5), (1, 3, 4, 5)) * 0.5
    del x11
    x687 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x687 += einsum(v.aabb.ovov, (0, 1, 2, 3), x315, (2, 4, 3, 5), (4, 5, 0, 1))
    del x315
    x688 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x688 += einsum(x686, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x686
    x688 += einsum(x687, (0, 1, 2, 3), (0, 1, 2, 3))
    del x687
    x689 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x689 += einsum(x688, (0, 1, 2, 3), t3.babbab, (4, 2, 5, 6, 3, 7), (0, 4, 5, 1, 6, 7)) * 4.00000000000004
    del x688
    x690 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x690 += einsum(x685, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x685
    x690 += einsum(x689, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    del x689
    t3new_bbbbbb += einsum(x690, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new_bbbbbb += einsum(x690, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x690, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new_bbbbbb += einsum(x690, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5))
    t3new_bbbbbb += einsum(x690, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_bbbbbb += einsum(x690, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3))
    t3new_bbbbbb += einsum(x690, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new_bbbbbb += einsum(x690, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    t3new_bbbbbb += einsum(x690, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    del x690
    x691 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x691 += einsum(x184, (0, 1, 2, 3), t3.bbbbbb, (4, 2, 3, 5, 6, 7), (0, 4, 1, 5, 6, 7))
    del x184
    t3new_bbbbbb += einsum(x691, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x691, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x691, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x691, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x691, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x691, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -6.0
    del x691
    x692 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x692 += einsum(t1.bb, (0, 1), v.bbbb.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x693 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x693 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x694 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x694 += einsum(x692, (0, 1, 2, 3), (0, 2, 1, 3))
    del x692
    x694 += einsum(x693, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    del x693
    x694 += einsum(x186, (0, 1, 2, 3), (0, 2, 1, 3))
    del x186
    x695 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x695 += einsum(t1.bb, (0, 1), x171, (2, 3, 0, 4), (3, 2, 4, 1))
    del x171
    x696 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x696 += einsum(t2.bbbb, (0, 1, 2, 3), x412, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    del x412
    x697 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x697 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x697 += einsum(x177, (0, 1, 2, 3), (1, 0, 2, 3))
    del x177
    x698 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x698 += einsum(t1.bb, (0, 1), x697, (2, 3, 1, 4), (2, 3, 0, 4))
    del x697
    x699 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x699 += einsum(v.bbbb.ooov, (0, 1, 2, 3), x113, (1, 4, 3, 5), (4, 0, 2, 5)) * 2.0
    del x113
    x700 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x700 += einsum(x695, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x695
    x700 += einsum(x187, (0, 1, 2, 3), (0, 1, 2, 3))
    del x187
    x700 += einsum(x696, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x696
    x700 += einsum(x698, (0, 1, 2, 3), (2, 1, 0, 3))
    del x698
    x700 += einsum(x699, (0, 1, 2, 3), (0, 1, 2, 3))
    del x699
    x701 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x701 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x701 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x701 += einsum(x694, (0, 1, 2, 3), (0, 1, 2, 3))
    x701 += einsum(x694, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x694
    x701 += einsum(x700, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x701 += einsum(x700, (0, 1, 2, 3), (1, 0, 2, 3))
    del x700
    x701 += einsum(x185, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x701 += einsum(x185, (0, 1, 2, 3), (1, 0, 2, 3))
    del x185
    x701 += einsum(x170, (0, 1, 2, 3), (2, 1, 0, 3))
    del x170
    x701 += einsum(x480, (0, 1, 2, 3), (1, 0, 2, 3))
    del x480
    x702 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x702 += einsum(t2.bbbb, (0, 1, 2, 3), x701, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3)) * -2.0
    del x701
    t3new_bbbbbb += einsum(x702, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new_bbbbbb += einsum(x702, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new_bbbbbb += einsum(x702, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new_bbbbbb += einsum(x702, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new_bbbbbb += einsum(x702, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new_bbbbbb += einsum(x702, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new_bbbbbb += einsum(x702, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new_bbbbbb += einsum(x702, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_bbbbbb += einsum(x702, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x702

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

def update_lams(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, l1=None, l2=None, l3=None, **kwargs):
    l1new = Namespace()
    l2new = Namespace()
    l3new = Namespace()

    # L amplitudes
    l1new_bb = np.zeros((nvir[1], nocc[1]), dtype=types[float])
    l1new_bb += einsum(f.bb.ov, (0, 1), (1, 0))
    l1new_bb += einsum(l2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (3, 0, 4, 1), (4, 2)) * -2.0
    l1new_aa = np.zeros((nvir[0], nocc[0]), dtype=types[float])
    l1new_aa += einsum(f.aa.ov, (0, 1), (1, 0))
    l2new_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    l2new_aaaa += einsum(l2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 1, 5, 0), (4, 5, 2, 3)) * -2.0
    l2new_abab = np.zeros((nvir[0], nvir[1], nocc[0], nocc[1]), dtype=types[float])
    l2new_abab += einsum(l1.aa, (0, 1), v.aabb.vvov, (2, 0, 3, 4), (2, 4, 1, 3))
    l2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (1, 3, 0, 2))
    l2new_abab += einsum(l1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 0), (3, 4, 2, 1))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 0, 5, 1), (4, 5, 2, 3))
    l2new_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    l2new_bbbb += einsum(l2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 0, 5, 1), (4, 5, 2, 3)) * 2.0
    l3new_aaaaaa = np.zeros((nvir[0], nvir[0], nvir[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (0, 5, 3, 1, 2, 4)) * -1.0
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (0, 3, 5, 1, 2, 4))
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (5, 0, 3, 1, 2, 4))
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (3, 0, 5, 1, 2, 4)) * -1.0
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (5, 3, 0, 1, 2, 4)) * -1.0
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (3, 5, 0, 1, 2, 4))
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (0, 5, 3, 2, 1, 4))
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (0, 3, 5, 2, 1, 4)) * -1.0
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (5, 0, 3, 2, 1, 4)) * -1.0
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (3, 0, 5, 2, 1, 4))
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (5, 3, 0, 2, 1, 4))
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (3, 5, 0, 2, 1, 4)) * -1.0
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (0, 5, 3, 2, 4, 1)) * -1.0
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (0, 3, 5, 2, 4, 1))
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (5, 0, 3, 2, 4, 1))
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (3, 0, 5, 2, 4, 1)) * -1.0
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (5, 3, 0, 2, 4, 1)) * -1.0
    l3new_aaaaaa += einsum(l1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (3, 5, 0, 2, 4, 1))
    l3new_babbab = np.zeros((nvir[1], nvir[0], nvir[1], nocc[1], nocc[0], nocc[1]), dtype=types[float])
    l3new_babbab += einsum(v.bbbb.vvvv, (0, 1, 2, 3), l3.babbab, (3, 4, 1, 5, 6, 7), (0, 4, 2, 5, 6, 7)) * -2.0
    l3new_abaaba = np.zeros((nvir[0], nvir[1], nvir[0], nocc[0], nocc[1], nocc[0]), dtype=types[float])
    l3new_abaaba += einsum(v.aaaa.vvvv, (0, 1, 2, 3), l3.abaaba, (1, 4, 3, 5, 6, 7), (0, 4, 2, 5, 6, 7)) * 2.0
    l3new_bbbbbb = np.zeros((nvir[1], nvir[1], nvir[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (0, 5, 3, 1, 2, 4)) * -1.0
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (0, 3, 5, 1, 2, 4))
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (5, 0, 3, 1, 2, 4))
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (3, 0, 5, 1, 2, 4)) * -1.0
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (5, 3, 0, 1, 2, 4)) * -1.0
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (3, 5, 0, 1, 2, 4))
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (0, 5, 3, 2, 1, 4))
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (0, 3, 5, 2, 1, 4)) * -1.0
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (5, 0, 3, 2, 1, 4)) * -1.0
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (3, 0, 5, 2, 1, 4))
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (5, 3, 0, 2, 1, 4))
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (3, 5, 0, 2, 1, 4)) * -1.0
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (0, 5, 3, 2, 4, 1)) * -1.0
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (0, 3, 5, 2, 4, 1))
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (5, 0, 3, 2, 4, 1))
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (3, 0, 5, 2, 4, 1)) * -1.0
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (5, 3, 0, 2, 4, 1)) * -1.0
    l3new_bbbbbb += einsum(l1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (3, 5, 0, 2, 4, 1))
    x0 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x0 += einsum(t2.bbbb, (0, 1, 2, 3), l3.bbbbbb, (4, 5, 3, 6, 0, 1), (6, 4, 5, 2))
    l1new_bb += einsum(v.bbbb.vvvv, (0, 1, 2, 3), x0, (4, 3, 1, 2), (0, 4)) * -6.0
    x1 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x1 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    x2 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x2 += einsum(t1.bb, (0, 1), x1, (2, 3, 4, 1), (3, 2, 4, 0))
    l1new_bb += einsum(v.bbbb.ooov, (0, 1, 2, 3), x2, (4, 1, 0, 2), (3, 4)) * -2.0
    x3 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x3 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x4 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x4 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x4 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x4 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    x4 += einsum(x3, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x5 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x5 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_abab += einsum(x1, (0, 1, 2, 3), x5, (1, 2, 4, 5), (5, 3, 4, 0)) * -2.0
    x6 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x6 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    l2new_abab += einsum(l1.bb, (0, 1), x6, (1, 2, 3, 4), (4, 0, 3, 2)) * -1.0
    x7 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x7 += einsum(t1.aa, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (3, 4, 0, 2))
    x8 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x8 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x8 += einsum(x7, (0, 1, 2, 3), (0, 1, 3, 2))
    x9 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x9 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 4, 3, 5))
    x10 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x10 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x10 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x11 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x11 += einsum(t2.bbbb, (0, 1, 2, 3), x10, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    x12 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x12 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x12 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x13 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x13 += einsum(t1.bb, (0, 1), x12, (2, 3, 1, 4), (0, 2, 3, 4))
    x14 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x14 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x14 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x14 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    x14 += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x14 += einsum(x13, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x15 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x15 += einsum(t1.aa, (0, 1), v.aabb.vvov, (2, 1, 3, 4), (3, 4, 0, 2))
    l2new_abab += einsum(l2.aaaa, (0, 1, 2, 3), x15, (4, 5, 3, 1), (0, 5, 2, 4)) * 2.0
    x16 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x16 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ovov, (1, 3, 4, 5), (4, 5, 0, 2))
    x17 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x17 += einsum(t2.abab, (0, 1, 2, 3), x10, (1, 4, 3, 5), (4, 5, 0, 2))
    x18 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x18 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x18 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    x18 += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x18 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x19 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x19 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    x20 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x20 += einsum(t1.bb, (0, 1), x10, (0, 2, 1, 3), (2, 3))
    x21 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x21 += einsum(f.bb.ov, (0, 1), (0, 1))
    x21 += einsum(x19, (0, 1), (0, 1))
    x21 += einsum(x20, (0, 1), (0, 1)) * -1.0
    l2new_abab += einsum(l1.aa, (0, 1), x21, (2, 3), (0, 3, 1, 2))
    l3new_babbab += einsum(x21, (0, 1), l2.abab, (2, 3, 4, 5), (1, 2, 3, 0, 4, 5))
    l3new_babbab += einsum(x21, (0, 1), l2.abab, (2, 3, 4, 5), (3, 2, 1, 5, 4, 0))
    l3new_babbab += einsum(x21, (0, 1), l2.abab, (2, 3, 4, 5), (3, 2, 1, 0, 4, 5)) * -1.0
    l3new_babbab += einsum(x21, (0, 1), l2.abab, (2, 3, 4, 5), (1, 2, 3, 5, 4, 0)) * -1.0
    l3new_abaaba += einsum(x21, (0, 1), l2.aaaa, (2, 3, 4, 5), (2, 1, 3, 4, 0, 5)) * 2.0
    l3new_bbbbbb += einsum(x21, (0, 1), l2.bbbb, (2, 3, 4, 5), (2, 3, 1, 4, 5, 0)) * 2.0
    l3new_bbbbbb += einsum(x21, (0, 1), l2.bbbb, (2, 3, 4, 5), (2, 3, 1, 0, 4, 5)) * 2.0
    l3new_bbbbbb += einsum(x21, (0, 1), l2.bbbb, (2, 3, 4, 5), (2, 1, 3, 4, 0, 5)) * 2.0
    l3new_bbbbbb += einsum(x21, (0, 1), l2.bbbb, (2, 3, 4, 5), (1, 2, 3, 4, 5, 0)) * 2.0
    l3new_bbbbbb += einsum(x21, (0, 1), l2.bbbb, (2, 3, 4, 5), (1, 2, 3, 0, 4, 5)) * 2.0
    l3new_bbbbbb += einsum(x21, (0, 1), l2.bbbb, (2, 3, 4, 5), (2, 3, 1, 4, 0, 5)) * -2.0
    l3new_bbbbbb += einsum(x21, (0, 1), l2.bbbb, (2, 3, 4, 5), (2, 1, 3, 4, 5, 0)) * -2.0
    l3new_bbbbbb += einsum(x21, (0, 1), l2.bbbb, (2, 3, 4, 5), (2, 1, 3, 0, 4, 5)) * -2.0
    l3new_bbbbbb += einsum(x21, (0, 1), l2.bbbb, (2, 3, 4, 5), (1, 2, 3, 4, 0, 5)) * -2.0
    x22 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x22 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 4, 1), (0, 4, 2, 3))
    x23 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x23 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 5, 3), (1, 5, 2, 4))
    x24 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x24 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x24 += einsum(x22, (0, 1, 2, 3), (0, 1, 3, 2))
    x24 += einsum(x23, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x25 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x25 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x26 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x26 += einsum(t1.bb, (0, 1), x3, (2, 3, 4, 1), (0, 2, 3, 4))
    x27 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x27 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x28 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x28 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x28 += einsum(x25, (0, 1, 2, 3), (3, 1, 2, 0))
    x28 += einsum(x26, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x28 += einsum(x27, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    x28 += einsum(x27, (0, 1, 2, 3), (3, 0, 2, 1))
    x29 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x29 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x30 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x30 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (1, 5, 0, 4))
    x31 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x31 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x31 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    l2new_abab += einsum(l1.aa, (0, 1), x31, (2, 3, 1, 4), (0, 3, 4, 2)) * -1.0
    x32 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x32 += einsum(t1.bb, (0, 1), x31, (2, 1, 3, 4), (0, 2, 3, 4))
    x33 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x33 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x33 += einsum(x29, (0, 1, 2, 3), (1, 0, 3, 2))
    x33 += einsum(x30, (0, 1, 2, 3), (0, 1, 3, 2))
    x33 += einsum(x32, (0, 1, 2, 3), (0, 1, 3, 2))
    x34 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x34 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 3, 7, 1), (4, 6, 0, 2, 5, 7)) * -1.0
    l2new_bbbb += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), x34, (3, 5, 6, 7, 4, 1), (0, 2, 6, 7)) * 1.9999999999999203
    x35 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x35 += einsum(t2.abab, (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 6, 3), (1, 4, 5, 6, 0, 2))
    x36 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x36 += einsum(t2.abab, (0, 1, 2, 3), x3, (4, 5, 6, 3), (4, 1, 6, 5, 0, 2))
    x37 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x37 += einsum(x34, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -0.9999999999999601
    del x34
    x37 += einsum(x35, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    x37 += einsum(x35, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x35
    x37 += einsum(x36, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x37 += einsum(x36, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x36
    x38 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x38 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 7, 1, 3), (4, 6, 2, 7, 5, 0))
    l2new_abab += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), x38, (5, 3, 6, 2, 4, 7), (1, 0, 7, 6)) * 1.9999999999999194
    x39 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x39 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x39 += einsum(x5, (0, 1, 2, 3), (1, 0, 2, 3))
    x40 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x40 += einsum(t2.abab, (0, 1, 2, 3), x39, (4, 5, 6, 2), (1, 4, 5, 3, 0, 6))
    x41 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x41 += einsum(x38, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 0.9999999999999597
    x41 += einsum(t2.bbbb, (0, 1, 2, 3), x31, (4, 3, 5, 6), (0, 1, 4, 2, 6, 5))
    x41 += einsum(x40, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -1.0
    x42 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x42 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 3, 7, 1), (0, 4, 6, 2, 5, 7)) * -0.9999999999999601
    x42 += einsum(v.aabb.vvov, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 7, 1, 3), (2, 4, 6, 7, 5, 0)) * -0.9999999999999597
    x42 += einsum(x4, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (2, 4, 0, 6, 5, 7)) * -2.0
    x42 += einsum(x6, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 6, 7, 3), (1, 5, 0, 7, 4, 6)) * -2.0
    x42 += einsum(x8, (0, 1, 2, 3), t3.babbab, (4, 2, 5, 6, 7, 1), (0, 4, 5, 6, 3, 7))
    x42 += einsum(t2.abab, (0, 1, 2, 3), x14, (4, 5, 6, 3), (5, 1, 4, 6, 0, 2))
    x42 += einsum(t2.bbbb, (0, 1, 2, 3), x18, (4, 3, 5, 6), (4, 0, 1, 2, 5, 6)) * -1.0
    x42 += einsum(x21, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (0, 2, 4, 5, 3, 6)) * -0.9999999999999597
    x42 += einsum(t2.abab, (0, 1, 2, 3), x24, (4, 5, 2, 6), (5, 1, 4, 3, 0, 6)) * -1.0
    x42 += einsum(t2.abab, (0, 1, 2, 3), x28, (1, 4, 5, 6), (5, 6, 4, 3, 0, 2)) * -1.0
    x42 += einsum(t2.abab, (0, 1, 2, 3), x33, (4, 5, 0, 6), (5, 1, 4, 3, 6, 2))
    x42 += einsum(t1.bb, (0, 1), x37, (2, 3, 4, 0, 5, 6), (4, 3, 2, 1, 5, 6))
    del x37
    x42 += einsum(t1.aa, (0, 1), x41, (2, 3, 4, 5, 0, 6), (4, 3, 2, 5, 6, 1)) * -1.0
    del x41
    l1new_bb += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), x42, (6, 3, 5, 2, 4, 1), (0, 6)) * -2.0
    del x42
    x43 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x43 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x43 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x43 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x43 += einsum(x3, (0, 1, 2, 3), (0, 2, 1, 3))
    x44 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x44 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 7, 3, 1), (5, 2, 4, 6, 0, 7))
    l2new_abab += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), x44, (4, 6, 5, 3, 7, 2), (0, 1, 7, 6)) * 1.9999999999999194
    x45 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x45 += einsum(t2.aaaa, (0, 1, 2, 3), x39, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2)) * -1.0
    x46 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x46 += einsum(x44, (0, 1, 2, 3, 4, 5), (0, 1, 3, 4, 2, 5)) * 0.9999999999999597
    x46 += einsum(t2.abab, (0, 1, 2, 3), x31, (4, 3, 5, 6), (1, 4, 0, 6, 5, 2)) * -1.0
    x46 += einsum(x45, (0, 1, 2, 3, 4, 5), (1, 0, 3, 4, 2, 5)) * -1.0
    x47 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x47 += einsum(v.aabb.vvov, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 7, 3, 1), (2, 5, 4, 6, 7, 0)) * -1.9999999999999194
    x47 += einsum(x43, (0, 1, 2, 3), t3.abaaba, (4, 2, 5, 6, 3, 7), (1, 0, 4, 5, 6, 7)) * -1.0
    x47 += einsum(x6, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 2, 6, 7, 3), (1, 0, 4, 5, 6, 7)) * -3.0
    x47 += einsum(x8, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 6, 1, 7), (0, 5, 4, 3, 6, 7)) * 2.0
    x47 += einsum(t2.abab, (0, 1, 2, 3), x18, (4, 3, 5, 6), (4, 1, 0, 5, 2, 6)) * -2.0
    del x18
    x47 += einsum(x21, (0, 1), t3.abaaba, (2, 3, 4, 5, 1, 6), (0, 3, 2, 4, 5, 6)) * -0.9999999999999601
    x47 += einsum(t2.aaaa, (0, 1, 2, 3), x24, (4, 5, 3, 6), (5, 4, 0, 1, 2, 6)) * -2.0
    del x24
    x47 += einsum(t2.aaaa, (0, 1, 2, 3), x33, (4, 5, 1, 6), (5, 4, 0, 6, 2, 3)) * 2.0
    del x33
    x47 += einsum(t1.aa, (0, 1), x46, (2, 3, 4, 0, 5, 6), (3, 2, 4, 5, 6, 1)) * -2.0
    del x46
    l1new_bb += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), x47, (6, 4, 3, 5, 0, 2), (1, 6))
    del x47
    x48 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x48 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x48 += einsum(x25, (0, 1, 2, 3), (3, 1, 2, 0))
    x48 += einsum(x26, (0, 1, 2, 3), (3, 1, 2, 0))
    x48 += einsum(x27, (0, 1, 2, 3), (2, 1, 3, 0))
    x48 += einsum(x27, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x49 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x49 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 6, 7, 3, 1), (4, 5, 6, 0, 2, 7)) * -1.0
    l2new_bbbb += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), x49, (5, 4, 3, 6, 7, 2), (0, 1, 6, 7)) * -5.9999999999997575
    x50 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x50 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    x51 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x51 += einsum(t2.bbbb, (0, 1, 2, 3), x3, (4, 5, 6, 3), (4, 0, 1, 6, 5, 2))
    x52 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x52 += einsum(x49, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -0.9999999999999596
    del x49
    x52 += einsum(x50, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    x52 += einsum(x50, (0, 1, 2, 3, 4, 5), (0, 1, 3, 4, 2, 5))
    del x50
    x52 += einsum(x51, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    x52 += einsum(x51, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    del x51
    x53 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x53 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 6, 7, 1, 3), (0, 4, 5, 6, 7, 2)) * -0.9999999999999596
    x53 += einsum(x43, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 2, 6, 7, 3), (1, 4, 0, 5, 6, 7)) * -1.5
    del x43
    x53 += einsum(x6, (0, 1, 2, 3), t3.babbab, (4, 2, 5, 6, 3, 7), (1, 4, 0, 5, 6, 7)) * -0.5
    x53 += einsum(t2.bbbb, (0, 1, 2, 3), x14, (4, 5, 6, 3), (5, 0, 4, 1, 2, 6))
    del x14
    x53 += einsum(x21, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6)) * 0.4999999999999798
    x53 += einsum(t2.bbbb, (0, 1, 2, 3), x48, (1, 4, 5, 6), (5, 0, 6, 4, 2, 3))
    x53 += einsum(t1.bb, (0, 1), x52, (2, 3, 4, 5, 0, 6), (5, 3, 4, 2, 6, 1))
    del x52
    l1new_bb += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), x53, (6, 3, 5, 4, 2, 1), (0, 6)) * -6.0
    del x53
    x54 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x54 += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 0, 4, 5))
    x55 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x55 += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 2, 5, 0), (3, 1, 4, 5))
    x56 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x56 += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.babbab, (4, 6, 5, 1, 7, 2), (3, 0, 6, 7))
    x57 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x57 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 5, 4, 7, 2, 1), (3, 0, 6, 7))
    x58 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x58 += einsum(t1.aa, (0, 1), l3.babbab, (2, 1, 3, 4, 5, 6), (4, 6, 2, 3, 5, 0))
    x59 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x59 += einsum(t2.abab, (0, 1, 2, 3), x58, (4, 1, 5, 3, 0, 6), (4, 5, 6, 2))
    x60 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x60 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (6, 3, 5, 7, 0, 2), (4, 1, 6, 7))
    x61 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x61 += einsum(t1.aa, (0, 1), l3.abaaba, (2, 3, 1, 4, 5, 6), (5, 3, 4, 6, 0, 2))
    x62 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x62 += einsum(t2.aaaa, (0, 1, 2, 3), x61, (4, 5, 0, 1, 6, 3), (4, 5, 6, 2)) * -1.0
    x63 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x63 += einsum(t1.aa, (0, 1), l2.abab, (1, 2, 3, 4), (4, 2, 3, 0))
    l2new_abab += einsum(x5, (0, 1, 2, 3), x63, (0, 4, 5, 2), (3, 4, 5, 1))
    x64 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x64 += einsum(t2.abab, (0, 1, 2, 3), l3.babbab, (4, 2, 3, 5, 6, 1), (5, 4, 6, 0))
    x65 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x65 += einsum(t2.aaaa, (0, 1, 2, 3), l3.abaaba, (2, 4, 3, 5, 6, 1), (6, 4, 5, 0))
    x66 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x66 += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x66 += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3))
    x66 += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3))
    l2new_abab += einsum(v.aabb.ovvv, (0, 1, 2, 3), x66, (4, 3, 5, 0), (1, 2, 5, 4)) * -2.0
    l2new_abab += einsum(v.aabb.ovoo, (0, 1, 2, 3), x66, (3, 4, 5, 0), (1, 4, 5, 2)) * 2.0
    x67 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x67 += einsum(l2.abab, (0, 1, 2, 3), (3, 1, 2, 0))
    x67 += einsum(x54, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x67 += einsum(x55, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x67 += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x67 += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    x67 += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x67 += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x67 += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x67 += einsum(t1.aa, (0, 1), x66, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    l1new_bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), x67, (4, 3, 0, 1), (2, 4))
    del x67
    x68 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x68 += einsum(t2.abab, (0, 1, 2, 3), l3.babbab, (4, 5, 3, 6, 0, 1), (6, 4, 5, 2))
    x69 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x69 += einsum(t2.aaaa, (0, 1, 2, 3), l3.abaaba, (4, 5, 3, 0, 6, 1), (6, 5, 4, 2))
    x70 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x70 += einsum(t1.aa, (0, 1), l2.abab, (2, 3, 0, 4), (4, 3, 2, 1)) * 0.5
    x70 += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3))
    x70 += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3))
    l1new_bb += einsum(v.aabb.vvvv, (0, 1, 2, 3), x70, (4, 3, 0, 1), (2, 4)) * 2.0
    del x70
    x71 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x71 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x71 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x72 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x72 += einsum(t1.bb, (0, 1), l3.bbbbbb, (2, 3, 1, 4, 5, 6), (4, 5, 6, 0, 2, 3))
    l3new_babbab += einsum(v.aabb.ovoo, (0, 1, 2, 3), x72, (4, 2, 5, 3, 6, 7), (6, 1, 7, 5, 0, 4)) * -6.0
    l3new_babbab += einsum(x5, (0, 1, 2, 3), x72, (0, 4, 5, 1, 6, 7), (7, 3, 6, 4, 2, 5)) * 6.0
    x73 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x73 += einsum(t2.bbbb, (0, 1, 2, 3), x72, (0, 4, 1, 5, 6, 3), (4, 5, 6, 2))
    x74 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x74 += einsum(t1.bb, (0, 1), l3.babbab, (2, 3, 1, 4, 5, 6), (4, 6, 0, 2, 5, 3))
    l2new_abab += einsum(x22, (0, 1, 2, 3), x74, (0, 4, 1, 5, 6, 3), (2, 5, 6, 4)) * 2.0
    x75 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x75 += einsum(t2.abab, (0, 1, 2, 3), x74, (4, 1, 5, 6, 0, 2), (4, 5, 6, 3))
    x76 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x76 += einsum(t2.bbbb, (0, 1, 2, 3), l3.bbbbbb, (4, 2, 3, 5, 6, 1), (5, 6, 0, 4))
    x77 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x77 += einsum(t2.abab, (0, 1, 2, 3), l3.babbab, (4, 2, 3, 5, 0, 6), (5, 6, 1, 4))
    x78 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x78 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3))
    x78 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3)) * -3.0
    x78 += einsum(x77, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    l1new_bb += einsum(v.bbbb.oovv, (0, 1, 2, 3), x78, (0, 4, 1, 3), (2, 4)) * -2.0
    x79 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x79 += einsum(x73, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x79 += einsum(x75, (0, 1, 2, 3), (0, 1, 2, 3))
    x79 += einsum(t1.bb, (0, 1), x78, (0, 2, 3, 4), (2, 3, 4, 1))
    l1new_bb += einsum(x71, (0, 1, 2, 3), x79, (4, 0, 2, 1), (3, 4)) * 2.0
    del x79
    x80 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x80 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x80 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x81 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x81 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (2, 4, 0, 5))
    x82 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x82 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), (3, 4, 1, 5))
    x83 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x83 += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (6, 4, 5, 7, 1, 2), (3, 6, 0, 7))
    x84 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x84 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 4, 5, 7, 1, 2), (3, 6, 0, 7))
    x85 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x85 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 6, 5, 0, 7, 2), (4, 6, 1, 7))
    x86 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x86 += einsum(x81, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    x86 += einsum(x82, (0, 1, 2, 3), (0, 1, 2, 3))
    x86 += einsum(x83, (0, 1, 2, 3), (0, 1, 2, 3)) * 9.0
    x86 += einsum(x84, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    x86 += einsum(x85, (0, 1, 2, 3), (0, 1, 2, 3))
    l1new_bb += einsum(x80, (0, 1, 2, 3), x86, (4, 0, 2, 1), (3, 4))
    del x86
    x87 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x87 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), (3, 4, 0, 5))
    x88 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x88 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 4, 5, 0, 7, 2), (3, 6, 1, 7))
    x89 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x89 += einsum(t2.abab, (0, 1, 2, 3), x74, (4, 1, 5, 3, 0, 6), (4, 5, 6, 2)) * -1.0
    x90 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x90 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 6, 5, 7, 1, 2), (4, 6, 0, 7))
    x91 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x91 += einsum(t1.bb, (0, 1), l3.abaaba, (2, 1, 3, 4, 5, 6), (5, 0, 4, 6, 2, 3))
    l3new_abaaba += einsum(x21, (0, 1), x91, (2, 0, 3, 4, 5, 6), (5, 1, 6, 4, 2, 3)) * 2.0
    x92 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x92 += einsum(t2.aaaa, (0, 1, 2, 3), x91, (4, 5, 0, 1, 3, 6), (4, 5, 6, 2)) * -1.0
    x93 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x93 += einsum(t1.bb, (0, 1), l2.abab, (2, 1, 3, 4), (4, 0, 3, 2))
    l2new_abab += einsum(x7, (0, 1, 2, 3), x93, (4, 0, 2, 5), (5, 1, 3, 4))
    x94 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x94 += einsum(t2.bbbb, (0, 1, 2, 3), l3.babbab, (2, 4, 3, 5, 6, 1), (5, 0, 6, 4))
    x95 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x95 += einsum(t2.abab, (0, 1, 2, 3), l3.abaaba, (4, 3, 2, 5, 6, 0), (6, 1, 5, 4))
    x96 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x96 += einsum(x93, (0, 1, 2, 3), (0, 1, 2, 3))
    x96 += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x96 += einsum(x95, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x97 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x97 += einsum(x87, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x97 += einsum(x88, (0, 1, 2, 3), (0, 1, 2, 3))
    x97 += einsum(x89, (0, 1, 2, 3), (0, 1, 3, 2))
    x97 += einsum(x90, (0, 1, 2, 3), (0, 1, 2, 3))
    x97 += einsum(x92, (0, 1, 2, 3), (0, 1, 3, 2))
    x97 += einsum(t1.aa, (0, 1), x96, (2, 3, 0, 4), (2, 3, 1, 4)) * 0.5
    l1new_bb += einsum(v.aabb.vvov, (0, 1, 2, 3), x97, (4, 2, 1, 0), (3, 4)) * -2.0
    del x97
    x98 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x98 += einsum(t1.bb, (0, 1), x58, (2, 3, 4, 1, 5, 6), (2, 3, 0, 4, 5, 6))
    x99 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x99 += einsum(t1.bb, (0, 1), x61, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6))
    x100 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x100 += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3))
    del x68
    x100 += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3))
    del x69
    l2new_abab += einsum(v.aabb.ovvv, (0, 1, 2, 3), x100, (4, 3, 5, 1), (5, 2, 0, 4)) * -2.0
    l2new_abab += einsum(x100, (0, 1, 2, 3), x6, (0, 4, 5, 3), (2, 1, 5, 4)) * 2.0
    x101 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x101 += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3))
    x101 += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x101 += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    l1new_bb += einsum(v.aabb.oovv, (0, 1, 2, 3), x101, (4, 3, 0, 1), (2, 4)) * -1.0
    x102 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x102 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), (3, 5, 2, 4))
    x103 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x103 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 7, 5, 0, 1, 2), (3, 6, 4, 7))
    x104 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x104 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 7, 5, 0, 1, 2), (4, 7, 3, 6))
    x105 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x105 += einsum(t1.bb, (0, 1), x101, (2, 1, 3, 4), (0, 2, 3, 4))
    x106 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x106 += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3))
    x106 += einsum(x95, (0, 1, 2, 3), (0, 1, 2, 3))
    x107 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x107 += einsum(t1.aa, (0, 1), x106, (2, 3, 4, 1), (2, 3, 0, 4)) * 2.0
    x108 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x108 += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3))
    x108 += einsum(x103, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.9999999999999194
    x108 += einsum(x104, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.9999999999999194
    x108 += einsum(x105, (0, 1, 2, 3), (1, 0, 2, 3))
    x108 += einsum(x107, (0, 1, 2, 3), (0, 1, 3, 2))
    l2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), x108, (4, 2, 5, 0), (1, 3, 5, 4))
    x109 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x109 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    x110 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x110 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    x111 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x111 += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (6, 4, 5, 0, 1, 2), (3, 6))
    x112 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x112 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 4, 5, 0, 1, 2), (3, 6))
    x113 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x113 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 6, 5, 0, 1, 2), (4, 6))
    x114 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x114 += einsum(x109, (0, 1), (0, 1))
    x114 += einsum(x110, (0, 1), (0, 1)) * 0.5
    x114 += einsum(x111, (0, 1), (0, 1)) * 1.4999999999999394
    x114 += einsum(x112, (0, 1), (0, 1)) * 0.9999999999999597
    x114 += einsum(x113, (0, 1), (0, 1)) * 0.49999999999998007
    x115 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x115 += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), (1, 3, 2, 4))
    x115 += einsum(x93, (0, 1, 2, 3), (0, 1, 2, 3))
    x115 += einsum(l2.bbbb, (0, 1, 2, 3), t3.babbab, (4, 5, 3, 0, 6, 1), (2, 4, 5, 6)) * 2.0
    x115 += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x115 += einsum(l2.abab, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 6, 1, 0), (3, 5, 4, 6)) * 2.0
    x115 += einsum(x95, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x115 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x72, (6, 0, 2, 7, 5, 3), (6, 7, 1, 4)) * -3.0
    x115 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x74, (6, 1, 7, 4, 2, 5), (6, 7, 0, 3)) * -4.0
    x115 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x58, (6, 2, 5, 3, 1, 7), (6, 0, 7, 4)) * 2.0
    x115 += einsum(t2.abab, (0, 1, 2, 3), x98, (4, 1, 5, 3, 0, 6), (4, 5, 6, 2)) * 2.0
    x115 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x91, (6, 7, 2, 1, 5, 4), (6, 7, 0, 3)) * 3.0
    x115 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x61, (6, 4, 2, 0, 7, 5), (6, 1, 7, 3)) * -2.0
    x115 += einsum(t2.aaaa, (0, 1, 2, 3), x99, (4, 5, 1, 0, 6, 3), (4, 5, 6, 2)) * -2.0
    x115 += einsum(t2.abab, (0, 1, 2, 3), x100, (4, 3, 2, 5), (4, 1, 0, 5)) * -2.0
    x115 += einsum(t2.abab, (0, 1, 2, 3), x78, (1, 4, 5, 3), (4, 5, 0, 2)) * -2.0
    x115 += einsum(t2.aaaa, (0, 1, 2, 3), x96, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    x115 += einsum(t2.abab, (0, 1, 2, 3), x101, (4, 3, 0, 5), (4, 1, 5, 2)) * -1.0
    x115 += einsum(t1.aa, (0, 1), x108, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    x115 += einsum(t1.aa, (0, 1), x114, (2, 3), (2, 3, 0, 1)) * 2.0
    l1new_bb += einsum(v.aabb.ovov, (0, 1, 2, 3), x115, (4, 2, 0, 1), (3, 4)) * -1.0
    del x115
    x116 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x116 += einsum(t2.abab, (0, 1, 2, 3), l3.babbab, (4, 2, 5, 6, 0, 1), (6, 4, 5, 3))
    x117 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x117 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 3, 4, 0), (4, 2, 3, 1))
    x117 += einsum(x116, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l1new_bb += einsum(v.bbbb.vvvv, (0, 1, 2, 3), x117, (4, 1, 2, 3), (0, 4)) * 2.0
    del x117
    x118 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x118 += einsum(t1.aa, (0, 1), v.aabb.vvoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x119 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x119 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvov, (4, 2, 5, 3), (1, 5, 0, 4))
    x120 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x120 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.babbab, (4, 5, 2, 1, 6, 3), (4, 0, 5, 6))
    x121 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x121 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 3, 1), (5, 2, 4, 6))
    x122 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x122 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x122 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x122 += einsum(x3, (0, 1, 2, 3), (1, 0, 2, 3))
    x122 += einsum(x3, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x123 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x123 += einsum(t2.abab, (0, 1, 2, 3), x122, (4, 5, 1, 3), (4, 5, 0, 2))
    x124 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x124 += einsum(t2.aaaa, (0, 1, 2, 3), x39, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    x125 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x125 += einsum(t2.abab, (0, 1, 2, 3), x8, (4, 3, 0, 5), (1, 4, 5, 2))
    x126 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x126 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x126 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    x127 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x127 += einsum(t1.bb, (0, 1), x126, (2, 1, 3, 4), (0, 2, 3, 4))
    x128 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x128 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x128 += einsum(x29, (0, 1, 2, 3), (1, 0, 3, 2))
    x128 += einsum(x30, (0, 1, 2, 3), (1, 0, 3, 2))
    x128 += einsum(x32, (0, 1, 2, 3), (1, 0, 3, 2))
    del x32
    x129 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x129 += einsum(t1.aa, (0, 1), x128, (2, 3, 0, 4), (2, 3, 4, 1))
    x130 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x130 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x130 += einsum(x118, (0, 1, 2, 3), (1, 0, 2, 3))
    x130 += einsum(x119, (0, 1, 2, 3), (0, 1, 2, 3))
    x130 += einsum(x120, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x130 += einsum(x121, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x130 += einsum(x123, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x130 += einsum(x124, (0, 1, 2, 3), (1, 0, 2, 3))
    x130 += einsum(x125, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x130 += einsum(x21, (0, 1), t2.abab, (2, 3, 4, 1), (3, 0, 2, 4))
    x130 += einsum(x127, (0, 1, 2, 3), (0, 1, 2, 3))
    x130 += einsum(x129, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    l1new_bb += einsum(l2.abab, (0, 1, 2, 3), x130, (3, 4, 2, 0), (1, 4)) * -1.0
    del x130
    x131 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x131 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x72, (6, 2, 1, 7, 4, 5), (6, 7, 0, 3)) * -1.0
    x132 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x132 += einsum(t1.bb, (0, 1), x72, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    x133 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x133 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x74, (6, 2, 7, 5, 1, 4), (6, 7, 0, 3)) * -1.0
    x134 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x134 += einsum(t1.bb, (0, 1), x74, (2, 3, 4, 1, 5, 6), (2, 3, 0, 4, 5, 6))
    x135 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x135 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x91, (6, 7, 0, 2, 3, 5), (6, 7, 1, 4))
    x136 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x136 += einsum(x0, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x136 += einsum(x116, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.3333333333333333
    x137 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x137 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x137 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x137 += einsum(x77, (0, 1, 2, 3), (0, 1, 2, 3))
    l2new_abab += einsum(v.aabb.ovoo, (0, 1, 2, 3), x137, (2, 4, 3, 5), (1, 5, 0, 4)) * 2.0
    x138 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x138 += einsum(t2.bbbb, (0, 1, 2, 3), x137, (4, 1, 5, 3), (0, 4, 5, 2)) * 4.0
    x139 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x139 += einsum(x93, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x139 += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3))
    del x94
    x139 += einsum(x95, (0, 1, 2, 3), (0, 1, 2, 3))
    del x95
    l1new_aa += einsum(v.aabb.vvoo, (0, 1, 2, 3), x139, (2, 3, 4, 1), (0, 4)) * -2.0
    l2new_abab += einsum(v.aabb.vvov, (0, 1, 2, 3), x139, (4, 2, 5, 1), (0, 3, 5, 4)) * -2.0
    l2new_abab += einsum(v.aabb.ooov, (0, 1, 2, 3), x139, (4, 2, 1, 5), (5, 3, 0, 4)) * 2.0
    l2new_abab += einsum(x21, (0, 1), x139, (2, 0, 3, 4), (4, 1, 3, 2)) * -2.0
    x140 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x140 += einsum(t2.abab, (0, 1, 2, 3), x139, (4, 5, 0, 2), (1, 4, 5, 3)) * 2.0
    x141 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x141 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), (2, 3, 4, 5))
    x142 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x142 += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (6, 7, 5, 0, 1, 2), (3, 4, 6, 7))
    x143 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x143 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 4, 7, 0, 1, 2), (3, 5, 6, 7))
    l2new_bbbb += einsum(v.bbbb.ovov, (0, 1, 2, 3), x143, (4, 5, 0, 2), (1, 3, 4, 5)) * 1.9999999999999203
    x144 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x144 += einsum(x141, (0, 1, 2, 3), (1, 0, 3, 2))
    x144 += einsum(x142, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.9999999999998788
    x144 += einsum(x143, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.9999999999999601
    x144 += einsum(t1.bb, (0, 1), x137, (2, 3, 4, 1), (2, 3, 0, 4))
    x145 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x145 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    x145 += einsum(l2.bbbb, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 3, 6, 0, 1), (4, 2, 5, 6)) * 6.0
    x145 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3)) * 6.0
    x145 += einsum(x77, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x145 += einsum(x131, (0, 1, 2, 3), (2, 0, 1, 3)) * 9.0
    x145 += einsum(t2.bbbb, (0, 1, 2, 3), x132, (4, 0, 1, 5, 6, 3), (5, 4, 6, 2)) * 6.0
    x145 += einsum(x133, (0, 1, 2, 3), (2, 0, 1, 3)) * 4.0
    x145 += einsum(t2.abab, (0, 1, 2, 3), x134, (4, 1, 5, 6, 0, 2), (5, 4, 6, 3)) * 2.0
    x145 += einsum(x135, (0, 1, 2, 3), (2, 0, 1, 3))
    x145 += einsum(t2.bbbb, (0, 1, 2, 3), x136, (4, 3, 2, 5), (0, 4, 1, 5)) * -6.0
    x145 += einsum(x138, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x145 += einsum(x140, (0, 1, 2, 3), (0, 1, 2, 3))
    x145 += einsum(t1.bb, (0, 1), x144, (0, 2, 3, 4), (4, 2, 3, 1)) * 2.0
    del x144
    x145 += einsum(t1.bb, (0, 1), x114, (2, 3), (0, 2, 3, 1)) * 2.0
    l1new_bb += einsum(v.bbbb.ovov, (0, 1, 2, 3), x145, (0, 4, 2, 1), (3, 4)) * -1.0
    del x145
    x146 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x146 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x147 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x147 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 2, 6, 1, 3), (4, 5, 0, 6))
    x148 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x148 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 1, 3), (4, 5, 2, 6))
    x149 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x149 += einsum(t2.bbbb, (0, 1, 2, 3), x4, (4, 5, 1, 3), (0, 4, 5, 2)) * 2.0
    del x4
    x150 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x150 += einsum(t2.abab, (0, 1, 2, 3), x6, (4, 5, 0, 2), (1, 4, 5, 3))
    x151 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x151 += einsum(x21, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    x152 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x152 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x153 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x153 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x153 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x153 += einsum(x152, (0, 1, 2, 3), (0, 1, 2, 3))
    x154 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x154 += einsum(t1.bb, (0, 1), x153, (2, 3, 1, 4), (0, 2, 3, 4))
    del x153
    x155 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x155 += einsum(t1.bb, (0, 1), x28, (0, 2, 3, 4), (2, 3, 4, 1))
    del x28
    x156 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x156 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x156 += einsum(x146, (0, 1, 2, 3), (2, 1, 0, 3))
    x156 += einsum(x147, (0, 1, 2, 3), (2, 1, 0, 3)) * -3.0
    x156 += einsum(x148, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x156 += einsum(x149, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x156 += einsum(x150, (0, 1, 2, 3), (2, 0, 1, 3))
    x156 += einsum(x151, (0, 1, 2, 3), (2, 1, 0, 3))
    x156 += einsum(x154, (0, 1, 2, 3), (2, 1, 0, 3))
    x156 += einsum(x155, (0, 1, 2, 3), (1, 2, 0, 3))
    l1new_bb += einsum(l2.bbbb, (0, 1, 2, 3), x156, (4, 2, 3, 1), (0, 4)) * 2.0
    del x156
    x157 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x157 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3))
    x157 += einsum(x77, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.3333333333333333
    x158 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x158 += einsum(t1.bb, (0, 1), x157, (2, 3, 4, 1), (0, 2, 3, 4))
    x159 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x159 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 0), (1, 2, 3, 4)) * 2.0
    x159 += einsum(l2.abab, (0, 1, 2, 3), t3.babbab, (4, 2, 5, 6, 0, 1), (3, 4, 5, 6)) * 2.0
    x159 += einsum(x131, (0, 1, 2, 3), (0, 2, 1, 3)) * 9.0
    del x131
    x159 += einsum(x133, (0, 1, 2, 3), (0, 2, 1, 3)) * 4.0
    del x133
    x159 += einsum(x135, (0, 1, 2, 3), (0, 2, 1, 3))
    del x135
    x159 += einsum(x138, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x138
    x159 += einsum(x140, (0, 1, 2, 3), (1, 0, 2, 3))
    del x140
    x159 += einsum(t1.bb, (0, 1), x158, (2, 3, 0, 4), (3, 4, 2, 1)) * -6.0
    x159 += einsum(t1.bb, (0, 1), x114, (2, 3), (2, 0, 3, 1)) * 2.0
    del x114
    l1new_bb += einsum(v.bbbb.ovov, (0, 1, 2, 3), x159, (4, 0, 2, 3), (1, 4))
    del x159
    x160 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x160 += einsum(l1.bb, (0, 1), t1.bb, (1, 2), (0, 2))
    x161 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x161 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4))
    x162 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x162 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    x163 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x163 += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (3, 4, 5, 6, 1, 2), (0, 6))
    x164 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x164 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 4, 5, 6, 1, 2), (0, 6))
    x165 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x165 += einsum(x160, (0, 1), (0, 1))
    x165 += einsum(x161, (0, 1), (0, 1)) * 2.0
    x165 += einsum(x162, (0, 1), (0, 1))
    x165 += einsum(x163, (0, 1), (0, 1)) * 2.9999999999998788
    x165 += einsum(x164, (0, 1), (0, 1)) * 1.9999999999999194
    l1new_bb += einsum(x165, (0, 1), x12, (2, 3, 1, 0), (3, 2)) * -1.0
    del x165
    x166 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x166 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 0), (2, 3))
    x167 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x167 += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), (2, 3))
    x168 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x168 += einsum(l2.bbbb, (0, 1, 2, 3), t3.bbbbbb, (4, 2, 3, 5, 0, 1), (4, 5))
    x169 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x169 += einsum(l2.abab, (0, 1, 2, 3), t3.babbab, (4, 2, 3, 5, 0, 1), (4, 5))
    x170 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x170 += einsum(l2.aaaa, (0, 1, 2, 3), t3.abaaba, (2, 4, 3, 0, 5, 1), (4, 5))
    x171 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x171 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x72, (0, 1, 2, 6, 4, 5), (6, 3))
    x172 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x172 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x74, (0, 2, 6, 5, 1, 4), (6, 3)) * -1.0
    x173 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x173 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x91, (1, 6, 0, 2, 3, 5), (6, 4))
    x174 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x174 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x174 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.9999999999999996
    x174 += einsum(x77, (0, 1, 2, 3), (0, 1, 2, 3))
    x175 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x175 += einsum(l1.bb, (0, 1), t1.bb, (2, 0), (1, 2))
    x176 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x176 += einsum(x175, (0, 1), (0, 1))
    x176 += einsum(x109, (0, 1), (0, 1)) * 2.0
    x176 += einsum(x110, (0, 1), (0, 1))
    x176 += einsum(x111, (0, 1), (0, 1)) * 2.9999999999998783
    x176 += einsum(x112, (0, 1), (0, 1)) * 1.9999999999999192
    x176 += einsum(x113, (0, 1), (0, 1)) * 0.99999999999996
    x177 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x177 += einsum(t1.bb, (0, 1), (0, 1)) * -0.5000000000000202
    x177 += einsum(x166, (0, 1), (0, 1)) * -1.0000000000000404
    x177 += einsum(x167, (0, 1), (0, 1)) * -0.5000000000000202
    x177 += einsum(x168, (0, 1), (0, 1)) * -1.5000000000000604
    x177 += einsum(x169, (0, 1), (0, 1)) * -1.0000000000000404
    x177 += einsum(x170, (0, 1), (0, 1)) * -0.5000000000000202
    x177 += einsum(x171, (0, 1), (0, 1)) * 1.4999999999999998
    x177 += einsum(x172, (0, 1), (0, 1))
    x177 += einsum(x173, (0, 1), (0, 1)) * 0.5000000000000002
    x177 += einsum(t2.bbbb, (0, 1, 2, 3), x174, (0, 1, 4, 3), (4, 2)) * -1.0000000000000404
    del x174
    x177 += einsum(t2.abab, (0, 1, 2, 3), x139, (1, 4, 0, 2), (4, 3)) * 1.0000000000000404
    x177 += einsum(t1.bb, (0, 1), x176, (0, 2), (2, 1)) * 0.5000000000000202
    del x176
    x178 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x178 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x178 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    l1new_bb += einsum(x177, (0, 1), x178, (0, 2, 1, 3), (3, 2)) * -1.9999999999999194
    del x177, x178
    x179 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x179 += einsum(l1.aa, (0, 1), t1.aa, (1, 2), (0, 2))
    x180 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x180 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    x181 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x181 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4))
    x182 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x182 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 4, 5, 0, 6, 2), (1, 6))
    x183 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x183 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 4, 5, 6, 1, 2), (0, 6))
    x184 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x184 += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (3, 4, 5, 6, 1, 2), (0, 6))
    x185 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x185 += einsum(x179, (0, 1), (0, 1)) * 1.00000000000004
    x185 += einsum(x180, (0, 1), (0, 1)) * 1.00000000000004
    x185 += einsum(x181, (0, 1), (0, 1)) * 2.00000000000008
    x185 += einsum(x182, (0, 1), (0, 1))
    x185 += einsum(x183, (0, 1), (0, 1)) * 1.9999999999999991
    x185 += einsum(x184, (0, 1), (1, 0)) * 2.9999999999999982
    l1new_bb += einsum(x185, (0, 1), v.aabb.vvov, (1, 0, 2, 3), (3, 2)) * 0.9999999999999601
    del x185
    x186 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x186 += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), (2, 3))
    x187 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x187 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 0), (2, 3))
    x188 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x188 += einsum(l2.bbbb, (0, 1, 2, 3), t3.babbab, (2, 4, 3, 0, 5, 1), (4, 5))
    x189 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x189 += einsum(l2.abab, (0, 1, 2, 3), t3.abaaba, (4, 3, 2, 5, 1, 0), (4, 5))
    x190 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x190 += einsum(l2.aaaa, (0, 1, 2, 3), t3.aaaaaa, (4, 2, 3, 5, 0, 1), (4, 5))
    x191 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x191 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x58, (0, 2, 3, 5, 1, 6), (6, 4))
    x192 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x192 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x61, (1, 4, 0, 2, 6, 5), (6, 3)) * -1.0
    x193 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x193 += einsum(t1.aa, (0, 1), l3.aaaaaa, (2, 3, 1, 4, 5, 6), (4, 5, 6, 0, 2, 3))
    l3new_abaaba += einsum(v.aabb.ooov, (0, 1, 2, 3), x193, (4, 0, 5, 1, 6, 7), (6, 3, 7, 5, 2, 4)) * -6.0
    l3new_abaaba += einsum(x7, (0, 1, 2, 3), x193, (2, 4, 5, 3, 6, 7), (6, 1, 7, 4, 0, 5)) * -6.0
    x194 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x194 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x193, (0, 1, 2, 6, 4, 5), (6, 3))
    x195 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x195 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    l2new_abab += einsum(x195, (0, 1, 2, 3), x7, (4, 5, 1, 2), (3, 5, 0, 4)) * -2.0
    x196 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x196 += einsum(t2.abab, (0, 1, 2, 3), l3.abaaba, (4, 3, 2, 5, 1, 6), (5, 6, 0, 4))
    x197 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x197 += einsum(t2.aaaa, (0, 1, 2, 3), l3.aaaaaa, (4, 2, 3, 5, 6, 1), (5, 6, 0, 4))
    x198 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x198 += einsum(x195, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.33333333333333337
    x198 += einsum(x196, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.33333333333333337
    x198 += einsum(x197, (0, 1, 2, 3), (0, 1, 2, 3))
    x199 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x199 += einsum(l1.aa, (0, 1), t1.aa, (2, 0), (1, 2))
    x200 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x200 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    x201 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x201 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    x202 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x202 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 6, 5, 0, 1, 2), (4, 6))
    x203 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x203 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 4, 5, 0, 1, 2), (3, 6))
    x204 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x204 += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (6, 4, 5, 0, 1, 2), (3, 6))
    x205 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x205 += einsum(x199, (0, 1), (0, 1))
    x205 += einsum(x200, (0, 1), (0, 1))
    x205 += einsum(x201, (0, 1), (0, 1)) * 2.0
    x205 += einsum(x202, (0, 1), (0, 1)) * 0.99999999999996
    x205 += einsum(x203, (0, 1), (0, 1)) * 1.9999999999999192
    x205 += einsum(x204, (0, 1), (0, 1)) * 2.9999999999998783
    x206 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x206 += einsum(l1.aa, (0, 1), (1, 0)) * -1.00000000000004
    x206 += einsum(t1.aa, (0, 1), (0, 1)) * -1.00000000000004
    x206 += einsum(x186, (0, 1), (0, 1)) * -1.00000000000004
    x206 += einsum(x187, (0, 1), (0, 1)) * -2.00000000000008
    x206 += einsum(x188, (0, 1), (0, 1)) * -1.00000000000004
    x206 += einsum(x189, (0, 1), (0, 1)) * -2.00000000000008
    x206 += einsum(x190, (0, 1), (0, 1)) * -3.0000000000001195
    x206 += einsum(x191, (0, 1), (0, 1))
    x206 += einsum(x192, (0, 1), (0, 1)) * 1.9999999999999991
    x206 += einsum(x194, (0, 1), (0, 1)) * 2.9999999999999982
    x206 += einsum(t2.abab, (0, 1, 2, 3), x66, (1, 3, 0, 4), (4, 2)) * 2.00000000000008
    x206 += einsum(t2.aaaa, (0, 1, 2, 3), x198, (0, 1, 4, 3), (4, 2)) * -6.000000000000239
    x206 += einsum(t1.aa, (0, 1), x205, (0, 2), (2, 1)) * 1.00000000000004
    l1new_bb += einsum(x206, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (3, 2)) * -0.9999999999999601
    del x206
    x207 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x207 += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3))
    x207 += einsum(x103, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.9999999999999194
    x207 += einsum(x104, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.9999999999999194
    x207 += einsum(x105, (0, 1, 2, 3), (1, 0, 3, 2))
    del x105
    x207 += einsum(x107, (0, 1, 2, 3), (0, 1, 2, 3))
    del x107
    l1new_bb += einsum(v.aabb.ooov, (0, 1, 2, 3), x207, (4, 2, 1, 0), (3, 4))
    del x207
    x208 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x208 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x208 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l2new_abab += einsum(x208, (0, 1, 2, 3), x96, (0, 1, 4, 5), (5, 3, 4, 2)) * -1.0
    l3new_abaaba += einsum(x208, (0, 1, 2, 3), x91, (0, 1, 4, 5, 6, 7), (6, 3, 7, 5, 2, 4)) * 2.0
    x209 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x209 += einsum(t1.bb, (0, 1), x157, (2, 3, 4, 1), (2, 3, 0, 4)) * 3.0
    l1new_bb += einsum(x208, (0, 1, 2, 3), x209, (4, 0, 2, 1), (3, 4)) * 2.0
    del x209
    x210 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x210 += einsum(x141, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.00000000000004
    x210 += einsum(x142, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.9999999999999982
    x210 += einsum(x143, (0, 1, 2, 3), (0, 1, 3, 2))
    del x143
    l1new_bb += einsum(v.bbbb.ooov, (0, 1, 2, 3), x210, (1, 4, 2, 0), (3, 4)) * 1.9999999999999203
    del x210
    x211 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x211 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 4, 5, 0, 6, 2), (1, 6))
    l1new_bb += einsum(x211, (0, 1), x71, (2, 3, 0, 1), (3, 2)) * -0.9999999999999601
    x212 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x212 += einsum(x175, (0, 1), (0, 1)) * 0.5
    x212 += einsum(x109, (0, 1), (0, 1))
    x212 += einsum(x110, (0, 1), (0, 1)) * 0.5
    x212 += einsum(x111, (0, 1), (0, 1)) * 1.4999999999999394
    x212 += einsum(x112, (0, 1), (0, 1)) * 0.9999999999999597
    x213 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x213 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x213 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    l1new_bb += einsum(x212, (0, 1), x213, (0, 2, 1, 3), (3, 2)) * -2.0
    del x212
    l1new_bb += einsum(x113, (0, 1), x213, (0, 1, 2, 3), (3, 2)) * 0.9999999999999601
    x214 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x214 += einsum(x199, (0, 1), (0, 1))
    x214 += einsum(x200, (0, 1), (0, 1))
    x214 += einsum(x201, (0, 1), (0, 1)) * 2.0
    x214 += einsum(x202, (0, 1), (0, 1)) * 0.9999999999999601
    x214 += einsum(x203, (0, 1), (0, 1)) * 1.9999999999999194
    x214 += einsum(x204, (0, 1), (1, 0)) * 2.9999999999998788
    l1new_bb += einsum(x214, (0, 1), v.aabb.ooov, (1, 0, 2, 3), (3, 2)) * -1.0
    del x214
    x215 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x215 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x215 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l1new_bb += einsum(l1.bb, (0, 1), x215, (1, 2, 0, 3), (3, 2)) * -1.0
    del x215
    x216 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x216 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 1, 2, 3), (2, 3))
    x217 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x217 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x217 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x218 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x218 += einsum(f.bb.vv, (0, 1), (0, 1))
    x218 += einsum(x216, (0, 1), (1, 0))
    x218 += einsum(t1.bb, (0, 1), x217, (0, 2, 1, 3), (3, 2)) * -1.0
    l1new_bb += einsum(l1.bb, (0, 1), x218, (0, 2), (2, 1))
    del x218
    x219 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x219 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3))
    x220 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x220 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 1, 3), (0, 4))
    x221 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x221 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x222 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x222 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x222 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x223 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x223 += einsum(t1.bb, (0, 1), x222, (2, 3, 0, 1), (2, 3))
    del x222
    x224 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x224 += einsum(t1.bb, (0, 1), x21, (2, 1), (0, 2))
    x225 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x225 += einsum(f.bb.oo, (0, 1), (0, 1))
    x225 += einsum(x219, (0, 1), (1, 0))
    x225 += einsum(x220, (0, 1), (0, 1)) * 2.0
    x225 += einsum(x221, (0, 1), (0, 1))
    x225 += einsum(x223, (0, 1), (1, 0)) * -1.0
    x225 += einsum(x224, (0, 1), (0, 1))
    del x224
    l1new_bb += einsum(l1.bb, (0, 1), x225, (1, 2), (0, 2)) * -1.0
    l2new_abab += einsum(x225, (0, 1), l2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    l3new_abaaba += einsum(x225, (0, 1), l3.abaaba, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -2.0
    del x225
    x226 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x226 += einsum(x175, (0, 1), (0, 1)) * 0.5000000000000202
    x226 += einsum(x109, (0, 1), (0, 1)) * 1.0000000000000404
    x226 += einsum(x110, (0, 1), (0, 1)) * 0.5000000000000202
    x226 += einsum(x111, (0, 1), (0, 1)) * 1.4999999999999998
    x226 += einsum(x112, (0, 1), (0, 1))
    x226 += einsum(x113, (0, 1), (0, 1)) * 0.5000000000000002
    l1new_bb += einsum(f.bb.ov, (0, 1), x226, (2, 0), (1, 2)) * -1.9999999999999194
    del x226
    x227 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x227 += einsum(x19, (0, 1), (0, 1))
    x227 += einsum(x20, (0, 1), (0, 1)) * -1.0
    l1new_bb += einsum(x175, (0, 1), x227, (1, 2), (2, 0)) * -1.0
    x228 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x228 += einsum(t2.aaaa, (0, 1, 2, 3), l3.aaaaaa, (4, 5, 3, 6, 0, 1), (6, 4, 5, 2))
    l1new_aa += einsum(v.aaaa.vvvv, (0, 1, 2, 3), x228, (4, 1, 2, 3), (0, 4)) * 6.0
    x229 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x229 += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (6, 7, 5, 0, 1, 2), (3, 4, 6, 7))
    l1new_aa += einsum(v.aaaa.ooov, (0, 1, 2, 3), x229, (1, 4, 2, 0), (3, 4)) * -5.9999999999997575
    x230 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x230 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x231 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x231 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x231 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x231 += einsum(x230, (0, 1, 2, 3), (0, 1, 2, 3))
    x231 += einsum(x230, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x232 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x232 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x233 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x233 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x233 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x234 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x234 += einsum(t2.aaaa, (0, 1, 2, 3), x233, (1, 4, 5, 3), (0, 4, 2, 5)) * 2.0
    x235 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x235 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x235 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x236 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x236 += einsum(t1.aa, (0, 1), x235, (2, 3, 1, 4), (0, 2, 3, 4))
    x237 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x237 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x237 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x237 += einsum(x232, (0, 1, 2, 3), (0, 1, 2, 3))
    x237 += einsum(x234, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x237 += einsum(x236, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x238 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x238 += einsum(t1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_abab += einsum(l2.bbbb, (0, 1, 2, 3), x238, (3, 1, 4, 5), (5, 0, 4, 2)) * 2.0
    x239 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x239 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 2, 4, 5))
    x240 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x240 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x240 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x241 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x241 += einsum(t2.abab, (0, 1, 2, 3), x240, (0, 4, 2, 5), (1, 3, 4, 5))
    x242 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x242 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x242 += einsum(x238, (0, 1, 2, 3), (0, 1, 2, 3))
    x242 += einsum(x239, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x242 += einsum(x241, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x243 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x243 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    x244 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x244 += einsum(t1.aa, (0, 1), x240, (0, 2, 1, 3), (2, 3))
    x245 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x245 += einsum(f.aa.ov, (0, 1), (0, 1))
    x245 += einsum(x243, (0, 1), (0, 1))
    x245 += einsum(x244, (0, 1), (0, 1)) * -1.0
    l2new_abab += einsum(x245, (0, 1), x66, (2, 3, 4, 0), (1, 3, 4, 2)) * -2.0
    l2new_abab += einsum(l1.bb, (0, 1), x245, (2, 3), (3, 0, 2, 1))
    l3new_aaaaaa += einsum(x245, (0, 1), l2.aaaa, (2, 3, 4, 5), (2, 3, 1, 4, 5, 0)) * 2.0
    l3new_aaaaaa += einsum(x245, (0, 1), l2.aaaa, (2, 3, 4, 5), (2, 3, 1, 0, 4, 5)) * 2.0
    l3new_aaaaaa += einsum(x245, (0, 1), l2.aaaa, (2, 3, 4, 5), (2, 1, 3, 4, 0, 5)) * 2.0
    l3new_aaaaaa += einsum(x245, (0, 1), l2.aaaa, (2, 3, 4, 5), (1, 2, 3, 4, 5, 0)) * 2.0
    l3new_aaaaaa += einsum(x245, (0, 1), l2.aaaa, (2, 3, 4, 5), (1, 2, 3, 0, 4, 5)) * 2.0
    l3new_aaaaaa += einsum(x245, (0, 1), l2.aaaa, (2, 3, 4, 5), (2, 3, 1, 4, 0, 5)) * -2.0
    l3new_aaaaaa += einsum(x245, (0, 1), l2.aaaa, (2, 3, 4, 5), (2, 1, 3, 4, 5, 0)) * -2.0
    l3new_aaaaaa += einsum(x245, (0, 1), l2.aaaa, (2, 3, 4, 5), (2, 1, 3, 0, 4, 5)) * -2.0
    l3new_aaaaaa += einsum(x245, (0, 1), l2.aaaa, (2, 3, 4, 5), (1, 2, 3, 4, 0, 5)) * -2.0
    l3new_babbab += einsum(x245, (0, 1), x58, (2, 3, 4, 5, 6, 0), (4, 1, 5, 3, 6, 2)) * 2.0
    l3new_babbab += einsum(x245, (0, 1), l2.bbbb, (2, 3, 4, 5), (2, 1, 3, 4, 0, 5)) * 2.0
    l3new_abaaba += einsum(x245, (0, 1), l2.abab, (2, 3, 4, 5), (1, 3, 2, 0, 5, 4))
    l3new_abaaba += einsum(x245, (0, 1), l2.abab, (2, 3, 4, 5), (2, 3, 1, 4, 5, 0))
    l3new_abaaba += einsum(x245, (0, 1), l2.abab, (2, 3, 4, 5), (2, 3, 1, 0, 5, 4)) * -1.0
    l3new_abaaba += einsum(x245, (0, 1), l2.abab, (2, 3, 4, 5), (1, 3, 2, 4, 5, 0)) * -1.0
    x246 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x246 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (3, 4, 0, 2))
    l2new_abab += einsum(x246, (0, 1, 2, 3), x61, (4, 1, 2, 5, 3, 6), (6, 0, 5, 4)) * 2.0
    x247 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x247 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 5), (3, 5, 0, 4))
    x248 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x248 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x248 += einsum(x246, (0, 1, 2, 3), (1, 0, 2, 3))
    x248 += einsum(x247, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x249 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x249 += einsum(t1.bb, (0, 1), x8, (2, 1, 3, 4), (0, 2, 3, 4))
    x250 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x250 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x250 += einsum(x29, (0, 1, 2, 3), (1, 0, 2, 3))
    x250 += einsum(x30, (0, 1, 2, 3), (1, 0, 2, 3))
    x250 += einsum(x249, (0, 1, 2, 3), (1, 0, 3, 2))
    x251 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x251 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x252 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x252 += einsum(t1.aa, (0, 1), x230, (2, 3, 4, 1), (0, 2, 3, 4))
    x253 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x253 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x254 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x254 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x254 += einsum(x251, (0, 1, 2, 3), (3, 1, 2, 0))
    x254 += einsum(x252, (0, 1, 2, 3), (3, 1, 2, 0))
    x254 += einsum(x253, (0, 1, 2, 3), (2, 1, 3, 0))
    x254 += einsum(x253, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x255 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x255 += einsum(x44, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * 0.9999999999999597
    del x44
    x255 += einsum(t2.abab, (0, 1, 2, 3), x8, (4, 3, 5, 6), (1, 4, 0, 6, 5, 2)) * -1.0
    x255 += einsum(x45, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x45
    x256 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x256 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 3, 7, 1), (5, 7, 4, 6, 0, 2)) * -1.0
    l2new_aaaa += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), x256, (4, 1, 5, 3, 6, 7), (0, 2, 6, 7)) * -1.9999999999999203
    x257 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x257 += einsum(t2.abab, (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x258 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x258 += einsum(t2.abab, (0, 1, 2, 3), x230, (4, 5, 6, 2), (1, 3, 4, 0, 6, 5))
    x259 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x259 += einsum(x256, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * 0.9999999999999601
    del x256
    x259 += einsum(x257, (0, 1, 2, 3, 4, 5), (0, 1, 4, 2, 3, 5))
    x259 += einsum(x257, (0, 1, 2, 3, 4, 5), (0, 1, 4, 2, 5, 3)) * -1.0
    del x257
    x259 += einsum(x258, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    x259 += einsum(x258, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x258
    x260 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x260 += einsum(v.aabb.ovvv, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 7, 3, 1), (5, 2, 0, 4, 6, 7)) * -0.9999999999999597
    x260 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 3, 7, 1), (5, 7, 0, 4, 6, 2)) * -0.9999999999999601
    x260 += einsum(x231, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 7, 3), (5, 7, 2, 4, 0, 6)) * -2.0
    x260 += einsum(x31, (0, 1, 2, 3), t3.babbab, (4, 5, 0, 6, 7, 1), (4, 6, 3, 5, 2, 7)) * -2.0
    x260 += einsum(x39, (0, 1, 2, 3), t3.abaaba, (4, 0, 5, 6, 7, 3), (1, 7, 2, 4, 5, 6))
    x260 += einsum(t2.abab, (0, 1, 2, 3), x237, (4, 5, 6, 2), (1, 3, 5, 0, 4, 6))
    x260 += einsum(t2.aaaa, (0, 1, 2, 3), x242, (4, 5, 6, 3), (4, 5, 6, 0, 1, 2)) * -1.0
    x260 += einsum(x245, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (3, 6, 0, 2, 4, 5)) * -0.9999999999999597
    x260 += einsum(t2.abab, (0, 1, 2, 3), x248, (3, 4, 5, 6), (1, 4, 6, 0, 5, 2)) * -1.0
    x260 += einsum(t2.abab, (0, 1, 2, 3), x250, (1, 4, 5, 6), (4, 3, 6, 0, 5, 2))
    x260 += einsum(t2.abab, (0, 1, 2, 3), x254, (0, 4, 5, 6), (1, 3, 5, 4, 6, 2))
    x260 += einsum(t1.bb, (0, 1), x255, (2, 0, 3, 4, 5, 6), (2, 1, 5, 3, 4, 6)) * -1.0
    del x255
    x260 += einsum(t1.aa, (0, 1), x259, (2, 3, 4, 5, 0, 6), (2, 3, 6, 5, 4, 1)) * -1.0
    del x259
    l1new_aa += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), x260, (4, 1, 6, 5, 3, 2), (0, 6)) * 2.0
    del x260
    x261 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x261 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x261 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x261 += einsum(x230, (0, 1, 2, 3), (1, 0, 2, 3))
    x261 += einsum(x230, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x262 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x262 += einsum(x38, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.9999999999999597
    del x38
    x262 += einsum(t2.bbbb, (0, 1, 2, 3), x8, (4, 3, 5, 6), (0, 1, 4, 2, 6, 5))
    x262 += einsum(x40, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    del x40
    x263 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x263 += einsum(v.aabb.ovvv, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 7, 1, 3), (4, 6, 7, 2, 0, 5)) * -0.9999999999999597
    x263 += einsum(x261, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 3, 7), (4, 5, 6, 7, 2, 1)) * -0.5
    x263 += einsum(x31, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 0, 6, 7, 1), (4, 5, 6, 7, 3, 2)) * -1.5
    x263 += einsum(x39, (0, 1, 2, 3), t3.babbab, (4, 5, 0, 6, 3, 7), (4, 1, 6, 7, 2, 5))
    x263 += einsum(t2.abab, (0, 1, 2, 3), x242, (4, 5, 6, 2), (1, 4, 3, 5, 6, 0)) * -1.0
    del x242
    x263 += einsum(x245, (0, 1), t3.babbab, (2, 3, 4, 5, 1, 6), (2, 4, 5, 6, 0, 3)) * -0.49999999999998007
    x263 += einsum(t2.bbbb, (0, 1, 2, 3), x248, (3, 4, 5, 6), (0, 1, 2, 4, 6, 5)) * -1.0
    del x248
    x263 += einsum(t2.bbbb, (0, 1, 2, 3), x250, (1, 4, 5, 6), (0, 4, 2, 3, 6, 5))
    del x250
    x263 += einsum(t1.bb, (0, 1), x262, (2, 3, 0, 4, 5, 6), (2, 3, 4, 1, 6, 5))
    del x262
    l1new_aa += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), x263, (3, 5, 0, 2, 6, 4), (1, 6)) * 2.0
    del x263
    x264 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x264 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x264 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x264 += einsum(x230, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x264 += einsum(x230, (0, 1, 2, 3), (2, 0, 1, 3))
    x265 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x265 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x265 += einsum(x251, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    x265 += einsum(x252, (0, 1, 2, 3), (2, 1, 3, 0))
    x265 += einsum(x253, (0, 1, 2, 3), (2, 0, 3, 1))
    x265 += einsum(x253, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    x266 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x266 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 6, 7, 1, 3), (4, 5, 6, 0, 2, 7))
    l2new_aaaa += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), x266, (3, 4, 5, 6, 7, 2), (0, 1, 6, 7)) * 5.9999999999997575
    x267 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x267 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    x268 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x268 += einsum(t2.aaaa, (0, 1, 2, 3), x230, (4, 5, 6, 3), (4, 0, 1, 6, 5, 2))
    x269 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x269 += einsum(x266, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -0.9999999999999596
    del x266
    x269 += einsum(x267, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    x269 += einsum(x267, (0, 1, 2, 3, 4, 5), (0, 1, 3, 4, 2, 5))
    del x267
    x269 += einsum(x268, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    x269 += einsum(x268, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    del x268
    x270 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x270 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 6, 7, 3, 1), (0, 4, 5, 6, 7, 2)) * -0.9999999999999596
    x270 += einsum(x264, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 7, 3), (2, 4, 1, 5, 6, 7)) * -1.5
    del x264
    x270 += einsum(x31, (0, 1, 2, 3), t3.abaaba, (4, 0, 5, 6, 1, 7), (3, 4, 2, 5, 6, 7)) * 0.5
    x270 += einsum(t2.aaaa, (0, 1, 2, 3), x237, (4, 5, 6, 3), (5, 0, 4, 1, 2, 6)) * -1.0
    del x237
    x270 += einsum(x245, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6)) * -0.4999999999999798
    x270 += einsum(t2.aaaa, (0, 1, 2, 3), x265, (1, 4, 5, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    del x265
    x270 += einsum(t1.aa, (0, 1), x269, (2, 3, 4, 5, 0, 6), (5, 3, 4, 2, 6, 1)) * -1.0
    del x269
    l1new_aa += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), x270, (6, 4, 3, 5, 2, 1), (0, 6)) * 6.0
    del x270
    x271 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x271 += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (4, 5, 2, 0))
    x272 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x272 += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 1, 5), (4, 5, 2, 0))
    x273 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x273 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (6, 3, 5, 7, 0, 2), (6, 7, 4, 1))
    x274 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x274 += einsum(t2.bbbb, (0, 1, 2, 3), x74, (0, 1, 4, 3, 5, 6), (4, 2, 5, 6)) * -1.0
    x275 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x275 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 5, 4, 7, 2, 1), (6, 7, 3, 0))
    x276 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x276 += einsum(t2.abab, (0, 1, 2, 3), x91, (1, 4, 5, 0, 2, 6), (4, 3, 5, 6)) * -1.0
    x277 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x277 += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.abaaba, (4, 6, 5, 1, 7, 2), (6, 7, 3, 0))
    x278 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x278 += einsum(l2.abab, (0, 1, 2, 3), (3, 1, 2, 0))
    x278 += einsum(x271, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x278 += einsum(x272, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x278 += einsum(x273, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x278 += einsum(x274, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x278 += einsum(x275, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    x278 += einsum(x276, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x278 += einsum(x277, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x278 += einsum(t1.bb, (0, 1), x96, (0, 2, 3, 4), (2, 1, 3, 4)) * -1.0
    l1new_aa += einsum(v.aabb.vvov, (0, 1, 2, 3), x278, (2, 3, 4, 1), (0, 4))
    del x278
    x279 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x279 += einsum(t2.bbbb, (0, 1, 2, 3), l3.babbab, (4, 5, 3, 0, 6, 1), (4, 2, 6, 5))
    x280 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x280 += einsum(t2.abab, (0, 1, 2, 3), l3.abaaba, (4, 5, 2, 6, 1, 0), (5, 3, 6, 4))
    x281 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x281 += einsum(t1.bb, (0, 1), l2.abab, (2, 3, 4, 0), (3, 1, 4, 2)) * 0.5
    x281 += einsum(x279, (0, 1, 2, 3), (0, 1, 2, 3))
    x281 += einsum(x280, (0, 1, 2, 3), (0, 1, 2, 3))
    l1new_aa += einsum(v.aabb.vvvv, (0, 1, 2, 3), x281, (2, 3, 4, 1), (0, 4)) * 2.0
    del x281
    x282 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x282 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 4, 0, 5))
    x283 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x283 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 5, 1), (2, 4, 0, 5))
    x284 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x284 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 6, 5, 0, 7, 2), (4, 6, 1, 7))
    x285 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x285 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 4, 5, 7, 1, 2), (3, 6, 0, 7))
    x286 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x286 += einsum(t2.abab, (0, 1, 2, 3), x61, (1, 3, 0, 4, 5, 6), (4, 5, 6, 2)) * -1.0
    x287 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x287 += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (6, 4, 5, 7, 1, 2), (3, 6, 0, 7))
    x288 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x288 += einsum(t2.aaaa, (0, 1, 2, 3), x193, (4, 0, 1, 5, 6, 3), (4, 5, 6, 2)) * -1.0
    x289 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x289 += einsum(x195, (0, 1, 2, 3), (1, 0, 2, 3))
    x289 += einsum(x196, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x289 += einsum(x197, (0, 1, 2, 3), (0, 1, 2, 3)) * -3.0
    x290 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x290 += einsum(x282, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.1111111111111111
    x290 += einsum(x283, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.4444444444444444
    x290 += einsum(x284, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.1111111111111111
    x290 += einsum(x285, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.4444444444444444
    x290 += einsum(x286, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.2222222222222222
    x290 += einsum(x287, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x290 += einsum(x288, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.6666666666666666
    x290 += einsum(t1.aa, (0, 1), x289, (0, 2, 3, 4), (2, 3, 1, 4)) * 0.2222222222222222
    l1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x290, (4, 0, 2, 3), (1, 4)) * 9.0
    del x290
    x291 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x291 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), (1, 5, 2, 4))
    x292 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x292 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 6, 5, 7, 1, 2), (0, 7, 4, 6))
    x293 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x293 += einsum(t2.bbbb, (0, 1, 2, 3), x58, (0, 1, 4, 3, 5, 6), (4, 2, 5, 6))
    x294 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x294 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 4, 5, 0, 7, 2), (1, 7, 3, 6))
    x295 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x295 += einsum(t2.abab, (0, 1, 2, 3), x61, (1, 4, 0, 5, 6, 2), (4, 3, 5, 6))
    x296 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x296 += einsum(x291, (0, 1, 2, 3), (0, 1, 2, 3))
    x296 += einsum(x292, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x296 += einsum(x293, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x296 += einsum(x294, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x296 += einsum(x295, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x296 += einsum(t1.bb, (0, 1), x101, (0, 2, 3, 4), (1, 2, 3, 4))
    l1new_aa += einsum(v.aabb.ovvv, (0, 1, 2, 3), x296, (3, 2, 4, 0), (1, 4)) * -1.0
    del x296
    x297 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x297 += einsum(x279, (0, 1, 2, 3), (0, 1, 2, 3))
    del x279
    x297 += einsum(x280, (0, 1, 2, 3), (0, 1, 2, 3))
    del x280
    l2new_abab += einsum(x297, (0, 1, 2, 3), x71, (4, 5, 0, 1), (3, 5, 2, 4)) * -2.0
    del x71
    l2new_abab += einsum(v.aabb.vvov, (0, 1, 2, 3), x297, (4, 3, 5, 1), (0, 4, 5, 2)) * -2.0
    l2new_abab += einsum(x297, (0, 1, 2, 3), x31, (4, 1, 2, 5), (3, 0, 5, 4)) * 2.0
    x298 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x298 += einsum(x200, (0, 1), (0, 1))
    x298 += einsum(x201, (0, 1), (0, 1)) * 2.0
    x298 += einsum(x202, (0, 1), (0, 1)) * 0.9999999999999601
    x298 += einsum(x203, (0, 1), (0, 1)) * 1.9999999999999194
    x298 += einsum(x204, (0, 1), (0, 1)) * 2.9999999999998788
    x299 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x299 += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), (3, 4, 1, 2))
    x299 += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3))
    x299 += einsum(l2.abab, (0, 1, 2, 3), t3.babbab, (4, 5, 3, 6, 0, 1), (4, 6, 2, 5)) * 2.0
    x299 += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x299 += einsum(l2.aaaa, (0, 1, 2, 3), t3.abaaba, (4, 5, 3, 0, 6, 1), (5, 6, 2, 4)) * 2.0
    x299 += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x299 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x58, (2, 1, 5, 4, 6, 7), (0, 3, 6, 7)) * 3.0
    x299 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x74, (2, 0, 6, 5, 7, 4), (6, 3, 7, 1)) * -2.0
    x299 += einsum(t2.bbbb, (0, 1, 2, 3), x98, (1, 0, 4, 3, 5, 6), (4, 2, 5, 6)) * -2.0
    x299 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x61, (2, 5, 6, 1, 7, 4), (0, 3, 6, 7)) * -4.0
    x299 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x91, (1, 6, 7, 2, 5, 3), (6, 4, 7, 0)) * 2.0
    x299 += einsum(t2.abab, (0, 1, 2, 3), x99, (1, 4, 5, 0, 6, 2), (4, 3, 5, 6)) * 2.0
    x299 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x193, (6, 0, 2, 7, 5, 3), (1, 4, 6, 7)) * -3.0
    x299 += einsum(t2.abab, (0, 1, 2, 3), x297, (3, 4, 5, 2), (1, 4, 5, 0)) * -2.0
    x299 += einsum(t2.bbbb, (0, 1, 2, 3), x101, (1, 3, 4, 5), (0, 2, 4, 5)) * 2.0
    x299 += einsum(t2.abab, (0, 1, 2, 3), x96, (1, 4, 5, 2), (4, 3, 5, 0)) * -1.0
    x299 += einsum(t2.abab, (0, 1, 2, 3), x289, (0, 4, 5, 2), (1, 3, 4, 5)) * -2.0
    x299 += einsum(t1.bb, (0, 1), x108, (0, 2, 3, 4), (2, 1, 3, 4)) * -1.0
    del x108
    x299 += einsum(t1.bb, (0, 1), x298, (2, 3), (0, 1, 2, 3))
    l1new_aa += einsum(v.aabb.ovov, (0, 1, 2, 3), x299, (2, 3, 4, 0), (1, 4)) * -1.0
    del x299
    x300 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x300 += einsum(x195, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.3333333333333333
    x300 += einsum(x196, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.3333333333333333
    x300 += einsum(x197, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    l1new_aa += einsum(v.aaaa.oovv, (0, 1, 2, 3), x300, (1, 4, 0, 3), (2, 4)) * -6.0
    x301 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x301 += einsum(l2.aaaa, (0, 1, 2, 3), (2, 3, 0, 1)) * -0.3333333333333333
    x301 += einsum(x286, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.3333333333333333
    x301 += einsum(x288, (0, 1, 2, 3), (0, 1, 2, 3))
    x301 += einsum(t1.aa, (0, 1), x300, (0, 2, 3, 4), (2, 3, 4, 1))
    l1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x301, (4, 0, 3, 1), (2, 4)) * -6.0
    del x301
    x302 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x302 += einsum(t2.abab, (0, 1, 2, 3), l3.abaaba, (4, 3, 5, 6, 1, 0), (6, 4, 5, 2))
    x303 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x303 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 3, 4, 0), (4, 2, 3, 1))
    x303 += einsum(x302, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l1new_aa += einsum(v.aaaa.vvvv, (0, 1, 2, 3), x303, (4, 3, 1, 2), (0, 4)) * -2.0
    del x303
    x304 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x304 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovvv, (4, 2, 5, 3), (1, 5, 0, 4))
    x305 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x305 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 5, 2, 6, 1, 3), (4, 6, 5, 0))
    x306 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x306 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 3, 6, 1), (5, 6, 4, 0)) * -1.0
    x307 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x307 += einsum(t2.abab, (0, 1, 2, 3), x261, (4, 5, 0, 2), (1, 3, 4, 5))
    del x261
    x308 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x308 += einsum(t2.bbbb, (0, 1, 2, 3), x8, (1, 3, 4, 5), (0, 2, 4, 5)) * 2.0
    x309 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x309 += einsum(t2.abab, (0, 1, 2, 3), x39, (1, 4, 5, 2), (4, 3, 0, 5))
    x310 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x310 += einsum(x245, (0, 1), t2.abab, (2, 3, 1, 4), (3, 4, 2, 0))
    x311 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x311 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x311 += einsum(x246, (0, 1, 2, 3), (1, 0, 3, 2))
    x312 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x312 += einsum(t1.bb, (0, 1), x311, (1, 2, 3, 4), (0, 2, 3, 4))
    del x311
    x313 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x313 += einsum(t1.bb, (0, 1), x128, (0, 2, 3, 4), (2, 1, 3, 4))
    x314 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x314 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x314 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    x314 += einsum(x304, (0, 1, 2, 3), (0, 1, 2, 3))
    x314 += einsum(x305, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x314 += einsum(x306, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x314 += einsum(x307, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x314 += einsum(x308, (0, 1, 2, 3), (0, 1, 3, 2))
    x314 += einsum(x309, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x314 += einsum(x310, (0, 1, 2, 3), (0, 1, 2, 3))
    x314 += einsum(x312, (0, 1, 2, 3), (0, 1, 3, 2))
    x314 += einsum(x313, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    l1new_aa += einsum(l2.abab, (0, 1, 2, 3), x314, (3, 1, 2, 4), (0, 4)) * -1.0
    del x314
    x315 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x315 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x316 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x316 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 2, 5, 6, 3, 1), (4, 5, 0, 6))
    x317 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x317 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 2, 6, 1, 3), (4, 5, 0, 6))
    x318 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x318 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x319 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x319 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x319 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x319 += einsum(x318, (0, 1, 2, 3), (0, 1, 2, 3))
    x320 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x320 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x320 += einsum(x315, (0, 1, 2, 3), (2, 1, 0, 3))
    x320 += einsum(x316, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x320 += einsum(x317, (0, 1, 2, 3), (2, 1, 0, 3)) * -3.0
    x320 += einsum(t2.aaaa, (0, 1, 2, 3), x231, (4, 5, 1, 3), (5, 0, 4, 2)) * -2.0
    x320 += einsum(t2.abab, (0, 1, 2, 3), x31, (1, 3, 4, 5), (5, 0, 4, 2))
    x320 += einsum(x245, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4))
    x320 += einsum(t1.aa, (0, 1), x319, (2, 3, 1, 4), (3, 2, 0, 4))
    x320 += einsum(t1.aa, (0, 1), x254, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    l1new_aa += einsum(l2.aaaa, (0, 1, 2, 3), x320, (4, 3, 2, 1), (0, 4)) * -2.0
    del x320
    x321 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x321 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x58, (0, 2, 3, 5, 6, 7), (6, 7, 1, 4))
    x322 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x322 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x61, (1, 4, 6, 2, 7, 5), (6, 7, 0, 3)) * -1.0
    x323 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x323 += einsum(t1.aa, (0, 1), x61, (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 0, 6))
    x324 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x324 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x193, (6, 2, 1, 7, 4, 5), (6, 7, 0, 3)) * -1.0
    x325 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x325 += einsum(t1.aa, (0, 1), x193, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    x326 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x326 += einsum(x195, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.3333333333333333
    x326 += einsum(x196, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.3333333333333333
    x326 += einsum(x197, (0, 1, 2, 3), (0, 1, 2, 3))
    x327 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x327 += einsum(x195, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x327 += einsum(x196, (0, 1, 2, 3), (0, 1, 2, 3))
    x327 += einsum(x197, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    l2new_abab += einsum(v.aabb.ooov, (0, 1, 2, 3), x327, (0, 4, 1, 5), (5, 3, 4, 2)) * 2.0
    x328 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x328 += einsum(x229, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x328 += einsum(t1.aa, (0, 1), x327, (2, 3, 4, 1), (2, 3, 0, 4)) * 0.3333333333333468
    x329 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x329 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 0), (1, 2, 3, 4)) * 0.3333333333333333
    x329 += einsum(x195, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.3333333333333333
    x329 += einsum(l2.abab, (0, 1, 2, 3), t3.abaaba, (4, 3, 5, 6, 1, 0), (2, 4, 5, 6)) * 0.3333333333333333
    x329 += einsum(x196, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.3333333333333333
    x329 += einsum(x197, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x329 += einsum(x321, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.16666666666666666
    x329 += einsum(x322, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.6666666666666666
    x329 += einsum(t2.abab, (0, 1, 2, 3), x323, (1, 3, 4, 0, 5, 6), (4, 6, 5, 2)) * -0.3333333333333333
    x329 += einsum(x324, (0, 1, 2, 3), (0, 2, 1, 3)) * 1.5
    x329 += einsum(t2.aaaa, (0, 1, 2, 3), x228, (4, 3, 2, 5), (4, 0, 1, 5)) * -1.0
    x329 += einsum(t2.aaaa, (0, 1, 2, 3), x325, (4, 0, 1, 5, 6, 3), (4, 6, 5, 2)) * -1.0
    x329 += einsum(t2.abab, (0, 1, 2, 3), x66, (1, 3, 4, 5), (4, 0, 5, 2)) * 0.3333333333333333
    x329 += einsum(t2.aaaa, (0, 1, 2, 3), x326, (4, 1, 5, 3), (4, 0, 5, 2)) * -2.0
    x329 += einsum(t1.aa, (0, 1), x328, (2, 0, 3, 4), (2, 4, 3, 1)) * -0.9999999999999596
    del x328
    x329 += einsum(t1.aa, (0, 1), x298, (2, 3), (2, 0, 3, 1)) * 0.16666666666666666
    l1new_aa += einsum(v.aaaa.ovov, (0, 1, 2, 3), x329, (4, 2, 0, 3), (1, 4)) * -6.0
    del x329
    x330 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x330 += einsum(x282, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.1111111111111111
    x330 += einsum(x283, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.4444444444444444
    x330 += einsum(x284, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.1111111111111111
    x330 += einsum(x285, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.4444444444444444
    x330 += einsum(x287, (0, 1, 2, 3), (0, 1, 2, 3))
    l1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x330, (4, 0, 3, 1), (2, 4)) * 9.0
    del x330
    x331 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x331 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), (2, 3, 4, 5))
    x332 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x332 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 4, 7, 0, 1, 2), (3, 5, 6, 7))
    x333 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x333 += einsum(x196, (0, 1, 2, 3), (0, 1, 2, 3))
    x333 += einsum(x197, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x334 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x334 += einsum(t1.aa, (0, 1), x333, (2, 3, 4, 1), (0, 2, 3, 4))
    x335 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x335 += einsum(x331, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x335 += einsum(x332, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.9999999999999601
    x335 += einsum(x334, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    x336 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x336 += einsum(l2.aaaa, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 3, 6, 0, 1), (2, 4, 5, 6)) * 0.5
    x336 += einsum(x321, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.08333333333333333
    del x321
    x336 += einsum(x322, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.3333333333333333
    del x322
    x336 += einsum(t2.aaaa, (0, 1, 2, 3), x302, (4, 2, 3, 5), (4, 0, 1, 5)) * 0.16666666666666666
    x336 += einsum(x324, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.75
    del x324
    x336 += einsum(t2.abab, (0, 1, 2, 3), x66, (1, 3, 4, 5), (4, 0, 5, 2)) * 0.16666666666666666
    x336 += einsum(t2.aaaa, (0, 1, 2, 3), x326, (4, 1, 5, 3), (4, 0, 5, 2)) * -1.0
    x336 += einsum(t1.aa, (0, 1), x335, (0, 2, 3, 4), (2, 4, 3, 1)) * -0.16666666666666666
    del x335
    x336 += einsum(t1.aa, (0, 1), x298, (2, 3), (2, 0, 3, 1)) * 0.08333333333333333
    del x298
    l1new_aa += einsum(v.aaaa.ovov, (0, 1, 2, 3), x336, (4, 2, 0, 1), (3, 4)) * 12.0
    del x336
    x337 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x337 += einsum(t1.aa, (0, 1), (0, 1)) * -0.5000000000000202
    x337 += einsum(x186, (0, 1), (0, 1)) * -0.5000000000000202
    del x186
    x337 += einsum(x187, (0, 1), (0, 1)) * -1.0000000000000404
    del x187
    x337 += einsum(x188, (0, 1), (0, 1)) * -0.5000000000000202
    del x188
    x337 += einsum(x189, (0, 1), (0, 1)) * -1.0000000000000404
    del x189
    x337 += einsum(x190, (0, 1), (0, 1)) * -1.5000000000000604
    del x190
    x337 += einsum(x191, (0, 1), (0, 1)) * 0.5000000000000002
    del x191
    x337 += einsum(x192, (0, 1), (0, 1))
    del x192
    x337 += einsum(x194, (0, 1), (0, 1)) * 1.4999999999999998
    del x194
    x337 += einsum(t2.abab, (0, 1, 2, 3), x66, (1, 3, 0, 4), (4, 2)) * 1.0000000000000404
    x337 += einsum(t2.aaaa, (0, 1, 2, 3), x198, (0, 1, 4, 3), (4, 2)) * -3.000000000000121
    del x198
    x337 += einsum(t1.aa, (0, 1), x205, (0, 2), (2, 1)) * 0.5000000000000202
    del x205
    l1new_aa += einsum(x337, (0, 1), x240, (0, 2, 1, 3), (3, 2)) * 1.9999999999999194
    del x337
    x338 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x338 += einsum(x179, (0, 1), (0, 1)) * 0.5
    del x179
    x338 += einsum(x180, (0, 1), (0, 1)) * 0.5
    x338 += einsum(x181, (0, 1), (0, 1))
    x338 += einsum(x182, (0, 1), (0, 1)) * 0.49999999999998007
    x339 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x339 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x339 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    l1new_aa += einsum(x338, (0, 1), x339, (2, 1, 0, 3), (3, 2)) * -2.0
    del x338
    l2new_abab += einsum(x100, (0, 1, 2, 3), x339, (4, 3, 2, 5), (5, 1, 4, 0)) * -2.0
    x340 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x340 += einsum(x160, (0, 1), (0, 1)) * 0.5
    del x160
    x340 += einsum(x161, (0, 1), (0, 1))
    x340 += einsum(x162, (0, 1), (0, 1)) * 0.5
    x340 += einsum(x163, (0, 1), (0, 1)) * 1.4999999999999394
    x340 += einsum(x164, (0, 1), (1, 0)) * 0.9999999999999597
    x340 += einsum(x211, (0, 1), (0, 1)) * 0.49999999999998007
    l1new_aa += einsum(x340, (0, 1), v.aabb.ovvv, (2, 3, 1, 0), (3, 2)) * 2.0
    del x340
    x341 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x341 += einsum(x175, (0, 1), (0, 1))
    x341 += einsum(x109, (0, 1), (0, 1)) * 2.0
    x341 += einsum(x110, (0, 1), (0, 1))
    x341 += einsum(x111, (0, 1), (0, 1)) * 2.9999999999998788
    x341 += einsum(x112, (0, 1), (0, 1)) * 1.9999999999999194
    x341 += einsum(x113, (0, 1), (0, 1)) * 0.9999999999999601
    l2new_abab += einsum(x341, (0, 1), v.aabb.ovov, (2, 3, 1, 4), (3, 4, 2, 0)) * -1.0
    x342 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x342 += einsum(l1.bb, (0, 1), (1, 0)) * -1.0
    x342 += einsum(t1.bb, (0, 1), (0, 1)) * -1.0
    x342 += einsum(x166, (0, 1), (0, 1)) * -2.0
    del x166
    x342 += einsum(x167, (0, 1), (0, 1)) * -1.0
    del x167
    x342 += einsum(x168, (0, 1), (0, 1)) * -3.0
    del x168
    x342 += einsum(x169, (0, 1), (0, 1)) * -2.0
    del x169
    x342 += einsum(x170, (0, 1), (0, 1)) * -1.0
    del x170
    x342 += einsum(x171, (0, 1), (0, 1)) * 2.9999999999998788
    del x171
    x342 += einsum(x172, (0, 1), (0, 1)) * 1.9999999999999194
    del x172
    x342 += einsum(x173, (0, 1), (0, 1)) * 0.9999999999999601
    del x173
    x342 += einsum(t2.bbbb, (0, 1, 2, 3), x137, (0, 1, 4, 3), (4, 2)) * -2.0
    x342 += einsum(t2.abab, (0, 1, 2, 3), x139, (1, 4, 0, 2), (4, 3)) * 2.0
    x342 += einsum(t1.bb, (0, 1), x341, (0, 2), (2, 1))
    l1new_aa += einsum(x342, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (3, 2)) * -1.0
    del x342
    x343 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x343 += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5000000000000202
    del x102
    x343 += einsum(x103, (0, 1, 2, 3), (0, 1, 2, 3))
    del x103
    x343 += einsum(x104, (0, 1, 2, 3), (0, 1, 2, 3))
    del x104
    x343 += einsum(t1.bb, (0, 1), x101, (2, 1, 3, 4), (0, 2, 3, 4)) * 0.5000000000000202
    x343 += einsum(t1.aa, (0, 1), x106, (2, 3, 4, 1), (3, 2, 4, 0)) * 1.0000000000000404
    l1new_aa += einsum(v.aabb.ovoo, (0, 1, 2, 3), x343, (3, 2, 4, 0), (1, 4)) * 1.9999999999999194
    del x343
    x344 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x344 += einsum(x183, (0, 1), (0, 1))
    x344 += einsum(x184, (0, 1), (0, 1)) * 1.4999999999999998
    l1new_aa += einsum(x344, (0, 1), x339, (2, 1, 0, 3), (3, 2)) * -1.9999999999999194
    del x344
    x345 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x345 += einsum(x331, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.3333333333333333
    x345 += einsum(x332, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.33333333333332005
    x345 += einsum(t1.aa, (0, 1), x327, (2, 3, 4, 1), (0, 2, 3, 4)) * 0.3333333333333333
    l1new_aa += einsum(v.aaaa.ooov, (0, 1, 2, 3), x345, (0, 4, 1, 2), (3, 4)) * -6.0
    del x345
    x346 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x346 += einsum(t1.aa, (0, 1), x333, (2, 3, 4, 1), (2, 3, 0, 4)) * 0.3333333333333333
    l1new_aa += einsum(v.aaaa.ooov, (0, 1, 2, 3), x346, (4, 1, 2, 0), (3, 4)) * 6.0
    del x346
    x347 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x347 += einsum(x199, (0, 1), (0, 1)) * 0.3333333333333468
    x347 += einsum(x200, (0, 1), (0, 1)) * 0.3333333333333468
    x347 += einsum(x201, (0, 1), (0, 1)) * 0.6666666666666936
    x347 += einsum(x202, (0, 1), (0, 1)) * 0.33333333333333354
    x347 += einsum(x204, (0, 1), (0, 1))
    x348 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x348 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x348 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    l1new_aa += einsum(x347, (0, 1), x348, (0, 2, 1, 3), (3, 2)) * -2.9999999999998788
    del x347
    l1new_aa += einsum(x203, (0, 1), x348, (0, 1, 2, 3), (3, 2)) * 1.9999999999999194
    l2new_abab += einsum(x348, (0, 1, 2, 3), x66, (4, 5, 0, 1), (3, 5, 2, 4)) * 2.0
    l3new_babbab += einsum(x348, (0, 1, 2, 3), x58, (4, 5, 6, 7, 0, 2), (6, 3, 7, 5, 1, 4)) * 2.0
    x349 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x349 += einsum(x175, (0, 1), (0, 1)) * 0.5
    del x175
    x349 += einsum(x109, (0, 1), (0, 1))
    del x109
    x349 += einsum(x110, (0, 1), (0, 1)) * 0.5
    del x110
    x349 += einsum(x111, (0, 1), (0, 1)) * 1.4999999999999394
    del x111
    x349 += einsum(x112, (0, 1), (1, 0)) * 0.9999999999999597
    del x112
    x349 += einsum(x113, (0, 1), (1, 0)) * 0.49999999999998007
    del x113
    l1new_aa += einsum(x349, (0, 1), v.aabb.ovoo, (2, 3, 1, 0), (3, 2)) * -2.0
    del x349
    x350 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x350 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x350 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l1new_aa += einsum(l1.aa, (0, 1), x350, (1, 2, 0, 3), (3, 2)) * -1.0
    del x350
    x351 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x351 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 1), (2, 3))
    x352 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x352 += einsum(f.aa.vv, (0, 1), (0, 1))
    x352 += einsum(x351, (0, 1), (1, 0))
    x352 += einsum(t1.aa, (0, 1), x235, (0, 1, 2, 3), (3, 2)) * -1.0
    l1new_aa += einsum(l1.aa, (0, 1), x352, (0, 2), (2, 1))
    del x352
    x353 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x353 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3))
    x354 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x354 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x355 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x355 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 1, 3), (0, 4))
    x356 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x356 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x356 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x357 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x357 += einsum(t1.aa, (0, 1), x356, (2, 3, 0, 1), (2, 3))
    del x356
    x358 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x358 += einsum(t1.aa, (0, 1), x245, (2, 1), (0, 2))
    x359 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x359 += einsum(f.aa.oo, (0, 1), (0, 1))
    x359 += einsum(x353, (0, 1), (1, 0))
    x359 += einsum(x354, (0, 1), (0, 1))
    x359 += einsum(x355, (0, 1), (0, 1)) * 2.0
    x359 += einsum(x357, (0, 1), (1, 0)) * -1.0
    x359 += einsum(x358, (0, 1), (0, 1))
    del x358
    l1new_aa += einsum(l1.aa, (0, 1), x359, (1, 2), (0, 2)) * -1.0
    l2new_abab += einsum(x359, (0, 1), l2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    l3new_babbab += einsum(x359, (0, 1), l3.babbab, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -2.0
    del x359
    x360 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x360 += einsum(x199, (0, 1), (0, 1)) * 0.3333333333333468
    x360 += einsum(x200, (0, 1), (0, 1)) * 0.3333333333333468
    x360 += einsum(x201, (0, 1), (0, 1)) * 0.6666666666666936
    x360 += einsum(x202, (0, 1), (0, 1)) * 0.33333333333333354
    x360 += einsum(x203, (0, 1), (0, 1)) * 0.6666666666666667
    x360 += einsum(x204, (0, 1), (0, 1))
    l1new_aa += einsum(f.aa.ov, (0, 1), x360, (2, 0), (1, 2)) * -2.9999999999998788
    del x360
    x361 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x361 += einsum(x243, (0, 1), (0, 1))
    del x243
    x361 += einsum(x244, (0, 1), (0, 1)) * -1.0
    del x244
    l1new_aa += einsum(x199, (0, 1), x361, (1, 2), (2, 0)) * -1.0
    x362 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x362 += einsum(l1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 0), (1, 2, 3, 4))
    x363 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x363 += einsum(l2.abab, (0, 1, 2, 3), x238, (3, 1, 4, 5), (2, 4, 0, 5))
    x364 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x364 += einsum(x5, (0, 1, 2, 3), x93, (0, 1, 4, 5), (4, 2, 5, 3))
    x365 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x365 += einsum(x302, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.3333333333333333
    x365 += einsum(x228, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l2new_abab += einsum(v.aabb.vvov, (0, 1, 2, 3), x365, (4, 5, 0, 1), (5, 3, 4, 2)) * 6.0
    x366 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x366 += einsum(x339, (0, 1, 2, 3), x365, (4, 2, 5, 1), (0, 4, 3, 5)) * 6.0
    del x365
    x367 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x367 += einsum(x196, (0, 1, 2, 3), (0, 1, 2, 3))
    del x196
    x367 += einsum(x197, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0000000000000004
    del x197
    x368 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x368 += einsum(t1.aa, (0, 1), x367, (2, 0, 3, 4), (2, 3, 1, 4)) * 0.22222222222223104
    del x367
    x369 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x369 += einsum(x282, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.11111111111111552
    x369 += einsum(x283, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.4444444444444621
    x369 += einsum(x284, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.11111111111111108
    x369 += einsum(x285, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.44444444444444453
    x369 += einsum(x286, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.22222222222223104
    x369 += einsum(x287, (0, 1, 2, 3), (0, 1, 2, 3))
    x369 += einsum(x288, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.6666666666666932
    x369 += einsum(x368, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x368
    x370 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x370 += einsum(x233, (0, 1, 2, 3), x369, (4, 0, 5, 3), (1, 4, 2, 5)) * 8.999999999999643
    del x369
    x371 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x371 += einsum(t1.bb, (0, 1), x106, (0, 2, 3, 4), (2, 1, 3, 4)) * 2.0
    x372 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x372 += einsum(l2.abab, (0, 1, 2, 3), (3, 1, 2, 0))
    x372 += einsum(x271, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x372 += einsum(x272, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x372 += einsum(x273, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.999999999999881
    x372 += einsum(x274, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x372 += einsum(x275, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.999999999999842
    x372 += einsum(x276, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x372 += einsum(x277, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.999999999999881
    x372 += einsum(x371, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x371
    x373 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x373 += einsum(v.aabb.ovov, (0, 1, 2, 3), x372, (2, 3, 4, 5), (0, 4, 1, 5))
    del x372
    x374 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x374 += einsum(v.aabb.ovvv, (0, 1, 2, 3), x297, (2, 3, 4, 5), (0, 4, 1, 5)) * 2.0
    x375 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x375 += einsum(t1.aa, (0, 1), x339, (2, 1, 3, 4), (0, 2, 3, 4))
    x376 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x376 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x376 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x376 += einsum(x375, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x375
    x377 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x377 += einsum(l2.aaaa, (0, 1, 2, 3), x376, (3, 4, 5, 1), (2, 4, 0, 5)) * 2.0
    del x376
    x378 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x378 += einsum(x327, (0, 1, 2, 3), x348, (1, 4, 2, 5), (0, 4, 3, 5)) * 2.0
    del x327
    x379 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x379 += einsum(x180, (0, 1), (0, 1))
    x379 += einsum(x181, (0, 1), (0, 1)) * 2.0
    x379 += einsum(x182, (0, 1), (0, 1)) * 0.9999999999999601
    x379 += einsum(x183, (0, 1), (0, 1)) * 1.9999999999999194
    x379 += einsum(x184, (0, 1), (0, 1)) * 2.9999999999998788
    x380 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x380 += einsum(x379, (0, 1), v.aaaa.ovov, (2, 1, 3, 4), (2, 3, 0, 4))
    del x379
    x381 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x381 += einsum(v.aabb.ovoo, (0, 1, 2, 3), x96, (2, 3, 4, 5), (0, 4, 1, 5))
    x382 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x382 += einsum(x230, (0, 1, 2, 3), (0, 1, 2, 3))
    x382 += einsum(x230, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x383 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x383 += einsum(x195, (0, 1, 2, 3), x382, (0, 4, 2, 5), (1, 4, 3, 5)) * 2.0
    x384 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x384 += einsum(x199, (0, 1), (0, 1))
    x384 += einsum(x200, (0, 1), (0, 1))
    x384 += einsum(x201, (0, 1), (0, 1)) * 2.0
    x384 += einsum(x202, (0, 1), (0, 1)) * 0.9999999999999601
    x384 += einsum(x203, (0, 1), (0, 1)) * 1.9999999999999194
    x384 += einsum(x204, (0, 1), (0, 1)) * 2.9999999999998788
    x385 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x385 += einsum(x384, (0, 1), v.aaaa.ovov, (2, 3, 1, 4), (0, 2, 4, 3))
    del x384
    x386 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x386 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x386 += einsum(x230, (0, 1, 2, 3), (0, 1, 2, 3))
    x387 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x387 += einsum(l1.aa, (0, 1), x386, (1, 2, 3, 4), (2, 3, 0, 4))
    x388 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x388 += einsum(f.aa.ov, (0, 1), l1.aa, (2, 3), (0, 3, 1, 2))
    x388 += einsum(x362, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x362
    x388 += einsum(x363, (0, 1, 2, 3), (0, 1, 2, 3))
    del x363
    x388 += einsum(x364, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x364
    x388 += einsum(x366, (0, 1, 2, 3), (1, 0, 3, 2))
    del x366
    x388 += einsum(x370, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x370
    x388 += einsum(x373, (0, 1, 2, 3), (1, 0, 3, 2))
    del x373
    x388 += einsum(x374, (0, 1, 2, 3), (1, 0, 3, 2))
    del x374
    x388 += einsum(x377, (0, 1, 2, 3), (0, 1, 2, 3))
    del x377
    x388 += einsum(x378, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x378
    x388 += einsum(x380, (0, 1, 2, 3), (1, 0, 2, 3))
    del x380
    x388 += einsum(x381, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x381
    x388 += einsum(x383, (0, 1, 2, 3), (0, 1, 2, 3))
    del x383
    x388 += einsum(x385, (0, 1, 2, 3), (0, 1, 3, 2))
    del x385
    x388 += einsum(x387, (0, 1, 2, 3), (0, 1, 2, 3))
    del x387
    x388 += einsum(l1.aa, (0, 1), x361, (2, 3), (1, 2, 0, 3))
    l2new_aaaa += einsum(x388, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_aaaa += einsum(x388, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_aaaa += einsum(x388, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new_aaaa += einsum(x388, (0, 1, 2, 3), (3, 2, 1, 0))
    del x388
    x389 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x389 += einsum(f.aa.ov, (0, 1), t1.aa, (2, 1), (0, 2))
    x390 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x390 += einsum(x389, (0, 1), l2.aaaa, (2, 3, 4, 1), (0, 4, 2, 3))
    x391 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x391 += einsum(l2.aaaa, (0, 1, 2, 3), x253, (2, 4, 3, 5), (4, 5, 0, 1))
    x392 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x392 += einsum(f.aa.ov, (0, 1), t2.abab, (2, 3, 1, 4), (3, 4, 0, 2))
    x393 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x393 += einsum(x392, (0, 1, 2, 3), l3.abaaba, (4, 1, 5, 6, 0, 3), (2, 6, 4, 5))
    del x392
    x394 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x394 += einsum(f.aa.ov, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4))
    x395 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x395 += einsum(x394, (0, 1, 2, 3), l3.aaaaaa, (4, 5, 3, 6, 1, 2), (0, 6, 4, 5)) * -1.0
    del x394
    x396 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x396 += einsum(x361, (0, 1), t2.abab, (2, 3, 1, 4), (3, 4, 2, 0))
    x397 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x397 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x397 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    x397 += einsum(x304, (0, 1, 2, 3), (0, 1, 2, 3))
    x397 += einsum(x305, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.999999999999921
    x397 += einsum(x306, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.99999999999992
    x397 += einsum(x307, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x397 += einsum(x308, (0, 1, 2, 3), (0, 1, 3, 2))
    x397 += einsum(x309, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x397 += einsum(x396, (0, 1, 2, 3), (0, 1, 2, 3))
    del x396
    x397 += einsum(x312, (0, 1, 2, 3), (0, 1, 3, 2))
    x397 += einsum(x313, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x398 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x398 += einsum(x397, (0, 1, 2, 3), l3.abaaba, (4, 1, 5, 6, 0, 2), (6, 3, 4, 5)) * -2.0
    del x397
    x399 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x399 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x399 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x399 += einsum(x230, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x399 += einsum(x230, (0, 1, 2, 3), (0, 2, 1, 3))
    x400 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x400 += einsum(t2.aaaa, (0, 1, 2, 3), x399, (4, 1, 5, 3), (0, 4, 5, 2)) * 2.0
    del x399
    x401 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x401 += einsum(t2.abab, (0, 1, 2, 3), x8, (1, 3, 4, 5), (0, 4, 5, 2))
    x402 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x402 += einsum(x361, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    x403 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x403 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x403 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x403 += einsum(x318, (0, 1, 2, 3), (1, 0, 2, 3))
    del x318
    x404 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x404 += einsum(t1.aa, (0, 1), x403, (2, 3, 1, 4), (0, 2, 3, 4))
    del x403
    x405 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x405 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x405 += einsum(x251, (0, 1, 2, 3), (3, 1, 0, 2))
    x405 += einsum(x252, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    x405 += einsum(x253, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x405 += einsum(x253, (0, 1, 2, 3), (3, 0, 2, 1))
    x406 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x406 += einsum(t1.aa, (0, 1), x405, (0, 2, 3, 4), (2, 3, 4, 1))
    del x405
    x407 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x407 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x407 += einsum(x315, (0, 1, 2, 3), (1, 0, 2, 3))
    x407 += einsum(x316, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.9999999999999606
    x407 += einsum(x317, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.999999999999881
    x407 += einsum(x400, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x400
    x407 += einsum(x401, (0, 1, 2, 3), (0, 2, 1, 3))
    del x401
    x407 += einsum(x402, (0, 1, 2, 3), (1, 0, 2, 3))
    del x402
    x407 += einsum(x404, (0, 1, 2, 3), (2, 0, 1, 3))
    del x404
    x407 += einsum(x406, (0, 1, 2, 3), (1, 0, 2, 3))
    del x406
    x408 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x408 += einsum(x407, (0, 1, 2, 3), l3.aaaaaa, (4, 5, 3, 6, 0, 1), (6, 2, 4, 5)) * -6.0
    del x407
    x409 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x409 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x409 += einsum(x230, (0, 1, 2, 3), (0, 2, 1, 3))
    x410 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x410 += einsum(x302, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x302
    x410 += einsum(x228, (0, 1, 2, 3), (0, 2, 1, 3)) * -3.0
    del x228
    x411 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x411 += einsum(x409, (0, 1, 2, 3), x410, (0, 4, 5, 3), (1, 2, 4, 5)) * 2.0
    del x409
    x412 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x412 += einsum(t1.aa, (0, 1), x361, (2, 1), (0, 2))
    x413 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x413 += einsum(x353, (0, 1), (1, 0))
    del x353
    x413 += einsum(x354, (0, 1), (0, 1))
    del x354
    x413 += einsum(x355, (0, 1), (0, 1)) * 2.0
    del x355
    x413 += einsum(x357, (0, 1), (1, 0)) * -1.0
    del x357
    x413 += einsum(x412, (0, 1), (0, 1))
    del x412
    x414 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x414 += einsum(x413, (0, 1), l2.aaaa, (2, 3, 4, 0), (4, 1, 2, 3)) * -2.0
    x415 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x415 += einsum(x390, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x390
    x415 += einsum(x391, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x391
    x415 += einsum(x393, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x393
    x415 += einsum(x395, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.0
    del x395
    x415 += einsum(x398, (0, 1, 2, 3), (0, 1, 3, 2))
    del x398
    x415 += einsum(x408, (0, 1, 2, 3), (0, 1, 2, 3))
    del x408
    x415 += einsum(x411, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x411
    x415 += einsum(x414, (0, 1, 2, 3), (0, 1, 3, 2))
    del x414
    l2new_aaaa += einsum(x415, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new_aaaa += einsum(x415, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x415
    x416 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x416 += einsum(f.aa.vv, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    x417 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x417 += einsum(x22, (0, 1, 2, 3), x91, (0, 1, 4, 5, 6, 3), (4, 5, 6, 2))
    x418 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x418 += einsum(t1.bb, (0, 1), v.aabb.vvvv, (2, 3, 4, 1), (0, 4, 2, 3))
    x419 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x419 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.vvov, (4, 5, 1, 3), (0, 2, 4, 5))
    x420 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x420 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovvv, (0, 4, 5, 3), (1, 5, 2, 4))
    x421 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x421 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 0, 2, 5, 6, 3), (4, 5, 6, 1))
    x422 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x422 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.abaaba, (0, 4, 2, 5, 6, 1), (4, 6, 5, 3)) * -1.0
    x423 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x423 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x423 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1))
    x424 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x424 += einsum(t2.abab, (0, 1, 2, 3), x423, (0, 2, 4, 5), (1, 3, 4, 5))
    del x423
    x425 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x425 += einsum(t2.abab, (0, 1, 2, 3), x39, (1, 4, 0, 5), (4, 3, 2, 5))
    x426 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x426 += einsum(v.aabb.vvov, (0, 1, 2, 3), (2, 3, 0, 1))
    x426 += einsum(x418, (0, 1, 2, 3), (0, 1, 3, 2))
    x426 += einsum(x419, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x426 += einsum(x420, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x426 += einsum(x421, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.999999999999921
    x426 += einsum(x422, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.99999999999992
    x426 += einsum(x424, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x424
    x426 += einsum(x425, (0, 1, 2, 3), (0, 1, 3, 2))
    x427 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x427 += einsum(x426, (0, 1, 2, 3), l3.abaaba, (4, 1, 3, 5, 0, 6), (5, 6, 4, 2)) * -2.0
    del x426
    x428 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x428 += einsum(t1.aa, (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x429 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x429 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvov, (4, 5, 1, 3), (0, 2, 4, 5))
    x430 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x430 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 2, 0, 5, 3, 6), (4, 5, 6, 1))
    x431 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x431 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa, (4, 0, 2, 5, 6, 3), (4, 5, 6, 1))
    x432 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x432 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x432 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x433 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x433 += einsum(t2.aaaa, (0, 1, 2, 3), x432, (1, 3, 4, 5), (0, 2, 4, 5)) * 2.0
    x434 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x434 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x434 += einsum(x230, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x435 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x435 += einsum(t2.aaaa, (0, 1, 2, 3), x434, (0, 4, 1, 5), (4, 2, 3, 5))
    x436 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x436 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x436 += einsum(x428, (0, 1, 2, 3), (0, 2, 3, 1))
    x436 += einsum(x429, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x436 += einsum(x430, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.9999999999999606
    x436 += einsum(x431, (0, 1, 2, 3), (0, 2, 1, 3)) * -2.999999999999881
    x436 += einsum(x433, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x433
    x436 += einsum(x435, (0, 1, 2, 3), (0, 2, 1, 3))
    del x435
    x437 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x437 += einsum(x436, (0, 1, 2, 3), l3.aaaaaa, (4, 1, 2, 5, 6, 0), (5, 6, 4, 3)) * -6.0
    del x436
    x438 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x438 += einsum(t1.bb, (0, 1), x39, (0, 2, 3, 4), (2, 1, 3, 4))
    x439 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x439 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x439 += einsum(x238, (0, 1, 2, 3), (0, 1, 2, 3))
    x439 += einsum(x239, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x439 += einsum(x241, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x241
    x439 += einsum(x438, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x438
    l2new_abab += einsum(x439, (0, 1, 2, 3), x58, (0, 4, 5, 1, 6, 2), (3, 5, 6, 4)) * 2.0
    x440 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x440 += einsum(x439, (0, 1, 2, 3), x61, (0, 1, 4, 5, 2, 6), (4, 5, 6, 3)) * 2.0
    del x439
    x441 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x441 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x442 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x442 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x442 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x442 += einsum(x441, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x441
    x442 += einsum(x232, (0, 1, 2, 3), (0, 1, 2, 3))
    x442 += einsum(x234, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x443 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x443 += einsum(x442, (0, 1, 2, 3), x193, (4, 5, 0, 1, 2, 6), (4, 5, 6, 3)) * 6.0
    del x442
    x444 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x444 += einsum(t1.aa, (0, 1), x230, (2, 3, 0, 4), (2, 3, 1, 4))
    x445 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x445 += einsum(t1.aa, (0, 1), x235, (2, 1, 3, 4), (0, 2, 3, 4))
    x446 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x446 += einsum(x444, (0, 1, 2, 3), (0, 1, 2, 3))
    del x444
    x446 += einsum(x445, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    l2new_abab += einsum(x446, (0, 1, 2, 3), x61, (4, 5, 0, 6, 1, 2), (3, 5, 6, 4)) * 2.0
    x447 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x447 += einsum(x446, (0, 1, 2, 3), x193, (4, 5, 0, 1, 2, 6), (4, 5, 6, 3)) * 6.0
    del x446
    x448 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x448 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x448 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    l2new_abab += einsum(x448, (0, 1, 2, 3), x74, (0, 4, 1, 5, 6, 2), (3, 5, 6, 4)) * 2.0
    x449 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x449 += einsum(x448, (0, 1, 2, 3), x91, (0, 1, 4, 5, 2, 6), (4, 5, 6, 3)) * 2.0
    del x448
    x450 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x450 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x326, (4, 5, 0, 3), (4, 5, 1, 2)) * 6.0
    x451 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x451 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    x452 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x452 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x453 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x453 += einsum(t1.aa, (0, 1), x339, (0, 1, 2, 3), (2, 3))
    del x339
    x454 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x454 += einsum(x351, (0, 1), (1, 0)) * -1.0
    x454 += einsum(x451, (0, 1), (1, 0))
    x454 += einsum(x452, (0, 1), (1, 0)) * 2.0
    x454 += einsum(x453, (0, 1), (1, 0)) * -1.0
    del x453
    x455 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x455 += einsum(x454, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 2, 0)) * -2.0
    x456 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x456 += einsum(v.aaaa.ovov, (0, 1, 2, 3), x334, (0, 4, 5, 2), (4, 5, 3, 1)) * 2.0
    del x334
    x457 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x457 += einsum(x361, (0, 1), x289, (2, 3, 0, 4), (2, 3, 4, 1)) * 2.0
    x458 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x458 += einsum(f.aa.ov, (0, 1), x300, (2, 3, 0, 4), (2, 3, 4, 1)) * 6.0
    del x300
    x459 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x459 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x459 += einsum(x416, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x416
    x459 += einsum(x417, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x417
    x459 += einsum(x427, (0, 1, 2, 3), (1, 0, 3, 2))
    del x427
    x459 += einsum(x437, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x437
    x459 += einsum(x440, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x440
    x459 += einsum(x443, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x443
    x459 += einsum(x447, (0, 1, 2, 3), (0, 1, 3, 2))
    del x447
    x459 += einsum(x449, (0, 1, 2, 3), (0, 1, 3, 2))
    del x449
    x459 += einsum(x450, (0, 1, 2, 3), (0, 1, 3, 2))
    del x450
    x459 += einsum(x455, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x455
    x459 += einsum(x456, (0, 1, 2, 3), (0, 1, 2, 3))
    del x456
    x459 += einsum(x457, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x457
    x459 += einsum(x458, (0, 1, 2, 3), (1, 0, 2, 3))
    del x458
    l2new_aaaa += einsum(x459, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new_aaaa += einsum(x459, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    del x459
    x460 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x460 += einsum(f.aa.oo, (0, 1), l2.aaaa, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_aaaa += einsum(x460, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    l2new_aaaa += einsum(x460, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    del x460
    x461 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x461 += einsum(x331, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.3333333333333468
    del x331
    x461 += einsum(t1.aa, (0, 1), x195, (2, 3, 4, 1), (2, 3, 4, 0)) * 0.3333333333333468
    del x195
    x461 += einsum(x332, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.33333333333333354
    del x332
    x461 += einsum(x229, (0, 1, 2, 3), (0, 1, 3, 2))
    del x229
    l2new_aaaa += einsum(v.aaaa.ovov, (0, 1, 2, 3), x461, (4, 5, 0, 2), (3, 1, 5, 4)) * -5.9999999999997575
    del x461
    x462 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x462 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x462 += einsum(x251, (0, 1, 2, 3), (1, 3, 0, 2)) * -1.0
    x462 += einsum(x252, (0, 1, 2, 3), (0, 3, 1, 2))
    l2new_aaaa += einsum(l2.aaaa, (0, 1, 2, 3), x462, (2, 4, 3, 5), (0, 1, 4, 5)) * -2.0
    x463 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x463 += einsum(t1.aa, (0, 1), v.aabb.vvvv, (2, 1, 3, 4), (3, 4, 0, 2))
    x464 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x464 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvov, (4, 2, 1, 5), (3, 5, 0, 4))
    x465 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x465 += einsum(t2.aaaa, (0, 1, 2, 3), v.aabb.ovvv, (1, 3, 4, 5), (4, 5, 0, 2))
    x466 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x466 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.babbab, (0, 4, 2, 5, 6, 3), (5, 1, 4, 6))
    x467 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x467 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba, (4, 2, 0, 5, 6, 1), (6, 3, 4, 5))
    x468 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x468 += einsum(t2.abab, (0, 1, 2, 3), x217, (1, 4, 3, 5), (4, 5, 0, 2))
    x469 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x469 += einsum(t2.abab, (0, 1, 2, 3), x8, (1, 4, 0, 5), (3, 4, 5, 2))
    x470 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x470 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (2, 3, 0, 1))
    x470 += einsum(x463, (0, 1, 2, 3), (1, 0, 2, 3))
    x470 += einsum(x464, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x470 += einsum(x465, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x470 += einsum(x466, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.99999999999992
    x470 += einsum(x467, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.999999999999921
    x470 += einsum(x468, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x470 += einsum(x469, (0, 1, 2, 3), (0, 1, 2, 3))
    l2new_abab += einsum(x470, (0, 1, 2, 3), l3.abaaba, (4, 0, 3, 5, 6, 2), (4, 1, 5, 6)) * 2.0
    del x470
    x471 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x471 += einsum(v.aabb.vvov, (0, 1, 2, 3), (2, 3, 0, 1))
    x471 += einsum(x418, (0, 1, 2, 3), (0, 1, 3, 2))
    del x418
    x471 += einsum(x419, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x419
    x471 += einsum(x420, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x420
    x471 += einsum(x421, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.999999999999921
    del x421
    x471 += einsum(x422, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.99999999999992
    del x422
    x471 += einsum(t2.abab, (0, 1, 2, 3), x235, (0, 2, 4, 5), (1, 3, 5, 4)) * -1.0
    x471 += einsum(x425, (0, 1, 2, 3), (0, 1, 2, 3))
    del x425
    l2new_abab += einsum(x471, (0, 1, 2, 3), l3.babbab, (4, 2, 1, 5, 6, 0), (3, 4, 6, 5)) * 2.0
    del x471
    x472 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x472 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x472 += einsum(x428, (0, 1, 2, 3), (0, 3, 2, 1)) * -0.5
    del x428
    x472 += einsum(x429, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    del x429
    x472 += einsum(x430, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.49999999999998
    del x430
    x472 += einsum(x431, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.4999999999999405
    del x431
    x472 += einsum(t2.aaaa, (0, 1, 2, 3), x432, (1, 3, 4, 5), (0, 2, 5, 4)) * -1.0
    del x432
    x472 += einsum(t2.aaaa, (0, 1, 2, 3), x434, (0, 4, 1, 5), (4, 2, 3, 5)) * -0.5
    del x434
    l2new_abab += einsum(x472, (0, 1, 2, 3), l3.abaaba, (1, 4, 2, 5, 6, 0), (3, 4, 5, 6)) * 4.0
    del x472
    x473 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x473 += einsum(t1.bb, (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x474 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x474 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovvv, (0, 2, 4, 5), (1, 3, 4, 5))
    x475 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x475 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb, (4, 0, 2, 5, 6, 3), (4, 5, 6, 1))
    x476 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x476 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab, (4, 0, 2, 5, 1, 6), (4, 5, 6, 3))
    x477 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x477 += einsum(t2.bbbb, (0, 1, 2, 3), x217, (1, 3, 4, 5), (0, 2, 4, 5)) * 2.0
    x478 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x478 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x478 += einsum(x3, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x479 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x479 += einsum(t2.bbbb, (0, 1, 2, 3), x478, (0, 4, 1, 5), (4, 2, 3, 5)) * -1.0
    del x478
    x480 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x480 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x480 += einsum(x473, (0, 1, 2, 3), (0, 2, 3, 1))
    x480 += einsum(x474, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x480 += einsum(x475, (0, 1, 2, 3), (0, 2, 1, 3)) * -2.999999999999881
    x480 += einsum(x476, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.99999999999996
    x480 += einsum(x477, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x480 += einsum(x479, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l2new_abab += einsum(x480, (0, 1, 2, 3), l3.babbab, (1, 4, 2, 5, 6, 0), (4, 3, 6, 5)) * 2.0
    del x480
    x481 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x481 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x481 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    x481 += einsum(x304, (0, 1, 2, 3), (0, 1, 2, 3))
    del x304
    x481 += einsum(x305, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.999999999999921
    del x305
    x481 += einsum(x306, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.99999999999992
    del x306
    x481 += einsum(x307, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x307
    x481 += einsum(x308, (0, 1, 2, 3), (0, 1, 3, 2))
    del x308
    x481 += einsum(x309, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x309
    x481 += einsum(x310, (0, 1, 2, 3), (0, 1, 2, 3))
    del x310
    x481 += einsum(x312, (0, 1, 2, 3), (0, 1, 3, 2))
    del x312
    x481 += einsum(x313, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x313
    l2new_abab += einsum(x481, (0, 1, 2, 3), l3.babbab, (4, 5, 1, 6, 2, 0), (5, 4, 3, 6)) * -2.0
    del x481
    x482 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x482 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1)) * 0.5
    x482 += einsum(x118, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    x482 += einsum(x119, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x482 += einsum(x120, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.99999999999996
    x482 += einsum(x121, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.9999999999999605
    x482 += einsum(t2.abab, (0, 1, 2, 3), x122, (4, 5, 1, 3), (5, 4, 0, 2)) * -0.5
    del x122
    x482 += einsum(t2.aaaa, (0, 1, 2, 3), x39, (4, 5, 1, 3), (5, 4, 0, 2))
    x482 += einsum(t2.abab, (0, 1, 2, 3), x8, (4, 3, 0, 5), (1, 4, 5, 2)) * -0.5
    x482 += einsum(x21, (0, 1), t2.abab, (2, 3, 4, 1), (3, 0, 2, 4)) * 0.5
    del x21
    x482 += einsum(t1.bb, (0, 1), x126, (2, 1, 3, 4), (0, 2, 3, 4)) * 0.5
    del x126
    x482 += einsum(t1.aa, (0, 1), x128, (2, 3, 0, 4), (3, 2, 4, 1)) * -0.5
    del x128
    l2new_abab += einsum(x482, (0, 1, 2, 3), l3.abaaba, (4, 5, 3, 6, 0, 2), (4, 5, 6, 1)) * -4.0
    del x482
    x483 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x483 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.3333333333333466
    x483 += einsum(x315, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.3333333333333466
    del x315
    x483 += einsum(x316, (0, 1, 2, 3), (2, 1, 0, 3)) * -0.33333333333333326
    del x316
    x483 += einsum(x317, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x317
    x483 += einsum(t2.aaaa, (0, 1, 2, 3), x231, (4, 5, 1, 3), (5, 0, 4, 2)) * -0.6666666666666932
    del x231
    x483 += einsum(t2.abab, (0, 1, 2, 3), x31, (1, 3, 4, 5), (5, 0, 4, 2)) * 0.3333333333333466
    x483 += einsum(x245, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4)) * 0.3333333333333466
    del x245
    x483 += einsum(t1.aa, (0, 1), x319, (2, 3, 1, 4), (3, 2, 0, 4)) * 0.3333333333333466
    del x319
    x483 += einsum(t1.aa, (0, 1), x254, (0, 2, 3, 4), (3, 2, 4, 1)) * -0.3333333333333466
    del x254
    l2new_abab += einsum(x483, (0, 1, 2, 3), l3.abaaba, (4, 5, 3, 1, 6, 2), (4, 5, 0, 6)) * 5.999999999999762
    del x483
    x484 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x484 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x484 += einsum(x146, (0, 1, 2, 3), (2, 1, 0, 3))
    x484 += einsum(x147, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.999999999999881
    x484 += einsum(x148, (0, 1, 2, 3), (2, 1, 0, 3)) * -0.99999999999996
    x484 += einsum(x149, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x484 += einsum(x150, (0, 1, 2, 3), (2, 0, 1, 3))
    del x150
    x484 += einsum(x151, (0, 1, 2, 3), (2, 1, 0, 3))
    del x151
    x484 += einsum(x154, (0, 1, 2, 3), (2, 1, 0, 3))
    del x154
    x484 += einsum(x155, (0, 1, 2, 3), (1, 2, 0, 3))
    del x155
    l2new_abab += einsum(x484, (0, 1, 2, 3), l3.babbab, (4, 5, 3, 1, 6, 2), (5, 4, 6, 0)) * 2.0
    del x484
    x485 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x485 += einsum(t1.aa, (0, 1), x8, (2, 3, 0, 4), (2, 3, 4, 1))
    del x8
    x486 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x486 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x486 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    x486 += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x486 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x17
    x486 += einsum(x485, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x485
    l2new_abab += einsum(x486, (0, 1, 2, 3), x91, (4, 0, 2, 5, 6, 3), (6, 1, 5, 4)) * 2.0
    x487 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x487 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x487 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x487 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    x487 += einsum(x232, (0, 1, 2, 3), (0, 1, 2, 3))
    x487 += einsum(x234, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x234
    l2new_abab += einsum(x487, (0, 1, 2, 3), x61, (4, 5, 0, 6, 1, 2), (3, 5, 6, 4)) * -2.0
    del x487
    x488 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x488 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x488 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x488 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    x488 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    x488 += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    l2new_abab += einsum(x488, (0, 1, 2, 3), x74, (0, 4, 1, 2, 5, 6), (6, 3, 5, 4)) * -2.0
    del x488
    x489 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x489 += einsum(t1.bb, (0, 1), x12, (2, 1, 3, 4), (0, 2, 3, 4))
    x490 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x490 += einsum(t1.bb, (0, 1), x3, (2, 3, 0, 4), (2, 3, 1, 4))
    x490 += einsum(x489, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    l2new_abab += einsum(x490, (0, 1, 2, 3), x74, (0, 4, 1, 2, 5, 6), (6, 3, 5, 4)) * 2.0
    del x490
    x491 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x491 += einsum(x271, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5000000000000198
    del x271
    x491 += einsum(x272, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5000000000000198
    del x272
    x491 += einsum(x273, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.7499999999999999
    del x273
    x491 += einsum(x274, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5000000000000198
    del x274
    x491 += einsum(x275, (0, 1, 2, 3), (0, 1, 2, 3))
    del x275
    x491 += einsum(x276, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5000000000000198
    del x276
    x491 += einsum(x277, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.7499999999999999
    del x277
    x491 += einsum(t1.bb, (0, 1), x106, (0, 2, 3, 4), (2, 1, 3, 4)) * -0.5000000000000198
    l2new_abab += einsum(x10, (0, 1, 2, 3), x491, (0, 2, 4, 5), (5, 3, 4, 1)) * -3.999999999999842
    del x491
    x492 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x492 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x492 += einsum(x247, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    l2new_abab += einsum(x492, (0, 1, 2, 3), x61, (4, 0, 2, 5, 3, 6), (6, 1, 5, 4)) * 2.0
    x493 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x493 += einsum(l2.aaaa, (0, 1, 2, 3), (2, 3, 0, 1))
    x493 += einsum(x282, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    del x282
    x493 += einsum(x283, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x283
    x493 += einsum(x284, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.49999999999998
    del x284
    x493 += einsum(x285, (0, 1, 2, 3), (1, 0, 3, 2)) * 1.999999999999921
    del x285
    x493 += einsum(x286, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x286
    x493 += einsum(x287, (0, 1, 2, 3), (1, 0, 3, 2)) * 4.4999999999998215
    del x287
    x493 += einsum(x288, (0, 1, 2, 3), (1, 0, 3, 2)) * -3.0
    del x288
    x493 += einsum(t1.aa, (0, 1), x333, (0, 2, 3, 4), (3, 2, 1, 4))
    del x333
    l2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), x493, (0, 4, 1, 5), (5, 3, 4, 2)) * 2.0
    del x493
    x494 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x494 += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3))
    del x64
    x494 += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3))
    del x65
    x495 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x495 += einsum(t1.aa, (0, 1), x494, (2, 3, 0, 4), (2, 3, 4, 1)) * 2.0
    x496 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x496 += einsum(l2.abab, (0, 1, 2, 3), (3, 1, 2, 0))
    x496 += einsum(x54, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x496 += einsum(x55, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x496 += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.999999999999881
    x496 += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.999999999999842
    x496 += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x496 += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.999999999999881
    x496 += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x496 += einsum(x495, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x495
    l2new_abab += einsum(v.aaaa.ovov, (0, 1, 2, 3), x496, (4, 5, 2, 3), (1, 5, 0, 4))
    x497 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x497 += einsum(l2.bbbb, (0, 1, 2, 3), (2, 3, 0, 1)) * 2.0
    x497 += einsum(x81, (0, 1, 2, 3), (1, 0, 3, 2)) * 4.0
    x497 += einsum(x82, (0, 1, 2, 3), (1, 0, 3, 2))
    x497 += einsum(x83, (0, 1, 2, 3), (1, 0, 3, 2)) * 8.999999999999643
    x497 += einsum(x73, (0, 1, 2, 3), (1, 0, 3, 2)) * -6.0
    x497 += einsum(x84, (0, 1, 2, 3), (1, 0, 3, 2)) * 3.999999999999842
    x497 += einsum(x75, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    x497 += einsum(x85, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.99999999999996
    x497 += einsum(t1.bb, (0, 1), x157, (0, 2, 3, 4), (3, 2, 1, 4)) * 6.0
    l2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), x497, (2, 4, 3, 5), (1, 5, 0, 4))
    del x497
    x498 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x498 += einsum(x54, (0, 1, 2, 3), (0, 1, 2, 3))
    del x54
    x498 += einsum(x55, (0, 1, 2, 3), (0, 1, 2, 3))
    del x55
    x498 += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.4999999999999405
    del x56
    x498 += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.999999999999921
    del x57
    x498 += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x59
    x498 += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.4999999999999405
    del x60
    x498 += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x62
    x498 += einsum(t1.aa, (0, 1), x494, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    l2new_abab += einsum(v.aaaa.ovov, (0, 1, 2, 3), x498, (4, 5, 2, 1), (3, 5, 0, 4)) * -2.0
    del x498
    x499 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x499 += einsum(x0, (0, 1, 2, 3), (0, 2, 1, 3)) * -3.0
    x499 += einsum(x116, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l2new_abab += einsum(v.aabb.ovvv, (0, 1, 2, 3), x499, (4, 5, 2, 3), (1, 5, 0, 4)) * 2.0
    x500 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x500 += einsum(x291, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x291
    x500 += einsum(x292, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.99999999999996
    del x292
    x500 += einsum(x293, (0, 1, 2, 3), (0, 1, 2, 3))
    del x293
    x500 += einsum(x294, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.99999999999996
    del x294
    x500 += einsum(x295, (0, 1, 2, 3), (0, 1, 2, 3))
    del x295
    x500 += einsum(t1.bb, (0, 1), x494, (0, 2, 3, 4), (2, 1, 3, 4))
    del x494
    l2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), x500, (4, 3, 5, 0), (1, 4, 5, 2)) * 2.0
    del x500
    x501 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x501 += einsum(x87, (0, 1, 2, 3), (0, 1, 2, 3))
    del x87
    x501 += einsum(x88, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.99999999999992
    del x88
    x501 += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x89
    x501 += einsum(x90, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.99999999999992
    del x90
    x501 += einsum(x92, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x92
    x501 += einsum(t1.aa, (0, 1), x106, (2, 3, 0, 4), (2, 3, 4, 1)) * 2.0
    del x106
    l2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), x501, (4, 2, 5, 1), (5, 3, 0, 4))
    del x501
    x502 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x502 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x502 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x502 += einsum(x489, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x489
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x502, (3, 4, 1, 5), (0, 5, 2, 4)) * -1.0
    del x502
    x503 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x503 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x503 += einsum(x445, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x445
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x503, (2, 4, 0, 5), (5, 1, 4, 3)) * -1.0
    del x503
    x504 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x504 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x504 += einsum(x246, (0, 1, 2, 3), (1, 0, 2, 3))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x504, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    del x504
    x505 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x505 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x505 += einsum(x22, (0, 1, 2, 3), (0, 1, 3, 2))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x505, (3, 4, 0, 5), (5, 1, 2, 4)) * -1.0
    del x505
    x506 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x506 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x506 += einsum(x29, (0, 1, 2, 3), (1, 0, 2, 3))
    del x29
    x506 += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3))
    del x30
    x506 += einsum(x249, (0, 1, 2, 3), (0, 1, 3, 2))
    del x249
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x506, (3, 4, 2, 5), (0, 1, 5, 4))
    x507 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x507 += einsum(t1.aa, (0, 1), x235, (0, 2, 1, 3), (2, 3))
    del x235
    x508 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x508 += einsum(f.aa.vv, (0, 1), (0, 1)) * -1.0
    x508 += einsum(x351, (0, 1), (1, 0)) * -1.0
    del x351
    x508 += einsum(x451, (0, 1), (0, 1))
    del x451
    x508 += einsum(x452, (0, 1), (0, 1)) * 2.0
    del x452
    x508 += einsum(x507, (0, 1), (1, 0)) * -1.0
    del x507
    l2new_abab += einsum(x508, (0, 1), l2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    l3new_babbab += einsum(x508, (0, 1), l3.babbab, (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -2.0
    del x508
    x509 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x509 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 4, 1, 3), (2, 4))
    x510 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x510 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    x511 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x511 += einsum(t1.bb, (0, 1), x217, (0, 1, 2, 3), (2, 3))
    del x217
    x512 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x512 += einsum(f.bb.vv, (0, 1), (0, 1)) * -1.0
    x512 += einsum(x216, (0, 1), (1, 0)) * -1.0
    x512 += einsum(x509, (0, 1), (0, 1)) * 2.0
    x512 += einsum(x510, (0, 1), (0, 1))
    x512 += einsum(x511, (0, 1), (1, 0)) * -1.0
    del x511
    l2new_abab += einsum(x512, (0, 1), l2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    l3new_abaaba += einsum(x512, (0, 1), l3.abaaba, (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -2.0
    del x512
    x513 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x513 += einsum(x180, (0, 1), (0, 1)) * 1.00000000000004
    del x180
    x513 += einsum(x181, (0, 1), (0, 1)) * 2.00000000000008
    del x181
    x513 += einsum(x182, (0, 1), (0, 1))
    del x182
    x513 += einsum(x183, (0, 1), (0, 1)) * 1.9999999999999991
    del x183
    x513 += einsum(x184, (0, 1), (0, 1)) * 2.9999999999999982
    del x184
    l2new_abab += einsum(x513, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (0, 4, 2, 3)) * -0.9999999999999601
    del x513
    x514 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x514 += einsum(x161, (0, 1), (0, 1)) * 2.0
    x514 += einsum(x162, (0, 1), (0, 1))
    x514 += einsum(x163, (0, 1), (0, 1)) * 2.9999999999998788
    x514 += einsum(x164, (0, 1), (0, 1)) * 1.9999999999999194
    x514 += einsum(x211, (0, 1), (0, 1)) * 0.9999999999999601
    l2new_abab += einsum(x514, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (3, 0, 2, 4)) * -1.0
    del x514
    x515 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x515 += einsum(x230, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x515 += einsum(x230, (0, 1, 2, 3), (0, 2, 1, 3))
    l2new_abab += einsum(x515, (0, 1, 2, 3), x63, (4, 5, 0, 1), (3, 5, 2, 4)) * -1.0
    l3new_babbab += einsum(x515, (0, 1, 2, 3), x58, (4, 5, 6, 7, 0, 1), (6, 3, 7, 5, 2, 4)) * 2.0
    x516 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x516 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x516 += einsum(x3, (0, 1, 2, 3), (0, 2, 1, 3))
    l2new_abab += einsum(x516, (0, 1, 2, 3), x93, (0, 2, 4, 5), (5, 3, 4, 1))
    del x93
    l3new_abaaba += einsum(x516, (0, 1, 2, 3), x91, (0, 1, 4, 5, 6, 7), (6, 3, 7, 5, 2, 4)) * 2.0
    x517 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x517 += einsum(x199, (0, 1), (0, 1)) * 1.00000000000004
    del x199
    x517 += einsum(x200, (0, 1), (0, 1)) * 1.00000000000004
    del x200
    x517 += einsum(x201, (0, 1), (0, 1)) * 2.00000000000008
    del x201
    x517 += einsum(x202, (0, 1), (0, 1))
    del x202
    x517 += einsum(x203, (0, 1), (0, 1)) * 1.9999999999999991
    del x203
    x517 += einsum(x204, (0, 1), (0, 1)) * 2.9999999999999982
    del x204
    l2new_abab += einsum(x517, (0, 1), v.aabb.ovov, (1, 2, 3, 4), (2, 4, 0, 3)) * -0.9999999999999601
    del x517
    x518 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x518 += einsum(l1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 0), (1, 2, 3, 4))
    x519 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x519 += einsum(l2.abab, (0, 1, 2, 3), x15, (4, 5, 2, 0), (3, 4, 1, 5))
    x520 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x520 += einsum(x63, (0, 1, 2, 3), x7, (4, 5, 2, 3), (0, 4, 1, 5))
    del x63
    x521 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x521 += einsum(x499, (0, 1, 2, 3), x80, (4, 3, 1, 5), (4, 0, 5, 2)) * 2.0
    del x499
    x522 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x522 += einsum(t1.bb, (0, 1), x157, (2, 0, 3, 4), (2, 3, 1, 4)) * 3.0
    del x157
    x523 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x523 += einsum(x81, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x81
    x523 += einsum(x82, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x82
    x523 += einsum(x83, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.4999999999998215
    del x83
    x523 += einsum(x73, (0, 1, 2, 3), (0, 1, 2, 3)) * -3.0
    del x73
    x523 += einsum(x84, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.999999999999921
    del x84
    x523 += einsum(x75, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x75
    x523 += einsum(x85, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.49999999999998
    del x85
    x523 += einsum(x522, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x522
    x524 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x524 += einsum(x10, (0, 1, 2, 3), x523, (4, 0, 5, 2), (1, 4, 3, 5)) * 2.0
    del x523
    x525 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x525 += einsum(v.aabb.ovov, (0, 1, 2, 3), x496, (4, 5, 0, 1), (2, 4, 3, 5))
    del x496
    x526 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x526 += einsum(v.aabb.vvov, (0, 1, 2, 3), x100, (4, 5, 0, 1), (2, 4, 3, 5)) * 2.0
    x527 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x527 += einsum(t1.bb, (0, 1), x80, (2, 1, 3, 4), (0, 2, 3, 4))
    del x80
    x528 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x528 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x528 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x528 += einsum(x527, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x527
    x529 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x529 += einsum(l2.bbbb, (0, 1, 2, 3), x528, (3, 4, 5, 1), (2, 4, 0, 5)) * 2.0
    del x528
    x530 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x530 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.3333333333333333
    x530 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3))
    del x76
    x530 += einsum(x77, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.3333333333333333
    del x77
    x531 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x531 += einsum(x213, (0, 1, 2, 3), x530, (4, 0, 2, 5), (1, 4, 3, 5)) * 6.0
    del x530
    x532 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x532 += einsum(x161, (0, 1), (0, 1))
    del x161
    x532 += einsum(x162, (0, 1), (0, 1)) * 0.5
    del x162
    x532 += einsum(x163, (0, 1), (0, 1)) * 1.4999999999999394
    del x163
    x532 += einsum(x164, (0, 1), (0, 1)) * 0.9999999999999597
    del x164
    x532 += einsum(x211, (0, 1), (0, 1)) * 0.49999999999998007
    del x211
    x533 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x533 += einsum(x532, (0, 1), v.bbbb.ovov, (2, 1, 3, 4), (2, 3, 4, 0)) * 2.0
    del x532
    x534 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x534 += einsum(v.aabb.ooov, (0, 1, 2, 3), x101, (4, 5, 0, 1), (2, 4, 3, 5))
    del x101
    x535 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x535 += einsum(x1, (0, 1, 2, 3), x516, (0, 2, 4, 5), (1, 4, 3, 5)) * 2.0
    del x1
    x536 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x536 += einsum(x341, (0, 1), v.bbbb.ovov, (2, 3, 1, 4), (2, 0, 4, 3))
    del x341
    x537 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x537 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x537 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    x538 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x538 += einsum(l1.bb, (0, 1), x537, (1, 2, 3, 4), (2, 3, 0, 4))
    x539 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x539 += einsum(f.bb.ov, (0, 1), l1.bb, (2, 3), (0, 3, 1, 2))
    x539 += einsum(x518, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x518
    x539 += einsum(x519, (0, 1, 2, 3), (0, 1, 2, 3))
    del x519
    x539 += einsum(x520, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x520
    x539 += einsum(x521, (0, 1, 2, 3), (1, 0, 3, 2))
    del x521
    x539 += einsum(x524, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x524
    x539 += einsum(x525, (0, 1, 2, 3), (1, 0, 3, 2))
    del x525
    x539 += einsum(x526, (0, 1, 2, 3), (1, 0, 3, 2))
    del x526
    x539 += einsum(x529, (0, 1, 2, 3), (0, 1, 2, 3))
    del x529
    x539 += einsum(x531, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x531
    x539 += einsum(x533, (0, 1, 2, 3), (1, 0, 3, 2))
    del x533
    x539 += einsum(x534, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x534
    x539 += einsum(x535, (0, 1, 2, 3), (0, 1, 2, 3))
    del x535
    x539 += einsum(x536, (0, 1, 2, 3), (1, 0, 3, 2))
    del x536
    x539 += einsum(x538, (0, 1, 2, 3), (0, 1, 2, 3))
    del x538
    x539 += einsum(l1.bb, (0, 1), x227, (2, 3), (1, 2, 0, 3))
    l2new_bbbb += einsum(x539, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_bbbb += einsum(x539, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_bbbb += einsum(x539, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new_bbbb += einsum(x539, (0, 1, 2, 3), (3, 2, 1, 0))
    del x539
    x540 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x540 += einsum(f.bb.ov, (0, 1), t1.bb, (2, 1), (0, 2))
    x541 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x541 += einsum(x540, (0, 1), l2.bbbb, (2, 3, 4, 1), (0, 4, 2, 3))
    x542 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x542 += einsum(l2.bbbb, (0, 1, 2, 3), x27, (3, 2, 4, 5), (4, 5, 0, 1)) * -1.0
    x543 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x543 += einsum(f.bb.ov, (0, 1), t2.bbbb, (2, 3, 4, 1), (0, 2, 3, 4))
    x544 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x544 += einsum(x543, (0, 1, 2, 3), l3.bbbbbb, (4, 5, 3, 6, 1, 2), (0, 6, 4, 5)) * -1.0
    del x543
    x545 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x545 += einsum(f.bb.ov, (0, 1), t2.abab, (2, 3, 4, 1), (0, 3, 2, 4))
    x546 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x546 += einsum(x545, (0, 1, 2, 3), l3.babbab, (4, 3, 5, 6, 2, 1), (0, 6, 4, 5))
    del x545
    x547 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x547 += einsum(x227, (0, 1), t2.abab, (2, 3, 4, 1), (3, 0, 2, 4))
    x548 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x548 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x548 += einsum(x118, (0, 1, 2, 3), (1, 0, 2, 3))
    del x118
    x548 += einsum(x119, (0, 1, 2, 3), (0, 1, 2, 3))
    del x119
    x548 += einsum(x120, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.99999999999992
    del x120
    x548 += einsum(x121, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.999999999999921
    del x121
    x548 += einsum(x123, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x123
    x548 += einsum(x124, (0, 1, 2, 3), (1, 0, 2, 3))
    del x124
    x548 += einsum(x125, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x125
    x548 += einsum(x547, (0, 1, 2, 3), (0, 1, 2, 3))
    del x547
    x548 += einsum(x127, (0, 1, 2, 3), (0, 1, 2, 3))
    del x127
    x548 += einsum(x129, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x129
    x549 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x549 += einsum(x548, (0, 1, 2, 3), l3.babbab, (4, 3, 5, 6, 2, 0), (6, 1, 4, 5)) * -2.0
    del x548
    x550 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x550 += einsum(t2.abab, (0, 1, 2, 3), x39, (4, 5, 0, 2), (1, 4, 5, 3))
    del x39
    x551 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x551 += einsum(x227, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    x552 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x552 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x552 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x552 += einsum(x152, (0, 1, 2, 3), (1, 0, 2, 3))
    del x152
    x553 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x553 += einsum(t1.bb, (0, 1), x552, (2, 3, 1, 4), (0, 2, 3, 4))
    del x552
    x554 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x554 += einsum(t1.bb, (0, 1), x48, (0, 2, 3, 4), (2, 3, 4, 1))
    del x48
    x555 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x555 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3))
    x555 += einsum(x146, (0, 1, 2, 3), (1, 0, 2, 3))
    del x146
    x555 += einsum(x147, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.999999999999881
    del x147
    x555 += einsum(x148, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.9999999999999606
    del x148
    x555 += einsum(x149, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x149
    x555 += einsum(x550, (0, 1, 2, 3), (0, 2, 1, 3))
    del x550
    x555 += einsum(x551, (0, 1, 2, 3), (1, 0, 2, 3))
    del x551
    x555 += einsum(x553, (0, 1, 2, 3), (2, 0, 1, 3))
    del x553
    x555 += einsum(x554, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x554
    x556 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x556 += einsum(x555, (0, 1, 2, 3), l3.bbbbbb, (4, 5, 3, 6, 0, 1), (6, 2, 4, 5)) * -6.0
    del x555
    x557 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x557 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x557 += einsum(x3, (0, 1, 2, 3), (0, 2, 1, 3))
    x558 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x558 += einsum(x136, (0, 1, 2, 3), x557, (0, 4, 5, 3), (4, 5, 1, 2)) * 6.0
    del x557
    x559 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x559 += einsum(t1.bb, (0, 1), x227, (2, 1), (0, 2))
    x560 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x560 += einsum(x219, (0, 1), (1, 0))
    x560 += einsum(x220, (0, 1), (0, 1)) * 2.0
    x560 += einsum(x221, (0, 1), (0, 1))
    x560 += einsum(x223, (0, 1), (1, 0)) * -1.0
    x560 += einsum(x559, (0, 1), (0, 1))
    del x559
    x561 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x561 += einsum(x560, (0, 1), l2.bbbb, (2, 3, 4, 0), (4, 1, 2, 3)) * -2.0
    x562 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x562 += einsum(x541, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x541
    x562 += einsum(x542, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x542
    x562 += einsum(x544, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.0
    del x544
    x562 += einsum(x546, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x546
    x562 += einsum(x549, (0, 1, 2, 3), (0, 1, 3, 2))
    del x549
    x562 += einsum(x556, (0, 1, 2, 3), (0, 1, 2, 3))
    del x556
    x562 += einsum(x558, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x558
    x562 += einsum(x561, (0, 1, 2, 3), (0, 1, 3, 2))
    del x561
    l2new_bbbb += einsum(x562, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new_bbbb += einsum(x562, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x562
    x563 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x563 += einsum(f.bb.vv, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    x564 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x564 += einsum(x246, (0, 1, 2, 3), x58, (4, 5, 6, 1, 2, 3), (4, 5, 6, 0))
    x565 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x565 += einsum(v.aabb.ovvv, (0, 1, 2, 3), (2, 3, 0, 1))
    x565 += einsum(x463, (0, 1, 2, 3), (1, 0, 2, 3))
    del x463
    x565 += einsum(x464, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x464
    x565 += einsum(x465, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x465
    x565 += einsum(x466, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.99999999999992
    del x466
    x565 += einsum(x467, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.999999999999921
    del x467
    x565 += einsum(x468, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x468
    x565 += einsum(x469, (0, 1, 2, 3), (1, 0, 2, 3))
    del x469
    x566 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x566 += einsum(x565, (0, 1, 2, 3), l3.babbab, (4, 3, 1, 5, 2, 6), (5, 6, 4, 0)) * -2.0
    del x565
    x567 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x567 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    x567 += einsum(x473, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    del x473
    x567 += einsum(x474, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x474
    x567 += einsum(x475, (0, 1, 2, 3), (0, 2, 1, 3)) * -2.999999999999881
    del x475
    x567 += einsum(x476, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.9999999999999606
    del x476
    x567 += einsum(x477, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x477
    x567 += einsum(x479, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x479
    x568 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x568 += einsum(x567, (0, 1, 2, 3), l3.bbbbbb, (4, 1, 2, 5, 6, 0), (5, 6, 4, 3)) * -6.0
    del x567
    x569 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x569 += einsum(x486, (0, 1, 2, 3), x74, (4, 5, 0, 6, 2, 3), (4, 5, 6, 1)) * 2.0
    del x486
    x570 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x570 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    x571 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x571 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x571 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x571 += einsum(x570, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x570
    x571 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    x571 += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x11
    x572 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x572 += einsum(x571, (0, 1, 2, 3), x72, (4, 5, 0, 1, 2, 6), (4, 5, 6, 3)) * 6.0
    del x571
    x573 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x573 += einsum(t1.bb, (0, 1), x3, (2, 0, 3, 4), (2, 3, 1, 4))
    x574 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x574 += einsum(x573, (0, 1, 2, 3), (0, 1, 2, 3))
    del x573
    x574 += einsum(x13, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x575 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x575 += einsum(x574, (0, 1, 2, 3), x72, (4, 5, 0, 1, 2, 6), (4, 5, 6, 3)) * 6.0
    del x574
    x576 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x576 += einsum(x492, (0, 1, 2, 3), x58, (4, 5, 0, 6, 2, 3), (4, 5, 6, 1)) * 2.0
    del x492
    x577 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x577 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x137, (4, 5, 0, 3), (4, 5, 1, 2)) * 2.0
    x578 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x578 += einsum(t1.bb, (0, 1), x12, (0, 2, 1, 3), (2, 3))
    del x12
    x579 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x579 += einsum(x216, (0, 1), (1, 0)) * -1.0
    del x216
    x579 += einsum(x509, (0, 1), (1, 0)) * 2.0
    del x509
    x579 += einsum(x510, (0, 1), (1, 0))
    del x510
    x579 += einsum(x578, (0, 1), (0, 1)) * -1.0
    del x578
    x580 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x580 += einsum(x579, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 2, 0)) * -2.0
    x581 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x581 += einsum(v.bbbb.ovov, (0, 1, 2, 3), x158, (2, 4, 5, 0), (4, 5, 1, 3)) * 6.0
    del x158
    x582 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x582 += einsum(x227, (0, 1), x78, (2, 3, 0, 4), (2, 3, 4, 1)) * 2.0
    x583 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x583 += einsum(f.bb.ov, (0, 1), x78, (2, 3, 0, 4), (2, 3, 1, 4)) * 2.0
    x584 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x584 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x584 += einsum(x563, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x563
    x584 += einsum(x564, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x564
    x584 += einsum(x566, (0, 1, 2, 3), (1, 0, 3, 2))
    del x566
    x584 += einsum(x568, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x568
    x584 += einsum(x569, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x569
    x584 += einsum(x572, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x572
    x584 += einsum(x575, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x575
    x584 += einsum(x576, (0, 1, 2, 3), (0, 1, 3, 2))
    del x576
    x584 += einsum(x577, (0, 1, 2, 3), (0, 1, 3, 2))
    del x577
    x584 += einsum(x580, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x580
    x584 += einsum(x581, (0, 1, 2, 3), (0, 1, 2, 3))
    del x581
    x584 += einsum(x582, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x582
    x584 += einsum(x583, (0, 1, 2, 3), (1, 0, 3, 2))
    del x583
    l2new_bbbb += einsum(x584, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new_bbbb += einsum(x584, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    del x584
    x585 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x585 += einsum(f.bb.oo, (0, 1), l2.bbbb, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_bbbb += einsum(x585, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    l2new_bbbb += einsum(x585, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    del x585
    x586 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x586 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x586 += einsum(x25, (0, 1, 2, 3), (1, 3, 2, 0))
    x586 += einsum(x26, (0, 1, 2, 3), (0, 2, 3, 1))
    l2new_bbbb += einsum(l2.bbbb, (0, 1, 2, 3), x586, (3, 4, 5, 2), (0, 1, 5, 4)) * 2.0
    x587 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x587 += einsum(x141, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.3333333333333468
    del x141
    x587 += einsum(x2, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.3333333333333468
    del x2
    x587 += einsum(x142, (0, 1, 2, 3), (0, 1, 3, 2))
    del x142
    l2new_bbbb += einsum(v.bbbb.ovov, (0, 1, 2, 3), x587, (4, 5, 2, 0), (1, 3, 5, 4)) * -5.9999999999997575
    del x587
    x588 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x588 += einsum(v.aabb.ovoo, (0, 1, 2, 3), x91, (2, 3, 4, 5, 6, 7), (4, 5, 0, 6, 7, 1))
    x589 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x589 += einsum(t2.aaaa, (0, 1, 2, 3), x233, (1, 4, 5, 3), (0, 4, 2, 5)) * 2.0000000000000204
    x590 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x590 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x590 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x590 += einsum(x232, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.00000000000001
    x590 += einsum(x589, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x589
    x590 += einsum(x236, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x591 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x591 += einsum(x590, (0, 1, 2, 3), l3.aaaaaa, (4, 5, 2, 6, 7, 0), (6, 7, 1, 4, 5, 3)) * 6.0
    del x590
    x592 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x592 += einsum(t2.abab, (0, 1, 2, 3), x240, (0, 4, 2, 5), (1, 3, 4, 5)) * 1.00000000000001
    x593 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x593 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x593 += einsum(x238, (0, 1, 2, 3), (0, 1, 2, 3))
    x593 += einsum(x592, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x594 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x594 += einsum(x593, (0, 1, 2, 3), l3.abaaba, (4, 1, 5, 6, 0, 7), (6, 7, 2, 4, 5, 3)) * 2.0
    del x593
    x595 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x595 += einsum(x515, (0, 1, 2, 3), x193, (4, 5, 0, 1, 6, 7), (4, 5, 2, 6, 7, 3)) * 6.0
    del x515
    x596 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x596 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x596 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x597 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x597 += einsum(x596, (0, 1, 2, 3), x193, (4, 5, 0, 1, 6, 7), (4, 5, 2, 6, 7, 3)) * 6.0
    del x596
    x598 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x598 += einsum(x588, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    del x588
    x598 += einsum(x591, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x591
    x598 += einsum(x594, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    del x594
    x598 += einsum(x595, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x595
    x598 += einsum(x597, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x597
    l3new_aaaaaa += einsum(x598, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2))
    l3new_aaaaaa += einsum(x598, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2)) * -1.0
    l3new_aaaaaa += einsum(x598, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2)) * -1.0
    l3new_aaaaaa += einsum(x598, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0)) * -1.0
    l3new_aaaaaa += einsum(x598, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0))
    l3new_aaaaaa += einsum(x598, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0))
    l3new_aaaaaa += einsum(x598, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 1, 0))
    l3new_aaaaaa += einsum(x598, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0)) * -1.0
    l3new_aaaaaa += einsum(x598, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0)) * -1.0
    del x598
    x599 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x599 += einsum(l2.aaaa, (0, 1, 2, 3), x230, (3, 4, 5, 6), (2, 4, 5, 0, 1, 6))
    del x230
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2)) * 2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2)) * 2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 1, 2)) * -2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1)) * -2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1)) * -2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1)) * 2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2)) * -2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2)) * -2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 0, 2)) * 2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 0, 1)) * 2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 0, 1)) * 2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 0, 1)) * -2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0)) * 2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * 2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * -2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 1, 0)) * -2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0)) * -2.0
    l3new_aaaaaa += einsum(x599, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0)) * 2.0
    del x599
    x600 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x600 += einsum(l2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    x601 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x601 += einsum(v.aaaa.ovov, (0, 1, 2, 3), x326, (4, 5, 2, 6), (4, 5, 0, 6, 3, 1)) * 6.0
    del x326
    x602 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x602 += einsum(x600, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    del x600
    x602 += einsum(x601, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x601
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2))
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 0, 2)) * -1.0
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2)) * -1.0
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2))
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 0, 2)) * -1.0
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0)) * -1.0
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 2, 0))
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0))
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -1.0
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * -1.0
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0))
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 1, 0))
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (3, 5, 4, 2, 1, 0)) * -1.0
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 1, 0)) * -1.0
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0))
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0))
    l3new_aaaaaa += einsum(x602, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0)) * -1.0
    del x602
    x603 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x603 += einsum(f.aa.vv, (0, 1), l3.aaaaaa, (2, 3, 1, 4, 5, 6), (4, 5, 6, 0, 2, 3))
    x604 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x604 += einsum(f.aa.ov, (0, 1), x193, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    x605 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x605 += einsum(v.aaaa.vvvv, (0, 1, 2, 3), l3.aaaaaa, (4, 1, 3, 5, 6, 7), (5, 6, 7, 4, 0, 2))
    x606 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x606 += einsum(x603, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    del x603
    x606 += einsum(x604, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    del x604
    x606 += einsum(x605, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    del x605
    l3new_aaaaaa += einsum(x606, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0))
    l3new_aaaaaa += einsum(x606, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0))
    l3new_aaaaaa += einsum(x606, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * -1.0
    del x606
    x607 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x607 += einsum(x462, (0, 1, 2, 3), l3.aaaaaa, (4, 5, 6, 7, 0, 2), (7, 1, 3, 4, 5, 6)) * -6.0
    del x462
    x608 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x608 += einsum(f.aa.oo, (0, 1), (0, 1))
    x608 += einsum(x389, (0, 1), (1, 0))
    del x389
    x609 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x609 += einsum(x608, (0, 1), l3.aaaaaa, (2, 3, 4, 5, 6, 0), (5, 6, 1, 2, 3, 4)) * 6.0
    x610 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x610 += einsum(x607, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x607
    x610 += einsum(x609, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    del x609
    l3new_aaaaaa += einsum(x610, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2))
    l3new_aaaaaa += einsum(x610, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2)) * -1.0
    l3new_aaaaaa += einsum(x610, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0))
    del x610
    x611 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x611 += einsum(t2.aaaa, (0, 1, 2, 3), l3.aaaaaa, (4, 2, 3, 5, 6, 7), (5, 6, 7, 0, 1, 4))
    x612 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x612 += einsum(x611, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x611
    x612 += einsum(x325, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x325
    x613 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x613 += einsum(v.aaaa.ovov, (0, 1, 2, 3), x612, (4, 5, 6, 0, 2, 7), (4, 5, 6, 7, 1, 3)) * 6.0
    del x612
    l3new_aaaaaa += einsum(x613, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 2, 0)) * -1.0
    l3new_aaaaaa += einsum(x613, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0))
    l3new_aaaaaa += einsum(x613, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * -1.0
    del x613
    x614 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x614 += einsum(l2.aaaa, (0, 1, 2, 3), v.aaaa.ooov, (4, 3, 5, 6), (2, 4, 5, 0, 1, 6))
    x615 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x615 += einsum(v.aaaa.ovov, (0, 1, 2, 3), x410, (4, 5, 6, 1), (4, 0, 2, 5, 6, 3)) * 2.0
    x616 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x616 += einsum(x614, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    del x614
    x616 += einsum(x615, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    del x615
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (4, 3, 5, 0, 1, 2))
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2)) * -1.0
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 1, 2)) * -1.0
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (4, 3, 5, 0, 2, 1)) * -1.0
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1))
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 2, 1))
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2)) * -1.0
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2))
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 0, 1))
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 0, 1)) * -1.0
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 0, 1)) * -1.0
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0))
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * -1.0
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -1.0
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 1, 0)) * -1.0
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0))
    l3new_aaaaaa += einsum(x616, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0))
    del x616
    x617 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x617 += einsum(x454, (0, 1), l3.aaaaaa, (2, 3, 1, 4, 5, 6), (4, 5, 6, 2, 3, 0)) * 6.0
    x618 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x618 += einsum(x361, (0, 1), x193, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 6, 1)) * 6.0
    x619 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x619 += einsum(x617, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    del x617
    x619 += einsum(x618, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x618
    l3new_aaaaaa += einsum(x619, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0))
    l3new_aaaaaa += einsum(x619, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * -1.0
    l3new_aaaaaa += einsum(x619, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -1.0
    del x619
    x620 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x620 += einsum(x413, (0, 1), l3.aaaaaa, (2, 3, 4, 5, 6, 0), (5, 6, 1, 2, 3, 4)) * 6.0
    l3new_aaaaaa += einsum(x620, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    l3new_aaaaaa += einsum(x620, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -1.0
    l3new_aaaaaa += einsum(x620, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0))
    del x620
    x621 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x621 += einsum(x253, (0, 1, 2, 3), l3.aaaaaa, (4, 5, 6, 7, 0, 1), (7, 2, 3, 4, 5, 6))
    l3new_aaaaaa += einsum(x621, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2)) * -6.0
    l3new_aaaaaa += einsum(x621, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1)) * 6.0
    l3new_aaaaaa += einsum(x621, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2)) * 6.0
    l3new_aaaaaa += einsum(x621, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 0, 1)) * -6.0
    l3new_aaaaaa += einsum(x621, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0)) * -6.0
    l3new_aaaaaa += einsum(x621, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 1, 0)) * 6.0
    del x621
    x622 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x622 += einsum(x239, (0, 1, 2, 3), l3.abaaba, (4, 1, 5, 6, 0, 7), (6, 7, 2, 4, 5, 3))
    x623 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x623 += einsum(x5, (0, 1, 2, 3), x91, (0, 1, 4, 5, 6, 7), (4, 5, 2, 6, 7, 3))
    x624 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x624 += einsum(x622, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 4.00000000000004
    del x622
    x624 += einsum(x623, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    del x623
    l3new_aaaaaa += einsum(x624, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2)) * -1.0
    l3new_aaaaaa += einsum(x624, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2)) * -1.0
    l3new_aaaaaa += einsum(x624, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 0, 2))
    l3new_aaaaaa += einsum(x624, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0))
    l3new_aaaaaa += einsum(x624, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0))
    l3new_aaaaaa += einsum(x624, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * -1.0
    l3new_aaaaaa += einsum(x624, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 1, 0)) * -1.0
    l3new_aaaaaa += einsum(x624, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0)) * -1.0
    l3new_aaaaaa += einsum(x624, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0))
    del x624
    x625 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x625 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x193, (4, 5, 6, 0, 7, 3), (4, 5, 6, 7, 1, 2)) * -1.0
    del x193
    l3new_aaaaaa += einsum(x625, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0)) * -6.0
    l3new_aaaaaa += einsum(x625, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 2, 0)) * 6.0
    l3new_aaaaaa += einsum(x625, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0)) * 6.0
    l3new_aaaaaa += einsum(x625, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -6.0
    l3new_aaaaaa += einsum(x625, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * -6.0
    l3new_aaaaaa += einsum(x625, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * 6.0
    del x625
    x626 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x626 += einsum(l2.bbbb, (0, 1, 2, 3), v.aabb.ovoo, (4, 5, 6, 3), (2, 6, 0, 1, 4, 5))
    x627 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x627 += einsum(v.aabb.ovov, (0, 1, 2, 3), x116, (4, 5, 6, 3), (4, 2, 5, 6, 0, 1))
    del x116
    x628 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x628 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x628 += einsum(x22, (0, 1, 2, 3), (0, 1, 3, 2))
    x628 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.00000000000001
    x629 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x629 += einsum(x628, (0, 1, 2, 3), l3.babbab, (4, 2, 5, 6, 7, 0), (6, 1, 4, 5, 7, 3)) * -2.0
    del x628
    x630 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x630 += einsum(x506, (0, 1, 2, 3), l3.babbab, (4, 5, 6, 7, 2, 0), (7, 1, 4, 6, 3, 5)) * -2.0
    x631 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x631 += einsum(x6, (0, 1, 2, 3), x58, (4, 0, 5, 6, 7, 2), (4, 1, 5, 6, 7, 3)) * 2.0
    x632 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x632 += einsum(t1.bb, (0, 1), x19, (2, 1), (0, 2))
    del x19
    x633 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x633 += einsum(x219, (0, 1), (1, 0))
    del x219
    x633 += einsum(x221, (0, 1), (0, 1))
    del x221
    x633 += einsum(x632, (0, 1), (0, 1))
    del x632
    x633 += einsum(x223, (0, 1), (1, 0)) * -1.0
    del x223
    x634 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x634 += einsum(x633, (0, 1), l3.babbab, (2, 3, 4, 5, 6, 0), (5, 1, 2, 4, 6, 3)) * -2.0
    del x633
    x635 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x635 += einsum(f.bb.oo, (0, 1), (0, 1))
    x635 += einsum(x540, (0, 1), (1, 0))
    del x540
    x636 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x636 += einsum(x635, (0, 1), l3.babbab, (2, 3, 4, 5, 6, 0), (5, 1, 2, 4, 6, 3)) * -2.0
    x637 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x637 += einsum(x626, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * 2.0
    del x626
    x637 += einsum(x627, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * 2.0
    del x627
    x637 += einsum(x629, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x629
    x637 += einsum(x630, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x630
    x637 += einsum(x631, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x631
    x637 += einsum(x634, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x634
    x637 += einsum(x636, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x636
    l3new_babbab += einsum(x637, (0, 1, 2, 3, 4, 5), (3, 5, 2, 0, 4, 1)) * -1.0
    l3new_babbab += einsum(x637, (0, 1, 2, 3, 4, 5), (3, 5, 2, 1, 4, 0))
    del x637
    x638 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x638 += einsum(l2.abab, (0, 1, 2, 3), v.bbbb.ovvv, (4, 5, 6, 1), (3, 4, 5, 6, 2, 0))
    x639 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x639 += einsum(l2.abab, (0, 1, 2, 3), v.aabb.vvov, (4, 0, 5, 6), (3, 5, 1, 6, 2, 4))
    x640 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x640 += einsum(v.aabb.ooov, (0, 1, 2, 3), x61, (4, 5, 6, 0, 1, 7), (4, 2, 5, 3, 6, 7))
    x641 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x641 += einsum(x7, (0, 1, 2, 3), x61, (4, 5, 2, 6, 3, 7), (4, 0, 5, 1, 6, 7)) * -1.0
    x642 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x642 += einsum(t2.bbbb, (0, 1, 2, 3), x10, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.00000000000002
    x643 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x643 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x643 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x643 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.00000000000001
    x643 += einsum(x642, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x642
    x643 += einsum(x13, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    l3new_abaaba += einsum(x643, (0, 1, 2, 3), l3.abaaba, (4, 2, 5, 6, 0, 7), (4, 3, 5, 6, 1, 7)) * 2.0
    x644 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x644 += einsum(x643, (0, 1, 2, 3), l3.babbab, (4, 5, 2, 6, 7, 0), (6, 1, 4, 3, 7, 5)) * 2.0
    del x643
    x645 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x645 += einsum(t2.abab, (0, 1, 2, 3), x10, (1, 4, 3, 5), (4, 5, 0, 2)) * 1.00000000000001
    x646 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x646 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x646 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    x646 += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.00000000000002
    x646 += einsum(x645, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x647 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x647 += einsum(x646, (0, 1, 2, 3), l3.abaaba, (4, 5, 3, 6, 7, 2), (7, 0, 5, 1, 6, 4)) * 2.0
    x648 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x648 += einsum(x516, (0, 1, 2, 3), x74, (4, 0, 1, 5, 6, 7), (4, 2, 5, 3, 6, 7)) * 2.0
    del x516
    x649 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x649 += einsum(x213, (0, 1, 2, 3), x74, (4, 0, 2, 5, 6, 7), (4, 1, 5, 3, 6, 7)) * 2.0
    del x213
    x650 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x650 += einsum(v.bbbb.ovov, (0, 1, 2, 3), x297, (4, 1, 5, 6), (0, 2, 3, 4, 5, 6)) * 2.0
    x651 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x651 += einsum(v.aabb.ovov, (0, 1, 2, 3), x100, (4, 5, 6, 1), (2, 4, 3, 5, 0, 6)) * 2.0
    x652 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x652 += einsum(v.bbbb.ovov, (0, 1, 2, 3), x96, (4, 2, 5, 6), (0, 4, 3, 1, 5, 6))
    del x96
    x653 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x653 += einsum(v.aabb.ovov, (0, 1, 2, 3), x66, (4, 5, 6, 0), (2, 4, 3, 5, 6, 1)) * 2.0
    x654 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x654 += einsum(l2.abab, (0, 1, 2, 3), x537, (3, 4, 5, 6), (4, 5, 1, 6, 2, 0))
    del x537
    x655 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x655 += einsum(l2.abab, (0, 1, 2, 3), x31, (4, 5, 2, 6), (3, 4, 1, 5, 6, 0))
    x656 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x656 += einsum(l1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 5), (1, 4, 0, 5, 2, 3))
    x656 += einsum(x638, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x638
    x656 += einsum(x639, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x639
    x656 += einsum(x640, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    del x640
    x656 += einsum(x641, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    del x641
    x656 += einsum(x644, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x644
    x656 += einsum(x647, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x647
    x656 += einsum(x648, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x648
    x656 += einsum(x649, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x649
    x656 += einsum(x650, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x650
    x656 += einsum(x651, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x651
    x656 += einsum(x652, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5))
    del x652
    x656 += einsum(x653, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x653
    x656 += einsum(x654, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x654
    x656 += einsum(x655, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x655
    l3new_babbab += einsum(x656, (0, 1, 2, 3, 4, 5), (2, 5, 3, 0, 4, 1))
    l3new_babbab += einsum(x656, (0, 1, 2, 3, 4, 5), (3, 5, 2, 0, 4, 1)) * -1.0
    l3new_babbab += einsum(x656, (0, 1, 2, 3, 4, 5), (2, 5, 3, 1, 4, 0)) * -1.0
    l3new_babbab += einsum(x656, (0, 1, 2, 3, 4, 5), (3, 5, 2, 1, 4, 0))
    del x656
    x657 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x657 += einsum(l2.bbbb, (0, 1, 2, 3), v.aabb.ovvv, (4, 5, 6, 1), (2, 3, 0, 6, 4, 5))
    x658 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x658 += einsum(f.bb.vv, (0, 1), l3.babbab, (2, 3, 1, 4, 5, 6), (4, 6, 0, 2, 5, 3))
    x659 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x659 += einsum(f.bb.ov, (0, 1), x74, (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6))
    x660 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x660 += einsum(v.aabb.vvvv, (0, 1, 2, 3), l3.babbab, (4, 1, 3, 5, 6, 7), (5, 7, 4, 2, 6, 0))
    x661 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x661 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x74, (4, 5, 0, 3, 6, 7), (4, 5, 1, 2, 6, 7)) * -1.0
    x662 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x662 += einsum(v.aabb.vvov, (0, 1, 2, 3), x74, (4, 5, 2, 6, 7, 1), (4, 5, 6, 3, 7, 0))
    x663 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x663 += einsum(v.aabb.ovvv, (0, 1, 2, 3), x58, (4, 5, 6, 3, 7, 0), (4, 5, 6, 2, 7, 1))
    x664 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x664 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1)) * 0.99999999999999
    x664 += einsum(x246, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.99999999999999
    del x246
    x664 += einsum(x247, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x247
    x665 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x665 += einsum(x664, (0, 1, 2, 3), l3.babbab, (4, 5, 0, 6, 2, 7), (6, 7, 4, 1, 3, 5)) * -2.00000000000002
    x666 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x666 += einsum(x579, (0, 1), l3.babbab, (2, 3, 1, 4, 5, 6), (4, 6, 2, 0, 5, 3)) * -2.0
    x667 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x667 += einsum(x31, (0, 1, 2, 3), x74, (4, 5, 0, 6, 2, 7), (4, 5, 6, 1, 3, 7)) * 2.0
    x668 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x668 += einsum(t2.abab, (0, 1, 2, 3), l3.babbab, (4, 2, 3, 5, 6, 7), (5, 7, 1, 4, 6, 0))
    x669 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x669 += einsum(x668, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x668
    x669 += einsum(x98, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x98
    x670 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x670 += einsum(v.aabb.ovov, (0, 1, 2, 3), x669, (4, 5, 2, 6, 7, 0), (4, 5, 3, 6, 7, 1)) * 2.0
    del x669
    x671 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x671 += einsum(x227, (0, 1), x74, (2, 3, 0, 4, 5, 6), (2, 3, 4, 1, 5, 6)) * 2.0
    x672 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x672 += einsum(v.aabb.ovov, (0, 1, 2, 3), x78, (4, 5, 2, 6), (4, 5, 3, 6, 0, 1)) * 2.0
    del x78
    x673 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x673 += einsum(l1.aa, (0, 1), v.bbbb.ovov, (2, 3, 4, 5), (2, 4, 3, 5, 1, 0)) * -1.0
    x673 += einsum(x657, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * 2.0
    del x657
    x673 += einsum(x658, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -2.0
    del x658
    x673 += einsum(x659, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * 2.0
    del x659
    x673 += einsum(x660, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * 2.0
    del x660
    x673 += einsum(x661, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -2.0
    del x661
    x673 += einsum(x662, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -2.0
    del x662
    x673 += einsum(x663, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -2.0
    del x663
    x673 += einsum(x665, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x665
    x673 += einsum(x666, (0, 1, 2, 3, 4, 5), (1, 0, 3, 2, 4, 5)) * -1.0
    del x666
    x673 += einsum(x667, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x667
    x673 += einsum(x670, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x670
    x673 += einsum(x671, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x671
    x673 += einsum(x672, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    del x672
    l3new_babbab += einsum(x673, (0, 1, 2, 3, 4, 5), (3, 5, 2, 0, 4, 1))
    l3new_babbab += einsum(x673, (0, 1, 2, 3, 4, 5), (2, 5, 3, 0, 4, 1)) * -1.0
    del x673
    x674 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x674 += einsum(l2.bbbb, (0, 1, 2, 3), x5, (3, 4, 5, 6), (2, 4, 0, 1, 5, 6))
    x675 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x675 += einsum(v.aabb.ovov, (0, 1, 2, 3), x0, (4, 5, 6, 3), (4, 2, 5, 6, 0, 1)) * -1.0
    del x0
    x676 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x676 += einsum(t1.bb, (0, 1), x20, (2, 1), (0, 2)) * -1.0
    del x20
    x677 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x677 += einsum(x220, (0, 1), (0, 1)) * 2.0
    del x220
    x677 += einsum(x676, (0, 1), (0, 1))
    del x676
    x678 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x678 += einsum(x677, (0, 1), l3.babbab, (2, 3, 4, 5, 6, 0), (5, 1, 2, 4, 6, 3)) * -2.0
    del x677
    x679 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x679 += einsum(x674, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    del x674
    x679 += einsum(x675, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * 6.0
    del x675
    x679 += einsum(x678, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x678
    l3new_babbab += einsum(x679, (0, 1, 2, 3, 4, 5), (3, 5, 2, 0, 4, 1))
    l3new_babbab += einsum(x679, (0, 1, 2, 3, 4, 5), (3, 5, 2, 1, 4, 0)) * -1.0
    del x679
    x680 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x680 += einsum(t2.aaaa, (0, 1, 2, 3), x233, (1, 4, 5, 3), (0, 4, 2, 5)) * 2.00000000000002
    del x233
    x681 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x681 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x681 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x681 += einsum(x232, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.00000000000001
    del x232
    x681 += einsum(x680, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x680
    x681 += einsum(x236, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x236
    l3new_babbab += einsum(x681, (0, 1, 2, 3), l3.babbab, (4, 2, 5, 6, 0, 7), (4, 3, 5, 6, 1, 7)) * 2.0
    x682 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x682 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1)) * 0.4999999999999949
    x682 += einsum(x238, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.4999999999999949
    x682 += einsum(x239, (0, 1, 2, 3), (0, 1, 2, 3))
    x682 += einsum(t2.abab, (0, 1, 2, 3), x240, (0, 4, 2, 5), (1, 3, 4, 5)) * -0.49999999999999983
    del x240
    l3new_babbab += einsum(x682, (0, 1, 2, 3), l3.bbbbbb, (4, 5, 1, 6, 7, 0), (4, 3, 5, 6, 2, 7)) * 12.000000000000123
    del x682
    x683 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x683 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x683 += einsum(x25, (0, 1, 2, 3), (1, 3, 0, 2))
    del x25
    x683 += einsum(x26, (0, 1, 2, 3), (0, 2, 1, 3))
    del x26
    x683 += einsum(x27, (0, 1, 2, 3), (2, 1, 0, 3))
    x683 += einsum(x27, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    l3new_babbab += einsum(x683, (0, 1, 2, 3), l3.babbab, (4, 5, 6, 2, 7, 0), (4, 5, 6, 1, 7, 3)) * -2.0
    del x683
    x684 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x684 += einsum(t2.bbbb, (0, 1, 2, 3), l3.babbab, (2, 4, 3, 5, 6, 7), (5, 7, 0, 1, 6, 4))
    x684 += einsum(x134, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x134
    l3new_babbab += einsum(v.bbbb.ovov, (0, 1, 2, 3), x684, (4, 5, 2, 0, 6, 7), (1, 7, 3, 5, 6, 4)) * 2.0
    del x684
    x685 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x685 += einsum(l2.aaaa, (0, 1, 2, 3), v.aabb.vvov, (4, 1, 5, 6), (5, 6, 2, 3, 0, 4))
    x686 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x686 += einsum(f.aa.vv, (0, 1), l3.abaaba, (2, 3, 1, 4, 5, 6), (5, 3, 4, 6, 0, 2))
    x687 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x687 += einsum(f.aa.ov, (0, 1), x61, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    x688 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x688 += einsum(v.aabb.vvvv, (0, 1, 2, 3), l3.abaaba, (4, 3, 1, 5, 6, 7), (6, 2, 5, 7, 4, 0))
    x689 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x689 += einsum(v.aabb.vvov, (0, 1, 2, 3), x91, (4, 2, 5, 6, 7, 1), (4, 3, 5, 6, 7, 0))
    x690 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x690 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x61, (4, 5, 6, 7, 0, 3), (4, 5, 6, 7, 1, 2)) * -1.0
    x691 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x691 += einsum(v.aabb.ovvv, (0, 1, 2, 3), x61, (4, 3, 5, 6, 0, 7), (4, 2, 5, 6, 7, 1))
    x692 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x692 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1)) * 0.99999999999999
    x692 += einsum(x22, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.99999999999999
    del x22
    x692 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x23
    x693 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x693 += einsum(x692, (0, 1, 2, 3), l3.abaaba, (4, 5, 2, 6, 0, 7), (1, 5, 6, 7, 4, 3)) * -2.00000000000002
    del x692
    x694 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x694 += einsum(x454, (0, 1), l3.abaaba, (2, 3, 1, 4, 5, 6), (5, 3, 4, 6, 2, 0)) * -2.0
    del x454
    x695 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x695 += einsum(x6, (0, 1, 2, 3), x61, (0, 4, 5, 6, 2, 7), (1, 4, 5, 6, 7, 3)) * 2.0
    x696 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x696 += einsum(t2.abab, (0, 1, 2, 3), l3.abaaba, (4, 3, 2, 5, 6, 7), (6, 1, 5, 7, 0, 4))
    x697 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x697 += einsum(x696, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x696
    x697 += einsum(x99, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x99
    x698 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x698 += einsum(v.aabb.ovov, (0, 1, 2, 3), x697, (4, 2, 5, 6, 0, 7), (4, 3, 5, 6, 1, 7)) * 2.0
    del x697
    x699 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x699 += einsum(x361, (0, 1), x61, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 6, 1)) * 2.0
    del x361
    x700 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x700 += einsum(v.aabb.ovov, (0, 1, 2, 3), x289, (4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * 2.0
    del x289
    x701 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x701 += einsum(l1.bb, (0, 1), v.aaaa.ovov, (2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    x701 += einsum(x685, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -2.0
    del x685
    x701 += einsum(x686, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * 2.0
    del x686
    x701 += einsum(x687, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -2.0
    del x687
    x701 += einsum(x688, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -2.0
    del x688
    x701 += einsum(x689, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * 2.0
    del x689
    x701 += einsum(x690, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * 2.0
    del x690
    x701 += einsum(x691, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * 2.0
    del x691
    x701 += einsum(x693, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x693
    x701 += einsum(x694, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x694
    x701 += einsum(x695, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x695
    x701 += einsum(x698, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x698
    x701 += einsum(x699, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x699
    x701 += einsum(x700, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0
    del x700
    l3new_abaaba += einsum(x701, (0, 1, 2, 3, 4, 5), (5, 1, 4, 2, 0, 3))
    l3new_abaaba += einsum(x701, (0, 1, 2, 3, 4, 5), (4, 1, 5, 2, 0, 3)) * -1.0
    del x701
    x702 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x702 += einsum(l2.aaaa, (0, 1, 2, 3), v.aabb.ooov, (4, 3, 5, 6), (5, 6, 2, 4, 0, 1))
    x703 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x703 += einsum(x664, (0, 1, 2, 3), l3.abaaba, (4, 0, 5, 6, 7, 2), (7, 1, 6, 3, 4, 5)) * -2.00000000000002
    del x664
    x704 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x704 += einsum(x506, (0, 1, 2, 3), l3.abaaba, (4, 5, 6, 7, 0, 2), (1, 5, 7, 3, 4, 6)) * -2.0
    del x506
    x705 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x705 += einsum(x31, (0, 1, 2, 3), x91, (4, 0, 2, 5, 6, 7), (4, 1, 5, 3, 6, 7)) * 2.0
    del x31, x91
    x706 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x706 += einsum(x413, (0, 1), l3.abaaba, (2, 3, 4, 5, 6, 0), (6, 3, 5, 1, 2, 4)) * -2.0
    del x413
    x707 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x707 += einsum(v.aabb.ovov, (0, 1, 2, 3), x410, (4, 5, 6, 1), (2, 3, 0, 4, 5, 6)) * 2.0
    del x410
    x708 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x708 += einsum(x608, (0, 1), l3.abaaba, (2, 3, 4, 5, 6, 0), (6, 3, 5, 1, 2, 4)) * -2.0
    del x608
    x709 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x709 += einsum(x702, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    del x702
    x709 += einsum(x703, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x703
    x709 += einsum(x704, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x704
    x709 += einsum(x705, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x705
    x709 += einsum(x706, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x706
    x709 += einsum(x707, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x707
    x709 += einsum(x708, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x708
    l3new_abaaba += einsum(x709, (0, 1, 2, 3, 4, 5), (5, 1, 4, 2, 0, 3)) * -1.0
    l3new_abaaba += einsum(x709, (0, 1, 2, 3, 4, 5), (5, 1, 4, 3, 0, 2))
    del x709
    x710 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x710 += einsum(l2.abab, (0, 1, 2, 3), v.aabb.ovvv, (4, 5, 6, 1), (3, 6, 2, 4, 0, 5))
    x711 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x711 += einsum(l2.abab, (0, 1, 2, 3), v.aaaa.ovvv, (4, 5, 6, 0), (3, 1, 2, 4, 5, 6))
    x712 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x712 += einsum(v.aabb.ovoo, (0, 1, 2, 3), x74, (4, 2, 3, 5, 6, 7), (4, 5, 6, 0, 7, 1))
    x713 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x713 += einsum(x5, (0, 1, 2, 3), x74, (0, 4, 1, 5, 6, 7), (4, 5, 6, 2, 7, 3)) * -1.0
    del x5, x74
    x714 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x714 += einsum(x681, (0, 1, 2, 3), l3.abaaba, (4, 5, 2, 6, 7, 0), (7, 5, 6, 1, 4, 3)) * 2.0
    del x681
    x715 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x715 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x715 += einsum(x238, (0, 1, 2, 3), (0, 1, 2, 3))
    del x238
    x715 += einsum(x239, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.00000000000002
    del x239
    x715 += einsum(x592, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x592
    x716 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x716 += einsum(x715, (0, 1, 2, 3), l3.babbab, (4, 5, 1, 6, 7, 0), (6, 4, 7, 2, 5, 3)) * 2.0
    del x715
    x717 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x717 += einsum(x382, (0, 1, 2, 3), x61, (4, 5, 0, 6, 2, 7), (4, 5, 6, 1, 7, 3)) * 2.0
    del x382
    x718 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x718 += einsum(x348, (0, 1, 2, 3), x61, (4, 5, 0, 6, 2, 7), (4, 5, 6, 1, 7, 3)) * 2.0
    del x61, x348
    x719 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x719 += einsum(v.aabb.ovov, (0, 1, 2, 3), x297, (4, 3, 5, 6), (2, 4, 0, 5, 1, 6)) * 2.0
    del x297
    x720 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x720 += einsum(v.aaaa.ovov, (0, 1, 2, 3), x100, (4, 5, 6, 1), (4, 5, 0, 2, 6, 3)) * 2.0
    del x100
    x721 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x721 += einsum(v.aabb.ovov, (0, 1, 2, 3), x139, (4, 2, 5, 6), (4, 3, 0, 5, 1, 6)) * 2.0
    del x139
    x722 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x722 += einsum(v.aaaa.ovov, (0, 1, 2, 3), x66, (4, 5, 6, 2), (4, 5, 6, 0, 3, 1)) * 2.0
    del x66
    x723 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x723 += einsum(l2.abab, (0, 1, 2, 3), x6, (3, 4, 5, 6), (4, 1, 2, 5, 0, 6))
    del x6
    x724 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x724 += einsum(l2.abab, (0, 1, 2, 3), x386, (2, 4, 5, 6), (3, 1, 4, 5, 0, 6))
    del x386
    x725 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x725 += einsum(l1.aa, (0, 1), v.aabb.ovov, (2, 3, 4, 5), (4, 5, 1, 2, 0, 3))
    x725 += einsum(x710, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x710
    x725 += einsum(x711, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x711
    x725 += einsum(x712, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    del x712
    x725 += einsum(x713, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    del x713
    x725 += einsum(x714, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x714
    x725 += einsum(x716, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x716
    x725 += einsum(x717, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x717
    x725 += einsum(x718, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x718
    x725 += einsum(x719, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x719
    x725 += einsum(x720, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5))
    del x720
    x725 += einsum(x721, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * -1.0
    del x721
    x725 += einsum(x722, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x722
    x725 += einsum(x723, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x723
    x725 += einsum(x724, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x724
    l3new_abaaba += einsum(x725, (0, 1, 2, 3, 4, 5), (4, 1, 5, 2, 0, 3))
    l3new_abaaba += einsum(x725, (0, 1, 2, 3, 4, 5), (5, 1, 4, 2, 0, 3)) * -1.0
    l3new_abaaba += einsum(x725, (0, 1, 2, 3, 4, 5), (4, 1, 5, 3, 0, 2)) * -1.0
    l3new_abaaba += einsum(x725, (0, 1, 2, 3, 4, 5), (5, 1, 4, 3, 0, 2))
    del x725
    x726 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x726 += einsum(l2.aaaa, (0, 1, 2, 3), x7, (4, 5, 3, 6), (4, 5, 2, 6, 0, 1))
    l3new_abaaba += einsum(x726, (0, 1, 2, 3, 4, 5), (5, 1, 4, 2, 0, 3)) * 2.0
    l3new_abaaba += einsum(x726, (0, 1, 2, 3, 4, 5), (5, 1, 4, 3, 0, 2)) * -2.0
    del x726
    x727 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x727 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x727 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    del x15
    x727 += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0000000000000204
    del x16
    x727 += einsum(x645, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x645
    l3new_abaaba += einsum(x727, (0, 1, 2, 3), l3.aaaaaa, (4, 5, 3, 6, 7, 2), (4, 1, 5, 6, 0, 7)) * 6.0
    del x727
    x728 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x728 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x728 += einsum(x251, (0, 1, 2, 3), (1, 3, 2, 0))
    del x251
    x728 += einsum(x252, (0, 1, 2, 3), (0, 2, 3, 1))
    del x252
    x728 += einsum(x253, (0, 1, 2, 3), (2, 1, 3, 0))
    x728 += einsum(x253, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    del x253
    l3new_abaaba += einsum(x728, (0, 1, 2, 3), l3.abaaba, (4, 5, 6, 3, 7, 0), (4, 5, 6, 2, 7, 1)) * 2.0
    del x728
    x729 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x729 += einsum(t2.aaaa, (0, 1, 2, 3), l3.abaaba, (2, 4, 3, 5, 6, 7), (6, 4, 5, 7, 0, 1))
    x729 += einsum(x323, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4))
    del x323
    l3new_abaaba += einsum(v.aaaa.ovov, (0, 1, 2, 3), x729, (4, 5, 6, 7, 2, 0), (1, 5, 3, 7, 4, 6)) * 2.0
    del x729
    x730 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x730 += einsum(v.aabb.ooov, (0, 1, 2, 3), x58, (4, 5, 6, 7, 0, 1), (4, 5, 2, 6, 7, 3))
    x731 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x731 += einsum(x7, (0, 1, 2, 3), x58, (4, 5, 6, 7, 2, 3), (4, 5, 0, 6, 7, 1))
    del x7, x58
    x732 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x732 += einsum(t2.bbbb, (0, 1, 2, 3), x10, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0000000000000204
    del x10
    x733 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x733 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x733 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x733 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.00000000000001
    del x9
    x733 += einsum(x732, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x732
    x733 += einsum(x13, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x13
    x734 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x734 += einsum(x733, (0, 1, 2, 3), l3.bbbbbb, (4, 5, 2, 6, 7, 0), (6, 7, 1, 4, 5, 3)) * 6.0
    del x733
    x735 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x735 += einsum(x646, (0, 1, 2, 3), l3.babbab, (4, 3, 5, 6, 2, 7), (6, 7, 0, 4, 5, 1)) * 2.0
    del x646
    x736 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x736 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    x736 += einsum(x3, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x737 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x737 += einsum(x736, (0, 1, 2, 3), x72, (4, 5, 0, 2, 6, 7), (4, 5, 1, 6, 7, 3)) * 6.0
    del x736
    x738 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x738 += einsum(x208, (0, 1, 2, 3), x72, (4, 5, 0, 1, 6, 7), (4, 5, 2, 6, 7, 3)) * 6.0
    del x208
    x739 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x739 += einsum(x730, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    del x730
    x739 += einsum(x731, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    del x731
    x739 += einsum(x734, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    del x734
    x739 += einsum(x735, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    del x735
    x739 += einsum(x737, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x737
    x739 += einsum(x738, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x738
    l3new_bbbbbb += einsum(x739, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2))
    l3new_bbbbbb += einsum(x739, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2)) * -1.0
    l3new_bbbbbb += einsum(x739, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2)) * -1.0
    l3new_bbbbbb += einsum(x739, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0)) * -1.0
    l3new_bbbbbb += einsum(x739, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0))
    l3new_bbbbbb += einsum(x739, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0))
    l3new_bbbbbb += einsum(x739, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 1, 0))
    l3new_bbbbbb += einsum(x739, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0)) * -1.0
    l3new_bbbbbb += einsum(x739, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0)) * -1.0
    del x739
    x740 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x740 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x72, (4, 5, 6, 0, 7, 3), (4, 5, 6, 7, 1, 2)) * -1.0
    l3new_bbbbbb += einsum(x740, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0)) * -6.0
    l3new_bbbbbb += einsum(x740, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 2, 0)) * 6.0
    l3new_bbbbbb += einsum(x740, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0)) * 6.0
    l3new_bbbbbb += einsum(x740, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -6.0
    l3new_bbbbbb += einsum(x740, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * -6.0
    l3new_bbbbbb += einsum(x740, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * 6.0
    del x740
    x741 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x741 += einsum(x579, (0, 1), l3.bbbbbb, (2, 3, 1, 4, 5, 6), (4, 5, 6, 2, 3, 0)) * 6.0
    del x579
    x742 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x742 += einsum(x227, (0, 1), x72, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 6, 1)) * 6.0
    del x227
    x743 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x743 += einsum(x741, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    del x741
    x743 += einsum(x742, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x742
    l3new_bbbbbb += einsum(x743, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0))
    l3new_bbbbbb += einsum(x743, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * -1.0
    l3new_bbbbbb += einsum(x743, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -1.0
    del x743
    x744 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x744 += einsum(x586, (0, 1, 2, 3), l3.bbbbbb, (4, 5, 6, 7, 0, 3), (7, 1, 2, 4, 5, 6)) * -6.0
    del x586
    x745 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x745 += einsum(x635, (0, 1), l3.bbbbbb, (2, 3, 4, 5, 6, 0), (5, 6, 1, 2, 3, 4)) * 6.0
    del x635
    x746 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x746 += einsum(x744, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    del x744
    x746 += einsum(x745, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    del x745
    l3new_bbbbbb += einsum(x746, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2))
    l3new_bbbbbb += einsum(x746, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2)) * -1.0
    l3new_bbbbbb += einsum(x746, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0))
    del x746
    x747 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x747 += einsum(l2.bbbb, (0, 1, 2, 3), x3, (3, 4, 5, 6), (2, 4, 5, 0, 1, 6))
    del x3
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2)) * 2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2)) * 2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 1, 2)) * -2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1)) * -2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1)) * -2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (5, 4, 3, 0, 2, 1)) * 2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2)) * -2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2)) * -2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 0, 2)) * 2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 0, 1)) * 2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 0, 1)) * 2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 0, 1)) * -2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0)) * 2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * 2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * -2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 1, 0)) * -2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0)) * -2.0
    l3new_bbbbbb += einsum(x747, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0)) * 2.0
    del x747
    x748 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x748 += einsum(l2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    x749 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x749 += einsum(v.bbbb.ovov, (0, 1, 2, 3), x137, (4, 5, 2, 6), (0, 4, 5, 3, 1, 6)) * 2.0
    del x137
    x750 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x750 += einsum(x748, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    del x748
    x750 += einsum(x749, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    del x749
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2))
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 0, 2)) * -1.0
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2)) * -1.0
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2))
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 0, 2)) * -1.0
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0)) * -1.0
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (3, 5, 4, 1, 2, 0))
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0))
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -1.0
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * -1.0
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0))
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 1, 0))
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (3, 5, 4, 2, 1, 0)) * -1.0
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 1, 0)) * -1.0
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0))
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0))
    l3new_bbbbbb += einsum(x750, (0, 1, 2, 3, 4, 5), (5, 4, 3, 2, 1, 0)) * -1.0
    del x750
    x751 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x751 += einsum(f.bb.vv, (0, 1), l3.bbbbbb, (2, 3, 1, 4, 5, 6), (4, 5, 6, 0, 2, 3))
    x752 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x752 += einsum(f.bb.ov, (0, 1), x72, (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6))
    del x72
    x753 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x753 += einsum(v.bbbb.vvvv, (0, 1, 2, 3), l3.bbbbbb, (4, 3, 1, 5, 6, 7), (5, 6, 7, 4, 0, 2)) * -1.0
    x754 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x754 += einsum(x751, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    del x751
    x754 += einsum(x752, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    del x752
    x754 += einsum(x753, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    del x753
    l3new_bbbbbb += einsum(x754, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0))
    l3new_bbbbbb += einsum(x754, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0))
    l3new_bbbbbb += einsum(x754, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * -1.0
    del x754
    x755 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x755 += einsum(x27, (0, 1, 2, 3), l3.bbbbbb, (4, 5, 6, 7, 0, 1), (7, 2, 3, 4, 5, 6))
    del x27
    l3new_bbbbbb += einsum(x755, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 1, 2)) * -6.0
    l3new_bbbbbb += einsum(x755, (0, 1, 2, 3, 4, 5), (3, 4, 5, 0, 2, 1)) * 6.0
    l3new_bbbbbb += einsum(x755, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 0, 2)) * 6.0
    l3new_bbbbbb += einsum(x755, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 0, 1)) * -6.0
    l3new_bbbbbb += einsum(x755, (0, 1, 2, 3, 4, 5), (3, 4, 5, 1, 2, 0)) * -6.0
    l3new_bbbbbb += einsum(x755, (0, 1, 2, 3, 4, 5), (3, 4, 5, 2, 1, 0)) * 6.0
    del x755
    x756 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x756 += einsum(x560, (0, 1), l3.bbbbbb, (2, 3, 4, 5, 6, 0), (5, 6, 1, 2, 3, 4)) * 6.0
    del x560
    l3new_bbbbbb += einsum(x756, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    l3new_bbbbbb += einsum(x756, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -1.0
    l3new_bbbbbb += einsum(x756, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0))
    del x756
    x757 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x757 += einsum(l2.bbbb, (0, 1, 2, 3), v.bbbb.ooov, (4, 3, 5, 6), (2, 4, 5, 0, 1, 6))
    x758 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x758 += einsum(v.bbbb.ovov, (0, 1, 2, 3), x136, (4, 5, 6, 1), (0, 2, 4, 3, 5, 6)) * 6.0
    del x136
    x759 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x759 += einsum(x757, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    del x757
    x759 += einsum(x758, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    del x758
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (4, 3, 5, 0, 1, 2))
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 1, 2)) * -1.0
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 1, 2)) * -1.0
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (4, 3, 5, 0, 2, 1)) * -1.0
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (4, 5, 3, 0, 2, 1))
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (5, 3, 4, 0, 2, 1))
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 0, 2)) * -1.0
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 0, 2))
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 0, 2))
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 0, 1))
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 0, 1)) * -1.0
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 0, 1)) * -1.0
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0))
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0)) * -1.0
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (5, 3, 4, 1, 2, 0)) * -1.0
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (4, 3, 5, 2, 1, 0)) * -1.0
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (4, 5, 3, 2, 1, 0))
    l3new_bbbbbb += einsum(x759, (0, 1, 2, 3, 4, 5), (5, 3, 4, 2, 1, 0))
    del x759
    x760 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x760 += einsum(t2.bbbb, (0, 1, 2, 3), l3.bbbbbb, (4, 2, 3, 5, 6, 7), (5, 6, 7, 0, 1, 4))
    x761 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x761 += einsum(x760, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x760
    x761 += einsum(x132, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x132
    x762 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x762 += einsum(v.bbbb.ovov, (0, 1, 2, 3), x761, (4, 5, 6, 0, 2, 7), (4, 5, 6, 1, 3, 7)) * 6.0
    del x761
    l3new_bbbbbb += einsum(x762, (0, 1, 2, 3, 4, 5), (5, 4, 3, 1, 2, 0)) * -1.0
    l3new_bbbbbb += einsum(x762, (0, 1, 2, 3, 4, 5), (4, 5, 3, 1, 2, 0))
    l3new_bbbbbb += einsum(x762, (0, 1, 2, 3, 4, 5), (4, 3, 5, 1, 2, 0)) * -1.0
    del x762

    l1new.aa = l1new_aa
    l1new.bb = l1new_bb
    l2new.aaaa = l2new_aaaa
    l2new.abab = l2new_abab
    l2new.bbbb = l2new_bbbb
    l3new.aaaaaa = l3new_aaaaaa
    l3new.abaaba = l3new_abaaba
    l3new.babbab = l3new_babbab
    l3new.bbbbbb = l3new_bbbbbb

    return {"l1new": l1new, "l2new": l2new, "l3new": l3new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, l1=None, l2=None, l3=None, **kwargs):
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
    rdm1_f_aa_ov += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), (2, 3))
    rdm1_f_aa_ov += einsum(l2.abab, (0, 1, 2, 3), t3.abaaba, (4, 3, 2, 5, 1, 0), (4, 5)) * 2.0
    rdm1_f_aa_ov += einsum(l2.aaaa, (0, 1, 2, 3), t3.aaaaaa, (4, 2, 3, 5, 0, 1), (4, 5)) * 3.0
    rdm1_f_aa_ov += einsum(t1.aa, (0, 1), (0, 1))
    rdm1_f_aa_ov += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 0), (2, 3)) * 2.0
    rdm1_f_aa_ov += einsum(l2.bbbb, (0, 1, 2, 3), t3.babbab, (2, 4, 3, 0, 5, 1), (4, 5))
    rdm1_f_bb_ov = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    rdm1_f_bb_ov += einsum(t1.bb, (0, 1), (0, 1))
    rdm1_f_bb_ov += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 0), (2, 3)) * 2.0
    rdm1_f_bb_ov += einsum(l2.abab, (0, 1, 2, 3), t3.babbab, (4, 2, 3, 5, 0, 1), (4, 5)) * 2.0
    rdm1_f_bb_ov += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), (2, 3))
    rdm1_f_bb_ov += einsum(l2.bbbb, (0, 1, 2, 3), t3.bbbbbb, (4, 2, 3, 5, 0, 1), (4, 5)) * 3.0
    rdm1_f_bb_ov += einsum(l2.aaaa, (0, 1, 2, 3), t3.abaaba, (2, 4, 3, 0, 5, 1), (4, 5))
    rdm1_f_aa_vo = np.zeros((nvir[0], nocc[0]), dtype=types[float])
    rdm1_f_aa_vo += einsum(l1.aa, (0, 1), (0, 1))
    rdm1_f_bb_vo = np.zeros((nvir[1], nocc[1]), dtype=types[float])
    rdm1_f_bb_vo += einsum(l1.bb, (0, 1), (0, 1))
    rdm1_f_aa_vv = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    rdm1_f_aa_vv += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 4, 5, 0, 6, 2), (1, 6)) * 0.9999999999999601
    rdm1_f_aa_vv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    rdm1_f_aa_vv += einsum(l1.aa, (0, 1), t1.aa, (1, 2), (0, 2))
    rdm1_f_aa_vv += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 4, 5, 6, 1, 2), (0, 6)) * 1.9999999999999194
    rdm1_f_aa_vv += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (3, 4, 5, 6, 1, 2), (0, 6)) * 2.9999999999998788
    rdm1_f_aa_vv += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4)) * 2.0
    rdm1_f_bb_vv = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    rdm1_f_bb_vv += einsum(l1.bb, (0, 1), t1.bb, (1, 2), (0, 2))
    rdm1_f_bb_vv += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 4, 5, 6, 1, 2), (0, 6)) * 1.9999999999999194
    rdm1_f_bb_vv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    rdm1_f_bb_vv += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 4, 5, 0, 6, 2), (1, 6)) * 0.9999999999999601
    rdm1_f_bb_vv += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4)) * 2.0
    rdm1_f_bb_vv += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (3, 4, 5, 6, 1, 2), (0, 6)) * 2.9999999999998788
    x0 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x0 += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (6, 4, 5, 0, 1, 2), (3, 6))
    rdm1_f_aa_oo += einsum(x0, (0, 1), (1, 0)) * -2.9999999999998788
    x1 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x1 += einsum(l1.aa, (0, 1), t1.aa, (2, 0), (1, 2))
    rdm1_f_aa_oo += einsum(x1, (0, 1), (1, 0)) * -1.0
    x2 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x2 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    rdm1_f_aa_oo += einsum(x2, (0, 1), (1, 0)) * -1.0
    x3 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x3 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    rdm1_f_aa_oo += einsum(x3, (0, 1), (1, 0)) * -2.0
    x4 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 4, 5, 0, 1, 2), (3, 6))
    rdm1_f_aa_oo += einsum(x4, (0, 1), (1, 0)) * -1.9999999999999194
    x5 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x5 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 6, 5, 0, 1, 2), (4, 6))
    rdm1_f_aa_oo += einsum(x5, (0, 1), (1, 0)) * -0.9999999999999601
    x6 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x6 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 6, 5, 0, 1, 2), (4, 6))
    rdm1_f_bb_oo += einsum(x6, (0, 1), (1, 0)) * -0.9999999999999601
    x7 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x7 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    rdm1_f_bb_oo += einsum(x7, (0, 1), (1, 0)) * -2.0
    x8 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x8 += einsum(l1.bb, (0, 1), t1.bb, (2, 0), (1, 2))
    rdm1_f_bb_oo += einsum(x8, (0, 1), (1, 0)) * -1.0
    x9 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x9 += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (6, 4, 5, 0, 1, 2), (3, 6))
    rdm1_f_bb_oo += einsum(x9, (0, 1), (1, 0)) * -2.9999999999998788
    x10 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x10 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 4, 5, 0, 1, 2), (3, 6))
    rdm1_f_bb_oo += einsum(x10, (0, 1), (1, 0)) * -1.9999999999999194
    x11 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x11 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    rdm1_f_bb_oo += einsum(x11, (0, 1), (1, 0)) * -1.0
    x12 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum(t1.aa, (0, 1), l3.aaaaaa, (2, 3, 1, 4, 5, 6), (4, 5, 6, 0, 2, 3))
    rdm1_f_aa_ov += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x12, (0, 1, 2, 6, 4, 5), (6, 3)) * -2.9999999999998788
    del x12
    x13 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x13 += einsum(t1.aa, (0, 1), l3.babbab, (2, 1, 3, 4, 5, 6), (4, 6, 2, 3, 5, 0))
    rdm1_f_aa_ov += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x13, (0, 2, 5, 3, 1, 6), (6, 4)) * 0.9999999999999601
    del x13
    x14 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x14 += einsum(t1.aa, (0, 1), l3.abaaba, (2, 3, 1, 4, 5, 6), (5, 3, 4, 6, 0, 2))
    rdm1_f_aa_ov += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x14, (1, 4, 2, 0, 6, 5), (6, 3)) * -1.9999999999999194
    del x14
    x15 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x15 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    x15 += einsum(t2.abab, (0, 1, 2, 3), l3.abaaba, (4, 3, 2, 5, 1, 6), (5, 6, 0, 4))
    x15 += einsum(t2.aaaa, (0, 1, 2, 3), l3.aaaaaa, (4, 2, 3, 5, 6, 1), (5, 6, 0, 4)) * 3.0
    rdm1_f_aa_ov += einsum(t2.aaaa, (0, 1, 2, 3), x15, (0, 1, 4, 3), (4, 2)) * 2.0
    del x15
    x16 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x16 += einsum(t1.aa, (0, 1), l2.abab, (1, 2, 3, 4), (4, 2, 3, 0)) * 0.5
    x16 += einsum(t2.abab, (0, 1, 2, 3), l3.babbab, (4, 2, 3, 5, 6, 1), (5, 4, 6, 0))
    x16 += einsum(t2.aaaa, (0, 1, 2, 3), l3.abaaba, (2, 4, 3, 5, 6, 1), (6, 4, 5, 0))
    rdm1_f_aa_ov += einsum(t2.abab, (0, 1, 2, 3), x16, (1, 3, 0, 4), (4, 2)) * -2.0
    del x16
    x17 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x17 += einsum(x1, (0, 1), (0, 1)) * 0.3333333333333468
    del x1
    x17 += einsum(x2, (0, 1), (0, 1)) * 0.3333333333333468
    del x2
    x17 += einsum(x3, (0, 1), (0, 1)) * 0.6666666666666936
    del x3
    x17 += einsum(x5, (0, 1), (0, 1)) * 0.33333333333333354
    del x5
    x17 += einsum(x4, (0, 1), (0, 1)) * 0.6666666666666667
    del x4
    x17 += einsum(x0, (0, 1), (0, 1))
    del x0
    rdm1_f_aa_ov += einsum(t1.aa, (0, 1), x17, (0, 2), (2, 1)) * -2.9999999999998788
    del x17
    x18 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x18 += einsum(t1.bb, (0, 1), l3.abaaba, (2, 1, 3, 4, 5, 6), (5, 0, 4, 6, 2, 3))
    rdm1_f_bb_ov += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x18, (1, 6, 0, 2, 3, 5), (6, 4)) * -0.9999999999999601
    del x18
    x19 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x19 += einsum(t1.bb, (0, 1), l3.babbab, (2, 3, 1, 4, 5, 6), (4, 6, 0, 2, 5, 3))
    rdm1_f_bb_ov += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x19, (0, 2, 6, 5, 1, 4), (6, 3)) * 1.9999999999999194
    del x19
    x20 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x20 += einsum(t1.bb, (0, 1), l3.bbbbbb, (2, 3, 1, 4, 5, 6), (4, 5, 6, 0, 2, 3))
    rdm1_f_bb_ov += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x20, (0, 1, 2, 6, 5, 4), (6, 3)) * 2.9999999999998788
    del x20
    x21 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x21 += einsum(t1.bb, (0, 1), l2.abab, (2, 1, 3, 4), (4, 0, 3, 2))
    x21 += einsum(t2.bbbb, (0, 1, 2, 3), l3.babbab, (2, 4, 3, 5, 6, 1), (5, 0, 6, 4)) * 2.0
    x21 += einsum(t2.abab, (0, 1, 2, 3), l3.abaaba, (4, 3, 2, 5, 6, 0), (6, 1, 5, 4)) * 2.0
    rdm1_f_bb_ov += einsum(t2.abab, (0, 1, 2, 3), x21, (1, 4, 0, 2), (4, 3)) * -1.0
    del x21
    x22 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x22 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2)) * 0.3333333333333333
    x22 += einsum(t2.bbbb, (0, 1, 2, 3), l3.bbbbbb, (4, 2, 3, 5, 6, 1), (5, 6, 0, 4))
    x22 += einsum(t2.abab, (0, 1, 2, 3), l3.babbab, (4, 2, 3, 5, 0, 6), (5, 6, 1, 4)) * 0.3333333333333333
    rdm1_f_bb_ov += einsum(t2.bbbb, (0, 1, 2, 3), x22, (0, 1, 4, 3), (4, 2)) * 6.0
    del x22
    x23 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x23 += einsum(x8, (0, 1), (0, 1)) * 1.00000000000004
    del x8
    x23 += einsum(x7, (0, 1), (0, 1)) * 2.00000000000008
    del x7
    x23 += einsum(x11, (0, 1), (0, 1)) * 1.00000000000004
    del x11
    x23 += einsum(x9, (0, 1), (0, 1)) * 2.9999999999999982
    del x9
    x23 += einsum(x10, (0, 1), (0, 1)) * 1.9999999999999991
    del x10
    x23 += einsum(x6, (0, 1), (0, 1))
    del x6
    rdm1_f_bb_ov += einsum(t1.bb, (0, 1), x23, (0, 2), (2, 1)) * -0.9999999999999601
    del x23

    rdm1_f_aa = np.block([[rdm1_f_aa_oo, rdm1_f_aa_ov], [rdm1_f_aa_vo, rdm1_f_aa_vv]])
    rdm1_f_bb = np.block([[rdm1_f_bb_oo, rdm1_f_bb_ov], [rdm1_f_bb_vo, rdm1_f_bb_vv]])

    rdm1_f.aa = rdm1_f_aa
    rdm1_f.bb = rdm1_f_bb

    return rdm1_f

def make_rdm2_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, l1=None, l2=None, l3=None, **kwargs):
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
    rdm2_f_aaaa_oovv += einsum(l1.bb, (0, 1), t3.abaaba, (2, 1, 3, 4, 0, 5), (2, 3, 4, 5)) * 2.0
    rdm2_f_aaaa_oovv += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_aaaa_oovv += einsum(l1.aa, (0, 1), t3.aaaaaa, (2, 3, 1, 4, 5, 0), (2, 3, 4, 5)) * 6.0
    rdm2_f_abab_oovv = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_abab_oovv += einsum(l1.bb, (0, 1), t3.babbab, (2, 3, 1, 4, 5, 0), (3, 2, 5, 4)) * 2.0
    rdm2_f_abab_oovv += einsum(l1.aa, (0, 1), t3.abaaba, (2, 3, 1, 4, 5, 0), (2, 3, 4, 5)) * 2.0
    rdm2_f_bbbb_oovv = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_oovv += einsum(l1.bb, (0, 1), t3.bbbbbb, (2, 3, 1, 4, 5, 0), (2, 3, 4, 5)) * 6.0
    rdm2_f_bbbb_oovv += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_bbbb_oovv += einsum(l1.aa, (0, 1), t3.babbab, (2, 1, 3, 4, 0, 5), (2, 3, 4, 5)) * 2.0
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
    rdm2_f_abab_ovvv += einsum(l2.bbbb, (0, 1, 2, 3), t3.babbab, (2, 4, 3, 5, 6, 1), (4, 0, 6, 5)) * 2.0
    rdm2_f_abab_ovvv += einsum(l2.abab, (0, 1, 2, 3), t3.abaaba, (4, 3, 2, 5, 6, 0), (4, 1, 5, 6)) * 2.0
    rdm2_f_abab_vovv = np.zeros((nvir[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_vovv += einsum(l2.abab, (0, 1, 2, 3), t3.babbab, (4, 2, 3, 5, 6, 1), (0, 4, 6, 5)) * 2.0
    rdm2_f_abab_vovv += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 3, 4), (0, 2, 3, 4))
    rdm2_f_abab_vovv += einsum(l2.aaaa, (0, 1, 2, 3), t3.abaaba, (2, 4, 3, 5, 6, 1), (0, 4, 5, 6)) * 2.0
    rdm2_f_abab_vvov = np.zeros((nvir[0], nvir[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_vvov += einsum(t1.bb, (0, 1), l2.abab, (2, 3, 4, 0), (2, 3, 4, 1))
    rdm2_f_aaaa_vvvv = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_vvvv += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 4, 5, 6, 1, 7), (0, 2, 6, 7)) * 2.0000000000000404
    rdm2_f_aaaa_vvvv += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (3, 4, 5, 6, 7, 2), (0, 1, 6, 7)) * 6.000000000000116
    rdm2_f_aaaa_vvvv += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 5), (0, 1, 4, 5)) * 2.0
    rdm2_f_abab_vvvv = np.zeros((nvir[0], nvir[1], nvir[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_vvvv += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 4, 5, 6, 7, 2), (1, 0, 7, 6)) * 2.0000000000000404
    rdm2_f_abab_vvvv += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 4, 5, 6, 7, 2), (0, 1, 6, 7)) * 2.0000000000000404
    rdm2_f_abab_vvvv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), (0, 1, 4, 5))
    rdm2_f_bbbb_vvvv = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_vvvv += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 4, 5, 6, 1, 7), (0, 2, 6, 7)) * 2.0000000000000404
    rdm2_f_bbbb_vvvv += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 5), (0, 1, 4, 5)) * 2.0
    rdm2_f_bbbb_vvvv += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (3, 4, 5, 6, 7, 2), (0, 1, 6, 7)) * 6.000000000000116
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
    x3 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x3 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 4, 7, 0, 1, 2), (3, 5, 6, 7))
    rdm2_f_aaaa_oooo += einsum(x3, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0000000000000404
    x4 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (6, 7, 5, 0, 1, 2), (3, 4, 6, 7))
    rdm2_f_aaaa_oooo += einsum(x4, (0, 1, 2, 3), (2, 3, 1, 0)) * -6.000000000000116
    x5 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x5 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    x6 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x6 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    x7 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x7 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 6, 5, 0, 1, 2), (4, 6))
    x8 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x8 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 4, 5, 0, 1, 2), (3, 6))
    x9 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x9 += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (6, 4, 5, 0, 1, 2), (3, 6))
    x10 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x10 += einsum(x5, (0, 1), (0, 1))
    x10 += einsum(x6, (0, 1), (0, 1)) * 2.0
    x10 += einsum(x7, (0, 1), (0, 1)) * 0.9999999999999601
    x10 += einsum(x8, (0, 1), (0, 1)) * 1.9999999999999194
    x10 += einsum(x9, (0, 1), (0, 1)) * 2.9999999999998788
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x10, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x10, (2, 3), (0, 3, 2, 1))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x10, (2, 3), (3, 0, 1, 2))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x10, (2, 3), (3, 0, 2, 1)) * -1.0
    del x10
    x11 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x11 += einsum(l1.aa, (0, 1), t1.aa, (2, 0), (1, 2))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x11, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x11, (2, 3), (3, 0, 1, 2))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x11, (2, 3), (0, 3, 2, 1))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x11, (2, 3), (3, 0, 2, 1)) * -1.0
    x12 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x12 += einsum(t2.abab, (0, 1, 2, 3), l3.abaaba, (4, 3, 2, 5, 1, 6), (5, 6, 0, 4))
    rdm2_f_aaaa_ovoo += einsum(x12, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0
    rdm2_f_aaaa_vooo += einsum(x12, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x13 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x13 += einsum(t2.aaaa, (0, 1, 2, 3), l3.aaaaaa, (4, 2, 3, 5, 6, 1), (5, 6, 0, 4))
    rdm2_f_aaaa_ovoo += einsum(x13, (0, 1, 2, 3), (2, 3, 1, 0)) * -6.0
    rdm2_f_aaaa_vooo += einsum(x13, (0, 1, 2, 3), (3, 2, 1, 0)) * 6.0
    x14 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x14 += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3))
    x14 += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x15 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x15 += einsum(t1.aa, (0, 1), x14, (2, 3, 4, 1), (0, 2, 3, 4)) * 2.0
    rdm2_f_aaaa_oooo += einsum(x15, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    rdm2_f_aaaa_oooo += einsum(x15, (0, 1, 2, 3), (3, 0, 2, 1))
    del x15
    x16 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x16 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 7, 5, 0, 1, 2), (4, 7, 3, 6))
    rdm2_f_abab_oooo = np.zeros((nocc[0], nocc[1], nocc[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_oooo += einsum(x16, (0, 1, 2, 3), (3, 1, 2, 0)) * 2.0000000000000404
    x17 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x17 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 7, 5, 0, 1, 2), (3, 6, 4, 7))
    rdm2_f_abab_oooo += einsum(x17, (0, 1, 2, 3), (3, 1, 2, 0)) * 2.0000000000000404
    x18 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x18 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), (3, 5, 2, 4))
    rdm2_f_abab_oooo += einsum(x18, (0, 1, 2, 3), (3, 1, 2, 0))
    x19 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x19 += einsum(t1.aa, (0, 1), l2.abab, (1, 2, 3, 4), (4, 2, 3, 0))
    rdm2_f_abab_ovoo += einsum(x19, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    x20 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x20 += einsum(t2.abab, (0, 1, 2, 3), l3.babbab, (4, 2, 3, 5, 6, 1), (5, 4, 6, 0))
    rdm2_f_abab_ovoo += einsum(x20, (0, 1, 2, 3), (3, 1, 2, 0)) * -2.0
    x21 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x21 += einsum(t2.aaaa, (0, 1, 2, 3), l3.abaaba, (2, 4, 3, 5, 6, 1), (6, 4, 5, 0))
    rdm2_f_abab_ovoo += einsum(x21, (0, 1, 2, 3), (3, 1, 2, 0)) * -2.0
    x22 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x22 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x22 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3))
    x22 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_abab_oooo += einsum(t1.bb, (0, 1), x22, (2, 1, 3, 4), (4, 0, 3, 2)) * 2.0
    rdm2_f_abab_ooov = np.zeros((nocc[0], nocc[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_ooov += einsum(t2.bbbb, (0, 1, 2, 3), x22, (1, 3, 4, 5), (5, 0, 4, 2)) * -4.0
    rdm2_f_abab_oovo = np.zeros((nocc[0], nocc[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_oovo += einsum(t2.abab, (0, 1, 2, 3), x22, (4, 3, 0, 5), (5, 1, 2, 4)) * 2.0
    rdm2_f_abab_ovvo += einsum(t1.aa, (0, 1), x22, (2, 3, 0, 4), (4, 3, 1, 2)) * -2.0
    rdm2_f_abab_ovvv += einsum(t2.abab, (0, 1, 2, 3), x22, (1, 4, 0, 5), (5, 4, 2, 3)) * -2.0
    x23 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x23 += einsum(t2.bbbb, (0, 1, 2, 3), l3.babbab, (2, 4, 3, 5, 6, 1), (5, 0, 6, 4))
    rdm2_f_abab_vooo += einsum(x23, (0, 1, 2, 3), (3, 1, 2, 0)) * -2.0
    x24 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x24 += einsum(t2.abab, (0, 1, 2, 3), l3.abaaba, (4, 3, 2, 5, 6, 0), (6, 1, 5, 4))
    rdm2_f_abab_vooo += einsum(x24, (0, 1, 2, 3), (3, 1, 2, 0)) * -2.0
    x25 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x25 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x25 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    x26 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x26 += einsum(t1.aa, (0, 1), x25, (2, 3, 4, 1), (2, 3, 0, 4)) * 2.0
    rdm2_f_abab_oooo += einsum(x26, (0, 1, 2, 3), (2, 1, 3, 0))
    x27 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x27 += einsum(delta.aa.oo, (0, 1), (0, 1)) * -1.0
    x27 += einsum(x11, (0, 1), (0, 1))
    x27 += einsum(x5, (0, 1), (0, 1))
    x27 += einsum(x6, (0, 1), (0, 1)) * 2.0
    x27 += einsum(x7, (0, 1), (0, 1)) * 0.9999999999999601
    x27 += einsum(x8, (0, 1), (0, 1)) * 1.9999999999999194
    x27 += einsum(x9, (0, 1), (0, 1)) * 2.9999999999998788
    rdm2_f_abab_oooo += einsum(delta.bb.oo, (0, 1), x27, (2, 3), (3, 0, 2, 1)) * -1.0
    del x27
    x28 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x28 += einsum(l1.bb, (0, 1), t1.bb, (2, 0), (1, 2))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x28, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x28, (2, 3), (3, 0, 1, 2))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x28, (2, 3), (0, 3, 2, 1))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x28, (2, 3), (3, 0, 2, 1)) * -1.0
    x29 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x29 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    x30 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x30 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    x31 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x31 += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (6, 4, 5, 0, 1, 2), (3, 6))
    x32 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x32 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 4, 5, 0, 1, 2), (3, 6))
    x33 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x33 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 6, 5, 0, 1, 2), (4, 6))
    x34 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x34 += einsum(x28, (0, 1), (0, 1)) * 0.5
    x34 += einsum(x29, (0, 1), (0, 1))
    x34 += einsum(x30, (0, 1), (0, 1)) * 0.5
    x34 += einsum(x31, (0, 1), (0, 1)) * 1.4999999999999394
    x34 += einsum(x32, (0, 1), (0, 1)) * 0.9999999999999597
    x34 += einsum(x33, (0, 1), (0, 1)) * 0.49999999999998007
    rdm2_f_abab_oooo += einsum(delta.aa.oo, (0, 1), x34, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_abab_oovo += einsum(t1.aa, (0, 1), x34, (2, 3), (0, 3, 1, 2)) * -2.0
    del x34
    x35 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x35 += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (6, 7, 5, 0, 1, 2), (3, 4, 6, 7))
    rdm2_f_bbbb_oooo += einsum(x35, (0, 1, 2, 3), (2, 3, 1, 0)) * -6.000000000000116
    x36 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x36 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 4, 7, 0, 1, 2), (3, 5, 6, 7))
    rdm2_f_bbbb_oooo += einsum(x36, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0000000000000404
    x37 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x37 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_bbbb_oooo += einsum(x37, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x38 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x38 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm2_f_bbbb_ovoo += einsum(x38, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0
    rdm2_f_bbbb_vooo += einsum(x38, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x39 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x39 += einsum(t1.bb, (0, 1), x38, (2, 3, 4, 1), (2, 3, 0, 4))
    rdm2_f_bbbb_oooo += einsum(x39, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0
    x40 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x40 += einsum(t2.bbbb, (0, 1, 2, 3), l3.bbbbbb, (4, 2, 3, 5, 6, 1), (5, 6, 0, 4))
    rdm2_f_bbbb_ovoo += einsum(x40, (0, 1, 2, 3), (2, 3, 1, 0)) * -6.0
    rdm2_f_bbbb_vooo += einsum(x40, (0, 1, 2, 3), (3, 2, 1, 0)) * 6.0
    x41 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x41 += einsum(t2.abab, (0, 1, 2, 3), l3.babbab, (4, 2, 3, 5, 0, 6), (5, 6, 1, 4))
    rdm2_f_bbbb_ovoo += einsum(x41, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0
    rdm2_f_bbbb_vooo += einsum(x41, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x42 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x42 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3))
    x42 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.3333333333333333
    x43 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x43 += einsum(t1.bb, (0, 1), x42, (2, 3, 4, 1), (0, 2, 3, 4)) * 6.0
    rdm2_f_bbbb_oooo += einsum(x43, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    rdm2_f_bbbb_oooo += einsum(x43, (0, 1, 2, 3), (3, 0, 2, 1))
    del x43
    x44 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x44 += einsum(x29, (0, 1), (0, 1))
    x44 += einsum(x30, (0, 1), (0, 1)) * 0.5
    x44 += einsum(x31, (0, 1), (0, 1)) * 1.4999999999999394
    x44 += einsum(x32, (0, 1), (0, 1)) * 0.9999999999999597
    x44 += einsum(x33, (0, 1), (0, 1)) * 0.49999999999998007
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x44, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x44, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x44, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x44, (2, 3), (3, 0, 2, 1)) * -2.0
    del x44
    x45 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x45 += einsum(t1.aa, (0, 1), l3.aaaaaa, (2, 3, 1, 4, 5, 6), (4, 5, 6, 0, 2, 3))
    x46 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x46 += einsum(t1.aa, (0, 1), x45, (2, 3, 4, 5, 1, 6), (3, 4, 2, 5, 0, 6))
    x47 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x47 += einsum(t2.aaaa, (0, 1, 2, 3), x46, (1, 0, 4, 5, 6, 3), (4, 6, 5, 2))
    rdm2_f_aaaa_ooov = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_ooov += einsum(x47, (0, 1, 2, 3), (1, 2, 0, 3)) * 6.0
    rdm2_f_aaaa_oovo = np.zeros((nocc[0], nocc[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_oovo += einsum(x47, (0, 1, 2, 3), (1, 2, 3, 0)) * -6.0
    x48 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x48 += einsum(l2.aaaa, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 3, 6, 0, 1), (2, 4, 5, 6))
    rdm2_f_aaaa_ooov += einsum(x48, (0, 1, 2, 3), (1, 2, 0, 3)) * 6.0
    rdm2_f_aaaa_oovo += einsum(x48, (0, 1, 2, 3), (1, 2, 3, 0)) * -6.0
    x49 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x49 += einsum(l2.abab, (0, 1, 2, 3), t3.abaaba, (4, 3, 5, 6, 1, 0), (2, 4, 5, 6))
    rdm2_f_aaaa_ooov += einsum(x49, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_aaaa_oovo += einsum(x49, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    x50 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x50 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_aaaa_ooov += einsum(x50, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_aaaa_oovo += einsum(x50, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x51 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x51 += einsum(t1.aa, (0, 1), l3.abaaba, (2, 3, 1, 4, 5, 6), (5, 3, 4, 6, 0, 2))
    rdm2_f_abab_ovvv += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x51, (1, 6, 0, 2, 7, 5), (7, 6, 3, 4)) * 2.0000000000000404
    x52 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x52 += einsum(t1.aa, (0, 1), x51, (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -1.0
    x53 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x53 += einsum(t2.abab, (0, 1, 2, 3), x52, (1, 3, 4, 0, 5, 6), (4, 6, 5, 2)) * -1.0
    rdm2_f_aaaa_ooov += einsum(x53, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_aaaa_oovo += einsum(x53, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    x54 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x54 += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), (2, 3))
    x55 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x55 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 0), (2, 3))
    x56 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x56 += einsum(l2.bbbb, (0, 1, 2, 3), t3.babbab, (2, 4, 3, 0, 5, 1), (4, 5))
    x57 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x57 += einsum(l2.abab, (0, 1, 2, 3), t3.abaaba, (4, 3, 2, 5, 1, 0), (4, 5))
    x58 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x58 += einsum(l2.aaaa, (0, 1, 2, 3), t3.aaaaaa, (4, 2, 3, 5, 0, 1), (4, 5))
    x59 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x59 += einsum(t1.aa, (0, 1), l3.babbab, (2, 1, 3, 4, 5, 6), (4, 6, 2, 3, 5, 0))
    rdm2_f_abab_ovvv += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x59, (2, 0, 6, 5, 1, 7), (7, 6, 4, 3)) * 2.0000000000000404
    x60 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x60 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x59, (0, 2, 3, 5, 1, 6), (6, 4))
    x61 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x61 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x51, (1, 4, 0, 2, 6, 5), (6, 3)) * -1.0
    x62 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x62 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x45, (0, 1, 2, 6, 4, 5), (6, 3))
    x63 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x63 += einsum(t2.abab, (0, 1, 2, 3), x22, (1, 3, 0, 4), (4, 2)) * 2.0
    x64 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x64 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.3333333333333333
    x64 += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.3333333333333333
    x64 += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3))
    x65 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x65 += einsum(t2.aaaa, (0, 1, 2, 3), x64, (0, 1, 4, 3), (4, 2)) * -6.0
    x66 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x66 += einsum(x11, (0, 1), (0, 1))
    x66 += einsum(x5, (0, 1), (0, 1))
    x66 += einsum(x6, (0, 1), (0, 1)) * 2.0
    x66 += einsum(x7, (0, 1), (0, 1)) * 0.9999999999999601
    x66 += einsum(x8, (0, 1), (0, 1)) * 1.9999999999999194
    x66 += einsum(x9, (0, 1), (0, 1)) * 2.9999999999998788
    rdm2_f_abab_ooov += einsum(t1.bb, (0, 1), x66, (2, 3), (3, 0, 2, 1)) * -1.0
    x67 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x67 += einsum(t1.aa, (0, 1), x66, (0, 2), (2, 1))
    x68 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x68 += einsum(x54, (0, 1), (0, 1)) * -1.0
    x68 += einsum(x55, (0, 1), (0, 1)) * -2.0
    x68 += einsum(x56, (0, 1), (0, 1)) * -1.0
    x68 += einsum(x57, (0, 1), (0, 1)) * -2.0
    x68 += einsum(x58, (0, 1), (0, 1)) * -3.0
    x68 += einsum(x60, (0, 1), (0, 1)) * 0.9999999999999601
    x68 += einsum(x61, (0, 1), (0, 1)) * 1.9999999999999194
    x68 += einsum(x62, (0, 1), (0, 1)) * 2.9999999999998788
    x68 += einsum(x63, (0, 1), (0, 1))
    x68 += einsum(x65, (0, 1), (0, 1))
    x68 += einsum(x67, (0, 1), (0, 1))
    rdm2_f_aaaa_ooov += einsum(delta.aa.oo, (0, 1), x68, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_aaaa_ooov += einsum(delta.aa.oo, (0, 1), x68, (2, 3), (2, 0, 1, 3))
    rdm2_f_aaaa_oovo += einsum(delta.aa.oo, (0, 1), x68, (2, 3), (0, 2, 3, 1))
    rdm2_f_aaaa_oovo += einsum(delta.aa.oo, (0, 1), x68, (2, 3), (2, 0, 3, 1)) * -1.0
    x69 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x69 += einsum(t2.abab, (0, 1, 2, 3), x20, (1, 3, 4, 5), (4, 0, 5, 2))
    x70 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x70 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x59, (2, 0, 3, 5, 6, 7), (6, 7, 1, 4)) * -1.0
    x71 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x71 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x51, (1, 4, 6, 2, 7, 5), (6, 7, 0, 3)) * -1.0
    x72 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x72 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x45, (1, 2, 6, 7, 5, 4), (6, 7, 0, 3)) * -1.0
    x73 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x73 += einsum(t2.aaaa, (0, 1, 2, 3), x64, (1, 4, 5, 3), (0, 4, 5, 2)) * 12.0
    x74 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x74 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    x74 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x75 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x75 += einsum(t2.abab, (0, 1, 2, 3), x74, (1, 3, 4, 5), (0, 4, 5, 2))
    x76 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x76 += einsum(t1.aa, (0, 1), x14, (2, 3, 4, 1), (0, 2, 3, 4))
    x77 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x77 += einsum(t1.aa, (0, 1), x76, (2, 0, 3, 4), (3, 2, 4, 1)) * 2.0
    del x76
    x78 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x78 += einsum(delta.aa.oo, (0, 1), t1.aa, (2, 3), (0, 1, 2, 3))
    x78 += einsum(x69, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x78 += einsum(x70, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x78 += einsum(x71, (0, 1, 2, 3), (1, 0, 2, 3)) * -4.0
    x78 += einsum(x72, (0, 1, 2, 3), (1, 0, 2, 3)) * -9.0
    x78 += einsum(x73, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x73
    x78 += einsum(x75, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x75
    x78 += einsum(x77, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x77
    x78 += einsum(t1.aa, (0, 1), x66, (2, 3), (0, 2, 3, 1))
    del x66
    rdm2_f_aaaa_ooov += einsum(x78, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_ooov += einsum(x78, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_aaaa_oovo += einsum(x78, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_oovo += einsum(x78, (0, 1, 2, 3), (2, 0, 3, 1))
    del x78
    x79 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x79 += einsum(t2.abab, (0, 1, 2, 3), l3.abaaba, (4, 3, 5, 6, 1, 0), (6, 4, 5, 2))
    rdm2_f_aaaa_vvov = np.zeros((nvir[0], nvir[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_vvov += einsum(x79, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_aaaa_vvvo = np.zeros((nvir[0], nvir[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_aaaa_vvvo += einsum(x79, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    x80 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x80 += einsum(t2.aaaa, (0, 1, 2, 3), l3.aaaaaa, (4, 5, 3, 6, 0, 1), (6, 4, 5, 2))
    rdm2_f_aaaa_vvov += einsum(x80, (0, 1, 2, 3), (1, 2, 0, 3)) * 6.0
    rdm2_f_aaaa_vvvo += einsum(x80, (0, 1, 2, 3), (1, 2, 3, 0)) * -6.0
    x81 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x81 += einsum(x79, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.3333333333333333
    x81 += einsum(x80, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_abab_oovv += einsum(x81, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 1, 6, 2), (4, 5, 3, 6)) * 6.0
    x82 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x82 += einsum(t2.aaaa, (0, 1, 2, 3), x81, (4, 2, 3, 5), (0, 1, 4, 5)) * 6.0
    del x81
    rdm2_f_aaaa_ooov += einsum(x82, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_aaaa_oovo += einsum(x82, (0, 1, 2, 3), (1, 0, 3, 2))
    del x82
    x83 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x83 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.3333333333333269
    x83 += einsum(x2, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.3333333333333269
    x83 += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.3333333333333336
    x83 += einsum(x4, (0, 1, 2, 3), (0, 1, 3, 2))
    x84 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x84 += einsum(t1.aa, (0, 1), x83, (0, 2, 3, 4), (2, 3, 4, 1)) * 6.000000000000116
    del x83
    rdm2_f_aaaa_ooov += einsum(x84, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_aaaa_oovo += einsum(x84, (0, 1, 2, 3), (2, 1, 3, 0))
    del x84
    x85 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x85 += einsum(t1.bb, (0, 1), l3.abaaba, (2, 1, 3, 4, 5, 6), (5, 0, 4, 6, 2, 3))
    rdm2_f_abab_vovv += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x85, (1, 6, 0, 2, 7, 5), (7, 6, 3, 4)) * -2.0000000000000404
    x86 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x86 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x85, (1, 6, 7, 2, 5, 3), (6, 4, 7, 0)) * -1.0
    rdm2_f_abab_ooov += einsum(x86, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x87 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x87 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x51, (2, 5, 6, 1, 7, 4), (0, 3, 6, 7)) * -1.0
    rdm2_f_abab_ooov += einsum(x87, (0, 1, 2, 3), (3, 0, 2, 1)) * -4.0
    x88 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x88 += einsum(t1.bb, (0, 1), x51, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6))
    x89 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x89 += einsum(t2.abab, (0, 1, 2, 3), x88, (1, 4, 5, 0, 6, 2), (4, 3, 5, 6)) * -1.0
    rdm2_f_abab_ooov += einsum(x89, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x90 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x90 += einsum(t1.bb, (0, 1), l3.babbab, (2, 3, 1, 4, 5, 6), (4, 6, 0, 2, 5, 3))
    rdm2_f_abab_vovv += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x90, (2, 0, 6, 5, 1, 7), (7, 6, 4, 3)) * -2.0000000000000404
    x91 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x91 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x90, (2, 0, 6, 5, 7, 4), (6, 3, 7, 1))
    rdm2_f_abab_ooov += einsum(x91, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x92 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x92 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x59, (1, 2, 4, 5, 6, 7), (0, 3, 6, 7))
    rdm2_f_abab_ooov += einsum(x92, (0, 1, 2, 3), (3, 0, 2, 1)) * -3.0
    x93 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x93 += einsum(t1.bb, (0, 1), x59, (2, 3, 1, 4, 5, 6), (3, 2, 0, 4, 5, 6))
    x94 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x94 += einsum(t2.bbbb, (0, 1, 2, 3), x93, (1, 0, 4, 3, 5, 6), (4, 2, 5, 6))
    rdm2_f_abab_ooov += einsum(x94, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x95 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x95 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x45, (6, 2, 0, 7, 5, 3), (1, 4, 6, 7))
    rdm2_f_abab_ooov += einsum(x95, (0, 1, 2, 3), (3, 0, 2, 1)) * -3.0
    x96 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x96 += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), (3, 4, 1, 2))
    rdm2_f_abab_ooov += einsum(x96, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    x97 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x97 += einsum(l2.aaaa, (0, 1, 2, 3), t3.abaaba, (4, 5, 3, 0, 6, 1), (5, 6, 2, 4))
    rdm2_f_abab_ooov += einsum(x97, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    x98 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x98 += einsum(l2.abab, (0, 1, 2, 3), t3.babbab, (4, 5, 3, 6, 0, 1), (4, 6, 2, 5))
    rdm2_f_abab_ooov += einsum(x98, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    x99 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x99 += einsum(t2.bbbb, (0, 1, 2, 3), l3.babbab, (4, 5, 3, 0, 6, 1), (4, 2, 6, 5))
    rdm2_f_abab_vvov += einsum(x99, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x100 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x100 += einsum(t2.abab, (0, 1, 2, 3), l3.abaaba, (4, 5, 2, 6, 1, 0), (5, 3, 6, 4))
    rdm2_f_abab_vvov += einsum(x100, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x101 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x101 += einsum(x99, (0, 1, 2, 3), (0, 1, 2, 3))
    x101 += einsum(x100, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_abab_ooov += einsum(t2.abab, (0, 1, 2, 3), x101, (3, 4, 5, 2), (0, 1, 5, 4)) * 2.0
    rdm2_f_abab_ovvv += einsum(t2.aaaa, (0, 1, 2, 3), x101, (4, 5, 1, 3), (0, 4, 2, 5)) * 4.0
    rdm2_f_abab_vovv += einsum(t2.abab, (0, 1, 2, 3), x101, (3, 4, 0, 5), (5, 1, 2, 4)) * -2.0
    rdm2_f_abab_vvvv += einsum(t1.aa, (0, 1), x101, (2, 3, 0, 4), (4, 2, 1, 3)) * 2.0
    x102 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x102 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x102 += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3))
    x102 += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    rdm2_f_abab_ooov += einsum(t2.abab, (0, 1, 2, 3), x102, (0, 4, 5, 2), (5, 1, 4, 3)) * -2.0
    rdm2_f_abab_oovv += einsum(x102, (0, 1, 2, 3), t3.abaaba, (0, 4, 1, 5, 6, 3), (2, 4, 5, 6)) * 2.0
    x103 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x103 += einsum(t1.bb, (0, 1), l2.abab, (2, 1, 3, 4), (4, 0, 3, 2))
    rdm2_f_abab_vooo += einsum(x103, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    x104 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x104 += einsum(x103, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x104 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    x104 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_abab_ooov += einsum(t2.abab, (0, 1, 2, 3), x104, (1, 4, 5, 2), (0, 4, 5, 3)) * 2.0
    rdm2_f_abab_oovo += einsum(t2.aaaa, (0, 1, 2, 3), x104, (4, 5, 1, 3), (0, 5, 2, 4)) * -4.0
    rdm2_f_abab_voov += einsum(t1.bb, (0, 1), x104, (0, 2, 3, 4), (4, 2, 3, 1)) * -2.0
    rdm2_f_abab_vovo = np.zeros((nvir[0], nocc[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_vovo += einsum(t1.aa, (0, 1), x104, (2, 3, 0, 4), (4, 3, 1, 2)) * -2.0
    x105 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x105 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    x105 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x105 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_abab_oovv += einsum(x105, (0, 1, 2, 3), t3.babbab, (4, 2, 0, 5, 6, 1), (3, 4, 6, 5)) * -2.0
    rdm2_f_abab_ovov = np.zeros((nocc[0], nvir[1], nocc[0], nvir[1]), dtype=types[float])
    rdm2_f_abab_ovov += einsum(t1.bb, (0, 1), x105, (0, 2, 3, 4), (4, 2, 3, 1)) * -1.0
    x106 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x106 += einsum(t1.bb, (0, 1), x105, (2, 1, 3, 4), (2, 0, 3, 4)) * 0.4999999999999899
    x107 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x107 += einsum(t1.aa, (0, 1), x25, (2, 3, 4, 1), (2, 3, 0, 4)) * 0.9999999999999798
    x108 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x108 += einsum(x18, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.4999999999999899
    x108 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3))
    x108 += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3))
    x108 += einsum(x106, (0, 1, 2, 3), (0, 1, 2, 3))
    del x106
    x108 += einsum(x107, (0, 1, 2, 3), (0, 1, 3, 2))
    del x107
    rdm2_f_abab_ooov += einsum(t1.bb, (0, 1), x108, (0, 2, 3, 4), (4, 2, 3, 1)) * 2.0000000000000404
    rdm2_f_abab_oovo += einsum(t1.aa, (0, 1), x108, (2, 3, 0, 4), (4, 3, 1, 2)) * 2.0000000000000404
    del x108
    x109 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x109 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 0), (2, 3))
    x110 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x110 += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), (2, 3))
    x111 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x111 += einsum(l2.bbbb, (0, 1, 2, 3), t3.bbbbbb, (4, 2, 3, 5, 0, 1), (4, 5))
    x112 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x112 += einsum(l2.abab, (0, 1, 2, 3), t3.babbab, (4, 2, 3, 5, 0, 1), (4, 5))
    x113 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x113 += einsum(l2.aaaa, (0, 1, 2, 3), t3.abaaba, (2, 4, 3, 0, 5, 1), (4, 5))
    x114 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x114 += einsum(t1.bb, (0, 1), l3.bbbbbb, (2, 3, 1, 4, 5, 6), (4, 5, 6, 0, 2, 3))
    x115 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x115 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x114, (0, 1, 2, 6, 4, 5), (6, 3))
    x116 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x116 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x90, (2, 0, 6, 5, 1, 4), (6, 3))
    x117 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x117 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x85, (1, 6, 0, 2, 3, 5), (6, 4))
    x118 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x118 += einsum(x38, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x118 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x118 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_abab_oovo += einsum(t2.abab, (0, 1, 2, 3), x118, (1, 4, 5, 3), (0, 5, 2, 4)) * -2.0
    rdm2_f_abab_oovv += einsum(x118, (0, 1, 2, 3), t3.babbab, (0, 4, 1, 5, 6, 3), (4, 2, 6, 5)) * 2.0
    x119 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x119 += einsum(x28, (0, 1), (0, 1))
    x119 += einsum(x29, (0, 1), (0, 1)) * 2.0
    x119 += einsum(x30, (0, 1), (0, 1))
    x119 += einsum(x31, (0, 1), (0, 1)) * 2.9999999999998788
    x119 += einsum(x32, (0, 1), (0, 1)) * 1.9999999999999194
    x119 += einsum(x33, (0, 1), (0, 1)) * 0.9999999999999601
    rdm2_f_abab_oovv += einsum(x119, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    x120 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x120 += einsum(t1.bb, (0, 1), (0, 1)) * -1.0
    x120 += einsum(x109, (0, 1), (0, 1)) * -2.0
    x120 += einsum(x110, (0, 1), (0, 1)) * -1.0
    x120 += einsum(x111, (0, 1), (0, 1)) * -3.0
    x120 += einsum(x112, (0, 1), (0, 1)) * -2.0
    x120 += einsum(x113, (0, 1), (0, 1)) * -1.0
    x120 += einsum(x115, (0, 1), (0, 1)) * 2.9999999999998788
    x120 += einsum(x116, (0, 1), (0, 1)) * 1.9999999999999194
    x120 += einsum(x117, (0, 1), (0, 1)) * 0.9999999999999601
    x120 += einsum(t2.bbbb, (0, 1, 2, 3), x118, (0, 1, 4, 3), (4, 2)) * -2.0
    x120 += einsum(t2.abab, (0, 1, 2, 3), x104, (1, 4, 0, 2), (4, 3)) * 2.0
    x120 += einsum(t1.bb, (0, 1), x119, (0, 2), (2, 1))
    rdm2_f_abab_ooov += einsum(delta.aa.oo, (0, 1), x120, (2, 3), (0, 2, 1, 3)) * -1.0
    del x120
    x121 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x121 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_bbbb_ooov = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_ooov += einsum(x121, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_bbbb_oovo = np.zeros((nocc[1], nocc[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_oovo += einsum(x121, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x122 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x122 += einsum(t1.bb, (0, 1), x90, (2, 3, 4, 1, 5, 6), (2, 3, 4, 0, 5, 6)) * -1.0
    x123 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x123 += einsum(t2.abab, (0, 1, 2, 3), x122, (4, 1, 5, 6, 0, 2), (4, 6, 5, 3)) * -1.0
    rdm2_f_bbbb_ooov += einsum(x123, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_bbbb_oovo += einsum(x123, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    x124 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x124 += einsum(l2.bbbb, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 3, 6, 0, 1), (2, 4, 5, 6))
    rdm2_f_bbbb_ooov += einsum(x124, (0, 1, 2, 3), (1, 2, 0, 3)) * 6.0
    rdm2_f_bbbb_oovo += einsum(x124, (0, 1, 2, 3), (1, 2, 3, 0)) * -6.0
    x125 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x125 += einsum(l2.abab, (0, 1, 2, 3), t3.babbab, (4, 2, 5, 6, 0, 1), (3, 4, 5, 6))
    rdm2_f_bbbb_ooov += einsum(x125, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_bbbb_oovo += einsum(x125, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    x126 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x126 += einsum(t1.bb, (0, 1), x114, (2, 3, 4, 5, 1, 6), (3, 4, 2, 5, 0, 6))
    x127 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x127 += einsum(t2.bbbb, (0, 1, 2, 3), x126, (0, 1, 4, 5, 6, 3), (4, 6, 5, 2)) * -1.0
    rdm2_f_bbbb_ooov += einsum(x127, (0, 1, 2, 3), (1, 2, 0, 3)) * 6.0
    rdm2_f_bbbb_oovo += einsum(x127, (0, 1, 2, 3), (1, 2, 3, 0)) * -6.0
    x128 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x128 += einsum(t2.bbbb, (0, 1, 2, 3), x118, (0, 1, 4, 3), (4, 2)) * -1.0
    x129 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x129 += einsum(t2.abab, (0, 1, 2, 3), x104, (1, 4, 0, 2), (4, 3))
    del x104
    x130 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x130 += einsum(t1.bb, (0, 1), x119, (0, 2), (2, 1)) * 0.5
    x131 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x131 += einsum(x109, (0, 1), (0, 1)) * -1.0
    del x109
    x131 += einsum(x110, (0, 1), (0, 1)) * -0.5
    del x110
    x131 += einsum(x111, (0, 1), (0, 1)) * -1.5
    del x111
    x131 += einsum(x112, (0, 1), (0, 1)) * -1.0
    del x112
    x131 += einsum(x113, (0, 1), (0, 1)) * -0.5
    del x113
    x131 += einsum(x115, (0, 1), (0, 1)) * 1.4999999999999394
    del x115
    x131 += einsum(x116, (0, 1), (0, 1)) * 0.9999999999999597
    del x116
    x131 += einsum(x117, (0, 1), (0, 1)) * 0.49999999999998007
    del x117
    x131 += einsum(x128, (0, 1), (0, 1))
    del x128
    x131 += einsum(x129, (0, 1), (0, 1))
    del x129
    x131 += einsum(x130, (0, 1), (0, 1))
    del x130
    rdm2_f_bbbb_ooov += einsum(delta.bb.oo, (0, 1), x131, (2, 3), (0, 2, 1, 3)) * -2.0
    rdm2_f_bbbb_ooov += einsum(delta.bb.oo, (0, 1), x131, (2, 3), (2, 0, 1, 3)) * 2.0
    rdm2_f_bbbb_oovo += einsum(delta.bb.oo, (0, 1), x131, (2, 3), (0, 2, 3, 1)) * 2.0
    rdm2_f_bbbb_oovo += einsum(delta.bb.oo, (0, 1), x131, (2, 3), (2, 0, 3, 1)) * -2.0
    rdm2_f_abab_oovv += einsum(t1.aa, (0, 1), x131, (2, 3), (0, 2, 1, 3)) * -2.0
    x132 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x132 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x114, (6, 1, 2, 7, 4, 5), (6, 7, 0, 3))
    x133 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x133 += einsum(t1.bb, (0, 1), x40, (2, 3, 4, 1), (2, 3, 0, 4))
    x134 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x134 += einsum(t1.bb, (0, 1), x133, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    del x133
    x135 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x135 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x90, (6, 2, 7, 5, 1, 4), (6, 7, 0, 3)) * -1.0
    x136 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x136 += einsum(t2.abab, (0, 1, 2, 3), x24, (4, 5, 0, 2), (4, 1, 5, 3))
    x137 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x137 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x85, (6, 7, 0, 2, 5, 3), (6, 7, 1, 4)) * -1.0
    x138 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x138 += einsum(x38, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x138 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x139 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x139 += einsum(t2.bbbb, (0, 1, 2, 3), x138, (1, 4, 5, 3), (4, 5, 0, 2)) * 4.0
    del x138
    x140 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x140 += einsum(x103, (0, 1, 2, 3), (0, 1, 2, 3))
    x140 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x141 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x141 += einsum(t2.abab, (0, 1, 2, 3), x140, (4, 5, 0, 2), (4, 5, 1, 3))
    del x140
    x142 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x142 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x142 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -1.0
    x143 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x143 += einsum(x142, (0, 1, 2, 3), x41, (4, 0, 5, 2), (1, 4, 5, 3)) * 2.0
    del x142
    x144 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x144 += einsum(delta.bb.oo, (0, 1), t1.bb, (2, 3), (0, 1, 2, 3))
    x144 += einsum(x132, (0, 1, 2, 3), (1, 0, 2, 3)) * -9.0
    x144 += einsum(x134, (0, 1, 2, 3), (1, 0, 2, 3)) * 6.0
    del x134
    x144 += einsum(x135, (0, 1, 2, 3), (1, 0, 2, 3)) * -4.0
    x144 += einsum(x136, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    x144 += einsum(x137, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x144 += einsum(x139, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x139
    x144 += einsum(x141, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x141
    x144 += einsum(x143, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x143
    x144 += einsum(t1.bb, (0, 1), x119, (2, 3), (0, 2, 3, 1))
    del x119
    rdm2_f_bbbb_ooov += einsum(x144, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_ooov += einsum(x144, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_bbbb_oovo += einsum(x144, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_bbbb_oovo += einsum(x144, (0, 1, 2, 3), (2, 0, 3, 1))
    del x144
    x145 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x145 += einsum(t2.bbbb, (0, 1, 2, 3), l3.bbbbbb, (4, 5, 3, 6, 0, 1), (6, 4, 5, 2))
    rdm2_f_bbbb_vvov = np.zeros((nvir[1], nvir[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_vvov += einsum(x145, (0, 1, 2, 3), (1, 2, 0, 3)) * 6.0
    rdm2_f_bbbb_vvvo = np.zeros((nvir[1], nvir[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_bbbb_vvvo += einsum(x145, (0, 1, 2, 3), (1, 2, 3, 0)) * -6.0
    x146 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x146 += einsum(t2.abab, (0, 1, 2, 3), l3.babbab, (4, 2, 5, 6, 0, 1), (6, 4, 5, 3))
    rdm2_f_bbbb_vvov += einsum(x146, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_bbbb_vvvo += einsum(x146, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    x147 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x147 += einsum(x145, (0, 1, 2, 3), (0, 2, 1, 3)) * -3.0
    x147 += einsum(x146, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_abab_oovv += einsum(x147, (0, 1, 2, 3), t3.babbab, (4, 5, 0, 1, 6, 2), (5, 4, 6, 3)) * 2.0
    rdm2_f_abab_ovvv += einsum(t2.abab, (0, 1, 2, 3), x147, (1, 4, 3, 5), (0, 4, 2, 5)) * -2.0
    x148 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x148 += einsum(t2.bbbb, (0, 1, 2, 3), x147, (4, 2, 3, 5), (4, 0, 1, 5)) * 2.0
    del x147
    rdm2_f_bbbb_ooov += einsum(x148, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_bbbb_oovo += einsum(x148, (0, 1, 2, 3), (2, 1, 3, 0))
    del x148
    x149 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x149 += einsum(x37, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.9999999999999798
    x149 += einsum(x39, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.9999999999999798
    x149 += einsum(x35, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.999999999999998
    x149 += einsum(x36, (0, 1, 2, 3), (0, 1, 3, 2))
    x150 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x150 += einsum(t1.bb, (0, 1), x149, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0000000000000404
    del x149
    rdm2_f_bbbb_ooov += einsum(x150, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_bbbb_oovo += einsum(x150, (0, 1, 2, 3), (2, 1, 3, 0))
    del x150
    x151 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x151 += einsum(l2.bbbb, (0, 1, 2, 3), t3.babbab, (4, 5, 3, 0, 6, 1), (2, 4, 5, 6))
    rdm2_f_abab_oovo += einsum(x151, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    x152 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x152 += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), (1, 3, 2, 4))
    rdm2_f_abab_oovo += einsum(x152, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x153 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x153 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x114, (0, 2, 6, 7, 3, 5), (6, 7, 1, 4))
    rdm2_f_abab_oovo += einsum(x153, (0, 1, 2, 3), (2, 1, 3, 0)) * -3.0
    x154 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x154 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x59, (2, 6, 3, 5, 1, 7), (6, 0, 7, 4)) * -1.0
    rdm2_f_abab_oovo += einsum(x154, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x155 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x155 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x90, (1, 6, 7, 4, 2, 5), (6, 7, 0, 3))
    rdm2_f_abab_oovo += einsum(x155, (0, 1, 2, 3), (2, 1, 3, 0)) * -4.0
    x156 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x156 += einsum(t2.abab, (0, 1, 2, 3), x93, (1, 4, 5, 3, 0, 6), (4, 5, 6, 2))
    rdm2_f_abab_oovo += einsum(x156, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x157 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x157 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x51, (6, 4, 0, 2, 7, 5), (6, 1, 7, 3)) * -1.0
    rdm2_f_abab_oovo += einsum(x157, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x158 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x158 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x85, (6, 7, 1, 2, 4, 5), (6, 7, 0, 3))
    rdm2_f_abab_oovo += einsum(x158, (0, 1, 2, 3), (2, 1, 3, 0)) * -3.0
    x159 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x159 += einsum(t2.aaaa, (0, 1, 2, 3), x88, (4, 5, 1, 0, 6, 3), (4, 5, 6, 2))
    rdm2_f_abab_oovo += einsum(x159, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x160 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x160 += einsum(l2.abab, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 6, 1, 0), (3, 5, 4, 6))
    rdm2_f_abab_oovo += einsum(x160, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    x161 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x161 += einsum(t2.abab, (0, 1, 2, 3), l3.babbab, (4, 5, 3, 6, 0, 1), (6, 4, 5, 2))
    rdm2_f_abab_vvvo = np.zeros((nvir[0], nvir[1], nvir[0], nocc[1]), dtype=types[float])
    rdm2_f_abab_vvvo += einsum(x161, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x162 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x162 += einsum(t2.aaaa, (0, 1, 2, 3), l3.abaaba, (4, 5, 3, 0, 6, 1), (6, 5, 4, 2))
    rdm2_f_abab_vvvo += einsum(x162, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x163 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x163 += einsum(x161, (0, 1, 2, 3), (0, 1, 2, 3))
    x163 += einsum(x162, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_abab_oovo += einsum(t2.abab, (0, 1, 2, 3), x163, (4, 3, 2, 5), (0, 1, 5, 4)) * 2.0
    rdm2_f_abab_ovvv += einsum(t2.abab, (0, 1, 2, 3), x163, (1, 4, 2, 5), (0, 4, 5, 3)) * -2.0
    rdm2_f_abab_vovv += einsum(t2.bbbb, (0, 1, 2, 3), x163, (1, 3, 4, 5), (4, 0, 5, 2)) * 4.0
    x164 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x164 += einsum(t1.aa, (0, 1), (0, 1)) * -1.0
    x164 += einsum(x54, (0, 1), (0, 1)) * -1.0
    del x54
    x164 += einsum(x55, (0, 1), (0, 1)) * -2.0
    del x55
    x164 += einsum(x56, (0, 1), (0, 1)) * -1.0
    del x56
    x164 += einsum(x57, (0, 1), (0, 1)) * -2.0
    del x57
    x164 += einsum(x58, (0, 1), (0, 1)) * -3.0
    del x58
    x164 += einsum(x60, (0, 1), (0, 1)) * 0.9999999999999601
    del x60
    x164 += einsum(x61, (0, 1), (0, 1)) * 1.9999999999999194
    del x61
    x164 += einsum(x62, (0, 1), (0, 1)) * 2.9999999999998788
    del x62
    x164 += einsum(x63, (0, 1), (0, 1))
    del x63
    x164 += einsum(x65, (0, 1), (0, 1))
    del x65
    x164 += einsum(x67, (0, 1), (0, 1))
    del x67
    rdm2_f_abab_oovo += einsum(delta.bb.oo, (0, 1), x164, (2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_abab_oovv += einsum(t1.bb, (0, 1), x164, (2, 3), (2, 0, 3, 1)) * -1.0
    del x164
    x165 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x165 += einsum(t2.abab, (0, 1, 2, 3), x59, (4, 1, 5, 3, 0, 6), (4, 5, 6, 2))
    rdm2_f_abab_ovvo += einsum(x165, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    x166 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x166 += einsum(t2.abab, (0, 1, 2, 3), x165, (1, 3, 4, 5), (4, 0, 2, 5))
    x167 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x167 += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 2, 5, 0), (3, 1, 4, 5))
    rdm2_f_abab_ovvo += einsum(x167, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x168 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x168 += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.babbab, (4, 6, 5, 1, 7, 2), (3, 0, 6, 7))
    rdm2_f_abab_ovvo += einsum(x168, (0, 1, 2, 3), (2, 1, 3, 0)) * 3.0
    x169 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x169 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 5, 4, 7, 2, 1), (3, 0, 6, 7))
    rdm2_f_abab_ovvo += einsum(x169, (0, 1, 2, 3), (2, 1, 3, 0)) * 4.0
    x170 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x170 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (6, 3, 5, 7, 0, 2), (4, 1, 6, 7))
    rdm2_f_abab_ovvo += einsum(x170, (0, 1, 2, 3), (2, 1, 3, 0)) * 3.0
    x171 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x171 += einsum(x167, (0, 1, 2, 3), (0, 1, 2, 3))
    x171 += einsum(x168, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.5
    x171 += einsum(x169, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x171 += einsum(x170, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.5
    x172 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x172 += einsum(t2.abab, (0, 1, 2, 3), x171, (1, 3, 4, 5), (4, 0, 5, 2)) * 2.0
    del x171
    x173 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x173 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 6, 5, 0, 7, 2), (4, 6, 1, 7))
    rdm2_f_aaaa_ovov += einsum(x173, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_aaaa_ovvo += einsum(x173, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_aaaa_voov += einsum(x173, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_aaaa_vovo += einsum(x173, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x174 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x174 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 4, 5, 7, 1, 2), (3, 6, 0, 7))
    rdm2_f_aaaa_ovov += einsum(x174, (0, 1, 2, 3), (1, 2, 0, 3)) * -4.0
    rdm2_f_aaaa_ovvo += einsum(x174, (0, 1, 2, 3), (1, 2, 3, 0)) * 4.0
    rdm2_f_aaaa_voov += einsum(x174, (0, 1, 2, 3), (2, 1, 0, 3)) * 4.0
    rdm2_f_aaaa_vovo += einsum(x174, (0, 1, 2, 3), (2, 1, 3, 0)) * -4.0
    x175 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x175 += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (6, 4, 5, 7, 1, 2), (3, 6, 0, 7))
    rdm2_f_aaaa_ovov += einsum(x175, (0, 1, 2, 3), (1, 2, 0, 3)) * -9.0
    rdm2_f_aaaa_ovvo += einsum(x175, (0, 1, 2, 3), (1, 2, 3, 0)) * 9.0
    rdm2_f_aaaa_voov += einsum(x175, (0, 1, 2, 3), (2, 1, 0, 3)) * 9.0
    rdm2_f_aaaa_vovo += einsum(x175, (0, 1, 2, 3), (2, 1, 3, 0)) * -9.0
    x176 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x176 += einsum(x173, (0, 1, 2, 3), (0, 1, 2, 3))
    x176 += einsum(x174, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    x176 += einsum(x175, (0, 1, 2, 3), (0, 1, 2, 3)) * 9.0
    x177 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x177 += einsum(t2.aaaa, (0, 1, 2, 3), x176, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    del x176
    x178 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x178 += einsum(t2.abab, (0, 1, 2, 3), x51, (1, 3, 4, 0, 5, 6), (4, 5, 6, 2))
    rdm2_f_aaaa_ovov += einsum(x178, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_aaaa_ovvo += einsum(x178, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    rdm2_f_aaaa_voov += einsum(x178, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_aaaa_vovo += einsum(x178, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x179 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x179 += einsum(t2.aaaa, (0, 1, 2, 3), x45, (4, 1, 0, 5, 6, 3), (4, 5, 6, 2))
    rdm2_f_aaaa_ovov += einsum(x179, (0, 1, 2, 3), (1, 2, 0, 3)) * 6.0
    rdm2_f_aaaa_ovvo += einsum(x179, (0, 1, 2, 3), (1, 2, 3, 0)) * -6.0
    rdm2_f_aaaa_voov += einsum(x179, (0, 1, 2, 3), (2, 1, 0, 3)) * -6.0
    rdm2_f_aaaa_vovo += einsum(x179, (0, 1, 2, 3), (2, 1, 3, 0)) * 6.0
    x180 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x180 += einsum(t1.aa, (0, 1), x14, (2, 0, 3, 4), (2, 3, 1, 4))
    x181 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x181 += einsum(x178, (0, 1, 2, 3), (0, 1, 2, 3))
    x181 += einsum(x179, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x181 += einsum(x180, (0, 1, 2, 3), (0, 1, 3, 2))
    x182 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x182 += einsum(t2.aaaa, (0, 1, 2, 3), x181, (1, 4, 3, 5), (4, 0, 5, 2)) * 4.0
    del x181
    x183 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x183 += einsum(t2.aaaa, (0, 1, 2, 3), x51, (4, 5, 0, 1, 6, 3), (4, 5, 6, 2)) * -1.0
    rdm2_f_abab_ovvo += einsum(x183, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    x184 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x184 += einsum(t1.aa, (0, 1), x21, (2, 3, 0, 4), (2, 3, 4, 1))
    x185 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x185 += einsum(x183, (0, 1, 2, 3), (0, 1, 2, 3))
    x185 += einsum(x184, (0, 1, 2, 3), (0, 1, 2, 3))
    del x184
    x186 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x186 += einsum(t2.abab, (0, 1, 2, 3), x185, (1, 3, 4, 5), (4, 0, 5, 2)) * 2.0
    del x185
    x187 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x187 += einsum(t2.abab, (0, 1, 2, 3), x19, (1, 3, 4, 5), (4, 5, 0, 2))
    x188 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x188 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x189 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x189 += einsum(x187, (0, 1, 2, 3), (0, 1, 2, 3))
    del x187
    x189 += einsum(x188, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x188
    x189 += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3))
    del x70
    x189 += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x69
    x189 += einsum(x71, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x71
    x189 += einsum(x72, (0, 1, 2, 3), (0, 1, 2, 3)) * 9.0
    del x72
    x190 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x190 += einsum(t1.aa, (0, 1), x189, (0, 2, 3, 4), (2, 3, 4, 1))
    del x189
    x191 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x191 += einsum(x166, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x166
    x191 += einsum(x172, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x172
    x191 += einsum(x177, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x177
    x191 += einsum(x182, (0, 1, 2, 3), (0, 1, 2, 3))
    del x182
    x191 += einsum(x186, (0, 1, 2, 3), (0, 1, 2, 3))
    del x186
    x191 += einsum(x190, (0, 1, 2, 3), (0, 1, 3, 2))
    del x190
    x191 += einsum(t1.aa, (0, 1), x68, (2, 3), (0, 2, 1, 3))
    del x68
    rdm2_f_aaaa_oovv += einsum(x191, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x191, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_aaaa_oovv += einsum(x191, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_aaaa_oovv += einsum(x191, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x191
    x192 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x192 += einsum(x11, (0, 1), t2.aaaa, (2, 0, 3, 4), (1, 2, 3, 4))
    x193 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x193 += einsum(x74, (0, 1, 2, 3), t3.abaaba, (4, 0, 2, 5, 1, 6), (4, 3, 5, 6)) * -2.0
    del x74
    x194 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x194 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x194 += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3))
    x195 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x195 += einsum(x194, (0, 1, 2, 3), t3.aaaaaa, (4, 0, 1, 5, 6, 3), (2, 4, 5, 6)) * -6.0
    del x194
    x196 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x196 += einsum(t1.aa, (0, 1), x14, (2, 3, 4, 1), (0, 2, 3, 4)) * 0.3333333333333333
    del x14
    x197 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x197 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x197 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3))
    x198 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x198 += einsum(x196, (0, 1, 2, 3), x197, (1, 2, 4, 5), (0, 3, 4, 5)) * 6.0
    del x196, x197
    x199 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x199 += einsum(x5, (0, 1), (0, 1)) * 0.5000000000000203
    x199 += einsum(x6, (0, 1), (0, 1)) * 1.0000000000000406
    x199 += einsum(x8, (0, 1), (0, 1))
    x200 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x200 += einsum(x199, (0, 1), t2.aaaa, (2, 0, 3, 4), (1, 2, 3, 4)) * -3.999999999999838
    del x199
    x201 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x201 += einsum(x192, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x192
    x201 += einsum(x193, (0, 1, 2, 3), (1, 0, 3, 2))
    del x193
    x201 += einsum(x195, (0, 1, 2, 3), (0, 1, 2, 3))
    del x195
    x201 += einsum(x198, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x198
    x201 += einsum(x200, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x200
    rdm2_f_aaaa_oovv += einsum(x201, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_aaaa_oovv += einsum(x201, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x201
    x202 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x202 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_aaaa_ovov += einsum(x202, (0, 1, 2, 3), (1, 2, 0, 3)) * -4.0
    rdm2_f_aaaa_ovvo += einsum(x202, (0, 1, 2, 3), (1, 2, 3, 0)) * 4.0
    rdm2_f_aaaa_voov += einsum(x202, (0, 1, 2, 3), (2, 1, 0, 3)) * 4.0
    rdm2_f_aaaa_vovo += einsum(x202, (0, 1, 2, 3), (2, 1, 3, 0)) * -4.0
    x203 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x203 += einsum(t2.aaaa, (0, 1, 2, 3), x202, (1, 4, 3, 5), (0, 4, 2, 5))
    x204 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x204 += einsum(x163, (0, 1, 2, 3), t3.abaaba, (4, 0, 5, 6, 1, 2), (4, 5, 3, 6)) * -4.0
    x205 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x205 += einsum(x79, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x79
    x205 += einsum(x80, (0, 1, 2, 3), (0, 2, 1, 3)) * -3.0
    del x80
    rdm2_f_abab_vovv += einsum(t2.abab, (0, 1, 2, 3), x205, (0, 4, 2, 5), (4, 1, 5, 3)) * -2.0
    x206 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x206 += einsum(x205, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 1, 2), (4, 5, 3, 6)) * -6.0
    x207 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x207 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    x208 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x208 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4))
    x209 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x209 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 4, 5, 0, 6, 2), (1, 6))
    x210 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x210 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 4, 5, 6, 1, 2), (0, 6))
    x211 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x211 += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.aaaaaa, (3, 4, 5, 6, 1, 2), (0, 6))
    x212 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x212 += einsum(x207, (0, 1), (0, 1))
    x212 += einsum(x208, (0, 1), (0, 1)) * 2.0
    x212 += einsum(x209, (0, 1), (0, 1)) * 0.9999999999999597
    x212 += einsum(x210, (0, 1), (0, 1)) * 1.999999999999919
    x212 += einsum(x211, (0, 1), (0, 1)) * 2.999999999999883
    x213 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x213 += einsum(x212, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 1, 4)) * -2.0
    del x212
    x214 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x214 += einsum(t2.aaaa, (0, 1, 2, 3), x205, (4, 2, 3, 5), (4, 0, 1, 5))
    x215 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x215 += einsum(x50, (0, 1, 2, 3), (0, 2, 1, 3))
    del x50
    x215 += einsum(x49, (0, 1, 2, 3), (0, 2, 1, 3))
    del x49
    x215 += einsum(x48, (0, 1, 2, 3), (0, 2, 1, 3)) * 3.0
    del x48
    x215 += einsum(x53, (0, 1, 2, 3), (0, 2, 1, 3))
    del x53
    x215 += einsum(x47, (0, 1, 2, 3), (0, 2, 1, 3)) * 3.0
    del x47
    x215 += einsum(x214, (0, 1, 2, 3), (0, 2, 1, 3))
    del x214
    x216 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x216 += einsum(t1.aa, (0, 1), x215, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    del x215
    x217 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x217 += einsum(x203, (0, 1, 2, 3), (0, 1, 2, 3)) * 8.0
    del x203
    x217 += einsum(x204, (0, 1, 2, 3), (1, 0, 2, 3))
    del x204
    x217 += einsum(x206, (0, 1, 2, 3), (0, 1, 2, 3))
    del x206
    x217 += einsum(x213, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x213
    x217 += einsum(x216, (0, 1, 2, 3), (1, 0, 3, 2))
    del x216
    rdm2_f_aaaa_oovv += einsum(x217, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_aaaa_oovv += einsum(x217, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x217
    x218 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x218 += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 0, 4, 5))
    rdm2_f_abab_ovvo += einsum(x218, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x219 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x219 += einsum(t2.abab, (0, 1, 2, 3), x218, (1, 3, 4, 5), (4, 0, 5, 2))
    x220 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x220 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3))
    x220 += einsum(x219, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x219
    rdm2_f_aaaa_oovv += einsum(x220, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x220, (0, 1, 2, 3), (0, 1, 2, 3))
    del x220
    x221 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x221 += einsum(x20, (0, 1, 2, 3), t3.abaaba, (4, 0, 2, 5, 1, 6), (3, 4, 5, 6))
    x222 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x222 += einsum(x13, (0, 1, 2, 3), t3.aaaaaa, (4, 0, 1, 5, 6, 3), (2, 4, 5, 6))
    x223 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x223 += einsum(x7, (0, 1), (0, 1))
    x223 += einsum(x9, (0, 1), (0, 1)) * 3.000000000000004
    x224 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x224 += einsum(x223, (0, 1), t2.aaaa, (2, 0, 3, 4), (1, 2, 3, 4)) * -1.9999999999999194
    del x223
    x225 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x225 += einsum(x221, (0, 1, 2, 3), (0, 1, 3, 2)) * -4.0
    del x221
    x225 += einsum(x222, (0, 1, 2, 3), (0, 1, 2, 3)) * -18.0
    del x222
    x225 += einsum(x224, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x224
    rdm2_f_aaaa_oovv += einsum(x225, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x225, (0, 1, 2, 3), (1, 0, 3, 2))
    del x225
    x226 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x226 += einsum(t2.aaaa, (0, 1, 2, 3), l3.abaaba, (2, 4, 3, 5, 6, 7), (6, 4, 5, 7, 0, 1))
    x226 += einsum(x52, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 5, 4)) * 1.0000000000000606
    del x52
    rdm2_f_aaaa_oovv += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x226, (1, 4, 0, 2, 6, 7), (6, 7, 3, 5)) * 1.9999999999999194
    del x226
    x227 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x227 += einsum(t2.aaaa, (0, 1, 2, 3), l3.aaaaaa, (4, 2, 3, 5, 6, 7), (5, 6, 7, 0, 1, 4))
    x227 += einsum(x46, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0000000000000584
    del x46
    rdm2_f_aaaa_oovv += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x227, (0, 1, 2, 6, 7, 5), (6, 7, 3, 4)) * 5.999999999999766
    del x227
    x228 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x228 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0000000000000404
    x228 += einsum(x2, (0, 1, 2, 3), (0, 1, 3, 2)) * 1.0000000000000404
    x228 += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2))
    x228 += einsum(x4, (0, 1, 2, 3), (0, 1, 3, 2)) * 3.000000000000004
    rdm2_f_aaaa_oovv += einsum(t2.aaaa, (0, 1, 2, 3), x228, (0, 1, 4, 5), (5, 4, 2, 3)) * 1.9999999999999194
    del x228
    x229 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x229 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.9999999999999799
    del x0
    x229 += einsum(x2, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.9999999999999799
    del x2
    x229 += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2))
    del x3
    x229 += einsum(x4, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.999999999999998
    del x4
    x230 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x230 += einsum(t1.aa, (0, 1), x229, (0, 2, 3, 4), (2, 4, 3, 1)) * -0.3333333333333336
    del x229
    rdm2_f_aaaa_oovv += einsum(t1.aa, (0, 1), x230, (0, 2, 3, 4), (2, 3, 4, 1)) * -6.000000000000116
    del x230
    x231 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x231 += einsum(t2.abab, (0, 1, 2, 3), l3.abaaba, (4, 3, 2, 5, 6, 7), (6, 1, 5, 7, 0, 4))
    x231 += einsum(x88, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0000000000000606
    del x88
    rdm2_f_abab_oovv += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x231, (1, 6, 0, 2, 7, 5), (7, 6, 3, 4)) * -1.9999999999999194
    del x231
    x232 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x232 += einsum(t2.abab, (0, 1, 2, 3), l3.babbab, (4, 2, 3, 5, 6, 7), (5, 7, 1, 4, 6, 0))
    x232 += einsum(x93, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 1.0000000000000606
    del x93
    rdm2_f_abab_oovv += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x232, (0, 2, 6, 5, 1, 7), (7, 6, 4, 3)) * -1.9999999999999194
    del x232
    x233 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x233 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), (1, 4, 0, 2, 3, 5))
    x233 += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (3, 5, 0, 2, 4, 1)) * -0.5
    rdm2_f_abab_oovv += einsum(x101, (0, 1, 2, 3), x233, (4, 0, 2, 5, 3, 6), (5, 4, 6, 1)) * -4.0
    del x233
    x234 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x234 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), (0, 2, 3, 5, 1, 4))
    x234 += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (0, 3, 5, 1, 2, 4)) * -0.5
    rdm2_f_abab_oovv += einsum(x163, (0, 1, 2, 3), x234, (0, 4, 1, 5, 6, 2), (6, 4, 3, 5)) * -4.0
    del x163, x234
    x235 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x235 += einsum(x103, (0, 1, 2, 3), (0, 1, 2, 3))
    x235 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x235 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x24
    rdm2_f_abab_oovv += einsum(x235, (0, 1, 2, 3), t3.abaaba, (4, 0, 2, 5, 6, 3), (4, 1, 5, 6)) * -2.0
    rdm2_f_abab_vovv += einsum(t2.abab, (0, 1, 2, 3), x235, (1, 4, 0, 5), (5, 4, 2, 3)) * -1.0
    x236 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x236 += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (4, 5, 2, 0))
    rdm2_f_abab_voov += einsum(x236, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x237 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x237 += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 1, 5), (4, 5, 2, 0))
    rdm2_f_abab_voov += einsum(x237, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x238 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x238 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (6, 3, 5, 7, 0, 2), (6, 7, 4, 1))
    rdm2_f_abab_voov += einsum(x238, (0, 1, 2, 3), (3, 0, 2, 1)) * 3.0
    x239 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x239 += einsum(t2.bbbb, (0, 1, 2, 3), x90, (0, 1, 4, 3, 5, 6), (4, 2, 5, 6)) * -1.0
    rdm2_f_abab_voov += einsum(x239, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    x240 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x240 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 5, 4, 7, 2, 1), (6, 7, 3, 0))
    rdm2_f_abab_voov += einsum(x240, (0, 1, 2, 3), (3, 0, 2, 1)) * 4.0
    x241 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x241 += einsum(t2.abab, (0, 1, 2, 3), x85, (1, 4, 5, 0, 6, 2), (4, 3, 5, 6))
    rdm2_f_abab_voov += einsum(x241, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    x242 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x242 += einsum(l3.aaaaaa, (0, 1, 2, 3, 4, 5), t3.abaaba, (4, 6, 5, 1, 7, 2), (6, 7, 3, 0))
    rdm2_f_abab_voov += einsum(x242, (0, 1, 2, 3), (3, 0, 2, 1)) * 3.0
    x243 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x243 += einsum(x236, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.6666666666666666
    x243 += einsum(x237, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.6666666666666666
    x243 += einsum(x238, (0, 1, 2, 3), (0, 1, 2, 3))
    x243 += einsum(x239, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.6666666666666666
    x243 += einsum(x240, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.3333333333333333
    x243 += einsum(x241, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.6666666666666666
    x243 += einsum(x242, (0, 1, 2, 3), (0, 1, 2, 3))
    x243 += einsum(t1.bb, (0, 1), x25, (0, 2, 3, 4), (2, 1, 3, 4)) * -0.6666666666666666
    rdm2_f_abab_oovv += einsum(t2.aaaa, (0, 1, 2, 3), x243, (4, 5, 1, 3), (0, 4, 2, 5)) * 6.0
    del x243
    x244 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x244 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_bbbb_ovov += einsum(x244, (0, 1, 2, 3), (1, 2, 0, 3)) * -4.0
    rdm2_f_bbbb_ovvo += einsum(x244, (0, 1, 2, 3), (1, 2, 3, 0)) * 4.0
    rdm2_f_bbbb_voov += einsum(x244, (0, 1, 2, 3), (2, 1, 0, 3)) * 4.0
    rdm2_f_bbbb_vovo += einsum(x244, (0, 1, 2, 3), (2, 1, 3, 0)) * -4.0
    x245 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x245 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), (3, 4, 1, 5))
    rdm2_f_bbbb_ovov += einsum(x245, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_bbbb_ovvo += einsum(x245, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_bbbb_voov += einsum(x245, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_bbbb_vovo += einsum(x245, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x246 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x246 += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (6, 4, 5, 7, 1, 2), (3, 6, 0, 7))
    rdm2_f_bbbb_ovov += einsum(x246, (0, 1, 2, 3), (1, 2, 0, 3)) * -9.0
    rdm2_f_bbbb_ovvo += einsum(x246, (0, 1, 2, 3), (1, 2, 3, 0)) * 9.0
    rdm2_f_bbbb_voov += einsum(x246, (0, 1, 2, 3), (2, 1, 0, 3)) * 9.0
    rdm2_f_bbbb_vovo += einsum(x246, (0, 1, 2, 3), (2, 1, 3, 0)) * -9.0
    x247 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x247 += einsum(t2.bbbb, (0, 1, 2, 3), x114, (4, 1, 0, 5, 6, 3), (4, 5, 6, 2))
    rdm2_f_bbbb_ovov += einsum(x247, (0, 1, 2, 3), (1, 2, 0, 3)) * 6.0
    rdm2_f_bbbb_ovvo += einsum(x247, (0, 1, 2, 3), (1, 2, 3, 0)) * -6.0
    rdm2_f_bbbb_voov += einsum(x247, (0, 1, 2, 3), (2, 1, 0, 3)) * -6.0
    rdm2_f_bbbb_vovo += einsum(x247, (0, 1, 2, 3), (2, 1, 3, 0)) * 6.0
    x248 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x248 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 4, 5, 7, 1, 2), (3, 6, 0, 7))
    rdm2_f_bbbb_ovov += einsum(x248, (0, 1, 2, 3), (1, 2, 0, 3)) * -4.0
    rdm2_f_bbbb_ovvo += einsum(x248, (0, 1, 2, 3), (1, 2, 3, 0)) * 4.0
    rdm2_f_bbbb_voov += einsum(x248, (0, 1, 2, 3), (2, 1, 0, 3)) * 4.0
    rdm2_f_bbbb_vovo += einsum(x248, (0, 1, 2, 3), (2, 1, 3, 0)) * -4.0
    x249 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x249 += einsum(t2.abab, (0, 1, 2, 3), x90, (4, 1, 5, 6, 0, 2), (4, 5, 6, 3))
    rdm2_f_bbbb_ovov += einsum(x249, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_bbbb_ovvo += einsum(x249, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    rdm2_f_bbbb_voov += einsum(x249, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_bbbb_vovo += einsum(x249, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x250 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x250 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 6, 5, 0, 7, 2), (4, 6, 1, 7))
    rdm2_f_bbbb_ovov += einsum(x250, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_bbbb_ovvo += einsum(x250, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_bbbb_voov += einsum(x250, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_bbbb_vovo += einsum(x250, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x251 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x251 += einsum(x244, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    x251 += einsum(x245, (0, 1, 2, 3), (0, 1, 2, 3))
    x251 += einsum(x246, (0, 1, 2, 3), (0, 1, 2, 3)) * 9.0
    x251 += einsum(x247, (0, 1, 2, 3), (0, 1, 2, 3)) * -6.0
    x251 += einsum(x248, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    x251 += einsum(x249, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x251 += einsum(x250, (0, 1, 2, 3), (0, 1, 2, 3))
    x251 += einsum(t1.bb, (0, 1), x42, (2, 0, 3, 4), (2, 3, 4, 1)) * -6.0
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x251, (1, 4, 3, 5), (0, 4, 2, 5))
    del x251
    x252 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x252 += einsum(x173, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x252 += einsum(x174, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x252 += einsum(x178, (0, 1, 2, 3), (0, 1, 2, 3))
    x252 += einsum(x175, (0, 1, 2, 3), (0, 1, 2, 3)) * -4.5
    x252 += einsum(x179, (0, 1, 2, 3), (0, 1, 2, 3)) * 3.0
    x252 += einsum(x180, (0, 1, 2, 3), (0, 1, 3, 2))
    del x180
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x252, (0, 4, 2, 5), (4, 1, 5, 3)) * -2.0
    del x252
    x253 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x253 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3))
    del x20
    x253 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3))
    del x21
    x254 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x254 += einsum(x168, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.5
    x254 += einsum(x169, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x254 += einsum(x165, (0, 1, 2, 3), (0, 1, 2, 3))
    x254 += einsum(x170, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.5
    x254 += einsum(x183, (0, 1, 2, 3), (0, 1, 2, 3))
    x254 += einsum(t1.aa, (0, 1), x253, (2, 3, 0, 4), (2, 3, 4, 1))
    rdm2_f_abab_oovv += einsum(t2.bbbb, (0, 1, 2, 3), x254, (1, 3, 4, 5), (4, 0, 5, 2)) * -4.0
    del x254
    x255 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x255 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), (3, 4, 0, 5))
    rdm2_f_abab_vovo += einsum(x255, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x256 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x256 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (6, 4, 5, 0, 7, 2), (3, 6, 1, 7))
    rdm2_f_abab_vovo += einsum(x256, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    x257 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x257 += einsum(t2.abab, (0, 1, 2, 3), x90, (4, 1, 5, 3, 0, 6), (4, 5, 6, 2)) * -1.0
    rdm2_f_abab_vovo += einsum(x257, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    x258 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x258 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 6, 5, 7, 1, 2), (4, 6, 0, 7))
    rdm2_f_abab_vovo += einsum(x258, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    x259 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x259 += einsum(t2.aaaa, (0, 1, 2, 3), x85, (4, 5, 0, 1, 6, 3), (4, 5, 6, 2))
    del x85
    rdm2_f_abab_vovo += einsum(x259, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    x260 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x260 += einsum(x255, (0, 1, 2, 3), (0, 1, 2, 3))
    x260 += einsum(x256, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x260 += einsum(x257, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x260 += einsum(x258, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x260 += einsum(x259, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x260 += einsum(t1.aa, (0, 1), x25, (2, 3, 0, 4), (2, 3, 4, 1)) * 2.0
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x260, (1, 4, 2, 5), (0, 4, 5, 3))
    del x260
    x261 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x261 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 6, 5, 7, 1, 2), (0, 7, 4, 6))
    rdm2_f_abab_ovov += einsum(x261, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    x262 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x262 += einsum(t2.bbbb, (0, 1, 2, 3), x59, (0, 1, 4, 3, 5, 6), (4, 2, 5, 6))
    del x59
    rdm2_f_abab_ovov += einsum(x262, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    x263 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x263 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (6, 4, 5, 0, 7, 2), (1, 7, 3, 6))
    rdm2_f_abab_ovov += einsum(x263, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    x264 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x264 += einsum(t2.abab, (0, 1, 2, 3), x51, (1, 4, 5, 0, 6, 2), (4, 3, 5, 6)) * -1.0
    rdm2_f_abab_ovov += einsum(x264, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    x265 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x265 += einsum(x261, (0, 1, 2, 3), (0, 1, 2, 3))
    x265 += einsum(x262, (0, 1, 2, 3), (0, 1, 2, 3))
    x265 += einsum(x263, (0, 1, 2, 3), (0, 1, 2, 3))
    x265 += einsum(x264, (0, 1, 2, 3), (0, 1, 2, 3))
    x265 += einsum(t1.bb, (0, 1), x253, (0, 2, 3, 4), (2, 1, 3, 4))
    del x253
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x265, (3, 4, 0, 5), (5, 1, 2, 4)) * 2.0
    del x265
    x266 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x266 += einsum(t1.bb, (0, 1), x105, (2, 1, 3, 4), (2, 0, 3, 4))
    del x105
    x267 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x267 += einsum(x18, (0, 1, 2, 3), (0, 1, 2, 3))
    x267 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.9999999999999194
    x267 += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.9999999999999194
    x267 += einsum(x266, (0, 1, 2, 3), (0, 1, 2, 3))
    x267 += einsum(x26, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), x267, (1, 4, 0, 5), (5, 4, 2, 3))
    del x267
    x268 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    x268 += einsum(x18, (0, 1, 2, 3), (0, 1, 2, 3))
    del x18
    x268 += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.000000000000041
    del x17
    x268 += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.000000000000041
    del x16
    x268 += einsum(x266, (0, 1, 2, 3), (0, 1, 2, 3))
    del x266
    x268 += einsum(x26, (0, 1, 2, 3), (0, 1, 3, 2))
    del x26
    x269 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x269 += einsum(x96, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.3333333333333333
    del x96
    x269 += einsum(x98, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.6666666666666666
    del x98
    x269 += einsum(t2.abab, (0, 1, 2, 3), x103, (1, 4, 5, 2), (4, 3, 5, 0)) * 0.3333333333333333
    x269 += einsum(t2.bbbb, (0, 1, 2, 3), x19, (1, 3, 4, 5), (0, 2, 4, 5)) * -0.6666666666666666
    x269 += einsum(x97, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.6666666666666666
    del x97
    x269 += einsum(t2.abab, (0, 1, 2, 3), x1, (4, 0, 5, 2), (1, 3, 4, 5)) * 0.6666666666666666
    x269 += einsum(x91, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.6666666666666666
    del x91
    x269 += einsum(x92, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x92
    x269 += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.6666666666666666
    del x94
    x269 += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.6666666666666666
    del x86
    x269 += einsum(x87, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.3333333333333333
    del x87
    x269 += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.6666666666666666
    del x89
    x269 += einsum(x95, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x95
    x269 += einsum(t1.bb, (0, 1), x268, (0, 2, 3, 4), (2, 1, 3, 4)) * 0.3333333333333333
    del x268
    rdm2_f_abab_oovv += einsum(t1.aa, (0, 1), x269, (2, 3, 0, 4), (4, 2, 1, 3)) * 3.0
    del x269
    x270 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x270 += einsum(x207, (0, 1), (0, 1)) * 0.3333333333333468
    x270 += einsum(x208, (0, 1), (0, 1)) * 0.6666666666666936
    x270 += einsum(x209, (0, 1), (0, 1)) * 0.33333333333333354
    x270 += einsum(x210, (0, 1), (0, 1)) * 0.6666666666666667
    x270 += einsum(x211, (0, 1), (0, 1))
    rdm2_f_abab_oovv += einsum(x270, (0, 1), t2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -2.9999999999998788
    del x270
    x271 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x271 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4))
    x272 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x272 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    x273 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x273 += einsum(l3.bbbbbb, (0, 1, 2, 3, 4, 5), t3.bbbbbb, (3, 4, 5, 6, 1, 2), (0, 6))
    x274 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x274 += einsum(l3.babbab, (0, 1, 2, 3, 4, 5), t3.babbab, (3, 4, 5, 6, 1, 2), (0, 6))
    x275 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x275 += einsum(l3.abaaba, (0, 1, 2, 3, 4, 5), t3.abaaba, (3, 4, 5, 0, 6, 2), (1, 6))
    x276 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x276 += einsum(x271, (0, 1), (0, 1)) * 2.0
    x276 += einsum(x272, (0, 1), (0, 1))
    x276 += einsum(x273, (0, 1), (0, 1)) * 2.9999999999998788
    x276 += einsum(x274, (0, 1), (0, 1)) * 1.9999999999999194
    x276 += einsum(x275, (0, 1), (0, 1)) * 0.9999999999999601
    rdm2_f_abab_oovv += einsum(x276, (0, 1), t2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    del x276
    x277 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x277 += einsum(x152, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x152
    x277 += einsum(x151, (0, 1, 2, 3), (0, 1, 2, 3))
    del x151
    x277 += einsum(t2.abab, (0, 1, 2, 3), x38, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x277 += einsum(x160, (0, 1, 2, 3), (0, 1, 2, 3))
    del x160
    x277 += einsum(t2.aaaa, (0, 1, 2, 3), x103, (4, 5, 1, 3), (4, 5, 0, 2))
    x277 += einsum(t2.abab, (0, 1, 2, 3), x19, (4, 3, 0, 5), (4, 1, 5, 2)) * -0.5
    del x19
    x277 += einsum(x153, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.5
    del x153
    x277 += einsum(x155, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x155
    x277 += einsum(x154, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x154
    x277 += einsum(x156, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x156
    x277 += einsum(x158, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.5
    del x158
    x277 += einsum(x157, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x157
    x277 += einsum(x159, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x159
    rdm2_f_abab_oovv += einsum(t1.bb, (0, 1), x277, (0, 2, 3, 4), (3, 2, 4, 1)) * -2.0
    del x277
    x278 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x278 += einsum(x11, (0, 1), (0, 1)) * 0.3333333333333468
    del x11
    x278 += einsum(x5, (0, 1), (0, 1)) * 0.3333333333333468
    del x5
    x278 += einsum(x6, (0, 1), (0, 1)) * 0.6666666666666936
    del x6
    x278 += einsum(x7, (0, 1), (0, 1)) * 0.33333333333333354
    del x7
    x278 += einsum(x8, (0, 1), (0, 1)) * 0.6666666666666667
    del x8
    x278 += einsum(x9, (0, 1), (0, 1))
    del x9
    rdm2_f_abab_oovv += einsum(x278, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -2.9999999999998788
    del x278
    x279 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x279 += einsum(t2.bbbb, (0, 1, 2, 3), x249, (1, 4, 3, 5), (4, 0, 2, 5)) * -1.0
    x280 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x280 += einsum(t2.abab, (0, 1, 2, 3), x241, (4, 5, 0, 2), (4, 1, 3, 5))
    x281 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x281 += einsum(x245, (0, 1, 2, 3), (0, 1, 2, 3))
    x281 += einsum(x246, (0, 1, 2, 3), (0, 1, 2, 3)) * 9.0
    x281 += einsum(x248, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    x281 += einsum(x250, (0, 1, 2, 3), (0, 1, 2, 3))
    x282 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x282 += einsum(t2.bbbb, (0, 1, 2, 3), x281, (1, 4, 3, 5), (4, 0, 5, 2)) * 2.0
    del x281
    x283 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x283 += einsum(x238, (0, 1, 2, 3), (0, 1, 2, 3))
    x283 += einsum(x240, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.3333333333333333
    x283 += einsum(x242, (0, 1, 2, 3), (0, 1, 2, 3))
    x284 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x284 += einsum(t2.abab, (0, 1, 2, 3), x283, (4, 5, 0, 2), (4, 1, 5, 3)) * 3.0
    del x283
    x285 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x285 += einsum(t1.bb, (0, 1), x40, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    x286 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x286 += einsum(x247, (0, 1, 2, 3), (0, 1, 2, 3))
    x286 += einsum(x285, (0, 1, 2, 3), (0, 1, 2, 3))
    del x285
    x287 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x287 += einsum(t2.bbbb, (0, 1, 2, 3), x286, (1, 4, 3, 5), (4, 0, 5, 2)) * 12.0
    del x286
    x288 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x288 += einsum(t1.bb, (0, 1), x23, (0, 2, 3, 4), (2, 1, 3, 4))
    del x23
    x289 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x289 += einsum(x239, (0, 1, 2, 3), (0, 1, 2, 3))
    x289 += einsum(x288, (0, 1, 2, 3), (0, 1, 2, 3))
    del x288
    x290 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x290 += einsum(t2.abab, (0, 1, 2, 3), x289, (4, 5, 0, 2), (4, 1, 5, 3)) * 2.0
    del x289
    x291 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x291 += einsum(t2.bbbb, (0, 1, 2, 3), x38, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x292 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x292 += einsum(t2.abab, (0, 1, 2, 3), x103, (4, 5, 0, 2), (4, 5, 1, 3))
    x293 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x293 += einsum(t2.bbbb, (0, 1, 2, 3), x41, (1, 4, 5, 3), (4, 0, 5, 2)) * -1.0
    x294 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x294 += einsum(x291, (0, 1, 2, 3), (0, 1, 2, 3))
    del x291
    x294 += einsum(x292, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x292
    x294 += einsum(x132, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.25
    del x132
    x294 += einsum(x135, (0, 1, 2, 3), (0, 1, 2, 3))
    del x135
    x294 += einsum(x293, (0, 1, 2, 3), (0, 1, 2, 3))
    del x293
    x294 += einsum(x137, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x137
    x294 += einsum(x136, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x136
    x295 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x295 += einsum(t1.bb, (0, 1), x294, (0, 2, 3, 4), (2, 3, 4, 1)) * 4.0
    del x294
    x296 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x296 += einsum(x279, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x279
    x296 += einsum(x280, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x280
    x296 += einsum(x282, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x282
    x296 += einsum(x284, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x284
    x296 += einsum(x287, (0, 1, 2, 3), (0, 1, 2, 3))
    del x287
    x296 += einsum(x290, (0, 1, 2, 3), (0, 1, 2, 3))
    del x290
    x296 += einsum(x295, (0, 1, 2, 3), (0, 1, 3, 2))
    del x295
    x296 += einsum(t1.bb, (0, 1), x131, (2, 3), (0, 2, 1, 3)) * 2.0
    del x131
    rdm2_f_bbbb_oovv += einsum(x296, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_bbbb_oovv += einsum(x296, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_bbbb_oovv += einsum(x296, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_bbbb_oovv += einsum(x296, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x296
    x297 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x297 += einsum(x42, (0, 1, 2, 3), t3.bbbbbb, (4, 0, 1, 5, 6, 3), (4, 2, 5, 6)) * -18.0
    x298 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x298 += einsum(x25, (0, 1, 2, 3), t3.babbab, (4, 2, 0, 5, 3, 6), (4, 1, 5, 6)) * -4.0
    del x25
    x299 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x299 += einsum(x31, (0, 1), (0, 1)) * 3.000000000000004
    del x31
    x299 += einsum(x32, (0, 1), (0, 1)) * 1.9999999999999996
    del x32
    x299 += einsum(x33, (0, 1), (0, 1))
    del x33
    x300 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x300 += einsum(x299, (0, 1), t2.bbbb, (2, 0, 3, 4), (1, 2, 3, 4)) * -1.9999999999999194
    del x299
    x301 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x301 += einsum(x297, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x297
    x301 += einsum(x298, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x298
    x301 += einsum(x300, (0, 1, 2, 3), (1, 0, 3, 2))
    del x300
    rdm2_f_bbbb_oovv += einsum(x301, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_bbbb_oovv += einsum(x301, (0, 1, 2, 3), (1, 0, 2, 3))
    del x301
    x302 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x302 += einsum(t2.bbbb, (0, 1, 2, 3), x244, (1, 4, 3, 5), (4, 0, 5, 2))
    x303 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x303 += einsum(t2.abab, (0, 1, 2, 3), x237, (4, 5, 0, 2), (4, 1, 5, 3))
    x304 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x304 += einsum(x145, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x304 += einsum(x146, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.3333333333333333
    x305 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x305 += einsum(x304, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 0, 6, 1, 2), (4, 5, 3, 6)) * -18.0
    x306 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x306 += einsum(x101, (0, 1, 2, 3), t3.babbab, (4, 2, 5, 6, 3, 0), (4, 5, 1, 6)) * -4.0
    del x101
    x307 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x307 += einsum(x271, (0, 1), (0, 1))
    x307 += einsum(x272, (0, 1), (0, 1)) * 0.5
    x307 += einsum(x273, (0, 1), (0, 1)) * 1.4999999999999416
    x307 += einsum(x274, (0, 1), (0, 1)) * 0.9999999999999595
    x307 += einsum(x275, (0, 1), (0, 1)) * 0.49999999999997985
    x308 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x308 += einsum(x307, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 1, 4)) * -4.0
    del x307
    x309 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x309 += einsum(t2.bbbb, (0, 1, 2, 3), x304, (4, 2, 3, 5), (4, 0, 1, 5)) * 3.0
    x310 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x310 += einsum(x121, (0, 1, 2, 3), (0, 2, 1, 3))
    del x121
    x310 += einsum(x124, (0, 1, 2, 3), (0, 2, 1, 3)) * 3.0
    del x124
    x310 += einsum(x125, (0, 1, 2, 3), (0, 2, 1, 3))
    del x125
    x310 += einsum(x127, (0, 1, 2, 3), (0, 2, 1, 3)) * 3.0
    del x127
    x310 += einsum(x123, (0, 1, 2, 3), (0, 2, 1, 3))
    del x123
    x310 += einsum(x309, (0, 1, 2, 3), (0, 2, 1, 3))
    del x309
    x311 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x311 += einsum(t1.bb, (0, 1), x310, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    del x310
    x312 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x312 += einsum(x302, (0, 1, 2, 3), (0, 1, 2, 3)) * 8.0
    del x302
    x312 += einsum(x303, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x303
    x312 += einsum(x305, (0, 1, 2, 3), (0, 1, 2, 3))
    del x305
    x312 += einsum(x306, (0, 1, 2, 3), (1, 0, 2, 3))
    del x306
    x312 += einsum(x308, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x308
    x312 += einsum(x311, (0, 1, 2, 3), (1, 0, 3, 2))
    del x311
    rdm2_f_bbbb_oovv += einsum(x312, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_oovv += einsum(x312, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x312
    x313 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x313 += einsum(x28, (0, 1), t2.bbbb, (2, 0, 3, 4), (1, 2, 3, 4))
    del x28
    x314 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x314 += einsum(x38, (0, 1, 2, 3), t3.bbbbbb, (4, 1, 0, 5, 6, 3), (2, 4, 5, 6))
    x315 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x315 += einsum(x103, (0, 1, 2, 3), t3.babbab, (4, 2, 0, 5, 3, 6), (1, 4, 5, 6))
    del x103
    x316 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x316 += einsum(t1.bb, (0, 1), x42, (2, 3, 4, 1), (0, 2, 3, 4)) * 3.0
    x317 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x317 += einsum(t2.bbbb, (0, 1, 2, 3), x316, (4, 0, 1, 5), (4, 5, 2, 3)) * 2.0
    del x316
    x318 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x318 += einsum(x29, (0, 1), (0, 1))
    del x29
    x318 += einsum(x30, (0, 1), (0, 1)) * 0.5
    del x30
    x319 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x319 += einsum(x318, (0, 1), t2.bbbb, (2, 0, 3, 4), (1, 2, 3, 4)) * -4.0
    del x318
    x320 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x320 += einsum(t1.bb, (0, 1), x42, (2, 3, 4, 1), (0, 2, 3, 4))
    del x42
    x321 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x321 += einsum(t1.bb, (0, 1), x320, (2, 3, 0, 4), (3, 2, 4, 1))
    del x320
    x322 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x322 += einsum(t1.bb, (0, 1), x321, (0, 2, 3, 4), (2, 3, 4, 1)) * 6.0
    del x321
    x323 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x323 += einsum(x313, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x313
    x323 += einsum(x314, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.0
    del x314
    x323 += einsum(x315, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x315
    x323 += einsum(x317, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x317
    x323 += einsum(x319, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x319
    x323 += einsum(x322, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x322
    rdm2_f_bbbb_oovv += einsum(x323, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_oovv += einsum(x323, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x323
    x324 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    x324 += einsum(t2.bbbb, (0, 1, 2, 3), l3.babbab, (2, 4, 3, 5, 6, 7), (5, 7, 0, 1, 6, 4))
    x324 += einsum(x122, (0, 1, 2, 3, 4, 5), (0, 1, 3, 2, 4, 5)) * -1.0000000000000606
    del x122
    rdm2_f_bbbb_oovv += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x324, (0, 2, 6, 7, 1, 4), (6, 7, 3, 5)) * 1.9999999999999194
    del x324
    x325 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x325 += einsum(t2.bbbb, (0, 1, 2, 3), l3.bbbbbb, (4, 2, 3, 5, 6, 7), (5, 6, 7, 0, 1, 4))
    x325 += einsum(x126, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0000000000000584
    del x126
    rdm2_f_bbbb_oovv += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x325, (0, 1, 2, 6, 7, 5), (6, 7, 3, 4)) * 5.999999999999766
    del x325
    x326 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x326 += einsum(x37, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x326 += einsum(x39, (0, 1, 2, 3), (0, 1, 3, 2))
    x326 += einsum(x35, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.999999999999883
    x326 += einsum(x36, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.9999999999999597
    rdm2_f_bbbb_oovv += einsum(t2.bbbb, (0, 1, 2, 3), x326, (0, 1, 4, 5), (5, 4, 2, 3)) * 2.0
    del x326
    x327 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x327 += einsum(x37, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.3333333333333269
    del x37
    x327 += einsum(x39, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.3333333333333269
    del x39
    x327 += einsum(x35, (0, 1, 2, 3), (0, 1, 3, 2))
    del x35
    x327 += einsum(x36, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.3333333333333336
    del x36
    x328 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x328 += einsum(t1.bb, (0, 1), x327, (0, 2, 3, 4), (2, 4, 3, 1)) * -1.0
    del x327
    rdm2_f_bbbb_oovv += einsum(t1.bb, (0, 1), x328, (0, 2, 3, 4), (2, 3, 4, 1)) * -6.000000000000116
    del x328
    x329 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x329 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_aaaa_ovov += einsum(x329, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_aaaa_ovvo += einsum(x329, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_aaaa_voov += einsum(x329, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_aaaa_vovo += einsum(x329, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x330 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x330 += einsum(t1.aa, (0, 1), x64, (0, 2, 3, 4), (2, 3, 1, 4)) * 6.0
    del x64
    rdm2_f_aaaa_ovov += einsum(x330, (0, 1, 2, 3), (1, 3, 0, 2)) * -1.0
    rdm2_f_aaaa_voov += einsum(x330, (0, 1, 2, 3), (3, 1, 0, 2))
    del x330
    x331 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x331 += einsum(l1.aa, (0, 1), t1.aa, (1, 2), (0, 2))
    x332 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x332 += einsum(x331, (0, 1), (0, 1))
    x332 += einsum(x207, (0, 1), (0, 1))
    x332 += einsum(x208, (0, 1), (0, 1)) * 2.0
    x332 += einsum(x209, (0, 1), (0, 1)) * 0.9999999999999601
    x332 += einsum(x210, (0, 1), (0, 1)) * 1.9999999999999194
    x332 += einsum(x211, (0, 1), (0, 1)) * 2.9999999999998788
    rdm2_f_aaaa_ovov += einsum(delta.aa.oo, (0, 1), x332, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_ovvo += einsum(delta.aa.oo, (0, 1), x332, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_voov += einsum(delta.aa.oo, (0, 1), x332, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_aaaa_vovo += einsum(delta.aa.oo, (0, 1), x332, (2, 3), (2, 0, 3, 1))
    x333 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x333 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), (1, 5, 2, 4))
    rdm2_f_abab_ovov += einsum(x333, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    x334 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x334 += einsum(l1.bb, (0, 1), t1.bb, (1, 2), (0, 2))
    x335 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x335 += einsum(x334, (0, 1), (0, 1))
    x335 += einsum(x271, (0, 1), (0, 1)) * 2.0
    x335 += einsum(x272, (0, 1), (0, 1))
    x335 += einsum(x273, (0, 1), (0, 1)) * 2.9999999999998788
    x335 += einsum(x274, (0, 1), (0, 1)) * 1.9999999999999194
    x335 += einsum(x275, (0, 1), (0, 1)) * 0.9999999999999601
    rdm2_f_abab_ovov += einsum(delta.aa.oo, (0, 1), x335, (2, 3), (0, 2, 1, 3))
    rdm2_f_abab_ovvv += einsum(t1.aa, (0, 1), x335, (2, 3), (0, 2, 1, 3))
    x336 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x336 += einsum(t1.bb, (0, 1), x118, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    del x118
    rdm2_f_bbbb_ovov += einsum(x336, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_bbbb_ovvo += einsum(x336, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_bbbb_voov += einsum(x336, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_bbbb_vovo += einsum(x336, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x336
    x337 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x337 += einsum(x334, (0, 1), (0, 1)) * 0.5000000000000202
    del x334
    x337 += einsum(x271, (0, 1), (0, 1)) * 1.0000000000000404
    del x271
    x337 += einsum(x272, (0, 1), (0, 1)) * 0.5000000000000202
    del x272
    x337 += einsum(x273, (0, 1), (0, 1)) * 1.4999999999999998
    del x273
    x337 += einsum(x274, (0, 1), (0, 1))
    del x274
    x337 += einsum(x275, (0, 1), (0, 1)) * 0.5000000000000002
    del x275
    rdm2_f_bbbb_ovov += einsum(delta.bb.oo, (0, 1), x337, (2, 3), (0, 2, 1, 3)) * 1.9999999999999194
    rdm2_f_bbbb_ovvo += einsum(delta.bb.oo, (0, 1), x337, (2, 3), (0, 2, 3, 1)) * -1.9999999999999194
    rdm2_f_bbbb_voov += einsum(delta.bb.oo, (0, 1), x337, (2, 3), (2, 0, 1, 3)) * -1.9999999999999194
    rdm2_f_bbbb_vovo += einsum(delta.bb.oo, (0, 1), x337, (2, 3), (2, 0, 3, 1)) * 1.9999999999999194
    del x337
    x338 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x338 += einsum(t1.aa, (0, 1), x102, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    rdm2_f_aaaa_ovvo += einsum(x338, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_aaaa_vovo += einsum(x338, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    del x338
    x339 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x339 += einsum(x331, (0, 1), (0, 1)) * 1.00000000000004
    x339 += einsum(x207, (0, 1), (0, 1)) * 1.00000000000004
    x339 += einsum(x208, (0, 1), (0, 1)) * 2.00000000000008
    x339 += einsum(x209, (0, 1), (0, 1))
    x339 += einsum(x210, (0, 1), (0, 1)) * 1.9999999999999991
    x339 += einsum(x211, (0, 1), (0, 1)) * 2.9999999999999982
    rdm2_f_abab_vovo += einsum(delta.bb.oo, (0, 1), x339, (2, 3), (2, 0, 3, 1)) * 0.9999999999999601
    del x339
    x340 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x340 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x51, (1, 4, 0, 2, 6, 7), (6, 7, 3, 5))
    del x51
    rdm2_f_aaaa_ovvv = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_ovvv += einsum(x340, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0000000000000404
    rdm2_f_aaaa_vovv = np.zeros((nvir[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_aaaa_vovv += einsum(x340, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0000000000000404
    del x340
    x341 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x341 += einsum(l2.aaaa, (0, 1, 2, 3), t3.aaaaaa, (4, 2, 3, 5, 6, 1), (4, 0, 5, 6))
    rdm2_f_aaaa_ovvv += einsum(x341, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.0
    rdm2_f_aaaa_vovv += einsum(x341, (0, 1, 2, 3), (1, 0, 3, 2)) * 6.0
    del x341
    x342 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x342 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_aaaa_ovvv += einsum(x342, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_aaaa_vovv += einsum(x342, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x342
    x343 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x343 += einsum(l2.abab, (0, 1, 2, 3), t3.abaaba, (4, 3, 2, 5, 1, 6), (4, 0, 5, 6))
    rdm2_f_aaaa_ovvv += einsum(x343, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_aaaa_vovv += einsum(x343, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x343
    x344 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x344 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x45, (0, 1, 2, 6, 7, 5), (6, 7, 3, 4)) * -1.0
    del x45
    rdm2_f_aaaa_ovvv += einsum(x344, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.000000000000116
    rdm2_f_aaaa_vovv += einsum(x344, (0, 1, 2, 3), (1, 0, 3, 2)) * 6.000000000000116
    del x344
    x345 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x345 += einsum(t2.abab, (0, 1, 2, 3), x161, (1, 3, 4, 5), (0, 4, 2, 5))
    x346 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x346 += einsum(t2.abab, (0, 1, 2, 3), x162, (1, 3, 4, 5), (0, 4, 5, 2))
    x347 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x347 += einsum(t2.aaaa, (0, 1, 2, 3), x205, (1, 4, 3, 5), (0, 4, 5, 2)) * 4.0
    x348 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x348 += einsum(x329, (0, 1, 2, 3), (0, 1, 2, 3))
    del x329
    x348 += einsum(x202, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x202
    x348 += einsum(x173, (0, 1, 2, 3), (0, 1, 2, 3))
    del x173
    x348 += einsum(x174, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    del x174
    x348 += einsum(x178, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x178
    x348 += einsum(x175, (0, 1, 2, 3), (0, 1, 2, 3)) * 9.0
    del x175
    x348 += einsum(x179, (0, 1, 2, 3), (0, 1, 2, 3)) * -6.0
    del x179
    x349 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x349 += einsum(t1.aa, (0, 1), x348, (0, 2, 3, 4), (2, 3, 4, 1))
    del x348
    x350 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x350 += einsum(x345, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x345
    x350 += einsum(x346, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x346
    x350 += einsum(x347, (0, 1, 2, 3), (0, 1, 2, 3))
    del x347
    x350 += einsum(x349, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x349
    x350 += einsum(t1.aa, (0, 1), x332, (2, 3), (0, 2, 1, 3))
    del x332
    rdm2_f_aaaa_ovvv += einsum(x350, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_aaaa_ovvv += einsum(x350, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_aaaa_vovv += einsum(x350, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_aaaa_vovv += einsum(x350, (0, 1, 2, 3), (1, 0, 3, 2))
    del x350
    x351 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x351 += einsum(t2.aaaa, (0, 1, 2, 3), x102, (0, 1, 4, 5), (4, 5, 2, 3)) * 2.0
    del x102
    rdm2_f_aaaa_ovvv += einsum(x351, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_aaaa_vovv += einsum(x351, (0, 1, 2, 3), (1, 0, 3, 2))
    del x351
    x352 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x352 += einsum(x1, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.3333333333333333
    del x1
    x352 += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.3333333333333333
    del x12
    x352 += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x13
    x353 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x353 += einsum(t1.aa, (0, 1), x352, (0, 2, 3, 4), (2, 3, 4, 1)) * 3.0
    del x352
    x354 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x354 += einsum(t1.aa, (0, 1), x353, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    del x353
    rdm2_f_aaaa_ovvv += einsum(x354, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_aaaa_vovv += einsum(x354, (0, 1, 2, 3), (1, 0, 2, 3))
    del x354
    x355 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x355 += einsum(x218, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.6666666666666666
    del x218
    x355 += einsum(x167, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.6666666666666666
    del x167
    x355 += einsum(x168, (0, 1, 2, 3), (0, 1, 2, 3))
    del x168
    x355 += einsum(x169, (0, 1, 2, 3), (0, 1, 2, 3)) * 1.3333333333333333
    del x169
    x355 += einsum(x165, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.6666666666666666
    del x165
    x355 += einsum(x170, (0, 1, 2, 3), (0, 1, 2, 3))
    del x170
    x355 += einsum(x183, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.6666666666666666
    del x183
    x355 += einsum(t1.aa, (0, 1), x22, (2, 3, 0, 4), (2, 3, 4, 1)) * -0.6666666666666666
    del x22
    rdm2_f_abab_ovvv += einsum(t1.bb, (0, 1), x355, (0, 2, 3, 4), (3, 2, 4, 1)) * 3.0
    del x355
    x356 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    x356 += einsum(x333, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x333
    x356 += einsum(x261, (0, 1, 2, 3), (0, 1, 2, 3))
    del x261
    x356 += einsum(x262, (0, 1, 2, 3), (0, 1, 2, 3))
    del x262
    x356 += einsum(x263, (0, 1, 2, 3), (0, 1, 2, 3))
    del x263
    x356 += einsum(x264, (0, 1, 2, 3), (0, 1, 2, 3))
    del x264
    rdm2_f_abab_ovvv += einsum(t1.aa, (0, 1), x356, (2, 3, 0, 4), (4, 2, 1, 3)) * -2.0
    del x356
    x357 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x357 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x90, (0, 2, 6, 7, 1, 4), (6, 7, 3, 5))
    del x90
    rdm2_f_bbbb_ovvv = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_ovvv += einsum(x357, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0000000000000404
    rdm2_f_bbbb_vovv = np.zeros((nvir[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_bbbb_vovv += einsum(x357, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0000000000000404
    del x357
    x358 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x358 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x114, (0, 1, 2, 6, 7, 5), (6, 7, 3, 4)) * -1.0
    del x114
    rdm2_f_bbbb_ovvv += einsum(x358, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.000000000000116
    rdm2_f_bbbb_vovv += einsum(x358, (0, 1, 2, 3), (1, 0, 3, 2)) * 6.000000000000116
    del x358
    x359 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x359 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_bbbb_ovvv += einsum(x359, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_bbbb_vovv += einsum(x359, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x359
    x360 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x360 += einsum(l2.bbbb, (0, 1, 2, 3), t3.bbbbbb, (4, 2, 3, 5, 6, 1), (4, 0, 5, 6))
    rdm2_f_bbbb_ovvv += einsum(x360, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.0
    rdm2_f_bbbb_vovv += einsum(x360, (0, 1, 2, 3), (1, 0, 3, 2)) * 6.0
    del x360
    x361 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x361 += einsum(l2.abab, (0, 1, 2, 3), t3.babbab, (4, 2, 3, 5, 0, 6), (4, 1, 5, 6))
    rdm2_f_bbbb_ovvv += einsum(x361, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_bbbb_vovv += einsum(x361, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x361
    x362 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x362 += einsum(t2.bbbb, (0, 1, 2, 3), x145, (1, 4, 3, 5), (0, 4, 5, 2)) * -1.0
    del x145
    x363 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x363 += einsum(t2.bbbb, (0, 1, 2, 3), x146, (1, 4, 3, 5), (0, 4, 2, 5))
    del x146
    x364 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x364 += einsum(t2.abab, (0, 1, 2, 3), x99, (4, 5, 0, 2), (1, 4, 5, 3))
    del x99
    x365 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x365 += einsum(t2.abab, (0, 1, 2, 3), x100, (4, 5, 0, 2), (1, 4, 3, 5))
    del x100
    x366 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x366 += einsum(x244, (0, 1, 2, 3), (0, 1, 2, 3))
    del x244
    x366 += einsum(x245, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x245
    x366 += einsum(x246, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.25
    del x246
    x366 += einsum(x247, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.5
    del x247
    x366 += einsum(x248, (0, 1, 2, 3), (0, 1, 2, 3))
    del x248
    x366 += einsum(x249, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x249
    x366 += einsum(x250, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    del x250
    x367 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x367 += einsum(t1.bb, (0, 1), x366, (0, 2, 3, 4), (2, 3, 4, 1)) * 4.0
    del x366
    x368 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x368 += einsum(x362, (0, 1, 2, 3), (0, 1, 2, 3)) * -12.0
    del x362
    x368 += einsum(x363, (0, 1, 2, 3), (0, 1, 2, 3)) * -4.0
    del x363
    x368 += einsum(x364, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x364
    x368 += einsum(x365, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x365
    x368 += einsum(x367, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x367
    x368 += einsum(t1.bb, (0, 1), x335, (2, 3), (0, 2, 1, 3))
    del x335
    rdm2_f_bbbb_ovvv += einsum(x368, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_ovvv += einsum(x368, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_bbbb_vovv += einsum(x368, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_bbbb_vovv += einsum(x368, (0, 1, 2, 3), (1, 0, 3, 2))
    del x368
    x369 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x369 += einsum(x38, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.3333333333333333
    x369 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3))
    x369 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.3333333333333333
    x370 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x370 += einsum(t2.bbbb, (0, 1, 2, 3), x369, (0, 1, 4, 5), (4, 5, 2, 3)) * 6.0
    del x369
    rdm2_f_bbbb_ovvv += einsum(x370, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_bbbb_vovv += einsum(x370, (0, 1, 2, 3), (1, 0, 3, 2))
    del x370
    x371 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x371 += einsum(x38, (0, 1, 2, 3), (1, 0, 2, 3))
    del x38
    x371 += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3)) * -3.0
    del x40
    x371 += einsum(x41, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x41
    x372 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x372 += einsum(t1.bb, (0, 1), x371, (0, 2, 3, 4), (2, 3, 4, 1)) * 0.3333333333333333
    rdm2_f_bbbb_ovvv += einsum(t1.bb, (0, 1), x372, (0, 2, 3, 4), (2, 3, 4, 1)) * -6.0
    del x372
    x373 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    x373 += einsum(x236, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x236
    x373 += einsum(x237, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x237
    x373 += einsum(x238, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.75
    del x238
    x373 += einsum(x239, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x239
    x373 += einsum(x240, (0, 1, 2, 3), (0, 1, 2, 3))
    del x240
    x373 += einsum(x241, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x241
    x373 += einsum(x242, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.75
    del x242
    x373 += einsum(t1.bb, (0, 1), x235, (0, 2, 3, 4), (2, 1, 3, 4)) * -0.25
    del x235
    rdm2_f_abab_vovv += einsum(t1.aa, (0, 1), x373, (2, 3, 0, 4), (4, 2, 1, 3)) * 4.0
    del x373
    x374 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x374 += einsum(x255, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x255
    x374 += einsum(x256, (0, 1, 2, 3), (0, 1, 2, 3))
    del x256
    x374 += einsum(x257, (0, 1, 2, 3), (0, 1, 2, 3))
    del x257
    x374 += einsum(x258, (0, 1, 2, 3), (0, 1, 2, 3))
    del x258
    x374 += einsum(x259, (0, 1, 2, 3), (0, 1, 2, 3))
    del x259
    rdm2_f_abab_vovv += einsum(t1.bb, (0, 1), x374, (0, 2, 3, 4), (3, 2, 4, 1)) * -2.0
    del x374
    x375 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x375 += einsum(x331, (0, 1), (0, 1)) * 0.5000000000000202
    del x331
    x375 += einsum(x207, (0, 1), (0, 1)) * 0.5000000000000202
    del x207
    x375 += einsum(x208, (0, 1), (0, 1)) * 1.0000000000000404
    del x208
    x375 += einsum(x209, (0, 1), (0, 1)) * 0.5000000000000002
    del x209
    x375 += einsum(x210, (0, 1), (0, 1))
    del x210
    x375 += einsum(x211, (0, 1), (0, 1)) * 1.4999999999999998
    del x211
    rdm2_f_abab_vovv += einsum(t1.bb, (0, 1), x375, (2, 3), (2, 0, 3, 1)) * 1.9999999999999194
    del x375
    x376 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x376 += einsum(t1.bb, (0, 1), x371, (0, 2, 3, 4), (2, 3, 4, 1))
    del x371
    rdm2_f_bbbb_vovv += einsum(t1.bb, (0, 1), x376, (0, 2, 3, 4), (3, 2, 1, 4)) * -2.0
    del x376
    x377 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x377 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_aaaa_vvov += einsum(x377, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_aaaa_vvvo += einsum(x377, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    rdm2_f_aaaa_vvvv += einsum(t1.aa, (0, 1), x377, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    del x377
    x378 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x378 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_bbbb_vvov += einsum(x378, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_bbbb_vvvo += einsum(x378, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    rdm2_f_bbbb_vvvv += einsum(t1.bb, (0, 1), x378, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    del x378
    x379 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x379 += einsum(t1.aa, (0, 1), l2.abab, (2, 3, 0, 4), (4, 3, 2, 1))
    rdm2_f_abab_vvvo += einsum(x379, (0, 1, 2, 3), (2, 1, 3, 0))
    x380 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x380 += einsum(t1.aa, (0, 1), x205, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    del x205
    rdm2_f_aaaa_vvvv += einsum(x380, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_aaaa_vvvv += einsum(x380, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x380
    x381 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    x381 += einsum(x379, (0, 1, 2, 3), (0, 1, 2, 3))
    del x379
    x381 += einsum(x161, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x161
    x381 += einsum(x162, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x162
    rdm2_f_abab_vvvv += einsum(t1.bb, (0, 1), x381, (0, 2, 3, 4), (3, 2, 4, 1))
    del x381
    x382 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x382 += einsum(t1.bb, (0, 1), x304, (0, 2, 3, 4), (2, 3, 4, 1)) * 6.0
    del x304
    rdm2_f_bbbb_vvvv += einsum(x382, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_bbbb_vvvv += einsum(x382, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x382

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

