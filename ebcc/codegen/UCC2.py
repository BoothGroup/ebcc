# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 2), ()) * -1.0
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 2, 1, 3), ())
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x0 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x1 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x1 += einsum(f.bb.ov, (0, 1), (0, 1))
    x1 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    x1 += einsum(t1.bb, (0, 1), x0, (0, 2, 3, 1), (2, 3)) * -0.5
    e_cc += einsum(t1.bb, (0, 1), x1, (0, 1), ())
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x3 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x3 += einsum(f.aa.ov, (0, 1), (0, 1)) * 2.0
    x3 += einsum(t1.aa, (0, 1), x2, (0, 2, 3, 1), (2, 3)) * -1.0
    e_cc += einsum(t1.aa, (0, 1), x3, (0, 1), ()) * 0.5

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    t1new = Namespace()
    t2new = Namespace()

    # T amplitudes
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    t1new_aa += einsum(f.aa.ov, (0, 1), (0, 1))
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    t1new_bb += einsum(f.bb.ov, (0, 1), (0, 1))
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    t2new_abab += einsum(f.aa.vv, (0, 1), t2.abab, (2, 3, 1, 4), (2, 3, 0, 4))
    t2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_abab += einsum(f.bb.vv, (0, 1), t2.abab, (2, 3, 4, 1), (2, 3, 4, 0))
    t2new_abab += einsum(t1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (2, 0, 3, 4))
    x0 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x0 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    t1new_aa += einsum(x0, (0, 1), (0, 1))
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x1 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x2 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x2 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x2 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_aa += einsum(t2.aaaa, (0, 1, 2, 3), x2, (4, 0, 1, 3), (4, 2)) * 2.0
    x3 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x3 += einsum(t1.aa, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (3, 4, 0, 2))
    x4 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x4 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x4 += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2))
    t1new_aa += einsum(t2.abab, (0, 1, 2, 3), x4, (1, 3, 0, 4), (4, 2)) * -1.0
    x5 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x5 += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    x5 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x5, (0, 4, 1, 3), (4, 2)) * 2.0
    x6 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x6 += einsum(t2.abab, (0, 1, 2, 3), (1, 3, 0, 2))
    x6 += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (2, 3, 0, 1))
    t1new_aa += einsum(v.aabb.vvov, (0, 1, 2, 3), x6, (2, 3, 4, 1), (4, 0))
    t1new_bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), x6, (4, 3, 0, 1), (4, 2))
    x7 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x7 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x7 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x8 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x8 += einsum(t1.aa, (0, 1), x7, (0, 2, 3, 1), (2, 3))
    x9 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x9 += einsum(f.aa.ov, (0, 1), (0, 1))
    x9 += einsum(x0, (0, 1), (0, 1))
    x9 += einsum(x8, (0, 1), (0, 1)) * -1.0
    t1new_aa += einsum(x9, (0, 1), t2.aaaa, (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_bb += einsum(x9, (0, 1), t2.abab, (0, 2, 1, 3), (2, 3))
    x10 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x10 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(x10, (0, 1), (0, 1))
    x11 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x11 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x11 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x12 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x12 += einsum(t1.bb, (0, 1), x11, (0, 2, 3, 1), (2, 3))
    x13 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x13 += einsum(f.bb.ov, (0, 1), (0, 1))
    x13 += einsum(x10, (0, 1), (0, 1))
    x13 += einsum(x12, (0, 1), (0, 1)) * -1.0
    t1new_aa += einsum(x13, (0, 1), t2.abab, (2, 0, 3, 1), (2, 3))
    t1new_bb += einsum(x13, (0, 1), t2.bbbb, (2, 0, 3, 1), (2, 3)) * 2.0
    x14 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x14 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x14 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_aa += einsum(t1.aa, (0, 1), x14, (0, 2, 1, 3), (2, 3)) * -1.0
    x15 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x15 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x15 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x16 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x16 += einsum(f.aa.oo, (0, 1), (0, 1)) * 0.5
    x16 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3)) * 0.5
    x16 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (4, 0)) * 0.5
    x16 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (4, 0)) * -1.0
    x16 += einsum(t1.aa, (0, 1), x15, (0, 2, 3, 1), (3, 2)) * -0.5
    x16 += einsum(t1.aa, (0, 1), x9, (2, 1), (2, 0)) * 0.5
    t1new_aa += einsum(t1.aa, (0, 1), x16, (0, 2), (2, 1)) * -2.0
    x17 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x17 += einsum(f.aa.vv, (0, 1), (0, 1))
    x17 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (0, 2, 3, 1), (2, 3)) * -1.0
    t1new_aa += einsum(t1.aa, (0, 1), x17, (1, 2), (0, 2))
    x18 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x18 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x18 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (4, 0, 2, 3))
    t1new_bb += einsum(t2.abab, (0, 1, 2, 3), x18, (1, 4, 0, 2), (4, 3)) * -1.0
    x19 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x19 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x20 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x20 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x20 += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    t1new_bb += einsum(t2.bbbb, (0, 1, 2, 3), x20, (4, 1, 0, 3), (4, 2)) * -2.0
    x21 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x21 += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    x21 += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3)) * 0.5
    t1new_bb += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x21, (0, 4, 3, 1), (4, 2)) * -2.0
    x22 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x22 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x22 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new_bb += einsum(t1.bb, (0, 1), x22, (0, 2, 1, 3), (2, 3)) * -1.0
    x23 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x23 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x23 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x24 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x24 += einsum(f.bb.oo, (0, 1), (0, 1))
    x24 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3))
    x24 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (4, 0)) * -2.0
    x24 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (4, 1))
    x24 += einsum(t1.bb, (0, 1), x23, (0, 2, 3, 1), (3, 2)) * -1.0
    x24 += einsum(t1.bb, (0, 1), x13, (2, 1), (2, 0))
    t1new_bb += einsum(t1.bb, (0, 1), x24, (0, 2), (2, 1)) * -1.0
    x25 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x25 += einsum(f.bb.vv, (0, 1), (0, 1))
    x25 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_bb += einsum(t1.bb, (0, 1), x25, (1, 2), (0, 2))
    x26 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x26 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x27 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x27 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x28 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x28 += einsum(t1.aa, (0, 1), x27, (2, 3, 4, 0), (2, 4, 3, 1))
    x29 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x29 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x29 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x29 += einsum(x26, (0, 1, 2, 3), (0, 1, 2, 3))
    x30 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x30 += einsum(t1.aa, (0, 1), x29, (2, 3, 1, 4), (0, 2, 3, 4))
    x31 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x31 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x31 += einsum(x28, (0, 1, 2, 3), (0, 2, 1, 3))
    x31 += einsum(x30, (0, 1, 2, 3), (0, 2, 1, 3))
    x32 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x32 += einsum(t1.aa, (0, 1), x31, (2, 0, 3, 4), (2, 3, 1, 4))
    x33 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x33 += einsum(x26, (0, 1, 2, 3), (0, 1, 2, 3))
    x33 += einsum(x32, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    t2new_aaaa += einsum(x33, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x33, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa += einsum(x33, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x33, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x34 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x34 += einsum(f.aa.ov, (0, 1), t1.aa, (2, 1), (0, 2))
    x35 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x35 += einsum(f.aa.oo, (0, 1), (0, 1))
    x35 += einsum(x34, (0, 1), (0, 1))
    t2new_abab += einsum(x35, (0, 1), t2.abab, (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    x36 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x36 += einsum(x35, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    t2new_aaaa += einsum(x36, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_aaaa += einsum(x36, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x37 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x37 += einsum(t1.aa, (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x38 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x38 += einsum(t1.aa, (0, 1), x37, (2, 3, 1, 4), (0, 2, 3, 4))
    x39 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x39 += einsum(f.aa.ov, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4))
    x40 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x40 += einsum(t1.aa, (0, 1), x1, (2, 3, 4, 1), (2, 0, 4, 3))
    x41 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x41 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x41 += einsum(x40, (0, 1, 2, 3), (3, 1, 2, 0))
    x42 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x42 += einsum(t1.aa, (0, 1), x41, (0, 2, 3, 4), (2, 3, 4, 1))
    x43 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x43 += einsum(x39, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x43 += einsum(x42, (0, 1, 2, 3), (1, 0, 2, 3))
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x44 += einsum(t1.aa, (0, 1), x43, (0, 2, 3, 4), (2, 3, 1, 4))
    x45 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x45 += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3))
    x45 += einsum(x44, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x45, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x46 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x46 += einsum(f.aa.vv, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 0, 4))
    x47 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x47 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x47 += einsum(x46, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    t2new_aaaa += einsum(x47, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3))
    x48 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x48 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x48 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (3, 4, 0, 2))
    x49 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x49 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x49 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x49 += einsum(t1.bb, (0, 1), x4, (2, 1, 3, 4), (2, 0, 4, 3))
    x50 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x50 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x50 += einsum(f.aa.ov, (0, 1), t2.abab, (2, 3, 1, 4), (3, 4, 0, 2))
    x50 += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2))
    x50 += einsum(t1.bb, (0, 1), x48, (1, 2, 3, 4), (0, 2, 4, 3))
    x50 += einsum(t1.bb, (0, 1), x49, (0, 2, 3, 4), (2, 1, 4, 3)) * -1.0
    t2new_abab += einsum(t1.aa, (0, 1), x50, (2, 3, 0, 4), (4, 2, 1, 3)) * -1.0
    x51 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    x51 += einsum(v.aabb.vvov, (0, 1, 2, 3), (2, 3, 0, 1))
    x51 += einsum(t1.bb, (0, 1), v.aabb.vvvv, (2, 3, 4, 1), (0, 4, 2, 3))
    t2new_abab += einsum(t1.aa, (0, 1), x51, (2, 3, 1, 4), (0, 2, 4, 3))
    x52 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x52 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x52 += einsum(t1.aa, (0, 1), v.aabb.vvov, (2, 1, 3, 4), (3, 4, 0, 2))
    x53 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x53 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x53 += einsum(f.bb.ov, (0, 1), t2.abab, (2, 3, 4, 1), (0, 3, 2, 4))
    x53 += einsum(t1.aa, (0, 1), v.aabb.vvoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x53 += einsum(t1.bb, (0, 1), x52, (2, 1, 3, 4), (2, 0, 3, 4))
    t2new_abab += einsum(t1.bb, (0, 1), x53, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    x54 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x54 += einsum(f.bb.ov, (0, 1), t1.bb, (2, 1), (0, 2))
    x55 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x55 += einsum(f.bb.oo, (0, 1), (0, 1))
    x55 += einsum(x54, (0, 1), (0, 1))
    t2new_abab += einsum(x55, (0, 1), t2.abab, (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    x56 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x56 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x57 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x57 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x58 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x58 += einsum(t1.bb, (0, 1), x57, (2, 3, 4, 0), (2, 4, 3, 1))
    x59 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x59 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x59 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x59 += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3))
    x60 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x60 += einsum(t1.bb, (0, 1), x59, (2, 3, 1, 4), (0, 2, 3, 4))
    x61 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x61 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x61 += einsum(x58, (0, 1, 2, 3), (0, 2, 1, 3))
    x61 += einsum(x60, (0, 1, 2, 3), (0, 2, 1, 3))
    x62 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x62 += einsum(t1.bb, (0, 1), x61, (2, 0, 3, 4), (2, 3, 1, 4))
    x63 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x63 += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3))
    x63 += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    t2new_bbbb += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x63, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb += einsum(x63, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb += einsum(x63, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x64 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x64 += einsum(f.bb.vv, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 0, 4))
    x65 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x65 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x65 += einsum(x64, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    t2new_bbbb += einsum(x65, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3))
    x66 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x66 += einsum(x55, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    t2new_bbbb += einsum(x66, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_bbbb += einsum(x66, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x67 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x67 += einsum(t1.bb, (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x68 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x68 += einsum(t1.bb, (0, 1), x67, (2, 3, 1, 4), (0, 2, 3, 4))
    x69 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x69 += einsum(f.bb.ov, (0, 1), t2.bbbb, (2, 3, 4, 1), (0, 2, 3, 4))
    x70 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x70 += einsum(t1.bb, (0, 1), x19, (2, 3, 4, 1), (2, 0, 4, 3))
    x71 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x71 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x71 += einsum(x70, (0, 1, 2, 3), (3, 1, 2, 0))
    x72 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x72 += einsum(t1.bb, (0, 1), x71, (0, 2, 3, 4), (2, 3, 4, 1))
    x73 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x73 += einsum(x69, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x73 += einsum(x72, (0, 1, 2, 3), (1, 0, 2, 3))
    x74 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x74 += einsum(t1.bb, (0, 1), x73, (0, 2, 3, 4), (2, 3, 1, 4))
    x75 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x75 += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3))
    x75 += einsum(x74, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb += einsum(x75, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x75, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0

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
    l1new_aa = np.zeros((nvir[0], nocc[0]), dtype=np.float64)
    l1new_aa += einsum(f.aa.ov, (0, 1), (1, 0))
    l1new_bb = np.zeros((nvir[1], nocc[1]), dtype=np.float64)
    l1new_bb += einsum(f.bb.ov, (0, 1), (1, 0))
    l2new_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=np.float64)
    l2new_aaaa += einsum(l2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 1, 5, 0), (4, 5, 2, 3)) * -2.0
    l2new_abab = np.zeros((nvir[0], nvir[1], nocc[0], nocc[1]), dtype=np.float64)
    l2new_abab += einsum(l1.aa, (0, 1), v.aabb.vvov, (2, 0, 3, 4), (2, 4, 1, 3))
    l2new_abab += einsum(l1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 0), (3, 4, 2, 1))
    l2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (1, 3, 0, 2))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 0, 5, 1), (4, 5, 2, 3))
    l2new_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=np.float64)
    l2new_bbbb += einsum(l2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 1, 5, 0), (4, 5, 2, 3)) * -2.0
    x0 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum(t1.bb, (0, 1), v.aabb.vvvv, (2, 3, 4, 1), (0, 4, 2, 3))
    l1new_aa += einsum(l2.abab, (0, 1, 2, 3), x0, (3, 1, 4, 0), (4, 2))
    x1 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum(t1.aa, (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    l1new_aa += einsum(l2.aaaa, (0, 1, 2, 3), x1, (3, 4, 1, 0), (4, 2)) * -2.0
    x2 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x2 += einsum(t1.bb, (0, 1), l2.abab, (2, 1, 3, 4), (4, 0, 3, 2))
    l1new_aa += einsum(v.aabb.vvoo, (0, 1, 2, 3), x2, (3, 2, 4, 1), (0, 4)) * -1.0
    l2new_abab += einsum(v.aabb.vvov, (0, 1, 2, 3), x2, (4, 2, 5, 1), (0, 3, 5, 4)) * -1.0
    x3 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x3 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    l1new_aa += einsum(v.aaaa.oovv, (0, 1, 2, 3), x3, (4, 1, 0, 3), (2, 4)) * -2.0
    l2new_abab += einsum(v.aabb.ooov, (0, 1, 2, 3), x3, (4, 1, 0, 5), (5, 3, 4, 2)) * -2.0
    x4 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x4 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    l1new_aa += einsum(x3, (0, 1, 2, 3), x4, (1, 2, 4, 3), (4, 0)) * -2.0
    x5 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x5 += einsum(t1.aa, (0, 1), v.aabb.ovov, (2, 1, 3, 4), (3, 4, 0, 2))
    l2new_abab += einsum(x3, (0, 1, 2, 3), x5, (4, 5, 1, 2), (3, 5, 0, 4)) * -2.0
    x6 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x6 += einsum(t1.aa, (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x7 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x7 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x7 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x7 += einsum(x6, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x7 += einsum(x6, (0, 1, 2, 3), (2, 0, 1, 3))
    x8 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x8 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x8 += einsum(x5, (0, 1, 2, 3), (0, 1, 3, 2))
    x9 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x9 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 4, 1), (0, 4, 2, 3))
    x10 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x10 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x10 += einsum(x9, (0, 1, 2, 3), (1, 0, 2, 3))
    x11 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x11 += einsum(t1.bb, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    x12 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x12 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x12 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x13 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x13 += einsum(t1.aa, (0, 1), x12, (0, 2, 3, 1), (2, 3))
    x14 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x14 += einsum(f.aa.ov, (0, 1), (0, 1))
    x14 += einsum(x11, (0, 1), (0, 1))
    x14 += einsum(x13, (0, 1), (0, 1)) * -1.0
    l2new_abab += einsum(l1.bb, (0, 1), x14, (2, 3), (3, 0, 2, 1))
    x15 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x15 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (3, 4, 0, 2))
    x16 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x16 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x16 += einsum(x15, (0, 1, 2, 3), (1, 0, 3, 2))
    x17 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x17 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x18 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x18 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (1, 5, 0, 4))
    x19 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x19 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x19 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    l2new_abab += einsum(x19, (0, 1, 2, 3), x2, (4, 0, 2, 5), (5, 1, 3, 4))
    l2new_abab += einsum(l1.aa, (0, 1), x19, (2, 3, 1, 4), (0, 3, 4, 2)) * -1.0
    x20 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x20 += einsum(t1.bb, (0, 1), x19, (2, 1, 3, 4), (0, 2, 3, 4))
    x21 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x21 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x21 += einsum(x17, (0, 1, 2, 3), (1, 0, 3, 2))
    x21 += einsum(x18, (0, 1, 2, 3), (1, 0, 3, 2))
    x21 += einsum(x20, (0, 1, 2, 3), (1, 0, 3, 2))
    x22 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x22 += einsum(v.aabb.ooov, (0, 1, 2, 3), (2, 3, 0, 1))
    x22 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    x22 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovvv, (4, 2, 5, 3), (1, 5, 0, 4))
    x22 += einsum(t2.abab, (0, 1, 2, 3), x7, (0, 4, 5, 2), (1, 3, 4, 5)) * -1.0
    x22 += einsum(t2.bbbb, (0, 1, 2, 3), x8, (1, 3, 4, 5), (0, 2, 5, 4)) * 2.0
    x22 += einsum(t2.abab, (0, 1, 2, 3), x10, (1, 4, 5, 2), (4, 3, 0, 5)) * -1.0
    x22 += einsum(x14, (0, 1), t2.abab, (2, 3, 1, 4), (3, 4, 2, 0))
    x22 += einsum(t1.bb, (0, 1), x16, (1, 2, 3, 4), (0, 2, 4, 3))
    x22 += einsum(t1.bb, (0, 1), x21, (0, 2, 3, 4), (2, 1, 4, 3)) * -1.0
    l1new_aa += einsum(l2.abab, (0, 1, 2, 3), x22, (3, 1, 2, 4), (0, 4)) * -1.0
    x23 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x23 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x23 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x23 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    x23 += einsum(x6, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x24 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x24 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x24 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x24 += einsum(t1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x25 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x25 += einsum(t1.aa, (0, 1), x6, (2, 3, 4, 1), (0, 2, 3, 4))
    x26 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x26 += einsum(t1.aa, (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x27 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x27 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x27 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 5, 2), (4, 0, 5, 1)) * -1.0
    x27 += einsum(x25, (0, 1, 2, 3), (3, 1, 2, 0))
    x27 += einsum(x26, (0, 1, 2, 3), (2, 1, 3, 0))
    x27 += einsum(x26, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x28 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x28 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x28 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 3), (4, 0, 1, 5)) * -1.0
    x28 += einsum(t2.aaaa, (0, 1, 2, 3), x23, (4, 5, 1, 3), (5, 0, 4, 2)) * -2.0
    x28 += einsum(t2.abab, (0, 1, 2, 3), x19, (1, 3, 4, 5), (5, 0, 4, 2))
    x28 += einsum(x14, (0, 1), t2.aaaa, (2, 3, 4, 1), (0, 2, 3, 4))
    x28 += einsum(t1.aa, (0, 1), x24, (2, 3, 1, 4), (3, 2, 0, 4))
    x28 += einsum(t1.aa, (0, 1), x27, (0, 2, 3, 4), (3, 2, 4, 1)) * -1.0
    l1new_aa += einsum(l2.aaaa, (0, 1, 2, 3), x28, (4, 2, 3, 1), (0, 4)) * 2.0
    x29 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x29 += einsum(l2.abab, (0, 1, 2, 3), (3, 1, 2, 0))
    x29 += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (4, 5, 2, 0)) * 2.0
    x29 += einsum(t1.bb, (0, 1), x2, (0, 2, 3, 4), (2, 1, 3, 4)) * -1.0
    x29 += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 1, 5), (4, 5, 2, 0)) * 2.0
    l1new_aa += einsum(v.aabb.vvov, (0, 1, 2, 3), x29, (2, 3, 4, 1), (0, 4))
    x30 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x30 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 4, 0, 5))
    x31 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x31 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 5, 1), (2, 4, 0, 5))
    x32 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x32 += einsum(l2.aaaa, (0, 1, 2, 3), (2, 3, 0, 1))
    x32 += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x32 += einsum(x31, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    l1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x32, (4, 0, 3, 1), (2, 4)) * 2.0
    x33 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x33 += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    x33 += einsum(x31, (0, 1, 2, 3), (0, 1, 2, 3))
    x33 += einsum(t1.aa, (0, 1), x3, (2, 0, 3, 4), (2, 3, 1, 4)) * -0.5
    l1new_aa += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x33, (4, 0, 3, 2), (1, 4)) * -4.0
    x34 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x34 += einsum(t1.aa, (0, 1), l2.abab, (1, 2, 3, 4), (4, 2, 3, 0))
    l1new_bb += einsum(v.aabb.oovv, (0, 1, 2, 3), x34, (4, 3, 1, 0), (2, 4)) * -1.0
    l2new_abab += einsum(v.aabb.ovvv, (0, 1, 2, 3), x34, (4, 3, 5, 0), (1, 2, 5, 4)) * -1.0
    l2new_abab += einsum(x14, (0, 1), x34, (2, 3, 4, 0), (1, 3, 4, 2)) * -1.0
    x35 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x35 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), (3, 5, 2, 4))
    x36 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x36 += einsum(t1.bb, (0, 1), x34, (2, 1, 3, 4), (2, 0, 3, 4))
    l2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), x36, (4, 2, 5, 0), (1, 3, 5, 4))
    x37 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x37 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    x37 += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3))
    x38 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x38 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    x39 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x39 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    x40 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x40 += einsum(x38, (0, 1), (0, 1))
    x40 += einsum(x39, (0, 1), (0, 1)) * 2.0
    l1new_aa += einsum(f.aa.ov, (0, 1), x40, (2, 0), (1, 2)) * -1.0
    x41 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x41 += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), (3, 4, 1, 2))
    x41 += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3))
    x41 += einsum(t2.bbbb, (0, 1, 2, 3), x34, (1, 3, 4, 5), (0, 2, 4, 5)) * 2.0
    x41 += einsum(t2.abab, (0, 1, 2, 3), x2, (1, 4, 5, 2), (4, 3, 5, 0)) * -1.0
    x41 += einsum(t2.abab, (0, 1, 2, 3), x3, (4, 0, 5, 2), (1, 3, 4, 5)) * -2.0
    x41 += einsum(t1.bb, (0, 1), x37, (0, 2, 3, 4), (2, 1, 3, 4)) * -1.0
    x41 += einsum(t1.bb, (0, 1), x40, (2, 3), (0, 1, 2, 3))
    l1new_aa += einsum(v.aabb.ovov, (0, 1, 2, 3), x41, (2, 3, 4, 0), (1, 4)) * -1.0
    x42 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x42 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), (1, 5, 2, 4))
    x42 += einsum(t1.bb, (0, 1), x34, (0, 2, 3, 4), (1, 2, 3, 4))
    l1new_aa += einsum(v.aabb.ovvv, (0, 1, 2, 3), x42, (3, 2, 4, 0), (1, 4)) * -1.0
    x43 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x43 += einsum(t2.abab, (0, 1, 2, 3), x34, (1, 3, 4, 5), (4, 5, 0, 2))
    x44 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x44 += einsum(t2.aaaa, (0, 1, 2, 3), x3, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x45 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x45 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), (2, 3, 4, 5))
    x46 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x46 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 0), (1, 2, 3, 4))
    x46 += einsum(x43, (0, 1, 2, 3), (0, 2, 1, 3)) * 0.5
    x46 += einsum(x44, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x46 += einsum(t1.aa, (0, 1), x45, (2, 0, 3, 4), (2, 4, 3, 1)) * -1.0
    x46 += einsum(t1.aa, (0, 1), x40, (2, 3), (2, 0, 3, 1)) * 0.5
    l1new_aa += einsum(v.aaaa.ovov, (0, 1, 2, 3), x46, (4, 0, 2, 3), (1, 4)) * 2.0
    x47 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x47 += einsum(t1.aa, (0, 1), x3, (2, 3, 4, 1), (3, 2, 4, 0))
    l2new_aaaa += einsum(v.aaaa.ovov, (0, 1, 2, 3), x47, (4, 5, 0, 2), (3, 1, 5, 4)) * 2.0
    x48 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x48 += einsum(x3, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x48 += einsum(x43, (0, 1, 2, 3), (2, 0, 1, 3)) * 0.5
    x48 += einsum(x44, (0, 1, 2, 3), (2, 0, 1, 3)) * 2.0
    x48 += einsum(t1.aa, (0, 1), x47, (0, 2, 3, 4), (4, 2, 3, 1))
    x48 += einsum(t1.aa, (0, 1), x40, (2, 3), (0, 2, 3, 1)) * 0.5
    l1new_aa += einsum(v.aaaa.ovov, (0, 1, 2, 3), x48, (0, 4, 2, 1), (3, 4)) * -2.0
    x49 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x49 += einsum(l1.aa, (0, 1), t1.aa, (1, 2), (0, 2))
    x50 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x50 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    x51 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x51 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4))
    x52 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x52 += einsum(x49, (0, 1), (0, 1))
    x52 += einsum(x50, (0, 1), (0, 1))
    x52 += einsum(x51, (0, 1), (0, 1)) * 2.0
    l1new_bb += einsum(x52, (0, 1), v.aabb.vvov, (1, 0, 2, 3), (3, 2))
    x53 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x53 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x53 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    l1new_aa += einsum(x52, (0, 1), x53, (2, 3, 1, 0), (3, 2)) * -1.0
    x54 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x54 += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), (2, 3))
    x55 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x55 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 0), (2, 3))
    x56 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x56 += einsum(t2.abab, (0, 1, 2, 3), x34, (1, 3, 0, 4), (4, 2))
    x57 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x57 += einsum(t2.aaaa, (0, 1, 2, 3), x3, (0, 1, 4, 3), (4, 2)) * -1.0
    x58 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x58 += einsum(l1.aa, (0, 1), t1.aa, (2, 0), (1, 2))
    l1new_aa += einsum(x14, (0, 1), x58, (2, 0), (1, 2)) * -1.0
    l2new_abab += einsum(x58, (0, 1), v.aabb.ovov, (1, 2, 3, 4), (2, 4, 0, 3)) * -1.0
    x59 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x59 += einsum(x58, (0, 1), (0, 1)) * 0.5
    x59 += einsum(x38, (0, 1), (0, 1)) * 0.5
    x59 += einsum(x39, (0, 1), (0, 1))
    x60 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x60 += einsum(t1.aa, (0, 1), x59, (0, 2), (2, 1)) * 2.0
    x61 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x61 += einsum(t1.aa, (0, 1), (0, 1)) * -1.0
    x61 += einsum(x54, (0, 1), (0, 1)) * -1.0
    x61 += einsum(x55, (0, 1), (0, 1)) * -2.0
    x61 += einsum(x56, (0, 1), (0, 1))
    x61 += einsum(x57, (0, 1), (0, 1)) * 2.0
    x61 += einsum(x60, (0, 1), (0, 1))
    l1new_aa += einsum(x61, (0, 1), x12, (0, 2, 3, 1), (3, 2))
    x62 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x62 += einsum(l1.bb, (0, 1), t1.bb, (1, 2), (0, 2))
    x63 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x63 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4))
    x64 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x64 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    x65 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x65 += einsum(x62, (0, 1), (0, 1))
    x65 += einsum(x63, (0, 1), (0, 1)) * 2.0
    x65 += einsum(x64, (0, 1), (0, 1))
    l1new_aa += einsum(x65, (0, 1), v.aabb.ovvv, (2, 3, 1, 0), (3, 2))
    x66 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x66 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 0), (2, 3))
    x67 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x67 += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), (2, 3))
    x68 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x68 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    l1new_bb += einsum(v.bbbb.oovv, (0, 1, 2, 3), x68, (4, 1, 0, 3), (2, 4)) * -2.0
    l2new_abab += einsum(x68, (0, 1, 2, 3), x9, (1, 2, 4, 5), (5, 3, 4, 0)) * -2.0
    l2new_abab += einsum(v.aabb.ovoo, (0, 1, 2, 3), x68, (4, 3, 2, 5), (1, 5, 0, 4)) * -2.0
    x69 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x69 += einsum(t2.bbbb, (0, 1, 2, 3), x68, (0, 1, 4, 3), (4, 2)) * -1.0
    x70 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x70 += einsum(t2.abab, (0, 1, 2, 3), x2, (1, 4, 0, 2), (4, 3))
    x71 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x71 += einsum(l1.bb, (0, 1), t1.bb, (2, 0), (1, 2))
    l2new_abab += einsum(x71, (0, 1), v.aabb.ovov, (2, 3, 1, 4), (3, 4, 2, 0)) * -1.0
    x72 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x72 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    x73 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x73 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    x74 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x74 += einsum(x71, (0, 1), (0, 1))
    x74 += einsum(x72, (0, 1), (0, 1)) * 2.0
    x74 += einsum(x73, (0, 1), (0, 1))
    l1new_aa += einsum(x74, (0, 1), v.aabb.ovoo, (2, 3, 1, 0), (3, 2)) * -1.0
    x75 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x75 += einsum(t1.bb, (0, 1), x74, (0, 2), (2, 1))
    x76 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x76 += einsum(l1.bb, (0, 1), (1, 0)) * -1.0
    x76 += einsum(t1.bb, (0, 1), (0, 1)) * -1.0
    x76 += einsum(x66, (0, 1), (0, 1)) * -2.0
    x76 += einsum(x67, (0, 1), (0, 1)) * -1.0
    x76 += einsum(x69, (0, 1), (0, 1)) * 2.0
    x76 += einsum(x70, (0, 1), (0, 1))
    x76 += einsum(x75, (0, 1), (0, 1))
    l1new_aa += einsum(x76, (0, 1), v.aabb.ovov, (2, 3, 0, 1), (3, 2)) * -1.0
    x77 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x77 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    x77 += einsum(x36, (0, 1, 2, 3), (1, 0, 2, 3))
    l1new_aa += einsum(v.aabb.ovoo, (0, 1, 2, 3), x77, (3, 2, 4, 0), (1, 4))
    x78 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x78 += einsum(x45, (0, 1, 2, 3), (1, 0, 3, 2))
    x78 += einsum(x47, (0, 1, 2, 3), (2, 1, 0, 3))
    l1new_aa += einsum(v.aaaa.ooov, (0, 1, 2, 3), x78, (1, 4, 0, 2), (3, 4)) * 2.0
    x79 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x79 += einsum(x58, (0, 1), (0, 1))
    x79 += einsum(x38, (0, 1), (0, 1))
    x79 += einsum(x39, (0, 1), (0, 1)) * 2.0
    l1new_bb += einsum(x79, (0, 1), v.aabb.ooov, (1, 0, 2, 3), (3, 2)) * -1.0
    x80 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x80 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x80 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    l1new_aa += einsum(x79, (0, 1), x80, (1, 0, 2, 3), (3, 2)) * -1.0
    x81 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x81 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x81 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l1new_aa += einsum(l1.aa, (0, 1), x81, (1, 2, 0, 3), (3, 2)) * -1.0
    x82 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x82 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 0, 1), (2, 3))
    x83 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x83 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x83 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x84 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x84 += einsum(t1.aa, (0, 1), x83, (0, 2, 1, 3), (2, 3))
    x85 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x85 += einsum(f.aa.vv, (0, 1), (0, 1))
    x85 += einsum(x82, (0, 1), (1, 0))
    x85 += einsum(x84, (0, 1), (1, 0)) * -1.0
    l1new_aa += einsum(l1.aa, (0, 1), x85, (0, 2), (2, 1))
    l2new_abab += einsum(x85, (0, 1), l2.abab, (0, 2, 3, 4), (1, 2, 3, 4))
    x86 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x86 += einsum(t1.bb, (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3))
    x87 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x87 += einsum(t1.aa, (0, 1), x80, (0, 2, 3, 1), (2, 3))
    x88 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x88 += einsum(t1.aa, (0, 1), x14, (2, 1), (0, 2))
    x89 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x89 += einsum(f.aa.oo, (0, 1), (0, 1))
    x89 += einsum(x86, (0, 1), (1, 0))
    x89 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x89 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -2.0
    x89 += einsum(x87, (0, 1), (0, 1)) * -1.0
    x89 += einsum(x88, (0, 1), (0, 1))
    l1new_aa += einsum(l1.aa, (0, 1), x89, (1, 2), (0, 2)) * -1.0
    x90 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x90 += einsum(t1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    l1new_bb += einsum(x68, (0, 1, 2, 3), x90, (1, 2, 4, 3), (4, 0)) * 2.0
    x91 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x91 += einsum(t1.aa, (0, 1), v.aabb.vvvv, (2, 1, 3, 4), (3, 4, 0, 2))
    l1new_bb += einsum(l2.abab, (0, 1, 2, 3), x91, (4, 1, 2, 0), (4, 3))
    x92 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x92 += einsum(t1.bb, (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    l1new_bb += einsum(l2.bbbb, (0, 1, 2, 3), x92, (3, 4, 0, 1), (4, 2)) * 2.0
    x93 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x93 += einsum(t1.bb, (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x94 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x94 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x94 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x94 += einsum(x93, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x94 += einsum(x93, (0, 1, 2, 3), (2, 0, 1, 3))
    x95 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x95 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    x96 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x96 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x96 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x97 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x97 += einsum(t1.bb, (0, 1), x96, (0, 2, 3, 1), (2, 3))
    x98 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x98 += einsum(f.bb.ov, (0, 1), (0, 1))
    x98 += einsum(x95, (0, 1), (0, 1))
    x98 += einsum(x97, (0, 1), (0, 1)) * -1.0
    l1new_bb += einsum(x71, (0, 1), x98, (1, 2), (2, 0)) * -1.0
    l2new_abab += einsum(x98, (0, 1), x2, (2, 0, 3, 4), (4, 1, 3, 2)) * -1.0
    l2new_abab += einsum(l1.aa, (0, 1), x98, (2, 3), (0, 3, 1, 2))
    x99 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x99 += einsum(t1.aa, (0, 1), v.aabb.vvov, (2, 1, 3, 4), (3, 4, 0, 2))
    x100 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x100 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x100 += einsum(x99, (0, 1, 2, 3), (0, 1, 2, 3))
    l2new_abab += einsum(l2.aaaa, (0, 1, 2, 3), x100, (4, 5, 3, 1), (0, 5, 2, 4)) * 2.0
    x101 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x101 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x101 += einsum(t1.aa, (0, 1), v.aabb.vvoo, (2, 1, 3, 4), (3, 4, 0, 2))
    x101 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvov, (4, 2, 5, 3), (1, 5, 0, 4))
    x101 += einsum(t2.abab, (0, 1, 2, 3), x94, (1, 4, 5, 3), (4, 5, 0, 2)) * -1.0
    x101 += einsum(t2.aaaa, (0, 1, 2, 3), x10, (4, 5, 1, 3), (5, 4, 0, 2)) * 2.0
    x101 += einsum(t2.abab, (0, 1, 2, 3), x8, (4, 3, 0, 5), (1, 4, 5, 2)) * -1.0
    x101 += einsum(x98, (0, 1), t2.abab, (2, 3, 4, 1), (3, 0, 2, 4))
    x101 += einsum(t1.bb, (0, 1), x100, (2, 1, 3, 4), (0, 2, 3, 4))
    x101 += einsum(t1.aa, (0, 1), x21, (2, 3, 0, 4), (3, 2, 4, 1)) * -1.0
    l1new_bb += einsum(l2.abab, (0, 1, 2, 3), x101, (3, 4, 2, 0), (1, 4)) * -1.0
    x102 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x102 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x102 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x102 += einsum(x93, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x102 += einsum(x93, (0, 1, 2, 3), (0, 2, 1, 3))
    x103 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x103 += einsum(v.aabb.ovoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x103 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    l2new_abab += einsum(x103, (0, 1, 2, 3), x34, (0, 4, 5, 2), (3, 4, 5, 1))
    l2new_abab += einsum(l1.bb, (0, 1), x103, (1, 2, 3, 4), (4, 0, 3, 2)) * -1.0
    x104 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x104 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x104 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x104 += einsum(x90, (0, 1, 2, 3), (0, 1, 2, 3))
    x105 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x105 += einsum(t1.bb, (0, 1), x93, (2, 3, 4, 1), (0, 2, 3, 4))
    x106 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x106 += einsum(t1.bb, (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x107 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x107 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x107 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 5, 3), (4, 0, 5, 1))
    x107 += einsum(x105, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x107 += einsum(x106, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    x107 += einsum(x106, (0, 1, 2, 3), (3, 0, 2, 1))
    x108 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x108 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x108 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovvv, (4, 3, 5, 2), (4, 0, 1, 5))
    x108 += einsum(t2.bbbb, (0, 1, 2, 3), x102, (4, 1, 5, 3), (5, 0, 4, 2)) * -2.0
    x108 += einsum(t2.abab, (0, 1, 2, 3), x103, (4, 5, 0, 2), (5, 1, 4, 3))
    x108 += einsum(x98, (0, 1), t2.bbbb, (2, 3, 4, 1), (0, 2, 3, 4))
    x108 += einsum(t1.bb, (0, 1), x104, (2, 3, 1, 4), (3, 2, 0, 4))
    x108 += einsum(t1.bb, (0, 1), x107, (0, 2, 3, 4), (3, 4, 2, 1))
    l1new_bb += einsum(l2.bbbb, (0, 1, 2, 3), x108, (4, 2, 3, 1), (0, 4)) * 2.0
    x109 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x109 += einsum(l2.abab, (0, 1, 2, 3), (3, 1, 2, 0))
    x109 += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 0, 4, 5)) * 2.0
    x109 += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 2, 5, 0), (3, 1, 4, 5)) * 2.0
    x109 += einsum(t1.aa, (0, 1), x34, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    l1new_bb += einsum(v.aabb.ovvv, (0, 1, 2, 3), x109, (4, 3, 0, 1), (2, 4))
    x110 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x110 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x110 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 1, 3))
    l1new_bb += einsum(x65, (0, 1), x110, (2, 3, 1, 0), (3, 2)) * -1.0
    x111 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x111 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (2, 4, 0, 5))
    x111 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), (3, 4, 1, 5)) * 0.25
    l1new_bb += einsum(x110, (0, 1, 2, 3), x111, (4, 0, 3, 2), (1, 4)) * 4.0
    x112 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x112 += einsum(x72, (0, 1), (0, 1))
    x112 += einsum(x73, (0, 1), (0, 1)) * 0.5
    x113 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x113 += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), (1, 3, 2, 4))
    x113 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3))
    x113 += einsum(t2.abab, (0, 1, 2, 3), x68, (4, 1, 5, 3), (4, 5, 0, 2)) * -2.0
    x113 += einsum(t2.aaaa, (0, 1, 2, 3), x2, (4, 5, 1, 3), (4, 5, 0, 2)) * 2.0
    x113 += einsum(t2.abab, (0, 1, 2, 3), x34, (4, 3, 0, 5), (4, 1, 5, 2)) * -1.0
    x113 += einsum(t1.aa, (0, 1), x37, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    x113 += einsum(t1.aa, (0, 1), x112, (2, 3), (2, 3, 0, 1)) * 2.0
    l1new_bb += einsum(v.aabb.ovov, (0, 1, 2, 3), x113, (4, 2, 0, 1), (3, 4)) * -1.0
    x114 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x114 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), (3, 4, 0, 5))
    x114 += einsum(t1.aa, (0, 1), x2, (2, 3, 0, 4), (2, 3, 1, 4))
    l1new_bb += einsum(v.aabb.vvov, (0, 1, 2, 3), x114, (4, 2, 1, 0), (3, 4)) * -1.0
    x115 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x115 += einsum(l2.bbbb, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x115 += einsum(t1.bb, (0, 1), x68, (2, 0, 3, 4), (2, 3, 4, 1))
    l1new_bb += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x115, (4, 0, 3, 1), (2, 4)) * -2.0
    x116 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x116 += einsum(t2.bbbb, (0, 1, 2, 3), x68, (4, 1, 5, 3), (4, 0, 5, 2)) * -2.0
    x116 += einsum(t2.abab, (0, 1, 2, 3), x2, (4, 5, 0, 2), (4, 1, 5, 3)) * 0.5
    x116 += einsum(t1.bb, (0, 1), x112, (2, 3), (2, 0, 3, 1))
    l1new_bb += einsum(x116, (0, 1, 2, 3), x96, (1, 2, 4, 3), (4, 0)) * 2.0
    x117 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x117 += einsum(t1.bb, (0, 1), (0, 1)) * -1.0
    x117 += einsum(x66, (0, 1), (0, 1)) * -2.0
    x117 += einsum(x67, (0, 1), (0, 1)) * -1.0
    x117 += einsum(x69, (0, 1), (0, 1)) * 2.0
    x117 += einsum(x70, (0, 1), (0, 1))
    x117 += einsum(x75, (0, 1), (0, 1))
    l1new_bb += einsum(x117, (0, 1), x96, (0, 2, 3, 1), (3, 2))
    x118 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x118 += einsum(t1.bb, (0, 1), x68, (2, 3, 4, 1), (3, 2, 4, 0))
    l2new_bbbb += einsum(v.bbbb.ovov, (0, 1, 2, 3), x118, (4, 5, 0, 2), (3, 1, 5, 4)) * 2.0
    x119 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x119 += einsum(x68, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x119 += einsum(t1.bb, (0, 1), x118, (0, 2, 3, 4), (4, 2, 3, 1))
    l1new_bb += einsum(v.bbbb.ovov, (0, 1, 2, 3), x119, (0, 4, 2, 1), (3, 4)) * -2.0
    x120 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x120 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), (2, 3, 4, 5))
    x121 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x121 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 0), (1, 2, 3, 4))
    x121 += einsum(t1.bb, (0, 1), x120, (2, 0, 3, 4), (2, 4, 3, 1)) * -1.0
    l1new_bb += einsum(v.bbbb.ovov, (0, 1, 2, 3), x121, (4, 0, 2, 3), (1, 4)) * 2.0
    x122 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x122 += einsum(l1.aa, (0, 1), (1, 0)) * -1.0
    x122 += einsum(t1.aa, (0, 1), (0, 1)) * -1.0
    x122 += einsum(x54, (0, 1), (0, 1)) * -1.0
    x122 += einsum(x55, (0, 1), (0, 1)) * -2.0
    x122 += einsum(x56, (0, 1), (0, 1))
    x122 += einsum(x57, (0, 1), (0, 1)) * 2.0
    x122 += einsum(x60, (0, 1), (0, 1))
    l1new_bb += einsum(x122, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (3, 2)) * -1.0
    x123 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x123 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3))
    x123 += einsum(x36, (0, 1, 2, 3), (0, 1, 3, 2))
    l1new_bb += einsum(v.aabb.ooov, (0, 1, 2, 3), x123, (4, 2, 0, 1), (3, 4))
    x124 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x124 += einsum(x120, (0, 1, 2, 3), (1, 0, 3, 2))
    x124 += einsum(x118, (0, 1, 2, 3), (2, 1, 0, 3))
    l1new_bb += einsum(v.bbbb.ooov, (0, 1, 2, 3), x124, (0, 4, 1, 2), (3, 4)) * 2.0
    x125 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x125 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x125 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    l1new_bb += einsum(x74, (0, 1), x125, (1, 0, 2, 3), (3, 2)) * -1.0
    x126 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x126 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x126 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l1new_bb += einsum(l1.bb, (0, 1), x126, (1, 2, 0, 3), (3, 2)) * -1.0
    x127 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x127 += einsum(t1.aa, (0, 1), v.aabb.ovvv, (0, 1, 2, 3), (2, 3))
    x128 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x128 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x128 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x129 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x129 += einsum(t1.bb, (0, 1), x128, (0, 2, 1, 3), (2, 3))
    x130 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x130 += einsum(f.bb.vv, (0, 1), (0, 1))
    x130 += einsum(x127, (0, 1), (1, 0))
    x130 += einsum(x129, (0, 1), (1, 0)) * -1.0
    l1new_bb += einsum(l1.bb, (0, 1), x130, (0, 2), (2, 1))
    l2new_abab += einsum(x130, (0, 1), l2.abab, (2, 0, 3, 4), (2, 1, 3, 4))
    x131 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x131 += einsum(t1.aa, (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3))
    x132 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x132 += einsum(t1.bb, (0, 1), x125, (0, 2, 3, 1), (2, 3))
    x133 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x133 += einsum(t1.bb, (0, 1), x98, (2, 1), (0, 2))
    x134 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x134 += einsum(f.bb.oo, (0, 1), (0, 1))
    x134 += einsum(x131, (0, 1), (1, 0))
    x134 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 1, 3), (0, 4)) * 2.0
    x134 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x134 += einsum(x132, (0, 1), (0, 1)) * -1.0
    x134 += einsum(x133, (0, 1), (0, 1))
    l1new_bb += einsum(l1.bb, (0, 1), x134, (1, 2), (0, 2)) * -1.0
    x135 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x135 += einsum(x72, (0, 1), (0, 1)) * 2.0
    x135 += einsum(x73, (0, 1), (0, 1))
    l1new_bb += einsum(f.bb.ov, (0, 1), x135, (2, 0), (1, 2)) * -1.0
    x136 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x136 += einsum(l1.aa, (0, 1), v.aaaa.ovvv, (2, 3, 4, 0), (1, 2, 3, 4))
    x137 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x137 += einsum(x58, (0, 1), v.aaaa.ovov, (2, 3, 1, 4), (0, 2, 3, 4))
    x138 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x138 += einsum(v.aabb.ovoo, (0, 1, 2, 3), x2, (2, 3, 4, 5), (4, 0, 5, 1))
    x139 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x139 += einsum(x2, (0, 1, 2, 3), x9, (0, 1, 4, 5), (2, 4, 3, 5))
    x140 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x140 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 1, 2, 3))
    x140 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x141 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x141 += einsum(t1.aa, (0, 1), x140, (2, 1, 3, 4), (2, 0, 3, 4))
    x142 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x142 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x142 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x142 += einsum(x141, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x143 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x143 += einsum(l2.aaaa, (0, 1, 2, 3), x142, (3, 4, 5, 1), (4, 2, 5, 0)) * 2.0
    x144 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x144 += einsum(t1.bb, (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (0, 4, 2, 3))
    x145 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x145 += einsum(v.aabb.ovov, (0, 1, 2, 3), (2, 3, 0, 1))
    x145 += einsum(x144, (0, 1, 2, 3), (0, 1, 2, 3))
    l2new_abab += einsum(l2.bbbb, (0, 1, 2, 3), x145, (3, 1, 4, 5), (5, 0, 4, 2)) * 2.0
    x146 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x146 += einsum(l2.abab, (0, 1, 2, 3), x145, (3, 1, 4, 5), (4, 2, 5, 0))
    x147 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x147 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    x147 += einsum(x6, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l2new_abab += einsum(x147, (0, 1, 2, 3), x34, (4, 5, 0, 2), (3, 5, 1, 4)) * -1.0
    x148 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x148 += einsum(x147, (0, 1, 2, 3), x3, (0, 4, 2, 5), (1, 4, 3, 5)) * 2.0
    x149 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x149 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x149 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l2new_abab += einsum(x149, (0, 1, 2, 3), x34, (4, 5, 0, 1), (3, 5, 2, 4)) * -1.0
    x150 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x150 += einsum(x149, (0, 1, 2, 3), x3, (0, 4, 1, 5), (2, 4, 3, 5)) * 2.0
    x151 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x151 += einsum(v.aaaa.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x151 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    x152 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x152 += einsum(l1.aa, (0, 1), x151, (1, 2, 3, 4), (2, 3, 4, 0))
    x153 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x153 += einsum(x11, (0, 1), (0, 1))
    x153 += einsum(x13, (0, 1), (0, 1)) * -1.0
    x154 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x154 += einsum(f.aa.ov, (0, 1), l1.aa, (2, 3), (0, 3, 1, 2))
    x154 += einsum(x136, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x154 += einsum(x137, (0, 1, 2, 3), (0, 1, 2, 3))
    x154 += einsum(x138, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x154 += einsum(x139, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x154 += einsum(x143, (0, 1, 2, 3), (1, 0, 3, 2))
    x154 += einsum(x146, (0, 1, 2, 3), (1, 0, 3, 2))
    x154 += einsum(x148, (0, 1, 2, 3), (1, 0, 3, 2))
    x154 += einsum(x150, (0, 1, 2, 3), (1, 0, 3, 2))
    x154 += einsum(x152, (0, 1, 2, 3), (0, 1, 3, 2))
    x154 += einsum(l1.aa, (0, 1), x153, (2, 3), (1, 2, 0, 3))
    l2new_aaaa += einsum(x154, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_aaaa += einsum(x154, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_aaaa += einsum(x154, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new_aaaa += einsum(x154, (0, 1, 2, 3), (3, 2, 1, 0))
    x155 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x155 += einsum(f.aa.vv, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    x156 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x156 += einsum(f.aa.ov, (0, 1), x3, (2, 3, 0, 4), (2, 3, 1, 4))
    x157 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x157 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), x3, (4, 5, 0, 3), (5, 4, 1, 2))
    x158 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x158 += einsum(x82, (0, 1), (1, 0))
    x158 += einsum(x84, (0, 1), (0, 1)) * -1.0
    x159 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x159 += einsum(x158, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2)) * -2.0
    x160 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x160 += einsum(x153, (0, 1), x3, (2, 3, 0, 4), (2, 3, 1, 4)) * 2.0
    x161 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x161 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x161 += einsum(x155, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    x161 += einsum(x156, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    x161 += einsum(x157, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x161 += einsum(x159, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x161 += einsum(x160, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    l2new_aaaa += einsum(x161, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_aaaa += einsum(x161, (0, 1, 2, 3), (2, 3, 0, 1))
    x162 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x162 += einsum(f.aa.ov, (0, 1), t1.aa, (2, 1), (0, 2))
    x163 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x163 += einsum(x162, (0, 1), l2.aaaa, (2, 3, 4, 1), (0, 4, 2, 3))
    x164 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x164 += einsum(l2.aaaa, (0, 1, 2, 3), x26, (3, 4, 2, 5), (4, 5, 0, 1)) * -1.0
    x165 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x165 += einsum(t1.aa, (0, 1), x153, (2, 1), (2, 0))
    x166 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x166 += einsum(x86, (0, 1), (1, 0))
    x166 += einsum(x87, (0, 1), (0, 1)) * -1.0
    x166 += einsum(x165, (0, 1), (1, 0))
    x167 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x167 += einsum(x166, (0, 1), l2.aaaa, (2, 3, 4, 0), (1, 4, 2, 3)) * -2.0
    x168 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x168 += einsum(x163, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x168 += einsum(x164, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    x168 += einsum(x167, (0, 1, 2, 3), (1, 0, 3, 2))
    l2new_aaaa += einsum(x168, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new_aaaa += einsum(x168, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x169 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x169 += einsum(f.aa.oo, (0, 1), l2.aaaa, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_aaaa += einsum(x169, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    l2new_aaaa += einsum(x169, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x170 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x170 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x170 += einsum(x25, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    l2new_aaaa += einsum(l2.aaaa, (0, 1, 2, 3), x170, (3, 4, 2, 5), (0, 1, 4, 5)) * -2.0
    x171 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x171 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x171 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x171 += einsum(t1.aa, (0, 1), x53, (2, 1, 3, 4), (0, 2, 4, 3)) * -1.0
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x171, (2, 4, 0, 5), (5, 1, 4, 3)) * -1.0
    x172 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x172 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x172 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x172 += einsum(t1.bb, (0, 1), x128, (2, 3, 1, 4), (0, 2, 4, 3)) * -1.0
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x172, (3, 4, 1, 5), (0, 5, 2, 4)) * -1.0
    x173 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x173 += einsum(v.aabb.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x173 += einsum(x15, (0, 1, 2, 3), (1, 0, 2, 3))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x173, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    x174 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x174 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x174 += einsum(t1.bb, (0, 1), v.aabb.vvov, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x174, (3, 4, 0, 5), (5, 1, 2, 4)) * -1.0
    x175 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x175 += einsum(v.aabb.oooo, (0, 1, 2, 3), (2, 3, 0, 1))
    x175 += einsum(x17, (0, 1, 2, 3), (1, 0, 2, 3))
    x175 += einsum(t1.bb, (0, 1), x8, (2, 1, 3, 4), (0, 2, 4, 3))
    l2new_abab += einsum(l2.abab, (0, 1, 2, 3), x175, (3, 4, 2, 5), (0, 1, 5, 4))
    x176 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x176 += einsum(x93, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x176 += einsum(x93, (0, 1, 2, 3), (0, 2, 1, 3))
    l2new_abab += einsum(x176, (0, 1, 2, 3), x2, (0, 1, 4, 5), (5, 3, 4, 2)) * -1.0
    x177 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x177 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3))
    x177 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    l2new_abab += einsum(x177, (0, 1, 2, 3), x2, (0, 1, 4, 5), (5, 3, 4, 2)) * -1.0
    x178 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x178 += einsum(f.aa.oo, (0, 1), (0, 1))
    x178 += einsum(x86, (0, 1), (1, 0))
    x178 += einsum(x87, (0, 1), (0, 1)) * -1.0
    x178 += einsum(x88, (0, 1), (0, 1))
    l2new_abab += einsum(x178, (0, 1), l2.abab, (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    x179 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x179 += einsum(f.bb.oo, (0, 1), (0, 1))
    x179 += einsum(x131, (0, 1), (1, 0))
    x179 += einsum(x132, (0, 1), (0, 1)) * -1.0
    x179 += einsum(x133, (0, 1), (0, 1))
    l2new_abab += einsum(x179, (0, 1), l2.abab, (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    x180 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x180 += einsum(l1.bb, (0, 1), v.bbbb.ovvv, (2, 3, 4, 0), (1, 2, 3, 4))
    x181 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x181 += einsum(x71, (0, 1), v.bbbb.ovov, (2, 3, 1, 4), (0, 2, 3, 4))
    x182 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x182 += einsum(v.aabb.ooov, (0, 1, 2, 3), x34, (4, 5, 0, 1), (4, 2, 5, 3))
    x183 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x183 += einsum(x34, (0, 1, 2, 3), x5, (4, 5, 2, 3), (0, 4, 1, 5))
    x184 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x184 += einsum(t1.bb, (0, 1), x110, (2, 3, 1, 4), (0, 2, 3, 4))
    x185 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x185 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x185 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x185 += einsum(x184, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x186 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x186 += einsum(l2.bbbb, (0, 1, 2, 3), x185, (3, 4, 5, 1), (4, 2, 5, 0)) * 2.0
    x187 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x187 += einsum(l2.abab, (0, 1, 2, 3), x100, (4, 5, 2, 0), (3, 4, 1, 5))
    x188 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x188 += einsum(x176, (0, 1, 2, 3), x68, (0, 4, 1, 5), (2, 4, 3, 5)) * 2.0
    x189 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x189 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x189 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 2, 1, 3))
    x190 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x190 += einsum(x189, (0, 1, 2, 3), x68, (0, 4, 2, 5), (1, 4, 3, 5)) * 2.0
    x191 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x191 += einsum(v.bbbb.ooov, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x191 += einsum(x93, (0, 1, 2, 3), (0, 1, 2, 3))
    x192 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x192 += einsum(l1.bb, (0, 1), x191, (1, 2, 3, 4), (2, 3, 4, 0))
    x193 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x193 += einsum(x95, (0, 1), (0, 1))
    x193 += einsum(x97, (0, 1), (0, 1)) * -1.0
    x194 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x194 += einsum(f.bb.ov, (0, 1), l1.bb, (2, 3), (0, 3, 1, 2))
    x194 += einsum(x180, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x194 += einsum(x181, (0, 1, 2, 3), (0, 1, 2, 3))
    x194 += einsum(x182, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x194 += einsum(x183, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x194 += einsum(x186, (0, 1, 2, 3), (1, 0, 3, 2))
    x194 += einsum(x187, (0, 1, 2, 3), (0, 1, 2, 3))
    x194 += einsum(x188, (0, 1, 2, 3), (1, 0, 3, 2))
    x194 += einsum(x190, (0, 1, 2, 3), (1, 0, 3, 2))
    x194 += einsum(x192, (0, 1, 2, 3), (0, 1, 3, 2))
    x194 += einsum(l1.bb, (0, 1), x193, (2, 3), (1, 2, 0, 3))
    l2new_bbbb += einsum(x194, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new_bbbb += einsum(x194, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_bbbb += einsum(x194, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new_bbbb += einsum(x194, (0, 1, 2, 3), (3, 2, 1, 0))
    x195 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x195 += einsum(f.bb.vv, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    x196 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x196 += einsum(f.bb.ov, (0, 1), x68, (2, 3, 0, 4), (2, 3, 1, 4))
    x197 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x197 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), x68, (4, 5, 0, 3), (5, 4, 1, 2))
    x198 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x198 += einsum(x127, (0, 1), (1, 0))
    x198 += einsum(x129, (0, 1), (0, 1)) * -1.0
    x199 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x199 += einsum(x198, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2)) * -2.0
    x200 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x200 += einsum(x193, (0, 1), x68, (2, 3, 0, 4), (2, 3, 1, 4)) * 2.0
    x201 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x201 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x201 += einsum(x195, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    x201 += einsum(x196, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    x201 += einsum(x197, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x201 += einsum(x199, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x201 += einsum(x200, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    l2new_bbbb += einsum(x201, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new_bbbb += einsum(x201, (0, 1, 2, 3), (2, 3, 0, 1))
    x202 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x202 += einsum(f.bb.ov, (0, 1), t1.bb, (2, 1), (0, 2))
    x203 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x203 += einsum(x202, (0, 1), l2.bbbb, (2, 3, 4, 1), (0, 4, 2, 3))
    x204 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x204 += einsum(l2.bbbb, (0, 1, 2, 3), x106, (3, 2, 4, 5), (4, 5, 0, 1)) * -1.0
    x205 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x205 += einsum(t1.bb, (0, 1), x193, (2, 1), (2, 0))
    x206 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x206 += einsum(x131, (0, 1), (1, 0))
    x206 += einsum(x132, (0, 1), (0, 1)) * -1.0
    x206 += einsum(x205, (0, 1), (1, 0))
    x207 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x207 += einsum(x206, (0, 1), l2.bbbb, (2, 3, 4, 0), (1, 4, 2, 3)) * -2.0
    x208 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x208 += einsum(x203, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x208 += einsum(x204, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    x208 += einsum(x207, (0, 1, 2, 3), (1, 0, 3, 2))
    l2new_bbbb += einsum(x208, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new_bbbb += einsum(x208, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    x209 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x209 += einsum(f.bb.oo, (0, 1), l2.bbbb, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new_bbbb += einsum(x209, (0, 1, 2, 3), (3, 2, 0, 1)) * -2.0
    l2new_bbbb += einsum(x209, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x210 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x210 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x210 += einsum(x105, (0, 1, 2, 3), (0, 2, 3, 1))
    l2new_bbbb += einsum(l2.bbbb, (0, 1, 2, 3), x210, (3, 4, 5, 2), (0, 1, 5, 4)) * 2.0

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
    rdm1_f_aa_oo = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    rdm1_f_aa_oo += einsum(delta.aa.oo, (0, 1), (0, 1))
    rdm1_f_bb_oo = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    rdm1_f_bb_oo += einsum(delta.bb.oo, (0, 1), (0, 1))
    rdm1_f_aa_ov = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    rdm1_f_aa_ov += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 0), (2, 3)) * 2.0
    rdm1_f_aa_ov += einsum(t1.aa, (0, 1), (0, 1))
    rdm1_f_aa_ov += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), (2, 3))
    rdm1_f_bb_ov = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    rdm1_f_bb_ov += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 0), (2, 3)) * 2.0
    rdm1_f_bb_ov += einsum(t1.bb, (0, 1), (0, 1))
    rdm1_f_bb_ov += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), (2, 3))
    rdm1_f_aa_vo = np.zeros((nvir[0], nocc[0]), dtype=np.float64)
    rdm1_f_aa_vo += einsum(l1.aa, (0, 1), (0, 1))
    rdm1_f_bb_vo = np.zeros((nvir[1], nocc[1]), dtype=np.float64)
    rdm1_f_bb_vo += einsum(l1.bb, (0, 1), (0, 1))
    rdm1_f_aa_vv = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    rdm1_f_aa_vv += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4)) * 2.0
    rdm1_f_aa_vv += einsum(l1.aa, (0, 1), t1.aa, (1, 2), (0, 2))
    rdm1_f_aa_vv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    rdm1_f_bb_vv = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    rdm1_f_bb_vv += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4)) * 2.0
    rdm1_f_bb_vv += einsum(l1.bb, (0, 1), t1.bb, (1, 2), (0, 2))
    rdm1_f_bb_vv += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    x0 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x0 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    rdm1_f_aa_oo += einsum(x0, (0, 1), (1, 0)) * -2.0
    x1 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x1 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    rdm1_f_aa_oo += einsum(x1, (0, 1), (1, 0)) * -1.0
    x2 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x2 += einsum(l1.aa, (0, 1), t1.aa, (2, 0), (1, 2))
    rdm1_f_aa_oo += einsum(x2, (0, 1), (1, 0)) * -1.0
    x3 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x3 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    rdm1_f_bb_oo += einsum(x3, (0, 1), (1, 0)) * -1.0
    x4 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x4 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    rdm1_f_bb_oo += einsum(x4, (0, 1), (1, 0)) * -2.0
    x5 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x5 += einsum(l1.bb, (0, 1), t1.bb, (2, 0), (1, 2))
    rdm1_f_bb_oo += einsum(x5, (0, 1), (1, 0)) * -1.0
    x6 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x6 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm1_f_aa_ov += einsum(t2.aaaa, (0, 1, 2, 3), x6, (0, 1, 4, 3), (4, 2)) * 2.0
    x7 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x7 += einsum(t1.aa, (0, 1), l2.abab, (1, 2, 3, 4), (4, 2, 3, 0))
    rdm1_f_aa_ov += einsum(t2.abab, (0, 1, 2, 3), x7, (1, 3, 0, 4), (4, 2)) * -1.0
    x8 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x8 += einsum(x2, (0, 1), (0, 1))
    x8 += einsum(x1, (0, 1), (0, 1))
    x8 += einsum(x0, (0, 1), (0, 1)) * 2.0
    rdm1_f_aa_ov += einsum(t1.aa, (0, 1), x8, (0, 2), (2, 1)) * -1.0
    x9 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x9 += einsum(t1.bb, (0, 1), l2.abab, (2, 1, 3, 4), (4, 0, 3, 2))
    rdm1_f_bb_ov += einsum(t2.abab, (0, 1, 2, 3), x9, (1, 4, 0, 2), (4, 3)) * -1.0
    x10 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x10 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm1_f_bb_ov += einsum(t2.bbbb, (0, 1, 2, 3), x10, (1, 0, 4, 3), (4, 2)) * -2.0
    x11 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x11 += einsum(x5, (0, 1), (0, 1)) * 0.5
    x11 += einsum(x4, (0, 1), (0, 1))
    x11 += einsum(x3, (0, 1), (0, 1)) * 0.5
    rdm1_f_bb_ov += einsum(t1.bb, (0, 1), x11, (0, 2), (2, 1)) * -2.0

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
    rdm2_f_aaaa_oooo = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), delta.aa.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), delta.aa.oo, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_bbbb_oooo = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), delta.bb.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), delta.bb.oo, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_ovoo = np.zeros((nocc[0], nvir[0], nocc[0], nocc[0]), dtype=np.float64)
    rdm2_f_aaaa_ovoo += einsum(delta.aa.oo, (0, 1), l1.aa, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_ovoo += einsum(delta.aa.oo, (0, 1), l1.aa, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_abab_ovoo = np.zeros((nocc[0], nvir[1], nocc[0], nocc[1]), dtype=np.float64)
    rdm2_f_abab_ovoo += einsum(delta.aa.oo, (0, 1), l1.bb, (2, 3), (0, 2, 1, 3))
    rdm2_f_baba_ovoo = np.zeros((nocc[1], nvir[0], nocc[1], nocc[0]), dtype=np.float64)
    rdm2_f_baba_ovoo += einsum(delta.bb.oo, (0, 1), l1.aa, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_ovoo = np.zeros((nocc[1], nvir[1], nocc[1], nocc[1]), dtype=np.float64)
    rdm2_f_bbbb_ovoo += einsum(delta.bb.oo, (0, 1), l1.bb, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_ovoo += einsum(delta.bb.oo, (0, 1), l1.bb, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_vooo = np.zeros((nvir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    rdm2_f_aaaa_vooo += einsum(delta.aa.oo, (0, 1), l1.aa, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_aaaa_vooo += einsum(delta.aa.oo, (0, 1), l1.aa, (2, 3), (2, 0, 3, 1))
    rdm2_f_abab_vooo = np.zeros((nvir[0], nocc[1], nocc[0], nocc[1]), dtype=np.float64)
    rdm2_f_abab_vooo += einsum(delta.bb.oo, (0, 1), l1.aa, (2, 3), (2, 0, 3, 1))
    rdm2_f_baba_vooo = np.zeros((nvir[1], nocc[0], nocc[1], nocc[0]), dtype=np.float64)
    rdm2_f_baba_vooo += einsum(delta.aa.oo, (0, 1), l1.bb, (2, 3), (2, 0, 3, 1))
    rdm2_f_bbbb_vooo = np.zeros((nvir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    rdm2_f_bbbb_vooo += einsum(delta.bb.oo, (0, 1), l1.bb, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_bbbb_vooo += einsum(delta.bb.oo, (0, 1), l1.bb, (2, 3), (2, 0, 3, 1))
    rdm2_f_aaaa_oovv = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    rdm2_f_aaaa_oovv += einsum(t2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_abab_oovv = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    rdm2_f_abab_oovv += einsum(t2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_baba_oovv = np.zeros((nocc[1], nocc[0], nvir[1], nvir[0]), dtype=np.float64)
    rdm2_f_baba_oovv += einsum(t2.abab, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_bbbb_oovv = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    rdm2_f_bbbb_oovv += einsum(t2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_bbbb_oovv += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_bbbb_oovv += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_ovov = np.zeros((nocc[0], nvir[0], nocc[0], nvir[0]), dtype=np.float64)
    rdm2_f_aaaa_ovov += einsum(l1.aa, (0, 1), t1.aa, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_bbbb_ovov = np.zeros((nocc[1], nvir[1], nocc[1], nvir[1]), dtype=np.float64)
    rdm2_f_bbbb_ovov += einsum(l1.bb, (0, 1), t1.bb, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_aaaa_ovvo = np.zeros((nocc[0], nvir[0], nvir[0], nocc[0]), dtype=np.float64)
    rdm2_f_aaaa_ovvo += einsum(l1.aa, (0, 1), t1.aa, (2, 3), (2, 0, 3, 1))
    rdm2_f_abab_ovvo = np.zeros((nocc[0], nvir[1], nvir[0], nocc[1]), dtype=np.float64)
    rdm2_f_abab_ovvo += einsum(l1.bb, (0, 1), t1.aa, (2, 3), (2, 0, 3, 1))
    rdm2_f_baba_ovvo = np.zeros((nocc[1], nvir[0], nvir[1], nocc[0]), dtype=np.float64)
    rdm2_f_baba_ovvo += einsum(l1.aa, (0, 1), t1.bb, (2, 3), (2, 0, 3, 1))
    rdm2_f_bbbb_ovvo = np.zeros((nocc[1], nvir[1], nvir[1], nocc[1]), dtype=np.float64)
    rdm2_f_bbbb_ovvo += einsum(l1.bb, (0, 1), t1.bb, (2, 3), (2, 0, 3, 1))
    rdm2_f_aaaa_voov = np.zeros((nvir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    rdm2_f_aaaa_voov += einsum(l1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3))
    rdm2_f_abab_voov = np.zeros((nvir[0], nocc[1], nocc[0], nvir[1]), dtype=np.float64)
    rdm2_f_abab_voov += einsum(l1.aa, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    rdm2_f_baba_voov = np.zeros((nvir[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    rdm2_f_baba_voov += einsum(l1.bb, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_voov = np.zeros((nvir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    rdm2_f_bbbb_voov += einsum(l1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_vovo = np.zeros((nvir[0], nocc[0], nvir[0], nocc[0]), dtype=np.float64)
    rdm2_f_aaaa_vovo += einsum(l1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_bbbb_vovo = np.zeros((nvir[1], nocc[1], nvir[1], nocc[1]), dtype=np.float64)
    rdm2_f_bbbb_vovo += einsum(l1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_vvoo = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=np.float64)
    rdm2_f_aaaa_vvoo += einsum(l2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_abab_vvoo = np.zeros((nvir[0], nvir[1], nocc[0], nocc[1]), dtype=np.float64)
    rdm2_f_abab_vvoo += einsum(l2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_baba_vvoo = np.zeros((nvir[1], nvir[0], nocc[1], nocc[0]), dtype=np.float64)
    rdm2_f_baba_vvoo += einsum(l2.abab, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_bbbb_vvoo = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=np.float64)
    rdm2_f_bbbb_vvoo += einsum(l2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_aaaa_vvvv = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    rdm2_f_aaaa_vvvv += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 5), (0, 1, 4, 5)) * 2.0
    rdm2_f_bbbb_vvvv = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    rdm2_f_bbbb_vvvv += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 5), (0, 1, 4, 5)) * 2.0
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x0 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_aaaa_oooo += einsum(x0, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x1 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm2_f_aaaa_ovoo += einsum(x1, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0
    rdm2_f_aaaa_vooo += einsum(x1, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x2 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x2 += einsum(t1.aa, (0, 1), x1, (2, 3, 4, 1), (2, 3, 0, 4))
    rdm2_f_aaaa_oooo += einsum(x2, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0
    x3 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x3 += einsum(l1.aa, (0, 1), t1.aa, (2, 0), (1, 2))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x3, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x3, (2, 3), (3, 0, 1, 2))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x3, (2, 3), (0, 3, 2, 1))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x3, (2, 3), (3, 0, 2, 1)) * -1.0
    x4 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x4 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 1), (2, 4))
    x5 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x5 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 0, 1), (2, 4))
    x6 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x6 += einsum(x4, (0, 1), (0, 1))
    x6 += einsum(x5, (0, 1), (0, 1)) * 2.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x6, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x6, (2, 3), (0, 3, 2, 1))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x6, (2, 3), (3, 0, 1, 2))
    rdm2_f_aaaa_oooo += einsum(delta.aa.oo, (0, 1), x6, (2, 3), (3, 0, 2, 1)) * -1.0
    x7 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x7 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 0, 1), (3, 5, 2, 4))
    rdm2_f_abab_oooo = np.zeros((nocc[0], nocc[1], nocc[0], nocc[1]), dtype=np.float64)
    rdm2_f_abab_oooo += einsum(x7, (0, 1, 2, 3), (3, 1, 2, 0))
    rdm2_f_baba_oooo = np.zeros((nocc[1], nocc[0], nocc[1], nocc[0]), dtype=np.float64)
    rdm2_f_baba_oooo += einsum(x7, (0, 1, 2, 3), (1, 3, 0, 2))
    x8 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x8 += einsum(t1.aa, (0, 1), l2.abab, (1, 2, 3, 4), (4, 2, 3, 0))
    rdm2_f_abab_ovoo += einsum(x8, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    rdm2_f_baba_vooo += einsum(x8, (0, 1, 2, 3), (1, 3, 0, 2)) * -1.0
    x9 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x9 += einsum(t1.bb, (0, 1), x8, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_abab_oooo += einsum(x9, (0, 1, 2, 3), (3, 1, 2, 0))
    rdm2_f_baba_oooo += einsum(x9, (0, 1, 2, 3), (1, 3, 0, 2))
    x10 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x10 += einsum(l1.bb, (0, 1), t1.bb, (2, 0), (1, 2))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (3, 0, 1, 2))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (0, 3, 2, 1))
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x10, (2, 3), (3, 0, 2, 1)) * -1.0
    x11 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x11 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 0, 1), (2, 4))
    x12 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x12 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 1), (3, 4))
    x13 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x13 += einsum(delta.bb.oo, (0, 1), (0, 1)) * -0.5
    x13 += einsum(x10, (0, 1), (0, 1)) * 0.5
    x13 += einsum(x11, (0, 1), (0, 1))
    x13 += einsum(x12, (0, 1), (0, 1)) * 0.5
    rdm2_f_abab_oooo += einsum(delta.aa.oo, (0, 1), x13, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_abab_oovo = np.zeros((nocc[0], nocc[1], nvir[0], nocc[1]), dtype=np.float64)
    rdm2_f_abab_oovo += einsum(t1.aa, (0, 1), x13, (2, 3), (0, 3, 1, 2)) * -2.0
    x14 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x14 += einsum(x3, (0, 1), (0, 1))
    x14 += einsum(x4, (0, 1), (0, 1))
    x14 += einsum(x5, (0, 1), (0, 1)) * 2.0
    rdm2_f_abab_oooo += einsum(delta.bb.oo, (0, 1), x14, (2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_baba_oooo += einsum(delta.bb.oo, (0, 1), x14, (2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_baba_oovo = np.zeros((nocc[1], nocc[0], nvir[1], nocc[0]), dtype=np.float64)
    rdm2_f_baba_oovo += einsum(t1.bb, (0, 1), x14, (2, 3), (0, 3, 1, 2)) * -1.0
    x15 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x15 += einsum(delta.bb.oo, (0, 1), (0, 1)) * -1.0
    x15 += einsum(x10, (0, 1), (0, 1))
    x15 += einsum(x11, (0, 1), (0, 1)) * 2.0
    x15 += einsum(x12, (0, 1), (0, 1))
    rdm2_f_baba_oooo += einsum(delta.aa.oo, (0, 1), x15, (2, 3), (3, 0, 2, 1)) * -1.0
    x16 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x16 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_bbbb_oooo += einsum(x16, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x17 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x17 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 1, 3, 4), (3, 4, 0, 2))
    rdm2_f_bbbb_ovoo += einsum(x17, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0
    rdm2_f_bbbb_vooo += einsum(x17, (0, 1, 2, 3), (3, 2, 1, 0)) * 2.0
    x18 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x18 += einsum(t1.bb, (0, 1), x17, (2, 3, 4, 1), (2, 3, 0, 4))
    rdm2_f_bbbb_oooo += einsum(x18, (0, 1, 2, 3), (2, 3, 1, 0)) * -2.0
    x19 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x19 += einsum(x11, (0, 1), (0, 1))
    x19 += einsum(x12, (0, 1), (0, 1)) * 0.5
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x19, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x19, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x19, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_bbbb_oooo += einsum(delta.bb.oo, (0, 1), x19, (2, 3), (3, 0, 2, 1)) * -2.0
    x20 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x20 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_aaaa_ooov = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    rdm2_f_aaaa_ooov += einsum(x20, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_aaaa_oovo = np.zeros((nocc[0], nocc[0], nvir[0], nocc[0]), dtype=np.float64)
    rdm2_f_aaaa_oovo += einsum(x20, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x21 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x21 += einsum(t2.abab, (0, 1, 2, 3), x8, (1, 3, 4, 5), (4, 5, 0, 2))
    x22 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x22 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x23 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x23 += einsum(x3, (0, 1), (0, 1)) * 0.5
    x23 += einsum(x4, (0, 1), (0, 1)) * 0.5
    x23 += einsum(x5, (0, 1), (0, 1))
    rdm2_f_abab_ooov = np.zeros((nocc[0], nocc[1], nocc[0], nvir[1]), dtype=np.float64)
    rdm2_f_abab_ooov += einsum(t1.bb, (0, 1), x23, (2, 3), (3, 0, 2, 1)) * -2.0
    x24 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x24 += einsum(delta.aa.oo, (0, 1), t1.aa, (2, 3), (0, 1, 2, 3))
    x24 += einsum(x21, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x24 += einsum(x22, (0, 1, 2, 3), (1, 0, 2, 3)) * -4.0
    x24 += einsum(t1.aa, (0, 1), x23, (2, 3), (0, 2, 3, 1)) * 2.0
    rdm2_f_aaaa_ooov += einsum(x24, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_ooov += einsum(x24, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_aaaa_oovo += einsum(x24, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_oovo += einsum(x24, (0, 1, 2, 3), (2, 0, 3, 1))
    x25 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x25 += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 0), (2, 3))
    x26 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x26 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 0), (2, 3))
    x27 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x27 += einsum(t2.abab, (0, 1, 2, 3), x8, (1, 3, 0, 4), (4, 2))
    x28 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x28 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (0, 1, 4, 3), (4, 2)) * -1.0
    x29 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x29 += einsum(t1.aa, (0, 1), x23, (0, 2), (2, 1)) * 2.0
    x30 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x30 += einsum(x25, (0, 1), (0, 1)) * -1.0
    x30 += einsum(x26, (0, 1), (0, 1)) * -2.0
    x30 += einsum(x27, (0, 1), (0, 1))
    x30 += einsum(x28, (0, 1), (0, 1)) * 2.0
    x30 += einsum(x29, (0, 1), (0, 1))
    rdm2_f_aaaa_ooov += einsum(delta.aa.oo, (0, 1), x30, (2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_aaaa_ooov += einsum(delta.aa.oo, (0, 1), x30, (2, 3), (2, 0, 1, 3))
    rdm2_f_aaaa_oovo += einsum(delta.aa.oo, (0, 1), x30, (2, 3), (0, 2, 3, 1))
    rdm2_f_aaaa_oovo += einsum(delta.aa.oo, (0, 1), x30, (2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_abab_oovo += einsum(delta.bb.oo, (0, 1), x30, (2, 3), (2, 0, 3, 1)) * -1.0
    x31 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x31 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2))
    x31 += einsum(x2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_aaaa_oovv += einsum(t2.aaaa, (0, 1, 2, 3), x31, (0, 1, 4, 5), (5, 4, 2, 3)) * -2.0
    x32 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x32 += einsum(t1.aa, (0, 1), x31, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    rdm2_f_aaaa_ooov += einsum(x32, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_aaaa_oovo += einsum(x32, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x33 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x33 += einsum(l1.aa, (0, 1), t2.abab, (2, 3, 0, 4), (3, 4, 1, 2))
    rdm2_f_abab_ooov += einsum(x33, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_baba_oovo += einsum(x33, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    x34 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x34 += einsum(t1.bb, (0, 1), l2.abab, (2, 1, 3, 4), (4, 0, 3, 2))
    rdm2_f_baba_ovoo += einsum(x34, (0, 1, 2, 3), (1, 3, 0, 2)) * -1.0
    rdm2_f_abab_vooo += einsum(x34, (0, 1, 2, 3), (3, 1, 2, 0)) * -1.0
    x35 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x35 += einsum(t2.abab, (0, 1, 2, 3), x34, (1, 4, 5, 2), (4, 3, 5, 0))
    rdm2_f_abab_ooov += einsum(x35, (0, 1, 2, 3), (3, 0, 2, 1))
    rdm2_f_baba_oovo += einsum(x35, (0, 1, 2, 3), (0, 3, 1, 2))
    x36 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x36 += einsum(t2.bbbb, (0, 1, 2, 3), x8, (1, 3, 4, 5), (0, 2, 4, 5))
    rdm2_f_abab_ooov += einsum(x36, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_baba_oovo += einsum(x36, (0, 1, 2, 3), (0, 3, 1, 2)) * -2.0
    x37 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x37 += einsum(t2.abab, (0, 1, 2, 3), x1, (4, 0, 5, 2), (1, 3, 4, 5)) * -1.0
    rdm2_f_abab_ooov += einsum(x37, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_baba_oovo += einsum(x37, (0, 1, 2, 3), (0, 3, 1, 2)) * -2.0
    x38 = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x38 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3))
    x38 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    x39 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x39 += einsum(t1.bb, (0, 1), x38, (0, 2, 3, 4), (2, 1, 3, 4))
    rdm2_f_abab_ooov += einsum(x39, (0, 1, 2, 3), (3, 0, 2, 1))
    rdm2_f_baba_oovo += einsum(x39, (0, 1, 2, 3), (0, 3, 1, 2))
    x40 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x40 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 0), (2, 3))
    x41 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x41 += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 0, 3), (2, 3))
    x42 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x42 += einsum(t2.bbbb, (0, 1, 2, 3), x17, (0, 1, 4, 3), (4, 2)) * -1.0
    x43 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x43 += einsum(t2.abab, (0, 1, 2, 3), x34, (1, 4, 0, 2), (4, 3))
    x44 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x44 += einsum(x10, (0, 1), (0, 1))
    x44 += einsum(x11, (0, 1), (0, 1)) * 2.0
    x44 += einsum(x12, (0, 1), (0, 1))
    rdm2_f_baba_ooov = np.zeros((nocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    rdm2_f_baba_ooov += einsum(t1.aa, (0, 1), x44, (2, 3), (3, 0, 2, 1)) * -1.0
    x45 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x45 += einsum(t1.bb, (0, 1), x44, (0, 2), (2, 1)) * 0.5
    x46 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x46 += einsum(t1.bb, (0, 1), (0, 1)) * -0.5
    x46 += einsum(x40, (0, 1), (0, 1)) * -1.0
    x46 += einsum(x41, (0, 1), (0, 1)) * -0.5
    x46 += einsum(x42, (0, 1), (0, 1))
    x46 += einsum(x43, (0, 1), (0, 1)) * 0.5
    x46 += einsum(x45, (0, 1), (0, 1))
    rdm2_f_abab_ooov += einsum(delta.aa.oo, (0, 1), x46, (2, 3), (0, 2, 1, 3)) * -2.0
    rdm2_f_baba_oovo += einsum(delta.aa.oo, (0, 1), x46, (2, 3), (2, 0, 3, 1)) * -2.0
    x47 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x47 += einsum(t2.abab, (0, 1, 2, 3), x17, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    rdm2_f_baba_ooov += einsum(x47, (0, 1, 2, 3), (1, 2, 0, 3)) * -2.0
    rdm2_f_abab_oovo += einsum(x47, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    x48 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x48 += einsum(l1.bb, (0, 1), t2.abab, (2, 3, 4, 0), (1, 3, 2, 4))
    rdm2_f_baba_ooov += einsum(x48, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_abab_oovo += einsum(x48, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x49 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x49 += einsum(t2.abab, (0, 1, 2, 3), x8, (4, 3, 0, 5), (4, 1, 5, 2))
    rdm2_f_baba_ooov += einsum(x49, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_abab_oovo += einsum(x49, (0, 1, 2, 3), (2, 1, 3, 0))
    x50 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x50 += einsum(t2.aaaa, (0, 1, 2, 3), x34, (4, 5, 1, 3), (4, 5, 0, 2))
    rdm2_f_baba_ooov += einsum(x50, (0, 1, 2, 3), (1, 2, 0, 3)) * -2.0
    rdm2_f_abab_oovo += einsum(x50, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    x51 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x51 += einsum(t1.aa, (0, 1), x38, (2, 3, 0, 4), (2, 3, 4, 1))
    rdm2_f_baba_ooov += einsum(x51, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_abab_oovo += einsum(x51, (0, 1, 2, 3), (2, 1, 3, 0))
    x52 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x52 += einsum(t1.aa, (0, 1), (0, 1)) * -1.0
    x52 += einsum(x25, (0, 1), (0, 1)) * -1.0
    x52 += einsum(x26, (0, 1), (0, 1)) * -2.0
    x52 += einsum(x27, (0, 1), (0, 1))
    x52 += einsum(x28, (0, 1), (0, 1)) * 2.0
    x52 += einsum(x29, (0, 1), (0, 1))
    rdm2_f_baba_ooov += einsum(delta.bb.oo, (0, 1), x52, (2, 3), (0, 2, 1, 3)) * -1.0
    x53 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x53 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 3, 4, 0), (1, 2, 3, 4))
    rdm2_f_bbbb_ooov = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    rdm2_f_bbbb_ooov += einsum(x53, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_bbbb_oovo = np.zeros((nocc[1], nocc[1], nvir[1], nocc[1]), dtype=np.float64)
    rdm2_f_bbbb_oovo += einsum(x53, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x54 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x54 += einsum(x40, (0, 1), (0, 1)) * -1.0
    x54 += einsum(x41, (0, 1), (0, 1)) * -0.5
    x54 += einsum(x42, (0, 1), (0, 1))
    x54 += einsum(x43, (0, 1), (0, 1)) * 0.5
    x54 += einsum(x45, (0, 1), (0, 1))
    rdm2_f_bbbb_ooov += einsum(delta.bb.oo, (0, 1), x54, (2, 3), (0, 2, 1, 3)) * -2.0
    rdm2_f_bbbb_ooov += einsum(delta.bb.oo, (0, 1), x54, (2, 3), (2, 0, 1, 3)) * 2.0
    rdm2_f_bbbb_oovo += einsum(delta.bb.oo, (0, 1), x54, (2, 3), (0, 2, 3, 1)) * 2.0
    rdm2_f_bbbb_oovo += einsum(delta.bb.oo, (0, 1), x54, (2, 3), (2, 0, 3, 1)) * -2.0
    rdm2_f_baba_oovv += einsum(t1.aa, (0, 1), x54, (2, 3), (2, 0, 3, 1)) * -2.0
    x55 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x55 += einsum(t2.bbbb, (0, 1, 2, 3), x17, (4, 1, 5, 3), (4, 5, 0, 2)) * -1.0
    x56 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x56 += einsum(t2.abab, (0, 1, 2, 3), x34, (4, 5, 0, 2), (4, 5, 1, 3))
    x57 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x57 += einsum(delta.bb.oo, (0, 1), t1.bb, (2, 3), (0, 1, 2, 3))
    x57 += einsum(x55, (0, 1, 2, 3), (1, 0, 2, 3)) * -4.0
    x57 += einsum(x56, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x57 += einsum(t1.bb, (0, 1), x44, (2, 3), (0, 2, 3, 1))
    rdm2_f_bbbb_ooov += einsum(x57, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_ooov += einsum(x57, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_bbbb_oovo += einsum(x57, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_bbbb_oovo += einsum(x57, (0, 1, 2, 3), (2, 0, 3, 1))
    x58 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x58 += einsum(x16, (0, 1, 2, 3), (1, 0, 3, 2))
    x58 += einsum(x18, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_bbbb_oovv += einsum(t2.bbbb, (0, 1, 2, 3), x58, (0, 1, 4, 5), (5, 4, 2, 3)) * -2.0
    x59 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x59 += einsum(t1.bb, (0, 1), x58, (0, 2, 3, 4), (2, 3, 4, 1)) * 2.0
    rdm2_f_bbbb_ooov += einsum(x59, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_bbbb_oovo += einsum(x59, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x60 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x60 += einsum(l2.abab, (0, 1, 2, 3), t2.aaaa, (4, 2, 5, 0), (3, 1, 4, 5))
    rdm2_f_abab_ovvo += einsum(x60, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    rdm2_f_baba_voov += einsum(x60, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    x61 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x61 += einsum(t2.abab, (0, 1, 2, 3), x60, (1, 3, 4, 5), (0, 4, 2, 5))
    x62 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x62 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3))
    x62 += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    x63 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x63 += einsum(t1.aa, (0, 1), x62, (0, 2, 3, 4), (2, 3, 1, 4))
    x64 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x64 += einsum(x61, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x64 += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3))
    x64 += einsum(t1.aa, (0, 1), x30, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_oovv += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x64, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_aaaa_oovv += einsum(x64, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_aaaa_oovv += einsum(x64, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x65 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x65 += einsum(x3, (0, 1), t2.aaaa, (2, 0, 3, 4), (1, 2, 3, 4))
    x66 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x66 += einsum(x6, (0, 1), t2.aaaa, (2, 0, 3, 4), (2, 1, 3, 4)) * -2.0
    x67 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x67 += einsum(x65, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x67 += einsum(x66, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_aaaa_oovv += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x67, (0, 1, 2, 3), (1, 0, 2, 3))
    x68 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x68 += einsum(t1.aa, (0, 1), x20, (0, 2, 3, 4), (2, 3, 1, 4))
    x69 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x69 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_aaaa_ovov += einsum(x69, (0, 1, 2, 3), (1, 2, 0, 3)) * -4.0
    rdm2_f_aaaa_ovvo += einsum(x69, (0, 1, 2, 3), (1, 2, 3, 0)) * 4.0
    rdm2_f_aaaa_voov += einsum(x69, (0, 1, 2, 3), (2, 1, 0, 3)) * 4.0
    rdm2_f_aaaa_vovo += einsum(x69, (0, 1, 2, 3), (2, 1, 3, 0)) * -4.0
    x70 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x70 += einsum(t2.aaaa, (0, 1, 2, 3), x69, (1, 4, 3, 5), (0, 4, 2, 5))
    x71 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x71 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 1), (0, 4))
    x72 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x72 += einsum(l2.aaaa, (0, 1, 2, 3), t2.aaaa, (2, 3, 4, 1), (0, 4))
    x73 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x73 += einsum(x71, (0, 1), (0, 1))
    x73 += einsum(x72, (0, 1), (0, 1)) * 2.0
    x74 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x74 += einsum(x73, (0, 1), t2.aaaa, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    x75 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x75 += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x75 += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3)) * 8.0
    x75 += einsum(x74, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x75, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_aaaa_oovv += einsum(x75, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x76 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x76 += einsum(l2.bbbb, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 0, 4, 5))
    rdm2_f_abab_ovvo += einsum(x76, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    rdm2_f_baba_voov += einsum(x76, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    x77 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x77 += einsum(t2.abab, (0, 1, 2, 3), x76, (1, 3, 4, 5), (4, 0, 5, 2))
    x78 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x78 += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3))
    x78 += einsum(x77, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    rdm2_f_aaaa_oovv += einsum(x78, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_aaaa_oovv += einsum(x78, (0, 1, 2, 3), (0, 1, 2, 3))
    x79 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x79 += einsum(t1.aa, (0, 1), x31, (0, 2, 3, 4), (2, 4, 3, 1))
    rdm2_f_aaaa_oovv += einsum(t1.aa, (0, 1), x79, (0, 2, 3, 4), (2, 3, 4, 1)) * -2.0
    x80 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x80 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 0, 5), (1, 5, 2, 4))
    rdm2_f_abab_ovov = np.zeros((nocc[0], nvir[1], nocc[0], nvir[1]), dtype=np.float64)
    rdm2_f_abab_ovov += einsum(x80, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_baba_vovo = np.zeros((nvir[1], nocc[0], nvir[1], nocc[0]), dtype=np.float64)
    rdm2_f_baba_vovo += einsum(x80, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    x81 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x81 += einsum(t2.abab, (0, 1, 2, 3), x80, (3, 4, 0, 5), (1, 4, 5, 2))
    rdm2_f_abab_oovv += einsum(x81, (0, 1, 2, 3), (2, 0, 3, 1))
    rdm2_f_baba_oovv += einsum(x81, (0, 1, 2, 3), (0, 2, 1, 3))
    x82 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x82 += einsum(l2.abab, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (4, 5, 2, 0))
    rdm2_f_baba_ovvo += einsum(x82, (0, 1, 2, 3), (0, 3, 1, 2)) * 2.0
    rdm2_f_abab_voov += einsum(x82, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x83 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x83 += einsum(l2.aaaa, (0, 1, 2, 3), t2.abab, (3, 4, 1, 5), (4, 5, 2, 0))
    rdm2_f_baba_ovvo += einsum(x83, (0, 1, 2, 3), (0, 3, 1, 2)) * 2.0
    rdm2_f_abab_voov += einsum(x83, (0, 1, 2, 3), (3, 0, 2, 1)) * 2.0
    x84 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x84 += einsum(x82, (0, 1, 2, 3), (0, 1, 2, 3))
    x84 += einsum(x83, (0, 1, 2, 3), (0, 1, 2, 3))
    x85 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x85 += einsum(t2.aaaa, (0, 1, 2, 3), x84, (4, 5, 1, 3), (4, 5, 0, 2)) * 4.0
    rdm2_f_abab_oovv += einsum(x85, (0, 1, 2, 3), (2, 0, 3, 1))
    rdm2_f_baba_oovv += einsum(x85, (0, 1, 2, 3), (0, 2, 1, 3))
    x86 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x86 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_bbbb_ovov += einsum(x86, (0, 1, 2, 3), (1, 2, 0, 3)) * -4.0
    rdm2_f_bbbb_ovvo += einsum(x86, (0, 1, 2, 3), (1, 2, 3, 0)) * 4.0
    rdm2_f_bbbb_voov += einsum(x86, (0, 1, 2, 3), (2, 1, 0, 3)) * 4.0
    rdm2_f_bbbb_vovo += einsum(x86, (0, 1, 2, 3), (2, 1, 3, 0)) * -4.0
    x87 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x87 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 0, 5), (3, 4, 1, 5))
    rdm2_f_bbbb_ovov += einsum(x87, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_bbbb_ovvo += einsum(x87, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_bbbb_voov += einsum(x87, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_bbbb_vovo += einsum(x87, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x88 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x88 += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    x88 += einsum(x87, (0, 1, 2, 3), (0, 1, 2, 3))
    x89 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x89 += einsum(t2.abab, (0, 1, 2, 3), x88, (1, 4, 3, 5), (4, 5, 0, 2))
    rdm2_f_abab_oovv += einsum(x89, (0, 1, 2, 3), (2, 0, 3, 1))
    rdm2_f_baba_oovv += einsum(x89, (0, 1, 2, 3), (0, 2, 1, 3))
    x90 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x90 += einsum(t2.abab, (0, 1, 2, 3), x38, (1, 4, 0, 5), (4, 3, 5, 2))
    rdm2_f_abab_oovv += einsum(x90, (0, 1, 2, 3), (2, 0, 3, 1))
    rdm2_f_baba_oovv += einsum(x90, (0, 1, 2, 3), (0, 2, 1, 3))
    x91 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x91 += einsum(t1.bb, (0, 1), x38, (0, 2, 3, 4), (2, 1, 3, 4)) * 0.5
    x92 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x92 += einsum(x33, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x92 += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3))
    x92 += einsum(x35, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x92 += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3))
    x92 += einsum(x91, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x93 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x93 += einsum(t1.aa, (0, 1), x92, (2, 3, 0, 4), (2, 3, 4, 1)) * 2.0
    rdm2_f_abab_oovv += einsum(x93, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_baba_oovv += einsum(x93, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x94 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x94 += einsum(x71, (0, 1), (0, 1)) * 0.5
    x94 += einsum(x72, (0, 1), (0, 1))
    x95 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x95 += einsum(x94, (0, 1), t2.abab, (2, 3, 0, 4), (3, 4, 2, 1)) * 2.0
    rdm2_f_abab_oovv += einsum(x95, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_baba_oovv += einsum(x95, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x96 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x96 += einsum(l2.bbbb, (0, 1, 2, 3), t2.bbbb, (2, 3, 4, 1), (0, 4))
    x97 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x97 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 0, 4), (1, 4))
    x98 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x98 += einsum(x96, (0, 1), (0, 1)) * 2.0
    x98 += einsum(x97, (0, 1), (0, 1))
    x99 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x99 += einsum(x98, (0, 1), t2.abab, (2, 3, 4, 0), (3, 1, 2, 4))
    rdm2_f_abab_oovv += einsum(x99, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_baba_oovv += einsum(x99, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x100 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x100 += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x100 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3))
    x100 += einsum(x50, (0, 1, 2, 3), (0, 1, 2, 3))
    x100 += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    rdm2_f_abab_oovv += einsum(t1.bb, (0, 1), x100, (0, 2, 3, 4), (3, 2, 4, 1)) * -2.0
    x101 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x101 += einsum(x14, (0, 1), t2.abab, (0, 2, 3, 4), (2, 4, 1, 3))
    rdm2_f_abab_oovv += einsum(x101, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_baba_oovv += einsum(x101, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x102 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x102 += einsum(x10, (0, 1), (0, 1)) * 0.5
    x102 += einsum(x11, (0, 1), (0, 1))
    x102 += einsum(x12, (0, 1), (0, 1)) * 0.5
    x103 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x103 += einsum(x102, (0, 1), t2.abab, (2, 0, 3, 4), (1, 4, 2, 3)) * 2.0
    rdm2_f_abab_oovv += einsum(x103, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_baba_oovv += einsum(x103, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x104 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x104 += einsum(t1.aa, (0, 1), x23, (0, 2), (2, 1))
    x105 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x105 += einsum(t1.aa, (0, 1), (0, 1)) * -0.5
    x105 += einsum(x25, (0, 1), (0, 1)) * -0.5
    x105 += einsum(x26, (0, 1), (0, 1)) * -1.0
    x105 += einsum(x27, (0, 1), (0, 1)) * 0.5
    x105 += einsum(x28, (0, 1), (0, 1))
    x105 += einsum(x104, (0, 1), (0, 1))
    rdm2_f_abab_oovv += einsum(t1.bb, (0, 1), x105, (2, 3), (2, 0, 3, 1)) * -2.0
    rdm2_f_baba_oovv += einsum(t1.bb, (0, 1), x105, (2, 3), (0, 2, 1, 3)) * -2.0
    x106 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x106 += einsum(x40, (0, 1), (0, 1)) * -2.0
    x106 += einsum(x41, (0, 1), (0, 1)) * -1.0
    x106 += einsum(x42, (0, 1), (0, 1)) * 2.0
    x106 += einsum(x43, (0, 1), (0, 1))
    x106 += einsum(t1.bb, (0, 1), x44, (0, 2), (2, 1))
    rdm2_f_abab_oovv += einsum(t1.aa, (0, 1), x106, (2, 3), (0, 2, 1, 3)) * -1.0
    x107 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x107 += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3))
    x107 += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x107 += einsum(x50, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x107 += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_baba_oovv += einsum(t1.bb, (0, 1), x107, (0, 2, 3, 4), (2, 3, 1, 4)) * -1.0
    x108 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x108 += einsum(t2.bbbb, (0, 1, 2, 3), x87, (1, 4, 3, 5), (0, 4, 2, 5))
    x109 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x109 += einsum(x55, (0, 1, 2, 3), (0, 1, 2, 3))
    x109 += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    x110 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x110 += einsum(t1.bb, (0, 1), x109, (0, 2, 3, 4), (2, 3, 1, 4)) * 4.0
    x111 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x111 += einsum(x108, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    x111 += einsum(x110, (0, 1, 2, 3), (0, 1, 2, 3))
    x111 += einsum(t1.bb, (0, 1), x54, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_bbbb_oovv += einsum(x111, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_bbbb_oovv += einsum(x111, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_bbbb_oovv += einsum(x111, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_bbbb_oovv += einsum(x111, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x112 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x112 += einsum(x10, (0, 1), t2.bbbb, (2, 0, 3, 4), (1, 2, 3, 4))
    x113 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x113 += einsum(x19, (0, 1), t2.bbbb, (2, 0, 3, 4), (2, 1, 3, 4)) * -4.0
    x114 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x114 += einsum(x112, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x114 += einsum(x113, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_bbbb_oovv += einsum(x114, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_bbbb_oovv += einsum(x114, (0, 1, 2, 3), (1, 0, 2, 3))
    x115 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x115 += einsum(t1.bb, (0, 1), x53, (0, 2, 3, 4), (2, 3, 1, 4))
    x116 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x116 += einsum(t2.bbbb, (0, 1, 2, 3), x86, (1, 4, 3, 5), (0, 4, 2, 5))
    x117 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x117 += einsum(t2.abab, (0, 1, 2, 3), x83, (4, 5, 0, 2), (4, 1, 5, 3))
    x118 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x118 += einsum(x96, (0, 1), (0, 1))
    x118 += einsum(x97, (0, 1), (0, 1)) * 0.5
    x119 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x119 += einsum(x118, (0, 1), t2.bbbb, (2, 3, 4, 0), (2, 3, 4, 1)) * -4.0
    x120 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x120 += einsum(x115, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x120 += einsum(x116, (0, 1, 2, 3), (0, 1, 2, 3)) * 8.0
    x120 += einsum(x117, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x120 += einsum(x119, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_bbbb_oovv += einsum(x120, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_oovv += einsum(x120, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x121 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x121 += einsum(t1.bb, (0, 1), x58, (0, 2, 3, 4), (2, 4, 3, 1))
    rdm2_f_bbbb_oovv += einsum(t1.bb, (0, 1), x121, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    x122 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x122 += einsum(t1.aa, (0, 1), x1, (2, 0, 3, 4), (2, 3, 4, 1))
    rdm2_f_aaaa_ovov += einsum(x122, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_aaaa_ovvo += einsum(x122, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    rdm2_f_aaaa_voov += einsum(x122, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_aaaa_vovo += einsum(x122, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x123 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x123 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_aaaa_ovov += einsum(x123, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_aaaa_ovvo += einsum(x123, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_aaaa_voov += einsum(x123, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_aaaa_vovo += einsum(x123, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x124 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x124 += einsum(l1.aa, (0, 1), t1.aa, (1, 2), (0, 2))
    x125 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x125 += einsum(x124, (0, 1), (0, 1))
    x125 += einsum(x71, (0, 1), (0, 1))
    x125 += einsum(x72, (0, 1), (0, 1)) * 2.0
    rdm2_f_aaaa_ovov += einsum(delta.aa.oo, (0, 1), x125, (2, 3), (0, 2, 1, 3))
    rdm2_f_baba_ovov = np.zeros((nocc[1], nvir[0], nocc[1], nvir[0]), dtype=np.float64)
    rdm2_f_baba_ovov += einsum(delta.bb.oo, (0, 1), x125, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_ovvo += einsum(delta.aa.oo, (0, 1), x125, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_aaaa_voov += einsum(delta.aa.oo, (0, 1), x125, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_aaaa_vovo += einsum(delta.aa.oo, (0, 1), x125, (2, 3), (2, 0, 3, 1))
    rdm2_f_abab_vovo = np.zeros((nvir[0], nocc[1], nvir[0], nocc[1]), dtype=np.float64)
    rdm2_f_abab_vovo += einsum(delta.bb.oo, (0, 1), x125, (2, 3), (2, 0, 3, 1))
    rdm2_f_baba_ovvv = np.zeros((nocc[1], nvir[0], nvir[1], nvir[0]), dtype=np.float64)
    rdm2_f_baba_ovvv += einsum(t1.bb, (0, 1), x125, (2, 3), (0, 2, 1, 3))
    rdm2_f_abab_vovv = np.zeros((nvir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    rdm2_f_abab_vovv += einsum(t1.bb, (0, 1), x125, (2, 3), (2, 0, 3, 1))
    x126 = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x126 += einsum(t1.bb, (0, 1), x8, (0, 2, 3, 4), (2, 1, 3, 4))
    rdm2_f_abab_ovov += einsum(x126, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_baba_vovo += einsum(x126, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    x127 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x127 += einsum(l1.bb, (0, 1), t1.bb, (1, 2), (0, 2))
    x128 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x128 += einsum(x127, (0, 1), (0, 1)) * 0.5
    x128 += einsum(x96, (0, 1), (0, 1))
    x128 += einsum(x97, (0, 1), (0, 1)) * 0.5
    rdm2_f_abab_ovov += einsum(delta.aa.oo, (0, 1), x128, (2, 3), (0, 2, 1, 3)) * 2.0
    rdm2_f_bbbb_ovvo += einsum(delta.bb.oo, (0, 1), x128, (2, 3), (0, 2, 3, 1)) * -2.0
    rdm2_f_bbbb_vovo += einsum(delta.bb.oo, (0, 1), x128, (2, 3), (2, 0, 3, 1)) * 2.0
    rdm2_f_abab_ovvv = np.zeros((nocc[0], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
    rdm2_f_abab_ovvv += einsum(t1.aa, (0, 1), x128, (2, 3), (0, 2, 1, 3)) * 2.0
    x129 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x129 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 4, 5, 1), (3, 4, 0, 5))
    rdm2_f_baba_ovov += einsum(x129, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_abab_vovo += einsum(x129, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x130 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x130 += einsum(t1.aa, (0, 1), x34, (2, 3, 0, 4), (2, 3, 4, 1))
    rdm2_f_baba_ovov += einsum(x130, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_abab_vovo += einsum(x130, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x131 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x131 += einsum(t1.bb, (0, 1), x17, (2, 0, 3, 4), (2, 3, 4, 1))
    rdm2_f_bbbb_ovov += einsum(x131, (0, 1, 2, 3), (1, 2, 0, 3)) * 2.0
    rdm2_f_bbbb_ovvo += einsum(x131, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    rdm2_f_bbbb_voov += einsum(x131, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_bbbb_vovo += einsum(x131, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    x132 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x132 += einsum(x127, (0, 1), (0, 1))
    x132 += einsum(x96, (0, 1), (0, 1)) * 2.0
    x132 += einsum(x97, (0, 1), (0, 1))
    rdm2_f_bbbb_ovov += einsum(delta.bb.oo, (0, 1), x132, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_voov += einsum(delta.bb.oo, (0, 1), x132, (2, 3), (2, 0, 1, 3)) * -1.0
    rdm2_f_baba_vovo += einsum(delta.aa.oo, (0, 1), x132, (2, 3), (2, 0, 3, 1))
    rdm2_f_baba_vovv = np.zeros((nvir[1], nocc[0], nvir[1], nvir[0]), dtype=np.float64)
    rdm2_f_baba_vovv += einsum(t1.aa, (0, 1), x132, (2, 3), (2, 0, 3, 1))
    x133 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x133 += einsum(t1.aa, (0, 1), x8, (2, 3, 0, 4), (2, 3, 4, 1))
    rdm2_f_abab_ovvo += einsum(x133, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_baba_voov += einsum(x133, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    x134 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x134 += einsum(t1.bb, (0, 1), x34, (0, 2, 3, 4), (2, 1, 3, 4))
    rdm2_f_baba_ovvo += einsum(x134, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    rdm2_f_abab_voov += einsum(x134, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    x135 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x135 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (0, 1, 4, 5), (4, 5, 2, 3))
    rdm2_f_aaaa_ovvv = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    rdm2_f_aaaa_ovvv += einsum(x135, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_aaaa_vovv = np.zeros((nvir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    rdm2_f_aaaa_vovv += einsum(x135, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    x136 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x136 += einsum(t1.aa, (0, 1), x122, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    rdm2_f_aaaa_ovvv += einsum(x136, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_aaaa_vovv += einsum(x136, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    x137 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x137 += einsum(l1.aa, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_aaaa_ovvv += einsum(x137, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_aaaa_vovv += einsum(x137, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    x138 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x138 += einsum(x123, (0, 1, 2, 3), (0, 1, 2, 3))
    x138 += einsum(x69, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    x139 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x139 += einsum(t1.aa, (0, 1), x138, (0, 2, 3, 4), (2, 1, 3, 4))
    x140 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x140 += einsum(x139, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x140 += einsum(t1.aa, (0, 1), x125, (2, 3), (0, 2, 1, 3))
    rdm2_f_aaaa_ovvv += einsum(x140, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_aaaa_ovvv += einsum(x140, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_aaaa_vovv += einsum(x140, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_aaaa_vovv += einsum(x140, (0, 1, 2, 3), (1, 0, 3, 2))
    x141 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x141 += einsum(l1.bb, (0, 1), t2.abab, (2, 1, 3, 4), (0, 4, 2, 3))
    rdm2_f_abab_ovvv += einsum(x141, (0, 1, 2, 3), (2, 0, 3, 1))
    rdm2_f_baba_vovv += einsum(x141, (0, 1, 2, 3), (0, 2, 1, 3))
    x142 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x142 += einsum(t2.abab, (0, 1, 2, 3), x8, (1, 4, 0, 5), (4, 3, 5, 2))
    rdm2_f_abab_ovvv += einsum(x142, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_baba_vovv += einsum(x142, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x143 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x143 += einsum(t1.aa, (0, 1), x80, (2, 3, 0, 4), (2, 3, 4, 1))
    rdm2_f_abab_ovvv += einsum(x143, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    rdm2_f_baba_vovv += einsum(x143, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x144 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x144 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x144 += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x144 += einsum(x133, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_abab_ovvv += einsum(t1.bb, (0, 1), x144, (0, 2, 3, 4), (3, 2, 4, 1))
    x145 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    x145 += einsum(l1.aa, (0, 1), t2.abab, (1, 2, 3, 4), (2, 4, 0, 3))
    rdm2_f_baba_ovvv += einsum(x145, (0, 1, 2, 3), (0, 2, 1, 3))
    rdm2_f_abab_vovv += einsum(x145, (0, 1, 2, 3), (2, 0, 3, 1))
    x146 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    x146 += einsum(t2.abab, (0, 1, 2, 3), x34, (1, 4, 0, 5), (4, 3, 5, 2))
    rdm2_f_baba_ovvv += einsum(x146, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_abab_vovv += einsum(x146, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    x147 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    x147 += einsum(t1.bb, (0, 1), x129, (0, 2, 3, 4), (2, 1, 3, 4))
    rdm2_f_baba_ovvv += einsum(x147, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    rdm2_f_abab_vovv += einsum(x147, (0, 1, 2, 3), (2, 0, 3, 1)) * -1.0
    x148 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x148 += einsum(x82, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x148 += einsum(x134, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x148 += einsum(x83, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    rdm2_f_baba_ovvv += einsum(t1.aa, (0, 1), x148, (2, 3, 0, 4), (2, 4, 3, 1))
    x149 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x149 += einsum(l1.bb, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4))
    rdm2_f_bbbb_ovvv = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    rdm2_f_bbbb_ovvv += einsum(x149, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_bbbb_vovv = np.zeros((nvir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    rdm2_f_bbbb_vovv += einsum(x149, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    x150 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x150 += einsum(t2.bbbb, (0, 1, 2, 3), x17, (0, 1, 4, 5), (4, 5, 2, 3))
    rdm2_f_bbbb_ovvv += einsum(x150, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_bbbb_vovv += einsum(x150, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    x151 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x151 += einsum(t1.bb, (0, 1), x131, (0, 2, 3, 4), (2, 3, 4, 1)) * -1.0
    rdm2_f_bbbb_ovvv += einsum(x151, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    rdm2_f_bbbb_vovv += einsum(x151, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    x152 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x152 += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3))
    x152 += einsum(x87, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    x153 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x153 += einsum(t1.bb, (0, 1), x152, (0, 2, 3, 4), (2, 1, 3, 4)) * 4.0
    x154 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x154 += einsum(x153, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x154 += einsum(t1.bb, (0, 1), x132, (2, 3), (0, 2, 1, 3))
    rdm2_f_bbbb_ovvv += einsum(x154, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_bbbb_ovvv += einsum(x154, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_bbbb_vovv += einsum(x154, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_bbbb_vovv += einsum(x154, (0, 1, 2, 3), (1, 0, 3, 2))
    x155 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x155 += einsum(x82, (0, 1, 2, 3), (0, 1, 2, 3))
    x155 += einsum(x134, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x155 += einsum(x83, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_abab_vovv += einsum(t1.aa, (0, 1), x155, (2, 3, 0, 4), (4, 2, 1, 3)) * 2.0
    x156 = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x156 += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3))
    x156 += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3))
    x156 += einsum(x133, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    rdm2_f_baba_vovv += einsum(t1.bb, (0, 1), x156, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    x157 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x157 += einsum(t1.aa, (0, 1), l2.aaaa, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_aaaa_vvov = np.zeros((nvir[0], nvir[0], nocc[0], nvir[0]), dtype=np.float64)
    rdm2_f_aaaa_vvov += einsum(x157, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_aaaa_vvvo = np.zeros((nvir[0], nvir[0], nvir[0], nocc[0]), dtype=np.float64)
    rdm2_f_aaaa_vvvo += einsum(x157, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    rdm2_f_aaaa_vvvv += einsum(t1.aa, (0, 1), x157, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    x158 = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x158 += einsum(t1.bb, (0, 1), l2.abab, (2, 3, 4, 0), (3, 1, 4, 2))
    rdm2_f_abab_vvov = np.zeros((nvir[0], nvir[1], nocc[0], nvir[1]), dtype=np.float64)
    rdm2_f_abab_vvov += einsum(x158, (0, 1, 2, 3), (3, 0, 2, 1))
    rdm2_f_baba_vvvo = np.zeros((nvir[1], nvir[0], nvir[1], nocc[0]), dtype=np.float64)
    rdm2_f_baba_vvvo += einsum(x158, (0, 1, 2, 3), (0, 3, 1, 2))
    x159 = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    x159 += einsum(t1.aa, (0, 1), l2.abab, (2, 3, 0, 4), (4, 3, 2, 1))
    rdm2_f_baba_vvov = np.zeros((nvir[1], nvir[0], nocc[1], nvir[0]), dtype=np.float64)
    rdm2_f_baba_vvov += einsum(x159, (0, 1, 2, 3), (1, 2, 0, 3))
    rdm2_f_abab_vvvo = np.zeros((nvir[0], nvir[1], nvir[0], nocc[1]), dtype=np.float64)
    rdm2_f_abab_vvvo += einsum(x159, (0, 1, 2, 3), (2, 1, 3, 0))
    x160 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x160 += einsum(t1.bb, (0, 1), l2.bbbb, (2, 3, 4, 0), (4, 2, 3, 1))
    rdm2_f_bbbb_vvov = np.zeros((nvir[1], nvir[1], nocc[1], nvir[1]), dtype=np.float64)
    rdm2_f_bbbb_vvov += einsum(x160, (0, 1, 2, 3), (2, 1, 0, 3)) * -2.0
    rdm2_f_bbbb_vvvo = np.zeros((nvir[1], nvir[1], nvir[1], nocc[1]), dtype=np.float64)
    rdm2_f_bbbb_vvvo += einsum(x160, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    rdm2_f_bbbb_vvvv += einsum(t1.bb, (0, 1), x160, (0, 2, 3, 4), (2, 3, 1, 4)) * 2.0
    x161 = np.zeros((nvir[1], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    x161 += einsum(l2.abab, (0, 1, 2, 3), t2.abab, (2, 3, 4, 5), (1, 5, 0, 4))
    rdm2_f_abab_vvvv = np.zeros((nvir[0], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
    rdm2_f_abab_vvvv += einsum(x161, (0, 1, 2, 3), (2, 0, 3, 1))
    rdm2_f_baba_vvvv = np.zeros((nvir[1], nvir[0], nvir[1], nvir[0]), dtype=np.float64)
    rdm2_f_baba_vvvv += einsum(x161, (0, 1, 2, 3), (0, 2, 1, 3))
    x162 = np.zeros((nvir[1], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    x162 += einsum(t1.bb, (0, 1), x159, (0, 2, 3, 4), (2, 1, 3, 4))
    rdm2_f_abab_vvvv += einsum(x162, (0, 1, 2, 3), (2, 0, 3, 1))
    rdm2_f_baba_vvvv += einsum(x162, (0, 1, 2, 3), (0, 2, 1, 3))

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

