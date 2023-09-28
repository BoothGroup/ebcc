# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 2), ()) * -1.0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    t2new = Namespace()

    # T amplitudes
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    t2new_aaaa += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    t2new_aaaa += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.oovv, (4, 0, 5, 3), (4, 1, 2, 5)) * -1.0
    t2new_abab += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -2.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x1 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x1 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x1 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2))
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum(t2.aaaa, (0, 1, 2, 3), x1, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x1
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x3 += einsum(x0, (0, 1, 2, 3), (0, 1, 2, 3))
    x3 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3))
    del x2
    t2new_aaaa += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x3, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_aaaa += einsum(x3, (0, 1, 2, 3), (1, 0, 3, 2))
    del x3
    x4 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x4 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x4 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x5 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x5 += einsum(t2.aaaa, (0, 1, 2, 3), x4, (1, 4, 3, 5), (0, 4, 2, 5))
    x6 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x6 += einsum(t2.aaaa, (0, 1, 2, 3), x5, (4, 1, 5, 3), (0, 4, 2, 5)) * -4.0
    del x5
    x7 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x7 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x7 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x8 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x8 += einsum(t2.abab, (0, 1, 2, 3), x7, (1, 4, 3, 5), (0, 4, 2, 5))
    del x7
    x9 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x9 += einsum(t2.abab, (0, 1, 2, 3), x8, (4, 1, 5, 3), (0, 4, 2, 5)) * -1.0
    del x8
    x10 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x10 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x11 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x11 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    x12 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x12 += einsum(x10, (0, 1), (0, 1))
    x12 += einsum(x11, (0, 1), (0, 1)) * 0.5
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x13 += einsum(x12, (0, 1), t2.aaaa, (2, 3, 4, 1), (2, 3, 4, 0)) * -4.0
    del x12
    x14 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x14 += einsum(x6, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x6
    x14 += einsum(x9, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x9
    x14 += einsum(x13, (0, 1, 2, 3), (1, 0, 2, 3))
    del x13
    t2new_aaaa += einsum(x14, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa += einsum(x14, (0, 1, 2, 3), (0, 1, 3, 2))
    del x14
    x15 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x15 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x16 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x16 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    x17 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x17 += einsum(x15, (0, 1), (0, 1))
    x17 += einsum(x16, (0, 1), (0, 1)) * 0.5
    x18 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x18 += einsum(x17, (0, 1), t2.aaaa, (2, 1, 3, 4), (2, 0, 3, 4)) * -4.0
    del x17
    t2new_aaaa += einsum(x18, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa += einsum(x18, (0, 1, 2, 3), (1, 0, 3, 2))
    del x18
    x19 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x19 += einsum(v.aaaa.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x19 += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 5, 2), (4, 0, 1, 5)) * -1.0
    t2new_aaaa += einsum(t2.aaaa, (0, 1, 2, 3), x19, (0, 4, 5, 1), (5, 4, 2, 3)) * -2.0
    del x19
    x20 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x20 += einsum(t2.bbbb, (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (4, 0, 5, 2))
    t2new_abab += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x21 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x21 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x21 += einsum(v.aaaa.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x21 += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2))
    del x0
    x21 += einsum(t2.aaaa, (0, 1, 2, 3), x4, (1, 4, 3, 5), (4, 0, 5, 2)) * -2.0
    del x4
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x21, (0, 4, 2, 5), (4, 1, 5, 3))
    del x21
    x22 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x22 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x22 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x23 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x23 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x23 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x23 += einsum(t2.bbbb, (0, 1, 2, 3), x22, (1, 4, 3, 5), (4, 0, 5, 2)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x23, (1, 4, 3, 5), (0, 4, 2, 5)) * -2.0
    del x23
    x24 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x24 += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x24 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x20
    t2new_abab += einsum(t2.aaaa, (0, 1, 2, 3), x24, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    x25 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x25 += einsum(v.aabb.vvoo, (0, 1, 2, 3), (2, 3, 0, 1))
    x25 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 4, 5, 3), (5, 1, 4, 2)) * -1.0
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x25, (1, 4, 2, 5), (0, 4, 5, 3)) * -1.0
    del x25
    x26 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x26 += einsum(v.aabb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x26 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (4, 0, 5, 1))
    t2new_abab += einsum(t2.abab, (0, 1, 2, 3), x26, (0, 4, 1, 5), (4, 5, 2, 3))
    del x26
    x27 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x27 += einsum(x10, (0, 1), (0, 1)) * 2.0
    del x10
    x27 += einsum(x11, (0, 1), (0, 1))
    del x11
    t2new_abab += einsum(x27, (0, 1), t2.abab, (2, 3, 1, 4), (2, 3, 0, 4)) * -1.0
    del x27
    x28 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x28 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    x29 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x29 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x30 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x30 += einsum(x28, (0, 1), (0, 1))
    del x28
    x30 += einsum(x29, (0, 1), (0, 1)) * 2.0
    del x29
    t2new_abab += einsum(x30, (0, 1), t2.abab, (2, 3, 4, 1), (2, 3, 4, 0)) * -1.0
    x31 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x31 += einsum(x15, (0, 1), (0, 1)) * 2.0
    del x15
    x31 += einsum(x16, (0, 1), (0, 1))
    del x16
    t2new_abab += einsum(x31, (0, 1), t2.abab, (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    del x31
    x32 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x32 += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    x33 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x33 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x34 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x34 += einsum(x32, (0, 1), (0, 1))
    del x32
    x34 += einsum(x33, (0, 1), (0, 1)) * 2.0
    del x33
    t2new_abab += einsum(x34, (0, 1), t2.abab, (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    x35 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x35 += einsum(t2.bbbb, (0, 1, 2, 3), x22, (1, 4, 5, 3), (0, 4, 2, 5))
    del x22
    x36 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x36 += einsum(t2.bbbb, (0, 1, 2, 3), x35, (4, 1, 5, 3), (0, 4, 2, 5)) * -4.0
    del x35
    x37 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x37 += einsum(x30, (0, 1), t2.bbbb, (2, 3, 4, 1), (2, 3, 4, 0)) * -2.0
    del x30
    x38 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x38 += einsum(x36, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x36
    x38 += einsum(x37, (0, 1, 2, 3), (1, 0, 2, 3))
    del x37
    t2new_bbbb += einsum(x38, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb += einsum(x38, (0, 1, 2, 3), (0, 1, 3, 2))
    del x38
    x39 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x39 += einsum(x34, (0, 1), t2.bbbb, (2, 1, 3, 4), (2, 0, 3, 4)) * -2.0
    del x34
    t2new_bbbb += einsum(x39, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x39, (0, 1, 2, 3), (1, 0, 3, 2))
    del x39
    x40 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x40 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x40 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x41 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x41 += einsum(t2.abab, (0, 1, 2, 3), x40, (0, 4, 5, 2), (4, 1, 5, 3))
    del x40
    x42 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x42 += einsum(t2.abab, (0, 1, 2, 3), x41, (0, 4, 2, 5), (4, 1, 5, 3)) * -1.0
    del x41
    x43 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x43 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x43 += einsum(x42, (0, 1, 2, 3), (1, 0, 3, 2))
    del x42
    t2new_bbbb += einsum(x43, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x43, (0, 1, 2, 3), (0, 1, 2, 3))
    del x43
    x44 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x44 += einsum(t2.abab, (0, 1, 2, 3), x24, (0, 4, 2, 5), (4, 1, 5, 3))
    del x24
    x45 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x45 += einsum(v.bbbb.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x45 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x46 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x46 += einsum(t2.bbbb, (0, 1, 2, 3), x45, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x45
    x47 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x47 += einsum(x44, (0, 1, 2, 3), (1, 0, 3, 2))
    del x44
    x47 += einsum(x46, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x46
    t2new_bbbb += einsum(x47, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb += einsum(x47, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb += einsum(x47, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb += einsum(x47, (0, 1, 2, 3), (1, 0, 3, 2))
    del x47
    x48 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x48 += einsum(v.bbbb.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x48 += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 5, 2), (4, 0, 1, 5)) * -1.0
    t2new_bbbb += einsum(t2.bbbb, (0, 1, 2, 3), x48, (0, 4, 5, 1), (5, 4, 2, 3)) * -2.0
    del x48

    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb

    return {"t2new": t2new}

