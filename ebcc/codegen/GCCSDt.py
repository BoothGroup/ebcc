# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(f.ov, (0, 1), t1, (0, 1), ())
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    e_cc += einsum(v.oovv, (0, 1, 2, 3), x0, (0, 1, 2, 3), ()) * 0.25
    del x0

    return e_cc

def update_amps(f=None, v=None, space=None, t1=None, t2=None, t3=None, **kwargs):
    nocc = space.ncocc
    nvir = space.ncvir
    naocc = space.naocc
    navir = space.navir
    so = np.ones((nocc,), dtype=bool)
    sv = np.ones((nvir,), dtype=bool)
    sO = space.active[space.correlated][space.occupied[space.correlated]]
    sV = space.active[space.correlated][space.virtual[space.correlated]]
    sOf = np.ones((naocc,), dtype=bool)
    sVf = np.ones((navir,), dtype=bool)

    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new[np.ix_(so,sv)] += einsum(f.ov, (0, 1), (0, 1))
    t1new[np.ix_(so,sv)] += einsum(f.oo, (0, 1), t1[np.ix_(so,sv)], (1, 2), (0, 2)) * -1.0
    t1new[np.ix_(so,sv)] += einsum(f.vv, (0, 1), t1[np.ix_(so,sv)], (2, 1), (2, 0))
    t1new[np.ix_(so,sv)] += einsum(f.ov, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovov, (2, 1, 0, 3), (2, 3)) * -1.0
    t1new[np.ix_(so,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (0, 1, 4, 3), (4, 2)) * -0.5
    t1new[np.ix_(so,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvv, (1, 4, 2, 3), (0, 4)) * -0.5
    t1new[np.ix_(so,sv)] += einsum(v.oOvV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 1, 5, 2, 3), (4, 5)) * 0.25
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new[np.ix_(so,so,sv,sv)] += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(f.OV, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oooo, (4, 5, 0, 1), (4, 5, 2, 3)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.vvvv, (4, 5, 2, 3), (0, 1, 4, 5)) * 0.5
    t3new = np.zeros((nocc, nocc, naocc, nvir, nvir, navir), dtype=types[float])
    t3new += einsum(f.OO, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -1.0
    t3new += einsum(f.VV, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oooV, (4, 5, 0, 6), (4, 5, 1, 2, 3, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.vvvO, (4, 5, 2, 6), (0, 1, 6, 4, 5, 3)) * -1.0
    t3new += einsum(v.oooo, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 7), (0, 1, 4, 5, 6, 7)) * 0.5
    t3new += einsum(v.OVOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 2, 6, 7, 1), (4, 5, 0, 6, 7, 3)) * -1.0
    t3new += einsum(v.vvvv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 2, 3, 7), (4, 5, 6, 0, 1, 7)) * 0.5
    x0 = np.zeros((nocc, nocc), dtype=types[float])
    x0 += einsum(f.ov, (0, 1), t1[np.ix_(so,sv)], (2, 1), (0, 2))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x0, (0, 2), (2, 1)) * -1.0
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ooov, (2, 0, 3, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x1, (0, 2), (2, 1)) * -1.0
    x2 = np.zeros((nvir, nvir), dtype=types[float])
    x2 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvv, (0, 2, 3, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x2, (2, 1), (0, 2)) * -1.0
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x3 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    t1new[np.ix_(so,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x3, (4, 0, 1, 3), (4, 2)) * 0.5
    x4 = np.zeros((nocc, nocc), dtype=types[float])
    x4 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x4, (2, 0), (2, 1)) * -0.5
    x5 = np.zeros((nocc, nvir), dtype=types[float])
    x5 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oovv, (2, 0, 3, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(x5, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 1), (2, 3))
    x6 = np.zeros((nocc, nocc), dtype=types[float])
    x6 += einsum(t1[np.ix_(so,sv)], (0, 1), x5, (2, 1), (0, 2))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x6, (2, 0), (2, 1)) * -1.0
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(f.oo, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x7, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x7, (0, 1, 2, 3), (1, 0, 3, 2))
    del x7
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(f.vv, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x8, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x8, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x9, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x9, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x9
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x10, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x10
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x11, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x11, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x11, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x11
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(v.oOoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 1, 5, 6, 3), (4, 2, 5, 6))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x12, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x12, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    del x12
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(v.vOvV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 2, 3), (4, 5, 6, 0))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x13, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x13
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(x0, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 4), (1, 2, 3, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x14, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x14, (0, 1, 2, 3), (1, 0, 3, 2))
    del x14
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum(f.ov, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (0, 2, 3, 4))
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(t1[np.ix_(so,sv)], (0, 1), x15, (0, 2, 3, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x16, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x16
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x17, (2, 3, 0, 4), (2, 3, 1, 4))
    del x17
    x18 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x18 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(t1[np.ix_(so,sv)], (0, 1), x18, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x19, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x19, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x19, (0, 1, 2, 3), (1, 0, 3, 2))
    del x19
    x20 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x20 += einsum(t1[np.ix_(so,sv)], (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x20, (2, 3, 4, 1), (0, 2, 4, 3)) * -1.0
    del x20
    x21 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x21 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x21, (4, 0, 1, 5), (4, 5, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x22, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    del x22
    x23 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x23 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum(t1[np.ix_(so,sv)], (0, 1), x23, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x24, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x24, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x24, (0, 1, 2, 3), (1, 0, 3, 2))
    del x24
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(x1, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x25, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x25, (0, 1, 2, 3), (1, 0, 3, 2))
    del x25
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x26, (4, 1, 5, 3), (4, 0, 2, 5)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x27, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x27, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x27, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x27, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x27
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvv, (4, 5, 2, 3), (0, 1, 4, 5))
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum(t1[np.ix_(so,sv)], (0, 1), x28, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x29, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    del x29
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum(x2, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 4, 0)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x30, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3))
    del x30
    x31 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x31 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvV, (2, 3, 1, 4), (3, 4, 0, 2))
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum(x31, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 0, 5, 6, 1), (2, 4, 5, 6))
    del x31
    t2new[np.ix_(so,so,sv,sv)] += einsum(x32, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x32, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    del x32
    x33 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x33 += einsum(v.oOvV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 2, 3), (4, 5, 0, 6))
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x34 += einsum(t1[np.ix_(so,sv)], (0, 1), x33, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x34, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    del x34
    x35 = np.zeros((naocc, navir), dtype=types[float])
    x35 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvV, (0, 2, 1, 3), (2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x35, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    del x35
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x36, (4, 1, 5, 3), (4, 0, 5, 2))
    del x36
    t2new[np.ix_(so,so,sv,sv)] += einsum(x37, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x37, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x37
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum(x4, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x38, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x38, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    del x38
    x39 = np.zeros((nvir, nvir), dtype=types[float])
    x39 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum(x39, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x40, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    del x40
    x41 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x41 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (0, 1, 4, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x41, (4, 5, 0, 1), (5, 4, 2, 3)) * -0.25
    t3new += einsum(x41, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 7), (1, 0, 4, 5, 6, 7)) * -0.25
    x42 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x42 += einsum(t1[np.ix_(so,sv)], (0, 1), x21, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum(t1[np.ix_(so,sv)], (0, 1), x42, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x42
    t2new[np.ix_(so,so,sv,sv)] += einsum(x43, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x43, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x43
    x44 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x44 += einsum(t1[np.ix_(so,sv)], (0, 1), x26, (2, 3, 4, 1), (2, 0, 3, 4)) * -1.0
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum(t1[np.ix_(so,sv)], (0, 1), x44, (2, 3, 0, 4), (3, 2, 1, 4)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x45, (0, 1, 2, 3), (0, 1, 3, 2))
    del x45
    x46 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x46 += einsum(t1[np.ix_(so,sv)], (0, 1), x3, (2, 3, 4, 1), (2, 0, 4, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x46, (4, 5, 0, 1), (5, 4, 2, 3)) * -0.5
    t3new += einsum(x46, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 7), (1, 0, 4, 5, 6, 7)) * -0.5
    x47 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x47 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x3, (4, 1, 5, 3), (4, 0, 5, 2))
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum(t1[np.ix_(so,sv)], (0, 1), x47, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x48, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x48, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x48, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x48
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum(x6, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (0, 2, 3, 4)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x49, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x49, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x49
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x50 += einsum(t1[np.ix_(so,sv)], (0, 1), x41, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x50, (2, 3, 0, 4), (2, 3, 1, 4)) * 0.5
    del x50
    x51 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x51 += einsum(x5, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    x52 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x52 += einsum(t1[np.ix_(so,sv)], (0, 1), x51, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x52, (0, 1, 2, 3), (0, 1, 3, 2))
    del x52
    x53 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x53 += einsum(t1[np.ix_(so,sv)], (0, 1), x46, (2, 3, 4, 0), (2, 3, 4, 1))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x53, (2, 3, 0, 4), (3, 2, 1, 4)) * -1.0
    del x53
    x54 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum(f.oo, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5))
    t3new += einsum(x54, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x54, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x54
    x55 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum(f.vv, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x55
    x56 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoO, (1, 4, 5, 6), (6, 3, 0, 5, 2, 4))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x56
    x57 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oOoV, (4, 5, 1, 6), (5, 6, 0, 4, 2, 3))
    t3new += einsum(x57, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x57, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x57
    x58 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x58 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    t3new += einsum(x58, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x58, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x58
    x59 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvV, (4, 3, 5, 6), (1, 6, 0, 4, 2, 5))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x59
    x60 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovvv, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    t3new += einsum(x60, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x60, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x60
    x61 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.vOvV, (3, 4, 5, 6), (4, 6, 0, 1, 2, 5))
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x61, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x61
    x62 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x62 += einsum(v.oOoO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 3, 5, 6, 7), (1, 7, 4, 0, 5, 6))
    t3new += einsum(x62, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x62, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    del x62
    x63 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 6, 1, 7), (5, 7, 4, 0, 6, 3))
    t3new += einsum(x63, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x63, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x63, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x63, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x63
    x64 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum(v.oVoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 6, 7, 1), (5, 3, 4, 0, 6, 7))
    t3new += einsum(x64, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x64, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x64
    x65 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x65 += einsum(v.vOvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 2, 7), (3, 7, 4, 5, 6, 0))
    t3new += einsum(x65, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x65, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x65
    x66 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x66 += einsum(v.vVvV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 7, 2, 3), (6, 1, 4, 5, 7, 0))
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    del x66
    x67 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x67 += einsum(x0, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 0, 3, 4, 5, 6), (3, 6, 1, 2, 4, 5))
    del x0
    t3new += einsum(x67, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x67, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x67
    x68 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x68 += einsum(f.ov, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 1, 6), (4, 6, 0, 2, 3, 5))
    x69 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x69 += einsum(t1[np.ix_(so,sv)], (0, 1), x68, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 1, 6))
    del x68
    t3new += einsum(x69, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x69, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x69
    x70 = np.zeros((navir, navir), dtype=types[float])
    x70 += einsum(f.oV, (0, 1), t1[np.ix_(so,sV)], (0, 2), (1, 2))
    t3new += einsum(x70, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * -1.0
    del x70
    x71 = np.zeros((naocc, naocc), dtype=types[float])
    x71 += einsum(f.vO, (0, 1), t1[np.ix_(sO,sv)], (2, 0), (1, 2))
    t3new += einsum(x71, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * -1.0
    del x71
    x72 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x72 += einsum(f.ov, (0, 1), t2[np.ix_(so,sO,sv,sV)], (2, 3, 1, 4), (3, 4, 0, 2))
    x73 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x72, (4, 5, 1, 6), (4, 5, 6, 0, 2, 3))
    del x72
    t3new += einsum(x73, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x73, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x73
    x74 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x74 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x15, (0, 4, 5, 6), (1, 3, 4, 5, 2, 6))
    del x15
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x74
    x75 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x75 += einsum(f.ov, (0, 1), t2[np.ix_(so,sO,sv,sv)], (2, 3, 4, 1), (3, 0, 2, 4)) * -1.0
    x76 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x75, (4, 1, 5, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x75
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x76
    x77 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x77 += einsum(f.ov, (0, 1), t2[np.ix_(so,so,sv,sV)], (2, 3, 1, 4), (4, 0, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x77, (4, 0, 5, 6), (5, 6, 1, 2, 3, 4)) * -1.0
    del x77
    x78 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x78 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oooO, (2, 0, 3, 4), (4, 3, 2, 1)) * -1.0
    x79 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x78, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2))
    del x78
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x79, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x79
    x80 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x80 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvO, (2, 3, 1, 4), (4, 0, 2, 3)) * -1.0
    x81 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x80, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    del x80
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x81, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x81
    x82 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x82 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovvO, (4, 5, 2, 6), (6, 3, 0, 1, 4, 5))
    x83 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x83 += einsum(t1[np.ix_(so,sv)], (0, 1), x82, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x82
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x83, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x83
    x84 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x84 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oVvO, (2, 3, 1, 4), (4, 3, 0, 2))
    x85 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x85 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x84, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x84
    t3new += einsum(x85, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x85, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x85
    x86 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x86 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oVvO, (4, 5, 3, 6), (6, 5, 0, 1, 4, 2))
    x87 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x87 += einsum(t1[np.ix_(so,sv)], (0, 1), x86, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x86
    t3new += einsum(x87, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x87, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x87
    x88 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x88 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oooo, (4, 5, 6, 0), (1, 3, 4, 5, 6, 2))
    x89 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x89 += einsum(t1[np.ix_(so,sv)], (0, 1), x88, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x88
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x89, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x89
    x90 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x90 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovoV, (2, 1, 3, 4), (4, 0, 3, 2)) * -1.0
    x91 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x91 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x90, (4, 5, 0, 6), (1, 4, 5, 6, 2, 3)) * -1.0
    del x90
    t3new += einsum(x91, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x91, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x91
    x92 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x92 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x18, (4, 0, 5, 6), (1, 3, 4, 5, 2, 6))
    del x18
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x92
    x93 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x93 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovoV, (4, 3, 5, 6), (1, 6, 0, 5, 4, 2))
    x94 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x94 += einsum(t1[np.ix_(so,sv)], (0, 1), x93, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x93
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x94
    x95 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x95 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovov, (4, 5, 6, 2), (1, 3, 0, 4, 6, 5))
    x96 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x96 += einsum(t1[np.ix_(so,sv)], (0, 1), x95, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x95
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x96
    x97 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x97 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oooO, (2, 0, 3, 4), (4, 1, 3, 2)) * -1.0
    x98 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x98 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x97, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3)) * -1.0
    del x97
    t3new += einsum(x98, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x98, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x98
    x99 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x99 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovvO, (0, 2, 3, 4), (4, 1, 3, 2)) * -1.0
    x100 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x100 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x99, (4, 5, 3, 6), (4, 5, 0, 1, 2, 6))
    del x99
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x100
    x101 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x101 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oooo, (2, 3, 4, 0), (1, 2, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x101, (4, 5, 6, 0), (5, 6, 1, 2, 3, 4)) * -1.0
    del x101
    x102 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x102 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovov, (2, 3, 0, 4), (1, 2, 3, 4))
    x103 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x103 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x102, (4, 5, 3, 6), (1, 4, 0, 5, 2, 6)) * -1.0
    del x102
    t3new += einsum(x103, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x103, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x103, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x103, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x103
    x104 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x104 += einsum(t1[np.ix_(so,sv)], (0, 1), v.vvvV, (2, 1, 3, 4), (4, 0, 3, 2)) * -1.0
    x105 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x105 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x104, (4, 5, 6, 3), (1, 4, 5, 0, 2, 6))
    del x104
    t3new += einsum(x105, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x105, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x105, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x105, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x105
    x106 = np.zeros((naocc, navir, nocc, nvir, nvir, nvir), dtype=types[float])
    x106 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.vvvv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x107 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x107 += einsum(t1[np.ix_(so,sv)], (0, 1), x106, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x106
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x107, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x107
    x108 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x108 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovoV, (2, 1, 3, 4), (0, 4, 3, 2)) * -1.0
    x109 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x109 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x108, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3))
    del x108
    t3new += einsum(x109, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x109, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x109
    x110 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x110 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x111 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x111 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x110, (4, 1, 5, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x110
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x111
    x112 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x112 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vvvV, (2, 1, 3, 4), (0, 4, 3, 2)) * -1.0
    x113 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x113 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x112, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6)) * -1.0
    del x112
    t3new += einsum(x113, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x113, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x113
    x114 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x114 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x114, (4, 5, 6, 2), (0, 1, 4, 6, 5, 3))
    del x114
    x115 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x115 += einsum(x21, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 2, 4, 5, 6, 7), (4, 7, 0, 3, 5, 6))
    t3new += einsum(x115, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x115, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    del x115
    x116 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x116 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvO, (2, 3, 1, 4), (3, 4, 0, 2))
    x117 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x117 += einsum(x116, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 0, 5, 6, 7), (1, 7, 2, 4, 5, 6))
    del x116
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    del x117
    x118 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x118 += einsum(v.ooov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 6, 3, 7), (5, 7, 4, 0, 2, 6))
    x119 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x119 += einsum(t1[np.ix_(so,sv)], (0, 1), x118, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x118
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x119
    x120 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x120 += einsum(x1, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 0, 3, 4, 5, 6), (3, 6, 2, 1, 4, 5)) * -1.0
    del x1
    t3new += einsum(x120, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x120, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x120
    x121 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x121 += einsum(v.oOvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 2, 7), (3, 7, 4, 5, 0, 6))
    x122 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x122 += einsum(t1[np.ix_(so,sv)], (0, 1), x121, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x121
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x122
    x123 = np.zeros((naocc, naocc), dtype=types[float])
    x123 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvO, (0, 2, 1, 3), (2, 3))
    t3new += einsum(x123, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * -1.0
    del x123
    x124 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x124 += einsum(x26, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 6, 3, 7), (5, 7, 0, 4, 6, 2)) * -1.0
    del x26
    t3new += einsum(x124, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x124, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x124, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x124, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x124
    x125 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x125 += einsum(v.ovvv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 2, 3, 7), (6, 7, 4, 5, 0, 1))
    x126 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x126 += einsum(t1[np.ix_(so,sv)], (0, 1), x125, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x125
    t3new += einsum(x126, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x126, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    del x126
    x127 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x127 += einsum(x2, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 5, 0)) * -1.0
    del x2
    t3new += einsum(x127, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x127, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x127
    x128 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x128 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oVvV, (2, 3, 1, 4), (3, 4, 0, 2))
    x129 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x129 += einsum(x128, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x128
    t3new += einsum(x129, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x129, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x129
    x130 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x130 += einsum(v.oVvV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 7, 2, 3), (6, 1, 4, 5, 0, 7))
    x131 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x131 += einsum(t1[np.ix_(so,sv)], (0, 1), x130, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x130
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    del x131
    x132 = np.zeros((navir, navir), dtype=types[float])
    x132 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oVvV, (0, 2, 1, 3), (2, 3))
    t3new += einsum(x132, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0))
    del x132
    x133 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x133 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oooV, (2, 0, 3, 4), (1, 4, 3, 2)) * -1.0
    x134 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x134 += einsum(x133, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 5, 6, 7, 1), (5, 0, 4, 2, 6, 7)) * -1.0
    del x133
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x134
    x135 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x135 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oOOV, (0, 2, 3, 4), (3, 2, 1, 4)) * -1.0
    t3new += einsum(x135, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    del x135
    x136 = np.zeros((navir, navir, nvir, nvir), dtype=types[float])
    x136 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovvV, (0, 2, 3, 4), (1, 4, 3, 2)) * -1.0
    x137 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x137 += einsum(x136, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 7, 2, 1), (6, 0, 4, 5, 7, 3))
    del x136
    t3new += einsum(x137, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x137, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    del x137
    x138 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x138 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovoO, (2, 1, 3, 4), (0, 4, 3, 2)) * -1.0
    x139 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x139 += einsum(x138, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 1, 5, 6, 7), (0, 7, 4, 3, 5, 6))
    del x138
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    del x139
    x140 = np.zeros((naocc, naocc, nvir, nvir), dtype=types[float])
    x140 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vvvO, (2, 1, 3, 4), (0, 4, 3, 2)) * -1.0
    x141 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x141 += einsum(x140, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 6, 2)) * -1.0
    del x140
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x141
    x142 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x142 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vVOV, (1, 2, 3, 4), (0, 3, 4, 2)) * -1.0
    t3new += einsum(x142, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    del x142
    x143 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x143 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oovO, (4, 1, 3, 5), (5, 0, 4, 2)) * -1.0
    x144 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x144 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x143, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x143
    t3new += einsum(x144, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x144, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x144, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x144, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x144
    x145 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x145 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oovO, (4, 1, 2, 5), (5, 3, 0, 4))
    x146 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x146 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x145, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3)) * -1.0
    del x145
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x146
    x147 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x147 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oovO, (0, 1, 4, 5), (5, 3, 2, 4))
    x148 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x148 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x147, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x147
    t3new += einsum(x148, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x148, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    del x148
    x149 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x149 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oovO, (0, 1, 4, 5), (5, 2, 3, 4)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x149, (4, 5, 6, 2), (0, 1, 4, 6, 5, 3)) * -0.5
    del x149
    x150 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x150 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x23, (4, 0, 5, 6), (1, 3, 4, 5, 2, 6))
    del x23
    t3new += einsum(x150, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x150, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x150, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x150, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x150
    x151 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x151 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 0, 5, 2), (1, 3, 4, 5))
    x152 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x152 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x151, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3)) * -1.0
    del x151
    t3new += einsum(x152, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x152, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x152
    x153 = np.zeros((naocc, navir, nocc, nocc, nocc, nocc), dtype=types[float])
    x153 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x154 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x154 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x153, (4, 5, 6, 1, 0, 7), (4, 5, 6, 7, 2, 3)) * -1.0
    t3new += einsum(x154, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x154, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    del x154
    x155 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x155 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 0, 5, 3), (1, 4, 5, 2)) * -1.0
    x156 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x156 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x155, (4, 1, 5, 6), (4, 3, 0, 5, 6, 2))
    del x155
    t3new += einsum(x156, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x156, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x156, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x156, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x156
    x157 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x157 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 1, 5, 2), (3, 0, 4, 5)) * -1.0
    x158 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x158 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x157, (4, 5, 0, 6), (1, 4, 5, 6, 2, 3)) * -1.0
    del x157
    t3new += einsum(x158, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x158, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x158
    x159 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x159 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ooov, (0, 1, 4, 5), (3, 4, 2, 5)) * -1.0
    x160 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x160 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x159, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x159
    t3new += einsum(x160, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    t3new += einsum(x160, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x160, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x160, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -0.5
    del x160
    x161 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x161 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovvv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x162 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x162 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x161, (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 2, 7)) * -1.0
    t3new += einsum(x162, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x162, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x162, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x162, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x162
    x163 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x163 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovvv, (0, 4, 5, 2), (1, 3, 4, 5))
    x164 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x164 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x163, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6)) * -1.0
    del x163
    t3new += einsum(x164, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x164, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x164
    x165 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x165 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x28, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x28
    t3new += einsum(x165, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x165, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    del x165
    x166 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x166 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvv, (4, 5, 2, 3), (1, 0, 4, 5)) * -1.0
    x167 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x167 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x166, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    del x166
    t3new += einsum(x167, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    t3new += einsum(x167, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x167, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x167, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    del x167
    x168 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x168 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovvv, (1, 4, 5, 2), (3, 0, 4, 5)) * -1.0
    x169 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x169 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x168, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x168
    t3new += einsum(x169, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x169, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x169, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x169, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x169
    x170 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x170 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvv, (0, 4, 5, 3), (1, 2, 4, 5)) * -1.0
    x171 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x171 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x170, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6))
    del x170
    t3new += einsum(x171, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x171, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x171
    x172 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x172 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oVvv, (1, 4, 5, 3), (4, 0, 2, 5)) * -1.0
    x173 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x173 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x172, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x172
    t3new += einsum(x173, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x173, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x173, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x173, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x173
    x174 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x174 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oVvv, (4, 5, 2, 3), (1, 5, 0, 4))
    x175 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x175 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x174, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x174
    t3new += einsum(x175, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x175, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    del x175
    x176 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x176 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oVvv, (0, 4, 5, 3), (1, 4, 2, 5))
    x177 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x177 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x176, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2)) * -1.0
    del x176
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x177
    x178 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x178 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oVvv, (4, 5, 2, 3), (5, 0, 1, 4)) * -1.0
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x178, (4, 5, 6, 0), (5, 6, 1, 2, 3, 4)) * -0.5
    del x178
    x179 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x179 += einsum(v.oovv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 1, 4, 5, 3, 6), (4, 6, 5, 2))
    x180 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x180 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x179, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x179
    t3new += einsum(x180, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x180, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    del x180
    x181 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x181 += einsum(v.oovv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 2, 3, 6), (5, 6, 4, 0))
    x182 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x182 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x181, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x181
    t3new += einsum(x182, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x182, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    del x182
    x183 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x183 += einsum(v.oovv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 6, 3, 7), (5, 7, 4, 0, 6, 2))
    x184 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x184 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x183, (4, 5, 6, 1, 7, 3), (4, 5, 0, 6, 2, 7))
    del x183
    t3new += einsum(x184, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x184, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x184, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x184, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x184
    x185 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x185 += einsum(x4, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5)) * -1.0
    del x4
    t3new += einsum(x185, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x185, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    del x185
    x186 = np.zeros((naocc, navir, nocc, nocc, nocc, nocc), dtype=types[float])
    x186 += einsum(v.oovv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 2, 3, 7), (6, 7, 4, 5, 0, 1))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x186, (4, 5, 6, 7, 0, 1), (6, 7, 4, 2, 3, 5)) * 0.25
    x187 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x187 += einsum(x39, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5)) * -1.0
    del x39
    t3new += einsum(x187, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x187, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    del x187
    x188 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x188 += einsum(v.oovV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 1, 4, 5, 6, 3), (4, 5, 6, 2)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x188, (4, 5, 6, 2), (0, 1, 4, 5, 6, 3)) * 0.5
    del x188
    x189 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x189 += einsum(v.oovV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 6, 2, 3), (5, 4, 0, 6)) * -1.0
    x190 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x190 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x189, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x189
    t3new += einsum(x190, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x190, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    t3new += einsum(x190, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x190, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    del x190
    x191 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x191 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oovV, (4, 1, 2, 5), (3, 5, 0, 4))
    x192 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x192 += einsum(x191, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7)) * -1.0
    del x191
    t3new += einsum(x192, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x192, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x192
    x193 = np.zeros((navir, navir, nvir, nvir), dtype=types[float])
    x193 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oovV, (0, 1, 4, 5), (3, 5, 2, 4))
    x194 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x194 += einsum(x193, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 7, 3, 1), (6, 0, 4, 5, 2, 7))
    del x193
    t3new += einsum(x194, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.25
    t3new += einsum(x194, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.25
    del x194
    x195 = np.zeros((navir, navir), dtype=types[float])
    x195 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oovV, (0, 1, 2, 4), (3, 4))
    t3new += einsum(x195, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -0.5
    del x195
    x196 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x196 += einsum(v.oOvv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 1, 5, 3, 6), (6, 4, 5, 2)) * -1.0
    x197 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x197 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x196, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x196
    t3new += einsum(x197, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x197, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    t3new += einsum(x197, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x197, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    del x197
    x198 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x198 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oOvv, (4, 5, 2, 3), (1, 5, 0, 4))
    x199 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x199 += einsum(x198, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 1, 5, 6, 7), (0, 7, 2, 4, 5, 6))
    del x198
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.25
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.25
    del x199
    x200 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x200 += einsum(v.oOvv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 2, 3, 6), (6, 4, 5, 0)) * -1.0
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x200, (4, 5, 6, 0), (5, 6, 1, 2, 3, 4)) * 0.5
    del x200
    x201 = np.zeros((naocc, naocc, nvir, nvir), dtype=types[float])
    x201 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oOvv, (0, 4, 5, 3), (1, 4, 2, 5))
    x202 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x202 += einsum(x201, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6)) * -1.0
    del x201
    t3new += einsum(x202, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x202, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x202
    x203 = np.zeros((naocc, naocc), dtype=types[float])
    x203 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oOvv, (0, 4, 2, 3), (1, 4))
    t3new += einsum(x203, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -0.5
    del x203
    x204 = np.zeros((naocc, naocc, navir, navir, nocc, nocc), dtype=types[float])
    x204 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oOvV, (4, 5, 2, 6), (1, 5, 3, 6, 0, 4))
    x205 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x205 += einsum(t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 1, 2, 3, 4, 5), x204, (6, 2, 7, 5, 8, 1), (6, 7, 8, 0, 3, 4))
    del x204
    t3new += einsum(x205, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    t3new += einsum(x205, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    del x205
    x206 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x206 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x33, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x33
    t3new += einsum(x206, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x206, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    del x206
    x207 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x207 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oOvV, (0, 4, 2, 5), (1, 4, 3, 5))
    t3new += einsum(x207, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    del x207
    x208 = np.zeros((naocc, nocc, nocc, nocc), dtype=types[float])
    x208 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oovO, (2, 3, 1, 4), (4, 0, 2, 3)) * -1.0
    x209 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x209 += einsum(t1[np.ix_(so,sv)], (0, 1), x208, (2, 3, 4, 0), (2, 3, 4, 1))
    del x208
    x210 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x210 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x209, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    del x209
    t3new += einsum(x210, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x210, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x210, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x210, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x210
    x211 = np.zeros((naocc, navir, nocc, nocc, nocc, nocc), dtype=types[float])
    x211 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oovO, (4, 5, 2, 6), (6, 3, 0, 1, 4, 5))
    x212 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x212 += einsum(t1[np.ix_(so,sv)], (0, 1), x211, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x211
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x212, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x212
    x213 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x213 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x21, (4, 0, 5, 6), (1, 3, 4, 5, 6, 2)) * -1.0
    del x21
    x214 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x214 += einsum(t1[np.ix_(so,sv)], (0, 1), x213, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x213
    t3new += einsum(x214, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x214, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x214, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x214, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x214
    x215 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x215 += einsum(t1[np.ix_(so,sv)], (0, 1), x153, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 6, 1))
    del x153
    x216 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x216 += einsum(t1[np.ix_(so,sv)], (0, 1), x215, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 6, 1)) * -1.0
    del x215
    t3new += einsum(x216, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x216, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x216
    x217 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x217 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x44, (4, 5, 0, 6), (1, 3, 5, 4, 2, 6)) * -1.0
    del x44
    t3new += einsum(x217, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x217, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x217
    x218 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x218 += einsum(t1[np.ix_(so,sv)], (0, 1), x161, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x161
    x219 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x219 += einsum(t1[np.ix_(so,sv)], (0, 1), x218, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x218
    t3new += einsum(x219, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x219, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x219, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x219, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x219
    x220 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x220 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oVvv, (2, 3, 4, 1), (3, 0, 2, 4)) * -1.0
    x221 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x221 += einsum(t1[np.ix_(so,sv)], (0, 1), x220, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x221, (4, 5, 6, 0), (5, 6, 1, 2, 3, 4)) * -1.0
    del x221
    x222 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x222 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x220, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2))
    del x220
    x223 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x223 += einsum(t1[np.ix_(so,sv)], (0, 1), x222, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x222
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x223
    x224 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x224 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oovO, (2, 0, 3, 4), (4, 1, 2, 3)) * -1.0
    x225 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x225 += einsum(t1[np.ix_(so,sv)], (0, 1), x224, (2, 3, 4, 1), (2, 3, 0, 4))
    x226 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x226 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x225, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3)) * -1.0
    del x225
    t3new += einsum(x226, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x226, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x226
    x227 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x227 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x224, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x224
    x228 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x228 += einsum(t1[np.ix_(so,sv)], (0, 1), x227, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -1.0
    del x227
    t3new += einsum(x228, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x228, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x228
    x229 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x229 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ooov, (2, 0, 3, 4), (1, 2, 3, 4))
    x230 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x230 += einsum(t1[np.ix_(so,sv)], (0, 1), x229, (2, 3, 4, 1), (2, 0, 3, 4))
    x231 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x231 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x230, (4, 5, 0, 6), (1, 4, 5, 6, 2, 3))
    del x230
    t3new += einsum(x231, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x231, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x231
    x232 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x232 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x229, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    del x229
    x233 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x233 += einsum(t1[np.ix_(so,sv)], (0, 1), x232, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6)) * -1.0
    del x232
    t3new += einsum(x233, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x233, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x233, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x233, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x233
    x234 = np.zeros((navir, nvir, nvir, nvir), dtype=types[float])
    x234 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovvv, (0, 2, 3, 4), (1, 2, 3, 4))
    x235 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x235 += einsum(t1[np.ix_(so,sv)], (0, 1), x234, (2, 3, 4, 1), (2, 0, 3, 4))
    del x234
    x236 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x236 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x235, (4, 5, 6, 3), (1, 4, 5, 0, 2, 6))
    del x235
    t3new += einsum(x236, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x236, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x236, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x236, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x236
    x237 = np.zeros((naocc, nocc, nocc, nocc), dtype=types[float])
    x237 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x238 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x238 += einsum(t1[np.ix_(so,sv)], (0, 1), x237, (2, 3, 0, 4), (2, 3, 4, 1))
    del x237
    x239 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x239 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x238, (4, 1, 5, 6), (4, 3, 0, 5, 6, 2))
    del x238
    t3new += einsum(x239, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x239, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x239, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x239, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x239
    x240 = np.zeros((naocc, nocc, nvir, nvir), dtype=types[float])
    x240 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x241 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x241 += einsum(t1[np.ix_(so,sv)], (0, 1), x240, (2, 3, 4, 1), (2, 0, 3, 4)) * -1.0
    x242 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x242 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x241, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    del x241
    t3new += einsum(x242, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x242, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x242, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x242, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x242
    x243 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x243 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x240, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6))
    del x240
    x244 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x244 += einsum(t1[np.ix_(so,sv)], (0, 1), x243, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x243
    t3new += einsum(x244, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x244, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x244
    x245 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x245 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.oVvv, (2, 3, 4, 1), (0, 3, 2, 4)) * -1.0
    x246 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x246 += einsum(t1[np.ix_(so,sv)], (0, 1), x245, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    x247 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x247 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x246, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x246
    t3new += einsum(x247, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x247, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x247
    x248 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x248 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x245, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2)) * -1.0
    del x245
    x249 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x249 += einsum(t1[np.ix_(so,sv)], (0, 1), x248, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x248
    t3new += einsum(x249, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x249, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x249
    x250 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x250 += einsum(t1[np.ix_(so,sV)], (0, 1), t1[np.ix_(sO,sv)], (2, 3), v.ooov, (4, 0, 5, 3), (2, 1, 4, 5))
    x251 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x251 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x250, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3)) * -1.0
    del x250
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x251
    x252 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x252 += einsum(t1[np.ix_(so,sV)], (0, 1), t1[np.ix_(sO,sv)], (2, 3), v.ovvv, (0, 4, 5, 3), (2, 1, 4, 5))
    x253 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x253 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x252, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6)) * -1.0
    del x252
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x253
    x254 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x254 += einsum(x3, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 6, 3, 7), (5, 7, 0, 4, 2, 6))
    x255 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x255 += einsum(t1[np.ix_(so,sv)], (0, 1), x254, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x254
    t3new += einsum(x255, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x255, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x255, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x255, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x255
    x256 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x256 += einsum(x6, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5)) * -1.0
    del x6
    t3new += einsum(x256, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x256, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x256
    x257 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x257 += einsum(t1[np.ix_(so,sv)], (0, 1), x186, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x186
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x257, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * 0.5
    del x257
    x258 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x258 += einsum(x5, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5)) * -1.0
    x259 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x259 += einsum(t1[np.ix_(so,sv)], (0, 1), x258, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x258
    t3new += einsum(x259, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x259, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x259
    x260 = np.zeros((navir, navir, nocc, nvir), dtype=types[float])
    x260 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oovV, (2, 0, 3, 4), (1, 4, 2, 3)) * -1.0
    x261 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x261 += einsum(t1[np.ix_(so,sv)], (0, 1), x260, (2, 3, 4, 1), (2, 3, 0, 4))
    x262 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x262 += einsum(x261, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7)) * -1.0
    del x261
    t3new += einsum(x262, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x262, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x262
    x263 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x263 += einsum(x260, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 7, 3, 1), (6, 0, 4, 5, 2, 7))
    del x260
    x264 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x264 += einsum(t1[np.ix_(so,sv)], (0, 1), x263, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -1.0
    del x263
    t3new += einsum(x264, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x264, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    del x264
    x265 = np.zeros((navir, nocc), dtype=types[float])
    x265 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oovV, (2, 0, 1, 3), (3, 2)) * -1.0
    x266 = np.zeros((navir, navir), dtype=types[float])
    x266 += einsum(t1[np.ix_(so,sV)], (0, 1), x265, (2, 0), (1, 2))
    del x265
    t3new += einsum(x266, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -1.0
    del x266
    x267 = np.zeros((naocc, naocc, nocc, nvir), dtype=types[float])
    x267 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.oOvv, (2, 3, 4, 1), (0, 3, 2, 4)) * -1.0
    x268 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x268 += einsum(t1[np.ix_(so,sv)], (0, 1), x267, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    x269 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x269 += einsum(x268, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 1, 5, 6, 7), (0, 7, 2, 4, 5, 6))
    del x268
    t3new += einsum(x269, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x269, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    del x269
    x270 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x270 += einsum(x267, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6)) * -1.0
    del x267
    x271 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x271 += einsum(t1[np.ix_(so,sv)], (0, 1), x270, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x270
    t3new += einsum(x271, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x271, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x271
    x272 = np.zeros((naocc, nvir), dtype=types[float])
    x272 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvv, (0, 2, 3, 1), (2, 3)) * -1.0
    x273 = np.zeros((naocc, naocc), dtype=types[float])
    x273 += einsum(t1[np.ix_(sO,sv)], (0, 1), x272, (2, 1), (0, 2))
    del x272
    t3new += einsum(x273, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -1.0
    del x273
    x274 = np.zeros((naocc, naocc, navir, nocc), dtype=types[float])
    x274 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.oOvV, (2, 3, 1, 4), (0, 3, 4, 2))
    x275 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x275 += einsum(t1[np.ix_(so,sV)], (0, 1), x274, (2, 3, 4, 0), (2, 3, 1, 4))
    del x274
    t3new += einsum(x275, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -1.0
    del x275
    x276 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x276 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x47, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x47
    t3new += einsum(x276, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x276, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x276, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x276, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x276
    x277 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x277 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oovv, (4, 0, 5, 2), (1, 3, 4, 5))
    x278 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x278 += einsum(t1[np.ix_(so,sv)], (0, 1), x277, (2, 3, 4, 1), (2, 3, 0, 4))
    x279 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x279 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x278, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3)) * -1.0
    del x278
    t3new += einsum(x279, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x279, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x279
    x280 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x280 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oovv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x281 = np.zeros((naocc, navir, nocc, nocc, nocc, nocc), dtype=types[float])
    x281 += einsum(t1[np.ix_(so,sv)], (0, 1), x280, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 6, 5)) * -1.0
    x282 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x282 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x281, (4, 5, 6, 7, 0, 1), (4, 5, 6, 7, 2, 3))
    t3new += einsum(x282, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    t3new += einsum(x282, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    del x282
    x283 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x283 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x3, (4, 0, 5, 3), (1, 4, 5, 2)) * -1.0
    x284 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x284 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x283, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    del x283
    t3new += einsum(x284, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x284, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x284, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x284, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x284
    x285 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x285 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (3, 0, 4, 5)) * -1.0
    x286 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x286 += einsum(t1[np.ix_(so,sv)], (0, 1), x285, (2, 3, 4, 1), (2, 0, 3, 4))
    x287 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x287 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x286, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x286
    t3new += einsum(x287, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x287, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x287
    x288 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x288 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x3, (4, 0, 1, 5), (3, 4, 2, 5)) * -1.0
    del x3
    x289 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x289 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x288, (4, 5, 6, 3), (1, 4, 5, 0, 2, 6))
    del x288
    t3new += einsum(x289, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x289, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x289, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    t3new += einsum(x289, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 0.5
    del x289
    x290 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x290 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x280, (4, 5, 6, 7, 1, 3), (4, 5, 6, 0, 7, 2)) * -1.0
    del x280
    x291 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x291 += einsum(t1[np.ix_(so,sv)], (0, 1), x290, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x290
    t3new += einsum(x291, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x291, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x291, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x291, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x291
    x292 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x292 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x277, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2)) * -1.0
    del x277
    x293 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x293 += einsum(t1[np.ix_(so,sv)], (0, 1), x292, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x292
    t3new += einsum(x293, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x293, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x293
    x294 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x294 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x41, (4, 5, 0, 6), (1, 3, 4, 5, 6, 2)) * -1.0
    del x41
    x295 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x295 += einsum(t1[np.ix_(so,sv)], (0, 1), x294, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x294
    t3new += einsum(x295, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x295, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    del x295
    x296 = np.zeros((naocc, nocc, nocc, nocc), dtype=types[float])
    x296 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (1, 0, 4, 5)) * -1.0
    x297 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x297 += einsum(t1[np.ix_(so,sv)], (0, 1), x296, (2, 3, 4, 0), (2, 3, 4, 1))
    del x296
    x298 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x298 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x297, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    del x297
    t3new += einsum(x298, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x298, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x298, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x298, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    del x298
    x299 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x299 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x285, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    del x285
    x300 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x300 += einsum(t1[np.ix_(so,sv)], (0, 1), x299, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x299
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x300
    x301 = np.zeros((naocc, nocc, nvir, nvir), dtype=types[float])
    x301 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oovv, (4, 0, 5, 3), (1, 4, 2, 5)) * -1.0
    x302 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x302 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x301, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6))
    del x301
    x303 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x303 += einsum(t1[np.ix_(so,sv)], (0, 1), x302, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x302
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x303
    x304 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x304 += einsum(x5, (0, 1), t2[np.ix_(so,sO,sv,sV)], (2, 3, 1, 4), (3, 4, 2, 0)) * -1.0
    x305 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x305 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x304, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3)) * -1.0
    del x304
    t3new += einsum(x305, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x305, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x305
    x306 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x306 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x51, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6)) * -1.0
    del x51
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x306
    x307 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x307 += einsum(x5, (0, 1), t2[np.ix_(so,sO,sv,sv)], (2, 3, 4, 1), (3, 2, 0, 4))
    x308 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x308 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x307, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    del x307
    t3new += einsum(x308, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x308, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x308, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x308, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x308
    x309 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x309 += einsum(x5, (0, 1), t2[np.ix_(so,so,sv,sV)], (2, 3, 1, 4), (4, 2, 3, 0))
    del x5
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x309, (4, 5, 6, 0), (5, 6, 1, 2, 3, 4))
    del x309
    x310 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x310 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oovv, (2, 0, 3, 4), (1, 2, 3, 4))
    x311 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x311 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x310, (4, 1, 3, 5), (4, 0, 2, 5))
    x312 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x312 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x311, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x311
    t3new += einsum(x312, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x312, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x312
    x313 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x313 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x310, (4, 5, 2, 3), (1, 4, 0, 5)) * -1.0
    x314 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x314 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x313, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3)) * -1.0
    del x313
    t3new += einsum(x314, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    t3new += einsum(x314, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    del x314
    x315 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x315 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x310, (4, 0, 3, 5), (1, 4, 2, 5)) * -1.0
    x316 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x316 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x315, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2)) * -1.0
    del x315
    t3new += einsum(x316, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x316, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x316
    x317 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x317 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x310, (4, 5, 2, 3), (4, 0, 1, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x317, (4, 5, 6, 0), (5, 6, 1, 2, 3, 4)) * -0.5
    del x317
    x318 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x318 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x319 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x319 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x318, (4, 1, 5, 3), (4, 0, 5, 2))
    x320 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x320 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x319, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x319
    t3new += einsum(x320, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x320, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x320, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x320, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x320
    x321 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x321 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x318, (4, 1, 5, 2), (4, 3, 0, 5)) * -1.0
    x322 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x322 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x321, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3)) * -1.0
    del x321
    t3new += einsum(x322, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x322, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x322
    x323 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x323 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x318, (4, 0, 1, 5), (4, 3, 2, 5)) * -1.0
    x324 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x324 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x323, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2)) * -1.0
    del x323
    t3new += einsum(x324, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x324, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    del x324
    x325 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x325 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x318, (4, 0, 1, 5), (4, 2, 3, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x325, (4, 5, 6, 2), (0, 1, 4, 5, 6, 3)) * -0.5
    del x325
    x326 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x326 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x46, (4, 5, 6, 0), (1, 3, 4, 5, 6, 2))
    del x46
    x327 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x327 += einsum(t1[np.ix_(so,sv)], (0, 1), x326, (2, 3, 4, 5, 0, 6), (2, 3, 5, 4, 1, 6)) * -1.0
    del x326
    t3new += einsum(x327, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x327, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x327
    x328 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x328 += einsum(t1[np.ix_(so,sv)], (0, 1), x281, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x281
    x329 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x329 += einsum(t1[np.ix_(so,sv)], (0, 1), x328, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 6, 1)) * -1.0
    del x328
    t3new += einsum(x329, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x329, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x329
    x330 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x330 += einsum(t1[np.ix_(so,sv)], (0, 1), x310, (2, 3, 1, 4), (2, 0, 3, 4)) * -1.0
    del x310
    x331 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x331 += einsum(t1[np.ix_(so,sv)], (0, 1), x330, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x331, (4, 5, 6, 0), (5, 6, 1, 2, 3, 4)) * -1.0
    del x331
    x332 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x332 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x330, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2))
    del x330
    x333 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x333 += einsum(t1[np.ix_(so,sv)], (0, 1), x332, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -1.0
    del x332
    t3new += einsum(x333, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x333, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x333, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x333, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x333
    x334 = np.zeros((naocc, nocc, nocc, nocc), dtype=types[float])
    x334 += einsum(t1[np.ix_(so,sv)], (0, 1), x318, (2, 3, 4, 1), (2, 0, 4, 3))
    x335 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x335 += einsum(t1[np.ix_(so,sv)], (0, 1), x334, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x334
    x336 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x336 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x335, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    del x335
    t3new += einsum(x336, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x336, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x336, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x336, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x336
    x337 = np.zeros((naocc, navir, nocc, nocc, nocc, nocc), dtype=types[float])
    x337 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x318, (4, 5, 6, 2), (4, 3, 0, 1, 6, 5)) * -1.0
    del x318
    x338 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x338 += einsum(t1[np.ix_(so,sv)], (0, 1), x337, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x337
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x338, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x338
    x339 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x339 += einsum(t1[np.ix_(so,sV)], (0, 1), t1[np.ix_(sO,sv)], (2, 3), v.oovv, (4, 0, 5, 3), (2, 1, 4, 5))
    x340 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x340 += einsum(t1[np.ix_(so,sv)], (0, 1), x339, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    x341 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x341 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x340, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3)) * -1.0
    del x340
    t3new += einsum(x341, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x341, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x341
    x342 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x342 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x339, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2)) * -1.0
    del x339
    x343 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x343 += einsum(t1[np.ix_(so,sv)], (0, 1), x342, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -1.0
    del x342
    t3new += einsum(x343, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x343, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x343

    return {"t1new": t1new, "t2new": t2new, "t3new": t3new}

