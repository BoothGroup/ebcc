# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), x0, (0, 1, 2, 3), ()) * 2.0
    x1 = np.zeros((nocc, nvir), dtype=types[float])
    x1 += einsum(f.ov, (0, 1), (0, 1))
    x1 += einsum(t1, (0, 1), x0, (0, 2, 1, 3), (2, 3))
    del x0
    e_cc += einsum(t1, (0, 1), x1, (0, 1), ()) * 2.0
    del x1

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
    t1new[np.ix_(so,sv)] += einsum(f.ov, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 1), (2, 3)) * 2.0
    t1new[np.ix_(so,sv)] += einsum(f.ov, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 1, 3), (2, 3)) * -1.0
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), v.oovv, (2, 0, 3, 1), (2, 3)) * -1.0
    t1new[np.ix_(so,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvv, (1, 2, 4, 3), (0, 4)) * -1.0
    t1new[np.ix_(so,sv)] += einsum(v.ovOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 2, 5, 1, 3), (4, 5)) * 0.5
    t1new[np.ix_(so,sv)] += einsum(v.oVvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 3, 5, 2, 1), (4, 5)) * -0.5
    t1new[np.ix_(so,sv)] += einsum(v.ovOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 2, 1, 5, 3), (4, 5)) * -0.25
    t1new[np.ix_(so,sv)] += einsum(v.oVvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 3, 2, 5, 1), (4, 5)) * 0.25
    t1new[np.ix_(so,sv)] += einsum(v.ovOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 2, 5, 1, 3), (4, 5))
    t1new[np.ix_(so,sv)] += einsum(v.ovOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 4, 2, 1, 5, 3), (4, 5)) * 0.25
    t1new[np.ix_(so,sv)] += einsum(v.oVvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 4, 3, 2, 5, 1), (4, 5)) * -0.25
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new[np.ix_(so,so,sv,sv)] += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oooo, (4, 1, 5, 0), (4, 5, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t3new = np.zeros((nocc, nocc, naocc, nvir, nvir, navir), dtype=types[float])
    t3new += einsum(f.oo, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -1.0
    t3new += einsum(f.oo, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -1.0
    t3new += einsum(f.OO, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -1.0
    t3new += einsum(f.vv, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 1, 5, 6), (2, 3, 4, 0, 5, 6))
    t3new += einsum(f.vv, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6))
    t3new += einsum(f.VV, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oovO, (4, 1, 5, 6), (4, 0, 6, 5, 2, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oovO, (4, 0, 5, 6), (1, 4, 6, 5, 2, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoO, (4, 5, 0, 6), (1, 4, 6, 2, 5, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oOoV, (1, 4, 5, 6), (5, 0, 4, 3, 2, 6))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oooV, (4, 0, 5, 6), (5, 4, 1, 3, 2, 6))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvV, (4, 5, 2, 6), (0, 4, 1, 3, 5, 6)) * -1.0
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oVvv, (4, 5, 6, 3), (4, 0, 1, 6, 2, 5)) * -1.0
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oVvv, (4, 5, 6, 2), (4, 0, 1, 3, 6, 5)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.vvvO, (4, 2, 5, 6), (1, 0, 6, 5, 4, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.vOvV, (4, 5, 2, 6), (0, 1, 5, 4, 3, 6)) * -1.0
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ooov, (4, 0, 5, 6), (4, 5, 1, 3, 6, 2))
    t3new += einsum(v.ooOO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 4, 3, 5, 6, 7), (0, 4, 2, 5, 6, 7)) * 0.5
    t3new += einsum(v.oOoO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 1, 5, 6, 7), (0, 4, 3, 5, 6, 7)) * -0.5
    t3new += einsum(v.ooOO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 3, 5, 6, 7), (4, 0, 2, 5, 6, 7))
    t3new += einsum(v.oooo, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 3, 4, 5, 6, 7), (0, 2, 4, 5, 6, 7))
    t3new += einsum(v.oovv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 4, 5, 3, 6, 7), (0, 4, 5, 2, 6, 7)) * -1.0
    t3new += einsum(v.oovv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 4, 5, 6, 3, 7), (0, 4, 5, 6, 2, 7)) * -1.0
    t3new += einsum(v.oovv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 3, 6, 7), (4, 0, 5, 2, 6, 7)) * -1.0
    t3new += einsum(v.oovv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -1.0
    t3new += einsum(v.ooVV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 4, 5, 6, 7, 3), (0, 4, 5, 6, 7, 2)) * -1.0
    t3new += einsum(v.ooVV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 6, 7, 3), (4, 0, 5, 6, 7, 2)) * -1.0
    t3new += einsum(v.oVoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 5, 6, 7, 3), (0, 4, 5, 6, 7, 1))
    t3new += einsum(v.vOvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 3, 2, 6, 7), (4, 5, 1, 0, 6, 7))
    t3new += einsum(v.vvOO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 3, 1, 6, 7), (4, 5, 2, 0, 6, 7)) * -1.0
    t3new += einsum(v.vvOO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 3, 6, 1, 7), (4, 5, 2, 6, 0, 7)) * -1.0
    t3new += einsum(v.OOVV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -1.0
    t3new += einsum(v.vvvv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 1, 3, 7), (4, 5, 6, 0, 2, 7))
    t3new += einsum(v.vvVV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 7, 2)) * 0.5
    t3new += einsum(v.vVvV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 2, 7, 1), (4, 5, 6, 0, 7, 3)) * -0.5
    t3new += einsum(v.vvVV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 7, 1, 3), (4, 5, 6, 7, 0, 2))
    t3new += einsum(v.ovoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 2, 5, 6, 7, 3), (4, 0, 5, 6, 1, 7)) * -1.0
    t3new += einsum(v.vOOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 5, 2, 6, 7, 3), (5, 4, 1, 0, 6, 7)) * -1.0
    t3new += einsum(v.oVOV, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (4, 5, 2, 6, 7, 3), (0, 4, 5, 7, 6, 1)) * -1.0
    t3new += einsum(v.ovOV, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sV,sVf)], (4, 5, 2, 6, 7, 3), (0, 4, 5, 1, 6, 7))
    x0 = np.zeros((nocc, nvir), dtype=types[float])
    x0 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovov, (2, 3, 0, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(x0, (0, 1), (0, 1)) * 2.0
    t1new[np.ix_(so,sv)] += einsum(x0, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 1), (2, 3)) * 4.0
    t1new[np.ix_(so,sv)] += einsum(x0, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 1, 3), (2, 3)) * -2.0
    x1 = np.zeros((nocc, nvir), dtype=types[float])
    x1 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 1, 0, 2), (4, 3))
    t1new[np.ix_(so,sv)] += einsum(x1, (0, 1), (0, 1)) * -1.5
    t1new[np.ix_(so,sv)] += einsum(x1, (0, 1), (0, 1)) * -0.5
    del x1
    x2 = np.zeros((nocc, nvir), dtype=types[float])
    x2 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 0, 1, 2), (4, 3))
    t1new[np.ix_(so,sv)] += einsum(x2, (0, 1), (0, 1)) * 0.5
    t1new[np.ix_(so,sv)] += einsum(x2, (0, 1), (0, 1)) * 0.5
    del x2
    x3 = np.zeros((nocc, nvir), dtype=types[float])
    x3 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvv, (1, 3, 4, 2), (0, 4))
    t1new[np.ix_(so,sv)] += einsum(x3, (0, 1), (0, 1))
    t1new[np.ix_(so,sv)] += einsum(x3, (0, 1), (0, 1))
    del x3
    x4 = np.zeros((nocc, nocc), dtype=types[float])
    x4 += einsum(f.ov, (0, 1), t1[np.ix_(so,sv)], (2, 1), (0, 2))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x4, (0, 2), (2, 1)) * -1.0
    t3new += einsum(x4, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6)) * -1.0
    t3new += einsum(x4, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -1.0
    x5 = np.zeros((nocc, nocc), dtype=types[float])
    x5 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ooov, (2, 0, 3, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x5, (2, 0), (2, 1))
    t3new += einsum(x5, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    t3new += einsum(x5, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6))
    x6 = np.zeros((nocc, nocc), dtype=types[float])
    x6 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ooov, (2, 3, 0, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x6, (2, 0), (2, 1)) * -2.0
    t3new += einsum(x6, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    t3new += einsum(x6, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x7 = np.zeros((nvir, nvir), dtype=types[float])
    x7 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvv, (0, 2, 3, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x7, (1, 2), (0, 2)) * -1.0
    t3new += einsum(x7, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6)) * -1.0
    t3new += einsum(x7, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -1.0
    x8 = np.zeros((nvir, nvir), dtype=types[float])
    x8 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvv, (0, 1, 2, 3), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x8, (2, 1), (0, 2)) * 2.0
    t3new += einsum(x8, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 1, 5, 6), (2, 3, 4, 0, 5, 6)) * 2.0
    t3new += einsum(x8, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x9, (4, 5, 0, 6), (4, 5, 1, 3, 6, 2))
    x10 = np.zeros((nocc, nvir), dtype=types[float])
    x10 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x9, (4, 0, 1, 2), (4, 3))
    t1new[np.ix_(so,sv)] += einsum(x10, (0, 1), (0, 1)) * -1.5
    t1new[np.ix_(so,sv)] += einsum(x10, (0, 1), (0, 1)) * -0.5
    del x10
    x11 = np.zeros((nocc, nvir), dtype=types[float])
    x11 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x9, (4, 0, 1, 3), (4, 2))
    t1new[np.ix_(so,sv)] += einsum(x11, (0, 1), (0, 1)) * 0.5
    t1new[np.ix_(so,sv)] += einsum(x11, (0, 1), (0, 1)) * 0.5
    del x11
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 2, 1, 3), (0, 4))
    t3new += einsum(x12, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x13 = np.zeros((nocc, nvir), dtype=types[float])
    x13 += einsum(t1[np.ix_(so,sv)], (0, 1), x12, (2, 0), (2, 1))
    t1new[np.ix_(so,sv)] += einsum(x13, (0, 1), (0, 1)) * -1.0
    t1new[np.ix_(so,sv)] += einsum(x13, (0, 1), (0, 1)) * -1.0
    del x13
    x14 = np.zeros((nocc, nocc), dtype=types[float])
    x14 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 3, 1, 2), (0, 4))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x14, (2, 0), (2, 1))
    t3new += einsum(x14, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    t3new += einsum(x14, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6))
    x15 = np.zeros((nocc, nvir), dtype=types[float])
    x15 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovov, (2, 1, 0, 3), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(x15, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 1), (2, 3)) * -2.0
    t1new[np.ix_(so,sv)] += einsum(x15, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 1, 3), (2, 3))
    x16 = np.zeros((nocc, nocc), dtype=types[float])
    x16 += einsum(t1[np.ix_(so,sv)], (0, 1), x15, (2, 1), (0, 2))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x16, (2, 0), (2, 1))
    t3new += einsum(x16, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    t3new += einsum(x16, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6))
    x17 = np.zeros((nocc, nocc), dtype=types[float])
    x17 += einsum(t1[np.ix_(so,sv)], (0, 1), x0, (2, 1), (0, 2))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x17, (2, 0), (2, 1)) * -2.0
    t3new += einsum(x17, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    t3new += einsum(x17, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(f.oo, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x18, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x18, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x18
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(f.vv, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x19, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x19, (0, 1, 2, 3), (0, 1, 3, 2))
    del x19
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x20, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x20
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x21, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x21, (0, 1, 2, 3), (1, 0, 2, 3))
    t3new += einsum(x21, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 4, 5, 2, 6, 7), (0, 4, 5, 3, 6, 7))
    t3new += einsum(x21, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 2, 6, 7), (4, 0, 5, 6, 3, 7)) * -1.0
    t3new += einsum(x21, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 6, 2, 7), (4, 0, 5, 6, 3, 7)) * 2.0
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(f.OV, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x22, (0, 1, 2, 3), (1, 0, 3, 2))
    del x22
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x23, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x23
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (0, 4, 3, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x24, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x24, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x24
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x25, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x25, (4, 1, 5, 3), (4, 0, 5, 2)) * 4.0
    del x25
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x26, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x26, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x26, (4, 1, 5, 2), (4, 0, 5, 3))
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(v.ooOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 2, 5, 6, 3), (4, 0, 5, 6))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x27, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x27, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x27
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum(v.ooOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 4, 2, 5, 6, 3), (4, 0, 5, 6))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x28, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x28, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    del x28
    x29 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x29 += einsum(v.oovO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 4, 3, 2, 5, 6), (6, 4, 0, 5))
    t2new[np.ix_(so,so,sV,sv)] += einsum(x29, (0, 1, 2, 3), (2, 1, 0, 3)) * 0.5
    t2new[np.ix_(so,so,sv,sV)] += einsum(x29, (0, 1, 2, 3), (1, 2, 3, 0)) * 0.5
    del x29
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum(v.vOvV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 0, 6, 3), (4, 5, 6, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x30, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x30, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    del x30
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum(v.vvOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 2, 1, 6, 3), (4, 5, 6, 0))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x31, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x31, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    del x31
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum(v.vvOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 2, 6, 1, 3), (4, 5, 6, 0))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x32, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x32, (0, 1, 2, 3), (1, 0, 3, 2))
    del x32
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum(x4, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 4), (1, 2, 3, 4))
    del x4
    t2new[np.ix_(so,so,sv,sv)] += einsum(x33, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x33, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x33
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum(f.ov, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (0, 2, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x34, (0, 4, 5, 6), (5, 4, 1, 3, 6, 2))
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum(t1[np.ix_(so,sv)], (0, 1), x34, (0, 2, 3, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x35, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x35, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x35
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x36 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x36, (4, 5, 0, 6), (5, 4, 1, 3, 6, 2))
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum(t1[np.ix_(so,sv)], (0, 1), x36, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x37, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x37, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x37
    x38 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x38 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x38, (2, 0, 3, 4), (2, 3, 1, 4))
    del x38
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum(t1[np.ix_(so,sv)], (0, 1), x9, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x39, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x39
    x40 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x40 += einsum(t1[np.ix_(so,sv)], (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x40, (2, 3, 1, 4), (0, 2, 3, 4))
    del x40
    x41 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x41 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    t3new += einsum(x41, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 2, 4, 5, 6, 7), (0, 1, 4, 5, 6, 7))
    t3new += einsum(x41, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 7), (1, 0, 4, 5, 6, 7))
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x41, (4, 5, 0, 1), (4, 5, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x42, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x42, (0, 1, 2, 3), (1, 0, 2, 3))
    del x42
    x43 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x43 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 1, 5, 2), (0, 4, 5, 3))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x43, (4, 5, 0, 6), (4, 5, 1, 3, 6, 2)) * -1.0
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum(t1[np.ix_(so,sv)], (0, 1), x43, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x44, (0, 1, 2, 3), (1, 0, 3, 2))
    del x44
    x45 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x45 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x45, (4, 5, 0, 6), (5, 4, 1, 3, 6, 2)) * 2.0
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum(t1[np.ix_(so,sv)], (0, 1), x45, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x46, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x46, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x46
    x47 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x47 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 5, 1, 2), (0, 4, 5, 3))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x47, (4, 5, 0, 6), (5, 4, 1, 3, 6, 2)) * -1.0
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum(t1[np.ix_(so,sv)], (0, 1), x47, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x48, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x48, (0, 1, 2, 3), (0, 1, 3, 2))
    del x48
    x49 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x49 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x49, (4, 5, 0, 6), (5, 4, 1, 3, 6, 2)) * -1.0
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum(t1[np.ix_(so,sv)], (0, 1), x49, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x50, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x50, (0, 1, 2, 3), (0, 1, 3, 2))
    del x50
    x51 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x51 += einsum(x5, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x51, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x51, (0, 1, 2, 3), (1, 0, 3, 2))
    del x51
    x52 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x52 += einsum(x6, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x6
    t2new[np.ix_(so,so,sv,sv)] += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x52, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x52
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x53 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    t3new += einsum(x53, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 4, 5, 3, 6, 7), (0, 4, 5, 2, 6, 7)) * -1.0
    t3new += einsum(x53, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 4, 5, 6, 3, 7), (0, 4, 5, 6, 2, 7)) * -1.0
    t3new += einsum(x53, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 3, 6, 7), (4, 0, 5, 2, 6, 7)) * -1.0
    t3new += einsum(x53, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -1.0
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x53, (4, 1, 5, 3), (4, 0, 2, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x54, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x54, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x54
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x53, (4, 1, 5, 2), (4, 0, 3, 5))
    del x53
    t2new[np.ix_(so,so,sv,sv)] += einsum(x55, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x55, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x55
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x21, (4, 1, 3, 5), (4, 0, 2, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x56, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x56, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x56
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x21, (4, 1, 2, 5), (4, 0, 3, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x57, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x57, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x57
    x58 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x58 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x58, (4, 5, 0, 6), (4, 5, 1, 3, 6, 2))
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum(t1[np.ix_(so,sv)], (0, 1), x58, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x59, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x59
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum(x8, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 4, 0))
    del x8
    t2new[np.ix_(so,so,sv,sv)] += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x60, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x60
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum(x7, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 0), (2, 3, 4, 1))
    del x7
    t2new[np.ix_(so,so,sv,sv)] += einsum(x61, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x61, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x61
    x62 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x62 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovOV, (2, 1, 3, 4), (3, 4, 0, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x62, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 0, 5, 6, 1), (2, 4, 6, 5)) * -1.0
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum(x62, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 0, 5, 6, 1), (2, 4, 5, 6))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x63, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x63, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    del x63
    x64 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x64 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oVvO, (2, 3, 1, 4), (4, 3, 0, 2))
    x65 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x65 += einsum(x64, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 0, 5, 6, 1), (2, 4, 5, 6))
    del x64
    t2new[np.ix_(so,so,sv,sv)] += einsum(x65, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x65, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    del x65
    x66 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x66 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovOV, (2, 1, 3, 4), (3, 4, 0, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x66, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 0, 5, 6, 1), (4, 2, 5, 6)) * -1.0
    del x66
    x67 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x67 += einsum(v.ovOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 2, 1, 6, 3), (4, 5, 0, 6))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x67, (4, 5, 0, 6), (4, 5, 1, 3, 6, 2)) * 0.5
    x68 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x68 += einsum(t1[np.ix_(so,sv)], (0, 1), x67, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x68, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    del x68
    x69 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x69 += einsum(v.oVvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 3, 2, 6, 1), (4, 5, 0, 6))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x69, (4, 5, 0, 6), (4, 5, 1, 3, 6, 2)) * -0.5
    x70 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x70 += einsum(t1[np.ix_(so,sv)], (0, 1), x69, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x70, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x70, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    del x70
    x71 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x71 += einsum(v.ovOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 2, 6, 1, 3), (4, 5, 0, 6))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x71, (4, 5, 0, 6), (5, 4, 1, 3, 6, 2))
    x72 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x72 += einsum(t1[np.ix_(so,sv)], (0, 1), x71, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x72, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x72, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x72
    x73 = np.zeros((naocc, navir), dtype=types[float])
    x73 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovOV, (0, 1, 2, 3), (2, 3))
    x74 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x74 += einsum(x73, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    del x73
    t2new[np.ix_(so,so,sv,sv)] += einsum(x74, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x74, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x74
    x75 = np.zeros((naocc, navir), dtype=types[float])
    x75 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oVvO, (0, 2, 1, 3), (3, 2))
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum(x75, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    del x75
    t2new[np.ix_(so,so,sv,sv)] += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x76, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x76
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum(x14, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x14
    t2new[np.ix_(so,so,sv,sv)] += einsum(x77, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x77, (0, 1, 2, 3), (1, 0, 3, 2))
    del x77
    x78 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x78 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x26, (4, 1, 5, 3), (0, 4, 2, 5))
    del x26
    t2new[np.ix_(so,so,sv,sv)] += einsum(x78, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x78, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x78
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum(x12, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x79, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x79, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x79, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x79
    x80 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x80 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x80, (4, 1, 5, 3), (4, 0, 5, 2)) * -2.0
    del x80
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x81, (4, 1, 5, 2), (4, 0, 3, 5))
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x81, (4, 1, 5, 3), (0, 4, 2, 5))
    del x81
    t2new[np.ix_(so,so,sv,sv)] += einsum(x82, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x82, (0, 1, 2, 3), (1, 0, 3, 2))
    del x82
    x83 = np.zeros((nvir, nvir), dtype=types[float])
    x83 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (0, 2, 1, 4), (3, 4))
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum(x83, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x84, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x84, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x84, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x84, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    del x84
    x85 = np.zeros((nvir, nvir), dtype=types[float])
    x85 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (0, 4, 1, 2), (3, 4))
    x86 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x86 += einsum(x85, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x86, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x86, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    del x86
    x87 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x87 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x87, (4, 5, 0, 1), (5, 4, 3, 2))
    t3new += einsum(x87, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 7), (0, 1, 4, 5, 6, 7))
    x88 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x88 += einsum(t1[np.ix_(so,sv)], (0, 1), x41, (2, 3, 4, 0), (2, 4, 3, 1))
    x89 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x89 += einsum(t1[np.ix_(so,sv)], (0, 1), x88, (2, 0, 3, 4), (2, 3, 1, 4))
    del x88
    t2new[np.ix_(so,so,sv,sv)] += einsum(x89, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x89, (0, 1, 2, 3), (1, 0, 2, 3))
    del x89
    x90 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x90 += einsum(t1[np.ix_(so,sv)], (0, 1), x21, (2, 3, 1, 4), (0, 2, 3, 4))
    del x21
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x90, (4, 5, 0, 6), (4, 5, 1, 3, 6, 2))
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x91 += einsum(t1[np.ix_(so,sv)], (0, 1), x90, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x91, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x91, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x91
    x92 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x92 += einsum(t1[np.ix_(so,sv)], (0, 1), x9, (2, 3, 4, 1), (2, 0, 4, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x92, (4, 5, 0, 1), (5, 4, 3, 2))
    t3new += einsum(x92, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 2, 4, 5, 6, 7), (1, 0, 4, 5, 6, 7))
    x93 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x93 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x9, (4, 1, 5, 3), (4, 0, 5, 2))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x93, (4, 5, 0, 6), (4, 5, 1, 3, 6, 2)) * 2.0
    x94 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x94 += einsum(t1[np.ix_(so,sv)], (0, 1), x93, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x94, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x94
    x95 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x95 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x9, (4, 1, 5, 2), (4, 0, 5, 3))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x95, (4, 5, 0, 6), (4, 5, 1, 3, 6, 2)) * -1.0
    x96 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x96 += einsum(t1[np.ix_(so,sv)], (0, 1), x95, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x96, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x96, (0, 1, 2, 3), (1, 0, 3, 2))
    del x96
    x97 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x97 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x9, (4, 5, 1, 3), (4, 0, 5, 2))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x97, (4, 5, 0, 6), (4, 5, 1, 3, 6, 2)) * -1.0
    x98 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x98 += einsum(t1[np.ix_(so,sv)], (0, 1), x97, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x98, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x98, (0, 1, 2, 3), (1, 0, 3, 2))
    del x98
    x99 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x99 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x9, (4, 5, 1, 2), (4, 0, 5, 3))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x99, (4, 5, 0, 6), (5, 4, 1, 3, 6, 2)) * -1.0
    x100 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x100 += einsum(t1[np.ix_(so,sv)], (0, 1), x99, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x100, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x100, (0, 1, 2, 3), (1, 0, 2, 3))
    del x100
    x101 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x101 += einsum(x16, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (0, 2, 3, 4))
    del x16
    t2new[np.ix_(so,so,sv,sv)] += einsum(x101, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x101, (0, 1, 2, 3), (1, 0, 2, 3))
    del x101
    x102 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x102 += einsum(x17, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (0, 2, 3, 4))
    del x17
    t2new[np.ix_(so,so,sv,sv)] += einsum(x102, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x102, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x102
    x103 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x103 += einsum(t1[np.ix_(so,sv)], (0, 1), x87, (2, 3, 4, 0), (2, 3, 4, 1))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x103, (2, 3, 0, 4), (2, 3, 1, 4))
    del x103
    x104 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x104 += einsum(x0, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 0, 4))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x104, (4, 5, 0, 6), (5, 4, 1, 3, 6, 2)) * 2.0
    x105 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x105 += einsum(t1[np.ix_(so,sv)], (0, 1), x104, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x105, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x105, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x105
    x106 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x106 += einsum(x15, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 0, 4))
    t3new += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x106, (4, 5, 0, 6), (5, 4, 1, 3, 6, 2)) * -1.0
    x107 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x107 += einsum(t1[np.ix_(so,sv)], (0, 1), x106, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x107, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x107, (0, 1, 2, 3), (0, 1, 3, 2))
    del x107
    x108 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x108 += einsum(t1[np.ix_(so,sv)], (0, 1), x92, (2, 3, 0, 4), (3, 2, 4, 1))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x108, (2, 3, 0, 4), (2, 3, 1, 4))
    del x108
    x109 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x109 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooOV, (4, 1, 5, 6), (5, 6, 0, 4, 2, 3))
    t3new += einsum(x109, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x109, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x109
    x110 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x110 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoO, (4, 5, 1, 6), (6, 3, 0, 4, 2, 5))
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x110
    x111 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x111 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 0, 5, 6), (1, 3, 4, 5, 2, 6))
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x111
    x112 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x112 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvV, (4, 5, 3, 6), (1, 6, 0, 4, 2, 5))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x112
    x113 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x113 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovvv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    t3new += einsum(x113, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x113, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    x114 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x114 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.vvOV, (4, 3, 5, 6), (5, 6, 0, 1, 2, 4))
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x114
    x115 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x115 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovvv, (4, 5, 6, 3), (1, 2, 0, 4, 5, 6))
    t3new += einsum(x115, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x115, (4, 5, 6, 1, 3, 7), (6, 0, 4, 7, 2, 5)) * -2.0
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x115, (4, 5, 6, 1, 2, 7), (6, 0, 4, 7, 3, 5))
    x116 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x116 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 5, 3, 6, 7), (5, 7, 4, 0, 6, 1))
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x116, (4, 5, 6, 1, 7, 3), (0, 6, 4, 2, 7, 5)) * 2.0
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x116, (4, 5, 6, 1, 7, 2), (0, 6, 4, 3, 7, 5)) * -1.0
    del x116
    x117 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x117 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 6, 3, 7), (5, 7, 4, 0, 6, 1))
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x117, (4, 5, 6, 1, 7, 3), (6, 0, 4, 7, 2, 5)) * 4.0
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x117, (4, 5, 6, 1, 7, 2), (6, 0, 4, 7, 3, 5)) * -2.0
    del x117
    x118 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x118 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 3, 6, 7), (5, 7, 4, 0, 6, 1))
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x118, (4, 5, 6, 1, 7, 3), (6, 0, 4, 7, 2, 5)) * -2.0
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x118, (4, 5, 6, 1, 7, 2), (6, 0, 4, 7, 3, 5))
    del x118
    x119 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x119 += einsum(v.OVOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 2, 6, 7, 3), (0, 1, 4, 5, 6, 7))
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x119
    x120 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x120 += einsum(f.ov, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 1, 5, 6), (4, 6, 0, 2, 3, 5))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x120, (2, 3, 0, 4, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x120
    x121 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x121 += einsum(f.ov, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 1, 6), (4, 6, 0, 2, 3, 5))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x121, (2, 3, 0, 4, 5, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x121
    x122 = np.zeros((navir, navir), dtype=types[float])
    x122 += einsum(f.oV, (0, 1), t1[np.ix_(so,sV)], (0, 2), (1, 2))
    t3new += einsum(x122, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * -1.0
    del x122
    x123 = np.zeros((naocc, naocc), dtype=types[float])
    x123 += einsum(f.vO, (0, 1), t1[np.ix_(sO,sv)], (2, 0), (1, 2))
    t3new += einsum(x123, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * -1.0
    del x123
    x124 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x124 += einsum(f.ov, (0, 1), t2[np.ix_(so,sO,sv,sV)], (2, 3, 1, 4), (3, 4, 0, 2))
    x125 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x125 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x124, (4, 5, 1, 6), (4, 5, 6, 0, 2, 3))
    del x124
    t3new += einsum(x125, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x125, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x125
    x126 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x126 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x34, (0, 4, 5, 6), (1, 3, 4, 5, 2, 6))
    del x34
    t3new += einsum(x126, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x126, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x126
    x127 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x127 += einsum(f.ov, (0, 1), t2[np.ix_(so,sO,sv,sv)], (2, 3, 4, 1), (3, 0, 2, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x127, (4, 0, 5, 6), (1, 5, 4, 2, 6, 3))
    x128 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x128 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x127, (4, 1, 5, 6), (4, 3, 5, 0, 6, 2))
    del x127
    t3new += einsum(x128, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x128, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x128
    x129 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x129 += einsum(f.ov, (0, 1), t2[np.ix_(so,sO,sv,sv)], (2, 3, 1, 4), (3, 0, 2, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x129, (4, 1, 5, 6), (5, 0, 4, 6, 2, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x129, (4, 0, 5, 6), (1, 5, 4, 6, 2, 3))
    del x129
    x130 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x130 += einsum(f.ov, (0, 1), t2[np.ix_(so,so,sv,sV)], (2, 3, 1, 4), (4, 0, 3, 2))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x130, (4, 0, 5, 6), (5, 6, 1, 3, 2, 4))
    del x130
    x131 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x131 += einsum(f.ov, (0, 1), t2[np.ix_(so,sO,sV,sv)], (2, 3, 4, 1), (3, 4, 0, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x131, (4, 5, 1, 6), (6, 0, 4, 3, 2, 5))
    del x131
    x132 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x132 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oooO, (2, 0, 3, 4), (4, 3, 2, 1))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x132, (4, 0, 5, 6), (1, 5, 4, 2, 6, 3)) * -1.0
    x133 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x133 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x132, (4, 1, 5, 6), (4, 3, 0, 5, 6, 2))
    del x132
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x133
    x134 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x134 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oooO, (2, 3, 0, 4), (4, 2, 3, 1))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x134, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x134, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x134
    x135 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x135 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvO, (2, 1, 3, 4), (4, 0, 2, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x135, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x135, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3))
    del x135
    x136 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x136 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x62, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    t3new += einsum(x136, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x136, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x136
    x137 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x137 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovvO, (4, 2, 5, 6), (6, 3, 1, 0, 4, 5))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x137, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x137
    x138 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x138 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovOV, (4, 3, 5, 6), (5, 6, 0, 1, 4, 2))
    x139 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x139 += einsum(t1[np.ix_(so,sv)], (0, 1), x138, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x139
    x140 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x140 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvV, (2, 3, 1, 4), (3, 4, 0, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x140, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5))
    del x140
    x141 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x141 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvv, (2, 3, 4, 1), (3, 0, 2, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x141, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3))
    x142 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x142 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x141, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6))
    del x141
    t3new += einsum(x142, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x142, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x142
    x143 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x143 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oOvV, (4, 5, 3, 6), (5, 6, 0, 1, 4, 2))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x143, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x143
    x144 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x144 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oOvv, (4, 5, 6, 2), (5, 3, 1, 0, 4, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x144, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3))
    del x144
    x145 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x145 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oovV, (2, 3, 1, 4), (4, 0, 2, 3))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x145, (4, 5, 6, 0), (5, 6, 1, 3, 2, 4))
    del x145
    x146 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x146 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x36, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x36
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x146
    x147 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x147 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oooo, (4, 5, 6, 0), (1, 3, 4, 5, 6, 2))
    x148 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x148 += einsum(t1[np.ix_(so,sv)], (0, 1), x147, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 1, 6))
    del x147
    t3new += einsum(x148, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x148, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x148
    x149 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x149 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oovV, (4, 5, 3, 6), (1, 6, 0, 4, 5, 2))
    x150 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x150 += einsum(t1[np.ix_(so,sv)], (0, 1), x149, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x149
    t3new += einsum(x150, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x150, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x150
    x151 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x151 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oovv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x152 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x152 += einsum(t1[np.ix_(so,sv)], (0, 1), x151, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x151
    t3new += einsum(x152, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x152, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x152
    x153 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x153 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oovV, (4, 5, 2, 6), (1, 6, 0, 4, 5, 3))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x153, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3))
    del x153
    x154 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x154 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x9, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    t3new += einsum(x154, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x154, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x154
    x155 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x155 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovoV, (2, 1, 3, 4), (4, 0, 3, 2))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x155, (4, 5, 6, 0), (6, 5, 1, 3, 2, 4))
    x156 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x156 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovoV, (4, 3, 5, 6), (1, 6, 0, 5, 4, 2))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x156, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x156
    x157 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x157 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovov, (4, 5, 6, 2), (1, 3, 0, 4, 6, 5))
    x158 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x158 += einsum(t1[np.ix_(so,sv)], (0, 1), x157, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    t3new += einsum(x158, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x158, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x158
    x159 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x159 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovoV, (4, 2, 5, 6), (1, 6, 0, 5, 4, 3))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x159, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3))
    del x159
    x160 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x160 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oooO, (2, 3, 0, 4), (4, 1, 2, 3))
    x161 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x161 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x160, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x160
    t3new += einsum(x161, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x161, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x161
    x162 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x162 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oooO, (2, 0, 3, 4), (4, 1, 3, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x162, (4, 5, 1, 6), (6, 0, 4, 3, 2, 5)) * -1.0
    del x162
    x163 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x163 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovvO, (0, 2, 3, 4), (4, 1, 3, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x163, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5))
    del x163
    x164 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x164 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oOvv, (0, 2, 3, 4), (2, 1, 3, 4))
    x165 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x165 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x164, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x164
    t3new += einsum(x165, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x165, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x165
    x166 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x166 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oooo, (2, 3, 4, 0), (1, 2, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x166, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * -1.0
    del x166
    x167 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x167 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oovv, (2, 0, 3, 4), (1, 2, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x167, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x167, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4))
    del x167
    x168 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x168 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovov, (2, 3, 0, 4), (1, 2, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x168, (4, 5, 6, 2), (0, 5, 1, 3, 6, 4))
    x169 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x169 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x168, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6))
    t3new += einsum(x169, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x169, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x169
    x170 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x170 += einsum(t1[np.ix_(so,sv)], (0, 1), v.vvvV, (2, 1, 3, 4), (4, 0, 3, 2))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x170, (4, 5, 2, 6), (0, 5, 1, 3, 6, 4)) * -1.0
    x171 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x171 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x170, (4, 5, 3, 6), (1, 4, 5, 0, 2, 6))
    del x170
    t3new += einsum(x171, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x171, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x171
    x172 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x172 += einsum(t1[np.ix_(so,sv)], (0, 1), v.vvvV, (2, 3, 1, 4), (4, 0, 2, 3))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x172, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -1.0
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x172, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4)) * -1.0
    del x172
    x173 = np.zeros((naocc, navir, nocc, nvir, nvir, nvir), dtype=types[float])
    x173 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.vvvv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x174 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x174 += einsum(t1[np.ix_(so,sv)], (0, 1), x173, (2, 3, 4, 1, 5, 6), (2, 3, 0, 4, 5, 6))
    del x173
    t3new += einsum(x174, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x174, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x174
    x175 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x175 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.oovV, (2, 3, 1, 4), (0, 4, 2, 3))
    x176 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x176 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x175, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x175
    t3new += einsum(x176, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x176, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x176
    x177 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x177 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x177, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x177, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3))
    del x177
    x178 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x178 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovoV, (2, 1, 3, 4), (0, 4, 3, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x178, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5))
    del x178
    x179 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x179 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x179, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3))
    x180 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x180 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x179, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6))
    t3new += einsum(x180, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x180, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x180
    x181 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x181 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vvvV, (2, 1, 3, 4), (0, 4, 3, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x181, (4, 5, 2, 6), (0, 1, 4, 6, 3, 5)) * -1.0
    del x181
    x182 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x182 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vvvV, (2, 3, 1, 4), (0, 4, 2, 3))
    x183 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x183 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x182, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x182
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x183
    x184 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x184 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x184, (4, 5, 2, 6), (1, 0, 4, 6, 5, 3)) * -1.0
    del x184
    x185 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x185 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.oooo, (4, 5, 6, 0), (1, 2, 4, 5, 6, 3))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x185, (2, 3, 0, 4, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x185
    x186 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x186 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.oovv, (4, 5, 6, 3), (1, 2, 0, 4, 5, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x186, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3))
    del x186
    x187 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x187 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovov, (4, 5, 6, 3), (1, 2, 0, 4, 6, 5))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x187, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3))
    x188 = np.zeros((naocc, navir, nocc, nvir, nvir, nvir), dtype=types[float])
    x188 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.vvvv, (4, 5, 6, 3), (1, 2, 0, 4, 5, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x188, (2, 3, 4, 1, 5, 6), (4, 0, 2, 6, 5, 3)) * -1.0
    del x188
    x189 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x189 += einsum(v.ooov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 5, 3, 6, 7), (5, 7, 4, 0, 1, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x189, (2, 3, 4, 0, 5, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x189
    x190 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x190 += einsum(v.ooov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 4, 5, 3, 6, 7), (5, 7, 4, 0, 2, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x190, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x190
    x191 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x191 += einsum(v.ooov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 3, 6, 7), (5, 7, 4, 0, 2, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x191, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3))
    del x191
    x192 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x192 += einsum(v.ooov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 4, 5, 6, 3, 7), (5, 7, 4, 0, 2, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x192, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3))
    del x192
    x193 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x193 += einsum(v.ooov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 3, 6, 7), (5, 7, 4, 0, 1, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x193, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3))
    del x193
    x194 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x194 += einsum(v.ooov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 6, 3, 7), (5, 7, 4, 0, 1, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x194, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x194
    x195 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x195 += einsum(v.ooov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 6, 3, 7), (5, 7, 4, 0, 2, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x195, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x195
    x196 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x196 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovOO, (2, 1, 3, 4), (3, 4, 0, 2))
    t3new += einsum(x196, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * 0.5
    t3new += einsum(x196, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7))
    del x196
    x197 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x197 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvO, (2, 3, 1, 4), (3, 4, 0, 2))
    t3new += einsum(x197, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * -0.5
    del x197
    x198 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x198 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovVV, (2, 1, 3, 4), (3, 4, 0, 2))
    t3new += einsum(x198, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * -1.0
    t3new += einsum(x198, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0)) * -1.0
    del x198
    x199 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x199 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oVvV, (2, 3, 1, 4), (3, 4, 0, 2))
    t3new += einsum(x199, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 5, 6, 7, 0), (2, 4, 5, 6, 7, 1))
    del x199
    x200 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x200 += einsum(v.ovOO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 3, 1, 6, 7), (2, 7, 4, 5, 0, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x200, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3))
    del x200
    x201 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x201 += einsum(v.oOvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 3, 2, 6, 7), (1, 7, 4, 5, 0, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x201, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x201
    x202 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x202 += einsum(v.ovOO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 3, 6, 1, 7), (2, 7, 4, 5, 0, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x202, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x202
    x203 = np.zeros((naocc, naocc), dtype=types[float])
    x203 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovOO, (0, 1, 2, 3), (2, 3))
    t3new += einsum(x203, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    del x203
    x204 = np.zeros((naocc, naocc), dtype=types[float])
    x204 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvO, (0, 2, 1, 3), (2, 3))
    t3new += einsum(x204, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6))
    del x204
    x205 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x205 += einsum(v.ovvv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 1, 3, 7), (6, 7, 4, 5, 0, 2))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x205, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x205
    x206 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x206 += einsum(v.ovVV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 1, 7, 3), (6, 2, 4, 5, 0, 7))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x206, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -0.5
    del x206
    x207 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x207 += einsum(v.oVvV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 2, 7, 1), (6, 3, 4, 5, 0, 7))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x207, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * 0.5
    del x207
    x208 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x208 += einsum(v.ovvv, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 3, 1, 7), (6, 7, 4, 5, 0, 2))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x208, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x208
    x209 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x209 += einsum(v.ovVV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 7, 1, 3), (6, 2, 4, 5, 0, 7))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x209, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x209
    x210 = np.zeros((navir, navir), dtype=types[float])
    x210 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovVV, (0, 1, 2, 3), (2, 3))
    t3new += einsum(x210, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 2.0
    del x210
    x211 = np.zeros((navir, navir), dtype=types[float])
    x211 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oVvV, (0, 2, 1, 3), (2, 3))
    t3new += einsum(x211, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * -1.0
    del x211
    x212 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x212 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oooV, (2, 0, 3, 4), (1, 4, 3, 2))
    t3new += einsum(x212, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 5, 6, 7, 1), (3, 4, 5, 6, 7, 0)) * -1.0
    del x212
    x213 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x213 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oooV, (2, 3, 0, 4), (1, 4, 2, 3))
    t3new += einsum(x213, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0))
    t3new += einsum(x213, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0))
    del x213
    x214 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x214 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oVOO, (0, 2, 3, 4), (3, 4, 1, 2))
    t3new += einsum(x214, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    del x214
    x215 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x215 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oOOV, (0, 2, 3, 4), (3, 2, 1, 4))
    x216 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x216 += einsum(x215, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 0, 6, 7, 3), (1, 2, 4, 5, 6, 7))
    del x215
    t3new += einsum(x216, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x216, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x216
    x217 = np.zeros((navir, navir, nvir, nvir), dtype=types[float])
    x217 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oVvv, (0, 2, 3, 4), (1, 2, 3, 4))
    t3new += einsum(x217, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 3, 7, 1), (4, 5, 6, 2, 7, 0)) * -0.5
    t3new += einsum(x217, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 7, 3, 1), (4, 5, 6, 7, 2, 0)) * -1.0
    del x217
    x218 = np.zeros((navir, navir, nvir, nvir), dtype=types[float])
    x218 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovvV, (0, 2, 3, 4), (1, 4, 3, 2))
    t3new += einsum(x218, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 3, 7, 1), (4, 5, 6, 2, 7, 0)) * 0.5
    del x218
    x219 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x219 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.oovO, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new += einsum(x219, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * 0.5
    t3new += einsum(x219, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7))
    del x219
    x220 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x220 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovoO, (2, 1, 3, 4), (0, 4, 3, 2))
    t3new += einsum(x220, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * -0.5
    del x220
    x221 = np.zeros((naocc, naocc, nvir, nvir), dtype=types[float])
    x221 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vvvO, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new += einsum(x221, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7)) * -1.0
    t3new += einsum(x221, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 3, 7), (4, 5, 0, 6, 2, 7)) * -1.0
    del x221
    x222 = np.zeros((naocc, naocc, nvir, nvir), dtype=types[float])
    x222 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vvvO, (2, 1, 3, 4), (0, 4, 3, 2))
    t3new += einsum(x222, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 2, 6, 7), (4, 5, 0, 3, 6, 7))
    del x222
    x223 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x223 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vOVV, (1, 2, 3, 4), (0, 2, 3, 4))
    t3new += einsum(x223, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -1.0
    del x223
    x224 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x224 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vVOV, (1, 2, 3, 4), (0, 3, 4, 2))
    x225 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x225 += einsum(x224, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 2), (0, 3, 4, 5, 6, 7))
    del x224
    t3new += einsum(x225, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x225, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x225
    x226 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x226 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oooV, (2, 0, 3, 4), (4, 3, 2, 1))
    t3new += einsum(x226, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 1, 5, 6, 7, 0), (4, 2, 5, 6, 3, 7))
    del x226
    x227 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x227 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oVvv, (2, 3, 4, 1), (3, 0, 2, 4))
    t3new += einsum(x227, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 2, 5, 6, 7, 0), (4, 1, 5, 6, 3, 7)) * -1.0
    del x227
    x228 = np.zeros((naocc, naocc, navir, nvir), dtype=types[float])
    x228 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOOV, (0, 2, 3, 4), (3, 2, 4, 1))
    t3new += einsum(x228, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 3, 6, 7))
    del x228
    x229 = np.zeros((naocc, naocc, navir, nvir), dtype=types[float])
    x229 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vvOV, (2, 1, 3, 4), (0, 3, 4, 2))
    t3new += einsum(x229, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 5, 1, 6, 7, 2), (5, 4, 0, 3, 6, 7)) * -1.0
    del x229
    x230 = np.zeros((naocc, navir, navir, nocc), dtype=types[float])
    x230 += einsum(t1[np.ix_(so,sv)], (0, 1), v.vVOV, (1, 2, 3, 4), (3, 4, 2, 0))
    t3new += einsum(x230, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (4, 5, 0, 6, 7, 1), (3, 4, 5, 7, 6, 2)) * -1.0
    del x230
    x231 = np.zeros((naocc, navir, navir, nocc), dtype=types[float])
    x231 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ooOV, (2, 0, 3, 4), (3, 1, 4, 2))
    t3new += einsum(x231, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (4, 5, 0, 6, 7, 2), (3, 4, 5, 7, 6, 1))
    del x231
    x232 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x232 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ooOV, (2, 0, 3, 4), (3, 4, 2, 1))
    t3new += einsum(x232, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sV,sVf)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * -1.0
    del x232
    x233 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x233 += einsum(t1[np.ix_(so,sv)], (0, 1), v.vvOV, (2, 1, 3, 4), (3, 4, 0, 2))
    t3new += einsum(x233, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sV,sVf)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7))
    del x233
    x234 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x234 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovoO, (1, 3, 4, 5), (5, 0, 4, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x234, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * 2.0
    x235 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x235 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x234, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6))
    del x234
    t3new += einsum(x235, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new += einsum(x235, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x235
    x236 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x236 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoO, (4, 2, 1, 5), (5, 3, 0, 4))
    x237 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x237 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x236, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x236
    t3new += einsum(x237, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x237, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x237
    x238 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x238 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovoO, (1, 2, 4, 5), (5, 0, 4, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x238, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * -1.0
    x239 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x239 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x238, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6))
    del x238
    t3new += einsum(x239, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x239, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x239
    x240 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x240 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovoO, (4, 3, 1, 5), (5, 0, 4, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x240, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * -1.0
    x241 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x241 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x240, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6))
    del x240
    t3new += einsum(x241, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x241, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x241
    x242 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x242 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovoO, (4, 2, 1, 5), (5, 0, 4, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x242, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x242, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x242
    x243 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x243 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoO, (0, 4, 1, 5), (5, 3, 2, 4))
    x244 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x244 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x243, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x243
    t3new += einsum(x244, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x244, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x244, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 0.5
    del x244
    x245 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x245 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoO, (1, 4, 0, 5), (5, 3, 2, 4))
    x246 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x246 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x245, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x245
    t3new += einsum(x246, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x246, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -0.5
    del x246
    x247 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x247 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoO, (1, 2, 4, 5), (5, 3, 0, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x247, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    del x247
    x248 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x248 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoO, (4, 2, 0, 5), (5, 3, 1, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x248, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    del x248
    x249 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x249 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoO, (0, 2, 4, 5), (5, 3, 1, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x249, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * 2.0
    del x249
    x250 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x250 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovoO, (1, 4, 0, 5), (5, 2, 3, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x250, (4, 5, 6, 2), (1, 0, 4, 5, 6, 3)) * -1.0
    del x250
    x251 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x251 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x49, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x49
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x251
    x252 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x252 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 5, 0, 2), (1, 3, 4, 5))
    x253 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x253 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x252, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x252
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x253
    x254 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x254 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x43, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x43
    t3new += einsum(x254, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x254, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x254
    x255 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x255 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x45, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x45
    t3new += einsum(x255, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new += einsum(x255, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x255
    x256 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x256 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 0, 5, 2), (1, 3, 4, 5))
    x257 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x257 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x256, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x256
    t3new += einsum(x257, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x257, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x257
    x258 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x258 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x47, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x47
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x258
    x259 = np.zeros((naocc, navir, nocc, nocc, nocc, nocc), dtype=types[float])
    x259 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x260 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x260 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x259, (4, 5, 6, 7, 0, 1), (4, 5, 6, 7, 2, 3))
    t3new += einsum(x260, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x260, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x260
    x261 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x261 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 0, 5, 3), (1, 4, 5, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x261, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * -1.0
    x262 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x262 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x261, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2))
    del x261
    t3new += einsum(x262, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x262, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x262
    x263 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x263 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 5, 1, 2), (3, 0, 4, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x263, (4, 5, 6, 0), (5, 6, 1, 3, 2, 4)) * -1.0
    del x263
    x264 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x264 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 0, 5, 2), (1, 4, 5, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x264, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x264, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x264
    x265 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x265 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 5, 0, 3), (1, 4, 5, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x265, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x265, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x265
    x266 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x266 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 5, 0, 2), (1, 4, 5, 3))
    x267 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x267 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x266, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2))
    t3new += einsum(x267, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x267, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x267
    x268 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x268 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 1, 5, 2), (3, 0, 4, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x268, (4, 5, 6, 0), (6, 5, 1, 3, 2, 4)) * -1.0
    del x268
    x269 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x269 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 0, 1, 5), (3, 4, 2, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x269, (4, 5, 6, 2), (0, 5, 1, 3, 6, 4)) * -1.0
    x270 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x270 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x269, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6))
    del x269
    t3new += einsum(x270, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x270, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x270, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    del x270
    x271 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x271 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 1, 0, 5), (3, 4, 2, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x271, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4)) * -1.0
    x272 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x272 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x271, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6))
    del x271
    t3new += einsum(x272, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    t3new += einsum(x272, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    del x272
    x273 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x273 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x266, (4, 5, 0, 6), (4, 3, 1, 5, 6, 2))
    del x266
    t3new += einsum(x273, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x273, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x273
    x274 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x274 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 0, 5, 2), (3, 1, 4, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x274, (4, 5, 6, 0), (5, 6, 1, 3, 2, 4)) * -1.0
    del x274
    x275 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x275 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ooov, (4, 5, 0, 2), (3, 1, 4, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x275, (4, 5, 6, 0), (5, 6, 1, 3, 2, 4)) * 2.0
    del x275
    x276 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x276 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvV, (1, 3, 4, 5), (5, 0, 2, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x276, (4, 5, 6, 2), (0, 5, 1, 3, 6, 4)) * -2.0
    x277 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x277 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x276, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6))
    del x276
    t3new += einsum(x277, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new += einsum(x277, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x277
    x278 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x278 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvV, (1, 2, 4, 5), (5, 0, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x278, (4, 5, 6, 2), (0, 5, 1, 3, 6, 4))
    x279 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x279 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x278, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6))
    del x278
    t3new += einsum(x279, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x279, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x279
    x280 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x280 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvV, (1, 4, 3, 5), (5, 0, 2, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x280, (4, 5, 6, 2), (0, 5, 1, 3, 6, 4))
    x281 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x281 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x280, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6))
    del x280
    t3new += einsum(x281, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x281, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x281
    x282 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x282 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvV, (4, 2, 3, 5), (1, 5, 0, 4))
    x283 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x283 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x282, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x282
    t3new += einsum(x283, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x283, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x283
    x284 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x284 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvV, (1, 4, 2, 5), (5, 0, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x284, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x284, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4))
    del x284
    x285 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x285 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvV, (0, 4, 3, 5), (1, 5, 2, 4))
    x286 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x286 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x285, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x285
    t3new += einsum(x286, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x286, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x286
    x287 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x287 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvV, (0, 4, 2, 5), (1, 5, 3, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x287, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5))
    del x287
    x288 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x288 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvV, (0, 3, 4, 5), (1, 5, 2, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x288, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5))
    del x288
    x289 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x289 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvV, (4, 3, 2, 5), (5, 0, 1, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x289, (4, 5, 6, 0), (5, 6, 1, 3, 2, 4))
    del x289
    x290 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x290 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvV, (0, 2, 4, 5), (1, 5, 3, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x290, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * -2.0
    del x290
    x291 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x291 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvV, (4, 3, 2, 5), (1, 5, 0, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x291, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5))
    del x291
    x292 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x292 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x113, (4, 5, 6, 1, 3, 7), (4, 5, 6, 0, 2, 7))
    t3new += einsum(x292, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new += einsum(x292, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x292
    x293 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x293 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x113, (4, 5, 6, 1, 2, 7), (4, 5, 6, 0, 3, 7))
    t3new += einsum(x293, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x293, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x293
    x294 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x294 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovvv, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    x295 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x295 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x294, (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 2, 7))
    t3new += einsum(x295, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x295, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x295
    x296 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x296 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x294, (4, 5, 6, 1, 7, 2), (4, 5, 6, 0, 3, 7))
    t3new += einsum(x296, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x296, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x296
    x297 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x297 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovvv, (0, 4, 5, 2), (1, 3, 4, 5))
    x298 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x298 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x297, (4, 5, 3, 6), (4, 5, 0, 1, 2, 6))
    del x297
    t3new += einsum(x298, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x298, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x298
    x299 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x299 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x58, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x58
    t3new += einsum(x299, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x299, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x299
    x300 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x300 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovvv, (0, 2, 4, 5), (1, 3, 4, 5))
    x301 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x301 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x300, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x300
    t3new += einsum(x301, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new += einsum(x301, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x301
    x302 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x302 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvv, (4, 3, 5, 2), (1, 0, 4, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x302, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3))
    x303 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x303 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x302, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6))
    del x302
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x303
    x304 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x304 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovvv, (1, 2, 4, 5), (3, 0, 4, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x304, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x304, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4))
    del x304
    x305 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x305 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovvv, (1, 4, 5, 2), (3, 0, 4, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x305, (4, 5, 2, 6), (0, 5, 1, 3, 6, 4))
    x306 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x306 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x305, (4, 5, 3, 6), (1, 4, 0, 5, 2, 6))
    del x305
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x306
    x307 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x307 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvv, (4, 2, 5, 3), (1, 0, 4, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x307, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x307, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3))
    del x307
    x308 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x308 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvv, (0, 4, 5, 3), (1, 2, 4, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x308, (4, 5, 2, 6), (1, 0, 4, 6, 5, 3))
    del x308
    x309 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x309 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvv, (0, 4, 5, 2), (1, 3, 4, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x309, (4, 5, 2, 6), (1, 0, 4, 5, 6, 3))
    del x309
    x310 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x310 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvv, (0, 3, 4, 5), (1, 2, 4, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x310, (4, 5, 6, 2), (1, 0, 4, 5, 6, 3))
    del x310
    x311 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x311 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvv, (0, 2, 4, 5), (1, 3, 4, 5))
    x312 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x312 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x311, (4, 5, 6, 2), (4, 3, 1, 0, 5, 6))
    del x311
    t3new += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x312
    x313 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x313 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovvv, (0, 2, 4, 5), (3, 1, 4, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x313, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -2.0
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x313, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4)) * -2.0
    del x313
    x314 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x314 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovvv, (0, 4, 5, 2), (3, 1, 4, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x314, (4, 5, 3, 6), (5, 0, 1, 6, 2, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x314, (4, 5, 2, 6), (5, 0, 1, 3, 6, 4))
    del x314
    x315 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x315 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ooov, (4, 5, 0, 3), (1, 2, 4, 5))
    x316 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x316 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x315, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x315
    t3new += einsum(x316, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x316, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x316
    x317 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x317 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ooov, (4, 0, 5, 3), (1, 2, 4, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x317, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    del x317
    x318 = np.zeros((naocc, navir, nocc, nocc, nocc, nocc), dtype=types[float])
    x318 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ooov, (4, 5, 6, 3), (1, 2, 0, 4, 5, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x318, (4, 5, 6, 7, 0, 1), (6, 7, 4, 3, 2, 5)) * -1.0
    x319 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x319 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovvv, (0, 4, 5, 3), (1, 2, 4, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x319, (4, 5, 2, 6), (0, 1, 4, 6, 3, 5))
    del x319
    x320 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x320 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovvv, (0, 3, 4, 5), (1, 2, 4, 5))
    x321 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x321 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x320, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x320
    t3new += einsum(x321, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x321, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x321
    x322 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x322 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovvv, (4, 3, 5, 6), (1, 2, 0, 4, 5, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x322, (4, 5, 6, 1, 7, 3), (6, 0, 4, 7, 2, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x322, (4, 5, 6, 1, 7, 2), (6, 0, 4, 3, 7, 5))
    x323 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x323 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 2, 4, 5, 3, 6), (4, 6, 5, 1))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x323, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * -1.0
    del x323
    x324 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x324 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 2, 4, 1, 5, 6), (4, 6, 5, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x324, (4, 5, 6, 3), (0, 1, 4, 2, 6, 5)) * -1.0
    del x324
    x325 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x325 += einsum(x12, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 2, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5))
    del x12
    t3new += einsum(x325, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x325, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x325
    x326 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x326 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 5, 3, 1, 6), (5, 6, 4, 0))
    x327 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x327 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x326, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x326
    t3new += einsum(x327, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x327, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x327
    x328 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x328 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 5, 1, 6, 7), (5, 7, 4, 0, 6, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x328, (4, 5, 6, 1, 7, 3), (0, 6, 4, 2, 7, 5)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x328, (4, 5, 6, 1, 7, 2), (0, 6, 4, 3, 7, 5))
    del x328
    x329 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x329 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 5, 6, 1, 7), (5, 7, 4, 0, 6, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x329, (4, 5, 6, 1, 7, 2), (0, 6, 4, 7, 3, 5))
    del x329
    x330 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x330 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 1, 3, 6), (5, 6, 4, 0))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x330, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    del x330
    x331 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x331 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 5, 1, 3, 6), (5, 6, 4, 0))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x331, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5))
    del x331
    x332 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x332 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 6, 1, 7), (5, 7, 4, 0, 6, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x332, (4, 5, 6, 1, 7, 3), (6, 0, 4, 7, 2, 5)) * -2.0
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x332, (4, 5, 6, 1, 7, 2), (6, 0, 4, 7, 3, 5))
    del x332
    x333 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x333 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 1, 6, 7), (5, 7, 4, 0, 6, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x333, (4, 5, 6, 1, 7, 3), (6, 0, 4, 7, 2, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x333, (4, 5, 6, 1, 7, 2), (6, 0, 4, 3, 7, 5))
    del x333
    x334 = np.zeros((naocc, navir, nocc, nocc, nocc, nocc), dtype=types[float])
    x334 += einsum(v.ovov, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 1, 3, 7), (6, 7, 4, 5, 0, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x334, (4, 5, 6, 7, 0, 1), (6, 7, 4, 2, 3, 5))
    x335 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x335 += einsum(x85, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 1, 5, 6), (4, 6, 2, 3, 0, 5))
    t3new += einsum(x335, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x335, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    del x335
    x336 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x336 += einsum(x85, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    del x85
    t3new += einsum(x336, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x336, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    del x336
    x337 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x337 += einsum(x83, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 1, 5, 6), (4, 6, 2, 3, 0, 5))
    t3new += einsum(x337, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x337, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.5
    del x337
    x338 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x338 += einsum(x83, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    del x83
    t3new += einsum(x338, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.5
    t3new += einsum(x338, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    del x338
    x339 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x339 += einsum(v.ovoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 5, 1, 6, 3), (5, 4, 0, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x339, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 0.5
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x339, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * -0.5
    del x339
    x340 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x340 += einsum(v.ovoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 4, 5, 1, 6, 3), (5, 4, 2, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x340, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -0.5
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x340, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * 0.5
    del x340
    x341 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x341 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (1, 2, 4, 5), (3, 5, 0, 4))
    t3new += einsum(x341, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * -1.0
    del x341
    x342 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x342 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (4, 2, 1, 5), (3, 5, 0, 4))
    t3new += einsum(x342, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0))
    t3new += einsum(x342, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0))
    del x342
    x343 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x343 += einsum(v.ovoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 0, 4, 5, 6, 3), (4, 5, 6, 1))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x343, (4, 5, 6, 2), (1, 0, 4, 5, 6, 3)) * -1.0
    del x343
    x344 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x344 += einsum(v.ovoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 6, 1, 3), (5, 4, 0, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x344, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3))
    del x344
    x345 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x345 += einsum(v.ovoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 5, 6, 1, 3), (5, 4, 2, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x345, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -2.0
    del x345
    x346 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x346 += einsum(v.ovoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 1, 6, 3), (5, 4, 0, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x346, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -0.5
    del x346
    x347 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x347 += einsum(v.ovoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 5, 1, 6, 3), (5, 4, 2, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x347, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * 0.5
    del x347
    x348 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x348 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (0, 2, 4, 5), (3, 5, 1, 4))
    t3new += einsum(x348, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * 2.0
    del x348
    x349 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x349 += einsum(v.ovoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 5, 6, 1, 3), (5, 4, 0, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x349, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x349
    x350 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x350 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (4, 2, 0, 5), (3, 5, 1, 4))
    t3new += einsum(x350, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * -1.0
    del x350
    x351 = np.zeros((navir, navir, nvir, nvir), dtype=types[float])
    x351 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (0, 4, 1, 5), (3, 5, 2, 4))
    t3new += einsum(x351, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 3, 7, 1), (4, 5, 6, 2, 7, 0)) * 0.25
    del x351
    x352 = np.zeros((navir, navir, nvir, nvir), dtype=types[float])
    x352 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (1, 4, 0, 5), (3, 5, 2, 4))
    t3new += einsum(x352, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 3, 7, 1), (4, 5, 6, 2, 7, 0)) * -0.25
    del x352
    x353 = np.zeros((navir, navir), dtype=types[float])
    x353 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (0, 2, 1, 4), (3, 4))
    t3new += einsum(x353, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -0.5
    del x353
    x354 = np.zeros((navir, navir), dtype=types[float])
    x354 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (1, 2, 0, 4), (3, 4))
    t3new += einsum(x354, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 0.5
    del x354
    x355 = np.zeros((navir, navir, nvir, nvir), dtype=types[float])
    x355 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (1, 4, 0, 5), (3, 5, 2, 4))
    t3new += einsum(x355, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 3, 7, 1), (4, 5, 6, 2, 7, 0)) * -0.25
    del x355
    x356 = np.zeros((navir, navir, nvir, nvir), dtype=types[float])
    x356 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (0, 4, 1, 5), (3, 5, 2, 4))
    t3new += einsum(x356, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 3, 7, 1), (4, 5, 6, 2, 7, 0)) * 0.25
    t3new += einsum(x356, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 7, 3, 1), (4, 5, 6, 7, 2, 0))
    del x356
    x357 = np.zeros((navir, navir), dtype=types[float])
    x357 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (1, 2, 0, 4), (3, 4))
    t3new += einsum(x357, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 0.5
    del x357
    x358 = np.zeros((navir, navir), dtype=types[float])
    x358 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (0, 2, 1, 4), (3, 4))
    t3new += einsum(x358, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -1.5
    del x358
    x359 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x359 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 4, 3, 2, 5, 6), (6, 4, 5, 1))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x359, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * 0.5
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x359, (4, 5, 6, 2), (0, 5, 1, 3, 6, 4)) * -0.5
    del x359
    x360 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x360 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvO, (4, 2, 3, 5), (1, 5, 0, 4))
    t3new += einsum(x360, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * 0.5
    del x360
    x361 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x361 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 4, 3, 1, 5, 6), (6, 4, 5, 2))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x361, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * -0.5
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x361, (4, 5, 6, 2), (0, 5, 1, 3, 6, 4)) * 0.5
    del x361
    x362 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x362 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvO, (4, 3, 2, 5), (1, 5, 0, 4))
    t3new += einsum(x362, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * -0.5
    del x362
    x363 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x363 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 3, 5, 2, 6), (6, 4, 5, 1))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x363, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * 0.5
    del x363
    x364 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x364 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 3, 2, 5, 6), (6, 4, 5, 1))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x364, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -0.5
    del x364
    x365 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x365 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 3, 5, 1, 6), (6, 4, 5, 2))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x365, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -1.0
    del x365
    x366 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x366 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvO, (4, 2, 3, 5), (1, 5, 0, 4))
    t3new += einsum(x366, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7))
    del x366
    x367 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x367 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 3, 5, 1, 6), (6, 4, 5, 2))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x367, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -0.5
    del x367
    x368 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x368 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 3, 1, 5, 6), (6, 4, 5, 2))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x368, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * 0.5
    del x368
    x369 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x369 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 0, 3, 2, 5, 6), (6, 4, 5, 1))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x369, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4)) * -1.0
    del x369
    x370 = np.zeros((naocc, naocc, nvir, nvir), dtype=types[float])
    x370 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvO, (0, 3, 4, 5), (1, 5, 2, 4))
    t3new += einsum(x370, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7)) * -1.0
    del x370
    x371 = np.zeros((naocc, naocc, nvir, nvir), dtype=types[float])
    x371 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvO, (0, 2, 4, 5), (1, 5, 3, 4))
    t3new += einsum(x371, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7)) * 2.0
    del x371
    x372 = np.zeros((naocc, naocc), dtype=types[float])
    x372 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvO, (0, 2, 3, 4), (1, 4))
    x373 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x373 += einsum(x372, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 1, 4, 5, 6), (0, 6, 2, 3, 4, 5))
    del x372
    t3new += einsum(x373, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x373, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x373
    x374 = np.zeros((naocc, naocc, nvir, nvir), dtype=types[float])
    x374 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvO, (0, 4, 3, 5), (1, 5, 2, 4))
    t3new += einsum(x374, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7))
    t3new += einsum(x374, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 3, 7), (4, 5, 0, 6, 2, 7))
    del x374
    x375 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x375 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 3, 2, 1, 6), (6, 4, 5, 0))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x375, (4, 5, 6, 0), (5, 6, 1, 3, 2, 4)) * -1.0
    del x375
    x376 = np.zeros((naocc, naocc, nvir, nvir), dtype=types[float])
    x376 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvO, (0, 4, 2, 5), (1, 5, 3, 4))
    t3new += einsum(x376, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7)) * -1.0
    del x376
    x377 = np.zeros((naocc, naocc), dtype=types[float])
    x377 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvO, (0, 3, 2, 4), (1, 4))
    t3new += einsum(x377, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6))
    del x377
    x378 = np.zeros((naocc, naocc, navir, navir, nocc, nocc), dtype=types[float])
    x378 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovOV, (4, 2, 5, 6), (1, 5, 3, 6, 0, 4))
    t3new += einsum(t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 1, 2, 3, 4, 5), x378, (6, 2, 7, 5, 8, 1), (8, 0, 6, 4, 3, 7)) * -1.0
    x379 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x379 += einsum(t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 1, 2, 3, 4, 5), x378, (6, 2, 7, 5, 8, 0), (6, 7, 8, 1, 3, 4))
    del x378
    t3new += einsum(x379, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x379, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    del x379
    x380 = np.zeros((naocc, naocc, navir, navir, nocc, nocc), dtype=types[float])
    x380 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oVvO, (4, 5, 2, 6), (1, 6, 3, 5, 0, 4))
    x381 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x381 += einsum(t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 1, 2, 3, 4, 5), x380, (6, 2, 7, 5, 8, 0), (6, 7, 8, 1, 3, 4))
    del x380
    t3new += einsum(x381, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x381, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    del x381
    x382 = np.zeros((naocc, naocc, navir, navir, nocc, nocc), dtype=types[float])
    x382 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovOV, (4, 2, 5, 6), (1, 5, 3, 6, 0, 4))
    t3new += einsum(t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 1, 2, 3, 4, 5), x382, (6, 2, 7, 5, 8, 1), (0, 8, 6, 3, 4, 7)) * -1.0
    del x382
    x383 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x383 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x67, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x67
    t3new += einsum(x383, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x383, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    del x383
    x384 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x384 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x69, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x69
    t3new += einsum(x384, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x384, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    del x384
    x385 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x385 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x71, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x71
    t3new += einsum(x385, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x385, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x385
    x386 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x386 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovOV, (0, 2, 4, 5), (1, 4, 3, 5))
    x387 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x387 += einsum(x386, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (0, 2, 4, 5, 6, 7))
    del x386
    t3new += einsum(x387, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new += einsum(x387, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x387
    x388 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x388 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oVvO, (0, 4, 2, 5), (1, 5, 3, 4))
    x389 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x389 += einsum(x388, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (0, 2, 4, 5, 6, 7))
    del x388
    t3new += einsum(x389, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x389, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x389
    x390 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x390 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovoV, (1, 3, 4, 5), (5, 0, 4, 2))
    t3new += einsum(x390, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 2, 5, 6, 7, 0), (4, 1, 5, 6, 3, 7)) * -2.0
    del x390
    x391 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x391 += einsum(v.ovoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (2, 4, 5, 1, 6, 3), (5, 6, 4, 0))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x391, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -0.5
    del x391
    x392 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x392 += einsum(v.ovoV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (0, 4, 5, 1, 6, 3), (5, 6, 4, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x392, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * 0.5
    del x392
    x393 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x393 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovoV, (1, 2, 4, 5), (5, 0, 4, 3))
    t3new += einsum(x393, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 2, 5, 6, 7, 0), (4, 1, 5, 6, 3, 7))
    del x393
    x394 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x394 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovoV, (4, 3, 1, 5), (5, 0, 4, 2))
    t3new += einsum(x394, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 2, 5, 6, 7, 0), (4, 1, 5, 6, 3, 7))
    del x394
    x395 = np.zeros((naocc, naocc, navir, nocc, nocc, nvir), dtype=types[float])
    x395 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovOV, (4, 3, 5, 6), (1, 5, 6, 0, 4, 2))
    x396 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x396 += einsum(t3[np.ix_(so,so,sOf,sv,sV,sVf)], (0, 1, 2, 3, 4, 5), x395, (6, 2, 5, 7, 1, 8), (6, 4, 7, 0, 8, 3))
    del x395
    t3new += einsum(x396, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x396, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    del x396
    x397 = np.zeros((naocc, naocc, navir, nocc, nocc, nvir), dtype=types[float])
    x397 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovOV, (4, 2, 5, 6), (1, 5, 6, 0, 4, 3))
    t3new += einsum(t3[np.ix_(so,so,sOf,sv,sV,sVf)], (0, 1, 2, 3, 4, 5), x397, (6, 2, 5, 7, 1, 8), (7, 0, 6, 8, 3, 4))
    t3new += einsum(t3[np.ix_(so,so,sOf,sv,sV,sVf)], (0, 1, 2, 3, 4, 5), x397, (6, 2, 5, 7, 0, 8), (1, 7, 6, 8, 3, 4)) * 0.5
    del x397
    x398 = np.zeros((naocc, naocc, navir, nocc, nocc, nvir), dtype=types[float])
    x398 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oVvO, (4, 5, 3, 6), (1, 6, 5, 0, 4, 2))
    t3new += einsum(t3[np.ix_(so,so,sOf,sv,sV,sVf)], (0, 1, 2, 3, 4, 5), x398, (6, 2, 5, 7, 1, 8), (0, 7, 6, 3, 8, 4)) * 0.5
    del x398
    x399 = np.zeros((naocc, naocc, navir, nocc, nocc, nvir), dtype=types[float])
    x399 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oVvO, (4, 5, 2, 6), (1, 6, 5, 0, 4, 3))
    t3new += einsum(t3[np.ix_(so,so,sOf,sv,sV,sVf)], (0, 1, 2, 3, 4, 5), x399, (6, 2, 5, 7, 0, 8), (1, 7, 6, 8, 3, 4)) * -0.5
    del x399
    x400 = np.zeros((naocc, naocc, navir, nvir), dtype=types[float])
    x400 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovOV, (0, 3, 4, 5), (1, 4, 5, 2))
    t3new += einsum(x400, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 5, 1, 6, 7, 2), (5, 4, 0, 3, 6, 7))
    del x400
    x401 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x401 += einsum(v.ovOV, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 5, 2, 1, 6, 3), (6, 4, 5, 0))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x401, (4, 5, 6, 0), (6, 5, 1, 3, 2, 4)) * 0.5
    del x401
    x402 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x402 += einsum(v.oVvO, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 5, 3, 2, 6, 1), (6, 4, 5, 0))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x402, (4, 5, 6, 0), (6, 5, 1, 3, 2, 4)) * -0.5
    del x402
    x403 = np.zeros((naocc, naocc, navir, nvir), dtype=types[float])
    x403 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovOV, (0, 2, 4, 5), (1, 4, 5, 3))
    t3new += einsum(x403, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 5, 1, 6, 7, 2), (5, 4, 0, 3, 6, 7)) * -2.0
    del x403
    x404 = np.zeros((naocc, naocc, navir, nvir), dtype=types[float])
    x404 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oVvO, (0, 4, 2, 5), (1, 5, 4, 3))
    t3new += einsum(x404, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 5, 1, 6, 7, 2), (5, 4, 0, 3, 6, 7))
    del x404
    x405 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x405 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (0, 4, 3, 5, 2, 6), (4, 6, 5, 1))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x405, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * 0.5
    del x405
    x406 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x406 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (0, 4, 3, 2, 5, 6), (4, 6, 5, 1))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x406, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * -0.5
    del x406
    x407 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x407 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (0, 4, 3, 5, 1, 6), (4, 6, 5, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x407, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * -0.5
    del x407
    x408 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x408 += einsum(v.ovvO, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (0, 4, 3, 1, 5, 6), (4, 6, 5, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x408, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * 0.5
    del x408
    x409 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x409 += einsum(v.ovOV, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (4, 5, 2, 6, 1, 3), (5, 4, 0, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x409, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x409, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3))
    del x409
    x410 = np.zeros((naocc, navir, navir, nocc), dtype=types[float])
    x410 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovOV, (1, 2, 4, 5), (4, 3, 5, 0))
    t3new += einsum(x410, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (4, 5, 0, 6, 7, 2), (3, 4, 5, 7, 6, 1))
    del x410
    x411 = np.zeros((naocc, navir, navir, nocc, nocc, nocc), dtype=types[float])
    x411 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovOV, (4, 2, 5, 6), (5, 3, 6, 1, 0, 4))
    t3new += einsum(t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (0, 1, 2, 3, 4, 5), x411, (2, 6, 5, 7, 8, 0), (7, 8, 1, 4, 3, 6)) * 0.5
    del x411
    x412 = np.zeros((naocc, navir, navir, nocc, nocc, nocc), dtype=types[float])
    x412 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oVvO, (4, 5, 2, 6), (6, 3, 5, 1, 0, 4))
    t3new += einsum(t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (0, 1, 2, 3, 4, 5), x412, (2, 6, 5, 7, 8, 0), (7, 8, 1, 4, 3, 6)) * -0.5
    del x412
    x413 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x413 += einsum(v.ovOV, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (4, 5, 2, 1, 6, 3), (5, 4, 0, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x413, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * 0.5
    del x413
    x414 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x414 += einsum(v.oVvO, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (4, 5, 3, 2, 6, 1), (5, 4, 0, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x414, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -0.5
    del x414
    x415 = np.zeros((naocc, navir, navir, nocc), dtype=types[float])
    x415 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovOV, (0, 2, 4, 5), (4, 3, 5, 1))
    t3new += einsum(x415, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (4, 5, 0, 6, 7, 2), (3, 4, 5, 7, 6, 1)) * -2.0
    del x415
    x416 = np.zeros((naocc, navir, navir, nocc), dtype=types[float])
    x416 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oVvO, (0, 4, 2, 5), (5, 3, 4, 1))
    t3new += einsum(x416, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (4, 5, 0, 6, 7, 2), (3, 4, 5, 7, 6, 1))
    del x416
    x417 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x417 += einsum(t3[np.ix_(so,sO,sOf,sv,sV,sVf)], (0, 1, 2, 3, 4, 5), x138, (2, 5, 6, 7, 0, 8), (1, 4, 6, 7, 8, 3))
    del x138
    t3new += einsum(x417, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x417, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -0.5
    del x417
    x418 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x418 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oVvO, (4, 5, 3, 6), (6, 5, 0, 1, 4, 2))
    x419 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x419 += einsum(t3[np.ix_(so,sO,sOf,sv,sV,sVf)], (0, 1, 2, 3, 4, 5), x418, (2, 5, 6, 7, 0, 8), (1, 4, 6, 7, 8, 3))
    del x418
    t3new += einsum(x419, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x419, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 0.5
    del x419
    x420 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x420 += einsum(v.ovOV, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sV,sVf)], (4, 5, 2, 1, 6, 3), (5, 6, 4, 0))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x420, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -0.5
    del x420
    x421 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x421 += einsum(v.oVvO, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sV,sVf)], (4, 5, 3, 2, 6, 1), (5, 6, 4, 0))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x421, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * 0.5
    del x421
    x422 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x422 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovOV, (1, 3, 4, 5), (4, 5, 0, 2))
    t3new += einsum(x422, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sV,sVf)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * 2.0
    del x422
    x423 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x423 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovOV, (1, 2, 4, 5), (4, 5, 0, 3))
    t3new += einsum(x423, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sV,sVf)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * -1.0
    del x423
    x424 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x424 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oVvO, (1, 4, 3, 5), (5, 4, 0, 2))
    t3new += einsum(x424, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sV,sVf)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * -1.0
    del x424
    x425 = np.zeros((naocc, naocc, navir, navir, nocc, nocc), dtype=types[float])
    x425 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovOV, (4, 3, 5, 6), (1, 5, 2, 6, 0, 4))
    t3new += einsum(t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 1, 2, 3, 4, 5), x425, (6, 2, 7, 5, 8, 1), (8, 0, 6, 4, 3, 7))
    t3new += einsum(t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 1, 2, 3, 4, 5), x425, (6, 2, 7, 5, 8, 0), (8, 1, 6, 3, 4, 7)) * 0.5
    del x425
    x426 = np.zeros((naocc, naocc, navir, navir, nocc, nocc), dtype=types[float])
    x426 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.oVvO, (4, 5, 3, 6), (1, 6, 2, 5, 0, 4))
    t3new += einsum(t3[np.ix_(so,so,sOf,sv,sv,sVf)], (0, 1, 2, 3, 4, 5), x426, (6, 2, 7, 5, 8, 0), (8, 1, 6, 3, 4, 7)) * -0.5
    del x426
    x427 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x427 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovOV, (0, 3, 4, 5), (1, 4, 2, 5))
    x428 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x428 += einsum(x427, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (0, 2, 4, 5, 6, 7))
    del x427
    t3new += einsum(x428, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x428, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x428
    x429 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x429 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.oVvO, (0, 4, 3, 5), (1, 5, 2, 4))
    t3new += einsum(x429, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    del x429
    x430 = np.zeros((naocc, nocc, nocc, nocc), dtype=types[float])
    x430 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovoO, (2, 1, 3, 4), (4, 0, 3, 2))
    x431 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x431 += einsum(t1[np.ix_(so,sv)], (0, 1), x430, (2, 3, 4, 0), (2, 3, 4, 1))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x431, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * -1.0
    x432 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x432 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x431, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    del x431
    t3new += einsum(x432, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x432, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x432
    x433 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x433 += einsum(t1[np.ix_(so,sv)], (0, 1), x430, (2, 3, 0, 4), (2, 3, 4, 1))
    del x430
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x433, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x433, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x433
    x434 = np.zeros((naocc, navir, nocc, nocc, nocc, nocc), dtype=types[float])
    x434 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoO, (4, 2, 5, 6), (6, 3, 1, 0, 5, 4))
    x435 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x435 += einsum(t1[np.ix_(so,sv)], (0, 1), x434, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x434
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x435, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x435
    x436 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x436 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x41, (4, 5, 0, 6), (1, 3, 4, 5, 6, 2))
    x437 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x437 += einsum(t1[np.ix_(so,sv)], (0, 1), x436, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x436
    t3new += einsum(x437, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x437, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x437
    x438 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x438 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x41, (4, 5, 6, 0), (1, 3, 4, 6, 5, 2))
    x439 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x439 += einsum(t1[np.ix_(so,sv)], (0, 1), x438, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x438
    t3new += einsum(x439, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x439, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x439
    x440 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x440 += einsum(t1[np.ix_(so,sv)], (0, 1), x259, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x259
    x441 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x441 += einsum(t1[np.ix_(so,sv)], (0, 1), x440, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x440
    t3new += einsum(x441, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x441, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x441
    x442 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x442 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvV, (2, 1, 3, 4), (4, 0, 2, 3))
    x443 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x443 += einsum(t1[np.ix_(so,sv)], (0, 1), x442, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x443, (4, 5, 6, 0), (5, 6, 1, 3, 2, 4))
    del x443
    x444 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x444 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x90, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x90
    t3new += einsum(x444, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x444, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x444
    x445 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x445 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x442, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2))
    x446 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x446 += einsum(t1[np.ix_(so,sv)], (0, 1), x445, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x445
    t3new += einsum(x446, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x446, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x446
    x447 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x447 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvV, (2, 3, 1, 4), (4, 0, 2, 3))
    x448 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x448 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x447, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x448, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3))
    del x448
    x449 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x449 += einsum(t1[np.ix_(so,sv)], (0, 1), x113, (2, 3, 4, 5, 1, 6), (2, 3, 0, 4, 5, 6))
    del x113
    x450 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x450 += einsum(t1[np.ix_(so,sv)], (0, 1), x449, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x449
    t3new += einsum(x450, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x450, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x450
    x451 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x451 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x447, (4, 5, 6, 2), (1, 4, 5, 0, 6, 3))
    del x447
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x451, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x451
    x452 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x452 += einsum(t1[np.ix_(so,sv)], (0, 1), x294, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x294
    x453 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x453 += einsum(t1[np.ix_(so,sv)], (0, 1), x452, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x452
    t3new += einsum(x453, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x453, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x453
    x454 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x454 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x442, (4, 5, 6, 2), (1, 4, 5, 0, 6, 3))
    del x442
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x454, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3))
    del x454
    x455 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x455 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovoO, (0, 2, 3, 4), (4, 1, 3, 2))
    x456 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x456 += einsum(t1[np.ix_(so,sv)], (0, 1), x455, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x456, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    del x456
    x457 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x457 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovoO, (2, 3, 0, 4), (4, 1, 2, 3))
    x458 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x458 += einsum(t1[np.ix_(so,sv)], (0, 1), x457, (2, 3, 4, 1), (2, 3, 0, 4))
    x459 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x459 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x458, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x458
    t3new += einsum(x459, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x459, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x459
    x460 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x460 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x457, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x457
    x461 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x461 += einsum(t1[np.ix_(so,sv)], (0, 1), x460, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x460
    t3new += einsum(x461, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x461, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x461
    x462 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x462 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x455, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x455
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x462, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x462
    x463 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x463 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ooov, (2, 3, 0, 4), (1, 2, 3, 4))
    x464 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x464 += einsum(t1[np.ix_(so,sv)], (0, 1), x463, (2, 3, 4, 1), (2, 0, 4, 3))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x464, (4, 5, 0, 6), (5, 6, 1, 3, 2, 4)) * -1.0
    del x464
    x465 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x465 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ooov, (2, 0, 3, 4), (1, 2, 3, 4))
    x466 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x466 += einsum(t1[np.ix_(so,sv)], (0, 1), x465, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x466, (4, 5, 6, 0), (6, 5, 1, 3, 2, 4)) * -1.0
    del x466
    x467 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x467 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x463, (4, 5, 6, 3), (1, 4, 0, 6, 5, 2))
    x468 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x468 += einsum(t1[np.ix_(so,sv)], (0, 1), x467, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x467
    t3new += einsum(x468, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x468, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x468
    x469 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x469 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x465, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x469, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x469
    x470 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x470 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x463, (4, 5, 6, 2), (1, 4, 0, 6, 5, 3))
    del x463
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x470, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x470
    x471 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x471 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x465, (4, 5, 6, 2), (1, 4, 0, 5, 6, 3))
    del x465
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x471, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x471
    x472 = np.zeros((navir, nvir, nvir, nvir), dtype=types[float])
    x472 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovvv, (0, 2, 3, 4), (1, 2, 3, 4))
    x473 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x473 += einsum(t1[np.ix_(so,sv)], (0, 1), x472, (2, 1, 3, 4), (2, 0, 4, 3))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x473, (4, 5, 3, 6), (5, 0, 1, 6, 2, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x473, (4, 5, 2, 6), (5, 0, 1, 3, 6, 4))
    del x473
    x474 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x474 += einsum(t1[np.ix_(so,sv)], (0, 1), x472, (2, 3, 4, 1), (2, 0, 3, 4))
    del x472
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x474, (4, 5, 2, 6), (0, 5, 1, 3, 6, 4))
    x475 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x475 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x474, (4, 5, 3, 6), (1, 4, 5, 0, 2, 6))
    del x474
    t3new += einsum(x475, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x475, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x475
    x476 = np.zeros((naocc, nocc, nocc, nocc), dtype=types[float])
    x476 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x477 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x477 += einsum(t1[np.ix_(so,sv)], (0, 1), x476, (2, 3, 0, 4), (2, 3, 4, 1))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x477, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * -1.0
    x478 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x478 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x477, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2))
    del x477
    t3new += einsum(x478, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x478, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x478
    x479 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x479 += einsum(t1[np.ix_(so,sv)], (0, 1), x476, (2, 3, 4, 0), (2, 4, 3, 1))
    del x476
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x479, (4, 1, 5, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x479, (4, 0, 5, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x479
    x480 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x480 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovvV, (2, 3, 1, 4), (0, 4, 2, 3))
    x481 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x481 += einsum(t1[np.ix_(so,sv)], (0, 1), x480, (2, 3, 4, 1), (2, 3, 0, 4))
    x482 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x482 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x481, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x481
    t3new += einsum(x482, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x482, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x482
    x483 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x483 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovvV, (2, 1, 3, 4), (0, 4, 2, 3))
    x484 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x484 += einsum(t1[np.ix_(so,sv)], (0, 1), x483, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x484, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5))
    del x484
    x485 = np.zeros((naocc, nocc, nvir, nvir), dtype=types[float])
    x485 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x486 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x486 += einsum(t1[np.ix_(so,sv)], (0, 1), x485, (2, 3, 1, 4), (2, 0, 3, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x486, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x486, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3))
    del x486
    x487 = np.zeros((naocc, nocc, nvir, nvir), dtype=types[float])
    x487 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x488 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x488 += einsum(t1[np.ix_(so,sv)], (0, 1), x487, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x488, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3))
    x489 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x489 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x488, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6))
    del x488
    t3new += einsum(x489, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    t3new += einsum(x489, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    del x489
    x490 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x490 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x483, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x483
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x490, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x490
    x491 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x491 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x480, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x480
    x492 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x492 += einsum(t1[np.ix_(so,sv)], (0, 1), x491, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x491
    t3new += einsum(x492, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x492, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -1.0
    del x492
    x493 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x493 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x487, (4, 5, 6, 2), (4, 3, 1, 0, 5, 6))
    del x487
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x493, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3))
    del x493
    x494 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x494 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x485, (4, 5, 2, 6), (4, 3, 1, 0, 5, 6))
    del x485
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x494, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x494
    x495 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x495 += einsum(t1[np.ix_(so,sV)], (0, 1), t1[np.ix_(sO,sv)], (2, 3), v.ooov, (4, 5, 0, 3), (2, 1, 4, 5))
    x496 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x496 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x495, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3))
    del x495
    t3new += einsum(x496, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x496, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x496
    x497 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x497 += einsum(t1[np.ix_(so,sV)], (0, 1), t1[np.ix_(sO,sv)], (2, 3), v.ooov, (4, 0, 5, 3), (2, 1, 4, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x497, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    del x497
    x498 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x498 += einsum(t1[np.ix_(so,sV)], (0, 1), t1[np.ix_(sO,sv)], (2, 3), v.ovvv, (0, 3, 4, 5), (2, 1, 4, 5))
    x499 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x499 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x498, (4, 5, 3, 6), (4, 5, 0, 1, 2, 6))
    del x498
    t3new += einsum(x499, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x499, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x499
    x500 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x500 += einsum(t1[np.ix_(so,sV)], (0, 1), t1[np.ix_(sO,sv)], (2, 3), v.ovvv, (0, 4, 5, 3), (2, 1, 4, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x500, (4, 5, 2, 6), (0, 1, 4, 6, 3, 5))
    del x500
    x501 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x501 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x41, (4, 5, 6, 0), (1, 2, 4, 6, 5, 3))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x501, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x501
    x502 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x502 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x41, (4, 5, 0, 6), (1, 2, 4, 5, 6, 3))
    del x41
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x502, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x502
    x503 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x503 += einsum(t1[np.ix_(so,sv)], (0, 1), x318, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x318
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x503, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x503
    x504 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x504 += einsum(t1[np.ix_(so,sv)], (0, 1), x322, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x322
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x504, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x504
    x505 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x505 += einsum(t1[np.ix_(so,sv)], (0, 1), x115, (2, 3, 4, 5, 1, 6), (2, 3, 0, 4, 5, 6))
    del x115
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x505, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3))
    del x505
    x506 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x506 += einsum(x9, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (1, 4, 5, 3, 6, 7), (5, 7, 0, 4, 2, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x506, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x506
    x507 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x507 += einsum(x9, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 5, 3, 6, 7), (5, 7, 0, 4, 1, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x507, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3))
    del x507
    x508 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x508 += einsum(x9, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 4, 5, 6, 3, 7), (5, 7, 0, 4, 1, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x508, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x508
    x509 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x509 += einsum(x9, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 3, 6, 7), (5, 7, 0, 4, 1, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x509, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x509
    x510 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x510 += einsum(x9, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 3, 6, 7), (5, 7, 0, 4, 2, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x510, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3))
    del x510
    x511 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x511 += einsum(x9, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 1, 5, 6, 3, 7), (5, 7, 0, 4, 2, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x511, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x511
    x512 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x512 += einsum(x9, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 2, 5, 6, 3, 7), (5, 7, 0, 4, 1, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x512, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3))
    del x512
    x513 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x513 += einsum(t1[np.ix_(so,sv)], (0, 1), x334, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x334
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x513, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3))
    del x513
    x514 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x514 += einsum(x0, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 1, 5, 6), (4, 6, 2, 3, 0, 5))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x514, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x514
    x515 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x515 += einsum(x15, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 1, 5, 6), (4, 6, 2, 3, 0, 5))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x515, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3))
    del x515
    x516 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x516 += einsum(x0, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x516, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x516
    x517 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x517 += einsum(x15, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x517, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x517
    x518 = np.zeros((navir, navir, nocc, nvir), dtype=types[float])
    x518 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovoV, (0, 2, 3, 4), (1, 4, 3, 2))
    x519 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x519 += einsum(t1[np.ix_(so,sv)], (0, 1), x518, (2, 3, 4, 1), (2, 3, 0, 4))
    del x518
    t3new += einsum(x519, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * -1.0
    del x519
    x520 = np.zeros((navir, navir, nocc, nvir), dtype=types[float])
    x520 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovoV, (2, 3, 0, 4), (1, 4, 2, 3))
    x521 = np.zeros((navir, navir, nocc, nocc), dtype=types[float])
    x521 += einsum(t1[np.ix_(so,sv)], (0, 1), x520, (2, 3, 4, 1), (2, 3, 0, 4))
    del x520
    t3new += einsum(x521, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0))
    t3new += einsum(x521, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0))
    del x521
    x522 = np.zeros((navir, navir, nocc, nvir), dtype=types[float])
    x522 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovoV, (0, 2, 3, 4), (1, 4, 3, 2))
    x523 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x523 += einsum(x522, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 3, 7, 1), (6, 0, 4, 5, 2, 7))
    del x522
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x523, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -0.5
    del x523
    x524 = np.zeros((navir, navir, nocc, nvir), dtype=types[float])
    x524 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovoV, (2, 3, 0, 4), (1, 4, 2, 3))
    x525 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x525 += einsum(x524, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 3, 7, 1), (6, 0, 4, 5, 2, 7))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x525, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * 0.5
    del x525
    x526 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x526 += einsum(x524, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 6, 7, 3, 1), (6, 0, 4, 5, 2, 7))
    del x524
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x526, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x526
    x527 = np.zeros((navir, nocc), dtype=types[float])
    x527 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovoV, (0, 1, 2, 3), (3, 2))
    x528 = np.zeros((navir, navir), dtype=types[float])
    x528 += einsum(t1[np.ix_(so,sV)], (0, 1), x527, (2, 0), (1, 2))
    del x527
    t3new += einsum(x528, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -2.0
    del x528
    x529 = np.zeros((navir, nocc), dtype=types[float])
    x529 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovoV, (2, 1, 0, 3), (3, 2))
    x530 = np.zeros((navir, navir), dtype=types[float])
    x530 += einsum(t1[np.ix_(so,sV)], (0, 1), x529, (2, 0), (1, 2))
    del x529
    t3new += einsum(x530, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0))
    del x530
    x531 = np.zeros((naocc, naocc, nocc, nvir), dtype=types[float])
    x531 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovvO, (2, 3, 1, 4), (0, 4, 2, 3))
    x532 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x532 += einsum(t1[np.ix_(so,sv)], (0, 1), x531, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new += einsum(x532, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * 0.5
    del x532
    x533 = np.zeros((naocc, naocc, nocc, nvir), dtype=types[float])
    x533 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovvO, (2, 1, 3, 4), (0, 4, 2, 3))
    x534 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x534 += einsum(t1[np.ix_(so,sv)], (0, 1), x533, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new += einsum(x534, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * -0.5
    del x534
    x535 = np.zeros((naocc, naocc, nocc, nvir), dtype=types[float])
    x535 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovvO, (2, 3, 1, 4), (0, 4, 2, 3))
    x536 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x536 += einsum(t1[np.ix_(so,sv)], (0, 1), x535, (2, 3, 4, 1), (2, 3, 0, 4))
    del x535
    t3new += einsum(x536, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7))
    del x536
    x537 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x537 += einsum(x533, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 3, 6, 7), (0, 7, 4, 5, 2, 6))
    del x533
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x537, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x537
    x538 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x538 += einsum(x531, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 3, 6, 7), (0, 7, 4, 5, 2, 6))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x538, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3))
    del x538
    x539 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x539 += einsum(x531, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    del x531
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x539, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x539
    x540 = np.zeros((naocc, nvir), dtype=types[float])
    x540 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvO, (0, 1, 2, 3), (3, 2))
    x541 = np.zeros((naocc, naocc), dtype=types[float])
    x541 += einsum(t1[np.ix_(sO,sv)], (0, 1), x540, (2, 1), (0, 2))
    del x540
    t3new += einsum(x541, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    del x541
    x542 = np.zeros((naocc, nvir), dtype=types[float])
    x542 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvO, (0, 2, 1, 3), (3, 2))
    x543 = np.zeros((naocc, naocc), dtype=types[float])
    x543 += einsum(t1[np.ix_(sO,sv)], (0, 1), x542, (2, 1), (0, 2))
    del x542
    t3new += einsum(x543, (0, 1), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6))
    del x543
    x544 = np.zeros((naocc, naocc, navir, nocc), dtype=types[float])
    x544 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovOV, (2, 1, 3, 4), (0, 3, 4, 2))
    x545 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x545 += einsum(t1[np.ix_(so,sV)], (0, 1), x544, (2, 3, 4, 0), (2, 3, 1, 4))
    x546 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x546 += einsum(x545, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (0, 2, 4, 5, 6, 7))
    del x545
    t3new += einsum(x546, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new += einsum(x546, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x546
    x547 = np.zeros((naocc, naocc, navir, nocc), dtype=types[float])
    x547 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.oVvO, (2, 3, 1, 4), (0, 4, 3, 2))
    x548 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x548 += einsum(t1[np.ix_(so,sV)], (0, 1), x547, (2, 3, 4, 0), (2, 3, 1, 4))
    del x547
    t3new += einsum(x548, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sv,sVf)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    del x548
    x549 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x549 += einsum(t1[np.ix_(so,sv)], (0, 1), x155, (2, 3, 4, 0), (2, 3, 4, 1))
    del x155
    t3new += einsum(x549, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 2, 5, 6, 7, 0), (4, 1, 5, 6, 3, 7))
    del x549
    x550 = np.zeros((naocc, naocc, navir, nvir), dtype=types[float])
    x550 += einsum(t1[np.ix_(so,sv)], (0, 1), x544, (2, 3, 4, 0), (2, 3, 4, 1))
    del x544
    t3new += einsum(x550, (0, 1, 2, 3), t3[np.ix_(so,so,sOf,sv,sV,sVf)], (4, 5, 1, 6, 7, 2), (5, 4, 0, 3, 6, 7))
    del x550
    x551 = np.zeros((naocc, navir, navir, nvir), dtype=types[float])
    x551 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ovOV, (0, 2, 3, 4), (3, 1, 4, 2))
    x552 = np.zeros((naocc, navir, navir, nocc), dtype=types[float])
    x552 += einsum(t1[np.ix_(so,sv)], (0, 1), x551, (2, 3, 4, 1), (2, 3, 4, 0))
    del x551
    t3new += einsum(x552, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sv,sVf)], (4, 5, 0, 6, 7, 2), (3, 4, 5, 7, 6, 1))
    del x552
    x553 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x553 += einsum(t1[np.ix_(so,sv)], (0, 1), x62, (2, 3, 4, 0), (2, 3, 4, 1))
    del x62
    t3new += einsum(x553, (0, 1, 2, 3), t3[np.ix_(so,sO,sOf,sv,sV,sVf)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * -1.0
    del x553
    x554 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x554 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x97, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x97
    t3new += einsum(x554, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x554, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x554
    x555 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x555 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovov, (4, 5, 0, 2), (1, 3, 4, 5))
    x556 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x556 += einsum(t1[np.ix_(so,sv)], (0, 1), x555, (2, 3, 4, 1), (2, 3, 0, 4))
    x557 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x557 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x556, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x556
    t3new += einsum(x557, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new += einsum(x557, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x557
    x558 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x558 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x99, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x99
    t3new += einsum(x558, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x558, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x558
    x559 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x559 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x93, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x93
    t3new += einsum(x559, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new += einsum(x559, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x559
    x560 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x560 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovov, (4, 2, 0, 5), (1, 3, 4, 5))
    x561 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x561 += einsum(t1[np.ix_(so,sv)], (0, 1), x560, (2, 3, 4, 1), (2, 3, 0, 4))
    x562 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x562 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x561, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x561
    t3new += einsum(x562, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x562, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x562
    x563 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x563 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x95, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x95
    t3new += einsum(x563, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x563, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x563
    x564 = np.zeros((naocc, navir, nocc, nocc, nocc, nocc), dtype=types[float])
    x564 += einsum(t1[np.ix_(so,sv)], (0, 1), x157, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    x565 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x565 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x564, (4, 5, 6, 7, 0, 1), (4, 5, 6, 7, 2, 3))
    t3new += einsum(x565, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x565, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x565
    x566 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x566 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x9, (4, 5, 0, 3), (1, 4, 5, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x566, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * -1.0
    x567 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x567 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x566, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    del x566
    t3new += einsum(x567, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x567, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x567
    x568 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x568 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x9, (4, 5, 0, 2), (1, 4, 5, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x568, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x568, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x568
    x569 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x569 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x9, (4, 0, 5, 3), (1, 4, 5, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x569, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x569, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x569
    x570 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x570 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x9, (4, 0, 5, 2), (1, 4, 5, 3))
    x571 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x571 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x570, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    t3new += einsum(x571, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x571, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x571
    x572 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x572 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (3, 0, 4, 5))
    x573 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x573 += einsum(t1[np.ix_(so,sv)], (0, 1), x572, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x573, (4, 5, 6, 0), (5, 6, 1, 3, 2, 4)) * -1.0
    del x573
    x574 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x574 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x9, (4, 1, 0, 5), (3, 4, 2, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x574, (4, 5, 6, 2), (0, 5, 1, 3, 6, 4)) * -1.0
    x575 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x575 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x574, (4, 5, 6, 3), (1, 4, 5, 0, 2, 6))
    del x574
    t3new += einsum(x575, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x575, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 0.5
    t3new += einsum(x575, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x575
    x576 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x576 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x9, (4, 0, 1, 5), (3, 4, 2, 5))
    del x9
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x576, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4)) * -1.0
    x577 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x577 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x576, (4, 5, 6, 3), (1, 4, 5, 0, 2, 6))
    del x576
    t3new += einsum(x577, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    t3new += einsum(x577, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -0.5
    del x577
    x578 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x578 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (3, 0, 4, 5))
    x579 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x579 += einsum(t1[np.ix_(so,sv)], (0, 1), x578, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x579, (4, 5, 6, 0), (6, 5, 1, 3, 2, 4)) * -1.0
    del x579
    x580 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x580 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x570, (4, 5, 0, 6), (4, 3, 5, 1, 6, 2))
    del x570
    t3new += einsum(x580, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x580, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x580
    x581 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x581 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovov, (4, 2, 0, 5), (3, 1, 4, 5))
    x582 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x582 += einsum(t1[np.ix_(so,sv)], (0, 1), x581, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x582, (4, 5, 6, 0), (6, 5, 1, 3, 2, 4)) * -1.0
    del x582
    x583 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x583 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovov, (4, 5, 0, 2), (3, 1, 4, 5))
    x584 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x584 += einsum(t1[np.ix_(so,sv)], (0, 1), x583, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x584, (4, 5, 6, 0), (6, 5, 1, 3, 2, 4)) * 2.0
    del x584
    x585 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x585 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x157, (4, 5, 6, 7, 1, 2), (4, 5, 6, 0, 7, 3))
    x586 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x586 += einsum(t1[np.ix_(so,sv)], (0, 1), x585, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x585
    t3new += einsum(x586, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x586, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x586
    x587 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x587 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x87, (4, 5, 6, 0), (1, 3, 4, 5, 6, 2))
    x588 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x588 += einsum(t1[np.ix_(so,sv)], (0, 1), x587, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x587
    t3new += einsum(x588, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x588, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x588
    x589 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x589 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x555, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x555
    x590 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x590 += einsum(t1[np.ix_(so,sv)], (0, 1), x589, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x589
    t3new += einsum(x590, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new += einsum(x590, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x590
    x591 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x591 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x560, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x560
    x592 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x592 += einsum(t1[np.ix_(so,sv)], (0, 1), x591, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x591
    t3new += einsum(x592, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x592, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x592
    x593 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x593 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x157, (4, 5, 6, 7, 1, 3), (4, 5, 6, 0, 7, 2))
    x594 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x594 += einsum(t1[np.ix_(so,sv)], (0, 1), x593, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x593
    t3new += einsum(x594, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x594, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x594
    x595 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x595 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x157, (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 7, 2))
    x596 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x596 += einsum(t1[np.ix_(so,sv)], (0, 1), x595, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x595
    t3new += einsum(x596, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new += einsum(x596, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x596
    x597 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x597 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x157, (4, 5, 6, 1, 7, 2), (4, 5, 6, 0, 7, 3))
    del x157
    x598 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x598 += einsum(t1[np.ix_(so,sv)], (0, 1), x597, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x597
    t3new += einsum(x598, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x598, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x598
    x599 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x599 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x572, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2))
    x600 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x600 += einsum(t1[np.ix_(so,sv)], (0, 1), x599, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x599
    t3new += einsum(x600, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x600, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x600
    x601 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x601 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x578, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x601, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x601
    x602 = np.zeros((naocc, nocc, nocc, nocc), dtype=types[float])
    x602 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 3, 5, 2), (1, 0, 4, 5))
    x603 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x603 += einsum(t1[np.ix_(so,sv)], (0, 1), x602, (2, 3, 4, 0), (2, 3, 4, 1))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x603, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * -1.0
    x604 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x604 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x603, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    del x603
    t3new += einsum(x604, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x604, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x604
    x605 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x605 += einsum(t1[np.ix_(so,sv)], (0, 1), x602, (2, 3, 0, 4), (2, 3, 4, 1))
    del x602
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x605, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x605, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x605
    x606 = np.zeros((naocc, nocc, nvir, nvir), dtype=types[float])
    x606 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 3, 0, 5), (1, 4, 2, 5))
    x607 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x607 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x606, (4, 5, 6, 2), (4, 3, 1, 0, 5, 6))
    del x606
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x607, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x607
    x608 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x608 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x581, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x608, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x608
    x609 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x609 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x583, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x609, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * 2.0
    del x609
    x610 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x610 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x578, (4, 5, 6, 2), (1, 4, 0, 5, 6, 3))
    del x578
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x610, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x610
    x611 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x611 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x572, (4, 5, 6, 2), (1, 4, 0, 5, 6, 3))
    del x572
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x611, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x611
    x612 = np.zeros((naocc, nocc, nvir, nvir), dtype=types[float])
    x612 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 5, 0, 2), (1, 4, 3, 5))
    x613 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x613 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x612, (4, 5, 6, 2), (4, 3, 1, 0, 5, 6))
    del x612
    x614 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x614 += einsum(t1[np.ix_(so,sv)], (0, 1), x613, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x613
    t3new += einsum(x614, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x614, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x614
    x615 = np.zeros((naocc, nocc, nvir, nvir), dtype=types[float])
    x615 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 2, 0, 5), (1, 4, 3, 5))
    x616 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x616 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x615, (4, 5, 6, 2), (4, 3, 1, 0, 5, 6))
    del x615
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x616, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x616
    x617 = np.zeros((naocc, nocc, nvir, nvir), dtype=types[float])
    x617 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 5, 0, 3), (1, 4, 2, 5))
    x618 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x618 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x617, (4, 5, 6, 2), (4, 3, 1, 0, 5, 6))
    del x617
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x618, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x618
    x619 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x619 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x581, (4, 5, 6, 2), (1, 4, 0, 5, 6, 3))
    del x581
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x619, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x619
    x620 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x620 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x583, (4, 5, 6, 2), (1, 4, 0, 5, 6, 3))
    del x583
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x620, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x620
    x621 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x621 += einsum(x0, (0, 1), t2[np.ix_(so,sO,sv,sV)], (2, 3, 1, 4), (3, 4, 2, 0))
    x622 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x622 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x621, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x621
    t3new += einsum(x622, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new += einsum(x622, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x622
    x623 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x623 += einsum(x15, (0, 1), t2[np.ix_(so,sO,sv,sV)], (2, 3, 1, 4), (3, 4, 2, 0))
    x624 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x624 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x623, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x623
    t3new += einsum(x624, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x624, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x624
    x625 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x625 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x104, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x104
    t3new += einsum(x625, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new += einsum(x625, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x625
    x626 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x626 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x106, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x106
    t3new += einsum(x626, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x626, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x626
    x627 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x627 += einsum(x0, (0, 1), t2[np.ix_(so,sO,sv,sv)], (2, 3, 4, 1), (3, 2, 0, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x627, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * 2.0
    x628 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x628 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x627, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    del x627
    t3new += einsum(x628, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new += einsum(x628, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x628
    x629 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x629 += einsum(x15, (0, 1), t2[np.ix_(so,sO,sv,sv)], (2, 3, 4, 1), (3, 2, 0, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x629, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * -1.0
    x630 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x630 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x629, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    del x629
    t3new += einsum(x630, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x630, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x630
    x631 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x631 += einsum(x0, (0, 1), t2[np.ix_(so,sO,sv,sv)], (2, 3, 1, 4), (3, 2, 0, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x631, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * 2.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x631, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * 2.0
    del x631
    x632 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x632 += einsum(x15, (0, 1), t2[np.ix_(so,sO,sv,sv)], (2, 3, 1, 4), (3, 2, 0, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x632, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x632, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x632
    x633 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x633 += einsum(x0, (0, 1), t2[np.ix_(so,so,sv,sV)], (2, 3, 1, 4), (4, 3, 2, 0))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x633, (4, 5, 6, 0), (5, 6, 1, 3, 2, 4)) * 2.0
    del x633
    x634 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x634 += einsum(x15, (0, 1), t2[np.ix_(so,so,sv,sV)], (2, 3, 1, 4), (4, 3, 2, 0))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x634, (4, 5, 6, 0), (5, 6, 1, 3, 2, 4)) * -1.0
    del x634
    x635 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x635 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x168, (4, 1, 5, 3), (4, 0, 2, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x635, (4, 5, 6, 2), (0, 5, 1, 3, 6, 4)) * -1.0
    x636 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x636 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x635, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6))
    del x635
    t3new += einsum(x636, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x636, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x636
    x637 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x637 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x168, (4, 5, 2, 3), (1, 4, 0, 5))
    x638 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x638 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x637, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x637
    t3new += einsum(x638, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x638, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x638
    x639 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x639 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x168, (4, 1, 5, 2), (4, 0, 3, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x639, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -1.0
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x639, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4)) * -1.0
    del x639
    x640 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x640 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x168, (4, 1, 3, 5), (4, 0, 2, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x640, (4, 5, 6, 2), (0, 5, 1, 3, 6, 4)) * 2.0
    x641 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x641 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x640, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6))
    del x640
    t3new += einsum(x641, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new += einsum(x641, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x641
    x642 = np.zeros((navir, nocc, nvir, nvir), dtype=types[float])
    x642 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x168, (4, 1, 2, 5), (4, 0, 3, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x642, (4, 5, 6, 2), (0, 5, 1, 3, 6, 4)) * -1.0
    x643 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x643 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x642, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6))
    del x642
    t3new += einsum(x643, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x643, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x643
    x644 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x644 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x168, (4, 0, 3, 5), (1, 4, 2, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x644, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * -1.0
    del x644
    x645 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x645 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x168, (4, 5, 2, 3), (4, 0, 1, 5))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x645, (4, 5, 6, 0), (6, 5, 1, 3, 2, 4)) * -1.0
    del x645
    x646 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x646 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x168, (4, 0, 2, 5), (1, 4, 3, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x646, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * 2.0
    del x646
    x647 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x647 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x168, (4, 0, 5, 3), (1, 4, 2, 5))
    x648 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x648 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x647, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x647
    t3new += einsum(x648, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x648, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x648
    x649 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x649 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x168, (4, 0, 5, 2), (1, 4, 3, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x649, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * -1.0
    del x649
    x650 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x650 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x168, (4, 5, 3, 2), (1, 4, 0, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x650, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    del x650
    x651 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x651 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x179, (4, 5, 1, 3), (4, 0, 5, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x651, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * -1.0
    x652 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x652 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x651, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6))
    del x651
    t3new += einsum(x652, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x652, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x652
    x653 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x653 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x179, (4, 5, 1, 2), (4, 0, 5, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x653, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x653, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x653
    x654 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x654 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x179, (4, 1, 5, 3), (4, 0, 5, 2))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x654, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * 2.0
    x655 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x655 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x654, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6))
    del x654
    t3new += einsum(x655, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new += einsum(x655, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x655
    x656 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x656 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x179, (4, 5, 1, 2), (4, 3, 0, 5))
    x657 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x657 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x656, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x656
    t3new += einsum(x657, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x657, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x657
    x658 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x658 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x179, (4, 1, 5, 2), (4, 0, 5, 3))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x658, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * -1.0
    x659 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x659 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x658, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6))
    del x658
    t3new += einsum(x659, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new += einsum(x659, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x659
    x660 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x660 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x179, (4, 1, 0, 5), (4, 3, 2, 5))
    x661 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x661 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x660, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x660
    t3new += einsum(x661, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -0.5
    t3new += einsum(x661, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -0.5
    del x661
    x662 = np.zeros((naocc, navir, nvir, nvir), dtype=types[float])
    x662 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x179, (4, 0, 1, 5), (4, 3, 2, 5))
    x663 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x663 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x662, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x662
    t3new += einsum(x663, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 0.5
    t3new += einsum(x663, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x663, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 0.5
    del x663
    x664 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x664 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x179, (4, 1, 5, 2), (4, 3, 0, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x664, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    del x664
    x665 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x665 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x179, (4, 0, 5, 2), (4, 3, 1, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x665, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * 2.0
    del x665
    x666 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x666 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x179, (4, 5, 0, 2), (4, 3, 1, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x666, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    del x666
    x667 = np.zeros((naocc, nvir, nvir, nvir), dtype=types[float])
    x667 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x179, (4, 1, 0, 5), (4, 3, 2, 5))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x667, (4, 5, 6, 2), (1, 0, 4, 6, 5, 3)) * -1.0
    del x667
    x668 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x668 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovov, (4, 5, 0, 3), (1, 2, 4, 5))
    x669 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x669 += einsum(t1[np.ix_(so,sv)], (0, 1), x668, (2, 3, 4, 1), (2, 3, 0, 4))
    x670 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x670 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x669, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x669
    t3new += einsum(x670, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x670, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x670
    x671 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x671 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovov, (4, 3, 0, 5), (1, 2, 4, 5))
    x672 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x672 += einsum(t1[np.ix_(so,sv)], (0, 1), x671, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x672, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    del x672
    x673 = np.zeros((naocc, navir, nocc, nocc, nocc, nocc), dtype=types[float])
    x673 += einsum(t1[np.ix_(so,sv)], (0, 1), x187, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x673, (4, 5, 6, 7, 0, 1), (7, 6, 4, 3, 2, 5)) * -1.0
    x674 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x674 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x668, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x668
    x675 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x675 += einsum(t1[np.ix_(so,sv)], (0, 1), x674, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x674
    t3new += einsum(x675, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x675, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x675
    x676 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x676 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x671, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x671
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x676, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x676
    x677 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x677 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x187, (4, 5, 6, 7, 1, 3), (4, 5, 6, 0, 7, 2))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x677, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x677
    x678 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x678 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x187, (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 7, 2))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x678, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x678
    x679 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x679 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x187, (4, 5, 6, 1, 7, 2), (4, 5, 6, 0, 7, 3))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x679, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x679
    x680 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x680 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x87, (4, 5, 6, 0), (1, 2, 4, 5, 6, 3))
    del x87
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x680, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x680
    x681 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x681 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x187, (4, 5, 6, 7, 1, 2), (4, 5, 6, 0, 7, 3))
    del x187
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x681, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x681
    x682 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x682 += einsum(x0, (0, 1), t2[np.ix_(so,sO,sV,sv)], (2, 3, 4, 1), (3, 4, 2, 0))
    del x0
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x682, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * 2.0
    del x682
    x683 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x683 += einsum(x15, (0, 1), t2[np.ix_(so,sO,sV,sv)], (2, 3, 4, 1), (3, 4, 2, 0))
    del x15
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x683, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    del x683
    x684 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x684 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), x92, (4, 5, 0, 6), (1, 3, 5, 4, 6, 2))
    x685 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x685 += einsum(t1[np.ix_(so,sv)], (0, 1), x684, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x684
    t3new += einsum(x685, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x685, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x685
    x686 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x686 += einsum(t1[np.ix_(so,sv)], (0, 1), x564, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x564
    x687 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x687 += einsum(t1[np.ix_(so,sv)], (0, 1), x686, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x686
    t3new += einsum(x687, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x687, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x687
    x688 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x688 += einsum(t1[np.ix_(so,sv)], (0, 1), x168, (2, 3, 1, 4), (2, 0, 3, 4))
    x689 = np.zeros((navir, nocc, nocc, nocc), dtype=types[float])
    x689 += einsum(t1[np.ix_(so,sv)], (0, 1), x688, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x689, (4, 5, 6, 0), (6, 5, 1, 3, 2, 4)) * -1.0
    del x689
    x690 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x690 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x688, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2))
    x691 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x691 += einsum(t1[np.ix_(so,sv)], (0, 1), x690, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x690
    t3new += einsum(x691, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x691, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x691
    x692 = np.zeros((navir, nocc, nocc, nvir), dtype=types[float])
    x692 += einsum(t1[np.ix_(so,sv)], (0, 1), x168, (2, 3, 4, 1), (2, 0, 3, 4))
    del x168
    x693 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x693 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x692, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2))
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x693, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x693
    x694 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x694 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x692, (4, 5, 6, 2), (1, 4, 5, 0, 6, 3))
    del x692
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x694, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x694
    x695 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x695 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), x688, (4, 5, 6, 2), (1, 4, 5, 0, 6, 3))
    del x688
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x695, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x695
    x696 = np.zeros((naocc, nocc, nocc, nocc), dtype=types[float])
    x696 += einsum(t1[np.ix_(so,sv)], (0, 1), x179, (2, 3, 4, 1), (2, 0, 4, 3))
    x697 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x697 += einsum(t1[np.ix_(so,sv)], (0, 1), x696, (2, 3, 4, 0), (2, 3, 4, 1))
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x697, (4, 5, 0, 6), (1, 5, 4, 2, 6, 3)) * -1.0
    x698 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x698 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x697, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    del x697
    t3new += einsum(x698, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new += einsum(x698, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x698
    x699 = np.zeros((naocc, nocc, nocc, nvir), dtype=types[float])
    x699 += einsum(t1[np.ix_(so,sv)], (0, 1), x696, (2, 3, 0, 4), (2, 3, 4, 1))
    del x696
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x699, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    t3new += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x699, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x699
    x700 = np.zeros((naocc, navir, nocc, nocc, nocc, nocc), dtype=types[float])
    x700 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), x179, (4, 5, 6, 2), (4, 3, 1, 0, 6, 5))
    del x179
    x701 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x701 += einsum(t1[np.ix_(so,sv)], (0, 1), x700, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x700
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x701, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x701
    x702 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x702 += einsum(t1[np.ix_(so,sV)], (0, 1), t1[np.ix_(sO,sv)], (2, 3), v.ovov, (4, 3, 0, 5), (2, 1, 4, 5))
    x703 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x703 += einsum(t1[np.ix_(so,sv)], (0, 1), x702, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x703, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    del x703
    x704 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x704 += einsum(t1[np.ix_(so,sV)], (0, 1), t1[np.ix_(sO,sv)], (2, 3), v.ovov, (4, 5, 0, 3), (2, 1, 4, 5))
    x705 = np.zeros((naocc, navir, nocc, nocc), dtype=types[float])
    x705 += einsum(t1[np.ix_(so,sv)], (0, 1), x704, (2, 3, 4, 1), (2, 3, 0, 4))
    x706 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x706 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x705, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x705
    t3new += einsum(x706, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    t3new += einsum(x706, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    del x706
    x707 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x707 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x702, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x702
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x707, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x707
    x708 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x708 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x704, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x704
    x709 = np.zeros((naocc, navir, nocc, nocc, nvir, nvir), dtype=types[float])
    x709 += einsum(t1[np.ix_(so,sv)], (0, 1), x708, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x708
    t3new += einsum(x709, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1))
    t3new += einsum(x709, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1))
    del x709
    x710 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x710 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), x92, (4, 5, 0, 6), (1, 2, 5, 4, 6, 3))
    del x92
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x710, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x710
    x711 = np.zeros((naocc, navir, nocc, nocc, nocc, nvir), dtype=types[float])
    x711 += einsum(t1[np.ix_(so,sv)], (0, 1), x673, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x673
    t3new += einsum(t1[np.ix_(so,sv)], (0, 1), x711, (2, 3, 4, 5, 0, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x711

    return {"t1new": t1new, "t2new": t2new, "t3new": t3new}

