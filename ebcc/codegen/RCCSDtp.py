# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), x0, (0, 1, 2, 3), ()) * 2.0
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x1 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x1 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x2 = np.zeros((nocc, nvir), dtype=np.float64)
    x2 += einsum(f.ov, (0, 1), (0, 1))
    x2 += einsum(t1, (0, 1), x1, (0, 2, 3, 1), (2, 3))
    del x1
    e_cc += einsum(t1, (0, 1), x2, (0, 1), ()) * 2.0
    del x2

    return e_cc

def update_amps(f=None, v=None, space=None, t1=None, t2=None, t3=None, **kwargs):
    nocc = space.nocc
    nvir = space.nvir
    naocc = space.naocc
    navir = space.navir
    so = np.ones((nocc,), dtype=bool)
    sv = np.ones((nvir,), dtype=bool)
    sO = space.active[space.occupied]
    sV = space.active[space.virtual]

    # T amplitudes
    t1new = np.zeros(((nocc, nvir)), dtype=np.float64)
    t1new[np.ix_(so,sv)] += einsum(f.ov, (0, 1), (0, 1))
    t1new[np.ix_(so,sv)] += einsum(f.oo, (0, 1), t1[np.ix_(so,sv)], (1, 2), (0, 2)) * -1.0
    t1new[np.ix_(so,sv)] += einsum(f.vv, (0, 1), t1[np.ix_(so,sv)], (2, 1), (2, 0))
    t1new[np.ix_(so,sv)] += einsum(f.ov, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 1), (2, 3)) * 2.0
    t1new[np.ix_(so,sv)] += einsum(f.ov, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 1, 3), (2, 3)) * -1.0
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), v.oovv, (2, 0, 3, 1), (2, 3)) * -1.0
    t1new[np.ix_(so,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvv, (1, 2, 4, 3), (0, 4)) * -1.0
    t1new[np.ix_(sO,sV)] += einsum(v.OVOV, (0, 1, 2, 3), t3, (4, 2, 0, 5, 1, 3), (4, 5)) * -0.5
    t1new[np.ix_(sO,sV)] += einsum(v.OVOV, (0, 1, 2, 3), t3, (4, 2, 0, 3, 5, 1), (4, 5)) * -0.25
    t1new[np.ix_(sO,sV)] += einsum(v.OVOV, (0, 1, 2, 3), t3, (4, 2, 0, 1, 5, 3), (4, 5)) * 0.25
    t1new[np.ix_(sO,sV)] += einsum(v.OVOV, (0, 1, 2, 3), t3, (0, 4, 2, 1, 5, 3), (4, 5)) * 0.25
    t1new[np.ix_(sO,sV)] += einsum(v.OVOV, (0, 1, 2, 3), t3, (0, 4, 2, 3, 5, 1), (4, 5)) * -0.25
    t2new = np.zeros(((nocc, nocc, nvir, nvir)), dtype=np.float64)
    t2new[np.ix_(so,so,sv,sv)] += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oooo, (4, 1, 5, 0), (4, 5, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t3new = np.zeros(((naocc, naocc, naocc, navir, navir, navir)), dtype=np.float64)
    t3new += einsum(f.OO, (0, 1), t3, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -1.0
    t3new += einsum(f.VV, (0, 1), t3, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6))
    t3new += einsum(v.OOVV, (0, 1, 2, 3), t3, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -1.0
    t3new += einsum(v.OVOV, (0, 1, 2, 3), t3, (4, 5, 2, 6, 3, 7), (4, 0, 5, 6, 1, 7)) * -1.0
    t3new += einsum(v.OVOV, (0, 1, 2, 3), t3, (4, 2, 5, 6, 3, 7), (4, 0, 5, 6, 1, 7))
    x0 = np.zeros((nocc, nvir), dtype=np.float64)
    x0 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovov, (2, 3, 0, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(x0, (0, 1), (0, 1)) * 2.0
    t1new[np.ix_(so,sv)] += einsum(x0, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 1), (2, 3)) * 4.0
    t1new[np.ix_(so,sv)] += einsum(x0, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 1, 3), (2, 3)) * -2.0
    x1 = np.zeros((nocc, nvir), dtype=np.float64)
    x1 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 0, 1, 3), (4, 2))
    t1new[np.ix_(so,sv)] += einsum(x1, (0, 1), (0, 1)) * -1.5
    t1new[np.ix_(so,sv)] += einsum(x1, (0, 1), (0, 1)) * -0.5
    del x1
    x2 = np.zeros((nocc, nvir), dtype=np.float64)
    x2 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 1, 0, 3), (4, 2))
    t1new[np.ix_(so,sv)] += einsum(x2, (0, 1), (0, 1)) * 0.5
    t1new[np.ix_(so,sv)] += einsum(x2, (0, 1), (0, 1)) * 0.5
    del x2
    x3 = np.zeros((nocc, nvir), dtype=np.float64)
    x3 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvv, (1, 3, 4, 2), (0, 4))
    t1new[np.ix_(so,sv)] += einsum(x3, (0, 1), (0, 1))
    t1new[np.ix_(so,sv)] += einsum(x3, (0, 1), (0, 1))
    del x3
    x4 = np.zeros((naocc, navir), dtype=np.float64)
    x4 += einsum(v.OVOV, (0, 1, 2, 3), t3, (4, 2, 0, 5, 3, 1), (4, 5))
    t1new[np.ix_(sO,sV)] += einsum(x4, (0, 1), (0, 1)) * 0.5
    t1new[np.ix_(sO,sV)] += einsum(x4, (0, 1), (0, 1))
    del x4
    x5 = np.zeros((nocc, nocc), dtype=np.float64)
    x5 += einsum(f.ov, (0, 1), t1[np.ix_(so,sv)], (2, 1), (0, 2))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x5, (0, 2), (2, 1)) * -1.0
    x6 = np.zeros((nocc, nocc), dtype=np.float64)
    x6 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ooov, (2, 0, 3, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x6, (2, 0), (2, 1))
    x7 = np.zeros((nocc, nocc), dtype=np.float64)
    x7 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ooov, (2, 3, 0, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x7, (2, 0), (2, 1)) * -2.0
    x8 = np.zeros((nvir, nvir), dtype=np.float64)
    x8 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvv, (0, 2, 3, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x8, (1, 2), (0, 2)) * -1.0
    x9 = np.zeros((nvir, nvir), dtype=np.float64)
    x9 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvv, (0, 1, 2, 3), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x9, (2, 1), (0, 2)) * 2.0
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x10 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x11 = np.zeros((nocc, nvir), dtype=np.float64)
    x11 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x10, (4, 0, 1, 2), (4, 3))
    t1new[np.ix_(so,sv)] += einsum(x11, (0, 1), (0, 1)) * -1.5
    t1new[np.ix_(so,sv)] += einsum(x11, (0, 1), (0, 1)) * -0.5
    del x11
    x12 = np.zeros((nocc, nvir), dtype=np.float64)
    x12 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x10, (4, 0, 1, 3), (4, 2))
    t1new[np.ix_(so,sv)] += einsum(x12, (0, 1), (0, 1)) * 0.5
    t1new[np.ix_(so,sv)] += einsum(x12, (0, 1), (0, 1)) * 0.5
    del x12
    x13 = np.zeros((nocc, nocc), dtype=np.float64)
    x13 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 2, 1, 3), (0, 4))
    x14 = np.zeros((nocc, nvir), dtype=np.float64)
    x14 += einsum(t1[np.ix_(so,sv)], (0, 1), x13, (2, 0), (2, 1))
    t1new[np.ix_(so,sv)] += einsum(x14, (0, 1), (0, 1)) * -1.0
    t1new[np.ix_(so,sv)] += einsum(x14, (0, 1), (0, 1)) * -1.0
    del x14
    x15 = np.zeros((nocc, nocc), dtype=np.float64)
    x15 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 3, 1, 2), (0, 4))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x15, (2, 0), (2, 1))
    x16 = np.zeros((nocc, nvir), dtype=np.float64)
    x16 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovov, (2, 1, 0, 3), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(x16, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 1), (2, 3)) * -2.0
    t1new[np.ix_(so,sv)] += einsum(x16, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 1, 3), (2, 3))
    x17 = np.zeros((nocc, nocc), dtype=np.float64)
    x17 += einsum(t1[np.ix_(so,sv)], (0, 1), x16, (2, 1), (0, 2))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x17, (2, 0), (2, 1))
    x18 = np.zeros((nocc, nocc), dtype=np.float64)
    x18 += einsum(t1[np.ix_(so,sv)], (0, 1), x0, (2, 1), (0, 2))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x18, (2, 0), (2, 1)) * -2.0
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x19 += einsum(f.oo, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x19, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x19, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x19
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x20 += einsum(f.vv, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x20, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x20, (0, 1, 2, 3), (0, 1, 3, 2))
    del x20
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x21 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x21, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x21
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x22 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x22, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x22, (0, 1, 2, 3), (1, 0, 2, 3))
    x23 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x23 += einsum(f.OV, (0, 1), t3, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    t2new[np.ix_(sO,sO,sV,sV)] += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(sO,sO,sV,sV)] += einsum(x23, (0, 1, 2, 3), (1, 0, 3, 2))
    del x23
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x24 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x24, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x24
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x25 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (0, 4, 3, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x25, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x25, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x25
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x26 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x26, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x26, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x26, (4, 1, 5, 3), (0, 4, 2, 5)) * 4.0
    del x26
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x27 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x27, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x27, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x27, (4, 1, 5, 2), (0, 4, 3, 5))
    x28 = np.zeros((naocc, navir, navir, nocc), dtype=np.float64)
    x28 += einsum(v.oOOV, (0, 1, 2, 3), t3, (4, 1, 2, 5, 6, 3), (4, 5, 6, 0))
    t2new[np.ix_(so,sO,sV,sV)] += einsum(x28, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    t2new[np.ix_(sO,so,sV,sV)] += einsum(x28, (0, 1, 2, 3), (0, 3, 1, 2)) * -1.0
    del x28
    x29 = np.zeros((naocc, navir, navir, nocc), dtype=np.float64)
    x29 += einsum(v.oOOV, (0, 1, 2, 3), t3, (2, 4, 1, 3, 5, 6), (4, 6, 5, 0))
    t2new[np.ix_(so,sO,sV,sV)] += einsum(x29, (0, 1, 2, 3), (3, 0, 1, 2)) * -0.5
    t2new[np.ix_(sO,so,sV,sV)] += einsum(x29, (0, 1, 2, 3), (0, 3, 2, 1)) * -0.5
    del x29
    x30 = np.zeros((naocc, navir, navir, nocc), dtype=np.float64)
    x30 += einsum(v.oOOV, (0, 1, 2, 3), t3, (2, 4, 1, 5, 6, 3), (4, 5, 6, 0))
    t2new[np.ix_(so,sO,sV,sV)] += einsum(x30, (0, 1, 2, 3), (3, 0, 1, 2)) * 0.5
    t2new[np.ix_(sO,so,sV,sV)] += einsum(x30, (0, 1, 2, 3), (0, 3, 2, 1)) * 0.5
    del x30
    x31 = np.zeros((naocc, naocc, navir, nvir), dtype=np.float64)
    x31 += einsum(v.vVOV, (0, 1, 2, 3), t3, (4, 5, 2, 3, 6, 1), (4, 5, 6, 0))
    t2new[np.ix_(sO,sO,sv,sV)] += einsum(x31, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    t2new[np.ix_(sO,sO,sV,sv)] += einsum(x31, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    del x31
    x32 = np.zeros((naocc, naocc, navir, nvir), dtype=np.float64)
    x32 += einsum(v.vVOV, (0, 1, 2, 3), t3, (4, 5, 2, 1, 6, 3), (4, 5, 6, 0))
    t2new[np.ix_(sO,sO,sv,sV)] += einsum(x32, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    t2new[np.ix_(sO,sO,sV,sv)] += einsum(x32, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    del x32
    x33 = np.zeros((naocc, naocc, navir, nvir), dtype=np.float64)
    x33 += einsum(v.vVOV, (0, 1, 2, 3), t3, (4, 5, 2, 6, 1, 3), (4, 5, 6, 0))
    t2new[np.ix_(sO,sO,sV,sv)] += einsum(x33, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(sO,sO,sv,sV)] += einsum(x33, (0, 1, 2, 3), (1, 0, 3, 2))
    del x33
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x34 += einsum(x5, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 4), (1, 2, 3, 4))
    del x5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x34, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x34, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x34
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x35 += einsum(f.ov, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (0, 2, 3, 4))
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x36 += einsum(t1[np.ix_(so,sv)], (0, 1), x35, (0, 2, 3, 4), (2, 3, 1, 4))
    del x35
    t2new[np.ix_(so,so,sv,sv)] += einsum(x36, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x36, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x36
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x37 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x38 += einsum(t1[np.ix_(so,sv)], (0, 1), x37, (2, 3, 0, 4), (2, 3, 1, 4))
    del x37
    t2new[np.ix_(so,so,sv,sv)] += einsum(x38, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x38, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x38
    x39 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x39 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x39, (2, 0, 3, 4), (3, 2, 4, 1))
    del x39
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x40 += einsum(t1[np.ix_(so,sv)], (0, 1), x10, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x40, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x40
    x41 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x41 += einsum(t1[np.ix_(so,sv)], (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x41, (2, 3, 1, 4), (0, 2, 3, 4))
    del x41
    x42 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x42 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x43 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x42, (4, 5, 0, 1), (4, 5, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x43, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x43, (0, 1, 2, 3), (1, 0, 2, 3))
    del x43
    x44 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x44 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 1, 5, 2), (0, 4, 5, 3))
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x45 += einsum(t1[np.ix_(so,sv)], (0, 1), x44, (2, 3, 0, 4), (2, 3, 1, 4))
    del x44
    t2new[np.ix_(so,so,sv,sv)] += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x45, (0, 1, 2, 3), (1, 0, 3, 2))
    del x45
    x46 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x46 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x47 += einsum(t1[np.ix_(so,sv)], (0, 1), x46, (2, 3, 0, 4), (2, 3, 1, 4))
    del x46
    t2new[np.ix_(so,so,sv,sv)] += einsum(x47, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x47, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x47
    x48 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x48 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 5, 1, 2), (0, 4, 5, 3))
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x49 += einsum(t1[np.ix_(so,sv)], (0, 1), x48, (2, 3, 0, 4), (2, 3, 1, 4))
    del x48
    t2new[np.ix_(so,so,sv,sv)] += einsum(x49, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x49, (0, 1, 2, 3), (0, 1, 3, 2))
    del x49
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x50 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x51 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x51 += einsum(t1[np.ix_(so,sv)], (0, 1), x50, (2, 3, 0, 4), (2, 3, 1, 4))
    del x50
    t2new[np.ix_(so,so,sv,sv)] += einsum(x51, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x51, (0, 1, 2, 3), (0, 1, 3, 2))
    del x51
    x52 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x52 += einsum(x6, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x6
    t2new[np.ix_(so,so,sv,sv)] += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x52, (0, 1, 2, 3), (1, 0, 3, 2))
    del x52
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x53 += einsum(x7, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x7
    t2new[np.ix_(so,so,sv,sv)] += einsum(x53, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x53, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x53
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x54 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x55 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x54, (4, 1, 5, 3), (4, 0, 2, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x55, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x55, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x55
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x56 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x54, (4, 1, 5, 2), (4, 0, 3, 5))
    del x54
    t2new[np.ix_(so,so,sv,sv)] += einsum(x56, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x56, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x56
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x57 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x22, (4, 1, 3, 5), (4, 0, 2, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x57, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x57, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x57
    x58 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x58 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x22, (4, 1, 2, 5), (4, 0, 3, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x58, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x58, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x58
    x59 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x59 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x60 += einsum(t1[np.ix_(so,sv)], (0, 1), x59, (2, 3, 0, 4), (2, 3, 1, 4))
    del x59
    t2new[np.ix_(so,so,sv,sv)] += einsum(x60, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x60, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x60
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x61 += einsum(x9, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 4, 0))
    del x9
    t2new[np.ix_(so,so,sv,sv)] += einsum(x61, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x61, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x61
    x62 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x62 += einsum(x8, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 0), (2, 3, 4, 1))
    del x8
    t2new[np.ix_(so,so,sv,sv)] += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x62, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x62
    x63 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x63 += einsum(t1[np.ix_(so,sv)], (0, 1), v.vOOV, (1, 2, 3, 4), (3, 2, 4, 0))
    t2new[np.ix_(so,sO,sV,sV)] += einsum(x63, (0, 1, 2, 3), t3, (4, 1, 0, 5, 6, 2), (3, 4, 6, 5)) * -1.0
    x64 = np.zeros((naocc, navir, navir, nocc), dtype=np.float64)
    x64 += einsum(x63, (0, 1, 2, 3), t3, (0, 4, 1, 2, 5, 6), (4, 6, 5, 3))
    del x63
    t2new[np.ix_(so,sO,sV,sV)] += einsum(x64, (0, 1, 2, 3), (3, 0, 1, 2)) * -0.5
    t2new[np.ix_(sO,so,sV,sV)] += einsum(x64, (0, 1, 2, 3), (0, 3, 2, 1)) * -0.5
    del x64
    x65 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x65 += einsum(t1[np.ix_(so,sv)], (0, 1), v.vOOV, (1, 2, 3, 4), (3, 2, 4, 0))
    t2new[np.ix_(sO,so,sV,sV)] += einsum(x65, (0, 1, 2, 3), t3, (4, 1, 0, 5, 6, 2), (4, 3, 5, 6)) * -1.0
    x66 = np.zeros((naocc, navir, navir, nocc), dtype=np.float64)
    x66 += einsum(x65, (0, 1, 2, 3), t3, (0, 4, 1, 5, 6, 2), (4, 5, 6, 3))
    del x65
    t2new[np.ix_(so,sO,sV,sV)] += einsum(x66, (0, 1, 2, 3), (3, 0, 1, 2)) * 0.5
    t2new[np.ix_(sO,so,sV,sV)] += einsum(x66, (0, 1, 2, 3), (0, 3, 2, 1)) * 0.5
    del x66
    x67 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x67 += einsum(v.oVOV, (0, 1, 2, 3), t3, (4, 5, 2, 1, 6, 3), (4, 5, 6, 0))
    x68 = np.zeros((naocc, naocc, navir, nvir), dtype=np.float64)
    x68 += einsum(t1[np.ix_(so,sv)], (0, 1), x67, (2, 3, 4, 0), (2, 3, 4, 1))
    del x67
    t2new[np.ix_(sO,sO,sv,sV)] += einsum(x68, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    t2new[np.ix_(sO,sO,sV,sv)] += einsum(x68, (0, 1, 2, 3), (1, 0, 2, 3)) * -0.5
    del x68
    x69 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x69 += einsum(v.oVOV, (0, 1, 2, 3), t3, (4, 5, 2, 3, 6, 1), (4, 5, 6, 0))
    x70 = np.zeros((naocc, naocc, navir, nvir), dtype=np.float64)
    x70 += einsum(t1[np.ix_(so,sv)], (0, 1), x69, (2, 3, 4, 0), (2, 3, 4, 1))
    del x69
    t2new[np.ix_(sO,sO,sv,sV)] += einsum(x70, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    t2new[np.ix_(sO,sO,sV,sv)] += einsum(x70, (0, 1, 2, 3), (1, 0, 2, 3)) * 0.5
    del x70
    x71 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x71 += einsum(v.oVOV, (0, 1, 2, 3), t3, (4, 5, 2, 6, 1, 3), (4, 5, 6, 0))
    x72 = np.zeros((naocc, naocc, navir, nvir), dtype=np.float64)
    x72 += einsum(t1[np.ix_(so,sv)], (0, 1), x71, (2, 3, 4, 0), (2, 3, 4, 1))
    del x71
    t2new[np.ix_(sO,sO,sv,sV)] += einsum(x72, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new[np.ix_(sO,sO,sV,sv)] += einsum(x72, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x72
    x73 = np.zeros((naocc, navir), dtype=np.float64)
    x73 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovOV, (0, 1, 2, 3), (2, 3))
    x74 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x74 += einsum(x73, (0, 1), t3, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    del x73
    t2new[np.ix_(sO,sO,sV,sV)] += einsum(x74, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new[np.ix_(sO,sO,sV,sV)] += einsum(x74, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x74
    x75 = np.zeros((naocc, navir), dtype=np.float64)
    x75 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oVvO, (0, 2, 1, 3), (3, 2))
    x76 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x76 += einsum(x75, (0, 1), t3, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    del x75
    t2new[np.ix_(sO,sO,sV,sV)] += einsum(x76, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(sO,sO,sV,sV)] += einsum(x76, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x76
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x77 += einsum(x15, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x15
    t2new[np.ix_(so,so,sv,sv)] += einsum(x77, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x77, (0, 1, 2, 3), (1, 0, 3, 2))
    del x77
    x78 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x78 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x27, (4, 1, 5, 3), (0, 4, 2, 5))
    del x27
    t2new[np.ix_(so,so,sv,sv)] += einsum(x78, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x78, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x78
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x79 += einsum(x13, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x13
    t2new[np.ix_(so,so,sv,sv)] += einsum(x79, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x79, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x79, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x79
    x80 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x80 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x80, (4, 1, 5, 3), (0, 4, 2, 5)) * -2.0
    del x80
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x81 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x81, (4, 1, 5, 2), (0, 4, 5, 3))
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x82 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x81, (4, 1, 5, 3), (0, 4, 2, 5))
    del x81
    t2new[np.ix_(so,so,sv,sv)] += einsum(x82, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x82, (0, 1, 2, 3), (1, 0, 3, 2))
    del x82
    x83 = np.zeros((nvir, nvir), dtype=np.float64)
    x83 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (0, 4, 1, 3), (2, 4))
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x84 += einsum(x83, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 4, 0))
    del x83
    t2new[np.ix_(so,so,sv,sv)] += einsum(x84, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x84, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x84, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x84, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    del x84
    x85 = np.zeros((nvir, nvir), dtype=np.float64)
    x85 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (0, 3, 1, 4), (2, 4))
    x86 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x86 += einsum(x85, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 4, 0))
    del x85
    t2new[np.ix_(so,so,sv,sv)] += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x86, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x86, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x86, (0, 1, 2, 3), (1, 0, 3, 2)) * 0.5
    del x86
    x87 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x87 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x87, (4, 5, 0, 1), (5, 4, 3, 2))
    x88 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x88 += einsum(t1[np.ix_(so,sv)], (0, 1), x42, (2, 3, 4, 0), (2, 4, 3, 1))
    del x42
    x89 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x89 += einsum(t1[np.ix_(so,sv)], (0, 1), x88, (2, 0, 3, 4), (2, 3, 1, 4))
    del x88
    t2new[np.ix_(so,so,sv,sv)] += einsum(x89, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x89, (0, 1, 2, 3), (1, 0, 2, 3))
    del x89
    x90 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x90 += einsum(t1[np.ix_(so,sv)], (0, 1), x22, (2, 3, 1, 4), (0, 2, 3, 4))
    del x22
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x91 += einsum(t1[np.ix_(so,sv)], (0, 1), x90, (2, 3, 0, 4), (2, 3, 1, 4))
    del x90
    t2new[np.ix_(so,so,sv,sv)] += einsum(x91, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x91, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x91
    x92 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x92 += einsum(t1[np.ix_(so,sv)], (0, 1), x10, (2, 3, 4, 1), (2, 0, 4, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x92, (4, 5, 0, 1), (5, 4, 3, 2))
    x93 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x93 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x10, (4, 1, 5, 3), (4, 0, 5, 2))
    x94 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x94 += einsum(t1[np.ix_(so,sv)], (0, 1), x93, (2, 3, 0, 4), (2, 3, 1, 4))
    del x93
    t2new[np.ix_(so,so,sv,sv)] += einsum(x94, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x94, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x94
    x95 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x95 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x10, (4, 1, 5, 2), (4, 0, 5, 3))
    x96 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x96 += einsum(t1[np.ix_(so,sv)], (0, 1), x95, (2, 3, 0, 4), (2, 3, 1, 4))
    del x95
    t2new[np.ix_(so,so,sv,sv)] += einsum(x96, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x96, (0, 1, 2, 3), (1, 0, 3, 2))
    del x96
    x97 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x97 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x10, (4, 5, 1, 3), (4, 0, 5, 2))
    x98 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x98 += einsum(t1[np.ix_(so,sv)], (0, 1), x97, (2, 3, 0, 4), (2, 3, 1, 4))
    del x97
    t2new[np.ix_(so,so,sv,sv)] += einsum(x98, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x98, (0, 1, 2, 3), (1, 0, 3, 2))
    del x98
    x99 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x99 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x10, (4, 5, 1, 2), (4, 0, 5, 3))
    del x10
    x100 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x100 += einsum(t1[np.ix_(so,sv)], (0, 1), x99, (2, 3, 0, 4), (2, 3, 1, 4))
    del x99
    t2new[np.ix_(so,so,sv,sv)] += einsum(x100, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x100, (0, 1, 2, 3), (1, 0, 2, 3))
    del x100
    x101 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x101 += einsum(x17, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (0, 2, 3, 4))
    del x17
    t2new[np.ix_(so,so,sv,sv)] += einsum(x101, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x101, (0, 1, 2, 3), (1, 0, 2, 3))
    del x101
    x102 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x102 += einsum(x18, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (0, 2, 3, 4))
    del x18
    t2new[np.ix_(so,so,sv,sv)] += einsum(x102, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x102, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    del x102
    x103 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x103 += einsum(t1[np.ix_(so,sv)], (0, 1), x87, (2, 3, 0, 4), (3, 2, 4, 1))
    del x87
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x103, (2, 3, 0, 4), (3, 2, 4, 1))
    del x103
    x104 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x104 += einsum(x0, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 0, 4))
    x105 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x105 += einsum(t1[np.ix_(so,sv)], (0, 1), x104, (2, 3, 0, 4), (2, 3, 1, 4))
    del x104
    t2new[np.ix_(so,so,sv,sv)] += einsum(x105, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x105, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x105
    x106 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x106 += einsum(x16, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 0, 4))
    x107 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x107 += einsum(t1[np.ix_(so,sv)], (0, 1), x106, (2, 3, 0, 4), (2, 3, 1, 4))
    del x106
    t2new[np.ix_(so,so,sv,sv)] += einsum(x107, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x107, (0, 1, 2, 3), (0, 1, 3, 2))
    del x107
    x108 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x108 += einsum(t1[np.ix_(so,sv)], (0, 1), x92, (2, 3, 0, 4), (3, 2, 4, 1))
    del x92
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x108, (2, 3, 0, 4), (2, 3, 1, 4))
    del x108
    x109 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x109 += einsum(f.OO, (0, 1), t3, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 6, 5))
    t3new += einsum(x109, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x109, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x109
    x110 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x110 += einsum(f.VV, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    del x110
    x111 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x111 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), v.oOOV, (0, 4, 5, 6), (1, 5, 4, 3, 2, 6))
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x111, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    del x111
    x112 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x112 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.vVOV, (2, 4, 5, 6), (1, 0, 5, 3, 6, 4))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x112
    x113 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x113 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.vVOV, (2, 4, 5, 6), (1, 0, 5, 3, 6, 4))
    t3new += einsum(x113, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x113, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x113, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x113, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x113, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x113, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    del x113
    x114 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x114 += einsum(v.OOOO, (0, 1, 2, 3), t3, (4, 3, 1, 5, 6, 7), (4, 0, 2, 5, 7, 6))
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    del x114
    x115 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x115 += einsum(v.OOOO, (0, 1, 2, 3), t3, (3, 4, 1, 5, 6, 7), (4, 0, 2, 7, 5, 6))
    t3new += einsum(x115, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 0.5
    t3new += einsum(x115, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -0.5
    del x115
    x116 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x116 += einsum(v.OOVV, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    del x116
    x117 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x117 += einsum(v.OOVV, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (4, 5, 0, 6, 7, 2))
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x117, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    del x117
    x118 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x118 += einsum(v.OOVV, (0, 1, 2, 3), t3, (4, 1, 5, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x118, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    del x118
    x119 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x119 += einsum(v.OVOV, (0, 1, 2, 3), t3, (4, 5, 2, 6, 7, 3), (4, 5, 0, 6, 7, 1))
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    del x119
    x120 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x120 += einsum(v.VVVV, (0, 1, 2, 3), t3, (4, 5, 6, 7, 3, 1), (4, 6, 5, 7, 0, 2))
    t3new += einsum(x120, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x120, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    del x120
    x121 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x121 += einsum(v.VVVV, (0, 1, 2, 3), t3, (4, 5, 6, 1, 7, 3), (4, 6, 5, 7, 0, 2))
    t3new += einsum(x121, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 0.5
    t3new += einsum(x121, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -0.5
    del x121
    x122 = np.zeros((naocc, naocc), dtype=np.float64)
    x122 += einsum(f.vO, (0, 1), t1[np.ix_(sO,sv)], (2, 0), (1, 2))
    t3new += einsum(x122, (0, 1), t3, (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -1.0
    x123 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x123 += einsum(x122, (0, 1), t3, (2, 3, 0, 4, 5, 6), (1, 2, 3, 4, 6, 5))
    del x122
    t3new += einsum(x123, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x123, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x123
    x124 = np.zeros((navir, navir), dtype=np.float64)
    x124 += einsum(f.oV, (0, 1), t1[np.ix_(so,sV)], (0, 2), (1, 2))
    t3new += einsum(x124, (0, 1), t3, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -1.0
    x125 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x125 += einsum(x124, (0, 1), t3, (2, 3, 4, 5, 6, 0), (2, 4, 3, 1, 5, 6))
    del x124
    t3new += einsum(x125, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x125, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    del x125
    x126 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x126 += einsum(f.ov, (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 1, 4), (3, 2, 4, 0))
    x127 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x127 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x126, (4, 5, 6, 0), (4, 5, 1, 6, 3, 2))
    del x126
    t3new += einsum(x127, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x127, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x127, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x127, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new += einsum(x127, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x127, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x127
    x128 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x128 += einsum(f.ov, (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 1, 4), (3, 2, 4, 0))
    x129 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x129 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x128, (4, 5, 6, 0), (4, 5, 1, 6, 3, 2))
    del x128
    t3new += einsum(x129, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x129, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x129, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x129, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x129, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new += einsum(x129, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    del x129
    x130 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x130 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.oOvV, (2, 3, 1, 4), (0, 3, 4, 2))
    x131 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x131 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x130, (4, 5, 6, 0), (4, 1, 5, 3, 2, 6))
    del x130
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    del x131
    x132 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x132 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), v.oOoO, (4, 5, 0, 6), (1, 5, 6, 3, 2, 4))
    x133 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x133 += einsum(t1[np.ix_(so,sV)], (0, 1), x132, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x132
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x133, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x133
    x134 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x134 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.oOvV, (4, 5, 2, 6), (1, 0, 5, 3, 6, 4))
    x135 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x135 += einsum(t1[np.ix_(so,sV)], (0, 1), x134, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x134
    t3new += einsum(x135, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x135, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x135, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x135, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x135, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x135, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    del x135
    x136 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x136 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.oOvV, (4, 5, 2, 6), (1, 0, 5, 3, 6, 4))
    x137 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x137 += einsum(t1[np.ix_(so,sV)], (0, 1), x136, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x136
    t3new += einsum(x137, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x137, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x137, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x137, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new += einsum(x137, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x137, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x137
    x138 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x138 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovOV, (2, 1, 3, 4), (0, 3, 4, 2))
    x139 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x139 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x138, (4, 5, 6, 0), (4, 1, 5, 3, 2, 6))
    del x138
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5))
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x139, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    del x139
    x140 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x140 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.ovOV, (4, 2, 5, 6), (1, 0, 5, 3, 6, 4))
    x141 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x141 += einsum(t1[np.ix_(so,sV)], (0, 1), x140, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x140
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5))
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -1.0
    del x141
    x142 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x142 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.ovOV, (4, 2, 5, 6), (1, 0, 5, 3, 6, 4))
    x143 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x143 += einsum(t1[np.ix_(so,sV)], (0, 1), x142, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x142
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    del x143
    x144 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=np.float64)
    x144 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.vVvV, (4, 5, 2, 6), (1, 0, 3, 5, 6, 4))
    x145 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x145 += einsum(t1[np.ix_(sO,sv)], (0, 1), x144, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x144
    t3new += einsum(x145, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x145, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x145, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x145, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x145, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x145, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x145
    x146 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=np.float64)
    x146 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.vVvV, (4, 5, 2, 6), (1, 0, 3, 5, 6, 4))
    x147 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x147 += einsum(t1[np.ix_(sO,sv)], (0, 1), x146, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x146
    t3new += einsum(x147, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x147, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x147, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x147, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x147, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x147, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x147
    x148 = np.zeros((naocc, naocc, naocc, naocc), dtype=np.float64)
    x148 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vOOO, (1, 2, 3, 4), (0, 3, 4, 2))
    x149 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x149 += einsum(x148, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 7), (0, 4, 1, 5, 7, 6))
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x149
    x150 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x150 += einsum(x148, (0, 1, 2, 3), t3, (2, 4, 3, 5, 6, 7), (0, 4, 1, 5, 7, 6))
    t3new += einsum(x150, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -0.5
    t3new += einsum(x150, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 0.5
    t3new += einsum(x150, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -0.5
    t3new += einsum(x150, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 0.5
    del x150
    x151 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x151 += einsum(x148, (0, 1, 2, 3), t3, (4, 3, 2, 5, 6, 7), (0, 4, 1, 5, 7, 6))
    del x148
    t3new += einsum(x151, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x151, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4))
    del x151
    x152 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x152 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oOOV, (0, 2, 3, 4), (3, 2, 1, 4))
    t3new += einsum(x152, (0, 1, 2, 3), t3, (4, 5, 0, 6, 3, 7), (4, 1, 5, 6, 2, 7))
    t3new += einsum(x152, (0, 1, 2, 3), t3, (4, 0, 5, 6, 3, 7), (4, 1, 5, 6, 2, 7)) * -1.0
    x153 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x153 += einsum(x152, (0, 1, 2, 3), t3, (4, 5, 0, 6, 7, 3), (4, 5, 1, 2, 6, 7))
    del x152
    t3new += einsum(x153, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x153, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x153, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x153, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x153, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x153, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x153, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x153, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x153, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x153, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    del x153
    x154 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x154 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oVOO, (0, 2, 3, 4), (3, 4, 1, 2))
    t3new += einsum(x154, (0, 1, 2, 3), t3, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7))
    x155 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x155 += einsum(x154, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (4, 5, 0, 2, 6, 7))
    t3new += einsum(x155, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x155, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x155, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x155, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x155
    x156 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x156 += einsum(x154, (0, 1, 2, 3), t3, (4, 1, 5, 6, 7, 3), (4, 5, 0, 2, 6, 7))
    t3new += einsum(x156, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x156, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    del x156
    x157 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x157 += einsum(x154, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (4, 5, 0, 2, 6, 7))
    del x154
    t3new += einsum(x157, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x157, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x157
    x158 = np.zeros((naocc, naocc), dtype=np.float64)
    x158 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvO, (0, 2, 1, 3), (2, 3))
    t3new += einsum(x158, (0, 1), t3, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6))
    x159 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x159 += einsum(x158, (0, 1), t3, (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 6, 5))
    del x158
    t3new += einsum(x159, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x159, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    del x159
    x160 = np.zeros((naocc, naocc), dtype=np.float64)
    x160 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovOO, (0, 1, 2, 3), (2, 3))
    t3new += einsum(x160, (0, 1), t3, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x161 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x161 += einsum(x160, (0, 1), t3, (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 6, 5))
    del x160
    t3new += einsum(x161, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 2.0
    t3new += einsum(x161, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -2.0
    del x161
    x162 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x162 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vOVV, (1, 2, 3, 4), (0, 2, 3, 4))
    t3new += einsum(x162, (0, 1, 2, 3), t3, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -1.0
    x163 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x163 += einsum(x162, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 6, 7, 2))
    t3new += einsum(x163, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x163, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x163, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x163, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    del x163
    x164 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x164 += einsum(x162, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (0, 4, 5, 6, 7, 2))
    t3new += einsum(x164, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x164, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x164
    x165 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x165 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vVOV, (1, 2, 3, 4), (0, 3, 4, 2))
    t3new += einsum(x165, (0, 1, 2, 3), t3, (4, 5, 1, 6, 2, 7), (4, 0, 5, 6, 3, 7)) * -1.0
    t3new += einsum(x165, (0, 1, 2, 3), t3, (4, 1, 5, 6, 2, 7), (4, 0, 5, 6, 3, 7))
    x166 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x166 += einsum(x165, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 2), (0, 4, 5, 6, 7, 3))
    del x165
    t3new += einsum(x166, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x166, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x166, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x166, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x166, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x166, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4))
    t3new += einsum(x166, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x166, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x166, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x166, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    del x166
    x167 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x167 += einsum(x162, (0, 1, 2, 3), t3, (4, 1, 5, 6, 7, 3), (0, 4, 5, 6, 7, 2))
    del x162
    t3new += einsum(x167, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    t3new += einsum(x167, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    del x167
    x168 = np.zeros((navir, navir, navir, navir), dtype=np.float64)
    x168 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oVVV, (0, 2, 3, 4), (1, 3, 4, 2))
    x169 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x169 += einsum(x168, (0, 1, 2, 3), t3, (4, 5, 6, 7, 2, 3), (4, 6, 5, 0, 7, 1))
    t3new += einsum(x169, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x169, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    del x169
    x170 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x170 += einsum(x168, (0, 1, 2, 3), t3, (4, 5, 6, 2, 7, 3), (4, 6, 5, 0, 7, 1))
    t3new += einsum(x170, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -0.5
    t3new += einsum(x170, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 0.5
    t3new += einsum(x170, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -0.5
    t3new += einsum(x170, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 0.5
    del x170
    x171 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x171 += einsum(x168, (0, 1, 2, 3), t3, (4, 5, 6, 7, 3, 2), (4, 6, 5, 0, 7, 1))
    del x168
    t3new += einsum(x171, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x171, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    del x171
    x172 = np.zeros((navir, navir), dtype=np.float64)
    x172 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovVV, (0, 1, 2, 3), (2, 3))
    t3new += einsum(x172, (0, 1), t3, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    x173 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x173 += einsum(x172, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 4, 3, 5, 6, 0))
    del x172
    t3new += einsum(x173, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new += einsum(x173, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    del x173
    x174 = np.zeros((navir, navir), dtype=np.float64)
    x174 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oVvV, (0, 2, 1, 3), (2, 3))
    t3new += einsum(x174, (0, 1), t3, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -1.0
    x175 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x175 += einsum(x174, (0, 1), t3, (2, 3, 4, 5, 6, 0), (2, 4, 3, 5, 6, 1))
    del x174
    t3new += einsum(x175, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x175, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    del x175
    x176 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x176 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovoO, (0, 2, 4, 5), (1, 5, 3, 4))
    x177 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x177 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x176, (4, 5, 6, 0), (1, 4, 5, 3, 2, 6))
    del x176
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -2.0
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * 2.0
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new += einsum(x177, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    del x177
    x178 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x178 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovoO, (4, 2, 0, 5), (1, 5, 3, 4))
    x179 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x179 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x178, (4, 5, 6, 0), (1, 4, 5, 3, 2, 6))
    del x178
    t3new += einsum(x179, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x179, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x179, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x179, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x179, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x179, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x179, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x179, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x179, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x179, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x179, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x179, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x179
    x180 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x180 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovoO, (0, 3, 4, 5), (1, 5, 2, 4))
    x181 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x181 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x180, (4, 5, 6, 0), (1, 4, 5, 3, 2, 6))
    del x180
    t3new += einsum(x181, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x181, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x181, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x181, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x181, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x181, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x181, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x181, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x181, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x181, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x181, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x181, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x181
    x182 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x182 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovoO, (4, 3, 0, 5), (1, 5, 2, 4))
    x183 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x183 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x182, (4, 5, 6, 0), (1, 4, 5, 3, 2, 6))
    del x182
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x183, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    del x183
    x184 = np.zeros((naocc, navir, navir, nvir), dtype=np.float64)
    x184 += einsum(t2[np.ix_(so,so,sV,sV)], (0, 1, 2, 3), v.ovoO, (0, 4, 1, 5), (5, 3, 2, 4))
    x185 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x185 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x184, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    t3new += einsum(x185, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x185, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x185, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x185, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x185, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 0.5
    t3new += einsum(x185, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -0.5
    t3new += einsum(x185, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -0.5
    t3new += einsum(x185, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 0.5
    del x185
    x186 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x186 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x184, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    del x184
    t3new += einsum(x186, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x186, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x186, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -0.5
    t3new += einsum(x186, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 0.5
    t3new += einsum(x186, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 0.5
    t3new += einsum(x186, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -0.5
    t3new += einsum(x186, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x186, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    del x186
    x187 = np.zeros((naocc, navir, navir, nvir), dtype=np.float64)
    x187 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovvV, (0, 4, 2, 5), (1, 3, 5, 4))
    x188 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x188 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x187, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    t3new += einsum(x188, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x188, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x188, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x188, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new += einsum(x188, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x188, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x188
    x189 = np.zeros((naocc, navir, navir, nvir), dtype=np.float64)
    x189 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovvV, (0, 4, 3, 5), (1, 2, 5, 4))
    x190 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x190 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x189, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    t3new += einsum(x190, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x190, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x190, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x190, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x190, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x190, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    del x190
    x191 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x191 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x187, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    del x187
    t3new += einsum(x191, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x191, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x191, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x191, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x191, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new += einsum(x191, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    del x191
    x192 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x192 += einsum(t2[np.ix_(sO,sO,sv,sv)], (0, 1, 2, 3), v.ovvV, (4, 3, 2, 5), (0, 1, 5, 4))
    x193 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x193 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x192, (4, 5, 6, 0), (4, 5, 1, 3, 2, 6))
    del x192
    t3new += einsum(x193, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4))
    t3new += einsum(x193, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x193, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x193, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x193, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x193, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3))
    t3new += einsum(x193, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x193, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    t3new += einsum(x193, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4))
    t3new += einsum(x193, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x193, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x193, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    del x193
    x194 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x194 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x189, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    del x189
    t3new += einsum(x194, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x194, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x194, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x194, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x194, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x194, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    del x194
    x195 = np.zeros((naocc, navir, navir, nvir), dtype=np.float64)
    x195 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovvV, (0, 2, 4, 5), (1, 3, 5, 4))
    x196 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x196 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x195, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    t3new += einsum(x196, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new += einsum(x196, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new += einsum(x196, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3new += einsum(x196, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new += einsum(x196, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new += einsum(x196, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    del x196
    x197 = np.zeros((naocc, navir, navir, nvir), dtype=np.float64)
    x197 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovvV, (0, 3, 4, 5), (1, 2, 5, 4))
    x198 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x198 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x197, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    t3new += einsum(x198, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x198, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x198, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new += einsum(x198, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x198, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x198, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x198
    x199 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x199 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x195, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    del x195
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -2.0
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new += einsum(x199, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x199
    x200 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x200 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x197, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    del x197
    t3new += einsum(x200, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x200, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x200, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x200, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x200, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x200, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    del x200
    x201 = np.zeros((naocc, naocc, naocc, naocc, navir, navir), dtype=np.float64)
    x201 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.vOOV, (2, 4, 5, 6), (1, 0, 5, 4, 3, 6))
    t3new += einsum(t3, (0, 1, 2, 3, 4, 5), x201, (6, 7, 2, 1, 8, 4), (0, 6, 7, 3, 8, 5)) * 0.5
    t3new += einsum(t3, (0, 1, 2, 3, 4, 5), x201, (6, 7, 1, 2, 8, 4), (0, 6, 7, 3, 8, 5)) * -1.0
    x202 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x202 += einsum(t3, (0, 1, 2, 3, 4, 5), x201, (6, 7, 2, 0, 8, 5), (6, 7, 1, 8, 3, 4))
    t3new += einsum(x202, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -0.5
    t3new += einsum(x202, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 0.5
    t3new += einsum(x202, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -0.5
    t3new += einsum(x202, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 0.5
    del x202
    x203 = np.zeros((naocc, naocc, naocc, naocc, navir, navir), dtype=np.float64)
    x203 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.vOOV, (2, 4, 5, 6), (1, 0, 5, 4, 3, 6))
    t3new += einsum(t3, (0, 1, 2, 3, 4, 5), x203, (6, 7, 2, 1, 8, 4), (7, 6, 0, 3, 8, 5)) * -0.5
    t3new += einsum(t3, (0, 1, 2, 3, 4, 5), x203, (6, 7, 1, 2, 8, 4), (7, 6, 0, 3, 8, 5))
    x204 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x204 += einsum(t3, (0, 1, 2, 3, 4, 5), x203, (6, 7, 2, 1, 8, 5), (6, 7, 0, 8, 3, 4))
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 0.5
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -0.5
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x204, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    del x204
    x205 = np.zeros((naocc, naocc, naocc, naocc), dtype=np.float64)
    x205 += einsum(t2[np.ix_(sO,sO,sv,sv)], (0, 1, 2, 3), v.vOvO, (3, 4, 2, 5), (0, 1, 5, 4))
    x206 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x206 += einsum(x205, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 7), (1, 0, 4, 5, 7, 6))
    t3new += einsum(x206, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x206, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    del x206
    x207 = np.zeros((naocc, naocc, naocc, naocc, navir, navir), dtype=np.float64)
    x207 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.vOOV, (2, 4, 5, 6), (1, 0, 5, 4, 3, 6))
    t3new += einsum(t3, (0, 1, 2, 3, 4, 5), x207, (6, 7, 1, 2, 8, 4), (0, 6, 7, 3, 8, 5)) * -0.5
    x208 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x208 += einsum(t3, (0, 1, 2, 3, 4, 5), x207, (6, 7, 2, 1, 8, 5), (6, 7, 0, 8, 3, 4))
    t3new += einsum(x208, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x208, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x208
    x209 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x209 += einsum(t3, (0, 1, 2, 3, 4, 5), x207, (6, 7, 0, 2, 8, 5), (6, 7, 1, 8, 3, 4))
    t3new += einsum(x209, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.5
    t3new += einsum(x209, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -0.5
    t3new += einsum(x209, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.5
    t3new += einsum(x209, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.5
    del x209
    x210 = np.zeros((naocc, naocc, naocc, naocc, navir, navir), dtype=np.float64)
    x210 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.vOOV, (2, 4, 5, 6), (1, 0, 5, 4, 3, 6))
    t3new += einsum(t3, (0, 1, 2, 3, 4, 5), x210, (6, 7, 1, 2, 8, 4), (7, 6, 0, 3, 8, 5)) * 0.5
    x211 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x211 += einsum(t3, (0, 1, 2, 3, 4, 5), x210, (6, 7, 1, 2, 8, 5), (6, 7, 0, 8, 3, 4))
    t3new += einsum(x211, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -0.5
    t3new += einsum(x211, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 0.5
    del x211
    x212 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x212 += einsum(t3, (0, 1, 2, 3, 4, 5), x201, (6, 7, 2, 1, 8, 5), (6, 7, 0, 8, 3, 4))
    del x201
    t3new += einsum(x212, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x212, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x212, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -0.5
    t3new += einsum(x212, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 0.5
    del x212
    x213 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x213 += einsum(t3, (0, 1, 2, 3, 4, 5), x203, (6, 7, 2, 0, 8, 5), (6, 7, 1, 8, 3, 4))
    del x203
    t3new += einsum(x213, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 0.5
    t3new += einsum(x213, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -0.5
    t3new += einsum(x213, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 0.5
    t3new += einsum(x213, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -0.5
    del x213
    x214 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x214 += einsum(x205, (0, 1, 2, 3), t3, (3, 4, 2, 5, 6, 7), (1, 0, 4, 5, 7, 6))
    del x205
    t3new += einsum(x214, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.5
    t3new += einsum(x214, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.5
    del x214
    x215 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x215 += einsum(t3, (0, 1, 2, 3, 4, 5), x210, (6, 7, 0, 2, 8, 5), (6, 7, 1, 8, 3, 4))
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -0.5
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 0.5
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -0.5
    t3new += einsum(x215, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 0.5
    del x215
    x216 = np.zeros((naocc, naocc, navir, navir, navir, navir), dtype=np.float64)
    x216 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), v.oVOV, (0, 4, 5, 6), (1, 5, 3, 2, 6, 4))
    x217 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x217 += einsum(t3, (0, 1, 2, 3, 4, 5), x216, (6, 2, 7, 8, 5, 3), (6, 0, 1, 7, 8, 4))
    t3new += einsum(x217, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -0.5
    t3new += einsum(x217, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -0.5
    t3new += einsum(x217, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 0.5
    t3new += einsum(x217, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 0.5
    t3new += einsum(x217, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 0.5
    t3new += einsum(x217, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 0.5
    t3new += einsum(x217, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -0.5
    t3new += einsum(x217, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -0.5
    del x217
    x218 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x218 += einsum(t3, (0, 1, 2, 3, 4, 5), x216, (6, 2, 7, 8, 3, 5), (6, 0, 1, 7, 8, 4))
    t3new += einsum(x218, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 0.5
    t3new += einsum(x218, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.5
    t3new += einsum(x218, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -0.5
    t3new += einsum(x218, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.5
    t3new += einsum(x218, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -0.5
    t3new += einsum(x218, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -0.5
    t3new += einsum(x218, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 0.5
    t3new += einsum(x218, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 0.5
    del x218
    x219 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x219 += einsum(t3, (0, 1, 2, 3, 4, 5), x216, (6, 2, 7, 8, 5, 4), (6, 0, 1, 7, 8, 3))
    t3new += einsum(x219, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x219, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x219, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x219, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x219, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x219, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x219, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x219, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    del x219
    x220 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x220 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovOV, (0, 2, 4, 5), (1, 4, 3, 5))
    t3new += einsum(x220, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -2.00000000000002
    t3new += einsum(x220, (0, 1, 2, 3), t3, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 2.00000000000002
    x221 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x221 += einsum(x220, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x220
    t3new += einsum(x221, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.00000000000002
    t3new += einsum(x221, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.00000000000002
    t3new += einsum(x221, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.00000000000002
    t3new += einsum(x221, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.00000000000002
    t3new += einsum(x221, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.00000000000002
    t3new += einsum(x221, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.00000000000002
    t3new += einsum(x221, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.00000000000002
    t3new += einsum(x221, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -2.00000000000002
    t3new += einsum(x221, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.00000000000002
    t3new += einsum(x221, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.00000000000002
    del x221
    x222 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x222 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovOV, (0, 3, 4, 5), (1, 4, 2, 5))
    t3new += einsum(x222, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 1.00000000000001
    t3new += einsum(x222, (0, 1, 2, 3), t3, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -1.00000000000001
    x223 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x223 += einsum(x222, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x222
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.00000000000001
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 1.00000000000001
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.00000000000001
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 1.00000000000001
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.00000000000001
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 1.00000000000001
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 1.00000000000001
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.00000000000001
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 1.00000000000001
    t3new += einsum(x223, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.00000000000001
    del x223
    x224 = np.zeros((naocc, naocc), dtype=np.float64)
    x224 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvO, (0, 2, 3, 4), (1, 4))
    t3new += einsum(x224, (0, 1), t3, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x225 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x225 += einsum(x224, (0, 1), t3, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 6, 5))
    del x224
    t3new += einsum(x225, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x225, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x225, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x225, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x225
    x226 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x226 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oVvO, (0, 4, 2, 5), (1, 5, 3, 4))
    t3new += einsum(x226, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 1.00000000000001
    t3new += einsum(x226, (0, 1, 2, 3), t3, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -1.00000000000001
    x227 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x227 += einsum(x226, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x226
    t3new += einsum(x227, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.00000000000001
    t3new += einsum(x227, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.00000000000001
    t3new += einsum(x227, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 1.00000000000001
    t3new += einsum(x227, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 1.00000000000001
    t3new += einsum(x227, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.00000000000001
    t3new += einsum(x227, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 1.00000000000001
    t3new += einsum(x227, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 1.00000000000001
    t3new += einsum(x227, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 1.00000000000001
    t3new += einsum(x227, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.00000000000001
    t3new += einsum(x227, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.00000000000001
    del x227
    x228 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x228 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.oVvO, (0, 4, 3, 5), (1, 5, 2, 4))
    t3new += einsum(x228, (0, 1, 2, 3), t3, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 1.00000000000001
    x229 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x229 += einsum(x228, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    t3new += einsum(x229, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 1.00000000000001
    t3new += einsum(x229, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.00000000000001
    t3new += einsum(x229, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.00000000000001
    t3new += einsum(x229, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 1.00000000000001
    del x229
    x230 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x230 += einsum(x228, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (0, 4, 5, 2, 6, 7))
    t3new += einsum(x230, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.00000000000001
    t3new += einsum(x230, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 1.00000000000001
    del x230
    x231 = np.zeros((naocc, naocc), dtype=np.float64)
    x231 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.ovvO, (0, 3, 2, 4), (1, 4))
    t3new += einsum(x231, (0, 1), t3, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6))
    x232 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x232 += einsum(x231, (0, 1), t3, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 6, 5))
    del x231
    t3new += einsum(x232, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x232, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x232
    x233 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x233 += einsum(t3, (0, 1, 2, 3, 4, 5), x207, (6, 7, 1, 2, 8, 5), (6, 7, 0, 8, 3, 4))
    del x207
    t3new += einsum(x233, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 0.5
    t3new += einsum(x233, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -0.5
    del x233
    x234 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x234 += einsum(t3, (0, 1, 2, 3, 4, 5), x210, (6, 7, 2, 1, 8, 5), (6, 7, 0, 8, 3, 4))
    del x210
    t3new += einsum(x234, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x234, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    del x234
    x235 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x235 += einsum(t3, (0, 1, 2, 3, 4, 5), x216, (6, 1, 7, 8, 5, 4), (6, 0, 2, 7, 8, 3))
    t3new += einsum(x235, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x235, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    del x235
    x236 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x236 += einsum(t3, (0, 1, 2, 3, 4, 5), x216, (6, 1, 7, 8, 4, 5), (6, 0, 2, 7, 8, 3))
    t3new += einsum(x236, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new += einsum(x236, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x236
    x237 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x237 += einsum(t3, (0, 1, 2, 3, 4, 5), x216, (6, 1, 7, 8, 3, 5), (6, 0, 2, 7, 8, 4))
    del x216
    t3new += einsum(x237, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 0.5
    t3new += einsum(x237, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -0.5
    t3new += einsum(x237, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -0.5
    t3new += einsum(x237, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 0.5
    del x237
    x238 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x238 += einsum(x228, (0, 1, 2, 3), t3, (4, 1, 5, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x228
    t3new += einsum(x238, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.00000000000001
    t3new += einsum(x238, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 1.00000000000001
    del x238
    x239 = np.zeros((navir, navir, navir, navir), dtype=np.float64)
    x239 += einsum(t2[np.ix_(so,so,sV,sV)], (0, 1, 2, 3), v.oVoV, (0, 4, 1, 5), (2, 3, 4, 5))
    x240 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x240 += einsum(x239, (0, 1, 2, 3), t3, (4, 5, 6, 7, 3, 2), (4, 6, 5, 0, 1, 7))
    t3new += einsum(x240, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x240, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    del x240
    x241 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x241 += einsum(x239, (0, 1, 2, 3), t3, (4, 5, 6, 3, 7, 2), (4, 6, 5, 1, 0, 7))
    del x239
    t3new += einsum(x241, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.25
    t3new += einsum(x241, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.25
    t3new += einsum(x241, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.25
    t3new += einsum(x241, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.25
    del x241
    x242 = np.zeros((navir, navir), dtype=np.float64)
    x242 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (1, 2, 0, 4), (3, 4))
    t3new += einsum(x242, (0, 1), t3, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 0.5
    x243 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x243 += einsum(x242, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    del x242
    t3new += einsum(x243, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -0.5
    t3new += einsum(x243, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 0.5
    del x243
    x244 = np.zeros((navir, navir), dtype=np.float64)
    x244 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (0, 2, 1, 4), (3, 4))
    t3new += einsum(x244, (0, 1), t3, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -1.5
    x245 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x245 += einsum(x244, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    del x244
    t3new += einsum(x245, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.5
    t3new += einsum(x245, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.5
    del x245
    x246 = np.zeros((navir, navir), dtype=np.float64)
    x246 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (0, 2, 1, 4), (3, 4))
    t3new += einsum(x246, (0, 1), t3, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -0.5
    x247 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x247 += einsum(x246, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    del x246
    t3new += einsum(x247, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 1.5
    t3new += einsum(x247, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.5
    del x247
    x248 = np.zeros((navir, navir), dtype=np.float64)
    x248 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.ovoV, (1, 2, 0, 4), (3, 4))
    t3new += einsum(x248, (0, 1), t3, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 0.5
    x249 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x249 += einsum(x248, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    del x248
    t3new += einsum(x249, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -0.5
    t3new += einsum(x249, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 0.5
    del x249
    x250 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=np.float64)
    x250 += einsum(t1[np.ix_(so,sV)], (0, 1), t2[np.ix_(so,sO,sV,sV)], (2, 3, 4, 5), v.ovoO, (2, 6, 0, 7), (3, 7, 1, 5, 4, 6))
    x251 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x251 += einsum(t1[np.ix_(sO,sv)], (0, 1), x250, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x250
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x251, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x251
    x252 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=np.float64)
    x252 += einsum(t1[np.ix_(so,sV)], (0, 1), t2[np.ix_(so,sO,sV,sV)], (2, 3, 4, 5), v.ovoO, (0, 6, 2, 7), (3, 7, 1, 5, 4, 6))
    x253 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x253 += einsum(t1[np.ix_(sO,sv)], (0, 1), x252, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x252
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new += einsum(x253, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    del x253
    x254 = np.zeros((naocc, navir, navir, nvir), dtype=np.float64)
    x254 += einsum(t1[np.ix_(so,sV)], (0, 1), t1[np.ix_(so,sV)], (2, 3), v.ovoO, (0, 4, 2, 5), (5, 3, 1, 4))
    x255 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x255 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x254, (4, 5, 6, 2), (1, 0, 4, 5, 6, 3))
    t3new += einsum(x255, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x255, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x255, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new += einsum(x255, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x255, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x255, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    del x255
    x256 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x256 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x254, (4, 5, 6, 2), (1, 0, 4, 5, 6, 3))
    del x254
    t3new += einsum(x256, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new += einsum(x256, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x256, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x256, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x256, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x256, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x256
    x257 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=np.float64)
    x257 += einsum(t1[np.ix_(sO,sv)], (0, 1), t2[np.ix_(so,sO,sV,sV)], (2, 3, 4, 5), v.ovvV, (2, 1, 6, 7), (0, 3, 5, 4, 7, 6))
    x258 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x258 += einsum(t1[np.ix_(sO,sv)], (0, 1), x257, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x257
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4))
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4))
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x258, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3))
    del x258
    x259 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=np.float64)
    x259 += einsum(t1[np.ix_(so,sV)], (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 4, 5), v.ovvV, (0, 6, 4, 7), (3, 2, 1, 5, 7, 6))
    x260 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x260 += einsum(t1[np.ix_(sO,sv)], (0, 1), x259, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x259
    t3new += einsum(x260, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x260, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x260, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x260, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x260, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x260, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x260
    x261 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=np.float64)
    x261 += einsum(t1[np.ix_(so,sV)], (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 4, 5), v.ovvV, (0, 6, 4, 7), (3, 2, 1, 5, 7, 6))
    x262 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x262 += einsum(t1[np.ix_(sO,sv)], (0, 1), x261, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x261
    t3new += einsum(x262, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x262, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x262, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x262, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x262, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x262, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x262
    x263 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=np.float64)
    x263 += einsum(t1[np.ix_(so,sV)], (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 4, 5), v.ovvV, (0, 4, 6, 7), (3, 2, 1, 5, 7, 6))
    x264 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x264 += einsum(t1[np.ix_(sO,sv)], (0, 1), x263, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x263
    t3new += einsum(x264, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x264, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x264, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4))
    t3new += einsum(x264, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x264, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x264, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4))
    del x264
    x265 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=np.float64)
    x265 += einsum(t1[np.ix_(so,sV)], (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 4, 5), v.ovvV, (0, 4, 6, 7), (3, 2, 1, 5, 7, 6))
    x266 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x266 += einsum(t1[np.ix_(sO,sv)], (0, 1), x265, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x265
    t3new += einsum(x266, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x266, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x266, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x266, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3))
    t3new += einsum(x266, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x266, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    del x266
    x267 = np.zeros((naocc, naocc, naocc, nvir), dtype=np.float64)
    x267 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vOvO, (2, 3, 1, 4), (0, 3, 4, 2))
    x268 = np.zeros((naocc, naocc, naocc, naocc), dtype=np.float64)
    x268 += einsum(t1[np.ix_(sO,sv)], (0, 1), x267, (2, 3, 4, 1), (2, 0, 4, 3))
    del x267
    x269 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x269 += einsum(x268, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 7), (1, 0, 4, 5, 7, 6))
    t3new += einsum(x269, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x269, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    del x269
    x270 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x270 += einsum(x268, (0, 1, 2, 3), t3, (3, 4, 2, 5, 6, 7), (1, 0, 4, 5, 7, 6))
    del x268
    t3new += einsum(x270, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.5
    t3new += einsum(x270, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.5
    del x270
    x271 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x271 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.ovOV, (2, 1, 3, 4), (0, 3, 4, 2))
    x272 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x272 += einsum(t1[np.ix_(so,sV)], (0, 1), x271, (2, 3, 4, 0), (2, 3, 1, 4))
    del x271
    t3new += einsum(x272, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (4, 0, 5, 6, 2, 7))
    t3new += einsum(x272, (0, 1, 2, 3), t3, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -1.0
    x273 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x273 += einsum(x272, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x272
    t3new += einsum(x273, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x273, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x273, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x273, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x273, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x273, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x273, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x273, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x273, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x273, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x273
    x274 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x274 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.oVvO, (2, 3, 1, 4), (0, 4, 3, 2))
    x275 = np.zeros((naocc, naocc, navir, navir), dtype=np.float64)
    x275 += einsum(t1[np.ix_(so,sV)], (0, 1), x274, (2, 3, 4, 0), (2, 3, 1, 4))
    del x274
    t3new += einsum(x275, (0, 1, 2, 3), t3, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7))
    x276 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x276 += einsum(x275, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    t3new += einsum(x276, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x276, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x276, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x276, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x276
    x277 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x277 += einsum(x275, (0, 1, 2, 3), t3, (4, 5, 1, 6, 3, 7), (0, 4, 5, 2, 6, 7))
    t3new += einsum(x277, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x277, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    del x277
    x278 = np.zeros((naocc, nvir), dtype=np.float64)
    x278 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvO, (0, 2, 1, 3), (3, 2))
    x279 = np.zeros((naocc, naocc), dtype=np.float64)
    x279 += einsum(t1[np.ix_(sO,sv)], (0, 1), x278, (2, 1), (0, 2))
    del x278
    t3new += einsum(x279, (0, 1), t3, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6))
    x280 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x280 += einsum(x279, (0, 1), t3, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 6, 5))
    del x279
    t3new += einsum(x280, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x280, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x280
    x281 = np.zeros((naocc, nvir), dtype=np.float64)
    x281 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovvO, (0, 1, 2, 3), (3, 2))
    x282 = np.zeros((naocc, naocc), dtype=np.float64)
    x282 += einsum(t1[np.ix_(sO,sv)], (0, 1), x281, (2, 1), (0, 2))
    del x281
    t3new += einsum(x282, (0, 1), t3, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x283 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x283 += einsum(x282, (0, 1), t3, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 6, 5))
    del x282
    t3new += einsum(x283, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new += einsum(x283, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    del x283
    x284 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x284 += einsum(x275, (0, 1, 2, 3), t3, (4, 1, 5, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x275
    t3new += einsum(x284, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x284, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    del x284
    x285 = np.zeros((navir, navir, navir, nocc), dtype=np.float64)
    x285 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oVoV, (2, 3, 0, 4), (1, 3, 4, 2))
    x286 = np.zeros((navir, navir, navir, navir), dtype=np.float64)
    x286 += einsum(t1[np.ix_(so,sV)], (0, 1), x285, (2, 3, 4, 0), (2, 1, 4, 3))
    del x285
    x287 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x287 += einsum(x286, (0, 1, 2, 3), t3, (4, 5, 6, 7, 2, 3), (4, 6, 5, 1, 0, 7))
    t3new += einsum(x287, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x287, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    del x287
    x288 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x288 += einsum(x286, (0, 1, 2, 3), t3, (4, 5, 6, 2, 7, 3), (4, 6, 5, 0, 1, 7))
    del x286
    t3new += einsum(x288, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.5
    t3new += einsum(x288, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.5
    del x288
    x289 = np.zeros((navir, nocc), dtype=np.float64)
    x289 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovoV, (0, 1, 2, 3), (3, 2))
    x290 = np.zeros((navir, navir), dtype=np.float64)
    x290 += einsum(t1[np.ix_(so,sV)], (0, 1), x289, (2, 0), (1, 2))
    del x289
    t3new += einsum(x290, (0, 1), t3, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -2.0
    x291 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x291 += einsum(x290, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    del x290
    t3new += einsum(x291, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new += einsum(x291, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x291
    x292 = np.zeros((navir, nocc), dtype=np.float64)
    x292 += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovoV, (2, 1, 0, 3), (3, 2))
    x293 = np.zeros((navir, navir), dtype=np.float64)
    x293 += einsum(t1[np.ix_(so,sV)], (0, 1), x292, (2, 0), (1, 2))
    del x292
    t3new += einsum(x293, (0, 1), t3, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6))
    x294 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x294 += einsum(x293, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    del x293
    t3new += einsum(x294, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x294, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    del x294
    x295 = np.zeros((naocc, navir, nocc, nvir), dtype=np.float64)
    x295 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovov, (4, 5, 0, 2), (1, 3, 4, 5))
    x296 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x296 += einsum(t1[np.ix_(sO,sv)], (0, 1), x295, (2, 3, 4, 1), (0, 2, 3, 4))
    x297 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x297 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x296, (4, 5, 6, 0), (4, 1, 5, 3, 2, 6))
    del x296
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 2.0
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -2.0
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * 2.0
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -2.0
    t3new += einsum(x297, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    del x297
    x298 = np.zeros((naocc, navir, nocc, nvir), dtype=np.float64)
    x298 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovov, (4, 5, 0, 3), (1, 2, 4, 5))
    x299 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x299 += einsum(t1[np.ix_(sO,sv)], (0, 1), x298, (2, 3, 4, 1), (0, 2, 3, 4))
    x300 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x300 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x299, (4, 5, 6, 0), (4, 1, 5, 3, 2, 6))
    del x299
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x300, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    del x300
    x301 = np.zeros((naocc, navir, nocc, nvir), dtype=np.float64)
    x301 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.ovov, (4, 2, 0, 5), (1, 3, 4, 5))
    x302 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x302 += einsum(t1[np.ix_(sO,sv)], (0, 1), x301, (2, 3, 4, 1), (0, 2, 3, 4))
    x303 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x303 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x302, (4, 5, 6, 0), (4, 1, 5, 3, 2, 6))
    del x302
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x303, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x303
    x304 = np.zeros((naocc, navir, nocc, nvir), dtype=np.float64)
    x304 += einsum(t2[np.ix_(so,sO,sV,sv)], (0, 1, 2, 3), v.ovov, (4, 3, 0, 5), (1, 2, 4, 5))
    x305 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x305 += einsum(t1[np.ix_(sO,sv)], (0, 1), x304, (2, 3, 4, 1), (0, 2, 3, 4))
    x306 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x306 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x305, (4, 5, 6, 0), (4, 1, 5, 3, 2, 6))
    del x305
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3))
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4))
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    del x306
    x307 = np.zeros((naocc, naocc, naocc, navir, nocc, nocc), dtype=np.float64)
    x307 += einsum(t1[np.ix_(sO,sv)], (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 4, 5), v.ovov, (6, 1, 7, 4), (0, 3, 2, 5, 6, 7))
    x308 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x308 += einsum(t2[np.ix_(so,so,sV,sV)], (0, 1, 2, 3), x307, (4, 5, 6, 7, 1, 0), (4, 5, 6, 7, 3, 2))
    del x307
    t3new += einsum(x308, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 0.5
    t3new += einsum(x308, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -0.5
    t3new += einsum(x308, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -0.5
    t3new += einsum(x308, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 0.5
    t3new += einsum(x308, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x308, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x308, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x308, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x308
    x309 = np.zeros((naocc, naocc, naocc, navir, nocc, nocc), dtype=np.float64)
    x309 += einsum(t1[np.ix_(sO,sv)], (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 4, 5), v.ovov, (6, 1, 7, 4), (0, 3, 2, 5, 6, 7))
    x310 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x310 += einsum(t2[np.ix_(so,so,sV,sV)], (0, 1, 2, 3), x309, (4, 5, 6, 7, 1, 0), (4, 5, 6, 7, 3, 2))
    del x309
    t3new += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -0.5
    t3new += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 0.5
    t3new += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 0.5
    t3new += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -0.5
    del x310
    x311 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x311 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x295, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    x312 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x312 += einsum(t1[np.ix_(so,sV)], (0, 1), x311, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x311
    t3new += einsum(x312, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * 2.0
    t3new += einsum(x312, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new += einsum(x312, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 2.0
    t3new += einsum(x312, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    del x312
    x313 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x313 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x298, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    x314 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x314 += einsum(t1[np.ix_(so,sV)], (0, 1), x313, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x313
    t3new += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x314, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x314, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    del x314
    x315 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x315 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x295, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    del x295
    x316 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x316 += einsum(t1[np.ix_(so,sV)], (0, 1), x315, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x315
    t3new += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -2.0
    t3new += einsum(x316, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -2.0
    t3new += einsum(x316, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    del x316
    x317 = np.zeros((naocc, naocc, nocc, nocc), dtype=np.float64)
    x317 += einsum(t2[np.ix_(sO,sO,sv,sv)], (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x318 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x318 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x317, (4, 5, 0, 6), (5, 4, 1, 3, 2, 6))
    del x317
    x319 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x319 += einsum(t1[np.ix_(so,sV)], (0, 1), x318, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x318
    t3new += einsum(x319, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x319, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x319, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x319, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x319, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x319, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x319, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x319, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x319, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x319, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x319, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x319, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    del x319
    x320 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x320 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x298, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    del x298
    x321 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x321 += einsum(t1[np.ix_(so,sV)], (0, 1), x320, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x320
    t3new += einsum(x321, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x321, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x321, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x321, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x321, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x321, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    del x321
    x322 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x322 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x301, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    x323 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x323 += einsum(t1[np.ix_(so,sV)], (0, 1), x322, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x322
    t3new += einsum(x323, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x323, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x323, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x323, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x323, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x323, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    del x323
    x324 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x324 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x301, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    del x301
    x325 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x325 += einsum(t1[np.ix_(so,sV)], (0, 1), x324, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x324
    t3new += einsum(x325, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x325, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5))
    t3new += einsum(x325, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x325, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x325, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x325, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    del x325
    x326 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x326 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x304, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    x327 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x327 += einsum(t1[np.ix_(so,sV)], (0, 1), x326, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x326
    t3new += einsum(x327, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 4, 5))
    t3new += einsum(x327, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x327, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x327, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x327, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x327, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    del x327
    x328 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x328 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x304, (4, 5, 6, 2), (1, 0, 4, 3, 5, 6))
    del x304
    x329 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x329 += einsum(t1[np.ix_(so,sV)], (0, 1), x328, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x328
    t3new += einsum(x329, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x329, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x329, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x329, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4))
    t3new += einsum(x329, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x329, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    del x329
    x330 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x330 += einsum(x0, (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 1, 4), (3, 2, 4, 0))
    x331 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x331 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x330, (4, 5, 6, 0), (4, 5, 1, 6, 3, 2))
    del x330
    t3new += einsum(x331, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    t3new += einsum(x331, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new += einsum(x331, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new += einsum(x331, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new += einsum(x331, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -2.0
    t3new += einsum(x331, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    del x331
    x332 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x332 += einsum(x0, (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 1, 4), (3, 2, 4, 0))
    del x0
    x333 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x333 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x332, (4, 5, 6, 0), (4, 5, 1, 6, 3, 2))
    del x332
    t3new += einsum(x333, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * 2.0
    t3new += einsum(x333, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new += einsum(x333, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new += einsum(x333, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 2.0
    t3new += einsum(x333, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new += einsum(x333, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x333
    x334 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x334 += einsum(x16, (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 1, 4), (3, 2, 4, 0))
    x335 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x335 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x334, (4, 5, 6, 0), (4, 5, 1, 6, 3, 2))
    del x334
    t3new += einsum(x335, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x335, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x335, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x335, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x335, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x335, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    del x335
    x336 = np.zeros((naocc, naocc, navir, nocc), dtype=np.float64)
    x336 += einsum(x16, (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 1, 4), (3, 2, 4, 0))
    del x16
    x337 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x337 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x336, (4, 5, 6, 0), (4, 5, 1, 6, 3, 2))
    del x336
    t3new += einsum(x337, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x337, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x337, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x337, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x337, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x337, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    del x337
    x338 = np.zeros((naocc, naocc, navir, navir, nocc, nvir), dtype=np.float64)
    x338 += einsum(t1[np.ix_(sO,sv)], (0, 1), t2[np.ix_(so,sO,sV,sV)], (2, 3, 4, 5), v.ovov, (6, 7, 2, 1), (0, 3, 5, 4, 6, 7))
    x339 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x339 += einsum(t1[np.ix_(sO,sv)], (0, 1), x338, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x338
    x340 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x340 += einsum(t1[np.ix_(so,sV)], (0, 1), x339, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x339
    t3new += einsum(x340, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x340, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x340, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x340, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x340, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x340, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x340, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x340, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x340, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x340, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x340, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x340, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x340
    x341 = np.zeros((naocc, naocc, navir, navir, nocc, nvir), dtype=np.float64)
    x341 += einsum(t1[np.ix_(so,sV)], (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 4, 5), v.ovov, (6, 7, 0, 4), (3, 2, 1, 5, 6, 7))
    x342 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x342 += einsum(t1[np.ix_(sO,sv)], (0, 1), x341, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x341
    x343 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x343 += einsum(t1[np.ix_(so,sV)], (0, 1), x342, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x342
    t3new += einsum(x343, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x343, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x343, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new += einsum(x343, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x343, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x343, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    del x343
    x344 = np.zeros((naocc, naocc, navir, navir, nocc, nvir), dtype=np.float64)
    x344 += einsum(t1[np.ix_(so,sV)], (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 4, 5), v.ovov, (6, 7, 0, 4), (3, 2, 1, 5, 6, 7))
    x345 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=np.float64)
    x345 += einsum(t1[np.ix_(sO,sv)], (0, 1), x344, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x344
    x346 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=np.float64)
    x346 += einsum(t1[np.ix_(so,sV)], (0, 1), x345, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x345
    t3new += einsum(x346, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x346, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x346, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x346, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x346, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x346, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    del x346

    return {"t1new": t1new, "t2new": t2new, "t3new": t3new}

