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

    # T amplitudes
    t1new = np.zeros(((nocc, nvir)), dtype=types[float])
    t1new[np.ix_(so,sv)] += einsum(f.ov, (0, 1), (0, 1))
    t1new[np.ix_(so,sv)] += einsum(f.oo, (0, 1), t1[np.ix_(so,sv)], (1, 2), (0, 2)) * -1.0
    t1new[np.ix_(so,sv)] += einsum(f.vv, (0, 1), t1[np.ix_(so,sv)], (2, 1), (2, 0))
    t1new[np.ix_(so,sv)] += einsum(f.ov, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 1), (2, 3))
    t1new[np.ix_(so,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), v.ovov, (2, 1, 0, 3), (2, 3)) * -1.0
    t1new[np.ix_(so,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ooov, (0, 1, 4, 3), (4, 2)) * -0.5
    t1new[np.ix_(so,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.ovvv, (1, 4, 2, 3), (0, 4)) * -0.5
    t1new[np.ix_(sO,sV)] += einsum(v.OOVV, (0, 1, 2, 3), t3, (4, 0, 1, 5, 2, 3), (4, 5)) * 0.25
    t2new = np.zeros(((nocc, nocc, nvir, nvir)), dtype=types[float])
    t2new[np.ix_(so,so,sv,sv)] += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(sO,sO,sV,sV)] += einsum(f.OV, (0, 1), t3, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oooo, (4, 5, 0, 1), (4, 5, 2, 3)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.vvvv, (4, 5, 2, 3), (0, 1, 4, 5)) * 0.5
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
    x12 = np.zeros((naocc, navir, navir, nocc), dtype=types[float])
    x12 += einsum(v.oVOO, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 1), (4, 5, 6, 0)) * -1.0
    t2new[np.ix_(so,sO,sV,sV)] += einsum(x12, (0, 1, 2, 3), (3, 0, 2, 1)) * 0.5
    t2new[np.ix_(sO,so,sV,sV)] += einsum(x12, (0, 1, 2, 3), (0, 3, 2, 1)) * -0.5
    del x12
    x13 = np.zeros((naocc, naocc, navir, nvir), dtype=types[float])
    x13 += einsum(v.vOVV, (0, 1, 2, 3), t3, (4, 5, 1, 6, 2, 3), (4, 5, 6, 0)) * -1.0
    t2new[np.ix_(sO,sO,sv,sV)] += einsum(x13, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    t2new[np.ix_(sO,sO,sV,sv)] += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x13
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(x0, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 4), (1, 2, 3, 4))
    del x0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x14, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x14, (0, 1, 2, 3), (1, 0, 3, 2))
    del x14
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum(f.ov, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (0, 2, 3, 4))
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(t1[np.ix_(so,sv)], (0, 1), x15, (0, 2, 3, 4), (2, 3, 1, 4))
    del x15
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
    del x18
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
    del x23
    t2new[np.ix_(so,so,sv,sv)] += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x24, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x24, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x24, (0, 1, 2, 3), (1, 0, 3, 2))
    del x24
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(x1, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    del x1
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
    del x28
    t2new[np.ix_(so,so,sv,sv)] += einsum(x29, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x29, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    del x29
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum(x2, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 4, 0)) * -1.0
    del x2
    t2new[np.ix_(so,so,sv,sv)] += einsum(x30, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3))
    del x30
    x31 = np.zeros((naocc, naocc, navir, nocc), dtype=types[float])
    x31 += einsum(t1[np.ix_(so,sv)], (0, 1), v.vVOO, (1, 2, 3, 4), (3, 4, 2, 0)) * -1.0
    x32 = np.zeros((naocc, navir, navir, nocc), dtype=types[float])
    x32 += einsum(x31, (0, 1, 2, 3), t3, (4, 0, 1, 5, 6, 2), (4, 5, 6, 3))
    del x31
    t2new[np.ix_(so,sO,sV,sV)] += einsum(x32, (0, 1, 2, 3), (3, 0, 2, 1)) * 0.5
    t2new[np.ix_(sO,so,sV,sV)] += einsum(x32, (0, 1, 2, 3), (0, 3, 2, 1)) * -0.5
    del x32
    x33 = np.zeros((naocc, naocc, navir, nocc), dtype=types[float])
    x33 += einsum(v.oOVV, (0, 1, 2, 3), t3, (4, 5, 1, 6, 2, 3), (4, 5, 6, 0)) * -1.0
    x34 = np.zeros((naocc, naocc, navir, nvir), dtype=types[float])
    x34 += einsum(t1[np.ix_(so,sv)], (0, 1), x33, (2, 3, 4, 0), (3, 2, 4, 1)) * -1.0
    del x33
    t2new[np.ix_(sO,sO,sv,sV)] += einsum(x34, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    t2new[np.ix_(sO,sO,sV,sv)] += einsum(x34, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x34
    x35 = np.zeros((naocc, navir), dtype=types[float])
    x35 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvV, (0, 2, 1, 3), (2, 3))
    t2new[np.ix_(sO,sO,sV,sV)] += einsum(x35, (0, 1), t3, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5))
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
    del x4
    t2new[np.ix_(so,so,sv,sv)] += einsum(x38, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x38, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    del x38
    x39 = np.zeros((nvir, nvir), dtype=types[float])
    x39 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum(x39, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 4, 0))
    del x39
    t2new[np.ix_(so,so,sv,sv)] += einsum(x40, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    t2new[np.ix_(so,so,sv,sv)] += einsum(x40, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    del x40
    x41 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x41 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (0, 1, 4, 5))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x41, (4, 5, 0, 1), (5, 4, 2, 3)) * -0.25
    x42 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x42 += einsum(t1[np.ix_(so,sv)], (0, 1), x21, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x21
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum(t1[np.ix_(so,sv)], (0, 1), x42, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x42
    t2new[np.ix_(so,so,sv,sv)] += einsum(x43, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x43, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x43
    x44 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x44 += einsum(t1[np.ix_(so,sv)], (0, 1), x26, (2, 3, 4, 1), (2, 0, 3, 4)) * -1.0
    del x26
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum(t1[np.ix_(so,sv)], (0, 1), x44, (2, 3, 0, 4), (3, 2, 1, 4)) * -1.0
    del x44
    t2new[np.ix_(so,so,sv,sv)] += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x45, (0, 1, 2, 3), (0, 1, 3, 2))
    del x45
    x46 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x46 += einsum(t1[np.ix_(so,sv)], (0, 1), x3, (2, 3, 4, 1), (2, 0, 4, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x46, (4, 5, 0, 1), (5, 4, 2, 3)) * -0.5
    x47 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x47 += einsum(t2[np.ix_(so,so,sv,sv)], (0, 1, 2, 3), x3, (4, 1, 5, 3), (4, 0, 5, 2))
    del x3
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum(t1[np.ix_(so,sv)], (0, 1), x47, (2, 3, 0, 4), (2, 3, 1, 4))
    del x47
    t2new[np.ix_(so,so,sv,sv)] += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x48, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x48, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x48, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x48
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum(x6, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 1, 3, 4), (0, 2, 3, 4)) * -1.0
    del x6
    t2new[np.ix_(so,so,sv,sv)] += einsum(x49, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new[np.ix_(so,so,sv,sv)] += einsum(x49, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x49
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x50 += einsum(t1[np.ix_(so,sv)], (0, 1), x41, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x41
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x50, (2, 3, 0, 4), (2, 3, 1, 4)) * 0.5
    del x50
    x51 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x51 += einsum(x5, (0, 1), t2[np.ix_(so,so,sv,sv)], (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    x52 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x52 += einsum(t1[np.ix_(so,sv)], (0, 1), x51, (2, 3, 0, 4), (2, 3, 1, 4))
    del x51
    t2new[np.ix_(so,so,sv,sv)] += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new[np.ix_(so,so,sv,sv)] += einsum(x52, (0, 1, 2, 3), (0, 1, 3, 2))
    del x52
    x53 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x53 += einsum(t1[np.ix_(so,sv)], (0, 1), x46, (2, 3, 4, 0), (2, 3, 4, 1))
    del x46
    t2new[np.ix_(so,so,sv,sv)] += einsum(t1[np.ix_(so,sv)], (0, 1), x53, (2, 3, 0, 4), (3, 2, 1, 4)) * -1.0
    del x53
    x54 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x54 += einsum(f.OO, (0, 1), t3, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    t3new = np.zeros(((naocc, naocc, naocc, navir, navir, navir)), dtype=types[float])
    t3new += einsum(x54, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x54, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x54, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x54
    x55 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x55 += einsum(f.VV, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x55, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    del x55
    x56 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x56 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), v.oVOO, (0, 4, 5, 6), (1, 5, 6, 2, 3, 4))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new += einsum(x56, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    del x56
    x57 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x57 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.vOVV, (2, 4, 5, 6), (0, 1, 4, 3, 5, 6))
    t3new += einsum(x57, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x57, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x57, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x57, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x57, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x57, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x57, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x57, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x57, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x57
    x58 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x58 += einsum(v.OOOO, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 7), (4, 0, 1, 5, 6, 7))
    t3new += einsum(x58, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 0.5
    t3new += einsum(x58, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -0.5
    t3new += einsum(x58, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.5
    del x58
    x59 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x59 += einsum(v.OVOV, (0, 1, 2, 3), t3, (4, 5, 2, 6, 7, 1), (4, 5, 0, 6, 7, 3))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x59, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x59
    x60 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x60 += einsum(v.VVVV, (0, 1, 2, 3), t3, (4, 5, 6, 7, 2, 3), (4, 5, 6, 7, 0, 1))
    t3new += einsum(x60, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -0.5
    t3new += einsum(x60, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 0.5
    t3new += einsum(x60, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -0.5
    del x60
    x61 = np.zeros((naocc, naocc), dtype=types[float])
    x61 += einsum(f.vO, (0, 1), t1[np.ix_(sO,sv)], (2, 0), (1, 2))
    x62 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x62 += einsum(x61, (0, 1), t3, (2, 3, 0, 4, 5, 6), (1, 2, 3, 4, 5, 6))
    del x61
    t3new += einsum(x62, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x62, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x62, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x62
    x63 = np.zeros((navir, navir), dtype=types[float])
    x63 += einsum(f.oV, (0, 1), t1[np.ix_(so,sV)], (0, 2), (1, 2))
    x64 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x64 += einsum(x63, (0, 1), t3, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x63
    t3new += einsum(x64, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x64, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x64, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    del x64
    x65 = np.zeros((naocc, naocc, navir, nocc), dtype=types[float])
    x65 += einsum(f.ov, (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    x66 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x66 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x65, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    del x65
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x66, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    del x66
    x67 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=types[float])
    x67 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), v.ooOO, (4, 0, 5, 6), (1, 5, 6, 2, 3, 4)) * -1.0
    x68 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x68 += einsum(t1[np.ix_(so,sV)], (0, 1), x67, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x67
    t3new += einsum(x68, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x68, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x68, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x68, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x68, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x68, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x68, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x68, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x68, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    del x68
    x69 = np.zeros((naocc, naocc, navir, nocc), dtype=types[float])
    x69 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.oVvO, (2, 3, 1, 4), (0, 4, 3, 2))
    x70 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x70 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x69, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x69
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3))
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x70, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    del x70
    x71 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=types[float])
    x71 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.oVvO, (4, 5, 2, 6), (0, 1, 6, 3, 5, 4)) * -1.0
    x72 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x72 += einsum(t1[np.ix_(so,sV)], (0, 1), x71, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x71
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x72, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    del x72
    x73 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=types[float])
    x73 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.vvVV, (4, 2, 5, 6), (0, 1, 3, 5, 6, 4)) * -1.0
    x74 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x74 += einsum(t1[np.ix_(sO,sv)], (0, 1), x73, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x73
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x74, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x74
    x75 = np.zeros((naocc, naocc, naocc, naocc), dtype=types[float])
    x75 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vOOO, (1, 2, 3, 4), (0, 3, 4, 2)) * -1.0
    x76 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x76 += einsum(x75, (0, 1, 2, 3), t3, (4, 2, 1, 5, 6, 7), (0, 4, 3, 5, 6, 7)) * -1.0
    del x75
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.5
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 0.5
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 0.5
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -0.5
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -0.5
    t3new += einsum(x76, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 0.5
    del x76
    x77 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x77 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oOOV, (0, 2, 3, 4), (3, 2, 1, 4)) * -1.0
    x78 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x78 += einsum(x77, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (4, 5, 0, 2, 6, 7))
    del x77
    t3new += einsum(x78, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x78, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x78, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x78, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x78, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x78, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x78, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    t3new += einsum(x78, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x78, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    del x78
    x79 = np.zeros((naocc, naocc), dtype=types[float])
    x79 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvO, (0, 2, 1, 3), (2, 3))
    x80 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x80 += einsum(x79, (0, 1), t3, (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6))
    del x79
    t3new += einsum(x80, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x80, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x80, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    del x80
    x81 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x81 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vVOV, (1, 2, 3, 4), (0, 3, 4, 2)) * -1.0
    x82 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x82 += einsum(x81, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 6, 7, 2))
    del x81
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new += einsum(x82, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    del x82
    x83 = np.zeros((navir, navir, navir, navir), dtype=types[float])
    x83 += einsum(t1[np.ix_(so,sV)], (0, 1), v.oVVV, (0, 2, 3, 4), (1, 3, 4, 2)) * -1.0
    x84 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x84 += einsum(x83, (0, 1, 2, 3), t3, (4, 5, 6, 7, 1, 2), (4, 5, 6, 0, 7, 3))
    del x83
    t3new += einsum(x84, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 0.5
    t3new += einsum(x84, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -0.5
    t3new += einsum(x84, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -0.5
    t3new += einsum(x84, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 0.5
    t3new += einsum(x84, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 0.5
    t3new += einsum(x84, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -0.5
    del x84
    x85 = np.zeros((navir, navir), dtype=types[float])
    x85 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oVvV, (0, 2, 1, 3), (2, 3))
    x86 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x86 += einsum(x85, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0))
    del x85
    t3new += einsum(x86, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x86, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x86, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    del x86
    x87 = np.zeros((naocc, naocc, navir, nocc), dtype=types[float])
    x87 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oovO, (4, 0, 2, 5), (1, 5, 3, 4)) * -1.0
    x88 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x88 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x87, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    del x87
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3))
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x88, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    del x88
    x89 = np.zeros((naocc, navir, navir, nvir), dtype=types[float])
    x89 += einsum(t2[np.ix_(so,so,sV,sV)], (0, 1, 2, 3), v.oovO, (0, 1, 4, 5), (5, 2, 3, 4)) * -1.0
    x90 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x90 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x89, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x89
    t3new += einsum(x90, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 0.5
    t3new += einsum(x90, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -0.5
    t3new += einsum(x90, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 0.5
    t3new += einsum(x90, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -0.5
    t3new += einsum(x90, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 0.5
    t3new += einsum(x90, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -0.5
    t3new += einsum(x90, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -0.5
    t3new += einsum(x90, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 0.5
    t3new += einsum(x90, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -0.5
    del x90
    x91 = np.zeros((naocc, navir, navir, nvir), dtype=types[float])
    x91 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oVvv, (0, 4, 5, 2), (1, 3, 4, 5)) * -1.0
    x92 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x92 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x91, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x91
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x92, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    del x92
    x93 = np.zeros((naocc, naocc, navir, nocc), dtype=types[float])
    x93 += einsum(t2[np.ix_(sO,sO,sv,sv)], (0, 1, 2, 3), v.oVvv, (4, 5, 2, 3), (0, 1, 5, 4)) * -1.0
    x94 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x94 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x93, (4, 5, 6, 0), (5, 4, 1, 2, 3, 6))
    del x93
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 0.5
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.5
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 0.5
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 0.5
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -0.5
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 0.5
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -0.5
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 0.5
    t3new += einsum(x94, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -0.5
    del x94
    x95 = np.zeros((naocc, naocc, naocc, naocc, navir, navir), dtype=types[float])
    x95 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), v.vVOO, (2, 4, 5, 6), (0, 1, 5, 6, 3, 4))
    x96 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x96 += einsum(t3, (0, 1, 2, 3, 4, 5), x95, (6, 7, 1, 2, 8, 5), (7, 6, 0, 8, 3, 4)) * -1.0
    del x95
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 0.5
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -0.5
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 0.5
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -0.5
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 0.5
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -0.5
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -0.5
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 0.5
    t3new += einsum(x96, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -0.5
    del x96
    x97 = np.zeros((naocc, naocc, naocc, naocc), dtype=types[float])
    x97 += einsum(t2[np.ix_(sO,sO,sv,sv)], (0, 1, 2, 3), v.vvOO, (2, 3, 4, 5), (0, 1, 4, 5))
    x98 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x98 += einsum(x97, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 7), (1, 0, 4, 5, 6, 7)) * -1.0
    del x97
    t3new += einsum(x98, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 0.25
    t3new += einsum(x98, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.25
    t3new += einsum(x98, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -0.25
    del x98
    x99 = np.zeros((naocc, naocc, navir, navir, navir, navir), dtype=types[float])
    x99 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), v.oOVV, (0, 4, 5, 6), (1, 4, 2, 3, 5, 6))
    x100 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x100 += einsum(t3, (0, 1, 2, 3, 4, 5), x99, (6, 2, 7, 8, 4, 5), (6, 0, 1, 7, 8, 3))
    del x99
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -0.5
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 0.5
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -0.5
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -0.5
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 0.5
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -0.5
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 0.5
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -0.5
    t3new += einsum(x100, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 0.5
    del x100
    x101 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x101 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oOvV, (0, 4, 2, 5), (1, 4, 3, 5))
    x102 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x102 += einsum(x101, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x101
    t3new += einsum(x102, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 1.00000000000001
    t3new += einsum(x102, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.00000000000001
    t3new += einsum(x102, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 1.00000000000001
    t3new += einsum(x102, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 1.00000000000001
    t3new += einsum(x102, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.00000000000001
    t3new += einsum(x102, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 1.00000000000001
    t3new += einsum(x102, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.00000000000001
    t3new += einsum(x102, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 1.00000000000001
    t3new += einsum(x102, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.00000000000001
    del x102
    x103 = np.zeros((naocc, naocc), dtype=types[float])
    x103 += einsum(t2[np.ix_(so,sO,sv,sv)], (0, 1, 2, 3), v.oOvv, (0, 4, 2, 3), (1, 4))
    x104 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x104 += einsum(x103, (0, 1), t3, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    del x103
    t3new += einsum(x104, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 0.5
    t3new += einsum(x104, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 0.5
    t3new += einsum(x104, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -0.5
    del x104
    x105 = np.zeros((navir, navir, navir, navir), dtype=types[float])
    x105 += einsum(t2[np.ix_(so,so,sV,sV)], (0, 1, 2, 3), v.ooVV, (0, 1, 4, 5), (2, 3, 4, 5))
    x106 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x106 += einsum(x105, (0, 1, 2, 3), t3, (4, 5, 6, 7, 3, 2), (4, 5, 6, 1, 0, 7))
    del x105
    t3new += einsum(x106, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -0.25
    t3new += einsum(x106, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 0.25
    t3new += einsum(x106, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -0.25
    del x106
    x107 = np.zeros((navir, navir), dtype=types[float])
    x107 += einsum(t2[np.ix_(so,so,sv,sV)], (0, 1, 2, 3), v.oovV, (0, 1, 2, 4), (3, 4))
    x108 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x108 += einsum(x107, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    del x107
    t3new += einsum(x108, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 0.5
    t3new += einsum(x108, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -0.5
    t3new += einsum(x108, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 0.5
    del x108
    x109 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=types[float])
    x109 += einsum(t1[np.ix_(so,sV)], (0, 1), t2[np.ix_(so,sO,sV,sV)], (2, 3, 4, 5), v.oovO, (2, 0, 6, 7), (3, 7, 1, 4, 5, 6)) * -1.0
    x110 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x110 += einsum(t1[np.ix_(sO,sv)], (0, 1), x109, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x109
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4))
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3))
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    t3new += einsum(x110, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x110
    x111 = np.zeros((naocc, navir, navir, nvir), dtype=types[float])
    x111 += einsum(t1[np.ix_(so,sV)], (0, 1), t1[np.ix_(so,sV)], (2, 3), v.oovO, (0, 2, 4, 5), (5, 1, 3, 4)) * -1.0
    x112 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x112 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x111, (4, 5, 6, 2), (0, 1, 4, 6, 5, 3))
    del x111
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x112, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x112
    x113 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=types[float])
    x113 += einsum(t1[np.ix_(sO,sv)], (0, 1), t2[np.ix_(so,sO,sV,sV)], (2, 3, 4, 5), v.oVvv, (2, 6, 7, 1), (0, 3, 4, 5, 6, 7))
    x114 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x114 += einsum(t1[np.ix_(sO,sv)], (0, 1), x113, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x113
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x114, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    del x114
    x115 = np.zeros((naocc, naocc, navir, navir, navir, nvir), dtype=types[float])
    x115 += einsum(t1[np.ix_(so,sV)], (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 4, 5), v.oVvv, (0, 6, 7, 4), (2, 3, 1, 5, 6, 7))
    x116 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x116 += einsum(t1[np.ix_(sO,sv)], (0, 1), x115, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x115
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4))
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -1.0
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5))
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4))
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x116, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x116
    x117 = np.zeros((naocc, naocc, naocc, nvir), dtype=types[float])
    x117 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.vvOO, (2, 1, 3, 4), (0, 3, 4, 2))
    x118 = np.zeros((naocc, naocc, naocc, naocc), dtype=types[float])
    x118 += einsum(t1[np.ix_(sO,sv)], (0, 1), x117, (2, 3, 4, 1), (2, 0, 4, 3))
    del x117
    x119 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x119 += einsum(x118, (0, 1, 2, 3), t3, (4, 2, 3, 5, 6, 7), (1, 0, 4, 5, 6, 7)) * -1.0
    del x118
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 0.5
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -0.5
    t3new += einsum(x119, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -0.5
    del x119
    x120 = np.zeros((naocc, naocc, navir, nocc), dtype=types[float])
    x120 += einsum(t1[np.ix_(sO,sv)], (0, 1), v.oOvV, (2, 3, 1, 4), (0, 3, 4, 2))
    x121 = np.zeros((naocc, naocc, navir, navir), dtype=types[float])
    x121 += einsum(t1[np.ix_(so,sV)], (0, 1), x120, (2, 3, 4, 0), (2, 3, 1, 4))
    del x120
    x122 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x122 += einsum(x121, (0, 1, 2, 3), t3, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x121
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x122, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    del x122
    x123 = np.zeros((naocc, nvir), dtype=types[float])
    x123 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oOvv, (0, 2, 3, 1), (2, 3)) * -1.0
    x124 = np.zeros((naocc, naocc), dtype=types[float])
    x124 += einsum(t1[np.ix_(sO,sv)], (0, 1), x123, (2, 1), (0, 2))
    del x123
    x125 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x125 += einsum(x124, (0, 1), t3, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    del x124
    t3new += einsum(x125, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x125, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x125, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x125
    x126 = np.zeros((navir, navir, navir, nocc), dtype=types[float])
    x126 += einsum(t1[np.ix_(so,sV)], (0, 1), v.ooVV, (2, 0, 3, 4), (1, 3, 4, 2))
    x127 = np.zeros((navir, navir, navir, navir), dtype=types[float])
    x127 += einsum(t1[np.ix_(so,sV)], (0, 1), x126, (2, 3, 4, 0), (2, 1, 4, 3))
    del x126
    x128 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x128 += einsum(x127, (0, 1, 2, 3), t3, (4, 5, 6, 7, 2, 3), (4, 5, 6, 1, 0, 7)) * -1.0
    del x127
    t3new += einsum(x128, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -0.5
    t3new += einsum(x128, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 0.5
    t3new += einsum(x128, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -0.5
    del x128
    x129 = np.zeros((navir, nocc), dtype=types[float])
    x129 += einsum(t1[np.ix_(so,sv)], (0, 1), v.oovV, (2, 0, 1, 3), (3, 2)) * -1.0
    x130 = np.zeros((navir, navir), dtype=types[float])
    x130 += einsum(t1[np.ix_(so,sV)], (0, 1), x129, (2, 0), (1, 2))
    del x129
    x131 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x131 += einsum(x130, (0, 1), t3, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    del x130
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x131, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    del x131
    x132 = np.zeros((naocc, navir, nocc, nvir), dtype=types[float])
    x132 += einsum(t2[np.ix_(so,sO,sv,sV)], (0, 1, 2, 3), v.oovv, (4, 0, 5, 2), (1, 3, 4, 5))
    x133 = np.zeros((naocc, naocc, navir, nocc), dtype=types[float])
    x133 += einsum(t1[np.ix_(sO,sv)], (0, 1), x132, (2, 3, 4, 1), (0, 2, 3, 4))
    x134 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x134 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x133, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x133
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -1.0
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3))
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -1.0
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -1.0
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x134, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    del x134
    x135 = np.zeros((naocc, naocc, naocc, navir, nocc, nocc), dtype=types[float])
    x135 += einsum(t1[np.ix_(sO,sv)], (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 4, 5), v.oovv, (6, 7, 4, 1), (0, 2, 3, 5, 6, 7))
    x136 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x136 += einsum(t2[np.ix_(so,so,sV,sV)], (0, 1, 2, 3), x135, (4, 5, 6, 7, 1, 0), (4, 5, 6, 7, 2, 3)) * -1.0
    del x135
    t3new += einsum(x136, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -0.5
    t3new += einsum(x136, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 0.5
    t3new += einsum(x136, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -0.5
    t3new += einsum(x136, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -0.5
    t3new += einsum(x136, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 0.5
    t3new += einsum(x136, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -0.5
    t3new += einsum(x136, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 0.5
    t3new += einsum(x136, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -0.5
    t3new += einsum(x136, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 0.5
    del x136
    x137 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=types[float])
    x137 += einsum(t2[np.ix_(sO,sO,sv,sV)], (0, 1, 2, 3), x132, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x132
    x138 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x138 += einsum(t1[np.ix_(so,sV)], (0, 1), x137, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x137
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -1.0
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -1.0
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5))
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5))
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3))
    t3new += einsum(x138, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    del x138
    x139 = np.zeros((naocc, naocc, nocc, nocc), dtype=types[float])
    x139 += einsum(t2[np.ix_(sO,sO,sv,sv)], (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (0, 1, 4, 5))
    x140 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=types[float])
    x140 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x139, (4, 5, 0, 6), (4, 5, 1, 2, 3, 6))
    del x139
    x141 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x141 += einsum(t1[np.ix_(so,sV)], (0, 1), x140, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x140
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -0.5
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -0.5
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 0.5
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 0.5
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 0.5
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -0.5
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -0.5
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -0.5
    t3new += einsum(x141, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 0.5
    del x141
    x142 = np.zeros((naocc, naocc, navir, nocc), dtype=types[float])
    x142 += einsum(x5, (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 1, 4), (2, 3, 4, 0))
    del x5
    x143 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x143 += einsum(t2[np.ix_(so,sO,sV,sV)], (0, 1, 2, 3), x142, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3)) * -1.0
    del x142
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -1.0
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3))
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -1.0
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3))
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4))
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -1.0
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    t3new += einsum(x143, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -1.0
    del x143
    x144 = np.zeros((naocc, naocc, navir, navir, nocc, nvir), dtype=types[float])
    x144 += einsum(t1[np.ix_(sO,sv)], (0, 1), t2[np.ix_(so,sO,sV,sV)], (2, 3, 4, 5), v.oovv, (6, 2, 7, 1), (0, 3, 4, 5, 6, 7)) * -1.0
    x145 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=types[float])
    x145 += einsum(t1[np.ix_(sO,sv)], (0, 1), x144, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6)) * -1.0
    del x144
    x146 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x146 += einsum(t1[np.ix_(so,sV)], (0, 1), x145, (2, 3, 4, 5, 6, 0), (3, 2, 4, 1, 5, 6)) * -1.0
    del x145
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3new += einsum(x146, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x146
    x147 = np.zeros((naocc, naocc, navir, navir, nocc, nvir), dtype=types[float])
    x147 += einsum(t1[np.ix_(so,sV)], (0, 1), t2[np.ix_(sO,sO,sv,sV)], (2, 3, 4, 5), v.oovv, (6, 0, 7, 4), (2, 3, 1, 5, 6, 7)) * -1.0
    x148 = np.zeros((naocc, naocc, naocc, navir, navir, nocc), dtype=types[float])
    x148 += einsum(t1[np.ix_(sO,sv)], (0, 1), x147, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x147
    x149 = np.zeros((naocc, naocc, naocc, navir, navir, navir), dtype=types[float])
    x149 += einsum(t1[np.ix_(so,sV)], (0, 1), x148, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6)) * -1.0
    del x148
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new += einsum(x149, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    del x149

    return {"t1new": t1new, "t2new": t2new, "t3new": t3new}

