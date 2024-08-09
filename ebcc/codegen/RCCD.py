# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -0.5
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), x0, (0, 1, 2, 3), ()) * 2.0
    del x0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # T amplitudes
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (4, 0, 5, 2)) * 2.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (4, 0, 3, 5)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (4, 0, 5, 2)) * -1.0
    t2new += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    del x1
    x3 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x2
    t2new += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x3, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x4 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(t2, (0, 1, 2, 3), x4, (1, 4, 5, 2), (0, 4, 3, 5))
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(t2, (0, 1, 2, 3), x5, (4, 1, 5, 3), (0, 4, 2, 5)) * 2.0
    del x5
    x7 = np.zeros((nvir, nvir), dtype=types[float])
    x7 += einsum(t2, (0, 1, 2, 3), x4, (0, 1, 3, 4), (2, 4))
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(x7, (0, 1), t2, (2, 3, 1, 4), (2, 3, 4, 0)) * 2.0
    del x7
    x9 = np.zeros((nocc, nocc), dtype=types[float])
    x9 += einsum(t2, (0, 1, 2, 3), x4, (1, 4, 2, 3), (0, 4))
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(x9, (0, 1), t2, (2, 1, 3, 4), (2, 0, 4, 3)) * 2.0
    del x9
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    del x6
    x11 += einsum(x8, (0, 1, 2, 3), (1, 0, 2, 3))
    del x8
    x11 += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2))
    del x10
    t2new += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x11, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x11
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x12 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x12 += einsum(t2, (0, 1, 2, 3), x4, (1, 4, 5, 3), (0, 4, 2, 5)) * 2.0
    del x4
    t2new += einsum(t2, (0, 1, 2, 3), x12, (4, 1, 5, 3), (0, 4, 2, 5)) * 2.0
    del x12
    x13 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x13 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x13 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (4, 0, 1, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x13, (0, 4, 5, 1), (5, 4, 3, 2))
    del x13
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x14 += einsum(x0, (0, 1, 2, 3), (0, 1, 2, 3))
    del x0
    t2new += einsum(t2, (0, 1, 2, 3), x14, (4, 1, 5, 2), (0, 4, 3, 5))
    del x14
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x15 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x15, (4, 1, 5, 2), (0, 4, 5, 3))
    del x15

    return {"t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, t2=None, l2=None, **kwargs):
    # L amplitudes
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum(v.ovov, (0, 1, 2, 3), (1, 3, 0, 2))
    l2new += einsum(l2, (0, 1, 2, 3), v.oooo, (4, 2, 5, 3), (0, 1, 4, 5))
    l2new += einsum(l2, (0, 1, 2, 3), v.vvvv, (4, 0, 5, 1), (4, 5, 2, 3))
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    l2new += einsum(l2, (0, 1, 2, 3), x0, (2, 3, 4, 5), (0, 1, 4, 5))
    del x0
    x1 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x1 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    l2new += einsum(v.ovov, (0, 1, 2, 3), x1, (4, 5, 0, 2), (1, 3, 4, 5))
    del x1
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum(f.oo, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3))
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(f.vv, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x5 += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3))
    del x4
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(l2, (0, 1, 2, 3), x5, (2, 4, 1, 5), (3, 4, 0, 5))
    del x5
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x7 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x8 = np.zeros((nvir, nvir), dtype=types[float])
    x8 += einsum(l2, (0, 1, 2, 3), x7, (2, 3, 1, 4), (0, 4))
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(x8, (0, 1), v.ovov, (2, 1, 3, 4), (2, 3, 0, 4)) * 2.0
    del x8
    x10 = np.zeros((nocc, nocc), dtype=types[float])
    x10 += einsum(l2, (0, 1, 2, 3), x7, (3, 4, 0, 1), (2, 4))
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(x10, (0, 1), v.ovov, (2, 3, 1, 4), (0, 2, 4, 3)) * 2.0
    del x10
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(x2, (0, 1, 2, 3), (0, 1, 2, 3))
    del x2
    x12 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x3
    x12 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x6
    x12 += einsum(x9, (0, 1, 2, 3), (1, 0, 2, 3))
    del x9
    x12 += einsum(x11, (0, 1, 2, 3), (0, 1, 3, 2))
    del x11
    l2new += einsum(x12, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new += einsum(x12, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    del x12
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 5, 1), (3, 4, 0, 5))
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x14 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(l2, (0, 1, 2, 3), x14, (3, 4, 1, 5), (2, 4, 0, 5)) * 0.5
    del x14
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    del x13
    x16 += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x15
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x17 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(x16, (0, 1, 2, 3), x17, (1, 4, 5, 3), (0, 4, 2, 5)) * 4.0
    del x16, x17
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * 2.0
    x19 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(l2, (0, 1, 2, 3), x19, (3, 4, 1, 5), (2, 4, 0, 5))
    del x19
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x22 += einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3))
    del x21
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(l2, (0, 1, 2, 3), x22, (2, 4, 1, 5), (3, 4, 0, 5))
    del x22
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    x24 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x25 = np.zeros((nvir, nvir), dtype=types[float])
    x25 += einsum(v.ovov, (0, 1, 2, 3), x24, (0, 2, 4, 3), (1, 4))
    del x24
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(x25, (0, 1), l2, (2, 1, 3, 4), (4, 3, 2, 0)) * 2.0
    del x25
    x27 = np.zeros((nocc, nocc), dtype=types[float])
    x27 += einsum(v.ovov, (0, 1, 2, 3), x7, (2, 4, 1, 3), (4, 0))
    del x7
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum(x27, (0, 1), l2, (2, 3, 0, 4), (4, 1, 2, 3)) * 2.0
    del x27
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum(x18, (0, 1, 2, 3), (0, 1, 2, 3))
    del x18
    x29 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x20
    x29 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x23
    x29 += einsum(x26, (0, 1, 2, 3), (1, 0, 2, 3))
    del x26
    x29 += einsum(x28, (0, 1, 2, 3), (0, 1, 3, 2))
    del x28
    l2new += einsum(x29, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    l2new += einsum(x29, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x29

    return {"l2new": l2new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, t2=None, l2=None, **kwargs):
    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))

    # RDM1
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=types[float])
    rdm1_f_oo += einsum(delta.oo, (0, 1), (0, 1)) * 2.0
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm1_f_oo += einsum(l2, (0, 1, 2, 3), x0, (3, 4, 0, 1), (4, 2)) * -2.0
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=types[float])
    rdm1_f_vv += einsum(l2, (0, 1, 2, 3), x1, (2, 3, 1, 4), (0, 4)) * 4.0
    del x1
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=types[float])
    rdm1_f_vo = np.zeros((nvir, nocc), dtype=types[float])

    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])

    return rdm1_f

def make_rdm2_f(f=None, v=None, nocc=None, nvir=None, t2=None, l2=None, **kwargs):
    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))

    # RDM2
    rdm2_f_oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 1, 3))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), delta.oo, (2, 3), (0, 2, 3, 1)) * -1.0
    rdm2_f_oovv = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2))
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 += einsum(l2, (0, 1, 2, 3), x0, (3, 4, 0, 1), (2, 4))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 2, 1)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 1, 2)) * -2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 2, 1)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 1, 2)) * 2.0
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 2, 1)) * -2.0
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_oooo += einsum(x2, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_oooo += einsum(x2, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_oooo += einsum(x2, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_oooo += einsum(x2, (0, 1, 2, 3), (3, 2, 1, 0))
    rdm2_f_oooo += einsum(x2, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    rdm2_f_oooo += einsum(x2, (0, 1, 2, 3), (3, 2, 1, 0))
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(t2, (0, 1, 2, 3), x2, (0, 1, 4, 5), (5, 4, 3, 2))
    del x2
    rdm2_f_oovv += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x4 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -0.5
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(t2, (0, 1, 2, 3), x4, (1, 4, 5, 3), (4, 0, 5, 2))
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(t2, (0, 1, 2, 3), x5, (1, 4, 3, 5), (4, 0, 5, 2)) * 4.0
    del x5
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    x7 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(t2, (0, 1, 2, 3), x7, (1, 4, 5, 2), (0, 4, 3, 5))
    del x7
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(t2, (0, 1, 2, 3), x8, (4, 1, 5, 2), (4, 0, 5, 3)) * -1.0
    del x8
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    del x3
    x10 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    del x6
    x10 += einsum(x9, (0, 1, 2, 3), (1, 0, 3, 2))
    del x9
    rdm2_f_oovv += einsum(x10, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x10, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x10
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum(t2, (0, 1, 2, 3), x4, (1, 4, 5, 2), (4, 0, 5, 3))
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(t2, (0, 1, 2, 3), x11, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x11
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * -0.5
    x13 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1))
    x14 = np.zeros((nvir, nvir), dtype=types[float])
    x14 += einsum(t2, (0, 1, 2, 3), x13, (0, 1, 4, 3), (2, 4))
    rdm2_f_ovov = np.zeros((nocc, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x14, (2, 3), (0, 3, 1, 2)) * 2.0
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x14, (2, 3), (0, 3, 1, 2)) * 2.0
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x14, (2, 3), (0, 3, 1, 2)) * 2.0
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x14, (2, 3), (0, 3, 1, 2)) * 2.0
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x14, (2, 3), (3, 0, 2, 1)) * 2.0
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x14, (2, 3), (3, 0, 2, 1)) * 2.0
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x14, (2, 3), (3, 0, 2, 1)) * 2.0
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x14, (2, 3), (3, 0, 2, 1)) * 2.0
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(x14, (0, 1), t2, (2, 3, 1, 4), (2, 3, 4, 0)) * 2.0
    del x14
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(x1, (0, 1), t2, (2, 0, 3, 4), (1, 2, 4, 3)) * 2.0
    del x1
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum(x12, (0, 1, 2, 3), (0, 1, 2, 3))
    del x12
    x17 += einsum(x15, (0, 1, 2, 3), (1, 0, 2, 3))
    del x15
    x17 += einsum(x16, (0, 1, 2, 3), (1, 0, 3, 2))
    del x16
    rdm2_f_oovv += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x17, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x17, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_oovv += einsum(x17, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x17, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x17, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    rdm2_f_oovv += einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    rdm2_f_oovv += einsum(x17, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x17, (0, 1, 2, 3), (1, 0, 2, 3))
    rdm2_f_oovv += einsum(x17, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x17
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 1, 5), (2, 4, 0, 5))
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(t2, (0, 1, 2, 3), x18, (1, 4, 2, 5), (4, 0, 5, 3))
    del x18
    rdm2_f_oovv += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3))
    del x19
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 1, 5), (3, 4, 0, 5))
    rdm2_f_ovov += einsum(x20, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x20, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_vovo += einsum(x20, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    rdm2_f_vovo += einsum(x20, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(t2, (0, 1, 2, 3), x20, (1, 4, 2, 5), (4, 0, 5, 3))
    del x20
    rdm2_f_oovv += einsum(x21, (0, 1, 2, 3), (0, 1, 3, 2))
    rdm2_f_oovv += einsum(x21, (0, 1, 2, 3), (0, 1, 3, 2))
    del x21
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum(t2, (0, 1, 2, 3), x4, (1, 4, 5, 3), (4, 0, 5, 2)) * 2.0
    del x4
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 += einsum(t2, (0, 1, 2, 3), x22, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    del x22
    rdm2_f_oovv += einsum(x23, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_oovv += einsum(x23, (0, 1, 2, 3), (1, 0, 3, 2))
    del x23
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x24 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(l2, (0, 1, 2, 3), x24, (2, 4, 5, 1), (3, 4, 0, 5))
    rdm2_f_ovov += einsum(x25, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x25, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    del x25
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(l2, (0, 1, 2, 3), x0, (3, 4, 5, 1), (2, 4, 0, 5)) * 2.0
    del x0
    rdm2_f_ovov += einsum(x26, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovov += einsum(x26, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_voov += einsum(x26, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_voov += einsum(x26, (0, 1, 2, 3), (2, 1, 0, 3))
    del x26
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1))
    x27 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum(t2, (0, 1, 2, 3), x27, (1, 4, 5, 2), (0, 4, 3, 5))
    del x27
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_ovvo += einsum(x28, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    rdm2_f_ovvo += einsum(x28, (0, 1, 2, 3), (0, 3, 2, 1)) * -1.0
    rdm2_f_vovo += einsum(x28, (0, 1, 2, 3), (3, 0, 2, 1))
    rdm2_f_vovo += einsum(x28, (0, 1, 2, 3), (3, 0, 2, 1))
    del x28
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum(l2, (0, 1, 2, 3), (3, 2, 0, 1)) * 2.0
    x29 += einsum(l2, (0, 1, 2, 3), (2, 3, 0, 1)) * -1.0
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum(t2, (0, 1, 2, 3), x29, (1, 4, 5, 3), (0, 4, 2, 5))
    del x29
    rdm2_f_ovvo += einsum(x30, (0, 1, 2, 3), (0, 3, 2, 1))
    rdm2_f_ovvo += einsum(x30, (0, 1, 2, 3), (0, 3, 2, 1))
    rdm2_f_vovo += einsum(x30, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    rdm2_f_vovo += einsum(x30, (0, 1, 2, 3), (3, 0, 2, 1)) * -1.0
    del x30
    x31 = np.zeros((nvir, nvir), dtype=types[float])
    x31 += einsum(t2, (0, 1, 2, 3), x13, (0, 1, 4, 3), (2, 4)) * 2.0
    del x13
    rdm2_f_ovvo += einsum(delta.oo, (0, 1), x31, (2, 3), (0, 3, 2, 1)) * -1.0
    rdm2_f_ovvo += einsum(delta.oo, (0, 1), x31, (2, 3), (0, 3, 2, 1)) * -1.0
    rdm2_f_voov += einsum(delta.oo, (0, 1), x31, (2, 3), (3, 0, 1, 2)) * -1.0
    rdm2_f_voov += einsum(delta.oo, (0, 1), x31, (2, 3), (3, 0, 1, 2)) * -1.0
    del x31
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum(l2, (0, 1, 2, 3), t2, (4, 2, 5, 1), (3, 4, 0, 5))
    rdm2_f_ovvo += einsum(x32, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_ovvo += einsum(x32, (0, 1, 2, 3), (1, 2, 3, 0)) * -1.0
    rdm2_f_voov += einsum(x32, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x32, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x32
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    x33 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x34 += einsum(l2, (0, 1, 2, 3), x33, (3, 4, 1, 5), (2, 4, 0, 5))
    del x33
    rdm2_f_ovvo += einsum(x34, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_ovvo += einsum(x34, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_voov += einsum(x34, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_voov += einsum(x34, (0, 1, 2, 3), (2, 1, 0, 3))
    del x34
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum(l2, (0, 1, 2, 3), x24, (2, 4, 1, 5), (3, 4, 0, 5))
    del x24
    rdm2_f_voov += einsum(x35, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    rdm2_f_voov += einsum(x35, (0, 1, 2, 3), (2, 1, 0, 3)) * -1.0
    del x35
    x36 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x36 += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 5), (0, 1, 4, 5))
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv += einsum(x36, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vvvv += einsum(x36, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vvvv += einsum(x36, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vvvv += einsum(x36, (0, 1, 2, 3), (1, 0, 3, 2))
    rdm2_f_vvvv += einsum(x36, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    rdm2_f_vvvv += einsum(x36, (0, 1, 2, 3), (1, 0, 3, 2))
    del x36
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])

    rdm2_f = pack_2e(rdm2_f_oooo, rdm2_f_ooov, rdm2_f_oovo, rdm2_f_ovoo, rdm2_f_vooo, rdm2_f_oovv, rdm2_f_ovov, rdm2_f_ovvo, rdm2_f_voov, rdm2_f_vovo, rdm2_f_vvoo, rdm2_f_ovvv, rdm2_f_vovv, rdm2_f_vvov, rdm2_f_vvvo, rdm2_f_vvvv)

    rdm2_f = rdm2_f.swapaxes(1, 2)

    return rdm2_f

