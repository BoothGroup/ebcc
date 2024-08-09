# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 2, 3), ()) * 0.25

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # T amplitudes
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 5, 2, 3), (0, 1, 4, 5)) * 0.5
    t2new += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    t2new += einsum(x0, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x0, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new += einsum(x0, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    x2 = np.zeros((nocc, nocc), dtype=types[float])
    x2 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(x2, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4))
    del x2
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(x1, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x1
    x4 += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    del x3
    t2new += einsum(x4, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x4, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x4
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new += einsum(x5, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new += einsum(x5, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x5
    x6 = np.zeros((nvir, nvir), dtype=types[float])
    x6 += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(x6, (0, 1), t2, (2, 3, 4, 1), (2, 3, 4, 0))
    del x6
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(t2, (0, 1, 2, 3), x8, (4, 1, 5, 3), (4, 0, 5, 2))
    del x8
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x7
    x10 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    del x9
    t2new += einsum(x10, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x10
    x11 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x11 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x11 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (4, 5, 0, 1)) * 0.5
    t2new += einsum(t2, (0, 1, 2, 3), x11, (0, 1, 4, 5), (4, 5, 2, 3)) * 0.5
    del x11

    return {"t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, t2=None, l2=None, **kwargs):
    # L amplitudes
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum(l2, (0, 1, 2, 3), v.vvvv, (4, 5, 0, 1), (4, 5, 2, 3)) * 0.5
    l2new += einsum(v.oovv, (0, 1, 2, 3), (2, 3, 0, 1))
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x0 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    l2new += einsum(v.oovv, (0, 1, 2, 3), x0, (4, 5, 0, 1), (2, 3, 4, 5)) * 0.25
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x2 += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3))
    del x1
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(l2, (0, 1, 2, 3), x2, (3, 4, 1, 5), (2, 4, 0, 5))
    del x2
    l2new += einsum(x3, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new += einsum(x3, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new += einsum(x3, (0, 1, 2, 3), (2, 3, 1, 0)) * -1.0
    l2new += einsum(x3, (0, 1, 2, 3), (3, 2, 1, 0))
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(f.oo, (0, 1), l2, (2, 3, 4, 1), (0, 4, 2, 3))
    l2new += einsum(x4, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    l2new += einsum(x4, (0, 1, 2, 3), (3, 2, 1, 0))
    del x4
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(f.vv, (0, 1), l2, (2, 1, 3, 4), (3, 4, 0, 2))
    x6 = np.zeros((nvir, nvir), dtype=types[float])
    x6 += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(x6, (0, 1), l2, (2, 0, 3, 4), (3, 4, 2, 1))
    del x6
    x8 = np.zeros((nvir, nvir), dtype=types[float])
    x8 += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4))
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(x8, (0, 1), v.oovv, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    del x8
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum(x5, (0, 1, 2, 3), (1, 0, 2, 3))
    del x5
    x10 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x7
    x10 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x9
    l2new += einsum(x10, (0, 1, 2, 3), (2, 3, 0, 1))
    l2new += einsum(x10, (0, 1, 2, 3), (3, 2, 0, 1)) * -1.0
    del x10
    x11 = np.zeros((nocc, nocc), dtype=types[float])
    x11 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum(x11, (0, 1), l2, (2, 3, 4, 0), (4, 1, 2, 3))
    del x11
    x13 = np.zeros((nocc, nocc), dtype=types[float])
    x13 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(x13, (0, 1), v.oovv, (2, 1, 3, 4), (0, 2, 3, 4)) * -1.0
    del x13
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(x12, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    del x12
    x15 += einsum(x14, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.5
    del x14
    l2new += einsum(x15, (0, 1, 2, 3), (3, 2, 0, 1))
    l2new += einsum(x15, (0, 1, 2, 3), (3, 2, 1, 0)) * -1.0
    del x15
    x16 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x16 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x16 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 5, 2, 3), (0, 1, 4, 5)) * 0.5
    l2new += einsum(l2, (0, 1, 2, 3), x16, (3, 2, 4, 5), (0, 1, 4, 5)) * -0.5
    del x16

    return {"l2new": l2new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, t2=None, l2=None, **kwargs):
    delta = Namespace(oo=np.eye(nocc), vv=np.eye(nvir))

    # RDM1
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=types[float])
    rdm1_f_oo += einsum(delta.oo, (0, 1), (0, 1))
    rdm1_f_oo += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (4, 2)) * -0.5
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=types[float])
    rdm1_f_vv += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4)) * 0.5
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
    rdm2_f_oovv = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vvoo += einsum(l2, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 5), (0, 1, 4, 5)) * 0.5
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x0 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 0, 1), (2, 3, 4, 5))
    rdm2_f_oooo += einsum(x0, (0, 1, 2, 3), (3, 2, 1, 0)) * 0.5
    rdm2_f_oovv += einsum(t2, (0, 1, 2, 3), x0, (0, 1, 4, 5), (5, 4, 2, 3)) * -0.25
    del x0
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 0, 1), (2, 4))
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 1, 2)) * -0.5
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (0, 3, 2, 1)) * 0.5
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 1, 2)) * 0.5
    rdm2_f_oooo += einsum(delta.oo, (0, 1), x1, (2, 3), (3, 0, 2, 1)) * -0.5
    x2 = np.zeros((nvir, nvir), dtype=types[float])
    x2 += einsum(l2, (0, 1, 2, 3), t2, (2, 3, 4, 1), (0, 4))
    rdm2_f_ovov = np.zeros((nocc, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_ovov += einsum(delta.oo, (0, 1), x2, (2, 3), (0, 2, 1, 3)) * 0.5
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_ovvo += einsum(delta.oo, (0, 1), x2, (2, 3), (0, 2, 3, 1)) * -0.5
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_voov += einsum(delta.oo, (0, 1), x2, (2, 3), (2, 0, 1, 3)) * -0.5
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_vovo += einsum(delta.oo, (0, 1), x2, (2, 3), (2, 0, 3, 1)) * 0.5
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum(x2, (0, 1), t2, (2, 3, 4, 0), (2, 3, 4, 1))
    del x2
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum(l2, (0, 1, 2, 3), t2, (4, 3, 5, 1), (2, 4, 0, 5))
    rdm2_f_ovov += einsum(x4, (0, 1, 2, 3), (1, 2, 0, 3)) * -1.0
    rdm2_f_ovvo += einsum(x4, (0, 1, 2, 3), (1, 2, 3, 0))
    rdm2_f_voov += einsum(x4, (0, 1, 2, 3), (2, 1, 0, 3))
    rdm2_f_vovo += einsum(x4, (0, 1, 2, 3), (2, 1, 3, 0)) * -1.0
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum(t2, (0, 1, 2, 3), x4, (1, 4, 3, 5), (4, 0, 5, 2))
    del x4
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    del x3
    x6 += einsum(x5, (0, 1, 2, 3), (0, 1, 2, 3))
    del x5
    rdm2_f_oovv += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3))
    rdm2_f_oovv += einsum(x6, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(x1, (0, 1), t2, (2, 0, 3, 4), (2, 1, 3, 4))
    del x1
    rdm2_f_oovv += einsum(x7, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.5
    rdm2_f_oovv += einsum(x7, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.5
    del x7
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])

    rdm2_f = pack_2e(rdm2_f_oooo, rdm2_f_ooov, rdm2_f_oovo, rdm2_f_ovoo, rdm2_f_vooo, rdm2_f_oovv, rdm2_f_ovov, rdm2_f_ovvo, rdm2_f_voov, rdm2_f_vovo, rdm2_f_vvoo, rdm2_f_ovvv, rdm2_f_vovv, rdm2_f_vvov, rdm2_f_vvvo, rdm2_f_vvvv)

    rdm2_f = rdm2_f.swapaxes(1, 2)

    return rdm2_f

