# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(f.ov, (0, 1), t1, (0, 1), ())
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    e_cc += einsum(v.oovv, (0, 1, 2, 3), x0, (0, 1, 2, 3), ()) * 0.25
    del x0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t1new += einsum(t1, (0, 1), v.ovov, (2, 1, 0, 3), (2, 3)) * -1.0
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x0 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x1 += einsum(x0, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new += einsum(t2, (0, 1, 2, 3), x1, (4, 0, 1, 3), (4, 2)) * 0.5
    del x1
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x2 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    t1new += einsum(v.ovvv, (0, 1, 2, 3), x2, (0, 4, 2, 3), (4, 1)) * 0.5
    t2new += einsum(v.vvvv, (0, 1, 2, 3), x2, (4, 5, 2, 3), (5, 4, 0, 1)) * -0.5
    x3 = np.zeros((nocc, nvir), dtype=types[float])
    x3 += einsum(t1, (0, 1), v.oovv, (2, 0, 3, 1), (2, 3))
    x4 = np.zeros((nocc, nvir), dtype=types[float])
    x4 += einsum(f.ov, (0, 1), (0, 1))
    x4 += einsum(x3, (0, 1), (0, 1))
    del x3
    t1new += einsum(x4, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    x5 = np.zeros((nocc, nocc), dtype=types[float])
    x5 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 1), (2, 3))
    x6 = np.zeros((nocc, nocc), dtype=types[float])
    x6 += einsum(f.oo, (0, 1), (0, 1)) * 2.0
    x6 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2)) * 2.0
    x6 += einsum(x5, (0, 1), (0, 1)) * 2.0
    x6 += einsum(v.oovv, (0, 1, 2, 3), x2, (1, 4, 2, 3), (0, 4)) * -1.0
    t1new += einsum(t1, (0, 1), x6, (0, 2), (2, 1)) * -0.5
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum(t2, (0, 1, 2, 3), x8, (4, 1, 5, 3), (4, 0, 2, 5)) * -1.0
    del x8
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x12 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x12 += einsum(t2, (0, 1, 2, 3), x0, (4, 1, 5, 3), (4, 0, 5, 2))
    del x0
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x13 += einsum(x10, (0, 1, 2, 3), (0, 1, 2, 3))
    del x10
    x13 += einsum(x11, (0, 1, 2, 3), (0, 1, 2, 3))
    del x11
    x13 += einsum(x12, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x12
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum(t1, (0, 1), x13, (2, 0, 3, 4), (2, 3, 4, 1))
    del x13
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum(x7, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x7
    x15 += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    del x9
    x15 += einsum(x14, (0, 1, 2, 3), (0, 1, 3, 2))
    del x14
    t2new += einsum(x15, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x15, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x15, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new += einsum(x15, (0, 1, 2, 3), (1, 0, 3, 2))
    del x15
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x17 = np.zeros((nocc, nocc), dtype=types[float])
    x17 += einsum(t1, (0, 1), x4, (2, 1), (0, 2))
    x18 = np.zeros((nocc, nocc), dtype=types[float])
    x18 += einsum(f.oo, (0, 1), (0, 1))
    x18 += einsum(x17, (0, 1), (0, 1))
    del x17
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum(x18, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4)) * -1.0
    del x18
    x20 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x20 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum(x2, (0, 1, 2, 3), x20, (4, 0, 1, 5), (4, 5, 2, 3)) * 0.5
    del x20
    x22 = np.zeros((nocc, nocc), dtype=types[float])
    x22 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    x23 = np.zeros((nocc, nocc), dtype=types[float])
    x23 += einsum(x5, (0, 1), (0, 1))
    del x5
    x23 += einsum(x22, (0, 1), (1, 0)) * 0.5
    del x22
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum(x23, (0, 1), t2, (2, 0, 3, 4), (1, 2, 3, 4)) * -1.0
    del x23
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum(x16, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x16
    x25 += einsum(x19, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x19
    x25 += einsum(x21, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x21
    x25 += einsum(x24, (0, 1, 2, 3), (1, 0, 3, 2))
    del x24
    t2new += einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x25, (0, 1, 2, 3), (1, 0, 2, 3))
    del x25
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum(t2, (0, 1, 2, 3), x26, (4, 1, 5, 3), (4, 0, 5, 2))
    del x26
    x28 = np.zeros((nvir, nvir), dtype=types[float])
    x28 += einsum(t1, (0, 1), v.ovvv, (0, 2, 3, 1), (2, 3))
    x29 = np.zeros((nvir, nvir), dtype=types[float])
    x29 += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    x30 = np.zeros((nvir, nvir), dtype=types[float])
    x30 += einsum(x28, (0, 1), (0, 1))
    del x28
    x30 += einsum(x29, (0, 1), (0, 1)) * 0.5
    del x29
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum(x30, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    del x30
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x32 += einsum(v.ovvv, (0, 1, 2, 3), x2, (4, 5, 2, 3), (0, 4, 5, 1)) * 0.5
    x33 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x33 += einsum(x4, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    del x4
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum(x32, (0, 1, 2, 3), (0, 2, 1, 3))
    del x32
    x34 += einsum(x33, (0, 1, 2, 3), (2, 1, 0, 3))
    del x33
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum(t1, (0, 1), x34, (0, 2, 3, 4), (2, 3, 4, 1))
    del x34
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 += einsum(x27, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x27
    x36 += einsum(x31, (0, 1, 2, 3), (1, 0, 3, 2))
    del x31
    x36 += einsum(x35, (0, 1, 2, 3), (1, 0, 3, 2))
    del x35
    t2new += einsum(x36, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x36, (0, 1, 2, 3), (0, 1, 3, 2))
    del x36
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum(t1, (0, 1), v.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum(x37, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x37
    x39 += einsum(x38, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x38
    t2new += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x39, (0, 1, 2, 3), (0, 1, 3, 2))
    del x39
    x40 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x40 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x40 += einsum(v.oovv, (0, 1, 2, 3), x2, (4, 5, 2, 3), (0, 1, 5, 4)) * -0.5
    t2new += einsum(x2, (0, 1, 2, 3), x40, (0, 1, 4, 5), (4, 5, 3, 2)) * -0.5
    del x2, x40

    return {"t1new": t1new, "t2new": t2new}

def energy_perturbative(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    # T3 amplitude
    x0 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x0 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 1, 6), (0, 4, 5, 2, 3, 6))
    t3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3))
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -1.0
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5))
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3 += einsum(x0, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3))
    del x0
    x1 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=types[float])
    x1 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 3, 5, 6), (0, 1, 4, 2, 5, 6))
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -1.0
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3))
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4))
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -1.0
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -1.0
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    t3 += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    del x1
    e_ia = direct_sum("i-a->ia", np.diag(f.oo), np.diag(f.vv))
    t3 /= direct_sum("ia+jb+kc->ijkabc", e_ia, e_ia, e_ia)

    # energy
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x0 += einsum(l2, (0, 1, 2, 3), t3, (4, 5, 3, 6, 0, 1), (2, 4, 5, 6))
    e_pert = 0
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x0, (2, 0, 1, 3), ()) * 0.25
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum(v.ovvv, (0, 1, 2, 3), t3, (4, 5, 0, 6, 2, 3), (4, 5, 6, 1))
    e_pert += einsum(l2, (0, 1, 2, 3), x1, (3, 2, 1, 0), ()) * -0.25
    del x1
    x2 = np.zeros((nocc, nvir), dtype=types[float])
    x2 += einsum(v.oovv, (0, 1, 2, 3), t3, (4, 0, 1, 5, 2, 3), (4, 5))
    e_pert += einsum(l1, (0, 1), x2, (1, 0), ()) * 0.25
    del x2

    return e_pert

