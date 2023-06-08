# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, direct_sum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(f.ov, (0, 1), t1, (0, 1), ())
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    e_cc += einsum(v.oovv, (0, 1, 2, 3), x0, (0, 1, 2, 3), ()) * 0.25
    del x0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=np.float64)
    t1new += einsum(t1, (0, 1), v.ovov, (2, 1, 0, 3), (2, 3)) * -1.0
    t1new += einsum(f.ov, (0, 1), (0, 1))
    t1new += einsum(f.vv, (0, 1), t1, (2, 1), (2, 0))
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2new += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x0 += einsum(t1, (0, 1), v.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x1 += einsum(v.ooov, (0, 1, 2, 3), (2, 0, 1, 3)) * -1.0
    x1 += einsum(x0, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    t1new += einsum(t2, (0, 1, 2, 3), x1, (4, 0, 1, 3), (4, 2)) * 0.5
    del x1
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x2 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x2 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * 2.0
    t1new += einsum(v.ovvv, (0, 1, 2, 3), x2, (0, 4, 2, 3), (4, 1)) * 0.5
    x3 = np.zeros((nocc, nvir), dtype=np.float64)
    x3 += einsum(t1, (0, 1), v.oovv, (2, 0, 3, 1), (2, 3))
    x4 = np.zeros((nocc, nvir), dtype=np.float64)
    x4 += einsum(f.ov, (0, 1), (0, 1))
    x4 += einsum(x3, (0, 1), (0, 1))
    del x3
    t1new += einsum(x4, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    x5 = np.zeros((nocc, nocc), dtype=np.float64)
    x5 += einsum(t1, (0, 1), v.ooov, (2, 0, 3, 1), (2, 3))
    x6 = np.zeros((nocc, nocc), dtype=np.float64)
    x6 += einsum(f.oo, (0, 1), (0, 1))
    x6 += einsum(f.ov, (0, 1), t1, (2, 1), (0, 2))
    x6 += einsum(x5, (0, 1), (0, 1))
    x6 += einsum(v.oovv, (0, 1, 2, 3), x2, (1, 4, 2, 3), (0, 4)) * -0.5
    t1new += einsum(t1, (0, 1), x6, (0, 2), (2, 1)) * -1.0
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x7 += einsum(t1, (0, 1), v.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x8 = np.zeros((nocc, nocc), dtype=np.float64)
    x8 += einsum(t1, (0, 1), x4, (2, 1), (0, 2))
    x9 = np.zeros((nocc, nocc), dtype=np.float64)
    x9 += einsum(f.oo, (0, 1), (0, 1))
    x9 += einsum(x8, (0, 1), (0, 1))
    del x8
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x10 += einsum(x9, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4)) * -1.0
    del x9
    x11 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x11 += einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x12 += einsum(x11, (0, 1, 2, 3), x2, (1, 2, 4, 5), (0, 3, 4, 5)) * 0.5
    del x11
    x13 = np.zeros((nocc, nocc), dtype=np.float64)
    x13 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    x14 = np.zeros((nocc, nocc), dtype=np.float64)
    x14 += einsum(x5, (0, 1), (0, 1))
    del x5
    x14 += einsum(x13, (0, 1), (1, 0)) * 0.5
    del x13
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x15 += einsum(x14, (0, 1), t2, (2, 0, 3, 4), (1, 2, 3, 4)) * -1.0
    del x14
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x16 += einsum(x7, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x7
    x16 += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x10
    x16 += einsum(x12, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x12
    x16 += einsum(x15, (0, 1, 2, 3), (1, 0, 3, 2))
    del x15
    t2new += einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x16, (0, 1, 2, 3), (1, 0, 2, 3))
    del x16
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x17 += einsum(t1, (0, 1), v.ooov, (2, 3, 0, 4), (2, 3, 1, 4))
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x18 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x19 += einsum(x17, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x17
    x19 += einsum(x18, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x18
    t2new += einsum(x19, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x19, (0, 1, 2, 3), (0, 1, 3, 2))
    del x19
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x20 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x21 += einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x22 += einsum(t2, (0, 1, 2, 3), x21, (4, 1, 5, 3), (4, 0, 2, 5)) * -1.0
    del x21
    x23 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x23 += einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x24 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x25 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x25 += einsum(t2, (0, 1, 2, 3), x0, (4, 1, 5, 3), (4, 0, 5, 2))
    del x0
    x26 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x26 += einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3))
    del x23
    x26 += einsum(x24, (0, 1, 2, 3), (0, 1, 2, 3))
    del x24
    x26 += einsum(x25, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    del x25
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x27 += einsum(t1, (0, 1), x26, (2, 0, 3, 4), (2, 3, 4, 1))
    del x26
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x28 += einsum(x20, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x20
    x28 += einsum(x22, (0, 1, 2, 3), (0, 1, 2, 3))
    del x22
    x28 += einsum(x27, (0, 1, 2, 3), (0, 1, 3, 2))
    del x27
    t2new += einsum(x28, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x28, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new += einsum(x28, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new += einsum(x28, (0, 1, 2, 3), (1, 0, 3, 2))
    del x28
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x29 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x30 += einsum(t2, (0, 1, 2, 3), x29, (4, 1, 5, 3), (4, 0, 5, 2))
    del x29
    x31 = np.zeros((nvir, nvir), dtype=np.float64)
    x31 += einsum(t1, (0, 1), v.ovvv, (0, 2, 3, 1), (2, 3))
    x32 = np.zeros((nvir, nvir), dtype=np.float64)
    x32 += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    x33 = np.zeros((nvir, nvir), dtype=np.float64)
    x33 += einsum(x31, (0, 1), (0, 1))
    del x31
    x33 += einsum(x32, (0, 1), (0, 1)) * 0.5
    del x32
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x34 += einsum(x33, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    del x33
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x35 += einsum(v.ovvv, (0, 1, 2, 3), x2, (4, 5, 2, 3), (0, 4, 5, 1)) * 0.5
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x36 += einsum(x4, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4)) * -1.0
    del x4
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x37 += einsum(x35, (0, 1, 2, 3), (0, 2, 1, 3))
    del x35
    x37 += einsum(x36, (0, 1, 2, 3), (2, 1, 0, 3))
    del x36
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x38 += einsum(t1, (0, 1), x37, (0, 2, 3, 4), (2, 3, 4, 1))
    del x37
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x39 += einsum(x30, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    del x30
    x39 += einsum(x34, (0, 1, 2, 3), (1, 0, 3, 2))
    del x34
    x39 += einsum(x38, (0, 1, 2, 3), (1, 0, 3, 2))
    del x38
    t2new += einsum(x39, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x39, (0, 1, 2, 3), (0, 1, 3, 2))
    del x39
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x40 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.5
    x40 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    t2new += einsum(v.vvvv, (0, 1, 2, 3), x40, (4, 5, 2, 3), (5, 4, 0, 1)) * -1.0
    del x40
    x41 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x41 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x41 += einsum(v.oovv, (0, 1, 2, 3), x2, (4, 5, 2, 3), (0, 1, 5, 4)) * -0.5
    t2new += einsum(x2, (0, 1, 2, 3), x41, (0, 1, 4, 5), (4, 5, 3, 2)) * -0.5
    del x2, x41

    return {"t1new": t1new, "t2new": t2new}

def energy_perturbative(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    e_ia = direct_sum("i-a->ia", np.diag(f.oo), np.diag(f.vv))
    denom3 = 1 / direct_sum("ia+jb+kc->ijkabc", e_ia, e_ia, e_ia)

    # energy
    x0 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x0 += einsum(l1, (0, 1), t2, (2, 3, 4, 0), v.oovv, (2, 3, 5, 6), denom3, (3, 2, 1, 5, 0, 6), (1, 5, 6, 4))
    e_pert = 0
    e_pert += einsum(v.ovvv, (0, 1, 2, 3), x0, (0, 2, 3, 1), ()) * 0.25
    del x0
    x1 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x1 += einsum(l2, (0, 1, 2, 3), v.ovvv, (4, 1, 5, 6), (2, 3, 4, 0, 5, 6))
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x2 += einsum(t2, (0, 1, 2, 3), denom3, (4, 1, 5, 6, 2, 3), x1, (5, 1, 4, 2, 6, 3), (5, 4, 0, 6))
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x2, (0, 1, 2, 3), ()) * -1.0
    del x2
    x3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x3 += einsum(l1, (0, 1), v.oovv, (2, 3, 4, 5), (1, 2, 3, 0, 4, 5)) * -1.0
    x3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4))
    x3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 0.5
    x3 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 0.25
    x3 += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 0.5
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x4 += einsum(v.ooov, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), denom3, (0, 5, 1, 3, 4, 6), (5, 2, 4, 6))
    x4 += einsum(v.oovv, (0, 1, 2, 3), v.ovvv, (1, 4, 2, 3), denom3, (1, 0, 5, 3, 6, 2), (5, 0, 4, 6))
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x5 += einsum(v.ovvv, (0, 1, 2, 3), denom3, (0, 4, 5, 2, 6, 3), x3, (4, 0, 5, 3, 2, 6), (5, 4, 6, 1)) * 4.0
    del x3
    x5 += einsum(l1, (0, 1), x4, (1, 2, 3, 0), (1, 2, 0, 3)) * 2.0
    del x4
    e_pert += einsum(t2, (0, 1, 2, 3), x5, (0, 1, 2, 3), ()) * -0.25
    del x5
    x6 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x6 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), denom3, (1, 4, 5, 2, 6, 3), (4, 5, 0, 6))
    x7 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    x7 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (2, 3, 4, 5, 0, 1, 6, 7))
    x7 += einsum(l2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (4, 5, 2, 3, 6, 7, 0, 1))
    x8 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x8 += einsum(l2, (0, 1, 2, 3), v.ooov, (4, 5, 3, 6), (2, 4, 5, 0, 1, 6))
    x9 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x9 += einsum(x8, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    x9 += einsum(x8, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -0.5
    x9 += einsum(x8, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    del x8
    x9 += einsum(x1, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4))
    x9 += einsum(x1, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4))
    del x1
    x10 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x10 += einsum(t2, (0, 1, 2, 3), v.ooov, (4, 5, 1, 6), (0, 4, 5, 2, 3, 6))
    x11 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x11 += einsum(t2, (0, 1, 2, 3), v.ovvv, (4, 3, 5, 6), (0, 1, 4, 2, 5, 6))
    x12 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x12 += einsum(x10, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -1.0
    x12 += einsum(x11, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    x12 += einsum(x11, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    x12 += einsum(x11, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -1.0
    x13 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x13 += einsum(t2, (0, 1, 2, 3), x9, (0, 4, 5, 6, 3, 2), (0, 1, 5, 4, 2, 3, 6))
    del x9
    x13 += einsum(l2, (0, 1, 2, 3), x12, (4, 2, 5, 1, 0, 6), (2, 3, 4, 5, 0, 1, 6))
    del x12
    x14 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x14 += einsum(l1, (0, 1), x6, (2, 1, 3, 0), (1, 2, 3, 0))
    del x6
    x14 += einsum(v.ovvv, (0, 1, 2, 3), denom3, (0, 4, 5, 2, 6, 3), x7, (0, 7, 5, 4, 2, 3, 1, 6), (5, 4, 7, 6)) * -0.5
    del x7
    x14 += einsum(denom3, (0, 1, 2, 3, 4, 5), x13, (0, 6, 2, 1, 3, 5, 4), (1, 2, 6, 4))
    del x13
    e_pert += einsum(v.ooov, (0, 1, 2, 3), x14, (0, 1, 2, 3), ()) * 0.5
    del x14
    x15 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x15 += einsum(x11, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -0.5
    del x11
    x15 += einsum(x10, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3))
    x15 += einsum(x10, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -0.25
    del x10
    x16 = np.zeros((nocc, nvir), dtype=np.float64)
    x16 += einsum(v.oovv, (0, 1, 2, 3), denom3, (0, 1, 4, 2, 5, 3), x15, (1, 0, 4, 2, 3, 5), (4, 5))
    del x15
    e_pert += einsum(l1, (0, 1), x16, (1, 0), ())
    del x16

    return e_pert

