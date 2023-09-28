# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), x0, (0, 1, 3, 2), ()) * 2.0
    del x0

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # T amplitudes
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (4, 0, 5, 2)) * 2.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 2), (4, 0, 3, 5)) * -1.0
    t2new += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (4, 0, 5, 2)) * -1.0
    t2new += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5))
    t2new += einsum(x0, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x1 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x1 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x2 += einsum(t2, (0, 1, 2, 3), x1, (1, 4, 5, 2), (0, 4, 3, 5))
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x3 += einsum(t2, (0, 1, 2, 3), x2, (4, 1, 5, 3), (0, 4, 2, 5)) * 2.0
    del x2
    x4 = np.zeros((nvir, nvir), dtype=np.float64)
    x4 += einsum(t2, (0, 1, 2, 3), x1, (0, 1, 3, 4), (2, 4))
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x5 += einsum(x4, (0, 1), t2, (2, 3, 1, 4), (2, 3, 4, 0)) * 2.0
    del x4
    x6 = np.zeros((nocc, nocc), dtype=np.float64)
    x6 += einsum(t2, (0, 1, 2, 3), x1, (1, 4, 2, 3), (0, 4))
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x7 += einsum(x6, (0, 1), t2, (2, 1, 3, 4), (2, 0, 4, 3)) * 2.0
    del x6
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x8 += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    del x3
    x8 += einsum(x5, (0, 1, 2, 3), (1, 0, 2, 3))
    del x5
    x8 += einsum(x7, (0, 1, 2, 3), (0, 1, 3, 2))
    del x7
    t2new += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x8, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x9 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x9 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.5
    x9 += einsum(t2, (0, 1, 2, 3), x1, (1, 4, 5, 3), (0, 4, 2, 5)) * 2.0
    del x1
    t2new += einsum(t2, (0, 1, 2, 3), x9, (4, 1, 5, 3), (0, 4, 2, 5)) * 2.0
    del x9
    x10 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x10 += einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3))
    x10 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 5, 3), (4, 0, 5, 1))
    t2new += einsum(t2, (0, 1, 2, 3), x10, (0, 4, 1, 5), (4, 5, 2, 3))
    del x10
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x11 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x11 += einsum(x0, (0, 1, 2, 3), (0, 1, 2, 3))
    del x0
    t2new += einsum(t2, (0, 1, 2, 3), x11, (4, 1, 5, 2), (0, 4, 3, 5))
    del x11
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x12 += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    x12 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x12, (4, 1, 5, 2), (0, 4, 5, 3))
    del x12

    return {"t2new": t2new}

