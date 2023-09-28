# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 2, 3), ()) * 0.25

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t2=None, **kwargs):
    # T amplitudes
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 5, 2, 3), (0, 1, 4, 5)) * 0.5
    t2new += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(t2, (0, 1, 2, 3), v.oooo, (4, 5, 0, 1), (4, 5, 2, 3)) * 0.5
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4))
    x1 = np.zeros((nocc, nocc), dtype=np.float64)
    x1 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x2 += einsum(x1, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4))
    del x1
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x3 += einsum(x0, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x0
    x3 += einsum(x2, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.25
    del x2
    t2new += einsum(x3, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x3, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x4 += einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new += einsum(x4, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new += einsum(x4, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x4
    x5 = np.zeros((nvir, nvir), dtype=np.float64)
    x5 += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x6 += einsum(x5, (0, 1), t2, (2, 3, 4, 1), (2, 3, 4, 0))
    del x5
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x7 += einsum(t2, (0, 1, 2, 3), vs, (4, 1, 5, 3), (0, 4, 2, 5))
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x8 += einsum(t2, (0, 1, 2, 3), x7, (4, 1, 5, 3), (0, 4, 2, 5))
    del x7
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x9 += einsum(x6, (0, 1, 2, 3), (0, 1, 2, 3)) * -0.25
    del x6
    x9 += einsum(x8, (0, 1, 2, 3), (0, 1, 2, 3))
    del x8
    t2new += einsum(x9, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(x9, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x9
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x10 += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    t2new += einsum(x10, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new += einsum(x10, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new += einsum(x10, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new += einsum(x10, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x10

    return {"t2new": t2new}

