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
    t1new = np.zeros((nocc, nvir), dtype=np.float64)
    t1new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 5, 2, 3), (0, 4)) * 0.5
    t1new += einsum(v.oovv, (0, 1, 2, 3), (0, 2))
    t1new += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (4, 5))
    t1new += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 2, 1, 5), (0, 5)) * -1.0
    x0 = np.zeros((nvir,), dtype=np.float64)
    x0 += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 2), (4,))
    t1new += einsum(x0, (0,), t2, (1, 2, 3, 0), (1, 3)) * 0.5
    del x0
    x1 = np.zeros((nocc,), dtype=np.float64)
    x1 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 0, 2, 3), (4,))
    t1new += einsum(x1, (0,), t2, (1, 0, 2, 3), (1, 2)) * 0.5
    del x1
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x2 += einsum(t2, (0, 1, 2, 3), v.oovv, (1, 4, 5, 2), (0, 1, 4, 5))
    t1new += einsum(t2, (0, 1, 2, 3), x2, (4, 1, 0, 3), (4, 2)) * -0.5
    del x2
    x3 = np.zeros((nocc, nocc), dtype=np.float64)
    x3 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    t1new += einsum(x3, (0, 1), t2, (2, 1, 3, 4), (0, 3)) * 0.5
    del x3
    x4 = np.zeros((nocc, nvir), dtype=np.float64)
    x4 += einsum(v.ovov, (0, 1, 2, 3), (0, 3)) * -1.0
    x4 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (4, 5))
    t1new += einsum(x4, (0, 1), t2, (2, 0, 3, 1), (2, 3))
    del x4
    x5 = np.zeros((nocc, nocc, nvir), dtype=np.float64)
    x5 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1))
    x5 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 5)) * -1.0
    t1new += einsum(x5, (0, 1, 2), t2, (3, 1, 4, 2), (0, 4))
    del x5
    x6 = np.zeros((nocc, nocc, nocc), dtype=np.float64)
    x6 += einsum(v.oooo, (0, 1, 1, 2), (0, 1, 2)) * -1.0
    x6 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 0, 2, 3), (4, 0, 1)) * -0.5
    t1new += einsum(x6, (0, 1, 2), t2, (0, 1, 3, 4), (2, 3)) * 0.5
    del x6

    return {"t2new": t2new}

