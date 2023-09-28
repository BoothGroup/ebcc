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
    t2new += einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 5, 2, 3), (0, 1, 4, 5)) * 0.5
    t2new += einsum(t2, (0, 1, 2, 3), v.oooo, (4, 5, 0, 1), (4, 5, 2, 3)) * 0.5
    t2new += einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new += einsum(t2, (0, 1, 2, 3), v.ovov, (4, 3, 1, 5), (4, 0, 5, 2)) * -1.0
    x0 = np.zeros((nvir, nvir), dtype=np.float64)
    x0 += einsum(t2, (0, 1, 2, 3), v.oovv, (0, 1, 4, 3), (2, 4))
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x1 += einsum(x0, (0, 1), t2, (2, 3, 4, 1), (2, 3, 4, 0))
    del x0
    t2new += einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3)) * 0.25
    t2new += einsum(x1, (0, 1, 2, 3), (0, 1, 3, 2)) * 0.25
    del x1
    x2 = np.zeros((nocc, nocc), dtype=np.float64)
    x2 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 2, 3), (0, 4))
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x3 += einsum(x2, (0, 1), t2, (2, 1, 3, 4), (2, 0, 3, 4))
    del x2
    t2new += einsum(x3, (0, 1, 2, 3), (0, 1, 3, 2)) * -0.25
    t2new += einsum(x3, (0, 1, 2, 3), (1, 0, 3, 2)) * -0.25
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x4 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x4 += einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new += einsum(t2, (0, 1, 2, 3), x4, (4, 1, 5, 3), (0, 4, 2, 5))
    del x4

    return {"t2new": t2new}

