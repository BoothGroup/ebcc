# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(f.ov, (0, 1), t1, (0, 1), ()) * 2.0
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x0 += einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -0.5
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x1 += einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3))
    x1 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3))
    e_cc += einsum(x0, (0, 1, 2, 3), x1, (0, 1, 3, 2), ()) * 2.0
    del x0, x1

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T amplitudes
    t1new = np.zeros((nocc, nvir), dtype=np.float64)
    t1new = einsum(f.ov, (0, 1), (0, 1), alpha=1.0, beta=1.0, out=t1new)
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2new = einsum(t2, (0, 1, 2, 3), v.oovv, (4, 1, 5, 3), (4, 0, 5, 2), alpha=-1.0, beta=1.0, out=t2new)
    t2new = einsum(t2, (0, 1, 2, 3), v.oooo, (4, 0, 5, 1), (4, 5, 2, 3), alpha=1.0, beta=1.0, out=t2new)
    t2new = einsum(t1, (0, 1), v.ooov, (2, 0, 3, 4), (3, 2, 4, 1), alpha=-1.0, beta=1.0, out=t2new)
    t2new = einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3), alpha=2.0, beta=1.0, out=t2new)
    t2new = einsum(t2, (0, 1, 2, 3), v.vvvv, (4, 2, 5, 3), (0, 1, 4, 5), alpha=1.0, beta=1.0, out=t2new)
    x0 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x0 = einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3), alpha=1.0, beta=1.0, out=x0)
    x0 = einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 1, 3), alpha=-0.5, beta=1.0, out=x0)
    t1new = einsum(t2, (0, 1, 2, 3), x0, (1, 3, 2, 4), (0, 4), alpha=2.0, beta=1.0, out=t1new)
    del x0
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x1 = einsum(t1, (0, 1), v.ovov, (2, 3, 4, 1), (0, 2, 4, 3), alpha=1.0, beta=1.0, out=x1)
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x2 = einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3), alpha=2.0, beta=1.0, out=x2)
    x2 = einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3), alpha=-1.0, beta=1.0, out=x2)
    x2 = einsum(x1, (0, 1, 2, 3), (0, 1, 2, 3), alpha=-1.0, beta=1.0, out=x2)
    x2 = einsum(x1, (0, 1, 2, 3), (0, 2, 1, 3), alpha=2.0, beta=1.0, out=x2)
    t1new = einsum(t2, (0, 1, 2, 3), x2, (4, 0, 1, 3), (4, 2), alpha=-1.0, beta=1.0, out=t1new)
    del x2
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x3 = einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1), alpha=1.0, beta=1.0, out=x3)
    x3 = einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3), alpha=-0.5, beta=1.0, out=x3)
    x4 = np.zeros((nocc, nvir), dtype=np.float64)
    x4 = einsum(t1, (0, 1), x3, (0, 2, 3, 1), (2, 3), alpha=2.0, beta=1.0, out=x4)
    x5 = np.zeros((nocc, nvir), dtype=np.float64)
    x5 = einsum(f.ov, (0, 1), (0, 1), alpha=1.0, beta=1.0, out=x5)
    x5 = einsum(x4, (0, 1), (0, 1), alpha=1.0, beta=1.0, out=x5)
    del x4
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x6 = einsum(t2, (0, 1, 2, 3), (0, 1, 3, 2), alpha=1.0, beta=1.0, out=x6)
    x6 = einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3), alpha=-0.5, beta=1.0, out=x6)
    t1new = einsum(x5, (0, 1), x6, (0, 2, 3, 1), (2, 3), alpha=2.0, beta=1.0, out=t1new)
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x7 = einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3), alpha=1.0, beta=1.0, out=x7)
    x7 = einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3), alpha=-0.5, beta=1.0, out=x7)
    t1new = einsum(t1, (0, 1), x7, (0, 2, 1, 3), (2, 3), alpha=2.0, beta=1.0, out=t1new)
    del x7
    x8 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x8 = einsum(v.ovvv, (0, 1, 2, 3), (0, 1, 2, 3), alpha=1.0, beta=1.0, out=x8)
    x8 = einsum(v.ovvv, (0, 1, 2, 3), (0, 2, 3, 1), alpha=-0.5, beta=1.0, out=x8)
    x9 = np.zeros((nvir, nvir), dtype=np.float64)
    x9 = einsum(f.vv, (0, 1), (0, 1), alpha=0.5, beta=1.0, out=x9)
    x9 = einsum(t1, (0, 1), x8, (0, 1, 2, 3), (3, 2), alpha=1.0, beta=1.0, out=x9)
    del x8
    t1new = einsum(t1, (0, 1), x9, (1, 2), (0, 2), alpha=2.0, beta=1.0, out=t1new)
    del x9
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x10 = einsum(v.ovov, (0, 1, 2, 3), (0, 2, 3, 1), alpha=-0.5, beta=1.0, out=x10)
    x10 = einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3), alpha=1.0, beta=1.0, out=x10)
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x11 = einsum(v.ooov, (0, 1, 2, 3), (0, 1, 2, 3), alpha=1.0, beta=1.0, out=x11)
    x11 = einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3), alpha=-0.5, beta=1.0, out=x11)
    x12 = np.zeros((nocc, nocc), dtype=np.float64)
    x12 = einsum(f.oo, (0, 1), (0, 1), alpha=1.0, beta=1.0, out=x12)
    x12 = einsum(t2, (0, 1, 2, 3), x10, (1, 4, 3, 2), (4, 0), alpha=2.0, beta=1.0, out=x12)
    x12 = einsum(t1, (0, 1), x11, (2, 3, 0, 1), (3, 2), alpha=2.0, beta=1.0, out=x12)
    del x11
    x12 = einsum(t1, (0, 1), x5, (2, 1), (2, 0), alpha=1.0, beta=1.0, out=x12)
    del x5
    t1new = einsum(t1, (0, 1), x12, (0, 2), (2, 1), alpha=-1.0, beta=1.0, out=t1new)
    del x12
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x13 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 3), (0, 4, 2, 5), alpha=1.0, beta=1.0, out=x13)
    t2new = einsum(x13, (0, 1, 2, 3), (1, 0, 3, 2), alpha=2.0, beta=1.0, out=t2new)
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x14 = einsum(t2, (0, 1, 2, 3), v.ovov, (4, 5, 1, 2), (0, 4, 3, 5), alpha=1.0, beta=1.0, out=x14)
    t2new = einsum(x14, (0, 1, 2, 3), (0, 1, 2, 3), alpha=-1.0, beta=1.0, out=t2new)
    x15 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x15 = einsum(t1, (0, 1), v.vvvv, (2, 3, 4, 1), (0, 2, 3, 4), alpha=1.0, beta=1.0, out=x15)
    t2new = einsum(t1, (0, 1), x15, (2, 3, 1, 4), (0, 2, 3, 4), alpha=1.0, beta=1.0, out=t2new)
    del x15
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x16 = einsum(f.oo, (0, 1), t2, (2, 1, 3, 4), (0, 2, 3, 4), alpha=1.0, beta=1.0, out=x16)
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x17 = einsum(f.vv, (0, 1), t2, (2, 3, 4, 1), (2, 3, 0, 4), alpha=1.0, beta=1.0, out=x17)
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x18 = einsum(t1, (0, 1), v.ovvv, (2, 3, 4, 1), (0, 2, 3, 4), alpha=1.0, beta=1.0, out=x18)
    x19 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x19 = einsum(t1, (0, 1), v.ooov, (2, 3, 4, 1), (0, 2, 3, 4), alpha=1.0, beta=1.0, out=x19)
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x20 = einsum(t1, (0, 1), x19, (2, 3, 4, 0), (2, 4, 3, 1), alpha=1.0, beta=1.0, out=x20)
    del x19
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x21 = einsum(t1, (0, 1), x20, (2, 0, 3, 4), (2, 3, 1, 4), alpha=1.0, beta=1.0, out=x21)
    del x20
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x22 = einsum(t2, (0, 1, 2, 3), (0, 1, 2, 3), alpha=1.0, beta=1.0, out=x22)
    x22 = einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3), alpha=1.0, beta=1.0, out=x22)
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x23 = einsum(v.oovv, (0, 1, 2, 3), x22, (1, 4, 5, 3), (4, 0, 5, 2), alpha=1.0, beta=1.0, out=x23)
    del x22
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x24 = einsum(x16, (0, 1, 2, 3), (0, 1, 2, 3), alpha=1.0, beta=1.0, out=x24)
    del x16
    x24 = einsum(x17, (0, 1, 2, 3), (0, 1, 2, 3), alpha=-1.0, beta=1.0, out=x24)
    del x17
    x24 = einsum(x18, (0, 1, 2, 3), (0, 1, 2, 3), alpha=-1.0, beta=1.0, out=x24)
    x24 = einsum(x21, (0, 1, 2, 3), (0, 1, 2, 3), alpha=-1.0, beta=1.0, out=x24)
    del x21
    x24 = einsum(x23, (0, 1, 2, 3), (0, 1, 2, 3), alpha=1.0, beta=1.0, out=x24)
    del x23
    t2new = einsum(x24, (0, 1, 2, 3), (0, 1, 3, 2), alpha=-1.0, beta=1.0, out=t2new)
    t2new = einsum(x24, (0, 1, 2, 3), (1, 0, 2, 3), alpha=-1.0, beta=1.0, out=t2new)
    del x24
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x25 = einsum(t2, (0, 1, 2, 3), x14, (4, 1, 5, 3), (0, 4, 2, 5), alpha=1.0, beta=1.0, out=x25)
    x26 = np.zeros((nvir, nvir), dtype=np.float64)
    x26 = einsum(t2, (0, 1, 2, 3), x3, (0, 1, 3, 4), (2, 4), alpha=1.0, beta=1.0, out=x26)
    del x3
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x27 = einsum(x26, (0, 1), t2, (2, 3, 1, 4), (2, 3, 0, 4), alpha=1.0, beta=1.0, out=x27)
    del x26
    x28 = np.zeros((nocc, nocc), dtype=np.float64)
    x28 = einsum(t2, (0, 1, 2, 3), x10, (1, 4, 3, 2), (0, 4), alpha=1.0, beta=1.0, out=x28)
    del x10
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x29 = einsum(x28, (0, 1), t2, (2, 1, 3, 4), (0, 2, 4, 3), alpha=1.0, beta=1.0, out=x29)
    del x28
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x30 = einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3), alpha=1.0, beta=1.0, out=x30)
    x30 = einsum(x18, (0, 1, 2, 3), (1, 0, 2, 3), alpha=1.0, beta=1.0, out=x30)
    del x18
    x31 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x31 = einsum(t1, (0, 1), x30, (2, 3, 1, 4), (2, 3, 0, 4), alpha=1.0, beta=1.0, out=x31)
    del x30
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x32 = einsum(t1, (0, 1), x31, (0, 2, 3, 4), (3, 2, 4, 1), alpha=1.0, beta=1.0, out=x32)
    del x31
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x33 = einsum(x25, (0, 1, 2, 3), (0, 1, 2, 3), alpha=2.0, beta=1.0, out=x33)
    del x25
    x33 = einsum(x27, (0, 1, 2, 3), (1, 0, 3, 2), alpha=1.0, beta=1.0, out=x33)
    del x27
    x33 = einsum(x29, (0, 1, 2, 3), (1, 0, 3, 2), alpha=1.0, beta=1.0, out=x33)
    del x29
    x33 = einsum(x32, (0, 1, 2, 3), (0, 1, 3, 2), alpha=1.0, beta=1.0, out=x33)
    del x32
    t2new = einsum(x33, (0, 1, 2, 3), (0, 1, 2, 3), alpha=-1.0, beta=1.0, out=t2new)
    t2new = einsum(x33, (0, 1, 2, 3), (1, 0, 3, 2), alpha=-1.0, beta=1.0, out=t2new)
    del x33
    x34 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x34 = einsum(v.oooo, (0, 1, 2, 3), (0, 1, 2, 3), alpha=1.0, beta=1.0, out=x34)
    x34 = einsum(t1, (0, 1), x1, (2, 3, 4, 1), (3, 0, 2, 4), alpha=1.0, beta=1.0, out=x34)
    del x1
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x35 = einsum(v.ooov, (0, 1, 2, 3), (0, 2, 1, 3), alpha=-1.0, beta=1.0, out=x35)
    x35 = einsum(t1, (0, 1), x34, (0, 2, 3, 4), (3, 2, 4, 1), alpha=1.0, beta=1.0, out=x35)
    del x34
    t2new = einsum(t1, (0, 1), x35, (2, 3, 0, 4), (2, 3, 1, 4), alpha=1.0, beta=1.0, out=t2new)
    del x35
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x36 = einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3), alpha=1.0, beta=1.0, out=x36)
    x36 = einsum(v.oovv, (0, 1, 2, 3), (0, 1, 2, 3), alpha=-0.5, beta=1.0, out=x36)
    x36 = einsum(x13, (0, 1, 2, 3), (0, 1, 2, 3), alpha=2.0, beta=1.0, out=x36)
    del x13
    t2new = einsum(t2, (0, 1, 2, 3), x36, (4, 1, 5, 3), (0, 4, 2, 5), alpha=2.0, beta=1.0, out=t2new)
    del x36
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x37 = einsum(v.ovov, (0, 1, 2, 3), (0, 2, 1, 3), alpha=-1.0, beta=1.0, out=x37)
    x37 = einsum(x14, (0, 1, 2, 3), (0, 1, 2, 3), alpha=1.0, beta=1.0, out=x37)
    del x14
    t2new = einsum(t2, (0, 1, 2, 3), x37, (4, 1, 5, 2), (4, 0, 5, 3), alpha=1.0, beta=1.0, out=t2new)
    del x37

    return {"t1new": t1new, "t2new": t2new}

