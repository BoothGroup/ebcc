# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 2), ()) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x0 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x0 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x1 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x1 += einsum(f.bb.ov, (0, 1), (0, 1)) * 2.0
    x1 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3)) * 2.0
    x1 += einsum(t1.bb, (0, 1), x0, (0, 2, 3, 1), (2, 3)) * -1.0
    del x0
    e_cc += einsum(t1.bb, (0, 1), x1, (0, 1), ()) * 0.5
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x3 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x3 += einsum(f.aa.ov, (0, 1), (0, 1))
    x3 += einsum(t1.aa, (0, 1), x2, (0, 2, 3, 1), (2, 3)) * -0.5
    del x2
    e_cc += einsum(t1.aa, (0, 1), x3, (0, 1), ())
    del x3

    return e_cc

def update_amps(f=None, v=None, space=None, t1=None, t2=None, t3=None, **kwargs):
    t1new = Namespace()
    t2new = Namespace()
    t3new = Namespace()

    nocc = (space[0].ncocc, space[1].ncocc)
    nvir = (space[0].ncvir, space[1].ncvir)
    naocc = (space[0].naocc, space[1].naocc)
    navir = (space[0].navir, space[1].navir)
    soa = np.ones((nocc[0],), dtype=bool)
    sva = np.ones((nvir[0],), dtype=bool)
    sob = np.ones((nocc[1],), dtype=bool)
    svb = np.ones((nvir[1],), dtype=bool)
    sOa = space[0].active[space[0].correlated][space[0].occupied[space[0].correlated]]
    sVa = space[0].active[space[0].correlated][space[0].virtual[space[0].correlated]]
    sOb = space[1].active[space[1].correlated][space[1].occupied[space[1].correlated]]
    sVb = space[1].active[space[1].correlated][space[1].virtual[space[1].correlated]]

    # T amplitudes
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    t1new_bb[np.ix_(sob,svb)] += einsum(f.bb.ov, (0, 1), (0, 1))
    t1new_bb[np.ix_(sob,svb)] += einsum(f.bb.oo, (0, 1), t1.bb[np.ix_(sob,svb)], (1, 2), (0, 2)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(f.bb.vv, (0, 1), t1.bb[np.ix_(sob,svb)], (2, 1), (2, 0))
    t1new_bb[np.ix_(sob,svb)] += einsum(f.aa.ov, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (0, 2, 1, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(f.bb.ov, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oovv, (2, 0, 3, 1), (2, 3)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 1), (4, 3)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovvv, (0, 2, 4, 3), (1, 4))
    t1new_bb[np.ix_(sOb,sVb)] += einsum(v.aaaa.OVOV, (0, 1, 2, 3), t3.abaaba, (0, 4, 2, 1, 5, 3), (4, 5))
    t1new_bb[np.ix_(sOb,sVb)] += einsum(v.aabb.OVOV, (0, 1, 2, 3), t3.babbab, (4, 0, 2, 5, 1, 3), (4, 5)) * 2.0
    t1new_bb[np.ix_(sOb,sVb)] += einsum(v.bbbb.OVOV, (0, 1, 2, 3), t3.bbbbbb, (4, 0, 2, 5, 1, 3), (4, 5)) * 3.0
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    t1new_aa[np.ix_(soa,sva)] += einsum(f.aa.ov, (0, 1), (0, 1))
    t1new_aa[np.ix_(soa,sva)] += einsum(f.aa.oo, (0, 1), t1.aa[np.ix_(soa,sva)], (1, 2), (0, 2)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(f.aa.vv, (0, 1), t1.aa[np.ix_(soa,sva)], (2, 1), (2, 0))
    t1new_aa[np.ix_(soa,sva)] += einsum(f.bb.ov, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 0, 3, 1), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(f.aa.ov, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oovv, (2, 0, 3, 1), (2, 3)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 0, 1, 3), (4, 2)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vvov, (4, 2, 1, 3), (0, 4))
    t1new_aa[np.ix_(sOa,sVa)] += einsum(v.aabb.OVOV, (0, 1, 2, 3), t3.abaaba, (4, 2, 0, 5, 3, 1), (4, 5)) * 2.0
    t1new_aa[np.ix_(sOa,sVa)] += einsum(v.bbbb.OVOV, (0, 1, 2, 3), t3.babbab, (0, 4, 2, 1, 5, 3), (4, 5))
    t1new_aa[np.ix_(sOa,sVa)] += einsum(v.aaaa.OVOV, (0, 1, 2, 3), t3.aaaaaa, (4, 0, 2, 5, 1, 3), (4, 5)) * 3.0
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    t2new_aaaa[np.ix_(sOa,sOa,sVa,sVa)] += einsum(f.bb.OV, (0, 1), t3.abaaba, (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    t2new_aaaa[np.ix_(sOa,sOa,sVa,sVa)] += einsum(f.aa.OV, (0, 1), t3.aaaaaa, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(f.aa.oo, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(f.bb.oo, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(f.aa.vv, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(f.bb.vv, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ooov, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ovoo, (2, 3, 4, 0), (2, 4, 3, 1)) * -1.0
    t2new_abab[np.ix_(sOa,sOb,sVa,sVb)] += einsum(f.aa.OV, (0, 1), t3.abaaba, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 2.0
    t2new_abab[np.ix_(sOa,sOb,sVa,sVb)] += einsum(f.bb.OV, (0, 1), t3.babbab, (2, 3, 0, 4, 5, 1), (3, 2, 5, 4)) * 2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.oooo, (4, 0, 5, 1), (4, 5, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.oovv, (4, 1, 5, 3), (0, 4, 2, 5)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.oovv, (4, 0, 5, 3), (4, 1, 2, 5)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.oovv, (4, 0, 5, 2), (4, 1, 5, 3)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vvoo, (4, 2, 5, 1), (0, 5, 4, 3)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aabb.ovov, (1, 3, 4, 5), (0, 4, 2, 5)) * 2.0
    t2new_abab[np.ix_(soa,sOb,sVa,sVb)] += einsum(v.aaaa.oOOV, (0, 1, 2, 3), t3.abaaba, (1, 4, 2, 5, 6, 3), (0, 4, 5, 6)) * -2.0
    t2new_abab[np.ix_(sOa,sob,sVa,sVb)] += einsum(v.aabb.OVoO, (0, 1, 2, 3), t3.abaaba, (4, 3, 0, 5, 6, 1), (4, 2, 5, 6)) * -2.0
    t2new_abab[np.ix_(sOa,sOb,sva,sVb)] += einsum(v.aaaa.vVOV, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 3, 6, 1), (4, 5, 0, 6)) * -2.0
    t2new_abab[np.ix_(sOa,sOb,sVa,svb)] += einsum(v.aabb.OVvV, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 3, 1), (4, 5, 6, 2)) * 2.0
    t2new_abab[np.ix_(soa,sOb,sVa,sVb)] += einsum(v.aabb.oOOV, (0, 1, 2, 3), t3.babbab, (4, 1, 2, 5, 6, 3), (0, 4, 6, 5)) * -2.0
    t2new_abab[np.ix_(sOa,sob,sVa,sVb)] += einsum(v.bbbb.oOOV, (0, 1, 2, 3), t3.babbab, (1, 4, 2, 5, 6, 3), (4, 0, 6, 5)) * -2.0
    t2new_abab[np.ix_(sOa,sOb,sVa,svb)] += einsum(v.bbbb.vVOV, (0, 1, 2, 3), t3.babbab, (4, 5, 2, 1, 6, 3), (5, 4, 6, 0)) * 2.0
    t2new_abab[np.ix_(sOa,sOb,sva,sVb)] += einsum(v.aabb.vVOV, (0, 1, 2, 3), t3.babbab, (4, 5, 2, 6, 1, 3), (5, 4, 0, 6)) * 2.0
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    t2new_bbbb[np.ix_(sOb,sOb,sVb,sVb)] += einsum(f.aa.OV, (0, 1), t3.babbab, (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    t2new_bbbb[np.ix_(sOb,sOb,sVb,sVb)] += einsum(f.bb.OV, (0, 1), t3.bbbbbb, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    t3new_babbab = np.zeros((naocc[1], naocc[0], naocc[1], navir[1], navir[0], navir[1]), dtype=types[float])
    t3new_babbab += einsum(f.aa.OO, (0, 1), t3.babbab, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    t3new_babbab += einsum(f.aa.VV, (0, 1), t3.babbab, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    t3new_babbab += einsum(v.bbbb.OOOO, (0, 1, 2, 3), t3.babbab, (3, 4, 1, 5, 6, 7), (0, 4, 2, 5, 6, 7)) * -2.0
    t3new_babbab += einsum(v.aaaa.OOVV, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -2.0
    t3new_babbab += einsum(v.aaaa.OVOV, (0, 1, 2, 3), t3.babbab, (4, 2, 5, 6, 3, 7), (4, 0, 5, 6, 1, 7)) * 2.0
    t3new_babbab += einsum(v.bbbb.VVVV, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 7, 2)) * 2.0
    t3new_babbab += einsum(v.aabb.OVOV, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 2, 6, 7, 3), (4, 0, 5, 6, 1, 7)) * 6.0
    t3new_abaaba = np.zeros((naocc[0], naocc[1], naocc[0], navir[0], navir[1], navir[0]), dtype=types[float])
    t3new_abaaba += einsum(f.bb.OO, (0, 1), t3.abaaba, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    t3new_abaaba += einsum(f.bb.VV, (0, 1), t3.abaaba, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    t3new_abaaba += einsum(v.aaaa.OOOO, (0, 1, 2, 3), t3.abaaba, (1, 4, 3, 5, 6, 7), (0, 4, 2, 5, 6, 7)) * 2.0
    t3new_abaaba += einsum(v.bbbb.OOVV, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -2.0
    t3new_abaaba += einsum(v.bbbb.OVOV, (0, 1, 2, 3), t3.abaaba, (4, 2, 5, 6, 3, 7), (4, 0, 5, 6, 1, 7)) * 2.0
    t3new_abaaba += einsum(v.aaaa.VVVV, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 7, 2)) * 2.0
    t3new_abaaba += einsum(v.aabb.OVOV, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 7, 1), (4, 2, 5, 6, 3, 7)) * 6.0
    x0 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x0 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(x0, (0, 1), (0, 1))
    t1new_bb[np.ix_(sob,svb)] += einsum(x0, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x0, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 0, 3, 1), (2, 3))
    x1 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x1 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovov, (2, 3, 0, 1), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(x1, (0, 1), (0, 1))
    t1new_bb[np.ix_(sob,svb)] += einsum(x1, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x1, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 0, 3, 1), (2, 3))
    x2 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x2 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 0, 3), (4, 2)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(x2, (0, 1), (0, 1)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(x2, (0, 1), (0, 1)) * -1.0
    del x2
    x3 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x3 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (1, 3, 4, 2), (0, 4)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(x3, (0, 1), (0, 1)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(x3, (0, 1), (0, 1)) * -1.0
    del x3
    x4 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x4 += einsum(f.bb.ov, (0, 1), t1.bb[np.ix_(sob,svb)], (2, 1), (0, 2))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x4, (0, 2), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x4, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    x5 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x5 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x5, (2, 1), (0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x5, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    x6 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x6 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x6, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x6, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    x7 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x7 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ooov, (2, 0, 3, 1), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x7, (2, 0), (2, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x7, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    x8 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x8 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ooov, (2, 3, 0, 1), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x8, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x8, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    x9 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x9 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvv, (0, 2, 3, 1), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x9, (1, 2), (0, 2)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x9, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    x10 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x10 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x10, (2, 1), (0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x10, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    x11 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x11 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovov, (2, 3, 0, 1), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(x11, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (0, 2, 1, 3), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(x11, (0, 1), (0, 1))
    t1new_aa[np.ix_(soa,sva)] += einsum(x11, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 0, 3, 1), (2, 3)) * 2.0
    x12 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x12 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovov, (2, 1, 0, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(x12, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (0, 2, 1, 3), (2, 3)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x12, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 0, 3, 1), (2, 3)) * -2.0
    x13 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x13 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ovov, (2, 3, 4, 1), (2, 0, 4, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x13, (0, 4, 1, 2), (4, 3)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x13, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    x14 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x14 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x14, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x14, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    x15 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x15 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(x15, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (0, 2, 1, 3), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(x15, (0, 1), (0, 1))
    t1new_aa[np.ix_(soa,sva)] += einsum(x15, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 0, 3, 1), (2, 3)) * 2.0
    x16 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x16 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x17 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x17 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x16, (4, 0, 1, 3), (4, 2)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(x17, (0, 1), (0, 1)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(x17, (0, 1), (0, 1)) * -1.0
    del x17
    x18 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x18 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x19 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x19 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x18, (2, 0), (2, 1))
    t1new_bb[np.ix_(sob,svb)] += einsum(x19, (0, 1), (0, 1)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(x19, (0, 1), (0, 1)) * -1.0
    del x19
    x20 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x20 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovov, (2, 1, 0, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(x20, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 0, 3, 1), (2, 3)) * -2.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x20, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 0, 3, 1), (2, 3)) * -1.0
    x21 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x21 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x0, (2, 1), (0, 2))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x21, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x21, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    x22 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x22 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x20, (2, 1), (0, 2))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x22, (2, 0), (2, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x22, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    x23 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x23 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1, (2, 1), (0, 2))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x23, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x23, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    x24 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x24 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ooov, (4, 0, 1, 3), (4, 2))
    t1new_aa[np.ix_(soa,sva)] += einsum(x24, (0, 1), (0, 1)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x24, (0, 1), (0, 1)) * -1.0
    del x24
    x25 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x25 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvv, (1, 2, 4, 3), (0, 4))
    t1new_aa[np.ix_(soa,sva)] += einsum(x25, (0, 1), (0, 1)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x25, (0, 1), (0, 1)) * -1.0
    del x25
    x26 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x26 += einsum(f.aa.ov, (0, 1), t1.aa[np.ix_(soa,sva)], (2, 1), (0, 2))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x26, (0, 2), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x26, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    x27 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x27 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ooov, (2, 0, 3, 1), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x27, (2, 0), (2, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x27, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4))
    x28 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x28 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ooov, (2, 3, 0, 1), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x28, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x28, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    x29 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x29 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvv, (0, 2, 3, 1), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x29, (1, 2), (0, 2)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x29, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    x30 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x30 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x30, (2, 1), (0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x30, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 3, 0, 4))
    x31 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x31 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x31, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x31, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    x32 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x32 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.vvov, (2, 3, 0, 1), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x32, (2, 1), (0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x32, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 3, 0, 4))
    x33 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x33 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovov, (2, 1, 3, 4), (0, 2, 3, 4))
    t1new_aa[np.ix_(soa,sva)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x33, (4, 0, 1, 3), (4, 2)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x33, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    x34 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x34 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x34, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x34, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    x35 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x35 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x36 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x36 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x35, (4, 0, 1, 3), (4, 2)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x36, (0, 1), (0, 1)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x36, (0, 1), (0, 1)) * -1.0
    del x36
    x37 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x37 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 1, 3), (0, 4))
    x38 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x38 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x37, (2, 0), (2, 1))
    t1new_aa[np.ix_(soa,sva)] += einsum(x38, (0, 1), (0, 1)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x38, (0, 1), (0, 1)) * -1.0
    del x38
    x39 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x39 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x12, (2, 1), (0, 2))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x39, (2, 0), (2, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x39, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4))
    x40 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x40 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x11, (2, 1), (0, 2))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x40, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x40, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    x41 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x41 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x15, (2, 1), (0, 2))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x41, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x41, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    x42 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x42 += einsum(f.aa.oo, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x42, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x42, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x42
    x43 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x43 += einsum(f.aa.vv, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x43, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x43, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x43
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x44 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x44, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x44, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x44, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x44
    x45 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x45 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x45, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x45, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x45, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x45, (4, 0, 2, 5), (4, 1, 5, 3))
    x46 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x46 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x46, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x46, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x46, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x46, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x47 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x47 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.oooo, (4, 1, 5, 0), (4, 5, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x47, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x47, (0, 1, 2, 3), (1, 0, 3, 2))
    del x47
    x48 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x48 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x48, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x48, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x48, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x48
    x49 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x49 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x49, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x49, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x49, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    x50 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x50 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x50, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x50, (0, 1, 2, 3), (1, 0, 3, 2))
    del x50
    x51 = np.zeros((naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x51 += einsum(v.aabb.oOOV, (0, 1, 2, 3), t3.abaaba, (4, 2, 1, 5, 3, 6), (4, 5, 6, 0))
    t2new_aaaa[np.ix_(soa,sOa,sVa,sVa)] += einsum(x51, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    t2new_aaaa[np.ix_(sOa,soa,sVa,sVa)] += einsum(x51, (0, 1, 2, 3), (0, 3, 2, 1)) * 2.0
    del x51
    x52 = np.zeros((naocc[0], naocc[0], navir[0], nvir[0]), dtype=types[float])
    x52 += einsum(v.aabb.vVOV, (0, 1, 2, 3), t3.abaaba, (4, 2, 5, 6, 3, 1), (4, 5, 6, 0))
    t2new_aaaa[np.ix_(sOa,sOa,sva,sVa)] += einsum(x52, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(sOa,sOa,sVa,sva)] += einsum(x52, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x52
    x53 = np.zeros((naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x53 += einsum(v.aaaa.oOOV, (0, 1, 2, 3), t3.aaaaaa, (4, 2, 1, 5, 6, 3), (4, 5, 6, 0))
    t2new_aaaa[np.ix_(soa,sOa,sVa,sVa)] += einsum(x53, (0, 1, 2, 3), (3, 0, 2, 1)) * 6.0
    t2new_aaaa[np.ix_(sOa,soa,sVa,sVa)] += einsum(x53, (0, 1, 2, 3), (0, 3, 2, 1)) * -6.0
    del x53
    x54 = np.zeros((naocc[0], naocc[0], navir[0], nvir[0]), dtype=types[float])
    x54 += einsum(v.aaaa.vVOV, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 2, 6, 3, 1), (4, 5, 6, 0))
    t2new_aaaa[np.ix_(sOa,sOa,sva,sVa)] += einsum(x54, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.0
    t2new_aaaa[np.ix_(sOa,sOa,sVa,sva)] += einsum(x54, (0, 1, 2, 3), (0, 1, 2, 3)) * -6.0
    del x54
    x55 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x55 += einsum(x26, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 0, 3, 4), (1, 2, 3, 4))
    del x26
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x55, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x55, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x55
    x56 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x56 += einsum(f.aa.ov, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (0, 2, 3, 4))
    x57 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x57 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x56, (0, 2, 3, 4), (2, 3, 1, 4))
    del x56
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x57, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x57
    x58 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x58 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x59 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x59 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x58, (2, 3, 0, 4), (2, 3, 1, 4))
    del x58
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x59, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x59, (0, 1, 2, 3), (1, 0, 3, 2))
    del x59
    x60 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x60 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    x61 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x61 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x60, (2, 0, 3, 4), (2, 3, 1, 4))
    del x60
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x61, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x61, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x61
    x62 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x62 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x35, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x62, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x62, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x62, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x62
    x63 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x63 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x64 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x64 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x63, (2, 3, 1, 4), (0, 2, 3, 4))
    del x63
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x64, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x64
    x65 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x65 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x66 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x66 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x65, (2, 3, 0, 4), (2, 3, 1, 4))
    del x65
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x66, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x66, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x66, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x66, (0, 1, 2, 3), (1, 0, 3, 2))
    del x66
    x67 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x67 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vvov, (2, 1, 3, 4), (0, 3, 2, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x67, (4, 1, 5, 3), (4, 0, 5, 2)) * 2.0
    x68 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x68 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x67, (4, 1, 5, 3), (4, 0, 2, 5))
    del x67
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x68, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x68, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x68, (0, 1, 2, 3), (1, 0, 2, 3))
    del x68
    x69 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x69 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x70 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x70 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x69, (4, 5, 0, 1), (4, 5, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x70, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x70, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x70, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x70, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x70
    x71 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x71 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x72 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x72 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x71, (2, 3, 0, 4), (2, 3, 1, 4))
    del x71
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x72, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x72, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x72, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x72, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x72
    x73 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x73 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x74 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x74 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x73, (2, 3, 0, 4), (2, 3, 1, 4))
    del x73
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x74, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x74, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x74, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x74, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x74
    x75 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x75 += einsum(x27, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x27
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x75, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x75, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x75
    x76 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x76 += einsum(x28, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x28
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x76, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x76, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x76
    x77 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x77 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x77, (4, 0, 5, 2), (4, 1, 5, 3)) * -1.0
    x78 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x78 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x77, (4, 1, 5, 3), (4, 0, 2, 5))
    del x77
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x78, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x78, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x78, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x78, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x78
    x79 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x79 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x45, (4, 1, 3, 5), (4, 0, 2, 5))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x79, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x79, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x79, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x79, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x79
    x80 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x80 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x81 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x81 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x80, (2, 3, 0, 4), (2, 3, 1, 4))
    del x80
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x81, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x81, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x81, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x81, (0, 1, 2, 3), (0, 1, 3, 2))
    del x81
    x82 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x82 += einsum(x30, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 4, 0))
    del x30
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x82, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x82, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x82
    x83 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x83 += einsum(x29, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 0), (2, 3, 4, 1))
    del x29
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x83, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x83, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x83
    x84 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x84 += einsum(x31, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x31
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x84, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x84, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x84
    x85 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x85 += einsum(x32, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 4, 0))
    del x32
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x85, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x85, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x85
    x86 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x86 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vOOV, (1, 2, 3, 4), (2, 3, 4, 0))
    t2new_abab[np.ix_(soa,sOb,sVa,sVb)] += einsum(x86, (0, 1, 2, 3), t3.babbab, (4, 0, 1, 5, 6, 2), (3, 4, 6, 5)) * -2.0
    x87 = np.zeros((naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x87 += einsum(x86, (0, 1, 2, 3), t3.abaaba, (4, 1, 0, 5, 2, 6), (4, 5, 6, 3))
    del x86
    t2new_aaaa[np.ix_(soa,sOa,sVa,sVa)] += einsum(x87, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    t2new_aaaa[np.ix_(sOa,soa,sVa,sVa)] += einsum(x87, (0, 1, 2, 3), (0, 3, 2, 1)) * 2.0
    del x87
    x88 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x88 += einsum(v.aabb.oVOV, (0, 1, 2, 3), t3.abaaba, (4, 2, 5, 6, 3, 1), (4, 5, 6, 0))
    x89 = np.zeros((naocc[0], naocc[0], navir[0], nvir[0]), dtype=types[float])
    x89 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x88, (2, 3, 4, 0), (3, 2, 4, 1)) * -1.0
    del x88
    t2new_aaaa[np.ix_(sOa,sOa,sva,sVa)] += einsum(x89, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(sOa,sOa,sVa,sva)] += einsum(x89, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x89
    x90 = np.zeros((naocc[1], navir[1]), dtype=types[float])
    x90 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovOV, (0, 1, 2, 3), (2, 3))
    t2new_aaaa[np.ix_(sOa,sOa,sVa,sVa)] += einsum(x90, (0, 1), t3.abaaba, (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    t2new_abab[np.ix_(sOa,sOb,sVa,sVb)] += einsum(x90, (0, 1), t3.babbab, (2, 3, 0, 4, 5, 1), (3, 2, 5, 4)) * 2.0
    t2new_bbbb[np.ix_(sOb,sOb,sVb,sVb)] += einsum(x90, (0, 1), t3.bbbbbb, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    del x90
    x91 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x91 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.vOOV, (1, 2, 3, 4), (3, 2, 4, 0))
    t2new_abab[np.ix_(soa,sOb,sVa,sVb)] += einsum(x91, (0, 1, 2, 3), t3.abaaba, (1, 4, 0, 5, 6, 2), (3, 4, 5, 6)) * -1.0
    x92 = np.zeros((naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x92 += einsum(x91, (0, 1, 2, 3), t3.aaaaaa, (4, 1, 0, 5, 6, 2), (4, 5, 6, 3)) * -1.0
    del x91
    t2new_aaaa[np.ix_(soa,sOa,sVa,sVa)] += einsum(x92, (0, 1, 2, 3), (3, 0, 2, 1)) * 3.0
    t2new_aaaa[np.ix_(sOa,soa,sVa,sVa)] += einsum(x92, (0, 1, 2, 3), (0, 3, 2, 1)) * -3.0
    del x92
    x93 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x93 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.vOOV, (1, 2, 3, 4), (3, 2, 4, 0))
    t2new_abab[np.ix_(soa,sOb,sVa,sVb)] += einsum(x93, (0, 1, 2, 3), t3.abaaba, (1, 4, 0, 5, 6, 2), (3, 4, 5, 6)) * -1.0
    x94 = np.zeros((naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x94 += einsum(x93, (0, 1, 2, 3), t3.aaaaaa, (4, 1, 0, 5, 6, 2), (4, 5, 6, 3)) * -1.0
    del x93
    t2new_aaaa[np.ix_(soa,sOa,sVa,sVa)] += einsum(x94, (0, 1, 2, 3), (3, 0, 2, 1)) * 3.0
    t2new_aaaa[np.ix_(sOa,soa,sVa,sVa)] += einsum(x94, (0, 1, 2, 3), (0, 3, 2, 1)) * -3.0
    del x94
    x95 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x95 += einsum(v.aaaa.oVOV, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 2, 6, 3, 1), (4, 5, 6, 0))
    x96 = np.zeros((naocc[0], naocc[0], navir[0], nvir[0]), dtype=types[float])
    x96 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x95, (2, 3, 4, 0), (3, 2, 4, 1)) * -1.0
    del x95
    t2new_aaaa[np.ix_(sOa,sOa,sva,sVa)] += einsum(x96, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.0
    t2new_aaaa[np.ix_(sOa,sOa,sVa,sva)] += einsum(x96, (0, 1, 2, 3), (0, 1, 2, 3)) * 6.0
    del x96
    x97 = np.zeros((naocc[0], navir[0]), dtype=types[float])
    x97 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovOV, (0, 1, 2, 3), (2, 3))
    t2new_aaaa[np.ix_(sOa,sOa,sVa,sVa)] += einsum(x97, (0, 1), t3.aaaaaa, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    t2new_abab[np.ix_(sOa,sOb,sVa,sVb)] += einsum(x97, (0, 1), t3.abaaba, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 2.0
    t2new_bbbb[np.ix_(sOb,sOb,sVb,sVb)] += einsum(x97, (0, 1), t3.babbab, (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    del x97
    x98 = np.zeros((naocc[0], navir[0]), dtype=types[float])
    x98 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oVvO, (0, 2, 1, 3), (3, 2))
    t2new_aaaa[np.ix_(sOa,sOa,sVa,sVa)] += einsum(x98, (0, 1), t3.aaaaaa, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * -6.0
    t2new_abab[np.ix_(sOa,sOb,sVa,sVb)] += einsum(x98, (0, 1), t3.abaaba, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * -2.0
    t2new_bbbb[np.ix_(sOb,sOb,sVb,sVb)] += einsum(x98, (0, 1), t3.babbab, (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * -2.0
    del x98
    x99 = np.zeros((naocc[1], navir[1]), dtype=types[float])
    x99 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovOV, (0, 1, 2, 3), (2, 3))
    t2new_aaaa[np.ix_(sOa,sOa,sVa,sVa)] += einsum(x99, (0, 1), t3.abaaba, (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    t2new_abab[np.ix_(sOa,sOb,sVa,sVb)] += einsum(x99, (0, 1), t3.babbab, (2, 3, 0, 4, 5, 1), (3, 2, 5, 4)) * 2.0
    t2new_bbbb[np.ix_(sOb,sOb,sVb,sVb)] += einsum(x99, (0, 1), t3.bbbbbb, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    del x99
    x100 = np.zeros((naocc[1], navir[1]), dtype=types[float])
    x100 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oVvO, (0, 2, 1, 3), (3, 2))
    t2new_aaaa[np.ix_(sOa,sOa,sVa,sVa)] += einsum(x100, (0, 1), t3.abaaba, (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * -2.0
    t2new_abab[np.ix_(sOa,sOb,sVa,sVb)] += einsum(x100, (0, 1), t3.babbab, (2, 3, 0, 4, 5, 1), (3, 2, 5, 4)) * -2.0
    t2new_bbbb[np.ix_(sOb,sOb,sVb,sVb)] += einsum(x100, (0, 1), t3.bbbbbb, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * -6.0
    del x100
    x101 = np.zeros((naocc[0], navir[0]), dtype=types[float])
    x101 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.OVov, (2, 3, 0, 1), (2, 3))
    t2new_aaaa[np.ix_(sOa,sOa,sVa,sVa)] += einsum(x101, (0, 1), t3.aaaaaa, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    t2new_abab[np.ix_(sOa,sOb,sVa,sVb)] += einsum(x101, (0, 1), t3.abaaba, (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 2.0
    t2new_bbbb[np.ix_(sOb,sOb,sVb,sVb)] += einsum(x101, (0, 1), t3.babbab, (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    del x101
    x102 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x102 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3))
    x103 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x103 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x102, (4, 1, 5, 3), (4, 0, 5, 2))
    del x102
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x103, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x103, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x103
    x104 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x104 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    x105 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x105 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x104, (4, 1, 5, 3), (4, 0, 5, 2))
    del x104
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x105, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x105, (0, 1, 2, 3), (0, 1, 3, 2))
    del x105
    x106 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x106 += einsum(x34, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x34
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x106, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x106, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x106
    x107 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x107 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x46, (4, 1, 5, 3), (0, 4, 2, 5))
    del x46
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x107, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x107, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x107, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x107, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x107
    x108 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x108 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x108, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 3, 0, 4)) * -1.0
    x109 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x109 += einsum(x108, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 4, 0))
    del x108
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x109, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x109, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x109
    x110 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x110 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x49, (4, 1, 5, 3), (4, 0, 5, 2))
    del x49
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x110, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x110, (0, 1, 2, 3), (0, 1, 3, 2)) * -4.0
    del x110
    x111 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x111 += einsum(x37, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x111, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x111, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x111, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x111, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x111
    x112 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x112 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    x113 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x113 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x112, (4, 1, 5, 3), (4, 0, 5, 2))
    del x112
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x113, (0, 1, 2, 3), (0, 1, 2, 3)) * -4.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x113, (0, 1, 2, 3), (0, 1, 3, 2)) * 4.0
    del x113
    x114 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x114 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x115 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x115 += einsum(x114, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x115, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x115, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x115, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x115, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x115
    x116 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x116 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 5, 2), (0, 1, 4, 5)) * -1.0
    x117 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x117 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x116, (4, 5, 0, 1), (5, 4, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x117, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x117, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x117
    x118 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x118 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x69, (2, 3, 4, 0), (2, 4, 3, 1))
    del x69
    x119 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x119 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x118, (2, 0, 3, 4), (2, 3, 1, 4))
    del x118
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x119, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x119, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x119, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x119, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x119
    x120 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x120 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x45, (2, 3, 1, 4), (0, 2, 3, 4))
    del x45
    x121 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x121 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x120, (2, 3, 0, 4), (2, 3, 1, 4))
    del x120
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x121, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x121, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x121, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x121, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x121
    x122 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x122 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x33, (4, 5, 1, 3), (4, 0, 5, 2))
    x123 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x123 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x122, (2, 3, 0, 4), (2, 3, 1, 4))
    del x122
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x123, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x123, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x123, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x123, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x123
    x124 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x124 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x35, (2, 3, 4, 1), (2, 0, 4, 3))
    x125 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x125 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x124, (4, 5, 0, 1), (5, 4, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x125, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x125, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x125
    x126 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x126 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x35, (4, 1, 5, 3), (4, 0, 5, 2))
    x127 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x127 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x126, (2, 3, 0, 4), (2, 3, 1, 4))
    del x126
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x127, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x127, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x127, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x127, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x127
    x128 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x128 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x35, (4, 5, 1, 3), (4, 0, 5, 2))
    x129 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x129 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x128, (2, 3, 0, 4), (2, 3, 1, 4))
    del x128
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x129, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x129, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x129, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x129, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x129
    x130 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x130 += einsum(x39, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (0, 2, 3, 4))
    del x39
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x130, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x130, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x130
    x131 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x131 += einsum(x40, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (0, 2, 3, 4))
    del x40
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x131, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x131, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x131
    x132 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x132 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x116, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x116
    x133 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x133 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x132, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x132
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x133, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x133, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x133
    x134 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x134 += einsum(x11, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 0, 4))
    x135 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x135 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x134, (2, 3, 0, 4), (2, 3, 1, 4))
    del x134
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x135, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x135, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x135
    x136 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x136 += einsum(x12, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 0, 4))
    x137 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x137 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x136, (2, 3, 0, 4), (2, 3, 1, 4))
    del x136
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x137, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x137, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x137
    x138 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x138 += einsum(x41, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (0, 2, 3, 4))
    del x41
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x138, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x138, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x138
    x139 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x139 += einsum(x15, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 0, 4))
    x140 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x140 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x139, (2, 3, 0, 4), (2, 3, 1, 4))
    del x139
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x140, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x140, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x140
    x141 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x141 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x124, (2, 3, 0, 4), (3, 2, 4, 1))
    del x124
    x142 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x142 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x141, (2, 3, 0, 4), (2, 3, 1, 4))
    del x141
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x142, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x142, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x142
    x143 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x143 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (2, 0, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x143, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x143, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    x144 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x144 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 2), (4, 1, 5, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x144, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x144, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    x145 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x145 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (4, 0, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x145, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x145, (1, 4, 3, 5), (0, 4, 2, 5)) * 4.0
    x146 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x146 += einsum(f.aa.ov, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x146, (0, 2, 3, 4), (2, 3, 1, 4)) * -1.0
    del x146
    x147 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x147 += einsum(f.bb.ov, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 0, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x147, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    del x147
    x148 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x148 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.oovv, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x148, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    del x148
    x149 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x149 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x149, (2, 0, 3, 4), (2, 3, 1, 4))
    del x149
    x150 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x150 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x150, (2, 3, 1, 4), (0, 2, 3, 4))
    del x150
    x151 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x151 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vvoo, (2, 1, 3, 4), (0, 3, 4, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x151, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x151
    x152 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x152 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 0, 2), (4, 5, 1, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x152, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    del x152
    x153 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x153 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ooov, (4, 0, 5, 2), (4, 5, 1, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x153, (2, 0, 3, 4), (2, 3, 1, 4))
    del x153
    x154 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x154 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovoo, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x154, (4, 0, 5, 1), (4, 5, 2, 3))
    x155 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x155 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x155, (4, 0, 5, 3), (4, 1, 2, 5)) * -1.0
    del x155
    x156 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x156 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovoo, (4, 2, 5, 1), (0, 4, 5, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x156, (2, 0, 3, 4), (2, 3, 1, 4))
    del x156
    x157 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x157 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovvv, (4, 2, 5, 3), (0, 4, 1, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x157, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    del x157
    x158 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x158 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 3), (4, 5, 0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x158, (2, 0, 3, 4), (2, 3, 1, 4)) * -2.0
    del x158
    x159 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x159 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ooov, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x159, (4, 0, 5, 1), (4, 5, 2, 3))
    x160 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x160 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x160, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x160
    x161 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x161 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x161, (2, 3, 0, 4), (2, 3, 4, 1))
    del x161
    x162 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x162 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 0, 5, 3), (4, 1, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x162, (2, 3, 0, 4), (2, 3, 4, 1))
    del x162
    x163 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x163 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x163, (4, 1, 5, 3), (0, 4, 2, 5)) * -1.0
    x164 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x164 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x164, (4, 1, 3, 5), (0, 4, 2, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x164, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x164, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x164, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x164, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    x165 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x165 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.vvov, (2, 3, 4, 1), (0, 4, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x165, (4, 1, 5, 2), (0, 4, 5, 3)) * -1.0
    x166 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x166 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vvov, (4, 2, 5, 3), (0, 1, 5, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x166, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x166
    x167 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x167 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aabb.ovoo, (1, 3, 4, 5), (0, 4, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x167, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    del x167
    x168 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x168 += einsum(v.aaaa.oVOV, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 3, 6, 1), (4, 5, 6, 0))
    t2new_abab[np.ix_(sOa,sOb,sva,sVb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x168, (2, 3, 4, 0), (2, 3, 1, 4)) * 2.0
    del x168
    x169 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x169 += einsum(v.aabb.oVOV, (0, 1, 2, 3), t3.babbab, (4, 5, 2, 6, 1, 3), (5, 4, 6, 0))
    t2new_abab[np.ix_(sOa,sOb,sva,sVb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x169, (2, 3, 4, 0), (2, 3, 1, 4)) * -2.0
    del x169
    x170 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x170 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.OVvO, (2, 3, 1, 4), (2, 4, 3, 0))
    t2new_abab[np.ix_(sOa,sob,sVa,sVb)] += einsum(x170, (0, 1, 2, 3), t3.abaaba, (4, 1, 0, 5, 6, 2), (4, 3, 5, 6)) * -2.0
    x171 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x171 += einsum(v.aabb.OVoV, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 3, 1), (4, 5, 6, 2))
    t2new_abab[np.ix_(sOa,sOb,sVa,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x171, (2, 3, 4, 0), (2, 3, 4, 1)) * -2.0
    del x171
    x172 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x172 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.vOOV, (1, 2, 3, 4), (3, 2, 4, 0))
    t2new_abab[np.ix_(sOa,sob,sVa,sVb)] += einsum(x172, (0, 1, 2, 3), t3.babbab, (1, 4, 0, 5, 6, 2), (4, 3, 6, 5)) * -1.0
    x173 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x173 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.vOOV, (1, 2, 3, 4), (3, 2, 4, 0))
    t2new_abab[np.ix_(sOa,sob,sVa,sVb)] += einsum(x173, (0, 1, 2, 3), t3.babbab, (0, 4, 1, 5, 6, 2), (4, 3, 6, 5))
    x174 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x174 += einsum(v.bbbb.oVOV, (0, 1, 2, 3), t3.babbab, (4, 5, 2, 1, 6, 3), (5, 4, 6, 0)) * -1.0
    t2new_abab[np.ix_(sOa,sOb,sVa,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x174, (2, 3, 4, 0), (2, 3, 4, 1)) * 2.0
    del x174
    x175 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x175 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (0, 4, 1, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x175, (4, 0, 5, 1), (4, 5, 2, 3))
    x176 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x176 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x176, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 4, 0)) * -1.0
    x177 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x177 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 4, 5, 3), (1, 5, 2, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x177, (4, 1, 5, 2), (0, 4, 5, 3))
    del x177
    x178 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x178 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 4, 3, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x178, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x178, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x178, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x178, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x178, (0, 1, 2, 3), (1, 0, 3, 2))
    del x178
    x179 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x179 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x179, (4, 1, 5, 3), (0, 4, 2, 5)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x179, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x179, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x179, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x179, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    x180 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x180 += einsum(x18, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x180, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x180, (0, 1, 2, 3), (0, 1, 2, 3))
    del x180
    x181 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x181 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x181, (4, 1, 5, 3), (0, 4, 2, 5)) * -2.0
    x182 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x182 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 4), (2, 4)) * -1.0
    x183 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x183 += einsum(x182, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 4, 0)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x183, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x183, (0, 1, 2, 3), (0, 1, 2, 3))
    del x183
    x184 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x184 += einsum(x114, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 3, 0, 4)) * -1.0
    del x114
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x184, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x184, (0, 1, 2, 3), (0, 1, 2, 3))
    del x184
    x185 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x185 += einsum(x37, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    del x37
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x185, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x185, (0, 1, 2, 3), (0, 1, 2, 3))
    del x185
    x186 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x186 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 0, 5), (4, 1, 5, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x186, (1, 4, 3, 5), (0, 4, 2, 5)) * -2.0
    x187 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x187 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x143, (2, 3, 1, 4), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x187, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    del x187
    x188 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x188 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x154, (2, 3, 4, 0), (2, 3, 4, 1))
    del x154
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x188, (2, 0, 3, 4), (2, 3, 1, 4))
    del x188
    x189 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x189 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x159, (2, 3, 4, 0), (3, 2, 4, 1))
    del x159
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x189, (0, 2, 3, 4), (2, 3, 1, 4))
    del x189
    x190 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x190 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x165, (2, 3, 4, 1), (0, 2, 3, 4))
    del x165
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x190, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x190
    x191 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x191 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x35, (4, 0, 5, 2), (4, 5, 1, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x191, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    del x191
    x192 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x192 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x35, (4, 5, 0, 2), (4, 5, 1, 3))
    del x35
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x192, (2, 0, 3, 4), (2, 3, 1, 4))
    del x192
    x193 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x193 += einsum(x11, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 0, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x193, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    del x193
    x194 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x194 += einsum(x12, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 0, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x194, (2, 0, 3, 4), (2, 3, 1, 4))
    del x194
    x195 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x195 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x33, (4, 5, 1, 3), (4, 5, 0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x195, (2, 0, 3, 4), (2, 3, 1, 4)) * -2.0
    del x195
    x196 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x196 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x13, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x196, (4, 0, 5, 1), (4, 5, 2, 3))
    x197 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x197 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x33, (4, 0, 5, 3), (4, 1, 5, 2))
    del x33
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x197, (2, 3, 0, 4), (2, 3, 4, 1))
    del x197
    x198 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x198 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x13, (4, 5, 1, 2), (0, 4, 5, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x198, (2, 0, 3, 4), (2, 3, 1, 4))
    del x198
    x199 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x199 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x175, (2, 3, 4, 0), (2, 3, 4, 1))
    del x175
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x199, (2, 0, 3, 4), (2, 3, 1, 4))
    del x199
    x200 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x200 += einsum(x15, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 0, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x200, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    del x200
    x201 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x201 += einsum(x0, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x201, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x201
    x202 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x202 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x16, (4, 1, 5, 3), (0, 4, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x202, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x202
    x203 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x203 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x16, (4, 5, 1, 3), (0, 4, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x203, (2, 3, 0, 4), (2, 3, 4, 1))
    del x203
    x204 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x204 += einsum(x1, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x204, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x204
    x205 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x205 += einsum(x20, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x205, (2, 3, 0, 4), (2, 3, 4, 1))
    del x205
    x206 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x206 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x13, (1, 4, 5, 3), (0, 4, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x206, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    del x206
    x207 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x207 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x196, (2, 3, 4, 0), (2, 3, 4, 1))
    del x196
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x207, (2, 0, 3, 4), (2, 3, 1, 4))
    del x207
    x208 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x208 += einsum(f.bb.oo, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x208, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x208, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x208
    x209 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x209 += einsum(f.bb.vv, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x209, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x209, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x209
    x210 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x210 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x210, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x210, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x210, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x210, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x210
    x211 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x211 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.oooo, (4, 0, 5, 1), (4, 5, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x211, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x211, (0, 1, 2, 3), (1, 0, 3, 2))
    del x211
    x212 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x212 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x212, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x212, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x212, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x212, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x212
    x213 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x213 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x213, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x213, (0, 1, 2, 3), (1, 0, 3, 2))
    del x213
    x214 = np.zeros((naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x214 += einsum(v.aabb.OVoO, (0, 1, 2, 3), t3.babbab, (4, 0, 3, 5, 1, 6), (4, 5, 6, 2))
    t2new_bbbb[np.ix_(sob,sOb,sVb,sVb)] += einsum(x214, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    t2new_bbbb[np.ix_(sOb,sob,sVb,sVb)] += einsum(x214, (0, 1, 2, 3), (0, 3, 2, 1)) * 2.0
    del x214
    x215 = np.zeros((naocc[1], naocc[1], navir[1], nvir[1]), dtype=types[float])
    x215 += einsum(v.aabb.OVvV, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 1, 3), (4, 5, 6, 2))
    t2new_bbbb[np.ix_(sOb,sOb,svb,sVb)] += einsum(x215, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sOb,sOb,sVb,svb)] += einsum(x215, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x215
    x216 = np.zeros((naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x216 += einsum(v.bbbb.oOOV, (0, 1, 2, 3), t3.bbbbbb, (4, 1, 2, 5, 6, 3), (4, 5, 6, 0)) * -1.0
    t2new_bbbb[np.ix_(sob,sOb,sVb,sVb)] += einsum(x216, (0, 1, 2, 3), (3, 0, 2, 1)) * 6.0
    t2new_bbbb[np.ix_(sOb,sob,sVb,sVb)] += einsum(x216, (0, 1, 2, 3), (0, 3, 2, 1)) * -6.0
    del x216
    x217 = np.zeros((naocc[1], naocc[1], navir[1], nvir[1]), dtype=types[float])
    x217 += einsum(v.bbbb.vVOV, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 2, 6, 1, 3), (4, 5, 6, 0)) * -1.0
    t2new_bbbb[np.ix_(sOb,sOb,svb,sVb)] += einsum(x217, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.0
    t2new_bbbb[np.ix_(sOb,sOb,sVb,svb)] += einsum(x217, (0, 1, 2, 3), (0, 1, 2, 3)) * -6.0
    del x217
    x218 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x218 += einsum(x4, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 0, 3, 4), (1, 2, 3, 4))
    del x4
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x218, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x218, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x218
    x219 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x219 += einsum(f.bb.ov, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (0, 2, 3, 4))
    x220 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x220 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x219, (0, 2, 3, 4), (2, 3, 1, 4))
    del x219
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x220, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x220, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x220
    x221 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x221 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x222 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x222 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x221, (2, 3, 0, 4), (2, 3, 1, 4))
    del x221
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x222, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x222, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x222, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x222, (0, 1, 2, 3), (1, 0, 3, 2))
    del x222
    x223 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x223 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    x224 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x224 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x223, (2, 0, 3, 4), (2, 3, 1, 4))
    del x223
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x224, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x224, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x224
    x225 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x225 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x16, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x225, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x225, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x225, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x225, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x225
    x226 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x226 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x227 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x227 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x226, (2, 3, 1, 4), (0, 2, 3, 4))
    del x226
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x227, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x227, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x227
    x228 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x228 += einsum(x6, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x6
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x228, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x228, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x228
    x229 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x229 += einsum(x5, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    del x5
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x229, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x229, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x229
    x230 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x230 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x143, (0, 4, 2, 5), (4, 1, 3, 5))
    del x143
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x230, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x230, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x230, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x230, (0, 1, 2, 3), (1, 0, 2, 3))
    del x230
    x231 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x231 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 5), (1, 4, 5, 3))
    x232 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x232 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x231, (2, 3, 0, 4), (2, 3, 1, 4))
    del x231
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x232, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x232, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x232, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x232, (0, 1, 2, 3), (1, 0, 3, 2))
    del x232
    x233 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x233 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x234 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x234 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x233, (4, 5, 0, 1), (4, 5, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x234, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x234, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x234, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x234, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x234
    x235 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x235 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x236 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x236 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x235, (2, 3, 0, 4), (2, 3, 1, 4))
    del x235
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x236, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x236, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x236, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x236, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x236
    x237 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x237 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x238 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x238 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x237, (2, 3, 0, 4), (2, 3, 1, 4))
    del x237
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x238, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x238, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x238, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x238, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x238
    x239 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x239 += einsum(x7, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x7
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x239, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x239, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x239
    x240 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x240 += einsum(x8, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x8
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x240, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x240, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x240
    x241 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x241 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x163, (4, 1, 5, 3), (4, 0, 2, 5))
    del x163
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x241, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x241, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x241, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x241, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x241
    x242 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x242 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x164, (4, 1, 3, 5), (4, 0, 2, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x242, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x242, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x242, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x242, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x242
    x243 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x243 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x244 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x244 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x243, (2, 3, 0, 4), (2, 3, 1, 4))
    del x243
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x244, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x244, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x244, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x244, (0, 1, 2, 3), (0, 1, 3, 2))
    del x244
    x245 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x245 += einsum(x10, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    del x10
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x245, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x245, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x245
    x246 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x246 += einsum(x9, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 0), (2, 3, 4, 1))
    del x9
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x246, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x246, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x246
    x247 = np.zeros((naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x247 += einsum(x170, (0, 1, 2, 3), t3.babbab, (4, 0, 1, 5, 2, 6), (4, 5, 6, 3))
    del x170
    t2new_bbbb[np.ix_(sob,sOb,sVb,sVb)] += einsum(x247, (0, 1, 2, 3), (3, 0, 2, 1)) * -2.0
    t2new_bbbb[np.ix_(sOb,sob,sVb,sVb)] += einsum(x247, (0, 1, 2, 3), (0, 3, 2, 1)) * 2.0
    del x247
    x248 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x248 += einsum(v.aabb.OVoV, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 1, 3), (4, 5, 6, 2))
    x249 = np.zeros((naocc[1], naocc[1], navir[1], nvir[1]), dtype=types[float])
    x249 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x248, (2, 3, 4, 0), (3, 2, 4, 1)) * -1.0
    del x248
    t2new_bbbb[np.ix_(sOb,sOb,svb,sVb)] += einsum(x249, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sOb,sOb,sVb,svb)] += einsum(x249, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x249
    x250 = np.zeros((naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x250 += einsum(x172, (0, 1, 2, 3), t3.bbbbbb, (4, 0, 1, 5, 6, 2), (4, 5, 6, 3))
    del x172
    t2new_bbbb[np.ix_(sob,sOb,sVb,sVb)] += einsum(x250, (0, 1, 2, 3), (3, 0, 2, 1)) * 3.0
    t2new_bbbb[np.ix_(sOb,sob,sVb,sVb)] += einsum(x250, (0, 1, 2, 3), (0, 3, 2, 1)) * -3.0
    del x250
    x251 = np.zeros((naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x251 += einsum(x173, (0, 1, 2, 3), t3.bbbbbb, (4, 1, 0, 5, 6, 2), (4, 5, 6, 3)) * -1.0
    del x173
    t2new_bbbb[np.ix_(sob,sOb,sVb,sVb)] += einsum(x251, (0, 1, 2, 3), (3, 0, 2, 1)) * 3.0
    t2new_bbbb[np.ix_(sOb,sob,sVb,sVb)] += einsum(x251, (0, 1, 2, 3), (0, 3, 2, 1)) * -3.0
    del x251
    x252 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x252 += einsum(v.bbbb.oVOV, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 2, 6, 3, 1), (4, 5, 6, 0))
    x253 = np.zeros((naocc[1], naocc[1], navir[1], nvir[1]), dtype=types[float])
    x253 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x252, (2, 3, 4, 0), (3, 2, 4, 1)) * -1.0
    del x252
    t2new_bbbb[np.ix_(sOb,sOb,svb,sVb)] += einsum(x253, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.0
    t2new_bbbb[np.ix_(sOb,sOb,sVb,svb)] += einsum(x253, (0, 1, 2, 3), (0, 1, 2, 3)) * 6.0
    del x253
    x254 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x254 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x144, (0, 4, 2, 5), (4, 1, 5, 3))
    del x144
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x254, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x254, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x254
    x255 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x255 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x186, (0, 4, 2, 5), (4, 1, 5, 3))
    del x186
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x255, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x255, (0, 1, 2, 3), (1, 0, 2, 3))
    del x255
    x256 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x256 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x145, (0, 4, 2, 5), (1, 4, 3, 5))
    del x145
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x256, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x256, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x256, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x256, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x256
    x257 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x257 += einsum(x14, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x14
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x257, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x257, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x257
    x258 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x258 += einsum(x176, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    del x176
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x258, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x258, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x258
    x259 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x259 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x179, (4, 1, 5, 3), (0, 4, 2, 5))
    del x179
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x259, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x259, (0, 1, 2, 3), (0, 1, 3, 2)) * -4.0
    del x259
    x260 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x260 += einsum(x18, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    del x18
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x260, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x260, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x260, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x260, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x260
    x261 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x261 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x181, (4, 1, 5, 3), (0, 4, 2, 5))
    del x181
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x261, (0, 1, 2, 3), (0, 1, 2, 3)) * -4.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x261, (0, 1, 2, 3), (0, 1, 3, 2)) * 4.0
    del x261
    x262 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x262 += einsum(x182, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    del x182
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x262, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x262, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x262, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x262, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x262
    x263 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x263 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x264 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x264 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x263, (4, 5, 0, 1), (5, 4, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x264, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x264, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x264
    x265 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x265 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x233, (2, 3, 4, 0), (2, 4, 3, 1))
    del x233
    x266 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x266 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x265, (2, 0, 3, 4), (2, 3, 1, 4))
    del x265
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x266, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x266, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x266, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x266, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x266
    x267 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x267 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x164, (2, 3, 1, 4), (0, 2, 3, 4))
    del x164
    x268 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x268 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x267, (2, 3, 0, 4), (2, 3, 1, 4))
    del x267
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x268, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x268, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x268, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x268, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x268
    x269 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x269 += einsum(x21, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (0, 2, 3, 4))
    del x21
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x269, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x269, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x269
    x270 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x270 += einsum(x0, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    x271 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x271 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x270, (2, 3, 0, 4), (2, 3, 1, 4))
    del x270
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x271, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x271, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x271
    x272 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x272 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x13, (0, 4, 5, 2), (4, 1, 5, 3))
    del x13
    x273 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x273 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x272, (2, 3, 0, 4), (2, 3, 1, 4))
    del x272
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x273, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x273, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x273, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x273, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x273
    x274 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x274 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x16, (2, 3, 4, 1), (2, 0, 4, 3))
    x275 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x275 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x274, (4, 5, 0, 1), (5, 4, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x275, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x275, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x275
    x276 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x276 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x16, (4, 1, 5, 3), (4, 0, 5, 2))
    x277 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x277 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x276, (2, 3, 0, 4), (2, 3, 1, 4))
    del x276
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x277, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x277, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x277, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x277, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x277
    x278 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x278 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x16, (4, 5, 1, 3), (4, 0, 5, 2))
    del x16
    x279 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x279 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x278, (2, 3, 0, 4), (2, 3, 1, 4))
    del x278
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x279, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x279, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x279, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x279, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x279
    x280 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x280 += einsum(x22, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (0, 2, 3, 4))
    del x22
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x280, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x280, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x280
    x281 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x281 += einsum(x23, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (0, 2, 3, 4))
    del x23
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x281, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x281, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x281
    x282 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x282 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x263, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x263
    x283 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x283 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x282, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x282
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x283, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x283, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x283
    x284 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x284 += einsum(x1, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    x285 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x285 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x284, (2, 3, 0, 4), (2, 3, 1, 4))
    del x284
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x285, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x285, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x285
    x286 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x286 += einsum(x20, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    x287 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x287 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x286, (2, 3, 0, 4), (2, 3, 1, 4))
    del x286
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x287, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x287, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x287
    x288 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x288 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x274, (2, 3, 0, 4), (3, 2, 4, 1))
    del x274
    x289 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x289 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x288, (2, 3, 0, 4), (2, 3, 1, 4))
    del x288
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x289, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x289, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x289
    x290 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x290 += einsum(f.aa.OO, (0, 1), t3.aaaaaa, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    t3new_aaaaaa = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    t3new_aaaaaa += einsum(x290, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x290, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x290, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    del x290
    x291 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x291 += einsum(f.aa.VV, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    t3new_aaaaaa += einsum(x291, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x291, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x291, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x291
    x292 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x292 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), v.aaaa.oOOV, (0, 4, 5, 6), (1, 5, 4, 2, 3, 6)) * -1.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x292, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    del x292
    x293 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x293 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.vVOV, (2, 4, 5, 6), (0, 1, 5, 3, 6, 4)) * -1.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x293, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    del x293
    x294 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x294 += einsum(v.aabb.OVOV, (0, 1, 2, 3), t3.abaaba, (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1))
    t3new_aaaaaa += einsum(x294, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x294, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x294, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x294, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x294, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x294, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x294, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x294, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x294, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    del x294
    x295 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x295 += einsum(v.aaaa.OOOO, (0, 1, 2, 3), t3.aaaaaa, (4, 3, 1, 5, 6, 7), (4, 0, 2, 5, 6, 7)) * -1.0
    t3new_aaaaaa += einsum(x295, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x295, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x295, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    del x295
    x296 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x296 += einsum(v.aaaa.OOVV, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    t3new_aaaaaa += einsum(x296, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x296, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x296, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -6.0
    t3new_aaaaaa += einsum(x296, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x296, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x296, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -6.0
    t3new_aaaaaa += einsum(x296, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x296, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x296, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 6.0
    del x296
    x297 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x297 += einsum(v.aaaa.OVOV, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 2, 6, 7, 3), (4, 5, 0, 6, 7, 1))
    t3new_aaaaaa += einsum(x297, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x297, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x297, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 6.0
    t3new_aaaaaa += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 6.0
    t3new_aaaaaa += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x297, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -6.0
    del x297
    x298 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x298 += einsum(v.aaaa.VVVV, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 6, 7, 3, 1), (4, 5, 6, 7, 0, 2)) * -1.0
    t3new_aaaaaa += einsum(x298, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x298, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x298, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    del x298
    x299 = np.zeros((naocc[0], naocc[0]), dtype=types[float])
    x299 += einsum(f.aa.vO, (0, 1), t1.aa[np.ix_(sOa,sva)], (2, 0), (1, 2))
    t3new_babbab += einsum(x299, (0, 1), t3.babbab, (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -2.0
    x300 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x300 += einsum(x299, (0, 1), t3.aaaaaa, (2, 3, 0, 4, 5, 6), (1, 2, 3, 4, 5, 6))
    t3new_aaaaaa += einsum(x300, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x300, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x300, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    del x300
    x301 = np.zeros((navir[0], navir[0]), dtype=types[float])
    x301 += einsum(f.aa.oV, (0, 1), t1.aa[np.ix_(soa,sVa)], (0, 2), (1, 2))
    t3new_babbab += einsum(x301, (0, 1), t3.babbab, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -2.0
    x302 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x302 += einsum(x301, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    t3new_aaaaaa += einsum(x302, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x302, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x302, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x302
    x303 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x303 += einsum(f.aa.ov, (0, 1), t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    x304 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x304 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x303, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    t3new_aaaaaa += einsum(x304, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x304, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x304, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x304, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x304, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x304, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x304, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x304, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x304, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    del x304
    x305 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x305 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.oOvV, (2, 3, 1, 4), (0, 3, 4, 2))
    x306 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x306 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x305, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x306
    x307 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x307 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), v.aaaa.oOoO, (4, 5, 0, 6), (1, 5, 6, 2, 3, 4)) * -1.0
    x308 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x308 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x307, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x307
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x308, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x308
    x309 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x309 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.oOvV, (4, 5, 2, 6), (0, 1, 5, 3, 6, 4)) * -1.0
    x310 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x310 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x309, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x309
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x310
    x311 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x311 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.ovOV, (2, 1, 3, 4), (0, 3, 4, 2))
    x312 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x312 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x311, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    del x312
    x313 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x313 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovOV, (4, 2, 5, 6), (0, 1, 5, 3, 6, 4)) * -1.0
    x314 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x314 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x313, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x313
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x314, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    del x314
    x315 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[0], nvir[0]), dtype=types[float])
    x315 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.vVvV, (4, 5, 2, 6), (0, 1, 3, 5, 6, 4)) * -1.0
    x316 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x316 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x315, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x315
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -2.0
    t3new_aaaaaa += einsum(x316, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    del x316
    x317 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x317 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.oOOV, (0, 2, 3, 4), (2, 3, 1, 4))
    t3new_babbab += einsum(x317, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (4, 0, 5, 6, 2, 7)) * -6.0
    x318 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x318 += einsum(x317, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (4, 5, 0, 2, 6, 7))
    t3new_aaaaaa += einsum(x318, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x318, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x318, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x318, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x318, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x318, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x318, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x318, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x318, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    del x318
    x319 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x319 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vVOV, (1, 2, 3, 4), (0, 3, 2, 4))
    t3new_babbab += einsum(x319, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (4, 0, 5, 6, 2, 7)) * 6.0
    x320 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x320 += einsum(x319, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (0, 4, 5, 6, 7, 2))
    t3new_aaaaaa += einsum(x320, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x320, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x320, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x320, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x320, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x320, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x320, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x320, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x320, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    del x320
    x321 = np.zeros((naocc[0], naocc[0], naocc[0], naocc[0]), dtype=types[float])
    x321 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.vOOO, (1, 2, 3, 4), (0, 3, 4, 2))
    x322 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x322 += einsum(x321, (0, 1, 2, 3), t3.aaaaaa, (4, 3, 2, 5, 6, 7), (0, 4, 1, 5, 6, 7)) * -1.0
    t3new_aaaaaa += einsum(x322, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x322, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x322, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x322, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x322, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x322, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 6.0
    del x322
    x323 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=types[float])
    x323 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oOOV, (0, 2, 3, 4), (3, 2, 1, 4))
    t3new_babbab += einsum(x323, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 3, 7), (4, 1, 5, 6, 2, 7)) * -2.0
    x324 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x324 += einsum(x323, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 7, 3), (4, 5, 1, 2, 6, 7))
    t3new_aaaaaa += einsum(x324, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x324, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x324, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x324, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x324, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x324, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x324, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x324, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x324, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x324
    x325 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=types[float])
    x325 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oVOO, (0, 2, 3, 4), (3, 4, 1, 2))
    t3new_babbab += einsum(x325, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 2.0
    x326 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x326 += einsum(x325, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 1, 6, 7, 3), (4, 5, 0, 2, 6, 7))
    t3new_aaaaaa += einsum(x326, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x326, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x326, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x326, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x326, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x326, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x326, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x326, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x326, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x326
    x327 = np.zeros((naocc[0], naocc[0]), dtype=types[float])
    x327 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oOvO, (0, 2, 1, 3), (2, 3))
    t3new_babbab += einsum(x327, (0, 1), t3.babbab, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * 2.0
    x328 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x328 += einsum(x327, (0, 1), t3.aaaaaa, (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6))
    t3new_aaaaaa += einsum(x328, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x328, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x328, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    del x328
    x329 = np.zeros((naocc[0], naocc[0]), dtype=types[float])
    x329 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovOO, (0, 1, 2, 3), (2, 3))
    t3new_babbab += einsum(x329, (0, 1), t3.babbab, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x330 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x330 += einsum(x329, (0, 1), t3.aaaaaa, (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6))
    t3new_aaaaaa += einsum(x330, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x330, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x330, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -6.0
    del x330
    x331 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=types[float])
    x331 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.vOVV, (1, 2, 3, 4), (0, 2, 3, 4))
    t3new_babbab += einsum(x331, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -2.0
    x332 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x332 += einsum(x331, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 1, 6, 7, 3), (0, 4, 5, 6, 7, 2))
    t3new_aaaaaa += einsum(x332, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x332, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x332, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -6.0
    t3new_aaaaaa += einsum(x332, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x332, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x332, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -6.0
    t3new_aaaaaa += einsum(x332, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x332, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x332, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 6.0
    del x332
    x333 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=types[float])
    x333 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.vVOV, (1, 2, 3, 4), (0, 3, 4, 2))
    t3new_babbab += einsum(x333, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 2, 7), (4, 0, 5, 6, 3, 7)) * 2.0
    x334 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x334 += einsum(x333, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 1, 6, 7, 2), (0, 4, 5, 6, 7, 3))
    t3new_aaaaaa += einsum(x334, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x334, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x334, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 6.0
    t3new_aaaaaa += einsum(x334, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x334, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x334, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 6.0
    t3new_aaaaaa += einsum(x334, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x334, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x334, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -6.0
    del x334
    x335 = np.zeros((navir[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x335 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oVVV, (0, 2, 3, 4), (1, 3, 4, 2))
    x336 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x336 += einsum(x335, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 6, 7, 2, 3), (4, 5, 6, 0, 7, 1))
    t3new_aaaaaa += einsum(x336, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x336, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -6.0
    t3new_aaaaaa += einsum(x336, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x336, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 6.0
    t3new_aaaaaa += einsum(x336, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x336, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -6.0
    del x336
    x337 = np.zeros((navir[0], navir[0]), dtype=types[float])
    x337 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovVV, (0, 1, 2, 3), (2, 3))
    t3new_babbab += einsum(x337, (0, 1), t3.babbab, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    x338 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x338 += einsum(x337, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0))
    t3new_aaaaaa += einsum(x338, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x338, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x338, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -6.0
    del x338
    x339 = np.zeros((navir[0], navir[0]), dtype=types[float])
    x339 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oVvV, (0, 2, 1, 3), (2, 3))
    t3new_babbab += einsum(x339, (0, 1), t3.babbab, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -2.0
    x340 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x340 += einsum(x339, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    t3new_aaaaaa += einsum(x340, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x340, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x340, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 6.0
    del x340
    x341 = np.zeros((naocc[0], naocc[0]), dtype=types[float])
    x341 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.OOov, (2, 3, 0, 1), (2, 3))
    t3new_babbab += einsum(x341, (0, 1), t3.babbab, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x342 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x342 += einsum(x341, (0, 1), t3.aaaaaa, (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6))
    t3new_aaaaaa += einsum(x342, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x342, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x342, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -6.0
    del x342
    x343 = np.zeros((navir[0], navir[0]), dtype=types[float])
    x343 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.VVov, (2, 3, 0, 1), (2, 3))
    t3new_babbab += einsum(x343, (0, 1), t3.babbab, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    x344 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x344 += einsum(x343, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0))
    t3new_aaaaaa += einsum(x344, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x344, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x344, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -6.0
    del x344
    x345 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x345 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oOov, (4, 5, 1, 3), (0, 5, 2, 4))
    x346 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x346 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x345, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x346, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    del x346
    x347 = np.zeros((naocc[0], navir[0], navir[0], nvir[0]), dtype=types[float])
    x347 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.vVov, (4, 5, 1, 3), (0, 2, 5, 4))
    x348 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x348 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x347, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x348, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    del x348
    x349 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x349 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoO, (0, 2, 4, 5), (1, 5, 3, 4))
    x350 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x350 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x349, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x350, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 4.0
    del x350
    x351 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x351 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoO, (4, 2, 0, 5), (1, 5, 3, 4))
    x352 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x352 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x351, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x352, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 4.0
    del x352
    x353 = np.zeros((naocc[0], navir[0], navir[0], nvir[0]), dtype=types[float])
    x353 += einsum(t2.aaaa[np.ix_(soa,soa,sVa,sVa)], (0, 1, 2, 3), v.aaaa.ovoO, (1, 4, 0, 5), (5, 2, 3, 4))
    x354 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x354 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x353, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x354, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x354
    x355 = np.zeros((naocc[0], navir[0], navir[0], nvir[0]), dtype=types[float])
    x355 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovvV, (0, 4, 2, 5), (1, 3, 5, 4))
    x356 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x356 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x355, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x356, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    del x356
    x357 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x357 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvV, (4, 2, 3, 5), (0, 1, 5, 4)) * -1.0
    x358 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x358 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x357, (4, 5, 6, 0), (5, 4, 1, 2, 3, 6))
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x358, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    del x358
    x359 = np.zeros((naocc[0], navir[0], navir[0], nvir[0]), dtype=types[float])
    x359 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovvV, (0, 2, 4, 5), (1, 3, 5, 4))
    x360 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x360 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x359, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x360, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 4.0
    del x360
    x361 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x361 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovOV, (1, 3, 4, 5), (0, 4, 2, 5))
    t3new_babbab += einsum(x361, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (4, 0, 5, 6, 2, 7)) * 6.0000000000000595
    x362 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x362 += einsum(x361, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (0, 4, 5, 2, 6, 7))
    t3new_aaaaaa += einsum(x362, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.00000000000002
    t3new_aaaaaa += einsum(x362, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -2.00000000000002
    t3new_aaaaaa += einsum(x362, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.00000000000002
    t3new_aaaaaa += einsum(x362, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 2.00000000000002
    t3new_aaaaaa += einsum(x362, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.00000000000002
    t3new_aaaaaa += einsum(x362, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.00000000000002
    t3new_aaaaaa += einsum(x362, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.00000000000002
    t3new_aaaaaa += einsum(x362, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 2.00000000000002
    t3new_aaaaaa += einsum(x362, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.00000000000002
    del x362
    x363 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x363 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.oVvO, (1, 4, 3, 5), (0, 5, 2, 4))
    t3new_babbab += einsum(x363, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (4, 0, 5, 6, 2, 7)) * -6.0000000000000595
    x364 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x364 += einsum(x363, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (0, 4, 5, 2, 6, 7))
    t3new_aaaaaa += einsum(x364, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.00000000000002
    t3new_aaaaaa += einsum(x364, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.00000000000002
    t3new_aaaaaa += einsum(x364, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.00000000000002
    t3new_aaaaaa += einsum(x364, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.00000000000002
    t3new_aaaaaa += einsum(x364, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.00000000000002
    t3new_aaaaaa += einsum(x364, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.00000000000002
    t3new_aaaaaa += einsum(x364, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.00000000000002
    t3new_aaaaaa += einsum(x364, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.00000000000002
    t3new_aaaaaa += einsum(x364, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.00000000000002
    del x364
    x365 = np.zeros((naocc[0], naocc[0]), dtype=types[float])
    x365 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vOov, (2, 4, 1, 3), (0, 4))
    t3new_babbab += einsum(x365, (0, 1), t3.babbab, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x366 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x366 += einsum(x365, (0, 1), t3.aaaaaa, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    t3new_aaaaaa += einsum(x366, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x366, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x366, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    del x366
    x367 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=types[float])
    x367 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 1, 3), (0, 4, 2, 5))
    t3new_babbab += einsum(x367, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 2.00000000000002
    x368 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x368 += einsum(x367, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    t3new_aaaaaa += einsum(x368, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 6.0000000000000595
    t3new_aaaaaa += einsum(x368, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -6.0000000000000595
    t3new_aaaaaa += einsum(x368, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 6.0000000000000595
    t3new_aaaaaa += einsum(x368, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 6.0000000000000595
    t3new_aaaaaa += einsum(x368, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -6.0000000000000595
    t3new_aaaaaa += einsum(x368, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 6.0000000000000595
    t3new_aaaaaa += einsum(x368, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -6.0000000000000595
    t3new_aaaaaa += einsum(x368, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 6.0000000000000595
    t3new_aaaaaa += einsum(x368, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -6.0000000000000595
    del x368
    x369 = np.zeros((navir[0], navir[0]), dtype=types[float])
    x369 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oVov, (0, 4, 1, 3), (2, 4))
    t3new_babbab += einsum(x369, (0, 1), t3.babbab, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -2.0
    x370 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x370 += einsum(x369, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    t3new_aaaaaa += einsum(x370, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x370, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x370, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x370
    x371 = np.zeros((naocc[0], naocc[0], naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x371 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.vOOV, (2, 4, 5, 6), (0, 1, 4, 5, 3, 6)) * -1.0
    x372 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x372 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x371, (6, 7, 2, 1, 8, 4), (7, 6, 0, 8, 3, 5)) * -1.0
    t3new_aaaaaa += einsum(x372, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x372, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x372, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x372, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x372, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x372, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x372, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x372, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x372, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    del x372
    x373 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[0], navir[1]), dtype=types[float])
    x373 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), v.aabb.oVOV, (0, 4, 5, 6), (1, 5, 2, 3, 4, 6)) * -1.0
    x374 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x374 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x373, (6, 1, 7, 8, 5, 4), (6, 0, 2, 7, 8, 3))
    t3new_aaaaaa += einsum(x374, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x374, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x374, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x374, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x374, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x374, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x374, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x374, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x374, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -4.0
    del x374
    x375 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x375 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovOV, (0, 2, 4, 5), (1, 4, 3, 5))
    t3new_babbab += einsum(x375, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (4, 0, 5, 6, 2, 7)) * 12.000000000000123
    x376 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x376 += einsum(x375, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (0, 4, 5, 2, 6, 7))
    t3new_aaaaaa += einsum(x376, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.00000000000004
    t3new_aaaaaa += einsum(x376, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.00000000000004
    t3new_aaaaaa += einsum(x376, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.00000000000004
    t3new_aaaaaa += einsum(x376, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 4.00000000000004
    t3new_aaaaaa += einsum(x376, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -4.00000000000004
    t3new_aaaaaa += einsum(x376, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 4.00000000000004
    t3new_aaaaaa += einsum(x376, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -4.00000000000004
    t3new_aaaaaa += einsum(x376, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 4.00000000000004
    t3new_aaaaaa += einsum(x376, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -4.00000000000004
    del x376
    x377 = np.zeros((naocc[0], naocc[0], naocc[0], naocc[0], navir[0], navir[0]), dtype=types[float])
    x377 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.vOOV, (2, 4, 5, 6), (0, 1, 5, 4, 3, 6)) * -1.0
    x378 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x378 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x377, (6, 7, 1, 2, 8, 5), (7, 6, 0, 8, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x378, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x378, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x378, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x378, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x378, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x378, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x378, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x378, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x378, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -6.0
    del x378
    x379 = np.zeros((naocc[0], naocc[0], naocc[0], naocc[0]), dtype=types[float])
    x379 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.vOvO, (2, 4, 3, 5), (0, 1, 4, 5))
    x380 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x380 += einsum(x379, (0, 1, 2, 3), t3.aaaaaa, (4, 2, 3, 5, 6, 7), (1, 0, 4, 5, 6, 7)) * -1.0
    t3new_aaaaaa += einsum(x380, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 3.0
    t3new_aaaaaa += einsum(x380, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 3.0
    t3new_aaaaaa += einsum(x380, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -3.0
    t3new_aaaaaa += einsum(x380, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -3.0
    t3new_aaaaaa += einsum(x380, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -3.0
    t3new_aaaaaa += einsum(x380, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -3.0
    del x380
    x381 = np.zeros((naocc[0], naocc[0], naocc[0], naocc[0], navir[0], navir[0]), dtype=types[float])
    x381 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.vOOV, (2, 4, 5, 6), (0, 1, 5, 4, 3, 6)) * -1.0
    x382 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x382 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x381, (6, 7, 1, 2, 8, 5), (7, 6, 0, 8, 3, 4)) * -1.0
    t3new_aaaaaa += einsum(x382, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x382, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x382, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x382, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x382, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x382, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -6.0
    del x382
    x383 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x383 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), v.aaaa.oVOV, (0, 4, 5, 6), (1, 5, 2, 3, 6, 4)) * -1.0
    x384 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x384 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x383, (6, 2, 7, 8, 4, 5), (6, 0, 1, 7, 8, 3))
    t3new_aaaaaa += einsum(x384, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -12.0
    t3new_aaaaaa += einsum(x384, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 12.0
    t3new_aaaaaa += einsum(x384, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -12.0
    t3new_aaaaaa += einsum(x384, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -12.0
    t3new_aaaaaa += einsum(x384, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 12.0
    t3new_aaaaaa += einsum(x384, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -12.0
    t3new_aaaaaa += einsum(x384, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 12.0
    t3new_aaaaaa += einsum(x384, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -12.0
    t3new_aaaaaa += einsum(x384, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 12.0
    del x384
    x385 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=types[float])
    x385 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovOV, (0, 2, 4, 5), (1, 4, 3, 5))
    t3new_babbab += einsum(x385, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 4.00000000000004
    x386 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x386 += einsum(x385, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    t3new_aaaaaa += einsum(x386, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 12.000000000000123
    t3new_aaaaaa += einsum(x386, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -12.000000000000123
    t3new_aaaaaa += einsum(x386, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 12.000000000000123
    t3new_aaaaaa += einsum(x386, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 12.000000000000123
    t3new_aaaaaa += einsum(x386, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -12.000000000000123
    t3new_aaaaaa += einsum(x386, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 12.000000000000123
    t3new_aaaaaa += einsum(x386, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -12.000000000000123
    t3new_aaaaaa += einsum(x386, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 12.000000000000123
    t3new_aaaaaa += einsum(x386, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -12.000000000000123
    del x386
    x387 = np.zeros((naocc[0], naocc[0]), dtype=types[float])
    x387 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvO, (0, 3, 2, 4), (1, 4)) * -1.0
    x388 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x388 += einsum(x387, (0, 1), t3.aaaaaa, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    t3new_aaaaaa += einsum(x388, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x388, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x388, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x388, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x388, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x388, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    del x388
    x389 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=types[float])
    x389 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.oVvO, (0, 4, 2, 5), (1, 5, 3, 4))
    t3new_babbab += einsum(x389, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -4.00000000000004
    x390 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x390 += einsum(x389, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    t3new_aaaaaa += einsum(x390, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -12.000000000000123
    t3new_aaaaaa += einsum(x390, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 12.000000000000123
    t3new_aaaaaa += einsum(x390, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -12.000000000000123
    t3new_aaaaaa += einsum(x390, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -12.000000000000123
    t3new_aaaaaa += einsum(x390, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 12.000000000000123
    t3new_aaaaaa += einsum(x390, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -12.000000000000123
    t3new_aaaaaa += einsum(x390, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 12.000000000000123
    t3new_aaaaaa += einsum(x390, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -12.000000000000123
    t3new_aaaaaa += einsum(x390, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 12.000000000000123
    del x390
    x391 = np.zeros((navir[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x391 += einsum(t2.aaaa[np.ix_(soa,soa,sVa,sVa)], (0, 1, 2, 3), v.aaaa.oVoV, (0, 4, 1, 5), (2, 3, 4, 5))
    t3new_abaaba += einsum(x391, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 2, 7, 3), (4, 5, 6, 0, 7, 1)) * 2.0
    x392 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x392 += einsum(x391, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 6, 7, 2, 3), (4, 5, 6, 1, 0, 7)) * -1.0
    del x391
    t3new_aaaaaa += einsum(x392, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -6.0
    t3new_aaaaaa += einsum(x392, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x392, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x392
    x393 = np.zeros((navir[0], navir[0]), dtype=types[float])
    x393 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoV, (1, 2, 0, 4), (3, 4)) * -1.0
    t3new_babbab += einsum(x393, (0, 1), t3.babbab, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -2.0
    x394 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x394 += einsum(x393, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    t3new_aaaaaa += einsum(x394, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x394, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x394, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x394
    x395 = np.zeros((navir[0], navir[0]), dtype=types[float])
    x395 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoV, (1, 2, 0, 4), (3, 4)) * -1.0
    t3new_babbab += einsum(x395, (0, 1), t3.babbab, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -2.0
    x396 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x396 += einsum(x395, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    t3new_aaaaaa += einsum(x396, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x396, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x396, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x396
    x397 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[0], nvir[0]), dtype=types[float])
    x397 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (2, 3, 4, 5), v.aaaa.ovoO, (2, 6, 0, 7), (3, 7, 1, 4, 5, 6)) * -1.0
    x398 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x398 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x397, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x397
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x398, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x398
    x399 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[0], nvir[0]), dtype=types[float])
    x399 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (2, 3, 4, 5), v.aaaa.ovoO, (0, 6, 2, 7), (3, 7, 1, 4, 5, 6)) * -1.0
    x400 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x400 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x399, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x399
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x400, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    del x400
    x401 = np.zeros((naocc[0], navir[0], navir[0], nvir[0]), dtype=types[float])
    x401 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t1.aa[np.ix_(soa,sVa)], (2, 3), v.aaaa.ovoO, (2, 4, 0, 5), (5, 1, 3, 4))
    x402 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x402 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x401, (4, 5, 6, 2), (0, 1, 4, 5, 6, 3)) * -1.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x402, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x402
    x403 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[0], nvir[0]), dtype=types[float])
    x403 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (2, 3, 4, 5), v.aaaa.ovvV, (2, 1, 6, 7), (0, 3, 4, 5, 7, 6)) * -1.0
    x404 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x404 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x403, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x403
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x404, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    del x404
    x405 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[0], nvir[0]), dtype=types[float])
    x405 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (2, 3, 4, 5), v.aaaa.ovvV, (0, 6, 4, 7), (2, 3, 1, 5, 7, 6)) * -1.0
    x406 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x406 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x405, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x405
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x406, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    del x406
    x407 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[0], nvir[0]), dtype=types[float])
    x407 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (2, 3, 4, 5), v.aaaa.ovvV, (0, 4, 6, 7), (2, 3, 1, 5, 7, 6)) * -1.0
    x408 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x408 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x407, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x407
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x408, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    del x408
    x409 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x409 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.ovOV, (2, 1, 3, 4), (0, 3, 4, 2))
    x410 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x410 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x409, (2, 3, 4, 0), (2, 3, 1, 4))
    t3new_babbab += einsum(x410, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (4, 0, 5, 6, 2, 7)) * -6.0
    x411 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x411 += einsum(x410, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (0, 4, 5, 2, 6, 7))
    t3new_aaaaaa += einsum(x411, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x411, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x411, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x411, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x411, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x411, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x411, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x411, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x411, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    del x411
    x412 = np.zeros((naocc[0], naocc[0], naocc[0], nvir[0]), dtype=types[float])
    x412 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.vOvO, (2, 3, 1, 4), (0, 3, 4, 2))
    x413 = np.zeros((naocc[0], naocc[0], naocc[0], naocc[0]), dtype=types[float])
    x413 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x412, (2, 3, 4, 1), (2, 0, 4, 3))
    del x412
    x414 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x414 += einsum(x413, (0, 1, 2, 3), t3.aaaaaa, (4, 2, 3, 5, 6, 7), (1, 0, 4, 5, 6, 7)) * -1.0
    t3new_aaaaaa += einsum(x414, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 3.0
    t3new_aaaaaa += einsum(x414, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 3.0
    t3new_aaaaaa += einsum(x414, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -3.0
    t3new_aaaaaa += einsum(x414, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -3.0
    t3new_aaaaaa += einsum(x414, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -3.0
    t3new_aaaaaa += einsum(x414, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -3.0
    del x414
    x415 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x415 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.ovOV, (2, 1, 3, 4), (0, 3, 4, 2))
    x416 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=types[float])
    x416 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x415, (2, 3, 4, 0), (2, 3, 1, 4))
    del x415
    t3new_babbab += einsum(x416, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -2.0
    x417 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x417 += einsum(x416, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    t3new_aaaaaa += einsum(x417, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x417, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x417, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x417, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x417, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x417, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -6.0
    t3new_aaaaaa += einsum(x417, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x417, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x417, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 6.0
    del x417
    x418 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x418 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.oVvO, (2, 3, 1, 4), (0, 4, 3, 2))
    x419 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=types[float])
    x419 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x418, (2, 3, 4, 0), (2, 3, 1, 4))
    del x418
    t3new_babbab += einsum(x419, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 2.0
    x420 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x420 += einsum(x419, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    t3new_aaaaaa += einsum(x420, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x420, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x420, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x420, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x420, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x420, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 6.0
    t3new_aaaaaa += einsum(x420, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x420, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x420, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -6.0
    del x420
    x421 = np.zeros((naocc[0], nvir[0]), dtype=types[float])
    x421 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvO, (0, 2, 1, 3), (3, 2))
    x422 = np.zeros((naocc[0], naocc[0]), dtype=types[float])
    x422 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x421, (2, 1), (0, 2))
    del x421
    t3new_babbab += einsum(x422, (0, 1), t3.babbab, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * 2.0
    x423 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x423 += einsum(x422, (0, 1), t3.aaaaaa, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    t3new_aaaaaa += einsum(x423, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x423, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -6.0
    t3new_aaaaaa += einsum(x423, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 6.0
    del x423
    x424 = np.zeros((naocc[0], nvir[0]), dtype=types[float])
    x424 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvO, (0, 1, 2, 3), (3, 2))
    x425 = np.zeros((naocc[0], naocc[0]), dtype=types[float])
    x425 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x424, (2, 1), (0, 2))
    del x424
    t3new_babbab += einsum(x425, (0, 1), t3.babbab, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x426 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x426 += einsum(x425, (0, 1), t3.aaaaaa, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    t3new_aaaaaa += einsum(x426, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x426, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x426, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    del x426
    x427 = np.zeros((navir[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x427 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oVoV, (2, 3, 0, 4), (1, 3, 4, 2))
    x428 = np.zeros((navir[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x428 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x427, (2, 3, 4, 0), (2, 1, 4, 3))
    del x427
    t3new_abaaba += einsum(x428, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 3, 7, 2), (4, 5, 6, 0, 7, 1)) * -2.0
    x429 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x429 += einsum(x428, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 6, 7, 2, 3), (4, 5, 6, 1, 0, 7)) * -1.0
    del x428
    t3new_aaaaaa += einsum(x429, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -6.0
    t3new_aaaaaa += einsum(x429, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x429, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x429
    x430 = np.zeros((navir[0], nocc[0]), dtype=types[float])
    x430 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovoV, (0, 1, 2, 3), (3, 2))
    x431 = np.zeros((navir[0], navir[0]), dtype=types[float])
    x431 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x430, (2, 0), (1, 2))
    del x430
    t3new_babbab += einsum(x431, (0, 1), t3.babbab, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -2.0
    x432 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x432 += einsum(x431, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    t3new_aaaaaa += einsum(x432, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x432, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x432, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x432
    x433 = np.zeros((navir[0], nocc[0]), dtype=types[float])
    x433 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovoV, (2, 1, 0, 3), (3, 2))
    x434 = np.zeros((navir[0], navir[0]), dtype=types[float])
    x434 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x433, (2, 0), (1, 2))
    del x433
    t3new_babbab += einsum(x434, (0, 1), t3.babbab, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    x435 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x435 += einsum(x434, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    t3new_aaaaaa += einsum(x435, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    t3new_aaaaaa += einsum(x435, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 6.0
    t3new_aaaaaa += einsum(x435, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x435
    x436 = np.zeros((naocc[0], nvir[0]), dtype=types[float])
    x436 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.vOov, (2, 3, 0, 1), (3, 2))
    x437 = np.zeros((naocc[0], naocc[0]), dtype=types[float])
    x437 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x436, (2, 1), (0, 2))
    del x436
    t3new_babbab += einsum(x437, (0, 1), t3.babbab, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x438 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x438 += einsum(x437, (0, 1), t3.aaaaaa, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    t3new_aaaaaa += einsum(x438, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x438, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_aaaaaa += einsum(x438, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    del x438
    x439 = np.zeros((navir[0], nocc[0]), dtype=types[float])
    x439 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.oVov, (2, 3, 0, 1), (3, 2))
    x440 = np.zeros((navir[0], navir[0]), dtype=types[float])
    x440 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x439, (2, 0), (1, 2))
    del x439
    t3new_babbab += einsum(x440, (0, 1), t3.babbab, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -2.0
    x441 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x441 += einsum(x440, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    t3new_aaaaaa += einsum(x441, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_aaaaaa += einsum(x441, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_aaaaaa += einsum(x441, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x441
    x442 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0]), dtype=types[float])
    x442 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 2, 4, 5))
    x443 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x443 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x442, (2, 3, 4, 1), (0, 2, 3, 4))
    x444 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x444 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x443, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x444, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    del x444
    x445 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x445 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x442, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    x446 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x446 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x445, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x445
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x446, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    del x446
    x447 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0]), dtype=types[float])
    x447 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 2), (1, 3, 4, 5))
    x448 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x448 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x447, (2, 3, 4, 1), (0, 2, 3, 4))
    x449 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x449 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x448, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x449, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 4.0
    del x449
    x450 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0]), dtype=types[float])
    x450 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 0, 5), (1, 3, 4, 5))
    x451 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x451 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x450, (2, 3, 4, 1), (0, 2, 3, 4))
    x452 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x452 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x451, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x452, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 4.0
    del x452
    x453 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], nocc[0], nocc[0]), dtype=types[float])
    x453 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (2, 3, 4, 5), v.aaaa.ovov, (6, 1, 7, 4), (0, 2, 3, 5, 6, 7)) * -1.0
    x454 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x454 += einsum(t2.aaaa[np.ix_(soa,soa,sVa,sVa)], (0, 1, 2, 3), x453, (4, 5, 6, 7, 0, 1), (4, 5, 6, 7, 2, 3))
    del x453
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x454, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    del x454
    x455 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x455 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x447, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    x456 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x456 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x455, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x455
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x456, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    del x456
    x457 = np.zeros((naocc[0], naocc[0], nocc[0], nocc[0]), dtype=types[float])
    x457 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x458 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x458 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x457, (4, 5, 0, 6), (4, 5, 1, 2, 3, 6))
    x459 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x459 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x458, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x458
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x459, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    del x459
    x460 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x460 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x450, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    x461 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x461 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x460, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x460
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -4.0
    t3new_aaaaaa += einsum(x461, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 4.0
    del x461
    x462 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x462 += einsum(x11, (0, 1), t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    x463 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x463 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x462, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    t3new_aaaaaa += einsum(x463, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x463, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x463, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x463, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x463, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x463, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x463, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x463, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x463, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    del x463
    x464 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x464 += einsum(x12, (0, 1), t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    x465 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x465 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x464, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    t3new_aaaaaa += einsum(x465, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x465, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x465, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x465, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x465, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -4.0
    t3new_aaaaaa += einsum(x465, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x465, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x465, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x465, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 4.0
    del x465
    x466 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=types[float])
    x466 += einsum(x15, (0, 1), t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    x467 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x467 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x466, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    t3new_aaaaaa += einsum(x467, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x467, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x467, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x467, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_aaaaaa += einsum(x467, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_aaaaaa += einsum(x467, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_aaaaaa += einsum(x467, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_aaaaaa += einsum(x467, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_aaaaaa += einsum(x467, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    del x467
    x468 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], nocc[0], nvir[0]), dtype=types[float])
    x468 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (2, 3, 4, 5), v.aaaa.ovov, (6, 7, 2, 1), (0, 3, 4, 5, 6, 7)) * -1.0
    x469 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x469 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x468, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x468
    x470 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x470 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x469, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x469
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 2.0
    t3new_aaaaaa += einsum(x470, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x470
    x471 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], nocc[0], nvir[0]), dtype=types[float])
    x471 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (2, 3, 4, 5), v.aaaa.ovov, (6, 7, 0, 4), (2, 3, 1, 5, 6, 7)) * -1.0
    x472 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], nocc[0]), dtype=types[float])
    x472 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x471, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x471
    x473 = np.zeros((naocc[0], naocc[0], naocc[0], navir[0], navir[0], navir[0]), dtype=types[float])
    x473 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x472, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x472
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * 2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_aaaaaa += einsum(x473, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    del x473
    x474 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x474 += einsum(f.bb.OO, (0, 1), t3.babbab, (2, 3, 1, 4, 5, 6), (3, 0, 2, 5, 4, 6))
    t3new_babbab += einsum(x474, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x474, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x474
    x475 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x475 += einsum(f.bb.VV, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (3, 2, 4, 6, 0, 5))
    t3new_babbab += einsum(x475, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x475, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x475
    x476 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x476 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), v.bbbb.oOOV, (1, 4, 5, 6), (0, 5, 4, 2, 3, 6))
    t3new_babbab += einsum(x476, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x476, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new_babbab += einsum(x476, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x476, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    del x476
    x477 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x477 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), v.aabb.oOOV, (0, 4, 5, 6), (4, 1, 5, 2, 3, 6)) * -1.0
    t3new_babbab += einsum(x477, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x477, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x477, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x477, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x477
    x478 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x478 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), v.bbbb.vVOV, (3, 4, 5, 6), (0, 1, 5, 2, 6, 4))
    t3new_babbab += einsum(x478, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x478, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x478, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x478, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    del x478
    x479 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x479 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.vVOV, (2, 4, 5, 6), (0, 1, 5, 4, 3, 6)) * -1.0
    t3new_babbab += einsum(x479, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x479, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x479, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x479, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x479
    x480 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x480 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), v.aabb.OVoO, (4, 5, 0, 6), (4, 1, 6, 5, 2, 3)) * -1.0
    t3new_babbab += einsum(x480, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x480, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x480
    x481 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x481 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.OVvV, (4, 5, 2, 6), (4, 0, 1, 5, 3, 6)) * -1.0
    t3new_babbab += einsum(x481, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x481, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x481
    x482 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x482 += einsum(v.aabb.OVOV, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 7, 1), (4, 5, 2, 6, 7, 3))
    t3new_babbab += einsum(x482, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x482, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x482, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x482, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    del x482
    x483 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x483 += einsum(v.aabb.OOOO, (0, 1, 2, 3), t3.babbab, (4, 1, 3, 5, 6, 7), (0, 4, 2, 6, 5, 7))
    t3new_babbab += einsum(x483, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x483, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x483
    x484 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x484 += einsum(v.bbbb.OOVV, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (5, 4, 0, 7, 6, 2))
    t3new_babbab += einsum(x484, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x484, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x484, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x484, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    del x484
    x485 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x485 += einsum(v.aabb.OOVV, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 7, 3), (0, 4, 5, 7, 6, 2))
    t3new_babbab += einsum(x485, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x485, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    del x485
    x486 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x486 += einsum(v.bbbb.OVOV, (0, 1, 2, 3), t3.babbab, (4, 5, 2, 6, 7, 3), (5, 4, 0, 7, 6, 1))
    t3new_babbab += einsum(x486, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x486, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x486, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x486, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    del x486
    x487 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x487 += einsum(v.aabb.VVOO, (0, 1, 2, 3), t3.babbab, (4, 5, 3, 6, 1, 7), (5, 4, 2, 0, 6, 7))
    t3new_babbab += einsum(x487, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x487, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x487
    x488 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x488 += einsum(v.aabb.VVVV, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 7, 1, 3), (5, 4, 6, 0, 7, 2))
    t3new_babbab += einsum(x488, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x488, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    del x488
    x489 = np.zeros((naocc[1], naocc[1]), dtype=types[float])
    x489 += einsum(f.bb.vO, (0, 1), t1.bb[np.ix_(sOb,svb)], (2, 0), (1, 2))
    t3new_abaaba += einsum(x489, (0, 1), t3.abaaba, (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -2.0
    x490 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x490 += einsum(x489, (0, 1), t3.babbab, (2, 3, 0, 4, 5, 6), (3, 1, 2, 5, 4, 6))
    t3new_babbab += einsum(x490, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x490, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x490
    x491 = np.zeros((navir[1], navir[1]), dtype=types[float])
    x491 += einsum(f.bb.oV, (0, 1), t1.bb[np.ix_(sob,sVb)], (0, 2), (1, 2))
    t3new_abaaba += einsum(x491, (0, 1), t3.abaaba, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -2.0
    x492 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x492 += einsum(x491, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 0), (3, 2, 4, 6, 1, 5))
    t3new_babbab += einsum(x492, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x492, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x492
    x493 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x493 += einsum(f.aa.ov, (0, 1), t2.abab[np.ix_(sOa,sOb,sva,sVb)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    x494 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x494 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x493, (4, 5, 6, 0), (4, 5, 1, 2, 6, 3)) * -1.0
    t3new_babbab += einsum(x494, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x494, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x494, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x494, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x494
    x495 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x495 += einsum(f.bb.ov, (0, 1), t2.abab[np.ix_(sOa,sOb,sVa,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    x496 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x496 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x495, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x496, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x496, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x496
    x497 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x497 += einsum(f.bb.ov, (0, 1), t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    x498 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x498 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x497, (4, 5, 6, 1), (0, 5, 4, 2, 6, 3)) * -1.0
    t3new_babbab += einsum(x498, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x498, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x498
    x499 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x499 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x409, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    t3new_babbab += einsum(x499, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x499, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x499, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x499, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x499
    x500 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], navir[1], nocc[0]), dtype=types[float])
    x500 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovOV, (4, 2, 5, 6), (0, 1, 5, 3, 6, 4)) * -1.0
    x501 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x501 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x500, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x500
    t3new_babbab += einsum(x501, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x501, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x501, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x501, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x501
    x502 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], navir[1], nocc[0]), dtype=types[float])
    x502 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), v.aabb.oOoO, (4, 5, 0, 6), (5, 1, 6, 2, 3, 4)) * -1.0
    x503 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x503 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x502, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x502
    t3new_babbab += einsum(x503, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x503, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x503
    x504 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], navir[1], nocc[0]), dtype=types[float])
    x504 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.oOvV, (4, 5, 2, 6), (5, 0, 1, 3, 6, 4)) * -1.0
    x505 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x505 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x504, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x504
    t3new_babbab += einsum(x505, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x505, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x505
    x506 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x506 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vVoO, (1, 2, 3, 4), (0, 4, 2, 3))
    x507 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x507 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x506, (4, 5, 6, 0), (4, 1, 5, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x507, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x507, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x507
    x508 = np.zeros((naocc[1], naocc[1], navir[0], navir[1], navir[1], nvir[0]), dtype=types[float])
    x508 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.vVvV, (4, 5, 2, 6), (0, 1, 5, 3, 6, 4)) * -1.0
    x509 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x509 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x508, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x508
    t3new_babbab += einsum(x509, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x509, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x509
    x510 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x510 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.oOvV, (2, 3, 1, 4), (0, 3, 4, 2))
    x511 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x511 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x510, (4, 5, 6, 1), (0, 4, 5, 2, 3, 6))
    t3new_babbab += einsum(x511, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x511, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x511, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new_babbab += einsum(x511, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    del x511
    x512 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x512 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.oOvV, (2, 3, 1, 4), (3, 0, 4, 2))
    x513 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x513 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x512, (4, 5, 6, 0), (4, 5, 1, 2, 3, 6)) * -1.0
    t3new_babbab += einsum(x513, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x513, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x513, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x513, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    del x513
    x514 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x514 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), v.bbbb.oOoO, (4, 5, 1, 6), (0, 5, 6, 2, 3, 4))
    x515 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x515 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x514, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x514
    t3new_babbab += einsum(x515, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x515, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x515, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x515, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x515
    x516 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x516 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), v.aabb.oOoO, (0, 4, 5, 6), (4, 1, 6, 2, 3, 5)) * -1.0
    x517 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x517 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x516, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x516
    t3new_babbab += einsum(x517, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x517, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x517, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x517, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x517
    x518 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x518 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), v.bbbb.oOvV, (4, 5, 3, 6), (0, 1, 5, 2, 6, 4))
    x519 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x519 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x518, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x518
    t3new_babbab += einsum(x519, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x519, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x519, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new_babbab += einsum(x519, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    del x519
    x520 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x520 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.ovOV, (2, 1, 3, 4), (0, 3, 4, 2))
    x521 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x521 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x520, (4, 5, 6, 1), (0, 4, 5, 2, 3, 6))
    t3new_babbab += einsum(x521, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x521, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x521, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x521, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    del x521
    x522 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x522 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovOV, (4, 3, 5, 6), (0, 1, 5, 2, 6, 4))
    x523 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x523 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x522, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x522
    t3new_babbab += einsum(x523, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x523, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x523, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x523, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    del x523
    x524 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], nvir[1]), dtype=types[float])
    x524 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), v.bbbb.vVvV, (4, 5, 3, 6), (0, 1, 2, 5, 6, 4))
    x525 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x525 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x524, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x524
    t3new_babbab += einsum(x525, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x525, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x525, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x525, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x525
    x526 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], nvir[1]), dtype=types[float])
    x526 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.vVvV, (2, 4, 5, 6), (0, 1, 4, 3, 6, 5)) * -1.0
    x527 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x527 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x526, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x526
    t3new_babbab += einsum(x527, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x527, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x527, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new_babbab += einsum(x527, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    del x527
    x528 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x528 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.vVoO, (2, 4, 5, 6), (0, 1, 6, 4, 3, 5)) * -1.0
    x529 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x529 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x528, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x528
    t3new_babbab += einsum(x529, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x529, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x529, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x529, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x529
    x530 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x530 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.OVov, (2, 3, 4, 1), (2, 0, 3, 4))
    x531 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x531 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x530, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x531, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x531, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x531
    x532 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x532 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 6, 2), (4, 0, 1, 5, 3, 6)) * -1.0
    x533 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x533 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x532, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x532
    t3new_babbab += einsum(x533, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x533, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x533
    x534 = np.zeros((naocc[0], naocc[0], naocc[1], naocc[1]), dtype=types[float])
    x534 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vOOO, (1, 2, 3, 4), (0, 2, 3, 4))
    x535 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x535 += einsum(x534, (0, 1, 2, 3), t3.babbab, (4, 1, 3, 5, 6, 7), (0, 4, 2, 6, 5, 7))
    t3new_babbab += einsum(x535, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x535, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x535
    x536 = np.zeros((naocc[0], naocc[0], navir[1], navir[1]), dtype=types[float])
    x536 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vOVV, (1, 2, 3, 4), (0, 2, 3, 4))
    x537 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x537 += einsum(x536, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 7, 3), (0, 4, 5, 7, 6, 2))
    t3new_babbab += einsum(x537, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x537, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    del x537
    x538 = np.zeros((naocc[1], naocc[1], navir[0], navir[0]), dtype=types[float])
    x538 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.oVOO, (0, 2, 3, 4), (3, 4, 1, 2))
    x539 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x539 += einsum(x538, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 3, 7), (5, 4, 0, 2, 6, 7))
    t3new_babbab += einsum(x539, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x539, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x539
    x540 = np.zeros((naocc[1], naocc[1]), dtype=types[float])
    x540 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovOO, (0, 1, 2, 3), (2, 3))
    t3new_abaaba += einsum(x540, (0, 1), t3.abaaba, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x541 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x541 += einsum(x540, (0, 1), t3.babbab, (2, 3, 1, 4, 5, 6), (3, 2, 0, 5, 4, 6))
    t3new_babbab += einsum(x541, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x541, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x541
    x542 = np.zeros((navir[0], navir[0], navir[1], navir[1]), dtype=types[float])
    x542 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.oVVV, (0, 2, 3, 4), (1, 2, 3, 4))
    x543 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x543 += einsum(x542, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 7, 1, 3), (5, 4, 6, 0, 7, 2))
    t3new_babbab += einsum(x543, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x543, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    del x543
    x544 = np.zeros((navir[1], navir[1]), dtype=types[float])
    x544 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovVV, (0, 1, 2, 3), (2, 3))
    t3new_abaaba += einsum(x544, (0, 1), t3.abaaba, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    x545 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x545 += einsum(x544, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (3, 2, 4, 6, 5, 0))
    t3new_babbab += einsum(x545, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x545, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    del x545
    x546 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x546 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.OVvV, (2, 3, 1, 4), (2, 0, 3, 4))
    t3new_abaaba += einsum(x546, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 7, 2), (4, 1, 5, 6, 3, 7)) * 6.0
    x547 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x547 += einsum(x546, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 7, 2), (4, 1, 5, 6, 7, 3))
    t3new_babbab += einsum(x547, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x547, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x547, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x547, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    del x547
    x548 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x548 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.OVoO, (2, 3, 0, 4), (2, 4, 3, 1))
    t3new_abaaba += einsum(x548, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 7, 2), (4, 1, 5, 6, 3, 7)) * -6.0
    x549 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x549 += einsum(x548, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 7, 2), (4, 5, 1, 6, 3, 7))
    t3new_babbab += einsum(x549, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x549, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x549, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x549, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x549
    x550 = np.zeros((naocc[0], naocc[0], naocc[1], naocc[1]), dtype=types[float])
    x550 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.OOvO, (2, 3, 1, 4), (2, 3, 0, 4))
    x551 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x551 += einsum(x550, (0, 1, 2, 3), t3.babbab, (4, 1, 3, 5, 6, 7), (0, 2, 4, 6, 5, 7))
    t3new_babbab += einsum(x551, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x551, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x551
    x552 = np.zeros((naocc[1], naocc[1], naocc[1], naocc[1]), dtype=types[float])
    x552 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.vOOO, (1, 2, 3, 4), (0, 3, 4, 2))
    x553 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x553 += einsum(x552, (0, 1, 2, 3), t3.babbab, (3, 4, 2, 5, 6, 7), (4, 0, 1, 6, 5, 7)) * -1.0
    t3new_babbab += einsum(x553, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x553, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x553
    x554 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=types[float])
    x554 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oOOV, (0, 2, 3, 4), (3, 2, 1, 4))
    t3new_abaaba += einsum(x554, (0, 1, 2, 3), t3.abaaba, (4, 0, 5, 6, 3, 7), (4, 1, 5, 6, 2, 7)) * -2.0
    x555 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x555 += einsum(x554, (0, 1, 2, 3), t3.babbab, (4, 5, 0, 6, 7, 3), (5, 4, 1, 7, 2, 6))
    t3new_babbab += einsum(x555, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x555, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x555, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x555, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x555
    x556 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=types[float])
    x556 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oVOO, (0, 2, 3, 4), (3, 4, 1, 2))
    t3new_abaaba += einsum(x556, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 2.0
    x557 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x557 += einsum(x556, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (5, 4, 0, 7, 2, 6))
    t3new_babbab += einsum(x557, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x557, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x557, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x557, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x557
    x558 = np.zeros((naocc[0], naocc[0], navir[1], navir[1]), dtype=types[float])
    x558 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.OOoV, (2, 3, 0, 4), (2, 3, 1, 4))
    x559 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x559 += einsum(x558, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 7, 3), (0, 4, 5, 7, 2, 6))
    t3new_babbab += einsum(x559, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x559, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x559
    x560 = np.zeros((naocc[1], naocc[1]), dtype=types[float])
    x560 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oOvO, (0, 2, 1, 3), (2, 3))
    t3new_abaaba += einsum(x560, (0, 1), t3.abaaba, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * 2.0
    x561 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x561 += einsum(x560, (0, 1), t3.babbab, (2, 3, 1, 4, 5, 6), (3, 2, 0, 5, 4, 6))
    t3new_babbab += einsum(x561, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x561, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x561
    x562 = np.zeros((naocc[1], naocc[1]), dtype=types[float])
    x562 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovOO, (0, 1, 2, 3), (2, 3))
    t3new_abaaba += einsum(x562, (0, 1), t3.abaaba, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x563 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x563 += einsum(x562, (0, 1), t3.babbab, (2, 3, 1, 4, 5, 6), (3, 2, 0, 5, 4, 6))
    t3new_babbab += einsum(x563, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x563, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x563
    x564 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=types[float])
    x564 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.vOVV, (1, 2, 3, 4), (0, 2, 3, 4))
    t3new_abaaba += einsum(x564, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -2.0
    x565 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x565 += einsum(x564, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (5, 0, 4, 7, 6, 2))
    t3new_babbab += einsum(x565, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x565, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x565, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x565, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    del x565
    x566 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=types[float])
    x566 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.vVOV, (1, 2, 3, 4), (0, 3, 4, 2))
    t3new_abaaba += einsum(x566, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 2, 7), (4, 0, 5, 6, 3, 7)) * 2.0
    x567 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x567 += einsum(x566, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 2), (5, 0, 4, 7, 6, 3))
    t3new_babbab += einsum(x567, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x567, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x567, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x567, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    del x567
    x568 = np.zeros((navir[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x568 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oVVV, (0, 2, 3, 4), (1, 3, 4, 2))
    x569 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x569 += einsum(x568, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 3, 7, 2), (5, 4, 6, 7, 0, 1)) * -1.0
    t3new_babbab += einsum(x569, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x569, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x569
    x570 = np.zeros((navir[1], navir[1]), dtype=types[float])
    x570 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovVV, (0, 1, 2, 3), (2, 3))
    t3new_abaaba += einsum(x570, (0, 1), t3.abaaba, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    x571 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x571 += einsum(x570, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (3, 2, 4, 6, 5, 0))
    t3new_babbab += einsum(x571, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x571, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    del x571
    x572 = np.zeros((navir[1], navir[1]), dtype=types[float])
    x572 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oVvV, (0, 2, 1, 3), (2, 3))
    t3new_abaaba += einsum(x572, (0, 1), t3.abaaba, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -2.0
    x573 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x573 += einsum(x572, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 0), (3, 2, 4, 6, 5, 1))
    t3new_babbab += einsum(x573, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x573, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    del x573
    x574 = np.zeros((naocc[1], naocc[1], navir[0], navir[0]), dtype=types[float])
    x574 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.VVvO, (2, 3, 1, 4), (0, 4, 2, 3))
    x575 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x575 += einsum(x574, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 3, 7), (5, 0, 4, 2, 6, 7))
    t3new_babbab += einsum(x575, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x575, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x575
    x576 = np.zeros((navir[0], navir[0], navir[1], navir[1]), dtype=types[float])
    x576 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.VVoV, (2, 3, 0, 4), (2, 3, 1, 4))
    x577 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x577 += einsum(x576, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 7, 1, 3), (5, 4, 6, 0, 2, 7))
    t3new_babbab += einsum(x577, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x577, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x577
    x578 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x578 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovoO, (4, 2, 0, 5), (5, 1, 3, 4))
    x579 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x579 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x578, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    t3new_babbab += einsum(x579, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x579, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x579, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x579, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    del x579
    x580 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x580 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovoO, (0, 2, 4, 5), (5, 1, 3, 4))
    x581 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x581 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x580, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    t3new_babbab += einsum(x581, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new_babbab += einsum(x581, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x581, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x581, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    del x581
    x582 = np.zeros((naocc[1], navir[0], navir[1], nvir[0]), dtype=types[float])
    x582 += einsum(t2.abab[np.ix_(soa,sob,sVa,sVb)], (0, 1, 2, 3), v.aabb.ovoO, (0, 4, 1, 5), (5, 2, 3, 4))
    x583 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x583 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x582, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6)) * -1.0
    t3new_babbab += einsum(x583, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x583, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x583, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x583, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x583
    x584 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x584 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoO, (0, 2, 4, 5), (1, 5, 3, 4))
    x585 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x585 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x584, (4, 5, 6, 1), (0, 4, 5, 2, 3, 6))
    t3new_babbab += einsum(x585, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x585, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x585, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x585, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x585
    x586 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x586 += einsum(t2.abab[np.ix_(sOa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoO, (4, 2, 1, 5), (0, 5, 3, 4)) * -1.0
    x587 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x587 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x586, (4, 5, 6, 0), (4, 1, 5, 2, 6, 3)) * -1.0
    t3new_babbab += einsum(x587, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x587, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x587, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x587, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x587
    x588 = np.zeros((naocc[1], navir[0], navir[1], nvir[0]), dtype=types[float])
    x588 += einsum(t2.abab[np.ix_(soa,sOb,sVa,svb)], (0, 1, 2, 3), v.aabb.ovvV, (0, 4, 3, 5), (1, 2, 5, 4)) * -1.0
    x589 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x589 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x588, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6)) * -1.0
    t3new_babbab += einsum(x589, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x589, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x589, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x589, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    del x589
    x590 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x590 += einsum(t2.abab[np.ix_(sOa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovvV, (4, 2, 3, 5), (0, 1, 5, 4))
    x591 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x591 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x590, (4, 5, 6, 0), (4, 5, 1, 2, 3, 6)) * -1.0
    t3new_babbab += einsum(x591, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x591, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x591, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x591, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    del x591
    x592 = np.zeros((naocc[1], navir[0], navir[1], nvir[0]), dtype=types[float])
    x592 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovvV, (0, 4, 2, 5), (1, 5, 3, 4))
    x593 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x593 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x592, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6)) * -1.0
    t3new_babbab += einsum(x593, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x593, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x593, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x593, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x593
    x594 = np.zeros((naocc[1], navir[1], navir[1], nvir[1]), dtype=types[float])
    x594 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovvV, (0, 2, 4, 5), (1, 3, 5, 4))
    x595 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x595 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x594, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    t3new_babbab += einsum(x595, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x595, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x595, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x595, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    del x595
    x596 = np.zeros((naocc[1], navir[0], navir[1], nvir[0]), dtype=types[float])
    x596 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovvV, (0, 2, 4, 5), (1, 5, 3, 4))
    x597 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x597 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x596, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6)) * -1.0
    t3new_babbab += einsum(x597, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x597, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x597, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x597, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x597
    x598 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x598 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoO, (0, 2, 4, 5), (1, 5, 3, 4))
    x599 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x599 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x598, (4, 5, 6, 1), (0, 4, 5, 2, 3, 6))
    t3new_babbab += einsum(x599, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x599, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x599, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x599, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    del x599
    x600 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x600 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovoO, (4, 3, 1, 5), (0, 5, 2, 4))
    x601 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x601 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x600, (4, 5, 6, 0), (4, 1, 5, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x601, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x601, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x601
    x602 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x602 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoO, (4, 2, 0, 5), (1, 5, 3, 4))
    x603 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x603 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x602, (4, 5, 6, 1), (0, 4, 5, 2, 3, 6))
    t3new_babbab += einsum(x603, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x603, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x603, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x603, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    del x603
    x604 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x604 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovoO, (1, 3, 4, 5), (0, 5, 2, 4))
    x605 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x605 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x604, (4, 5, 6, 0), (4, 1, 5, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x605, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x605, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x605
    x606 = np.zeros((naocc[1], navir[1], navir[1], nvir[1]), dtype=types[float])
    x606 += einsum(t2.bbbb[np.ix_(sob,sob,sVb,sVb)], (0, 1, 2, 3), v.bbbb.ovoO, (1, 4, 0, 5), (5, 2, 3, 4))
    x607 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x607 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x606, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    t3new_babbab += einsum(x607, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x607, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x607, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x607, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    del x607
    x608 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x608 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.oOov, (4, 5, 0, 2), (5, 1, 3, 4))
    x609 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x609 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x608, (4, 5, 6, 0), (4, 5, 1, 2, 6, 3)) * -1.0
    t3new_babbab += einsum(x609, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x609, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x609, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x609, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x609
    x610 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x610 += einsum(t2.abab[np.ix_(soa,sOb,sVa,svb)], (0, 1, 2, 3), v.aabb.oOov, (0, 4, 5, 3), (4, 1, 2, 5)) * -1.0
    x611 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x611 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x610, (4, 5, 6, 0), (4, 1, 5, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x611, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x611, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x611
    x612 = np.zeros((naocc[0], navir[0], navir[1], nvir[1]), dtype=types[float])
    x612 += einsum(t2.abab[np.ix_(soa,sob,sVa,sVb)], (0, 1, 2, 3), v.aabb.oOov, (0, 4, 1, 5), (4, 2, 3, 5))
    x613 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x613 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x612, (4, 5, 6, 2), (4, 0, 1, 5, 3, 6)) * -1.0
    t3new_babbab += einsum(x613, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x613, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x613
    x614 = np.zeros((naocc[1], navir[1], navir[1], nvir[1]), dtype=types[float])
    x614 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovvV, (0, 4, 2, 5), (1, 3, 5, 4))
    x615 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x615 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x614, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    t3new_babbab += einsum(x615, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x615, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x615, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x615, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x615
    x616 = np.zeros((naocc[1], navir[1], navir[1], nvir[1]), dtype=types[float])
    x616 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovvV, (0, 2, 4, 5), (1, 3, 5, 4))
    x617 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x617 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x616, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    t3new_babbab += einsum(x617, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x617, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x617, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x617, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x617
    x618 = np.zeros((naocc[0], navir[0], navir[1], nvir[1]), dtype=types[float])
    x618 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (1, 3, 4, 5), (0, 2, 5, 4))
    x619 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x619 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x618, (4, 5, 6, 2), (4, 0, 1, 5, 3, 6)) * -1.0
    t3new_babbab += einsum(x619, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x619, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x619
    x620 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x620 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (4, 3, 2, 5), (0, 1, 5, 4))
    x621 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x621 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x620, (4, 5, 6, 1), (0, 5, 4, 2, 3, 6)) * -1.0
    t3new_babbab += einsum(x621, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x621, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x621, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x621, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    del x621
    x622 = np.zeros((naocc[0], navir[0], navir[1], nvir[1]), dtype=types[float])
    x622 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (1, 4, 3, 5), (0, 2, 5, 4))
    x623 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x623 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x622, (4, 5, 6, 2), (4, 0, 1, 5, 3, 6)) * -1.0
    t3new_babbab += einsum(x623, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x623, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x623
    x624 = np.zeros((naocc[1], navir[0], navir[1], nvir[0]), dtype=types[float])
    x624 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.vVov, (4, 5, 0, 2), (1, 5, 3, 4))
    x625 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x625 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x624, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6)) * -1.0
    t3new_babbab += einsum(x625, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x625, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x625, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x625, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    del x625
    x626 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x626 += einsum(t2.abab[np.ix_(sOa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.vVov, (2, 4, 5, 3), (0, 1, 4, 5))
    x627 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x627 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x626, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x627, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x627, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x627
    x628 = np.zeros((naocc[0], navir[0], navir[1], nvir[1]), dtype=types[float])
    x628 += einsum(t2.abab[np.ix_(sOa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.vVov, (2, 4, 1, 5), (0, 4, 3, 5)) * -1.0
    x629 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x629 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x628, (4, 5, 6, 2), (4, 0, 1, 5, 3, 6)) * -1.0
    t3new_babbab += einsum(x629, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x629, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x629
    x630 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x630 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovoO, (0, 2, 4, 5), (1, 5, 3, 4))
    x631 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x631 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x630, (4, 5, 6, 0), (4, 1, 5, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x631, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 4.0
    t3new_babbab += einsum(x631, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -4.0
    del x631
    x632 = np.zeros((naocc[0], navir[0], navir[1], nvir[1]), dtype=types[float])
    x632 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovvV, (0, 2, 4, 5), (1, 3, 5, 4))
    x633 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x633 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x632, (4, 5, 6, 2), (4, 0, 1, 5, 3, 6)) * -1.0
    t3new_babbab += einsum(x633, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 4.0
    t3new_babbab += einsum(x633, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -4.0
    del x633
    x634 = np.zeros((naocc[0], naocc[0], naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x634 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.vOOV, (2, 4, 5, 6), (0, 5, 4, 1, 6, 3)) * -1.0
    x635 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x635 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x634, (6, 0, 2, 7, 5, 8), (6, 7, 1, 3, 8, 4))
    t3new_babbab += einsum(x635, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x635, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x635, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x635, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x635
    x636 = np.zeros((naocc[0], naocc[0], naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x636 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.vOOV, (2, 4, 5, 6), (0, 5, 4, 1, 6, 3)) * -1.0
    x637 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x637 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x636, (6, 0, 2, 7, 5, 8), (6, 7, 1, 3, 8, 4))
    t3new_babbab += einsum(x637, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x637, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x637, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x637, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x637
    x638 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[0], navir[1]), dtype=types[float])
    x638 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), v.aaaa.oVOV, (0, 4, 5, 6), (5, 1, 2, 6, 4, 3)) * -1.0
    x639 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x639 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x638, (2, 6, 7, 3, 5, 8), (0, 6, 1, 7, 8, 4))
    t3new_babbab += einsum(x639, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x639, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x639, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x639, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x639
    x640 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x640 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovOV, (0, 2, 4, 5), (4, 1, 5, 3))
    t3new_abaaba += einsum(x640, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 7, 2), (4, 1, 5, 6, 3, 7)) * 6.0000000000000595
    x641 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x641 += einsum(x640, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 7, 2), (4, 1, 5, 6, 3, 7))
    t3new_babbab += einsum(x641, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.00000000000002
    t3new_babbab += einsum(x641, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.00000000000002
    t3new_babbab += einsum(x641, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.00000000000002
    t3new_babbab += einsum(x641, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.00000000000002
    del x641
    x642 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x642 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.oVvO, (0, 4, 2, 5), (5, 1, 4, 3))
    t3new_abaaba += einsum(x642, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 7, 2), (4, 1, 5, 6, 3, 7)) * -6.0000000000000595
    x643 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x643 += einsum(x642, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 7, 2), (4, 1, 5, 6, 3, 7))
    t3new_babbab += einsum(x643, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.00000000000002
    t3new_babbab += einsum(x643, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.00000000000002
    t3new_babbab += einsum(x643, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.00000000000002
    t3new_babbab += einsum(x643, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.00000000000002
    del x643
    x644 = np.zeros((naocc[0], naocc[0], naocc[1], naocc[1], navir[1], navir[1]), dtype=types[float])
    x644 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.vOOV, (2, 4, 5, 6), (0, 4, 1, 5, 3, 6)) * -1.0
    x645 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x645 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x644, (6, 1, 7, 2, 8, 5), (6, 7, 0, 4, 8, 3))
    t3new_babbab += einsum(x645, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x645, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x645, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x645, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x645
    x646 = np.zeros((naocc[0], naocc[0], naocc[1], naocc[1]), dtype=types[float])
    x646 += einsum(t2.abab[np.ix_(sOa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.vOvO, (2, 4, 3, 5), (0, 4, 1, 5))
    x647 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x647 += einsum(x646, (0, 1, 2, 3), t3.babbab, (4, 1, 3, 5, 6, 7), (0, 2, 4, 6, 5, 7))
    t3new_babbab += einsum(x647, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x647, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x647
    x648 = np.zeros((naocc[0], naocc[0], naocc[1], naocc[1], navir[0], navir[0]), dtype=types[float])
    x648 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), v.aabb.OVvO, (4, 5, 3, 6), (0, 4, 1, 6, 2, 5))
    x649 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x649 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x648, (6, 1, 7, 2, 8, 4), (6, 7, 0, 8, 3, 5))
    t3new_babbab += einsum(x649, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x649, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x649
    x650 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[1], navir[1]), dtype=types[float])
    x650 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), v.aabb.OVoV, (4, 5, 1, 6), (0, 4, 2, 5, 3, 6))
    x651 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x651 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x650, (6, 1, 7, 4, 8, 5), (6, 0, 2, 7, 8, 3))
    t3new_babbab += einsum(x651, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x651, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x651
    x652 = np.zeros((naocc[0], naocc[0], navir[1], navir[1]), dtype=types[float])
    x652 += einsum(t2.abab[np.ix_(sOa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.vOoV, (2, 4, 1, 5), (0, 4, 3, 5)) * -1.0
    x653 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x653 += einsum(x652, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 7, 3), (0, 4, 5, 7, 2, 6))
    t3new_babbab += einsum(x653, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.00000000000002
    t3new_babbab += einsum(x653, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.00000000000002
    del x653
    x654 = np.zeros((naocc[1], naocc[1], navir[0], navir[0], navir[1], navir[1]), dtype=types[float])
    x654 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), v.aabb.oVOV, (0, 4, 5, 6), (1, 5, 2, 4, 3, 6)) * -1.0
    x655 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x655 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x654, (6, 2, 7, 4, 8, 5), (1, 6, 0, 7, 8, 3))
    t3new_babbab += einsum(x655, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x655, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x655, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x655, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x655
    x656 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=types[float])
    x656 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovOV, (0, 2, 4, 5), (1, 4, 3, 5))
    t3new_abaaba += einsum(x656, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 2.00000000000002
    x657 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x657 += einsum(x656, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (5, 0, 4, 7, 2, 6))
    t3new_babbab += einsum(x657, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.00000000000002
    t3new_babbab += einsum(x657, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.00000000000002
    t3new_babbab += einsum(x657, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.00000000000002
    t3new_babbab += einsum(x657, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.00000000000002
    del x657
    x658 = np.zeros((naocc[1], naocc[1]), dtype=types[float])
    x658 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovvO, (0, 2, 3, 4), (1, 4)) * -1.0
    t3new_abaaba += einsum(x658, (0, 1), t3.abaaba, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * 2.0
    x659 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x659 += einsum(x658, (0, 1), t3.babbab, (2, 3, 1, 4, 5, 6), (3, 0, 2, 5, 4, 6))
    t3new_babbab += einsum(x659, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x659, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x659
    x660 = np.zeros((naocc[1], naocc[1], navir[0], navir[0]), dtype=types[float])
    x660 += einsum(t2.abab[np.ix_(soa,sOb,sVa,svb)], (0, 1, 2, 3), v.aabb.oVvO, (0, 4, 3, 5), (1, 5, 2, 4)) * -1.0
    x661 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x661 += einsum(x660, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 3, 7), (5, 0, 4, 2, 6, 7))
    t3new_babbab += einsum(x661, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.00000000000002
    t3new_babbab += einsum(x661, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.00000000000002
    del x661
    x662 = np.zeros((navir[0], navir[0], navir[1], navir[1]), dtype=types[float])
    x662 += einsum(t2.abab[np.ix_(soa,sob,sVa,sVb)], (0, 1, 2, 3), v.aabb.oVoV, (0, 4, 1, 5), (2, 4, 3, 5))
    x663 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x663 += einsum(x662, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 7, 1, 3), (5, 4, 6, 0, 2, 7))
    t3new_babbab += einsum(x663, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x663, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x663
    x664 = np.zeros((navir[1], navir[1]), dtype=types[float])
    x664 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoV, (0, 2, 1, 4), (3, 4)) * -1.0
    t3new_abaaba += einsum(x664, (0, 1), t3.abaaba, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    x665 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x665 += einsum(x664, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (3, 2, 4, 6, 0, 5))
    t3new_babbab += einsum(x665, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x665, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x665
    x666 = np.zeros((naocc[0], naocc[1], naocc[1], naocc[1], navir[0], navir[1]), dtype=types[float])
    x666 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), v.bbbb.vOOV, (3, 4, 5, 6), (0, 1, 5, 4, 2, 6))
    x667 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x667 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x666, (6, 7, 1, 2, 8, 5), (6, 7, 0, 8, 3, 4))
    t3new_babbab += einsum(x667, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 3.0
    t3new_babbab += einsum(x667, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -3.0
    del x667
    x668 = np.zeros((naocc[0], naocc[1], naocc[1], naocc[1], navir[0], navir[1]), dtype=types[float])
    x668 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), v.bbbb.vOOV, (3, 4, 5, 6), (0, 1, 5, 4, 2, 6))
    x669 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x669 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x668, (6, 7, 1, 2, 8, 5), (6, 7, 0, 8, 3, 4))
    t3new_babbab += einsum(x669, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 3.0
    t3new_babbab += einsum(x669, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -3.0
    del x669
    x670 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], navir[1]), dtype=types[float])
    x670 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), v.bbbb.oVOV, (1, 4, 5, 6), (0, 5, 2, 3, 6, 4))
    x671 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x671 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x670, (6, 2, 7, 8, 4, 5), (6, 0, 1, 7, 8, 3))
    t3new_babbab += einsum(x671, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -6.0
    t3new_babbab += einsum(x671, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 6.0
    del x671
    x672 = np.zeros((naocc[0], naocc[1], naocc[1], naocc[1], navir[0], navir[1]), dtype=types[float])
    x672 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.OVvO, (4, 5, 2, 6), (4, 0, 1, 6, 5, 3)) * -1.0
    x673 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x673 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x672, (2, 6, 7, 1, 5, 8), (0, 7, 6, 3, 8, 4)) * -1.0
    t3new_babbab += einsum(x673, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -4.0
    t3new_babbab += einsum(x673, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 4.0
    del x673
    x674 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], navir[1]), dtype=types[float])
    x674 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), v.aabb.OVoV, (4, 5, 0, 6), (4, 1, 5, 2, 3, 6)) * -1.0
    x675 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x675 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x674, (2, 6, 5, 7, 8, 4), (0, 6, 1, 3, 7, 8))
    t3new_babbab += einsum(x675, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 4.0
    t3new_babbab += einsum(x675, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -4.0
    del x675
    x676 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x676 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 0, 2), (4, 1, 5, 3))
    t3new_abaaba += einsum(x676, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 7, 2), (4, 1, 5, 6, 3, 7)) * 12.000000000000123
    x677 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x677 += einsum(x676, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 7, 2), (4, 1, 5, 6, 3, 7))
    t3new_babbab += einsum(x677, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 4.00000000000004
    t3new_babbab += einsum(x677, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -4.00000000000004
    t3new_babbab += einsum(x677, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -4.00000000000004
    t3new_babbab += einsum(x677, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 4.00000000000004
    del x677
    x678 = np.zeros((naocc[1], naocc[1], naocc[1], naocc[1], navir[1], navir[1]), dtype=types[float])
    x678 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.vOOV, (2, 4, 5, 6), (0, 1, 5, 4, 3, 6)) * -1.0
    x679 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x679 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x678, (6, 7, 2, 0, 8, 5), (1, 7, 6, 4, 8, 3))
    t3new_babbab += einsum(x679, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x679, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x679
    x680 = np.zeros((naocc[1], naocc[1], naocc[1], naocc[1]), dtype=types[float])
    x680 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.vOvO, (2, 4, 3, 5), (0, 1, 4, 5))
    x681 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x681 += einsum(x680, (0, 1, 2, 3), t3.babbab, (2, 4, 3, 5, 6, 7), (4, 1, 0, 6, 5, 7)) * -1.0
    t3new_babbab += einsum(x681, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x681, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    del x681
    x682 = np.zeros((naocc[1], naocc[1], naocc[1], naocc[1], navir[1], navir[1]), dtype=types[float])
    x682 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.vOOV, (2, 4, 5, 6), (0, 1, 5, 4, 3, 6)) * -1.0
    x683 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x683 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x682, (6, 7, 0, 2, 8, 5), (1, 7, 6, 4, 8, 3)) * -1.0
    t3new_babbab += einsum(x683, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x683, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x683
    x684 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x684 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), v.bbbb.oVOV, (0, 4, 5, 6), (1, 5, 2, 3, 6, 4)) * -1.0
    x685 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x685 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x684, (6, 2, 7, 8, 3, 5), (1, 6, 0, 4, 7, 8))
    t3new_babbab += einsum(x685, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -4.0
    t3new_babbab += einsum(x685, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 4.0
    del x685
    x686 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=types[float])
    x686 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovOV, (0, 2, 4, 5), (1, 4, 3, 5))
    t3new_abaaba += einsum(x686, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 4.00000000000004
    x687 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x687 += einsum(x686, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (5, 0, 4, 7, 2, 6))
    t3new_babbab += einsum(x687, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 4.00000000000004
    t3new_babbab += einsum(x687, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -4.00000000000004
    t3new_babbab += einsum(x687, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -4.00000000000004
    t3new_babbab += einsum(x687, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 4.00000000000004
    del x687
    x688 = np.zeros((naocc[1], naocc[1]), dtype=types[float])
    x688 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvO, (0, 3, 2, 4), (1, 4)) * -1.0
    x689 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x689 += einsum(x688, (0, 1), t3.babbab, (2, 3, 1, 4, 5, 6), (3, 0, 2, 5, 4, 6))
    t3new_babbab += einsum(x689, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x689, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x689, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x689, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x689
    x690 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=types[float])
    x690 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.oVvO, (0, 4, 2, 5), (1, 5, 3, 4))
    t3new_abaaba += einsum(x690, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -4.00000000000004
    x691 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x691 += einsum(x690, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (5, 0, 4, 7, 2, 6))
    t3new_babbab += einsum(x691, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -4.00000000000004
    t3new_babbab += einsum(x691, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 4.00000000000004
    t3new_babbab += einsum(x691, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 4.00000000000004
    t3new_babbab += einsum(x691, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -4.00000000000004
    del x691
    x692 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x692 += einsum(x387, (0, 1), t3.babbab, (2, 1, 3, 4, 5, 6), (0, 2, 3, 5, 4, 6))
    t3new_babbab += einsum(x692, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x692, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x692
    x693 = np.zeros((navir[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x693 += einsum(t2.bbbb[np.ix_(sob,sob,sVb,sVb)], (0, 1, 2, 3), v.bbbb.oVoV, (0, 4, 1, 5), (2, 3, 5, 4)) * -1.0
    t3new_babbab += einsum(x693, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 2, 7, 3), (4, 5, 6, 0, 7, 1)) * 2.0
    x694 = np.zeros((navir[1], navir[1]), dtype=types[float])
    x694 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoV, (1, 2, 0, 4), (3, 4)) * -1.0
    t3new_abaaba += einsum(x694, (0, 1), t3.abaaba, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -2.0
    x695 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x695 += einsum(x694, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (3, 2, 4, 6, 0, 5))
    t3new_babbab += einsum(x695, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x695, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x695
    x696 = np.zeros((navir[1], navir[1]), dtype=types[float])
    x696 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoV, (1, 2, 0, 4), (3, 4)) * -1.0
    t3new_abaaba += einsum(x696, (0, 1), t3.abaaba, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -2.0
    x697 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x697 += einsum(x696, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (3, 2, 4, 6, 0, 5))
    t3new_babbab += einsum(x697, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x697, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x697
    x698 = np.zeros((naocc[1], naocc[1], navir[0], navir[1], navir[1], nvir[0]), dtype=types[float])
    x698 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (2, 3, 4, 5), v.aabb.ovoO, (0, 6, 2, 7), (3, 7, 1, 4, 5, 6)) * -1.0
    x699 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x699 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x698, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x698
    t3new_babbab += einsum(x699, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x699, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x699
    x700 = np.zeros((naocc[1], naocc[1], navir[0], navir[1], navir[1], nvir[0]), dtype=types[float])
    x700 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (2, 3, 4, 5), v.aabb.ovvV, (0, 6, 4, 7), (2, 3, 1, 5, 7, 6)) * -1.0
    x701 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x701 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x700, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x700
    t3new_babbab += einsum(x701, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x701, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x701
    x702 = np.zeros((naocc[1], naocc[1], navir[0], navir[1], navir[1], nvir[0]), dtype=types[float])
    x702 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.abab[np.ix_(soa,sOb,sVa,sVb)], (2, 3, 4, 5), v.aabb.ovvV, (2, 6, 1, 7), (0, 3, 4, 5, 7, 6)) * -1.0
    x703 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x703 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x702, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x702
    t3new_babbab += einsum(x703, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x703, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x703, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x703, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    del x703
    x704 = np.zeros((naocc[1], naocc[1], navir[0], navir[1], navir[1], nvir[0]), dtype=types[float])
    x704 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.abab[np.ix_(soa,sOb,sVa,sVb)], (2, 3, 4, 5), v.aabb.ovoO, (2, 6, 0, 7), (3, 7, 4, 1, 5, 6)) * -1.0
    x705 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x705 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x704, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x704
    t3new_babbab += einsum(x705, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x705, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x705, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x705, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x705
    x706 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], nvir[1]), dtype=types[float])
    x706 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.abab[np.ix_(sOa,sOb,sva,sVb)], (2, 3, 4, 5), v.aabb.ovvV, (0, 4, 6, 7), (2, 3, 1, 5, 7, 6)) * -1.0
    x707 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x707 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x706, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x706
    t3new_babbab += einsum(x707, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x707, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x707, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x707, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    del x707
    x708 = np.zeros((naocc[1], navir[0], navir[1], nvir[0]), dtype=types[float])
    x708 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t1.bb[np.ix_(sob,sVb)], (2, 3), v.aabb.ovoO, (0, 4, 2, 5), (5, 1, 3, 4))
    x709 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x709 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x708, (4, 5, 6, 2), (0, 1, 4, 5, 6, 3)) * -1.0
    t3new_babbab += einsum(x709, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x709, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x709, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x709, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x709
    x710 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], nvir[1]), dtype=types[float])
    x710 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (2, 3, 4, 5), v.aabb.oOov, (0, 6, 2, 7), (6, 3, 1, 4, 5, 7)) * -1.0
    x711 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x711 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x710, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x710
    t3new_babbab += einsum(x711, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x711, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x711
    x712 = np.zeros((naocc[0], navir[0], navir[1], nvir[1]), dtype=types[float])
    x712 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t1.bb[np.ix_(sob,sVb)], (2, 3), v.aabb.oOov, (0, 4, 2, 5), (4, 1, 3, 5))
    x713 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x713 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x712, (4, 5, 6, 2), (4, 0, 1, 5, 6, 3)) * -1.0
    t3new_babbab += einsum(x713, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x713, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x713
    x714 = np.zeros((naocc[1], naocc[1], navir[0], navir[1], navir[1], nvir[0]), dtype=types[float])
    x714 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (2, 3, 4, 5), v.aabb.vVov, (6, 7, 2, 1), (0, 3, 7, 4, 5, 6)) * -1.0
    x715 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x715 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x714, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x714
    t3new_babbab += einsum(x715, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x715, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x715
    x716 = np.zeros((naocc[1], naocc[1], navir[0], navir[1], navir[1], nvir[0]), dtype=types[float])
    x716 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (2, 3, 4, 5), v.aabb.vVov, (6, 7, 0, 4), (2, 3, 7, 1, 5, 6)) * -1.0
    x717 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x717 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x716, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x716
    t3new_babbab += einsum(x717, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x717, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x717
    x718 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], nvir[1]), dtype=types[float])
    x718 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.abab[np.ix_(sOa,sob,sVa,sVb)], (2, 3, 4, 5), v.bbbb.ovoO, (3, 6, 0, 7), (2, 7, 4, 1, 5, 6))
    x719 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x719 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x718, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x718
    t3new_babbab += einsum(x719, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x719, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x719, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x719, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x719
    x720 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], nvir[1]), dtype=types[float])
    x720 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.abab[np.ix_(sOa,sob,sVa,sVb)], (2, 3, 4, 5), v.bbbb.ovoO, (0, 6, 3, 7), (2, 7, 4, 1, 5, 6))
    x721 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x721 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x720, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x720
    t3new_babbab += einsum(x721, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x721, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x721, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x721, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x721
    x722 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], nvir[1]), dtype=types[float])
    x722 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.abab[np.ix_(soa,sOb,sVa,sVb)], (2, 3, 4, 5), v.aabb.oOov, (2, 6, 0, 7), (6, 3, 4, 1, 5, 7)) * -1.0
    x723 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x723 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x722, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x722
    t3new_babbab += einsum(x723, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x723, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x723, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x723, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x723
    x724 = np.zeros((naocc[1], navir[1], navir[1], nvir[1]), dtype=types[float])
    x724 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t1.bb[np.ix_(sob,sVb)], (2, 3), v.bbbb.ovoO, (2, 4, 0, 5), (5, 1, 3, 4))
    x725 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x725 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x724, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    t3new_babbab += einsum(x725, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x725, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x725, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x725, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    del x725
    x726 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], nvir[1]), dtype=types[float])
    x726 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.abab[np.ix_(sOa,sob,sVa,sVb)], (2, 3, 4, 5), v.bbbb.ovvV, (3, 1, 6, 7), (2, 0, 4, 5, 7, 6))
    x727 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x727 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x726, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x726
    t3new_babbab += einsum(x727, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new_babbab += einsum(x727, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x727, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x727, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    del x727
    x728 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], nvir[1]), dtype=types[float])
    x728 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.abab[np.ix_(sOa,sOb,sVa,svb)], (2, 3, 4, 5), v.bbbb.ovvV, (0, 6, 5, 7), (2, 3, 4, 1, 7, 6))
    x729 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x729 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x728, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x728
    t3new_babbab += einsum(x729, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x729, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x729, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x729, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x729
    x730 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], nvir[1]), dtype=types[float])
    x730 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.abab[np.ix_(sOa,sOb,sVa,svb)], (2, 3, 4, 5), v.bbbb.ovvV, (0, 5, 6, 7), (2, 3, 4, 1, 7, 6))
    x731 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x731 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x730, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x730
    t3new_babbab += einsum(x731, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x731, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x731, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x731, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x731
    x732 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], navir[1], nvir[1]), dtype=types[float])
    x732 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.abab[np.ix_(sOa,sOb,sva,sVb)], (2, 3, 4, 5), v.aabb.vVov, (4, 6, 0, 7), (2, 3, 6, 1, 5, 7)) * -1.0
    x733 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x733 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x732, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x732
    t3new_babbab += einsum(x733, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x733, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x733, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x733, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x733
    x734 = np.zeros((naocc[0], naocc[1], naocc[1], nvir[0]), dtype=types[float])
    x734 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.vOvO, (2, 3, 1, 4), (3, 0, 4, 2))
    x735 = np.zeros((naocc[0], naocc[0], naocc[1], naocc[1]), dtype=types[float])
    x735 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x734, (2, 3, 4, 1), (0, 2, 3, 4))
    del x734
    x736 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x736 += einsum(x735, (0, 1, 2, 3), t3.babbab, (4, 1, 3, 5, 6, 7), (0, 2, 4, 6, 5, 7))
    t3new_babbab += einsum(x736, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x736, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x736
    x737 = np.zeros((naocc[0], naocc[0], navir[1], nocc[1]), dtype=types[float])
    x737 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vOoV, (1, 2, 3, 4), (0, 2, 4, 3))
    x738 = np.zeros((naocc[0], naocc[0], navir[1], navir[1]), dtype=types[float])
    x738 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x737, (2, 3, 4, 0), (2, 3, 1, 4))
    del x737
    x739 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x739 += einsum(x738, (0, 1, 2, 3), t3.babbab, (4, 1, 5, 6, 7, 3), (0, 4, 5, 7, 2, 6))
    t3new_babbab += einsum(x739, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x739, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x739
    x740 = np.zeros((naocc[1], naocc[1], navir[0], nocc[0]), dtype=types[float])
    x740 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.oVvO, (2, 3, 1, 4), (0, 4, 3, 2))
    x741 = np.zeros((naocc[1], naocc[1], navir[0], navir[0]), dtype=types[float])
    x741 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x740, (2, 3, 4, 0), (2, 3, 1, 4))
    del x740
    x742 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x742 += einsum(x741, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 3, 7), (5, 0, 4, 2, 6, 7))
    t3new_babbab += einsum(x742, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x742, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x742
    x743 = np.zeros((naocc[1], nvir[1]), dtype=types[float])
    x743 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovvO, (0, 1, 2, 3), (3, 2))
    x744 = np.zeros((naocc[1], naocc[1]), dtype=types[float])
    x744 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x743, (2, 1), (0, 2))
    del x743
    t3new_abaaba += einsum(x744, (0, 1), t3.abaaba, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x745 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x745 += einsum(x744, (0, 1), t3.babbab, (2, 3, 1, 4, 5, 6), (3, 0, 2, 5, 4, 6))
    t3new_babbab += einsum(x745, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x745, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x745
    x746 = np.zeros((navir[0], navir[1], navir[1], nocc[0]), dtype=types[float])
    x746 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.oVoV, (2, 3, 0, 4), (3, 1, 4, 2))
    x747 = np.zeros((navir[0], navir[0], navir[1], navir[1]), dtype=types[float])
    x747 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x746, (2, 3, 4, 0), (1, 2, 3, 4))
    del x746
    x748 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x748 += einsum(x747, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 7, 1, 3), (5, 4, 6, 0, 2, 7))
    t3new_babbab += einsum(x748, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x748, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x748
    x749 = np.zeros((navir[1], nocc[1]), dtype=types[float])
    x749 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovoV, (0, 1, 2, 3), (3, 2))
    x750 = np.zeros((navir[1], navir[1]), dtype=types[float])
    x750 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x749, (2, 0), (1, 2))
    del x749
    t3new_abaaba += einsum(x750, (0, 1), t3.abaaba, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -2.0
    x751 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x751 += einsum(x750, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (3, 2, 4, 6, 0, 5))
    t3new_babbab += einsum(x751, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x751, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x751
    x752 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=types[float])
    x752 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x530, (2, 3, 4, 0), (2, 3, 4, 1))
    t3new_abaaba += einsum(x752, (0, 1, 2, 3), t3.aaaaaa, (4, 5, 0, 6, 7, 2), (4, 1, 5, 6, 3, 7)) * -6.0
    x753 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x753 += einsum(x752, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 7, 2), (4, 1, 5, 6, 3, 7))
    t3new_babbab += einsum(x753, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x753, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x753, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x753, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x753
    x754 = np.zeros((naocc[1], naocc[1], naocc[1], nvir[1]), dtype=types[float])
    x754 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.vOvO, (2, 3, 1, 4), (0, 3, 4, 2))
    x755 = np.zeros((naocc[1], naocc[1], naocc[1], naocc[1]), dtype=types[float])
    x755 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x754, (2, 3, 4, 1), (2, 0, 4, 3))
    del x754
    x756 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x756 += einsum(x755, (0, 1, 2, 3), t3.babbab, (2, 4, 3, 5, 6, 7), (4, 1, 0, 6, 5, 7)) * -1.0
    t3new_babbab += einsum(x756, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x756, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    del x756
    x757 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x757 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.ovOV, (2, 1, 3, 4), (0, 3, 4, 2))
    x758 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=types[float])
    x758 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x757, (2, 3, 4, 0), (2, 3, 1, 4))
    del x757
    t3new_abaaba += einsum(x758, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -2.0
    x759 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x759 += einsum(x758, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (5, 0, 4, 7, 2, 6))
    t3new_babbab += einsum(x759, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x759, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x759, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x759, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x759
    x760 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x760 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.oVvO, (2, 3, 1, 4), (0, 4, 3, 2))
    x761 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=types[float])
    x761 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x760, (2, 3, 4, 0), (2, 3, 1, 4))
    del x760
    t3new_abaaba += einsum(x761, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 2.0
    x762 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x762 += einsum(x761, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (5, 0, 4, 7, 2, 6))
    t3new_babbab += einsum(x762, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x762, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x762, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x762, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x762
    x763 = np.zeros((naocc[1], nvir[1]), dtype=types[float])
    x763 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvO, (0, 2, 1, 3), (3, 2))
    x764 = np.zeros((naocc[1], naocc[1]), dtype=types[float])
    x764 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x763, (2, 1), (0, 2))
    del x763
    t3new_abaaba += einsum(x764, (0, 1), t3.abaaba, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * 2.0
    x765 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x765 += einsum(x764, (0, 1), t3.babbab, (2, 3, 1, 4, 5, 6), (3, 0, 2, 5, 4, 6))
    t3new_babbab += einsum(x765, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x765, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x765
    x766 = np.zeros((naocc[1], nvir[1]), dtype=types[float])
    x766 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvO, (0, 1, 2, 3), (3, 2))
    x767 = np.zeros((naocc[1], naocc[1]), dtype=types[float])
    x767 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x766, (2, 1), (0, 2))
    del x766
    t3new_abaaba += einsum(x767, (0, 1), t3.abaaba, (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x768 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x768 += einsum(x767, (0, 1), t3.babbab, (2, 3, 1, 4, 5, 6), (3, 0, 2, 5, 4, 6))
    t3new_babbab += einsum(x768, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x768, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x768
    x769 = np.zeros((navir[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x769 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oVoV, (2, 3, 0, 4), (1, 3, 4, 2))
    x770 = np.zeros((navir[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x770 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x769, (2, 3, 4, 0), (2, 1, 4, 3))
    del x769
    t3new_babbab += einsum(x770, (0, 1, 2, 3), t3.babbab, (4, 5, 6, 2, 7, 3), (4, 5, 6, 0, 7, 1)) * 2.0
    x771 = np.zeros((navir[1], nocc[1]), dtype=types[float])
    x771 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovoV, (0, 1, 2, 3), (3, 2))
    x772 = np.zeros((navir[1], navir[1]), dtype=types[float])
    x772 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x771, (2, 0), (1, 2))
    del x771
    t3new_abaaba += einsum(x772, (0, 1), t3.abaaba, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -2.0
    x773 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x773 += einsum(x772, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (3, 2, 4, 6, 0, 5))
    t3new_babbab += einsum(x773, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x773, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x773
    x774 = np.zeros((navir[1], nocc[1]), dtype=types[float])
    x774 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovoV, (2, 1, 0, 3), (3, 2))
    x775 = np.zeros((navir[1], navir[1]), dtype=types[float])
    x775 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x774, (2, 0), (1, 2))
    del x774
    t3new_abaaba += einsum(x775, (0, 1), t3.abaaba, (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    x776 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x776 += einsum(x775, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 1), (3, 2, 4, 6, 0, 5))
    t3new_babbab += einsum(x776, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x776, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x776
    x777 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=types[float])
    x777 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 0, 5), (1, 3, 4, 5))
    x778 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x778 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x777, (2, 3, 4, 1), (0, 2, 3, 4))
    x779 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x779 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x778, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    t3new_babbab += einsum(x779, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x779, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x779, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x779, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    del x779
    x780 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=types[float])
    x780 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 2), (1, 3, 4, 5))
    x781 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x781 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x780, (2, 3, 4, 1), (0, 2, 3, 4))
    x782 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x782 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x781, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    t3new_babbab += einsum(x782, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new_babbab += einsum(x782, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x782, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x782, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    del x782
    x783 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], navir[1], nocc[0]), dtype=types[float])
    x783 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x780, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    x784 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x784 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x783, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x783
    t3new_babbab += einsum(x784, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x784, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x784, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x784, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x784
    x785 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], navir[1], nocc[0]), dtype=types[float])
    x785 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x777, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    x786 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x786 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x785, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x785
    t3new_babbab += einsum(x786, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x786, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x786, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x786, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x786
    x787 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x787 += einsum(x11, (0, 1), t2.abab[np.ix_(sOa,sOb,sva,sVb)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    del x11
    x788 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x788 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x787, (4, 5, 6, 0), (4, 5, 1, 2, 6, 3)) * -1.0
    t3new_babbab += einsum(x788, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x788, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x788, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x788, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x788
    x789 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x789 += einsum(x12, (0, 1), t2.abab[np.ix_(sOa,sOb,sva,sVb)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    del x12
    x790 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x790 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x789, (4, 5, 6, 0), (4, 5, 1, 2, 6, 3)) * -1.0
    t3new_babbab += einsum(x790, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x790, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x790, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x790, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x790
    x791 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=types[float])
    x791 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 0, 2), (1, 3, 4, 5))
    x792 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x792 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x791, (2, 3, 4, 1), (0, 2, 3, 4))
    x793 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x793 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x792, (4, 5, 6, 0), (4, 5, 1, 2, 6, 3)) * -1.0
    t3new_babbab += einsum(x793, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x793, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x793, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x793, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x793
    x794 = np.zeros((naocc[1], navir[0], nocc[1], nvir[0]), dtype=types[float])
    x794 += einsum(t2.abab[np.ix_(soa,sOb,sVa,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 4, 5, 3), (1, 2, 5, 4)) * -1.0
    x795 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x795 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x794, (2, 3, 4, 1), (0, 2, 3, 4))
    x796 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x796 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x795, (4, 5, 6, 0), (4, 1, 5, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x796, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x796, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x796
    x797 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], nocc[0], nocc[1]), dtype=types[float])
    x797 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (2, 3, 4, 5), v.aabb.ovov, (6, 1, 7, 4), (0, 2, 3, 5, 6, 7)) * -1.0
    x798 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x798 += einsum(t2.abab[np.ix_(soa,sob,sVa,sVb)], (0, 1, 2, 3), x797, (4, 5, 6, 7, 0, 1), (4, 5, 6, 2, 7, 3))
    del x797
    t3new_babbab += einsum(x798, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x798, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x798
    x799 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], navir[1], nocc[0]), dtype=types[float])
    x799 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x791, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    x800 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x800 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x799, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x799
    t3new_babbab += einsum(x800, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x800, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x800, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x800, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    del x800
    x801 = np.zeros((naocc[0], naocc[1], nocc[0], nocc[1]), dtype=types[float])
    x801 += einsum(t2.abab[np.ix_(sOa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x802 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], navir[1], nocc[0]), dtype=types[float])
    x802 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x801, (4, 5, 6, 0), (4, 5, 1, 2, 3, 6)) * -1.0
    x803 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x803 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x802, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x802
    t3new_babbab += einsum(x803, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x803, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x803
    x804 = np.zeros((naocc[0], navir[1], nocc[0], nvir[1]), dtype=types[float])
    x804 += einsum(t2.abab[np.ix_(sOa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 5), (0, 3, 4, 5)) * -1.0
    x805 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], navir[1], nocc[0]), dtype=types[float])
    x805 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x804, (4, 5, 6, 2), (4, 0, 1, 3, 5, 6)) * -1.0
    x806 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x806 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x805, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x805
    t3new_babbab += einsum(x806, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x806, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x806
    x807 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x807 += einsum(x0, (0, 1), t2.abab[np.ix_(sOa,sOb,sVa,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    x808 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x808 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x807, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x808, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x808, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x808
    x809 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x809 += einsum(x0, (0, 1), t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    del x0
    x810 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x810 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x809, (4, 5, 6, 1), (0, 5, 4, 2, 6, 3)) * -1.0
    t3new_babbab += einsum(x810, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x810, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x810
    x811 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], nocc[0], nocc[1]), dtype=types[float])
    x811 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.abab[np.ix_(sOa,sOb,sva,sVb)], (2, 3, 4, 5), v.aabb.ovov, (6, 4, 7, 1), (2, 0, 3, 5, 6, 7)) * -1.0
    x812 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x812 += einsum(t2.abab[np.ix_(soa,sob,sVa,sVb)], (0, 1, 2, 3), x811, (4, 5, 6, 7, 0, 1), (4, 5, 6, 2, 7, 3))
    del x811
    t3new_babbab += einsum(x812, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x812, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x812, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x812, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x812
    x813 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1]), dtype=types[float])
    x813 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 3, 4, 5))
    x814 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x814 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x813, (2, 3, 4, 1), (0, 2, 3, 4))
    x815 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x815 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x814, (4, 5, 6, 1), (0, 4, 5, 2, 3, 6))
    t3new_babbab += einsum(x815, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x815, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x815, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x815, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x815
    x816 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x816 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x804, (2, 3, 4, 1), (2, 0, 3, 4))
    x817 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x817 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x816, (4, 5, 6, 0), (4, 5, 1, 2, 6, 3)) * -1.0
    t3new_babbab += einsum(x817, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x817, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x817, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x817, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x817
    x818 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x818 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x794, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6)) * -1.0
    x819 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x819 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x818, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x818
    t3new_babbab += einsum(x819, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x819, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x819, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x819, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x819
    x820 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x820 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x801, (4, 5, 0, 6), (4, 5, 1, 2, 3, 6)) * -1.0
    x821 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x821 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x820, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x820
    t3new_babbab += einsum(x821, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x821, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x821, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x821, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x821
    x822 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x822 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x813, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    x823 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x823 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x822, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x822
    t3new_babbab += einsum(x823, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x823, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x823, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x823, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x823
    x824 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=types[float])
    x824 += einsum(x15, (0, 1), t2.abab[np.ix_(sOa,sOb,sva,sVb)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    del x15
    x825 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x825 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x824, (4, 5, 6, 0), (4, 5, 1, 2, 6, 3)) * -1.0
    t3new_babbab += einsum(x825, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x825, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x825, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x825, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x825
    x826 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1]), dtype=types[float])
    x826 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 0, 2), (1, 3, 4, 5))
    x827 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x827 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x826, (2, 3, 4, 1), (0, 2, 3, 4))
    x828 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x828 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x827, (4, 5, 6, 1), (0, 4, 5, 2, 3, 6))
    t3new_babbab += einsum(x828, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x828, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x828, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x828, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    del x828
    x829 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=types[float])
    x829 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 5), (0, 2, 4, 5))
    x830 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x830 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x829, (2, 3, 4, 1), (2, 0, 3, 4))
    x831 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x831 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x830, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x831, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x831, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x831
    x832 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1]), dtype=types[float])
    x832 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 0, 5), (1, 3, 4, 5))
    x833 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x833 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x832, (2, 3, 4, 1), (0, 2, 3, 4))
    x834 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x834 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x833, (4, 5, 6, 1), (0, 4, 5, 2, 3, 6))
    t3new_babbab += einsum(x834, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x834, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x834, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x834, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    del x834
    x835 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=types[float])
    x835 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 2, 4, 5))
    x836 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x836 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x835, (2, 3, 4, 1), (2, 0, 3, 4))
    x837 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x837 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x836, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x837, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x837, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    del x837
    x838 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], nocc[1], nocc[1]), dtype=types[float])
    x838 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.abab[np.ix_(sOa,sOb,sVa,svb)], (2, 3, 4, 5), v.bbbb.ovov, (6, 1, 7, 5), (2, 0, 3, 4, 6, 7))
    x839 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x839 += einsum(t2.bbbb[np.ix_(sob,sob,sVb,sVb)], (0, 1, 2, 3), x838, (4, 5, 6, 7, 0, 1), (4, 5, 6, 7, 2, 3))
    del x838
    t3new_babbab += einsum(x839, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x839, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x839, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    t3new_babbab += einsum(x839, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x839
    x840 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x840 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x826, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    x841 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x841 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x840, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x840
    t3new_babbab += einsum(x841, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x841, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x841, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x841, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x841
    x842 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x842 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x832, (4, 5, 6, 3), (0, 1, 4, 2, 5, 6))
    x843 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x843 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x842, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x842
    t3new_babbab += einsum(x843, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x843, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x843, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x843, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x843
    x844 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x844 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x829, (4, 5, 6, 2), (4, 0, 1, 5, 3, 6)) * -1.0
    x845 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x845 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x844, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x844
    t3new_babbab += einsum(x845, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x845, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x845
    x846 = np.zeros((naocc[1], naocc[1], nocc[1], nocc[1]), dtype=types[float])
    x846 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x847 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x847 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x846, (4, 5, 1, 6), (0, 4, 5, 2, 3, 6)) * -1.0
    x848 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x848 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x847, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x847
    t3new_babbab += einsum(x848, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x848, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x848, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x848, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    del x848
    x849 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x849 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x835, (4, 5, 6, 2), (4, 0, 1, 5, 3, 6)) * -1.0
    x850 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x850 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x849, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x849
    t3new_babbab += einsum(x850, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x850, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x850
    x851 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x851 += einsum(x1, (0, 1), t2.abab[np.ix_(sOa,sOb,sVa,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    x852 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x852 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x851, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x852, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x852, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x852
    x853 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x853 += einsum(x20, (0, 1), t2.abab[np.ix_(sOa,sOb,sVa,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    x854 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x854 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x853, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x854, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_babbab += einsum(x854, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x854
    x855 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x855 += einsum(x1, (0, 1), t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    del x1
    x856 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x856 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x855, (4, 5, 6, 1), (0, 5, 4, 2, 6, 3)) * -1.0
    t3new_babbab += einsum(x856, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x856, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x856
    x857 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=types[float])
    x857 += einsum(x20, (0, 1), t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (2, 3, 1, 4), (2, 3, 4, 0)) * -1.0
    del x20
    x858 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x858 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x857, (4, 5, 6, 1), (0, 5, 4, 2, 6, 3)) * -1.0
    t3new_babbab += einsum(x858, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_babbab += einsum(x858, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    del x858
    x859 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=types[float])
    x859 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 3, 4, 5))
    x860 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=types[float])
    x860 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x859, (2, 3, 4, 1), (2, 0, 3, 4))
    x861 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x861 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x860, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    t3new_babbab += einsum(x861, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -4.0
    t3new_babbab += einsum(x861, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 4.0
    del x861
    x862 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x862 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x859, (4, 5, 6, 2), (4, 0, 1, 5, 3, 6)) * -1.0
    x863 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x863 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x862, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x862
    t3new_babbab += einsum(x863, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 4.0
    t3new_babbab += einsum(x863, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -4.0
    del x863
    x864 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], nocc[0], nvir[0]), dtype=types[float])
    x864 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (2, 3, 4, 5), v.aabb.ovov, (6, 7, 2, 1), (0, 3, 4, 5, 6, 7)) * -1.0
    x865 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], navir[1], nocc[0]), dtype=types[float])
    x865 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x864, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x864
    x866 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x866 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x865, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x865
    t3new_babbab += einsum(x866, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_babbab += einsum(x866, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    del x866
    x867 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], nocc[0], nvir[0]), dtype=types[float])
    x867 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (2, 3, 4, 5), v.aabb.ovov, (6, 7, 0, 4), (2, 3, 1, 5, 6, 7)) * -1.0
    x868 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], navir[1], nocc[0]), dtype=types[float])
    x868 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x867, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x867
    x869 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x869 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x868, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x868
    t3new_babbab += einsum(x869, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_babbab += einsum(x869, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    del x869
    x870 = np.zeros((naocc[1], naocc[1], navir[0], navir[1], nocc[1], nvir[0]), dtype=types[float])
    x870 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.abab[np.ix_(soa,sOb,sVa,sVb)], (2, 3, 4, 5), v.aabb.ovov, (2, 6, 7, 1), (0, 3, 4, 5, 7, 6)) * -1.0
    x871 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x871 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x870, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x870
    x872 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x872 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x871, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x871
    t3new_babbab += einsum(x872, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x872, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x872, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x872, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x872
    x873 = np.zeros((naocc[0], naocc[1], navir[1], navir[1], nocc[0], nvir[1]), dtype=types[float])
    x873 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.abab[np.ix_(sOa,sOb,sva,sVb)], (2, 3, 4, 5), v.aabb.ovov, (6, 4, 0, 7), (2, 3, 1, 5, 6, 7)) * -1.0
    x874 = np.zeros((naocc[0], naocc[1], naocc[1], navir[1], navir[1], nocc[0]), dtype=types[float])
    x874 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x873, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x873
    x875 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x875 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x874, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x874
    t3new_babbab += einsum(x875, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x875, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4))
    t3new_babbab += einsum(x875, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5))
    t3new_babbab += einsum(x875, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -1.0
    del x875
    x876 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], nocc[1], nvir[1]), dtype=types[float])
    x876 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.abab[np.ix_(sOa,sob,sVa,sVb)], (2, 3, 4, 5), v.bbbb.ovov, (6, 7, 3, 1), (2, 0, 4, 5, 6, 7))
    x877 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x877 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x876, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x876
    x878 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x878 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x877, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x877
    t3new_babbab += einsum(x878, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x878, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x878, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x878, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x878
    x879 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], nocc[1], nvir[1]), dtype=types[float])
    x879 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.abab[np.ix_(sOa,sOb,sVa,svb)], (2, 3, 4, 5), v.bbbb.ovov, (6, 7, 0, 5), (2, 3, 4, 1, 6, 7))
    x880 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], nocc[1]), dtype=types[float])
    x880 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x879, (2, 3, 4, 5, 6, 1), (2, 0, 3, 4, 5, 6))
    del x879
    x881 = np.zeros((naocc[0], naocc[1], naocc[1], navir[0], navir[1], navir[1]), dtype=types[float])
    x881 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x880, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 1, 6))
    del x880
    t3new_babbab += einsum(x881, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3new_babbab += einsum(x881, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -1.0
    t3new_babbab += einsum(x881, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -1.0
    t3new_babbab += einsum(x881, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4))
    del x881
    x882 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x882 += einsum(f.aa.OO, (0, 1), t3.abaaba, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 6, 5))
    t3new_abaaba += einsum(x882, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x882, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x882
    x883 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x883 += einsum(f.aa.VV, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    t3new_abaaba += einsum(x883, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x883, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x883
    x884 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x884 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), v.aaaa.oOOV, (0, 4, 5, 6), (5, 4, 1, 2, 6, 3)) * -1.0
    t3new_abaaba += einsum(x884, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x884, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x884, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x884, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    del x884
    x885 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x885 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), v.aabb.OVoO, (4, 5, 1, 6), (0, 4, 6, 2, 5, 3))
    t3new_abaaba += einsum(x885, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x885, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x885, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x885, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x885
    x886 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x886 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), v.aabb.OVvV, (4, 5, 3, 6), (0, 4, 1, 2, 5, 6))
    t3new_abaaba += einsum(x886, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x886, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x886, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x886, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x886
    x887 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x887 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.vVOV, (2, 4, 5, 6), (0, 5, 1, 6, 4, 3)) * -1.0
    t3new_abaaba += einsum(x887, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x887, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x887, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x887, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x887
    x888 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x888 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), v.aabb.oOOV, (0, 4, 5, 6), (1, 4, 5, 2, 3, 6)) * -1.0
    t3new_abaaba += einsum(x888, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x888, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x888
    x889 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x889 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.vVOV, (2, 4, 5, 6), (0, 1, 5, 3, 4, 6)) * -1.0
    t3new_abaaba += einsum(x889, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x889, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x889
    x890 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x890 += einsum(v.aabb.OOOO, (0, 1, 2, 3), t3.abaaba, (4, 3, 1, 5, 6, 7), (4, 0, 2, 5, 7, 6))
    t3new_abaaba += einsum(x890, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x890, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x890
    x891 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x891 += einsum(v.aaaa.OOVV, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 7, 3), (4, 0, 5, 6, 2, 7))
    t3new_abaaba += einsum(x891, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x891, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x891, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x891, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    del x891
    x892 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x892 += einsum(v.aabb.OOVV, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 3, 7), (4, 0, 5, 6, 7, 2))
    t3new_abaaba += einsum(x892, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x892, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x892
    x893 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x893 += einsum(v.aaaa.OVOV, (0, 1, 2, 3), t3.abaaba, (4, 5, 2, 6, 7, 3), (4, 0, 5, 6, 1, 7))
    t3new_abaaba += einsum(x893, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x893, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x893, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x893, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x893
    x894 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x894 += einsum(v.aabb.VVOO, (0, 1, 2, 3), t3.abaaba, (4, 3, 5, 6, 7, 1), (4, 5, 2, 6, 0, 7))
    t3new_abaaba += einsum(x894, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x894, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    del x894
    x895 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x895 += einsum(v.aabb.VVVV, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 7, 3, 1), (4, 6, 5, 7, 0, 2))
    t3new_abaaba += einsum(x895, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x895, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x895
    x896 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x896 += einsum(v.aabb.OVOV, (0, 1, 2, 3), t3.babbab, (4, 5, 2, 6, 7, 3), (5, 0, 4, 7, 1, 6))
    t3new_abaaba += einsum(x896, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x896, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x896, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x896, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x896
    x897 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x897 += einsum(x299, (0, 1), t3.abaaba, (2, 3, 0, 4, 5, 6), (1, 2, 3, 4, 6, 5))
    del x299
    t3new_abaaba += einsum(x897, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x897, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x897
    x898 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x898 += einsum(x301, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 0), (2, 4, 3, 1, 5, 6))
    del x301
    t3new_abaaba += einsum(x898, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x898, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x898
    x899 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x899 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x495, (4, 5, 6, 1), (4, 0, 5, 6, 2, 3))
    del x495
    t3new_abaaba += einsum(x899, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x899, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x899, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x899, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x899
    x900 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x900 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x493, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x493
    t3new_abaaba += einsum(x900, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x900, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x900
    x901 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x901 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x303, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    del x303
    t3new_abaaba += einsum(x901, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x901, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x901
    x902 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x902 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x305, (4, 5, 6, 0), (4, 5, 1, 2, 6, 3)) * -1.0
    del x305
    t3new_abaaba += einsum(x902, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x902, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x902, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x902, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x902
    x903 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x903 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), v.aabb.oOoO, (4, 5, 1, 6), (0, 5, 6, 2, 3, 4))
    x904 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x904 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x903, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x903
    t3new_abaaba += einsum(x904, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x904, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x904, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x904, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x904
    x905 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x905 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), v.aaaa.oOoO, (4, 5, 0, 6), (5, 6, 1, 2, 3, 4)) * -1.0
    x906 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x906 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x905, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x905
    t3new_abaaba += einsum(x906, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x906, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x906, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x906, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x906
    x907 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x907 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), v.aabb.oOvV, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4))
    x908 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x908 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x907, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x907
    t3new_abaaba += einsum(x908, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x908, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x908, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x908, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x908
    x909 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x909 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.oOvV, (4, 5, 2, 6), (0, 5, 1, 6, 3, 4)) * -1.0
    x910 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x910 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x909, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x909
    t3new_abaaba += einsum(x910, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x910, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x910, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x910, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x910
    x911 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x911 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x311, (4, 5, 6, 0), (4, 5, 1, 2, 6, 3)) * -1.0
    del x311
    t3new_abaaba += einsum(x911, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x911, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x911, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x911, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x911
    x912 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x912 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovOV, (4, 2, 5, 6), (0, 5, 1, 6, 3, 4)) * -1.0
    x913 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x913 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x912, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x912
    t3new_abaaba += einsum(x913, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x913, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x913, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x913, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x913
    x914 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x914 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x506, (4, 5, 6, 1), (4, 0, 5, 2, 6, 3))
    del x506
    t3new_abaaba += einsum(x914, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x914, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x914, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x914, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x914
    x915 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x915 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), v.aabb.vVvV, (4, 5, 3, 6), (0, 1, 2, 5, 6, 4))
    x916 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x916 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x915, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x915
    t3new_abaaba += einsum(x916, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x916, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x916, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x916, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x916
    x917 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x917 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.vVvV, (4, 5, 2, 6), (0, 1, 5, 6, 3, 4)) * -1.0
    x918 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x918 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x917, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x917
    t3new_abaaba += einsum(x918, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x918, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x918, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x918, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x918
    x919 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x919 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x409, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x409
    t3new_abaaba += einsum(x919, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x919, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x919
    x920 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x920 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovOV, (4, 2, 5, 6), (0, 1, 5, 3, 6, 4)) * -1.0
    x921 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x921 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x920, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x920
    t3new_abaaba += einsum(x921, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x921, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x921
    x922 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x922 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x530, (4, 5, 6, 1), (0, 4, 5, 2, 6, 3))
    del x530
    t3new_abaaba += einsum(x922, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x922, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x922, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x922, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x922
    x923 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], nocc[1]), dtype=types[float])
    x923 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 6, 3), (0, 4, 1, 2, 5, 6))
    x924 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x924 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x923, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x923
    t3new_abaaba += einsum(x924, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x924, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x924, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x924, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x924
    x925 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x925 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x512, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    del x512
    t3new_abaaba += einsum(x925, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x925, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x925
    x926 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], nocc[1]), dtype=types[float])
    x926 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), v.aabb.oOoO, (0, 4, 5, 6), (1, 4, 6, 2, 3, 5)) * -1.0
    x927 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x927 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x926, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x926
    t3new_abaaba += einsum(x927, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x927, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x927
    x928 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[1], nvir[1]), dtype=types[float])
    x928 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.vVvV, (2, 4, 5, 6), (0, 1, 3, 4, 6, 5)) * -1.0
    x929 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x929 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x928, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x928
    t3new_abaaba += einsum(x929, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x929, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x929
    x930 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], nocc[1]), dtype=types[float])
    x930 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.vVoO, (2, 4, 5, 6), (0, 1, 6, 3, 4, 5)) * -1.0
    x931 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x931 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x930, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x930
    t3new_abaaba += einsum(x931, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x931, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    del x931
    x932 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x932 += einsum(x321, (0, 1, 2, 3), t3.abaaba, (2, 4, 3, 5, 6, 7), (0, 1, 4, 5, 7, 6))
    del x321
    t3new_abaaba += einsum(x932, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x932, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x932
    x933 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x933 += einsum(x323, (0, 1, 2, 3), t3.abaaba, (4, 5, 0, 6, 7, 3), (4, 1, 5, 2, 6, 7))
    del x323
    t3new_abaaba += einsum(x933, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x933, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x933, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x933, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x933
    x934 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x934 += einsum(x325, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 7, 3), (4, 0, 5, 2, 6, 7))
    del x325
    t3new_abaaba += einsum(x934, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x934, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x934, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x934, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x934
    x935 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x935 += einsum(x327, (0, 1), t3.abaaba, (2, 3, 1, 4, 5, 6), (2, 0, 3, 4, 6, 5))
    del x327
    t3new_abaaba += einsum(x935, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x935, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x935
    x936 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x936 += einsum(x329, (0, 1), t3.abaaba, (2, 3, 1, 4, 5, 6), (2, 0, 3, 4, 6, 5))
    del x329
    t3new_abaaba += einsum(x936, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x936, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x936
    x937 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x937 += einsum(x534, (0, 1, 2, 3), t3.abaaba, (4, 3, 1, 5, 6, 7), (0, 4, 2, 5, 7, 6))
    del x534
    t3new_abaaba += einsum(x937, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x937, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x937
    x938 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x938 += einsum(x331, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 7, 3), (0, 4, 5, 6, 2, 7))
    del x331
    t3new_abaaba += einsum(x938, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x938, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x938, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x938, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    del x938
    x939 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x939 += einsum(x536, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 3, 7), (0, 4, 5, 6, 7, 2))
    del x536
    t3new_abaaba += einsum(x939, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x939, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x939
    x940 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x940 += einsum(x333, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 7, 2), (0, 4, 5, 6, 3, 7))
    del x333
    t3new_abaaba += einsum(x940, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x940, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x940, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x940, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    del x940
    x941 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x941 += einsum(x538, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 7, 3), (4, 5, 0, 2, 6, 7))
    del x538
    t3new_abaaba += einsum(x941, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x941, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x941
    x942 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x942 += einsum(x542, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 7, 3, 1), (4, 6, 5, 0, 7, 2))
    del x542
    t3new_abaaba += einsum(x942, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x942, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x942
    x943 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x943 += einsum(x335, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 2, 7, 3), (4, 6, 5, 0, 1, 7))
    del x335
    t3new_abaaba += einsum(x943, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x943, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x943
    x944 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x944 += einsum(x337, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (2, 4, 3, 5, 0, 6))
    del x337
    t3new_abaaba += einsum(x944, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x944, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x944
    x945 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x945 += einsum(x339, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 0), (2, 4, 3, 5, 1, 6))
    del x339
    t3new_abaaba += einsum(x945, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x945, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    del x945
    x946 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x946 += einsum(x317, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (5, 0, 4, 2, 7, 6))
    del x317
    t3new_abaaba += einsum(x946, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x946, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x946, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x946, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x946
    x947 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x947 += einsum(x319, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (0, 5, 4, 7, 2, 6))
    del x319
    t3new_abaaba += einsum(x947, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x947, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x947, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x947, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    del x947
    x948 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x948 += einsum(x550, (0, 1, 2, 3), t3.abaaba, (4, 3, 1, 5, 6, 7), (4, 0, 2, 5, 7, 6))
    del x550
    t3new_abaaba += einsum(x948, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x948, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x948
    x949 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x949 += einsum(x558, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 3, 7), (4, 0, 5, 6, 7, 2))
    del x558
    t3new_abaaba += einsum(x949, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x949, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x949
    x950 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x950 += einsum(x341, (0, 1), t3.abaaba, (2, 3, 1, 4, 5, 6), (2, 0, 3, 4, 6, 5))
    del x341
    t3new_abaaba += einsum(x950, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x950, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x950
    x951 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x951 += einsum(x574, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 7, 3), (4, 5, 0, 6, 2, 7))
    del x574
    t3new_abaaba += einsum(x951, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x951, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    del x951
    x952 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x952 += einsum(x576, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 7, 3, 1), (4, 6, 5, 7, 0, 2))
    del x576
    t3new_abaaba += einsum(x952, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x952, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    del x952
    x953 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x953 += einsum(x343, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (2, 4, 3, 5, 0, 6))
    del x343
    t3new_abaaba += einsum(x953, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x953, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x953
    x954 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x954 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x604, (4, 5, 6, 1), (0, 4, 5, 2, 6, 3))
    del x604
    t3new_abaaba += einsum(x954, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x954, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x954, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x954, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x954
    x955 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x955 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x600, (4, 5, 6, 1), (0, 4, 5, 2, 6, 3))
    del x600
    t3new_abaaba += einsum(x955, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x955, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x955, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x955, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    del x955
    x956 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x956 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x612, (4, 5, 6, 3), (0, 4, 1, 2, 5, 6))
    del x612
    t3new_abaaba += einsum(x956, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x956, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x956, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x956, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x956
    x957 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x957 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x610, (4, 5, 6, 1), (0, 4, 5, 2, 6, 3))
    del x610
    t3new_abaaba += einsum(x957, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x957, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x957, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x957, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x957
    x958 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x958 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x345, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    del x345
    t3new_abaaba += einsum(x958, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x958, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x958, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x958, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x958
    x959 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x959 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x622, (4, 5, 6, 3), (0, 4, 1, 2, 5, 6))
    del x622
    t3new_abaaba += einsum(x959, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x959, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x959, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x959, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x959
    x960 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x960 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x618, (4, 5, 6, 3), (0, 4, 1, 2, 5, 6))
    del x618
    t3new_abaaba += einsum(x960, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x960, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x960, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x960, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x960
    x961 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x961 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x347, (4, 5, 6, 2), (0, 4, 1, 5, 6, 3)) * -1.0
    del x347
    t3new_abaaba += einsum(x961, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x961, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x961, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x961, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x961
    x962 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x962 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x626, (4, 5, 6, 1), (4, 0, 5, 2, 6, 3))
    del x626
    t3new_abaaba += einsum(x962, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x962, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x962, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x962, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x962
    x963 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x963 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x628, (4, 5, 6, 3), (0, 4, 1, 2, 5, 6))
    del x628
    t3new_abaaba += einsum(x963, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x963, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x963, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x963, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x963
    x964 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x964 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x353, (4, 5, 6, 2), (0, 4, 1, 5, 6, 3)) * -1.0
    del x353
    t3new_abaaba += einsum(x964, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x964, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x964, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x964, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x964
    x965 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x965 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x349, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    del x349
    t3new_abaaba += einsum(x965, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x965, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x965, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x965, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    del x965
    x966 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x966 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x578, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    del x578
    t3new_abaaba += einsum(x966, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x966, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x966
    x967 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x967 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x351, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    del x351
    t3new_abaaba += einsum(x967, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x967, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x967, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x967, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    del x967
    x968 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x968 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x580, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    del x580
    t3new_abaaba += einsum(x968, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x968, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x968
    x969 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x969 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x630, (4, 5, 6, 1), (0, 4, 5, 2, 6, 3))
    del x630
    t3new_abaaba += einsum(x969, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x969, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x969, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x969, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x969
    x970 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x970 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x586, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    del x586
    t3new_abaaba += einsum(x970, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x970, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x970
    x971 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x971 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x582, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x582
    t3new_abaaba += einsum(x971, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x971, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x971
    x972 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x972 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x590, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x590
    t3new_abaaba += einsum(x972, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x972, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x972
    x973 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x973 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x355, (4, 5, 6, 2), (0, 4, 1, 5, 6, 3)) * -1.0
    del x355
    t3new_abaaba += einsum(x973, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x973, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x973, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x973, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x973
    x974 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x974 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x632, (4, 5, 6, 3), (0, 4, 1, 2, 5, 6))
    del x632
    t3new_abaaba += einsum(x974, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x974, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x974, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x974, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x974
    x975 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x975 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x359, (4, 5, 6, 2), (0, 4, 1, 5, 6, 3)) * -1.0
    del x359
    t3new_abaaba += einsum(x975, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x975, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x975, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x975, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x975
    x976 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x976 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x357, (4, 5, 6, 0), (5, 4, 1, 2, 6, 3))
    del x357
    t3new_abaaba += einsum(x976, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x976, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x976, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x976, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    del x976
    x977 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x977 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x596, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x596
    t3new_abaaba += einsum(x977, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x977, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x977
    x978 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x978 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x588, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x588
    t3new_abaaba += einsum(x978, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x978, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x978
    x979 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x979 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x592, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x592
    t3new_abaaba += einsum(x979, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x979, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x979
    x980 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x980 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x608, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    del x608
    t3new_abaaba += einsum(x980, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 4.0
    t3new_abaaba += einsum(x980, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -4.0
    del x980
    x981 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x981 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x624, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x624
    t3new_abaaba += einsum(x981, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_abaaba += einsum(x981, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    del x981
    x982 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x982 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x644, (6, 2, 7, 1, 8, 4), (6, 0, 7, 3, 5, 8))
    del x644
    t3new_abaaba += einsum(x982, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x982, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x982
    x983 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x983 += einsum(x646, (0, 1, 2, 3), t3.abaaba, (4, 3, 1, 5, 6, 7), (0, 4, 2, 5, 7, 6))
    del x646
    t3new_abaaba += einsum(x983, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x983, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x983
    x984 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x984 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x648, (6, 2, 7, 1, 8, 5), (6, 0, 7, 8, 3, 4))
    del x648
    t3new_abaaba += einsum(x984, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x984, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x984, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x984, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x984
    x985 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x985 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x650, (6, 2, 7, 5, 8, 4), (6, 0, 1, 7, 3, 8))
    del x650
    t3new_abaaba += einsum(x985, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x985, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x985, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x985, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x985
    x986 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x986 += einsum(x652, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 3, 7), (0, 4, 5, 6, 7, 2))
    del x652
    t3new_abaaba += einsum(x986, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.00000000000002
    t3new_abaaba += einsum(x986, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.00000000000002
    del x986
    x987 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x987 += einsum(x365, (0, 1), t3.abaaba, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 6, 5))
    del x365
    t3new_abaaba += einsum(x987, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x987, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x987
    x988 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x988 += einsum(x367, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x367
    t3new_abaaba += einsum(x988, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.00000000000002
    t3new_abaaba += einsum(x988, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.00000000000002
    t3new_abaaba += einsum(x988, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.00000000000002
    t3new_abaaba += einsum(x988, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.00000000000002
    del x988
    x989 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x989 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x654, (6, 1, 7, 5, 8, 4), (0, 2, 6, 7, 3, 8))
    del x654
    t3new_abaaba += einsum(x989, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x989, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x989
    x990 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x990 += einsum(x660, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 7, 3), (4, 5, 0, 2, 6, 7))
    del x660
    t3new_abaaba += einsum(x990, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.00000000000002
    t3new_abaaba += einsum(x990, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.00000000000002
    del x990
    x991 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x991 += einsum(x662, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 7, 3, 1), (4, 6, 5, 0, 7, 2))
    del x662
    t3new_abaaba += einsum(x991, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x991, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x991
    x992 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x992 += einsum(x369, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    del x369
    t3new_abaaba += einsum(x992, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x992, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x992
    x993 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x993 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x666, (6, 7, 0, 2, 8, 5), (6, 1, 7, 8, 4, 3))
    del x666
    t3new_abaaba += einsum(x993, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x993, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x993, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x993, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x993
    x994 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x994 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x668, (6, 7, 0, 2, 8, 5), (6, 1, 7, 8, 4, 3))
    del x668
    t3new_abaaba += einsum(x994, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x994, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x994, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x994, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x994
    x995 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x995 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x670, (6, 2, 7, 8, 3, 5), (6, 1, 0, 7, 4, 8))
    del x670
    t3new_abaaba += einsum(x995, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x995, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x995, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x995, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x995
    x996 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x996 += einsum(x361, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (0, 5, 4, 2, 7, 6))
    del x361
    t3new_abaaba += einsum(x996, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.00000000000002
    t3new_abaaba += einsum(x996, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.00000000000002
    t3new_abaaba += einsum(x996, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.00000000000002
    t3new_abaaba += einsum(x996, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.00000000000002
    del x996
    x997 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x997 += einsum(x363, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (0, 5, 4, 2, 7, 6))
    del x363
    t3new_abaaba += einsum(x997, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.00000000000002
    t3new_abaaba += einsum(x997, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.00000000000002
    t3new_abaaba += einsum(x997, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.00000000000002
    t3new_abaaba += einsum(x997, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.00000000000002
    del x997
    x998 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x998 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x634, (6, 2, 1, 7, 5, 8), (6, 0, 7, 3, 4, 8)) * -1.0
    del x634
    t3new_abaaba += einsum(x998, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -3.0
    t3new_abaaba += einsum(x998, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 3.0
    del x998
    x999 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x999 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x636, (6, 1, 2, 7, 5, 8), (6, 0, 7, 3, 4, 8))
    del x636
    t3new_abaaba += einsum(x999, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -3.0
    t3new_abaaba += einsum(x999, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 3.0
    del x999
    x1000 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1000 += einsum(t3.aaaaaa, (0, 1, 2, 3, 4, 5), x638, (2, 6, 7, 5, 4, 8), (0, 1, 6, 7, 3, 8)) * -1.0
    del x638
    t3new_abaaba += einsum(x1000, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 6.0
    t3new_abaaba += einsum(x1000, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    del x1000
    x1001 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1001 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x377, (6, 7, 0, 2, 8, 5), (7, 6, 1, 8, 3, 4)) * -1.0
    del x377
    t3new_abaaba += einsum(x1001, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1001, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x1001
    x1002 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1002 += einsum(x379, (0, 1, 2, 3), t3.abaaba, (2, 4, 3, 5, 6, 7), (1, 0, 4, 5, 7, 6)) * -1.0
    del x379
    t3new_abaaba += einsum(x1002, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1002, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    del x1002
    x1003 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1003 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x381, (6, 7, 0, 2, 8, 5), (7, 6, 1, 8, 3, 4)) * -1.0
    del x381
    t3new_abaaba += einsum(x1003, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1003, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x1003
    x1004 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1004 += einsum(t3.abaaba, (0, 1, 2, 3, 4, 5), x383, (6, 2, 7, 8, 3, 5), (6, 0, 1, 7, 8, 4))
    del x383
    t3new_abaaba += einsum(x1004, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_abaaba += einsum(x1004, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 4.0
    del x1004
    x1005 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1005 += einsum(x385, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x385
    t3new_abaaba += einsum(x1005, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.00000000000004
    t3new_abaaba += einsum(x1005, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.00000000000004
    t3new_abaaba += einsum(x1005, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -4.00000000000004
    t3new_abaaba += einsum(x1005, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 4.00000000000004
    del x1005
    x1006 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1006 += einsum(x387, (0, 1), t3.abaaba, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 6, 5))
    del x387
    t3new_abaaba += einsum(x1006, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1006, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1006, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1006, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x1006
    x1007 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1007 += einsum(x389, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x389
    t3new_abaaba += einsum(x1007, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -4.00000000000004
    t3new_abaaba += einsum(x1007, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 4.00000000000004
    t3new_abaaba += einsum(x1007, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 4.00000000000004
    t3new_abaaba += einsum(x1007, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -4.00000000000004
    del x1007
    x1008 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1008 += einsum(x688, (0, 1), t3.abaaba, (2, 1, 3, 4, 5, 6), (2, 3, 0, 4, 6, 5))
    t3new_abaaba += einsum(x1008, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1008, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x1008
    x1009 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1009 += einsum(x393, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    del x393
    t3new_abaaba += einsum(x1009, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1009, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x1009
    x1010 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1010 += einsum(x395, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    del x395
    t3new_abaaba += einsum(x1010, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1010, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x1010
    x1011 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1011 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x371, (6, 7, 1, 2, 8, 5), (7, 6, 0, 8, 4, 3)) * -1.0
    del x371
    t3new_abaaba += einsum(x1011, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -4.0
    t3new_abaaba += einsum(x1011, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 4.0
    del x1011
    x1012 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1012 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x373, (6, 2, 7, 8, 4, 5), (6, 1, 0, 7, 8, 3))
    del x373
    t3new_abaaba += einsum(x1012, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 4.0
    t3new_abaaba += einsum(x1012, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -4.0
    del x1012
    x1013 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1013 += einsum(x375, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (0, 5, 4, 2, 7, 6))
    del x375
    t3new_abaaba += einsum(x1013, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.00000000000004
    t3new_abaaba += einsum(x1013, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.00000000000004
    t3new_abaaba += einsum(x1013, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -4.00000000000004
    t3new_abaaba += einsum(x1013, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 4.00000000000004
    del x1013
    x1014 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x1014 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.abab[np.ix_(soa,sOb,sVa,sVb)], (2, 3, 4, 5), v.aaaa.ovoO, (2, 6, 0, 7), (7, 3, 1, 4, 5, 6)) * -1.0
    x1015 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1015 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1014, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1014
    t3new_abaaba += einsum(x1015, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1015, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1015, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1015, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x1015
    x1016 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x1016 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.abab[np.ix_(soa,sOb,sVa,sVb)], (2, 3, 4, 5), v.aaaa.ovoO, (0, 6, 2, 7), (7, 3, 1, 4, 5, 6)) * -1.0
    x1017 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1017 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1016, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1016
    t3new_abaaba += einsum(x1017, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1017, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1017, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x1017, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x1017
    x1018 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1018 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x401, (4, 5, 6, 2), (0, 4, 1, 5, 6, 3)) * -1.0
    del x401
    t3new_abaaba += einsum(x1018, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1018, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1018, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1018, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x1018
    x1019 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x1019 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), t2.abab[np.ix_(soa,sOb,sVa,sVb)], (2, 3, 4, 5), v.aaaa.ovvV, (2, 1, 6, 7), (0, 3, 4, 7, 5, 6)) * -1.0
    x1020 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1020 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1019, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1019
    t3new_abaaba += einsum(x1020, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x1020, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1020, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1020, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    del x1020
    x1021 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x1021 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.abab[np.ix_(sOa,sob,sVa,sVb)], (2, 3, 4, 5), v.aabb.ovoO, (0, 6, 3, 7), (2, 7, 1, 4, 5, 6))
    x1022 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1022 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1021, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1021
    t3new_abaaba += einsum(x1022, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1022, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1022, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1022, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x1022
    x1023 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x1023 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.abab[np.ix_(sOa,sOb,sVa,svb)], (2, 3, 4, 5), v.aabb.ovvV, (0, 6, 5, 7), (2, 3, 1, 4, 7, 6))
    x1024 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1024 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1023, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1023
    t3new_abaaba += einsum(x1024, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1024, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1024, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x1024, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x1024
    x1025 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x1025 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.abab[np.ix_(sOa,sOb,sva,sVb)], (2, 3, 4, 5), v.aaaa.ovvV, (0, 6, 4, 7), (2, 3, 1, 7, 5, 6)) * -1.0
    x1026 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1026 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1025, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1025
    t3new_abaaba += einsum(x1026, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1026, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1026, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1026, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x1026
    x1027 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x1027 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.abab[np.ix_(sOa,sOb,sva,sVb)], (2, 3, 4, 5), v.aaaa.ovvV, (0, 4, 6, 7), (2, 3, 1, 7, 5, 6)) * -1.0
    x1028 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1028 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1027, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1027
    t3new_abaaba += einsum(x1028, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1028, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1028, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x1028, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x1028
    x1029 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[1], nvir[1]), dtype=types[float])
    x1029 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.abab[np.ix_(sOa,sob,sVa,sVb)], (2, 3, 4, 5), v.aabb.oOov, (0, 6, 3, 7), (2, 6, 1, 4, 5, 7))
    x1030 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1030 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1029, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x1029
    t3new_abaaba += einsum(x1030, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1030, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x1030, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1030, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x1030
    x1031 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1031 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x712, (4, 5, 6, 3), (0, 4, 1, 5, 2, 6))
    del x712
    t3new_abaaba += einsum(x1031, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1031, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x1031, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1031, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x1031
    x1032 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x1032 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.abab[np.ix_(sOa,sob,sVa,sVb)], (2, 3, 4, 5), v.aabb.vVov, (6, 7, 3, 1), (2, 0, 4, 7, 5, 6))
    x1033 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1033 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1032, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1032
    t3new_abaaba += einsum(x1033, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1033, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1033, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x1033, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x1033
    x1034 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x1034 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.abab[np.ix_(sOa,sOb,sVa,svb)], (2, 3, 4, 5), v.aabb.vVov, (6, 7, 0, 5), (2, 3, 4, 7, 1, 6))
    x1035 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1035 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1034, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1034
    t3new_abaaba += einsum(x1035, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1035, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1035, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x1035, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x1035
    x1036 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x1036 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (2, 3, 4, 5), v.aabb.ovvV, (2, 6, 1, 7), (3, 0, 4, 5, 7, 6)) * -1.0
    x1037 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1037 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1036, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1036
    t3new_abaaba += einsum(x1037, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1037, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x1037
    x1038 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], navir[1], nvir[0]), dtype=types[float])
    x1038 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (2, 3, 4, 5), v.aabb.ovoO, (2, 6, 0, 7), (3, 7, 4, 5, 1, 6)) * -1.0
    x1039 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1039 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1038, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1038
    t3new_abaaba += einsum(x1039, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1039, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1039
    x1040 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[1], nvir[1]), dtype=types[float])
    x1040 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (2, 3, 4, 5), v.aabb.ovvV, (0, 4, 6, 7), (2, 3, 1, 5, 7, 6)) * -1.0
    x1041 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1041 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1040, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x1040
    t3new_abaaba += einsum(x1041, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1041, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x1041
    x1042 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1042 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x708, (4, 5, 6, 2), (0, 1, 4, 5, 3, 6)) * -1.0
    del x708
    t3new_abaaba += einsum(x1042, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1042, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x1042
    x1043 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[1], nvir[1]), dtype=types[float])
    x1043 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (2, 3, 4, 5), v.aabb.oOov, (2, 6, 0, 7), (3, 6, 4, 5, 1, 7)) * -1.0
    x1044 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1044 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1043, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x1043
    t3new_abaaba += einsum(x1044, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1044, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x1044
    x1045 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], navir[1], nvir[1]), dtype=types[float])
    x1045 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (2, 3, 4, 5), v.aabb.vVov, (4, 6, 0, 7), (2, 3, 5, 6, 1, 7)) * -1.0
    x1046 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1046 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1045, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x1045
    t3new_abaaba += einsum(x1046, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1046, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    del x1046
    x1047 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1047 += einsum(x413, (0, 1, 2, 3), t3.abaaba, (3, 4, 2, 5, 6, 7), (1, 0, 4, 5, 7, 6))
    del x413
    t3new_abaaba += einsum(x1047, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1047, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    del x1047
    x1048 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1048 += einsum(x416, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x416
    t3new_abaaba += einsum(x1048, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1048, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1048, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1048, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1048
    x1049 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1049 += einsum(x419, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x419
    t3new_abaaba += einsum(x1049, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1049, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1049, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1049, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x1049
    x1050 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1050 += einsum(x422, (0, 1), t3.abaaba, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 6, 5))
    del x422
    t3new_abaaba += einsum(x1050, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1050, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1050
    x1051 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1051 += einsum(x425, (0, 1), t3.abaaba, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 6, 5))
    del x425
    t3new_abaaba += einsum(x1051, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1051, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x1051
    x1052 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1052 += einsum(x431, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    del x431
    t3new_abaaba += einsum(x1052, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1052, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x1052
    x1053 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1053 += einsum(x434, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    del x434
    t3new_abaaba += einsum(x1053, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1053, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x1053
    x1054 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1054 += einsum(x410, (0, 1, 2, 3), t3.babbab, (4, 5, 1, 6, 7, 3), (0, 5, 4, 2, 7, 6))
    del x410
    t3new_abaaba += einsum(x1054, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1054, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1054, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1054, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1054
    x1055 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1055 += einsum(x735, (0, 1, 2, 3), t3.abaaba, (4, 3, 1, 5, 6, 7), (0, 4, 2, 5, 7, 6))
    del x735
    t3new_abaaba += einsum(x1055, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1055, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1055
    x1056 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1056 += einsum(x738, (0, 1, 2, 3), t3.abaaba, (4, 5, 1, 6, 3, 7), (0, 4, 5, 6, 7, 2))
    del x738
    t3new_abaaba += einsum(x1056, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1056, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1056
    x1057 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1057 += einsum(x437, (0, 1), t3.abaaba, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 6, 5))
    del x437
    t3new_abaaba += einsum(x1057, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1057, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x1057
    x1058 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1058 += einsum(x741, (0, 1, 2, 3), t3.abaaba, (4, 1, 5, 6, 7, 3), (4, 5, 0, 2, 6, 7))
    del x741
    t3new_abaaba += einsum(x1058, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1058, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x1058
    x1059 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1059 += einsum(x747, (0, 1, 2, 3), t3.abaaba, (4, 5, 6, 7, 3, 1), (4, 6, 5, 0, 7, 2))
    del x747
    t3new_abaaba += einsum(x1059, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1059, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x1059
    x1060 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1060 += einsum(x440, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 1), (2, 4, 3, 0, 5, 6))
    del x440
    t3new_abaaba += einsum(x1060, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1060, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x1060
    x1061 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], nocc[0], nocc[1]), dtype=types[float])
    x1061 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), t2.abab[np.ix_(sOa,sOb,sVa,svb)], (2, 3, 4, 5), v.aabb.ovov, (6, 1, 7, 5), (0, 2, 3, 4, 6, 7))
    x1062 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1062 += einsum(t2.abab[np.ix_(soa,sob,sVa,sVb)], (0, 1, 2, 3), x1061, (4, 5, 6, 7, 0, 1), (4, 5, 6, 7, 2, 3))
    del x1061
    t3new_abaaba += einsum(x1062, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1062, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1062, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x1062, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x1062
    x1063 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1063 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x795, (4, 5, 6, 1), (4, 0, 5, 2, 6, 3))
    del x795
    t3new_abaaba += einsum(x1063, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1063, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1063, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1063, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x1063
    x1064 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1064 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x443, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    del x443
    t3new_abaaba += einsum(x1064, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1064, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1064, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x1064, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x1064
    x1065 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1065 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x442, (4, 5, 6, 2), (0, 4, 1, 5, 3, 6)) * -1.0
    del x442
    x1066 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1066 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1065, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1065
    t3new_abaaba += einsum(x1066, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1066, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1066, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1066, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x1066
    x1067 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1067 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x801, (4, 5, 6, 1), (4, 0, 5, 2, 3, 6))
    x1068 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1068 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1067, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1067
    t3new_abaaba += einsum(x1068, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1068, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1068, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1068, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x1068
    x1069 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1069 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x804, (4, 5, 6, 3), (0, 4, 1, 2, 5, 6))
    del x804
    x1070 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1070 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1069, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1069
    t3new_abaaba += einsum(x1070, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1070, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1070, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1070, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x1070
    x1071 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1071 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x807, (4, 5, 6, 1), (4, 0, 5, 6, 2, 3))
    del x807
    t3new_abaaba += einsum(x1071, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1071, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1071, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1071, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x1071
    x1072 = np.zeros((naocc[0], naocc[0], naocc[1], navir[1], nocc[0], nocc[0]), dtype=types[float])
    x1072 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), t2.abab[np.ix_(sOa,sOb,sva,sVb)], (2, 3, 4, 5), v.aaaa.ovov, (6, 1, 7, 4), (0, 2, 3, 5, 6, 7)) * -1.0
    x1073 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1073 += einsum(t2.aaaa[np.ix_(soa,soa,sVa,sVa)], (0, 1, 2, 3), x1072, (4, 5, 6, 7, 0, 1), (4, 5, 6, 2, 3, 7))
    del x1072
    t3new_abaaba += einsum(x1073, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1073, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1073, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1073, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x1073
    x1074 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1074 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x448, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    del x448
    t3new_abaaba += einsum(x1074, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1074, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1074, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1074, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    del x1074
    x1075 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1075 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x778, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x778
    t3new_abaaba += einsum(x1075, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1075, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1075
    x1076 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1076 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x451, (4, 5, 6, 0), (4, 5, 1, 6, 2, 3)) * -1.0
    del x451
    t3new_abaaba += einsum(x1076, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1076, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1076, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1076, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    del x1076
    x1077 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1077 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x781, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x781
    t3new_abaaba += einsum(x1077, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1077, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x1077
    x1078 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1078 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x447, (4, 5, 6, 2), (0, 4, 1, 5, 3, 6)) * -1.0
    del x447
    x1079 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1079 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1078, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1078
    t3new_abaaba += einsum(x1079, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1079, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1079, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1079, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x1079
    x1080 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1080 += einsum(t2.abab[np.ix_(sOa,sOb,sva,sVb)], (0, 1, 2, 3), x450, (4, 5, 6, 2), (0, 4, 1, 5, 3, 6)) * -1.0
    del x450
    x1081 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1081 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1080, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1080
    t3new_abaaba += einsum(x1081, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1081, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1081, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1081, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1081
    x1082 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1082 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x457, (4, 5, 0, 6), (4, 5, 1, 2, 3, 6))
    del x457
    x1083 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1083 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1082, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1082
    t3new_abaaba += einsum(x1083, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1083, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1083, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1083, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    del x1083
    x1084 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1084 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x777, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x777
    x1085 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1085 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1084, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1084
    t3new_abaaba += einsum(x1085, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1085, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x1085
    x1086 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1086 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x780, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x780
    x1087 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1087 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1086, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1086
    t3new_abaaba += einsum(x1087, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1087, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    del x1087
    x1088 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1088 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x787, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x787
    t3new_abaaba += einsum(x1088, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1088, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1088
    x1089 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1089 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x789, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x789
    t3new_abaaba += einsum(x1089, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1089, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    del x1089
    x1090 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1090 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x462, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    del x462
    t3new_abaaba += einsum(x1090, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1090, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x1090
    x1091 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1091 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x464, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    del x464
    t3new_abaaba += einsum(x1091, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1091, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    del x1091
    x1092 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1092 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x792, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x792
    t3new_abaaba += einsum(x1092, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_abaaba += einsum(x1092, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 4.0
    del x1092
    x1093 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1093 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x791, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x791
    x1094 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1094 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1093, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1093
    t3new_abaaba += einsum(x1094, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_abaaba += einsum(x1094, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    del x1094
    x1095 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1095 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x836, (4, 5, 6, 1), (0, 4, 5, 2, 6, 3))
    del x836
    t3new_abaaba += einsum(x1095, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1095, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1095, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1095, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x1095
    x1096 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1096 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x830, (4, 5, 6, 1), (0, 4, 5, 2, 6, 3))
    del x830
    t3new_abaaba += einsum(x1096, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x1096, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1096, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1096, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    del x1096
    x1097 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], nocc[1]), dtype=types[float])
    x1097 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x835, (4, 5, 6, 3), (0, 4, 1, 2, 5, 6))
    del x835
    x1098 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1098 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1097, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1097
    t3new_abaaba += einsum(x1098, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1098, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1098, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1098, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x1098
    x1099 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], nocc[1]), dtype=types[float])
    x1099 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x829, (4, 5, 6, 3), (0, 4, 1, 2, 5, 6))
    del x829
    x1100 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1100 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1099, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1099
    t3new_abaaba += einsum(x1100, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1100, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1100, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x1100, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x1100
    x1101 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1101 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x851, (4, 5, 6, 1), (4, 0, 5, 6, 2, 3))
    del x851
    t3new_abaaba += einsum(x1101, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1101, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1101, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1101, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    del x1101
    x1102 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1102 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x853, (4, 5, 6, 1), (4, 0, 5, 6, 2, 3))
    del x853
    t3new_abaaba += einsum(x1102, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1102, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1102, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    t3new_abaaba += einsum(x1102, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    del x1102
    x1103 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1103 += einsum(t2.abab[np.ix_(sOa,sob,sVa,sVb)], (0, 1, 2, 3), x860, (4, 5, 6, 1), (0, 4, 5, 2, 6, 3))
    del x860
    t3new_abaaba += einsum(x1103, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1103, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1103, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1103, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1103
    x1104 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1104 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x816, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    del x816
    t3new_abaaba += einsum(x1104, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1104, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x1104
    x1105 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], nocc[0], nocc[1]), dtype=types[float])
    x1105 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (2, 3, 4, 5), v.aabb.ovov, (6, 4, 7, 1), (2, 3, 0, 5, 6, 7)) * -1.0
    x1106 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1106 += einsum(t2.abab[np.ix_(soa,sob,sVa,sVb)], (0, 1, 2, 3), x1105, (4, 5, 6, 7, 0, 1), (4, 5, 6, 7, 2, 3))
    del x1105
    t3new_abaaba += einsum(x1106, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1106, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x1106
    x1107 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], nocc[1]), dtype=types[float])
    x1107 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x801, (4, 5, 0, 6), (4, 1, 5, 2, 3, 6)) * -1.0
    del x801
    x1108 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1108 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1107, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1107
    t3new_abaaba += einsum(x1108, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1108, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1108
    x1109 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], nocc[1]), dtype=types[float])
    x1109 += einsum(t2.abab[np.ix_(sOa,sOb,sVa,svb)], (0, 1, 2, 3), x859, (4, 5, 6, 3), (0, 4, 1, 2, 5, 6))
    del x859
    x1110 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1110 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1109, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1109
    t3new_abaaba += einsum(x1110, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1110, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1110, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_abaaba += einsum(x1110, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1110
    x1111 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], nocc[1]), dtype=types[float])
    x1111 += einsum(t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (0, 1, 2, 3), x794, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x794
    x1112 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1112 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1111, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1111
    t3new_abaaba += einsum(x1112, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1112, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    del x1112
    x1113 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1113 += einsum(t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (0, 1, 2, 3), x824, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x824
    t3new_abaaba += einsum(x1113, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1113, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1113
    x1114 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1114 += einsum(t2.abab[np.ix_(soa,sOb,sVa,sVb)], (0, 1, 2, 3), x466, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    del x466
    t3new_abaaba += einsum(x1114, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_abaaba += einsum(x1114, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    del x1114
    x1115 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], nocc[0], nvir[0]), dtype=types[float])
    x1115 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), t2.abab[np.ix_(soa,sOb,sVa,sVb)], (2, 3, 4, 5), v.aaaa.ovov, (6, 7, 2, 1), (0, 3, 4, 5, 6, 7)) * -1.0
    x1116 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1116 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1115, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1115
    x1117 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1117 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1116, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1116
    t3new_abaaba += einsum(x1117, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1117, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x1117, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1117, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x1117
    x1118 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], nocc[0], nvir[0]), dtype=types[float])
    x1118 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t2.abab[np.ix_(sOa,sOb,sva,sVb)], (2, 3, 4, 5), v.aaaa.ovov, (6, 7, 0, 4), (2, 3, 1, 5, 6, 7)) * -1.0
    x1119 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1119 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1118, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1118
    x1120 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1120 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1119, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1119
    t3new_abaaba += einsum(x1120, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1120, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3))
    t3new_abaaba += einsum(x1120, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4))
    t3new_abaaba += einsum(x1120, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -1.0
    del x1120
    x1121 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], nocc[0], nvir[0]), dtype=types[float])
    x1121 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.abab[np.ix_(sOa,sob,sVa,sVb)], (2, 3, 4, 5), v.aabb.ovov, (6, 7, 3, 1), (2, 0, 4, 5, 6, 7))
    x1122 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1122 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1121, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1121
    x1123 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1123 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1122, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1122
    t3new_abaaba += einsum(x1123, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1123, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1123, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1123, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x1123
    x1124 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], nocc[0], nvir[0]), dtype=types[float])
    x1124 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.abab[np.ix_(sOa,sOb,sVa,svb)], (2, 3, 4, 5), v.aabb.ovov, (6, 7, 0, 5), (2, 3, 4, 1, 6, 7))
    x1125 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1125 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1124, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1124
    x1126 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1126 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1125, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1125
    t3new_abaaba += einsum(x1126, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3new_abaaba += einsum(x1126, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -1.0
    t3new_abaaba += einsum(x1126, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -1.0
    t3new_abaaba += einsum(x1126, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3))
    del x1126
    x1127 = np.zeros((naocc[0], naocc[1], navir[0], navir[0], nocc[1], nvir[0]), dtype=types[float])
    x1127 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.aaaa[np.ix_(soa,sOa,sVa,sVa)], (2, 3, 4, 5), v.aabb.ovov, (2, 6, 7, 1), (3, 0, 4, 5, 7, 6)) * -1.0
    x1128 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], nocc[1]), dtype=types[float])
    x1128 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x1127, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1127
    x1129 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1129 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1128, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1128
    t3new_abaaba += einsum(x1129, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_abaaba += einsum(x1129, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    del x1129
    x1130 = np.zeros((naocc[0], naocc[0], navir[0], navir[1], nocc[0], nvir[1]), dtype=types[float])
    x1130 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.aaaa[np.ix_(sOa,sOa,sva,sVa)], (2, 3, 4, 5), v.aabb.ovov, (6, 4, 0, 7), (2, 3, 5, 1, 6, 7)) * -1.0
    x1131 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[1], nocc[0]), dtype=types[float])
    x1131 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1130, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x1130
    x1132 = np.zeros((naocc[0], naocc[0], naocc[1], navir[0], navir[0], navir[1]), dtype=types[float])
    x1132 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1131, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1131
    t3new_abaaba += einsum(x1132, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_abaaba += einsum(x1132, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    del x1132
    x1133 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1133 += einsum(f.bb.OO, (0, 1), t3.bbbbbb, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    t3new_bbbbbb = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    t3new_bbbbbb += einsum(x1133, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1133, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1133, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    del x1133
    x1134 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1134 += einsum(f.bb.VV, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    t3new_bbbbbb += einsum(x1134, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1134, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1134, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x1134
    x1135 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1135 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), v.bbbb.oOOV, (0, 4, 5, 6), (1, 5, 4, 2, 3, 6)) * -1.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1135, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    del x1135
    x1136 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1136 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.vVOV, (2, 4, 5, 6), (0, 1, 5, 3, 6, 4)) * -1.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1136, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    del x1136
    x1137 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1137 += einsum(v.aabb.OVOV, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 1, 7), (4, 5, 2, 6, 7, 3))
    t3new_bbbbbb += einsum(x1137, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1137, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1137, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1137, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1137, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1137, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1137, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1137, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1137, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    del x1137
    x1138 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1138 += einsum(v.bbbb.OOOO, (0, 1, 2, 3), t3.bbbbbb, (4, 3, 1, 5, 6, 7), (4, 0, 2, 5, 6, 7)) * -1.0
    t3new_bbbbbb += einsum(x1138, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1138, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1138, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    del x1138
    x1139 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1139 += einsum(v.bbbb.OOVV, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    t3new_bbbbbb += einsum(x1139, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1139, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1139, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -6.0
    t3new_bbbbbb += einsum(x1139, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1139, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1139, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -6.0
    t3new_bbbbbb += einsum(x1139, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1139, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1139, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 6.0
    del x1139
    x1140 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1140 += einsum(v.bbbb.OVOV, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 2, 6, 7, 3), (4, 5, 0, 6, 7, 1))
    t3new_bbbbbb += einsum(x1140, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1140, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1140, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 6.0
    t3new_bbbbbb += einsum(x1140, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1140, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1140, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 6.0
    t3new_bbbbbb += einsum(x1140, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1140, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1140, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -6.0
    del x1140
    x1141 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1141 += einsum(v.bbbb.VVVV, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 6, 7, 3, 1), (4, 5, 6, 7, 0, 2)) * -1.0
    t3new_bbbbbb += einsum(x1141, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1141, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1141, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    del x1141
    x1142 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1142 += einsum(x489, (0, 1), t3.bbbbbb, (2, 3, 0, 4, 5, 6), (1, 2, 3, 4, 5, 6))
    del x489
    t3new_bbbbbb += einsum(x1142, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1142, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1142, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    del x1142
    x1143 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1143 += einsum(x491, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x491
    t3new_bbbbbb += einsum(x1143, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1143, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1143, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x1143
    x1144 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1144 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x497, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    del x497
    t3new_bbbbbb += einsum(x1144, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1144, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1144, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1144, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1144, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1144, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1144, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x1144, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x1144, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    del x1144
    x1145 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1145 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x510, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x510
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1145, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x1145
    x1146 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x1146 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), v.bbbb.oOoO, (4, 5, 0, 6), (1, 5, 6, 2, 3, 4)) * -1.0
    x1147 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1147 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1146, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1146
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1147, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x1147
    x1148 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x1148 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.oOvV, (4, 5, 2, 6), (0, 1, 5, 3, 6, 4)) * -1.0
    x1149 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1149 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1148, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1148
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1149, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x1149
    x1150 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1150 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x520, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x520
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1150, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    del x1150
    x1151 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x1151 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovOV, (4, 2, 5, 6), (0, 1, 5, 3, 6, 4)) * -1.0
    x1152 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1152 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1151, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1151
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1152, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    del x1152
    x1153 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], navir[1], nvir[1]), dtype=types[float])
    x1153 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.vVvV, (4, 5, 2, 6), (0, 1, 3, 5, 6, 4)) * -1.0
    x1154 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1154 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1153, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1153
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1154, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    del x1154
    x1155 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1155 += einsum(x540, (0, 1), t3.bbbbbb, (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6))
    del x540
    t3new_bbbbbb += einsum(x1155, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1155, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1155, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -6.0
    del x1155
    x1156 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1156 += einsum(x544, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0))
    del x544
    t3new_bbbbbb += einsum(x1156, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1156, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1156, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -6.0
    del x1156
    x1157 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1157 += einsum(x546, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 2, 7), (1, 4, 5, 6, 7, 3))
    del x546
    t3new_bbbbbb += einsum(x1157, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1157, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1157, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1157, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1157, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1157, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1157, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1157, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1157, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    del x1157
    x1158 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1158 += einsum(x548, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 2, 7), (4, 5, 1, 3, 6, 7))
    del x548
    t3new_bbbbbb += einsum(x1158, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1158, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1158, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1158, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1158, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1158, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1158, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1158, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1158, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    del x1158
    x1159 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1159 += einsum(x552, (0, 1, 2, 3), t3.bbbbbb, (4, 3, 2, 5, 6, 7), (0, 4, 1, 5, 6, 7)) * -1.0
    del x552
    t3new_bbbbbb += einsum(x1159, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1159, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1159, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1159, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1159, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1159, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 6.0
    del x1159
    x1160 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1160 += einsum(x554, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 0, 6, 7, 3), (4, 5, 1, 2, 6, 7))
    del x554
    t3new_bbbbbb += einsum(x1160, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1160, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1160, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1160, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1160, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1160, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1160, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1160, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1160, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x1160
    x1161 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1161 += einsum(x556, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (4, 5, 0, 2, 6, 7))
    del x556
    t3new_bbbbbb += einsum(x1161, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1161, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1161, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1161, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1161, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1161, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1161, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1161, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1161, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x1161
    x1162 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1162 += einsum(x560, (0, 1), t3.bbbbbb, (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6))
    del x560
    t3new_bbbbbb += einsum(x1162, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1162, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1162, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    del x1162
    x1163 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1163 += einsum(x562, (0, 1), t3.bbbbbb, (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6))
    del x562
    t3new_bbbbbb += einsum(x1163, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1163, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1163, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -6.0
    del x1163
    x1164 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1164 += einsum(x564, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (0, 4, 5, 6, 7, 2))
    del x564
    t3new_bbbbbb += einsum(x1164, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1164, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1164, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -6.0
    t3new_bbbbbb += einsum(x1164, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1164, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1164, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -6.0
    t3new_bbbbbb += einsum(x1164, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1164, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1164, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 6.0
    del x1164
    x1165 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1165 += einsum(x566, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 2), (0, 4, 5, 6, 7, 3))
    del x566
    t3new_bbbbbb += einsum(x1165, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1165, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1165, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 6.0
    t3new_bbbbbb += einsum(x1165, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1165, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1165, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 6.0
    t3new_bbbbbb += einsum(x1165, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1165, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1165, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -6.0
    del x1165
    x1166 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1166 += einsum(x568, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 6, 7, 2, 3), (4, 5, 6, 0, 7, 1))
    del x568
    t3new_bbbbbb += einsum(x1166, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1166, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -6.0
    t3new_bbbbbb += einsum(x1166, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1166, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 6.0
    t3new_bbbbbb += einsum(x1166, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1166, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -6.0
    del x1166
    x1167 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1167 += einsum(x570, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0))
    del x570
    t3new_bbbbbb += einsum(x1167, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1167, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1167, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -6.0
    del x1167
    x1168 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1168 += einsum(x572, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x572
    t3new_bbbbbb += einsum(x1168, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1168, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1168, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 6.0
    del x1168
    x1169 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1169 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x584, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    del x584
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1169, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    del x1169
    x1170 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1170 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x594, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x594
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1170, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    del x1170
    x1171 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1171 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x598, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    del x598
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1171, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 4.0
    del x1171
    x1172 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1172 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x602, (4, 5, 6, 0), (1, 4, 5, 2, 3, 6)) * -1.0
    del x602
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1172, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 4.0
    del x1172
    x1173 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1173 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x606, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x606
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1173, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x1173
    x1174 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1174 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x614, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x614
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1174, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    del x1174
    x1175 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1175 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x620, (4, 5, 6, 0), (5, 4, 1, 2, 3, 6))
    del x620
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1175, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    del x1175
    x1176 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1176 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x616, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x616
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1176, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 4.0
    del x1176
    x1177 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1177 += einsum(x640, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 2, 7), (1, 4, 5, 3, 6, 7))
    del x640
    t3new_bbbbbb += einsum(x1177, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.00000000000002
    t3new_bbbbbb += einsum(x1177, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -2.00000000000002
    t3new_bbbbbb += einsum(x1177, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.00000000000002
    t3new_bbbbbb += einsum(x1177, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 2.00000000000002
    t3new_bbbbbb += einsum(x1177, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.00000000000002
    t3new_bbbbbb += einsum(x1177, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.00000000000002
    t3new_bbbbbb += einsum(x1177, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.00000000000002
    t3new_bbbbbb += einsum(x1177, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 2.00000000000002
    t3new_bbbbbb += einsum(x1177, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.00000000000002
    del x1177
    x1178 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1178 += einsum(x642, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 2, 7), (1, 4, 5, 3, 6, 7))
    del x642
    t3new_bbbbbb += einsum(x1178, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.00000000000002
    t3new_bbbbbb += einsum(x1178, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.00000000000002
    t3new_bbbbbb += einsum(x1178, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.00000000000002
    t3new_bbbbbb += einsum(x1178, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.00000000000002
    t3new_bbbbbb += einsum(x1178, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.00000000000002
    t3new_bbbbbb += einsum(x1178, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.00000000000002
    t3new_bbbbbb += einsum(x1178, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.00000000000002
    t3new_bbbbbb += einsum(x1178, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.00000000000002
    t3new_bbbbbb += einsum(x1178, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.00000000000002
    del x1178
    x1179 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1179 += einsum(x656, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x656
    t3new_bbbbbb += einsum(x1179, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 6.0000000000000595
    t3new_bbbbbb += einsum(x1179, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -6.0000000000000595
    t3new_bbbbbb += einsum(x1179, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 6.0000000000000595
    t3new_bbbbbb += einsum(x1179, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 6.0000000000000595
    t3new_bbbbbb += einsum(x1179, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -6.0000000000000595
    t3new_bbbbbb += einsum(x1179, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 6.0000000000000595
    t3new_bbbbbb += einsum(x1179, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -6.0000000000000595
    t3new_bbbbbb += einsum(x1179, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 6.0000000000000595
    t3new_bbbbbb += einsum(x1179, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -6.0000000000000595
    del x1179
    x1180 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1180 += einsum(x658, (0, 1), t3.bbbbbb, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    del x658
    t3new_bbbbbb += einsum(x1180, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1180, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1180, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 6.0
    del x1180
    x1181 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1181 += einsum(x664, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    del x664
    t3new_bbbbbb += einsum(x1181, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1181, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1181, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x1181
    x1182 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1182 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x672, (1, 6, 7, 2, 4, 8), (7, 6, 0, 8, 3, 5)) * -1.0
    del x672
    t3new_bbbbbb += einsum(x1182, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x1182, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x1182, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1182, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1182, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1182, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1182, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1182, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1182, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    del x1182
    x1183 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1183 += einsum(t3.babbab, (0, 1, 2, 3, 4, 5), x674, (1, 6, 4, 7, 8, 5), (6, 0, 2, 7, 8, 3))
    del x674
    t3new_bbbbbb += einsum(x1183, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1183, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1183, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1183, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1183, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1183, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1183, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1183, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1183, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -4.0
    del x1183
    x1184 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1184 += einsum(x676, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 2, 7), (1, 4, 5, 3, 6, 7))
    del x676
    t3new_bbbbbb += einsum(x1184, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.00000000000004
    t3new_bbbbbb += einsum(x1184, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.00000000000004
    t3new_bbbbbb += einsum(x1184, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.00000000000004
    t3new_bbbbbb += einsum(x1184, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 4.00000000000004
    t3new_bbbbbb += einsum(x1184, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -4.00000000000004
    t3new_bbbbbb += einsum(x1184, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 4.00000000000004
    t3new_bbbbbb += einsum(x1184, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -4.00000000000004
    t3new_bbbbbb += einsum(x1184, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 4.00000000000004
    t3new_bbbbbb += einsum(x1184, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -4.00000000000004
    del x1184
    x1185 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1185 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x678, (6, 7, 1, 2, 8, 5), (7, 6, 0, 8, 3, 4)) * -1.0
    del x678
    t3new_bbbbbb += einsum(x1185, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1185, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1185, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1185, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1185, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1185, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1185, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1185, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1185, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -6.0
    del x1185
    x1186 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1186 += einsum(x680, (0, 1, 2, 3), t3.bbbbbb, (4, 2, 3, 5, 6, 7), (1, 0, 4, 5, 6, 7)) * -1.0
    del x680
    t3new_bbbbbb += einsum(x1186, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 3.0
    t3new_bbbbbb += einsum(x1186, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 3.0
    t3new_bbbbbb += einsum(x1186, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -3.0
    t3new_bbbbbb += einsum(x1186, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -3.0
    t3new_bbbbbb += einsum(x1186, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -3.0
    t3new_bbbbbb += einsum(x1186, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -3.0
    del x1186
    x1187 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1187 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x682, (6, 7, 1, 2, 8, 5), (7, 6, 0, 8, 3, 4)) * -1.0
    del x682
    t3new_bbbbbb += einsum(x1187, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1187, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1187, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1187, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1187, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1187, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1187, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1187, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1187, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -6.0
    del x1187
    x1188 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1188 += einsum(t3.bbbbbb, (0, 1, 2, 3, 4, 5), x684, (6, 2, 7, 8, 4, 5), (6, 0, 1, 7, 8, 3))
    del x684
    t3new_bbbbbb += einsum(x1188, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -12.0
    t3new_bbbbbb += einsum(x1188, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 12.0
    t3new_bbbbbb += einsum(x1188, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -12.0
    t3new_bbbbbb += einsum(x1188, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -12.0
    t3new_bbbbbb += einsum(x1188, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 12.0
    t3new_bbbbbb += einsum(x1188, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -12.0
    t3new_bbbbbb += einsum(x1188, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 12.0
    t3new_bbbbbb += einsum(x1188, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -12.0
    t3new_bbbbbb += einsum(x1188, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 12.0
    del x1188
    x1189 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1189 += einsum(x686, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x686
    t3new_bbbbbb += einsum(x1189, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 12.000000000000123
    t3new_bbbbbb += einsum(x1189, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -12.000000000000123
    t3new_bbbbbb += einsum(x1189, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 12.000000000000123
    t3new_bbbbbb += einsum(x1189, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 12.000000000000123
    t3new_bbbbbb += einsum(x1189, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -12.000000000000123
    t3new_bbbbbb += einsum(x1189, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 12.000000000000123
    t3new_bbbbbb += einsum(x1189, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -12.000000000000123
    t3new_bbbbbb += einsum(x1189, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 12.000000000000123
    t3new_bbbbbb += einsum(x1189, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -12.000000000000123
    del x1189
    x1190 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1190 += einsum(x688, (0, 1), t3.bbbbbb, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    del x688
    t3new_bbbbbb += einsum(x1190, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1190, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1190, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1190, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1190, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1190, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    del x1190
    x1191 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1191 += einsum(x690, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x690
    t3new_bbbbbb += einsum(x1191, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -12.000000000000123
    t3new_bbbbbb += einsum(x1191, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 12.000000000000123
    t3new_bbbbbb += einsum(x1191, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -12.000000000000123
    t3new_bbbbbb += einsum(x1191, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -12.000000000000123
    t3new_bbbbbb += einsum(x1191, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 12.000000000000123
    t3new_bbbbbb += einsum(x1191, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -12.000000000000123
    t3new_bbbbbb += einsum(x1191, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 12.000000000000123
    t3new_bbbbbb += einsum(x1191, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -12.000000000000123
    t3new_bbbbbb += einsum(x1191, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 12.000000000000123
    del x1191
    x1192 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1192 += einsum(x693, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 6, 7, 3, 2), (4, 5, 6, 1, 0, 7))
    del x693
    t3new_bbbbbb += einsum(x1192, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -6.0
    t3new_bbbbbb += einsum(x1192, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1192, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x1192
    x1193 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1193 += einsum(x694, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    del x694
    t3new_bbbbbb += einsum(x1193, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1193, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1193, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x1193
    x1194 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1194 += einsum(x696, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    del x696
    t3new_bbbbbb += einsum(x1194, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1194, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1194, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x1194
    x1195 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], navir[1], nvir[1]), dtype=types[float])
    x1195 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (2, 3, 4, 5), v.bbbb.ovoO, (2, 6, 0, 7), (3, 7, 1, 4, 5, 6)) * -1.0
    x1196 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1196 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1195, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1195
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1196, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x1196
    x1197 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], navir[1], nvir[1]), dtype=types[float])
    x1197 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (2, 3, 4, 5), v.bbbb.ovoO, (0, 6, 2, 7), (3, 7, 1, 4, 5, 6)) * -1.0
    x1198 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1198 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1197, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1197
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1198, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    del x1198
    x1199 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1199 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x724, (4, 5, 6, 2), (0, 1, 4, 5, 6, 3)) * -1.0
    del x724
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1199, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x1199
    x1200 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], navir[1], nvir[1]), dtype=types[float])
    x1200 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (2, 3, 4, 5), v.bbbb.ovvV, (2, 1, 6, 7), (0, 3, 4, 5, 7, 6)) * -1.0
    x1201 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1201 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1200, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1200
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1201, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    del x1201
    x1202 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], navir[1], nvir[1]), dtype=types[float])
    x1202 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (2, 3, 4, 5), v.bbbb.ovvV, (0, 6, 4, 7), (2, 3, 1, 5, 7, 6)) * -1.0
    x1203 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1203 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1202, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1202
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1203, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    del x1203
    x1204 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], navir[1], nvir[1]), dtype=types[float])
    x1204 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (2, 3, 4, 5), v.bbbb.ovvV, (0, 4, 6, 7), (2, 3, 1, 5, 7, 6)) * -1.0
    x1205 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1205 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1204, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1204
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1205, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    del x1205
    x1206 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1206 += einsum(x744, (0, 1), t3.bbbbbb, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    del x744
    t3new_bbbbbb += einsum(x1206, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1206, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1206, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    del x1206
    x1207 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1207 += einsum(x750, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    del x750
    t3new_bbbbbb += einsum(x1207, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1207, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1207, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x1207
    x1208 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1208 += einsum(x752, (0, 1, 2, 3), t3.babbab, (4, 0, 5, 6, 2, 7), (1, 4, 5, 3, 6, 7))
    del x752
    t3new_bbbbbb += einsum(x1208, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1208, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1208, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1208, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1208, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1208, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1208, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1208, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1208, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    del x1208
    x1209 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1209 += einsum(x755, (0, 1, 2, 3), t3.bbbbbb, (4, 2, 3, 5, 6, 7), (1, 0, 4, 5, 6, 7)) * -1.0
    del x755
    t3new_bbbbbb += einsum(x1209, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 3.0
    t3new_bbbbbb += einsum(x1209, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 3.0
    t3new_bbbbbb += einsum(x1209, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -3.0
    t3new_bbbbbb += einsum(x1209, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -3.0
    t3new_bbbbbb += einsum(x1209, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -3.0
    t3new_bbbbbb += einsum(x1209, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -3.0
    del x1209
    x1210 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1210 += einsum(x758, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x758
    t3new_bbbbbb += einsum(x1210, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1210, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1210, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1210, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1210, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1210, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -6.0
    t3new_bbbbbb += einsum(x1210, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1210, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1210, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 6.0
    del x1210
    x1211 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1211 += einsum(x761, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 1, 6, 7, 3), (0, 4, 5, 2, 6, 7))
    del x761
    t3new_bbbbbb += einsum(x1211, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1211, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1211, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1211, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1211, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1211, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 6.0
    t3new_bbbbbb += einsum(x1211, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1211, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1211, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -6.0
    del x1211
    x1212 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1212 += einsum(x764, (0, 1), t3.bbbbbb, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    del x764
    t3new_bbbbbb += einsum(x1212, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1212, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -6.0
    t3new_bbbbbb += einsum(x1212, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 6.0
    del x1212
    x1213 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1213 += einsum(x767, (0, 1), t3.bbbbbb, (2, 3, 1, 4, 5, 6), (0, 2, 3, 4, 5, 6))
    del x767
    t3new_bbbbbb += einsum(x1213, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1213, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1213, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -6.0
    del x1213
    x1214 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1214 += einsum(x770, (0, 1, 2, 3), t3.bbbbbb, (4, 5, 6, 7, 3, 2), (4, 5, 6, 1, 0, 7))
    del x770
    t3new_bbbbbb += einsum(x1214, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -6.0
    t3new_bbbbbb += einsum(x1214, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 6.0
    t3new_bbbbbb += einsum(x1214, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x1214
    x1215 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1215 += einsum(x772, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    del x772
    t3new_bbbbbb += einsum(x1215, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 6.0
    t3new_bbbbbb += einsum(x1215, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -6.0
    t3new_bbbbbb += einsum(x1215, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 6.0
    del x1215
    x1216 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1216 += einsum(x775, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    del x775
    t3new_bbbbbb += einsum(x1216, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -6.0
    t3new_bbbbbb += einsum(x1216, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 6.0
    t3new_bbbbbb += einsum(x1216, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -6.0
    del x1216
    x1217 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1217 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x809, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    del x809
    t3new_bbbbbb += einsum(x1217, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1217, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1217, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1217, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1217, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1217, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1217, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x1217, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x1217, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    del x1217
    x1218 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1218 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x814, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x814
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1218, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    del x1218
    x1219 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x1219 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x813, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x813
    x1220 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1220 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1219, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1219
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1220, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 2.0
    del x1220
    x1221 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1221 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x827, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x827
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1221, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 4.0
    del x1221
    x1222 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1222 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x833, (4, 5, 6, 0), (4, 1, 5, 2, 3, 6)) * -1.0
    del x833
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (2, 0, 1, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1222, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 4.0
    del x1222
    x1223 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], nocc[1], nocc[1]), dtype=types[float])
    x1223 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (2, 3, 4, 5), v.bbbb.ovov, (6, 1, 7, 4), (0, 2, 3, 5, 6, 7)) * -1.0
    x1224 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1224 += einsum(t2.bbbb[np.ix_(sob,sob,sVb,sVb)], (0, 1, 2, 3), x1223, (4, 5, 6, 7, 0, 1), (4, 5, 6, 7, 2, 3))
    del x1223
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1224, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    del x1224
    x1225 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x1225 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x826, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x826
    x1226 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1226 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1225, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1225
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * -4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * -4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * 4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1226, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    del x1226
    x1227 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x1227 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x846, (4, 5, 0, 6), (4, 5, 1, 2, 3, 6))
    del x846
    x1228 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1228 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1227, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1227
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1228, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    del x1228
    x1229 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x1229 += einsum(t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (0, 1, 2, 3), x832, (4, 5, 6, 2), (0, 1, 4, 3, 5, 6)) * -1.0
    del x832
    x1230 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1230 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1229, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1229
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 4, 5)) * 4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)) * -4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 3, 5)) * -4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 3, 5)) * 4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (2, 1, 0, 4, 5, 3)) * 4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3)) * -4.0
    t3new_bbbbbb += einsum(x1230, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 4.0
    del x1230
    x1231 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1231 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x855, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    del x855
    t3new_bbbbbb += einsum(x1231, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1231, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1231, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1231, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1231, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * 4.0
    t3new_bbbbbb += einsum(x1231, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1231, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x1231, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x1231, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * -4.0
    del x1231
    x1232 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1232 += einsum(t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (0, 1, 2, 3), x857, (4, 5, 6, 0), (5, 4, 1, 6, 2, 3))
    del x857
    t3new_bbbbbb += einsum(x1232, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x1232, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x1232, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1232, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 4.0
    t3new_bbbbbb += einsum(x1232, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -4.0
    t3new_bbbbbb += einsum(x1232, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -4.0
    t3new_bbbbbb += einsum(x1232, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 4.0
    t3new_bbbbbb += einsum(x1232, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -4.0
    t3new_bbbbbb += einsum(x1232, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 4.0
    del x1232
    x1233 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], nocc[1], nvir[1]), dtype=types[float])
    x1233 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), t2.bbbb[np.ix_(sob,sOb,sVb,sVb)], (2, 3, 4, 5), v.bbbb.ovov, (6, 7, 2, 1), (0, 3, 4, 5, 6, 7)) * -1.0
    x1234 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x1234 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1233, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1233
    x1235 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1235 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1234, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1234
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (0, 1, 2, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (2, 0, 1, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (2, 1, 0, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (2, 0, 1, 5, 4, 3)) * 2.0
    t3new_bbbbbb += einsum(x1235, (0, 1, 2, 3, 4, 5), (2, 1, 0, 5, 4, 3)) * -2.0
    del x1235
    x1236 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], nocc[1], nvir[1]), dtype=types[float])
    x1236 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t2.bbbb[np.ix_(sOb,sOb,svb,sVb)], (2, 3, 4, 5), v.bbbb.ovov, (6, 7, 0, 4), (2, 3, 1, 5, 6, 7)) * -1.0
    x1237 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], nocc[1]), dtype=types[float])
    x1237 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1236, (2, 3, 4, 5, 6, 1), (0, 2, 3, 4, 5, 6))
    del x1236
    x1238 = np.zeros((naocc[1], naocc[1], naocc[1], navir[1], navir[1], navir[1]), dtype=types[float])
    x1238 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1237, (2, 3, 4, 5, 6, 0), (2, 3, 4, 1, 5, 6))
    del x1237
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (0, 2, 1, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 4, 5)) * 2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5)) * -2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (1, 0, 2, 3, 5, 4)) * -2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 5, 3)) * 2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 3, 4)) * 2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (1, 0, 2, 5, 4, 3)) * -2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 4, 5)) * -2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 3, 5)) * 2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (1, 2, 0, 3, 5, 4)) * 2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (1, 2, 0, 4, 5, 3)) * -2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 3, 4)) * -2.0
    t3new_bbbbbb += einsum(x1238, (0, 1, 2, 3, 4, 5), (1, 2, 0, 5, 4, 3)) * 2.0
    del x1238

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.aaaa = t2new_aaaa
    t2new.abab = t2new_abab
    t2new.bbbb = t2new_bbbb
    t3new.aaaaaa = t3new_aaaaaa
    t3new.abaaba = t3new_abaaba
    t3new.babbab = t3new_babbab
    t3new.bbbbbb = t3new_bbbbbb

    return {"t1new": t1new, "t2new": t2new, "t3new": t3new}

