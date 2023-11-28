# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # energy
    e_cc = 0
    e_cc += einsum(t2.bbbb, (0, 1, 2, 3), v.bbbb.ovov, (0, 3, 1, 2), ()) * -1.0
    e_cc += einsum(t2.abab, (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 3), ())
    e_cc += einsum(t2.aaaa, (0, 1, 2, 3), v.aaaa.ovov, (0, 2, 1, 3), ())
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1))
    x0 += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3)) * -1.0
    x1 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x1 += einsum(f.bb.ov, (0, 1), (0, 1)) * 2.0
    x1 += einsum(t1.aa, (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3)) * 2.0
    x1 += einsum(t1.bb, (0, 1), x0, (0, 2, 1, 3), (2, 3)) * -1.0
    del x0
    e_cc += einsum(t1.bb, (0, 1), x1, (0, 1), ()) * 0.5
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    x2 += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    x3 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x3 += einsum(f.aa.ov, (0, 1), (0, 1)) * 2.0
    x3 += einsum(t1.aa, (0, 1), x2, (0, 2, 3, 1), (2, 3)) * -1.0
    del x2
    e_cc += einsum(t1.aa, (0, 1), x3, (0, 1), ()) * 0.5
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
    sOfa = np.ones((naocc[0],), dtype=bool)
    sVfa = np.ones((navir[0],), dtype=bool)
    sOfb = np.ones((naocc[1],), dtype=bool)
    sVfb = np.ones((navir[1],), dtype=bool)

    # T amplitudes
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    t1new_bb[np.ix_(sob,svb)] += einsum(f.bb.ov, (0, 1), (0, 1))
    t1new_bb[np.ix_(sob,svb)] += einsum(f.bb.oo, (0, 1), t1.bb[np.ix_(sob,svb)], (1, 2), (0, 2)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(f.bb.vv, (0, 1), t1.bb[np.ix_(sob,svb)], (2, 1), (2, 0))
    t1new_bb[np.ix_(sob,svb)] += einsum(f.aa.ov, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (0, 2, 1, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(f.bb.ov, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oovv, (2, 0, 3, 1), (2, 3)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 1), (4, 3)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovvv, (0, 2, 4, 3), (1, 4))
    t1new_bb[np.ix_(sob,svb)] += einsum(v.aaaa.ovOV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 4, 2, 1, 5, 3), (4, 5))
    t1new_bb[np.ix_(sob,svb)] += einsum(v.aabb.ovOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 2, 5, 1, 3), (4, 5)) * 2.0
    t1new_bb[np.ix_(sob,svb)] += einsum(v.bbbb.ovOV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 0, 2, 5, 1, 3), (4, 5)) * 3.0
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    t1new_aa[np.ix_(soa,sva)] += einsum(f.aa.ov, (0, 1), (0, 1))
    t1new_aa[np.ix_(soa,sva)] += einsum(f.aa.oo, (0, 1), t1.aa[np.ix_(soa,sva)], (1, 2), (0, 2)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(f.aa.vv, (0, 1), t1.aa[np.ix_(soa,sva)], (2, 1), (2, 0))
    t1new_aa[np.ix_(soa,sva)] += einsum(f.bb.ov, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 0, 3, 1), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(f.aa.ov, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oovv, (2, 0, 3, 1), (2, 3)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 0, 1, 3), (4, 2)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vvov, (4, 2, 1, 3), (0, 4))
    t1new_aa[np.ix_(soa,sva)] += einsum(v.aabb.OVov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 0, 5, 3, 1), (4, 5)) * 2.0
    t1new_aa[np.ix_(soa,sva)] += einsum(v.bbbb.ovOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (0, 4, 2, 1, 5, 3), (4, 5))
    t1new_aa[np.ix_(soa,sva)] += einsum(v.aaaa.ovOV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 0, 2, 5, 1, 3), (4, 5)) * 3.0
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(v.aaaa.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    t2new_aaaa[np.ix_(soa,sOa,sva,sVa)] += einsum(f.bb.ov, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(f.aa.OV, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(v.aabb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(f.aa.oo, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(f.bb.oo, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(f.aa.vv, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(f.bb.vv, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ooov, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ovoo, (2, 3, 4, 0), (2, 4, 3, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(f.aa.OV, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(f.bb.OV, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 0, 4, 5, 1), (3, 2, 5, 4)) * 2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.oooo, (4, 0, 5, 1), (4, 5, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.oovv, (4, 1, 5, 3), (0, 4, 2, 5)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.oovv, (4, 0, 5, 3), (4, 1, 2, 5)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.oovv, (4, 0, 5, 2), (4, 1, 5, 3)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vvoo, (4, 2, 5, 1), (0, 5, 4, 3)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aabb.ovov, (1, 3, 4, 5), (0, 4, 2, 5)) * 2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(v.aaaa.ooOV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 4, 2, 5, 6, 3), (0, 4, 5, 6)) * -2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(v.aabb.OVoo, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 0, 5, 6, 1), (4, 2, 5, 6)) * -2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(v.aaaa.vOvV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 0, 6, 3), (4, 5, 2, 6)) * -2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(v.aabb.OVvv, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 3, 1), (4, 5, 6, 2)) * 2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(v.aabb.ooOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 2, 5, 6, 3), (0, 4, 6, 5)) * -2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(v.bbbb.ooOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 4, 2, 5, 6, 3), (4, 0, 6, 5)) * -2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(v.bbbb.vOvV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 0, 6, 3), (5, 4, 6, 2)) * -2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(v.aabb.vvOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 2, 6, 1, 3), (5, 4, 0, 6)) * 2.0
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 1, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(v.bbbb.ovov, (0, 1, 2, 3), (0, 2, 3, 1)) * -1.0
    t2new_bbbb[np.ix_(sob,sOb,svb,sVb)] += einsum(f.aa.ov, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(f.bb.OV, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    t3new_aaaaaa = np.zeros((nocc[0], nocc[0], naocc[0], nvir[0], nvir[0], navir[0]), dtype=np.float64)
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(f.aa.OO, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(f.aa.VV, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 6.0
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sVa,sVfa)] += einsum(v.aabb.OVov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(v.aaaa.oooo, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (3, 1, 4, 5, 6, 7), (0, 2, 4, 5, 6, 7)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(v.aaaa.OVOV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 2, 6, 7, 3), (4, 5, 0, 6, 7, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(v.aaaa.OOVV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(v.aaaa.vvvv, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 6, 1, 3, 7), (4, 5, 6, 0, 2, 7)) * 6.0
    t3new_babbab = np.zeros((nocc[1], nocc[0], naocc[1], nvir[1], nvir[0], navir[1]), dtype=np.float64)
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(f.bb.oo, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(f.aa.oo, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(f.bb.OO, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(f.bb.vv, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 1, 5, 6), (2, 3, 4, 0, 5, 6)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(f.aa.vv, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(f.bb.VV, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.bbbb.oovO, (4, 1, 5, 6), (4, 0, 6, 5, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.oovO, (4, 0, 5, 6), (1, 4, 6, 5, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ooOV, (4, 1, 5, 6), (4, 0, 5, 3, 2, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ooOV, (4, 0, 5, 6), (1, 4, 5, 3, 2, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.bbbb.ovoO, (4, 5, 1, 6), (4, 0, 6, 5, 2, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.oOoV, (1, 4, 5, 6), (5, 0, 4, 3, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.oooV, (4, 0, 5, 6), (5, 4, 1, 3, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ooov, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (4, 5, 3, 6), (4, 0, 1, 5, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.bbbb.oVvv, (4, 5, 6, 3), (4, 0, 1, 6, 2, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.vvvO, (4, 2, 5, 6), (1, 0, 6, 5, 4, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.vvOV, (4, 3, 5, 6), (1, 0, 5, 4, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vvOV, (4, 2, 5, 6), (1, 0, 5, 3, 4, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.vOvV, (4, 5, 3, 6), (1, 0, 5, 4, 2, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.vvoV, (4, 2, 5, 6), (5, 0, 1, 3, 4, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.aabb.ovoO, (4, 5, 1, 6), (0, 4, 6, 2, 5, 3)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.ovoo, (4, 5, 6, 0), (6, 4, 1, 2, 5, 3)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.aabb.ovvV, (4, 5, 3, 6), (0, 4, 1, 2, 5, 6)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aabb.OVOV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 1), (5, 4, 2, 7, 6, 3)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aabb.OVvO, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 1), (5, 4, 3, 2, 6, 7)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aabb.OVoV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 7, 6, 3)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aabb.OVov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.bbbb.ooOO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 4, 3, 5, 6, 7), (0, 4, 2, 5, 6, 7)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aabb.ooOO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 3, 5, 6, 7), (4, 0, 2, 5, 6, 7)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aabb.oooo, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 1, 4, 5, 6, 7), (2, 0, 4, 5, 6, 7)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.bbbb.oovv, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 4, 5, 3, 6, 7), (0, 4, 5, 2, 6, 7)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aabb.oovv, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 3, 6, 7), (4, 0, 5, 2, 6, 7)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aaaa.oovv, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.bbbb.ooVV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 4, 5, 6, 7, 3), (0, 4, 5, 6, 7, 2)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aabb.ooVV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 7, 3), (4, 0, 5, 6, 7, 2)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.bbbb.oVoV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 4, 5, 6, 7, 3), (0, 4, 5, 6, 7, 1)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.bbbb.vOvO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 3, 2, 6, 7), (4, 5, 1, 0, 6, 7)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.bbbb.OVOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 2, 6, 7, 3), (4, 5, 0, 6, 7, 1)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.bbbb.vvOO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 3, 1, 6, 7), (4, 5, 2, 0, 6, 7)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.bbbb.OOVV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aabb.vvOO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 3, 6, 1, 7), (4, 5, 2, 6, 0, 7)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aabb.vvoo, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 5, 6, 1, 7), (2, 4, 5, 6, 0, 7)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aabb.vvvv, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 3, 1, 7), (4, 5, 6, 2, 0, 7)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.bbbb.vvVV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 7, 2)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(v.aabb.vvVV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 7, 1, 3), (4, 5, 6, 7, 0, 2)) * 2.0
    t3new_abaaba = np.zeros((nocc[0], nocc[1], naocc[0], nvir[0], nvir[1], navir[0]), dtype=np.float64)
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(f.aa.oo, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(f.bb.oo, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(f.aa.OO, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(f.aa.vv, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 1, 5, 6), (2, 3, 4, 0, 5, 6)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(f.bb.vv, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(f.aa.VV, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ooOV, (4, 0, 5, 6), (4, 1, 5, 2, 3, 6)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.oOoV, (0, 4, 5, 6), (5, 1, 4, 2, 3, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.vvOV, (4, 2, 5, 6), (0, 1, 5, 4, 3, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.vOvV, (4, 5, 2, 6), (0, 1, 5, 4, 3, 6)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.OVoo, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.OVvv, (4, 5, 6, 3), (0, 1, 4, 2, 6, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aaaa.oovO, (4, 0, 5, 6), (4, 1, 6, 5, 3, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aaaa.ovoO, (4, 5, 0, 6), (4, 1, 6, 5, 3, 2)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.vOoo, (4, 5, 6, 1), (0, 6, 5, 4, 3, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.vOvv, (4, 5, 6, 3), (0, 1, 5, 4, 6, 2)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ovvV, (4, 5, 2, 6), (4, 1, 0, 5, 3, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.oVoo, (4, 5, 6, 1), (4, 6, 0, 2, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.oVvv, (4, 5, 6, 3), (4, 1, 0, 2, 6, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.oVvv, (4, 5, 6, 2), (4, 1, 0, 6, 3, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ovoo, (4, 5, 6, 1), (4, 6, 0, 5, 3, 2)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aabb.oOov, (1, 4, 5, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ooov, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aabb.vVov, (3, 4, 5, 6), (0, 5, 1, 2, 6, 4)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aaaa.ooOO, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 4, 3, 5, 6, 7), (0, 4, 2, 5, 6, 7)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aabb.oooo, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 3, 4, 5, 6, 7), (0, 2, 4, 5, 6, 7)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aaaa.oovv, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 4, 5, 3, 6, 7), (0, 4, 5, 2, 6, 7)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aabb.oovv, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 4, 5, 6, 3, 7), (0, 4, 5, 6, 2, 7)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.bbbb.oovv, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aaaa.ooVV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 4, 5, 6, 7, 3), (0, 4, 5, 6, 7, 2)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aaaa.oVoV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 4, 5, 6, 7, 3), (0, 4, 5, 6, 7, 1)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aaaa.vOvO, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 3, 2, 6, 7), (4, 5, 1, 0, 6, 7)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aaaa.OVOV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 2, 6, 7, 3), (4, 5, 0, 6, 7, 1)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aabb.OOoo, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aaaa.vvOO, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 3, 1, 6, 7), (4, 5, 2, 0, 6, 7)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aabb.OOvv, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 3, 7), (4, 5, 0, 6, 2, 7)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aaaa.OOVV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aabb.vvoo, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 5, 1, 6, 7), (4, 2, 5, 0, 6, 7)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aabb.vvvv, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 1, 3, 7), (4, 5, 6, 0, 2, 7)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aaaa.vvVV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 7, 2)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aabb.VVoo, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aabb.VVvv, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 7, 3, 1), (4, 5, 6, 7, 2, 0)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aabb.OVOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 2, 6, 7, 3), (5, 4, 0, 7, 6, 1)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aabb.vOOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sVa,sVfb)], (4, 5, 2, 6, 7, 3), (5, 4, 1, 0, 6, 7)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aabb.oVOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sva,sVfb)], (4, 5, 2, 6, 7, 3), (0, 4, 5, 7, 6, 1)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(v.aabb.ovOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sVa,sVfb)], (4, 5, 2, 6, 7, 3), (0, 4, 5, 1, 6, 7)) * 2.0
    t3new_bbbbbb = np.zeros((nocc[1], nocc[1], naocc[1], nvir[1], nvir[1], navir[1]), dtype=np.float64)
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(f.bb.OO, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(f.bb.VV, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,sVb,sVfb)] += einsum(v.aabb.ovOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 5, 6, 1, 7), (4, 5, 2, 6, 7, 3)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(v.bbbb.oooo, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (1, 3, 4, 5, 6, 7), (0, 2, 4, 5, 6, 7)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(v.bbbb.OVOV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 2, 6, 7, 3), (4, 5, 0, 6, 7, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(v.bbbb.OOVV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(v.bbbb.vvvv, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 6, 3, 1, 7), (4, 5, 6, 0, 2, 7)) * -6.0
    x0 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x0 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovov, (0, 1, 2, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(x0, (0, 1), (0, 1))
    t1new_bb[np.ix_(sob,svb)] += einsum(x0, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x0, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 0, 3, 1), (2, 3))
    t2new_aaaa[np.ix_(soa,sOa,sva,sVa)] += einsum(x0, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    x1 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x1 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovov, (2, 3, 0, 1), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(x1, (0, 1), (0, 1))
    t1new_bb[np.ix_(sob,svb)] += einsum(x1, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 0, 3, 1), (2, 3)) * 2.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x1, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 0, 3, 1), (2, 3))
    t2new_aaaa[np.ix_(soa,sOa,sva,sVa)] += einsum(x1, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    x2 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x2 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 0, 1, 3), (4, 2))
    t1new_bb[np.ix_(sob,svb)] += einsum(x2, (0, 1), (0, 1)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(x2, (0, 1), (0, 1)) * -1.0
    del x2
    x3 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x3 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (1, 2, 4, 3), (0, 4))
    t1new_bb[np.ix_(sob,svb)] += einsum(x3, (0, 1), (0, 1)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(x3, (0, 1), (0, 1)) * -1.0
    del x3
    x4 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x4 += einsum(f.bb.ov, (0, 1), t1.bb[np.ix_(sob,svb)], (2, 1), (0, 2))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x4, (0, 2), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x4, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 0, 3, 4), (2, 1, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x4, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (0, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x4, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -2.0
    x5 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x5 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x5, (2, 1), (0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x5, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x5, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 1, 5, 6), (2, 3, 4, 0, 5, 6)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x5, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    x6 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x6 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovoo, (0, 1, 2, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x6, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x6, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x6, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x6, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x7 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x7 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ooov, (2, 0, 3, 1), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x7, (2, 0), (2, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x7, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x7, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x7, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * 2.0
    x8 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x8 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ooov, (2, 3, 0, 1), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x8, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x8, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x8, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x8, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x9 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x9 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvv, (0, 2, 3, 1), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x9, (1, 2), (0, 2)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x9, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 0), (2, 3, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x9, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x9, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -2.0
    x10 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x10 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x10, (2, 1), (0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x10, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x10, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 1, 5, 6), (2, 3, 4, 0, 5, 6)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x10, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    x11 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x11 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovov, (2, 3, 0, 1), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(x11, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (0, 2, 1, 3), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(x11, (0, 1), (0, 1))
    t1new_aa[np.ix_(soa,sva)] += einsum(x11, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 0, 3, 1), (2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sOb,svb,sVb)] += einsum(x11, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    x12 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x12 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovov, (2, 1, 0, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(x12, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (0, 2, 1, 3), (2, 3)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x12, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 0, 3, 1), (2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sOb,svb,sVb)] += einsum(x12, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * -2.0
    x13 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x13 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ovov, (2, 3, 4, 1), (2, 0, 4, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x13, (0, 4, 1, 2), (4, 3)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x13, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x13, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x13, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2)) * -1.0
    x14 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x14 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 3), (1, 4))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x14, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x14, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x14, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x14, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x15 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x15 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ovov, (2, 3, 0, 1), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(x15, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (0, 2, 1, 3), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(x15, (0, 1), (0, 1))
    t1new_aa[np.ix_(soa,sva)] += einsum(x15, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 0, 3, 1), (2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sOb,svb,sVb)] += einsum(x15, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * 2.0
    x16 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x16 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x17 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x17 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x16, (4, 0, 1, 3), (4, 2)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(x17, (0, 1), (0, 1)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(x17, (0, 1), (0, 1)) * -1.0
    del x17
    x18 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x18 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x19 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x19 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x18, (2, 0), (2, 1))
    t1new_bb[np.ix_(sob,svb)] += einsum(x19, (0, 1), (0, 1)) * -1.0
    t1new_bb[np.ix_(sob,svb)] += einsum(x19, (0, 1), (0, 1)) * -1.0
    del x19
    x20 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x20 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovov, (2, 1, 0, 3), (2, 3))
    t1new_bb[np.ix_(sob,svb)] += einsum(x20, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 0, 3, 1), (2, 3)) * -2.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x20, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 0, 3, 1), (2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,sOa,sva,sVa)] += einsum(x20, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 0, 3, 4, 1, 5), (2, 3, 4, 5)) * -2.0
    x21 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x21 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x0, (2, 1), (0, 2))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x21, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x21, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x21, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x21, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x22 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x22 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x20, (2, 1), (0, 2))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x22, (2, 0), (2, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x22, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x22, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x22, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * 2.0
    x23 = np.zeros((nocc[1], nocc[1]), dtype=np.float64)
    x23 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1, (2, 1), (0, 2))
    t1new_bb[np.ix_(sob,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x23, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x23, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x23, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x23, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    x24 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x24 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ooov, (4, 0, 1, 3), (4, 2))
    t1new_aa[np.ix_(soa,sva)] += einsum(x24, (0, 1), (0, 1)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x24, (0, 1), (0, 1)) * -1.0
    del x24
    x25 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x25 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvv, (1, 2, 4, 3), (0, 4))
    t1new_aa[np.ix_(soa,sva)] += einsum(x25, (0, 1), (0, 1)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x25, (0, 1), (0, 1)) * -1.0
    del x25
    x26 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x26 += einsum(f.aa.ov, (0, 1), t1.aa[np.ix_(soa,sva)], (2, 1), (0, 2))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x26, (0, 2), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x26, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (0, 2, 3, 4), (1, 2, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x26, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 0, 3, 4, 5, 6), (2, 1, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x26, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6)) * -2.0
    x27 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x27 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ooov, (2, 0, 3, 1), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x27, (2, 0), (2, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x27, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x27, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x27, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * 2.0
    x28 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x28 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ooov, (2, 3, 0, 1), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x28, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x28, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x28, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x28, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    x29 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x29 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvv, (0, 2, 3, 1), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x29, (1, 2), (0, 2)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x29, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 0, 4), (2, 3, 1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x29, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x29, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 0, 5, 6), (2, 3, 4, 1, 5, 6)) * -2.0
    x30 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x30 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvv, (0, 1, 2, 3), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x30, (2, 1), (0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x30, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x30, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x30, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 1, 5, 6), (2, 3, 4, 0, 5, 6)) * 2.0
    x31 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x31 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ooov, (2, 3, 0, 1), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x31, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x31, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x31, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x31, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    x32 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x32 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.vvov, (2, 3, 0, 1), (2, 3))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x32, (2, 1), (0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x32, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x32, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x32, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 1, 5, 6), (2, 3, 4, 0, 5, 6)) * 2.0
    x33 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x33 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovov, (2, 1, 3, 4), (0, 2, 3, 4))
    t1new_aa[np.ix_(soa,sva)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x33, (4, 0, 1, 3), (4, 2)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x33, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x33, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x33, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -2.0
    x34 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x34 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 3), (0, 4))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x34, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x34, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x34, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x34, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    x35 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x35 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    x36 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x36 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x35, (4, 0, 1, 3), (4, 2)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x36, (0, 1), (0, 1)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x36, (0, 1), (0, 1)) * -1.0
    del x36
    x37 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x37 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 2), (0, 4)) * -1.0
    x38 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x38 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x37, (2, 0), (2, 1))
    t1new_aa[np.ix_(soa,sva)] += einsum(x38, (0, 1), (0, 1)) * -1.0
    t1new_aa[np.ix_(soa,sva)] += einsum(x38, (0, 1), (0, 1)) * -1.0
    del x38
    x39 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x39 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x12, (2, 1), (0, 2))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x39, (2, 0), (2, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x39, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x39, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x39, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * 2.0
    x40 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x40 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x11, (2, 1), (0, 2))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x40, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x40, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x40, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x40, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    x41 = np.zeros((nocc[0], nocc[0]), dtype=np.float64)
    x41 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x15, (2, 1), (0, 2))
    t1new_aa[np.ix_(soa,sva)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x41, (2, 0), (2, 1)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x41, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x41, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 1, 3, 4, 5, 6), (2, 0, 3, 4, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x41, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 2, 3, 4, 5, 6), (0, 2, 3, 4, 5, 6)) * -2.0
    x42 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x42 += einsum(f.aa.oo, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x42, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x42, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x42
    x43 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x43 += einsum(f.aa.vv, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x43, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x43, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x43
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x44 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x44, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x44, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x44, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x44, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x44
    x45 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x45 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x45, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x45, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x45, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x45, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x45, (4, 0, 2, 5), (4, 1, 5, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x45, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 2, 7), (4, 0, 5, 6, 3, 7)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x45, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 4, 5, 2, 6, 7), (0, 4, 5, 3, 6, 7)) * 2.0
    x46 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x46 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x46, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x46, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x46, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x46, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    x47 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x47 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.oooo, (4, 0, 5, 1), (4, 5, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x47, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x47, (0, 1, 2, 3), (1, 0, 3, 2))
    del x47
    x48 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x48 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x48, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x48, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x48, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x48, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x48
    x49 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x49 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x49, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x49, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x49, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x49, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    x50 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x50 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.vvvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x50, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x50, (0, 1, 2, 3), (1, 0, 3, 2))
    del x50
    x51 = np.zeros((navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x51 += einsum(v.aabb.oOov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 1, 5, 3, 6), (6, 4, 0, 5))
    t2new_aaaa[np.ix_(soa,soa,sva,sVa)] += einsum(x51, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sVa)] += einsum(x51, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    del x51
    x52 = np.zeros((naocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x52 += einsum(v.aabb.vVov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 1), (5, 4, 6, 0))
    t2new_aaaa[np.ix_(soa,sOa,sva,sva)] += einsum(x52, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,sOa,sva,sva)] += einsum(x52, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x52
    x53 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x53 += einsum(v.aaaa.ooOV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 1, 2, 5, 6, 3), (4, 0, 5, 6))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x53, (0, 1, 2, 3), (1, 0, 3, 2)) * -6.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x53, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.0
    del x53
    x54 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x54 += einsum(v.aaaa.vOvV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 0, 3), (4, 5, 6, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x54, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x54, (0, 1, 2, 3), (0, 1, 2, 3)) * -6.0
    del x54
    x55 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x55 += einsum(x26, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 0, 3, 4), (1, 2, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x55, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x55, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x55
    x56 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x56 += einsum(f.aa.ov, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (0, 2, 3, 4))
    x57 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x57 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x56, (0, 2, 3, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x57, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x57, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x57
    x58 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x58 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x59 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x59 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x58, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x59, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x59, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x59, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x59, (0, 1, 2, 3), (1, 0, 3, 2))
    del x59
    x60 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x60 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    x61 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x61 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x60, (2, 0, 3, 4), (3, 2, 4, 1))
    del x60
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x61, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x61, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x61
    x62 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x62 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x35, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x62, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x62, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x62, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x62, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x62
    x63 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x63 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x64 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x64 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x63, (2, 3, 1, 4), (0, 2, 3, 4))
    del x63
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x64, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x64, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x64
    x65 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x65 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x66 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x66 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x65, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x66, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x66, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x66, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x66, (0, 1, 2, 3), (1, 0, 3, 2))
    del x66
    x67 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x67 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vvov, (2, 1, 3, 4), (0, 3, 2, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x67, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x67, (4, 1, 5, 3), (4, 0, 5, 2)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x67, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * 6.0
    x68 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x68 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x67, (4, 1, 5, 3), (4, 0, 2, 5))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x68, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x68, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x68, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x68, (0, 1, 2, 3), (1, 0, 2, 3))
    del x68
    x69 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x69 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x70 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x70 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x69, (4, 5, 0, 1), (4, 5, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x70, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x70, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x70, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x70, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x70
    x71 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x71 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x72 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x72 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x71, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x72, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x72, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x72, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x72, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x72
    x73 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x73 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x74 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x74 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x73, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x74, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x74, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x74, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x74, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x74
    x75 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x75 += einsum(x27, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x75, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x75, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x75
    x76 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x76 += einsum(x28, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x76, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x76, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x76
    x77 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x77 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x77, (4, 0, 5, 2), (4, 1, 5, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x77, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x77, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 4, 5, 3, 6, 7), (0, 4, 5, 2, 6, 7)) * -2.0
    x78 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x78 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x77, (4, 1, 5, 3), (4, 0, 2, 5))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x78, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x78, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x78, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x78, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x78
    x79 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x79 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x45, (4, 1, 3, 5), (4, 0, 2, 5))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x79, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x79, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x79, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x79, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x79
    x80 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x80 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 3), (0, 1, 4, 5))
    x81 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x81 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x80, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x81, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x81, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x81, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x81, (0, 1, 2, 3), (0, 1, 3, 2))
    del x81
    x82 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x82 += einsum(x30, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x82, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x82, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x82
    x83 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x83 += einsum(x29, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 0), (2, 3, 4, 1))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x83, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x83, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x83
    x84 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x84 += einsum(x31, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x84, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x84, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x84
    x85 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x85 += einsum(x32, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x85, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x85, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x85
    x86 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x86 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vOov, (1, 2, 3, 4), (2, 0, 3, 4))
    x87 = np.zeros((navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x87 += einsum(x86, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 0, 5, 3, 6), (6, 1, 4, 5))
    del x86
    t2new_aaaa[np.ix_(soa,soa,sva,sVa)] += einsum(x87, (0, 1, 2, 3), (1, 2, 3, 0)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sVa)] += einsum(x87, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    del x87
    x88 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x88 += einsum(v.aabb.oVov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 1), (5, 4, 0, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x88, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -2.0
    x89 = np.zeros((naocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x89 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x88, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,sOa,sva,sva)] += einsum(x89, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,sOa,sva,sva)] += einsum(x89, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x89
    x90 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x90 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovOV, (2, 1, 3, 4), (3, 4, 0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x90, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 0, 5, 6, 1), (2, 4, 5, 6)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x90, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -1.0
    x91 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x91 += einsum(x90, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 0, 5, 6, 1), (2, 4, 5, 6))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x91, (0, 1, 2, 3), (0, 1, 3, 2)) * -3.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x91, (0, 1, 2, 3), (1, 0, 3, 2)) * 3.0
    del x91
    x92 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x92 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oVvO, (2, 3, 1, 4), (4, 3, 0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x92, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 0, 5, 6, 1), (2, 4, 5, 6))
    x93 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x93 += einsum(x92, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 0, 5, 6, 1), (2, 4, 5, 6))
    del x92
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x93, (0, 1, 2, 3), (0, 1, 3, 2)) * 3.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x93, (0, 1, 2, 3), (1, 0, 3, 2)) * -3.0
    del x93
    x94 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x94 += einsum(v.aaaa.ovOV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 2, 6, 1, 3), (4, 5, 0, 6))
    x95 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x95 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x94, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x95, (0, 1, 2, 3), (0, 1, 2, 3)) * 6.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x95, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.0
    del x95
    x96 = np.zeros((naocc[0], navir[0]), dtype=np.float64)
    x96 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovOV, (0, 1, 2, 3), (2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x96, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x96, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 2.0
    del x96
    x97 = np.zeros((naocc[0], navir[0]), dtype=np.float64)
    x97 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oVvO, (0, 2, 1, 3), (3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x97, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * -6.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x97, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * -2.0
    del x97
    x98 = np.zeros((naocc[0], navir[0]), dtype=np.float64)
    x98 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.OVov, (2, 3, 0, 1), (2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x98, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x98, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 2.0
    del x98
    x99 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x99 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x99, (0, 1, 2, 3), (0, 1, 2, 3))
    x100 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x100 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x99, (4, 1, 5, 3), (4, 0, 5, 2))
    del x99
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x100, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x100, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x100
    x101 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x101 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    x102 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x102 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x101, (4, 1, 5, 3), (4, 0, 5, 2))
    del x101
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x102, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x102, (0, 1, 2, 3), (0, 1, 3, 2))
    del x102
    x103 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x103 += einsum(x34, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x103, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x103, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x103
    x104 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x104 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x46, (4, 1, 5, 3), (0, 4, 2, 5))
    del x46
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x104, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x104, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x104, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x104, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x104
    x105 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x105 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 4, 1, 3), (2, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x105, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 3, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x105, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x105, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 1, 5, 6), (2, 3, 4, 0, 5, 6)) * -2.0
    x106 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x106 += einsum(x105, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x106, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x106, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    del x106
    x107 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x107 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x49, (4, 1, 5, 3), (4, 0, 5, 2))
    del x49
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x107, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x107, (0, 1, 2, 3), (0, 1, 3, 2)) * -4.0
    del x107
    x108 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x108 += einsum(x37, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x108, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x108, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x108, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x108, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x108
    x109 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x109 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    x110 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x110 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x109, (4, 1, 5, 3), (4, 0, 5, 2))
    del x109
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x110, (0, 1, 2, 3), (0, 1, 2, 3)) * -4.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x110, (0, 1, 2, 3), (0, 1, 3, 2)) * 4.0
    del x110
    x111 = np.zeros((nvir[0], nvir[0]), dtype=np.float64)
    x111 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (0, 4, 1, 3), (2, 4))
    x112 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x112 += einsum(x111, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x112, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x112, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x112, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x112, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x112
    x113 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x113 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x114 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x114 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x113, (4, 5, 0, 1), (5, 4, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x114, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x114, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x114
    x115 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x115 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x69, (2, 3, 4, 0), (2, 4, 3, 1))
    x116 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x116 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x115, (2, 0, 3, 4), (2, 3, 1, 4))
    del x115
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x116, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x116, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x116, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x116, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x116
    x117 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x117 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x45, (2, 3, 1, 4), (0, 2, 3, 4))
    x118 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x118 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x117, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x118, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x118, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x118, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x118, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x118
    x119 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x119 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x33, (4, 5, 1, 3), (4, 0, 5, 2))
    x120 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x120 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x119, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x120, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x120, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x120, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x120, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x120
    x121 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x121 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x35, (2, 3, 4, 1), (2, 0, 4, 3))
    x122 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x122 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x121, (4, 5, 0, 1), (5, 4, 2, 3)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x122, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x122, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x122
    x123 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x123 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x35, (4, 1, 5, 3), (4, 0, 5, 2))
    x124 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x124 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x123, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x124, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x124, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x124, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x124, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x124
    x125 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x125 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x35, (4, 5, 1, 3), (4, 0, 5, 2))
    x126 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x126 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x125, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x126, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x126, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x126, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x126, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x126
    x127 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x127 += einsum(x39, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x127, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x127, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x127
    x128 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x128 += einsum(x40, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x128, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x128, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x128
    x129 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x129 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x113, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    x130 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x130 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x129, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x129
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x130, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x130, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x130
    x131 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x131 += einsum(x11, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 0, 4))
    x132 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x132 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x131, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x132, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x132, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x132
    x133 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x133 += einsum(x12, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 0, 4))
    x134 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x134 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x133, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x134, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x134, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x134
    x135 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x135 += einsum(x41, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x135, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x135, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x135
    x136 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x136 += einsum(x15, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sva)], (2, 3, 4, 1), (2, 3, 0, 4))
    x137 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x137 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x136, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x137, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x137, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x137
    x138 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x138 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x121, (2, 3, 0, 4), (3, 2, 4, 1))
    x139 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x139 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x138, (2, 3, 0, 4), (2, 3, 1, 4))
    del x138
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x139, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_aaaa[np.ix_(soa,soa,sva,sva)] += einsum(x139, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x139
    x140 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x140 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (2, 0, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x140, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x140, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x140, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 0, 5, 6, 2, 7), (4, 1, 5, 6, 3, 7)) * 6.0
    x141 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x141 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 2), (4, 1, 5, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x141, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x141, (1, 4, 3, 5), (0, 4, 2, 5)) * 2.0
    x142 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x142 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (4, 0, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x142, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x142, (1, 4, 3, 5), (0, 4, 2, 5)) * 4.0
    x143 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x143 += einsum(f.aa.ov, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x143, (0, 2, 3, 4), (2, 3, 1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x143, (0, 4, 5, 6), (5, 4, 1, 6, 2, 3)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x143, (0, 4, 5, 6), (4, 5, 1, 2, 6, 3)) * -2.0
    del x143
    x144 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x144 += einsum(f.bb.ov, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 0, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x144, (2, 0, 3, 4), (2, 3, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x144, (4, 0, 5, 6), (5, 4, 1, 2, 6, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x144, (4, 1, 5, 6), (4, 5, 0, 6, 3, 2)) * -1.0
    del x144
    x145 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x145 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.oovv, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x145, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x145, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x145, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -2.0
    del x145
    x146 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x146 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x146, (2, 0, 3, 4), (2, 3, 1, 4))
    del x146
    x147 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    x147 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x147, (2, 3, 1, 4), (0, 2, 3, 4))
    del x147
    x148 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x148 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vvoo, (2, 1, 3, 4), (0, 3, 4, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x148, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x148, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x148, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2)) * -1.0
    del x148
    x149 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x149 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 0, 2), (4, 5, 1, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x149, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x149, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x149, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -2.0
    del x149
    x150 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x150 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ooov, (4, 0, 5, 2), (4, 5, 1, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x150, (2, 0, 3, 4), (2, 3, 1, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x150, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x150, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * 2.0
    del x150
    x151 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x151 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovoo, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x151, (4, 0, 5, 1), (4, 5, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x151, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 1, 4, 5, 6, 7), (2, 0, 4, 5, 6, 7)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x151, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 3, 4, 5, 6, 7), (0, 2, 4, 5, 6, 7)) * 2.0
    x152 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x152 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x152, (4, 0, 5, 3), (4, 1, 2, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x152, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 3, 6, 7), (4, 0, 5, 2, 6, 7)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x152, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 4, 5, 6, 3, 7), (0, 4, 5, 6, 2, 7)) * -2.0
    del x152
    x153 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x153 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovoo, (4, 2, 5, 1), (0, 4, 5, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x153, (2, 0, 3, 4), (2, 3, 1, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x153, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x153, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * 2.0
    del x153
    x154 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x154 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovvv, (4, 2, 5, 3), (0, 4, 1, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x154, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x154, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x154, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -2.0
    del x154
    x155 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x155 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 3), (4, 5, 0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x155, (2, 0, 3, 4), (2, 3, 1, 4)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x155, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x155, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -4.0
    del x155
    x156 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x156 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ooov, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x156, (4, 0, 5, 1), (4, 5, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x156, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 1, 4, 5, 6, 7), (2, 0, 4, 5, 6, 7)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x156, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 3, 4, 5, 6, 7), (0, 2, 4, 5, 6, 7)) * 2.0
    x157 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x157 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x157, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x157, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x157, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2)) * -1.0
    del x157
    x158 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x158 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x158, (2, 3, 0, 4), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x158, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x158, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2))
    del x158
    x159 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x159 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 0, 5, 3), (4, 1, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x159, (2, 3, 0, 4), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x159, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x159, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2))
    del x159
    x160 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x160 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x160, (4, 1, 5, 3), (0, 4, 2, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x160, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 4, 5, 3, 6, 7), (0, 4, 5, 2, 6, 7)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x160, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 1, 5, 6, 3, 7), (4, 0, 5, 6, 2, 7)) * -2.0
    x161 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x161 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x161, (4, 1, 3, 5), (0, 4, 2, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x161, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x161, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x161, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x161, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x161, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 4, 5, 2, 6, 7), (0, 4, 5, 3, 6, 7)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x161, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 1, 5, 6, 2, 7), (4, 0, 5, 6, 3, 7)) * 2.0
    x162 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x162 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.vvov, (2, 3, 4, 1), (0, 4, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x162, (4, 1, 5, 2), (0, 4, 5, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x162, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 4, 5, 6, 3, 7), (0, 4, 5, 6, 2, 7)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x162, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 1, 5, 3, 6, 7), (4, 0, 5, 2, 6, 7)) * -2.0
    x163 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x163 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vvov, (4, 2, 5, 3), (0, 1, 5, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x163, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x163, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x163, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2)) * -1.0
    del x163
    x164 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x164 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aabb.ovoo, (1, 3, 4, 5), (0, 4, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x164, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x164, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * -4.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x164, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2)) * -2.0
    del x164
    x165 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x165 += einsum(v.aaaa.ovOV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 2, 1, 6, 3), (4, 0, 5, 6))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x165, (2, 0, 3, 4), (2, 3, 1, 4)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x165, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x165, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -4.0
    del x165
    x166 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x166 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovOV, (2, 1, 3, 4), (3, 4, 0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x166, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 3, 0, 5, 6, 1), (2, 4, 6, 5)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x166, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -1.0
    x167 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x167 += einsum(v.aabb.ovOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 2, 6, 1, 3), (5, 0, 4, 6))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x167, (2, 0, 3, 4), (2, 3, 1, 4)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x167, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x167, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -4.0
    del x167
    x168 = np.zeros((naocc[1], navir[1]), dtype=np.float64)
    x168 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovOV, (0, 1, 2, 3), (2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x168, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 0, 4, 5, 1), (3, 2, 5, 4)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x168, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    del x168
    x169 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x169 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.OVov, (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x169, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 0, 5, 6, 1), (4, 2, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x169, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -1.0
    x170 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x170 += einsum(v.aabb.OVov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 3, 1), (4, 5, 2, 6))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x170, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x170, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * -4.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x170, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2)) * -2.0
    del x170
    x171 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x171 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovOV, (2, 1, 3, 4), (3, 4, 0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x171, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 0, 5, 6, 1), (4, 2, 6, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x171, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    x172 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x172 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oVvO, (2, 3, 1, 4), (4, 3, 0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x172, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 0, 5, 6, 1), (4, 2, 6, 5))
    x173 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x173 += einsum(v.bbbb.ovOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 2, 1, 6, 3), (5, 4, 0, 6))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x173, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x173, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * -4.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x173, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2)) * -2.0
    del x173
    x174 = np.zeros((naocc[1], navir[1]), dtype=np.float64)
    x174 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovOV, (0, 1, 2, 3), (2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x174, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 0, 4, 5, 1), (3, 2, 5, 4)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x174, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * 6.0
    del x174
    x175 = np.zeros((naocc[1], navir[1]), dtype=np.float64)
    x175 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oVvO, (0, 2, 1, 3), (3, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x175, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 0, 4, 5, 1), (3, 2, 5, 4)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x175, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 0, 4, 5, 1), (2, 3, 4, 5)) * -6.0
    del x175
    x176 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x176 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (0, 4, 1, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x176, (4, 0, 5, 1), (4, 5, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x176, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 1, 4, 5, 6, 7), (2, 0, 4, 5, 6, 7)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x176, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 3, 4, 5, 6, 7), (0, 2, 4, 5, 6, 7)) * 2.0
    x177 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x177 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 2, 1, 4), (3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x177, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 4, 0)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x177, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 1, 5, 6), (2, 3, 4, 0, 5, 6)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x177, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 1, 6), (2, 3, 4, 5, 0, 6)) * -2.0
    x178 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x178 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 4, 5, 3), (1, 5, 2, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x178, (4, 1, 5, 2), (0, 4, 5, 3))
    del x178
    x179 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x179 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 4, 3, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x179, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x179, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x179, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x179, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x179, (0, 1, 2, 3), (1, 0, 3, 2))
    del x179
    x180 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x180 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x180, (4, 1, 5, 3), (0, 4, 2, 5)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x180, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x180, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x180, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x180, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    x181 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x181 += einsum(x18, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 1, 3, 4), (2, 0, 3, 4)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x181, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x181, (0, 1, 2, 3), (0, 1, 2, 3))
    del x181
    x182 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x182 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x182, (4, 1, 5, 3), (0, 4, 2, 5)) * -2.0
    x183 = np.zeros((nvir[1], nvir[1]), dtype=np.float64)
    x183 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (0, 4, 1, 3), (2, 4))
    x184 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x184 += einsum(x183, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 4, 0)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x184, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x184, (0, 1, 2, 3), (0, 1, 2, 3))
    del x184
    x185 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x185 += einsum(x111, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 3, 0, 4)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x185, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x185, (0, 1, 2, 3), (0, 1, 2, 3))
    del x185
    x186 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x186 += einsum(x37, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (1, 2, 3, 4), (0, 2, 3, 4)) * -1.0
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x186, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(x186, (0, 1, 2, 3), (0, 1, 2, 3))
    del x186
    x187 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x187 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 0, 5), (4, 1, 5, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x187, (1, 4, 3, 5), (0, 4, 2, 5)) * -2.0
    x188 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x188 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x140, (2, 3, 1, 4), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x188, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x188, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x188, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -2.0
    del x188
    x189 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x189 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x151, (2, 3, 4, 0), (2, 3, 4, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x189, (2, 0, 3, 4), (2, 3, 1, 4))
    del x189
    x190 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x190 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x156, (2, 3, 4, 0), (3, 2, 4, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x190, (0, 2, 3, 4), (2, 3, 1, 4))
    del x190
    x191 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x191 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x162, (2, 3, 4, 1), (0, 2, 3, 4))
    del x162
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x191, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x191, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x191, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2)) * -1.0
    del x191
    x192 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x192 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x35, (4, 0, 5, 2), (4, 5, 1, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x192, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x192, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x192, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -2.0
    del x192
    x193 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x193 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x35, (4, 5, 0, 2), (4, 5, 1, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x193, (2, 0, 3, 4), (2, 3, 1, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x193, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x193, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * 2.0
    del x193
    x194 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x194 += einsum(x11, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 0, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x194, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x194, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x194, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -2.0
    del x194
    x195 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x195 += einsum(x12, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 0, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x195, (2, 0, 3, 4), (2, 3, 1, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x195, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x195, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * 2.0
    del x195
    x196 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x196 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x33, (4, 5, 1, 3), (4, 5, 0, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x196, (2, 0, 3, 4), (2, 3, 1, 4)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x196, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x196, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -4.0
    del x196
    x197 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x197 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x13, (2, 3, 4, 1), (0, 2, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x197, (4, 0, 5, 1), (4, 5, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x197, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 1, 4, 5, 6, 7), (2, 0, 4, 5, 6, 7)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x197, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 3, 4, 5, 6, 7), (0, 2, 4, 5, 6, 7)) * 2.0
    x198 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x198 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x33, (4, 0, 5, 3), (4, 1, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x198, (2, 3, 0, 4), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x198, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x198, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2))
    del x198
    x199 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x199 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x13, (4, 5, 1, 2), (0, 4, 5, 3))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x199, (2, 0, 3, 4), (2, 3, 1, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x199, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x199, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * 2.0
    del x199
    x200 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x200 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x176, (2, 3, 4, 0), (2, 3, 4, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x200, (2, 0, 3, 4), (2, 3, 1, 4))
    del x200
    x201 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x201 += einsum(x15, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 1, 4), (2, 0, 3, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x201, (2, 0, 3, 4), (2, 3, 1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x201, (4, 0, 5, 6), (5, 4, 1, 6, 2, 3)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x201, (4, 0, 5, 6), (4, 5, 1, 2, 6, 3)) * -2.0
    del x201
    x202 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x202 += einsum(x0, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x202, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x202, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x202, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2)) * -1.0
    del x202
    x203 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x203 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x16, (4, 1, 5, 3), (0, 4, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x203, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x203, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x203, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2)) * -1.0
    del x203
    x204 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x204 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x16, (4, 5, 1, 3), (0, 4, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x204, (2, 3, 0, 4), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x204, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x204, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2))
    del x204
    x205 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x205 += einsum(x1, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x205, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x205, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x205, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2)) * -1.0
    del x205
    x206 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x206 += einsum(x20, (0, 1), t2.abab[np.ix_(soa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x206, (2, 3, 0, 4), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x206, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x206, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2))
    del x206
    x207 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x207 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x13, (1, 4, 5, 3), (0, 4, 5, 2))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x207, (2, 3, 0, 4), (2, 3, 4, 1)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x207, (4, 5, 0, 6), (5, 4, 1, 2, 6, 3)) * -4.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x207, (4, 5, 1, 6), (4, 5, 0, 6, 3, 2)) * -2.0
    del x207
    x208 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x208 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x197, (2, 3, 4, 0), (2, 3, 4, 1))
    t2new_abab[np.ix_(soa,sob,sva,svb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x208, (2, 0, 3, 4), (2, 3, 1, 4))
    del x208
    x209 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x209 += einsum(f.bb.oo, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x209, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x209, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x209
    x210 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x210 += einsum(f.bb.vv, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x210, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x210, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x210
    x211 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x211 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ooov, (2, 0, 3, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x211, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x211, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x211, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x211, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x211
    x212 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x212 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.oooo, (4, 0, 5, 1), (4, 5, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x212, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x212, (0, 1, 2, 3), (1, 0, 3, 2))
    del x212
    x213 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x213 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.oovv, (4, 1, 5, 3), (0, 4, 2, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x213, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x213, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x213, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x213, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x213
    x214 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x214 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.vvvv, (4, 2, 5, 3), (0, 1, 4, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x214, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x214, (0, 1, 2, 3), (1, 0, 3, 2))
    del x214
    x215 = np.zeros((navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x215 += einsum(v.aabb.ovoO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 3, 5, 1, 6), (6, 4, 2, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,sVb)] += einsum(x215, (0, 1, 2, 3), (2, 1, 3, 0)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,sVb)] += einsum(x215, (0, 1, 2, 3), (1, 2, 3, 0)) * -2.0
    del x215
    x216 = np.zeros((naocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x216 += einsum(v.aabb.ovvV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 5, 6, 1, 3), (5, 4, 6, 2))
    t2new_bbbb[np.ix_(sob,sOb,svb,svb)] += einsum(x216, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sOb,svb,svb)] += einsum(x216, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    del x216
    x217 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x217 += einsum(v.bbbb.ooOV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 1, 2, 5, 6, 3), (4, 0, 5, 6))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x217, (0, 1, 2, 3), (1, 0, 3, 2)) * -6.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x217, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.0
    del x217
    x218 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x218 += einsum(v.bbbb.vOvV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 0, 3), (4, 5, 6, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x218, (0, 1, 2, 3), (0, 1, 3, 2)) * 6.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x218, (0, 1, 2, 3), (0, 1, 2, 3)) * -6.0
    del x218
    x219 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x219 += einsum(x4, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 0, 3, 4), (1, 2, 3, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x219, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x219, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x219
    x220 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x220 += einsum(f.bb.ov, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (0, 2, 3, 4))
    x221 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x221 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x220, (0, 2, 3, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x221, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x221, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x221
    x222 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x222 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    x223 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x223 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x222, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x223, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x223, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x223, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x223, (0, 1, 2, 3), (1, 0, 3, 2))
    del x223
    x224 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x224 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oooo, (2, 3, 4, 0), (2, 3, 4, 1))
    x225 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x225 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x224, (2, 0, 3, 4), (2, 3, 1, 4))
    del x224
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x225, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x225, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x225
    x226 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x226 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x16, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x226, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x226, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x226, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x226, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x226
    x227 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x227 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x228 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x228 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x227, (2, 3, 1, 4), (0, 2, 3, 4))
    del x227
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x228, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x228, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x228
    x229 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x229 += einsum(x6, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x229, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x229, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x229
    x230 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x230 += einsum(x5, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x230, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x230, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x230
    x231 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x231 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x140, (0, 4, 2, 5), (4, 1, 3, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x231, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x231, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x231, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x231, (0, 1, 2, 3), (1, 0, 2, 3))
    del x231
    x232 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x232 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 5), (1, 4, 5, 3))
    x233 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x233 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x232, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x233, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x233, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x233, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x233, (0, 1, 2, 3), (1, 0, 3, 2))
    del x233
    x234 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x234 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x235 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x235 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x234, (4, 5, 0, 1), (4, 5, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x235, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x235, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x235, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x235, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x235
    x236 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x236 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    x237 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x237 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x236, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x237, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x237, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x237, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x237, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x237
    x238 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x238 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    x239 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x239 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x238, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x239, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x239, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x239, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x239, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x239
    x240 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x240 += einsum(x7, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x240, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x240, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x240
    x241 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x241 += einsum(x8, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x241, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x241, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x241
    x242 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x242 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x160, (4, 1, 5, 3), (4, 0, 2, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x242, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x242, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x242, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x242, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x242
    x243 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x243 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x161, (4, 1, 3, 5), (4, 0, 2, 5))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x243, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x243, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x243, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x243, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x243
    x244 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x244 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (4, 3, 5, 2), (0, 1, 4, 5)) * -1.0
    x245 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x245 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x244, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x245, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x245, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x245, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x245, (0, 1, 2, 3), (0, 1, 3, 2))
    del x245
    x246 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x246 += einsum(x10, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x246, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x246, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x246
    x247 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x247 += einsum(x9, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 0), (2, 3, 4, 1))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x247, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x247, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x247
    x248 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x248 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ovvO, (2, 3, 1, 4), (4, 2, 0, 3))
    x249 = np.zeros((navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x249 += einsum(x248, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 0, 5, 3, 6), (6, 2, 4, 5))
    del x248
    t2new_bbbb[np.ix_(sob,sob,svb,sVb)] += einsum(x249, (0, 1, 2, 3), (1, 2, 3, 0)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,sVb)] += einsum(x249, (0, 1, 2, 3), (2, 1, 3, 0)) * -2.0
    del x249
    x250 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x250 += einsum(v.aabb.ovoV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 5, 6, 1, 3), (5, 4, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x250, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -2.0
    x251 = np.zeros((naocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x251 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x250, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sOb,svb,svb)] += einsum(x251, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sOb,svb,svb)] += einsum(x251, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x251
    x252 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x252 += einsum(x171, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 0, 5, 6, 1), (2, 4, 5, 6))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x252, (0, 1, 2, 3), (0, 1, 3, 2)) * -3.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x252, (0, 1, 2, 3), (1, 0, 3, 2)) * 3.0
    del x252
    x253 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x253 += einsum(x172, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 0, 5, 6, 1), (2, 4, 5, 6))
    del x172
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x253, (0, 1, 2, 3), (0, 1, 3, 2)) * 3.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x253, (0, 1, 2, 3), (1, 0, 3, 2)) * -3.0
    del x253
    x254 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x254 += einsum(v.bbbb.ovOV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 2, 6, 1, 3), (4, 5, 0, 6))
    x255 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x255 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x254, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x255, (0, 1, 2, 3), (0, 1, 2, 3)) * 6.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x255, (0, 1, 2, 3), (0, 1, 3, 2)) * -6.0
    del x255
    x256 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x256 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x141, (0, 4, 2, 5), (4, 1, 5, 3))
    del x141
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x256, (0, 1, 2, 3), (1, 0, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x256, (0, 1, 2, 3), (1, 0, 2, 3)) * -1.0
    del x256
    x257 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x257 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x187, (0, 4, 2, 5), (4, 1, 5, 3))
    del x187
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x257, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x257, (0, 1, 2, 3), (1, 0, 2, 3))
    del x257
    x258 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x258 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x142, (0, 4, 2, 5), (1, 4, 3, 5))
    del x142
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x258, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x258, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x258, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x258, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    del x258
    x259 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x259 += einsum(x14, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x259, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x259, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x259
    x260 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x260 += einsum(x177, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x260, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x260, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x260
    x261 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x261 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x180, (4, 1, 5, 3), (4, 0, 5, 2))
    del x180
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x261, (0, 1, 2, 3), (0, 1, 2, 3)) * 4.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x261, (0, 1, 2, 3), (0, 1, 3, 2)) * -4.0
    del x261
    x262 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x262 += einsum(x18, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (2, 0, 3, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x262, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x262, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x262, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x262, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x262
    x263 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x263 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x182, (4, 1, 5, 3), (4, 0, 5, 2))
    del x182
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x263, (0, 1, 2, 3), (0, 1, 2, 3)) * -4.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x263, (0, 1, 2, 3), (0, 1, 3, 2)) * 4.0
    del x263
    x264 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x264 += einsum(x183, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 4, 0))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x264, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x264, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x264, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x264, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x264
    x265 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x265 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 5, 3), (0, 1, 4, 5))
    x266 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x266 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x265, (4, 5, 0, 1), (5, 4, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x266, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x266, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x266
    x267 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x267 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x234, (2, 3, 4, 0), (2, 4, 3, 1))
    x268 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x268 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x267, (2, 0, 3, 4), (2, 3, 1, 4))
    del x267
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x268, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x268, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x268, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x268, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x268
    x269 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x269 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x161, (2, 3, 1, 4), (0, 2, 3, 4))
    x270 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x270 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x269, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x270, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x270, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x270, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x270, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x270
    x271 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x271 += einsum(x21, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x271, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x271, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x271
    x272 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x272 += einsum(x0, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    x273 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x273 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x272, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x273, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x273, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x273
    x274 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x274 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x13, (0, 4, 5, 2), (4, 1, 5, 3))
    x275 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x275 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x274, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x275, (0, 1, 2, 3), (0, 1, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x275, (0, 1, 2, 3), (0, 1, 3, 2))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x275, (0, 1, 2, 3), (1, 0, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x275, (0, 1, 2, 3), (1, 0, 3, 2)) * -1.0
    del x275
    x276 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x276 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x16, (2, 3, 4, 1), (2, 0, 4, 3))
    x277 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x277 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x276, (4, 5, 0, 1), (5, 4, 2, 3)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x277, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x277, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x277
    x278 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x278 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x16, (4, 1, 5, 3), (4, 0, 5, 2))
    x279 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x279 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x278, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x279, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x279, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x279, (0, 1, 2, 3), (1, 0, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x279, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x279
    x280 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x280 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x16, (4, 5, 1, 3), (4, 0, 5, 2))
    x281 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x281 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x280, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x281, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x281, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x281, (0, 1, 2, 3), (1, 0, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x281, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x281
    x282 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x282 += einsum(x22, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x282, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x282, (0, 1, 2, 3), (1, 0, 3, 2)) * -2.0
    del x282
    x283 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x283 += einsum(x23, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 1, 3, 4), (0, 2, 3, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x283, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x283, (0, 1, 2, 3), (1, 0, 3, 2)) * 2.0
    del x283
    x284 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x284 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x265, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    x285 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x285 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x284, (2, 3, 0, 4), (2, 3, 4, 1)) * -1.0
    del x284
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x285, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x285, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x285
    x286 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x286 += einsum(x1, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    x287 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x287 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x286, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x287, (0, 1, 2, 3), (0, 1, 2, 3)) * 2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x287, (0, 1, 2, 3), (0, 1, 3, 2)) * -2.0
    del x287
    x288 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x288 += einsum(x20, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    x289 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x289 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x288, (2, 3, 0, 4), (2, 3, 1, 4))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x289, (0, 1, 2, 3), (0, 1, 2, 3)) * -2.0
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x289, (0, 1, 2, 3), (0, 1, 3, 2)) * 2.0
    del x289
    x290 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x290 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x276, (2, 3, 0, 4), (3, 2, 4, 1))
    x291 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x291 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x290, (2, 3, 0, 4), (2, 3, 1, 4))
    del x290
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x291, (0, 1, 2, 3), (0, 1, 2, 3))
    t2new_bbbb[np.ix_(sob,sob,svb,svb)] += einsum(x291, (0, 1, 2, 3), (0, 1, 3, 2)) * -1.0
    del x291
    x292 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x292 += einsum(f.aa.oo, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x292, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x292, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x292
    x293 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x293 += einsum(f.aa.vv, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x293, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x293, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x293
    x294 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x294 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.oovO, (4, 1, 5, 6), (6, 3, 0, 4, 2, 5)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x294, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x294, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x294, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x294, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x294
    x295 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x295 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ooOV, (4, 1, 5, 6), (5, 6, 0, 4, 2, 3))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x295, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x295, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x295
    x296 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x296 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoO, (4, 5, 1, 6), (6, 3, 0, 4, 2, 5)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x296, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x296, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x296, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x296, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x296
    x297 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x297 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.oOoV, (1, 4, 5, 6), (4, 6, 0, 5, 2, 3))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x297, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x297, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x297
    x298 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x298 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.oooV, (4, 0, 5, 6), (1, 6, 5, 4, 2, 3)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x298, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x298, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x298
    x299 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x299 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ooov, (4, 0, 5, 6), (1, 3, 4, 5, 2, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x299, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x299, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x299, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x299, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x299
    x300 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x300 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvV, (4, 5, 3, 6), (1, 6, 0, 4, 2, 5)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x300, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x300, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x300, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x300, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x300
    x301 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x301 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovvv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x301, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x301, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x301, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x301, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x301, (4, 5, 6, 0, 2, 7), (6, 1, 4, 7, 3, 5)) * 2.0
    x302 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x302 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.oVvv, (4, 5, 6, 3), (1, 5, 0, 4, 2, 6)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x302, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x302, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x302, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x302, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x302
    x303 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x303 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.vvvO, (4, 2, 5, 6), (6, 3, 0, 1, 5, 4)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x303, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x303, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x303
    x304 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x304 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.vvOV, (4, 3, 5, 6), (5, 6, 0, 1, 2, 4))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x304, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x304, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x304
    x305 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x305 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.vOvV, (4, 5, 3, 6), (5, 6, 0, 1, 2, 4))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x305, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x305, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x305
    x306 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x306 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 7, 4, 0, 6, 1))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x306, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x306, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x306, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x306, (4, 5, 6, 0, 7, 2), (6, 1, 4, 7, 3, 5)) * 2.0
    x307 = np.zeros((naocc[0], navir[0], navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x307 += einsum(v.aabb.oVov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 7, 1, 4, 0, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x307, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 1, 2)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x307, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 1, 2)) * -2.0
    del x307
    x308 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x308 += einsum(v.aabb.vOov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 1, 7, 4, 6, 0))
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x308, (0, 1, 2, 3, 4, 5), (3, 0, 1, 5, 4, 2)) * 2.0
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x308, (0, 1, 2, 3, 4, 5), (3, 0, 1, 4, 5, 2)) * -2.0
    del x308
    x309 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x309 += einsum(v.aaaa.ooOO, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 1, 3, 5, 6, 7), (2, 7, 4, 0, 5, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x309, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x309, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x309
    x310 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x310 += einsum(v.aaaa.oovv, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 1, 5, 6, 3, 7), (5, 7, 4, 0, 6, 2))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x310, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x310, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x310, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    del x310
    x311 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x311 += einsum(v.aaaa.ooVV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 1, 5, 6, 7, 3), (5, 2, 4, 0, 6, 7))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x311, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x311, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x311
    x312 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x312 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 2, 5, 6, 3, 7), (5, 7, 4, 0, 6, 1))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x312, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x312, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x312, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x312, (4, 5, 6, 0, 7, 2), (6, 1, 4, 7, 3, 5)) * 6.0
    x313 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x313 += einsum(v.aaaa.oVoV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 2, 5, 6, 7, 3), (5, 1, 4, 0, 6, 7))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x313, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x313, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x313
    x314 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x314 += einsum(v.aaaa.vOvO, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 3, 6, 2, 7), (1, 7, 4, 5, 6, 0))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x314, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x314, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    del x314
    x315 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x315 += einsum(v.aaaa.vvOO, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 3, 6, 1, 7), (2, 7, 4, 5, 6, 0))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x315, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x315, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    del x315
    x316 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x316 += einsum(v.aaaa.vvVV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 6, 7, 1, 3), (6, 2, 4, 5, 7, 0))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x316, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x316, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    del x316
    x317 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x317 += einsum(x26, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 0, 3, 4, 5, 6), (3, 6, 1, 2, 4, 5))
    del x26
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x317, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x317, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x317
    x318 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x318 += einsum(f.aa.ov, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 0, 2, 3, 5))
    x319 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x319 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x318, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 1, 6))
    del x318
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x319, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x319, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x319
    x320 = np.zeros((navir[0], navir[0]), dtype=np.float64)
    x320 += einsum(f.aa.oV, (0, 1), t1.aa[np.ix_(soa,sVa)], (0, 2), (1, 2))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x320, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x320, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * -2.0
    del x320
    x321 = np.zeros((naocc[0], naocc[0]), dtype=np.float64)
    x321 += einsum(f.aa.vO, (0, 1), t1.aa[np.ix_(sOa,sva)], (2, 0), (1, 2))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x321, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x321, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * -2.0
    del x321
    x322 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x322 += einsum(f.aa.ov, (0, 1), t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (2, 3, 1, 4), (3, 4, 0, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x322, (4, 5, 0, 6), (6, 1, 4, 2, 3, 5)) * -2.0
    x323 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x323 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x322, (4, 5, 1, 6), (4, 5, 6, 0, 2, 3))
    del x322
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x323, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x323, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x323
    x324 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x324 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x56, (0, 4, 5, 6), (1, 3, 4, 5, 2, 6))
    del x56
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x324, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x324, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x324
    x325 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x325 += einsum(f.aa.ov, (0, 1), t2.aaaa[np.ix_(soa,sOa,sva,sva)], (2, 3, 4, 1), (3, 0, 2, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x325, (4, 0, 5, 6), (5, 1, 4, 6, 3, 2)) * 2.0
    x326 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x326 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x325, (4, 1, 5, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x325
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x326, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x326, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x326, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x326, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x326
    x327 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x327 += einsum(f.aa.ov, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sVa)], (2, 3, 1, 4), (4, 0, 2, 3)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x327, (4, 0, 5, 6), (6, 5, 1, 2, 3, 4)) * 4.0
    del x327
    x328 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x328 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oooO, (2, 0, 3, 4), (4, 3, 2, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x328, (4, 0, 5, 6), (5, 1, 4, 6, 3, 2))
    x329 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x329 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x328, (4, 1, 5, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x328
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x329, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x329, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x329, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x329, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x329
    x330 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x330 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oooO, (2, 3, 0, 4), (4, 2, 3, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x330, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -1.0
    x331 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x331 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x330, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x330
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x331, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x331, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x331, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x331, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x331
    x332 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x332 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvO, (2, 1, 3, 4), (4, 0, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x332, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2))
    x333 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x333 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x332, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    del x332
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x333, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x333, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x333, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x333, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x333
    x334 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x334 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x90, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x90
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x334, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x334, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x334
    x335 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x335 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovvO, (4, 2, 5, 6), (6, 3, 0, 1, 4, 5)) * -1.0
    x336 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x336 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x335, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x335
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x336, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x336, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x336
    x337 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x337 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovOV, (4, 3, 5, 6), (5, 6, 0, 1, 4, 2))
    x338 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x338 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x337, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x337
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x338, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x338, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x338
    x339 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x339 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oOvV, (2, 3, 1, 4), (3, 4, 0, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x339, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5))
    x340 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x340 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x339, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x339
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x340, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x340, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x340
    x341 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x341 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oOvv, (2, 3, 4, 1), (3, 0, 2, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x341, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -1.0
    x342 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x342 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x341, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    del x341
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x342, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x342, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x342, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x342, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x342
    x343 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x343 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.oOvV, (4, 5, 3, 6), (5, 6, 0, 1, 4, 2))
    x344 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x344 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x343, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x343
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x344, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x344, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x344
    x345 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x345 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.oOvv, (4, 5, 6, 2), (5, 3, 0, 1, 4, 6)) * -1.0
    x346 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x346 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x345, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x345
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x346, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x346, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x346
    x347 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x347 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oovV, (2, 3, 1, 4), (4, 0, 2, 3))
    x348 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x348 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x347, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x347
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x348, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x348, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x348
    x349 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x349 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x58, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x58
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x349, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x349, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x349, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x349, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x349
    x350 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x350 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.oooo, (4, 5, 6, 0), (1, 3, 4, 5, 6, 2))
    x351 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x351 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x350, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 1, 6))
    del x350
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x351, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x351, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x351, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x351, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x351
    x352 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x352 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.oovV, (4, 5, 3, 6), (1, 6, 0, 4, 5, 2)) * -1.0
    x353 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x353 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x352, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x352
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x353, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x353, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x353, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x353, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x353
    x354 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x354 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.oovv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x355 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x355 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x354, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x354
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x355, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x355, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x355, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x355, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x355
    x356 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x356 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovoV, (2, 1, 3, 4), (4, 0, 3, 2))
    x357 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x357 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x356, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x356
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x357, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x357, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x357
    x358 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x358 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x35, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x358, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x358, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x358, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x358, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x358
    x359 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x359 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovoV, (4, 3, 5, 6), (1, 6, 0, 5, 4, 2)) * -1.0
    x360 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x360 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x359, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x359
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x360, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x360, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x360, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x360, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x360
    x361 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x361 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 6, 2), (1, 3, 0, 4, 6, 5))
    x362 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x362 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x361, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x362, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x362, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x362, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x362, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x362
    x363 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x363 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oooO, (2, 0, 3, 4), (4, 1, 3, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x363, (4, 5, 0, 6), (6, 1, 4, 2, 3, 5)) * -1.0
    x364 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x364 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x363, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3))
    del x363
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x364, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x364, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x364
    x365 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x365 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oooO, (2, 3, 0, 4), (4, 1, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x365, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5))
    x366 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x366 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x365, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x365
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x366, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x366, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x366
    x367 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x367 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.ovvO, (0, 2, 3, 4), (4, 1, 3, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x367, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5))
    x368 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x368 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x367, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x367
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x368, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x368, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x368
    x369 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x369 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oOvv, (0, 2, 3, 4), (2, 1, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x369, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * -1.0
    x370 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x370 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x369, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x369
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x370, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x370, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x370
    x371 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x371 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oooo, (2, 3, 4, 0), (1, 2, 3, 4))
    x372 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x372 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x371, (4, 5, 0, 6), (1, 4, 6, 5, 2, 3)) * -1.0
    del x371
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x372, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x372, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x372
    x373 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x373 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oovv, (2, 0, 3, 4), (1, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x373, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4))
    x374 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x374 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x373, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x373
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x374, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x374, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x374, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x374, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x374
    x375 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x375 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.ovov, (2, 3, 0, 4), (1, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x375, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4)) * -1.0
    x376 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x376 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x375, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x376, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x376, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x376, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x376, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x376
    x377 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x377 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.vvvV, (2, 1, 3, 4), (4, 0, 3, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x377, (4, 5, 2, 6), (5, 1, 0, 6, 3, 4))
    x378 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x378 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x377, (4, 5, 3, 6), (1, 4, 5, 0, 2, 6)) * -1.0
    del x377
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x378, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x378, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x378, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x378, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x378
    x379 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x379 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.vvvV, (2, 3, 1, 4), (4, 0, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x379, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4)) * -1.0
    x380 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x380 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x379, (4, 5, 6, 3), (1, 4, 5, 0, 2, 6)) * -1.0
    del x379
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x380, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x380, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x380, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x380, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x380
    x381 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x381 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.vvvv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x382 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x382 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x381, (2, 3, 4, 1, 5, 6), (2, 3, 0, 4, 5, 6))
    del x381
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x382, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x382, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x382, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x382
    x383 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x383 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.oovV, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x383, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -1.0
    x384 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x384 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x383, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x383
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x384, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x384, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x384
    x385 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x385 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x385, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2))
    x386 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x386 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x385, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x385
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x386, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x386, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x386, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x386, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x386
    x387 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x387 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.ovoV, (2, 1, 3, 4), (0, 4, 3, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x387, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5))
    x388 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x388 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x387, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x387
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x388, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x388, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x388
    x389 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x389 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x389, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -1.0
    x390 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x390 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x389, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x390, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x390, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x390, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x390, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x390
    x391 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x391 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.vvvV, (2, 1, 3, 4), (0, 4, 3, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x391, (4, 5, 2, 6), (0, 1, 4, 6, 3, 5)) * -1.0
    x392 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x392 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x391, (4, 5, 3, 6), (4, 5, 0, 1, 2, 6))
    del x391
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x392, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x392, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x392
    x393 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x393 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.vvvV, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x393, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5))
    x394 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x394 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x393, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x393
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x394, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x394, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x394
    x395 = np.zeros((naocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x395 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x396 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x396 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x395, (4, 5, 2, 6), (4, 3, 0, 1, 6, 5)) * -1.0
    del x395
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x396, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x396, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x396
    x397 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x397 += einsum(v.aabb.ooov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 7, 4, 0, 1, 6))
    x398 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x398 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x397, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x397
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x398, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x398, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x398, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x398, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x398
    x399 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x399 += einsum(v.aabb.oOov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 1, 7, 4, 0, 6))
    x400 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x400 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x399, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x399
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x400, (0, 1, 2, 3, 4, 5), (3, 0, 1, 4, 5, 2)) * -2.0
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x400, (0, 1, 2, 3, 4, 5), (3, 0, 1, 5, 4, 2)) * 2.0
    del x400
    x401 = np.zeros((navir[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x401 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.ooov, (2, 0, 3, 4), (1, 2, 3, 4))
    x402 = np.zeros((naocc[0], navir[0], navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x402 += einsum(x401, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 0, 7, 4, 1, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x402, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 2, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x402, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 2, 1)) * 2.0
    del x402
    x403 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x403 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.oOov, (0, 2, 3, 4), (2, 1, 3, 4))
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sVa,sVfa)] += einsum(x403, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * -2.0
    x404 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x404 += einsum(x67, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 1, 5, 6, 3, 7), (5, 7, 0, 4, 6, 2))
    del x67
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x404, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x404, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x404, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x404, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x404
    x405 = np.zeros((navir[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x405 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vVov, (1, 2, 3, 4), (2, 0, 3, 4))
    x406 = np.zeros((naocc[0], navir[0], navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x406 += einsum(x405, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 7, 0, 1, 4, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x406, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 1, 2)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x406, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 1, 2)) * -2.0
    del x406
    x407 = np.zeros((naocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x407 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vvov, (2, 1, 3, 4), (0, 3, 2, 4))
    x408 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x408 += einsum(x407, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 1, 5, 6, 3, 7), (0, 5, 7, 4, 6, 2))
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x408, (0, 1, 2, 3, 4, 5), (3, 1, 0, 5, 4, 2)) * 2.0
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x408, (0, 1, 2, 3, 4, 5), (3, 1, 0, 4, 5, 2)) * -2.0
    del x408
    x409 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x409 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vVov, (1, 2, 3, 4), (0, 2, 3, 4))
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sVa,sVfa)] += einsum(x409, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * 2.0
    x410 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x410 += einsum(x69, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (3, 2, 4, 5, 6, 7), (4, 7, 0, 1, 5, 6)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x410, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x410, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x410
    x411 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x411 += einsum(v.aaaa.ooov, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 2, 5, 6, 3, 7), (5, 7, 4, 0, 1, 6))
    x412 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x412 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x411, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x411
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x412, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x412, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x412, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x412, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x412
    x413 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x413 += einsum(v.aaaa.ooov, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 1, 5, 6, 3, 7), (5, 7, 4, 0, 2, 6))
    x414 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x414 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x413, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x413
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x414, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x414, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x414, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x414, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x414
    x415 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x415 += einsum(x27, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 1, 3, 4, 5, 6), (3, 6, 2, 0, 4, 5))
    del x27
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x415, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x415, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x415
    x416 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x416 += einsum(x28, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 1, 3, 4, 5, 6), (3, 6, 2, 0, 4, 5))
    del x28
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x416, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x416, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x416
    x417 = np.zeros((naocc[0], naocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x417 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovOO, (2, 1, 3, 4), (3, 4, 0, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x417, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * 2.0
    x418 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x418 += einsum(x417, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 1, 5, 6, 7), (0, 7, 2, 4, 5, 6))
    del x417
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x418, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x418, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x418
    x419 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x419 += einsum(x77, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 1, 5, 6, 3, 7), (5, 7, 0, 4, 6, 2))
    del x77
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x419, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x419, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x419, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x419, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    del x419
    x420 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x420 += einsum(x45, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 1, 5, 6, 2, 7), (5, 7, 0, 4, 6, 3))
    del x45
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x420, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x420, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x420, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x420, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 6.0
    del x420
    x421 = np.zeros((navir[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x421 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovVV, (2, 1, 3, 4), (3, 4, 0, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x421, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * -2.0
    x422 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x422 += einsum(x421, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x421
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x422, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x422, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x422
    x423 = np.zeros((navir[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x423 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oVvV, (2, 3, 1, 4), (3, 4, 0, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x423, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 5, 6, 7, 0), (2, 4, 5, 6, 7, 1)) * 2.0
    x424 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x424 += einsum(x423, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 5, 6, 7, 0), (5, 1, 2, 4, 6, 7))
    del x423
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x424, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x424, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x424
    x425 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x425 += einsum(v.aaaa.ovOO, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 3, 6, 1, 7), (2, 7, 4, 5, 0, 6))
    x426 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x426 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x425, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x425
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x426, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x426, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x426
    x427 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x427 += einsum(v.aaaa.oOvO, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 3, 6, 2, 7), (1, 7, 4, 5, 0, 6))
    x428 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x428 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x427, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x427
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x428, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x428, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x428
    x429 = np.zeros((naocc[0], naocc[0]), dtype=np.float64)
    x429 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovOO, (0, 1, 2, 3), (2, 3))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x429, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x429, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    del x429
    x430 = np.zeros((naocc[0], naocc[0]), dtype=np.float64)
    x430 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oOvO, (0, 2, 1, 3), (2, 3))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x430, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * 6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x430, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * 2.0
    del x430
    x431 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x431 += einsum(v.aaaa.ovvv, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 6, 1, 3, 7), (6, 7, 4, 5, 0, 2))
    x432 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x432 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x431, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x431
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x432, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x432, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x432
    x433 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x433 += einsum(v.aaaa.ovVV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 6, 7, 1, 3), (6, 2, 4, 5, 0, 7))
    x434 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x434 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x433, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x433
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x434, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x434, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x434
    x435 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x435 += einsum(x30, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 5, 0))
    del x30
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x435, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x435, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    del x435
    x436 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x436 += einsum(x29, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 0, 6), (4, 6, 2, 3, 5, 1))
    del x29
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x436, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x436, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    del x436
    x437 = np.zeros((navir[0], navir[0]), dtype=np.float64)
    x437 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovVV, (0, 1, 2, 3), (2, 3))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x437, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x437, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 2.0
    del x437
    x438 = np.zeros((navir[0], navir[0]), dtype=np.float64)
    x438 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.oVvV, (0, 2, 1, 3), (2, 3))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x438, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x438, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * -2.0
    del x438
    x439 = np.zeros((navir[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x439 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oooV, (2, 0, 3, 4), (1, 4, 3, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x439, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 4, 5, 6, 7, 1), (3, 4, 5, 6, 7, 0)) * -2.0
    x440 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x440 += einsum(x439, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 2, 5, 6, 7, 1), (5, 0, 4, 3, 6, 7))
    del x439
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x440, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x440, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x440
    x441 = np.zeros((navir[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x441 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oooV, (2, 3, 0, 4), (1, 4, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x441, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * 2.0
    x442 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x442 += einsum(x441, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 5, 6, 7, 1), (5, 0, 4, 2, 6, 7))
    del x441
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x442, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x442, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x442
    x443 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=np.float64)
    x443 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oVOO, (0, 2, 3, 4), (3, 4, 1, 2))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x443, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x443, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 2.0
    del x443
    x444 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=np.float64)
    x444 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.oOOV, (0, 2, 3, 4), (3, 2, 1, 4))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x444, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 0, 6, 7, 3), (4, 5, 1, 6, 7, 2)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x444, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 3), (4, 5, 1, 6, 7, 2)) * -2.0
    del x444
    x445 = np.zeros((navir[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x445 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.ovvV, (0, 2, 3, 4), (1, 4, 3, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x445, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 3, 7, 1), (4, 5, 6, 2, 7, 0)) * 2.0
    x446 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x446 += einsum(x445, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 6, 7, 3, 1), (6, 0, 4, 5, 7, 2))
    del x445
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x446, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x446, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    del x446
    x447 = np.zeros((naocc[0], naocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x447 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.oovO, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x447, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * 2.0
    x448 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x448 += einsum(x447, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 1, 5, 6, 7), (0, 7, 4, 2, 5, 6))
    del x447
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x448, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x448, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x448
    x449 = np.zeros((naocc[0], naocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x449 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.vvvO, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x449, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7)) * -2.0
    x450 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x450 += einsum(x449, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 6, 2))
    del x449
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x450, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x450, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    del x450
    x451 = np.zeros((naocc[0], naocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x451 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.vvvO, (2, 1, 3, 4), (0, 4, 3, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x451, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 2, 6, 7), (4, 5, 0, 3, 6, 7)) * 2.0
    x452 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x452 += einsum(x451, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 2, 7), (0, 7, 4, 5, 6, 3))
    del x451
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x452, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x452, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    del x452
    x453 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=np.float64)
    x453 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.vOVV, (1, 2, 3, 4), (0, 2, 3, 4))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x453, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x453, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -2.0
    del x453
    x454 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=np.float64)
    x454 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.vVOV, (1, 2, 3, 4), (0, 3, 4, 2))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x454, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 7, 2), (4, 5, 0, 6, 7, 3)) * 6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x454, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 7, 2), (4, 5, 0, 6, 7, 3)) * 2.0
    del x454
    x455 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x455 += einsum(x31, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 1, 3, 4, 5, 6), (3, 6, 2, 0, 4, 5))
    del x31
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x455, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x455, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x455
    x456 = np.zeros((naocc[0], naocc[0]), dtype=np.float64)
    x456 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.OOov, (2, 3, 0, 1), (2, 3))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x456, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x456, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    del x456
    x457 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x457 += einsum(x32, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 5, 0))
    del x32
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x457, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x457, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    del x457
    x458 = np.zeros((navir[0], navir[0]), dtype=np.float64)
    x458 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.VVov, (2, 3, 0, 1), (2, 3))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x458, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x458, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 2.0
    del x458
    x459 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x459 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.oOov, (4, 5, 1, 3), (5, 0, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x459, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -1.0
    x460 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x460 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x459, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x459
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x460, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x460, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x460, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x460, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x460
    x461 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x461 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x65, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x65
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x461, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x461, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x461, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x461, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x461
    x462 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x462 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.vvov, (4, 2, 5, 6), (1, 3, 0, 5, 4, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x462, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x462, (4, 5, 6, 1, 7, 3), (6, 0, 4, 7, 2, 5)) * 4.0
    x463 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x463 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x462, (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 2, 7))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x463, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x463, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x463, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x463, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x463
    x464 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x464 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vVov, (4, 5, 1, 3), (5, 0, 2, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x464, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4))
    x465 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x465 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x464, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x464
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x465, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x465, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x465, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x465, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x465
    x466 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x466 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oOov, (4, 5, 1, 3), (5, 2, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x466, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5))
    x467 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x467 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x466, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x466
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x467, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x467, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x467
    x468 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x468 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 3), (2, 0, 4, 5))
    x469 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x469 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x468, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x468
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x469, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x469, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x469
    x470 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x470 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.vvov, (4, 5, 1, 3), (2, 0, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x470, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4)) * -1.0
    x471 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x471 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x470, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x470
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x471, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x471, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x471, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x471, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x471
    x472 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x472 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x472, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2))
    x473 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x473 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x472, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x472
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x473, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x473, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x473, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x473, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x473
    x474 = np.zeros((naocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x474 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vvov, (4, 5, 1, 3), (0, 2, 4, 5))
    x475 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x475 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x474, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x474
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x475, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x475, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x475
    x476 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x476 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vVov, (4, 5, 1, 3), (0, 5, 2, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x476, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * -1.0
    x477 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x477 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x476, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x476
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x477, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x477, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x477
    x478 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x478 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 3), (0, 2, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x478, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -1.0
    x479 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x479 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x478, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x478
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x479, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x479, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x479
    x480 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x480 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.vvov, (4, 5, 1, 3), (0, 2, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x480, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5))
    x481 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x481 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x480, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x480
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x481, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x481, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x481
    x482 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x482 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovoO, (1, 3, 4, 5), (5, 0, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x482, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -2.0
    x483 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x483 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x482, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x482
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x483, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x483, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x483, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x483, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x483
    x484 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x484 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoO, (4, 2, 1, 5), (5, 3, 0, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x484, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -2.0
    x485 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x485 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x484, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x484
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x485, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x485, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x485
    x486 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x486 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovoO, (4, 3, 1, 5), (5, 0, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x486, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * 2.0
    x487 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x487 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x486, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x486
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x487, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x487, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x487, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x487, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x487
    x488 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x488 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoO, (1, 2, 4, 5), (5, 3, 0, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x488, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * 2.0
    x489 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x489 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x488, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x488
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x489, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x489, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x489
    x490 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x490 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoO, (1, 4, 0, 5), (5, 3, 2, 4)) * -1.0
    x491 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x491 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x490, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x491, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x491, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x491, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x491, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x491
    x492 = np.zeros((naocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x492 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovoO, (1, 4, 0, 5), (5, 2, 3, 4))
    x493 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x493 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x492, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x492
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x493, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x493, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x493
    x494 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x494 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x73, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x73
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x494, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x494, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x494, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x494, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    del x494
    x495 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x495 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 0, 2), (1, 3, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x495, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -2.0
    x496 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x496 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x495, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x495
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x496, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x496, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    del x496
    x497 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x497 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x71, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x71
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x497, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x497, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x497, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x497, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    del x497
    x498 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x498 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ooov, (4, 0, 5, 2), (1, 3, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x498, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * 2.0
    x499 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x499 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x498, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x498
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x499, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x499, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x499
    x500 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x500 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x501 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x501 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x500, (4, 5, 6, 7, 0, 1), (4, 5, 6, 7, 2, 3))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x501, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x501, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x501, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x501, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x501
    x502 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x502 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 0, 3), (1, 4, 5, 2)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x502, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * 2.0
    x503 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x503 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x502, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x502
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x503, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x503, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x503, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x503, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    del x503
    x504 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x504 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ooov, (4, 1, 5, 2), (3, 0, 4, 5)) * -1.0
    x505 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x505 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x504, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x504
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x505, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x505, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    del x505
    x506 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x506 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ooov, (4, 0, 5, 3), (1, 4, 5, 2)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x506, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -2.0
    x507 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x507 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x506, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x506
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x507, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x507, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x507, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x507, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    del x507
    x508 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x508 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 1, 2), (3, 0, 4, 5)) * -1.0
    x509 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x509 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x508, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x508
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x509, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x509, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x509
    x510 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x510 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ooov, (4, 1, 0, 5), (3, 4, 2, 5))
    x511 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x511 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x510, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x511, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x511, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x511, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x511, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x511, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x511, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x511, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x511, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x511
    x512 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x512 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvV, (1, 3, 4, 5), (5, 0, 2, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x512, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4)) * 2.0
    x513 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x513 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x512, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x512
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x513, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x513, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x513, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x513, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x513
    x514 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x514 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvV, (4, 3, 2, 5), (1, 5, 0, 4)) * -1.0
    x515 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x515 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x514, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x515, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x515, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x515, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x515, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x515
    x516 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x516 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvV, (1, 4, 3, 5), (5, 0, 2, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x516, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4)) * -2.0
    x517 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x517 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x516, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x516
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x517, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x517, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x517, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x517, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x517
    x518 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x518 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvV, (0, 4, 3, 5), (1, 5, 2, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x518, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * 2.0
    x519 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x519 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x518, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x518
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x519, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x519, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x519
    x520 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x520 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvV, (4, 3, 2, 5), (5, 0, 1, 4))
    x521 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x521 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x520, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x520
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x521, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x521, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x521
    x522 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x522 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvV, (0, 3, 4, 5), (1, 5, 2, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x522, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * -2.0
    x523 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x523 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x522, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x522
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x523, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x523, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x523
    x524 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x524 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x301, (4, 5, 6, 1, 3, 7), (4, 5, 6, 0, 2, 7))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x524, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x524, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x524, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x524, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x524
    x525 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x525 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x525, (4, 5, 6, 0, 2, 7), (6, 1, 4, 7, 3, 5)) * -2.0
    x526 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x526 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x525, (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 2, 7))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x526, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x526, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x526, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x526, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x526
    x527 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x527 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovvv, (0, 4, 5, 2), (1, 3, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x527, (4, 5, 2, 6), (0, 1, 4, 6, 3, 5)) * -2.0
    x528 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x528 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x527, (4, 5, 3, 6), (4, 5, 0, 1, 2, 6))
    del x527
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x528, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x528, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x528
    x529 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x529 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x80, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x80
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x529, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x529, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x529, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x529, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x529
    x530 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x530 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovvv, (0, 2, 4, 5), (1, 3, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x530, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * 2.0
    x531 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x531 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x530, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x530
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x531, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x531, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x531
    x532 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x532 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 3), (1, 0, 4, 5)) * -1.0
    x533 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x533 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x532, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x533, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x533, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x533, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x533, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x533, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x533, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x533, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x533, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x533
    x534 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x534 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovvv, (1, 2, 4, 5), (3, 0, 4, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x534, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4)) * -2.0
    x535 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x535 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x534, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x534
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x535, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x535, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x535, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x535, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x535
    x536 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x536 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovvv, (1, 4, 5, 2), (3, 0, 4, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x536, (4, 5, 2, 6), (5, 1, 0, 6, 3, 4)) * 2.0
    x537 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x537 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x536, (4, 5, 3, 6), (1, 4, 0, 5, 2, 6)) * -1.0
    del x536
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x537, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x537, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x537, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x537, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x537
    x538 = np.zeros((naocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x538 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvv, (0, 4, 5, 3), (1, 2, 4, 5)) * -1.0
    x539 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x539 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x538, (4, 5, 2, 6), (4, 3, 0, 1, 5, 6)) * -1.0
    del x538
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x539, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x539, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x539
    x540 = np.zeros((naocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x540 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvv, (0, 3, 4, 5), (1, 2, 4, 5)) * -1.0
    x541 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x541 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x540, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x540
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x541, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x541, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x541
    x542 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x542 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 7, 4, 0, 6, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x542, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x542, (4, 5, 6, 1, 7, 3), (6, 0, 4, 7, 2, 5)) * 4.0
    x543 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x543 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x542, (4, 5, 6, 1, 7, 3), (4, 5, 0, 6, 2, 7))
    del x542
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x543, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x543, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x543, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x543, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x543
    x544 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x544 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 1, 7), (5, 7, 4, 0, 6, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x544, (4, 5, 6, 1, 7, 3), (6, 0, 4, 7, 2, 5)) * -4.0
    x545 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x545 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x544, (4, 5, 6, 1, 7, 3), (4, 5, 0, 6, 2, 7))
    del x544
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x545, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x545, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x545, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x545, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x545
    x546 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x546 += einsum(x34, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5))
    del x34
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x546, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x546, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x546
    x547 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x547 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 0, 5, 6, 1, 7), (5, 7, 4, 2, 6, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x547, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x547, (4, 5, 6, 1, 7, 3), (6, 0, 4, 7, 2, 5)) * 12.0
    x548 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x548 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x547, (4, 5, 6, 1, 7, 3), (4, 5, 0, 6, 2, 7))
    del x547
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x548, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x548, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x548, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x548, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x548
    x549 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x549 += einsum(x105, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    del x105
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x549, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x549, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x549
    x550 = np.zeros((navir[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x550 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (2, 0, 4, 5))
    x551 = np.zeros((naocc[0], navir[0], navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x551 += einsum(x550, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 0, 7, 1, 4, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x551, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 2, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x551, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 2, 1)) * -2.0
    del x551
    x552 = np.zeros((navir[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x552 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 5), (2, 0, 4, 5))
    x553 = np.zeros((naocc[0], navir[0], navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x553 += einsum(x552, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 0, 7, 1, 4, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x553, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 2, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x553, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 2, 1)) * 2.0
    del x553
    x554 = np.zeros((navir[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x554 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oVov, (4, 5, 1, 3), (2, 5, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x554, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * 2.0
    x555 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x555 += einsum(x554, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x554
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x555, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x555, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x555
    x556 = np.zeros((navir[0], navir[0]), dtype=np.float64)
    x556 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oVov, (0, 4, 1, 3), (2, 4))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x556, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x556, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -2.0
    del x556
    x557 = np.zeros((naocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x557 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x558 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x558 += einsum(x557, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 1, 5, 6, 3, 7), (0, 5, 7, 4, 2, 6))
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x558, (0, 1, 2, 3, 4, 5), (3, 1, 0, 4, 5, 2)) * 2.0
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x558, (0, 1, 2, 3, 4, 5), (3, 1, 0, 5, 4, 2)) * -2.0
    del x558
    x559 = np.zeros((naocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x559 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 5), (0, 4, 2, 5))
    x560 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x560 += einsum(x559, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 1, 5, 6, 3, 7), (0, 5, 7, 4, 2, 6))
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x560, (0, 1, 2, 3, 4, 5), (3, 1, 0, 4, 5, 2)) * -2.0
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x560, (0, 1, 2, 3, 4, 5), (3, 1, 0, 5, 4, 2)) * 2.0
    del x560
    x561 = np.zeros((naocc[0], naocc[0]), dtype=np.float64)
    x561 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vOov, (2, 4, 1, 3), (0, 4))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x561, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x561, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    del x561
    x562 = np.zeros((naocc[0], naocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x562 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vOov, (4, 5, 1, 3), (0, 5, 2, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x562, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7)) * 2.0
    x563 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x563 += einsum(x562, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    del x562
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x563, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x563, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x563
    x564 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x564 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 3), (0, 2, 4, 5))
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sVa,sVfa)] += einsum(x564, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * 2.0
    x565 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x565 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 1, 5), (0, 2, 4, 5))
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sVa,sVfa)] += einsum(x565, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * -2.0
    x566 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=np.float64)
    x566 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 1, 3), (0, 4, 2, 5))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x566, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x566, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 2.0
    del x566
    x567 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x567 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 2, 4, 5, 3, 6), (4, 6, 5, 1))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x567, (4, 5, 6, 2), (0, 1, 4, 6, 5, 3)) * 4.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x567, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * -2.0
    x568 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x568 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x567, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x567
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x568, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x568, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x568
    x569 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x569 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 1, 3, 6), (5, 6, 4, 0))
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x569, (4, 5, 6, 0), (6, 4, 1, 2, 3, 5)) * 4.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x569, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -2.0
    x570 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x570 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x569, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x569
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x570, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x570, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    del x570
    x571 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x571 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x306, (4, 5, 6, 1, 7, 3), (4, 5, 0, 6, 2, 7))
    del x306
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x571, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x571, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x571, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x571, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x571
    x572 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x572 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x88, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x572, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x572, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x572, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x572, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x572
    x573 = np.zeros((navir[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x573 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovov, (1, 2, 4, 5), (3, 0, 4, 5)) * -1.0
    x574 = np.zeros((naocc[0], navir[0], navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x574 += einsum(x573, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 0, 7, 1, 4, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x574, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 2, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x574, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 2, 1)) * -4.0
    del x574
    x575 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x575 += einsum(v.aabb.vOov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 1, 5, 3, 6), (6, 4, 5, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x575, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4)) * -2.0
    x576 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x576 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x575, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x575
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x576, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x576, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x576, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x576, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x576
    x577 = np.zeros((naocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x577 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aabb.ovov, (0, 3, 4, 5), (1, 4, 2, 5)) * -1.0
    x578 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x578 += einsum(x577, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 1, 5, 6, 3, 7), (0, 5, 7, 4, 2, 6))
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x578, (0, 1, 2, 3, 4, 5), (3, 1, 0, 4, 5, 2)) * 4.0
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x578, (0, 1, 2, 3, 4, 5), (3, 1, 0, 5, 4, 2)) * -4.0
    del x578
    x579 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x579 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.vOov, (2, 4, 5, 6), (1, 4, 3, 0, 5, 6))
    x580 = np.zeros((naocc[0], navir[0], navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x580 += einsum(t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 1, 2, 3, 4, 5), x579, (6, 2, 7, 8, 1, 4), (6, 7, 5, 8, 0, 3))
    del x579
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x580, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 2, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x580, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 2, 1)) * -4.0
    del x580
    x581 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x581 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x88, (4, 5, 0, 6), (1, 4, 3, 5, 2, 6))
    del x88
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x581, (0, 1, 2, 3, 4, 5), (3, 1, 0, 4, 5, 2)) * 4.0
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x581, (0, 1, 2, 3, 4, 5), (3, 1, 0, 5, 4, 2)) * -4.0
    del x581
    x582 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x582 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 3, 4, 5))
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sVa,sVfa)] += einsum(x582, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * 4.0
    x583 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x583 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (0, 2, 4, 5, 1, 6), (4, 6, 5, 3)) * -1.0
    x584 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x584 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x583, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x584, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x584, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x584, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x584, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x584
    x585 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x585 += einsum(x113, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 7), (4, 7, 1, 0, 5, 6)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x585, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -3.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x585, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -3.0
    del x585
    x586 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x586 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 2, 5, 3, 1, 6), (5, 6, 4, 0)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x586, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -6.0
    x587 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x587 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x586, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x586
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x587, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x587, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -12.0
    del x587
    x588 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x588 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x312, (4, 5, 6, 1, 7, 3), (4, 5, 0, 6, 2, 7))
    del x312
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x588, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x588, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x588, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x588, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 12.0
    del x588
    x589 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x589 += einsum(x37, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x589, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x589, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x589, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x589, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x589
    x590 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x590 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 2, 5, 6, 1, 7), (5, 7, 4, 0, 6, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x590, (4, 5, 6, 0, 7, 2), (6, 1, 4, 7, 3, 5)) * -6.0
    x591 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x591 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x590, (4, 5, 6, 1, 7, 3), (4, 5, 0, 6, 2, 7))
    del x590
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x591, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x591, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x591, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x591, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -12.0
    del x591
    x592 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x592 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 6, 3, 1, 7), (6, 7, 4, 5, 0, 2)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x592, (4, 5, 6, 7, 0, 1), (6, 7, 4, 2, 3, 5)) * 6.0
    x593 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x593 += einsum(x111, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x593, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x593, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x593, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x593, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x593
    x594 = np.zeros((naocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x594 += einsum(v.aaaa.ovoV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 0, 4, 5, 6, 3), (4, 5, 6, 1))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x594, (4, 5, 6, 2), (0, 1, 4, 5, 6, 3)) * 6.0
    del x594
    x595 = np.zeros((naocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x595 += einsum(v.aaaa.ovoV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 0, 4, 5, 6, 3), (4, 5, 6, 1))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x595, (4, 5, 6, 2), (0, 1, 4, 5, 6, 3)) * 6.0
    del x595
    x596 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x596 += einsum(v.aaaa.ovoV, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 2, 5, 6, 1, 3), (5, 4, 0, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x596, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * 6.0
    x597 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x597 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x596, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x596
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x597, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x597, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x597, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x597, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -12.0
    del x597
    x598 = np.zeros((navir[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x598 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoV, (1, 2, 4, 5), (3, 5, 0, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x598, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * 4.0
    x599 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x599 += einsum(x598, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x598
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x599, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x599, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -12.0
    del x599
    x600 = np.zeros((navir[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x600 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoV, (4, 2, 1, 5), (3, 5, 0, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x600, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * -4.0
    x601 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x601 += einsum(x600, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x600
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x601, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x601, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 12.0
    del x601
    x602 = np.zeros((navir[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x602 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoV, (1, 4, 0, 5), (3, 5, 2, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x602, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 3, 7, 1), (4, 5, 6, 2, 7, 0)) * 2.0
    x603 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x603 += einsum(x602, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 6, 7, 3, 1), (6, 0, 4, 5, 2, 7))
    del x602
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x603, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x603, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x603
    x604 = np.zeros((navir[0], navir[0]), dtype=np.float64)
    x604 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoV, (1, 2, 0, 4), (3, 4)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x604, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x604, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -2.0
    del x604
    x605 = np.zeros((navir[0], navir[0]), dtype=np.float64)
    x605 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoV, (1, 2, 0, 4), (3, 4)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x605, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x605, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -2.0
    del x605
    x606 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x606 += einsum(v.aaaa.ovvO, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 0, 3, 5, 2, 6), (6, 4, 5, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x606, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4)) * 3.0
    x607 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x607 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x606, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x606
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x607, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x607, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x607, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x607, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x607
    x608 = np.zeros((naocc[0], naocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x608 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvO, (4, 3, 2, 5), (1, 5, 0, 4)) * -1.0
    x609 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x609 += einsum(x608, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 1, 5, 6, 7), (0, 7, 2, 4, 5, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x609, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 3.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x609, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 3.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x609, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -3.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x609, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -3.0
    del x609
    x610 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x610 += einsum(v.aaaa.ovvO, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 0, 3, 5, 1, 6), (6, 4, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x610, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4)) * -3.0
    x611 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x611 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x610, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x610
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x611, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x611, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x611, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x611, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x611
    x612 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x612 += einsum(v.aaaa.ovvO, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 3, 2, 1, 6), (6, 4, 5, 0))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x612, (4, 5, 6, 0), (6, 5, 1, 2, 3, 4)) * -12.0
    del x612
    x613 = np.zeros((naocc[0], naocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x613 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvO, (0, 3, 4, 5), (1, 5, 2, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x613, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7)) * 4.0
    x614 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x614 += einsum(x613, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    del x613
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x614, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x614, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 12.0
    del x614
    x615 = np.zeros((naocc[0], naocc[0]), dtype=np.float64)
    x615 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvO, (0, 3, 2, 4), (1, 4)) * -1.0
    x616 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x616 += einsum(x615, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 1, 4, 5, 6), (0, 6, 2, 3, 4, 5))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x616, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x616, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x616
    x617 = np.zeros((naocc[0], naocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x617 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovvO, (0, 4, 3, 5), (1, 5, 2, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x617, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7)) * -4.0
    x618 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x618 += einsum(x617, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    del x617
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x618, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x618, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -12.0
    del x618
    x619 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x619 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovOV, (4, 2, 5, 6), (1, 5, 3, 6, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 1, 2, 3, 4, 5), x619, (6, 2, 7, 5, 8, 0), (8, 1, 6, 3, 4, 7)) * -2.0
    x620 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x620 += einsum(t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (0, 1, 2, 3, 4, 5), x619, (6, 2, 7, 5, 8, 1), (6, 7, 8, 0, 3, 4))
    del x619
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x620, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x620, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x620
    x621 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x621 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.oVvO, (4, 5, 2, 6), (1, 6, 3, 5, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 1, 2, 3, 4, 5), x621, (6, 2, 7, 5, 8, 0), (8, 1, 6, 3, 4, 7)) * 2.0
    x622 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x622 += einsum(t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (0, 1, 2, 3, 4, 5), x621, (6, 2, 7, 5, 8, 1), (6, 7, 8, 0, 3, 4))
    del x621
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x622, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x622, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x622
    x623 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x623 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x94, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x94
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x623, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 12.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x623, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -12.0
    del x623
    x624 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=np.float64)
    x624 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovOV, (0, 2, 4, 5), (1, 4, 3, 5))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x624, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 12.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x624, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 4.0
    del x624
    x625 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=np.float64)
    x625 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.oVvO, (0, 4, 2, 5), (1, 5, 3, 4))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x625, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -12.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x625, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -4.0
    del x625
    x626 = np.zeros((naocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x626 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovoO, (2, 1, 3, 4), (4, 0, 3, 2))
    x627 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x627 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x626, (2, 3, 4, 0), (2, 3, 4, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x627, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2))
    x628 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x628 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x627, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x627
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x628, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x628, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x628, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x628, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x628
    x629 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x629 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x626, (2, 3, 0, 4), (2, 3, 4, 1))
    del x626
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x629, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -1.0
    x630 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x630 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x629, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x629
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x630, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x630, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x630, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x630, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x630
    x631 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x631 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovoO, (4, 2, 5, 6), (6, 3, 0, 1, 5, 4)) * -1.0
    x632 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x632 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x631, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x631
    x633 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x633 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x632, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x632
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x633, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x633, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x633
    x634 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x634 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x69, (4, 5, 6, 0), (1, 3, 4, 6, 5, 2))
    x635 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x635 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x634, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x634
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x635, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x635, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x635, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x635, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x635
    x636 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x636 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x69, (4, 5, 0, 6), (1, 3, 4, 5, 6, 2))
    del x69
    x637 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x637 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x636, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x636
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x637, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x637, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x637, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x637, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x637
    x638 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x638 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x500, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x500
    x639 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x639 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x638, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x638
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x639, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x639, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x639, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x639, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x639
    x640 = np.zeros((navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x640 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvV, (2, 1, 3, 4), (4, 0, 2, 3))
    x641 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x641 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x640, (2, 3, 4, 1), (2, 0, 3, 4))
    x642 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x642 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x641, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x641
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x642, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x642, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x642
    x643 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x643 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x117, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x117
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x643, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x643, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x643, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x643, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x643
    x644 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x644 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x640, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2)) * -1.0
    x645 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x645 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x644, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x644
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x645, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x645, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x645, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x645, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x645
    x646 = np.zeros((navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x646 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvV, (2, 3, 1, 4), (4, 0, 2, 3))
    x647 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x647 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x646, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2)) * -1.0
    x648 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x648 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x647, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x647
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x648, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x648, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x648, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x648, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x648
    x649 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x649 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x301, (2, 3, 4, 5, 1, 6), (2, 3, 0, 4, 5, 6))
    del x301
    x650 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x650 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x649, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x649
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x650, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x650, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x650, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x650, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x650
    x651 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x651 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x525, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x525
    x652 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x652 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x651, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x651
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x652, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x652, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x652, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x652, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x652
    x653 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0]), dtype=np.float64)
    x653 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.ovoO, (0, 2, 3, 4), (4, 1, 3, 2))
    x654 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x654 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x653, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x654, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -1.0
    x655 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x655 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x654, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x654
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x655, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x655, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x655
    x656 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0]), dtype=np.float64)
    x656 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.ovoO, (2, 3, 0, 4), (4, 1, 2, 3))
    x657 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x657 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x656, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x657, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5))
    x658 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x658 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x657, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x657
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x658, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x658, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x658
    x659 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x659 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x656, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    x660 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x660 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x659, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x659
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x660, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x660, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x660
    x661 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x661 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x653, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    x662 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x662 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x661, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x661
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x662, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x662, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x662
    x663 = np.zeros((navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x663 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.ooov, (2, 0, 3, 4), (1, 2, 3, 4))
    x664 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x664 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x663, (2, 3, 4, 1), (2, 0, 3, 4))
    x665 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x665 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x664, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x664
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x665, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x665, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x665
    x666 = np.zeros((navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x666 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.ooov, (2, 3, 0, 4), (1, 2, 3, 4))
    x667 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x667 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x666, (2, 3, 4, 1), (2, 0, 4, 3))
    x668 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x668 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x667, (4, 5, 0, 6), (1, 4, 5, 6, 2, 3)) * -1.0
    del x667
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x668, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x668, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x668
    x669 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x669 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x666, (4, 5, 6, 3), (1, 4, 0, 6, 5, 2)) * -1.0
    x670 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x670 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x669, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x669
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x670, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x670, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x670, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x670, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x670
    x671 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x671 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x663, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    x672 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x672 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x671, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x671
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x672, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x672, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x672, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x672, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x672
    x673 = np.zeros((navir[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x673 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.ovvv, (0, 2, 3, 4), (1, 2, 3, 4))
    x674 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x674 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x673, (2, 1, 3, 4), (2, 0, 4, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x674, (4, 5, 2, 6), (5, 1, 0, 6, 3, 4))
    x675 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x675 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x674, (4, 5, 3, 6), (1, 4, 5, 0, 2, 6)) * -1.0
    del x674
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x675, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x675, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x675, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x675, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x675
    x676 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x676 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x673, (2, 3, 4, 1), (2, 0, 3, 4))
    del x673
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x676, (4, 5, 2, 6), (5, 1, 0, 6, 3, 4)) * -1.0
    x677 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x677 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x676, (4, 5, 3, 6), (1, 4, 5, 0, 2, 6)) * -1.0
    del x676
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x677, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x677, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x677, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x677, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x677
    x678 = np.zeros((naocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x678 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x679 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x679 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x678, (2, 3, 0, 4), (2, 3, 4, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x679, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2))
    x680 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x680 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x679, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x679
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x680, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x680, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x680, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x680, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x680
    x681 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x681 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x678, (2, 3, 4, 0), (2, 4, 3, 1))
    del x678
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x681, (4, 0, 5, 6), (5, 1, 4, 6, 3, 2)) * -1.0
    x682 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x682 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x681, (4, 1, 5, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x681
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x682, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x682, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x682, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x682, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x682
    x683 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0]), dtype=np.float64)
    x683 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.ovvV, (2, 3, 1, 4), (0, 4, 2, 3))
    x684 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x684 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x683, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x684, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -1.0
    x685 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x685 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x684, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x684
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x685, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x685, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x685
    x686 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0]), dtype=np.float64)
    x686 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.ovvV, (2, 1, 3, 4), (0, 4, 2, 3))
    x687 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x687 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x686, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x687, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5))
    x688 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x688 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x687, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x687
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x688, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x688, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x688
    x689 = np.zeros((naocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x689 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x690 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x690 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x689, (2, 3, 1, 4), (2, 0, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x690, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2))
    x691 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x691 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x690, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    del x690
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x691, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x691, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x691, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x691, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x691
    x692 = np.zeros((naocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x692 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x693 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x693 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x692, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x693, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -1.0
    x694 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x694 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x693, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    del x693
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x694, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x694, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x694, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x694, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x694
    x695 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x695 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x686, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    x696 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x696 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x695, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x695
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x696, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x696, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x696
    x697 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x697 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x683, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    x698 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x698 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x697, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x697
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x698, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x698, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x698
    x699 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x699 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x692, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x692
    x700 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x700 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x699, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x699
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x700, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x700, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x700
    x701 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x701 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x689, (4, 5, 2, 6), (4, 3, 0, 1, 5, 6)) * -1.0
    del x689
    x702 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x702 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x701, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x701
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x702, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x702, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x702
    x703 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x703 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t1.aa[np.ix_(sOa,sva)], (2, 3), v.aaaa.ooov, (4, 0, 5, 3), (2, 1, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x703, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -1.0
    x704 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x704 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x703, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x703
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x704, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x704, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x704
    x705 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x705 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t1.aa[np.ix_(sOa,sva)], (2, 3), v.aaaa.ooov, (4, 5, 0, 3), (2, 1, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x705, (4, 5, 0, 6), (6, 1, 4, 2, 3, 5))
    x706 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x706 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x705, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3))
    del x705
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x706, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x706, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x706
    x707 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x707 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t1.aa[np.ix_(sOa,sva)], (2, 3), v.aaaa.ovvv, (0, 3, 4, 5), (2, 1, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x707, (4, 5, 2, 6), (0, 1, 4, 6, 3, 5)) * -1.0
    x708 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x708 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x707, (4, 5, 3, 6), (4, 5, 0, 1, 2, 6))
    del x707
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x708, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x708, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x708
    x709 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x709 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t1.aa[np.ix_(sOa,sva)], (2, 3), v.aaaa.ovvv, (0, 4, 5, 3), (2, 1, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x709, (4, 5, 2, 6), (0, 1, 4, 6, 3, 5))
    x710 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x710 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x709, (4, 5, 3, 6), (4, 5, 0, 1, 2, 6))
    del x709
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x710, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x710, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x710
    x711 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x711 += einsum(x33, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 7, 0, 4, 1, 6))
    x712 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x712 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x711, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x711
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x712, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x712, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x712, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x712, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x712
    x713 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x713 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.ovov, (0, 2, 3, 4), (1, 3, 2, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x713, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * -2.0
    x714 = np.zeros((navir[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x714 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x713, (2, 3, 1, 4), (2, 0, 3, 4))
    x715 = np.zeros((naocc[0], navir[0], navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x715 += einsum(x714, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 0, 7, 1, 4, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x715, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 2, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sVa,sVfa)] += einsum(x715, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 2, 1)) * 2.0
    del x715
    x716 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x716 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.ovov, (2, 1, 3, 4), (0, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x716, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    x717 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x717 += einsum(x716, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (0, 5, 7, 4, 1, 6))
    x718 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x718 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x717, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x717
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x718, (0, 1, 2, 3, 4, 5), (3, 1, 0, 4, 5, 2)) * -2.0
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sva,sVfa)] += einsum(x718, (0, 1, 2, 3, 4, 5), (3, 1, 0, 5, 4, 2)) * 2.0
    del x718
    x719 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x719 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t1.aa[np.ix_(sOa,sva)], (2, 3), v.aabb.ovov, (0, 3, 4, 5), (2, 1, 4, 5))
    t3new_aaaaaa[np.ix_(soa,sOa,sOfa,sva,sVa,sVfa)] += einsum(x719, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * -2.0
    x720 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x720 += einsum(x121, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 7), (4, 7, 1, 0, 5, 6)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x720, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -3.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x720, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -3.0
    del x720
    x721 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x721 += einsum(x35, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 1, 5, 6, 3, 7), (5, 7, 0, 4, 2, 6))
    x722 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x722 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x721, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x721
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x722, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x722, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x722, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x722, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x722
    x723 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x723 += einsum(x35, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 2, 5, 6, 3, 7), (5, 7, 0, 4, 1, 6))
    x724 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x724 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x723, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x723
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x724, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x724, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x724, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x724, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x724
    x725 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x725 += einsum(x39, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5))
    del x39
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x725, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x725, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x725
    x726 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x726 += einsum(x40, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5))
    del x40
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x726, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x726, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x726
    x727 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x727 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x592, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x592
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x727, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * 6.0
    del x727
    x728 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x728 += einsum(x11, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    x729 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x729 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x728, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x728
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x729, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x729, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x729
    x730 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x730 += einsum(x12, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    x731 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x731 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x730, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x730
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x731, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x731, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x731
    x732 = np.zeros((navir[0], navir[0], nocc[0], nvir[0]), dtype=np.float64)
    x732 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.ovoV, (0, 2, 3, 4), (1, 4, 3, 2))
    x733 = np.zeros((navir[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x733 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x732, (2, 3, 4, 1), (2, 3, 0, 4))
    del x732
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x733, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * -2.0
    x734 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x734 += einsum(x733, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x733
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x734, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x734, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x734
    x735 = np.zeros((navir[0], navir[0], nocc[0], nvir[0]), dtype=np.float64)
    x735 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.ovoV, (2, 3, 0, 4), (1, 4, 2, 3))
    x736 = np.zeros((navir[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x736 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x735, (2, 3, 4, 1), (2, 3, 0, 4))
    del x735
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x736, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * 2.0
    x737 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x737 += einsum(x736, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x736
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x737, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x737, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x737
    x738 = np.zeros((navir[0], navir[0], nocc[0], nvir[0]), dtype=np.float64)
    x738 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aaaa.ovoV, (2, 3, 0, 4), (1, 4, 2, 3))
    x739 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x739 += einsum(x738, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 6, 7, 3, 1), (6, 0, 4, 5, 2, 7))
    x740 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x740 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x739, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x739
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x740, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x740, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x740
    x741 = np.zeros((navir[0], nocc[0]), dtype=np.float64)
    x741 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovoV, (0, 1, 2, 3), (3, 2))
    x742 = np.zeros((navir[0], navir[0]), dtype=np.float64)
    x742 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x741, (2, 0), (1, 2))
    del x741
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x742, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x742, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -2.0
    del x742
    x743 = np.zeros((navir[0], nocc[0]), dtype=np.float64)
    x743 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovoV, (2, 1, 0, 3), (3, 2))
    x744 = np.zeros((navir[0], navir[0]), dtype=np.float64)
    x744 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x743, (2, 0), (1, 2))
    del x743
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x744, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x744, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 2.0
    del x744
    x745 = np.zeros((naocc[0], naocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x745 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.ovvO, (2, 3, 1, 4), (0, 4, 2, 3))
    x746 = np.zeros((naocc[0], naocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x746 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x745, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x746, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7))
    x747 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x747 += einsum(x746, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 1, 5, 6, 7), (0, 7, 2, 4, 5, 6))
    del x746
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x747, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 3.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x747, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -3.0
    del x747
    x748 = np.zeros((naocc[0], naocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x748 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.ovvO, (2, 1, 3, 4), (0, 4, 2, 3))
    x749 = np.zeros((naocc[0], naocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x749 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x748, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x749, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * -1.0
    x750 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x750 += einsum(x749, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 3, 1, 5, 6, 7), (0, 7, 2, 4, 5, 6))
    del x749
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x750, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -3.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x750, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 3.0
    del x750
    x751 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x751 += einsum(x748, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    x752 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x752 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x751, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x751
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x752, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x752, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x752
    x753 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x753 += einsum(x745, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    x754 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x754 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x753, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x753
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x754, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x754, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x754
    x755 = np.zeros((naocc[0], nvir[0]), dtype=np.float64)
    x755 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvO, (0, 1, 2, 3), (3, 2))
    x756 = np.zeros((naocc[0], naocc[0]), dtype=np.float64)
    x756 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x755, (2, 1), (0, 2))
    del x755
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x756, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x756, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    del x756
    x757 = np.zeros((naocc[0], nvir[0]), dtype=np.float64)
    x757 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aaaa.ovvO, (0, 2, 1, 3), (3, 2))
    x758 = np.zeros((naocc[0], naocc[0]), dtype=np.float64)
    x758 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x757, (2, 1), (0, 2))
    del x757
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x758, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * 6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x758, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * 2.0
    del x758
    x759 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=np.float64)
    x759 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.ovOV, (2, 1, 3, 4), (0, 3, 4, 2))
    x760 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=np.float64)
    x760 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x759, (2, 3, 4, 0), (2, 3, 1, 4))
    del x759
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x760, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x760, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -2.0
    del x760
    x761 = np.zeros((naocc[0], naocc[0], navir[0], nocc[0]), dtype=np.float64)
    x761 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aaaa.oVvO, (2, 3, 1, 4), (0, 4, 3, 2))
    x762 = np.zeros((naocc[0], naocc[0], navir[0], navir[0]), dtype=np.float64)
    x762 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x761, (2, 3, 4, 0), (2, 3, 1, 4))
    del x761
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x762, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x762, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 2.0
    del x762
    x763 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x763 += einsum(x41, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5))
    del x41
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x763, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x763, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x763
    x764 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x764 += einsum(x15, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    x765 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x765 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x764, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x764
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x765, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x765, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x765
    x766 = np.zeros((navir[0], nocc[0]), dtype=np.float64)
    x766 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.oVov, (2, 3, 0, 1), (3, 2))
    x767 = np.zeros((navir[0], navir[0]), dtype=np.float64)
    x767 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x766, (2, 0), (1, 2))
    del x766
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x767, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x767, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -2.0
    del x767
    x768 = np.zeros((naocc[0], nvir[0]), dtype=np.float64)
    x768 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.vOov, (2, 3, 0, 1), (3, 2))
    x769 = np.zeros((naocc[0], naocc[0]), dtype=np.float64)
    x769 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), x768, (2, 1), (0, 2))
    del x768
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x769, (0, 1), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -6.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x769, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    del x769
    x770 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x770 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x119, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x119
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x770, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x770, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x770, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x770, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x770
    x771 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x771 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x771, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    x772 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x772 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x771, (4, 5, 6, 7, 1, 3), (4, 5, 6, 0, 7, 2))
    x773 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x773 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x772, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x772
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x773, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x773, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x773, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x773, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x773
    x774 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x774 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x713, (4, 1, 5, 3), (4, 0, 2, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x774, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4)) * -1.0
    x775 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x775 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x774, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x774
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x775, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x775, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x775, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x775, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x775
    x776 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x776 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x716, (4, 5, 1, 3), (4, 0, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x776, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -1.0
    x777 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x777 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x776, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x776
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x777, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x777, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x777, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x777, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x777
    x778 = np.zeros((navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x778 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (2, 0, 4, 5))
    x779 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x779 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x778, (2, 3, 4, 1), (2, 0, 3, 4))
    x780 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x780 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x779, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x779
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x780, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x780, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x780
    x781 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x781 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x778, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    x782 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x782 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x781, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x781
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x782, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x782, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x782, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x782, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x782
    x783 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x783 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x716, (4, 5, 1, 3), (4, 2, 0, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x783, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5))
    x784 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x784 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x783, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x783
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x784, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x784, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x784
    x785 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x785 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x33, (4, 5, 1, 3), (0, 4, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x785, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2))
    x786 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x786 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x785, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x785
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x786, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x786, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x786, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x786, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x786
    x787 = np.zeros((naocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x787 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 4, 2, 5))
    x788 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x788 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x787, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x787
    x789 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x789 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x788, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x788
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x789, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x789, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x789
    x790 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x790 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x713, (4, 1, 5, 3), (0, 4, 2, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x790, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5))
    x791 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x791 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x790, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x790
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x791, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x791, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x791
    x792 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0]), dtype=np.float64)
    x792 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 3), (0, 2, 4, 5))
    x793 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x793 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x792, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x793, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -1.0
    x794 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x794 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x793, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x793
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x794, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x794, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x794
    x795 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x795 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x792, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    x796 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x796 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x795, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x795
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x796, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x796, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x796
    x797 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x797 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x125, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x125
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x797, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x797, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x797, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x797, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    del x797
    x798 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0]), dtype=np.float64)
    x798 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 2), (1, 3, 4, 5))
    x799 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x799 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x798, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x799, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -2.0
    x800 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x800 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x799, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x799
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x800, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x800, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x800
    x801 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x801 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x123, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x123
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x801, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x801, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x801, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x801, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    del x801
    x802 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0]), dtype=np.float64)
    x802 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 0, 5), (1, 3, 4, 5))
    x803 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x803 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x802, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x803, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * 2.0
    x804 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x804 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x803, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x803
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x804, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x804, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    del x804
    x805 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x805 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x361, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    x806 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x806 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x805, (4, 5, 6, 7, 0, 1), (4, 5, 6, 7, 2, 3))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x806, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x806, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x806, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x806, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x806
    x807 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x807 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x35, (4, 0, 5, 3), (1, 4, 5, 2)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x807, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * 2.0
    x808 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x808 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x807, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x807
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x808, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x808, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x808, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x808, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    del x808
    x809 = np.zeros((navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x809 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 1, 5), (3, 0, 4, 5)) * -1.0
    x810 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x810 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x809, (2, 3, 4, 1), (2, 0, 3, 4))
    x811 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x811 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x810, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x810
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x811, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x811, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x811
    x812 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x812 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x35, (4, 5, 0, 3), (1, 4, 5, 2)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x812, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -2.0
    x813 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x813 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x812, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x812
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x813, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x813, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x813, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x813, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    del x813
    x814 = np.zeros((navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x814 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 1, 2), (3, 0, 4, 5)) * -1.0
    x815 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x815 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x814, (2, 3, 4, 1), (2, 0, 3, 4))
    x816 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x816 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x815, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x815
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x816, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x816, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    del x816
    x817 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x817 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x35, (4, 0, 1, 5), (3, 4, 2, 5)) * -1.0
    x818 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x818 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x817, (4, 5, 6, 3), (1, 4, 5, 0, 2, 6))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x818, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x818, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x818, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x818, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x818, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x818, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x818, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x818, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x818
    x819 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x819 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x361, (4, 5, 6, 7, 1, 3), (4, 5, 6, 0, 7, 2))
    x820 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x820 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x819, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x819
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x820, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x820, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x820, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x820, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x820
    x821 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x821 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x361, (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 7, 2))
    x822 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x822 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x821, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x821
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x822, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x822, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x822, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x822, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x822
    x823 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x823 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x798, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    x824 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x824 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x823, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x823
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x824, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x824, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x824
    x825 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x825 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x113, (4, 5, 0, 6), (1, 3, 4, 5, 6, 2)) * -1.0
    del x113
    x826 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x826 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x825, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x825
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x826, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x826, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x826, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x826, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x826
    x827 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x827 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x802, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    x828 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x828 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x827, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x827
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x828, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x828, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x828
    x829 = np.zeros((naocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x829 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 5, 3), (1, 0, 4, 5)) * -1.0
    x830 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x830 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x829, (2, 3, 4, 0), (2, 3, 4, 1))
    del x829
    x831 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x831 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x830, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x831, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x831, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x831, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x831, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x831, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x831, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x831, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x831, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x831
    x832 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x832 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x809, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    x833 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x833 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x832, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x832
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x833, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x833, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x833, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x833, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x833
    x834 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x834 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x814, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    x835 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x835 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x834, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x834
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x835, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x835, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x835, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x835, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x835
    x836 = np.zeros((naocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x836 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 3), (1, 4, 2, 5)) * -1.0
    x837 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x837 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x836, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x836
    x838 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x838 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x837, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x837
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x838, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x838, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x838
    x839 = np.zeros((naocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x839 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aaaa.ovov, (4, 3, 0, 5), (1, 4, 2, 5)) * -1.0
    x840 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x840 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x839, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x839
    x841 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x841 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x840, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x840
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x841, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x841, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x841
    x842 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x842 += einsum(x11, (0, 1), t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (2, 3, 1, 4), (3, 4, 2, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x842, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -2.0
    x843 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x843 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x842, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x842
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x843, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x843, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x843
    x844 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x844 += einsum(x12, (0, 1), t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (2, 3, 1, 4), (3, 4, 2, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x844, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * 2.0
    x845 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x845 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x844, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x844
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x845, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x845, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x845
    x846 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x846 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x131, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x131
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x846, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x846, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x846
    x847 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x847 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x133, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x133
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x847, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x847, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x847
    x848 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x848 += einsum(x11, (0, 1), t2.aaaa[np.ix_(soa,sOa,sva,sva)], (2, 3, 4, 1), (3, 2, 0, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x848, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * 2.0
    x849 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x849 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x848, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x848
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x849, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x849, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x849, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x849, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x849
    x850 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x850 += einsum(x12, (0, 1), t2.aaaa[np.ix_(soa,sOa,sva,sva)], (2, 3, 4, 1), (3, 2, 0, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x850, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -2.0
    x851 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x851 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x850, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x850
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x851, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x851, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x851, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x851, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x851
    x852 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x852 += einsum(x11, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sVa)], (2, 3, 1, 4), (4, 2, 3, 0)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x852, (4, 5, 6, 0), (6, 5, 1, 2, 3, 4)) * 4.0
    del x852
    x853 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x853 += einsum(x12, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sVa)], (2, 3, 1, 4), (4, 2, 3, 0)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x853, (4, 5, 6, 0), (6, 5, 1, 2, 3, 4)) * -4.0
    del x853
    x854 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x854 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x375, (4, 1, 5, 3), (4, 0, 2, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x854, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4)) * 2.0
    x855 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x855 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x854, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x854
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x855, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x855, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x855, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x855, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x855
    x856 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x856 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x375, (4, 5, 3, 2), (1, 4, 0, 5))
    x857 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x857 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x856, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x857, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x857, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x857, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x857, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x857
    x858 = np.zeros((navir[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x858 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x375, (4, 1, 3, 5), (4, 0, 2, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x858, (4, 5, 6, 2), (5, 1, 0, 6, 3, 4)) * -2.0
    x859 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x859 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x858, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x858
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x859, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x859, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x859, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x859, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x859
    x860 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x860 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x375, (4, 0, 3, 5), (1, 4, 2, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x860, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * 2.0
    x861 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x861 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x860, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x860
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x861, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x861, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x861
    x862 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x862 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x375, (4, 5, 2, 3), (4, 0, 1, 5))
    x863 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x863 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x862, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3))
    del x862
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x863, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x863, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x863
    x864 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x864 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x375, (4, 0, 5, 3), (1, 4, 2, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x864, (4, 5, 6, 2), (0, 1, 4, 6, 3, 5)) * -2.0
    x865 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x865 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x864, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x864
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x865, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x865, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x865
    x866 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x866 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x389, (4, 5, 1, 3), (4, 0, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x866, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * 2.0
    x867 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x867 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x866, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x866
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x867, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x867, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x867, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x867, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x867
    x868 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x868 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x389, (4, 1, 5, 2), (4, 3, 0, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x868, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * 2.0
    x869 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x869 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x868, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x868
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x869, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x869, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x869
    x870 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x870 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x389, (4, 1, 5, 3), (4, 0, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x870, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -2.0
    x871 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x871 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x870, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x870
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x871, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x871, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x871, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x871, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x871
    x872 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x872 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x389, (4, 5, 1, 2), (4, 3, 0, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x872, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -2.0
    x873 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x873 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x872, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x872
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x873, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x873, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x873
    x874 = np.zeros((naocc[0], navir[0], nvir[0], nvir[0]), dtype=np.float64)
    x874 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x389, (4, 0, 1, 5), (4, 3, 2, 5)) * -1.0
    x875 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x875 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x874, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x875, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x875, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x875, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x875, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x875
    x876 = np.zeros((naocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x876 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x389, (4, 1, 0, 5), (4, 2, 3, 5)) * -1.0
    x877 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x877 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x876, (4, 5, 6, 2), (4, 3, 0, 1, 6, 5)) * -1.0
    del x876
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x877, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x877, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x877
    x878 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x878 += einsum(x15, (0, 1), t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (2, 3, 1, 4), (3, 4, 2, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x878, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -2.0
    x879 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x879 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x878, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x878
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x879, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x879, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x879
    x880 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x880 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x136, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x136
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x880, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x880, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x880
    x881 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x881 += einsum(x15, (0, 1), t2.aaaa[np.ix_(soa,sOa,sva,sva)], (2, 3, 4, 1), (3, 2, 0, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x881, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * 2.0
    x882 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x882 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x881, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x881
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x882, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x882, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x882, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x882, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x882
    x883 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x883 += einsum(x15, (0, 1), t2.aaaa[np.ix_(soa,soa,sva,sVa)], (2, 3, 1, 4), (4, 2, 3, 0)) * -1.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x883, (4, 5, 6, 0), (6, 5, 1, 2, 3, 4)) * 4.0
    del x883
    x884 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x884 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x121, (4, 5, 0, 6), (1, 3, 5, 4, 6, 2))
    del x121
    x885 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x885 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x884, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x884
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x885, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x885, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x885, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x885, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x885
    x886 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x886 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x805, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x805
    x887 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x887 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x886, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x886
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x887, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x887, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x887, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x887, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x887
    x888 = np.zeros((navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x888 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x375, (2, 3, 1, 4), (2, 0, 3, 4))
    x889 = np.zeros((navir[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x889 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x888, (2, 3, 4, 1), (2, 3, 0, 4))
    x890 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x890 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x889, (4, 5, 6, 0), (1, 4, 6, 5, 2, 3)) * -1.0
    del x889
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x890, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x890, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x890
    x891 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x891 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x888, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2)) * -1.0
    x892 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x892 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x891, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x891
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x892, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x892, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x892, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x892, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x892
    x893 = np.zeros((navir[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x893 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x375, (2, 3, 4, 1), (2, 0, 3, 4))
    x894 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x894 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x893, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2)) * -1.0
    x895 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x895 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x894, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x894
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x895, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x895, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x895, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x895, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x895
    x896 = np.zeros((naocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x896 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x389, (2, 3, 4, 1), (2, 0, 4, 3))
    x897 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x897 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x896, (2, 3, 4, 0), (2, 3, 4, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x897, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2))
    x898 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x898 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x897, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x897
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x898, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x898, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x898, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x898, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x898
    x899 = np.zeros((naocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x899 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x896, (2, 3, 0, 4), (2, 3, 4, 1))
    del x896
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x899, (4, 5, 0, 6), (5, 1, 4, 6, 3, 2)) * -1.0
    x900 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x900 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x899, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x899
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x900, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x900, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x900, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x900, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x900
    x901 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nocc[0]), dtype=np.float64)
    x901 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x389, (4, 5, 6, 2), (4, 3, 0, 1, 6, 5)) * -1.0
    x902 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x902 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x901, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x901
    x903 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x903 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x902, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x902
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x903, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x903, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x903
    x904 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0]), dtype=np.float64)
    x904 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t1.aa[np.ix_(sOa,sva)], (2, 3), v.aaaa.ovov, (4, 3, 0, 5), (2, 1, 4, 5))
    x905 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x905 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x904, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x905, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5)) * -1.0
    x906 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x906 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x905, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x905
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x906, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x906, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x906
    x907 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0]), dtype=np.float64)
    x907 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t1.aa[np.ix_(sOa,sva)], (2, 3), v.aaaa.ovov, (4, 5, 0, 3), (2, 1, 4, 5))
    x908 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0]), dtype=np.float64)
    x908 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x907, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x908, (4, 5, 6, 0), (6, 1, 4, 2, 3, 5))
    x909 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x909 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x908, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x908
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x909, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x909, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x909
    x910 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x910 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x904, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    x911 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x911 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x910, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x910
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x911, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x911, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x911
    x912 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x912 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x907, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    x913 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x913 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x912, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x912
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x913, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)] += einsum(x913, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x913
    x914 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x914 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.vvov, (4, 2, 5, 6), (1, 3, 0, 5, 4, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x914, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x914, (4, 5, 6, 1, 7, 3), (0, 6, 4, 2, 7, 5)) * 2.0
    x915 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x915 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.ovvv, (4, 5, 6, 2), (1, 3, 4, 0, 5, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x915, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x915, (4, 5, 1, 6, 3, 7), (6, 0, 4, 7, 2, 5)) * 4.0
    x916 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x916 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 4, 5, 3, 6, 7), (5, 7, 4, 0, 6, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x916, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x916, (4, 5, 6, 1, 7, 3), (0, 6, 4, 2, 7, 5)) * 4.0
    del x916
    x917 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x917 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 2, 5, 6, 3, 7), (5, 7, 0, 4, 1, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x917, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x917, (4, 5, 1, 6, 3, 7), (6, 0, 4, 7, 2, 5)) * 4.0
    x918 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x918 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 2, 5, 6, 3, 7), (5, 7, 0, 4, 1, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x918, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x918, (4, 5, 1, 6, 3, 7), (6, 0, 4, 7, 2, 5)) * 12.0
    x919 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x919 += einsum(f.aa.ov, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 0, 3, 2, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x919, (2, 3, 0, 4, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x919
    x920 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x920 += einsum(f.bb.ov, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 1, 5, 6), (4, 6, 3, 0, 2, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x920, (2, 3, 4, 0, 5, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x920
    x921 = np.zeros((navir[1], navir[1]), dtype=np.float64)
    x921 += einsum(f.bb.oV, (0, 1), t1.bb[np.ix_(sob,sVb)], (0, 2), (1, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x921, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x921, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * -6.0
    del x921
    x922 = np.zeros((naocc[1], naocc[1]), dtype=np.float64)
    x922 += einsum(f.bb.vO, (0, 1), t1.bb[np.ix_(sOb,svb)], (2, 0), (1, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x922, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x922, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 0, 4, 5, 6), (2, 3, 1, 4, 5, 6)) * -6.0
    del x922
    x923 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x923 += einsum(f.aa.ov, (0, 1), t2.abab[np.ix_(soa,sOb,sva,sVb)], (2, 3, 1, 4), (3, 4, 0, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x923, (4, 5, 0, 6), (1, 6, 4, 3, 2, 5)) * -1.0
    del x923
    x924 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x924 += einsum(f.aa.ov, (0, 1), t2.abab[np.ix_(soa,sob,sva,sVb)], (2, 3, 1, 4), (4, 0, 2, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x924, (4, 0, 5, 6), (6, 5, 1, 3, 2, 4)) * -1.0
    del x924
    x925 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x925 += einsum(f.aa.ov, (0, 1), t2.abab[np.ix_(soa,sOb,sva,svb)], (2, 3, 1, 4), (3, 0, 2, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x925, (4, 0, 5, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x925
    x926 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x926 += einsum(f.bb.ov, (0, 1), t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (2, 3, 1, 4), (3, 4, 0, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x926, (4, 5, 1, 6), (6, 0, 4, 3, 2, 5)) * -2.0
    x927 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x927 += einsum(f.bb.ov, (0, 1), t2.bbbb[np.ix_(sob,sOb,svb,svb)], (2, 3, 4, 1), (3, 0, 2, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x927, (4, 1, 5, 6), (5, 0, 4, 6, 2, 3)) * 2.0
    x928 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x928 += einsum(f.bb.ov, (0, 1), t2.abab[np.ix_(soa,sOb,sva,svb)], (2, 3, 4, 1), (3, 2, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x928, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x928
    x929 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x929 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovvO, (2, 1, 3, 4), (4, 0, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x929, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3))
    del x929
    x930 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x930 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovvO, (4, 2, 5, 6), (6, 3, 0, 4, 1, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x930, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x930
    x931 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x931 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovOV, (4, 2, 5, 6), (5, 6, 0, 4, 1, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x931, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x931
    x932 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x932 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovoV, (2, 1, 3, 4), (4, 0, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x932, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4))
    del x932
    x933 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x933 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovoV, (4, 2, 5, 6), (1, 6, 0, 4, 5, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x933, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x933
    x934 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x934 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x934, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    x935 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x935 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.oooO, (2, 0, 3, 4), (4, 2, 3, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x935, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x935
    x936 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x936 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.oooo, (4, 5, 6, 0), (1, 3, 4, 5, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x936, (2, 3, 0, 4, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x936
    x937 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x937 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.aabb.oovV, (4, 5, 3, 6), (1, 6, 4, 5, 0, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x937, (2, 3, 0, 4, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x937
    x938 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x938 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.oovv, (4, 5, 6, 2), (1, 3, 4, 5, 0, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x938, (2, 3, 0, 4, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x938
    x939 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x939 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vvoO, (2, 1, 3, 4), (4, 0, 3, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x939, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x939
    x940 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x940 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vvvV, (2, 1, 3, 4), (4, 0, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x940, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * 2.0
    del x940
    x941 = np.zeros((naocc[1], navir[1], nocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    x941 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.vvvv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x941, (2, 3, 4, 1, 5, 6), (4, 0, 2, 6, 5, 3)) * 2.0
    del x941
    x942 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x942 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oooO, (2, 0, 3, 4), (4, 3, 2, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x942, (4, 1, 5, 6), (5, 0, 4, 6, 2, 3))
    x943 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x943 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oooO, (2, 3, 0, 4), (4, 2, 3, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x943, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    x944 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x944 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.oooO, (2, 3, 0, 4), (4, 2, 3, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x944, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x944
    x945 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x945 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvO, (2, 1, 3, 4), (4, 0, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x945, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3))
    x946 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x946 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovOV, (4, 3, 5, 6), (5, 6, 0, 1, 4, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x946, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.babbab[np.ix_(sob,sOa,sOfb,svb,sVa,sVfb)], (0, 1, 2, 3, 4, 5), x946, (2, 5, 6, 7, 0, 8), (6, 7, 1, 8, 3, 4)) * -1.0
    del x946
    x947 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x947 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oOvV, (2, 3, 1, 4), (3, 4, 0, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x947, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5))
    x948 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x948 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oOvv, (2, 3, 4, 1), (3, 0, 2, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x948, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    x949 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x949 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.oOvV, (4, 5, 3, 6), (5, 6, 0, 1, 4, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x949, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x949
    x950 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x950 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.oovV, (2, 3, 1, 4), (4, 2, 3, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x950, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4))
    del x950
    x951 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x951 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.oooo, (4, 0, 5, 6), (1, 3, 4, 5, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x951, (2, 3, 4, 0, 5, 6), (5, 4, 2, 1, 6, 3))
    del x951
    x952 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x952 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.bbbb.oovV, (4, 5, 3, 6), (1, 6, 0, 4, 5, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x952, (2, 3, 4, 0, 5, 6), (5, 4, 2, 1, 6, 3))
    del x952
    x953 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x953 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.bbbb.ovoV, (4, 3, 5, 6), (1, 6, 0, 5, 4, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x953, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x953
    x954 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x954 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oooO, (2, 0, 3, 4), (4, 1, 3, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x954, (4, 5, 1, 6), (6, 0, 4, 3, 2, 5)) * -1.0
    x955 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x955 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oooO, (2, 3, 0, 4), (4, 1, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x955, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5))
    x956 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x956 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.oooO, (2, 3, 0, 4), (4, 1, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x956, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5))
    del x956
    x957 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x957 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.ovvO, (0, 2, 3, 4), (4, 1, 3, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x957, (4, 5, 6, 3), (1, 0, 4, 6, 2, 5))
    x958 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x958 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oOvv, (0, 2, 3, 4), (2, 1, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x958, (4, 5, 6, 3), (1, 0, 4, 6, 2, 5)) * -1.0
    x959 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x959 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.oooo, (2, 3, 4, 0), (1, 2, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x959, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * -1.0
    del x959
    x960 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x960 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oovv, (2, 0, 3, 4), (1, 2, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x960, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4))
    x961 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x961 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.ovov, (2, 3, 0, 4), (1, 2, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x961, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -1.0
    x962 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x962 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.vvoO, (4, 2, 5, 6), (6, 3, 0, 1, 5, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x962, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x962
    x963 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x963 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.vvvV, (2, 1, 3, 4), (4, 0, 3, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x963, (4, 5, 3, 6), (5, 0, 1, 6, 2, 4))
    x964 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x964 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.vvvV, (2, 3, 1, 4), (4, 0, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x964, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -1.0
    x965 = np.zeros((navir[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x965 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.vvvV, (2, 3, 1, 4), (4, 0, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x965, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4)) * -1.0
    del x965
    x966 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    x966 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x966, (2, 3, 4, 5, 1, 6), (0, 4, 2, 6, 5, 3))
    del x966
    x967 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x967 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.vvoo, (4, 2, 5, 6), (1, 3, 0, 5, 6, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x967, (2, 3, 4, 0, 5, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x967
    x968 = np.zeros((naocc[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x968 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.vvoO, (2, 3, 0, 4), (4, 1, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x968, (4, 5, 6, 2), (1, 0, 4, 3, 6, 5)) * -1.0
    del x968
    x969 = np.zeros((navir[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x969 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.vvoo, (2, 3, 4, 0), (1, 4, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x969, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4))
    del x969
    x970 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x970 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.oovV, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x970, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    x971 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x971 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.oovV, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x971, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -1.0
    del x971
    x972 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x972 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x972, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3))
    x973 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x973 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.oovv, (2, 3, 4, 1), (0, 2, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x973, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3))
    del x973
    x974 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x974 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.ovoV, (2, 1, 3, 4), (0, 4, 3, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x974, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5))
    x975 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x975 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x975, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    x976 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x976 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.vvvV, (2, 1, 3, 4), (0, 4, 3, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x976, (4, 5, 3, 6), (1, 0, 4, 6, 2, 5)) * -1.0
    x977 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x977 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.vvvV, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x977, (4, 5, 6, 3), (1, 0, 4, 6, 2, 5))
    x978 = np.zeros((naocc[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x978 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.vvvV, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x978, (4, 5, 6, 2), (1, 0, 4, 3, 6, 5))
    del x978
    x979 = np.zeros((naocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    x979 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x979, (4, 5, 2, 6), (1, 0, 4, 6, 5, 3)) * -1.0
    del x979
    x980 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x980 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 6, 2), (1, 3, 4, 0, 6, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x980, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    x981 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x981 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.ovov, (2, 3, 0, 4), (1, 2, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x981, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * -2.0
    x982 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x982 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.ovov, (2, 3, 4, 1), (0, 2, 4, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x982, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    x983 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x983 += einsum(v.aaaa.ooov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 2, 5, 6, 3, 7), (5, 7, 0, 1, 4, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x983, (2, 3, 0, 4, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x983
    x984 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x984 += einsum(v.aaaa.ooov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 3, 7), (5, 7, 0, 2, 4, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x984, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x984
    x985 = np.zeros((naocc[1], naocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x985 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovOO, (2, 1, 3, 4), (3, 4, 0, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x985, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7)) * 2.0
    del x985
    x986 = np.zeros((navir[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x986 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovVV, (2, 1, 3, 4), (3, 4, 0, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x986, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0)) * -2.0
    del x986
    x987 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x987 += einsum(v.aabb.ovOO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 3, 6, 1, 7), (2, 7, 5, 0, 4, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x987, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x987
    x988 = np.zeros((naocc[1], naocc[1]), dtype=np.float64)
    x988 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovOO, (0, 1, 2, 3), (2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x988, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x988, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -6.0
    del x988
    x989 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x989 += einsum(v.aabb.ovoo, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 5, 6, 1, 7), (5, 7, 4, 0, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x989, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x989
    x990 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x990 += einsum(v.aabb.ovvv, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 3, 1, 7), (6, 7, 5, 0, 4, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x990, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x990
    x991 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x991 += einsum(v.aabb.ovVV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 7, 1, 3), (6, 2, 5, 0, 4, 7))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x991, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x991
    x992 = np.zeros((navir[1], navir[1]), dtype=np.float64)
    x992 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovVV, (0, 1, 2, 3), (2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x992, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x992, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 6.0
    del x992
    x993 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x993 += einsum(v.aabb.ooov, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 2, 5, 6, 3, 7), (5, 7, 0, 1, 4, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x993, (2, 3, 0, 4, 5, 6), (5, 4, 2, 6, 1, 3)) * -6.0
    del x993
    x994 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=np.float64)
    x994 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.OVoO, (2, 3, 0, 4), (2, 4, 3, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x994, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 7, 6, 3)) * -2.0
    del x994
    x995 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=np.float64)
    x995 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.OVvV, (2, 3, 1, 4), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x995, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 7, 6, 3)) * 2.0
    del x995
    x996 = np.zeros((naocc[0], naocc[1], navir[0], nvir[1]), dtype=np.float64)
    x996 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.OVoO, (2, 3, 0, 4), (2, 4, 3, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x996, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 3, 6, 7)) * 2.0
    del x996
    x997 = np.zeros((naocc[0], naocc[1], navir[0], nvir[1]), dtype=np.float64)
    x997 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.OVvv, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x997, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 3, 6, 7)) * -2.0
    del x997
    x998 = np.zeros((naocc[0], navir[0], navir[1], nocc[1]), dtype=np.float64)
    x998 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.OVvV, (2, 3, 1, 4), (2, 3, 4, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x998, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 1), (3, 4, 5, 7, 6, 2)) * -2.0
    del x998
    x999 = np.zeros((naocc[0], navir[0], navir[1], nocc[1]), dtype=np.float64)
    x999 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.OVoo, (2, 3, 4, 0), (2, 3, 1, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x999, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 1), (3, 4, 5, 7, 6, 2)) * 2.0
    del x999
    x1000 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x1000 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.OVvv, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1000, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * 2.0
    del x1000
    x1001 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x1001 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.OVoo, (2, 3, 4, 0), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1001, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * -2.0
    del x1001
    x1002 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1002 += einsum(v.bbbb.ooov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 4, 5, 3, 6, 7), (5, 7, 4, 0, 1, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1002, (2, 3, 4, 0, 5, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x1002
    x1003 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1003 += einsum(v.bbbb.ooov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 4, 5, 3, 6, 7), (5, 7, 4, 0, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1003, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * 2.0
    del x1003
    x1004 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1004 += einsum(v.aabb.ooov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 3, 6, 7), (5, 7, 0, 4, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1004, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * 2.0
    del x1004
    x1005 = np.zeros((naocc[1], naocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1005 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovOO, (2, 1, 3, 4), (3, 4, 0, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1005, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * 2.0
    x1006 = np.zeros((navir[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1006 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovVV, (2, 1, 3, 4), (3, 4, 0, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1006, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * -2.0
    x1007 = np.zeros((navir[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1007 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oVvV, (2, 3, 1, 4), (3, 4, 0, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1007, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 5, 6, 7, 0), (2, 4, 5, 6, 7, 1)) * 2.0
    x1008 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1008 += einsum(v.bbbb.ovOO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 3, 1, 6, 7), (2, 7, 5, 4, 0, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1008, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * 2.0
    del x1008
    x1009 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1009 += einsum(v.bbbb.oOvO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 3, 2, 6, 7), (1, 7, 5, 4, 0, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1009, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x1009
    x1010 = np.zeros((naocc[1], naocc[1]), dtype=np.float64)
    x1010 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovOO, (0, 1, 2, 3), (2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1010, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1010, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -6.0
    del x1010
    x1011 = np.zeros((naocc[1], naocc[1]), dtype=np.float64)
    x1011 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oOvO, (0, 2, 1, 3), (2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1011, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1011, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * 6.0
    del x1011
    x1012 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1012 += einsum(v.bbbb.ovVV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 1, 7, 3), (6, 2, 5, 4, 0, 7))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1012, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x1012
    x1013 = np.zeros((navir[1], navir[1]), dtype=np.float64)
    x1013 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovVV, (0, 1, 2, 3), (2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1013, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1013, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 6.0
    del x1013
    x1014 = np.zeros((navir[1], navir[1]), dtype=np.float64)
    x1014 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oVvV, (0, 2, 1, 3), (2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1014, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1014, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1)) * -6.0
    del x1014
    x1015 = np.zeros((navir[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1015 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oooV, (2, 0, 3, 4), (1, 4, 3, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1015, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 4, 5, 6, 7, 1), (3, 4, 5, 6, 7, 0)) * -2.0
    x1016 = np.zeros((navir[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1016 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oooV, (2, 3, 0, 4), (1, 4, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1016, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * 2.0
    x1017 = np.zeros((navir[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1017 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.oooV, (2, 3, 0, 4), (1, 4, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1017, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0)) * 2.0
    del x1017
    x1018 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=np.float64)
    x1018 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oVOO, (0, 2, 3, 4), (3, 4, 1, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1018, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1018, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 6.0
    del x1018
    x1019 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=np.float64)
    x1019 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oOOV, (0, 2, 3, 4), (3, 2, 1, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1019, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 0, 6, 7, 3), (4, 5, 1, 6, 7, 2)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1019, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 0, 6, 7, 3), (4, 5, 1, 6, 7, 2)) * -6.0
    del x1019
    x1020 = np.zeros((navir[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1020 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.ovvV, (0, 2, 3, 4), (1, 4, 3, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1020, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 3, 7, 1), (4, 5, 6, 2, 7, 0)) * 2.0
    x1021 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1021 += einsum(v.aabb.vvov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 3, 1, 7), (6, 7, 5, 4, 2, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1021, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x1021
    x1022 = np.zeros((navir[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x1022 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.vvoV, (2, 3, 0, 4), (1, 4, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1022, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 7, 3, 1), (4, 5, 6, 7, 2, 0)) * -2.0
    del x1022
    x1023 = np.zeros((naocc[1], naocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1023 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.oovO, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1023, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * 2.0
    x1024 = np.zeros((naocc[1], naocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x1024 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.oovO, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1024, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7)) * 2.0
    del x1024
    x1025 = np.zeros((naocc[1], naocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1025 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.vvvO, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1025, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7)) * -2.0
    x1026 = np.zeros((naocc[1], naocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1026 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.vvvO, (2, 1, 3, 4), (0, 4, 3, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1026, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 2, 6, 7), (4, 5, 0, 3, 6, 7)) * 2.0
    x1027 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=np.float64)
    x1027 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.vOVV, (1, 2, 3, 4), (0, 2, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1027, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1027, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -6.0
    del x1027
    x1028 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=np.float64)
    x1028 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.vVOV, (1, 2, 3, 4), (0, 3, 4, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1028, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 2), (4, 5, 0, 6, 7, 3)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1028, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 7, 2), (4, 5, 0, 6, 7, 3)) * 6.0
    del x1028
    x1029 = np.zeros((naocc[1], naocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1029 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.vvvO, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1029, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 3, 7), (4, 5, 0, 6, 2, 7)) * -2.0
    del x1029
    x1030 = np.zeros((naocc[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x1030 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoO, (0, 4, 1, 5), (5, 3, 2, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1030, (4, 5, 6, 2), (1, 0, 4, 3, 6, 5)) * -1.0
    del x1030
    x1031 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1031 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoO, (0, 2, 4, 5), (5, 3, 1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1031, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    x1032 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1032 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovoO, (4, 2, 1, 5), (5, 0, 4, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1032, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x1032
    x1033 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1033 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoO, (4, 2, 1, 5), (5, 3, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1033, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -1.0
    del x1033
    x1034 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1034 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovoO, (0, 2, 4, 5), (5, 1, 4, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1034, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    x1035 = np.zeros((naocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    x1035 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovoO, (0, 4, 1, 5), (5, 2, 4, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1035, (4, 5, 2, 6), (1, 0, 4, 6, 5, 3)) * -1.0
    del x1035
    x1036 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1036 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 0, 2), (1, 3, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1036, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -1.0
    del x1036
    x1037 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1037 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ooov, (4, 0, 5, 2), (1, 3, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1037, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5))
    del x1037
    x1038 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1038 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 0, 2), (1, 4, 5, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1038, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x1038
    x1039 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1039 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aaaa.ooov, (4, 0, 5, 2), (3, 4, 5, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1039, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4))
    del x1039
    x1040 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1040 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aaaa.ooov, (4, 0, 5, 2), (1, 4, 5, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1040, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3))
    del x1040
    x1041 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1041 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aaaa.ooov, (4, 5, 0, 2), (3, 4, 5, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1041, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * -1.0
    del x1041
    x1042 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1042 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 5), (1, 3, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1042, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    x1043 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1043 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoo, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1043, (4, 5, 6, 0, 1, 7), (7, 6, 4, 3, 2, 5))
    x1044 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1044 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 5), (1, 4, 5, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1044, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    x1045 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1045 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoo, (4, 2, 5, 1), (3, 0, 4, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1045, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4))
    del x1045
    x1046 = np.zeros((navir[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1046 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoo, (0, 4, 5, 1), (3, 5, 2, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1046, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4))
    del x1046
    x1047 = np.zeros((naocc[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x1047 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovvV, (0, 4, 3, 5), (1, 5, 2, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1047, (4, 5, 6, 2), (1, 0, 4, 3, 6, 5))
    del x1047
    x1048 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1048 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovvV, (4, 2, 3, 5), (5, 0, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1048, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4))
    del x1048
    x1049 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1049 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovvV, (0, 2, 4, 5), (1, 5, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1049, (4, 5, 6, 3), (1, 0, 4, 6, 2, 5))
    x1050 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1050 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovvV, (4, 2, 3, 5), (1, 5, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1050, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5))
    del x1050
    x1051 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1051 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovvV, (0, 2, 4, 5), (5, 1, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1051, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4))
    x1052 = np.zeros((navir[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1052 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovvV, (0, 4, 3, 5), (5, 1, 2, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1052, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4))
    del x1052
    x1053 = np.zeros((naocc[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x1053 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovvv, (0, 4, 5, 2), (1, 3, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1053, (4, 5, 2, 6), (1, 0, 4, 3, 6, 5)) * -1.0
    del x1053
    x1054 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1054 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovvv, (0, 2, 4, 5), (1, 3, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1054, (4, 5, 6, 3), (1, 0, 4, 6, 2, 5))
    x1055 = np.zeros((naocc[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x1055 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovvv, (0, 2, 4, 5), (1, 3, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1055, (4, 5, 6, 2), (1, 0, 4, 3, 6, 5))
    del x1055
    x1056 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x1056 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovvv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1056, (4, 5, 6, 0, 2, 7), (1, 6, 4, 3, 7, 5))
    del x1056
    x1057 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1057 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovvv, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1057, (4, 5, 6, 0, 3, 7), (1, 6, 4, 7, 2, 5)) * -1.0
    x1058 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x1058 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovvv, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1058, (4, 5, 6, 0, 2, 7), (1, 6, 4, 3, 7, 5)) * -1.0
    del x1058
    x1059 = np.zeros((naocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    x1059 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovvv, (0, 4, 5, 3), (1, 2, 4, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1059, (4, 5, 2, 6), (1, 0, 4, 6, 5, 3)) * -1.0
    del x1059
    x1060 = np.zeros((naocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    x1060 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aaaa.ovvv, (0, 4, 5, 2), (1, 4, 5, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1060, (4, 2, 5, 6), (1, 0, 4, 6, 5, 3)) * -1.0
    del x1060
    x1061 = np.zeros((naocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    x1061 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aaaa.ovvv, (0, 2, 4, 5), (1, 4, 5, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1061, (4, 5, 2, 6), (1, 0, 4, 6, 5, 3))
    del x1061
    x1062 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1062 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovvv, (4, 2, 5, 3), (1, 0, 4, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1062, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x1062
    x1063 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1063 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovvv, (0, 2, 4, 5), (3, 1, 4, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1063, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4))
    x1064 = np.zeros((navir[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1064 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovvv, (0, 2, 4, 5), (3, 1, 4, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1064, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4))
    del x1064
    x1065 = np.zeros((navir[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1065 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovvv, (0, 4, 5, 2), (3, 1, 4, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1065, (4, 5, 2, 6), (5, 0, 1, 3, 6, 4)) * -1.0
    del x1065
    x1066 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1066 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovoO, (1, 3, 4, 5), (5, 0, 4, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1066, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1066
    x1067 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1067 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoO, (4, 2, 1, 5), (5, 3, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1067, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -2.0
    x1068 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1068 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovoO, (4, 3, 1, 5), (5, 0, 4, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1068, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1068
    x1069 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1069 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoO, (1, 2, 4, 5), (5, 3, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1069, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * 2.0
    x1070 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1070 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoO, (1, 4, 0, 5), (5, 3, 2, 4)) * -1.0
    x1071 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1071 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1070, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1071, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1071, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x1071
    x1072 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1072 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovoO, (4, 3, 1, 5), (5, 0, 4, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1072, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * 2.0
    x1073 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1073 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovoO, (1, 3, 4, 5), (5, 0, 4, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1073, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -2.0
    x1074 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1074 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 0, 2), (1, 3, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1074, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -2.0
    x1075 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1075 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 0, 5, 2), (1, 3, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1075, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * 2.0
    x1076 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1076 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.ooov, (4, 5, 6, 2), (1, 3, 4, 5, 0, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1076, (4, 5, 6, 0, 7, 1), (7, 6, 4, 3, 2, 5)) * 2.0
    x1077 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1077 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.ooov, (4, 5, 0, 2), (1, 3, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1077, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -2.0
    del x1077
    x1078 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1078 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 0, 3), (1, 4, 5, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1078, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * 2.0
    x1079 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1079 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 0, 5, 3), (1, 4, 5, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1079, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -2.0
    x1080 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1080 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ooov, (4, 0, 1, 5), (3, 4, 2, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1080, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * -2.0
    del x1080
    x1081 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1081 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 5, 0, 3), (1, 4, 5, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1081, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * 2.0
    del x1081
    x1082 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1082 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (1, 3, 4, 5), (5, 0, 2, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1082, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * 2.0
    del x1082
    x1083 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1083 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (4, 3, 2, 5), (1, 5, 0, 4)) * -1.0
    x1084 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1084 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1083, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1084, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1084, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    del x1084
    x1085 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1085 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (1, 4, 3, 5), (5, 0, 2, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1085, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * -2.0
    del x1085
    x1086 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1086 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (0, 4, 3, 5), (1, 5, 2, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1086, (4, 5, 6, 3), (1, 0, 4, 6, 2, 5)) * 2.0
    x1087 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1087 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (0, 3, 4, 5), (1, 5, 2, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1087, (4, 5, 6, 3), (1, 0, 4, 6, 2, 5)) * -2.0
    x1088 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1088 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovvv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1088, (4, 5, 6, 1, 3, 7), (6, 0, 4, 7, 2, 5)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1088, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1088, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1088, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1088, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    x1089 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1089 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovvv, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1089, (4, 5, 6, 1, 3, 7), (6, 0, 4, 7, 2, 5)) * -2.0
    x1090 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1090 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovvv, (0, 4, 5, 2), (1, 3, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1090, (4, 5, 3, 6), (1, 0, 4, 6, 2, 5)) * -2.0
    x1091 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1091 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovvv, (0, 2, 4, 5), (1, 3, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1091, (4, 5, 6, 3), (1, 0, 4, 6, 2, 5)) * 2.0
    x1092 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1092 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (4, 2, 5, 3), (1, 0, 4, 5)) * -1.0
    x1093 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1093 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1092, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1093, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1093, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x1093
    x1094 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1094 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 0, 1, 5), (3, 4, 2, 5)) * -1.0
    x1095 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1095 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1094, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1095, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1095, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x1095
    x1096 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1096 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 0, 5, 3), (1, 4, 5, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1096, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1096
    x1097 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1097 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.aabb.ooov, (4, 5, 1, 2), (3, 4, 5, 0)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1097, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * 2.0
    del x1097
    x1098 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1098 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (1, 4, 3, 5), (5, 0, 2, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1098, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -2.0
    x1099 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1099 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (1, 3, 4, 5), (5, 0, 2, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1099, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * 2.0
    x1100 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1100 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovvv, (1, 4, 5, 2), (3, 0, 4, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1100, (4, 5, 3, 6), (5, 0, 1, 6, 2, 4)) * 2.0
    x1101 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1101 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovvv, (1, 2, 4, 5), (3, 0, 4, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1101, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -2.0
    x1102 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1102 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.vvov, (4, 5, 6, 2), (1, 3, 0, 6, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1102, (4, 5, 6, 1, 7, 2), (6, 0, 4, 3, 7, 5)) * -2.0
    x1103 = np.zeros((naocc[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x1103 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.vvov, (4, 5, 0, 2), (1, 3, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1103, (4, 5, 6, 2), (1, 0, 4, 3, 6, 5)) * 2.0
    del x1103
    x1104 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1104 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.vvov, (4, 2, 1, 5), (3, 0, 4, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1104, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * 2.0
    del x1104
    x1105 = np.zeros((naocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    x1105 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.aabb.vvov, (4, 5, 0, 3), (1, 4, 5, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1105, (4, 5, 2, 6), (1, 0, 4, 6, 5, 3)) * -2.0
    del x1105
    x1106 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1106 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.vvov, (4, 2, 5, 3), (1, 0, 5, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1106, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1106
    x1107 = np.zeros((navir[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1107 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.aabb.vvov, (4, 5, 1, 2), (3, 0, 4, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1107, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4)) * -2.0
    del x1107
    x1108 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1108 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aabb.ovoO, (1, 3, 4, 5), (5, 0, 4, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1108, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -4.0
    del x1108
    x1109 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1109 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aabb.ovvV, (1, 3, 4, 5), (5, 0, 2, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1109, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * 4.0
    del x1109
    x1110 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1110 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovOV, (4, 2, 5, 6), (5, 1, 6, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 1, 2, 3, 4, 5), x1110, (2, 6, 5, 7, 8, 0), (1, 8, 6, 4, 3, 7)) * -1.0
    del x1110
    x1111 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1111 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.oVvO, (4, 5, 2, 6), (6, 1, 5, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 1, 2, 3, 4, 5), x1111, (2, 6, 5, 7, 8, 0), (1, 8, 6, 4, 3, 7))
    del x1111
    x1112 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=np.float64)
    x1112 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovOV, (0, 2, 4, 5), (4, 1, 5, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1112, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 7, 6, 3)) * 2.0
    del x1112
    x1113 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=np.float64)
    x1113 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.oVvO, (0, 4, 2, 5), (5, 1, 4, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1113, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 7, 6, 3)) * -2.0
    del x1113
    x1114 = np.zeros((naocc[0], naocc[1], navir[0], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1114 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aaaa.ovOV, (4, 2, 5, 6), (5, 1, 6, 0, 4, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.abaaba[np.ix_(soa,sob,sOfa,sva,sVb,sVfa)], (0, 1, 2, 3, 4, 5), x1114, (2, 6, 5, 7, 0, 8), (1, 7, 6, 8, 3, 4)) * -1.0
    del x1114
    x1115 = np.zeros((naocc[0], naocc[1], navir[0], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1115 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aaaa.oVvO, (4, 5, 2, 6), (6, 1, 5, 0, 4, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.abaaba[np.ix_(soa,sob,sOfa,sva,sVb,sVfa)], (0, 1, 2, 3, 4, 5), x1115, (2, 6, 5, 7, 0, 8), (1, 7, 6, 8, 3, 4))
    del x1115
    x1116 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1116 += einsum(v.aaaa.ovOV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,sVb,sVfa)], (4, 5, 2, 1, 6, 3), (6, 4, 0, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1116, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * -2.0
    del x1116
    x1117 = np.zeros((naocc[0], naocc[1], navir[0], nvir[1]), dtype=np.float64)
    x1117 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aaaa.ovOV, (0, 2, 4, 5), (4, 1, 5, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1117, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 3, 6, 7)) * 2.0
    del x1117
    x1118 = np.zeros((naocc[0], naocc[1], navir[0], nvir[1]), dtype=np.float64)
    x1118 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aaaa.oVvO, (0, 4, 2, 5), (5, 1, 4, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1118, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 3, 6, 7)) * -2.0
    del x1118
    x1119 = np.zeros((naocc[0], navir[0], navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1119 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovOV, (4, 2, 5, 6), (5, 6, 3, 0, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.abaaba[np.ix_(soa,sOb,sOfa,sva,svb,sVfa)], (0, 1, 2, 3, 4, 5), x1119, (2, 5, 6, 7, 0, 8), (8, 7, 1, 4, 3, 6)) * -1.0
    del x1119
    x1120 = np.zeros((naocc[0], navir[0], navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1120 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aaaa.oVvO, (4, 5, 2, 6), (6, 5, 3, 0, 4, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.abaaba[np.ix_(soa,sOb,sOfa,sva,svb,sVfa)], (0, 1, 2, 3, 4, 5), x1120, (2, 5, 6, 7, 0, 8), (8, 7, 1, 4, 3, 6))
    del x1120
    x1121 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1121 += einsum(v.aaaa.ovOV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,svb,sVfa)], (4, 5, 2, 1, 6, 3), (5, 4, 0, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1121, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -2.0
    del x1121
    x1122 = np.zeros((naocc[0], navir[0], navir[1], nocc[1]), dtype=np.float64)
    x1122 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovOV, (0, 2, 4, 5), (4, 5, 3, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1122, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 1), (3, 4, 5, 7, 6, 2)) * 2.0
    del x1122
    x1123 = np.zeros((naocc[0], navir[0], navir[1], nocc[1]), dtype=np.float64)
    x1123 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aaaa.oVvO, (0, 4, 2, 5), (5, 4, 3, 1)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1123, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 1), (3, 4, 5, 7, 6, 2)) * -2.0
    del x1123
    x1124 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1124 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ovOV, (4, 2, 5, 6), (5, 6, 0, 4, 1, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.abaaba[np.ix_(soa,sOb,sOfa,sva,sVb,sVfa)], (0, 1, 2, 3, 4, 5), x1124, (2, 5, 6, 0, 7, 8), (7, 6, 1, 8, 3, 4)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1124, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1124
    x1125 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1125 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.oVvO, (4, 5, 2, 6), (6, 5, 0, 4, 1, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.abaaba[np.ix_(soa,sOb,sOfa,sva,sVb,sVfa)], (0, 1, 2, 3, 4, 5), x1125, (2, 5, 6, 0, 7, 8), (7, 6, 1, 8, 3, 4))
    del x1125
    x1126 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1126 += einsum(v.aaaa.ovOV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,sVb,sVfa)], (4, 5, 2, 1, 6, 3), (5, 6, 4, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1126, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -2.0
    del x1126
    x1127 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x1127 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ovOV, (0, 2, 4, 5), (4, 5, 1, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1127, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * 2.0
    del x1127
    x1128 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x1128 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.oVvO, (0, 4, 2, 5), (5, 4, 1, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1128, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * -2.0
    del x1128
    x1129 = np.zeros((naocc[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x1129 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 0, 4, 3, 5, 6), (4, 6, 5, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1129, (4, 5, 6, 2), (1, 0, 4, 3, 6, 5)) * -2.0
    del x1129
    x1130 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1130 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 0, 4, 5, 1, 6), (4, 6, 5, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1130, (4, 5, 6, 3), (1, 0, 4, 6, 2, 5)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1130, (4, 5, 6, 2), (0, 1, 4, 6, 5, 3)) * 4.0
    x1131 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1131 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 5, 3, 1, 6), (5, 6, 4, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1131, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1131, (4, 5, 6, 0), (6, 4, 1, 2, 3, 5)) * 4.0
    x1132 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1132 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 5, 3, 6, 7), (5, 7, 4, 2, 6, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1132, (4, 5, 6, 1, 7, 2), (6, 0, 4, 3, 7, 5)) * 2.0
    del x1132
    x1133 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1133 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 5, 6, 1, 7), (5, 7, 4, 2, 6, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1133, (4, 5, 6, 1, 7, 3), (6, 0, 4, 7, 2, 5)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1133, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1133, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1133, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1133, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    x1134 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1134 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 4, 5, 3, 1, 6), (5, 6, 4, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1134, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -2.0
    del x1134
    x1135 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x1135 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 4, 5, 3, 6, 7), (5, 7, 4, 0, 6, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1135, (4, 5, 6, 0, 7, 2), (1, 6, 4, 3, 7, 5)) * 2.0
    del x1135
    x1136 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1136 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 4, 5, 6, 1, 7), (5, 7, 4, 0, 6, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1136, (4, 5, 6, 0, 7, 3), (1, 6, 4, 7, 2, 5)) * 2.0
    del x1136
    x1137 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1137 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 3, 1, 7), (6, 7, 5, 0, 4, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1137, (4, 5, 6, 0, 7, 1), (7, 6, 4, 3, 2, 5)) * 2.0
    x1138 = np.zeros((naocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    x1138 += einsum(v.aabb.ovoV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 0, 4, 5, 6, 3), (4, 6, 1, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1138, (4, 5, 2, 6), (1, 0, 4, 6, 5, 3)) * -2.0
    del x1138
    x1139 = np.zeros((navir[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1139 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoV, (4, 2, 1, 5), (3, 5, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1139, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0)) * -2.0
    del x1139
    x1140 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1140 += einsum(v.aabb.ovoV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 4, 5, 6, 1, 3), (5, 4, 0, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1140, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -2.0
    del x1140
    x1141 = np.zeros((navir[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1141 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoV, (0, 2, 4, 5), (3, 5, 1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1141, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * -2.0
    x1142 = np.zeros((navir[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x1142 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoV, (0, 4, 1, 5), (3, 5, 2, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1142, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 7, 3, 1), (4, 5, 6, 7, 2, 0)) * -2.0
    del x1142
    x1143 = np.zeros((navir[1], navir[1]), dtype=np.float64)
    x1143 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoV, (0, 2, 1, 4), (3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1143, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1143, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 6.0
    del x1143
    x1144 = np.zeros((navir[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1144 += einsum(v.aabb.ovvO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 3, 2, 5, 6), (6, 4, 5, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1144, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4)) * -2.0
    del x1144
    x1145 = np.zeros((naocc[1], naocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x1145 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovvO, (4, 2, 3, 5), (1, 5, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1145, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7)) * -2.0
    del x1145
    x1146 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1146 += einsum(v.aabb.ovvO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 3, 5, 1, 6), (6, 4, 5, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1146, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -2.0
    x1147 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1147 += einsum(v.aabb.ovvO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 3, 2, 1, 6), (6, 5, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1147, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * -2.0
    del x1147
    x1148 = np.zeros((naocc[1], naocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1148 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovvO, (0, 2, 4, 5), (1, 5, 3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1148, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7)) * -2.0
    x1149 = np.zeros((naocc[1], naocc[1]), dtype=np.float64)
    x1149 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovvO, (0, 2, 3, 4), (1, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1149, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1149, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * 6.0
    del x1149
    x1150 = np.zeros((naocc[1], naocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1150 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovvO, (0, 4, 3, 5), (1, 5, 2, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1150, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 3, 7), (4, 5, 0, 6, 2, 7)) * -2.0
    del x1150
    x1151 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1151 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovOV, (4, 2, 5, 6), (1, 5, 3, 6, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (0, 1, 2, 3, 4, 5), x1151, (6, 2, 7, 5, 8, 1), (0, 8, 6, 3, 4, 7)) * -2.0
    del x1151
    x1152 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=np.float64)
    x1152 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovOV, (0, 2, 4, 5), (1, 4, 3, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1152, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1152, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 6.0
    del x1152
    x1153 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1153 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (0, 2, 4, 5, 3, 6), (4, 6, 5, 1))
    x1154 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1154 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1153, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1154, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -3.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1154, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -3.0
    del x1154
    x1155 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1155 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 2, 5, 1, 3, 6), (5, 6, 4, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1155, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -6.0
    x1156 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1156 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 2, 5, 6, 3, 7), (5, 7, 4, 0, 6, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1156, (4, 5, 6, 1, 7, 3), (6, 0, 4, 7, 2, 5)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1156, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1156, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1156, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1156, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    x1157 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1157 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 2, 5, 6, 1, 7), (5, 7, 4, 0, 6, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1157, (4, 5, 6, 1, 7, 3), (6, 0, 4, 7, 2, 5)) * -6.0
    x1158 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1158 += einsum(v.bbbb.ovoV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 2, 5, 6, 1, 3), (5, 4, 0, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1158, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * 6.0
    x1159 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1159 += einsum(v.bbbb.ovvO, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 0, 3, 5, 2, 6), (6, 4, 5, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1159, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * 3.0
    x1160 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1160 += einsum(v.bbbb.ovvO, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 0, 3, 5, 1, 6), (6, 4, 5, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1160, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -3.0
    x1161 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1161 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 6, 2), (4, 1, 5, 3, 0, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 1, 2, 3, 4, 5), x1161, (2, 6, 5, 7, 8, 1), (8, 0, 6, 4, 3, 7)) * -4.0
    del x1161
    x1162 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=np.float64)
    x1162 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 0, 2), (4, 1, 5, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1162, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 7, 6, 3)) * 4.0
    del x1162
    x1163 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1163 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 6, 3), (4, 1, 5, 0, 6, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.abaaba[np.ix_(soa,sob,sOfa,sva,sVb,sVfa)], (0, 1, 2, 3, 4, 5), x1163, (2, 6, 5, 7, 1, 8), (7, 0, 6, 8, 3, 4)) * 4.0
    del x1163
    x1164 = np.zeros((naocc[0], naocc[1], navir[0], nvir[1]), dtype=np.float64)
    x1164 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 0, 3), (4, 1, 5, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1164, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 3, 6, 7)) * -4.0
    del x1164
    x1165 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1165 += einsum(v.aabb.OVov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 3, 1), (5, 4, 2, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1165, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 4.0
    del x1165
    x1166 = np.zeros((naocc[0], navir[0], navir[1], nocc[1]), dtype=np.float64)
    x1166 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 1, 2), (4, 5, 3, 0)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1166, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 1), (3, 4, 5, 7, 6, 2)) * -4.0
    del x1166
    x1167 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x1167 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 1, 3), (4, 5, 0, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1167, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * 4.0
    del x1167
    x1168 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1168 += einsum(x18, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 2, 3, 4, 5, 6), (3, 6, 2, 0, 5, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1168, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1168, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1168
    x1169 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1169 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 4, 5, 1, 6, 7), (5, 7, 4, 0, 6, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1169, (4, 5, 6, 1, 7, 3), (0, 6, 4, 2, 7, 5)) * -4.0
    del x1169
    x1170 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1170 += einsum(x37, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 5, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1170, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1170, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1170
    x1171 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1171 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 2, 5, 6, 1, 7), (5, 7, 0, 4, 3, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x1171, (4, 5, 1, 6, 3, 7), (6, 0, 4, 7, 2, 5)) * -4.0
    x1172 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1172 += einsum(x183, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 1, 5, 6), (4, 6, 3, 2, 5, 0)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1172, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1172, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1172
    x1173 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1173 += einsum(x111, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 3, 2, 0, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1173, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1173, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1173
    x1174 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1174 += einsum(v.bbbb.ovoV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 4, 5, 1, 6, 3), (5, 4, 0, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1174, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 4.0
    del x1174
    x1175 = np.zeros((navir[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1175 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoV, (1, 2, 4, 5), (3, 5, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1175, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * 4.0
    x1176 = np.zeros((navir[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1176 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoV, (4, 2, 1, 5), (3, 5, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1176, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * -4.0
    x1177 = np.zeros((navir[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1177 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoV, (1, 4, 0, 5), (3, 5, 2, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1177, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 3, 7, 1), (4, 5, 6, 2, 7, 0)) * 2.0
    x1178 = np.zeros((navir[1], navir[1]), dtype=np.float64)
    x1178 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoV, (1, 2, 0, 4), (3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1178, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1178, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -6.0
    del x1178
    x1179 = np.zeros((navir[1], navir[1]), dtype=np.float64)
    x1179 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoV, (1, 2, 0, 4), (3, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1179, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1179, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -6.0
    del x1179
    x1180 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1180 += einsum(v.bbbb.ovvO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (0, 4, 3, 2, 5, 6), (6, 4, 5, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1180, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * 2.0
    del x1180
    x1181 = np.zeros((naocc[1], naocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1181 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvO, (4, 3, 2, 5), (1, 5, 0, 4)) * -1.0
    x1182 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1182 += einsum(x1181, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 1, 5, 6, 7), (0, 7, 4, 2, 6, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1182, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1182, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x1182
    x1183 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1183 += einsum(v.bbbb.ovvO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (0, 4, 3, 1, 5, 6), (6, 4, 5, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1183, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * -2.0
    del x1183
    x1184 = np.zeros((naocc[1], naocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1184 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvO, (0, 3, 4, 5), (1, 5, 2, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1184, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7)) * 4.0
    x1185 = np.zeros((naocc[1], naocc[1]), dtype=np.float64)
    x1185 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvO, (0, 3, 2, 4), (1, 4)) * -1.0
    x1186 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1186 += einsum(x1185, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 1, 4, 5, 6), (0, 6, 3, 2, 5, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1186, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1186, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1186
    x1187 = np.zeros((naocc[1], naocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1187 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvO, (0, 4, 3, 5), (1, 5, 2, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1187, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 3, 6, 7), (4, 5, 0, 2, 6, 7)) * -4.0
    x1188 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1188 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovOV, (4, 2, 5, 6), (1, 5, 3, 6, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (0, 1, 2, 3, 4, 5), x1188, (6, 2, 7, 5, 8, 0), (8, 1, 6, 3, 4, 7)) * -2.0
    x1189 = np.zeros((naocc[1], naocc[1], navir[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1189 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.oVvO, (4, 5, 2, 6), (1, 6, 3, 5, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (0, 1, 2, 3, 4, 5), x1189, (6, 2, 7, 5, 8, 0), (8, 1, 6, 3, 4, 7)) * 2.0
    x1190 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=np.float64)
    x1190 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovOV, (0, 2, 4, 5), (1, 4, 3, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1190, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1190, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 12.0
    del x1190
    x1191 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=np.float64)
    x1191 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.oVvO, (0, 4, 2, 5), (1, 5, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1191, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1191, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -12.0
    del x1191
    x1192 = np.zeros((naocc[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1192 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovoO, (2, 1, 3, 4), (4, 0, 2, 3))
    x1193 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1193 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1192, (2, 3, 0, 4), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1193, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1193
    x1194 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1194 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x151, (4, 5, 6, 0), (1, 3, 4, 5, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1194, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1194
    x1195 = np.zeros((navir[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1195 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovvV, (2, 1, 3, 4), (4, 0, 2, 3))
    x1196 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1196 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1195, (4, 5, 6, 3), (1, 4, 5, 6, 0, 2)) * -1.0
    del x1195
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1196, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1196
    x1197 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1197 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x915, (2, 3, 4, 5, 1, 6), (2, 3, 0, 4, 5, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1197, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x1197
    x1198 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1198 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1192, (2, 3, 4, 0), (2, 3, 4, 1))
    del x1192
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1198, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x1198
    x1199 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1199 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoO, (4, 2, 5, 6), (6, 3, 0, 4, 1, 5)) * -1.0
    x1200 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1200 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1199, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1199
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1200, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3))
    del x1200
    x1201 = np.zeros((navir[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1201 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.ovvV, (2, 3, 1, 4), (4, 2, 0, 3))
    x1202 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1202 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1201, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1202, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4))
    del x1202
    x1203 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1203 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x151, (4, 0, 5, 6), (1, 3, 4, 6, 5, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1203, (2, 3, 4, 0, 5, 6), (5, 4, 2, 1, 6, 3))
    del x1203
    x1204 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1204 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1201, (4, 5, 6, 2), (1, 4, 0, 5, 6, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1204, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x1204
    x1205 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1205 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1057, (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 0, 6))
    del x1057
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1205, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x1205
    x1206 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1206 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1043, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1043
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1206, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3))
    del x1206
    x1207 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1207 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.ovoO, (2, 3, 0, 4), (4, 1, 2, 3))
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,sVb,sVfb)] += einsum(x1207, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * -2.0
    x1208 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1208 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1207, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1208, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5))
    del x1208
    x1209 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1209 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1207, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x1207
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1209, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3))
    del x1209
    x1210 = np.zeros((navir[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1210 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.ovoo, (2, 3, 4, 0), (1, 2, 4, 3))
    x1211 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1211 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1210, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1211, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * -1.0
    del x1211
    x1212 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1212 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1210, (4, 5, 6, 2), (1, 4, 0, 5, 6, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1212, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3))
    del x1212
    x1213 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1213 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.ovvV, (2, 3, 1, 4), (0, 4, 2, 3))
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,sVb,sVfb)] += einsum(x1213, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * 2.0
    x1214 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1214 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1213, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1214, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -1.0
    del x1214
    x1215 = np.zeros((naocc[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1215 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x1216 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1216 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1215, (2, 3, 1, 4), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1216, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3))
    del x1216
    x1217 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1217 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1213, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x1213
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1217, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x1217
    x1218 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1218 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1215, (4, 5, 2, 6), (4, 3, 0, 5, 1, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1218, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x1218
    x1219 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1219 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x156, (4, 5, 6, 0), (1, 3, 5, 4, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1219, (2, 3, 0, 4, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1219
    x1220 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1220 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1076, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1076
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1220, (2, 3, 0, 4, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1220
    x1221 = np.zeros((navir[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1221 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.ooov, (2, 3, 0, 4), (1, 2, 3, 4))
    x1222 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1222 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1221, (4, 5, 6, 3), (1, 4, 6, 5, 0, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1222, (2, 3, 0, 4, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x1222
    x1223 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1223 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1102, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x1102
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1223, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x1223
    x1224 = np.zeros((navir[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    x1224 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.vvov, (2, 3, 0, 4), (1, 2, 3, 4))
    x1225 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1225 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1224, (2, 3, 1, 4), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1225, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * -2.0
    del x1225
    x1226 = np.zeros((naocc[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1226 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x1227 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1227 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1226, (2, 3, 0, 4), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1227, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1227
    x1228 = np.zeros((naocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1228 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.vvov, (2, 3, 4, 1), (0, 4, 2, 3))
    x1229 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1229 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1228, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1229, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1229
    x1230 = np.zeros((naocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1230 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovoO, (2, 1, 3, 4), (4, 0, 3, 2))
    x1231 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1231 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1230, (2, 3, 4, 0), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1231, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3))
    x1232 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1232 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1230, (2, 3, 0, 4), (2, 3, 4, 1))
    del x1230
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1232, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    x1233 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1233 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x156, (4, 0, 5, 6), (1, 3, 4, 5, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1233, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x1233
    x1234 = np.zeros((navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1234 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvV, (2, 1, 3, 4), (4, 0, 2, 3))
    x1235 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1235 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1234, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1235, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x1235
    x1236 = np.zeros((navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1236 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvV, (2, 3, 1, 4), (4, 0, 2, 3))
    x1237 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1237 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1236, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1237, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x1237
    x1238 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1]), dtype=np.float64)
    x1238 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.ovoO, (0, 2, 3, 4), (4, 1, 3, 2))
    x1239 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1239 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1238, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1239, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    x1240 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1]), dtype=np.float64)
    x1240 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.ovoO, (2, 3, 0, 4), (4, 1, 2, 3))
    x1241 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1241 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1240, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1241, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5))
    x1242 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1242 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1240, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1242, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x1242
    x1243 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1243 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1238, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1243, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x1243
    x1244 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1244 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1221, (2, 3, 4, 1), (2, 4, 3, 0))
    del x1221
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1244, (4, 0, 5, 6), (6, 5, 1, 3, 2, 4)) * -1.0
    del x1244
    x1245 = np.zeros((navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1245 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.ooov, (2, 3, 0, 4), (1, 2, 3, 4))
    x1246 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1246 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1245, (4, 5, 6, 3), (1, 4, 0, 6, 5, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1246, (2, 3, 4, 0, 5, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x1246
    x1247 = np.zeros((navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1247 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.ooov, (2, 0, 3, 4), (1, 2, 3, 4))
    x1248 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1248 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1247, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1248, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x1248
    x1249 = np.zeros((navir[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1249 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.ovvv, (0, 2, 3, 4), (1, 2, 3, 4))
    x1250 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1250 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1249, (2, 1, 3, 4), (2, 0, 4, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1250, (4, 5, 3, 6), (5, 0, 1, 6, 2, 4))
    x1251 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1251 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1249, (2, 3, 4, 1), (2, 0, 3, 4))
    del x1249
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1251, (4, 5, 3, 6), (5, 0, 1, 6, 2, 4)) * -1.0
    x1252 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1252 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x914, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    del x914
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1252, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x1252
    x1253 = np.zeros((navir[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1253 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1224, (2, 3, 4, 1), (2, 0, 4, 3))
    del x1224
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1253, (4, 5, 2, 6), (5, 0, 1, 3, 6, 4))
    del x1253
    x1254 = np.zeros((naocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1254 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.ooov, (2, 3, 4, 1), (0, 2, 3, 4))
    x1255 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1255 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1254, (2, 3, 0, 4), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1255, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3))
    x1256 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1256 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1254, (2, 3, 4, 0), (2, 4, 3, 1))
    del x1254
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1256, (4, 1, 5, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    x1257 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1257 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1226, (2, 3, 4, 0), (2, 4, 3, 1))
    del x1226
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1257, (4, 0, 5, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x1257
    x1258 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1]), dtype=np.float64)
    x1258 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.ovvV, (2, 3, 1, 4), (0, 4, 2, 3))
    x1259 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1259 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1258, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1259, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    x1260 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1]), dtype=np.float64)
    x1260 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.ovvV, (2, 1, 3, 4), (0, 4, 2, 3))
    x1261 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1261 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1260, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1261, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5))
    x1262 = np.zeros((naocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1262 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.ovvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x1263 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1263 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1262, (2, 3, 1, 4), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1263, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3))
    x1264 = np.zeros((naocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1264 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x1265 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1265 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1264, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1265, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    x1266 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1266 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1260, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1266, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x1266
    x1267 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1267 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1258, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1267, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x1267
    x1268 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1268 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t1.bb[np.ix_(sOb,svb)], (2, 3), v.bbbb.ooov, (4, 0, 5, 3), (2, 1, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1268, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    x1269 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1269 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t1.bb[np.ix_(sOb,svb)], (2, 3), v.bbbb.ooov, (4, 5, 0, 3), (2, 1, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1269, (4, 5, 1, 6), (6, 0, 4, 3, 2, 5))
    x1270 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1270 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t1.bb[np.ix_(sOb,svb)], (2, 3), v.aabb.ooov, (4, 5, 0, 3), (2, 1, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1270, (4, 5, 0, 6), (1, 6, 4, 3, 2, 5))
    del x1270
    x1271 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1271 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t1.bb[np.ix_(sOb,svb)], (2, 3), v.bbbb.ovvv, (0, 3, 4, 5), (2, 1, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1271, (4, 5, 3, 6), (1, 0, 4, 6, 2, 5)) * -1.0
    x1272 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1272 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t1.bb[np.ix_(sOb,svb)], (2, 3), v.bbbb.ovvv, (0, 4, 5, 3), (2, 1, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1272, (4, 5, 3, 6), (1, 0, 4, 6, 2, 5))
    x1273 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1273 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1228, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x1228
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1273, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x1273
    x1274 = np.zeros((naocc[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x1274 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t1.bb[np.ix_(sOb,svb)], (2, 3), v.aabb.vvov, (4, 5, 0, 3), (2, 1, 4, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1274, (4, 5, 2, 6), (1, 0, 4, 3, 6, 5)) * -1.0
    del x1274
    x1275 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1275 += einsum(x35, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 3, 7), (5, 7, 0, 2, 4, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1275, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x1275
    x1276 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1276 += einsum(x35, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 2, 5, 6, 3, 7), (5, 7, 0, 1, 4, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1276, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1276
    x1277 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1277 += einsum(x11, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 3, 0, 2, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1277, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x1277
    x1278 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1278 += einsum(x12, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 3, 0, 2, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1278, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1278
    x1279 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1279 += einsum(x33, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 2, 5, 6, 3, 7), (5, 7, 0, 1, 4, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1279, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -6.0
    del x1279
    x1280 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1280 += einsum(x33, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 3, 6, 7), (5, 7, 0, 4, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1280, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * 2.0
    del x1280
    x1281 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1281 += einsum(x13, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 4, 5, 6, 3, 7), (5, 7, 4, 0, 1, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1281, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1281
    x1282 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1282 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1137, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1137
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1282, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1282
    x1283 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1283 += einsum(x15, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 3, 0, 2, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1283, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x1283
    x1284 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1284 += einsum(x0, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 1, 5, 6), (4, 6, 3, 2, 0, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1284, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x1284
    x1285 = np.zeros((navir[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1285 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.ovoV, (2, 3, 0, 4), (1, 4, 2, 3))
    x1286 = np.zeros((navir[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1286 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1285, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1286, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0)) * 2.0
    del x1286
    x1287 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1287 += einsum(x1285, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 7, 3, 1), (6, 0, 5, 2, 4, 7))
    del x1285
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1287, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1287
    x1288 = np.zeros((navir[1], nocc[1]), dtype=np.float64)
    x1288 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovoV, (0, 1, 2, 3), (3, 2))
    x1289 = np.zeros((navir[1], navir[1]), dtype=np.float64)
    x1289 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1288, (2, 0), (1, 2))
    del x1288
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1289, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1289, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -6.0
    del x1289
    x1290 = np.zeros((naocc[1], naocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x1290 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.ovvO, (2, 3, 1, 4), (0, 4, 2, 3))
    x1291 = np.zeros((naocc[1], naocc[1], nocc[0], nocc[0]), dtype=np.float64)
    x1291 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1290, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1291, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7)) * 2.0
    del x1291
    x1292 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1292 += einsum(x1290, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 3, 7), (0, 7, 5, 2, 4, 6))
    del x1290
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1292, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1292
    x1293 = np.zeros((naocc[1], nvir[1]), dtype=np.float64)
    x1293 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ovvO, (0, 1, 2, 3), (3, 2))
    x1294 = np.zeros((naocc[1], naocc[1]), dtype=np.float64)
    x1294 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1293, (2, 1), (0, 2))
    del x1293
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1294, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1294, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -6.0
    del x1294
    x1295 = np.zeros((naocc[0], naocc[1], navir[0], nocc[1]), dtype=np.float64)
    x1295 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.aabb.OVov, (2, 3, 4, 1), (2, 0, 3, 4))
    x1296 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=np.float64)
    x1296 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1295, (2, 3, 4, 0), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1296, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 7, 6, 3)) * -2.0
    del x1296
    x1297 = np.zeros((naocc[0], naocc[1], navir[0], nvir[1]), dtype=np.float64)
    x1297 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1295, (2, 3, 4, 0), (2, 3, 4, 1))
    del x1295
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1297, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 2), (5, 4, 1, 3, 6, 7)) * 2.0
    del x1297
    x1298 = np.zeros((naocc[0], navir[0], navir[1], nvir[1]), dtype=np.float64)
    x1298 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.aabb.OVov, (2, 3, 0, 4), (2, 3, 1, 4))
    x1299 = np.zeros((naocc[0], navir[0], navir[1], nocc[1]), dtype=np.float64)
    x1299 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1298, (2, 3, 4, 1), (2, 3, 4, 0))
    del x1298
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1299, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,svb,sVfa)], (4, 5, 0, 6, 7, 1), (3, 4, 5, 7, 6, 2)) * 2.0
    del x1299
    x1300 = np.zeros((naocc[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x1300 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x169, (2, 3, 4, 0), (2, 3, 4, 1))
    del x169
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1300, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sOb,sOfa,sva,sVb,sVfa)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * -2.0
    del x1300
    x1301 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1301 += einsum(x16, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (1, 4, 5, 3, 6, 7), (5, 7, 4, 0, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1301, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x1301
    x1302 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1302 += einsum(x16, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 4, 5, 3, 6, 7), (5, 7, 4, 0, 1, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1302, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * 2.0
    del x1302
    x1303 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1303 += einsum(x1, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 1, 5, 6), (4, 6, 3, 2, 0, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1303, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x1303
    x1304 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1304 += einsum(x20, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 1, 5, 6), (4, 6, 3, 2, 0, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1304, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * 2.0
    del x1304
    x1305 = np.zeros((navir[1], navir[1], nocc[1], nvir[1]), dtype=np.float64)
    x1305 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.ovoV, (0, 2, 3, 4), (1, 4, 3, 2))
    x1306 = np.zeros((navir[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1306 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1305, (2, 3, 4, 1), (2, 3, 0, 4))
    del x1305
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1306, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * -2.0
    x1307 = np.zeros((navir[1], navir[1], nocc[1], nvir[1]), dtype=np.float64)
    x1307 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.ovoV, (2, 3, 0, 4), (1, 4, 2, 3))
    x1308 = np.zeros((navir[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1308 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1307, (2, 3, 4, 1), (2, 3, 0, 4))
    del x1307
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1308, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 5, 6, 7, 1), (2, 4, 5, 6, 7, 0)) * 2.0
    x1309 = np.zeros((navir[1], navir[1], nocc[1], nvir[1]), dtype=np.float64)
    x1309 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.ovoV, (2, 3, 0, 4), (1, 4, 2, 3))
    x1310 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1310 += einsum(x1309, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 6, 3, 7, 1), (6, 0, 5, 4, 2, 7))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1310, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * 2.0
    del x1310
    x1311 = np.zeros((navir[1], nocc[1]), dtype=np.float64)
    x1311 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovoV, (0, 1, 2, 3), (3, 2))
    x1312 = np.zeros((navir[1], navir[1]), dtype=np.float64)
    x1312 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1311, (2, 0), (1, 2))
    del x1311
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1312, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1312, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * -6.0
    del x1312
    x1313 = np.zeros((navir[1], nocc[1]), dtype=np.float64)
    x1313 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovoV, (2, 1, 0, 3), (3, 2))
    x1314 = np.zeros((navir[1], navir[1]), dtype=np.float64)
    x1314 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1313, (2, 0), (1, 2))
    del x1313
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1314, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1314, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 6, 0)) * 6.0
    del x1314
    x1315 = np.zeros((naocc[1], naocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1315 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.ovvO, (2, 3, 1, 4), (0, 4, 2, 3))
    x1316 = np.zeros((naocc[1], naocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1316 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1315, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1316, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7))
    x1317 = np.zeros((naocc[1], naocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1317 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.ovvO, (2, 1, 3, 4), (0, 4, 2, 3))
    x1318 = np.zeros((naocc[1], naocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1318 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1317, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1318, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (3, 4, 1, 5, 6, 7), (2, 4, 0, 5, 6, 7)) * -1.0
    x1319 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1319 += einsum(x1317, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 3, 6, 7), (0, 7, 5, 4, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1319, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x1319
    x1320 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1320 += einsum(x1315, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 3, 6, 7), (0, 7, 5, 4, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1320, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * 2.0
    del x1320
    x1321 = np.zeros((naocc[1], nvir[1]), dtype=np.float64)
    x1321 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvO, (0, 1, 2, 3), (3, 2))
    x1322 = np.zeros((naocc[1], naocc[1]), dtype=np.float64)
    x1322 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1321, (2, 1), (0, 2))
    del x1321
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1322, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1322, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * -6.0
    del x1322
    x1323 = np.zeros((naocc[1], nvir[1]), dtype=np.float64)
    x1323 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovvO, (0, 2, 1, 3), (3, 2))
    x1324 = np.zeros((naocc[1], naocc[1]), dtype=np.float64)
    x1324 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), x1323, (2, 1), (0, 2))
    del x1323
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1324, (0, 1), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1324, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 1, 4, 5, 6), (2, 3, 0, 4, 5, 6)) * 6.0
    del x1324
    x1325 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=np.float64)
    x1325 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.ovOV, (2, 1, 3, 4), (0, 3, 4, 2))
    x1326 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=np.float64)
    x1326 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1325, (2, 3, 4, 0), (2, 3, 1, 4))
    del x1325
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1326, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1326, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * -6.0
    del x1326
    x1327 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1]), dtype=np.float64)
    x1327 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.oVvO, (2, 3, 1, 4), (0, 4, 3, 2))
    x1328 = np.zeros((naocc[1], naocc[1], navir[1], navir[1]), dtype=np.float64)
    x1328 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), x1327, (2, 3, 4, 0), (2, 3, 1, 4))
    del x1327
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1328, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1328, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2)) * 6.0
    del x1328
    x1329 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1329 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 2), (1, 3, 4, 5))
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,sVb,sVfb)] += einsum(x1329, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * 2.0
    x1330 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1330 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1329, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1330, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -1.0
    del x1330
    x1331 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1331 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 0, 5), (1, 3, 4, 5))
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,sVb,sVfb)] += einsum(x1331, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * -2.0
    x1332 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1332 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1331, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1332, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5))
    del x1332
    x1333 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1333 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x35, (4, 0, 5, 2), (1, 4, 5, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1333, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x1333
    x1334 = np.zeros((navir[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1334 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 0, 5), (3, 4, 1, 5)) * -1.0
    x1335 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1335 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1334, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1335, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4))
    del x1335
    x1336 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1336 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x35, (4, 5, 0, 2), (1, 4, 5, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1336, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3))
    del x1336
    x1337 = np.zeros((navir[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1337 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 2), (3, 4, 1, 5)) * -1.0
    x1338 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1338 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1337, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1338, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * -1.0
    del x1338
    x1339 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1339 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1329, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x1329
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1339, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x1339
    x1340 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1340 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1331, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x1331
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1340, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3))
    del x1340
    x1341 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x1341 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 6, 2), (1, 3, 0, 4, 6, 5))
    x1342 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1342 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1341, (4, 5, 6, 7, 0, 2), (4, 5, 6, 7, 1, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1342, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3))
    del x1342
    x1343 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1343 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1341, (4, 5, 6, 0, 7, 2), (4, 5, 6, 7, 1, 3))
    del x1341
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1343, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x1343
    x1344 = np.zeros((naocc[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1344 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 5, 0, 2), (1, 4, 5, 3)) * -1.0
    x1345 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1345 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1344, (4, 5, 2, 6), (4, 3, 0, 5, 1, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1345, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3))
    del x1345
    x1346 = np.zeros((naocc[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1346 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aaaa.ovov, (4, 2, 0, 5), (1, 4, 5, 3)) * -1.0
    x1347 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1347 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1346, (4, 5, 2, 6), (4, 3, 0, 5, 1, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1347, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x1347
    x1348 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1348 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1334, (4, 5, 6, 2), (1, 4, 0, 5, 6, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1348, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -1.0
    del x1348
    x1349 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1349 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1337, (4, 5, 6, 2), (1, 4, 0, 5, 6, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1349, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3))
    del x1349
    x1350 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1350 += einsum(x11, (0, 1), t2.abab[np.ix_(soa,sOb,sva,sVb)], (2, 3, 1, 4), (3, 4, 2, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1350, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -1.0
    del x1350
    x1351 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1351 += einsum(x12, (0, 1), t2.abab[np.ix_(soa,sOb,sva,sVb)], (2, 3, 1, 4), (3, 4, 2, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1351, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5))
    del x1351
    x1352 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1352 += einsum(x11, (0, 1), t2.abab[np.ix_(soa,sob,sva,sVb)], (2, 3, 1, 4), (4, 2, 0, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1352, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * -1.0
    del x1352
    x1353 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1353 += einsum(x12, (0, 1), t2.abab[np.ix_(soa,sob,sva,sVb)], (2, 3, 1, 4), (4, 2, 0, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1353, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4))
    del x1353
    x1354 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1354 += einsum(x11, (0, 1), t2.abab[np.ix_(soa,sOb,sva,svb)], (2, 3, 1, 4), (3, 2, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1354, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x1354
    x1355 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1355 += einsum(x12, (0, 1), t2.abab[np.ix_(soa,sOb,sva,svb)], (2, 3, 1, 4), (3, 2, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1355, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3))
    del x1355
    x1356 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1356 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x980, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1356, (4, 5, 6, 0, 7, 1), (7, 6, 4, 3, 2, 5)) * 2.0
    x1357 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1357 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 0, 2), (1, 3, 4, 5))
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,sVb,sVfb)] += einsum(x1357, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * 4.0
    x1358 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1358 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1357, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1358, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -2.0
    del x1358
    x1359 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1359 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x33, (4, 0, 1, 5), (3, 4, 2, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1359, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * -2.0
    del x1359
    x1360 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1360 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x33, (4, 5, 0, 3), (1, 4, 5, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1360, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * 2.0
    del x1360
    x1361 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1361 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x980, (4, 5, 6, 7, 1, 2), (4, 5, 0, 6, 7, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1361, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1361
    x1362 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1362 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x176, (4, 5, 6, 0), (1, 3, 4, 5, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1362, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1362
    x1363 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1363 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1357, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x1357
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1363, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x1363
    x1364 = np.zeros((navir[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1364 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 5), (3, 0, 4, 5)) * -1.0
    x1365 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1365 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1364, (4, 5, 6, 3), (1, 4, 5, 6, 0, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1365, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1365
    x1366 = np.zeros((naocc[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1366 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 0, 3), (1, 4, 5, 2)) * -1.0
    x1367 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1367 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1366, (4, 5, 2, 6), (4, 3, 0, 5, 1, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1367, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x1367
    x1368 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1368 += einsum(x0, (0, 1), t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (2, 3, 1, 4), (3, 4, 2, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1368, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -2.0
    x1369 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1369 += einsum(x0, (0, 1), t2.bbbb[np.ix_(sob,sOb,svb,svb)], (2, 3, 4, 1), (3, 2, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1369, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * 2.0
    x1370 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1370 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x33, (4, 0, 5, 3), (1, 4, 5, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1370, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1370
    x1371 = np.zeros((navir[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1371 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 1, 2), (3, 4, 0, 5)) * -1.0
    x1372 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1372 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1371, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1372, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * 2.0
    del x1372
    x1373 = np.zeros((naocc[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1373 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (1, 0, 4, 5)) * -1.0
    x1374 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1374 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1373, (2, 3, 0, 4), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1374, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1374
    x1375 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1375 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1371, (4, 5, 6, 2), (1, 4, 0, 5, 6, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1375, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x1375
    x1376 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1376 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x934, (4, 5, 6, 7, 1, 3), (4, 5, 6, 7, 0, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1376, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x1376
    x1377 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1377 += einsum(x0, (0, 1), t2.abab[np.ix_(soa,sOb,sva,svb)], (2, 3, 4, 1), (3, 2, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1377, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1377
    x1378 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1]), dtype=np.float64)
    x1378 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 3, 4, 5))
    x1379 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1379 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1378, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1379, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    x1380 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1380 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x934, (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 0, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1380, (4, 5, 6, 0, 7, 1), (7, 6, 4, 3, 2, 5))
    x1381 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1381 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x13, (0, 4, 5, 2), (1, 4, 5, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1381, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    x1382 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1382 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1364, (2, 3, 4, 1), (2, 3, 4, 0))
    del x1364
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1382, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4))
    del x1382
    x1383 = np.zeros((navir[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1383 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x13, (0, 4, 1, 5), (3, 4, 2, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1383, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4))
    del x1383
    x1384 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1384 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x176, (4, 0, 5, 6), (1, 3, 4, 5, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1384, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x1384
    x1385 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1385 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1378, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1385, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x1385
    x1386 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1386 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x934, (4, 5, 6, 0, 7, 3), (4, 5, 6, 1, 7, 2))
    del x934
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1386, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x1386
    x1387 = np.zeros((naocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1387 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 4, 5, 3), (1, 5, 2, 4)) * -1.0
    x1388 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1388 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1387, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x1387
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1388, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x1388
    x1389 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1389 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1373, (2, 3, 4, 0), (2, 3, 4, 1))
    del x1373
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1389, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3))
    del x1389
    x1390 = np.zeros((navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1390 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (3, 1, 4, 5)) * -1.0
    x1391 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1391 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1390, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1391, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x1391
    x1392 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1392 += einsum(x15, (0, 1), t2.abab[np.ix_(soa,sOb,sva,sVb)], (2, 3, 1, 4), (3, 4, 2, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1392, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -1.0
    del x1392
    x1393 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1393 += einsum(x15, (0, 1), t2.abab[np.ix_(soa,sob,sva,sVb)], (2, 3, 1, 4), (4, 2, 0, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1393, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * -1.0
    del x1393
    x1394 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1394 += einsum(x15, (0, 1), t2.abab[np.ix_(soa,sOb,sva,svb)], (2, 3, 1, 4), (3, 2, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1394, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x1394
    x1395 = np.zeros((naocc[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x1395 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x981, (4, 0, 5, 3), (1, 4, 2, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1395, (4, 5, 6, 2), (1, 0, 4, 3, 6, 5)) * -1.0
    del x1395
    x1396 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1396 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x981, (4, 5, 2, 3), (4, 0, 5, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1396, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * -1.0
    del x1396
    x1397 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1397 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x981, (4, 0, 2, 5), (1, 4, 3, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1397, (4, 5, 6, 3), (1, 0, 4, 6, 2, 5)) * -1.0
    x1398 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1398 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x981, (4, 5, 2, 3), (1, 4, 0, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1398, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -1.0
    del x1398
    x1399 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1399 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x981, (4, 0, 2, 5), (4, 1, 3, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1399, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -1.0
    x1400 = np.zeros((navir[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1400 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x981, (4, 0, 5, 3), (4, 1, 2, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1400, (4, 5, 6, 2), (5, 0, 1, 3, 6, 4)) * -1.0
    del x1400
    x1401 = np.zeros((naocc[1], navir[1], nvir[0], nvir[0]), dtype=np.float64)
    x1401 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x982, (4, 0, 1, 5), (4, 3, 2, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1401, (4, 5, 6, 2), (1, 0, 4, 3, 6, 5)) * -1.0
    del x1401
    x1402 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1402 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x982, (4, 0, 5, 2), (4, 3, 1, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1402, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    x1403 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1403 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x982, (4, 5, 1, 2), (4, 0, 5, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1403, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x1403
    x1404 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1404 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x982, (4, 5, 1, 2), (4, 3, 0, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1404, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5)) * -1.0
    del x1404
    x1405 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1405 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x982, (4, 0, 5, 2), (4, 1, 5, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1405, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    x1406 = np.zeros((naocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    x1406 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x982, (4, 0, 1, 5), (4, 2, 5, 3))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1406, (4, 5, 2, 6), (1, 0, 4, 6, 5, 3)) * -1.0
    del x1406
    x1407 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1]), dtype=np.float64)
    x1407 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 0, 2), (1, 3, 4, 5))
    x1408 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1408 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1407, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1408, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -2.0
    x1409 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1]), dtype=np.float64)
    x1409 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 0, 5), (1, 3, 4, 5))
    x1410 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1410 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1409, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1410, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * 2.0
    x1411 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1411 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x16, (4, 0, 5, 3), (1, 4, 5, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1411, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * 2.0
    x1412 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1412 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x16, (4, 5, 0, 3), (1, 4, 5, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1412, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -2.0
    x1413 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1413 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 6, 2), (1, 3, 0, 4, 6, 5))
    x1414 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1414 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1413, (4, 5, 6, 7, 1, 3), (4, 5, 0, 6, 7, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1414, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * 2.0
    del x1414
    x1415 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1415 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1413, (4, 5, 6, 1, 7, 3), (4, 5, 0, 6, 7, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1415, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x1415
    x1416 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1416 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1407, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1416, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x1416
    x1417 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1417 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1409, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1417, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * 2.0
    del x1417
    x1418 = np.zeros((naocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1418 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 5, 2), (1, 0, 4, 5))
    x1419 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1419 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1418, (2, 3, 4, 0), (2, 3, 4, 1))
    del x1418
    x1420 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1420 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1419, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1420, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1420, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x1420
    x1421 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1421 += einsum(x1, (0, 1), t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (2, 3, 1, 4), (3, 4, 2, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1421, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -2.0
    x1422 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1422 += einsum(x20, (0, 1), t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (2, 3, 1, 4), (3, 4, 2, 0))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1422, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * 2.0
    x1423 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1423 += einsum(x1, (0, 1), t2.bbbb[np.ix_(sob,sOb,svb,svb)], (2, 3, 4, 1), (3, 2, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1423, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * 2.0
    x1424 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1424 += einsum(x20, (0, 1), t2.bbbb[np.ix_(sob,sOb,svb,svb)], (2, 3, 4, 1), (3, 2, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1424, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -2.0
    x1425 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1425 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x16, (4, 1, 0, 5), (3, 4, 2, 5))
    x1426 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1426 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1425, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1426, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1426, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x1426
    x1427 = np.zeros((navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1427 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 1, 2), (3, 0, 4, 5)) * -1.0
    x1428 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1428 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1427, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1428, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -2.0
    del x1428
    x1429 = np.zeros((navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1429 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 2, 1, 5), (3, 0, 4, 5)) * -1.0
    x1430 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1430 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1429, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1430, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * 2.0
    del x1430
    x1431 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1431 += einsum(x1, (0, 1), t2.abab[np.ix_(soa,sOb,sva,svb)], (2, 3, 4, 1), (3, 2, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1431, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1431
    x1432 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1432 += einsum(x20, (0, 1), t2.abab[np.ix_(soa,sOb,sva,svb)], (2, 3, 4, 1), (3, 2, 0, 4)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1432, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1432
    x1433 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1433 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x961, (4, 1, 5, 3), (4, 0, 2, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1433, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * 2.0
    del x1433
    x1434 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1434 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x961, (4, 5, 3, 2), (1, 4, 0, 5))
    x1435 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1435 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1434, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1435, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1435, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x1435
    x1436 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1436 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x961, (4, 1, 3, 5), (4, 0, 2, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1436, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * -2.0
    del x1436
    x1437 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1437 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x961, (4, 0, 3, 5), (1, 4, 2, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1437, (4, 5, 6, 3), (1, 0, 4, 6, 2, 5)) * 2.0
    x1438 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1438 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x961, (4, 0, 5, 3), (1, 4, 2, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1438, (4, 5, 6, 3), (1, 0, 4, 6, 2, 5)) * -2.0
    x1439 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1439 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x961, (4, 1, 3, 5), (4, 0, 2, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1439, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * -2.0
    x1440 = np.zeros((navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1440 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x961, (4, 1, 5, 3), (4, 0, 2, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1440, (4, 5, 6, 3), (5, 0, 1, 6, 2, 4)) * 2.0
    x1441 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1441 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x975, (4, 5, 1, 3), (4, 0, 5, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1441, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1441
    x1442 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1442 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x975, (4, 1, 5, 2), (4, 3, 0, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1442, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * 2.0
    x1443 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1443 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x975, (4, 1, 5, 3), (4, 0, 5, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1443, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1443
    x1444 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1444 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x975, (4, 5, 1, 2), (4, 3, 0, 5)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1444, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -2.0
    x1445 = np.zeros((naocc[1], navir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1445 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x975, (4, 1, 0, 5), (4, 3, 2, 5))
    x1446 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1446 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1445, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1446, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(x1446, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1))
    del x1446
    x1447 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1447 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x975, (4, 1, 5, 3), (4, 0, 5, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1447, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -2.0
    x1448 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1448 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x975, (4, 5, 1, 3), (4, 0, 5, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1448, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * 2.0
    x1449 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1449 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x980, (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1449, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -4.0
    del x1449
    x1450 = np.zeros((navir[1], nocc[0], nvir[0], nvir[1]), dtype=np.float64)
    x1450 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x981, (4, 1, 3, 5), (4, 0, 2, 5))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1450, (4, 5, 6, 3), (0, 5, 1, 2, 6, 4)) * -4.0
    del x1450
    x1451 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1451 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x982, (4, 1, 5, 3), (4, 0, 5, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1451, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * -4.0
    del x1451
    x1452 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1452 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x197, (4, 5, 6, 0), (1, 3, 4, 5, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1452, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1452
    x1453 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1453 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1356, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1356
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1453, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * 2.0
    del x1453
    x1454 = np.zeros((navir[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1454 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x981, (2, 3, 1, 4), (2, 0, 3, 4))
    x1455 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1455 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1454, (4, 5, 6, 3), (1, 4, 5, 6, 0, 2)) * -1.0
    del x1454
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1455, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3)) * -2.0
    del x1455
    x1456 = np.zeros((naocc[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1456 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x982, (2, 3, 4, 1), (2, 0, 3, 4))
    x1457 = np.zeros((naocc[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1457 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1456, (2, 3, 0, 4), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1457, (4, 5, 1, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1457
    x1458 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1458 += einsum(t2.abab[np.ix_(soa,sOb,sva,sVb)], (0, 1, 2, 3), x197, (4, 0, 5, 6), (1, 3, 4, 5, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1458, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x1458
    x1459 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1459 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1380, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1380
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1459, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3))
    del x1459
    x1460 = np.zeros((navir[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x1460 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x981, (2, 3, 4, 1), (2, 3, 0, 4))
    del x981
    x1461 = np.zeros((navir[1], nocc[0], nocc[0], nocc[1]), dtype=np.float64)
    x1461 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1460, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1461, (4, 5, 0, 6), (6, 5, 1, 3, 2, 4)) * -1.0
    del x1461
    x1462 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1462 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1460, (4, 5, 6, 2), (1, 4, 0, 5, 6, 3)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1462, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3))
    del x1462
    x1463 = np.zeros((naocc[1], nocc[0], nocc[0], nvir[1]), dtype=np.float64)
    x1463 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1456, (2, 3, 4, 0), (2, 3, 4, 1))
    del x1456
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1463, (4, 5, 0, 6), (1, 5, 4, 6, 2, 3)) * -1.0
    del x1463
    x1464 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1464 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x982, (4, 5, 6, 2), (4, 3, 0, 5, 1, 6)) * -1.0
    x1465 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1465 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1464, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1464
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1465, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3))
    del x1465
    x1466 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1466 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t1.bb[np.ix_(sOb,svb)], (2, 3), v.aabb.ovov, (4, 5, 0, 3), (2, 1, 4, 5))
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,sVb,sVfb)] += einsum(x1466, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 2, 5, 6, 3, 7), (4, 5, 0, 6, 7, 1)) * -2.0
    x1467 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1467 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1466, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1467, (4, 5, 6, 0), (1, 6, 4, 3, 2, 5))
    del x1467
    x1468 = np.zeros((naocc[1], navir[1], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1468 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1466, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x1466
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1468, (2, 3, 4, 0, 5, 6), (5, 4, 2, 6, 1, 3))
    del x1468
    x1469 = np.zeros((navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1469 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x961, (2, 3, 1, 4), (2, 0, 3, 4))
    x1470 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1470 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1469, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1470, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x1470
    x1471 = np.zeros((navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1471 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x961, (2, 3, 4, 1), (2, 0, 3, 4))
    x1472 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1472 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), x1471, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1472, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x1472
    x1473 = np.zeros((naocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1473 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x975, (2, 3, 4, 1), (2, 0, 4, 3))
    x1474 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1474 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1473, (2, 3, 4, 0), (2, 3, 4, 1))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1474, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3))
    x1475 = np.zeros((naocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1475 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1473, (2, 3, 0, 4), (2, 3, 4, 1))
    del x1473
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), x1475, (4, 5, 1, 6), (5, 0, 4, 6, 2, 3)) * -1.0
    x1476 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1]), dtype=np.float64)
    x1476 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t1.bb[np.ix_(sOb,svb)], (2, 3), v.bbbb.ovov, (4, 3, 0, 5), (2, 1, 4, 5))
    x1477 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1477 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1476, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1477, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5)) * -1.0
    x1478 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1]), dtype=np.float64)
    x1478 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), t1.bb[np.ix_(sOb,svb)], (2, 3), v.bbbb.ovov, (4, 5, 0, 3), (2, 1, 4, 5))
    x1479 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1479 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1478, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1479, (4, 5, 6, 1), (6, 0, 4, 3, 2, 5))
    x1480 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1480 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1476, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1480, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3)) * -1.0
    del x1480
    x1481 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1481 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1478, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    t3new_babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1481, (2, 3, 4, 5, 0, 6), (5, 4, 2, 1, 6, 3))
    del x1481
    x1482 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1482 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ovvv, (4, 5, 6, 3), (0, 2, 4, 1, 5, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1482, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x1482, (4, 5, 1, 6, 3, 7), (0, 6, 4, 2, 7, 5)) * 2.0
    x1483 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1483 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 4, 5, 3, 6, 7), (5, 7, 0, 4, 1, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1483, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x1483, (4, 5, 1, 6, 3, 7), (0, 6, 4, 2, 7, 5)) * 4.0
    del x1483
    x1484 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1484 += einsum(f.aa.ov, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 1, 5, 6), (4, 6, 0, 2, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1484, (2, 3, 0, 4, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1484
    x1485 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1485 += einsum(f.bb.ov, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 0, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1485, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1485
    x1486 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1486 += einsum(f.bb.ov, (0, 1), t2.abab[np.ix_(sOa,sob,sVa,svb)], (2, 3, 4, 1), (2, 4, 0, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1486, (4, 5, 1, 6), (0, 6, 4, 2, 3, 5)) * -1.0
    del x1486
    x1487 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1487 += einsum(f.bb.ov, (0, 1), t2.abab[np.ix_(soa,sob,sVa,svb)], (2, 3, 4, 1), (4, 2, 0, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1487, (4, 5, 1, 6), (5, 6, 0, 2, 3, 4))
    del x1487
    x1488 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1488 += einsum(f.bb.ov, (0, 1), t2.abab[np.ix_(sOa,sob,sva,svb)], (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1488, (4, 1, 5, 6), (0, 5, 4, 6, 3, 2))
    del x1488
    x1489 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1489 += einsum(f.aa.ov, (0, 1), t2.abab[np.ix_(sOa,sob,sva,svb)], (2, 3, 1, 4), (2, 0, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1489, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1489
    x1490 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1490 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.oOvV, (4, 5, 2, 6), (5, 6, 0, 4, 1, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1490, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1490
    x1491 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1491 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.oOoo, (0, 2, 3, 4), (2, 1, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1491, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5))
    del x1491
    x1492 = np.zeros((naocc[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1492 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.oOvv, (0, 2, 3, 4), (2, 1, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1492, (4, 5, 6, 3), (0, 1, 4, 2, 6, 5)) * -1.0
    del x1492
    x1493 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1493 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vVoo, (1, 2, 3, 4), (0, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1493, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -1.0
    del x1493
    x1494 = np.zeros((naocc[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1494 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vVvv, (1, 2, 3, 4), (0, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1494, (4, 5, 6, 3), (0, 1, 4, 2, 6, 5))
    del x1494
    x1495 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1495 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.oOoo, (0, 2, 3, 4), (2, 3, 4, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1495, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * -1.0
    del x1495
    x1496 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1496 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oOvv, (4, 5, 6, 3), (5, 2, 0, 4, 1, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1496, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1496
    x1497 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1497 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vvoo, (2, 1, 3, 4), (0, 3, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1497, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2))
    del x1497
    x1498 = np.zeros((naocc[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1498 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vvvv, (2, 1, 3, 4), (0, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1498, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2)) * -1.0
    del x1498
    x1499 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1499 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.oovV, (4, 5, 2, 6), (0, 6, 4, 5, 1, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1499, (2, 3, 0, 4, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1499
    x1500 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1500 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ovoV, (4, 2, 5, 6), (0, 6, 5, 4, 1, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1500, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1500
    x1501 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1501 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.oooo, (2, 0, 3, 4), (1, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1501, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * -1.0
    del x1501
    x1502 = np.zeros((navir[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1502 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.oovv, (2, 0, 3, 4), (1, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1502, (4, 5, 6, 3), (5, 1, 0, 2, 6, 4))
    del x1502
    x1503 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1503 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vVoo, (1, 2, 3, 4), (2, 0, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1503, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4))
    del x1503
    x1504 = np.zeros((navir[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1504 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vVvv, (1, 2, 3, 4), (2, 0, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1504, (4, 5, 6, 3), (5, 1, 0, 2, 6, 4)) * -1.0
    del x1504
    x1505 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1505 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oooo, (4, 5, 6, 1), (0, 2, 4, 5, 6, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1505, (2, 3, 0, 4, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1505
    x1506 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1506 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oovv, (4, 5, 6, 3), (0, 2, 4, 5, 1, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1506, (2, 3, 0, 4, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1506
    x1507 = np.zeros((naocc[0], navir[0], nocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    x1507 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.vvvv, (4, 5, 6, 3), (0, 2, 1, 4, 5, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1507, (2, 3, 4, 1, 5, 6), (0, 4, 2, 5, 6, 3))
    del x1507
    x1508 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1508 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1508, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x1508
    x1509 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1509 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.vOov, (2, 3, 4, 1), (3, 0, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1509, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2))
    del x1509
    x1510 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1510 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.vOov, (4, 5, 6, 3), (5, 2, 0, 1, 6, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1510, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x1510
    x1511 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1511 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.oVov, (2, 3, 4, 1), (3, 2, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1511, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4))
    del x1511
    x1512 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1512 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.oVov, (4, 5, 6, 3), (0, 5, 4, 1, 6, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1512, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x1512
    x1513 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1513 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 5, 6, 3), (0, 2, 4, 1, 6, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1513, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    x1514 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1514 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.oOvv, (2, 3, 4, 1), (3, 2, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1514, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1514
    x1515 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1515 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.oOoo, (2, 3, 4, 0), (3, 2, 4, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1515, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1515
    x1516 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1516 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.oooo, (4, 0, 5, 6), (1, 3, 4, 5, 6, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1516, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1516
    x1517 = np.zeros((naocc[0], navir[0], nocc[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1517 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.vvvv, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1517, (2, 3, 4, 5, 1, 6), (4, 0, 2, 5, 6, 3)) * 2.0
    del x1517
    x1518 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1518 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.vvoo, (4, 2, 5, 6), (1, 3, 0, 5, 6, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1518, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1518
    x1519 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1519 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.vVvv, (2, 3, 4, 1), (3, 0, 2, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1519, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * 2.0
    del x1519
    x1520 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1520 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aabb.vVoo, (3, 4, 5, 6), (1, 4, 0, 5, 6, 2)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1520, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1520
    x1521 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1521 += einsum(v.aaaa.ooov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 4, 5, 3, 6, 7), (5, 7, 0, 1, 4, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1521, (2, 3, 0, 4, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1521
    x1522 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1522 += einsum(v.aaaa.ooov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 4, 5, 3, 6, 7), (5, 7, 0, 2, 4, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1522, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1522
    x1523 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1523 += einsum(v.aaaa.ovOO, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 3, 1, 6, 7), (2, 7, 4, 0, 5, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1523, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1523
    x1524 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1524 += einsum(v.aaaa.oOvO, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 3, 2, 6, 7), (1, 7, 4, 0, 5, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1524, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1524
    x1525 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1525 += einsum(v.aabb.ovoo, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 5, 1, 6, 7), (5, 7, 4, 0, 2, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1525, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1525
    x1526 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1526 += einsum(v.aabb.ovvv, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 1, 3, 7), (6, 7, 4, 0, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1526, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1526
    x1527 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1527 += einsum(v.aaaa.ovVV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 1, 7, 3), (6, 2, 4, 0, 5, 7))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1527, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1527
    x1528 = np.zeros((navir[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1528 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.oVoo, (0, 2, 3, 4), (1, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1528, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0)) * 2.0
    del x1528
    x1529 = np.zeros((navir[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1529 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.oVvv, (0, 2, 3, 4), (1, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1529, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 7, 3, 1), (4, 5, 6, 7, 2, 0)) * -2.0
    del x1529
    x1530 = np.zeros((naocc[0], naocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1530 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vOoo, (1, 2, 3, 4), (0, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1530, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7)) * 2.0
    del x1530
    x1531 = np.zeros((naocc[0], naocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1531 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vOvv, (1, 2, 3, 4), (0, 2, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1531, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 3, 7), (4, 5, 0, 6, 2, 7)) * -2.0
    del x1531
    x1532 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=np.float64)
    x1532 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.oOOV, (0, 2, 3, 4), (2, 3, 1, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1532, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (5, 4, 0, 7, 6, 2)) * -2.0
    del x1532
    x1533 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=np.float64)
    x1533 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vVOV, (1, 2, 3, 4), (0, 3, 2, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1533, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (5, 4, 0, 7, 6, 2)) * 2.0
    del x1533
    x1534 = np.zeros((naocc[0], naocc[1], navir[1], nvir[0]), dtype=np.float64)
    x1534 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.oOOV, (0, 2, 3, 4), (2, 3, 4, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1534, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sVa,sVfb)], (4, 5, 1, 6, 7, 2), (5, 4, 0, 3, 6, 7)) * 2.0
    del x1534
    x1535 = np.zeros((naocc[0], naocc[1], navir[1], nvir[0]), dtype=np.float64)
    x1535 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vvOV, (2, 1, 3, 4), (0, 3, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1535, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sVa,sVfb)], (4, 5, 1, 6, 7, 2), (5, 4, 0, 3, 6, 7)) * -2.0
    del x1535
    x1536 = np.zeros((naocc[1], navir[0], navir[1], nocc[0]), dtype=np.float64)
    x1536 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.ooOV, (2, 0, 3, 4), (3, 1, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1536, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sva,sVfb)], (4, 5, 0, 6, 7, 2), (3, 4, 5, 7, 6, 1)) * 2.0
    del x1536
    x1537 = np.zeros((naocc[1], navir[0], navir[1], nocc[0]), dtype=np.float64)
    x1537 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vVOV, (1, 2, 3, 4), (3, 2, 4, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1537, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sva,sVfb)], (4, 5, 0, 6, 7, 2), (3, 4, 5, 7, 6, 1)) * -2.0
    del x1537
    x1538 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1538 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.ooOV, (2, 0, 3, 4), (3, 4, 2, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1538, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sVa,sVfb)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * -2.0
    del x1538
    x1539 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1539 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), v.aabb.vvOV, (2, 1, 3, 4), (3, 4, 0, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1539, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sVa,sVfb)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * 2.0
    del x1539
    x1540 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1540 += einsum(v.aabb.ooov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 4, 5, 6, 3, 7), (5, 7, 0, 4, 2, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1540, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1540
    x1541 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1541 += einsum(v.bbbb.ooov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 7, 4, 0, 1, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1541, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1541
    x1542 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1542 += einsum(v.bbbb.ooov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 1, 5, 6, 3, 7), (5, 7, 4, 0, 2, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1542, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1542
    x1543 = np.zeros((naocc[0], naocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1543 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.OOov, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1543, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7)) * 2.0
    del x1543
    x1544 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1544 += einsum(v.aabb.OOov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1544, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1544
    x1545 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1545 += einsum(v.aabb.vvov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 1, 3, 7), (6, 7, 4, 5, 2, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1545, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1545
    x1546 = np.zeros((navir[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1546 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.VVov, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1546, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0)) * -2.0
    del x1546
    x1547 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1547 += einsum(v.aabb.VVov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 7, 3, 1), (6, 0, 4, 5, 2, 7))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1547, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1547
    x1548 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1548 += einsum(v.aabb.ovoo, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 0, 5, 6, 1, 7), (5, 7, 4, 2, 3, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1548, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3)) * -6.0
    del x1548
    x1549 = np.zeros((naocc[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1549 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oOov, (0, 4, 1, 5), (4, 2, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1549, (4, 5, 6, 3), (0, 1, 4, 2, 6, 5))
    del x1549
    x1550 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1550 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oOov, (0, 4, 5, 3), (4, 2, 1, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1550, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5))
    del x1550
    x1551 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1551 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.oOov, (0, 4, 5, 3), (4, 1, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1551, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * -1.0
    del x1551
    x1552 = np.zeros((naocc[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1552 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.oOov, (0, 4, 1, 5), (4, 2, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1552, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2)) * -1.0
    del x1552
    x1553 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1553 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vVov, (2, 4, 5, 3), (4, 0, 1, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1553, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4))
    del x1553
    x1554 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1554 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vVov, (2, 4, 5, 3), (0, 4, 1, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1554, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -1.0
    del x1554
    x1555 = np.zeros((navir[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1555 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vVov, (2, 4, 1, 5), (4, 0, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1555, (4, 5, 6, 3), (5, 1, 0, 2, 6, 4))
    del x1555
    x1556 = np.zeros((naocc[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1556 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vVov, (2, 4, 1, 5), (0, 4, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1556, (4, 5, 6, 3), (0, 1, 4, 2, 6, 5)) * -1.0
    del x1556
    x1557 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1557 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 1, 3), (0, 2, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1557, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -1.0
    del x1557
    x1558 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1558 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 5, 3), (0, 2, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1558, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5))
    del x1558
    x1559 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1559 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 5, 6, 3), (0, 2, 4, 5, 1, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1559, (4, 5, 6, 0, 7, 1), (6, 7, 4, 2, 3, 5))
    x1560 = np.zeros((naocc[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1560 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (1, 4, 5, 3), (0, 2, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1560, (4, 5, 3, 6), (0, 1, 4, 2, 6, 5)) * -1.0
    del x1560
    x1561 = np.zeros((naocc[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1561 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (1, 3, 4, 5), (0, 2, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1561, (4, 5, 6, 3), (0, 1, 4, 2, 6, 5))
    del x1561
    x1562 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1562 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (4, 5, 6, 3), (0, 2, 1, 4, 5, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1562, (4, 5, 6, 1, 3, 7), (0, 6, 4, 2, 7, 5))
    del x1562
    x1563 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1563 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (4, 3, 5, 6), (0, 2, 1, 4, 5, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1563, (4, 5, 6, 1, 3, 7), (0, 6, 4, 2, 7, 5)) * -1.0
    del x1563
    x1564 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1564 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.vvov, (4, 5, 6, 3), (0, 2, 1, 6, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1564, (4, 5, 6, 1, 7, 2), (0, 6, 4, 7, 3, 5)) * -1.0
    x1565 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1565 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x490, (4, 5, 6, 2), (4, 5, 0, 1, 6, 3))
    del x490
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1565, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1565, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x1565
    x1566 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1566 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ovoO, (0, 2, 4, 5), (5, 4, 1, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1566, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1566
    x1567 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1567 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ovoO, (4, 2, 0, 5), (5, 4, 1, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1567, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1567
    x1568 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1568 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 5), (1, 3, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1568, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -2.0
    del x1568
    x1569 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1569 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovoo, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1569, (4, 5, 6, 0, 1, 7), (6, 7, 4, 2, 3, 5)) * 2.0
    x1570 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1570 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x514, (4, 5, 6, 0), (4, 5, 6, 1, 2, 3))
    del x514
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1570, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1570, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x1570
    x1571 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1571 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ovvV, (0, 2, 4, 5), (5, 1, 4, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1571, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * 2.0
    del x1571
    x1572 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1572 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.aaaa.ovvV, (0, 4, 2, 5), (5, 1, 4, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1572, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * -2.0
    del x1572
    x1573 = np.zeros((naocc[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1573 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovvv, (0, 2, 4, 5), (1, 3, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1573, (4, 5, 6, 3), (0, 1, 4, 2, 6, 5)) * 2.0
    del x1573
    x1574 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1574 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovvv, (4, 2, 5, 6), (1, 3, 0, 4, 5, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1574, (4, 5, 6, 0, 3, 7), (6, 1, 4, 2, 7, 5)) * -2.0
    x1575 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1575 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 1, 3), (0, 4, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1575, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2))
    del x1575
    x1576 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1576 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 5, 3), (2, 0, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1576, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * -1.0
    del x1576
    x1577 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1577 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 5, 3), (0, 4, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1577, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * -1.0
    del x1577
    x1578 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1578 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 1, 3), (2, 0, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1578, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4))
    del x1578
    x1579 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1579 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 0, 5, 3), (2, 4, 1, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1579, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * -1.0
    del x1579
    x1580 = np.zeros((navir[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1580 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ooov, (4, 0, 1, 5), (2, 4, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1580, (4, 5, 6, 3), (5, 1, 0, 2, 6, 4)) * -1.0
    del x1580
    x1581 = np.zeros((naocc[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1581 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (1, 4, 5, 3), (0, 2, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1581, (4, 5, 3, 6), (0, 1, 4, 5, 6, 2))
    del x1581
    x1582 = np.zeros((naocc[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1582 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (1, 3, 4, 5), (0, 2, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1582, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2)) * -1.0
    del x1582
    x1583 = np.zeros((navir[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1583 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (1, 3, 4, 5), (2, 0, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1583, (4, 5, 6, 3), (5, 1, 0, 2, 6, 4)) * -1.0
    del x1583
    x1584 = np.zeros((navir[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1584 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (1, 4, 5, 3), (2, 0, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1584, (4, 5, 3, 6), (5, 1, 0, 2, 6, 4))
    del x1584
    x1585 = np.zeros((naocc[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1585 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vvov, (4, 2, 1, 5), (0, 4, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1585, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2))
    del x1585
    x1586 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1586 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vvov, (4, 2, 5, 3), (0, 1, 5, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1586, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2))
    del x1586
    x1587 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1587 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aabb.ovoo, (0, 3, 4, 5), (1, 4, 5, 2)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1587, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * 2.0
    del x1587
    x1588 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1588 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ovoo, (0, 4, 5, 1), (2, 5, 4, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1588, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * 2.0
    del x1588
    x1589 = np.zeros((naocc[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1589 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aabb.ovvv, (0, 3, 4, 5), (1, 2, 4, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1589, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2)) * -2.0
    del x1589
    x1590 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1590 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x532, (4, 5, 0, 6), (4, 2, 5, 1, 6, 3))
    del x532
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1590, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1590, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x1590
    x1591 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1591 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ovvv, (0, 4, 5, 3), (2, 1, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1591, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * -2.0
    del x1591
    x1592 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1592 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x510, (4, 5, 6, 2), (0, 4, 5, 1, 6, 3))
    del x510
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1592, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1592, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x1592
    x1593 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1593 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovoo, (1, 2, 4, 5), (3, 0, 4, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1593, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * 2.0
    del x1593
    x1594 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1594 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovoo, (4, 2, 5, 1), (0, 4, 5, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1594, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1594
    x1595 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1595 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovvv, (4, 2, 5, 3), (0, 4, 1, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1595, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1595
    x1596 = np.zeros((navir[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1596 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovvv, (1, 2, 4, 5), (3, 0, 4, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1596, (4, 5, 6, 3), (5, 1, 0, 2, 6, 4)) * -2.0
    del x1596
    x1597 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1597 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.aabb.oOov, (4, 5, 1, 3), (5, 4, 0, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1597, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * -4.0
    del x1597
    x1598 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1598 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.aabb.vVov, (4, 5, 1, 3), (5, 0, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1598, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * 4.0
    del x1598
    x1599 = np.zeros((naocc[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1599 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 2, 4, 1, 5, 6), (4, 6, 5, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1599, (4, 5, 6, 3), (0, 1, 4, 2, 6, 5)) * -2.0
    del x1599
    x1600 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1600 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 4, 5, 1, 3, 6), (5, 6, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1600, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -2.0
    del x1600
    x1601 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1], nvir[0], nvir[0]), dtype=np.float64)
    x1601 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 4, 5, 6, 3, 7), (5, 7, 4, 2, 6, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1601, (4, 5, 6, 1, 7, 2), (0, 6, 4, 7, 3, 5)) * 2.0
    del x1601
    x1602 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1602 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 4, 5, 1, 6, 7), (5, 7, 4, 2, 6, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1602, (4, 5, 6, 1, 7, 3), (0, 6, 4, 2, 7, 5)) * 2.0
    del x1602
    x1603 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1603 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 1, 6, 7), (5, 7, 4, 0, 6, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1603, (4, 5, 6, 0, 7, 3), (6, 1, 4, 2, 7, 5)) * 2.0
    del x1603
    x1604 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1604 += einsum(v.aabb.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 1, 3, 7), (6, 7, 4, 0, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1604, (4, 5, 6, 0, 7, 1), (6, 7, 4, 2, 3, 5)) * 2.0
    x1605 = np.zeros((naocc[1], navir[1], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1605 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.oVvO, (4, 5, 3, 6), (6, 5, 0, 1, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.babbab[np.ix_(sob,sOa,sOfb,svb,sVa,sVfb)], (0, 1, 2, 3, 4, 5), x1605, (2, 5, 6, 7, 0, 8), (6, 7, 1, 8, 3, 4))
    del x1605
    x1606 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1606 += einsum(v.bbbb.ovOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sVa,sVfb)], (4, 5, 2, 1, 6, 3), (5, 6, 4, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1606, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -2.0
    del x1606
    x1607 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1607 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovOV, (1, 3, 4, 5), (4, 5, 0, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1607, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sVa,sVfb)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * 2.0
    del x1607
    x1608 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1608 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.oVvO, (1, 4, 3, 5), (5, 4, 0, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1608, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sVa,sVfb)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * -2.0
    del x1608
    x1609 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1609 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x583, (4, 5, 6, 2), (4, 5, 0, 1, 6, 3))
    del x583
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1609, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -3.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1609, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -3.0
    del x1609
    x1610 = np.zeros((naocc[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1610 += einsum(v.aabb.oVov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 2, 4, 5, 6, 1), (4, 5, 6, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1610, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2)) * -2.0
    del x1610
    x1611 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1611 += einsum(v.aabb.oVov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 4, 5, 6, 3, 1), (5, 4, 2, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1611, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * -2.0
    del x1611
    x1612 = np.zeros((navir[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1612 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oVov, (0, 4, 5, 3), (2, 4, 1, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1612, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0)) * 2.0
    del x1612
    x1613 = np.zeros((navir[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1613 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oVov, (0, 4, 1, 5), (2, 4, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1613, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 7, 3, 1), (4, 5, 6, 7, 2, 0)) * 2.0
    del x1613
    x1614 = np.zeros((naocc[1], navir[0], navir[1], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1614 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovOV, (4, 3, 5, 6), (5, 2, 6, 0, 1, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.babbab[np.ix_(sob,sOa,sOfb,svb,sva,sVfb)], (0, 1, 2, 3, 4, 5), x1614, (2, 6, 5, 7, 8, 0), (7, 8, 1, 4, 3, 6))
    del x1614
    x1615 = np.zeros((naocc[1], navir[0], navir[1], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1615 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.oVvO, (4, 5, 3, 6), (6, 2, 5, 0, 1, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.babbab[np.ix_(sob,sOa,sOfb,svb,sva,sVfb)], (0, 1, 2, 3, 4, 5), x1615, (2, 6, 5, 7, 8, 0), (7, 8, 1, 4, 3, 6)) * -1.0
    del x1615
    x1616 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1616 += einsum(v.bbbb.ovOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sva,sVfb)], (4, 5, 2, 1, 6, 3), (5, 4, 0, 6)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1616, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * -2.0
    del x1616
    x1617 = np.zeros((naocc[1], navir[0], navir[1], nocc[0]), dtype=np.float64)
    x1617 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovOV, (1, 3, 4, 5), (4, 2, 5, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1617, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sva,sVfb)], (4, 5, 0, 6, 7, 2), (3, 4, 5, 7, 6, 1)) * -2.0
    del x1617
    x1618 = np.zeros((naocc[1], navir[0], navir[1], nocc[0]), dtype=np.float64)
    x1618 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.oVvO, (1, 4, 3, 5), (5, 2, 4, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1618, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sva,sVfb)], (4, 5, 0, 6, 7, 2), (3, 4, 5, 7, 6, 1)) * 2.0
    del x1618
    x1619 = np.zeros((naocc[0], naocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1619 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vOov, (2, 4, 5, 3), (0, 4, 1, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1619, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7)) * 2.0
    del x1619
    x1620 = np.zeros((navir[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1620 += einsum(v.aabb.vOov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 1, 0, 5, 6), (6, 4, 5, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1620, (4, 5, 6, 3), (5, 1, 0, 2, 6, 4)) * -2.0
    del x1620
    x1621 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1621 += einsum(v.aabb.vOov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 0, 3, 6), (6, 4, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1621, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * -2.0
    del x1621
    x1622 = np.zeros((naocc[0], naocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1622 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.vOov, (2, 4, 1, 5), (0, 4, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1622, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 3, 7), (4, 5, 0, 6, 2, 7)) * 2.0
    del x1622
    x1623 = np.zeros((naocc[0], naocc[1], navir[1], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1623 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovOV, (4, 3, 5, 6), (0, 5, 6, 1, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.babbab[np.ix_(sob,soa,sOfb,svb,sVa,sVfb)], (0, 1, 2, 3, 4, 5), x1623, (6, 2, 5, 7, 0, 8), (1, 7, 6, 8, 3, 4))
    del x1623
    x1624 = np.zeros((naocc[0], naocc[1], navir[1], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1624 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.oVvO, (4, 5, 3, 6), (0, 6, 5, 1, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.babbab[np.ix_(sob,soa,sOfb,svb,sVa,sVfb)], (0, 1, 2, 3, 4, 5), x1624, (6, 2, 5, 7, 0, 8), (1, 7, 6, 8, 3, 4)) * -1.0
    del x1624
    x1625 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1625 += einsum(v.bbbb.ovOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sVa,sVfb)], (4, 5, 2, 1, 6, 3), (6, 5, 4, 0)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1625, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * -2.0
    del x1625
    x1626 = np.zeros((naocc[0], naocc[1], navir[1], nvir[0]), dtype=np.float64)
    x1626 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.ovOV, (1, 3, 4, 5), (0, 4, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1626, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sVa,sVfb)], (4, 5, 1, 6, 7, 2), (5, 4, 0, 3, 6, 7)) * -2.0
    del x1626
    x1627 = np.zeros((naocc[0], naocc[1], navir[1], nvir[0]), dtype=np.float64)
    x1627 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.bbbb.oVvO, (1, 4, 3, 5), (0, 5, 4, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1627, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sVa,sVfb)], (4, 5, 1, 6, 7, 2), (5, 4, 0, 3, 6, 7)) * 2.0
    del x1627
    x1628 = np.zeros((naocc[0], naocc[0], navir[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1628 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.OVov, (4, 5, 6, 3), (0, 4, 2, 5, 1, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 1, 2, 3, 4, 5), x1628, (6, 2, 7, 5, 8, 1), (0, 8, 6, 3, 4, 7)) * -2.0
    del x1628
    x1629 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1629 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovOV, (4, 3, 5, 6), (0, 5, 2, 6, 1, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (0, 1, 2, 3, 4, 5), x1629, (6, 2, 7, 5, 8, 0), (1, 8, 6, 4, 3, 7)) * -1.0
    del x1629
    x1630 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], nocc[1], nocc[1]), dtype=np.float64)
    x1630 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.oVvO, (4, 5, 3, 6), (0, 6, 2, 5, 1, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (0, 1, 2, 3, 4, 5), x1630, (6, 2, 7, 5, 8, 0), (1, 8, 6, 4, 3, 7))
    del x1630
    x1631 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=np.float64)
    x1631 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovOV, (1, 3, 4, 5), (0, 4, 2, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1631, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (5, 4, 0, 7, 6, 2)) * 2.0
    del x1631
    x1632 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=np.float64)
    x1632 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.oVvO, (1, 4, 3, 5), (0, 5, 2, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1632, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (5, 4, 0, 7, 6, 2)) * -2.0
    del x1632
    x1633 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1633 += einsum(x37, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 2, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5)) * -1.0
    del x37
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1633, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1633, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x1633
    x1634 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1634 += einsum(v.aaaa.ovov, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 4, 5, 1, 6, 7), (5, 7, 0, 4, 3, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x1634, (4, 5, 1, 6, 3, 7), (0, 6, 4, 2, 7, 5)) * -4.0
    del x1634
    x1635 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1635 += einsum(x18, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 1, 3, 4, 5, 6), (3, 6, 2, 0, 4, 5)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1635, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1635, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x1635
    x1636 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1636 += einsum(x111, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 1, 5, 6), (4, 6, 2, 3, 0, 5)) * -1.0
    del x111
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1636, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1636, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x1636
    x1637 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1637 += einsum(x183, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 5, 0)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1637, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1637, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x1637
    x1638 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1638 += einsum(v.aaaa.ovoV, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 4, 5, 1, 6, 3), (5, 0, 4, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1638, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * 4.0
    del x1638
    x1639 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1639 += einsum(v.aaaa.ovvO, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 4, 3, 2, 5, 6), (6, 4, 1, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1639, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * 2.0
    del x1639
    x1640 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1640 += einsum(x608, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (3, 4, 1, 5, 6, 7), (0, 7, 2, 4, 5, 6))
    del x608
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1640, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1640, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x1640
    x1641 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1641 += einsum(v.aaaa.ovvO, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (0, 4, 3, 1, 5, 6), (6, 4, 2, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1641, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * -2.0
    del x1641
    x1642 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1642 += einsum(x615, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 1, 4, 5, 6), (0, 6, 2, 3, 4, 5))
    del x615
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1642, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1642, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x1642
    x1643 = np.zeros((naocc[0], naocc[1], navir[0], navir[1], nocc[0], nocc[0]), dtype=np.float64)
    x1643 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovOV, (4, 2, 5, 6), (1, 5, 3, 6, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (0, 1, 2, 3, 4, 5), x1643, (6, 2, 7, 5, 8, 1), (8, 0, 6, 4, 3, 7)) * -4.0
    del x1643
    x1644 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=np.float64)
    x1644 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovOV, (0, 2, 4, 5), (1, 4, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1644, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (5, 4, 0, 7, 6, 2)) * 4.0
    del x1644
    x1645 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x1645 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aabb.ovOV, (4, 3, 5, 6), (1, 5, 6, 0, 4, 2)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t3.babbab[np.ix_(sob,soa,sOfb,svb,sVa,sVfb)], (0, 1, 2, 3, 4, 5), x1645, (6, 2, 5, 7, 1, 8), (7, 0, 6, 8, 3, 4)) * 4.0
    del x1645
    x1646 = np.zeros((naocc[0], naocc[1], navir[1], nvir[0]), dtype=np.float64)
    x1646 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), v.aabb.ovOV, (0, 3, 4, 5), (1, 4, 5, 2)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1646, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sVa,sVfb)], (4, 5, 1, 6, 7, 2), (5, 4, 0, 3, 6, 7)) * -4.0
    del x1646
    x1647 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1647 += einsum(v.aabb.ovOV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sva,sVfb)], (4, 5, 2, 6, 1, 3), (5, 0, 4, 6)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1647, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * 4.0
    del x1647
    x1648 = np.zeros((naocc[1], navir[0], navir[1], nocc[0]), dtype=np.float64)
    x1648 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), v.aabb.ovOV, (1, 2, 4, 5), (4, 3, 5, 0)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1648, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sva,sVfb)], (4, 5, 0, 6, 7, 2), (3, 4, 5, 7, 6, 1)) * -4.0
    del x1648
    x1649 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1649 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), v.aabb.ovOV, (1, 3, 4, 5), (4, 5, 0, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1649, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sVa,sVfb)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * 4.0
    del x1649
    x1650 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1650 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x656, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x656
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1650, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1650
    x1651 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1651 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x653, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x653
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1651, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1651
    x1652 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1652 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x686, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x686
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1652, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1652
    x1653 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1653 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x683, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x683
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1653, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1653
    x1654 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1654 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t1.aa[np.ix_(sOa,sva)], (2, 3), v.aabb.ovoo, (0, 3, 4, 5), (2, 1, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1654, (4, 5, 1, 6), (0, 6, 4, 2, 3, 5))
    del x1654
    x1655 = np.zeros((naocc[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1655 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), t1.aa[np.ix_(sOa,sva)], (2, 3), v.aabb.ovvv, (0, 3, 4, 5), (2, 1, 4, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1655, (4, 5, 3, 6), (0, 1, 4, 2, 6, 5)) * -1.0
    del x1655
    x1656 = np.zeros((naocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1656 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.ovoo, (2, 1, 3, 4), (0, 2, 3, 4))
    x1657 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1657 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1656, (2, 0, 3, 4), (2, 4, 3, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1657, (4, 1, 5, 6), (0, 5, 4, 6, 3, 2)) * -1.0
    del x1657
    x1658 = np.zeros((naocc[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1658 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.ovvv, (2, 1, 3, 4), (0, 2, 3, 4))
    x1659 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1659 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1658, (4, 5, 6, 3), (4, 2, 0, 5, 1, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1659, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1659
    x1660 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1660 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x640, (4, 5, 6, 2), (0, 4, 5, 6, 1, 3))
    del x640
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1660, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1660
    x1661 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1661 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x646, (4, 5, 6, 2), (0, 4, 5, 6, 1, 3))
    del x646
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1661, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1661
    x1662 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1662 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x666, (4, 5, 6, 2), (0, 4, 6, 5, 1, 3))
    del x666
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1662, (2, 3, 0, 4, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1662
    x1663 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1663 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x663, (4, 5, 6, 2), (0, 4, 5, 6, 1, 3))
    del x663
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1663, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1663
    x1664 = np.zeros((navir[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1664 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.ovoo, (0, 2, 3, 4), (1, 3, 4, 2))
    x1665 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1665 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1664, (2, 3, 4, 1), (2, 0, 4, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1665, (4, 5, 1, 6), (5, 6, 0, 2, 3, 4)) * -1.0
    del x1665
    x1666 = np.zeros((navir[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1666 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.ovvv, (0, 2, 3, 4), (1, 2, 3, 4))
    x1667 = np.zeros((navir[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1667 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1666, (2, 1, 3, 4), (2, 0, 4, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1667, (4, 5, 3, 6), (5, 1, 0, 2, 6, 4))
    del x1667
    x1668 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1668 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x151, (4, 5, 6, 1), (0, 2, 4, 5, 6, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1668, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1668
    x1669 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1669 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1482, (2, 3, 4, 5, 1, 6), (2, 3, 0, 4, 5, 6))
    del x1482
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1669, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1669
    x1670 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1670 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x403, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1670, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5))
    del x1670
    x1671 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1671 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x403, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x403
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1671, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x1671
    x1672 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1672 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x409, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1672, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -1.0
    del x1672
    x1673 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1673 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x409, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x409
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1673, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x1673
    x1674 = np.zeros((naocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1674 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.oOov, (2, 3, 4, 1), (3, 2, 0, 4))
    x1675 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1675 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1674, (2, 0, 3, 4), (2, 3, 4, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1675, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * -1.0
    del x1675
    x1676 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1676 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.oOov, (4, 5, 6, 3), (5, 2, 0, 4, 1, 6))
    x1677 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1677 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1676, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1676
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1677, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1677
    x1678 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1678 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x407, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1678, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2))
    del x1678
    x1679 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1679 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x407, (4, 5, 6, 3), (4, 2, 0, 1, 5, 6))
    del x407
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1679, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x1679
    x1680 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1680 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x401, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1680, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * -1.0
    del x1680
    x1681 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1681 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x401, (4, 5, 6, 3), (0, 4, 5, 1, 6, 2))
    del x401
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1681, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x1681
    x1682 = np.zeros((navir[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1682 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.aabb.vVov, (2, 3, 4, 1), (3, 0, 4, 2))
    x1683 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1683 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1682, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1683, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4))
    del x1683
    x1684 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1684 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x405, (4, 5, 6, 3), (0, 4, 5, 1, 6, 2))
    del x405
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1684, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x1684
    x1685 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1685 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x156, (4, 5, 6, 1), (0, 2, 5, 4, 6, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1685, (2, 3, 0, 4, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1685
    x1686 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1686 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1559, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1559
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1686, (2, 3, 0, 4, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1686
    x1687 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1687 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1564, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x1564
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1687, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x1687
    x1688 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1688 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x151, (4, 0, 5, 6), (1, 3, 4, 6, 5, 2))
    del x151
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1688, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1688
    x1689 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1689 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1574, (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 0, 6))
    del x1574
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1689, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1689
    x1690 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1690 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1569, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1569
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1690, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1690
    x1691 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1691 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1666, (2, 3, 4, 1), (2, 0, 3, 4))
    del x1666
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1691, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * -2.0
    del x1691
    x1692 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1692 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1664, (4, 5, 6, 3), (1, 4, 0, 6, 5, 2)) * -1.0
    del x1664
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1692, (2, 3, 4, 0, 5, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1692
    x1693 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1693 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1658, (2, 3, 4, 1), (2, 3, 0, 4))
    del x1658
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1693, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1693
    x1694 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1694 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1656, (2, 3, 4, 0), (2, 3, 4, 1))
    del x1656
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1694, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1694
    x1695 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1695 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1674, (2, 3, 4, 0), (2, 3, 4, 1))
    del x1674
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1695, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1695
    x1696 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1696 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x156, (4, 0, 5, 6), (1, 3, 4, 5, 6, 2))
    del x156
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1696, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1696
    x1697 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1697 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x462, (2, 3, 4, 5, 6, 1), (2, 3, 4, 0, 5, 6))
    del x462
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1697, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1697
    x1698 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1698 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1682, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    del x1682
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1698, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1698
    x1699 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1699 += einsum(x35, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 4, 5, 3, 6, 7), (5, 7, 0, 2, 4, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1699, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1699
    x1700 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1700 += einsum(x35, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 4, 5, 3, 6, 7), (5, 7, 0, 1, 4, 6))
    del x35
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1700, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1700
    x1701 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1701 += einsum(x11, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 1, 5, 6), (4, 6, 2, 0, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1701, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1701
    x1702 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1702 += einsum(x12, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 1, 5, 6), (4, 6, 2, 0, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1702, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1702
    x1703 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1703 += einsum(x738, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 3, 7, 1), (6, 0, 4, 2, 5, 7))
    del x738
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1703, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1703
    x1704 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1704 += einsum(x748, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 3, 6, 7), (0, 7, 4, 2, 5, 6))
    del x748
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1704, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1704
    x1705 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1705 += einsum(x745, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 3, 6, 7), (0, 7, 4, 2, 5, 6))
    del x745
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1705, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1705
    x1706 = np.zeros((naocc[0], naocc[1], navir[1], nocc[0]), dtype=np.float64)
    x1706 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.ovOV, (2, 1, 3, 4), (0, 3, 4, 2))
    x1707 = np.zeros((naocc[0], naocc[1], navir[0], navir[1]), dtype=np.float64)
    x1707 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), x1706, (2, 3, 4, 0), (2, 3, 1, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1707, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 5, 1, 6, 7, 3), (5, 4, 0, 7, 6, 2)) * -2.0
    del x1707
    x1708 = np.zeros((naocc[0], naocc[1], navir[1], nvir[0]), dtype=np.float64)
    x1708 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1706, (2, 3, 4, 0), (2, 3, 4, 1))
    del x1706
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1708, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sVa,sVfb)], (4, 5, 1, 6, 7, 2), (5, 4, 0, 3, 6, 7)) * 2.0
    del x1708
    x1709 = np.zeros((naocc[1], navir[0], navir[1], nvir[0]), dtype=np.float64)
    x1709 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.ovOV, (0, 2, 3, 4), (3, 1, 4, 2))
    x1710 = np.zeros((naocc[1], navir[0], navir[1], nocc[0]), dtype=np.float64)
    x1710 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1709, (2, 3, 4, 1), (2, 3, 4, 0))
    del x1709
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1710, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sva,sVfb)], (4, 5, 0, 6, 7, 2), (3, 4, 5, 7, 6, 1)) * 2.0
    del x1710
    x1711 = np.zeros((naocc[1], navir[1], nocc[0], nvir[0]), dtype=np.float64)
    x1711 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x166, (2, 3, 4, 0), (2, 3, 4, 1))
    del x166
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1711, (0, 1, 2, 3), t3.babbab[np.ix_(sob,sOa,sOfb,svb,sVa,sVfb)], (4, 5, 0, 6, 7, 1), (2, 4, 5, 3, 6, 7)) * -2.0
    del x1711
    x1712 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1712 += einsum(x33, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (1, 4, 5, 6, 3, 7), (5, 7, 0, 4, 2, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1712, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1712
    x1713 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1713 += einsum(x13, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 3, 6, 7), (5, 7, 4, 0, 1, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1713, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1713
    x1714 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1714 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1604, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1604
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1714, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1714
    x1715 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1715 += einsum(x15, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 1, 5, 6), (4, 6, 2, 0, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1715, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1715
    x1716 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1716 += einsum(x0, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1716, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1716
    x1717 = np.zeros((navir[0], navir[0], nocc[1], nvir[1]), dtype=np.float64)
    x1717 += einsum(t1.aa[np.ix_(soa,sVa)], (0, 1), v.aabb.oVov, (0, 2, 3, 4), (1, 2, 3, 4))
    x1718 = np.zeros((navir[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1718 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1717, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1718, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 5, 6, 7, 1), (4, 2, 5, 6, 7, 0)) * 2.0
    del x1718
    x1719 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1719 += einsum(x1717, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 6, 7, 3, 1), (6, 0, 4, 5, 2, 7))
    del x1717
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1719, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1719
    x1720 = np.zeros((naocc[0], naocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1720 += einsum(t1.aa[np.ix_(sOa,sva)], (0, 1), v.aabb.vOov, (1, 2, 3, 4), (0, 2, 3, 4))
    x1721 = np.zeros((naocc[0], naocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1721 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1720, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1721, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 3, 1, 5, 6, 7), (4, 2, 0, 5, 6, 7)) * 2.0
    del x1721
    x1722 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1722 += einsum(x1720, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    del x1720
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1722, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1722
    x1723 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1723 += einsum(x16, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 1, 5, 6, 3, 7), (5, 7, 4, 0, 2, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1723, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1723
    x1724 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1724 += einsum(x16, (0, 1, 2, 3), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (4, 2, 5, 6, 3, 7), (5, 7, 4, 0, 1, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1724, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1724
    x1725 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1725 += einsum(x1, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1725, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1725
    x1726 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1726 += einsum(x20, (0, 1), t3.abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1726, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1726
    x1727 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1727 += einsum(x13, (0, 1, 2, 3), t3.aaaaaa[np.ix_(soa,soa,sOfa,sva,sva,sVfa)], (4, 0, 5, 6, 3, 7), (5, 7, 4, 1, 2, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1727, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -6.0
    del x1727
    x1728 = np.zeros((naocc[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1728 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x716, (4, 0, 1, 5), (4, 2, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1728, (4, 5, 6, 3), (0, 1, 4, 2, 6, 5))
    del x1728
    x1729 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1729 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x716, (4, 0, 5, 3), (4, 2, 1, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1729, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5))
    del x1729
    x1730 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1730 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x716, (4, 0, 5, 3), (4, 1, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1730, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * -1.0
    del x1730
    x1731 = np.zeros((naocc[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1731 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x716, (4, 0, 1, 5), (4, 2, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1731, (4, 5, 6, 3), (0, 1, 4, 5, 6, 2)) * -1.0
    del x1731
    x1732 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1732 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x713, (4, 5, 2, 3), (4, 0, 1, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1732, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * -1.0
    del x1732
    x1733 = np.zeros((naocc[0], navir[0], nvir[1], nvir[1]), dtype=np.float64)
    x1733 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x713, (4, 1, 2, 5), (0, 4, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1733, (4, 5, 6, 3), (0, 1, 4, 2, 6, 5))
    del x1733
    x1734 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1734 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x713, (4, 5, 2, 3), (0, 4, 1, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1734, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5))
    del x1734
    x1735 = np.zeros((navir[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1735 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x713, (4, 1, 2, 5), (4, 0, 3, 5))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1735, (4, 5, 6, 3), (5, 1, 0, 2, 6, 4)) * -1.0
    del x1735
    x1736 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1736 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1513, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1736, (4, 5, 6, 0, 7, 1), (6, 7, 4, 2, 3, 5))
    x1737 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1737 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x792, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x792
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1737, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1737
    x1738 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1738 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x176, (4, 5, 6, 1), (0, 2, 4, 5, 6, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1738, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1738
    x1739 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1739 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1513, (4, 5, 6, 7, 1, 2), (4, 5, 0, 6, 7, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1739, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1739
    x1740 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1740 += einsum(x0, (0, 1), t2.abab[np.ix_(sOa,sob,sVa,svb)], (2, 3, 4, 1), (2, 4, 3, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1740, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -1.0
    del x1740
    x1741 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1741 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x798, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x798
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1741, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1741
    x1742 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1742 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x802, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x802
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1742, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1742
    x1743 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1743 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x361, (4, 5, 6, 7, 0, 2), (4, 5, 6, 7, 1, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1743, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1743
    x1744 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1744 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x361, (4, 5, 6, 0, 7, 2), (4, 5, 6, 7, 1, 3))
    del x361
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1744, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1744
    x1745 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1745 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x856, (4, 5, 6, 0), (4, 5, 6, 1, 2, 3)) * -1.0
    del x856
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1745, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1745, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x1745
    x1746 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1746 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x375, (4, 0, 5, 2), (4, 1, 5, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1746, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * 2.0
    del x1746
    x1747 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1747 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x375, (4, 0, 2, 5), (4, 1, 5, 3))
    del x375
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1747, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * -2.0
    del x1747
    x1748 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1748 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x874, (4, 5, 6, 2), (4, 5, 0, 1, 6, 3)) * -1.0
    del x874
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1748, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1748, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1))
    del x1748
    x1749 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1749 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x389, (4, 5, 0, 2), (4, 5, 1, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1749, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1749
    x1750 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1750 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x389, (4, 0, 5, 2), (4, 5, 1, 3))
    del x389
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1750, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1750
    x1751 = np.zeros((navir[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1751 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 4, 5, 3), (2, 1, 5, 4))
    x1752 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1752 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1751, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1752, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * -1.0
    del x1752
    x1753 = np.zeros((navir[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1753 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x33, (4, 0, 1, 5), (2, 4, 3, 5))
    del x33
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1753, (4, 5, 6, 3), (5, 1, 0, 2, 6, 4)) * -1.0
    del x1753
    x1754 = np.zeros((naocc[0], nocc[0], nvir[1], nvir[1]), dtype=np.float64)
    x1754 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 2, 1, 5), (0, 4, 3, 5))
    x1755 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1755 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1754, (4, 5, 6, 3), (4, 2, 0, 5, 1, 6))
    del x1754
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1755, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1755
    x1756 = np.zeros((naocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1756 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (4, 2, 5, 3), (0, 4, 1, 5))
    x1757 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1757 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1756, (2, 0, 3, 4), (2, 3, 4, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1757, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * -1.0
    del x1757
    x1758 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1758 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x778, (4, 5, 6, 2), (0, 4, 5, 6, 1, 3))
    del x778
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1758, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1758
    x1759 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1759 += einsum(x0, (0, 1), t2.abab[np.ix_(soa,sob,sVa,svb)], (2, 3, 4, 1), (4, 2, 3, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1759, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4))
    del x1759
    x1760 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1760 += einsum(x0, (0, 1), t2.abab[np.ix_(sOa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1760, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2))
    del x1760
    x1761 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1761 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x830, (4, 5, 0, 6), (4, 2, 5, 1, 6, 3)) * -1.0
    del x830
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1761, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1761, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x1761
    x1762 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1762 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x817, (4, 5, 6, 2), (0, 4, 5, 1, 6, 3)) * -1.0
    del x817
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1762, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(x1762, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    del x1762
    x1763 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1763 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x814, (4, 5, 6, 2), (0, 4, 5, 6, 1, 3))
    del x814
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1763, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1763
    x1764 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1764 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x809, (4, 5, 6, 2), (0, 4, 5, 6, 1, 3))
    del x809
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1764, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -2.0
    del x1764
    x1765 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1765 += einsum(x11, (0, 1), t2.abab[np.ix_(sOa,sob,sva,svb)], (2, 3, 1, 4), (2, 0, 3, 4))
    del x11
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1765, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1765
    x1766 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1766 += einsum(x12, (0, 1), t2.abab[np.ix_(sOa,sob,sva,svb)], (2, 3, 1, 4), (2, 0, 3, 4))
    del x12
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1766, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1766
    x1767 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1767 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x771, (4, 5, 6, 7, 1, 3), (4, 5, 6, 7, 0, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1767, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -4.0
    del x1767
    x1768 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1768 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x713, (4, 1, 5, 3), (4, 0, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1768, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * -4.0
    del x1768
    x1769 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1769 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x716, (4, 5, 1, 3), (4, 5, 0, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1769, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * -4.0
    del x1769
    x1770 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1770 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x564, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1770, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -1.0
    del x1770
    x1771 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1771 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x565, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1771, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5))
    del x1771
    x1772 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1772 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x564, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x564
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1772, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x1772
    x1773 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1773 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x565, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x565
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1773, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x1773
    x1774 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1774 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 6, 3), (0, 2, 1, 4, 6, 5))
    x1775 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1775 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1774, (4, 5, 6, 7, 1, 3), (4, 5, 0, 6, 7, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1775, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x1775
    x1776 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1776 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1774, (4, 5, 6, 1, 7, 3), (4, 5, 0, 6, 7, 2))
    del x1774
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1776, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x1776
    x1777 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1777 += einsum(x1, (0, 1), t2.abab[np.ix_(sOa,sob,sVa,svb)], (2, 3, 4, 1), (2, 4, 3, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1777, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -1.0
    del x1777
    x1778 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1778 += einsum(x20, (0, 1), t2.abab[np.ix_(sOa,sob,sVa,svb)], (2, 3, 4, 1), (2, 4, 3, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1778, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5))
    del x1778
    x1779 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1779 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x582, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1779, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5)) * -2.0
    del x1779
    x1780 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1780 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x771, (2, 3, 4, 5, 6, 1), (2, 3, 4, 5, 0, 6))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1780, (4, 5, 6, 0, 7, 1), (6, 7, 4, 2, 3, 5)) * 2.0
    x1781 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1781 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x176, (4, 0, 5, 6), (1, 3, 4, 5, 6, 2))
    del x176
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1781, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1781
    x1782 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1782 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x582, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x582
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1782, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1782
    x1783 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1783 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x771, (4, 5, 6, 0, 7, 3), (4, 5, 6, 1, 7, 2))
    del x771
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1783, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1783
    x1784 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1784 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x16, (4, 1, 5, 3), (0, 4, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1784, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2))
    del x1784
    x1785 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1785 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x552, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1785, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * -1.0
    del x1785
    x1786 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1786 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x16, (4, 5, 1, 3), (0, 4, 5, 2))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1786, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * -1.0
    del x1786
    x1787 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1787 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x550, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1787, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4))
    del x1787
    x1788 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1788 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x557, (4, 5, 6, 3), (4, 2, 0, 1, 5, 6))
    del x557
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1788, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x1788
    x1789 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1789 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x559, (4, 5, 6, 3), (4, 2, 0, 1, 5, 6))
    del x559
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1789, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x1789
    x1790 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1790 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x552, (4, 5, 6, 3), (0, 4, 5, 1, 6, 2))
    del x552
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1790, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x1790
    x1791 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1791 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x550, (4, 5, 6, 3), (0, 4, 5, 1, 6, 2))
    del x550
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1791, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x1791
    x1792 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1792 += einsum(x1, (0, 1), t2.abab[np.ix_(soa,sob,sVa,svb)], (2, 3, 4, 1), (4, 2, 3, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1792, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4))
    del x1792
    x1793 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1793 += einsum(x20, (0, 1), t2.abab[np.ix_(soa,sob,sVa,svb)], (2, 3, 4, 1), (4, 2, 3, 0))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1793, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * -1.0
    del x1793
    x1794 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1794 += einsum(x1, (0, 1), t2.abab[np.ix_(sOa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1794, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2))
    del x1794
    x1795 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1795 += einsum(x20, (0, 1), t2.abab[np.ix_(sOa,sob,sva,svb)], (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1795, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * -1.0
    del x1795
    x1796 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1796 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x13, (0, 4, 5, 3), (1, 4, 5, 2)) * -1.0
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1796, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * 2.0
    del x1796
    x1797 = np.zeros((navir[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    x1797 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x13, (0, 4, 1, 5), (2, 4, 5, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1797, (4, 5, 3, 6), (0, 5, 1, 2, 6, 4)) * 2.0
    del x1797
    x1798 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1798 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x577, (4, 5, 6, 3), (4, 2, 0, 1, 5, 6))
    del x577
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1798, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1798
    x1799 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1799 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1751, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    del x1751
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1799, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1799
    x1800 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1800 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x573, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1800, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * 2.0
    del x1800
    x1801 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1801 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x13, (4, 5, 1, 2), (0, 4, 5, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1801, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1801
    x1802 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1802 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1756, (2, 3, 4, 0), (2, 3, 4, 1))
    del x1756
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1802, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1802
    x1803 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1803 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x573, (4, 5, 6, 3), (0, 4, 5, 1, 6, 2))
    del x573
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1803, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1803
    x1804 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1804 += einsum(x15, (0, 1), t2.abab[np.ix_(sOa,sob,sva,svb)], (2, 3, 1, 4), (2, 0, 3, 4))
    del x15
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1804, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * -2.0
    del x1804
    x1805 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1805 += einsum(t2.aaaa[np.ix_(soa,soa,sva,sva)], (0, 1, 2, 3), x1513, (4, 5, 1, 6, 7, 3), (4, 5, 0, 6, 7, 2))
    del x1513
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1805, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1805
    x1806 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1806 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x904, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x904
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1806, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1806
    x1807 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1807 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x907, (4, 5, 6, 2), (4, 5, 0, 6, 1, 3))
    del x907
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1807, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1807
    x1808 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1808 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x888, (4, 5, 6, 2), (0, 4, 5, 6, 1, 3))
    del x888
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1808, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1808
    x1809 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1809 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x893, (4, 5, 6, 2), (0, 4, 5, 6, 1, 3))
    del x893
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1809, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1809
    x1810 = np.zeros((naocc[0], navir[0], nocc[1], nocc[1]), dtype=np.float64)
    x1810 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x719, (2, 3, 4, 1), (2, 3, 0, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1810, (4, 5, 6, 1), (0, 6, 4, 2, 3, 5))
    del x1810
    x1811 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1811 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x719, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x719
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1811, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3))
    del x1811
    x1812 = np.zeros((naocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1812 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x716, (2, 3, 4, 1), (2, 3, 0, 4))
    x1813 = np.zeros((naocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1813 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1812, (2, 0, 3, 4), (2, 3, 4, 1))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x1813, (4, 5, 1, 6), (0, 5, 4, 6, 3, 2)) * -1.0
    del x1813
    x1814 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1814 += einsum(t2.abab[np.ix_(soa,sob,sVa,svb)], (0, 1, 2, 3), x716, (4, 5, 6, 3), (4, 2, 0, 5, 1, 6))
    del x716
    x1815 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1815 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1814, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1814
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1815, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * -1.0
    del x1815
    x1816 = np.zeros((navir[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1816 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x713, (2, 3, 4, 1), (2, 0, 3, 4))
    del x713
    x1817 = np.zeros((navir[0], nocc[0], nocc[1], nocc[1]), dtype=np.float64)
    x1817 += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1816, (2, 3, 4, 1), (2, 0, 3, 4))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x1817, (4, 5, 6, 1), (5, 6, 0, 2, 3, 4)) * -1.0
    del x1817
    x1818 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1818 += einsum(t2.abab[np.ix_(sOa,sob,sva,svb)], (0, 1, 2, 3), x714, (4, 5, 6, 3), (0, 4, 5, 1, 6, 2))
    del x714
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1818, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -1.0
    del x1818
    x1819 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1819 += einsum(t2.abab[np.ix_(sOa,sob,sVa,svb)], (0, 1, 2, 3), x197, (4, 5, 6, 1), (0, 2, 4, 5, 6, 3))
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1819, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1819
    x1820 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1820 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1736, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1736
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1820, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3))
    del x1820
    x1821 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1821 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sVa)], (0, 1, 2, 3), x197, (4, 0, 5, 6), (1, 3, 4, 5, 6, 2))
    del x197
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1821, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * 2.0
    del x1821
    x1822 = np.zeros((naocc[0], navir[0], nocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1822 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1780, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1780
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.aa[np.ix_(soa,sva)], (0, 1), x1822, (2, 3, 4, 0, 5, 6), (4, 5, 2, 1, 6, 3)) * 2.0
    del x1822
    x1823 = np.zeros((naocc[0], navir[0], nocc[0], nocc[1], nocc[1], nvir[0]), dtype=np.float64)
    x1823 += einsum(t2.aaaa[np.ix_(soa,sOa,sva,sva)], (0, 1, 2, 3), x1816, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    del x1816
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1823, (2, 3, 4, 5, 0, 6), (4, 5, 2, 6, 1, 3)) * -2.0
    del x1823
    x1824 = np.zeros((naocc[0], nocc[0], nocc[1], nvir[1]), dtype=np.float64)
    x1824 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1812, (2, 3, 4, 0), (2, 3, 4, 1))
    del x1812
    t3new_abaaba[np.ix_(soa,sob,sOfa,sva,svb,sVfa)] += einsum(t2.aaaa[np.ix_(soa,soa,sva,sVa)], (0, 1, 2, 3), x1824, (4, 1, 5, 6), (0, 5, 4, 2, 6, 3)) * 2.0
    del x1824
    x1825 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1825 += einsum(f.bb.oo, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1825, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1825, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x1825
    x1826 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1826 += einsum(f.bb.vv, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1826, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1826, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x1826
    x1827 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1827 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.oovO, (4, 1, 5, 6), (6, 3, 0, 4, 2, 5)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1827, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1827, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1827, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1827, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1827
    x1828 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1828 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ooOV, (4, 1, 5, 6), (5, 6, 0, 4, 2, 3))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1828, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1828, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1828
    x1829 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1829 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoO, (4, 5, 1, 6), (6, 3, 0, 4, 2, 5)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1829, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1829, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1829, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1829, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x1829
    x1830 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1830 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.oOoV, (1, 4, 5, 6), (4, 6, 0, 5, 2, 3))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1830, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1830, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1830
    x1831 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1831 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.oooV, (4, 0, 5, 6), (1, 6, 5, 4, 2, 3)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1831, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1831, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x1831
    x1832 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1832 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 0, 5, 6), (1, 3, 4, 5, 2, 6))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1832, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1832, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1832, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1832, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1832
    x1833 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1833 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (4, 5, 3, 6), (1, 6, 0, 4, 2, 5)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1833, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1833, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1833, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1833, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x1833
    x1834 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1834 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.oVvv, (4, 5, 6, 3), (1, 5, 0, 4, 2, 6)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1834, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1834, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1834, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1834, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1834
    x1835 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1835 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.vvvO, (4, 2, 5, 6), (6, 3, 0, 1, 5, 4)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1835, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1835, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x1835
    x1836 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1836 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.vvOV, (4, 3, 5, 6), (5, 6, 0, 1, 2, 4))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1836, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1836, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x1836
    x1837 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1837 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.vOvV, (4, 5, 3, 6), (5, 6, 0, 1, 2, 4))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1837, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1837, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x1837
    x1838 = np.zeros((naocc[1], navir[1], navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1838 += einsum(v.aabb.ovoV, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 5, 6, 1, 7), (5, 7, 3, 4, 2, 6))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x1838, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 1, 2)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x1838, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 1, 2)) * -2.0
    del x1838
    x1839 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1839 += einsum(v.aabb.ovvO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 5, 6, 1, 7), (5, 3, 7, 4, 6, 2))
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x1839, (0, 1, 2, 3, 4, 5), (3, 0, 1, 5, 4, 2)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x1839, (0, 1, 2, 3, 4, 5), (3, 0, 1, 4, 5, 2)) * -2.0
    del x1839
    x1840 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1840 += einsum(v.bbbb.ooOO, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 1, 3, 5, 6, 7), (2, 7, 4, 0, 5, 6))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1840, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1840, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x1840
    x1841 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1841 += einsum(v.bbbb.oovv, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 1, 5, 6, 3, 7), (5, 7, 4, 0, 6, 2))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1841, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1841, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1841, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1841, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    del x1841
    x1842 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1842 += einsum(v.bbbb.ooVV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 1, 5, 6, 7, 3), (5, 2, 4, 0, 6, 7))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1842, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1842, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x1842
    x1843 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1843 += einsum(v.bbbb.oVoV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 2, 5, 6, 7, 3), (5, 1, 4, 0, 6, 7))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1843, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1843, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x1843
    x1844 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1844 += einsum(v.bbbb.vOvO, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 3, 6, 2, 7), (1, 7, 4, 5, 6, 0))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1844, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1844, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    del x1844
    x1845 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1845 += einsum(v.bbbb.vvOO, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 3, 6, 1, 7), (2, 7, 4, 5, 6, 0))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1845, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1845, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    del x1845
    x1846 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1846 += einsum(v.bbbb.vvVV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 6, 7, 1, 3), (6, 2, 4, 5, 7, 0))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1846, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1846, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    del x1846
    x1847 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1847 += einsum(x4, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 0, 3, 4, 5, 6), (3, 6, 1, 2, 4, 5))
    del x4
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1847, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1847, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x1847
    x1848 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1848 += einsum(f.bb.ov, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 0, 2, 3, 5))
    x1849 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1849 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1848, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 1, 6))
    del x1848
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1849, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1849, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x1849
    x1850 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1850 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x926, (4, 5, 1, 6), (4, 5, 6, 0, 2, 3))
    del x926
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1850, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1850, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x1850
    x1851 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1851 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x220, (0, 4, 5, 6), (1, 3, 4, 5, 2, 6))
    del x220
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1851, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1851, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x1851
    x1852 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1852 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x927, (4, 1, 5, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x927
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1852, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1852, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1852, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1852, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x1852
    x1853 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1853 += einsum(f.bb.ov, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,sVb)], (2, 3, 1, 4), (4, 0, 2, 3)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1853, (4, 0, 5, 6), (6, 5, 1, 2, 3, 4)) * 4.0
    del x1853
    x1854 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1854 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x942, (4, 1, 5, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x942
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1854, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1854, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1854, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1854, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1854
    x1855 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1855 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x943, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x943
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1855, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1855, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1855, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1855, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1855
    x1856 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1856 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x945, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    del x945
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1856, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1856, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1856, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1856, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1856
    x1857 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1857 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x171, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x171
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1857, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1857, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1857
    x1858 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1858 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovvO, (4, 2, 5, 6), (6, 3, 0, 1, 4, 5)) * -1.0
    x1859 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1859 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1858, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x1858
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1859, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1859, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x1859
    x1860 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1860 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovOV, (4, 3, 5, 6), (5, 6, 0, 1, 4, 2))
    x1861 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1861 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1860, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x1860
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1861, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1861, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x1861
    x1862 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1862 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x947, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x947
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1862, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1862, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1862
    x1863 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1863 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x948, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    del x948
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1863, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1863, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1863, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1863, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1863
    x1864 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1864 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.oOvV, (4, 5, 3, 6), (5, 6, 0, 1, 4, 2))
    x1865 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1865 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1864, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x1864
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1865, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1865, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x1865
    x1866 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1866 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.oOvv, (4, 5, 6, 2), (5, 3, 0, 1, 4, 6)) * -1.0
    x1867 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1867 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1866, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x1866
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1867, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1867, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x1867
    x1868 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1868 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.oovV, (2, 3, 1, 4), (4, 0, 2, 3))
    x1869 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1869 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1868, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x1868
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1869, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1869, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1869
    x1870 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1870 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x222, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x222
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1870, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1870, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1870, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1870, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1870
    x1871 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1871 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.oooo, (4, 5, 6, 0), (1, 3, 4, 5, 6, 2))
    x1872 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1872 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1871, (2, 3, 0, 4, 5, 6), (2, 3, 4, 5, 1, 6))
    del x1871
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1872, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1872, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1872, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1872, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1872
    x1873 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1873 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.oovV, (4, 5, 3, 6), (1, 6, 0, 4, 5, 2)) * -1.0
    x1874 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1874 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1873, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x1873
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1874, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1874, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1874, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1874, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1874
    x1875 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1875 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.oovv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x1876 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1876 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1875, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x1875
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1876, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1876, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1876, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1876, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1876
    x1877 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1877 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), v.bbbb.ovoV, (2, 1, 3, 4), (4, 0, 3, 2))
    x1878 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1878 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1877, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x1877
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1878, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1878, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1878
    x1879 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1879 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x16, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1879, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1879, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1879, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1879, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x1879
    x1880 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1880 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovoV, (4, 3, 5, 6), (1, 6, 0, 5, 4, 2)) * -1.0
    x1881 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1881 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1880, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x1880
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1881, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1881, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1881, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1881, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1881
    x1882 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1882 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1413, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1882, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1882, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1882, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1882, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1882
    x1883 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1883 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x954, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3))
    del x954
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1883, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1883, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1883
    x1884 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1884 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x955, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x955
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1884, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1884, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1884
    x1885 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1885 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x957, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x957
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1885, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1885, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x1885
    x1886 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1886 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x958, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x958
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1886, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1886, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x1886
    x1887 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1887 += einsum(t1.bb[np.ix_(sob,sVb)], (0, 1), v.bbbb.oooo, (2, 3, 4, 0), (1, 2, 3, 4))
    x1888 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1888 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1887, (4, 5, 0, 6), (1, 4, 6, 5, 2, 3)) * -1.0
    del x1887
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1888, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1888, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1888
    x1889 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1889 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x960, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x960
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1889, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1889, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1889, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1889, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1889
    x1890 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1890 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x961, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1890, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1890, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1890, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1890, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x1890
    x1891 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1891 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x963, (4, 5, 3, 6), (1, 4, 5, 0, 2, 6)) * -1.0
    del x963
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1891, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1891, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1891, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1891, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x1891
    x1892 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1892 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x964, (4, 5, 6, 3), (1, 4, 5, 0, 2, 6)) * -1.0
    del x964
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1892, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1892, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1892, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1892, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x1892
    x1893 = np.zeros((naocc[1], navir[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1893 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.vvvv, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x1894 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1894 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1893, (2, 3, 4, 1, 5, 6), (2, 3, 0, 4, 5, 6))
    del x1893
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1894, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1894, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1894, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1894, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1894
    x1895 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1895 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x970, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x970
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1895, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1895, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1895
    x1896 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1896 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x972, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x972
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1896, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1896, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1896, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1896, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1896
    x1897 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1897 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x974, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x974
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1897, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1897, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1897
    x1898 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1898 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x975, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1898, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1898, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1898, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1898, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x1898
    x1899 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1899 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x976, (4, 5, 3, 6), (4, 5, 0, 1, 2, 6))
    del x976
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1899, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1899, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x1899
    x1900 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1900 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x977, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x977
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1900, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1900, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x1900
    x1901 = np.zeros((naocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1901 += einsum(t1.bb[np.ix_(sOb,svb)], (0, 1), v.bbbb.vvvv, (2, 3, 4, 1), (0, 2, 3, 4))
    x1902 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1902 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1901, (4, 5, 2, 6), (4, 3, 0, 1, 6, 5)) * -1.0
    del x1901
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1902, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1902, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x1902
    x1903 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1903 += einsum(x6, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 1, 3, 4, 5, 6), (3, 6, 2, 0, 4, 5))
    del x6
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1903, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1903, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x1903
    x1904 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1904 += einsum(x5, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 5, 0))
    del x5
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1904, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1904, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    del x1904
    x1905 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1905 += einsum(x140, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 5, 6, 2, 7), (5, 7, 1, 4, 6, 3))
    del x140
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1905, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1905, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1905, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1905, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x1905
    x1906 = np.zeros((naocc[1], navir[1], navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1906 += einsum(x1201, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 3, 7), (5, 7, 0, 2, 4, 6))
    del x1201
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x1906, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 1, 2)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x1906, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 1, 2)) * -2.0
    del x1906
    x1907 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1907 += einsum(v.aabb.ovoO, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 5, 6, 1, 7), (5, 3, 7, 4, 2, 6))
    x1908 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1908 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1907, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x1907
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x1908, (0, 1, 2, 3, 4, 5), (3, 0, 1, 4, 5, 2)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x1908, (0, 1, 2, 3, 4, 5), (3, 0, 1, 5, 4, 2)) * 2.0
    del x1908
    x1909 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1909 += einsum(v.aabb.ovoo, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 5, 6, 1, 7), (5, 7, 4, 2, 3, 6))
    x1910 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1910 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1909, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x1909
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1910, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1910, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1910, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1910, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x1910
    x1911 = np.zeros((naocc[1], navir[1], navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1911 += einsum(x1210, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 3, 7), (5, 0, 7, 4, 2, 6))
    del x1210
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x1911, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 2, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x1911, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 2, 1)) * 2.0
    del x1911
    x1912 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1912 += einsum(x1215, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 2, 7), (0, 5, 7, 4, 6, 3))
    del x1215
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x1912, (0, 1, 2, 3, 4, 5), (3, 1, 0, 5, 4, 2)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x1912, (0, 1, 2, 3, 4, 5), (3, 1, 0, 4, 5, 2)) * -2.0
    del x1912
    x1913 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1913 += einsum(x234, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 7), (4, 7, 0, 1, 5, 6))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1913, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1913, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x1913
    x1914 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1914 += einsum(v.bbbb.ooov, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 2, 5, 6, 3, 7), (5, 7, 4, 0, 1, 6))
    x1915 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1915 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1914, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x1914
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1915, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1915, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1915, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1915, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x1915
    x1916 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1916 += einsum(v.bbbb.ooov, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 1, 5, 6, 3, 7), (5, 7, 4, 0, 2, 6))
    x1917 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1917 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1916, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x1916
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1917, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1917, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1917, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1917, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x1917
    x1918 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1918 += einsum(x7, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 1, 3, 4, 5, 6), (3, 6, 2, 0, 4, 5))
    del x7
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1918, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1918, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x1918
    x1919 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1919 += einsum(x8, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 1, 3, 4, 5, 6), (3, 6, 2, 0, 4, 5))
    del x8
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1919, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1919, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x1919
    x1920 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1920 += einsum(x1005, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 1, 5, 6, 7), (0, 7, 2, 4, 5, 6))
    del x1005
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1920, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1920, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x1920
    x1921 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1921 += einsum(x160, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 1, 5, 6, 3, 7), (5, 7, 0, 4, 6, 2))
    del x160
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1921, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1921, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1921, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1921, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    del x1921
    x1922 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1922 += einsum(x161, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 1, 5, 6, 2, 7), (5, 7, 0, 4, 6, 3))
    del x161
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1922, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1922, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1922, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1922, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 6.0
    del x1922
    x1923 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1923 += einsum(x1006, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x1006
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1923, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1923, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x1923
    x1924 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1924 += einsum(x1007, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 5, 6, 7, 0), (5, 1, 2, 4, 6, 7))
    del x1007
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1924, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1924, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x1924
    x1925 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1925 += einsum(v.bbbb.ovOO, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 3, 6, 1, 7), (2, 7, 4, 5, 0, 6))
    x1926 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1926 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1925, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x1925
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1926, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1926, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x1926
    x1927 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1927 += einsum(v.bbbb.oOvO, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 3, 6, 2, 7), (1, 7, 4, 5, 0, 6))
    x1928 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1928 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1927, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x1927
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1928, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1928, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x1928
    x1929 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1929 += einsum(v.bbbb.ovvv, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 6, 3, 1, 7), (6, 7, 4, 5, 0, 2)) * -1.0
    x1930 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1930 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1929, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x1929
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1930, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1930, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x1930
    x1931 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1931 += einsum(v.bbbb.ovVV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 6, 7, 1, 3), (6, 2, 4, 5, 0, 7))
    x1932 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1932 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1931, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x1931
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1932, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1932, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x1932
    x1933 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1933 += einsum(x10, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 5, 0))
    del x10
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1933, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1933, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    del x1933
    x1934 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1934 += einsum(x9, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 0, 6), (4, 6, 2, 3, 5, 1))
    del x9
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1934, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1934, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    del x1934
    x1935 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1935 += einsum(x1015, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 2, 5, 6, 7, 1), (5, 0, 4, 3, 6, 7))
    del x1015
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1935, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1935, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x1935
    x1936 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1936 += einsum(x1016, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 5, 6, 7, 1), (5, 0, 4, 2, 6, 7))
    del x1016
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1936, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1936, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x1936
    x1937 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1937 += einsum(x1020, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 6, 7, 3, 1), (6, 0, 4, 5, 7, 2))
    del x1020
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1937, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1937, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    del x1937
    x1938 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1938 += einsum(x1023, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 1, 5, 6, 7), (0, 7, 4, 2, 5, 6))
    del x1023
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1938, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1938, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x1938
    x1939 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1939 += einsum(x1025, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 6, 2))
    del x1025
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1939, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1939, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    del x1939
    x1940 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1940 += einsum(x1026, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 2, 7), (0, 7, 4, 5, 6, 3))
    del x1026
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1940, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1940, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    del x1940
    x1941 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1941 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1034, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x1034
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1941, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1941, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1941, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1941, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1941
    x1942 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1942 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1031, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1031
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1942, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1942, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x1942
    x1943 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1943 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x232, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x232
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1943, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1943, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1943, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1943, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1943
    x1944 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1944 += einsum(t2.abab[np.ix_(soa,sob,sva,sVb)], (0, 1, 2, 3), v.aabb.ovoo, (0, 2, 4, 5), (3, 1, 4, 5)) * -1.0
    x1945 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1945 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1944, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x1944
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1945, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1945, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1945
    x1946 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1946 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1051, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1051
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1946, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1946, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1946, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1946, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1946
    x1947 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1947 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x915, (4, 5, 0, 6, 2, 7), (4, 5, 6, 1, 3, 7))
    del x915
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1947, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1947, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1947, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1947, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x1947
    x1948 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1948 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1063, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1063
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1948, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1948, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1948, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1948, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1948
    x1949 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1949 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1044, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x1044
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1949, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1949, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1949, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1949, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x1949
    x1950 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1950 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1042, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x1042
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1950, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1950, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1950
    x1951 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1951 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1049, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1049
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1951, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1951, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x1951
    x1952 = np.zeros((naocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1952 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovvv, (0, 2, 4, 5), (1, 3, 4, 5)) * -1.0
    x1953 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1953 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1952, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x1952
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1953, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1953, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x1953
    x1954 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1954 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1054, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x1054
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1954, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1954, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x1954
    x1955 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1955 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1073, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x1073
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1955, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1955, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1955, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1955, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x1955
    x1956 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1956 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1067, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1067
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1956, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1956, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x1956
    x1957 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1957 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1072, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x1072
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1957, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1957, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1957, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1957, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x1957
    x1958 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1958 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1069, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1069
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1958, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1958, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x1958
    x1959 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1959 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1070, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1070
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1959, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1959, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1959, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1959, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x1959
    x1960 = np.zeros((naocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1960 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovoO, (1, 4, 0, 5), (5, 2, 3, 4))
    x1961 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1961 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1960, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x1960
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1961, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1961, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x1961
    x1962 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1962 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x238, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x238
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1962, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1962, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1962, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1962, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    del x1962
    x1963 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1963 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1074, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x1074
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1963, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1963, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    del x1963
    x1964 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1964 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x236, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x236
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1964, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1964, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1964, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1964, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    del x1964
    x1965 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1965 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1075, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x1075
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1965, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1965, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x1965
    x1966 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1966 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 6, 2), (1, 3, 0, 4, 5, 6))
    x1967 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1967 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1966, (4, 5, 6, 0, 7, 1), (4, 5, 6, 7, 2, 3))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1967, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1967, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1967, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1967, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1967
    x1968 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1968 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1078, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x1078
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1968, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1968, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1968, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1968, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    del x1968
    x1969 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1969 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 1, 5, 2), (3, 0, 4, 5)) * -1.0
    x1970 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1970 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1969, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x1969
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1970, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1970, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    del x1970
    x1971 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1971 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1079, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x1079
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1971, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1971, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1971, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1971, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    del x1971
    x1972 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1972 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ooov, (4, 5, 1, 2), (3, 0, 4, 5)) * -1.0
    x1973 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1973 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1972, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x1972
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1973, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1973, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x1973
    x1974 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1974 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1094, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1094
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1974, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1974, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1974, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1974, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1974, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1974, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1974, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1974, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x1974
    x1975 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1975 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1099, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1099
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1975, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1975, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1975, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1975, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x1975
    x1976 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1976 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1083, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1083
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1976, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1976, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1976, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1976, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x1976
    x1977 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1977 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1098, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1098
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1977, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1977, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1977, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1977, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x1977
    x1978 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1978 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1086, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1086
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1978, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1978, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x1978
    x1979 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x1979 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvV, (4, 3, 2, 5), (5, 0, 1, 4))
    x1980 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1980 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1979, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x1979
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1980, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1980, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x1980
    x1981 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1981 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1087, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1087
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1981, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1981, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x1981
    x1982 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1982 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1088, (4, 5, 6, 1, 3, 7), (4, 5, 6, 0, 2, 7))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1982, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1982, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1982, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1982, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x1982
    x1983 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1983 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1089, (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 2, 7))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1983, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1983, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1983, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1983, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x1983
    x1984 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1984 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1090, (4, 5, 3, 6), (4, 5, 0, 1, 2, 6))
    del x1090
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1984, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1984, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x1984
    x1985 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1985 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x244, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x244
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1985, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1985, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1985, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1985, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x1985
    x1986 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1986 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1091, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x1091
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1986, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1986, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x1986
    x1987 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1987 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1092, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    del x1092
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1987, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1987, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1987, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1987, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1987, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1987, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1987, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1987, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x1987
    x1988 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1988 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1101, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1101
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1988, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1988, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1988, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1988, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x1988
    x1989 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1989 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1100, (4, 5, 3, 6), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1100
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1989, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1989, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1989, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1989, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x1989
    x1990 = np.zeros((naocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1990 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (0, 4, 5, 3), (1, 2, 4, 5)) * -1.0
    x1991 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1991 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1990, (4, 5, 2, 6), (4, 3, 0, 1, 5, 6)) * -1.0
    del x1990
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1991, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1991, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x1991
    x1992 = np.zeros((naocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1992 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovvv, (0, 3, 4, 5), (1, 2, 4, 5)) * -1.0
    x1993 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1993 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1992, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x1992
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1993, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1993, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x1993
    x1994 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1994 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x917, (4, 5, 0, 6, 2, 7), (4, 5, 1, 6, 3, 7))
    del x917
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1994, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1994, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1994, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1994, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x1994
    x1995 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1995 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x1171, (4, 5, 0, 6, 2, 7), (4, 5, 1, 6, 3, 7))
    del x1171
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1995, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1995, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1995, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x1995, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x1995
    x1996 = np.zeros((naocc[1], navir[1], navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1996 += einsum(x1337, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 3, 7), (5, 0, 7, 2, 4, 6))
    del x1337
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x1996, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 2, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x1996, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 2, 1)) * 2.0
    del x1996
    x1997 = np.zeros((naocc[1], navir[1], navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x1997 += einsum(x1334, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 3, 7), (5, 0, 7, 2, 4, 6))
    del x1334
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x1997, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 2, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x1997, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 2, 1)) * -2.0
    del x1997
    x1998 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1998 += einsum(x1344, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 2, 7), (0, 5, 7, 4, 3, 6))
    del x1344
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x1998, (0, 1, 2, 3, 4, 5), (3, 1, 0, 4, 5, 2)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x1998, (0, 1, 2, 3, 4, 5), (3, 1, 0, 5, 4, 2)) * 2.0
    del x1998
    x1999 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1999 += einsum(x1346, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 2, 7), (0, 5, 7, 4, 3, 6))
    del x1346
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x1999, (0, 1, 2, 3, 4, 5), (3, 1, 0, 4, 5, 2)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x1999, (0, 1, 2, 3, 4, 5), (3, 1, 0, 5, 4, 2)) * -2.0
    del x1999
    x2000 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2000 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x918, (4, 5, 0, 6, 2, 7), (4, 5, 1, 6, 3, 7))
    del x918
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2000, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2000, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2000, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2000, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x2000
    x2001 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2001 += einsum(x14, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5))
    del x14
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2001, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2001, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x2001
    x2002 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2002 += einsum(x177, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    del x177
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2002, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2002, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x2002
    x2003 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2003 += einsum(x1141, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x1141
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2003, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2003, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    del x2003
    x2004 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2004 += einsum(x1148, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    del x1148
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2004, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2004, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x2004
    x2005 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2005 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1130, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x1130
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2005, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2005, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x2005
    x2006 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2006 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1131, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x1131
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2006, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2006, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    del x2006
    x2007 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2007 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1133, (4, 5, 6, 1, 7, 3), (4, 5, 0, 6, 2, 7))
    del x1133
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2007, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2007, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2007, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2007, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x2007
    x2008 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2008 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x250, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2008, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2008, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2008, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2008, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x2008
    x2009 = np.zeros((naocc[1], navir[1], navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2009 += einsum(x1371, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 3, 7), (5, 0, 7, 2, 4, 6))
    del x1371
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x2009, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 2, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x2009, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 2, 1)) * -4.0
    del x2009
    x2010 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2010 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1146, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1146
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2010, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2010, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2010, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2010, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x2010
    x2011 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2011 += einsum(x1366, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 2, 7), (0, 5, 7, 4, 3, 6))
    del x1366
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x2011, (0, 1, 2, 3, 4, 5), (3, 1, 0, 4, 5, 2)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x2011, (0, 1, 2, 3, 4, 5), (3, 1, 0, 5, 4, 2)) * -4.0
    del x2011
    x2012 = np.zeros((naocc[1], naocc[1], navir[1], nocc[0], nocc[1], nvir[0]), dtype=np.float64)
    x2012 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), v.aabb.ovvO, (4, 5, 2, 6), (1, 6, 3, 4, 0, 5))
    x2013 = np.zeros((naocc[1], navir[1], navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2013 += einsum(t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (0, 1, 2, 3, 4, 5), x2012, (6, 2, 7, 1, 8, 4), (6, 7, 5, 8, 0, 3))
    del x2012
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x2013, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 2, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x2013, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 2, 1)) * -4.0
    del x2013
    x2014 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2014 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x250, (4, 5, 0, 6), (1, 4, 3, 5, 2, 6))
    del x250
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x2014, (0, 1, 2, 3, 4, 5), (3, 1, 0, 4, 5, 2)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x2014, (0, 1, 2, 3, 4, 5), (3, 1, 0, 5, 4, 2)) * -4.0
    del x2014
    x2015 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2015 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1153, (4, 5, 6, 3), (4, 5, 0, 1, 2, 6))
    del x1153
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2015, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2015, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2015, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2015, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x2015
    x2016 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2016 += einsum(x265, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 7), (4, 7, 1, 0, 5, 6)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2016, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -3.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2016, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -3.0
    del x2016
    x2017 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2017 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1155, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x1155
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2017, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2017, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -12.0
    del x2017
    x2018 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2018 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1156, (4, 5, 6, 1, 7, 3), (4, 5, 0, 6, 2, 7))
    del x1156
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2018, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2018, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2018, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2018, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 12.0
    del x2018
    x2019 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2019 += einsum(x18, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5)) * -1.0
    del x18
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2019, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2019, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2019, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2019, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x2019
    x2020 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2020 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1157, (4, 5, 6, 1, 7, 3), (4, 5, 0, 6, 2, 7))
    del x1157
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2020, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2020, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2020, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2020, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -12.0
    del x2020
    x2021 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2021 += einsum(v.bbbb.ovov, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 6, 3, 1, 7), (6, 7, 4, 5, 0, 2)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x2021, (4, 5, 6, 7, 1, 0), (6, 7, 4, 2, 3, 5)) * -6.0
    x2022 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2022 += einsum(x183, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5)) * -1.0
    del x183
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2022, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2022, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2022, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2022, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x2022
    x2023 = np.zeros((naocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x2023 += einsum(v.bbbb.ovoV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 0, 4, 5, 6, 3), (4, 5, 6, 1))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x2023, (4, 5, 6, 2), (0, 1, 4, 5, 6, 3)) * 6.0
    del x2023
    x2024 = np.zeros((naocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x2024 += einsum(v.bbbb.ovoV, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 0, 4, 5, 6, 3), (4, 5, 6, 1))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x2024, (4, 5, 6, 2), (0, 1, 4, 5, 6, 3)) * 6.0
    del x2024
    x2025 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2025 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1158, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x1158
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2025, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2025, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2025, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2025, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -12.0
    del x2025
    x2026 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2026 += einsum(x1175, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x1175
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2026, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2026, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -12.0
    del x2026
    x2027 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2027 += einsum(x1176, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x1176
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2027, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2027, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 12.0
    del x2027
    x2028 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2028 += einsum(x1177, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 6, 7, 3, 1), (6, 0, 4, 5, 2, 7))
    del x1177
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2028, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2028, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x2028
    x2029 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2029 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1159, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1159
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2029, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2029, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2029, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2029, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x2029
    x2030 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2030 += einsum(x1181, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 1, 5, 6, 7), (0, 7, 2, 4, 5, 6))
    del x1181
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2030, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 3.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2030, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 3.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2030, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -3.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2030, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -3.0
    del x2030
    x2031 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2031 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1160, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1160
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2031, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2031, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2031, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2031, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x2031
    x2032 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2032 += einsum(v.bbbb.ovvO, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 3, 2, 1, 6), (6, 4, 5, 0))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x2032, (4, 5, 6, 0), (6, 5, 1, 2, 3, 4)) * -12.0
    del x2032
    x2033 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2033 += einsum(x1184, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    del x1184
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2033, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2033, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 12.0
    del x2033
    x2034 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2034 += einsum(x1185, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 1, 4, 5, 6), (0, 6, 2, 3, 4, 5))
    del x1185
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2034, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2034, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x2034
    x2035 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2035 += einsum(x1187, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    del x1187
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2035, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2035, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -12.0
    del x2035
    x2036 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2036 += einsum(t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (0, 1, 2, 3, 4, 5), x1188, (6, 2, 7, 5, 8, 1), (6, 7, 8, 0, 3, 4))
    del x1188
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2036, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2036, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x2036
    x2037 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2037 += einsum(t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (0, 1, 2, 3, 4, 5), x1189, (6, 2, 7, 5, 8, 1), (6, 7, 8, 0, 3, 4))
    del x1189
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2037, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2037, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x2037
    x2038 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2038 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x254, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x254
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2038, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 12.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2038, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -12.0
    del x2038
    x2039 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2039 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1231, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x1231
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2039, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2039, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2039, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2039, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2039
    x2040 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2040 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1232, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x1232
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2040, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2040, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2040, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2040, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2040
    x2041 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2041 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), v.bbbb.ovoO, (4, 2, 5, 6), (6, 3, 0, 1, 5, 4)) * -1.0
    x2042 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2042 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2041, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x2041
    x2043 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2043 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2042, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2042
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2043, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2043, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x2043
    x2044 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2044 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x234, (4, 5, 6, 0), (1, 3, 4, 6, 5, 2))
    x2045 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2045 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2044, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x2044
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2045, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2045, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2045, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2045, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2045
    x2046 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2046 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x234, (4, 5, 0, 6), (1, 3, 4, 5, 6, 2))
    del x234
    x2047 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2047 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2046, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2046
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2047, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2047, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2047, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2047, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2047
    x2048 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2048 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1966, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x1966
    x2049 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2049 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2048, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x2048
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2049, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2049, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2049, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2049, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2049
    x2050 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2050 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1234, (2, 3, 4, 1), (2, 0, 3, 4))
    x2051 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2051 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x2050, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x2050
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2051, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2051, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x2051
    x2052 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2052 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x269, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x269
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2052, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2052, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2052, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2052, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2052
    x2053 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2053 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1234, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2)) * -1.0
    del x1234
    x2054 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2054 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2053, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2053
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2054, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2054, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2054, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2054, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2054
    x2055 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2055 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1236, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2)) * -1.0
    del x1236
    x2056 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2056 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2055, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2055
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2056, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2056, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2056, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2056, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2056
    x2057 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2057 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1088, (2, 3, 4, 5, 1, 6), (2, 3, 0, 4, 5, 6))
    del x1088
    x2058 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2058 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2057, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2057
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2058, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2058, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2058, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2058, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2058
    x2059 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2059 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1089, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    del x1089
    x2060 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2060 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2059, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2059
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2060, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2060, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2060, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2060, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2060
    x2061 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2061 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1239, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1239
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2061, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2061, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2061
    x2062 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2062 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1241, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1241
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2062, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2062, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2062
    x2063 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2063 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1240, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1240
    x2064 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2064 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2063, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2063
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2064, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2064, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x2064
    x2065 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2065 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1238, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1238
    x2066 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2066 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2065, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2065
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2066, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2066, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x2066
    x2067 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2067 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1247, (2, 3, 4, 1), (2, 0, 3, 4))
    x2068 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2068 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x2067, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x2067
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2068, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2068, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2068
    x2069 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2069 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1245, (2, 3, 4, 1), (2, 0, 4, 3))
    x2070 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2070 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x2069, (4, 5, 0, 6), (1, 4, 5, 6, 2, 3)) * -1.0
    del x2069
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2070, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2070, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2070
    x2071 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2071 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1245, (4, 5, 6, 3), (1, 4, 0, 6, 5, 2)) * -1.0
    del x1245
    x2072 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2072 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2071, (2, 3, 4, 0, 5, 6), (2, 3, 4, 5, 1, 6))
    del x2071
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2072, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2072, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2072, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2072, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2072
    x2073 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2073 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1247, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    del x1247
    x2074 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2074 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2073, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2073
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2074, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2074, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2074, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2074, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2074
    x2075 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2075 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1250, (4, 5, 3, 6), (1, 4, 5, 0, 2, 6)) * -1.0
    del x1250
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2075, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2075, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2075, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2075, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2075
    x2076 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2076 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1251, (4, 5, 3, 6), (1, 4, 5, 0, 2, 6)) * -1.0
    del x1251
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2076, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2076, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2076, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2076, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2076
    x2077 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2077 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1255, (4, 5, 1, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x1255
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2077, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2077, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2077, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2077, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2077
    x2078 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2078 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1256, (4, 1, 5, 6), (4, 3, 0, 5, 6, 2)) * -1.0
    del x1256
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2078, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2078, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2078, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2078, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2078
    x2079 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2079 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1259, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1259
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2079, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2079, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2079
    x2080 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2080 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1261, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1261
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2080, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2080, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2080
    x2081 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2081 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1263, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    del x1263
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2081, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2081, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2081, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2081, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2081
    x2082 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2082 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1265, (4, 5, 1, 6), (4, 3, 5, 0, 2, 6)) * -1.0
    del x1265
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2082, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2082, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2082, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2082, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2082
    x2083 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2083 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1260, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1260
    x2084 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2084 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2083, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2083
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2084, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2084, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x2084
    x2085 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2085 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1258, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1258
    x2086 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2086 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2085, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2085
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2086, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2086, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x2086
    x2087 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2087 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1264, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x1264
    x2088 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2088 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2087, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2087
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2088, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2088, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x2088
    x2089 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2089 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1262, (4, 5, 2, 6), (4, 3, 0, 1, 5, 6)) * -1.0
    del x1262
    x2090 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2090 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2089, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2089
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2090, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2090, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x2090
    x2091 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2091 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1268, (4, 5, 6, 1), (4, 5, 0, 6, 2, 3))
    del x1268
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2091, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2091, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2091
    x2092 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2092 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1269, (4, 5, 1, 6), (4, 5, 0, 6, 2, 3))
    del x1269
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2092, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2092, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2092
    x2093 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2093 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1271, (4, 5, 3, 6), (4, 5, 0, 1, 2, 6))
    del x1271
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2093, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2093, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x2093
    x2094 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2094 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1272, (4, 5, 3, 6), (4, 5, 0, 1, 2, 6))
    del x1272
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2094, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2094, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x2094
    x2095 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2095 += einsum(x21, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5))
    del x21
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2095, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2095, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x2095
    x2096 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2096 += einsum(x0, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    x2097 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2097 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2096, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2096
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2097, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2097, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x2097
    x2098 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2098 += einsum(x13, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 0, 5, 6, 3, 7), (5, 7, 1, 4, 2, 6))
    del x13
    x2099 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2099 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2098, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2098
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2099, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2099, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2099, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2099, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2099
    x2100 = np.zeros((naocc[1], navir[1], navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2100 += einsum(x1460, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 3, 7), (5, 0, 7, 2, 4, 6))
    del x1460
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x2100, (0, 1, 2, 3, 4, 5), (3, 4, 0, 5, 2, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,sVb,sVfb)] += einsum(x2100, (0, 1, 2, 3, 4, 5), (4, 3, 0, 5, 2, 1)) * 2.0
    del x2100
    x2101 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2101 += einsum(x982, (0, 1, 2, 3), t3.babbab[np.ix_(sob,soa,sOfb,svb,sva,sVfb)], (4, 1, 5, 6, 3, 7), (0, 5, 7, 4, 2, 6))
    del x982
    x2102 = np.zeros((naocc[1], naocc[1], navir[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2102 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2101, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2101
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x2102, (0, 1, 2, 3, 4, 5), (3, 1, 0, 4, 5, 2)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sOb,sOfb,svb,svb,sVfb)] += einsum(x2102, (0, 1, 2, 3, 4, 5), (3, 1, 0, 5, 4, 2)) * 2.0
    del x2102
    x2103 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2103 += einsum(x276, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 6, 7), (4, 7, 1, 0, 5, 6)) * -1.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2103, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -3.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2103, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -3.0
    del x2103
    x2104 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2104 += einsum(x16, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 1, 5, 6, 3, 7), (5, 7, 0, 4, 2, 6))
    x2105 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2105 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2104, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2104
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2105, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2105, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2105, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2105, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x2105
    x2106 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2106 += einsum(x16, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 2, 5, 6, 3, 7), (5, 7, 0, 4, 1, 6))
    del x16
    x2107 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2107 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2106, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2106
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2107, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2107, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2107, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2107, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x2107
    x2108 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2108 += einsum(x22, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5))
    del x22
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2108, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2108, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x2108
    x2109 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2109 += einsum(x23, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 1, 3, 4, 5, 6), (3, 6, 0, 2, 4, 5))
    del x23
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2109, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2109, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x2109
    x2110 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2110 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2021, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x2021
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2110, (2, 3, 4, 5, 0, 6), (4, 5, 2, 1, 6, 3)) * 6.0
    del x2110
    x2111 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2111 += einsum(x1, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    x2112 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2112 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2111, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2111
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2112, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2112, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x2112
    x2113 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2113 += einsum(x20, (0, 1), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (2, 3, 4, 5, 1, 6), (4, 6, 2, 3, 0, 5))
    x2114 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2114 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2113, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2113
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2114, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2114, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x2114
    x2115 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2115 += einsum(x1306, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x1306
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2115, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2115, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 6.0
    del x2115
    x2116 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2116 += einsum(x1308, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 5, 6, 7, 1), (5, 0, 2, 4, 6, 7))
    del x1308
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2116, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2116, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -6.0
    del x2116
    x2117 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2117 += einsum(x1309, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 6, 7, 3, 1), (6, 0, 4, 5, 2, 7))
    del x1309
    x2118 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2118 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2117, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2117
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2118, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2118, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x2118
    x2119 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2119 += einsum(x1316, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 1, 5, 6, 7), (0, 7, 2, 4, 5, 6))
    del x1316
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2119, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 3.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2119, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -3.0
    del x2119
    x2120 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2120 += einsum(x1318, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 3, 1, 5, 6, 7), (0, 7, 2, 4, 5, 6))
    del x1318
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2120, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -3.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2120, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 3.0
    del x2120
    x2121 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2121 += einsum(x1317, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    del x1317
    x2122 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2122 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2121, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2121
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2122, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2122, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -6.0
    del x2122
    x2123 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2123 += einsum(x1315, (0, 1, 2, 3), t3.bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)], (4, 5, 1, 6, 3, 7), (0, 7, 4, 5, 2, 6))
    del x1315
    x2124 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2124 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2123, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2123
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2124, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -6.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2124, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 6.0
    del x2124
    x2125 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2125 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1368, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1368
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2125, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2125, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x2125
    x2126 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2126 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x272, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x272
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2126, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2126, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x2126
    x2127 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2127 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1369, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x1369
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2127, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2127, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2127, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2127, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x2127
    x2128 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2128 += einsum(x0, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,sVb)], (2, 3, 1, 4), (4, 2, 3, 0)) * -1.0
    del x0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x2128, (4, 5, 6, 0), (6, 5, 1, 2, 3, 4)) * 4.0
    del x2128
    x2129 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2129 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x274, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x274
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2129, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2129, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2129, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2129, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2129
    x2130 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2130 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1390, (2, 3, 4, 1), (2, 0, 3, 4))
    x2131 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2131 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x2130, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x2130
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2131, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2131, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2131
    x2132 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2132 += einsum(t2.abab[np.ix_(soa,sob,sva,svb)], (0, 1, 2, 3), x980, (4, 5, 0, 6, 7, 2), (4, 5, 6, 1, 7, 3))
    del x980
    x2133 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2133 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2132, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2132
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2133, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2133, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2133, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2133, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2133
    x2134 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2134 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1390, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    del x1390
    x2135 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2135 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2134, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2134
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2135, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2135, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2135, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2135, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2135
    x2136 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2136 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1381, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x1381
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2136, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2136, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2136, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2136, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    del x2136
    x2137 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2137 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1379, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1379
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2137, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2137, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2137
    x2138 = np.zeros((naocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2138 += einsum(t2.abab[np.ix_(soa,sOb,sva,svb)], (0, 1, 2, 3), v.aabb.ovov, (0, 2, 4, 5), (1, 4, 3, 5)) * -1.0
    x2139 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2139 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x2138, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x2138
    x2140 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2140 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2139, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2139
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2140, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2140, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x2140
    x2141 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2141 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1378, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1378
    x2142 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2142 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2141, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2141
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2142, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2142, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x2142
    x2143 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2143 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1399, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1399
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2143, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2143, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2143, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2143, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2143
    x2144 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2144 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1397, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1397
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2144, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2144, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    del x2144
    x2145 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2145 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1405, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x1405
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2145, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2145, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2145, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2145, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2145
    x2146 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2146 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1402, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1402
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2146, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2146, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x2146
    x2147 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2147 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x280, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x280
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2147, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2147, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2147, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2147, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    del x2147
    x2148 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2148 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1408, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1408
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2148, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2148, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x2148
    x2149 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2149 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x278, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x278
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2149, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2149, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2149, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2149, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    del x2149
    x2150 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2150 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1410, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1410
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2150, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2150, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    del x2150
    x2151 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2151 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1413, (2, 3, 4, 5, 6, 1), (2, 3, 0, 4, 5, 6))
    x2152 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2152 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x2151, (4, 5, 6, 7, 0, 1), (4, 5, 6, 7, 2, 3))
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2152, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2152, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2152, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2152, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2152
    x2153 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2153 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1411, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x1411
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2153, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2153, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2153, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2153, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    del x2153
    x2154 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2154 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1429, (2, 3, 4, 1), (2, 0, 3, 4))
    x2155 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2155 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x2154, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x2154
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2155, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2155, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    del x2155
    x2156 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2156 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1412, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x1412
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2156, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2156, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2156, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2156, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    del x2156
    x2157 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2157 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1427, (2, 3, 4, 1), (2, 0, 3, 4))
    x2158 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2158 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x2157, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3)) * -1.0
    del x2157
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2158, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2158, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    del x2158
    x2159 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2159 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1425, (4, 5, 6, 3), (1, 4, 5, 0, 2, 6))
    del x1425
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2159, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2159, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2159, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2159, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2159, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2159, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2159, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2159, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    del x2159
    x2160 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2160 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1413, (4, 5, 6, 7, 1, 3), (4, 5, 6, 0, 7, 2))
    x2161 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2161 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2160, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2160
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2161, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2161, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2161, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2161, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x2161
    x2162 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2162 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1413, (4, 5, 6, 1, 7, 3), (4, 5, 6, 0, 7, 2))
    del x1413
    x2163 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2163 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2162, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2162
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2163, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2163, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2163, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2163, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x2163
    x2164 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2164 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1407, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1407
    x2165 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2165 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2164, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2164
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2165, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2165, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x2165
    x2166 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2166 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x265, (4, 5, 0, 6), (1, 3, 4, 5, 6, 2)) * -1.0
    del x265
    x2167 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2167 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2166, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2166
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2167, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2167, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2167, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2167, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x2167
    x2168 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2168 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1409, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1409
    x2169 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2169 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2168, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2168
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2169, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2169, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x2169
    x2170 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2170 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1419, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2))
    del x1419
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2170, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2170, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2170, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2170, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2170, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2170, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2170, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2170, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x2170
    x2171 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2171 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1429, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    del x1429
    x2172 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2172 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2171, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2171
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2172, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2172, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2172, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2172, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x2172
    x2173 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2173 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1427, (4, 5, 6, 3), (1, 4, 0, 5, 6, 2)) * -1.0
    del x1427
    x2174 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2174 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2173, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2173
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2174, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2174, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2174, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2174, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x2174
    x2175 = np.zeros((naocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2175 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 5, 0, 3), (1, 4, 2, 5)) * -1.0
    x2176 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2176 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x2175, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x2175
    x2177 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2177 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2176, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2176
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2177, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2177, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x2177
    x2178 = np.zeros((naocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2178 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), v.bbbb.ovov, (4, 3, 0, 5), (1, 4, 2, 5)) * -1.0
    x2179 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2179 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x2178, (4, 5, 6, 2), (4, 3, 0, 1, 5, 6)) * -1.0
    del x2178
    x2180 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2180 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2179, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2179
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2180, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2180, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x2180
    x2181 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2181 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1421, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1421
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2181, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2181, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x2181
    x2182 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2182 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1422, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1422
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2182, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2182, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x2182
    x2183 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2183 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x286, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x286
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2183, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2183, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x2183
    x2184 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2184 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x288, (4, 5, 0, 6), (1, 3, 4, 5, 2, 6))
    del x288
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2184, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2184, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x2184
    x2185 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2185 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1423, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x1423
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2185, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2185, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2185, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2185, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x2185
    x2186 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2186 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1424, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x1424
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2186, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2186, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2186, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2186, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x2186
    x2187 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2187 += einsum(x1, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,sVb)], (2, 3, 1, 4), (4, 2, 3, 0)) * -1.0
    del x1
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x2187, (4, 5, 6, 0), (6, 5, 1, 2, 3, 4)) * 4.0
    del x2187
    x2188 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2188 += einsum(x20, (0, 1), t2.bbbb[np.ix_(sob,sob,svb,sVb)], (2, 3, 1, 4), (4, 2, 3, 0)) * -1.0
    del x20
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x2188, (4, 5, 6, 0), (6, 5, 1, 2, 3, 4)) * -4.0
    del x2188
    x2189 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2189 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1440, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1440
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2189, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2189, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2189, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2189, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x2189
    x2190 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2190 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1434, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3)) * -1.0
    del x1434
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2190, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2190, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2190, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2190, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x2190
    x2191 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2191 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1439, (4, 5, 6, 3), (1, 4, 0, 5, 2, 6)) * -1.0
    del x1439
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2191, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2191, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2191, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2191, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x2191
    x2192 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2192 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1437, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1437
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2192, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2192, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x2192
    x2193 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2193 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x961, (4, 5, 2, 3), (4, 0, 1, 5))
    del x961
    x2194 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2194 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x2193, (4, 5, 6, 0), (1, 4, 5, 6, 2, 3))
    del x2193
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2194, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2194, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x2194
    x2195 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2195 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1438, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1438
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2195, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2195, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x2195
    x2196 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2196 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1448, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x1448
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2196, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2196, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2196, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2196, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -4.0
    del x2196
    x2197 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2197 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1442, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1442
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2197, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2197, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 4.0
    del x2197
    x2198 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2198 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1447, (4, 5, 1, 6), (4, 3, 0, 5, 2, 6)) * -1.0
    del x1447
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2198, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2198, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2198, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2198, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 4.0
    del x2198
    x2199 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2199 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1444, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1444
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2199, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 4.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2199, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -4.0
    del x2199
    x2200 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2200 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1445, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2)) * -1.0
    del x1445
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2200, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2200, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2200, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2200, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    del x2200
    x2201 = np.zeros((naocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x2201 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x975, (4, 0, 1, 5), (4, 2, 3, 5))
    x2202 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2202 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x2201, (4, 5, 6, 2), (4, 3, 0, 1, 6, 5)) * -1.0
    del x2201
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2202, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2202, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x2202
    x2203 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2203 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,sVb)], (0, 1, 2, 3), x276, (4, 5, 0, 6), (1, 3, 5, 4, 6, 2))
    del x276
    x2204 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2204 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2203, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2203
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2204, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2204, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2204, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2204, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2204
    x2205 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2205 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2151, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x2151
    x2206 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2206 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2205, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2205
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2206, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2206, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2206, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2206, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2206
    x2207 = np.zeros((navir[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2207 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x1469, (2, 3, 4, 1), (2, 3, 0, 4))
    x2208 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2208 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x2207, (4, 5, 6, 0), (1, 4, 6, 5, 2, 3)) * -1.0
    del x2207
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2208, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2208, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2208
    x2209 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2209 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1469, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2)) * -1.0
    del x1469
    x2210 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2210 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2209, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2209
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2210, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2210, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2210, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2210, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2210
    x2211 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2211 += einsum(t2.bbbb[np.ix_(sob,sOb,svb,svb)], (0, 1, 2, 3), x1471, (4, 5, 6, 3), (1, 4, 5, 0, 6, 2)) * -1.0
    del x1471
    x2212 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2212 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2211, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2211
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2212, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2212, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2212, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2212, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2212
    x2213 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2213 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1474, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x1474
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2213, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2213, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2213, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2213, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2213
    x2214 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2214 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x1475, (4, 5, 1, 6), (4, 3, 5, 0, 6, 2)) * -1.0
    del x1475
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2214, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2214, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2214, (0, 1, 2, 3, 4, 5), (3, 2, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2214, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2214
    x2215 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nocc[1]), dtype=np.float64)
    x2215 += einsum(t2.bbbb[np.ix_(sob,sob,svb,sVb)], (0, 1, 2, 3), x975, (4, 5, 6, 2), (4, 3, 0, 1, 6, 5)) * -1.0
    del x975
    x2216 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2216 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2215, (2, 3, 4, 5, 6, 0), (2, 3, 4, 5, 6, 1))
    del x2215
    x2217 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2217 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2216, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2216
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2217, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2217, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x2217
    x2218 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2218 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1477, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1477
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2218, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2218, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * 2.0
    del x2218
    x2219 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2219 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1479, (4, 5, 6, 1), (4, 5, 6, 0, 2, 3))
    del x1479
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2219, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2219, (0, 1, 2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -2.0
    del x2219
    x2220 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2220 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1476, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1476
    x2221 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2221 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2220, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2220
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2221, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * 2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2221, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * -2.0
    del x2221
    x2222 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2222 += einsum(t2.bbbb[np.ix_(sob,sob,svb,svb)], (0, 1, 2, 3), x1478, (4, 5, 6, 3), (4, 5, 0, 1, 6, 2))
    del x1478
    x2223 = np.zeros((naocc[1], navir[1], nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2223 += einsum(t1.bb[np.ix_(sob,svb)], (0, 1), x2222, (2, 3, 4, 5, 0, 6), (2, 3, 4, 5, 1, 6))
    del x2222
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2223, (0, 1, 2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -2.0
    t3new_bbbbbb[np.ix_(sob,sob,sOfb,svb,svb,sVfb)] += einsum(x2223, (0, 1, 2, 3, 4, 5), (2, 3, 0, 5, 4, 1)) * 2.0
    del x2223

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

