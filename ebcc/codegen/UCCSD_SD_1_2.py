# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, **kwargs):
    # Energy
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x0 += einsum("iajb->jiab", v.aaaa.ovov) * -1
    x0 += einsum("iajb->jiba", v.aaaa.ovov)
    x1 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x1 += einsum("ijab->jiba", t2.aaaa)
    x1 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    e_cc = 0
    e_cc += einsum("ijab,ijba->", x0, x1) * -0.5
    del x0
    del x1
    x2 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x2 += einsum("iajb->jiab", v.bbbb.ovov) * -1
    x2 += einsum("iajb->jiba", v.bbbb.ovov)
    x3 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x3 += einsum("ijab->jiba", t2.bbbb)
    x3 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    e_cc += einsum("ijab,ijba->", x2, x3) * -0.5
    del x2
    del x3
    x4 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x4 += einsum("ijab->ijab", t2.abab)
    x4 += einsum("ia,jb->ijab", t1.aa, t1.bb)
    e_cc += einsum("iajb,ijab->", v.aabb.ovov, x4)
    del x4
    x5 = np.zeros((nbos), dtype=types[float])
    x5 += einsum("w->w", G)
    x5 += einsum("ia,wia->w", t1.aa, g.aa.bov)
    x5 += einsum("ia,wia->w", t1.bb, g.bb.bov)
    e_cc += einsum("w,w->", s1, x5)
    del x5
    e_cc += einsum("ia,ia->", f.aa.ov, t1.aa)
    e_cc += einsum("wia,wia->", g.bb.bov, u11.bb)
    e_cc += einsum("ia,ia->", f.bb.ov, t1.bb)
    e_cc += einsum("wia,wia->", g.aa.bov, u11.aa)

    return e_cc

def update_amps(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, **kwargs):
    t1new = Namespace()
    t2new = Namespace()
    u11new = Namespace()
    u12new = Namespace()

    # Get boson coupling creation array:
    gc = Namespace(
        aa = Namespace(
            boo=g.aa.boo.transpose(0, 2, 1),
            bov=g.aa.bvo.transpose(0, 2, 1),
            bvo=g.aa.bov.transpose(0, 2, 1),
            bvv=g.aa.bvv.transpose(0, 2, 1),
        ),
        bb = Namespace(
            boo=g.bb.boo.transpose(0, 2, 1),
            bov=g.bb.bvo.transpose(0, 2, 1),
            bvo=g.bb.bov.transpose(0, 2, 1),
            bvv=g.bb.bvv.transpose(0, 2, 1),
        ),
    )

    # T1, T2, S1, S2, U11 and U12 amplitudes
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x0 += einsum("ia,jbka->ijkb", t1.aa, v.aaaa.ovov)
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum("ijka->ijka", x0)
    x1 += einsum("ijka->ikja", x0) * -1
    x69 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x69 += einsum("ijka->ijka", x0)
    x69 -= einsum("ijka->ikja", x0)
    x127 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x127 += einsum("ia,jkla->jilk", t1.aa, x0)
    x128 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x128 += einsum("ijab,klij->lkba", t2.aaaa, x127)
    x132 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x132 += einsum("ijab->ijab", x128)
    del x128
    x133 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x133 += einsum("ia,jkli->jkla", t1.aa, x127)
    del x127
    x134 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x134 += einsum("ia,jkib->jkab", t1.aa, x133)
    del x133
    x141 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x141 += einsum("ijab->ijab", x134)
    del x134
    x259 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x259 += einsum("ijka->jkia", x0) * -1
    x259 += einsum("ijka->kjia", x0)
    del x0
    x1 += einsum("ijka->jika", v.aaaa.ooov) * -1
    x1 += einsum("ijka->jkia", v.aaaa.ooov)
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    t1new_aa += einsum("ijab,kjib->ka", t2.aaaa, x1) * -1
    del x1
    x2 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum("iabc->ibac", v.aaaa.ovvv) * -1
    x2 += einsum("iabc->ibca", v.aaaa.ovvv)
    x99 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x99 += einsum("ia,jbca->ijbc", t1.aa, x2)
    x105 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x105 += einsum("ia,ibca->bc", t1.aa, x2)
    x106 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x106 += einsum("ab->ab", x105) * -1
    del x105
    t1new_aa += einsum("ijab,icab->jc", t2.aaaa, x2) * -1
    del x2
    x3 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x3 += einsum("ia,jakb->ijkb", t1.aa, v.aabb.ovov)
    x4 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x4 += einsum("ijka->jika", x3)
    x112 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x112 += einsum("ijab,kljb->kila", t2.abab, x3)
    del x3
    x115 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x115 += einsum("ijka->ikja", x112)
    del x112
    x4 += einsum("ijka->ijka", v.aabb.ooov)
    x283 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x283 += einsum("ijab,iklb->kjla", t2.abab, x4)
    x287 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x287 += einsum("ijka->ikja", x283) * -1
    del x283
    x374 = np.zeros((nbos, nbos, nocc[0], nocc[0]), dtype=types[float])
    x374 += einsum("wxia,jkia->xwjk", u12.bb, x4)
    t1new_aa += einsum("ijab,ikjb->ka", t2.abab, x4) * -1
    x5 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x5 += einsum("w,wia->ia", s1, g.aa.bov)
    x9 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x9 += einsum("ia->ia", x5)
    x73 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x73 += einsum("ia->ia", x5)
    x297 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x297 += einsum("ia->ia", x5)
    x321 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x321 += einsum("ia->ia", x5)
    x343 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x343 += einsum("ia->ia", x5)
    del x5
    x6 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum("ia,jbia->jb", t1.bb, v.aabb.ovov)
    x9 += einsum("ia->ia", x6)
    x97 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x97 += einsum("ia,ja->ij", t1.aa, x6)
    x98 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x98 += einsum("ij,kjab->ikab", x97, t2.aaaa)
    del x97
    x119 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x119 += einsum("ijab->ijab", x98) * -1
    del x98
    x113 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x113 += einsum("ia,jkba->jkib", x6, t2.aaaa)
    x115 += einsum("ijka->ikja", x113) * -1
    del x113
    x297 += einsum("ia->ia", x6)
    x316 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x316 += einsum("ia->ia", x6)
    del x6
    x7 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x7 += einsum("iajb->jiab", v.aaaa.ovov)
    x7 += einsum("iajb->jiba", v.aaaa.ovov) * -1
    x8 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum("ia,ijab->jb", t1.aa, x7)
    x9 += einsum("ia->ia", x8) * -1
    x316 += einsum("ia->ia", x8) * -1
    del x8
    x317 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x317 += einsum("ia,ib->ab", t1.aa, x316) * -1
    x365 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x365 += einsum("ia,wja->wij", x316, u11.aa)
    del x316
    x366 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x366 += einsum("wia,xij->xwja", u11.aa, x365)
    del x365
    x367 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x367 += einsum("wxia->wxia", x366)
    del x366
    x23 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x23 += einsum("ijab,ikab->jk", t2.aaaa, x7)
    x27 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x27 += einsum("ij->ji", x23) * -1
    x117 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x117 += einsum("ij->ji", x23) * -1
    del x23
    x104 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x104 += einsum("ijab,ijcb->ac", t2.aaaa, x7)
    x106 += einsum("ab->ab", x104) * -1
    del x104
    x231 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x231 += einsum("ijab,ikac->kjcb", t2.abab, x7)
    x232 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x232 += einsum("ijab->ijab", x231) * -1
    del x231
    x311 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x311 += einsum("wia,ijab->wjb", u11.aa, x7)
    x312 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x312 += einsum("wia->wia", x311) * -1
    x360 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x360 += einsum("wia->wia", x311) * -1
    del x311
    x369 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x369 += einsum("wxia,ijab->wxjb", u12.aa, x7)
    del x7
    x370 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x370 += einsum("wxia->xwia", x369) * -1
    del x369
    x9 += einsum("ia->ia", f.aa.ov)
    x26 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x26 += einsum("ia,ja->ij", t1.aa, x9)
    x27 += einsum("ij->ji", x26)
    del x26
    x263 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x263 += einsum("ia,jkab->jikb", x9, t2.abab)
    x269 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x269 += einsum("ijka->jika", x263)
    del x263
    x374 += einsum("ia,wxja->xwij", x9, u12.aa)
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    t1new_bb += einsum("ia,ijab->jb", x9, t2.abab)
    x10 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x10 += einsum("ijab->jiab", t2.aaaa)
    x10 += einsum("ijab->jiba", t2.aaaa) * -1
    x100 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x100 += einsum("ijab,kica->kjcb", x10, x99) * -1
    del x99
    x119 += einsum("ijab->ijba", x100) * -1
    del x100
    t1new_aa += einsum("ia,ijab->jb", x9, x10) * -1
    del x9
    x11 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x11 += einsum("w,wia->ia", s1, g.bb.bov)
    x15 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x15 += einsum("ia->ia", x11)
    x154 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x154 += einsum("ia->ia", x11)
    x296 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x296 += einsum("ia->ia", x11)
    x332 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x332 += einsum("ia->ia", x11)
    x379 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x379 += einsum("ia->ia", x11)
    del x11
    x12 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x12 += einsum("ia,iajb->jb", t1.aa, v.aabb.ovov)
    x15 += einsum("ia->ia", x12)
    x193 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x193 += einsum("ia,jkab->kjib", x12, t2.bbbb)
    x196 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x196 += einsum("ijka->ikja", x193) * -1
    del x193
    x206 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x206 += einsum("ia,ja->ij", t1.bb, x12)
    x207 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x207 += einsum("ij->ij", x206)
    del x206
    x296 += einsum("ia->ia", x12)
    x327 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x327 += einsum("ia->ia", x12)
    del x12
    x13 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x13 += einsum("iajb->jiab", v.bbbb.ovov)
    x13 += einsum("iajb->jiba", v.bbbb.ovov) * -1
    x14 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x14 += einsum("ia,ijab->jb", t1.bb, x13)
    x15 += einsum("ia->ia", x14) * -1
    x327 += einsum("ia->ia", x14) * -1
    del x14
    x328 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x328 += einsum("ia,ib->ab", t1.bb, x327) * -1
    x401 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x401 += einsum("ia,wja->wij", x327, u11.bb)
    del x327
    x402 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x402 += einsum("wij->wji", x401)
    del x401
    x271 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x271 += einsum("ijab,ijbc->ca", t2.bbbb, x13)
    x272 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x272 += einsum("ab->ba", x271) * -1
    x328 += einsum("ab->ba", x271) * -1
    del x271
    x314 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x314 += einsum("wia,ijab->wjb", u11.bb, x13)
    x315 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x315 += einsum("wia->wia", x314) * -1
    x405 = np.zeros((nbos, nbos, nocc[1], nocc[1]), dtype=types[float])
    x405 += einsum("wia,xja->xwji", u11.bb, x314) * -1
    x406 = np.zeros((nbos, nbos, nocc[1], nocc[1]), dtype=types[float])
    x406 += einsum("wxij->xwji", x405)
    del x405
    x408 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x408 += einsum("ia,wja->wji", t1.bb, x314) * -1
    del x314
    x409 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x409 += einsum("wia,xij->xwja", u11.bb, x408)
    del x408
    x410 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x410 += einsum("wxia->xwia", x409)
    del x409
    x372 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x372 += einsum("wxia,ijab->wxjb", u12.bb, x13)
    del x13
    x373 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x373 += einsum("wxia->xwia", x372) * -1
    del x372
    x15 += einsum("ia->ia", f.bb.ov)
    x53 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x53 += einsum("ia,ja->ij", t1.bb, x15)
    x54 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x54 += einsum("ij->ji", x53)
    del x53
    x284 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x284 += einsum("ia,jkba->jkib", x15, t2.abab)
    x287 += einsum("ijka->ikja", x284)
    del x284
    x413 = np.zeros((nbos, nbos, nocc[1], nocc[1]), dtype=types[float])
    x413 += einsum("ia,wxja->xwij", x15, u12.bb)
    t1new_aa += einsum("ia,jiba->jb", x15, t2.abab)
    x16 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x16 += einsum("iabj->ijba", v.aaaa.ovvo)
    x16 -= einsum("ijab->ijab", v.aaaa.oovv)
    x75 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x75 += einsum("ia,jkba->ijkb", t1.aa, x16)
    x76 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x76 -= einsum("ijka->jika", x75)
    del x75
    t1new_aa += einsum("ia,ijba->jb", t1.aa, x16)
    u11new_aa = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    u11new_aa += einsum("wia,ijba->wjb", u11.aa, x16)
    x17 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x17 += einsum("ia,wja->wji", t1.aa, g.aa.bov)
    x18 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x18 += einsum("wij->wij", x17)
    del x17
    x18 += einsum("wij->wij", g.aa.boo)
    x84 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x84 += einsum("ia,wij->wja", t1.aa, x18)
    x85 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x85 -= einsum("wia->wia", x84)
    del x84
    t1new_aa -= einsum("wia,wij->ja", u11.aa, x18)
    u11new_aa -= einsum("wij,wxia->xja", x18, u12.aa)
    del x18
    x19 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x19 += einsum("w,wij->ij", s1, g.aa.boo)
    x27 += einsum("ij->ij", x19)
    x79 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x79 += einsum("ij->ji", x19)
    del x19
    x20 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x20 += einsum("wia,wja->ij", g.aa.bov, u11.aa)
    x27 += einsum("ij->ij", x20)
    x92 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x92 += einsum("ij->ij", x20)
    del x20
    x21 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x21 += einsum("ia,jkia->jk", t1.bb, v.aabb.ooov)
    x27 += einsum("ij->ij", x21)
    x92 += einsum("ij->ij", x21)
    del x21
    x93 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x93 += einsum("ij,ikab->jkab", x92, t2.aaaa)
    del x92
    x94 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x94 -= einsum("ijab->jiba", x93)
    del x93
    x22 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x22 += einsum("ijab,kajb->ik", t2.abab, v.aabb.ovov)
    x27 += einsum("ij->ji", x22)
    x117 += einsum("ij->ji", x22)
    del x22
    x24 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x24 += einsum("ijka->ikja", v.aaaa.ooov)
    x24 += einsum("ijka->kija", v.aaaa.ooov) * -1
    x25 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x25 += einsum("ia,ijka->jk", t1.aa, x24)
    x27 += einsum("ij->ij", x25) * -1
    x117 += einsum("ij->ij", x25) * -1
    del x25
    x118 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x118 += einsum("ij,ikab->jkab", x117, t2.aaaa)
    del x117
    x119 += einsum("ijab->jiba", x118)
    del x118
    x114 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x114 += einsum("ijab,kila->jklb", x10, x24)
    x115 += einsum("ijka->ijka", x114)
    del x114
    x322 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x322 += einsum("wia,ijka->wjk", u11.aa, x24) * -1
    del x24
    x27 += einsum("ij->ij", f.aa.oo)
    x290 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x290 += einsum("ij,ikab->jkab", x27, t2.abab)
    t2new_baba = np.zeros((nocc[1], nocc[0], nvir[1], nvir[0]), dtype=types[float])
    t2new_baba += einsum("ijab->jiba", x290) * -1
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum("ijab->ijab", x290) * -1
    del x290
    t1new_aa += einsum("ia,ij->ja", t1.aa, x27) * -1
    u11new_aa += einsum("ij,wia->wja", x27, u11.aa) * -1
    u12new_aa = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    u12new_aa += einsum("ij,wxia->xwja", x27, u12.aa) * -1
    del x27
    x28 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x28 += einsum("w,wab->ab", s1, g.aa.bvv)
    x32 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x32 += einsum("ab->ab", x28)
    x87 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x87 += einsum("ab->ab", x28)
    x276 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x276 += einsum("ab->ab", x28)
    x317 += einsum("ab->ab", x28)
    del x28
    x29 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x29 += einsum("ia,iabc->bc", t1.bb, v.bbaa.ovvv)
    x32 += einsum("ab->ab", x29)
    x90 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x90 -= einsum("ab->ba", x29)
    x276 += einsum("ab->ab", x29)
    x317 += einsum("ab->ab", x29)
    del x29
    x30 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x30 += einsum("iabc->ibac", v.aaaa.ovvv)
    x30 += einsum("iabc->ibca", v.aaaa.ovvv) * -1
    x31 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x31 += einsum("ia,ibca->bc", t1.aa, x30)
    del x30
    x32 += einsum("ab->ab", x31) * -1
    x276 += einsum("ab->ab", x31) * -1
    x317 += einsum("ab->ab", x31) * -1
    del x31
    x32 += einsum("ab->ab", f.aa.vv)
    t1new_aa += einsum("ia,ba->ib", t1.aa, x32)
    del x32
    x33 = np.zeros((nbos), dtype=types[float])
    x33 += einsum("ia,wia->w", t1.aa, g.aa.bov)
    x35 = np.zeros((nbos), dtype=types[float])
    x35 += einsum("w->w", x33)
    del x33
    x34 = np.zeros((nbos), dtype=types[float])
    x34 += einsum("ia,wia->w", t1.bb, g.bb.bov)
    x35 += einsum("w->w", x34)
    del x34
    x35 += einsum("w->w", G)
    t1new_aa += einsum("w,wia->ia", x35, u11.aa)
    t1new_bb += einsum("w,wia->ia", x35, u11.bb)
    s1new = np.zeros((nbos), dtype=types[float])
    s1new += einsum("w,wx->x", x35, s2)
    u11new_aa += einsum("w,wxia->xia", x35, u12.aa)
    u11new_bb = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    u11new_bb += einsum("w,wxia->xia", x35, u12.bb)
    del x35
    x36 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x36 += einsum("ia,jbka->ijkb", t1.bb, v.bbbb.ovov)
    x37 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x37 += einsum("ijka->ijka", x36) * -1
    x37 += einsum("ijka->ikja", x36)
    x151 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x151 -= einsum("ijka->ijka", x36)
    x151 += einsum("ijka->ikja", x36)
    x216 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x216 += einsum("ia,jkla->jilk", t1.bb, x36)
    x217 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x217 += einsum("ijkl->ijkl", x216)
    x223 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x223 += einsum("ia,jkli->jkla", t1.bb, x216)
    del x216
    x224 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x224 += einsum("ia,jkib->jkab", t1.bb, x223)
    del x223
    x228 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x228 += einsum("ijab->ijab", x224)
    del x224
    x280 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x280 += einsum("ijka->jkia", x36) * -1
    x280 += einsum("ijka->kjia", x36)
    del x36
    x37 += einsum("ijka->jika", v.bbbb.ooov)
    x37 += einsum("ijka->jkia", v.bbbb.ooov) * -1
    t1new_bb += einsum("ijab,kijb->ka", t2.bbbb, x37) * -1
    del x37
    x38 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x38 += einsum("iabc->ibac", v.bbbb.ovvv) * -1
    x38 += einsum("iabc->ibca", v.bbbb.ovvv)
    x57 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x57 += einsum("ia,ibac->bc", t1.bb, x38)
    x58 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x58 += einsum("ab->ab", x57) * -1
    x272 += einsum("ab->ab", x57) * -1
    x328 += einsum("ab->ab", x57) * -1
    del x57
    t1new_bb += einsum("ijab,icab->jc", t2.bbbb, x38) * -1
    del x38
    x39 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x39 += einsum("ia,jbka->jikb", t1.bb, v.aabb.ovov)
    x40 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x40 += einsum("ijka->ikja", x39)
    x192 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x192 += einsum("ijab,ikla->kjlb", t2.abab, x39)
    x196 += einsum("ijka->ikja", x192)
    del x192
    x250 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x250 += einsum("ijka->ikja", x39)
    del x39
    x40 += einsum("iajk->ijka", v.aabb.ovoo)
    x262 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x262 += einsum("ijab,kjla->iklb", t2.abab, x40)
    x269 += einsum("ijka->jika", x262) * -1
    del x262
    x264 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x264 += einsum("ia,jkla->ijkl", t1.aa, x40)
    x265 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x265 += einsum("ijkl->jikl", x264)
    del x264
    x282 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x282 += einsum("ijab,ikla->jklb", x10, x40)
    x287 += einsum("ijka->ijka", x282) * -1
    del x282
    x413 += einsum("wxia,ijka->xwjk", u12.aa, x40)
    t1new_bb += einsum("ijab,ijka->kb", t2.abab, x40) * -1
    del x40
    x41 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x41 += einsum("ijab->jiab", t2.bbbb)
    x41 += einsum("ijab->jiba", t2.bbbb) * -1
    x186 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x186 += einsum("iajb,jkbc->ikac", v.aabb.ovov, x41)
    x187 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x187 += einsum("ijab,ikac->kjcb", t2.abab, x186) * -1
    x209 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x209 += einsum("ijab->jiba", x187) * -1
    del x187
    x232 += einsum("ijab->ijab", x186) * -1
    del x186
    x261 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x261 += einsum("ijka,klab->ijlb", x4, x41)
    del x4
    x269 += einsum("ijka->ijka", x261) * -1
    del x261
    t1new_bb += einsum("ia,ijab->jb", x15, x41) * -1
    del x15
    x42 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x42 += einsum("iabj->ijba", v.bbbb.ovvo)
    x42 -= einsum("ijab->ijab", v.bbbb.oovv)
    x156 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x156 += einsum("ia,jkba->ijkb", t1.bb, x42)
    x157 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x157 -= einsum("ijka->jika", x156)
    del x156
    t1new_bb += einsum("ia,ijba->jb", t1.bb, x42)
    u11new_bb += einsum("wia,ijba->wjb", u11.bb, x42)
    x43 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x43 += einsum("ia,wja->wji", t1.bb, g.bb.bov)
    x44 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x44 += einsum("wij->wij", x43)
    del x43
    x44 += einsum("wij->wij", g.bb.boo)
    x166 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x166 += einsum("ia,wij->wja", t1.bb, x44)
    x167 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x167 -= einsum("wia->wia", x166)
    del x166
    t1new_bb -= einsum("wia,wij->ja", u11.bb, x44)
    u11new_bb -= einsum("wij,wxia->xja", x44, u12.bb)
    del x44
    x45 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x45 += einsum("w,wij->ij", s1, g.bb.boo)
    x54 += einsum("ij->ij", x45)
    x160 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x160 += einsum("ij->ji", x45)
    del x45
    x46 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x46 += einsum("wia,wja->ij", g.bb.bov, u11.bb)
    x54 += einsum("ij->ij", x46)
    x174 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x174 += einsum("ij->ij", x46)
    del x46
    x47 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x47 += einsum("ia,iajk->jk", t1.aa, v.aabb.ovoo)
    x54 += einsum("ij->ij", x47)
    x174 += einsum("ij->ij", x47)
    del x47
    x175 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x175 += einsum("ij,ikab->jkab", x174, t2.bbbb)
    del x174
    x176 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x176 -= einsum("ijab->jiba", x175)
    del x175
    x48 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x48 += einsum("ijab,iakb->jk", t2.abab, v.aabb.ovov)
    x54 += einsum("ij->ji", x48)
    x207 += einsum("ij->ij", x48)
    del x48
    x208 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x208 += einsum("ij,jkab->ikab", x207, t2.bbbb)
    del x207
    x209 += einsum("ijab->ijba", x208) * -1
    del x208
    x49 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x49 += einsum("iajb->jiab", v.bbbb.ovov) * -1
    x49 += einsum("iajb->jiba", v.bbbb.ovov)
    x50 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x50 += einsum("ijab,ikba->jk", t2.bbbb, x49)
    x54 += einsum("ij->ji", x50) * -1
    del x50
    x198 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x198 += einsum("ijab,ijbc->ac", t2.bbbb, x49)
    x200 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x200 += einsum("ab->ab", x198) * -1
    del x198
    x202 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x202 += einsum("ijab,ikab->jk", t2.bbbb, x49)
    x204 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x204 += einsum("ij->ji", x202) * -1
    del x202
    x235 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x235 += einsum("ijab,ikca->jkbc", x41, x49)
    del x49
    x236 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x236 += einsum("ijab->jiab", x235)
    del x235
    x51 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x51 += einsum("ijka->ikja", v.bbbb.ooov)
    x51 += einsum("ijka->kija", v.bbbb.ooov) * -1
    x52 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x52 += einsum("ia,ijka->jk", t1.bb, x51)
    x54 += einsum("ij->ij", x52) * -1
    del x52
    x333 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x333 += einsum("wia,ijka->wjk", u11.bb, x51) * -1
    del x51
    x54 += einsum("ij->ij", f.bb.oo)
    x289 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x289 += einsum("ij,kiab->kjab", x54, t2.abab)
    t2new_baba += einsum("ijab->jiba", x289) * -1
    t2new_abab += einsum("ijab->ijab", x289) * -1
    del x289
    t1new_bb += einsum("ia,ij->ja", t1.bb, x54) * -1
    u11new_bb += einsum("ij,wia->wja", x54, u11.bb) * -1
    u12new_bb = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    u12new_bb += einsum("ij,wxia->xwja", x54, u12.bb) * -1
    del x54
    x55 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x55 += einsum("w,wab->ab", s1, g.bb.bvv)
    x58 += einsum("ab->ab", x55)
    x169 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x169 += einsum("ab->ab", x55)
    x272 += einsum("ab->ab", x55)
    x328 += einsum("ab->ab", x55)
    del x55
    x56 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x56 += einsum("ia,iabc->bc", t1.aa, v.aabb.ovvv)
    x58 += einsum("ab->ab", x56)
    x172 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x172 -= einsum("ab->ba", x56)
    x272 += einsum("ab->ab", x56)
    x328 += einsum("ab->ab", x56)
    del x56
    x58 += einsum("ab->ab", f.bb.vv)
    t1new_bb += einsum("ia,ba->ib", t1.bb, x58)
    del x58
    x59 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x59 += einsum("ia,bjca->ijbc", t1.aa, v.aaaa.vovv)
    x94 -= einsum("ijab->ijab", x59)
    del x59
    x60 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x60 += einsum("ijab,jbck->ikac", t2.abab, v.bbaa.ovvo)
    x94 += einsum("ijab->ijab", x60)
    del x60
    x61 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x61 += einsum("ia,jbca->ijcb", t1.aa, v.bbaa.ovvv)
    x62 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x62 += einsum("ijab,kjcb->kiac", t2.abab, x61)
    x94 -= einsum("ijab->ijab", x62)
    del x62
    x238 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x238 += einsum("ijab->ijab", x61)
    del x61
    x63 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x63 -= einsum("ijab->jiab", t2.aaaa)
    x63 += einsum("ijab->jiba", t2.aaaa)
    x64 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x64 += einsum("ijab,ikcb->kjca", x16, x63)
    del x16
    x94 -= einsum("ijab->ijab", x64)
    del x64
    x65 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x65 += einsum("iajb->jiab", v.aaaa.ovov)
    x65 -= einsum("iajb->jiba", v.aaaa.ovov)
    x66 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x66 += einsum("ijab,ikbc->kjca", t2.aaaa, x65)
    x67 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x67 -= einsum("ijab,ikac->kjcb", t2.aaaa, x66)
    x94 -= einsum("ijab->jiba", x67)
    del x67
    x137 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x137 -= einsum("ijab,ikbc->kjca", t2.aaaa, x66)
    del x66
    x141 += einsum("ijab->jiba", x137)
    del x137
    x72 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x72 += einsum("ia,ijab->jb", t1.aa, x65)
    x73 -= einsum("ia->ia", x72)
    x297 -= einsum("ia->ia", x72)
    del x72
    x135 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x135 += einsum("ijab,ikac->kjcb", t2.aaaa, x65)
    x136 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x136 -= einsum("ijab,ikac->kjcb", t2.aaaa, x135)
    del x135
    x141 += einsum("ijab->ijab", x136)
    del x136
    x177 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x177 += einsum("ijab,ikac->kjcb", t2.abab, x65)
    x178 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x178 -= einsum("ijab,ikac->kjcb", t2.abab, x177)
    del x177
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb -= einsum("ijab->ijba", x178)
    t2new_bbbb += einsum("ijab->jiba", x178)
    del x178
    x308 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x308 -= einsum("wia,ijab->wjb", u11.aa, x65)
    del x65
    s2new = np.zeros((nbos, nbos), dtype=types[float])
    s2new += einsum("wia,xia->wx", u11.aa, x308)
    del x308
    x68 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x68 += einsum("ijab,kljb->ikla", t2.abab, v.aabb.ooov)
    x76 += einsum("ijka->jika", x68)
    del x68
    x70 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x70 += einsum("ijab->jiab", t2.aaaa)
    x70 -= einsum("ijab->jiba", t2.aaaa)
    x71 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x71 += einsum("ijka,jlab->iklb", x69, x70)
    del x69
    x76 += einsum("ijka->jika", x71)
    del x71
    x83 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x83 += einsum("wia,ijab->wjb", g.aa.bov, x70)
    del x70
    x85 -= einsum("wia->wia", x83)
    del x83
    x73 += einsum("ia->ia", f.aa.ov)
    x74 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x74 += einsum("ia,jkab->ijkb", x73, t2.aaaa)
    x76 += einsum("ijka->ikja", x74)
    del x74
    x78 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x78 += einsum("ia,ja->ji", t1.aa, x73)
    del x73
    x79 += einsum("ij->ji", x78)
    del x78
    x76 -= einsum("ijak->ijka", v.aaaa.oovo)
    x77 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x77 += einsum("ia,ijkb->jkba", t1.aa, x76)
    del x76
    x94 += einsum("ijab->ijba", x77)
    del x77
    x79 += einsum("ij->ji", f.aa.oo)
    x80 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x80 += einsum("ij,jkab->ikab", x79, t2.aaaa)
    del x79
    x94 += einsum("ijab->ijba", x80)
    del x80
    x81 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x81 += einsum("ia,wba->wib", t1.aa, g.aa.bvv)
    x85 += einsum("wia->wia", x81)
    del x81
    x82 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x82 += einsum("wia,jiba->wjb", g.bb.bov, t2.abab)
    x85 += einsum("wia->wia", x82)
    del x82
    x85 += einsum("wai->wia", g.aa.bvo)
    x86 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x86 += einsum("wia,wjb->jiba", u11.aa, x85)
    x94 += einsum("ijab->ijab", x86)
    del x86
    x292 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x292 += einsum("wia,wjb->jiba", u11.bb, x85)
    del x85
    t2new_baba += einsum("ijab->jiba", x292)
    t2new_abab += einsum("ijab->ijab", x292)
    del x292
    x87 += einsum("ab->ab", f.aa.vv)
    x88 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x88 += einsum("ab,ijbc->ijac", x87, t2.aaaa)
    del x87
    x94 -= einsum("ijab->jiab", x88)
    del x88
    x89 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x89 += einsum("wia,wib->ab", g.aa.bov, u11.aa)
    x90 += einsum("ab->ab", x89)
    x91 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x91 += einsum("ab,ijac->ijbc", x90, t2.aaaa)
    del x90
    x94 -= einsum("ijab->jiba", x91)
    del x91
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum("ijab->ijab", x94)
    t2new_aaaa -= einsum("ijab->ijba", x94)
    t2new_aaaa -= einsum("ijab->jiab", x94)
    t2new_aaaa += einsum("ijab->jiba", x94)
    del x94
    x276 += einsum("ab->ba", x89) * -1
    x317 += einsum("ab->ba", x89) * -1
    del x89
    x95 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x95 += einsum("ia,jkla->ijlk", t1.aa, v.aaaa.ooov)
    x96 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x96 += einsum("ijab,kjil->klba", t2.aaaa, x95)
    x119 += einsum("ijab->ijab", x96)
    del x96
    x108 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x108 += einsum("ia,jkil->jkla", t1.aa, x95)
    del x95
    x115 += einsum("ijka->ijka", x108)
    del x108
    x101 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x101 += einsum("ijab,kcjb->ikac", t2.abab, v.aabb.ovov)
    x102 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x102 += einsum("ijab,kicb->kjca", x10, x101)
    del x101
    x119 += einsum("ijab->jiba", x102) * -1
    del x102
    x103 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x103 += einsum("ijab,icjb->ac", t2.abab, v.aabb.ovov)
    x106 += einsum("ab->ab", x103)
    x107 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x107 += einsum("ab,ijbc->ijac", x106, t2.aaaa)
    del x106
    x119 += einsum("ijab->jiba", x107)
    del x107
    x276 += einsum("ab->ab", x103) * -1
    x317 += einsum("ab->ab", x103) * -1
    del x103
    x109 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x109 += einsum("ijab,kacb->ijkc", t2.aaaa, v.aaaa.ovvv)
    x115 += einsum("ijka->ikja", x109)
    del x109
    x110 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x110 += einsum("ia,jbca->ijcb", t1.aa, v.aaaa.ovvv)
    x111 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x111 += einsum("ia,jkba->ijkb", t1.aa, x110)
    del x110
    x115 += einsum("ijka->ikja", x111)
    del x111
    x116 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x116 += einsum("ia,jikb->jkba", t1.aa, x115)
    del x115
    x119 += einsum("ijab->ijba", x116)
    del x116
    t2new_aaaa += einsum("ijab->ijab", x119) * -1
    t2new_aaaa += einsum("ijab->ijba", x119)
    t2new_aaaa += einsum("ijab->jiab", x119)
    t2new_aaaa += einsum("ijab->jiba", x119) * -1
    del x119
    x120 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x120 += einsum("ijab,ikjl->lkba", t2.aaaa, v.aaaa.oooo)
    x122 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x122 += einsum("ijab->jiba", x120)
    del x120
    x121 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x121 += einsum("ijab,cadb->ijcd", t2.aaaa, v.aaaa.vvvv)
    x122 += einsum("ijab->jiba", x121)
    del x121
    t2new_aaaa += einsum("ijab->ijba", x122) * -1
    t2new_aaaa += einsum("ijab->ijab", x122)
    del x122
    x123 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x123 += einsum("ia,bacd->icbd", t1.aa, v.aaaa.vvvv)
    x124 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x124 += einsum("ia,jbca->ijbc", t1.aa, x123)
    del x123
    x132 += einsum("ijab->ijab", x124)
    del x124
    x125 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x125 += einsum("ijab,kbla->ijlk", t2.aaaa, v.aaaa.ovov)
    x126 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x126 += einsum("ijab,klji->lkab", t2.aaaa, x125)
    x132 += einsum("ijab->ijab", x126)
    del x126
    x129 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x129 += einsum("ijkl->lkji", x125)
    del x125
    x129 += einsum("ijkl->kilj", v.aaaa.oooo)
    x130 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x130 += einsum("ia,ijkl->jkla", t1.aa, x129)
    del x129
    x131 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x131 += einsum("ia,ijkb->kjba", t1.aa, x130)
    del x130
    x132 += einsum("ijab->ijba", x131)
    del x131
    t2new_aaaa += einsum("ijab->ijab", x132)
    t2new_aaaa += einsum("ijab->ijba", x132) * -1
    del x132
    x138 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x138 += einsum("iajb->jiab", v.bbbb.ovov)
    x138 -= einsum("iajb->jiba", v.bbbb.ovov)
    x139 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x139 += einsum("ijab,jkbc->ikac", t2.abab, x138)
    x140 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x140 -= einsum("ijab,kjcb->kica", t2.abab, x139)
    del x139
    x141 += einsum("ijab->ijab", x140)
    del x140
    t2new_aaaa += einsum("ijab->ijab", x141)
    t2new_aaaa -= einsum("ijab->ijba", x141)
    del x141
    x148 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x148 += einsum("ijab,ikbc->kjca", t2.bbbb, x138)
    x149 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x149 -= einsum("ijab,ikac->kjcb", t2.bbbb, x148)
    x176 -= einsum("ijab->jiba", x149)
    del x149
    x227 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x227 -= einsum("ijab,ikbc->kjca", t2.bbbb, x148)
    del x148
    x228 += einsum("ijab->ijab", x227)
    del x227
    x153 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x153 += einsum("ia,ijab->jb", t1.bb, x138)
    x154 -= einsum("ia->ia", x153)
    x296 -= einsum("ia->ia", x153)
    del x153
    x225 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x225 += einsum("ijab,ikac->kjcb", t2.bbbb, x138)
    x226 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x226 -= einsum("ijab,ikac->kjcb", t2.bbbb, x225)
    del x225
    x228 += einsum("ijab->jiba", x226)
    del x226
    t2new_bbbb += einsum("ijab->ijab", x228)
    t2new_bbbb -= einsum("ijab->ijba", x228)
    del x228
    x307 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x307 -= einsum("wia,ijab->wjb", u11.bb, x138)
    del x138
    s2new += einsum("wia,xia->wx", u11.bb, x307)
    del x307
    x142 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x142 += einsum("ia,bjca->ijbc", t1.bb, v.bbbb.vovv)
    x176 -= einsum("ijab->ijab", x142)
    del x142
    x143 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x143 += einsum("ijab,iack->jkbc", t2.abab, v.aabb.ovvo)
    x176 += einsum("ijab->ijab", x143)
    del x143
    x144 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x144 += einsum("ia,jbca->jibc", t1.bb, v.aabb.ovvv)
    x145 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x145 += einsum("ijab,ikac->kjbc", t2.abab, x144)
    x176 -= einsum("ijab->ijab", x145)
    del x145
    x232 += einsum("ijab->ijab", x144)
    x267 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x267 += einsum("ijab->ijab", x144)
    x412 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x412 += einsum("ijab->ijab", x144)
    del x144
    x146 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x146 -= einsum("ijab->jiab", t2.bbbb)
    x146 += einsum("ijab->jiba", t2.bbbb)
    x147 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x147 += einsum("ijab,ikcb->jkac", x146, x42)
    del x42
    x176 -= einsum("ijab->ijab", x147)
    del x147
    x152 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x152 += einsum("ijab,klib->jkla", x146, x151)
    del x146
    del x151
    x157 += einsum("ijka->kjia", x152)
    del x152
    x150 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x150 += einsum("ijab,iakl->jklb", t2.abab, v.aabb.ovoo)
    x157 += einsum("ijka->jika", x150)
    del x150
    x154 += einsum("ia->ia", f.bb.ov)
    x155 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x155 += einsum("ia,jkab->ijkb", x154, t2.bbbb)
    x157 += einsum("ijka->ikja", x155)
    del x155
    x159 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x159 += einsum("ia,ja->ji", t1.bb, x154)
    del x154
    x160 += einsum("ij->ji", x159)
    del x159
    x157 -= einsum("ijak->ijka", v.bbbb.oovo)
    x158 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x158 += einsum("ia,ijkb->jkba", t1.bb, x157)
    del x157
    x176 += einsum("ijab->ijba", x158)
    del x158
    x160 += einsum("ij->ji", f.bb.oo)
    x161 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x161 += einsum("ij,jkab->ikab", x160, t2.bbbb)
    del x160
    x176 += einsum("ijab->ijba", x161)
    del x161
    x162 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x162 += einsum("ia,wba->wib", t1.bb, g.bb.bvv)
    x167 += einsum("wia->wia", x162)
    del x162
    x163 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x163 += einsum("wia,ijab->wjb", g.aa.bov, t2.abab)
    x167 += einsum("wia->wia", x163)
    del x163
    x164 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x164 += einsum("ijab->jiab", t2.bbbb)
    x164 -= einsum("ijab->jiba", t2.bbbb)
    x165 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x165 += einsum("wia,ijab->wjb", g.bb.bov, x164)
    x167 -= einsum("wia->wia", x165)
    del x165
    x167 += einsum("wai->wia", g.bb.bvo)
    x168 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x168 += einsum("wia,wjb->jiba", u11.bb, x167)
    x176 += einsum("ijab->ijab", x168)
    del x168
    x291 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x291 += einsum("wia,wjb->ijab", u11.aa, x167)
    del x167
    t2new_baba += einsum("ijab->jiba", x291)
    t2new_abab += einsum("ijab->ijab", x291)
    del x291
    x169 += einsum("ab->ab", f.bb.vv)
    x170 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x170 += einsum("ab,ijbc->ijac", x169, t2.bbbb)
    del x169
    x176 -= einsum("ijab->jiab", x170)
    del x170
    x171 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x171 += einsum("wia,wib->ab", g.bb.bov, u11.bb)
    x172 += einsum("ab->ab", x171)
    x173 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x173 += einsum("ab,ijac->ijbc", x172, t2.bbbb)
    del x172
    x176 -= einsum("ijab->jiba", x173)
    del x173
    t2new_bbbb += einsum("ijab->ijab", x176)
    t2new_bbbb -= einsum("ijab->ijba", x176)
    t2new_bbbb -= einsum("ijab->jiab", x176)
    t2new_bbbb += einsum("ijab->jiba", x176)
    del x176
    x272 += einsum("ab->ba", x171) * -1
    x328 += einsum("ab->ba", x171) * -1
    del x171
    x179 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x179 += einsum("ia,jkla->ijlk", t1.bb, v.bbbb.ooov)
    x180 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x180 += einsum("ijab,kijl->klab", t2.bbbb, x179)
    x209 += einsum("ijab->ijab", x180)
    del x180
    x188 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x188 += einsum("ia,jkil->jkla", t1.bb, x179)
    del x179
    x196 += einsum("ijka->ijka", x188)
    del x188
    x181 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x181 += einsum("ijab,iajc->bc", t2.abab, v.aabb.ovov)
    x182 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x182 += einsum("ab,ijbc->jiac", x181, t2.bbbb)
    x209 += einsum("ijab->ijab", x182) * -1
    del x182
    x272 += einsum("ab->ab", x181) * -1
    x328 += einsum("ab->ab", x181) * -1
    del x181
    x183 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x183 += einsum("iabc->ibac", v.bbbb.ovvv)
    x183 += einsum("iabc->ibca", v.bbbb.ovvv) * -1
    x184 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x184 += einsum("ia,jbac->jibc", t1.bb, x183)
    x185 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x185 += einsum("ijab,ikbc->jkac", x184, x41) * -1
    x209 += einsum("ijab->ijba", x185) * -1
    del x185
    x236 += einsum("ijab->ijab", x184) * -1
    del x184
    x199 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x199 += einsum("ia,ibac->bc", t1.bb, x183)
    del x183
    x200 += einsum("ab->ab", x199) * -1
    del x199
    x201 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x201 += einsum("ab,ijbc->ijac", x200, t2.bbbb)
    del x200
    x209 += einsum("ijab->jiba", x201)
    del x201
    x189 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x189 += einsum("ijab,kbca->jikc", t2.bbbb, v.bbbb.ovvv)
    x196 += einsum("ijka->ikja", x189)
    del x189
    x190 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x190 += einsum("ia,jbca->ijcb", t1.bb, v.bbbb.ovvv)
    x191 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x191 += einsum("ia,jkba->ijkb", t1.bb, x190)
    del x190
    x196 += einsum("ijka->ikja", x191)
    del x191
    x194 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x194 += einsum("ijka->ikja", v.bbbb.ooov) * -1
    x194 += einsum("ijka->kija", v.bbbb.ooov)
    x195 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x195 += einsum("ijka,ilab->jklb", x194, x41)
    x196 += einsum("ijka->kija", x195)
    del x195
    x197 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x197 += einsum("ia,jikb->jkba", t1.bb, x196)
    del x196
    x209 += einsum("ijab->ijba", x197)
    del x197
    x203 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x203 += einsum("ia,ijka->jk", t1.bb, x194)
    del x194
    x204 += einsum("ij->ij", x203) * -1
    del x203
    x205 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x205 += einsum("ij,ikab->jkab", x204, t2.bbbb)
    del x204
    x209 += einsum("ijab->jiba", x205) * -1
    del x205
    t2new_bbbb += einsum("ijab->ijab", x209) * -1
    t2new_bbbb += einsum("ijab->ijba", x209)
    t2new_bbbb += einsum("ijab->jiab", x209)
    t2new_bbbb += einsum("ijab->jiba", x209) * -1
    del x209
    x210 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x210 += einsum("ijab,ikjl->lkba", t2.bbbb, v.bbbb.oooo)
    x212 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x212 += einsum("ijab->jiba", x210)
    del x210
    x211 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x211 += einsum("ijab,cbda->ijdc", t2.bbbb, v.bbbb.vvvv)
    x212 += einsum("ijab->jiba", x211)
    del x211
    t2new_bbbb += einsum("ijab->ijba", x212) * -1
    t2new_bbbb += einsum("ijab->ijab", x212)
    del x212
    x213 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x213 += einsum("ia,bacd->icbd", t1.bb, v.bbbb.vvvv)
    x214 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x214 += einsum("ia,jbca->ijbc", t1.bb, x213)
    del x213
    x222 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x222 += einsum("ijab->ijab", x214)
    del x214
    x215 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x215 += einsum("ijab,kbla->ijlk", t2.bbbb, v.bbbb.ovov)
    x217 += einsum("ijkl->jilk", x215)
    x218 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x218 += einsum("ijab,klji->klab", t2.bbbb, x217)
    del x217
    x222 += einsum("ijab->jiab", x218)
    del x218
    x219 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x219 += einsum("ijkl->lkji", x215)
    del x215
    x219 += einsum("ijkl->kilj", v.bbbb.oooo)
    x220 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x220 += einsum("ia,ijkl->jkla", t1.bb, x219)
    del x219
    x221 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x221 += einsum("ia,ijkb->kjba", t1.bb, x220)
    del x220
    x222 += einsum("ijab->jiab", x221)
    del x221
    t2new_bbbb += einsum("ijab->ijab", x222)
    t2new_bbbb += einsum("ijab->ijba", x222) * -1
    del x222
    x229 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x229 += einsum("ijab,cadb->ijcd", t2.abab, v.aabb.vvvv)
    t2new_baba += einsum("ijab->jiba", x229)
    t2new_abab += einsum("ijab->ijab", x229)
    del x229
    x230 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x230 += einsum("ia,bjca->jibc", t1.bb, v.aabb.vovv)
    t2new_baba += einsum("ijab->jiba", x230)
    t2new_abab += einsum("ijab->ijab", x230)
    del x230
    x232 += einsum("iabj->ijab", v.aabb.ovvo)
    x233 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x233 += einsum("ijab,ikac->kjcb", x232, x63)
    del x232
    del x63
    t2new_baba += einsum("ijab->jiba", x233)
    t2new_abab += einsum("ijab->ijab", x233)
    del x233
    x234 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x234 += einsum("ijab,iakc->jkbc", t2.abab, v.aabb.ovov)
    x236 += einsum("ijab->jiab", x234)
    del x234
    x236 += einsum("iabj->ijba", v.bbbb.ovvo)
    x236 += einsum("ijab->ijab", v.bbbb.oovv) * -1
    x237 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x237 += einsum("ijab,jkcb->ikac", t2.abab, x236)
    del x236
    t2new_baba += einsum("ijab->jiba", x237)
    t2new_abab += einsum("ijab->ijab", x237)
    del x237
    x238 += einsum("iabj->jiba", v.bbaa.ovvo)
    x239 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x239 += einsum("ijab,kicb->kjca", x164, x238)
    t2new_baba += einsum("ijab->jiba", x239)
    t2new_abab += einsum("ijab->ijab", x239)
    del x239
    u12new_aa += einsum("wxia,jiba->xwjb", u12.bb, x238)
    del x238
    x240 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x240 -= einsum("iabc->ibac", v.aaaa.ovvv)
    x240 += einsum("iabc->ibca", v.aaaa.ovvv)
    x241 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x241 += einsum("ia,jbca->jibc", t1.aa, x240)
    x242 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x242 -= einsum("ijab->ijab", x241)
    del x241
    x340 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x340 += einsum("wia,ibca->wbc", u11.aa, x240)
    del x240
    x341 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x341 -= einsum("wab->wba", x340)
    del x340
    x242 += einsum("iabj->ijba", v.aaaa.ovvo)
    x242 -= einsum("ijab->ijab", v.aaaa.oovv)
    x243 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x243 += einsum("ijab,ikca->kjcb", t2.abab, x242)
    t2new_baba += einsum("ijab->jiba", x243)
    t2new_abab += einsum("ijab->ijab", x243)
    del x243
    u12new_aa += einsum("wxia,ijba->xwjb", u12.aa, x242)
    del x242
    x244 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x244 += einsum("ia,jabc->ijbc", t1.aa, v.aabb.ovvv)
    x246 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x246 += einsum("ijab->jiab", x244)
    del x244
    x245 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x245 += einsum("ijab,kajc->ikbc", t2.abab, v.aabb.ovov)
    x246 -= einsum("ijab->jiab", x245)
    del x245
    x246 += einsum("ijab->ijab", v.aabb.oovv)
    x247 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x247 += einsum("ijab,ikcb->kjac", t2.abab, x246)
    del x246
    t2new_baba -= einsum("ijab->jiba", x247)
    t2new_abab -= einsum("ijab->ijab", x247)
    del x247
    x248 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x248 += einsum("ia,jkla->jkil", t1.bb, v.aabb.ooov)
    x252 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x252 += einsum("ijkl->ijlk", x248)
    x265 += einsum("ijkl->ijlk", x248)
    del x248
    x249 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x249 += einsum("ijab,kalb->ikjl", t2.abab, v.aabb.ovov)
    x252 += einsum("ijkl->jilk", x249)
    x265 += einsum("ijkl->jilk", x249)
    del x249
    x250 += einsum("iajk->ijka", v.aabb.ovoo)
    x251 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x251 += einsum("ia,jkla->jikl", t1.aa, x250)
    del x250
    x252 += einsum("ijkl->ijkl", x251)
    del x251
    x252 += einsum("ijkl->ijkl", v.aabb.oooo)
    x253 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x253 += einsum("ijab,ikjl->klab", t2.abab, x252)
    del x252
    t2new_baba += einsum("ijab->jiba", x253)
    t2new_abab += einsum("ijab->ijab", x253)
    del x253
    x254 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x254 += einsum("ia,jabc->ijbc", t1.bb, v.bbaa.ovvv)
    x255 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x255 += einsum("ijab->jiab", x254)
    x285 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x285 += einsum("ijab->jiab", x254)
    del x254
    x255 += einsum("ijab->ijab", v.bbaa.oovv)
    x256 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x256 += einsum("ijab,jkca->ikcb", t2.abab, x255)
    del x255
    t2new_baba -= einsum("ijab->jiba", x256)
    t2new_abab -= einsum("ijab->ijab", x256)
    del x256
    x257 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x257 += einsum("ia,jkba->jkib", t1.bb, v.aabb.oovv)
    x269 += einsum("ijka->ijka", x257)
    del x257
    x258 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x258 += einsum("ijab,kacb->ikjc", t2.abab, v.aabb.ovvv)
    x269 += einsum("ijka->jika", x258)
    del x258
    x259 += einsum("ijka->ikja", v.aaaa.ooov)
    x259 += einsum("ijka->kija", v.aaaa.ooov) * -1
    x260 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x260 += einsum("ijab,ikla->kljb", t2.abab, x259)
    x269 += einsum("ijka->ijka", x260) * -1
    del x260
    x374 += einsum("wxia,ijka->xwjk", u12.aa, x259) * -1
    del x259
    u12new_aa += einsum("ia,wxij->xwja", t1.aa, x374) * -1
    del x374
    x265 += einsum("ijkl->ijkl", v.aabb.oooo)
    x266 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x266 += einsum("ia,jkil->jkla", t1.bb, x265)
    del x265
    x269 += einsum("ijka->ijka", x266) * -1
    del x266
    x267 += einsum("iabj->ijab", v.aabb.ovvo)
    x268 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x268 += einsum("ia,jkab->jikb", t1.aa, x267)
    del x267
    x269 += einsum("ijka->ijka", x268)
    del x268
    x269 += einsum("ijak->ijka", v.aabb.oovo)
    x270 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x270 += einsum("ia,ijkb->jkab", t1.aa, x269)
    del x269
    t2new_baba += einsum("ijab->jiba", x270) * -1
    t2new_abab += einsum("ijab->ijab", x270) * -1
    del x270
    x272 += einsum("ab->ab", f.bb.vv)
    x273 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x273 += einsum("ab,ijcb->ijca", x272, t2.abab)
    t2new_baba += einsum("ijab->jiba", x273)
    t2new_abab += einsum("ijab->ijab", x273)
    del x273
    u12new_bb += einsum("ab,wxib->xwia", x272, u12.bb)
    del x272
    x274 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x274 += einsum("iajb->jiab", v.aaaa.ovov) * -1
    x274 += einsum("iajb->jiba", v.aaaa.ovov)
    x275 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x275 += einsum("ijab,ijcb->ca", t2.aaaa, x274)
    del x274
    x276 += einsum("ab->ba", x275) * -1
    x317 += einsum("ab->ba", x275) * -1
    del x275
    x276 += einsum("ab->ab", f.aa.vv)
    x277 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x277 += einsum("ab,ijbc->ijac", x276, t2.abab)
    t2new_baba += einsum("ijab->jiba", x277)
    t2new_abab += einsum("ijab->ijab", x277)
    del x277
    u12new_aa += einsum("ab,wxib->xwia", x276, u12.aa)
    del x276
    x278 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x278 += einsum("ia,jabk->kijb", t1.bb, v.bbaa.ovvo)
    x287 += einsum("ijka->ikja", x278)
    del x278
    x279 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x279 += einsum("ijab,kbca->ijkc", t2.abab, v.bbaa.ovvv)
    x287 += einsum("ijka->ikja", x279)
    del x279
    x280 += einsum("ijka->ikja", v.bbbb.ooov)
    x280 += einsum("ijka->kija", v.bbbb.ooov) * -1
    x281 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x281 += einsum("ijab,jklb->ikla", t2.abab, x280)
    x287 += einsum("ijka->ijka", x281) * -1
    del x281
    x413 += einsum("wxia,ijka->xwjk", u12.bb, x280) * -1
    del x280
    u12new_bb += einsum("ia,wxij->xwja", t1.bb, x413) * -1
    del x413
    x285 += einsum("ijab->ijab", v.bbaa.oovv)
    x286 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x286 += einsum("ia,jkba->ijkb", t1.aa, x285)
    del x285
    x287 += einsum("ijka->ijka", x286)
    del x286
    x287 += einsum("ijak->kija", v.bbaa.oovo)
    x288 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x288 += einsum("ia,jikb->jkba", t1.bb, x287)
    del x287
    t2new_baba += einsum("ijab->jiba", x288) * -1
    t2new_abab += einsum("ijab->ijab", x288) * -1
    del x288
    x293 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x293 += einsum("ia,bcda->ibcd", t1.bb, v.aabb.vvvv)
    x294 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x294 += einsum("iabc->iabc", x293)
    del x293
    x294 += einsum("abci->iabc", v.aabb.vvvo)
    x295 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x295 += einsum("ia,jbac->ijbc", t1.aa, x294)
    del x294
    t2new_baba += einsum("ijab->jiba", x295)
    t2new_abab += einsum("ijab->ijab", x295)
    del x295
    x296 += einsum("ia->ia", f.bb.ov)
    s1new += einsum("ia,wia->w", x296, u11.bb)
    s2new += einsum("ia,wxia->xw", x296, u12.bb)
    del x296
    x297 += einsum("ia->ia", f.aa.ov)
    s1new += einsum("ia,wia->w", x297, u11.aa)
    s2new += einsum("ia,wxia->xw", x297, u12.aa)
    del x297
    x298 = np.zeros((nbos, nbos), dtype=types[float])
    x298 += einsum("wia,xia->wx", gc.aa.bov, u11.aa)
    x306 = np.zeros((nbos, nbos), dtype=types[float])
    x306 += einsum("wx->wx", x298)
    del x298
    x299 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x299 += einsum("wia,iajb->wjb", u11.aa, v.aabb.ovov)
    x300 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x300 += einsum("wia->wia", x299)
    x315 += einsum("wia->wia", x299)
    x400 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x400 += einsum("ia,wja->wij", t1.bb, x299)
    x402 += einsum("wij->wij", x400)
    del x400
    x403 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x403 += einsum("wia,xji->xwja", u11.bb, x402)
    del x402
    x410 += einsum("wxia->wxia", x403)
    del x403
    x404 = np.zeros((nbos, nbos, nocc[1], nocc[1]), dtype=types[float])
    x404 += einsum("wia,xja->xwij", u11.bb, x299)
    del x299
    x406 += einsum("wxij->wxij", x404)
    del x404
    x407 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x407 += einsum("ia,wxji->wxja", t1.bb, x406)
    del x406
    x410 += einsum("wxia->wxia", x407)
    del x407
    u12new_bb += einsum("wxia->wxia", x410) * -1
    u12new_bb += einsum("wxia->xwia", x410) * -1
    del x410
    x300 += einsum("wia->wia", gc.bb.bov)
    x301 = np.zeros((nbos, nbos), dtype=types[float])
    x301 += einsum("wia,xia->xw", u11.bb, x300)
    del x300
    x306 += einsum("wx->wx", x301)
    del x301
    x302 = np.zeros((nbos, nbos), dtype=types[float])
    x302 += einsum("wia,xia->wx", g.aa.bov, u11.aa)
    x304 = np.zeros((nbos, nbos), dtype=types[float])
    x304 += einsum("wx->wx", x302)
    x357 = np.zeros((nbos, nbos), dtype=types[float])
    x357 += einsum("wx->wx", x302)
    del x302
    x303 = np.zeros((nbos, nbos), dtype=types[float])
    x303 += einsum("wia,xia->wx", g.bb.bov, u11.bb)
    x304 += einsum("wx->wx", x303)
    x357 += einsum("wx->wx", x303)
    del x303
    x358 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x358 += einsum("wx,wyia->yxia", x357, u12.aa)
    x359 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x359 -= einsum("wxia->xwia", x358)
    del x358
    x398 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x398 += einsum("wx,wyia->yxia", x357, u12.bb)
    del x357
    x399 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x399 -= einsum("wxia->xwia", x398)
    del x398
    x304 += einsum("wx->wx", w)
    x305 = np.zeros((nbos, nbos), dtype=types[float])
    x305 += einsum("wx,wy->xy", s2, x304)
    x306 += einsum("wx->wx", x305)
    del x305
    s2new += einsum("wx->wx", x306)
    s2new += einsum("wx->xw", x306)
    del x306
    u11new_aa += einsum("wx,wia->xia", x304, u11.aa)
    u11new_bb += einsum("wx,wia->xia", x304, u11.bb)
    del x304
    x309 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x309 += einsum("wx,xia->wia", s2, g.aa.bov)
    x312 += einsum("wia->wia", x309)
    x345 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x345 += einsum("wia->wia", x309)
    del x309
    x310 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x310 += einsum("wia,jbia->wjb", u11.bb, v.aabb.ovov)
    x312 += einsum("wia->wia", x310)
    x360 += einsum("wia->wia", x310)
    del x310
    x361 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x361 += einsum("ia,wib->wba", t1.aa, x360)
    x362 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x362 += einsum("wia,xab->xwib", u11.aa, x361)
    del x361
    x367 += einsum("wxia->xwia", x362)
    del x362
    x363 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x363 += einsum("ia,wja->wji", t1.aa, x360)
    del x360
    x364 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x364 += einsum("wia,xij->xwja", u11.aa, x363)
    del x363
    x367 += einsum("wxia->xwia", x364)
    del x364
    u12new_aa += einsum("wxia->wxia", x367) * -1
    u12new_aa += einsum("wxia->xwia", x367) * -1
    del x367
    x312 += einsum("wia->wia", gc.aa.bov)
    x322 += einsum("ia,wja->wji", t1.aa, x312)
    u11new_aa += einsum("wia,ijab->wjb", x312, x10) * -1
    u11new_bb += einsum("wia,ijab->wjb", x312, t2.abab)
    del x312
    x313 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x313 += einsum("wx,xia->wia", s2, g.bb.bov)
    x315 += einsum("wia->wia", x313)
    x381 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x381 += einsum("wia->wia", x313)
    del x313
    x315 += einsum("wia->wia", gc.bb.bov)
    x333 += einsum("ia,wja->wji", t1.bb, x315)
    u11new_aa += einsum("wia,jiba->wjb", x315, t2.abab)
    u11new_bb += einsum("wia,ijab->wjb", x315, x164) * -1
    del x315
    del x164
    x317 += einsum("ab->ab", f.aa.vv)
    u11new_aa += einsum("ab,wib->wia", x317, u11.aa)
    del x317
    x318 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x318 += einsum("wx,xij->wij", s2, g.aa.boo)
    x322 += einsum("wij->wij", x318)
    x347 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x347 += einsum("wij->wij", x318)
    del x318
    x319 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x319 += einsum("wia,xwja->xij", g.aa.bov, u12.aa)
    x322 += einsum("wij->wij", x319)
    x351 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x351 += einsum("wij->wij", x319)
    del x319
    x320 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x320 += einsum("wia,jkia->wjk", u11.bb, v.aabb.ooov)
    x322 += einsum("wij->wij", x320)
    x351 += einsum("wij->wij", x320)
    del x320
    x321 += einsum("ia->ia", f.aa.ov)
    x322 += einsum("ia,wja->wij", x321, u11.aa)
    del x321
    x322 += einsum("wij->wij", gc.aa.boo)
    u11new_aa += einsum("ia,wij->wja", t1.aa, x322) * -1
    del x322
    x323 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x323 += einsum("wx,xab->wab", s2, g.aa.bvv)
    x326 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x326 += einsum("wab->wab", x323)
    x353 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x353 += einsum("wab->wab", x323)
    del x323
    x324 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x324 += einsum("wia,iabc->wbc", u11.bb, v.bbaa.ovvv)
    x326 += einsum("wab->wab", x324)
    x341 -= einsum("wab->wba", x324)
    del x324
    x325 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x325 += einsum("iabc->ibac", v.aaaa.ovvv)
    x325 -= einsum("iabc->ibca", v.aaaa.ovvv)
    x326 -= einsum("wia,ibca->wbc", u11.aa, x325)
    del x325
    x326 += einsum("wab->wab", gc.aa.bvv)
    u11new_aa += einsum("ia,wba->wib", t1.aa, x326)
    del x326
    x328 += einsum("ab->ab", f.bb.vv)
    u11new_bb += einsum("ab,wib->wia", x328, u11.bb)
    del x328
    x329 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x329 += einsum("wx,xij->wij", s2, g.bb.boo)
    x333 += einsum("wij->wij", x329)
    x383 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x383 += einsum("wij->wij", x329)
    del x329
    x330 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x330 += einsum("wia,xwja->xij", g.bb.bov, u12.bb)
    x333 += einsum("wij->wij", x330)
    x394 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x394 += einsum("wij->wij", x330)
    del x330
    x331 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x331 += einsum("wia,iajk->wjk", u11.aa, v.aabb.ovoo)
    x333 += einsum("wij->wij", x331)
    x383 += einsum("wij->wij", x331)
    del x331
    x332 += einsum("ia->ia", f.bb.ov)
    x333 += einsum("ia,wja->wij", x332, u11.bb)
    del x332
    x333 += einsum("wij->wij", gc.bb.boo)
    u11new_bb += einsum("ia,wij->wja", t1.bb, x333) * -1
    del x333
    x334 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x334 += einsum("wx,xab->wab", s2, g.bb.bvv)
    x337 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x337 += einsum("wab->wab", x334)
    x385 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x385 += einsum("wab->wab", x334)
    del x334
    x335 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x335 += einsum("wia,iabc->wbc", u11.aa, v.aabb.ovvv)
    x337 += einsum("wab->wab", x335)
    x385 += einsum("wab->wab", x335)
    del x335
    x336 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x336 -= einsum("iabc->ibac", v.bbbb.ovvv)
    x336 += einsum("iabc->ibca", v.bbbb.ovvv)
    x337 -= einsum("wia,ibac->wbc", u11.bb, x336)
    del x336
    x337 += einsum("wab->wab", gc.bb.bvv)
    u11new_bb += einsum("ia,wba->wib", t1.bb, x337)
    del x337
    x338 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x338 += einsum("wx,ywia->yxia", w, u12.aa)
    x359 -= einsum("wxia->wxia", x338)
    del x338
    x339 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x339 += einsum("wia,xwib->xab", g.aa.bov, u12.aa)
    x341 += einsum("wab->wab", x339)
    del x339
    x342 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x342 += einsum("wia,xab->xwib", u11.aa, x341)
    del x341
    x359 += einsum("wxia->xwia", x342)
    del x342
    x343 += einsum("ia->ia", f.aa.ov)
    x344 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x344 += einsum("ia,wja->wij", x343, u11.aa)
    del x343
    x347 += einsum("wij->wij", x344)
    del x344
    x345 += einsum("wia->wia", gc.aa.bov)
    x346 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x346 += einsum("ia,wja->wji", t1.aa, x345)
    x347 += einsum("wij->wij", x346)
    del x346
    x355 = np.zeros((nbos, nbos, nocc[0], nocc[0]), dtype=types[float])
    x355 += einsum("wia,xja->xwji", u11.aa, x345)
    del x345
    x356 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x356 += einsum("ia,wxij->wxja", t1.aa, x355)
    del x355
    x359 += einsum("wxia->wxia", x356)
    del x356
    x347 += einsum("wij->wij", gc.aa.boo)
    x348 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x348 += einsum("wia,xij->xwja", u11.aa, x347)
    del x347
    x359 += einsum("wxia->wxia", x348)
    del x348
    x349 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x349 += einsum("ijka->ikja", v.aaaa.ooov)
    x349 -= einsum("ijka->kija", v.aaaa.ooov)
    x350 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x350 += einsum("wia,ijka->wjk", u11.aa, x349)
    del x349
    x351 -= einsum("wij->wij", x350)
    del x350
    x352 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x352 += einsum("wia,xij->xwja", u11.aa, x351)
    del x351
    x359 += einsum("wxia->xwia", x352)
    del x352
    x353 += einsum("wab->wab", gc.aa.bvv)
    x354 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x354 += einsum("wia,xba->xwib", u11.aa, x353)
    del x353
    x359 -= einsum("wxia->wxia", x354)
    del x354
    u12new_aa -= einsum("wxia->wxia", x359)
    u12new_aa -= einsum("wxia->xwia", x359)
    del x359
    x368 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x368 += einsum("wxia,jbia->wxjb", u12.bb, v.aabb.ovov)
    x370 += einsum("wxia->xwia", x368)
    del x368
    u12new_aa += einsum("ijab,wxia->xwjb", x10, x370) * -1
    del x10
    u12new_bb += einsum("ijab,wxia->xwjb", t2.abab, x370)
    del x370
    x371 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x371 += einsum("wxia,iajb->wxjb", u12.aa, v.aabb.ovov)
    x373 += einsum("wxia->xwia", x371)
    del x371
    u12new_aa += einsum("ijab,wxjb->xwia", t2.abab, x373)
    u12new_bb += einsum("wxia,ijba->xwjb", x373, x41)
    del x41
    del x373
    x375 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x375 += einsum("wia,xyia->wxy", g.aa.bov, u12.aa)
    x377 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x377 += einsum("wxy->wyx", x375)
    del x375
    x376 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x376 += einsum("wia,xyia->wxy", g.bb.bov, u12.bb)
    x377 += einsum("wxy->wyx", x376)
    del x376
    u12new_aa += einsum("wia,wxy->yxia", u11.aa, x377)
    u12new_bb += einsum("wia,wxy->yxia", u11.bb, x377)
    del x377
    x378 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x378 += einsum("wx,ywia->yxia", w, u12.bb)
    x399 -= einsum("wxia->wxia", x378)
    del x378
    x379 += einsum("ia->ia", f.bb.ov)
    x380 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x380 += einsum("ia,wja->wij", x379, u11.bb)
    del x379
    x383 += einsum("wij->wij", x380)
    del x380
    x381 += einsum("wia->wia", gc.bb.bov)
    x382 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x382 += einsum("ia,wja->wji", t1.bb, x381)
    x383 += einsum("wij->wij", x382)
    del x382
    x396 = np.zeros((nbos, nbos, nocc[1], nocc[1]), dtype=types[float])
    x396 += einsum("wia,xja->xwji", u11.bb, x381)
    del x381
    x397 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x397 += einsum("ia,wxij->wxja", t1.bb, x396)
    del x396
    x399 += einsum("wxia->wxia", x397)
    del x397
    x383 += einsum("wij->wij", gc.bb.boo)
    x384 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x384 += einsum("wia,xij->xwja", u11.bb, x383)
    del x383
    x399 += einsum("wxia->wxia", x384)
    del x384
    x385 += einsum("wab->wab", gc.bb.bvv)
    x386 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x386 += einsum("wia,xba->xwib", u11.bb, x385)
    del x385
    x399 -= einsum("wxia->wxia", x386)
    del x386
    x387 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x387 += einsum("wia,xwib->xab", g.bb.bov, u12.bb)
    x390 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x390 += einsum("wab->wab", x387)
    del x387
    x388 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x388 += einsum("iabc->ibac", v.bbbb.ovvv)
    x388 -= einsum("iabc->ibca", v.bbbb.ovvv)
    x389 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x389 += einsum("wia,ibac->wbc", u11.bb, x388)
    x390 -= einsum("wab->wba", x389)
    del x389
    x391 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x391 += einsum("wia,xab->xwib", u11.bb, x390)
    del x390
    x399 += einsum("wxia->xwia", x391)
    del x391
    x411 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x411 -= einsum("ia,jbac->jibc", t1.bb, x388)
    del x388
    x392 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x392 += einsum("ijka->ikja", v.bbbb.ooov)
    x392 -= einsum("ijka->kija", v.bbbb.ooov)
    x393 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x393 += einsum("wia,ijka->wjk", u11.bb, x392)
    del x392
    x394 -= einsum("wij->wij", x393)
    del x393
    x395 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x395 += einsum("wia,xij->xwja", u11.bb, x394)
    del x394
    x399 += einsum("wxia->xwia", x395)
    del x395
    u12new_bb -= einsum("wxia->wxia", x399)
    u12new_bb -= einsum("wxia->xwia", x399)
    del x399
    x411 += einsum("iabj->ijba", v.bbbb.ovvo)
    x411 -= einsum("ijab->ijab", v.bbbb.oovv)
    u12new_bb += einsum("wxia,ijba->xwjb", u12.bb, x411)
    del x411
    x412 += einsum("iabj->ijab", v.aabb.ovvo)
    u12new_bb += einsum("wxia,ijab->xwjb", u12.aa, x412)
    del x412
    t1new_aa += einsum("w,wai->ia", s1, g.aa.bvo)
    t1new_aa += einsum("ijab,jbca->ic", t2.abab, v.bbaa.ovvv)
    t1new_aa += einsum("wab,wib->ia", g.aa.bvv, u11.aa)
    t1new_aa += einsum("ai->ia", f.aa.vo)
    t1new_aa += einsum("ia,iabj->jb", t1.bb, v.bbaa.ovvo)
    t1new_bb += einsum("w,wai->ia", s1, g.bb.bvo)
    t1new_bb += einsum("wab,wib->ia", g.bb.bvv, u11.bb)
    t1new_bb += einsum("ia,iabj->jb", t1.aa, v.aabb.ovvo)
    t1new_bb += einsum("ijab,iacb->jc", t2.abab, v.aabb.ovvv)
    t1new_bb += einsum("ai->ia", f.bb.vo)
    t2new_aaaa -= einsum("aibj->jiab", v.aaaa.vovo)
    t2new_aaaa += einsum("aibj->jiba", v.aaaa.vovo)
    t2new_bbbb -= einsum("aibj->jiab", v.bbbb.vovo)
    t2new_bbbb += einsum("aibj->jiba", v.bbbb.vovo)
    t2new_baba += einsum("aibj->jiba", v.aabb.vovo)
    t2new_abab += einsum("aibj->ijab", v.aabb.vovo)
    s1new += einsum("wia,xwia->x", g.aa.bov, u12.aa)
    s1new += einsum("w,wx->x", s1, w)
    s1new += einsum("ia,wia->w", t1.aa, gc.aa.bov)
    s1new += einsum("w->w", G)
    s1new += einsum("ia,wia->w", t1.bb, gc.bb.bov)
    s1new += einsum("wia,xwia->x", g.bb.bov, u12.bb)
    u11new_aa += einsum("wx,xai->wia", s2, g.aa.bvo)
    u11new_aa += einsum("wab,xwib->xia", g.aa.bvv, u12.aa)
    u11new_aa += einsum("wai->wia", gc.aa.bvo)
    u11new_aa += einsum("wia,iabj->wjb", u11.bb, v.bbaa.ovvo)
    u11new_bb += einsum("wx,xai->wia", s2, g.bb.bvo)
    u11new_bb += einsum("wab,xwib->xia", g.bb.bvv, u12.bb)
    u11new_bb += einsum("wia,iabj->wjb", u11.aa, v.aabb.ovvo)
    u11new_bb += einsum("wai->wia", gc.bb.bvo)

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.abab = t2new_abab
    t2new.baba = t2new_baba
    t2new.aaaa = t2new_aaaa
    t2new.bbbb = t2new_bbbb
    u11new.aa = u11new_aa
    u11new.bb = u11new_bb
    u12new.aa = u12new_aa
    u12new.bb = u12new_bb

    return {"t1new": t1new, "t2new": t2new, "s1new": s1new, "s2new": s2new, "u11new": u11new, "u12new": u12new}

def update_lams(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, lu12=None, **kwargs):
    l1new = Namespace()
    l2new = Namespace()
    lu11new = Namespace()
    lu12new = Namespace()

    # Get boson coupling creation array:
    gc = Namespace(
        aa = Namespace(
            boo=g.aa.boo.transpose(0, 2, 1),
            bov=g.aa.bvo.transpose(0, 2, 1),
            bvo=g.aa.bov.transpose(0, 2, 1),
            bvv=g.aa.bvv.transpose(0, 2, 1),
        ),
        bb = Namespace(
            boo=g.bb.boo.transpose(0, 2, 1),
            bov=g.bb.bvo.transpose(0, 2, 1),
            bvo=g.bb.bov.transpose(0, 2, 1),
            bvv=g.bb.bvv.transpose(0, 2, 1),
        ),
    )

    # L1, L2, LS1, LS2, LU11 and LU12 amplitudes
    x0 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x0 += einsum("abij,klab->ikjl", l2.abab, t2.abab)
    x61 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x61 += einsum("ijkl->ijkl", x0)
    x531 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x531 += einsum("ijkl->ijkl", x0)
    l1new_aa = np.zeros((nvir[0], nocc[0]), dtype=types[float])
    l1new_aa += einsum("iajk,likj->al", v.aabb.ovoo, x0)
    l1new_bb = np.zeros((nvir[1], nocc[1]), dtype=types[float])
    l1new_bb += einsum("ijka,jilk->al", v.aabb.ooov, x0)
    del x0
    x1 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x1 += einsum("ia,bajk->jkib", t1.bb, l2.abab)
    x2 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x2 += einsum("ia,jkla->jikl", t1.aa, x1)
    x61 += einsum("ijkl->ijkl", x2)
    x62 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x62 += einsum("ia,jkil->jkla", t1.bb, x61)
    x301 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x301 += einsum("ia,ijkl->jkla", t1.aa, x61)
    del x61
    x531 += einsum("ijkl->ijkl", x2)
    x532 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x532 += einsum("iajb,kilj->klab", v.aabb.ovov, x531)
    del x531
    l2new_baba = np.zeros((nvir[1], nvir[0], nocc[1], nocc[0]), dtype=types[float])
    l2new_baba += einsum("ijab->baji", x532)
    l2new_abab = np.zeros((nvir[0], nvir[1], nocc[0], nocc[1]), dtype=types[float])
    l2new_abab += einsum("ijab->abij", x532)
    del x532
    l1new_aa += einsum("iajk,likj->al", v.aabb.ovoo, x2)
    l1new_bb += einsum("ijka,jilk->al", v.aabb.ooov, x2)
    del x2
    x53 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x53 += einsum("ia,jikb->jkba", t1.bb, x1)
    x62 += einsum("ijab,kjla->kilb", t2.abab, x1)
    x192 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x192 += einsum("ijab,ijka->kb", t2.abab, x1)
    x207 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x207 += einsum("ia->ia", x192)
    del x192
    x299 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x299 += einsum("ijab,ikla->kljb", t2.abab, x1)
    x353 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x353 += einsum("iajk,lkjb->liba", v.aabb.ovoo, x1)
    x390 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x390 -= einsum("ijab->ijab", x353)
    del x353
    x503 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x503 += einsum("iabc,jkib->jkca", v.bbaa.ovvv, x1)
    l2new_baba -= einsum("ijab->baji", x503)
    l2new_abab -= einsum("ijab->abij", x503)
    del x503
    l1new_aa -= einsum("ijab,kjia->bk", v.bbaa.oovv, x1)
    x3 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x3 += einsum("abij,kjac->ikbc", l2.abab, t2.abab)
    l1new_aa += einsum("iabc,jibc->aj", v.aabb.ovvv, x3) * -1
    del x3
    x4 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x4 += einsum("w,wia->ia", s1, g.aa.bov)
    x5 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x5 += einsum("ia->ia", x4)
    x41 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x41 += einsum("ia->ia", x4)
    x372 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x372 += einsum("ia->ia", x4)
    x564 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x564 += einsum("ia->ia", x4)
    del x4
    x5 += einsum("ia->ia", f.aa.ov)
    x6 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum("ia,jkab->jkib", x5, t2.aaaa)
    del x5
    x7 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x7 += einsum("ijka->kjia", x6)
    del x6
    x7 += einsum("ijak->ijka", v.aaaa.oovo) * -1
    x32 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x32 += einsum("ijka->kija", x7)
    x32 += einsum("ijka->jika", x7) * -1
    del x7
    x8 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum("ijab,kbca->jikc", t2.aaaa, v.aaaa.ovvv)
    x21 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x21 += einsum("ijka->ijka", x8) * -1
    del x8
    x9 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x9 += einsum("ia,jbca->ijcb", t1.aa, v.aaaa.ovvv)
    x10 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x10 += einsum("ia,jkba->ijkb", t1.aa, x9)
    del x9
    x21 += einsum("ijka->ijka", x10) * -1
    del x10
    x11 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x11 += einsum("ia,jbia->jb", t1.bb, v.aabb.ovov)
    x14 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x14 += einsum("ia->ia", x11)
    x41 += einsum("ia->ia", x11)
    x389 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x389 += einsum("ia->ia", x11)
    x564 += einsum("ia->ia", x11)
    del x11
    x12 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum("iajb->jiab", v.aaaa.ovov)
    x12 += einsum("iajb->jiba", v.aaaa.ovov) * -1
    x13 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x13 += einsum("ia,ijab->jb", t1.aa, x12)
    x14 += einsum("ia->ia", x13) * -1
    x15 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x15 += einsum("ia,jkab->jkib", x14, t2.aaaa)
    x21 += einsum("ijka->jika", x15)
    del x15
    x404 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x404 += einsum("ia,ja->ij", t1.aa, x14)
    x405 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x405 += einsum("ij->ij", x404)
    del x404
    x41 += einsum("ia->ia", x13) * -1
    del x13
    x75 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x75 += einsum("wia,ijab->wjb", u11.aa, x12)
    x76 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x76 += einsum("wia->wia", x75) * -1
    x220 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x220 += einsum("wia->wia", x75) * -1
    x229 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x229 += einsum("wia->wia", x75) * -1
    del x75
    x399 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x399 += einsum("wxia,ijab->wxjb", u12.aa, x12)
    x400 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x400 += einsum("wxia->xwia", x399) * -1
    del x399
    x509 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x509 += einsum("ijab,ikac->kjcb", t2.abab, x12)
    x511 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x511 += einsum("ijab->ijab", x509) * -1
    del x509
    x16 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x16 += einsum("ijab,kalb->ijkl", t2.aaaa, v.aaaa.ovov)
    x19 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x19 += einsum("ijkl->jilk", x16)
    x418 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x418 += einsum("ijkl->jilk", x16)
    del x16
    x17 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x17 += einsum("ia,jbka->ijkb", t1.aa, v.aaaa.ovov)
    x18 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x18 += einsum("ia,jkla->ijkl", t1.aa, x17)
    x19 += einsum("ijkl->ijkl", x18)
    x20 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x20 += einsum("ia,jkil->jkla", t1.aa, x19)
    del x19
    x21 += einsum("ijka->jika", x20)
    del x20
    x32 += einsum("ijka->jkia", x21)
    x32 += einsum("ijka->ikja", x21) * -1
    del x21
    x418 += einsum("ijkl->ijkl", x18)
    del x18
    x419 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x419 += einsum("abij,ijkl->klab", l2.aaaa, x418)
    del x418
    x422 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x422 += einsum("ijab->jiba", x419)
    del x419
    x22 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x22 += einsum("ijka->ikja", x17) * -1
    x22 += einsum("ijka->ijka", x17)
    x37 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x37 += einsum("ijka->jkia", x17)
    x37 += einsum("ijka->kjia", x17) * -1
    x86 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x86 += einsum("ijka->jkia", x17) * -1
    x86 += einsum("ijka->kjia", x17)
    x386 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x386 += einsum("ijka->ijka", x17)
    x395 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x395 += einsum("ijka->ikja", x17)
    x395 += einsum("ijka->ijka", x17) * -1
    x535 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x535 += einsum("ijka->ikja", x17) * -1
    x535 += einsum("ijka->ijka", x17)
    del x17
    x22 += einsum("ijka->jkia", v.aaaa.ooov)
    x22 += einsum("ijka->jika", v.aaaa.ooov) * -1
    x23 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x23 += einsum("ijab->jiab", t2.aaaa) * -1
    x23 += einsum("ijab->jiba", t2.aaaa)
    x24 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x24 += einsum("ijka,klba->ijlb", x22, x23)
    del x22
    x32 += einsum("ijka->ijka", x24)
    x51 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x51 += einsum("ijka->ijka", x24)
    del x24
    x85 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x85 += einsum("iabc,ijca->jb", v.aaaa.ovvv, x23)
    x121 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x121 += einsum("ia->ia", x85) * -1
    del x85
    x105 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x105 += einsum("iajb,ikba->jk", v.aaaa.ovov, x23)
    x109 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x109 += einsum("ij->ij", x105) * -1
    x405 += einsum("ij->ji", x105) * -1
    del x105
    x166 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x166 += einsum("ai,ijab->jb", l1.aa, x23)
    x178 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x178 += einsum("ia->ia", x166) * -1
    del x166
    x175 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x175 += einsum("abij,ikba->jk", l2.aaaa, x23)
    x176 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x176 += einsum("ij->ij", x175) * -1
    x411 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x411 += einsum("ij->ij", x175) * -1
    del x175
    x184 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x184 += einsum("abij,ijbc->ac", l2.aaaa, x23)
    x185 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x185 += einsum("ab->ab", x184) * -1
    x402 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x402 += einsum("ab->ab", x184) * -1
    x548 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x548 += einsum("ab->ab", x184) * -1
    del x184
    x214 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x214 += einsum("abij,ikba->jk", l2.aaaa, x23) * 2
    x215 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x215 += einsum("ij->ij", x214) * -1
    del x214
    x297 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x297 += einsum("abij,ikac->kjcb", l2.abab, x23) * -2
    x301 += einsum("ijka,ilab->ljkb", x1, x23) * -1
    x324 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x324 += einsum("abij,ijbc->ac", l2.aaaa, x23) * 2
    x325 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x325 += einsum("ab->ab", x324) * -1
    x567 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x567 += einsum("ab->ab", x324) * -1
    del x324
    x408 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x408 += einsum("iajb,ijbc->ac", v.aaaa.ovov, x23)
    x409 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x409 += einsum("ab->ba", x408) * -1
    del x408
    x25 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x25 += einsum("ia,jakb->ijkb", t1.aa, v.aabb.ovov)
    x26 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x26 += einsum("ijka->jika", x25)
    x559 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x559 += einsum("ijka->jika", x25)
    x26 += einsum("ijka->ijka", v.aabb.ooov)
    x27 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x27 += einsum("ijab,kljb->ikla", t2.abab, x26)
    x32 += einsum("ijka->kjia", x27)
    x51 += einsum("ijka->kjia", x27)
    del x27
    x77 = np.zeros((nbos, nbos, nocc[0], nocc[0]), dtype=types[float])
    x77 += einsum("wxia,jkia->xwjk", u12.bb, x26)
    x88 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x88 += einsum("ijab,ikjb->ka", t2.abab, x26)
    x121 += einsum("ia->ia", x88) * -1
    del x88
    x293 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x293 += einsum("ijab,iklb->klja", t2.abab, x26) * -1
    x534 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x534 += einsum("ijka,likb->ljab", x1, x26)
    l2new_baba += einsum("ijab->baji", x534)
    l2new_abab += einsum("ijab->abij", x534)
    del x534
    x28 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x28 += einsum("iabj->ijba", v.aaaa.ovvo)
    x28 += einsum("ijab->ijab", v.aaaa.oovv) * -1
    x29 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x29 += einsum("ia,jkba->ijkb", t1.aa, x28)
    x32 += einsum("ijka->ijka", x29)
    x51 += einsum("ijka->ijka", x29)
    del x29
    x97 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x97 += einsum("ia,ijba->jb", t1.aa, x28)
    x121 += einsum("ia->ia", x97)
    del x97
    x575 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x575 += einsum("wia,ijba->wjb", u11.aa, x28) * -1
    x30 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x30 += einsum("ia,jkla->ijlk", t1.aa, v.aaaa.ooov)
    x31 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x31 += einsum("ijkl->jkil", x30)
    x31 += einsum("ijkl->kjil", x30) * -1
    x50 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x50 += einsum("ijkl->ikjl", x30) * -1
    x50 += einsum("ijkl->ijkl", x30)
    x51 += einsum("ia,jikl->jkla", t1.aa, x50) * -1
    del x50
    l1new_aa += einsum("abij,ikjb->ak", l2.aaaa, x51) * -1
    del x51
    x351 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x351 += einsum("abij,jkli->klba", l2.aaaa, x30)
    del x30
    x390 -= einsum("ijab->ijab", x351)
    del x351
    x31 += einsum("ijkl->kijl", v.aaaa.oooo)
    x31 += einsum("ijkl->kilj", v.aaaa.oooo) * -1
    x32 += einsum("ia,ijkl->kjla", t1.aa, x31) * -1
    del x31
    l1new_aa += einsum("abij,ikja->bk", l2.aaaa, x32)
    del x32
    x33 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x33 += einsum("abij,kjcb->ikac", l2.abab, t2.abab)
    x35 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x35 += einsum("ijab->ijab", x33)
    x34 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x34 += einsum("abij->jiab", l2.aaaa) * -1
    x34 += einsum("abij->jiba", l2.aaaa)
    x35 += einsum("ijab,ikac->kjcb", x23, x34)
    x53 += einsum("ijab,ikac->kjcb", t2.abab, x34) * -1
    x35 += einsum("wxai,wxjb->ijab", lu12.aa, u12.aa) * 0.5
    x36 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x36 += einsum("iabc->ibac", v.aaaa.ovvv)
    x36 += einsum("iabc->ibca", v.aaaa.ovvv) * -1
    x512 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x512 += einsum("ia,jbac->ijbc", t1.aa, x36)
    x513 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x513 += einsum("ijab->jiab", x512) * -1
    del x512
    l1new_aa += einsum("ijab,jacb->ci", x35, x36) * -1
    del x35
    x37 += einsum("ijka->ikja", v.aaaa.ooov) * -1
    x37 += einsum("ijka->kija", v.aaaa.ooov)
    x49 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x49 += einsum("ijab,kila->kljb", t2.abab, x37) * -1
    x77 += einsum("wxia,jika->xwjk", u12.aa, x37) * -1
    del x37
    x38 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x38 += einsum("ijab->jiab", t2.bbbb) * -1
    x38 += einsum("ijab->jiba", t2.bbbb)
    x49 += einsum("ijka,klba->ijlb", x26, x38) * -1
    x204 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x204 += einsum("abij,ikba->jk", l2.bbbb, x38)
    x205 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x205 += einsum("ij->ij", x204) * -1
    x488 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x488 += einsum("ij->ij", x204) * -1
    x576 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x576 += einsum("ij->ij", x204) * -1
    del x204
    x212 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x212 += einsum("abij,ijbc->ac", l2.bbbb, x38)
    x213 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x213 += einsum("ab->ab", x212) * -1
    x478 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x478 += einsum("ab->ab", x212) * -1
    x547 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x547 += einsum("ab->ab", x212) * -1
    del x212
    x224 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x224 += einsum("abij,ikba->jk", l2.bbbb, x38) * 2
    x225 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x225 += einsum("ij->ij", x224) * -1
    del x224
    x510 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x510 += einsum("iajb,jkcb->ikac", v.aabb.ovov, x38)
    x511 += einsum("ijab->ijab", x510) * -1
    del x510
    x566 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x566 += einsum("abij,ijbc->ac", l2.bbbb, x38) * -2
    x618 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x618 += einsum("wxai,ijba->wxjb", lu12.bb, x38)
    x619 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x619 += einsum("wxia->xwia", x618) * -1
    del x618
    x39 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x39 += einsum("ia,jbka->jikb", t1.bb, v.aabb.ovov)
    x40 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x40 += einsum("ijka->ikja", x39)
    x44 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x44 += einsum("ijka->ikja", x39)
    x354 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x354 += einsum("ijka,ljkb->ilab", x1, x39)
    x390 -= einsum("ijab->ijab", x354)
    del x354
    x561 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x561 += einsum("ijka->ikja", x39)
    x40 += einsum("iajk->ijka", v.aabb.ovoo)
    x49 += einsum("ijab,kjla->kilb", t2.abab, x40) * -1
    x133 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x133 += einsum("ijab,ijka->kb", t2.abab, x40)
    x159 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x159 += einsum("ia->ia", x133) * -1
    del x133
    x281 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x281 += einsum("ijab,ikla->jklb", t2.abab, x40)
    x285 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x285 += einsum("ijka->kjia", x281)
    x295 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x295 += einsum("ijka->kjia", x281)
    del x281
    x317 = np.zeros((nbos, nbos, nocc[1], nocc[1]), dtype=types[float])
    x317 += einsum("wxia,ijka->xwjk", u12.aa, x40) * 0.5
    x527 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x527 += einsum("ia,jkla->ijkl", t1.aa, x40)
    x528 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x528 += einsum("ijkl->jikl", x527)
    del x527
    x41 += einsum("ia->ia", f.aa.ov)
    x49 += einsum("ia,jkab->ijkb", x41, t2.abab)
    x77 += einsum("ia,wxja->xwij", x41, u12.aa)
    x108 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x108 += einsum("ia,ja->ij", t1.aa, x41)
    x109 += einsum("ij->ji", x108)
    del x108
    x135 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x135 += einsum("ia,ijab->jb", x41, t2.abab)
    x159 += einsum("ia->ia", x135)
    del x135
    x219 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x219 += einsum("ia,wja->wji", x41, u11.aa)
    x223 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x223 += einsum("wij->wji", x219)
    del x219
    x233 = np.zeros((nbos, nbos), dtype=types[float])
    x233 += einsum("ia,wxia->xw", x41, u12.aa) * 0.5
    x255 = np.zeros((nbos), dtype=types[float])
    x255 += einsum("ia,wia->w", x41, u11.aa)
    x258 = np.zeros((nbos), dtype=types[float])
    x258 += einsum("w->w", x255)
    del x255
    x334 = np.zeros((nbos, nbos), dtype=types[float])
    x334 += einsum("ia,wxia->xw", x41, u12.aa)
    lu11new_aa = np.zeros((nbos, nvir[0], nocc[0]), dtype=types[float])
    lu11new_aa += einsum("w,ia->wai", ls1, x41) * 2
    lu12new_aa = np.zeros((nbos, nbos, nvir[0], nocc[0]), dtype=types[float])
    lu12new_aa += einsum("wx,ia->xwai", ls2, x41) * 2
    x42 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x42 += einsum("ia,jkla->jkil", t1.bb, v.aabb.ooov)
    x46 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x46 += einsum("ijkl->ijlk", x42)
    x528 += einsum("ijkl->ijlk", x42)
    del x42
    x43 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x43 += einsum("ijab,kalb->ikjl", t2.abab, v.aabb.ovov)
    x46 += einsum("ijkl->jilk", x43)
    x528 += einsum("ijkl->jilk", x43)
    del x43
    x44 += einsum("iajk->ijka", v.aabb.ovoo)
    x45 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x45 += einsum("ia,jkla->ijkl", t1.aa, x44)
    del x44
    x46 += einsum("ijkl->jikl", x45)
    del x45
    x46 += einsum("ijkl->ijkl", v.aabb.oooo)
    x49 += einsum("ia,jkil->jkla", t1.bb, x46) * -1
    x293 += einsum("ia,ijkl->jkla", t1.aa, x46) * -1
    del x46
    x47 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x47 += einsum("ia,jbca->jibc", t1.bb, v.aabb.ovvv)
    x48 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x48 += einsum("ijab->ijab", x47)
    x361 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x361 += einsum("ijab->ijab", x47)
    x511 += einsum("ijab->ijab", x47)
    x620 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x620 += einsum("ijab->ijab", x47)
    del x47
    x48 += einsum("iabj->ijab", v.aabb.ovvo)
    x49 += einsum("ia,jkab->jikb", t1.aa, x48)
    x49 += einsum("ijak->ijka", v.aabb.oovo)
    x49 += einsum("ia,jkba->jkib", t1.bb, v.aabb.oovv)
    x49 += einsum("ijab,kacb->kijc", t2.abab, v.aabb.ovvv)
    l1new_aa += einsum("abij,kijb->ak", l2.abab, x49) * -1
    del x49
    x52 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x52 += einsum("ijab->jiab", t2.bbbb)
    x52 += einsum("ijab->jiba", t2.bbbb) * -1
    x53 += einsum("abij,jkcb->ikac", l2.abab, x52) * -1
    x129 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x129 += einsum("iabc,ijac->jb", v.bbbb.ovvv, x52)
    x159 += einsum("ia->ia", x129) * -1
    del x129
    x195 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x195 += einsum("ai,ijba->jb", l1.bb, x52)
    x207 += einsum("ia->ia", x195) * -1
    del x195
    x53 += einsum("wxai,wxjb->ijab", lu12.aa, u12.bb) * -0.5
    l1new_aa += einsum("iabc,jiba->cj", v.bbaa.ovvv, x53) * -1
    del x53
    x54 = np.zeros((nbos, nbos, nocc[0], nocc[0]), dtype=types[float])
    x54 += einsum("ia,wxaj->wxji", t1.aa, lu12.aa)
    x58 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x58 += einsum("wxia,wxjk->jkia", u12.aa, x54) * 0.5
    x62 += einsum("wxia,wxjk->jkia", u12.bb, x54) * -0.5
    x163 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x163 += einsum("wxia,wxij->ja", u12.aa, x54)
    x178 += einsum("ia->ia", x163) * 0.5
    del x163
    x585 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x585 += einsum("wia,xwij->xja", u11.aa, x54)
    x586 = np.zeros((nbos, nbos), dtype=types[float])
    x586 += einsum("wia,xia->wx", g.aa.bov, x585)
    del x585
    x605 = np.zeros((nbos, nbos), dtype=types[float])
    x605 -= einsum("wx->wx", x586)
    del x586
    lu12new_aa += einsum("ijka,wxik->xwaj", x535, x54) * -1
    lu12new_aa += einsum("ia,wxji->xwaj", x41, x54) * -1
    lu12new_bb = np.zeros((nbos, nbos, nvir[1], nocc[1]), dtype=types[float])
    lu12new_bb -= einsum("ijka,wxji->xwak", v.aabb.ooov, x54)
    lu12new_bb -= einsum("ijka,wxij->xwak", x25, x54)
    x55 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x55 += einsum("ia,abjk->jikb", t1.aa, l2.abab)
    x58 += einsum("ijab,kljb->klia", t2.abab, x55)
    x62 += einsum("ijab,klib->klja", x52, x55) * -1
    x164 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x164 += einsum("ijab,ikjb->ka", t2.abab, x55)
    x178 += einsum("ia->ia", x164)
    del x164
    x297 += einsum("ia,ijkb->jkab", t1.aa, x55) * 2
    x301 += einsum("ijab,iklb->klja", t2.abab, x55)
    x426 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x426 += einsum("ijka,jilb->lkba", v.aabb.ooov, x55)
    x467 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x467 -= einsum("ijab->ijab", x426)
    del x426
    x429 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x429 += einsum("ijka,ijlb->lkba", x25, x55)
    del x25
    x467 -= einsum("ijab->ijab", x429)
    del x429
    x500 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x500 += einsum("iabc,jikb->jkac", v.aabb.ovvv, x55)
    l2new_baba -= einsum("ijab->baji", x500)
    l2new_abab -= einsum("ijab->abij", x500)
    del x500
    x533 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x533 += einsum("ijka,likb->ljab", x40, x55)
    l2new_baba += einsum("ijab->baji", x533)
    l2new_abab += einsum("ijab->abij", x533)
    del x533
    x551 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x551 += einsum("ia,jikb->jkab", x41, x55)
    l2new_baba += einsum("ijab->baji", x551) * -1
    l2new_abab += einsum("ijab->abij", x551) * -1
    del x551
    l1new_aa += einsum("ijab,kijb->ak", x48, x55) * -1
    del x48
    l1new_bb -= einsum("ijab,jika->bk", v.aabb.oovv, x55)
    l2new_baba += einsum("ijka,iklb->balj", x535, x55) * -1
    del x535
    l2new_abab += einsum("ijka,iklb->abjl", x395, x55)
    x56 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x56 += einsum("ia,abjk->kjib", t1.aa, l2.aaaa)
    x57 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x57 += einsum("ijka->ijka", x56)
    x57 += einsum("ijka->jika", x56) * -1
    x58 += einsum("ijab,kilb->klja", x23, x57)
    del x57
    x60 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x60 += einsum("ijka->ijka", x56) * -1
    x60 += einsum("ijka->jika", x56)
    x62 += einsum("ijab,kila->kljb", t2.abab, x60) * -1
    x63 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x63 += einsum("ia,ijkb->jkba", t1.aa, x60) * -1
    l1new_aa += einsum("iabc,jiab->cj", x36, x63)
    del x36
    del x63
    x396 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x396 += einsum("ijka,ilkb->ljba", x395, x60)
    del x395
    x414 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x414 += einsum("ijab->ijab", x396)
    del x396
    l1new_aa += einsum("ijab,jkia->bk", x28, x60)
    del x28
    l2new_baba += einsum("ijka,jlib->abkl", x26, x60) * -1
    del x60
    x68 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x68 += einsum("ia,jkla->kjli", t1.aa, x56)
    x69 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x69 += einsum("ijkl->ijkl", x68)
    x122 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x122 += einsum("ijkl->ijkl", x68)
    x122 += einsum("ijkl->ijlk", x68) * -1
    l1new_aa += einsum("ijka,ljki->al", v.aaaa.ooov, x122)
    del x122
    x420 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x420 += einsum("ijkl->ijkl", x68)
    del x68
    x352 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x352 += einsum("iabc,jkib->kjac", v.aaaa.ovvv, x56)
    x390 -= einsum("ijab->ijab", x352)
    del x352
    x363 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x363 -= einsum("ijka->ijka", x56)
    x363 += einsum("ijka->jika", x56)
    l2new_abab += einsum("ijka,jlib->balk", x26, x363) * -1
    del x26
    x413 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x413 += einsum("ia,jkib->jkba", x14, x56)
    del x14
    x414 += einsum("ijab->ijab", x413) * -1
    del x413
    x59 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x59 += einsum("iajb->jiab", v.aaaa.ovov)
    x59 -= einsum("iajb->jiba", v.aaaa.ovov)
    x243 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x243 += einsum("wia,ijab->wjb", u11.aa, x59)
    x244 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x244 -= einsum("wia->wia", x243)
    x381 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x381 -= einsum("wia->wia", x243)
    x553 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x553 -= einsum("wia->wia", x243)
    del x243
    x388 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x388 += einsum("ia,ijab->jb", t1.aa, x59)
    x389 -= einsum("ia->ia", x388)
    x390 += einsum("ai,jb->ijab", l1.aa, x389)
    del x389
    x564 -= einsum("ia->ia", x388)
    del x388
    l1new_aa += einsum("ijka,jkab->bi", x58, x59) * -1
    del x58
    del x59
    x62 += einsum("ai,jkab->ijkb", l1.aa, t2.abab) * -1
    l1new_aa += einsum("iajb,kijb->ak", v.aabb.ovov, x62)
    del x62
    x64 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x64 += einsum("ia,bacd->icbd", t1.aa, v.aaaa.vvvv)
    x65 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x65 += einsum("iabc->iabc", x64) * -1
    del x64
    x65 += einsum("aibc->iabc", v.aaaa.vovv)
    x66 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x66 += einsum("iabc->iabc", x65)
    x66 += einsum("iabc->ibac", x65) * -1
    del x65
    l1new_aa += einsum("abij,ibac->cj", l2.aaaa, x66) * -1
    del x66
    x67 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x67 += einsum("abij,klab->ijkl", l2.aaaa, t2.aaaa)
    x69 += einsum("ijkl->jilk", x67)
    x70 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x70 += einsum("ia,ijkl->jkla", t1.aa, x69)
    del x69
    x71 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x71 += einsum("ijka->ijka", x70)
    x71 += einsum("ijka->ikja", x70) * -1
    del x70
    l1new_aa += einsum("iajb,kjib->ak", v.aaaa.ovov, x71)
    del x71
    x123 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x123 += einsum("ijkl->jikl", x67) * -1
    x123 += einsum("ijkl->jilk", x67)
    l1new_aa += einsum("ijka,jlki->al", v.aaaa.ooov, x123) * -1
    del x123
    x420 += einsum("ijkl->jilk", x67)
    del x67
    x421 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x421 += einsum("iajb,klji->klab", v.aaaa.ovov, x420)
    del x420
    x422 += einsum("ijab->jiab", x421)
    del x421
    l2new_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    l2new_aaaa += einsum("ijab->abij", x422)
    l2new_aaaa += einsum("ijab->baij", x422) * -1
    del x422
    x72 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x72 += einsum("abci->iabc", v.aabb.vvvo)
    x72 += einsum("ia,bcda->ibcd", t1.bb, v.aabb.vvvv)
    l1new_aa += einsum("abij,jacb->ci", l2.abab, x72)
    del x72
    x73 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x73 += einsum("wx,wia->xia", s2, g.aa.bov)
    x76 += einsum("wia->wia", x73)
    x553 += einsum("wia->wia", x73)
    del x73
    x74 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x74 += einsum("wia,jbia->wjb", u11.bb, v.aabb.ovov)
    x76 += einsum("wia->wia", x74)
    x220 += einsum("wia->wia", x74)
    x244 += einsum("wia->wia", x74)
    x381 += einsum("wia->wia", x74)
    x382 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x382 += einsum("wai,wjb->ijab", lu11.aa, x381)
    del x381
    x390 += einsum("ijab->ijab", x382)
    del x382
    x553 += einsum("wia->wia", x74)
    del x74
    x76 += einsum("wia->wia", gc.aa.bov)
    x77 += einsum("wia,xja->wxji", u11.aa, x76) * 2
    l1new_aa += einsum("wxai,wxji->aj", lu12.aa, x77) * -0.5
    del x77
    x571 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x571 += einsum("wia,ijab->wjb", x76, t2.abab) * -1
    x575 += einsum("wia,ijab->wjb", x76, x23) * -1
    del x76
    del x23
    x78 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x78 += einsum("ai,jkba->ijkb", l1.aa, t2.aaaa)
    x79 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x79 += einsum("ijka->ikja", x78)
    x79 += einsum("ijka->ijka", x78) * -1
    del x78
    l1new_aa += einsum("iajb,kjia->bk", v.aaaa.ovov, x79) * -1
    del x79
    x80 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x80 += einsum("w,wai->ia", s1, g.aa.bvo)
    x121 += einsum("ia->ia", x80)
    del x80
    x81 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x81 += einsum("wab,wib->ia", g.aa.bvv, u11.aa)
    x121 += einsum("ia->ia", x81)
    del x81
    x82 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x82 += einsum("ia,iabj->jb", t1.bb, v.bbaa.ovvo)
    x121 += einsum("ia->ia", x82)
    del x82
    x83 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x83 += einsum("ijab,jkib->ka", t2.aaaa, v.aaaa.ooov)
    x121 += einsum("ia->ia", x83)
    del x83
    x84 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x84 += einsum("ijab,jbca->ic", t2.abab, v.bbaa.ovvv)
    x121 += einsum("ia->ia", x84)
    del x84
    x86 += einsum("ijka->ikja", v.aaaa.ooov)
    x87 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x87 += einsum("ijab,ijkb->ka", t2.aaaa, x86)
    del x86
    x121 += einsum("ia->ia", x87) * -1
    del x87
    x89 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x89 += einsum("ijab->jiab", t2.aaaa)
    x89 += einsum("ijab->jiba", t2.aaaa) * -1
    x90 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x90 += einsum("ia,ijab->jb", x41, x89)
    x121 += einsum("ia->ia", x90) * -1
    del x90
    x165 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x165 += einsum("ijka,ijab->kb", x56, x89)
    x178 += einsum("ia->ia", x165) * -1
    del x165
    x293 += einsum("ijka,ilab->ljkb", x40, x89) * -1
    x392 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x392 += einsum("ijab,ikac->jkbc", x12, x89)
    x393 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x393 += einsum("ijab->jiba", x392)
    x513 += einsum("ijab->ijba", x392)
    del x392
    x505 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x505 += einsum("iajb,ikac->kjcb", v.aabb.ovov, x89)
    x507 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x507 += einsum("ijab->ijab", x505) * -1
    del x505
    x542 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x542 += einsum("iajb,ijbc->ac", v.aaaa.ovov, x89)
    x543 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x543 += einsum("ab->ba", x542) * -1
    del x542
    x614 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x614 += einsum("wxai,ijab->wxjb", lu12.aa, x89)
    del x89
    x615 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x615 += einsum("wxia->xwia", x614) * -1
    del x614
    x91 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x91 += einsum("w,wia->ia", s1, g.bb.bov)
    x95 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x95 += einsum("ia->ia", x91)
    x266 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x266 += einsum("ia->ia", x91)
    x447 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x447 += einsum("ia->ia", x91)
    x563 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x563 += einsum("ia->ia", x91)
    del x91
    x92 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x92 += einsum("ia,iajb->jb", t1.aa, v.aabb.ovov)
    x95 += einsum("ia->ia", x92)
    x272 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x272 += einsum("ia->ia", x92)
    x466 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x466 += einsum("ia->ia", x92)
    x563 += einsum("ia->ia", x92)
    del x92
    x93 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x93 += einsum("iajb->jiab", v.bbbb.ovov) * -1
    x93 += einsum("iajb->jiba", v.bbbb.ovov)
    x94 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x94 += einsum("ia,ijba->jb", t1.bb, x93)
    x95 += einsum("ia->ia", x94) * -1
    x272 += einsum("ia->ia", x94) * -1
    del x94
    x273 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x273 += einsum("ia,jkab->jkib", x272, t2.bbbb)
    x278 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x278 += einsum("ijka->jika", x273)
    del x273
    x481 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x481 += einsum("ia,ja->ij", t1.bb, x272)
    x482 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x482 += einsum("ij->ij", x481)
    del x481
    x227 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x227 += einsum("wia,ijba->wjb", u11.bb, x93) * 0.5
    x228 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x228 += einsum("wia->wia", x227) * -1
    del x227
    x315 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x315 += einsum("wia,ijba->wjb", u11.bb, x93)
    x316 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x316 += einsum("wia->wia", x315) * -1
    x330 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x330 += einsum("wia->wia", x315) * -1
    del x315
    x475 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x475 += einsum("wxia,ijba->wxjb", u12.bb, x93)
    x476 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x476 += einsum("wxia->xwia", x475) * -1
    del x475
    x480 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x480 += einsum("ijab,ikba->jk", t2.bbbb, x93)
    x482 += einsum("ij->ij", x480) * -1
    del x480
    x485 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x485 += einsum("ijab,ijbc->ac", t2.bbbb, x93)
    x486 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x486 += einsum("ab->ab", x485) * -1
    del x485
    x506 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x506 += einsum("ijab,jkcb->ikac", t2.abab, x93)
    x507 += einsum("ijab->ijab", x506) * -1
    del x506
    x95 += einsum("ia->ia", f.bb.ov)
    x96 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x96 += einsum("ia,jiba->jb", x95, t2.abab)
    x121 += einsum("ia->ia", x96)
    del x96
    x134 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x134 += einsum("ia,ijba->jb", x95, x38)
    x159 += einsum("ia->ia", x134) * -1
    del x134
    x149 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x149 += einsum("ia,ja->ij", t1.bb, x95)
    x150 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x150 += einsum("ij->ji", x149)
    del x149
    x233 += einsum("ia,wxia->xw", x95, u12.bb) * 0.5
    x256 = np.zeros((nbos), dtype=types[float])
    x256 += einsum("ia,wia->w", x95, u11.bb)
    x258 += einsum("w->w", x256)
    del x256
    x293 += einsum("ia,jkba->jikb", x95, t2.abab)
    x317 += einsum("ia,wxja->xwij", x95, u12.bb) * 0.5
    x329 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x329 += einsum("ia,wja->wji", x95, u11.bb)
    x333 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x333 += einsum("wij->wji", x329)
    del x329
    x334 += einsum("ia,wxia->xw", x95, u12.bb)
    x552 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x552 += einsum("ia,jkib->jkba", x95, x1)
    l2new_baba += einsum("ijab->baji", x552) * -1
    l2new_abab += einsum("ijab->abij", x552) * -1
    del x552
    lu11new_bb = np.zeros((nbos, nvir[1], nocc[1]), dtype=types[float])
    lu11new_bb += einsum("w,ia->wai", ls1, x95) * 2
    lu12new_bb += einsum("wx,ia->xwai", ls2, x95) * 2
    x98 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x98 += einsum("ia,wja->wji", t1.aa, g.aa.bov)
    x99 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x99 += einsum("wij->wij", x98)
    x598 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x598 += einsum("wij->wij", x98)
    del x98
    x99 += einsum("wij->wij", g.aa.boo)
    x100 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x100 += einsum("wia,wij->ja", u11.aa, x99)
    x121 += einsum("ia->ia", x100) * -1
    del x100
    x222 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x222 += einsum("wx,wij->xij", s2, x99)
    x223 += einsum("wij->wij", x222)
    del x222
    x575 += einsum("wij,wxia->xja", x99, u12.aa)
    x101 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x101 += einsum("w,wij->ij", s1, g.aa.boo)
    x109 += einsum("ij->ij", x101)
    x374 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x374 += einsum("ij->ij", x101)
    del x101
    x102 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x102 += einsum("wia,wja->ij", g.aa.bov, u11.aa)
    x109 += einsum("ij->ij", x102)
    x374 += einsum("ij->ij", x102)
    del x102
    x103 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x103 += einsum("ia,jkia->jk", t1.bb, v.aabb.ooov)
    x109 += einsum("ij->ij", x103)
    x378 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x378 += einsum("ij->ij", x103)
    del x103
    x104 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x104 += einsum("ijab,kajb->ik", t2.abab, v.aabb.ovov)
    x109 += einsum("ij->ji", x104)
    x405 += einsum("ij->ij", x104)
    del x104
    x406 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x406 += einsum("ij,abik->kjab", x405, l2.aaaa)
    del x405
    x414 += einsum("ijab->ijba", x406) * -1
    del x406
    x106 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x106 += einsum("ijka->ikja", v.aaaa.ooov) * -1
    x106 += einsum("ijka->kija", v.aaaa.ooov)
    x107 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x107 += einsum("ia,jika->jk", t1.aa, x106)
    x109 += einsum("ij->ij", x107) * -1
    del x107
    x218 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x218 += einsum("wia,jika->wjk", u11.aa, x106)
    x223 += einsum("wij->wij", x218) * -1
    del x218
    x109 += einsum("ij->ij", f.aa.oo)
    x110 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x110 += einsum("ia,ij->ja", t1.aa, x109)
    x121 += einsum("ia->ia", x110) * -1
    del x110
    x545 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x545 += einsum("ij,abjk->ikab", x109, l2.abab)
    l2new_baba += einsum("ijab->baji", x545) * -1
    l2new_abab += einsum("ijab->abij", x545) * -1
    del x545
    x575 += einsum("ij,wia->wja", x109, u11.aa)
    l1new_aa += einsum("ai,ji->aj", l1.aa, x109) * -1
    lu12new_aa += einsum("ij,wxaj->xwai", x109, lu12.aa) * -1
    del x109
    x111 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x111 += einsum("w,wab->ab", s1, g.aa.bvv)
    x115 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x115 += einsum("ab->ab", x111)
    x367 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x367 += einsum("ab->ab", x111)
    x543 += einsum("ab->ab", x111)
    del x111
    x112 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x112 += einsum("ia,iabc->bc", t1.bb, v.bbaa.ovvv)
    x115 += einsum("ab->ab", x112)
    x370 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x370 += einsum("ab->ab", x112)
    x543 += einsum("ab->ab", x112)
    del x112
    x113 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x113 += einsum("iabc->ibac", v.aaaa.ovvv) * -1
    x113 += einsum("iabc->ibca", v.aaaa.ovvv)
    x114 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x114 += einsum("ia,ibac->bc", t1.aa, x113)
    x115 += einsum("ab->ab", x114) * -1
    x543 += einsum("ab->ab", x114) * -1
    del x114
    x572 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x572 += einsum("wia,ibac->wbc", u11.aa, x113) * -1
    del x113
    x115 += einsum("ab->ab", f.aa.vv)
    x116 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x116 += einsum("ia,ba->ib", t1.aa, x115)
    x121 += einsum("ia->ia", x116)
    del x116
    l1new_aa += einsum("ai,ab->bi", l1.aa, x115)
    del x115
    x117 = np.zeros((nbos), dtype=types[float])
    x117 += einsum("ia,wia->w", t1.aa, g.aa.bov)
    x119 = np.zeros((nbos), dtype=types[float])
    x119 += einsum("w->w", x117)
    x577 = np.zeros((nbos), dtype=types[float])
    x577 += einsum("w->w", x117)
    del x117
    x118 = np.zeros((nbos), dtype=types[float])
    x118 += einsum("ia,wia->w", t1.bb, g.bb.bov)
    x119 += einsum("w->w", x118)
    x577 += einsum("w->w", x118)
    del x118
    x119 += einsum("w->w", G)
    x120 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x120 += einsum("w,wia->ia", x119, u11.aa)
    x121 += einsum("ia->ia", x120)
    del x120
    x158 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x158 += einsum("w,wia->ia", x119, u11.bb)
    x159 += einsum("ia->ia", x158)
    del x158
    x257 = np.zeros((nbos), dtype=types[float])
    x257 += einsum("w,wx->x", x119, s2)
    x258 += einsum("w->w", x257)
    del x257
    x571 += einsum("w,wxia->xia", x119, u12.bb) * -1
    x575 += einsum("w,wxia->xia", x119, u12.aa) * -1
    del x119
    x121 += einsum("ai->ia", f.aa.vo)
    l1new_aa += einsum("ia,ijab->bj", x121, x34)
    l1new_bb += einsum("ia,abij->bj", x121, l2.abab)
    ls1new = np.zeros((nbos), dtype=types[float])
    ls1new += einsum("ia,wai->w", x121, lu11.aa)
    ls2new = np.zeros((nbos, nbos), dtype=types[float])
    ls2new += einsum("ia,wxai->xw", x121, lu12.aa)
    del x121
    x124 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x124 += einsum("w,wai->ia", s1, g.bb.bvo)
    x159 += einsum("ia->ia", x124)
    del x124
    x125 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x125 += einsum("wab,wib->ia", g.bb.bvv, u11.bb)
    x159 += einsum("ia->ia", x125)
    del x125
    x126 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x126 += einsum("ia,iabj->jb", t1.aa, v.aabb.ovvo)
    x159 += einsum("ia->ia", x126)
    del x126
    x127 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x127 += einsum("ijab,iacb->jc", t2.abab, v.aabb.ovvv)
    x159 += einsum("ia->ia", x127)
    del x127
    x128 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x128 += einsum("ijab,ikja->kb", t2.bbbb, v.bbbb.ooov)
    x159 += einsum("ia->ia", x128)
    del x128
    x130 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x130 += einsum("ia,jbka->ijkb", t1.bb, v.bbbb.ovov)
    x131 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x131 += einsum("ijka->jkia", x130) * -1
    x131 += einsum("ijka->kjia", x130)
    x275 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x275 += einsum("ia,jkla->ijkl", t1.bb, x130)
    x276 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x276 += einsum("ijkl->ijkl", x275)
    x495 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x495 += einsum("ijkl->ijkl", x275)
    del x275
    x279 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x279 += einsum("ijka->ikja", x130)
    x279 += einsum("ijka->ijka", x130) * -1
    x290 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x290 += einsum("ijka->jkia", x130)
    x290 += einsum("ijka->kjia", x130) * -1
    x463 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x463 += einsum("ijka->ijka", x130)
    x470 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x470 += einsum("ijka->ikja", x130) * -1
    x470 += einsum("ijka->ijka", x130)
    l2new_baba += einsum("ijka,jklb->bali", x1, x470)
    x565 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x565 += einsum("ijka->ikja", x130)
    x565 += einsum("ijka->ijka", x130) * -1
    del x130
    l2new_abab += einsum("ijka,jklb->abil", x1, x565) * -1
    x131 += einsum("ijka->ikja", v.bbbb.ooov)
    x132 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x132 += einsum("ijab,jika->kb", t2.bbbb, x131)
    del x131
    x159 += einsum("ia->ia", x132) * -1
    del x132
    x136 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x136 += einsum("iabj->ijba", v.bbbb.ovvo)
    x136 += einsum("ijab->ijab", v.bbbb.oovv) * -1
    x137 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x137 += einsum("ia,ijba->jb", t1.bb, x136)
    x159 += einsum("ia->ia", x137)
    del x137
    x282 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x282 += einsum("ia,jkba->ijkb", t1.bb, x136)
    x285 += einsum("ijka->ijka", x282)
    x295 += einsum("ijka->ijka", x282)
    del x282
    x571 += einsum("wia,ijba->wjb", u11.bb, x136) * -1
    x138 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x138 += einsum("ia,wja->wji", t1.bb, g.bb.bov)
    x139 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x139 += einsum("wij->wij", x138)
    x601 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x601 += einsum("wij->wij", x138)
    x139 += einsum("wij->wij", g.bb.boo)
    x140 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x140 += einsum("wia,wij->ja", u11.bb, x139)
    x159 += einsum("ia->ia", x140) * -1
    del x140
    x332 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x332 += einsum("wx,wij->xij", s2, x139)
    x333 += einsum("wij->wij", x332)
    del x332
    x571 += einsum("wij,wxia->xja", x139, u12.bb)
    x141 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x141 += einsum("w,wij->ij", s1, g.bb.boo)
    x150 += einsum("ij->ij", x141)
    x449 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x449 += einsum("ij->ij", x141)
    del x141
    x142 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x142 += einsum("wia,wja->ij", g.bb.bov, u11.bb)
    x150 += einsum("ij->ij", x142)
    x449 += einsum("ij->ij", x142)
    del x142
    x143 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x143 += einsum("ia,iajk->jk", t1.aa, v.aabb.ovoo)
    x150 += einsum("ij->ij", x143)
    x453 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x453 += einsum("ij->ij", x143)
    del x143
    x144 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x144 += einsum("ijab,iakb->jk", t2.abab, v.aabb.ovov)
    x150 += einsum("ij->ji", x144)
    x482 += einsum("ij->ij", x144)
    del x144
    x483 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x483 += einsum("ij,abik->kjab", x482, l2.bbbb)
    del x482
    x491 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x491 += einsum("ijab->ijba", x483) * -1
    del x483
    x145 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x145 += einsum("iajb->jiab", v.bbbb.ovov)
    x145 += einsum("iajb->jiba", v.bbbb.ovov) * -1
    x146 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x146 += einsum("ijab,ikab->jk", t2.bbbb, x145)
    x150 += einsum("ij->ji", x146) * -1
    del x146
    x516 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x516 += einsum("ijab,ikac->kjcb", x145, x52)
    x518 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x518 += einsum("ijab->jiab", x516)
    del x516
    x539 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x539 += einsum("ijab,ijbc->ac", t2.bbbb, x145)
    x540 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x540 += einsum("ab->ab", x539) * -1
    del x539
    x147 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x147 += einsum("ijka->ikja", v.bbbb.ooov) * -1
    x147 += einsum("ijka->kija", v.bbbb.ooov)
    x148 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x148 += einsum("ia,jika->jk", t1.bb, x147)
    x150 += einsum("ij->ij", x148) * -1
    del x148
    x328 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x328 += einsum("wia,jika->wjk", u11.bb, x147)
    x333 += einsum("wij->wij", x328) * -1
    del x328
    x150 += einsum("ij->ij", f.bb.oo)
    x151 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x151 += einsum("ia,ij->ja", t1.bb, x150)
    x159 += einsum("ia->ia", x151) * -1
    del x151
    x546 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x546 += einsum("ij,abkj->kiab", x150, l2.abab)
    l2new_baba += einsum("ijab->baji", x546) * -1
    l2new_abab += einsum("ijab->abij", x546) * -1
    del x546
    x571 += einsum("ij,wia->wja", x150, u11.bb)
    l1new_bb += einsum("ai,ji->aj", l1.bb, x150) * -1
    lu12new_bb += einsum("ij,wxaj->xwai", x150, lu12.bb) * -1
    del x150
    x152 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x152 += einsum("w,wab->ab", s1, g.bb.bvv)
    x156 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x156 += einsum("ab->ab", x152)
    x442 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x442 += einsum("ab->ab", x152)
    x540 += einsum("ab->ab", x152)
    del x152
    x153 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x153 += einsum("ia,iabc->bc", t1.aa, v.aabb.ovvv)
    x156 += einsum("ab->ab", x153)
    x445 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x445 += einsum("ab->ab", x153)
    x540 += einsum("ab->ab", x153)
    del x153
    x154 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x154 += einsum("iabc->ibac", v.bbbb.ovvv) * -1
    x154 += einsum("iabc->ibca", v.bbbb.ovvv)
    x155 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x155 += einsum("ia,ibac->bc", t1.bb, x154)
    x156 += einsum("ab->ab", x155) * -1
    x540 += einsum("ab->ab", x155) * -1
    del x155
    x568 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x568 += einsum("wia,ibac->wbc", u11.bb, x154) * -1
    x156 += einsum("ab->ab", f.bb.vv)
    x157 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x157 += einsum("ia,ba->ib", t1.bb, x156)
    x159 += einsum("ia->ia", x157)
    del x157
    l1new_bb += einsum("ai,ab->bi", l1.bb, x156)
    del x156
    x159 += einsum("ai->ia", f.bb.vo)
    l1new_aa += einsum("ia,baji->bj", x159, l2.abab)
    ls1new += einsum("ia,wai->w", x159, lu11.bb)
    ls2new += einsum("ia,wxai->xw", x159, lu12.bb)
    x160 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x160 += einsum("w,wia->ia", ls1, u11.aa)
    x178 += einsum("ia->ia", x160) * -1
    del x160
    x161 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x161 += einsum("wx,wxia->ia", ls2, u12.aa)
    x178 += einsum("ia->ia", x161) * -0.5
    del x161
    x162 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x162 += einsum("ai,jiba->jb", l1.bb, t2.abab)
    x178 += einsum("ia->ia", x162) * -1
    del x162
    x167 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x167 += einsum("ia,waj->wji", t1.aa, lu11.aa)
    x169 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x169 += einsum("wij->wij", x167)
    x242 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x242 += einsum("wij->wij", x167)
    x168 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x168 += einsum("wia,wxaj->xji", u11.aa, lu12.aa)
    x169 += einsum("wij->wij", x168)
    x170 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x170 += einsum("wia,wij->ja", u11.aa, x169)
    x178 += einsum("ia->ia", x170)
    del x170
    x247 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x247 += einsum("wx,wij->xij", s2, x169)
    del x169
    x242 += einsum("wij->wij", x168)
    x610 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x610 += einsum("wia,xji->wxja", g.aa.bov, x242)
    x612 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x612 += einsum("wxia->wxia", x610)
    del x610
    x171 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x171 += einsum("ai,ja->ij", l1.aa, t1.aa)
    x176 += einsum("ij->ij", x171)
    x215 += einsum("ij->ij", x171) * 2
    x383 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x383 += einsum("ij->ij", x171)
    del x171
    x172 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x172 += einsum("wai,wja->ij", lu11.aa, u11.aa)
    x176 += einsum("ij->ij", x172)
    x215 += einsum("ij->ij", x172) * 2
    x383 += einsum("ij->ij", x172)
    del x172
    x384 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x384 += einsum("ij,jakb->kiab", x383, v.aaaa.ovov)
    del x383
    x390 += einsum("ijab->jiba", x384)
    del x384
    x173 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x173 += einsum("wxai,wxja->ij", lu12.aa, u12.aa)
    x176 += einsum("ij->ij", x173) * 0.5
    x215 += einsum("ij->ij", x173)
    x411 += einsum("ij->ij", x173) * 0.5
    del x173
    x174 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x174 += einsum("abij,kjab->ik", l2.abab, t2.abab)
    x176 += einsum("ij->ij", x174)
    x177 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x177 += einsum("ia,ij->ja", t1.aa, x176)
    x178 += einsum("ia->ia", x177)
    del x177
    x549 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x549 += einsum("ij,jakb->ikab", x176, v.aabb.ovov)
    l2new_baba += einsum("ijab->baji", x549) * -1
    l2new_abab += einsum("ijab->abij", x549) * -1
    del x549
    l1new_bb += einsum("ij,jika->ak", x176, v.aabb.ooov) * -1
    ls1new += einsum("ij,wji->w", x176, g.aa.boo) * -1
    del x176
    x215 += einsum("ij->ij", x174) * 2
    l1new_aa += einsum("ij,kjia->ak", x215, x106) * -0.5
    del x106
    l1new_aa += einsum("ij,ja->ai", x215, x41) * -0.5
    del x41
    del x215
    x411 += einsum("ij->ij", x174)
    del x174
    x412 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x412 += einsum("ij,jakb->kiab", x411, v.aaaa.ovov)
    del x411
    x414 += einsum("ijab->jiba", x412)
    del x412
    x178 += einsum("ia->ia", t1.aa) * -1
    l1new_bb += einsum("ia,iajb->bj", x178, v.aabb.ovov) * -1
    ls1new += einsum("ia,wia->w", x178, g.aa.bov) * -1
    x179 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x179 += einsum("iajb->jiab", v.aaaa.ovov) * -1
    x179 += einsum("iajb->jiba", v.aaaa.ovov)
    x397 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x397 += einsum("ijab,kica->kjcb", x179, x33)
    del x33
    x414 += einsum("ijab->ijab", x397)
    del x397
    l1new_aa += einsum("ia,ijab->bj", x178, x179) * -1
    del x178
    del x179
    x180 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x180 += einsum("ai,ib->ab", l1.aa, t1.aa)
    x185 += einsum("ab->ab", x180)
    x325 += einsum("ab->ab", x180) * 2
    del x180
    x181 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x181 += einsum("wai,wib->ab", lu11.aa, u11.aa)
    x185 += einsum("ab->ab", x181)
    x325 += einsum("ab->ab", x181) * 2
    x350 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x350 += einsum("ab,icjb->ijac", x181, v.aaaa.ovov)
    x390 += einsum("ijab->ijab", x350)
    del x350
    x548 += einsum("ab->ab", x181)
    x567 += einsum("ab->ab", x181) * 2
    del x181
    x182 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x182 += einsum("wxai,wxib->ab", lu12.aa, u12.aa)
    x185 += einsum("ab->ab", x182) * 0.5
    x325 += einsum("ab->ab", x182)
    x402 += einsum("ab->ab", x182) * 0.5
    x548 += einsum("ab->ab", x182) * 0.5
    x567 += einsum("ab->ab", x182)
    del x182
    x183 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x183 += einsum("abij,ijcb->ac", l2.abab, t2.abab)
    x185 += einsum("ab->ab", x183)
    ls1new += einsum("ab,wab->w", x185, g.aa.bvv)
    x325 += einsum("ab->ab", x183) * 2
    l1new_bb += einsum("ab,icab->ci", x325, v.bbaa.ovvv) * 0.5
    del x325
    x402 += einsum("ab->ab", x183)
    x403 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x403 += einsum("ab,ibjc->ijca", x402, v.aaaa.ovov)
    del x402
    x414 += einsum("ijab->jiba", x403)
    del x403
    x548 += einsum("ab->ab", x183)
    l2new_baba += einsum("ab,ibjc->caji", x548, v.aabb.ovov) * -1
    del x548
    x567 += einsum("ab->ab", x183) * 2
    del x183
    l2new_abab += einsum("ab,ibjc->acij", x567, v.aabb.ovov) * -0.5
    del x567
    x186 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x186 += einsum("iabc->ibac", v.aaaa.ovvv)
    x186 -= einsum("iabc->ibca", v.aaaa.ovvv)
    x616 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x616 -= einsum("ia,jbac->jibc", t1.aa, x186)
    l1new_aa += einsum("ab,iabc->ci", x185, x186) * -1
    del x185
    del x186
    x187 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x187 += einsum("w,wia->ia", ls1, u11.bb)
    x207 += einsum("ia->ia", x187) * -1
    del x187
    x188 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x188 += einsum("wx,wxia->ia", ls2, u12.bb)
    x207 += einsum("ia->ia", x188) * -0.5
    del x188
    x189 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x189 += einsum("ai,ijab->jb", l1.aa, t2.abab)
    x207 += einsum("ia->ia", x189) * -1
    del x189
    x190 = np.zeros((nbos, nbos, nocc[1], nocc[1]), dtype=types[float])
    x190 += einsum("ia,wxaj->wxji", t1.bb, lu12.bb)
    x191 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x191 += einsum("wxia,wxij->ja", u12.bb, x190)
    x207 += einsum("ia->ia", x191) * 0.5
    del x191
    x299 += einsum("wxia,wxjk->jkia", u12.bb, x190) * 0.5
    x301 += einsum("wxia,wxjk->ijka", u12.aa, x190) * -0.5
    x587 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x587 += einsum("wia,xwij->xja", u11.bb, x190)
    x588 = np.zeros((nbos, nbos), dtype=types[float])
    x588 += einsum("wia,xia->wx", g.bb.bov, x587)
    del x587
    x605 -= einsum("wx->wx", x588)
    del x588
    lu12new_aa -= einsum("iajk,wxkj->xwai", v.aabb.ovoo, x190)
    lu12new_aa -= einsum("wxij,kija->xwak", x190, x39)
    del x39
    lu12new_bb += einsum("wxij,ijka->xwak", x190, x565) * -1
    del x565
    lu12new_bb += einsum("ia,wxji->xwaj", x95, x190) * -1
    x193 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x193 += einsum("ia,abjk->kjib", t1.bb, l2.bbbb)
    x194 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x194 += einsum("ijka,ijab->kb", x193, x52)
    x207 += einsum("ia->ia", x194) * -1
    del x194
    x298 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x298 += einsum("ijka->ijka", x193) * -1
    x298 += einsum("ijka->jika", x193)
    x299 += einsum("ijka,ilba->jklb", x298, x38)
    x301 += einsum("ijab,kjlb->ikla", t2.abab, x298) * -1
    l1new_bb += einsum("ijab,kjia->bk", x136, x298) * -1
    del x136
    del x298
    x302 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x302 += einsum("ijka->ijka", x193)
    x302 += einsum("ijka->jika", x193) * -1
    x303 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x303 += einsum("ia,jikb->jkba", t1.bb, x302) * -1
    x471 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x471 += einsum("ijka,jklb->ilab", x302, x470)
    del x302
    del x470
    x491 += einsum("ijab->ijab", x471)
    del x471
    x309 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x309 += einsum("ia,jkla->jkil", t1.bb, x193)
    x310 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x310 += einsum("ijkl->ijkl", x309)
    x323 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x323 += einsum("ijkl->ijkl", x309)
    x323 += einsum("ijkl->ijlk", x309) * -1
    l1new_bb += einsum("ijka,ljki->al", v.bbbb.ooov, x323)
    del x323
    x497 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x497 += einsum("ijkl->ijkl", x309)
    del x309
    x428 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x428 += einsum("iabc,jkib->kjac", v.bbbb.ovvv, x193)
    x467 -= einsum("ijab->ijab", x428)
    del x428
    x438 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x438 += einsum("ijka->ijka", x193)
    x438 -= einsum("ijka->jika", x193)
    l2new_abab += einsum("ijka,lkjb->abil", x40, x438) * -1
    x490 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x490 += einsum("ia,jkib->jkba", x272, x193)
    del x272
    x491 += einsum("ijab->ijab", x490) * -1
    del x490
    x530 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x530 -= einsum("ijka->ijka", x193)
    x530 += einsum("ijka->jika", x193)
    l2new_baba += einsum("ijka,lkjb->bali", x40, x530)
    del x40
    del x530
    x196 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x196 += einsum("ia,waj->wji", t1.bb, lu11.bb)
    x198 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x198 += einsum("wij->wij", x196)
    x343 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x343 += einsum("wij->wij", x196)
    x197 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x197 += einsum("wia,wxaj->xji", u11.bb, lu12.bb)
    x198 += einsum("wij->wij", x197)
    x199 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x199 += einsum("wia,wij->ja", u11.bb, x198)
    x207 += einsum("ia->ia", x199)
    del x199
    x345 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x345 += einsum("wx,wij->xij", s2, x198)
    del x198
    x343 += einsum("wij->wij", x197)
    x628 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x628 += einsum("wia,xji->wxja", g.bb.bov, x343)
    x630 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x630 += einsum("wxia->wxia", x628)
    del x628
    x582 = np.zeros((nbos, nbos), dtype=types[float])
    x582 += einsum("wij,xji->wx", g.bb.boo, x197)
    x605 -= einsum("wx->wx", x582)
    del x582
    x589 = np.zeros((nbos, nbos), dtype=types[float])
    x589 += einsum("wij,xji->wx", x138, x197)
    del x138
    del x197
    x605 -= einsum("wx->wx", x589)
    del x589
    x200 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x200 += einsum("ai,ja->ij", l1.bb, t1.bb)
    x205 += einsum("ij->ij", x200)
    x225 += einsum("ij->ij", x200) * 2
    x460 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x460 += einsum("ij->ij", x200)
    ls1new -= einsum("ij,wji->w", x200, g.bb.boo)
    del x200
    x201 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x201 += einsum("wai,wja->ij", lu11.bb, u11.bb)
    x205 += einsum("ij->ij", x201)
    x225 += einsum("ij->ij", x201) * 2
    x460 += einsum("ij->ij", x201)
    x461 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x461 += einsum("ij,jakb->kiab", x460, v.bbbb.ovov)
    del x460
    x467 += einsum("ijab->jiba", x461)
    del x461
    x576 += einsum("ij->ij", x201)
    del x201
    x202 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x202 += einsum("wxai,wxja->ij", lu12.bb, u12.bb)
    x205 += einsum("ij->ij", x202) * 0.5
    x225 += einsum("ij->ij", x202)
    x488 += einsum("ij->ij", x202) * 0.5
    x576 += einsum("ij->ij", x202) * 0.5
    del x202
    x203 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x203 += einsum("abij,ikab->jk", l2.abab, t2.abab)
    x205 += einsum("ij->ij", x203)
    x206 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x206 += einsum("ia,ij->ja", t1.bb, x205)
    x207 += einsum("ia->ia", x206)
    del x206
    x550 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x550 += einsum("ij,kajb->kiab", x205, v.aabb.ovov)
    l2new_baba += einsum("ijab->baji", x550) * -1
    l2new_abab += einsum("ijab->abij", x550) * -1
    del x550
    l1new_bb += einsum("ij,ja->ai", x205, x95) * -1
    del x205
    del x95
    x225 += einsum("ij->ij", x203) * 2
    l1new_aa += einsum("ij,kaji->ak", x225, v.aabb.ovoo) * -0.5
    l1new_bb += einsum("ij,kjia->ak", x225, x147) * -0.5
    del x225
    del x147
    x488 += einsum("ij->ij", x203)
    x489 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x489 += einsum("ij,jakb->kiab", x488, v.bbbb.ovov)
    del x488
    x491 += einsum("ijab->jiba", x489)
    del x489
    x576 += einsum("ij->ij", x203)
    del x203
    ls1new += einsum("ij,wji->w", x576, g.bb.boo) * -1
    del x576
    x207 += einsum("ia->ia", t1.bb) * -1
    l1new_aa += einsum("ia,jbia->bj", x207, v.aabb.ovov) * -1
    l1new_bb += einsum("ia,ijba->bj", x207, x93)
    ls1new += einsum("ia,wia->w", x207, g.bb.bov) * -1
    del x207
    x208 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x208 += einsum("ai,ib->ab", l1.bb, t1.bb)
    x213 += einsum("ab->ab", x208)
    ls1new += einsum("ab,wab->w", x208, g.bb.bvv)
    del x208
    x209 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x209 += einsum("wai,wib->ab", lu11.bb, u11.bb)
    x213 += einsum("ab->ab", x209)
    x425 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x425 += einsum("ab,icjb->ijac", x209, v.bbbb.ovov)
    x467 += einsum("ijab->ijab", x425)
    del x425
    x547 += einsum("ab->ab", x209)
    x566 += einsum("ab->ab", x209) * 2
    del x209
    x210 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x210 += einsum("wxai,wxib->ab", lu12.bb, u12.bb)
    x213 += einsum("ab->ab", x210) * 0.5
    x478 += einsum("ab->ab", x210) * 0.5
    x547 += einsum("ab->ab", x210) * 0.5
    x566 += einsum("ab->ab", x210)
    del x210
    x211 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x211 += einsum("abij,ijac->bc", l2.abab, t2.abab)
    x213 += einsum("ab->ab", x211)
    l1new_aa += einsum("ab,icab->ci", x213, v.aabb.ovvv)
    l1new_bb += einsum("ab,iabc->ci", x213, x154)
    del x213
    x478 += einsum("ab->ab", x211)
    x479 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x479 += einsum("ab,ibjc->ijca", x478, v.bbbb.ovov)
    del x478
    x491 += einsum("ijab->jiba", x479)
    del x479
    x547 += einsum("ab->ab", x211)
    l2new_baba += einsum("ab,icjb->acji", x547, v.aabb.ovov) * -1
    ls1new += einsum("ab,wab->w", x547, g.bb.bvv)
    del x547
    x566 += einsum("ab->ab", x211) * 2
    del x211
    l2new_abab += einsum("ab,icjb->caij", x566, v.aabb.ovov) * -0.5
    del x566
    x216 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x216 += einsum("wia,wxja->xij", g.aa.bov, u12.aa)
    x223 += einsum("wij->wij", x216)
    del x216
    x217 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x217 += einsum("wia,jkia->wjk", u11.bb, v.aabb.ooov)
    x223 += einsum("wij->wij", x217)
    del x217
    x220 += einsum("wia->wia", gc.aa.bov)
    x221 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x221 += einsum("ia,wja->wij", t1.aa, x220)
    del x220
    x223 += einsum("wij->wji", x221)
    del x221
    x223 += einsum("wij->wij", gc.aa.boo)
    x575 += einsum("ia,wij->wja", t1.aa, x223)
    l1new_aa += einsum("wai,wji->aj", lu11.aa, x223) * -1
    del x223
    x226 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x226 += einsum("wia,iajb->wjb", u11.aa, v.aabb.ovov)
    x228 += einsum("wia->wia", x226)
    x316 += einsum("wia->wia", x226)
    x330 += einsum("wia->wia", x226)
    x458 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x458 += einsum("wia->wia", x226)
    x555 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x555 += einsum("wia->wia", x226)
    del x226
    x228 += einsum("wia->wia", gc.bb.bov)
    x233 += einsum("wia,xia->xw", u11.bb, x228)
    x334 += einsum("wia,xia->xw", u11.bb, x228) * 2
    del x228
    x229 += einsum("wia->wia", gc.aa.bov) * 2
    x233 += einsum("wia,xia->xw", u11.aa, x229) * 0.5
    x334 += einsum("wia,xia->xw", u11.aa, x229)
    del x229
    x230 = np.zeros((nbos, nbos), dtype=types[float])
    x230 += einsum("wia,xia->wx", g.aa.bov, u11.aa)
    x232 = np.zeros((nbos, nbos), dtype=types[float])
    x232 += einsum("wx->wx", x230)
    x603 = np.zeros((nbos, nbos), dtype=types[float])
    x603 += einsum("wx->wx", x230)
    del x230
    x231 = np.zeros((nbos, nbos), dtype=types[float])
    x231 += einsum("wia,xia->wx", g.bb.bov, u11.bb)
    x232 += einsum("wx->wx", x231)
    x603 += einsum("wx->wx", x231)
    del x231
    x604 = np.zeros((nbos, nbos), dtype=types[float])
    x604 += einsum("wx,yw->xy", ls2, x603)
    x605 += einsum("wx->xw", x604)
    del x604
    x611 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x611 += einsum("wx,xyai->ywia", x603, lu12.aa)
    x612 -= einsum("wxia->xwia", x611)
    del x611
    x629 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x629 += einsum("wx,xyai->ywia", x603, lu12.bb)
    del x603
    x630 -= einsum("wxia->xwia", x629)
    del x629
    x232 += einsum("wx->wx", w)
    x233 += einsum("wx,wy->xy", s2, x232)
    l1new_aa += einsum("wx,xwai->ai", x233, lu12.aa)
    del x233
    x334 += einsum("wx,wy->xy", s2, x232) * 2
    l1new_bb += einsum("wx,xwai->ai", x334, lu12.bb) * 0.5
    del x334
    x571 += einsum("wx,wia->xia", x232, u11.bb) * -1
    x575 += einsum("wx,wia->xia", x232, u11.aa) * -1
    del x232
    x234 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x234 += einsum("wx,wai->xia", s2, lu11.aa)
    x238 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x238 += einsum("wia->wia", x234)
    del x234
    x235 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x235 += einsum("wia,baji->wjb", u11.bb, l2.abab)
    x238 += einsum("wia->wia", x235)
    x246 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x246 += einsum("wia->wia", x235)
    x248 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x248 += einsum("wia->wia", x235)
    del x235
    x236 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x236 += einsum("abij->jiab", l2.aaaa)
    x236 -= einsum("abij->jiba", l2.aaaa)
    x237 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x237 += einsum("wia,ijab->wjb", u11.aa, x236)
    x238 -= einsum("wia->wia", x237)
    x380 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x380 += einsum("wia,wjb->ijab", g.aa.bov, x238)
    x390 += einsum("ijab->ijab", x380)
    del x380
    l1new_aa += einsum("wab,wia->bi", g.aa.bvv, x238)
    del x238
    x248 -= einsum("wia->wia", x237)
    del x237
    x558 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x558 += einsum("wia,wjb->jiba", g.bb.bov, x248)
    l2new_baba += einsum("ijab->baji", x558)
    l2new_abab += einsum("ijab->abij", x558)
    del x558
    l1new_aa += einsum("wia,wji->aj", x248, x99) * -1
    del x248
    del x99
    lu11new_aa -= einsum("wai,ijab->wbj", g.aa.bvo, x236)
    del x236
    x239 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x239 += einsum("wia,iabc->wbc", u11.bb, v.bbaa.ovvv)
    x241 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x241 += einsum("wab->wab", x239)
    x572 += einsum("wab->wab", x239)
    del x239
    x240 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x240 -= einsum("iabc->ibac", v.aaaa.ovvv)
    x240 += einsum("iabc->ibca", v.aaaa.ovvv)
    x241 -= einsum("wia,ibac->wbc", u11.aa, x240)
    x356 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x356 += einsum("ia,jbca->ijbc", t1.aa, x240)
    x357 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x357 -= einsum("ijab->ijab", x356)
    del x356
    x369 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x369 += einsum("ia,ibac->bc", t1.aa, x240)
    del x240
    x370 -= einsum("ab->ab", x369)
    del x369
    x371 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x371 += einsum("ab,acij->ijcb", x370, l2.aaaa)
    del x370
    x390 += einsum("ijab->jiab", x371)
    del x371
    x241 += einsum("wab->wab", gc.aa.bvv)
    l1new_aa += einsum("wai,wab->bi", lu11.aa, x241)
    del x241
    x244 += einsum("wia->wia", gc.aa.bov)
    l1new_aa -= einsum("wij,wja->ai", x242, x244)
    del x242
    del x244
    x245 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x245 += einsum("abij->jiab", l2.aaaa)
    x245 += einsum("abij->jiba", l2.aaaa) * -1
    x246 += einsum("wia,ijab->wjb", u11.aa, x245) * -1
    x247 += einsum("ia,wja->wji", t1.aa, x246)
    del x246
    x247 += einsum("ai,wja->wij", l1.aa, u11.aa)
    x247 += einsum("wai,wxja->xij", lu11.aa, u12.aa)
    l1new_aa += einsum("wia,wji->aj", g.aa.bov, x247) * -1
    del x247
    x249 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x249 += einsum("iabj->ijba", v.aaaa.ovvo)
    x249 -= einsum("ijab->ijab", v.aaaa.oovv)
    l1new_aa += einsum("ai,jiab->bj", l1.aa, x249)
    lu11new_aa += einsum("wai,jiab->wbj", lu11.aa, x249)
    del x249
    x250 = np.zeros((nbos), dtype=types[float])
    x250 += einsum("w,wx->x", s1, w)
    x258 += einsum("w->w", x250)
    x578 = np.zeros((nbos), dtype=types[float])
    x578 += einsum("w->w", x250)
    del x250
    x251 = np.zeros((nbos), dtype=types[float])
    x251 += einsum("ia,wia->w", t1.aa, gc.aa.bov)
    x258 += einsum("w->w", x251)
    x578 += einsum("w->w", x251)
    del x251
    x252 = np.zeros((nbos), dtype=types[float])
    x252 += einsum("ia,wia->w", t1.bb, gc.bb.bov)
    x258 += einsum("w->w", x252)
    x578 += einsum("w->w", x252)
    del x252
    x253 = np.zeros((nbos), dtype=types[float])
    x253 += einsum("wia,wxia->x", g.aa.bov, u12.aa)
    x258 += einsum("w->w", x253)
    x578 += einsum("w->w", x253)
    del x253
    x254 = np.zeros((nbos), dtype=types[float])
    x254 += einsum("wia,wxia->x", g.bb.bov, u12.bb)
    x258 += einsum("w->w", x254)
    x578 += einsum("w->w", x254)
    del x254
    x258 += einsum("w->w", G)
    l1new_aa += einsum("w,wai->ai", x258, lu11.aa)
    l1new_bb += einsum("w,wai->ai", x258, lu11.bb)
    del x258
    x259 = np.zeros((nbos), dtype=types[float])
    x259 += einsum("w,xw->x", ls1, s2)
    x264 = np.zeros((nbos), dtype=types[float])
    x264 += einsum("w->w", x259)
    del x259
    x260 = np.zeros((nbos), dtype=types[float])
    x260 += einsum("ai,wia->w", l1.aa, u11.aa)
    x264 += einsum("w->w", x260)
    del x260
    x261 = np.zeros((nbos), dtype=types[float])
    x261 += einsum("ai,wia->w", l1.bb, u11.bb)
    x264 += einsum("w->w", x261)
    del x261
    x262 = np.zeros((nbos), dtype=types[float])
    x262 += einsum("wai,wxia->x", lu11.aa, u12.aa)
    x264 += einsum("w->w", x262)
    del x262
    x263 = np.zeros((nbos), dtype=types[float])
    x263 += einsum("wai,wxia->x", lu11.bb, u12.bb)
    x264 += einsum("w->w", x263)
    del x263
    x264 += einsum("w->w", s1)
    l1new_aa += einsum("w,wia->ai", x264, g.aa.bov)
    l1new_bb += einsum("w,wia->ai", x264, g.bb.bov)
    del x264
    x265 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x265 += einsum("abij,ikcb->jkac", l2.abab, t2.abab)
    l1new_bb += einsum("iabc,jibc->aj", v.bbaa.ovvv, x265) * -1
    del x265
    x266 += einsum("ia->ia", f.bb.ov)
    x267 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x267 += einsum("ia,jkab->jkib", x266, t2.bbbb)
    del x266
    x268 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x268 += einsum("ijka->kjia", x267)
    del x267
    x268 += einsum("ijak->ijka", v.bbbb.oovo) * -1
    x285 += einsum("ijka->kija", x268)
    x285 += einsum("ijka->jika", x268) * -1
    del x268
    x269 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x269 += einsum("ijab,kbca->jikc", t2.bbbb, v.bbbb.ovvv)
    x278 += einsum("ijka->ijka", x269) * -1
    del x269
    x270 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x270 += einsum("ia,jbca->ijcb", t1.bb, v.bbbb.ovvv)
    x271 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x271 += einsum("ia,jkba->ijkb", t1.bb, x270)
    del x270
    x278 += einsum("ijka->ijka", x271) * -1
    del x271
    x274 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x274 += einsum("ijab,kalb->ijkl", t2.bbbb, v.bbbb.ovov)
    x276 += einsum("ijkl->jilk", x274)
    x277 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x277 += einsum("ia,jkil->jkla", t1.bb, x276)
    del x276
    x278 += einsum("ijka->jika", x277)
    del x277
    x285 += einsum("ijka->jkia", x278)
    x285 += einsum("ijka->ikja", x278) * -1
    del x278
    x495 += einsum("ijkl->jilk", x274)
    del x274
    x496 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x496 += einsum("abij,ijkl->klab", l2.bbbb, x495)
    del x495
    x499 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x499 += einsum("ijab->jiba", x496)
    del x496
    x279 += einsum("ijka->jkia", v.bbbb.ooov) * -1
    x279 += einsum("ijka->jika", v.bbbb.ooov)
    x280 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x280 += einsum("ijka,jlba->likb", x279, x38)
    del x279
    x285 += einsum("ijka->jkia", x280)
    x295 += einsum("ijka->jkia", x280)
    del x280
    x283 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x283 += einsum("ia,jkla->ijlk", t1.bb, v.bbbb.ooov)
    x284 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x284 += einsum("ijkl->jkli", x283)
    x284 += einsum("ijkl->kjli", x283) * -1
    x294 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x294 += einsum("ijkl->ikjl", x283) * -1
    x294 += einsum("ijkl->ijkl", x283)
    x295 += einsum("ia,jikl->jkla", t1.bb, x294) * -1
    del x294
    l1new_bb += einsum("abij,ikjb->ak", l2.bbbb, x295) * -1
    del x295
    x427 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x427 += einsum("abij,jkli->klba", l2.bbbb, x283)
    del x283
    x467 -= einsum("ijab->ijab", x427)
    del x427
    x284 += einsum("ijkl->kijl", v.bbbb.oooo) * -1
    x284 += einsum("ijkl->kilj", v.bbbb.oooo)
    x285 += einsum("ia,ijkl->ljka", t1.bb, x284) * -1
    del x284
    l1new_bb += einsum("abij,ikja->bk", l2.bbbb, x285)
    del x285
    x286 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x286 += einsum("abij,ikac->jkbc", l2.abab, t2.abab)
    x289 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x289 += einsum("ijab->ijab", x286)
    x468 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x468 += einsum("ijab->ijab", x286)
    del x286
    x287 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x287 += einsum("abij->jiab", l2.bbbb) * -1
    x287 += einsum("abij->jiba", l2.bbbb)
    x288 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x288 += einsum("ijab,ikac->kjcb", x287, x38)
    del x38
    x289 += einsum("ijab->jiba", x288)
    x468 += einsum("ijab->jiba", x288)
    del x288
    x469 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x469 += einsum("ijab,kica->jkbc", x145, x468)
    del x145
    del x468
    x491 += einsum("ijab->jiba", x469) * -1
    del x469
    x344 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x344 += einsum("wia,ijba->wjb", u11.bb, x287) * -1
    x472 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x472 += einsum("ijab,jkcb->ikac", t2.abab, x287)
    x473 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x473 += einsum("iajb,ikac->jkbc", v.aabb.ovov, x472) * -1
    del x472
    x491 += einsum("ijab->jiba", x473)
    del x473
    l1new_bb += einsum("ia,ijba->bj", x159, x287) * -1
    del x159
    x289 += einsum("wxai,wxjb->ijab", lu12.bb, u12.bb) * 0.5
    l1new_bb += einsum("iabc,jiab->cj", x154, x289) * -1
    del x289
    del x154
    x290 += einsum("ijka->ikja", v.bbbb.ooov) * -1
    x290 += einsum("ijka->kija", v.bbbb.ooov)
    x293 += einsum("ijab,kjlb->ikla", t2.abab, x290) * -1
    x317 += einsum("wxia,jika->xwjk", u12.bb, x290) * -0.5
    del x290
    x291 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x291 += einsum("ia,jabc->ijbc", t1.bb, v.bbaa.ovvv)
    x292 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x292 += einsum("ijab->jiab", x291)
    x525 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x525 += einsum("ijab->jiab", x291)
    del x291
    x292 += einsum("ijab->ijab", v.bbaa.oovv)
    x293 += einsum("ia,jkba->ijkb", t1.aa, x292)
    del x292
    x293 += einsum("ijak->kija", v.bbaa.oovo)
    x293 += einsum("ia,jabk->kjib", t1.bb, v.bbaa.ovvo)
    x293 += einsum("ijab,kbca->ikjc", t2.abab, v.bbaa.ovvv)
    l1new_bb += einsum("abij,ikja->bk", l2.abab, x293) * -1
    del x293
    x296 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x296 += einsum("abij->jiab", l2.bbbb)
    x296 += einsum("abij->jiba", l2.bbbb) * -1
    x297 += einsum("ijab,jkcb->ikac", t2.abab, x296) * -2
    x297 += einsum("wxai,wxjb->jiba", lu12.bb, u12.aa) * -1
    l1new_bb += einsum("iabc,ijab->cj", v.aabb.ovvv, x297) * -0.5
    del x297
    x300 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x300 += einsum("iajb->jiab", v.bbbb.ovov)
    x300 -= einsum("iajb->jiba", v.bbbb.ovov)
    l1new_bb += einsum("ijka,jkab->bi", x299, x300) * -1
    del x299
    del x300
    x301 += einsum("ai,jkba->jikb", l1.bb, t2.abab) * -1
    l1new_bb += einsum("iajb,ikja->bk", v.aabb.ovov, x301)
    del x301
    x304 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x304 += einsum("iabc->ibac", v.bbbb.ovvv)
    x304 += einsum("iabc->ibca", v.bbbb.ovvv) * -1
    x517 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x517 += einsum("ia,jbac->ijbc", t1.bb, x304)
    x518 += einsum("ijab->jiab", x517) * -1
    del x517
    l1new_bb += einsum("ijab,jabc->ci", x303, x304)
    del x304
    del x303
    x305 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x305 += einsum("ia,bacd->icbd", t1.bb, v.bbbb.vvvv)
    x306 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x306 += einsum("iabc->iabc", x305) * -1
    del x305
    x306 += einsum("aibc->iabc", v.bbbb.vovv)
    x307 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x307 += einsum("iabc->iabc", x306) * -1
    x307 += einsum("iabc->ibac", x306)
    del x306
    l1new_bb += einsum("abij,iabc->cj", l2.bbbb, x307) * -1
    del x307
    x308 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x308 += einsum("abij,klba->ijlk", l2.bbbb, t2.bbbb)
    x310 += einsum("ijkl->jilk", x308)
    x311 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x311 += einsum("ia,ijkl->jkla", t1.bb, x310)
    del x310
    x312 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x312 += einsum("ijka->ijka", x311)
    x312 += einsum("ijka->ikja", x311) * -1
    del x311
    l1new_bb += einsum("iajb,kijb->ak", v.bbbb.ovov, x312) * -1
    del x312
    x322 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x322 += einsum("ijkl->jikl", x308) * -1
    x322 += einsum("ijkl->jilk", x308)
    l1new_bb += einsum("ijka,jlik->al", v.bbbb.ooov, x322)
    del x322
    x497 += einsum("ijkl->jilk", x308)
    del x308
    x498 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x498 += einsum("iajb,klji->klab", v.bbbb.ovov, x497)
    del x497
    x499 += einsum("ijab->jiab", x498)
    del x498
    l2new_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    l2new_bbbb += einsum("ijab->abij", x499)
    l2new_bbbb += einsum("ijab->baij", x499) * -1
    del x499
    x313 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x313 += einsum("aibc->iabc", v.aabb.vovv)
    x313 += einsum("ia,bacd->ibcd", t1.aa, v.aabb.vvvv)
    l1new_bb += einsum("abij,iabc->cj", l2.abab, x313)
    del x313
    x314 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x314 += einsum("wx,xia->wia", s2, g.bb.bov)
    x316 += einsum("wia->wia", x314)
    x555 += einsum("wia->wia", x314)
    del x314
    x316 += einsum("wia->wia", gc.bb.bov)
    x317 += einsum("wia,xja->wxji", u11.bb, x316)
    l1new_bb += einsum("wxai,wxji->aj", lu12.bb, x317) * -1
    del x317
    x571 += einsum("wia,ijba->wjb", x316, x52) * -1
    del x52
    x575 += einsum("wia,jiba->wjb", x316, t2.abab) * -1
    del x316
    x318 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x318 += einsum("ai,jkba->ijkb", l1.bb, t2.bbbb)
    x319 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x319 += einsum("ijka->ikja", x318)
    x319 += einsum("ijka->ijka", x318) * -1
    del x318
    l1new_bb += einsum("iajb,kija->bk", v.bbbb.ovov, x319)
    del x319
    x320 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x320 += einsum("ia,jbca->ijcb", t1.aa, v.bbaa.ovvv)
    x321 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x321 += einsum("ijab->ijab", x320)
    x436 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x436 += einsum("ijab->ijab", x320)
    x507 += einsum("ijab->ijab", x320)
    x632 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x632 += einsum("ijab->ijab", x320)
    del x320
    x321 += einsum("iabj->jiba", v.bbaa.ovvo)
    l1new_bb += einsum("ijka,ikab->bj", x1, x321) * -1
    del x321
    x326 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x326 += einsum("wia,wxja->xij", g.bb.bov, u12.bb)
    x333 += einsum("wij->wij", x326)
    del x326
    x327 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x327 += einsum("wia,iajk->wjk", u11.aa, v.aabb.ovoo)
    x333 += einsum("wij->wij", x327)
    del x327
    x330 += einsum("wia->wia", gc.bb.bov)
    x331 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x331 += einsum("ia,wja->wij", t1.bb, x330)
    x333 += einsum("wij->wji", x331)
    del x331
    l1new_bb += einsum("wia,wji->aj", x330, x343) * -1
    del x330
    del x343
    x333 += einsum("wij->wij", gc.bb.boo)
    x571 += einsum("ia,wij->wja", t1.bb, x333)
    l1new_bb += einsum("wai,wji->aj", lu11.bb, x333) * -1
    del x333
    x335 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x335 += einsum("wx,wai->xia", s2, lu11.bb)
    x339 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x339 += einsum("wia->wia", x335)
    del x335
    x336 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x336 += einsum("wia,abij->wjb", u11.aa, l2.abab)
    x339 += einsum("wia->wia", x336)
    x344 += einsum("wia->wia", x336)
    x345 += einsum("ia,wja->wji", t1.bb, x344)
    del x344
    x346 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x346 += einsum("wia->wia", x336)
    del x336
    x337 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x337 -= einsum("abij->jiab", l2.bbbb)
    x337 += einsum("abij->jiba", l2.bbbb)
    x338 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x338 += einsum("wia,ijba->wjb", u11.bb, x337)
    x339 -= einsum("wia->wia", x338)
    x455 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x455 += einsum("wia,wjb->ijab", g.bb.bov, x339)
    x467 += einsum("ijab->ijab", x455)
    del x455
    l1new_bb += einsum("wab,wia->bi", g.bb.bvv, x339)
    del x339
    x346 -= einsum("wia->wia", x338)
    del x338
    x557 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x557 += einsum("wia,wjb->ijab", g.aa.bov, x346)
    l2new_baba += einsum("ijab->baji", x557)
    l2new_abab += einsum("ijab->abij", x557)
    del x557
    l1new_bb += einsum("wij,wja->ai", x139, x346) * -1
    del x346
    del x139
    lu11new_bb -= einsum("wai,ijba->wbj", g.bb.bvo, x337)
    x340 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x340 += einsum("wia,iabc->wbc", u11.aa, v.aabb.ovvv)
    x342 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x342 += einsum("wab->wab", x340)
    x568 += einsum("wab->wab", x340)
    del x340
    x341 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x341 -= einsum("iabc->ibac", v.bbbb.ovvv)
    x341 += einsum("iabc->ibca", v.bbbb.ovvv)
    x342 -= einsum("wia,ibac->wbc", u11.bb, x341)
    x444 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x444 += einsum("ia,ibac->bc", t1.bb, x341)
    del x341
    x445 -= einsum("ab->ab", x444)
    del x444
    x446 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x446 += einsum("ab,acij->ijcb", x445, l2.bbbb)
    del x445
    x467 += einsum("ijab->jiab", x446)
    del x446
    x342 += einsum("wab->wab", gc.bb.bvv)
    l1new_bb += einsum("wai,wab->bi", lu11.bb, x342)
    del x342
    x345 += einsum("ai,wja->wij", l1.bb, u11.bb)
    x345 += einsum("wai,wxja->xij", lu11.bb, u12.bb)
    l1new_bb += einsum("wia,wji->aj", g.bb.bov, x345) * -1
    del x345
    x347 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x347 += einsum("iabj->ijba", v.bbbb.ovvo)
    x347 -= einsum("ijab->ijab", v.bbbb.oovv)
    l1new_bb += einsum("ai,jiab->bj", l1.bb, x347)
    lu11new_bb += einsum("wai,jiab->wbj", lu11.bb, x347)
    del x347
    x348 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x348 += einsum("wia,wbj->ijab", gc.aa.bov, lu11.aa)
    x390 += einsum("ijab->ijab", x348)
    del x348
    x349 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x349 += einsum("ai,jbac->ijbc", l1.aa, v.aaaa.ovvv)
    x390 -= einsum("ijab->ijab", x349)
    del x349
    x355 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x355 -= einsum("abij->jiab", l2.aaaa)
    x355 += einsum("abij->jiba", l2.aaaa)
    x357 += einsum("iabj->jiba", v.aaaa.ovvo)
    x357 -= einsum("ijab->jiab", v.aaaa.oovv)
    x358 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x358 += einsum("ijab,ikac->jkbc", x355, x357)
    del x355
    del x357
    x390 += einsum("ijab->ijab", x358)
    del x358
    x359 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x359 -= einsum("ijab->jiab", t2.bbbb)
    x359 += einsum("ijab->jiba", t2.bbbb)
    x360 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x360 += einsum("iajb,jkcb->ikac", v.aabb.ovov, x359)
    x361 -= einsum("ijab->ijab", x360)
    del x360
    x595 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x595 += einsum("wia,ijba->wjb", g.bb.bov, x359)
    del x359
    x596 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x596 -= einsum("wia->wia", x595)
    del x595
    x361 += einsum("iabj->ijab", v.aabb.ovvo)
    x362 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x362 += einsum("abij,kjcb->ikac", l2.abab, x361)
    del x361
    x390 += einsum("ijab->ijab", x362)
    del x362
    x364 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x364 += einsum("ijka->ikja", v.aaaa.ooov)
    x364 -= einsum("ijka->kija", v.aaaa.ooov)
    x365 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x365 += einsum("ijka,lkib->jlab", x363, x364)
    del x363
    x390 += einsum("ijab->ijab", x365)
    del x365
    l2new_abab += einsum("ijka,kjlb->abil", x364, x55)
    del x364
    x366 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x366 += einsum("wia,wib->ab", g.aa.bov, u11.aa)
    x367 -= einsum("ab->ba", x366)
    x543 += einsum("ab->ba", x366) * -1
    del x366
    x367 += einsum("ab->ab", f.aa.vv)
    x368 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x368 += einsum("ab,acij->ijcb", x367, l2.aaaa)
    del x367
    x390 -= einsum("ijab->jiba", x368)
    del x368
    x372 += einsum("ia->ia", f.aa.ov)
    x373 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x373 += einsum("ia,ja->ij", t1.aa, x372)
    x374 += einsum("ij->ji", x373)
    del x373
    x385 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x385 += einsum("ia,jkib->jkba", x372, x56)
    del x56
    x390 += einsum("ijab->ijba", x385)
    del x385
    x390 += einsum("ai,jb->jiba", l1.aa, x372)
    del x372
    x374 += einsum("ij->ij", f.aa.oo)
    x375 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x375 += einsum("ij,abjk->kiab", x374, l2.aaaa)
    del x374
    x390 += einsum("ijab->jiba", x375)
    del x375
    x376 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x376 -= einsum("ijka->ikja", v.aaaa.ooov)
    x376 += einsum("ijka->kija", v.aaaa.ooov)
    x377 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x377 += einsum("ia,jika->jk", t1.aa, x376)
    x378 -= einsum("ij->ij", x377)
    del x377
    x379 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x379 += einsum("ij,abjk->kiab", x378, l2.aaaa)
    del x378
    x390 -= einsum("ijab->ijba", x379)
    del x379
    l2new_baba -= einsum("ijka,kjlb->bali", x376, x55)
    del x55
    lu12new_aa -= einsum("ijka,wxkj->xwai", x376, x54)
    del x376
    del x54
    x386 -= einsum("ijka->jika", v.aaaa.ooov)
    x387 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x387 += einsum("ai,ijkb->jkab", l1.aa, x386)
    del x386
    x390 += einsum("ijab->ijab", x387)
    del x387
    l2new_aaaa += einsum("ijab->abij", x390)
    l2new_aaaa -= einsum("ijab->baij", x390)
    l2new_aaaa -= einsum("ijab->abji", x390)
    l2new_aaaa += einsum("ijab->baji", x390)
    del x390
    x391 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x391 += einsum("ijab,kcjb->ikac", t2.abab, v.aabb.ovov)
    x393 += einsum("ijab->ijab", x391)
    x394 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x394 += einsum("ijab,ikac->jkbc", x34, x393)
    del x393
    del x34
    x414 += einsum("ijab->ijab", x394)
    del x394
    x513 += einsum("ijab->jiab", x391)
    del x391
    x398 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x398 += einsum("wxia,jbia->wxjb", u12.bb, v.aabb.ovov)
    x400 += einsum("wxia->xwia", x398)
    del x398
    x401 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x401 += einsum("wxai,wxjb->ijab", lu12.aa, x400) * 0.5
    x414 += einsum("ijab->ijab", x401)
    del x401
    x538 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x538 += einsum("wxai,wxjb->jiba", lu12.bb, x400) * 0.5
    del x400
    l2new_baba += einsum("ijab->baji", x538)
    l2new_abab += einsum("ijab->abij", x538)
    del x538
    x407 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x407 += einsum("ijab,icjb->ac", t2.abab, v.aabb.ovov)
    x409 += einsum("ab->ab", x407)
    x410 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x410 += einsum("ab,acij->ijcb", x409, l2.aaaa)
    del x409
    x414 += einsum("ijab->jiab", x410) * -1
    del x410
    l2new_aaaa += einsum("ijab->abij", x414)
    l2new_aaaa += einsum("ijab->baij", x414) * -1
    l2new_aaaa += einsum("ijab->abji", x414) * -1
    l2new_aaaa += einsum("ijab->baji", x414)
    del x414
    x543 += einsum("ab->ab", x407) * -1
    del x407
    x415 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x415 += einsum("abij,kilj->klab", l2.aaaa, v.aaaa.oooo)
    x417 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x417 += einsum("ijab->jiba", x415)
    del x415
    x416 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x416 += einsum("abij,bcad->ijdc", l2.aaaa, v.aaaa.vvvv)
    x417 += einsum("ijab->jiba", x416)
    del x416
    l2new_aaaa += einsum("ijab->baij", x417) * -1
    l2new_aaaa += einsum("ijab->abij", x417)
    del x417
    x423 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x423 += einsum("wia,wbj->ijab", gc.bb.bov, lu11.bb)
    x467 += einsum("ijab->ijab", x423)
    del x423
    x424 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x424 += einsum("ai,jbac->ijbc", l1.bb, v.bbbb.ovvv)
    x467 -= einsum("ijab->ijab", x424)
    del x424
    x430 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x430 += einsum("iabc->ibac", v.bbbb.ovvv)
    x430 -= einsum("iabc->ibca", v.bbbb.ovvv)
    x431 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x431 += einsum("ia,jbac->ijbc", t1.bb, x430)
    del x430
    x432 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x432 -= einsum("ijab->ijab", x431)
    x631 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x631 -= einsum("ijab->jiab", x431)
    del x431
    x432 += einsum("iabj->jiba", v.bbbb.ovvo)
    x432 -= einsum("ijab->jiab", v.bbbb.oovv)
    x433 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x433 += einsum("ijab,ikac->jkbc", x337, x432)
    del x432
    del x337
    x467 += einsum("ijab->ijab", x433)
    del x433
    x434 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x434 += einsum("ijab->jiab", t2.aaaa)
    x434 -= einsum("ijab->jiba", t2.aaaa)
    x435 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x435 += einsum("iajb,ikac->kjcb", v.aabb.ovov, x434)
    x436 -= einsum("ijab->ijab", x435)
    del x435
    x591 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x591 += einsum("wia,ijab->wjb", g.aa.bov, x434)
    del x434
    x592 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x592 -= einsum("wia->wia", x591)
    del x591
    x436 += einsum("iabj->jiba", v.bbaa.ovvo)
    x437 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x437 += einsum("abij,ikac->jkbc", l2.abab, x436)
    del x436
    x467 += einsum("ijab->ijab", x437)
    del x437
    x439 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x439 += einsum("ijka->ikja", v.bbbb.ooov)
    x439 -= einsum("ijka->kija", v.bbbb.ooov)
    x440 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x440 += einsum("ijka,lkjb->ilab", x438, x439)
    del x438
    del x439
    x467 += einsum("ijab->ijab", x440)
    del x440
    x441 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x441 += einsum("wia,wib->ab", g.bb.bov, u11.bb)
    x442 -= einsum("ab->ba", x441)
    x540 += einsum("ab->ba", x441) * -1
    del x441
    x442 += einsum("ab->ab", f.bb.vv)
    x443 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x443 += einsum("ab,acij->ijcb", x442, l2.bbbb)
    del x442
    x467 -= einsum("ijab->jiba", x443)
    del x443
    x447 += einsum("ia->ia", f.bb.ov)
    x448 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x448 += einsum("ia,ja->ij", t1.bb, x447)
    x449 += einsum("ij->ji", x448)
    del x448
    x462 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x462 += einsum("ia,jkib->jkba", x447, x193)
    del x193
    x467 += einsum("ijab->ijba", x462)
    del x462
    x467 += einsum("ai,jb->jiba", l1.bb, x447)
    del x447
    x449 += einsum("ij->ij", f.bb.oo)
    x450 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x450 += einsum("ij,abjk->kiab", x449, l2.bbbb)
    del x449
    x467 += einsum("ijab->jiba", x450)
    del x450
    x451 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x451 -= einsum("ijka->ikja", v.bbbb.ooov)
    x451 += einsum("ijka->kija", v.bbbb.ooov)
    x452 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x452 += einsum("ia,jika->jk", t1.bb, x451)
    x453 -= einsum("ij->ij", x452)
    del x452
    x454 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x454 += einsum("ij,abjk->kiab", x453, l2.bbbb)
    del x453
    x467 -= einsum("ijab->ijba", x454)
    del x454
    x536 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x536 += einsum("ijka,lkjb->ilab", x1, x451)
    del x1
    l2new_baba -= einsum("ijab->baji", x536)
    l2new_abab -= einsum("ijab->abij", x536)
    del x536
    lu12new_bb -= einsum("wxij,kjia->xwak", x190, x451)
    del x451
    del x190
    x456 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x456 -= einsum("iajb->jiab", v.bbbb.ovov)
    x456 += einsum("iajb->jiba", v.bbbb.ovov)
    x457 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x457 += einsum("wia,ijba->wjb", u11.bb, x456)
    x458 -= einsum("wia->wia", x457)
    x459 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x459 += einsum("wai,wjb->ijab", lu11.bb, x458)
    del x458
    x467 += einsum("ijab->ijab", x459)
    del x459
    x555 -= einsum("wia->wia", x457)
    del x457
    x465 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x465 += einsum("ia,ijba->jb", t1.bb, x456)
    del x456
    x466 -= einsum("ia->ia", x465)
    x467 += einsum("ai,jb->ijab", l1.bb, x466)
    del x466
    x563 -= einsum("ia->ia", x465)
    del x465
    x463 -= einsum("ijka->jika", v.bbbb.ooov)
    x464 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x464 += einsum("ai,ijkb->jkab", l1.bb, x463)
    del x463
    x467 += einsum("ijab->ijab", x464)
    del x464
    l2new_bbbb += einsum("ijab->abij", x467)
    l2new_bbbb -= einsum("ijab->baij", x467)
    l2new_bbbb -= einsum("ijab->abji", x467)
    l2new_bbbb += einsum("ijab->baji", x467)
    del x467
    x474 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x474 += einsum("wxia,iajb->wxjb", u12.aa, v.aabb.ovov)
    x476 += einsum("wxia->xwia", x474)
    del x474
    x477 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x477 += einsum("wxai,wxjb->ijab", lu12.bb, x476) * 0.5
    x491 += einsum("ijab->ijab", x477)
    del x477
    x537 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x537 += einsum("wxai,wxjb->ijab", lu12.aa, x476) * 0.5
    del x476
    l2new_baba += einsum("ijab->baji", x537)
    l2new_abab += einsum("ijab->abij", x537)
    del x537
    x484 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x484 += einsum("ijab,iajc->bc", t2.abab, v.aabb.ovov)
    x486 += einsum("ab->ab", x484)
    x487 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x487 += einsum("ab,acij->ijcb", x486, l2.bbbb)
    del x486
    x491 += einsum("ijab->jiab", x487) * -1
    del x487
    l2new_bbbb += einsum("ijab->abij", x491)
    l2new_bbbb += einsum("ijab->baij", x491) * -1
    l2new_bbbb += einsum("ijab->abji", x491) * -1
    l2new_bbbb += einsum("ijab->baji", x491)
    del x491
    x540 += einsum("ab->ab", x484) * -1
    del x484
    x492 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x492 += einsum("abij,kilj->klab", l2.bbbb, v.bbbb.oooo)
    x494 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x494 += einsum("ijab->jiba", x492)
    del x492
    x493 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x493 += einsum("abij,bcad->ijdc", l2.bbbb, v.bbbb.vvvv)
    x494 += einsum("ijab->jiba", x493)
    del x493
    l2new_bbbb += einsum("ijab->baij", x494) * -1
    l2new_bbbb += einsum("ijab->abij", x494)
    del x494
    x501 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x501 += einsum("abij,acbd->ijcd", l2.abab, v.aabb.vvvv)
    l2new_baba += einsum("ijab->baji", x501)
    l2new_abab += einsum("ijab->abij", x501)
    del x501
    x502 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x502 += einsum("ai,jbac->ijcb", l1.aa, v.bbaa.ovvv)
    l2new_baba += einsum("ijab->baji", x502)
    l2new_abab += einsum("ijab->abij", x502)
    del x502
    x504 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x504 += einsum("ai,jbac->jibc", l1.bb, v.aabb.ovvv)
    l2new_baba += einsum("ijab->baji", x504)
    l2new_abab += einsum("ijab->abij", x504)
    del x504
    x507 += einsum("iabj->jiba", v.bbaa.ovvo)
    x508 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x508 += einsum("ijab,ikac->jkbc", x245, x507)
    del x507
    del x245
    l2new_baba += einsum("ijab->baji", x508) * -1
    l2new_abab += einsum("ijab->abij", x508) * -1
    del x508
    x511 += einsum("iabj->ijab", v.aabb.ovvo)
    l2new_baba += einsum("ijab,kicb->acjk", x296, x511)
    del x296
    l2new_abab += einsum("ijab,kicb->cakj", x287, x511) * -1
    del x511
    del x287
    x513 += einsum("iabj->ijba", v.aaaa.ovvo)
    x513 += einsum("ijab->ijab", v.aaaa.oovv) * -1
    x514 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x514 += einsum("abij,kiac->kjcb", l2.abab, x513)
    del x513
    l2new_baba += einsum("ijab->baji", x514)
    l2new_abab += einsum("ijab->abij", x514)
    del x514
    x515 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x515 += einsum("ijab,iakc->jkbc", t2.abab, v.aabb.ovov)
    x518 += einsum("ijab->jiab", x515)
    del x515
    x518 += einsum("iabj->ijba", v.bbbb.ovvo)
    x518 += einsum("ijab->ijab", v.bbbb.oovv) * -1
    x519 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x519 += einsum("abij,kjbc->ikac", l2.abab, x518)
    del x518
    l2new_baba += einsum("ijab->baji", x519)
    l2new_abab += einsum("ijab->abij", x519)
    del x519
    x520 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x520 += einsum("ia,jabc->ijbc", t1.aa, v.aabb.ovvv)
    x522 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x522 += einsum("ijab->jiab", x520)
    del x520
    x521 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x521 += einsum("ijab,kajc->ikbc", t2.abab, v.aabb.ovov)
    x522 += einsum("ijab->jiab", x521) * -1
    del x521
    x522 += einsum("ijab->ijab", v.aabb.oovv)
    x523 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x523 += einsum("abij,kibc->kjac", l2.abab, x522)
    del x522
    l2new_baba += einsum("ijab->baji", x523) * -1
    l2new_abab += einsum("ijab->abij", x523) * -1
    del x523
    x524 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x524 += einsum("ijab,ickb->jkac", t2.abab, v.aabb.ovov)
    x525 += einsum("ijab->jiab", x524) * -1
    del x524
    x525 += einsum("ijab->ijab", v.bbaa.oovv)
    x526 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x526 += einsum("abij,kjac->ikcb", l2.abab, x525)
    del x525
    l2new_baba += einsum("ijab->baji", x526) * -1
    l2new_abab += einsum("ijab->abij", x526) * -1
    del x526
    x528 += einsum("ijkl->ijkl", v.aabb.oooo)
    x529 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x529 += einsum("abij,kilj->klab", l2.abab, x528)
    del x528
    l2new_baba += einsum("ijab->baji", x529)
    l2new_abab += einsum("ijab->abij", x529)
    del x529
    x540 += einsum("ab->ab", f.bb.vv)
    x541 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x541 += einsum("ab,caij->ijcb", x540, l2.abab)
    l2new_baba += einsum("ijab->baji", x541)
    l2new_abab += einsum("ijab->abij", x541)
    del x541
    x571 += einsum("ab,wib->wia", x540, u11.bb) * -1
    lu12new_bb += einsum("ab,wxai->xwbi", x540, lu12.bb)
    del x540
    x543 += einsum("ab->ab", f.aa.vv)
    x544 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x544 += einsum("ab,acij->ijbc", x543, l2.abab)
    l2new_baba += einsum("ijab->baji", x544)
    l2new_abab += einsum("ijab->abij", x544)
    del x544
    x575 += einsum("ab,wib->wia", x543, u11.aa) * -1
    lu12new_aa += einsum("ab,wxai->xwbi", x543, lu12.aa)
    del x543
    x553 += einsum("wia->wia", gc.aa.bov)
    x554 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x554 += einsum("wai,wjb->jiba", lu11.bb, x553)
    del x553
    l2new_baba += einsum("ijab->baji", x554)
    l2new_abab += einsum("ijab->abij", x554)
    del x554
    x555 += einsum("wia->wia", gc.bb.bov)
    x556 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x556 += einsum("wai,wjb->ijab", lu11.aa, x555)
    del x555
    l2new_baba += einsum("ijab->baji", x556)
    l2new_abab += einsum("ijab->abij", x556)
    del x556
    x559 += einsum("ijka->ijka", v.aabb.ooov)
    x560 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x560 += einsum("ai,jikb->jkab", l1.aa, x559)
    del x559
    l2new_baba -= einsum("ijab->baji", x560)
    l2new_abab -= einsum("ijab->abij", x560)
    del x560
    x561 += einsum("iajk->ijka", v.aabb.ovoo)
    x562 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x562 += einsum("ai,jkib->jkba", l1.bb, x561)
    del x561
    l2new_baba -= einsum("ijab->baji", x562)
    l2new_abab -= einsum("ijab->abij", x562)
    del x562
    x563 += einsum("ia->ia", f.bb.ov)
    x578 += einsum("ia,wia->w", x563, u11.bb)
    l2new_baba += einsum("ai,jb->baji", l1.aa, x563)
    l2new_abab += einsum("ai,jb->abij", l1.aa, x563)
    del x563
    x564 += einsum("ia->ia", f.aa.ov)
    x578 += einsum("ia,wia->w", x564, u11.aa)
    l2new_baba += einsum("ai,jb->abij", l1.bb, x564)
    l2new_abab += einsum("ai,jb->baji", l1.bb, x564)
    del x564
    x568 += einsum("wab->wab", gc.bb.bvv)
    x571 += einsum("ia,wba->wib", t1.bb, x568) * -1
    del x568
    x569 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x569 += einsum("ia,wba->wib", t1.bb, g.bb.bvv)
    x570 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x570 += einsum("wia->wia", x569)
    x596 += einsum("wia->wia", x569)
    del x569
    x570 += einsum("wai->wia", g.bb.bvo)
    x571 += einsum("wx,wia->xia", s2, x570) * -1
    del x570
    x571 += einsum("wai->wia", gc.bb.bvo) * -1
    x571 += einsum("wab,wxib->xia", g.bb.bvv, u12.bb) * -1
    x571 += einsum("wia,iabj->wjb", u11.aa, v.aabb.ovvo) * -1
    ls1new += einsum("wia,wxai->x", x571, lu12.bb) * -1
    del x571
    x572 += einsum("wab->wab", gc.aa.bvv)
    x575 += einsum("ia,wba->wib", t1.aa, x572) * -1
    del x572
    x573 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x573 += einsum("ia,wba->wib", t1.aa, g.aa.bvv)
    x574 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x574 += einsum("wia->wia", x573)
    x592 += einsum("wia->wia", x573)
    del x573
    x574 += einsum("wai->wia", g.aa.bvo)
    x575 += einsum("wx,wia->xia", s2, x574) * -1
    del x574
    x575 += einsum("wai->wia", gc.aa.bvo) * -1
    x575 += einsum("wab,wxib->xia", g.aa.bvv, u12.aa) * -1
    x575 += einsum("wia,iabj->wjb", u11.bb, v.bbaa.ovvo) * -1
    ls1new += einsum("wia,wxai->x", x575, lu12.aa) * -1
    del x575
    x577 += einsum("w->w", G)
    x578 += einsum("w,wx->x", x577, s2)
    x605 += einsum("w,x->xw", ls1, x577)
    del x577
    x578 += einsum("w->w", G)
    ls1new += einsum("w,wx->x", x578, ls2)
    del x578
    x579 = np.zeros((nbos, nbos), dtype=types[float])
    x579 += einsum("wx,yx->wy", ls2, w)
    x605 += einsum("wx->wx", x579)
    del x579
    x580 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x580 += einsum("wia,wxbi->xba", u11.aa, lu12.aa)
    x581 = np.zeros((nbos, nbos), dtype=types[float])
    x581 += einsum("wab,xab->wx", g.aa.bvv, x580)
    x605 += einsum("wx->wx", x581)
    del x581
    x608 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x608 += einsum("wia,xba->wxib", g.aa.bov, x580)
    del x580
    x612 += einsum("wxia->wxia", x608)
    del x608
    x583 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x583 += einsum("wia,xwbi->xba", u11.bb, lu12.bb)
    x584 = np.zeros((nbos, nbos), dtype=types[float])
    x584 += einsum("wab,xab->wx", g.bb.bvv, x583)
    x605 += einsum("wx->wx", x584)
    del x584
    x626 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x626 += einsum("wia,xba->wxib", g.bb.bov, x583)
    del x583
    x630 += einsum("wxia->wxia", x626)
    del x626
    x590 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x590 += einsum("wia,jiba->wjb", g.bb.bov, t2.abab)
    x592 += einsum("wia->wia", x590)
    del x590
    x592 += einsum("wai->wia", g.aa.bvo)
    x593 = np.zeros((nbos, nbos), dtype=types[float])
    x593 += einsum("wai,xia->wx", lu11.aa, x592)
    del x592
    x605 += einsum("wx->xw", x593)
    del x593
    x594 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x594 += einsum("wia,ijab->wjb", g.aa.bov, t2.abab)
    x596 += einsum("wia->wia", x594)
    del x594
    x596 += einsum("wai->wia", g.bb.bvo)
    x597 = np.zeros((nbos, nbos), dtype=types[float])
    x597 += einsum("wai,xia->wx", lu11.bb, x596)
    del x596
    x605 += einsum("wx->xw", x597)
    del x597
    x598 += einsum("wij->wij", g.aa.boo)
    x599 = np.zeros((nbos, nbos), dtype=types[float])
    x599 += einsum("wij,xji->wx", x167, x598)
    del x167
    x605 -= einsum("wx->xw", x599)
    del x599
    x600 = np.zeros((nbos, nbos), dtype=types[float])
    x600 += einsum("wij,xji->wx", x168, x598)
    del x168
    x605 -= einsum("wx->xw", x600)
    del x600
    x609 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x609 += einsum("wai,xji->wxja", lu11.aa, x598)
    del x598
    x612 += einsum("wxia->xwia", x609)
    del x609
    x601 += einsum("wij->wij", g.bb.boo)
    x602 = np.zeros((nbos, nbos), dtype=types[float])
    x602 += einsum("wij,xji->wx", x196, x601)
    del x196
    x605 -= einsum("wx->xw", x602)
    del x602
    ls2new += einsum("wx->wx", x605)
    ls2new += einsum("wx->xw", x605)
    del x605
    x627 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x627 += einsum("wai,xji->wxja", lu11.bb, x601)
    del x601
    x630 += einsum("wxia->xwia", x627)
    del x627
    x606 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x606 += einsum("wx,xyai->ywia", w, lu12.aa)
    x612 -= einsum("wxia->wxia", x606)
    del x606
    x607 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x607 += einsum("wab,xai->wxib", g.aa.bvv, lu11.aa)
    x612 -= einsum("wxia->wxia", x607)
    del x607
    lu12new_aa -= einsum("wxia->wxai", x612)
    lu12new_aa -= einsum("wxia->xwai", x612)
    del x612
    x613 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x613 += einsum("wxai,jiba->wxjb", lu12.bb, t2.abab)
    x615 += einsum("wxia->xwia", x613)
    del x613
    lu12new_aa += einsum("ijab,wxia->xwbj", x12, x615) * -1
    del x12
    lu12new_bb += einsum("iajb,wxia->xwbj", v.aabb.ovov, x615)
    del x615
    x616 += einsum("iabj->ijba", v.aaaa.ovvo)
    x616 -= einsum("ijab->ijab", v.aaaa.oovv)
    lu12new_aa += einsum("wxai,jiab->xwbj", lu12.aa, x616)
    del x616
    x617 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x617 += einsum("wxai,ijab->wxjb", lu12.aa, t2.abab)
    x619 += einsum("wxia->xwia", x617)
    del x617
    lu12new_aa += einsum("iajb,wxjb->xwai", v.aabb.ovov, x619)
    lu12new_bb += einsum("wxia,ijba->xwbj", x619, x93) * -1
    del x619
    del x93
    x620 += einsum("iabj->ijab", v.aabb.ovvo)
    lu12new_aa += einsum("wxai,jiba->xwbj", lu12.bb, x620)
    del x620
    x621 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x621 += einsum("wia,xyai->xyw", u11.aa, lu12.aa)
    x623 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x623 += einsum("wxy->xwy", x621)
    del x621
    x622 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x622 += einsum("wia,xyai->xyw", u11.bb, lu12.bb)
    x623 += einsum("wxy->xwy", x622)
    del x622
    lu12new_aa += einsum("wia,xyw->yxai", g.aa.bov, x623)
    lu12new_bb += einsum("wia,xyw->yxai", g.bb.bov, x623)
    del x623
    x624 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x624 += einsum("wx,xyai->ywia", w, lu12.bb)
    x630 -= einsum("wxia->wxia", x624)
    del x624
    x625 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x625 += einsum("wab,xai->wxib", g.bb.bvv, lu11.bb)
    x630 -= einsum("wxia->wxia", x625)
    del x625
    lu12new_bb -= einsum("wxia->wxai", x630)
    lu12new_bb -= einsum("wxia->xwai", x630)
    del x630
    x631 += einsum("iabj->ijba", v.bbbb.ovvo)
    x631 -= einsum("ijab->ijab", v.bbbb.oovv)
    lu12new_bb += einsum("wxai,jiab->xwbj", lu12.bb, x631)
    del x631
    x632 += einsum("iabj->jiba", v.bbaa.ovvo)
    lu12new_bb += einsum("wxai,ijab->xwbj", lu12.aa, x632)
    del x632
    l1new_aa += einsum("w,wia->ai", ls1, gc.aa.bov)
    l1new_aa += einsum("ai,jbai->bj", l1.bb, v.aabb.ovvo)
    l1new_aa += einsum("ia->ai", f.aa.ov)
    l1new_bb += einsum("w,wia->ai", ls1, gc.bb.bov)
    l1new_bb += einsum("ai,jbai->bj", l1.aa, v.bbaa.ovvo)
    l1new_bb += einsum("ia->ai", f.bb.ov)
    l2new_aaaa -= einsum("iajb->abji", v.aaaa.ovov)
    l2new_aaaa += einsum("iajb->baji", v.aaaa.ovov)
    l2new_bbbb -= einsum("iajb->abji", v.bbbb.ovov)
    l2new_bbbb += einsum("iajb->baji", v.bbbb.ovov)
    l2new_baba += einsum("iajb->baji", v.aabb.ovov)
    l2new_abab += einsum("iajb->abij", v.aabb.ovov)
    ls1new += einsum("w,xw->x", ls1, w)
    ls1new += einsum("ai,wai->w", l1.aa, g.aa.bvo)
    ls1new += einsum("ai,wai->w", l1.bb, g.bb.bvo)
    ls1new += einsum("w->w", G)
    lu11new_aa += einsum("w,xwai->xai", G, lu12.aa)
    lu11new_aa += einsum("wx,xia->wai", ls2, gc.aa.bov)
    lu11new_aa += einsum("wx,xai->wai", w, lu11.aa)
    lu11new_aa -= einsum("ij,waj->wai", f.aa.oo, lu11.aa)
    lu11new_aa -= einsum("ai,wji->waj", l1.aa, g.aa.boo)
    lu11new_aa += einsum("ab,wai->wbi", f.aa.vv, lu11.aa)
    lu11new_aa += einsum("ai,wab->wbi", l1.aa, g.aa.bvv)
    lu11new_aa += einsum("wab,xwai->xbi", gc.aa.bvv, lu12.aa)
    lu11new_aa += einsum("wia->wai", g.aa.bov)
    lu11new_aa -= einsum("wij,xwaj->xai", gc.aa.boo, lu12.aa)
    lu11new_aa += einsum("wai,baji->wbj", g.bb.bvo, l2.abab)
    lu11new_aa += einsum("wai,jbai->wbj", lu11.bb, v.aabb.ovvo)
    lu11new_bb += einsum("w,xwai->xai", G, lu12.bb)
    lu11new_bb += einsum("wx,xia->wai", ls2, gc.bb.bov)
    lu11new_bb += einsum("wx,xai->wai", w, lu11.bb)
    lu11new_bb += einsum("ab,wai->wbi", f.bb.vv, lu11.bb)
    lu11new_bb += einsum("ai,wab->wbi", l1.bb, g.bb.bvv)
    lu11new_bb += einsum("wai,abij->wbj", g.aa.bvo, l2.abab)
    lu11new_bb += einsum("wai,jbai->wbj", lu11.aa, v.bbaa.ovvo)
    lu11new_bb += einsum("wia->wai", g.bb.bov)
    lu11new_bb -= einsum("wij,xwaj->xai", gc.bb.boo, lu12.bb)
    lu11new_bb -= einsum("ij,waj->wai", f.bb.oo, lu11.bb)
    lu11new_bb -= einsum("ai,wji->waj", l1.bb, g.bb.boo)
    lu11new_bb += einsum("wab,xwai->xbi", gc.bb.bvv, lu12.bb)

    l1new.aa = l1new_aa
    l1new.bb = l1new_bb
    l2new.abab = l2new_abab
    l2new.baba = l2new_baba
    l2new.aaaa = l2new_aaaa
    l2new.bbbb = l2new_bbbb
    lu11new.aa = lu11new_aa
    lu11new.bb = lu11new_bb
    lu12new.aa = lu12new_aa
    lu12new.bb = lu12new_bb

    return {"l1new": l1new, "l2new": l2new, "ls1new": ls1new, "ls2new": ls2new, "lu11new": lu11new, "lu12new": lu12new}

def make_rdm1_f(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, lu12=None, **kwargs):
    rdm1_f = Namespace()

    # Get boson coupling creation array:
    gc = Namespace(
        aa = Namespace(
            boo=g.aa.boo.transpose(0, 2, 1),
            bov=g.aa.bvo.transpose(0, 2, 1),
            bvo=g.aa.bov.transpose(0, 2, 1),
            bvv=g.aa.bvv.transpose(0, 2, 1),
        ),
        bb = Namespace(
            boo=g.bb.boo.transpose(0, 2, 1),
            bov=g.bb.bvo.transpose(0, 2, 1),
            bvo=g.bb.bov.transpose(0, 2, 1),
            bvv=g.bb.bvv.transpose(0, 2, 1),
        ),
    )

    delta_oo = Namespace()
    delta_oo.aa = np.eye(nocc[0])
    delta_oo.bb = np.eye(nocc[1])
    delta_vv = Namespace()
    delta_vv.aa = np.eye(nvir[0])
    delta_vv.bb = np.eye(nvir[1])

    # 1RDM
    x0 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x0 += einsum("abij,kjab->ik", l2.abab, t2.abab)
    x17 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x17 += einsum("ij->ij", x0) * 2
    rdm1_f_oo_aa = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    rdm1_f_oo_aa += einsum("ij->ij", x0) * -1
    del x0
    x1 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x1 += einsum("wxai,wxja->ij", lu12.aa, u12.aa)
    x17 += einsum("ij->ij", x1)
    rdm1_f_oo_aa += einsum("ij->ij", x1) * -0.5
    del x1
    x2 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x2 += einsum("ai,ja->ij", l1.aa, t1.aa)
    x17 += einsum("ij->ij", x2) * 2
    rdm1_f_oo_aa -= einsum("ij->ij", x2)
    del x2
    x3 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x3 += einsum("wai,wja->ij", lu11.aa, u11.aa)
    x17 += einsum("ij->ij", x3) * 2
    rdm1_f_oo_aa -= einsum("ij->ij", x3)
    del x3
    x4 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x4 += einsum("ijab->jiab", t2.aaaa) * -1
    x4 += einsum("ijab->jiba", t2.aaaa)
    rdm1_f_oo_aa += einsum("abij,ikab->jk", l2.aaaa, x4) * -1
    del x4
    x5 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x5 += einsum("wxai,wxja->ij", lu12.bb, u12.bb)
    x24 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x24 += einsum("ij->ij", x5) * 0.5
    rdm1_f_oo_bb = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    rdm1_f_oo_bb += einsum("ij->ij", x5) * -0.5
    del x5
    x6 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x6 += einsum("wai,wja->ij", lu11.bb, u11.bb)
    x24 += einsum("ij->ij", x6)
    rdm1_f_oo_bb -= einsum("ij->ij", x6)
    del x6
    x7 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x7 += einsum("ai,ja->ij", l1.bb, t1.bb)
    x24 += einsum("ij->ij", x7)
    rdm1_f_oo_bb -= einsum("ij->ij", x7)
    del x7
    x8 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x8 += einsum("abij,ikab->jk", l2.abab, t2.abab)
    x24 += einsum("ij->ij", x8)
    rdm1_f_oo_bb += einsum("ij->ij", x8) * -1
    del x8
    x9 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x9 += einsum("ijab->jiab", t2.bbbb) * -1
    x9 += einsum("ijab->jiba", t2.bbbb)
    x24 += einsum("abij,ikba->jk", l2.bbbb, x9) * -1
    rdm1_f_vo_bb = np.zeros((nvir[1], nocc[1]), dtype=types[float])
    rdm1_f_vo_bb += einsum("ia,ij->aj", t1.bb, x24) * -1
    del x24
    rdm1_f_oo_bb += einsum("abij,ikab->jk", l2.bbbb, x9) * -1
    del x9
    x10 = np.zeros((nbos, nbos, nocc[0], nocc[0]), dtype=types[float])
    x10 += einsum("ia,wxaj->wxji", t1.aa, lu12.aa)
    rdm1_f_vo_aa = np.zeros((nvir[0], nocc[0]), dtype=types[float])
    rdm1_f_vo_aa += einsum("wxia,wxij->aj", u12.aa, x10) * -0.5
    del x10
    x11 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x11 += einsum("ia,abjk->jikb", t1.aa, l2.abab)
    rdm1_f_vo_aa += einsum("ijab,ikjb->ak", t2.abab, x11) * -1
    del x11
    x12 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x12 += einsum("ia,bajk->jkib", t1.aa, l2.aaaa)
    x13 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x13 += einsum("ijka->ijka", x12)
    x13 += einsum("ijka->jika", x12) * -1
    del x12
    rdm1_f_vo_aa += einsum("ijab,jikb->ak", t2.aaaa, x13) * -1
    del x13
    x14 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x14 += einsum("ijab->jiab", t2.aaaa)
    x14 -= einsum("ijab->jiba", t2.aaaa)
    rdm1_f_vo_aa -= einsum("ai,ijab->bj", l1.aa, x14)
    del x14
    x15 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x15 += einsum("ia,waj->wji", t1.aa, lu11.aa)
    x15 += einsum("wia,wxaj->xji", u11.aa, lu12.aa)
    rdm1_f_vo_aa -= einsum("wia,wij->aj", u11.aa, x15)
    del x15
    x16 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x16 += einsum("ijab->jiab", t2.aaaa)
    x16 += einsum("ijab->jiba", t2.aaaa) * -1
    x17 += einsum("abij,ikab->jk", l2.aaaa, x16) * -2
    del x16
    rdm1_f_vo_aa += einsum("ia,ij->aj", t1.aa, x17) * -0.5
    del x17
    x18 = np.zeros((nbos, nbos, nocc[1], nocc[1]), dtype=types[float])
    x18 += einsum("ia,wxaj->wxji", t1.bb, lu12.bb)
    rdm1_f_vo_bb += einsum("wxia,wxij->aj", u12.bb, x18) * -0.5
    del x18
    x19 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x19 += einsum("ia,bajk->jkib", t1.bb, l2.abab)
    rdm1_f_vo_bb += einsum("ijab,ijka->bk", t2.abab, x19) * -1
    del x19
    x20 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x20 += einsum("ia,abjk->kjib", t1.bb, l2.bbbb)
    x21 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x21 += einsum("ijka->ijka", x20)
    x21 += einsum("ijka->jika", x20) * -1
    del x20
    rdm1_f_vo_bb += einsum("ijab,ijka->bk", t2.bbbb, x21) * -1
    del x21
    x22 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x22 += einsum("ijab->jiab", t2.bbbb)
    x22 -= einsum("ijab->jiba", t2.bbbb)
    rdm1_f_vo_bb -= einsum("ai,ijab->bj", l1.bb, x22)
    del x22
    x23 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x23 += einsum("ia,waj->wji", t1.bb, lu11.bb)
    x23 += einsum("wia,wxaj->xji", u11.bb, lu12.bb)
    rdm1_f_vo_bb -= einsum("wia,wij->aj", u11.bb, x23)
    del x23
    x25 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x25 += einsum("abij->jiab", l2.aaaa) * -1
    x25 += einsum("abij->jiba", l2.aaaa)
    rdm1_f_vv_aa = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    rdm1_f_vv_aa += einsum("ijab,ijbc->ac", t2.aaaa, x25) * -1
    del x25
    x26 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x26 += einsum("abij->jiab", l2.bbbb) * -1
    x26 += einsum("abij->jiba", l2.bbbb)
    rdm1_f_vv_bb = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    rdm1_f_vv_bb += einsum("ijab,ijca->bc", t2.bbbb, x26) * -1
    del x26
    rdm1_f_oo_aa += einsum("ij->ji", delta_oo.aa)
    rdm1_f_oo_bb += einsum("ij->ji", delta_oo.bb)
    rdm1_f_ov_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    rdm1_f_ov_aa += einsum("ai->ia", l1.aa)
    rdm1_f_ov_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    rdm1_f_ov_bb += einsum("ai->ia", l1.bb)
    rdm1_f_vo_aa += einsum("wx,wxia->ai", ls2, u12.aa) * 0.5
    rdm1_f_vo_aa += einsum("w,wia->ai", ls1, u11.aa)
    rdm1_f_vo_aa += einsum("ia->ai", t1.aa)
    rdm1_f_vo_aa += einsum("ai,jiba->bj", l1.bb, t2.abab)
    rdm1_f_vo_bb += einsum("ai,ijab->bj", l1.aa, t2.abab)
    rdm1_f_vo_bb += einsum("wx,wxia->ai", ls2, u12.bb) * 0.5
    rdm1_f_vo_bb += einsum("w,wia->ai", ls1, u11.bb)
    rdm1_f_vo_bb += einsum("ia->ai", t1.bb)
    rdm1_f_vv_aa += einsum("wai,wib->ba", lu11.aa, u11.aa)
    rdm1_f_vv_aa += einsum("ai,ib->ba", l1.aa, t1.aa)
    rdm1_f_vv_aa += einsum("wxai,wxib->ba", lu12.aa, u12.aa) * 0.5
    rdm1_f_vv_aa += einsum("abij,ijcb->ca", l2.abab, t2.abab)
    rdm1_f_vv_bb += einsum("ai,ib->ba", l1.bb, t1.bb)
    rdm1_f_vv_bb += einsum("abij,ijac->cb", l2.abab, t2.abab)
    rdm1_f_vv_bb += einsum("wxai,wxib->ba", lu12.bb, u12.bb) * 0.5
    rdm1_f_vv_bb += einsum("wai,wib->ba", lu11.bb, u11.bb)

    rdm1_f_aa = np.block([[rdm1_f_oo_aa, rdm1_f_ov_aa], [rdm1_f_vo_aa, rdm1_f_vv_aa]])
    rdm1_f_bb = np.block([[rdm1_f_oo_bb, rdm1_f_ov_bb], [rdm1_f_vo_bb, rdm1_f_vv_bb]])

    rdm1_f.aa = rdm1_f_aa
    rdm1_f.bb = rdm1_f_bb

    return rdm1_f

def make_rdm2_f(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, lu12=None, **kwargs):
    rdm2_f = Namespace()

    # Get boson coupling creation array:
    gc = Namespace(
        aa = Namespace(
            boo=g.aa.boo.transpose(0, 2, 1),
            bov=g.aa.bvo.transpose(0, 2, 1),
            bvo=g.aa.bov.transpose(0, 2, 1),
            bvv=g.aa.bvv.transpose(0, 2, 1),
        ),
        bb = Namespace(
            boo=g.bb.boo.transpose(0, 2, 1),
            bov=g.bb.bvo.transpose(0, 2, 1),
            bvo=g.bb.bov.transpose(0, 2, 1),
            bvv=g.bb.bvv.transpose(0, 2, 1),
        ),
    )

    delta_oo = Namespace()
    delta_oo.aa = np.eye(nocc[0])
    delta_oo.bb = np.eye(nocc[1])
    delta_vv = Namespace()
    delta_vv.aa = np.eye(nvir[0])
    delta_vv.bb = np.eye(nvir[1])

    # 2RDM
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x0 += einsum("abij,klab->ijkl", l2.aaaa, t2.aaaa)
    x52 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x52 += einsum("ijkl->jilk", x0)
    rdm2_f_oooo_aaaa = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_oooo_aaaa += einsum("ijkl->jkil", x0) * -1
    rdm2_f_oooo_aaaa += einsum("ijkl->jlik", x0)
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum("ia,abjk->kjib", t1.aa, l2.aaaa)
    x2 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x2 += einsum("ia,jkla->jkil", t1.aa, x1)
    x52 += einsum("ijkl->ijkl", x2)
    x53 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x53 += einsum("ia,ijkl->jkla", t1.aa, x52)
    del x52
    x60 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x60 += einsum("ijka->ikja", x53) * -1
    x127 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x127 += einsum("ijka->ikja", x53)
    del x53
    x220 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x220 += einsum("ia,ijkl->jlka", t1.aa, x2)
    x221 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x221 += einsum("ia,ijkb->jkab", t1.aa, x220)
    del x220
    x227 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x227 += einsum("ijab->ijab", x221)
    del x221
    x230 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x230 += einsum("ijab,jikl->lkab", t2.aaaa, x2)
    x233 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x233 += einsum("ijab->ijab", x230)
    del x230
    rdm2_f_oooo_aaaa += einsum("ijkl->ikjl", x2)
    rdm2_f_oooo_aaaa += einsum("ijkl->iljk", x2) * -1
    del x2
    x49 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x49 += einsum("ijka->ijka", x1) * -1
    x49 += einsum("ijka->jika", x1)
    x56 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x56 += einsum("ijab,ijka->kb", t2.aaaa, x49) * 2
    x59 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x59 += einsum("ia->ia", x56) * -1
    del x56
    x119 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x119 += einsum("ijab,ijka->kb", t2.aaaa, x49)
    x124 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x124 += einsum("ia->ia", x119) * -1
    x336 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x336 += einsum("ia->ia", x119) * -1
    del x119
    x190 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x190 += einsum("ijab,ijka->kb", t2.aaaa, x49) * 2.0000000000000013
    x193 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x193 += einsum("ia->ia", x190) * -1
    del x190
    x317 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x317 += einsum("ijab,ikla->kljb", t2.abab, x49)
    x319 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x319 += einsum("ijka->ijka", x317) * -1
    del x317
    x337 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x337 += einsum("ijab,ijka->kb", t2.aaaa, x49) * -1
    x98 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x98 += einsum("ijka->ijka", x1)
    x98 += einsum("ijka->jika", x1) * -1
    x99 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x99 += einsum("ijab,ikla->kljb", t2.abab, x98)
    del x98
    rdm2_f_oovo_aabb = np.zeros((nocc[0], nocc[0], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x99) * -1
    rdm2_f_vooo_bbaa = np.zeros((nvir[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x99) * -1
    del x99
    x136 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x136 += einsum("ijka->ijka", x1)
    x136 -= einsum("ijka->jika", x1)
    x137 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x137 += einsum("ia,ijkb->jkab", t1.aa, x136)
    del x136
    rdm2_f_oovv_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_oovv_aaaa -= einsum("ijab->ijab", x137)
    rdm2_f_vvoo_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_vvoo_aaaa -= einsum("ijab->abij", x137)
    del x137
    x168 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x168 -= einsum("ijka->ijka", x1)
    x168 += einsum("ijka->jika", x1)
    x169 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x169 += einsum("ia,ijkb->jkab", t1.aa, x168)
    rdm2_f_ovvo_aaaa = np.zeros((nocc[0], nvir[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_ovvo_aaaa -= einsum("ijab->ibaj", x169)
    rdm2_f_voov_aaaa = np.zeros((nvir[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_voov_aaaa -= einsum("ijab->ajib", x169)
    del x169
    x350 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x350 += einsum("ijab,ijkc->kcab", t2.aaaa, x1)
    x355 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x355 += einsum("iabc->iabc", x350)
    del x350
    x351 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x351 += einsum("ia,jikb->jkba", t1.aa, x1)
    x353 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x353 += einsum("ijab->ijab", x351) * -2
    del x351
    rdm2_f_ooov_aaaa = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_ooov_aaaa += einsum("ijka->ikja", x1)
    rdm2_f_ooov_aaaa -= einsum("ijka->jkia", x1)
    rdm2_f_ovoo_aaaa = np.zeros((nocc[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_ovoo_aaaa -= einsum("ijka->iajk", x1)
    rdm2_f_ovoo_aaaa += einsum("ijka->jaik", x1)
    del x1
    x3 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x3 += einsum("wai,wja->ij", lu11.aa, u11.aa)
    x14 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x14 += einsum("ij->ji", x3)
    x38 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x38 += einsum("ia,ij->ja", t1.aa, x3)
    x43 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x43 += einsum("ia->ia", x38)
    del x38
    x44 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x44 += einsum("ij->ij", x3)
    x110 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x110 += einsum("ij->ij", x3)
    x197 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x197 += einsum("ij,kiab->kjab", x3, t2.aaaa)
    x219 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x219 -= einsum("ijab->ijab", x197)
    del x197
    x235 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x235 += einsum("ij->ij", x3)
    x335 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x335 += einsum("ij->ij", x3)
    rdm2_f_oooo_aaaa -= einsum("ij,kl->ijkl", delta_oo.aa, x3)
    rdm2_f_oooo_aaaa += einsum("ij,kl->ilkj", delta_oo.aa, x3)
    rdm2_f_oooo_aaaa += einsum("ij,kl->kijl", delta_oo.aa, x3)
    rdm2_f_oooo_aaaa -= einsum("ij,kl->klji", delta_oo.aa, x3)
    del x3
    x4 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum("wxai,wxja->ij", lu12.aa, u12.aa)
    x8 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x8 += einsum("ij->ij", x4)
    x14 += einsum("ij->ji", x4) * 0.5
    x57 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x57 += einsum("ij->ij", x4) * 0.5
    x110 += einsum("ij->ij", x4) * 0.5
    x191 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x191 += einsum("ij->ij", x4) * 0.49999999999999967
    x335 += einsum("ij->ij", x4) * 0.5
    del x4
    x5 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x5 += einsum("abij,kjab->ik", l2.abab, t2.abab)
    x8 += einsum("ij->ij", x5) * 2
    x14 += einsum("ij->ji", x5)
    x57 += einsum("ij->ij", x5)
    x110 += einsum("ij->ij", x5)
    x191 += einsum("ij->ij", x5)
    x335 += einsum("ij->ij", x5)
    del x5
    x6 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x6 += einsum("ijab->jiab", t2.aaaa)
    x6 += einsum("ijab->jiba", t2.aaaa) * -1
    x7 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x7 += einsum("abij,ikab->jk", l2.aaaa, x6) * 2
    x8 += einsum("ij->ij", x7) * -1
    del x7
    rdm2_f_oooo_aaaa += einsum("ij,kl->jikl", delta_oo.aa, x8) * -0.5
    rdm2_f_oooo_aaaa += einsum("ij,kl->jlki", delta_oo.aa, x8) * 0.5
    rdm2_f_oooo_aaaa += einsum("ij,kl->kjil", delta_oo.aa, x8) * 0.5
    rdm2_f_oooo_aaaa += einsum("ij,kl->klij", delta_oo.aa, x8) * -0.5
    del x8
    x13 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x13 += einsum("abij,ikab->jk", l2.aaaa, x6)
    x14 += einsum("ij->ji", x13) * -1
    x57 += einsum("ij->ij", x13) * -1
    x58 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x58 += einsum("ia,ij->ja", t1.aa, x57) * 2
    x59 += einsum("ia->ia", x58)
    del x58
    x60 += einsum("ia,jk->jika", t1.aa, x57) * -1
    x127 += einsum("ia,jk->jika", t1.aa, x57)
    x189 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x189 += einsum("ij,ikab->kjab", x57, t2.aaaa)
    del x57
    x194 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x194 += einsum("ijab->ijba", x189)
    del x189
    x110 += einsum("ij->ij", x13) * -1
    x191 += einsum("ij->ij", x13) * -1
    del x13
    x192 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x192 += einsum("ia,ij->ja", t1.aa, x191) * 2.0000000000000013
    del x191
    x193 += einsum("ia->ia", x192)
    del x192
    x126 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x126 += einsum("ijka,ilba->ljkb", x49, x6)
    x127 += einsum("ijka->jkia", x126)
    del x126
    x184 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x184 += einsum("wxai,ijab->wxjb", lu12.aa, x6)
    x185 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x185 += einsum("wxia->xwia", x184) * -1
    del x184
    x334 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x334 += einsum("abij,ikab->jk", l2.aaaa, x6)
    x335 += einsum("ij->ij", x334) * -1
    del x334
    x9 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x9 += einsum("ai,ja->ij", l1.aa, t1.aa)
    x14 += einsum("ij->ji", x9)
    x31 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x31 += einsum("ia,ij->ja", t1.aa, x9)
    x61 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x61 += einsum("ia->ia", x31) * -1
    rdm2_f_oovo_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_oovo_aaaa += einsum("ij,ka->ikaj", delta_oo.aa, x31)
    rdm2_f_vooo_aaaa = np.zeros((nvir[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_vooo_aaaa -= einsum("ij,ka->akji", delta_oo.aa, x31)
    del x31
    x44 += einsum("ij->ij", x9)
    x45 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x45 += einsum("ia,jk->jika", t1.aa, x44)
    x125 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x125 += einsum("ia,jk->jika", t1.aa, x44)
    del x44
    x110 += einsum("ij->ij", x9)
    x123 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x123 += einsum("ia,ij->ja", t1.aa, x110)
    x124 += einsum("ia->ia", x123)
    del x123
    x322 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x322 += einsum("ij,ikab->jkab", x110, t2.abab)
    rdm2_f_vovo_bbaa = np.zeros((nvir[1], nocc[1], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x322) * -1
    rdm2_f_vovo_aabb = np.zeros((nvir[0], nocc[0], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x322) * -1
    del x322
    rdm2_f_oovo_aabb += einsum("ia,jk->jkai", t1.bb, x110) * -1
    rdm2_f_vooo_bbaa += einsum("ia,jk->aijk", t1.bb, x110) * -1
    del x110
    x195 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x195 += einsum("ij,kiab->jkab", x9, t2.aaaa)
    x219 += einsum("ijab->ijab", x195)
    del x195
    x235 += einsum("ij->ij", x9)
    x236 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x236 += einsum("ia,ij->ja", t1.aa, x235)
    del x235
    x237 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x237 += einsum("ia->ia", x236)
    x238 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x238 += einsum("ia->ia", x236)
    del x236
    x335 += einsum("ij->ij", x9)
    x336 += einsum("ia,ij->ja", t1.aa, x335) * 0.9999999999999993
    x337 += einsum("ia,ij->ja", t1.aa, x335)
    del x335
    rdm2_f_oooo_aaaa -= einsum("ij,kl->jikl", delta_oo.aa, x9)
    rdm2_f_oooo_aaaa += einsum("ij,kl->kijl", delta_oo.aa, x9)
    rdm2_f_oooo_aaaa += einsum("ij,kl->jlki", delta_oo.aa, x9)
    rdm2_f_oooo_aaaa -= einsum("ij,kl->klji", delta_oo.aa, x9)
    del x9
    x10 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x10 += einsum("abij,klab->ikjl", l2.abab, t2.abab)
    x101 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x101 += einsum("ijkl->ijkl", x10)
    x305 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x305 += einsum("ijkl->ijkl", x10)
    x313 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x313 += einsum("ijkl->ijkl", x10)
    rdm2_f_oooo_aabb = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_oooo_aabb += einsum("ijkl->ijkl", x10)
    rdm2_f_oooo_bbaa = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_oooo_bbaa += einsum("ijkl->klij", x10)
    del x10
    x11 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x11 += einsum("ia,bajk->jkib", t1.bb, l2.abab)
    x12 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x12 += einsum("ia,jkla->jikl", t1.aa, x11)
    x101 += einsum("ijkl->ijkl", x12)
    x102 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x102 += einsum("ia,jkil->jkla", t1.bb, x101)
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x102)
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x102)
    del x102
    x117 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x117 += einsum("ia,ijkl->jkla", t1.aa, x101)
    del x101
    rdm2_f_oovo_bbaa = np.zeros((nocc[1], nocc[1], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x117)
    rdm2_f_vooo_aabb = np.zeros((nvir[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x117)
    del x117
    x305 += einsum("ijkl->ijkl", x12)
    x306 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x306 += einsum("ijab,ikjl->klab", t2.abab, x305)
    del x305
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x306)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x306)
    del x306
    x313 += einsum("ijkl->ijkl", x12) * 0.9999999999999993
    x314 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x314 += einsum("ia,ijkl->jkla", t1.aa, x313)
    del x313
    x315 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x315 += einsum("ijka->ijka", x314)
    del x314
    rdm2_f_oooo_aabb += einsum("ijkl->ijkl", x12)
    rdm2_f_oooo_bbaa += einsum("ijkl->klij", x12)
    del x12
    x64 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x64 += einsum("ijab,ikla->kljb", t2.abab, x11)
    x76 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x76 += einsum("ijka->ijka", x64)
    x130 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x130 += einsum("ijka->ijka", x64) * -1
    x253 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x253 -= einsum("ijka->ijka", x64)
    del x64
    x71 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x71 += einsum("ijab,ijka->kb", t2.abab, x11)
    x75 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x75 += einsum("ia->ia", x71) * 2
    x109 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x109 += einsum("ia->ia", x71)
    x279 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x279 += einsum("ia->ia", x71) * 2.0000000000000013
    x333 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x333 += einsum("ia->ia", x71)
    x338 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x338 += einsum("ia->ia", x71)
    del x71
    x95 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x95 += einsum("ijab,kjla->kilb", t2.abab, x11)
    x319 += einsum("ijka->ijka", x95) * -1
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x95)
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x95)
    del x95
    x161 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x161 += einsum("ia,ijkb->jkba", t1.aa, x11)
    rdm2_f_oovv_bbaa = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_oovv_bbaa -= einsum("ijab->ijba", x161)
    rdm2_f_vvoo_aabb = np.zeros((nvir[0], nvir[0], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_vvoo_aabb -= einsum("ijab->baij", x161)
    del x161
    x172 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x172 += einsum("ia,jikb->jkba", t1.bb, x11)
    x376 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x376 += einsum("ijab->ijab", x172)
    rdm2_f_ovvo_aabb = np.zeros((nocc[0], nvir[0], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_ovvo_aabb -= einsum("ijab->iabj", x172)
    rdm2_f_voov_bbaa = np.zeros((nvir[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_voov_bbaa -= einsum("ijab->bjia", x172)
    del x172
    x371 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x371 += einsum("ijab,ijkc->kcab", t2.abab, x11)
    rdm2_f_vovv_bbaa = np.zeros((nvir[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_vovv_bbaa += einsum("iabc->ciba", x371) * -1
    rdm2_f_vvvo_aabb = np.zeros((nvir[0], nvir[0], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_vvvo_aabb += einsum("iabc->baci", x371) * -1
    del x371
    rdm2_f_ooov_bbaa = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_ooov_bbaa -= einsum("ijka->jkia", x11)
    rdm2_f_ovoo_aabb = np.zeros((nocc[0], nvir[0], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_ovoo_aabb -= einsum("ijka->iajk", x11)
    x14 += einsum("ij->ji", delta_oo.aa) * -1
    rdm2_f_oooo_aabb += einsum("ij,kl->lkji", delta_oo.bb, x14) * -1
    rdm2_f_oooo_bbaa += einsum("ij,kl->jilk", delta_oo.bb, x14) * -1
    del x14
    x15 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x15 += einsum("ai,ja->ij", l1.bb, t1.bb)
    x21 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x21 += einsum("ij->ij", x15) * 2
    x23 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x23 += einsum("ij->ij", x15)
    x89 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x89 += einsum("ij->ij", x15)
    x93 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x93 += einsum("ia,ij->ja", t1.bb, x15)
    x94 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x94 += einsum("ia->ia", x93) * -1
    del x93
    x239 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x239 += einsum("ij,kiab->jkab", x15, t2.bbbb)
    x263 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x263 += einsum("ijab->ijab", x239)
    del x239
    x297 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x297 += einsum("ij->ij", x15)
    x332 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x332 += einsum("ij->ij", x15)
    rdm2_f_oooo_bbbb = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_oooo_bbbb -= einsum("ij,kl->jikl", delta_oo.bb, x15)
    rdm2_f_oooo_bbbb += einsum("ij,kl->kijl", delta_oo.bb, x15)
    rdm2_f_oooo_bbbb += einsum("ij,kl->jlki", delta_oo.bb, x15)
    rdm2_f_oooo_bbbb -= einsum("ij,kl->klji", delta_oo.bb, x15)
    del x15
    x16 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x16 += einsum("wai,wja->ij", lu11.bb, u11.bb)
    x21 += einsum("ij->ij", x16) * 2
    x23 += einsum("ij->ij", x16)
    x83 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x83 += einsum("ia,ij->ja", t1.bb, x16)
    x88 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x88 += einsum("ia->ia", x83)
    del x83
    x89 += einsum("ij->ij", x16)
    x90 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x90 += einsum("ia,jk->jika", t1.bb, x89)
    x128 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x128 += einsum("ia,jk->jika", t1.bb, x89)
    del x89
    x241 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x241 += einsum("ij,kiab->kjab", x16, t2.bbbb)
    x263 -= einsum("ijab->ijab", x241)
    del x241
    x297 += einsum("ij->ij", x16)
    x298 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x298 += einsum("ia,ij->ja", t1.bb, x297)
    del x297
    x299 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x299 += einsum("ia->ia", x298)
    x300 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x300 += einsum("ia->ia", x298)
    del x298
    x332 += einsum("ij->ij", x16)
    rdm2_f_oooo_bbbb -= einsum("ij,kl->ijkl", delta_oo.bb, x16)
    rdm2_f_oooo_bbbb += einsum("ij,kl->ilkj", delta_oo.bb, x16)
    rdm2_f_oooo_bbbb += einsum("ij,kl->kijl", delta_oo.bb, x16)
    rdm2_f_oooo_bbbb -= einsum("ij,kl->klji", delta_oo.bb, x16)
    del x16
    x17 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x17 += einsum("wxai,wxja->ij", lu12.bb, u12.bb)
    x21 += einsum("ij->ij", x17)
    x23 += einsum("ij->ij", x17) * 0.5
    x24 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x24 += einsum("ij->ij", x17)
    x73 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x73 += einsum("ij->ij", x17) * 0.5
    x274 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x274 += einsum("ij->ij", x17) * 0.5
    x277 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x277 += einsum("ij->ij", x17) * 0.49999999999999967
    x332 += einsum("ij->ij", x17) * 0.5
    del x17
    x18 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x18 += einsum("abij,ikab->jk", l2.abab, t2.abab)
    x21 += einsum("ij->ij", x18) * 2
    x23 += einsum("ij->ij", x18)
    x24 += einsum("ij->ij", x18) * 2
    x73 += einsum("ij->ij", x18)
    x265 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x265 += einsum("ij,kiab->jkab", x18, t2.bbbb)
    x280 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x280 += einsum("ijab->ijab", x265) * -1
    del x265
    x277 += einsum("ij->ij", x18)
    x332 += einsum("ij->ij", x18)
    del x18
    x19 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x19 += einsum("ijab->jiab", t2.bbbb) * -1
    x19 += einsum("ijab->jiba", t2.bbbb)
    x20 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x20 += einsum("abij,ikba->jk", l2.bbbb, x19) * 2
    x21 += einsum("ij->ij", x20) * -1
    rdm2_f_oooo_aabb += einsum("ij,kl->jikl", delta_oo.aa, x21) * -0.5
    rdm2_f_oovo_bbaa += einsum("ia,jk->jkai", t1.aa, x21) * -0.5
    rdm2_f_vooo_aabb += einsum("ia,jk->aijk", t1.aa, x21) * -0.5
    del x21
    x24 += einsum("ij->ij", x20) * -1
    del x20
    rdm2_f_oooo_bbbb += einsum("ij,kl->jikl", delta_oo.bb, x24) * -0.5
    rdm2_f_oooo_bbbb += einsum("ij,kl->jlki", delta_oo.bb, x24) * 0.5
    rdm2_f_oooo_bbbb += einsum("ij,kl->kjil", delta_oo.bb, x24) * 0.5
    rdm2_f_oooo_bbbb += einsum("ij,kl->klij", delta_oo.bb, x24) * -0.5
    del x24
    x22 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x22 += einsum("abij,ikba->jk", l2.bbbb, x19)
    x23 += einsum("ij->ij", x22) * -1
    x108 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x108 += einsum("ia,ij->ja", t1.bb, x23)
    x109 += einsum("ia->ia", x108)
    del x108
    x321 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x321 += einsum("ij,kiab->kjab", x23, t2.abab)
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x321) * -1
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x321) * -1
    del x321
    rdm2_f_oooo_bbaa += einsum("ij,kl->klji", delta_oo.aa, x23) * -1
    del x23
    x73 += einsum("ij->ij", x22) * -1
    x74 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x74 += einsum("ia,ij->ja", t1.bb, x73) * 2
    x75 += einsum("ia->ia", x74)
    del x74
    x76 += einsum("ia,jk->jika", t1.bb, x73) * -1
    x130 += einsum("ia,jk->jika", t1.bb, x73)
    del x73
    x274 += einsum("ij->ij", x22) * -1
    x275 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x275 += einsum("ij,ikab->kjab", x274, t2.bbbb)
    del x274
    x280 += einsum("ijab->ijba", x275)
    del x275
    x277 += einsum("ij->ij", x22) * -1
    del x22
    x278 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x278 += einsum("ia,ij->ja", t1.bb, x277) * 2.0000000000000013
    del x277
    x279 += einsum("ia->ia", x278)
    del x278
    x268 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x268 += einsum("wxai,ijba->wxjb", lu12.bb, x19)
    x269 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x269 += einsum("wxia->xwia", x268) * -1
    del x268
    x331 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x331 += einsum("abij,ikba->jk", l2.bbbb, x19)
    x332 += einsum("ij->ij", x331) * -1
    del x331
    x333 += einsum("ia,ij->ja", t1.bb, x332)
    x338 += einsum("ia,ij->ja", t1.bb, x332) * 0.9999999999999993
    del x332
    x25 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x25 += einsum("abij,klba->ijlk", l2.bbbb, t2.bbbb)
    x68 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x68 += einsum("ijkl->jilk", x25)
    x291 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x291 += einsum("ia,jikl->jkla", t1.bb, x25)
    x292 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x292 += einsum("ia,ijkb->kjba", t1.bb, x291)
    del x291
    x295 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x295 += einsum("ijab->ijab", x292)
    del x292
    x293 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x293 += einsum("ijkl->jilk", x25)
    rdm2_f_oooo_bbbb += einsum("ijkl->jkil", x25) * -1
    rdm2_f_oooo_bbbb += einsum("ijkl->jlik", x25)
    del x25
    x26 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x26 += einsum("ia,bajk->jkib", t1.bb, l2.bbbb)
    x27 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x27 += einsum("ia,jkla->kjli", t1.bb, x26)
    x68 += einsum("ijkl->ijkl", x27)
    x69 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x69 += einsum("ia,ijkl->jkla", t1.bb, x68)
    del x68
    x76 += einsum("ijka->ikja", x69) * -1
    x130 += einsum("ijka->ikja", x69)
    del x69
    x285 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x285 += einsum("ia,ijkl->jlka", t1.bb, x27)
    x286 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x286 += einsum("ia,ijkb->jkab", t1.bb, x285)
    del x285
    x290 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x290 += einsum("ijab->ijab", x286)
    del x286
    x293 += einsum("ijkl->ijkl", x27)
    x294 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x294 += einsum("ijab,ijkl->klab", t2.bbbb, x293)
    del x293
    x295 += einsum("ijab->ijab", x294)
    del x294
    rdm2_f_vovo_bbbb = np.zeros((nvir[1], nocc[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_vovo_bbbb += einsum("ijab->ajbi", x295) * -1
    rdm2_f_vovo_bbbb += einsum("ijab->bjai", x295)
    del x295
    rdm2_f_oooo_bbbb += einsum("ijkl->ikjl", x27)
    rdm2_f_oooo_bbbb += einsum("ijkl->iljk", x27) * -1
    del x27
    x65 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x65 += einsum("ijka->ijka", x26)
    x65 += einsum("ijka->jika", x26) * -1
    x72 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x72 += einsum("ijab,ijkb->ka", t2.bbbb, x65) * 2
    x75 += einsum("ia->ia", x72) * -1
    del x72
    x104 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x104 += einsum("ijab,ijkb->ka", t2.bbbb, x65)
    x109 += einsum("ia->ia", x104) * -1
    x338 += einsum("ia->ia", x104) * -1
    del x104
    x129 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x129 += einsum("ijab,kila->jklb", x19, x65)
    x130 += einsum("ijka->jkia", x129)
    del x129
    x276 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x276 += einsum("ijab,ijkb->ka", t2.bbbb, x65) * 2.0000000000000013
    x279 += einsum("ia->ia", x276) * -1
    del x276
    x333 += einsum("ijab,ijkb->ka", t2.bbbb, x65) * -1
    x114 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x114 += einsum("ijka->ijka", x26) * -1
    x114 += einsum("ijka->jika", x26)
    x115 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x115 += einsum("ijab,kjlb->ikla", t2.abab, x114)
    del x114
    x315 += einsum("ijka->ijka", x115) * -1
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x115) * -1
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x115) * -1
    del x115
    x151 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x151 -= einsum("ijka->ijka", x26)
    x151 += einsum("ijka->jika", x26)
    x152 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x152 += einsum("ia,jikb->jkab", t1.bb, x151)
    rdm2_f_oovv_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_oovv_bbbb -= einsum("ijab->ijab", x152)
    rdm2_f_vvoo_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_vvoo_bbbb -= einsum("ijab->abij", x152)
    del x152
    x181 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x181 += einsum("ia,ijkb->jkab", t1.bb, x151)
    del x151
    rdm2_f_ovvo_bbbb = np.zeros((nocc[1], nvir[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_ovvo_bbbb -= einsum("ijab->ibaj", x181)
    rdm2_f_voov_bbbb = np.zeros((nvir[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_voov_bbbb -= einsum("ijab->ajib", x181)
    del x181
    x251 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x251 += einsum("ijka->ijka", x26)
    x251 -= einsum("ijka->jika", x26)
    x363 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x363 += einsum("ijab,ijkc->kcab", t2.bbbb, x26)
    x369 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x369 += einsum("iabc->iabc", x363)
    del x363
    x364 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x364 += einsum("ia,jikb->jkba", t1.bb, x26)
    x366 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x366 += einsum("ijab->ijab", x364) * -2
    del x364
    rdm2_f_ooov_bbbb = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_ooov_bbbb += einsum("ijka->ikja", x26)
    rdm2_f_ooov_bbbb -= einsum("ijka->jkia", x26)
    rdm2_f_ovoo_bbbb = np.zeros((nocc[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_ovoo_bbbb -= einsum("ijka->iajk", x26)
    rdm2_f_ovoo_bbbb += einsum("ijka->jaik", x26)
    del x26
    x28 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x28 += einsum("ia,abjk->jikb", t1.aa, l2.abab)
    x48 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x48 += einsum("ijab,kljb->klia", t2.abab, x28)
    x60 += einsum("ijka->ijka", x48)
    x127 += einsum("ijka->ijka", x48) * -1
    x209 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x209 -= einsum("ijka->ijka", x48)
    del x48
    x55 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x55 += einsum("ijab,ikjb->ka", t2.abab, x28)
    x59 += einsum("ia->ia", x55) * 2
    x124 += einsum("ia->ia", x55)
    x193 += einsum("ia->ia", x55) * 2.0000000000000013
    x336 += einsum("ia->ia", x55)
    x337 += einsum("ia->ia", x55)
    del x55
    x112 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x112 += einsum("ijab,iklb->klja", t2.abab, x28)
    x315 += einsum("ijka->ijka", x112)
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x112)
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x112)
    del x112
    x165 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x165 += einsum("ia,jkib->jkba", t1.bb, x28)
    rdm2_f_oovv_aabb = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_oovv_aabb -= einsum("ijab->ijba", x165)
    rdm2_f_vvoo_bbaa = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_vvoo_bbaa -= einsum("ijab->baij", x165)
    del x165
    x177 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x177 += einsum("ia,ijkb->jkab", t1.aa, x28)
    x386 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x386 += einsum("ijab->ijab", x177)
    rdm2_f_ovvo_bbaa = np.zeros((nocc[1], nvir[1], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_ovvo_bbaa -= einsum("ijab->jbai", x177)
    rdm2_f_voov_aabb = np.zeros((nvir[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_voov_aabb -= einsum("ijab->aijb", x177)
    del x177
    x318 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x318 += einsum("ijab,klib->klja", x19, x28)
    del x19
    x319 += einsum("ijka->ijka", x318) * -1
    del x318
    x381 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x381 += einsum("ijab,ikjc->kacb", t2.abab, x28)
    rdm2_f_vovv_aabb = np.zeros((nvir[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_vovv_aabb += einsum("iabc->aicb", x381) * -1
    rdm2_f_vvvo_bbaa = np.zeros((nvir[1], nvir[1], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_vvvo_bbaa += einsum("iabc->cbai", x381) * -1
    del x381
    rdm2_f_ooov_aabb = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_ooov_aabb -= einsum("ijka->ijka", x28)
    rdm2_f_ovoo_bbaa = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_ovoo_bbaa -= einsum("ijka->kaij", x28)
    x29 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x29 += einsum("wx,wxia->ia", ls2, u12.aa)
    x61 += einsum("ia->ia", x29) * 0.5
    x124 += einsum("ia->ia", x29) * -0.5
    x237 += einsum("ia->ia", x29) * -0.5
    x238 += einsum("ia->ia", x29) * -0.5
    x336 += einsum("ia->ia", x29) * -0.49999999999999967
    x337 += einsum("ia->ia", x29) * -0.5
    rdm2_f_oovo_aaaa += einsum("ij,ka->jkai", delta_oo.aa, x29) * -0.5
    rdm2_f_vooo_aaaa += einsum("ij,ka->akij", delta_oo.aa, x29) * 0.5
    del x29
    x30 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x30 += einsum("w,wia->ia", ls1, u11.aa)
    x61 += einsum("ia->ia", x30)
    x124 += einsum("ia->ia", x30) * -1
    x237 += einsum("ia->ia", x30) * -1
    x238 += einsum("ia->ia", x30) * -1
    x336 += einsum("ia->ia", x30) * -0.9999999999999993
    x337 += einsum("ia->ia", x30) * -1
    rdm2_f_oovo_aaaa -= einsum("ij,ka->jkai", delta_oo.aa, x30)
    rdm2_f_vooo_aaaa += einsum("ij,ka->akji", delta_oo.aa, x30)
    del x30
    x32 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x32 += einsum("ai,jkba->ijkb", l1.aa, t2.aaaa)
    x45 += einsum("ijka->ijka", x32)
    x125 += einsum("ijka->ijka", x32)
    x209 += einsum("ijka->ijka", x32)
    del x32
    x33 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x33 += einsum("ia,waj->wji", t1.aa, lu11.aa)
    x34 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x34 += einsum("wia,wjk->jkia", u11.aa, x33)
    x45 -= einsum("ijka->ijka", x34)
    x125 -= einsum("ijka->ijka", x34)
    del x34
    x41 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x41 += einsum("wij->wij", x33)
    x121 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x121 += einsum("wij->wij", x33)
    x211 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x211 += einsum("ia,wij->wja", t1.aa, x33)
    x214 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x214 -= einsum("wia->wia", x211)
    del x211
    x234 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x234 += einsum("wia,wij->ja", u11.aa, x33)
    del x33
    x237 += einsum("ia->ia", x234)
    x238 += einsum("ia->ia", x234)
    del x234
    rdm2_f_vovo_aaaa = np.zeros((nvir[0], nocc[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_vovo_aaaa += einsum("ia,jb->biaj", t1.aa, x238)
    rdm2_f_vovo_aaaa += einsum("ia,jb->aibj", t1.aa, x238) * -1
    del x238
    x35 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x35 += einsum("wia,wxaj->xji", u11.aa, lu12.aa)
    x36 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x36 += einsum("wia,wjk->jika", u11.aa, x35)
    x45 += einsum("ijka->ijka", x36)
    x125 += einsum("ijka->ijka", x36)
    del x36
    rdm2_f_vooo_aaaa -= einsum("ijka->ajik", x125)
    rdm2_f_vooo_aaaa += einsum("ijka->akij", x125)
    del x125
    x41 += einsum("wij->wij", x35)
    x42 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x42 += einsum("wia,wij->ja", u11.aa, x41)
    x43 += einsum("ia->ia", x42)
    del x42
    x103 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x103 += einsum("wia,wjk->jkia", u11.bb, x41)
    rdm2_f_oovo_aabb -= einsum("ijka->ijak", x103)
    rdm2_f_vooo_bbaa -= einsum("ijka->akij", x103)
    del x103
    x324 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x324 += einsum("ia,wij->wja", t1.aa, x41)
    del x41
    x325 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x325 += einsum("wia->wia", x324)
    del x324
    x121 += einsum("wij->wij", x35)
    x122 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x122 += einsum("wia,wij->ja", u11.aa, x121)
    x124 += einsum("ia->ia", x122)
    x337 += einsum("ia->ia", x122)
    del x122
    x336 += einsum("wia,wij->ja", u11.aa, x121) * 0.9999999999999993
    del x121
    x200 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x200 += einsum("ia,wij->wja", t1.aa, x35)
    x201 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x201 += einsum("wia,wjb->ijba", u11.aa, x200)
    del x200
    x219 += einsum("ijab->ijab", x201)
    del x201
    x216 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x216 += einsum("wia,wij->ja", u11.aa, x35)
    del x35
    x218 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x218 -= einsum("ia->ia", x216)
    del x216
    x37 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x37 += einsum("ai,jiba->jb", l1.bb, t2.abab)
    x43 -= einsum("ia->ia", x37)
    x124 += einsum("ia->ia", x37) * -1
    x218 += einsum("ia->ia", x37)
    x336 += einsum("ia->ia", x37) * -0.9999999999999993
    x337 += einsum("ia->ia", x37) * -1
    del x37
    x39 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x39 -= einsum("ijab->jiab", t2.aaaa)
    x39 += einsum("ijab->jiba", t2.aaaa)
    x40 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x40 += einsum("ai,ijab->jb", l1.aa, x39)
    x43 -= einsum("ia->ia", x40)
    del x40
    x45 -= einsum("ij,ka->jika", delta_oo.aa, x43)
    rdm2_f_oovo_aaaa += einsum("ijka->ijak", x45)
    rdm2_f_oovo_aaaa -= einsum("ijka->ikaj", x45)
    del x45
    rdm2_f_vooo_aaaa += einsum("ij,ka->ajik", delta_oo.aa, x43)
    rdm2_f_vooo_aaaa -= einsum("ij,ka->akij", delta_oo.aa, x43)
    del x43
    x323 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x323 += einsum("wai,ijab->wjb", lu11.aa, x39)
    x325 -= einsum("wia->wia", x323)
    del x323
    x46 = np.zeros((nbos, nbos, nocc[0], nocc[0]), dtype=types[float])
    x46 += einsum("ia,wxaj->wxji", t1.aa, lu12.aa)
    x47 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x47 += einsum("wxia,wxjk->jkia", u12.aa, x46)
    x60 += einsum("ijka->ijka", x47) * 0.5
    x127 += einsum("ijka->ijka", x47) * -0.5
    rdm2_f_vooo_aaaa += einsum("ijka->ajik", x127) * -1
    rdm2_f_vooo_aaaa += einsum("ijka->akij", x127)
    del x127
    x182 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x182 += einsum("ia,ijkb->jkab", t1.aa, x47)
    del x47
    x194 += einsum("ijab->ijab", x182) * 0.5
    del x182
    x54 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x54 += einsum("wxia,wxij->ja", u12.aa, x46)
    x59 += einsum("ia->ia", x54)
    x60 += einsum("ij,ka->jika", delta_oo.aa, x59) * 0.5
    rdm2_f_vooo_aaaa += einsum("ij,ka->ajik", delta_oo.aa, x59) * 0.5
    rdm2_f_vooo_aaaa += einsum("ij,ka->akij", delta_oo.aa, x59) * -0.5
    del x59
    x124 += einsum("ia->ia", x54) * 0.5
    x193 += einsum("ia->ia", x54)
    x194 += einsum("ia,jb->ijab", t1.aa, x193) * 0.5
    del x193
    x336 += einsum("ia->ia", x54) * 0.49999999999999967
    x337 += einsum("ia->ia", x54) * 0.5
    del x54
    x96 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x96 += einsum("wxia,wxjk->jkia", u12.bb, x46)
    x319 += einsum("ijka->ijka", x96) * 0.5
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x96) * -0.5
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x96) * -0.5
    del x96
    x198 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x198 += einsum("wia,wxij->xja", u11.aa, x46)
    del x46
    x199 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x199 += einsum("wia,wjb->jiab", u11.aa, x198)
    x219 += einsum("ijab->ijab", x199)
    del x199
    x325 += einsum("wia->wia", x198)
    del x198
    x50 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x50 += einsum("ijab->jiab", t2.aaaa) * -1
    x50 += einsum("ijab->jiba", t2.aaaa)
    x51 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x51 += einsum("ijka,ilba->jklb", x49, x50)
    del x49
    x60 += einsum("ijka->ijka", x51)
    del x51
    rdm2_f_oovo_aaaa += einsum("ijka->ijak", x60) * -1
    rdm2_f_oovo_aaaa += einsum("ijka->ikaj", x60)
    del x60
    x116 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x116 += einsum("ijka,ilab->ljkb", x11, x50)
    del x11
    x315 += einsum("ijka->ijka", x116) * -1
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x116) * -1
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x116) * -1
    del x116
    x120 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x120 += einsum("ai,ijab->jb", l1.aa, x50)
    x124 += einsum("ia->ia", x120) * -1
    x337 += einsum("ia->ia", x120) * -1
    del x120
    x336 += einsum("ai,ijab->jb", l1.aa, x50) * -0.9999999999999993
    rdm2_f_vovo_bbaa += einsum("ia,jb->aibj", t1.bb, x336) * -1
    del x336
    x383 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x383 += einsum("abij,ikac->kjcb", l2.abab, x50)
    del x50
    x386 += einsum("ijab->ijab", x383) * -1
    del x383
    x61 += einsum("ia->ia", t1.aa)
    rdm2_f_oovo_aaaa += einsum("ij,ka->jiak", delta_oo.aa, x61)
    rdm2_f_vooo_aaaa += einsum("ij,ka->aijk", delta_oo.aa, x61) * -1
    del x61
    x62 = np.zeros((nbos, nbos, nocc[1], nocc[1]), dtype=types[float])
    x62 += einsum("ia,wxaj->wxji", t1.bb, lu12.bb)
    x63 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x63 += einsum("wxia,wxjk->jkia", u12.bb, x62)
    x76 += einsum("ijka->ijka", x63) * 0.5
    x130 += einsum("ijka->ijka", x63) * -0.5
    rdm2_f_vooo_bbbb = np.zeros((nvir[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_vooo_bbbb += einsum("ijka->ajik", x130) * -1
    rdm2_f_vooo_bbbb += einsum("ijka->akij", x130)
    del x130
    x264 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x264 += einsum("ia,ijkb->jkab", t1.bb, x63)
    del x63
    x280 += einsum("ijab->ijab", x264) * 0.5
    del x264
    x70 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x70 += einsum("wxia,wxij->ja", u12.bb, x62)
    x75 += einsum("ia->ia", x70)
    x76 += einsum("ij,ka->jika", delta_oo.bb, x75) * 0.5
    rdm2_f_vooo_bbbb += einsum("ij,ka->ajik", delta_oo.bb, x75) * 0.5
    rdm2_f_vooo_bbbb += einsum("ij,ka->akij", delta_oo.bb, x75) * -0.5
    del x75
    x109 += einsum("ia->ia", x70) * 0.5
    x279 += einsum("ia->ia", x70)
    x280 += einsum("ia,jb->ijab", t1.bb, x279) * 0.5
    del x279
    x333 += einsum("ia->ia", x70) * 0.5
    x338 += einsum("ia->ia", x70) * 0.49999999999999967
    del x70
    x111 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x111 += einsum("wxia,wxjk->ijka", u12.aa, x62)
    x315 += einsum("ijka->ijka", x111) * -0.5
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x111) * -0.5
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x111) * -0.5
    del x111
    x242 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x242 += einsum("wia,wxij->xja", u11.bb, x62)
    del x62
    x243 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x243 += einsum("wia,wjb->jiab", u11.bb, x242)
    x263 += einsum("ijab->ijab", x243)
    del x243
    x329 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x329 += einsum("wia->wia", x242)
    del x242
    x66 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x66 += einsum("ijab->jiab", t2.bbbb)
    x66 += einsum("ijab->jiba", t2.bbbb) * -1
    x67 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x67 += einsum("ijka,jlab->iklb", x65, x66)
    del x65
    x76 += einsum("ijka->ijka", x67)
    del x67
    rdm2_f_oovo_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_oovo_bbbb += einsum("ijka->ijak", x76) * -1
    rdm2_f_oovo_bbbb += einsum("ijka->ikaj", x76)
    del x76
    x100 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x100 += einsum("ijka,klba->ijlb", x28, x66)
    del x28
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x100) * -1
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x100) * -1
    del x100
    x105 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x105 += einsum("ai,ijba->jb", l1.bb, x66)
    x109 += einsum("ia->ia", x105) * -1
    x333 += einsum("ia->ia", x105) * -1
    del x105
    x338 += einsum("ai,ijba->jb", l1.bb, x66) * -0.9999999999999993
    x375 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x375 += einsum("abij,jkcb->ikac", l2.abab, x66)
    x376 += einsum("ijab->ijab", x375) * -1
    del x375
    x77 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x77 += einsum("ai,jkab->ikjb", l1.bb, t2.bbbb)
    x90 += einsum("ijka->ijka", x77)
    x128 += einsum("ijka->ijka", x77)
    x253 += einsum("ijka->ijka", x77)
    del x77
    x78 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x78 += einsum("ia,waj->wji", t1.bb, lu11.bb)
    x79 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x79 += einsum("wia,wjk->jkia", u11.bb, x78)
    x90 -= einsum("ijka->ijka", x79)
    x128 -= einsum("ijka->ijka", x79)
    del x79
    x86 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x86 += einsum("wij->wij", x78)
    x106 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x106 += einsum("wij->wij", x78)
    x256 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x256 += einsum("ia,wij->wja", t1.bb, x78)
    x258 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x258 -= einsum("wia->wia", x256)
    del x256
    x296 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x296 += einsum("wia,wij->ja", u11.bb, x78)
    del x78
    x299 += einsum("ia->ia", x296)
    x300 += einsum("ia->ia", x296)
    del x296
    x80 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x80 += einsum("wia,wxaj->xji", u11.bb, lu12.bb)
    x81 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x81 += einsum("wia,wjk->jika", u11.bb, x80)
    x90 += einsum("ijka->ijka", x81)
    x128 += einsum("ijka->ijka", x81)
    del x81
    rdm2_f_vooo_bbbb -= einsum("ijka->ajik", x128)
    rdm2_f_vooo_bbbb += einsum("ijka->akij", x128)
    del x128
    x86 += einsum("wij->wij", x80)
    x87 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x87 += einsum("wia,wij->ja", u11.bb, x86)
    x88 += einsum("ia->ia", x87)
    del x87
    x118 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x118 += einsum("wia,wjk->ijka", u11.aa, x86)
    rdm2_f_oovo_bbaa -= einsum("ijka->jkai", x118)
    rdm2_f_vooo_aabb -= einsum("ijka->aijk", x118)
    del x118
    x328 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x328 += einsum("ia,wij->wja", t1.bb, x86)
    del x86
    x329 += einsum("wia->wia", x328)
    del x328
    x106 += einsum("wij->wij", x80)
    x107 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x107 += einsum("wia,wij->ja", u11.bb, x106)
    x109 += einsum("ia->ia", x107)
    x333 += einsum("ia->ia", x107)
    del x107
    x338 += einsum("wia,wij->ja", u11.bb, x106) * 0.9999999999999993
    del x106
    x244 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x244 += einsum("ia,wij->wja", t1.bb, x80)
    x245 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x245 += einsum("wia,wjb->ijba", u11.bb, x244)
    del x244
    x263 += einsum("ijab->ijab", x245)
    del x245
    x260 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x260 += einsum("wia,wij->ja", u11.bb, x80)
    del x80
    x262 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x262 -= einsum("ia->ia", x260)
    del x260
    x82 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x82 += einsum("ai,ijab->jb", l1.aa, t2.abab)
    x88 -= einsum("ia->ia", x82)
    x109 += einsum("ia->ia", x82) * -1
    x262 += einsum("ia->ia", x82)
    x333 += einsum("ia->ia", x82) * -1
    x338 += einsum("ia->ia", x82) * -0.9999999999999993
    del x82
    x84 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x84 += einsum("ijab->jiab", t2.bbbb)
    x84 -= einsum("ijab->jiba", t2.bbbb)
    x85 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x85 += einsum("ai,ijba->jb", l1.bb, x84)
    x88 -= einsum("ia->ia", x85)
    del x85
    x90 -= einsum("ij,ka->jika", delta_oo.bb, x88)
    rdm2_f_oovo_bbbb += einsum("ijka->ijak", x90)
    rdm2_f_oovo_bbbb -= einsum("ijka->ikaj", x90)
    del x90
    rdm2_f_vooo_bbbb += einsum("ij,ka->ajik", delta_oo.bb, x88)
    rdm2_f_vooo_bbbb -= einsum("ij,ka->akij", delta_oo.bb, x88)
    del x88
    x327 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x327 += einsum("wai,ijba->wjb", lu11.bb, x84)
    x329 -= einsum("wia->wia", x327)
    del x327
    x91 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x91 += einsum("w,wia->ia", ls1, u11.bb)
    x94 += einsum("ia->ia", x91)
    x109 += einsum("ia->ia", x91) * -1
    x299 += einsum("ia->ia", x91) * -1
    x300 += einsum("ia->ia", x91) * -1
    x333 += einsum("ia->ia", x91) * -1
    x338 += einsum("ia->ia", x91) * -0.9999999999999993
    del x91
    x92 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x92 += einsum("wx,wxia->ia", ls2, u12.bb)
    x94 += einsum("ia->ia", x92) * 0.5
    x109 += einsum("ia->ia", x92) * -0.5
    x299 += einsum("ia->ia", x92) * -0.5
    x300 += einsum("ia->ia", x92) * -0.5
    rdm2_f_vovo_bbbb += einsum("ia,jb->ajbi", t1.bb, x300)
    rdm2_f_vovo_bbbb += einsum("ia,jb->bjai", t1.bb, x300) * -1
    del x300
    x333 += einsum("ia->ia", x92) * -0.5
    x338 += einsum("ia->ia", x92) * -0.49999999999999967
    del x92
    rdm2_f_vovo_aabb += einsum("ia,jb->aibj", t1.aa, x338) * -1
    del x338
    x94 += einsum("ia->ia", t1.bb)
    rdm2_f_oovo_bbbb += einsum("ij,ka->jkai", delta_oo.bb, x94) * -1
    rdm2_f_oovo_bbbb += einsum("ij,ka->jiak", delta_oo.bb, x94)
    rdm2_f_vooo_bbbb += einsum("ij,ka->aijk", delta_oo.bb, x94) * -1
    rdm2_f_vooo_bbbb += einsum("ij,ka->akji", delta_oo.bb, x94)
    del x94
    x97 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x97 += einsum("ai,jkab->ijkb", l1.aa, t2.abab)
    x319 += einsum("ijka->ijka", x97)
    x320 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x320 += einsum("ia,ijkb->jkab", t1.aa, x319)
    del x319
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x320) * -1
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x320) * -1
    del x320
    rdm2_f_oovo_aabb -= einsum("ijka->ijak", x97)
    rdm2_f_vooo_bbaa -= einsum("ijka->akij", x97)
    del x97
    x109 += einsum("ia->ia", t1.bb) * -1
    rdm2_f_oovo_aabb += einsum("ij,ka->jiak", delta_oo.aa, x109) * -1
    rdm2_f_vooo_bbaa += einsum("ij,ka->akji", delta_oo.aa, x109) * -1
    del x109
    x113 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x113 += einsum("ai,jkba->jikb", l1.bb, t2.abab)
    x315 += einsum("ijka->ijka", x113) * -1
    x316 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x316 += einsum("ia,jikb->jkba", t1.bb, x315)
    del x315
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x316)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x316)
    del x316
    rdm2_f_oovo_bbaa -= einsum("ijka->jkai", x113)
    rdm2_f_vooo_aabb -= einsum("ijka->aijk", x113)
    del x113
    x124 += einsum("ia->ia", t1.aa) * -1
    rdm2_f_oovo_bbaa += einsum("ij,ka->jiak", delta_oo.bb, x124) * -1
    rdm2_f_vooo_aabb += einsum("ij,ka->akji", delta_oo.bb, x124) * -1
    del x124
    x131 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x131 += einsum("abij,kjcb->ikac", l2.abab, t2.abab)
    x206 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x206 += einsum("ijab->ijab", x131)
    x303 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x303 += einsum("ijab->ijab", x131)
    x353 += einsum("ijab->ijab", x131) * 2
    rdm2_f_oovv_aaaa -= einsum("ijab->ijba", x131)
    rdm2_f_ovvo_aaaa += einsum("ijab->iabj", x131)
    rdm2_f_voov_aaaa += einsum("ijab->bjia", x131)
    rdm2_f_vvoo_aaaa -= einsum("ijab->baij", x131)
    del x131
    x132 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x132 += einsum("wxai,wxjb->ijab", lu12.aa, u12.aa)
    x353 += einsum("ijab->ijab", x132)
    rdm2_f_oovv_aaaa += einsum("ijab->ijba", x132) * -0.5
    rdm2_f_ovvo_aaaa += einsum("ijab->iabj", x132) * 0.5
    rdm2_f_voov_aaaa += einsum("ijab->bjia", x132) * 0.5
    rdm2_f_vvoo_aaaa += einsum("ijab->baij", x132) * -0.5
    del x132
    x133 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x133 += einsum("wai,wjb->ijab", lu11.aa, u11.aa)
    rdm2_f_oovv_aaaa -= einsum("ijab->ijba", x133)
    rdm2_f_ovvo_aaaa += einsum("ijab->iabj", x133)
    rdm2_f_voov_aaaa += einsum("ijab->bjia", x133)
    rdm2_f_vvoo_aaaa -= einsum("ijab->baij", x133)
    del x133
    x134 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x134 += einsum("abij->jiab", l2.aaaa)
    x134 -= einsum("abij->jiba", l2.aaaa)
    x135 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x135 += einsum("ijab,ikac->kjcb", x134, x39)
    del x39
    rdm2_f_oovv_aaaa += einsum("ijab->jiab", x135)
    rdm2_f_vvoo_aaaa += einsum("ijab->abji", x135)
    del x135
    x173 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x173 += einsum("ijab,ikac->kjcb", t2.abab, x134)
    x283 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x283 -= einsum("ijab,ikac->jkbc", t2.abab, x173)
    x284 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x284 += einsum("ijab->ijab", x283)
    del x283
    rdm2_f_ovvo_aabb -= einsum("ijab->iabj", x173)
    rdm2_f_voov_bbaa -= einsum("ijab->bjia", x173)
    del x173
    x222 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x222 += einsum("ijab,ikac->jkbc", t2.aaaa, x134)
    x223 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x223 -= einsum("ijab,kica->jkbc", t2.aaaa, x222)
    del x222
    x227 += einsum("ijab->ijab", x223)
    del x223
    x224 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x224 += einsum("ijab,ikbc->jkac", t2.aaaa, x134)
    x225 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x225 -= einsum("ijab,kicb->jkac", t2.aaaa, x224)
    del x224
    x227 += einsum("ijab->jiba", x225)
    del x225
    x138 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x138 += einsum("ai,ib->ab", l1.aa, t1.aa)
    x144 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x144 += einsum("ab->ab", x138) * 2
    x163 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x163 += einsum("ab->ab", x138)
    x348 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x348 += einsum("ab->ab", x138)
    del x138
    x139 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x139 += einsum("wai,wib->ab", lu11.aa, u11.aa)
    x144 += einsum("ab->ab", x139) * 2
    x163 += einsum("ab->ab", x139)
    x196 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x196 += einsum("ab,ijca->ijcb", x139, t2.aaaa)
    x219 -= einsum("ijab->ijab", x196)
    del x196
    x311 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x311 += einsum("ab->ab", x139)
    x348 += einsum("ab->ab", x139)
    del x139
    x349 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x349 += einsum("ia,bc->ibac", t1.aa, x348)
    del x348
    x140 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x140 += einsum("wxai,wxib->ab", lu12.aa, u12.aa)
    x144 += einsum("ab->ab", x140)
    x163 += einsum("ab->ab", x140) * 0.5
    x187 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x187 += einsum("ab->ab", x140)
    x311 += einsum("ab->ab", x140) * 0.5
    del x140
    x141 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x141 += einsum("abij,ijcb->ac", l2.abab, t2.abab)
    x144 += einsum("ab->ab", x141) * 2
    x163 += einsum("ab->ab", x141)
    x187 += einsum("ab->ab", x141) * 2
    x311 += einsum("ab->ab", x141)
    del x141
    x142 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x142 += einsum("abij->jiab", l2.aaaa)
    x142 += einsum("abij->jiba", l2.aaaa) * -1
    x143 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x143 += einsum("ijab,ijac->bc", t2.aaaa, x142) * 2
    x144 += einsum("ab->ba", x143) * -1
    rdm2_f_oovv_aaaa += einsum("ij,ab->jiba", delta_oo.aa, x144) * 0.5
    rdm2_f_ovvo_aaaa += einsum("ij,ab->jabi", delta_oo.aa, x144) * -0.5
    rdm2_f_voov_aaaa += einsum("ij,ab->bija", delta_oo.aa, x144) * -0.5
    rdm2_f_vvoo_aaaa += einsum("ij,ab->baji", delta_oo.aa, x144) * 0.5
    del x144
    x187 += einsum("ab->ba", x143) * -1
    del x143
    x188 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x188 += einsum("ab,ijac->ijcb", x187, t2.aaaa) * 0.5
    x194 += einsum("ijab->jiab", x188)
    del x188
    x355 += einsum("ia,bc->ibac", t1.aa, x187) * 0.5
    del x187
    x162 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x162 += einsum("ijab,ijac->bc", t2.aaaa, x142)
    x163 += einsum("ab->ba", x162) * -1
    rdm2_f_oovv_bbaa += einsum("ij,ab->jiba", delta_oo.bb, x163)
    rdm2_f_vvoo_aabb += einsum("ij,ab->baji", delta_oo.bb, x163)
    rdm2_f_vovv_bbaa += einsum("ia,bc->aicb", t1.bb, x163)
    rdm2_f_vvvo_aabb += einsum("ia,bc->cbai", t1.bb, x163)
    del x163
    x311 += einsum("ab->ba", x162) * -1
    del x162
    x312 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x312 += einsum("ab,ijac->ijbc", x311, t2.abab)
    del x311
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x312) * -1
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x312) * -1
    del x312
    x352 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x352 += einsum("ijab,ikac->kjcb", x142, x6) * 2
    del x142
    del x6
    x353 += einsum("ijab->jiba", x352)
    del x352
    x354 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x354 += einsum("ia,ijbc->jabc", t1.aa, x353) * 0.5
    del x353
    x355 += einsum("iabc->ibac", x354) * -1
    del x354
    rdm2_f_vovv_aaaa = np.zeros((nvir[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_vovv_aaaa += einsum("iabc->bica", x355)
    rdm2_f_vovv_aaaa += einsum("iabc->ciba", x355) * -1
    rdm2_f_vvvo_aaaa = np.zeros((nvir[0], nvir[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_vvvo_aaaa += einsum("iabc->baci", x355) * -1
    rdm2_f_vvvo_aaaa += einsum("iabc->cabi", x355)
    del x355
    x145 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x145 += einsum("abij,ikac->jkbc", l2.abab, t2.abab)
    x249 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x249 += einsum("ijab->ijab", x145)
    x366 += einsum("ijab->ijab", x145) * 2
    rdm2_f_oovv_bbbb -= einsum("ijab->ijba", x145)
    rdm2_f_ovvo_bbbb += einsum("ijab->iabj", x145)
    rdm2_f_voov_bbbb += einsum("ijab->bjia", x145)
    rdm2_f_vvoo_bbbb -= einsum("ijab->baij", x145)
    del x145
    x146 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x146 += einsum("wxai,wxjb->ijab", lu12.bb, u12.bb)
    x366 += einsum("ijab->ijab", x146)
    rdm2_f_oovv_bbbb += einsum("ijab->ijba", x146) * -0.5
    rdm2_f_ovvo_bbbb += einsum("ijab->iabj", x146) * 0.5
    rdm2_f_voov_bbbb += einsum("ijab->bjia", x146) * 0.5
    rdm2_f_vvoo_bbbb += einsum("ijab->baij", x146) * -0.5
    del x146
    x147 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x147 += einsum("wai,wjb->ijab", lu11.bb, u11.bb)
    rdm2_f_oovv_bbbb -= einsum("ijab->ijba", x147)
    rdm2_f_ovvo_bbbb += einsum("ijab->iabj", x147)
    rdm2_f_voov_bbbb += einsum("ijab->bjia", x147)
    rdm2_f_vvoo_bbbb -= einsum("ijab->baij", x147)
    del x147
    x148 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x148 -= einsum("ijab->jiab", t2.bbbb)
    x148 += einsum("ijab->jiba", t2.bbbb)
    x174 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x174 += einsum("abij,jkcb->ikac", l2.abab, x148)
    rdm2_f_ovvo_aabb -= einsum("ijab->iabj", x174)
    rdm2_f_voov_bbaa -= einsum("ijab->bjia", x174)
    del x174
    x252 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x252 += einsum("ijab,kila->jklb", x148, x251)
    del x251
    x253 += einsum("ijka->jkia", x252)
    del x252
    x254 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x254 += einsum("ia,ijkb->jkab", t1.bb, x253)
    del x253
    x263 += einsum("ijab->ijab", x254)
    del x254
    x257 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x257 += einsum("wai,ijba->wjb", lu11.bb, x148)
    x258 -= einsum("wia->wia", x257)
    del x257
    x261 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x261 += einsum("ai,ijba->jb", l1.bb, x148)
    x262 -= einsum("ia->ia", x261)
    del x261
    x263 += einsum("ia,jb->ijab", t1.bb, x262)
    del x262
    x149 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x149 -= einsum("abij->jiab", l2.bbbb)
    x149 += einsum("abij->jiba", l2.bbbb)
    x150 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x150 += einsum("ijab,ikca->jkbc", x148, x149)
    rdm2_f_oovv_bbbb += einsum("ijab->jiab", x150)
    rdm2_f_vvoo_bbbb += einsum("ijab->abji", x150)
    del x150
    x179 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x179 += einsum("ijab,jkcb->ikac", t2.abab, x149)
    x226 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x226 -= einsum("ijab,kjcb->ikac", t2.abab, x179)
    x227 += einsum("ijab->ijab", x226)
    del x226
    x302 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x302 -= einsum("ijab->ijab", x179)
    rdm2_f_ovvo_bbaa -= einsum("ijab->jbai", x179)
    rdm2_f_voov_aabb -= einsum("ijab->aijb", x179)
    del x179
    x180 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x180 += einsum("ijab,ikbc->kjca", x149, x84)
    rdm2_f_ovvo_bbbb += einsum("ijab->jbai", x180)
    rdm2_f_voov_bbbb += einsum("ijab->aijb", x180)
    del x180
    x248 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x248 += einsum("ijab,ikca->jkbc", t2.bbbb, x149)
    x249 -= einsum("ijab->jiba", x248)
    x250 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x250 += einsum("ijab,ikbc->jkac", t2.bbbb, x249)
    del x249
    x263 -= einsum("ijab->jiba", x250)
    del x250
    x287 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x287 -= einsum("ijab,kica->jkbc", t2.bbbb, x248)
    del x248
    x290 += einsum("ijab->jiba", x287)
    del x287
    x288 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x288 += einsum("ijab,ikcb->jkac", t2.bbbb, x149)
    del x149
    x289 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x289 -= einsum("ijab,kicb->jkac", t2.bbbb, x288)
    del x288
    x290 += einsum("ijab->ijab", x289)
    del x289
    x153 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x153 += einsum("ai,ib->ab", l1.bb, t1.bb)
    x159 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x159 += einsum("ab->ab", x153)
    x361 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x361 += einsum("ab->ab", x153)
    del x153
    x154 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x154 += einsum("wai,wib->ab", lu11.bb, u11.bb)
    x159 += einsum("ab->ab", x154)
    x240 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x240 += einsum("ab,ijac->jicb", x154, t2.bbbb)
    x263 -= einsum("ijab->ijab", x240)
    del x240
    x309 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x309 += einsum("ab->ab", x154)
    x361 += einsum("ab->ab", x154)
    del x154
    x362 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x362 += einsum("ia,bc->ibac", t1.bb, x361)
    del x361
    x155 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x155 += einsum("wxai,wxib->ab", lu12.bb, u12.bb)
    x159 += einsum("ab->ab", x155) * 0.5
    x272 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x272 += einsum("ab->ab", x155)
    x309 += einsum("ab->ab", x155) * 0.5
    x368 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x368 += einsum("ab->ab", x155)
    del x155
    x156 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x156 += einsum("abij,ijac->bc", l2.abab, t2.abab)
    x159 += einsum("ab->ab", x156)
    x266 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x266 += einsum("ab,ijac->jibc", x156, t2.bbbb)
    x280 += einsum("ijab->ijab", x266) * -1
    del x266
    x309 += einsum("ab->ab", x156)
    x368 += einsum("ab->ab", x156) * 2
    del x156
    x157 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x157 += einsum("abij->jiab", l2.bbbb) * -1
    x157 += einsum("abij->jiba", l2.bbbb)
    x158 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x158 += einsum("ijab,ijbc->ac", t2.bbbb, x157)
    x159 += einsum("ab->ba", x158) * -1
    rdm2_f_oovv_bbbb += einsum("ij,ab->jiba", delta_oo.bb, x159)
    rdm2_f_oovv_aabb += einsum("ij,ab->jiba", delta_oo.aa, x159)
    rdm2_f_ovvo_bbbb += einsum("ij,ab->jabi", delta_oo.bb, x159) * -1
    rdm2_f_voov_bbbb += einsum("ij,ab->bija", delta_oo.bb, x159) * -1
    rdm2_f_vvoo_bbbb += einsum("ij,ab->baji", delta_oo.bb, x159)
    rdm2_f_vvoo_bbaa += einsum("ij,ab->baji", delta_oo.aa, x159)
    rdm2_f_vovv_aabb += einsum("ia,bc->aicb", t1.aa, x159)
    rdm2_f_vvvo_bbaa += einsum("ia,bc->cbai", t1.aa, x159)
    del x159
    x309 += einsum("ab->ba", x158) * -1
    del x158
    x310 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x310 += einsum("ab,ijca->ijcb", x309, t2.abab)
    del x309
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x310) * -1
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x310) * -1
    del x310
    x271 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x271 += einsum("ijab,ijbc->ac", t2.bbbb, x157) * 2
    x272 += einsum("ab->ba", x271) * -1
    x273 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x273 += einsum("ab,ijac->ijcb", x272, t2.bbbb) * 0.5
    del x272
    x280 += einsum("ijab->jiab", x273)
    del x273
    x368 += einsum("ab->ba", x271) * -1
    del x271
    x369 += einsum("ia,bc->ibac", t1.bb, x368) * 0.5
    del x368
    x365 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x365 += einsum("ijab,ikbc->kjca", x157, x66) * 2
    del x66
    del x157
    x366 += einsum("ijab->jiba", x365)
    del x365
    x367 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x367 += einsum("ia,ijbc->jabc", t1.bb, x366) * 0.5
    del x366
    x369 += einsum("iabc->ibac", x367) * -1
    del x367
    rdm2_f_vovv_bbbb = np.zeros((nvir[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_vovv_bbbb += einsum("iabc->bica", x369)
    rdm2_f_vovv_bbbb += einsum("iabc->ciba", x369) * -1
    rdm2_f_vvvo_bbbb = np.zeros((nvir[1], nvir[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_vvvo_bbbb += einsum("iabc->baci", x369) * -1
    rdm2_f_vvvo_bbbb += einsum("iabc->cabi", x369)
    del x369
    x160 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x160 += einsum("abij,ikcb->jkac", l2.abab, t2.abab)
    x372 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x372 += einsum("ia,ijbc->jbca", t1.bb, x160)
    rdm2_f_vovv_bbaa += einsum("iabc->ciba", x372) * -1
    rdm2_f_vvvo_aabb += einsum("iabc->baci", x372) * -1
    del x372
    rdm2_f_oovv_bbaa -= einsum("ijab->ijba", x160)
    rdm2_f_vvoo_aabb -= einsum("ijab->baij", x160)
    del x160
    x164 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x164 += einsum("abij,kjac->ikbc", l2.abab, t2.abab)
    x301 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x301 += einsum("ijab,ikbc->kjac", t2.abab, x164)
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x301)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x301)
    del x301
    x382 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x382 += einsum("ia,ijbc->jabc", t1.aa, x164)
    rdm2_f_vovv_aabb += einsum("iabc->aicb", x382) * -1
    rdm2_f_vvvo_bbaa += einsum("iabc->cbai", x382) * -1
    del x382
    rdm2_f_oovv_aabb -= einsum("ijab->ijba", x164)
    rdm2_f_vvoo_bbaa -= einsum("ijab->baij", x164)
    del x164
    x166 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x166 += einsum("ijab->jiab", t2.aaaa)
    x166 -= einsum("ijab->jiba", t2.aaaa)
    x167 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x167 += einsum("ijab,ikac->jkbc", x134, x166)
    del x134
    x303 += einsum("ijab->ijab", x167)
    x304 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x304 += einsum("ijab,ikac->kjcb", t2.abab, x303)
    del x303
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x304)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x304)
    del x304
    rdm2_f_ovvo_aaaa += einsum("ijab->iabj", x167)
    rdm2_f_voov_aaaa += einsum("ijab->bjia", x167)
    del x167
    x178 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x178 += einsum("abij,ikac->kjcb", l2.abab, x166)
    x302 -= einsum("ijab->ijab", x178)
    rdm2_f_vovo_bbaa += einsum("ijab,jkcb->ckai", x302, x84)
    del x84
    rdm2_f_vovo_aabb -= einsum("ijab,kicb->ckaj", x148, x302)
    del x148
    del x302
    rdm2_f_ovvo_bbaa -= einsum("ijab->jbai", x178)
    rdm2_f_voov_aabb -= einsum("ijab->aijb", x178)
    del x178
    x208 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x208 += einsum("ijab,iklb->jkla", x166, x168)
    del x168
    x209 += einsum("ijka->jkia", x208)
    del x208
    x210 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x210 += einsum("ia,ijkb->jkab", t1.aa, x209)
    del x209
    x219 += einsum("ijab->ijab", x210)
    del x210
    x213 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x213 += einsum("wai,ijab->wjb", lu11.aa, x166)
    x214 -= einsum("wia->wia", x213)
    del x213
    x217 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x217 += einsum("ai,ijab->jb", l1.aa, x166)
    del x166
    x218 -= einsum("ia->ia", x217)
    del x217
    x219 += einsum("ia,jb->ijab", t1.aa, x218)
    del x218
    x170 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x170 += einsum("wxai,wxjb->ijab", lu12.aa, u12.bb)
    x376 += einsum("ijab->ijab", x170) * -0.5
    rdm2_f_ovvo_aabb += einsum("ijab->iabj", x170) * 0.5
    rdm2_f_voov_bbaa += einsum("ijab->bjia", x170) * 0.5
    del x170
    x171 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x171 += einsum("wai,wjb->ijab", lu11.aa, u11.bb)
    rdm2_f_ovvo_aabb += einsum("ijab->iabj", x171)
    rdm2_f_voov_bbaa += einsum("ijab->bjia", x171)
    del x171
    x175 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x175 += einsum("wxai,wxjb->jiba", lu12.bb, u12.aa)
    x386 += einsum("ijab->ijab", x175) * -0.5
    rdm2_f_ovvo_bbaa += einsum("ijab->jbai", x175) * 0.5
    rdm2_f_voov_aabb += einsum("ijab->aijb", x175) * 0.5
    del x175
    x176 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x176 += einsum("wai,wjb->jiba", lu11.bb, u11.aa)
    rdm2_f_ovvo_bbaa += einsum("ijab->jbai", x176)
    rdm2_f_voov_aabb += einsum("ijab->aijb", x176)
    del x176
    x183 = np.zeros((nbos, nbos, nocc[0], nvir[0]), dtype=types[float])
    x183 += einsum("wxai,jiba->wxjb", lu12.bb, t2.abab)
    x185 += einsum("wxia->xwia", x183)
    del x183
    x186 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x186 += einsum("wxia,wxjb->ijab", u12.aa, x185) * 0.5
    x194 += einsum("ijab->jiba", x186) * -1
    del x186
    rdm2_f_vovo_aaaa += einsum("ijab->aibj", x194) * -1
    rdm2_f_vovo_aaaa += einsum("ijab->biaj", x194)
    rdm2_f_vovo_aaaa += einsum("ijab->ajbi", x194)
    rdm2_f_vovo_aaaa += einsum("ijab->bjai", x194) * -1
    del x194
    x308 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x308 += einsum("wxia,wxjb->jiba", u12.bb, x185) * 0.5
    del x185
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x308)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x308)
    del x308
    x202 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x202 += einsum("abij,kiac->kjcb", l2.abab, t2.aaaa)
    x203 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x203 += einsum("ijab,kjcb->kica", t2.abab, x202)
    del x202
    x219 -= einsum("ijab->ijab", x203)
    del x203
    x204 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x204 -= einsum("abij->jiab", l2.aaaa)
    x204 += einsum("abij->jiba", l2.aaaa)
    x205 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x205 += einsum("ijab,ikbc->jkac", t2.aaaa, x204)
    del x204
    x206 -= einsum("ijab->jiba", x205)
    del x205
    x207 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x207 += einsum("ijab,ikac->jkbc", t2.aaaa, x206)
    del x206
    x219 += einsum("ijab->ijab", x207)
    del x207
    x212 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x212 += einsum("wai,jiba->wjb", lu11.bb, t2.abab)
    x214 += einsum("wia->wia", x212)
    x215 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x215 += einsum("wia,wjb->ijab", u11.aa, x214)
    del x214
    x219 += einsum("ijab->jiba", x215)
    del x215
    rdm2_f_vovo_aaaa += einsum("ijab->aibj", x219)
    rdm2_f_vovo_aaaa -= einsum("ijab->biaj", x219)
    rdm2_f_vovo_aaaa -= einsum("ijab->ajbi", x219)
    rdm2_f_vovo_aaaa += einsum("ijab->bjai", x219)
    del x219
    x325 -= einsum("wia->wia", x212)
    del x212
    x227 += einsum("ijab->jiba", t2.aaaa)
    rdm2_f_vovo_aaaa -= einsum("ijab->biaj", x227)
    rdm2_f_vovo_aaaa += einsum("ijab->aibj", x227)
    del x227
    x228 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x228 += einsum("wx,xia->wia", ls2, u11.aa)
    x229 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x229 += einsum("wia,wjb->jiba", u11.aa, x228)
    rdm2_f_vovo_aaaa -= einsum("ijab->biaj", x229)
    rdm2_f_vovo_aaaa += einsum("ijab->bjai", x229)
    del x229
    x325 -= einsum("wia->wia", x228)
    del x228
    x326 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x326 += einsum("wia,wjb->jiba", u11.bb, x325)
    del x325
    rdm2_f_vovo_bbaa -= einsum("ijab->bjai", x326)
    rdm2_f_vovo_aabb -= einsum("ijab->aibj", x326)
    del x326
    x231 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x231 += einsum("ijab->jiba", t2.aaaa)
    x231 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    x232 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x232 += einsum("ijkl,ijab->klab", x0, x231)
    del x0
    del x231
    x233 += einsum("ijab->jiba", x232)
    del x232
    rdm2_f_vovo_aaaa += einsum("ijab->ajbi", x233) * -1
    rdm2_f_vovo_aaaa += einsum("ijab->bjai", x233)
    del x233
    x237 += einsum("ia->ia", t1.aa) * -1
    rdm2_f_vovo_aaaa += einsum("ia,jb->ajbi", t1.aa, x237)
    rdm2_f_vovo_aaaa += einsum("ia,jb->bjai", t1.aa, x237) * -1
    del x237
    x246 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x246 += einsum("abij,kjcb->ikac", l2.abab, t2.bbbb)
    x247 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x247 += einsum("ijab,ikac->jkbc", t2.abab, x246)
    del x246
    x263 += einsum("ijab->ijab", x247)
    del x247
    x255 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x255 += einsum("wai,ijab->wjb", lu11.aa, t2.abab)
    x258 += einsum("wia->wia", x255)
    x259 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x259 += einsum("wia,wjb->ijab", u11.bb, x258)
    del x258
    x263 += einsum("ijab->jiba", x259)
    del x259
    rdm2_f_vovo_bbbb += einsum("ijab->aibj", x263)
    rdm2_f_vovo_bbbb -= einsum("ijab->biaj", x263)
    rdm2_f_vovo_bbbb -= einsum("ijab->ajbi", x263)
    rdm2_f_vovo_bbbb += einsum("ijab->bjai", x263)
    del x263
    x329 -= einsum("wia->wia", x255)
    del x255
    x330 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x330 += einsum("wia,wjb->ijab", u11.aa, x329)
    del x329
    rdm2_f_vovo_bbaa -= einsum("ijab->bjai", x330)
    rdm2_f_vovo_aabb -= einsum("ijab->aibj", x330)
    del x330
    x267 = np.zeros((nbos, nbos, nocc[1], nvir[1]), dtype=types[float])
    x267 += einsum("wxai,ijab->wxjb", lu12.aa, t2.abab)
    x269 += einsum("wxia->xwia", x267)
    del x267
    x270 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x270 += einsum("wxia,wxjb->ijab", u12.bb, x269) * 0.5
    x280 += einsum("ijab->jiba", x270) * -1
    del x270
    rdm2_f_vovo_bbbb += einsum("ijab->aibj", x280) * -1
    rdm2_f_vovo_bbbb += einsum("ijab->biaj", x280)
    rdm2_f_vovo_bbbb += einsum("ijab->ajbi", x280)
    rdm2_f_vovo_bbbb += einsum("ijab->bjai", x280) * -1
    del x280
    x307 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x307 += einsum("wxia,wxjb->ijab", u12.aa, x269) * 0.5
    del x269
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x307)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x307)
    del x307
    x281 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x281 += einsum("wx,xia->wia", ls2, u11.bb)
    x282 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x282 += einsum("wia,wjb->jiba", u11.bb, x281)
    del x281
    x284 += einsum("ijab->jiba", x282)
    del x282
    rdm2_f_vovo_bbbb -= einsum("ijab->ajbi", x284)
    rdm2_f_vovo_bbbb += einsum("ijab->aibj", x284)
    del x284
    x290 += einsum("ijab->jiba", t2.bbbb)
    rdm2_f_vovo_bbbb -= einsum("ijab->biaj", x290)
    rdm2_f_vovo_bbbb += einsum("ijab->aibj", x290)
    del x290
    x299 += einsum("ia->ia", t1.bb) * -1
    rdm2_f_vovo_bbbb += einsum("ia,jb->biaj", t1.bb, x299)
    rdm2_f_vovo_bbbb += einsum("ia,jb->aibj", t1.bb, x299) * -1
    del x299
    x333 += einsum("ia->ia", t1.bb) * -1
    rdm2_f_vovo_bbaa += einsum("ia,jb->bjai", t1.aa, x333) * -1
    del x333
    x337 += einsum("ia->ia", t1.aa) * -1
    rdm2_f_vovo_aabb += einsum("ia,jb->bjai", t1.bb, x337) * -1
    del x337
    x339 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x339 += einsum("ia,bcji->jbca", t1.aa, l2.aaaa)
    x391 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x391 += einsum("ia,ibcd->cbda", t1.aa, x339)
    x392 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x392 += einsum("abcd->badc", x391)
    del x391
    rdm2_f_ovvv_aaaa = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_ovvv_aaaa += einsum("iabc->iacb", x339)
    rdm2_f_ovvv_aaaa -= einsum("iabc->ibca", x339)
    rdm2_f_vvov_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_vvov_aaaa -= einsum("iabc->caib", x339)
    rdm2_f_vvov_aaaa += einsum("iabc->cbia", x339)
    del x339
    x340 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x340 += einsum("ia,bcji->jbca", t1.bb, l2.bbbb)
    x396 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x396 += einsum("ia,ibcd->cbda", t1.bb, x340)
    x397 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x397 += einsum("abcd->badc", x396)
    del x396
    rdm2_f_ovvv_bbbb = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_ovvv_bbbb += einsum("iabc->iacb", x340)
    rdm2_f_ovvv_bbbb -= einsum("iabc->ibca", x340)
    rdm2_f_vvov_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_vvov_bbbb -= einsum("iabc->caib", x340)
    rdm2_f_vvov_bbbb += einsum("iabc->cbia", x340)
    del x340
    x341 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x341 += einsum("ia,bcij->jbac", t1.aa, l2.abab)
    rdm2_f_ovvv_bbaa = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_ovvv_bbaa += einsum("iabc->icba", x341)
    rdm2_f_vvov_aabb = np.zeros((nvir[0], nvir[0], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_vvov_aabb += einsum("iabc->baic", x341)
    del x341
    x342 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x342 += einsum("ia,bcji->jbca", t1.bb, l2.abab)
    x394 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x394 += einsum("ia,ibcd->bacd", t1.aa, x342)
    rdm2_f_vvvv_bbaa = np.zeros((nvir[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_vvvv_bbaa += einsum("abcd->dcba", x394)
    rdm2_f_vvvv_aabb = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_vvvv_aabb += einsum("abcd->badc", x394)
    del x394
    rdm2_f_ovvv_aabb = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_ovvv_aabb += einsum("iabc->iacb", x342)
    rdm2_f_vvov_bbaa = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_vvov_bbaa += einsum("iabc->cbia", x342)
    del x342
    x343 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x343 += einsum("ai,jibc->jabc", l1.aa, t2.aaaa)
    x349 += einsum("iabc->iabc", x343)
    del x343
    x344 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x344 += einsum("ia,wbi->wba", t1.aa, lu11.aa)
    x345 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x345 += einsum("wia,wbc->ibca", u11.aa, x344)
    x349 -= einsum("iabc->iabc", x345)
    del x345
    x378 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x378 += einsum("wab->wab", x344)
    del x344
    x346 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x346 += einsum("wia,xwbi->xba", u11.aa, lu12.aa)
    x347 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x347 += einsum("wia,wbc->ibac", u11.aa, x346)
    x349 += einsum("iabc->iabc", x347)
    del x347
    rdm2_f_vovv_aaaa += einsum("iabc->bica", x349)
    rdm2_f_vovv_aaaa -= einsum("iabc->ciba", x349)
    rdm2_f_vvvo_aaaa -= einsum("iabc->baci", x349)
    rdm2_f_vvvo_aaaa += einsum("iabc->cabi", x349)
    del x349
    x378 += einsum("wab->wab", x346)
    del x346
    x379 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x379 += einsum("wia,wbc->ibca", u11.bb, x378)
    del x378
    rdm2_f_vovv_bbaa += einsum("iabc->ciba", x379)
    rdm2_f_vvvo_aabb += einsum("iabc->baci", x379)
    del x379
    x356 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x356 += einsum("ai,jibc->jabc", l1.bb, t2.bbbb)
    x362 += einsum("iabc->iabc", x356)
    del x356
    x357 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x357 += einsum("ia,wbi->wba", t1.bb, lu11.bb)
    x358 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x358 += einsum("wia,wbc->ibca", u11.bb, x357)
    x362 -= einsum("iabc->iabc", x358)
    del x358
    x388 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x388 += einsum("wab->wab", x357)
    del x357
    x359 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x359 += einsum("wia,xwbi->xba", u11.bb, lu12.bb)
    x360 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x360 += einsum("wia,wbc->ibac", u11.bb, x359)
    x362 += einsum("iabc->iabc", x360)
    del x360
    rdm2_f_vovv_bbbb += einsum("iabc->bica", x362)
    rdm2_f_vovv_bbbb -= einsum("iabc->ciba", x362)
    rdm2_f_vvvo_bbbb -= einsum("iabc->baci", x362)
    rdm2_f_vvvo_bbbb += einsum("iabc->cabi", x362)
    del x362
    x388 += einsum("wab->wab", x359)
    del x359
    x389 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x389 += einsum("wia,wbc->iabc", u11.aa, x388)
    del x388
    rdm2_f_vovv_aabb += einsum("iabc->aicb", x389)
    rdm2_f_vvvo_bbaa += einsum("iabc->cbai", x389)
    del x389
    x370 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x370 += einsum("ai,ijbc->jabc", l1.aa, t2.abab)
    rdm2_f_vovv_bbaa += einsum("iabc->ciba", x370)
    rdm2_f_vvvo_aabb += einsum("iabc->baci", x370)
    del x370
    x373 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x373 += einsum("abij->jiab", l2.aaaa) * -1
    x373 += einsum("abij->jiba", l2.aaaa)
    x374 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x374 += einsum("ijab,ikac->kjcb", t2.abab, x373)
    del x373
    x376 += einsum("ijab->ijab", x374) * -1
    del x374
    x377 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x377 += einsum("ia,ijbc->jabc", t1.aa, x376)
    del x376
    rdm2_f_vovv_bbaa += einsum("iabc->ciab", x377) * -1
    rdm2_f_vvvo_aabb += einsum("iabc->abci", x377) * -1
    del x377
    x380 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x380 += einsum("ai,jibc->jbac", l1.bb, t2.abab)
    rdm2_f_vovv_aabb += einsum("iabc->aicb", x380)
    rdm2_f_vvvo_bbaa += einsum("iabc->cbai", x380)
    del x380
    x384 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x384 += einsum("abij->jiab", l2.bbbb)
    x384 += einsum("abij->jiba", l2.bbbb) * -1
    x385 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x385 += einsum("ijab,jkcb->ikac", t2.abab, x384)
    del x384
    x386 += einsum("ijab->ijab", x385) * -1
    del x385
    x387 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x387 += einsum("ia,jibc->jbac", t1.bb, x386)
    del x386
    rdm2_f_vovv_aabb += einsum("iabc->aibc", x387) * -1
    rdm2_f_vvvo_bbaa += einsum("iabc->bcai", x387) * -1
    del x387
    x390 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x390 += einsum("abij,ijcd->abcd", l2.aaaa, t2.aaaa)
    x392 += einsum("abcd->badc", x390)
    del x390
    rdm2_f_vvvv_aaaa = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_vvvv_aaaa += einsum("abcd->dacb", x392) * -1
    rdm2_f_vvvv_aaaa += einsum("abcd->cadb", x392)
    del x392
    x393 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x393 += einsum("abij,ijcd->acbd", l2.abab, t2.abab)
    rdm2_f_vvvv_bbaa += einsum("abcd->dcba", x393)
    rdm2_f_vvvv_aabb += einsum("abcd->badc", x393)
    del x393
    x395 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x395 += einsum("abij,ijcd->abcd", l2.bbbb, t2.bbbb)
    x397 += einsum("abcd->badc", x395)
    del x395
    rdm2_f_vvvv_bbbb = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_vvvv_bbbb += einsum("abcd->dacb", x397) * -1
    rdm2_f_vvvv_bbbb += einsum("abcd->cadb", x397)
    del x397
    rdm2_f_oooo_aaaa += einsum("ij,kl->jilk", delta_oo.aa, delta_oo.aa)
    rdm2_f_oooo_aaaa -= einsum("ij,kl->ljik", delta_oo.aa, delta_oo.aa)
    rdm2_f_oooo_bbbb += einsum("ij,kl->jilk", delta_oo.bb, delta_oo.bb)
    rdm2_f_oooo_bbbb -= einsum("ij,kl->ljik", delta_oo.bb, delta_oo.bb)
    rdm2_f_ooov_aaaa += einsum("ij,ak->jika", delta_oo.aa, l1.aa)
    rdm2_f_ooov_aaaa -= einsum("ij,ak->kija", delta_oo.aa, l1.aa)
    rdm2_f_ooov_bbbb += einsum("ij,ak->jika", delta_oo.bb, l1.bb)
    rdm2_f_ooov_bbbb -= einsum("ij,ak->kija", delta_oo.bb, l1.bb)
    rdm2_f_ooov_aabb += einsum("ij,ak->jika", delta_oo.aa, l1.bb)
    rdm2_f_ooov_bbaa += einsum("ij,ak->jika", delta_oo.bb, l1.aa)
    rdm2_f_ovoo_aaaa -= einsum("ij,ak->jaki", delta_oo.aa, l1.aa)
    rdm2_f_ovoo_aaaa += einsum("ij,ak->kaji", delta_oo.aa, l1.aa)
    rdm2_f_ovoo_bbaa += einsum("ij,ak->kaji", delta_oo.aa, l1.bb)
    rdm2_f_ovoo_aabb += einsum("ij,ak->kaji", delta_oo.bb, l1.aa)
    rdm2_f_ovoo_bbbb -= einsum("ij,ak->jaki", delta_oo.bb, l1.bb)
    rdm2_f_ovoo_bbbb += einsum("ij,ak->kaji", delta_oo.bb, l1.bb)
    rdm2_f_oovo_aaaa -= einsum("ij,ka->jkai", delta_oo.aa, t1.aa)
    rdm2_f_vooo_aaaa += einsum("ij,ka->akji", delta_oo.aa, t1.aa)
    rdm2_f_ovov_aaaa = np.zeros((nocc[0], nvir[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_ovov_aaaa -= einsum("abij->jaib", l2.aaaa)
    rdm2_f_ovov_aaaa += einsum("abij->jbia", l2.aaaa)
    rdm2_f_ovov_bbbb = np.zeros((nocc[1], nvir[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_ovov_bbbb -= einsum("abij->jaib", l2.bbbb)
    rdm2_f_ovov_bbbb += einsum("abij->jbia", l2.bbbb)
    rdm2_f_ovov_bbaa = np.zeros((nocc[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_ovov_bbaa += einsum("abij->jbia", l2.abab)
    rdm2_f_ovov_aabb = np.zeros((nocc[0], nvir[0], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_ovov_aabb += einsum("abij->iajb", l2.abab)
    rdm2_f_oovv_aaaa -= einsum("ai,jb->ijba", l1.aa, t1.aa)
    rdm2_f_oovv_bbbb -= einsum("ai,jb->ijba", l1.bb, t1.bb)
    rdm2_f_ovvo_aaaa += einsum("ai,jb->iabj", l1.aa, t1.aa)
    rdm2_f_ovvo_aabb += einsum("ai,jb->iabj", l1.aa, t1.bb)
    rdm2_f_ovvo_bbaa += einsum("ai,jb->iabj", l1.bb, t1.aa)
    rdm2_f_ovvo_bbbb += einsum("ai,jb->iabj", l1.bb, t1.bb)
    rdm2_f_voov_aaaa += einsum("ai,jb->bjia", l1.aa, t1.aa)
    rdm2_f_voov_bbaa += einsum("ai,jb->bjia", l1.aa, t1.bb)
    rdm2_f_voov_aabb += einsum("ai,jb->bjia", l1.bb, t1.aa)
    rdm2_f_voov_bbbb += einsum("ai,jb->bjia", l1.bb, t1.bb)
    rdm2_f_vvoo_aaaa -= einsum("ai,jb->baij", l1.aa, t1.aa)
    rdm2_f_vvoo_bbbb -= einsum("ai,jb->baij", l1.bb, t1.bb)
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", t2.abab)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", t2.abab)

    rdm2_f_aaaa = pack_2e(rdm2_f_oooo_aaaa, rdm2_f_ooov_aaaa, rdm2_f_oovo_aaaa, rdm2_f_ovoo_aaaa, rdm2_f_vooo_aaaa, rdm2_f_oovv_aaaa, rdm2_f_ovov_aaaa, rdm2_f_ovvo_aaaa, rdm2_f_voov_aaaa, rdm2_f_vovo_aaaa, rdm2_f_vvoo_aaaa, rdm2_f_ovvv_aaaa, rdm2_f_vovv_aaaa, rdm2_f_vvov_aaaa, rdm2_f_vvvo_aaaa, rdm2_f_vvvv_aaaa)
    rdm2_f_aabb = pack_2e(rdm2_f_oooo_aabb, rdm2_f_ooov_aabb, rdm2_f_oovo_aabb, rdm2_f_ovoo_aabb, rdm2_f_vooo_aabb, rdm2_f_oovv_aabb, rdm2_f_ovov_aabb, rdm2_f_ovvo_aabb, rdm2_f_voov_aabb, rdm2_f_vovo_aabb, rdm2_f_vvoo_aabb, rdm2_f_ovvv_aabb, rdm2_f_vovv_aabb, rdm2_f_vvov_aabb, rdm2_f_vvvo_aabb, rdm2_f_vvvv_aabb)
    rdm2_f_bbaa = pack_2e(rdm2_f_oooo_bbaa, rdm2_f_ooov_bbaa, rdm2_f_oovo_bbaa, rdm2_f_ovoo_bbaa, rdm2_f_vooo_bbaa, rdm2_f_oovv_bbaa, rdm2_f_ovov_bbaa, rdm2_f_ovvo_bbaa, rdm2_f_voov_bbaa, rdm2_f_vovo_bbaa, rdm2_f_vvoo_bbaa, rdm2_f_ovvv_bbaa, rdm2_f_vovv_bbaa, rdm2_f_vvov_bbaa, rdm2_f_vvvo_bbaa, rdm2_f_vvvv_bbaa)
    rdm2_f_bbbb = pack_2e(rdm2_f_oooo_bbbb, rdm2_f_ooov_bbbb, rdm2_f_oovo_bbbb, rdm2_f_ovoo_bbbb, rdm2_f_vooo_bbbb, rdm2_f_oovv_bbbb, rdm2_f_ovov_bbbb, rdm2_f_ovvo_bbbb, rdm2_f_voov_bbbb, rdm2_f_vovo_bbbb, rdm2_f_vvoo_bbbb, rdm2_f_ovvv_bbbb, rdm2_f_vovv_bbbb, rdm2_f_vvov_bbbb, rdm2_f_vvvo_bbbb, rdm2_f_vvvv_bbbb)

    rdm2_f.aaaa = rdm2_f_aaaa
    rdm2_f.aabb = rdm2_f_aabb
    rdm2_f.bbaa = rdm2_f_bbaa
    rdm2_f.bbbb = rdm2_f_bbbb

    return rdm2_f

def make_sing_b_dm(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, lu12=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        aa = Namespace(
            boo=g.aa.boo.transpose(0, 2, 1),
            bov=g.aa.bvo.transpose(0, 2, 1),
            bvo=g.aa.bov.transpose(0, 2, 1),
            bvv=g.aa.bvv.transpose(0, 2, 1),
        ),
        bb = Namespace(
            boo=g.bb.boo.transpose(0, 2, 1),
            bov=g.bb.bvo.transpose(0, 2, 1),
            bvo=g.bb.bov.transpose(0, 2, 1),
            bvv=g.bb.bvv.transpose(0, 2, 1),
        ),
    )

    # Single boson DM
    dm_b_cre = np.zeros((nbos), dtype=types[float])
    dm_b_cre += einsum("w->w", ls1)
    dm_b_des = np.zeros((nbos), dtype=types[float])
    dm_b_des += einsum("ai,wia->w", l1.aa, u11.aa)
    dm_b_des += einsum("wai,xwia->x", lu11.aa, u12.aa)
    dm_b_des += einsum("w->w", s1)
    dm_b_des += einsum("ai,wia->w", l1.bb, u11.bb)
    dm_b_des += einsum("w,xw->x", ls1, s2)
    dm_b_des += einsum("wai,xwia->x", lu11.bb, u12.bb)

    dm_b = np.array([dm_b_cre, dm_b_des])

    return dm_b

def make_rdm1_b(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, lu12=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        aa = Namespace(
            boo=g.aa.boo.transpose(0, 2, 1),
            bov=g.aa.bvo.transpose(0, 2, 1),
            bvo=g.aa.bov.transpose(0, 2, 1),
            bvv=g.aa.bvv.transpose(0, 2, 1),
        ),
        bb = Namespace(
            boo=g.bb.boo.transpose(0, 2, 1),
            bov=g.bb.bvo.transpose(0, 2, 1),
            bvo=g.bb.bov.transpose(0, 2, 1),
            bvv=g.bb.bvv.transpose(0, 2, 1),
        ),
    )

    # Boson 1RDM
    rdm1_b = np.zeros((nbos, nbos), dtype=types[float])
    rdm1_b += einsum("wxai,yxia->wy", lu12.bb, u12.bb)
    rdm1_b += einsum("wai,xia->wx", lu11.aa, u11.aa)
    rdm1_b += einsum("wx,yx->wy", ls2, s2)
    rdm1_b += einsum("wxai,yxia->wy", lu12.aa, u12.aa)
    rdm1_b += einsum("w,x->wx", ls1, s1)
    rdm1_b += einsum("wai,xia->wx", lu11.bb, u11.bb)

    return rdm1_b

def make_eb_coup_rdm(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, lu12=None, **kwargs):
    rdm_eb = Namespace()

    # Get boson coupling creation array:
    gc = Namespace(
        aa = Namespace(
            boo=g.aa.boo.transpose(0, 2, 1),
            bov=g.aa.bvo.transpose(0, 2, 1),
            bvo=g.aa.bov.transpose(0, 2, 1),
            bvv=g.aa.bvv.transpose(0, 2, 1),
        ),
        bb = Namespace(
            boo=g.bb.boo.transpose(0, 2, 1),
            bov=g.bb.bvo.transpose(0, 2, 1),
            bvo=g.bb.bov.transpose(0, 2, 1),
            bvv=g.bb.bvv.transpose(0, 2, 1),
        ),
    )

    delta_oo = Namespace()
    delta_oo.aa = np.eye(nocc[0])
    delta_oo.bb = np.eye(nocc[1])
    delta_vv = Namespace()
    delta_vv.aa = np.eye(nvir[0])
    delta_vv.bb = np.eye(nvir[1])

    # Boson-fermion coupling RDM
    x0 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x0 += einsum("wia,xwaj->xji", u11.aa, lu12.aa)
    x7 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x7 += einsum("wij->wij", x0)
    x16 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x16 += einsum("wx,xij->wij", s2, x0)
    x64 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x64 += einsum("wij->wij", x16)
    rdm_eb_des_oo_aa = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    rdm_eb_des_oo_aa -= einsum("wij->wji", x16)
    del x16
    x75 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x75 += einsum("wij->wij", x0)
    rdm_eb_cre_oo_aa = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    rdm_eb_cre_oo_aa -= einsum("wij->wji", x0)
    del x0
    x1 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x1 += einsum("ia,waj->wji", t1.aa, lu11.aa)
    x7 += einsum("wij->wij", x1)
    rdm_eb_cre_ov_aa = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    rdm_eb_cre_ov_aa -= einsum("ia,wij->wja", t1.aa, x7)
    rdm_eb_des_ov_aa = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    rdm_eb_des_ov_aa -= einsum("wij,wxia->xja", x7, u12.aa)
    del x7
    x75 += einsum("wij->wij", x1)
    x77 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x77 += einsum("wia,wij->ja", u11.aa, x75)
    del x75
    rdm_eb_cre_oo_aa -= einsum("wij->wji", x1)
    del x1
    x2 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x2 += einsum("wia,xwaj->xji", u11.bb, lu12.bb)
    x11 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x11 += einsum("wij->wij", x2)
    x36 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x36 += einsum("wx,xij->wij", s2, x2)
    x83 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x83 += einsum("wij->wij", x36)
    rdm_eb_des_oo_bb = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    rdm_eb_des_oo_bb -= einsum("wij->wji", x36)
    del x36
    x89 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x89 += einsum("wij->wij", x2)
    rdm_eb_cre_oo_bb = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    rdm_eb_cre_oo_bb -= einsum("wij->wji", x2)
    del x2
    x3 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x3 += einsum("ia,waj->wji", t1.bb, lu11.bb)
    x11 += einsum("wij->wij", x3)
    rdm_eb_cre_ov_bb = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    rdm_eb_cre_ov_bb -= einsum("ia,wij->wja", t1.bb, x11)
    rdm_eb_des_ov_bb = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    rdm_eb_des_ov_bb -= einsum("wij,wxia->xja", x11, u12.bb)
    del x11
    x89 += einsum("wij->wij", x3)
    x91 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x91 += einsum("wia,wij->ja", u11.bb, x89)
    del x89
    rdm_eb_cre_oo_bb -= einsum("wij->wji", x3)
    del x3
    x4 = np.zeros((nbos, nbos, nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum("ia,wxaj->wxji", t1.aa, lu12.aa)
    x5 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x5 += einsum("wia,xwij->xja", u11.aa, x4)
    rdm_eb_cre_ov_aa -= einsum("wia->wia", x5)
    rdm_eb_des_ov_aa -= einsum("wx,xia->wia", s2, x5)
    del x5
    x77 += einsum("wxia,wxij->ja", u12.aa, x4) * 0.5
    del x4
    x6 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x6 += einsum("ijab->jiab", t2.aaaa)
    x6 -= einsum("ijab->jiba", t2.aaaa)
    rdm_eb_cre_ov_aa -= einsum("wai,ijab->wjb", lu11.aa, x6)
    x8 = np.zeros((nbos, nbos, nocc[1], nocc[1]), dtype=types[float])
    x8 += einsum("ia,wxaj->wxji", t1.bb, lu12.bb)
    x9 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x9 += einsum("wia,xwij->xja", u11.bb, x8)
    rdm_eb_cre_ov_bb -= einsum("wia->wia", x9)
    rdm_eb_des_ov_bb -= einsum("wx,xia->wia", s2, x9)
    del x9
    x91 += einsum("wxia,wxij->ja", u12.bb, x8) * 0.5
    del x8
    x10 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x10 += einsum("ijab->jiab", t2.bbbb)
    x10 -= einsum("ijab->jiba", t2.bbbb)
    rdm_eb_cre_ov_bb -= einsum("wai,ijab->wjb", lu11.bb, x10)
    x12 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum("wia,xwbi->xba", u11.aa, lu12.aa)
    rdm_eb_cre_vv_aa = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    rdm_eb_cre_vv_aa += einsum("wab->wab", x12)
    rdm_eb_des_ov_aa -= einsum("wab,xwia->xib", x12, u12.aa)
    rdm_eb_des_vv_aa = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    rdm_eb_des_vv_aa += einsum("wx,xab->wab", s2, x12)
    del x12
    x13 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x13 += einsum("wia,xwbi->xba", u11.bb, lu12.bb)
    rdm_eb_cre_vv_bb = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    rdm_eb_cre_vv_bb += einsum("wab->wab", x13)
    rdm_eb_des_ov_bb -= einsum("wab,xwia->xib", x13, u12.bb)
    rdm_eb_des_vv_bb = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    rdm_eb_des_vv_bb += einsum("wx,xab->wab", s2, x13)
    del x13
    x14 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x14 += einsum("wai,xwja->xij", lu11.aa, u12.aa)
    x64 += einsum("wij->wij", x14)
    rdm_eb_des_oo_aa -= einsum("wij->wji", x14)
    del x14
    x15 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x15 += einsum("ai,wja->wij", l1.aa, u11.aa)
    x64 += einsum("wij->wij", x15)
    rdm_eb_des_oo_aa -= einsum("wij->wji", x15)
    del x15
    x17 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x17 += einsum("wx,xai->wia", s2, lu11.aa)
    x21 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x21 += einsum("wia->wia", x17)
    x52 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x52 += einsum("wia->wia", x17)
    rdm_eb_des_vo_aa = np.zeros((nbos, nvir[0], nocc[0]), dtype=types[float])
    rdm_eb_des_vo_aa += einsum("wia->wai", x17)
    del x17
    x18 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x18 += einsum("wia,baji->wjb", u11.bb, l2.abab)
    x21 += einsum("wia->wia", x18)
    x52 += einsum("wia->wia", x18)
    rdm_eb_des_vo_aa += einsum("wia->wai", x18)
    del x18
    x19 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x19 += einsum("abij->jiab", l2.aaaa)
    x19 += einsum("abij->jiba", l2.aaaa) * -1
    x20 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x20 += einsum("wia,ijab->wjb", u11.aa, x19)
    x21 += einsum("wia->wia", x20) * -1
    del x20
    rdm_eb_des_oo_aa += einsum("ia,wja->wij", t1.aa, x21) * -1
    rdm_eb_des_vv_aa += einsum("ia,wib->wba", t1.aa, x21)
    del x21
    x62 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x62 += einsum("ijab,ijcb->ac", t2.aaaa, x19)
    del x19
    x63 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x63 += einsum("ab->ba", x62) * -1
    x92 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x92 += einsum("ab->ba", x62) * -1
    del x62
    x22 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x22 += einsum("ai,ja->ij", l1.aa, t1.aa)
    x28 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x28 += einsum("ij->ij", x22)
    x65 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x65 += einsum("ij->ij", x22)
    x76 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x76 += einsum("ij->ij", x22)
    del x22
    x23 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x23 += einsum("wai,wja->ij", lu11.aa, u11.aa)
    x28 += einsum("ij->ij", x23)
    x65 += einsum("ij->ij", x23)
    x76 += einsum("ij->ij", x23)
    del x23
    x24 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x24 += einsum("wxai,wxja->ij", lu12.aa, u12.aa)
    x28 += einsum("ij->ij", x24) * 0.5
    x65 += einsum("ij->ij", x24) * 0.5
    x76 += einsum("ij->ij", x24) * 0.5
    del x24
    x25 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x25 += einsum("abij,kjab->ik", l2.abab, t2.abab)
    x28 += einsum("ij->ij", x25)
    x65 += einsum("ij->ij", x25)
    x76 += einsum("ij->ij", x25)
    del x25
    x26 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x26 += einsum("ijab->jiab", t2.aaaa)
    x26 += einsum("ijab->jiba", t2.aaaa) * -1
    x27 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x27 += einsum("abij,ikab->jk", l2.aaaa, x26)
    x28 += einsum("ij->ij", x27) * -1
    x65 += einsum("ij->ij", x27) * -1
    del x27
    rdm_eb_des_ov_aa += einsum("ij,wia->wja", x65, u11.aa) * -1
    del x65
    x76 += einsum("abij,ikab->jk", l2.aaaa, x26) * -1
    x77 += einsum("ia,ij->ja", t1.aa, x76)
    del x76
    x77 += einsum("ai,ijba->jb", l1.aa, x26) * -1
    del x26
    x28 += einsum("ij->ji", delta_oo.aa) * -1
    rdm_eb_des_oo_aa += einsum("w,ij->wji", s1, x28) * -1
    del x28
    x29 = np.zeros((nbos), dtype=types[float])
    x29 += einsum("w,xw->x", ls1, s2)
    x34 = np.zeros((nbos), dtype=types[float])
    x34 += einsum("w->w", x29)
    del x29
    x30 = np.zeros((nbos), dtype=types[float])
    x30 += einsum("ai,wia->w", l1.aa, u11.aa)
    x34 += einsum("w->w", x30)
    del x30
    x31 = np.zeros((nbos), dtype=types[float])
    x31 += einsum("ai,wia->w", l1.bb, u11.bb)
    x34 += einsum("w->w", x31)
    del x31
    x32 = np.zeros((nbos), dtype=types[float])
    x32 += einsum("wai,xwia->x", lu11.aa, u12.aa)
    x34 += einsum("w->w", x32)
    del x32
    x33 = np.zeros((nbos), dtype=types[float])
    x33 += einsum("wai,xwia->x", lu11.bb, u12.bb)
    x34 += einsum("w->w", x33)
    del x33
    rdm_eb_des_oo_aa += einsum("w,ij->wji", x34, delta_oo.aa)
    rdm_eb_des_oo_bb += einsum("w,ij->wji", x34, delta_oo.bb)
    rdm_eb_des_ov_aa += einsum("w,ia->wia", x34, t1.aa)
    rdm_eb_des_ov_bb += einsum("w,ia->wia", x34, t1.bb)
    del x34
    x35 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x35 += einsum("wai,xwja->xij", lu11.bb, u12.bb)
    x83 += einsum("wij->wij", x35)
    rdm_eb_des_oo_bb -= einsum("wij->wji", x35)
    del x35
    x37 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x37 += einsum("ai,wja->wij", l1.bb, u11.bb)
    x83 += einsum("wij->wij", x37)
    rdm_eb_des_oo_bb -= einsum("wij->wji", x37)
    del x37
    x38 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x38 += einsum("wx,xai->wia", s2, lu11.bb)
    x42 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x42 += einsum("wia->wia", x38)
    x55 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x55 += einsum("wia->wia", x38)
    rdm_eb_des_vo_bb = np.zeros((nbos, nvir[1], nocc[1]), dtype=types[float])
    rdm_eb_des_vo_bb += einsum("wia->wai", x38)
    del x38
    x39 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x39 += einsum("wia,abij->wjb", u11.aa, l2.abab)
    x42 += einsum("wia->wia", x39)
    x55 += einsum("wia->wia", x39)
    rdm_eb_des_vo_bb += einsum("wia->wai", x39)
    del x39
    x40 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x40 += einsum("abij->jiab", l2.bbbb)
    x40 += einsum("abij->jiba", l2.bbbb) * -1
    x41 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x41 += einsum("wia,ijab->wjb", u11.bb, x40)
    del x40
    x42 += einsum("wia->wia", x41) * -1
    del x41
    rdm_eb_des_oo_bb += einsum("ia,wja->wij", t1.bb, x42) * -1
    rdm_eb_des_vv_bb += einsum("ia,wib->wba", t1.bb, x42)
    del x42
    x43 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x43 += einsum("ai,ja->ij", l1.bb, t1.bb)
    x49 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x49 += einsum("ij->ij", x43)
    x84 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x84 += einsum("ij->ij", x43)
    x90 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x90 += einsum("ij->ij", x43)
    del x43
    x44 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x44 += einsum("wai,wja->ij", lu11.bb, u11.bb)
    x49 += einsum("ij->ij", x44)
    x84 += einsum("ij->ij", x44)
    x90 += einsum("ij->ij", x44)
    del x44
    x45 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x45 += einsum("wxai,wxja->ij", lu12.bb, u12.bb)
    x49 += einsum("ij->ij", x45) * 0.5
    x84 += einsum("ij->ij", x45) * 0.5
    x90 += einsum("ij->ij", x45) * 0.5
    del x45
    x46 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x46 += einsum("abij,ikab->jk", l2.abab, t2.abab)
    x49 += einsum("ij->ij", x46)
    x84 += einsum("ij->ij", x46)
    x90 += einsum("ij->ij", x46)
    del x46
    x47 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x47 += einsum("ijab->jiab", t2.bbbb) * -1
    x47 += einsum("ijab->jiba", t2.bbbb)
    x48 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x48 += einsum("abij,ikba->jk", l2.bbbb, x47)
    x49 += einsum("ij->ij", x48) * -1
    x84 += einsum("ij->ij", x48) * -1
    del x48
    rdm_eb_des_ov_bb += einsum("ij,wia->wja", x84, u11.bb) * -1
    del x84
    x90 += einsum("abij,ikba->jk", l2.bbbb, x47) * -1
    del x47
    x91 += einsum("ia,ij->ja", t1.bb, x90)
    del x90
    x49 += einsum("ij->ji", delta_oo.bb) * -1
    rdm_eb_des_oo_bb += einsum("w,ij->wji", s1, x49) * -1
    del x49
    x50 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x50 += einsum("abij->jiab", l2.aaaa)
    x50 -= einsum("abij->jiba", l2.aaaa)
    x51 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x51 += einsum("wia,ijab->wjb", u11.aa, x50)
    del x50
    x52 -= einsum("wia->wia", x51)
    x64 += einsum("ia,wja->wji", t1.aa, x52)
    rdm_eb_des_ov_aa -= einsum("ia,wij->wja", t1.aa, x64)
    del x64
    rdm_eb_des_ov_aa -= einsum("wia,ijab->wjb", x52, x6)
    del x6
    rdm_eb_des_ov_bb += einsum("wia,ijab->wjb", x52, t2.abab)
    del x52
    rdm_eb_des_vo_aa -= einsum("wia->wai", x51)
    del x51
    x53 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x53 += einsum("abij->jiab", l2.bbbb)
    x53 -= einsum("abij->jiba", l2.bbbb)
    x54 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x54 += einsum("wia,ijab->wjb", u11.bb, x53)
    del x53
    x55 -= einsum("wia->wia", x54)
    x83 += einsum("ia,wja->wji", t1.bb, x55)
    rdm_eb_des_ov_bb -= einsum("ia,wij->wja", t1.bb, x83)
    del x83
    rdm_eb_des_ov_aa += einsum("wia,jiba->wjb", x55, t2.abab)
    rdm_eb_des_ov_bb -= einsum("wia,ijab->wjb", x55, x10)
    del x10
    del x55
    rdm_eb_des_vo_bb -= einsum("wia->wai", x54)
    del x54
    x56 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x56 += einsum("wia,xyai->xyw", u11.aa, lu12.aa)
    x58 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x58 += einsum("wxy->xwy", x56)
    del x56
    x57 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x57 += einsum("wia,xyai->xyw", u11.bb, lu12.bb)
    x58 += einsum("wxy->xwy", x57)
    del x57
    rdm_eb_des_ov_aa += einsum("wxy,wxia->yia", x58, u12.aa) * 0.5
    rdm_eb_des_ov_bb += einsum("wxy,wxia->yia", x58, u12.bb) * 0.5
    del x58
    x59 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x59 += einsum("wai,wib->ab", lu11.aa, u11.aa)
    x63 += einsum("ab->ab", x59)
    x92 += einsum("ab->ab", x59)
    del x59
    x60 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x60 += einsum("wxai,wxib->ab", lu12.aa, u12.aa)
    x63 += einsum("ab->ab", x60) * 0.5
    x92 += einsum("ab->ab", x60) * 0.5
    del x60
    x61 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x61 += einsum("abij,ijcb->ac", l2.abab, t2.abab)
    x63 += einsum("ab->ab", x61)
    rdm_eb_des_ov_aa += einsum("ab,wia->wib", x63, u11.aa) * -1
    del x63
    x92 += einsum("ab->ab", x61)
    del x61
    x66 = np.zeros((nbos, nbos), dtype=types[float])
    x66 += einsum("wx,yx->wy", ls2, s2)
    x71 = np.zeros((nbos, nbos), dtype=types[float])
    x71 += einsum("wx->wx", x66)
    del x66
    x67 = np.zeros((nbos, nbos), dtype=types[float])
    x67 += einsum("wai,xia->wx", lu11.aa, u11.aa)
    x71 += einsum("wx->wx", x67)
    del x67
    x68 = np.zeros((nbos, nbos), dtype=types[float])
    x68 += einsum("wai,xia->wx", lu11.bb, u11.bb)
    x71 += einsum("wx->wx", x68)
    del x68
    x69 = np.zeros((nbos, nbos), dtype=types[float])
    x69 += einsum("wxai,ywia->xy", lu12.aa, u12.aa)
    x71 += einsum("wx->wx", x69)
    del x69
    x70 = np.zeros((nbos, nbos), dtype=types[float])
    x70 += einsum("wxai,ywia->xy", lu12.bb, u12.bb)
    x71 += einsum("wx->wx", x70)
    del x70
    rdm_eb_des_ov_aa += einsum("wx,wia->xia", x71, u11.aa)
    rdm_eb_des_ov_bb += einsum("wx,wia->xia", x71, u11.bb)
    del x71
    x72 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x72 += einsum("ia,abjk->jikb", t1.aa, l2.abab)
    x77 += einsum("ijab,ikjb->ka", t2.abab, x72)
    del x72
    x73 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x73 += einsum("ia,bajk->jkib", t1.aa, l2.aaaa)
    x74 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x74 += einsum("ijka->ijka", x73) * -1
    x74 += einsum("ijka->jika", x73)
    del x73
    x77 += einsum("ijab,jikb->ka", t2.aaaa, x74) * -1
    del x74
    x77 += einsum("ia->ia", t1.aa) * -1
    x77 += einsum("w,wia->ia", ls1, u11.aa) * -1
    x77 += einsum("wx,wxia->ia", ls2, u12.aa) * -0.5
    x77 += einsum("ai,jiba->jb", l1.bb, t2.abab) * -1
    rdm_eb_des_ov_aa += einsum("w,ia->wia", s1, x77) * -1
    del x77
    x78 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x78 += einsum("wai,wib->ab", lu11.bb, u11.bb)
    x82 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x82 += einsum("ab->ab", x78)
    x93 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x93 += einsum("ab->ab", x78) * 2
    del x78
    x79 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x79 += einsum("wxai,wxib->ab", lu12.bb, u12.bb)
    x82 += einsum("ab->ab", x79) * 0.5
    x93 += einsum("ab->ab", x79)
    del x79
    x80 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x80 += einsum("abij,ijac->bc", l2.abab, t2.abab)
    x82 += einsum("ab->ab", x80)
    x93 += einsum("ab->ab", x80) * 2
    del x80
    x81 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x81 += einsum("abij->jiab", l2.bbbb) * -1
    x81 += einsum("abij->jiba", l2.bbbb)
    x82 += einsum("ijab,ijca->cb", t2.bbbb, x81) * -1
    rdm_eb_des_ov_bb += einsum("ab,wia->wib", x82, u11.bb) * -1
    del x82
    x93 += einsum("ijab,ijca->cb", t2.bbbb, x81) * -2
    del x81
    x85 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x85 += einsum("ia,bajk->jkib", t1.bb, l2.abab)
    x91 += einsum("ijab,ijka->kb", t2.abab, x85)
    del x85
    x86 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x86 += einsum("ia,bajk->jkib", t1.bb, l2.bbbb)
    x87 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x87 += einsum("ijka->ijka", x86) * -1
    x87 += einsum("ijka->jika", x86)
    del x86
    x91 += einsum("ijab,ijka->kb", t2.bbbb, x87) * -1
    del x87
    x88 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x88 += einsum("ijab->jiab", t2.bbbb)
    x88 += einsum("ijab->jiba", t2.bbbb) * -1
    x91 += einsum("ai,ijba->jb", l1.bb, x88) * -1
    del x88
    x91 += einsum("ia->ia", t1.bb) * -1
    x91 += einsum("w,wia->ia", ls1, u11.bb) * -1
    x91 += einsum("wx,wxia->ia", ls2, u12.bb) * -0.5
    x91 += einsum("ai,ijab->jb", l1.aa, t2.abab) * -1
    rdm_eb_des_ov_bb += einsum("w,ia->wia", s1, x91) * -1
    del x91
    x92 += einsum("ai,ib->ab", l1.aa, t1.aa)
    rdm_eb_des_vv_aa += einsum("w,ab->wab", s1, x92)
    del x92
    x93 += einsum("ai,ib->ab", l1.bb, t1.bb) * 2
    rdm_eb_des_vv_bb += einsum("w,ab->wab", s1, x93) * 0.5
    del x93
    rdm_eb_cre_oo_aa += einsum("w,ij->wji", ls1, delta_oo.aa)
    rdm_eb_cre_oo_bb += einsum("w,ij->wji", ls1, delta_oo.bb)
    rdm_eb_cre_ov_aa += einsum("w,ia->wia", ls1, t1.aa)
    rdm_eb_cre_ov_aa += einsum("wai,jiba->wjb", lu11.bb, t2.abab)
    rdm_eb_cre_ov_aa += einsum("wx,xia->wia", ls2, u11.aa)
    rdm_eb_cre_ov_bb += einsum("wai,ijab->wjb", lu11.aa, t2.abab)
    rdm_eb_cre_ov_bb += einsum("w,ia->wia", ls1, t1.bb)
    rdm_eb_cre_ov_bb += einsum("wx,xia->wia", ls2, u11.bb)
    rdm_eb_cre_vo_aa = np.zeros((nbos, nvir[0], nocc[0]), dtype=types[float])
    rdm_eb_cre_vo_aa += einsum("wai->wai", lu11.aa)
    rdm_eb_cre_vo_bb = np.zeros((nbos, nvir[1], nocc[1]), dtype=types[float])
    rdm_eb_cre_vo_bb += einsum("wai->wai", lu11.bb)
    rdm_eb_cre_vv_aa += einsum("ia,wbi->wba", t1.aa, lu11.aa)
    rdm_eb_cre_vv_bb += einsum("ia,wbi->wba", t1.bb, lu11.bb)
    rdm_eb_des_ov_aa += einsum("w,xwia->xia", ls1, u12.aa)
    rdm_eb_des_ov_aa += einsum("wia->wia", u11.aa)
    rdm_eb_des_ov_bb += einsum("w,xwia->xia", ls1, u12.bb)
    rdm_eb_des_ov_bb += einsum("wia->wia", u11.bb)
    rdm_eb_des_vo_aa += einsum("w,ai->wai", s1, l1.aa)
    rdm_eb_des_vo_bb += einsum("w,ai->wai", s1, l1.bb)
    rdm_eb_des_vv_aa += einsum("ai,wib->wab", l1.aa, u11.aa)
    rdm_eb_des_vv_aa += einsum("wai,xwib->xab", lu11.aa, u12.aa)
    rdm_eb_des_vv_bb += einsum("ai,wib->wab", l1.bb, u11.bb)
    rdm_eb_des_vv_bb += einsum("wai,xwib->xab", lu11.bb, u12.bb)

    rdm_eb_aa = np.array([
            np.block([[rdm_eb_cre_oo_aa, rdm_eb_cre_ov_aa], [rdm_eb_cre_vo_aa, rdm_eb_cre_vv_aa]]),
            np.block([[rdm_eb_des_oo_aa, rdm_eb_des_ov_aa], [rdm_eb_des_vo_aa, rdm_eb_des_vv_aa]]),
    ])
    rdm_eb_bb = np.array([
            np.block([[rdm_eb_cre_oo_bb, rdm_eb_cre_ov_bb], [rdm_eb_cre_vo_bb, rdm_eb_cre_vv_bb]]),
            np.block([[rdm_eb_des_oo_bb, rdm_eb_des_ov_bb], [rdm_eb_des_vo_bb, rdm_eb_des_vv_bb]]),
    ])

    rdm_eb.aa = rdm_eb_aa
    rdm_eb.bb = rdm_eb_bb

    return rdm_eb

