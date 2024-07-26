# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, **kwargs):
    # Energy
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x0 += einsum("iajb->jiab", v.bbbb.ovov)
    x0 += einsum("iajb->jiba", v.bbbb.ovov) * -1
    x1 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x1 += einsum("ijab->jiba", t2.bbbb)
    x1 += einsum("ia,jb->ijba", t1.bb, t1.bb) * -1
    e_cc = 0
    e_cc += einsum("ijab,ijab->", x0, x1) * -0.5
    del x0
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum("iajb->jiab", v.aaaa.ovov)
    x2 += einsum("iajb->jiba", v.aaaa.ovov) * -1
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x3 += einsum("ijab->jiba", t2.aaaa)
    x3 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    e_cc += einsum("ijab,ijab->", x2, x3) * -0.5
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
    e_cc += einsum("wia,wia->", g.aa.bov, u11.aa)
    e_cc += einsum("ia,ia->", f.aa.ov, t1.aa)
    e_cc += einsum("ia,ia->", f.bb.ov, t1.bb)
    e_cc += einsum("wia,wia->", g.bb.bov, u11.bb)

    return e_cc

def update_amps(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, **kwargs):
    t1new = Namespace()
    t2new = Namespace()
    u11new = Namespace()

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

    # T1, T2, S1 and U11 amplitudes
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x0 += einsum("ia,jakb->ikjb", t1.aa, v.aaaa.ovov)
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum("ijka->ijka", x0) * -1
    x1 += einsum("ijka->ikja", x0)
    x67 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x67 -= einsum("ijka->ijka", x0)
    x67 += einsum("ijka->ikja", x0)
    x93 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x93 += einsum("ia,jkla->jilk", t1.aa, x0)
    x94 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x94 += einsum("ia,jkli->jkla", t1.aa, x93)
    x95 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x95 += einsum("ia,jkib->jkab", t1.aa, x94)
    del x94
    x102 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x102 += einsum("ijab->ijab", x95)
    del x95
    x134 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x134 += einsum("ijkl->ijkl", x93)
    del x93
    x261 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x261 += einsum("ijka->jkia", x0) * -1
    x261 += einsum("ijka->kjia", x0)
    del x0
    x1 += einsum("ijka->jika", v.aaaa.ooov)
    x1 += einsum("ijka->jkia", v.aaaa.ooov) * -1
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    t1new_aa += einsum("ijab,kjia->kb", t2.aaaa, x1) * -1
    del x1
    x2 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum("iabc->ibac", v.aaaa.ovvv)
    x2 += einsum("iabc->ibca", v.aaaa.ovvv) * -1
    x30 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x30 += einsum("ia,ibca->bc", t1.aa, x2)
    x31 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x31 += einsum("ab->ab", x30) * -1
    x275 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x275 += einsum("ab->ab", x30) * -1
    x307 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x307 += einsum("ab->ab", x30) * -1
    del x30
    x107 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x107 += einsum("ia,jbac->ijbc", t1.aa, x2)
    t1new_aa += einsum("ijab,icba->jc", t2.aaaa, x2) * -1
    del x2
    x3 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x3 += einsum("ia,jakb->ijkb", t1.aa, v.aabb.ovov)
    x4 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x4 += einsum("ijka->jika", x3)
    x122 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x122 += einsum("ijab,kljb->kila", t2.abab, x3)
    del x3
    x126 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x126 += einsum("ijka->ikja", x122)
    del x122
    x4 += einsum("ijka->ijka", v.aabb.ooov)
    x285 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x285 += einsum("ijab,iklb->kjla", t2.abab, x4)
    x289 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x289 += einsum("ijka->ikja", x285) * -1
    del x285
    t1new_aa += einsum("ijab,ikjb->ka", t2.abab, x4) * -1
    x5 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x5 += einsum("w,wia->ia", s1, g.aa.bov)
    x9 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x9 += einsum("ia->ia", x5)
    x71 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x71 += einsum("ia->ia", x5)
    x299 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x299 += einsum("ia->ia", x5)
    x308 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x308 += einsum("ia->ia", x5)
    del x5
    x6 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum("ia,jbia->jb", t1.bb, v.aabb.ovov)
    x9 += einsum("ia->ia", x6)
    x105 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x105 += einsum("ia,ja->ij", t1.aa, x6)
    x106 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x106 += einsum("ij,kjab->ikab", x105, t2.aaaa)
    del x105
    x130 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x130 += einsum("ijab->ijab", x106) * -1
    del x106
    x123 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x123 += einsum("ia,jkab->kjib", x6, t2.aaaa)
    x126 += einsum("ijka->ikja", x123) * -1
    del x123
    x299 += einsum("ia->ia", x6)
    x306 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x306 += einsum("ia->ia", x6)
    del x6
    x7 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x7 += einsum("iajb->jiab", v.aaaa.ovov) * -1
    x7 += einsum("iajb->jiba", v.aaaa.ovov)
    x8 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum("ia,ijba->jb", t1.aa, x7)
    x9 += einsum("ia->ia", x8) * -1
    x306 += einsum("ia->ia", x8) * -1
    del x8
    x307 += einsum("ia,ib->ab", t1.aa, x306) * -1
    del x306
    x23 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x23 += einsum("ijab,ikba->jk", t2.aaaa, x7)
    x27 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x27 += einsum("ij->ji", x23) * -1
    x128 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x128 += einsum("ij->ji", x23) * -1
    del x23
    x113 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x113 += einsum("ijab,ijca->bc", t2.aaaa, x7)
    x116 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x116 += einsum("ab->ab", x113) * -1
    del x113
    x233 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x233 += einsum("ijab,ikca->kjcb", t2.abab, x7)
    x234 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x234 += einsum("ijab->ijab", x233) * -1
    del x233
    x301 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x301 += einsum("wia,ijba->wjb", u11.aa, x7)
    del x7
    x302 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x302 += einsum("wia->wia", x301) * -1
    del x301
    x9 += einsum("ia->ia", f.aa.ov)
    x26 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x26 += einsum("ia,ja->ij", t1.aa, x9)
    x27 += einsum("ij->ji", x26)
    del x26
    x265 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x265 += einsum("ia,jkab->jikb", x9, t2.abab)
    x271 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x271 += einsum("ijka->jika", x265)
    del x265
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    t1new_bb += einsum("ia,ijab->jb", x9, t2.abab)
    x10 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x10 += einsum("ijab->jiab", t2.aaaa)
    x10 += einsum("ijab->jiba", t2.aaaa) * -1
    x108 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x108 += einsum("ijab,kica->jkbc", x10, x107) * -1
    del x107
    x130 += einsum("ijab->jiab", x108) * -1
    del x108
    t1new_aa += einsum("ia,ijba->jb", x9, x10)
    del x9
    x11 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x11 += einsum("w,wia->ia", s1, g.bb.bov)
    x15 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x15 += einsum("ia->ia", x11)
    x155 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x155 += einsum("ia->ia", x11)
    x298 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x298 += einsum("ia->ia", x11)
    x317 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x317 += einsum("ia->ia", x11)
    del x11
    x12 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x12 += einsum("ia,iajb->jb", t1.aa, v.aabb.ovov)
    x15 += einsum("ia->ia", x12)
    x204 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x204 += einsum("ia,jkba->jkib", x12, t2.bbbb)
    x208 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x208 += einsum("ijka->ikja", x204) * -1
    del x204
    x219 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x219 += einsum("ia,ja->ij", t1.bb, x12)
    x220 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x220 += einsum("ij->ij", x219)
    del x219
    x298 += einsum("ia->ia", x12)
    x315 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x315 += einsum("ia->ia", x12)
    del x12
    x13 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x13 += einsum("iajb->jiab", v.bbbb.ovov) * -1
    x13 += einsum("iajb->jiba", v.bbbb.ovov)
    x14 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x14 += einsum("ia,ijba->jb", t1.bb, x13)
    x15 += einsum("ia->ia", x14) * -1
    x315 += einsum("ia->ia", x14) * -1
    del x14
    x316 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x316 += einsum("ia,ib->ab", t1.bb, x315) * -1
    del x315
    x48 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x48 += einsum("ijab,ikba->kj", t2.bbbb, x13)
    x52 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x52 += einsum("ij->ij", x48) * -1
    del x48
    x210 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x210 += einsum("ijab,ijca->cb", t2.bbbb, x13)
    x213 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x213 += einsum("ab->ba", x210) * -1
    del x210
    x215 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x215 += einsum("ijab,ikab->kj", t2.bbbb, x13)
    x217 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x217 += einsum("ij->ij", x215) * -1
    del x215
    x277 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x277 += einsum("ijab,ijac->cb", t2.bbbb, x13)
    x278 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x278 += einsum("ab->ba", x277) * -1
    x316 += einsum("ab->ba", x277) * -1
    del x277
    x304 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x304 += einsum("wia,ijba->wjb", u11.bb, x13)
    x305 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x305 += einsum("wia->wia", x304) * -1
    del x304
    x15 += einsum("ia->ia", f.bb.ov)
    x51 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x51 += einsum("ia,ja->ij", t1.bb, x15)
    x52 += einsum("ij->ji", x51)
    del x51
    x286 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x286 += einsum("ia,jkba->jkib", x15, t2.abab)
    x289 += einsum("ijka->ikja", x286)
    del x286
    t1new_aa += einsum("ia,jiba->jb", x15, t2.abab)
    x16 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x16 += einsum("iabj->ijba", v.aaaa.ovvo)
    x16 -= einsum("ijab->ijab", v.aaaa.oovv)
    x73 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x73 += einsum("ia,jkba->ijkb", t1.aa, x16)
    x74 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x74 -= einsum("ijka->jika", x73)
    del x73
    t1new_aa += einsum("ia,ijba->jb", t1.aa, x16)
    u11new_aa = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    u11new_aa += einsum("wia,ijba->wjb", u11.aa, x16)
    x17 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x17 += einsum("ia,wja->wji", t1.aa, g.aa.bov)
    x18 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x18 += einsum("wij->wij", x17)
    del x17
    x18 += einsum("wij->wij", g.aa.boo)
    x82 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x82 += einsum("ia,wij->wja", t1.aa, x18)
    x83 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x83 -= einsum("wia->wia", x82)
    del x82
    t1new_aa -= einsum("wia,wij->ja", u11.aa, x18)
    del x18
    x19 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x19 += einsum("w,wij->ij", s1, g.aa.boo)
    x27 += einsum("ij->ij", x19)
    x77 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x77 += einsum("ij->ji", x19)
    del x19
    x20 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x20 += einsum("wia,wja->ij", g.aa.bov, u11.aa)
    x27 += einsum("ij->ij", x20)
    x90 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x90 += einsum("ij->ij", x20)
    del x20
    x21 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x21 += einsum("ia,jkia->jk", t1.bb, v.aabb.ooov)
    x27 += einsum("ij->ij", x21)
    x90 += einsum("ij->ij", x21)
    del x21
    x91 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x91 += einsum("ij,ikab->kjab", x90, t2.aaaa)
    del x90
    x92 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x92 -= einsum("ijab->ijba", x91)
    del x91
    x22 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x22 += einsum("ijab,kajb->ik", t2.abab, v.aabb.ovov)
    x27 += einsum("ij->ji", x22)
    x128 += einsum("ij->ji", x22)
    del x22
    x24 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x24 += einsum("ijka->ikja", v.aaaa.ooov)
    x24 += einsum("ijka->kija", v.aaaa.ooov) * -1
    x25 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x25 += einsum("ia,ijka->jk", t1.aa, x24)
    x27 += einsum("ij->ij", x25) * -1
    x128 += einsum("ij->ij", x25) * -1
    del x25
    x129 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x129 += einsum("ij,ikab->kjab", x128, t2.aaaa)
    del x128
    x130 += einsum("ijab->ijba", x129)
    del x129
    x309 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x309 += einsum("wia,ijka->wjk", u11.aa, x24) * -1
    del x24
    x27 += einsum("ij->ij", f.aa.oo)
    x291 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x291 += einsum("ij,ikab->jkab", x27, t2.abab)
    t2new_baba = np.zeros((nocc[1], nocc[0], nvir[1], nvir[0]), dtype=types[float])
    t2new_baba += einsum("ijab->jiba", x291) * -1
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum("ijab->ijab", x291) * -1
    del x291
    t1new_aa += einsum("ia,ij->ja", t1.aa, x27) * -1
    u11new_aa += einsum("ij,wia->wja", x27, u11.aa) * -1
    del x27
    x28 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x28 += einsum("w,wab->ab", s1, g.aa.bvv)
    x31 += einsum("ab->ab", x28)
    x85 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x85 += einsum("ab->ab", x28)
    x275 += einsum("ab->ab", x28)
    x307 += einsum("ab->ab", x28)
    del x28
    x29 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x29 += einsum("ia,iabc->bc", t1.bb, v.bbaa.ovvv)
    x31 += einsum("ab->ab", x29)
    x88 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x88 -= einsum("ab->ba", x29)
    x275 += einsum("ab->ab", x29)
    x307 += einsum("ab->ab", x29)
    del x29
    x31 += einsum("ab->ab", f.aa.vv)
    t1new_aa += einsum("ia,ba->ib", t1.aa, x31)
    del x31
    x32 = np.zeros((nbos), dtype=types[float])
    x32 += einsum("ia,wia->w", t1.aa, g.aa.bov)
    x34 = np.zeros((nbos), dtype=types[float])
    x34 += einsum("w->w", x32)
    del x32
    x33 = np.zeros((nbos), dtype=types[float])
    x33 += einsum("ia,wia->w", t1.bb, g.bb.bov)
    x34 += einsum("w->w", x33)
    del x33
    x34 += einsum("w->w", G)
    t1new_aa += einsum("w,wia->ia", x34, u11.aa)
    t1new_bb += einsum("w,wia->ia", x34, u11.bb)
    del x34
    x35 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x35 += einsum("ia,jakb->ikjb", t1.bb, v.bbbb.ovov)
    x36 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x36 += einsum("ijka->ijka", x35)
    x36 += einsum("ijka->ikja", x35) * -1
    x151 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x151 -= einsum("ijka->ijka", x35)
    x151 += einsum("ijka->ikja", x35)
    x184 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x184 += einsum("ia,jkla->jilk", t1.bb, x35)
    x185 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x185 += einsum("ijab,klij->lkba", t2.bbbb, x184)
    x189 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x189 += einsum("ijab->ijab", x185)
    del x185
    x225 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x225 += einsum("ia,jkli->jkla", t1.bb, x184)
    del x184
    x226 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x226 += einsum("ia,jkib->jkab", t1.bb, x225)
    del x225
    x230 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x230 += einsum("ijab->ijab", x226)
    del x226
    x282 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x282 += einsum("ijka->jkia", x35) * -1
    x282 += einsum("ijka->kjia", x35)
    del x35
    x36 += einsum("ijka->jika", v.bbbb.ooov) * -1
    x36 += einsum("ijka->jkia", v.bbbb.ooov)
    t1new_bb += einsum("ijab,kija->kb", t2.bbbb, x36) * -1
    del x36
    x37 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x37 += einsum("iabc->ibac", v.bbbb.ovvv)
    x37 += einsum("iabc->ibca", v.bbbb.ovvv) * -1
    x55 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x55 += einsum("ia,ibca->bc", t1.bb, x37)
    x56 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x56 += einsum("ab->ab", x55) * -1
    x278 += einsum("ab->ab", x55) * -1
    x316 += einsum("ab->ab", x55) * -1
    del x55
    x195 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x195 += einsum("ia,jbac->ijbc", t1.bb, x37)
    t1new_bb += einsum("ijab,icba->jc", t2.bbbb, x37) * -1
    del x37
    x38 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x38 += einsum("ia,jbka->jikb", t1.bb, v.aabb.ovov)
    x39 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x39 += einsum("ijka->ikja", x38)
    x203 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x203 += einsum("ijab,ikla->kjlb", t2.abab, x38)
    x208 += einsum("ijka->ikja", x203)
    del x203
    x252 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x252 += einsum("ijka->ikja", x38)
    del x38
    x39 += einsum("iajk->ijka", v.aabb.ovoo)
    x264 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x264 += einsum("ijab,kjla->iklb", t2.abab, x39)
    x271 += einsum("ijka->jika", x264) * -1
    del x264
    x266 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x266 += einsum("ia,jkla->ijkl", t1.aa, x39)
    x267 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x267 += einsum("ijkl->jikl", x266)
    del x266
    t1new_bb += einsum("ijab,ijka->kb", t2.abab, x39) * -1
    x40 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x40 -= einsum("ijab->jiab", t2.bbbb)
    x40 += einsum("ijab->jiba", t2.bbbb)
    x165 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x165 += einsum("wia,ijba->wjb", g.bb.bov, x40)
    x167 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x167 -= einsum("wia->wia", x165)
    del x165
    t1new_bb += einsum("ia,ijba->jb", x15, x40) * -1
    del x15
    x41 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x41 += einsum("iabj->ijba", v.bbbb.ovvo)
    x41 -= einsum("ijab->ijab", v.bbbb.oovv)
    x147 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x147 += einsum("ijab,ikcb->jkac", x40, x41)
    x176 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x176 -= einsum("ijab->ijab", x147)
    del x147
    x157 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x157 += einsum("ia,jkba->ijkb", t1.bb, x41)
    x158 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x158 -= einsum("ijka->jika", x157)
    del x157
    t1new_bb += einsum("ia,ijba->jb", t1.bb, x41)
    u11new_bb = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    u11new_bb += einsum("wia,ijba->wjb", u11.bb, x41)
    del x41
    x42 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x42 += einsum("ia,wja->wji", t1.bb, g.bb.bov)
    x43 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x43 += einsum("wij->wij", x42)
    del x42
    x43 += einsum("wij->wij", g.bb.boo)
    x166 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x166 += einsum("ia,wij->wja", t1.bb, x43)
    x167 -= einsum("wia->wia", x166)
    del x166
    t1new_bb -= einsum("wia,wij->ja", u11.bb, x43)
    del x43
    x44 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x44 += einsum("w,wij->ij", s1, g.bb.boo)
    x52 += einsum("ij->ij", x44)
    x161 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x161 += einsum("ij->ji", x44)
    del x44
    x45 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x45 += einsum("wia,wja->ij", g.bb.bov, u11.bb)
    x52 += einsum("ij->ij", x45)
    x174 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x174 += einsum("ij->ij", x45)
    del x45
    x46 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x46 += einsum("ia,iajk->jk", t1.aa, v.aabb.ovoo)
    x52 += einsum("ij->ij", x46)
    x174 += einsum("ij->ij", x46)
    del x46
    x175 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x175 += einsum("ij,ikab->kjab", x174, t2.bbbb)
    del x174
    x176 -= einsum("ijab->ijba", x175)
    del x175
    x47 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x47 += einsum("ijab,iakb->jk", t2.abab, v.aabb.ovov)
    x52 += einsum("ij->ji", x47)
    x220 += einsum("ij->ij", x47)
    del x47
    x221 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x221 += einsum("ij,jkab->kiab", x220, t2.bbbb)
    del x220
    x222 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x222 += einsum("ijab->jiba", x221) * -1
    del x221
    x49 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x49 += einsum("ijka->ikja", v.bbbb.ooov)
    x49 += einsum("ijka->kija", v.bbbb.ooov) * -1
    x50 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x50 += einsum("ia,ijka->jk", t1.bb, x49)
    x52 += einsum("ij->ij", x50) * -1
    del x50
    x216 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x216 += einsum("ia,jika->jk", t1.bb, x49)
    x217 += einsum("ij->ij", x216) * -1
    del x216
    x218 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x218 += einsum("ij,ikab->kjab", x217, t2.bbbb)
    del x217
    x222 += einsum("ijab->ijba", x218) * -1
    del x218
    x318 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x318 += einsum("wia,ijka->wjk", u11.bb, x49) * -1
    del x49
    x52 += einsum("ij->ij", f.bb.oo)
    x292 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x292 += einsum("ij,kiab->kjab", x52, t2.abab)
    t2new_baba += einsum("ijab->jiba", x292) * -1
    t2new_abab += einsum("ijab->ijab", x292) * -1
    del x292
    t1new_bb += einsum("ia,ij->ja", t1.bb, x52) * -1
    u11new_bb += einsum("ij,wia->wja", x52, u11.bb) * -1
    del x52
    x53 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x53 += einsum("w,wab->ab", s1, g.bb.bvv)
    x56 += einsum("ab->ab", x53)
    x169 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x169 += einsum("ab->ab", x53)
    x278 += einsum("ab->ab", x53)
    x316 += einsum("ab->ab", x53)
    del x53
    x54 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x54 += einsum("ia,iabc->bc", t1.aa, v.aabb.ovvv)
    x56 += einsum("ab->ab", x54)
    x172 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x172 -= einsum("ab->ba", x54)
    x278 += einsum("ab->ab", x54)
    x316 += einsum("ab->ab", x54)
    del x54
    x56 += einsum("ab->ab", f.bb.vv)
    t1new_bb += einsum("ia,ba->ib", t1.bb, x56)
    del x56
    x57 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x57 += einsum("ia,bjca->ijbc", t1.aa, v.aaaa.vovv)
    x92 -= einsum("ijab->ijab", x57)
    del x57
    x58 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x58 += einsum("ijab,jbck->ikac", t2.abab, v.bbaa.ovvo)
    x92 += einsum("ijab->ijab", x58)
    del x58
    x59 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x59 += einsum("ia,jbca->ijcb", t1.aa, v.bbaa.ovvv)
    x60 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x60 += einsum("ijab,kjcb->kiac", t2.abab, x59)
    x92 -= einsum("ijab->ijab", x60)
    del x60
    x241 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x241 += einsum("ijab->ijab", x59)
    del x59
    x61 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x61 += einsum("ijab->jiab", t2.aaaa)
    x61 -= einsum("ijab->jiba", t2.aaaa)
    x62 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x62 += einsum("ijab,ikbc->jkac", x16, x61)
    del x16
    del x61
    x92 -= einsum("ijab->jiba", x62)
    del x62
    x63 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x63 -= einsum("iajb->jiab", v.aaaa.ovov)
    x63 += einsum("iajb->jiba", v.aaaa.ovov)
    x64 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x64 += einsum("ijab,ikcb->jkac", t2.aaaa, x63)
    x65 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x65 -= einsum("ijab,kica->jkbc", t2.aaaa, x64)
    x92 -= einsum("ijab->ijab", x65)
    del x65
    x98 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x98 -= einsum("ijab,kicb->jkac", t2.aaaa, x64)
    del x64
    x102 += einsum("ijab->jiba", x98)
    del x98
    x70 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x70 += einsum("ia,ijba->jb", t1.aa, x63)
    x71 -= einsum("ia->ia", x70)
    x299 -= einsum("ia->ia", x70)
    del x70
    x96 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x96 += einsum("ijab,ikca->jkbc", t2.aaaa, x63)
    x97 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x97 -= einsum("ijab,kica->jkbc", t2.aaaa, x96)
    del x96
    x102 += einsum("ijab->ijab", x97)
    del x97
    x223 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x223 += einsum("ijab,ikca->kjcb", t2.abab, x63)
    del x63
    x224 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x224 -= einsum("ijab,ikac->jkbc", t2.abab, x223)
    del x223
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb -= einsum("ijab->jiab", x224)
    t2new_bbbb += einsum("ijab->ijab", x224)
    del x224
    x66 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x66 += einsum("ijab,kljb->ikla", t2.abab, v.aabb.ooov)
    x74 += einsum("ijka->jika", x66)
    del x66
    x68 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x68 -= einsum("ijab->jiab", t2.aaaa)
    x68 += einsum("ijab->jiba", t2.aaaa)
    x69 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x69 += einsum("ijka,klba->ijlb", x67, x68)
    del x67
    x74 += einsum("ijka->jika", x69)
    del x69
    x81 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x81 += einsum("wia,ijba->wjb", g.aa.bov, x68)
    del x68
    x83 -= einsum("wia->wia", x81)
    del x81
    x71 += einsum("ia->ia", f.aa.ov)
    x72 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x72 += einsum("ia,jkab->jkib", x71, t2.aaaa)
    x74 += einsum("ijka->kjia", x72)
    del x72
    x76 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x76 += einsum("ia,ja->ij", t1.aa, x71)
    del x71
    x77 += einsum("ij->ij", x76)
    del x76
    x74 -= einsum("ijak->ijka", v.aaaa.oovo)
    x75 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x75 += einsum("ia,ijkb->jkab", t1.aa, x74)
    del x74
    x92 += einsum("ijab->ijab", x75)
    del x75
    x77 += einsum("ij->ji", f.aa.oo)
    x78 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x78 += einsum("ij,jkab->kiab", x77, t2.aaaa)
    del x77
    x92 += einsum("ijab->jiba", x78)
    del x78
    x79 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x79 += einsum("ia,wba->wib", t1.aa, g.aa.bvv)
    x83 += einsum("wia->wia", x79)
    del x79
    x80 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x80 += einsum("wia,jiba->wjb", g.bb.bov, t2.abab)
    x83 += einsum("wia->wia", x80)
    del x80
    x83 += einsum("wai->wia", g.aa.bvo)
    x84 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x84 += einsum("wia,wjb->ijab", u11.aa, x83)
    x92 += einsum("ijab->jiba", x84)
    del x84
    x294 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x294 += einsum("wia,wjb->jiba", u11.bb, x83)
    del x83
    t2new_baba += einsum("ijab->jiba", x294)
    t2new_abab += einsum("ijab->ijab", x294)
    del x294
    x85 += einsum("ab->ab", f.aa.vv)
    x86 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x86 += einsum("ab,ijbc->ijca", x85, t2.aaaa)
    del x85
    x92 -= einsum("ijab->jiba", x86)
    del x86
    x87 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x87 += einsum("wia,wib->ab", g.aa.bov, u11.aa)
    x88 += einsum("ab->ab", x87)
    x89 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x89 += einsum("ab,ijac->ijcb", x88, t2.aaaa)
    del x88
    x92 -= einsum("ijab->jiab", x89)
    del x89
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum("ijab->ijab", x92)
    t2new_aaaa -= einsum("ijab->ijba", x92)
    t2new_aaaa -= einsum("ijab->jiab", x92)
    t2new_aaaa += einsum("ijab->jiba", x92)
    del x92
    x275 += einsum("ab->ba", x87) * -1
    x307 += einsum("ab->ba", x87) * -1
    del x87
    x99 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x99 -= einsum("iajb->jiab", v.bbbb.ovov)
    x99 += einsum("iajb->jiba", v.bbbb.ovov)
    x100 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x100 += einsum("ijab,jkcb->ikac", t2.abab, x99)
    x101 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x101 -= einsum("ijab,kjcb->ikac", t2.abab, x100)
    del x100
    x102 += einsum("ijab->ijab", x101)
    del x101
    t2new_aaaa += einsum("ijab->ijab", x102)
    t2new_aaaa -= einsum("ijab->ijba", x102)
    del x102
    x148 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x148 += einsum("ijab,ikcb->jkac", t2.bbbb, x99)
    x149 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x149 -= einsum("ijab,kica->jkbc", t2.bbbb, x148)
    x176 -= einsum("ijab->ijab", x149)
    del x149
    x229 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x229 -= einsum("ijab,kicb->jkac", t2.bbbb, x148)
    del x148
    x230 += einsum("ijab->ijab", x229)
    del x229
    x154 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x154 += einsum("ia,ijba->jb", t1.bb, x99)
    x155 -= einsum("ia->ia", x154)
    x298 -= einsum("ia->ia", x154)
    del x154
    x227 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x227 += einsum("ijab,ikca->jkbc", t2.bbbb, x99)
    del x99
    x228 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x228 -= einsum("ijab,kica->jkbc", t2.bbbb, x227)
    del x227
    x230 += einsum("ijab->jiba", x228)
    del x228
    t2new_bbbb += einsum("ijab->ijab", x230)
    t2new_bbbb -= einsum("ijab->ijba", x230)
    del x230
    x103 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x103 += einsum("ia,jkla->ijlk", t1.aa, v.aaaa.ooov)
    x104 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x104 += einsum("ijab,kijl->klab", t2.aaaa, x103)
    x130 += einsum("ijab->ijab", x104)
    del x104
    x118 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x118 += einsum("ia,jkil->jkla", t1.aa, x103)
    del x103
    x126 += einsum("ijka->ijka", x118)
    del x118
    x109 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x109 += einsum("ijab,kcjb->ikac", t2.abab, v.aabb.ovov)
    x110 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x110 += einsum("ijab->jiab", t2.aaaa) * -1
    x110 += einsum("ijab->jiba", t2.aaaa)
    x111 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x111 += einsum("ijab,jkbc->ikac", x109, x110)
    del x109
    x130 += einsum("ijab->jiba", x111) * -1
    del x111
    x284 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x284 += einsum("ijab,iklb->jkla", x110, x39)
    del x110
    del x39
    x289 += einsum("ijka->ijka", x284) * -1
    del x284
    x112 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x112 += einsum("ijab,icjb->ac", t2.abab, v.aabb.ovov)
    x116 += einsum("ab->ab", x112)
    x275 += einsum("ab->ab", x112) * -1
    x307 += einsum("ab->ab", x112) * -1
    del x112
    x114 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x114 += einsum("iabc->ibac", v.aaaa.ovvv) * -1
    x114 += einsum("iabc->ibca", v.aaaa.ovvv)
    x115 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x115 += einsum("ia,ibca->bc", t1.aa, x114)
    del x114
    x116 += einsum("ab->ab", x115) * -1
    del x115
    x117 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x117 += einsum("ab,ijbc->ijca", x116, t2.aaaa)
    del x116
    x130 += einsum("ijab->jiab", x117)
    del x117
    x119 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x119 += einsum("ijab,kbca->jikc", t2.aaaa, v.aaaa.ovvv)
    x126 += einsum("ijka->ikja", x119)
    del x119
    x120 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x120 += einsum("ia,jbca->ijcb", t1.aa, v.aaaa.ovvv)
    x121 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x121 += einsum("ia,jkba->ijkb", t1.aa, x120)
    del x120
    x126 += einsum("ijka->ikja", x121)
    del x121
    x124 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x124 += einsum("ijka->ikja", v.aaaa.ooov) * -1
    x124 += einsum("ijka->kija", v.aaaa.ooov)
    x125 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x125 += einsum("ijab,ikla->jklb", x10, x124)
    del x124
    x126 += einsum("ijka->ijka", x125)
    del x125
    x127 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x127 += einsum("ia,jikb->jkab", t1.aa, x126)
    del x126
    x130 += einsum("ijab->ijab", x127)
    del x127
    t2new_aaaa += einsum("ijab->ijab", x130) * -1
    t2new_aaaa += einsum("ijab->ijba", x130)
    t2new_aaaa += einsum("ijab->jiab", x130)
    t2new_aaaa += einsum("ijab->jiba", x130) * -1
    del x130
    x131 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x131 += einsum("ia,bcda->ibdc", t1.aa, v.aaaa.vvvv)
    x132 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x132 += einsum("ia,jbca->ijbc", t1.aa, x131)
    del x131
    x139 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x139 += einsum("ijab->ijab", x132)
    del x132
    x133 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x133 += einsum("ijab,kbla->ijlk", t2.aaaa, v.aaaa.ovov)
    x134 += einsum("ijkl->jilk", x133)
    x135 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x135 += einsum("ijab,klji->klab", t2.aaaa, x134)
    del x134
    x139 += einsum("ijab->jiab", x135)
    del x135
    x136 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x136 += einsum("ijkl->lkji", x133)
    del x133
    x136 += einsum("ijkl->kilj", v.aaaa.oooo)
    x137 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x137 += einsum("ia,ijkl->jkla", t1.aa, x136)
    del x136
    x138 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x138 += einsum("ia,ijkb->kjab", t1.aa, x137)
    del x137
    x139 += einsum("ijab->jiba", x138)
    del x138
    t2new_aaaa += einsum("ijab->ijab", x139)
    t2new_aaaa += einsum("ijab->ijba", x139) * -1
    del x139
    x140 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x140 += einsum("ijab,ikjl->lkba", t2.aaaa, v.aaaa.oooo)
    x142 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x142 += einsum("ijab->jiba", x140)
    del x140
    x141 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x141 += einsum("ijab,cadb->ijcd", t2.aaaa, v.aaaa.vvvv)
    x142 += einsum("ijab->jiba", x141)
    del x141
    t2new_aaaa += einsum("ijab->ijba", x142) * -1
    t2new_aaaa += einsum("ijab->ijab", x142)
    del x142
    x143 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x143 += einsum("ia,bjca->ijbc", t1.bb, v.bbbb.vovv)
    x176 -= einsum("ijab->ijab", x143)
    del x143
    x144 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x144 += einsum("ijab,iack->jkbc", t2.abab, v.aabb.ovvo)
    x176 += einsum("ijab->ijab", x144)
    del x144
    x145 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x145 += einsum("ia,jbca->jibc", t1.bb, v.aabb.ovvv)
    x146 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x146 += einsum("ijab,ikac->kjbc", t2.abab, x145)
    x176 -= einsum("ijab->ijab", x146)
    del x146
    x234 += einsum("ijab->ijab", x145)
    x269 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x269 += einsum("ijab->ijab", x145)
    del x145
    x150 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x150 += einsum("ijab,iakl->jklb", t2.abab, v.aabb.ovoo)
    x158 += einsum("ijka->jika", x150)
    del x150
    x152 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x152 += einsum("ijab->jiab", t2.bbbb)
    x152 -= einsum("ijab->jiba", t2.bbbb)
    x153 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x153 += einsum("ijka,klab->ijlb", x151, x152)
    del x152
    del x151
    x158 += einsum("ijka->jika", x153)
    del x153
    x155 += einsum("ia->ia", f.bb.ov)
    x156 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x156 += einsum("ia,jkab->jkib", x155, t2.bbbb)
    x158 += einsum("ijka->kjia", x156)
    del x156
    x160 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x160 += einsum("ia,ja->ij", t1.bb, x155)
    del x155
    x161 += einsum("ij->ij", x160)
    del x160
    x158 -= einsum("ijak->ijka", v.bbbb.oovo)
    x159 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x159 += einsum("ia,ijkb->jkab", t1.bb, x158)
    del x158
    x176 += einsum("ijab->ijab", x159)
    del x159
    x161 += einsum("ij->ji", f.bb.oo)
    x162 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x162 += einsum("ij,jkab->kiab", x161, t2.bbbb)
    del x161
    x176 += einsum("ijab->jiba", x162)
    del x162
    x163 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x163 += einsum("ia,wba->wib", t1.bb, g.bb.bvv)
    x167 += einsum("wia->wia", x163)
    del x163
    x164 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x164 += einsum("wia,ijab->wjb", g.aa.bov, t2.abab)
    x167 += einsum("wia->wia", x164)
    del x164
    x167 += einsum("wai->wia", g.bb.bvo)
    x168 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x168 += einsum("wia,wjb->ijab", u11.bb, x167)
    x176 += einsum("ijab->jiba", x168)
    del x168
    x293 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x293 += einsum("wia,wjb->ijab", u11.aa, x167)
    del x167
    t2new_baba += einsum("ijab->jiba", x293)
    t2new_abab += einsum("ijab->ijab", x293)
    del x293
    x169 += einsum("ab->ab", f.bb.vv)
    x170 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x170 += einsum("ab,ijbc->ijca", x169, t2.bbbb)
    del x169
    x176 -= einsum("ijab->jiba", x170)
    del x170
    x171 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x171 += einsum("wia,wib->ab", g.bb.bov, u11.bb)
    x172 += einsum("ab->ab", x171)
    x173 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x173 += einsum("ab,ijac->ijcb", x172, t2.bbbb)
    del x172
    x176 -= einsum("ijab->jiab", x173)
    del x173
    t2new_bbbb += einsum("ijab->ijab", x176)
    t2new_bbbb -= einsum("ijab->ijba", x176)
    t2new_bbbb -= einsum("ijab->jiab", x176)
    t2new_bbbb += einsum("ijab->jiba", x176)
    del x176
    x278 += einsum("ab->ba", x171) * -1
    x316 += einsum("ab->ba", x171) * -1
    del x171
    x177 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x177 += einsum("ijab,ikjl->lkba", t2.bbbb, v.bbbb.oooo)
    x179 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x179 += einsum("ijab->jiba", x177)
    del x177
    x178 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x178 += einsum("ijab,cbda->ijdc", t2.bbbb, v.bbbb.vvvv)
    x179 += einsum("ijab->jiba", x178)
    del x178
    t2new_bbbb += einsum("ijab->ijba", x179) * -1
    t2new_bbbb += einsum("ijab->ijab", x179)
    del x179
    x180 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x180 += einsum("ia,bcda->ibdc", t1.bb, v.bbbb.vvvv)
    x181 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x181 += einsum("ia,jbca->ijbc", t1.bb, x180)
    del x180
    x189 += einsum("ijab->ijab", x181)
    del x181
    x182 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x182 += einsum("ijab,kbla->ijlk", t2.bbbb, v.bbbb.ovov)
    x183 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x183 += einsum("ijab,klji->lkab", t2.bbbb, x182)
    x189 += einsum("ijab->ijab", x183)
    del x183
    x186 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x186 += einsum("ijkl->lkji", x182)
    del x182
    x186 += einsum("ijkl->kilj", v.bbbb.oooo)
    x187 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x187 += einsum("ia,ijkl->jkla", t1.bb, x186)
    del x186
    x188 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x188 += einsum("ia,ijkb->kjab", t1.bb, x187)
    del x187
    x189 += einsum("ijab->ijab", x188)
    del x188
    t2new_bbbb += einsum("ijab->ijab", x189)
    t2new_bbbb += einsum("ijab->ijba", x189) * -1
    del x189
    x190 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x190 += einsum("ia,jkla->ijlk", t1.bb, v.bbbb.ooov)
    x191 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x191 += einsum("ijab,kjil->klba", t2.bbbb, x190)
    x222 += einsum("ijab->ijab", x191)
    del x191
    x199 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x199 += einsum("ia,jkil->jkla", t1.bb, x190)
    del x190
    x208 += einsum("ijka->ijka", x199)
    del x199
    x192 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x192 += einsum("ijab,iajc->bc", t2.abab, v.aabb.ovov)
    x193 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x193 += einsum("ab,ijcb->ijac", x192, t2.bbbb)
    x222 += einsum("ijab->ijab", x193) * -1
    del x193
    x278 += einsum("ab->ab", x192) * -1
    x316 += einsum("ab->ab", x192) * -1
    del x192
    x194 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x194 += einsum("ijab->jiab", t2.bbbb) * -1
    x194 += einsum("ijab->jiba", t2.bbbb)
    x196 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x196 += einsum("ijab,kicb->jkac", x194, x195) * -1
    del x195
    x222 += einsum("ijab->jiab", x196) * -1
    del x196
    x197 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x197 += einsum("iajb,jkcb->ikac", v.aabb.ovov, x194)
    x198 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x198 += einsum("ijab,ikac->jkbc", t2.abab, x197) * -1
    x222 += einsum("ijab->ijab", x198) * -1
    del x198
    x234 += einsum("ijab->ijab", x197) * -1
    del x197
    x237 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x237 += einsum("ijab,ikcb->jkac", x13, x194)
    del x13
    x239 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x239 += einsum("ijab->ijba", x237)
    del x237
    x263 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x263 += einsum("ijab,klib->klja", x194, x4)
    del x4
    x271 += einsum("ijka->ijka", x263) * -1
    del x263
    x200 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x200 += einsum("ijab,kacb->ijkc", t2.bbbb, v.bbbb.ovvv)
    x208 += einsum("ijka->ikja", x200)
    del x200
    x201 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x201 += einsum("ia,jbca->ijcb", t1.bb, v.bbbb.ovvv)
    x202 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x202 += einsum("ia,jkba->ijkb", t1.bb, x201)
    del x201
    x208 += einsum("ijka->ikja", x202)
    del x202
    x205 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x205 += einsum("ijka->ikja", v.bbbb.ooov) * -1
    x205 += einsum("ijka->kija", v.bbbb.ooov)
    x206 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x206 += einsum("ijab->jiab", t2.bbbb)
    x206 += einsum("ijab->jiba", t2.bbbb) * -1
    x207 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x207 += einsum("ijka,ilab->jklb", x205, x206)
    del x205
    del x206
    x208 += einsum("ijka->kija", x207)
    del x207
    x209 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x209 += einsum("ia,jikb->jkab", t1.bb, x208)
    del x208
    x222 += einsum("ijab->ijab", x209)
    del x209
    x211 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x211 += einsum("iabc->ibac", v.bbbb.ovvv) * -1
    x211 += einsum("iabc->ibca", v.bbbb.ovvv)
    x212 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x212 += einsum("ia,ibca->bc", t1.bb, x211)
    x213 += einsum("ab->ab", x212) * -1
    del x212
    x214 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x214 += einsum("ab,ijbc->ijca", x213, t2.bbbb)
    del x213
    x222 += einsum("ijab->jiab", x214)
    del x214
    t2new_bbbb += einsum("ijab->ijab", x222) * -1
    t2new_bbbb += einsum("ijab->ijba", x222)
    t2new_bbbb += einsum("ijab->jiab", x222)
    t2new_bbbb += einsum("ijab->jiba", x222) * -1
    del x222
    x238 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x238 += einsum("ia,jbca->ijbc", t1.bb, x211)
    del x211
    x239 += einsum("ijab->jiab", x238) * -1
    del x238
    x231 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x231 += einsum("ia,bjca->jibc", t1.bb, v.aabb.vovv)
    t2new_baba += einsum("ijab->jiba", x231)
    t2new_abab += einsum("ijab->ijab", x231)
    del x231
    x232 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x232 += einsum("ijab,cadb->ijcd", t2.abab, v.aabb.vvvv)
    t2new_baba += einsum("ijab->jiba", x232)
    t2new_abab += einsum("ijab->ijab", x232)
    del x232
    x234 += einsum("iabj->ijab", v.aabb.ovvo)
    x235 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x235 += einsum("ijab,ikbc->jkac", x10, x234)
    del x234
    t2new_baba += einsum("ijab->jiba", x235)
    t2new_abab += einsum("ijab->ijab", x235)
    del x235
    x236 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x236 += einsum("ijab,iakc->jkbc", t2.abab, v.aabb.ovov)
    x239 += einsum("ijab->jiab", x236)
    del x236
    x239 += einsum("iabj->ijba", v.bbbb.ovvo)
    x239 += einsum("ijab->ijab", v.bbbb.oovv) * -1
    x240 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x240 += einsum("ijab,jkcb->ikac", t2.abab, x239)
    del x239
    t2new_baba += einsum("ijab->jiba", x240)
    t2new_abab += einsum("ijab->ijab", x240)
    del x240
    x241 += einsum("iabj->jiba", v.bbaa.ovvo)
    t2new_baba += einsum("ijab,jkbc->kica", x241, x40)
    t2new_abab -= einsum("ijab,jkcb->ikac", x241, x40)
    del x40
    del x241
    x242 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x242 -= einsum("iabc->ibac", v.aaaa.ovvv)
    x242 += einsum("iabc->ibca", v.aaaa.ovvv)
    x243 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x243 += einsum("ia,jbca->jibc", t1.aa, x242)
    del x242
    x244 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x244 -= einsum("ijab->ijab", x243)
    del x243
    x244 += einsum("iabj->ijba", v.aaaa.ovvo)
    x244 -= einsum("ijab->ijab", v.aaaa.oovv)
    x245 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x245 += einsum("ijab,ikca->kjcb", t2.abab, x244)
    del x244
    t2new_baba += einsum("ijab->jiba", x245)
    t2new_abab += einsum("ijab->ijab", x245)
    del x245
    x246 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x246 += einsum("ia,jabc->ijbc", t1.aa, v.aabb.ovvv)
    x248 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x248 += einsum("ijab->jiab", x246)
    del x246
    x247 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x247 += einsum("ijab,kajc->ikbc", t2.abab, v.aabb.ovov)
    x248 -= einsum("ijab->jiab", x247)
    del x247
    x248 += einsum("ijab->ijab", v.aabb.oovv)
    x249 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x249 += einsum("ijab,ikcb->kjac", t2.abab, x248)
    del x248
    t2new_baba -= einsum("ijab->jiba", x249)
    t2new_abab -= einsum("ijab->ijab", x249)
    del x249
    x250 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x250 += einsum("ia,jkla->jkil", t1.bb, v.aabb.ooov)
    x254 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x254 += einsum("ijkl->ijlk", x250)
    x267 += einsum("ijkl->ijlk", x250)
    del x250
    x251 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x251 += einsum("ijab,kalb->ikjl", t2.abab, v.aabb.ovov)
    x254 += einsum("ijkl->jilk", x251)
    x267 += einsum("ijkl->jilk", x251)
    del x251
    x252 += einsum("iajk->ijka", v.aabb.ovoo)
    x253 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x253 += einsum("ia,jkla->jikl", t1.aa, x252)
    del x252
    x254 += einsum("ijkl->ijkl", x253)
    del x253
    x254 += einsum("ijkl->ijkl", v.aabb.oooo)
    x255 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x255 += einsum("ijab,ikjl->klab", t2.abab, x254)
    del x254
    t2new_baba += einsum("ijab->jiba", x255)
    t2new_abab += einsum("ijab->ijab", x255)
    del x255
    x256 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x256 += einsum("ia,jabc->ijbc", t1.bb, v.bbaa.ovvv)
    x257 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x257 += einsum("ijab->jiab", x256)
    x287 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x287 += einsum("ijab->jiab", x256)
    del x256
    x257 += einsum("ijab->ijab", v.bbaa.oovv)
    x258 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x258 += einsum("ijab,jkca->ikcb", t2.abab, x257)
    del x257
    t2new_baba -= einsum("ijab->jiba", x258)
    t2new_abab -= einsum("ijab->ijab", x258)
    del x258
    x259 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x259 += einsum("ia,jkba->jkib", t1.bb, v.aabb.oovv)
    x271 += einsum("ijka->ijka", x259)
    del x259
    x260 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x260 += einsum("ijab,kacb->ikjc", t2.abab, v.aabb.ovvv)
    x271 += einsum("ijka->jika", x260)
    del x260
    x261 += einsum("ijka->ikja", v.aaaa.ooov)
    x261 += einsum("ijka->kija", v.aaaa.ooov) * -1
    x262 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x262 += einsum("ijab,ikla->kljb", t2.abab, x261)
    del x261
    x271 += einsum("ijka->ijka", x262) * -1
    del x262
    x267 += einsum("ijkl->ijkl", v.aabb.oooo)
    x268 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x268 += einsum("ia,jkil->jkla", t1.bb, x267)
    del x267
    x271 += einsum("ijka->ijka", x268) * -1
    del x268
    x269 += einsum("iabj->ijab", v.aabb.ovvo)
    x270 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x270 += einsum("ia,jkab->jikb", t1.aa, x269)
    del x269
    x271 += einsum("ijka->ijka", x270)
    del x270
    x271 += einsum("ijak->ijka", v.aabb.oovo)
    x272 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x272 += einsum("ia,ijkb->jkab", t1.aa, x271)
    del x271
    t2new_baba += einsum("ijab->jiba", x272) * -1
    t2new_abab += einsum("ijab->ijab", x272) * -1
    del x272
    x273 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x273 += einsum("iajb->jiab", v.aaaa.ovov)
    x273 += einsum("iajb->jiba", v.aaaa.ovov) * -1
    x274 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x274 += einsum("ijab,ijca->cb", t2.aaaa, x273)
    del x273
    x275 += einsum("ab->ba", x274) * -1
    x307 += einsum("ab->ba", x274) * -1
    del x274
    x275 += einsum("ab->ab", f.aa.vv)
    x276 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x276 += einsum("ab,ijbc->ijac", x275, t2.abab)
    del x275
    t2new_baba += einsum("ijab->jiba", x276)
    t2new_abab += einsum("ijab->ijab", x276)
    del x276
    x278 += einsum("ab->ab", f.bb.vv)
    x279 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x279 += einsum("ab,ijcb->ijca", x278, t2.abab)
    del x278
    t2new_baba += einsum("ijab->jiba", x279)
    t2new_abab += einsum("ijab->ijab", x279)
    del x279
    x280 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x280 += einsum("ia,jabk->kijb", t1.bb, v.bbaa.ovvo)
    x289 += einsum("ijka->ikja", x280)
    del x280
    x281 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x281 += einsum("ijab,kbca->ijkc", t2.abab, v.bbaa.ovvv)
    x289 += einsum("ijka->ikja", x281)
    del x281
    x282 += einsum("ijka->ikja", v.bbbb.ooov)
    x282 += einsum("ijka->kija", v.bbbb.ooov) * -1
    x283 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x283 += einsum("ijab,jklb->ikla", t2.abab, x282)
    del x282
    x289 += einsum("ijka->ijka", x283) * -1
    del x283
    x287 += einsum("ijab->ijab", v.bbaa.oovv)
    x288 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x288 += einsum("ia,jkba->ijkb", t1.aa, x287)
    del x287
    x289 += einsum("ijka->ijka", x288)
    del x288
    x289 += einsum("ijak->kija", v.bbaa.oovo)
    x290 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x290 += einsum("ia,jikb->jkba", t1.bb, x289)
    del x289
    t2new_baba += einsum("ijab->jiba", x290) * -1
    t2new_abab += einsum("ijab->ijab", x290) * -1
    del x290
    x295 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x295 += einsum("ia,bcda->ibcd", t1.bb, v.aabb.vvvv)
    x296 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x296 += einsum("iabc->iabc", x295)
    del x295
    x296 += einsum("abci->iabc", v.aabb.vvvo)
    x297 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x297 += einsum("ia,jbac->ijbc", t1.aa, x296)
    del x296
    t2new_baba += einsum("ijab->jiba", x297)
    t2new_abab += einsum("ijab->ijab", x297)
    del x297
    x298 += einsum("ia->ia", f.bb.ov)
    s1new = np.zeros((nbos), dtype=types[float])
    s1new += einsum("ia,wia->w", x298, u11.bb)
    del x298
    x299 += einsum("ia->ia", f.aa.ov)
    s1new += einsum("ia,wia->w", x299, u11.aa)
    del x299
    x300 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x300 += einsum("wia,jbia->wjb", u11.bb, v.aabb.ovov)
    x302 += einsum("wia->wia", x300)
    del x300
    x302 += einsum("wia->wia", gc.aa.bov)
    x309 += einsum("ia,wja->wji", t1.aa, x302)
    u11new_aa += einsum("wia,ijba->wjb", x302, x10)
    del x10
    u11new_bb += einsum("wia,ijab->wjb", x302, t2.abab)
    del x302
    x303 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x303 += einsum("wia,iajb->wjb", u11.aa, v.aabb.ovov)
    x305 += einsum("wia->wia", x303)
    del x303
    x305 += einsum("wia->wia", gc.bb.bov)
    x318 += einsum("ia,wja->wji", t1.bb, x305)
    u11new_aa += einsum("wia,jiba->wjb", x305, t2.abab)
    u11new_bb += einsum("wia,ijab->wjb", x305, x194)
    del x305
    del x194
    x307 += einsum("ab->ab", f.aa.vv)
    u11new_aa += einsum("ab,wib->wia", x307, u11.aa)
    del x307
    x308 += einsum("ia->ia", f.aa.ov)
    x309 += einsum("ia,wja->wij", x308, u11.aa)
    del x308
    x309 += einsum("wij->wij", gc.aa.boo)
    x309 += einsum("wia,jkia->wjk", u11.bb, v.aabb.ooov)
    u11new_aa += einsum("ia,wij->wja", t1.aa, x309) * -1
    del x309
    x310 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x310 += einsum("iabc->ibac", v.aaaa.ovvv)
    x310 -= einsum("iabc->ibca", v.aaaa.ovvv)
    x311 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x311 -= einsum("wia,ibca->wbc", u11.aa, x310)
    del x310
    x311 += einsum("wab->wab", gc.aa.bvv)
    x311 += einsum("wia,iabc->wbc", u11.bb, v.bbaa.ovvv)
    u11new_aa += einsum("ia,wba->wib", t1.aa, x311)
    del x311
    x312 = np.zeros((nbos, nbos), dtype=types[float])
    x312 += einsum("wia,xia->wx", g.aa.bov, u11.aa)
    x314 = np.zeros((nbos, nbos), dtype=types[float])
    x314 += einsum("wx->wx", x312)
    del x312
    x313 = np.zeros((nbos, nbos), dtype=types[float])
    x313 += einsum("wia,xia->wx", g.bb.bov, u11.bb)
    x314 += einsum("wx->wx", x313)
    del x313
    x314 += einsum("wx->wx", w)
    u11new_aa += einsum("wx,wia->xia", x314, u11.aa)
    u11new_bb += einsum("wx,wia->xia", x314, u11.bb)
    del x314
    x316 += einsum("ab->ab", f.bb.vv)
    u11new_bb += einsum("ab,wib->wia", x316, u11.bb)
    del x316
    x317 += einsum("ia->ia", f.bb.ov)
    x318 += einsum("ia,wja->wij", x317, u11.bb)
    del x317
    x318 += einsum("wij->wij", gc.bb.boo)
    x318 += einsum("wia,iajk->wjk", u11.aa, v.aabb.ovoo)
    u11new_bb += einsum("ia,wij->wja", t1.bb, x318) * -1
    del x318
    x319 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x319 += einsum("iabc->ibac", v.bbbb.ovvv)
    x319 -= einsum("iabc->ibca", v.bbbb.ovvv)
    x320 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x320 -= einsum("wia,ibca->wbc", u11.bb, x319)
    del x319
    x320 += einsum("wab->wab", gc.bb.bvv)
    x320 += einsum("wia,iabc->wbc", u11.aa, v.aabb.ovvv)
    u11new_bb += einsum("ia,wba->wib", t1.bb, x320)
    del x320
    t1new_aa += einsum("ai->ia", f.aa.vo)
    t1new_aa += einsum("ijab,jbca->ic", t2.abab, v.bbaa.ovvv)
    t1new_aa += einsum("ia,iabj->jb", t1.bb, v.bbaa.ovvo)
    t1new_aa += einsum("w,wai->ia", s1, g.aa.bvo)
    t1new_aa += einsum("wab,wib->ia", g.aa.bvv, u11.aa)
    t1new_bb += einsum("wab,wib->ia", g.bb.bvv, u11.bb)
    t1new_bb += einsum("ia,iabj->jb", t1.aa, v.aabb.ovvo)
    t1new_bb += einsum("ijab,iacb->jc", t2.abab, v.aabb.ovvv)
    t1new_bb += einsum("ai->ia", f.bb.vo)
    t1new_bb += einsum("w,wai->ia", s1, g.bb.bvo)
    t2new_aaaa -= einsum("aibj->jiab", v.aaaa.vovo)
    t2new_aaaa += einsum("aibj->jiba", v.aaaa.vovo)
    t2new_bbbb -= einsum("aibj->jiab", v.bbbb.vovo)
    t2new_bbbb += einsum("aibj->jiba", v.bbbb.vovo)
    t2new_baba += einsum("aibj->jiba", v.aabb.vovo)
    t2new_abab += einsum("aibj->ijab", v.aabb.vovo)
    s1new += einsum("ia,wia->w", t1.aa, gc.aa.bov)
    s1new += einsum("w->w", G)
    s1new += einsum("w,wx->x", s1, w)
    s1new += einsum("ia,wia->w", t1.bb, gc.bb.bov)
    u11new_aa += einsum("wai->wia", gc.aa.bvo)
    u11new_aa += einsum("wia,iabj->wjb", u11.bb, v.bbaa.ovvo)
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

    return {"t1new": t1new, "t2new": t2new, "s1new": s1new, "u11new": u11new}

def update_lams(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, l1=None, l2=None, ls1=None, lu11=None, **kwargs):
    l1new = Namespace()
    l2new = Namespace()
    lu11new = Namespace()

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

    # L1, L2, LS1 and LU11 amplitudes
    x0 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x0 += einsum("ia,bajk->jkib", t1.bb, l2.abab)
    x2 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x2 += einsum("ia,jkla->jikl", t1.aa, x0)
    x67 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x67 += einsum("ijkl->ijkl", x2)
    x461 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x461 += einsum("ijkl->ijkl", x2)
    l1new_aa = np.zeros((nvir[0], nocc[0]), dtype=types[float])
    l1new_aa += einsum("iajk,likj->al", v.aabb.ovoo, x2)
    l1new_bb = np.zeros((nvir[1], nocc[1]), dtype=types[float])
    l1new_bb += einsum("ijka,jilk->al", v.aabb.ooov, x2)
    del x2
    x52 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x52 += einsum("ia,jikb->jkba", t1.bb, x0)
    x68 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x68 += einsum("ijab,kjla->kilb", t2.abab, x0)
    x176 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x176 += einsum("ijab,ijka->kb", t2.abab, x0)
    x186 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x186 += einsum("ia->ia", x176)
    del x176
    x252 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x252 += einsum("ijab,ikla->kjlb", t2.abab, x0)
    x295 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x295 += einsum("iajk,lkjb->liba", v.aabb.ovoo, x0)
    x333 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x333 -= einsum("ijab->ijab", x295)
    del x295
    x430 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x430 += einsum("iabc,jkib->jkca", v.bbaa.ovvv, x0)
    l2new_baba = np.zeros((nvir[1], nvir[0], nocc[1], nocc[0]), dtype=types[float])
    l2new_baba -= einsum("ijab->baji", x430)
    l2new_abab = np.zeros((nvir[0], nvir[1], nocc[0], nocc[1]), dtype=types[float])
    l2new_abab -= einsum("ijab->abij", x430)
    del x430
    l1new_aa -= einsum("ijab,kjia->bk", v.bbaa.oovv, x0)
    x1 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x1 += einsum("abij,klab->ikjl", l2.abab, t2.abab)
    x67 += einsum("ijkl->ijkl", x1)
    x68 += einsum("ia,jkil->jkla", t1.bb, x67)
    x258 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x258 += einsum("ia,ijkl->jkla", t1.aa, x67)
    del x67
    x461 += einsum("ijkl->ijkl", x1)
    x462 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x462 += einsum("iajb,kilj->klab", v.aabb.ovov, x461)
    del x461
    l2new_baba += einsum("ijab->baji", x462)
    l2new_abab += einsum("ijab->abij", x462)
    del x462
    l1new_aa += einsum("iajk,likj->al", v.aabb.ovoo, x1)
    l1new_bb += einsum("ijka,jilk->al", v.aabb.ooov, x1)
    del x1
    x3 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x3 += einsum("ai,ja->ij", l1.aa, t1.aa)
    x163 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x163 += einsum("ij->ij", x3)
    x208 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x208 += einsum("ij->ij", x3)
    x326 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x326 += einsum("ij->ij", x3)
    x4 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x4 += einsum("w,wia->ia", s1, g.aa.bov)
    x6 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum("ia->ia", x4)
    x31 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x31 += einsum("ia->ia", x4)
    x314 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x314 += einsum("ia->ia", x4)
    x495 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x495 += einsum("ia->ia", x4)
    l1new_aa -= einsum("ij,ja->ai", x3, x4)
    del x3
    del x4
    x5 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x5 += einsum("abij,kjac->ikbc", l2.abab, t2.abab)
    l1new_aa += einsum("iabc,jibc->aj", v.aabb.ovvv, x5) * -1
    del x5
    x6 += einsum("ia->ia", f.aa.ov)
    x7 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x7 += einsum("ia,jkab->jkib", x6, t2.aaaa)
    del x6
    x8 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum("ijka->kjia", x7)
    del x7
    x8 += einsum("ijak->ijka", v.aaaa.oovo) * -1
    x26 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x26 += einsum("ijka->kija", x8)
    x26 += einsum("ijka->jika", x8) * -1
    del x8
    x9 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x9 += einsum("ijab,kacb->ijkc", t2.aaaa, v.aaaa.ovvv)
    x17 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x17 += einsum("ijka->ijka", x9) * -1
    del x9
    x10 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x10 += einsum("ijab,kbla->ijlk", t2.aaaa, v.aaaa.ovov)
    x11 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x11 += einsum("ia,jkil->kjla", t1.aa, x10)
    x17 += einsum("ijka->ijka", x11)
    del x11
    x356 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x356 += einsum("abij,ijkl->klab", l2.aaaa, x10)
    del x10
    x360 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x360 += einsum("ijab->ijab", x356)
    del x356
    x12 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x12 += einsum("ia,jbia->jb", t1.bb, v.aabb.ovov)
    x15 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x15 += einsum("ia->ia", x12)
    x31 += einsum("ia->ia", x12)
    x209 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x209 += einsum("ia->ia", x12)
    x332 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x332 += einsum("ia->ia", x12)
    x495 += einsum("ia->ia", x12)
    del x12
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x13 += einsum("iajb->jiab", v.aaaa.ovov) * -1
    x13 += einsum("iajb->jiba", v.aaaa.ovov)
    x14 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x14 += einsum("ia,ijba->jb", t1.aa, x13)
    x15 += einsum("ia->ia", x14) * -1
    x16 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x16 += einsum("ia,jkab->jkib", x15, t2.aaaa)
    x17 += einsum("ijka->jika", x16)
    del x16
    x26 += einsum("ijka->jkia", x17)
    x26 += einsum("ijka->ikja", x17) * -1
    del x17
    x344 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x344 += einsum("ia,ja->ij", t1.aa, x15)
    x345 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x345 += einsum("ij->ij", x344)
    del x344
    x517 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x517 += einsum("ia,ib->ab", t1.aa, x15) * -1
    x31 += einsum("ia->ia", x14) * -1
    x209 += einsum("ia->ia", x14) * -1
    del x14
    x194 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x194 += einsum("wia,ijba->wjb", u11.aa, x13)
    x195 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x195 += einsum("wia->wia", x194) * -1
    del x194
    x435 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x435 += einsum("ijab,ikca->kjcb", t2.abab, x13)
    x437 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x437 += einsum("ijab->ijab", x435) * -1
    del x435
    x18 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x18 += einsum("ia,jbka->ijkb", t1.aa, v.aaaa.ovov)
    x19 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x19 += einsum("ijka->ikja", x18) * -1
    x19 += einsum("ijka->ijka", x18)
    x27 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x27 += einsum("ijka->jkia", x18)
    x27 += einsum("ijka->kjia", x18) * -1
    x46 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x46 += einsum("ia,jkla->ijkl", t1.aa, x18)
    x47 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x47 += einsum("ia,jkil->kjla", t1.aa, x46)
    x48 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x48 += einsum("ijka->ijka", x47)
    del x47
    x357 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x357 += einsum("abij,jikl->klba", l2.aaaa, x46)
    del x46
    x360 += einsum("ijab->ijab", x357)
    del x357
    x83 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x83 += einsum("ijka->jkia", x18) * -1
    x83 += einsum("ijka->kjia", x18)
    x329 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x329 += einsum("ijka->ijka", x18)
    x341 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x341 += einsum("ijka->ikja", x18) * -1
    x341 += einsum("ijka->ijka", x18)
    x466 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x466 += einsum("ijka->ikja", x18)
    x466 += einsum("ijka->ijka", x18) * -1
    del x18
    x19 += einsum("ijka->jkia", v.aaaa.ooov)
    x19 += einsum("ijka->jika", v.aaaa.ooov) * -1
    x20 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x20 += einsum("ijab->jiab", t2.aaaa)
    x20 += einsum("ijab->jiba", t2.aaaa) * -1
    x26 += einsum("ijka,klab->ijlb", x19, x20)
    x50 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x50 += einsum("ijka,klab->ijlb", x19, x20) * 0.9999999999999993
    del x19
    x82 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x82 += einsum("iabc,ijac->jb", v.aaaa.ovvv, x20)
    x116 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x116 += einsum("ia->ia", x82) * -1
    del x82
    x161 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x161 += einsum("ai,ijba->jb", l1.aa, x20)
    x165 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x165 += einsum("ia->ia", x161) * -1
    del x161
    x169 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x169 += einsum("abij,ijac->bc", l2.aaaa, x20)
    x170 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x170 += einsum("ab->ab", x169) * -1
    x347 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x347 += einsum("ab->ab", x169) * -1
    x476 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x476 += einsum("ab->ab", x169) * -1
    del x169
    x249 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x249 += einsum("abij,ikca->kjcb", l2.abab, x20) * -1
    x258 += einsum("ijka,ilba->ljkb", x0, x20) * -1
    x338 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x338 += einsum("ijab,ikbc->kjca", x13, x20)
    x339 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x339 += einsum("ijab->ijab", x338)
    del x338
    x350 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x350 += einsum("iajb,ijac->cb", v.aaaa.ovov, x20)
    x351 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x351 += einsum("ab->ab", x350) * -1
    del x350
    x468 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x468 += einsum("iajb,ijca->cb", v.aaaa.ovov, x20)
    x469 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x469 += einsum("ab->ab", x468) * -1
    x517 += einsum("ab->ab", x468) * -1
    del x468
    x21 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x21 += einsum("ia,jakb->ijkb", t1.aa, v.aabb.ovov)
    x22 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x22 += einsum("ijka->jika", x21)
    x492 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x492 += einsum("ijka->jika", x21)
    x22 += einsum("ijka->ijka", v.aabb.ooov)
    x26 += einsum("ijab,kljb->lkia", t2.abab, x22)
    x50 += einsum("ijab,kljb->lkia", t2.abab, x22) * 0.9999999999999993
    x85 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x85 += einsum("ijab,ikjb->ka", t2.abab, x22)
    x116 += einsum("ia->ia", x85) * -1
    del x85
    x228 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x228 += einsum("ijab,iklb->klja", t2.abab, x22) * -1
    x465 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x465 += einsum("ijka,likb->ljab", x0, x22)
    l2new_baba += einsum("ijab->baji", x465)
    l2new_abab += einsum("ijab->abij", x465)
    del x465
    x23 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x23 += einsum("iabj->ijba", v.aaaa.ovvo)
    x23 += einsum("ijab->ijab", v.aaaa.oovv) * -1
    x26 += einsum("ia,jkba->ijkb", t1.aa, x23)
    x50 += einsum("ia,jkba->ijkb", t1.aa, x23) * 0.9999999999999993
    x93 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x93 += einsum("ia,ijba->jb", t1.aa, x23)
    x116 += einsum("ia->ia", x93)
    del x93
    x24 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x24 += einsum("ia,jkla->ijlk", t1.aa, v.aaaa.ooov)
    x25 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x25 += einsum("ijkl->jkil", x24)
    x25 += einsum("ijkl->kjil", x24) * -1
    x49 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x49 += einsum("ijkl->ikjl", x24)
    x49 += einsum("ijkl->ijkl", x24) * -1
    x50 += einsum("ia,jkil->jkla", t1.aa, x49) * -0.9999999999999993
    del x49
    x293 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x293 += einsum("abij,jkli->klba", l2.aaaa, x24)
    del x24
    x333 -= einsum("ijab->ijab", x293)
    del x293
    x25 += einsum("ijkl->kijl", v.aaaa.oooo)
    x25 += einsum("ijkl->kilj", v.aaaa.oooo) * -1
    x26 += einsum("ia,ijkl->kjla", t1.aa, x25) * -1
    del x25
    l1new_aa += einsum("abij,ikjb->ak", l2.aaaa, x26) * -1
    del x26
    x27 += einsum("ijka->ikja", v.aaaa.ooov) * -1
    x27 += einsum("ijka->kija", v.aaaa.ooov)
    x39 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x39 += einsum("ijab,kila->kljb", t2.abab, x27) * -1
    del x27
    x28 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x28 += einsum("ijab->jiab", t2.bbbb)
    x28 += einsum("ijab->jiba", t2.bbbb) * -1
    x39 += einsum("ijka,klab->ijlb", x22, x28) * -1
    x124 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x124 += einsum("iabc,ijac->jb", v.bbbb.ovvv, x28)
    x154 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x154 += einsum("ia->ia", x124) * -1
    del x124
    x183 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x183 += einsum("abij,ikab->jk", l2.bbbb, x28)
    x184 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x184 += einsum("ij->ij", x183) * -1
    x251 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x251 += einsum("ij->ij", x183) * -1
    x285 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x285 += einsum("ij->ij", x183) * -1
    del x183
    x436 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x436 += einsum("iajb,jkbc->ikac", v.aabb.ovov, x28)
    x437 += einsum("ijab->ijab", x436) * -1
    del x436
    x509 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x509 += einsum("wia,ijab->wjb", g.bb.bov, x28)
    x511 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x511 += einsum("wia->wia", x509) * -1
    del x509
    x514 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x514 += einsum("wai,ijab->wjb", lu11.bb, x28)
    x515 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x515 += einsum("wia->wia", x514) * -1
    del x514
    x29 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x29 += einsum("ia,jbka->jikb", t1.bb, v.aabb.ovov)
    x30 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x30 += einsum("ijka->ikja", x29)
    x34 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x34 += einsum("ijka->ikja", x29)
    x296 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x296 += einsum("ijka,ljkb->ilab", x0, x29)
    x333 -= einsum("ijab->ijab", x296)
    del x296
    x490 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x490 += einsum("ijka->ikja", x29)
    del x29
    x30 += einsum("iajk->ijka", v.aabb.ovoo)
    x39 += einsum("ijab,kjla->kilb", t2.abab, x30) * -1
    x128 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x128 += einsum("ijab,ijka->kb", t2.abab, x30)
    x154 += einsum("ia->ia", x128) * -1
    del x128
    x223 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x223 += einsum("ijab,ikla->lkjb", t2.abab, x30)
    x458 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x458 += einsum("ia,jkla->ijkl", t1.aa, x30)
    x459 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x459 += einsum("ijkl->jikl", x458)
    del x458
    x31 += einsum("ia->ia", f.aa.ov)
    x39 += einsum("ia,jkab->ijkb", x31, t2.abab)
    x104 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x104 += einsum("ia,ja->ij", t1.aa, x31)
    x105 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x105 += einsum("ij->ji", x104)
    del x104
    x130 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x130 += einsum("ia,ijab->jb", x31, t2.abab)
    x154 += einsum("ia->ia", x130)
    del x130
    x196 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x196 += einsum("ia,wja->wij", x31, u11.aa)
    x213 = np.zeros((nbos), dtype=types[float])
    x213 += einsum("ia,wia->w", x31, u11.aa)
    x215 = np.zeros((nbos), dtype=types[float])
    x215 += einsum("w->w", x213)
    del x213
    lu11new_aa = np.zeros((nbos, nvir[0], nocc[0]), dtype=types[float])
    lu11new_aa += einsum("w,ia->wai", ls1, x31) * 2
    x32 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x32 += einsum("ia,jkla->jkil", t1.bb, v.aabb.ooov)
    x36 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x36 += einsum("ijkl->ijlk", x32)
    x459 += einsum("ijkl->ijlk", x32)
    del x32
    x33 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x33 += einsum("ijab,kalb->ikjl", t2.abab, v.aabb.ovov)
    x36 += einsum("ijkl->jilk", x33)
    x459 += einsum("ijkl->jilk", x33)
    del x33
    x34 += einsum("iajk->ijka", v.aabb.ovoo)
    x35 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x35 += einsum("ia,jkla->ijkl", t1.aa, x34)
    del x34
    x36 += einsum("ijkl->jikl", x35)
    del x35
    x36 += einsum("ijkl->ijkl", v.aabb.oooo)
    x39 += einsum("ia,jkil->jkla", t1.bb, x36) * -1
    x228 += einsum("ia,ijkl->jkla", t1.aa, x36) * -1
    del x36
    x37 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x37 += einsum("ia,jbca->jibc", t1.bb, v.aabb.ovvv)
    x38 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x38 += einsum("ijab->ijab", x37)
    x304 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x304 += einsum("ijab->ijab", x37)
    x437 += einsum("ijab->ijab", x37)
    del x37
    x38 += einsum("iabj->ijab", v.aabb.ovvo)
    x39 += einsum("ia,jkab->jikb", t1.aa, x38)
    x39 += einsum("ijak->ijka", v.aabb.oovo)
    x39 += einsum("ia,jkba->jkib", t1.bb, v.aabb.oovv)
    x39 += einsum("ijab,kacb->kijc", t2.abab, v.aabb.ovvv)
    l1new_aa += einsum("abij,kijb->ak", l2.abab, x39) * -1
    del x39
    x40 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x40 += einsum("abij,kjcb->ikac", l2.abab, t2.abab)
    x42 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x42 += einsum("ijab->ijab", x40)
    x343 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x343 += einsum("ijab,kicb->kjca", x13, x40)
    del x40
    x355 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x355 += einsum("ijab->ijab", x343)
    del x343
    x41 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x41 += einsum("abij->jiab", l2.aaaa)
    x41 += einsum("abij->jiba", l2.aaaa) * -1
    x42 += einsum("ijab,ikcb->kjca", x20, x41)
    x52 += einsum("ijab,ikca->kjcb", t2.abab, x41) * -1
    x43 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x43 += einsum("iabc->ibac", v.aaaa.ovvv)
    x43 += einsum("iabc->ibca", v.aaaa.ovvv) * -1
    x109 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x109 += einsum("ia,ibca->bc", t1.aa, x43)
    x110 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x110 += einsum("ab->ab", x109) * -1
    x469 += einsum("ab->ab", x109) * -1
    x517 += einsum("ab->ab", x109) * -1
    del x109
    x448 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x448 += einsum("ia,jbac->ijbc", t1.aa, x43)
    x449 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x449 += einsum("ijab->jiab", x448) * -1
    del x448
    l1new_aa += einsum("ijab,jacb->ci", x42, x43) * -1
    del x42
    x44 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x44 += einsum("ia,jbca->ijcb", t1.aa, v.aaaa.ovvv)
    x45 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x45 += einsum("ia,jkba->ijkb", t1.aa, x44)
    del x44
    x48 += einsum("ijka->ijka", x45) * -0.9999999999999993
    del x45
    x50 += einsum("ijka->jkia", x48)
    x50 += einsum("ijka->ikja", x48) * -1
    del x48
    l1new_aa += einsum("abij,ikja->bk", l2.aaaa, x50)
    del x50
    x51 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x51 += einsum("ijab->jiab", t2.bbbb) * -1
    x51 += einsum("ijab->jiba", t2.bbbb)
    x52 += einsum("abij,jkbc->ikac", l2.abab, x51) * -1
    l1new_aa += einsum("iabc,jiba->cj", v.bbaa.ovvv, x52) * -1
    del x52
    x179 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x179 += einsum("ai,ijab->jb", l1.bb, x51)
    x186 += einsum("ia->ia", x179) * -1
    del x179
    x190 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x190 += einsum("abij,ijbc->ac", l2.bbbb, x51)
    x191 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x191 += einsum("ab->ab", x190) * -1
    x422 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x422 += einsum("ab->ab", x190) * -1
    x478 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x478 += einsum("ab->ab", x190) * -1
    del x190
    x53 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x53 += einsum("ia,abjk->jikb", t1.aa, l2.abab)
    x60 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x60 += einsum("ijab,kljb->kila", t2.abab, x53)
    x68 += einsum("ijab,klia->kljb", x51, x53) * -1
    x159 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x159 += einsum("ijab,ikjb->ka", t2.abab, x53)
    x165 += einsum("ia->ia", x159)
    del x159
    x249 += einsum("ia,ijkb->jkab", t1.aa, x53)
    x258 += einsum("ijab,iklb->klja", t2.abab, x53)
    x364 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x364 += einsum("ijka,jilb->lkba", v.aabb.ooov, x53)
    x403 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x403 -= einsum("ijab->ijab", x364)
    del x364
    x367 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x367 += einsum("ijka,ijlb->lkba", x21, x53)
    del x21
    x403 -= einsum("ijab->ijab", x367)
    del x367
    x433 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x433 += einsum("iabc,jikb->jkac", v.aabb.ovvv, x53)
    l2new_baba -= einsum("ijab->baji", x433)
    l2new_abab -= einsum("ijab->abij", x433)
    del x433
    x464 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x464 += einsum("ijka,likb->ljab", x30, x53)
    l2new_baba += einsum("ijab->baji", x464)
    l2new_abab += einsum("ijab->abij", x464)
    del x464
    x483 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x483 += einsum("ia,jikb->jkab", x31, x53)
    l2new_baba += einsum("ijab->baji", x483) * -1
    l2new_abab += einsum("ijab->abij", x483) * -1
    del x483
    l1new_aa += einsum("ijab,kijb->ak", x38, x53) * -1
    del x38
    l1new_bb -= einsum("ijab,jika->bk", v.aabb.oovv, x53)
    l2new_baba += einsum("ijka,ijlb->balk", x466, x53) * -1
    l2new_abab += einsum("ijka,iklb->abjl", x466, x53)
    del x466
    x54 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x54 += einsum("ia,abjk->kjib", t1.aa, l2.aaaa)
    x55 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x55 += einsum("ijka->ijka", x54) * -1
    x55 += einsum("ijka->jika", x54)
    x68 += einsum("ijab,kila->kljb", t2.abab, x55) * -1
    l1new_aa += einsum("ijab,kjia->bk", x23, x55) * -1
    del x23
    x62 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x62 += einsum("ijka->ijka", x54)
    x62 += einsum("ijka->jika", x54) * -1
    x63 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x63 += einsum("ia,jikb->jkba", t1.aa, x62) * -1
    l1new_aa += einsum("iabc,jiab->cj", x43, x63)
    del x43
    del x63
    x342 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x342 += einsum("ijka,likb->ljba", x341, x62)
    del x341
    x355 += einsum("ijab->ijab", x342)
    del x342
    l2new_abab += einsum("ijka,ljib->balk", x22, x62) * -1
    del x62
    x71 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x71 += einsum("ia,jkla->kjli", t1.aa, x54)
    x72 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x72 += einsum("ijkl->ijkl", x71)
    x117 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x117 += einsum("ijkl->ijkl", x71) * -1
    x117 += einsum("ijkl->ijlk", x71)
    l1new_aa += einsum("ijka,ljik->al", v.aaaa.ooov, x117)
    del x117
    x358 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x358 += einsum("ijkl->ijkl", x71)
    del x71
    x160 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x160 += einsum("ijab,jikb->ka", x20, x54)
    del x20
    x165 += einsum("ia->ia", x160) * -1
    del x160
    x294 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x294 += einsum("iabc,jkib->kjac", v.aaaa.ovvv, x54)
    x333 -= einsum("ijab->ijab", x294)
    del x294
    x306 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x306 += einsum("ijka->ijka", x54)
    x306 -= einsum("ijka->jika", x54)
    l2new_baba += einsum("ijka,ljib->abkl", x22, x306) * -1
    del x22
    x354 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x354 += einsum("ia,jkib->jkba", x15, x54)
    del x15
    x355 += einsum("ijab->ijab", x354)
    del x354
    x56 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x56 += einsum("ijab->jiab", t2.aaaa) * -1
    x56 += einsum("ijab->jiba", t2.aaaa)
    x58 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x58 += einsum("abij,ikba->jk", l2.aaaa, x56)
    x59 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x59 += einsum("ij->ij", x58) * -1
    x163 += einsum("ij->ij", x58) * -1
    x205 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x205 += einsum("ij->ij", x58) * -1
    del x58
    x60 += einsum("ijka,ilba->jlkb", x55, x56)
    del x55
    x86 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x86 += einsum("ia,ijba->jb", x31, x56)
    del x31
    x116 += einsum("ia->ia", x86) * -1
    del x86
    x101 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x101 += einsum("iajb,ikba->kj", v.aaaa.ovov, x56)
    x105 += einsum("ij->ji", x101) * -1
    x345 += einsum("ij->ij", x101) * -1
    del x101
    x228 += einsum("ijka,ilba->ljkb", x30, x56) * -1
    x439 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x439 += einsum("iajb,ikca->kjcb", v.aabb.ovov, x56)
    x441 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x441 += einsum("ijab->ijab", x439) * -1
    del x439
    x500 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x500 += einsum("wia,ijba->wjb", g.aa.bov, x56)
    x502 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x502 += einsum("wia->wia", x500) * -1
    del x500
    x505 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x505 += einsum("wai,ijba->wjb", lu11.aa, x56)
    x506 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x506 += einsum("wia->wia", x505) * -1
    del x505
    x57 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x57 += einsum("abij,kjab->ik", l2.abab, t2.abab)
    x59 += einsum("ij->ij", x57)
    x60 += einsum("ia,jk->jika", t1.aa, x59)
    x68 += einsum("ia,jk->jkia", t1.bb, x59) * -1
    x353 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x353 += einsum("ij,jakb->ikab", x59, v.aaaa.ovov)
    x355 += einsum("ijab->ijba", x353) * -1
    del x353
    l1new_aa += einsum("ia,ji->aj", f.aa.ov, x59) * -1
    del x59
    x163 += einsum("ij->ij", x57)
    x205 += einsum("ij->ij", x57)
    del x57
    x61 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x61 += einsum("iajb->jiab", v.aaaa.ovov)
    x61 += einsum("iajb->jiba", v.aaaa.ovov) * -1
    x447 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x447 += einsum("ijab,ikbc->jkac", x56, x61)
    del x56
    x449 += einsum("ijab->jiab", x447)
    del x447
    l1new_aa += einsum("ijka,kjab->bi", x60, x61) * -1
    del x60
    x64 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x64 += einsum("ia,bcda->ibdc", t1.aa, v.aaaa.vvvv)
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
    x68 += einsum("ai,jkab->ijkb", l1.aa, t2.abab) * -1
    l1new_aa += einsum("iajb,kijb->ak", v.aabb.ovov, x68)
    del x68
    x69 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x69 += einsum("ai,jkab->ikjb", l1.aa, t2.aaaa)
    x74 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x74 += einsum("ijka->ijka", x69)
    del x69
    x70 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x70 += einsum("abij,klab->ijkl", l2.aaaa, t2.aaaa)
    x72 += einsum("ijkl->jilk", x70)
    x73 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x73 += einsum("ia,ijkl->jkla", t1.aa, x72)
    del x72
    x74 += einsum("ijka->ikja", x73)
    del x73
    x75 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x75 += einsum("ijka->ikja", x74)
    x75 += einsum("ijka->ijka", x74) * -1
    del x74
    l1new_aa += einsum("iajb,kijb->ak", v.aaaa.ovov, x75) * -1
    del x75
    x118 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x118 += einsum("ijkl->jikl", x70)
    x118 += einsum("ijkl->jilk", x70) * -1
    l1new_aa += einsum("ijka,jlik->al", v.aaaa.ooov, x118) * -1
    del x118
    x358 += einsum("ijkl->jilk", x70)
    del x70
    x359 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x359 += einsum("iajb,klji->klab", v.aaaa.ovov, x358)
    del x358
    x360 += einsum("ijab->jiab", x359)
    del x359
    l2new_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    l2new_aaaa += einsum("ijab->abij", x360)
    l2new_aaaa += einsum("ijab->baij", x360) * -1
    del x360
    x76 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x76 += einsum("abci->iabc", v.aabb.vvvo)
    x76 += einsum("ia,bcda->ibcd", t1.bb, v.aabb.vvvv)
    l1new_aa += einsum("abij,jacb->ci", l2.abab, x76)
    del x76
    x77 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x77 += einsum("w,wai->ia", s1, g.aa.bvo)
    x116 += einsum("ia->ia", x77)
    del x77
    x78 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x78 += einsum("wab,wib->ia", g.aa.bvv, u11.aa)
    x116 += einsum("ia->ia", x78)
    del x78
    x79 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x79 += einsum("ia,iabj->jb", t1.bb, v.bbaa.ovvo)
    x116 += einsum("ia->ia", x79)
    del x79
    x80 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x80 += einsum("ijab,jkib->ka", t2.aaaa, v.aaaa.ooov)
    x116 += einsum("ia->ia", x80)
    del x80
    x81 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x81 += einsum("ijab,jbca->ic", t2.abab, v.bbaa.ovvv)
    x116 += einsum("ia->ia", x81)
    del x81
    x83 += einsum("ijka->ikja", v.aaaa.ooov)
    x84 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x84 += einsum("ijab,ijkb->ka", t2.aaaa, x83)
    del x83
    x116 += einsum("ia->ia", x84) * -1
    del x84
    x87 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x87 += einsum("w,wia->ia", s1, g.bb.bov)
    x91 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x91 += einsum("ia->ia", x87)
    x235 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x235 += einsum("ia->ia", x87)
    x384 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x384 += einsum("ia->ia", x87)
    x494 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x494 += einsum("ia->ia", x87)
    x88 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x88 += einsum("ia,iajb->jb", t1.aa, v.aabb.ovov)
    x91 += einsum("ia->ia", x88)
    x241 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x241 += einsum("ia->ia", x88)
    x289 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x289 += einsum("ia->ia", x88)
    x402 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x402 += einsum("ia->ia", x88)
    x494 += einsum("ia->ia", x88)
    del x88
    x89 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x89 += einsum("iajb->jiab", v.bbbb.ovov)
    x89 += einsum("iajb->jiba", v.bbbb.ovov) * -1
    x90 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x90 += einsum("ia,ijab->jb", t1.bb, x89)
    x91 += einsum("ia->ia", x90) * -1
    x241 += einsum("ia->ia", x90) * -1
    x242 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x242 += einsum("ia,jkab->jkib", x241, t2.bbbb)
    x247 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x247 += einsum("ijka->jika", x242)
    del x242
    x415 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x415 += einsum("ia,ja->ij", t1.bb, x241)
    x416 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x416 += einsum("ij->ij", x415)
    del x415
    x525 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x525 += einsum("ia,ib->ab", t1.bb, x241) * -1
    x289 += einsum("ia->ia", x90) * -1
    del x90
    x274 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x274 += einsum("wia,ijab->wjb", u11.bb, x89)
    x275 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x275 += einsum("wia->wia", x274) * -1
    del x274
    x414 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x414 += einsum("ijab,ikab->jk", t2.bbbb, x89)
    x416 += einsum("ij->ij", x414) * -1
    del x414
    x440 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x440 += einsum("ijab,jkbc->ikac", t2.abab, x89)
    x441 += einsum("ijab->ijab", x440) * -1
    del x440
    x471 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x471 += einsum("ijab,ijbc->ac", t2.bbbb, x89)
    x472 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x472 += einsum("ab->ab", x471) * -1
    x525 += einsum("ab->ab", x471) * -1
    del x471
    x91 += einsum("ia->ia", f.bb.ov)
    x92 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x92 += einsum("ia,jiba->jb", x91, t2.abab)
    x116 += einsum("ia->ia", x92)
    del x92
    x129 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x129 += einsum("ia,ijab->jb", x91, x28)
    x154 += einsum("ia->ia", x129) * -1
    del x129
    x144 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x144 += einsum("ia,ja->ij", t1.bb, x91)
    x145 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x145 += einsum("ij->ji", x144)
    del x144
    x214 = np.zeros((nbos), dtype=types[float])
    x214 += einsum("ia,wia->w", x91, u11.bb)
    x215 += einsum("w->w", x214)
    del x214
    x228 += einsum("ia,jkba->jikb", x91, t2.abab)
    x276 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x276 += einsum("ia,wja->wij", x91, u11.bb)
    x482 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x482 += einsum("ia,jkib->jkba", x91, x0)
    l2new_baba += einsum("ijab->baji", x482) * -1
    l2new_abab += einsum("ijab->abij", x482) * -1
    del x482
    lu11new_bb = np.zeros((nbos, nvir[1], nocc[1]), dtype=types[float])
    lu11new_bb += einsum("w,ia->wai", ls1, x91) * 2
    del x91
    x94 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x94 += einsum("ia,wja->wji", t1.aa, g.aa.bov)
    x95 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x95 += einsum("wij->wij", x94)
    x521 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x521 += einsum("wij->wij", x94)
    del x94
    x95 += einsum("wij->wij", g.aa.boo)
    x96 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x96 += einsum("wia,wij->ja", u11.aa, x95)
    x116 += einsum("ia->ia", x96) * -1
    del x96
    x501 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x501 += einsum("ia,wij->wja", t1.aa, x95)
    x502 += einsum("wia->wia", x501) * -1
    del x501
    x97 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x97 += einsum("w,wij->ij", s1, g.aa.boo)
    x105 += einsum("ij->ij", x97)
    x316 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x316 += einsum("ij->ij", x97)
    del x97
    x98 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x98 += einsum("wia,wja->ij", g.aa.bov, u11.aa)
    x105 += einsum("ij->ij", x98)
    x316 += einsum("ij->ij", x98)
    del x98
    x99 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x99 += einsum("ia,jkia->jk", t1.bb, v.aabb.ooov)
    x105 += einsum("ij->ij", x99)
    x319 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x319 += einsum("ij->ij", x99)
    del x99
    x100 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x100 += einsum("ijab,kajb->ik", t2.abab, v.aabb.ovov)
    x105 += einsum("ij->ji", x100)
    x345 += einsum("ij->ij", x100)
    del x100
    x346 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x346 += einsum("ij,abik->kjab", x345, l2.aaaa)
    del x345
    x355 += einsum("ijab->ijba", x346)
    del x346
    x102 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x102 += einsum("ijka->ikja", v.aaaa.ooov) * -1
    x102 += einsum("ijka->kija", v.aaaa.ooov)
    x103 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x103 += einsum("ia,jika->jk", t1.aa, x102)
    x105 += einsum("ij->ij", x103) * -1
    del x103
    x196 += einsum("wia,jika->wjk", u11.aa, x102) * -1
    del x102
    x105 += einsum("ij->ij", f.aa.oo)
    x106 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x106 += einsum("ia,ij->ja", t1.aa, x105)
    x116 += einsum("ia->ia", x106) * -1
    del x106
    x475 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x475 += einsum("ij,abjk->ikab", x105, l2.abab)
    l2new_baba += einsum("ijab->baji", x475) * -1
    l2new_abab += einsum("ijab->abij", x475) * -1
    del x475
    l1new_aa += einsum("ai,ji->aj", l1.aa, x105) * -1
    lu11new_aa += einsum("ij,waj->wai", x105, lu11.aa) * -1
    del x105
    x107 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x107 += einsum("w,wab->ab", s1, g.aa.bvv)
    x110 += einsum("ab->ab", x107)
    x309 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x309 += einsum("ab->ab", x107)
    x469 += einsum("ab->ab", x107)
    x517 += einsum("ab->ab", x107)
    del x107
    x108 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x108 += einsum("ia,iabc->bc", t1.bb, v.bbaa.ovvv)
    x110 += einsum("ab->ab", x108)
    x312 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x312 += einsum("ab->ab", x108)
    x469 += einsum("ab->ab", x108)
    x517 += einsum("ab->ab", x108)
    del x108
    x110 += einsum("ab->ab", f.aa.vv)
    x111 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x111 += einsum("ia,ba->ib", t1.aa, x110)
    x116 += einsum("ia->ia", x111)
    del x111
    l1new_aa += einsum("ai,ab->bi", l1.aa, x110)
    del x110
    x112 = np.zeros((nbos), dtype=types[float])
    x112 += einsum("ia,wia->w", t1.aa, g.aa.bov)
    x114 = np.zeros((nbos), dtype=types[float])
    x114 += einsum("w->w", x112)
    del x112
    x113 = np.zeros((nbos), dtype=types[float])
    x113 += einsum("ia,wia->w", t1.bb, g.bb.bov)
    x114 += einsum("w->w", x113)
    del x113
    x114 += einsum("w->w", G)
    x115 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x115 += einsum("w,wia->ia", x114, u11.aa)
    x116 += einsum("ia->ia", x115)
    del x115
    x153 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x153 += einsum("w,wia->ia", x114, u11.bb)
    del x114
    x154 += einsum("ia->ia", x153)
    del x153
    x116 += einsum("ai->ia", f.aa.vo)
    l1new_aa += einsum("ia,ijba->bj", x116, x41)
    l1new_bb += einsum("ia,abij->bj", x116, l2.abab)
    ls1new = np.zeros((nbos), dtype=types[float])
    ls1new += einsum("ia,wai->w", x116, lu11.aa)
    del x116
    x119 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x119 += einsum("w,wai->ia", s1, g.bb.bvo)
    x154 += einsum("ia->ia", x119)
    del x119
    x120 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x120 += einsum("wab,wib->ia", g.bb.bvv, u11.bb)
    x154 += einsum("ia->ia", x120)
    del x120
    x121 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x121 += einsum("ia,iabj->jb", t1.aa, v.aabb.ovvo)
    x154 += einsum("ia->ia", x121)
    del x121
    x122 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x122 += einsum("ijab,iacb->jc", t2.abab, v.aabb.ovvv)
    x154 += einsum("ia->ia", x122)
    del x122
    x123 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x123 += einsum("ijab,ikja->kb", t2.bbbb, v.bbbb.ooov)
    x154 += einsum("ia->ia", x123)
    del x123
    x125 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x125 += einsum("ia,jbka->ijkb", t1.bb, v.bbbb.ovov)
    x126 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x126 += einsum("ijka->jkia", x125) * -1
    x126 += einsum("ijka->kjia", x125)
    x220 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x220 += einsum("ijka->ikja", x125)
    x220 += einsum("ijka->ijka", x125) * -1
    x225 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x225 += einsum("ijka->jkia", x125) * -1
    x225 += einsum("ijka->kjia", x125)
    x244 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x244 += einsum("ia,jkla->ijkl", t1.bb, x125)
    x245 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x245 += einsum("ijkl->ijkl", x244)
    x406 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x406 += einsum("ijkl->ijkl", x244)
    del x244
    x267 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x267 += einsum("ijka->jkia", x125)
    x399 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x399 += einsum("ijka->ijka", x125)
    x410 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x410 += einsum("ijka->ikja", x125) * -1
    x410 += einsum("ijka->ijka", x125)
    del x125
    x463 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x463 += einsum("ijka,jlkb->ilab", x0, x410)
    l2new_baba += einsum("ijab->baji", x463) * -1
    l2new_abab += einsum("ijab->abij", x463) * -1
    del x463
    x126 += einsum("ijka->ikja", v.bbbb.ooov)
    x127 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x127 += einsum("ijab,jika->kb", t2.bbbb, x126)
    del x126
    x154 += einsum("ia->ia", x127) * -1
    del x127
    x131 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x131 += einsum("iabj->ijba", v.bbbb.ovvo)
    x131 += einsum("ijab->ijab", v.bbbb.oovv) * -1
    x132 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x132 += einsum("ia,ijba->jb", t1.bb, x131)
    x154 += einsum("ia->ia", x132)
    del x132
    x223 += einsum("ia,jkba->ijkb", t1.bb, x131)
    x133 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x133 += einsum("ia,wja->wji", t1.bb, g.bb.bov)
    x134 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x134 += einsum("wij->wij", x133)
    x526 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x526 += einsum("wij->wij", x133)
    del x133
    x134 += einsum("wij->wij", g.bb.boo)
    x135 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x135 += einsum("wia,wij->ja", u11.bb, x134)
    x154 += einsum("ia->ia", x135) * -1
    del x135
    x510 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x510 += einsum("ia,wij->wja", t1.bb, x134)
    x511 += einsum("wia->wia", x510) * -1
    del x510
    x136 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x136 += einsum("w,wij->ij", s1, g.bb.boo)
    x145 += einsum("ij->ij", x136)
    x386 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x386 += einsum("ij->ij", x136)
    del x136
    x137 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x137 += einsum("wia,wja->ij", g.bb.bov, u11.bb)
    x145 += einsum("ij->ij", x137)
    x386 += einsum("ij->ij", x137)
    del x137
    x138 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x138 += einsum("ia,iajk->jk", t1.aa, v.aabb.ovoo)
    x145 += einsum("ij->ij", x138)
    x389 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x389 += einsum("ij->ij", x138)
    del x138
    x139 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x139 += einsum("ijab,iakb->jk", t2.abab, v.aabb.ovov)
    x145 += einsum("ij->ji", x139)
    x416 += einsum("ij->ij", x139)
    del x139
    x417 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x417 += einsum("ij,abik->kjab", x416, l2.bbbb)
    del x416
    x426 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x426 += einsum("ijab->ijba", x417)
    del x417
    x140 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x140 += einsum("iajb->jiab", v.bbbb.ovov) * -1
    x140 += einsum("iajb->jiba", v.bbbb.ovov)
    x141 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x141 += einsum("ijab,ikba->jk", t2.bbbb, x140)
    x145 += einsum("ij->ji", x141) * -1
    del x141
    x419 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x419 += einsum("ijab,ijbc->ac", t2.bbbb, x140)
    x420 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x420 += einsum("ab->ab", x419) * -1
    del x419
    x443 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x443 += einsum("ijab,ikcb->kjca", x140, x51)
    del x140
    x445 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x445 += einsum("ijab->jiab", x443)
    del x443
    x142 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x142 += einsum("ijka->ikja", v.bbbb.ooov)
    x142 += einsum("ijka->kija", v.bbbb.ooov) * -1
    x143 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x143 += einsum("ia,ijka->jk", t1.bb, x142)
    x145 += einsum("ij->ij", x143) * -1
    del x143
    x276 += einsum("wia,ijka->wjk", u11.bb, x142) * -1
    x145 += einsum("ij->ij", f.bb.oo)
    x146 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x146 += einsum("ia,ij->ja", t1.bb, x145)
    x154 += einsum("ia->ia", x146) * -1
    del x146
    x474 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x474 += einsum("ij,abkj->kiab", x145, l2.abab)
    l2new_baba += einsum("ijab->baji", x474) * -1
    l2new_abab += einsum("ijab->abij", x474) * -1
    del x474
    l1new_bb += einsum("ai,ji->aj", l1.bb, x145) * -1
    lu11new_bb += einsum("ij,waj->wai", x145, lu11.bb) * -1
    del x145
    x147 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x147 += einsum("w,wab->ab", s1, g.bb.bvv)
    x151 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x151 += einsum("ab->ab", x147)
    x379 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x379 += einsum("ab->ab", x147)
    x472 += einsum("ab->ab", x147)
    x525 += einsum("ab->ab", x147)
    del x147
    x148 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x148 += einsum("ia,iabc->bc", t1.aa, v.aabb.ovvv)
    x151 += einsum("ab->ab", x148)
    x382 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x382 += einsum("ab->ab", x148)
    x472 += einsum("ab->ab", x148)
    x525 += einsum("ab->ab", x148)
    del x148
    x149 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x149 += einsum("iabc->ibac", v.bbbb.ovvv) * -1
    x149 += einsum("iabc->ibca", v.bbbb.ovvv)
    x150 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x150 += einsum("ia,ibac->bc", t1.bb, x149)
    x151 += einsum("ab->ab", x150) * -1
    x472 += einsum("ab->ab", x150) * -1
    x525 += einsum("ab->ab", x150) * -1
    del x150
    x444 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x444 += einsum("ia,jbca->ijbc", t1.bb, x149)
    x445 += einsum("ijab->jiab", x444) * -1
    del x444
    x151 += einsum("ab->ab", f.bb.vv)
    x152 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x152 += einsum("ia,ba->ib", t1.bb, x151)
    x154 += einsum("ia->ia", x152)
    del x152
    l1new_bb += einsum("ai,ab->bi", l1.bb, x151)
    del x151
    x154 += einsum("ai->ia", f.bb.vo)
    l1new_aa += einsum("ia,baji->bj", x154, l2.abab)
    ls1new += einsum("ia,wai->w", x154, lu11.bb)
    x155 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x155 += einsum("w,wia->ia", ls1, u11.aa)
    x165 += einsum("ia->ia", x155) * -1
    del x155
    x156 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x156 += einsum("ai,jiba->jb", l1.bb, t2.abab)
    x165 += einsum("ia->ia", x156) * -1
    del x156
    x157 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x157 += einsum("ia,waj->wji", t1.aa, lu11.aa)
    x158 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x158 += einsum("wia,wij->ja", u11.aa, x157)
    x165 += einsum("ia->ia", x158)
    del x158
    x503 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x503 += einsum("ia,wij->wja", t1.aa, x157)
    x506 += einsum("wia->wia", x503) * -1
    del x503
    lu11new_bb -= einsum("wij,jika->wak", x157, v.aabb.ooov)
    x162 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x162 += einsum("wai,wja->ij", lu11.aa, u11.aa)
    x163 += einsum("ij->ij", x162)
    x164 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x164 += einsum("ia,ij->ja", t1.aa, x163)
    x165 += einsum("ia->ia", x164)
    del x164
    x481 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x481 += einsum("ij,jakb->ikab", x163, v.aabb.ovov)
    l2new_baba += einsum("ijab->baji", x481) * -1
    l2new_abab += einsum("ijab->abij", x481) * -1
    del x481
    l1new_bb += einsum("ij,jika->ak", x163, v.aabb.ooov) * -1
    ls1new += einsum("ij,wji->w", x163, g.aa.boo) * -1
    lu11new_aa += einsum("ij,wja->wai", x163, g.aa.bov) * -1
    x205 += einsum("ij->ij", x162)
    x206 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x206 += einsum("w,ij->wij", s1, x205)
    del x205
    x208 += einsum("ij->ij", x162)
    x326 += einsum("ij->ij", x162)
    del x162
    x327 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x327 += einsum("ij,jakb->ikab", x326, v.aaaa.ovov)
    del x326
    x333 += einsum("ijab->ijba", x327)
    del x327
    x165 += einsum("ia->ia", t1.aa) * -1
    l1new_aa += einsum("ia,ijba->bj", x165, x61) * -1
    del x61
    l1new_bb += einsum("ia,iajb->bj", x165, v.aabb.ovov) * -1
    ls1new += einsum("ia,wia->w", x165, g.aa.bov) * -1
    del x165
    x166 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x166 += einsum("ai,ib->ab", l1.aa, t1.aa)
    x170 += einsum("ab->ab", x166)
    del x166
    x167 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x167 += einsum("wai,wib->ab", lu11.aa, u11.aa)
    x170 += einsum("ab->ab", x167)
    x292 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x292 += einsum("ab,icjb->ijac", x167, v.aaaa.ovov)
    x333 += einsum("ijab->ijab", x292)
    del x292
    x476 += einsum("ab->ab", x167)
    del x167
    x168 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x168 += einsum("abij,ijcb->ac", l2.abab, t2.abab)
    x170 += einsum("ab->ab", x168)
    l1new_bb += einsum("ab,icab->ci", x170, v.bbaa.ovvv)
    ls1new += einsum("ab,wab->w", x170, g.aa.bvv)
    x347 += einsum("ab->ab", x168)
    x348 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x348 += einsum("ab,ibjc->ijac", x347, v.aaaa.ovov)
    del x347
    x355 += einsum("ijab->jiab", x348) * -1
    del x348
    x476 += einsum("ab->ab", x168)
    del x168
    x477 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x477 += einsum("ab,ibjc->ijac", x476, v.aabb.ovov)
    l2new_baba += einsum("ijab->baji", x477) * -1
    l2new_abab += einsum("ijab->abij", x477) * -1
    del x477
    lu11new_aa += einsum("ab,wib->wai", x476, g.aa.bov) * -1
    del x476
    x171 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x171 += einsum("iabc->ibac", v.aaaa.ovvv) * -1
    x171 += einsum("iabc->ibca", v.aaaa.ovvv)
    l1new_aa += einsum("ab,iacb->ci", x170, x171) * -1
    del x170
    del x171
    x172 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x172 += einsum("w,wia->ia", ls1, u11.bb)
    x186 += einsum("ia->ia", x172) * -1
    del x172
    x173 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x173 += einsum("ai,ijab->jb", l1.aa, t2.abab)
    x186 += einsum("ia->ia", x173) * -1
    del x173
    x174 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x174 += einsum("ia,waj->wji", t1.bb, lu11.bb)
    x175 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x175 += einsum("wia,wij->ja", u11.bb, x174)
    x186 += einsum("ia->ia", x175)
    del x175
    x513 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x513 += einsum("ia,wij->wja", t1.bb, x174)
    x515 += einsum("wia->wia", x513) * -1
    del x513
    lu11new_aa -= einsum("wij,kaji->wak", x174, v.aabb.ovoo)
    x177 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x177 += einsum("ia,bajk->jkib", t1.bb, l2.bbbb)
    x178 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x178 += einsum("ijka,jiab->kb", x177, x51)
    x186 += einsum("ia->ia", x178) * -1
    del x178
    x250 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x250 += einsum("ijka->ijka", x177)
    x250 += einsum("ijka->jika", x177) * -1
    x252 += einsum("ijka,jlba->ilkb", x250, x51)
    del x51
    x258 += einsum("ijab,jklb->ikla", t2.abab, x250) * -1
    x411 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x411 += einsum("ijka,jlkb->ilab", x250, x410)
    del x410
    x426 += einsum("ijab->ijab", x411)
    del x411
    l1new_bb += einsum("ijab,jkia->bk", x131, x250) * -1
    del x131
    l2new_baba += einsum("ijka,lkib->abjl", x250, x30)
    del x250
    x253 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x253 += einsum("ijka->ijka", x177) * -1
    x253 += einsum("ijka->jika", x177)
    x254 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x254 += einsum("ia,ijkb->jkba", t1.bb, x253) * -1
    l1new_bb += einsum("iabc,jiac->bj", x149, x254)
    del x254
    l2new_abab += einsum("ijka,lkib->balj", x253, x30) * -1
    del x253
    del x30
    x265 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x265 += einsum("ia,jkla->kjli", t1.bb, x177)
    x266 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x266 += einsum("ijkl->ijkl", x265) * -1
    x266 += einsum("ijkl->ijlk", x265)
    x405 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x405 += einsum("iajb,klji->lkab", v.bbbb.ovov, x265)
    del x265
    x408 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x408 += einsum("ijab->ijab", x405)
    del x405
    x366 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x366 += einsum("iabc,jkib->kjac", v.bbbb.ovvv, x177)
    x403 -= einsum("ijab->ijab", x366)
    del x366
    x375 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x375 += einsum("ijka->ijka", x177)
    x375 -= einsum("ijka->jika", x177)
    x425 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x425 += einsum("ia,jkib->jkba", x241, x177)
    del x241
    x426 += einsum("ijab->ijab", x425)
    del x425
    x180 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x180 += einsum("ai,ja->ij", l1.bb, t1.bb)
    x184 += einsum("ij->ij", x180)
    x288 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x288 += einsum("ij->ij", x180)
    x396 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x396 += einsum("ij->ij", x180)
    l1new_bb -= einsum("ij,ja->ai", x180, x87)
    del x180
    del x87
    x181 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x181 += einsum("wai,wja->ij", lu11.bb, u11.bb)
    x184 += einsum("ij->ij", x181)
    x285 += einsum("ij->ij", x181)
    x288 += einsum("ij->ij", x181)
    x396 += einsum("ij->ij", x181)
    del x181
    x397 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x397 += einsum("ij,jakb->kiab", x396, v.bbbb.ovov)
    del x396
    x403 += einsum("ijab->jiba", x397)
    del x397
    x182 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x182 += einsum("abij,ikab->jk", l2.abab, t2.abab)
    x184 += einsum("ij->ij", x182)
    x185 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x185 += einsum("ia,ij->ja", t1.bb, x184)
    x186 += einsum("ia->ia", x185)
    del x185
    x480 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x480 += einsum("ij,kajb->kiab", x184, v.aabb.ovov)
    l2new_baba += einsum("ijab->baji", x480) * -1
    l2new_abab += einsum("ijab->abij", x480) * -1
    del x480
    l1new_aa += einsum("ij,kaji->ak", x184, v.aabb.ovoo) * -1
    l1new_bb += einsum("ij,jkia->ak", x184, x142) * -1
    del x142
    ls1new += einsum("ij,wji->w", x184, g.bb.boo) * -1
    lu11new_bb += einsum("ij,wja->wai", x184, g.bb.bov) * -1
    del x184
    x251 += einsum("ij->ij", x182)
    x252 += einsum("ia,jk->jika", t1.bb, x251)
    l1new_bb += einsum("ijka,kjab->bi", x252, x89) * -1
    del x252
    x258 += einsum("ia,jk->ijka", t1.aa, x251) * -1
    x424 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x424 += einsum("ij,jakb->kiab", x251, v.bbbb.ovov)
    x426 += einsum("ijab->jiba", x424) * -1
    del x424
    l1new_bb += einsum("ia,ji->aj", f.bb.ov, x251) * -1
    del x251
    x285 += einsum("ij->ij", x182)
    del x182
    x286 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x286 += einsum("w,ij->wij", s1, x285)
    del x285
    x186 += einsum("ia->ia", t1.bb) * -1
    l1new_aa += einsum("ia,jbia->bj", x186, v.aabb.ovov) * -1
    ls1new += einsum("ia,wia->w", x186, g.bb.bov) * -1
    x187 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x187 += einsum("ai,ib->ab", l1.bb, t1.bb)
    x191 += einsum("ab->ab", x187)
    ls1new += einsum("ab,wab->w", x187, g.bb.bvv)
    del x187
    x188 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x188 += einsum("wai,wib->ab", lu11.bb, u11.bb)
    x191 += einsum("ab->ab", x188)
    x363 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x363 += einsum("ab,icjb->ijac", x188, v.bbbb.ovov)
    x403 += einsum("ijab->ijab", x363)
    del x363
    x478 += einsum("ab->ab", x188)
    del x188
    x189 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x189 += einsum("abij,ijac->bc", l2.abab, t2.abab)
    x191 += einsum("ab->ab", x189)
    l1new_aa += einsum("ab,icab->ci", x191, v.aabb.ovvv)
    l1new_bb += einsum("ab,iacb->ci", x191, x149) * -1
    del x149
    del x191
    x422 += einsum("ab->ab", x189)
    x423 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x423 += einsum("ab,ibjc->ijca", x422, v.bbbb.ovov)
    del x422
    x426 += einsum("ijab->jiba", x423) * -1
    del x423
    x478 += einsum("ab->ab", x189)
    del x189
    x479 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x479 += einsum("ab,icjb->ijca", x478, v.aabb.ovov)
    l2new_baba += einsum("ijab->baji", x479) * -1
    l2new_abab += einsum("ijab->abij", x479) * -1
    del x479
    ls1new += einsum("ab,wab->w", x478, g.bb.bvv)
    lu11new_bb += einsum("ab,wib->wai", x478, g.bb.bov) * -1
    del x478
    x192 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x192 -= einsum("ijka->ikja", v.aaaa.ooov)
    x192 += einsum("ijka->kija", v.aaaa.ooov)
    x307 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x307 += einsum("ijka,lkib->jlab", x192, x306)
    del x306
    x333 += einsum("ijab->jiba", x307)
    del x307
    x318 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x318 += einsum("ia,jika->jk", t1.aa, x192)
    x319 -= einsum("ij->ij", x318)
    del x318
    x320 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x320 += einsum("ij,abjk->kiab", x319, l2.aaaa)
    del x319
    x333 -= einsum("ijab->ijba", x320)
    del x320
    l1new_aa += einsum("ij,kjia->ak", x163, x192) * -1
    del x163
    l2new_abab += einsum("ijka,kilb->abjl", x192, x53)
    del x192
    x193 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x193 += einsum("wia,jbia->wjb", u11.bb, v.aabb.ovov)
    x195 += einsum("wia->wia", x193)
    x324 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x324 += einsum("wia->wia", x193)
    x484 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x484 += einsum("wia->wia", x193)
    del x193
    x195 += einsum("wia->wia", gc.aa.bov)
    x196 += einsum("ia,wja->wji", t1.aa, x195)
    l1new_aa += einsum("wij,wja->ai", x157, x195) * -1
    del x195
    x196 += einsum("wij->wij", gc.aa.boo)
    x196 += einsum("wia,jkia->wjk", u11.bb, v.aabb.ooov)
    l1new_aa += einsum("wai,wji->aj", lu11.aa, x196) * -1
    del x196
    x197 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x197 += einsum("iabc->ibac", v.aaaa.ovvv)
    x197 -= einsum("iabc->ibca", v.aaaa.ovvv)
    x198 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x198 -= einsum("wia,ibca->wbc", u11.aa, x197)
    x311 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x311 += einsum("ia,ibca->bc", t1.aa, x197)
    del x197
    x312 -= einsum("ab->ab", x311)
    del x311
    x313 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x313 += einsum("ab,acij->ijcb", x312, l2.aaaa)
    del x312
    x333 += einsum("ijab->jiab", x313)
    del x313
    x198 += einsum("wab->wab", gc.aa.bvv)
    x198 += einsum("wia,iabc->wbc", u11.bb, v.bbaa.ovvv)
    l1new_aa += einsum("wai,wab->bi", lu11.aa, x198)
    del x198
    x199 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x199 += einsum("wia,baji->wjb", u11.bb, l2.abab)
    x202 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x202 += einsum("wia->wia", x199)
    x204 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x204 += einsum("wia->wia", x199)
    del x199
    x200 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x200 -= einsum("abij->jiab", l2.aaaa)
    x200 += einsum("abij->jiba", l2.aaaa)
    x201 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x201 += einsum("wia,ijba->wjb", u11.aa, x200)
    x202 -= einsum("wia->wia", x201)
    del x201
    x321 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x321 += einsum("wia,wjb->ijab", g.aa.bov, x202)
    x333 += einsum("ijab->ijab", x321)
    del x321
    x488 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x488 += einsum("wia,wjb->jiba", g.bb.bov, x202)
    l2new_baba += einsum("ijab->baji", x488)
    l2new_abab += einsum("ijab->abij", x488)
    del x488
    l1new_aa += einsum("wab,wia->bi", g.aa.bvv, x202)
    l1new_aa += einsum("wia,wji->aj", x202, x95) * -1
    del x202
    del x95
    x203 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x203 += einsum("abij->jiab", l2.aaaa) * -1
    x203 += einsum("abij->jiba", l2.aaaa)
    x204 += einsum("wia,ijba->wjb", u11.aa, x203) * -1
    x206 += einsum("ia,wja->wji", t1.aa, x204)
    del x204
    x206 += einsum("ai,wja->wij", l1.aa, u11.aa)
    l1new_aa += einsum("wia,wji->aj", g.aa.bov, x206) * -1
    del x206
    x207 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x207 += einsum("iabj->ijba", v.aaaa.ovvo)
    x207 -= einsum("ijab->ijab", v.aaaa.oovv)
    l1new_aa += einsum("ai,jiab->bj", l1.aa, x207)
    lu11new_aa += einsum("wai,jiab->wbj", lu11.aa, x207)
    del x207
    x209 += einsum("ia->ia", f.aa.ov)
    l1new_aa += einsum("ij,ja->ai", x208, x209) * -1
    del x208
    del x209
    x210 = np.zeros((nbos), dtype=types[float])
    x210 += einsum("w,wx->x", s1, w)
    x215 += einsum("w->w", x210)
    del x210
    x211 = np.zeros((nbos), dtype=types[float])
    x211 += einsum("ia,wia->w", t1.aa, gc.aa.bov)
    x215 += einsum("w->w", x211)
    del x211
    x212 = np.zeros((nbos), dtype=types[float])
    x212 += einsum("ia,wia->w", t1.bb, gc.bb.bov)
    x215 += einsum("w->w", x212)
    del x212
    x215 += einsum("w->w", G)
    l1new_aa += einsum("w,wai->ai", x215, lu11.aa)
    l1new_bb += einsum("w,wai->ai", x215, lu11.bb)
    del x215
    x216 = np.zeros((nbos), dtype=types[float])
    x216 += einsum("ai,wia->w", l1.aa, u11.aa)
    x218 = np.zeros((nbos), dtype=types[float])
    x218 += einsum("w->w", x216)
    del x216
    x217 = np.zeros((nbos), dtype=types[float])
    x217 += einsum("ai,wia->w", l1.bb, u11.bb)
    x218 += einsum("w->w", x217)
    del x217
    x218 += einsum("w->w", s1)
    l1new_aa += einsum("w,wia->ai", x218, g.aa.bov)
    l1new_bb += einsum("w,wia->ai", x218, g.bb.bov)
    del x218
    x219 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x219 += einsum("abij,ikcb->jkac", l2.abab, t2.abab)
    l1new_bb += einsum("iabc,jibc->aj", v.bbaa.ovvv, x219) * -1
    del x219
    x220 += einsum("ijka->jkia", v.bbbb.ooov) * -1
    x220 += einsum("ijka->jika", v.bbbb.ooov)
    x223 += einsum("ijka,jlab->iklb", x220, x28)
    del x220
    x221 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x221 += einsum("ia,jkla->ijlk", t1.bb, v.bbbb.ooov)
    x222 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x222 += einsum("ijkl->ikjl", x221) * -1
    x222 += einsum("ijkl->ijkl", x221)
    x223 += einsum("ia,jikl->jkla", t1.bb, x222) * -1
    del x222
    x365 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x365 += einsum("abij,jkli->klba", l2.bbbb, x221)
    del x221
    x403 -= einsum("ijab->ijab", x365)
    del x365
    x224 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x224 -= einsum("abij->jiab", l2.bbbb)
    x224 += einsum("abij->jiba", l2.bbbb)
    l1new_bb += einsum("ijka,ikab->bj", x223, x224)
    del x223
    x225 += einsum("ijka->ikja", v.bbbb.ooov)
    x225 += einsum("ijka->kija", v.bbbb.ooov) * -1
    x228 += einsum("ijab,jklb->ikla", t2.abab, x225) * -1
    del x225
    x226 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x226 += einsum("ia,jabc->ijbc", t1.bb, v.bbaa.ovvv)
    x227 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x227 += einsum("ijab->jiab", x226)
    x456 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x456 += einsum("ijab->jiab", x226)
    del x226
    x227 += einsum("ijab->ijab", v.bbaa.oovv)
    x228 += einsum("ia,jkba->ijkb", t1.aa, x227)
    del x227
    x228 += einsum("ijak->kija", v.bbaa.oovo)
    x228 += einsum("ia,jabk->kjib", t1.bb, v.bbaa.ovvo)
    x228 += einsum("ijab,kbca->ikjc", t2.abab, v.bbaa.ovvv)
    l1new_bb += einsum("abij,ikja->bk", l2.abab, x228) * -1
    del x228
    x229 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x229 += einsum("abij,ikac->jkbc", l2.abab, t2.abab)
    x232 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x232 += einsum("ijab->ijab", x229)
    del x229
    x230 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x230 += einsum("abij->jiab", l2.bbbb) * -1
    x230 += einsum("abij->jiba", l2.bbbb)
    x231 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x231 += einsum("ijab,ikca->kjcb", x230, x28)
    del x28
    x232 += einsum("ijab->jiba", x231)
    del x231
    x409 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x409 += einsum("ijab,jkcb->kica", x232, x89)
    x426 += einsum("ijab->jiba", x409) * -1
    del x409
    x249 += einsum("ijab,jkbc->ikac", t2.abab, x230) * -1
    del x230
    l1new_bb += einsum("iabc,ijab->cj", v.aabb.ovvv, x249) * -1
    del x249
    x233 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x233 += einsum("iabc->ibac", v.bbbb.ovvv)
    x233 += einsum("iabc->ibca", v.bbbb.ovvv) * -1
    l1new_bb += einsum("ijab,jacb->ci", x232, x233) * -1
    del x232
    del x233
    x234 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x234 += einsum("ia,jkil->jkla", t1.bb, v.bbbb.oooo)
    x237 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x237 += einsum("ijka->ijka", x234)
    del x234
    x235 += einsum("ia->ia", f.bb.ov)
    x236 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x236 += einsum("ia,jkab->jkib", x235, t2.bbbb)
    del x235
    x237 += einsum("ijka->kjia", x236)
    del x236
    x237 += einsum("ijak->ijka", v.bbbb.oovo) * -1
    x248 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x248 += einsum("ijka->ikja", x237)
    x248 += einsum("ijka->ijka", x237) * -1
    del x237
    x238 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x238 += einsum("ijab,kbca->jikc", t2.bbbb, v.bbbb.ovvv)
    x247 += einsum("ijka->ijka", x238) * -1
    del x238
    x239 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x239 += einsum("ia,jbca->ijcb", t1.bb, v.bbbb.ovvv)
    x240 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x240 += einsum("ia,jkba->ijkb", t1.bb, x239)
    del x239
    x247 += einsum("ijka->ijka", x240) * -1
    del x240
    x243 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x243 += einsum("ijab,kalb->ijkl", t2.bbbb, v.bbbb.ovov)
    x245 += einsum("ijkl->jilk", x243)
    x246 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x246 += einsum("ia,jkil->jkla", t1.bb, x245)
    del x245
    x247 += einsum("ijka->jika", x246)
    del x246
    x248 += einsum("ijka->kjia", x247)
    x248 += einsum("ijka->kija", x247) * -1
    del x247
    l1new_bb += einsum("abij,kija->bk", l2.bbbb, x248)
    del x248
    x406 += einsum("ijkl->jilk", x243)
    del x243
    x407 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x407 += einsum("abij,ijkl->klab", l2.bbbb, x406)
    del x406
    x408 += einsum("ijab->jiba", x407)
    del x407
    x255 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x255 += einsum("ia,bcda->ibdc", t1.bb, v.bbbb.vvvv)
    x256 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x256 += einsum("iabc->iabc", x255) * -1
    del x255
    x256 += einsum("aibc->iabc", v.bbbb.vovv)
    x257 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x257 += einsum("iabc->iabc", x256) * -1
    x257 += einsum("iabc->ibac", x256)
    del x256
    l1new_bb += einsum("abij,iabc->cj", l2.bbbb, x257) * -1
    del x257
    x258 += einsum("ai,jkba->jikb", l1.bb, t2.abab) * -1
    l1new_bb += einsum("iajb,ikja->bk", v.aabb.ovov, x258)
    del x258
    x259 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x259 += einsum("ai,jkba->ijkb", l1.bb, t2.bbbb)
    x262 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x262 += einsum("ijka->ijka", x259)
    del x259
    x260 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x260 += einsum("abij,klba->ijlk", l2.bbbb, t2.bbbb)
    x261 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x261 += einsum("ia,jikl->jkla", t1.bb, x260)
    x262 += einsum("ijka->ijka", x261)
    del x261
    x263 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x263 += einsum("ijka->ikja", x262)
    x263 += einsum("ijka->ijka", x262) * -1
    del x262
    l1new_bb += einsum("iajb,kija->bk", v.bbbb.ovov, x263)
    del x263
    x271 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x271 += einsum("ijkl->jikl", x260) * -1
    x271 += einsum("ijkl->jilk", x260)
    l1new_bb += einsum("ijka,jlki->al", v.bbbb.ooov, x271) * -1
    del x271
    x404 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x404 += einsum("iajb,klij->lkba", v.bbbb.ovov, x260)
    del x260
    x408 += einsum("ijab->ijab", x404)
    del x404
    l2new_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    l2new_bbbb += einsum("ijab->abij", x408)
    l2new_bbbb += einsum("ijab->baij", x408) * -1
    del x408
    x264 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x264 += einsum("aibc->iabc", v.aabb.vovv)
    x264 += einsum("ia,bacd->ibcd", t1.aa, v.aabb.vvvv)
    l1new_bb += einsum("abij,iabc->cj", l2.abab, x264)
    del x264
    x267 += einsum("ijka->ikja", v.bbbb.ooov) * -1
    l1new_bb += einsum("ijkl,klja->ai", x266, x267) * -1
    del x266
    del x267
    x268 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x268 += einsum("ia,jbca->ijcb", t1.aa, v.bbaa.ovvv)
    x269 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x269 += einsum("ijab->ijab", x268)
    x373 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x373 += einsum("ijab->ijab", x268)
    x441 += einsum("ijab->ijab", x268)
    del x268
    x269 += einsum("iabj->jiba", v.bbaa.ovvo)
    l1new_bb += einsum("ijka,ikab->bj", x0, x269) * -1
    del x269
    x270 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x270 += einsum("abij->jiab", l2.bbbb)
    x270 += einsum("abij->jiba", l2.bbbb) * -1
    x283 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x283 += einsum("wia,ijab->wjb", u11.bb, x270)
    x284 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x284 += einsum("wia->wia", x283) * -1
    del x283
    x412 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x412 += einsum("ijab,jkbc->ikac", t2.abab, x270)
    x413 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x413 += einsum("iajb,ikac->jkbc", v.aabb.ovov, x412) * -1
    del x412
    x426 += einsum("ijab->jiba", x413) * -1
    del x413
    l1new_bb += einsum("ia,ijab->bj", x154, x270) * -1
    del x154
    x272 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x272 -= einsum("iajb->jiab", v.bbbb.ovov)
    x272 += einsum("iajb->jiba", v.bbbb.ovov)
    l1new_bb += einsum("ia,ijab->bj", x186, x272) * -1
    del x272
    del x186
    x273 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x273 += einsum("wia,iajb->wjb", u11.aa, v.aabb.ovov)
    x275 += einsum("wia->wia", x273)
    x394 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x394 += einsum("wia->wia", x273)
    x486 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x486 += einsum("wia->wia", x273)
    del x273
    x275 += einsum("wia->wia", gc.bb.bov)
    x276 += einsum("ia,wja->wji", t1.bb, x275)
    l1new_bb += einsum("wij,wja->ai", x174, x275) * -1
    del x275
    x276 += einsum("wij->wij", gc.bb.boo)
    x276 += einsum("wia,iajk->wjk", u11.aa, v.aabb.ovoo)
    l1new_bb += einsum("wai,wji->aj", lu11.bb, x276) * -1
    del x276
    x277 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x277 -= einsum("iabc->ibac", v.bbbb.ovvv)
    x277 += einsum("iabc->ibca", v.bbbb.ovvv)
    x278 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x278 -= einsum("wia,ibac->wbc", u11.bb, x277)
    x368 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x368 += einsum("ia,jbca->ijbc", t1.bb, x277)
    x369 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x369 -= einsum("ijab->ijab", x368)
    del x368
    x381 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x381 += einsum("ia,ibac->bc", t1.bb, x277)
    x382 -= einsum("ab->ab", x381)
    del x381
    x383 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x383 += einsum("ab,acij->ijcb", x382, l2.bbbb)
    del x382
    x403 += einsum("ijab->jiab", x383)
    del x383
    x278 += einsum("wab->wab", gc.bb.bvv)
    x278 += einsum("wia,iabc->wbc", u11.aa, v.aabb.ovvv)
    l1new_bb += einsum("wai,wab->bi", lu11.bb, x278)
    del x278
    x279 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x279 += einsum("wia,abij->wjb", u11.aa, l2.abab)
    x282 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x282 += einsum("wia->wia", x279)
    x284 += einsum("wia->wia", x279)
    del x279
    x286 += einsum("ia,wja->wji", t1.bb, x284)
    l1new_bb += einsum("wij,wja->ai", x134, x284) * -1
    del x284
    del x134
    x280 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x280 += einsum("abij->jiab", l2.bbbb)
    x280 -= einsum("abij->jiba", l2.bbbb)
    x281 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x281 += einsum("wia,ijab->wjb", u11.bb, x280)
    del x280
    x282 -= einsum("wia->wia", x281)
    del x281
    x391 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x391 += einsum("wia,wjb->ijab", g.bb.bov, x282)
    x403 += einsum("ijab->ijab", x391)
    del x391
    x489 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x489 += einsum("wia,wjb->ijab", g.aa.bov, x282)
    l2new_baba += einsum("ijab->baji", x489)
    l2new_abab += einsum("ijab->abij", x489)
    del x489
    l1new_bb += einsum("wab,wia->bi", g.bb.bvv, x282)
    del x282
    x286 += einsum("ai,wja->wij", l1.bb, u11.bb)
    l1new_bb += einsum("wia,wji->aj", g.bb.bov, x286) * -1
    del x286
    x287 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x287 += einsum("iabj->ijba", v.bbbb.ovvo)
    x287 -= einsum("ijab->ijab", v.bbbb.oovv)
    l1new_bb += einsum("ai,jiab->bj", l1.bb, x287)
    lu11new_bb += einsum("wai,jiab->wbj", lu11.bb, x287)
    del x287
    x289 += einsum("ia->ia", f.bb.ov)
    l1new_bb += einsum("ij,ja->ai", x288, x289) * -1
    del x288
    del x289
    x290 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x290 += einsum("wia,wbj->ijab", gc.aa.bov, lu11.aa)
    x333 += einsum("ijab->ijab", x290)
    del x290
    x291 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x291 += einsum("ai,jbac->ijbc", l1.aa, v.aaaa.ovvv)
    x333 -= einsum("ijab->ijab", x291)
    del x291
    x297 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x297 += einsum("abij->jiab", l2.aaaa)
    x297 -= einsum("abij->jiba", l2.aaaa)
    x298 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x298 -= einsum("iabc->ibac", v.aaaa.ovvv)
    x298 += einsum("iabc->ibca", v.aaaa.ovvv)
    x299 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x299 += einsum("ia,jbca->ijbc", t1.aa, x298)
    x300 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x300 -= einsum("ijab->ijab", x299)
    del x299
    x300 += einsum("iabj->jiba", v.aaaa.ovvo)
    x300 -= einsum("ijab->jiab", v.aaaa.oovv)
    x301 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x301 += einsum("ijab,ikbc->jkac", x297, x300)
    del x297
    del x300
    x333 += einsum("ijab->ijab", x301)
    del x301
    x302 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x302 += einsum("ijab->jiab", t2.bbbb)
    x302 -= einsum("ijab->jiba", t2.bbbb)
    x303 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x303 += einsum("iajb,jkbc->ikac", v.aabb.ovov, x302)
    del x302
    x304 -= einsum("ijab->ijab", x303)
    del x303
    x304 += einsum("iabj->ijab", v.aabb.ovvo)
    x305 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x305 += einsum("abij,kjcb->ikac", l2.abab, x304)
    del x304
    x333 += einsum("ijab->ijab", x305)
    del x305
    x308 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x308 += einsum("wia,wib->ab", g.aa.bov, u11.aa)
    x309 -= einsum("ab->ba", x308)
    x469 += einsum("ab->ba", x308) * -1
    x517 += einsum("ab->ba", x308) * -1
    del x308
    x309 += einsum("ab->ab", f.aa.vv)
    x310 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x310 += einsum("ab,acij->ijcb", x309, l2.aaaa)
    del x309
    x333 -= einsum("ijab->jiba", x310)
    del x310
    x314 += einsum("ia->ia", f.aa.ov)
    x315 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x315 += einsum("ia,ja->ij", t1.aa, x314)
    x316 += einsum("ij->ji", x315)
    del x315
    x328 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x328 += einsum("ia,jkib->jkba", x314, x54)
    del x54
    x333 += einsum("ijab->ijba", x328)
    del x328
    x333 += einsum("ai,jb->jiba", l1.aa, x314)
    lu11new_aa -= einsum("ia,wji->waj", x314, x157)
    del x314
    x316 += einsum("ij->ij", f.aa.oo)
    x317 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x317 += einsum("ij,abjk->kiab", x316, l2.aaaa)
    del x316
    x333 += einsum("ijab->jiba", x317)
    del x317
    x322 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x322 -= einsum("iajb->jiab", v.aaaa.ovov)
    x322 += einsum("iajb->jiba", v.aaaa.ovov)
    x323 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x323 += einsum("wia,ijba->wjb", u11.aa, x322)
    x324 -= einsum("wia->wia", x323)
    x325 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x325 += einsum("wai,wjb->ijab", lu11.aa, x324)
    del x324
    x333 += einsum("ijab->ijab", x325)
    del x325
    x484 -= einsum("wia->wia", x323)
    del x323
    x331 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x331 += einsum("ia,ijba->jb", t1.aa, x322)
    del x322
    x332 -= einsum("ia->ia", x331)
    x333 += einsum("ai,jb->ijab", l1.aa, x332)
    del x332
    x495 -= einsum("ia->ia", x331)
    del x331
    x329 -= einsum("ijka->jika", v.aaaa.ooov)
    x330 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x330 += einsum("ai,ijkb->jkab", l1.aa, x329)
    del x329
    x333 += einsum("ijab->ijab", x330)
    del x330
    l2new_aaaa += einsum("ijab->abij", x333)
    l2new_aaaa -= einsum("ijab->baij", x333)
    l2new_aaaa -= einsum("ijab->abji", x333)
    l2new_aaaa += einsum("ijab->baji", x333)
    del x333
    x334 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x334 += einsum("abij,kjli->klba", l2.aaaa, v.aaaa.oooo)
    x336 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x336 += einsum("ijab->jiba", x334)
    del x334
    x335 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x335 += einsum("abij,acbd->ijcd", l2.aaaa, v.aaaa.vvvv)
    x336 += einsum("ijab->jiba", x335)
    del x335
    l2new_aaaa += einsum("ijab->baij", x336) * -1
    l2new_aaaa += einsum("ijab->abij", x336)
    del x336
    x337 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x337 += einsum("ijab,kcjb->ikac", t2.abab, v.aabb.ovov)
    x339 += einsum("ijab->ijab", x337)
    x340 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x340 += einsum("ijab,ikca->kjcb", x339, x41)
    del x339
    x355 += einsum("ijab->ijab", x340) * -1
    del x340
    x449 += einsum("ijab->jiab", x337)
    del x337
    x349 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x349 += einsum("ijab,icjb->ac", t2.abab, v.aabb.ovov)
    x351 += einsum("ab->ab", x349)
    x352 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x352 += einsum("ab,acij->ijcb", x351, l2.aaaa)
    del x351
    x355 += einsum("ijab->jiab", x352)
    del x352
    l2new_aaaa += einsum("ijab->abij", x355) * -1
    l2new_aaaa += einsum("ijab->baij", x355)
    l2new_aaaa += einsum("ijab->abji", x355)
    l2new_aaaa += einsum("ijab->baji", x355) * -1
    del x355
    x469 += einsum("ab->ab", x349) * -1
    x517 += einsum("ab->ab", x349) * -1
    del x349
    x361 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x361 += einsum("wia,wbj->ijab", gc.bb.bov, lu11.bb)
    x403 += einsum("ijab->ijab", x361)
    del x361
    x362 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x362 += einsum("ai,jbac->ijbc", l1.bb, v.bbbb.ovvv)
    x403 -= einsum("ijab->ijab", x362)
    del x362
    x369 += einsum("iabj->jiba", v.bbbb.ovvo)
    x369 -= einsum("ijab->jiab", v.bbbb.oovv)
    x370 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x370 += einsum("ijab,ikac->jkbc", x224, x369)
    del x369
    x403 += einsum("ijab->ijab", x370)
    del x370
    x371 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x371 -= einsum("ijab->jiab", t2.aaaa)
    x371 += einsum("ijab->jiba", t2.aaaa)
    x372 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x372 += einsum("iajb,ikca->kjcb", v.aabb.ovov, x371)
    del x371
    x373 -= einsum("ijab->ijab", x372)
    del x372
    x373 += einsum("iabj->jiba", v.bbaa.ovvo)
    x374 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x374 += einsum("abij,ikac->jkbc", l2.abab, x373)
    del x373
    x403 += einsum("ijab->ijab", x374)
    del x374
    x376 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x376 += einsum("ijka->ikja", v.bbbb.ooov)
    x376 -= einsum("ijka->kija", v.bbbb.ooov)
    x377 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x377 += einsum("ijka,lkjb->ilab", x375, x376)
    del x375
    x403 += einsum("ijab->ijab", x377)
    del x377
    x388 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x388 += einsum("ia,ijka->jk", t1.bb, x376)
    x389 -= einsum("ij->ij", x388)
    del x388
    x390 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x390 += einsum("ij,abjk->kiab", x389, l2.bbbb)
    del x389
    x403 -= einsum("ijab->ijba", x390)
    del x390
    l2new_baba += einsum("ijka,lkjb->bali", x0, x376)
    del x376
    x378 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x378 += einsum("wia,wib->ab", g.bb.bov, u11.bb)
    x379 -= einsum("ab->ba", x378)
    x472 += einsum("ab->ba", x378) * -1
    x525 += einsum("ab->ba", x378) * -1
    del x378
    x379 += einsum("ab->ab", f.bb.vv)
    x380 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x380 += einsum("ab,acij->ijcb", x379, l2.bbbb)
    del x379
    x403 -= einsum("ijab->jiba", x380)
    del x380
    x384 += einsum("ia->ia", f.bb.ov)
    x385 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x385 += einsum("ia,ja->ij", t1.bb, x384)
    x386 += einsum("ij->ji", x385)
    del x385
    x398 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x398 += einsum("ia,jkib->jkba", x384, x177)
    del x177
    x403 += einsum("ijab->ijba", x398)
    del x398
    x403 += einsum("ai,jb->jiba", l1.bb, x384)
    lu11new_bb -= einsum("ia,wji->waj", x384, x174)
    del x384
    x386 += einsum("ij->ij", f.bb.oo)
    x387 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x387 += einsum("ij,abjk->kiab", x386, l2.bbbb)
    del x386
    x403 += einsum("ijab->jiba", x387)
    del x387
    x392 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x392 += einsum("iajb->jiab", v.bbbb.ovov)
    x392 -= einsum("iajb->jiba", v.bbbb.ovov)
    x393 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x393 += einsum("wia,ijab->wjb", u11.bb, x392)
    x394 -= einsum("wia->wia", x393)
    x395 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x395 += einsum("wai,wjb->ijab", lu11.bb, x394)
    del x394
    x403 += einsum("ijab->ijab", x395)
    del x395
    x486 -= einsum("wia->wia", x393)
    del x393
    x401 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x401 += einsum("ia,ijab->jb", t1.bb, x392)
    del x392
    x402 -= einsum("ia->ia", x401)
    x403 += einsum("ai,jb->ijab", l1.bb, x402)
    del x402
    x494 -= einsum("ia->ia", x401)
    del x401
    x399 -= einsum("ijka->jika", v.bbbb.ooov)
    x400 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x400 += einsum("ai,ijkb->jkab", l1.bb, x399)
    del x399
    x403 += einsum("ijab->ijab", x400)
    del x400
    l2new_bbbb += einsum("ijab->abij", x403)
    l2new_bbbb -= einsum("ijab->baij", x403)
    l2new_bbbb -= einsum("ijab->abji", x403)
    l2new_bbbb += einsum("ijab->baji", x403)
    del x403
    x418 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x418 += einsum("ijab,iajc->bc", t2.abab, v.aabb.ovov)
    x420 += einsum("ab->ab", x418)
    x421 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x421 += einsum("ab,acij->ijcb", x420, l2.bbbb)
    del x420
    x426 += einsum("ijab->jiab", x421)
    del x421
    l2new_bbbb += einsum("ijab->abij", x426) * -1
    l2new_bbbb += einsum("ijab->baij", x426)
    l2new_bbbb += einsum("ijab->abji", x426)
    l2new_bbbb += einsum("ijab->baji", x426) * -1
    del x426
    x472 += einsum("ab->ab", x418) * -1
    x525 += einsum("ab->ab", x418) * -1
    del x418
    x427 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x427 += einsum("abij,kilj->klab", l2.bbbb, v.bbbb.oooo)
    x429 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x429 += einsum("ijab->jiba", x427)
    del x427
    x428 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x428 += einsum("abij,acbd->ijcd", l2.bbbb, v.bbbb.vvvv)
    x429 += einsum("ijab->jiba", x428)
    del x428
    l2new_bbbb += einsum("ijab->baij", x429) * -1
    l2new_bbbb += einsum("ijab->abij", x429)
    del x429
    x431 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x431 += einsum("ai,jbac->jibc", l1.bb, v.aabb.ovvv)
    l2new_baba += einsum("ijab->baji", x431)
    l2new_abab += einsum("ijab->abij", x431)
    del x431
    x432 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x432 += einsum("abij,acbd->ijcd", l2.abab, v.aabb.vvvv)
    l2new_baba += einsum("ijab->baji", x432)
    l2new_abab += einsum("ijab->abij", x432)
    del x432
    x434 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x434 += einsum("ai,jbac->ijcb", l1.aa, v.bbaa.ovvv)
    l2new_baba += einsum("ijab->baji", x434)
    l2new_abab += einsum("ijab->abij", x434)
    del x434
    x437 += einsum("iabj->ijab", v.aabb.ovvo)
    x438 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x438 += einsum("ijab,kica->kjcb", x270, x437)
    del x437
    del x270
    l2new_baba += einsum("ijab->baji", x438) * -1
    l2new_abab += einsum("ijab->abij", x438) * -1
    del x438
    x441 += einsum("iabj->jiba", v.bbaa.ovvo)
    l2new_baba += einsum("ijab,ikbc->cakj", x200, x441) * -1
    del x200
    l2new_abab += einsum("ijab,ikbc->acjk", x203, x441) * -1
    del x441
    del x203
    x442 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x442 += einsum("ijab,iakc->jkbc", t2.abab, v.aabb.ovov)
    x445 += einsum("ijab->jiab", x442)
    del x442
    x445 += einsum("iabj->ijba", v.bbbb.ovvo)
    x445 += einsum("ijab->ijab", v.bbbb.oovv) * -1
    x446 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x446 += einsum("abij,kjbc->ikac", l2.abab, x445)
    del x445
    l2new_baba += einsum("ijab->baji", x446)
    l2new_abab += einsum("ijab->abij", x446)
    del x446
    x449 += einsum("iabj->ijba", v.aaaa.ovvo)
    x449 += einsum("ijab->ijab", v.aaaa.oovv) * -1
    x450 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x450 += einsum("abij,kiac->kjcb", l2.abab, x449)
    del x449
    l2new_baba += einsum("ijab->baji", x450)
    l2new_abab += einsum("ijab->abij", x450)
    del x450
    x451 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x451 += einsum("ia,jabc->ijbc", t1.aa, v.aabb.ovvv)
    x453 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x453 += einsum("ijab->jiab", x451)
    del x451
    x452 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x452 += einsum("ijab,kajc->ikbc", t2.abab, v.aabb.ovov)
    x453 += einsum("ijab->jiab", x452) * -1
    del x452
    x453 += einsum("ijab->ijab", v.aabb.oovv)
    x454 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x454 += einsum("abij,kibc->kjac", l2.abab, x453)
    del x453
    l2new_baba += einsum("ijab->baji", x454) * -1
    l2new_abab += einsum("ijab->abij", x454) * -1
    del x454
    x455 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x455 += einsum("ijab,ickb->jkac", t2.abab, v.aabb.ovov)
    x456 += einsum("ijab->jiab", x455) * -1
    del x455
    x456 += einsum("ijab->ijab", v.bbaa.oovv)
    x457 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x457 += einsum("abij,kjac->ikcb", l2.abab, x456)
    del x456
    l2new_baba += einsum("ijab->baji", x457) * -1
    l2new_abab += einsum("ijab->abij", x457) * -1
    del x457
    x459 += einsum("ijkl->ijkl", v.aabb.oooo)
    x460 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x460 += einsum("abij,kilj->klab", l2.abab, x459)
    del x459
    l2new_baba += einsum("ijab->baji", x460)
    l2new_abab += einsum("ijab->abij", x460)
    del x460
    x467 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x467 += einsum("ijka->ikja", v.aaaa.ooov)
    x467 -= einsum("ijka->kija", v.aaaa.ooov)
    l2new_baba -= einsum("ijka,kilb->balj", x467, x53)
    del x53
    lu11new_aa -= einsum("wij,jkia->wak", x157, x467)
    del x467
    del x157
    x469 += einsum("ab->ab", f.aa.vv)
    x470 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x470 += einsum("ab,acij->ijbc", x469, l2.abab)
    del x469
    l2new_baba += einsum("ijab->baji", x470)
    l2new_abab += einsum("ijab->abij", x470)
    del x470
    x472 += einsum("ab->ab", f.bb.vv)
    x473 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x473 += einsum("ab,caij->ijcb", x472, l2.abab)
    del x472
    l2new_baba += einsum("ijab->baji", x473)
    l2new_abab += einsum("ijab->abij", x473)
    del x473
    x484 += einsum("wia->wia", gc.aa.bov)
    x485 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x485 += einsum("wai,wjb->jiba", lu11.bb, x484)
    del x484
    l2new_baba += einsum("ijab->baji", x485)
    l2new_abab += einsum("ijab->abij", x485)
    del x485
    x486 += einsum("wia->wia", gc.bb.bov)
    x487 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x487 += einsum("wai,wjb->ijab", lu11.aa, x486)
    del x486
    l2new_baba += einsum("ijab->baji", x487)
    l2new_abab += einsum("ijab->abij", x487)
    del x487
    x490 += einsum("iajk->ijka", v.aabb.ovoo)
    x491 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x491 += einsum("ai,jkib->jkba", l1.bb, x490)
    del x490
    l2new_baba -= einsum("ijab->baji", x491)
    l2new_abab -= einsum("ijab->abij", x491)
    del x491
    x492 += einsum("ijka->ijka", v.aabb.ooov)
    x493 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x493 += einsum("ai,jikb->jkab", l1.aa, x492)
    del x492
    l2new_baba -= einsum("ijab->baji", x493)
    l2new_abab -= einsum("ijab->abij", x493)
    del x493
    x494 += einsum("ia->ia", f.bb.ov)
    l2new_baba += einsum("ai,jb->baji", l1.aa, x494)
    l2new_abab += einsum("ai,jb->abij", l1.aa, x494)
    del x494
    x495 += einsum("ia->ia", f.aa.ov)
    l2new_baba += einsum("ai,jb->abij", l1.bb, x495)
    l2new_abab += einsum("ai,jb->baji", l1.bb, x495)
    del x495
    x496 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x496 -= einsum("ijka->ikja", v.bbbb.ooov)
    x496 += einsum("ijka->kija", v.bbbb.ooov)
    l2new_abab -= einsum("ijka,lkjb->abil", x0, x496)
    del x0
    lu11new_bb -= einsum("wij,kjia->wak", x174, x496)
    del x496
    del x174
    x497 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x497 += einsum("ia,wbi->wba", t1.bb, lu11.bb)
    lu11new_aa += einsum("wab,icab->wci", x497, v.aabb.ovvv)
    lu11new_bb -= einsum("wab,iacb->wci", x497, x277)
    del x497
    del x277
    x498 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x498 += einsum("ia,wba->wib", t1.aa, g.aa.bvv)
    x502 += einsum("wia->wia", x498)
    del x498
    x499 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x499 += einsum("wia,jiba->wjb", g.bb.bov, t2.abab)
    x502 += einsum("wia->wia", x499)
    del x499
    x502 += einsum("wai->wia", g.aa.bvo)
    lu11new_aa += einsum("wia,ijba->wbj", x502, x41)
    del x41
    lu11new_bb += einsum("wia,abij->wbj", x502, l2.abab)
    del x502
    x504 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x504 += einsum("wai,jiba->wjb", lu11.bb, t2.abab)
    x506 += einsum("wia->wia", x504)
    del x504
    lu11new_aa += einsum("wia,ijba->wbj", x506, x13) * -1
    del x13
    lu11new_bb += einsum("wia,iajb->wbj", x506, v.aabb.ovov)
    del x506
    x507 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x507 += einsum("ia,wba->wib", t1.bb, g.bb.bvv)
    x511 += einsum("wia->wia", x507)
    del x507
    x508 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x508 += einsum("wia,ijab->wjb", g.aa.bov, t2.abab)
    x511 += einsum("wia->wia", x508)
    del x508
    x511 += einsum("wai->wia", g.bb.bvo)
    lu11new_aa += einsum("wia,baji->wbj", x511, l2.abab)
    lu11new_bb += einsum("wia,ijab->wbj", x511, x224)
    del x224
    del x511
    x512 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x512 += einsum("wai,ijab->wjb", lu11.aa, t2.abab)
    x515 += einsum("wia->wia", x512)
    del x512
    lu11new_aa += einsum("wia,jbia->wbj", x515, v.aabb.ovov)
    lu11new_bb += einsum("wia,ijab->wbj", x515, x89) * -1
    del x89
    del x515
    x516 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x516 += einsum("ia,wbi->wba", t1.aa, lu11.aa)
    lu11new_aa -= einsum("wab,iacb->wci", x516, x298)
    del x298
    lu11new_bb += einsum("wab,icab->wci", x516, v.bbaa.ovvv)
    del x516
    x517 += einsum("ab->ab", f.aa.vv)
    lu11new_aa += einsum("ab,wai->wbi", x517, lu11.aa)
    del x517
    x518 = np.zeros((nbos, nbos), dtype=types[float])
    x518 += einsum("wia,xia->wx", g.aa.bov, u11.aa)
    x520 = np.zeros((nbos, nbos), dtype=types[float])
    x520 += einsum("wx->wx", x518)
    del x518
    x519 = np.zeros((nbos, nbos), dtype=types[float])
    x519 += einsum("wia,xia->wx", g.bb.bov, u11.bb)
    x520 += einsum("wx->wx", x519)
    del x519
    x520 += einsum("wx->wx", w)
    lu11new_aa += einsum("wx,xai->wai", x520, lu11.aa)
    lu11new_bb += einsum("wx,xai->wai", x520, lu11.bb)
    del x520
    x521 += einsum("wij->wij", g.aa.boo)
    lu11new_aa -= einsum("ai,wji->waj", l1.aa, x521)
    del x521
    x522 = np.zeros((nbos, nbos), dtype=types[float])
    x522 += einsum("wai,xia->wx", lu11.aa, u11.aa)
    x524 = np.zeros((nbos, nbos), dtype=types[float])
    x524 += einsum("wx->wx", x522)
    del x522
    x523 = np.zeros((nbos, nbos), dtype=types[float])
    x523 += einsum("wai,xia->wx", lu11.bb, u11.bb)
    x524 += einsum("wx->wx", x523)
    del x523
    lu11new_aa += einsum("wx,xia->wai", x524, g.aa.bov)
    lu11new_bb += einsum("wx,xia->wai", x524, g.bb.bov)
    del x524
    x525 += einsum("ab->ab", f.bb.vv)
    lu11new_bb += einsum("ab,wai->wbi", x525, lu11.bb)
    del x525
    x526 += einsum("wij->wij", g.bb.boo)
    lu11new_bb -= einsum("ai,wji->waj", l1.bb, x526)
    del x526
    l1new_aa += einsum("w,wia->ai", ls1, gc.aa.bov)
    l1new_aa += einsum("ia->ai", f.aa.ov)
    l1new_aa += einsum("ai,jbai->bj", l1.bb, v.aabb.ovvo)
    l1new_bb += einsum("w,wia->ai", ls1, gc.bb.bov)
    l1new_bb += einsum("ia->ai", f.bb.ov)
    l1new_bb += einsum("ai,jbai->bj", l1.aa, v.bbaa.ovvo)
    l2new_aaaa -= einsum("iajb->abji", v.aaaa.ovov)
    l2new_aaaa += einsum("iajb->baji", v.aaaa.ovov)
    l2new_bbbb -= einsum("iajb->abji", v.bbbb.ovov)
    l2new_bbbb += einsum("iajb->baji", v.bbbb.ovov)
    l2new_baba += einsum("iajb->baji", v.aabb.ovov)
    l2new_abab += einsum("iajb->abij", v.aabb.ovov)
    ls1new += einsum("w,xw->x", ls1, w)
    ls1new += einsum("w->w", G)
    ls1new += einsum("ai,wai->w", l1.bb, g.bb.bvo)
    ls1new += einsum("ai,wai->w", l1.aa, g.aa.bvo)
    lu11new_aa += einsum("wia->wai", g.aa.bov)
    lu11new_aa += einsum("wai,jbai->wbj", lu11.bb, v.aabb.ovvo)
    lu11new_aa += einsum("ai,wab->wbi", l1.aa, g.aa.bvv)
    lu11new_bb += einsum("ai,wab->wbi", l1.bb, g.bb.bvv)
    lu11new_bb += einsum("wia->wai", g.bb.bov)
    lu11new_bb += einsum("wai,jbai->wbj", lu11.aa, v.bbaa.ovvo)

    l1new.aa = l1new_aa
    l1new.bb = l1new_bb
    l2new.abab = l2new_abab
    l2new.baba = l2new_baba
    l2new.aaaa = l2new_aaaa
    l2new.bbbb = l2new_bbbb
    lu11new.aa = lu11new_aa
    lu11new.bb = lu11new_bb

    return {"l1new": l1new, "l2new": l2new, "ls1new": ls1new, "lu11new": lu11new}

def make_rdm1_f(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, l1=None, l2=None, ls1=None, lu11=None, **kwargs):
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
    x0 += einsum("wai,wja->ij", lu11.aa, u11.aa)
    x14 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x14 += einsum("ij->ij", x0)
    rdm1_f_oo_aa = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    rdm1_f_oo_aa -= einsum("ij->ij", x0)
    del x0
    x1 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x1 += einsum("abij,kjab->ik", l2.abab, t2.abab)
    x14 += einsum("ij->ij", x1)
    rdm1_f_oo_aa += einsum("ij->ij", x1) * -1
    del x1
    x2 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x2 += einsum("ai,ja->ij", l1.aa, t1.aa)
    x14 += einsum("ij->ij", x2)
    rdm1_f_oo_aa -= einsum("ij->ij", x2)
    del x2
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x3 += einsum("ijab->jiab", t2.aaaa)
    x3 += einsum("ijab->jiba", t2.aaaa) * -1
    rdm1_f_oo_aa += einsum("abij,ikba->jk", l2.aaaa, x3) * -1
    del x3
    x4 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x4 += einsum("abij,ikab->jk", l2.abab, t2.abab)
    x21 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x21 += einsum("ij->ij", x4)
    rdm1_f_oo_bb = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    rdm1_f_oo_bb += einsum("ij->ij", x4) * -1
    del x4
    x5 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x5 += einsum("wai,wja->ij", lu11.bb, u11.bb)
    x21 += einsum("ij->ij", x5)
    rdm1_f_oo_bb -= einsum("ij->ij", x5)
    del x5
    x6 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x6 += einsum("ai,ja->ij", l1.bb, t1.bb)
    x21 += einsum("ij->ij", x6)
    rdm1_f_oo_bb -= einsum("ij->ij", x6)
    del x6
    x7 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x7 += einsum("ijab->jiab", t2.bbbb) * -1
    x7 += einsum("ijab->jiba", t2.bbbb)
    rdm1_f_oo_bb += einsum("abij,ikab->jk", l2.bbbb, x7) * -1
    del x7
    x8 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x8 += einsum("ia,waj->wji", t1.aa, lu11.aa)
    rdm1_f_vo_aa = np.zeros((nvir[0], nocc[0]), dtype=types[float])
    rdm1_f_vo_aa -= einsum("wia,wij->aj", u11.aa, x8)
    del x8
    x9 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x9 += einsum("ia,abjk->jikb", t1.aa, l2.abab)
    rdm1_f_vo_aa += einsum("ijab,ikjb->ak", t2.abab, x9) * -1
    del x9
    x10 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x10 += einsum("ia,abjk->kjib", t1.aa, l2.aaaa)
    x11 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x11 += einsum("ijka->ijka", x10) * -1
    x11 += einsum("ijka->jika", x10)
    del x10
    rdm1_f_vo_aa += einsum("ijab,jika->bk", t2.aaaa, x11) * -1
    del x11
    x12 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum("ijab->jiab", t2.aaaa)
    x12 -= einsum("ijab->jiba", t2.aaaa)
    rdm1_f_vo_aa -= einsum("ai,ijab->bj", l1.aa, x12)
    del x12
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x13 += einsum("ijab->jiab", t2.aaaa) * -1
    x13 += einsum("ijab->jiba", t2.aaaa)
    x14 += einsum("abij,ikba->jk", l2.aaaa, x13) * -1
    del x13
    rdm1_f_vo_aa += einsum("ia,ij->aj", t1.aa, x14) * -1
    del x14
    x15 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x15 += einsum("ia,waj->wji", t1.bb, lu11.bb)
    rdm1_f_vo_bb = np.zeros((nvir[1], nocc[1]), dtype=types[float])
    rdm1_f_vo_bb -= einsum("wia,wij->aj", u11.bb, x15)
    del x15
    x16 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x16 += einsum("ia,bajk->jkib", t1.bb, l2.abab)
    rdm1_f_vo_bb += einsum("ijab,ijka->bk", t2.abab, x16) * -1
    del x16
    x17 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x17 += einsum("ia,bajk->jkib", t1.bb, l2.bbbb)
    x18 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x18 += einsum("ijka->ijka", x17)
    x18 += einsum("ijka->jika", x17) * -1
    del x17
    rdm1_f_vo_bb += einsum("ijab,jikb->ak", t2.bbbb, x18) * -1
    del x18
    x19 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x19 += einsum("ijab->jiab", t2.bbbb)
    x19 -= einsum("ijab->jiba", t2.bbbb)
    rdm1_f_vo_bb -= einsum("ai,ijab->bj", l1.bb, x19)
    del x19
    x20 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x20 += einsum("ijab->jiab", t2.bbbb)
    x20 += einsum("ijab->jiba", t2.bbbb) * -1
    x21 += einsum("abij,ikab->jk", l2.bbbb, x20) * -1
    del x20
    rdm1_f_vo_bb += einsum("ia,ij->aj", t1.bb, x21) * -1
    del x21
    x22 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x22 += einsum("abij->jiab", l2.aaaa) * -1
    x22 += einsum("abij->jiba", l2.aaaa)
    rdm1_f_vv_aa = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    rdm1_f_vv_aa += einsum("ijab,ijca->bc", t2.aaaa, x22) * -1
    del x22
    x23 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x23 += einsum("abij->jiab", l2.bbbb) * -1
    x23 += einsum("abij->jiba", l2.bbbb)
    rdm1_f_vv_bb = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    rdm1_f_vv_bb += einsum("ijab,ijbc->ac", t2.bbbb, x23) * -1
    del x23
    rdm1_f_oo_aa += einsum("ij->ji", delta_oo.aa)
    rdm1_f_oo_bb += einsum("ij->ji", delta_oo.bb)
    rdm1_f_ov_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    rdm1_f_ov_aa += einsum("ai->ia", l1.aa)
    rdm1_f_ov_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    rdm1_f_ov_bb += einsum("ai->ia", l1.bb)
    rdm1_f_vo_aa += einsum("ai,jiba->bj", l1.bb, t2.abab)
    rdm1_f_vo_aa += einsum("w,wia->ai", ls1, u11.aa)
    rdm1_f_vo_aa += einsum("ia->ai", t1.aa)
    rdm1_f_vo_bb += einsum("ai,ijab->bj", l1.aa, t2.abab)
    rdm1_f_vo_bb += einsum("w,wia->ai", ls1, u11.bb)
    rdm1_f_vo_bb += einsum("ia->ai", t1.bb)
    rdm1_f_vv_aa += einsum("abij,ijcb->ca", l2.abab, t2.abab)
    rdm1_f_vv_aa += einsum("wai,wib->ba", lu11.aa, u11.aa)
    rdm1_f_vv_aa += einsum("ai,ib->ba", l1.aa, t1.aa)
    rdm1_f_vv_bb += einsum("ai,ib->ba", l1.bb, t1.bb)
    rdm1_f_vv_bb += einsum("wai,wib->ba", lu11.bb, u11.bb)
    rdm1_f_vv_bb += einsum("abij,ijac->cb", l2.abab, t2.abab)

    rdm1_f_aa = np.block([[rdm1_f_oo_aa, rdm1_f_ov_aa], [rdm1_f_vo_aa, rdm1_f_vv_aa]])
    rdm1_f_bb = np.block([[rdm1_f_oo_bb, rdm1_f_ov_bb], [rdm1_f_vo_bb, rdm1_f_vv_bb]])

    rdm1_f.aa = rdm1_f_aa
    rdm1_f.bb = rdm1_f_bb

    return rdm1_f

def make_rdm2_f(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, l1=None, l2=None, ls1=None, lu11=None, **kwargs):
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
    x0 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x0 += einsum("wai,wja->ij", lu11.aa, u11.aa)
    x18 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x18 += einsum("ij->ij", x0)
    x29 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x29 += einsum("ia,ij->ja", t1.aa, x0)
    x32 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x32 += einsum("ia->ia", x29)
    del x29
    x33 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x33 += einsum("ij->ij", x0)
    x96 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x96 += einsum("ij->ji", x0)
    x145 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x145 += einsum("ij,kiab->kjab", x0, t2.aaaa)
    x161 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x161 -= einsum("ijab->ijab", x145)
    del x145
    x248 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x248 += einsum("ij->ij", x0)
    rdm2_f_oooo_aaaa = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_oooo_aaaa -= einsum("ij,kl->ijkl", delta_oo.aa, x0)
    rdm2_f_oooo_aaaa += einsum("ij,kl->ilkj", delta_oo.aa, x0)
    rdm2_f_oooo_aaaa += einsum("ij,kl->kijl", delta_oo.aa, x0)
    rdm2_f_oooo_aaaa -= einsum("ij,kl->klji", delta_oo.aa, x0)
    del x0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x1 += einsum("abij,klba->ijlk", l2.aaaa, t2.aaaa)
    x39 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x39 += einsum("ijkl->jilk", x1)
    x174 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x174 += einsum("ijab,ijkl->klab", t2.aaaa, x1)
    x178 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x178 += einsum("ijab->ijab", x174)
    del x174
    x176 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x176 += einsum("ia,jikl->jkla", t1.aa, x1)
    x177 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x177 += einsum("ia,ijkb->jkab", t1.aa, x176)
    del x176
    x178 += einsum("ijab->ijab", x177)
    del x177
    rdm2_f_oooo_aaaa += einsum("ijkl->jkil", x1) * -1
    rdm2_f_oooo_aaaa += einsum("ijkl->jlik", x1)
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x2 += einsum("ia,abjk->kjib", t1.aa, l2.aaaa)
    x3 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x3 += einsum("ia,jkla->jkil", t1.aa, x2)
    x39 += einsum("ijkl->ijkl", x3)
    x40 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x40 += einsum("ia,ijkl->jkla", t1.aa, x39)
    del x39
    x46 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x46 += einsum("ijka->ikja", x40)
    x95 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x95 += einsum("ijka->ikja", x40)
    del x40
    x162 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x162 += einsum("ia,ijkl->jlka", t1.aa, x3)
    x163 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x163 += einsum("ia,ijkb->jkab", t1.aa, x162)
    del x162
    x169 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x169 += einsum("ijab->ijab", x163)
    del x163
    x175 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x175 += einsum("ijab,jikl->klba", t2.aaaa, x3)
    x178 += einsum("ijab->ijab", x175)
    del x175
    rdm2_f_vovo_aaaa = np.zeros((nvir[0], nocc[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_vovo_aaaa += einsum("ijab->ajbi", x178) * -1
    rdm2_f_vovo_aaaa += einsum("ijab->bjai", x178)
    del x178
    rdm2_f_oooo_aaaa += einsum("ijkl->ikjl", x3)
    rdm2_f_oooo_aaaa += einsum("ijkl->iljk", x3) * -1
    del x3
    x36 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x36 += einsum("ijka->ijka", x2)
    x36 += einsum("ijka->jika", x2) * -1
    x42 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x42 += einsum("ijka->ijka", x2) * -1
    x42 += einsum("ijka->jika", x2)
    x43 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x43 += einsum("ijab,jikb->ka", t2.aaaa, x42)
    x45 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x45 += einsum("ia->ia", x43) * -1
    x93 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x93 += einsum("ia->ia", x43) * -1
    x249 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x249 += einsum("ia->ia", x43) * -1
    del x43
    x77 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x77 += einsum("ijab,kila->kljb", t2.abab, x42)
    x234 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x234 += einsum("ijka->ijka", x77) * -1
    rdm2_f_oovo_aabb = np.zeros((nocc[0], nocc[0], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x77) * -1
    rdm2_f_vooo_bbaa = np.zeros((nvir[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x77) * -1
    del x77
    x251 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x251 += einsum("ijab,jikb->ka", t2.aaaa, x42) * -1
    del x42
    x104 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x104 += einsum("ijka->ijka", x2)
    x104 -= einsum("ijka->jika", x2)
    x105 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x105 += einsum("ia,ijkb->jkab", t1.aa, x104)
    rdm2_f_oovv_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_oovv_aaaa -= einsum("ijab->ijab", x105)
    rdm2_f_vvoo_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_vvoo_aaaa -= einsum("ijab->abij", x105)
    del x105
    x130 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x130 -= einsum("ijka->ijka", x2)
    x130 += einsum("ijka->jika", x2)
    x131 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x131 += einsum("ia,ijkb->jkab", t1.aa, x130)
    del x130
    rdm2_f_ovvo_aaaa = np.zeros((nocc[0], nvir[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_ovvo_aaaa -= einsum("ijab->ibaj", x131)
    rdm2_f_voov_aaaa = np.zeros((nvir[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_voov_aaaa -= einsum("ijab->ajib", x131)
    del x131
    x261 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x261 += einsum("ijab,jikc->kcba", t2.aaaa, x2)
    x266 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x266 += einsum("iabc->iabc", x261)
    del x261
    x262 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x262 += einsum("ia,jikb->jkba", t1.aa, x2)
    x264 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x264 += einsum("ijab->ijab", x262) * -1
    del x262
    rdm2_f_ooov_aaaa = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_ooov_aaaa += einsum("ijka->ikja", x2)
    rdm2_f_ooov_aaaa -= einsum("ijka->jkia", x2)
    rdm2_f_ovoo_aaaa = np.zeros((nocc[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_ovoo_aaaa -= einsum("ijka->iajk", x2)
    rdm2_f_ovoo_aaaa += einsum("ijka->jaik", x2)
    del x2
    x4 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum("abij,kjab->ik", l2.abab, t2.abab)
    x7 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x7 += einsum("ij->ij", x4)
    x18 += einsum("ij->ij", x4)
    x96 += einsum("ij->ji", x4)
    x248 += einsum("ij->ij", x4)
    del x4
    x5 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x5 += einsum("ijab->jiab", t2.aaaa) * -1
    x5 += einsum("ijab->jiba", t2.aaaa)
    x6 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x6 += einsum("abij,ikba->jk", l2.aaaa, x5)
    x7 += einsum("ij->ij", x6) * -1
    x44 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x44 += einsum("ia,ij->ja", t1.aa, x7)
    x45 += einsum("ia->ia", x44)
    del x44
    x46 += einsum("ia,jk->jika", t1.aa, x7)
    x95 += einsum("ia,jk->jika", t1.aa, x7)
    x172 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x172 += einsum("ij,ikab->kjab", x7, t2.aaaa)
    x173 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x173 += einsum("ijab->ijba", x172)
    del x172
    rdm2_f_oooo_aaaa += einsum("ij,kl->jikl", delta_oo.aa, x7) * -1
    rdm2_f_oooo_aaaa += einsum("ij,kl->jlki", delta_oo.aa, x7)
    rdm2_f_oooo_aaaa += einsum("ij,kl->kjil", delta_oo.aa, x7)
    rdm2_f_oooo_aaaa += einsum("ij,kl->klij", delta_oo.aa, x7) * -1
    del x7
    x18 += einsum("ij->ij", x6) * -1
    del x6
    x91 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x91 += einsum("ai,ijab->jb", l1.aa, x5)
    x93 += einsum("ia->ia", x91) * -1
    x251 += einsum("ia->ia", x91) * -1
    del x91
    x247 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x247 += einsum("abij,ikba->jk", l2.aaaa, x5)
    x248 += einsum("ij->ij", x247) * -1
    del x247
    x249 += einsum("ai,ijab->jb", l1.aa, x5) * -0.9999999999999993
    x292 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x292 += einsum("abij,ikac->kjcb", l2.abab, x5)
    x294 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x294 += einsum("ijab->ijab", x292) * -1
    del x292
    x8 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x8 += einsum("ai,ja->ij", l1.aa, t1.aa)
    x18 += einsum("ij->ij", x8)
    x92 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x92 += einsum("ia,ij->ja", t1.aa, x18)
    x93 += einsum("ia->ia", x92)
    del x92
    x240 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x240 += einsum("ij,ikab->jkab", x18, t2.abab)
    rdm2_f_vovo_bbaa = np.zeros((nvir[1], nocc[1], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x240) * -1
    rdm2_f_vovo_aabb = np.zeros((nvir[0], nocc[0], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x240) * -1
    del x240
    rdm2_f_oooo_aabb = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_oooo_aabb += einsum("ij,kl->klji", delta_oo.bb, x18) * -1
    rdm2_f_oooo_bbaa = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_oooo_bbaa += einsum("ij,kl->jikl", delta_oo.bb, x18) * -1
    rdm2_f_oovo_aabb += einsum("ia,jk->jkai", t1.bb, x18) * -1
    del x18
    x33 += einsum("ij->ij", x8)
    x34 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x34 += einsum("ia,jk->jika", t1.aa, x33)
    x94 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x94 += einsum("ia,jk->jika", t1.aa, x33)
    x179 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x179 += einsum("ia,ij->ja", t1.aa, x33)
    del x33
    x180 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x180 += einsum("ia->ia", x179)
    x181 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x181 += einsum("ia->ia", x179)
    del x179
    x48 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x48 += einsum("ia,ij->ja", t1.aa, x8)
    x49 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x49 -= einsum("ia->ia", x48)
    del x48
    x96 += einsum("ij->ji", x8)
    x143 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x143 += einsum("ij,kiab->jkab", x8, t2.aaaa)
    x161 += einsum("ijab->ijab", x143)
    del x143
    x248 += einsum("ij->ij", x8)
    x249 += einsum("ia,ij->ja", t1.aa, x248) * 0.9999999999999993
    x251 += einsum("ia,ij->ja", t1.aa, x248)
    del x248
    rdm2_f_oooo_aaaa -= einsum("ij,kl->jikl", delta_oo.aa, x8)
    rdm2_f_oooo_aaaa += einsum("ij,kl->kijl", delta_oo.aa, x8)
    rdm2_f_oooo_aaaa += einsum("ij,kl->jlki", delta_oo.aa, x8)
    rdm2_f_oooo_aaaa -= einsum("ij,kl->klji", delta_oo.aa, x8)
    del x8
    x9 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x9 += einsum("abij,klab->ikjl", l2.abab, t2.abab)
    x79 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x79 += einsum("ijkl->ijkl", x9)
    x226 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x226 += einsum("ijkl->ijkl", x9)
    x232 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x232 += einsum("ijkl->ijkl", x9)
    rdm2_f_oooo_aabb += einsum("ijkl->ijkl", x9)
    rdm2_f_oooo_bbaa += einsum("ijkl->klij", x9)
    del x9
    x10 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x10 += einsum("ia,bajk->jkib", t1.bb, l2.abab)
    x11 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x11 += einsum("ia,jkla->jikl", t1.aa, x10)
    x79 += einsum("ijkl->ijkl", x11)
    x80 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x80 += einsum("ia,jkil->jkla", t1.bb, x79)
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x80)
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x80)
    del x80
    x90 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x90 += einsum("ia,ijkl->jkla", t1.aa, x79)
    del x79
    rdm2_f_oovo_bbaa = np.zeros((nocc[1], nocc[1], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x90)
    rdm2_f_vooo_aabb = np.zeros((nvir[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x90)
    del x90
    x226 += einsum("ijkl->ijkl", x11)
    x227 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x227 += einsum("ijab,ikjl->klab", t2.abab, x226)
    del x226
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x227)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x227)
    del x227
    x232 += einsum("ijkl->ijkl", x11) * 0.9999999999999993
    x233 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x233 += einsum("ia,jkil->jkla", t1.bb, x232)
    del x232
    x234 += einsum("ijka->ijka", x233)
    del x233
    rdm2_f_oooo_aabb += einsum("ijkl->ijkl", x11)
    rdm2_f_oooo_bbaa += einsum("ijkl->klij", x11)
    del x11
    x63 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x63 += einsum("ijab,ikla->kljb", t2.abab, x10)
    x72 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x72 += einsum("ijka->ijka", x63) * -1
    x99 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x99 += einsum("ijka->ijka", x63) * -1
    x190 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x190 -= einsum("ijka->ijka", x63)
    del x63
    x68 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x68 += einsum("ijab,ijka->kb", t2.abab, x10)
    x71 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x71 += einsum("ia->ia", x68)
    x84 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x84 += einsum("ia->ia", x68)
    x97 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x97 += einsum("ia->ia", x68)
    x246 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x246 += einsum("ia->ia", x68)
    x250 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x250 += einsum("ia->ia", x68)
    del x68
    x75 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x75 += einsum("ijab,kjla->kilb", t2.abab, x10)
    x234 += einsum("ijka->ijka", x75)
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x75)
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x75)
    del x75
    x89 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x89 += einsum("ijka,ilab->ljkb", x10, x5)
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x89) * -1
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x89) * -1
    del x89
    x126 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x126 += einsum("ia,ijkb->jkba", t1.aa, x10)
    rdm2_f_oovv_bbaa = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_oovv_bbaa -= einsum("ijab->ijba", x126)
    rdm2_f_vvoo_aabb = np.zeros((nvir[0], nvir[0], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_vvoo_aabb -= einsum("ijab->baij", x126)
    del x126
    x132 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x132 += einsum("ia,jikb->jkba", t1.bb, x10)
    x286 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x286 += einsum("ijab->ijab", x132)
    rdm2_f_ovvo_aabb = np.zeros((nocc[0], nvir[0], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_ovvo_aabb -= einsum("ijab->iabj", x132)
    rdm2_f_voov_bbaa = np.zeros((nvir[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_voov_bbaa -= einsum("ijab->bjia", x132)
    del x132
    x280 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x280 += einsum("ijab,ijkc->kcab", t2.abab, x10)
    rdm2_f_vovv_bbaa = np.zeros((nvir[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_vovv_bbaa += einsum("iabc->ciba", x280) * -1
    rdm2_f_vvvo_aabb = np.zeros((nvir[0], nvir[0], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_vvvo_aabb += einsum("iabc->baci", x280) * -1
    del x280
    rdm2_f_ooov_bbaa = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_ooov_bbaa -= einsum("ijka->jkia", x10)
    rdm2_f_ovoo_aabb = np.zeros((nocc[0], nvir[0], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_ovoo_aabb -= einsum("ijka->iajk", x10)
    x12 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x12 += einsum("ai,ja->ij", l1.bb, t1.bb)
    x17 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x17 += einsum("ij->ji", x12)
    x51 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x51 += einsum("ia,ij->ja", t1.bb, x12)
    x73 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x73 -= einsum("ia->ia", x51)
    rdm2_f_oovo_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_oovo_bbbb += einsum("ij,ka->ikaj", delta_oo.bb, x51)
    rdm2_f_vooo_bbbb = np.zeros((nvir[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_vooo_bbbb -= einsum("ij,ka->akji", delta_oo.bb, x51)
    del x51
    x61 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x61 += einsum("ij->ij", x12)
    x82 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x82 += einsum("ij->ij", x12)
    x182 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x182 += einsum("ij,kiab->jkab", x12, t2.bbbb)
    x199 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x199 += einsum("ijab->ijab", x182)
    del x182
    x245 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x245 += einsum("ij->ij", x12)
    rdm2_f_oooo_bbbb = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_oooo_bbbb -= einsum("ij,kl->jikl", delta_oo.bb, x12)
    rdm2_f_oooo_bbbb += einsum("ij,kl->kijl", delta_oo.bb, x12)
    rdm2_f_oooo_bbbb += einsum("ij,kl->jlki", delta_oo.bb, x12)
    rdm2_f_oooo_bbbb -= einsum("ij,kl->klji", delta_oo.bb, x12)
    del x12
    x13 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x13 += einsum("wai,wja->ij", lu11.bb, u11.bb)
    x17 += einsum("ij->ji", x13)
    x57 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x57 += einsum("ia,ij->ja", t1.bb, x13)
    x60 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x60 += einsum("ia->ia", x57)
    del x57
    x61 += einsum("ij->ij", x13)
    x62 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x62 += einsum("ia,jk->jika", t1.bb, x61)
    x98 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x98 += einsum("ia,jk->jika", t1.bb, x61)
    x217 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x217 += einsum("ia,ij->ja", t1.bb, x61)
    del x61
    x218 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x218 += einsum("ia->ia", x217)
    x219 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x219 += einsum("ia->ia", x217)
    del x217
    x82 += einsum("ij->ij", x13)
    x184 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x184 += einsum("ij,kiab->kjab", x13, t2.bbbb)
    x199 -= einsum("ijab->ijab", x184)
    del x184
    x245 += einsum("ij->ij", x13)
    rdm2_f_oooo_bbbb -= einsum("ij,kl->ijkl", delta_oo.bb, x13)
    rdm2_f_oooo_bbbb += einsum("ij,kl->ilkj", delta_oo.bb, x13)
    rdm2_f_oooo_bbbb += einsum("ij,kl->kijl", delta_oo.bb, x13)
    rdm2_f_oooo_bbbb -= einsum("ij,kl->klji", delta_oo.bb, x13)
    del x13
    x14 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x14 += einsum("abij,ikab->jk", l2.abab, t2.abab)
    x17 += einsum("ij->ji", x14)
    x22 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x22 += einsum("ij->ij", x14)
    x82 += einsum("ij->ij", x14)
    x207 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x207 += einsum("ij,kiab->jkab", x14, t2.bbbb)
    x211 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x211 += einsum("ijab->ijab", x207) * -1
    del x207
    x245 += einsum("ij->ij", x14)
    del x14
    x15 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x15 += einsum("ijab->jiab", t2.bbbb) * -1
    x15 += einsum("ijab->jiba", t2.bbbb)
    x16 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x16 += einsum("abij,ikba->jk", l2.bbbb, x15)
    x17 += einsum("ij->ji", x16) * -1
    x22 += einsum("ij->ij", x16) * -1
    x70 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x70 += einsum("ia,ij->ja", t1.bb, x22)
    x71 += einsum("ia->ia", x70)
    del x70
    x72 += einsum("ia,jk->jika", t1.bb, x22)
    x99 += einsum("ia,jk->jika", t1.bb, x22)
    rdm2_f_oooo_bbbb += einsum("ij,kl->jikl", delta_oo.bb, x22) * -1
    rdm2_f_oooo_bbbb += einsum("ij,kl->jlki", delta_oo.bb, x22)
    rdm2_f_oooo_bbbb += einsum("ij,kl->kjil", delta_oo.bb, x22)
    rdm2_f_oooo_bbbb += einsum("ij,kl->klij", delta_oo.bb, x22) * -1
    del x22
    x82 += einsum("ij->ij", x16) * -1
    x83 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x83 += einsum("ia,ij->ja", t1.bb, x82)
    x84 += einsum("ia->ia", x83)
    x97 += einsum("ia->ia", x83)
    del x83
    x241 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x241 += einsum("ij,kiab->kjab", x82, t2.abab)
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x241) * -1
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x241) * -1
    del x241
    rdm2_f_oovo_bbaa += einsum("ia,jk->jkai", t1.aa, x82) * -1
    rdm2_f_vooo_aabb += einsum("ia,jk->aijk", t1.aa, x82) * -1
    del x82
    x210 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x210 += einsum("ij,ikab->kjab", x16, t2.bbbb) * -1
    del x16
    x211 += einsum("ijab->ijba", x210)
    del x210
    x81 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x81 += einsum("ai,ijab->jb", l1.bb, x15)
    x84 += einsum("ia->ia", x81) * -1
    x97 += einsum("ia->ia", x81) * -1
    x246 += einsum("ia->ia", x81) * -1
    del x81
    x244 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x244 += einsum("abij,ikba->jk", l2.bbbb, x15)
    x245 += einsum("ij->ij", x244) * -1
    del x244
    x246 += einsum("ia,ij->ja", t1.bb, x245)
    x250 += einsum("ia,ij->ja", t1.bb, x245) * 0.9999999999999993
    del x245
    x250 += einsum("ai,ijab->jb", l1.bb, x15) * -0.9999999999999993
    x285 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x285 += einsum("abij,jkbc->ikac", l2.abab, x15)
    x286 += einsum("ijab->ijab", x285) * -1
    del x285
    x17 += einsum("ij->ji", delta_oo.bb) * -1
    rdm2_f_oooo_aabb += einsum("ij,kl->jilk", delta_oo.aa, x17) * -1
    rdm2_f_oooo_bbaa += einsum("ij,kl->lkji", delta_oo.aa, x17) * -1
    del x17
    x19 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x19 += einsum("abij,klab->ijkl", l2.bbbb, t2.bbbb)
    x66 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x66 += einsum("ijkl->jilk", x19)
    x212 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x212 += einsum("ia,jikl->jkla", t1.bb, x19)
    x213 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x213 += einsum("ia,ijkb->kjba", t1.bb, x212)
    del x212
    x216 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x216 += einsum("ijab->ijab", x213)
    del x213
    x214 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x214 += einsum("ijkl->jilk", x19)
    rdm2_f_oooo_bbbb += einsum("ijkl->jkil", x19) * -1
    rdm2_f_oooo_bbbb += einsum("ijkl->jlik", x19)
    del x19
    x20 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x20 += einsum("ia,bajk->jkib", t1.bb, l2.bbbb)
    x21 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x21 += einsum("ia,jkla->kjli", t1.bb, x20)
    x66 += einsum("ijkl->ijkl", x21)
    x67 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x67 += einsum("ia,ijkl->jkla", t1.bb, x66)
    del x66
    x72 += einsum("ijka->ikja", x67)
    x99 += einsum("ijka->ikja", x67)
    del x67
    x201 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x201 += einsum("ia,ijkl->jlka", t1.bb, x21)
    x202 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x202 += einsum("ia,ijkb->jkab", t1.bb, x201)
    del x201
    x206 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x206 += einsum("ijab->ijab", x202)
    del x202
    x214 += einsum("ijkl->ijkl", x21)
    x215 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x215 += einsum("ijab,ijkl->klab", t2.bbbb, x214)
    del x214
    x216 += einsum("ijab->jiba", x215)
    del x215
    rdm2_f_vovo_bbbb = np.zeros((nvir[1], nocc[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_vovo_bbbb += einsum("ijab->ajbi", x216) * -1
    rdm2_f_vovo_bbbb += einsum("ijab->bjai", x216)
    del x216
    rdm2_f_oooo_bbbb += einsum("ijkl->ikjl", x21)
    rdm2_f_oooo_bbbb += einsum("ijkl->iljk", x21) * -1
    del x21
    x64 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x64 += einsum("ijka->ijka", x20) * -1
    x64 += einsum("ijka->jika", x20)
    x65 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x65 += einsum("ijab,ikla->jklb", x15, x64)
    x72 += einsum("ijka->jkia", x65)
    x99 += einsum("ijka->jkia", x65)
    del x65
    rdm2_f_vooo_bbbb += einsum("ijka->ajik", x99) * -1
    rdm2_f_vooo_bbbb += einsum("ijka->akij", x99)
    del x99
    x69 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x69 += einsum("ijab,ijka->kb", t2.bbbb, x64)
    x71 += einsum("ia->ia", x69) * -1
    x72 += einsum("ij,ka->jika", delta_oo.bb, x71) * -1
    rdm2_f_oovo_bbbb += einsum("ijka->ijak", x72)
    rdm2_f_oovo_bbbb += einsum("ijka->ikaj", x72) * -1
    del x72
    x211 += einsum("ia,jb->ijab", t1.bb, x71)
    rdm2_f_vooo_bbbb += einsum("ij,ka->ajik", delta_oo.bb, x71)
    rdm2_f_vooo_bbbb += einsum("ij,ka->akij", delta_oo.bb, x71) * -1
    del x71
    x84 += einsum("ia->ia", x69) * -1
    x97 += einsum("ia->ia", x69) * -1
    x250 += einsum("ia->ia", x69) * -1
    del x69
    x88 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x88 += einsum("ijab,kjlb->ikla", t2.abab, x64)
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x88) * -1
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x88) * -1
    del x88
    x246 += einsum("ijab,ijka->kb", t2.bbbb, x64) * -1
    del x64
    x117 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x117 += einsum("ijka->ijka", x20)
    x117 -= einsum("ijka->jika", x20)
    x118 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x118 += einsum("ia,ijkb->jkab", t1.bb, x117)
    rdm2_f_oovv_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_oovv_bbbb -= einsum("ijab->ijab", x118)
    rdm2_f_vvoo_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_vvoo_bbbb -= einsum("ijab->abij", x118)
    del x118
    x236 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x236 += einsum("ijab,kjlb->ikla", t2.abab, x117)
    del x117
    x238 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x238 -= einsum("ijka->ijka", x236)
    del x236
    x141 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x141 -= einsum("ijka->ijka", x20)
    x141 += einsum("ijka->jika", x20)
    x142 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x142 += einsum("ia,ijkb->jkab", t1.bb, x141)
    rdm2_f_ovvo_bbbb = np.zeros((nocc[1], nvir[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_ovvo_bbbb -= einsum("ijab->ibaj", x142)
    rdm2_f_voov_bbbb = np.zeros((nvir[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_voov_bbbb -= einsum("ijab->ajib", x142)
    del x142
    x267 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x267 += einsum("ijab,jikc->kcba", t2.bbbb, x20)
    x274 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x274 += einsum("iabc->iabc", x267)
    del x267
    x268 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x268 += einsum("ia,jikb->jkba", t1.bb, x20)
    x271 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x271 += einsum("ijab->ijab", x268) * -1
    del x268
    rdm2_f_ooov_bbbb = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_ooov_bbbb += einsum("ijka->ikja", x20)
    rdm2_f_ooov_bbbb -= einsum("ijka->jkia", x20)
    rdm2_f_ovoo_bbbb = np.zeros((nocc[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_ovoo_bbbb -= einsum("ijka->iajk", x20)
    rdm2_f_ovoo_bbbb += einsum("ijka->jaik", x20)
    del x20
    x23 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x23 += einsum("ia,abjk->jikb", t1.aa, l2.abab)
    x35 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x35 += einsum("ijab,kljb->klia", t2.abab, x23)
    x46 += einsum("ijka->ijka", x35) * -1
    x95 += einsum("ijka->ijka", x35) * -1
    x152 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x152 -= einsum("ijka->ijka", x35)
    del x35
    x41 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x41 += einsum("ijab,ikjb->ka", t2.abab, x23)
    x45 += einsum("ia->ia", x41)
    x46 += einsum("ij,ka->jika", delta_oo.aa, x45) * -1
    x173 += einsum("ia,jb->ijab", t1.aa, x45)
    rdm2_f_vooo_aaaa = np.zeros((nvir[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_vooo_aaaa += einsum("ij,ka->ajik", delta_oo.aa, x45)
    rdm2_f_vooo_aaaa += einsum("ij,ka->akij", delta_oo.aa, x45) * -1
    del x45
    x93 += einsum("ia->ia", x41)
    x249 += einsum("ia->ia", x41)
    x251 += einsum("ia->ia", x41)
    del x41
    x78 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x78 += einsum("ijab,klia->kljb", x15, x23)
    del x15
    x234 += einsum("ijka->ijka", x78) * -1
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x78) * -1
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x78) * -1
    del x78
    x85 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x85 += einsum("ijab,iklb->klja", t2.abab, x23)
    x238 -= einsum("ijka->ijka", x85)
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x85)
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x85)
    del x85
    x128 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x128 += einsum("ia,jkib->jkba", t1.bb, x23)
    rdm2_f_oovv_aabb = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_oovv_aabb -= einsum("ijab->ijba", x128)
    rdm2_f_vvoo_bbaa = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_vvoo_bbaa -= einsum("ijab->baij", x128)
    del x128
    x136 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x136 += einsum("ia,ijkb->jkab", t1.aa, x23)
    x294 += einsum("ijab->ijab", x136)
    rdm2_f_ovvo_bbaa = np.zeros((nocc[1], nvir[1], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_ovvo_bbaa -= einsum("ijab->jbai", x136)
    rdm2_f_voov_aabb = np.zeros((nvir[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_voov_aabb -= einsum("ijab->aijb", x136)
    del x136
    x290 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x290 += einsum("ijab,ikjc->kacb", t2.abab, x23)
    rdm2_f_vovv_aabb = np.zeros((nvir[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_vovv_aabb += einsum("iabc->aicb", x290) * -1
    rdm2_f_vvvo_bbaa = np.zeros((nvir[1], nvir[1], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_vvvo_bbaa += einsum("iabc->cbai", x290) * -1
    del x290
    rdm2_f_ooov_aabb = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_ooov_aabb -= einsum("ijka->ijka", x23)
    rdm2_f_ovoo_bbaa = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_ovoo_bbaa -= einsum("ijka->kaij", x23)
    del x23
    x24 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x24 += einsum("ai,jkab->ikjb", l1.aa, t2.aaaa)
    x34 += einsum("ijka->ijka", x24)
    x94 += einsum("ijka->ijka", x24)
    x152 += einsum("ijka->ijka", x24)
    del x24
    x25 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x25 += einsum("ia,waj->wji", t1.aa, lu11.aa)
    x26 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x26 += einsum("wia,wjk->jkia", u11.aa, x25)
    x34 -= einsum("ijka->ijka", x26)
    x94 -= einsum("ijka->ijka", x26)
    del x26
    rdm2_f_vooo_aaaa -= einsum("ijka->ajik", x94)
    rdm2_f_vooo_aaaa += einsum("ijka->akij", x94)
    del x94
    x28 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x28 += einsum("wia,wij->ja", u11.aa, x25)
    x32 += einsum("ia->ia", x28)
    x93 += einsum("ia->ia", x28)
    x180 += einsum("ia->ia", x28)
    x181 += einsum("ia->ia", x28)
    x249 += einsum("ia->ia", x28) * 0.9999999999999993
    x251 += einsum("ia->ia", x28)
    del x28
    x74 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x74 += einsum("wia,wjk->jkia", u11.bb, x25)
    rdm2_f_oovo_aabb -= einsum("ijka->ijak", x74)
    rdm2_f_vooo_bbaa -= einsum("ijka->akij", x74)
    del x74
    x154 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x154 += einsum("ia,wij->wja", t1.aa, x25)
    del x25
    x157 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x157 -= einsum("wia->wia", x154)
    del x154
    x27 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x27 += einsum("ai,jiba->jb", l1.bb, t2.abab)
    x32 -= einsum("ia->ia", x27)
    x93 += einsum("ia->ia", x27) * -1
    x160 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x160 += einsum("ia->ia", x27)
    x249 += einsum("ia->ia", x27) * -0.9999999999999993
    x251 += einsum("ia->ia", x27) * -1
    del x27
    x30 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x30 -= einsum("ijab->jiab", t2.aaaa)
    x30 += einsum("ijab->jiba", t2.aaaa)
    x31 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x31 += einsum("ai,ijab->jb", l1.aa, x30)
    x32 -= einsum("ia->ia", x31)
    del x31
    x34 -= einsum("ij,ka->jika", delta_oo.aa, x32)
    rdm2_f_oovo_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_oovo_aaaa += einsum("ijka->ijak", x34)
    rdm2_f_oovo_aaaa -= einsum("ijka->ikaj", x34)
    del x34
    rdm2_f_vooo_aaaa += einsum("ij,ka->ajik", delta_oo.aa, x32)
    rdm2_f_vooo_aaaa -= einsum("ij,ka->akij", delta_oo.aa, x32)
    del x32
    x138 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x138 += einsum("abij,ikca->kjcb", l2.abab, x30)
    rdm2_f_ovvo_bbaa -= einsum("ijab->jbai", x138)
    rdm2_f_voov_aabb -= einsum("ijab->aijb", x138)
    del x138
    x156 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x156 += einsum("wai,ijba->wjb", lu11.aa, x30)
    x157 -= einsum("wia->wia", x156)
    del x156
    x159 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x159 += einsum("ai,ijba->jb", l1.aa, x30)
    x160 -= einsum("ia->ia", x159)
    del x159
    x161 += einsum("ia,jb->ijab", t1.aa, x160)
    del x160
    x237 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x237 += einsum("ijka,ilba->ljkb", x10, x30)
    del x10
    x238 -= einsum("ijka->ijka", x237)
    del x237
    x37 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x37 += einsum("ijab->jiab", t2.aaaa)
    x37 += einsum("ijab->jiba", t2.aaaa) * -1
    x38 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x38 += einsum("ijka,jlba->iklb", x36, x37)
    del x36
    x46 += einsum("ijka->ijka", x38)
    rdm2_f_oovo_aaaa += einsum("ijka->ijak", x46)
    rdm2_f_oovo_aaaa += einsum("ijka->ikaj", x46) * -1
    del x46
    x95 += einsum("ijka->ijka", x38)
    del x38
    rdm2_f_vooo_aaaa += einsum("ijka->ajik", x95) * -1
    rdm2_f_vooo_aaaa += einsum("ijka->akij", x95)
    del x95
    x96 += einsum("abij,ikab->kj", l2.aaaa, x37) * -1
    del x37
    x47 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x47 += einsum("w,wia->ia", ls1, u11.aa)
    x49 += einsum("ia->ia", x47)
    x93 += einsum("ia->ia", x47) * -1
    x180 -= einsum("ia->ia", x47)
    x181 -= einsum("ia->ia", x47)
    rdm2_f_vovo_aaaa -= einsum("ia,jb->bjai", t1.aa, x181)
    rdm2_f_vovo_aaaa += einsum("ia,jb->ajbi", t1.aa, x181)
    del x181
    x249 += einsum("ia->ia", x47) * -0.9999999999999993
    rdm2_f_vovo_bbaa += einsum("ia,jb->aibj", t1.bb, x249) * -1
    del x249
    x251 += einsum("ia->ia", x47) * -1
    del x47
    rdm2_f_vovo_aabb += einsum("ia,jb->bjai", t1.bb, x251) * -1
    del x251
    x49 += einsum("ia->ia", t1.aa)
    rdm2_f_oovo_aaaa += einsum("ij,ka->jiak", delta_oo.aa, x49)
    rdm2_f_oovo_aaaa -= einsum("ij,ka->jkai", delta_oo.aa, x49)
    rdm2_f_vooo_aaaa -= einsum("ij,ka->aijk", delta_oo.aa, x49)
    rdm2_f_vooo_aaaa += einsum("ij,ka->akji", delta_oo.aa, x49)
    del x49
    x50 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x50 += einsum("w,wia->ia", ls1, u11.bb)
    x73 += einsum("ia->ia", x50)
    x84 += einsum("ia->ia", x50) * -1
    x97 += einsum("ia->ia", x50) * -1
    x218 -= einsum("ia->ia", x50)
    x219 -= einsum("ia->ia", x50)
    x246 += einsum("ia->ia", x50) * -1
    x250 += einsum("ia->ia", x50) * -0.9999999999999993
    rdm2_f_oovo_bbbb -= einsum("ij,ka->jkai", delta_oo.bb, x50)
    rdm2_f_vooo_bbbb += einsum("ij,ka->akji", delta_oo.bb, x50)
    del x50
    x52 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x52 += einsum("ai,jkba->ijkb", l1.bb, t2.bbbb)
    x62 += einsum("ijka->ijka", x52)
    x98 += einsum("ijka->ijka", x52)
    x190 += einsum("ijka->ijka", x52)
    del x52
    x53 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x53 += einsum("ia,waj->wji", t1.bb, lu11.bb)
    x54 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x54 += einsum("wia,wjk->jkia", u11.bb, x53)
    x62 -= einsum("ijka->ijka", x54)
    x98 -= einsum("ijka->ijka", x54)
    del x54
    rdm2_f_vooo_bbbb -= einsum("ijka->ajik", x98)
    rdm2_f_vooo_bbbb += einsum("ijka->akij", x98)
    del x98
    x56 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x56 += einsum("wia,wij->ja", u11.bb, x53)
    x60 += einsum("ia->ia", x56)
    x84 += einsum("ia->ia", x56)
    x97 += einsum("ia->ia", x56)
    x218 += einsum("ia->ia", x56)
    x219 += einsum("ia->ia", x56)
    rdm2_f_vovo_bbbb += einsum("ia,jb->biaj", t1.bb, x219)
    rdm2_f_vovo_bbbb -= einsum("ia,jb->aibj", t1.bb, x219)
    del x219
    x246 += einsum("ia->ia", x56)
    x250 += einsum("ia->ia", x56) * 0.9999999999999993
    del x56
    x87 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x87 += einsum("wia,wjk->ijka", u11.aa, x53)
    rdm2_f_oovo_bbaa -= einsum("ijka->jkai", x87)
    rdm2_f_vooo_aabb -= einsum("ijka->aijk", x87)
    del x87
    x193 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x193 += einsum("ia,wij->wja", t1.bb, x53)
    del x53
    x195 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x195 -= einsum("wia->wia", x193)
    del x193
    x55 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x55 += einsum("ai,ijab->jb", l1.aa, t2.abab)
    x60 -= einsum("ia->ia", x55)
    x84 += einsum("ia->ia", x55) * -1
    x97 += einsum("ia->ia", x55) * -1
    rdm2_f_vooo_bbaa += einsum("ij,ka->akji", delta_oo.aa, x97) * -1
    del x97
    x198 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x198 += einsum("ia->ia", x55)
    x246 += einsum("ia->ia", x55) * -1
    x250 += einsum("ia->ia", x55) * -0.9999999999999993
    del x55
    x58 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x58 -= einsum("ijab->jiab", t2.bbbb)
    x58 += einsum("ijab->jiba", t2.bbbb)
    x59 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x59 += einsum("ai,ijab->jb", l1.bb, x58)
    x60 -= einsum("ia->ia", x59)
    del x59
    x62 -= einsum("ij,ka->jika", delta_oo.bb, x60)
    rdm2_f_oovo_bbbb += einsum("ijka->ijak", x62)
    rdm2_f_oovo_bbbb -= einsum("ijka->ikaj", x62)
    del x62
    rdm2_f_vooo_bbbb += einsum("ij,ka->ajik", delta_oo.bb, x60)
    rdm2_f_vooo_bbbb -= einsum("ij,ka->akij", delta_oo.bb, x60)
    del x60
    x189 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x189 += einsum("ijka,ilab->ljkb", x141, x58)
    del x141
    x190 += einsum("ijka->jkia", x189)
    del x189
    x191 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x191 += einsum("ia,ijkb->jkab", t1.bb, x190)
    del x190
    x199 += einsum("ijab->ijab", x191)
    del x191
    x73 += einsum("ia->ia", t1.bb)
    rdm2_f_oovo_bbbb += einsum("ij,ka->jiak", delta_oo.bb, x73)
    rdm2_f_vooo_bbbb -= einsum("ij,ka->aijk", delta_oo.bb, x73)
    del x73
    x76 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x76 += einsum("ai,jkab->ijkb", l1.aa, t2.abab)
    x234 += einsum("ijka->ijka", x76) * -1
    x235 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x235 += einsum("ia,ijkb->jkab", t1.aa, x234)
    del x234
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x235)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x235)
    del x235
    rdm2_f_oovo_aabb -= einsum("ijka->ijak", x76)
    rdm2_f_vooo_bbaa -= einsum("ijka->akij", x76)
    del x76
    x84 += einsum("ia->ia", t1.bb) * -1
    rdm2_f_oovo_aabb += einsum("ij,ka->jiak", delta_oo.aa, x84) * -1
    del x84
    x86 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x86 += einsum("ai,jkba->jikb", l1.bb, t2.abab)
    x238 += einsum("ijka->ijka", x86)
    x239 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x239 += einsum("ia,jikb->jkba", t1.bb, x238)
    del x238
    rdm2_f_vovo_bbaa -= einsum("ijab->bjai", x239)
    rdm2_f_vovo_aabb -= einsum("ijab->aibj", x239)
    del x239
    rdm2_f_oovo_bbaa -= einsum("ijka->jkai", x86)
    rdm2_f_vooo_aabb -= einsum("ijka->aijk", x86)
    del x86
    x93 += einsum("ia->ia", t1.aa) * -1
    rdm2_f_oovo_bbaa += einsum("ij,ka->jiak", delta_oo.bb, x93) * -1
    rdm2_f_vooo_aabb += einsum("ij,ka->akji", delta_oo.bb, x93) * -1
    del x93
    x96 += einsum("ij->ji", delta_oo.aa) * -1
    rdm2_f_vooo_bbaa += einsum("ia,jk->aikj", t1.bb, x96) * -1
    del x96
    x100 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x100 += einsum("wai,wjb->ijab", lu11.aa, u11.aa)
    rdm2_f_oovv_aaaa -= einsum("ijab->ijba", x100)
    rdm2_f_ovvo_aaaa += einsum("ijab->iabj", x100)
    rdm2_f_voov_aaaa += einsum("ijab->bjia", x100)
    rdm2_f_vvoo_aaaa -= einsum("ijab->baij", x100)
    del x100
    x101 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x101 += einsum("abij,kjcb->ikac", l2.abab, t2.abab)
    x146 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x146 += einsum("ijab,jkac->ikbc", t2.aaaa, x101)
    x161 -= einsum("ijab->ijab", x146)
    del x146
    x148 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x148 += einsum("ijab->ijab", x101)
    x264 += einsum("ijab->ijab", x101)
    rdm2_f_oovv_aaaa -= einsum("ijab->ijba", x101)
    rdm2_f_ovvo_aaaa += einsum("ijab->iabj", x101)
    rdm2_f_voov_aaaa += einsum("ijab->bjia", x101)
    rdm2_f_vvoo_aaaa -= einsum("ijab->baij", x101)
    del x101
    x102 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x102 -= einsum("abij->jiab", l2.aaaa)
    x102 += einsum("abij->jiba", l2.aaaa)
    x103 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x103 += einsum("ijab,ikbc->kjca", x102, x30)
    rdm2_f_oovv_aaaa += einsum("ijab->jiab", x103)
    rdm2_f_vvoo_aaaa += einsum("ijab->abji", x103)
    del x103
    x129 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x129 += einsum("ijab,ikcb->kjca", x102, x30)
    rdm2_f_ovvo_aaaa += einsum("ijab->jbai", x129)
    rdm2_f_voov_aaaa += einsum("ijab->aijb", x129)
    del x129
    x134 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x134 += einsum("ijab,ikca->kjcb", t2.abab, x102)
    x200 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x200 -= einsum("ijab,ikac->jkbc", t2.abab, x134)
    rdm2_f_vovo_bbbb -= einsum("ijab->ajbi", x200)
    rdm2_f_vovo_bbbb += einsum("ijab->aibj", x200)
    del x200
    x221 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x221 -= einsum("ijab->ijab", x134)
    rdm2_f_ovvo_aabb -= einsum("ijab->iabj", x134)
    rdm2_f_voov_bbaa -= einsum("ijab->bjia", x134)
    del x134
    x147 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x147 += einsum("ijab,ikbc->jkac", t2.aaaa, x102)
    x148 -= einsum("ijab->jiba", x147)
    del x147
    x149 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x149 += einsum("ijab,ikac->jkbc", t2.aaaa, x148)
    del x148
    x161 += einsum("ijab->ijab", x149)
    del x149
    x164 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x164 += einsum("ijab,ikca->jkbc", t2.aaaa, x102)
    x165 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x165 -= einsum("ijab,kica->jkbc", t2.aaaa, x164)
    del x164
    x169 += einsum("ijab->jiba", x165)
    del x165
    x166 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x166 += einsum("ijab,ikcb->jkac", t2.aaaa, x102)
    del x102
    x167 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x167 -= einsum("ijab,kicb->jkac", t2.aaaa, x166)
    del x166
    x169 += einsum("ijab->ijab", x167)
    del x167
    x106 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x106 += einsum("ai,ib->ab", l1.aa, t1.aa)
    x111 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x111 += einsum("ab->ab", x106)
    x259 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x259 += einsum("ab->ab", x106)
    del x106
    x107 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x107 += einsum("wai,wib->ab", lu11.aa, u11.aa)
    x111 += einsum("ab->ab", x107)
    x144 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x144 += einsum("ab,ijca->ijcb", x107, t2.aaaa)
    x161 -= einsum("ijab->ijab", x144)
    del x144
    x230 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x230 += einsum("ab->ab", x107)
    x259 += einsum("ab->ab", x107)
    del x107
    x260 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x260 += einsum("ia,bc->ibac", t1.aa, x259)
    del x259
    x108 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x108 += einsum("abij,ijcb->ac", l2.abab, t2.abab)
    x111 += einsum("ab->ab", x108)
    x170 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x170 += einsum("ab->ab", x108)
    x230 += einsum("ab->ab", x108)
    del x108
    x109 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x109 += einsum("abij->jiab", l2.aaaa) * -1
    x109 += einsum("abij->jiba", l2.aaaa)
    x110 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x110 += einsum("ijab,ijbc->ac", t2.aaaa, x109)
    x111 += einsum("ab->ba", x110) * -1
    rdm2_f_oovv_aaaa += einsum("ij,ab->jiba", delta_oo.aa, x111)
    rdm2_f_oovv_bbaa += einsum("ij,ab->jiba", delta_oo.bb, x111)
    rdm2_f_ovvo_aaaa += einsum("ij,ab->jabi", delta_oo.aa, x111) * -1
    rdm2_f_voov_aaaa += einsum("ij,ab->bija", delta_oo.aa, x111) * -1
    rdm2_f_vvoo_aaaa += einsum("ij,ab->baji", delta_oo.aa, x111)
    rdm2_f_vvoo_aabb += einsum("ij,ab->baji", delta_oo.bb, x111)
    rdm2_f_vovv_bbaa += einsum("ia,bc->aicb", t1.bb, x111)
    rdm2_f_vvvo_aabb += einsum("ia,bc->cbai", t1.bb, x111)
    del x111
    x170 += einsum("ab->ba", x110) * -1
    x171 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x171 += einsum("ab,ijac->ijcb", x170, t2.aaaa)
    x173 += einsum("ijab->jiab", x171)
    del x171
    rdm2_f_vovo_aaaa += einsum("ijab->aibj", x173) * -1
    rdm2_f_vovo_aaaa += einsum("ijab->biaj", x173)
    rdm2_f_vovo_aaaa += einsum("ijab->ajbi", x173)
    rdm2_f_vovo_aaaa += einsum("ijab->bjai", x173) * -1
    del x173
    x266 += einsum("ia,bc->ibac", t1.aa, x170)
    del x170
    x230 += einsum("ab->ba", x110) * -1
    del x110
    x231 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x231 += einsum("ab,ijac->ijbc", x230, t2.abab)
    del x230
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x231) * -1
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x231) * -1
    del x231
    x263 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x263 += einsum("ijab,ikcb->kjca", x109, x5)
    del x5
    x264 += einsum("ijab->jiba", x263)
    del x263
    x265 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x265 += einsum("ia,ijbc->jabc", t1.aa, x264)
    del x264
    x266 += einsum("iabc->ibac", x265) * -1
    del x265
    rdm2_f_vovv_aaaa = np.zeros((nvir[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_vovv_aaaa += einsum("iabc->bica", x266)
    rdm2_f_vovv_aaaa += einsum("iabc->ciba", x266) * -1
    rdm2_f_vvvo_aaaa = np.zeros((nvir[0], nvir[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_vvvo_aaaa += einsum("iabc->baci", x266) * -1
    rdm2_f_vvvo_aaaa += einsum("iabc->cabi", x266)
    del x266
    x284 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x284 += einsum("ijab,ikac->kjcb", t2.abab, x109)
    del x109
    x286 += einsum("ijab->ijab", x284) * -1
    del x284
    x287 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x287 += einsum("ia,ijbc->jabc", t1.aa, x286)
    del x286
    rdm2_f_vovv_bbaa += einsum("iabc->ciab", x287) * -1
    rdm2_f_vvvo_aabb += einsum("iabc->abci", x287) * -1
    del x287
    x112 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x112 += einsum("abij,ikac->jkbc", l2.abab, t2.abab)
    x224 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x224 += einsum("ijab->ijab", x112)
    x271 += einsum("ijab->ijab", x112)
    rdm2_f_oovv_bbbb -= einsum("ijab->ijba", x112)
    rdm2_f_ovvo_bbbb += einsum("ijab->iabj", x112)
    rdm2_f_voov_bbbb += einsum("ijab->bjia", x112)
    rdm2_f_vvoo_bbbb -= einsum("ijab->baij", x112)
    x113 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x113 += einsum("wai,wjb->ijab", lu11.bb, u11.bb)
    rdm2_f_oovv_bbbb -= einsum("ijab->ijba", x113)
    rdm2_f_ovvo_bbbb += einsum("ijab->iabj", x113)
    rdm2_f_voov_bbbb += einsum("ijab->bjia", x113)
    rdm2_f_vvoo_bbbb -= einsum("ijab->baij", x113)
    del x113
    x114 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x114 += einsum("ijab->jiab", t2.bbbb)
    x114 -= einsum("ijab->jiba", t2.bbbb)
    x135 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x135 += einsum("abij,jkbc->ikac", l2.abab, x114)
    x221 -= einsum("ijab->ijab", x135)
    x222 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x222 += einsum("ijab,ikac->kjcb", x221, x30)
    del x221
    del x30
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x222)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x222)
    del x222
    rdm2_f_ovvo_aabb -= einsum("ijab->iabj", x135)
    rdm2_f_voov_bbaa -= einsum("ijab->bjia", x135)
    del x135
    x188 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x188 += einsum("ijab,ikac->jkbc", x112, x114)
    del x112
    x199 -= einsum("ijab->ijab", x188)
    del x188
    x194 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x194 += einsum("wai,ijab->wjb", lu11.bb, x114)
    x195 -= einsum("wia->wia", x194)
    del x194
    x197 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x197 += einsum("ai,ijab->jb", l1.bb, x114)
    x198 -= einsum("ia->ia", x197)
    del x197
    x199 += einsum("ia,jb->ijab", t1.bb, x198)
    del x198
    x115 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x115 += einsum("abij->jiab", l2.bbbb)
    x115 -= einsum("abij->jiba", l2.bbbb)
    x116 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x116 += einsum("ijab,ikbc->jkac", x114, x115)
    rdm2_f_oovv_bbbb += einsum("ijab->jiab", x116)
    rdm2_f_vvoo_bbbb += einsum("ijab->abji", x116)
    del x116
    x139 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x139 += einsum("ijab,jkbc->ikac", t2.abab, x115)
    x168 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x168 -= einsum("ijab,kjcb->ikac", t2.abab, x139)
    x169 += einsum("ijab->jiba", x168)
    del x168
    rdm2_f_ovvo_bbaa -= einsum("ijab->jbai", x139)
    rdm2_f_voov_aabb -= einsum("ijab->aijb", x139)
    del x139
    x140 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x140 += einsum("ijab,ikca->kjcb", x115, x58)
    del x58
    del x115
    rdm2_f_ovvo_bbbb += einsum("ijab->jbai", x140)
    rdm2_f_voov_bbbb += einsum("ijab->aijb", x140)
    del x140
    x119 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x119 += einsum("ai,ib->ab", l1.bb, t1.bb)
    x124 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x124 += einsum("ab->ab", x119)
    x278 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x278 += einsum("ab->ab", x119)
    del x119
    x120 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x120 += einsum("wai,wib->ab", lu11.bb, u11.bb)
    x124 += einsum("ab->ab", x120)
    x183 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x183 += einsum("ab,ijac->jicb", x120, t2.bbbb)
    x199 -= einsum("ijab->ijab", x183)
    del x183
    x228 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x228 += einsum("ab->ab", x120)
    x278 += einsum("ab->ab", x120)
    del x120
    x279 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x279 += einsum("ia,bc->ibac", t1.bb, x278)
    del x278
    x121 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x121 += einsum("abij,ijac->bc", l2.abab, t2.abab)
    x124 += einsum("ab->ab", x121)
    x208 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x208 += einsum("ab,ijac->jibc", x121, t2.bbbb)
    x211 += einsum("ijab->ijab", x208) * -1
    del x208
    x228 += einsum("ab->ab", x121)
    x273 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x273 += einsum("ab->ab", x121)
    del x121
    x122 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x122 += einsum("abij->jiab", l2.bbbb) * -1
    x122 += einsum("abij->jiba", l2.bbbb)
    x123 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x123 += einsum("ijab,ijca->bc", t2.bbbb, x122)
    x124 += einsum("ab->ba", x123) * -1
    rdm2_f_oovv_bbbb += einsum("ij,ab->jiba", delta_oo.bb, x124)
    rdm2_f_oovv_aabb += einsum("ij,ab->jiba", delta_oo.aa, x124)
    rdm2_f_ovvo_bbbb += einsum("ij,ab->jabi", delta_oo.bb, x124) * -1
    rdm2_f_voov_bbbb += einsum("ij,ab->bija", delta_oo.bb, x124) * -1
    rdm2_f_vvoo_bbbb += einsum("ij,ab->baji", delta_oo.bb, x124)
    rdm2_f_vvoo_bbaa += einsum("ij,ab->baji", delta_oo.aa, x124)
    rdm2_f_vovv_aabb += einsum("ia,bc->aicb", t1.aa, x124)
    rdm2_f_vvvo_bbaa += einsum("ia,bc->cbai", t1.aa, x124)
    del x124
    x209 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x209 += einsum("ab,ijbc->ijca", x123, t2.bbbb) * -1
    x211 += einsum("ijab->jiab", x209)
    del x209
    rdm2_f_vovo_bbbb += einsum("ijab->aibj", x211) * -1
    rdm2_f_vovo_bbbb += einsum("ijab->biaj", x211)
    rdm2_f_vovo_bbbb += einsum("ijab->ajbi", x211)
    rdm2_f_vovo_bbbb += einsum("ijab->bjai", x211) * -1
    del x211
    x228 += einsum("ab->ba", x123) * -1
    x229 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x229 += einsum("ab,ijca->ijcb", x228, t2.abab)
    del x228
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x229) * -1
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x229) * -1
    del x229
    x273 += einsum("ab->ba", x123) * -1
    del x123
    x274 += einsum("ia,bc->ibac", t1.bb, x273)
    del x273
    x293 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x293 += einsum("ijab,jkbc->ikac", t2.abab, x122)
    x294 += einsum("ijab->ijab", x293) * -1
    del x293
    x295 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x295 += einsum("ia,jibc->jbac", t1.bb, x294)
    del x294
    rdm2_f_vovv_aabb += einsum("iabc->aibc", x295) * -1
    rdm2_f_vvvo_bbaa += einsum("iabc->bcai", x295) * -1
    del x295
    x125 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x125 += einsum("abij,ikcb->jkac", l2.abab, t2.abab)
    x281 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x281 += einsum("ia,ijbc->jbca", t1.bb, x125)
    rdm2_f_vovv_bbaa += einsum("iabc->ciba", x281) * -1
    rdm2_f_vvvo_aabb += einsum("iabc->baci", x281) * -1
    del x281
    rdm2_f_oovv_bbaa -= einsum("ijab->ijba", x125)
    rdm2_f_vvoo_aabb -= einsum("ijab->baij", x125)
    del x125
    x127 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x127 += einsum("abij,kjac->ikbc", l2.abab, t2.abab)
    x220 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x220 += einsum("ijab,ikbc->kjac", t2.abab, x127)
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x220)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x220)
    del x220
    x291 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x291 += einsum("ia,ijbc->jabc", t1.aa, x127)
    rdm2_f_vovv_aabb += einsum("iabc->aicb", x291) * -1
    rdm2_f_vvvo_bbaa += einsum("iabc->cbai", x291) * -1
    del x291
    rdm2_f_oovv_aabb -= einsum("ijab->ijba", x127)
    rdm2_f_vvoo_bbaa -= einsum("ijab->baij", x127)
    del x127
    x133 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x133 += einsum("wai,wjb->ijab", lu11.aa, u11.bb)
    rdm2_f_ovvo_aabb += einsum("ijab->iabj", x133)
    rdm2_f_voov_bbaa += einsum("ijab->bjia", x133)
    del x133
    x137 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x137 += einsum("wai,wjb->jiba", lu11.bb, u11.aa)
    rdm2_f_ovvo_bbaa += einsum("ijab->jbai", x137)
    rdm2_f_voov_aabb += einsum("ijab->aijb", x137)
    del x137
    x150 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x150 += einsum("ijab->jiab", t2.aaaa)
    x150 -= einsum("ijab->jiba", t2.aaaa)
    x151 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x151 += einsum("ijka,jlba->iklb", x104, x150)
    del x104
    del x150
    x152 += einsum("ijka->ijka", x151)
    del x151
    x153 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x153 += einsum("ia,ijkb->jkab", t1.aa, x152)
    del x152
    x161 += einsum("ijab->ijab", x153)
    del x153
    x155 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x155 += einsum("wai,jiba->wjb", lu11.bb, t2.abab)
    x157 += einsum("wia->wia", x155)
    del x155
    x158 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x158 += einsum("wia,wjb->ijab", u11.aa, x157)
    x161 += einsum("ijab->jiba", x158)
    del x158
    rdm2_f_vovo_aaaa += einsum("ijab->aibj", x161)
    rdm2_f_vovo_aaaa -= einsum("ijab->biaj", x161)
    rdm2_f_vovo_aaaa -= einsum("ijab->ajbi", x161)
    rdm2_f_vovo_aaaa += einsum("ijab->bjai", x161)
    del x161
    x243 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x243 += einsum("wia,wjb->jiba", u11.bb, x157)
    del x157
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x243)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x243)
    del x243
    x169 += einsum("ijab->jiba", t2.aaaa)
    rdm2_f_vovo_aaaa -= einsum("ijab->biaj", x169)
    rdm2_f_vovo_aaaa += einsum("ijab->aibj", x169)
    del x169
    x180 -= einsum("ia->ia", t1.aa)
    rdm2_f_vovo_aaaa -= einsum("ia,jb->aibj", t1.aa, x180)
    rdm2_f_vovo_aaaa += einsum("ia,jb->biaj", t1.aa, x180)
    del x180
    x185 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x185 -= einsum("abij->jiab", l2.bbbb)
    x185 += einsum("abij->jiba", l2.bbbb)
    x186 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x186 += einsum("ijab,ikcb->jkac", t2.bbbb, x185)
    x187 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x187 -= einsum("ijab,kica->jkbc", t2.bbbb, x186)
    x199 -= einsum("ijab->ijab", x187)
    del x187
    x205 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x205 -= einsum("ijab,kicb->jkac", t2.bbbb, x186)
    del x186
    x206 += einsum("ijab->jiba", x205)
    del x205
    x203 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x203 += einsum("ijab,ikca->jkbc", t2.bbbb, x185)
    x204 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x204 -= einsum("ijab,kica->jkbc", t2.bbbb, x203)
    del x203
    x206 += einsum("ijab->ijab", x204)
    del x204
    x223 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x223 += einsum("ijab,ikca->jkbc", x114, x185)
    del x185
    del x114
    x224 += einsum("ijab->jiba", x223)
    del x223
    x225 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x225 += einsum("ijab,jkbc->ikac", t2.abab, x224)
    del x224
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x225)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x225)
    del x225
    x192 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x192 += einsum("wai,ijab->wjb", lu11.aa, t2.abab)
    x195 += einsum("wia->wia", x192)
    del x192
    x196 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x196 += einsum("wia,wjb->ijab", u11.bb, x195)
    x199 += einsum("ijab->jiba", x196)
    del x196
    rdm2_f_vovo_bbbb += einsum("ijab->aibj", x199)
    rdm2_f_vovo_bbbb -= einsum("ijab->biaj", x199)
    rdm2_f_vovo_bbbb -= einsum("ijab->ajbi", x199)
    rdm2_f_vovo_bbbb += einsum("ijab->bjai", x199)
    del x199
    x242 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x242 += einsum("wia,wjb->ijab", u11.aa, x195)
    del x195
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x242)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x242)
    del x242
    x206 += einsum("ijab->jiba", t2.bbbb)
    rdm2_f_vovo_bbbb -= einsum("ijab->biaj", x206)
    rdm2_f_vovo_bbbb += einsum("ijab->aibj", x206)
    del x206
    x218 -= einsum("ia->ia", t1.bb)
    rdm2_f_vovo_bbbb += einsum("ia,jb->ajbi", t1.bb, x218)
    rdm2_f_vovo_bbbb -= einsum("ia,jb->bjai", t1.bb, x218)
    del x218
    x246 += einsum("ia->ia", t1.bb) * -1
    rdm2_f_vovo_bbaa += einsum("ia,jb->bjai", t1.aa, x246) * -1
    del x246
    x250 += einsum("ia->ia", t1.bb) * -0.9999999999999993
    rdm2_f_vovo_aabb += einsum("ia,jb->aibj", t1.aa, x250) * -1
    del x250
    x252 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x252 += einsum("ia,bcji->jbca", t1.aa, l2.aaaa)
    x297 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x297 += einsum("ia,ibcd->cbda", t1.aa, x252)
    x298 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x298 += einsum("abcd->badc", x297)
    del x297
    rdm2_f_ovvv_aaaa = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_ovvv_aaaa += einsum("iabc->iacb", x252)
    rdm2_f_ovvv_aaaa -= einsum("iabc->ibca", x252)
    rdm2_f_vvov_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_vvov_aaaa -= einsum("iabc->caib", x252)
    rdm2_f_vvov_aaaa += einsum("iabc->cbia", x252)
    del x252
    x253 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x253 += einsum("ia,bcji->jbca", t1.bb, l2.bbbb)
    x302 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x302 += einsum("ia,ibcd->cbda", t1.bb, x253)
    x303 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x303 += einsum("abcd->badc", x302)
    del x302
    rdm2_f_ovvv_bbbb = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_ovvv_bbbb += einsum("iabc->iacb", x253)
    rdm2_f_ovvv_bbbb -= einsum("iabc->ibca", x253)
    rdm2_f_vvov_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_vvov_bbbb -= einsum("iabc->caib", x253)
    rdm2_f_vvov_bbbb += einsum("iabc->cbia", x253)
    del x253
    x254 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x254 += einsum("ia,bcij->jbac", t1.aa, l2.abab)
    rdm2_f_ovvv_bbaa = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_ovvv_bbaa += einsum("iabc->icba", x254)
    rdm2_f_vvov_aabb = np.zeros((nvir[0], nvir[0], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_vvov_aabb += einsum("iabc->baic", x254)
    del x254
    x255 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x255 += einsum("ia,bcji->jbca", t1.bb, l2.abab)
    x300 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x300 += einsum("ia,ibcd->bacd", t1.aa, x255)
    rdm2_f_vvvv_bbaa = np.zeros((nvir[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_vvvv_bbaa += einsum("abcd->dcba", x300)
    rdm2_f_vvvv_aabb = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_vvvv_aabb += einsum("abcd->badc", x300)
    del x300
    rdm2_f_ovvv_aabb = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_ovvv_aabb += einsum("iabc->iacb", x255)
    rdm2_f_vvov_bbaa = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_vvov_bbaa += einsum("iabc->cbia", x255)
    del x255
    x256 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x256 += einsum("ai,jibc->jabc", l1.aa, t2.aaaa)
    x260 += einsum("iabc->iabc", x256)
    del x256
    x257 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x257 += einsum("ia,wbi->wba", t1.aa, lu11.aa)
    x258 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x258 += einsum("wia,wbc->ibca", u11.aa, x257)
    x260 -= einsum("iabc->iabc", x258)
    del x258
    rdm2_f_vovv_aaaa += einsum("iabc->bica", x260)
    rdm2_f_vovv_aaaa -= einsum("iabc->ciba", x260)
    rdm2_f_vvvo_aaaa -= einsum("iabc->baci", x260)
    rdm2_f_vvvo_aaaa += einsum("iabc->cabi", x260)
    del x260
    x282 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x282 += einsum("wia,wbc->ibca", u11.bb, x257)
    del x257
    rdm2_f_vovv_bbaa += einsum("iabc->ciba", x282)
    rdm2_f_vvvo_aabb += einsum("iabc->baci", x282)
    del x282
    x269 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x269 += einsum("ijab->jiab", t2.bbbb)
    x269 += einsum("ijab->jiba", t2.bbbb) * -1
    x270 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x270 += einsum("ijab,ikbc->jkac", x122, x269)
    del x122
    del x269
    x271 += einsum("ijab->ijab", x270)
    del x270
    x272 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x272 += einsum("ia,ijbc->jabc", t1.bb, x271)
    del x271
    x274 += einsum("iabc->ibac", x272) * -1
    del x272
    rdm2_f_vovv_bbbb = np.zeros((nvir[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_vovv_bbbb += einsum("iabc->bica", x274)
    rdm2_f_vovv_bbbb += einsum("iabc->ciba", x274) * -1
    rdm2_f_vvvo_bbbb = np.zeros((nvir[1], nvir[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_vvvo_bbbb += einsum("iabc->baci", x274) * -1
    rdm2_f_vvvo_bbbb += einsum("iabc->cabi", x274)
    del x274
    x275 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x275 += einsum("ai,jibc->jabc", l1.bb, t2.bbbb)
    x279 += einsum("iabc->iabc", x275)
    del x275
    x276 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x276 += einsum("ia,wbi->wba", t1.bb, lu11.bb)
    x277 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x277 += einsum("wia,wbc->ibca", u11.bb, x276)
    x279 -= einsum("iabc->iabc", x277)
    del x277
    rdm2_f_vovv_bbbb += einsum("iabc->bica", x279)
    rdm2_f_vovv_bbbb -= einsum("iabc->ciba", x279)
    rdm2_f_vvvo_bbbb -= einsum("iabc->baci", x279)
    rdm2_f_vvvo_bbbb += einsum("iabc->cabi", x279)
    del x279
    x289 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x289 += einsum("wia,wbc->iabc", u11.aa, x276)
    del x276
    rdm2_f_vovv_aabb += einsum("iabc->aicb", x289)
    rdm2_f_vvvo_bbaa += einsum("iabc->cbai", x289)
    del x289
    x283 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x283 += einsum("ai,ijbc->jabc", l1.aa, t2.abab)
    rdm2_f_vovv_bbaa += einsum("iabc->ciba", x283)
    rdm2_f_vvvo_aabb += einsum("iabc->baci", x283)
    del x283
    x288 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x288 += einsum("ai,jibc->jbac", l1.bb, t2.abab)
    rdm2_f_vovv_aabb += einsum("iabc->aicb", x288)
    rdm2_f_vvvo_bbaa += einsum("iabc->cbai", x288)
    del x288
    x296 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x296 += einsum("abij,ijcd->abcd", l2.aaaa, t2.aaaa)
    x298 += einsum("abcd->badc", x296)
    del x296
    rdm2_f_vvvv_aaaa = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_vvvv_aaaa += einsum("abcd->dacb", x298) * -1
    rdm2_f_vvvv_aaaa += einsum("abcd->cadb", x298)
    del x298
    x299 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x299 += einsum("abij,ijcd->acbd", l2.abab, t2.abab)
    rdm2_f_vvvv_bbaa += einsum("abcd->dcba", x299)
    rdm2_f_vvvv_aabb += einsum("abcd->badc", x299)
    del x299
    x301 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x301 += einsum("abij,ijcd->abcd", l2.bbbb, t2.bbbb)
    x303 += einsum("abcd->badc", x301)
    del x301
    rdm2_f_vvvv_bbbb = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_vvvv_bbbb += einsum("abcd->dacb", x303) * -1
    rdm2_f_vvvv_bbbb += einsum("abcd->cadb", x303)
    del x303
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
    rdm2_f_oovo_bbbb -= einsum("ij,ka->jkai", delta_oo.bb, t1.bb)
    rdm2_f_vooo_bbbb += einsum("ij,ka->akji", delta_oo.bb, t1.bb)
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

def make_sing_b_dm(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, l1=None, l2=None, ls1=None, lu11=None, **kwargs):
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
    dm_b_des += einsum("w->w", s1)
    dm_b_des += einsum("ai,wia->w", l1.aa, u11.aa)
    dm_b_des += einsum("ai,wia->w", l1.bb, u11.bb)

    dm_b = np.array([dm_b_cre, dm_b_des])

    return dm_b

def make_rdm1_b(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, l1=None, l2=None, ls1=None, lu11=None, **kwargs):
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
    rdm1_b += einsum("w,x->wx", ls1, s1)
    rdm1_b += einsum("wai,xia->wx", lu11.aa, u11.aa)
    rdm1_b += einsum("wai,xia->wx", lu11.bb, u11.bb)

    return rdm1_b

def make_eb_coup_rdm(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, l1=None, l2=None, ls1=None, lu11=None, **kwargs):
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
    x0 += einsum("ia,waj->wji", t1.aa, lu11.aa)
    x48 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x48 += einsum("wia,wij->ja", u11.aa, x0)
    rdm_eb_cre_oo_aa = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    rdm_eb_cre_oo_aa -= einsum("wij->wji", x0)
    rdm_eb_cre_ov_aa = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    rdm_eb_cre_ov_aa -= einsum("ia,wij->wja", t1.aa, x0)
    del x0
    x1 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x1 += einsum("ia,waj->wji", t1.bb, lu11.bb)
    x60 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x60 += einsum("wia,wij->ja", u11.bb, x1)
    rdm_eb_cre_oo_bb = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    rdm_eb_cre_oo_bb -= einsum("wij->wji", x1)
    rdm_eb_cre_ov_bb = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    rdm_eb_cre_ov_bb -= einsum("ia,wij->wja", t1.bb, x1)
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum("ijab->jiab", t2.aaaa)
    x2 -= einsum("ijab->jiba", t2.aaaa)
    rdm_eb_cre_ov_aa -= einsum("wai,ijab->wjb", lu11.aa, x2)
    x3 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x3 += einsum("ijab->jiab", t2.bbbb)
    x3 -= einsum("ijab->jiba", t2.bbbb)
    rdm_eb_cre_ov_bb -= einsum("wai,ijab->wjb", lu11.bb, x3)
    x4 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum("ai,wja->wij", l1.aa, u11.aa)
    x40 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x40 += einsum("wij->wij", x4)
    rdm_eb_des_oo_aa = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    rdm_eb_des_oo_aa -= einsum("wij->wji", x4)
    del x4
    x5 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x5 += einsum("wia,baji->wjb", u11.bb, l2.abab)
    x8 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum("wia->wia", x5)
    x31 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x31 += einsum("wia->wia", x5)
    rdm_eb_des_vo_aa = np.zeros((nbos, nvir[0], nocc[0]), dtype=types[float])
    rdm_eb_des_vo_aa += einsum("wia->wai", x5)
    del x5
    x6 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x6 += einsum("abij->jiab", l2.aaaa)
    x6 += einsum("abij->jiba", l2.aaaa) * -1
    x7 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x7 += einsum("wia,ijab->wjb", u11.aa, x6)
    x8 += einsum("wia->wia", x7) * -1
    del x7
    rdm_eb_des_oo_aa += einsum("ia,wja->wij", t1.aa, x8) * -1
    rdm_eb_des_vv_aa = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    rdm_eb_des_vv_aa += einsum("ia,wib->wba", t1.aa, x8)
    del x8
    x37 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x37 += einsum("ijab,ijac->bc", t2.aaaa, x6)
    del x6
    x38 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x38 += einsum("ab->ba", x37) * -1
    x61 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x61 += einsum("ab->ba", x37) * -1
    del x37
    x9 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x9 += einsum("ai,ja->ij", l1.aa, t1.aa)
    x14 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x14 += einsum("ij->ij", x9)
    x39 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x39 += einsum("ij->ij", x9)
    x47 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x47 += einsum("ij->ij", x9)
    del x9
    x10 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x10 += einsum("wai,wja->ij", lu11.aa, u11.aa)
    x14 += einsum("ij->ij", x10)
    x39 += einsum("ij->ij", x10)
    x47 += einsum("ij->ij", x10)
    del x10
    x11 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x11 += einsum("abij,kjab->ik", l2.abab, t2.abab)
    x14 += einsum("ij->ij", x11)
    x39 += einsum("ij->ij", x11)
    x47 += einsum("ij->ij", x11)
    del x11
    x12 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x12 += einsum("ijab->jiab", t2.aaaa) * -1
    x12 += einsum("ijab->jiba", t2.aaaa)
    x13 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x13 += einsum("abij,ikba->jk", l2.aaaa, x12)
    x14 += einsum("ij->ij", x13) * -1
    x39 += einsum("ij->ij", x13) * -1
    del x13
    rdm_eb_des_ov_aa = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    rdm_eb_des_ov_aa += einsum("ij,wia->wja", x39, u11.aa) * -1
    del x39
    x47 += einsum("abij,ikba->jk", l2.aaaa, x12) * -1
    x48 += einsum("ia,ij->ja", t1.aa, x47)
    del x47
    x48 += einsum("ai,ijab->jb", l1.aa, x12) * -1
    del x12
    x14 += einsum("ij->ji", delta_oo.aa) * -1
    rdm_eb_des_oo_aa += einsum("w,ij->wji", s1, x14) * -1
    del x14
    x15 = np.zeros((nbos), dtype=types[float])
    x15 += einsum("ai,wia->w", l1.aa, u11.aa)
    x17 = np.zeros((nbos), dtype=types[float])
    x17 += einsum("w->w", x15)
    del x15
    x16 = np.zeros((nbos), dtype=types[float])
    x16 += einsum("ai,wia->w", l1.bb, u11.bb)
    x17 += einsum("w->w", x16)
    del x16
    rdm_eb_des_oo_aa += einsum("w,ij->wji", x17, delta_oo.aa)
    rdm_eb_des_oo_bb = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    rdm_eb_des_oo_bb += einsum("w,ij->wji", x17, delta_oo.bb)
    rdm_eb_des_ov_aa += einsum("w,ia->wia", x17, t1.aa)
    rdm_eb_des_ov_bb = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    rdm_eb_des_ov_bb += einsum("w,ia->wia", x17, t1.bb)
    del x17
    x18 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x18 += einsum("ai,wja->wij", l1.bb, u11.bb)
    x54 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x54 += einsum("wij->wij", x18)
    rdm_eb_des_oo_bb -= einsum("wij->wji", x18)
    del x18
    x19 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x19 += einsum("wia,abij->wjb", u11.aa, l2.abab)
    x22 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x22 += einsum("wia->wia", x19)
    x34 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x34 += einsum("wia->wia", x19)
    rdm_eb_des_vo_bb = np.zeros((nbos, nvir[1], nocc[1]), dtype=types[float])
    rdm_eb_des_vo_bb += einsum("wia->wai", x19)
    del x19
    x20 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x20 += einsum("abij->jiab", l2.bbbb)
    x20 += einsum("abij->jiba", l2.bbbb) * -1
    x21 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x21 += einsum("wia,ijab->wjb", u11.bb, x20)
    x22 += einsum("wia->wia", x21) * -1
    del x21
    rdm_eb_des_oo_bb += einsum("ia,wja->wij", t1.bb, x22) * -1
    rdm_eb_des_vv_bb = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    rdm_eb_des_vv_bb += einsum("ia,wib->wba", t1.bb, x22)
    del x22
    x51 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x51 += einsum("ijab,ijcb->ac", t2.bbbb, x20)
    del x20
    x52 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x52 += einsum("ab->ba", x51) * -1
    x62 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x62 += einsum("ab->ba", x51) * -1
    del x51
    x23 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x23 += einsum("ai,ja->ij", l1.bb, t1.bb)
    x28 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x28 += einsum("ij->ij", x23)
    x53 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x53 += einsum("ij->ij", x23)
    x59 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x59 += einsum("ij->ij", x23)
    del x23
    x24 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x24 += einsum("wai,wja->ij", lu11.bb, u11.bb)
    x28 += einsum("ij->ij", x24)
    x53 += einsum("ij->ij", x24)
    x59 += einsum("ij->ij", x24)
    del x24
    x25 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x25 += einsum("abij,ikab->jk", l2.abab, t2.abab)
    x28 += einsum("ij->ij", x25)
    x53 += einsum("ij->ij", x25)
    x59 += einsum("ij->ij", x25)
    del x25
    x26 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x26 += einsum("ijab->jiab", t2.bbbb)
    x26 += einsum("ijab->jiba", t2.bbbb) * -1
    x27 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x27 += einsum("abij,ikab->jk", l2.bbbb, x26)
    x28 += einsum("ij->ij", x27) * -1
    x53 += einsum("ij->ij", x27) * -1
    del x27
    rdm_eb_des_ov_bb += einsum("ij,wia->wja", x53, u11.bb) * -1
    del x53
    x59 += einsum("abij,ikab->jk", l2.bbbb, x26) * -1
    del x26
    x60 += einsum("ia,ij->ja", t1.bb, x59)
    del x59
    x28 += einsum("ij->ji", delta_oo.bb) * -1
    rdm_eb_des_oo_bb += einsum("w,ij->wji", s1, x28) * -1
    del x28
    x29 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x29 += einsum("abij->jiab", l2.aaaa)
    x29 -= einsum("abij->jiba", l2.aaaa)
    x30 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x30 += einsum("wia,ijab->wjb", u11.aa, x29)
    del x29
    x31 -= einsum("wia->wia", x30)
    x40 += einsum("ia,wja->wji", t1.aa, x31)
    rdm_eb_des_ov_aa -= einsum("ia,wij->wja", t1.aa, x40)
    del x40
    rdm_eb_des_ov_aa -= einsum("wia,ijab->wjb", x31, x2)
    del x2
    rdm_eb_des_ov_bb += einsum("wia,ijab->wjb", x31, t2.abab)
    del x31
    rdm_eb_des_vo_aa -= einsum("wia->wai", x30)
    del x30
    x32 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x32 += einsum("abij->jiab", l2.bbbb)
    x32 -= einsum("abij->jiba", l2.bbbb)
    x33 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x33 += einsum("wia,ijab->wjb", u11.bb, x32)
    del x32
    x34 -= einsum("wia->wia", x33)
    x54 += einsum("ia,wja->wji", t1.bb, x34)
    rdm_eb_des_ov_bb -= einsum("ia,wij->wja", t1.bb, x54)
    del x54
    rdm_eb_des_ov_aa += einsum("wia,jiba->wjb", x34, t2.abab)
    rdm_eb_des_ov_bb -= einsum("wia,ijab->wjb", x34, x3)
    del x34
    del x3
    rdm_eb_des_vo_bb -= einsum("wia->wai", x33)
    del x33
    x35 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x35 += einsum("wai,wib->ab", lu11.aa, u11.aa)
    x38 += einsum("ab->ab", x35)
    x61 += einsum("ab->ab", x35)
    del x35
    x36 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x36 += einsum("abij,ijcb->ac", l2.abab, t2.abab)
    x38 += einsum("ab->ab", x36)
    rdm_eb_des_ov_aa += einsum("ab,wia->wib", x38, u11.aa) * -1
    del x38
    x61 += einsum("ab->ab", x36)
    del x36
    x41 = np.zeros((nbos, nbos), dtype=types[float])
    x41 += einsum("wai,xia->wx", lu11.aa, u11.aa)
    x43 = np.zeros((nbos, nbos), dtype=types[float])
    x43 += einsum("wx->wx", x41)
    del x41
    x42 = np.zeros((nbos, nbos), dtype=types[float])
    x42 += einsum("wai,xia->wx", lu11.bb, u11.bb)
    x43 += einsum("wx->wx", x42)
    del x42
    rdm_eb_des_ov_aa += einsum("wx,wia->xia", x43, u11.aa)
    rdm_eb_des_ov_bb += einsum("wx,wia->xia", x43, u11.bb)
    del x43
    x44 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x44 += einsum("ia,abjk->jikb", t1.aa, l2.abab)
    x48 += einsum("ijab,ikjb->ka", t2.abab, x44)
    del x44
    x45 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x45 += einsum("ia,bajk->jkib", t1.aa, l2.aaaa)
    x46 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x46 += einsum("ijka->ijka", x45)
    x46 += einsum("ijka->jika", x45) * -1
    del x45
    x48 += einsum("ijab,jika->kb", t2.aaaa, x46) * -1
    del x46
    x48 += einsum("ia->ia", t1.aa) * -1
    x48 += einsum("w,wia->ia", ls1, u11.aa) * -1
    x48 += einsum("ai,jiba->jb", l1.bb, t2.abab) * -1
    rdm_eb_des_ov_aa += einsum("w,ia->wia", s1, x48) * -1
    del x48
    x49 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x49 += einsum("wai,wib->ab", lu11.bb, u11.bb)
    x52 += einsum("ab->ab", x49)
    x62 += einsum("ab->ab", x49)
    del x49
    x50 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x50 += einsum("abij,ijac->bc", l2.abab, t2.abab)
    x52 += einsum("ab->ab", x50)
    rdm_eb_des_ov_bb += einsum("ab,wia->wib", x52, u11.bb) * -1
    del x52
    x62 += einsum("ab->ab", x50)
    del x50
    x55 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x55 += einsum("ia,bajk->jkib", t1.bb, l2.abab)
    x60 += einsum("ijab,ijka->kb", t2.abab, x55)
    del x55
    x56 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x56 += einsum("ia,bajk->jkib", t1.bb, l2.bbbb)
    x57 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x57 += einsum("ijka->ijka", x56)
    x57 += einsum("ijka->jika", x56) * -1
    del x56
    x60 += einsum("ijab,ijkb->ka", t2.bbbb, x57) * -1
    del x57
    x58 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x58 += einsum("ijab->jiab", t2.bbbb) * -1
    x58 += einsum("ijab->jiba", t2.bbbb)
    x60 += einsum("ai,ijab->jb", l1.bb, x58) * -1
    del x58
    x60 += einsum("ia->ia", t1.bb) * -1
    x60 += einsum("w,wia->ia", ls1, u11.bb) * -1
    x60 += einsum("ai,ijab->jb", l1.aa, t2.abab) * -1
    rdm_eb_des_ov_bb += einsum("w,ia->wia", s1, x60) * -1
    del x60
    x61 += einsum("ai,ib->ab", l1.aa, t1.aa)
    rdm_eb_des_vv_aa += einsum("w,ab->wab", s1, x61)
    del x61
    x62 += einsum("ai,ib->ab", l1.bb, t1.bb)
    rdm_eb_des_vv_bb += einsum("w,ab->wab", s1, x62)
    del x62
    rdm_eb_cre_oo_aa += einsum("w,ij->wji", ls1, delta_oo.aa)
    rdm_eb_cre_oo_bb += einsum("w,ij->wji", ls1, delta_oo.bb)
    rdm_eb_cre_ov_aa += einsum("w,ia->wia", ls1, t1.aa)
    rdm_eb_cre_ov_aa += einsum("wai,jiba->wjb", lu11.bb, t2.abab)
    rdm_eb_cre_ov_bb += einsum("w,ia->wia", ls1, t1.bb)
    rdm_eb_cre_ov_bb += einsum("wai,ijab->wjb", lu11.aa, t2.abab)
    rdm_eb_cre_vo_aa = np.zeros((nbos, nvir[0], nocc[0]), dtype=types[float])
    rdm_eb_cre_vo_aa += einsum("wai->wai", lu11.aa)
    rdm_eb_cre_vo_bb = np.zeros((nbos, nvir[1], nocc[1]), dtype=types[float])
    rdm_eb_cre_vo_bb += einsum("wai->wai", lu11.bb)
    rdm_eb_cre_vv_aa = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    rdm_eb_cre_vv_aa += einsum("ia,wbi->wba", t1.aa, lu11.aa)
    rdm_eb_cre_vv_bb = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    rdm_eb_cre_vv_bb += einsum("ia,wbi->wba", t1.bb, lu11.bb)
    rdm_eb_des_ov_aa += einsum("wia->wia", u11.aa)
    rdm_eb_des_ov_bb += einsum("wia->wia", u11.bb)
    rdm_eb_des_vo_aa += einsum("w,ai->wai", s1, l1.aa)
    rdm_eb_des_vo_bb += einsum("w,ai->wai", s1, l1.bb)
    rdm_eb_des_vv_aa += einsum("ai,wib->wab", l1.aa, u11.aa)
    rdm_eb_des_vv_bb += einsum("ai,wib->wab", l1.bb, u11.bb)

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

