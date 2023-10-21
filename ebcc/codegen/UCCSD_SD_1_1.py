# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, **kwargs):
    # Energy
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x0 += einsum("iajb->jiab", v.bbbb.ovov) * -1
    x0 += einsum("iajb->jiba", v.bbbb.ovov)
    x1 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x1 += einsum("ijab->jiba", t2.bbbb)
    x1 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    e_cc = 0
    e_cc += einsum("ijab,ijba->", x0, x1) * -0.5
    del x0
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum("iajb->jiab", v.aaaa.ovov)
    x2 += einsum("iajb->jiba", v.aaaa.ovov) * -1
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x3 += einsum("ijab->jiba", t2.aaaa)
    x3 += einsum("ia,jb->ijba", t1.aa, t1.aa) * -1
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
    e_cc += einsum("wia,wia->", g.bb.bov, u11.bb)
    e_cc += einsum("ia,ia->", f.bb.ov, t1.bb)
    e_cc += einsum("ia,ia->", f.aa.ov, t1.aa)

    return e_cc

def update_amps(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, **kwargs):
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

    # T1, T2, S1, S2 and U11 amplitudes
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x0 += einsum("ia,jakb->ikjb", t1.aa, v.aaaa.ovov)
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x1 += einsum("ijka->ijka", x0) * -1
    x1 += einsum("ijka->ikja", x0)
    x98 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x98 -= einsum("ijka->ijka", x0)
    x98 += einsum("ijka->ikja", x0)
    x124 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x124 += einsum("ia,jkla->jilk", t1.aa, x0)
    x125 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x125 += einsum("ia,jkli->jkla", t1.aa, x124)
    x126 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x126 += einsum("ia,jkib->jkab", t1.aa, x125)
    del x125
    x133 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x133 += einsum("ijab->ijab", x126)
    del x126
    x141 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x141 += einsum("ijab,klji->lkab", t2.aaaa, x124)
    del x124
    x145 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x145 += einsum("ijab->ijab", x141)
    del x141
    x284 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x284 += einsum("ijka->jkia", x0) * -1
    x284 += einsum("ijka->kjia", x0)
    del x0
    x1 += einsum("ijka->jika", v.aaaa.ooov)
    x1 += einsum("ijka->jkia", v.aaaa.ooov) * -1
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    t1new_aa += einsum("ijab,kjia->kb", t2.aaaa, x1) * -1
    del x1
    x2 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum("iabc->ibac", v.aaaa.ovvv)
    x2 += einsum("iabc->ibca", v.aaaa.ovvv) * -1
    x31 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x31 += einsum("ia,ibca->bc", t1.aa, x2)
    x32 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x32 += einsum("ab->ab", x31) * -1
    x280 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x280 += einsum("ab->ab", x31) * -1
    x321 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x321 += einsum("ab->ab", x31) * -1
    del x31
    x64 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x64 += einsum("ia,jbac->ijbc", t1.aa, x2)
    t1new_aa += einsum("ijab,icba->jc", t2.aaaa, x2) * -1
    del x2
    x3 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x3 += einsum("ia,jakb->ijkb", t1.aa, v.aabb.ovov)
    x4 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x4 += einsum("ijka->jika", x3)
    x79 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x79 += einsum("ijab,kljb->kila", t2.abab, x3)
    del x3
    x83 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x83 += einsum("ijka->ikja", x79)
    del x79
    x4 += einsum("ijka->ijka", v.aabb.ooov)
    x267 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x267 += einsum("ijab,iklb->kjla", t2.abab, x4)
    x274 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x274 += einsum("ijka->ikja", x267) * -1
    del x267
    t1new_aa += einsum("ijab,ikjb->ka", t2.abab, x4) * -1
    x5 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x5 += einsum("w,wia->ia", s1, g.aa.bov)
    x9 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x9 += einsum("ia->ia", x5)
    x102 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x102 += einsum("ia->ia", x5)
    x300 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x300 += einsum("ia->ia", x5)
    x322 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x322 += einsum("ia->ia", x5)
    del x5
    x6 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum("ia,jbia->jb", t1.bb, v.aabb.ovov)
    x9 += einsum("ia->ia", x6)
    x61 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x61 += einsum("ia,ja->ij", t1.aa, x6)
    x62 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x62 += einsum("ij,kjab->ikab", x61, t2.aaaa)
    del x61
    x87 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x87 += einsum("ijab->ijab", x62) * -1
    del x62
    x80 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x80 += einsum("ia,jkba->jkib", x6, t2.aaaa)
    x83 += einsum("ijka->ikja", x80) * -1
    del x80
    x300 += einsum("ia->ia", x6)
    x320 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x320 += einsum("ia->ia", x6)
    del x6
    x7 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x7 += einsum("iajb->jiab", v.aaaa.ovov)
    x7 += einsum("iajb->jiba", v.aaaa.ovov) * -1
    x8 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum("ia,ijab->jb", t1.aa, x7)
    x9 += einsum("ia->ia", x8) * -1
    x320 += einsum("ia->ia", x8) * -1
    del x8
    x321 += einsum("ia,ib->ab", t1.aa, x320) * -1
    del x320
    x70 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x70 += einsum("ijab,ijac->bc", t2.aaaa, x7)
    x73 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x73 += einsum("ab->ab", x70) * -1
    del x70
    x279 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x279 += einsum("ijab,ijca->bc", t2.aaaa, x7)
    x280 += einsum("ab->ab", x279) * -1
    x321 += einsum("ab->ab", x279) * -1
    del x279
    x315 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x315 += einsum("wia,ijab->wjb", u11.aa, x7)
    x316 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x316 += einsum("wia->wia", x315) * -1
    del x315
    x9 += einsum("ia->ia", f.aa.ov)
    x27 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x27 += einsum("ia,ja->ij", t1.aa, x9)
    x28 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x28 += einsum("ij->ji", x27)
    del x27
    x288 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x288 += einsum("ia,jkab->jikb", x9, t2.abab)
    x291 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x291 += einsum("ijka->jika", x288)
    del x288
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    t1new_bb += einsum("ia,ijab->jb", x9, t2.abab)
    x10 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x10 += einsum("ijab->jiab", t2.aaaa)
    x10 -= einsum("ijab->jiba", t2.aaaa)
    x112 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x112 += einsum("wia,ijab->wjb", g.aa.bov, x10)
    x114 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x114 -= einsum("wia->wia", x112)
    del x112
    t1new_aa += einsum("ia,ijab->jb", x9, x10) * -1
    del x9
    x11 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x11 += einsum("w,wia->ia", s1, g.bb.bov)
    x15 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x15 += einsum("ia->ia", x11)
    x159 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x159 += einsum("ia->ia", x11)
    x301 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x301 += einsum("ia->ia", x11)
    x328 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x328 += einsum("ia->ia", x11)
    del x11
    x12 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x12 += einsum("ia,iajb->jb", t1.aa, v.aabb.ovov)
    x15 += einsum("ia->ia", x12)
    x201 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x201 += einsum("ia,jkab->kjib", x12, t2.bbbb)
    x205 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x205 += einsum("ijka->ikja", x201) * -1
    del x201
    x216 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x216 += einsum("ia,ja->ij", t1.bb, x12)
    x217 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x217 += einsum("ij->ij", x216)
    del x216
    x301 += einsum("ia->ia", x12)
    x326 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x326 += einsum("ia->ia", x12)
    del x12
    x13 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x13 += einsum("iajb->jiab", v.bbbb.ovov) * -1
    x13 += einsum("iajb->jiba", v.bbbb.ovov)
    x14 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x14 += einsum("ia,ijba->jb", t1.bb, x13)
    x15 += einsum("ia->ia", x14) * -1
    x326 += einsum("ia->ia", x14) * -1
    del x14
    x327 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x327 += einsum("ia,ib->ab", t1.bb, x326) * -1
    del x326
    x49 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x49 += einsum("ijab,ikba->kj", t2.bbbb, x13)
    x53 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x53 += einsum("ij->ij", x49) * -1
    del x49
    x237 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x237 += einsum("ijab,jkcb->ikac", t2.abab, x13)
    x238 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x238 += einsum("ijab->ijab", x237) * -1
    del x237
    x318 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x318 += einsum("wia,ijba->wjb", u11.bb, x13)
    del x13
    x319 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x319 += einsum("wia->wia", x318) * -1
    del x318
    x15 += einsum("ia->ia", f.bb.ov)
    x52 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x52 += einsum("ia,ja->ij", t1.bb, x15)
    x53 += einsum("ij->ji", x52)
    del x52
    x268 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x268 += einsum("ia,jkba->jkib", x15, t2.abab)
    x274 += einsum("ijka->ikja", x268)
    del x268
    t1new_aa += einsum("ia,jiba->jb", x15, t2.abab)
    x16 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x16 += einsum("iabj->ijba", v.aaaa.ovvo)
    x16 -= einsum("ijab->ijab", v.aaaa.oovv)
    x104 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x104 += einsum("ia,jkba->ijkb", t1.aa, x16)
    x105 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x105 -= einsum("ijka->jika", x104)
    del x104
    t1new_aa += einsum("ia,ijba->jb", t1.aa, x16)
    u11new_aa = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    u11new_aa += einsum("wia,ijba->wjb", u11.aa, x16)
    x17 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x17 += einsum("ia,wja->wji", t1.aa, g.aa.bov)
    x18 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x18 += einsum("wij->wij", x17)
    del x17
    x18 += einsum("wij->wij", g.aa.boo)
    x113 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x113 += einsum("ia,wij->wja", t1.aa, x18)
    x114 -= einsum("wia->wia", x113)
    del x113
    t1new_aa -= einsum("wia,wij->ja", u11.aa, x18)
    del x18
    x19 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x19 += einsum("w,wij->ij", s1, g.aa.boo)
    x28 += einsum("ij->ij", x19)
    x108 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x108 += einsum("ij->ji", x19)
    del x19
    x20 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x20 += einsum("wia,wja->ij", g.aa.bov, u11.aa)
    x28 += einsum("ij->ij", x20)
    x121 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x121 += einsum("ij->ij", x20)
    del x20
    x21 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x21 += einsum("ia,jkia->jk", t1.bb, v.aabb.ooov)
    x28 += einsum("ij->ij", x21)
    x121 += einsum("ij->ij", x21)
    del x21
    x122 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x122 += einsum("ij,ikab->kjab", x121, t2.aaaa)
    del x121
    x123 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x123 -= einsum("ijab->ijba", x122)
    del x122
    x22 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x22 += einsum("ijab,kajb->ik", t2.abab, v.aabb.ovov)
    x28 += einsum("ij->ji", x22)
    x85 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x85 += einsum("ij->ji", x22)
    del x22
    x23 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x23 += einsum("iajb->jiab", v.aaaa.ovov) * -1
    x23 += einsum("iajb->jiba", v.aaaa.ovov)
    x24 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x24 += einsum("ijab,ikba->jk", t2.aaaa, x23)
    del x23
    x28 += einsum("ij->ji", x24) * -1
    x85 += einsum("ij->ji", x24) * -1
    del x24
    x25 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x25 += einsum("ijka->ikja", v.aaaa.ooov)
    x25 += einsum("ijka->kija", v.aaaa.ooov) * -1
    x26 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x26 += einsum("ia,ijka->jk", t1.aa, x25)
    x28 += einsum("ij->ij", x26) * -1
    x85 += einsum("ij->ij", x26) * -1
    del x26
    x86 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x86 += einsum("ij,ikab->kjab", x85, t2.aaaa)
    del x85
    x87 += einsum("ijab->ijba", x86)
    del x86
    x323 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x323 += einsum("wia,ijka->wjk", u11.aa, x25) * -1
    del x25
    x28 += einsum("ij->ij", f.aa.oo)
    x294 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x294 += einsum("ij,ikab->jkab", x28, t2.abab)
    t2new_baba = np.zeros((nocc[1], nocc[0], nvir[1], nvir[0]), dtype=types[float])
    t2new_baba += einsum("ijab->jiba", x294) * -1
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    t2new_abab += einsum("ijab->ijab", x294) * -1
    del x294
    t1new_aa += einsum("ia,ij->ja", t1.aa, x28) * -1
    u11new_aa += einsum("ij,wia->wja", x28, u11.aa) * -1
    del x28
    x29 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x29 += einsum("w,wab->ab", s1, g.aa.bvv)
    x32 += einsum("ab->ab", x29)
    x116 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x116 += einsum("ab->ab", x29)
    x280 += einsum("ab->ab", x29)
    x321 += einsum("ab->ab", x29)
    del x29
    x30 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x30 += einsum("ia,iabc->bc", t1.bb, v.bbaa.ovvv)
    x32 += einsum("ab->ab", x30)
    x119 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x119 -= einsum("ab->ba", x30)
    x280 += einsum("ab->ab", x30)
    x321 += einsum("ab->ab", x30)
    del x30
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
    del x35
    x36 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x36 += einsum("ia,jakb->ikjb", t1.bb, v.bbbb.ovov)
    x37 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x37 += einsum("ijka->ijka", x36) * -1
    x37 += einsum("ijka->ikja", x36)
    x155 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x155 -= einsum("ijka->ijka", x36)
    x155 += einsum("ijka->ikja", x36)
    x181 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x181 += einsum("ia,jkla->jilk", t1.bb, x36)
    x182 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x182 += einsum("ia,jkli->jkla", t1.bb, x181)
    x183 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x183 += einsum("ia,jkib->jkab", t1.bb, x182)
    del x182
    x187 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x187 += einsum("ijab->ijab", x183)
    del x183
    x223 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x223 += einsum("ijkl->ijkl", x181)
    del x181
    x264 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x264 += einsum("ijka->jkia", x36)
    x264 += einsum("ijka->kjia", x36) * -1
    del x36
    x37 += einsum("ijka->jika", v.bbbb.ooov)
    x37 += einsum("ijka->jkia", v.bbbb.ooov) * -1
    t1new_bb += einsum("ijab,kjia->kb", t2.bbbb, x37) * -1
    del x37
    x38 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x38 += einsum("iabc->ibac", v.bbbb.ovvv)
    x38 += einsum("iabc->ibca", v.bbbb.ovvv) * -1
    x209 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x209 += einsum("ia,ibac->bc", t1.bb, x38)
    x210 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x210 += einsum("ab->ab", x209) * -1
    del x209
    t1new_bb += einsum("ijab,icba->jc", t2.bbbb, x38) * -1
    del x38
    x39 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x39 += einsum("ia,jbka->jikb", t1.bb, v.aabb.ovov)
    x40 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x40 += einsum("ijka->ikja", x39)
    x200 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x200 += einsum("ijab,ikla->kjlb", t2.abab, x39)
    x205 += einsum("ijka->ikja", x200)
    del x200
    x255 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x255 += einsum("ijka->ikja", x39)
    del x39
    x40 += einsum("iajk->ijka", v.aabb.ovoo)
    x269 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x269 += einsum("ia,jkla->ijkl", t1.aa, x40)
    x270 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x270 += einsum("ijkl->jikl", x269)
    del x269
    x287 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x287 += einsum("ijab,kjla->iklb", t2.abab, x40)
    x291 += einsum("ijka->jika", x287) * -1
    del x287
    t1new_bb += einsum("ijab,ijka->kb", t2.abab, x40) * -1
    x41 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x41 += einsum("ijab->jiab", t2.bbbb) * -1
    x41 += einsum("ijab->jiba", t2.bbbb)
    x194 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x194 += einsum("iajb,jkcb->ikac", v.aabb.ovov, x41)
    x195 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x195 += einsum("ijab,ikac->jkbc", t2.abab, x194) * -1
    del x194
    x219 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x219 += einsum("ijab->ijab", x195) * -1
    del x195
    x286 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x286 += einsum("ijka,klba->ijlb", x4, x41)
    del x4
    x291 += einsum("ijka->ijka", x286) * -1
    del x286
    t1new_bb += einsum("ia,ijba->jb", x15, x41) * -1
    del x15
    x42 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x42 += einsum("iabj->ijba", v.bbbb.ovvo)
    x42 -= einsum("ijab->ijab", v.bbbb.oovv)
    x161 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x161 += einsum("ia,jkba->ijkb", t1.bb, x42)
    x162 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x162 -= einsum("ijka->jika", x161)
    del x161
    t1new_bb += einsum("ia,ijba->jb", t1.bb, x42)
    u11new_bb = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    u11new_bb += einsum("wia,ijba->wjb", u11.bb, x42)
    x43 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x43 += einsum("ia,wja->wji", t1.bb, g.bb.bov)
    x44 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x44 += einsum("wij->wij", x43)
    del x43
    x44 += einsum("wij->wij", g.bb.boo)
    x170 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x170 += einsum("ia,wij->wja", t1.bb, x44)
    x171 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x171 -= einsum("wia->wia", x170)
    del x170
    t1new_bb -= einsum("wia,wij->ja", u11.bb, x44)
    del x44
    x45 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x45 += einsum("w,wij->ij", s1, g.bb.boo)
    x53 += einsum("ij->ij", x45)
    x165 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x165 += einsum("ij->ji", x45)
    del x45
    x46 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x46 += einsum("wia,wja->ij", g.bb.bov, u11.bb)
    x53 += einsum("ij->ij", x46)
    x178 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x178 += einsum("ij->ij", x46)
    del x46
    x47 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x47 += einsum("ia,iajk->jk", t1.aa, v.aabb.ovoo)
    x53 += einsum("ij->ij", x47)
    x178 += einsum("ij->ij", x47)
    del x47
    x179 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x179 += einsum("ij,ikab->kjab", x178, t2.bbbb)
    del x178
    x180 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x180 -= einsum("ijab->ijba", x179)
    del x179
    x48 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x48 += einsum("ijab,iakb->jk", t2.abab, v.aabb.ovov)
    x53 += einsum("ij->ji", x48)
    x217 += einsum("ij->ij", x48)
    del x48
    x218 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x218 += einsum("ij,jkab->ikab", x217, t2.bbbb)
    del x217
    x219 += einsum("ijab->ijba", x218) * -1
    del x218
    x50 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x50 += einsum("ijka->ikja", v.bbbb.ooov) * -1
    x50 += einsum("ijka->kija", v.bbbb.ooov)
    x51 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x51 += einsum("ia,jika->jk", t1.bb, x50)
    x53 += einsum("ij->ij", x51) * -1
    del x51
    x329 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x329 += einsum("wia,jika->wjk", u11.bb, x50) * -1
    del x50
    x53 += einsum("ij->ij", f.bb.oo)
    x293 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x293 += einsum("ij,kiab->kjab", x53, t2.abab)
    t2new_baba += einsum("ijab->jiba", x293) * -1
    t2new_abab += einsum("ijab->ijab", x293) * -1
    del x293
    t1new_bb += einsum("ia,ij->ja", t1.bb, x53) * -1
    u11new_bb += einsum("ij,wia->wja", x53, u11.bb) * -1
    del x53
    x54 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x54 += einsum("w,wab->ab", s1, g.bb.bvv)
    x58 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x58 += einsum("ab->ab", x54)
    x173 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x173 += einsum("ab->ab", x54)
    x277 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x277 += einsum("ab->ab", x54)
    x327 += einsum("ab->ab", x54)
    del x54
    x55 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x55 += einsum("ia,iabc->bc", t1.aa, v.aabb.ovvv)
    x58 += einsum("ab->ab", x55)
    x176 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x176 -= einsum("ab->ba", x55)
    x277 += einsum("ab->ab", x55)
    x327 += einsum("ab->ab", x55)
    del x55
    x56 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x56 += einsum("iabc->ibac", v.bbbb.ovvv) * -1
    x56 += einsum("iabc->ibca", v.bbbb.ovvv)
    x57 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x57 += einsum("ia,ibac->bc", t1.bb, x56)
    x58 += einsum("ab->ab", x57) * -1
    x277 += einsum("ab->ab", x57) * -1
    x327 += einsum("ab->ab", x57) * -1
    del x57
    x192 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x192 += einsum("ia,jbca->ijbc", t1.bb, x56)
    del x56
    x193 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x193 += einsum("ijab,jkcb->kica", x192, x41) * -1
    del x192
    x219 += einsum("ijab->jiab", x193) * -1
    del x193
    x58 += einsum("ab->ab", f.bb.vv)
    t1new_bb += einsum("ia,ba->ib", t1.bb, x58)
    del x58
    x59 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x59 += einsum("ia,jkla->ijlk", t1.aa, v.aaaa.ooov)
    x60 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x60 += einsum("ijab,kijl->klab", t2.aaaa, x59)
    x87 += einsum("ijab->ijab", x60)
    del x60
    x75 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x75 += einsum("ia,jkil->jkla", t1.aa, x59)
    del x59
    x83 += einsum("ijka->ijka", x75)
    del x75
    x63 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x63 += einsum("ijab->jiab", t2.aaaa)
    x63 += einsum("ijab->jiba", t2.aaaa) * -1
    x65 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x65 += einsum("ijab,kica->jkbc", x63, x64) * -1
    del x64
    x87 += einsum("ijab->jiab", x65) * -1
    del x65
    x236 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x236 += einsum("iajb,ikac->kjcb", v.aabb.ovov, x63)
    x238 += einsum("ijab->ijab", x236) * -1
    del x236
    x239 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x239 += einsum("ijab,ikac->kjcb", x63, x7)
    del x7
    x241 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x241 += einsum("ijab->ijba", x239)
    del x239
    x266 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x266 += einsum("ijka,ilab->ljkb", x40, x63)
    del x40
    del x63
    x274 += einsum("ijka->ijka", x266) * -1
    del x266
    x66 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x66 += einsum("ijab,kcjb->ikac", t2.abab, v.aabb.ovov)
    x241 += einsum("ijab->jiab", x66)
    x67 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x67 += einsum("ijab->jiab", t2.aaaa) * -1
    x67 += einsum("ijab->jiba", t2.aaaa)
    x68 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x68 += einsum("ijab,jkbc->ikac", x66, x67)
    del x66
    x87 += einsum("ijab->jiba", x68) * -1
    del x68
    x69 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x69 += einsum("ijab,icjb->ac", t2.abab, v.aabb.ovov)
    x73 += einsum("ab->ab", x69)
    x280 += einsum("ab->ab", x69) * -1
    x321 += einsum("ab->ab", x69) * -1
    del x69
    x71 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x71 += einsum("iabc->ibac", v.aaaa.ovvv) * -1
    x71 += einsum("iabc->ibca", v.aaaa.ovvv)
    x72 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x72 += einsum("ia,ibca->bc", t1.aa, x71)
    x73 += einsum("ab->ab", x72) * -1
    del x72
    x74 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x74 += einsum("ab,ijbc->ijca", x73, t2.aaaa)
    del x73
    x87 += einsum("ijab->jiab", x74)
    del x74
    x240 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x240 += einsum("ia,jbca->ijbc", t1.aa, x71)
    del x71
    x241 += einsum("ijab->jiab", x240) * -1
    del x240
    x76 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x76 += einsum("ijab,kacb->ijkc", t2.aaaa, v.aaaa.ovvv)
    x83 += einsum("ijka->ikja", x76)
    del x76
    x77 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x77 += einsum("ia,jbca->ijcb", t1.aa, v.aaaa.ovvv)
    x78 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x78 += einsum("ia,jkba->ijkb", t1.aa, x77)
    del x77
    x83 += einsum("ijka->ikja", x78)
    del x78
    x81 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x81 += einsum("ijka->ikja", v.aaaa.ooov) * -1
    x81 += einsum("ijka->kija", v.aaaa.ooov)
    x82 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x82 += einsum("ijab,iklb->jkla", x67, x81)
    del x81
    del x67
    x83 += einsum("ijka->ijka", x82)
    del x82
    x84 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x84 += einsum("ia,jikb->jkab", t1.aa, x83)
    del x83
    x87 += einsum("ijab->ijab", x84)
    del x84
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    t2new_aaaa += einsum("ijab->ijab", x87) * -1
    t2new_aaaa += einsum("ijab->ijba", x87)
    t2new_aaaa += einsum("ijab->jiab", x87)
    t2new_aaaa += einsum("ijab->jiba", x87) * -1
    del x87
    x88 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x88 += einsum("ia,bjca->ijbc", t1.aa, v.aaaa.vovv)
    x123 -= einsum("ijab->ijab", x88)
    del x88
    x89 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x89 += einsum("ijab,jbck->ikac", t2.abab, v.bbaa.ovvo)
    x123 += einsum("ijab->ijab", x89)
    del x89
    x90 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x90 += einsum("ia,jbca->ijcb", t1.aa, v.bbaa.ovvv)
    x91 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x91 += einsum("ijab,kjcb->kiac", t2.abab, x90)
    x123 -= einsum("ijab->ijab", x91)
    del x91
    x238 += einsum("ijab->ijab", x90)
    del x90
    x92 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x92 -= einsum("ijab->jiab", t2.aaaa)
    x92 += einsum("ijab->jiba", t2.aaaa)
    x93 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x93 += einsum("ijab,ikcb->jkac", x16, x92)
    del x16
    x123 -= einsum("ijab->jiba", x93)
    del x93
    x99 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x99 += einsum("ijab,klib->jkla", x92, x98)
    del x98
    del x92
    x105 += einsum("ijka->kjia", x99)
    del x99
    x94 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x94 -= einsum("iajb->jiab", v.aaaa.ovov)
    x94 += einsum("iajb->jiba", v.aaaa.ovov)
    x95 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x95 += einsum("ijab,ikcb->jkac", t2.aaaa, x94)
    x96 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x96 -= einsum("ijab,kica->jkbc", t2.aaaa, x95)
    x123 -= einsum("ijab->ijab", x96)
    del x96
    x129 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x129 -= einsum("ijab,kicb->jkac", t2.aaaa, x95)
    del x95
    x133 += einsum("ijab->jiba", x129)
    del x129
    x127 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x127 += einsum("ijab,ikca->jkbc", t2.aaaa, x94)
    del x94
    x128 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x128 -= einsum("ijab,kica->jkbc", t2.aaaa, x127)
    del x127
    x133 += einsum("ijab->ijab", x128)
    del x128
    x97 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x97 += einsum("ijab,kljb->ikla", t2.abab, v.aabb.ooov)
    x105 += einsum("ijka->jika", x97)
    del x97
    x100 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x100 += einsum("iajb->jiab", v.aaaa.ovov)
    x100 -= einsum("iajb->jiba", v.aaaa.ovov)
    x101 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x101 += einsum("ia,ijab->jb", t1.aa, x100)
    x102 -= einsum("ia->ia", x101)
    x300 -= einsum("ia->ia", x101)
    del x101
    x229 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x229 += einsum("ijab,ikac->kjcb", t2.abab, x100)
    x230 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x230 -= einsum("ijab,ikac->kjcb", t2.abab, x229)
    del x229
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    t2new_bbbb -= einsum("ijab->ijba", x230)
    t2new_bbbb += einsum("ijab->jiba", x230)
    del x230
    x311 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x311 -= einsum("wia,ijab->wjb", u11.aa, x100)
    del x100
    s2new = np.zeros((nbos, nbos), dtype=types[float])
    s2new += einsum("wia,xia->xw", u11.aa, x311)
    del x311
    x102 += einsum("ia->ia", f.aa.ov)
    x103 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x103 += einsum("ia,jkab->jkib", x102, t2.aaaa)
    x105 += einsum("ijka->kjia", x103)
    del x103
    x107 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x107 += einsum("ia,ja->ij", t1.aa, x102)
    del x102
    x108 += einsum("ij->ij", x107)
    del x107
    x105 -= einsum("ijak->ijka", v.aaaa.oovo)
    x106 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x106 += einsum("ia,ijkb->jkab", t1.aa, x105)
    del x105
    x123 += einsum("ijab->ijab", x106)
    del x106
    x108 += einsum("ij->ji", f.aa.oo)
    x109 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x109 += einsum("ij,jkab->kiab", x108, t2.aaaa)
    del x108
    x123 += einsum("ijab->jiba", x109)
    del x109
    x110 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x110 += einsum("ia,wba->wib", t1.aa, g.aa.bvv)
    x114 += einsum("wia->wia", x110)
    del x110
    x111 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x111 += einsum("wia,jiba->wjb", g.bb.bov, t2.abab)
    x114 += einsum("wia->wia", x111)
    del x111
    x114 += einsum("wai->wia", g.aa.bvo)
    x115 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x115 += einsum("wia,wjb->ijab", u11.aa, x114)
    x123 += einsum("ijab->jiba", x115)
    del x115
    x296 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x296 += einsum("wia,wjb->jiba", u11.bb, x114)
    del x114
    t2new_baba += einsum("ijab->jiba", x296)
    t2new_abab += einsum("ijab->ijab", x296)
    del x296
    x116 += einsum("ab->ab", f.aa.vv)
    x117 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x117 += einsum("ab,ijbc->ijca", x116, t2.aaaa)
    del x116
    x123 -= einsum("ijab->jiba", x117)
    del x117
    x118 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x118 += einsum("wia,wib->ab", g.aa.bov, u11.aa)
    x119 += einsum("ab->ab", x118)
    x120 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x120 += einsum("ab,ijac->ijcb", x119, t2.aaaa)
    del x119
    x123 -= einsum("ijab->jiab", x120)
    del x120
    t2new_aaaa += einsum("ijab->ijab", x123)
    t2new_aaaa -= einsum("ijab->ijba", x123)
    t2new_aaaa -= einsum("ijab->jiab", x123)
    t2new_aaaa += einsum("ijab->jiba", x123)
    del x123
    x280 += einsum("ab->ba", x118) * -1
    x321 += einsum("ab->ba", x118) * -1
    del x118
    x130 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x130 -= einsum("iajb->jiab", v.bbbb.ovov)
    x130 += einsum("iajb->jiba", v.bbbb.ovov)
    x131 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x131 += einsum("ijab,jkcb->ikac", t2.abab, x130)
    x132 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x132 -= einsum("ijab,kjcb->ikac", t2.abab, x131)
    del x131
    x133 += einsum("ijab->ijab", x132)
    del x132
    t2new_aaaa += einsum("ijab->ijab", x133)
    t2new_aaaa -= einsum("ijab->ijba", x133)
    del x133
    x152 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x152 += einsum("ijab,ikcb->jkac", t2.bbbb, x130)
    x153 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x153 -= einsum("ijab,kica->jkbc", t2.bbbb, x152)
    x180 -= einsum("ijab->ijab", x153)
    del x153
    x186 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x186 -= einsum("ijab,kicb->jkac", t2.bbbb, x152)
    del x152
    x187 += einsum("ijab->ijab", x186)
    del x186
    x158 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x158 += einsum("ia,ijba->jb", t1.bb, x130)
    x159 -= einsum("ia->ia", x158)
    x301 -= einsum("ia->ia", x158)
    del x158
    x184 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x184 += einsum("ijab,ikca->jkbc", t2.bbbb, x130)
    x185 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x185 -= einsum("ijab,kica->jkbc", t2.bbbb, x184)
    del x184
    x187 += einsum("ijab->jiba", x185)
    del x185
    t2new_bbbb += einsum("ijab->ijab", x187)
    t2new_bbbb -= einsum("ijab->ijba", x187)
    del x187
    x312 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x312 -= einsum("wia,ijba->wjb", u11.bb, x130)
    del x130
    s2new += einsum("wia,xia->xw", u11.bb, x312)
    del x312
    x134 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x134 += einsum("ijab,ikjl->klab", t2.aaaa, v.aaaa.oooo)
    x136 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x136 += einsum("ijab->jiba", x134)
    del x134
    x135 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x135 += einsum("ijab,cadb->ijcd", t2.aaaa, v.aaaa.vvvv)
    x136 += einsum("ijab->jiba", x135)
    del x135
    t2new_aaaa += einsum("ijab->ijba", x136) * -1
    t2new_aaaa += einsum("ijab->ijab", x136)
    del x136
    x137 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x137 += einsum("ia,jkil->jkla", t1.aa, v.aaaa.oooo)
    x138 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x138 += einsum("ia,ijkb->jkab", t1.aa, x137)
    del x137
    x145 += einsum("ijab->ijab", x138)
    del x138
    x139 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x139 += einsum("ia,bacd->icbd", t1.aa, v.aaaa.vvvv)
    x140 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x140 += einsum("ia,jbca->ijbc", t1.aa, x139)
    del x139
    x145 += einsum("ijab->ijab", x140)
    del x140
    x142 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x142 += einsum("ijab,kalb->ijkl", t2.aaaa, v.aaaa.ovov)
    x143 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x143 += einsum("ijab->jiba", t2.aaaa)
    x143 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    x144 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x144 += einsum("ijkl,klab->ijab", x142, x143)
    del x142
    del x143
    x145 += einsum("ijab->jiba", x144)
    del x144
    t2new_aaaa += einsum("ijab->ijab", x145)
    t2new_aaaa += einsum("ijab->ijba", x145) * -1
    del x145
    x146 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x146 += einsum("ia,bjca->ijbc", t1.bb, v.bbbb.vovv)
    x180 -= einsum("ijab->ijab", x146)
    del x146
    x147 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x147 += einsum("ijab,iack->jkbc", t2.abab, v.aabb.ovvo)
    x180 += einsum("ijab->ijab", x147)
    del x147
    x148 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x148 += einsum("ia,jbca->jibc", t1.bb, v.aabb.ovvv)
    x149 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x149 += einsum("ijab,ikac->kjbc", t2.abab, x148)
    x180 -= einsum("ijab->ijab", x149)
    del x149
    x243 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x243 += einsum("ijab->ijab", x148)
    x289 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x289 += einsum("ijab->ijab", x148)
    del x148
    x150 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x150 -= einsum("ijab->jiab", t2.bbbb)
    x150 += einsum("ijab->jiba", t2.bbbb)
    x151 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x151 += einsum("ijab,ikcb->kjca", x150, x42)
    del x42
    x180 -= einsum("ijab->jiba", x151)
    del x151
    x169 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x169 += einsum("wia,ijba->wjb", g.bb.bov, x150)
    del x150
    x171 -= einsum("wia->wia", x169)
    del x169
    x154 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x154 += einsum("ijab,iakl->jklb", t2.abab, v.aabb.ovoo)
    x162 += einsum("ijka->jika", x154)
    del x154
    x156 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x156 += einsum("ijab->jiab", t2.bbbb)
    x156 -= einsum("ijab->jiba", t2.bbbb)
    x157 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x157 += einsum("ijka,klab->ijlb", x155, x156)
    del x155
    x162 += einsum("ijka->jika", x157)
    del x157
    x159 += einsum("ia->ia", f.bb.ov)
    x160 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x160 += einsum("ia,jkab->jkib", x159, t2.bbbb)
    x162 += einsum("ijka->kjia", x160)
    del x160
    x164 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x164 += einsum("ia,ja->ij", t1.bb, x159)
    del x159
    x165 += einsum("ij->ij", x164)
    del x164
    x162 -= einsum("ijak->ijka", v.bbbb.oovo)
    x163 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x163 += einsum("ia,ijkb->jkab", t1.bb, x162)
    del x162
    x180 += einsum("ijab->ijab", x163)
    del x163
    x165 += einsum("ij->ji", f.bb.oo)
    x166 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x166 += einsum("ij,jkab->kiab", x165, t2.bbbb)
    del x165
    x180 += einsum("ijab->jiba", x166)
    del x166
    x167 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x167 += einsum("ia,wba->wib", t1.bb, g.bb.bvv)
    x171 += einsum("wia->wia", x167)
    del x167
    x168 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x168 += einsum("wia,ijab->wjb", g.aa.bov, t2.abab)
    x171 += einsum("wia->wia", x168)
    del x168
    x171 += einsum("wai->wia", g.bb.bvo)
    x172 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x172 += einsum("wia,wjb->ijab", u11.bb, x171)
    x180 += einsum("ijab->jiba", x172)
    del x172
    x295 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x295 += einsum("wia,wjb->ijab", u11.aa, x171)
    del x171
    t2new_baba += einsum("ijab->jiba", x295)
    t2new_abab += einsum("ijab->ijab", x295)
    del x295
    x173 += einsum("ab->ab", f.bb.vv)
    x174 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x174 += einsum("ab,ijbc->ijca", x173, t2.bbbb)
    del x173
    x180 -= einsum("ijab->jiba", x174)
    del x174
    x175 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x175 += einsum("wia,wib->ab", g.bb.bov, u11.bb)
    x176 += einsum("ab->ab", x175)
    x177 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x177 += einsum("ab,ijac->ijcb", x176, t2.bbbb)
    del x176
    x180 -= einsum("ijab->jiab", x177)
    del x177
    t2new_bbbb += einsum("ijab->ijab", x180)
    t2new_bbbb -= einsum("ijab->ijba", x180)
    t2new_bbbb -= einsum("ijab->jiab", x180)
    t2new_bbbb += einsum("ijab->jiba", x180)
    del x180
    x277 += einsum("ab->ba", x175) * -1
    x327 += einsum("ab->ba", x175) * -1
    del x175
    x188 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x188 += einsum("ia,jkla->ijlk", t1.bb, v.bbbb.ooov)
    x189 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x189 += einsum("ijab,kijl->klab", t2.bbbb, x188)
    x219 += einsum("ijab->ijab", x189)
    del x189
    x196 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x196 += einsum("ia,jkil->jkla", t1.bb, x188)
    del x188
    x205 += einsum("ijka->ijka", x196)
    del x196
    x190 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x190 += einsum("ijab,iajc->bc", t2.abab, v.aabb.ovov)
    x191 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x191 += einsum("ab,ijbc->jiac", x190, t2.bbbb)
    x219 += einsum("ijab->ijab", x191) * -1
    del x191
    x277 += einsum("ab->ab", x190) * -1
    x327 += einsum("ab->ab", x190) * -1
    del x190
    x197 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x197 += einsum("ijab,kbca->jikc", t2.bbbb, v.bbbb.ovvv)
    x205 += einsum("ijka->ikja", x197)
    del x197
    x198 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x198 += einsum("ia,jbca->ijcb", t1.bb, v.bbbb.ovvv)
    x199 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x199 += einsum("ia,jkba->ijkb", t1.bb, x198)
    del x198
    x205 += einsum("ijka->ikja", x199)
    del x199
    x202 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x202 += einsum("ijka->ikja", v.bbbb.ooov)
    x202 += einsum("ijka->kija", v.bbbb.ooov) * -1
    x213 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x213 += einsum("ia,jika->jk", t1.bb, x202)
    x214 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x214 += einsum("ij->ij", x213) * -1
    del x213
    x203 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x203 += einsum("ijab->jiab", t2.bbbb)
    x203 += einsum("ijab->jiba", t2.bbbb) * -1
    x204 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x204 += einsum("ijka,jlab->iklb", x202, x203)
    del x202
    del x203
    x205 += einsum("ijka->kija", x204)
    del x204
    x206 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x206 += einsum("ia,jikb->jkab", t1.bb, x205)
    del x205
    x219 += einsum("ijab->ijab", x206)
    del x206
    x207 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x207 += einsum("iajb->jiab", v.bbbb.ovov)
    x207 += einsum("iajb->jiba", v.bbbb.ovov) * -1
    x208 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x208 += einsum("ijab,ijac->bc", t2.bbbb, x207)
    x210 += einsum("ab->ab", x208) * -1
    del x208
    x211 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x211 += einsum("ab,ijbc->ijca", x210, t2.bbbb)
    del x210
    x219 += einsum("ijab->jiab", x211)
    del x211
    x212 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x212 += einsum("ijab,ikba->jk", t2.bbbb, x207)
    x214 += einsum("ij->ji", x212) * -1
    del x212
    x215 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x215 += einsum("ij,ikab->jkab", x214, t2.bbbb)
    del x214
    x219 += einsum("ijab->jiba", x215) * -1
    del x215
    t2new_bbbb += einsum("ijab->ijab", x219) * -1
    t2new_bbbb += einsum("ijab->ijba", x219)
    t2new_bbbb += einsum("ijab->jiab", x219)
    t2new_bbbb += einsum("ijab->jiba", x219) * -1
    del x219
    x276 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x276 += einsum("ijab,ijca->bc", t2.bbbb, x207)
    del x207
    x277 += einsum("ab->ab", x276) * -1
    x327 += einsum("ab->ab", x276) * -1
    del x276
    x220 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x220 += einsum("ia,bcda->ibdc", t1.bb, v.bbbb.vvvv)
    x221 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x221 += einsum("ia,jbca->ijbc", t1.bb, x220)
    del x220
    x228 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x228 += einsum("ijab->ijab", x221)
    del x221
    x222 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x222 += einsum("ijab,kbla->ijlk", t2.bbbb, v.bbbb.ovov)
    x223 += einsum("ijkl->jilk", x222)
    x224 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x224 += einsum("ijab,klji->klab", t2.bbbb, x223)
    del x223
    x228 += einsum("ijab->jiab", x224)
    del x224
    x225 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x225 += einsum("ijkl->lkji", x222)
    del x222
    x225 += einsum("ijkl->kilj", v.bbbb.oooo)
    x226 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x226 += einsum("ia,ijkl->jkla", t1.bb, x225)
    del x225
    x227 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x227 += einsum("ia,ijkb->kjba", t1.bb, x226)
    del x226
    x228 += einsum("ijab->jiab", x227)
    del x227
    t2new_bbbb += einsum("ijab->ijab", x228)
    t2new_bbbb += einsum("ijab->ijba", x228) * -1
    del x228
    x231 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x231 += einsum("ijab,ikjl->lkba", t2.bbbb, v.bbbb.oooo)
    x233 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x233 += einsum("ijab->jiba", x231)
    del x231
    x232 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x232 += einsum("ijab,cadb->ijcd", t2.bbbb, v.bbbb.vvvv)
    x233 += einsum("ijab->jiba", x232)
    del x232
    t2new_bbbb += einsum("ijab->ijba", x233) * -1
    t2new_bbbb += einsum("ijab->ijab", x233)
    del x233
    x234 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x234 += einsum("ia,bacj->ijbc", t1.aa, v.aabb.vvvo)
    t2new_baba += einsum("ijab->jiba", x234)
    t2new_abab += einsum("ijab->ijab", x234)
    del x234
    x235 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x235 += einsum("ijab,cadb->ijcd", t2.abab, v.aabb.vvvv)
    t2new_baba += einsum("ijab->jiba", x235)
    t2new_abab += einsum("ijab->ijab", x235)
    del x235
    x238 += einsum("iabj->jiba", v.bbaa.ovvo)
    t2new_baba += einsum("ijab,kicb->jkac", x156, x238)
    del x156
    t2new_abab += einsum("ijab,jkcb->ikac", x238, x41) * -1
    del x238
    x241 += einsum("iabj->ijba", v.aaaa.ovvo)
    x241 += einsum("ijab->ijab", v.aaaa.oovv) * -1
    x242 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x242 += einsum("ijab,ikca->kjcb", t2.abab, x241)
    del x241
    t2new_baba += einsum("ijab->jiba", x242)
    t2new_abab += einsum("ijab->ijab", x242)
    del x242
    x243 += einsum("iabj->ijab", v.aabb.ovvo)
    x244 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x244 += einsum("ijab,ikac->jkbc", x10, x243)
    del x243
    t2new_baba -= einsum("ijab->jiba", x244)
    t2new_abab -= einsum("ijab->ijab", x244)
    del x244
    x245 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x245 += einsum("iabc->ibac", v.bbbb.ovvv)
    x245 -= einsum("iabc->ibca", v.bbbb.ovvv)
    x246 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x246 += einsum("ia,jbac->jibc", t1.bb, x245)
    del x245
    x247 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x247 -= einsum("ijab->ijab", x246)
    del x246
    x247 += einsum("iabj->ijba", v.bbbb.ovvo)
    x247 -= einsum("ijab->ijab", v.bbbb.oovv)
    x248 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x248 += einsum("ijab,jkcb->ikac", t2.abab, x247)
    del x247
    t2new_baba += einsum("ijab->jiba", x248)
    t2new_abab += einsum("ijab->ijab", x248)
    del x248
    x249 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x249 += einsum("ia,jabc->ijbc", t1.aa, v.aabb.ovvv)
    x251 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x251 += einsum("ijab->jiab", x249)
    del x249
    x250 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x250 += einsum("ijab,kajc->ikbc", t2.abab, v.aabb.ovov)
    x251 -= einsum("ijab->jiab", x250)
    del x250
    x251 += einsum("ijab->ijab", v.aabb.oovv)
    x252 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x252 += einsum("ijab,ikcb->kjac", t2.abab, x251)
    del x251
    t2new_baba -= einsum("ijab->jiba", x252)
    t2new_abab -= einsum("ijab->ijab", x252)
    del x252
    x253 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x253 += einsum("ia,jkla->jkil", t1.bb, v.aabb.ooov)
    x257 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x257 += einsum("ijkl->ijlk", x253)
    x270 += einsum("ijkl->ijlk", x253)
    del x253
    x254 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x254 += einsum("ijab,kalb->ikjl", t2.abab, v.aabb.ovov)
    x257 += einsum("ijkl->jilk", x254)
    x270 += einsum("ijkl->jilk", x254)
    del x254
    x255 += einsum("iajk->ijka", v.aabb.ovoo)
    x256 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x256 += einsum("ia,jkla->jikl", t1.aa, x255)
    del x255
    x257 += einsum("ijkl->ijkl", x256)
    del x256
    x257 += einsum("ijkl->ijkl", v.aabb.oooo)
    x258 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x258 += einsum("ijab,ikjl->klab", t2.abab, x257)
    del x257
    t2new_baba += einsum("ijab->jiba", x258)
    t2new_abab += einsum("ijab->ijab", x258)
    del x258
    x259 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x259 += einsum("ia,jabc->ijbc", t1.bb, v.bbaa.ovvv)
    x260 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x260 += einsum("ijab->jiab", x259)
    x272 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x272 += einsum("ijab->jiab", x259)
    del x259
    x260 += einsum("ijab->ijab", v.bbaa.oovv)
    x261 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x261 += einsum("ijab,jkca->ikcb", t2.abab, x260)
    del x260
    t2new_baba -= einsum("ijab->jiba", x261)
    t2new_abab -= einsum("ijab->ijab", x261)
    del x261
    x262 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x262 += einsum("ia,jabk->kijb", t1.bb, v.bbaa.ovvo)
    x274 += einsum("ijka->ikja", x262)
    del x262
    x263 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x263 += einsum("ijab,kbca->ijkc", t2.abab, v.bbaa.ovvv)
    x274 += einsum("ijka->ikja", x263)
    del x263
    x264 += einsum("ijka->ikja", v.bbbb.ooov) * -1
    x264 += einsum("ijka->kija", v.bbbb.ooov)
    x265 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x265 += einsum("ijab,kjlb->ikla", t2.abab, x264)
    del x264
    x274 += einsum("ijka->ijka", x265) * -1
    del x265
    x270 += einsum("ijkl->ijkl", v.aabb.oooo)
    x271 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x271 += einsum("ia,ijkl->jkla", t1.aa, x270)
    del x270
    x274 += einsum("ijka->ijka", x271) * -1
    del x271
    x272 += einsum("ijab->ijab", v.bbaa.oovv)
    x273 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x273 += einsum("ia,jkba->ijkb", t1.aa, x272)
    del x272
    x274 += einsum("ijka->ijka", x273)
    del x273
    x274 += einsum("ijak->kija", v.bbaa.oovo)
    x275 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x275 += einsum("ia,jikb->jkba", t1.bb, x274)
    del x274
    t2new_baba += einsum("ijab->jiba", x275) * -1
    t2new_abab += einsum("ijab->ijab", x275) * -1
    del x275
    x277 += einsum("ab->ab", f.bb.vv)
    x278 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x278 += einsum("ab,ijcb->ijca", x277, t2.abab)
    del x277
    t2new_baba += einsum("ijab->jiba", x278)
    t2new_abab += einsum("ijab->ijab", x278)
    del x278
    x280 += einsum("ab->ab", f.aa.vv)
    x281 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x281 += einsum("ab,ijbc->ijac", x280, t2.abab)
    del x280
    t2new_baba += einsum("ijab->jiba", x281)
    t2new_abab += einsum("ijab->ijab", x281)
    del x281
    x282 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x282 += einsum("ia,jkba->jkib", t1.bb, v.aabb.oovv)
    x291 += einsum("ijka->ijka", x282)
    del x282
    x283 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x283 += einsum("ijab,kacb->ikjc", t2.abab, v.aabb.ovvv)
    x291 += einsum("ijka->jika", x283)
    del x283
    x284 += einsum("ijka->ikja", v.aaaa.ooov)
    x284 += einsum("ijka->kija", v.aaaa.ooov) * -1
    x285 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x285 += einsum("ijab,ikla->kljb", t2.abab, x284)
    del x284
    x291 += einsum("ijka->ijka", x285) * -1
    del x285
    x289 += einsum("iabj->ijab", v.aabb.ovvo)
    x290 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x290 += einsum("ia,jkab->jikb", t1.aa, x289)
    del x289
    x291 += einsum("ijka->ijka", x290)
    del x290
    x291 += einsum("ijak->ijka", v.aabb.oovo)
    x292 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x292 += einsum("ia,ijkb->jkab", t1.aa, x291)
    del x291
    t2new_baba += einsum("ijab->jiba", x292) * -1
    t2new_abab += einsum("ijab->ijab", x292) * -1
    del x292
    x297 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x297 += einsum("ia,bacd->ibcd", t1.aa, v.aabb.vvvv)
    x298 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x298 += einsum("iabc->iabc", x297)
    del x297
    x298 += einsum("aibc->iabc", v.aabb.vovv)
    x299 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x299 += einsum("ia,jbca->jibc", t1.bb, x298)
    del x298
    t2new_baba += einsum("ijab->jiba", x299)
    t2new_abab += einsum("ijab->ijab", x299)
    del x299
    x300 += einsum("ia->ia", f.aa.ov)
    s1new += einsum("ia,wia->w", x300, u11.aa)
    del x300
    x301 += einsum("ia->ia", f.bb.ov)
    s1new += einsum("ia,wia->w", x301, u11.bb)
    del x301
    x302 = np.zeros((nbos, nbos), dtype=types[float])
    x302 += einsum("wia,xia->wx", gc.aa.bov, u11.aa)
    x310 = np.zeros((nbos, nbos), dtype=types[float])
    x310 += einsum("wx->wx", x302)
    del x302
    x303 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x303 += einsum("wia,iajb->wjb", u11.aa, v.aabb.ovov)
    x304 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x304 += einsum("wia->wia", x303)
    x319 += einsum("wia->wia", x303)
    del x303
    x304 += einsum("wia->wia", gc.bb.bov)
    x305 = np.zeros((nbos, nbos), dtype=types[float])
    x305 += einsum("wia,xia->xw", u11.bb, x304)
    del x304
    x310 += einsum("wx->wx", x305)
    del x305
    x306 = np.zeros((nbos, nbos), dtype=types[float])
    x306 += einsum("wia,xia->wx", g.aa.bov, u11.aa)
    x308 = np.zeros((nbos, nbos), dtype=types[float])
    x308 += einsum("wx->wx", x306)
    del x306
    x307 = np.zeros((nbos, nbos), dtype=types[float])
    x307 += einsum("wia,xia->wx", g.bb.bov, u11.bb)
    x308 += einsum("wx->wx", x307)
    del x307
    x308 += einsum("wx->wx", w)
    x309 = np.zeros((nbos, nbos), dtype=types[float])
    x309 += einsum("wx,wy->xy", s2, x308)
    x310 += einsum("wx->wx", x309)
    del x309
    s2new += einsum("wx->wx", x310)
    s2new += einsum("wx->xw", x310)
    del x310
    u11new_aa += einsum("wx,wia->xia", x308, u11.aa)
    u11new_bb += einsum("wx,wia->xia", x308, u11.bb)
    del x308
    x313 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x313 += einsum("wx,xia->wia", s2, g.aa.bov)
    x316 += einsum("wia->wia", x313)
    del x313
    x314 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x314 += einsum("wia,jbia->wjb", u11.bb, v.aabb.ovov)
    x316 += einsum("wia->wia", x314)
    del x314
    x316 += einsum("wia->wia", gc.aa.bov)
    x323 += einsum("ia,wja->wji", t1.aa, x316)
    u11new_aa += einsum("wia,ijab->wjb", x316, x10) * -1
    del x10
    u11new_bb += einsum("wia,ijab->wjb", x316, t2.abab)
    del x316
    x317 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x317 += einsum("wx,xia->wia", s2, g.bb.bov)
    x319 += einsum("wia->wia", x317)
    del x317
    x319 += einsum("wia->wia", gc.bb.bov)
    x329 += einsum("ia,wja->wji", t1.bb, x319)
    u11new_aa += einsum("wia,jiba->wjb", x319, t2.abab)
    u11new_bb += einsum("wia,ijba->wjb", x319, x41) * -1
    del x41
    del x319
    x321 += einsum("ab->ab", f.aa.vv)
    u11new_aa += einsum("ab,wib->wia", x321, u11.aa)
    del x321
    x322 += einsum("ia->ia", f.aa.ov)
    x323 += einsum("ia,wja->wij", x322, u11.aa)
    del x322
    x323 += einsum("wij->wij", gc.aa.boo)
    x323 += einsum("wx,xij->wij", s2, g.aa.boo)
    x323 += einsum("wia,jkia->wjk", u11.bb, v.aabb.ooov)
    u11new_aa += einsum("ia,wij->wja", t1.aa, x323) * -1
    del x323
    x324 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x324 += einsum("iabc->ibac", v.aaaa.ovvv)
    x324 -= einsum("iabc->ibca", v.aaaa.ovvv)
    x325 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x325 -= einsum("wia,ibca->wbc", u11.aa, x324)
    del x324
    x325 += einsum("wab->wab", gc.aa.bvv)
    x325 += einsum("wx,xab->wab", s2, g.aa.bvv)
    x325 += einsum("wia,iabc->wbc", u11.bb, v.bbaa.ovvv)
    u11new_aa += einsum("ia,wba->wib", t1.aa, x325)
    del x325
    x327 += einsum("ab->ab", f.bb.vv)
    u11new_bb += einsum("ab,wib->wia", x327, u11.bb)
    del x327
    x328 += einsum("ia->ia", f.bb.ov)
    x329 += einsum("ia,wja->wij", x328, u11.bb)
    del x328
    x329 += einsum("wij->wij", gc.bb.boo)
    x329 += einsum("wx,xij->wij", s2, g.bb.boo)
    x329 += einsum("wia,iajk->wjk", u11.aa, v.aabb.ovoo)
    u11new_bb += einsum("ia,wij->wja", t1.bb, x329) * -1
    del x329
    x330 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x330 -= einsum("iabc->ibac", v.bbbb.ovvv)
    x330 += einsum("iabc->ibca", v.bbbb.ovvv)
    x331 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x331 -= einsum("wia,ibac->wbc", u11.bb, x330)
    del x330
    x331 += einsum("wab->wab", gc.bb.bvv)
    x331 += einsum("wx,xab->wab", s2, g.bb.bvv)
    x331 += einsum("wia,iabc->wbc", u11.aa, v.aabb.ovvv)
    u11new_bb += einsum("ia,wba->wib", t1.bb, x331)
    del x331
    t1new_aa += einsum("ijab,jbca->ic", t2.abab, v.bbaa.ovvv)
    t1new_aa += einsum("wab,wib->ia", g.aa.bvv, u11.aa)
    t1new_aa += einsum("ia,iabj->jb", t1.bb, v.bbaa.ovvo)
    t1new_aa += einsum("ai->ia", f.aa.vo)
    t1new_aa += einsum("w,wai->ia", s1, g.aa.bvo)
    t1new_bb += einsum("ia,iabj->jb", t1.aa, v.aabb.ovvo)
    t1new_bb += einsum("ijab,iacb->jc", t2.abab, v.aabb.ovvv)
    t1new_bb += einsum("ai->ia", f.bb.vo)
    t1new_bb += einsum("wab,wib->ia", g.bb.bvv, u11.bb)
    t1new_bb += einsum("w,wai->ia", s1, g.bb.bvo)
    t2new_aaaa -= einsum("aibj->jiab", v.aaaa.vovo)
    t2new_aaaa += einsum("aibj->jiba", v.aaaa.vovo)
    t2new_bbbb -= einsum("aibj->jiab", v.bbbb.vovo)
    t2new_bbbb += einsum("aibj->jiba", v.bbbb.vovo)
    t2new_baba += einsum("aibj->jiba", v.aabb.vovo)
    t2new_abab += einsum("aibj->ijab", v.aabb.vovo)
    s1new += einsum("ia,wia->w", t1.aa, gc.aa.bov)
    s1new += einsum("w,wx->x", s1, w)
    s1new += einsum("ia,wia->w", t1.bb, gc.bb.bov)
    s1new += einsum("w->w", G)
    u11new_aa += einsum("wx,xai->wia", s2, g.aa.bvo)
    u11new_aa += einsum("wia,iabj->wjb", u11.bb, v.bbaa.ovvo)
    u11new_aa += einsum("wai->wia", gc.aa.bvo)
    u11new_bb += einsum("wia,iabj->wjb", u11.aa, v.aabb.ovvo)
    u11new_bb += einsum("wx,xai->wia", s2, g.bb.bvo)
    u11new_bb += einsum("wai->wia", gc.bb.bvo)

    t1new.aa = t1new_aa
    t1new.bb = t1new_bb
    t2new.abab = t2new_abab
    t2new.baba = t2new_baba
    t2new.aaaa = t2new_aaaa
    t2new.bbbb = t2new_bbbb
    u11new.aa = u11new_aa
    u11new.bb = u11new_bb

    return {"t1new": t1new, "t2new": t2new, "s1new": s1new, "s2new": s2new, "u11new": u11new}

def update_lams(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, **kwargs):
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

    # L1, L2, LS1 , LS2 and LU11 amplitudes
    x0 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x0 += einsum("abij,kjac->ikbc", l2.abab, t2.abab)
    l1new_aa = np.zeros((nvir[0], nocc[0]), dtype=types[float])
    l1new_aa += einsum("iabc,jibc->aj", v.aabb.ovvv, x0) * -1
    del x0
    x1 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x1 += einsum("ia,bajk->jkib", t1.bb, l2.abab)
    x3 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x3 += einsum("ia,jkla->jikl", t1.aa, x1)
    x68 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x68 += einsum("ijkl->ijkl", x3)
    x476 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x476 += einsum("ijkl->ijkl", x3)
    l1new_aa += einsum("iajk,likj->al", v.aabb.ovoo, x3)
    l1new_bb = np.zeros((nvir[1], nocc[1]), dtype=types[float])
    l1new_bb += einsum("ijka,jilk->al", v.aabb.ooov, x3)
    del x3
    x55 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x55 += einsum("ia,jikb->jkba", t1.bb, x1)
    x69 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x69 += einsum("ijab,kjla->kilb", t2.abab, x1)
    x180 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x180 += einsum("ijab,ijka->kb", t2.abab, x1)
    x190 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x190 += einsum("ia->ia", x180)
    del x180
    x262 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x262 += einsum("ijab,ikla->kjlb", t2.abab, x1)
    x303 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x303 += einsum("iajk,lkjb->liba", v.aabb.ovoo, x1)
    x342 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x342 -= einsum("ijab->ijab", x303)
    del x303
    x444 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x444 += einsum("iabc,jkib->jkca", v.bbaa.ovvv, x1)
    l2new_baba = np.zeros((nvir[1], nvir[0], nocc[1], nocc[0]), dtype=types[float])
    l2new_baba -= einsum("ijab->baji", x444)
    l2new_abab = np.zeros((nvir[0], nvir[1], nocc[0], nocc[1]), dtype=types[float])
    l2new_abab -= einsum("ijab->abij", x444)
    del x444
    l1new_aa -= einsum("ijab,kjia->bk", v.bbaa.oovv, x1)
    x2 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x2 += einsum("abij,klab->ikjl", l2.abab, t2.abab)
    x68 += einsum("ijkl->ijkl", x2)
    x69 += einsum("ia,jkil->jkla", t1.bb, x68)
    x267 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x267 += einsum("ia,ijkl->jkla", t1.aa, x68)
    del x68
    x476 += einsum("ijkl->ijkl", x2)
    x477 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x477 += einsum("iajb,kilj->klab", v.aabb.ovov, x476)
    del x476
    l2new_baba += einsum("ijab->baji", x477)
    l2new_abab += einsum("ijab->abij", x477)
    del x477
    l1new_aa += einsum("iajk,likj->al", v.aabb.ovoo, x2)
    l1new_bb += einsum("ijka,jilk->al", v.aabb.ooov, x2)
    del x2
    x4 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum("ai,ja->ij", l1.aa, t1.aa)
    x167 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x167 += einsum("ij->ij", x4)
    x220 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x220 += einsum("ij->ij", x4)
    x335 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x335 += einsum("ij->ij", x4)
    x5 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x5 += einsum("w,wia->ia", s1, g.aa.bov)
    x6 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum("ia->ia", x5)
    x38 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x38 += einsum("ia->ia", x5)
    x322 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x322 += einsum("ia->ia", x5)
    x512 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x512 += einsum("ia->ia", x5)
    l1new_aa -= einsum("ij,ja->ai", x4, x5)
    del x4
    del x5
    x6 += einsum("ia->ia", f.aa.ov)
    x7 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x7 += einsum("ia,jkab->jkib", x6, t2.aaaa)
    del x6
    x8 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum("ijka->kjia", x7)
    del x7
    x8 += einsum("ijak->ijka", v.aaaa.oovo) * -1
    x33 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x33 += einsum("ijka->jika", x8) * -1
    x33 += einsum("ijka->kija", x8)
    del x8
    x9 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x9 += einsum("ijab,kacb->ijkc", t2.aaaa, v.aaaa.ovvv)
    x22 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x22 += einsum("ijka->ijka", x9) * -1
    del x9
    x10 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x10 += einsum("ia,jabc->ijbc", t1.aa, v.aaaa.ovvv)
    x11 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x11 += einsum("ia,jkba->jikb", t1.aa, x10)
    del x10
    x22 += einsum("ijka->ijka", x11) * -1
    del x11
    x12 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x12 += einsum("ia,jbia->jb", t1.bb, v.aabb.ovov)
    x15 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x15 += einsum("ia->ia", x12)
    x38 += einsum("ia->ia", x12)
    x221 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x221 += einsum("ia->ia", x12)
    x341 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x341 += einsum("ia->ia", x12)
    x512 += einsum("ia->ia", x12)
    del x12
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x13 += einsum("iajb->jiab", v.aaaa.ovov) * -1
    x13 += einsum("iajb->jiba", v.aaaa.ovov)
    x14 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x14 += einsum("ia,ijba->jb", t1.aa, x13)
    x15 += einsum("ia->ia", x14) * -1
    x16 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x16 += einsum("ia,jkab->jkib", x15, t2.aaaa)
    x22 += einsum("ijka->jika", x16)
    del x16
    x351 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x351 += einsum("ia,ja->ij", t1.aa, x15)
    x352 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x352 += einsum("ij->ij", x351)
    del x351
    x38 += einsum("ia->ia", x14) * -1
    x221 += einsum("ia->ia", x14) * -1
    del x14
    x198 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x198 += einsum("wia,ijba->wjb", u11.aa, x13)
    x199 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x199 += einsum("wia->wia", x198) * -1
    x211 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x211 += einsum("wia->wia", x198) * -1
    del x198
    x453 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x453 += einsum("ijab,ikca->kjcb", t2.abab, x13)
    x455 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x455 += einsum("ijab->ijab", x453) * -1
    del x453
    x17 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x17 += einsum("ijab,kbla->ijlk", t2.aaaa, v.aaaa.ovov)
    x20 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x20 += einsum("ijkl->jilk", x17)
    x368 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x368 += einsum("ijkl->jilk", x17)
    del x17
    x18 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x18 += einsum("ia,jakb->ikjb", t1.aa, v.aaaa.ovov)
    x19 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x19 += einsum("ia,jkla->ijkl", t1.aa, x18)
    x20 += einsum("ijkl->ijkl", x19)
    x21 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x21 += einsum("ia,jkil->jkla", t1.aa, x20)
    del x20
    x22 += einsum("ijka->jika", x21)
    del x21
    x33 += einsum("ijka->ikja", x22) * -1
    x33 += einsum("ijka->jkia", x22)
    del x22
    x368 += einsum("ijkl->ijkl", x19)
    del x19
    x369 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x369 += einsum("abij,ijkl->klab", l2.aaaa, x368)
    del x368
    x370 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x370 += einsum("ijab->ijab", x369)
    del x369
    x23 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x23 += einsum("ijka->ijka", x18) * -1
    x23 += einsum("ijka->ikja", x18)
    x34 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x34 += einsum("ijka->jkia", x18) * -1
    x34 += einsum("ijka->kjia", x18)
    x79 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x79 += einsum("ijka->jkia", x18) * -1
    x86 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x86 += einsum("ijka->jkia", x18) * -1
    x86 += einsum("ijka->kjia", x18)
    x338 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x338 += einsum("ijka->ijka", x18)
    x347 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x347 += einsum("ijka->ijka", x18)
    x347 += einsum("ijka->ikja", x18) * -1
    x514 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x514 += einsum("ijka->ijka", x18) * -1
    x514 += einsum("ijka->ikja", x18)
    del x18
    x23 += einsum("ijka->jika", v.aaaa.ooov)
    x23 += einsum("ijka->jkia", v.aaaa.ooov) * -1
    x24 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x24 += einsum("ijab->jiab", t2.aaaa) * -1
    x24 += einsum("ijab->jiba", t2.aaaa)
    x25 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x25 += einsum("ijka,jlba->iklb", x23, x24)
    del x23
    x33 += einsum("ijka->ijka", x25)
    x53 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x53 += einsum("ijka->ijka", x25)
    del x25
    x60 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x60 += einsum("abij,ikba->jk", l2.aaaa, x24)
    x61 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x61 += einsum("ij->ij", x60) * -1
    x167 += einsum("ij->ij", x60) * -1
    x209 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x209 += einsum("ij->ij", x60) * -1
    del x60
    x85 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x85 += einsum("iabc,ijca->jb", v.aaaa.ovvv, x24)
    x120 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x120 += einsum("ia->ia", x85) * -1
    del x85
    x104 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x104 += einsum("iajb,ikba->kj", v.aaaa.ovov, x24)
    x108 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x108 += einsum("ij->ji", x104) * -1
    del x104
    x448 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x448 += einsum("iajb,ikca->kjcb", v.aabb.ovov, x24)
    x450 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x450 += einsum("ijab->ijab", x448) * -1
    del x448
    x26 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x26 += einsum("ia,jakb->ijkb", t1.aa, v.aabb.ovov)
    x27 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x27 += einsum("ijka->jika", x26)
    x508 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x508 += einsum("ijka->jika", x26)
    x27 += einsum("ijka->ijka", v.aabb.ooov)
    x28 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x28 += einsum("ijab,kljb->ikla", t2.abab, x27)
    x33 += einsum("ijka->kjia", x28)
    x53 += einsum("ijka->kjia", x28)
    del x28
    x88 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x88 += einsum("ijab,ikjb->ka", t2.abab, x27)
    x120 += einsum("ia->ia", x88) * -1
    del x88
    x250 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x250 += einsum("ijab,iklb->klja", t2.abab, x27) * -1
    x478 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x478 += einsum("ijka,likb->ljab", x1, x27)
    l2new_baba += einsum("ijab->baji", x478)
    l2new_abab += einsum("ijab->abij", x478)
    del x478
    x29 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x29 += einsum("iabj->ijba", v.aaaa.ovvo)
    x29 += einsum("ijab->ijab", v.aaaa.oovv) * -1
    x30 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x30 += einsum("ia,jkba->ijkb", t1.aa, x29)
    x33 += einsum("ijka->ijka", x30)
    x53 += einsum("ijka->ijka", x30)
    del x30
    x96 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x96 += einsum("ia,ijba->jb", t1.aa, x29)
    x120 += einsum("ia->ia", x96)
    del x96
    x31 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x31 += einsum("ia,jkla->ijlk", t1.aa, v.aaaa.ooov)
    x32 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x32 += einsum("ijkl->jkil", x31)
    x32 += einsum("ijkl->kjil", x31) * -1
    x52 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x52 += einsum("ijkl->ijkl", x31)
    x52 += einsum("ijkl->ikjl", x31) * -1
    x53 += einsum("ia,jikl->jkla", t1.aa, x52) * -1
    del x52
    l1new_aa += einsum("abij,jkib->ak", l2.aaaa, x53)
    del x53
    x301 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x301 += einsum("abij,jkli->klba", l2.aaaa, x31)
    del x31
    x342 -= einsum("ijab->ijab", x301)
    del x301
    x32 += einsum("ijkl->kijl", v.aaaa.oooo)
    x32 += einsum("ijkl->kilj", v.aaaa.oooo) * -1
    x33 += einsum("ia,ijkl->kjla", t1.aa, x32) * -1
    del x32
    l1new_aa += einsum("abij,jkia->bk", l2.aaaa, x33) * -1
    del x33
    x34 += einsum("ijka->ikja", v.aaaa.ooov)
    x34 += einsum("ijka->kija", v.aaaa.ooov) * -1
    x46 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x46 += einsum("ijab,ikla->kljb", t2.abab, x34) * -1
    del x34
    x35 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x35 += einsum("ijab->jiab", t2.bbbb) * -1
    x35 += einsum("ijab->jiba", t2.bbbb)
    x46 += einsum("ijka,klba->ijlb", x27, x35) * -1
    del x27
    x128 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x128 += einsum("iabc,ijca->jb", v.bbbb.ovvv, x35)
    x158 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x158 += einsum("ia->ia", x128) * -1
    del x128
    x187 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x187 += einsum("abij,ikba->jk", l2.bbbb, x35)
    x188 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x188 += einsum("ij->ij", x187) * -1
    x261 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x261 += einsum("ij->ij", x187) * -1
    x293 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x293 += einsum("ij->ij", x187) * -1
    del x187
    x194 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x194 += einsum("abij,ijbc->ac", l2.bbbb, x35)
    x195 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x195 += einsum("ab->ab", x194) * -1
    x435 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x435 += einsum("ab->ab", x194) * -1
    x496 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x496 += einsum("ab->ab", x194) * -1
    del x194
    x454 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x454 += einsum("iajb,jkcb->ikac", v.aabb.ovov, x35)
    x455 += einsum("ijab->ijab", x454) * -1
    del x454
    x36 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x36 += einsum("ia,jbka->jikb", t1.bb, v.aabb.ovov)
    x37 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x37 += einsum("ijka->ikja", x36)
    x41 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x41 += einsum("ijka->ikja", x36)
    x304 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x304 += einsum("ijka,ljkb->ilab", x1, x36)
    x342 -= einsum("ijab->ijab", x304)
    del x304
    x510 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x510 += einsum("ijka->ikja", x36)
    x37 += einsum("iajk->ijka", v.aabb.ovoo)
    x46 += einsum("ijab,kjla->kilb", t2.abab, x37) * -1
    x132 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x132 += einsum("ijab,ijka->kb", t2.abab, x37)
    x158 += einsum("ia->ia", x132) * -1
    del x132
    x242 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x242 += einsum("ijab,ikla->jklb", t2.abab, x37)
    x246 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x246 += einsum("ijka->kjia", x242)
    x257 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x257 += einsum("ijka->kjia", x242)
    del x242
    x250 += einsum("ijab,iklb->jkla", x24, x37) * -1
    x473 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x473 += einsum("ia,jkla->ijkl", t1.aa, x37)
    x474 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x474 += einsum("ijkl->jikl", x473)
    del x473
    x38 += einsum("ia->ia", f.aa.ov)
    x46 += einsum("ia,jkab->ijkb", x38, t2.abab)
    x89 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x89 += einsum("ia,ijba->jb", x38, x24)
    x120 += einsum("ia->ia", x89) * -1
    del x89
    x107 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x107 += einsum("ia,ja->ij", t1.aa, x38)
    x108 += einsum("ij->ji", x107)
    del x107
    x134 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x134 += einsum("ia,ijab->jb", x38, t2.abab)
    x158 += einsum("ia->ia", x134)
    del x134
    x200 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x200 += einsum("ia,wja->wij", x38, u11.aa)
    x216 = np.zeros((nbos), dtype=types[float])
    x216 += einsum("ia,wia->w", x38, u11.aa)
    x219 = np.zeros((nbos), dtype=types[float])
    x219 += einsum("w->w", x216)
    del x216
    lu11new_aa = np.zeros((nbos, nvir[0], nocc[0]), dtype=types[float])
    lu11new_aa += einsum("w,ia->wai", ls1, x38) * 2
    x39 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x39 += einsum("ia,jkla->jkil", t1.bb, v.aabb.ooov)
    x43 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x43 += einsum("ijkl->ijlk", x39)
    x474 += einsum("ijkl->ijlk", x39)
    del x39
    x40 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x40 += einsum("ijab,kalb->ikjl", t2.abab, v.aabb.ovov)
    x43 += einsum("ijkl->jilk", x40)
    x474 += einsum("ijkl->jilk", x40)
    del x40
    x41 += einsum("iajk->ijka", v.aabb.ovoo)
    x42 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x42 += einsum("ia,jkla->ijkl", t1.aa, x41)
    del x41
    x43 += einsum("ijkl->jikl", x42)
    del x42
    x43 += einsum("ijkl->ijkl", v.aabb.oooo)
    x46 += einsum("ia,jkil->jkla", t1.bb, x43) * -1
    x250 += einsum("ia,ijkl->jkla", t1.aa, x43) * -1
    del x43
    x44 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x44 += einsum("ia,jbca->jibc", t1.bb, v.aabb.ovvv)
    x45 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x45 += einsum("ijab->ijab", x44)
    x311 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x311 += einsum("ijab->ijab", x44)
    x455 += einsum("ijab->ijab", x44)
    del x44
    x45 += einsum("iabj->ijab", v.aabb.ovvo)
    x46 += einsum("ia,jkab->jikb", t1.aa, x45)
    x46 += einsum("ijak->ijka", v.aabb.oovo)
    x46 += einsum("ia,jkba->jkib", t1.bb, v.aabb.oovv)
    x46 += einsum("ijab,kacb->kijc", t2.abab, v.aabb.ovvv)
    l1new_aa += einsum("abij,kijb->ak", l2.abab, x46) * -1
    del x46
    x47 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x47 += einsum("abij,kjcb->ikac", l2.abab, t2.abab)
    x50 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x50 += einsum("ijab->ijab", x47)
    x349 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x349 += einsum("ijab,kica->kjcb", x13, x47)
    del x47
    x362 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x362 += einsum("ijab->ijab", x349) * -1
    del x349
    x48 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x48 += einsum("abij->jiab", l2.aaaa)
    x48 += einsum("abij->jiba", l2.aaaa) * -1
    x55 += einsum("ijab,ikca->kjcb", t2.abab, x48) * -1
    x49 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x49 += einsum("ijab->jiab", t2.aaaa)
    x49 += einsum("ijab->jiba", t2.aaaa) * -1
    x50 += einsum("ijab,ikac->jkbc", x48, x49)
    x165 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x165 += einsum("ai,ijba->jb", l1.aa, x49)
    x169 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x169 += einsum("ia->ia", x165) * -1
    del x165
    x174 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x174 += einsum("abij,ijcb->ac", l2.aaaa, x49)
    x175 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x175 += einsum("ab->ab", x174) * -1
    x358 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x358 += einsum("ab->ab", x174) * -1
    x494 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x494 += einsum("ab->ab", x174) * -1
    del x174
    x259 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x259 += einsum("abij,ikca->kjcb", l2.abab, x49) * -1
    x267 += einsum("ijka,ilba->ljkb", x1, x49) * -1
    x344 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x344 += einsum("ijab,ikca->jkbc", x13, x49)
    del x13
    x345 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x345 += einsum("ijab->jiba", x344)
    del x344
    x350 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x350 += einsum("iajb,ikab->kj", v.aaaa.ovov, x49)
    x352 += einsum("ij->ij", x350) * -1
    del x350
    x355 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x355 += einsum("iajb,ijac->cb", v.aaaa.ovov, x49)
    x356 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x356 += einsum("ab->ab", x355) * -1
    del x355
    x486 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x486 += einsum("iajb,ijca->cb", v.aaaa.ovov, x49)
    x487 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x487 += einsum("ab->ab", x486) * -1
    del x486
    x51 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x51 += einsum("iabc->ibca", v.aaaa.ovvv) * -1
    x51 += einsum("iabc->ibac", v.aaaa.ovvv)
    x458 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x458 += einsum("ia,jbac->ijbc", t1.aa, x51)
    x459 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x459 += einsum("ijab->jiab", x458) * -1
    del x458
    l1new_aa += einsum("ijab,jabc->ci", x50, x51)
    del x50
    x54 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x54 += einsum("ijab->jiab", t2.bbbb)
    x54 += einsum("ijab->jiba", t2.bbbb) * -1
    x55 += einsum("abij,jkcb->ikac", l2.abab, x54) * -1
    l1new_aa += einsum("iabc,jiba->cj", v.bbaa.ovvv, x55) * -1
    del x55
    x183 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x183 += einsum("ai,ijba->jb", l1.bb, x54)
    x190 += einsum("ia->ia", x183) * -1
    del x183
    x56 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x56 += einsum("ia,abjk->jikb", t1.aa, l2.abab)
    x62 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x62 += einsum("ijab,kljb->klia", t2.abab, x56)
    x69 += einsum("ijab,klib->klja", x54, x56) * -1
    x163 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x163 += einsum("ijab,ikjb->ka", t2.abab, x56)
    x169 += einsum("ia->ia", x163)
    del x163
    x259 += einsum("ia,ijkb->jkab", t1.aa, x56)
    x267 += einsum("ijab,iklb->klja", t2.abab, x56)
    x374 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x374 += einsum("ijka,jilb->lkba", v.aabb.ooov, x56)
    x415 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x415 -= einsum("ijab->ijab", x374)
    del x374
    x377 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x377 += einsum("ijka,ijlb->lkba", x26, x56)
    x415 -= einsum("ijab->ijab", x377)
    del x377
    x446 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x446 += einsum("iabc,jikb->jkac", v.aabb.ovvv, x56)
    l2new_baba -= einsum("ijab->baji", x446)
    l2new_abab -= einsum("ijab->abij", x446)
    del x446
    x479 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x479 += einsum("ijka,likb->ljab", x37, x56)
    del x37
    l2new_baba += einsum("ijab->baji", x479)
    l2new_abab += einsum("ijab->abij", x479)
    del x479
    x500 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x500 += einsum("ia,jikb->jkab", x38, x56)
    del x38
    l2new_baba += einsum("ijab->baji", x500) * -1
    l2new_abab += einsum("ijab->abij", x500) * -1
    del x500
    l1new_aa += einsum("ijab,kijb->ak", x45, x56) * -1
    del x45
    l1new_bb -= einsum("ijab,jika->bk", v.aabb.oovv, x56)
    l2new_baba += einsum("ijka,iklb->balj", x347, x56) * -1
    l2new_abab += einsum("ijka,iklb->abjl", x514, x56)
    del x514
    x57 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x57 += einsum("ia,abjk->kjib", t1.aa, l2.aaaa)
    x58 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x58 += einsum("ijka->ijka", x57)
    x58 += einsum("ijka->jika", x57) * -1
    x62 += einsum("ijab,iklb->klja", x49, x58)
    x64 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x64 += einsum("ia,ijkb->jkba", t1.aa, x58) * -1
    l1new_aa += einsum("iabc,jiac->bj", x51, x64)
    del x64
    x69 += einsum("ijab,ikla->kljb", t2.abab, x58) * -1
    x348 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x348 += einsum("ijka,iljb->lkba", x347, x58)
    del x58
    del x347
    x362 += einsum("ijab->ijab", x348)
    del x348
    x71 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x71 += einsum("ia,jkla->kjli", t1.aa, x57)
    x72 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x72 += einsum("ia,ijkl->jlka", t1.aa, x71)
    x73 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x73 += einsum("ijka->ijka", x72)
    del x72
    x122 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x122 += einsum("ijkl->ijkl", x71)
    x122 += einsum("ijkl->ijlk", x71) * -1
    l1new_aa += einsum("ijka,ljki->al", v.aaaa.ooov, x122)
    del x122
    x367 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x367 += einsum("iajb,klij->klab", v.aaaa.ovov, x71)
    del x71
    x370 += einsum("ijab->ijab", x367)
    del x367
    x75 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x75 += einsum("ijka->ijka", x57) * -1
    x75 += einsum("ijka->jika", x57)
    x481 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x481 += einsum("ijka,iljb->lkba", x26, x75)
    del x26
    l2new_baba += einsum("ijab->baji", x481) * -1
    l2new_abab += einsum("ijab->abij", x481) * -1
    del x481
    l1new_aa += einsum("ijab,jkia->bk", x29, x75)
    del x75
    del x29
    x164 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x164 += einsum("ijab,ijka->kb", x49, x57)
    del x49
    x169 += einsum("ia->ia", x164) * -1
    del x164
    x302 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x302 += einsum("iabc,jkib->kjac", v.aaaa.ovvv, x57)
    x342 -= einsum("ijab->ijab", x302)
    del x302
    x313 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x313 += einsum("ijka->ijka", x57)
    x313 -= einsum("ijka->jika", x57)
    x480 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x480 += einsum("ijka,ljib->lkba", v.aabb.ooov, x313)
    l2new_baba -= einsum("ijab->baji", x480)
    l2new_abab -= einsum("ijab->abij", x480)
    del x480
    x361 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x361 += einsum("ia,jkib->jkba", x15, x57)
    del x15
    x362 += einsum("ijab->ijab", x361)
    del x361
    x59 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x59 += einsum("abij,kjab->ik", l2.abab, t2.abab)
    x61 += einsum("ij->ij", x59)
    x62 += einsum("ia,jk->jkia", t1.aa, x61)
    x69 += einsum("ia,jk->jkia", t1.bb, x61) * -1
    x360 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x360 += einsum("ij,jakb->ikab", x61, v.aaaa.ovov)
    x362 += einsum("ijab->ijba", x360) * -1
    del x360
    l1new_aa += einsum("ia,ji->aj", f.aa.ov, x61) * -1
    del x61
    x167 += einsum("ij->ij", x59)
    x209 += einsum("ij->ij", x59)
    del x59
    x63 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x63 -= einsum("iajb->jiab", v.aaaa.ovov)
    x63 += einsum("iajb->jiba", v.aaaa.ovov)
    x332 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x332 += einsum("wia,ijba->wjb", u11.aa, x63)
    x333 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x333 -= einsum("wia->wia", x332)
    x502 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x502 -= einsum("wia->wia", x332)
    del x332
    x340 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x340 += einsum("ia,ijba->jb", t1.aa, x63)
    x341 -= einsum("ia->ia", x340)
    x342 += einsum("ai,jb->ijab", l1.aa, x341)
    del x341
    x512 -= einsum("ia->ia", x340)
    del x340
    l1new_aa += einsum("ijka,kjab->bi", x62, x63) * -1
    del x62
    del x63
    x65 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x65 += einsum("ia,bcda->ibdc", t1.aa, v.aaaa.vvvv)
    x66 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x66 += einsum("iabc->iabc", x65) * -1
    del x65
    x66 += einsum("aibc->iabc", v.aaaa.vovv)
    x67 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x67 += einsum("iabc->iabc", x66)
    x67 += einsum("iabc->ibac", x66) * -1
    del x66
    l1new_aa += einsum("abij,ibac->cj", l2.aaaa, x67) * -1
    del x67
    x69 += einsum("ai,jkab->ijkb", l1.aa, t2.abab) * -1
    l1new_aa += einsum("iajb,kijb->ak", v.aabb.ovov, x69)
    del x69
    x70 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x70 += einsum("ai,jkab->ikjb", l1.aa, t2.aaaa)
    x73 += einsum("ijka->ijka", x70) * 0.9999999999999993
    del x70
    x74 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x74 += einsum("ijka->ijka", x73) * -1
    x74 += einsum("ijka->ikja", x73)
    del x73
    l1new_aa += einsum("iajb,kjia->bk", v.aaaa.ovov, x74) * -1
    del x74
    x76 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x76 += einsum("abci->iabc", v.aabb.vvvo)
    x76 += einsum("ia,bcda->ibcd", t1.bb, v.aabb.vvvv)
    l1new_aa += einsum("abij,jacb->ci", l2.abab, x76)
    del x76
    x77 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x77 += einsum("abij,klba->ijlk", l2.aaaa, t2.aaaa)
    x78 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x78 += einsum("ijkl->jikl", x77)
    x78 += einsum("ijkl->jilk", x77) * -1
    x366 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x366 += einsum("iajb,klij->lkba", v.aaaa.ovov, x77)
    del x77
    x370 += einsum("ijab->ijab", x366)
    del x366
    l2new_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    l2new_aaaa += einsum("ijab->abij", x370)
    l2new_aaaa += einsum("ijab->baij", x370) * -1
    del x370
    x79 += einsum("ijka->ikja", v.aaaa.ooov)
    l1new_aa += einsum("ijkl,klia->aj", x78, x79) * -1
    del x78
    del x79
    x80 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x80 += einsum("w,wai->ia", s1, g.aa.bvo)
    x120 += einsum("ia->ia", x80)
    del x80
    x81 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x81 += einsum("wab,wib->ia", g.aa.bvv, u11.aa)
    x120 += einsum("ia->ia", x81)
    del x81
    x82 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x82 += einsum("ia,iabj->jb", t1.bb, v.bbaa.ovvo)
    x120 += einsum("ia->ia", x82)
    del x82
    x83 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x83 += einsum("ijab,ikja->kb", t2.aaaa, v.aaaa.ooov)
    x120 += einsum("ia->ia", x83)
    del x83
    x84 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x84 += einsum("ijab,jbca->ic", t2.abab, v.bbaa.ovvv)
    x120 += einsum("ia->ia", x84)
    del x84
    x86 += einsum("ijka->ikja", v.aaaa.ooov)
    x87 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x87 += einsum("ijab,ijkb->ka", t2.aaaa, x86)
    del x86
    x120 += einsum("ia->ia", x87) * -1
    del x87
    x90 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x90 += einsum("w,wia->ia", s1, g.bb.bov)
    x94 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x94 += einsum("ia->ia", x90)
    x227 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x227 += einsum("ia->ia", x90)
    x297 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x297 += einsum("ia->ia", x90)
    x394 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x394 += einsum("ia->ia", x90)
    x513 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x513 += einsum("ia->ia", x90)
    del x90
    x91 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x91 += einsum("ia,iajb->jb", t1.aa, v.aabb.ovov)
    x94 += einsum("ia->ia", x91)
    x233 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x233 += einsum("ia->ia", x91)
    x297 += einsum("ia->ia", x91)
    x414 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x414 += einsum("ia->ia", x91)
    x513 += einsum("ia->ia", x91)
    del x91
    x92 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x92 += einsum("iajb->jiab", v.bbbb.ovov) * -1
    x92 += einsum("iajb->jiba", v.bbbb.ovov)
    x93 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x93 += einsum("ia,ijba->jb", t1.bb, x92)
    x94 += einsum("ia->ia", x93) * -1
    x233 += einsum("ia->ia", x93) * -1
    x234 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x234 += einsum("ia,jkab->jkib", x233, t2.bbbb)
    x239 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x239 += einsum("ijka->jika", x234)
    del x234
    x428 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x428 += einsum("ia,ja->ij", t1.bb, x233)
    x429 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x429 += einsum("ij->ij", x428)
    del x428
    x297 += einsum("ia->ia", x93) * -1
    del x93
    x283 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x283 += einsum("wia,ijba->wjb", u11.bb, x92)
    x284 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x284 += einsum("wia->wia", x283) * -1
    x295 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x295 += einsum("wia->wia", x283) * -1
    del x283
    x427 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x427 += einsum("ijab,ikba->jk", t2.bbbb, x92)
    x429 += einsum("ij->ij", x427) * -1
    del x427
    x449 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x449 += einsum("ijab,jkcb->ikac", t2.abab, x92)
    x450 += einsum("ijab->ijab", x449) * -1
    del x449
    x489 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x489 += einsum("ijab,ijac->bc", t2.bbbb, x92)
    x490 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x490 += einsum("ab->ab", x489) * -1
    del x489
    x94 += einsum("ia->ia", f.bb.ov)
    x95 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x95 += einsum("ia,jiba->jb", x94, t2.abab)
    x120 += einsum("ia->ia", x95)
    del x95
    x133 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x133 += einsum("ia,ijba->jb", x94, x35)
    x158 += einsum("ia->ia", x133) * -1
    del x133
    x148 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x148 += einsum("ia,ja->ij", t1.bb, x94)
    x149 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x149 += einsum("ij->ji", x148)
    del x148
    x217 = np.zeros((nbos), dtype=types[float])
    x217 += einsum("ia,wia->w", x94, u11.bb)
    x219 += einsum("w->w", x217)
    del x217
    x250 += einsum("ia,jkba->jikb", x94, t2.abab)
    x285 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x285 += einsum("ia,wja->wij", x94, u11.bb)
    x501 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x501 += einsum("ia,jkib->jkba", x94, x1)
    l2new_baba += einsum("ijab->baji", x501) * -1
    l2new_abab += einsum("ijab->abij", x501) * -1
    del x501
    lu11new_bb = np.zeros((nbos, nvir[1], nocc[1]), dtype=types[float])
    lu11new_bb += einsum("w,ia->wai", ls1, x94) * 2
    del x94
    x97 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x97 += einsum("ia,wja->wji", t1.aa, g.aa.bov)
    x98 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x98 += einsum("wij->wij", x97)
    x521 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x521 += einsum("wij->wij", x97)
    del x97
    x98 += einsum("wij->wij", g.aa.boo)
    x99 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x99 += einsum("wia,wij->ja", u11.aa, x98)
    x120 += einsum("ia->ia", x99) * -1
    del x99
    x100 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x100 += einsum("w,wij->ij", s1, g.aa.boo)
    x108 += einsum("ij->ij", x100)
    x324 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x324 += einsum("ij->ij", x100)
    del x100
    x101 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x101 += einsum("wia,wja->ij", g.aa.bov, u11.aa)
    x108 += einsum("ij->ij", x101)
    x324 += einsum("ij->ij", x101)
    del x101
    x102 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x102 += einsum("ia,jkia->jk", t1.bb, v.aabb.ooov)
    x108 += einsum("ij->ij", x102)
    x328 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x328 += einsum("ij->ij", x102)
    del x102
    x103 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x103 += einsum("ijab,kajb->ik", t2.abab, v.aabb.ovov)
    x108 += einsum("ij->ji", x103)
    x352 += einsum("ij->ij", x103)
    del x103
    x353 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x353 += einsum("ij,abik->kjab", x352, l2.aaaa)
    del x352
    x362 += einsum("ijab->ijba", x353)
    del x353
    x105 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x105 += einsum("ijka->ikja", v.aaaa.ooov)
    x105 += einsum("ijka->kija", v.aaaa.ooov) * -1
    x106 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x106 += einsum("ia,ijka->jk", t1.aa, x105)
    x108 += einsum("ij->ij", x106) * -1
    del x106
    x200 += einsum("wia,ijka->wjk", u11.aa, x105) * -1
    x108 += einsum("ij->ij", f.aa.oo)
    x109 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x109 += einsum("ia,ij->ja", t1.aa, x108)
    x120 += einsum("ia->ia", x109) * -1
    del x109
    x492 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x492 += einsum("ij,abjk->ikab", x108, l2.abab)
    l2new_baba += einsum("ijab->baji", x492) * -1
    l2new_abab += einsum("ijab->abij", x492) * -1
    del x492
    l1new_aa += einsum("ai,ji->aj", l1.aa, x108) * -1
    del x108
    x110 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x110 += einsum("w,wab->ab", s1, g.aa.bvv)
    x114 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x114 += einsum("ab->ab", x110)
    x317 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x317 += einsum("ab->ab", x110)
    x487 += einsum("ab->ab", x110)
    del x110
    x111 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x111 += einsum("ia,iabc->bc", t1.bb, v.bbaa.ovvv)
    x114 += einsum("ab->ab", x111)
    x320 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x320 += einsum("ab->ab", x111)
    x487 += einsum("ab->ab", x111)
    del x111
    x112 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x112 += einsum("iabc->ibca", v.aaaa.ovvv)
    x112 += einsum("iabc->ibac", v.aaaa.ovvv) * -1
    x113 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x113 += einsum("ia,ibac->bc", t1.aa, x112)
    del x112
    x114 += einsum("ab->ab", x113) * -1
    x487 += einsum("ab->ab", x113) * -1
    del x113
    x114 += einsum("ab->ab", f.aa.vv)
    x115 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x115 += einsum("ia,ba->ib", t1.aa, x114)
    x120 += einsum("ia->ia", x115)
    del x115
    l1new_aa += einsum("ai,ab->bi", l1.aa, x114)
    del x114
    x116 = np.zeros((nbos), dtype=types[float])
    x116 += einsum("ia,wia->w", t1.aa, g.aa.bov)
    x118 = np.zeros((nbos), dtype=types[float])
    x118 += einsum("w->w", x116)
    x515 = np.zeros((nbos), dtype=types[float])
    x515 += einsum("w->w", x116)
    del x116
    x117 = np.zeros((nbos), dtype=types[float])
    x117 += einsum("ia,wia->w", t1.bb, g.bb.bov)
    x118 += einsum("w->w", x117)
    x515 += einsum("w->w", x117)
    del x117
    x118 += einsum("w->w", G)
    x119 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x119 += einsum("w,wia->ia", x118, u11.aa)
    x120 += einsum("ia->ia", x119)
    del x119
    x157 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x157 += einsum("w,wia->ia", x118, u11.bb)
    x158 += einsum("ia->ia", x157)
    del x157
    x218 = np.zeros((nbos), dtype=types[float])
    x218 += einsum("w,wx->x", x118, s2)
    del x118
    x219 += einsum("w->w", x218)
    del x218
    x120 += einsum("ai->ia", f.aa.vo)
    l1new_bb += einsum("ia,abij->bj", x120, l2.abab)
    ls1new = np.zeros((nbos), dtype=types[float])
    ls1new += einsum("ia,wai->w", x120, lu11.aa)
    x121 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x121 -= einsum("abij->jiab", l2.aaaa)
    x121 += einsum("abij->jiba", l2.aaaa)
    x204 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x204 += einsum("wia,ijba->wjb", u11.aa, x121)
    x205 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x205 -= einsum("wia->wia", x204)
    x330 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x330 -= einsum("wia->wia", x204)
    del x204
    l1new_aa += einsum("ia,ijba->bj", x120, x121) * -1
    del x120
    lu11new_aa -= einsum("wai,ijba->wbj", g.aa.bvo, x121)
    del x121
    x123 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x123 += einsum("w,wai->ia", s1, g.bb.bvo)
    x158 += einsum("ia->ia", x123)
    del x123
    x124 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x124 += einsum("wab,wib->ia", g.bb.bvv, u11.bb)
    x158 += einsum("ia->ia", x124)
    del x124
    x125 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x125 += einsum("ia,iabj->jb", t1.aa, v.aabb.ovvo)
    x158 += einsum("ia->ia", x125)
    del x125
    x126 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x126 += einsum("ijab,iacb->jc", t2.abab, v.aabb.ovvv)
    x158 += einsum("ia->ia", x126)
    del x126
    x127 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x127 += einsum("ijab,jkib->ka", t2.bbbb, v.bbbb.ooov)
    x158 += einsum("ia->ia", x127)
    del x127
    x129 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x129 += einsum("ia,jakb->ikjb", t1.bb, v.bbbb.ovov)
    x130 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x130 += einsum("ijka->jkia", x129) * -1
    x130 += einsum("ijka->kjia", x129)
    x236 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x236 += einsum("ia,jkla->ijkl", t1.bb, x129)
    x237 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x237 += einsum("ijkl->ijkl", x236)
    x418 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x418 += einsum("ijkl->ijkl", x236)
    del x236
    x240 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x240 += einsum("ijka->ijka", x129) * -1
    x240 += einsum("ijka->ikja", x129)
    x247 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x247 += einsum("ijka->jkia", x129) * -1
    x247 += einsum("ijka->kjia", x129)
    x276 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x276 += einsum("ijka->jkia", x129) * -1
    x411 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x411 += einsum("ijka->ijka", x129)
    x423 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x423 += einsum("ijka->ijka", x129)
    x423 += einsum("ijka->ikja", x129) * -1
    x482 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x482 += einsum("ijka->ijka", x129) * -1
    x482 += einsum("ijka->ikja", x129)
    del x129
    x483 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x483 += einsum("ijka,jklb->ilab", x1, x482)
    del x482
    l2new_baba += einsum("ijab->baji", x483) * -1
    l2new_abab += einsum("ijab->abij", x483) * -1
    del x483
    x130 += einsum("ijka->ikja", v.bbbb.ooov)
    x131 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x131 += einsum("ijab,jika->kb", t2.bbbb, x130)
    del x130
    x158 += einsum("ia->ia", x131) * -1
    del x131
    x135 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x135 += einsum("iabj->ijba", v.bbbb.ovvo)
    x135 += einsum("ijab->ijab", v.bbbb.oovv) * -1
    x136 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x136 += einsum("ia,ijba->jb", t1.bb, x135)
    x158 += einsum("ia->ia", x136)
    del x136
    x243 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x243 += einsum("ia,jkba->ijkb", t1.bb, x135)
    x246 += einsum("ijka->ijka", x243)
    x257 += einsum("ijka->ijka", x243)
    del x243
    x137 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x137 += einsum("ia,wja->wji", t1.bb, g.bb.bov)
    x138 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x138 += einsum("wij->wij", x137)
    x528 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x528 += einsum("wij->wij", x137)
    del x137
    x138 += einsum("wij->wij", g.bb.boo)
    x139 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x139 += einsum("wia,wij->ja", u11.bb, x138)
    x158 += einsum("ia->ia", x139) * -1
    del x139
    x140 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x140 += einsum("w,wij->ij", s1, g.bb.boo)
    x149 += einsum("ij->ij", x140)
    x396 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x396 += einsum("ij->ij", x140)
    del x140
    x141 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x141 += einsum("wia,wja->ij", g.bb.bov, u11.bb)
    x149 += einsum("ij->ij", x141)
    x396 += einsum("ij->ij", x141)
    del x141
    x142 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x142 += einsum("ia,iajk->jk", t1.aa, v.aabb.ovoo)
    x149 += einsum("ij->ij", x142)
    x400 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x400 += einsum("ij->ij", x142)
    del x142
    x143 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x143 += einsum("ijab,iakb->jk", t2.abab, v.aabb.ovov)
    x149 += einsum("ij->ji", x143)
    x429 += einsum("ij->ij", x143)
    del x143
    x430 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x430 += einsum("ij,abik->kjab", x429, l2.bbbb)
    del x429
    x439 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x439 += einsum("ijab->ijba", x430)
    del x430
    x144 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x144 += einsum("iajb->jiab", v.bbbb.ovov)
    x144 += einsum("iajb->jiba", v.bbbb.ovov) * -1
    x145 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x145 += einsum("ijab,ikab->jk", t2.bbbb, x144)
    x149 += einsum("ij->ji", x145) * -1
    del x145
    x432 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x432 += einsum("ijab,ijac->bc", t2.bbbb, x144)
    x433 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x433 += einsum("ab->ab", x432) * -1
    del x432
    x462 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x462 += einsum("ijab,ikcb->kjca", x144, x54)
    x464 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x464 += einsum("ijab->jiab", x462)
    del x462
    x146 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x146 += einsum("ijka->ikja", v.bbbb.ooov)
    x146 += einsum("ijka->kija", v.bbbb.ooov) * -1
    x147 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x147 += einsum("ia,ijka->jk", t1.bb, x146)
    x149 += einsum("ij->ij", x147) * -1
    del x147
    x285 += einsum("wia,ijka->wjk", u11.bb, x146) * -1
    x149 += einsum("ij->ij", f.bb.oo)
    x150 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x150 += einsum("ia,ij->ja", t1.bb, x149)
    x158 += einsum("ia->ia", x150) * -1
    del x150
    x493 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x493 += einsum("ij,abkj->kiab", x149, l2.abab)
    l2new_baba += einsum("ijab->baji", x493) * -1
    l2new_abab += einsum("ijab->abij", x493) * -1
    del x493
    l1new_bb += einsum("ai,ji->aj", l1.bb, x149) * -1
    del x149
    x151 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x151 += einsum("w,wab->ab", s1, g.bb.bvv)
    x155 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x155 += einsum("ab->ab", x151)
    x389 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x389 += einsum("ab->ab", x151)
    x490 += einsum("ab->ab", x151)
    del x151
    x152 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x152 += einsum("ia,iabc->bc", t1.aa, v.aabb.ovvv)
    x155 += einsum("ab->ab", x152)
    x392 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x392 += einsum("ab->ab", x152)
    x490 += einsum("ab->ab", x152)
    del x152
    x153 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x153 += einsum("iabc->ibca", v.bbbb.ovvv)
    x153 += einsum("iabc->ibac", v.bbbb.ovvv) * -1
    x154 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x154 += einsum("ia,ibac->bc", t1.bb, x153)
    x155 += einsum("ab->ab", x154) * -1
    x490 += einsum("ab->ab", x154) * -1
    del x154
    x155 += einsum("ab->ab", f.bb.vv)
    x156 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x156 += einsum("ia,ba->ib", t1.bb, x155)
    x158 += einsum("ia->ia", x156)
    del x156
    l1new_bb += einsum("ai,ab->bi", l1.bb, x155)
    del x155
    x158 += einsum("ai->ia", f.bb.vo)
    l1new_aa += einsum("ia,baji->bj", x158, l2.abab)
    ls1new += einsum("ia,wai->w", x158, lu11.bb)
    x159 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x159 += einsum("w,wia->ia", ls1, u11.aa)
    x169 += einsum("ia->ia", x159) * -1
    del x159
    x160 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x160 += einsum("ai,jiba->jb", l1.bb, t2.abab)
    x169 += einsum("ia->ia", x160) * -1
    del x160
    x161 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x161 += einsum("ia,waj->wji", t1.aa, lu11.aa)
    x162 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x162 += einsum("wia,wij->ja", u11.aa, x161)
    x169 += einsum("ia->ia", x162)
    del x162
    x166 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x166 += einsum("wai,wja->ij", lu11.aa, u11.aa)
    x167 += einsum("ij->ij", x166)
    x168 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x168 += einsum("ia,ij->ja", t1.aa, x167)
    x169 += einsum("ia->ia", x168)
    del x168
    x498 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x498 += einsum("ij,jakb->ikab", x167, v.aabb.ovov)
    l2new_baba += einsum("ijab->baji", x498) * -1
    l2new_abab += einsum("ijab->abij", x498) * -1
    del x498
    l1new_aa += einsum("ij,jkia->ak", x167, x105) * -1
    del x105
    l1new_bb += einsum("ij,jika->ak", x167, v.aabb.ooov) * -1
    ls1new += einsum("ij,wji->w", x167, g.aa.boo) * -1
    del x167
    x209 += einsum("ij->ij", x166)
    x210 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x210 += einsum("w,ij->wij", s1, x209)
    del x209
    x220 += einsum("ij->ij", x166)
    x335 += einsum("ij->ij", x166)
    del x166
    x336 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x336 += einsum("ij,jakb->ikab", x335, v.aaaa.ovov)
    del x335
    x342 += einsum("ijab->ijba", x336)
    del x336
    x169 += einsum("ia->ia", t1.aa) * -1
    l1new_bb += einsum("ia,iajb->bj", x169, v.aabb.ovov) * -1
    ls1new += einsum("ia,wia->w", x169, g.aa.bov) * -1
    x170 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x170 += einsum("iajb->jiab", v.aaaa.ovov)
    x170 -= einsum("iajb->jiba", v.aaaa.ovov)
    l1new_aa += einsum("ia,ijba->bj", x169, x170) * -1
    del x169
    del x170
    x171 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x171 += einsum("ai,ib->ab", l1.aa, t1.aa)
    x175 += einsum("ab->ab", x171)
    del x171
    x172 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x172 += einsum("wai,wib->ab", lu11.aa, u11.aa)
    x175 += einsum("ab->ab", x172)
    x300 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x300 += einsum("ab,icjb->ijac", x172, v.aaaa.ovov)
    x342 += einsum("ijab->ijab", x300)
    del x300
    x494 += einsum("ab->ab", x172)
    del x172
    x173 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x173 += einsum("abij,ijcb->ac", l2.abab, t2.abab)
    x175 += einsum("ab->ab", x173)
    l1new_aa += einsum("ab,iabc->ci", x175, x51) * -1
    del x51
    l1new_bb += einsum("ab,icab->ci", x175, v.bbaa.ovvv)
    ls1new += einsum("ab,wab->w", x175, g.aa.bvv)
    del x175
    x358 += einsum("ab->ab", x173)
    x359 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x359 += einsum("ab,ibjc->ijac", x358, v.aaaa.ovov)
    del x358
    x362 += einsum("ijab->jiab", x359) * -1
    del x359
    x494 += einsum("ab->ab", x173)
    del x173
    x495 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x495 += einsum("ab,ibjc->ijac", x494, v.aabb.ovov)
    del x494
    l2new_baba += einsum("ijab->baji", x495) * -1
    l2new_abab += einsum("ijab->abij", x495) * -1
    del x495
    x176 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x176 += einsum("w,wia->ia", ls1, u11.bb)
    x190 += einsum("ia->ia", x176) * -1
    del x176
    x177 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x177 += einsum("ai,ijab->jb", l1.aa, t2.abab)
    x190 += einsum("ia->ia", x177) * -1
    del x177
    x178 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x178 += einsum("ia,waj->wji", t1.bb, lu11.bb)
    x179 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x179 += einsum("wia,wij->ja", u11.bb, x178)
    x190 += einsum("ia->ia", x179)
    del x179
    x181 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x181 += einsum("ia,abjk->kjib", t1.bb, l2.bbbb)
    x182 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x182 += einsum("ijka,jiba->kb", x181, x54)
    x190 += einsum("ia->ia", x182) * -1
    del x182
    x260 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x260 += einsum("ijka->ijka", x181)
    x260 += einsum("ijka->jika", x181) * -1
    x262 += einsum("ijka,ilab->jlkb", x260, x35)
    x263 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x263 += einsum("ia,ijkb->jkba", t1.bb, x260) * -1
    x267 += einsum("ijab,jklb->ikla", t2.abab, x260) * -1
    l1new_bb += einsum("ijab,jkia->bk", x135, x260) * -1
    del x135
    l2new_baba += einsum("ijka,likb->abjl", x260, x36)
    del x260
    x269 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x269 += einsum("ia,jkla->kjli", t1.bb, x181)
    x270 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x270 += einsum("ia,ijkl->jlka", t1.bb, x269)
    x271 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x271 += einsum("ijka->ijka", x270)
    del x270
    x280 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x280 += einsum("ijkl->ijkl", x269) * -1
    x280 += einsum("ijkl->ijlk", x269)
    l1new_bb += einsum("ijka,ljki->al", v.bbbb.ooov, x280) * -1
    del x280
    x417 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x417 += einsum("iajb,klij->klab", v.bbbb.ovov, x269)
    del x269
    x420 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x420 += einsum("ijab->ijab", x417)
    del x417
    x376 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x376 += einsum("iabc,jkib->kjac", v.bbbb.ovvv, x181)
    x415 -= einsum("ijab->ijab", x376)
    del x376
    x385 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x385 -= einsum("ijka->ijka", x181)
    x385 += einsum("ijka->jika", x181)
    l2new_abab -= einsum("iajk,kljb->abil", v.aabb.ovoo, x385)
    x422 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x422 += einsum("ijka->ijka", x181) * -1
    x422 += einsum("ijka->jika", x181)
    x424 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x424 += einsum("ijka,jklb->ilab", x422, x423)
    del x423
    x439 += einsum("ijab->ijab", x424)
    del x424
    l2new_abab += einsum("ijka,jlkb->abil", x36, x422) * -1
    del x36
    del x422
    x438 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x438 += einsum("ia,jkib->jkba", x233, x181)
    x439 += einsum("ijab->ijab", x438)
    del x438
    x485 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x485 += einsum("ijka->ijka", x181)
    x485 -= einsum("ijka->jika", x181)
    l2new_baba += einsum("iajk,kljb->bali", v.aabb.ovoo, x485)
    del x485
    x184 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x184 += einsum("ai,ja->ij", l1.bb, t1.bb)
    x188 += einsum("ij->ij", x184)
    x408 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x408 += einsum("ij->ij", x184)
    l1new_bb += einsum("ij,ja->ai", x184, x297) * -1
    del x184
    del x297
    x185 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x185 += einsum("wai,wja->ij", lu11.bb, u11.bb)
    x188 += einsum("ij->ij", x185)
    x293 += einsum("ij->ij", x185)
    x408 += einsum("ij->ij", x185)
    x409 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x409 += einsum("ij,jakb->kiab", x408, v.bbbb.ovov)
    del x408
    x415 += einsum("ijab->jiba", x409)
    del x409
    l1new_bb += einsum("ij,ja->ai", x185, x233) * -1
    del x233
    del x185
    x186 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x186 += einsum("abij,ikab->jk", l2.abab, t2.abab)
    x188 += einsum("ij->ij", x186)
    x189 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x189 += einsum("ia,ij->ja", t1.bb, x188)
    x190 += einsum("ia->ia", x189)
    del x189
    x499 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x499 += einsum("ij,kajb->kiab", x188, v.aabb.ovov)
    l2new_baba += einsum("ijab->baji", x499) * -1
    l2new_abab += einsum("ijab->abij", x499) * -1
    del x499
    l1new_aa += einsum("ij,kaji->ak", x188, v.aabb.ovoo) * -1
    l1new_bb += einsum("ij,jkia->ak", x188, x146) * -1
    del x146
    l1new_bb += einsum("ia,ji->aj", f.bb.ov, x188) * -1
    ls1new += einsum("ij,wji->w", x188, g.bb.boo) * -1
    del x188
    x261 += einsum("ij->ij", x186)
    x262 += einsum("ia,jk->jika", t1.bb, x261)
    l1new_bb += einsum("ijka,jkba->bi", x262, x92)
    del x262
    x267 += einsum("ia,jk->ijka", t1.aa, x261) * -1
    x437 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x437 += einsum("ij,jakb->kiab", x261, v.bbbb.ovov)
    del x261
    x439 += einsum("ijab->jiba", x437) * -1
    del x437
    x293 += einsum("ij->ij", x186)
    del x186
    x294 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x294 += einsum("w,ij->wij", s1, x293)
    del x293
    x190 += einsum("ia->ia", t1.bb) * -1
    l1new_aa += einsum("ia,jbia->bj", x190, v.aabb.ovov) * -1
    l1new_bb += einsum("ia,ijba->bj", x190, x144) * -1
    del x144
    ls1new += einsum("ia,wia->w", x190, g.bb.bov) * -1
    del x190
    x191 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x191 += einsum("ai,ib->ab", l1.bb, t1.bb)
    x195 += einsum("ab->ab", x191)
    del x191
    x192 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x192 += einsum("wai,wib->ab", lu11.bb, u11.bb)
    x195 += einsum("ab->ab", x192)
    x373 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x373 += einsum("ab,icjb->ijac", x192, v.bbbb.ovov)
    x415 += einsum("ijab->ijab", x373)
    del x373
    x496 += einsum("ab->ab", x192)
    del x192
    x193 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x193 += einsum("abij,ijac->bc", l2.abab, t2.abab)
    x195 += einsum("ab->ab", x193)
    l1new_aa += einsum("ab,icab->ci", x195, v.aabb.ovvv)
    l1new_bb += einsum("ab,iacb->ci", x195, x153) * -1
    del x153
    ls1new += einsum("ab,wab->w", x195, g.bb.bvv)
    del x195
    x435 += einsum("ab->ab", x193)
    x436 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x436 += einsum("ab,ibjc->ijca", x435, v.bbbb.ovov)
    del x435
    x439 += einsum("ijab->jiba", x436) * -1
    del x436
    x496 += einsum("ab->ab", x193)
    del x193
    x497 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x497 += einsum("ab,icjb->ijca", x496, v.aabb.ovov)
    del x496
    l2new_baba += einsum("ijab->baji", x497) * -1
    l2new_abab += einsum("ijab->abij", x497) * -1
    del x497
    x196 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x196 += einsum("wx,wia->xia", s2, g.aa.bov)
    x199 += einsum("wia->wia", x196)
    x502 += einsum("wia->wia", x196)
    del x196
    x197 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x197 += einsum("wia,jbia->wjb", u11.bb, v.aabb.ovov)
    x199 += einsum("wia->wia", x197)
    x211 += einsum("wia->wia", x197)
    x333 += einsum("wia->wia", x197)
    x334 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x334 += einsum("wai,wjb->ijab", lu11.aa, x333)
    del x333
    x342 += einsum("ijab->ijab", x334)
    del x334
    x502 += einsum("wia->wia", x197)
    del x197
    x199 += einsum("wia->wia", gc.aa.bov)
    x200 += einsum("ia,wja->wji", t1.aa, x199)
    del x199
    x200 += einsum("wij->wij", gc.aa.boo)
    x200 += einsum("wx,wij->xij", s2, g.aa.boo)
    x200 += einsum("wia,jkia->wjk", u11.bb, v.aabb.ooov)
    l1new_aa += einsum("wai,wji->aj", lu11.aa, x200) * -1
    del x200
    x201 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x201 += einsum("iabc->ibca", v.aaaa.ovvv)
    x201 -= einsum("iabc->ibac", v.aaaa.ovvv)
    x202 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x202 -= einsum("wia,ibac->wbc", u11.aa, x201)
    x306 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x306 += einsum("ia,jbac->ijbc", t1.aa, x201)
    x307 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x307 -= einsum("ijab->ijab", x306)
    del x306
    x319 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x319 += einsum("ia,ibac->bc", t1.aa, x201)
    del x201
    x320 -= einsum("ab->ab", x319)
    del x319
    x321 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x321 += einsum("ab,acij->ijcb", x320, l2.aaaa)
    del x320
    x342 += einsum("ijab->jiab", x321)
    del x321
    x202 += einsum("wab->wab", gc.aa.bvv)
    x202 += einsum("wx,wab->xab", s2, g.aa.bvv)
    x202 += einsum("wia,iabc->wbc", u11.bb, v.bbaa.ovvv)
    l1new_aa += einsum("wai,wab->bi", lu11.aa, x202)
    del x202
    x203 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x203 += einsum("wia,baji->wjb", u11.bb, l2.abab)
    x205 += einsum("wia->wia", x203)
    x507 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x507 += einsum("wia,wjb->jiba", g.bb.bov, x205)
    l2new_baba += einsum("ijab->baji", x507)
    l2new_abab += einsum("ijab->abij", x507)
    del x507
    l1new_aa += einsum("wab,wia->bi", g.aa.bvv, x205)
    l1new_aa += einsum("wia,wji->aj", x205, x98) * -1
    del x98
    del x205
    x208 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x208 += einsum("wia->wia", x203)
    x330 += einsum("wia->wia", x203)
    del x203
    x206 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x206 += einsum("wx,xai->wia", s2, lu11.aa)
    x208 += einsum("wia->wia", x206)
    x330 += einsum("wia->wia", x206)
    del x206
    x331 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x331 += einsum("wia,wjb->ijab", g.aa.bov, x330)
    del x330
    x342 += einsum("ijab->ijab", x331)
    del x331
    x207 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x207 += einsum("abij->jiab", l2.aaaa) * -1
    x207 += einsum("abij->jiba", l2.aaaa)
    x208 += einsum("wia,ijba->wjb", u11.aa, x207) * -1
    x210 += einsum("ia,wja->wji", t1.aa, x208)
    del x208
    x210 += einsum("ai,wja->wij", l1.aa, u11.aa)
    l1new_aa += einsum("wia,wji->aj", g.aa.bov, x210) * -1
    del x210
    x211 += einsum("wia->wia", gc.aa.bov)
    l1new_aa += einsum("wij,wja->ai", x161, x211) * -1
    del x161
    del x211
    x212 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x212 += einsum("iabj->ijba", v.aaaa.ovvo)
    x212 -= einsum("ijab->ijab", v.aaaa.oovv)
    l1new_aa += einsum("ai,jiab->bj", l1.aa, x212)
    lu11new_aa += einsum("wai,jiab->wbj", lu11.aa, x212)
    del x212
    x213 = np.zeros((nbos), dtype=types[float])
    x213 += einsum("w,wx->x", s1, w)
    x219 += einsum("w->w", x213)
    x516 = np.zeros((nbos), dtype=types[float])
    x516 += einsum("w->w", x213)
    del x213
    x214 = np.zeros((nbos), dtype=types[float])
    x214 += einsum("ia,wia->w", t1.aa, gc.aa.bov)
    x219 += einsum("w->w", x214)
    x516 += einsum("w->w", x214)
    del x214
    x215 = np.zeros((nbos), dtype=types[float])
    x215 += einsum("ia,wia->w", t1.bb, gc.bb.bov)
    x219 += einsum("w->w", x215)
    x516 += einsum("w->w", x215)
    del x215
    x219 += einsum("w->w", G)
    l1new_aa += einsum("w,wai->ai", x219, lu11.aa)
    l1new_bb += einsum("w,wai->ai", x219, lu11.bb)
    del x219
    x221 += einsum("ia->ia", f.aa.ov)
    l1new_aa += einsum("ij,ja->ai", x220, x221) * -1
    del x220
    del x221
    x222 = np.zeros((nbos), dtype=types[float])
    x222 += einsum("w,xw->x", ls1, s2)
    x225 = np.zeros((nbos), dtype=types[float])
    x225 += einsum("w->w", x222)
    del x222
    x223 = np.zeros((nbos), dtype=types[float])
    x223 += einsum("ai,wia->w", l1.aa, u11.aa)
    x225 += einsum("w->w", x223)
    del x223
    x224 = np.zeros((nbos), dtype=types[float])
    x224 += einsum("ai,wia->w", l1.bb, u11.bb)
    x225 += einsum("w->w", x224)
    del x224
    x225 += einsum("w->w", s1)
    l1new_aa += einsum("w,wia->ai", x225, g.aa.bov)
    l1new_bb += einsum("w,wia->ai", x225, g.bb.bov)
    del x225
    x226 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x226 += einsum("abij,ikcb->jkac", l2.abab, t2.abab)
    l1new_bb += einsum("iabc,jibc->aj", v.bbaa.ovvv, x226) * -1
    del x226
    x227 += einsum("ia->ia", f.bb.ov)
    x228 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x228 += einsum("ia,jkab->jkib", x227, t2.bbbb)
    del x227
    x229 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x229 += einsum("ijka->kjia", x228)
    del x228
    x229 += einsum("ijak->ijka", v.bbbb.oovo) * -1
    x246 += einsum("ijka->jika", x229) * -1
    x246 += einsum("ijka->kija", x229)
    del x229
    x230 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x230 += einsum("ijab,kacb->ijkc", t2.bbbb, v.bbbb.ovvv)
    x239 += einsum("ijka->ijka", x230) * -1
    del x230
    x231 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x231 += einsum("ia,jabc->ijbc", t1.bb, v.bbbb.ovvv)
    x232 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x232 += einsum("ia,jkba->jikb", t1.bb, x231)
    del x231
    x239 += einsum("ijka->ijka", x232) * -1
    del x232
    x235 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x235 += einsum("ijab,kbla->ijlk", t2.bbbb, v.bbbb.ovov)
    x237 += einsum("ijkl->jilk", x235)
    x238 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x238 += einsum("ia,jkil->jkla", t1.bb, x237)
    del x237
    x239 += einsum("ijka->jika", x238)
    del x238
    x246 += einsum("ijka->ikja", x239) * -1
    x246 += einsum("ijka->jkia", x239)
    del x239
    x418 += einsum("ijkl->jilk", x235)
    del x235
    x419 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x419 += einsum("abij,ijkl->klab", l2.bbbb, x418)
    del x418
    x420 += einsum("ijab->ijab", x419)
    del x419
    x240 += einsum("ijka->jika", v.bbbb.ooov)
    x240 += einsum("ijka->jkia", v.bbbb.ooov) * -1
    x241 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x241 += einsum("ijka,jlba->likb", x240, x35)
    del x240
    del x35
    x246 += einsum("ijka->jkia", x241)
    x257 += einsum("ijka->jkia", x241)
    del x241
    x244 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x244 += einsum("ia,jkla->ijlk", t1.bb, v.bbbb.ooov)
    x245 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x245 += einsum("ijkl->jkli", x244)
    x245 += einsum("ijkl->kjli", x244) * -1
    x256 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x256 += einsum("ijkl->ijkl", x244) * -1
    x256 += einsum("ijkl->ikjl", x244)
    x257 += einsum("ia,jkil->jkla", t1.bb, x256) * -1
    del x256
    l1new_bb += einsum("abij,jkib->ak", l2.bbbb, x257)
    del x257
    x375 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x375 += einsum("abij,jkli->klba", l2.bbbb, x244)
    del x244
    x415 -= einsum("ijab->ijab", x375)
    del x375
    x245 += einsum("ijkl->kijl", v.bbbb.oooo) * -1
    x245 += einsum("ijkl->kilj", v.bbbb.oooo)
    x246 += einsum("ia,ijkl->ljka", t1.bb, x245) * -1
    del x245
    l1new_bb += einsum("abij,jkia->bk", l2.bbbb, x246) * -1
    del x246
    x247 += einsum("ijka->ikja", v.bbbb.ooov)
    x247 += einsum("ijka->kija", v.bbbb.ooov) * -1
    x250 += einsum("ijab,jklb->ikla", t2.abab, x247) * -1
    del x247
    x248 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x248 += einsum("ia,jabc->ijbc", t1.bb, v.bbaa.ovvv)
    x249 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x249 += einsum("ijab->jiab", x248)
    x471 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x471 += einsum("ijab->jiab", x248)
    del x248
    x249 += einsum("ijab->ijab", v.bbaa.oovv)
    x250 += einsum("ia,jkba->ijkb", t1.aa, x249)
    del x249
    x250 += einsum("ijak->kija", v.bbaa.oovo)
    x250 += einsum("ia,jabk->kjib", t1.bb, v.bbaa.ovvo)
    x250 += einsum("ijab,kbca->ikjc", t2.abab, v.bbaa.ovvv)
    l1new_bb += einsum("abij,ikja->bk", l2.abab, x250) * -1
    del x250
    x251 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x251 += einsum("abij,ikac->jkbc", l2.abab, t2.abab)
    x254 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x254 += einsum("ijab->ijab", x251)
    del x251
    x252 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x252 += einsum("abij->jiab", l2.bbbb) * -1
    x252 += einsum("abij->jiba", l2.bbbb)
    x253 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x253 += einsum("ijab,ikbc->kjca", x252, x54)
    del x54
    x254 += einsum("ijab->jiba", x253)
    del x253
    x421 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x421 += einsum("ijab,jkbc->kica", x254, x92)
    del x92
    x439 += einsum("ijab->jiba", x421) * -1
    del x421
    x292 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x292 += einsum("wia,ijba->wjb", u11.bb, x252) * -1
    x425 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x425 += einsum("ijab,jkcb->ikac", t2.abab, x252)
    x426 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x426 += einsum("iajb,ikac->jkbc", v.aabb.ovov, x425) * -1
    del x425
    x439 += einsum("ijab->jiba", x426) * -1
    del x426
    x255 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x255 += einsum("iabc->ibca", v.bbbb.ovvv) * -1
    x255 += einsum("iabc->ibac", v.bbbb.ovvv)
    x463 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x463 += einsum("ia,jbac->ijbc", t1.bb, x255)
    x464 += einsum("ijab->jiab", x463) * -1
    del x463
    l1new_bb += einsum("ijab,jacb->ci", x254, x255) * -1
    del x254
    l1new_bb += einsum("iabc,jiac->bj", x255, x263)
    del x255
    del x263
    x258 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x258 += einsum("abij->jiab", l2.bbbb)
    x258 += einsum("abij->jiba", l2.bbbb) * -1
    x259 += einsum("ijab,jkcb->ikac", t2.abab, x258) * -1
    del x258
    l1new_bb += einsum("iabc,ijab->cj", v.aabb.ovvv, x259) * -1
    del x259
    x264 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x264 += einsum("ia,bacd->icbd", t1.bb, v.bbbb.vvvv)
    x265 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x265 += einsum("iabc->iabc", x264) * -1
    del x264
    x265 += einsum("aibc->iabc", v.bbbb.vovv)
    x266 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x266 += einsum("iabc->iabc", x265) * -1
    x266 += einsum("iabc->ibac", x265)
    del x265
    l1new_bb += einsum("abij,iabc->cj", l2.bbbb, x266) * -1
    del x266
    x267 += einsum("ai,jkba->jikb", l1.bb, t2.abab) * -1
    l1new_bb += einsum("iajb,ikja->bk", v.aabb.ovov, x267)
    del x267
    x268 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x268 += einsum("ai,jkab->ikjb", l1.bb, t2.bbbb)
    x271 += einsum("ijka->ijka", x268) * 0.9999999999999993
    del x268
    x272 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x272 += einsum("ijka->ijka", x271) * -1
    x272 += einsum("ijka->ikja", x271)
    del x271
    l1new_bb += einsum("iajb,kija->bk", v.bbbb.ovov, x272)
    del x272
    x273 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x273 += einsum("aibc->iabc", v.aabb.vovv)
    x273 += einsum("ia,bacd->ibcd", t1.aa, v.aabb.vvvv)
    l1new_bb += einsum("abij,iabc->cj", l2.abab, x273)
    del x273
    x274 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x274 += einsum("abij,klab->ijkl", l2.bbbb, t2.bbbb)
    x275 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x275 += einsum("ijkl->jikl", x274)
    x275 += einsum("ijkl->jilk", x274) * -1
    x416 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x416 += einsum("iajb,klij->lkba", v.bbbb.ovov, x274)
    del x274
    x420 += einsum("ijab->ijab", x416)
    del x416
    l2new_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    l2new_bbbb += einsum("ijab->abij", x420)
    l2new_bbbb += einsum("ijab->baij", x420) * -1
    del x420
    x276 += einsum("ijka->ikja", v.bbbb.ooov)
    l1new_bb += einsum("ijkl,klia->aj", x275, x276) * -1
    del x275
    del x276
    x277 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x277 += einsum("ia,jbca->ijcb", t1.aa, v.bbaa.ovvv)
    x278 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x278 += einsum("ijab->ijab", x277)
    x383 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x383 += einsum("ijab->ijab", x277)
    x450 += einsum("ijab->ijab", x277)
    del x277
    x278 += einsum("iabj->jiba", v.bbaa.ovvo)
    l1new_bb += einsum("ijka,ikab->bj", x1, x278) * -1
    del x278
    x279 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x279 -= einsum("abij->jiab", l2.bbbb)
    x279 += einsum("abij->jiba", l2.bbbb)
    x289 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x289 += einsum("wia,ijba->wjb", u11.bb, x279)
    x290 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x290 -= einsum("wia->wia", x289)
    x402 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x402 -= einsum("wia->wia", x289)
    del x289
    l1new_bb += einsum("ia,ijba->bj", x158, x279) * -1
    del x158
    lu11new_bb -= einsum("wai,ijba->wbj", g.bb.bvo, x279)
    x281 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x281 += einsum("wx,wia->xia", s2, g.bb.bov)
    x284 += einsum("wia->wia", x281)
    x504 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x504 += einsum("wia->wia", x281)
    del x281
    x282 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x282 += einsum("wia,iajb->wjb", u11.aa, v.aabb.ovov)
    x284 += einsum("wia->wia", x282)
    x295 += einsum("wia->wia", x282)
    x406 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x406 += einsum("wia->wia", x282)
    x504 += einsum("wia->wia", x282)
    del x282
    x284 += einsum("wia->wia", gc.bb.bov)
    x285 += einsum("ia,wja->wji", t1.bb, x284)
    del x284
    x285 += einsum("wij->wij", gc.bb.boo)
    x285 += einsum("wx,wij->xij", s2, g.bb.boo)
    x285 += einsum("wia,iajk->wjk", u11.aa, v.aabb.ovoo)
    l1new_bb += einsum("wai,wji->aj", lu11.bb, x285) * -1
    del x285
    x286 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x286 += einsum("iabc->ibca", v.bbbb.ovvv)
    x286 -= einsum("iabc->ibac", v.bbbb.ovvv)
    x287 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x287 -= einsum("wia,ibac->wbc", u11.bb, x286)
    x378 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x378 += einsum("ia,jbac->ijbc", t1.bb, x286)
    x379 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x379 -= einsum("ijab->ijab", x378)
    del x378
    x391 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x391 += einsum("ia,ibac->bc", t1.bb, x286)
    del x286
    x392 -= einsum("ab->ab", x391)
    del x391
    x393 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x393 += einsum("ab,acij->ijcb", x392, l2.bbbb)
    del x392
    x415 += einsum("ijab->jiab", x393)
    del x393
    x287 += einsum("wab->wab", gc.bb.bvv)
    x287 += einsum("wx,wab->xab", s2, g.bb.bvv)
    x287 += einsum("wia,iabc->wbc", u11.aa, v.aabb.ovvv)
    l1new_bb += einsum("wai,wab->bi", lu11.bb, x287)
    del x287
    x288 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x288 += einsum("wia,abij->wjb", u11.aa, l2.abab)
    x290 += einsum("wia->wia", x288)
    x506 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x506 += einsum("wia,wjb->ijab", g.aa.bov, x290)
    l2new_baba += einsum("ijab->baji", x506)
    l2new_abab += einsum("ijab->abij", x506)
    del x506
    l1new_bb += einsum("wab,wia->bi", g.bb.bvv, x290)
    l1new_bb += einsum("wij,wja->ai", x138, x290) * -1
    del x138
    del x290
    x292 += einsum("wia->wia", x288)
    x402 += einsum("wia->wia", x288)
    del x288
    x291 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x291 += einsum("wx,xai->wia", s2, lu11.bb)
    x292 += einsum("wia->wia", x291)
    x294 += einsum("ia,wja->wji", t1.bb, x292)
    del x292
    x402 += einsum("wia->wia", x291)
    del x291
    x403 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x403 += einsum("wia,wjb->ijab", g.bb.bov, x402)
    del x402
    x415 += einsum("ijab->ijab", x403)
    del x403
    x294 += einsum("ai,wja->wij", l1.bb, u11.bb)
    l1new_bb += einsum("wia,wji->aj", g.bb.bov, x294) * -1
    del x294
    x295 += einsum("wia->wia", gc.bb.bov)
    l1new_bb += einsum("wij,wja->ai", x178, x295) * -1
    del x178
    del x295
    x296 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x296 += einsum("iabj->ijba", v.bbbb.ovvo)
    x296 -= einsum("ijab->ijab", v.bbbb.oovv)
    l1new_bb += einsum("ai,jiab->bj", l1.bb, x296)
    lu11new_bb += einsum("wai,jiab->wbj", lu11.bb, x296)
    del x296
    x298 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x298 += einsum("wia,wbj->ijab", gc.aa.bov, lu11.aa)
    x342 += einsum("ijab->ijab", x298)
    del x298
    x299 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x299 += einsum("ai,jbac->ijbc", l1.aa, v.aaaa.ovvv)
    x342 -= einsum("ijab->ijab", x299)
    del x299
    x305 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x305 += einsum("abij->jiab", l2.aaaa)
    x305 -= einsum("abij->jiba", l2.aaaa)
    x307 += einsum("ijab->jiab", v.aaaa.oovv)
    x307 -= einsum("iabj->jiba", v.aaaa.ovvo)
    x308 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x308 += einsum("ijab,ikac->jkbc", x305, x307)
    del x305
    del x307
    x342 += einsum("ijab->ijab", x308)
    del x308
    x309 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x309 -= einsum("ijab->jiab", t2.bbbb)
    x309 += einsum("ijab->jiba", t2.bbbb)
    x310 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x310 += einsum("iajb,jkcb->ikac", v.aabb.ovov, x309)
    x311 -= einsum("ijab->ijab", x310)
    del x310
    x527 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x527 += einsum("wia,ijba->wjb", g.bb.bov, x309)
    del x309
    x530 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x530 -= einsum("wia->wia", x527)
    del x527
    x311 += einsum("iabj->ijab", v.aabb.ovvo)
    x312 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x312 += einsum("abij,kjcb->ikac", l2.abab, x311)
    del x311
    x342 += einsum("ijab->ijab", x312)
    del x312
    x314 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x314 -= einsum("ijka->ikja", v.aaaa.ooov)
    x314 += einsum("ijka->kija", v.aaaa.ooov)
    x315 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x315 += einsum("ijka,lkib->jlab", x313, x314)
    del x313
    x342 += einsum("ijab->ijab", x315)
    del x315
    l2new_abab += einsum("ijka,kilb->abjl", x314, x56)
    del x314
    x316 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x316 += einsum("wia,wib->ab", g.aa.bov, u11.aa)
    x317 -= einsum("ab->ba", x316)
    x487 += einsum("ab->ba", x316) * -1
    del x316
    x317 += einsum("ab->ab", f.aa.vv)
    x318 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x318 += einsum("ab,acij->ijcb", x317, l2.aaaa)
    del x317
    x342 -= einsum("ijab->jiba", x318)
    del x318
    x322 += einsum("ia->ia", f.aa.ov)
    x323 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x323 += einsum("ia,ja->ij", t1.aa, x322)
    x324 += einsum("ij->ji", x323)
    del x323
    x337 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x337 += einsum("ia,jkib->jkba", x322, x57)
    del x57
    x342 += einsum("ijab->ijba", x337)
    del x337
    x342 += einsum("ai,jb->jiba", l1.aa, x322)
    del x322
    x324 += einsum("ij->ij", f.aa.oo)
    x325 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x325 += einsum("ij,abjk->kiab", x324, l2.aaaa)
    del x324
    x342 += einsum("ijab->jiba", x325)
    del x325
    x326 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x326 += einsum("ijka->ikja", v.aaaa.ooov)
    x326 -= einsum("ijka->kija", v.aaaa.ooov)
    x327 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x327 += einsum("ia,ijka->jk", t1.aa, x326)
    x328 -= einsum("ij->ij", x327)
    del x327
    x329 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x329 += einsum("ij,abjk->kiab", x328, l2.aaaa)
    del x328
    x342 -= einsum("ijab->ijba", x329)
    del x329
    l2new_baba -= einsum("ijka,kilb->balj", x326, x56)
    del x56
    del x326
    x338 -= einsum("ijka->jika", v.aaaa.ooov)
    x339 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x339 += einsum("ai,ijkb->jkab", l1.aa, x338)
    del x338
    x342 += einsum("ijab->ijab", x339)
    del x339
    l2new_aaaa += einsum("ijab->abij", x342)
    l2new_aaaa -= einsum("ijab->baij", x342)
    l2new_aaaa -= einsum("ijab->abji", x342)
    l2new_aaaa += einsum("ijab->baji", x342)
    del x342
    x343 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x343 += einsum("ijab,kcjb->ikac", t2.abab, v.aabb.ovov)
    x345 += einsum("ijab->ijab", x343)
    x346 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x346 += einsum("ijab,ikac->kjcb", x345, x48)
    del x48
    del x345
    x362 += einsum("ijab->ijab", x346)
    del x346
    x459 += einsum("ijab->jiab", x343)
    del x343
    x354 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x354 += einsum("ijab,icjb->ac", t2.abab, v.aabb.ovov)
    x356 += einsum("ab->ab", x354)
    x357 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x357 += einsum("ab,acij->ijcb", x356, l2.aaaa)
    del x356
    x362 += einsum("ijab->jiab", x357)
    del x357
    l2new_aaaa += einsum("ijab->abij", x362) * -1
    l2new_aaaa += einsum("ijab->baij", x362)
    l2new_aaaa += einsum("ijab->abji", x362)
    l2new_aaaa += einsum("ijab->baji", x362) * -1
    del x362
    x487 += einsum("ab->ab", x354) * -1
    del x354
    x363 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x363 += einsum("abij,kjli->klba", l2.aaaa, v.aaaa.oooo)
    x365 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x365 += einsum("ijab->jiba", x363)
    del x363
    x364 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x364 += einsum("abij,acbd->ijcd", l2.aaaa, v.aaaa.vvvv)
    x365 += einsum("ijab->jiba", x364)
    del x364
    l2new_aaaa += einsum("ijab->baij", x365) * -1
    l2new_aaaa += einsum("ijab->abij", x365)
    del x365
    x371 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x371 += einsum("wia,wbj->ijab", gc.bb.bov, lu11.bb)
    x415 += einsum("ijab->ijab", x371)
    del x371
    x372 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x372 += einsum("ai,jbac->ijbc", l1.bb, v.bbbb.ovvv)
    x415 -= einsum("ijab->ijab", x372)
    del x372
    x379 += einsum("ijab->jiab", v.bbbb.oovv)
    x379 -= einsum("iabj->jiba", v.bbbb.ovvo)
    x380 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x380 += einsum("ijab,ikbc->jkac", x279, x379)
    del x379
    del x279
    x415 += einsum("ijab->ijab", x380)
    del x380
    x381 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x381 -= einsum("ijab->jiab", t2.aaaa)
    x381 += einsum("ijab->jiba", t2.aaaa)
    x382 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x382 += einsum("iajb,ikca->kjcb", v.aabb.ovov, x381)
    x383 -= einsum("ijab->ijab", x382)
    del x382
    x520 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x520 += einsum("wia,ijba->wjb", g.aa.bov, x381)
    del x381
    x523 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x523 -= einsum("wia->wia", x520)
    del x520
    x383 += einsum("iabj->jiba", v.bbaa.ovvo)
    x384 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x384 += einsum("abij,ikac->jkbc", l2.abab, x383)
    del x383
    x415 += einsum("ijab->ijab", x384)
    del x384
    x386 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x386 -= einsum("ijka->ikja", v.bbbb.ooov)
    x386 += einsum("ijka->kija", v.bbbb.ooov)
    x387 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x387 += einsum("ijka,lkjb->ilab", x385, x386)
    del x385
    del x386
    x415 += einsum("ijab->ijab", x387)
    del x387
    x388 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x388 += einsum("wia,wib->ab", g.bb.bov, u11.bb)
    x389 -= einsum("ab->ba", x388)
    x490 += einsum("ab->ba", x388) * -1
    del x388
    x389 += einsum("ab->ab", f.bb.vv)
    x390 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x390 += einsum("ab,acij->ijcb", x389, l2.bbbb)
    del x389
    x415 -= einsum("ijab->jiba", x390)
    del x390
    x394 += einsum("ia->ia", f.bb.ov)
    x395 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x395 += einsum("ia,ja->ij", t1.bb, x394)
    x396 += einsum("ij->ji", x395)
    del x395
    x410 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x410 += einsum("ia,jkib->jkba", x394, x181)
    del x181
    x415 += einsum("ijab->ijba", x410)
    del x410
    x415 += einsum("ai,jb->jiba", l1.bb, x394)
    del x394
    x396 += einsum("ij->ij", f.bb.oo)
    x397 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x397 += einsum("ij,abjk->kiab", x396, l2.bbbb)
    del x396
    x415 += einsum("ijab->jiba", x397)
    del x397
    x398 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x398 += einsum("ijka->ikja", v.bbbb.ooov)
    x398 -= einsum("ijka->kija", v.bbbb.ooov)
    x399 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x399 += einsum("ia,ijka->jk", t1.bb, x398)
    x400 -= einsum("ij->ij", x399)
    del x399
    x401 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x401 += einsum("ij,abjk->kiab", x400, l2.bbbb)
    del x400
    x415 -= einsum("ijab->ijba", x401)
    del x401
    x484 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x484 += einsum("ijka,kljb->ilab", x1, x398)
    del x1
    del x398
    l2new_baba -= einsum("ijab->baji", x484)
    l2new_abab -= einsum("ijab->abij", x484)
    del x484
    x404 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x404 -= einsum("iajb->jiab", v.bbbb.ovov)
    x404 += einsum("iajb->jiba", v.bbbb.ovov)
    x405 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x405 += einsum("wia,ijba->wjb", u11.bb, x404)
    x406 -= einsum("wia->wia", x405)
    x407 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x407 += einsum("wai,wjb->ijab", lu11.bb, x406)
    del x406
    x415 += einsum("ijab->ijab", x407)
    del x407
    x504 -= einsum("wia->wia", x405)
    del x405
    x413 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x413 += einsum("ia,ijba->jb", t1.bb, x404)
    del x404
    x414 -= einsum("ia->ia", x413)
    x415 += einsum("ai,jb->ijab", l1.bb, x414)
    del x414
    x513 -= einsum("ia->ia", x413)
    del x413
    x411 -= einsum("ijka->jika", v.bbbb.ooov)
    x412 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x412 += einsum("ai,ijkb->jkab", l1.bb, x411)
    del x411
    x415 += einsum("ijab->ijab", x412)
    del x412
    l2new_bbbb += einsum("ijab->abij", x415)
    l2new_bbbb -= einsum("ijab->baij", x415)
    l2new_bbbb -= einsum("ijab->abji", x415)
    l2new_bbbb += einsum("ijab->baji", x415)
    del x415
    x431 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x431 += einsum("ijab,iajc->bc", t2.abab, v.aabb.ovov)
    x433 += einsum("ab->ab", x431)
    x434 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x434 += einsum("ab,acij->ijcb", x433, l2.bbbb)
    del x433
    x439 += einsum("ijab->jiab", x434)
    del x434
    l2new_bbbb += einsum("ijab->abij", x439) * -1
    l2new_bbbb += einsum("ijab->baij", x439)
    l2new_bbbb += einsum("ijab->abji", x439)
    l2new_bbbb += einsum("ijab->baji", x439) * -1
    del x439
    x490 += einsum("ab->ab", x431) * -1
    del x431
    x440 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x440 += einsum("abij,kjli->klba", l2.bbbb, v.bbbb.oooo)
    x442 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x442 += einsum("ijab->jiba", x440)
    del x440
    x441 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x441 += einsum("abij,bcad->ijdc", l2.bbbb, v.bbbb.vvvv)
    x442 += einsum("ijab->jiba", x441)
    del x441
    l2new_bbbb += einsum("ijab->baij", x442) * -1
    l2new_bbbb += einsum("ijab->abij", x442)
    del x442
    x443 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x443 += einsum("ai,jbac->ijcb", l1.aa, v.bbaa.ovvv)
    l2new_baba += einsum("ijab->baji", x443)
    l2new_abab += einsum("ijab->abij", x443)
    del x443
    x445 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x445 += einsum("abij,acbd->ijcd", l2.abab, v.aabb.vvvv)
    l2new_baba += einsum("ijab->baji", x445)
    l2new_abab += einsum("ijab->abij", x445)
    del x445
    x447 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x447 += einsum("ai,jbac->jibc", l1.bb, v.aabb.ovvv)
    l2new_baba += einsum("ijab->baji", x447)
    l2new_abab += einsum("ijab->abij", x447)
    del x447
    x450 += einsum("iabj->jiba", v.bbaa.ovvo)
    x451 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x451 += einsum("ijab,ikbc->jkac", x207, x450)
    del x450
    del x207
    l2new_baba += einsum("ijab->baji", x451) * -1
    l2new_abab += einsum("ijab->abij", x451) * -1
    del x451
    x452 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x452 += einsum("abij->jiab", l2.bbbb)
    x452 -= einsum("abij->jiba", l2.bbbb)
    x455 += einsum("iabj->ijab", v.aabb.ovvo)
    l2new_baba += einsum("ijab,kicb->acjk", x452, x455)
    del x452
    l2new_abab += einsum("ijab,kicb->cakj", x252, x455) * -1
    del x252
    del x455
    x456 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x456 += einsum("iajb->jiab", v.aaaa.ovov)
    x456 += einsum("iajb->jiba", v.aaaa.ovov) * -1
    x457 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x457 += einsum("ijab,ikca->jkbc", x24, x456)
    del x24
    del x456
    x459 += einsum("ijab->jiab", x457)
    del x457
    x459 += einsum("iabj->ijba", v.aaaa.ovvo)
    x459 += einsum("ijab->ijab", v.aaaa.oovv) * -1
    x460 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x460 += einsum("abij,kiac->kjcb", l2.abab, x459)
    del x459
    l2new_baba += einsum("ijab->baji", x460)
    l2new_abab += einsum("ijab->abij", x460)
    del x460
    x461 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x461 += einsum("ijab,iakc->jkbc", t2.abab, v.aabb.ovov)
    x464 += einsum("ijab->jiab", x461)
    del x461
    x464 += einsum("iabj->ijba", v.bbbb.ovvo)
    x464 += einsum("ijab->ijab", v.bbbb.oovv) * -1
    x465 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x465 += einsum("abij,kjbc->ikac", l2.abab, x464)
    del x464
    l2new_baba += einsum("ijab->baji", x465)
    l2new_abab += einsum("ijab->abij", x465)
    del x465
    x466 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x466 += einsum("ia,jabc->ijbc", t1.aa, v.aabb.ovvv)
    x468 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x468 += einsum("ijab->jiab", x466)
    del x466
    x467 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x467 += einsum("ijab,kajc->ikbc", t2.abab, v.aabb.ovov)
    x468 += einsum("ijab->jiab", x467) * -1
    del x467
    x468 += einsum("ijab->ijab", v.aabb.oovv)
    x469 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x469 += einsum("abij,kibc->kjac", l2.abab, x468)
    del x468
    l2new_baba += einsum("ijab->baji", x469) * -1
    l2new_abab += einsum("ijab->abij", x469) * -1
    del x469
    x470 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x470 += einsum("ijab,ickb->jkac", t2.abab, v.aabb.ovov)
    x471 += einsum("ijab->jiab", x470) * -1
    del x470
    x471 += einsum("ijab->ijab", v.bbaa.oovv)
    x472 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x472 += einsum("abij,kjac->ikcb", l2.abab, x471)
    del x471
    l2new_baba += einsum("ijab->baji", x472) * -1
    l2new_abab += einsum("ijab->abij", x472) * -1
    del x472
    x474 += einsum("ijkl->ijkl", v.aabb.oooo)
    x475 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x475 += einsum("abij,kilj->klab", l2.abab, x474)
    del x474
    l2new_baba += einsum("ijab->baji", x475)
    l2new_abab += einsum("ijab->abij", x475)
    del x475
    x487 += einsum("ab->ab", f.aa.vv)
    x488 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x488 += einsum("ab,acij->ijbc", x487, l2.abab)
    del x487
    l2new_baba += einsum("ijab->baji", x488)
    l2new_abab += einsum("ijab->abij", x488)
    del x488
    x490 += einsum("ab->ab", f.bb.vv)
    x491 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x491 += einsum("ab,caij->ijcb", x490, l2.abab)
    del x490
    l2new_baba += einsum("ijab->baji", x491)
    l2new_abab += einsum("ijab->abij", x491)
    del x491
    x502 += einsum("wia->wia", gc.aa.bov)
    x503 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x503 += einsum("wai,wjb->jiba", lu11.bb, x502)
    del x502
    l2new_baba += einsum("ijab->baji", x503)
    l2new_abab += einsum("ijab->abij", x503)
    del x503
    x504 += einsum("wia->wia", gc.bb.bov)
    x505 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x505 += einsum("wai,wjb->ijab", lu11.aa, x504)
    del x504
    l2new_baba += einsum("ijab->baji", x505)
    l2new_abab += einsum("ijab->abij", x505)
    del x505
    x508 += einsum("ijka->ijka", v.aabb.ooov)
    x509 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x509 += einsum("ai,jikb->jkab", l1.aa, x508)
    del x508
    l2new_baba -= einsum("ijab->baji", x509)
    l2new_abab -= einsum("ijab->abij", x509)
    del x509
    x510 += einsum("iajk->ijka", v.aabb.ovoo)
    x511 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x511 += einsum("ai,jkib->jkba", l1.bb, x510)
    del x510
    l2new_baba -= einsum("ijab->baji", x511)
    l2new_abab -= einsum("ijab->abij", x511)
    del x511
    x512 += einsum("ia->ia", f.aa.ov)
    x516 += einsum("ia,wia->w", x512, u11.aa)
    l2new_baba += einsum("ai,jb->abij", l1.bb, x512)
    l2new_abab += einsum("ai,jb->baji", l1.bb, x512)
    del x512
    x513 += einsum("ia->ia", f.bb.ov)
    x516 += einsum("ia,wia->w", x513, u11.bb)
    l2new_baba += einsum("ai,jb->baji", l1.aa, x513)
    l2new_abab += einsum("ai,jb->abij", l1.aa, x513)
    del x513
    x515 += einsum("w->w", G)
    x516 += einsum("w,wx->x", x515, s2)
    x536 = np.zeros((nbos, nbos), dtype=types[float])
    x536 += einsum("w,x->xw", ls1, x515)
    del x515
    x516 += einsum("w->w", G)
    ls1new += einsum("w,wx->x", x516, ls2)
    del x516
    x517 = np.zeros((nbos, nbos), dtype=types[float])
    x517 += einsum("wx,yx->wy", ls2, w)
    x536 += einsum("wx->wx", x517)
    del x517
    x518 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x518 += einsum("ia,wba->wib", t1.aa, g.aa.bvv)
    x523 += einsum("wia->wia", x518)
    del x518
    x519 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x519 += einsum("wia,jiba->wjb", g.bb.bov, t2.abab)
    x523 += einsum("wia->wia", x519)
    del x519
    x521 += einsum("wij->wij", g.aa.boo)
    x522 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x522 += einsum("ia,wij->wja", t1.aa, x521)
    del x521
    x523 -= einsum("wia->wia", x522)
    del x522
    x523 += einsum("wai->wia", g.aa.bvo)
    x524 = np.zeros((nbos, nbos), dtype=types[float])
    x524 += einsum("wai,xia->wx", lu11.aa, x523)
    del x523
    x536 += einsum("wx->xw", x524)
    del x524
    x525 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x525 += einsum("ia,wba->wib", t1.bb, g.bb.bvv)
    x530 += einsum("wia->wia", x525)
    del x525
    x526 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x526 += einsum("wia,ijab->wjb", g.aa.bov, t2.abab)
    x530 += einsum("wia->wia", x526)
    del x526
    x528 += einsum("wij->wij", g.bb.boo)
    x529 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x529 += einsum("ia,wij->wja", t1.bb, x528)
    del x528
    x530 -= einsum("wia->wia", x529)
    del x529
    x530 += einsum("wai->wia", g.bb.bvo)
    x531 = np.zeros((nbos, nbos), dtype=types[float])
    x531 += einsum("wai,xia->wx", lu11.bb, x530)
    del x530
    x536 += einsum("wx->xw", x531)
    del x531
    x532 = np.zeros((nbos, nbos), dtype=types[float])
    x532 += einsum("wia,xia->wx", g.aa.bov, u11.aa)
    x534 = np.zeros((nbos, nbos), dtype=types[float])
    x534 += einsum("wx->wx", x532)
    del x532
    x533 = np.zeros((nbos, nbos), dtype=types[float])
    x533 += einsum("wia,xia->wx", g.bb.bov, u11.bb)
    x534 += einsum("wx->wx", x533)
    del x533
    x535 = np.zeros((nbos, nbos), dtype=types[float])
    x535 += einsum("wx,yw->xy", ls2, x534)
    del x534
    x536 += einsum("wx->xw", x535)
    del x535
    ls2new = np.zeros((nbos, nbos), dtype=types[float])
    ls2new += einsum("wx->wx", x536)
    ls2new += einsum("wx->xw", x536)
    del x536
    l1new_aa += einsum("ia->ai", f.aa.ov)
    l1new_aa += einsum("w,wia->ai", ls1, gc.aa.bov)
    l1new_aa += einsum("ai,jbai->bj", l1.bb, v.aabb.ovvo)
    l1new_bb += einsum("ai,jbai->bj", l1.aa, v.bbaa.ovvo)
    l1new_bb += einsum("ia->ai", f.bb.ov)
    l1new_bb += einsum("w,wia->ai", ls1, gc.bb.bov)
    l2new_aaaa -= einsum("iajb->abji", v.aaaa.ovov)
    l2new_aaaa += einsum("iajb->baji", v.aaaa.ovov)
    l2new_bbbb -= einsum("iajb->abji", v.bbbb.ovov)
    l2new_bbbb += einsum("iajb->baji", v.bbbb.ovov)
    l2new_baba += einsum("iajb->baji", v.aabb.ovov)
    l2new_abab += einsum("iajb->abij", v.aabb.ovov)
    ls1new += einsum("ai,wai->w", l1.aa, g.aa.bvo)
    ls1new += einsum("w->w", G)
    ls1new += einsum("w,xw->x", ls1, w)
    ls1new += einsum("ai,wai->w", l1.bb, g.bb.bvo)
    lu11new_aa += einsum("ab,wai->wbi", f.aa.vv, lu11.aa)
    lu11new_aa += einsum("ai,wab->wbi", l1.aa, g.aa.bvv)
    lu11new_aa += einsum("wx,xia->wai", ls2, gc.aa.bov)
    lu11new_aa += einsum("wx,xai->wai", w, lu11.aa)
    lu11new_aa -= einsum("ij,waj->wai", f.aa.oo, lu11.aa)
    lu11new_aa -= einsum("ai,wji->waj", l1.aa, g.aa.boo)
    lu11new_aa += einsum("wia->wai", g.aa.bov)
    lu11new_aa += einsum("wai,baji->wbj", g.bb.bvo, l2.abab)
    lu11new_aa += einsum("wai,jbai->wbj", lu11.bb, v.aabb.ovvo)
    lu11new_bb -= einsum("ij,waj->wai", f.bb.oo, lu11.bb)
    lu11new_bb -= einsum("ai,wji->waj", l1.bb, g.bb.boo)
    lu11new_bb += einsum("wx,xia->wai", ls2, gc.bb.bov)
    lu11new_bb += einsum("wx,xai->wai", w, lu11.bb)
    lu11new_bb += einsum("wia->wai", g.bb.bov)
    lu11new_bb += einsum("ab,wai->wbi", f.bb.vv, lu11.bb)
    lu11new_bb += einsum("ai,wab->wbi", l1.bb, g.bb.bvv)
    lu11new_bb += einsum("wai,abij->wbj", g.aa.bvo, l2.abab)
    lu11new_bb += einsum("wai,jbai->wbj", lu11.aa, v.bbaa.ovvo)

    l1new.aa = l1new_aa
    l1new.bb = l1new_bb
    l2new.abab = l2new_abab
    l2new.baba = l2new_baba
    l2new.aaaa = l2new_aaaa
    l2new.bbbb = l2new_bbbb
    lu11new.aa = lu11new_aa
    lu11new.bb = lu11new_bb

    return {"l1new": l1new, "l2new": l2new, "ls1new": ls1new, "ls2new": ls2new, "lu11new": lu11new}

def make_rdm1_f(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, **kwargs):
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
    x14 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x14 += einsum("ij->ij", x0)
    rdm1_f_oo_aa = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    rdm1_f_oo_aa += einsum("ij->ij", x0) * -1
    del x0
    x1 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x1 += einsum("ai,ja->ij", l1.aa, t1.aa)
    x14 += einsum("ij->ij", x1)
    rdm1_f_oo_aa -= einsum("ij->ij", x1)
    del x1
    x2 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x2 += einsum("wai,wja->ij", lu11.aa, u11.aa)
    x14 += einsum("ij->ij", x2)
    rdm1_f_oo_aa -= einsum("ij->ij", x2)
    del x2
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x3 += einsum("ijab->jiab", t2.aaaa)
    x3 += einsum("ijab->jiba", t2.aaaa) * -1
    rdm1_f_oo_aa += einsum("abij,ikba->jk", l2.aaaa, x3) * -1
    del x3
    x4 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x4 += einsum("wai,wja->ij", lu11.bb, u11.bb)
    x21 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x21 += einsum("ij->ij", x4)
    rdm1_f_oo_bb = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    rdm1_f_oo_bb -= einsum("ij->ij", x4)
    del x4
    x5 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x5 += einsum("abij,ikab->jk", l2.abab, t2.abab)
    x21 += einsum("ij->ij", x5)
    rdm1_f_oo_bb += einsum("ij->ij", x5) * -1
    del x5
    x6 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x6 += einsum("ai,ja->ij", l1.bb, t1.bb)
    x21 += einsum("ij->ij", x6)
    rdm1_f_oo_bb -= einsum("ij->ij", x6)
    del x6
    x7 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x7 += einsum("ijab->jiab", t2.bbbb)
    x7 += einsum("ijab->jiba", t2.bbbb) * -1
    rdm1_f_oo_bb += einsum("abij,ikba->jk", l2.bbbb, x7) * -1
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
    x10 += einsum("ia,bajk->jkib", t1.aa, l2.aaaa)
    x11 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x11 += einsum("ijka->ijka", x10) * -1
    x11 += einsum("ijka->jika", x10)
    del x10
    rdm1_f_vo_aa += einsum("ijab,ijkb->ak", t2.aaaa, x11) * -1
    del x11
    x12 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x12 -= einsum("ijab->jiab", t2.aaaa)
    x12 += einsum("ijab->jiba", t2.aaaa)
    rdm1_f_vo_aa -= einsum("ai,ijba->bj", l1.aa, x12)
    del x12
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x13 += einsum("ijab->jiab", t2.aaaa) * -1
    x13 += einsum("ijab->jiba", t2.aaaa)
    x14 += einsum("abij,ikba->jk", l2.aaaa, x13) * -1
    del x13
    rdm1_f_vo_aa += einsum("ia,ij->aj", t1.aa, x14) * -1
    del x14
    x15 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x15 += einsum("ia,bajk->jkib", t1.bb, l2.abab)
    rdm1_f_vo_bb = np.zeros((nvir[1], nocc[1]), dtype=types[float])
    rdm1_f_vo_bb += einsum("ijab,ijka->bk", t2.abab, x15) * -1
    del x15
    x16 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x16 += einsum("ia,waj->wji", t1.bb, lu11.bb)
    rdm1_f_vo_bb -= einsum("wia,wij->aj", u11.bb, x16)
    del x16
    x17 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x17 += einsum("ia,abjk->kjib", t1.bb, l2.bbbb)
    x18 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x18 += einsum("ijka->ijka", x17)
    x18 += einsum("ijka->jika", x17) * -1
    del x17
    rdm1_f_vo_bb += einsum("ijab,ijka->bk", t2.bbbb, x18) * -1
    del x18
    x19 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x19 += einsum("ijab->jiab", t2.bbbb)
    x19 -= einsum("ijab->jiba", t2.bbbb)
    rdm1_f_vo_bb -= einsum("ai,ijab->bj", l1.bb, x19)
    del x19
    x20 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x20 += einsum("ijab->jiab", t2.bbbb) * -1
    x20 += einsum("ijab->jiba", t2.bbbb)
    x21 += einsum("abij,ikba->jk", l2.bbbb, x20) * -1
    del x20
    rdm1_f_vo_bb += einsum("ia,ij->aj", t1.bb, x21) * -1
    del x21
    x22 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x22 += einsum("abij->jiab", l2.aaaa)
    x22 += einsum("abij->jiba", l2.aaaa) * -1
    rdm1_f_vv_aa = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    rdm1_f_vv_aa += einsum("ijab,ijcb->ac", t2.aaaa, x22) * -1
    del x22
    x23 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x23 += einsum("abij->jiab", l2.bbbb) * -1
    x23 += einsum("abij->jiba", l2.bbbb)
    rdm1_f_vv_bb = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    rdm1_f_vv_bb += einsum("ijab,ijca->bc", t2.bbbb, x23) * -1
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
    rdm1_f_vv_aa += einsum("wai,wib->ba", lu11.aa, u11.aa)
    rdm1_f_vv_aa += einsum("ai,ib->ba", l1.aa, t1.aa)
    rdm1_f_vv_aa += einsum("abij,ijcb->ca", l2.abab, t2.abab)
    rdm1_f_vv_bb += einsum("wai,wib->ba", lu11.bb, u11.bb)
    rdm1_f_vv_bb += einsum("abij,ijac->cb", l2.abab, t2.abab)
    rdm1_f_vv_bb += einsum("ai,ib->ba", l1.bb, t1.bb)

    rdm1_f_aa = np.block([[rdm1_f_oo_aa, rdm1_f_ov_aa], [rdm1_f_vo_aa, rdm1_f_vv_aa]])
    rdm1_f_bb = np.block([[rdm1_f_oo_bb, rdm1_f_ov_bb], [rdm1_f_vo_bb, rdm1_f_vv_bb]])

    rdm1_f.aa = rdm1_f_aa
    rdm1_f.bb = rdm1_f_bb

    return rdm1_f

def make_rdm2_f(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, **kwargs):
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
    x0 += einsum("ai,ja->ij", l1.aa, t1.aa)
    x12 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x12 += einsum("ij->ji", x0)
    x33 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x33 += einsum("ij->ij", x0)
    x47 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x47 += einsum("ia,ij->ja", t1.aa, x0)
    x48 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x48 -= einsum("ia->ia", x47)
    del x47
    x85 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x85 += einsum("ij->ij", x0)
    x153 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x153 += einsum("ij,kiab->jkab", x0, t2.aaaa)
    x171 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x171 += einsum("ijab->ijab", x153)
    del x153
    x258 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x258 += einsum("ij->ij", x0)
    rdm2_f_oooo_aaaa = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_oooo_aaaa -= einsum("ij,kl->jikl", delta_oo.aa, x0)
    rdm2_f_oooo_aaaa += einsum("ij,kl->kijl", delta_oo.aa, x0)
    rdm2_f_oooo_aaaa += einsum("ij,kl->jlki", delta_oo.aa, x0)
    rdm2_f_oooo_aaaa -= einsum("ij,kl->klji", delta_oo.aa, x0)
    del x0
    x1 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x1 += einsum("abij,kjab->ik", l2.abab, t2.abab)
    x4 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum("ij->ij", x1)
    x12 += einsum("ij->ji", x1)
    x85 += einsum("ij->ij", x1)
    x258 += einsum("ij->ij", x1)
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x2 += einsum("ijab->jiab", t2.aaaa)
    x2 += einsum("ijab->jiba", t2.aaaa) * -1
    x3 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x3 += einsum("abij,ikab->jk", l2.aaaa, x2)
    x4 += einsum("ij->ij", x3) * -1
    x43 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x43 += einsum("ia,ij->ja", t1.aa, x4)
    x44 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x44 += einsum("ia->ia", x43)
    del x43
    x45 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x45 += einsum("ia,jk->jika", t1.aa, x4)
    x96 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x96 += einsum("ia,jk->jika", t1.aa, x4)
    x174 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x174 += einsum("ij,ikab->kjab", x4, t2.aaaa)
    x175 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x175 += einsum("ijab->ijba", x174)
    del x174
    rdm2_f_oooo_aaaa += einsum("ij,kl->jikl", delta_oo.aa, x4) * -1
    rdm2_f_oooo_aaaa += einsum("ij,kl->jlki", delta_oo.aa, x4)
    rdm2_f_oooo_aaaa += einsum("ij,kl->kjil", delta_oo.aa, x4)
    rdm2_f_oooo_aaaa += einsum("ij,kl->klij", delta_oo.aa, x4) * -1
    del x4
    x12 += einsum("ij->ji", x3) * -1
    x85 += einsum("ij->ij", x3) * -1
    del x3
    x92 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x92 += einsum("ai,ijba->jb", l1.aa, x2)
    x94 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x94 += einsum("ia->ia", x92) * -1
    x259 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x259 += einsum("ia->ia", x92) * -1
    del x92
    x257 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x257 += einsum("abij,ikab->jk", l2.aaaa, x2)
    x258 += einsum("ij->ij", x257) * -1
    del x257
    x260 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x260 += einsum("ai,ijba->jb", l1.aa, x2) * -0.9999999999999993
    x302 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x302 += einsum("abij,ikca->kjcb", l2.abab, x2)
    x304 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x304 += einsum("ijab->ijab", x302) * -1
    del x302
    x5 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x5 += einsum("wai,wja->ij", lu11.aa, u11.aa)
    x12 += einsum("ij->ji", x5)
    x29 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x29 += einsum("ia,ij->ja", t1.aa, x5)
    x32 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x32 += einsum("ia->ia", x29)
    del x29
    x33 += einsum("ij->ij", x5)
    x34 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x34 += einsum("ia,jk->jika", t1.aa, x33)
    x95 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x95 += einsum("ia,jk->jika", t1.aa, x33)
    x183 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x183 += einsum("ia,ij->ja", t1.aa, x33)
    del x33
    x184 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x184 += einsum("ia->ia", x183)
    x185 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x185 += einsum("ia->ia", x183)
    del x183
    x85 += einsum("ij->ij", x5)
    x93 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x93 += einsum("ia,ij->ja", t1.aa, x85)
    x94 += einsum("ia->ia", x93)
    del x93
    x248 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x248 += einsum("ij,ikab->jkab", x85, t2.abab)
    rdm2_f_vovo_bbaa = np.zeros((nvir[1], nocc[1], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x248) * -1
    rdm2_f_vovo_aabb = np.zeros((nvir[0], nocc[0], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x248) * -1
    del x248
    rdm2_f_oovo_aabb = np.zeros((nocc[0], nocc[0], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_oovo_aabb += einsum("ia,jk->jkai", t1.bb, x85) * -1
    del x85
    x155 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x155 += einsum("ij,kiab->kjab", x5, t2.aaaa)
    x171 -= einsum("ijab->ijab", x155)
    del x155
    x258 += einsum("ij->ij", x5)
    x259 += einsum("ia,ij->ja", t1.aa, x258)
    x260 += einsum("ia,ij->ja", t1.aa, x258) * 0.9999999999999993
    del x258
    rdm2_f_oooo_aaaa -= einsum("ij,kl->ijkl", delta_oo.aa, x5)
    rdm2_f_oooo_aaaa += einsum("ij,kl->ilkj", delta_oo.aa, x5)
    rdm2_f_oooo_aaaa += einsum("ij,kl->kijl", delta_oo.aa, x5)
    rdm2_f_oooo_aaaa -= einsum("ij,kl->klji", delta_oo.aa, x5)
    del x5
    x6 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x6 += einsum("abij,klba->ijlk", l2.aaaa, t2.aaaa)
    x39 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x39 += einsum("ijkl->jilk", x6)
    x176 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x176 += einsum("ijab,ijkl->klab", t2.aaaa, x6)
    x180 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x180 += einsum("ijab->ijab", x176)
    del x176
    x178 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x178 += einsum("ia,jikl->jkla", t1.aa, x6)
    x179 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x179 += einsum("ia,ijkb->jkab", t1.aa, x178)
    del x178
    x180 += einsum("ijab->ijab", x179)
    del x179
    rdm2_f_oooo_aaaa += einsum("ijkl->jkil", x6) * -1
    rdm2_f_oooo_aaaa += einsum("ijkl->jlik", x6)
    del x6
    x7 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x7 += einsum("ia,abjk->kjib", t1.aa, l2.aaaa)
    x8 = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    x8 += einsum("ia,jkla->jkil", t1.aa, x7)
    x39 += einsum("ijkl->ijkl", x8)
    x40 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x40 += einsum("ia,ijkl->jkla", t1.aa, x39)
    del x39
    x45 += einsum("ijka->ikja", x40)
    x96 += einsum("ijka->ikja", x40)
    del x40
    x144 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x144 += einsum("ia,ijkl->jlka", t1.aa, x8)
    x145 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x145 += einsum("ia,ijkb->jkab", t1.aa, x144)
    del x144
    x152 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x152 += einsum("ijab->ijab", x145)
    del x145
    x177 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x177 += einsum("ijab,jikl->klba", t2.aaaa, x8)
    x180 += einsum("ijab->ijab", x177)
    del x177
    rdm2_f_vovo_aaaa = np.zeros((nvir[0], nocc[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_vovo_aaaa += einsum("ijab->ajbi", x180) * -1
    rdm2_f_vovo_aaaa += einsum("ijab->bjai", x180)
    del x180
    rdm2_f_oooo_aaaa += einsum("ijkl->ikjl", x8)
    rdm2_f_oooo_aaaa += einsum("ijkl->iljk", x8) * -1
    del x8
    x36 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x36 += einsum("ijka->ijka", x7)
    x36 += einsum("ijka->jika", x7) * -1
    x42 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x42 += einsum("ijab,jika->kb", t2.aaaa, x36)
    x44 += einsum("ia->ia", x42) * -1
    x94 += einsum("ia->ia", x42) * -1
    x260 += einsum("ia->ia", x42) * -1
    del x42
    x78 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x78 += einsum("ijab,ikla->kljb", t2.abab, x36)
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x78) * -1
    rdm2_f_vooo_bbaa = np.zeros((nvir[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x78) * -1
    del x78
    x259 += einsum("ijab,jika->kb", t2.aaaa, x36) * -1
    x105 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x105 -= einsum("ijka->ijka", x7)
    x105 += einsum("ijka->jika", x7)
    x106 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x106 += einsum("ia,jikb->jkab", t1.aa, x105)
    rdm2_f_oovv_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_oovv_aaaa -= einsum("ijab->ijab", x106)
    rdm2_f_vvoo_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_vvoo_aaaa -= einsum("ijab->abij", x106)
    del x106
    x243 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x243 += einsum("ijab,ikla->kljb", t2.abab, x105)
    del x105
    x245 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x245 -= einsum("ijka->ijka", x243)
    del x243
    x130 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x130 += einsum("ijka->ijka", x7)
    x130 -= einsum("ijka->jika", x7)
    x131 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x131 += einsum("ia,jikb->jkab", t1.aa, x130)
    rdm2_f_ovvo_aaaa = np.zeros((nocc[0], nvir[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_ovvo_aaaa -= einsum("ijab->ibaj", x131)
    rdm2_f_voov_aaaa = np.zeros((nvir[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_voov_aaaa -= einsum("ijab->ajib", x131)
    del x131
    x266 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x266 += einsum("ijab,jikc->kcba", t2.aaaa, x7)
    x271 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x271 += einsum("iabc->iabc", x266)
    del x266
    x267 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x267 += einsum("ia,jikb->jkba", t1.aa, x7)
    x269 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x269 += einsum("ijab->ijab", x267) * -1
    del x267
    rdm2_f_ooov_aaaa = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_ooov_aaaa += einsum("ijka->ikja", x7)
    rdm2_f_ooov_aaaa -= einsum("ijka->jkia", x7)
    rdm2_f_ovoo_aaaa = np.zeros((nocc[0], nvir[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_ovoo_aaaa -= einsum("ijka->iajk", x7)
    rdm2_f_ovoo_aaaa += einsum("ijka->jaik", x7)
    del x7
    x9 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x9 += einsum("abij,klab->ikjl", l2.abab, t2.abab)
    x80 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x80 += einsum("ijkl->ijkl", x9)
    x233 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x233 += einsum("ijkl->ijkl", x9)
    x239 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x239 += einsum("ijkl->ijkl", x9)
    rdm2_f_oooo_aabb = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_oooo_aabb += einsum("ijkl->ijkl", x9)
    rdm2_f_oooo_bbaa = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_oooo_bbaa += einsum("ijkl->klij", x9)
    del x9
    x10 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x10 += einsum("ia,bajk->jkib", t1.bb, l2.abab)
    x11 = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    x11 += einsum("ia,jkla->jikl", t1.aa, x10)
    x80 += einsum("ijkl->ijkl", x11)
    x81 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x81 += einsum("ia,jkil->jkla", t1.bb, x80)
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x81)
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x81)
    del x81
    x91 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x91 += einsum("ia,ijkl->jkla", t1.aa, x80)
    del x80
    rdm2_f_oovo_bbaa = np.zeros((nocc[1], nocc[1], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x91)
    rdm2_f_vooo_aabb = np.zeros((nvir[0], nocc[0], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x91)
    del x91
    x233 += einsum("ijkl->ijkl", x11)
    x234 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x234 += einsum("ijab,ikjl->klab", t2.abab, x233)
    del x233
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x234)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x234)
    del x234
    x239 += einsum("ijkl->ijkl", x11) * 0.9999999999999993
    x240 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x240 += einsum("ia,ijkl->jkla", t1.aa, x239)
    del x239
    x241 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x241 += einsum("ijka->ijka", x240)
    del x240
    rdm2_f_oooo_aabb += einsum("ijkl->ijkl", x11)
    rdm2_f_oooo_bbaa += einsum("ijkl->klij", x11)
    del x11
    x60 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x60 += einsum("ijab,ikla->kljb", t2.abab, x10)
    x71 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x71 += einsum("ijka->ijka", x60) * -1
    x99 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x99 += einsum("ijka->ijka", x60) * -1
    x204 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x204 -= einsum("ijka->ijka", x60)
    del x60
    x66 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x66 += einsum("ijab,ijka->kb", t2.abab, x10)
    x70 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x70 += einsum("ia->ia", x66)
    x84 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x84 += einsum("ia->ia", x66)
    x97 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x97 += einsum("ia->ia", x66)
    x256 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x256 += einsum("ia->ia", x66)
    x261 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x261 += einsum("ia->ia", x66)
    del x66
    x77 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x77 += einsum("ijab,kjla->kilb", t2.abab, x10)
    x245 -= einsum("ijka->ijka", x77)
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x77)
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x77)
    del x77
    x90 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x90 += einsum("ijka,ilba->ljkb", x10, x2)
    del x2
    x241 += einsum("ijka->ijka", x90) * -1
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x90) * -1
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x90) * -1
    del x90
    x126 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x126 += einsum("ia,ijkb->jkba", t1.aa, x10)
    rdm2_f_oovv_bbaa = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_oovv_bbaa -= einsum("ijab->ijba", x126)
    rdm2_f_vvoo_aabb = np.zeros((nvir[0], nvir[0], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_vvoo_aabb -= einsum("ijab->baij", x126)
    del x126
    x133 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x133 += einsum("ia,jikb->jkba", t1.bb, x10)
    x296 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x296 += einsum("ijab->ijab", x133)
    rdm2_f_ovvo_aabb = np.zeros((nocc[0], nvir[0], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_ovvo_aabb -= einsum("ijab->iabj", x133)
    rdm2_f_voov_bbaa = np.zeros((nvir[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_voov_bbaa -= einsum("ijab->bjia", x133)
    del x133
    x292 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x292 += einsum("ijab,ijkc->kcab", t2.abab, x10)
    rdm2_f_vovv_bbaa = np.zeros((nvir[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_vovv_bbaa += einsum("iabc->ciba", x292) * -1
    rdm2_f_vvvo_aabb = np.zeros((nvir[0], nvir[0], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_vvvo_aabb += einsum("iabc->baci", x292) * -1
    del x292
    rdm2_f_ooov_bbaa = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_ooov_bbaa -= einsum("ijka->jkia", x10)
    rdm2_f_ovoo_aabb = np.zeros((nocc[0], nvir[0], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_ovoo_aabb -= einsum("ijka->iajk", x10)
    del x10
    x12 += einsum("ij->ji", delta_oo.aa) * -1
    rdm2_f_oooo_aabb += einsum("ij,kl->lkji", delta_oo.bb, x12) * -1
    rdm2_f_oooo_bbaa += einsum("ij,kl->jilk", delta_oo.bb, x12) * -1
    rdm2_f_vooo_bbaa += einsum("ia,jk->aikj", t1.bb, x12) * -1
    del x12
    x13 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x13 += einsum("ai,ja->ij", l1.bb, t1.bb)
    x18 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x18 += einsum("ij->ij", x13)
    x58 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x58 += einsum("ij->ij", x13)
    x73 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x73 += einsum("ia,ij->ja", t1.bb, x13)
    x74 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x74 -= einsum("ia->ia", x73)
    del x73
    x198 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x198 += einsum("ij,kiab->jkab", x13, t2.bbbb)
    x213 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x213 += einsum("ijab->ijab", x198)
    del x198
    x254 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x254 += einsum("ij->ij", x13)
    rdm2_f_oooo_bbbb = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_oooo_bbbb -= einsum("ij,kl->jikl", delta_oo.bb, x13)
    rdm2_f_oooo_bbbb += einsum("ij,kl->kijl", delta_oo.bb, x13)
    rdm2_f_oooo_bbbb += einsum("ij,kl->jlki", delta_oo.bb, x13)
    rdm2_f_oooo_bbbb -= einsum("ij,kl->klji", delta_oo.bb, x13)
    del x13
    x14 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x14 += einsum("wai,wja->ij", lu11.bb, u11.bb)
    x18 += einsum("ij->ij", x14)
    x54 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x54 += einsum("ia,ij->ja", t1.bb, x14)
    x57 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x57 += einsum("ia->ia", x54)
    del x54
    x58 += einsum("ij->ij", x14)
    x59 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x59 += einsum("ia,jk->jika", t1.bb, x58)
    x98 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x98 += einsum("ia,jk->jika", t1.bb, x58)
    x224 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x224 += einsum("ia,ij->ja", t1.bb, x58)
    del x58
    x225 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x225 += einsum("ia->ia", x224)
    x226 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x226 += einsum("ia->ia", x224)
    del x224
    x200 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x200 += einsum("ij,kiab->kjab", x14, t2.bbbb)
    x213 -= einsum("ijab->ijab", x200)
    del x200
    x254 += einsum("ij->ij", x14)
    rdm2_f_oooo_bbbb -= einsum("ij,kl->ijkl", delta_oo.bb, x14)
    rdm2_f_oooo_bbbb += einsum("ij,kl->ilkj", delta_oo.bb, x14)
    rdm2_f_oooo_bbbb += einsum("ij,kl->kijl", delta_oo.bb, x14)
    rdm2_f_oooo_bbbb -= einsum("ij,kl->klji", delta_oo.bb, x14)
    del x14
    x15 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x15 += einsum("abij,ikab->jk", l2.abab, t2.abab)
    x18 += einsum("ij->ij", x15)
    x22 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x22 += einsum("ij->ij", x15)
    x214 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x214 += einsum("ij,kiab->jkab", x15, t2.bbbb)
    x218 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x218 += einsum("ijab->ijab", x214) * -1
    del x214
    x254 += einsum("ij->ij", x15)
    del x15
    x16 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x16 += einsum("ijab->jiab", t2.bbbb)
    x16 += einsum("ijab->jiba", t2.bbbb) * -1
    x17 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x17 += einsum("abij,ikab->jk", l2.bbbb, x16)
    x18 += einsum("ij->ij", x17) * -1
    x83 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x83 += einsum("ia,ij->ja", t1.bb, x18)
    x84 += einsum("ia->ia", x83)
    x97 += einsum("ia->ia", x83)
    del x83
    x247 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x247 += einsum("ij,kiab->kjab", x18, t2.abab)
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x247) * -1
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x247) * -1
    del x247
    rdm2_f_oooo_aabb += einsum("ij,kl->jikl", delta_oo.aa, x18) * -1
    rdm2_f_oooo_bbaa += einsum("ij,kl->klji", delta_oo.aa, x18) * -1
    rdm2_f_oovo_bbaa += einsum("ia,jk->jkai", t1.aa, x18) * -1
    rdm2_f_vooo_aabb += einsum("ia,jk->aijk", t1.aa, x18) * -1
    del x18
    x22 += einsum("ij->ij", x17) * -1
    x69 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x69 += einsum("ia,ij->ja", t1.bb, x22)
    x70 += einsum("ia->ia", x69)
    del x69
    x71 += einsum("ia,jk->jika", t1.bb, x22)
    x99 += einsum("ia,jk->jika", t1.bb, x22)
    rdm2_f_oooo_bbbb += einsum("ij,kl->jikl", delta_oo.bb, x22) * -1
    rdm2_f_oooo_bbbb += einsum("ij,kl->jlki", delta_oo.bb, x22)
    rdm2_f_oooo_bbbb += einsum("ij,kl->kjil", delta_oo.bb, x22)
    rdm2_f_oooo_bbbb += einsum("ij,kl->klij", delta_oo.bb, x22) * -1
    del x22
    x217 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x217 += einsum("ij,ikab->kjab", x17, t2.bbbb) * -1
    del x17
    x218 += einsum("ijab->ijba", x217)
    del x217
    x82 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x82 += einsum("ai,ijba->jb", l1.bb, x16)
    x84 += einsum("ia->ia", x82) * -1
    x97 += einsum("ia->ia", x82) * -1
    del x82
    x252 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x252 += einsum("ai,ijba->jb", l1.bb, x16) * 0.9999999999999993
    x256 += einsum("ia->ia", x252) * -1
    x261 += einsum("ia->ia", x252) * -1
    del x252
    x253 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x253 += einsum("abij,ikab->jk", l2.bbbb, x16)
    x254 += einsum("ij->ij", x253) * -1
    del x253
    x255 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x255 += einsum("ia,ij->ja", t1.bb, x254) * 0.9999999999999993
    del x254
    x256 += einsum("ia->ia", x255)
    x261 += einsum("ia->ia", x255)
    del x255
    x295 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x295 += einsum("abij,jkcb->ikac", l2.abab, x16)
    x296 += einsum("ijab->ijab", x295) * -1
    del x295
    x19 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x19 += einsum("abij,klab->ijkl", l2.bbbb, t2.bbbb)
    x64 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x64 += einsum("ijkl->jilk", x19)
    x219 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x219 += einsum("ia,jikl->jkla", t1.bb, x19)
    x220 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x220 += einsum("ia,ijkb->kjba", t1.bb, x219)
    del x219
    x223 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x223 += einsum("ijab->ijab", x220)
    del x220
    x221 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x221 += einsum("ijkl->jilk", x19)
    rdm2_f_oooo_bbbb += einsum("ijkl->jkil", x19) * -1
    rdm2_f_oooo_bbbb += einsum("ijkl->jlik", x19)
    del x19
    x20 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x20 += einsum("ia,bajk->jkib", t1.bb, l2.bbbb)
    x21 = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    x21 += einsum("ia,jkla->kjli", t1.bb, x20)
    x64 += einsum("ijkl->ijkl", x21)
    x65 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x65 += einsum("ia,ijkl->jkla", t1.bb, x64)
    del x64
    x71 += einsum("ijka->ikja", x65)
    x99 += einsum("ijka->ikja", x65)
    del x65
    x190 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x190 += einsum("ia,ijkl->jlka", t1.bb, x21)
    x191 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x191 += einsum("ia,ijkb->jkab", t1.bb, x190)
    del x190
    x197 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x197 += einsum("ijab->ijab", x191)
    del x191
    x221 += einsum("ijkl->ijkl", x21)
    x222 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x222 += einsum("ijab,ijkl->klab", t2.bbbb, x221)
    del x221
    x223 += einsum("ijab->jiba", x222)
    del x222
    rdm2_f_vovo_bbbb = np.zeros((nvir[1], nocc[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_vovo_bbbb += einsum("ijab->ajbi", x223) * -1
    rdm2_f_vovo_bbbb += einsum("ijab->bjai", x223)
    del x223
    rdm2_f_oooo_bbbb += einsum("ijkl->ikjl", x21)
    rdm2_f_oooo_bbbb += einsum("ijkl->iljk", x21) * -1
    del x21
    x61 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x61 += einsum("ijka->ijka", x20)
    x61 += einsum("ijka->jika", x20) * -1
    x89 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x89 += einsum("ijab,jklb->ikla", t2.abab, x61)
    x241 += einsum("ijka->ijka", x89) * -1
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x89) * -1
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x89) * -1
    del x89
    x67 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x67 += einsum("ijka->ijka", x20) * -1
    x67 += einsum("ijka->jika", x20)
    x68 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x68 += einsum("ijab,jikb->ka", t2.bbbb, x67)
    del x67
    x70 += einsum("ia->ia", x68) * -1
    x71 += einsum("ij,ka->jika", delta_oo.bb, x70) * -1
    x218 += einsum("ia,jb->ijab", t1.bb, x70)
    rdm2_f_vooo_bbbb = np.zeros((nvir[1], nocc[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_vooo_bbbb += einsum("ij,ka->ajik", delta_oo.bb, x70)
    rdm2_f_vooo_bbbb += einsum("ij,ka->akij", delta_oo.bb, x70) * -1
    del x70
    x84 += einsum("ia->ia", x68) * -1
    x97 += einsum("ia->ia", x68) * -1
    x256 += einsum("ia->ia", x68) * -1
    x261 += einsum("ia->ia", x68) * -1
    del x68
    x117 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x117 -= einsum("ijka->ijka", x20)
    x117 += einsum("ijka->jika", x20)
    x118 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x118 += einsum("ia,jikb->jkab", t1.bb, x117)
    del x117
    rdm2_f_oovv_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_oovv_bbbb -= einsum("ijab->ijab", x118)
    rdm2_f_vvoo_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nocc[1]), dtype=types[float])
    rdm2_f_vvoo_bbbb -= einsum("ijab->abij", x118)
    del x118
    x142 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x142 += einsum("ijka->ijka", x20)
    x142 -= einsum("ijka->jika", x20)
    x143 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x143 += einsum("ia,jikb->jkab", t1.bb, x142)
    rdm2_f_ovvo_bbbb = np.zeros((nocc[1], nvir[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_ovvo_bbbb -= einsum("ijab->ibaj", x143)
    rdm2_f_voov_bbbb = np.zeros((nvir[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_voov_bbbb -= einsum("ijab->ajib", x143)
    del x143
    x282 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x282 += einsum("ijab,jikc->kcba", t2.bbbb, x20)
    x289 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x289 += einsum("iabc->iabc", x282)
    del x282
    x283 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x283 += einsum("ia,jikb->jkba", t1.bb, x20)
    x286 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x286 += einsum("ijab->ijab", x283) * -1
    del x283
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
    x45 += einsum("ijka->ijka", x35) * -1
    x96 += einsum("ijka->ijka", x35) * -1
    x162 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x162 -= einsum("ijka->ijka", x35)
    del x35
    x41 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x41 += einsum("ijab,ikjb->ka", t2.abab, x23)
    x44 += einsum("ia->ia", x41)
    x45 += einsum("ij,ka->jika", delta_oo.aa, x44) * -1
    x175 += einsum("ia,jb->ijab", t1.aa, x44)
    rdm2_f_vooo_aaaa = np.zeros((nvir[0], nocc[0], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_vooo_aaaa += einsum("ij,ka->ajik", delta_oo.aa, x44)
    rdm2_f_vooo_aaaa += einsum("ij,ka->akij", delta_oo.aa, x44) * -1
    del x44
    x94 += einsum("ia->ia", x41)
    x259 += einsum("ia->ia", x41)
    x260 += einsum("ia->ia", x41)
    del x41
    x79 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x79 += einsum("ijab,klib->klja", x16, x23)
    rdm2_f_oovo_aabb += einsum("ijka->ijak", x79) * -1
    rdm2_f_vooo_bbaa += einsum("ijka->akij", x79) * -1
    del x79
    x87 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x87 += einsum("ijab,iklb->klja", t2.abab, x23)
    x241 += einsum("ijka->ijka", x87)
    rdm2_f_oovo_bbaa += einsum("ijka->jkai", x87)
    rdm2_f_vooo_aabb += einsum("ijka->aijk", x87)
    del x87
    x128 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x128 += einsum("ia,jkib->jkba", t1.bb, x23)
    rdm2_f_oovv_aabb = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_oovv_aabb -= einsum("ijab->ijba", x128)
    rdm2_f_vvoo_bbaa = np.zeros((nvir[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_vvoo_bbaa -= einsum("ijab->baij", x128)
    del x128
    x138 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x138 += einsum("ia,ijkb->jkab", t1.aa, x23)
    x304 += einsum("ijab->ijab", x138)
    rdm2_f_ovvo_bbaa = np.zeros((nocc[1], nvir[1], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_ovvo_bbaa -= einsum("ijab->jbai", x138)
    rdm2_f_voov_aabb = np.zeros((nvir[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_voov_aabb -= einsum("ijab->aijb", x138)
    del x138
    x298 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x298 += einsum("ijab,ikjc->kacb", t2.abab, x23)
    rdm2_f_vovv_aabb = np.zeros((nvir[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_vovv_aabb += einsum("iabc->aicb", x298) * -1
    rdm2_f_vvvo_bbaa = np.zeros((nvir[1], nvir[1], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_vvvo_bbaa += einsum("iabc->cbai", x298) * -1
    del x298
    rdm2_f_ooov_aabb = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_ooov_aabb -= einsum("ijka->ijka", x23)
    rdm2_f_ovoo_bbaa = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=types[float])
    rdm2_f_ovoo_bbaa -= einsum("ijka->kaij", x23)
    x24 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x24 += einsum("ai,jkba->ijkb", l1.aa, t2.aaaa)
    x34 += einsum("ijka->ijka", x24)
    x95 += einsum("ijka->ijka", x24)
    x162 += einsum("ijka->ijka", x24)
    del x24
    x25 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x25 += einsum("ia,waj->wji", t1.aa, lu11.aa)
    x26 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x26 += einsum("wia,wjk->jkia", u11.aa, x25)
    x34 -= einsum("ijka->ijka", x26)
    x95 -= einsum("ijka->ijka", x26)
    del x26
    rdm2_f_vooo_aaaa -= einsum("ijka->ajik", x95)
    rdm2_f_vooo_aaaa += einsum("ijka->akij", x95)
    del x95
    x28 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x28 += einsum("wia,wij->ja", u11.aa, x25)
    x32 += einsum("ia->ia", x28)
    x94 += einsum("ia->ia", x28)
    x184 += einsum("ia->ia", x28)
    x185 += einsum("ia->ia", x28)
    x259 += einsum("ia->ia", x28)
    x260 += einsum("ia->ia", x28) * 0.9999999999999993
    del x28
    x76 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x76 += einsum("wia,wjk->jkia", u11.bb, x25)
    rdm2_f_oovo_aabb -= einsum("ijka->ijak", x76)
    rdm2_f_vooo_bbaa -= einsum("ijka->akij", x76)
    del x76
    x164 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x164 += einsum("ia,wij->wja", t1.aa, x25)
    del x25
    x167 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x167 -= einsum("wia->wia", x164)
    del x164
    x27 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x27 += einsum("ai,jiba->jb", l1.bb, t2.abab)
    x32 -= einsum("ia->ia", x27)
    x94 += einsum("ia->ia", x27) * -1
    x170 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x170 += einsum("ia->ia", x27)
    x259 += einsum("ia->ia", x27) * -1
    x260 += einsum("ia->ia", x27) * -0.9999999999999993
    del x27
    x30 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x30 += einsum("ijab->jiab", t2.aaaa)
    x30 -= einsum("ijab->jiba", t2.aaaa)
    x31 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x31 += einsum("ai,ijba->jb", l1.aa, x30)
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
    x37 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x37 += einsum("ijab->jiab", t2.aaaa) * -1
    x37 += einsum("ijab->jiba", t2.aaaa)
    x38 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x38 += einsum("ijka,jlab->iklb", x36, x37)
    del x36
    x45 += einsum("ijka->ijka", x38)
    rdm2_f_oovo_aaaa += einsum("ijka->ijak", x45)
    rdm2_f_oovo_aaaa += einsum("ijka->ikaj", x45) * -1
    del x45
    x96 += einsum("ijka->ijka", x38)
    del x38
    rdm2_f_vooo_aaaa += einsum("ijka->ajik", x96) * -1
    rdm2_f_vooo_aaaa += einsum("ijka->akij", x96)
    del x96
    x46 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x46 += einsum("w,wia->ia", ls1, u11.aa)
    x48 += einsum("ia->ia", x46)
    x94 += einsum("ia->ia", x46) * -1
    x184 -= einsum("ia->ia", x46)
    x185 -= einsum("ia->ia", x46)
    rdm2_f_vovo_aaaa -= einsum("ia,jb->aibj", t1.aa, x185)
    rdm2_f_vovo_aaaa += einsum("ia,jb->biaj", t1.aa, x185)
    del x185
    x259 += einsum("ia->ia", x46) * -1
    rdm2_f_vovo_bbaa += einsum("ia,jb->aibj", t1.bb, x259) * -1
    del x259
    x260 += einsum("ia->ia", x46) * -0.9999999999999993
    del x46
    x48 += einsum("ia->ia", t1.aa)
    rdm2_f_oovo_aaaa -= einsum("ij,ka->jkai", delta_oo.aa, x48)
    rdm2_f_oovo_aaaa += einsum("ij,ka->jiak", delta_oo.aa, x48)
    rdm2_f_vooo_aaaa -= einsum("ij,ka->aijk", delta_oo.aa, x48)
    rdm2_f_vooo_aaaa += einsum("ij,ka->akji", delta_oo.aa, x48)
    del x48
    x49 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x49 += einsum("ai,jkab->ikjb", l1.bb, t2.bbbb)
    x59 += einsum("ijka->ijka", x49)
    x98 += einsum("ijka->ijka", x49)
    x204 += einsum("ijka->ijka", x49)
    del x49
    x50 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x50 += einsum("ia,waj->wji", t1.bb, lu11.bb)
    x51 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x51 += einsum("wia,wjk->jkia", u11.bb, x50)
    x59 -= einsum("ijka->ijka", x51)
    x98 -= einsum("ijka->ijka", x51)
    del x51
    rdm2_f_vooo_bbbb -= einsum("ijka->ajik", x98)
    rdm2_f_vooo_bbbb += einsum("ijka->akij", x98)
    del x98
    x53 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x53 += einsum("wia,wij->ja", u11.bb, x50)
    x57 += einsum("ia->ia", x53)
    x84 += einsum("ia->ia", x53)
    x97 += einsum("ia->ia", x53)
    x225 += einsum("ia->ia", x53)
    x226 += einsum("ia->ia", x53)
    x256 += einsum("ia->ia", x53) * 0.9999999999999993
    x261 += einsum("ia->ia", x53) * 0.9999999999999993
    del x53
    x86 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x86 += einsum("wia,wjk->ijka", u11.aa, x50)
    rdm2_f_oovo_bbaa -= einsum("ijka->jkai", x86)
    rdm2_f_vooo_aabb -= einsum("ijka->aijk", x86)
    del x86
    x207 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x207 += einsum("ia,wij->wja", t1.bb, x50)
    del x50
    x209 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x209 -= einsum("wia->wia", x207)
    x249 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x249 -= einsum("wia->wia", x207)
    del x207
    x52 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x52 += einsum("ai,ijab->jb", l1.aa, t2.abab)
    x57 -= einsum("ia->ia", x52)
    x84 += einsum("ia->ia", x52) * -1
    x97 += einsum("ia->ia", x52) * -1
    x212 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x212 += einsum("ia->ia", x52)
    x256 += einsum("ia->ia", x52) * -0.9999999999999993
    x261 += einsum("ia->ia", x52) * -0.9999999999999993
    del x52
    x55 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x55 += einsum("ijab->jiab", t2.bbbb)
    x55 -= einsum("ijab->jiba", t2.bbbb)
    x56 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x56 += einsum("ai,ijba->jb", l1.bb, x55)
    x57 -= einsum("ia->ia", x56)
    del x56
    x59 -= einsum("ij,ka->jika", delta_oo.bb, x57)
    rdm2_f_oovo_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_oovo_bbbb += einsum("ijka->ijak", x59)
    rdm2_f_oovo_bbbb -= einsum("ijka->ikaj", x59)
    del x59
    rdm2_f_vooo_bbbb += einsum("ij,ka->ajik", delta_oo.bb, x57)
    rdm2_f_vooo_bbbb -= einsum("ij,ka->akij", delta_oo.bb, x57)
    del x57
    x62 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x62 += einsum("ijab->jiab", t2.bbbb) * -1
    x62 += einsum("ijab->jiba", t2.bbbb)
    x63 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x63 += einsum("ijka,jlab->iklb", x61, x62)
    del x61
    del x62
    x71 += einsum("ijka->ijka", x63)
    rdm2_f_oovo_bbbb += einsum("ijka->ijak", x71)
    rdm2_f_oovo_bbbb += einsum("ijka->ikaj", x71) * -1
    del x71
    x99 += einsum("ijka->ijka", x63)
    del x63
    rdm2_f_vooo_bbbb += einsum("ijka->ajik", x99) * -1
    rdm2_f_vooo_bbbb += einsum("ijka->akij", x99)
    del x99
    x72 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x72 += einsum("w,wia->ia", ls1, u11.bb)
    x74 += einsum("ia->ia", x72)
    x84 += einsum("ia->ia", x72) * -1
    x97 += einsum("ia->ia", x72) * -1
    rdm2_f_vooo_bbaa += einsum("ij,ka->akji", delta_oo.aa, x97) * -1
    del x97
    x225 -= einsum("ia->ia", x72)
    x226 -= einsum("ia->ia", x72)
    rdm2_f_vovo_bbbb -= einsum("ia,jb->aibj", t1.bb, x226)
    rdm2_f_vovo_bbbb += einsum("ia,jb->biaj", t1.bb, x226)
    del x226
    x256 += einsum("ia->ia", x72) * -0.9999999999999993
    x261 += einsum("ia->ia", x72) * -0.9999999999999993
    del x72
    rdm2_f_vovo_aabb += einsum("ia,jb->aibj", t1.aa, x261) * -1
    del x261
    x74 += einsum("ia->ia", t1.bb)
    rdm2_f_oovo_bbbb -= einsum("ij,ka->jkai", delta_oo.bb, x74)
    rdm2_f_oovo_bbbb += einsum("ij,ka->jiak", delta_oo.bb, x74)
    rdm2_f_vooo_bbbb -= einsum("ij,ka->aijk", delta_oo.bb, x74)
    rdm2_f_vooo_bbbb += einsum("ij,ka->akji", delta_oo.bb, x74)
    del x74
    x75 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x75 += einsum("ai,jkab->ijkb", l1.aa, t2.abab)
    x245 += einsum("ijka->ijka", x75)
    rdm2_f_oovo_aabb -= einsum("ijka->ijak", x75)
    rdm2_f_vooo_bbaa -= einsum("ijka->akij", x75)
    del x75
    x84 += einsum("ia->ia", t1.bb) * -1
    rdm2_f_oovo_aabb += einsum("ij,ka->jiak", delta_oo.aa, x84) * -1
    del x84
    x88 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x88 += einsum("ai,jkba->jikb", l1.bb, t2.abab)
    x241 += einsum("ijka->ijka", x88) * -1
    x242 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x242 += einsum("ia,jikb->jkba", t1.bb, x241)
    del x241
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x242)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x242)
    del x242
    rdm2_f_oovo_bbaa -= einsum("ijka->jkai", x88)
    rdm2_f_vooo_aabb -= einsum("ijka->aijk", x88)
    del x88
    x94 += einsum("ia->ia", t1.aa) * -1
    rdm2_f_oovo_bbaa += einsum("ij,ka->jiak", delta_oo.bb, x94) * -1
    rdm2_f_vooo_aabb += einsum("ij,ka->akji", delta_oo.bb, x94) * -1
    del x94
    x100 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x100 += einsum("wai,wjb->ijab", lu11.aa, u11.aa)
    rdm2_f_oovv_aaaa -= einsum("ijab->ijba", x100)
    rdm2_f_ovvo_aaaa += einsum("ijab->iabj", x100)
    rdm2_f_voov_aaaa += einsum("ijab->bjia", x100)
    rdm2_f_vvoo_aaaa -= einsum("ijab->baij", x100)
    del x100
    x101 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x101 += einsum("abij,kjcb->ikac", l2.abab, t2.abab)
    x159 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x159 += einsum("ijab->ijab", x101)
    x231 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x231 += einsum("ijab->ijab", x101)
    x269 += einsum("ijab->ijab", x101)
    rdm2_f_oovv_aaaa -= einsum("ijab->ijba", x101)
    rdm2_f_ovvo_aaaa += einsum("ijab->iabj", x101)
    rdm2_f_voov_aaaa += einsum("ijab->bjia", x101)
    rdm2_f_vvoo_aaaa -= einsum("ijab->baij", x101)
    del x101
    x102 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x102 -= einsum("ijab->jiab", t2.aaaa)
    x102 += einsum("ijab->jiba", t2.aaaa)
    x139 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x139 += einsum("abij,ikca->kjcb", l2.abab, x102)
    x228 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x228 -= einsum("ijab->ijab", x139)
    rdm2_f_ovvo_bbaa -= einsum("ijab->jbai", x139)
    rdm2_f_voov_aabb -= einsum("ijab->aijb", x139)
    del x139
    x161 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x161 += einsum("ijab,kila->jklb", x102, x130)
    del x130
    x162 += einsum("ijka->jkia", x161)
    del x161
    x163 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x163 += einsum("ia,ijkb->jkab", t1.aa, x162)
    del x162
    x171 += einsum("ijab->ijab", x163)
    del x163
    x166 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x166 += einsum("wai,ijba->wjb", lu11.aa, x102)
    x167 -= einsum("wia->wia", x166)
    del x166
    x169 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x169 += einsum("ai,ijba->jb", l1.aa, x102)
    x170 -= einsum("ia->ia", x169)
    del x169
    x171 += einsum("ia,jb->ijab", t1.aa, x170)
    del x170
    x103 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x103 -= einsum("abij->jiab", l2.aaaa)
    x103 += einsum("abij->jiba", l2.aaaa)
    x104 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x104 += einsum("ijab,ikca->jkbc", x102, x103)
    rdm2_f_oovv_aaaa += einsum("ijab->jiab", x104)
    rdm2_f_vvoo_aaaa += einsum("ijab->abji", x104)
    del x104
    x129 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x129 += einsum("ijab,ikbc->kjca", x103, x30)
    del x30
    rdm2_f_ovvo_aaaa += einsum("ijab->jbai", x129)
    rdm2_f_voov_aaaa += einsum("ijab->aijb", x129)
    del x129
    x134 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x134 += einsum("ijab,ikca->kjcb", t2.abab, x103)
    x188 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x188 -= einsum("ijab,ikac->jkbc", t2.abab, x134)
    x189 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x189 += einsum("ijab->ijab", x188)
    del x188
    rdm2_f_ovvo_aabb -= einsum("ijab->iabj", x134)
    rdm2_f_voov_bbaa -= einsum("ijab->bjia", x134)
    del x134
    x158 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x158 += einsum("ijab,ikbc->jkac", t2.aaaa, x103)
    del x103
    x159 -= einsum("ijab->jiba", x158)
    del x158
    x160 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x160 += einsum("ijab,ikac->jkbc", t2.aaaa, x159)
    del x159
    x171 += einsum("ijab->ijab", x160)
    del x160
    x107 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x107 += einsum("ai,ib->ab", l1.aa, t1.aa)
    x112 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x112 += einsum("ab->ab", x107)
    x275 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x275 += einsum("ab->ab", x107)
    del x107
    x108 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x108 += einsum("wai,wib->ab", lu11.aa, u11.aa)
    x112 += einsum("ab->ab", x108)
    x154 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x154 += einsum("ab,ijca->ijcb", x108, t2.aaaa)
    x171 -= einsum("ijab->ijab", x154)
    del x154
    x237 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x237 += einsum("ab->ab", x108)
    x275 += einsum("ab->ab", x108)
    del x108
    x276 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x276 += einsum("ia,bc->ibac", t1.aa, x275)
    del x275
    x109 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x109 += einsum("abij,ijcb->ac", l2.abab, t2.abab)
    x112 += einsum("ab->ab", x109)
    x172 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x172 += einsum("ab->ab", x109)
    x237 += einsum("ab->ab", x109)
    del x109
    x110 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x110 += einsum("abij->jiab", l2.aaaa)
    x110 += einsum("abij->jiba", l2.aaaa) * -1
    x111 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x111 += einsum("ijab,ijac->bc", t2.aaaa, x110)
    x112 += einsum("ab->ba", x111) * -1
    rdm2_f_oovv_aaaa += einsum("ij,ab->jiba", delta_oo.aa, x112)
    rdm2_f_oovv_bbaa += einsum("ij,ab->jiba", delta_oo.bb, x112)
    rdm2_f_ovvo_aaaa += einsum("ij,ab->jabi", delta_oo.aa, x112) * -1
    rdm2_f_voov_aaaa += einsum("ij,ab->bija", delta_oo.aa, x112) * -1
    rdm2_f_vvoo_aaaa += einsum("ij,ab->baji", delta_oo.aa, x112)
    rdm2_f_vvoo_aabb += einsum("ij,ab->baji", delta_oo.bb, x112)
    rdm2_f_vovv_bbaa += einsum("ia,bc->aicb", t1.bb, x112)
    rdm2_f_vvvo_aabb += einsum("ia,bc->cbai", t1.bb, x112)
    del x112
    x172 += einsum("ab->ba", x111) * -1
    x173 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x173 += einsum("ab,ijac->ijcb", x172, t2.aaaa)
    x175 += einsum("ijab->jiab", x173)
    del x173
    rdm2_f_vovo_aaaa += einsum("ijab->aibj", x175) * -1
    rdm2_f_vovo_aaaa += einsum("ijab->biaj", x175)
    rdm2_f_vovo_aaaa += einsum("ijab->ajbi", x175)
    rdm2_f_vovo_aaaa += einsum("ijab->bjai", x175) * -1
    del x175
    x271 += einsum("ia,bc->ibac", t1.aa, x172)
    del x172
    x237 += einsum("ab->ba", x111) * -1
    del x111
    x238 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x238 += einsum("ab,ijac->ijbc", x237, t2.abab)
    del x237
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x238) * -1
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x238) * -1
    del x238
    x268 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x268 += einsum("ijab,ikca->kjcb", x110, x37)
    del x37
    x269 += einsum("ijab->jiba", x268)
    del x268
    x270 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x270 += einsum("ia,ijbc->jabc", t1.aa, x269)
    del x269
    x271 += einsum("iabc->ibac", x270) * -1
    del x270
    rdm2_f_vovv_aaaa = np.zeros((nvir[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_vovv_aaaa += einsum("iabc->bica", x271)
    rdm2_f_vovv_aaaa += einsum("iabc->ciba", x271) * -1
    rdm2_f_vvvo_aaaa = np.zeros((nvir[0], nvir[0], nvir[0], nocc[0]), dtype=types[float])
    rdm2_f_vvvo_aaaa += einsum("iabc->baci", x271) * -1
    rdm2_f_vvvo_aaaa += einsum("iabc->cabi", x271)
    del x271
    x294 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x294 += einsum("ijab,ikca->kjcb", t2.abab, x110)
    del x110
    x296 += einsum("ijab->ijab", x294) * -1
    del x294
    x297 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x297 += einsum("ia,ijbc->jabc", t1.aa, x296)
    del x296
    rdm2_f_vovv_bbaa += einsum("iabc->ciab", x297) * -1
    rdm2_f_vvvo_aabb += einsum("iabc->abci", x297) * -1
    del x297
    x113 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x113 += einsum("wai,wjb->ijab", lu11.bb, u11.bb)
    rdm2_f_oovv_bbbb -= einsum("ijab->ijba", x113)
    rdm2_f_ovvo_bbbb += einsum("ijab->iabj", x113)
    rdm2_f_voov_bbbb += einsum("ijab->bjia", x113)
    rdm2_f_vvoo_bbbb -= einsum("ijab->baij", x113)
    del x113
    x114 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x114 += einsum("abij,ikac->jkbc", l2.abab, t2.abab)
    x202 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x202 += einsum("ijab,ikac->jkbc", x114, x55)
    x213 -= einsum("ijab->ijab", x202)
    del x202
    x286 += einsum("ijab->ijab", x114)
    rdm2_f_oovv_bbbb -= einsum("ijab->ijba", x114)
    rdm2_f_ovvo_bbbb += einsum("ijab->iabj", x114)
    rdm2_f_voov_bbbb += einsum("ijab->bjia", x114)
    rdm2_f_vvoo_bbbb -= einsum("ijab->baij", x114)
    del x114
    x115 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x115 -= einsum("abij->jiab", l2.bbbb)
    x115 += einsum("abij->jiba", l2.bbbb)
    x116 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x116 += einsum("ijab,ikcb->kjca", x115, x55)
    rdm2_f_oovv_bbbb += einsum("ijab->jiab", x116)
    rdm2_f_vvoo_bbbb += einsum("ijab->abji", x116)
    del x116
    x140 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x140 += einsum("ijab,jkcb->ikac", t2.abab, x115)
    x151 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x151 -= einsum("ijab,kjcb->ikac", t2.abab, x140)
    x152 += einsum("ijab->jiba", x151)
    del x151
    x228 -= einsum("ijab->ijab", x140)
    rdm2_f_ovvo_bbaa -= einsum("ijab->jbai", x140)
    rdm2_f_voov_aabb -= einsum("ijab->aijb", x140)
    del x140
    x141 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x141 += einsum("ijab,ikbc->kjca", x115, x55)
    del x115
    del x55
    rdm2_f_ovvo_bbbb += einsum("ijab->jbai", x141)
    rdm2_f_voov_bbbb += einsum("ijab->aijb", x141)
    del x141
    x119 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x119 += einsum("ai,ib->ab", l1.bb, t1.bb)
    x124 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x124 += einsum("ab->ab", x119)
    x280 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x280 += einsum("ab->ab", x119)
    del x119
    x120 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x120 += einsum("wai,wib->ab", lu11.bb, u11.bb)
    x124 += einsum("ab->ab", x120)
    x199 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x199 += einsum("ab,ijac->jicb", x120, t2.bbbb)
    x213 -= einsum("ijab->ijab", x199)
    del x199
    x235 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x235 += einsum("ab->ab", x120)
    x280 += einsum("ab->ab", x120)
    del x120
    x281 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x281 += einsum("ia,bc->ibac", t1.bb, x280)
    del x280
    x121 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x121 += einsum("abij,ijac->bc", l2.abab, t2.abab)
    x124 += einsum("ab->ab", x121)
    x215 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x215 += einsum("ab,ijac->jibc", x121, t2.bbbb)
    x218 += einsum("ijab->ijab", x215) * -1
    del x215
    x235 += einsum("ab->ab", x121)
    x288 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x288 += einsum("ab->ab", x121)
    del x121
    x122 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x122 += einsum("abij->jiab", l2.bbbb) * -1
    x122 += einsum("abij->jiba", l2.bbbb)
    x123 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x123 += einsum("ijab,ijbc->ac", t2.bbbb, x122)
    del x122
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
    x216 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x216 += einsum("ab,ijbc->ijca", x123, t2.bbbb) * -1
    x218 += einsum("ijab->jiab", x216)
    del x216
    rdm2_f_vovo_bbbb += einsum("ijab->aibj", x218) * -1
    rdm2_f_vovo_bbbb += einsum("ijab->biaj", x218)
    rdm2_f_vovo_bbbb += einsum("ijab->ajbi", x218)
    rdm2_f_vovo_bbbb += einsum("ijab->bjai", x218) * -1
    del x218
    x235 += einsum("ab->ba", x123) * -1
    x236 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x236 += einsum("ab,ijca->ijcb", x235, t2.abab)
    del x235
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x236) * -1
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x236) * -1
    del x236
    x288 += einsum("ab->ba", x123) * -1
    del x123
    x289 += einsum("ia,bc->ibac", t1.bb, x288)
    del x288
    x125 = np.zeros((nocc[1], nocc[1], nvir[0], nvir[0]), dtype=types[float])
    x125 += einsum("abij,ikcb->jkac", l2.abab, t2.abab)
    x293 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x293 += einsum("ia,ijbc->jbca", t1.bb, x125)
    rdm2_f_vovv_bbaa += einsum("iabc->ciba", x293) * -1
    rdm2_f_vvvo_aabb += einsum("iabc->baci", x293) * -1
    del x293
    rdm2_f_oovv_bbaa -= einsum("ijab->ijba", x125)
    rdm2_f_vvoo_aabb -= einsum("ijab->baij", x125)
    del x125
    x127 = np.zeros((nocc[0], nocc[0], nvir[1], nvir[1]), dtype=types[float])
    x127 += einsum("abij,kjac->ikbc", l2.abab, t2.abab)
    x227 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x227 += einsum("ijab,ikbc->kjac", t2.abab, x127)
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x227)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x227)
    del x227
    x299 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x299 += einsum("ia,ijbc->jabc", t1.aa, x127)
    rdm2_f_vovv_aabb += einsum("iabc->aicb", x299) * -1
    rdm2_f_vvvo_bbaa += einsum("iabc->cbai", x299) * -1
    del x299
    rdm2_f_oovv_aabb -= einsum("ijab->ijba", x127)
    rdm2_f_vvoo_bbaa -= einsum("ijab->baij", x127)
    del x127
    x132 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x132 += einsum("wai,wjb->ijab", lu11.aa, u11.bb)
    rdm2_f_ovvo_aabb += einsum("ijab->iabj", x132)
    rdm2_f_voov_bbaa += einsum("ijab->bjia", x132)
    del x132
    x135 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x135 -= einsum("ijab->jiab", t2.bbbb)
    x135 += einsum("ijab->jiba", t2.bbbb)
    x136 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x136 += einsum("abij,jkcb->ikac", l2.abab, x135)
    rdm2_f_ovvo_aabb -= einsum("ijab->iabj", x136)
    rdm2_f_voov_bbaa -= einsum("ijab->bjia", x136)
    del x136
    x203 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x203 += einsum("ijab,kila->jklb", x135, x142)
    del x142
    x204 += einsum("ijka->jkia", x203)
    del x203
    x205 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x205 += einsum("ia,ijkb->jkab", t1.bb, x204)
    del x204
    x213 += einsum("ijab->ijab", x205)
    del x205
    x208 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x208 += einsum("wai,ijba->wjb", lu11.bb, x135)
    x209 -= einsum("wia->wia", x208)
    x249 -= einsum("wia->wia", x208)
    del x208
    x211 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x211 += einsum("ai,ijba->jb", l1.bb, x135)
    x212 -= einsum("ia->ia", x211)
    del x211
    x213 += einsum("ia,jb->ijab", t1.bb, x212)
    del x212
    x229 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x229 += einsum("ijab,kicb->kjca", x135, x228)
    del x228
    rdm2_f_vovo_bbaa -= einsum("ijab->bjai", x229)
    rdm2_f_vovo_aabb -= einsum("ijab->aibj", x229)
    del x229
    x244 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x244 += einsum("ijab,klib->klja", x135, x23)
    del x135
    del x23
    x245 -= einsum("ijka->ijka", x244)
    del x244
    x246 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x246 += einsum("ia,ijkb->jkab", t1.aa, x245)
    del x245
    rdm2_f_vovo_bbaa -= einsum("ijab->bjai", x246)
    rdm2_f_vovo_aabb -= einsum("ijab->aibj", x246)
    del x246
    x137 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x137 += einsum("wai,wjb->jiba", lu11.bb, u11.aa)
    rdm2_f_ovvo_bbaa += einsum("ijab->jbai", x137)
    rdm2_f_voov_aabb += einsum("ijab->aijb", x137)
    del x137
    x146 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x146 += einsum("abij->jiab", l2.aaaa)
    x146 -= einsum("abij->jiba", l2.aaaa)
    x147 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x147 += einsum("ijab,ikac->jkbc", t2.aaaa, x146)
    x148 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x148 -= einsum("ijab,kica->jkbc", t2.aaaa, x147)
    del x147
    x152 += einsum("ijab->jiba", x148)
    del x148
    x149 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x149 += einsum("ijab,ikbc->jkac", t2.aaaa, x146)
    x150 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x150 -= einsum("ijab,kicb->jkac", t2.aaaa, x149)
    del x149
    x152 += einsum("ijab->ijab", x150)
    del x150
    x230 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x230 += einsum("ijab,ikbc->jkac", x102, x146)
    del x146
    del x102
    x231 += einsum("ijab->jiba", x230)
    del x230
    x232 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x232 += einsum("ijab,ikac->kjcb", t2.abab, x231)
    del x231
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x232)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x232)
    del x232
    x152 += einsum("ijab->jiba", t2.aaaa)
    rdm2_f_vovo_aaaa -= einsum("ijab->biaj", x152)
    rdm2_f_vovo_aaaa += einsum("ijab->aibj", x152)
    del x152
    x156 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x156 += einsum("abij,kiac->kjcb", l2.abab, t2.aaaa)
    x157 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x157 += einsum("ijab,kjcb->kica", t2.abab, x156)
    del x156
    x171 -= einsum("ijab->ijab", x157)
    del x157
    x165 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x165 += einsum("wai,jiba->wjb", lu11.bb, t2.abab)
    x167 += einsum("wia->wia", x165)
    del x165
    x168 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x168 += einsum("wia,wjb->ijab", u11.aa, x167)
    x171 += einsum("ijab->jiba", x168)
    del x168
    rdm2_f_vovo_aaaa += einsum("ijab->aibj", x171)
    rdm2_f_vovo_aaaa -= einsum("ijab->biaj", x171)
    rdm2_f_vovo_aaaa -= einsum("ijab->ajbi", x171)
    rdm2_f_vovo_aaaa += einsum("ijab->bjai", x171)
    del x171
    x251 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x251 += einsum("wia,wjb->jiba", u11.bb, x167)
    del x167
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x251)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x251)
    del x251
    x181 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x181 += einsum("wx,xia->wia", ls2, u11.aa)
    x182 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x182 += einsum("wia,wjb->jiba", u11.aa, x181)
    del x181
    rdm2_f_vovo_aaaa -= einsum("ijab->biaj", x182)
    rdm2_f_vovo_aaaa += einsum("ijab->bjai", x182)
    del x182
    x184 -= einsum("ia->ia", t1.aa)
    rdm2_f_vovo_aaaa -= einsum("ia,jb->bjai", t1.aa, x184)
    rdm2_f_vovo_aaaa += einsum("ia,jb->ajbi", t1.aa, x184)
    del x184
    x186 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x186 += einsum("wx,xia->wia", ls2, u11.bb)
    x187 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x187 += einsum("wia,wjb->jiba", u11.bb, x186)
    x189 += einsum("ijab->jiba", x187)
    del x187
    rdm2_f_vovo_bbbb -= einsum("ijab->ajbi", x189)
    rdm2_f_vovo_bbbb += einsum("ijab->aibj", x189)
    del x189
    x249 += einsum("wia->wia", x186)
    del x186
    x192 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x192 += einsum("abij->jiab", l2.bbbb)
    x192 -= einsum("abij->jiba", l2.bbbb)
    x193 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x193 += einsum("ijab,ikac->jkbc", t2.bbbb, x192)
    x194 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x194 -= einsum("ijab,kica->jkbc", t2.bbbb, x193)
    del x193
    x197 += einsum("ijab->jiba", x194)
    del x194
    x195 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x195 += einsum("ijab,ikbc->jkac", t2.bbbb, x192)
    del x192
    x196 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x196 -= einsum("ijab,kicb->jkac", t2.bbbb, x195)
    x197 += einsum("ijab->ijab", x196)
    del x196
    x201 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x201 -= einsum("ijab,kica->jkbc", t2.bbbb, x195)
    del x195
    x213 -= einsum("ijab->ijab", x201)
    del x201
    x197 += einsum("ijab->jiba", t2.bbbb)
    rdm2_f_vovo_bbbb -= einsum("ijab->biaj", x197)
    rdm2_f_vovo_bbbb += einsum("ijab->aibj", x197)
    del x197
    x206 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x206 += einsum("wai,ijab->wjb", lu11.aa, t2.abab)
    x209 += einsum("wia->wia", x206)
    x210 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x210 += einsum("wia,wjb->ijab", u11.bb, x209)
    del x209
    x213 += einsum("ijab->jiba", x210)
    del x210
    rdm2_f_vovo_bbbb += einsum("ijab->aibj", x213)
    rdm2_f_vovo_bbbb -= einsum("ijab->biaj", x213)
    rdm2_f_vovo_bbbb -= einsum("ijab->ajbi", x213)
    rdm2_f_vovo_bbbb += einsum("ijab->bjai", x213)
    del x213
    x249 += einsum("wia->wia", x206)
    del x206
    x250 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x250 += einsum("wia,wjb->ijab", u11.aa, x249)
    del x249
    rdm2_f_vovo_bbaa += einsum("ijab->bjai", x250)
    rdm2_f_vovo_aabb += einsum("ijab->aibj", x250)
    del x250
    x225 -= einsum("ia->ia", t1.bb)
    rdm2_f_vovo_bbbb -= einsum("ia,jb->bjai", t1.bb, x225)
    rdm2_f_vovo_bbbb += einsum("ia,jb->ajbi", t1.bb, x225)
    del x225
    x256 += einsum("ia->ia", t1.bb) * -0.9999999999999993
    rdm2_f_vovo_bbaa += einsum("ia,jb->bjai", t1.aa, x256) * -1
    del x256
    x260 += einsum("ia->ia", t1.aa) * -0.9999999999999993
    rdm2_f_vovo_aabb += einsum("ia,jb->bjai", t1.bb, x260) * -1
    del x260
    x262 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x262 += einsum("ia,bcji->jbca", t1.aa, l2.aaaa)
    x307 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x307 += einsum("ia,ibcd->cbda", t1.aa, x262)
    x308 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x308 += einsum("abcd->badc", x307)
    del x307
    rdm2_f_ovvv_aaaa = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_ovvv_aaaa += einsum("iabc->iacb", x262)
    rdm2_f_ovvv_aaaa -= einsum("iabc->ibca", x262)
    rdm2_f_vvov_aaaa = np.zeros((nvir[0], nvir[0], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_vvov_aaaa -= einsum("iabc->caib", x262)
    rdm2_f_vvov_aaaa += einsum("iabc->cbia", x262)
    del x262
    x263 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x263 += einsum("ia,bcji->jbca", t1.bb, l2.bbbb)
    x312 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x312 += einsum("ia,ibcd->cbda", t1.bb, x263)
    x313 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x313 += einsum("abcd->badc", x312)
    del x312
    rdm2_f_ovvv_bbbb = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_ovvv_bbbb += einsum("iabc->iacb", x263)
    rdm2_f_ovvv_bbbb -= einsum("iabc->ibca", x263)
    rdm2_f_vvov_bbbb = np.zeros((nvir[1], nvir[1], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_vvov_bbbb -= einsum("iabc->caib", x263)
    rdm2_f_vvov_bbbb += einsum("iabc->cbia", x263)
    del x263
    x264 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x264 += einsum("ia,bcij->jbac", t1.aa, l2.abab)
    rdm2_f_ovvv_bbaa = np.zeros((nocc[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_ovvv_bbaa += einsum("iabc->icba", x264)
    rdm2_f_vvov_aabb = np.zeros((nvir[0], nvir[0], nocc[1], nvir[1]), dtype=types[float])
    rdm2_f_vvov_aabb += einsum("iabc->baic", x264)
    del x264
    x265 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x265 += einsum("ia,bcji->jbca", t1.bb, l2.abab)
    x310 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x310 += einsum("ia,ibcd->bacd", t1.aa, x265)
    rdm2_f_vvvv_bbaa = np.zeros((nvir[1], nvir[1], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_vvvv_bbaa += einsum("abcd->dcba", x310)
    rdm2_f_vvvv_aabb = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_vvvv_aabb += einsum("abcd->badc", x310)
    del x310
    rdm2_f_ovvv_aabb = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_ovvv_aabb += einsum("iabc->iacb", x265)
    rdm2_f_vvov_bbaa = np.zeros((nvir[1], nvir[1], nocc[0], nvir[0]), dtype=types[float])
    rdm2_f_vvov_bbaa += einsum("iabc->cbia", x265)
    del x265
    x272 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x272 += einsum("ai,jibc->jabc", l1.aa, t2.aaaa)
    x276 += einsum("iabc->iabc", x272)
    del x272
    x273 = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    x273 += einsum("ia,wbi->wba", t1.aa, lu11.aa)
    x274 = np.zeros((nocc[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x274 += einsum("wia,wbc->ibca", u11.aa, x273)
    x276 -= einsum("iabc->iabc", x274)
    del x274
    rdm2_f_vovv_aaaa += einsum("iabc->bica", x276)
    rdm2_f_vovv_aaaa -= einsum("iabc->ciba", x276)
    rdm2_f_vvvo_aaaa -= einsum("iabc->baci", x276)
    rdm2_f_vvvo_aaaa += einsum("iabc->cabi", x276)
    del x276
    x290 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x290 += einsum("wia,wbc->ibca", u11.bb, x273)
    del x273
    rdm2_f_vovv_bbaa += einsum("iabc->ciba", x290)
    rdm2_f_vvvo_aabb += einsum("iabc->baci", x290)
    del x290
    x277 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x277 += einsum("ai,jibc->jabc", l1.bb, t2.bbbb)
    x281 += einsum("iabc->iabc", x277)
    del x277
    x278 = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    x278 += einsum("ia,wbi->wba", t1.bb, lu11.bb)
    x279 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x279 += einsum("wia,wbc->ibca", u11.bb, x278)
    x281 -= einsum("iabc->iabc", x279)
    del x279
    rdm2_f_vovv_bbbb = np.zeros((nvir[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_vovv_bbbb += einsum("iabc->bica", x281)
    rdm2_f_vovv_bbbb -= einsum("iabc->ciba", x281)
    rdm2_f_vvvo_bbbb = np.zeros((nvir[1], nvir[1], nvir[1], nocc[1]), dtype=types[float])
    rdm2_f_vvvo_bbbb -= einsum("iabc->baci", x281)
    rdm2_f_vvvo_bbbb += einsum("iabc->cabi", x281)
    del x281
    x300 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x300 += einsum("wia,wbc->iabc", u11.aa, x278)
    del x278
    rdm2_f_vovv_aabb += einsum("iabc->aicb", x300)
    rdm2_f_vvvo_bbaa += einsum("iabc->cbai", x300)
    del x300
    x284 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x284 += einsum("abij->jiab", l2.bbbb)
    x284 += einsum("abij->jiba", l2.bbbb) * -1
    x285 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x285 += einsum("ijab,ikac->jkbc", x16, x284)
    del x16
    x286 += einsum("ijab->jiba", x285)
    del x285
    x287 = np.zeros((nocc[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x287 += einsum("ia,ijbc->jabc", t1.bb, x286)
    del x286
    x289 += einsum("iabc->ibac", x287) * -1
    del x287
    rdm2_f_vovv_bbbb += einsum("iabc->bica", x289)
    rdm2_f_vovv_bbbb += einsum("iabc->ciba", x289) * -1
    rdm2_f_vvvo_bbbb += einsum("iabc->baci", x289) * -1
    rdm2_f_vvvo_bbbb += einsum("iabc->cabi", x289)
    del x289
    x303 = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=types[float])
    x303 += einsum("ijab,jkcb->ikac", t2.abab, x284)
    del x284
    x304 += einsum("ijab->ijab", x303) * -1
    del x303
    x305 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x305 += einsum("ia,jibc->jbac", t1.bb, x304)
    del x304
    rdm2_f_vovv_aabb += einsum("iabc->aibc", x305) * -1
    rdm2_f_vvvo_bbaa += einsum("iabc->bcai", x305) * -1
    del x305
    x291 = np.zeros((nocc[1], nvir[0], nvir[0], nvir[1]), dtype=types[float])
    x291 += einsum("ai,ijbc->jabc", l1.aa, t2.abab)
    rdm2_f_vovv_bbaa += einsum("iabc->ciba", x291)
    rdm2_f_vvvo_aabb += einsum("iabc->baci", x291)
    del x291
    x301 = np.zeros((nocc[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x301 += einsum("ai,jibc->jbac", l1.bb, t2.abab)
    rdm2_f_vovv_aabb += einsum("iabc->aicb", x301)
    rdm2_f_vvvo_bbaa += einsum("iabc->cbai", x301)
    del x301
    x306 = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    x306 += einsum("abij,ijcd->abcd", l2.aaaa, t2.aaaa)
    x308 += einsum("abcd->badc", x306)
    del x306
    rdm2_f_vvvv_aaaa = np.zeros((nvir[0], nvir[0], nvir[0], nvir[0]), dtype=types[float])
    rdm2_f_vvvv_aaaa += einsum("abcd->dacb", x308) * -1
    rdm2_f_vvvv_aaaa += einsum("abcd->cadb", x308)
    del x308
    x309 = np.zeros((nvir[0], nvir[0], nvir[1], nvir[1]), dtype=types[float])
    x309 += einsum("abij,ijcd->acbd", l2.abab, t2.abab)
    rdm2_f_vvvv_bbaa += einsum("abcd->dcba", x309)
    rdm2_f_vvvv_aabb += einsum("abcd->badc", x309)
    del x309
    x311 = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    x311 += einsum("abij,ijcd->abcd", l2.bbbb, t2.bbbb)
    x313 += einsum("abcd->badc", x311)
    del x311
    rdm2_f_vvvv_bbbb = np.zeros((nvir[1], nvir[1], nvir[1], nvir[1]), dtype=types[float])
    rdm2_f_vvvv_bbbb += einsum("abcd->dacb", x313) * -1
    rdm2_f_vvvv_bbbb += einsum("abcd->cadb", x313)
    del x313
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

def make_sing_b_dm(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, **kwargs):
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
    dm_b_des += einsum("ai,wia->w", l1.bb, u11.bb)
    dm_b_des += einsum("ai,wia->w", l1.aa, u11.aa)
    dm_b_des += einsum("w,xw->x", ls1, s2)

    dm_b = np.array([dm_b_cre, dm_b_des])

    return dm_b

def make_rdm1_b(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, **kwargs):
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
    rdm1_b += einsum("wx,yx->wy", ls2, s2)
    rdm1_b += einsum("w,x->wx", ls1, s1)
    rdm1_b += einsum("wai,xia->wx", lu11.bb, u11.bb)
    rdm1_b += einsum("wai,xia->wx", lu11.aa, u11.aa)

    return rdm1_b

def make_eb_coup_rdm(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, **kwargs):
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
    x53 = np.zeros((nocc[0], nvir[0]), dtype=types[float])
    x53 += einsum("wia,wij->ja", u11.aa, x0)
    rdm_eb_cre_oo_aa = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    rdm_eb_cre_oo_aa -= einsum("wij->wji", x0)
    rdm_eb_cre_ov_aa = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    rdm_eb_cre_ov_aa -= einsum("ia,wij->wja", t1.aa, x0)
    del x0
    x1 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x1 += einsum("ia,waj->wji", t1.bb, lu11.bb)
    x65 = np.zeros((nocc[1], nvir[1]), dtype=types[float])
    x65 += einsum("wia,wij->ja", u11.bb, x1)
    rdm_eb_cre_oo_bb = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    rdm_eb_cre_oo_bb -= einsum("wij->wji", x1)
    rdm_eb_cre_ov_bb = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    rdm_eb_cre_ov_bb -= einsum("ia,wij->wja", t1.bb, x1)
    del x1
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x2 -= einsum("ijab->jiab", t2.aaaa)
    x2 += einsum("ijab->jiba", t2.aaaa)
    rdm_eb_cre_ov_aa -= einsum("wai,ijba->wjb", lu11.aa, x2)
    x3 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x3 += einsum("ijab->jiab", t2.bbbb)
    x3 -= einsum("ijab->jiba", t2.bbbb)
    rdm_eb_cre_ov_bb -= einsum("wai,ijab->wjb", lu11.bb, x3)
    x4 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x4 += einsum("ai,wja->wij", l1.aa, u11.aa)
    x43 = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    x43 += einsum("wij->wij", x4)
    rdm_eb_des_oo_aa = np.zeros((nbos, nocc[0], nocc[0]), dtype=types[float])
    rdm_eb_des_oo_aa -= einsum("wij->wji", x4)
    del x4
    x5 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x5 += einsum("wx,xai->wia", s2, lu11.aa)
    x9 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x9 += einsum("wia->wia", x5)
    x34 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x34 += einsum("wia->wia", x5)
    rdm_eb_des_vo_aa = np.zeros((nbos, nvir[0], nocc[0]), dtype=types[float])
    rdm_eb_des_vo_aa += einsum("wia->wai", x5)
    del x5
    x6 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x6 += einsum("wia,baji->wjb", u11.bb, l2.abab)
    x9 += einsum("wia->wia", x6)
    x34 += einsum("wia->wia", x6)
    rdm_eb_des_vo_aa += einsum("wia->wai", x6)
    del x6
    x7 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x7 += einsum("abij->jiab", l2.aaaa) * -1
    x7 += einsum("abij->jiba", l2.aaaa)
    x8 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x8 += einsum("wia,ijba->wjb", u11.aa, x7)
    x9 += einsum("wia->wia", x8) * -1
    del x8
    rdm_eb_des_oo_aa += einsum("ia,wja->wij", t1.aa, x9) * -1
    rdm_eb_des_vv_aa = np.zeros((nbos, nvir[0], nvir[0]), dtype=types[float])
    rdm_eb_des_vv_aa += einsum("ia,wib->wba", t1.aa, x9)
    del x9
    x40 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x40 += einsum("ijab,ijbc->ac", t2.aaaa, x7)
    del x7
    x41 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x41 += einsum("ab->ba", x40) * -1
    x66 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x66 += einsum("ab->ba", x40) * -1
    del x40
    x10 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x10 += einsum("ai,ja->ij", l1.aa, t1.aa)
    x15 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x15 += einsum("ij->ij", x10)
    x42 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x42 += einsum("ij->ij", x10)
    x52 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x52 += einsum("ij->ij", x10)
    del x10
    x11 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x11 += einsum("wai,wja->ij", lu11.aa, u11.aa)
    x15 += einsum("ij->ij", x11)
    x42 += einsum("ij->ij", x11)
    x52 += einsum("ij->ij", x11)
    del x11
    x12 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x12 += einsum("abij,kjab->ik", l2.abab, t2.abab)
    x15 += einsum("ij->ij", x12)
    x42 += einsum("ij->ij", x12)
    x52 += einsum("ij->ij", x12)
    del x12
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x13 += einsum("ijab->jiab", t2.aaaa) * -1
    x13 += einsum("ijab->jiba", t2.aaaa)
    x14 = np.zeros((nocc[0], nocc[0]), dtype=types[float])
    x14 += einsum("abij,ikba->jk", l2.aaaa, x13)
    x15 += einsum("ij->ij", x14) * -1
    x42 += einsum("ij->ij", x14) * -1
    del x14
    rdm_eb_des_ov_aa = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    rdm_eb_des_ov_aa += einsum("ij,wia->wja", x42, u11.aa) * -1
    del x42
    x52 += einsum("abij,ikba->jk", l2.aaaa, x13) * -1
    del x13
    x53 += einsum("ia,ij->ja", t1.aa, x52)
    del x52
    x15 += einsum("ij->ji", delta_oo.aa) * -1
    rdm_eb_des_oo_aa += einsum("w,ij->wji", s1, x15) * -1
    del x15
    x16 = np.zeros((nbos), dtype=types[float])
    x16 += einsum("w,xw->x", ls1, s2)
    x19 = np.zeros((nbos), dtype=types[float])
    x19 += einsum("w->w", x16)
    del x16
    x17 = np.zeros((nbos), dtype=types[float])
    x17 += einsum("ai,wia->w", l1.aa, u11.aa)
    x19 += einsum("w->w", x17)
    del x17
    x18 = np.zeros((nbos), dtype=types[float])
    x18 += einsum("ai,wia->w", l1.bb, u11.bb)
    x19 += einsum("w->w", x18)
    del x18
    rdm_eb_des_oo_aa += einsum("w,ij->wji", x19, delta_oo.aa)
    rdm_eb_des_oo_bb = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    rdm_eb_des_oo_bb += einsum("w,ij->wji", x19, delta_oo.bb)
    rdm_eb_des_ov_aa += einsum("w,ia->wia", x19, t1.aa)
    rdm_eb_des_ov_bb = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    rdm_eb_des_ov_bb += einsum("w,ia->wia", x19, t1.bb)
    del x19
    x20 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x20 += einsum("ai,wja->wij", l1.bb, u11.bb)
    x60 = np.zeros((nbos, nocc[1], nocc[1]), dtype=types[float])
    x60 += einsum("wij->wij", x20)
    rdm_eb_des_oo_bb -= einsum("wij->wji", x20)
    del x20
    x21 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x21 += einsum("wx,xai->wia", s2, lu11.bb)
    x25 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x25 += einsum("wia->wia", x21)
    x37 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x37 += einsum("wia->wia", x21)
    rdm_eb_des_vo_bb = np.zeros((nbos, nvir[1], nocc[1]), dtype=types[float])
    rdm_eb_des_vo_bb += einsum("wia->wai", x21)
    del x21
    x22 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x22 += einsum("wia,abij->wjb", u11.aa, l2.abab)
    x25 += einsum("wia->wia", x22)
    x37 += einsum("wia->wia", x22)
    rdm_eb_des_vo_bb += einsum("wia->wai", x22)
    del x22
    x23 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x23 += einsum("abij->jiab", l2.bbbb)
    x23 += einsum("abij->jiba", l2.bbbb) * -1
    x24 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x24 += einsum("wia,ijab->wjb", u11.bb, x23)
    del x23
    x25 += einsum("wia->wia", x24) * -1
    del x24
    rdm_eb_des_oo_bb += einsum("ia,wja->wij", t1.bb, x25) * -1
    rdm_eb_des_vv_bb = np.zeros((nbos, nvir[1], nvir[1]), dtype=types[float])
    rdm_eb_des_vv_bb += einsum("ia,wib->wba", t1.bb, x25)
    del x25
    x26 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x26 += einsum("ai,ja->ij", l1.bb, t1.bb)
    x31 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x31 += einsum("ij->ij", x26)
    x59 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x59 += einsum("ij->ij", x26)
    x64 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x64 += einsum("ij->ij", x26)
    del x26
    x27 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x27 += einsum("wai,wja->ij", lu11.bb, u11.bb)
    x31 += einsum("ij->ij", x27)
    x59 += einsum("ij->ij", x27)
    x64 += einsum("ij->ij", x27)
    del x27
    x28 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x28 += einsum("abij,ikab->jk", l2.abab, t2.abab)
    x31 += einsum("ij->ij", x28)
    x59 += einsum("ij->ij", x28)
    x64 += einsum("ij->ij", x28)
    del x28
    x29 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x29 += einsum("ijab->jiab", t2.bbbb) * -1
    x29 += einsum("ijab->jiba", t2.bbbb)
    x30 = np.zeros((nocc[1], nocc[1]), dtype=types[float])
    x30 += einsum("abij,ikba->jk", l2.bbbb, x29)
    x31 += einsum("ij->ij", x30) * -1
    x59 += einsum("ij->ij", x30) * -1
    del x30
    rdm_eb_des_ov_bb += einsum("ij,wia->wja", x59, u11.bb) * -1
    del x59
    x64 += einsum("abij,ikba->jk", l2.bbbb, x29) * -1
    x65 += einsum("ia,ij->ja", t1.bb, x64)
    del x64
    x65 += einsum("ai,ijab->jb", l1.bb, x29) * -1
    del x29
    x31 += einsum("ij->ji", delta_oo.bb) * -1
    rdm_eb_des_oo_bb += einsum("w,ij->wji", s1, x31) * -1
    del x31
    x32 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x32 -= einsum("abij->jiab", l2.aaaa)
    x32 += einsum("abij->jiba", l2.aaaa)
    x33 = np.zeros((nbos, nocc[0], nvir[0]), dtype=types[float])
    x33 += einsum("wia,ijba->wjb", u11.aa, x32)
    del x32
    x34 -= einsum("wia->wia", x33)
    x43 += einsum("ia,wja->wji", t1.aa, x34)
    rdm_eb_des_ov_aa -= einsum("ia,wij->wja", t1.aa, x43)
    del x43
    rdm_eb_des_ov_aa -= einsum("wia,ijba->wjb", x34, x2)
    del x2
    rdm_eb_des_ov_bb += einsum("wia,ijab->wjb", x34, t2.abab)
    del x34
    rdm_eb_des_vo_aa -= einsum("wia->wai", x33)
    del x33
    x35 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x35 += einsum("abij->jiab", l2.bbbb)
    x35 -= einsum("abij->jiba", l2.bbbb)
    x36 = np.zeros((nbos, nocc[1], nvir[1]), dtype=types[float])
    x36 += einsum("wia,ijab->wjb", u11.bb, x35)
    del x35
    x37 -= einsum("wia->wia", x36)
    x60 += einsum("ia,wja->wji", t1.bb, x37)
    rdm_eb_des_ov_bb -= einsum("ia,wij->wja", t1.bb, x60)
    del x60
    rdm_eb_des_ov_aa += einsum("wia,jiba->wjb", x37, t2.abab)
    rdm_eb_des_ov_bb -= einsum("wia,ijab->wjb", x37, x3)
    del x3
    del x37
    rdm_eb_des_vo_bb -= einsum("wia->wai", x36)
    del x36
    x38 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x38 += einsum("wai,wib->ab", lu11.aa, u11.aa)
    x41 += einsum("ab->ab", x38)
    x66 += einsum("ab->ab", x38)
    del x38
    x39 = np.zeros((nvir[0], nvir[0]), dtype=types[float])
    x39 += einsum("abij,ijcb->ac", l2.abab, t2.abab)
    x41 += einsum("ab->ab", x39)
    rdm_eb_des_ov_aa += einsum("ab,wia->wib", x41, u11.aa) * -1
    del x41
    x66 += einsum("ab->ab", x39)
    del x39
    x44 = np.zeros((nbos, nbos), dtype=types[float])
    x44 += einsum("wx,yw->xy", ls2, s2)
    x47 = np.zeros((nbos, nbos), dtype=types[float])
    x47 += einsum("wx->wx", x44)
    del x44
    x45 = np.zeros((nbos, nbos), dtype=types[float])
    x45 += einsum("wai,xia->wx", lu11.aa, u11.aa)
    x47 += einsum("wx->wx", x45)
    del x45
    x46 = np.zeros((nbos, nbos), dtype=types[float])
    x46 += einsum("wai,xia->wx", lu11.bb, u11.bb)
    x47 += einsum("wx->wx", x46)
    del x46
    rdm_eb_des_ov_aa += einsum("wx,wia->xia", x47, u11.aa)
    rdm_eb_des_ov_bb += einsum("wx,wia->xia", x47, u11.bb)
    del x47
    x48 = np.zeros((nocc[0], nocc[0], nocc[1], nvir[1]), dtype=types[float])
    x48 += einsum("ia,abjk->jikb", t1.aa, l2.abab)
    x53 += einsum("ijab,ikjb->ka", t2.abab, x48)
    del x48
    x49 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x49 += einsum("ia,bajk->jkib", t1.aa, l2.aaaa)
    x50 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=types[float])
    x50 += einsum("ijka->ijka", x49)
    x50 += einsum("ijka->jika", x49) * -1
    del x49
    x53 += einsum("ijab,ijkb->ka", t2.aaaa, x50) * -1
    del x50
    x51 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=types[float])
    x51 += einsum("ijab->jiab", t2.aaaa)
    x51 += einsum("ijab->jiba", t2.aaaa) * -1
    x53 += einsum("ai,ijba->jb", l1.aa, x51) * -1
    del x51
    x53 += einsum("ia->ia", t1.aa) * -1
    x53 += einsum("w,wia->ia", ls1, u11.aa) * -1
    x53 += einsum("ai,jiba->jb", l1.bb, t2.abab) * -1
    rdm_eb_des_ov_aa += einsum("w,ia->wia", s1, x53) * -1
    del x53
    x54 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x54 += einsum("wai,wib->ab", lu11.bb, u11.bb)
    x58 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x58 += einsum("ab->ab", x54)
    x67 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x67 += einsum("ab->ab", x54)
    del x54
    x55 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x55 += einsum("abij,ijac->bc", l2.abab, t2.abab)
    x58 += einsum("ab->ab", x55)
    x67 += einsum("ab->ab", x55)
    del x55
    x56 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=types[float])
    x56 += einsum("abij->jiab", l2.bbbb) * -1
    x56 += einsum("abij->jiba", l2.bbbb)
    x57 = np.zeros((nvir[1], nvir[1]), dtype=types[float])
    x57 += einsum("ijab,ijca->bc", t2.bbbb, x56)
    del x56
    x58 += einsum("ab->ba", x57) * -1
    rdm_eb_des_ov_bb += einsum("ab,wia->wib", x58, u11.bb) * -1
    del x58
    x67 += einsum("ab->ba", x57) * -1
    del x57
    x61 = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0]), dtype=types[float])
    x61 += einsum("ia,bajk->jkib", t1.bb, l2.abab)
    x65 += einsum("ijab,ijka->kb", t2.abab, x61)
    del x61
    x62 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x62 += einsum("ia,bajk->jkib", t1.bb, l2.bbbb)
    x63 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=types[float])
    x63 += einsum("ijka->ijka", x62) * -1
    x63 += einsum("ijka->jika", x62)
    del x62
    x65 += einsum("ijab,ijka->kb", t2.bbbb, x63) * -1
    del x63
    x65 += einsum("ia->ia", t1.bb) * -1
    x65 += einsum("w,wia->ia", ls1, u11.bb) * -1
    x65 += einsum("ai,ijab->jb", l1.aa, t2.abab) * -1
    rdm_eb_des_ov_bb += einsum("w,ia->wia", s1, x65) * -1
    del x65
    x66 += einsum("ai,ib->ab", l1.aa, t1.aa)
    rdm_eb_des_vv_aa += einsum("w,ab->wab", s1, x66)
    del x66
    x67 += einsum("ai,ib->ab", l1.bb, t1.bb)
    rdm_eb_des_vv_bb += einsum("w,ab->wab", s1, x67)
    del x67
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

