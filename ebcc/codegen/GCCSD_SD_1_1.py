# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, **kwargs):
    # Energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum("ijab->jiba", t2)
    x0 += einsum("ia,jb->ijab", t1, t1) * 2
    e_cc = 0
    e_cc += einsum("ijab,ijab->", v.oovv, x0) * 0.25
    del x0
    x1 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x1 += einsum("wia->wia", u11)
    x1 += einsum("w,ia->wia", s1, t1)
    e_cc += einsum("wia,wia->", g.bov, x1)
    del x1
    e_cc += einsum("ia,ia->", f.ov, t1)
    e_cc += einsum("w,w->", G, s1)

    return e_cc

def update_amps(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    # T1, T2, S1, S2 and U11 amplitudes
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x0 += einsum("ia,jkba->ijkb", t1, v.oovv)
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum("ijka->ikja", x0) * -1
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x30 += einsum("ijab,kjlb->kila", t2, x0)
    x31 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x31 -= einsum("ijka->ikja", x30)
    del x30
    x65 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x65 += einsum("ia,jkla->jilk", t1, x0)
    del x0
    x66 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x66 += einsum("ijkl->klji", x65) * -1
    x67 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x67 += einsum("ijkl->klji", x65) * -1
    del x65
    x1 += einsum("ijka->kjia", v.ooov)
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum("ijab,kija->kb", t2, x1) * -0.5
    del x1
    x2 = np.zeros((nocc, nvir), dtype=types[float])
    x2 += einsum("w,wia->ia", s1, g.bov)
    x4 = np.zeros((nocc, nvir), dtype=types[float])
    x4 += einsum("ia->ia", x2)
    x11 = np.zeros((nocc, nvir), dtype=types[float])
    x11 += einsum("ia->ia", x2)
    x78 = np.zeros((nocc, nvir), dtype=types[float])
    x78 += einsum("ia->ia", x2)
    del x2
    x3 = np.zeros((nocc, nvir), dtype=types[float])
    x3 += einsum("ia,jiba->jb", t1, v.oovv)
    x4 += einsum("ia->ia", x3)
    x11 += einsum("ia->ia", x3)
    x77 = np.zeros((nvir, nvir), dtype=types[float])
    x77 += einsum("ia,ib->ab", t1, x3)
    del x3
    x4 += einsum("ia->ia", f.ov)
    x38 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x38 += einsum("ia,jkab->jkib", x4, t2)
    x39 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x39 += einsum("ijka->kjia", x38)
    del x38
    x54 = np.zeros((nocc, nocc), dtype=types[float])
    x54 += einsum("ia,ja->ij", t1, x4)
    x55 = np.zeros((nocc, nocc), dtype=types[float])
    x55 += einsum("ij->ij", x54)
    del x54
    t1new += einsum("ia,ijab->jb", x4, t2)
    s1new = np.zeros((nbos), dtype=types[float])
    s1new += einsum("ia,wia->w", x4, u11)
    del x4
    x5 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x5 += einsum("ia,wja->wji", t1, g.bov)
    x6 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x6 += einsum("wij->wij", x5)
    del x5
    x6 += einsum("wij->wij", g.boo)
    x25 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x25 += einsum("ia,wij->wja", t1, x6)
    x26 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x26 -= einsum("wia->wia", x25)
    del x25
    t1new -= einsum("wia,wij->ja", u11, x6)
    del x6
    x7 = np.zeros((nocc, nocc), dtype=types[float])
    x7 += einsum("w,wij->ij", s1, g.boo)
    x13 = np.zeros((nocc, nocc), dtype=types[float])
    x13 += einsum("ij->ij", x7)
    x55 += einsum("ij->ji", x7)
    del x7
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum("wia,wja->ij", g.bov, u11)
    x13 += einsum("ij->ij", x8)
    x57 = np.zeros((nocc, nocc), dtype=types[float])
    x57 += einsum("ij->ij", x8)
    del x8
    x9 = np.zeros((nocc, nocc), dtype=types[float])
    x9 -= einsum("ia,ijka->jk", t1, v.ooov)
    x13 += einsum("ij->ij", x9)
    x57 += einsum("ij->ij", x9)
    del x9
    x58 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x58 += einsum("ij,ikab->kjab", x57, t2)
    del x57
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 -= einsum("ijab->ijba", x58)
    del x58
    x10 = np.zeros((nocc, nocc), dtype=types[float])
    x10 -= einsum("ijab,jkab->ik", t2, v.oovv)
    x13 += einsum("ij->ji", x10) * 0.5
    x62 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x62 += einsum("ij,kjab->kiab", x10, t2)
    del x10
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum("ijab->ijba", x62) * -0.5
    del x62
    x11 += einsum("ia->ia", f.ov)
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum("ia,ja->ij", t1, x11)
    del x11
    x13 += einsum("ij->ji", x12)
    del x12
    x13 += einsum("ij->ij", f.oo)
    t1new += einsum("ia,ij->ja", t1, x13) * -1
    u11new = np.zeros((nbos, nocc, nvir), dtype=types[float])
    u11new += einsum("ij,wia->wja", x13, u11) * -1
    del x13
    x14 = np.zeros((nvir, nvir), dtype=types[float])
    x14 += einsum("w,wab->ab", s1, g.bvv)
    x16 = np.zeros((nvir, nvir), dtype=types[float])
    x16 += einsum("ab->ab", x14)
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x34 += einsum("ab,ijcb->ijac", x14, t2)
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum("ijab->ijab", x34)
    del x34
    x77 += einsum("ab->ab", x14) * -1
    del x14
    x15 = np.zeros((nvir, nvir), dtype=types[float])
    x15 -= einsum("ia,ibac->bc", t1, v.ovvv)
    x16 -= einsum("ab->ab", x15)
    x42 = np.zeros((nvir, nvir), dtype=types[float])
    x42 += einsum("ab->ab", x15)
    x77 += einsum("ab->ab", x15)
    del x15
    x16 += einsum("ab->ab", f.vv)
    t1new += einsum("ia,ba->ib", t1, x16)
    del x16
    x17 = np.zeros((nbos), dtype=types[float])
    x17 += einsum("ia,wia->w", t1, g.bov)
    x18 = np.zeros((nbos), dtype=types[float])
    x18 += einsum("w->w", x17)
    del x17
    x18 += einsum("w->w", G)
    t1new += einsum("w,wia->ia", x18, u11)
    s1new += einsum("w,wx->x", x18, s2)
    del x18
    x19 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x19 += einsum("ia,bcda->ibcd", t1, v.vvvv)
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new -= einsum("ia,jbca->ijcb", t1, x19)
    del x19
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum("ijab,jckb->ikac", t2, v.ovov)
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 -= einsum("ijab->ijab", x20)
    del x20
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum("ia,jbca->ijbc", t1, v.ovvv)
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 -= einsum("ijab,kjcb->kiac", t2, x21)
    x33 += einsum("ijab->ijab", x22)
    del x22
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x37 -= einsum("ia,jkba->jikb", t1, x21)
    del x21
    x39 += einsum("ijka->kjia", x37)
    del x37
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum("ia,ijkb->jkab", t1, x39)
    del x39
    x44 += einsum("ijab->jiab", x40)
    del x40
    x23 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x23 += einsum("ia,wba->wib", t1, g.bvv)
    x26 += einsum("wia->wia", x23)
    del x23
    x24 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x24 += einsum("wia,jiba->wjb", g.bov, t2)
    x26 += einsum("wia->wia", x24)
    del x24
    x26 += einsum("wai->wia", g.bvo)
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 += einsum("wia,wjb->ijab", u11, x26)
    del x26
    x33 += einsum("ijab->jiba", x27)
    del x27
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum("ia,jbka->ijkb", t1, v.ovov)
    x31 += einsum("ijka->ijka", x28)
    del x28
    x29 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x29 -= einsum("ijab,jklb->ikla", t2, v.ooov)
    x31 += einsum("ijka->ijka", x29)
    del x29
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum("ia,jikb->jkab", t1, x31)
    del x31
    x33 += einsum("ijab->ijab", x32)
    del x32
    t2new += einsum("ijab->ijab", x33)
    t2new -= einsum("ijab->ijba", x33)
    t2new -= einsum("ijab->jiab", x33)
    t2new += einsum("ijab->jiba", x33)
    del x33
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum("ijab,jkbc->ikac", t2, v.oovv)
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 += einsum("ijab,kjcb->kica", t2, x35)
    del x35
    x44 -= einsum("ijab->ijab", x36)
    del x36
    x41 = np.zeros((nvir, nvir), dtype=types[float])
    x41 += einsum("wia,wib->ab", g.bov, u11)
    x42 += einsum("ab->ba", x41)
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum("ab,ijbc->ijca", x42, t2)
    del x42
    x44 += einsum("ijab->jiab", x43)
    del x43
    t2new -= einsum("ijab->ijab", x44)
    t2new += einsum("ijab->ijba", x44)
    del x44
    x77 += einsum("ab->ba", x41)
    del x41
    x45 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x45 += einsum("ijab,kcab->ijkc", t2, v.ovvv)
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum("ia,jkib->jkab", t1, x45)
    del x45
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum("ijab->ijab", x46) * 0.5
    del x46
    x47 = np.zeros((nvir, nvir), dtype=types[float])
    x47 -= einsum("ijab,ijbc->ac", t2, v.oovv)
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum("ab,ijcb->ijca", x47, t2)
    x49 += einsum("ijab->ijab", x48) * 0.5
    del x48
    t2new += einsum("ijab->ijab", x49) * -1
    t2new += einsum("ijab->ijba", x49)
    del x49
    x77 += einsum("ab->ab", x47) * 0.5
    del x47
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum("ia,bcja->ijbc", t1, v.vvov)
    x59 += einsum("ijab->ijba", x50)
    del x50
    x51 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x51 += einsum("ia,jkla->ijkl", t1, v.ooov)
    x52 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x52 += einsum("ia,jkil->jkla", t1, x51)
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x53 -= einsum("ia,jikb->jkba", t1, x52)
    del x52
    x59 += einsum("ijab->ijba", x53)
    del x53
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum("ijab,kijl->klab", t2, x51)
    del x51
    x63 += einsum("ijab->ijba", x61) * -0.5
    del x61
    t2new += einsum("ijab->ijab", x63) * -1
    t2new += einsum("ijab->jiab", x63)
    del x63
    x55 += einsum("ij->ji", f.oo)
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum("ij,jkab->kiab", x55, t2)
    del x55
    x59 += einsum("ijab->jiba", x56)
    del x56
    t2new += einsum("ijab->ijab", x59)
    t2new -= einsum("ijab->jiab", x59)
    del x59
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum("ab,ijcb->ijac", f.vv, t2)
    t2new += einsum("ijab->jiab", x60)
    t2new -= einsum("ijab->jiba", x60)
    del x60
    x64 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x64 += einsum("ijab,klab->ijkl", t2, v.oovv)
    x66 += einsum("ijkl->lkji", x64) * 0.5
    x67 += einsum("ijkl->lkji", x64) * 0.5000000000000003
    del x64
    x66 += einsum("ijkl->jilk", v.oooo)
    t2new += einsum("ijab,ijkl->lkba", t2, x66) * 0.5
    del x66
    x67 += einsum("ijkl->jilk", v.oooo)
    x68 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x68 += einsum("ia,ijkl->lkja", t1, x67)
    del x67
    x68 += einsum("iajk->kjia", v.ovoo)
    t2new += einsum("ia,jkib->jkab", t1, x68)
    del x68
    x69 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x69 += einsum("wia,ijab->wjb", u11, v.oovv)
    x76 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x76 += einsum("wia->wia", x69)
    s2new = np.zeros((nbos, nbos), dtype=types[float])
    s2new += einsum("wia,xia->wx", u11, x69)
    del x69
    x70 = np.zeros((nbos, nbos), dtype=types[float])
    x70 += einsum("wia,xia->wx", gc.bov, u11)
    x74 = np.zeros((nbos, nbos), dtype=types[float])
    x74 += einsum("wx->wx", x70)
    del x70
    x71 = np.zeros((nbos, nbos), dtype=types[float])
    x71 += einsum("wia,xia->wx", g.bov, u11)
    x72 = np.zeros((nbos, nbos), dtype=types[float])
    x72 += einsum("wx->xw", x71)
    del x71
    x72 += einsum("wx->wx", w)
    x73 = np.zeros((nbos, nbos), dtype=types[float])
    x73 += einsum("wx,yw->xy", s2, x72)
    x74 += einsum("wx->wx", x73)
    del x73
    s2new += einsum("wx->wx", x74)
    s2new += einsum("wx->xw", x74)
    del x74
    u11new += einsum("wx,xia->wia", x72, u11)
    del x72
    x75 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x75 += einsum("wx,xia->wia", s2, g.bov)
    x76 += einsum("wia->wia", x75)
    del x75
    x76 += einsum("wia->wia", gc.bov)
    x79 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x79 += einsum("ia,wja->wji", t1, x76)
    u11new += einsum("wia,ijab->wjb", x76, t2)
    del x76
    x77 += einsum("ab->ab", f.vv) * -1
    u11new += einsum("ab,wib->wia", x77, u11) * -1
    del x77
    x78 += einsum("ia->ia", f.ov)
    x79 += einsum("ia,wja->wij", x78, u11)
    del x78
    x79 += einsum("wij->wij", gc.boo)
    x79 += einsum("wx,xij->wij", s2, g.boo)
    x79 -= einsum("wia,ijka->wjk", u11, v.ooov)
    u11new -= einsum("ia,wij->wja", t1, x79)
    del x79
    x80 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x80 += einsum("wab->wab", gc.bvv)
    x80 += einsum("wx,xab->wab", s2, g.bvv)
    x80 += einsum("wia,ibac->wbc", u11, v.ovvv)
    u11new += einsum("ia,wba->wib", t1, x80)
    del x80
    t1new += einsum("w,wai->ia", s1, g.bvo)
    t1new += einsum("ijab,jcab->ic", t2, v.ovvv) * -0.5
    t1new -= einsum("ia,ibja->jb", t1, v.ovov)
    t1new += einsum("ai->ia", f.vo)
    t1new += einsum("wab,wib->ia", g.bvv, u11)
    t2new -= einsum("ia,ibjk->kjba", t1, v.ovoo)
    t2new += einsum("abij->jiba", v.vvoo)
    t2new += einsum("ijab,cdab->jidc", t2, v.vvvv) * 0.5
    s1new += einsum("w,xw->x", s1, w)
    s1new += einsum("w->w", G)
    s1new += einsum("ia,wia->w", t1, gc.bov)
    u11new += einsum("wx,xai->wia", s2, g.bvo)
    u11new -= einsum("wia,ibja->wjb", u11, v.ovov)
    u11new += einsum("wai->wia", gc.bvo)

    return {"t1new": t1new, "t2new": t2new, "s1new": s1new, "s2new": s2new, "u11new": u11new}

def update_lams(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    # L1, L2, LS1 , LS2 and LU11 amplitudes
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x0 += einsum("ia,bajk->jkib", t1, l2)
    x5 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x5 += einsum("ia,jkla->jkil", t1, x0)
    x18 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x18 += einsum("ijkl->ijlk", x5) * 2.0000000000000013
    x121 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x121 += einsum("ijkl->ijlk", x5) * 2
    l1new = np.zeros((nvir, nocc), dtype=types[float])
    l1new += einsum("ijka,lkji->al", v.ooov, x5) * 0.5
    del x5
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x19 += einsum("ijab,kjlb->kila", t2, x0) * -4
    x49 = np.zeros((nocc, nvir), dtype=types[float])
    x49 += einsum("ijab,ijka->kb", t2, x0)
    x53 = np.zeros((nocc, nvir), dtype=types[float])
    x53 += einsum("ia->ia", x49) * 0.5
    del x49
    x95 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x95 -= einsum("ijka,kljb->liba", v.ooov, x0)
    x102 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x102 += einsum("ijab->ijab", x95)
    del x95
    l1new -= einsum("iajb,kjia->bk", v.ovov, x0)
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum("iabc,jkia->cbkj", v.ovvv, x0)
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 -= einsum("abij,jkab->ik", l2, t2)
    x19 += einsum("ia,jk->jika", t1, x1) * 2
    x51 = np.zeros((nocc, nocc), dtype=types[float])
    x51 += einsum("ij->ij", x1)
    x63 = np.zeros((nocc, nocc), dtype=types[float])
    x63 += einsum("ij->ij", x1) * 0.5
    x66 = np.zeros((nocc, nocc), dtype=types[float])
    x66 += einsum("ij->ij", x1) * 0.5
    x103 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x103 -= einsum("ij,kjab->ikab", x1, v.oovv)
    x105 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x105 += einsum("ijab->ijba", x103) * -0.5
    del x103
    x123 = np.zeros((nocc, nocc), dtype=types[float])
    x123 += einsum("ij->ij", x1)
    l1new += einsum("ia,ji->aj", f.ov, x1) * -0.5
    del x1
    x2 = np.zeros((nocc, nocc), dtype=types[float])
    x2 += einsum("ai,ja->ij", l1, t1)
    x51 += einsum("ij->ij", x2) * 2
    x63 += einsum("ij->ij", x2)
    x77 = np.zeros((nocc, nocc), dtype=types[float])
    x77 += einsum("ij->ij", x2)
    ls1new = np.zeros((nbos), dtype=types[float])
    ls1new -= einsum("ij,wji->w", x2, g.boo)
    x3 = np.zeros((nocc, nvir), dtype=types[float])
    x3 += einsum("w,wia->ia", s1, g.bov)
    x12 = np.zeros((nocc, nvir), dtype=types[float])
    x12 += einsum("ia->ia", x3)
    x58 = np.zeros((nocc, nvir), dtype=types[float])
    x58 += einsum("ia->ia", x3)
    x84 = np.zeros((nocc, nvir), dtype=types[float])
    x84 += einsum("ia->ia", x3)
    l1new -= einsum("ij,ja->ai", x2, x3)
    del x2
    del x3
    x4 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x4 += einsum("abij,klab->ijkl", l2, t2)
    x18 += einsum("ijkl->jilk", x4) * -1
    x19 += einsum("ia,ijkl->jlka", t1, x18) * -1
    del x18
    x121 += einsum("ijkl->jilk", x4) * -1
    l2new += einsum("ijab,klij->balk", v.oovv, x121) * -0.25
    del x121
    l1new += einsum("ijka,lkij->al", v.ooov, x4) * -0.25
    del x4
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum("abij,jkbc->ikac", l2, t2)
    l1new -= einsum("iabc,jiac->bj", v.ovvv, x6)
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum("ia,jbca->ijbc", t1, v.ovvv)
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum("ijab->jiab", x7) * -0.5
    x98 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x98 += einsum("ijab->ijab", x7)
    l1new += einsum("ijka,jkab->bi", x0, x7)
    del x7
    x8 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x8 -= einsum("wia,abji->wjb", u11, l2)
    x65 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x65 += einsum("wia->wia", x8)
    x100 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x100 += einsum("wia->wia", x8)
    l1new += einsum("wab,wia->bi", g.bvv, x8)
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 += einsum("ia,jkba->ijkb", t1, v.oovv)
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum("ijka->ikja", x9) * -1
    x14 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x14 += einsum("ijka->kjia", x9) * 0.5000000000000003
    x96 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x96 += einsum("ijka,jlkb->ilab", x0, x9)
    x102 -= einsum("ijab->ijab", x96)
    del x96
    x110 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x110 -= einsum("ai,ijkb->kjab", l1, x9)
    x118 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x118 += einsum("ijab->ijab", x110)
    del x110
    x119 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x119 += einsum("ijka->kjia", x9) * 0.5
    del x9
    x10 += einsum("ijka->kjia", v.ooov)
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum("ijab,kila->ljkb", t2, x10)
    x25 = np.zeros((nocc, nvir), dtype=types[float])
    x25 += einsum("ijab,kija->kb", t2, x10) * 0.5
    del x10
    x44 = np.zeros((nocc, nvir), dtype=types[float])
    x44 += einsum("ia->ia", x25)
    del x25
    x11 = np.zeros((nocc, nvir), dtype=types[float])
    x11 -= einsum("ia,ijba->jb", t1, v.oovv)
    x12 += einsum("ia->ia", x11)
    x58 += einsum("ia->ia", x11)
    x78 = np.zeros((nocc, nvir), dtype=types[float])
    x78 += einsum("ia->ia", x11)
    x89 = np.zeros((nocc, nocc), dtype=types[float])
    x89 += einsum("ia,ja->ij", t1, x11)
    x90 = np.zeros((nocc, nocc), dtype=types[float])
    x90 += einsum("ij->ji", x89)
    del x89
    x102 += einsum("ai,jb->ijab", l1, x11)
    x113 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x113 -= einsum("ia,jkib->kjba", x11, x0)
    del x11
    x118 -= einsum("ijab->ijab", x113)
    del x113
    x12 += einsum("ia->ia", f.ov)
    x17 += einsum("ia,jkab->ikjb", x12, t2) * 0.5
    x26 = np.zeros((nocc, nvir), dtype=types[float])
    x26 += einsum("ia,ijab->jb", x12, t2)
    x44 += einsum("ia->ia", x26) * -1
    del x26
    x34 = np.zeros((nocc, nocc), dtype=types[float])
    x34 += einsum("ia,ja->ij", t1, x12)
    x35 = np.zeros((nocc, nocc), dtype=types[float])
    x35 += einsum("ij->ji", x34)
    del x34
    lu11new = np.zeros((nbos, nvir, nocc), dtype=types[float])
    lu11new += einsum("w,ia->wai", ls1, x12) * 2
    del x12
    x13 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x13 += einsum("ijab,klab->ijkl", t2, v.oovv)
    x15 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x15 += einsum("ijkl->lkji", x13) * -1
    x120 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x120 += einsum("ijkl->lkji", x13) * 0.5
    del x13
    x14 += einsum("ijka->jika", v.ooov) * -1
    x15 += einsum("ia,jkla->kjil", t1, x14) * -4
    del x14
    x15 += einsum("ijkl->jilk", v.oooo) * -2
    x17 += einsum("ia,ijkl->jlka", t1, x15) * -0.25
    del x15
    x16 += einsum("iajb->ijab", v.ovov)
    x17 += einsum("ia,jkba->jkib", t1, x16) * -1
    del x16
    x17 += einsum("iajk->ikja", v.ovoo) * 0.5
    x17 += einsum("ijab,kcab->kjic", t2, v.ovvv) * 0.25
    l1new += einsum("abij,kija->bk", l2, x17) * -1
    del x17
    x19 += einsum("ai,jkba->ikjb", l1, t2) * -2
    l1new += einsum("ijab,kija->bk", v.oovv, x19) * -0.25
    del x19
    x20 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x20 += einsum("abic->ibac", v.vvov) * -1
    x20 += einsum("ia,bcda->icbd", t1, v.vvvv)
    l1new += einsum("abij,iabc->cj", l2, x20) * 0.5
    del x20
    x21 = np.zeros((nocc, nvir), dtype=types[float])
    x21 += einsum("w,wai->ia", s1, g.bvo)
    x44 += einsum("ia->ia", x21) * -1
    del x21
    x22 = np.zeros((nocc, nvir), dtype=types[float])
    x22 += einsum("wab,wib->ia", g.bvv, u11)
    x44 += einsum("ia->ia", x22) * -1
    del x22
    x23 = np.zeros((nocc, nvir), dtype=types[float])
    x23 += einsum("ia,ibja->jb", t1, v.ovov)
    x44 += einsum("ia->ia", x23)
    del x23
    x24 = np.zeros((nocc, nvir), dtype=types[float])
    x24 -= einsum("ijab,icab->jc", t2, v.ovvv)
    x44 += einsum("ia->ia", x24) * 0.5
    del x24
    x27 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x27 += einsum("ia,wja->wji", t1, g.bov)
    x28 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x28 += einsum("wij->wij", x27)
    x68 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x68 += einsum("wij->wij", x27)
    del x27
    x28 += einsum("wij->wij", g.boo)
    x29 = np.zeros((nocc, nvir), dtype=types[float])
    x29 += einsum("wia,wij->ja", u11, x28)
    del x28
    x44 += einsum("ia->ia", x29)
    del x29
    x30 = np.zeros((nocc, nocc), dtype=types[float])
    x30 += einsum("w,wij->ij", s1, g.boo)
    x35 += einsum("ij->ij", x30)
    x86 = np.zeros((nocc, nocc), dtype=types[float])
    x86 += einsum("ij->ij", x30)
    del x30
    x31 = np.zeros((nocc, nocc), dtype=types[float])
    x31 += einsum("wia,wja->ij", g.bov, u11)
    x35 += einsum("ij->ij", x31)
    x86 += einsum("ij->ij", x31)
    del x31
    x32 = np.zeros((nocc, nocc), dtype=types[float])
    x32 -= einsum("ia,ijka->jk", t1, v.ooov)
    x35 += einsum("ij->ij", x32)
    x90 += einsum("ij->ij", x32)
    del x32
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x91 += einsum("ij,abjk->kiab", x90, l2)
    del x90
    x92 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x92 -= einsum("ijab->ijba", x91)
    del x91
    x33 = np.zeros((nocc, nocc), dtype=types[float])
    x33 -= einsum("ijab,kiab->jk", t2, v.oovv)
    x35 += einsum("ij->ji", x33) * 0.5
    x104 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x104 += einsum("ij,abki->kjab", x33, l2)
    del x33
    x105 += einsum("ijab->ijba", x104) * -0.5
    del x104
    l2new += einsum("ijab->abij", x105) * -1
    l2new += einsum("ijab->abji", x105)
    del x105
    x35 += einsum("ij->ij", f.oo)
    x36 = np.zeros((nocc, nvir), dtype=types[float])
    x36 += einsum("ia,ij->ja", t1, x35)
    x44 += einsum("ia->ia", x36)
    del x36
    l1new += einsum("ai,ji->aj", l1, x35) * -1
    del x35
    x37 = np.zeros((nvir, nvir), dtype=types[float])
    x37 += einsum("w,wab->ab", s1, g.bvv)
    x39 = np.zeros((nvir, nvir), dtype=types[float])
    x39 += einsum("ab->ab", x37)
    x70 = np.zeros((nvir, nvir), dtype=types[float])
    x70 += einsum("ab->ab", x37)
    x115 = np.zeros((nvir, nvir), dtype=types[float])
    x115 += einsum("ab->ab", x37)
    del x37
    x38 = np.zeros((nvir, nvir), dtype=types[float])
    x38 -= einsum("ia,ibac->bc", t1, v.ovvv)
    x39 += einsum("ab->ab", x38) * -1
    x70 -= einsum("ab->ab", x38)
    x112 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x112 -= einsum("ab,caij->jicb", x38, l2)
    del x38
    x118 -= einsum("ijab->ijab", x112)
    del x112
    x39 += einsum("ab->ab", f.vv)
    x40 = np.zeros((nocc, nvir), dtype=types[float])
    x40 += einsum("ia,ba->ib", t1, x39)
    del x39
    x44 += einsum("ia->ia", x40) * -1
    del x40
    x41 = np.zeros((nbos), dtype=types[float])
    x41 += einsum("ia,wia->w", t1, g.bov)
    x42 = np.zeros((nbos), dtype=types[float])
    x42 += einsum("w->w", x41)
    x74 = np.zeros((nbos), dtype=types[float])
    x74 += einsum("w->w", x41)
    del x41
    x42 += einsum("w->w", G)
    x43 = np.zeros((nocc, nvir), dtype=types[float])
    x43 += einsum("w,wia->ia", x42, u11)
    del x42
    x44 += einsum("ia->ia", x43) * -1
    del x43
    x44 += einsum("ai->ia", f.vo) * -1
    l1new += einsum("ia,abij->bj", x44, l2) * -1
    ls1new += einsum("ia,wai->w", x44, lu11) * -1
    del x44
    x45 = np.zeros((nocc, nvir), dtype=types[float])
    x45 += einsum("w,wia->ia", ls1, u11)
    x53 += einsum("ia->ia", x45) * -1
    del x45
    x46 = np.zeros((nocc, nvir), dtype=types[float])
    x46 += einsum("ai,jiba->jb", l1, t2)
    x53 += einsum("ia->ia", x46) * -1
    del x46
    x47 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x47 += einsum("ia,waj->wji", t1, lu11)
    x48 = np.zeros((nocc, nvir), dtype=types[float])
    x48 += einsum("wia,wij->ja", u11, x47)
    x53 += einsum("ia->ia", x48)
    del x48
    x50 = np.zeros((nocc, nocc), dtype=types[float])
    x50 += einsum("wai,wja->ij", lu11, u11)
    x51 += einsum("ij->ij", x50) * 2
    x52 = np.zeros((nocc, nvir), dtype=types[float])
    x52 += einsum("ia,ij->ja", t1, x51) * 0.5
    del x51
    x53 += einsum("ia->ia", x52)
    del x52
    x63 += einsum("ij->ij", x50)
    l1new += einsum("ij,jkia->ak", x63, v.ooov) * -1
    del x63
    x66 += einsum("ij->ij", x50)
    x67 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x67 += einsum("w,ij->wij", s1, x66)
    del x66
    x77 += einsum("ij->ij", x50)
    x88 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x88 += einsum("ij,jkab->kiab", x77, v.oovv)
    x92 += einsum("ijab->jiba", x88)
    del x88
    x123 += einsum("ij->ij", x50) * 2
    del x50
    ls1new += einsum("ij,wji->w", x123, g.boo) * -0.5
    del x123
    x53 += einsum("ia->ia", t1) * -1
    l1new += einsum("ia,ijab->bj", x53, v.oovv) * -1
    ls1new += einsum("ia,wia->w", x53, g.bov) * -1
    del x53
    x54 = np.zeros((nvir, nvir), dtype=types[float])
    x54 += einsum("ai,ib->ab", l1, t1)
    x57 = np.zeros((nvir, nvir), dtype=types[float])
    x57 += einsum("ab->ab", x54)
    ls1new += einsum("ab,wab->w", x54, g.bvv)
    del x54
    x55 = np.zeros((nvir, nvir), dtype=types[float])
    x55 += einsum("wai,wib->ab", lu11, u11)
    x57 += einsum("ab->ab", x55)
    x111 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x111 -= einsum("ab,ijcb->jiac", x55, v.oovv)
    x118 += einsum("ijab->ijab", x111)
    del x111
    x122 = np.zeros((nvir, nvir), dtype=types[float])
    x122 += einsum("ab->ab", x55)
    del x55
    x56 = np.zeros((nvir, nvir), dtype=types[float])
    x56 -= einsum("abij,ijbc->ac", l2, t2)
    x57 += einsum("ab->ab", x56) * 0.5
    l1new += einsum("ab,iabc->ci", x57, v.ovvv) * -1
    del x57
    x108 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x108 -= einsum("ab,ijcb->ijac", x56, v.oovv)
    x109 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x109 += einsum("ijab->ijab", x108) * 0.5
    del x108
    x122 += einsum("ab->ab", x56) * 0.5
    del x56
    ls1new += einsum("ab,wab->w", x122, g.bvv)
    del x122
    x58 += einsum("ia->ia", f.ov)
    x61 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x61 += einsum("ia,wja->wij", x58, u11)
    x73 = np.zeros((nbos), dtype=types[float])
    x73 += einsum("ia,wia->w", x58, u11)
    del x58
    x76 = np.zeros((nbos), dtype=types[float])
    x76 += einsum("w->w", x73)
    del x73
    x59 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x59 += einsum("wia,ijab->wjb", u11, v.oovv)
    x60 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x60 += einsum("wia->wia", x59)
    x69 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x69 += einsum("wia->wia", x59)
    x94 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x94 += einsum("wai,wjb->ijab", lu11, x59)
    del x59
    x102 += einsum("ijab->ijab", x94)
    del x94
    x60 += einsum("wia->wia", gc.bov)
    x60 += einsum("wx,wia->xia", s2, g.bov)
    x61 += einsum("ia,wja->wji", t1, x60)
    del x60
    x61 += einsum("wij->wij", gc.boo)
    x61 += einsum("wx,wij->xij", s2, g.boo)
    x61 += einsum("wia,jika->wjk", u11, v.ooov)
    l1new -= einsum("wai,wji->aj", lu11, x61)
    del x61
    x62 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x62 += einsum("wab->wab", gc.bvv)
    x62 += einsum("wx,wab->xab", s2, g.bvv)
    x62 -= einsum("wia,ibca->wbc", u11, v.ovvv)
    l1new += einsum("wai,wab->bi", lu11, x62)
    del x62
    x64 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x64 += einsum("wx,xai->wia", s2, lu11)
    x65 += einsum("wia->wia", x64)
    x67 += einsum("ia,wja->wji", t1, x65)
    del x65
    x100 += einsum("wia->wia", x64)
    del x64
    x101 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x101 += einsum("wia,wjb->ijab", g.bov, x100)
    del x100
    x102 += einsum("ijab->ijab", x101)
    del x101
    x67 += einsum("ai,wja->wij", l1, u11)
    l1new += einsum("wia,wji->aj", g.bov, x67) * -1
    del x67
    x68 += einsum("wij->wij", g.boo)
    x129 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x129 += einsum("ia,wij->wja", t1, x68)
    x130 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x130 -= einsum("wia->wia", x129)
    del x129
    l1new -= einsum("wij,wja->ai", x68, x8)
    del x8
    del x68
    x69 += einsum("wia->wia", gc.bov)
    l1new -= einsum("wij,wja->ai", x47, x69)
    del x69
    del x47
    x70 += einsum("ab->ab", f.vv)
    l1new += einsum("ai,ab->bi", l1, x70)
    del x70
    x71 = np.zeros((nbos), dtype=types[float])
    x71 += einsum("w,xw->x", s1, w)
    x76 += einsum("w->w", x71)
    del x71
    x72 = np.zeros((nbos), dtype=types[float])
    x72 += einsum("ia,wia->w", t1, gc.bov)
    x76 += einsum("w->w", x72)
    del x72
    x74 += einsum("w->w", G)
    x75 = np.zeros((nbos), dtype=types[float])
    x75 += einsum("w,wx->x", x74, s2)
    x76 += einsum("w->w", x75)
    del x75
    x132 = np.zeros((nbos, nbos), dtype=types[float])
    x132 += einsum("w,x->xw", ls1, x74)
    del x74
    x76 += einsum("w->w", G)
    l1new += einsum("w,wai->ai", x76, lu11)
    ls1new += einsum("w,wx->x", x76, ls2)
    del x76
    x78 += einsum("ia->ia", f.ov)
    l1new -= einsum("ij,ja->ai", x77, x78)
    del x77
    del x78
    x79 = np.zeros((nbos), dtype=types[float])
    x79 += einsum("w->w", s1)
    x79 += einsum("w,xw->x", ls1, s2)
    x79 += einsum("ai,wia->w", l1, u11)
    l1new += einsum("w,wia->ai", x79, g.bov)
    del x79
    x80 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x80 += einsum("ai,jkib->jkab", l1, v.ooov)
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 -= einsum("ijab->jiab", x80)
    del x80
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum("ab,caij->ijbc", f.vv, l2)
    x82 -= einsum("ijab->jiab", x81)
    del x81
    l2new -= einsum("ijab->abij", x82)
    l2new += einsum("ijab->baij", x82)
    del x82
    x83 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x83 += einsum("ai,jabc->ijbc", l1, v.ovvv)
    x92 += einsum("ijab->ijba", x83)
    del x83
    x84 += einsum("ia->ia", f.ov)
    x85 = np.zeros((nocc, nocc), dtype=types[float])
    x85 += einsum("ia,ja->ij", t1, x84)
    x86 += einsum("ij->ji", x85)
    del x85
    x102 += einsum("ai,jb->jiba", l1, x84)
    x117 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x117 += einsum("ia,jkib->jkba", x84, x0)
    del x0
    del x84
    x118 -= einsum("ijab->jiba", x117)
    del x117
    x86 += einsum("ij->ij", f.oo)
    x87 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x87 += einsum("ij,abjk->kiab", x86, l2)
    del x86
    x92 += einsum("ijab->jiba", x87)
    del x87
    l2new += einsum("ijab->abij", x92)
    l2new -= einsum("ijab->abji", x92)
    del x92
    x93 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x93 += einsum("wia,wbj->ijab", gc.bov, lu11)
    x102 += einsum("ijab->ijab", x93)
    del x93
    x97 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x97 += einsum("ijab,kica->jkbc", t2, v.oovv)
    x98 += einsum("ijab->ijab", x97)
    del x97
    x98 -= einsum("iajb->jiab", v.ovov)
    x99 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x99 += einsum("abij,ikac->jkbc", l2, x98)
    del x98
    x102 += einsum("ijab->ijab", x99)
    del x99
    l2new += einsum("ijab->abij", x102)
    l2new -= einsum("ijab->baij", x102)
    l2new -= einsum("ijab->abji", x102)
    l2new += einsum("ijab->baji", x102)
    del x102
    x106 = np.zeros((nvir, nvir), dtype=types[float])
    x106 -= einsum("ijab,ijca->bc", t2, v.oovv)
    x107 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x107 -= einsum("ab,caij->jicb", x106, l2)
    del x106
    x109 += einsum("ijab->ijab", x107) * 0.5
    del x107
    l2new += einsum("ijab->abij", x109) * -1
    l2new += einsum("ijab->baij", x109)
    del x109
    x114 = np.zeros((nvir, nvir), dtype=types[float])
    x114 += einsum("wia,wib->ab", g.bov, u11)
    x115 -= einsum("ab->ba", x114)
    del x114
    x116 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x116 += einsum("ab,acij->ijcb", x115, l2)
    del x115
    x118 -= einsum("ijab->jiba", x116)
    del x116
    l2new += einsum("ijab->abij", x118)
    l2new -= einsum("ijab->baij", x118)
    del x118
    x119 += einsum("ijka->jika", v.ooov) * -1
    x120 += einsum("ia,jkla->kjli", t1, x119) * -2
    del x119
    x120 += einsum("ijkl->jilk", v.oooo)
    l2new += einsum("abij,klij->balk", l2, x120) * 0.5
    del x120
    x124 = np.zeros((nbos, nbos), dtype=types[float])
    x124 += einsum("wx,xy->wy", ls2, w)
    x132 += einsum("wx->wx", x124)
    del x124
    x125 = np.zeros((nbos, nbos), dtype=types[float])
    x125 += einsum("wia,xia->wx", g.bov, u11)
    x126 = np.zeros((nbos, nbos), dtype=types[float])
    x126 += einsum("wx,yx->yw", ls2, x125)
    del x125
    x132 += einsum("wx->wx", x126)
    del x126
    x127 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x127 += einsum("ia,wba->wib", t1, g.bvv)
    x130 += einsum("wia->wia", x127)
    del x127
    x128 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x128 += einsum("wia,jiba->wjb", g.bov, t2)
    x130 += einsum("wia->wia", x128)
    del x128
    x130 += einsum("wai->wia", g.bvo)
    x131 = np.zeros((nbos, nbos), dtype=types[float])
    x131 += einsum("wai,xia->wx", lu11, x130)
    del x130
    x132 += einsum("wx->xw", x131)
    del x131
    ls2new = np.zeros((nbos, nbos), dtype=types[float])
    ls2new += einsum("wx->wx", x132)
    ls2new += einsum("wx->xw", x132)
    del x132
    l1new += einsum("ia->ai", f.ov)
    l1new += einsum("w,wia->ai", ls1, gc.bov)
    l1new -= einsum("ai,jaib->bj", l1, v.ovov)
    l2new += einsum("abij,abcd->dcji", l2, v.vvvv) * 0.5
    l2new += einsum("ijab->baji", v.oovv)
    ls1new += einsum("w,wx->x", ls1, w)
    ls1new += einsum("ai,wai->w", l1, g.bvo)
    ls1new += einsum("w->w", G)
    lu11new += einsum("wx,xia->wai", ls2, gc.bov)
    lu11new += einsum("wx,wai->xai", w, lu11)
    lu11new -= einsum("ij,waj->wai", f.oo, lu11)
    lu11new -= einsum("ai,wji->waj", l1, g.boo)
    lu11new += einsum("wia->wai", g.bov)
    lu11new += einsum("ab,wai->wbi", f.vv, lu11)
    lu11new += einsum("ai,wab->wbi", l1, g.bvv)
    lu11new += einsum("wai,baji->wbj", g.bvo, l2)
    lu11new -= einsum("wai,jaib->wbj", lu11, v.ovov)

    return {"l1new": l1new, "l2new": l2new, "ls1new": ls1new, "ls2new": ls2new, "lu11new": lu11new}

def make_rdm1_f(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    delta_oo = np.eye(nocc)
    delta_vv = np.eye(nvir)

    # 1RDM
    x0 = np.zeros((nocc, nocc), dtype=types[float])
    x0 += einsum("ai,ja->ij", l1, t1)
    x5 = np.zeros((nocc, nocc), dtype=types[float])
    x5 += einsum("ij->ij", x0)
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=types[float])
    rdm1_f_oo -= einsum("ij->ij", x0)
    del x0
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 += einsum("wai,wja->ij", lu11, u11)
    x5 += einsum("ij->ij", x1)
    rdm1_f_oo -= einsum("ij->ij", x1)
    del x1
    x2 = np.zeros((nocc, nocc), dtype=types[float])
    x2 += einsum("abij,kjab->ik", l2, t2)
    x5 += einsum("ij->ij", x2) * 0.5
    rdm1_f_vo = np.zeros((nvir, nocc), dtype=types[float])
    rdm1_f_vo += einsum("ia,ij->aj", t1, x5) * -1
    del x5
    rdm1_f_oo += einsum("ij->ij", x2) * -0.5
    del x2
    x3 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x3 += einsum("ia,waj->wji", t1, lu11)
    rdm1_f_vo -= einsum("wia,wij->aj", u11, x3)
    del x3
    x4 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x4 += einsum("ia,bajk->jkib", t1, l2)
    rdm1_f_vo += einsum("ijab,ijkb->ak", t2, x4) * 0.5
    del x4
    rdm1_f_oo += einsum("ij->ji", delta_oo)
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=types[float])
    rdm1_f_ov += einsum("ai->ia", l1)
    rdm1_f_vo += einsum("ai,jiba->bj", l1, t2)
    rdm1_f_vo += einsum("w,wia->ai", ls1, u11)
    rdm1_f_vo += einsum("ia->ai", t1)
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=types[float])
    rdm1_f_vv += einsum("wai,wib->ba", lu11, u11)
    rdm1_f_vv += einsum("ai,ib->ba", l1, t1)
    rdm1_f_vv += einsum("abij,ijca->cb", l2, t2) * -0.5

    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])

    return rdm1_f

def make_rdm2_f(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    delta_oo = np.eye(nocc)
    delta_vv = np.eye(nvir)

    # 2RDM
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x0 += einsum("abij,klab->ijkl", l2, t2)
    x20 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x20 += einsum("ijkl->jilk", x0) * -1
    x51 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x51 += einsum("ijkl->jilk", x0) * -1
    x52 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x52 += einsum("ijkl->jilk", x0) * -1
    rdm2_f_oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_oooo += einsum("ijkl->jilk", x0) * 0.5
    del x0
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum("ia,bajk->jkib", t1, l2)
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum("ia,jkla->jkil", t1, x1)
    x20 += einsum("ijkl->ijlk", x2) * 2
    x21 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x21 += einsum("ia,ijkl->jkla", t1, x20) * 0.5
    del x20
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_ovoo += einsum("ijka->iakj", x21) * -1
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_vooo += einsum("ijka->aikj", x21)
    del x21
    x51 += einsum("ijkl->ijlk", x2) * 2.0000000000000013
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vvoo += einsum("ijab,ijkl->balk", t2, x51) * -0.25
    del x51
    x52 += einsum("ijkl->ijlk", x2) * 1.9999999999999987
    x53 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x53 += einsum("ia,ijkl->jlka", t1, x52) * -1
    del x52
    rdm2_f_vvoo += einsum("ia,ijkb->abkj", t1, x53) * -0.5000000000000003
    del x53
    rdm2_f_oooo -= einsum("ijkl->ijlk", x2)
    del x2
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 -= einsum("ijab,kjlb->klia", t2, x1)
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum("ijka->ijka", x9)
    x25 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x25 -= einsum("ijka->ijka", x9)
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x34 += einsum("ia,ijkb->jkab", t1, x9)
    del x9
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 -= einsum("ijab->ijab", x34)
    del x34
    x16 = np.zeros((nocc, nvir), dtype=types[float])
    x16 -= einsum("ijab,ijkb->ka", t2, x1)
    x18 = np.zeros((nocc, nvir), dtype=types[float])
    x18 += einsum("ia->ia", x16)
    del x16
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum("ia,jikb->jkba", t1, x1)
    x59 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x59 -= einsum("ia,ijbc->jbca", t1, x28)
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov -= einsum("iabc->cbia", x59)
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo += einsum("iabc->cbai", x59)
    del x59
    rdm2_f_ovov = np.zeros((nocc, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_ovov += einsum("ijab->ibja", x28)
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_ovvo -= einsum("ijab->ibaj", x28)
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_voov -= einsum("ijab->bija", x28)
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_vovo += einsum("ijab->biaj", x28)
    del x28
    x58 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x58 += einsum("ijab,ijkc->kcab", t2, x1)
    rdm2_f_vvov += einsum("iabc->bcia", x58) * 0.5
    rdm2_f_vvvo += einsum("iabc->bcai", x58) * -0.5
    del x58
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov -= einsum("ijka->jika", x1)
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo += einsum("ijka->jiak", x1)
    del x1
    x3 = np.zeros((nocc, nocc), dtype=types[float])
    x3 += einsum("abij,kjab->ik", l2, t2)
    x17 = np.zeros((nocc, nvir), dtype=types[float])
    x17 += einsum("ia,ij->ja", t1, x3)
    x18 += einsum("ia->ia", x17)
    del x17
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x19 += einsum("ij,ka->jika", delta_oo, x18) * 0.5
    rdm2_f_vooo += einsum("ij,ka->aijk", delta_oo, x18) * 0.5
    rdm2_f_vooo += einsum("ij,ka->aikj", delta_oo, x18) * -0.5
    rdm2_f_vvoo += einsum("ia,jb->abij", t1, x18) * -0.5000000000000003
    rdm2_f_vvoo += einsum("ia,jb->baij", t1, x18) * 0.5000000000000003
    rdm2_f_vvoo += einsum("ia,jb->abji", t1, x18) * 0.5000000000000003
    rdm2_f_vvoo += einsum("ia,jb->baji", t1, x18) * -0.5000000000000003
    del x18
    x19 += einsum("ia,jk->jika", t1, x3) * -0.5
    rdm2_f_ovoo += einsum("ijka->iajk", x19) * -1
    rdm2_f_ovoo += einsum("ijka->iakj", x19)
    del x19
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum("ij,kiab->kjab", x3, t2)
    rdm2_f_vvoo += einsum("ijab->abij", x50) * -0.5
    rdm2_f_vvoo += einsum("ijab->abji", x50) * 0.5
    del x50
    rdm2_f_oooo += einsum("ij,kl->ikjl", delta_oo, x3) * -0.5
    rdm2_f_oooo += einsum("ij,kl->iklj", delta_oo, x3) * 0.5
    rdm2_f_oooo += einsum("ij,kl->kjil", delta_oo, x3) * 0.5
    rdm2_f_oooo += einsum("ij,kl->kjli", delta_oo, x3) * -0.5
    rdm2_f_vooo += einsum("ia,jk->ajik", t1, x3) * -0.5
    rdm2_f_vooo += einsum("ia,jk->ajki", t1, x3) * 0.5
    del x3
    x4 = np.zeros((nocc, nocc), dtype=types[float])
    x4 += einsum("ai,ja->ij", l1, t1)
    x14 = np.zeros((nocc, nocc), dtype=types[float])
    x14 += einsum("ij->ij", x4)
    x23 = np.zeros((nocc, nvir), dtype=types[float])
    x23 += einsum("ia,ij->ja", t1, x4)
    x24 = np.zeros((nocc, nvir), dtype=types[float])
    x24 -= einsum("ia->ia", x23)
    del x23
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum("ij,kiab->jkab", x4, t2)
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 -= einsum("ijab->ijba", x44)
    del x44
    rdm2_f_oooo -= einsum("ij,kl->jkil", delta_oo, x4)
    rdm2_f_oooo += einsum("ij,kl->kjil", delta_oo, x4)
    rdm2_f_oooo += einsum("ij,kl->jkli", delta_oo, x4)
    rdm2_f_oooo -= einsum("ij,kl->kjli", delta_oo, x4)
    del x4
    x5 = np.zeros((nocc, nocc), dtype=types[float])
    x5 += einsum("wai,wja->ij", lu11, u11)
    x12 = np.zeros((nocc, nvir), dtype=types[float])
    x12 += einsum("ia,ij->ja", t1, x5)
    x13 = np.zeros((nocc, nvir), dtype=types[float])
    x13 += einsum("ia->ia", x12)
    del x12
    x14 += einsum("ij->ij", x5)
    x15 -= einsum("ia,jk->jika", t1, x14)
    x25 += einsum("ia,jk->jika", t1, x14)
    x54 = np.zeros((nocc, nvir), dtype=types[float])
    x54 += einsum("ia,ij->ja", t1, x14)
    del x14
    x55 = np.zeros((nocc, nvir), dtype=types[float])
    x55 += einsum("ia->ia", x54)
    x56 = np.zeros((nocc, nvir), dtype=types[float])
    x56 += einsum("ia->ia", x54)
    del x54
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum("ij,kiab->kjab", x5, t2)
    x46 += einsum("ijab->ijba", x45)
    del x45
    rdm2_f_vvoo -= einsum("ijab->baij", x46)
    rdm2_f_vvoo += einsum("ijab->baji", x46)
    del x46
    rdm2_f_oooo -= einsum("ij,kl->ikjl", delta_oo, x5)
    rdm2_f_oooo += einsum("ij,kl->iklj", delta_oo, x5)
    rdm2_f_oooo += einsum("ij,kl->kjil", delta_oo, x5)
    rdm2_f_oooo -= einsum("ij,kl->kjli", delta_oo, x5)
    del x5
    x6 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x6 += einsum("ai,jkba->ijkb", l1, t2)
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum("ia,ijkb->jkab", t1, x6)
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum("ijab->ijab", x40)
    del x40
    rdm2_f_ovoo -= einsum("ijka->iakj", x6)
    rdm2_f_vooo += einsum("ijka->aikj", x6)
    del x6
    x7 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x7 += einsum("ia,waj->wji", t1, lu11)
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x8 += einsum("wia,wjk->jkia", u11, x7)
    x15 += einsum("ijka->ijka", x8)
    x25 -= einsum("ijka->ijka", x8)
    del x8
    rdm2_f_vooo -= einsum("ijka->aijk", x25)
    rdm2_f_vooo += einsum("ijka->aikj", x25)
    del x25
    x11 = np.zeros((nocc, nvir), dtype=types[float])
    x11 += einsum("wia,wij->ja", u11, x7)
    x13 += einsum("ia->ia", x11)
    x55 += einsum("ia->ia", x11)
    x56 += einsum("ia->ia", x11)
    del x11
    x36 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x36 += einsum("ia,wij->wja", t1, x7)
    del x7
    x37 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x37 -= einsum("wia->wia", x36)
    del x36
    x10 = np.zeros((nocc, nvir), dtype=types[float])
    x10 += einsum("ai,jiba->jb", l1, t2)
    x13 -= einsum("ia->ia", x10)
    x15 += einsum("ij,ka->jika", delta_oo, x13)
    rdm2_f_ovoo -= einsum("ijka->iajk", x15)
    rdm2_f_ovoo += einsum("ijka->iakj", x15)
    del x15
    rdm2_f_vooo += einsum("ij,ka->aijk", delta_oo, x13)
    rdm2_f_vooo -= einsum("ij,ka->aikj", delta_oo, x13)
    del x13
    x39 += einsum("ia,jb->ijab", t1, x10)
    del x10
    x22 = np.zeros((nocc, nvir), dtype=types[float])
    x22 += einsum("w,wia->ia", ls1, u11)
    x24 += einsum("ia->ia", x22)
    x55 -= einsum("ia->ia", x22)
    x56 -= einsum("ia->ia", x22)
    del x22
    rdm2_f_vvoo += einsum("ia,jb->abji", t1, x56)
    rdm2_f_vvoo -= einsum("ia,jb->baji", t1, x56)
    del x56
    x24 += einsum("ia->ia", t1)
    rdm2_f_ovoo += einsum("ij,ka->jaik", delta_oo, x24)
    rdm2_f_ovoo -= einsum("ij,ka->jaki", delta_oo, x24)
    rdm2_f_vooo += einsum("ij,ka->ajki", delta_oo, x24)
    rdm2_f_vooo -= einsum("ij,ka->ajik", delta_oo, x24)
    del x24
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum("wai,wjb->ijab", lu11, u11)
    rdm2_f_ovov -= einsum("ijab->ibja", x26)
    rdm2_f_ovvo += einsum("ijab->ibaj", x26)
    rdm2_f_voov += einsum("ijab->bija", x26)
    rdm2_f_vovo -= einsum("ijab->biaj", x26)
    del x26
    x27 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x27 -= einsum("abij,kjca->ikbc", l2, t2)
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum("ijab,jkbc->ikac", t2, x27)
    x43 += einsum("ijab->ijab", x42)
    del x42
    x63 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x63 += einsum("ia,ijbc->jbac", t1, x27)
    x65 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x65 -= einsum("iabc->iabc", x63)
    del x63
    rdm2_f_ovov -= einsum("ijab->ibja", x27)
    rdm2_f_ovvo += einsum("ijab->ibaj", x27)
    rdm2_f_voov += einsum("ijab->bija", x27)
    rdm2_f_vovo -= einsum("ijab->biaj", x27)
    del x27
    x29 = np.zeros((nvir, nvir), dtype=types[float])
    x29 += einsum("ai,ib->ab", l1, t1)
    x32 = np.zeros((nvir, nvir), dtype=types[float])
    x32 += einsum("ab->ab", x29)
    x33 = np.zeros((nvir, nvir), dtype=types[float])
    x33 += einsum("ab->ab", x29) * 2
    x64 = np.zeros((nvir, nvir), dtype=types[float])
    x64 += einsum("ab->ab", x29)
    del x29
    x30 = np.zeros((nvir, nvir), dtype=types[float])
    x30 += einsum("wai,wib->ab", lu11, u11)
    x32 += einsum("ab->ab", x30)
    x33 += einsum("ab->ab", x30) * 2
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum("ab,ijca->ijcb", x30, t2)
    x43 -= einsum("ijab->ijab", x41)
    del x41
    rdm2_f_vvoo -= einsum("ijab->abji", x43)
    rdm2_f_vvoo += einsum("ijab->baji", x43)
    del x43
    x64 += einsum("ab->ab", x30)
    del x30
    x65 += einsum("ia,bc->ibac", t1, x64)
    del x64
    x31 = np.zeros((nvir, nvir), dtype=types[float])
    x31 -= einsum("abij,ijca->bc", l2, t2)
    x32 += einsum("ab->ab", x31) * 0.5
    rdm2_f_ovov += einsum("ij,ab->jbia", delta_oo, x32)
    rdm2_f_voov += einsum("ij,ab->bjia", delta_oo, x32) * -1
    del x32
    x33 += einsum("ab->ab", x31)
    rdm2_f_ovvo += einsum("ij,ab->jbai", delta_oo, x33) * -0.5
    rdm2_f_vovo += einsum("ij,ab->bjai", delta_oo, x33) * 0.5
    del x33
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum("ab,ijca->ijcb", x31, t2)
    rdm2_f_vvoo += einsum("ijab->abji", x49) * 0.5
    rdm2_f_vvoo += einsum("ijab->baji", x49) * -0.5
    del x49
    rdm2_f_vvov += einsum("ia,bc->acib", t1, x31) * 0.5
    rdm2_f_vvov += einsum("ia,bc->caib", t1, x31) * -0.5
    rdm2_f_vvvo += einsum("ia,bc->acbi", t1, x31) * -0.5
    rdm2_f_vvvo += einsum("ia,bc->cabi", t1, x31) * 0.5
    del x31
    x35 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x35 += einsum("wai,jiba->wjb", lu11, t2)
    x37 += einsum("wia->wia", x35)
    del x35
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum("wia,wjb->jiba", u11, x37)
    del x37
    x39 += einsum("ijab->ijab", x38)
    del x38
    rdm2_f_vvoo += einsum("ijab->abij", x39)
    rdm2_f_vvoo -= einsum("ijab->baij", x39)
    rdm2_f_vvoo -= einsum("ijab->abji", x39)
    rdm2_f_vvoo += einsum("ijab->baji", x39)
    del x39
    x47 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x47 += einsum("wx,xia->wia", ls2, u11)
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum("wia,wjb->jiba", u11, x47)
    del x47
    rdm2_f_vvoo -= einsum("ijab->baij", x48)
    rdm2_f_vvoo += einsum("ijab->baji", x48)
    del x48
    x55 -= einsum("ia->ia", t1)
    rdm2_f_vvoo += einsum("ia,jb->baij", t1, x55)
    rdm2_f_vvoo -= einsum("ia,jb->abij", t1, x55)
    del x55
    x57 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x57 += einsum("ia,bcji->jbca", t1, l2)
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv -= einsum("iabc->icba", x57)
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv += einsum("iabc->ciba", x57)
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv -= einsum("ia,ibcd->adcb", t1, x57)
    del x57
    x60 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x60 += einsum("ai,jibc->jabc", l1, t2)
    rdm2_f_vvov -= einsum("iabc->cbia", x60)
    rdm2_f_vvvo += einsum("iabc->cbai", x60)
    del x60
    x61 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x61 += einsum("ia,wbi->wba", t1, lu11)
    x62 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x62 += einsum("wia,wbc->ibca", u11, x61)
    del x61
    x65 -= einsum("iabc->iabc", x62)
    del x62
    rdm2_f_vvov += einsum("iabc->bcia", x65)
    rdm2_f_vvov -= einsum("iabc->cbia", x65)
    rdm2_f_vvvo -= einsum("iabc->bcai", x65)
    rdm2_f_vvvo += einsum("iabc->cbai", x65)
    del x65
    rdm2_f_oooo += einsum("ij,kl->jlik", delta_oo, delta_oo)
    rdm2_f_oooo -= einsum("ij,kl->lijk", delta_oo, delta_oo)
    rdm2_f_ooov += einsum("ij,ak->jkia", delta_oo, l1)
    rdm2_f_ooov -= einsum("ij,ak->kjia", delta_oo, l1)
    rdm2_f_oovo -= einsum("ij,ak->jkai", delta_oo, l1)
    rdm2_f_oovo += einsum("ij,ak->kjai", delta_oo, l1)
    rdm2_f_oovv = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_oovv += einsum("abij->jiba", l2)
    rdm2_f_ovov -= einsum("ai,jb->ibja", l1, t1)
    rdm2_f_ovvo += einsum("ai,jb->ibaj", l1, t1)
    rdm2_f_voov += einsum("ai,jb->bija", l1, t1)
    rdm2_f_vovo -= einsum("ai,jb->biaj", l1, t1)
    rdm2_f_vvoo += einsum("ijab->baji", t2)
    rdm2_f_vvvv += einsum("abij,ijcd->dcba", l2, t2) * 0.5

    rdm2_f = pack_2e(rdm2_f_oooo, rdm2_f_ooov, rdm2_f_oovo, rdm2_f_ovoo, rdm2_f_vooo, rdm2_f_oovv, rdm2_f_ovov, rdm2_f_ovvo, rdm2_f_voov, rdm2_f_vovo, rdm2_f_vvoo, rdm2_f_ovvv, rdm2_f_vovv, rdm2_f_vvov, rdm2_f_vvvo, rdm2_f_vvvv)

    rdm2_f = rdm2_f.transpose(0, 2, 1, 3)

    return rdm2_f

def make_sing_b_dm(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    # Single boson DM
    dm_b_cre = np.zeros((nbos), dtype=types[float])
    dm_b_cre += einsum("w->w", ls1)
    dm_b_des = np.zeros((nbos), dtype=types[float])
    dm_b_des += einsum("w,xw->x", ls1, s2)
    dm_b_des += einsum("ai,wia->w", l1, u11)
    dm_b_des += einsum("w->w", s1)

    dm_b = np.array([dm_b_cre, dm_b_des])

    return dm_b

def make_rdm1_b(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    # Boson 1RDM
    rdm1_b = np.zeros((nbos, nbos), dtype=types[float])
    rdm1_b += einsum("wx,yx->wy", ls2, s2)
    rdm1_b += einsum("w,x->wx", ls1, s1)
    rdm1_b += einsum("wai,xia->wx", lu11, u11)

    return rdm1_b

def make_eb_coup_rdm(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    delta_oo = np.eye(nocc)
    delta_vv = np.eye(nvir)

    # Boson-fermion coupling RDM
    x0 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x0 += einsum("ia,waj->wji", t1, lu11)
    x21 = np.zeros((nocc, nvir), dtype=types[float])
    x21 += einsum("wia,wij->ja", u11, x0)
    rdm_eb_cre_oo = np.zeros((nbos, nocc, nocc), dtype=types[float])
    rdm_eb_cre_oo -= einsum("wij->wji", x0)
    rdm_eb_cre_ov = np.zeros((nbos, nocc, nvir), dtype=types[float])
    rdm_eb_cre_ov -= einsum("ia,wij->wja", t1, x0)
    del x0
    x1 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x1 += einsum("ai,wja->wij", l1, u11)
    x17 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x17 += einsum("wij->wij", x1)
    rdm_eb_des_oo = np.zeros((nbos, nocc, nocc), dtype=types[float])
    rdm_eb_des_oo -= einsum("wij->wji", x1)
    del x1
    x2 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x2 += einsum("wx,xai->wia", s2, lu11)
    x4 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x4 += einsum("wia->wia", x2)
    rdm_eb_des_vo = np.zeros((nbos, nvir, nocc), dtype=types[float])
    rdm_eb_des_vo += einsum("wia->wai", x2)
    del x2
    x3 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x3 -= einsum("wia,abji->wjb", u11, l2)
    x4 += einsum("wia->wia", x3)
    x5 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x5 += einsum("ia,wja->wij", t1, x4)
    x17 += einsum("wij->wji", x5)
    rdm_eb_des_ov = np.zeros((nbos, nocc, nvir), dtype=types[float])
    rdm_eb_des_ov -= einsum("ia,wij->wja", t1, x17)
    del x17
    rdm_eb_des_oo -= einsum("wij->wij", x5)
    del x5
    rdm_eb_des_ov += einsum("wia,ijab->wjb", x4, t2)
    rdm_eb_des_vv = np.zeros((nbos, nvir, nvir), dtype=types[float])
    rdm_eb_des_vv += einsum("ia,wib->wba", t1, x4)
    del x4
    rdm_eb_des_vo += einsum("wia->wai", x3)
    del x3
    x6 = np.zeros((nocc, nocc), dtype=types[float])
    x6 += einsum("ai,ja->ij", l1, t1)
    x9 = np.zeros((nocc, nocc), dtype=types[float])
    x9 += einsum("ij->ij", x6) * 2
    x16 = np.zeros((nocc, nocc), dtype=types[float])
    x16 += einsum("ij->ij", x6)
    x20 = np.zeros((nocc, nocc), dtype=types[float])
    x20 += einsum("ij->ij", x6) * 1.9999999999999987
    del x6
    x7 = np.zeros((nocc, nocc), dtype=types[float])
    x7 += einsum("wai,wja->ij", lu11, u11)
    x9 += einsum("ij->ij", x7) * 2
    x16 += einsum("ij->ij", x7)
    x20 += einsum("ij->ij", x7) * 1.9999999999999987
    del x7
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum("abij,kjab->ik", l2, t2)
    x9 += einsum("ij->ij", x8)
    x16 += einsum("ij->ij", x8) * 0.5
    rdm_eb_des_ov += einsum("ij,wia->wja", x16, u11) * -1
    del x16
    x20 += einsum("ij->ij", x8)
    del x8
    x21 += einsum("ia,ij->ja", t1, x20) * 0.5000000000000003
    del x20
    x9 += einsum("ij->ji", delta_oo) * -2
    rdm_eb_des_oo += einsum("w,ij->wji", s1, x9) * -0.5
    del x9
    x10 = np.zeros((nbos), dtype=types[float])
    x10 += einsum("w,xw->x", ls1, s2)
    x12 = np.zeros((nbos), dtype=types[float])
    x12 += einsum("w->w", x10)
    del x10
    x11 = np.zeros((nbos), dtype=types[float])
    x11 += einsum("ai,wia->w", l1, u11)
    x12 += einsum("w->w", x11)
    del x11
    rdm_eb_des_oo += einsum("w,ij->wji", x12, delta_oo)
    rdm_eb_des_ov += einsum("w,ia->wia", x12, t1)
    del x12
    x13 = np.zeros((nvir, nvir), dtype=types[float])
    x13 += einsum("wai,wib->ab", lu11, u11)
    x15 = np.zeros((nvir, nvir), dtype=types[float])
    x15 += einsum("ab->ab", x13) * 2
    x22 = np.zeros((nvir, nvir), dtype=types[float])
    x22 += einsum("ab->ab", x13)
    del x13
    x14 = np.zeros((nvir, nvir), dtype=types[float])
    x14 -= einsum("abij,ijca->bc", l2, t2)
    x15 += einsum("ab->ab", x14)
    rdm_eb_des_ov += einsum("ab,wia->wib", x15, u11) * -0.5
    del x15
    x22 += einsum("ab->ab", x14) * 0.5
    del x14
    x18 = np.zeros((nbos, nbos), dtype=types[float])
    x18 += einsum("wx,yx->wy", ls2, s2)
    x18 += einsum("wai,xia->wx", lu11, u11)
    rdm_eb_des_ov += einsum("wx,wia->xia", x18, u11)
    del x18
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x19 += einsum("ia,bajk->jkib", t1, l2)
    x21 += einsum("ijab,ijkb->ka", t2, x19) * -0.5000000000000003
    del x19
    x21 += einsum("ia->ia", t1) * -1
    x21 += einsum("w,wia->ia", ls1, u11) * -1
    x21 += einsum("ai,jiba->jb", l1, t2) * -1
    rdm_eb_des_ov += einsum("w,ia->wia", s1, x21) * -1
    del x21
    x22 += einsum("ai,ib->ab", l1, t1)
    rdm_eb_des_vv += einsum("w,ab->wab", s1, x22)
    del x22
    rdm_eb_cre_oo += einsum("w,ij->wji", ls1, delta_oo)
    rdm_eb_cre_ov += einsum("wx,xia->wia", ls2, u11)
    rdm_eb_cre_ov += einsum("wai,jiba->wjb", lu11, t2)
    rdm_eb_cre_ov += einsum("w,ia->wia", ls1, t1)
    rdm_eb_cre_vo = np.zeros((nbos, nvir, nocc), dtype=types[float])
    rdm_eb_cre_vo += einsum("wai->wai", lu11)
    rdm_eb_cre_vv = np.zeros((nbos, nvir, nvir), dtype=types[float])
    rdm_eb_cre_vv += einsum("ia,wbi->wba", t1, lu11)
    rdm_eb_des_ov += einsum("wia->wia", u11)
    rdm_eb_des_vo += einsum("w,ai->wai", s1, l1)
    rdm_eb_des_vv += einsum("ai,wib->wab", l1, u11)

    rdm_eb = np.array([
            np.block([[rdm_eb_cre_oo, rdm_eb_cre_ov], [rdm_eb_cre_vo, rdm_eb_cre_vv]]),
            np.block([[rdm_eb_des_oo, rdm_eb_des_ov], [rdm_eb_des_vo, rdm_eb_des_vv]]),
    ])

    return rdm_eb

