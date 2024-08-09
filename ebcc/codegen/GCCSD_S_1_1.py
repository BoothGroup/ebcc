# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, **kwargs):
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

def update_amps(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    # T1, T2, S1 and U11 amplitudes
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x0 += einsum("ia,jkba->ijkb", t1, v.oovv)
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum("ijka->ikja", x0) * -1
    x29 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x29 += einsum("ijab,kjlb->kila", t2, x0)
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x30 -= einsum("ijka->ikja", x29)
    del x29
    x64 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x64 += einsum("ia,jkla->jilk", t1, x0)
    del x0
    x65 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x65 += einsum("ijkl->klji", x64) * -1
    x66 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x66 += einsum("ijkl->klji", x64) * -1
    del x64
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
    x71 = np.zeros((nocc, nvir), dtype=types[float])
    x71 += einsum("ia->ia", x2)
    del x2
    x3 = np.zeros((nocc, nvir), dtype=types[float])
    x3 += einsum("ia,jiba->jb", t1, v.oovv)
    x4 += einsum("ia->ia", x3)
    x11 += einsum("ia->ia", x3)
    x70 = np.zeros((nvir, nvir), dtype=types[float])
    x70 += einsum("ia,ib->ab", t1, x3)
    del x3
    x4 += einsum("ia->ia", f.ov)
    x37 = np.zeros((nocc, nocc), dtype=types[float])
    x37 += einsum("ia,ja->ij", t1, x4)
    x38 = np.zeros((nocc, nocc), dtype=types[float])
    x38 += einsum("ij->ij", x37)
    del x37
    x51 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x51 += einsum("ia,jkab->jkib", x4, t2)
    x52 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x52 += einsum("ijka->kjia", x51)
    del x51
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
    x24 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x24 += einsum("ia,wij->wja", t1, x6)
    x25 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x25 -= einsum("wia->wia", x24)
    del x24
    t1new -= einsum("wia,wij->ja", u11, x6)
    del x6
    x7 = np.zeros((nocc, nocc), dtype=types[float])
    x7 += einsum("w,wij->ij", s1, g.boo)
    x13 = np.zeros((nocc, nocc), dtype=types[float])
    x13 += einsum("ij->ij", x7)
    x38 += einsum("ij->ji", x7)
    del x7
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum("wia,wja->ij", g.bov, u11)
    x13 += einsum("ij->ij", x8)
    x40 = np.zeros((nocc, nocc), dtype=types[float])
    x40 += einsum("ij->ij", x8)
    del x8
    x9 = np.zeros((nocc, nocc), dtype=types[float])
    x9 -= einsum("ia,ijka->jk", t1, v.ooov)
    x13 += einsum("ij->ij", x9)
    x40 += einsum("ij->ij", x9)
    del x9
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum("ij,ikab->kjab", x40, t2)
    del x40
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 -= einsum("ijab->ijba", x41)
    del x41
    x10 = np.zeros((nocc, nocc), dtype=types[float])
    x10 -= einsum("ijab,jkab->ik", t2, v.oovv)
    x13 += einsum("ij->ji", x10) * 0.5
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum("ij,kjab->kiab", x10, t2)
    del x10
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum("ijab->ijba", x44) * -0.5
    del x44
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
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x47 += einsum("ab,ijcb->ijac", x14, t2)
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum("ijab->ijab", x47)
    del x47
    x70 += einsum("ab->ab", x14) * -1
    del x14
    x15 = np.zeros((nvir, nvir), dtype=types[float])
    x15 -= einsum("ia,ibac->bc", t1, v.ovvv)
    x16 -= einsum("ab->ab", x15)
    x55 = np.zeros((nvir, nvir), dtype=types[float])
    x55 += einsum("ab->ab", x15)
    x70 += einsum("ab->ab", x15)
    del x15
    x16 += einsum("ab->ab", f.vv)
    t1new += einsum("ia,ba->ib", t1, x16)
    del x16
    x17 = np.zeros((nbos), dtype=types[float])
    x17 += einsum("w->w", G)
    x17 += einsum("ia,wia->w", t1, g.bov)
    t1new += einsum("w,wia->ia", x17, u11)
    del x17
    x18 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x18 += einsum("ia,bcda->ibcd", t1, v.vvvv)
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new -= einsum("ia,jbca->ijcb", t1, x18)
    del x18
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 += einsum("ijab,jckb->ikac", t2, v.ovov)
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 -= einsum("ijab->ijab", x19)
    del x19
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum("ia,jbca->ijbc", t1, v.ovvv)
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 -= einsum("ijab,kjcb->kiac", t2, x20)
    x32 += einsum("ijab->ijab", x21)
    del x21
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x50 -= einsum("ia,jkba->jikb", t1, x20)
    del x20
    x52 += einsum("ijka->kjia", x50)
    del x50
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x53 += einsum("ia,ijkb->jkab", t1, x52)
    del x52
    x57 += einsum("ijab->jiab", x53)
    del x53
    x22 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x22 += einsum("ia,wba->wib", t1, g.bvv)
    x25 += einsum("wia->wia", x22)
    del x22
    x23 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x23 += einsum("wia,jiba->wjb", g.bov, t2)
    x25 += einsum("wia->wia", x23)
    del x23
    x25 += einsum("wai->wia", g.bvo)
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum("wia,wjb->ijab", u11, x25)
    del x25
    x32 += einsum("ijab->jiba", x26)
    del x26
    x27 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x27 += einsum("ia,jbka->ijkb", t1, v.ovov)
    x30 += einsum("ijka->ijka", x27)
    del x27
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 -= einsum("ijab,jklb->ikla", t2, v.ooov)
    x30 += einsum("ijka->ijka", x28)
    del x28
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum("ia,jikb->jkab", t1, x30)
    del x30
    x32 += einsum("ijab->ijab", x31)
    del x31
    t2new += einsum("ijab->ijab", x32)
    t2new -= einsum("ijab->ijba", x32)
    t2new -= einsum("ijab->jiab", x32)
    t2new += einsum("ijab->jiba", x32)
    del x32
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum("ia,bcja->ijbc", t1, v.vvov)
    x42 += einsum("ijab->ijba", x33)
    del x33
    x34 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x34 += einsum("ia,jkla->ijkl", t1, v.ooov)
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x35 += einsum("ia,jkil->jkla", t1, x34)
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 -= einsum("ia,jikb->jkba", t1, x35)
    del x35
    x42 += einsum("ijab->ijba", x36)
    del x36
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum("ijab,kijl->klab", t2, x34)
    del x34
    x45 += einsum("ijab->ijba", x43) * -0.5
    del x43
    t2new += einsum("ijab->ijab", x45) * -1
    t2new += einsum("ijab->jiab", x45)
    del x45
    x38 += einsum("ij->ji", f.oo)
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum("ij,jkab->kiab", x38, t2)
    del x38
    x42 += einsum("ijab->jiba", x39)
    del x39
    t2new += einsum("ijab->ijab", x42)
    t2new -= einsum("ijab->jiab", x42)
    del x42
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum("ab,ijcb->ijac", f.vv, t2)
    t2new += einsum("ijab->jiab", x46)
    t2new -= einsum("ijab->jiba", x46)
    del x46
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum("ijab,jkbc->ikac", t2, v.oovv)
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum("ijab,kjcb->ikac", t2, x48)
    del x48
    x57 -= einsum("ijab->ijab", x49)
    del x49
    x54 = np.zeros((nvir, nvir), dtype=types[float])
    x54 += einsum("wia,wib->ab", g.bov, u11)
    x55 += einsum("ab->ba", x54)
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum("ab,ijbc->ijca", x55, t2)
    del x55
    x57 += einsum("ijab->jiab", x56)
    del x56
    t2new -= einsum("ijab->ijab", x57)
    t2new += einsum("ijab->ijba", x57)
    del x57
    x70 += einsum("ab->ba", x54)
    del x54
    x58 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x58 += einsum("ijab,kcab->ijkc", t2, v.ovvv)
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum("ia,jkib->jkab", t1, x58)
    del x58
    x62 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x62 += einsum("ijab->ijab", x59) * 0.5
    del x59
    x60 = np.zeros((nvir, nvir), dtype=types[float])
    x60 -= einsum("ijab,ijbc->ac", t2, v.oovv)
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum("ab,ijcb->ijca", x60, t2)
    x62 += einsum("ijab->ijab", x61) * 0.5
    del x61
    t2new += einsum("ijab->ijab", x62) * -1
    t2new += einsum("ijab->ijba", x62)
    del x62
    x70 += einsum("ab->ab", x60) * 0.5
    del x60
    x63 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x63 += einsum("ijab,klab->ijkl", t2, v.oovv)
    x65 += einsum("ijkl->lkji", x63) * 0.49999999999999967
    x66 += einsum("ijkl->lkji", x63) * 0.5000000000000003
    del x63
    x65 += einsum("ijkl->jilk", v.oooo) * 0.9999999999999993
    t2new += einsum("ijab,ijkl->lkba", t2, x65) * 0.5000000000000003
    del x65
    x66 += einsum("ijkl->jilk", v.oooo)
    x67 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x67 += einsum("ia,ijkl->lkja", t1, x66)
    del x66
    x67 += einsum("iajk->kjia", v.ovoo)
    t2new += einsum("ia,jkib->jkab", t1, x67)
    del x67
    x68 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x68 += einsum("wia,ijab->wjb", u11, v.oovv)
    x69 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x69 += einsum("wia->wia", x68)
    del x68
    x69 += einsum("wia->wia", gc.bov)
    x72 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x72 += einsum("ia,wja->wji", t1, x69)
    u11new += einsum("wia,ijab->wjb", x69, t2)
    del x69
    x70 += einsum("ab->ab", f.vv) * -1
    u11new += einsum("ab,wib->wia", x70, u11) * -1
    del x70
    x71 += einsum("ia->ia", f.ov)
    x72 += einsum("ia,wja->wij", x71, u11)
    del x71
    x72 += einsum("wij->wij", gc.boo)
    x72 -= einsum("wia,ijka->wjk", u11, v.ooov)
    u11new -= einsum("ia,wij->wja", t1, x72)
    del x72
    x73 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x73 += einsum("wab->wab", gc.bvv)
    x73 += einsum("wia,ibac->wbc", u11, v.ovvv)
    u11new += einsum("ia,wba->wib", t1, x73)
    del x73
    x74 = np.zeros((nbos, nbos), dtype=types[float])
    x74 += einsum("wx->wx", w)
    x74 += einsum("wia,xia->xw", g.bov, u11)
    u11new += einsum("wx,xia->wia", x74, u11)
    del x74
    t1new += einsum("wab,wib->ia", g.bvv, u11)
    t1new += einsum("ai->ia", f.vo)
    t1new += einsum("w,wai->ia", s1, g.bvo)
    t1new += einsum("ijab,jcab->ic", t2, v.ovvv) * -0.5
    t1new -= einsum("ia,ibja->jb", t1, v.ovov)
    t2new -= einsum("ia,ibjk->kjba", t1, v.ovoo)
    t2new += einsum("abij->jiba", v.vvoo)
    t2new += einsum("ijab,cdab->jidc", t2, v.vvvv) * 0.5
    s1new += einsum("w,xw->x", s1, w)
    s1new += einsum("ia,wia->w", t1, gc.bov)
    s1new += einsum("w->w", G)
    u11new += einsum("wai->wia", gc.bvo)
    u11new -= einsum("wia,ibja->wjb", u11, v.ovov)

    return {"t1new": t1new, "t2new": t2new, "s1new": s1new, "u11new": u11new}

def update_lams(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, l1=None, l2=None, ls1=None, lu11=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    # L1, L2, LS1 and LU11 amplitudes
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x0 += einsum("abij,klab->ijkl", l2, t2)
    x18 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x18 += einsum("ijkl->jilk", x0) * -1
    x112 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x112 += einsum("ijkl->jilk", x0) * -0.5
    l1new = np.zeros((nvir, nocc), dtype=types[float])
    l1new += einsum("ijka,lkij->al", v.ooov, x0) * -0.25
    del x0
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum("ia,bajk->jkib", t1, l2)
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum("ia,jkla->jkil", t1, x1)
    x18 += einsum("ijkl->ijlk", x2) * 2.0000000000000013
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x19 += einsum("ia,ijkl->jlka", t1, x18) * -0.5
    del x18
    x112 += einsum("ijkl->ijlk", x2)
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum("ijab,klij->balk", v.oovv, x112) * -0.5
    del x112
    l1new += einsum("ijka,lkji->al", v.ooov, x2) * 0.5
    del x2
    x19 += einsum("ijab,kjlb->klia", t2, x1) * 2
    x50 = np.zeros((nocc, nvir), dtype=types[float])
    x50 += einsum("ijab,ijka->kb", t2, x1)
    x54 = np.zeros((nocc, nvir), dtype=types[float])
    x54 += einsum("ia->ia", x50) * 0.5
    del x50
    l2new += einsum("iabc,jkia->cbkj", v.ovvv, x1)
    x3 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x3 -= einsum("wia,abji->wjb", u11, l2)
    x65 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x65 += einsum("ia,wja->wji", t1, x3)
    x73 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum("wia,wjb->ijab", g.bov, x3)
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum("ijab->ijab", x73)
    del x73
    l1new += einsum("wab,wia->bi", g.bvv, x3)
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 -= einsum("abij,kicb->jkac", l2, t2)
    l1new -= einsum("iabc,jiac->bj", v.ovvv, x4)
    del x4
    x5 = np.zeros((nocc, nvir), dtype=types[float])
    x5 += einsum("w,wia->ia", s1, g.bov)
    x11 = np.zeros((nocc, nvir), dtype=types[float])
    x11 += einsum("ia->ia", x5)
    x59 = np.zeros((nocc, nvir), dtype=types[float])
    x59 += einsum("ia->ia", x5)
    x80 = np.zeros((nocc, nvir), dtype=types[float])
    x80 += einsum("ia->ia", x5)
    x6 = np.zeros((nocc, nocc), dtype=types[float])
    x6 += einsum("ai,ja->ij", l1, t1)
    x52 = np.zeros((nocc, nocc), dtype=types[float])
    x52 += einsum("ij->ij", x6) * 2
    x68 = np.zeros((nocc, nocc), dtype=types[float])
    x68 += einsum("ij->ij", x6)
    x120 = np.zeros((nocc, nocc), dtype=types[float])
    x120 += einsum("ij->ij", x6)
    l1new -= einsum("ia,ji->aj", x5, x6)
    del x5
    ls1new = np.zeros((nbos), dtype=types[float])
    ls1new -= einsum("ij,wji->w", x6, g.boo)
    del x6
    x7 = np.zeros((nocc, nocc), dtype=types[float])
    x7 += einsum("abij,ikab->jk", l2, t2)
    x19 += einsum("ia,jk->jkia", t1, x7) * -1
    x52 += einsum("ij->ij", x7)
    x64 = np.zeros((nocc, nocc), dtype=types[float])
    x64 += einsum("ij->ij", x7) * 0.5
    x83 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x83 -= einsum("ij,kjab->ikab", x7, v.oovv)
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum("ijab->ijba", x83) * -0.5
    del x83
    x120 += einsum("ij->ij", x7) * 0.5
    l1new += einsum("ia,ji->aj", f.ov, x7) * -0.5
    del x7
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x8 += einsum("ia,jkba->ijkb", t1, v.oovv)
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 += einsum("ijka->ikja", x8) * -1
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x13 += einsum("ijka->kjia", x8) * 0.5000000000000003
    x78 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x78 -= einsum("ijka->ikja", x8)
    x98 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x98 -= einsum("ai,ijkb->kjab", l1, x8)
    x106 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x106 += einsum("ijab->ijab", x98)
    del x98
    x110 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x110 += einsum("ijka->kjia", x8) * 0.5
    del x8
    x9 += einsum("ijka->kjia", v.ooov)
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum("ijab,kila->lkjb", t2, x9) * 2
    x26 = np.zeros((nocc, nvir), dtype=types[float])
    x26 += einsum("ijab,kija->kb", t2, x9) * 0.5
    del x9
    x45 = np.zeros((nocc, nvir), dtype=types[float])
    x45 += einsum("ia->ia", x26)
    del x26
    x10 = np.zeros((nocc, nvir), dtype=types[float])
    x10 += einsum("ia,jiba->jb", t1, v.oovv)
    x11 += einsum("ia->ia", x10)
    x59 += einsum("ia->ia", x10)
    x69 = np.zeros((nocc, nvir), dtype=types[float])
    x69 += einsum("ia->ia", x10)
    x81 += einsum("ai,jb->ijab", l1, x10)
    x94 = np.zeros((nocc, nocc), dtype=types[float])
    x94 += einsum("ia,ja->ij", t1, x10)
    x95 = np.zeros((nocc, nocc), dtype=types[float])
    x95 += einsum("ij->ji", x94)
    del x94
    x101 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x101 -= einsum("ia,jkib->kjba", x10, x1)
    x106 -= einsum("ijab->ijab", x101)
    del x101
    x118 = np.zeros((nvir, nvir), dtype=types[float])
    x118 += einsum("ia,ib->ab", t1, x10)
    del x10
    x11 += einsum("ia->ia", f.ov)
    x17 += einsum("ia,jkab->ikjb", x11, t2) * -1
    x27 = np.zeros((nocc, nvir), dtype=types[float])
    x27 += einsum("ia,ijab->jb", x11, t2)
    x45 += einsum("ia->ia", x27) * -1
    del x27
    x35 = np.zeros((nocc, nocc), dtype=types[float])
    x35 += einsum("ia,ja->ij", t1, x11)
    x36 = np.zeros((nocc, nocc), dtype=types[float])
    x36 += einsum("ij->ji", x35)
    del x35
    lu11new = np.zeros((nbos, nvir, nocc), dtype=types[float])
    lu11new += einsum("w,ia->wai", ls1, x11) * 2
    del x11
    x12 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x12 += einsum("ijab,klab->ijkl", t2, v.oovv)
    x14 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x14 += einsum("ijkl->lkji", x12)
    x111 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x111 += einsum("ijkl->lkji", x12)
    del x12
    x13 += einsum("ijka->jika", v.ooov) * -1
    x14 += einsum("ia,jkla->kjli", t1, x13) * -4
    del x13
    x14 += einsum("ijkl->jilk", v.oooo) * 2
    x17 += einsum("ia,ijkl->jlka", t1, x14) * -0.5
    del x14
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 += einsum("ia,jbca->ijbc", t1, v.ovvv)
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum("ijab->jiab", x15) * -0.5
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 -= einsum("ijab->jiab", x15)
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum("ijab->ijab", x15)
    del x15
    x16 += einsum("iajb->ijab", v.ovov)
    x17 += einsum("ia,jkba->jikb", t1, x16) * -2
    del x16
    x17 += einsum("iajk->ikja", v.ovoo) * -1
    x17 += einsum("ijab,kcab->kjic", t2, v.ovvv) * -0.5
    l1new += einsum("abij,kija->bk", l2, x17) * 0.5
    del x17
    x19 += einsum("ai,jkba->ikjb", l1, t2) * -1
    l1new += einsum("ijab,kija->bk", v.oovv, x19) * -0.5
    del x19
    x20 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x20 += einsum("abic->ibac", v.vvov) * -1
    x20 += einsum("ia,bcda->icbd", t1, v.vvvv)
    l1new += einsum("abij,iabc->cj", l2, x20) * 0.5
    del x20
    x21 += einsum("iajb->ijab", v.ovov)
    l1new += einsum("ijka,kiab->bj", x1, x21)
    del x21
    x22 = np.zeros((nocc, nvir), dtype=types[float])
    x22 += einsum("w,wai->ia", s1, g.bvo)
    x45 += einsum("ia->ia", x22) * -1
    del x22
    x23 = np.zeros((nocc, nvir), dtype=types[float])
    x23 += einsum("wab,wib->ia", g.bvv, u11)
    x45 += einsum("ia->ia", x23) * -1
    del x23
    x24 = np.zeros((nocc, nvir), dtype=types[float])
    x24 += einsum("ia,ibja->jb", t1, v.ovov)
    x45 += einsum("ia->ia", x24)
    del x24
    x25 = np.zeros((nocc, nvir), dtype=types[float])
    x25 -= einsum("ijab,icab->jc", t2, v.ovvv)
    x45 += einsum("ia->ia", x25) * 0.5
    del x25
    x28 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x28 += einsum("ia,wja->wji", t1, g.bov)
    x29 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x29 += einsum("wij->wij", x28)
    x66 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x66 += einsum("wij->wij", x28)
    del x28
    x29 += einsum("wij->wij", g.boo)
    x30 = np.zeros((nocc, nvir), dtype=types[float])
    x30 += einsum("wia,wij->ja", u11, x29)
    del x29
    x45 += einsum("ia->ia", x30)
    del x30
    x31 = np.zeros((nocc, nocc), dtype=types[float])
    x31 += einsum("w,wij->ij", s1, g.boo)
    x36 += einsum("ij->ij", x31)
    x91 = np.zeros((nocc, nocc), dtype=types[float])
    x91 += einsum("ij->ij", x31)
    del x31
    x32 = np.zeros((nocc, nocc), dtype=types[float])
    x32 += einsum("wia,wja->ij", g.bov, u11)
    x36 += einsum("ij->ij", x32)
    x91 += einsum("ij->ij", x32)
    del x32
    x33 = np.zeros((nocc, nocc), dtype=types[float])
    x33 -= einsum("ia,ijka->jk", t1, v.ooov)
    x36 += einsum("ij->ij", x33)
    x95 += einsum("ij->ij", x33)
    del x33
    x96 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x96 += einsum("ij,abjk->kiab", x95, l2)
    del x95
    x97 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x97 -= einsum("ijab->ijba", x96)
    del x96
    x34 = np.zeros((nocc, nocc), dtype=types[float])
    x34 += einsum("ijab,ikab->jk", t2, v.oovv)
    x36 += einsum("ij->ji", x34) * 0.5
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum("ij,abki->kjab", x34, l2)
    del x34
    x84 += einsum("ijab->ijba", x82) * -0.5
    del x82
    l2new += einsum("ijab->abij", x84) * -1
    l2new += einsum("ijab->abji", x84)
    del x84
    x36 += einsum("ij->ij", f.oo)
    x37 = np.zeros((nocc, nvir), dtype=types[float])
    x37 += einsum("ia,ij->ja", t1, x36)
    x45 += einsum("ia->ia", x37)
    del x37
    l1new += einsum("ai,ji->aj", l1, x36) * -1
    lu11new += einsum("ij,waj->wai", x36, lu11) * -1
    del x36
    x38 = np.zeros((nvir, nvir), dtype=types[float])
    x38 += einsum("w,wab->ab", s1, g.bvv)
    x40 = np.zeros((nvir, nvir), dtype=types[float])
    x40 += einsum("ab->ab", x38)
    x67 = np.zeros((nvir, nvir), dtype=types[float])
    x67 += einsum("ab->ab", x38)
    x103 = np.zeros((nvir, nvir), dtype=types[float])
    x103 += einsum("ab->ab", x38)
    x118 += einsum("ab->ab", x38) * -1
    del x38
    x39 = np.zeros((nvir, nvir), dtype=types[float])
    x39 -= einsum("ia,ibac->bc", t1, v.ovvv)
    x40 += einsum("ab->ab", x39) * -1
    x67 -= einsum("ab->ab", x39)
    x100 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x100 -= einsum("ab,caij->jicb", x39, l2)
    x106 -= einsum("ijab->ijab", x100)
    del x100
    x118 += einsum("ab->ab", x39)
    del x39
    x40 += einsum("ab->ab", f.vv)
    x41 = np.zeros((nocc, nvir), dtype=types[float])
    x41 += einsum("ia,ba->ib", t1, x40)
    del x40
    x45 += einsum("ia->ia", x41) * -1
    del x41
    x42 = np.zeros((nbos), dtype=types[float])
    x42 += einsum("ia,wia->w", t1, g.bov)
    x43 = np.zeros((nbos), dtype=types[float])
    x43 += einsum("w->w", x42)
    del x42
    x43 += einsum("w->w", G)
    x44 = np.zeros((nocc, nvir), dtype=types[float])
    x44 += einsum("w,wia->ia", x43, u11)
    del x43
    x45 += einsum("ia->ia", x44) * -1
    del x44
    x45 += einsum("ai->ia", f.vo) * -1
    l1new += einsum("ia,abij->bj", x45, l2) * -1
    ls1new += einsum("ia,wai->w", x45, lu11) * -1
    del x45
    x46 = np.zeros((nocc, nvir), dtype=types[float])
    x46 += einsum("w,wia->ia", ls1, u11)
    x54 += einsum("ia->ia", x46) * -1
    del x46
    x47 = np.zeros((nocc, nvir), dtype=types[float])
    x47 += einsum("ai,jiba->jb", l1, t2)
    x54 += einsum("ia->ia", x47) * -1
    del x47
    x48 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x48 += einsum("ia,waj->wji", t1, lu11)
    x49 = np.zeros((nocc, nvir), dtype=types[float])
    x49 += einsum("wia,wij->ja", u11, x48)
    x54 += einsum("ia->ia", x49)
    del x49
    x117 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x117 -= einsum("ia,wij->wja", t1, x48)
    lu11new += einsum("wij,kjia->wak", x48, v.ooov)
    x51 = np.zeros((nocc, nocc), dtype=types[float])
    x51 += einsum("wai,wja->ij", lu11, u11)
    x52 += einsum("ij->ij", x51) * 2
    x53 = np.zeros((nocc, nvir), dtype=types[float])
    x53 += einsum("ia,ij->ja", t1, x52) * 0.5
    x54 += einsum("ia->ia", x53)
    del x53
    l1new += einsum("ij,jkia->ak", x52, v.ooov) * -0.5
    del x52
    x64 += einsum("ij->ij", x51)
    x65 += einsum("w,ij->wij", s1, x64)
    ls1new += einsum("ij,wji->w", x64, g.boo) * -1
    del x64
    x68 += einsum("ij->ij", x51)
    x93 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x93 += einsum("ij,jkab->kiab", x68, v.oovv)
    x97 += einsum("ijab->jiba", x93)
    del x93
    x120 += einsum("ij->ij", x51)
    del x51
    lu11new += einsum("ij,wja->wai", x120, g.bov) * -1
    del x120
    x54 += einsum("ia->ia", t1) * -1
    l1new += einsum("ia,ijab->bj", x54, v.oovv) * -1
    ls1new += einsum("ia,wia->w", x54, g.bov) * -1
    del x54
    x55 = np.zeros((nvir, nvir), dtype=types[float])
    x55 += einsum("ai,ib->ab", l1, t1)
    x58 = np.zeros((nvir, nvir), dtype=types[float])
    x58 += einsum("ab->ab", x55)
    ls1new += einsum("ab,wab->w", x55, g.bvv)
    del x55
    x56 = np.zeros((nvir, nvir), dtype=types[float])
    x56 += einsum("wai,wib->ab", lu11, u11)
    x58 += einsum("ab->ab", x56)
    x99 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x99 -= einsum("ab,ijcb->jiac", x56, v.oovv)
    x106 += einsum("ijab->ijab", x99)
    del x99
    x113 = np.zeros((nvir, nvir), dtype=types[float])
    x113 += einsum("ab->ab", x56) * 2
    x119 = np.zeros((nvir, nvir), dtype=types[float])
    x119 += einsum("ab->ab", x56)
    del x56
    x57 = np.zeros((nvir, nvir), dtype=types[float])
    x57 -= einsum("abij,ijbc->ac", l2, t2)
    x58 += einsum("ab->ab", x57) * 0.5
    l1new += einsum("ab,iabc->ci", x58, v.ovvv) * -1
    del x58
    x85 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x85 += einsum("ab,ijcb->jiac", x57, v.oovv)
    x88 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x88 += einsum("ijab->ijab", x85) * 0.5
    del x85
    x113 += einsum("ab->ab", x57)
    ls1new += einsum("ab,wab->w", x113, g.bvv) * 0.5
    del x113
    x119 += einsum("ab->ab", x57) * 0.5
    del x57
    lu11new += einsum("ab,wib->wai", x119, g.bov) * -1
    del x119
    x59 += einsum("ia->ia", f.ov)
    x62 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x62 += einsum("ia,wja->wij", x59, u11)
    x70 = np.zeros((nbos), dtype=types[float])
    x70 += einsum("ia,wia->w", x59, u11)
    del x59
    x60 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x60 += einsum("wia,ijab->wjb", u11, v.oovv)
    x61 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x61 += einsum("wia->wia", x60)
    x74 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x74 += einsum("wai,wjb->ijab", lu11, x60)
    del x60
    x81 += einsum("ijab->ijab", x74)
    del x74
    x61 += einsum("wia->wia", gc.bov)
    x62 += einsum("ia,wja->wji", t1, x61)
    l1new -= einsum("wij,wja->ai", x48, x61)
    del x61
    x62 += einsum("wij->wij", gc.boo)
    x62 += einsum("wia,jika->wjk", u11, v.ooov)
    l1new -= einsum("wai,wji->aj", lu11, x62)
    del x62
    x63 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x63 += einsum("wab->wab", gc.bvv)
    x63 -= einsum("wia,ibca->wbc", u11, v.ovvv)
    l1new += einsum("wai,wab->bi", lu11, x63)
    del x63
    x65 += einsum("ai,wja->wij", l1, u11)
    l1new += einsum("wia,wji->aj", g.bov, x65) * -1
    del x65
    x66 += einsum("wij->wij", g.boo)
    x116 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x116 -= einsum("ia,wij->wja", t1, x66)
    l1new -= einsum("wia,wji->aj", x3, x66)
    del x3
    lu11new -= einsum("ai,wji->waj", l1, x66)
    del x66
    x67 += einsum("ab->ab", f.vv)
    l1new += einsum("ai,ab->bi", l1, x67)
    del x67
    x69 += einsum("ia->ia", f.ov)
    l1new -= einsum("ij,ja->ai", x68, x69)
    del x68
    del x69
    x70 += einsum("w->w", G)
    x70 += einsum("w,xw->x", s1, w)
    x70 += einsum("ia,wia->w", t1, gc.bov)
    l1new += einsum("w,wai->ai", x70, lu11)
    del x70
    x71 = np.zeros((nbos), dtype=types[float])
    x71 += einsum("w->w", s1)
    x71 += einsum("ai,wia->w", l1, u11)
    l1new += einsum("w,wia->ai", x71, g.bov)
    del x71
    x72 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x72 += einsum("wia,wbj->ijab", gc.bov, lu11)
    x81 += einsum("ijab->ijab", x72)
    del x72
    x75 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x75 -= einsum("ijab,ikca->jkbc", t2, v.oovv)
    x76 += einsum("ijab->ijab", x75)
    del x75
    x76 -= einsum("iajb->jiab", v.ovov)
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum("abij,ikac->jkbc", l2, x76)
    del x76
    x81 += einsum("ijab->ijab", x77)
    del x77
    x78 += einsum("ijka->kjia", v.ooov)
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum("ijka,iklb->jlab", x1, x78)
    del x78
    x81 -= einsum("ijab->ijab", x79)
    del x79
    x80 += einsum("ia->ia", f.ov)
    x81 += einsum("ai,jb->jiba", l1, x80)
    l2new += einsum("ijab->abij", x81)
    l2new -= einsum("ijab->baij", x81)
    l2new -= einsum("ijab->abji", x81)
    l2new += einsum("ijab->baji", x81)
    del x81
    x90 = np.zeros((nocc, nocc), dtype=types[float])
    x90 += einsum("ia,ja->ij", t1, x80)
    x91 += einsum("ij->ji", x90)
    del x90
    x105 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x105 += einsum("ia,jkib->jkba", x80, x1)
    del x1
    x106 -= einsum("ijab->jiba", x105)
    del x105
    lu11new -= einsum("ia,wji->waj", x80, x48)
    del x48
    del x80
    x86 = np.zeros((nvir, nvir), dtype=types[float])
    x86 += einsum("ijab,ijac->bc", t2, v.oovv)
    x87 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x87 -= einsum("ab,caij->jicb", x86, l2)
    x88 += einsum("ijab->ijab", x87) * 0.5
    del x87
    l2new += einsum("ijab->abij", x88) * -1
    l2new += einsum("ijab->baij", x88)
    del x88
    x118 += einsum("ab->ab", x86) * 0.5
    del x86
    x89 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x89 += einsum("ai,jabc->ijbc", l1, v.ovvv)
    x97 += einsum("ijab->ijba", x89)
    del x89
    x91 += einsum("ij->ij", f.oo)
    x92 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x92 += einsum("ij,abjk->kiab", x91, l2)
    del x91
    x97 += einsum("ijab->jiba", x92)
    del x92
    l2new += einsum("ijab->abij", x97)
    l2new -= einsum("ijab->abji", x97)
    del x97
    x102 = np.zeros((nvir, nvir), dtype=types[float])
    x102 += einsum("wia,wib->ab", g.bov, u11)
    x103 -= einsum("ab->ba", x102)
    x104 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x104 += einsum("ab,acij->ijcb", x103, l2)
    del x103
    x106 -= einsum("ijab->jiba", x104)
    del x104
    l2new += einsum("ijab->abij", x106)
    l2new -= einsum("ijab->baij", x106)
    del x106
    x118 += einsum("ab->ba", x102)
    del x102
    x107 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x107 += einsum("ai,jkib->jkab", l1, v.ooov)
    x109 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x109 -= einsum("ijab->jiab", x107)
    del x107
    x108 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x108 += einsum("ab,caij->ijbc", f.vv, l2)
    x109 -= einsum("ijab->jiab", x108)
    del x108
    l2new -= einsum("ijab->abij", x109)
    l2new += einsum("ijab->baij", x109)
    del x109
    x110 += einsum("ijka->jika", v.ooov) * -1
    x111 += einsum("ia,jkla->kjli", t1, x110) * -4
    del x110
    x111 += einsum("ijkl->jilk", v.oooo) * 2
    l2new += einsum("abij,klij->balk", l2, x111) * 0.25
    del x111
    x114 = np.zeros((nbos, nbos), dtype=types[float])
    x114 += einsum("wai,xia->wx", lu11, u11)
    lu11new += einsum("wx,xia->wai", x114, g.bov)
    del x114
    x115 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x115 += einsum("ia,wbi->wba", t1, lu11)
    lu11new += einsum("wab,iacb->wci", x115, v.ovvv)
    del x115
    x116 += einsum("wai->wia", g.bvo)
    x116 += einsum("ia,wba->wib", t1, g.bvv)
    x116 += einsum("wia,jiba->wjb", g.bov, t2)
    lu11new += einsum("wia,abij->wbj", x116, l2)
    del x116
    x117 += einsum("wai,ijab->wjb", lu11, t2)
    lu11new += einsum("wia,ijab->wbj", x117, v.oovv)
    del x117
    x118 += einsum("ab->ab", f.vv) * -1
    lu11new += einsum("ab,wai->wbi", x118, lu11) * -1
    del x118
    x121 = np.zeros((nbos, nbos), dtype=types[float])
    x121 += einsum("wx->wx", w)
    x121 += einsum("wia,xia->xw", g.bov, u11)
    lu11new += einsum("wx,wai->xai", x121, lu11)
    del x121
    l1new -= einsum("ai,jaib->bj", l1, v.ovov)
    l1new += einsum("ia->ai", f.ov)
    l1new += einsum("w,wia->ai", ls1, gc.bov)
    l2new += einsum("abij,abcd->dcji", l2, v.vvvv) * 0.5
    l2new += einsum("ijab->baji", v.oovv)
    ls1new += einsum("ai,wai->w", l1, g.bvo)
    ls1new += einsum("w->w", G)
    ls1new += einsum("w,wx->x", ls1, w)
    lu11new += einsum("wia->wai", g.bov)
    lu11new -= einsum("wai,jaib->wbj", lu11, v.ovov)
    lu11new += einsum("ai,wab->wbi", l1, g.bvv)

    return {"l1new": l1new, "l2new": l2new, "ls1new": ls1new, "lu11new": lu11new}

def make_rdm1_f(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, l1=None, l2=None, ls1=None, lu11=None, **kwargs):
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
    rdm1_f_vo += einsum("w,wia->ai", ls1, u11)
    rdm1_f_vo += einsum("ia->ai", t1)
    rdm1_f_vo += einsum("ai,jiba->bj", l1, t2)
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=types[float])
    rdm1_f_vv += einsum("ai,ib->ba", l1, t1)
    rdm1_f_vv += einsum("wai,wib->ba", lu11, u11)
    rdm1_f_vv += einsum("abij,ijca->cb", l2, t2) * -0.5

    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])

    return rdm1_f

def make_rdm2_f(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, l1=None, l2=None, ls1=None, lu11=None, **kwargs):
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
    x48 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x48 += einsum("ijkl->jilk", x0) * -1
    x49 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x49 += einsum("ijkl->jilk", x0) * -1
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
    x48 += einsum("ijkl->ijlk", x2) * 2.0000000000000013
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vvoo += einsum("ijab,ijkl->balk", t2, x48) * -0.25
    del x48
    x49 += einsum("ijkl->ijlk", x2) * 1.9999999999999987
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x50 += einsum("ia,ijkl->jlka", t1, x49) * -1
    del x49
    rdm2_f_vvoo += einsum("ia,ijkb->abkj", t1, x50) * -0.5000000000000003
    del x50
    rdm2_f_oooo -= einsum("ijkl->ijlk", x2)
    del x2
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 -= einsum("ijab,kjlb->klia", t2, x1)
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum("ijka->ijka", x9)
    x25 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x25 -= einsum("ijka->ijka", x9)
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum("ia,ijkb->jkab", t1, x9)
    del x9
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 -= einsum("ijab->ijab", x33)
    del x33
    x16 = np.zeros((nocc, nvir), dtype=types[float])
    x16 -= einsum("ijab,ijkb->ka", t2, x1)
    x18 = np.zeros((nocc, nvir), dtype=types[float])
    x18 += einsum("ia->ia", x16)
    del x16
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum("ia,jikb->jkba", t1, x1)
    x56 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x56 -= einsum("ia,ijbc->jbca", t1, x28)
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov -= einsum("iabc->cbia", x56)
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo += einsum("iabc->cbai", x56)
    del x56
    rdm2_f_ovov = np.zeros((nocc, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_ovov += einsum("ijab->ibja", x28)
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_ovvo -= einsum("ijab->ibaj", x28)
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_voov -= einsum("ijab->bija", x28)
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_vovo += einsum("ijab->biaj", x28)
    del x28
    x55 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x55 += einsum("ijab,ijkc->kcab", t2, x1)
    rdm2_f_vvov += einsum("iabc->bcia", x55) * 0.5
    rdm2_f_vvvo += einsum("iabc->bcai", x55) * -0.5
    del x55
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov -= einsum("ijka->jika", x1)
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo += einsum("ijka->jiak", x1)
    del x1
    x3 = np.zeros((nocc, nocc), dtype=types[float])
    x3 += einsum("ai,ja->ij", l1, t1)
    x14 = np.zeros((nocc, nocc), dtype=types[float])
    x14 += einsum("ij->ij", x3)
    x23 = np.zeros((nocc, nvir), dtype=types[float])
    x23 += einsum("ia,ij->ja", t1, x3)
    x24 = np.zeros((nocc, nvir), dtype=types[float])
    x24 -= einsum("ia->ia", x23)
    del x23
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 += einsum("ij,kiab->jkab", x3, t2)
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 -= einsum("ijab->ijba", x43)
    del x43
    rdm2_f_oooo -= einsum("ij,kl->jkil", delta_oo, x3)
    rdm2_f_oooo += einsum("ij,kl->kjil", delta_oo, x3)
    rdm2_f_oooo += einsum("ij,kl->jkli", delta_oo, x3)
    rdm2_f_oooo -= einsum("ij,kl->kjli", delta_oo, x3)
    del x3
    x4 = np.zeros((nocc, nocc), dtype=types[float])
    x4 += einsum("abij,kjab->ik", l2, t2)
    x17 = np.zeros((nocc, nvir), dtype=types[float])
    x17 += einsum("ia,ij->ja", t1, x4)
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
    x19 += einsum("ia,jk->jika", t1, x4) * -0.5
    rdm2_f_ovoo += einsum("ijka->iajk", x19) * -1
    rdm2_f_ovoo += einsum("ijka->iakj", x19)
    del x19
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x47 += einsum("ij,kiab->kjab", x4, t2)
    rdm2_f_vvoo += einsum("ijab->abij", x47) * -0.5
    rdm2_f_vvoo += einsum("ijab->abji", x47) * 0.5
    del x47
    rdm2_f_oooo += einsum("ij,kl->ikjl", delta_oo, x4) * -0.5
    rdm2_f_oooo += einsum("ij,kl->iklj", delta_oo, x4) * 0.5
    rdm2_f_oooo += einsum("ij,kl->kjil", delta_oo, x4) * 0.5
    rdm2_f_oooo += einsum("ij,kl->kjli", delta_oo, x4) * -0.5
    rdm2_f_vooo += einsum("ia,jk->ajik", t1, x4) * -0.5
    rdm2_f_vooo += einsum("ia,jk->ajki", t1, x4) * 0.5
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
    x51 = np.zeros((nocc, nvir), dtype=types[float])
    x51 += einsum("ia,ij->ja", t1, x14)
    del x14
    x52 = np.zeros((nocc, nvir), dtype=types[float])
    x52 += einsum("ia->ia", x51)
    x53 = np.zeros((nocc, nvir), dtype=types[float])
    x53 += einsum("ia->ia", x51)
    del x51
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum("ij,kiab->kjab", x5, t2)
    x45 += einsum("ijab->ijba", x44)
    del x44
    rdm2_f_vvoo -= einsum("ijab->baij", x45)
    rdm2_f_vvoo += einsum("ijab->baji", x45)
    del x45
    rdm2_f_oooo -= einsum("ij,kl->ikjl", delta_oo, x5)
    rdm2_f_oooo += einsum("ij,kl->iklj", delta_oo, x5)
    rdm2_f_oooo += einsum("ij,kl->kjil", delta_oo, x5)
    rdm2_f_oooo -= einsum("ij,kl->kjli", delta_oo, x5)
    del x5
    x6 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x6 += einsum("ai,jkba->ijkb", l1, t2)
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum("ia,ijkb->jkab", t1, x6)
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum("ijab->ijab", x39)
    del x39
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
    x52 += einsum("ia->ia", x11)
    x53 += einsum("ia->ia", x11)
    del x11
    x35 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x35 += einsum("ia,wij->wja", t1, x7)
    del x7
    x36 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x36 -= einsum("wia->wia", x35)
    del x35
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
    x38 += einsum("ia,jb->ijab", t1, x10)
    del x10
    x22 = np.zeros((nocc, nvir), dtype=types[float])
    x22 += einsum("w,wia->ia", ls1, u11)
    x24 += einsum("ia->ia", x22)
    x52 -= einsum("ia->ia", x22)
    x53 -= einsum("ia->ia", x22)
    del x22
    rdm2_f_vvoo += einsum("ia,jb->abji", t1, x53)
    rdm2_f_vvoo -= einsum("ia,jb->baji", t1, x53)
    del x53
    x24 += einsum("ia->ia", t1)
    rdm2_f_ovoo -= einsum("ij,ka->jaki", delta_oo, x24)
    rdm2_f_ovoo += einsum("ij,ka->jaik", delta_oo, x24)
    rdm2_f_vooo -= einsum("ij,ka->ajik", delta_oo, x24)
    rdm2_f_vooo += einsum("ij,ka->ajki", delta_oo, x24)
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
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum("ijab,jkbc->ikac", t2, x27)
    x42 += einsum("ijab->ijab", x41)
    del x41
    x60 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x60 += einsum("ia,ijbc->jbac", t1, x27)
    x62 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x62 -= einsum("iabc->iabc", x60)
    del x60
    rdm2_f_ovov -= einsum("ijab->ibja", x27)
    rdm2_f_ovvo += einsum("ijab->ibaj", x27)
    rdm2_f_voov += einsum("ijab->bija", x27)
    rdm2_f_vovo -= einsum("ijab->biaj", x27)
    del x27
    x29 = np.zeros((nvir, nvir), dtype=types[float])
    x29 += einsum("ai,ib->ab", l1, t1)
    x32 = np.zeros((nvir, nvir), dtype=types[float])
    x32 += einsum("ab->ab", x29)
    x61 = np.zeros((nvir, nvir), dtype=types[float])
    x61 += einsum("ab->ab", x29)
    del x29
    x30 = np.zeros((nvir, nvir), dtype=types[float])
    x30 += einsum("wai,wib->ab", lu11, u11)
    x32 += einsum("ab->ab", x30)
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum("ab,ijca->ijcb", x30, t2)
    x42 -= einsum("ijab->ijab", x40)
    del x40
    rdm2_f_vvoo -= einsum("ijab->abji", x42)
    rdm2_f_vvoo += einsum("ijab->baji", x42)
    del x42
    x61 += einsum("ab->ab", x30)
    del x30
    x62 += einsum("ia,bc->ibac", t1, x61)
    del x61
    x31 = np.zeros((nvir, nvir), dtype=types[float])
    x31 -= einsum("abij,ijca->bc", l2, t2)
    x32 += einsum("ab->ab", x31) * 0.5
    rdm2_f_ovov += einsum("ij,ab->jbia", delta_oo, x32)
    rdm2_f_ovvo += einsum("ij,ab->jbai", delta_oo, x32) * -1
    rdm2_f_voov += einsum("ij,ab->bjia", delta_oo, x32) * -1
    rdm2_f_vovo += einsum("ij,ab->bjai", delta_oo, x32)
    del x32
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum("ab,ijca->ijcb", x31, t2)
    rdm2_f_vvoo += einsum("ijab->abji", x46) * 0.5
    rdm2_f_vvoo += einsum("ijab->baji", x46) * -0.5
    del x46
    rdm2_f_vvov += einsum("ia,bc->acib", t1, x31) * 0.5
    rdm2_f_vvov += einsum("ia,bc->caib", t1, x31) * -0.5
    rdm2_f_vvvo += einsum("ia,bc->acbi", t1, x31) * -0.5
    rdm2_f_vvvo += einsum("ia,bc->cabi", t1, x31) * 0.5
    del x31
    x34 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x34 += einsum("wai,jiba->wjb", lu11, t2)
    x36 += einsum("wia->wia", x34)
    del x34
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum("wia,wjb->jiba", u11, x36)
    del x36
    x38 += einsum("ijab->ijab", x37)
    del x37
    rdm2_f_vvoo += einsum("ijab->abij", x38)
    rdm2_f_vvoo -= einsum("ijab->baij", x38)
    rdm2_f_vvoo -= einsum("ijab->abji", x38)
    rdm2_f_vvoo += einsum("ijab->baji", x38)
    del x38
    x52 -= einsum("ia->ia", t1)
    rdm2_f_vvoo += einsum("ia,jb->baij", t1, x52)
    rdm2_f_vvoo -= einsum("ia,jb->abij", t1, x52)
    del x52
    x54 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x54 += einsum("ia,bcji->jbca", t1, l2)
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv -= einsum("iabc->icba", x54)
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv += einsum("iabc->ciba", x54)
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv -= einsum("ia,ibcd->adcb", t1, x54)
    del x54
    x57 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x57 += einsum("ai,jibc->jabc", l1, t2)
    rdm2_f_vvov -= einsum("iabc->cbia", x57)
    rdm2_f_vvvo += einsum("iabc->cbai", x57)
    del x57
    x58 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x58 += einsum("ia,wbi->wba", t1, lu11)
    x59 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x59 += einsum("wia,wbc->ibca", u11, x58)
    del x58
    x62 -= einsum("iabc->iabc", x59)
    del x59
    rdm2_f_vvov += einsum("iabc->bcia", x62)
    rdm2_f_vvov -= einsum("iabc->cbia", x62)
    rdm2_f_vvvo -= einsum("iabc->bcai", x62)
    rdm2_f_vvvo += einsum("iabc->cbai", x62)
    del x62
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

def make_sing_b_dm(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, l1=None, l2=None, ls1=None, lu11=None, **kwargs):
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
    dm_b_des += einsum("ai,wia->w", l1, u11)
    dm_b_des += einsum("w->w", s1)

    dm_b = np.array([dm_b_cre, dm_b_des])

    return dm_b

def make_rdm1_b(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, l1=None, l2=None, ls1=None, lu11=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    # Boson 1RDM
    rdm1_b = np.zeros((nbos, nbos), dtype=types[float])
    rdm1_b += einsum("wai,xia->wx", lu11, u11)
    rdm1_b += einsum("w,x->wx", ls1, s1)

    return rdm1_b

def make_eb_coup_rdm(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, l1=None, l2=None, ls1=None, lu11=None, **kwargs):
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
    x17 = np.zeros((nocc, nvir), dtype=types[float])
    x17 += einsum("wia,wij->ja", u11, x0)
    rdm_eb_cre_oo = np.zeros((nbos, nocc, nocc), dtype=types[float])
    rdm_eb_cre_oo -= einsum("wij->wji", x0)
    rdm_eb_cre_ov = np.zeros((nbos, nocc, nvir), dtype=types[float])
    rdm_eb_cre_ov -= einsum("ia,wij->wja", t1, x0)
    del x0
    x1 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x1 += einsum("ai,wja->wij", l1, u11)
    x14 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x14 += einsum("wij->wij", x1)
    rdm_eb_des_oo = np.zeros((nbos, nocc, nocc), dtype=types[float])
    rdm_eb_des_oo -= einsum("wij->wji", x1)
    del x1
    x2 = np.zeros((nbos), dtype=types[float])
    x2 += einsum("ai,wia->w", l1, u11)
    rdm_eb_des_oo += einsum("w,ij->wji", x2, delta_oo)
    rdm_eb_des_ov = np.zeros((nbos, nocc, nvir), dtype=types[float])
    rdm_eb_des_ov += einsum("w,ia->wia", x2, t1)
    del x2
    x3 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x3 -= einsum("wia,abji->wjb", u11, l2)
    x4 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x4 += einsum("ia,wja->wji", t1, x3)
    x14 += einsum("wij->wij", x4)
    rdm_eb_des_ov -= einsum("ia,wij->wja", t1, x14)
    del x14
    rdm_eb_des_oo -= einsum("wij->wji", x4)
    del x4
    rdm_eb_des_ov += einsum("wia,jiba->wjb", x3, t2)
    rdm_eb_des_vo = np.zeros((nbos, nvir, nocc), dtype=types[float])
    rdm_eb_des_vo += einsum("wia->wai", x3)
    rdm_eb_des_vv = np.zeros((nbos, nvir, nvir), dtype=types[float])
    rdm_eb_des_vv += einsum("ia,wib->wba", t1, x3)
    del x3
    x5 = np.zeros((nocc, nocc), dtype=types[float])
    x5 += einsum("ai,ja->ij", l1, t1)
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum("ij->ij", x5)
    x13 = np.zeros((nocc, nocc), dtype=types[float])
    x13 += einsum("ij->ij", x5)
    x16 = np.zeros((nocc, nocc), dtype=types[float])
    x16 += einsum("ij->ij", x5) * 1.9999999999999987
    del x5
    x6 = np.zeros((nocc, nocc), dtype=types[float])
    x6 += einsum("wai,wja->ij", lu11, u11)
    x8 += einsum("ij->ij", x6)
    x13 += einsum("ij->ij", x6)
    x16 += einsum("ij->ij", x6) * 1.9999999999999987
    del x6
    x7 = np.zeros((nocc, nocc), dtype=types[float])
    x7 += einsum("abij,kjab->ik", l2, t2)
    x8 += einsum("ij->ij", x7) * 0.5
    x13 += einsum("ij->ij", x7) * 0.5
    rdm_eb_des_ov += einsum("ij,wia->wja", x13, u11) * -1
    del x13
    x16 += einsum("ij->ij", x7)
    del x7
    x17 += einsum("ia,ij->ja", t1, x16) * 0.5000000000000003
    del x16
    x8 += einsum("ij->ji", delta_oo) * -1
    rdm_eb_des_oo += einsum("w,ij->wji", s1, x8) * -1
    del x8
    x9 = np.zeros((nbos, nbos), dtype=types[float])
    x9 += einsum("wai,xia->wx", lu11, u11)
    rdm_eb_des_ov += einsum("wx,wia->xia", x9, u11)
    del x9
    x10 = np.zeros((nvir, nvir), dtype=types[float])
    x10 += einsum("wai,wib->ab", lu11, u11)
    x12 = np.zeros((nvir, nvir), dtype=types[float])
    x12 += einsum("ab->ab", x10) * 2
    x18 = np.zeros((nvir, nvir), dtype=types[float])
    x18 += einsum("ab->ab", x10)
    del x10
    x11 = np.zeros((nvir, nvir), dtype=types[float])
    x11 -= einsum("abij,ijca->bc", l2, t2)
    x12 += einsum("ab->ab", x11)
    rdm_eb_des_ov += einsum("ab,wia->wib", x12, u11) * -0.5
    del x12
    x18 += einsum("ab->ab", x11) * 0.5
    del x11
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum("ia,bajk->jkib", t1, l2)
    x17 += einsum("ijab,ijkb->ka", t2, x15) * -0.5000000000000003
    del x15
    x17 += einsum("ia->ia", t1) * -1
    x17 += einsum("w,wia->ia", ls1, u11) * -1
    x17 += einsum("ai,jiba->jb", l1, t2) * -1
    rdm_eb_des_ov += einsum("w,ia->wia", s1, x17) * -1
    del x17
    x18 += einsum("ai,ib->ab", l1, t1)
    rdm_eb_des_vv += einsum("w,ab->wab", s1, x18)
    del x18
    rdm_eb_cre_oo += einsum("w,ij->wji", ls1, delta_oo)
    rdm_eb_cre_ov += einsum("w,ia->wia", ls1, t1)
    rdm_eb_cre_ov += einsum("wai,jiba->wjb", lu11, t2)
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

