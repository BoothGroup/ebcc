# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, **kwargs):
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

def update_amps(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    # T1, T2, S1, S2, U11 and U12 amplitudes
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x0 += einsum("ia,jkba->ijkb", t1, v.oovv)
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum("ijka->ikja", x0) * -1
    x40 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x40 += einsum("ijab,kjlb->kila", t2, x0)
    x41 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x41 -= einsum("ijka->ikja", x40)
    del x40
    x65 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x65 += einsum("ia,jkla->jilk", t1, x0)
    x66 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x66 += einsum("ijkl->klji", x65) * -2.0000000000000013
    x67 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x67 += einsum("ijkl->klji", x65) * -1
    del x65
    x110 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x110 += einsum("ijka->kjia", x0)
    del x0
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
    x81 = np.zeros((nocc, nvir), dtype=types[float])
    x81 += einsum("ia->ia", x2)
    del x2
    x3 = np.zeros((nocc, nvir), dtype=types[float])
    x3 += einsum("ia,jiba->jb", t1, v.oovv)
    x4 += einsum("ia->ia", x3)
    x11 += einsum("ia->ia", x3)
    x77 = np.zeros((nvir, nvir), dtype=types[float])
    x77 += einsum("ia,ib->ab", t1, x3) * 2
    del x3
    x4 += einsum("ia->ia", f.ov)
    x24 = np.zeros((nocc, nocc), dtype=types[float])
    x24 += einsum("ia,ja->ij", t1, x4)
    x25 = np.zeros((nocc, nocc), dtype=types[float])
    x25 += einsum("ij->ij", x24)
    del x24
    x48 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x48 += einsum("ia,jkab->jkib", x4, t2)
    x49 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x49 += einsum("ijka->kjia", x48)
    del x48
    x91 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x91 += einsum("ia,wja->wji", x4, u11)
    x94 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x94 += einsum("wij->wji", x91)
    del x91
    x111 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x111 += einsum("ia,wxja->xwij", x4, u12)
    t1new += einsum("ia,ijab->jb", x4, t2)
    s1new = np.zeros((nbos), dtype=types[float])
    s1new += einsum("ia,wia->w", x4, u11)
    s2new = np.zeros((nbos, nbos), dtype=types[float])
    s2new += einsum("ia,wxia->xw", x4, u12)
    del x4
    x5 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x5 += einsum("ia,wja->wji", t1, g.bov)
    x6 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x6 += einsum("wij->wij", x5)
    del x5
    x6 += einsum("wij->wij", g.boo)
    x35 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x35 += einsum("ia,wij->wja", t1, x6)
    x36 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x36 -= einsum("wia->wia", x35)
    del x35
    t1new -= einsum("wia,wij->ja", u11, x6)
    u11new = np.zeros((nbos, nocc, nvir), dtype=types[float])
    u11new -= einsum("wij,wxia->xja", x6, u12)
    del x6
    x7 = np.zeros((nocc, nocc), dtype=types[float])
    x7 += einsum("w,wij->ij", s1, g.boo)
    x13 = np.zeros((nocc, nocc), dtype=types[float])
    x13 += einsum("ij->ij", x7)
    x25 += einsum("ij->ji", x7)
    x83 = np.zeros((nocc, nocc), dtype=types[float])
    x83 += einsum("ij->ij", x7) * 2
    del x7
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum("wia,wja->ij", g.bov, u11)
    x13 += einsum("ij->ij", x8)
    x27 = np.zeros((nocc, nocc), dtype=types[float])
    x27 += einsum("ij->ij", x8)
    x83 += einsum("ij->ij", x8) * 2
    del x8
    x9 = np.zeros((nocc, nocc), dtype=types[float])
    x9 -= einsum("ia,ijka->jk", t1, v.ooov)
    x13 += einsum("ij->ij", x9)
    x27 += einsum("ij->ij", x9)
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum("ij,ikab->kjab", x27, t2)
    del x27
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 -= einsum("ijab->ijba", x28)
    del x28
    x83 += einsum("ij->ij", x9) * 2
    del x9
    x10 = np.zeros((nocc, nocc), dtype=types[float])
    x10 -= einsum("ijab,jkab->ik", t2, v.oovv)
    x13 += einsum("ij->ji", x10) * 0.5
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum("ij,kjab->kiab", x10, t2)
    x62 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x62 += einsum("ijab->ijba", x61) * -0.5
    del x61
    x83 += einsum("ij->ji", x10)
    del x10
    x11 += einsum("ia->ia", f.ov)
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum("ia,ja->ij", t1, x11)
    x13 += einsum("ij->ji", x12)
    del x12
    x83 += einsum("ia,ja->ji", t1, x11) * 2
    del x11
    x13 += einsum("ij->ij", f.oo)
    t1new += einsum("ia,ij->ja", t1, x13) * -1
    u12new = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    u12new += einsum("ij,wxia->xwja", x13, u12) * -1
    del x13
    x14 = np.zeros((nvir, nvir), dtype=types[float])
    x14 += einsum("w,wab->ab", s1, g.bvv)
    x16 = np.zeros((nvir, nvir), dtype=types[float])
    x16 += einsum("ab->ab", x14)
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum("ab,ijcb->ijac", x14, t2)
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum("ijab->ijab", x44)
    del x44
    x77 += einsum("ab->ab", x14) * -2
    x109 = np.zeros((nvir, nvir), dtype=types[float])
    x109 += einsum("ab->ab", x14) * -2
    del x14
    x15 = np.zeros((nvir, nvir), dtype=types[float])
    x15 -= einsum("ia,ibac->bc", t1, v.ovvv)
    x16 -= einsum("ab->ab", x15)
    x52 = np.zeros((nvir, nvir), dtype=types[float])
    x52 += einsum("ab->ab", x15)
    x77 += einsum("ab->ab", x15) * 2
    x109 += einsum("ab->ab", x15) * 2
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
    u11new += einsum("w,wxia->xia", x18, u12)
    del x18
    x19 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x19 += einsum("ia,bcda->ibcd", t1, v.vvvv)
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new -= einsum("ia,jbca->ijcb", t1, x19)
    del x19
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum("ia,bcja->ijbc", t1, v.vvov)
    x29 += einsum("ijab->ijba", x20)
    del x20
    x21 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x21 += einsum("ia,jkla->ijkl", t1, v.ooov)
    x22 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x22 += einsum("ia,jkil->jkla", t1, x21)
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x23 -= einsum("ia,jikb->jkba", t1, x22)
    del x22
    x29 += einsum("ijab->ijba", x23)
    del x23
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum("ijab,kijl->klab", t2, x21)
    del x21
    x62 += einsum("ijab->ijba", x60) * -0.5
    del x60
    t2new += einsum("ijab->ijab", x62) * -1
    t2new += einsum("ijab->jiab", x62)
    del x62
    x25 += einsum("ij->ji", f.oo)
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum("ij,jkab->kiab", x25, t2)
    del x25
    x29 += einsum("ijab->jiba", x26)
    del x26
    t2new += einsum("ijab->ijab", x29)
    t2new -= einsum("ijab->jiab", x29)
    del x29
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum("ijab,jckb->ikac", t2, v.ovov)
    x43 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x43 -= einsum("ijab->ijab", x30)
    del x30
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum("ia,jbca->ijbc", t1, v.ovvv)
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 -= einsum("ijab,kjcb->kiac", t2, x31)
    x43 += einsum("ijab->ijab", x32)
    del x32
    x47 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x47 -= einsum("ia,jkba->jikb", t1, x31)
    x49 += einsum("ijka->kjia", x47)
    del x47
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum("ia,ijkb->jkab", t1, x49)
    del x49
    x54 += einsum("ijab->jiab", x50)
    del x50
    x108 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x108 -= einsum("ijab->jiab", x31)
    del x31
    x33 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x33 += einsum("ia,wba->wib", t1, g.bvv)
    x36 += einsum("wia->wia", x33)
    del x33
    x34 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x34 += einsum("wia,jiba->wjb", g.bov, t2)
    x36 += einsum("wia->wia", x34)
    del x34
    x36 += einsum("wai->wia", g.bvo)
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum("wia,wjb->ijab", u11, x36)
    del x36
    x43 += einsum("ijab->jiba", x37)
    del x37
    x38 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x38 += einsum("ia,jbka->ijkb", t1, v.ovov)
    x41 += einsum("ijka->ijka", x38)
    del x38
    x39 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x39 -= einsum("ijab,jklb->ikla", t2, v.ooov)
    x41 += einsum("ijka->ijka", x39)
    del x39
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum("ia,jikb->jkab", t1, x41)
    del x41
    x43 += einsum("ijab->ijab", x42)
    del x42
    t2new += einsum("ijab->ijab", x43)
    t2new -= einsum("ijab->ijba", x43)
    t2new -= einsum("ijab->jiab", x43)
    t2new += einsum("ijab->jiba", x43)
    del x43
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum("ijab,jkbc->ikac", t2, v.oovv)
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x46 += einsum("ijab,kjcb->kica", t2, x45)
    del x45
    x54 -= einsum("ijab->ijab", x46)
    del x46
    x51 = np.zeros((nvir, nvir), dtype=types[float])
    x51 += einsum("wia,wib->ab", g.bov, u11)
    x52 += einsum("ab->ba", x51)
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x53 += einsum("ab,ijbc->ijca", x52, t2)
    del x52
    x54 += einsum("ijab->jiab", x53)
    del x53
    t2new -= einsum("ijab->ijab", x54)
    t2new += einsum("ijab->ijba", x54)
    del x54
    x77 += einsum("ab->ba", x51) * 2
    x109 += einsum("ab->ba", x51) * 2
    del x51
    x55 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x55 += einsum("ijab,kcab->ijkc", t2, v.ovvv)
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum("ia,jkib->jkab", t1, x55)
    del x55
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum("ijab->ijab", x56) * 0.5
    del x56
    x57 = np.zeros((nvir, nvir), dtype=types[float])
    x57 -= einsum("ijab,ijbc->ac", t2, v.oovv)
    x58 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x58 += einsum("ab,ijcb->ijca", x57, t2)
    x59 += einsum("ijab->ijab", x58) * 0.5
    del x58
    t2new += einsum("ijab->ijab", x59) * -1
    t2new += einsum("ijab->ijba", x59)
    del x59
    x77 += einsum("ab->ab", x57)
    x109 += einsum("ab->ab", x57)
    del x57
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum("ab,ijcb->ijac", f.vv, t2)
    t2new += einsum("ijab->jiab", x63)
    t2new -= einsum("ijab->jiba", x63)
    del x63
    x64 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x64 += einsum("ijab,klab->ijkl", t2, v.oovv)
    x66 += einsum("ijkl->lkji", x64)
    x67 += einsum("ijkl->lkji", x64) * 0.5000000000000003
    del x64
    x66 += einsum("ijkl->jilk", v.oooo) * 2
    t2new += einsum("ijab,ijkl->lkba", t2, x66) * 0.25
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
    x97 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x97 += einsum("ia,wib->wab", t1, x69)
    x98 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x98 += einsum("wab->wba", x97)
    del x97
    x102 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x102 += einsum("ia,wja->wij", t1, x69)
    x103 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x103 += einsum("wij->wji", x102)
    del x102
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
    x90 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x90 += einsum("wx,ywia->xyia", x71, u12)
    del x71
    x107 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x107 -= einsum("wxia->wxia", x90)
    del x90
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
    x92 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x92 += einsum("wia->wia", x75)
    del x75
    x76 += einsum("wia->wia", gc.bov)
    x82 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x82 += einsum("ia,wja->wji", t1, x76)
    u11new += einsum("wia,ijab->wjb", x76, t2)
    del x76
    x77 += einsum("ab->ab", f.vv) * -2
    u11new += einsum("ab,wib->wia", x77, u11) * -0.5
    del x77
    x78 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x78 += einsum("wx,xij->wij", s2, g.boo)
    x82 += einsum("wij->wij", x78)
    x94 += einsum("wij->wij", x78)
    del x78
    x79 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x79 += einsum("wia,xwja->xij", g.bov, u12)
    x82 += einsum("wij->wij", x79)
    x103 += einsum("wij->wij", x79)
    del x79
    x80 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x80 -= einsum("wia,ijka->wjk", u11, v.ooov)
    x82 += einsum("wij->wij", x80)
    x103 += einsum("wij->wij", x80)
    del x80
    x104 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x104 += einsum("wia,xij->wxja", u11, x103)
    del x103
    x107 += einsum("wxia->wxia", x104)
    del x104
    x81 += einsum("ia->ia", f.ov)
    x82 += einsum("ia,wja->wij", x81, u11)
    del x81
    x82 += einsum("wij->wij", gc.boo)
    u11new -= einsum("ia,wij->wja", t1, x82)
    del x82
    x83 += einsum("ij->ij", f.oo) * 2
    u11new += einsum("ij,wia->wja", x83, u11) * -0.5
    del x83
    x84 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x84 += einsum("wx,xab->wab", s2, g.bvv)
    x86 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x86 += einsum("wab->wab", x84)
    x100 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x100 += einsum("wab->wab", x84)
    del x84
    x85 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x85 -= einsum("wia,ibac->wbc", u11, v.ovvv)
    x86 -= einsum("wab->wab", x85)
    x98 += einsum("wab->wba", x85)
    del x85
    x86 += einsum("wab->wab", gc.bvv)
    u11new += einsum("ia,wba->wib", t1, x86)
    del x86
    x87 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x87 += einsum("wxia,ijab->wxjb", u12, v.oovv)
    u12new += einsum("ijab,wxjb->xwia", t2, x87)
    del x87
    x88 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x88 += einsum("wia,xyia->wxy", g.bov, u12)
    u12new += einsum("wia,wxy->yxia", u11, x88)
    del x88
    x89 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x89 += einsum("wx,yxia->ywia", w, u12)
    x107 -= einsum("wxia->wxia", x89)
    del x89
    x92 += einsum("wia->wia", gc.bov)
    x93 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x93 += einsum("ia,wja->wij", t1, x92)
    x94 += einsum("wij->wji", x93)
    del x93
    x105 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x105 += einsum("wia,xja->wxij", u11, x92)
    del x92
    x106 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x106 += einsum("ia,wxji->xwja", t1, x105)
    del x105
    x107 += einsum("wxia->wxia", x106)
    del x106
    x94 += einsum("wij->wij", gc.boo)
    x95 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x95 += einsum("wia,xij->wxja", u11, x94)
    del x94
    x107 += einsum("wxia->xwia", x95)
    del x95
    x96 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x96 += einsum("wia,xwib->xab", g.bov, u12)
    x98 += einsum("wab->wab", x96)
    del x96
    x99 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x99 += einsum("wia,xab->wxib", u11, x98)
    del x98
    x107 += einsum("wxia->wxia", x99)
    del x99
    x100 += einsum("wab->wab", gc.bvv)
    x101 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x101 += einsum("wia,xba->wxib", u11, x100)
    del x100
    x107 -= einsum("wxia->xwia", x101)
    del x101
    u12new -= einsum("wxia->wxia", x107)
    u12new -= einsum("wxia->xwia", x107)
    del x107
    x108 += einsum("iajb->ijab", v.ovov)
    u12new -= einsum("wxia,ijba->xwjb", u12, x108)
    del x108
    x109 += einsum("ab->ab", f.vv) * -2
    u12new += einsum("ab,wxib->xwia", x109, u12) * -0.5
    del x109
    x110 -= einsum("ijka->jika", v.ooov)
    x111 -= einsum("wxia,ijka->xwjk", u12, x110)
    del x110
    u12new -= einsum("ia,wxij->xwja", t1, x111)
    del x111
    t1new += einsum("w,wai->ia", s1, g.bvo)
    t1new += einsum("ai->ia", f.vo)
    t1new += einsum("ijab,jcab->ic", t2, v.ovvv) * -0.5
    t1new -= einsum("ia,ibja->jb", t1, v.ovov)
    t1new += einsum("wab,wib->ia", g.bvv, u11)
    t2new -= einsum("ia,ibjk->kjba", t1, v.ovoo)
    t2new += einsum("abij->jiba", v.vvoo)
    t2new += einsum("ijab,cdab->jidc", t2, v.vvvv) * 0.5
    s1new += einsum("w,xw->x", s1, w)
    s1new += einsum("ia,wia->w", t1, gc.bov)
    s1new += einsum("w->w", G)
    s1new += einsum("wia,xwia->x", g.bov, u12)
    u11new += einsum("wx,xai->wia", s2, g.bvo)
    u11new += einsum("wai->wia", gc.bvo)
    u11new -= einsum("wia,ibja->wjb", u11, v.ovov)
    u11new += einsum("wab,xwib->xia", g.bvv, u12)

    return {"t1new": t1new, "t2new": t2new, "s1new": s1new, "s2new": s2new, "u11new": u11new, "u12new": u12new}

def update_lams(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, lu12=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    # L1, L2, LS1, LS2, LU11 and LU12 amplitudes
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x0 += einsum("abij,klab->ijkl", l2, t2)
    x15 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x15 += einsum("ijkl->jilk", x0) * -1
    x134 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x134 += einsum("ijkl->jilk", x0) * -0.5
    l1new = np.zeros((nvir, nocc), dtype=types[float])
    l1new += einsum("ijka,lkij->al", v.ooov, x0) * -0.25
    del x0
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum("ia,bajk->jkib", t1, l2)
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum("ia,jkla->jkil", t1, x1)
    x15 += einsum("ijkl->ijlk", x2) * 2.0000000000000013
    x16 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x16 += einsum("ia,ijkl->jlka", t1, x15) * -0.5
    del x15
    x134 += einsum("ijkl->ijlk", x2)
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum("ijab,klij->balk", v.oovv, x134) * -0.5
    del x134
    l1new += einsum("ijka,lkji->al", v.ooov, x2) * 0.5
    del x2
    x16 += einsum("ijab,kjlb->klia", t2, x1) * 2
    x47 = np.zeros((nocc, nvir), dtype=types[float])
    x47 += einsum("ijab,ijka->kb", t2, x1)
    x56 = np.zeros((nocc, nvir), dtype=types[float])
    x56 += einsum("ia->ia", x47) * 0.5
    x149 = np.zeros((nocc, nvir), dtype=types[float])
    x149 += einsum("ia->ia", x47)
    del x47
    l2new += einsum("iabc,jkia->cbkj", v.ovvv, x1)
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x3 += einsum("ia,jkba->ijkb", t1, v.oovv)
    x4 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x4 += einsum("ijka->ikja", x3) * -1
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 += einsum("ijka->kjia", x3) * 0.5000000000000003
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x20 += einsum("ijka->kjia", x3)
    x89 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x89 -= einsum("ai,ijkb->kjab", l1, x3)
    x98 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x98 += einsum("ijab->ijab", x89)
    del x89
    x104 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x104 -= einsum("ijka->ikja", x3)
    x132 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x132 += einsum("ijka->kjia", x3) * 0.5
    x4 += einsum("ijka->kjia", v.ooov)
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x13 += einsum("ijab,kila->lkjb", t2, x4)
    x42 = np.zeros((nocc, nvir), dtype=types[float])
    x42 += einsum("ijab,kija->kb", t2, x4)
    x142 = np.zeros((nocc, nvir), dtype=types[float])
    x142 += einsum("ijab,kija->kb", t2, x4) * 0.5
    del x4
    x148 = np.zeros((nocc, nvir), dtype=types[float])
    x148 += einsum("ia->ia", x142)
    del x142
    x5 = np.zeros((nocc, nvir), dtype=types[float])
    x5 += einsum("w,wia->ia", s1, g.bov)
    x7 = np.zeros((nocc, nvir), dtype=types[float])
    x7 += einsum("ia->ia", x5)
    x64 = np.zeros((nocc, nvir), dtype=types[float])
    x64 += einsum("ia->ia", x5)
    x96 = np.zeros((nocc, nvir), dtype=types[float])
    x96 += einsum("ia->ia", x5)
    del x5
    x6 = np.zeros((nocc, nvir), dtype=types[float])
    x6 += einsum("ia,jiba->jb", t1, v.oovv)
    x7 += einsum("ia->ia", x6)
    x64 += einsum("ia->ia", x6)
    x92 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x92 -= einsum("ia,jkib->kjba", x6, x1)
    x98 -= einsum("ijab->ijab", x92)
    del x92
    x107 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x107 += einsum("ai,jb->ijab", l1, x6)
    x114 = np.zeros((nocc, nocc), dtype=types[float])
    x114 += einsum("ia,ja->ij", t1, x6)
    del x6
    x115 = np.zeros((nocc, nocc), dtype=types[float])
    x115 += einsum("ij->ji", x114)
    del x114
    x7 += einsum("ia->ia", f.ov)
    x13 += einsum("ia,jkab->ikjb", x7, t2) * -0.5
    x24 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x24 += einsum("ia,wxja->xwij", x7, u12) * 0.5
    x35 = np.zeros((nocc, nocc), dtype=types[float])
    x35 += einsum("ia,ja->ij", t1, x7)
    x36 = np.zeros((nocc, nocc), dtype=types[float])
    x36 += einsum("ij->ji", x35)
    del x35
    x42 += einsum("ia,ijab->jb", x7, t2) * -2
    x74 = np.zeros((nbos, nbos), dtype=types[float])
    x74 += einsum("ia,wxia->xw", x7, u12)
    x137 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x137 += einsum("ia,wja->wij", x7, u11)
    x143 = np.zeros((nocc, nvir), dtype=types[float])
    x143 += einsum("ia,ijab->jb", x7, t2)
    x148 += einsum("ia->ia", x143) * -1
    del x143
    lu11new = np.zeros((nbos, nvir, nocc), dtype=types[float])
    lu11new += einsum("w,ia->wai", ls1, x7) * 2
    lu12new = np.zeros((nbos, nbos, nvir, nocc), dtype=types[float])
    lu12new += einsum("wx,ia->xwai", ls2, x7) * 2
    del x7
    x8 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x8 += einsum("ijab,klab->ijkl", t2, v.oovv)
    x10 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x10 += einsum("ijkl->lkji", x8)
    x133 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x133 += einsum("ijkl->lkji", x8)
    del x8
    x9 += einsum("ijka->jika", v.ooov) * -1
    x10 += einsum("ia,jkla->kjli", t1, x9) * -4
    del x9
    x10 += einsum("ijkl->jilk", v.oooo) * 2
    x13 += einsum("ia,ijkl->jlka", t1, x10) * -0.25
    del x10
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum("ia,jbca->ijbc", t1, v.ovvv)
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum("ijab->jiab", x11) * -0.5
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x19 -= einsum("ijab->jiab", x11)
    x102 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x102 += einsum("ijab->ijab", x11)
    del x11
    x12 += einsum("iajb->ijab", v.ovov)
    x13 += einsum("ia,jkba->jikb", t1, x12) * -1
    del x12
    x13 += einsum("iajk->ikja", v.ovoo) * -0.5
    x13 += einsum("ijab,kcab->kjic", t2, v.ovvv) * -0.25
    l1new += einsum("abij,kija->bk", l2, x13)
    del x13
    x14 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x14 += einsum("ia,wxaj->wxji", t1, lu12)
    x16 += einsum("wxia,wxjk->jkia", u12, x14) * -1
    x46 = np.zeros((nocc, nvir), dtype=types[float])
    x46 += einsum("wxia,wxij->ja", u12, x14)
    x56 += einsum("ia->ia", x46) * 0.5
    x149 += einsum("ia->ia", x46)
    del x46
    x155 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x155 += einsum("wia,xwij->xja", u11, x14)
    x156 = np.zeros((nbos, nbos), dtype=types[float])
    x156 += einsum("wia,xia->wx", g.bov, x155)
    del x155
    x162 = np.zeros((nbos, nbos), dtype=types[float])
    x162 -= einsum("wx->wx", x156)
    del x156
    lu12new += einsum("ijka,wxkj->xwai", v.ooov, x14)
    lu12new -= einsum("wxij,ikja->xwak", x14, x3)
    del x3
    x16 += einsum("ai,jkba->ikjb", l1, t2) * -1
    l1new += einsum("ijab,kija->bk", v.oovv, x16) * -0.5
    del x16
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum("wxai,wxjb->ijab", lu12, u12)
    x17 += einsum("abij,kicb->jkac", l2, t2) * -2
    l1new += einsum("iabc,jiab->cj", v.ovvv, x17) * 0.5
    del x17
    x18 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x18 += einsum("abic->ibac", v.vvov) * -1
    x18 += einsum("ia,bcda->icbd", t1, v.vvvv)
    l1new += einsum("abij,iabc->cj", l2, x18) * 0.5
    del x18
    x19 += einsum("iajb->ijab", v.ovov)
    l1new += einsum("ijka,kiab->bj", x1, x19)
    lu12new -= einsum("wxai,jiab->xwbj", lu12, x19)
    del x19
    x20 += einsum("ijka->jika", v.ooov) * -1
    x24 += einsum("wxia,ijka->xwjk", u12, x20) * -0.5
    del x20
    x21 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x21 += einsum("wx,wia->xia", s2, g.bov)
    x23 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x23 += einsum("wia->wia", x21)
    del x21
    x22 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x22 += einsum("wia,jiba->wjb", u11, v.oovv)
    x23 += einsum("wia->wia", x22)
    x65 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x65 += einsum("wia->wia", x22)
    x71 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x71 += einsum("wia->wia", x22) * 0.5
    x100 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x100 += einsum("wai,wjb->ijab", lu11, x22)
    x107 += einsum("ijab->ijab", x100)
    del x100
    x136 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x136 += einsum("wia->wia", x22)
    del x22
    x23 += einsum("wia->wia", gc.bov)
    x24 += einsum("wia,xja->wxji", u11, x23)
    l1new += einsum("wxai,wxji->aj", lu12, x24) * -1
    del x24
    x141 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x141 += einsum("wia,ijab->wjb", x23, t2)
    del x23
    x25 = np.zeros((nocc, nvir), dtype=types[float])
    x25 += einsum("w,wai->ia", s1, g.bvo)
    x42 += einsum("ia->ia", x25) * -2
    x148 += einsum("ia->ia", x25) * -1
    del x25
    x26 = np.zeros((nocc, nvir), dtype=types[float])
    x26 += einsum("wab,wib->ia", g.bvv, u11)
    x42 += einsum("ia->ia", x26) * -2
    x148 += einsum("ia->ia", x26) * -1
    del x26
    x27 = np.zeros((nocc, nvir), dtype=types[float])
    x27 += einsum("ia,ibja->jb", t1, v.ovov)
    x42 += einsum("ia->ia", x27) * 2
    x148 += einsum("ia->ia", x27)
    del x27
    x28 = np.zeros((nocc, nvir), dtype=types[float])
    x28 -= einsum("ijab,icab->jc", t2, v.ovvv)
    x42 += einsum("ia->ia", x28)
    x148 += einsum("ia->ia", x28) * 0.5
    del x28
    x29 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x29 += einsum("ia,wja->wji", t1, g.bov)
    x30 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x30 += einsum("wij->wij", x29)
    x66 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x66 += einsum("wij->wij", x29)
    del x29
    x30 += einsum("wij->wij", g.boo)
    x42 += einsum("wia,wij->ja", u11, x30) * 2
    x137 += einsum("wx,wij->xij", s2, x30)
    x141 += einsum("wij,wxia->xja", x30, u12) * -1
    x144 = np.zeros((nocc, nvir), dtype=types[float])
    x144 += einsum("wia,wij->ja", u11, x30)
    del x30
    x148 += einsum("ia->ia", x144)
    del x144
    x31 = np.zeros((nocc, nocc), dtype=types[float])
    x31 += einsum("w,wij->ij", s1, g.boo)
    x36 += einsum("ij->ij", x31)
    x110 = np.zeros((nocc, nocc), dtype=types[float])
    x110 += einsum("ij->ij", x31)
    del x31
    x32 = np.zeros((nocc, nocc), dtype=types[float])
    x32 += einsum("wia,wja->ij", g.bov, u11)
    x36 += einsum("ij->ij", x32)
    x110 += einsum("ij->ij", x32)
    del x32
    x33 = np.zeros((nocc, nocc), dtype=types[float])
    x33 += einsum("ia,jika->jk", t1, v.ooov)
    x36 += einsum("ij->ij", x33)
    x115 += einsum("ij->ij", x33)
    del x33
    x116 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x116 += einsum("ij,abjk->kiab", x115, l2)
    del x115
    x117 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x117 -= einsum("ijab->ijba", x116)
    del x116
    x34 = np.zeros((nocc, nocc), dtype=types[float])
    x34 += einsum("ijab,ikab->jk", t2, v.oovv)
    x36 += einsum("ij->ji", x34) * 0.5
    x121 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x121 += einsum("ij,abki->kjab", x34, l2)
    del x34
    x124 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x124 += einsum("ijab->ijba", x121) * -0.5
    del x121
    x36 += einsum("ij->ij", f.oo)
    x42 += einsum("ia,ij->ja", t1, x36) * 2
    x141 += einsum("ij,wia->wja", x36, u11) * -1
    x145 = np.zeros((nocc, nvir), dtype=types[float])
    x145 += einsum("ia,ij->ja", t1, x36)
    x148 += einsum("ia->ia", x145)
    del x145
    l1new += einsum("ai,ji->aj", l1, x36) * -1
    lu12new += einsum("ij,wxaj->xwai", x36, lu12) * -1
    del x36
    x37 = np.zeros((nvir, nvir), dtype=types[float])
    x37 += einsum("w,wab->ab", s1, g.bvv)
    x39 = np.zeros((nvir, nvir), dtype=types[float])
    x39 += einsum("ab->ab", x37)
    x80 = np.zeros((nvir, nvir), dtype=types[float])
    x80 += einsum("ab->ab", x37)
    x94 = np.zeros((nvir, nvir), dtype=types[float])
    x94 += einsum("ab->ab", x37)
    x135 = np.zeros((nvir, nvir), dtype=types[float])
    x135 += einsum("ab->ab", x37) * -1
    del x37
    x38 = np.zeros((nvir, nvir), dtype=types[float])
    x38 += einsum("ia,ibca->bc", t1, v.ovvv)
    x39 += einsum("ab->ab", x38) * -1
    x80 -= einsum("ab->ab", x38)
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x91 -= einsum("ab,caij->jicb", x38, l2)
    x98 -= einsum("ijab->ijab", x91)
    del x91
    x135 += einsum("ab->ab", x38)
    del x38
    x39 += einsum("ab->ab", f.vv)
    x42 += einsum("ia,ba->ib", t1, x39) * -2
    x146 = np.zeros((nocc, nvir), dtype=types[float])
    x146 += einsum("ia,ba->ib", t1, x39)
    del x39
    x148 += einsum("ia->ia", x146) * -1
    del x146
    x40 = np.zeros((nbos), dtype=types[float])
    x40 += einsum("ia,wia->w", t1, g.bov)
    x41 = np.zeros((nbos), dtype=types[float])
    x41 += einsum("w->w", x40)
    x85 = np.zeros((nbos), dtype=types[float])
    x85 += einsum("w->w", x40)
    del x40
    x41 += einsum("w->w", G)
    x42 += einsum("w,wia->ia", x41, u11) * -2
    x141 += einsum("w,wxia->xia", x41, u12)
    x147 = np.zeros((nocc, nvir), dtype=types[float])
    x147 += einsum("w,wia->ia", x41, u11)
    del x41
    x148 += einsum("ia->ia", x147) * -1
    del x147
    x42 += einsum("ai->ia", f.vo) * -2
    l1new += einsum("ia,abij->bj", x42, l2) * -0.5
    del x42
    x43 = np.zeros((nocc, nvir), dtype=types[float])
    x43 += einsum("w,wia->ia", ls1, u11)
    x56 += einsum("ia->ia", x43) * -1
    x149 += einsum("ia->ia", x43) * -2
    del x43
    x44 = np.zeros((nocc, nvir), dtype=types[float])
    x44 += einsum("wx,wxia->ia", ls2, u12)
    x56 += einsum("ia->ia", x44) * -0.5
    x149 += einsum("ia->ia", x44) * -1
    del x44
    x45 = np.zeros((nocc, nvir), dtype=types[float])
    x45 += einsum("ai,jiba->jb", l1, t2)
    x56 += einsum("ia->ia", x45) * -1
    x149 += einsum("ia->ia", x45) * -2
    del x45
    x48 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x48 += einsum("ia,waj->wji", t1, lu11)
    x50 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x50 += einsum("wij->wij", x48)
    x69 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x69 += einsum("wij->wij", x48)
    x49 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x49 += einsum("wia,wxaj->xji", u11, lu12)
    x50 += einsum("wij->wij", x49)
    x56 += einsum("wia,wij->ja", u11, x50)
    x149 += einsum("wia,wij->ja", u11, x50) * 2
    del x50
    x69 += einsum("wij->wij", x49)
    x70 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x70 += einsum("wx,wij->xij", s2, x69)
    x170 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x170 += einsum("wia,xji->wxja", g.bov, x69)
    x171 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x171 += einsum("wxia->wxia", x170)
    del x170
    x51 = np.zeros((nocc, nocc), dtype=types[float])
    x51 += einsum("ai,ja->ij", l1, t1)
    x55 = np.zeros((nocc, nocc), dtype=types[float])
    x55 += einsum("ij->ij", x51) * 2
    x79 = np.zeros((nocc, nocc), dtype=types[float])
    x79 += einsum("ij->ij", x51)
    x112 = np.zeros((nocc, nocc), dtype=types[float])
    x112 += einsum("ij->ij", x51)
    del x51
    x52 = np.zeros((nocc, nocc), dtype=types[float])
    x52 += einsum("wai,wja->ij", lu11, u11)
    x55 += einsum("ij->ij", x52) * 2
    x79 += einsum("ij->ij", x52)
    x112 += einsum("ij->ij", x52)
    del x52
    x113 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x113 += einsum("ij,jkab->kiab", x112, v.oovv)
    del x112
    x117 += einsum("ijab->jiba", x113)
    del x113
    x53 = np.zeros((nocc, nocc), dtype=types[float])
    x53 += einsum("wxai,wxja->ij", lu12, u12)
    x55 += einsum("ij->ij", x53)
    x79 += einsum("ij->ij", x53) * 0.5
    x122 = np.zeros((nocc, nocc), dtype=types[float])
    x122 += einsum("ij->ij", x53)
    del x53
    x54 = np.zeros((nocc, nocc), dtype=types[float])
    x54 += einsum("abij,ikab->jk", l2, t2)
    x55 += einsum("ij->ij", x54)
    x56 += einsum("ia,ij->ja", t1, x55) * 0.5
    x149 += einsum("ia,ij->ja", t1, x55)
    l1new += einsum("ij,jkia->ak", x55, v.ooov) * -0.5
    ls1new = np.zeros((nbos), dtype=types[float])
    ls1new += einsum("ij,wji->w", x55, g.boo) * -0.5
    del x55
    x79 += einsum("ij->ij", x54) * 0.5
    x122 += einsum("ij->ij", x54)
    del x54
    x123 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x123 += einsum("ij,jkab->kiab", x122, v.oovv) * 0.5
    del x122
    x124 += einsum("ijab->jiba", x123) * -1
    del x123
    l2new += einsum("ijab->abij", x124) * -1
    l2new += einsum("ijab->abji", x124)
    del x124
    x56 += einsum("ia->ia", t1) * -1
    l1new += einsum("ia,ijab->bj", x56, v.oovv) * -1
    del x56
    x57 = np.zeros((nvir, nvir), dtype=types[float])
    x57 += einsum("ai,ib->ab", l1, t1)
    x61 = np.zeros((nvir, nvir), dtype=types[float])
    x61 += einsum("ab->ab", x57)
    ls1new += einsum("ab,wab->w", x57, g.bvv)
    del x57
    x58 = np.zeros((nvir, nvir), dtype=types[float])
    x58 += einsum("wai,wib->ab", lu11, u11)
    x61 += einsum("ab->ab", x58)
    x90 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x90 -= einsum("ab,ijcb->jiac", x58, v.oovv)
    x98 += einsum("ijab->ijab", x90)
    del x90
    x150 = np.zeros((nvir, nvir), dtype=types[float])
    x150 += einsum("ab->ab", x58) * 2
    del x58
    x59 = np.zeros((nvir, nvir), dtype=types[float])
    x59 += einsum("wxai,wxib->ab", lu12, u12)
    x61 += einsum("ab->ab", x59) * 0.5
    x127 = np.zeros((nvir, nvir), dtype=types[float])
    x127 += einsum("ab->ab", x59)
    x150 += einsum("ab->ab", x59)
    del x59
    x60 = np.zeros((nvir, nvir), dtype=types[float])
    x60 -= einsum("abij,ijbc->ac", l2, t2)
    x61 += einsum("ab->ab", x60) * 0.5
    l1new += einsum("ab,iabc->ci", x61, v.ovvv) * -1
    del x61
    x127 += einsum("ab->ab", x60)
    x128 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x128 += einsum("ab,ijbc->ijca", x127, v.oovv) * 0.5
    del x127
    x129 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x129 += einsum("ijab->jiba", x128) * -1
    del x128
    x150 += einsum("ab->ab", x60)
    del x60
    ls1new += einsum("ab,wab->w", x150, g.bvv) * 0.5
    del x150
    x62 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x62 += einsum("wia,wxja->xij", g.bov, u12)
    x67 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x67 += einsum("wij->wij", x62)
    x137 += einsum("wij->wij", x62)
    del x62
    x63 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x63 += einsum("wia,jika->wjk", u11, v.ooov)
    x67 += einsum("wij->wij", x63)
    x137 += einsum("wij->wij", x63)
    del x63
    x64 += einsum("ia->ia", f.ov)
    x67 += einsum("ia,wja->wij", x64, u11)
    x84 = np.zeros((nbos), dtype=types[float])
    x84 += einsum("ia,wia->w", x64, u11)
    x87 = np.zeros((nbos), dtype=types[float])
    x87 += einsum("w->w", x84)
    del x84
    l1new += einsum("ia,ji->aj", x64, x79) * -1
    del x79
    lu12new -= einsum("ia,wxji->xwaj", x64, x14)
    del x64
    del x14
    x65 += einsum("wia->wia", gc.bov)
    x67 += einsum("ia,wja->wji", t1, x65)
    l1new -= einsum("wia,wji->aj", x65, x69)
    del x65
    del x69
    x66 += einsum("wij->wij", g.boo)
    x67 += einsum("wx,wij->xij", s2, x66)
    x160 = np.zeros((nbos, nbos), dtype=types[float])
    x160 += einsum("wij,xji->wx", x48, x66)
    del x48
    x162 -= einsum("wx->xw", x160)
    del x160
    x161 = np.zeros((nbos, nbos), dtype=types[float])
    x161 += einsum("wij,xji->wx", x49, x66)
    del x49
    x162 -= einsum("wx->xw", x161)
    del x161
    x169 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x169 += einsum("wai,xji->wxja", lu11, x66)
    x171 += einsum("wxia->xwia", x169)
    del x169
    x67 += einsum("wij->wij", gc.boo)
    l1new -= einsum("wai,wji->aj", lu11, x67)
    del x67
    x68 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x68 += einsum("wia,baji->wjb", u11, l2)
    x70 += einsum("ia,wja->wji", t1, x68)
    x76 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x76 += einsum("wia->wia", x68)
    l1new -= einsum("wij,wja->ai", x66, x68)
    del x66
    del x68
    x70 += einsum("ai,wja->wij", l1, u11)
    x70 += einsum("wai,wxja->xij", lu11, u12)
    l1new -= einsum("wia,wji->aj", g.bov, x70)
    del x70
    x71 += einsum("wia->wia", gc.bov)
    x74 += einsum("wia,xia->xw", u11, x71) * 2
    del x71
    x72 = np.zeros((nbos, nbos), dtype=types[float])
    x72 += einsum("wia,xia->wx", g.bov, u11)
    x73 = np.zeros((nbos, nbos), dtype=types[float])
    x73 += einsum("wx->xw", x72)
    x152 = np.zeros((nbos, nbos), dtype=types[float])
    x152 += einsum("wx,yx->yw", ls2, x72)
    x162 += einsum("wx->wx", x152)
    del x152
    x168 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x168 += einsum("wx,xyai->wyia", x72, lu12)
    del x72
    x171 -= einsum("wxia->wxia", x168)
    del x168
    x73 += einsum("wx->wx", w)
    x74 += einsum("wx,yw->xy", s2, x73) * 2
    l1new += einsum("wx,xwai->ai", x74, lu12) * 0.5
    del x74
    x141 += einsum("wx,xia->wia", x73, u11)
    del x73
    x75 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x75 += einsum("wx,xai->wia", s2, lu11)
    x76 += einsum("wia->wia", x75)
    del x75
    x106 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x106 += einsum("wia,wjb->ijab", g.bov, x76)
    x107 += einsum("ijab->ijab", x106)
    del x106
    l1new += einsum("wab,wia->bi", g.bvv, x76)
    del x76
    x77 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x77 += einsum("wia,ibca->wbc", u11, v.ovvv)
    x78 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x78 -= einsum("wab->wab", x77)
    x138 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x138 += einsum("wab->wab", x77) * -1
    del x77
    x78 += einsum("wab->wab", gc.bvv)
    l1new += einsum("wai,wab->bi", lu11, x78)
    del x78
    x80 += einsum("ab->ab", f.vv)
    l1new += einsum("ai,ab->bi", l1, x80)
    del x80
    x81 = np.zeros((nbos), dtype=types[float])
    x81 += einsum("w,xw->x", s1, w)
    x87 += einsum("w->w", x81)
    del x81
    x82 = np.zeros((nbos), dtype=types[float])
    x82 += einsum("ia,wia->w", t1, gc.bov)
    x87 += einsum("w->w", x82)
    del x82
    x83 = np.zeros((nbos), dtype=types[float])
    x83 += einsum("wia,wxia->x", g.bov, u12)
    x87 += einsum("w->w", x83)
    del x83
    x85 += einsum("w->w", G)
    x86 = np.zeros((nbos), dtype=types[float])
    x86 += einsum("w,wx->x", x85, s2)
    x87 += einsum("w->w", x86)
    del x86
    x162 += einsum("w,x->xw", ls1, x85)
    del x85
    x87 += einsum("w->w", G)
    l1new += einsum("w,wai->ai", x87, lu11)
    ls1new += einsum("w,wx->x", x87, ls2)
    del x87
    x88 = np.zeros((nbos), dtype=types[float])
    x88 += einsum("w->w", s1)
    x88 += einsum("w,xw->x", ls1, s2)
    x88 += einsum("ai,wia->w", l1, u11)
    x88 += einsum("wai,wxia->x", lu11, u12)
    l1new += einsum("w,wia->ai", x88, g.bov)
    del x88
    x93 = np.zeros((nvir, nvir), dtype=types[float])
    x93 += einsum("wia,wib->ab", g.bov, u11)
    x94 -= einsum("ab->ba", x93)
    x95 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x95 += einsum("ab,acij->ijcb", x94, l2)
    del x94
    x98 -= einsum("ijab->jiba", x95)
    del x95
    x135 += einsum("ab->ba", x93)
    del x93
    x96 += einsum("ia->ia", f.ov)
    x97 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x97 += einsum("ia,jkib->jkba", x96, x1)
    x98 -= einsum("ijab->jiba", x97)
    del x97
    l2new += einsum("ijab->abij", x98)
    l2new -= einsum("ijab->baij", x98)
    del x98
    x107 += einsum("ai,jb->jiba", l1, x96)
    x109 = np.zeros((nocc, nocc), dtype=types[float])
    x109 += einsum("ia,ja->ij", t1, x96)
    del x96
    x110 += einsum("ij->ji", x109)
    del x109
    x99 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x99 += einsum("wia,wbj->ijab", gc.bov, lu11)
    x107 += einsum("ijab->ijab", x99)
    del x99
    x101 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x101 -= einsum("ijab,ikca->jkbc", t2, v.oovv)
    x102 += einsum("ijab->ijab", x101)
    del x101
    x102 -= einsum("iajb->jiab", v.ovov)
    x103 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x103 += einsum("abij,ikac->jkbc", l2, x102)
    del x102
    x107 += einsum("ijab->ijab", x103)
    del x103
    x104 += einsum("ijka->kjia", v.ooov)
    x105 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x105 += einsum("ijka,iklb->jlab", x1, x104)
    del x104
    del x1
    x107 -= einsum("ijab->ijab", x105)
    del x105
    l2new += einsum("ijab->abij", x107)
    l2new -= einsum("ijab->baij", x107)
    l2new -= einsum("ijab->abji", x107)
    l2new += einsum("ijab->baji", x107)
    del x107
    x108 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x108 += einsum("ai,jabc->ijbc", l1, v.ovvv)
    x117 += einsum("ijab->ijba", x108)
    del x108
    x110 += einsum("ij->ij", f.oo)
    x111 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x111 += einsum("ij,abjk->kiab", x110, l2)
    del x110
    x117 += einsum("ijab->jiba", x111)
    del x111
    l2new += einsum("ijab->abij", x117)
    l2new -= einsum("ijab->abji", x117)
    del x117
    x118 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x118 += einsum("ai,jkib->jkab", l1, v.ooov)
    x120 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x120 -= einsum("ijab->jiab", x118)
    del x118
    x119 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x119 += einsum("ab,caij->ijbc", f.vv, l2)
    x120 -= einsum("ijab->jiab", x119)
    del x119
    l2new -= einsum("ijab->abij", x120)
    l2new += einsum("ijab->baij", x120)
    del x120
    x125 = np.zeros((nvir, nvir), dtype=types[float])
    x125 += einsum("ijab,ijac->bc", t2, v.oovv)
    x126 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x126 -= einsum("ab,caij->jicb", x125, l2)
    x129 += einsum("ijab->ijab", x126) * 0.5
    del x126
    l2new += einsum("ijab->abij", x129) * -1
    l2new += einsum("ijab->baij", x129)
    del x129
    x135 += einsum("ab->ab", x125) * 0.5
    del x125
    x130 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x130 += einsum("wxia,ijab->wxjb", u12, v.oovv)
    x131 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x131 += einsum("wxai,wxjb->ijab", lu12, x130)
    del x130
    l2new += einsum("ijab->abij", x131) * 0.5
    l2new += einsum("ijab->baij", x131) * -0.5
    l2new += einsum("ijab->abji", x131) * -0.5
    l2new += einsum("ijab->baji", x131) * 0.5
    del x131
    x132 += einsum("ijka->jika", v.ooov) * -1
    x133 += einsum("ia,jkla->kjli", t1, x132) * -4
    del x132
    x133 += einsum("ijkl->jilk", v.oooo) * 2
    l2new += einsum("abij,klij->balk", l2, x133) * 0.25
    del x133
    x135 += einsum("ab->ab", f.vv) * -1
    x141 += einsum("ab,wib->wia", x135, u11) * -1
    lu12new += einsum("ab,wxai->xwbi", x135, lu12) * -1
    del x135
    x136 += einsum("wia->wia", gc.bov)
    x137 += einsum("ia,wja->wji", t1, x136)
    del x136
    x137 += einsum("wij->wij", gc.boo)
    x141 += einsum("ia,wij->wja", t1, x137) * -1
    del x137
    x138 += einsum("wab->wab", gc.bvv)
    x141 += einsum("ia,wba->wib", t1, x138)
    del x138
    x139 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x139 += einsum("ia,wba->wib", t1, g.bvv)
    x140 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x140 += einsum("wia->wia", x139)
    x158 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x158 += einsum("wia->wia", x139)
    del x139
    x140 += einsum("wai->wia", g.bvo)
    x141 += einsum("wx,wia->xia", s2, x140)
    del x140
    x141 += einsum("wai->wia", gc.bvo)
    x141 += einsum("wab,wxib->xia", g.bvv, u12)
    x141 += einsum("wia,ibja->wjb", u11, v.ovov) * -1
    ls1new += einsum("wia,wxai->x", x141, lu12)
    del x141
    x148 += einsum("ai->ia", f.vo) * -1
    ls1new += einsum("ia,wai->w", x148, lu11) * -1
    ls2new = np.zeros((nbos, nbos), dtype=types[float])
    ls2new += einsum("ia,wxai->xw", x148, lu12) * -1
    del x148
    x149 += einsum("ia->ia", t1) * -2
    ls1new += einsum("ia,wia->w", x149, g.bov) * -0.5
    del x149
    x151 = np.zeros((nbos, nbos), dtype=types[float])
    x151 += einsum("wx,xy->wy", ls2, w)
    x162 += einsum("wx->wx", x151)
    del x151
    x153 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x153 += einsum("wia,wxbi->xba", u11, lu12)
    x154 = np.zeros((nbos, nbos), dtype=types[float])
    x154 += einsum("wab,xab->wx", g.bvv, x153)
    x162 += einsum("wx->wx", x154)
    del x154
    x167 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x167 += einsum("wia,xba->wxib", g.bov, x153)
    del x153
    x171 += einsum("wxia->wxia", x167)
    del x167
    x157 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x157 += einsum("wia,jiba->wjb", g.bov, t2)
    x158 += einsum("wia->wia", x157)
    del x157
    x158 += einsum("wai->wia", g.bvo)
    x159 = np.zeros((nbos, nbos), dtype=types[float])
    x159 += einsum("wai,xia->wx", lu11, x158)
    del x158
    x162 += einsum("wx->xw", x159)
    del x159
    ls2new += einsum("wx->wx", x162)
    ls2new += einsum("wx->xw", x162)
    del x162
    x163 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x163 += einsum("wxai,ijab->wxjb", lu12, t2)
    lu12new += einsum("ijab,wxia->xwbj", v.oovv, x163)
    del x163
    x164 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x164 += einsum("wia,xyai->xyw", u11, lu12)
    lu12new += einsum("wia,xyw->yxai", g.bov, x164)
    del x164
    x165 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x165 += einsum("wx,wyai->yxia", w, lu12)
    x171 -= einsum("wxia->wxia", x165)
    del x165
    x166 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x166 += einsum("wab,xai->wxib", g.bvv, lu11)
    x171 -= einsum("wxia->wxia", x166)
    del x166
    lu12new -= einsum("wxia->wxai", x171)
    lu12new -= einsum("wxia->xwai", x171)
    del x171
    l1new += einsum("w,wia->ai", ls1, gc.bov)
    l1new -= einsum("ai,jaib->bj", l1, v.ovov)
    l1new += einsum("ia->ai", f.ov)
    l2new += einsum("abij,abcd->dcji", l2, v.vvvv) * 0.5
    l2new += einsum("ijab->baji", v.oovv)
    ls1new += einsum("ai,wai->w", l1, g.bvo)
    ls1new += einsum("w->w", G)
    ls1new += einsum("w,wx->x", ls1, w)
    lu11new -= einsum("wij,xwaj->xai", gc.boo, lu12)
    lu11new += einsum("wia->wai", g.bov)
    lu11new += einsum("wai,baji->wbj", g.bvo, l2)
    lu11new -= einsum("wai,jaib->wbj", lu11, v.ovov)
    lu11new += einsum("wab,xwai->xbi", gc.bvv, lu12)
    lu11new += einsum("ab,wai->wbi", f.vv, lu11)
    lu11new += einsum("ai,wab->wbi", l1, g.bvv)
    lu11new -= einsum("ij,waj->wai", f.oo, lu11)
    lu11new -= einsum("ai,wji->waj", l1, g.boo)
    lu11new += einsum("w,xwai->xai", G, lu12)
    lu11new += einsum("wx,xia->wai", ls2, gc.bov)
    lu11new += einsum("wx,wai->xai", w, lu11)

    return {"l1new": l1new, "l2new": l2new, "ls1new": ls1new, "ls2new": ls2new, "lu11new": lu11new, "lu12new": lu12new}

def make_rdm1_f(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, lu12=None, **kwargs):
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
    x0 += einsum("wai,wja->ij", lu11, u11)
    x7 = np.zeros((nocc, nocc), dtype=types[float])
    x7 += einsum("ij->ij", x0) * 2
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=types[float])
    rdm1_f_oo -= einsum("ij->ij", x0)
    del x0
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 += einsum("ai,ja->ij", l1, t1)
    x7 += einsum("ij->ij", x1) * 2
    rdm1_f_oo -= einsum("ij->ij", x1)
    del x1
    x2 = np.zeros((nocc, nocc), dtype=types[float])
    x2 += einsum("abij,kjab->ik", l2, t2)
    x7 += einsum("ij->ij", x2)
    rdm1_f_oo += einsum("ij->ij", x2) * -0.5
    del x2
    x3 = np.zeros((nocc, nocc), dtype=types[float])
    x3 += einsum("wxai,wxja->ij", lu12, u12)
    x7 += einsum("ij->ij", x3)
    rdm1_f_vo = np.zeros((nvir, nocc), dtype=types[float])
    rdm1_f_vo += einsum("ia,ij->aj", t1, x7) * -0.5
    del x7
    rdm1_f_oo += einsum("ij->ij", x3) * -0.5
    del x3
    x4 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x4 += einsum("ia,wxaj->wxji", t1, lu12)
    rdm1_f_vo += einsum("wxia,wxij->aj", u12, x4) * -0.5
    del x4
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x5 += einsum("ia,bajk->jkib", t1, l2)
    rdm1_f_vo += einsum("ijab,ijkb->ak", t2, x5) * 0.5
    del x5
    x6 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x6 += einsum("ia,waj->wji", t1, lu11)
    x6 += einsum("wia,wxaj->xji", u11, lu12)
    rdm1_f_vo -= einsum("wia,wij->aj", u11, x6)
    del x6
    rdm1_f_oo += einsum("ij->ji", delta_oo)
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=types[float])
    rdm1_f_ov += einsum("ai->ia", l1)
    rdm1_f_vo += einsum("ai,jiba->bj", l1, t2)
    rdm1_f_vo += einsum("ia->ai", t1)
    rdm1_f_vo += einsum("w,wia->ai", ls1, u11)
    rdm1_f_vo += einsum("wx,wxia->ai", ls2, u12) * 0.5
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=types[float])
    rdm1_f_vv += einsum("ai,ib->ba", l1, t1)
    rdm1_f_vv += einsum("wai,wib->ba", lu11, u11)
    rdm1_f_vv += einsum("abij,ijca->cb", l2, t2) * -0.5
    rdm1_f_vv += einsum("wxai,wxib->ba", lu12, u12) * 0.5

    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])

    return rdm1_f

def make_rdm2_f(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, lu12=None, **kwargs):
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
    x28 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x28 += einsum("ijkl->jilk", x0) * -1
    x77 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x77 += einsum("ijkl->jilk", x0) * -1
    x78 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x78 += einsum("ijkl->jilk", x0) * -1
    rdm2_f_oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_oooo += einsum("ijkl->jilk", x0) * 0.5
    del x0
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum("ia,bajk->jkib", t1, l2)
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum("ia,jkla->jkil", t1, x1)
    x28 += einsum("ijkl->ijlk", x2) * 2
    x29 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x29 += einsum("ia,ijkl->jkla", t1, x28) * 0.5
    del x28
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_ovoo += einsum("ijka->iakj", x29) * -1
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_vooo += einsum("ijka->aikj", x29)
    del x29
    x77 += einsum("ijkl->ijlk", x2) * 2.0000000000000013
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vvoo += einsum("ijab,ijkl->balk", t2, x77) * -0.25
    del x77
    x78 += einsum("ijkl->ijlk", x2) * 1.9999999999999987
    x79 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x79 += einsum("ia,ijkl->jlka", t1, x78) * -1
    del x78
    rdm2_f_vvoo += einsum("ia,ijkb->bakj", t1, x79) * 0.5000000000000003
    del x79
    rdm2_f_oooo -= einsum("ijkl->ijlk", x2)
    del x2
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x13 -= einsum("ijab,kjlb->klia", t2, x1)
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x20 += einsum("ijka->ijka", x13)
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 -= einsum("ijka->ijka", x13)
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum("ia,ijkb->jkab", t1, x13)
    del x13
    x67 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x67 -= einsum("ijab->ijab", x60)
    del x60
    x24 = np.zeros((nocc, nvir), dtype=types[float])
    x24 -= einsum("ijab,ijkb->ka", t2, x1)
    x26 = np.zeros((nocc, nvir), dtype=types[float])
    x26 += einsum("ia->ia", x24)
    x53 = np.zeros((nocc, nvir), dtype=types[float])
    x53 += einsum("ia->ia", x24)
    del x24
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum("ia,jikb->jkba", t1, x1)
    x88 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x88 -= einsum("ia,ijbc->jbca", t1, x37)
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov -= einsum("iabc->cbia", x88)
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo += einsum("iabc->cbai", x88)
    del x88
    rdm2_f_ovov = np.zeros((nocc, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_ovov += einsum("ijab->ibja", x37)
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_ovvo -= einsum("ijab->ibaj", x37)
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_voov -= einsum("ijab->bija", x37)
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_vovo += einsum("ijab->biaj", x37)
    del x37
    x87 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x87 += einsum("ijab,ijkc->kcab", t2, x1)
    rdm2_f_vvov += einsum("iabc->bcia", x87) * 0.5
    rdm2_f_vvvo += einsum("iabc->bcai", x87) * -0.5
    del x87
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov -= einsum("ijka->jika", x1)
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo += einsum("ijka->jiak", x1)
    del x1
    x3 = np.zeros((nocc, nocc), dtype=types[float])
    x3 += einsum("wai,wja->ij", lu11, u11)
    x15 = np.zeros((nocc, nvir), dtype=types[float])
    x15 += einsum("ia,ij->ja", t1, x3)
    x18 = np.zeros((nocc, nvir), dtype=types[float])
    x18 += einsum("ia->ia", x15)
    del x15
    x19 = np.zeros((nocc, nocc), dtype=types[float])
    x19 += einsum("ij->ij", x3)
    x75 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum("ij,kiab->kjab", x3, t2)
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum("ijab->ijba", x75)
    del x75
    x81 = np.zeros((nocc, nocc), dtype=types[float])
    x81 += einsum("ij->ij", x3)
    rdm2_f_oooo -= einsum("ij,kl->ikjl", delta_oo, x3)
    rdm2_f_oooo += einsum("ij,kl->iklj", delta_oo, x3)
    rdm2_f_oooo += einsum("ij,kl->kjil", delta_oo, x3)
    rdm2_f_oooo -= einsum("ij,kl->kjli", delta_oo, x3)
    del x3
    x4 = np.zeros((nocc, nocc), dtype=types[float])
    x4 += einsum("wxai,wxja->ij", lu12, u12)
    x6 = np.zeros((nocc, nocc), dtype=types[float])
    x6 += einsum("ij->ij", x4)
    x51 = np.zeros((nocc, nocc), dtype=types[float])
    x51 += einsum("ij->ij", x4)
    del x4
    x5 = np.zeros((nocc, nocc), dtype=types[float])
    x5 += einsum("abij,kjab->ik", l2, t2)
    x6 += einsum("ij->ij", x5)
    x25 = np.zeros((nocc, nvir), dtype=types[float])
    x25 += einsum("ia,ij->ja", t1, x6)
    x26 += einsum("ia->ia", x25)
    del x25
    x27 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x27 += einsum("ia,jk->jika", t1, x6) * -0.5
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x35 += einsum("ia,jk->jika", t1, x6) * 0.5
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum("ij,ikab->jkab", x6, t2) * 0.5
    rdm2_f_vvoo += einsum("ijab->abji", x55)
    rdm2_f_vvoo += einsum("ijab->abij", x55) * -1
    del x55
    rdm2_f_oooo += einsum("ij,kl->jkil", delta_oo, x6) * -0.5
    rdm2_f_oooo += einsum("ij,kl->jkli", delta_oo, x6) * 0.5
    rdm2_f_oooo += einsum("ij,kl->kijl", delta_oo, x6) * 0.5
    rdm2_f_oooo += einsum("ij,kl->kilj", delta_oo, x6) * -0.5
    del x6
    x51 += einsum("ij->ij", x5)
    del x5
    x52 = np.zeros((nocc, nvir), dtype=types[float])
    x52 += einsum("ia,ij->ja", t1, x51)
    del x51
    x53 += einsum("ia->ia", x52)
    del x52
    x7 = np.zeros((nocc, nocc), dtype=types[float])
    x7 += einsum("ai,ja->ij", l1, t1)
    x19 += einsum("ij->ij", x7)
    x20 -= einsum("ia,jk->jika", t1, x19)
    x34 += einsum("ia,jk->jika", t1, x19)
    del x19
    x32 = np.zeros((nocc, nvir), dtype=types[float])
    x32 += einsum("ia,ij->ja", t1, x7)
    x33 = np.zeros((nocc, nvir), dtype=types[float])
    x33 += einsum("ia->ia", x32) * -1
    del x32
    x74 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x74 += einsum("ij,kiab->jkab", x7, t2)
    x76 -= einsum("ijab->ijba", x74)
    del x74
    rdm2_f_vvoo -= einsum("ijab->baij", x76)
    rdm2_f_vvoo += einsum("ijab->baji", x76)
    del x76
    x81 += einsum("ij->ij", x7)
    x82 = np.zeros((nocc, nvir), dtype=types[float])
    x82 += einsum("ia,ij->ja", t1, x81)
    x83 = np.zeros((nocc, nvir), dtype=types[float])
    x83 += einsum("ia->ia", x82)
    del x82
    x84 = np.zeros((nocc, nvir), dtype=types[float])
    x84 += einsum("ia,ij->ja", t1, x81) * 2
    del x81
    x85 = np.zeros((nocc, nvir), dtype=types[float])
    x85 += einsum("ia->ia", x84)
    del x84
    rdm2_f_oooo -= einsum("ij,kl->jkil", delta_oo, x7)
    rdm2_f_oooo += einsum("ij,kl->kjil", delta_oo, x7)
    rdm2_f_oooo += einsum("ij,kl->jkli", delta_oo, x7)
    rdm2_f_oooo -= einsum("ij,kl->kjli", delta_oo, x7)
    del x7
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x8 += einsum("ai,jkba->ijkb", l1, t2)
    x70 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x70 += einsum("ia,ijkb->jkab", t1, x8)
    x73 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum("ijab->ijab", x70)
    del x70
    rdm2_f_ovoo -= einsum("ijka->iakj", x8)
    rdm2_f_vooo += einsum("ijka->aikj", x8)
    del x8
    x9 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x9 += einsum("ia,waj->wji", t1, lu11)
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum("wia,wjk->jkia", u11, x9)
    x20 += einsum("ijka->ijka", x10)
    x34 -= einsum("ijka->ijka", x10)
    del x10
    x16 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x16 += einsum("wij->wij", x9)
    x62 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x62 += einsum("ia,wij->wja", t1, x9)
    x63 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x63 -= einsum("wia->wia", x62)
    del x62
    x80 = np.zeros((nocc, nvir), dtype=types[float])
    x80 += einsum("wia,wij->ja", u11, x9)
    del x9
    x83 += einsum("ia->ia", x80)
    x85 += einsum("ia->ia", x80) * 2
    del x80
    x11 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x11 += einsum("wia,wxaj->xji", u11, lu12)
    x12 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x12 += einsum("wia,wjk->jika", u11, x11)
    x20 -= einsum("ijka->ijka", x12)
    x34 += einsum("ijka->ijka", x12)
    del x12
    rdm2_f_vooo -= einsum("ijka->aijk", x34)
    rdm2_f_vooo += einsum("ijka->aikj", x34)
    del x34
    x16 += einsum("wij->wij", x11)
    x17 = np.zeros((nocc, nvir), dtype=types[float])
    x17 += einsum("wia,wij->ja", u11, x16)
    del x16
    x18 += einsum("ia->ia", x17)
    del x17
    x58 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x58 += einsum("ia,wij->wja", t1, x11)
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum("wia,wjb->ijba", u11, x58)
    del x58
    x67 += einsum("ijab->ijab", x59)
    del x59
    x65 = np.zeros((nocc, nvir), dtype=types[float])
    x65 += einsum("wia,wij->ja", u11, x11)
    del x11
    x66 = np.zeros((nocc, nvir), dtype=types[float])
    x66 -= einsum("ia->ia", x65)
    del x65
    x14 = np.zeros((nocc, nvir), dtype=types[float])
    x14 += einsum("ai,jiba->jb", l1, t2)
    x18 -= einsum("ia->ia", x14)
    x20 += einsum("ij,ka->jika", delta_oo, x18)
    rdm2_f_ovoo -= einsum("ijka->iajk", x20)
    rdm2_f_ovoo += einsum("ijka->iakj", x20)
    del x20
    rdm2_f_vooo += einsum("ij,ka->aijk", delta_oo, x18)
    rdm2_f_vooo -= einsum("ij,ka->aikj", delta_oo, x18)
    del x18
    x66 += einsum("ia->ia", x14)
    del x14
    x67 += einsum("ia,jb->ijab", t1, x66)
    del x66
    x21 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x21 += einsum("ia,wxaj->wxji", t1, lu12)
    x22 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x22 += einsum("wxia,wxjk->jkia", u12, x21)
    x27 += einsum("ijka->ijka", x22) * 0.5
    x35 += einsum("ijka->ijka", x22) * -0.5
    rdm2_f_vooo += einsum("ijka->aijk", x35) * -1
    rdm2_f_vooo += einsum("ijka->aikj", x35)
    del x35
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum("ia,ijkb->jkab", t1, x22)
    del x22
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum("ijab->ijab", x50) * 0.5
    del x50
    x23 = np.zeros((nocc, nvir), dtype=types[float])
    x23 += einsum("wxia,wxij->ja", u12, x21)
    x26 += einsum("ia->ia", x23)
    x27 += einsum("ij,ka->jika", delta_oo, x26) * 0.5
    rdm2_f_ovoo += einsum("ijka->iajk", x27) * -1
    rdm2_f_ovoo += einsum("ijka->iakj", x27)
    del x27
    rdm2_f_vooo += einsum("ij,ka->aijk", delta_oo, x26) * 0.5
    rdm2_f_vooo += einsum("ij,ka->aikj", delta_oo, x26) * -0.5
    del x26
    x53 += einsum("ia->ia", x23)
    del x23
    x54 += einsum("ia,jb->ijab", t1, x53) * 0.5
    del x53
    x56 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x56 += einsum("wia,wxij->xja", u11, x21)
    del x21
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum("wia,wjb->jiab", u11, x56)
    del x56
    x67 += einsum("ijab->ijab", x57)
    del x57
    x30 = np.zeros((nocc, nvir), dtype=types[float])
    x30 += einsum("w,wia->ia", ls1, u11)
    x33 += einsum("ia->ia", x30)
    x83 += einsum("ia->ia", x30) * -1
    x85 += einsum("ia->ia", x30) * -2
    del x30
    x31 = np.zeros((nocc, nvir), dtype=types[float])
    x31 += einsum("wx,wxia->ia", ls2, u12)
    x33 += einsum("ia->ia", x31) * 0.5
    x83 += einsum("ia->ia", x31) * -0.5
    x85 += einsum("ia->ia", x31) * -1
    del x31
    rdm2_f_vvoo += einsum("ia,jb->baij", t1, x85) * 0.5
    rdm2_f_vvoo += einsum("ia,jb->abij", t1, x85) * -0.5
    del x85
    x33 += einsum("ia->ia", t1)
    rdm2_f_ovoo += einsum("ij,ka->jaki", delta_oo, x33) * -1
    rdm2_f_ovoo += einsum("ij,ka->jaik", delta_oo, x33)
    rdm2_f_vooo += einsum("ij,ka->ajik", delta_oo, x33) * -1
    rdm2_f_vooo += einsum("ij,ka->ajki", delta_oo, x33)
    del x33
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x36 -= einsum("abij,kjca->ikbc", l2, t2)
    x72 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x72 += einsum("ijab,jkbc->ikac", t2, x36)
    x73 += einsum("ijab->ijab", x72)
    del x72
    x94 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x94 += einsum("ia,ijbc->jbac", t1, x36)
    x96 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x96 -= einsum("iabc->iabc", x94)
    del x94
    rdm2_f_ovov -= einsum("ijab->ibja", x36)
    rdm2_f_ovvo += einsum("ijab->ibaj", x36)
    rdm2_f_voov += einsum("ijab->bija", x36)
    rdm2_f_vovo -= einsum("ijab->biaj", x36)
    del x36
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum("wai,wjb->ijab", lu11, u11)
    rdm2_f_ovov -= einsum("ijab->ibja", x38)
    rdm2_f_ovvo += einsum("ijab->ibaj", x38)
    rdm2_f_voov += einsum("ijab->bija", x38)
    rdm2_f_vovo -= einsum("ijab->biaj", x38)
    del x38
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum("wxai,wxjb->ijab", lu12, u12)
    x97 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x97 += einsum("ia,ijbc->jbac", t1, x39)
    x98 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x98 += einsum("iabc->iabc", x97) * -0.5
    del x97
    rdm2_f_ovov += einsum("ijab->ibja", x39) * -0.5
    rdm2_f_ovvo += einsum("ijab->ibaj", x39) * 0.5
    rdm2_f_voov += einsum("ijab->bija", x39) * 0.5
    rdm2_f_vovo += einsum("ijab->biaj", x39) * -0.5
    del x39
    x40 = np.zeros((nvir, nvir), dtype=types[float])
    x40 += einsum("ai,ib->ab", l1, t1)
    x44 = np.zeros((nvir, nvir), dtype=types[float])
    x44 += einsum("ab->ab", x40) * 2
    x45 = np.zeros((nvir, nvir), dtype=types[float])
    x45 += einsum("ab->ab", x40)
    x95 = np.zeros((nvir, nvir), dtype=types[float])
    x95 += einsum("ab->ab", x40)
    del x40
    x41 = np.zeros((nvir, nvir), dtype=types[float])
    x41 += einsum("wai,wib->ab", lu11, u11)
    x44 += einsum("ab->ab", x41) * 2
    x45 += einsum("ab->ab", x41)
    x71 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x71 += einsum("ab,ijca->ijcb", x41, t2)
    x73 -= einsum("ijab->ijab", x71)
    del x71
    rdm2_f_vvoo -= einsum("ijab->abji", x73)
    rdm2_f_vvoo += einsum("ijab->baji", x73)
    del x73
    x95 += einsum("ab->ab", x41)
    del x41
    x96 += einsum("ia,bc->ibac", t1, x95)
    del x95
    x42 = np.zeros((nvir, nvir), dtype=types[float])
    x42 += einsum("wxai,wxib->ab", lu12, u12)
    x44 += einsum("ab->ab", x42)
    x45 += einsum("ab->ab", x42) * 0.5
    x46 = np.zeros((nvir, nvir), dtype=types[float])
    x46 += einsum("ab->ab", x42)
    del x42
    x43 = np.zeros((nvir, nvir), dtype=types[float])
    x43 -= einsum("abij,ijca->bc", l2, t2)
    x44 += einsum("ab->ab", x43)
    rdm2_f_ovov += einsum("ij,ab->jbia", delta_oo, x44) * 0.5
    rdm2_f_voov += einsum("ij,ab->bjia", delta_oo, x44) * -0.5
    del x44
    x45 += einsum("ab->ab", x43) * 0.5
    rdm2_f_ovvo += einsum("ij,ab->jbai", delta_oo, x45) * -1
    rdm2_f_vovo += einsum("ij,ab->bjai", delta_oo, x45)
    del x45
    x46 += einsum("ab->ab", x43)
    del x43
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x47 += einsum("ab,ijac->ijbc", x46, t2) * 0.5
    rdm2_f_vvoo += einsum("ijab->baij", x47)
    rdm2_f_vvoo += einsum("ijab->abij", x47) * -1
    del x47
    x98 += einsum("ia,bc->ibac", t1, x46) * 0.5
    del x46
    rdm2_f_vvov += einsum("iabc->bcia", x98)
    rdm2_f_vvov += einsum("iabc->cbia", x98) * -1
    rdm2_f_vvvo += einsum("iabc->bcai", x98) * -1
    rdm2_f_vvvo += einsum("iabc->cbai", x98)
    del x98
    x48 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x48 += einsum("wxai,jiba->wxjb", lu12, t2)
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum("wxia,wxjb->jiba", u12, x48)
    del x48
    x54 += einsum("ijab->ijab", x49) * -0.5
    del x49
    rdm2_f_vvoo += einsum("ijab->abij", x54) * -1
    rdm2_f_vvoo += einsum("ijab->baij", x54)
    rdm2_f_vvoo += einsum("ijab->abji", x54)
    rdm2_f_vvoo += einsum("ijab->baji", x54) * -1
    del x54
    x61 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x61 += einsum("wai,jiba->wjb", lu11, t2)
    x63 += einsum("wia->wia", x61)
    del x61
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum("wia,wjb->jiba", u11, x63)
    del x63
    x67 += einsum("ijab->ijab", x64)
    del x64
    rdm2_f_vvoo += einsum("ijab->abij", x67)
    rdm2_f_vvoo -= einsum("ijab->baij", x67)
    rdm2_f_vvoo -= einsum("ijab->abji", x67)
    rdm2_f_vvoo += einsum("ijab->baji", x67)
    del x67
    x68 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x68 += einsum("wx,xia->wia", ls2, u11)
    x69 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x69 += einsum("wia,wjb->jiba", u11, x68)
    del x68
    rdm2_f_vvoo -= einsum("ijab->baij", x69)
    rdm2_f_vvoo += einsum("ijab->baji", x69)
    del x69
    x83 += einsum("ia->ia", t1) * -1
    rdm2_f_vvoo += einsum("ia,jb->abji", t1, x83)
    rdm2_f_vvoo += einsum("ia,jb->baji", t1, x83) * -1
    del x83
    x86 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x86 += einsum("ia,bcji->jbca", t1, l2)
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv -= einsum("iabc->icba", x86)
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv += einsum("iabc->ciba", x86)
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv -= einsum("ia,ibcd->adcb", t1, x86)
    del x86
    x89 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x89 += einsum("ai,jibc->jabc", l1, t2)
    rdm2_f_vvov -= einsum("iabc->cbia", x89)
    rdm2_f_vvvo += einsum("iabc->cbai", x89)
    del x89
    x90 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x90 += einsum("ia,wbi->wba", t1, lu11)
    x91 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x91 += einsum("wia,wbc->ibca", u11, x90)
    del x90
    x96 -= einsum("iabc->iabc", x91)
    del x91
    x92 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x92 += einsum("wia,xwbi->xba", u11, lu12)
    x93 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x93 += einsum("wia,wbc->ibac", u11, x92)
    del x92
    x96 += einsum("iabc->iabc", x93)
    del x93
    rdm2_f_vvov += einsum("iabc->bcia", x96)
    rdm2_f_vvov -= einsum("iabc->cbia", x96)
    rdm2_f_vvvo -= einsum("iabc->bcai", x96)
    rdm2_f_vvvo += einsum("iabc->cbai", x96)
    del x96
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

def make_sing_b_dm(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, lu12=None, **kwargs):
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
    dm_b_des += einsum("wai,xwia->x", lu11, u12)
    dm_b_des += einsum("ai,wia->w", l1, u11)
    dm_b_des += einsum("w,xw->x", ls1, s2)
    dm_b_des += einsum("w->w", s1)

    dm_b = np.array([dm_b_cre, dm_b_des])

    return dm_b

def make_rdm1_b(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, lu12=None, **kwargs):
    # Get boson coupling creation array:
    gc = Namespace(
        boo=g.boo.transpose(0, 2, 1),
        bov=g.bvo.transpose(0, 2, 1),
        bvo=g.bov.transpose(0, 2, 1),
        bvv=g.bvv.transpose(0, 2, 1),
    )

    # Boson 1RDM
    rdm1_b = np.zeros((nbos, nbos), dtype=types[float])
    rdm1_b += einsum("w,x->wx", ls1, s1)
    rdm1_b += einsum("wai,xia->wx", lu11, u11)
    rdm1_b += einsum("wx,yx->wy", ls2, s2)
    rdm1_b += einsum("wxai,yxia->wy", lu12, u12)

    return rdm1_b

def make_eb_coup_rdm(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, l1=None, l2=None, ls1=None, ls2=None, lu11=None, lu12=None, **kwargs):
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
    x0 += einsum("wia,xwaj->xji", u11, lu12)
    x4 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x4 += einsum("wij->wij", x0)
    x8 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x8 += einsum("wx,xij->wij", s2, x0)
    x27 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x27 += einsum("wij->wij", x8)
    rdm_eb_des_oo = np.zeros((nbos, nocc, nocc), dtype=types[float])
    rdm_eb_des_oo -= einsum("wij->wji", x8)
    del x8
    x31 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x31 += einsum("wij->wij", x0)
    rdm_eb_cre_oo = np.zeros((nbos, nocc, nocc), dtype=types[float])
    rdm_eb_cre_oo -= einsum("wij->wji", x0)
    del x0
    x1 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x1 += einsum("ia,waj->wji", t1, lu11)
    x4 += einsum("wij->wij", x1)
    rdm_eb_cre_ov = np.zeros((nbos, nocc, nvir), dtype=types[float])
    rdm_eb_cre_ov -= einsum("ia,wij->wja", t1, x4)
    rdm_eb_des_ov = np.zeros((nbos, nocc, nvir), dtype=types[float])
    rdm_eb_des_ov -= einsum("wij,wxia->xja", x4, u12)
    del x4
    x31 += einsum("wij->wij", x1)
    x33 = np.zeros((nocc, nvir), dtype=types[float])
    x33 += einsum("wia,wij->ja", u11, x31)
    del x31
    rdm_eb_cre_oo -= einsum("wij->wji", x1)
    del x1
    x2 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x2 += einsum("ia,wxaj->wxji", t1, lu12)
    x3 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x3 += einsum("wia,xwij->xja", u11, x2)
    rdm_eb_cre_ov -= einsum("wia->wia", x3)
    rdm_eb_des_ov -= einsum("wx,xia->wia", s2, x3)
    del x3
    x33 += einsum("wxia,wxij->ja", u12, x2) * 0.5
    del x2
    x5 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x5 += einsum("wia,xwbi->xba", u11, lu12)
    rdm_eb_cre_vv = np.zeros((nbos, nvir, nvir), dtype=types[float])
    rdm_eb_cre_vv += einsum("wab->wab", x5)
    rdm_eb_des_ov -= einsum("wab,xwia->xib", x5, u12)
    rdm_eb_des_vv = np.zeros((nbos, nvir, nvir), dtype=types[float])
    rdm_eb_des_vv += einsum("wx,xab->wab", s2, x5)
    del x5
    x6 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x6 += einsum("wai,xwja->xij", lu11, u12)
    x27 += einsum("wij->wij", x6)
    rdm_eb_des_oo -= einsum("wij->wji", x6)
    del x6
    x7 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x7 += einsum("ai,wja->wij", l1, u11)
    x27 += einsum("wij->wij", x7)
    rdm_eb_des_oo -= einsum("wij->wji", x7)
    del x7
    x9 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x9 += einsum("wx,xai->wia", s2, lu11)
    x11 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x11 += einsum("wia->wia", x9)
    rdm_eb_des_vo = np.zeros((nbos, nvir, nocc), dtype=types[float])
    rdm_eb_des_vo += einsum("wia->wai", x9)
    del x9
    x10 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x10 -= einsum("wia,abji->wjb", u11, l2)
    x11 += einsum("wia->wia", x10)
    x12 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x12 += einsum("ia,wja->wij", t1, x11)
    x27 += einsum("wij->wji", x12)
    rdm_eb_des_ov -= einsum("ia,wij->wja", t1, x27)
    del x27
    rdm_eb_des_oo -= einsum("wij->wij", x12)
    del x12
    rdm_eb_des_ov += einsum("wia,ijab->wjb", x11, t2)
    rdm_eb_des_vv += einsum("ia,wib->wba", t1, x11)
    del x11
    rdm_eb_des_vo += einsum("wia->wai", x10)
    del x10
    x13 = np.zeros((nocc, nocc), dtype=types[float])
    x13 += einsum("ai,ja->ij", l1, t1)
    x17 = np.zeros((nocc, nocc), dtype=types[float])
    x17 += einsum("ij->ij", x13)
    x28 = np.zeros((nocc, nocc), dtype=types[float])
    x28 += einsum("ij->ij", x13)
    x32 = np.zeros((nocc, nocc), dtype=types[float])
    x32 += einsum("ij->ij", x13) * 2
    del x13
    x14 = np.zeros((nocc, nocc), dtype=types[float])
    x14 += einsum("wai,wja->ij", lu11, u11)
    x17 += einsum("ij->ij", x14)
    x28 += einsum("ij->ij", x14)
    x32 += einsum("ij->ij", x14) * 2
    del x14
    x15 = np.zeros((nocc, nocc), dtype=types[float])
    x15 += einsum("wxai,wxja->ij", lu12, u12)
    x17 += einsum("ij->ij", x15) * 0.5
    x28 += einsum("ij->ij", x15) * 0.5
    x32 += einsum("ij->ij", x15)
    del x15
    x16 = np.zeros((nocc, nocc), dtype=types[float])
    x16 += einsum("abij,kjab->ik", l2, t2)
    x17 += einsum("ij->ij", x16) * 0.5
    x28 += einsum("ij->ij", x16) * 0.5
    rdm_eb_des_ov += einsum("ij,wia->wja", x28, u11) * -1
    del x28
    x32 += einsum("ij->ij", x16)
    del x16
    x33 += einsum("ia,ij->ja", t1, x32) * 0.5
    del x32
    x17 += einsum("ij->ji", delta_oo) * -1
    rdm_eb_des_oo += einsum("w,ij->wji", s1, x17) * -1
    del x17
    x18 = np.zeros((nbos), dtype=types[float])
    x18 += einsum("w,xw->x", ls1, s2)
    x21 = np.zeros((nbos), dtype=types[float])
    x21 += einsum("w->w", x18)
    del x18
    x19 = np.zeros((nbos), dtype=types[float])
    x19 += einsum("ai,wia->w", l1, u11)
    x21 += einsum("w->w", x19)
    del x19
    x20 = np.zeros((nbos), dtype=types[float])
    x20 += einsum("wai,xwia->x", lu11, u12)
    x21 += einsum("w->w", x20)
    del x20
    rdm_eb_des_oo += einsum("w,ij->wji", x21, delta_oo)
    rdm_eb_des_ov += einsum("w,ia->wia", x21, t1)
    del x21
    x22 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x22 += einsum("wia,xyai->xyw", u11, lu12)
    rdm_eb_des_ov += einsum("wxy,wxia->yia", x22, u12) * 0.5
    del x22
    x23 = np.zeros((nvir, nvir), dtype=types[float])
    x23 += einsum("wai,wib->ab", lu11, u11)
    x26 = np.zeros((nvir, nvir), dtype=types[float])
    x26 += einsum("ab->ab", x23)
    x34 = np.zeros((nvir, nvir), dtype=types[float])
    x34 += einsum("ab->ab", x23)
    del x23
    x24 = np.zeros((nvir, nvir), dtype=types[float])
    x24 += einsum("wxai,wxib->ab", lu12, u12)
    x26 += einsum("ab->ab", x24) * 0.5
    x34 += einsum("ab->ab", x24) * 0.5
    del x24
    x25 = np.zeros((nvir, nvir), dtype=types[float])
    x25 -= einsum("abij,ijca->bc", l2, t2)
    x26 += einsum("ab->ab", x25) * 0.5
    rdm_eb_des_ov += einsum("ab,wia->wib", x26, u11) * -1
    del x26
    x34 += einsum("ab->ab", x25) * 0.5
    del x25
    x29 = np.zeros((nbos, nbos), dtype=types[float])
    x29 += einsum("wx,yw->xy", ls2, s2)
    x29 += einsum("wai,xia->wx", lu11, u11)
    x29 += einsum("wxai,ywia->xy", lu12, u12)
    rdm_eb_des_ov += einsum("wx,wia->xia", x29, u11)
    del x29
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x30 += einsum("ia,bajk->jkib", t1, l2)
    x33 += einsum("ijab,ijkb->ka", t2, x30) * -0.5000000000000003
    del x30
    x33 += einsum("ia->ia", t1) * -1
    x33 += einsum("w,wia->ia", ls1, u11) * -1
    x33 += einsum("wx,wxia->ia", ls2, u12) * -0.5
    x33 += einsum("ai,jiba->jb", l1, t2) * -1
    rdm_eb_des_ov += einsum("w,ia->wia", s1, x33) * -1
    del x33
    x34 += einsum("ai,ib->ab", l1, t1)
    rdm_eb_des_vv += einsum("w,ab->wab", s1, x34)
    del x34
    rdm_eb_cre_oo += einsum("w,ij->wji", ls1, delta_oo)
    rdm_eb_cre_ov += einsum("w,ia->wia", ls1, t1)
    rdm_eb_cre_ov += einsum("wx,xia->wia", ls2, u11)
    rdm_eb_cre_ov += einsum("wai,jiba->wjb", lu11, t2)
    rdm_eb_cre_vo = np.zeros((nbos, nvir, nocc), dtype=types[float])
    rdm_eb_cre_vo += einsum("wai->wai", lu11)
    rdm_eb_cre_vv += einsum("ia,wbi->wba", t1, lu11)
    rdm_eb_des_ov += einsum("wia->wia", u11)
    rdm_eb_des_ov += einsum("w,xwia->xia", ls1, u12)
    rdm_eb_des_vo += einsum("w,ai->wai", s1, l1)
    rdm_eb_des_vv += einsum("ai,wib->wab", l1, u11)
    rdm_eb_des_vv += einsum("wai,xwib->xab", lu11, u12)

    rdm_eb = np.array([
            np.block([[rdm_eb_cre_oo, rdm_eb_cre_ov], [rdm_eb_cre_vo, rdm_eb_cre_vv]]),
            np.block([[rdm_eb_des_oo, rdm_eb_des_ov], [rdm_eb_des_vo, rdm_eb_des_vv]]),
    ])

    return rdm_eb

