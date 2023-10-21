# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, **kwargs):
    # Energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum("iajb->jiab", v.ovov) * -0.5
    x0 += einsum("iajb->jiba", v.ovov)
    x1 = np.zeros((nocc, nvir), dtype=types[float])
    x1 += einsum("ia,ijab->jb", t1, x0)
    e_cc = 0
    e_cc += einsum("ijab,ijab->", t2, x0) * 2
    del x0
    x1 += einsum("ia->ia", f.ov)
    e_cc += einsum("ia,ia->", t1, x1) * 2
    del x1
    x2 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x2 += einsum("wia->wia", u11)
    x2 += einsum("w,ia->wia", s1, t1)
    e_cc += einsum("wia,wia->", g.bov, x2) * 2
    del x2
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
    x0 += einsum("ia,jbka->ijkb", t1, v.ovov)
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum("ijka->ijka", x0) * 2
    x1 += einsum("ijka->ikja", x0) * -1
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum("ijab,kjla->kilb", t2, x0)
    x40 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x40 -= einsum("ijka->ikja", x34)
    del x34
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x35 -= einsum("ijka->ijka", x0)
    x35 += einsum("ijka->ikja", x0) * 2
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x36 += einsum("ijab,klia->jklb", t2, x35)
    del x35
    x40 += einsum("ijka->jkia", x36)
    del x36
    x64 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x64 += einsum("ijab,klja->kilb", t2, x0)
    x67 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x67 -= einsum("ijka->kija", x64)
    del x64
    x86 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x86 += einsum("ia,jkla->jilk", t1, x0)
    del x0
    x87 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x87 += einsum("ijkl->lkji", x86)
    x88 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x88 += einsum("ijkl->lkji", x86)
    del x86
    x1 += einsum("ijka->jika", v.ooov) * -1
    x1 += einsum("ijka->jkia", v.ooov) * 2
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum("ijab,kjib->ka", t2, x1) * -1
    del x1
    x2 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x2 += einsum("iabc->ibac", v.ovvv) * -0.5
    x2 += einsum("iabc->ibca", v.ovvv)
    t1new += einsum("ijab,icba->jc", t2, x2) * 2
    del x2
    x3 = np.zeros((nocc, nvir), dtype=types[float])
    x3 += einsum("w,wia->ia", s1, g.bov)
    x6 = np.zeros((nocc, nvir), dtype=types[float])
    x6 += einsum("ia->ia", x3)
    x18 = np.zeros((nocc, nvir), dtype=types[float])
    x18 += einsum("ia->ia", x3) * 0.5
    x104 = np.zeros((nocc, nvir), dtype=types[float])
    x104 += einsum("ia->ia", x3)
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum("iajb->jiab", v.ovov) * 2
    x4 -= einsum("iajb->jiba", v.ovov)
    x5 = np.zeros((nocc, nvir), dtype=types[float])
    x5 += einsum("ia,ijba->jb", t1, x4)
    x6 += einsum("ia->ia", x5)
    del x5
    x97 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x97 += einsum("wia,ijba->wjb", u11, x4)
    del x4
    x99 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x99 += einsum("wia->wia", x97)
    s2new = np.zeros((nbos, nbos), dtype=types[float])
    s2new += einsum("wia,xia->xw", u11, x97) * 2
    del x97
    x6 += einsum("ia->ia", f.ov)
    x66 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x66 += einsum("ia,jkab->jkib", x6, t2)
    x67 += einsum("ijka->kjia", x66)
    del x66
    x69 = np.zeros((nocc, nocc), dtype=types[float])
    x69 += einsum("ia,ja->ij", t1, x6)
    x70 = np.zeros((nocc, nocc), dtype=types[float])
    x70 += einsum("ij->ij", x69)
    del x69
    s1new = np.zeros((nbos), dtype=types[float])
    s1new += einsum("ia,wia->w", x6, u11) * 2
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum("ijab->jiab", t2) * 2
    x7 -= einsum("ijab->jiba", t2)
    x46 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x46 += einsum("wia,ijba->wjb", g.bov, x7)
    x48 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x48 += einsum("wia->wia", x46)
    del x46
    t1new += einsum("ia,ijba->jb", x6, x7)
    del x6
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum("iabj->ijba", v.ovvo) * 2
    x8 -= einsum("ijab->ijab", v.oovv)
    t1new += einsum("ia,ijba->jb", t1, x8)
    u11new = np.zeros((nbos, nocc, nvir), dtype=types[float])
    u11new += einsum("wia,ijba->wjb", u11, x8)
    del x8
    x9 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x9 += einsum("ia,wja->wji", t1, g.bov)
    x10 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x10 += einsum("wij->wij", x9)
    del x9
    x10 += einsum("wij->wij", g.boo)
    x47 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x47 += einsum("ia,wij->wja", t1, x10)
    x48 -= einsum("wia->wia", x47)
    del x47
    t1new -= einsum("wia,wij->ja", u11, x10)
    del x10
    x11 = np.zeros((nocc, nocc), dtype=types[float])
    x11 += einsum("w,wij->ij", s1, g.boo)
    x20 = np.zeros((nocc, nocc), dtype=types[float])
    x20 += einsum("ij->ij", x11)
    x70 += einsum("ij->ji", x11)
    del x11
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum("wia,wja->ij", g.bov, u11)
    x20 += einsum("ij->ij", x12)
    x52 = np.zeros((nocc, nocc), dtype=types[float])
    x52 += einsum("ij->ij", x12)
    del x12
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum("iajb->jiab", v.ovov)
    x13 += einsum("iajb->jiba", v.ovov) * -0.5
    x14 = np.zeros((nocc, nocc), dtype=types[float])
    x14 += einsum("ijab,ikba->jk", t2, x13) * 2
    x20 += einsum("ij->ji", x14)
    del x14
    x17 = np.zeros((nocc, nvir), dtype=types[float])
    x17 += einsum("ia,ijba->jb", t1, x13)
    x18 += einsum("ia->ia", x17)
    del x17
    x78 = np.zeros((nvir, nvir), dtype=types[float])
    x78 += einsum("ijab,ijbc->ac", t2, x13)
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum("ab,ijbc->ijca", x78, t2) * 2
    del x78
    x83 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x83 += einsum("ijab->jiab", x79)
    del x79
    x102 = np.zeros((nocc, nvir), dtype=types[float])
    x102 += einsum("ia,ijba->jb", t1, x13) * 2
    del x13
    x103 = np.zeros((nvir, nvir), dtype=types[float])
    x103 += einsum("ia,ib->ab", t1, x102) * -1
    del x102
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum("ijka->ikja", v.ooov)
    x15 += einsum("ijka->kija", v.ooov) * -0.5
    x16 = np.zeros((nocc, nocc), dtype=types[float])
    x16 += einsum("ia,jika->jk", t1, x15) * 2
    del x15
    x20 += einsum("ij->ij", x16)
    del x16
    x18 += einsum("ia->ia", f.ov) * 0.5
    x19 = np.zeros((nocc, nocc), dtype=types[float])
    x19 += einsum("ia,ja->ij", t1, x18) * 2
    del x18
    x20 += einsum("ij->ji", x19)
    del x19
    x20 += einsum("ij->ij", f.oo)
    t1new += einsum("ia,ij->ja", t1, x20) * -1
    u11new += einsum("ij,wia->wja", x20, u11) * -1
    del x20
    x21 = np.zeros((nvir, nvir), dtype=types[float])
    x21 += einsum("w,wab->ab", s1, g.bvv)
    x24 = np.zeros((nvir, nvir), dtype=types[float])
    x24 += einsum("ab->ab", x21)
    x72 = np.zeros((nvir, nvir), dtype=types[float])
    x72 += einsum("ab->ab", x21)
    x103 += einsum("ab->ab", x21)
    del x21
    x22 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x22 += einsum("iabc->ibac", v.ovvv) * 2
    x22 -= einsum("iabc->ibca", v.ovvv)
    x23 = np.zeros((nvir, nvir), dtype=types[float])
    x23 += einsum("ia,ibac->bc", t1, x22)
    x24 += einsum("ab->ab", x23)
    x43 = np.zeros((nvir, nvir), dtype=types[float])
    x43 -= einsum("ab->ab", x23)
    del x23
    x106 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x106 += einsum("wia,ibac->wbc", u11, x22)
    del x22
    x24 += einsum("ab->ab", f.vv)
    t1new += einsum("ia,ba->ib", t1, x24)
    del x24
    x25 = np.zeros((nbos), dtype=types[float])
    x25 += einsum("ia,wia->w", t1, g.bov)
    x26 = np.zeros((nbos), dtype=types[float])
    x26 += einsum("w->w", x25) * 2
    del x25
    x26 += einsum("w->w", G)
    t1new += einsum("w,wia->ia", x26, u11)
    s1new += einsum("w,wx->x", x26, s2)
    del x26
    x27 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x27 += einsum("ia,bcda->ibdc", t1, v.vvvv)
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum("ia,jbca->ijbc", t1, x27)
    del x27
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum("ia,jabc->ijbc", t1, v.ovvv)
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum("ijab,kjca->kibc", t2, x28)
    del x28
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum("ijab->ijab", x29)
    del x29
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 -= einsum("iajb->jiab", v.ovov)
    x30 += einsum("iajb->jiba", v.ovov) * 2
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 += einsum("ijab,ikbc->jkac", t2, x30)
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum("ijab,kica->jkbc", t2, x31)
    del x31
    x54 += einsum("ijab->ijab", x32)
    del x32
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum("ijab,ikac->jkbc", t2, x30) * 2
    del x30
    x33 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x33 += einsum("ijab,jkla->ilkb", t2, v.ooov)
    x40 -= einsum("ijka->ijka", x33)
    del x33
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum("ia,jbca->ijcb", t1, v.ovvv)
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum("ijab->jiab", x37)
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum("ijab,kjca->kibc", t2, x37)
    del x37
    x74 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x74 += einsum("ijab->ijab", x56)
    del x56
    x38 += einsum("iabj->ijba", v.ovvo)
    x39 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x39 += einsum("ia,jkba->ijkb", t1, x38)
    del x38
    x40 += einsum("ijka->ijka", x39)
    del x39
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum("ia,jikb->jkab", t1, x40)
    del x40
    x54 += einsum("ijab->ijab", x41)
    del x41
    x42 = np.zeros((nvir, nvir), dtype=types[float])
    x42 += einsum("wia,wib->ab", g.bov, u11)
    x43 += einsum("ab->ba", x42)
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum("ab,ijbc->ijca", x43, t2)
    del x43
    x54 += einsum("ijab->jiab", x44)
    del x44
    x103 += einsum("ab->ba", x42) * -1
    del x42
    x45 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x45 += einsum("ia,wba->wib", t1, g.bvv)
    x48 += einsum("wia->wia", x45)
    del x45
    x48 += einsum("wai->wia", g.bvo)
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum("wia,wjb->ijab", u11, x48)
    del x48
    x54 -= einsum("ijab->jiba", x49)
    del x49
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x50 += einsum("ijka->ikja", v.ooov) * 2
    x50 -= einsum("ijka->kija", v.ooov)
    x51 = np.zeros((nocc, nocc), dtype=types[float])
    x51 += einsum("ia,jika->jk", t1, x50)
    x52 += einsum("ij->ij", x51)
    del x51
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x53 += einsum("ij,ikab->kjab", x52, t2)
    del x52
    x54 += einsum("ijab->ijba", x53)
    del x53
    t2new -= einsum("ijab->ijab", x54)
    t2new -= einsum("ijab->jiba", x54)
    del x54
    x65 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x65 += einsum("ijab,kila->jklb", t2, x50)
    x67 += einsum("ijka->jika", x65)
    del x65
    x105 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x105 += einsum("wia,jika->wjk", u11, x50)
    del x50
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum("ia,bjca->ijbc", t1, v.vovv)
    x74 -= einsum("ijab->ijab", x55)
    del x55
    x57 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x57 -= einsum("iabc->ibac", v.ovvv)
    x57 += einsum("iabc->ibca", v.ovvv) * 2
    x58 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x58 += einsum("ia,jbac->ijbc", t1, x57)
    del x57
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum("ijab,kica->jkbc", t2, x58)
    del x58
    x74 -= einsum("ijab->jiab", x59)
    del x59
    x60 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x60 += einsum("ia,jkba->ijkb", t1, v.oovv)
    x67 += einsum("ijka->jika", x60)
    del x60
    x61 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x61 += einsum("ijab,klja->iklb", t2, v.ooov)
    x67 -= einsum("ijka->jika", x61)
    del x61
    x62 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x62 += einsum("ia,jkla->ijlk", t1, v.ooov)
    x63 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x63 += einsum("ia,jkil->jkla", t1, x62)
    x67 -= einsum("ijka->jika", x63)
    del x63
    x68 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x68 += einsum("ia,ijkb->jkab", t1, x67)
    del x67
    x74 += einsum("ijab->ijab", x68)
    del x68
    x75 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum("ijab,kijl->klab", t2, x62)
    del x62
    t2new += einsum("ijab->ijba", x75)
    t2new += einsum("ijab->jiab", x75)
    del x75
    x70 += einsum("ij->ji", f.oo)
    x71 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x71 += einsum("ij,jkab->kiab", x70, t2)
    del x70
    x74 += einsum("ijab->jiba", x71)
    del x71
    x72 += einsum("ab->ab", f.vv)
    x73 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum("ab,ijbc->ijca", x72, t2)
    del x72
    x74 -= einsum("ijab->jiba", x73)
    del x73
    t2new -= einsum("ijab->ijba", x74)
    t2new -= einsum("ijab->jiab", x74)
    del x74
    x76 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x76 += einsum("ijab,kbca->jikc", t2, v.ovvv)
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum("ia,jkib->jkab", t1, x76)
    del x76
    x83 += einsum("ijab->ijab", x77)
    del x77
    x80 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x80 += einsum("iajb->jiab", v.ovov) * -0.5
    x80 += einsum("iajb->jiba", v.ovov)
    x81 = np.zeros((nocc, nocc), dtype=types[float])
    x81 += einsum("ijab,ikab->jk", t2, x80)
    del x80
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum("ij,jkab->kiab", x81, t2) * 2
    del x81
    x83 += einsum("ijab->ijba", x82)
    del x82
    t2new += einsum("ijab->ijab", x83) * -1
    t2new += einsum("ijab->jiba", x83) * -1
    del x83
    x84 += einsum("iabj->jiba", v.ovvo) * 2
    x84 -= einsum("ijab->jiab", v.oovv)
    t2new += einsum("ijab,kica->kjcb", t2, x84)
    del x84
    x85 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x85 += einsum("ijab,kbla->ijlk", t2, v.ovov)
    x87 += einsum("ijkl->lkji", x85) * 0.9999999999999993
    x88 += einsum("ijkl->lkji", x85)
    del x85
    x87 += einsum("ijkl->kilj", v.oooo) * 0.9999999999999993
    t2new += einsum("ijab,ijkl->klab", t2, x87)
    del x87
    x88 += einsum("ijkl->kilj", v.oooo)
    x89 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x89 += einsum("ia,ijkl->lkja", t1, x88)
    del x88
    x89 += einsum("ijak->jkia", v.oovo) * -1
    t2new += einsum("ia,jkib->jkab", t1, x89)
    del x89
    x90 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x90 -= einsum("iabj->jiba", v.ovvo)
    x90 += einsum("ijab,jakc->ikbc", t2, v.ovov)
    t2new += einsum("ijab,kicb->jkac", t2, x90)
    del x90
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x91 -= einsum("ijab->jiab", v.oovv)
    x91 += einsum("ijab,jcka->ikbc", t2, v.ovov)
    t2new += einsum("ijab,kicb->jkca", t2, x91)
    del x91
    x92 = np.zeros((nbos, nbos), dtype=types[float])
    x92 += einsum("wia,xia->wx", gc.bov, u11)
    x96 = np.zeros((nbos, nbos), dtype=types[float])
    x96 += einsum("wx->wx", x92) * 2
    del x92
    x93 = np.zeros((nbos, nbos), dtype=types[float])
    x93 += einsum("wia,xia->wx", g.bov, u11)
    x94 = np.zeros((nbos, nbos), dtype=types[float])
    x94 += einsum("wx->wx", x93) * 2
    del x93
    x94 += einsum("wx->wx", w)
    x95 = np.zeros((nbos, nbos), dtype=types[float])
    x95 += einsum("wx,wy->xy", s2, x94)
    x96 += einsum("wx->wx", x95)
    del x95
    s2new += einsum("wx->wx", x96)
    s2new += einsum("wx->xw", x96)
    del x96
    u11new += einsum("wx,wia->xia", x94, u11)
    del x94
    x98 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x98 += einsum("wx,xia->wia", s2, g.bov)
    x99 += einsum("wia->wia", x98)
    del x98
    x99 += einsum("wia->wia", gc.bov)
    x105 += einsum("ia,wja->wji", t1, x99)
    u11new += einsum("wia,ijba->wjb", x99, x7)
    del x99
    del x7
    x100 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x100 += einsum("iajb->jiab", v.ovov) * -1
    x100 += einsum("iajb->jiba", v.ovov) * 2
    x103 += einsum("ijab,ijcb->ac", t2, x100) * -1
    del x100
    x101 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x101 += einsum("iabc->ibac", v.ovvv)
    x101 += einsum("iabc->ibca", v.ovvv) * -0.5
    x103 += einsum("ia,ibac->bc", t1, x101) * 2
    del x101
    x103 += einsum("ab->ab", f.vv)
    u11new += einsum("ab,wib->wia", x103, u11)
    del x103
    x104 += einsum("ia->ia", f.ov)
    x105 += einsum("ia,wja->wij", x104, u11)
    del x104
    x105 += einsum("wij->wij", gc.boo)
    x105 += einsum("wx,xij->wij", s2, g.boo)
    u11new -= einsum("ia,wij->wja", t1, x105)
    del x105
    x106 += einsum("wab->wab", gc.bvv)
    x106 += einsum("wx,xab->wab", s2, g.bvv)
    u11new += einsum("ia,wba->wib", t1, x106)
    del x106
    t1new += einsum("w,wai->ia", s1, g.bvo)
    t1new += einsum("ai->ia", f.vo)
    t1new += einsum("wab,wib->ia", g.bvv, u11)
    t2new -= einsum("ijab,jack->kicb", t2, v.ovvo)
    t2new -= einsum("ijab,jkca->kibc", t2, v.oovv)
    t2new -= einsum("ijab,jkcb->ikac", t2, v.oovv)
    t2new += einsum("ijab,jbck->ikac", t2, v.ovvo) * 2
    t2new += einsum("ijab,cadb->jidc", t2, v.vvvv)
    t2new += einsum("aibj->jiba", v.vovo)
    t2new -= einsum("ia,ijbk->kjba", t1, v.oovo)
    s1new += einsum("ia,wia->w", t1, gc.bov) * 2
    s1new += einsum("w,wx->x", s1, w)
    s1new += einsum("w->w", G)
    u11new += einsum("wx,xai->wia", s2, g.bvo)
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
    x0 = np.zeros((nocc, nvir), dtype=types[float])
    x0 += einsum("w,wia->ia", s1, g.bov)
    x1 = np.zeros((nocc, nvir), dtype=types[float])
    x1 += einsum("ia->ia", x0)
    x57 = np.zeros((nocc, nvir), dtype=types[float])
    x57 += einsum("ia->ia", x0)
    x67 = np.zeros((nocc, nvir), dtype=types[float])
    x67 += einsum("ia->ia", x0) * 0.5
    x100 = np.zeros((nocc, nvir), dtype=types[float])
    x100 += einsum("ia->ia", x0)
    x136 = np.zeros((nocc, nvir), dtype=types[float])
    x136 += einsum("ia->ia", x0)
    del x0
    x1 += einsum("ia->ia", f.ov)
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x2 += einsum("ia,jkab->jkib", x1, t2) * 0.25
    del x1
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x20 += einsum("ijka->jkia", x2)
    x20 += einsum("ijka->ikja", x2) * -2
    del x2
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x3 += einsum("ijab,kacb->ijkc", t2, v.ovvv)
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum("ijka->ijka", x3) * -0.5
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum("ia,jabc->ijbc", t1, v.ovvv)
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x5 += einsum("ia,jkba->jikb", t1, x4)
    x11 += einsum("ijka->ijka", x5) * -0.5
    del x5
    x131 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x131 += einsum("ijab->ijab", x4)
    del x4
    x6 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x6 += einsum("ijab,kbla->ijlk", t2, v.ovov)
    x9 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x9 += einsum("ijkl->jilk", x6)
    x178 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x178 += einsum("ijkl->lkji", x6)
    del x6
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x7 += einsum("ia,jakb->ikjb", t1, v.ovov)
    x8 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x8 += einsum("ia,jkla->ijkl", t1, x7)
    x9 += einsum("ijkl->ijkl", x8)
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum("ia,jkil->jkla", t1, x9) * 0.5
    del x9
    x11 += einsum("ijka->jika", x10)
    del x10
    x20 += einsum("ijka->ikja", x11)
    x20 += einsum("ijka->jkia", x11) * -0.5
    del x11
    x178 += einsum("ijkl->lkji", x8)
    del x8
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum("ijka->ijka", x7) * -1
    x15 += einsum("ijka->ikja", x7) * 2
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x24 += einsum("ijka->ijka", x7) * 2
    x24 -= einsum("ijka->ikja", x7)
    x25 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x25 -= einsum("ijka->ijka", x7)
    x25 += einsum("ijka->ikja", x7) * 2
    x55 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x55 += einsum("ijka->jkia", x7) * -1
    x55 += einsum("ijka->kjia", x7) * 2
    x125 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x125 += einsum("ai,ijkb->jkab", l1, x7)
    x143 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x143 += einsum("ijab->ijab", x125)
    del x125
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum("iajb->jiab", v.ovov)
    x12 += einsum("iajb->jiba", v.ovov) * -0.5
    x13 = np.zeros((nocc, nvir), dtype=types[float])
    x13 += einsum("ia,ijba->jb", t1, x12)
    x14 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x14 += einsum("ia,jkab->jkib", x13, t2) * 0.5
    x20 += einsum("ijka->jkia", x14)
    x20 += einsum("ijka->ikja", x14) * -2
    del x14
    x67 += einsum("ia->ia", x13)
    del x13
    x56 = np.zeros((nocc, nvir), dtype=types[float])
    x56 += einsum("ia,ijba->jb", t1, x12) * 2
    x57 += einsum("ia->ia", x56)
    del x56
    x175 = np.zeros((nocc, nocc), dtype=types[float])
    x175 += einsum("ijab,ikba->jk", t2, x12)
    del x12
    x176 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x176 += einsum("ij,abik->kjab", x175, l2) * 2
    del x175
    x177 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x177 += einsum("ijab->ijba", x176)
    del x176
    x15 += einsum("ijka->jika", v.ooov) * 2
    x15 += einsum("ijka->jkia", v.ooov) * -1
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum("ijab->jiab", t2)
    x16 += einsum("ijab->jiba", t2) * -0.5
    x20 += einsum("ijka,klba->ijlb", x15, x16) * -0.5
    del x15
    x33 = np.zeros((nocc, nocc), dtype=types[float])
    x33 += einsum("abij,ikba->jk", l2, x16) * 2
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum("ia,jk->jkia", t1, x33) * -0.5
    x95 = np.zeros((nocc, nocc), dtype=types[float])
    x95 += einsum("ij->ij", x33)
    x108 = np.zeros((nocc, nocc), dtype=types[float])
    x108 += einsum("ij->ij", x33)
    del x33
    x76 = np.zeros((nocc, nvir), dtype=types[float])
    x76 += einsum("iabc,ijca->jb", v.ovvv, x16)
    x86 = np.zeros((nocc, nocc), dtype=types[float])
    x86 += einsum("abij,ikba->jk", l2, x16)
    x87 = np.zeros((nocc, nocc), dtype=types[float])
    x87 += einsum("ij->ij", x86)
    x171 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x171 += einsum("ij,jakb->kiab", x86, v.ovov) * 2
    del x86
    x172 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x172 += einsum("ijab->jiba", x171)
    del x171
    x92 = np.zeros((nvir, nvir), dtype=types[float])
    x92 += einsum("abij,ijca->bc", l2, x16)
    x93 = np.zeros((nvir, nvir), dtype=types[float])
    x93 += einsum("ab->ab", x92)
    x170 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x170 += einsum("ab,ibjc->ijca", x92, v.ovov) * 2
    x172 += einsum("ijab->jiba", x170)
    del x170
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum("ijab->baij", x172) * -1
    l2new += einsum("ijab->abji", x172) * -1
    del x172
    x180 = np.zeros((nvir, nvir), dtype=types[float])
    x180 += einsum("ab->ab", x92)
    del x92
    x179 = np.zeros((nocc, nvir), dtype=types[float])
    x179 += einsum("iabc,ijca->jb", v.ovvv, x16) * 2
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum("iabj->ijba", v.ovvo) * 2
    x17 += einsum("ijab->ijab", v.oovv) * -1
    x20 += einsum("ia,jkba->ijkb", t1, x17) * -0.25
    del x17
    x18 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x18 += einsum("ia,jkla->ijlk", t1, v.ooov)
    x19 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x19 += einsum("ijkl->jkli", x18)
    x19 += einsum("ijkl->kjli", x18) * -0.5
    x27 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x27 -= einsum("ijkl->ijkl", x18)
    x27 += einsum("ijkl->ikjl", x18) * 2
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum("ia,jikl->jkla", t1, x27)
    del x27
    x128 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x128 += einsum("abij,jkli->klba", l2, x18)
    del x18
    x143 -= einsum("ijab->ijab", x128)
    del x128
    x19 += einsum("ijkl->kijl", v.oooo) * -0.5
    x19 += einsum("ijkl->kilj", v.oooo)
    x20 += einsum("ia,ijkl->ljka", t1, x19) * 0.5
    del x19
    x20 += einsum("ijak->jika", v.oovo) * -0.5
    x20 += einsum("ijak->kija", v.oovo) * 0.25
    l1new = np.zeros((nvir, nocc), dtype=types[float])
    l1new += einsum("abij,jkia->bk", l2, x20) * 4
    del x20
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 -= einsum("ijab->jiab", t2)
    x21 += einsum("ijab->jiba", t2) * 2
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 -= einsum("abij,ikac->jkbc", l2, x21)
    del x21
    x22 += einsum("abij,jkac->ikbc", l2, t2)
    x23 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x23 -= einsum("iabc->ibca", v.ovvv)
    x23 += einsum("iabc->ibac", v.ovvv) * 2
    l1new -= einsum("ijab,jabc->ci", x22, x23)
    del x22
    del x23
    x24 -= einsum("ijka->jika", v.ooov)
    x24 += einsum("ijka->jkia", v.ooov) * 2
    x28 += einsum("ijab,kila->kljb", t2, x24)
    del x24
    x25 += einsum("ijka->jika", v.ooov) * 2
    x25 -= einsum("ijka->jkia", v.ooov)
    x28 += einsum("ijab,kilb->klja", t2, x25)
    del x25
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 -= einsum("iabj->ijba", v.ovvo)
    x26 += einsum("ijab->ijab", v.oovv) * 2
    x28 -= einsum("ia,jkba->ijkb", t1, x26)
    del x26
    l1new += einsum("abij,jkib->ak", l2, x28)
    del x28
    x29 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x29 += einsum("ia,bacd->icbd", t1, v.vvvv)
    x30 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x30 += einsum("iabc->iabc", x29)
    x30 += einsum("iabc->ibac", x29) * -0.5
    del x29
    x30 += einsum("aibc->iabc", v.vovv) * -0.5
    x30 += einsum("aibc->ibac", v.vovv)
    l1new += einsum("abij,ibac->cj", l2, x30) * 2
    del x30
    x31 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x31 += einsum("ia,abjk->kjib", t1, l2)
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x32 += einsum("ijka->ijka", x31) * 2
    x32 += einsum("ijka->jika", x31) * -1
    x34 += einsum("ijab,ikla->kljb", t2, x32) * -0.5
    del x32
    x34 += einsum("ijab,jkla->klib", t2, x31) * 0.5
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x36 -= einsum("ijka->ijka", x31)
    x36 += einsum("ijka->jika", x31) * 2
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum("ia,ijkb->jkba", t1, x36)
    l1new -= einsum("iabc,jiba->cj", v.ovvv, x37)
    del x37
    l1new -= einsum("iabj,kjib->ak", v.ovvo, x36)
    del x36
    x38 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x38 += einsum("ijka->ijka", x31) * 2
    x38 -= einsum("ijka->jika", x31)
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum("ia,ijkb->jkba", t1, x38)
    l1new -= einsum("iabc,jibc->aj", v.ovvv, x39)
    del x39
    x157 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x157 += einsum("ijka,jlkb->liba", x38, x7)
    x169 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x169 -= einsum("ijab->jiba", x157)
    del x157
    l1new -= einsum("ijab,kjia->bk", v.oovv, x38)
    del x38
    x42 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x42 += einsum("ijab,kjla->klib", t2, x31)
    x45 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x45 += einsum("ijka->ijka", x42)
    x49 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x49 += einsum("ijka->ijka", x42)
    del x42
    x47 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x47 += einsum("ia,jkla->kjli", t1, x31)
    x48 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x48 += einsum("ia,ijkl->jlka", t1, x47)
    x49 += einsum("ijka->ijka", x48) * -2.0000000000000013
    x49 += einsum("ijka->ikja", x48)
    del x48
    x51 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x51 += einsum("ijkl->ijkl", x47)
    x51 += einsum("ijkl->ijlk", x47) * -0.5
    l1new += einsum("ijka,ljki->al", v.ooov, x51) * 2
    del x51
    l2new += einsum("iajb,klij->abkl", v.ovov, x47)
    del x47
    x81 = np.zeros((nocc, nvir), dtype=types[float])
    x81 += einsum("ijab,ijkb->ka", x16, x31)
    x89 = np.zeros((nocc, nvir), dtype=types[float])
    x89 += einsum("ia->ia", x81)
    del x81
    x127 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x127 += einsum("ijka,jlkb->liba", v.ooov, x31)
    x143 -= einsum("ijab->ijab", x127)
    del x127
    x129 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x129 += einsum("ijka,iklb->jlab", x31, x7)
    x143 -= einsum("ijab->ijab", x129)
    del x129
    x146 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x146 += einsum("ijka,jlib->lkba", v.ooov, x31)
    x169 += einsum("ijab->ijab", x146)
    del x146
    x147 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x147 += einsum("iabc,jkib->kjac", v.ovvv, x31)
    x169 -= einsum("ijab->ijab", x147)
    del x147
    x148 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x148 += einsum("ijka,jklb->ilab", x31, x7)
    del x7
    x169 += einsum("ijab->ijab", x148)
    del x148
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 -= einsum("iajb->jiab", v.ovov)
    x35 += einsum("iajb->jiba", v.ovov) * 2
    l1new += einsum("ijka,kjab->bi", x34, x35) * 2
    del x34
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum("abij,jkca->ikbc", l2, t2)
    x41 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x41 += einsum("iabc->ibca", v.ovvv) * 2
    x41 -= einsum("iabc->ibac", v.ovvv)
    x104 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x104 += einsum("wia,ibca->wbc", u11, x41)
    x115 = np.zeros((nvir, nvir), dtype=types[float])
    x115 += einsum("ia,ibca->bc", t1, x41)
    x116 = np.zeros((nvir, nvir), dtype=types[float])
    x116 += einsum("ab->ab", x115)
    x164 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x164 += einsum("ab,acij->ijcb", x115, l2)
    del x115
    x169 += einsum("ijab->jiab", x164)
    del x164
    l1new -= einsum("ijab,jabc->ci", x40, x41)
    del x40
    del x41
    x43 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x43 += einsum("abij,klab->ijkl", l2, t2)
    x44 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x44 += einsum("ia,jikl->jkla", t1, x43)
    x45 += einsum("ijka->ijka", x44) * -0.5
    x45 += einsum("ijka->ikja", x44)
    del x44
    l1new += einsum("iajb,kjib->ak", v.ovov, x45) * 2
    del x45
    x50 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x50 += einsum("ijkl->jikl", x43) * -0.5
    x50 += einsum("ijkl->jilk", x43)
    l1new += einsum("ijka,jlik->al", v.ooov, x50) * 2
    del x50
    l2new += einsum("iajb,klij->balk", v.ovov, x43)
    del x43
    x46 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x46 += einsum("ai,jkab->ikjb", l1, t2)
    x49 += einsum("ijka->ijka", x46) * -1
    x49 += einsum("ijka->ikja", x46) * 2
    del x46
    l1new += einsum("iajb,kjia->bk", v.ovov, x49) * -1
    del x49
    x52 = np.zeros((nocc, nvir), dtype=types[float])
    x52 += einsum("w,wai->ia", s1, g.bvo)
    x76 += einsum("ia->ia", x52) * 0.5
    x179 += einsum("ia->ia", x52)
    del x52
    x53 = np.zeros((nocc, nvir), dtype=types[float])
    x53 += einsum("wab,wib->ia", g.bvv, u11)
    x76 += einsum("ia->ia", x53) * 0.5
    x179 += einsum("ia->ia", x53)
    del x53
    x54 = np.zeros((nocc, nvir), dtype=types[float])
    x54 += einsum("ijab,ikja->kb", t2, v.ooov)
    x76 += einsum("ia->ia", x54) * 0.5
    x179 += einsum("ia->ia", x54)
    del x54
    x55 += einsum("ijka->ikja", v.ooov) * 2
    x76 += einsum("ijab,ijkb->ka", t2, x55) * -0.5
    x179 += einsum("ijab,ijkb->ka", t2, x55) * -1
    del x55
    x57 += einsum("ia->ia", f.ov)
    x76 += einsum("ia,ijba->jb", x57, x16)
    x179 += einsum("ia,ijba->jb", x57, x16) * 2
    del x16
    lu11new = np.zeros((nbos, nvir, nocc), dtype=types[float])
    lu11new += einsum("w,ia->wai", ls1, x57) * 2
    del x57
    x58 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x58 += einsum("iabj->ijba", v.ovvo)
    x58 += einsum("ijab->ijab", v.oovv) * -0.5
    x76 += einsum("ia,ijba->jb", t1, x58)
    x179 += einsum("ia,ijba->jb", t1, x58) * 2
    del x58
    x59 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x59 += einsum("ia,wja->wji", t1, g.bov)
    x60 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x60 += einsum("wij->wij", x59)
    x110 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x110 += einsum("wij->wij", x59)
    del x59
    x60 += einsum("wij->wij", g.boo)
    x76 += einsum("wia,wij->ja", u11, x60) * -0.5
    x179 += einsum("wia,wij->ja", u11, x60) * -1
    del x60
    x61 = np.zeros((nocc, nocc), dtype=types[float])
    x61 += einsum("w,wij->ij", s1, g.boo)
    x69 = np.zeros((nocc, nocc), dtype=types[float])
    x69 += einsum("ij->ij", x61)
    x138 = np.zeros((nocc, nocc), dtype=types[float])
    x138 += einsum("ij->ij", x61)
    del x61
    x62 = np.zeros((nocc, nocc), dtype=types[float])
    x62 += einsum("wia,wja->ij", g.bov, u11)
    x69 += einsum("ij->ij", x62)
    x138 += einsum("ij->ij", x62)
    del x62
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum("iajb->jiab", v.ovov) * -0.5
    x63 += einsum("iajb->jiba", v.ovov)
    x64 = np.zeros((nocc, nocc), dtype=types[float])
    x64 += einsum("ijab,ikab->jk", t2, x63) * 2
    x69 += einsum("ij->ji", x64)
    del x64
    x173 = np.zeros((nvir, nvir), dtype=types[float])
    x173 += einsum("ijab,ijcb->ac", t2, x63)
    del x63
    x174 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x174 += einsum("ab,acij->ijcb", x173, l2) * 2
    del x173
    x177 += einsum("ijab->jiab", x174)
    del x174
    l2new += einsum("ijab->abij", x177) * -1
    l2new += einsum("ijab->baji", x177) * -1
    del x177
    x65 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x65 += einsum("ijka->ikja", v.ooov) * -0.5
    x65 += einsum("ijka->kija", v.ooov)
    x66 = np.zeros((nocc, nocc), dtype=types[float])
    x66 += einsum("ia,ijka->jk", t1, x65) * 2
    del x65
    x69 += einsum("ij->ij", x66)
    del x66
    x67 += einsum("ia->ia", f.ov) * 0.5
    x68 = np.zeros((nocc, nocc), dtype=types[float])
    x68 += einsum("ia,ja->ij", t1, x67) * 2
    del x67
    x69 += einsum("ij->ji", x68)
    del x68
    x69 += einsum("ij->ij", f.oo)
    x76 += einsum("ia,ij->ja", t1, x69) * -0.5
    x179 += einsum("ia,ij->ja", t1, x69) * -1
    l1new += einsum("ai,ji->aj", l1, x69) * -1
    del x69
    x70 = np.zeros((nvir, nvir), dtype=types[float])
    x70 += einsum("w,wab->ab", s1, g.bvv)
    x73 = np.zeros((nvir, nvir), dtype=types[float])
    x73 += einsum("ab->ab", x70)
    x116 += einsum("ab->ab", x70)
    x134 = np.zeros((nvir, nvir), dtype=types[float])
    x134 += einsum("ab->ab", x70)
    del x70
    x71 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x71 += einsum("iabc->ibca", v.ovvv)
    x71 += einsum("iabc->ibac", v.ovvv) * -0.5
    x72 = np.zeros((nvir, nvir), dtype=types[float])
    x72 += einsum("ia,ibca->bc", t1, x71) * 2
    del x71
    x73 += einsum("ab->ab", x72)
    del x72
    x73 += einsum("ab->ab", f.vv)
    x76 += einsum("ia,ba->ib", t1, x73) * 0.5
    x179 += einsum("ia,ba->ib", t1, x73)
    del x73
    x74 = np.zeros((nbos), dtype=types[float])
    x74 += einsum("ia,wia->w", t1, g.bov)
    x75 = np.zeros((nbos), dtype=types[float])
    x75 += einsum("w->w", x74) * 2
    x120 = np.zeros((nbos), dtype=types[float])
    x120 += einsum("w->w", x74) * 2
    del x74
    x75 += einsum("w->w", G)
    x76 += einsum("w,wia->ia", x75, u11) * 0.5
    x179 += einsum("w,wia->ia", x75, u11)
    del x75
    x76 += einsum("ai->ia", f.vo) * 0.5
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum("abij->jiab", l2) * -1
    x77 += einsum("abij->jiba", l2) * 2
    l1new += einsum("ia,ijab->bj", x76, x77) * 2
    del x76
    del x77
    x78 = np.zeros((nocc, nvir), dtype=types[float])
    x78 += einsum("w,wia->ia", ls1, u11)
    x89 += einsum("ia->ia", x78) * -0.5
    del x78
    x79 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x79 += einsum("ia,waj->wji", t1, lu11)
    x80 = np.zeros((nocc, nvir), dtype=types[float])
    x80 += einsum("wia,wij->ja", u11, x79)
    x89 += einsum("ia->ia", x80) * 0.5
    del x80
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum("ijab->jiab", t2) * -1
    x82 += einsum("ijab->jiba", t2) * 2
    x83 = np.zeros((nocc, nvir), dtype=types[float])
    x83 += einsum("ai,ijab->jb", l1, x82) * 0.5
    del x82
    x89 += einsum("ia->ia", x83) * -1
    del x83
    x84 = np.zeros((nocc, nocc), dtype=types[float])
    x84 += einsum("ai,ja->ij", l1, t1)
    x87 += einsum("ij->ij", x84) * 0.5
    x95 += einsum("ij->ij", x84)
    x140 = np.zeros((nocc, nocc), dtype=types[float])
    x140 += einsum("ij->ij", x84)
    x85 = np.zeros((nocc, nocc), dtype=types[float])
    x85 += einsum("wai,wja->ij", lu11, u11)
    x87 += einsum("ij->ij", x85) * 0.5
    x88 = np.zeros((nocc, nvir), dtype=types[float])
    x88 += einsum("ia,ij->ja", t1, x87)
    del x87
    x89 += einsum("ia->ia", x88)
    del x88
    x95 += einsum("ij->ij", x85)
    ls1new = np.zeros((nbos), dtype=types[float])
    ls1new += einsum("ij,wji->w", x95, g.boo) * -2
    x108 += einsum("ij->ij", x85)
    x109 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x109 += einsum("w,ij->wij", s1, x108)
    l1new += einsum("ia,ji->aj", f.ov, x108) * -1
    del x108
    x140 += einsum("ij->ij", x85)
    x141 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x141 += einsum("ij,jakb->kiab", x140, v.ovov)
    del x140
    x143 += einsum("ijab->jiba", x141)
    del x141
    x89 += einsum("ia->ia", t1) * -0.5
    l1new += einsum("ia,ijab->bj", x89, x35) * -2
    del x35
    ls1new += einsum("ia,wia->w", x89, g.bov) * -4
    del x89
    x90 = np.zeros((nvir, nvir), dtype=types[float])
    x90 += einsum("ai,ib->ab", l1, t1)
    x93 += einsum("ab->ab", x90) * 0.5
    ls1new += einsum("ab,wab->w", x90, g.bvv) * 2
    del x90
    x91 = np.zeros((nvir, nvir), dtype=types[float])
    x91 += einsum("wai,wib->ab", lu11, u11)
    x93 += einsum("ab->ab", x91) * 0.5
    x126 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x126 += einsum("ab,icjb->ijac", x91, v.ovov)
    x143 += einsum("ijab->ijab", x126)
    del x126
    x180 += einsum("ab->ab", x91) * 0.5
    del x91
    ls1new += einsum("ab,wab->w", x180, g.bvv) * 4
    del x180
    x94 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x94 += einsum("iabc->ibca", v.ovvv) * 2
    x94 += einsum("iabc->ibac", v.ovvv) * -1
    l1new += einsum("ab,iabc->ci", x93, x94) * 2
    del x93
    del x94
    x96 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x96 += einsum("ijka->ikja", v.ooov) * 2
    x96 += einsum("ijka->kija", v.ooov) * -1
    l1new += einsum("ij,jkia->ak", x95, x96) * -1
    del x96
    del x95
    x97 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x97 -= einsum("ijka->ikja", v.ooov)
    x97 += einsum("ijka->kija", v.ooov) * 2
    x103 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x103 += einsum("wia,ijka->wjk", u11, x97)
    x160 = np.zeros((nocc, nocc), dtype=types[float])
    x160 += einsum("ia,ijka->jk", t1, x97)
    del x97
    x162 = np.zeros((nocc, nocc), dtype=types[float])
    x162 += einsum("ij->ij", x160)
    del x160
    x98 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x98 += einsum("iajb->jiab", v.ovov) * 2
    x98 -= einsum("iajb->jiba", v.ovov)
    x99 = np.zeros((nocc, nvir), dtype=types[float])
    x99 += einsum("ia,ijba->jb", t1, x98)
    x100 += einsum("ia->ia", x99)
    x161 = np.zeros((nocc, nocc), dtype=types[float])
    x161 += einsum("ia,ja->ij", t1, x99)
    x162 += einsum("ij->ji", x161)
    del x161
    x163 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x163 += einsum("ij,abjk->kiab", x162, l2)
    del x162
    x169 -= einsum("ijab->ijba", x163)
    del x163
    x167 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x167 += einsum("ia,jkib->jkba", x99, x31)
    x169 -= einsum("ijab->ijab", x167)
    del x167
    x169 += einsum("ai,jb->ijab", l1, x99)
    l1new -= einsum("ij,ja->ai", x85, x99)
    del x99
    del x85
    x101 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x101 += einsum("wia,ijba->wjb", u11, x98)
    del x98
    x102 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x102 += einsum("wia->wia", x101)
    x114 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x114 += einsum("wia->wia", x101)
    x168 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x168 += einsum("wai,wjb->ijab", lu11, x101)
    del x101
    x169 += einsum("ijab->ijab", x168)
    del x168
    x100 += einsum("ia->ia", f.ov)
    x103 += einsum("ia,wja->wij", x100, u11)
    x119 = np.zeros((nbos), dtype=types[float])
    x119 += einsum("ia,wia->w", x100, u11)
    x122 = np.zeros((nbos), dtype=types[float])
    x122 += einsum("w->w", x119) * 2
    del x119
    l1new -= einsum("ia,ji->aj", x100, x84)
    del x100
    del x84
    x102 += einsum("wia->wia", gc.bov)
    x102 += einsum("wx,wia->xia", s2, g.bov)
    x103 += einsum("ia,wja->wji", t1, x102)
    del x102
    x103 += einsum("wij->wij", gc.boo)
    x103 += einsum("wx,wij->xij", s2, g.boo)
    l1new -= einsum("wai,wji->aj", lu11, x103)
    del x103
    x104 += einsum("wab->wab", gc.bvv)
    x104 += einsum("wx,wab->xab", s2, g.bvv)
    l1new += einsum("wai,wab->bi", lu11, x104)
    del x104
    x105 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x105 += einsum("wx,xai->wia", s2, lu11)
    x107 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x107 += einsum("wia->wia", x105)
    x165 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x165 += einsum("wia->wia", x105)
    del x105
    x106 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x106 += einsum("abij->jiab", l2)
    x106 += einsum("abij->jiba", l2) * -0.5
    x107 += einsum("wia,ijba->wjb", u11, x106) * 2
    del x106
    x109 += einsum("ia,wja->wji", t1, x107)
    del x107
    x109 += einsum("ai,wja->wij", l1, u11)
    l1new += einsum("wia,wji->aj", g.bov, x109) * -1
    del x109
    x110 += einsum("wij->wij", g.boo)
    x186 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x186 += einsum("ia,wij->wja", t1, x110)
    x187 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x187 -= einsum("wia->wia", x186)
    del x186
    x111 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x111 += einsum("abij->jiab", l2) * 2
    x111 -= einsum("abij->jiba", l2)
    x112 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x112 += einsum("wia,ijba->wjb", u11, x111)
    del x111
    x165 += einsum("wia->wia", x112)
    x166 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x166 += einsum("wia,wjb->ijab", g.bov, x165)
    del x165
    x169 += einsum("ijab->ijab", x166)
    del x166
    l1new -= einsum("wij,wja->ai", x110, x112)
    del x110
    l1new += einsum("wab,wia->bi", g.bvv, x112)
    del x112
    x113 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x113 += einsum("iabj->ijba", v.ovvo) * 2
    x113 -= einsum("ijab->ijab", v.oovv)
    l1new += einsum("ai,jiab->bj", l1, x113)
    lu11new += einsum("wai,jiab->wbj", lu11, x113)
    del x113
    x114 += einsum("wia->wia", gc.bov)
    l1new -= einsum("wia,wji->aj", x114, x79)
    del x114
    del x79
    x116 += einsum("ab->ab", f.vv)
    l1new += einsum("ai,ab->bi", l1, x116)
    del x116
    x117 = np.zeros((nbos), dtype=types[float])
    x117 += einsum("w,wx->x", s1, w)
    x122 += einsum("w->w", x117)
    del x117
    x118 = np.zeros((nbos), dtype=types[float])
    x118 += einsum("ia,wia->w", t1, gc.bov)
    x122 += einsum("w->w", x118) * 2
    del x118
    x120 += einsum("w->w", G)
    x121 = np.zeros((nbos), dtype=types[float])
    x121 += einsum("w,wx->x", x120, s2)
    x122 += einsum("w->w", x121)
    del x121
    x189 = np.zeros((nbos, nbos), dtype=types[float])
    x189 += einsum("w,x->xw", ls1, x120)
    del x120
    x122 += einsum("w->w", G)
    l1new += einsum("w,wai->ai", x122, lu11)
    ls1new += einsum("w,wx->x", x122, ls2)
    del x122
    x123 = np.zeros((nbos), dtype=types[float])
    x123 += einsum("w->w", s1)
    x123 += einsum("w,xw->x", ls1, s2)
    x123 += einsum("ai,wia->w", l1, u11) * 2
    l1new += einsum("w,wia->ai", x123, g.bov)
    del x123
    x124 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x124 += einsum("ai,jbac->ijbc", l1, v.ovvv)
    x143 -= einsum("ijab->ijab", x124)
    del x124
    x130 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x130 += einsum("ijab,kbic->jkac", t2, v.ovov)
    x131 -= einsum("ijab->ijab", x130)
    del x130
    x131 += einsum("ijab->jiab", v.oovv)
    x132 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x132 += einsum("abij,ikbc->jkac", l2, x131)
    x143 += einsum("ijab->ijab", x132)
    del x132
    x156 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x156 += einsum("abij,ikac->jkbc", l2, x131)
    del x131
    x169 -= einsum("ijab->ijab", x156)
    del x156
    x133 = np.zeros((nvir, nvir), dtype=types[float])
    x133 += einsum("wia,wib->ab", g.bov, u11)
    x134 -= einsum("ab->ba", x133)
    del x133
    x134 += einsum("ab->ab", f.vv)
    x135 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x135 += einsum("ab,acij->ijcb", x134, l2)
    del x134
    x143 -= einsum("ijab->jiba", x135)
    del x135
    x136 += einsum("ia->ia", f.ov)
    x137 = np.zeros((nocc, nocc), dtype=types[float])
    x137 += einsum("ia,ja->ij", t1, x136)
    x138 += einsum("ij->ji", x137)
    del x137
    x142 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x142 += einsum("ia,jkib->jkba", x136, x31)
    x143 += einsum("ijab->ijba", x142)
    del x142
    x169 += einsum("ai,jb->jiba", l1, x136)
    del x136
    x138 += einsum("ij->ij", f.oo)
    x139 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x139 += einsum("ij,abjk->kiab", x138, l2)
    del x138
    x143 += einsum("ijab->jiba", x139)
    del x139
    l2new -= einsum("ijab->baij", x143)
    l2new -= einsum("ijab->abji", x143)
    del x143
    x144 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x144 += einsum("wia,wbj->ijab", gc.bov, lu11)
    x169 += einsum("ijab->ijab", x144)
    del x144
    x145 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x145 += einsum("ai,jikb->jkab", l1, v.ooov)
    x169 -= einsum("ijab->ijab", x145)
    del x145
    x149 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x149 -= einsum("abij->jiab", l2)
    x149 += einsum("abij->jiba", l2) * 2
    lu11new += einsum("wai,ijab->wbj", g.bvo, x149)
    x150 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x150 += einsum("ia,jbca->ijcb", t1, v.ovvv)
    x154 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x154 += einsum("ijab->ijab", x150)
    del x150
    x151 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x151 += einsum("ijab,kaic->jkbc", t2, v.ovov)
    x154 -= einsum("ijab->ijab", x151)
    del x151
    x152 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x152 += einsum("ijab->jiab", t2) * 2
    x152 -= einsum("ijab->jiba", t2)
    x153 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x153 += einsum("iajb,ikca->jkbc", v.ovov, x152)
    x154 += einsum("ijab->jiba", x153)
    del x153
    x185 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x185 += einsum("wia,ijba->wjb", g.bov, x152)
    del x152
    x187 += einsum("wia->wia", x185)
    del x185
    x154 += einsum("iabj->jiba", v.ovvo)
    x155 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x155 += einsum("ijab,ikac->jkbc", x149, x154)
    del x154
    del x149
    x169 += einsum("ijab->ijab", x155)
    del x155
    x158 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x158 += einsum("ijka->ikja", v.ooov) * 2
    x158 -= einsum("ijka->kija", v.ooov)
    x159 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x159 += einsum("ijka,lkib->ljba", x158, x31)
    del x158
    del x31
    x169 -= einsum("ijab->ijab", x159)
    del x159
    l2new += einsum("ijab->abij", x169)
    l2new += einsum("ijab->baji", x169)
    del x169
    x178 += einsum("ijkl->kilj", v.oooo)
    l2new += einsum("abij,klji->ablk", l2, x178)
    del x178
    x179 += einsum("ai->ia", f.vo)
    ls1new += einsum("ia,wai->w", x179, lu11) * 2
    del x179
    x181 = np.zeros((nbos, nbos), dtype=types[float])
    x181 += einsum("wx,yx->wy", ls2, w)
    x189 += einsum("wx->wx", x181)
    del x181
    x182 = np.zeros((nbos, nbos), dtype=types[float])
    x182 += einsum("wia,xia->wx", g.bov, u11)
    x183 = np.zeros((nbos, nbos), dtype=types[float])
    x183 += einsum("wx,yx->yw", ls2, x182)
    del x182
    x189 += einsum("wx->wx", x183) * 2
    del x183
    x184 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x184 += einsum("ia,wba->wib", t1, g.bvv)
    x187 += einsum("wia->wia", x184)
    del x184
    x187 += einsum("wai->wia", g.bvo)
    x188 = np.zeros((nbos, nbos), dtype=types[float])
    x188 += einsum("wai,xia->wx", lu11, x187)
    del x187
    x189 += einsum("wx->xw", x188) * 2
    del x188
    ls2new = np.zeros((nbos, nbos), dtype=types[float])
    ls2new += einsum("wx->wx", x189)
    ls2new += einsum("wx->xw", x189)
    del x189
    l1new += einsum("ia->ai", f.ov)
    l1new += einsum("w,wia->ai", ls1, gc.bov)
    l2new += einsum("abij,bcad->cdji", l2, v.vvvv)
    l2new += einsum("iajb->baji", v.ovov)
    ls1new += einsum("ai,wai->w", l1, g.bvo) * 2
    ls1new += einsum("w,xw->x", ls1, w)
    ls1new += einsum("w->w", G)
    lu11new -= einsum("ij,waj->wai", f.oo, lu11)
    lu11new -= einsum("ai,wji->waj", l1, g.boo)
    lu11new += einsum("wx,xia->wai", ls2, gc.bov)
    lu11new += einsum("wx,xai->wai", w, lu11)
    lu11new += einsum("wia->wai", g.bov)
    lu11new += einsum("ab,wai->wbi", f.vv, lu11)
    lu11new += einsum("ai,wab->wbi", l1, g.bvv)

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
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum("ij->ij", x0)
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=types[float])
    rdm1_f_oo -= einsum("ij->ij", x0) * 2
    del x0
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 += einsum("wai,wja->ij", lu11, u11)
    x8 += einsum("ij->ij", x1)
    rdm1_f_oo -= einsum("ij->ij", x1) * 2
    del x1
    x2 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x2 += einsum("ijab->jiab", t2) * 2
    x2 += einsum("ijab->jiba", t2) * -1
    rdm1_f_oo += einsum("abij,ikba->jk", l2, x2) * -2
    del x2
    x3 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x3 += einsum("ia,waj->wji", t1, lu11)
    rdm1_f_vo = np.zeros((nvir, nocc), dtype=types[float])
    rdm1_f_vo -= einsum("wia,wij->aj", u11, x3) * 2
    del x3
    x4 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x4 += einsum("ia,abjk->kjib", t1, l2)
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x5 += einsum("ijka->ijka", x4) * -1
    x5 += einsum("ijka->jika", x4) * 2
    del x4
    rdm1_f_vo += einsum("ijab,jika->bk", t2, x5) * -2
    del x5
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum("ijab->jiab", t2) * 2
    x6 -= einsum("ijab->jiba", t2)
    rdm1_f_vo += einsum("ai,ijba->bj", l1, x6) * 2
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum("ijab->jiab", t2) * -0.5
    x7 += einsum("ijab->jiba", t2)
    x8 += einsum("abij,ikab->jk", l2, x7) * 2
    del x7
    rdm1_f_vo += einsum("ia,ij->aj", t1, x8) * -2
    del x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum("abij->jiab", l2) * -0.5
    x9 += einsum("abij->jiba", l2)
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=types[float])
    rdm1_f_vv += einsum("ijab,ijac->bc", t2, x9) * 4
    del x9
    rdm1_f_oo += einsum("ij->ji", delta_oo) * 2
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=types[float])
    rdm1_f_ov += einsum("ai->ia", l1) * 2
    rdm1_f_vo += einsum("ia->ai", t1) * 2
    rdm1_f_vo += einsum("w,wia->ai", ls1, u11) * 2
    rdm1_f_vv += einsum("wai,wib->ba", lu11, u11) * 2
    rdm1_f_vv += einsum("ai,ib->ba", l1, t1) * 2

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
    x23 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x23 += einsum("ia,jikl->jkla", t1, x0)
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum("ijka->ijka", x23) * 4
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_vooo += einsum("ijka->ajik", x23) * -2
    rdm2_f_vooo += einsum("ijka->akij", x23) * 4
    del x23
    rdm2_f_oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_oooo += einsum("ijkl->jkil", x0) * -2
    rdm2_f_oooo += einsum("ijkl->jlik", x0) * 4
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum("ia,abjk->kjib", t1, l2)
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum("ia,jkla->jkil", t1, x1)
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum("ia,ijkl->jlka", t1, x2)
    x18 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x18 -= einsum("ijka->ijka", x10)
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x35 += einsum("ijka->ijka", x10)
    x67 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x67 += einsum("ia,ijkb->jkab", t1, x10)
    del x10
    x69 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x69 += einsum("ijab->ijab", x67)
    del x67
    x75 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum("ijab,jikl->lkab", t2, x2)
    x78 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x78 += einsum("ijab->ijab", x75) * 2.0000000000000013
    del x75
    rdm2_f_oooo += einsum("ijkl->ikjl", x2) * 4
    rdm2_f_oooo -= einsum("ijkl->iljk", x2) * 2
    del x2
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 += einsum("ijab,jkla->klib", t2, x1)
    x18 -= einsum("ijka->ijka", x9)
    del x9
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum("ijka->ijka", x1) * 2
    x11 -= einsum("ijka->jika", x1)
    x12 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x12 += einsum("ijab,ikla->kljb", t2, x11)
    x18 += einsum("ijka->ijka", x12)
    del x12
    x51 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x51 += einsum("ia,jikb->jkba", t1, x11)
    del x11
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_ovvo -= einsum("ijab->iabj", x51) * 2
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_voov -= einsum("ijab->bjia", x51) * 2
    del x51
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x20 += einsum("ijab,kjla->klib", t2, x1)
    x22 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x22 -= einsum("ijka->ijka", x20)
    x80 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x80 -= einsum("ijka->ijka", x20)
    del x20
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x24 += einsum("ijka->ijka", x1)
    x24 += einsum("ijka->jika", x1) * -0.5
    x25 = np.zeros((nocc, nvir), dtype=types[float])
    x25 += einsum("ijab,jikb->ka", t2, x24) * 2
    x27 = np.zeros((nocc, nvir), dtype=types[float])
    x27 += einsum("ia->ia", x25)
    del x25
    x36 = np.zeros((nocc, nvir), dtype=types[float])
    x36 += einsum("ijab,jikb->ka", t2, x24)
    del x24
    x38 = np.zeros((nocc, nvir), dtype=types[float])
    x38 += einsum("ia->ia", x36)
    del x36
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x32 += einsum("ijab,kjlb->klia", t2, x1)
    x35 += einsum("ijka->ijka", x32)
    x58 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x58 += einsum("ijka->ijka", x32)
    del x32
    x43 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x43 -= einsum("ijka->ijka", x1)
    x43 += einsum("ijka->jika", x1) * 2
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x44 += einsum("ia,jikb->jkba", t1, x43)
    del x43
    rdm2_f_oovv = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_oovv -= einsum("ijab->ijba", x44) * 2
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vvoo -= einsum("ijab->baij", x44) * 2
    del x44
    x92 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x92 += einsum("ia,jikb->jkba", t1, x1)
    x93 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x93 += einsum("ijab->ijab", x92)
    del x92
    x100 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x100 += einsum("ijab,ijkc->kcab", t2, x1)
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv += einsum("iabc->bica", x100) * 2
    rdm2_f_vovv += einsum("iabc->ciba", x100) * -4
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo += einsum("iabc->baci", x100) * -4
    rdm2_f_vvvo += einsum("iabc->cabi", x100) * 2
    del x100
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov += einsum("ijka->ikja", x1) * 2
    rdm2_f_ooov -= einsum("ijka->jkia", x1) * 4
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_ovoo -= einsum("ijka->iajk", x1) * 4
    rdm2_f_ovoo += einsum("ijka->jaik", x1) * 2
    x3 = np.zeros((nocc, nocc), dtype=types[float])
    x3 += einsum("ai,ja->ij", l1, t1)
    x21 = np.zeros((nocc, nocc), dtype=types[float])
    x21 += einsum("ij->ij", x3)
    x30 = np.zeros((nocc, nvir), dtype=types[float])
    x30 += einsum("ia,ij->ja", t1, x3)
    x31 = np.zeros((nocc, nvir), dtype=types[float])
    x31 -= einsum("ia->ia", x30)
    del x30
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum("ij,kiab->jkab", x3, t2)
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum("ijab->ijab", x79)
    del x79
    rdm2_f_oooo -= einsum("ij,kl->jikl", delta_oo, x3) * 4
    rdm2_f_oooo += einsum("ij,kl->kijl", delta_oo, x3) * 2
    rdm2_f_oooo += einsum("ij,kl->jlki", delta_oo, x3) * 2
    rdm2_f_oooo -= einsum("ij,kl->klji", delta_oo, x3) * 4
    del x3
    x4 = np.zeros((nocc, nocc), dtype=types[float])
    x4 += einsum("wai,wja->ij", lu11, u11)
    x14 = np.zeros((nocc, nvir), dtype=types[float])
    x14 += einsum("ia,ij->ja", t1, x4)
    x17 = np.zeros((nocc, nvir), dtype=types[float])
    x17 += einsum("ia->ia", x14)
    del x14
    x21 += einsum("ij->ij", x4)
    x22 += einsum("ia,jk->jika", t1, x21)
    x85 = np.zeros((nocc, nvir), dtype=types[float])
    x85 += einsum("ia,ij->ja", t1, x21)
    del x21
    x86 = np.zeros((nocc, nvir), dtype=types[float])
    x86 += einsum("ia->ia", x85)
    x87 = np.zeros((nocc, nvir), dtype=types[float])
    x87 += einsum("ia->ia", x85)
    del x85
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum("ij,kiab->kjab", x4, t2)
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum("ijab->ijab", x56)
    del x56
    rdm2_f_oooo -= einsum("ij,kl->ijkl", delta_oo, x4) * 4
    rdm2_f_oooo += einsum("ij,kl->ilkj", delta_oo, x4) * 2
    rdm2_f_oooo += einsum("ij,kl->kijl", delta_oo, x4) * 2
    rdm2_f_oooo -= einsum("ij,kl->klji", delta_oo, x4) * 4
    del x4
    x5 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x5 += einsum("ijab->jiab", t2)
    x5 += einsum("ijab->jiba", t2) * -0.5
    x6 = np.zeros((nocc, nocc), dtype=types[float])
    x6 += einsum("abij,ikba->kj", l2, x5)
    del x5
    x26 = np.zeros((nocc, nvir), dtype=types[float])
    x26 += einsum("ia,ji->ja", t1, x6) * 2
    x27 += einsum("ia->ia", x26)
    del x26
    x28 += einsum("ij,ka->jika", delta_oo, x27) * -4
    del x27
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo += einsum("ijka->ijak", x28)
    rdm2_f_oovo += einsum("ijka->ikaj", x28) * -0.5
    del x28
    x37 = np.zeros((nocc, nvir), dtype=types[float])
    x37 += einsum("ia,ji->ja", t1, x6)
    x38 += einsum("ia->ia", x37)
    del x37
    x74 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x74 += einsum("ia,jb->ijab", t1, x38) * 8.000000000000005
    rdm2_f_vooo += einsum("ij,ka->ajik", delta_oo, x38) * 4
    rdm2_f_vooo += einsum("ij,ka->akij", delta_oo, x38) * -8
    del x38
    x73 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum("ij,jkab->ikab", x6, t2) * 8
    x74 += einsum("ijab->jiba", x73)
    del x73
    rdm2_f_oooo += einsum("ij,kl->jilk", delta_oo, x6) * -8
    rdm2_f_oooo += einsum("ij,kl->jkli", delta_oo, x6) * 4
    rdm2_f_oooo += einsum("ij,kl->ljik", delta_oo, x6) * 4
    rdm2_f_oooo += einsum("ij,kl->lkij", delta_oo, x6) * -8
    rdm2_f_oovo += einsum("ia,jk->kiaj", t1, x6) * 4
    rdm2_f_oovo += einsum("ia,jk->kjai", t1, x6) * -8
    rdm2_f_vooo += einsum("ia,jk->aikj", t1, x6) * -8
    rdm2_f_vooo += einsum("ia,jk->ajki", t1, x6) * 4
    del x6
    x7 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x7 += einsum("ia,waj->wji", t1, lu11)
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x8 += einsum("wia,wjk->jkia", u11, x7)
    x18 += einsum("ijka->ijka", x8)
    x35 -= einsum("ijka->ijka", x8)
    del x8
    x13 = np.zeros((nocc, nvir), dtype=types[float])
    x13 += einsum("wia,wij->ja", u11, x7)
    x17 += einsum("ia->ia", x13)
    x86 += einsum("ia->ia", x13)
    x87 += einsum("ia->ia", x13)
    del x13
    x60 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x60 += einsum("ia,wij->wja", t1, x7)
    del x7
    x62 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x62 += einsum("wia->wia", x60)
    del x60
    x15 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x15 -= einsum("ijab->jiab", t2)
    x15 += einsum("ijab->jiba", t2) * 2
    x16 = np.zeros((nocc, nvir), dtype=types[float])
    x16 += einsum("ai,ijab->jb", l1, x15)
    x17 -= einsum("ia->ia", x16)
    x18 += einsum("ij,ka->jika", delta_oo, x17)
    rdm2_f_oovo -= einsum("ijka->ijak", x18) * 4
    rdm2_f_oovo += einsum("ijka->ikaj", x18) * 2
    del x18
    rdm2_f_vooo += einsum("ij,ka->ajik", delta_oo, x17) * 2
    rdm2_f_vooo -= einsum("ij,ka->akij", delta_oo, x17) * 4
    del x17
    x64 -= einsum("ia,jb->ijab", t1, x16)
    del x16
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum("abij,ikac->kjcb", l2, x15)
    x93 -= einsum("ijab->jiba", x54)
    rdm2_f_vvoo -= einsum("ijab->abji", x54) * 2
    del x54
    x61 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x61 += einsum("wai,ijab->wjb", lu11, x15)
    x62 -= einsum("wia->wia", x61)
    del x61
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum("wia,wjb->ijab", u11, x62)
    del x62
    x64 += einsum("ijab->jiba", x63)
    del x63
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x19 += einsum("ai,jkab->ikjb", l1, t2)
    x22 += einsum("ijka->ijka", x19)
    rdm2_f_oovo += einsum("ijka->ijak", x22) * 2
    rdm2_f_oovo -= einsum("ijka->ikaj", x22) * 4
    rdm2_f_vooo -= einsum("ijka->ajik", x22) * 4
    rdm2_f_vooo += einsum("ijka->akij", x22) * 2
    del x22
    x80 += einsum("ijka->ijka", x19)
    del x19
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum("ia,ijkb->jkba", t1, x80)
    del x80
    x82 += einsum("ijab->ijba", x81)
    del x81
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_vovo += einsum("ijab->aibj", x82) * 2
    rdm2_f_vovo -= einsum("ijab->biaj", x82) * 4
    rdm2_f_vovo -= einsum("ijab->ajbi", x82) * 4
    rdm2_f_vovo += einsum("ijab->bjai", x82) * 2
    del x82
    x29 = np.zeros((nocc, nvir), dtype=types[float])
    x29 += einsum("w,wia->ia", ls1, u11)
    x31 += einsum("ia->ia", x29)
    x86 -= einsum("ia->ia", x29)
    x87 -= einsum("ia->ia", x29)
    del x29
    rdm2_f_vovo -= einsum("ia,jb->aibj", t1, x87) * 4
    rdm2_f_vovo += einsum("ia,jb->biaj", t1, x87) * 2
    del x87
    x31 += einsum("ia->ia", t1)
    rdm2_f_oovo -= einsum("ij,ka->jkai", delta_oo, x31) * 2
    rdm2_f_oovo += einsum("ij,ka->jiak", delta_oo, x31) * 4
    rdm2_f_vooo += einsum("ij,ka->akji", delta_oo, x31) * 4
    rdm2_f_vooo -= einsum("ij,ka->aijk", delta_oo, x31) * 2
    del x31
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum("ijab->jiab", t2) * 2
    x33 -= einsum("ijab->jiba", t2)
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum("ijka,ilba->ljkb", x1, x33)
    del x1
    x35 -= einsum("ijka->jkia", x34)
    rdm2_f_vooo -= einsum("ijka->ajik", x35) * 2
    rdm2_f_vooo += einsum("ijka->akij", x35) * 4
    del x35
    x58 -= einsum("ijka->jkia", x34)
    del x34
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum("ia,ijkb->jkba", t1, x58)
    del x58
    x64 -= einsum("ijab->ijba", x59)
    del x59
    rdm2_f_vvoo -= einsum("abij,ikbc->cajk", l2, x33) * 2
    del x33
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum("wai,wjb->ijab", lu11, u11)
    rdm2_f_oovv -= einsum("ijab->ijba", x39) * 2
    rdm2_f_ovvo += einsum("ijab->iabj", x39) * 4
    rdm2_f_voov += einsum("ijab->bjia", x39) * 4
    rdm2_f_vvoo -= einsum("ijab->baij", x39) * 2
    del x39
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 -= einsum("abij->jiab", l2)
    x40 += einsum("abij->jiba", l2) * 2
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 += einsum("ijab,ikac->kjcb", t2, x40)
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum("ijab,ikbc->kjca", t2, x41)
    x64 += einsum("ijab->ijab", x57)
    del x57
    x68 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x68 += einsum("ijab,ikac->kjcb", t2, x41)
    x69 += einsum("ijab->ijab", x68) * 2
    del x68
    rdm2_f_oovv -= einsum("ijab->ijba", x41) * 2
    del x41
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum("ijab,ikac->jkbc", x15, x40)
    del x40
    del x15
    rdm2_f_ovvo += einsum("ijab->jbai", x50) * 2
    rdm2_f_voov += einsum("ijab->aijb", x50) * 2
    del x50
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum("abij->jiab", l2) * 2
    x42 -= einsum("abij->jiba", l2)
    rdm2_f_oovv -= einsum("ijab,ikbc->kjac", t2, x42) * 2
    del x42
    x45 = np.zeros((nvir, nvir), dtype=types[float])
    x45 += einsum("ai,ib->ab", l1, t1)
    x49 = np.zeros((nvir, nvir), dtype=types[float])
    x49 += einsum("ab->ab", x45) * 0.5
    x53 = np.zeros((nvir, nvir), dtype=types[float])
    x53 += einsum("ab->ab", x45)
    x98 = np.zeros((nvir, nvir), dtype=types[float])
    x98 += einsum("ab->ab", x45)
    del x45
    x46 = np.zeros((nvir, nvir), dtype=types[float])
    x46 += einsum("wai,wib->ab", lu11, u11)
    x49 += einsum("ab->ab", x46) * 0.5
    x53 += einsum("ab->ab", x46)
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum("ab,ijca->ijcb", x46, t2)
    x64 += einsum("ijab->ijab", x55)
    del x55
    rdm2_f_vovo -= einsum("ijab->aibj", x64) * 4
    rdm2_f_vovo += einsum("ijab->biaj", x64) * 2
    rdm2_f_vovo += einsum("ijab->ajbi", x64) * 2
    rdm2_f_vovo -= einsum("ijab->bjai", x64) * 4
    del x64
    x98 += einsum("ab->ab", x46)
    del x46
    x99 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x99 += einsum("ia,bc->ibac", t1, x98)
    del x98
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x47 += einsum("abij->jiab", l2)
    x47 += einsum("abij->jiba", l2) * -0.5
    x48 = np.zeros((nvir, nvir), dtype=types[float])
    x48 += einsum("ijab,ijbc->ca", t2, x47)
    x49 += einsum("ab->ab", x48)
    rdm2_f_oovv += einsum("ij,ab->jiba", delta_oo, x49) * 8
    del x49
    x72 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x72 += einsum("ab,ijac->ijbc", x48, t2) * 8
    x74 += einsum("ijab->jiba", x72)
    del x72
    rdm2_f_vovo += einsum("ijab->aibj", x74) * -1
    rdm2_f_vovo += einsum("ijab->biaj", x74) * 0.5
    rdm2_f_vovo += einsum("ijab->ajbi", x74) * 0.5
    rdm2_f_vovo += einsum("ijab->bjai", x74) * -1
    del x74
    rdm2_f_vovv += einsum("ia,bc->aicb", t1, x48) * 8
    rdm2_f_vovv += einsum("ia,bc->ciab", t1, x48) * -4
    rdm2_f_vvvo += einsum("ia,bc->abci", t1, x48) * -4
    rdm2_f_vvvo += einsum("ia,bc->cbai", t1, x48) * 8
    del x48
    x52 = np.zeros((nvir, nvir), dtype=types[float])
    x52 += einsum("ijab,ijbc->ca", t2, x47) * 2
    del x47
    x53 += einsum("ab->ab", x52)
    del x52
    rdm2_f_ovvo += einsum("ij,ab->jabi", delta_oo, x53) * -2
    rdm2_f_voov += einsum("ij,ab->bija", delta_oo, x53) * -2
    rdm2_f_vvoo += einsum("ij,ab->baji", delta_oo, x53) * 4
    del x53
    x65 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x65 += einsum("abij,kjbc->ikac", l2, t2)
    x66 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x66 += einsum("ijab,jkac->ikbc", t2, x65)
    del x65
    x69 += einsum("ijab->ijab", x66)
    del x66
    x69 += einsum("ijab->jiba", t2)
    rdm2_f_vovo -= einsum("ijab->biaj", x69) * 2
    rdm2_f_vovo += einsum("ijab->aibj", x69) * 4
    del x69
    x70 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x70 += einsum("abij,kjac->ikbc", l2, t2)
    x71 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x71 += einsum("ijab,jkac->ikbc", t2, x70)
    rdm2_f_vovo += einsum("ijab->ajbi", x71) * 4
    rdm2_f_vovo -= einsum("ijab->bjai", x71) * 2
    del x71
    x97 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x97 += einsum("ia,ijbc->jbac", t1, x70)
    del x70
    x99 -= einsum("iabc->iabc", x97)
    del x97
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum("ijab->jiba", t2)
    x76 += einsum("ia,jb->ijab", t1, t1)
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum("ijkl,ijab->klab", x0, x76) * 2
    del x0
    del x76
    x78 += einsum("ijab->jiba", x77)
    del x77
    rdm2_f_vovo += einsum("ijab->ajbi", x78) * -1
    rdm2_f_vovo += einsum("ijab->bjai", x78) * 2
    del x78
    x83 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x83 += einsum("wx,xia->wia", ls2, u11)
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum("wia,wjb->jiba", u11, x83)
    del x83
    rdm2_f_vovo -= einsum("ijab->biaj", x84) * 2
    rdm2_f_vovo += einsum("ijab->bjai", x84) * 4
    del x84
    x86 -= einsum("ia->ia", t1)
    rdm2_f_vovo -= einsum("ia,jb->bjai", t1, x86) * 4
    rdm2_f_vovo += einsum("ia,jb->ajbi", t1, x86) * 2
    del x86
    x88 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x88 += einsum("ia,bcji->jbca", t1, l2)
    x102 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x102 += einsum("ia,ibcd->cbda", t1, x88)
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv -= einsum("abcd->cbda", x102) * 2
    rdm2_f_vvvv += einsum("abcd->dbca", x102) * 4
    del x102
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv += einsum("iabc->iacb", x88) * 4
    rdm2_f_ovvv -= einsum("iabc->ibca", x88) * 2
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov -= einsum("iabc->caib", x88) * 2
    rdm2_f_vvov += einsum("iabc->cbia", x88) * 4
    del x88
    x89 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x89 += einsum("ia,wbi->wba", t1, lu11)
    x90 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x90 += einsum("wia,wbc->ibca", u11, x89)
    del x89
    x95 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x95 -= einsum("iabc->iabc", x90)
    del x90
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x91 += einsum("abij,kjca->ikbc", l2, t2)
    x93 += einsum("ijab->ijab", x91)
    del x91
    x94 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x94 += einsum("ia,ijbc->jbca", t1, x93)
    del x93
    x95 += einsum("iabc->iacb", x94)
    del x94
    rdm2_f_vovv += einsum("iabc->bica", x95) * 2
    rdm2_f_vovv -= einsum("iabc->ciba", x95) * 4
    rdm2_f_vvvo -= einsum("iabc->baci", x95) * 4
    rdm2_f_vvvo += einsum("iabc->cabi", x95) * 2
    del x95
    x96 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x96 += einsum("ai,jibc->jabc", l1, t2)
    x99 += einsum("iabc->iabc", x96)
    del x96
    rdm2_f_vovv += einsum("iabc->bica", x99) * 4
    rdm2_f_vovv -= einsum("iabc->ciba", x99) * 2
    rdm2_f_vvvo -= einsum("iabc->baci", x99) * 2
    rdm2_f_vvvo += einsum("iabc->cabi", x99) * 4
    del x99
    x101 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x101 += einsum("abij,ijcd->abcd", l2, t2)
    rdm2_f_vvvv += einsum("abcd->cbda", x101) * -2
    rdm2_f_vvvv += einsum("abcd->dbca", x101) * 4
    del x101
    rdm2_f_oooo += einsum("ij,kl->jilk", delta_oo, delta_oo) * 4
    rdm2_f_oooo -= einsum("ij,kl->ljik", delta_oo, delta_oo) * 2
    rdm2_f_ooov += einsum("ij,ak->jika", delta_oo, l1) * 4
    rdm2_f_ooov -= einsum("ij,ak->kija", delta_oo, l1) * 2
    rdm2_f_ovoo -= einsum("ij,ak->jaki", delta_oo, l1) * 2
    rdm2_f_ovoo += einsum("ij,ak->kaji", delta_oo, l1) * 4
    rdm2_f_ovov = np.zeros((nocc, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_ovov -= einsum("abij->jaib", l2) * 2
    rdm2_f_ovov += einsum("abij->jbia", l2) * 4
    rdm2_f_oovv -= einsum("ai,jb->ijba", l1, t1) * 2
    rdm2_f_ovvo += einsum("ai,jb->iabj", l1, t1) * 4
    rdm2_f_voov += einsum("ai,jb->bjia", l1, t1) * 4
    rdm2_f_vvoo -= einsum("ai,jb->baij", l1, t1) * 2

    rdm2_f = pack_2e(rdm2_f_oooo, rdm2_f_ooov, rdm2_f_oovo, rdm2_f_ovoo, rdm2_f_vooo, rdm2_f_oovv, rdm2_f_ovov, rdm2_f_ovvo, rdm2_f_voov, rdm2_f_vovo, rdm2_f_vvoo, rdm2_f_ovvv, rdm2_f_vovv, rdm2_f_vvov, rdm2_f_vvvo, rdm2_f_vvvv)

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
    dm_b_des += einsum("w->w", s1)
    dm_b_des += einsum("ai,wia->w", l1, u11) * 2
    dm_b_des += einsum("w,xw->x", ls1, s2)

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
    rdm1_b += einsum("w,x->wx", ls1, s1)
    rdm1_b += einsum("wx,yx->wy", ls2, s2)
    rdm1_b += einsum("wai,xia->wx", lu11, u11) * 2

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
    x27 = np.zeros((nocc, nvir), dtype=types[float])
    x27 += einsum("wia,wij->ja", u11, x0)
    rdm_eb_cre_oo = np.zeros((nbos, nocc, nocc), dtype=types[float])
    rdm_eb_cre_oo -= einsum("wij->wji", x0) * 2
    rdm_eb_cre_ov = np.zeros((nbos, nocc, nvir), dtype=types[float])
    rdm_eb_cre_ov -= einsum("ia,wij->wja", t1, x0) * 2
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum("ijab->jiab", t2) * 2
    x1 -= einsum("ijab->jiba", t2)
    rdm_eb_cre_ov += einsum("wai,ijba->wjb", lu11, x1) * 2
    x2 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x2 += einsum("ai,wja->wij", l1, u11)
    x21 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x21 += einsum("wij->wij", x2)
    rdm_eb_des_oo = np.zeros((nbos, nocc, nocc), dtype=types[float])
    rdm_eb_des_oo -= einsum("wij->wji", x2) * 2
    del x2
    x3 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x3 += einsum("wx,xai->wia", s2, lu11)
    x6 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x6 += einsum("wia->wia", x3)
    rdm_eb_des_vo = np.zeros((nbos, nvir, nocc), dtype=types[float])
    rdm_eb_des_vo += einsum("wia->wai", x3) * 2
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum("abij->jiab", l2) * 2
    x4 -= einsum("abij->jiba", l2)
    x5 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x5 += einsum("wia,ijba->wjb", u11, x4)
    del x4
    x6 += einsum("wia->wia", x5)
    x7 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x7 += einsum("ia,wja->wij", t1, x6)
    x21 += einsum("wij->wji", x7)
    rdm_eb_des_ov = np.zeros((nbos, nocc, nvir), dtype=types[float])
    rdm_eb_des_ov -= einsum("ia,wij->wja", t1, x21) * 2
    del x21
    rdm_eb_des_oo -= einsum("wij->wij", x7) * 2
    del x7
    rdm_eb_des_ov += einsum("wia,ijba->wjb", x6, x1) * 2
    del x1
    rdm_eb_des_vv = np.zeros((nbos, nvir, nvir), dtype=types[float])
    rdm_eb_des_vv += einsum("ia,wib->wba", t1, x6) * 2
    del x6
    rdm_eb_des_vo += einsum("wia->wai", x5) * 2
    del x5
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum("ai,ja->ij", l1, t1)
    x11 = np.zeros((nocc, nocc), dtype=types[float])
    x11 += einsum("ij->ij", x8)
    x20 = np.zeros((nocc, nocc), dtype=types[float])
    x20 += einsum("ij->ij", x8) * 0.5
    x26 = np.zeros((nocc, nocc), dtype=types[float])
    x26 += einsum("ij->ij", x8) * 0.49999999999999967
    del x8
    x9 = np.zeros((nocc, nocc), dtype=types[float])
    x9 += einsum("wai,wja->ij", lu11, u11)
    x11 += einsum("ij->ij", x9)
    x20 += einsum("ij->ij", x9) * 0.5
    x26 += einsum("ij->ij", x9) * 0.49999999999999967
    del x9
    x10 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x10 += einsum("ijab->jiab", t2) * -0.5
    x10 += einsum("ijab->jiba", t2)
    x11 += einsum("abij,ikab->jk", l2, x10) * 2
    x19 = np.zeros((nocc, nocc), dtype=types[float])
    x19 += einsum("abij,ikab->jk", l2, x10)
    del x10
    x20 += einsum("ij->ij", x19)
    rdm_eb_des_ov += einsum("ij,wia->wja", x20, u11) * -4
    del x20
    x26 += einsum("ij->ij", x19)
    del x19
    x27 += einsum("ia,ij->ja", t1, x26) * 2.0000000000000013
    del x26
    x11 += einsum("ij->ji", delta_oo) * -1
    rdm_eb_des_oo += einsum("w,ij->wji", s1, x11) * -2
    del x11
    x12 = np.zeros((nbos), dtype=types[float])
    x12 += einsum("w,xw->x", ls1, s2)
    x14 = np.zeros((nbos), dtype=types[float])
    x14 += einsum("w->w", x12)
    del x12
    x13 = np.zeros((nbos), dtype=types[float])
    x13 += einsum("ai,wia->w", l1, u11)
    x14 += einsum("w->w", x13) * 2
    del x13
    rdm_eb_des_oo += einsum("w,ij->wji", x14, delta_oo) * 2
    rdm_eb_des_ov += einsum("w,ia->wia", x14, t1) * 2
    del x14
    x15 = np.zeros((nvir, nvir), dtype=types[float])
    x15 += einsum("wai,wib->ab", lu11, u11)
    x18 = np.zeros((nvir, nvir), dtype=types[float])
    x18 += einsum("ab->ab", x15)
    x28 = np.zeros((nvir, nvir), dtype=types[float])
    x28 += einsum("ab->ab", x15)
    del x15
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x16 += einsum("abij->jiab", l2) * -0.5
    x16 += einsum("abij->jiba", l2)
    x17 = np.zeros((nvir, nvir), dtype=types[float])
    x17 += einsum("ijab,ijac->cb", t2, x16) * 2
    del x16
    x18 += einsum("ab->ab", x17)
    rdm_eb_des_ov += einsum("ab,wia->wib", x18, u11) * -2
    del x18
    x28 += einsum("ab->ab", x17)
    del x17
    x22 = np.zeros((nbos, nbos), dtype=types[float])
    x22 += einsum("wx,yw->xy", ls2, s2)
    x22 += einsum("wai,xia->wx", lu11, u11) * 2
    rdm_eb_des_ov += einsum("wx,wia->xia", x22, u11) * 2
    del x22
    x23 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x23 += einsum("ia,abjk->kjib", t1, l2)
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x24 += einsum("ijka->ijka", x23) * -0.5
    x24 += einsum("ijka->jika", x23)
    del x23
    x27 += einsum("ijab,jika->kb", t2, x24) * 2.0000000000000013
    del x24
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum("ijab->jiab", t2) * 2
    x25 += einsum("ijab->jiba", t2) * -1
    x27 += einsum("ai,ijba->jb", l1, x25) * -1
    del x25
    x27 += einsum("ia->ia", t1) * -1
    x27 += einsum("w,wia->ia", ls1, u11) * -1
    rdm_eb_des_ov += einsum("w,ia->wia", s1, x27) * -2
    del x27
    x28 += einsum("ai,ib->ab", l1, t1)
    rdm_eb_des_vv += einsum("w,ab->wab", s1, x28) * 2
    del x28
    rdm_eb_cre_oo += einsum("w,ij->wji", ls1, delta_oo) * 2
    rdm_eb_cre_ov += einsum("w,ia->wia", ls1, t1) * 2
    rdm_eb_cre_ov += einsum("wx,xia->wia", ls2, u11) * 2
    rdm_eb_cre_vo = np.zeros((nbos, nvir, nocc), dtype=types[float])
    rdm_eb_cre_vo += einsum("wai->wai", lu11) * 2
    rdm_eb_cre_vv = np.zeros((nbos, nvir, nvir), dtype=types[float])
    rdm_eb_cre_vv += einsum("ia,wbi->wba", t1, lu11) * 2
    rdm_eb_des_ov += einsum("wia->wia", u11) * 2
    rdm_eb_des_vo += einsum("w,ai->wai", s1, l1) * 2
    rdm_eb_des_vv += einsum("ai,wib->wab", l1, u11) * 2

    rdm_eb = np.array([
            np.block([[rdm_eb_cre_oo, rdm_eb_cre_ov], [rdm_eb_cre_vo, rdm_eb_cre_vv]]),
            np.block([[rdm_eb_des_oo, rdm_eb_des_ov], [rdm_eb_des_vo, rdm_eb_des_vv]]),
    ])

    return rdm_eb

