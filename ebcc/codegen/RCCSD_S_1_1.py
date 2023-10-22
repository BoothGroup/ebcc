# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, u11=None, **kwargs):
    # Energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum("iajb->jiab", v.ovov)
    x0 += einsum("iajb->jiba", v.ovov) * -0.5
    e_cc = 0
    e_cc += einsum("ijab,ijba->", t2, x0) * 2
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum("iajb->jiab", v.ovov) * -0.5
    x1 += einsum("iajb->jiba", v.ovov)
    x2 = np.zeros((nocc, nvir), dtype=types[float])
    x2 += einsum("ia,ijab->jb", t1, x1)
    del x1
    x2 += einsum("ia->ia", f.ov)
    e_cc += einsum("ia,ia->", t1, x2) * 2
    del x2
    x3 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x3 += einsum("wia->wia", u11)
    x3 += einsum("w,ia->wia", s1, t1)
    e_cc += einsum("wia,wia->", g.bov, x3) * 2
    del x3
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
    x0 += einsum("ia,jakb->ikjb", t1, v.ovov)
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum("ijka->ijka", x0) * 2
    x1 += einsum("ijka->ikja", x0) * -1
    x27 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x27 += einsum("ia,jkla->jilk", t1, x0)
    x88 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x88 += einsum("ijkl->lkji", x27)
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum("ijab,klji->lkab", t2, x27)
    del x27
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x35 += einsum("ijab,kjla->kilb", t2, x0)
    x41 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x41 -= einsum("ijka->ikja", x35)
    del x35
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x36 -= einsum("ijka->ijka", x0)
    x36 += einsum("ijka->ikja", x0) * 2
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x37 += einsum("ijab,klia->jklb", t2, x36)
    del x36
    x41 += einsum("ijka->jkia", x37)
    del x37
    x65 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x65 += einsum("ijab,klja->kilb", t2, x0)
    del x0
    x69 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x69 -= einsum("ijka->kija", x65)
    del x65
    x1 += einsum("ijka->jika", v.ooov) * -1
    x1 += einsum("ijka->jkia", v.ooov) * 2
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum("ijab,kija->kb", t2, x1) * -2
    del x1
    x2 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x2 += einsum("iabc->ibac", v.ovvv) * -0.5
    x2 += einsum("iabc->ibca", v.ovvv)
    t1new += einsum("ijab,icba->jc", t2, x2) * 4
    del x2
    x3 = np.zeros((nocc, nvir), dtype=types[float])
    x3 += einsum("w,wia->ia", s1, g.bov)
    x6 = np.zeros((nocc, nvir), dtype=types[float])
    x6 += einsum("ia->ia", x3)
    x19 = np.zeros((nocc, nvir), dtype=types[float])
    x19 += einsum("ia->ia", x3) * 0.5
    x98 = np.zeros((nocc, nvir), dtype=types[float])
    x98 += einsum("ia->ia", x3)
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum("iajb->jiab", v.ovov) * 2
    x4 -= einsum("iajb->jiba", v.ovov)
    x5 = np.zeros((nocc, nvir), dtype=types[float])
    x5 += einsum("ia,ijba->jb", t1, x4)
    x6 += einsum("ia->ia", x5)
    del x5
    x92 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x92 += einsum("wia,ijba->wjb", u11, x4)
    del x4
    x93 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x93 += einsum("wia->wia", x92)
    del x92
    x6 += einsum("ia->ia", f.ov)
    x68 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x68 += einsum("ia,jkab->jkib", x6, t2)
    x69 += einsum("ijka->kjia", x68)
    del x68
    x71 = np.zeros((nocc, nocc), dtype=types[float])
    x71 += einsum("ia,ja->ij", t1, x6)
    x72 = np.zeros((nocc, nocc), dtype=types[float])
    x72 += einsum("ij->ij", x71)
    del x71
    s1new = np.zeros((nbos), dtype=types[float])
    s1new += einsum("ia,wia->w", x6, u11) * 2
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 -= einsum("ijab->jiab", t2)
    x7 += einsum("ijab->jiba", t2) * 2
    t1new += einsum("ia,ijab->jb", x6, x7) * 2
    del x6
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum("iabj->ijba", v.ovvo) * 2
    x8 -= einsum("ijab->ijab", v.oovv)
    t1new += einsum("ia,ijba->jb", t1, x8) * 2
    u11new = np.zeros((nbos, nocc, nvir), dtype=types[float])
    u11new += einsum("wia,ijba->wjb", u11, x8)
    del x8
    x9 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x9 += einsum("ia,wja->wji", t1, g.bov)
    x10 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x10 += einsum("wij->wij", x9)
    del x9
    x10 += einsum("wij->wij", g.boo)
    x49 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x49 += einsum("ia,wij->wja", t1, x10)
    x50 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x50 -= einsum("wia->wia", x49)
    del x49
    t1new -= einsum("wia,wij->ja", u11, x10) * 2
    del x10
    x11 = np.zeros((nocc, nocc), dtype=types[float])
    x11 += einsum("w,wij->ij", s1, g.boo)
    x21 = np.zeros((nocc, nocc), dtype=types[float])
    x21 += einsum("ij->ij", x11)
    x72 += einsum("ij->ji", x11)
    del x11
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum("wia,wja->ij", g.bov, u11)
    x21 += einsum("ij->ij", x12)
    x54 = np.zeros((nocc, nocc), dtype=types[float])
    x54 += einsum("ij->ij", x12)
    del x12
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x13 += einsum("iajb->jiab", v.ovov) * -0.5
    x13 += einsum("iajb->jiba", v.ovov)
    x14 = np.zeros((nocc, nocc), dtype=types[float])
    x14 += einsum("ijab,ikab->jk", t2, x13) * 2
    x21 += einsum("ij->ji", x14)
    del x14
    x80 = np.zeros((nvir, nvir), dtype=types[float])
    x80 += einsum("ijab,ijac->bc", t2, x13)
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum("ab,ijbc->ijca", x80, t2) * 2
    del x80
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum("ijab->jiab", x81)
    del x81
    x82 = np.zeros((nocc, nocc), dtype=types[float])
    x82 += einsum("ijab,ikab->jk", t2, x13)
    del x13
    x83 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x83 += einsum("ij,jkab->kiab", x82, t2) * 2
    del x82
    x84 += einsum("ijab->ijba", x83)
    del x83
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum("ijka->ikja", v.ooov)
    x15 += einsum("ijka->kija", v.ooov) * -0.5
    x16 = np.zeros((nocc, nocc), dtype=types[float])
    x16 += einsum("ia,jika->jk", t1, x15) * 2
    del x15
    x21 += einsum("ij->ij", x16)
    del x16
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum("iajb->jiab", v.ovov)
    x17 += einsum("iajb->jiba", v.ovov) * -0.5
    x18 = np.zeros((nocc, nvir), dtype=types[float])
    x18 += einsum("ia,ijba->jb", t1, x17)
    x19 += einsum("ia->ia", x18)
    del x18
    x96 = np.zeros((nocc, nvir), dtype=types[float])
    x96 += einsum("ia,ijba->jb", t1, x17) * 2
    del x17
    x97 = np.zeros((nvir, nvir), dtype=types[float])
    x97 += einsum("ia,ib->ab", t1, x96) * -1
    del x96
    x19 += einsum("ia->ia", f.ov) * 0.5
    x20 = np.zeros((nocc, nocc), dtype=types[float])
    x20 += einsum("ia,ja->ij", t1, x19) * 2
    del x19
    x21 += einsum("ij->ji", x20)
    del x20
    x21 += einsum("ij->ij", f.oo)
    t1new += einsum("ia,ij->ja", t1, x21) * -2
    u11new += einsum("ij,wia->wja", x21, u11) * -1
    del x21
    x22 = np.zeros((nvir, nvir), dtype=types[float])
    x22 += einsum("w,wab->ab", s1, g.bvv)
    x25 = np.zeros((nvir, nvir), dtype=types[float])
    x25 += einsum("ab->ab", x22)
    x74 = np.zeros((nvir, nvir), dtype=types[float])
    x74 += einsum("ab->ab", x22)
    x97 += einsum("ab->ab", x22)
    del x22
    x23 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x23 -= einsum("iabc->ibac", v.ovvv)
    x23 += einsum("iabc->ibca", v.ovvv) * 2
    x24 = np.zeros((nvir, nvir), dtype=types[float])
    x24 += einsum("ia,ibca->bc", t1, x23)
    x25 += einsum("ab->ab", x24)
    x44 = np.zeros((nvir, nvir), dtype=types[float])
    x44 -= einsum("ab->ab", x24)
    del x24
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum("ia,jbac->ijbc", t1, x23)
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum("ijab,kica->jkbc", t2, x59)
    del x59
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 -= einsum("ijab->jiab", x60)
    del x60
    x100 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x100 += einsum("wia,ibca->wbc", u11, x23)
    del x23
    x25 += einsum("ab->ab", f.vv)
    t1new += einsum("ia,ba->ib", t1, x25) * 2
    del x25
    x26 = np.zeros((nbos), dtype=types[float])
    x26 += einsum("w->w", G)
    x26 += einsum("ia,wia->w", t1, g.bov) * 2
    t1new += einsum("w,wia->ia", x26, u11) * 2
    del x26
    x28 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x28 += einsum("ia,bacd->icbd", t1, v.vvvv)
    t2new += einsum("ia,jbca->ijbc", t1, x28)
    del x28
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum("ia,jabc->ijbc", t1, v.ovvv)
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum("ijab,kjca->kibc", t2, x29)
    del x29
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum("ijab->ijab", x30)
    del x30
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 -= einsum("iajb->jiab", v.ovov)
    x31 += einsum("iajb->jiba", v.ovov) * 2
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum("ijab,ikbc->jkac", t2, x31)
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum("ijab,kica->jkbc", t2, x32)
    del x32
    x56 += einsum("ijab->ijab", x33)
    del x33
    x85 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x85 += einsum("ijab,ikac->jkbc", t2, x31) * 2
    del x31
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum("ijab,jkla->ilkb", t2, v.ooov)
    x41 -= einsum("ijka->ijka", x34)
    del x34
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum("ia,jbca->ijcb", t1, v.ovvv)
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum("ijab->jiab", x38)
    x58 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x58 += einsum("ijab,kjca->kibc", t2, x38)
    del x38
    x76 += einsum("ijab->ijab", x58)
    del x58
    x39 += einsum("iabj->ijba", v.ovvo)
    x40 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x40 += einsum("ia,jkba->ijkb", t1, x39)
    del x39
    x41 += einsum("ijka->ijka", x40)
    del x40
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum("ia,jikb->jkab", t1, x41)
    del x41
    x56 += einsum("ijab->ijab", x42)
    del x42
    x43 = np.zeros((nvir, nvir), dtype=types[float])
    x43 += einsum("wia,wib->ab", g.bov, u11)
    x44 += einsum("ab->ba", x43)
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum("ab,ijbc->ijca", x44, t2)
    del x44
    x56 += einsum("ijab->jiab", x45)
    del x45
    x97 += einsum("ab->ba", x43) * -1
    del x43
    x46 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x46 += einsum("ia,wba->wib", t1, g.bvv)
    x50 += einsum("wia->wia", x46)
    del x46
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x47 += einsum("ijab->jiab", t2) * 2
    x47 -= einsum("ijab->jiba", t2)
    x48 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x48 += einsum("wia,ijba->wjb", g.bov, x47)
    del x47
    x50 += einsum("wia->wia", x48)
    del x48
    x50 += einsum("wai->wia", g.bvo)
    x51 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x51 += einsum("wia,wjb->ijab", u11, x50)
    del x50
    x56 -= einsum("ijab->jiba", x51)
    del x51
    x52 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x52 += einsum("ijka->ikja", v.ooov) * 2
    x52 -= einsum("ijka->kija", v.ooov)
    x53 = np.zeros((nocc, nocc), dtype=types[float])
    x53 += einsum("ia,jika->jk", t1, x52)
    x54 += einsum("ij->ij", x53)
    del x53
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum("ij,ikab->kjab", x54, t2)
    del x54
    x56 += einsum("ijab->ijba", x55)
    del x55
    t2new -= einsum("ijab->ijab", x56)
    t2new -= einsum("ijab->jiba", x56)
    del x56
    x99 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x99 += einsum("wia,jika->wjk", u11, x52)
    del x52
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum("ia,bjca->ijbc", t1, v.vovv)
    x76 -= einsum("ijab->ijab", x57)
    del x57
    x61 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x61 += einsum("ia,jkba->ijkb", t1, v.oovv)
    x69 += einsum("ijka->jika", x61)
    del x61
    x62 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x62 += einsum("ijab,klja->iklb", t2, v.ooov)
    x69 -= einsum("ijka->jika", x62)
    del x62
    x63 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x63 += einsum("ia,jkla->ijlk", t1, v.ooov)
    x64 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x64 += einsum("ia,jkil->jkla", t1, x63)
    x69 -= einsum("ijka->jika", x64)
    del x64
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum("ijab,kijl->klab", t2, x63)
    del x63
    t2new += einsum("ijab->ijba", x77)
    t2new += einsum("ijab->jiab", x77)
    del x77
    x66 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x66 -= einsum("ijka->ikja", v.ooov)
    x66 += einsum("ijka->kija", v.ooov) * 2
    x67 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x67 += einsum("ijab,ikla->jklb", t2, x66)
    del x66
    x69 += einsum("ijka->jika", x67)
    del x67
    x70 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x70 += einsum("ia,ijkb->jkab", t1, x69)
    del x69
    x76 += einsum("ijab->ijab", x70)
    del x70
    x72 += einsum("ij->ji", f.oo)
    x73 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum("ij,jkab->kiab", x72, t2)
    del x72
    x76 += einsum("ijab->jiba", x73)
    del x73
    x74 += einsum("ab->ab", f.vv)
    x75 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum("ab,ijbc->ijca", x74, t2)
    del x74
    x76 -= einsum("ijab->jiba", x75)
    del x75
    t2new -= einsum("ijab->ijba", x76)
    t2new -= einsum("ijab->jiab", x76)
    del x76
    x78 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x78 += einsum("ijab,kacb->ijkc", t2, v.ovvv)
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum("ia,jkib->jkab", t1, x78)
    del x78
    x84 += einsum("ijab->ijab", x79)
    del x79
    t2new += einsum("ijab->ijab", x84) * -1
    t2new += einsum("ijab->jiba", x84) * -1
    del x84
    x85 += einsum("iabj->jiba", v.ovvo) * 2
    x85 -= einsum("ijab->jiab", v.oovv)
    t2new += einsum("ijab,kica->kjcb", t2, x85)
    del x85
    x86 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x86 += einsum("ijab,kalb->ijkl", t2, v.ovov)
    x87 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x87 += einsum("ijkl->lkji", x86)
    x88 += einsum("ijkl->lkji", x86)
    del x86
    x87 += einsum("ijkl->kilj", v.oooo)
    t2new += einsum("ijab,ijkl->lkba", t2, x87)
    del x87
    x88 += einsum("ijkl->kilj", v.oooo)
    x89 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x89 += einsum("ia,ijkl->lkja", t1, x88) * 0.9999999999999993
    del x88
    x89 += einsum("ijak->jkia", v.oovo) * -0.9999999999999993
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
    x93 += einsum("wia->wia", gc.bov)
    x99 += einsum("ia,wja->wji", t1, x93)
    u11new += einsum("wia,ijab->wjb", x93, x7)
    del x93
    del x7
    x94 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x94 += einsum("iajb->jiab", v.ovov) * -1
    x94 += einsum("iajb->jiba", v.ovov) * 2
    x97 += einsum("ijab,ijac->bc", t2, x94) * -1
    del x94
    x95 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x95 += einsum("iabc->ibac", v.ovvv)
    x95 += einsum("iabc->ibca", v.ovvv) * -0.5
    x97 += einsum("ia,ibac->bc", t1, x95) * 2
    del x95
    x97 += einsum("ab->ab", f.vv)
    u11new += einsum("ab,wib->wia", x97, u11)
    del x97
    x98 += einsum("ia->ia", f.ov)
    x99 += einsum("ia,wja->wij", x98, u11)
    del x98
    x99 += einsum("wij->wij", gc.boo)
    u11new -= einsum("ia,wij->wja", t1, x99)
    del x99
    x100 += einsum("wab->wab", gc.bvv)
    u11new += einsum("ia,wba->wib", t1, x100)
    del x100
    x101 = np.zeros((nbos, nbos), dtype=types[float])
    x101 += einsum("wx->wx", w)
    x101 += einsum("wia,xia->wx", g.bov, u11) * 2
    u11new += einsum("wx,wia->xia", x101, u11)
    del x101
    t1new += einsum("wab,wib->ia", g.bvv, u11) * 2
    t1new += einsum("w,wai->ia", s1, g.bvo) * 2
    t1new += einsum("ai->ia", f.vo) * 2
    t2new -= einsum("ijab,jack->kicb", t2, v.ovvo)
    t2new -= einsum("ijab,jkca->kibc", t2, v.oovv)
    t2new -= einsum("ijab,jkcb->ikac", t2, v.oovv)
    t2new += einsum("ijab,jbck->ikac", t2, v.ovvo) * 2
    t2new += einsum("ijab,cadb->jidc", t2, v.vvvv)
    t2new -= einsum("ia,ijbk->kjba", t1, v.oovo)
    t2new += einsum("aibj->jiba", v.vovo)
    s1new += einsum("ia,wia->w", t1, gc.bov) * 2
    s1new += einsum("w,wx->x", s1, w)
    s1new += einsum("w->w", G)
    u11new += einsum("wai->wia", gc.bvo)

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
    x0 = np.zeros((nocc, nocc), dtype=types[float])
    x0 += einsum("ai,ja->ij", l1, t1)
    x87 = np.zeros((nocc, nocc), dtype=types[float])
    x87 += einsum("ij->ij", x0) * 0.5
    x95 = np.zeros((nocc, nocc), dtype=types[float])
    x95 += einsum("ij->ij", x0)
    x114 = np.zeros((nocc, nocc), dtype=types[float])
    x114 += einsum("ij->ij", x0)
    ls1new = np.zeros((nbos), dtype=types[float])
    ls1new -= einsum("ij,wji->w", x0, g.boo) * 2
    x1 = np.zeros((nocc, nvir), dtype=types[float])
    x1 += einsum("w,wia->ia", s1, g.bov)
    x2 = np.zeros((nocc, nvir), dtype=types[float])
    x2 += einsum("ia->ia", x1)
    x60 = np.zeros((nocc, nvir), dtype=types[float])
    x60 += einsum("ia->ia", x1)
    x69 = np.zeros((nocc, nvir), dtype=types[float])
    x69 += einsum("ia->ia", x1) * 0.5
    x99 = np.zeros((nocc, nvir), dtype=types[float])
    x99 += einsum("ia->ia", x1)
    x141 = np.zeros((nocc, nvir), dtype=types[float])
    x141 += einsum("ia->ia", x1)
    l1new = np.zeros((nvir, nocc), dtype=types[float])
    l1new -= einsum("ij,ja->ai", x0, x1)
    del x0
    del x1
    x2 += einsum("ia->ia", f.ov)
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x3 += einsum("ia,jkab->jkib", x2, t2) * 2
    del x2
    x17 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x17 += einsum("ijka->jkia", x3) * -1
    x17 += einsum("ijka->ikja", x3) * 0.5
    del x3
    x4 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x4 += einsum("ijab,kbca->jikc", t2, v.ovvv)
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x7 += einsum("ijka->ijka", x4)
    del x4
    x5 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x5 += einsum("ijab,kalb->ijkl", t2, v.ovov)
    x6 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x6 += einsum("ia,jkli->jkla", t1, x5)
    x7 += einsum("ijka->ijka", x6) * -1
    del x6
    x17 += einsum("ijka->ikja", x7)
    x17 += einsum("ijka->jkia", x7) * -2
    del x7
    x169 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x169 += einsum("ijkl->lkji", x5)
    del x5
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x8 += einsum("iajb->jiba", v.ovov)
    x8 += einsum("iajb->jiab", v.ovov) * -0.5
    x9 = np.zeros((nocc, nvir), dtype=types[float])
    x9 += einsum("ia,ijab->jb", t1, x8)
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum("ia,jkab->jkib", x9, t2) * 4
    x17 += einsum("ijka->jkia", x10) * -1
    x17 += einsum("ijka->ikja", x10) * 0.5
    del x10
    x69 += einsum("ia->ia", x9)
    del x9
    x59 = np.zeros((nocc, nvir), dtype=types[float])
    x59 += einsum("ia,ijab->jb", t1, x8) * 2
    x60 += einsum("ia->ia", x59)
    x177 = np.zeros((nvir, nvir), dtype=types[float])
    x177 += einsum("ia,ib->ab", t1, x59) * -1
    del x59
    x66 = np.zeros((nocc, nocc), dtype=types[float])
    x66 += einsum("ijab,ikab->jk", t2, x8) * 2
    x71 = np.zeros((nocc, nocc), dtype=types[float])
    x71 += einsum("ij->ji", x66)
    del x66
    x159 = np.zeros((nvir, nvir), dtype=types[float])
    x159 += einsum("ijab,ijac->bc", t2, x8)
    del x8
    x160 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x160 += einsum("ab,acij->ijcb", x159, l2) * 2
    del x159
    x164 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x164 += einsum("ijab->jiab", x160)
    del x160
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum("ia,jbka->ijkb", t1, v.ovov)
    x12 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x12 += einsum("ijka->ijka", x11) * -0.5
    x12 += einsum("ijka->ikja", x11)
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x13 += einsum("ijka->ijka", x11) * 2
    x13 += einsum("ijka->ikja", x11) * -1
    x20 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x20 += einsum("ia,jkla->jilk", t1, x11)
    x21 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x21 += einsum("ia,jkil->kjla", t1, x20)
    x22 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x22 += einsum("ijka->ijka", x21) * 2.0000000000000013
    del x21
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum("abij,ijkl->abkl", l2, x20)
    del x20
    x23 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x23 += einsum("ijka->ijka", x11)
    x23 += einsum("ijka->ikja", x11) * -0.5
    x58 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x58 += einsum("ijka->jkia", x11) * -1
    x58 += einsum("ijka->kjia", x11) * 2
    x144 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x144 += einsum("ai,ijkb->jkab", l1, x11)
    x158 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x158 += einsum("ijab->ijab", x144)
    del x144
    x12 += einsum("ijka->jika", v.ooov)
    x12 += einsum("ijka->jkia", v.ooov) * -0.5
    x17 += einsum("ijab,kilb->klja", t2, x12) * 2
    del x12
    x13 += einsum("ijka->jika", v.ooov) * -1
    x13 += einsum("ijka->jkia", v.ooov) * 2
    x17 += einsum("ijab,kila->kljb", t2, x13)
    del x13
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum("ijab->ijab", v.oovv) * 2
    x14 += einsum("iabj->ijba", v.ovvo) * -1
    x17 += einsum("ia,jkba->ijkb", t1, x14) * -1
    del x14
    x15 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x15 += einsum("ia,jkla->ijlk", t1, v.ooov)
    x16 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x16 += einsum("ijkl->jkil", x15) * -0.5
    x16 += einsum("ijkl->kjil", x15)
    x26 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x26 += einsum("ijkl->ijkl", x15)
    x26 += einsum("ijkl->ikjl", x15) * -0.5
    x27 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x27 += einsum("ia,jikl->jkla", t1, x26) * 2
    del x26
    x146 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x146 += einsum("abij,jkli->klba", l2, x15)
    del x15
    x158 -= einsum("ijab->ijab", x146)
    del x146
    x16 += einsum("ijkl->kilj", v.oooo)
    x16 += einsum("ijkl->kijl", v.oooo) * -0.5
    x17 += einsum("ia,ijkl->kjla", t1, x16) * 2
    del x16
    x17 += einsum("ijak->jika", v.oovo)
    x17 += einsum("ijak->kija", v.oovo) * -2
    l1new += einsum("abij,jkib->ak", l2, x17)
    del x17
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum("ia,jabc->ijbc", t1, v.ovvv)
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x19 += einsum("ia,jkba->jikb", t1, x18)
    x22 += einsum("ijka->ijka", x19) * -2
    del x19
    x27 += einsum("ijka->ikja", x22)
    x27 += einsum("ijka->jkia", x22) * -0.5
    del x22
    x129 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x129 += einsum("ijab->ijab", x18)
    del x18
    x23 += einsum("ijka->jika", v.ooov) * -0.5
    x23 += einsum("ijka->jkia", v.ooov)
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x24 += einsum("ijab->jiba", t2) * -1
    x24 += einsum("ijab->jiab", t2) * 2
    x27 += einsum("ijka,jlba->iklb", x23, x24) * -2
    del x23
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum("iabj->ijba", v.ovvo) * 2
    x25 += einsum("ijab->ijab", v.oovv) * -1
    x27 += einsum("ia,jkba->ijkb", t1, x25) * -1
    del x25
    l1new += einsum("abij,jkia->bk", l2, x27)
    del x27
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum("ijab->jiba", t2) * 2
    x28 -= einsum("ijab->jiab", t2)
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 -= einsum("abij,ikac->jkbc", l2, x28)
    x125 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x125 += einsum("iajb,ikac->jkbc", v.ovov, x28)
    x126 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x126 += einsum("ijab->jiba", x125)
    del x125
    x173 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x173 += einsum("wia,ijab->wjb", g.bov, x28)
    x174 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x174 -= einsum("wai,ijab->wjb", lu11, x28)
    del x28
    x29 += einsum("abij,kicb->jkac", l2, t2)
    x30 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x30 -= einsum("iabc->ibac", v.ovvv)
    x30 += einsum("iabc->ibca", v.ovvv) * 2
    l1new -= einsum("ijab,jacb->ci", x29, x30)
    del x29
    del x30
    x31 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x31 += einsum("ia,bacd->icbd", t1, v.vvvv)
    x32 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x32 += einsum("iabc->ibac", x31) * -0.5
    x32 += einsum("iabc->iabc", x31)
    del x31
    x32 += einsum("aibc->ibac", v.vovv)
    x32 += einsum("aibc->iabc", v.vovv) * -0.5
    l1new += einsum("abij,ibac->cj", l2, x32) * 2
    del x32
    x33 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x33 += einsum("ai,jkba->ijkb", l1, t2)
    x41 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x41 += einsum("ijka->ijka", x33) * -2
    x41 += einsum("ijka->ikja", x33)
    del x33
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum("ia,abjk->kjib", t1, l2)
    x35 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x35 += einsum("ia,jkla->jkil", t1, x34)
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x36 += einsum("ia,jikl->jkla", t1, x35)
    x41 += einsum("ijka->ijka", x36) * -1
    x41 += einsum("ijka->ikja", x36) * 2.0000000000000013
    del x36
    x52 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x52 += einsum("ijkl->ijlk", x35) * 2
    x52 += einsum("ijkl->ijkl", x35) * -1
    l1new += einsum("ijka,ljik->al", v.ooov, x52)
    del x52
    l2new += einsum("iajb,klij->balk", v.ovov, x35)
    del x35
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x37 += einsum("ijka->ijka", x34) * -0.5
    x37 += einsum("ijka->jika", x34)
    x41 += einsum("ijab,kila->kljb", t2, x37) * 2
    x45 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x45 += einsum("ijab,kilb->klja", x24, x37) * -1
    del x24
    del x37
    x38 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x38 += einsum("ijka->ijka", x34) * 2
    x38 += einsum("ijka->jika", x34) * -1
    x41 += einsum("ijab,kilb->klja", t2, x38)
    del x38
    x46 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x46 += einsum("ijka->ijka", x34) * 2
    x46 -= einsum("ijka->jika", x34)
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x47 += einsum("ia,ijkb->jkba", t1, x46)
    l1new -= einsum("iabc,jibc->aj", v.ovvv, x47)
    del x47
    x131 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x131 += einsum("ijka,likb->jlab", x11, x46)
    x142 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x142 -= einsum("ijab->jiba", x131)
    del x131
    l1new -= einsum("iabj,jkib->ak", v.ovvo, x46)
    del x46
    x48 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x48 -= einsum("ijka->ijka", x34)
    x48 += einsum("ijka->jika", x34) * 2
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum("ia,ijkb->jkba", t1, x48)
    l1new -= einsum("iabc,jiba->cj", v.ovvv, x49)
    del x49
    l1new -= einsum("ijab,jkia->bk", v.oovv, x48)
    del x48
    x120 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x120 += einsum("ijka,jlib->lkba", v.ooov, x34)
    x142 += einsum("ijab->ijab", x120)
    del x120
    x121 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x121 += einsum("iabc,jkib->kjac", v.ovvv, x34)
    x142 -= einsum("ijab->ijab", x121)
    del x121
    x122 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x122 += einsum("ijka,lijb->lkba", x11, x34)
    x142 += einsum("ijab->ijab", x122)
    del x122
    x147 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x147 += einsum("ijka,jlkb->liba", v.ooov, x34)
    x158 -= einsum("ijab->ijab", x147)
    del x147
    x148 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x148 += einsum("ijka,iljb->lkba", x11, x34)
    del x11
    x158 -= einsum("ijab->ijab", x148)
    del x148
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum("ijab->jiba", t2)
    x39 += einsum("ijab->jiab", t2) * -0.5
    x40 = np.zeros((nocc, nocc), dtype=types[float])
    x40 += einsum("abij,ikab->jk", l2, x39)
    x41 += einsum("ia,jk->jkia", t1, x40) * 2
    l1new += einsum("iajb,kjib->ak", v.ovov, x41)
    del x41
    x87 += einsum("ij->ij", x40)
    x167 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x167 += einsum("ij,jakb->kiab", x40, v.ovov) * 2
    x168 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x168 += einsum("ijab->jiba", x167)
    del x167
    x171 = np.zeros((nocc, nocc), dtype=types[float])
    x171 += einsum("ij->ij", x40)
    l1new += einsum("ia,ji->aj", f.ov, x40) * -2
    del x40
    x44 = np.zeros((nocc, nocc), dtype=types[float])
    x44 += einsum("abij,ikab->jk", l2, x39) * 2
    x45 += einsum("ia,jk->jkia", t1, x44) * -1
    x95 += einsum("ij->ij", x44)
    x110 = np.zeros((nocc, nocc), dtype=types[float])
    x110 += einsum("ij->ij", x44)
    del x44
    x83 = np.zeros((nocc, nvir), dtype=types[float])
    x83 += einsum("ijka,jiba->kb", x34, x39)
    x89 = np.zeros((nocc, nvir), dtype=types[float])
    x89 += einsum("ia->ia", x83)
    del x83
    x42 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x42 += einsum("abij,klab->ijkl", l2, t2)
    x43 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x43 += einsum("ia,jikl->jkla", t1, x42)
    x45 += einsum("ijka->ijka", x43)
    x45 += einsum("ijka->ikja", x43) * -0.5
    del x43
    l1new += einsum("iajb,kjia->bk", v.ovov, x45) * 2
    del x45
    x53 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x53 += einsum("ijkl->jilk", x42)
    x53 += einsum("ijkl->jikl", x42) * -0.5
    l1new += einsum("ijka,jlik->al", v.ooov, x53) * 2
    del x53
    l2new += einsum("iajb,klji->ablk", v.ovov, x42)
    del x42
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum("abij,kibc->jkac", l2, t2)
    x51 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x51 += einsum("iabc->ibac", v.ovvv) * 2
    x51 -= einsum("iabc->ibca", v.ovvv)
    x103 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x103 += einsum("wia,ibac->wbc", u11, x51)
    x112 = np.zeros((nvir, nvir), dtype=types[float])
    x112 += einsum("ia,ibac->bc", t1, x51)
    x113 = np.zeros((nvir, nvir), dtype=types[float])
    x113 += einsum("ab->ab", x112)
    x137 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x137 += einsum("ab,acij->ijcb", x112, l2)
    del x112
    x142 += einsum("ijab->jiab", x137)
    del x137
    l1new -= einsum("ijab,jacb->ci", x50, x51)
    del x50
    x54 = np.zeros((nocc, nvir), dtype=types[float])
    x54 += einsum("w,wai->ia", s1, g.bvo)
    x78 = np.zeros((nocc, nvir), dtype=types[float])
    x78 += einsum("ia->ia", x54) * 0.5
    x170 = np.zeros((nocc, nvir), dtype=types[float])
    x170 += einsum("ia->ia", x54)
    del x54
    x55 = np.zeros((nocc, nvir), dtype=types[float])
    x55 += einsum("wab,wib->ia", g.bvv, u11)
    x78 += einsum("ia->ia", x55) * 0.5
    x170 += einsum("ia->ia", x55)
    del x55
    x56 = np.zeros((nocc, nvir), dtype=types[float])
    x56 += einsum("ijab,jkib->ka", t2, v.ooov)
    x78 += einsum("ia->ia", x56) * 0.5
    x170 += einsum("ia->ia", x56)
    del x56
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum("ijab->jiba", t2) * -0.5
    x57 += einsum("ijab->jiab", t2)
    x78 += einsum("iabc,ijca->jb", v.ovvv, x57)
    x93 = np.zeros((nvir, nvir), dtype=types[float])
    x93 += einsum("abij,ijbc->ac", l2, x57) * 2
    x94 = np.zeros((nvir, nvir), dtype=types[float])
    x94 += einsum("ab->ab", x93)
    x178 = np.zeros((nvir, nvir), dtype=types[float])
    x178 += einsum("ab->ab", x93)
    del x93
    x165 = np.zeros((nvir, nvir), dtype=types[float])
    x165 += einsum("abij,ijbc->ac", l2, x57)
    x166 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x166 += einsum("ab,ibjc->ijca", x165, v.ovov) * 2
    del x165
    x168 += einsum("ijab->jiba", x166)
    del x166
    l2new += einsum("ijab->baij", x168) * -1
    l2new += einsum("ijab->abji", x168) * -1
    del x168
    x170 += einsum("iabc,ijca->jb", v.ovvv, x57) * 2
    del x57
    x58 += einsum("ijka->ikja", v.ooov) * 2
    x78 += einsum("ijab,jika->kb", t2, x58) * -0.5
    x170 += einsum("ijab,jika->kb", t2, x58) * -1
    del x58
    x60 += einsum("ia->ia", f.ov)
    x78 += einsum("ia,ijab->jb", x60, x39)
    x170 += einsum("ia,ijab->jb", x60, x39) * 2
    del x60
    del x39
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum("iabj->ijba", v.ovvo)
    x61 += einsum("ijab->ijab", v.oovv) * -0.5
    x78 += einsum("ia,ijba->jb", t1, x61)
    x170 += einsum("ia,ijba->jb", t1, x61) * 2
    del x61
    x62 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x62 += einsum("ia,wja->wji", t1, g.bov)
    x63 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x63 += einsum("wij->wij", x62)
    x104 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x104 += einsum("wij->wij", x62)
    del x62
    x63 += einsum("wij->wij", g.boo)
    x78 += einsum("wia,wij->ja", u11, x63) * -0.5
    x170 += einsum("wia,wij->ja", u11, x63) * -1
    del x63
    x64 = np.zeros((nocc, nocc), dtype=types[float])
    x64 += einsum("w,wij->ij", s1, g.boo)
    x71 += einsum("ij->ij", x64)
    x154 = np.zeros((nocc, nocc), dtype=types[float])
    x154 += einsum("ij->ij", x64)
    del x64
    x65 = np.zeros((nocc, nocc), dtype=types[float])
    x65 += einsum("wia,wja->ij", g.bov, u11)
    x71 += einsum("ij->ij", x65)
    x154 += einsum("ij->ij", x65)
    del x65
    x67 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x67 += einsum("ijka->ikja", v.ooov) * -0.5
    x67 += einsum("ijka->kija", v.ooov)
    x68 = np.zeros((nocc, nocc), dtype=types[float])
    x68 += einsum("ia,ijka->jk", t1, x67) * 2
    del x67
    x71 += einsum("ij->ij", x68)
    del x68
    x69 += einsum("ia->ia", f.ov) * 0.5
    x70 = np.zeros((nocc, nocc), dtype=types[float])
    x70 += einsum("ia,ja->ij", t1, x69) * 2
    x71 += einsum("ij->ji", x70)
    del x70
    lu11new = np.zeros((nbos, nvir, nocc), dtype=types[float])
    lu11new += einsum("w,ia->wai", ls1, x69) * 4
    del x69
    x71 += einsum("ij->ij", f.oo)
    x78 += einsum("ia,ij->ja", t1, x71) * -0.5
    x170 += einsum("ia,ij->ja", t1, x71) * -1
    l1new += einsum("ai,ji->aj", l1, x71) * -1
    lu11new += einsum("ij,waj->wai", x71, lu11) * -1
    del x71
    x72 = np.zeros((nvir, nvir), dtype=types[float])
    x72 += einsum("w,wab->ab", s1, g.bvv)
    x75 = np.zeros((nvir, nvir), dtype=types[float])
    x75 += einsum("ab->ab", x72)
    x113 += einsum("ab->ab", x72)
    x151 = np.zeros((nvir, nvir), dtype=types[float])
    x151 += einsum("ab->ab", x72)
    x177 += einsum("ab->ab", x72)
    del x72
    x73 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x73 += einsum("iabc->ibac", v.ovvv)
    x73 += einsum("iabc->ibca", v.ovvv) * -0.5
    x74 = np.zeros((nvir, nvir), dtype=types[float])
    x74 += einsum("ia,ibac->bc", t1, x73) * 2
    del x73
    x75 += einsum("ab->ab", x74)
    x177 += einsum("ab->ab", x74)
    del x74
    x75 += einsum("ab->ab", f.vv)
    x78 += einsum("ia,ba->ib", t1, x75) * 0.5
    x170 += einsum("ia,ba->ib", t1, x75)
    del x75
    x76 = np.zeros((nbos), dtype=types[float])
    x76 += einsum("ia,wia->w", t1, g.bov)
    x77 = np.zeros((nbos), dtype=types[float])
    x77 += einsum("w->w", x76) * 2
    del x76
    x77 += einsum("w->w", G)
    x78 += einsum("w,wia->ia", x77, u11) * 0.5
    x170 += einsum("w,wia->ia", x77, u11)
    del x77
    x78 += einsum("ai->ia", f.vo) * 0.5
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum("abij->jiba", l2) * 2
    x79 += einsum("abij->jiab", l2) * -1
    l1new += einsum("ia,ijab->bj", x78, x79) * 2
    del x78
    del x79
    x80 = np.zeros((nocc, nvir), dtype=types[float])
    x80 += einsum("w,wia->ia", ls1, u11)
    x89 += einsum("ia->ia", x80) * -0.5
    del x80
    x81 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x81 += einsum("ia,waj->wji", t1, lu11)
    x82 = np.zeros((nocc, nvir), dtype=types[float])
    x82 += einsum("wia,wij->ja", u11, x81)
    x89 += einsum("ia->ia", x82) * 0.5
    del x82
    x174 += einsum("ia,wij->wja", t1, x81)
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum("ijab->jiba", t2) * 2
    x84 += einsum("ijab->jiab", t2) * -1
    x85 = np.zeros((nocc, nvir), dtype=types[float])
    x85 += einsum("ai,ijab->jb", l1, x84) * 0.5
    del x84
    x89 += einsum("ia->ia", x85) * -1
    del x85
    x86 = np.zeros((nocc, nocc), dtype=types[float])
    x86 += einsum("wai,wja->ij", lu11, u11)
    x87 += einsum("ij->ij", x86) * 0.5
    x88 = np.zeros((nocc, nvir), dtype=types[float])
    x88 += einsum("ia,ij->ja", t1, x87)
    del x87
    x89 += einsum("ia->ia", x88)
    del x88
    x95 += einsum("ij->ij", x86)
    lu11new += einsum("ij,wja->wai", x95, g.bov) * -1
    x110 += einsum("ij->ij", x86)
    x111 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x111 += einsum("w,ij->wij", s1, x110) * 0.5
    del x110
    x114 += einsum("ij->ij", x86)
    x156 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x156 += einsum("ij,jakb->kiab", x114, v.ovov)
    x158 += einsum("ijab->jiba", x156)
    del x156
    x171 += einsum("ij->ij", x86) * 0.5
    del x86
    ls1new += einsum("ij,wji->w", x171, g.boo) * -4
    del x171
    x89 += einsum("ia->ia", t1) * -0.5
    ls1new += einsum("ia,wia->w", x89, g.bov) * -4
    x90 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x90 += einsum("iajb->jiba", v.ovov) * 2
    x90 -= einsum("iajb->jiab", v.ovov)
    x98 = np.zeros((nocc, nvir), dtype=types[float])
    x98 += einsum("ia,ijab->jb", t1, x90)
    x99 += einsum("ia->ia", x98)
    x115 = np.zeros((nocc, nvir), dtype=types[float])
    x115 += einsum("ia->ia", x98)
    x134 = np.zeros((nocc, nocc), dtype=types[float])
    x134 += einsum("ia,ja->ij", t1, x98)
    x135 = np.zeros((nocc, nocc), dtype=types[float])
    x135 += einsum("ij->ji", x134)
    del x134
    x138 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x138 += einsum("ia,jkib->jkba", x98, x34)
    x142 -= einsum("ijab->ijab", x138)
    del x138
    x142 += einsum("ai,jb->ijab", l1, x98)
    del x98
    x100 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x100 += einsum("wia,ijab->wjb", u11, x90)
    x101 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x101 += einsum("wia->wia", x100)
    x139 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x139 += einsum("wai,wjb->ijab", lu11, x100)
    del x100
    x142 += einsum("ijab->ijab", x139)
    del x139
    l1new += einsum("ia,ijab->bj", x89, x90) * -2
    del x89
    lu11new -= einsum("wia,ijab->wbj", x174, x90)
    del x90
    del x174
    x91 = np.zeros((nvir, nvir), dtype=types[float])
    x91 += einsum("ai,ib->ab", l1, t1)
    x94 += einsum("ab->ab", x91)
    del x91
    x92 = np.zeros((nvir, nvir), dtype=types[float])
    x92 += einsum("wai,wib->ab", lu11, u11)
    x94 += einsum("ab->ab", x92)
    l1new += einsum("ab,iacb->ci", x94, x51)
    ls1new += einsum("ab,wab->w", x94, g.bvv) * 2
    del x94
    x145 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x145 += einsum("ab,ibjc->jiac", x92, v.ovov)
    x158 += einsum("ijab->ijab", x145)
    del x145
    x178 += einsum("ab->ab", x92)
    del x92
    lu11new += einsum("ab,wib->wai", x178, g.bov) * -1
    del x178
    x96 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x96 += einsum("ijka->ikja", v.ooov) * 2
    x96 -= einsum("ijka->kija", v.ooov)
    l1new += einsum("ij,jkia->ak", x95, x96) * -1
    del x95
    lu11new -= einsum("wij,jkia->wak", x81, x96)
    del x96
    x97 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x97 -= einsum("ijka->ikja", v.ooov)
    x97 += einsum("ijka->kija", v.ooov) * 2
    x102 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x102 += einsum("wia,ijka->wjk", u11, x97)
    x132 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x132 += einsum("ijka,lkjb->ilab", x34, x97)
    x142 -= einsum("ijab->ijab", x132)
    del x132
    x133 = np.zeros((nocc, nocc), dtype=types[float])
    x133 += einsum("ia,ijka->jk", t1, x97)
    del x97
    x135 += einsum("ij->ij", x133)
    del x133
    x136 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x136 += einsum("ij,abjk->kiab", x135, l2)
    del x135
    x142 -= einsum("ijab->ijba", x136)
    del x136
    x99 += einsum("ia->ia", f.ov)
    x102 += einsum("ia,wja->wij", x99, u11)
    x116 = np.zeros((nbos), dtype=types[float])
    x116 += einsum("ia,wia->w", x99, u11) * 2
    del x99
    x101 += einsum("wia->wia", gc.bov)
    x102 += einsum("ia,wja->wji", t1, x101)
    l1new -= einsum("wia,wji->aj", x101, x81)
    del x101
    x102 += einsum("wij->wij", gc.boo)
    l1new -= einsum("wai,wji->aj", lu11, x102)
    del x102
    x103 += einsum("wab->wab", gc.bvv)
    l1new += einsum("wai,wab->bi", lu11, x103)
    del x103
    x104 += einsum("wij->wij", g.boo)
    x173 -= einsum("ia,wij->wja", t1, x104)
    lu11new -= einsum("ai,wji->waj", l1, x104)
    x105 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x105 += einsum("abij->jiba", l2) * 2
    x105 -= einsum("abij->jiab", l2)
    x106 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x106 += einsum("wia,ijab->wjb", u11, x105)
    x140 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x140 += einsum("wia,wjb->ijab", g.bov, x106)
    x142 += einsum("ijab->ijab", x140)
    del x140
    l1new -= einsum("wij,wja->ai", x104, x106)
    del x104
    l1new += einsum("wab,wia->bi", g.bvv, x106)
    del x106
    x107 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x107 += einsum("iabj->ijba", v.ovvo) * 2
    x107 -= einsum("ijab->ijab", v.oovv)
    l1new += einsum("ai,jiab->bj", l1, x107)
    lu11new += einsum("wai,jiab->wbj", lu11, x107)
    del x107
    x108 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x108 += einsum("abij->jiba", l2)
    x108 += einsum("abij->jiab", l2) * -0.5
    x109 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x109 += einsum("wia,ijab->wjb", u11, x108)
    del x108
    x111 += einsum("ia,wja->wji", t1, x109)
    del x109
    x111 += einsum("ai,wja->wij", l1, u11) * 0.5
    l1new += einsum("wia,wji->aj", g.bov, x111) * -2
    del x111
    x113 += einsum("ab->ab", f.vv)
    l1new += einsum("ai,ab->bi", l1, x113)
    del x113
    x115 += einsum("ia->ia", f.ov)
    l1new -= einsum("ij,ja->ai", x114, x115)
    del x114
    del x115
    x116 += einsum("w->w", G)
    x116 += einsum("w,wx->x", s1, w)
    x116 += einsum("ia,wia->w", t1, gc.bov) * 2
    l1new += einsum("w,wai->ai", x116, lu11)
    del x116
    x117 = np.zeros((nbos), dtype=types[float])
    x117 += einsum("w->w", s1)
    x117 += einsum("ai,wia->w", l1, u11) * 2
    l1new += einsum("w,wia->ai", x117, g.bov)
    del x117
    x118 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x118 += einsum("wia,wbj->ijab", gc.bov, lu11)
    x142 += einsum("ijab->ijab", x118)
    del x118
    x119 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x119 += einsum("ai,jikb->jkab", l1, v.ooov)
    x142 -= einsum("ijab->ijab", x119)
    del x119
    x123 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x123 += einsum("ia,jbca->ijcb", t1, v.ovvv)
    x126 += einsum("ijab->ijab", x123)
    del x123
    x124 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x124 += einsum("ijab,icka->jkbc", t2, v.ovov)
    x126 -= einsum("ijab->ijab", x124)
    del x124
    x126 += einsum("iabj->jiba", v.ovvo)
    x127 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x127 += einsum("ijab,ikac->jkbc", x105, x126)
    del x126
    x142 += einsum("ijab->ijab", x127)
    del x127
    x128 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x128 += einsum("ijab,kbic->jkac", t2, v.ovov)
    x129 -= einsum("ijab->ijab", x128)
    del x128
    x129 += einsum("ijab->jiab", v.oovv)
    x130 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x130 += einsum("abij,ikac->jkbc", l2, x129)
    x142 -= einsum("ijab->ijab", x130)
    del x130
    x149 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x149 += einsum("abij,ikbc->jkac", l2, x129)
    del x129
    x158 += einsum("ijab->ijab", x149)
    del x149
    x141 += einsum("ia->ia", f.ov)
    x142 += einsum("ai,jb->jiba", l1, x141)
    l2new += einsum("ijab->abij", x142)
    l2new += einsum("ijab->baji", x142)
    del x142
    x153 = np.zeros((nocc, nocc), dtype=types[float])
    x153 += einsum("ia,ja->ij", t1, x141)
    x154 += einsum("ij->ji", x153)
    del x153
    x157 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x157 += einsum("ia,jkib->jkba", x141, x34)
    del x34
    x158 += einsum("ijab->ijba", x157)
    del x157
    lu11new -= einsum("ia,wji->waj", x141, x81)
    del x81
    del x141
    x143 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x143 += einsum("ai,jbac->ijbc", l1, v.ovvv)
    x158 -= einsum("ijab->ijab", x143)
    del x143
    x150 = np.zeros((nvir, nvir), dtype=types[float])
    x150 += einsum("wia,wib->ab", g.bov, u11)
    x151 -= einsum("ab->ba", x150)
    x177 += einsum("ab->ba", x150) * -1
    del x150
    x151 += einsum("ab->ab", f.vv)
    x152 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x152 += einsum("ab,acij->ijcb", x151, l2)
    del x151
    x158 -= einsum("ijab->jiba", x152)
    del x152
    x154 += einsum("ij->ij", f.oo)
    x155 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x155 += einsum("ij,abjk->kiab", x154, l2)
    del x154
    x158 += einsum("ijab->jiba", x155)
    del x155
    l2new -= einsum("ijab->baij", x158)
    l2new -= einsum("ijab->abji", x158)
    del x158
    x161 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x161 += einsum("iajb->jiba", v.ovov) * -0.5
    x161 += einsum("iajb->jiab", v.ovov)
    x162 = np.zeros((nocc, nocc), dtype=types[float])
    x162 += einsum("ijab,ikba->jk", t2, x161)
    del x161
    x163 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x163 += einsum("ij,abik->kjab", x162, l2) * 2
    del x162
    x164 += einsum("ijab->ijba", x163)
    del x163
    l2new += einsum("ijab->abij", x164) * -1
    l2new += einsum("ijab->baji", x164) * -1
    del x164
    x169 += einsum("ijkl->kilj", v.oooo)
    l2new += einsum("abij,klji->ablk", l2, x169)
    del x169
    x170 += einsum("ai->ia", f.vo)
    ls1new += einsum("ia,wai->w", x170, lu11) * 2
    del x170
    x172 = np.zeros((nbos, nbos), dtype=types[float])
    x172 += einsum("wai,xia->wx", lu11, u11)
    lu11new += einsum("wx,xia->wai", x172, g.bov) * 2
    del x172
    x173 += einsum("wai->wia", g.bvo)
    x173 += einsum("ia,wba->wib", t1, g.bvv)
    lu11new += einsum("wia,ijab->wbj", x173, x105)
    del x105
    del x173
    x175 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x175 += einsum("ia,wbi->wba", t1, lu11)
    lu11new += einsum("wab,iacb->wci", x175, x51)
    del x51
    del x175
    x176 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x176 += einsum("iajb->jiba", v.ovov) * 2
    x176 += einsum("iajb->jiab", v.ovov) * -1
    x177 += einsum("ijab,ijac->bc", t2, x176) * -1
    del x176
    x177 += einsum("ab->ab", f.vv)
    lu11new += einsum("ab,wai->wbi", x177, lu11)
    del x177
    x179 = np.zeros((nbos, nbos), dtype=types[float])
    x179 += einsum("wx->wx", w)
    x179 += einsum("wia,xia->wx", g.bov, u11) * 2
    lu11new += einsum("wx,xai->wai", x179, lu11)
    del x179
    l1new += einsum("w,wia->ai", ls1, gc.bov)
    l1new += einsum("ia->ai", f.ov)
    l2new += einsum("iajb->baji", v.ovov)
    l2new += einsum("abij,bcad->cdji", l2, v.vvvv)
    ls1new += einsum("ai,wai->w", l1, g.bvo) * 2
    ls1new += einsum("w,xw->x", ls1, w)
    ls1new += einsum("w->w", G)
    lu11new += einsum("ai,wab->wbi", l1, g.bvv)
    lu11new += einsum("wia->wai", g.bov)

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
    x0 += einsum("wai,wja->ij", lu11, u11)
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum("ij->ij", x0)
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=types[float])
    rdm1_f_oo -= einsum("ij->ij", x0) * 2
    del x0
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 += einsum("ai,ja->ij", l1, t1)
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
    x4 += einsum("ia,bajk->jkib", t1, l2)
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x5 += einsum("ijka->ijka", x4) * -1
    x5 += einsum("ijka->jika", x4) * 2
    del x4
    rdm1_f_vo += einsum("ijab,ijkb->ak", t2, x5) * -2
    del x5
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 -= einsum("ijab->jiab", t2)
    x6 += einsum("ijab->jiba", t2) * 2
    rdm1_f_vo += einsum("ai,ijab->bj", l1, x6) * 2
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum("ijab->jiab", t2)
    x7 += einsum("ijab->jiba", t2) * -0.5
    x8 += einsum("abij,ikba->jk", l2, x7) * 2
    del x7
    rdm1_f_vo += einsum("ia,ij->aj", t1, x8) * -2
    del x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum("abij->jiab", l2) * -0.5
    x9 += einsum("abij->jiba", l2)
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=types[float])
    rdm1_f_vv += einsum("ijab,ijcb->ac", t2, x9) * 4
    del x9
    rdm1_f_oo += einsum("ij->ji", delta_oo) * 2
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=types[float])
    rdm1_f_ov += einsum("ai->ia", l1) * 2
    rdm1_f_vo += einsum("ia->ai", t1) * 2
    rdm1_f_vo += einsum("w,wia->ai", ls1, u11) * 2
    rdm1_f_vv += einsum("ai,ib->ba", l1, t1) * 2
    rdm1_f_vv += einsum("wai,wib->ba", lu11, u11) * 2

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
    x0 = np.zeros((nocc, nocc), dtype=types[float])
    x0 += einsum("wai,wja->ij", lu11, u11)
    x15 = np.zeros((nocc, nocc), dtype=types[float])
    x15 += einsum("ij->ij", x0)
    x24 = np.zeros((nocc, nvir), dtype=types[float])
    x24 += einsum("ia,ij->ja", t1, x0)
    x27 = np.zeros((nocc, nvir), dtype=types[float])
    x27 += einsum("ia->ia", x24)
    del x24
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum("ij,kiab->kjab", x0, t2)
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum("ijab->ijab", x55)
    del x55
    rdm2_f_oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_oooo -= einsum("ij,kl->ijkl", delta_oo, x0) * 4
    rdm2_f_oooo += einsum("ij,kl->ilkj", delta_oo, x0) * 2
    rdm2_f_oooo += einsum("ij,kl->kijl", delta_oo, x0) * 2
    rdm2_f_oooo -= einsum("ij,kl->klji", delta_oo, x0) * 4
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum("ijab->jiab", t2) * -0.5
    x1 += einsum("ijab->jiba", t2)
    x2 = np.zeros((nocc, nocc), dtype=types[float])
    x2 += einsum("abij,ikab->kj", l2, x1)
    del x1
    x10 = np.zeros((nocc, nvir), dtype=types[float])
    x10 += einsum("ia,ji->ja", t1, x2) * 2
    x11 = np.zeros((nocc, nvir), dtype=types[float])
    x11 += einsum("ia->ia", x10)
    del x10
    x33 = np.zeros((nocc, nvir), dtype=types[float])
    x33 += einsum("ia,ji->ja", t1, x2)
    x34 = np.zeros((nocc, nvir), dtype=types[float])
    x34 += einsum("ia->ia", x33)
    del x33
    x77 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x77 += einsum("ij,jkab->ikab", x2, t2) * 8
    x78 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x78 += einsum("ijab->jiba", x77)
    del x77
    rdm2_f_oooo += einsum("ij,kl->jilk", delta_oo, x2) * -8
    rdm2_f_oooo += einsum("ij,kl->jkli", delta_oo, x2) * 4
    rdm2_f_oooo += einsum("ij,kl->ljik", delta_oo, x2) * 4
    rdm2_f_oooo += einsum("ij,kl->lkij", delta_oo, x2) * -8
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo += einsum("ia,jk->kiaj", t1, x2) * 4
    rdm2_f_oovo += einsum("ia,jk->kjai", t1, x2) * -8
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_vooo += einsum("ia,jk->aikj", t1, x2) * -8
    rdm2_f_vooo += einsum("ia,jk->ajki", t1, x2) * 4
    del x2
    x3 = np.zeros((nocc, nocc), dtype=types[float])
    x3 += einsum("ai,ja->ij", l1, t1)
    x15 += einsum("ij->ij", x3)
    x16 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x16 += einsum("ia,jk->jika", t1, x15)
    x83 = np.zeros((nocc, nvir), dtype=types[float])
    x83 += einsum("ia,ij->ja", t1, x15)
    del x15
    x84 = np.zeros((nocc, nvir), dtype=types[float])
    x84 += einsum("ia->ia", x83)
    x85 = np.zeros((nocc, nvir), dtype=types[float])
    x85 += einsum("ia->ia", x83)
    del x83
    x30 = np.zeros((nocc, nvir), dtype=types[float])
    x30 += einsum("ia,ij->ja", t1, x3)
    x31 = np.zeros((nocc, nvir), dtype=types[float])
    x31 -= einsum("ia->ia", x30)
    del x30
    x65 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x65 += einsum("ij,kiab->jkab", x3, t2)
    x68 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x68 += einsum("ijab->ijab", x65)
    del x65
    rdm2_f_oooo -= einsum("ij,kl->jikl", delta_oo, x3) * 4
    rdm2_f_oooo += einsum("ij,kl->kijl", delta_oo, x3) * 2
    rdm2_f_oooo += einsum("ij,kl->jlki", delta_oo, x3) * 2
    rdm2_f_oooo -= einsum("ij,kl->klji", delta_oo, x3) * 4
    del x3
    x4 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x4 += einsum("abij,klab->ijkl", l2, t2)
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x7 += einsum("ia,jikl->jkla", t1, x4)
    x12 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x12 += einsum("ijka->ijka", x7) * 4
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum("ia,ijkb->kjba", t1, x7)
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum("ijab->ijab", x79) * 2.0000000000000013
    del x79
    rdm2_f_vooo += einsum("ijka->ajik", x7) * -2
    rdm2_f_vooo += einsum("ijka->akij", x7) * 4
    del x7
    x80 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x80 += einsum("ijkl->jilk", x4)
    rdm2_f_oooo += einsum("ijkl->jkil", x4) * -2
    rdm2_f_oooo += einsum("ijkl->jlik", x4) * 4
    del x4
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x5 += einsum("ia,bajk->jkib", t1, l2)
    x6 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x6 += einsum("ia,jkla->kjli", t1, x5)
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x20 += einsum("ia,ijkl->jlka", t1, x6)
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 -= einsum("ijka->ijka", x20)
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x37 += einsum("ijka->ijka", x20)
    x71 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x71 += einsum("ia,ijkb->jkab", t1, x20)
    del x20
    x73 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum("ijab->ijab", x71)
    del x71
    x80 += einsum("ijkl->ijkl", x6)
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum("ijab,ijkl->klab", t2, x80) * 2
    del x80
    x82 += einsum("ijab->jiba", x81)
    del x81
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_vovo += einsum("ijab->ajbi", x82) * -1
    rdm2_f_vovo += einsum("ijab->bjai", x82) * 2
    del x82
    rdm2_f_oooo += einsum("ijkl->ikjl", x6) * 4
    rdm2_f_oooo -= einsum("ijkl->iljk", x6) * 2
    del x6
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x8 += einsum("ijka->ijka", x5) * -0.5
    x8 += einsum("ijka->jika", x5)
    x9 = np.zeros((nocc, nvir), dtype=types[float])
    x9 += einsum("ijab,ijkb->ka", t2, x8) * 2
    x11 += einsum("ia->ia", x9)
    del x9
    x12 += einsum("ij,ka->jika", delta_oo, x11) * -4
    del x11
    rdm2_f_oovo += einsum("ijka->ijak", x12)
    rdm2_f_oovo += einsum("ijka->ikaj", x12) * -0.5
    del x12
    x32 = np.zeros((nocc, nvir), dtype=types[float])
    x32 += einsum("ijab,ijkb->ka", t2, x8)
    del x8
    x34 += einsum("ia->ia", x32)
    del x32
    x78 += einsum("ia,jb->ijab", t1, x34) * 8.000000000000005
    rdm2_f_vooo += einsum("ij,ka->ajik", delta_oo, x34) * 4
    rdm2_f_vooo += einsum("ij,ka->akij", delta_oo, x34) * -8
    del x34
    x14 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x14 += einsum("ijab,kjla->klib", t2, x5)
    x16 -= einsum("ijka->ijka", x14)
    x66 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x66 -= einsum("ijka->ijka", x14)
    del x14
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x19 += einsum("ijab,jkla->klib", t2, x5)
    x28 -= einsum("ijka->ijka", x19)
    del x19
    x21 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x21 += einsum("ijka->ijka", x5) * 2
    x21 -= einsum("ijka->jika", x5)
    x22 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x22 += einsum("ijab,ikla->kljb", t2, x21)
    x28 += einsum("ijka->ijka", x22)
    del x22
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum("ia,ijkb->jkab", t1, x21)
    del x21
    rdm2_f_oovv = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_oovv -= einsum("ijab->ijab", x42) * 2
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vvoo -= einsum("ijab->abij", x42) * 2
    del x42
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x35 += einsum("ijab,kjlb->klia", t2, x5)
    x37 += einsum("ijka->ijka", x35)
    x58 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x58 += einsum("ijka->ijka", x35)
    del x35
    x49 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x49 -= einsum("ijka->ijka", x5)
    x49 += einsum("ijka->jika", x5) * 2
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum("ia,ijkb->jkab", t1, x49)
    del x49
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_ovvo -= einsum("ijab->ibaj", x50) * 2
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_voov -= einsum("ijab->ajib", x50) * 2
    del x50
    x90 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x90 += einsum("ia,jikb->jkba", t1, x5)
    x92 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x92 += einsum("ijab->ijab", x90)
    del x90
    x99 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x99 += einsum("ijab,jikc->kcba", t2, x5)
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv += einsum("iabc->bica", x99) * 2
    rdm2_f_vovv += einsum("iabc->ciba", x99) * -4
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo += einsum("iabc->baci", x99) * -4
    rdm2_f_vvvo += einsum("iabc->cabi", x99) * 2
    del x99
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov += einsum("ijka->ikja", x5) * 2
    rdm2_f_ooov -= einsum("ijka->jkia", x5) * 4
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_ovoo -= einsum("ijka->iajk", x5) * 4
    rdm2_f_ovoo += einsum("ijka->jaik", x5) * 2
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x13 += einsum("ai,jkba->ijkb", l1, t2)
    x16 += einsum("ijka->ijka", x13)
    rdm2_f_oovo += einsum("ijka->ijak", x16) * 2
    rdm2_f_oovo -= einsum("ijka->ikaj", x16) * 4
    rdm2_f_vooo -= einsum("ijka->ajik", x16) * 4
    rdm2_f_vooo += einsum("ijka->akij", x16) * 2
    del x16
    x66 += einsum("ijka->ijka", x13)
    del x13
    x67 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x67 += einsum("ia,ijkb->jkab", t1, x66)
    del x66
    x68 += einsum("ijab->ijab", x67)
    del x67
    rdm2_f_vovo += einsum("ijab->aibj", x68) * 2
    rdm2_f_vovo -= einsum("ijab->biaj", x68) * 4
    rdm2_f_vovo -= einsum("ijab->ajbi", x68) * 4
    rdm2_f_vovo += einsum("ijab->bjai", x68) * 2
    del x68
    x17 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x17 += einsum("ia,waj->wji", t1, lu11)
    x18 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x18 += einsum("wia,wjk->jkia", u11, x17)
    x28 += einsum("ijka->ijka", x18)
    x37 -= einsum("ijka->ijka", x18)
    del x18
    x23 = np.zeros((nocc, nvir), dtype=types[float])
    x23 += einsum("wia,wij->ja", u11, x17)
    x27 += einsum("ia->ia", x23)
    x84 += einsum("ia->ia", x23)
    x85 += einsum("ia->ia", x23)
    del x23
    x60 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x60 += einsum("ia,wij->wja", t1, x17)
    del x17
    x62 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x62 += einsum("wia->wia", x60)
    del x60
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum("ijab->jiab", t2) * 2
    x25 -= einsum("ijab->jiba", t2)
    x26 = np.zeros((nocc, nvir), dtype=types[float])
    x26 += einsum("ai,ijba->jb", l1, x25)
    x27 -= einsum("ia->ia", x26)
    x28 += einsum("ij,ka->jika", delta_oo, x27)
    rdm2_f_oovo -= einsum("ijka->ijak", x28) * 4
    rdm2_f_oovo += einsum("ijka->ikaj", x28) * 2
    del x28
    rdm2_f_vooo += einsum("ij,ka->ajik", delta_oo, x27) * 2
    rdm2_f_vooo -= einsum("ij,ka->akij", delta_oo, x27) * 4
    del x27
    x64 -= einsum("ia,jb->ijab", t1, x26)
    del x26
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x36 += einsum("ijab,iklb->jkla", x25, x5)
    del x5
    x37 -= einsum("ijka->jkia", x36)
    rdm2_f_vooo -= einsum("ijka->ajik", x37) * 2
    rdm2_f_vooo += einsum("ijka->akij", x37) * 4
    del x37
    x58 -= einsum("ijka->jkia", x36)
    del x36
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum("ia,ijkb->jkab", t1, x58)
    del x58
    x64 -= einsum("ijab->ijab", x59)
    del x59
    x61 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x61 += einsum("wai,ijba->wjb", lu11, x25)
    x62 -= einsum("wia->wia", x61)
    del x61
    x63 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x63 += einsum("wia,wjb->jiba", u11, x62)
    del x62
    x64 += einsum("ijab->ijab", x63)
    del x63
    rdm2_f_vvoo -= einsum("abij,ikca->cbjk", l2, x25) * 2
    x29 = np.zeros((nocc, nvir), dtype=types[float])
    x29 += einsum("w,wia->ia", ls1, u11)
    x31 += einsum("ia->ia", x29)
    x84 -= einsum("ia->ia", x29)
    x85 -= einsum("ia->ia", x29)
    del x29
    rdm2_f_vovo -= einsum("ia,jb->bjai", t1, x85) * 4
    rdm2_f_vovo += einsum("ia,jb->ajbi", t1, x85) * 2
    del x85
    x31 += einsum("ia->ia", t1)
    rdm2_f_oovo += einsum("ij,ka->jiak", delta_oo, x31) * 4
    rdm2_f_oovo -= einsum("ij,ka->jkai", delta_oo, x31) * 2
    rdm2_f_vooo -= einsum("ij,ka->aijk", delta_oo, x31) * 2
    rdm2_f_vooo += einsum("ij,ka->akji", delta_oo, x31) * 4
    del x31
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x38 += einsum("wai,wjb->ijab", lu11, u11)
    rdm2_f_oovv -= einsum("ijab->ijba", x38) * 2
    rdm2_f_ovvo += einsum("ijab->iabj", x38) * 4
    rdm2_f_voov += einsum("ijab->bjia", x38) * 4
    rdm2_f_vvoo -= einsum("ijab->baij", x38) * 2
    del x38
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum("abij->jiab", l2) * 2
    x39 -= einsum("abij->jiba", l2)
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum("ijab,ikca->kjcb", t2, x39)
    x72 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x72 += einsum("ijab,ikac->kjcb", t2, x40)
    x73 += einsum("ijab->ijab", x72) * 2
    del x72
    rdm2_f_oovv -= einsum("ijab->ijba", x40) * 2
    del x40
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum("ijab,ikcb->jkac", x25, x39)
    del x25
    rdm2_f_ovvo += einsum("ijab->jbai", x48) * 2
    rdm2_f_voov += einsum("ijab->aijb", x48) * 2
    del x48
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum("ijab,ikcb->kjca", t2, x39)
    del x39
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum("ijab,ikac->kjcb", t2, x56)
    del x56
    x64 += einsum("ijab->jiba", x57)
    del x57
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x41 -= einsum("abij->jiab", l2)
    x41 += einsum("abij->jiba", l2) * 2
    rdm2_f_oovv -= einsum("ijab,ikcb->kjac", t2, x41) * 2
    del x41
    x43 = np.zeros((nvir, nvir), dtype=types[float])
    x43 += einsum("ai,ib->ab", l1, t1)
    x47 = np.zeros((nvir, nvir), dtype=types[float])
    x47 += einsum("ab->ab", x43) * 0.5
    x52 = np.zeros((nvir, nvir), dtype=types[float])
    x52 += einsum("ab->ab", x43)
    x97 = np.zeros((nvir, nvir), dtype=types[float])
    x97 += einsum("ab->ab", x43)
    del x43
    x44 = np.zeros((nvir, nvir), dtype=types[float])
    x44 += einsum("wai,wib->ab", lu11, u11)
    x47 += einsum("ab->ab", x44) * 0.5
    x52 += einsum("ab->ab", x44)
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum("ab,ijac->jicb", x44, t2)
    x64 += einsum("ijab->ijab", x54)
    del x54
    rdm2_f_vovo -= einsum("ijab->aibj", x64) * 4
    rdm2_f_vovo += einsum("ijab->biaj", x64) * 2
    rdm2_f_vovo += einsum("ijab->ajbi", x64) * 2
    rdm2_f_vovo -= einsum("ijab->bjai", x64) * 4
    del x64
    x97 += einsum("ab->ab", x44)
    del x44
    x98 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x98 += einsum("ia,bc->ibac", t1, x97)
    del x97
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum("abij->jiab", l2)
    x45 += einsum("abij->jiba", l2) * -0.5
    x46 = np.zeros((nvir, nvir), dtype=types[float])
    x46 += einsum("ijab,ijbc->ca", t2, x45)
    x47 += einsum("ab->ab", x46)
    rdm2_f_oovv += einsum("ij,ab->jiba", delta_oo, x47) * 8
    del x47
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum("ab,ijac->ijbc", x46, t2) * 8
    x78 += einsum("ijab->jiba", x76)
    del x76
    rdm2_f_vovo += einsum("ijab->aibj", x78) * -1
    rdm2_f_vovo += einsum("ijab->biaj", x78) * 0.5
    rdm2_f_vovo += einsum("ijab->ajbi", x78) * 0.5
    rdm2_f_vovo += einsum("ijab->bjai", x78) * -1
    del x78
    rdm2_f_vovv += einsum("ia,bc->aicb", t1, x46) * 8
    rdm2_f_vovv += einsum("ia,bc->ciab", t1, x46) * -4
    rdm2_f_vvvo += einsum("ia,bc->abci", t1, x46) * -4
    rdm2_f_vvvo += einsum("ia,bc->cbai", t1, x46) * 8
    del x46
    x51 = np.zeros((nvir, nvir), dtype=types[float])
    x51 += einsum("ijab,ijbc->ca", t2, x45) * 2
    del x45
    x52 += einsum("ab->ab", x51)
    del x51
    rdm2_f_ovvo += einsum("ij,ab->jabi", delta_oo, x52) * -2
    rdm2_f_voov += einsum("ij,ab->bija", delta_oo, x52) * -2
    rdm2_f_vvoo += einsum("ij,ab->baji", delta_oo, x52) * 4
    del x52
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x53 -= einsum("ijab->jiab", t2)
    x53 += einsum("ijab->jiba", t2) * 2
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x91 += einsum("abij,ikac->kjcb", l2, x53)
    x92 -= einsum("ijab->jiba", x91)
    del x91
    rdm2_f_vvoo -= einsum("abij,ikcb->cajk", l2, x53) * 2
    del x53
    x69 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x69 += einsum("abij,kjbc->ikac", l2, t2)
    x70 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x70 += einsum("ijab,jkac->ikbc", t2, x69)
    del x69
    x73 += einsum("ijab->ijab", x70)
    del x70
    x73 += einsum("ijab->jiba", t2)
    rdm2_f_vovo -= einsum("ijab->biaj", x73) * 2
    rdm2_f_vovo += einsum("ijab->aibj", x73) * 4
    del x73
    x74 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x74 += einsum("abij,kjac->ikbc", l2, t2)
    x75 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x75 += einsum("ijab,jkac->ikbc", t2, x74)
    rdm2_f_vovo += einsum("ijab->ajbi", x75) * 4
    rdm2_f_vovo -= einsum("ijab->bjai", x75) * 2
    del x75
    x96 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x96 += einsum("ia,ijbc->jbac", t1, x74)
    del x74
    x98 -= einsum("iabc->iabc", x96)
    del x96
    x84 -= einsum("ia->ia", t1)
    rdm2_f_vovo -= einsum("ia,jb->aibj", t1, x84) * 4
    rdm2_f_vovo += einsum("ia,jb->biaj", t1, x84) * 2
    del x84
    x86 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x86 += einsum("ia,bcji->jbca", t1, l2)
    x101 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x101 += einsum("ia,ibcd->cbda", t1, x86)
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv -= einsum("abcd->cbda", x101) * 2
    rdm2_f_vvvv += einsum("abcd->dbca", x101) * 4
    del x101
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv += einsum("iabc->iacb", x86) * 4
    rdm2_f_ovvv -= einsum("iabc->ibca", x86) * 2
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov -= einsum("iabc->caib", x86) * 2
    rdm2_f_vvov += einsum("iabc->cbia", x86) * 4
    del x86
    x87 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x87 += einsum("ia,wbi->wba", t1, lu11)
    x88 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x88 += einsum("wia,wbc->ibca", u11, x87)
    del x87
    x94 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x94 -= einsum("iabc->iabc", x88)
    del x88
    x89 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x89 += einsum("abij,kjca->ikbc", l2, t2)
    x92 += einsum("ijab->ijab", x89)
    del x89
    x93 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x93 += einsum("ia,ijbc->jabc", t1, x92)
    del x92
    x94 += einsum("iabc->ibac", x93)
    del x93
    rdm2_f_vovv += einsum("iabc->bica", x94) * 2
    rdm2_f_vovv -= einsum("iabc->ciba", x94) * 4
    rdm2_f_vvvo -= einsum("iabc->baci", x94) * 4
    rdm2_f_vvvo += einsum("iabc->cabi", x94) * 2
    del x94
    x95 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x95 += einsum("ai,jibc->jabc", l1, t2)
    x98 += einsum("iabc->iabc", x95)
    del x95
    rdm2_f_vovv += einsum("iabc->bica", x98) * 4
    rdm2_f_vovv -= einsum("iabc->ciba", x98) * 2
    rdm2_f_vvvo -= einsum("iabc->baci", x98) * 2
    rdm2_f_vvvo += einsum("iabc->cabi", x98) * 4
    del x98
    x100 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x100 += einsum("abij,ijcd->abcd", l2, t2)
    rdm2_f_vvvv += einsum("abcd->cbda", x100) * -2
    rdm2_f_vvvv += einsum("abcd->dbca", x100) * 4
    del x100
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
    dm_b_des += einsum("w->w", s1)
    dm_b_des += einsum("ai,wia->w", l1, u11) * 2

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
    rdm1_b += einsum("w,x->wx", ls1, s1)
    rdm1_b += einsum("wai,xia->wx", lu11, u11) * 2

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
    x22 = np.zeros((nocc, nvir), dtype=types[float])
    x22 += einsum("wia,wij->ja", u11, x0)
    rdm_eb_cre_oo = np.zeros((nbos, nocc, nocc), dtype=types[float])
    rdm_eb_cre_oo -= einsum("wij->wji", x0) * 2
    rdm_eb_cre_ov = np.zeros((nbos, nocc, nvir), dtype=types[float])
    rdm_eb_cre_ov -= einsum("ia,wij->wja", t1, x0) * 2
    del x0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 -= einsum("ijab->jiab", t2)
    x1 += einsum("ijab->jiba", t2) * 2
    rdm_eb_cre_ov += einsum("wai,ijab->wjb", lu11, x1) * 2
    x2 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x2 += einsum("ai,wja->wij", l1, u11)
    x17 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x17 += einsum("wij->wij", x2)
    rdm_eb_des_oo = np.zeros((nbos, nocc, nocc), dtype=types[float])
    rdm_eb_des_oo -= einsum("wij->wji", x2) * 2
    del x2
    x3 = np.zeros((nbos), dtype=types[float])
    x3 += einsum("ai,wia->w", l1, u11)
    rdm_eb_des_oo += einsum("w,ij->wji", x3, delta_oo) * 4
    rdm_eb_des_ov = np.zeros((nbos, nocc, nvir), dtype=types[float])
    rdm_eb_des_ov += einsum("w,ia->wia", x3, t1) * 4
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 -= einsum("abij->jiab", l2)
    x4 += einsum("abij->jiba", l2) * 2
    x5 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x5 += einsum("wia,ijab->wjb", u11, x4)
    del x4
    x6 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x6 += einsum("ia,wja->wij", t1, x5)
    x17 += einsum("wij->wji", x6)
    rdm_eb_des_ov -= einsum("ia,wij->wja", t1, x17) * 2
    del x17
    rdm_eb_des_oo -= einsum("wij->wij", x6) * 2
    del x6
    rdm_eb_des_ov += einsum("wia,ijab->wjb", x5, x1) * 2
    del x1
    rdm_eb_des_vo = np.zeros((nbos, nvir, nocc), dtype=types[float])
    rdm_eb_des_vo += einsum("wia->wai", x5) * 2
    rdm_eb_des_vv = np.zeros((nbos, nvir, nvir), dtype=types[float])
    rdm_eb_des_vv += einsum("ia,wib->wba", t1, x5) * 2
    del x5
    x7 = np.zeros((nocc, nocc), dtype=types[float])
    x7 += einsum("ai,ja->ij", l1, t1)
    x11 = np.zeros((nocc, nocc), dtype=types[float])
    x11 += einsum("ij->ij", x7)
    x16 = np.zeros((nocc, nocc), dtype=types[float])
    x16 += einsum("ij->ij", x7)
    x21 = np.zeros((nocc, nocc), dtype=types[float])
    x21 += einsum("ij->ij", x7) * 0.49999999999999967
    del x7
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum("wai,wja->ij", lu11, u11)
    x11 += einsum("ij->ij", x8)
    x16 += einsum("ij->ij", x8)
    x21 += einsum("ij->ij", x8) * 0.49999999999999967
    del x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum("ijab->jiab", t2)
    x9 += einsum("ijab->jiba", t2) * -0.5
    x10 = np.zeros((nocc, nocc), dtype=types[float])
    x10 += einsum("abij,ikba->jk", l2, x9) * 2
    x11 += einsum("ij->ij", x10)
    x16 += einsum("ij->ij", x10)
    del x10
    rdm_eb_des_ov += einsum("ij,wia->wja", x16, u11) * -2
    del x16
    x21 += einsum("abij,ikba->jk", l2, x9)
    del x9
    x22 += einsum("ia,ij->ja", t1, x21) * 2.0000000000000013
    del x21
    x11 += einsum("ij->ji", delta_oo) * -1
    rdm_eb_des_oo += einsum("w,ij->wji", s1, x11) * -2
    del x11
    x12 = np.zeros((nbos, nbos), dtype=types[float])
    x12 += einsum("wai,xia->wx", lu11, u11)
    rdm_eb_des_ov += einsum("wx,wia->xia", x12, u11) * 4
    del x12
    x13 = np.zeros((nvir, nvir), dtype=types[float])
    x13 += einsum("wai,wib->ab", lu11, u11)
    x15 = np.zeros((nvir, nvir), dtype=types[float])
    x15 += einsum("ab->ab", x13)
    x23 = np.zeros((nvir, nvir), dtype=types[float])
    x23 += einsum("ab->ab", x13) * 0.5
    del x13
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x14 += einsum("abij->jiab", l2)
    x14 += einsum("abij->jiba", l2) * -0.5
    x15 += einsum("ijab,ijbc->ca", t2, x14) * 2
    rdm_eb_des_ov += einsum("ab,wia->wib", x15, u11) * -2
    del x15
    x23 += einsum("ijab,ijbc->ca", t2, x14)
    del x14
    x18 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x18 += einsum("ia,abjk->kjib", t1, l2)
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x19 += einsum("ijka->ijka", x18) * -0.5
    x19 += einsum("ijka->jika", x18)
    del x18
    x22 += einsum("ijab,ijkb->ka", t2, x19) * 2.0000000000000013
    del x19
    x20 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x20 += einsum("ijab->jiab", t2) * -1
    x20 += einsum("ijab->jiba", t2) * 2
    x22 += einsum("ai,ijab->jb", l1, x20) * -1
    del x20
    x22 += einsum("ia->ia", t1) * -1
    x22 += einsum("w,wia->ia", ls1, u11) * -1
    rdm_eb_des_ov += einsum("w,ia->wia", s1, x22) * -2
    del x22
    x23 += einsum("ai,ib->ab", l1, t1) * 0.5
    rdm_eb_des_vv += einsum("w,ab->wab", s1, x23) * 4
    del x23
    rdm_eb_cre_oo += einsum("w,ij->wji", ls1, delta_oo) * 2
    rdm_eb_cre_ov += einsum("w,ia->wia", ls1, t1) * 2
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

