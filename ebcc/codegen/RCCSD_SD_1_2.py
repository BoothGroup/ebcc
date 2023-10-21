# Code generated for ebcc.

from ebcc import numpy as np
from ebcc.util import pack_2e, einsum, Namespace
from ebcc.precision import types

def energy(f=None, v=None, w=None, g=None, G=None, nocc=None, nvir=None, nbos=None, t1=None, t2=None, s1=None, s2=None, u11=None, u12=None, **kwargs):
    # Energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x0 += einsum("iajb->jiab", v.ovov)
    x0 += einsum("iajb->jiba", v.ovov) * -0.5
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x1 += einsum("ijab->jiba", t2)
    x1 += einsum("ia,jb->ijab", t1, t1)
    e_cc = 0
    e_cc += einsum("ijab,ijba->", x0, x1) * 2
    del x0
    del x1
    x2 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x2 += einsum("wia->wia", u11)
    x2 += einsum("w,ia->wia", s1, t1)
    e_cc += einsum("wia,wia->", g.bov, x2) * 2
    del x2
    e_cc += einsum("ia,ia->", f.ov, t1) * 2
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
    x0 += einsum("ia,jbka->ijkb", t1, v.ovov)
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x1 += einsum("ijka->ijka", x0) * 2
    x1 += einsum("ijka->ikja", x0) * -1
    x28 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x28 += einsum("ia,jkla->jilk", t1, x0)
    x88 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x88 += einsum("ijkl->lkji", x28)
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    t2new += einsum("ijab,klij->lkba", t2, x28)
    del x28
    x35 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x35 += einsum("ijab,kjla->kilb", t2, x0)
    x41 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x41 -= einsum("ijka->ikja", x35)
    del x35
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x36 += einsum("ijka->ijka", x0) * 2
    x36 -= einsum("ijka->ikja", x0)
    x37 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x37 += einsum("ijab,kila->jklb", t2, x36)
    del x36
    x41 += einsum("ijka->jkia", x37)
    del x37
    x65 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x65 += einsum("ijab,klja->kilb", t2, x0)
    x69 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x69 -= einsum("ijka->kija", x65)
    del x65
    x136 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x136 -= einsum("ijka->jkia", x0)
    x136 += einsum("ijka->kjia", x0) * 2
    del x0
    x1 += einsum("ijka->jika", v.ooov) * -1
    x1 += einsum("ijka->jkia", v.ooov) * 2
    t1new = np.zeros((nocc, nvir), dtype=types[float])
    t1new += einsum("ijab,kjib->ka", t2, x1) * -1
    del x1
    x2 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x2 += einsum("iabc->ibac", v.ovvv)
    x2 += einsum("iabc->ibca", v.ovvv) * -0.5
    x102 = np.zeros((nvir, nvir), dtype=types[float])
    x102 += einsum("ia,ibac->bc", t1, x2) * 2
    x104 = np.zeros((nvir, nvir), dtype=types[float])
    x104 += einsum("ab->ab", x102)
    x135 = np.zeros((nvir, nvir), dtype=types[float])
    x135 += einsum("ab->ab", x102)
    del x102
    t1new += einsum("ijab,icab->jc", t2, x2) * 2
    del x2
    x3 = np.zeros((nocc, nvir), dtype=types[float])
    x3 += einsum("w,wia->ia", s1, g.bov)
    x6 = np.zeros((nocc, nvir), dtype=types[float])
    x6 += einsum("ia->ia", x3)
    x18 = np.zeros((nocc, nvir), dtype=types[float])
    x18 += einsum("ia->ia", x3) * 0.5
    x108 = np.zeros((nocc, nvir), dtype=types[float])
    x108 += einsum("ia->ia", x3)
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
    x99 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x99 += einsum("wia->wia", x97)
    x117 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x117 += einsum("ia,wib->wab", t1, x97)
    x118 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x118 += einsum("wab->wba", x117)
    del x117
    x125 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x125 += einsum("ia,wja->wij", t1, x97)
    x126 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x126 += einsum("wij->wji", x125)
    del x125
    s2new = np.zeros((nbos, nbos), dtype=types[float])
    s2new += einsum("wia,xia->wx", u11, x97) * 2
    del x97
    x133 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x133 += einsum("wxia,ijba->xwjb", u12, x4)
    del x4
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
    x120 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x120 += einsum("ia,wja->wji", x6, u11)
    x123 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x123 += einsum("wij->wji", x120)
    del x120
    x137 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x137 += einsum("ia,wxja->xwij", x6, u12)
    s1new = np.zeros((nbos), dtype=types[float])
    s1new += einsum("ia,wia->w", x6, u11) * 2
    s2new += einsum("ia,wxia->xw", x6, u12) * 2
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 += einsum("ijab->jiab", t2) * 2
    x7 -= einsum("ijab->jiba", t2)
    x47 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x47 += einsum("wia,ijba->wjb", g.bov, x7)
    x49 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x49 += einsum("wia->wia", x47)
    del x47
    t1new += einsum("ia,ijba->jb", x6, x7)
    del x6
    u12new = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    u12new += einsum("wxia,ijba->xwjb", x133, x7)
    del x133
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
    x48 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x48 += einsum("ia,wij->wja", t1, x10)
    x49 -= einsum("wia->wia", x48)
    del x48
    t1new -= einsum("wia,wij->ja", u11, x10)
    u11new -= einsum("wij,wxia->xja", x10, u12)
    del x10
    x11 = np.zeros((nocc, nocc), dtype=types[float])
    x11 += einsum("w,wij->ij", s1, g.boo)
    x20 = np.zeros((nocc, nocc), dtype=types[float])
    x20 += einsum("ij->ij", x11)
    x72 += einsum("ij->ji", x11)
    del x11
    x12 = np.zeros((nocc, nocc), dtype=types[float])
    x12 += einsum("wia,wja->ij", g.bov, u11)
    x20 += einsum("ij->ij", x12)
    x53 = np.zeros((nocc, nocc), dtype=types[float])
    x53 += einsum("ij->ij", x12)
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
    x79 = np.zeros((nvir, nvir), dtype=types[float])
    x79 += einsum("ijab,ijbc->ac", t2, x13)
    x80 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x80 += einsum("ab,ijbc->ijca", x79, t2) * 2
    del x79
    x83 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x83 += einsum("ijab->jiab", x80)
    del x80
    x81 = np.zeros((nocc, nocc), dtype=types[float])
    x81 += einsum("ijab,ikba->jk", t2, x13)
    x82 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x82 += einsum("ij,jkab->kiab", x81, t2) * 2
    del x81
    x83 += einsum("ijab->ijba", x82)
    del x82
    x103 = np.zeros((nocc, nvir), dtype=types[float])
    x103 += einsum("ia,ijba->jb", t1, x13) * 2
    del x13
    x104 += einsum("ia,ib->ab", t1, x103) * -1
    del x103
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
    u12new += einsum("ij,wxia->xwja", x20, u12) * -1
    del x20
    x21 = np.zeros((nvir, nvir), dtype=types[float])
    x21 += einsum("w,wab->ab", s1, g.bvv)
    x24 = np.zeros((nvir, nvir), dtype=types[float])
    x24 += einsum("ab->ab", x21)
    x74 = np.zeros((nvir, nvir), dtype=types[float])
    x74 += einsum("ab->ab", x21)
    x104 += einsum("ab->ab", x21)
    x135 += einsum("ab->ab", x21)
    del x21
    x22 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x22 -= einsum("iabc->ibac", v.ovvv)
    x22 += einsum("iabc->ibca", v.ovvv) * 2
    x23 = np.zeros((nvir, nvir), dtype=types[float])
    x23 += einsum("ia,ibca->bc", t1, x22)
    x24 += einsum("ab->ab", x23)
    x44 = np.zeros((nvir, nvir), dtype=types[float])
    x44 -= einsum("ab->ab", x23)
    del x23
    x111 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x111 += einsum("wia,ibca->wbc", u11, x22)
    del x22
    x112 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x112 += einsum("wab->wab", x111)
    x118 -= einsum("wab->wba", x111)
    del x111
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
    u11new += einsum("w,wxia->xia", x26, u12)
    del x26
    x27 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x27 += einsum("ia,bacd->icbd", t1, v.vvvv)
    t2new += einsum("ia,jbca->ijbc", t1, x27)
    del x27
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x29 += einsum("ia,jabc->ijbc", t1, v.ovvv)
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x30 += einsum("ijab,kjca->kibc", t2, x29)
    del x29
    x55 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x55 += einsum("ijab->ijab", x30)
    del x30
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x31 -= einsum("iajb->jiab", v.ovov)
    x31 += einsum("iajb->jiba", v.ovov) * 2
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x32 += einsum("ijab,ikbc->jkac", t2, x31)
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum("ijab,kica->jkbc", t2, x32)
    del x32
    x55 += einsum("ijab->ijab", x33)
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
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum("ijab,kjca->kibc", t2, x38)
    del x38
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum("ijab->ijab", x57)
    del x57
    x39 += einsum("iabj->ijba", v.ovvo)
    x40 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x40 += einsum("ia,jkba->ijkb", t1, x39)
    del x39
    x41 += einsum("ijka->ijka", x40)
    del x40
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum("ia,jikb->jkab", t1, x41)
    del x41
    x55 += einsum("ijab->ijab", x42)
    del x42
    x43 = np.zeros((nvir, nvir), dtype=types[float])
    x43 += einsum("wia,wib->ab", g.bov, u11)
    x44 += einsum("ab->ba", x43)
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum("ab,ijbc->ijca", x44, t2)
    del x44
    x55 += einsum("ijab->jiab", x45)
    del x45
    x104 += einsum("ab->ba", x43) * -1
    x135 += einsum("ab->ba", x43) * -1
    del x43
    x46 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x46 += einsum("ia,wba->wib", t1, g.bvv)
    x49 += einsum("wia->wia", x46)
    del x46
    x49 += einsum("wai->wia", g.bvo)
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 += einsum("wia,wjb->ijab", u11, x49)
    del x49
    x55 -= einsum("ijab->jiba", x50)
    del x50
    x51 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x51 += einsum("ijka->ikja", v.ooov) * 2
    x51 -= einsum("ijka->kija", v.ooov)
    x52 = np.zeros((nocc, nocc), dtype=types[float])
    x52 += einsum("ia,jika->jk", t1, x51)
    x53 += einsum("ij->ij", x52)
    del x52
    x54 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x54 += einsum("ij,ikab->kjab", x53, t2)
    del x53
    x55 += einsum("ijab->ijba", x54)
    del x54
    t2new -= einsum("ijab->ijab", x55)
    t2new -= einsum("ijab->jiba", x55)
    del x55
    x107 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x107 += einsum("wia,jika->wjk", u11, x51)
    del x51
    x109 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x109 += einsum("wij->wij", x107)
    x126 += einsum("wij->wij", x107)
    del x107
    x56 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x56 += einsum("ia,bjca->ijbc", t1, v.vovv)
    x76 -= einsum("ijab->ijab", x56)
    del x56
    x58 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x58 += einsum("iabc->ibac", v.ovvv) * 2
    x58 -= einsum("iabc->ibca", v.ovvv)
    x59 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x59 += einsum("ia,jbca->ijbc", t1, x58)
    del x58
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum("ijab,kica->jkbc", t2, x59)
    x76 -= einsum("ijab->jiab", x60)
    del x60
    x134 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x134 += einsum("ijab->jiab", x59)
    del x59
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
    x84 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x84 += einsum("ijab,kjil->klba", t2, x63)
    del x63
    t2new += einsum("ijab->ijba", x84)
    t2new += einsum("ijab->jiab", x84)
    del x84
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
    x77 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x77 += einsum("ijab,kacb->ijkc", t2, v.ovvv)
    x78 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x78 += einsum("ia,jkib->jkab", t1, x77)
    del x77
    x83 += einsum("ijab->ijab", x78)
    del x78
    t2new += einsum("ijab->ijab", x83) * -1
    t2new += einsum("ijab->jiba", x83) * -1
    del x83
    x85 += einsum("iabj->jiba", v.ovvo) * 2
    x85 -= einsum("ijab->jiab", v.oovv)
    t2new += einsum("ijab,kica->kjcb", t2, x85)
    del x85
    x86 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x86 += einsum("ijab,kbla->ijlk", t2, v.ovov)
    x87 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x87 += einsum("ijkl->lkji", x86)
    x88 += einsum("ijkl->lkji", x86)
    del x86
    x87 += einsum("ijkl->kilj", v.oooo)
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
    x115 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x115 += einsum("wx,ywia->xyia", x93, u12)
    del x93
    x132 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x132 -= einsum("wxia->wxia", x115) * 2
    del x115
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
    x121 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x121 += einsum("wia->wia", x98)
    del x98
    x99 += einsum("wia->wia", gc.bov)
    x109 += einsum("ia,wja->wji", t1, x99)
    u11new += einsum("wia,ijba->wjb", x99, x7)
    del x99
    del x7
    x100 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x100 += einsum("iajb->jiab", v.ovov) * -1
    x100 += einsum("iajb->jiba", v.ovov) * 2
    x101 = np.zeros((nvir, nvir), dtype=types[float])
    x101 += einsum("ijab,ijcb->ac", t2, x100)
    del x100
    x104 += einsum("ab->ab", x101) * -1
    x135 += einsum("ab->ab", x101) * -1
    del x101
    x104 += einsum("ab->ab", f.vv)
    u11new += einsum("ab,wib->wia", x104, u11)
    del x104
    x105 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x105 += einsum("wx,xij->wij", s2, g.boo)
    x109 += einsum("wij->wij", x105)
    x123 += einsum("wij->wij", x105)
    del x105
    x106 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x106 += einsum("wia,xwja->xij", g.bov, u12)
    x109 += einsum("wij->wij", x106)
    x126 += einsum("wij->wij", x106)
    del x106
    x127 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x127 += einsum("wia,xij->wxja", u11, x126)
    del x126
    x132 += einsum("wxia->wxia", x127)
    del x127
    x108 += einsum("ia->ia", f.ov)
    x109 += einsum("ia,wja->wij", x108, u11)
    del x108
    x109 += einsum("wij->wij", gc.boo)
    u11new -= einsum("ia,wij->wja", t1, x109)
    del x109
    x110 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x110 += einsum("wx,xab->wab", s2, g.bvv)
    x112 += einsum("wab->wab", x110)
    x128 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x128 += einsum("wab->wab", x110)
    del x110
    x112 += einsum("wab->wab", gc.bvv)
    u11new += einsum("ia,wba->wib", t1, x112)
    del x112
    x113 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x113 += einsum("wia,xyia->wxy", g.bov, u12)
    u12new += einsum("wia,wxy->yxia", u11, x113) * 2
    del x113
    x114 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x114 += einsum("wx,ywia->yxia", w, u12)
    x132 -= einsum("wxia->wxia", x114)
    del x114
    x116 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x116 += einsum("wia,xwib->xab", g.bov, u12)
    x118 += einsum("wab->wab", x116)
    del x116
    x119 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x119 += einsum("wia,xab->wxib", u11, x118)
    del x118
    x132 += einsum("wxia->wxia", x119)
    del x119
    x121 += einsum("wia->wia", gc.bov)
    x122 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x122 += einsum("ia,wja->wij", t1, x121)
    x123 += einsum("wij->wji", x122)
    del x122
    x130 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x130 += einsum("wia,xja->wxij", u11, x121)
    del x121
    x131 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x131 += einsum("ia,wxji->xwja", t1, x130)
    del x130
    x132 += einsum("wxia->wxia", x131)
    del x131
    x123 += einsum("wij->wij", gc.boo)
    x124 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x124 += einsum("wia,xij->wxja", u11, x123)
    del x123
    x132 += einsum("wxia->xwia", x124)
    del x124
    x128 += einsum("wab->wab", gc.bvv)
    x129 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x129 += einsum("wia,xba->wxib", u11, x128)
    del x128
    x132 -= einsum("wxia->xwia", x129)
    del x129
    u12new -= einsum("wxia->wxia", x132)
    u12new -= einsum("wxia->xwia", x132)
    del x132
    x134 += einsum("iabj->ijba", v.ovvo) * 2
    x134 -= einsum("ijab->ijab", v.oovv)
    u12new += einsum("wxia,ijba->xwjb", u12, x134)
    del x134
    x135 += einsum("ab->ab", f.vv)
    u12new += einsum("ab,wxib->xwia", x135, u12)
    del x135
    x136 += einsum("ijka->ikja", v.ooov) * 2
    x136 -= einsum("ijka->kija", v.ooov)
    x137 += einsum("wxia,jika->xwjk", u12, x136)
    del x136
    u12new -= einsum("ia,wxij->xwja", t1, x137)
    del x137
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
    s1new += einsum("wia,xwia->x", g.bov, u12) * 2
    s1new += einsum("w,wx->x", s1, w)
    s1new += einsum("w->w", G)
    s1new += einsum("ia,wia->w", t1, gc.bov) * 2
    u11new += einsum("wab,xwib->xia", g.bvv, u12)
    u11new += einsum("wx,xai->wia", s2, g.bvo)
    u11new += einsum("wai->wia", gc.bvo)

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
    x0 = np.zeros((nocc, nvir), dtype=types[float])
    x0 += einsum("w,wia->ia", s1, g.bov)
    x1 = np.zeros((nocc, nvir), dtype=types[float])
    x1 += einsum("ia->ia", x0)
    x52 = np.zeros((nocc, nvir), dtype=types[float])
    x52 += einsum("ia->ia", x0) * 0.5
    x64 = np.zeros((nocc, nvir), dtype=types[float])
    x64 += einsum("ia->ia", x0)
    x112 = np.zeros((nocc, nvir), dtype=types[float])
    x112 += einsum("ia->ia", x0)
    x163 = np.zeros((nocc, nvir), dtype=types[float])
    x163 += einsum("ia->ia", x0)
    del x0
    x1 += einsum("ia->ia", f.ov)
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x2 += einsum("ia,jkab->jkib", x1, t2) * 0.5
    del x1
    x20 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x20 += einsum("ijka->ikja", x2)
    x20 += einsum("ijka->jkia", x2) * -2
    del x2
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x3 += einsum("ijab,kbca->jikc", t2, v.ovvv)
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum("ijka->ijka", x3) * -1
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 += einsum("ia,jbca->ijcb", t1, v.ovvv)
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x5 += einsum("ia,jkba->ijkb", t1, x4)
    x11 += einsum("ijka->ijka", x5) * -1
    del x5
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x37 += einsum("ijab->jiab", x4)
    x178 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x178 += einsum("ijab->ijab", x4)
    del x4
    x6 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x6 += einsum("ijab,kalb->ijkl", t2, v.ovov)
    x9 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x9 += einsum("ijkl->jilk", x6)
    x192 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x192 += einsum("ijkl->lkji", x6)
    del x6
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x7 += einsum("ia,jbka->ijkb", t1, v.ovov)
    x8 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x8 += einsum("ia,jkla->ijkl", t1, x7)
    x9 += einsum("ijkl->ijkl", x8)
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x10 += einsum("ia,jkil->jkla", t1, x9)
    del x9
    x11 += einsum("ijka->jika", x10)
    del x10
    x20 += einsum("ijka->jkia", x11)
    x20 += einsum("ijka->ikja", x11) * -0.5
    del x11
    x192 += einsum("ijkl->lkji", x8)
    del x8
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x15 += einsum("ijka->ikja", x7) * -0.5
    x15 += einsum("ijka->ijka", x7)
    x16 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x16 += einsum("ijka->ikja", x7) * 2
    x16 += einsum("ijka->ijka", x7) * -1
    x24 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x24 += einsum("ijka->ikja", x7) * 2
    x24 -= einsum("ijka->ijka", x7)
    x51 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x51 += einsum("ijka->jkia", x7) * -0.5
    x51 += einsum("ijka->kjia", x7)
    x62 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x62 += einsum("ijka->jkia", x7) * -1
    x62 += einsum("ijka->kjia", x7) * 2
    x152 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x152 += einsum("ai,ijkb->jkab", l1, x7)
    x170 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x170 += einsum("ijab->ijab", x152)
    del x152
    x158 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x158 += einsum("ijka->ijka", x7)
    x239 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x239 -= einsum("ijka->ikja", x7)
    x239 += einsum("ijka->ijka", x7) * 2
    del x7
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x12 += einsum("iajb->jiab", v.ovov)
    x12 += einsum("iajb->jiba", v.ovov) * -0.5
    x13 = np.zeros((nocc, nvir), dtype=types[float])
    x13 += einsum("ia,ijba->jb", t1, x12)
    x14 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x14 += einsum("ia,jkab->jkib", x13, t2)
    x20 += einsum("ijka->ikja", x14)
    x20 += einsum("ijka->jkia", x14) * -2
    del x14
    x52 += einsum("ia->ia", x13)
    del x13
    x54 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x54 += einsum("wia,ijba->wjb", u11, x12)
    x63 = np.zeros((nocc, nvir), dtype=types[float])
    x63 += einsum("ia,ijba->jb", t1, x12) * 2
    x64 += einsum("ia->ia", x63)
    del x63
    x124 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x124 += einsum("wia,ijba->wjb", u11, x12) * 2
    x125 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x125 += einsum("wia->wia", x124)
    x194 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x194 += einsum("wia->wia", x124)
    x199 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x199 += einsum("wia->wia", x124)
    del x124
    x139 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x139 += einsum("wxia,ijba->wxjb", u12, x12)
    del x12
    x140 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x140 += einsum("wxai,xwjb->jiba", lu12, x139)
    del x139
    x145 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x145 += einsum("ijab->jiba", x140)
    del x140
    x15 += einsum("ijka->jkia", v.ooov)
    x15 += einsum("ijka->jika", v.ooov) * -0.5
    x20 += einsum("ijab,kila->kljb", t2, x15)
    del x15
    x16 += einsum("ijka->jkia", v.ooov) * -1
    x16 += einsum("ijka->jika", v.ooov) * 2
    x20 += einsum("ijab,kilb->klja", t2, x16) * 0.5
    del x16
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x17 += einsum("ijab->ijab", v.oovv) * 2
    x17 += einsum("iabj->ijba", v.ovvo) * -1
    x20 += einsum("ia,jkba->ijkb", t1, x17) * -0.5
    del x17
    x18 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x18 += einsum("ia,jkla->ijlk", t1, v.ooov)
    x19 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x19 += einsum("ijkl->jkil", x18) * -0.5
    x19 += einsum("ijkl->kjil", x18)
    x27 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x27 -= einsum("ijkl->ikjl", x18)
    x27 += einsum("ijkl->ijkl", x18) * 2
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum("ia,jikl->jkla", t1, x27)
    del x27
    x154 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x154 += einsum("abij,jkli->klba", l2, x18)
    del x18
    x170 -= einsum("ijab->ijab", x154)
    del x154
    x19 += einsum("ijkl->kijl", v.oooo) * -0.5
    x19 += einsum("ijkl->kilj", v.oooo)
    x20 += einsum("ia,ijkl->kjla", t1, x19)
    del x19
    x20 += einsum("ijak->kija", v.oovo) * -1
    x20 += einsum("ijak->jika", v.oovo) * 0.5
    l1new = np.zeros((nvir, nocc), dtype=types[float])
    l1new += einsum("abij,ikja->bk", l2, x20) * 2
    del x20
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x21 += einsum("abij->jiab", l2) * -1
    x21 += einsum("abij->jiba", l2) * 2
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x22 += einsum("ijab,ikac->kjcb", t2, x21) * -1
    del x21
    x22 += einsum("wxai,wxjb->ijab", lu12, u12) * -0.5
    x22 += einsum("abij,kjbc->ikac", l2, t2)
    x23 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x23 -= einsum("iabc->ibac", v.ovvv)
    x23 += einsum("iabc->ibca", v.ovvv) * 2
    x123 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x123 += einsum("wia,ibca->wbc", u11, x23)
    x129 = np.zeros((nvir, nvir), dtype=types[float])
    x129 += einsum("ia,ibca->bc", t1, x23)
    x130 = np.zeros((nvir, nvir), dtype=types[float])
    x130 += einsum("ab->ab", x129)
    x187 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x187 += einsum("ab,acij->ijbc", x129, l2)
    del x129
    x191 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x191 += einsum("ijab->jiba", x187)
    del x187
    l1new += einsum("ijab,jacb->ci", x22, x23) * -1
    del x22
    del x23
    x24 -= einsum("ijka->jkia", v.ooov)
    x24 += einsum("ijka->jika", v.ooov) * 2
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x25 += einsum("ijab->jiab", t2) * 2
    x25 -= einsum("ijab->jiba", t2)
    x28 -= einsum("ijka,klba->ijlb", x24, x25)
    del x24
    x224 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x224 += einsum("wia,ijba->wjb", g.bov, x25)
    x225 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x225 += einsum("wia->wia", x224)
    del x224
    x237 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x237 += einsum("wxai,ijba->xwjb", lu12, x25)
    del x25
    x26 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x26 += einsum("iabj->ijba", v.ovvo) * 2
    x26 -= einsum("ijab->ijab", v.oovv)
    x28 -= einsum("ia,jkba->ijkb", t1, x26)
    l1new += einsum("abij,ikjb->ak", l2, x28)
    del x28
    l1new += einsum("ai,jiab->bj", l1, x26)
    lu11new = np.zeros((nbos, nvir, nocc), dtype=types[float])
    lu11new += einsum("wai,jiab->wbj", lu11, x26)
    del x26
    x29 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x29 += einsum("ia,bcda->ibdc", t1, v.vvvv)
    x30 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x30 += einsum("iabc->iabc", x29)
    x30 += einsum("iabc->ibac", x29) * -0.5
    del x29
    x30 += einsum("aibc->iabc", v.vovv) * -0.5
    x30 += einsum("aibc->ibac", v.vovv)
    l1new += einsum("abij,ibac->cj", l2, x30) * 2
    del x30
    x31 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x31 += einsum("ia,wxaj->wxji", t1, lu12)
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum("wxia,wxjk->jkia", u12, x31) * -0.25
    x86 = np.zeros((nocc, nvir), dtype=types[float])
    x86 += einsum("wxia,wxij->ja", u12, x31)
    x99 = np.zeros((nocc, nvir), dtype=types[float])
    x99 += einsum("ia->ia", x86) * 0.5
    del x86
    x221 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x221 += einsum("wia,xwij->xja", u11, x31)
    x222 = np.zeros((nbos, nbos), dtype=types[float])
    x222 += einsum("wia,xia->wx", g.bov, x221)
    del x221
    x228 = np.zeros((nbos, nbos), dtype=types[float])
    x228 -= einsum("wx->wx", x222) * 2
    del x222
    lu12new = np.zeros((nbos, nbos, nvir, nocc), dtype=types[float])
    lu12new -= einsum("ijka,wxik->xwaj", x239, x31)
    del x239
    x32 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x32 += einsum("ia,bajk->jkib", t1, l2)
    x34 += einsum("ijab,kjlb->klia", t2, x32) * 0.5
    x36 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x36 -= einsum("ijka->ijka", x32)
    x36 += einsum("ijka->jika", x32) * 2
    x38 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x38 += einsum("ijka->ijka", x32) * 2
    x38 -= einsum("ijka->jika", x32)
    x43 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x43 += einsum("ijab,kjla->klib", t2, x32)
    x45 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x45 += einsum("ijka->ijka", x43) * 2
    x50 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x50 += einsum("ijka->ijka", x43)
    del x43
    x47 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x47 += einsum("ia,jkla->jkil", t1, x32)
    x48 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x48 += einsum("ijkl->ijkl", x47)
    x57 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x57 += einsum("ijkl->ijkl", x47) * 2
    x57 += einsum("ijkl->ijlk", x47) * -1
    l1new += einsum("ijka,ljki->al", v.ooov, x57)
    del x57
    x193 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x193 += einsum("ijkl->ijkl", x47)
    del x47
    x173 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x173 += einsum("iabc,jkib->kjac", v.ovvv, x32)
    x191 -= einsum("ijab->ijab", x173)
    del x173
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 += einsum("ijab->jiab", t2) * 2
    x33 += einsum("ijab->jiba", t2) * -1
    x34 += einsum("ijka,ilba->jklb", x32, x33) * -0.5
    x88 = np.zeros((nocc, nvir), dtype=types[float])
    x88 += einsum("ai,ijba->jb", l1, x33)
    x99 += einsum("ia->ia", x88) * -1
    del x88
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 -= einsum("iajb->jiab", v.ovov)
    x35 += einsum("iajb->jiba", v.ovov) * 2
    l1new += einsum("ijka,jkba->bi", x34, x35) * 2
    del x34
    del x35
    x37 += einsum("iabj->ijba", v.ovvo)
    l1new -= einsum("ijka,kjab->bi", x36, x37)
    del x36
    del x37
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 += einsum("ia,jabc->ijbc", t1, v.ovvv)
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x40 += einsum("ijab->jiab", x39)
    x156 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x156 += einsum("ijab->ijab", x39)
    del x39
    x40 += einsum("ijab->ijab", v.oovv)
    l1new -= einsum("ijka,kjab->bi", x38, x40)
    del x40
    x41 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x41 += einsum("iabc->ibac", v.ovvv) * 2
    x41 -= einsum("iabc->ibca", v.ovvv)
    x238 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x238 += einsum("ia,jbca->jibc", t1, x41)
    x42 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x42 += einsum("abij,kjac->ikbc", l2, t2)
    l1new -= einsum("iabc,jiac->bj", x41, x42)
    del x41
    del x42
    x44 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x44 += einsum("ai,jkba->ijkb", l1, t2)
    x45 += einsum("ijka->ikja", x44)
    x45 += einsum("ijka->ijka", x44) * -2
    del x44
    l1new += einsum("iajb,kija->bk", v.ovov, x45)
    del x45
    x46 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x46 += einsum("abij,klba->ijlk", l2, t2)
    x48 += einsum("ijkl->jilk", x46)
    x49 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x49 += einsum("ia,ijkl->jkla", t1, x48)
    del x48
    x50 += einsum("ijka->ijka", x49)
    x50 += einsum("ijka->ikja", x49) * -2
    del x49
    l1new += einsum("iajb,kijb->ak", v.ovov, x50) * -1
    del x50
    x56 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x56 += einsum("ijkl->jikl", x46) * 2
    x56 += einsum("ijkl->jilk", x46) * -1
    l1new += einsum("ijka,jlki->al", v.ooov, x56)
    del x56
    x193 += einsum("ijkl->jilk", x46)
    del x46
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    l2new += einsum("iajb,klji->ablk", v.ovov, x193)
    del x193
    x51 += einsum("ijka->ikja", v.ooov)
    x51 += einsum("ijka->kija", v.ooov) * -0.5
    x55 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x55 += einsum("wxia,jika->xwjk", u12, x51)
    del x51
    x52 += einsum("ia->ia", f.ov) * 0.5
    x55 += einsum("ia,wxja->xwij", x52, u12)
    x74 = np.zeros((nocc, nocc), dtype=types[float])
    x74 += einsum("ia,ja->ji", t1, x52) * 2
    x75 = np.zeros((nocc, nocc), dtype=types[float])
    x75 += einsum("ij->ij", x74)
    del x74
    x128 = np.zeros((nbos, nbos), dtype=types[float])
    x128 += einsum("ia,wxia->xw", x52, u12) * 2
    x200 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x200 += einsum("ia,wja->wij", x52, u11) * 2
    lu11new += einsum("w,ia->wai", ls1, x52) * 4
    del x52
    x53 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x53 += einsum("wx,xia->wia", s2, g.bov)
    x54 += einsum("wia->wia", x53) * 0.5
    x194 += einsum("wia->wia", x53)
    del x53
    x54 += einsum("wia->wia", gc.bov) * 0.5
    x55 += einsum("wia,xja->wxji", u11, x54) * 2
    del x54
    l1new += einsum("wxai,wxji->aj", lu12, x55) * -1
    del x55
    x58 = np.zeros((nocc, nvir), dtype=types[float])
    x58 += einsum("w,wai->ia", s1, g.bvo)
    x82 = np.zeros((nocc, nvir), dtype=types[float])
    x82 += einsum("ia->ia", x58)
    x213 = np.zeros((nocc, nvir), dtype=types[float])
    x213 += einsum("ia->ia", x58) * 0.5
    del x58
    x59 = np.zeros((nocc, nvir), dtype=types[float])
    x59 += einsum("wab,wib->ia", g.bvv, u11)
    x82 += einsum("ia->ia", x59)
    x213 += einsum("ia->ia", x59) * 0.5
    del x59
    x60 = np.zeros((nocc, nvir), dtype=types[float])
    x60 += einsum("ijab,jkib->ka", t2, v.ooov)
    x82 += einsum("ia->ia", x60)
    x213 += einsum("ia->ia", x60) * 0.5
    del x60
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum("ijab->jiab", t2) * -0.5
    x61 += einsum("ijab->jiba", t2)
    x71 = np.zeros((nocc, nocc), dtype=types[float])
    x71 += einsum("iajb,ikab->kj", v.ovov, x61) * 2
    x75 += einsum("ij->ji", x71)
    del x71
    x82 += einsum("iabc,ijac->jb", v.ovvv, x61) * 2
    x96 = np.zeros((nocc, nocc), dtype=types[float])
    x96 += einsum("abij,ikab->kj", l2, x61) * 2
    x97 = np.zeros((nocc, nocc), dtype=types[float])
    x97 += einsum("ij->ji", x96)
    x148 = np.zeros((nocc, nocc), dtype=types[float])
    x148 += einsum("ij->ji", x96)
    x215 = np.zeros((nocc, nocc), dtype=types[float])
    x215 += einsum("ij->ji", x96)
    del x96
    x104 = np.zeros((nvir, nvir), dtype=types[float])
    x104 += einsum("abij,ijac->cb", l2, x61) * 2
    x105 = np.zeros((nvir, nvir), dtype=types[float])
    x105 += einsum("ab->ba", x104)
    x146 = np.zeros((nvir, nvir), dtype=types[float])
    x146 += einsum("ab->ba", x104)
    del x104
    x141 = np.zeros((nvir, nvir), dtype=types[float])
    x141 += einsum("iajb,ijcb->ca", v.ovov, x61) * 2
    x142 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x142 += einsum("ab,acij->ijbc", x141, l2)
    del x141
    x145 += einsum("ijab->jiba", x142) * -1
    del x142
    x205 = np.zeros((nocc, nvir), dtype=types[float])
    x205 += einsum("iabc,ijac->jb", v.ovvv, x61)
    x213 += einsum("ia->ia", x205)
    del x205
    x214 = np.zeros((nvir, nvir), dtype=types[float])
    x214 += einsum("abij,ijac->bc", l2, x61) * 4
    del x61
    x62 += einsum("ijka->ikja", v.ooov) * 2
    x82 += einsum("ijab,ijkb->ka", t2, x62) * -1
    x206 = np.zeros((nocc, nvir), dtype=types[float])
    x206 += einsum("ijab,ijkb->ka", t2, x62) * 0.5
    del x62
    x213 += einsum("ia->ia", x206) * -1
    del x206
    x64 += einsum("ia->ia", f.ov)
    lu12new += einsum("wx,ia->xwai", ls2, x64) * 2
    x65 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x65 += einsum("ijab->jiab", t2)
    x65 += einsum("ijab->jiba", t2) * -0.5
    x82 += einsum("ia,ijba->jb", x64, x65) * 2
    x87 = np.zeros((nocc, nvir), dtype=types[float])
    x87 += einsum("ijka,ijba->kb", x32, x65) * 2
    x99 += einsum("ia->ia", x87)
    del x87
    x143 = np.zeros((nocc, nocc), dtype=types[float])
    x143 += einsum("iajb,ikba->kj", v.ovov, x65) * 2
    x144 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x144 += einsum("ij,abik->jkab", x143, l2)
    del x143
    x145 += einsum("ijab->jiba", x144) * -1
    del x144
    l2new += einsum("ijab->abij", x145)
    l2new += einsum("ijab->baji", x145)
    del x145
    x207 = np.zeros((nocc, nvir), dtype=types[float])
    x207 += einsum("ia,ijba->jb", x64, x65)
    del x65
    x213 += einsum("ia->ia", x207)
    del x207
    x66 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x66 += einsum("iabj->ijba", v.ovvo)
    x66 += einsum("ijab->ijab", v.oovv) * -0.5
    x82 += einsum("ia,ijba->jb", t1, x66) * 2
    x208 = np.zeros((nocc, nvir), dtype=types[float])
    x208 += einsum("ia,ijba->jb", t1, x66)
    del x66
    x213 += einsum("ia->ia", x208)
    del x208
    x67 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x67 += einsum("ia,wja->wji", t1, g.bov)
    x68 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x68 += einsum("wij->wij", x67)
    x115 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x115 += einsum("wij->wij", x67)
    x68 += einsum("wij->wij", g.boo)
    x82 += einsum("wia,wij->ja", u11, x68) * -1
    x200 += einsum("wx,wij->xij", s2, x68)
    x204 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x204 += einsum("wij,wxia->xja", x68, u12)
    x209 = np.zeros((nocc, nvir), dtype=types[float])
    x209 += einsum("wia,wij->ja", u11, x68) * 0.5
    del x68
    x213 += einsum("ia->ia", x209) * -1
    del x209
    x69 = np.zeros((nocc, nocc), dtype=types[float])
    x69 += einsum("w,wij->ij", s1, g.boo)
    x75 += einsum("ij->ij", x69)
    x165 = np.zeros((nocc, nocc), dtype=types[float])
    x165 += einsum("ij->ij", x69)
    del x69
    x70 = np.zeros((nocc, nocc), dtype=types[float])
    x70 += einsum("wia,wja->ij", g.bov, u11)
    x75 += einsum("ij->ij", x70)
    x165 += einsum("ij->ij", x70)
    del x70
    x72 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x72 += einsum("ijka->ikja", v.ooov)
    x72 += einsum("ijka->kija", v.ooov) * -0.5
    x73 = np.zeros((nocc, nocc), dtype=types[float])
    x73 += einsum("ia,jika->jk", t1, x72) * 2
    x75 += einsum("ij->ij", x73)
    del x73
    x200 += einsum("wia,jika->wjk", u11, x72) * 2
    del x72
    x75 += einsum("ij->ij", f.oo)
    x82 += einsum("ia,ij->ja", t1, x75) * -1
    x204 += einsum("ij,wia->wja", x75, u11)
    x210 = np.zeros((nocc, nvir), dtype=types[float])
    x210 += einsum("ia,ij->ja", t1, x75) * 0.5
    x213 += einsum("ia->ia", x210) * -1
    del x210
    l1new += einsum("ai,ji->aj", l1, x75) * -1
    lu12new += einsum("ij,wxaj->xwai", x75, lu12) * -1
    del x75
    x76 = np.zeros((nvir, nvir), dtype=types[float])
    x76 += einsum("w,wab->ab", s1, g.bvv)
    x79 = np.zeros((nvir, nvir), dtype=types[float])
    x79 += einsum("ab->ab", x76)
    x130 += einsum("ab->ab", x76)
    x161 = np.zeros((nvir, nvir), dtype=types[float])
    x161 += einsum("ab->ab", x76)
    x198 = np.zeros((nvir, nvir), dtype=types[float])
    x198 += einsum("ab->ab", x76)
    del x76
    x77 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x77 += einsum("iabc->ibac", v.ovvv) * -0.5
    x77 += einsum("iabc->ibca", v.ovvv)
    x78 = np.zeros((nvir, nvir), dtype=types[float])
    x78 += einsum("ia,ibca->bc", t1, x77) * 2
    x79 += einsum("ab->ab", x78)
    x198 += einsum("ab->ab", x78)
    del x78
    x201 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x201 += einsum("wia,ibca->wbc", u11, x77) * 2
    del x77
    x79 += einsum("ab->ab", f.vv)
    x82 += einsum("ia,ba->ib", t1, x79)
    x211 = np.zeros((nocc, nvir), dtype=types[float])
    x211 += einsum("ia,ba->ib", t1, x79) * 0.5
    del x79
    x213 += einsum("ia->ia", x211)
    del x211
    x80 = np.zeros((nbos), dtype=types[float])
    x80 += einsum("ia,wia->w", t1, g.bov)
    x81 = np.zeros((nbos), dtype=types[float])
    x81 += einsum("w->w", x80) * 2
    x135 = np.zeros((nbos), dtype=types[float])
    x135 += einsum("w->w", x80) * 2
    del x80
    x81 += einsum("w->w", G)
    x82 += einsum("w,wia->ia", x81, u11)
    x204 += einsum("w,wxia->xia", x81, u12) * -1
    x212 = np.zeros((nocc, nvir), dtype=types[float])
    x212 += einsum("w,wia->ia", x81, u11) * 0.5
    del x81
    x213 += einsum("ia->ia", x212)
    del x212
    x82 += einsum("ai->ia", f.vo)
    x83 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x83 += einsum("abij->jiab", l2) * 2
    x83 += einsum("abij->jiba", l2) * -1
    l1new += einsum("ia,ijba->bj", x82, x83)
    del x82
    del x83
    x84 = np.zeros((nocc, nvir), dtype=types[float])
    x84 += einsum("w,wia->ia", ls1, u11)
    x99 += einsum("ia->ia", x84) * -1
    del x84
    x85 = np.zeros((nocc, nvir), dtype=types[float])
    x85 += einsum("wx,wxia->ia", ls2, u12)
    x99 += einsum("ia->ia", x85) * -0.5
    del x85
    x89 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x89 += einsum("ia,waj->wji", t1, lu11)
    x91 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x91 += einsum("wij->wij", x89)
    x119 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x119 += einsum("wij->wij", x89)
    x90 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x90 += einsum("wia,wxaj->xji", u11, lu12)
    x91 += einsum("wij->wij", x90)
    x92 = np.zeros((nocc, nvir), dtype=types[float])
    x92 += einsum("wia,wij->ja", u11, x91)
    del x91
    x99 += einsum("ia->ia", x92)
    del x92
    x119 += einsum("wij->wij", x90)
    x120 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x120 += einsum("wx,wij->xij", s2, x119)
    x235 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x235 += einsum("wia,xji->xwja", g.bov, x119)
    x236 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x236 += einsum("wxia->xwia", x235)
    del x235
    x218 = np.zeros((nbos, nbos), dtype=types[float])
    x218 += einsum("wij,xji->wx", g.boo, x90)
    x228 -= einsum("wx->wx", x218) * 2
    del x218
    x223 = np.zeros((nbos, nbos), dtype=types[float])
    x223 += einsum("wij,xji->wx", x67, x90)
    del x90
    del x67
    x228 -= einsum("wx->wx", x223) * 2
    del x223
    x93 = np.zeros((nocc, nocc), dtype=types[float])
    x93 += einsum("ai,ja->ij", l1, t1)
    x97 += einsum("ij->ij", x93)
    x167 = np.zeros((nocc, nocc), dtype=types[float])
    x167 += einsum("ij->ij", x93)
    ls1new = np.zeros((nbos), dtype=types[float])
    ls1new -= einsum("ij,wji->w", x93, g.boo) * 2
    del x93
    x94 = np.zeros((nocc, nocc), dtype=types[float])
    x94 += einsum("wai,wja->ij", lu11, u11)
    x97 += einsum("ij->ij", x94)
    x167 += einsum("ij->ij", x94)
    x168 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x168 += einsum("ij,jakb->ikab", x167, v.ovov)
    del x167
    x170 += einsum("ijab->ijba", x168)
    del x168
    x215 += einsum("ij->ij", x94)
    del x94
    x95 = np.zeros((nocc, nocc), dtype=types[float])
    x95 += einsum("wxai,wxja->ij", lu12, u12)
    x97 += einsum("ij->ij", x95) * 0.5
    x98 = np.zeros((nocc, nvir), dtype=types[float])
    x98 += einsum("ia,ij->ja", t1, x97)
    x99 += einsum("ia->ia", x98)
    del x98
    l1new += einsum("ia,ji->aj", x64, x97) * -1
    del x64
    x148 += einsum("ij->ij", x95) * 0.5
    x149 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x149 += einsum("ij,jakb->ikab", x148, v.ovov)
    del x148
    x150 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x150 += einsum("ijab->ijba", x149)
    del x149
    x215 += einsum("ij->ij", x95) * 0.5
    del x95
    ls1new += einsum("ij,wji->w", x215, g.boo) * -2
    del x215
    x99 += einsum("ia->ia", t1) * -1
    ls1new += einsum("ia,wia->w", x99, g.bov) * -2
    x100 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x100 += einsum("iajb->jiab", v.ovov) * 2
    x100 += einsum("iajb->jiba", v.ovov) * -1
    l1new += einsum("ia,ijba->bj", x99, x100) * -1
    del x99
    del x100
    x101 = np.zeros((nvir, nvir), dtype=types[float])
    x101 += einsum("ai,ib->ab", l1, t1)
    x105 += einsum("ab->ab", x101)
    ls1new += einsum("ab,wab->w", x101, g.bvv) * 2
    del x101
    x102 = np.zeros((nvir, nvir), dtype=types[float])
    x102 += einsum("wai,wib->ab", lu11, u11)
    x105 += einsum("ab->ab", x102)
    x153 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x153 += einsum("ab,icjb->ijac", x102, v.ovov)
    x170 += einsum("ijab->ijab", x153)
    del x153
    x214 += einsum("ab->ab", x102) * 2
    del x102
    x103 = np.zeros((nvir, nvir), dtype=types[float])
    x103 += einsum("wxai,wxib->ab", lu12, u12)
    x105 += einsum("ab->ab", x103) * 0.5
    x146 += einsum("ab->ab", x103) * 0.5
    x147 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x147 += einsum("ab,ibjc->ijac", x146, v.ovov)
    del x146
    x150 += einsum("ijab->jiab", x147)
    del x147
    l2new += einsum("ijab->baij", x150) * -1
    l2new += einsum("ijab->abji", x150) * -1
    del x150
    x214 += einsum("ab->ab", x103)
    del x103
    ls1new += einsum("ab,wab->w", x214, g.bvv)
    del x214
    x106 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x106 += einsum("iabc->ibac", v.ovvv) * -1
    x106 += einsum("iabc->ibca", v.ovvv) * 2
    l1new += einsum("ab,iabc->ci", x105, x106)
    del x105
    del x106
    x107 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x107 -= einsum("ijka->ikja", v.ooov)
    x107 += einsum("ijka->kija", v.ooov) * 2
    l1new += einsum("ij,kjia->ak", x97, x107) * -1
    del x97
    del x107
    x108 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x108 += einsum("wia,wxja->xij", g.bov, u12)
    x116 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x116 += einsum("wij->wij", x108)
    x200 += einsum("wij->wij", x108)
    del x108
    x109 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x109 += einsum("ijka->ikja", v.ooov) * 2
    x109 -= einsum("ijka->kija", v.ooov)
    x116 += einsum("wia,jika->wjk", u11, x109)
    x183 = np.zeros((nocc, nocc), dtype=types[float])
    x183 += einsum("ia,jika->jk", t1, x109)
    x185 = np.zeros((nocc, nocc), dtype=types[float])
    x185 += einsum("ij->ij", x183)
    del x183
    lu12new -= einsum("ijka,wxki->xwaj", x109, x31)
    del x109
    x110 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x110 += einsum("iajb->jiab", v.ovov) * 2
    x110 -= einsum("iajb->jiba", v.ovov)
    x111 = np.zeros((nocc, nvir), dtype=types[float])
    x111 += einsum("ia,ijba->jb", t1, x110)
    x112 += einsum("ia->ia", x111)
    x184 = np.zeros((nocc, nocc), dtype=types[float])
    x184 += einsum("ia,ja->ji", t1, x111)
    x185 += einsum("ij->ij", x184)
    del x184
    x186 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x186 += einsum("ij,abjk->ikab", x185, l2)
    del x185
    x191 -= einsum("ijab->jiba", x186)
    del x186
    x189 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x189 += einsum("ia,jkib->jkab", x111, x32)
    x191 -= einsum("ijab->ijba", x189)
    del x189
    x191 += einsum("ai,jb->ijab", l1, x111)
    del x111
    x113 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x113 += einsum("wia,ijba->wjb", u11, x110)
    x114 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x114 += einsum("wia->wia", x113)
    x190 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x190 += einsum("wai,wjb->jiba", lu11, x113)
    del x113
    x191 += einsum("ijab->jiba", x190)
    del x190
    lu12new += einsum("ijab,wxib->xwaj", x110, x237)
    del x237
    del x110
    x112 += einsum("ia->ia", f.ov)
    x116 += einsum("ia,wja->wij", x112, u11)
    x134 = np.zeros((nbos), dtype=types[float])
    x134 += einsum("ia,wia->w", x112, u11)
    x137 = np.zeros((nbos), dtype=types[float])
    x137 += einsum("w->w", x134) * 2
    del x134
    lu12new -= einsum("ia,wxji->xwaj", x112, x31)
    del x112
    del x31
    x114 += einsum("wia->wia", gc.bov)
    x116 += einsum("ia,wja->wji", t1, x114)
    l1new -= einsum("wia,wji->aj", x114, x119)
    del x114
    del x119
    x115 += einsum("wij->wij", g.boo)
    x116 += einsum("wx,wij->xij", s2, x115)
    x227 = np.zeros((nbos, nbos), dtype=types[float])
    x227 += einsum("wij,xji->wx", x115, x89)
    del x89
    x228 -= einsum("wx->wx", x227) * 2
    del x227
    x234 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x234 += einsum("wai,xji->xwja", lu11, x115)
    x236 += einsum("wxia->wxia", x234)
    del x234
    x116 += einsum("wij->wij", gc.boo)
    l1new -= einsum("wai,wji->aj", lu11, x116)
    del x116
    x117 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x117 += einsum("abij->jiab", l2) * 2
    x117 -= einsum("abij->jiba", l2)
    x118 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x118 += einsum("wia,ijba->wjb", u11, x117)
    x120 += einsum("ia,wja->wji", t1, x118)
    x122 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x122 += einsum("wia->wia", x118)
    l1new -= einsum("wij,wja->ai", x115, x118)
    del x115
    del x118
    lu11new += einsum("wai,ijba->wbj", g.bvo, x117)
    del x117
    x120 += einsum("ai,wja->wij", l1, u11)
    x120 += einsum("wai,wxja->xij", lu11, u12)
    l1new -= einsum("wia,wji->aj", g.bov, x120)
    del x120
    x121 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x121 += einsum("wx,wai->xia", s2, lu11)
    x122 += einsum("wia->wia", x121)
    del x121
    x188 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x188 += einsum("wia,wjb->jiba", g.bov, x122)
    x191 += einsum("ijab->jiba", x188)
    del x188
    l1new += einsum("wab,wia->bi", g.bvv, x122)
    del x122
    x123 += einsum("wab->wab", gc.bvv)
    l1new += einsum("wai,wab->bi", lu11, x123)
    del x123
    x125 += einsum("wia->wia", gc.bov) * 2
    x128 += einsum("wia,xia->xw", u11, x125)
    del x125
    x126 = np.zeros((nbos, nbos), dtype=types[float])
    x126 += einsum("wia,xia->wx", g.bov, u11)
    x127 = np.zeros((nbos, nbos), dtype=types[float])
    x127 += einsum("wx->wx", x126) * 2
    x217 = np.zeros((nbos, nbos), dtype=types[float])
    x217 += einsum("wx,yx->yw", ls2, x126)
    x228 += einsum("wx->wx", x217) * 2
    del x217
    x233 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x233 += einsum("wx,xyai->wyia", x126, lu12)
    del x126
    x236 -= einsum("wxia->wxia", x233) * 2
    del x233
    x127 += einsum("wx->wx", w)
    x128 += einsum("wx,wy->xy", s2, x127)
    l1new += einsum("wx,xwai->ai", x128, lu12)
    del x128
    x204 += einsum("wx,wia->xia", x127, u11) * -1
    del x127
    x130 += einsum("ab->ab", f.vv)
    l1new += einsum("ai,ab->bi", l1, x130)
    del x130
    x131 = np.zeros((nbos), dtype=types[float])
    x131 += einsum("w,wx->x", s1, w)
    x137 += einsum("w->w", x131)
    del x131
    x132 = np.zeros((nbos), dtype=types[float])
    x132 += einsum("ia,wia->w", t1, gc.bov)
    x137 += einsum("w->w", x132) * 2
    del x132
    x133 = np.zeros((nbos), dtype=types[float])
    x133 += einsum("wia,wxia->x", g.bov, u12)
    x137 += einsum("w->w", x133) * 2
    del x133
    x135 += einsum("w->w", G)
    x136 = np.zeros((nbos), dtype=types[float])
    x136 += einsum("w,wx->x", x135, s2)
    x137 += einsum("w->w", x136)
    del x136
    x228 += einsum("w,x->xw", ls1, x135)
    del x135
    x137 += einsum("w->w", G)
    l1new += einsum("w,wai->ai", x137, lu11)
    ls1new += einsum("w,wx->x", x137, ls2)
    del x137
    x138 = np.zeros((nbos), dtype=types[float])
    x138 += einsum("w->w", s1)
    x138 += einsum("w,xw->x", ls1, s2)
    x138 += einsum("ai,wia->w", l1, u11) * 2
    x138 += einsum("wai,wxia->x", lu11, u12) * 2
    l1new += einsum("w,wia->ai", x138, g.bov)
    del x138
    x151 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x151 += einsum("ai,jbac->ijbc", l1, v.ovvv)
    x170 -= einsum("ijab->ijab", x151)
    del x151
    x155 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x155 += einsum("ijab,kbic->jkac", t2, v.ovov)
    x156 -= einsum("ijab->ijab", x155)
    del x155
    x156 += einsum("ijab->jiab", v.oovv)
    x157 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x157 += einsum("abij,ikbc->kjca", l2, x156)
    x170 += einsum("ijab->jiba", x157)
    del x157
    x180 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x180 += einsum("abij,ikac->kjcb", l2, x156)
    del x156
    x191 -= einsum("ijab->jiba", x180)
    del x180
    x158 += einsum("ijka->jkia", v.ooov)
    x159 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x159 += einsum("ijka,iljb->klab", x158, x32)
    x170 -= einsum("ijab->jiba", x159)
    del x159
    x181 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x181 += einsum("ijka,likb->ljba", x158, x38)
    del x38
    x191 -= einsum("ijab->ijab", x181)
    del x181
    x182 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x182 += einsum("ijka,lijb->klab", x158, x32)
    del x158
    x191 += einsum("ijab->jiba", x182)
    del x182
    x160 = np.zeros((nvir, nvir), dtype=types[float])
    x160 += einsum("wia,wib->ab", g.bov, u11)
    x161 -= einsum("ab->ba", x160)
    x198 += einsum("ab->ba", x160) * -1
    del x160
    x161 += einsum("ab->ab", f.vv)
    x162 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x162 += einsum("ab,acij->ijbc", x161, l2)
    del x161
    x170 -= einsum("ijab->jiab", x162)
    del x162
    x163 += einsum("ia->ia", f.ov)
    x164 = np.zeros((nocc, nocc), dtype=types[float])
    x164 += einsum("ia,ja->ji", t1, x163)
    x165 += einsum("ij->ij", x164)
    del x164
    x169 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x169 += einsum("ia,jkib->jkab", x163, x32)
    del x32
    x170 += einsum("ijab->ijab", x169)
    del x169
    x191 += einsum("ai,jb->jiba", l1, x163)
    del x163
    x165 += einsum("ij->ij", f.oo)
    x166 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x166 += einsum("ij,abjk->ikab", x165, l2)
    del x165
    x170 += einsum("ijab->ijba", x166)
    del x166
    l2new -= einsum("ijab->baij", x170)
    l2new -= einsum("ijab->abji", x170)
    del x170
    x171 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x171 += einsum("wia,wbj->ijab", gc.bov, lu11)
    x191 += einsum("ijab->ijab", x171)
    del x171
    x172 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x172 += einsum("ai,jikb->jkab", l1, v.ooov)
    x191 -= einsum("ijab->ijab", x172)
    del x172
    x174 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x174 -= einsum("abij->jiab", l2)
    x174 += einsum("abij->jiba", l2) * 2
    x175 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x175 += einsum("ijab,kaic->jkbc", t2, v.ovov)
    x178 -= einsum("ijab->ijab", x175)
    del x175
    x176 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x176 -= einsum("ijab->jiab", t2)
    x176 += einsum("ijab->jiba", t2) * 2
    x177 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x177 += einsum("iajb,ikac->kjcb", v.ovov, x176)
    del x176
    x178 += einsum("ijab->ijab", x177)
    del x177
    x178 += einsum("iabj->jiba", v.ovvo)
    x179 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x179 += einsum("ijab,ikac->jkbc", x174, x178)
    del x178
    del x174
    x191 += einsum("ijab->ijab", x179)
    del x179
    l2new += einsum("ijab->abij", x191)
    l2new += einsum("ijab->baji", x191)
    del x191
    x192 += einsum("ijkl->kilj", v.oooo)
    l2new += einsum("abij,klij->balk", l2, x192)
    del x192
    x194 += einsum("wia->wia", gc.bov)
    x204 += einsum("wia,ijba->wjb", x194, x33) * -1
    del x33
    del x194
    x195 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x195 += einsum("iabj->ijba", v.ovvo) * 2
    x195 += einsum("ijab->ijab", v.oovv) * -1
    x204 += einsum("wia,ijba->wjb", u11, x195) * -1
    del x195
    x196 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x196 += einsum("ijab->jiab", t2) * -1
    x196 += einsum("ijab->jiba", t2) * 2
    x197 = np.zeros((nvir, nvir), dtype=types[float])
    x197 += einsum("iajb,ijcb->ca", v.ovov, x196)
    del x196
    x198 += einsum("ab->ab", x197) * -1
    del x197
    x198 += einsum("ab->ab", f.vv)
    x204 += einsum("ab,wib->wia", x198, u11) * -1
    lu12new += einsum("ab,wxai->xwbi", x198, lu12)
    del x198
    x199 += einsum("wia->wia", gc.bov)
    x200 += einsum("ia,wja->wji", t1, x199)
    del x199
    x200 += einsum("wij->wij", gc.boo)
    x204 += einsum("ia,wij->wja", t1, x200)
    del x200
    x201 += einsum("wab->wab", gc.bvv)
    x204 += einsum("ia,wba->wib", t1, x201) * -1
    del x201
    x202 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x202 += einsum("ia,wba->wib", t1, g.bvv)
    x203 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x203 += einsum("wia->wia", x202)
    x225 += einsum("wia->wia", x202)
    del x202
    x203 += einsum("wai->wia", g.bvo)
    x204 += einsum("wx,wia->xia", s2, x203) * -1
    del x203
    x204 += einsum("wai->wia", gc.bvo) * -1
    x204 += einsum("wab,wxib->xia", g.bvv, u12) * -1
    ls1new += einsum("wia,wxai->x", x204, lu12) * -2
    del x204
    x213 += einsum("ai->ia", f.vo) * 0.5
    ls1new += einsum("ia,wai->w", x213, lu11) * 4
    ls2new = np.zeros((nbos, nbos), dtype=types[float])
    ls2new += einsum("ia,wxai->xw", x213, lu12) * 4
    del x213
    x216 = np.zeros((nbos, nbos), dtype=types[float])
    x216 += einsum("wx,yx->wy", ls2, w)
    x228 += einsum("wx->wx", x216)
    del x216
    x219 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x219 += einsum("wia,wxbi->xba", u11, lu12)
    x220 = np.zeros((nbos, nbos), dtype=types[float])
    x220 += einsum("wab,xab->wx", g.bvv, x219)
    x228 += einsum("wx->wx", x220) * 2
    del x220
    x232 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x232 += einsum("wia,xba->wxib", g.bov, x219)
    del x219
    x236 += einsum("wxia->wxia", x232)
    del x232
    x225 += einsum("wai->wia", g.bvo)
    x226 = np.zeros((nbos, nbos), dtype=types[float])
    x226 += einsum("wai,xia->xw", lu11, x225)
    del x225
    x228 += einsum("wx->wx", x226) * 2
    del x226
    ls2new += einsum("wx->wx", x228)
    ls2new += einsum("wx->xw", x228)
    del x228
    x229 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x229 += einsum("wia,xyai->xyw", u11, lu12)
    lu12new += einsum("wia,xyw->yxai", g.bov, x229) * 2
    del x229
    x230 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x230 += einsum("wx,xyai->ywia", w, lu12)
    x236 -= einsum("wxia->wxia", x230)
    del x230
    x231 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x231 += einsum("wab,xai->wxib", g.bvv, lu11)
    x236 -= einsum("wxia->wxia", x231)
    del x231
    lu12new -= einsum("wxia->wxai", x236)
    lu12new -= einsum("wxia->xwai", x236)
    del x236
    x238 += einsum("iabj->ijba", v.ovvo) * 2
    x238 -= einsum("ijab->ijab", v.oovv)
    lu12new += einsum("wxai,jiab->xwbj", lu12, x238)
    del x238
    l1new += einsum("ia->ai", f.ov)
    l1new += einsum("w,wia->ai", ls1, gc.bov)
    l2new += einsum("abij,acbd->dcji", l2, v.vvvv)
    l2new += einsum("iajb->baji", v.ovov)
    ls1new += einsum("w->w", G)
    ls1new += einsum("w,xw->x", ls1, w)
    ls1new += einsum("ai,wai->w", l1, g.bvo) * 2
    lu11new -= einsum("ij,waj->wai", f.oo, lu11)
    lu11new -= einsum("ai,wji->waj", l1, g.boo)
    lu11new -= einsum("wij,xwaj->xai", gc.boo, lu12)
    lu11new += einsum("wab,xwai->xbi", gc.bvv, lu12)
    lu11new += einsum("ab,wai->wbi", f.vv, lu11)
    lu11new += einsum("ai,wab->wbi", l1, g.bvv)
    lu11new += einsum("wia->wai", g.bov)
    lu11new += einsum("w,xwai->xai", G, lu12)
    lu11new += einsum("wx,xia->wai", ls2, gc.bov)
    lu11new += einsum("wx,xai->wai", w, lu11)

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
    x0 += einsum("wxai,wxja->ij", lu12, u12)
    x10 = np.zeros((nocc, nocc), dtype=types[float])
    x10 += einsum("ij->ij", x0) * 0.5
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=types[float])
    rdm1_f_oo += einsum("ij->ij", x0) * -1
    del x0
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 += einsum("wai,wja->ij", lu11, u11)
    x10 += einsum("ij->ij", x1)
    rdm1_f_oo -= einsum("ij->ij", x1) * 2
    del x1
    x2 = np.zeros((nocc, nocc), dtype=types[float])
    x2 += einsum("ai,ja->ij", l1, t1)
    x10 += einsum("ij->ij", x2)
    rdm1_f_oo -= einsum("ij->ij", x2) * 2
    del x2
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x3 += einsum("ijab->jiab", t2) * -1
    x3 += einsum("ijab->jiba", t2) * 2
    rdm1_f_oo += einsum("abij,ikab->jk", l2, x3) * -2
    del x3
    x4 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x4 += einsum("ia,wxaj->wxji", t1, lu12)
    rdm1_f_vo = np.zeros((nvir, nocc), dtype=types[float])
    rdm1_f_vo += einsum("wxia,wxij->aj", u12, x4) * -1
    del x4
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x5 += einsum("ia,bajk->jkib", t1, l2)
    x6 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x6 += einsum("ijka->ijka", x5) * 2
    x6 += einsum("ijka->jika", x5) * -1
    del x5
    rdm1_f_vo += einsum("ijab,jikb->ak", t2, x6) * -2
    del x6
    x7 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x7 -= einsum("ijab->jiab", t2)
    x7 += einsum("ijab->jiba", t2) * 2
    rdm1_f_vo += einsum("ai,ijab->bj", l1, x7) * 2
    del x7
    x8 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x8 += einsum("ia,waj->wji", t1, lu11)
    x8 += einsum("wia,wxaj->xji", u11, lu12)
    rdm1_f_vo -= einsum("wia,wij->aj", u11, x8) * 2
    del x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x9 += einsum("ijab->jiab", t2)
    x9 += einsum("ijab->jiba", t2) * -0.5
    x10 += einsum("abij,ikba->jk", l2, x9) * 2
    del x9
    rdm1_f_vo += einsum("ia,ij->aj", t1, x10) * -2
    del x10
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum("abij->jiab", l2)
    x11 += einsum("abij->jiba", l2) * -0.5
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=types[float])
    rdm1_f_vv += einsum("ijab,ijbc->ac", t2, x11) * 4
    del x11
    rdm1_f_oo += einsum("ij->ji", delta_oo) * 2
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=types[float])
    rdm1_f_ov += einsum("ai->ia", l1) * 2
    rdm1_f_vo += einsum("wx,wxia->ai", ls2, u12)
    rdm1_f_vo += einsum("w,wia->ai", ls1, u11) * 2
    rdm1_f_vo += einsum("ia->ai", t1) * 2
    rdm1_f_vv += einsum("wxai,wxib->ba", lu12, u12)
    rdm1_f_vv += einsum("wai,wib->ba", lu11, u11) * 2
    rdm1_f_vv += einsum("ai,ib->ba", l1, t1) * 2

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
    x0 = np.zeros((nocc, nocc), dtype=types[float])
    x0 += einsum("wai,wja->ij", lu11, u11)
    x13 = np.zeros((nocc, nocc), dtype=types[float])
    x13 += einsum("ij->ij", x0)
    x32 = np.zeros((nocc, nvir), dtype=types[float])
    x32 += einsum("ia,ij->ja", t1, x0)
    x37 = np.zeros((nocc, nvir), dtype=types[float])
    x37 += einsum("ia->ia", x32)
    del x32
    x65 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x65 += einsum("ij,kiab->kjab", x0, t2)
    x76 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x76 += einsum("ijab->ijab", x65)
    del x65
    x111 = np.zeros((nocc, nocc), dtype=types[float])
    x111 += einsum("ij->ij", x0)
    rdm2_f_oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_oooo -= einsum("ij,kl->ijkl", delta_oo, x0) * 4
    rdm2_f_oooo += einsum("ij,kl->ilkj", delta_oo, x0) * 2
    rdm2_f_oooo += einsum("ij,kl->kijl", delta_oo, x0) * 2
    rdm2_f_oooo -= einsum("ij,kl->klji", delta_oo, x0) * 4
    del x0
    x1 = np.zeros((nocc, nocc), dtype=types[float])
    x1 += einsum("ai,ja->ij", l1, t1)
    x13 += einsum("ij->ij", x1)
    x14 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x14 += einsum("ia,jk->jika", t1, x13)
    del x13
    x41 = np.zeros((nocc, nvir), dtype=types[float])
    x41 += einsum("ia,ij->ja", t1, x1)
    x42 = np.zeros((nocc, nvir), dtype=types[float])
    x42 += einsum("ia->ia", x41) * -1
    del x41
    x79 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x79 += einsum("ij,kiab->jkab", x1, t2)
    x86 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x86 += einsum("ijab->ijab", x79)
    del x79
    x111 += einsum("ij->ij", x1)
    x112 = np.zeros((nocc, nvir), dtype=types[float])
    x112 += einsum("ia,ij->ja", t1, x111)
    del x111
    x113 = np.zeros((nocc, nvir), dtype=types[float])
    x113 += einsum("ia->ia", x112)
    x114 = np.zeros((nocc, nvir), dtype=types[float])
    x114 += einsum("ia->ia", x112)
    del x112
    rdm2_f_oooo -= einsum("ij,kl->jikl", delta_oo, x1) * 4
    rdm2_f_oooo += einsum("ij,kl->kijl", delta_oo, x1) * 2
    rdm2_f_oooo += einsum("ij,kl->jlki", delta_oo, x1) * 2
    rdm2_f_oooo -= einsum("ij,kl->klji", delta_oo, x1) * 4
    del x1
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x2 += einsum("abij,klba->ijlk", l2, t2)
    x19 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x19 += einsum("ia,jikl->jkla", t1, x2)
    x25 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x25 += einsum("ijka->ijka", x19) * -4
    x43 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x43 += einsum("ijka->ijka", x19) * 2
    x106 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x106 += einsum("ia,ijkb->kjba", t1, x19)
    del x19
    x109 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x109 += einsum("ijab->ijab", x106) * 2.0000000000000013
    del x106
    x107 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x107 += einsum("ijkl->jilk", x2)
    rdm2_f_oooo += einsum("ijkl->jkil", x2) * -2
    rdm2_f_oooo += einsum("ijkl->jlik", x2) * 4
    del x2
    x3 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x3 += einsum("ia,bajk->jkib", t1, l2)
    x4 = np.zeros((nocc, nocc, nocc, nocc), dtype=types[float])
    x4 += einsum("ia,jkla->kjli", t1, x3)
    x29 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x29 += einsum("ia,ijkl->jlka", t1, x4)
    x38 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x38 -= einsum("ijka->ijka", x29)
    x47 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x47 += einsum("ijka->ijka", x29)
    x89 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x89 += einsum("ia,ijkb->jkab", t1, x29)
    del x29
    x91 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x91 += einsum("ijab->ijab", x89)
    del x89
    x107 += einsum("ijkl->ijkl", x4)
    x108 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x108 += einsum("ijab,ijkl->klab", t2, x107) * 2
    del x107
    x109 += einsum("ijab->ijab", x108)
    del x108
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_vovo += einsum("ijab->ajbi", x109) * -1
    rdm2_f_vovo += einsum("ijab->bjai", x109) * 2
    del x109
    rdm2_f_oooo += einsum("ijkl->ikjl", x4) * 4
    rdm2_f_oooo -= einsum("ijkl->iljk", x4) * 2
    del x4
    x12 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x12 += einsum("ijab,kjla->klib", t2, x3)
    x14 -= einsum("ijka->ijka", x12)
    x84 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x84 -= einsum("ijka->ijka", x12)
    del x12
    x21 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x21 += einsum("ijka->ijka", x3)
    x21 += einsum("ijka->jika", x3) * -0.5
    x22 = np.zeros((nocc, nvir), dtype=types[float])
    x22 += einsum("ijab,ijka->kb", t2, x21) * 4
    x24 = np.zeros((nocc, nvir), dtype=types[float])
    x24 += einsum("ia->ia", x22)
    del x22
    x99 = np.zeros((nocc, nvir), dtype=types[float])
    x99 += einsum("ijab,ijka->kb", t2, x21) * 4.000000000000003
    del x21
    x102 = np.zeros((nocc, nvir), dtype=types[float])
    x102 += einsum("ia->ia", x99)
    del x99
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x28 += einsum("ijab,jkla->klib", t2, x3)
    x38 -= einsum("ijka->ijka", x28)
    del x28
    x30 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x30 -= einsum("ijka->ijka", x3)
    x30 += einsum("ijka->jika", x3) * 2
    x31 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x31 += einsum("ijab,kila->kljb", t2, x30)
    x38 += einsum("ijka->ijka", x31)
    del x31
    x61 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x61 += einsum("ia,ijkb->jkab", t1, x30)
    del x30
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_ovvo -= einsum("ijab->ibaj", x61) * 2
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_voov -= einsum("ijab->ajib", x61) * 2
    del x61
    x44 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x44 += einsum("ijab,kjlb->klia", t2, x3)
    x47 += einsum("ijka->ijka", x44)
    x68 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x68 += einsum("ijka->ijka", x44)
    del x44
    x52 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x52 += einsum("ijka->ijka", x3) * 2
    x52 -= einsum("ijka->jika", x3)
    x53 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x53 += einsum("ia,ijkb->jkab", t1, x52)
    del x52
    rdm2_f_oovv = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_oovv -= einsum("ijab->ijab", x53) * 2
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_vvoo -= einsum("ijab->abij", x53) * 2
    del x53
    x119 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x119 += einsum("ia,jikb->jkba", t1, x3)
    x121 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x121 += einsum("ijab->ijab", x119)
    del x119
    x131 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x131 += einsum("ijab,ijkc->kcab", t2, x3)
    x132 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x132 += einsum("iabc->iabc", x131) * 2
    x133 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x133 += einsum("iabc->iabc", x131) * 4
    del x131
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    rdm2_f_ooov += einsum("ijka->ikja", x3) * 2
    rdm2_f_ooov -= einsum("ijka->jkia", x3) * 4
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=types[float])
    rdm2_f_ovoo -= einsum("ijka->iajk", x3) * 4
    rdm2_f_ovoo += einsum("ijka->jaik", x3) * 2
    x5 = np.zeros((nocc, nocc), dtype=types[float])
    x5 += einsum("wxai,wxja->ij", lu12, u12)
    x8 = np.zeros((nocc, nocc), dtype=types[float])
    x8 += einsum("ij->ij", x5)
    x16 = np.zeros((nocc, nocc), dtype=types[float])
    x16 += einsum("ij->ij", x5) * 0.5
    x100 = np.zeros((nocc, nocc), dtype=types[float])
    x100 += einsum("ij->ij", x5) * 0.49999999999999967
    del x5
    x6 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x6 += einsum("ijab->jiab", t2)
    x6 += einsum("ijab->jiba", t2) * -0.5
    x7 = np.zeros((nocc, nocc), dtype=types[float])
    x7 += einsum("abij,ikba->kj", l2, x6) * 4
    x8 += einsum("ij->ji", x7)
    del x7
    rdm2_f_oooo += einsum("ij,kl->jikl", delta_oo, x8) * -2
    rdm2_f_oooo += einsum("ij,kl->jlki", delta_oo, x8)
    rdm2_f_oooo += einsum("ij,kl->kjil", delta_oo, x8)
    rdm2_f_oooo += einsum("ij,kl->klij", delta_oo, x8) * -2
    del x8
    x15 = np.zeros((nocc, nocc), dtype=types[float])
    x15 += einsum("abij,ikba->kj", l2, x6) * 2
    x16 += einsum("ij->ji", x15)
    x23 = np.zeros((nocc, nvir), dtype=types[float])
    x23 += einsum("ia,ij->ja", t1, x16) * 2
    x24 += einsum("ia->ia", x23)
    del x23
    x98 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x98 += einsum("ij,ikab->jkab", x16, t2) * 4
    x103 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x103 += einsum("ijab->jiba", x98)
    del x98
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=types[float])
    rdm2_f_oovo += einsum("ia,jk->jiak", t1, x16) * 2
    rdm2_f_oovo += einsum("ia,jk->jkai", t1, x16) * -4
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=types[float])
    rdm2_f_vooo += einsum("ia,jk->aijk", t1, x16) * -4
    rdm2_f_vooo += einsum("ia,jk->akji", t1, x16) * 2
    del x16
    x100 += einsum("ij->ji", x15)
    del x15
    x101 = np.zeros((nocc, nvir), dtype=types[float])
    x101 += einsum("ia,ij->ja", t1, x100) * 2.0000000000000013
    del x100
    x102 += einsum("ia->ia", x101)
    del x101
    x93 = np.zeros((nbos, nbos, nocc, nvir), dtype=types[float])
    x93 += einsum("wxai,ijba->wxjb", lu12, x6) * 2
    del x6
    x94 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x94 += einsum("wxia,xwjb->ijab", u12, x93) * 2
    del x93
    x103 += einsum("ijab->jiba", x94) * -1
    del x94
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x9 += einsum("ai,jkba->ijkb", l1, t2)
    x14 += einsum("ijka->ijka", x9)
    x84 += einsum("ijka->ijka", x9)
    del x9
    x85 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x85 += einsum("ia,ijkb->jkab", t1, x84)
    del x84
    x86 += einsum("ijab->ijab", x85)
    del x85
    x10 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x10 += einsum("wia,wxaj->xji", u11, lu12)
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x11 += einsum("wia,wjk->jika", u11, x10)
    x14 += einsum("ijka->ijka", x11)
    del x11
    rdm2_f_oovo += einsum("ijka->ijak", x14) * 2
    rdm2_f_oovo -= einsum("ijka->ikaj", x14) * 4
    rdm2_f_vooo -= einsum("ijka->ajik", x14) * 4
    rdm2_f_vooo += einsum("ijka->akij", x14) * 2
    del x14
    x35 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x35 += einsum("wij->wij", x10)
    x74 = np.zeros((nocc, nvir), dtype=types[float])
    x74 += einsum("wia,wij->ja", u11, x10)
    x75 = np.zeros((nocc, nvir), dtype=types[float])
    x75 += einsum("ia->ia", x74)
    del x74
    x82 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x82 += einsum("ia,wij->wja", t1, x10)
    del x10
    x83 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x83 += einsum("wia,wjb->ijba", u11, x82)
    del x82
    x86 += einsum("ijab->ijab", x83)
    del x83
    x17 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x17 += einsum("ia,wxaj->wxji", t1, lu12)
    x18 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x18 += einsum("wxia,wxjk->jkia", u12, x17)
    x25 += einsum("ijka->ijka", x18) * 2
    x43 += einsum("ijka->ijka", x18) * -1
    rdm2_f_vooo += einsum("ijka->ajik", x43) * -1
    rdm2_f_vooo += einsum("ijka->akij", x43) * 2
    del x43
    x92 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x92 += einsum("ia,ijkb->jkab", t1, x18)
    del x18
    x103 += einsum("ijab->ijab", x92) * 2
    del x92
    x20 = np.zeros((nocc, nvir), dtype=types[float])
    x20 += einsum("wxia,wxij->ja", u12, x17)
    x24 += einsum("ia->ia", x20)
    x25 += einsum("ij,ka->jika", delta_oo, x24) * 2
    rdm2_f_oovo += einsum("ijka->ijak", x25) * -1
    rdm2_f_oovo += einsum("ijka->ikaj", x25) * 0.5
    del x25
    rdm2_f_vooo += einsum("ij,ka->ajik", delta_oo, x24)
    rdm2_f_vooo += einsum("ij,ka->akij", delta_oo, x24) * -2
    del x24
    x102 += einsum("ia->ia", x20)
    del x20
    x103 += einsum("ia,jb->ijab", t1, x102) * 2
    del x102
    x80 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x80 += einsum("wia,wxij->xja", u11, x17)
    del x17
    x81 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x81 += einsum("wia,wjb->jiab", u11, x80)
    del x80
    x86 += einsum("ijab->ijab", x81)
    del x81
    rdm2_f_vovo += einsum("ijab->aibj", x86) * 2
    rdm2_f_vovo -= einsum("ijab->biaj", x86) * 4
    rdm2_f_vovo -= einsum("ijab->ajbi", x86) * 4
    rdm2_f_vovo += einsum("ijab->bjai", x86) * 2
    del x86
    x26 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x26 += einsum("ia,waj->wji", t1, lu11)
    x27 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x27 += einsum("wia,wjk->jkia", u11, x26)
    x38 += einsum("ijka->ijka", x27)
    x47 -= einsum("ijka->ijka", x27)
    del x27
    x35 += einsum("wij->wij", x26)
    x36 = np.zeros((nocc, nvir), dtype=types[float])
    x36 += einsum("wia,wij->ja", u11, x35)
    del x35
    x37 += einsum("ia->ia", x36)
    del x36
    x70 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x70 += einsum("ia,wij->wja", t1, x26)
    x72 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x72 += einsum("wia->wia", x70)
    del x70
    x110 = np.zeros((nocc, nvir), dtype=types[float])
    x110 += einsum("wia,wij->ja", u11, x26)
    del x26
    x113 += einsum("ia->ia", x110)
    x114 += einsum("ia->ia", x110)
    del x110
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x33 -= einsum("ijab->jiab", t2)
    x33 += einsum("ijab->jiba", t2) * 2
    x34 = np.zeros((nocc, nvir), dtype=types[float])
    x34 += einsum("ai,ijab->jb", l1, x33)
    x37 -= einsum("ia->ia", x34)
    x38 += einsum("ij,ka->jika", delta_oo, x37)
    rdm2_f_oovo -= einsum("ijka->ijak", x38) * 4
    rdm2_f_oovo += einsum("ijka->ikaj", x38) * 2
    del x38
    rdm2_f_vooo += einsum("ij,ka->ajik", delta_oo, x37) * 2
    rdm2_f_vooo -= einsum("ij,ka->akij", delta_oo, x37) * 4
    del x37
    x75 -= einsum("ia->ia", x34)
    del x34
    x76 += einsum("ia,jb->ijab", t1, x75)
    del x75
    x71 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x71 += einsum("wai,ijab->wjb", lu11, x33)
    x72 -= einsum("wia->wia", x71)
    del x71
    x73 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x73 += einsum("wia,wjb->ijab", u11, x72)
    del x72
    x76 += einsum("ijab->jiba", x73)
    del x73
    x120 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x120 += einsum("abij,ikac->kjcb", l2, x33)
    x121 -= einsum("ijab->jiba", x120)
    del x120
    rdm2_f_vvoo -= einsum("abij,ikcb->cajk", l2, x33) * 2
    del x33
    x39 = np.zeros((nocc, nvir), dtype=types[float])
    x39 += einsum("w,wia->ia", ls1, u11)
    x42 += einsum("ia->ia", x39)
    x113 += einsum("ia->ia", x39) * -1
    x114 += einsum("ia->ia", x39) * -1
    del x39
    x40 = np.zeros((nocc, nvir), dtype=types[float])
    x40 += einsum("wx,wxia->ia", ls2, u12)
    x42 += einsum("ia->ia", x40) * 0.5
    x113 += einsum("ia->ia", x40) * -0.5
    x114 += einsum("ia->ia", x40) * -0.5
    del x40
    rdm2_f_vovo += einsum("ia,jb->bjai", t1, x114) * -4
    rdm2_f_vovo += einsum("ia,jb->ajbi", t1, x114) * 2
    del x114
    x42 += einsum("ia->ia", t1)
    rdm2_f_oovo += einsum("ij,ka->jkai", delta_oo, x42) * -2
    rdm2_f_oovo += einsum("ij,ka->jiak", delta_oo, x42) * 4
    rdm2_f_vooo += einsum("ij,ka->aijk", delta_oo, x42) * -2
    rdm2_f_vooo += einsum("ij,ka->akji", delta_oo, x42) * 4
    del x42
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x45 += einsum("ijab->jiab", t2) * 2
    x45 -= einsum("ijab->jiba", t2)
    x46 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x46 += einsum("ijka,ilba->ljkb", x3, x45)
    del x3
    x47 -= einsum("ijka->jkia", x46)
    rdm2_f_vooo -= einsum("ijka->ajik", x47) * 2
    rdm2_f_vooo += einsum("ijka->akij", x47) * 4
    del x47
    x68 -= einsum("ijka->jkia", x46)
    del x46
    x69 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x69 += einsum("ia,ijkb->jkab", t1, x68)
    del x68
    x76 -= einsum("ijab->ijab", x69)
    del x69
    rdm2_f_vvoo -= einsum("abij,ikca->cbjk", l2, x45) * 2
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x48 += einsum("wxai,wxjb->ijab", lu12, u12)
    x130 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x130 += einsum("ia,ijbc->jbac", t1, x48)
    x132 += einsum("iabc->iabc", x130) * -1
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=types[float])
    rdm2_f_vovv += einsum("iabc->bica", x132)
    rdm2_f_vovv += einsum("iabc->ciba", x132) * -2
    del x132
    x133 += einsum("iabc->iabc", x130) * -2
    del x130
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=types[float])
    rdm2_f_vvvo += einsum("iabc->baci", x133) * -1
    rdm2_f_vvvo += einsum("iabc->cabi", x133) * 0.5
    del x133
    rdm2_f_oovv += einsum("ijab->ijba", x48) * -1
    rdm2_f_ovvo += einsum("ijab->iabj", x48) * 2
    rdm2_f_voov += einsum("ijab->bjia", x48) * 2
    rdm2_f_vvoo += einsum("ijab->baij", x48) * -1
    del x48
    x49 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x49 += einsum("wai,wjb->ijab", lu11, u11)
    rdm2_f_oovv -= einsum("ijab->ijba", x49) * 2
    rdm2_f_ovvo += einsum("ijab->iabj", x49) * 4
    rdm2_f_voov += einsum("ijab->bjia", x49) * 4
    rdm2_f_vvoo -= einsum("ijab->baij", x49) * 2
    del x49
    x50 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x50 -= einsum("abij->jiab", l2)
    x50 += einsum("abij->jiba", l2) * 2
    x60 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x60 += einsum("ijab,ikbc->jkac", x45, x50)
    del x45
    rdm2_f_ovvo += einsum("ijab->jbai", x60) * 2
    rdm2_f_voov += einsum("ijab->aijb", x60) * 2
    del x60
    rdm2_f_oovv -= einsum("ijab,ikac->kjbc", t2, x50) * 2
    del x50
    x51 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x51 += einsum("abij->jiab", l2) * 2
    x51 -= einsum("abij->jiba", l2)
    x66 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x66 += einsum("ijab,ikca->kjcb", t2, x51)
    x67 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x67 += einsum("ijab,ikbc->kjca", t2, x66)
    x76 += einsum("ijab->ijab", x67)
    del x67
    x90 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x90 += einsum("ijab,ikac->kjcb", t2, x66)
    del x66
    x91 += einsum("ijab->ijab", x90) * 2
    del x90
    rdm2_f_oovv -= einsum("ijab,ikbc->kjac", t2, x51) * 2
    del x51
    x54 = np.zeros((nvir, nvir), dtype=types[float])
    x54 += einsum("ai,ib->ab", l1, t1)
    x59 = np.zeros((nvir, nvir), dtype=types[float])
    x59 += einsum("ab->ab", x54) * 0.5
    x63 = np.zeros((nvir, nvir), dtype=types[float])
    x63 += einsum("ab->ab", x54)
    x128 = np.zeros((nvir, nvir), dtype=types[float])
    x128 += einsum("ab->ab", x54)
    del x54
    x55 = np.zeros((nvir, nvir), dtype=types[float])
    x55 += einsum("wai,wib->ab", lu11, u11)
    x59 += einsum("ab->ab", x55) * 0.5
    x63 += einsum("ab->ab", x55)
    x64 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x64 += einsum("ab,ijac->jicb", x55, t2)
    x76 += einsum("ijab->ijab", x64)
    del x64
    rdm2_f_vovo -= einsum("ijab->aibj", x76) * 4
    rdm2_f_vovo += einsum("ijab->biaj", x76) * 2
    rdm2_f_vovo += einsum("ijab->ajbi", x76) * 2
    rdm2_f_vovo -= einsum("ijab->bjai", x76) * 4
    del x76
    x128 += einsum("ab->ab", x55)
    del x55
    x129 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x129 += einsum("ia,bc->ibac", t1, x128)
    del x128
    x56 = np.zeros((nvir, nvir), dtype=types[float])
    x56 += einsum("wxai,wxib->ab", lu12, u12)
    x59 += einsum("ab->ab", x56) * 0.25
    x63 += einsum("ab->ab", x56) * 0.5
    x96 = np.zeros((nvir, nvir), dtype=types[float])
    x96 += einsum("ab->ab", x56)
    del x56
    x57 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x57 += einsum("abij->jiab", l2)
    x57 += einsum("abij->jiba", l2) * -0.5
    x58 = np.zeros((nvir, nvir), dtype=types[float])
    x58 += einsum("ijab,ijca->cb", t2, x57)
    x59 += einsum("ab->ab", x58)
    del x58
    rdm2_f_oovv += einsum("ij,ab->jiba", delta_oo, x59) * 8
    rdm2_f_vvoo += einsum("ij,ab->baji", delta_oo, x59) * 8
    del x59
    x62 = np.zeros((nvir, nvir), dtype=types[float])
    x62 += einsum("ijab,ijca->cb", t2, x57) * 2
    x63 += einsum("ab->ab", x62)
    del x62
    rdm2_f_ovvo += einsum("ij,ab->jabi", delta_oo, x63) * -2
    rdm2_f_voov += einsum("ij,ab->bija", delta_oo, x63) * -2
    del x63
    x95 = np.zeros((nvir, nvir), dtype=types[float])
    x95 += einsum("ijab,ijca->cb", t2, x57) * 4
    del x57
    x96 += einsum("ab->ab", x95)
    del x95
    x97 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x97 += einsum("ab,ijac->ijbc", x96, t2) * 2
    x103 += einsum("ijab->jiba", x97)
    del x97
    rdm2_f_vovo += einsum("ijab->aibj", x103) * -1
    rdm2_f_vovo += einsum("ijab->biaj", x103) * 0.5
    rdm2_f_vovo += einsum("ijab->ajbi", x103) * 0.5
    rdm2_f_vovo += einsum("ijab->bjai", x103) * -1
    del x103
    rdm2_f_vovv += einsum("ia,bc->aicb", t1, x96) * 2
    rdm2_f_vovv += einsum("ia,bc->ciab", t1, x96) * -1
    rdm2_f_vvvo += einsum("ia,bc->abci", t1, x96) * -1
    rdm2_f_vvvo += einsum("ia,bc->cbai", t1, x96) * 2
    del x96
    x77 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x77 += einsum("wx,xia->wia", ls2, u11)
    x78 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x78 += einsum("wia,wjb->jiba", u11, x77)
    del x77
    rdm2_f_vovo -= einsum("ijab->biaj", x78) * 2
    rdm2_f_vovo += einsum("ijab->bjai", x78) * 4
    del x78
    x87 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x87 += einsum("abij,kjbc->ikac", l2, t2)
    x88 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x88 += einsum("ijab,jkac->ikbc", t2, x87)
    del x87
    x91 += einsum("ijab->ijab", x88)
    del x88
    x91 += einsum("ijab->jiba", t2)
    rdm2_f_vovo -= einsum("ijab->biaj", x91) * 2
    rdm2_f_vovo += einsum("ijab->aibj", x91) * 4
    del x91
    x104 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x104 += einsum("abij,kjac->ikbc", l2, t2)
    x105 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x105 += einsum("ijab,jkac->ikbc", t2, x104)
    rdm2_f_vovo += einsum("ijab->ajbi", x105) * 4
    rdm2_f_vovo -= einsum("ijab->bjai", x105) * 2
    del x105
    x127 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x127 += einsum("ia,ijbc->jbac", t1, x104)
    del x104
    x129 -= einsum("iabc->iabc", x127)
    del x127
    x113 += einsum("ia->ia", t1) * -1
    rdm2_f_vovo += einsum("ia,jb->aibj", t1, x113) * -4
    rdm2_f_vovo += einsum("ia,jb->biaj", t1, x113) * 2
    del x113
    x115 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x115 += einsum("ia,bcji->jbca", t1, l2)
    x135 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x135 += einsum("ia,ibcd->cbda", t1, x115)
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_vvvv -= einsum("abcd->cbda", x135) * 2
    rdm2_f_vvvv += einsum("abcd->dbca", x135) * 4
    del x135
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    rdm2_f_ovvv += einsum("iabc->iacb", x115) * 4
    rdm2_f_ovvv -= einsum("iabc->ibca", x115) * 2
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=types[float])
    rdm2_f_vvov -= einsum("iabc->caib", x115) * 2
    rdm2_f_vvov += einsum("iabc->cbia", x115) * 4
    del x115
    x116 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x116 += einsum("ia,wbi->wba", t1, lu11)
    x117 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x117 += einsum("wia,wbc->ibca", u11, x116)
    del x116
    x123 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x123 -= einsum("iabc->iabc", x117)
    del x117
    x118 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x118 += einsum("abij,kjca->ikbc", l2, t2)
    x121 += einsum("ijab->ijab", x118)
    del x118
    x122 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x122 += einsum("ia,ijbc->jabc", t1, x121)
    del x121
    x123 += einsum("iabc->ibac", x122)
    del x122
    rdm2_f_vovv += einsum("iabc->bica", x123) * 2
    rdm2_f_vovv -= einsum("iabc->ciba", x123) * 4
    rdm2_f_vvvo -= einsum("iabc->baci", x123) * 4
    rdm2_f_vvvo += einsum("iabc->cabi", x123) * 2
    del x123
    x124 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x124 += einsum("ai,jibc->jabc", l1, t2)
    x129 += einsum("iabc->iabc", x124)
    del x124
    x125 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x125 += einsum("wia,xwbi->xba", u11, lu12)
    x126 = np.zeros((nocc, nvir, nvir, nvir), dtype=types[float])
    x126 += einsum("wia,wbc->ibac", u11, x125)
    del x125
    x129 += einsum("iabc->iabc", x126)
    del x126
    rdm2_f_vovv += einsum("iabc->bica", x129) * 4
    rdm2_f_vovv -= einsum("iabc->ciba", x129) * 2
    rdm2_f_vvvo -= einsum("iabc->baci", x129) * 2
    rdm2_f_vvvo += einsum("iabc->cabi", x129) * 4
    del x129
    x134 = np.zeros((nvir, nvir, nvir, nvir), dtype=types[float])
    x134 += einsum("abij,ijcd->abcd", l2, t2)
    rdm2_f_vvvv += einsum("abcd->cbda", x134) * -2
    rdm2_f_vvvv += einsum("abcd->dbca", x134) * 4
    del x134
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
    dm_b_des += einsum("wai,xwia->x", lu11, u12) * 2
    dm_b_des += einsum("w,xw->x", ls1, s2)
    dm_b_des += einsum("w->w", s1)
    dm_b_des += einsum("ai,wia->w", l1, u11) * 2

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
    rdm1_b += einsum("wxai,yxia->wy", lu12, u12) * 2
    rdm1_b += einsum("wx,yx->wy", ls2, s2)
    rdm1_b += einsum("w,x->wx", ls1, s1)
    rdm1_b += einsum("wai,xia->wx", lu11, u11) * 2

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
    x5 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x5 += einsum("wij->wij", x0)
    x8 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x8 += einsum("wx,xij->wij", s2, x0)
    x30 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x30 += einsum("wij->wij", x8)
    rdm_eb_des_oo = np.zeros((nbos, nocc, nocc), dtype=types[float])
    rdm_eb_des_oo -= einsum("wij->wji", x8) * 2
    del x8
    x36 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x36 += einsum("wij->wij", x0)
    rdm_eb_cre_oo = np.zeros((nbos, nocc, nocc), dtype=types[float])
    rdm_eb_cre_oo -= einsum("wij->wji", x0) * 2
    del x0
    x1 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x1 += einsum("ia,waj->wji", t1, lu11)
    x5 += einsum("wij->wij", x1)
    rdm_eb_cre_ov = np.zeros((nbos, nocc, nvir), dtype=types[float])
    rdm_eb_cre_ov -= einsum("ia,wij->wja", t1, x5) * 2
    rdm_eb_des_ov = np.zeros((nbos, nocc, nvir), dtype=types[float])
    rdm_eb_des_ov -= einsum("wij,wxia->xja", x5, u12) * 2
    del x5
    x36 += einsum("wij->wij", x1)
    x38 = np.zeros((nocc, nvir), dtype=types[float])
    x38 += einsum("wia,wij->ja", u11, x36) * 0.9999999999999993
    del x36
    rdm_eb_cre_oo -= einsum("wij->wji", x1) * 2
    del x1
    x2 = np.zeros((nbos, nbos, nocc, nocc), dtype=types[float])
    x2 += einsum("ia,wxaj->wxji", t1, lu12)
    x3 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x3 += einsum("wia,xwij->xja", u11, x2)
    rdm_eb_cre_ov -= einsum("wia->wia", x3) * 2
    rdm_eb_des_ov -= einsum("wx,xia->wia", s2, x3) * 2
    del x3
    x38 += einsum("wxia,wxij->ja", u12, x2) * 0.49999999999999967
    del x2
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x4 -= einsum("ijab->jiab", t2)
    x4 += einsum("ijab->jiba", t2) * 2
    rdm_eb_cre_ov += einsum("wai,ijab->wjb", lu11, x4) * 2
    x6 = np.zeros((nbos, nvir, nvir), dtype=types[float])
    x6 += einsum("wia,xwbi->xba", u11, lu12)
    rdm_eb_cre_vv = np.zeros((nbos, nvir, nvir), dtype=types[float])
    rdm_eb_cre_vv += einsum("wab->wab", x6) * 2
    rdm_eb_des_ov -= einsum("wab,xwia->xib", x6, u12) * 2
    rdm_eb_des_vv = np.zeros((nbos, nvir, nvir), dtype=types[float])
    rdm_eb_des_vv += einsum("wx,xab->wab", s2, x6) * 2
    del x6
    x7 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x7 += einsum("wai,xwja->xij", lu11, u12)
    x30 += einsum("wij->wij", x7)
    rdm_eb_des_oo -= einsum("wij->wji", x7) * 2
    del x7
    x9 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x9 += einsum("ai,wja->wij", l1, u11)
    x30 += einsum("wij->wij", x9)
    rdm_eb_des_oo -= einsum("wij->wji", x9) * 2
    del x9
    x10 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x10 += einsum("wx,xai->wia", s2, lu11)
    x13 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x13 += einsum("wia->wia", x10)
    rdm_eb_des_vo = np.zeros((nbos, nvir, nocc), dtype=types[float])
    rdm_eb_des_vo += einsum("wia->wai", x10) * 2
    del x10
    x11 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x11 += einsum("abij->jiab", l2) * 2
    x11 -= einsum("abij->jiba", l2)
    x12 = np.zeros((nbos, nocc, nvir), dtype=types[float])
    x12 += einsum("wia,ijba->wjb", u11, x11)
    del x11
    x13 += einsum("wia->wia", x12)
    del x12
    x14 = np.zeros((nbos, nocc, nocc), dtype=types[float])
    x14 += einsum("ia,wja->wji", t1, x13)
    x30 += einsum("wij->wij", x14)
    rdm_eb_des_ov -= einsum("ia,wij->wja", t1, x30) * 2
    del x30
    rdm_eb_des_oo -= einsum("wij->wji", x14) * 2
    del x14
    rdm_eb_des_ov += einsum("wia,ijab->wjb", x13, x4) * 2
    del x4
    rdm_eb_des_vv += einsum("ia,wib->wba", t1, x13) * 2
    del x13
    x15 = np.zeros((nocc, nocc), dtype=types[float])
    x15 += einsum("ai,ja->ij", l1, t1)
    x20 = np.zeros((nocc, nocc), dtype=types[float])
    x20 += einsum("ij->ij", x15)
    x31 = np.zeros((nocc, nocc), dtype=types[float])
    x31 += einsum("ij->ij", x15)
    x37 = np.zeros((nocc, nocc), dtype=types[float])
    x37 += einsum("ij->ij", x15)
    del x15
    x16 = np.zeros((nocc, nocc), dtype=types[float])
    x16 += einsum("wai,wja->ij", lu11, u11)
    x20 += einsum("ij->ij", x16)
    x31 += einsum("ij->ij", x16)
    x37 += einsum("ij->ij", x16)
    del x16
    x17 = np.zeros((nocc, nocc), dtype=types[float])
    x17 += einsum("wxai,wxja->ij", lu12, u12)
    x20 += einsum("ij->ij", x17) * 0.5
    x31 += einsum("ij->ij", x17) * 0.5
    x37 += einsum("ij->ij", x17) * 0.5
    del x17
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x18 += einsum("ijab->jiab", t2)
    x18 += einsum("ijab->jiba", t2) * -0.5
    x19 = np.zeros((nocc, nocc), dtype=types[float])
    x19 += einsum("abij,ikba->jk", l2, x18) * 2
    x20 += einsum("ij->ij", x19)
    x31 += einsum("ij->ij", x19)
    del x19
    rdm_eb_des_ov += einsum("ij,wia->wja", x31, u11) * -2
    del x31
    x37 += einsum("abij,ikba->jk", l2, x18) * 2.0000000000000013
    del x18
    x38 += einsum("ia,ij->ja", t1, x37) * 0.9999999999999993
    del x37
    x20 += einsum("ij->ji", delta_oo) * -1
    rdm_eb_des_oo += einsum("w,ij->wji", s1, x20) * -2
    del x20
    x21 = np.zeros((nbos), dtype=types[float])
    x21 += einsum("w,xw->x", ls1, s2)
    x24 = np.zeros((nbos), dtype=types[float])
    x24 += einsum("w->w", x21)
    del x21
    x22 = np.zeros((nbos), dtype=types[float])
    x22 += einsum("ai,wia->w", l1, u11)
    x24 += einsum("w->w", x22) * 2
    del x22
    x23 = np.zeros((nbos), dtype=types[float])
    x23 += einsum("wai,xwia->x", lu11, u12)
    x24 += einsum("w->w", x23) * 2
    del x23
    rdm_eb_des_oo += einsum("w,ij->wji", x24, delta_oo) * 2
    rdm_eb_des_ov += einsum("w,ia->wia", x24, t1) * 2
    del x24
    x25 = np.zeros((nbos, nbos, nbos), dtype=types[float])
    x25 += einsum("wia,xyai->xyw", u11, lu12)
    rdm_eb_des_ov += einsum("wxy,wxia->yia", x25, u12) * 2
    del x25
    x26 = np.zeros((nvir, nvir), dtype=types[float])
    x26 += einsum("wai,wib->ab", lu11, u11)
    x29 = np.zeros((nvir, nvir), dtype=types[float])
    x29 += einsum("ab->ab", x26)
    x40 = np.zeros((nvir, nvir), dtype=types[float])
    x40 += einsum("ab->ab", x26) * 2
    del x26
    x27 = np.zeros((nvir, nvir), dtype=types[float])
    x27 += einsum("wxai,wxib->ab", lu12, u12)
    x29 += einsum("ab->ab", x27) * 0.5
    x40 += einsum("ab->ab", x27)
    del x27
    x28 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x28 += einsum("abij->jiab", l2) * -0.5
    x28 += einsum("abij->jiba", l2)
    x29 += einsum("ijab,ijcb->ca", t2, x28) * 2
    rdm_eb_des_ov += einsum("ab,wia->wib", x29, u11) * -2
    del x29
    x40 += einsum("ijab,ijcb->ca", t2, x28) * 4
    del x28
    x32 = np.zeros((nbos, nbos), dtype=types[float])
    x32 += einsum("wx,yw->xy", ls2, s2)
    x32 += einsum("wai,xia->wx", lu11, u11) * 2
    x32 += einsum("wxai,ywia->xy", lu12, u12) * 2
    rdm_eb_des_ov += einsum("wx,wia->xia", x32, u11) * 2
    del x32
    x33 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x33 += einsum("ia,abjk->kjib", t1, l2)
    x34 = np.zeros((nocc, nocc, nocc, nvir), dtype=types[float])
    x34 += einsum("ijka->ijka", x33) * -0.5
    x34 += einsum("ijka->jika", x33)
    del x33
    x38 += einsum("ijab,ijkb->ka", t2, x34) * 2
    del x34
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x35 += einsum("ijab->jiab", t2) * -1
    x35 += einsum("ijab->jiba", t2) * 2
    x38 += einsum("ai,ijab->jb", l1, x35) * -0.9999999999999993
    del x35
    x38 += einsum("ia->ia", t1) * -0.9999999999999993
    x38 += einsum("w,wia->ia", ls1, u11) * -0.9999999999999993
    x38 += einsum("wx,wxia->ia", ls2, u12) * -0.49999999999999967
    rdm_eb_des_ov += einsum("w,ia->wia", s1, x38) * -2.0000000000000013
    del x38
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=types[float])
    x39 -= einsum("abij->jiab", l2)
    x39 += einsum("abij->jiba", l2) * 2
    rdm_eb_des_vo += einsum("wia,ijab->wbj", u11, x39) * 2
    del x39
    x40 += einsum("ai,ib->ab", l1, t1) * 2
    rdm_eb_des_vv += einsum("w,ab->wab", s1, x40)
    del x40
    rdm_eb_cre_oo += einsum("w,ij->wji", ls1, delta_oo) * 2
    rdm_eb_cre_ov += einsum("wx,xia->wia", ls2, u11) * 2
    rdm_eb_cre_ov += einsum("w,ia->wia", ls1, t1) * 2
    rdm_eb_cre_vo = np.zeros((nbos, nvir, nocc), dtype=types[float])
    rdm_eb_cre_vo += einsum("wai->wai", lu11) * 2
    rdm_eb_cre_vv += einsum("ia,wbi->wba", t1, lu11) * 2
    rdm_eb_des_ov += einsum("wia->wia", u11) * 2
    rdm_eb_des_ov += einsum("w,xwia->xia", ls1, u12) * 2
    rdm_eb_des_vo += einsum("w,ai->wai", s1, l1) * 2
    rdm_eb_des_vv += einsum("wai,xwib->xab", lu11, u12) * 2
    rdm_eb_des_vv += einsum("ai,wib->wab", l1, u11) * 2

    rdm_eb = np.array([
            np.block([[rdm_eb_cre_oo, rdm_eb_cre_ov], [rdm_eb_cre_vo, rdm_eb_cre_vv]]),
            np.block([[rdm_eb_des_oo, rdm_eb_des_ov], [rdm_eb_des_vo, rdm_eb_des_vv]]),
    ])

    return rdm_eb

