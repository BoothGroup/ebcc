# Code generated by qwick.

import numpy as np
from pyscf import lib
from types import SimpleNamespace
from ebcc.codegen import common

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # energy
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += lib.einsum("ijab->jiba", t2)
    x0 += lib.einsum("ia,jb->ijab", t1, t1) * 2
    e_cc = 0
    e_cc += lib.einsum("ijab,ijab->", v.oovv, x0) * 0.25
    del x0
    e_cc += lib.einsum("ia,ia->", f.ov, t1)

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T1 and T2 amplitudes
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x0 += lib.einsum("ia,jkba->ijkb", t1, v.oovv)
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x1 += lib.einsum("ijka->ikja", x0) * -1
    x20 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x20 -= lib.einsum("ia,jkla->lkij", t1, x0)
    del x0
    x1 += lib.einsum("ijka->kjia", v.ooov)
    t1new = np.zeros((nocc, nvir), dtype=np.float64)
    t1new += lib.einsum("ijab,kija->kb", t2, x1) * -0.5
    del x1
    x2 = np.zeros((nocc, nvir), dtype=np.float64)
    x2 += lib.einsum("ia,jiba->jb", t1, v.oovv)
    x3 = np.zeros((nocc, nvir), dtype=np.float64)
    x3 += lib.einsum("ia->ia", x2)
    x4 = np.zeros((nocc, nvir), dtype=np.float64)
    x4 += lib.einsum("ia->ia", x2)
    del x2
    x3 += lib.einsum("ia->ia", f.ov)
    t1new += lib.einsum("ia,ijab->jb", x3, t2)
    del x3
    x4 += lib.einsum("ia->ia", f.ov)
    x5 = np.zeros((nocc, nocc), dtype=np.float64)
    x5 += lib.einsum("ia,ja->ji", t1, x4) * 2
    del x4
    x5 += lib.einsum("ij->ij", f.oo) * 2
    x5 += lib.einsum("ia,ijka->jk", t1, v.ooov) * -2
    x5 += lib.einsum("ijab,jkab->ki", t2, v.oovv) * -1
    t1new += lib.einsum("ia,ij->ja", t1, x5) * -0.5
    del x5
    x6 = np.zeros((nvir, nvir), dtype=np.float64)
    x6 += lib.einsum("ab->ab", f.vv)
    x6 += lib.einsum("ia,ibac->bc", t1, v.ovvv)
    t1new += lib.einsum("ia,ba->ib", t1, x6)
    del x6
    x7 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x7 += lib.einsum("ia,bcda->ibcd", t1, v.vvvv)
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2new -= lib.einsum("ia,jbca->ijcb", t1, x7)
    del x7
    x8 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x8 += lib.einsum("ij,kiab->jkab", f.oo, t2)
    x13 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x13 += lib.einsum("ijab->ijba", x8)
    del x8
    x9 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x9 += lib.einsum("ia,bcja->ijbc", t1, v.vvov)
    x13 -= lib.einsum("ijab->ijba", x9)
    del x9
    x10 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x10 += lib.einsum("ia,jkla->ijkl", t1, v.ooov)
    x11 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x11 += lib.einsum("ia,jkil->jkla", t1, x10)
    del x10
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x12 -= lib.einsum("ia,jikb->jkba", t1, x11)
    del x11
    x13 -= lib.einsum("ijab->ijba", x12)
    del x12
    t2new -= lib.einsum("ijab->ijab", x13)
    t2new += lib.einsum("ijab->jiab", x13)
    del x13
    x14 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x14 += lib.einsum("ab,ijcb->ijac", f.vv, t2)
    t2new += lib.einsum("ijab->jiab", x14)
    t2new -= lib.einsum("ijab->jiba", x14)
    del x14
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x15 += lib.einsum("ia,jbka->ijkb", t1, v.ovov)
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x16 += lib.einsum("ia,jikb->jkab", t1, x15)
    del x15
    t2new += lib.einsum("ijab->ijab", x16)
    t2new -= lib.einsum("ijab->ijba", x16)
    t2new -= lib.einsum("ijab->jiab", x16)
    t2new += lib.einsum("ijab->jiba", x16)
    del x16
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x17 += lib.einsum("ia,jbca->ijbc", t1, v.ovvv)
    x18 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x18 -= lib.einsum("ia,jkba->jikb", t1, x17)
    del x17
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x19 -= lib.einsum("ia,jkib->kjab", t1, x18)
    del x18
    t2new -= lib.einsum("ijab->ijab", x19)
    t2new += lib.einsum("ijab->ijba", x19)
    del x19
    x20 += lib.einsum("ijkl->jilk", v.oooo)
    x21 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x21 += lib.einsum("ia,ijkl->lkja", t1, x20)
    del x20
    x21 += lib.einsum("iajk->kjia", v.ovoo)
    t2new += lib.einsum("ia,jkib->jkab", t1, x21)
    del x21
    t1new += lib.einsum("ijab,jcab->ic", t2, v.ovvv) * -0.5
    t1new += lib.einsum("ai->ia", f.vo)
    t1new -= lib.einsum("ia,ibja->jb", t1, v.ovov)
    t2new -= lib.einsum("ia,ibjk->kjba", t1, v.ovoo)
    t2new += lib.einsum("abij->jiba", v.vvoo)

    return {"t1new": t1new, "t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    # L1 and L2 amplitudes
    x0 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x0 += lib.einsum("ia,bajk->jkib", t1, l2)
    x6 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x6 += lib.einsum("ia,jkla->kjli", t1, x0)
    x15 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x15 += lib.einsum("ijkl->ijlk", x6) * 2.0000000000000013
    l1new = np.zeros((nvir, nocc), dtype=np.float64)
    l1new += lib.einsum("ijka,lkji->al", v.ooov, x6) * 0.5
    l2new = np.zeros((nvir, nvir, nocc, nocc), dtype=np.float64)
    l2new += lib.einsum("ijab,klij->bakl", v.oovv, x6) * -0.5
    del x6
    x16 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x16 += lib.einsum("ijab,kjlb->kila", t2, x0) * -2
    x26 = np.zeros((nocc, nvir), dtype=np.float64)
    x26 += lib.einsum("ijab,ijka->kb", t2, x0)
    x30 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x30 -= lib.einsum("ijka,kljb->liba", v.ooov, x0)
    x34 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x34 += lib.einsum("ijab->ijab", x30)
    del x30
    l1new -= lib.einsum("iajb,kjia->bk", v.ovov, x0)
    l2new += lib.einsum("iabc,jkia->cbkj", v.ovvv, x0)
    x1 = np.zeros((nocc, nocc), dtype=np.float64)
    x1 += lib.einsum("ai,ja->ij", l1, t1)
    x25 = np.zeros((nocc, nocc), dtype=np.float64)
    x25 += lib.einsum("ij->ij", x1)
    x29 = np.zeros((nocc, nocc), dtype=np.float64)
    x29 += lib.einsum("ij->ij", x1) * 2
    x41 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x41 += lib.einsum("ij,kjab->ikab", x1, v.oovv)
    x45 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x45 += lib.einsum("ijab->ijba", x41)
    del x41
    x2 = np.zeros((nocc, nvir), dtype=np.float64)
    x2 += lib.einsum("ia,jiba->jb", t1, v.oovv)
    x11 = np.zeros((nocc, nvir), dtype=np.float64)
    x11 += lib.einsum("ia->ia", x2)
    x34 += lib.einsum("ai,jb->ijab", l1, x2)
    x37 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x37 -= lib.einsum("ia,jkib->kjba", x2, x0)
    x38 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x38 += lib.einsum("ijab->ijab", x37)
    del x37
    x42 = np.zeros((nocc, nocc), dtype=np.float64)
    x42 += lib.einsum("ia,ja->ij", t1, x2)
    x43 = np.zeros((nocc, nocc), dtype=np.float64)
    x43 += lib.einsum("ij->ji", x42)
    del x42
    l1new -= lib.einsum("ij,ja->ai", x1, x2)
    del x1
    del x2
    x3 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x3 += lib.einsum("abij,kjcb->ikac", l2, t2)
    l1new -= lib.einsum("iabc,jiac->bj", v.ovvv, x3)
    del x3
    x4 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x4 += lib.einsum("ia,jbca->ijbc", t1, v.ovvv)
    x12 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x12 += lib.einsum("ijab->jiab", x4) * -0.5
    x32 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x32 += lib.einsum("ijab->ijab", x4)
    l1new += lib.einsum("ijka,jkab->bi", x0, x4)
    del x4
    x5 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x5 += lib.einsum("abij,klab->ijkl", l2, t2)
    x15 += lib.einsum("ijkl->jilk", x5) * -1
    x16 += lib.einsum("ia,ijkl->jlka", t1, x15) * -0.5
    del x15
    l1new += lib.einsum("ijka,lkij->al", v.ooov, x5) * -0.25
    del x5
    x7 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x7 += lib.einsum("ia,jkba->ijkb", t1, v.oovv)
    x8 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x8 += lib.einsum("ijka->ikja", x7) * -1
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x9 += lib.einsum("ijka->kjia", x7) * 0.5000000000000003
    x31 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x31 += lib.einsum("ijka,jlkb->ilab", x0, x7)
    del x0
    x34 -= lib.einsum("ijab->ijab", x31)
    del x31
    x35 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x35 -= lib.einsum("ai,ijkb->kjab", l1, x7)
    x38 -= lib.einsum("ijab->ijab", x35)
    del x35
    x49 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x49 += lib.einsum("ijka->kjia", x7) * 0.5
    del x7
    x8 += lib.einsum("ijka->kjia", v.ooov)
    x13 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x13 += lib.einsum("ijab,kila->ljkb", t2, x8) * 2
    x24 = np.zeros((nocc, nvir), dtype=np.float64)
    x24 += lib.einsum("ijab,kija->kb", t2, x8) * 0.5
    del x8
    x9 += lib.einsum("ijka->jika", v.ooov) * -1
    x10 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x10 += lib.einsum("ia,jkla->kjil", t1, x9) * -4
    del x9
    x10 += lib.einsum("ijkl->jilk", v.oooo) * -2
    x10 += lib.einsum("ijab,klab->lkji", t2, v.oovv) * -1
    x13 += lib.einsum("ia,ijkl->jlka", t1, x10) * -0.5
    del x10
    x11 += lib.einsum("ia->ia", f.ov)
    x13 += lib.einsum("ia,jkab->ikjb", x11, t2)
    x20 = np.zeros((nocc, nocc), dtype=np.float64)
    x20 += lib.einsum("ia,ja->ji", t1, x11)
    x21 = np.zeros((nocc, nocc), dtype=np.float64)
    x21 += lib.einsum("ij->ij", x20)
    del x20
    x24 += lib.einsum("ia,ijab->jb", x11, t2) * -1
    del x11
    x12 += lib.einsum("iajb->ijab", v.ovov)
    x13 += lib.einsum("ia,jkba->jkib", t1, x12) * -2
    del x12
    x13 += lib.einsum("iajk->ikja", v.ovoo)
    x13 += lib.einsum("ijab,kcab->kjic", t2, v.ovvv) * 0.5
    l1new += lib.einsum("abij,kija->bk", l2, x13) * -0.5
    del x13
    x14 = np.zeros((nocc, nocc), dtype=np.float64)
    x14 += lib.einsum("abij,ikab->jk", l2, t2)
    x16 += lib.einsum("ia,jk->jika", t1, x14)
    x25 += lib.einsum("ij->ij", x14) * 0.5
    x26 += lib.einsum("ia,ij->ja", t1, x25) * 2
    l1new += lib.einsum("ij,jkia->ak", x25, v.ooov) * -1
    del x25
    x29 += lib.einsum("ij->ij", x14)
    del x14
    l1new += lib.einsum("ia,ji->aj", f.ov, x29) * -0.5
    del x29
    x16 += lib.einsum("ai,jkba->ikjb", l1, t2) * -1
    l1new += lib.einsum("ijab,kija->bk", v.oovv, x16) * -0.5
    del x16
    x17 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x17 += lib.einsum("abic->ibac", v.vvov) * -1
    x17 += lib.einsum("ia,bcda->icbd", t1, v.vvvv)
    l1new += lib.einsum("abij,iabc->cj", l2, x17) * 0.5
    del x17
    x18 = np.zeros((nocc, nocc), dtype=np.float64)
    x18 += lib.einsum("ia,jika->jk", t1, v.ooov)
    x21 += lib.einsum("ij->ij", x18)
    x43 += lib.einsum("ij->ij", x18)
    del x18
    x44 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x44 += lib.einsum("ij,abjk->ikab", x43, l2)
    del x43
    x45 += lib.einsum("ijab->jiba", x44)
    del x44
    x19 = np.zeros((nocc, nocc), dtype=np.float64)
    x19 += lib.einsum("ijab,ikab->jk", t2, v.oovv)
    x21 += lib.einsum("ij->ji", x19) * 0.5
    del x19
    x21 += lib.einsum("ij->ij", f.oo)
    x24 += lib.einsum("ia,ij->ja", t1, x21)
    l1new += lib.einsum("ai,ji->aj", l1, x21) * -1
    del x21
    x22 = np.zeros((nvir, nvir), dtype=np.float64)
    x22 += lib.einsum("ia,ibca->bc", t1, v.ovvv)
    x23 = np.zeros((nvir, nvir), dtype=np.float64)
    x23 += lib.einsum("ab->ab", x22) * -1
    x28 = np.zeros((nvir, nvir), dtype=np.float64)
    x28 -= lib.einsum("ab->ab", x22)
    x36 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x36 -= lib.einsum("ab,caij->jicb", x22, l2)
    del x22
    x38 += lib.einsum("ijab->ijab", x36)
    del x36
    l2new -= lib.einsum("ijab->abij", x38)
    l2new += lib.einsum("ijab->baij", x38)
    del x38
    x23 += lib.einsum("ab->ab", f.vv)
    x24 += lib.einsum("ia,ba->ib", t1, x23) * -1
    del x23
    x24 += lib.einsum("ai->ia", f.vo) * -1
    x24 += lib.einsum("ia,ibja->jb", t1, v.ovov)
    x24 += lib.einsum("ijab,icab->jc", t2, v.ovvv) * -0.5
    l1new += lib.einsum("ia,abij->bj", x24, l2) * -1
    del x24
    x26 += lib.einsum("ia->ia", t1) * -2
    x26 += lib.einsum("ai,jiba->jb", l1, t2) * -2
    l1new += lib.einsum("ia,ijab->bj", x26, v.oovv) * -0.5
    del x26
    x27 = np.zeros((nvir, nvir), dtype=np.float64)
    x27 += lib.einsum("ai,ib->ab", l1, t1)
    x27 += lib.einsum("abij,ijac->bc", l2, t2) * 0.5
    l1new += lib.einsum("ab,iabc->ci", x27, v.ovvv) * -1
    del x27
    x28 += lib.einsum("ab->ab", f.vv)
    l1new += lib.einsum("ai,ab->bi", l1, x28)
    del x28
    x32 -= lib.einsum("iajb->jiab", v.ovov)
    x33 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x33 += lib.einsum("abij,ikac->kjcb", l2, x32)
    del x32
    x34 += lib.einsum("ijab->jiba", x33)
    del x33
    x34 += lib.einsum("ia,bj->ijab", f.ov, l1)
    l2new += lib.einsum("ijab->abij", x34)
    l2new -= lib.einsum("ijab->baij", x34)
    l2new -= lib.einsum("ijab->abji", x34)
    l2new += lib.einsum("ijab->baji", x34)
    del x34
    x39 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x39 += lib.einsum("ij,abkj->ikab", f.oo, l2)
    x45 += lib.einsum("ijab->ijba", x39)
    del x39
    x40 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x40 += lib.einsum("ai,jabc->ijbc", l1, v.ovvv)
    x45 -= lib.einsum("ijab->ijba", x40)
    del x40
    l2new -= lib.einsum("ijab->abij", x45)
    l2new += lib.einsum("ijab->abji", x45)
    del x45
    x46 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x46 += lib.einsum("ai,jkib->jkab", l1, v.ooov)
    x48 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x48 -= lib.einsum("ijab->jiab", x46)
    del x46
    x47 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x47 += lib.einsum("ab,caij->ijbc", f.vv, l2)
    x48 -= lib.einsum("ijab->jiab", x47)
    del x47
    l2new -= lib.einsum("ijab->abij", x48)
    l2new += lib.einsum("ijab->baij", x48)
    del x48
    x49 += lib.einsum("ijka->jika", v.ooov) * -1
    x50 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x50 += lib.einsum("ia,jkla->kjli", t1, x49) * -2
    del x49
    x50 += lib.einsum("ijkl->jilk", v.oooo)
    l2new += lib.einsum("abij,klij->balk", l2, x50) * 0.5
    del x50
    l1new += lib.einsum("ia->ai", f.ov)
    l1new -= lib.einsum("ai,jaib->bj", l1, v.ovov)
    l2new += lib.einsum("abij,abcd->dcji", l2, v.vvvv) * 0.5
    l2new += lib.einsum("ijab->baji", v.oovv)

    return {"l1new": l1new, "l2new": l2new}

def make_rdm1_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    delta_oo = np.eye(nocc)
    delta_vv = np.eye(nvir)

    # 1RDM
    x0 = np.zeros((nocc, nocc), dtype=np.float64)
    x0 += lib.einsum("abij,kjab->ik", l2, t2)
    x3 = np.zeros((nocc, nocc), dtype=np.float64)
    x3 += lib.einsum("ij->ij", x0) * 0.5
    rdm1_f_oo = np.zeros((nocc, nocc), dtype=np.float64)
    rdm1_f_oo += lib.einsum("ij->ij", x0) * -0.5
    del x0
    x1 = np.zeros((nocc, nocc), dtype=np.float64)
    x1 += lib.einsum("ai,ja->ij", l1, t1)
    x3 += lib.einsum("ij->ij", x1)
    rdm1_f_vo = np.zeros((nvir, nocc), dtype=np.float64)
    rdm1_f_vo += lib.einsum("ia,ij->aj", t1, x3) * -1
    del x3
    rdm1_f_oo -= lib.einsum("ij->ij", x1)
    del x1
    x2 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x2 += lib.einsum("ia,bajk->jkib", t1, l2)
    rdm1_f_vo += lib.einsum("ijab,ijkb->ak", t2, x2) * 0.5
    del x2
    rdm1_f_oo += lib.einsum("ij->ji", delta_oo)
    rdm1_f_ov = np.zeros((nocc, nvir), dtype=np.float64)
    rdm1_f_ov += lib.einsum("ai->ia", l1)
    rdm1_f_vo += lib.einsum("ia->ai", t1)
    rdm1_f_vo += lib.einsum("ai,jiba->bj", l1, t2)
    rdm1_f_vv = np.zeros((nvir, nvir), dtype=np.float64)
    rdm1_f_vv += lib.einsum("abij,ijca->cb", l2, t2) * -0.5
    rdm1_f_vv += lib.einsum("ai,ib->ba", l1, t1)

    rdm1_f = np.block([[rdm1_f_oo, rdm1_f_ov], [rdm1_f_vo, rdm1_f_vv]])

    return rdm1_f

def make_rdm2_f(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    delta_oo = np.eye(nocc)
    delta_vv = np.eye(nvir)

    # 2RDM
    x0 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x0 += lib.einsum("abij,klab->ijkl", l2, t2)
    x14 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x14 += lib.einsum("ijkl->jilk", x0) * -1
    x26 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x26 += lib.einsum("ijkl->jilk", x0) * -1
    x27 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x27 += lib.einsum("ijkl->jilk", x0) * -1
    rdm2_f_oooo = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    rdm2_f_oooo += lib.einsum("ijkl->lkji", x0) * 0.5
    del x0
    x1 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x1 += lib.einsum("ia,bajk->jkib", t1, l2)
    x2 = np.zeros((nocc, nocc, nocc, nocc), dtype=np.float64)
    x2 += lib.einsum("ia,jkla->jkil", t1, x1)
    x14 += lib.einsum("ijkl->ijlk", x2) * 2
    x15 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x15 += lib.einsum("ia,ijkl->jkla", t1, x14) * 0.5
    del x14
    rdm2_f_ooov = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    rdm2_f_ooov += lib.einsum("ijka->kjia", x15) * -1
    rdm2_f_oovo = np.zeros((nocc, nocc, nvir, nocc), dtype=np.float64)
    rdm2_f_oovo += lib.einsum("ijka->kjai", x15)
    del x15
    x26 += lib.einsum("ijkl->ijlk", x2) * 2.0000000000000013
    rdm2_f_oovv = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    rdm2_f_oovv += lib.einsum("ijab,ijkl->lkba", t2, x26) * -0.25
    del x26
    x27 += lib.einsum("ijkl->ijlk", x2) * 1.9999999999999987
    x28 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x28 += lib.einsum("ia,ijkl->jlka", t1, x27) * -1
    del x27
    rdm2_f_oovv += lib.einsum("ia,ijkb->jkab", t1, x28) * 0.5000000000000003
    del x28
    rdm2_f_oooo -= lib.einsum("ijkl->klji", x2)
    del x2
    x9 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x9 -= lib.einsum("ijab,kjlb->klia", t2, x1)
    x10 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x10 -= lib.einsum("ijka->jika", x9)
    x23 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x23 += lib.einsum("ia,ijkb->jkab", t1, x9)
    del x9
    x24 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x24 += lib.einsum("ijab->ijab", x23)
    del x23
    x11 = np.zeros((nocc, nvir), dtype=np.float64)
    x11 -= lib.einsum("ijab,ijkb->ka", t2, x1)
    x13 = np.zeros((nocc, nvir), dtype=np.float64)
    x13 += lib.einsum("ia->ia", x11)
    del x11
    x29 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x29 += lib.einsum("ia,jikb->jkba", t1, x1)
    x34 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x34 -= lib.einsum("ia,ijbc->jbca", t1, x29)
    rdm2_f_ovvv = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    rdm2_f_ovvv -= lib.einsum("iabc->iacb", x34)
    rdm2_f_vovv = np.zeros((nvir, nocc, nvir, nvir), dtype=np.float64)
    rdm2_f_vovv += lib.einsum("iabc->aicb", x34)
    del x34
    rdm2_f_ovov = np.zeros((nocc, nvir, nocc, nvir), dtype=np.float64)
    rdm2_f_ovov += lib.einsum("ijab->jaib", x29)
    rdm2_f_ovvo = np.zeros((nocc, nvir, nvir, nocc), dtype=np.float64)
    rdm2_f_ovvo -= lib.einsum("ijab->jabi", x29)
    rdm2_f_voov = np.zeros((nvir, nocc, nocc, nvir), dtype=np.float64)
    rdm2_f_voov -= lib.einsum("ijab->ajib", x29)
    rdm2_f_vovo = np.zeros((nvir, nocc, nvir, nocc), dtype=np.float64)
    rdm2_f_vovo += lib.einsum("ijab->ajbi", x29)
    del x29
    x33 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x33 += lib.einsum("ijab,ijkc->kcab", t2, x1)
    rdm2_f_ovvv += lib.einsum("iabc->iacb", x33) * -0.5
    rdm2_f_vovv += lib.einsum("iabc->aicb", x33) * 0.5
    del x33
    rdm2_f_ovoo = np.zeros((nocc, nvir, nocc, nocc), dtype=np.float64)
    rdm2_f_ovoo -= lib.einsum("ijka->kaji", x1)
    rdm2_f_vooo = np.zeros((nvir, nocc, nocc, nocc), dtype=np.float64)
    rdm2_f_vooo += lib.einsum("ijka->akji", x1)
    del x1
    x3 = np.zeros((nocc, nocc), dtype=np.float64)
    x3 += lib.einsum("abij,kjab->ik", l2, t2)
    x12 = np.zeros((nocc, nvir), dtype=np.float64)
    x12 += lib.einsum("ia,ij->ja", t1, x3)
    x13 += lib.einsum("ia->ia", x12)
    del x12
    rdm2_f_ooov += lib.einsum("ij,ka->jkia", delta_oo, x13) * -0.5
    rdm2_f_ooov += lib.einsum("ij,ka->kija", delta_oo, x13) * 0.5
    rdm2_f_oovo += lib.einsum("ij,ka->jkai", delta_oo, x13) * 0.5
    rdm2_f_oovo += lib.einsum("ij,ka->kiaj", delta_oo, x13) * -0.5
    rdm2_f_oovv += lib.einsum("ia,jb->ijab", t1, x13) * -0.5000000000000003
    rdm2_f_oovv += lib.einsum("ia,jb->ijba", t1, x13) * 0.5000000000000003
    rdm2_f_oovv += lib.einsum("ia,jb->jiab", t1, x13) * 0.5000000000000003
    rdm2_f_oovv += lib.einsum("ia,jb->jiba", t1, x13) * -0.5000000000000003
    del x13
    x22 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x22 += lib.einsum("ij,kiab->kjab", x3, t2)
    rdm2_f_oovv += lib.einsum("ijab->ijba", x22) * 0.5
    rdm2_f_oovv += lib.einsum("ijab->jiba", x22) * -0.5
    del x22
    rdm2_f_oooo += lib.einsum("ij,kl->iljk", delta_oo, x3) * -0.5
    rdm2_f_oooo += lib.einsum("ij,kl->ilkj", delta_oo, x3) * 0.5
    rdm2_f_oooo += lib.einsum("ij,kl->ljik", delta_oo, x3) * 0.5
    rdm2_f_oooo += lib.einsum("ij,kl->ljki", delta_oo, x3) * -0.5
    rdm2_f_ooov += lib.einsum("ia,jk->ikja", t1, x3) * 0.5
    rdm2_f_ooov += lib.einsum("ia,jk->kija", t1, x3) * -0.5
    rdm2_f_oovo += lib.einsum("ia,jk->ikaj", t1, x3) * -0.5
    rdm2_f_oovo += lib.einsum("ia,jk->kiaj", t1, x3) * 0.5
    del x3
    x4 = np.zeros((nocc, nocc), dtype=np.float64)
    x4 += lib.einsum("ai,ja->ij", l1, t1)
    x7 = np.zeros((nocc, nvir), dtype=np.float64)
    x7 += lib.einsum("ia,ij->ja", t1, x4)
    x8 = np.zeros((nocc, nvir), dtype=np.float64)
    x8 -= lib.einsum("ia->ia", x7)
    del x7
    x10 += lib.einsum("ia,jk->ijka", t1, x4)
    x25 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x25 += lib.einsum("ij,kiab->jkab", x4, t2)
    rdm2_f_oovv -= lib.einsum("ijab->ijba", x25)
    rdm2_f_oovv += lib.einsum("ijab->jiba", x25)
    del x25
    rdm2_f_oooo -= lib.einsum("ij,kl->jlik", delta_oo, x4)
    rdm2_f_oooo += lib.einsum("ij,kl->ljik", delta_oo, x4)
    rdm2_f_oooo += lib.einsum("ij,kl->jlki", delta_oo, x4)
    rdm2_f_oooo -= lib.einsum("ij,kl->ljki", delta_oo, x4)
    del x4
    x5 = np.zeros((nocc, nocc, nocc, nvir), dtype=np.float64)
    x5 += lib.einsum("ai,jkba->ijkb", l1, t2)
    x16 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x16 += lib.einsum("ia,ijkb->jkab", t1, x5)
    x19 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x19 += lib.einsum("ijab->ijab", x16)
    del x16
    rdm2_f_ooov -= lib.einsum("ijka->kjia", x5)
    rdm2_f_oovo += lib.einsum("ijka->kjai", x5)
    del x5
    x6 = np.zeros((nocc, nvir), dtype=np.float64)
    x6 += lib.einsum("ai,jiba->jb", l1, t2)
    x8 += lib.einsum("ia->ia", x6)
    del x6
    x24 -= lib.einsum("ia,jb->ijab", t1, x8)
    rdm2_f_oovv -= lib.einsum("ijab->ijab", x24)
    rdm2_f_oovv += lib.einsum("ijab->ijba", x24)
    rdm2_f_oovv += lib.einsum("ijab->jiab", x24)
    rdm2_f_oovv -= lib.einsum("ijab->jiba", x24)
    del x24
    rdm2_f_ooov += lib.einsum("ij,ka->jkia", delta_oo, x8)
    rdm2_f_ooov -= lib.einsum("ij,ka->kija", delta_oo, x8)
    rdm2_f_oovo -= lib.einsum("ij,ka->jkai", delta_oo, x8)
    rdm2_f_oovo += lib.einsum("ij,ka->kiaj", delta_oo, x8)
    del x8
    x10 += lib.einsum("ij,ka->jika", delta_oo, t1)
    rdm2_f_ooov += lib.einsum("ijka->ikja", x10)
    rdm2_f_ooov -= lib.einsum("ijka->kija", x10)
    rdm2_f_oovo -= lib.einsum("ijka->ikaj", x10)
    rdm2_f_oovo += lib.einsum("ijka->kiaj", x10)
    del x10
    x17 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x17 -= lib.einsum("abij,kjca->ikbc", l2, t2)
    x18 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x18 += lib.einsum("ijab,jkbc->ikac", t2, x17)
    x19 += lib.einsum("ijab->ijab", x18)
    del x18
    rdm2_f_oovv += lib.einsum("ijab->ijab", x19)
    rdm2_f_oovv -= lib.einsum("ijab->ijba", x19)
    del x19
    x35 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x35 += lib.einsum("ia,ijbc->jbac", t1, x17)
    x36 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x36 -= lib.einsum("iabc->iabc", x35)
    del x35
    rdm2_f_ovov -= lib.einsum("ijab->jaib", x17)
    rdm2_f_ovvo += lib.einsum("ijab->jabi", x17)
    rdm2_f_voov += lib.einsum("ijab->ajib", x17)
    rdm2_f_vovo -= lib.einsum("ijab->ajbi", x17)
    del x17
    x20 = np.zeros((nvir, nvir), dtype=np.float64)
    x20 -= lib.einsum("abij,ijca->bc", l2, t2)
    x21 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x21 += lib.einsum("ab,ijca->ijcb", x20, t2)
    rdm2_f_oovv += lib.einsum("ijab->ijab", x21) * -0.5
    rdm2_f_oovv += lib.einsum("ijab->ijba", x21) * 0.5
    del x21
    x31 = np.zeros((nvir, nvir), dtype=np.float64)
    x31 += lib.einsum("ab->ab", x20)
    rdm2_f_ovvv += lib.einsum("ia,bc->ibac", t1, x20) * 0.5
    rdm2_f_ovvv += lib.einsum("ia,bc->ibca", t1, x20) * -0.5
    rdm2_f_vovv += lib.einsum("ia,bc->biac", t1, x20) * -0.5
    rdm2_f_vovv += lib.einsum("ia,bc->bica", t1, x20) * 0.5
    del x20
    x30 = np.zeros((nvir, nvir), dtype=np.float64)
    x30 += lib.einsum("ai,ib->ab", l1, t1)
    x31 += lib.einsum("ab->ab", x30) * 2
    rdm2_f_ovov += lib.einsum("ij,ab->jaib", delta_oo, x31) * 0.5
    rdm2_f_ovvo += lib.einsum("ij,ab->jabi", delta_oo, x31) * -0.5
    rdm2_f_voov += lib.einsum("ij,ab->ajib", delta_oo, x31) * -0.5
    rdm2_f_vovo += lib.einsum("ij,ab->ajbi", delta_oo, x31) * 0.5
    del x31
    x36 += lib.einsum("ia,bc->ibac", t1, x30)
    del x30
    rdm2_f_ovvv += lib.einsum("iabc->iabc", x36)
    rdm2_f_ovvv -= lib.einsum("iabc->iacb", x36)
    rdm2_f_vovv -= lib.einsum("iabc->aibc", x36)
    rdm2_f_vovv += lib.einsum("iabc->aicb", x36)
    del x36
    x32 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x32 += lib.einsum("ai,jibc->jabc", l1, t2)
    rdm2_f_ovvv -= lib.einsum("iabc->iacb", x32)
    rdm2_f_vovv += lib.einsum("iabc->aicb", x32)
    del x32
    x37 = np.zeros((nocc, nvir, nvir, nvir), dtype=np.float64)
    x37 += lib.einsum("ia,bcji->jbca", t1, l2)
    rdm2_f_vvov = np.zeros((nvir, nvir, nocc, nvir), dtype=np.float64)
    rdm2_f_vvov -= lib.einsum("iabc->baic", x37)
    rdm2_f_vvvo = np.zeros((nvir, nvir, nvir, nocc), dtype=np.float64)
    rdm2_f_vvvo += lib.einsum("iabc->baci", x37)
    rdm2_f_vvvv = np.zeros((nvir, nvir, nvir, nvir), dtype=np.float64)
    rdm2_f_vvvv += lib.einsum("ia,ibcd->bcad", t1, x37)
    del x37
    rdm2_f_oooo += lib.einsum("ij,kl->jlik", delta_oo, delta_oo)
    rdm2_f_oooo -= lib.einsum("ij,kl->lijk", delta_oo, delta_oo)
    rdm2_f_ovoo += lib.einsum("ij,ak->jaik", delta_oo, l1)
    rdm2_f_ovoo -= lib.einsum("ij,ak->jaki", delta_oo, l1)
    rdm2_f_vooo -= lib.einsum("ij,ak->ajik", delta_oo, l1)
    rdm2_f_vooo += lib.einsum("ij,ak->ajki", delta_oo, l1)
    rdm2_f_oovv += lib.einsum("ijab->jiba", t2)
    rdm2_f_oovv -= lib.einsum("ia,jb->ijba", t1, t1)
    rdm2_f_oovv += lib.einsum("ia,jb->ijab", t1, t1)
    rdm2_f_ovov -= lib.einsum("ai,jb->jaib", l1, t1)
    rdm2_f_ovvo += lib.einsum("ai,jb->jabi", l1, t1)
    rdm2_f_voov += lib.einsum("ai,jb->ajib", l1, t1)
    rdm2_f_vovo -= lib.einsum("ai,jb->ajbi", l1, t1)
    rdm2_f_vvoo = np.zeros((nvir, nvir, nocc, nocc), dtype=np.float64)
    rdm2_f_vvoo += lib.einsum("abij->baji", l2)
    rdm2_f_vvvv += lib.einsum("abij,ijcd->badc", l2, t2) * 0.5

    rdm2_f = common.pack_2e(rdm2_f_oooo, rdm2_f_ooov, rdm2_f_oovo, rdm2_f_ovoo, rdm2_f_vooo, rdm2_f_oovv, rdm2_f_ovov, rdm2_f_ovvo, rdm2_f_voov, rdm2_f_vovo, rdm2_f_vvoo, rdm2_f_ovvv, rdm2_f_vovv, rdm2_f_vvov, rdm2_f_vvvo, rdm2_f_vvvv)

    rdm2_f = rdm2_f.transpose(0, 2, 1, 3)

    return rdm2_f
