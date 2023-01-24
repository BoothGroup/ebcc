# Code generated by qwick.

import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # energy
    e_cc = 0.0
    e_cc += np.einsum("ia,ia->", f.ov, t1)
    e_cc += np.einsum("ijab,ijab->", t2, v.oovv) * 0.25
    e_cc += np.einsum("ia,jb,ijab->", t1, t1, v.oovv) * 0.5

    return e_cc

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, t3=None, **kwargs):
    # T1 amplitudes
    t1new = np.zeros((nocc, nvir), dtype=np.float64)
    t1new += np.einsum("ia->ia", f.ov)
    t1new += np.einsum("ab,ib->ia", f.vv, t1)
    t1new += np.einsum("ij,ja->ia", f.oo, t1) * -1.0
    t1new += np.einsum("jb,ijab->ia", f.ov, t2)
    t1new += np.einsum("jb,ibja->ia", t1, v.ovov) * -1.0
    t1new += np.einsum("ijbc,jabc->ia", t2, v.ovvv) * -0.5
    t1new += np.einsum("jkab,jkib->ia", t2, v.ooov) * -0.5
    t1new += np.einsum("jkbc,ijkabc->ia", v.oovv, t3) * 0.25
    t1new += np.einsum("jb,ib,ja->ia", f.ov, t1, t1) * -1.0
    t1new += np.einsum("ib,jc,jabc->ia", t1, t1, v.ovvv) * -1.0
    t1new += np.einsum("jb,ka,jkib->ia", t1, t1, v.ooov)
    t1new += np.einsum("ib,jkac,jkbc->ia", t1, t2, v.oovv) * -0.5
    t1new += np.einsum("ja,ikbc,jkbc->ia", t1, t2, v.oovv) * -0.5
    t1new += np.einsum("jb,ikac,jkbc->ia", t1, t2, v.oovv)
    t1new += np.einsum("ib,jc,ka,jkbc->ia", t1, t1, t1, v.oovv)

    # T2 amplitudes
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2new += np.einsum("ijab->ijab", v.oovv)
    t2new += np.einsum("bc,ijac->ijab", f.vv, t2)
    t2new += np.einsum("ac,ijbc->ijab", f.vv, t2) * -1.0
    t2new += np.einsum("jk,ikab->ijab", f.oo, t2) * -1.0
    t2new += np.einsum("ik,jkab->ijab", f.oo, t2)
    t2new += np.einsum("ic,jcab->ijab", t1, v.ovvv) * -1.0
    t2new += np.einsum("jc,icab->ijab", t1, v.ovvv)
    t2new += np.einsum("ka,ijkb->ijab", t1, v.ooov) * -1.0
    t2new += np.einsum("kb,ijka->ijab", t1, v.ooov)
    t2new += np.einsum("kc,ijkabc->ijab", f.ov, t3)
    t2new += np.einsum("ijcd,abcd->ijab", t2, v.vvvv) * 0.5
    t2new += np.einsum("ikac,jckb->ijab", t2, v.ovov) * -1.0
    t2new += np.einsum("ikbc,jcka->ijab", t2, v.ovov)
    t2new += np.einsum("jkac,ickb->ijab", t2, v.ovov)
    t2new += np.einsum("jkbc,icka->ijab", t2, v.ovov) * -1.0
    t2new += np.einsum("klab,ijkl->ijab", t2, v.oooo) * 0.5
    t2new += np.einsum("kbcd,ijkacd->ijab", v.ovvv, t3) * -0.5
    t2new += np.einsum("kacd,ijkbcd->ijab", v.ovvv, t3) * 0.5
    t2new += np.einsum("kljc,iklabc->ijab", v.ooov, t3) * -0.5
    t2new += np.einsum("klic,jklabc->ijab", v.ooov, t3) * 0.5
    t2new += np.einsum("kc,ic,jkab->ijab", f.ov, t1, t2)
    t2new += np.einsum("kc,jc,ikab->ijab", f.ov, t1, t2) * -1.0
    t2new += np.einsum("kc,ka,ijbc->ijab", f.ov, t1, t2)
    t2new += np.einsum("kc,kb,ijac->ijab", f.ov, t1, t2) * -1.0
    t2new += np.einsum("ic,jd,abcd->ijab", t1, t1, v.vvvv)
    t2new += np.einsum("ic,ka,jckb->ijab", t1, t1, v.ovov)
    t2new += np.einsum("ic,kb,jcka->ijab", t1, t1, v.ovov) * -1.0
    t2new += np.einsum("jc,ka,ickb->ijab", t1, t1, v.ovov) * -1.0
    t2new += np.einsum("jc,kb,icka->ijab", t1, t1, v.ovov)
    t2new += np.einsum("ka,lb,ijkl->ijab", t1, t1, v.oooo)
    t2new += np.einsum("ic,jkad,kbcd->ijab", t1, t2, v.ovvv)
    t2new += np.einsum("ic,jkbd,kacd->ijab", t1, t2, v.ovvv) * -1.0
    t2new += np.einsum("ic,klab,kljc->ijab", t1, t2, v.ooov) * -0.5
    t2new += np.einsum("jc,ikad,kbcd->ijab", t1, t2, v.ovvv) * -1.0
    t2new += np.einsum("jc,ikbd,kacd->ijab", t1, t2, v.ovvv)
    t2new += np.einsum("jc,klab,klic->ijab", t1, t2, v.ooov) * 0.5
    t2new += np.einsum("ka,ijcd,kbcd->ijab", t1, t2, v.ovvv) * -0.5
    t2new += np.einsum("ka,ilbc,kljc->ijab", t1, t2, v.ooov)
    t2new += np.einsum("ka,jlbc,klic->ijab", t1, t2, v.ooov) * -1.0
    t2new += np.einsum("kb,ijcd,kacd->ijab", t1, t2, v.ovvv) * 0.5
    t2new += np.einsum("kb,ilac,kljc->ijab", t1, t2, v.ooov) * -1.0
    t2new += np.einsum("kb,jlac,klic->ijab", t1, t2, v.ooov)
    t2new += np.einsum("kc,ijad,kbcd->ijab", t1, t2, v.ovvv)
    t2new += np.einsum("kc,ijbd,kacd->ijab", t1, t2, v.ovvv) * -1.0
    t2new += np.einsum("kc,ilab,kljc->ijab", t1, t2, v.ooov)
    t2new += np.einsum("kc,jlab,klic->ijab", t1, t2, v.ooov) * -1.0
    t2new += np.einsum("ic,klcd,jklabd->ijab", t1, v.oovv, t3) * 0.5
    t2new += np.einsum("jc,klcd,iklabd->ijab", t1, v.oovv, t3) * -0.5
    t2new += np.einsum("ka,klcd,ijlbcd->ijab", t1, v.oovv, t3) * 0.5
    t2new += np.einsum("kb,klcd,ijlacd->ijab", t1, v.oovv, t3) * -0.5
    t2new += np.einsum("kc,klcd,ijlabd->ijab", t1, v.oovv, t3)
    t2new += np.einsum("ijac,klbd,klcd->ijab", t2, t2, v.oovv) * -0.5
    t2new += np.einsum("ijbc,klad,klcd->ijab", t2, t2, v.oovv) * 0.5
    t2new += np.einsum("ijcd,klab,klcd->ijab", t2, t2, v.oovv) * 0.25
    t2new += np.einsum("ikab,jlcd,klcd->ijab", t2, t2, v.oovv) * -0.5
    t2new += np.einsum("ikac,jlbd,klcd->ijab", t2, t2, v.oovv)
    t2new += np.einsum("ikbc,jlad,klcd->ijab", t2, t2, v.oovv) * -1.0
    t2new += np.einsum("ikcd,jlab,klcd->ijab", t2, t2, v.oovv) * -0.5
    t2new += np.einsum("ic,jd,ka,kbcd->ijab", t1, t1, t1, v.ovvv) * -1.0
    t2new += np.einsum("ic,jd,kb,kacd->ijab", t1, t1, t1, v.ovvv)
    t2new += np.einsum("ic,ka,lb,kljc->ijab", t1, t1, t1, v.ooov) * -1.0
    t2new += np.einsum("jc,ka,lb,klic->ijab", t1, t1, t1, v.ooov)
    t2new += np.einsum("ic,jd,klab,klcd->ijab", t1, t1, t2, v.oovv) * 0.5
    t2new += np.einsum("ic,ka,jlbd,klcd->ijab", t1, t1, t2, v.oovv) * -1.0
    t2new += np.einsum("ic,kb,jlad,klcd->ijab", t1, t1, t2, v.oovv)
    t2new += np.einsum("ic,kd,jlab,klcd->ijab", t1, t1, t2, v.oovv) * -1.0
    t2new += np.einsum("jc,ka,ilbd,klcd->ijab", t1, t1, t2, v.oovv)
    t2new += np.einsum("jc,kb,ilad,klcd->ijab", t1, t1, t2, v.oovv) * -1.0
    t2new += np.einsum("jc,kd,ilab,klcd->ijab", t1, t1, t2, v.oovv)
    t2new += np.einsum("ka,lb,ijcd,klcd->ijab", t1, t1, t2, v.oovv) * 0.5
    t2new += np.einsum("kc,la,ijbd,klcd->ijab", t1, t1, t2, v.oovv)
    t2new += np.einsum("kc,lb,ijad,klcd->ijab", t1, t1, t2, v.oovv) * -1.0
    t2new += np.einsum("ic,jd,ka,lb,klcd->ijab", t1, t1, t1, t1, v.oovv)

    # T3 amplitudes
    t3new = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    t3new += np.einsum("cd,ijkabd->ijkabc", f.vv, t3)
    t3new += np.einsum("bd,ijkacd->ijkabc", f.vv, t3) * -1.0
    t3new += np.einsum("ad,ijkbcd->ijkabc", f.vv, t3)
    t3new += np.einsum("kl,ijlabc->ijkabc", f.oo, t3) * -1.0
    t3new += np.einsum("jl,iklabc->ijkabc", f.oo, t3)
    t3new += np.einsum("il,jklabc->ijkabc", f.oo, t3) * -1.0
    t3new += np.einsum("ijad,kdbc->ijkabc", t2, v.ovvv) * -1.0
    t3new += np.einsum("ijbd,kdac->ijkabc", t2, v.ovvv)
    t3new += np.einsum("ijcd,kdab->ijkabc", t2, v.ovvv) * -1.0
    t3new += np.einsum("ikad,jdbc->ijkabc", t2, v.ovvv)
    t3new += np.einsum("ikbd,jdac->ijkabc", t2, v.ovvv) * -1.0
    t3new += np.einsum("ikcd,jdab->ijkabc", t2, v.ovvv)
    t3new += np.einsum("ilab,jklc->ijkabc", t2, v.ooov) * -1.0
    t3new += np.einsum("ilac,jklb->ijkabc", t2, v.ooov)
    t3new += np.einsum("ilbc,jkla->ijkabc", t2, v.ooov) * -1.0
    t3new += np.einsum("jkad,idbc->ijkabc", t2, v.ovvv) * -1.0
    t3new += np.einsum("jkbd,idac->ijkabc", t2, v.ovvv)
    t3new += np.einsum("jkcd,idab->ijkabc", t2, v.ovvv) * -1.0
    t3new += np.einsum("jlab,iklc->ijkabc", t2, v.ooov)
    t3new += np.einsum("jlac,iklb->ijkabc", t2, v.ooov) * -1.0
    t3new += np.einsum("jlbc,ikla->ijkabc", t2, v.ooov)
    t3new += np.einsum("klab,ijlc->ijkabc", t2, v.ooov) * -1.0
    t3new += np.einsum("klac,ijlb->ijkabc", t2, v.ooov)
    t3new += np.einsum("klbc,ijla->ijkabc", t2, v.ooov) * -1.0
    t3new += np.einsum("bcde,ijkade->ijkabc", v.vvvv, t3) * 0.5
    t3new += np.einsum("acde,ijkbde->ijkabc", v.vvvv, t3) * -0.5
    t3new += np.einsum("abde,ijkcde->ijkabc", v.vvvv, t3) * 0.5
    t3new += np.einsum("kdlc,ijlabd->ijkabc", v.ovov, t3) * -1.0
    t3new += np.einsum("kdlb,ijlacd->ijkabc", v.ovov, t3)
    t3new += np.einsum("kdla,ijlbcd->ijkabc", v.ovov, t3) * -1.0
    t3new += np.einsum("jdlc,iklabd->ijkabc", v.ovov, t3)
    t3new += np.einsum("jdlb,iklacd->ijkabc", v.ovov, t3) * -1.0
    t3new += np.einsum("jdla,iklbcd->ijkabc", v.ovov, t3)
    t3new += np.einsum("jklm,ilmabc->ijkabc", v.oooo, t3) * 0.5
    t3new += np.einsum("idlc,jklabd->ijkabc", v.ovov, t3) * -1.0
    t3new += np.einsum("idlb,jklacd->ijkabc", v.ovov, t3)
    t3new += np.einsum("idla,jklbcd->ijkabc", v.ovov, t3) * -1.0
    t3new += np.einsum("iklm,jlmabc->ijkabc", v.oooo, t3) * -0.5
    t3new += np.einsum("ijlm,klmabc->ijkabc", v.oooo, t3) * 0.5
    t3new += np.einsum("ld,id,jklabc->ijkabc", f.ov, t1, t3) * -1.0
    t3new += np.einsum("ld,jd,iklabc->ijkabc", f.ov, t1, t3)
    t3new += np.einsum("ld,kd,ijlabc->ijkabc", f.ov, t1, t3) * -1.0
    t3new += np.einsum("ld,la,ijkbcd->ijkabc", f.ov, t1, t3) * -1.0
    t3new += np.einsum("ld,lb,ijkacd->ijkabc", f.ov, t1, t3)
    t3new += np.einsum("ld,lc,ijkabd->ijkabc", f.ov, t1, t3) * -1.0
    t3new += np.einsum("ld,ijad,klbc->ijkabc", f.ov, t2, t2)
    t3new += np.einsum("ld,ijbd,klac->ijkabc", f.ov, t2, t2) * -1.0
    t3new += np.einsum("ld,ijcd,klab->ijkabc", f.ov, t2, t2)
    t3new += np.einsum("ld,ikad,jlbc->ijkabc", f.ov, t2, t2) * -1.0
    t3new += np.einsum("ld,ikbd,jlac->ijkabc", f.ov, t2, t2)
    t3new += np.einsum("ld,ikcd,jlab->ijkabc", f.ov, t2, t2) * -1.0
    t3new += np.einsum("ld,ilab,jkcd->ijkabc", f.ov, t2, t2)
    t3new += np.einsum("ld,ilac,jkbd->ijkabc", f.ov, t2, t2) * -1.0
    t3new += np.einsum("ld,ilbc,jkad->ijkabc", f.ov, t2, t2)
    t3new += np.einsum("id,jkae,bcde->ijkabc", t1, t2, v.vvvv) * -1.0
    t3new += np.einsum("id,jkbe,acde->ijkabc", t1, t2, v.vvvv)
    t3new += np.einsum("id,jkce,abde->ijkabc", t1, t2, v.vvvv) * -1.0
    t3new += np.einsum("id,jlab,kdlc->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("id,jlac,kdlb->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("id,jlbc,kdla->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("id,klab,jdlc->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("id,klac,jdlb->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("id,klbc,jdla->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("jd,ikae,bcde->ijkabc", t1, t2, v.vvvv)
    t3new += np.einsum("jd,ikbe,acde->ijkabc", t1, t2, v.vvvv) * -1.0
    t3new += np.einsum("jd,ikce,abde->ijkabc", t1, t2, v.vvvv)
    t3new += np.einsum("jd,ilab,kdlc->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("jd,ilac,kdlb->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("jd,ilbc,kdla->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("jd,klab,idlc->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("jd,klac,idlb->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("jd,klbc,idla->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("kd,ijae,bcde->ijkabc", t1, t2, v.vvvv) * -1.0
    t3new += np.einsum("kd,ijbe,acde->ijkabc", t1, t2, v.vvvv)
    t3new += np.einsum("kd,ijce,abde->ijkabc", t1, t2, v.vvvv) * -1.0
    t3new += np.einsum("kd,ilab,jdlc->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("kd,ilac,jdlb->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("kd,ilbc,jdla->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("kd,jlab,idlc->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("kd,jlac,idlb->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("kd,jlbc,idla->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("la,ijbd,kdlc->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("la,ijcd,kdlb->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("la,ikbd,jdlc->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("la,ikcd,jdlb->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("la,imbc,jklm->ijkabc", t1, t2, v.oooo) * -1.0
    t3new += np.einsum("la,jkbd,idlc->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("la,jkcd,idlb->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("la,jmbc,iklm->ijkabc", t1, t2, v.oooo)
    t3new += np.einsum("la,kmbc,ijlm->ijkabc", t1, t2, v.oooo) * -1.0
    t3new += np.einsum("lb,ijad,kdlc->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("lb,ijcd,kdla->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("lb,ikad,jdlc->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("lb,ikcd,jdla->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("lb,imac,jklm->ijkabc", t1, t2, v.oooo)
    t3new += np.einsum("lb,jkad,idlc->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("lb,jkcd,idla->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("lb,jmac,iklm->ijkabc", t1, t2, v.oooo) * -1.0
    t3new += np.einsum("lb,kmac,ijlm->ijkabc", t1, t2, v.oooo)
    t3new += np.einsum("lc,ijad,kdlb->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("lc,ijbd,kdla->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("lc,ikad,jdlb->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("lc,ikbd,jdla->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("lc,imab,jklm->ijkabc", t1, t2, v.oooo) * -1.0
    t3new += np.einsum("lc,jkad,idlb->ijkabc", t1, t2, v.ovov) * -1.0
    t3new += np.einsum("lc,jkbd,idla->ijkabc", t1, t2, v.ovov)
    t3new += np.einsum("lc,jmab,iklm->ijkabc", t1, t2, v.oooo)
    t3new += np.einsum("lc,kmab,ijlm->ijkabc", t1, t2, v.oooo) * -1.0
    t3new += np.einsum("id,lcde,jklabe->ijkabc", t1, v.ovvv, t3) * -1.0
    t3new += np.einsum("id,lbde,jklace->ijkabc", t1, v.ovvv, t3)
    t3new += np.einsum("id,lade,jklbce->ijkabc", t1, v.ovvv, t3) * -1.0
    t3new += np.einsum("id,lmkd,jlmabc->ijkabc", t1, v.ooov, t3) * 0.5
    t3new += np.einsum("id,lmjd,klmabc->ijkabc", t1, v.ooov, t3) * -0.5
    t3new += np.einsum("jd,lcde,iklabe->ijkabc", t1, v.ovvv, t3)
    t3new += np.einsum("jd,lbde,iklace->ijkabc", t1, v.ovvv, t3) * -1.0
    t3new += np.einsum("jd,lade,iklbce->ijkabc", t1, v.ovvv, t3)
    t3new += np.einsum("jd,lmkd,ilmabc->ijkabc", t1, v.ooov, t3) * -0.5
    t3new += np.einsum("jd,lmid,klmabc->ijkabc", t1, v.ooov, t3) * 0.5
    t3new += np.einsum("kd,lcde,ijlabe->ijkabc", t1, v.ovvv, t3) * -1.0
    t3new += np.einsum("kd,lbde,ijlace->ijkabc", t1, v.ovvv, t3)
    t3new += np.einsum("kd,lade,ijlbce->ijkabc", t1, v.ovvv, t3) * -1.0
    t3new += np.einsum("kd,lmjd,ilmabc->ijkabc", t1, v.ooov, t3) * 0.5
    t3new += np.einsum("kd,lmid,jlmabc->ijkabc", t1, v.ooov, t3) * -0.5
    t3new += np.einsum("la,lcde,ijkbde->ijkabc", t1, v.ovvv, t3) * 0.5
    t3new += np.einsum("la,lbde,ijkcde->ijkabc", t1, v.ovvv, t3) * -0.5
    t3new += np.einsum("la,lmkd,ijmbcd->ijkabc", t1, v.ooov, t3) * -1.0
    t3new += np.einsum("la,lmjd,ikmbcd->ijkabc", t1, v.ooov, t3)
    t3new += np.einsum("la,lmid,jkmbcd->ijkabc", t1, v.ooov, t3) * -1.0
    t3new += np.einsum("lb,lcde,ijkade->ijkabc", t1, v.ovvv, t3) * -0.5
    t3new += np.einsum("lb,lade,ijkcde->ijkabc", t1, v.ovvv, t3) * 0.5
    t3new += np.einsum("lb,lmkd,ijmacd->ijkabc", t1, v.ooov, t3)
    t3new += np.einsum("lb,lmjd,ikmacd->ijkabc", t1, v.ooov, t3) * -1.0
    t3new += np.einsum("lb,lmid,jkmacd->ijkabc", t1, v.ooov, t3)
    t3new += np.einsum("lc,lbde,ijkade->ijkabc", t1, v.ovvv, t3) * 0.5
    t3new += np.einsum("lc,lade,ijkbde->ijkabc", t1, v.ovvv, t3) * -0.5
    t3new += np.einsum("lc,lmkd,ijmabd->ijkabc", t1, v.ooov, t3) * -1.0
    t3new += np.einsum("lc,lmjd,ikmabd->ijkabc", t1, v.ooov, t3)
    t3new += np.einsum("lc,lmid,jkmabd->ijkabc", t1, v.ooov, t3) * -1.0
    t3new += np.einsum("ld,lcde,ijkabe->ijkabc", t1, v.ovvv, t3)
    t3new += np.einsum("ld,lbde,ijkace->ijkabc", t1, v.ovvv, t3) * -1.0
    t3new += np.einsum("ld,lade,ijkbce->ijkabc", t1, v.ovvv, t3)
    t3new += np.einsum("ld,lmkd,ijmabc->ijkabc", t1, v.ooov, t3)
    t3new += np.einsum("ld,lmjd,ikmabc->ijkabc", t1, v.ooov, t3) * -1.0
    t3new += np.einsum("ld,lmid,jkmabc->ijkabc", t1, v.ooov, t3)
    t3new += np.einsum("ijad,klbe,lcde->ijkabc", t2, t2, v.ovvv)
    t3new += np.einsum("ijad,klce,lbde->ijkabc", t2, t2, v.ovvv) * -1.0
    t3new += np.einsum("ijad,lmbc,lmkd->ijkabc", t2, t2, v.ooov) * -0.5
    t3new += np.einsum("ijbd,klae,lcde->ijkabc", t2, t2, v.ovvv) * -1.0
    t3new += np.einsum("ijbd,klce,lade->ijkabc", t2, t2, v.ovvv)
    t3new += np.einsum("ijbd,lmac,lmkd->ijkabc", t2, t2, v.ooov) * 0.5
    t3new += np.einsum("ijcd,klae,lbde->ijkabc", t2, t2, v.ovvv)
    t3new += np.einsum("ijcd,klbe,lade->ijkabc", t2, t2, v.ovvv) * -1.0
    t3new += np.einsum("ijcd,lmab,lmkd->ijkabc", t2, t2, v.ooov) * -0.5
    t3new += np.einsum("ijde,klab,lcde->ijkabc", t2, t2, v.ovvv) * -0.5
    t3new += np.einsum("ijde,klac,lbde->ijkabc", t2, t2, v.ovvv) * 0.5
    t3new += np.einsum("ijde,klbc,lade->ijkabc", t2, t2, v.ovvv) * -0.5
    t3new += np.einsum("ikad,jlbe,lcde->ijkabc", t2, t2, v.ovvv) * -1.0
    t3new += np.einsum("ikad,jlce,lbde->ijkabc", t2, t2, v.ovvv)
    t3new += np.einsum("ikad,lmbc,lmjd->ijkabc", t2, t2, v.ooov) * 0.5
    t3new += np.einsum("ikbd,jlae,lcde->ijkabc", t2, t2, v.ovvv)
    t3new += np.einsum("ikbd,jlce,lade->ijkabc", t2, t2, v.ovvv) * -1.0
    t3new += np.einsum("ikbd,lmac,lmjd->ijkabc", t2, t2, v.ooov) * -0.5
    t3new += np.einsum("ikcd,jlae,lbde->ijkabc", t2, t2, v.ovvv) * -1.0
    t3new += np.einsum("ikcd,jlbe,lade->ijkabc", t2, t2, v.ovvv)
    t3new += np.einsum("ikcd,lmab,lmjd->ijkabc", t2, t2, v.ooov) * 0.5
    t3new += np.einsum("ikde,jlab,lcde->ijkabc", t2, t2, v.ovvv) * 0.5
    t3new += np.einsum("ikde,jlac,lbde->ijkabc", t2, t2, v.ovvv) * -0.5
    t3new += np.einsum("ikde,jlbc,lade->ijkabc", t2, t2, v.ovvv) * 0.5
    t3new += np.einsum("ilab,jkde,lcde->ijkabc", t2, t2, v.ovvv) * -0.5
    t3new += np.einsum("ilab,jmcd,lmkd->ijkabc", t2, t2, v.ooov)
    t3new += np.einsum("ilab,kmcd,lmjd->ijkabc", t2, t2, v.ooov) * -1.0
    t3new += np.einsum("ilac,jkde,lbde->ijkabc", t2, t2, v.ovvv) * 0.5
    t3new += np.einsum("ilac,jmbd,lmkd->ijkabc", t2, t2, v.ooov) * -1.0
    t3new += np.einsum("ilac,kmbd,lmjd->ijkabc", t2, t2, v.ooov)
    t3new += np.einsum("ilad,jkbe,lcde->ijkabc", t2, t2, v.ovvv)
    t3new += np.einsum("ilad,jkce,lbde->ijkabc", t2, t2, v.ovvv) * -1.0
    t3new += np.einsum("ilad,jmbc,lmkd->ijkabc", t2, t2, v.ooov)
    t3new += np.einsum("ilad,kmbc,lmjd->ijkabc", t2, t2, v.ooov) * -1.0
    t3new += np.einsum("ilbc,jkde,lade->ijkabc", t2, t2, v.ovvv) * -0.5
    t3new += np.einsum("ilbc,jmad,lmkd->ijkabc", t2, t2, v.ooov)
    t3new += np.einsum("ilbc,kmad,lmjd->ijkabc", t2, t2, v.ooov) * -1.0
    t3new += np.einsum("ilbd,jkae,lcde->ijkabc", t2, t2, v.ovvv) * -1.0
    t3new += np.einsum("ilbd,jkce,lade->ijkabc", t2, t2, v.ovvv)
    t3new += np.einsum("ilbd,jmac,lmkd->ijkabc", t2, t2, v.ooov) * -1.0
    t3new += np.einsum("ilbd,kmac,lmjd->ijkabc", t2, t2, v.ooov)
    t3new += np.einsum("ilcd,jkae,lbde->ijkabc", t2, t2, v.ovvv)
    t3new += np.einsum("ilcd,jkbe,lade->ijkabc", t2, t2, v.ovvv) * -1.0
    t3new += np.einsum("ilcd,jmab,lmkd->ijkabc", t2, t2, v.ooov)
    t3new += np.einsum("ilcd,kmab,lmjd->ijkabc", t2, t2, v.ooov) * -1.0
    t3new += np.einsum("jkad,lmbc,lmid->ijkabc", t2, t2, v.ooov) * -0.5
    t3new += np.einsum("jkbd,lmac,lmid->ijkabc", t2, t2, v.ooov) * 0.5
    t3new += np.einsum("jkcd,lmab,lmid->ijkabc", t2, t2, v.ooov) * -0.5
    t3new += np.einsum("jlab,kmcd,lmid->ijkabc", t2, t2, v.ooov)
    t3new += np.einsum("jlac,kmbd,lmid->ijkabc", t2, t2, v.ooov) * -1.0
    t3new += np.einsum("jlad,kmbc,lmid->ijkabc", t2, t2, v.ooov)
    t3new += np.einsum("jlbc,kmad,lmid->ijkabc", t2, t2, v.ooov)
    t3new += np.einsum("jlbd,kmac,lmid->ijkabc", t2, t2, v.ooov) * -1.0
    t3new += np.einsum("jlcd,kmab,lmid->ijkabc", t2, t2, v.ooov)
    t3new += np.einsum("ijad,lmde,klmbce->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("ijbd,lmde,klmace->ijkabc", t2, v.oovv, t3) * -0.5
    t3new += np.einsum("ijcd,lmde,klmabe->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("ijde,lmde,klmabc->ijkabc", t2, v.oovv, t3) * 0.25
    t3new += np.einsum("ikad,lmde,jlmbce->ijkabc", t2, v.oovv, t3) * -0.5
    t3new += np.einsum("ikbd,lmde,jlmace->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("ikcd,lmde,jlmabe->ijkabc", t2, v.oovv, t3) * -0.5
    t3new += np.einsum("ikde,lmde,jlmabc->ijkabc", t2, v.oovv, t3) * -0.25
    t3new += np.einsum("ilab,lmde,jkmcde->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("ilac,lmde,jkmbde->ijkabc", t2, v.oovv, t3) * -0.5
    t3new += np.einsum("ilad,lmde,jkmbce->ijkabc", t2, v.oovv, t3) * 1.00000000000001
    t3new += np.einsum("ilbc,lmde,jkmade->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("ilbd,lmde,jkmace->ijkabc", t2, v.oovv, t3) * -1.00000000000001
    t3new += np.einsum("ilcd,lmde,jkmabe->ijkabc", t2, v.oovv, t3) * 1.00000000000001
    t3new += np.einsum("ilde,lmde,jkmabc->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("jkad,lmde,ilmbce->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("jkbd,lmde,ilmace->ijkabc", t2, v.oovv, t3) * -0.5
    t3new += np.einsum("jkcd,lmde,ilmabe->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("jkde,lmde,ilmabc->ijkabc", t2, v.oovv, t3) * 0.25
    t3new += np.einsum("jlab,lmde,ikmcde->ijkabc", t2, v.oovv, t3) * -0.5
    t3new += np.einsum("jlac,lmde,ikmbde->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("jlad,lmde,ikmbce->ijkabc", t2, v.oovv, t3) * -1.00000000000001
    t3new += np.einsum("jlbc,lmde,ikmade->ijkabc", t2, v.oovv, t3) * -0.5
    t3new += np.einsum("jlbd,lmde,ikmace->ijkabc", t2, v.oovv, t3) * 1.00000000000001
    t3new += np.einsum("jlcd,lmde,ikmabe->ijkabc", t2, v.oovv, t3) * -1.00000000000001
    t3new += np.einsum("jlde,lmde,ikmabc->ijkabc", t2, v.oovv, t3) * -0.5
    t3new += np.einsum("klab,lmde,ijmcde->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("klac,lmde,ijmbde->ijkabc", t2, v.oovv, t3) * -0.5
    t3new += np.einsum("klad,lmde,ijmbce->ijkabc", t2, v.oovv, t3) * 1.00000000000001
    t3new += np.einsum("klbc,lmde,ijmade->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("klbd,lmde,ijmace->ijkabc", t2, v.oovv, t3) * -1.00000000000001
    t3new += np.einsum("klcd,lmde,ijmabe->ijkabc", t2, v.oovv, t3) * 1.00000000000001
    t3new += np.einsum("klde,lmde,ijmabc->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("lmab,lmde,ijkcde->ijkabc", t2, v.oovv, t3) * 0.25
    t3new += np.einsum("lmac,lmde,ijkbde->ijkabc", t2, v.oovv, t3) * -0.25
    t3new += np.einsum("lmad,lmde,ijkbce->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("lmbc,lmde,ijkade->ijkabc", t2, v.oovv, t3) * 0.25
    t3new += np.einsum("lmbd,lmde,ijkace->ijkabc", t2, v.oovv, t3) * -0.5
    t3new += np.einsum("lmcd,lmde,ijkabe->ijkabc", t2, v.oovv, t3) * 0.5
    t3new += np.einsum("id,je,klab,lcde->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("id,je,klac,lbde->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("id,je,klbc,lade->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("id,ke,jlab,lcde->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("id,ke,jlac,lbde->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("id,ke,jlbc,lade->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("id,la,jkbe,lcde->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("id,la,jkce,lbde->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("id,la,jmbc,lmkd->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("id,la,kmbc,lmjd->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("id,lb,jkae,lcde->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("id,lb,jkce,lade->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("id,lb,jmac,lmkd->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("id,lb,kmac,lmjd->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("id,lc,jkae,lbde->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("id,lc,jkbe,lade->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("id,lc,jmab,lmkd->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("id,lc,kmab,lmjd->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("jd,ke,ilab,lcde->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("jd,ke,ilac,lbde->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("jd,ke,ilbc,lade->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("jd,la,ikbe,lcde->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("jd,la,ikce,lbde->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("jd,la,imbc,lmkd->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("jd,la,kmbc,lmid->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("jd,lb,ikae,lcde->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("jd,lb,ikce,lade->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("jd,lb,imac,lmkd->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("jd,lb,kmac,lmid->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("jd,lc,ikae,lbde->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("jd,lc,ikbe,lade->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("jd,lc,imab,lmkd->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("jd,lc,kmab,lmid->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("kd,la,ijbe,lcde->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("kd,la,ijce,lbde->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("kd,la,imbc,lmjd->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("kd,la,jmbc,lmid->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("kd,lb,ijae,lcde->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("kd,lb,ijce,lade->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("kd,lb,imac,lmjd->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("kd,lb,jmac,lmid->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("kd,lc,ijae,lbde->ijkabc", t1, t1, t2, v.ovvv) * -1.0
    t3new += np.einsum("kd,lc,ijbe,lade->ijkabc", t1, t1, t2, v.ovvv)
    t3new += np.einsum("kd,lc,imab,lmjd->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("kd,lc,jmab,lmid->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("la,mb,ijcd,lmkd->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("la,mb,ikcd,lmjd->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("la,mb,jkcd,lmid->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("la,mc,ijbd,lmkd->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("la,mc,ikbd,lmjd->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("la,mc,jkbd,lmid->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("lb,mc,ijad,lmkd->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("lb,mc,ikad,lmjd->ijkabc", t1, t1, t2, v.ooov)
    t3new += np.einsum("lb,mc,jkad,lmid->ijkabc", t1, t1, t2, v.ooov) * -1.0
    t3new += np.einsum("id,je,lmde,klmabc->ijkabc", t1, t1, v.oovv, t3) * 0.5
    t3new += np.einsum("id,ke,lmde,jlmabc->ijkabc", t1, t1, v.oovv, t3) * -0.5
    t3new += np.einsum("id,la,lmde,jkmbce->ijkabc", t1, t1, v.oovv, t3) * -1.0
    t3new += np.einsum("id,lb,lmde,jkmace->ijkabc", t1, t1, v.oovv, t3)
    t3new += np.einsum("id,lc,lmde,jkmabe->ijkabc", t1, t1, v.oovv, t3) * -1.0
    t3new += np.einsum("id,le,lmde,jkmabc->ijkabc", t1, t1, v.oovv, t3)
    t3new += np.einsum("jd,ke,lmde,ilmabc->ijkabc", t1, t1, v.oovv, t3) * 0.5
    t3new += np.einsum("jd,la,lmde,ikmbce->ijkabc", t1, t1, v.oovv, t3)
    t3new += np.einsum("jd,lb,lmde,ikmace->ijkabc", t1, t1, v.oovv, t3) * -1.0
    t3new += np.einsum("jd,lc,lmde,ikmabe->ijkabc", t1, t1, v.oovv, t3)
    t3new += np.einsum("jd,le,lmde,ikmabc->ijkabc", t1, t1, v.oovv, t3) * -1.0
    t3new += np.einsum("kd,la,lmde,ijmbce->ijkabc", t1, t1, v.oovv, t3) * -1.0
    t3new += np.einsum("kd,lb,lmde,ijmace->ijkabc", t1, t1, v.oovv, t3)
    t3new += np.einsum("kd,lc,lmde,ijmabe->ijkabc", t1, t1, v.oovv, t3) * -1.0
    t3new += np.einsum("kd,le,lmde,ijmabc->ijkabc", t1, t1, v.oovv, t3)
    t3new += np.einsum("la,mb,lmde,ijkcde->ijkabc", t1, t1, v.oovv, t3) * 0.5
    t3new += np.einsum("la,mc,lmde,ijkbde->ijkabc", t1, t1, v.oovv, t3) * -0.5
    t3new += np.einsum("lb,mc,lmde,ijkade->ijkabc", t1, t1, v.oovv, t3) * 0.5
    t3new += np.einsum("ld,ma,lmde,ijkbce->ijkabc", t1, t1, v.oovv, t3) * -1.0
    t3new += np.einsum("ld,mb,lmde,ijkace->ijkabc", t1, t1, v.oovv, t3)
    t3new += np.einsum("ld,mc,lmde,ijkabe->ijkabc", t1, t1, v.oovv, t3) * -1.0
    t3new += np.einsum("id,jkae,lmbc,lmde->ijkabc", t1, t2, t2, v.oovv) * -0.5
    t3new += np.einsum("id,jkbe,lmac,lmde->ijkabc", t1, t2, t2, v.oovv) * 0.5
    t3new += np.einsum("id,jkce,lmab,lmde->ijkabc", t1, t2, t2, v.oovv) * -0.5
    t3new += np.einsum("id,jlab,kmce,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("id,jlac,kmbe,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("id,jlae,kmbc,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("id,jlbc,kmae,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("id,jlbe,kmac,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("id,jlce,kmab,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("jd,ikae,lmbc,lmde->ijkabc", t1, t2, t2, v.oovv) * 0.5
    t3new += np.einsum("jd,ikbe,lmac,lmde->ijkabc", t1, t2, t2, v.oovv) * -0.5
    t3new += np.einsum("jd,ikce,lmab,lmde->ijkabc", t1, t2, t2, v.oovv) * 0.5
    t3new += np.einsum("jd,ilab,kmce,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("jd,ilac,kmbe,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("jd,ilae,kmbc,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("jd,ilbc,kmae,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("jd,ilbe,kmac,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("jd,ilce,kmab,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("kd,ijae,lmbc,lmde->ijkabc", t1, t2, t2, v.oovv) * -0.5
    t3new += np.einsum("kd,ijbe,lmac,lmde->ijkabc", t1, t2, t2, v.oovv) * 0.5
    t3new += np.einsum("kd,ijce,lmab,lmde->ijkabc", t1, t2, t2, v.oovv) * -0.5
    t3new += np.einsum("kd,ilab,jmce,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("kd,ilac,jmbe,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("kd,ilae,jmbc,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("kd,ilbc,jmae,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("kd,ilbe,jmac,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("kd,ilce,jmab,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("la,ijbd,kmce,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("la,ijcd,kmbe,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("la,ijde,kmbc,lmde->ijkabc", t1, t2, t2, v.oovv) * -0.5
    t3new += np.einsum("la,ikbd,jmce,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("la,ikcd,jmbe,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("la,ikde,jmbc,lmde->ijkabc", t1, t2, t2, v.oovv) * 0.5
    t3new += np.einsum("la,imbc,jkde,lmde->ijkabc", t1, t2, t2, v.oovv) * -0.5
    t3new += np.einsum("la,imbd,jkce,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("la,imcd,jkbe,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("lb,ijad,kmce,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("lb,ijcd,kmae,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("lb,ijde,kmac,lmde->ijkabc", t1, t2, t2, v.oovv) * 0.5
    t3new += np.einsum("lb,ikad,jmce,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("lb,ikcd,jmae,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("lb,ikde,jmac,lmde->ijkabc", t1, t2, t2, v.oovv) * -0.5
    t3new += np.einsum("lb,imac,jkde,lmde->ijkabc", t1, t2, t2, v.oovv) * 0.5
    t3new += np.einsum("lb,imad,jkce,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("lb,imcd,jkae,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("lc,ijad,kmbe,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("lc,ijbd,kmae,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("lc,ijde,kmab,lmde->ijkabc", t1, t2, t2, v.oovv) * -0.5
    t3new += np.einsum("lc,ikad,jmbe,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("lc,ikbd,jmae,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("lc,ikde,jmab,lmde->ijkabc", t1, t2, t2, v.oovv) * 0.5
    t3new += np.einsum("lc,imab,jkde,lmde->ijkabc", t1, t2, t2, v.oovv) * -0.5
    t3new += np.einsum("lc,imad,jkbe,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("lc,imbd,jkae,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("ld,ijae,kmbc,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("ld,ijbe,kmac,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("ld,ijce,kmab,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("ld,ikae,jmbc,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("ld,ikbe,jmac,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("ld,ikce,jmab,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("ld,imab,jkce,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("ld,imac,jkbe,lmde->ijkabc", t1, t2, t2, v.oovv) * -1.0
    t3new += np.einsum("ld,imbc,jkae,lmde->ijkabc", t1, t2, t2, v.oovv)
    t3new += np.einsum("id,je,la,kmbc,lmde->ijkabc", t1, t1, t1, t2, v.oovv) * -1.0
    t3new += np.einsum("id,je,lb,kmac,lmde->ijkabc", t1, t1, t1, t2, v.oovv)
    t3new += np.einsum("id,je,lc,kmab,lmde->ijkabc", t1, t1, t1, t2, v.oovv) * -1.0
    t3new += np.einsum("id,ke,la,jmbc,lmde->ijkabc", t1, t1, t1, t2, v.oovv)
    t3new += np.einsum("id,ke,lb,jmac,lmde->ijkabc", t1, t1, t1, t2, v.oovv) * -1.0
    t3new += np.einsum("id,ke,lc,jmab,lmde->ijkabc", t1, t1, t1, t2, v.oovv)
    t3new += np.einsum("id,la,mb,jkce,lmde->ijkabc", t1, t1, t1, t2, v.oovv) * -1.0
    t3new += np.einsum("id,la,mc,jkbe,lmde->ijkabc", t1, t1, t1, t2, v.oovv)
    t3new += np.einsum("id,lb,mc,jkae,lmde->ijkabc", t1, t1, t1, t2, v.oovv) * -1.0
    t3new += np.einsum("jd,ke,la,imbc,lmde->ijkabc", t1, t1, t1, t2, v.oovv) * -1.0
    t3new += np.einsum("jd,ke,lb,imac,lmde->ijkabc", t1, t1, t1, t2, v.oovv)
    t3new += np.einsum("jd,ke,lc,imab,lmde->ijkabc", t1, t1, t1, t2, v.oovv) * -1.0
    t3new += np.einsum("jd,la,mb,ikce,lmde->ijkabc", t1, t1, t1, t2, v.oovv)
    t3new += np.einsum("jd,la,mc,ikbe,lmde->ijkabc", t1, t1, t1, t2, v.oovv) * -1.0
    t3new += np.einsum("jd,lb,mc,ikae,lmde->ijkabc", t1, t1, t1, t2, v.oovv)
    t3new += np.einsum("kd,la,mb,ijce,lmde->ijkabc", t1, t1, t1, t2, v.oovv) * -1.0
    t3new += np.einsum("kd,la,mc,ijbe,lmde->ijkabc", t1, t1, t1, t2, v.oovv)
    t3new += np.einsum("kd,lb,mc,ijae,lmde->ijkabc", t1, t1, t1, t2, v.oovv) * -1.0

    return {"t1new": t1new, "t2new": t2new, "t3new": t3new}
