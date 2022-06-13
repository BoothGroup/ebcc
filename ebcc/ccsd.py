# Code generated by qwick.

import numpy as np
from pyscf import lib

def energy(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # CCSD energy
    e_cc = 0
    e_cc += lib.einsum("ia,ia->", f.ov, t1) * 2
    e_cc += lib.einsum("ijab,iajb->", t2, v.ovov) * 2
    e_cc += lib.einsum("ijab,ibja->", t2, v.ovov) * -1
    e_cc += lib.einsum("ia,jb,iajb->", t1, t1, v.ovov) * 2
    e_cc += lib.einsum("ia,jb,ibja->", t1, t1, v.ovov) * -1
    return {"e_cc": e_cc}

def update_amps(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, **kwargs):
    # T1 amplitude
    t1new = np.zeros((nocc, nvir), dtype=np.float64)
    t1new -= lib.einsum("kj,kb->jb", f.oo, t1)
    t1new += lib.einsum("kjcd,bdkc->jb", t2, v.vvov) * 2
    t1new += lib.einsum("kjcd,bckd->jb", t2, v.vvov) * -1
    t1new -= lib.einsum("kc,jd,bckd->jb", t1, t1, v.vvov)
    t1new += lib.einsum("kc,jd,bdkc->jb", t1, t1, v.vvov) * 2
    t1new += lib.einsum("kc,ljbd,kdlc->jb", t1, t2, v.ovov)
    t1new -= lib.einsum("kc,jlbd,kdlc->jb", t1, t2, v.ovov) * 2
    t1new -= lib.einsum("kc,ljbd,kcld->jb", t1, t2, v.ovov) * 2
    t1new += lib.einsum("kc,jlbd,kcld->jb", t1, t2, v.ovov) * 4
    t1new += lib.einsum("kb,ljcd,kcld->jb", t1, t2, v.ovov)
    t1new += lib.einsum("jc,klbd,kdlc->jb", t1, t2, v.ovov)
    t1new += lib.einsum("kb,ljcd,kdlc->jb", t1, t2, v.ovov) * -2
    t1new += lib.einsum("jc,klbd,kcld->jb", t1, t2, v.ovov) * -2
    t1new += lib.einsum("kb,lc,jd,kcld->jb", t1, t1, t1, v.ovov)
    t1new -= lib.einsum("kb,lc,jd,kdlc->jb", t1, t1, t1, v.ovov) * 2
    t1new += lib.einsum("bc,jc->jb", f.vv, t1)
    t1new += lib.einsum("bj->jb", f.vo)
    t1new -= lib.einsum("kc,kjbc->jb", f.ov, t2)
    t1new -= lib.einsum("kc,bckj->jb", t1, v.vvoo)
    t1new += lib.einsum("kc,jkbc->jb", f.ov, t2) * 2
    t1new += lib.einsum("kc,bjkc->jb", t1, v.voov) * 2
    t1new -= lib.einsum("kc,kb,jc->jb", f.ov, t1, t1)
    t1new += lib.einsum("klbc,kclj->jb", t2, v.ovoo)
    t1new += lib.einsum("klbc,lckj->jb", t2, v.ovoo) * -2
    t1new += lib.einsum("kb,lc,kclj->jb", t1, t1, v.ovoo)
    t1new -= lib.einsum("kb,lc,lckj->jb", t1, t1, v.ovoo) * 2
    # T2 amplitude
    t2new = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2new += lib.einsum("ce,lkde->klcd", f.vv, t2)
    t2new += lib.einsum("de,klce->klcd", f.vv, t2)
    t2new += lib.einsum("ke,cedl->klcd", t1, v.vvvo)
    t2new += lib.einsum("le,ckde->klcd", t1, v.vovv)
    t2new -= lib.einsum("ke,mlcg,dgme->klcd", t1, t2, v.vvov)
    t2new -= lib.einsum("ke,lmdg,cgme->klcd", t1, t2, v.vvov)
    t2new -= lib.einsum("ke,mldg,cemg->klcd", t1, t2, v.vvov)
    t2new -= lib.einsum("le,kmcg,dgme->klcd", t1, t2, v.vvov)
    t2new -= lib.einsum("le,mkcg,demg->klcd", t1, t2, v.vvov)
    t2new -= lib.einsum("le,mkdg,cgme->klcd", t1, t2, v.vvov)
    t2new -= lib.einsum("me,klcg,demg->klcd", t1, t2, v.vvov)
    t2new -= lib.einsum("me,lkdg,cemg->klcd", t1, t2, v.vvov)
    t2new += lib.einsum("ke,lmdg,cemg->klcd", t1, t2, v.vvov) * 2
    t2new += lib.einsum("le,kmcg,demg->klcd", t1, t2, v.vvov) * 2
    t2new += lib.einsum("me,klcg,dgme->klcd", t1, t2, v.vvov) * 2
    t2new += lib.einsum("me,lkdg,cgme->klcd", t1, t2, v.vvov) * 2
    t2new += lib.einsum("mc,kleg,dgme->klcd", t1, t2, v.vvov) * -1
    t2new += lib.einsum("md,kleg,cemg->klcd", t1, t2, v.vvov) * -1
    t2new -= lib.einsum("mc,ke,lg,dgme->klcd", t1, t1, t1, v.vvov)
    t2new -= lib.einsum("md,ke,lg,cemg->klcd", t1, t1, t1, v.vvov)
    t2new += lib.einsum("mc,lnde,menk->klcd", t1, t2, v.ovoo)
    t2new += lib.einsum("mc,nkde,menl->klcd", t1, t2, v.ovoo)
    t2new += lib.einsum("mc,nlde,nemk->klcd", t1, t2, v.ovoo)
    t2new += lib.einsum("md,knce,menl->klcd", t1, t2, v.ovoo)
    t2new += lib.einsum("md,nkce,neml->klcd", t1, t2, v.ovoo)
    t2new += lib.einsum("md,nlce,menk->klcd", t1, t2, v.ovoo)
    t2new += lib.einsum("me,kncd,neml->klcd", t1, t2, v.ovoo)
    t2new += lib.einsum("me,nlcd,nemk->klcd", t1, t2, v.ovoo)
    t2new -= lib.einsum("mc,lnde,nemk->klcd", t1, t2, v.ovoo) * 2
    t2new -= lib.einsum("md,knce,neml->klcd", t1, t2, v.ovoo) * 2
    t2new -= lib.einsum("me,kncd,menl->klcd", t1, t2, v.ovoo) * 2
    t2new -= lib.einsum("me,nlcd,menk->klcd", t1, t2, v.ovoo) * 2
    t2new += lib.einsum("ke,mncd,menl->klcd", t1, t2, v.ovoo)
    t2new += lib.einsum("le,mncd,nemk->klcd", t1, t2, v.ovoo)
    t2new += lib.einsum("mc,nd,ke,menl->klcd", t1, t1, t1, v.ovoo)
    t2new += lib.einsum("mc,nd,le,nemk->klcd", t1, t1, t1, v.ovoo)
    t2new -= lib.einsum("kmce,deml->klcd", t2, v.vvoo)
    t2new -= lib.einsum("mkce,dlme->klcd", t2, v.voov)
    t2new -= lib.einsum("mlce,demk->klcd", t2, v.vvoo)
    t2new -= lib.einsum("lmde,cemk->klcd", t2, v.vvoo)
    t2new -= lib.einsum("mkde,ceml->klcd", t2, v.vvoo)
    t2new -= lib.einsum("mlde,ckme->klcd", t2, v.voov)
    t2new += lib.einsum("kmce,dlme->klcd", t2, v.voov) * 2
    t2new += lib.einsum("lmde,ckme->klcd", t2, v.voov) * 2
    t2new -= lib.einsum("me,mc,lkde->klcd", f.ov, t1, t2)
    t2new -= lib.einsum("me,md,klce->klcd", f.ov, t1, t2)
    t2new -= lib.einsum("me,ke,mlcd->klcd", f.ov, t1, t2)
    t2new -= lib.einsum("me,le,kmcd->klcd", f.ov, t1, t2)
    t2new -= lib.einsum("mc,ke,dlme->klcd", t1, t1, v.voov)
    t2new -= lib.einsum("mc,le,demk->klcd", t1, t1, v.vvoo)
    t2new -= lib.einsum("md,ke,ceml->klcd", t1, t1, v.vvoo)
    t2new -= lib.einsum("md,le,ckme->klcd", t1, t1, v.voov)
    t2new -= lib.einsum("mk,mlcd->klcd", f.oo, t2)
    t2new -= lib.einsum("ml,kmcd->klcd", f.oo, t2)
    t2new -= lib.einsum("mc,dlmk->klcd", t1, v.vooo)
    t2new -= lib.einsum("md,ckml->klcd", t1, v.vooo)
    t2new += lib.einsum("kmce,nldg,mgne->klcd", t2, t2, v.ovov)
    t2new += lib.einsum("mkce,lndg,mgne->klcd", t2, t2, v.ovov)
    t2new += lib.einsum("mkce,nldg,meng->klcd", t2, t2, v.ovov)
    t2new += lib.einsum("mlce,nkdg,mgne->klcd", t2, t2, v.ovov)
    t2new -= lib.einsum("kmce,lndg,mgne->klcd", t2, t2, v.ovov) * 2
    t2new -= lib.einsum("kmce,nldg,meng->klcd", t2, t2, v.ovov) * 2
    t2new -= lib.einsum("mkce,lndg,meng->klcd", t2, t2, v.ovov) * 2
    t2new += lib.einsum("kmce,lndg,meng->klcd", t2, t2, v.ovov) * 4
    t2new += lib.einsum("kmcd,nleg,meng->klcd", t2, t2, v.ovov)
    t2new += lib.einsum("mlcd,nkeg,meng->klcd", t2, t2, v.ovov)
    t2new += lib.einsum("mncd,kleg,meng->klcd", t2, t2, v.ovov)
    t2new += lib.einsum("klce,mndg,mgne->klcd", t2, t2, v.ovov)
    t2new += lib.einsum("mnce,lkdg,meng->klcd", t2, t2, v.ovov)
    t2new += lib.einsum("kmcd,nleg,mgne->klcd", t2, t2, v.ovov) * -2
    t2new += lib.einsum("mlcd,nkeg,mgne->klcd", t2, t2, v.ovov) * -2
    t2new += lib.einsum("klce,mndg,meng->klcd", t2, t2, v.ovov) * -2
    t2new += lib.einsum("mnce,lkdg,mgne->klcd", t2, t2, v.ovov) * -2
    t2new += lib.einsum("mc,ke,lndg,mgne->klcd", t1, t1, t2, v.ovov)
    t2new += lib.einsum("mc,ke,nldg,meng->klcd", t1, t1, t2, v.ovov)
    t2new += lib.einsum("mc,le,nkdg,mgne->klcd", t1, t1, t2, v.ovov)
    t2new += lib.einsum("mc,ne,lkdg,meng->klcd", t1, t1, t2, v.ovov)
    t2new += lib.einsum("md,ke,nlcg,mgne->klcd", t1, t1, t2, v.ovov)
    t2new += lib.einsum("md,le,kncg,mgne->klcd", t1, t1, t2, v.ovov)
    t2new += lib.einsum("md,le,nkcg,meng->klcd", t1, t1, t2, v.ovov)
    t2new += lib.einsum("md,ne,klcg,meng->klcd", t1, t1, t2, v.ovov)
    t2new += lib.einsum("me,kg,nlcd,mgne->klcd", t1, t1, t2, v.ovov)
    t2new += lib.einsum("me,lg,kncd,mgne->klcd", t1, t1, t2, v.ovov)
    t2new -= lib.einsum("mc,ke,lndg,meng->klcd", t1, t1, t2, v.ovov) * 2
    t2new -= lib.einsum("mc,ne,lkdg,mgne->klcd", t1, t1, t2, v.ovov) * 2
    t2new -= lib.einsum("md,le,kncg,meng->klcd", t1, t1, t2, v.ovov) * 2
    t2new -= lib.einsum("md,ne,klcg,mgne->klcd", t1, t1, t2, v.ovov) * 2
    t2new -= lib.einsum("me,kg,nlcd,meng->klcd", t1, t1, t2, v.ovov) * 2
    t2new -= lib.einsum("me,lg,kncd,meng->klcd", t1, t1, t2, v.ovov) * 2
    t2new += lib.einsum("mc,nd,kleg,meng->klcd", t1, t1, t2, v.ovov)
    t2new += lib.einsum("ke,lg,mncd,meng->klcd", t1, t1, t2, v.ovov)
    t2new += lib.einsum("mc,nd,ke,lg,meng->klcd", t1, t1, t1, t1, v.ovov)
    t2new += lib.einsum("mncd,mknl->klcd", t2, v.oooo)
    t2new += lib.einsum("mc,nd,mknl->klcd", t1, t1, v.oooo)
    t2new += lib.einsum("ckdl->klcd", v.vovo)
    t2new += lib.einsum("kleg,cedg->klcd", t2, v.vvvv)
    t2new += lib.einsum("ke,lg,cedg->klcd", t1, t1, v.vvvv)
    return {"t1new": t1new, "t2new": t2new}

def update_lams(f=None, v=None, nocc=None, nvir=None, t1=None, t2=None, l1=None, l2=None, **kwargs):
    # L1 amplitude
    l1new = np.zeros((nvir, nocc), dtype=np.float64)
    l1new += lib.einsum("nA,ghnm,hegA->em", t1, l2, v.vvvv) * 2
    l1new += lib.einsum("nA,ghnm,gehA->em", t1, l2, v.vvvv) * -1
    l1new -= lib.einsum("og,ghnm,nphA,peoA->em", t1, l2, t2, v.ovov)
    l1new -= lib.einsum("og,ghnm,pnhA,oepA->em", t1, l2, t2, v.ovov)
    l1new -= lib.einsum("nh,egno,pogA,phmA->em", t1, l2, t2, v.ovov)
    l1new -= lib.einsum("oh,egnm,pngA,oAph->em", t1, l2, t2, v.ovov)
    l1new -= lib.einsum("oh,ghnm,pngA,peoA->em", t1, l2, t2, v.ovov)
    l1new -= lib.einsum("oh,egno,npgA,phmA->em", t1, l2, t2, v.ovov)
    l1new -= lib.einsum("oh,egno,pngA,pAmh->em", t1, l2, t2, v.ovov)
    l1new -= lib.einsum("nh,egno,opgA,pAmh->em", t1, l2, t2, v.ovov) * 4
    l1new -= lib.einsum("oh,egnm,npgA,ohpA->em", t1, l2, t2, v.ovov) * 4
    l1new -= lib.einsum("oh,genm,npgA,oAph->em", t1, l2, t2, v.ovov) * 4
    l1new -= lib.einsum("oh,genm,pngA,ohpA->em", t1, l2, t2, v.ovov) * 4
    l1new -= lib.einsum("oh,ghnm,npgA,oepA->em", t1, l2, t2, v.ovov) * 4
    l1new += lib.einsum("og,ghnm,nphA,oepA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("og,ghnm,pnhA,peoA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("nh,egno,opgA,phmA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("nh,egno,pogA,pAmh->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("oh,egnm,npgA,oAph->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("oh,egnm,pngA,ohpA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("oh,genm,pngA,oAph->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("oh,ghnm,npgA,peoA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("oh,ghnm,pngA,oepA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("oh,egno,npgA,pAmh->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("oh,egno,pngA,phmA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("oh,genm,npgA,ohpA->em", t1, l2, t2, v.ovov) * 8
    l1new += lib.einsum("nA,ghnm,pogh,oepA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("nA,ghno,opgh,pAme->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("nA,ghno,ophg,pemA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("oA,ghnm,npgh,oepA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("oA,ghnm,pngh,peoA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("og,egnm,nphA,ohpA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("og,genm,nphA,oAph->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("pg,egno,onhA,phmA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("pg,ghno,nohA,pAme->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("pg,ghno,onhA,pemA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("nh,egnm,opgA,ohpA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("nh,genm,opgA,phoA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("ph,egno,nogA,phmA->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("ph,egno,ongA,pAmh->em", t1, l2, t2, v.ovov) * 2
    l1new += lib.einsum("nA,ghnm,opgh,oepA->em", t1, l2, t2, v.ovov) * -1
    l1new += lib.einsum("nA,ghno,opgh,pemA->em", t1, l2, t2, v.ovov) * -1
    l1new += lib.einsum("oA,ghnm,pngh,oepA->em", t1, l2, t2, v.ovov) * -1
    l1new += lib.einsum("og,egnm,nphA,oAph->em", t1, l2, t2, v.ovov) * -1
    l1new += lib.einsum("pg,egno,nohA,phmA->em", t1, l2, t2, v.ovov) * -1
    l1new += lib.einsum("pg,ghno,nohA,pemA->em", t1, l2, t2, v.ovov) * -1
    l1new += lib.einsum("nh,egnm,opgA,phoA->em", t1, l2, t2, v.ovov) * -1
    l1new += lib.einsum("ph,egno,nogA,pAmh->em", t1, l2, t2, v.ovov) * -1
    l1new += lib.einsum("nA,ghno,ophg,pAme->em", t1, l2, t2, v.ovov) * -4
    l1new += lib.einsum("oA,ghnm,npgh,peoA->em", t1, l2, t2, v.ovov) * -4
    l1new += lib.einsum("og,genm,nphA,ohpA->em", t1, l2, t2, v.ovov) * -4
    l1new += lib.einsum("pg,ghno,onhA,pAme->em", t1, l2, t2, v.ovov) * -4
    l1new += lib.einsum("nh,genm,opgA,ohpA->em", t1, l2, t2, v.ovov) * -4
    l1new += lib.einsum("ph,egno,ongA,phmA->em", t1, l2, t2, v.ovov) * -4
    l1new -= lib.einsum("pA,og,nh,egnm,oAph->em", t1, t1, t1, l2, v.ovov)
    l1new -= lib.einsum("pA,og,nh,genm,ohpA->em", t1, t1, t1, l2, v.ovov) * 4
    l1new += lib.einsum("pA,og,nh,egnm,ohpA->em", t1, t1, t1, l2, v.ovov) * 2
    l1new += lib.einsum("pA,og,nh,genm,oAph->em", t1, t1, t1, l2, v.ovov) * 2
    l1new += lib.einsum("nA,pg,oh,ghnm,oepA->em", t1, t1, t1, l2, v.ovov) * 2.0000000000000013
    l1new += lib.einsum("nA,pg,oh,egno,phmA->em", t1, t1, t1, l2, v.ovov) * 2.0000000000000013
    l1new += lib.einsum("nA,og,ph,ghnm,oepA->em", t1, t1, t1, l2, v.ovov) * -1
    l1new += lib.einsum("oA,pg,nh,egno,phmA->em", t1, t1, t1, l2, v.ovov) * -1
    l1new -= lib.einsum("gn,egnm->em", f.vo, l2)
    l1new -= lib.einsum("gn,gemn->em", l1, v.vvoo)
    l1new -= lib.einsum("ng,nemg->em", t1, v.ovov)
    l1new += lib.einsum("gn,genm->em", f.vo, l2) * 2
    l1new += lib.einsum("gn,gnme->em", l1, v.voov) * 2
    l1new += lib.einsum("ng,ngme->em", t1, v.ovov) * 2
    l1new -= lib.einsum("mg,en,ng->em", f.ov, l1, t1)
    l1new -= lib.einsum("ne,gm,ng->em", f.ov, l1, t1)
    l1new += lib.einsum("ghnm,hegn->em", l2, v.vvvo) * 2
    l1new += lib.einsum("ghnm,gehn->em", l2, v.vvvo) * -1
    l1new -= lib.einsum("gh,nh,egnm->em", f.vv, t1, l2)
    l1new -= lib.einsum("gm,nh,negh->em", l1, t1, v.ovvv)
    l1new -= lib.einsum("gn,nh,gemh->em", l1, t1, v.vvov)
    l1new += lib.einsum("gh,nh,genm->em", f.vv, t1, l2) * 2
    l1new += lib.einsum("gm,nh,genh->em", l1, t1, v.vvov) * 2
    l1new += lib.einsum("gn,nh,ghme->em", l1, t1, v.vvov) * 2
    l1new -= lib.einsum("egno,npgh,pomh->em", l2, t2, v.ooov)
    l1new -= lib.einsum("egno,pngh,phmo->em", l2, t2, v.ovoo)
    l1new -= lib.einsum("egno,pogh,pnmh->em", l2, t2, v.ooov)
    l1new -= lib.einsum("egno,opgh,phmn->em", l2, t2, v.ovoo) * 4
    l1new += lib.einsum("egno,npgh,phmo->em", l2, t2, v.ovoo) * 2
    l1new += lib.einsum("egno,opgh,pnmh->em", l2, t2, v.ooov) * 2
    l1new += lib.einsum("egno,pngh,pomh->em", l2, t2, v.ooov) * 2
    l1new += lib.einsum("egno,pogh,phmn->em", l2, t2, v.ovoo) * 2
    l1new += lib.einsum("egnm,opgh,onph->em", l2, t2, v.ooov) * 2
    l1new += lib.einsum("genm,pogh,onph->em", l2, t2, v.ooov) * 2
    l1new += lib.einsum("ghnm,pogh,oepn->em", l2, t2, v.ovoo) * 2
    l1new += lib.einsum("ghno,npgh,pemo->em", l2, t2, v.ovoo) * 2
    l1new += lib.einsum("ghno,nphg,pome->em", l2, t2, v.ooov) * 2
    l1new += lib.einsum("egnm,pogh,onph->em", l2, t2, v.ooov) * -1
    l1new += lib.einsum("ghnm,opgh,oepn->em", l2, t2, v.ovoo) * -1
    l1new += lib.einsum("ghno,nphg,pemo->em", l2, t2, v.ovoo) * -1
    l1new += lib.einsum("genm,opgh,onph->em", l2, t2, v.ooov) * -4
    l1new += lib.einsum("ghno,npgh,pome->em", l2, t2, v.ooov) * -4
    l1new -= lib.einsum("pg,nh,egno,phmo->em", t1, t1, l2, v.ovoo)
    l1new -= lib.einsum("pg,oh,egnm,onph->em", t1, t1, l2, v.ooov)
    l1new -= lib.einsum("pg,oh,egno,pnmh->em", t1, t1, l2, v.ooov)
    l1new -= lib.einsum("og,ph,genm,onph->em", t1, t1, l2, v.ooov) * 4
    l1new += lib.einsum("og,ph,egnm,onph->em", t1, t1, l2, v.ooov) * 2
    l1new += lib.einsum("pg,nh,egno,pomh->em", t1, t1, l2, v.ooov) * 2
    l1new += lib.einsum("pg,oh,genm,onph->em", t1, t1, l2, v.ooov) * 2
    l1new += lib.einsum("pg,oh,egno,phmn->em", t1, t1, l2, v.ovoo) * 2
    l1new += lib.einsum("pg,oh,ghnm,oepn->em", t1, t1, l2, v.ovoo) * 2
    l1new += lib.einsum("og,ph,ghnm,oepn->em", t1, t1, l2, v.ovoo) * -1
    l1new += lib.einsum("ge,gm->em", f.vv, l1)
    l1new += lib.einsum("ghnm,ongA,oehA->em", l2, t2, v.ovvv)
    l1new += lib.einsum("ghnm,nohA,oegA->em", l2, t2, v.ovvv)
    l1new += lib.einsum("ghnm,onhA,geoA->em", l2, t2, v.vvov)
    l1new -= lib.einsum("ghnm,nogA,oehA->em", l2, t2, v.ovvv) * 2
    l1new -= lib.einsum("ghnm,ongA,heoA->em", l2, t2, v.vvov) * 2
    l1new -= lib.einsum("ghnm,nohA,geoA->em", l2, t2, v.vvov) * 2
    l1new -= lib.einsum("ghnm,onhA,oegA->em", l2, t2, v.ovvv) * 2
    l1new += lib.einsum("ghnm,nogA,heoA->em", l2, t2, v.vvov) * 4
    l1new += lib.einsum("egnm,nohA,ohgA->em", l2, t2, v.ovvv)
    l1new += lib.einsum("egno,nohA,ghmA->em", l2, t2, v.vvov)
    l1new += lib.einsum("ghno,ongA,hemA->em", l2, t2, v.vvov)
    l1new += lib.einsum("genm,noAh,ohgA->em", l2, t2, v.ovvv) * 4
    l1new += lib.einsum("ghno,nogA,hAme->em", l2, t2, v.vvov) * 4
    l1new += lib.einsum("egnm,noAh,ohgA->em", l2, t2, v.ovvv) * -2
    l1new += lib.einsum("genm,nohA,ohgA->em", l2, t2, v.ovvv) * -2
    l1new += lib.einsum("egno,onhA,ghmA->em", l2, t2, v.vvov) * -2
    l1new += lib.einsum("ghno,nogA,hemA->em", l2, t2, v.vvov) * -2
    l1new += lib.einsum("ghno,ongA,hAme->em", l2, t2, v.vvov) * -2
    l1new += lib.einsum("nA,og,ghnm,oehA->em", t1, t1, l2, v.ovvv)
    l1new += lib.einsum("nA,oh,ghnm,geoA->em", t1, t1, l2, v.vvov)
    l1new += lib.einsum("oA,nh,egnm,ohgA->em", t1, t1, l2, v.ovvv)
    l1new -= lib.einsum("nA,og,ghnm,heoA->em", t1, t1, l2, v.vvov) * 2
    l1new -= lib.einsum("nA,oh,egnm,ohgA->em", t1, t1, l2, v.ovvv) * 2
    l1new -= lib.einsum("nA,oh,ghnm,oegA->em", t1, t1, l2, v.ovvv) * 2
    l1new -= lib.einsum("oA,nh,genm,ohgA->em", t1, t1, l2, v.ovvv) * 2
    l1new += lib.einsum("nA,oh,genm,ohgA->em", t1, t1, l2, v.ovvv) * 4
    l1new += lib.einsum("oA,nh,egno,ghmA->em", t1, t1, l2, v.vvov)
    l1new += lib.einsum("nA,oh,egno,ghmA->em", t1, t1, l2, v.vvov) * -2
    l1new -= lib.einsum("mn,en->em", f.oo, l1)
    l1new += lib.einsum("pg,egno,pomn->em", t1, l2, v.oooo) * 2
    l1new += lib.einsum("pg,egno,pnmo->em", t1, l2, v.oooo) * -1
    l1new += lib.einsum("me->em", f.ov)
    l1new += lib.einsum("egno,gnmo->em", l2, v.vooo)
    l1new += lib.einsum("egno,gomn->em", l2, v.vooo) * -2
    l1new += lib.einsum("no,ng,egom->em", f.oo, t1, l2)
    l1new += lib.einsum("en,og,onmg->em", l1, t1, v.ooov)
    l1new += lib.einsum("gn,og,oemn->em", l1, t1, v.ovoo)
    l1new -= lib.einsum("no,ng,geom->em", f.oo, t1, l2) * 2
    l1new -= lib.einsum("en,og,ogmn->em", l1, t1, v.ovoo) * 2
    l1new -= lib.einsum("gn,og,onme->em", l1, t1, v.ooov) * 2
    l1new += lib.einsum("ng,ehom,ongh->em", f.ov, l2, t2)
    l1new += lib.einsum("gn,ongh,oemh->em", l1, t2, v.ovov)
    l1new += lib.einsum("og,ghnm,oehn->em", t1, l2, v.ovvo)
    l1new += lib.einsum("nh,egno,ghmo->em", t1, l2, v.vvoo)
    l1new += lib.einsum("oh,egnm,ongh->em", t1, l2, v.oovv)
    l1new += lib.einsum("oh,ghnm,geon->em", t1, l2, v.vvoo)
    l1new += lib.einsum("oh,egno,gnmh->em", t1, l2, v.voov)
    l1new -= lib.einsum("ng,ehom,nogh->em", f.ov, l2, t2) * 2
    l1new -= lib.einsum("ng,heom,ongh->em", f.ov, l2, t2) * 2
    l1new -= lib.einsum("gn,nogh,oemh->em", l1, t2, v.ovov) * 2
    l1new -= lib.einsum("gn,ongh,ohme->em", l1, t2, v.ovov) * 2
    l1new -= lib.einsum("og,ghnm,heon->em", t1, l2, v.vvoo) * 2
    l1new -= lib.einsum("nh,egno,gomh->em", t1, l2, v.voov) * 2
    l1new -= lib.einsum("oh,egnm,ohgn->em", t1, l2, v.ovvo) * 2
    l1new -= lib.einsum("oh,genm,ongh->em", t1, l2, v.oovv) * 2
    l1new -= lib.einsum("oh,ghnm,oegn->em", t1, l2, v.ovvo) * 2
    l1new -= lib.einsum("oh,egno,ghmn->em", t1, l2, v.vvoo) * 2
    l1new += lib.einsum("ng,heom,nogh->em", f.ov, l2, t2) * 4
    l1new += lib.einsum("gn,nogh,ohme->em", l1, t2, v.ovov) * 4
    l1new += lib.einsum("oh,genm,ohgn->em", t1, l2, v.ovvo) * 4
    l1new += lib.einsum("mg,ehno,ongh->em", f.ov, l2, t2)
    l1new += lib.einsum("ne,ghom,nogh->em", f.ov, l2, t2)
    l1new += lib.einsum("gm,ongh,neoh->em", l1, t2, v.ovov)
    l1new += lib.einsum("en,nogh,ogmh->em", l1, t2, v.ovov)
    l1new += lib.einsum("mg,ehno,nogh->em", f.ov, l2, t2) * -2
    l1new += lib.einsum("ne,ghom,ongh->em", f.ov, l2, t2) * -2
    l1new += lib.einsum("gm,nogh,neoh->em", l1, t2, v.ovov) * -2
    l1new += lib.einsum("en,ongh,ogmh->em", l1, t2, v.ovov) * -2
    l1new += lib.einsum("ng,og,nh,ehom->em", f.ov, t1, t1, l2)
    l1new += lib.einsum("gm,og,nh,neoh->em", l1, t1, t1, v.ovov)
    l1new += lib.einsum("en,ng,oh,ogmh->em", l1, t1, t1, v.ovov)
    l1new += lib.einsum("gn,og,nh,oemh->em", l1, t1, t1, v.ovov)
    l1new -= lib.einsum("ng,og,nh,heom->em", f.ov, t1, t1, l2) * 2
    l1new -= lib.einsum("gm,ng,oh,neoh->em", l1, t1, t1, v.ovov) * 2
    l1new -= lib.einsum("en,og,nh,ogmh->em", l1, t1, t1, v.ovov) * 2
    l1new -= lib.einsum("gn,og,nh,ohme->em", l1, t1, t1, v.ovov) * 2
    return {"l1new": l1new, "l2new": l2new}

