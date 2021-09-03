import numpy as np
import copy
try:
    from pyscf import lib
    einsum = lib.einsum
except ImportError:
    einsum = np.einsum

def mp2_energy(cc, T1, T2):
    ''' Calculate MP2 energy (this is the CC energy without the T1^2 contribution) '''

    # doubles contrib
    E = 0.25*einsum('abij,ijab->', T2, cc.I.oovv)
    # t1 contribution
    E += einsum('ai,ia->', T1, cc.fock_mo.ov)

    return E

def ccsd_energy(cc, T1, T2, autogen=False):
    ''' Calculate CCSD energy. 
        Equation generating script using the 'wick' program found in gen_eqns/ccsd_T.py
    '''

    if autogen:
        E = 1.0*einsum('ia,ai->', cc.fock_mo.ov, T1)
        E += 0.25*einsum('ijab,baji->', cc.I.oovv, T2)
        E += -0.5*einsum('ijab,bi,aj->', cc.I.oovv, T1, T1)
    else:
        # doubles contrib
        E = 0.25*einsum('abij,ijab->', T2, cc.I.oovv)
        # t1 contribution
        E += einsum('ai,ia->', T1, cc.fock_mo.ov)
        # t1**2 contribution
        E += 0.5*einsum('ai,bj,ijab->', T1, T1, cc.I.oovv)

    return E

def amp_updates_ccsd(cc, autogen=False):
    ''' Solve residual equations for CCSD.
        Equation generating script using the 'wick' program found in gen_eqns/ccsd_T.py
    '''

    # Copy fock matrix in MO basis, and remove diagonals
    F = copy.copy(cc.fock_mo)
    F.oo = F.oo - np.diag(cc.eo)
    F.vv = F.vv - np.diag(cc.ev)

    I = cc.I

    T1old = cc.T1old
    T2old = cc.T2old

    # Update T1
    
    if autogen:
        T1 = 1.0*einsum('ai->ai', F.vo)
        T1 += -1.0*einsum('ji,aj->ai', F.oo, T1old)
        T1 += 1.0*einsum('ab,bi->ai', F.vv, T1old)
        T1 += -1.0*einsum('jb,abji->ai', F.ov, T2old)
        T1 += -1.0*einsum('jaib,bj->ai', I.ovov, T1old)
        T1 += 0.5*einsum('jkib,abkj->ai', I.ooov, T2old)
        T1 += -0.5*einsum('jabc,cbji->ai', I.ovvv, T2old)
        T1 += -1.0*einsum('jb,bi,aj->ai', F.ov, T1old, T1old)
        T1 += -1.0*einsum('jkib,aj,bk->ai', I.ooov, T1old, T1old)
        T1 += 1.0*einsum('jabc,ci,bj->ai', I.ovvv, T1old, T1old)
        T1 += -0.5*einsum('jkbc,aj,cbki->ai', I.oovv, T1old, T2old)
        T1 += -0.5*einsum('jkbc,ci,abkj->ai', I.oovv, T1old, T2old)
        T1 += 1.0*einsum('jkbc,cj,abki->ai', I.oovv, T1old, T2old)
        T1 += 1.0*einsum('jkbc,ci,aj,bk->ai', I.oovv, T1old, T1old, T1old)
    else:
        T1 = F.vo.copy()
        # _S_S
        T1 += einsum('ab,bi->ai', F.vv, T1old)
        T1 -= einsum('ji,aj->ai', F.oo, T1old)
        T1 -= einsum('ajbi,bj->ai', I.vovo, T1old)
        # _S_D
        T1 += einsum('jb,abij->ai', F.ov, T2old)
        T1 += 0.5*einsum('ajbc,bcij->ai', I.vovv, T2old)
        T1 -= 0.5*einsum('jkib,abjk->ai', I.ooov, T2old)
        # _S_SS
        T1 -= einsum('jb,bi,aj->ai', F.ov, T1old, T1old)
        T1 -= einsum('ajbc,bj,ci->ai', I.vovv, T1old, T1old)
        T1 += einsum('jkib,bj,ak->ai', I.ooov, T1old, T1old)
        # _S_SD
        T1 -= 0.5*einsum('jkbc,bi,acjk->ai', I.oovv, T1old, T2old)
        T1 -= 0.5*einsum('jkbc,aj,bcik->ai', I.oovv, T1old, T2old)
        T1 += einsum('jkbc,bj,caki->ai', I.oovv, T1old, T2old)
        # _S_SSS
        T1 += einsum('jkbc,bi,cj,ak->ai', I.oovv, T1old, T1old, T1old)

    # Update T2

    if autogen:

        T2 = 1.0*einsum('baji->abij', I.vvoo)
        T2 += 1.0*einsum('ki,bakj->abij', F.oo, T2old)
        T2 += -1.0*einsum('kj,baki->abij', F.oo, T2old)
        T2 += 1.0*einsum('ac,bcji->abij', F.vv, T2old)
        T2 += -1.0*einsum('bc,acji->abij', F.vv, T2old)
        T2 += -1.0*einsum('kaji,bk->abij', I.ovoo, T1old)
        T2 += 1.0*einsum('kbji,ak->abij', I.ovoo, T1old)
        T2 += -1.0*einsum('baic,cj->abij', I.vvov, T1old)
        T2 += 1.0*einsum('bajc,ci->abij', I.vvov, T1old)
        T2 += -0.5*einsum('klji,balk->abij', I.oooo, T2old)
        T2 += 1.0*einsum('kaic,bckj->abij', I.ovov, T2old)
        T2 += -1.0*einsum('kajc,bcki->abij', I.ovov, T2old)
        T2 += -1.0*einsum('kbic,ackj->abij', I.ovov, T2old)
        T2 += 1.0*einsum('kbjc,acki->abij', I.ovov, T2old)
        T2 += -0.5*einsum('bacd,dcji->abij', I.vvvv, T2old)
        T2 += -1.0*einsum('kc,ak,bcji->abij', F.ov, T1old, T2old)
        T2 += 1.0*einsum('kc,bk,acji->abij', F.ov, T1old, T2old)
        T2 += 1.0*einsum('kc,ci,bakj->abij', F.ov, T1old, T2old)
        T2 += -1.0*einsum('kc,cj,baki->abij', F.ov, T1old, T2old)
        T2 += 1.0*einsum('klji,bk,al->abij', I.oooo, T1old, T1old)
        T2 += 1.0*einsum('kaic,cj,bk->abij', I.ovov, T1old, T1old)
        T2 += -1.0*einsum('kajc,ci,bk->abij', I.ovov, T1old, T1old)
        T2 += -1.0*einsum('kbic,cj,ak->abij', I.ovov, T1old, T1old)
        T2 += 1.0*einsum('kbjc,ci,ak->abij', I.ovov, T1old, T1old)
        T2 += 1.0*einsum('bacd,di,cj->abij', I.vvvv, T1old, T1old)
        T2 += 1.0*einsum('klic,ak,bclj->abij', I.ooov, T1old, T2old)
        T2 += -1.0*einsum('klic,bk,aclj->abij', I.ooov, T1old, T2old)
        T2 += 0.5*einsum('klic,cj,balk->abij', I.ooov, T1old, T2old)
        T2 += -1.0*einsum('klic,ck,balj->abij', I.ooov, T1old, T2old)
        T2 += -1.0*einsum('kljc,ak,bcli->abij', I.ooov, T1old, T2old)
        T2 += 1.0*einsum('kljc,bk,acli->abij', I.ooov, T1old, T2old)
        T2 += -0.5*einsum('kljc,ci,balk->abij', I.ooov, T1old, T2old)
        T2 += 1.0*einsum('kljc,ck,bali->abij', I.ooov, T1old, T2old)
        T2 += 0.5*einsum('kacd,bk,dcji->abij', I.ovvv, T1old, T2old)
        T2 += -1.0*einsum('kacd,di,bckj->abij', I.ovvv, T1old, T2old)
        T2 += 1.0*einsum('kacd,dj,bcki->abij', I.ovvv, T1old, T2old)
        T2 += -1.0*einsum('kacd,dk,bcji->abij', I.ovvv, T1old, T2old)
        T2 += -0.5*einsum('kbcd,ak,dcji->abij', I.ovvv, T1old, T2old)
        T2 += 1.0*einsum('kbcd,di,ackj->abij', I.ovvv, T1old, T2old)
        T2 += -1.0*einsum('kbcd,dj,acki->abij', I.ovvv, T1old, T2old)
        T2 += 1.0*einsum('kbcd,dk,acji->abij', I.ovvv, T1old, T2old)
        T2 += 0.5*einsum('klcd,adji,bclk->abij', I.oovv, T2old, T2old)
        T2 += -1.0*einsum('klcd,adki,bclj->abij', I.oovv, T2old, T2old)
        T2 += -0.5*einsum('klcd,baki,dclj->abij', I.oovv, T2old, T2old)
        T2 += -0.5*einsum('klcd,bdji,aclk->abij', I.oovv, T2old, T2old)
        T2 += 1.0*einsum('klcd,bdki,aclj->abij', I.oovv, T2old, T2old)
        T2 += 0.25*einsum('klcd,dcji,balk->abij', I.oovv, T2old, T2old)
        T2 += -0.5*einsum('klcd,dcki,balj->abij', I.oovv, T2old, T2old)
        T2 += -1.0*einsum('klic,cj,bk,al->abij', I.ooov, T1old, T1old, T1old)
        T2 += 1.0*einsum('kljc,ci,bk,al->abij', I.ooov, T1old, T1old, T1old)
        T2 += -1.0*einsum('kacd,di,cj,bk->abij', I.ovvv, T1old, T1old, T1old)
        T2 += 1.0*einsum('kbcd,di,cj,ak->abij', I.ovvv, T1old, T1old, T1old)
        T2 += -1.0*einsum('klcd,ak,dl,bcji->abij', I.oovv, T1old, T1old, T2old)
        T2 += -0.5*einsum('klcd,bk,al,dcji->abij', I.oovv, T1old, T1old, T2old)
        T2 += 1.0*einsum('klcd,bk,dl,acji->abij', I.oovv, T1old, T1old, T2old)
        T2 += -1.0*einsum('klcd,di,ak,bclj->abij', I.oovv, T1old, T1old, T2old)
        T2 += 1.0*einsum('klcd,di,bk,aclj->abij', I.oovv, T1old, T1old, T2old)
        T2 += -0.5*einsum('klcd,di,cj,balk->abij', I.oovv, T1old, T1old, T2old)
        T2 += 1.0*einsum('klcd,di,ck,balj->abij', I.oovv, T1old, T1old, T2old)
        T2 += 1.0*einsum('klcd,dj,ak,bcli->abij', I.oovv, T1old, T1old, T2old)
        T2 += -1.0*einsum('klcd,dj,bk,acli->abij', I.oovv, T1old, T1old, T2old)
        T2 += -1.0*einsum('klcd,dj,ck,bali->abij', I.oovv, T1old, T1old, T2old)
        T2 += 1.0*einsum('klcd,di,cj,bk,al->abij', I.oovv, T1old, T1old, T1old, T1old)

    else:
        T2 = I.vvoo.copy()
        # _D_S
        T2 += einsum('abcj,ci->abij', I.vvvo, T1old)
        T2 -= einsum('abci,cj->abij', I.vvvo, T1old)
        T2 += einsum('bkij,ak->abij', I.vooo, T1old)
        T2 -= einsum('akij,bk->abij', I.vooo, T1old)
        # _D_D
        T2 += einsum('bc,acij->abij', F.vv, T2old)
        T2 -= einsum('ac,bcij->abij', F.vv, T2old)
        T2 -= einsum('kj,abik->abij', F.oo, T2old)
        T2 += einsum('ki,abjk->abij', F.oo, T2old)
        T2 += 0.5*einsum('abcd,cdij->abij', I.vvvv, T2old)
        T2 += 0.5*einsum('klij,abkl->abij', I.oooo, T2old)
        T2 -= einsum('bkcj,acik->abij', I.vovo, T2old)
        T2 += einsum('akcj,bcik->abij', I.vovo, T2old)
        T2 += einsum('bkci,acjk->abij', I.vovo, T2old)
        T2 -= einsum('akci,bcjk->abij', I.vovo, T2old)
        # _D_SS
        T2 += 0.5*einsum('abcd,ci,dj->abij', I.vvvv, T1old, T1old)
        T2 -= 0.5*einsum('abcd,cj,di->abij', I.vvvv, T1old, T1old)
        T2 += 0.5*einsum('klij,ak,bl->abij', I.oooo, T1old, T1old)
        T2 -= 0.5*einsum('klij,bk,al->abij', I.oooo, T1old, T1old)
        T2 -= einsum('akcj,ci,bk->abij', I.vovo, T1old, T1old)
        T2 += einsum('bkcj,ci,ak->abij', I.vovo, T1old, T1old)
        T2 += einsum('akci,cj,bk->abij', I.vovo, T1old, T1old)
        T2 -= einsum('bkci,cj,ak->abij', I.vovo, T1old, T1old)
        # _D_SD
        T2 -= einsum('kc,ci,abkj->abij', F.ov, T1old, T2old)
        T2 += einsum('kc,cj,abki->abij', F.ov, T1old, T2old)
        T2 -= einsum('kc,ak,cbij->abij', F.ov, T1old, T2old)
        T2 += einsum('kc,bk,caij->abij', F.ov, T1old, T2old)
        T2 -= einsum('akcd,ck,dbij->abij', I.vovv, T1old, T2old)
        T2 += einsum('bkcd,ck,daij->abij', I.vovv, T1old, T2old)
        T2 += einsum('klic,ck,ablj->abij', I.ooov, T1old, T2old)
        T2 -= einsum('kljc,ck,abli->abij', I.ooov, T1old, T2old)
        T2 += einsum('akcd,ci,dbkj->abij', I.vovv, T1old, T2old)
        T2 -= einsum('bkcd,ci,dakj->abij', I.vovv, T1old, T2old)
        T2 -= einsum('akcd,cj,dbki->abij', I.vovv, T1old, T2old)
        T2 += einsum('bkcd,cj,daki->abij', I.vovv, T1old, T2old)
        T2 -= einsum('klic,ak,cblj->abij', I.ooov, T1old, T2old)
        T2 += einsum('klic,bk,calj->abij', I.ooov, T1old, T2old)
        T2 += einsum('kljc,ak,cbli->abij', I.ooov, T1old, T2old)
        T2 -= einsum('kljc,bk,cali->abij', I.ooov, T1old, T2old)
        T2 -= 0.5*einsum('kljc,ci,abkl->abij', I.ooov, T1old, T2old)
        T2 += 0.5*einsum('klic,cj,abkl->abij', I.ooov, T1old, T2old)
        T2 += 0.5*einsum('bkcd,ak,cdij->abij', I.vovv, T1old, T2old)
        T2 -= 0.5*einsum('akcd,bk,cdij->abij', I.vovv, T1old, T2old)
        # _D_DD
        T2 += 0.25*einsum('klcd,cdij,abkl->abij', I.oovv, T2old, T2old)
        T2 += einsum('klcd,acik,dblj->abij', I.oovv, T2old, T2old)
        T2 -= einsum('klcd,bcik,dalj->abij', I.oovv, T2old, T2old)
        T2 -= 0.5*einsum('klcd,cakl,dbij->abij', I.oovv, T2old, T2old)
        T2 += 0.5*einsum('klcd,cbkl,daij->abij', I.oovv, T2old, T2old)
        T2 -= 0.5*einsum('klcd,cdki,ablj->abij', I.oovv, T2old, T2old)
        T2 += 0.5*einsum('klcd,cdkj,abli->abij', I.oovv, T2old, T2old)
        # _D_SSS
        T2 += einsum('bkcd,ci,ak,dj->abij', I.vovv, T1old, T1old, T1old)
        T2 -= einsum('akcd,ci,bk,dj->abij', I.vovv, T1old, T1old, T1old)
        T2 -= einsum('kljc,ci,ak,bl->abij', I.ooov, T1old, T1old, T1old)
        T2 += einsum('klic,cj,ak,bl->abij', I.ooov, T1old, T1old, T1old)
        # _D_SSD
        T2 += 0.25*einsum('klcd,ci,dj,abkl->abij', I.oovv, T1old, T1old, T2old)
        T2 -= 0.25*einsum('klcd,cj,di,abkl->abij', I.oovv, T1old, T1old, T2old)
        T2 += 0.25*einsum('klcd,ak,bl,cdij->abij', I.oovv, T1old, T1old, T2old)
        T2 -= 0.25*einsum('klcd,bk,al,cdij->abij', I.oovv, T1old, T1old, T2old)
        T2 -= einsum('klcd,ci,ak,dblj->abij', I.oovv, T1old, T1old, T2old)
        T2 += einsum('klcd,ci,bk,dalj->abij', I.oovv, T1old, T1old, T2old)
        T2 += einsum('klcd,cj,ak,dbli->abij', I.oovv, T1old, T1old, T2old)
        T2 -= einsum('klcd,cj,bk,dali->abij', I.oovv, T1old, T1old, T2old)
        T2 -= einsum('klcd,ck,di,ablj->abij', I.oovv, T1old, T1old, T2old)
        T2 += einsum('klcd,ck,dj,abli->abij', I.oovv, T1old, T1old, T2old)
        T2 -= einsum('klcd,ck,al,dbij->abij', I.oovv, T1old, T1old, T2old)
        T2 += einsum('klcd,ck,bl,daij->abij', I.oovv, T1old, T1old, T2old)
        # _D_SSSS
        T2 += einsum('klcd,ci,ak,bl,dj->abij', I.oovv, T1old, T1old, T1old, T1old)

    return T1, T2
