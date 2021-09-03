import numpy as np
import copy
try:
    from pyscf import lib
    einsum = lib.einsum
except ImportError:
    einsum = np.einsum

def ccsd_1_1_energy(cc, T1, T2, S1, U11):
    ''' Calculate CCSD-1-1 energy. 
        Equation generating script using the 'wick' program found in gen_eqns/ccsd_11_amps.py
    '''

    # Fermionic part
    # doubles contrib
    E = 0.25*einsum('abij,ijab->', T2, cc.I.oovv)
    # t1 contribution
    E += einsum('ai,ia->', T1, cc.fock_mo.ov)
    # t1**2 contribution
    E += 0.5*einsum('ai,bj,ijab->', T1, T1, cc.I.oovv)

    # Bosonic and coupling part
    E += einsum('I,I->',cc.G,S1)
    E += einsum('Iia,Iai->',cc.g_mo_blocks.ov, U11)
    E += einsum('Iia,ai,I->',cc.g_mo_blocks.ov, T1, S1)

    return E

def amp_updates_ccsd_1_1(cc, autogen=False):
    ''' Solve residual equations for CCSD.
        Equation generating script using the 'wick' program found in gen_eqns/ccsd_11_amps.py
    '''

    # Copy fock matrix in MO basis, and remove diagonals
    F = copy.copy(cc.fock_mo)
    F.oo = F.oo - np.diag(cc.eo)
    F.vv = F.vv - np.diag(cc.ev)

    I = cc.I

    T1old = cc.T1old
    T2old = cc.T2old
    S1old = cc.S1old
    U11old = cc.U11old

    g = copy.copy(cc.g_mo_blocks)
    h = copy.copy(cc.g_mo_blocks)
    G = cc.G.copy()
    H = cc.G.copy()

    # Note that we also have to remove diagonals from omega
    # matrix. Since the rest of the code is constrained to have
    # no coupling between bosons, this means that this is actually
    # the zero matrix, but we keep it in here anyway for potential
    # future applications
    w = np.diag(cc.omega).copy()
    w = w - np.diag(np.diag(w))

    # Update T1, T2, S1 and U11
    if autogen:
        if True: 
            # My code generation
            T1 = 1.0*einsum('ai->ai', F.vo)
            T1 += 1.0*einsum('I,Iai->ai', G, U11old)
            T1 += -1.0*einsum('ji,aj->ai', F.oo, T1old)
            T1 += 1.0*einsum('ab,bi->ai', F.vv, T1old)
            T1 += 1.0*einsum('Iai,I->ai', g.vo, S1old)
            T1 += -1.0*einsum('jb,abji->ai', F.ov, T2old)
            T1 += -1.0*einsum('Iji,Iaj->ai', g.oo, U11old)
            T1 += 1.0*einsum('Iab,Ibi->ai', g.vv, U11old)
            T1 += -1.0*einsum('jaib,bj->ai', I.ovov, T1old)
            T1 += 0.5*einsum('jkib,abkj->ai', I.ooov, T2old)
            T1 += -0.5*einsum('jabc,cbji->ai', I.ovvv, T2old)
            T1 += -1.0*einsum('jb,bi,aj->ai', F.ov, T1old, T1old)
            T1 += -1.0*einsum('Iji,aj,I->ai', g.oo, T1old, S1old)
            T1 += 1.0*einsum('Iab,bi,I->ai', g.vv, T1old, S1old)
            T1 += -1.0*einsum('Ijb,aj,Ibi->ai', g.ov, T1old, U11old)
            T1 += -1.0*einsum('Ijb,bi,Iaj->ai', g.ov, T1old, U11old)
            T1 += 1.0*einsum('Ijb,bj,Iai->ai', g.ov, T1old, U11old)
            T1 += -1.0*einsum('Ijb,abji,I->ai', g.ov, T2old, S1old)
            T1 += -1.0*einsum('jkib,aj,bk->ai', I.ooov, T1old, T1old)
            T1 += 1.0*einsum('jabc,ci,bj->ai', I.ovvv, T1old, T1old)
            T1 += -0.5*einsum('jkbc,aj,cbki->ai', I.oovv, T1old, T2old)
            T1 += -0.5*einsum('jkbc,ci,abkj->ai', I.oovv, T1old, T2old)
            T1 += 1.0*einsum('jkbc,cj,abki->ai', I.oovv, T1old, T2old)
            T1 += -1.0*einsum('Ijb,bi,aj,I->ai', g.ov, T1old, T1old, S1old)
            T1 += 1.0*einsum('jkbc,ci,aj,bk->ai', I.oovv, T1old, T1old, T1old)

            T2 = 1.0*einsum('baji->abij', I.vvoo)
            T2 += 1.0*einsum('ki,bakj->abij', F.oo, T2old)
            T2 += -1.0*einsum('kj,baki->abij', F.oo, T2old)
            T2 += 1.0*einsum('ac,bcji->abij', F.vv, T2old)
            T2 += -1.0*einsum('bc,acji->abij', F.vv, T2old)
            T2 += 1.0*einsum('Iai,Ibj->abij', g.vo, U11old)
            T2 += -1.0*einsum('Iaj,Ibi->abij', g.vo, U11old)
            T2 += -1.0*einsum('Ibi,Iaj->abij', g.vo, U11old)
            T2 += 1.0*einsum('Ibj,Iai->abij', g.vo, U11old)
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
            T2 += -1.0*einsum('Iki,ak,Ibj->abij', g.oo, T1old, U11old)
            T2 += 1.0*einsum('Iki,bk,Iaj->abij', g.oo, T1old, U11old)
            T2 += 1.0*einsum('Iki,bakj,I->abij', g.oo, T2old, S1old)
            T2 += 1.0*einsum('Ikj,ak,Ibi->abij', g.oo, T1old, U11old)
            T2 += -1.0*einsum('Ikj,bk,Iai->abij', g.oo, T1old, U11old)
            T2 += -1.0*einsum('Ikj,baki,I->abij', g.oo, T2old, S1old)
            T2 += 1.0*einsum('Iac,ci,Ibj->abij', g.vv, T1old, U11old)
            T2 += -1.0*einsum('Iac,cj,Ibi->abij', g.vv, T1old, U11old)
            T2 += 1.0*einsum('Iac,bcji,I->abij', g.vv, T2old, S1old)
            T2 += -1.0*einsum('Ibc,ci,Iaj->abij', g.vv, T1old, U11old)
            T2 += 1.0*einsum('Ibc,cj,Iai->abij', g.vv, T1old, U11old)
            T2 += -1.0*einsum('Ibc,acji,I->abij', g.vv, T2old, S1old)
            T2 += 1.0*einsum('klji,bk,al->abij', I.oooo, T1old, T1old)
            T2 += 1.0*einsum('kaic,cj,bk->abij', I.ovov, T1old, T1old)
            T2 += -1.0*einsum('kajc,ci,bk->abij', I.ovov, T1old, T1old)
            T2 += -1.0*einsum('kbic,cj,ak->abij', I.ovov, T1old, T1old)
            T2 += 1.0*einsum('kbjc,ci,ak->abij', I.ovov, T1old, T1old)
            T2 += 1.0*einsum('bacd,di,cj->abij', I.vvvv, T1old, T1old)
            T2 += 1.0*einsum('Ikc,acji,Ibk->abij', g.ov, T2old, U11old)
            T2 += -1.0*einsum('Ikc,acki,Ibj->abij', g.ov, T2old, U11old)
            T2 += 1.0*einsum('Ikc,ackj,Ibi->abij', g.ov, T2old, U11old)
            T2 += -1.0*einsum('Ikc,baki,Icj->abij', g.ov, T2old, U11old)
            T2 += 1.0*einsum('Ikc,bakj,Ici->abij', g.ov, T2old, U11old)
            T2 += -1.0*einsum('Ikc,bcji,Iak->abij', g.ov, T2old, U11old)
            T2 += 1.0*einsum('Ikc,bcki,Iaj->abij', g.ov, T2old, U11old)
            T2 += -1.0*einsum('Ikc,bckj,Iai->abij', g.ov, T2old, U11old)
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
            T2 += -1.0*einsum('Ikc,ak,bcji,I->abij', g.ov, T1old, T2old, S1old)
            T2 += 1.0*einsum('Ikc,bk,acji,I->abij', g.ov, T1old, T2old, S1old)
            T2 += -1.0*einsum('Ikc,ci,ak,Ibj->abij', g.ov, T1old, T1old, U11old)
            T2 += 1.0*einsum('Ikc,ci,bk,Iaj->abij', g.ov, T1old, T1old, U11old)
            T2 += 1.0*einsum('Ikc,ci,bakj,I->abij', g.ov, T1old, T2old, S1old)
            T2 += 1.0*einsum('Ikc,cj,ak,Ibi->abij', g.ov, T1old, T1old, U11old)
            T2 += -1.0*einsum('Ikc,cj,bk,Iai->abij', g.ov, T1old, T1old, U11old)
            T2 += -1.0*einsum('Ikc,cj,baki,I->abij', g.ov, T1old, T2old, S1old)
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
            
            S1 = G.copy() 
            S1 += 1.0*einsum('IJ,J->I', w, S1old)
            S1 += 1.0*einsum('ia,Iai->I', F.ov, U11old)
            S1 += 1.0*einsum('Iia,ai->I', g.ov, T1old)
            S1 += 1.0*einsum('Jia,J,Iai->I', g.ov, S1old, U11old)
            S1 += -1.0*einsum('ijab,bi,Iaj->I', I.oovv, T1old, U11old)

            U11 = g.vo.copy()
            U11 += -1.0*einsum('ji,Iaj->Iai', F.oo, U11old)
            U11 += 1.0*einsum('ab,Ibi->Iai', F.vv, U11old)
            U11 += 1.0*einsum('IJ,Jai->Iai', w, U11old)
            U11 += -1.0*einsum('Iji,aj->Iai', g.oo, T1old)
            U11 += 1.0*einsum('Iab,bi->Iai', g.vv, T1old)
            U11 += -1.0*einsum('Ijb,abji->Iai', g.ov, T2old)
            U11 += -1.0*einsum('jaib,Ibj->Iai', I.ovov, U11old)
            U11 += -1.0*einsum('jb,aj,Ibi->Iai', F.ov, T1old, U11old)
            U11 += -1.0*einsum('jb,bi,Iaj->Iai', F.ov, T1old, U11old)
            U11 += -1.0*einsum('Ijb,bi,aj->Iai', g.ov, T1old, T1old)
            U11 += -1.0*einsum('Jji,J,Iaj->Iai', g.oo, S1old, U11old)
            U11 += 1.0*einsum('Jab,J,Ibi->Iai', g.vv, S1old, U11old)
            U11 += -1.0*einsum('Jjb,Iaj,Jbi->Iai', g.ov, U11old, U11old)
            U11 += -1.0*einsum('Jjb,Ibi,Jaj->Iai', g.ov, U11old, U11old)
            U11 += 1.0*einsum('Jjb,Ibj,Jai->Iai', g.ov, U11old, U11old)
            U11 += -1.0*einsum('jkib,aj,Ibk->Iai', I.ooov, T1old, U11old)
            U11 += 1.0*einsum('jkib,bj,Iak->Iai', I.ooov, T1old, U11old)
            U11 += 1.0*einsum('jabc,ci,Ibj->Iai', I.ovvv, T1old, U11old)
            U11 += -1.0*einsum('jabc,cj,Ibi->Iai', I.ovvv, T1old, U11old)
            U11 += 1.0*einsum('jkbc,acji,Ibk->Iai', I.oovv, T2old, U11old)
            U11 += 0.5*einsum('jkbc,ackj,Ibi->Iai', I.oovv, T2old, U11old)
            U11 += 0.5*einsum('jkbc,cbji,Iak->Iai', I.oovv, T2old, U11old)
            U11 += -1.0*einsum('Jjb,aj,J,Ibi->Iai', g.ov, T1old, S1old, U11old)
            U11 += -1.0*einsum('Jjb,bi,J,Iaj->Iai', g.ov, T1old, S1old, U11old)
            U11 += -1.0*einsum('jkbc,aj,ck,Ibi->Iai', I.oovv, T1old, T1old, U11old)
            U11 += 1.0*einsum('jkbc,ci,aj,Ibk->Iai', I.oovv, T1old, T1old, U11old)
            U11 += -1.0*einsum('jkbc,ci,bj,Iak->Iai', I.oovv, T1old, T1old, U11old)
        else:
            # Autogen from epcc
            # CCSD equations
            #T1,T2 = cc_equations.ccsd_simple(F, I, T1old, T2old)
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

            # F1
            T1 += 1.0*einsum('I,Iai->ai', G, U11old)
            T1 += 1.0*einsum('Iai,I->ai', g.vo, S1old)
            T1 += -1.0*einsum('Iji,Iaj->ai', g.oo, U11old)
            T1 += 1.0*einsum('Iab,Ibi->ai', g.vv, U11old)
            T1 += -1.0*einsum('Iji,aj,I->ai', g.oo, T1old, S1old)
            T1 += 1.0*einsum('Iab,bi,I->ai', g.vv, T1old, S1old)
            T1 += -1.0*einsum('Ijb,bi,Iaj->ai', g.ov, T1old, U11old)
            T1 += -1.0*einsum('Ijb,aj,Ibi->ai', g.ov, T1old, U11old)
            T1 += 1.0*einsum('Ijb,bj,Iai->ai', g.ov, T1old, U11old)
            T1 += 1.0*einsum('Ijb,abij,I->ai', g.ov, T2old, S1old)
            T1 += -1.0*einsum('Ijb,bi,aj,I->ai', g.ov, T1old, T1old, S1old)

            # F2
            T2 += 1.0*einsum('Iai,Ibj->abij', g.vo, U11old)
            T2 += -1.0*einsum('Ibi,Iaj->abij', g.vo, U11old)
            T2 += -1.0*einsum('Iaj,Ibi->abij', g.vo, U11old)
            T2 += 1.0*einsum('Ibj,Iai->abij', g.vo, U11old)
            T2 += -1.0*einsum('Iki,ak,Ibj->abij', g.oo, T1old, U11old)
            T2 += 1.0*einsum('Iki,bk,Iaj->abij', g.oo, T1old, U11old)
            T2 += 1.0*einsum('Ikj,ak,Ibi->abij', g.oo, T1old, U11old)
            T2 += -1.0*einsum('Ikj,bk,Iai->abij', g.oo, T1old, U11old)
            T2 += 1.0*einsum('Iki,abjk,I->abij', g.oo, T2old, S1old)
            T2 += -1.0*einsum('Ikj,abik,I->abij', g.oo, T2old, S1old)
            T2 += 1.0*einsum('Iac,ci,Ibj->abij', g.vv, T1old, U11old)
            T2 += -1.0*einsum('Ibc,ci,Iaj->abij', g.vv, T1old, U11old)
            T2 += -1.0*einsum('Iac,cj,Ibi->abij', g.vv, T1old, U11old)
            T2 += 1.0*einsum('Ibc,cj,Iai->abij', g.vv, T1old, U11old)
            T2 += -1.0*einsum('Iac,bcij,I->abij', g.vv, T2old, S1old)
            T2 += 1.0*einsum('Ibc,acij,I->abij', g.vv, T2old, S1old)
            T2 += -1.0*einsum('Ikc,acij,Ibk->abij', g.ov, T2old, U11old)
            T2 += -1.0*einsum('Ikc,abik,Icj->abij', g.ov, T2old, U11old)
            T2 += 1.0*einsum('Ikc,acik,Ibj->abij', g.ov, T2old, U11old)
            T2 += 1.0*einsum('Ikc,bcij,Iak->abij', g.ov, T2old, U11old)
            T2 += -1.0*einsum('Ikc,bcik,Iaj->abij', g.ov, T2old, U11old)
            T2 += 1.0*einsum('Ikc,abjk,Ici->abij', g.ov, T2old, U11old)
            T2 += -1.0*einsum('Ikc,acjk,Ibi->abij', g.ov, T2old, U11old)
            T2 += 1.0*einsum('Ikc,bcjk,Iai->abij', g.ov, T2old, U11old)
            T2 += -1.0*einsum('Ikc,ci,ak,Ibj->abij', g.ov, T1old, T1old, U11old)
            T2 += 1.0*einsum('Ikc,ci,bk,Iaj->abij', g.ov, T1old, T1old, U11old)
            T2 += 1.0*einsum('Ikc,ak,cj,Ibi->abij', g.ov, T1old, T1old, U11old)
            T2 += -1.0*einsum('Ikc,cj,bk,Iai->abij', g.ov, T1old, T1old, U11old)
            T2 += 1.0*einsum('Ikc,ci,abjk,I->abij', g.ov, T1old, T2old, S1old)
            T2 += 1.0*einsum('Ikc,ak,bcij,I->abij', g.ov, T1old, T2old, S1old)
            T2 += -1.0*einsum('Ikc,cj,abik,I->abij', g.ov, T1old, T2old, S1old)
            T2 += -1.0*einsum('Ikc,bk,acij,I->abij', g.ov, T1old, T2old, S1old)

            # S1 equation
            S1 = H.copy()
            #S1 += 1.0*einsum('I,I->I', w, S1)
            S1 += 1.0*einsum('ia,Iai->I', F.ov, U11old)
            S1 += 1.0*einsum('Iia,ai->I', h.ov, T1old)
            S1 += 1.0*einsum('Jia,J,Iai->I', g.ov, S1old, U11old)
            S1 += -1.0*einsum('ijab,bi,Iaj->I', I.oovv, T1old, U11old)

            # U11 equations
            U11 = h.vo.copy()
            #U11 += 1.0*einsum('I,Iai->Iai', w, U11)
            U11 += -1.0*einsum('ji,Iaj->Iai', F.oo, U11old)
            U11 += 1.0*einsum('ab,Ibi->Iai', F.vv, U11old)
            U11 += -1.0*einsum('Iji,aj->Iai', h.oo, T1old)
            U11 += 1.0*einsum('Iab,bi->Iai', h.vv, T1old)
            U11 += -1.0*einsum('ajbi,Ibj->Iai', I.vovo, U11old)
            U11 += 1.0*einsum('Ijb,abij->Iai', h.ov, T2old)
            U11 += -1.0*einsum('jb,bi,Iaj->Iai', F.ov, T1old, U11old)
            U11 += -1.0*einsum('jb,aj,Ibi->Iai', F.ov, T1old, U11old)
            U11 += -1.0*einsum('Jji,J,Iaj->Iai', g.oo, S1old, U11old)
            U11 += -1.0*einsum('Ijb,bi,aj->Iai', h.ov, T1old, T1old)
            U11 += 1.0*einsum('Jab,J,Ibi->Iai', g.vv, S1old, U11old)
            U11 += -1.0*einsum('jkib,aj,Ibk->Iai', I.ooov, T1old, U11old)
            U11 += 1.0*einsum('jkib,bj,Iak->Iai', I.ooov, T1old, U11old)
            U11 += -1.0*einsum('ajbc,ci,Ibj->Iai', I.vovv, T1old, U11old)
            U11 += 1.0*einsum('ajbc,cj,Ibi->Iai', I.vovv, T1old, U11old)
            U11 += -1.0*einsum('Jjb,Ibi,Jaj->Iai', g.ov, U11old, U11old)
            U11 += -1.0*einsum('Jjb,Iaj,Jbi->Iai', g.ov, U11old, U11old)
            U11 += 1.0*einsum('Jjb,Ibj,Jai->Iai', g.ov, U11old, U11old)
            U11 += -1.0*einsum('jkbc,acij,Ibk->Iai', I.oovv, T2old, U11old)
            U11 += -0.5*einsum('jkbc,cbij,Iak->Iai', I.oovv, T2old, U11old)
            U11 += -0.5*einsum('jkbc,acjk,Ibi->Iai', I.oovv, T2old, U11old)
            U11 += -1.0*einsum('Jjb,bi,J,Iaj->Iai', g.ov, T1old, S1old, U11old)
            U11 += -1.0*einsum('Jjb,aj,J,Ibi->Iai', g.ov, T1old, S1old, U11old)
            U11 += 1.0*einsum('jkbc,ci,aj,Ibk->Iai', I.oovv, T1old, T1old, U11old)
            U11 += -1.0*einsum('jkbc,ci,bj,Iak->Iai', I.oovv, T1old, T1old, U11old)
            U11 += -1.0*einsum('jkbc,aj,ck,Ibi->Iai', I.oovv, T1old, T1old, U11old)

        
    else:
        T1 = F.vo.copy()
        T1 += einsum('xai,x->ai', g.vo, S1old)
        T1 += einsum('x,xai->ai',G, U11old)

        T2 = I.vvoo.copy()
        abij_temp = einsum('xai,xbj->abij', g.vo, U11old)
        T2 += abij_temp
        T2 -= abij_temp.transpose((1,0,2,3))
        T2 -= abij_temp.transpose((0,1,3,2))
        T2 += abij_temp.transpose((1,0,3,2))
        abij_temp = None

        T2A = T2old.copy()
        T2A += 0.5*einsum('ai,bj->abij',T1old,T1old)
        T2A -= 0.5*einsum('bi,aj->abij',T1old,T1old)

        Fvv = F.vv.copy()
        Fvv += einsum('xab,x->ab', g.vv, S1old)
        Fvv -= 0.5*einsum('jb,aj->ab',F.ov,T1old)
        Fvv -= einsum('ajcb,cj->ab',I.vovv,T1old)
        Fvv -= 0.5*einsum('jkbc,acjk->ab',I.oovv,T2A)
        Fvv -= einsum('yjb,yaj->ab', g.ov, U11old)

        Foo = F.oo.copy()
        Foo += einsum('xji,x->ji', g.oo, S1old)
        Foo += 0.5*einsum('jb,bi->ji',F.ov,T1old)
        Foo += einsum('jkib,bk->ji',I.ooov,T1old)
        Foo += 0.5*einsum('jkbc,bcik->ji',I.oovv,T2A)
        Foo += einsum('yjb,ybi->ji', g.ov, U11old)
        T2A = None

        gsov = einsum('yia,y->ia', g.ov, S1old)
        gtm = einsum('xjb,bj->x', g.ov, T1old)
        Fov = F.ov.copy()
        Fov += einsum('jkbc,ck->jb',I.oovv,T1old)
        Fov += gsov

        T1 += einsum('ab,bi->ai',Fvv,T1old)
        T1 -= einsum('ji,aj->ai',Foo,T1old)
        T1 += einsum('jb,abij->ai',Fov,T2old)
        T1 -= einsum('ajbi,bj->ai',I.vovo,T1old)
        T1 += 0.5*einsum('ajbc,bcij->ai',I.vovv,T2old)
        T1 -= 0.5*einsum('jkib,abjk->ai',I.ooov,T2old)

        T2B = T2old.copy()
        T2B += einsum('ai,bj->abij',T1old,T1old)
        T2B -= einsum('bi,aj->abij',T1old,T1old)

        Woooo = I.oooo.copy()
        Woooo += einsum('klic,cj->klij',I.ooov,T1old)
        Woooo -= einsum('kljc,ci->klij',I.ooov,T1old)
        Woooo += 0.25*einsum('klcd,cdij->klij',I.oovv,T2B)
        T2 += 0.5*einsum('klij,abkl->abij',Woooo,T2B)
        Woooo = None

        Wvvvv = I.vvvv.copy()
        Wvvvv -= einsum('akcd,bk->abcd',I.vovv,T1old)
        Wvvvv += einsum('bkcd,ak->abcd',I.vovv,T1old)
        Wvvvv += 0.25*einsum('klcd,abkl->abcd',I.oovv,T2B)
        T2 += 0.5*einsum('abcd,cdij->abij',Wvvvv,T2B)
        T2B = None
        Wvvvv = None

        Wovvo = -I.vovo.transpose((1,0,2,3))
        Wovvo -= einsum('bkcd,dj->kbcj',I.vovv,T1old)
        Wovvo += einsum('kljc,bl->kbcj',I.ooov,T1old)
        temp = 0.5*T2old + einsum('dj,bl->dbjl',T1old,T1old)
        Wovvo -= einsum('klcd,dbjl->kbcj',I.oovv,temp)
        temp = einsum('kbcj,acik->abij',Wovvo,T2old)
        temp += einsum('bkcj,ci,ak->abij',I.vovo,T1old,T1old)
        T2 += temp
        T2 -= temp.transpose((0,1,3,2))
        T2 -= temp.transpose((1,0,2,3))
        T2 += temp.transpose((1,0,3,2))
        temp = None
        #Wovvo = None

        Ftemp = Fvv - 0.5*einsum('jb,aj->ab',Fov,T1old)
        temp_ab = einsum('bc,acij->abij',Ftemp,T2old)
        temp_ab += einsum('bkij,ak->abij',I.vooo,T1old)
        T2 += temp_ab
        T2 -= temp_ab.transpose((1,0,2,3))
        temp_ab = None

        Ftemp = Foo + 0.5*einsum('jb,bi->ji',Fov,T1old)
        temp_ij = -einsum('kj,abik->abij',Ftemp,T2old)
        temp_ij += einsum('abcj,ci->abij',I.vvvo,T1old)
        T2 += temp_ij
        T2 -= temp_ij.transpose((0,1,3,2))
        temp_ij = None

        # remaining T1 terms
        T1 += einsum('xab,xbi->ai', g.vv, U11old)
        T1 -= einsum('xji,xaj->ai',g.oo, U11old)

        T1 += einsum('x,xai->ai', gtm, U11old)
        ttt = einsum('jb,bi->ji', gsov, T1old)
        T1 -= einsum('ji,aj->ai', ttt, T1old)

        # Remaining T2 terms
        gtov = einsum('xac,ci->xai', g.vv, T1old)
        gtov -= einsum('xki,ak->xai', g.oo, T1old)
        temp_abij = einsum('xai,xbj->abij', gtov, U11old)
        T2temp = T2old - einsum('bi,aj->abij', T1old, T1old)
        Wmvo = einsum('xkc,acik->xai', g.ov, T2temp)
        temp_abij += einsum('xai,xbj->abij', Wmvo, U11old)
        T2 += temp_abij
        T2 -= temp_abij.transpose((1,0,2,3))
        T2 -= temp_abij.transpose((0,1,3,2))
        T2 += temp_abij.transpose((1,0,3,2))
        temp_abij = None

        gstv = einsum('kc,bk->bc', gsov, T1old)
        temp_ab = -0.5*einsum('bc,acij->abij', gstv, T2old)
        T2 += temp_ab
        T2 -= temp_ab.transpose((1,0,2,3))
        temp_ab = None

        gsto = einsum('kc,cj->kj', gsov, T1old)
        temp_ij = -0.5*einsum('kj,abik->abij', gsto, T2old)
        T2 += temp_ij
        T2 -= temp_ij.transpose((0,1,3,2))
        temp_ij = None

        # S1 and U11 terms
        S1 = H.copy()
        U11 = h.vo.copy()
        S1 += einsum('xia,ai->x', h.ov, T1old)
        S1 += einsum('ia,xai->x', F.ov, U11old)
        S1 += einsum('ia,xai->x', gsov, U11old)

        Xov = einsum('ijab,xbj->xia', I.oovv, U11old)
        S1 += einsum('xia,ai->x', Xov, T1old)

        U11 += 0.5*einsum('xjb,abij->xai', Xov, T2old)

        U11 += einsum('xab,bi->xai', h.vv, T1old)
        U11 -= einsum('xji,aj->xai', h.oo, T1old)

        U11 += einsum('xjb,abij->xai', h.ov, T2temp)

        U11 += einsum('ab,xbi->xai', Fvv, U11old)
        U11 -= einsum('ji,xaj->xai', Foo, U11old)

        U11 += einsum('jabi,xbj->xai', Wovvo, U11old)

        Xvv = einsum('jb,aj->ab', Fov + gsov, T1old)
        U11 -= 0.5*einsum('ab,xbi->xai', Xvv, U11old)

        Xoo = einsum('jb,bi->ji', Fov + gsov, T1old)
        U11 -= 0.5*einsum('ji,xaj->xai', Xoo, U11old)

        Xmm = einsum('yjb,xbj->xy', g.ov, U11old)
        U11 += einsum('xy,yai->xai', Xmm, U11old)

    return T1, T2, S1, U11 
