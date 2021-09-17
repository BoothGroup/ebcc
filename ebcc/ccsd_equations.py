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

def two_rdm_ferm(cc, autogen=False, write=True):
    ''' Return fermionic sector 2RDM in pyscf (chemist) form, i.e.
        dm2[p,q,r,s] = <p^\dagger r^\dagger s q>
        where p,q,r,s are spin-orbitals. p,q correspond to one particle and r,s
        correspond to another particle.  The contraction between ERIs (in
        Chemist's notation) and rdm2 is E = einsum('pqrs,pqrs', eri, rdm2).

        Note that default ordering of the basis is occ_a, occ_b, virt_a, virt_b, although
        this is a general GHF code.

        NOTES:
        Equations are derived from wick as [i,j,k,l] = <i^+ j^+ k l> for each block.
        This is done *without* taking the connected part, meaning that no 1rdm contribution
        needs to be included later on, as is done with the non-autogen code (from cqcpy).
        These then subsequently have their first two indices transposed
        or a minus sign inserted (note antisymmetry), for agreement with cqcpy
        Finally, the 2RDM is transposed as dm2.transpose(2,0,3,1), for
        consistency with pyscf.
    '''

    if write:
        print('Computing fermionic space 2RDM...')

    if cc.L1 is None:
        if write:
            print('No optimized lambda amplitudes found to compute density matrices.')
            print('Using L = T^+ approximation...')
        cc.init_lam()

    L1 = cc.L1.copy()
    L2 = cc.L2.copy()
    T1 = cc.T1.copy()
    T2 = cc.T2.copy()

    delta = np.eye(cc.no)

    if autogen:
        # oooo block
        #dm2_oooo = -0.5*einsum('klab,baji->ijkl', L2, T2)
        #dm2_oooo += 1.0*einsum('klab,bi,aj->ijkl', L2, T1, T1)
        #dm2_oooo = dm2_oooo.transpose(1,0,2,3)  # This agrees now

        # No get_connected (includes the 1RDM contribution):
        dm2_oooo = 1.0*einsum('il,jk->ijkl', delta, delta)
        dm2_oooo += -1.0*einsum('jl,ik->ijkl', delta, delta)
        dm2_oooo += -0.5*einsum('klab,baji->ijkl', L2, T2)
        dm2_oooo += 1.0*einsum('ka,ai,jl->ijkl', L1, T1, delta)
        dm2_oooo += -1.0*einsum('ka,aj,il->ijkl', L1, T1, delta)
        dm2_oooo += -1.0*einsum('la,ai,jk->ijkl', L1, T1, delta)
        dm2_oooo += 1.0*einsum('la,aj,ik->ijkl', L1, T1, delta)
        dm2_oooo += 1.0*einsum('klab,bi,aj->ijkl', L2, T1, T1)
        dm2_oooo += -0.5*einsum('kmab,baim,jl->ijkl', L2, T2, delta)
        dm2_oooo += 0.5*einsum('kmab,bajm,il->ijkl', L2, T2, delta)
        dm2_oooo += 0.5*einsum('lmab,baim,jk->ijkl', L2, T2, delta)
        dm2_oooo += -0.5*einsum('lmab,bajm,ik->ijkl', L2, T2, delta)
        # Transposed for agreement with cqcpy
        dm2_oooo = dm2_oooo.transpose(1,0,2,3)

        # vvvv block
        dm2_vvvv = -0.5*einsum('ijba,cdji->abcd', L2, T2)
        dm2_vvvv += 1.0*einsum('ijba,ci,dj->abcd', L2, T1, T1)
        dm2_vvvv = dm2_vvvv.transpose(1,0,2,3)  # This agrees now

        # vovv block
        #dm2_vovv = 1.0*einsum('ja,bcij->aibc', L1, T2)
        #dm2_vovv += 0.5*einsum('jkda,di,bckj->aibc', L2, T1, T2)
        #dm2_vovv += -1.0*einsum('jkda,bj,dcik->aibc', L2, T1, T2)
        #dm2_vovv += 1.0*einsum('jkda,cj,dbik->aibc', L2, T1, T2)
        #dm2_vovv += -1.0*einsum('jkda,bj,ck,di->aibc', L2, T1, T1, T1)
        # no get_connected
        dm2_vovv = 1.0*einsum('ja,bcij->aibc', L1, T2)
        dm2_vovv += -1.0*einsum('ja,bj,ci->aibc', L1, T1, T1)
        dm2_vovv += 1.0*einsum('ja,cj,bi->aibc', L1, T1, T1)
        dm2_vovv += 0.5*einsum('jkda,di,bckj->aibc', L2, T1, T2)
        dm2_vovv += -1.0*einsum('jkda,bj,dcik->aibc', L2, T1, T2)
        dm2_vovv += -0.5*einsum('jkda,bi,dckj->aibc', L2, T1, T2)
        dm2_vovv += 1.0*einsum('jkda,cj,dbik->aibc', L2, T1, T2)
        dm2_vovv += 0.5*einsum('jkda,ci,dbkj->aibc', L2, T1, T2)
        dm2_vovv += -1.0*einsum('jkda,bj,ck,di->aibc', L2, T1, T1, T1)
        dm2_vovv = dm2_vovv.transpose(0,1,3,2)

        # vvvo block
        dm2_vvvo = -1.0*einsum('ijba,cj->abci', L2, T1)
        dm2_vvvo = dm2_vvvo.transpose(1,0,2,3)  # This agrees now

        #ovoo block
        #dm2_ovoo = 1.0*einsum('jkba,bi->iajk', L2, T1)
        # no get_connected
        dm2_ovoo = 1.0*einsum('ja,ik->iajk', L1, delta)
        dm2_ovoo += -1.0*einsum('ka,ij->iajk', L1, delta)
        dm2_ovoo += 1.0*einsum('jkba,bi->iajk', L2, T1)
        dm2_ovoo = dm2_ovoo.transpose(0,1,3,2)

        #oovo block
        #dm2_oovo = 1.0*einsum('kb,baji->ijak', L1, T2)
        #dm2_oovo += -1.0*einsum('klbc,ci,bajl->ijak', L2, T1, T2)
        #dm2_oovo += 1.0*einsum('klbc,cj,bail->ijak', L2, T1, T2)
        #dm2_oovo += 0.5*einsum('klbc,al,cbji->ijak', L2, T1, T2)
        #dm2_oovo += -1.0*einsum('klbc,al,ci,bj->ijak', L2, T1, T1, T1)
        # No get_connected
        dm2_oovo = -1.0*einsum('ai,jk->ijak', T1, delta)
        dm2_oovo += 1.0*einsum('aj,ik->ijak', T1, delta)
        dm2_oovo += 1.0*einsum('kb,baji->ijak', L1, T2)
        dm2_oovo += -1.0*einsum('kb,bi,aj->ijak', L1, T1, T1)
        dm2_oovo += 1.0*einsum('kb,bj,ai->ijak', L1, T1, T1)
        dm2_oovo += 1.0*einsum('lb,bail,jk->ijak', L1, T2, delta)
        dm2_oovo += -1.0*einsum('lb,bajl,ik->ijak', L1, T2, delta)
        dm2_oovo += -1.0*einsum('klbc,ci,bajl->ijak', L2, T1, T2)
        dm2_oovo += 1.0*einsum('klbc,cj,bail->ijak', L2, T1, T2)
        dm2_oovo += 0.5*einsum('klbc,al,cbji->ijak', L2, T1, T2)
        dm2_oovo += -0.5*einsum('klbc,ai,cbjl->ijak', L2, T1, T2)
        dm2_oovo += 0.5*einsum('klbc,aj,cbil->ijak', L2, T1, T2)
        dm2_oovo += 1.0*einsum('lb,al,bi,jk->ijak', L1, T1, T1, delta)
        dm2_oovo += -1.0*einsum('lb,al,bj,ik->ijak', L1, T1, T1, delta)
        dm2_oovo += -1.0*einsum('klbc,al,ci,bj->ijak', L2, T1, T1, T1)
        dm2_oovo += -0.5*einsum('lmbc,ci,baml,jk->ijak', L2, T1, T2, delta)
        dm2_oovo += 0.5*einsum('lmbc,cj,baml,ik->ijak', L2, T1, T2, delta)
        dm2_oovo += -0.5*einsum('lmbc,al,cbim,jk->ijak', L2, T1, T2, delta)
        dm2_oovo += 0.5*einsum('lmbc,al,cbjm,ik->ijak', L2, T1, T2, delta)
        dm2_oovo = dm2_oovo.transpose(1,0,2,3)
        
        # oovv
        dm2_oovv = 1.0*einsum('abji->ijab', T2)
        dm2_oovv += -1.0*einsum('ai,bj->ijab', T1, T1)
        dm2_oovv += 1.0*einsum('bi,aj->ijab', T1, T1)
        dm2_oovv += -1.0*einsum('kc,ci,abjk->ijab', L1, T1, T2)
        dm2_oovv += 1.0*einsum('kc,cj,abik->ijab', L1, T1, T2)
        dm2_oovv += -1.0*einsum('kc,ak,cbji->ijab', L1, T1, T2)
        dm2_oovv += 1.0*einsum('kc,ai,cbjk->ijab', L1, T1, T2)
        dm2_oovv += -1.0*einsum('kc,aj,cbik->ijab', L1, T1, T2)
        dm2_oovv += 1.0*einsum('kc,bk,caji->ijab', L1, T1, T2)
        dm2_oovv += -1.0*einsum('kc,bi,cajk->ijab', L1, T1, T2)
        dm2_oovv += 1.0*einsum('kc,bj,caik->ijab', L1, T1, T2)
        dm2_oovv += -0.5*einsum('klcd,dcik,abjl->ijab', L2, T2, T2)
        dm2_oovv += 0.5*einsum('klcd,dcjk,abil->ijab', L2, T2, T2)
        dm2_oovv += -0.5*einsum('klcd,dalk,cbji->ijab', L2, T2, T2)
        dm2_oovv += 1.0*einsum('klcd,daik,cbjl->ijab', L2, T2, T2)
        dm2_oovv += 0.5*einsum('klcd,dblk,caji->ijab', L2, T2, T2)
        dm2_oovv += -1.0*einsum('klcd,dbik,cajl->ijab', L2, T2, T2)
        dm2_oovv += 0.25*einsum('klcd,ablk,dcji->ijab', L2, T2, T2)
        dm2_oovv += 1.0*einsum('kc,ak,ci,bj->ijab', L1, T1, T1, T1)
        dm2_oovv += -1.0*einsum('kc,ak,cj,bi->ijab', L1, T1, T1, T1)
        dm2_oovv += -1.0*einsum('kc,bk,ci,aj->ijab', L1, T1, T1, T1)
        dm2_oovv += 1.0*einsum('kc,bk,cj,ai->ijab', L1, T1, T1, T1)
        dm2_oovv += -0.5*einsum('klcd,di,cj,ablk->ijab', L2, T1, T1, T2)
        dm2_oovv += 0.5*einsum('klcd,di,aj,cblk->ijab', L2, T1, T1, T2)
        dm2_oovv += -0.5*einsum('klcd,di,bj,calk->ijab', L2, T1, T1, T2)
        dm2_oovv += -0.5*einsum('klcd,dj,ai,cblk->ijab', L2, T1, T1, T2)
        dm2_oovv += 0.5*einsum('klcd,dj,bi,calk->ijab', L2, T1, T1, T2)
        dm2_oovv += 1.0*einsum('klcd,ak,di,cbjl->ijab', L2, T1, T1, T2)
        dm2_oovv += -1.0*einsum('klcd,ak,dj,cbil->ijab', L2, T1, T1, T2)
        dm2_oovv += -0.5*einsum('klcd,ak,bl,dcji->ijab', L2, T1, T1, T2)
        dm2_oovv += 0.5*einsum('klcd,ak,bi,dcjl->ijab', L2, T1, T1, T2)
        dm2_oovv += -0.5*einsum('klcd,ak,bj,dcil->ijab', L2, T1, T1, T2)
        dm2_oovv += -1.0*einsum('klcd,bk,di,cajl->ijab', L2, T1, T1, T2)
        dm2_oovv += 1.0*einsum('klcd,bk,dj,cail->ijab', L2, T1, T1, T2)
        dm2_oovv += -0.5*einsum('klcd,bk,ai,dcjl->ijab', L2, T1, T1, T2)
        dm2_oovv += 0.5*einsum('klcd,bk,aj,dcil->ijab', L2, T1, T1, T2)
        dm2_oovv += 1.0*einsum('klcd,ak,bl,di,cj->ijab', L2, T1, T1, T1, T1)
        dm2_oovv = dm2_oovv.transpose(1,0,2,3)

        # vvoo
        dm2_vvoo = 1.0*einsum('ijba->abij', L2)
        dm2_vvoo = dm2_vvoo.transpose(1,0,2,3)

        # vovo
        dm2_vovo = 1.0*einsum('ja,bi->aibj', L1, T1)
        dm2_vovo += 1.0*einsum('jkca,cbik->aibj', L2, T2)
        dm2_vovo += -1.0*einsum('ka,bk,ij->aibj', L1, T1, delta)
        dm2_vovo += 1.0*einsum('jkca,bk,ci->aibj', L2, T1, T1)
        dm2_vovo += 0.5*einsum('klca,cblk,ij->aibj', L2, T2, delta)
        dm2_vovo *= -1. # Flip sign
    else:
        # Code from cqcpy
        tfac = 1.0
        # oooo block
        T2temp = T2 + einsum('ci,dj->cdij', T1, T1)
        T2temp -= einsum('di,cj->cdij', T1, T1)
        dm2_oooo = 0.5*einsum('klab,abij->klij', L2, T2temp)

        # ovoo block
        T2temp = T2 + einsum('ci,dj->cdij', T1, T1)
        T2temp -= einsum('di,cj->cdij', T1, T1)
        dm2_ovoo = -einsum('kb,baij->kaij', L1, T2temp)
        LTo = einsum('klcd,cdil->ki', L2, T2)
        tmp = -0.5*einsum('ki,aj->kaij', LTo, T1)
        dm2_ovoo += tmp - tmp.transpose((0, 1, 3, 2))
        Lklid = einsum('klcd,ci->klid', L2, T1)
        tmp = -einsum('klid,adjl->kaij', Lklid, T2)
        dm2_ovoo += tmp - tmp.transpose((0, 1, 3, 2))
        dm2_ovoo += 0.5*einsum('lkdb,dbji,al->kaij', L2, T2temp, T1)

        # vvoo block
        dm2_vvoo = tfac*T2.copy()
        dm2_vvoo += tfac*einsum('ai,bj->abij', T1, T1)
        dm2_vvoo -= tfac*einsum('aj,bi->abij', T1, T1)
        LTki = einsum('kc,ci->ki', L1, T1)
        tmp = -einsum('ki,abkj->abij', LTki, T2)
        dm2_vvoo += tmp - tmp.transpose((0, 1, 3, 2))
        LTac = einsum('kc,ak->ac', L1, T1)
        tmp = -einsum('ac,cbij->abij', LTac, T2)
        dm2_vvoo += tmp - tmp.transpose((1, 0, 2, 3))
        T2temp = T2 - einsum('bk,cj->bcjk', T1, T1)
        LTbj = einsum('kc,bcjk->bj', L1, T2temp)
        dm2_vvoo += einsum('ai,bj->abij', LTbj, T1)
        dm2_vvoo -= einsum('aj,bi->abij', LTbj, T1)
        dm2_vvoo -= einsum('bi,aj->abij', LTbj, T1)
        dm2_vvoo += einsum('bj,ai->abij', LTbj, T1)
        LToo = einsum('klcd,cdij->klij', L2, T2)
        dm2_vvoo += 0.25*einsum('klij,abkl->abij', LToo, T2)
        LTov = einsum('klcd,caki->lida', L2, T2)
        tmp = 0.5*einsum('lida,bdjl->abij', LTov, T2)
        dm2_vvoo += tmp - tmp.transpose((0, 1, 3, 2))
        dm2_vvoo -= tmp.transpose((1, 0, 2, 3))
        dm2_vvoo += tmp.transpose((1, 0, 3, 2))
        T2temp = T2 + einsum('cj,ai->acij', T1, T1) - einsum('ci,aj->acij', T1, T1)
        Lcb = einsum('klcd,bdkl->cb', L2, T2)
        tmp = -0.5*einsum('cb,acij->abij', Lcb, T2temp)
        dm2_vvoo += tmp - tmp.transpose((1, 0, 2, 3))
        Lkj = einsum('klcd,cdjl->kj', L2, T2)
        tmp = -0.5*einsum('kj,abik->abij', Lkj, T2temp)
        dm2_vvoo += tmp - tmp.transpose((0, 1, 3, 2))
        T2temp = T2 + einsum('ci,dj->cdij', T1, T1)
        LToo = einsum('klcd,cdij->klij', L2, T2temp)
        tmp = einsum('klij,ak->alij', LToo, T1)
        tmp = 0.25*einsum('alij,bl->abij', tmp, T1)
        dm2_vvoo += tmp - tmp.transpose((1, 0, 2, 3))
        Looov = einsum('klcd,ci->klid', L2, T1)
        Loooo = einsum('klid,dj->klij', Looov, T1)
        Loooo -= Loooo.transpose((0, 1, 3, 2))
        dm2_vvoo += 0.25*einsum('klij,abkl->abij', Loooo, T2temp)
        Lalid = einsum('klcd,ak,ci->alid', L2, T1, T1)
        tmp = einsum('alid,bdjl->abij', Lalid, T2)
        dm2_vvoo -= tmp
        dm2_vvoo += tmp.transpose((0, 1, 3, 2)) + tmp.transpose((1, 0, 2, 3))
        dm2_vvoo -= tmp.transpose((1, 0, 3, 2))

        # NOTE: oovv block missing from cqcpy - but it is just the L2 amplitudes
        dm2_oovv = 1.0*einsum('ijba->ijab', L2)
        dm2_oovv = dm2_oovv.transpose(1,0,2,3)

        # vovo block
        dm2_vovo = -einsum('ja,bi->bjai', L1, T1)
        T2temp = T2 + einsum('bk,ci->bcki', T1, T1)
        dm2_vovo -= einsum('kjac,bcki->bjai', L2, T2temp)

        # vvvo block
        T2temp = T2 + einsum('ci,dj->cdij', T1, T1)
        T2temp -= einsum('di,cj->cdij', T1, T1)
        dm2_vvvo = einsum('ja,bcji->bcai', L1, T2temp)
        LTba = einsum('jlad,bdjl->ba', L2, T2)
        dm2_vvvo += 0.5*einsum('ba,ci->bcai', LTba, T1)
        dm2_vvvo -= 0.5*einsum('ca,bi->bcai', LTba, T1)
        LTtemp = einsum('jkad,cdik->jcai', L2, T2)
        dm2_vvvo += einsum('jcai,bj->bcai', LTtemp, T1)
        dm2_vvvo -= einsum('jbai,cj->bcai', LTtemp, T1)
        dm2_vvvo -= 0.5*einsum('kjda,cbkj,di->bcai', L2, T2temp, T1)

        # vvvv block
        T2temp = T2 + einsum('ci,dj->cdij', T1, T1)
        T2temp -= einsum('di,cj->cdij', T1, T1)
        dm2_vvvv = 0.5*einsum('ijab,cdij->cdab', L2, T2temp)
    
        # oovo block
        dm2_oovo = -einsum('jkab,bi->jkai', L2, T1)
    
        # vovv block
        dm2_vovv = einsum('jiab,cj->ciab', L2, T1)
    
    # Now put the blocks together, symmetrizing where appropriate
    # NOTE: We are storing as occupied_a, occupied_b, virtual_a, virtual_b
    dm2 = np.zeros((cc.nso, cc.nso, cc.nso, cc.nso))

    # Antisymmetric wrt switching annihilation operators
    dm2_oooo = (dm2_oooo - dm2_oooo.transpose(0,1,3,2)) / 2.
    dm2_vvvv = (dm2_vvvv - dm2_vvvv.transpose(0,1,3,2)) / 2.
    dm2_vovv = (dm2_vovv - dm2_vovv.transpose(0,1,3,2)) / 2.
    dm2_ovoo = (dm2_ovoo - dm2_ovoo.transpose(0,1,3,2)) / 2.
    dm2_oovv = (dm2_oovv - dm2_oovv.transpose(0,1,3,2)) / 2.
    dm2_vvoo = (dm2_vvoo - dm2_vvoo.transpose(0,1,3,2)) / 2.

    # Antisymmetric wrt switching creation operators
    dm2_vvvo = (dm2_vvvo - dm2_vvvo.transpose(1,0,2,3)) / 2.
    dm2_oooo = (dm2_oooo - dm2_oooo.transpose(1,0,2,3)) / 2.
    dm2_oovo = (dm2_oovo - dm2_oovo.transpose(1,0,2,3)) / 2.
    dm2_oovv = (dm2_oovv - dm2_oovv.transpose(1,0,2,3)) / 2.
    dm2_vvoo = (dm2_vvoo - dm2_vvoo.transpose(1,0,2,3)) / 2.

    # Hermitian wrt inverting of operator order
    dm2[:cc.no, :cc.no, :cc.no, :cc.no] = (dm2_oooo + dm2_oooo.transpose(3,2,1,0)) / 2.
    dm2[cc.no:, cc.no:, cc.no:, cc.no:] = (dm2_vvvv + dm2_vvvv.transpose(3,2,1,0)) / 2.
    
    # ovvv/vovv/vvov/vvvo blocks (two contributions to make sure we have both hermitian parts)
    dm2[:cc.no, cc.no:, cc.no:, cc.no:] = (-dm2_vovv.transpose(1,0,2,3) + dm2_vvvo.transpose(3,2,1,0)) / 2.
    dm2[cc.no:, :cc.no, cc.no:, cc.no:] = (dm2_vovv - dm2_vvvo.transpose(2,3,1,0)) / 2.
    dm2[cc.no:, cc.no:, :cc.no, cc.no:] = (dm2_vovv.transpose(3,2,1,0) - dm2_vvvo.transpose(0,1,3,2)) / 2.
    dm2[cc.no:, cc.no:, cc.no:, :cc.no] = (-dm2_vovv.transpose(3,2,0,1) + dm2_vvvo) / 2.

    # vooo/ovoo/oovo/ooov blocks (two contributions to make sure we have both hermitian parts)
    dm2[cc.no:, :cc.no, :cc.no, :cc.no] = (-dm2_ovoo.transpose(1,0,2,3) - dm2_oovo.transpose(2,3,1,0)) / 2.
    dm2[:cc.no, cc.no:, :cc.no, :cc.no] = (dm2_ovoo + dm2_oovo.transpose(3,2,1,0)) / 2.
    dm2[:cc.no, :cc.no, cc.no:, :cc.no] = (dm2_ovoo.transpose(3,2,1,0) + dm2_oovo) / 2.
    dm2[:cc.no, :cc.no, :cc.no, cc.no:] = (-dm2_ovoo.transpose(3,2,0,1) - dm2_oovo.transpose(0,1,3,2)) / 2.

    # vvoo/oovv blocks
    dm2[cc.no:, cc.no:, :cc.no, :cc.no] = (dm2_vvoo + dm2_oovv.transpose(3,2,1,0)) / 2.
    dm2[:cc.no, :cc.no, cc.no:, cc.no:] = (dm2_oovv + dm2_vvoo.transpose(3,2,1,0)) / 2.

    # voov / ovvo
    dm2[cc.no:, :cc.no, :cc.no, cc.no:] = (-dm2_vovo.transpose(0,1,3,2) - dm2_vovo.transpose(2,3,1,0)) / 2.
    dm2[:cc.no, cc.no:, cc.no:, :cc.no] = (-dm2_vovo.transpose(1,0,2,3) - dm2_vovo.transpose(3,2,0,1)) / 2.

    # vovo / ovov blocks
    dm2[cc.no:, :cc.no, cc.no:, :cc.no] = (dm2_vovo + dm2_vovo.transpose(2,3,0,1)) / 2.
    dm2[:cc.no, cc.no:, :cc.no, cc.no:] = (dm2_vovo.transpose(3,2,1,0) + dm2_vovo.transpose(1,0,3,2)) / 2.
    
    if not autogen:
        # Add 1rdm contribution when one electron traced out
        # This is because the non-autogen code was only the connected contribution
        dm1 = one_rdm_ferm(cc, autogen=autogen, write=False)
        # Remove MF component of 1RDM
        dm1[np.diag_indices(cc.no)] -= 1.

        for i in range(cc.no):
            dm2[i,:,:,i] -= dm1
            dm2[:,i,i,:] -= dm1
            dm2[:,i,:,i] += dm1
            dm2[i,:,i,:] += dm1

        #print('Trace of 1RDM: {}. Number of electrons: {}'.format(np.trace(dm1), cc.no))
        # Add mean-field part
        for i in range(cc.no):
            for j in range(cc.no):
                pass
                dm2[i,j,i,j] += 1
                dm2[i,j,j,i] -= 1

    # Transpose for consistency with pyscf
    dm2 = dm2.transpose(2,0,3,1)
    return dm2


def one_rdm_ferm(cc, autogen=False, write=True):
    ''' Calculate 1RDM '''

    if write:
        print('Computing fermionic space 1RDM...')

    if cc.L1 is None:
        if write:
            print('No optimized lambda amplitudes found to compute density matrices.')
            print('Using L = T^+ approximation...')
        cc.init_lam()

    L1 = cc.L1
    L2 = cc.L2
    T1 = cc.T1
    T2 = cc.T2
    # Small optimization for the vo block in the non-autogen code.
    opt = True 

    if autogen:
        # Only oo and vv block currently autogenerated
        dm1_oo = -1.0*einsum('ia,aj->ij', L1, T1)
        dm1_oo += 0.5*einsum('ikab,bajk->ij', L2, T2)

        dm1_vv = 1.0*einsum('ib,ai->ab', L1, T1)
        dm1_vv += -0.5*einsum('ijcb,caji->ab', L2, T2)

        dm1_ov = L1.copy()

        dm1_vo = T1.copy() 
        dm1_vo += -1.0*einsum('jb,baij->ai', L1, T2)
        dm1_vo += -1.0*einsum('jb,aj,bi->ai', L1, T1, T1)
        dm1_vo += 0.5*einsum('jkbc,ci,bakj->ai', L2, T1, T2)
        dm1_vo += 0.5*einsum('jkbc,aj,cbik->ai', L2, T1, T2)
        
    else:
        dm1_oo = -einsum('ja,ai->ji', L1, T1)
        dm1_oo -= 0.5*einsum('kjca,caki->ji', L2, T2)

        dm1_vv = einsum('ia,bi->ba', L1, T1)
        dm1_vv += 0.5*einsum('kica,cbki->ba', L2, T2)
        
        # This is taken from pyscf. May already be hermitian here.
        dm1_ov = L1

        if opt:
            dm1_vo = T1.copy()
            T2temp = T2 - einsum('bi,aj->baji', T1, T1)
            dm1_vo += einsum('jb,baji->ai', L1, T2temp)

            Pac = 0.5*einsum('kjcb,abkj->ac', L2, T2)
            dm1_vo -= einsum('ac,ci->ai', Pac, T1)

            Pik = 0.5*einsum('kjcb,cbij->ik', L2, T2)
            dm1_vo -= einsum('ik,ak->ai', Pik, T1)
        else:
            dm1_vo = T1 + einsum('jb,baji->ai', L1, T2)
            dm1_vo -= einsum('jb,bi,aj->ai', L1, T1, T1)
            dm1_vo -= 0.5*einsum('kjcb,ci,abkj->ai', L2, T1, T2)
            dm1_vo -= 0.5*einsum('kjcb,ak,cbij->ai', L2, T1, T2)

    dm1 = np.zeros((cc.nso, cc.nso))
    # Hermitize everything
    dm1[:cc.no, :cc.no] = (dm1_oo + dm1_oo.T) / 2.
    dm1[:cc.no, cc.no:] = (dm1_ov + dm1_vo.T) / 2.
    dm1[cc.no:, :cc.no] = dm1[:cc.no, cc.no:].T
    dm1[cc.no:, cc.no:] = (dm1_vv + dm1_vv.T) / 2.

    # Add mean-field part
    dm1[np.diag_indices(cc.no)] += 1.
    if write:
        print('Trace of 1RDM: {}. Number of electrons: {}'.format(np.trace(dm1), cc.no))

    return dm1


def lam_updates_ccsd(cc, autogen=False):
    ''' Solve residual equations for CCSD.
        Equation generating script using the 'wick' program found in gen_eqns/ccsd_T.py
    '''

    # Copy fock matrix in MO basis, and remove diagonals
    F = copy.copy(cc.fock_mo)
    F_full = copy.copy(cc.fock_mo)
    F.oo = F.oo - np.diag(cc.eo)
    F.vv = F.vv - np.diag(cc.ev)

    I = cc.I

    T1old = cc.T1old
    T2old = cc.T2old

    L1old = cc.L1old
    L2old = cc.L2old

    # Update L1
    if autogen:
        opt = False # It seems as though this is not an optimization, but rather
                    # necessary for getting the right T1 residuals
        # Piece not dependent on lamdba
        L1 = F.ov.copy()
        L1 += -1.0*einsum('bj,jiab->ia', T1old, I.oovv)
        # Piece proportional to lambda
        if opt:
            L1 += -1.0*einsum('ij,ja->ia', F.oo, L1old)
            L1 += 1.0*einsum('ba,ib->ia', F.vv, L1old)
            L1 += -1.0*einsum('ibja,jb->ia', I.ovov, L1old)
            L1 += 0.5*einsum('ibjk,kjab->ia', I.ovoo, L2old)
            L1 += -0.5*einsum('bcja,jicb->ia', I.vvov, L2old)
            L1 += -1.0*einsum('ja,bj,ib->ia', F.ov, T1old, L1old)
            L1 += -1.0*einsum('ib,bj,ja->ia', F.ov, T1old, L1old)
            L1 += 1.0*einsum('jikb,bj,ka->ia', I.ooov, T1old, L1old)
            L1 += -1.0*einsum('jika,bj,kb->ia', I.ooov, T1old, L1old)
            L1 += -1.0*einsum('jbac,cj,ib->ia', I.ovvv, T1old, L1old)
            L1 += 1.0*einsum('ibac,cj,jb->ia', I.ovvv, T1old, L1old)
            L1 += -0.5*einsum('ja,bcjk,kicb->ia', F.ov, T2old, L2old)
            L1 += -0.5*einsum('ib,cbjk,kjac->ia', F.ov, T2old, L2old)
            L1 += 0.5*einsum('jkab,cbkj,ic->ia', I.oovv, T2old, L1old)
            L1 += 0.5*einsum('jibc,cbjk,ka->ia', I.oovv, T2old, L1old)
            L1 += 1.0*einsum('jiab,cbjk,kc->ia', I.oovv, T2old, L1old)
            L1 += -1.0*einsum('jbka,cj,kicb->ia', I.ovov, T1old, L2old)
            L1 += -1.0*einsum('ibjc,ck,jkab->ia', I.ovov, T1old, L2old)
            L1 += 0.5*einsum('jikl,bj,lkab->ia', I.oooo, T1old, L2old)
            L1 += 0.5*einsum('bcad,dj,jicb->ia', I.vvvv, T1old, L2old)
            L1 += 0.25*einsum('jkla,bckj,licb->ia', I.ooov, T2old, L2old)
            L1 += -1.0*einsum('jikb,cbjl,klac->ia', I.ooov, T2old, L2old)
            L1 += 0.5*einsum('jika,bcjl,klcb->ia', I.ooov, T2old, L2old)
            L1 += 1.0*einsum('jbac,dcjk,kidb->ia', I.ovvv, T2old, L2old)
            L1 += -0.25*einsum('ibcd,dcjk,kjab->ia', I.ovvv, T2old, L2old)
            L1 += -0.5*einsum('ibac,dcjk,kjdb->ia', I.ovvv, T2old, L2old)
            L1 += -1.0*einsum('jkab,cj,bk,ic->ia', I.oovv, T1old, T1old, L1old)
            L1 += -1.0*einsum('jibc,ck,bj,ka->ia', I.oovv, T1old, T1old, L1old)
            L1 += 1.0*einsum('jiab,bk,cj,kc->ia', I.oovv, T1old, T1old, L1old)
            L1 += -0.5*einsum('jkla,bj,ck,licb->ia', I.ooov, T1old, T1old, L2old)
            L1 += -1.0*einsum('jikb,bl,cj,klac->ia', I.ooov, T1old, T1old, L2old)
            L1 += 1.0*einsum('jbac,ck,dj,kidb->ia', I.ovvv, T1old, T1old, L2old)
            L1 += 0.5*einsum('ibcd,dj,ck,jkab->ia', I.ovvv, T1old, T1old, L2old)
            L1 += 1.0*einsum('jkab,cj,dbkl,lidc->ia', I.oovv, T1old, T2old, L2old)
            L1 += -0.25*einsum('jkab,bl,cdkj,lidc->ia', I.oovv, T1old, T2old, L2old)
            L1 += 0.5*einsum('jkab,bj,cdkl,lidc->ia', I.oovv, T1old, T2old, L2old)
            L1 += -0.25*einsum('jibc,dj,cbkl,lkad->ia', I.oovv, T1old, T2old, L2old)
            L1 += 1.0*einsum('jibc,ck,dbjl,klad->ia', I.oovv, T1old, T2old, L2old)
            L1 += 0.5*einsum('jibc,cj,dbkl,lkad->ia', I.oovv, T1old, T2old, L2old)
            L1 += -0.5*einsum('jiab,cj,dbkl,lkdc->ia', I.oovv, T1old, T2old, L2old)
            L1 += -0.5*einsum('jiab,bk,cdjl,kldc->ia', I.oovv, T1old, T2old, L2old)
            L1 += 0.5*einsum('jkab,bl,cj,dk,lidc->ia', I.oovv, T1old, T1old, T1old, L2old)
            L1 += 0.5*einsum('jibc,ck,bl,dj,klad->ia', I.oovv, T1old, T1old, T1old, L2old)
        else:
            L1 += -1.0*einsum('ij,ja->ia', F.oo, L1old)
            L1 += 1.0*einsum('ba,ib->ia', F.vv, L1old)
            L1 += -1.0*einsum('ibja,jb->ia', I.ovov, L1old)
            L1 += -1.0*einsum('bj,jiab->ia', F.vo, L2old)
            L1 += 0.5*einsum('ibjk,kjab->ia', I.ovoo, L2old)
            L1 += -0.5*einsum('bcja,jicb->ia', I.vvov, L2old)
            L1 += -1.0*einsum('ib,bj,ja->ia', F.ov, T1old, L1old)
            L1 += -1.0*einsum('ja,bj,ib->ia', F.ov, T1old, L1old)
            L1 += -1.0*einsum('jika,bj,kb->ia', I.ooov, T1old, L1old)
            L1 += 1.0*einsum('ibac,cj,jb->ia', I.ovvv, T1old, L1old)
            L1 += 1.0*einsum('jikb,bj,ka->ia', I.ooov, T1old, L1old)
            L1 += -1.0*einsum('jbac,cj,ib->ia', I.ovvv, T1old, L1old)
            L1 += 1.0*einsum('jk,bj,kiab->ia', F_full.oo, T1old, L2old)
            L1 += -1.0*einsum('bc,cj,jiab->ia', F_full.vv, T1old, L2old)
            L1 += 1.0*einsum('jiab,cbjk,kc->ia', I.oovv, T2old, L1old)
            L1 += 0.5*einsum('jibc,cbjk,ka->ia', I.oovv, T2old, L1old)
            L1 += 0.5*einsum('jkab,cbkj,ic->ia', I.oovv, T2old, L1old)
            L1 += -0.5*einsum('ib,cbjk,kjac->ia', F.ov, T2old, L2old)
            L1 += -1.0*einsum('ibjc,ck,jkab->ia', I.ovov, T1old, L2old)
            L1 += 0.5*einsum('jikl,bj,lkab->ia', I.oooo, T1old, L2old)
            L1 += -0.5*einsum('ja,bcjk,kicb->ia', F.ov, T2old, L2old)
            L1 += -1.0*einsum('jbka,cj,kicb->ia', I.ovov, T1old, L2old)
            L1 += 1.0*einsum('jb,cbjk,kiac->ia', F.ov, T2old, L2old)
            L1 += 1.0*einsum('jbkc,cj,kiab->ia', I.ovov, T1old, L2old)
            L1 += 0.5*einsum('bcad,dj,jicb->ia', I.vvvv, T1old, L2old)
            L1 += 0.5*einsum('jika,bcjl,klcb->ia', I.ooov, T2old, L2old)
            L1 += -0.5*einsum('ibac,dcjk,kjdb->ia', I.ovvv, T2old, L2old)
            L1 += -1.0*einsum('jikb,cbjl,klac->ia', I.ooov, T2old, L2old)
            L1 += -0.25*einsum('ibcd,dcjk,kjab->ia', I.ovvv, T2old, L2old)
            L1 += 0.25*einsum('jkla,bckj,licb->ia', I.ooov, T2old, L2old)
            L1 += 1.0*einsum('jbac,dcjk,kidb->ia', I.ovvv, T2old, L2old)
            L1 += -0.5*einsum('jklb,cbkj,liac->ia', I.ooov, T2old, L2old)
            L1 += 0.5*einsum('jbcd,dcjk,kiab->ia', I.ovvv, T2old, L2old)
            L1 += 1.0*einsum('jiab,bk,cj,kc->ia', I.oovv, T1old, T1old, L1old)
            L1 += -1.0*einsum('jibc,ck,bj,ka->ia', I.oovv, T1old, T1old, L1old)
            L1 += -1.0*einsum('jkab,cj,bk,ic->ia', I.oovv, T1old, T1old, L1old)
            L1 += 1.0*einsum('jb,bk,cj,kiac->ia', F.ov, T1old, T1old, L2old)
            L1 += -1.0*einsum('jikb,bl,cj,klac->ia', I.ooov, T1old, T1old, L2old)
            L1 += 0.5*einsum('ibcd,dj,ck,jkab->ia', I.ovvv, T1old, T1old, L2old)
            L1 += -0.5*einsum('jkla,bj,ck,licb->ia', I.ooov, T1old, T1old, L2old)
            L1 += 1.0*einsum('jbac,ck,dj,kidb->ia', I.ovvv, T1old, T1old, L2old)
            L1 += 1.0*einsum('jklb,cj,bk,liac->ia', I.ooov, T1old, T1old, L2old)
            L1 += -1.0*einsum('jbcd,dk,cj,kiab->ia', I.ovvv, T1old, T1old, L2old)
            L1 += -0.5*einsum('jiab,cj,dbkl,lkdc->ia', I.oovv, T1old, T2old, L2old)
            L1 += -0.5*einsum('jiab,bk,cdjl,kldc->ia', I.oovv, T1old, T2old, L2old)
            L1 += -0.25*einsum('jibc,dj,cbkl,lkad->ia', I.oovv, T1old, T2old, L2old)
            L1 += 1.0*einsum('jibc,ck,dbjl,klad->ia', I.oovv, T1old, T2old, L2old)
            L1 += 0.5*einsum('jibc,cj,dbkl,lkad->ia', I.oovv, T1old, T2old, L2old)
            L1 += 1.0*einsum('jkab,cj,dbkl,lidc->ia', I.oovv, T1old, T2old, L2old)
            L1 += -0.25*einsum('jkab,bl,cdkj,lidc->ia', I.oovv, T1old, T2old, L2old)
            L1 += 0.5*einsum('jkab,bj,cdkl,lidc->ia', I.oovv, T1old, T2old, L2old)
            L1 += 0.5*einsum('jkbc,dj,cbkl,liad->ia', I.oovv, T1old, T2old, L2old)
            L1 += 0.5*einsum('jkbc,cl,dbkj,liad->ia', I.oovv, T1old, T2old, L2old)
            L1 += -1.0*einsum('jkbc,cj,dbkl,liad->ia', I.oovv, T1old, T2old, L2old)
            L1 += 0.5*einsum('jibc,ck,bl,dj,klad->ia', I.oovv, T1old, T1old, T1old, L2old)
            L1 += 0.5*einsum('jkab,bl,cj,dk,lidc->ia', I.oovv, T1old, T1old, T1old, L2old)
            L1 += -1.0*einsum('jkbc,cl,dj,bk,liad->ia', I.oovv, T1old, T1old, T1old, L2old)

        # Piece independent of Lambda
        L2 = I.oovv.copy().transpose(1,0,3,2)
        # Piece proportional to lambda
        L2 += 1.0*einsum('jikb,ka->ijab', I.ooov, L1old)
        L2 += -1.0*einsum('jika,kb->ijab', I.ooov, L1old)
        L2 += 1.0*einsum('jcba,ic->ijab', I.ovvv, L1old)
        L2 += -1.0*einsum('icba,jc->ijab', I.ovvv, L1old)
        L2 += -1.0*einsum('jk,kiba->ijab', F.oo, L2old)
        L2 += 1.0*einsum('ik,kjba->ijab', F.oo, L2old)
        L2 += -1.0*einsum('cb,jiac->ijab', F.vv, L2old)
        L2 += 1.0*einsum('ca,jibc->ijab', F.vv, L2old)
        L2 += -0.5*einsum('jikl,lkba->ijab', I.oooo, L2old)
        L2 += 1.0*einsum('jckb,kiac->ijab', I.ovov, L2old)
        L2 += -1.0*einsum('jcka,kibc->ijab', I.ovov, L2old)
        L2 += -1.0*einsum('ickb,kjac->ijab', I.ovov, L2old)
        L2 += 1.0*einsum('icka,kjbc->ijab', I.ovov, L2old)
        L2 += -0.5*einsum('cdba,jidc->ijab', I.vvvv, L2old)
        L2 += -1.0*einsum('jibc,ck,ka->ijab', I.oovv, T1old, L1old)
        L2 += 1.0*einsum('jiac,ck,kb->ijab', I.oovv, T1old, L1old)
        L2 += 1.0*einsum('kjba,ck,ic->ijab', I.oovv, T1old, L1old)
        L2 += -1.0*einsum('kiba,ck,jc->ijab', I.oovv, T1old, L1old)
        L2 += -1.0*einsum('jc,ck,kiba->ijab', F.ov, T1old, L2old)
        L2 += 1.0*einsum('ic,ck,kjba->ijab', F.ov, T1old, L2old)
        L2 += 1.0*einsum('kb,ck,jiac->ijab', F.ov, T1old, L2old)
        L2 += -1.0*einsum('ka,ck,jibc->ijab', F.ov, T1old, L2old)
        L2 += 1.0*einsum('jikc,cl,klba->ijab', I.ooov, T1old, L2old)
        L2 += 1.0*einsum('kjlb,ck,liac->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('jcbd,dk,kiac->ijab', I.ovvv, T1old, L2old)
        L2 += -1.0*einsum('kjla,ck,libc->ijab', I.ooov, T1old, L2old)
        L2 += 1.0*einsum('jcad,dk,kibc->ijab', I.ovvv, T1old, L2old)
        L2 += 1.0*einsum('kjlc,ck,liba->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('kilb,ck,ljac->ijab', I.ooov, T1old, L2old)
        L2 += 1.0*einsum('icbd,dk,kjac->ijab', I.ovvv, T1old, L2old)
        L2 += 1.0*einsum('kila,ck,ljbc->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('icad,dk,kjbc->ijab', I.ovvv, T1old, L2old)
        L2 += -1.0*einsum('kilc,ck,ljba->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('kcba,dk,jidc->ijab', I.ovvv, T1old, L2old)
        L2 += 1.0*einsum('kcbd,dk,jiac->ijab', I.ovvv, T1old, L2old)
        L2 += -1.0*einsum('kcad,dk,jibc->ijab', I.ovvv, T1old, L2old)
        L2 += -0.5*einsum('jibc,dckl,lkad->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('jiac,dckl,lkbd->ijab', I.oovv, T2old, L2old)
        L2 += 0.25*einsum('jicd,dckl,lkba->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('kjba,cdkl,lidc->ijab', I.oovv, T2old, L2old)
        L2 += -1.0*einsum('kjbc,dckl,liad->ijab', I.oovv, T2old, L2old)
        L2 += 1.0*einsum('kjac,dckl,libd->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('kjcd,dckl,liba->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('kiba,cdkl,ljdc->ijab', I.oovv, T2old, L2old)
        L2 += 1.0*einsum('kibc,dckl,ljad->ijab', I.oovv, T2old, L2old)
        L2 += -1.0*einsum('kiac,dckl,ljbd->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('kicd,dckl,ljba->ijab', I.oovv, T2old, L2old)
        L2 += 0.25*einsum('klba,cdlk,jidc->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('klbc,dclk,jiad->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('klac,dclk,jibd->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('jicd,dk,cl,klba->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('kjbc,cl,dk,liad->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('kjac,cl,dk,libd->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('kjcd,dl,ck,liba->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('kibc,cl,dk,ljad->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('kiac,cl,dk,ljbd->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('kicd,dl,ck,ljba->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -0.5*einsum('klba,ck,dl,jidc->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('klbc,dk,cl,jiad->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('klac,dk,cl,jibd->ijab', I.oovv, T1old, T1old, L2old)
        # Disconnected piece, with intermediate projections onto singles space
        L2 += 1.0*einsum('jb,ia->ijab', L1old, F.ov)
        L2 += -1.0*einsum('ja,ib->ijab', L1old, F.ov)
        L2 += -1.0*einsum('ib,ja->ijab', L1old, F.ov)
        L2 += 1.0*einsum('ia,jb->ijab', L1old, F.ov)
        L2 += -1.0*einsum('ck,jb,kiac->ijab', T1old, L1old, I.oovv)
        L2 += 1.0*einsum('ck,ja,kibc->ijab', T1old, L1old, I.oovv)
        L2 += 1.0*einsum('ck,ib,kjac->ijab', T1old, L1old, I.oovv)
        L2 += -1.0*einsum('ck,ia,kjbc->ijab', T1old, L1old, I.oovv)

    else:
        fac = 1.0
        # Optimized code
        L1 = F.ov.copy()
        L2 = I.oovv.copy()
    
        L1 += einsum('jiba,bj->ia', I.oovv, T1old)
    
        TTemp = 0.5*T2old + einsum('bj,ck->bcjk', T1old, T1old)
        IvovoT1 = einsum('aibc,ck->aibk', I.vovv, T1old)
        IvovoT2 = einsum('bj,jika->biak', T1old, I.ooov)
        IooovT = einsum('ikbc,bj->ikjc', I.oovv, T1old)
        IvovvT = einsum('ck,kiab->ciab', T1old, I.oovv)
        IovT = einsum('ikac,ck->ia', I.oovv, T1old)

        # OO
        IToo = einsum('ib,bj->ij', F.ov, T1old) + F.oo\
            + einsum('ikbc,bcjk->ij', I.oovv, TTemp)\
            + einsum('ijkb,bj->ik', I.ooov, T1old)
        L1 -= fac*einsum('ja,ij->ia', L1old, IToo)

        temp = fac*einsum('ikab,jk->ijab', L2old, IToo)
        L2 -= temp
        L2 += temp.transpose((1, 0, 2, 3))
        del IToo
        del temp

        # VV
        ITvv = einsum('ja,bj->ba', F.ov, T1old) - F.vv\
            - einsum('bkac,ck->ba', I.vovv, T1old)\
            + einsum('jkac,bcjk->ba', I.oovv, TTemp)
        L1 -= fac*einsum('ib,ba->ia', L1old, ITvv)

        temp = fac*einsum('ijac,cb->ijab', L2old, ITvv)
        L2 -= temp
        L2 += temp.transpose((0, 1, 3, 2))
        del ITvv
        del temp

        # VOVO
        ITvovo = -IvovoT1 - IvovoT2 - I.vovo
        L1 += fac*einsum('jb,biaj->ia', L1old, ITvovo)
        del ITvovo

        # OOVV
        IToovv = einsum('kica,bcjk->ijab', I.oovv, T2old)\
            - einsum('ciba,bj->ijac', IvovvT, T1old)
        L1 += fac*einsum('jb,ijab->ia', L1old, IToovv)
        del IToovv

        # VOOO
        ITvooo = einsum('cibk,bj->cijk', I.vovo, T1old) + 0.5*I.vooo\
            + 0.5*einsum('ic,bcjk->bijk', F.ov, T2old)\
            + 0.5*einsum('ijkl,bj->bikl', I.oooo, T1old)\
            + einsum('kilc,bcjk->bilj', I.ooov, T2old)\
            - einsum('bicl,ck->bilk', IvovoT2, T1old)\
            + 0.5*einsum('dibc,bcjk->dijk', I.vovv, TTemp)\
            + 0.5*einsum('id,cdkl->cikl', IovT, T2old)\
            - einsum('ikjc,cdkl->dijl', IooovT, T2old)\
            - 0.5*einsum('bicd,cdkl->bikl', IvovvT, TTemp)
        L1 += fac*einsum('jkab,bijk->ia', L2old, ITvooo)
        del ITvooo

        # VVVO
        ITvvvo = einsum('cjak,bj->cbak', I.vovo, T1old) - 0.5*I.vvvo\
            - 0.5*einsum('ka,bcjk->bcaj', F.ov, T2old)\
            - 0.5*einsum('cdab,bj->cdaj', I.vvvv, T1old)\
            - einsum('dkca,bcjk->bdaj', I.vovv, T2old)\
            - einsum('djak,bj->bdak', IvovoT1, T1old)\
            + 0.5*einsum('jkla,bcjk->bcal', I.ooov, TTemp)\
            - 0.5*einsum('la,cdkl->cdak', IovT, T2old)\
            + einsum('bkac,cdkl->bdal', IvovvT, T2old)\
            + 0.5*einsum('klja,cdkl->cdaj', IooovT, TTemp)
        L1 += fac*einsum('ikbc,cbak->ia', L2old, ITvvvo)
        del ITvvvo

        # OV
        ITov = IovT + F.ov
        temp = fac*einsum('jb,ia->ijab', L1old, ITov)
        L2 += temp
        L2 -= temp.transpose((1, 0, 2, 3))
        L2 -= temp.transpose((0, 1, 3, 2))
        L2 += temp.transpose((1, 0, 3, 2))
        del ITov

        # VOVV
        ITvovv = IvovvT - I.vovv
        temp = fac*einsum('ic,cjab->ijab', L1old, ITvovv)
        L2 -= temp
        L2 += temp.transpose((1, 0, 2, 3))
        del ITvovv

        # OOOV
        ITooov = IooovT + I.ooov
        temp = fac*einsum('ka,ijkb->ijab', L1old, ITooov)
        L2 -= temp
        L2 += temp.transpose((0, 1, 3, 2))
        del ITooov

        # VOVO
        ITvovo = - IvovoT1 - IvovoT2 - I.vovo\
            - einsum('djcb,ck->djbk', IvovvT, T1old)
        temp = fac*einsum('ikad,djbk->ijab', L2old, ITvovo)
        L2 += temp
        L2 -= temp.transpose((1, 0, 2, 3))
        L2 -= temp.transpose((0, 1, 3, 2))
        L2 += temp.transpose((1, 0, 3, 2))
        del ITvovo

        # VVVV
        ITvvvv = einsum('dkab,ck->cdab', I.vovv, T1old)\
            + 0.5*einsum('klab,cdkl->cdab', I.oovv, TTemp)\
            + 0.5*I.vvvv
        L2 += fac*einsum('ijcd,cdab->ijab', L2old, ITvvvv)
        del ITvvvv

        # OOOO
        IToooo = einsum('ijlc,ck->ijkl', I.ooov, T1old)\
            - 0.5*einsum('ijcd,cdkl->ijkl', I.oovv, TTemp)\
            - 0.5*I.oooo
        L2 -= fac*einsum('klab,ijkl->ijab', L2old, IToooo)
        del IToooo

        # OOVV
        IToovv = einsum('ljdb,cdkl->kjcb', I.oovv, T2old)
        temp = fac*einsum('ikac,kjcb->ijab', L2old, IToovv)
        L2 += temp
        L2 -= temp.transpose((1, 0, 2, 3))
        L2 -= temp.transpose((0, 1, 3, 2))
        L2 += temp.transpose((1, 0, 3, 2))
        del IToovv

        # LT terms
        Ltemp1 = einsum('jkbd,bcjk->cd', L2old, T2old)
        Ltemp2 = einsum('jlbc,bcjk->lk', L2old, T2old)

        L1 += 0.5*fac*einsum('cd,dica->ia', Ltemp1, I.vovv)

        L1 -= 0.5*fac*einsum('lk,kila->ia', Ltemp2, I.ooov)

        L1 -= 0.5*fac*einsum('db,bida->ia', Ltemp1, IvovvT)

        L1 -= 0.5*fac*einsum('jl,lija->ia', Ltemp2, IooovT)

        temp = 0.5*fac*einsum('da,ijdb->ijab', Ltemp1, I.oovv)
        L2 -= temp
        L2 += temp.transpose((0, 1, 3, 2))

        temp = 0.5*fac*einsum('il,ljab->ijab', Ltemp2, I.oovv)
        L2 -= temp
        L2 += temp.transpose((1, 0, 2, 3))

    return L1, L2


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

def calc_lam_resid(F, I, T1old, L1old, T2old, L2old):
    ''' Just return residual (debugging)'''

    #resid_eqns = 'l1_commutator_notconnected'
    resid_eqns = 'no removed terms'
    if resid_eqns.lower() == 'no removed terms':
        # Finding the connected LH term, but without removing the additional terms in L1
        L1 = 1.0*einsum('ia->ia', F.ov)
        L1 += -1.0*einsum('bj,jiab->ia', T1old, I.oovv)


        L1 += -1.0*einsum('ij,ja->ia', F.oo, L1old)
        L1 += 1.0*einsum('ba,ib->ia', F.vv, L1old)
        L1 += -1.0*einsum('ibja,jb->ia', I.ovov, L1old)
        L1 += -1.0*einsum('bj,jiab->ia', F.vo, L2old)
        L1 += 0.5*einsum('ibjk,kjab->ia', I.ovoo, L2old)
        L1 += -0.5*einsum('bcja,jicb->ia', I.vvov, L2old)
        L1 += -1.0*einsum('ib,bj,ja->ia', F.ov, T1old, L1old)
        L1 += -1.0*einsum('ja,bj,ib->ia', F.ov, T1old, L1old)
        L1 += -1.0*einsum('jika,bj,kb->ia', I.ooov, T1old, L1old)
        L1 += 1.0*einsum('ibac,cj,jb->ia', I.ovvv, T1old, L1old)
        L1 += 1.0*einsum('jikb,bj,ka->ia', I.ooov, T1old, L1old)
        L1 += -1.0*einsum('jbac,cj,ib->ia', I.ovvv, T1old, L1old)
        L1 += 1.0*einsum('jk,bj,kiab->ia', F.oo, T1old, L2old)
        L1 += -1.0*einsum('bc,cj,jiab->ia', F.vv, T1old, L2old)
        L1 += 1.0*einsum('jiab,cbjk,kc->ia', I.oovv, T2old, L1old)
        L1 += 0.5*einsum('jibc,cbjk,ka->ia', I.oovv, T2old, L1old)
        L1 += 0.5*einsum('jkab,cbkj,ic->ia', I.oovv, T2old, L1old)
        L1 += -0.5*einsum('ib,cbjk,kjac->ia', F.ov, T2old, L2old)
        L1 += -1.0*einsum('ibjc,ck,jkab->ia', I.ovov, T1old, L2old)
        L1 += 0.5*einsum('jikl,bj,lkab->ia', I.oooo, T1old, L2old)
        L1 += -0.5*einsum('ja,bcjk,kicb->ia', F.ov, T2old, L2old)
        L1 += -1.0*einsum('jbka,cj,kicb->ia', I.ovov, T1old, L2old)
        L1 += 1.0*einsum('jb,cbjk,kiac->ia', F.ov, T2old, L2old)
        L1 += 1.0*einsum('jbkc,cj,kiab->ia', I.ovov, T1old, L2old)
        L1 += 0.5*einsum('bcad,dj,jicb->ia', I.vvvv, T1old, L2old)
        L1 += 0.5*einsum('jika,bcjl,klcb->ia', I.ooov, T2old, L2old)
        L1 += -0.5*einsum('ibac,dcjk,kjdb->ia', I.ovvv, T2old, L2old)
        L1 += -1.0*einsum('jikb,cbjl,klac->ia', I.ooov, T2old, L2old)
        L1 += -0.25*einsum('ibcd,dcjk,kjab->ia', I.ovvv, T2old, L2old)
        L1 += 0.25*einsum('jkla,bckj,licb->ia', I.ooov, T2old, L2old)
        L1 += 1.0*einsum('jbac,dcjk,kidb->ia', I.ovvv, T2old, L2old)
        L1 += -0.5*einsum('jklb,cbkj,liac->ia', I.ooov, T2old, L2old)
        L1 += 0.5*einsum('jbcd,dcjk,kiab->ia', I.ovvv, T2old, L2old)
        L1 += 1.0*einsum('jiab,bk,cj,kc->ia', I.oovv, T1old, T1old, L1old)
        L1 += -1.0*einsum('jibc,ck,bj,ka->ia', I.oovv, T1old, T1old, L1old)
        L1 += -1.0*einsum('jkab,cj,bk,ic->ia', I.oovv, T1old, T1old, L1old)
        L1 += 1.0*einsum('jb,bk,cj,kiac->ia', F.ov, T1old, T1old, L2old)
        L1 += -1.0*einsum('jikb,bl,cj,klac->ia', I.ooov, T1old, T1old, L2old)
        L1 += 0.5*einsum('ibcd,dj,ck,jkab->ia', I.ovvv, T1old, T1old, L2old)
        L1 += -0.5*einsum('jkla,bj,ck,licb->ia', I.ooov, T1old, T1old, L2old)
        L1 += 1.0*einsum('jbac,ck,dj,kidb->ia', I.ovvv, T1old, T1old, L2old)
        L1 += 1.0*einsum('jklb,cj,bk,liac->ia', I.ooov, T1old, T1old, L2old)
        L1 += -1.0*einsum('jbcd,dk,cj,kiab->ia', I.ovvv, T1old, T1old, L2old)
        L1 += -0.5*einsum('jiab,cj,dbkl,lkdc->ia', I.oovv, T1old, T2old, L2old)
        L1 += -0.5*einsum('jiab,bk,cdjl,kldc->ia', I.oovv, T1old, T2old, L2old)
        L1 += -0.25*einsum('jibc,dj,cbkl,lkad->ia', I.oovv, T1old, T2old, L2old)
        L1 += 1.0*einsum('jibc,ck,dbjl,klad->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jibc,cj,dbkl,lkad->ia', I.oovv, T1old, T2old, L2old)
        L1 += 1.0*einsum('jkab,cj,dbkl,lidc->ia', I.oovv, T1old, T2old, L2old)
        L1 += -0.25*einsum('jkab,bl,cdkj,lidc->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jkab,bj,cdkl,lidc->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jkbc,dj,cbkl,liad->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jkbc,cl,dbkj,liad->ia', I.oovv, T1old, T2old, L2old)
        L1 += -1.0*einsum('jkbc,cj,dbkl,liad->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jibc,ck,bl,dj,klad->ia', I.oovv, T1old, T1old, T1old, L2old)
        L1 += 0.5*einsum('jkab,bl,cj,dk,lidc->ia', I.oovv, T1old, T1old, T1old, L2old)
        L1 += -1.0*einsum('jkbc,cl,dj,bk,liad->ia', I.oovv, T1old, T1old, T1old, L2old)

        # Piece independent of Lambda
        L2 = I.oovv.copy().transpose(1,0,3,2)
        # Piece proportional to lambda
        L2 += 1.0*einsum('jikb,ka->ijab', I.ooov, L1old)
        L2 += -1.0*einsum('jika,kb->ijab', I.ooov, L1old)
        L2 += 1.0*einsum('jcba,ic->ijab', I.ovvv, L1old)
        L2 += -1.0*einsum('icba,jc->ijab', I.ovvv, L1old)
        L2 += -1.0*einsum('jk,kiba->ijab', F.oo, L2old)
        L2 += 1.0*einsum('ik,kjba->ijab', F.oo, L2old)
        L2 += -1.0*einsum('cb,jiac->ijab', F.vv, L2old)
        L2 += 1.0*einsum('ca,jibc->ijab', F.vv, L2old)
        L2 += -0.5*einsum('jikl,lkba->ijab', I.oooo, L2old)
        L2 += 1.0*einsum('jckb,kiac->ijab', I.ovov, L2old)
        L2 += -1.0*einsum('jcka,kibc->ijab', I.ovov, L2old)
        L2 += -1.0*einsum('ickb,kjac->ijab', I.ovov, L2old)
        L2 += 1.0*einsum('icka,kjbc->ijab', I.ovov, L2old)
        L2 += -0.5*einsum('cdba,jidc->ijab', I.vvvv, L2old)
        L2 += -1.0*einsum('jibc,ck,ka->ijab', I.oovv, T1old, L1old)
        L2 += 1.0*einsum('jiac,ck,kb->ijab', I.oovv, T1old, L1old)
        L2 += 1.0*einsum('kjba,ck,ic->ijab', I.oovv, T1old, L1old)
        L2 += -1.0*einsum('kiba,ck,jc->ijab', I.oovv, T1old, L1old)
        L2 += -1.0*einsum('jc,ck,kiba->ijab', F.ov, T1old, L2old)
        L2 += 1.0*einsum('ic,ck,kjba->ijab', F.ov, T1old, L2old)
        L2 += 1.0*einsum('kb,ck,jiac->ijab', F.ov, T1old, L2old)
        L2 += -1.0*einsum('ka,ck,jibc->ijab', F.ov, T1old, L2old)
        L2 += 1.0*einsum('jikc,cl,klba->ijab', I.ooov, T1old, L2old)
        L2 += 1.0*einsum('kjlb,ck,liac->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('jcbd,dk,kiac->ijab', I.ovvv, T1old, L2old)
        L2 += -1.0*einsum('kjla,ck,libc->ijab', I.ooov, T1old, L2old)
        L2 += 1.0*einsum('jcad,dk,kibc->ijab', I.ovvv, T1old, L2old)
        L2 += 1.0*einsum('kjlc,ck,liba->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('kilb,ck,ljac->ijab', I.ooov, T1old, L2old)
        L2 += 1.0*einsum('icbd,dk,kjac->ijab', I.ovvv, T1old, L2old)
        L2 += 1.0*einsum('kila,ck,ljbc->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('icad,dk,kjbc->ijab', I.ovvv, T1old, L2old)
        L2 += -1.0*einsum('kilc,ck,ljba->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('kcba,dk,jidc->ijab', I.ovvv, T1old, L2old)
        L2 += 1.0*einsum('kcbd,dk,jiac->ijab', I.ovvv, T1old, L2old)
        L2 += -1.0*einsum('kcad,dk,jibc->ijab', I.ovvv, T1old, L2old)
        L2 += -0.5*einsum('jibc,dckl,lkad->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('jiac,dckl,lkbd->ijab', I.oovv, T2old, L2old)
        L2 += 0.25*einsum('jicd,dckl,lkba->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('kjba,cdkl,lidc->ijab', I.oovv, T2old, L2old)
        L2 += -1.0*einsum('kjbc,dckl,liad->ijab', I.oovv, T2old, L2old)
        L2 += 1.0*einsum('kjac,dckl,libd->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('kjcd,dckl,liba->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('kiba,cdkl,ljdc->ijab', I.oovv, T2old, L2old)
        L2 += 1.0*einsum('kibc,dckl,ljad->ijab', I.oovv, T2old, L2old)
        L2 += -1.0*einsum('kiac,dckl,ljbd->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('kicd,dckl,ljba->ijab', I.oovv, T2old, L2old)
        L2 += 0.25*einsum('klba,cdlk,jidc->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('klbc,dclk,jiad->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('klac,dclk,jibd->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('jicd,dk,cl,klba->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('kjbc,cl,dk,liad->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('kjac,cl,dk,libd->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('kjcd,dl,ck,liba->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('kibc,cl,dk,ljad->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('kiac,cl,dk,ljbd->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('kicd,dl,ck,ljba->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -0.5*einsum('klba,ck,dl,jidc->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('klbc,dk,cl,jiad->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('klac,dk,cl,jibd->ijab', I.oovv, T1old, T1old, L2old)
        # Disconnected piece, with intermediate projections onto singles space
        L2 += 1.0*einsum('jb,ia->ijab', L1old, F.ov)
        L2 += -1.0*einsum('ja,ib->ijab', L1old, F.ov)
        L2 += -1.0*einsum('ib,ja->ijab', L1old, F.ov)
        L2 += 1.0*einsum('ia,jb->ijab', L1old, F.ov)
        L2 += -1.0*einsum('ck,jb,kiac->ijab', T1old, L1old, I.oovv)
        L2 += 1.0*einsum('ck,ja,kibc->ijab', T1old, L1old, I.oovv)
        L2 += 1.0*einsum('ck,ib,kjac->ijab', T1old, L1old, I.oovv)
        L2 += -1.0*einsum('ck,ia,kjbc->ijab', T1old, L1old, I.oovv)
    elif resid_eqns.lower() == 'l1_commutator_notconnected':
        # Finding the commutator term for L1 directly, without using 'get_connected'
        L1 = F.ov.copy()
        L1 += -1.0*einsum('bj,jiab->ia', T1old, I.oovv)

        L1 += -1.0*einsum('ij,ja->ia', F.oo, L1old)
        L1 += 1.0*einsum('ba,ib->ia', F.vv, L1old)
        L1 += -1.0*einsum('ibja,jb->ia', I.ovov, L1old)
        L1 += -1.0*einsum('bj,jiab->ia', F.vo, L2old)
        L1 += 0.5*einsum('ibjk,kjab->ia', I.ovoo, L2old)
        L1 += -0.5*einsum('bcja,jicb->ia', I.vvov, L2old)
        L1 += -1.0*einsum('ib,bj,ja->ia', F.ov, T1old, L1old)
        L1 += -1.0*einsum('ja,bj,ib->ia', F.ov, T1old, L1old)
        L1 += -1.0*einsum('jika,bj,kb->ia', I.ooov, T1old, L1old)
        L1 += 1.0*einsum('ibac,cj,jb->ia', I.ovvv, T1old, L1old)
        L1 += 1.0*einsum('jikb,bj,ka->ia', I.ooov, T1old, L1old)
        L1 += -1.0*einsum('jbac,cj,ib->ia', I.ovvv, T1old, L1old)
        L1 += 1.0*einsum('jk,bj,kiab->ia', F.oo, T1old, L2old)
        L1 += -1.0*einsum('bc,cj,jiab->ia', F.vv, T1old, L2old)
        L1 += 1.0*einsum('jiab,cbjk,kc->ia', I.oovv, T2old, L1old)
        L1 += 0.5*einsum('jibc,cbjk,ka->ia', I.oovv, T2old, L1old)
        L1 += 0.5*einsum('jkab,cbkj,ic->ia', I.oovv, T2old, L1old)
        L1 += -0.5*einsum('ib,cbjk,kjac->ia', F.ov, T2old, L2old)
        L1 += -1.0*einsum('ibjc,ck,jkab->ia', I.ovov, T1old, L2old)
        L1 += 0.5*einsum('jikl,bj,lkab->ia', I.oooo, T1old, L2old)
        L1 += -0.5*einsum('ja,bcjk,kicb->ia', F.ov, T2old, L2old)
        L1 += -1.0*einsum('jbka,cj,kicb->ia', I.ovov, T1old, L2old)
        L1 += 1.0*einsum('jb,cbjk,kiac->ia', F.ov, T2old, L2old)
        L1 += 1.0*einsum('jbkc,cj,kiab->ia', I.ovov, T1old, L2old)
        L1 += 0.5*einsum('bcad,dj,jicb->ia', I.vvvv, T1old, L2old)
        L1 += 0.5*einsum('jika,bcjl,klcb->ia', I.ooov, T2old, L2old)
        L1 += -0.5*einsum('ibac,dcjk,kjdb->ia', I.ovvv, T2old, L2old)
        L1 += -1.0*einsum('jikb,cbjl,klac->ia', I.ooov, T2old, L2old)
        L1 += -0.25*einsum('ibcd,dcjk,kjab->ia', I.ovvv, T2old, L2old)
        L1 += 0.25*einsum('jkla,bckj,licb->ia', I.ooov, T2old, L2old)
        L1 += 1.0*einsum('jbac,dcjk,kidb->ia', I.ovvv, T2old, L2old)
        L1 += -0.5*einsum('jklb,cbkj,liac->ia', I.ooov, T2old, L2old)
        L1 += 0.5*einsum('jbcd,dcjk,kiab->ia', I.ovvv, T2old, L2old)
        L1 += 1.0*einsum('jiab,bk,cj,kc->ia', I.oovv, T1old, T1old, L1old)
        L1 += -1.0*einsum('jibc,ck,bj,ka->ia', I.oovv, T1old, T1old, L1old)
        L1 += -1.0*einsum('jkab,cj,bk,ic->ia', I.oovv, T1old, T1old, L1old)
        L1 += 1.0*einsum('jb,bk,cj,kiac->ia', F.ov, T1old, T1old, L2old)
        L1 += -1.0*einsum('jikb,bl,cj,klac->ia', I.ooov, T1old, T1old, L2old)
        L1 += 0.5*einsum('ibcd,dj,ck,jkab->ia', I.ovvv, T1old, T1old, L2old)
        L1 += -0.5*einsum('jkla,bj,ck,licb->ia', I.ooov, T1old, T1old, L2old)
        L1 += 1.0*einsum('jbac,ck,dj,kidb->ia', I.ovvv, T1old, T1old, L2old)
        L1 += 1.0*einsum('jklb,cj,bk,liac->ia', I.ooov, T1old, T1old, L2old)
        L1 += -1.0*einsum('jbcd,dk,cj,kiab->ia', I.ovvv, T1old, T1old, L2old)
        L1 += -0.5*einsum('jiab,cj,dbkl,lkdc->ia', I.oovv, T1old, T2old, L2old)
        L1 += -0.5*einsum('jiab,bk,cdjl,kldc->ia', I.oovv, T1old, T2old, L2old)
        L1 += -0.25*einsum('jibc,dj,cbkl,lkad->ia', I.oovv, T1old, T2old, L2old)
        L1 += 1.0*einsum('jibc,ck,dbjl,klad->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jibc,cj,dbkl,lkad->ia', I.oovv, T1old, T2old, L2old)
        L1 += 1.0*einsum('jkab,cj,dbkl,lidc->ia', I.oovv, T1old, T2old, L2old)
        L1 += -0.25*einsum('jkab,bl,cdkj,lidc->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jkab,bj,cdkl,lidc->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jkbc,dj,cbkl,liad->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jkbc,cl,dbkj,liad->ia', I.oovv, T1old, T2old, L2old)
        L1 += -1.0*einsum('jkbc,cj,dbkl,liad->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jibc,ck,bl,dj,klad->ia', I.oovv, T1old, T1old, T1old, L2old)
        L1 += 0.5*einsum('jkab,bl,cj,dk,lidc->ia', I.oovv, T1old, T1old, T1old, L2old)
        L1 += -1.0*einsum('jkbc,cl,dj,bk,liad->ia', I.oovv, T1old, T1old, T1old, L2old)
        # Piece independent of Lambda
        L2 = I.oovv.copy().transpose(1,0,3,2)
        # Piece proportional to lambda
        L2 += 1.0*einsum('jikb,ka->ijab', I.ooov, L1old)
        L2 += -1.0*einsum('jika,kb->ijab', I.ooov, L1old)
        L2 += 1.0*einsum('jcba,ic->ijab', I.ovvv, L1old)
        L2 += -1.0*einsum('icba,jc->ijab', I.ovvv, L1old)
        L2 += -1.0*einsum('jk,kiba->ijab', F.oo, L2old)
        L2 += 1.0*einsum('ik,kjba->ijab', F.oo, L2old)
        L2 += -1.0*einsum('cb,jiac->ijab', F.vv, L2old)
        L2 += 1.0*einsum('ca,jibc->ijab', F.vv, L2old)
        L2 += -0.5*einsum('jikl,lkba->ijab', I.oooo, L2old)
        L2 += 1.0*einsum('jckb,kiac->ijab', I.ovov, L2old)
        L2 += -1.0*einsum('jcka,kibc->ijab', I.ovov, L2old)
        L2 += -1.0*einsum('ickb,kjac->ijab', I.ovov, L2old)
        L2 += 1.0*einsum('icka,kjbc->ijab', I.ovov, L2old)
        L2 += -0.5*einsum('cdba,jidc->ijab', I.vvvv, L2old)
        L2 += -1.0*einsum('jibc,ck,ka->ijab', I.oovv, T1old, L1old)
        L2 += 1.0*einsum('jiac,ck,kb->ijab', I.oovv, T1old, L1old)
        L2 += 1.0*einsum('kjba,ck,ic->ijab', I.oovv, T1old, L1old)
        L2 += -1.0*einsum('kiba,ck,jc->ijab', I.oovv, T1old, L1old)
        L2 += -1.0*einsum('jc,ck,kiba->ijab', F.ov, T1old, L2old)
        L2 += 1.0*einsum('ic,ck,kjba->ijab', F.ov, T1old, L2old)
        L2 += 1.0*einsum('kb,ck,jiac->ijab', F.ov, T1old, L2old)
        L2 += -1.0*einsum('ka,ck,jibc->ijab', F.ov, T1old, L2old)
        L2 += 1.0*einsum('jikc,cl,klba->ijab', I.ooov, T1old, L2old)
        L2 += 1.0*einsum('kjlb,ck,liac->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('jcbd,dk,kiac->ijab', I.ovvv, T1old, L2old)
        L2 += -1.0*einsum('kjla,ck,libc->ijab', I.ooov, T1old, L2old)
        L2 += 1.0*einsum('jcad,dk,kibc->ijab', I.ovvv, T1old, L2old)
        L2 += 1.0*einsum('kjlc,ck,liba->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('kilb,ck,ljac->ijab', I.ooov, T1old, L2old)
        L2 += 1.0*einsum('icbd,dk,kjac->ijab', I.ovvv, T1old, L2old)
        L2 += 1.0*einsum('kila,ck,ljbc->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('icad,dk,kjbc->ijab', I.ovvv, T1old, L2old)
        L2 += -1.0*einsum('kilc,ck,ljba->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('kcba,dk,jidc->ijab', I.ovvv, T1old, L2old)
        L2 += 1.0*einsum('kcbd,dk,jiac->ijab', I.ovvv, T1old, L2old)
        L2 += -1.0*einsum('kcad,dk,jibc->ijab', I.ovvv, T1old, L2old)
        L2 += -0.5*einsum('jibc,dckl,lkad->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('jiac,dckl,lkbd->ijab', I.oovv, T2old, L2old)
        L2 += 0.25*einsum('jicd,dckl,lkba->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('kjba,cdkl,lidc->ijab', I.oovv, T2old, L2old)
        L2 += -1.0*einsum('kjbc,dckl,liad->ijab', I.oovv, T2old, L2old)
        L2 += 1.0*einsum('kjac,dckl,libd->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('kjcd,dckl,liba->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('kiba,cdkl,ljdc->ijab', I.oovv, T2old, L2old)
        L2 += 1.0*einsum('kibc,dckl,ljad->ijab', I.oovv, T2old, L2old)
        L2 += -1.0*einsum('kiac,dckl,ljbd->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('kicd,dckl,ljba->ijab', I.oovv, T2old, L2old)
        L2 += 0.25*einsum('klba,cdlk,jidc->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('klbc,dclk,jiad->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('klac,dclk,jibd->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('jicd,dk,cl,klba->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('kjbc,cl,dk,liad->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('kjac,cl,dk,libd->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('kjcd,dl,ck,liba->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('kibc,cl,dk,ljad->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('kiac,cl,dk,ljbd->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('kicd,dl,ck,ljba->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -0.5*einsum('klba,ck,dl,jidc->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('klbc,dk,cl,jiad->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('klac,dk,cl,jibd->ijab', I.oovv, T1old, T1old, L2old)
        # Disconnected piece, with intermediate projections onto singles space
        L2 += 1.0*einsum('jb,ia->ijab', L1old, F.ov)
        L2 += -1.0*einsum('ja,ib->ijab', L1old, F.ov)
        L2 += -1.0*einsum('ib,ja->ijab', L1old, F.ov)
        L2 += 1.0*einsum('ia,jb->ijab', L1old, F.ov)
        L2 += -1.0*einsum('ck,jb,kiac->ijab', T1old, L1old, I.oovv)
        L2 += 1.0*einsum('ck,ja,kibc->ijab', T1old, L1old, I.oovv)
        L2 += 1.0*einsum('ck,ib,kjac->ijab', T1old, L1old, I.oovv)
        L2 += -1.0*einsum('ck,ia,kjbc->ijab', T1old, L1old, I.oovv)
    else:
        # Code that is actually used in the optimization of the lambda amplitudes
        # Piece not dependent on lamdba
        L1 = F.ov.copy()
        L1 += -1.0*einsum('bj,jiab->ia', T1old, I.oovv)
        # Piece proportional to lambda
        L1 += -1.0*einsum('ij,ja->ia', F.oo, L1old)
        L1 += 1.0*einsum('ba,ib->ia', F.vv, L1old)
        L1 += -1.0*einsum('ibja,jb->ia', I.ovov, L1old)
        L1 += 0.5*einsum('ibjk,kjab->ia', I.ovoo, L2old)
        L1 += -0.5*einsum('bcja,jicb->ia', I.vvov, L2old)
        L1 += -1.0*einsum('ja,bj,ib->ia', F.ov, T1old, L1old)
        L1 += -1.0*einsum('ib,bj,ja->ia', F.ov, T1old, L1old)
        L1 += 1.0*einsum('jikb,bj,ka->ia', I.ooov, T1old, L1old)
        L1 += -1.0*einsum('jika,bj,kb->ia', I.ooov, T1old, L1old)
        L1 += -1.0*einsum('jbac,cj,ib->ia', I.ovvv, T1old, L1old)
        L1 += 1.0*einsum('ibac,cj,jb->ia', I.ovvv, T1old, L1old)
        L1 += -0.5*einsum('ja,bcjk,kicb->ia', F.ov, T2old, L2old)
        L1 += -0.5*einsum('ib,cbjk,kjac->ia', F.ov, T2old, L2old)
        L1 += 0.5*einsum('jkab,cbkj,ic->ia', I.oovv, T2old, L1old)
        L1 += 0.5*einsum('jibc,cbjk,ka->ia', I.oovv, T2old, L1old)
        L1 += 1.0*einsum('jiab,cbjk,kc->ia', I.oovv, T2old, L1old)
        L1 += -1.0*einsum('jbka,cj,kicb->ia', I.ovov, T1old, L2old)
        L1 += -1.0*einsum('ibjc,ck,jkab->ia', I.ovov, T1old, L2old)
        L1 += 0.5*einsum('jikl,bj,lkab->ia', I.oooo, T1old, L2old)
        L1 += 0.5*einsum('bcad,dj,jicb->ia', I.vvvv, T1old, L2old)
        L1 += 0.25*einsum('jkla,bckj,licb->ia', I.ooov, T2old, L2old)
        L1 += -1.0*einsum('jikb,cbjl,klac->ia', I.ooov, T2old, L2old)
        L1 += 0.5*einsum('jika,bcjl,klcb->ia', I.ooov, T2old, L2old)
        L1 += 1.0*einsum('jbac,dcjk,kidb->ia', I.ovvv, T2old, L2old)
        L1 += -0.25*einsum('ibcd,dcjk,kjab->ia', I.ovvv, T2old, L2old)
        L1 += -0.5*einsum('ibac,dcjk,kjdb->ia', I.ovvv, T2old, L2old)
        L1 += -1.0*einsum('jkab,cj,bk,ic->ia', I.oovv, T1old, T1old, L1old)
        L1 += -1.0*einsum('jibc,ck,bj,ka->ia', I.oovv, T1old, T1old, L1old)
        L1 += 1.0*einsum('jiab,bk,cj,kc->ia', I.oovv, T1old, T1old, L1old)
        L1 += -0.5*einsum('jkla,bj,ck,licb->ia', I.ooov, T1old, T1old, L2old)
        L1 += -1.0*einsum('jikb,bl,cj,klac->ia', I.ooov, T1old, T1old, L2old)
        L1 += 1.0*einsum('jbac,ck,dj,kidb->ia', I.ovvv, T1old, T1old, L2old)
        L1 += 0.5*einsum('ibcd,dj,ck,jkab->ia', I.ovvv, T1old, T1old, L2old)
        L1 += 1.0*einsum('jkab,cj,dbkl,lidc->ia', I.oovv, T1old, T2old, L2old)
        L1 += -0.25*einsum('jkab,bl,cdkj,lidc->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jkab,bj,cdkl,lidc->ia', I.oovv, T1old, T2old, L2old)
        L1 += -0.25*einsum('jibc,dj,cbkl,lkad->ia', I.oovv, T1old, T2old, L2old)
        L1 += 1.0*einsum('jibc,ck,dbjl,klad->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jibc,cj,dbkl,lkad->ia', I.oovv, T1old, T2old, L2old)
        L1 += -0.5*einsum('jiab,cj,dbkl,lkdc->ia', I.oovv, T1old, T2old, L2old)
        L1 += -0.5*einsum('jiab,bk,cdjl,kldc->ia', I.oovv, T1old, T2old, L2old)
        L1 += 0.5*einsum('jkab,bl,cj,dk,lidc->ia', I.oovv, T1old, T1old, T1old, L2old)
        L1 += 0.5*einsum('jibc,ck,bl,dj,klad->ia', I.oovv, T1old, T1old, T1old, L2old)

        # Piece independent of Lambda
        L2 = I.oovv.copy().transpose(1,0,3,2)
        # Piece proportional to lambda
        L2 += 1.0*einsum('jikb,ka->ijab', I.ooov, L1old)
        L2 += -1.0*einsum('jika,kb->ijab', I.ooov, L1old)
        L2 += 1.0*einsum('jcba,ic->ijab', I.ovvv, L1old)
        L2 += -1.0*einsum('icba,jc->ijab', I.ovvv, L1old)
        L2 += -1.0*einsum('jk,kiba->ijab', F.oo, L2old)
        L2 += 1.0*einsum('ik,kjba->ijab', F.oo, L2old)
        L2 += -1.0*einsum('cb,jiac->ijab', F.vv, L2old)
        L2 += 1.0*einsum('ca,jibc->ijab', F.vv, L2old)
        L2 += -0.5*einsum('jikl,lkba->ijab', I.oooo, L2old)
        L2 += 1.0*einsum('jckb,kiac->ijab', I.ovov, L2old)
        L2 += -1.0*einsum('jcka,kibc->ijab', I.ovov, L2old)
        L2 += -1.0*einsum('ickb,kjac->ijab', I.ovov, L2old)
        L2 += 1.0*einsum('icka,kjbc->ijab', I.ovov, L2old)
        L2 += -0.5*einsum('cdba,jidc->ijab', I.vvvv, L2old)
        L2 += -1.0*einsum('jibc,ck,ka->ijab', I.oovv, T1old, L1old)
        L2 += 1.0*einsum('jiac,ck,kb->ijab', I.oovv, T1old, L1old)
        L2 += 1.0*einsum('kjba,ck,ic->ijab', I.oovv, T1old, L1old)
        L2 += -1.0*einsum('kiba,ck,jc->ijab', I.oovv, T1old, L1old)
        L2 += -1.0*einsum('jc,ck,kiba->ijab', F.ov, T1old, L2old)
        L2 += 1.0*einsum('ic,ck,kjba->ijab', F.ov, T1old, L2old)
        L2 += 1.0*einsum('kb,ck,jiac->ijab', F.ov, T1old, L2old)
        L2 += -1.0*einsum('ka,ck,jibc->ijab', F.ov, T1old, L2old)
        L2 += 1.0*einsum('jikc,cl,klba->ijab', I.ooov, T1old, L2old)
        L2 += 1.0*einsum('kjlb,ck,liac->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('jcbd,dk,kiac->ijab', I.ovvv, T1old, L2old)
        L2 += -1.0*einsum('kjla,ck,libc->ijab', I.ooov, T1old, L2old)
        L2 += 1.0*einsum('jcad,dk,kibc->ijab', I.ovvv, T1old, L2old)
        L2 += 1.0*einsum('kjlc,ck,liba->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('kilb,ck,ljac->ijab', I.ooov, T1old, L2old)
        L2 += 1.0*einsum('icbd,dk,kjac->ijab', I.ovvv, T1old, L2old)
        L2 += 1.0*einsum('kila,ck,ljbc->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('icad,dk,kjbc->ijab', I.ovvv, T1old, L2old)
        L2 += -1.0*einsum('kilc,ck,ljba->ijab', I.ooov, T1old, L2old)
        L2 += -1.0*einsum('kcba,dk,jidc->ijab', I.ovvv, T1old, L2old)
        L2 += 1.0*einsum('kcbd,dk,jiac->ijab', I.ovvv, T1old, L2old)
        L2 += -1.0*einsum('kcad,dk,jibc->ijab', I.ovvv, T1old, L2old)
        L2 += -0.5*einsum('jibc,dckl,lkad->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('jiac,dckl,lkbd->ijab', I.oovv, T2old, L2old)
        L2 += 0.25*einsum('jicd,dckl,lkba->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('kjba,cdkl,lidc->ijab', I.oovv, T2old, L2old)
        L2 += -1.0*einsum('kjbc,dckl,liad->ijab', I.oovv, T2old, L2old)
        L2 += 1.0*einsum('kjac,dckl,libd->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('kjcd,dckl,liba->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('kiba,cdkl,ljdc->ijab', I.oovv, T2old, L2old)
        L2 += 1.0*einsum('kibc,dckl,ljad->ijab', I.oovv, T2old, L2old)
        L2 += -1.0*einsum('kiac,dckl,ljbd->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('kicd,dckl,ljba->ijab', I.oovv, T2old, L2old)
        L2 += 0.25*einsum('klba,cdlk,jidc->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('klbc,dclk,jiad->ijab', I.oovv, T2old, L2old)
        L2 += 0.5*einsum('klac,dclk,jibd->ijab', I.oovv, T2old, L2old)
        L2 += -0.5*einsum('jicd,dk,cl,klba->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('kjbc,cl,dk,liad->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('kjac,cl,dk,libd->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('kjcd,dl,ck,liba->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('kibc,cl,dk,ljad->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('kiac,cl,dk,ljbd->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('kicd,dl,ck,ljba->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -0.5*einsum('klba,ck,dl,jidc->ijab', I.oovv, T1old, T1old, L2old)
        L2 += 1.0*einsum('klbc,dk,cl,jiad->ijab', I.oovv, T1old, T1old, L2old)
        L2 += -1.0*einsum('klac,dk,cl,jibd->ijab', I.oovv, T1old, T1old, L2old)
        # Disconnected piece, with intermediate projections onto singles space
        L2 += 1.0*einsum('jb,ia->ijab', L1old, F.ov)
        L2 += -1.0*einsum('ja,ib->ijab', L1old, F.ov)
        L2 += -1.0*einsum('ib,ja->ijab', L1old, F.ov)
        L2 += 1.0*einsum('ia,jb->ijab', L1old, F.ov)
        L2 += -1.0*einsum('ck,jb,kiac->ijab', T1old, L1old, I.oovv)
        L2 += 1.0*einsum('ck,ja,kibc->ijab', T1old, L1old, I.oovv)
        L2 += 1.0*einsum('ck,ib,kjac->ijab', T1old, L1old, I.oovv)
        L2 += -1.0*einsum('ck,ia,kjbc->ijab', T1old, L1old, I.oovv)

    res = np.linalg.norm(L1) / np.sqrt(T1old.size) + np.linalg.norm(L2) / np.sqrt(T2old.size)

    return res


def calc_lam_hbar(F, I, T1old, L1old, T2old, L2old):
    ''' Calculate <0| Lambda Hbar |0> and return the result (for debugging)'''

    E = 1.0*einsum('ia,ai->', L1old, F.vo)
    E += 0.25*einsum('ijab,baji->', L2old, I.vvoo)
    E += -1.0*einsum('ia,ji,aj->', L1old, F.oo, T1old)
    E += 1.0*einsum('ia,ab,bi->', L1old, F.vv, T1old)
    E += -1.0*einsum('ia,jb,abji->', L1old, F.ov, T2old)
    E += -1.0*einsum('ia,jaib,bj->', L1old, I.ovov, T1old)
    E += 0.5*einsum('ia,jkib,abkj->', L1old, I.ooov, T2old)
    E += -0.5*einsum('ia,jabc,cbji->', L1old, I.ovvv, T2old)
    E += 0.5*einsum('ijab,ki,bakj->', L2old, F.oo, T2old)
    E += -0.5*einsum('ijab,bc,acji->', L2old, F.vv, T2old)
    E += 0.5*einsum('ijab,kbji,ak->', L2old, I.ovoo, T1old)
    E += -0.5*einsum('ijab,baic,cj->', L2old, I.vvov, T1old)
    E += -0.125*einsum('ijab,klji,balk->', L2old, I.oooo, T2old)
    E += -1.0*einsum('ijab,kbic,ackj->', L2old, I.ovov, T2old)
    E += -0.125*einsum('ijab,bacd,dcji->', L2old, I.vvvv, T2old)
    E += -1.0*einsum('ia,jb,bi,aj->', L1old, F.ov, T1old, T1old)
    E += -1.0*einsum('ia,jkib,aj,bk->', L1old, I.ooov, T1old, T1old)
    E += 1.0*einsum('ia,jabc,ci,bj->', L1old, I.ovvv, T1old, T1old)
    E += -0.5*einsum('ia,jkbc,aj,cbki->', L1old, I.oovv, T1old, T2old)
    E += -0.5*einsum('ia,jkbc,ci,abkj->', L1old, I.oovv, T1old, T2old)
    E += 1.0*einsum('ia,jkbc,cj,abki->', L1old, I.oovv, T1old, T2old)
    E += 0.5*einsum('ijab,kc,bk,acji->', L2old, F.ov, T1old, T2old)
    E += 0.5*einsum('ijab,kc,ci,bakj->', L2old, F.ov, T1old, T2old)
    E += 0.25*einsum('ijab,klji,bk,al->', L2old, I.oooo, T1old, T1old)
    E += -1.0*einsum('ijab,kbic,cj,ak->', L2old, I.ovov, T1old, T1old)
    E += 0.25*einsum('ijab,bacd,di,cj->', L2old, I.vvvv, T1old, T1old)
    E += -1.0*einsum('ijab,klic,bk,aclj->', L2old, I.ooov, T1old, T2old)
    E += 0.25*einsum('ijab,klic,cj,balk->', L2old, I.ooov, T1old, T2old)
    E += -0.5*einsum('ijab,klic,ck,balj->', L2old, I.ooov, T1old, T2old)
    E += -0.25*einsum('ijab,kbcd,ak,dcji->', L2old, I.ovvv, T1old, T2old)
    E += 1.0*einsum('ijab,kbcd,di,ackj->', L2old, I.ovvv, T1old, T2old)
    E += 0.5*einsum('ijab,kbcd,dk,acji->', L2old, I.ovvv, T1old, T2old)
    E += -0.25*einsum('ijab,klcd,baki,dclj->', L2old, I.oovv, T2old, T2old)
    E += -0.25*einsum('ijab,klcd,bdji,aclk->', L2old, I.oovv, T2old, T2old)
    E += 0.5*einsum('ijab,klcd,bdki,aclj->', L2old, I.oovv, T2old, T2old)
    E += 0.0625*einsum('ijab,klcd,dcji,balk->', L2old, I.oovv, T2old, T2old)
    E += 1.0*einsum('ia,jkbc,ci,aj,bk->', L1old, I.oovv, T1old, T1old, T1old)
    E += -0.5*einsum('ijab,klic,cj,bk,al->', L2old, I.ooov, T1old, T1old, T1old)
    E += 0.5*einsum('ijab,kbcd,di,cj,ak->', L2old, I.ovvv, T1old, T1old, T1old)
    E += -0.125*einsum('ijab,klcd,bk,al,dcji->', L2old, I.oovv, T1old, T1old, T2old)
    E += 0.5*einsum('ijab,klcd,bk,dl,acji->', L2old, I.oovv, T1old, T1old, T2old)
    E += 1.0*einsum('ijab,klcd,di,bk,aclj->', L2old, I.oovv, T1old, T1old, T2old)
    E += -0.125*einsum('ijab,klcd,di,cj,balk->', L2old, I.oovv, T1old, T1old, T2old)
    E += 0.5*einsum('ijab,klcd,di,ck,balj->', L2old, I.oovv, T1old, T1old, T2old)
    E += 0.25*einsum('ijab,klcd,di,cj,bk,al->', L2old, I.oovv, T1old, T1old, T1old, T1old)

    return E
