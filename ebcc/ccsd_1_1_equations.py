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
    E_ferm = 0.25*einsum('abij,ijab->', T2, cc.I.oovv)
    # t1 contribution
    E_ferm += einsum('ai,ia->', T1, cc.fock_mo.ov)
    # t1**2 contribution
    E_ferm += 0.5*einsum('ai,bj,ijab->', T1, T1, cc.I.oovv)

    # Bosonic and coupling part
    E_bos = einsum('I,I->',cc.G,S1)
    E_bos += einsum('Iia,Iai->',cc.g_mo_blocks.ov, U11)
    E_bos += einsum('Iia,ai,I->',cc.g_mo_blocks.ov, T1, S1)
    print('Fermionic correlation energy contribution: {}'.format(E_ferm))
    print('Bosonic correlation energy contribution: {}'.format(E_bos))

    E = E_ferm + E_bos

    return E

def two_rdm_ferm(cc, write=True):
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
    LS1 = cc.LS1.copy()
    LU11 = cc.LU11.copy()

    T1 = cc.T1.copy()
    T2 = cc.T2.copy()
    S1 = cc.S1.copy()
    U11 = cc.U11.copy()

    delta = np.eye(cc.no)

    # oooo block
    dm2_oooo = 1.0*einsum('il,jk->ijkl', delta, delta)
    dm2_oooo += -1.0*einsum('jl,ik->ijkl', delta, delta)
    dm2_oooo += -0.5*einsum('klab,baji->ijkl', L2, T2)
    dm2_oooo += 1.0*einsum('ka,ai,jl->ijkl', L1, T1, delta)
    dm2_oooo += -1.0*einsum('ka,aj,il->ijkl', L1, T1, delta)
    dm2_oooo += -1.0*einsum('la,ai,jk->ijkl', L1, T1, delta)
    dm2_oooo += 1.0*einsum('la,aj,ik->ijkl', L1, T1, delta)
    dm2_oooo += 1.0*einsum('Ika,Iai,jl->ijkl', LU11, U11, delta)
    dm2_oooo += -1.0*einsum('Ika,Iaj,il->ijkl', LU11, U11, delta)
    dm2_oooo += -1.0*einsum('Ila,Iai,jk->ijkl', LU11, U11, delta)
    dm2_oooo += 1.0*einsum('Ila,Iaj,ik->ijkl', LU11, U11, delta)
    dm2_oooo += 1.0*einsum('klab,bi,aj->ijkl', L2, T1, T1)
    dm2_oooo += -0.5*einsum('kmab,baim,jl->ijkl', L2, T2, delta)
    dm2_oooo += 0.5*einsum('kmab,bajm,il->ijkl', L2, T2, delta)
    dm2_oooo += 0.5*einsum('lmab,baim,jk->ijkl', L2, T2, delta)
    dm2_oooo += -0.5*einsum('lmab,bajm,ik->ijkl', L2, T2, delta)
    # Transposed for agreement with cqcpy convention
    dm2_oooo = dm2_oooo.transpose(1,0,2,3)

    # vvvv block
    dm2_vvvv = -0.5*einsum('ijba,cdji->abcd', L2, T2)
    dm2_vvvv += 1.0*einsum('ijba,ci,dj->abcd', L2, T1, T1)
    dm2_vvvv = dm2_vvvv.transpose(1,0,2,3)

    # vovv block
    dm2_vovv = 1.0*einsum('ja,bcij->aibc', L1, T2)
    dm2_vovv += -1.0*einsum('ja,bj,ci->aibc', L1, T1, T1)
    dm2_vovv += 1.0*einsum('ja,cj,bi->aibc', L1, T1, T1)
    dm2_vovv += -1.0*einsum('Ija,bj,Ici->aibc', LU11, T1, U11)
    dm2_vovv += 1.0*einsum('Ija,bi,Icj->aibc', LU11, T1, U11)
    dm2_vovv += 1.0*einsum('Ija,cj,Ibi->aibc', LU11, T1, U11)
    dm2_vovv += -1.0*einsum('Ija,ci,Ibj->aibc', LU11, T1, U11)
    dm2_vovv += 0.5*einsum('jkda,di,bckj->aibc', L2, T1, T2)
    dm2_vovv += -1.0*einsum('jkda,bj,dcik->aibc', L2, T1, T2)
    dm2_vovv += -0.5*einsum('jkda,bi,dckj->aibc', L2, T1, T2)
    dm2_vovv += 1.0*einsum('jkda,cj,dbik->aibc', L2, T1, T2)
    dm2_vovv += 0.5*einsum('jkda,ci,dbkj->aibc', L2, T1, T2)
    dm2_vovv += -1.0*einsum('jkda,bj,ck,di->aibc', L2, T1, T1, T1)
    dm2_vovv = dm2_vovv.transpose(0,1,3,2)

    # vvvo block
    dm2_vvvo = -1.0*einsum('ijba,cj->abci', L2, T1)
    dm2_vvvo = dm2_vvvo.transpose(1,0,2,3)

    # ovoo block
    dm2_ovoo = 1.0*einsum('ja,ik->iajk', L1, delta)
    dm2_ovoo += -1.0*einsum('ka,ij->iajk', L1, delta)
    dm2_ovoo += 1.0*einsum('jkba,bi->iajk', L2, T1)
    dm2_ovoo = dm2_ovoo.transpose(0,1,3,2)

    # oovo block
    dm2_oovo = -1.0*einsum('ai,jk->ijak', T1, delta)
    dm2_oovo += 1.0*einsum('aj,ik->ijak', T1, delta)
    dm2_oovo += 1.0*einsum('kb,baji->ijak', L1, T2)
    dm2_oovo += -1.0*einsum('I,Iai,jk->ijak', LS1, U11, delta)
    dm2_oovo += 1.0*einsum('I,Iaj,ik->ijak', LS1, U11, delta)
    dm2_oovo += -1.0*einsum('kb,bi,aj->ijak', L1, T1, T1)
    dm2_oovo += 1.0*einsum('kb,bj,ai->ijak', L1, T1, T1)
    dm2_oovo += 1.0*einsum('lb,bail,jk->ijak', L1, T2, delta)
    dm2_oovo += -1.0*einsum('lb,bajl,ik->ijak', L1, T2, delta)
    dm2_oovo += -1.0*einsum('Ikb,bi,Iaj->ijak', LU11, T1, U11)
    dm2_oovo += 1.0*einsum('Ikb,bj,Iai->ijak', LU11, T1, U11)
    dm2_oovo += 1.0*einsum('Ikb,ai,Ibj->ijak', LU11, T1, U11)
    dm2_oovo += -1.0*einsum('Ikb,aj,Ibi->ijak', LU11, T1, U11)
    dm2_oovo += -1.0*einsum('klbc,ci,bajl->ijak', L2, T1, T2)
    dm2_oovo += 1.0*einsum('klbc,cj,bail->ijak', L2, T1, T2)
    dm2_oovo += 0.5*einsum('klbc,al,cbji->ijak', L2, T1, T2)
    dm2_oovo += -0.5*einsum('klbc,ai,cbjl->ijak', L2, T1, T2)
    dm2_oovo += 0.5*einsum('klbc,aj,cbil->ijak', L2, T1, T2)
    dm2_oovo += 1.0*einsum('lb,al,bi,jk->ijak', L1, T1, T1, delta)
    dm2_oovo += -1.0*einsum('lb,al,bj,ik->ijak', L1, T1, T1, delta)
    dm2_oovo += 1.0*einsum('Ilb,bi,Ial,jk->ijak', LU11, T1, U11, delta)
    dm2_oovo += -1.0*einsum('Ilb,bj,Ial,ik->ijak', LU11, T1, U11, delta)
    dm2_oovo += 1.0*einsum('Ilb,al,Ibi,jk->ijak', LU11, T1, U11, delta)
    dm2_oovo += -1.0*einsum('Ilb,al,Ibj,ik->ijak', LU11, T1, U11, delta)
    dm2_oovo += -1.0*einsum('klbc,al,ci,bj->ijak', L2, T1, T1, T1)
    dm2_oovo += -0.5*einsum('lmbc,ci,baml,jk->ijak', L2, T1, T2, delta)
    dm2_oovo += 0.5*einsum('lmbc,cj,baml,ik->ijak', L2, T1, T2, delta)
    dm2_oovo += -0.5*einsum('lmbc,al,cbim,jk->ijak', L2, T1, T2, delta)
    dm2_oovo += 0.5*einsum('lmbc,al,cbjm,ik->ijak', L2, T1, T2, delta)
    dm2_oovo = dm2_oovo.transpose(1,0,2,3)
    
    # oovv block
    dm2_oovv = 1.0*einsum('abji->ijab', T2)
    dm2_oovv += -1.0*einsum('ai,bj->ijab', T1, T1)
    dm2_oovv += 1.0*einsum('bi,aj->ijab', T1, T1)
    dm2_oovv += -1.0*einsum('I,ai,Ibj->ijab', LS1, T1, U11)
    dm2_oovv += 1.0*einsum('I,aj,Ibi->ijab', LS1, T1, U11)
    dm2_oovv += 1.0*einsum('I,bi,Iaj->ijab', LS1, T1, U11)
    dm2_oovv += -1.0*einsum('I,bj,Iai->ijab', LS1, T1, U11)
    dm2_oovv += -1.0*einsum('kc,ci,abjk->ijab', L1, T1, T2)
    dm2_oovv += 1.0*einsum('kc,cj,abik->ijab', L1, T1, T2)
    dm2_oovv += -1.0*einsum('kc,ak,cbji->ijab', L1, T1, T2)
    dm2_oovv += 1.0*einsum('kc,ai,cbjk->ijab', L1, T1, T2)
    dm2_oovv += -1.0*einsum('kc,aj,cbik->ijab', L1, T1, T2)
    dm2_oovv += 1.0*einsum('kc,bk,caji->ijab', L1, T1, T2)
    dm2_oovv += -1.0*einsum('kc,bi,cajk->ijab', L1, T1, T2)
    dm2_oovv += 1.0*einsum('kc,bj,caik->ijab', L1, T1, T2)
    dm2_oovv += 1.0*einsum('Ikc,caik,Ibj->ijab', LU11, T2, U11)
    dm2_oovv += -1.0*einsum('Ikc,cajk,Ibi->ijab', LU11, T2, U11)
    dm2_oovv += 1.0*einsum('Ikc,caji,Ibk->ijab', LU11, T2, U11)
    dm2_oovv += -1.0*einsum('Ikc,cbik,Iaj->ijab', LU11, T2, U11)
    dm2_oovv += 1.0*einsum('Ikc,cbjk,Iai->ijab', LU11, T2, U11)
    dm2_oovv += -1.0*einsum('Ikc,cbji,Iak->ijab', LU11, T2, U11)
    dm2_oovv += 1.0*einsum('Ikc,abik,Icj->ijab', LU11, T2, U11)
    dm2_oovv += -1.0*einsum('Ikc,abjk,Ici->ijab', LU11, T2, U11)
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
    dm2_oovv += -1.0*einsum('Ikc,ci,aj,Ibk->ijab', LU11, T1, T1, U11)
    dm2_oovv += 1.0*einsum('Ikc,ci,bj,Iak->ijab', LU11, T1, T1, U11)
    dm2_oovv += 1.0*einsum('Ikc,cj,ai,Ibk->ijab', LU11, T1, T1, U11)
    dm2_oovv += -1.0*einsum('Ikc,cj,bi,Iak->ijab', LU11, T1, T1, U11)
    dm2_oovv += 1.0*einsum('Ikc,ak,ci,Ibj->ijab', LU11, T1, T1, U11)
    dm2_oovv += -1.0*einsum('Ikc,ak,cj,Ibi->ijab', LU11, T1, T1, U11)
    dm2_oovv += -1.0*einsum('Ikc,ak,bi,Icj->ijab', LU11, T1, T1, U11)
    dm2_oovv += 1.0*einsum('Ikc,ak,bj,Ici->ijab', LU11, T1, T1, U11)
    dm2_oovv += -1.0*einsum('Ikc,bk,ci,Iaj->ijab', LU11, T1, T1, U11)
    dm2_oovv += 1.0*einsum('Ikc,bk,cj,Iai->ijab', LU11, T1, T1, U11)
    dm2_oovv += 1.0*einsum('Ikc,bk,ai,Icj->ijab', LU11, T1, T1, U11)
    dm2_oovv += -1.0*einsum('Ikc,bk,aj,Ici->ijab', LU11, T1, T1, U11)
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

    # vvoo block
    dm2_vvoo = 1.0*einsum('ijba->abij', L2)
    dm2_vvoo = dm2_vvoo.transpose(1,0,2,3)

    # vovo block
    dm2_vovo = 1.0*einsum('ja,bi->aibj', L1, T1)
    dm2_vovo += 1.0*einsum('Ija,Ibi->aibj', LU11, U11)
    dm2_vovo += 1.0*einsum('jkca,cbik->aibj', L2, T2)
    dm2_vovo += -1.0*einsum('ka,bk,ij->aibj', L1, T1, delta)
    dm2_vovo += -1.0*einsum('Ika,Ibk,ij->aibj', LU11, U11, delta)
    dm2_vovo += 1.0*einsum('jkca,bk,ci->aibj', L2, T1, T1)
    dm2_vovo += 0.5*einsum('klca,cblk,ij->aibj', L2, T2, delta)
    dm2_vovo *= -1. # Flip sign
    
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

    # Transpose for consistency with pyscf
    dm2 = dm2.transpose(2,0,3,1)
    return dm2

def eb_coup_rdm(cc, write=True):
    ''' Calculate the e-b coupling RDMs: <b^+_I p^+ q> and <b_I p^+ q>.
        Returns as a tuple (Ipq, Ipq), where the first matrix corresponds
        to the bosonic creation, and the second to the annihilation
    '''

    if write:
        print('Computing electron-boson coupling RDMs...')

    delta = np.eye(cc.no)
    
    L1 = cc.L1.copy()
    L2 = cc.L2.copy()
    LS1 = cc.LS1.copy()
    LU11 = cc.LU11.copy()

    T1 = cc.T1.copy()
    T2 = cc.T2.copy()
    S1 = cc.S1.copy()
    U11 = cc.U11.copy()

    # <b+ o^+ o> block
    dm1_b_oo = 1.0*einsum('I,ij->Iij', LS1, delta)
    dm1_b_oo += -1.0*einsum('Ija,ai->Iij', LU11, T1)
    # <b+ v^+ v> block
    dm1_b_vv = 1.0*einsum('Iia,bi->Iab', LU11, T1)
    # <b+ o^+ v> block
    dm1_b_ov = 1.0*einsum('I,ai->Iia', LS1, T1)
    dm1_b_ov += -1.0*einsum('Ijb,baij->Iia', LU11, T2)
    dm1_b_ov += -1.0*einsum('Ijb,aj,bi->Iia', LU11, T1, T1)
    # <b+ v^+ o> block
    dm1_b_vo = 1.0*einsum('Iia->Iai', LU11)
    # <b o^+ o> block
    dm1_boo = 1.0*einsum('I,ij->Iij', S1, delta)
    dm1_boo += -1.0*einsum('ja,Iai->Iij', L1, U11)
    dm1_boo += -1.0*einsum('ja,ai,I->Iij', L1, T1, S1)
    dm1_boo += 1.0*einsum('ka,Iak,ij->Iij', L1, U11, delta)
    dm1_boo += -1.0*einsum('Jja,I,Jai->Iij', LU11, S1, U11)
    dm1_boo += 1.0*einsum('jkab,bi,Iak->Iij', L2, T1, U11)
    dm1_boo += 0.5*einsum('jkab,baik,I->Iij', L2, T2, S1)
    # <b v^+ v> block
    dm1_bvv = 1.0*einsum('ia,Ibi->Iab', L1, U11)
    dm1_bvv += 1.0*einsum('ia,bi,I->Iab', L1, T1, S1)
    dm1_bvv += 1.0*einsum('Jia,I,Jbi->Iab', LU11, S1, U11)
    dm1_bvv += -1.0*einsum('ijca,bi,Icj->Iab', L2, T1, U11)
    dm1_bvv += -0.5*einsum('ijca,cbji,I->Iab', L2, T2, S1)
    # <b o^+ v> block
    dm1_bov = 1.0*einsum('Iai->Iia', U11)
    dm1_bov += 1.0*einsum('ai,I->Iia', T1, S1)
    dm1_bov += 1.0*einsum('J,I,Jai->Iia', LS1, S1, U11)
    dm1_bov += -1.0*einsum('jb,bi,Iaj->Iia', L1, T1, U11)
    dm1_bov += -1.0*einsum('jb,aj,Ibi->Iia', L1, T1, U11)
    dm1_bov += 1.0*einsum('jb,ai,Ibj->Iia', L1, T1, U11)
    dm1_bov += -1.0*einsum('jb,baij,I->Iia', L1, T2, S1)
    dm1_bov += -1.0*einsum('Jjb,Jbi,Iaj->Iia', LU11, U11, U11)
    dm1_bov += -1.0*einsum('Jjb,Jaj,Ibi->Iia', LU11, U11, U11)
    dm1_bov += 1.0*einsum('Jjb,Jai,Ibj->Iia', LU11, U11, U11)
    dm1_bov += -0.5*einsum('jkbc,cbij,Iak->Iia', L2, T2, U11)
    dm1_bov += -0.5*einsum('jkbc,cakj,Ibi->Iia', L2, T2, U11)
    dm1_bov += 1.0*einsum('jkbc,caij,Ibk->Iia', L2, T2, U11)
    dm1_bov += -1.0*einsum('jb,aj,bi,I->Iia', L1, T1, T1, S1)
    dm1_bov += -1.0*einsum('Jjb,bi,I,Jaj->Iia', LU11, T1, S1, U11)
    dm1_bov += -1.0*einsum('Jjb,aj,I,Jbi->Iia', LU11, T1, S1, U11)
    dm1_bov += 0.5*einsum('jkbc,ci,bakj,I->Iia', L2, T1, T2, S1)
    dm1_bov += 1.0*einsum('jkbc,aj,ci,Ibk->Iia', L2, T1, T1, U11)
    dm1_bov += 0.5*einsum('jkbc,aj,cbik,I->Iia', L2, T1, T2, S1)
    # <b v^+ o> block
    dm1_bvo = 1.0*einsum('ia,I->Iai', L1, S1)
    dm1_bvo += -1.0*einsum('ijba,Ibj->Iai', L2, U11)

    # Hermitize everything
    dm_coup_boscre = np.zeros((cc.nbos, cc.nso, cc.nso))
    dm_coup_bosann = np.zeros((cc.nbos, cc.nso, cc.nso))

    dm_coup_boscre[:, :cc.no, :cc.no] = (dm1_b_oo + dm1_b_oo.transpose(0,2,1)) / 2.
    dm_coup_boscre[:, cc.no:, cc.no:] = (dm1_b_vv + dm1_b_vv.transpose(0,2,1)) / 2.
    dm_coup_boscre[:, :cc.no, cc.no:] = (dm1_b_ov + dm1_b_vo.transpose(0,2,1)) / 2.
    dm_coup_boscre[:, cc.no:, :cc.no] = dm_coup_boscre[:, :cc.no, cc.no:].transpose(0,2,1)
    
    dm_coup_bosann[:, :cc.no, :cc.no] = (dm1_boo + dm1_boo.transpose(0,2,1)) / 2.
    dm_coup_bosann[:, cc.no:, cc.no:] = (dm1_bvv + dm1_bvv.transpose(0,2,1)) / 2.
    dm_coup_bosann[:, :cc.no, cc.no:] = (dm1_bov + dm1_bvo.transpose(0,2,1)) / 2.
    dm_coup_bosann[:, cc.no:, :cc.no] = dm_coup_bosann[:, :cc.no, cc.no:].transpose(0,2,1)

    # Do we need to add a mean-field part to <b^+ a^+ a>?
    #for i in range(cc.no):
    #    dm_coup_boscre[:,i,i] += 1.

    return (dm_coup_boscre, dm_coup_bosann)

def dm_singbos(cc, write=True):
    ''' Calculate single boson RDMs as a tuple, (<b^+>, <b>) '''

    if write:
        print('Computing single boson RDMs...')
    
    if cc.L1 is None:
        if write:
            print('No optimized lambda amplitudes found to compute density matrices.')
            print('Using L = T^+ approximation...')
        cc.init_lam()
    
    L1 = cc.L1.copy()
    LS1 = cc.LS1.copy()

    S1 = cc.S1.copy()
    U11 = cc.U11.copy()

    dm1_b_cre = LS1.copy()
    dm1_b_ann = S1.copy()
    dm1_b_ann += 1.0*einsum('ia,Iai->I', L1, U11)

    return (dm1_b_cre, dm1_b_ann)

def one_rdm_bos(cc, write=True):
    ''' Calculate bosonic 1RDM: <b^+_i b_j> '''

    if write:
        print('Computing bosonic space 1RDM...')
    
    if cc.L1 is None:
        if write:
            print('No optimized lambda amplitudes found to compute density matrices.')
            print('Using L = T^+ approximation...')
        cc.init_lam()

    LS1 = cc.LS1.copy()
    LU11 = cc.LU11.copy()

    S1 = cc.S1.copy()
    U11 = cc.U11.copy()

    dm1_b = 1.0*einsum('I,J->IJ', LS1, S1)
    dm1_b += 1.0*einsum('Iia,Jai->IJ', LU11, U11)

    # Hermitize
    dm1_b = (dm1_b + dm1_b.T) / 2.

    if write:
        print('Trace of bosonic 1RDM: {}'.format(np.trace(dm1_b)))

    return dm1_b

def one_rdm_ferm(cc, write=True):
    ''' Calculate 1RDM '''

    if write:
        print('Computing fermionic space 1RDM...')

    if cc.L1 is None:
        if write:
            print('No optimized lambda amplitudes found to compute density matrices.')
            print('Using L = T^+ approximation...')
        cc.init_lam()

    L1 = cc.L1.copy()
    L2 = cc.L2.copy()
    LS1 = cc.LS1.copy()
    LU11 = cc.LU11.copy()

    T1 = cc.T1.copy()
    T2 = cc.T2.copy()
    S1 = cc.S1.copy()
    U11 = cc.U11.copy()

    dm1_oo = -1.0*einsum('ia,aj->ij', L1, T1)
    dm1_oo += -1.0*einsum('Iia,Iaj->ij', LU11, U11)
    dm1_oo += 0.5*einsum('ikab,bajk->ij', L2, T2)

    dm1_vv = 1.0*einsum('ib,ai->ab', L1, T1)
    dm1_vv += 1.0*einsum('Iib,Iai->ab', LU11, U11)
    dm1_vv += -0.5*einsum('ijcb,caji->ab', L2, T2)

    dm1_ov = L1.copy()

    dm1_vo = T1.copy()
    dm1_vo += 1.0*einsum('I,Iai->ai', LS1, U11)
    dm1_vo += -1.0*einsum('jb,baij->ai', L1, T2)
    dm1_vo += -1.0*einsum('jb,aj,bi->ai', L1, T1, T1)
    dm1_vo += -1.0*einsum('Ijb,bi,Iaj->ai', LU11, T1, U11)
    dm1_vo += -1.0*einsum('Ijb,aj,Ibi->ai', LU11, T1, U11)
    dm1_vo += 0.5*einsum('jkbc,ci,bakj->ai', L2, T1, T2)
    dm1_vo += 0.5*einsum('jkbc,aj,cbik->ai', L2, T1, T2)
    
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

def lam_updates_ccsd_1_1(cc):
    ''' Solve residual equations for CCSD_1_11.
        Equations generating script using the 'wick' program found in gen_eqns/ccsd_11_lam.py
    '''
    
    # Copy fock matrix in MO basis, and remove diagonals
    F = copy.copy(cc.fock_mo)
    F_full = copy.copy(cc.fock_mo)
    F.oo = F.oo - np.diag(cc.eo)
    F.vv = F.vv - np.diag(cc.ev)
    # eb coupling
    g = copy.copy(cc.g_mo_blocks)
    h = copy.copy(cc.g_mo_blocks)
    # b terms
    G = cc.G.copy()
    H = cc.G.copy()
    
    # Note that we also have to remove diagonals from omega
    # matrix. Since the rest of the code is constrained to have
    # no coupling between bosons, this means that this is actually
    # the zero matrix, but we keep it in here anyway for potential
    # future applications
    w = np.diag(cc.omega).copy()
    w_full = w.copy()
    w = w - np.diag(np.diag(w))

    I = cc.I
    
    T1old = cc.T1old
    T2old = cc.T2old
    S1old = cc.S1old
    U11old = cc.U11old

    L1old = cc.L1old
    L2old = cc.L2old
    LS1old = cc.LS1old
    LU11old = cc.LU11old
    
    # Lambda residuals for 1-fermion deexcitation space
    # Connected piece not proportional to Lambda...
    L1 = F.ov.copy()
    L1 += 1.0*einsum('I,Iia->ia', S1old, g.ov)
    L1 += -1.0*einsum('bj,jiab->ia', T1old, I.oovv)

    # Connected piece proportional to Lambda...
    L1 += 1.0*einsum('Iia,I->ia', g.ov, LS1old)
    L1 += -1.0*einsum('ij,ja->ia', F.oo, L1old)
    L1 += 1.0*einsum('ba,ib->ia', F.vv, L1old)
    L1 += 1.0*einsum('I,Iia->ia', G, LU11old)
    L1 += -1.0*einsum('ibja,jb->ia', I.ovov, L1old)
    L1 += -1.0*einsum('Iij,Ija->ia', g.oo, LU11old)
    L1 += 1.0*einsum('Iba,Iib->ia', g.vv, LU11old)
    L1 += -1.0*einsum('bj,jiab->ia', F.vo, L2old)
    L1 += 0.5*einsum('ibjk,kjab->ia', I.ovoo, L2old)
    L1 += -0.5*einsum('bcja,jicb->ia', I.vvov, L2old)
    L1 += -1.0*einsum('ib,bj,ja->ia', F.ov, T1old, L1old)
    L1 += -1.0*einsum('Iij,I,ja->ia', g.oo, S1old, L1old)
    L1 += -1.0*einsum('ja,bj,ib->ia', F.ov, T1old, L1old)
    L1 += 1.0*einsum('Iba,I,ib->ia', g.vv, S1old, L1old)
    # w_full used here
    L1 += 1.0*einsum('IJ,J,Iia->ia', w_full, S1old, LU11old)
    L1 += -1.0*einsum('jiab,Ibj,I->ia', I.oovv, U11old, LS1old)
    L1 += 1.0*einsum('Iia,Ibj,jb->ia', g.ov, U11old, L1old)
    L1 += -1.0*einsum('jika,bj,kb->ia', I.ooov, T1old, L1old)
    L1 += 1.0*einsum('ibac,cj,jb->ia', I.ovvv, T1old, L1old)
    L1 += -1.0*einsum('Iib,Ibj,ja->ia', g.ov, U11old, L1old)
    L1 += 1.0*einsum('jikb,bj,ka->ia', I.ooov, T1old, L1old)
    L1 += -1.0*einsum('Ija,Ibj,ib->ia', g.ov, U11old, L1old)
    L1 += -1.0*einsum('jbac,cj,ib->ia', I.ovvv, T1old, L1old)
    L1 += -1.0*einsum('ib,Ibj,Ija->ia', F.ov, U11old, LU11old)
    L1 += -1.0*einsum('Iib,bj,Ija->ia', g.ov, T1old, LU11old)
    L1 += -1.0*einsum('ja,Ibj,Iib->ia', F.ov, U11old, LU11old)
    L1 += -1.0*einsum('Ija,bj,Iib->ia', g.ov, T1old, LU11old)
    L1 += 1.0*einsum('jb,Ibj,Iia->ia', F.ov, U11old, LU11old)
    L1 += 1.0*einsum('Ijb,bj,Iia->ia', g.ov, T1old, LU11old)
    L1 += -1.0*einsum('I,Ibj,jiab->ia', G, U11old, L2old)
    L1 += -1.0*einsum('Ibj,I,jiab->ia', g.vo, S1old, L2old)
    # F_full used here
    L1 += 1.0*einsum('jk,bj,kiab->ia', F_full.oo, T1old, L2old)
    L1 += -1.0*einsum('bc,cj,jiab->ia', F_full.vv, T1old, L2old)
    L1 += 1.0*einsum('jiab,cbjk,kc->ia', I.oovv, T2old, L1old)
    L1 += 0.5*einsum('jibc,cbjk,ka->ia', I.oovv, T2old, L1old)
    L1 += 0.5*einsum('jkab,cbkj,ic->ia', I.oovv, T2old, L1old)
    L1 += -1.0*einsum('jika,Ibj,Ikb->ia', I.ooov, U11old, LU11old)
    L1 += 1.0*einsum('ibac,Icj,Ijb->ia', I.ovvv, U11old, LU11old)
    L1 += 1.0*einsum('jikb,Ibj,Ika->ia', I.ooov, U11old, LU11old)
    L1 += -1.0*einsum('jbac,Icj,Iib->ia', I.ovvv, U11old, LU11old)
    L1 += -0.5*einsum('ib,cbjk,kjac->ia', F.ov, T2old, L2old)
    L1 += -1.0*einsum('Iij,Ibk,jkab->ia', g.oo, U11old, L2old)
    L1 += -1.0*einsum('ibjc,ck,jkab->ia', I.ovov, T1old, L2old)
    L1 += 0.5*einsum('jikl,bj,lkab->ia', I.oooo, T1old, L2old)
    L1 += -0.5*einsum('ja,bcjk,kicb->ia', F.ov, T2old, L2old)
    L1 += 1.0*einsum('Iba,Icj,jicb->ia', g.vv, U11old, L2old)
    L1 += -1.0*einsum('jbka,cj,kicb->ia', I.ovov, T1old, L2old)
    L1 += 1.0*einsum('jb,cbjk,kiac->ia', F.ov, T2old, L2old)
    L1 += 1.0*einsum('Ijk,Ibj,kiab->ia', g.oo, U11old, L2old)
    L1 += -1.0*einsum('Ibc,Icj,jiab->ia', g.vv, U11old, L2old)
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
    L1 += -1.0*einsum('Iib,bj,I,ja->ia', g.ov, T1old, S1old, L1old)
    L1 += -1.0*einsum('Ija,bj,I,ib->ia', g.ov, T1old, S1old, L1old)
    L1 += 1.0*einsum('jiab,bk,cj,kc->ia', I.oovv, T1old, T1old, L1old)
    L1 += -1.0*einsum('jibc,ck,bj,ka->ia', I.oovv, T1old, T1old, L1old)
    L1 += -1.0*einsum('jkab,cj,bk,ic->ia', I.oovv, T1old, T1old, L1old)
    L1 += -1.0*einsum('Iib,I,Jbj,Jja->ia', g.ov, S1old, U11old, LU11old)
    L1 += -1.0*einsum('Ija,I,Jbj,Jib->ia', g.ov, S1old, U11old, LU11old)
    L1 += 1.0*einsum('Ijb,I,Jbj,Jia->ia', g.ov, S1old, U11old, LU11old)
    L1 += 1.0*einsum('jb,bk,cj,kiac->ia', F.ov, T1old, T1old, L2old)
    L1 += 1.0*einsum('Ijk,bj,I,kiab->ia', g.oo, T1old, S1old, L2old)
    L1 += -1.0*einsum('Ibc,cj,I,jiab->ia', g.vv, T1old, S1old, L2old)
    L1 += 1.0*einsum('jiab,cj,Ibk,Ikc->ia', I.oovv, T1old, U11old, LU11old)
    L1 += 1.0*einsum('jiab,bk,Icj,Ikc->ia', I.oovv, T1old, U11old, LU11old)
    L1 += -1.0*einsum('jibc,ck,Ibj,Ika->ia', I.oovv, T1old, U11old, LU11old)
    L1 += 1.0*einsum('jibc,cj,Ibk,Ika->ia', I.oovv, T1old, U11old, LU11old)
    L1 += -1.0*einsum('jkab,cj,Ibk,Iic->ia', I.oovv, T1old, U11old, LU11old)
    L1 += 1.0*einsum('jkab,bj,Ick,Iic->ia', I.oovv, T1old, U11old, LU11old)
    L1 += -1.0*einsum('jkbc,cj,Ibk,Iia->ia', I.oovv, T1old, U11old, LU11old)
    L1 += -1.0*einsum('Iib,bj,Ick,jkac->ia', g.ov, T1old, U11old, L2old)
    L1 += -0.5*einsum('Iib,cbjk,I,kjac->ia', g.ov, T2old, S1old, L2old)
    L1 += -1.0*einsum('jikb,bl,cj,klac->ia', I.ooov, T1old, T1old, L2old)
    L1 += 0.5*einsum('ibcd,dj,ck,jkab->ia', I.ovvv, T1old, T1old, L2old)
    L1 += -1.0*einsum('Ija,bj,Ick,kicb->ia', g.ov, T1old, U11old, L2old)
    L1 += 1.0*einsum('Ijb,cj,Ibk,kiac->ia', g.ov, T1old, U11old, L2old)
    L1 += 1.0*einsum('Ijb,bk,Icj,kiac->ia', g.ov, T1old, U11old, L2old)
    L1 += -1.0*einsum('Ijb,bj,Ick,kiac->ia', g.ov, T1old, U11old, L2old)
    L1 += -0.5*einsum('Ija,bcjk,I,kicb->ia', g.ov, T2old, S1old, L2old)
    L1 += 1.0*einsum('Ijb,cbjk,I,kiac->ia', g.ov, T2old, S1old, L2old)
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
    L1 += 1.0*einsum('Ijb,bk,cj,I,kiac->ia', g.ov, T1old, T1old, S1old, L2old)
    L1 += 0.5*einsum('jibc,ck,bl,dj,klad->ia', I.oovv, T1old, T1old, T1old, L2old)
    L1 += 0.5*einsum('jkab,bl,cj,dk,lidc->ia', I.oovv, T1old, T1old, T1old, L2old)
    L1 += -1.0*einsum('jkbc,cl,dj,bk,liad->ia', I.oovv, T1old, T1old, T1old, L2old)

    # Connected piece not proportional to Lambda...
    L2 = I.oovv.transpose(1,0,3,2).copy()

    # Connected piece proportional to Lambda...
    L2 += 1.0*einsum('jikb,ka->ijab', I.ooov, L1old)
    L2 += -1.0*einsum('jika,kb->ijab', I.ooov, L1old)
    L2 += 1.0*einsum('jcba,ic->ijab', I.ovvv, L1old)
    L2 += -1.0*einsum('icba,jc->ijab', I.ovvv, L1old)
    L2 += 1.0*einsum('Ijb,Iia->ijab', g.ov, LU11old)
    L2 += -1.0*einsum('Ija,Iib->ijab', g.ov, LU11old)
    L2 += -1.0*einsum('Iib,Ija->ijab', g.ov, LU11old)
    L2 += 1.0*einsum('Iia,Ijb->ijab', g.ov, LU11old)
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
    L2 += -1.0*einsum('Ijk,I,kiba->ijab', g.oo, S1old, L2old)
    L2 += 1.0*einsum('Iik,I,kjba->ijab', g.oo, S1old, L2old)
    L2 += 1.0*einsum('kb,ck,jiac->ijab', F.ov, T1old, L2old)
    L2 += -1.0*einsum('ka,ck,jibc->ijab', F.ov, T1old, L2old)
    L2 += -1.0*einsum('Icb,I,jiac->ijab', g.vv, S1old, L2old)
    L2 += 1.0*einsum('Ica,I,jibc->ijab', g.vv, S1old, L2old)
    L2 += -1.0*einsum('jibc,Ick,Ika->ijab', I.oovv, U11old, LU11old)
    L2 += 1.0*einsum('jiac,Ick,Ikb->ijab', I.oovv, U11old, LU11old)
    L2 += 1.0*einsum('kjba,Ick,Iic->ijab', I.oovv, U11old, LU11old)
    L2 += -1.0*einsum('kjbc,Ick,Iia->ijab', I.oovv, U11old, LU11old)
    L2 += 1.0*einsum('kjac,Ick,Iib->ijab', I.oovv, U11old, LU11old)
    L2 += -1.0*einsum('kiba,Ick,Ijc->ijab', I.oovv, U11old, LU11old)
    L2 += 1.0*einsum('kibc,Ick,Ija->ijab', I.oovv, U11old, LU11old)
    L2 += -1.0*einsum('kiac,Ick,Ijb->ijab', I.oovv, U11old, LU11old)
    L2 += 1.0*einsum('jikc,cl,klba->ijab', I.ooov, T1old, L2old)
    L2 += -1.0*einsum('Ijb,Ick,kiac->ijab', g.ov, U11old, L2old)
    L2 += 1.0*einsum('Ija,Ick,kibc->ijab', g.ov, U11old, L2old)
    L2 += -1.0*einsum('Ijc,Ick,kiba->ijab', g.ov, U11old, L2old)
    L2 += 1.0*einsum('Iib,Ick,kjac->ijab', g.ov, U11old, L2old)
    L2 += -1.0*einsum('Iia,Ick,kjbc->ijab', g.ov, U11old, L2old)
    L2 += 1.0*einsum('Iic,Ick,kjba->ijab', g.ov, U11old, L2old)
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
    L2 += 1.0*einsum('Ikb,Ick,jiac->ijab', g.ov, U11old, L2old)
    L2 += -1.0*einsum('Ika,Ick,jibc->ijab', g.ov, U11old, L2old)
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
    L2 += -1.0*einsum('Ijc,ck,I,kiba->ijab', g.ov, T1old, S1old, L2old)
    L2 += 1.0*einsum('Iic,ck,I,kjba->ijab', g.ov, T1old, S1old, L2old)
    L2 += 1.0*einsum('Ikb,ck,I,jiac->ijab', g.ov, T1old, S1old, L2old)
    L2 += -1.0*einsum('Ika,ck,I,jibc->ijab', g.ov, T1old, S1old, L2old)
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

    # Disconnected piece projecting onto single fermionic excitations...
    L2 += 1.0*einsum('jb,ia->ijab', L1old, F.ov)
    L2 += -1.0*einsum('ja,ib->ijab', L1old, F.ov)
    L2 += -1.0*einsum('ib,ja->ijab', L1old, F.ov)
    L2 += 1.0*einsum('ia,jb->ijab', L1old, F.ov)
    L2 += 1.0*einsum('I,jb,Iia->ijab', S1old, L1old, g.ov)
    L2 += -1.0*einsum('I,ja,Iib->ijab', S1old, L1old, g.ov)
    L2 += -1.0*einsum('I,ib,Ija->ijab', S1old, L1old, g.ov)
    L2 += 1.0*einsum('I,ia,Ijb->ijab', S1old, L1old, g.ov)
    L2 += -1.0*einsum('ck,jb,kiac->ijab', T1old, L1old, I.oovv)
    L2 += 1.0*einsum('ck,ja,kibc->ijab', T1old, L1old, I.oovv)
    L2 += 1.0*einsum('ck,ib,kjac->ijab', T1old, L1old, I.oovv)
    L2 += -1.0*einsum('ck,ia,kjbc->ijab', T1old, L1old, I.oovv)

    # Connected piece not proportional to Lambda...
    LS1 = G.copy()
    LS1 += 1.0*einsum('ai,Iia->I', T1old, g.ov)

    # Connected piece proportional to Lambda...
    LS1 += 1.0*einsum('JI,J->I', w, LS1old)
    LS1 += 1.0*einsum('Iai,ia->I', g.vo, L1old)
    LS1 += 1.0*einsum('ai,Iia->I', F.vo, LU11old)
    LS1 += 1.0*einsum('Iia,Jai,J->I', g.ov, U11old, LS1old)
    LS1 += -1.0*einsum('Iij,ai,ja->I', g.oo, T1old, L1old)
    LS1 += 1.0*einsum('Iab,bi,ia->I', g.vv, T1old, L1old)
    LS1 += 1.0*einsum('J,Jai,Iia->I', G, U11old, LU11old)
    LS1 += 1.0*einsum('Jai,J,Iia->I', g.vo, S1old, LU11old)
    # NOTE: Does this want to be full F?
    LS1 += -1.0*einsum('ij,ai,Ija->I', F_full.oo, T1old, LU11old)
    LS1 += 1.0*einsum('ab,bi,Iia->I', F_full.vv, T1old, LU11old)
    LS1 += -1.0*einsum('Iia,baij,jb->I', g.ov, T2old, L1old)
    LS1 += -1.0*einsum('Iij,Jai,Jja->I', g.oo, U11old, LU11old)
    LS1 += 1.0*einsum('Iab,Jbi,Jia->I', g.vv, U11old, LU11old)
    LS1 += -1.0*einsum('ia,baij,Ijb->I', F.ov, T2old, LU11old)
    LS1 += -1.0*einsum('Jij,Jai,Ija->I', g.oo, U11old, LU11old)
    LS1 += 1.0*einsum('Jab,Jbi,Iia->I', g.vv, U11old, LU11old)
    LS1 += -1.0*einsum('iajb,bi,Ija->I', I.ovov, T1old, LU11old)
    LS1 += 0.5*einsum('ijka,baji,Ikb->I', I.ooov, T2old, LU11old)
    LS1 += -0.5*einsum('iabc,cbij,Ija->I', I.ovvv, T2old, LU11old)
    LS1 += 0.5*einsum('Iij,abik,jkba->I', g.oo, T2old, L2old)
    LS1 += -0.5*einsum('Iab,cbij,jica->I', g.vv, T2old, L2old)
    LS1 += -1.0*einsum('Iia,aj,bi,jb->I', g.ov, T1old, T1old, L1old)
    LS1 += -1.0*einsum('ia,aj,bi,Ijb->I', F.ov, T1old, T1old, LU11old)
    LS1 += -1.0*einsum('Jij,ai,J,Ija->I', g.oo, T1old, S1old, LU11old)
    LS1 += 1.0*einsum('Jab,bi,J,Iia->I', g.vv, T1old, S1old, LU11old)
    LS1 += -1.0*einsum('Iia,bi,Jaj,Jjb->I', g.ov, T1old, U11old, LU11old)
    LS1 += -1.0*einsum('Iia,aj,Jbi,Jjb->I', g.ov, T1old, U11old, LU11old)
    LS1 += -1.0*einsum('Jia,bi,Jaj,Ijb->I', g.ov, T1old, U11old, LU11old)
    LS1 += -1.0*einsum('Jia,aj,Jbi,Ijb->I', g.ov, T1old, U11old, LU11old)
    LS1 += 1.0*einsum('Jia,ai,Jbj,Ijb->I', g.ov, T1old, U11old, LU11old)
    LS1 += -1.0*einsum('Jia,baij,J,Ijb->I', g.ov, T2old, S1old, LU11old)
    LS1 += -1.0*einsum('ijka,bi,aj,Ikb->I', I.ooov, T1old, T1old, LU11old)
    LS1 += 1.0*einsum('iabc,cj,bi,Ija->I', I.ovvv, T1old, T1old, LU11old)
    LS1 += -0.5*einsum('ijab,ci,bajk,Ikc->I', I.oovv, T1old, T2old, LU11old)
    LS1 += -0.5*einsum('ijab,bk,caji,Ikc->I', I.oovv, T1old, T2old, LU11old)
    LS1 += 1.0*einsum('ijab,bi,cajk,Ikc->I', I.oovv, T1old, T2old, LU11old)
    LS1 += 0.5*einsum('Iia,bi,cajk,kjcb->I', g.ov, T1old, T2old, L2old)
    LS1 += 0.5*einsum('Iia,aj,bcik,jkcb->I', g.ov, T1old, T2old, L2old)
    LS1 += -1.0*einsum('Jia,aj,bi,J,Ijb->I', g.ov, T1old, T1old, S1old, LU11old)
    LS1 += 1.0*einsum('ijab,bk,ci,aj,Ikc->I', I.oovv, T1old, T1old, T1old, LU11old)

    # Lambda residuals for 1 boson + 1 fermion deexcitation space...
    # Connected piece not proportional to Lambda...
    LU11 = g.ov.copy()

    # Connected piece proportional to Lambda...
    LU11 += -1.0*einsum('Iij,ja->Iia', g.oo, L1old)
    LU11 += 1.0*einsum('Iba,ib->Iia', g.vv, L1old)
    # Three terms below have had their diagonal removed
    LU11 += 1.0*einsum('JI,Jia->Iia', w, LU11old)
    LU11 += -1.0*einsum('ij,Ija->Iia', F.oo, LU11old)
    LU11 += 1.0*einsum('ba,Iib->Iia', F.vv, LU11old)
    LU11 += -1.0*einsum('ibja,Ijb->Iia', I.ovov, LU11old)
    LU11 += -1.0*einsum('Ibj,jiab->Iia', g.vo, L2old)
    LU11 += -1.0*einsum('Iib,bj,ja->Iia', g.ov, T1old, L1old)
    LU11 += -1.0*einsum('Ija,bj,ib->Iia', g.ov, T1old, L1old)
    LU11 += -1.0*einsum('ib,bj,Ija->Iia', F.ov, T1old, LU11old)
    LU11 += -1.0*einsum('ja,bj,Iib->Iia', F.ov, T1old, LU11old)
    LU11 += -1.0*einsum('Jij,J,Ija->Iia', g.oo, S1old, LU11old)
    LU11 += 1.0*einsum('Jba,J,Iib->Iia', g.vv, S1old, LU11old)
    LU11 += -1.0*einsum('Iib,Jbj,Jja->Iia', g.ov, U11old, LU11old)
    LU11 += -1.0*einsum('Ija,Jbj,Jib->Iia', g.ov, U11old, LU11old)
    LU11 += 1.0*einsum('Ijb,Jbj,Jia->Iia', g.ov, U11old, LU11old)
    LU11 += 1.0*einsum('Jia,Jbj,Ijb->Iia', g.ov, U11old, LU11old)
    LU11 += -1.0*einsum('Jib,Jbj,Ija->Iia', g.ov, U11old, LU11old)
    LU11 += -1.0*einsum('Jja,Jbj,Iib->Iia', g.ov, U11old, LU11old)
    LU11 += -1.0*einsum('jika,bj,Ikb->Iia', I.ooov, T1old, LU11old)
    LU11 += 1.0*einsum('ibac,cj,Ijb->Iia', I.ovvv, T1old, LU11old)
    LU11 += 1.0*einsum('jikb,bj,Ika->Iia', I.ooov, T1old, LU11old)
    LU11 += -1.0*einsum('jbac,cj,Iib->Iia', I.ovvv, T1old, LU11old)
    LU11 += 1.0*einsum('Ijk,bj,kiab->Iia', g.oo, T1old, L2old)
    LU11 += -1.0*einsum('Ibc,cj,jiab->Iia', g.vv, T1old, L2old)
    LU11 += 1.0*einsum('jiab,cbjk,Ikc->Iia', I.oovv, T2old, LU11old)
    LU11 += 0.5*einsum('jibc,cbjk,Ika->Iia', I.oovv, T2old, LU11old)
    LU11 += 0.5*einsum('jkab,cbkj,Iic->Iia', I.oovv, T2old, LU11old)
    LU11 += -0.5*einsum('Iib,cbjk,kjac->Iia', g.ov, T2old, L2old)
    LU11 += -0.5*einsum('Ija,bcjk,kicb->Iia', g.ov, T2old, L2old)
    LU11 += 1.0*einsum('Ijb,cbjk,kiac->Iia', g.ov, T2old, L2old)
    LU11 += -1.0*einsum('Jib,bj,J,Ija->Iia', g.ov, T1old, S1old, LU11old)
    LU11 += -1.0*einsum('Jja,bj,J,Iib->Iia', g.ov, T1old, S1old, LU11old)
    LU11 += 1.0*einsum('jiab,bk,cj,Ikc->Iia', I.oovv, T1old, T1old, LU11old)
    LU11 += -1.0*einsum('jibc,ck,bj,Ika->Iia', I.oovv, T1old, T1old, LU11old)
    LU11 += -1.0*einsum('jkab,cj,bk,Iic->Iia', I.oovv, T1old, T1old, LU11old)
    LU11 += 1.0*einsum('Ijb,bk,cj,kiac->Iia', g.ov, T1old, T1old, L2old)

    # Disconnected piece projecting onto single fermionic excitations...
    LU11 += 1.0*einsum('I,ia->Iia', LS1old, F.ov)
    LU11 += 1.0*einsum('J,I,Jia->Iia', S1old, LS1old, g.ov)
    LU11 += -1.0*einsum('bj,I,jiab->Iia', T1old, LS1old, I.oovv)

    # Disconnected piece projecting onto single bosonic excitations...
    LU11 += 1.0*einsum('I,ia->Iia', LS1old, F.ov)
    LU11 += 1.0*einsum('J,I,Jia->Iia', S1old, LS1old, g.ov)
    LU11 += -1.0*einsum('bj,I,jiab->Iia', T1old, LS1old, I.oovv)

    return L1, L2, LS1, LU11

def calc_lam_resid_1_1(F, I, G, g, w_diag, T1old, L1old, T2old, L2old, S1old, LS1old, U11old, LU11old):
    ''' Compute the norm of the residual for the lambda equations (debugging) '''

    w = np.diag(w_diag)

# lambda residuals for 1 fermion deexcitation space...
# connected piece not proportional to Lambda...
    L1 = 1.0*einsum('ia->ia', F.ov)
    L1 += 1.0*einsum('I,Iia->ia', S1old, g.ov)
    L1 += -1.0*einsum('bj,jiab->ia', T1old, I.oovv)

#Computing connected piece proportional to Lambda...
    L1 += 1.0*einsum('Iia,I->ia', g.ov, LS1old)
    L1 += -1.0*einsum('ij,ja->ia', F.oo, L1old)
    L1 += 1.0*einsum('ba,ib->ia', F.vv, L1old)
    L1 += 1.0*einsum('I,Iia->ia', G, LU11old)
    L1 += -1.0*einsum('ibja,jb->ia', I.ovov, L1old)
    L1 += -1.0*einsum('Iij,Ija->ia', g.oo, LU11old)
    L1 += 1.0*einsum('Iba,Iib->ia', g.vv, LU11old)
    L1 += -1.0*einsum('bj,jiab->ia', F.vo, L2old)
    L1 += 0.5*einsum('ibjk,kjab->ia', I.ovoo, L2old)
    L1 += -0.5*einsum('bcja,jicb->ia', I.vvov, L2old)
    L1 += -1.0*einsum('ib,bj,ja->ia', F.ov, T1old, L1old)
    L1 += -1.0*einsum('Iij,I,ja->ia', g.oo, S1old, L1old)
    L1 += -1.0*einsum('ja,bj,ib->ia', F.ov, T1old, L1old)
    L1 += 1.0*einsum('Iba,I,ib->ia', g.vv, S1old, L1old)
    L1 += 1.0*einsum('IJ,J,Iia->ia', w, S1old, LU11old)
    L1 += -1.0*einsum('jiab,Ibj,I->ia', I.oovv, U11old, LS1old)
    L1 += 1.0*einsum('Iia,Ibj,jb->ia', g.ov, U11old, L1old)
    L1 += -1.0*einsum('jika,bj,kb->ia', I.ooov, T1old, L1old)
    L1 += 1.0*einsum('ibac,cj,jb->ia', I.ovvv, T1old, L1old)
    L1 += -1.0*einsum('Iib,Ibj,ja->ia', g.ov, U11old, L1old)
    L1 += 1.0*einsum('jikb,bj,ka->ia', I.ooov, T1old, L1old)
    L1 += -1.0*einsum('Ija,Ibj,ib->ia', g.ov, U11old, L1old)
    L1 += -1.0*einsum('jbac,cj,ib->ia', I.ovvv, T1old, L1old)
    L1 += -1.0*einsum('ib,Ibj,Ija->ia', F.ov, U11old, LU11old)
    L1 += -1.0*einsum('Iib,bj,Ija->ia', g.ov, T1old, LU11old)
    L1 += -1.0*einsum('ja,Ibj,Iib->ia', F.ov, U11old, LU11old)
    L1 += -1.0*einsum('Ija,bj,Iib->ia', g.ov, T1old, LU11old)
    L1 += 1.0*einsum('jb,Ibj,Iia->ia', F.ov, U11old, LU11old)
    L1 += 1.0*einsum('Ijb,bj,Iia->ia', g.ov, T1old, LU11old)
    L1 += -1.0*einsum('I,Ibj,jiab->ia', G, U11old, L2old)
    L1 += -1.0*einsum('Ibj,I,jiab->ia', g.vo, S1old, L2old)
    L1 += 1.0*einsum('jk,bj,kiab->ia', F.oo, T1old, L2old)
    L1 += -1.0*einsum('bc,cj,jiab->ia', F.vv, T1old, L2old)
    L1 += 1.0*einsum('jiab,cbjk,kc->ia', I.oovv, T2old, L1old)
    L1 += 0.5*einsum('jibc,cbjk,ka->ia', I.oovv, T2old, L1old)
    L1 += 0.5*einsum('jkab,cbkj,ic->ia', I.oovv, T2old, L1old)
    L1 += -1.0*einsum('jika,Ibj,Ikb->ia', I.ooov, U11old, LU11old)
    L1 += 1.0*einsum('ibac,Icj,Ijb->ia', I.ovvv, U11old, LU11old)
    L1 += 1.0*einsum('jikb,Ibj,Ika->ia', I.ooov, U11old, LU11old)
    L1 += -1.0*einsum('jbac,Icj,Iib->ia', I.ovvv, U11old, LU11old)
    L1 += -0.5*einsum('ib,cbjk,kjac->ia', F.ov, T2old, L2old)
    L1 += -1.0*einsum('Iij,Ibk,jkab->ia', g.oo, U11old, L2old)
    L1 += -1.0*einsum('ibjc,ck,jkab->ia', I.ovov, T1old, L2old)
    L1 += 0.5*einsum('jikl,bj,lkab->ia', I.oooo, T1old, L2old)
    L1 += -0.5*einsum('ja,bcjk,kicb->ia', F.ov, T2old, L2old)
    L1 += 1.0*einsum('Iba,Icj,jicb->ia', g.vv, U11old, L2old)
    L1 += -1.0*einsum('jbka,cj,kicb->ia', I.ovov, T1old, L2old)
    L1 += 1.0*einsum('jb,cbjk,kiac->ia', F.ov, T2old, L2old)
    L1 += 1.0*einsum('Ijk,Ibj,kiab->ia', g.oo, U11old, L2old)
    L1 += -1.0*einsum('Ibc,Icj,jiab->ia', g.vv, U11old, L2old)
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
    L1 += -1.0*einsum('Iib,bj,I,ja->ia', g.ov, T1old, S1old, L1old)
    L1 += -1.0*einsum('Ija,bj,I,ib->ia', g.ov, T1old, S1old, L1old)
    L1 += 1.0*einsum('jiab,bk,cj,kc->ia', I.oovv, T1old, T1old, L1old)
    L1 += -1.0*einsum('jibc,ck,bj,ka->ia', I.oovv, T1old, T1old, L1old)
    L1 += -1.0*einsum('jkab,cj,bk,ic->ia', I.oovv, T1old, T1old, L1old)
    L1 += -1.0*einsum('Iib,I,Jbj,Jja->ia', g.ov, S1old, U11old, LU11old)
    L1 += -1.0*einsum('Ija,I,Jbj,Jib->ia', g.ov, S1old, U11old, LU11old)
    L1 += 1.0*einsum('Ijb,I,Jbj,Jia->ia', g.ov, S1old, U11old, LU11old)
    L1 += 1.0*einsum('jb,bk,cj,kiac->ia', F.ov, T1old, T1old, L2old)
    L1 += 1.0*einsum('Ijk,bj,I,kiab->ia', g.oo, T1old, S1old, L2old)
    L1 += -1.0*einsum('Ibc,cj,I,jiab->ia', g.vv, T1old, S1old, L2old)
    L1 += 1.0*einsum('jiab,cj,Ibk,Ikc->ia', I.oovv, T1old, U11old, LU11old)
    L1 += 1.0*einsum('jiab,bk,Icj,Ikc->ia', I.oovv, T1old, U11old, LU11old)
    L1 += -1.0*einsum('jibc,ck,Ibj,Ika->ia', I.oovv, T1old, U11old, LU11old)
    L1 += 1.0*einsum('jibc,cj,Ibk,Ika->ia', I.oovv, T1old, U11old, LU11old)
    L1 += -1.0*einsum('jkab,cj,Ibk,Iic->ia', I.oovv, T1old, U11old, LU11old)
    L1 += 1.0*einsum('jkab,bj,Ick,Iic->ia', I.oovv, T1old, U11old, LU11old)
    L1 += -1.0*einsum('jkbc,cj,Ibk,Iia->ia', I.oovv, T1old, U11old, LU11old)
    L1 += -1.0*einsum('Iib,bj,Ick,jkac->ia', g.ov, T1old, U11old, L2old)
    L1 += -0.5*einsum('Iib,cbjk,I,kjac->ia', g.ov, T2old, S1old, L2old)
    L1 += -1.0*einsum('jikb,bl,cj,klac->ia', I.ooov, T1old, T1old, L2old)
    L1 += 0.5*einsum('ibcd,dj,ck,jkab->ia', I.ovvv, T1old, T1old, L2old)
    L1 += -1.0*einsum('Ija,bj,Ick,kicb->ia', g.ov, T1old, U11old, L2old)
    L1 += 1.0*einsum('Ijb,cj,Ibk,kiac->ia', g.ov, T1old, U11old, L2old)
    L1 += 1.0*einsum('Ijb,bk,Icj,kiac->ia', g.ov, T1old, U11old, L2old)
    L1 += -1.0*einsum('Ijb,bj,Ick,kiac->ia', g.ov, T1old, U11old, L2old)
    L1 += -0.5*einsum('Ija,bcjk,I,kicb->ia', g.ov, T2old, S1old, L2old)
    L1 += 1.0*einsum('Ijb,cbjk,I,kiac->ia', g.ov, T2old, S1old, L2old)
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
    L1 += 1.0*einsum('Ijb,bk,cj,I,kiac->ia', g.ov, T1old, T1old, S1old, L2old)
    L1 += 0.5*einsum('jibc,ck,bl,dj,klad->ia', I.oovv, T1old, T1old, T1old, L2old)
    L1 += 0.5*einsum('jkab,bl,cj,dk,lidc->ia', I.oovv, T1old, T1old, T1old, L2old)
    L1 += -1.0*einsum('jkbc,cl,dj,bk,liad->ia', I.oovv, T1old, T1old, T1old, L2old)

# lambda residuals for 2 fermion deexcitation space...
# connected piece not proportional to Lambda...
    L2 = 1.0*einsum('jiba->ijab', I.oovv)

# connected piece proportional to Lambda...
    L2 += 1.0*einsum('jikb,ka->ijab', I.ooov, L1old)
    L2 += -1.0*einsum('jika,kb->ijab', I.ooov, L1old)
    L2 += 1.0*einsum('jcba,ic->ijab', I.ovvv, L1old)
    L2 += -1.0*einsum('icba,jc->ijab', I.ovvv, L1old)
    L2 += 1.0*einsum('Ijb,Iia->ijab', g.ov, LU11old)
    L2 += -1.0*einsum('Ija,Iib->ijab', g.ov, LU11old)
    L2 += -1.0*einsum('Iib,Ija->ijab', g.ov, LU11old)
    L2 += 1.0*einsum('Iia,Ijb->ijab', g.ov, LU11old)
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
    L2 += -1.0*einsum('Ijk,I,kiba->ijab', g.oo, S1old, L2old)
    L2 += 1.0*einsum('Iik,I,kjba->ijab', g.oo, S1old, L2old)
    L2 += 1.0*einsum('kb,ck,jiac->ijab', F.ov, T1old, L2old)
    L2 += -1.0*einsum('ka,ck,jibc->ijab', F.ov, T1old, L2old)
    L2 += -1.0*einsum('Icb,I,jiac->ijab', g.vv, S1old, L2old)
    L2 += 1.0*einsum('Ica,I,jibc->ijab', g.vv, S1old, L2old)
    L2 += -1.0*einsum('jibc,Ick,Ika->ijab', I.oovv, U11old, LU11old)
    L2 += 1.0*einsum('jiac,Ick,Ikb->ijab', I.oovv, U11old, LU11old)
    L2 += 1.0*einsum('kjba,Ick,Iic->ijab', I.oovv, U11old, LU11old)
    L2 += -1.0*einsum('kjbc,Ick,Iia->ijab', I.oovv, U11old, LU11old)
    L2 += 1.0*einsum('kjac,Ick,Iib->ijab', I.oovv, U11old, LU11old)
    L2 += -1.0*einsum('kiba,Ick,Ijc->ijab', I.oovv, U11old, LU11old)
    L2 += 1.0*einsum('kibc,Ick,Ija->ijab', I.oovv, U11old, LU11old)
    L2 += -1.0*einsum('kiac,Ick,Ijb->ijab', I.oovv, U11old, LU11old)
    L2 += 1.0*einsum('jikc,cl,klba->ijab', I.ooov, T1old, L2old)
    L2 += -1.0*einsum('Ijb,Ick,kiac->ijab', g.ov, U11old, L2old)
    L2 += 1.0*einsum('Ija,Ick,kibc->ijab', g.ov, U11old, L2old)
    L2 += -1.0*einsum('Ijc,Ick,kiba->ijab', g.ov, U11old, L2old)
    L2 += 1.0*einsum('Iib,Ick,kjac->ijab', g.ov, U11old, L2old)
    L2 += -1.0*einsum('Iia,Ick,kjbc->ijab', g.ov, U11old, L2old)
    L2 += 1.0*einsum('Iic,Ick,kjba->ijab', g.ov, U11old, L2old)
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
    L2 += 1.0*einsum('Ikb,Ick,jiac->ijab', g.ov, U11old, L2old)
    L2 += -1.0*einsum('Ika,Ick,jibc->ijab', g.ov, U11old, L2old)
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
    L2 += -1.0*einsum('Ijc,ck,I,kiba->ijab', g.ov, T1old, S1old, L2old)
    L2 += 1.0*einsum('Iic,ck,I,kjba->ijab', g.ov, T1old, S1old, L2old)
    L2 += 1.0*einsum('Ikb,ck,I,jiac->ijab', g.ov, T1old, S1old, L2old)
    L2 += -1.0*einsum('Ika,ck,I,jibc->ijab', g.ov, T1old, S1old, L2old)
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

# disconnected piece projecting onto single fermionic excitations...
    L2 += 1.0*einsum('jb,ia->ijab', L1old, F.ov)
    L2 += -1.0*einsum('ja,ib->ijab', L1old, F.ov)
    L2 += -1.0*einsum('ib,ja->ijab', L1old, F.ov)
    L2 += 1.0*einsum('ia,jb->ijab', L1old, F.ov)
    L2 += 1.0*einsum('I,jb,Iia->ijab', S1old, L1old, g.ov)
    L2 += -1.0*einsum('I,ja,Iib->ijab', S1old, L1old, g.ov)
    L2 += -1.0*einsum('I,ib,Ija->ijab', S1old, L1old, g.ov)
    L2 += 1.0*einsum('I,ia,Ijb->ijab', S1old, L1old, g.ov)
    L2 += -1.0*einsum('ck,jb,kiac->ijab', T1old, L1old, I.oovv)
    L2 += 1.0*einsum('ck,ja,kibc->ijab', T1old, L1old, I.oovv)
    L2 += 1.0*einsum('ck,ib,kjac->ijab', T1old, L1old, I.oovv)
    L2 += -1.0*einsum('ck,ia,kjbc->ijab', T1old, L1old, I.oovv)

# lambda residuals for 1 boson deexcitation space...
# connected piece not proportional to Lambda...
    LS1 = 1.0*einsum('I->I', G)
    LS1 += 1.0*einsum('ai,Iia->I', T1old, g.ov)

# connected piece proportional to Lambda...
    LS1 += 1.0*einsum('JI,J->I', w, LS1old)
    LS1 += 1.0*einsum('Iai,ia->I', g.vo, L1old)
    LS1 += 1.0*einsum('ai,Iia->I', F.vo, LU11old)
    LS1 += 1.0*einsum('Iia,Jai,J->I', g.ov, U11old, LS1old)
    LS1 += -1.0*einsum('Iij,ai,ja->I', g.oo, T1old, L1old)
    LS1 += 1.0*einsum('Iab,bi,ia->I', g.vv, T1old, L1old)
    LS1 += 1.0*einsum('J,Jai,Iia->I', G, U11old, LU11old)
    LS1 += 1.0*einsum('Jai,J,Iia->I', g.vo, S1old, LU11old)
    LS1 += -1.0*einsum('ij,ai,Ija->I', F.oo, T1old, LU11old)
    LS1 += 1.0*einsum('ab,bi,Iia->I', F.vv, T1old, LU11old)
    LS1 += -1.0*einsum('Iia,baij,jb->I', g.ov, T2old, L1old)
    LS1 += -1.0*einsum('Iij,Jai,Jja->I', g.oo, U11old, LU11old)
    LS1 += 1.0*einsum('Iab,Jbi,Jia->I', g.vv, U11old, LU11old)
    LS1 += -1.0*einsum('ia,baij,Ijb->I', F.ov, T2old, LU11old)
    LS1 += -1.0*einsum('Jij,Jai,Ija->I', g.oo, U11old, LU11old)
    LS1 += 1.0*einsum('Jab,Jbi,Iia->I', g.vv, U11old, LU11old)
    LS1 += -1.0*einsum('iajb,bi,Ija->I', I.ovov, T1old, LU11old)
    LS1 += 0.5*einsum('ijka,baji,Ikb->I', I.ooov, T2old, LU11old)
    LS1 += -0.5*einsum('iabc,cbij,Ija->I', I.ovvv, T2old, LU11old)
    LS1 += 0.5*einsum('Iij,abik,jkba->I', g.oo, T2old, L2old)
    LS1 += -0.5*einsum('Iab,cbij,jica->I', g.vv, T2old, L2old)
    LS1 += -1.0*einsum('Iia,aj,bi,jb->I', g.ov, T1old, T1old, L1old)
    LS1 += -1.0*einsum('ia,aj,bi,Ijb->I', F.ov, T1old, T1old, LU11old)
    LS1 += -1.0*einsum('Jij,ai,J,Ija->I', g.oo, T1old, S1old, LU11old)
    LS1 += 1.0*einsum('Jab,bi,J,Iia->I', g.vv, T1old, S1old, LU11old)
    LS1 += -1.0*einsum('Iia,bi,Jaj,Jjb->I', g.ov, T1old, U11old, LU11old)
    LS1 += -1.0*einsum('Iia,aj,Jbi,Jjb->I', g.ov, T1old, U11old, LU11old)
    LS1 += -1.0*einsum('Jia,bi,Jaj,Ijb->I', g.ov, T1old, U11old, LU11old)
    LS1 += -1.0*einsum('Jia,aj,Jbi,Ijb->I', g.ov, T1old, U11old, LU11old)
    LS1 += 1.0*einsum('Jia,ai,Jbj,Ijb->I', g.ov, T1old, U11old, LU11old)
    LS1 += -1.0*einsum('Jia,baij,J,Ijb->I', g.ov, T2old, S1old, LU11old)
    LS1 += -1.0*einsum('ijka,bi,aj,Ikb->I', I.ooov, T1old, T1old, LU11old)
    LS1 += 1.0*einsum('iabc,cj,bi,Ija->I', I.ovvv, T1old, T1old, LU11old)
    LS1 += -0.5*einsum('ijab,ci,bajk,Ikc->I', I.oovv, T1old, T2old, LU11old)
    LS1 += -0.5*einsum('ijab,bk,caji,Ikc->I', I.oovv, T1old, T2old, LU11old)
    LS1 += 1.0*einsum('ijab,bi,cajk,Ikc->I', I.oovv, T1old, T2old, LU11old)
    LS1 += 0.5*einsum('Iia,bi,cajk,kjcb->I', g.ov, T1old, T2old, L2old)
    LS1 += 0.5*einsum('Iia,aj,bcik,jkcb->I', g.ov, T1old, T2old, L2old)
    LS1 += -1.0*einsum('Jia,aj,bi,J,Ijb->I', g.ov, T1old, T1old, S1old, LU11old)
    LS1 += 1.0*einsum('ijab,bk,ci,aj,Ikc->I', I.oovv, T1old, T1old, T1old, LU11old)

# lambda residuals for 1 boson + 1 fermion deexcitation space...
# connected piece not proportional to Lambda...
    LU11 = 1.0*einsum('Iia->Iia', g.ov)

# connected piece proportional to Lambda...
    LU11 += -1.0*einsum('Iij,ja->Iia', g.oo, L1old)
    LU11 += 1.0*einsum('Iba,ib->Iia', g.vv, L1old)
    LU11 += 1.0*einsum('JI,Jia->Iia', w, LU11old)
    LU11 += -1.0*einsum('ij,Ija->Iia', F.oo, LU11old)
    LU11 += 1.0*einsum('ba,Iib->Iia', F.vv, LU11old)
    LU11 += -1.0*einsum('ibja,Ijb->Iia', I.ovov, LU11old)
    LU11 += -1.0*einsum('Ibj,jiab->Iia', g.vo, L2old)
    LU11 += -1.0*einsum('Iib,bj,ja->Iia', g.ov, T1old, L1old)
    LU11 += -1.0*einsum('Ija,bj,ib->Iia', g.ov, T1old, L1old)
    LU11 += -1.0*einsum('ib,bj,Ija->Iia', F.ov, T1old, LU11old)
    LU11 += -1.0*einsum('ja,bj,Iib->Iia', F.ov, T1old, LU11old)
    LU11 += -1.0*einsum('Jij,J,Ija->Iia', g.oo, S1old, LU11old)
    LU11 += 1.0*einsum('Jba,J,Iib->Iia', g.vv, S1old, LU11old)
    LU11 += -1.0*einsum('Iib,Jbj,Jja->Iia', g.ov, U11old, LU11old)
    LU11 += -1.0*einsum('Ija,Jbj,Jib->Iia', g.ov, U11old, LU11old)
    LU11 += 1.0*einsum('Ijb,Jbj,Jia->Iia', g.ov, U11old, LU11old)
    LU11 += 1.0*einsum('Jia,Jbj,Ijb->Iia', g.ov, U11old, LU11old)
    LU11 += -1.0*einsum('Jib,Jbj,Ija->Iia', g.ov, U11old, LU11old)
    LU11 += -1.0*einsum('Jja,Jbj,Iib->Iia', g.ov, U11old, LU11old)
    LU11 += -1.0*einsum('jika,bj,Ikb->Iia', I.ooov, T1old, LU11old)
    LU11 += 1.0*einsum('ibac,cj,Ijb->Iia', I.ovvv, T1old, LU11old)
    LU11 += 1.0*einsum('jikb,bj,Ika->Iia', I.ooov, T1old, LU11old)
    LU11 += -1.0*einsum('jbac,cj,Iib->Iia', I.ovvv, T1old, LU11old)
    LU11 += 1.0*einsum('Ijk,bj,kiab->Iia', g.oo, T1old, L2old)
    LU11 += -1.0*einsum('Ibc,cj,jiab->Iia', g.vv, T1old, L2old)
    LU11 += 1.0*einsum('jiab,cbjk,Ikc->Iia', I.oovv, T2old, LU11old)
    LU11 += 0.5*einsum('jibc,cbjk,Ika->Iia', I.oovv, T2old, LU11old)
    LU11 += 0.5*einsum('jkab,cbkj,Iic->Iia', I.oovv, T2old, LU11old)
    LU11 += -0.5*einsum('Iib,cbjk,kjac->Iia', g.ov, T2old, L2old)
    LU11 += -0.5*einsum('Ija,bcjk,kicb->Iia', g.ov, T2old, L2old)
    LU11 += 1.0*einsum('Ijb,cbjk,kiac->Iia', g.ov, T2old, L2old)
    LU11 += -1.0*einsum('Jib,bj,J,Ija->Iia', g.ov, T1old, S1old, LU11old)
    LU11 += -1.0*einsum('Jja,bj,J,Iib->Iia', g.ov, T1old, S1old, LU11old)
    LU11 += 1.0*einsum('jiab,bk,cj,Ikc->Iia', I.oovv, T1old, T1old, LU11old)
    LU11 += -1.0*einsum('jibc,ck,bj,Ika->Iia', I.oovv, T1old, T1old, LU11old)
    LU11 += -1.0*einsum('jkab,cj,bk,Iic->Iia', I.oovv, T1old, T1old, LU11old)
    LU11 += 1.0*einsum('Ijb,bk,cj,kiac->Iia', g.ov, T1old, T1old, L2old)

# disconnected piece projecting onto single fermionic excitations...
    LU11 += 1.0*einsum('I,ia->Iia', LS1old, F.ov)
    LU11 += 1.0*einsum('J,I,Jia->Iia', S1old, LS1old, g.ov)
    LU11 += -1.0*einsum('bj,I,jiab->Iia', T1old, LS1old, I.oovv)

# disconnected piece projecting onto single bosonic excitations...
    LU11 += 1.0*einsum('I,ia->Iia', LS1old, F.ov)
    LU11 += 1.0*einsum('J,I,Jia->Iia', S1old, LS1old, g.ov)
    LU11 += -1.0*einsum('bj,I,jiab->Iia', T1old, LS1old, I.oovv)

    res_L1 = np.linalg.norm(L1) / np.sqrt(L1.size)
    res_L2 = np.linalg.norm(L2) / np.sqrt(L2.size)
    res_LS1 = np.linalg.norm(LS1) / np.sqrt(LS1.size)
    res_LU11 = np.linalg.norm(LU11) / np.sqrt(LU11.size)
    res = res_L1 + res_L2 + res_LS1 + res_LU11 

    return res

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
