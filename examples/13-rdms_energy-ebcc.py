'''
Compute RDMs from ebcc, and use to find total energy to compare to projected estimate
'''

import numpy as np
import pyscf
from pyscf import ao2mo, scf
from pyscf import cc as pyscf_cc
from ebcc import ebccsd, utils

def get_e_f(mf, eri, dm1, dm2):
# Get full array of alpha then beta orbitals (this is the ordering of the RDMs)
# NOTE: ebcc returns RDMs in molecular spin-orbitals as occ_a, occ_b, virt_a, virt_b.
    umf = mf.to_uhf()
    nspato = umf.mol.nao_nr()
    na = sum(umf.mo_occ[0] > 0.0)
    nb = sum(umf.mo_occ[1] > 0.0)
    va = nspato - na
    vb = nspato - nb
    no = na + nb
    nv = va + vb
    C = np.hstack((umf.mo_coeff[0][:,:na], umf.mo_coeff[1][:,:nb], umf.mo_coeff[0][:,na:], umf.mo_coeff[1][:,nb:]))
# Get full spinorbital integrals in this ordering
    eri_g = ao2mo.full(eri, C, compact=False)
# zero out spin-forbidden sectors due to UHF
    mask_a = [True]*na + [False]*nb + [True]*va + [False]*vb
    mask_b = [not i for i in mask_a]
    eri_g[np.ix_(mask_b,mask_a,mask_a,mask_a)] = 0.0
    eri_g[np.ix_(mask_a,mask_b,mask_a,mask_a)] = 0.0
    eri_g[np.ix_(mask_a,mask_a,mask_b,mask_a)] = 0.0
    eri_g[np.ix_(mask_a,mask_a,mask_a,mask_b)] = 0.0
    eri_g[np.ix_(mask_a,mask_b,mask_b,mask_b)] = 0.0
    eri_g[np.ix_(mask_b,mask_a,mask_b,mask_b)] = 0.0
    eri_g[np.ix_(mask_b,mask_b,mask_a,mask_b)] = 0.0
    eri_g[np.ix_(mask_b,mask_b,mask_b,mask_a)] = 0.0
    eri_g[np.ix_(mask_b,mask_a,mask_b,mask_a)] = 0.0
    eri_g[np.ix_(mask_a,mask_b,mask_a,mask_b)] = 0.0
    eri_g[np.ix_(mask_a,mask_b,mask_b,mask_a)] = 0.0
    eri_g[np.ix_(mask_b,mask_a,mask_a,mask_b)] = 0.0
# Get 1e hamiltonian in spin-orbital basis
    t_so = np.linalg.multi_dot((C.T, umf.get_hcore(), C))
    t_so[np.ix_(mask_a, mask_b)] = 0.
    t_so[np.ix_(mask_b, mask_a)] = 0.

    E1b = np.einsum('pq,qp',t_so, dm1_eb)
    E2b = np.einsum('pqrs,pqrs', eri_g, dm2_eb) * .5
    Enuc = mol.energy_nuc()
    E = E1b+E2b+Enuc
    return E

def get_e_b(mf, cc, dm1_bb, dm1_coup_cre, dm1_coup_ann, dm_singbos_cre, dm_singbos_ann, dm1_ferm, unshifted_bos):

    umf = mf.to_uhf()
    nspato = umf.mol.nao_nr()
    na = sum(umf.mo_occ[0] > 0.0)
    nb = sum(umf.mo_occ[1] > 0.0)
    va = nspato - na
    vb = nspato - nb

    if cc.rank[2] == 0:
        return 0.0

    # Get full array of alpha then beta orbitals (this is the ordering of the RDMs)
    # NOTE: ebcc returns RDMs in molecular spin-orbitals as occ_a, occ_b, virt_a, virt_b.
    C = np.hstack((umf.mo_coeff[0][:,:na], umf.mo_coeff[1][:,:nb], umf.mo_coeff[0][:,na:], umf.mo_coeff[1][:,nb:]))

    gmat_mo = cc.gmatso

    # Contraction of coupling RDMs with coupling hamiltonian
    E_eb_cre = np.einsum('Ipq,Iqp->',gmat_mo,dm1_coup_cre)
    E_eb_ann = np.einsum('Ipq,Ipq->',gmat_mo,dm1_coup_ann)
    #print('Boson create coupling energy: ',E_eb_cre)
    #print('Boson annihilate coupling energy: ',E_eb_ann)

    # Coupling matrices between annihilation and creation should be related
    assert(np.allclose(dm1_coup_ann, dm1_coup_cre.transpose((0,2,1))))

    E_b = np.einsum('i,ii->',cc.omega,dm1_bb)
    #print('Boson diagonal energy: ',E_b)

    # Note that if there is a shift in the Hamiltonian to remove coupling to a static
    # MF density, then we need to correct for that in the energy expression.
    # We can either directly modify the energy (done here), or shift back the RDMs we obtain (see example 12)
    E_offset = 0.
    if (not unshifted_bos) and cc.shift:
        E_offset = -2. * np.einsum("npq,n,pq->", gmat_mo, cc.xi, dm1_ferm)
        E_offset -= np.einsum("n,n,n->", cc.omega, cc.xi, dm_singbos_ann + dm_singbos_cre)
        E_offset += cc.const
    #print("Bosonic offset energy: ",E_offset)

    return E_b + E_eb_cre + E_eb_ann + E_offset


mol = pyscf.M(
    atom = [
    ['O' , (0. , 0.     , 0.)],
    ['H' , (0. , -0.757 , 0.587)],
    ['H' , (0. , 0.757  , 0.587)]],
    basis = 'cc-pvdz')
mf = mol.RHF().run()
nao = mf.mo_coeff.shape[1]
nbos = 5
np.random.seed(97)
# Note that gmat is the tensor corresponding to the boson annihilation
# i.e. the hamiltonian is gmat[p,q,x] c_p^+ c_q b_x
# The term corresponding to the boson creation is appropriately
# transposed to ensure overall hermiticity.
gmat = np.random.random((nbos,nao,nao)) * 0.01
omega = 0.1+np.random.random((nbos)) * 10 
eri = ao2mo.restore(1, mf._eri, nao)

for rank in [(2,1,1)]:
    for shift in [True, False]:
        cc = ebccsd.EBCCSD.fromUHFobj(mf, options={'tthresh': 1e-8}, rank=rank, omega=omega, gmat=gmat, shift=shift, autogen_code=True)
        e_corr = cc.kernel()
        etot = mf.e_tot - cc.const + e_corr
        print('EBCCSD correlation energy for rank {} and shift {}:   {}'.format(rank,shift,cc.e_corr))
        print('EBCCSD total energy', etot)
        
        # First check RDMs with L=T^+ (which should give the same energy)
        unshifted_bos = True # This is default
        dm1_eb = cc.make_1rdm_f()
        dm2_eb = cc.make_2rdm_f()
        dm1_bb = cc.make_1rdm_b()
        dm1_coup_cre, dm1_coup_ann = cc.make_eb_coup_rdm()
        dm1_b_sing_cre, dm1_b_sing_ann = cc.make_sing_b_dm()

        E_ferm = get_e_f(mf, eri, dm1_eb, dm2_eb)
        E_bos = get_e_b(mf, cc, dm1_bb, dm1_coup_cre, dm1_coup_ann, dm1_b_sing_cre, dm1_b_sing_ann, dm1_eb, unshifted_bos)
        print('Total energy from density matrices (L=T^+): ',E_ferm + E_bos)
        if np.isclose(etot,(E_ferm + E_bos + cc.const)):
            print('*** Total energy agrees between projection and RDMs with L=T^+, CC rank={} and shift={}'.format(rank,shift))
        else:
            print('ERROR between total energy from RDMs vs. projection: ',etot-(E_ferm + E_bos))
            assert(np.isclose(etot,(E_ferm + E_bos)))

        cc.solve_lambda()
        
        # First check RDMs with L=T^+ (which should give the same energy)
        dm1_eb = cc.make_1rdm_f()
        dm2_eb = cc.make_2rdm_f()
        dm1_b_sing_cre, dm1_b_sing_ann = cc.make_sing_b_dm()
        dm1_bb = cc.make_1rdm_b()
        dm1_coup_cre, dm1_coup_ann = cc.make_eb_coup_rdm()

        E_ferm = get_e_f(mf, eri, dm1_eb, dm2_eb)
        E_bos = get_e_b(mf, cc, dm1_bb, dm1_coup_cre, dm1_coup_ann, dm1_b_sing_cre, dm1_b_sing_ann, dm1_eb, unshifted_bos)
        print('Total energy from density matrices (opt L): ',E_ferm + E_bos)
        if np.isclose(etot,(E_ferm + E_bos + cc.const)):
            print('*** Total energy agrees between projection and RDMs with opt L, CC rank={} and shift={}'.format(rank,shift))
        else:
            print('ERROR between total energy from RDMs vs. projection: ',etot-(E_ferm + E_bos))
            assert(np.isclose(etot,(E_ferm + E_bos)))
        
        # Now, compute energy wrt the shifted boson RDMs
        unshifted_bos = False
        dm1_bb = cc.make_1rdm_b(unshifted_bos=unshifted_bos)
        dm1_coup_cre, dm1_coup_ann = cc.make_eb_coup_rdm(unshifted_bos=unshifted_bos)

        E_bos = get_e_b(mf, cc, dm1_bb, dm1_coup_cre, dm1_coup_ann, dm1_b_sing_cre, dm1_b_sing_ann, dm1_eb, unshifted_bos)
        print('Total energy from density matrices (opt L and shifted boson RDMs): ',E_ferm + E_bos)
        if np.isclose(etot,(E_ferm + E_bos)):
            print('*** Total energy agrees between projection and RDMs with opt L, shifted boson RDMs, CC rank={} and shift={}'.format(rank,shift))
        else:
            print('ERROR between total energy from RDMs vs. projection: ',etot-(E_ferm + E_bos))
            assert(np.isclose(etot,(E_ferm + E_bos)))
