import numpy as np
import json
import math
import pyscf
from pyscf import ao2mo, gto, scf
from ebcc import ebccsd, ccsd_equations
from epcc import ccsd as epcc_ccsd
from epcc import epcc as epcc_epccsd
from epcc import hh_model
from epcc import fci

np.set_printoptions(suppress=True, precision=4, linewidth=np.inf, threshold=np.inf)

def get_e_f(cc, eri, dm1, dm2):
# Get full array of alpha then beta orbitals (this is the ordering of the RDMs)
# NOTE: ebcc returns RDMs in molecular spin-orbitals as occ_a, occ_b, virt_a, virt_b.
    C = np.hstack((cc.mf.mo_coeff[0][:,:cc.na], cc.mf.mo_coeff[1][:,:cc.nb], cc.mf.mo_coeff[0][:,cc.na:], cc.mf.mo_coeff[1][:,cc.nb:]))
# Get full spinorbital integrals in this ordering
    eri_g = ao2mo.full(eri, C, compact=False)
# zero out spin-forbidden sectors due to UHF
    mask_a = [True]*cc.na + [False]*cc.nb + [True]*cc.va + [False]*cc.vb
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
    t_so = np.linalg.multi_dot((C.T, cc.mf.get_hcore(), C))
    t_so[np.ix_(mask_a, mask_b)] = 0.
    t_so[np.ix_(mask_b, mask_a)] = 0.

    E1b = np.einsum('pq,qp',t_so, dm1_eb)
    E2b = np.einsum('pqrs,pqrs', eri_g, dm2_eb) * .5
    Enuc = mol.energy_nuc()
    #print('2b contribution: {}'.format(E2b))
    #print('1b contribution: {}'.format(E1b))
    #print('Enuc contribution: {}'.fohttps://scholar.google.co.uk/scholar?q=energy+evaluation+from+reduced+density+matrices&hl=en&as_sdt=0&as_vis=1&oi=scholartrmat(Enuc))
    #print('Etot: {}'.format(E1b+E2b+Enuc))
    E = E1b+E2b+Enuc
    #print('E(CCSD) = %s, reference %s' % (E, cc.e_corr + cc.ehf))
    print('Fermionic energy: ',E)
    return E

def get_e_b(cc, dm1_bb, dm1_coup_cre, dm1_coup_ann, dm_singbos_cre, dm_singbos_ann, dm1_ferm):

# Get full array of alpha then beta orbitals (this is the ordering of the RDMs)
# NOTE: ebcc returns RDMs in molecular spin-orbitals as occ_a, occ_b, virt_a, virt_b.
    C = np.hstack((cc.mf.mo_coeff[0][:,:cc.na], cc.mf.mo_coeff[1][:,:cc.nb], cc.mf.mo_coeff[0][:,cc.na:], cc.mf.mo_coeff[1][:,cc.nb:]))

    print(dm1_ferm)

    gmat_mo = np.einsum('pq,Ipr,rs->Iqs',C,cc.gmat,C)

    mask_a = [True]*cc.na + [False]*cc.nb + [True]*cc.va + [False]*cc.vb
    mask_b = [not i for i in mask_a]
    mask_bos = [True]*cc.nbos
    gmat_mo[np.ix_(mask_bos,mask_a, mask_b)] = 0.
    gmat_mo[np.ix_(mask_bos,mask_b, mask_a)] = 0.

    E_eb_cre = np.einsum('Ipq,Ipq->',gmat_mo,dm1_coup_cre)
    # I think this term should be "Ipq,Iqp->", which would then make the energy expression invariant if we properly enforce
    # hermiticity on the RDMs, but that gives incorrect results- god knows!
    E_eb_ann = np.einsum('Ipq,Ipq->',gmat_mo,dm1_coup_ann)


    E_eb_cre2 = np.einsum('Ipq,Iqp->', gmat_mo, dm1_coup_cre)
    E_eb_ann2 = np.einsum('Ipq,Iqp->', gmat_mo, dm1_coup_ann)
    print('Boson create coupling energy: ',E_eb_cre)
    print('Boson annihilate coupling energy: ',E_eb_ann)
    print("££$$", abs(dm1_coup_cre.transpose((0, 2, 1)) - dm1_coup_ann).max(), abs(dm1_coup_cre - dm1_coup_ann).max(),
          E_eb_ann2 - E_eb_ann, E_eb_cre2 - E_eb_cre)

    E_b = np.einsum('i,ii->',cc.omega,dm1_bb)
    print('Boson diagonal energy: ',E_b)
    E_offset = 0.0
    if cc.shift:
        E_offset = - 2. * np.einsum("npq,n,pq->", gmat_mo, cc.xi, dm1_ferm)
        E_offset -= np.einsum("n,n,n->", cc.omega, cc.xi, dm_singbos_ann + dm_singbos_cre)

    print("E_offset:", E_offset)
    # Note that G is zero if shift=True?!
    E_singb = np.einsum('i,i->',dm_singbos_cre,cc.G)
    E_singb += np.einsum('i,i->',dm_singbos_ann,cc.G)
    print('Single boson energy: ',E_singb)
    E_singb = 0.

    return E_b + E_eb_cre + E_eb_ann + E_offset

np.set_printoptions(edgeitems=30, linewidth=100000)
# Set up ab initio system with random e-b couplings
mol = pyscf.M(
    atom = 'H 0 0 0; H 0 0 0.75',  # in Angstrom
    basis = '6-31G',
    symmetry = True,
    verbose=1
)

myhf = mol.RHF().run()
nmo = myhf.mo_coeff.shape[1]
nbos = 5
np.random.seed(95)
gmat = np.random.random((nbos,nmo,nmo)) * 0.5
#gmat = (gmat + gmat.transpose((0,2,1))) / 2.
#gmat = np.zeros((nbos, nmo, nmo))
#for i in range(nmo):
#    gmat[:,i,i] = np.random.random((nbos)) * 0.2
#gmat = np.random.random((nbos,nmo,nmo)) * 0.0
omega = 0.1+np.random.random((nbos)) * 10.

print('Running ebcc...')
#eri = ao2mo.kernel(mol, myhf.mo_coeff, compact=False).reshape([nmo,]*4)
eri = ao2mo.restore(1, myhf._eri, nmo)
#cc = ebccsd.EBCCSD(mol, myhf, eri, options={'diis space': 12}, rank=(2,0,0))
#etot, e_corr = cc.kernel()
#print('CCSD (diis) correlation energy for rank (2,0,0):   {}'.format(cc.e_corr))
#print('EBCCSD total energy', etot)
#cc.solve_lambda()
## Generate 1 and 2 RDMs
#dm1_eb_ccsd = cc.make_1rdm_f()
#dm2_eb_ccsd = cc.make_2rdm_f()

#for rank in [(2,0,0), (2,1,1), (2,2,1), (2,2,2)]:
for rank in [(2,1,1)]:
    for shift in [False]:
        cc = ebccsd.EBCCSD(mol, myhf, eri, options={'diis space': 14, 'tthresh': 1e-7}, rank=rank, omega=omega, gmat=gmat, shift=shift, autogen_code=True)
        etot, e_corr = cc.kernel()
        print('EBCCSD (diis) correlation energy for rank {} and shift {}:   {}'.format(rank,shift,cc.e_corr))
        print('EBCCSD total energy', etot)

# Generate 1 and 2 RDMs
#        dm1_eb = cc.make_1rdm_f()
#        dm2_eb = cc.make_2rdm_f()
#        dm1_bb = cc.make_1rdm_b()
#        dm1_coup_cre, dm1_coup_ann = cc.make_eb_coup_rdm()
#        dm1_b_sing_cre, dm1_b_sing_ann = cc.make_sing_b_dm()
#
#        E_ferm = get_e_f(cc, eri, dm1_eb, dm2_eb)
#        if rank[2] > 0:
#            E_bos = get_e_b(cc, dm1_bb, dm1_coup_cre, dm1_coup_ann, dm1_b_sing_cre, dm1_b_sing_ann)
#        else:
#            E_bos = 0.
#        #print('Bosonic energy from density matrices (L=T^+): ',E_bos)
#        print('Total energy from density matrices (L=T^+): ',E_ferm + E_bos + cc.const)
        
        cc.solve_lambda()
# Generate 1 and 2 RDMs
        dm1_eb = cc.make_1rdm_f()
        dm2_eb = cc.make_2rdm_f()
        dm1_bb = cc.make_1rdm_b()
        dm1_coup_cre, dm1_coup_ann = cc.make_eb_coup_rdm()
        dm1_b_sing_cre, dm1_b_sing_ann = cc.make_sing_b_dm()
        #print('Single boson expectation: ')
        #print(dm1_b_sing_cre)
        #print(dm1_b_sing_ann)

        E_ferm = get_e_f(cc, eri, dm1_eb, dm2_eb)
        print('Fermionic total energy from density matrices: ',E_ferm)
        if rank[2] > 0:
            E_bos = get_e_b(cc, dm1_bb, dm1_coup_cre, dm1_coup_ann, dm1_b_sing_cre, dm1_b_sing_ann, dm1_eb)
        else:
            E_bos = 0.
        print('Bosonic energy from density matrices: ',E_bos)
        print('Zero boson shift: ',cc.const)
        print('Total energy from density matrices: ',E_ferm + E_bos + cc.const)
        print("*"*20)
        print('ERROR between total energy from RDMs vs. projection: ',etot-(E_ferm + E_bos + cc.const))
        print("*" * 20)
