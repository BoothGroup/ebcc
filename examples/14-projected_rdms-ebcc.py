'''
Compute RDMs from ebcc, in both shifted and unshifted boson representations, from a subspace projection.

    This is done in two ways:
        o Democratic partitioning of the full density matrix
        o Projection of the T/L amplitudes in the construction of the individual density matrices.

    Since these are both done on the exact full system CCSD-11 solution (which can be thought of as a
    complete bath space), they should both be trivially exact, and should only differ once the 
    solutions differ for the different subspaces (in which case the projected approach is likely 
    to be more accurate and physical.

    The partitioning of the full system into 'clusters' in this case is just achieved via randomly
    splitting the occupied and virtual states into disjoint orthogonal orbitals. Note that for the
    projection of the T/L amplitudes, we only want to provide a projector onto the *occupied*
    space of each cluster, as opposed to democratic partitioning, which requires a fragmentation and
    projection of the full (occ+virt) orbital space.
'''
import numpy as np
import pyscf
import scipy.stats
from pyscf import ao2mo
from ebcc import ebccsd, utils

def make_random_subspaces(nocc, nvir, nclust):
    ''' Returns five lists of length nclust:
    occ_clust_size: Integer list of number of occupied orbitals in each cluster
    vir_clust_size: Integer list of number of virtual orbitals in each cluster
    clust_orbs: ntot x nc arrays, giving the cluster orbitals in the canonical full system basis.
                Occupied orbitals first, then virtuals
    occ_projs: nocc x nocc arrays, giving a projector in the occupied canonical basis, into each clusters occupied space.
    tot_projs: ntot x ntot arrays, giving a projector into the total canonical basis, into each clusters total space.
    '''
    # Make subspaces.
    # Create occupied subspace sizes (note these can be zero or odd). 
    # Generate number of occ and vir orbitals in each cluster via multinomial distribution
    occ_clust_size = np.random.multinomial(nocc, np.ones(nclust)/nclust, size=1)[0]
    vir_clust_size = np.random.multinomial(nvir, np.ones(nclust)/nclust, size=1)[0]
    assert(sum(occ_clust_size)==nocc)
    assert(sum(vir_clust_size)==nvir)
    # Find arbitrary orthogonal occupied and virtual 'orbitals' in each cluster
    occ_rep = scipy.stats.ortho_group.rvs(nocc) # These are the representation of the total orbitals
    vir_rep = scipy.stats.ortho_group.rvs(nvir)
    # The total orbital space of each cluster, defined by a set of occupied and virtual (non-canonical) orthogonal orbitals represented in the full canonical basis
    clust_orbs = [] 
    # Projectors of each cluster into its occupied space (in the representation of the canonical occupied orbitals)
    occ_projs = []
    # Total projectors of each cluster (in the representation of all canonical orbitals, occ+vir)
    tot_projs = []
    print('occupied cluster sizes: ',occ_clust_size)
    print('virtual cluster sizes: ',vir_clust_size)
    for i in range(nclust):
        occ_start_ind = np.sum(occ_clust_size[:i])
        vir_start_ind = np.sum(vir_clust_size[:i])
        # Find the cluster occupied orbitals (in the space of all canonical orbitals), noting that they will have no weight on virtuals
        clust_occ_orbs = np.vstack((occ_rep[:, occ_start_ind:occ_start_ind+occ_clust_size[i]], np.zeros((nvir, occ_clust_size[i])) ))
        clust_vir_orbs = np.vstack((np.zeros((nocc, vir_clust_size[i])), vir_rep[:, vir_start_ind:vir_start_ind+vir_clust_size[i]] ))
        clust_orbs.append(np.hstack((clust_occ_orbs, clust_vir_orbs)))
        assert(clust_orbs[-1].shape == (nocc+nvir, occ_clust_size[i]+vir_clust_size[i]))
        assert(np.isclose(np.trace(np.dot(clust_orbs[-1], clust_orbs[-1].T)), occ_clust_size[i]+vir_clust_size[i]))

        # Create a projector (in the canonical occupied representation) into the occupied space of this cluster
        occ_projs.append(np.dot(occ_rep[:, occ_start_ind:occ_start_ind+occ_clust_size[i]], occ_rep[:, occ_start_ind:occ_start_ind+occ_clust_size[i]].T))
        assert(occ_projs[-1].shape == (nocc,nocc))
        assert(np.isclose(np.trace(occ_projs[-1]), occ_clust_size[i]))
        # Create a total space projector (in the full canonical representation) into the full space of this cluster
        tot_projs.append(np.dot(clust_orbs[-1], clust_orbs[-1].T))
        assert(tot_projs[-1].shape == (nocc+nvir, nocc+nvir))
        assert(np.isclose(np.trace(tot_projs[-1]), occ_clust_size[i]+vir_clust_size[i]))
    print('Created {} disjoint clusters which partition the space'.format(nclust))
    return (occ_clust_size, vir_clust_size, clust_orbs, occ_projs, tot_projs)

mol = pyscf.M(
    atom = [
    ['O' , (0. , 0.     , 0.)],
    ['H' , (0. , -0.757 , 0.587)],
    ['H' , (0. , 0.757  , 0.587)]],
    basis = 'cc-pvdz')
mf = mol.RHF().run()
nao = mf.mo_coeff.shape[1]
nbos = 5
np.random.seed(92)
gmat = np.random.random((nbos,nao,nao)) * 0.01
omega = 0.1+np.random.random((nbos)) * 10 
eri = ao2mo.restore(1, mf._eri, nao)

# Now run ebccsd models
for rank in [(2,1,1)]:
    for shift in [True, False]:
        cc = ebccsd.EBCCSD.fromUHFobj(mf, options={'tthresh': 1e-8}, rank=rank, omega=omega, gmat=gmat, shift=shift, autogen_code=True)
        e_corr = cc.kernel()
        print('EBCCSD correlation energy for rank {} and shift {}:   {}'.format(rank,shift,cc.e_corr))
        print('EBCCSD total energy', mf.e_tot + e_corr - cc.const)
        
        cc.solve_lambda()

        nclust = 5
        occ_clust_size, vir_clust_size, clust_orbs, occ_projs, tot_projs = make_random_subspaces(cc.no, cc.nv, nclust)
        
        # Create RDMs with respect to the original bosonic operators (no shift included) 
        dm1_eb = cc.make_1rdm_f()
        dm2_eb = cc.make_2rdm_f()
        dm1_bb = cc.make_1rdm_b()
        dm1_coup_cre, dm1_coup_ann = cc.make_eb_coup_rdm()
        dm1_b_sing_cre, dm1_b_sing_ann = cc.make_sing_b_dm()
        # Also compute boson-fermion coupling dms in shifted boson representation
        dm1_coup_cre_shift, dm1_coup_ann_shift = cc.make_eb_coup_rdm(unshifted_bos=False)
        # Check symmetries
        assert(np.allclose(dm1_eb, dm1_eb.T))
        assert(np.allclose(dm1_coup_cre,dm1_coup_ann.transpose((0,2,1))))
        assert(np.allclose(dm1_coup_cre_shift, dm1_coup_ann_shift.transpose((0,2,1))))
        assert(np.allclose(dm2_eb, dm2_eb.transpose(1,0,3,2)))
        assert(np.allclose(dm2_eb, -dm2_eb.transpose(2,1,0,3)))
        assert(np.allclose(dm2_eb, -dm2_eb.transpose(0,3,2,1)))

        # Create the 'democratically partitioned' DMs, just by projecting the first index of the total density matrix
        # with the complete projector of each cluster, and summing the resulting matrices
        dm1_eb_dp = np.zeros_like(dm1_eb)
        dm2_eb_dp = np.zeros_like(dm2_eb)
        dm1_coup_cre_dp = np.zeros_like(dm1_coup_cre)
        dm1_coup_ann_dp = np.zeros_like(dm1_coup_ann)
        dm1_coup_cre_shift_dp = np.zeros_like(dm1_coup_cre)
        dm1_coup_ann_shift_dp = np.zeros_like(dm1_coup_ann)
        for i in range(nclust):
            dm1_eb_dp += np.dot(tot_projs[i], dm1_eb)
            dm2_eb_dp += np.einsum('pi,ijkl->pjkl', tot_projs[i], dm2_eb)
            dm1_coup_cre_dp += np.einsum('pi,Iij->Ipj', tot_projs[i], dm1_coup_cre)
            dm1_coup_ann_dp += np.einsum('pi,Iij->Ipj', tot_projs[i], dm1_coup_ann)
            dm1_coup_cre_shift_dp += np.einsum('pi,Iij->Ipj', tot_projs[i], dm1_coup_cre_shift)
            dm1_coup_ann_shift_dp += np.einsum('pi,Iij->Ipj', tot_projs[i], dm1_coup_ann_shift)

        # Check that democratically partitioned density matrices are exact 
        # (since all cluster density matrices are exact, this is just testing the completeness of the projector really)
        assert(np.allclose(dm1_eb_dp, dm1_eb))
        assert(np.allclose(dm2_eb_dp, dm2_eb))
        assert(np.allclose(dm1_coup_cre_dp, dm1_coup_cre))
        assert(np.allclose(dm1_coup_ann_dp, dm1_coup_ann))
        assert(np.allclose(dm1_coup_cre_shift_dp, dm1_coup_cre_shift))
        assert(np.allclose(dm1_coup_ann_shift_dp, dm1_coup_ann_shift))
        print('"Democratically partitioned" density matrices exact...')

        # Now, directly construct the projective cluster density matrix contributions, via projection of the occupied T/L amplitudes 
        # directly in the DM constructions.
        dm1_proj = np.zeros_like(dm1_eb)
        dm2_proj = np.zeros_like(dm2_eb)
        dm1_coup_cre_proj = np.zeros_like(dm1_coup_cre)
        dm1_coup_ann_proj = np.zeros_like(dm1_coup_ann)
        dm1_coup_cre_shift_proj = np.zeros_like(dm1_coup_cre_shift)
        dm1_coup_ann_shift_proj = np.zeros_like(dm1_coup_ann_shift)
        for i in range(nclust):
            dm1_clustproj = cc.make_1rdm_f(write=True, subspace_proj=occ_projs[i])
            dm2_clustproj = cc.make_2rdm_f(write=False, subspace_proj=occ_projs[i])
            dm1_coup_cre_clustproj, dm1_coup_ann_clustproj = cc.make_eb_coup_rdm(write=False, subspace_proj=occ_projs[i])
            dm1_coup_cre_shift_clustproj, dm1_coup_ann_shift_clustproj = cc.make_eb_coup_rdm(write=False, unshifted_bos=False, subspace_proj=occ_projs[i])

            # Combine projected DMs of each cluster
            dm1_proj += dm1_clustproj
            dm2_proj += dm2_clustproj
            dm1_coup_cre_proj += dm1_coup_cre_clustproj
            dm1_coup_ann_proj += dm1_coup_ann_clustproj
            dm1_coup_cre_shift_proj += dm1_coup_cre_shift_clustproj
            dm1_coup_ann_shift_proj += dm1_coup_ann_shift_clustproj
            # Note that each projected cluster density matrix should be physically sensible, and have the correct permutational symmetries
            # for a GHF dm. They should also trace to the correct number of electrons in each cluster (given by the occupied orbital number)
            assert(np.allclose(dm1_clustproj, dm1_clustproj.T))
            assert(np.allclose(dm2_clustproj, dm2_clustproj.transpose(1,0,3,2)))
            assert(np.allclose(dm2_clustproj, -dm2_clustproj.transpose(2,1,0,3)))
            assert(np.allclose(dm2_clustproj, -dm2_clustproj.transpose(0,3,2,1)))
            assert(np.allclose(dm1_coup_cre_clustproj, dm1_coup_ann_clustproj.transpose((0,2,1))))
            assert(np.allclose(dm1_coup_cre_shift_clustproj, dm1_coup_ann_shift_clustproj.transpose((0,2,1))))

        # Check all projectively occupied partitioned density matrices are exact
        assert(np.allclose(dm1_proj, dm1_eb))
        assert(np.allclose(dm2_proj, dm2_eb))
        assert(np.allclose(dm1_coup_cre_proj, dm1_coup_cre))
        assert(np.allclose(dm1_coup_ann_proj, dm1_coup_ann))
        assert(np.allclose(dm1_coup_cre_shift_proj, dm1_coup_cre_shift))
        assert(np.allclose(dm1_coup_ann_shift_proj, dm1_coup_ann_shift))
        print('"Projectively occupied partitioning" of density matrices exact...')
