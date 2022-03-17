''' Compute the EOM dd spectral moments at CCSD level. 

    There are various ways that this can be done. The formal expression for the CC dd-moments is:
    dd_moms[p,q,r,s,n] = <(1+L) e^-T a^+_p a_q e^T P ((e^-T H e^T - E_cc)P)^n e^-T a^+_r a_s e^T> - delta_{n0} <(1+L) e^-T a^+_p a_q e^T |gs><gs| e^-T a^+_r a_s e^T>.

    We can approximate this in various ways. In the EOM approximation, the P operator is a projection
    operator onto the space of singles and doubles only (meaning there is no GS (at least reference) contribution in the
    Lehmann representation, and the second term is neglected).

    With the 'include_ref_proj' keyword, we can also include the projector onto the reference state, in which case the
    expressions become exact for two-electrons. The second term then needs to be explicitly removed, which can be done
    (approximately) via the hermitized 1RDMs, the non-hermitized 1RDMs (depending on the hermit_gs_contrib kwarg).
    Note that whether you remove the GS contribution with hermitized or non-hermitized 1RDMs is seemingly unimportant numerically.

    We then compare these zeroth moments in terms of their energy, and agreement with the 2RDM, as well as comparing
    the Hbar eigenvalues to the neutral excitation energies of ee-eom-ccsd from pyscf.
'''
import numpy as np
import pyscf
import itertools
import sys
from pyscf import gto, scf, cc, ao2mo
from ebcc import ebccsd, ccsd_equations

mol = pyscf.M(
    atom = 'Li 0 0 0; H 0 0 1.5',
    basis = 'sto-3g')
mf = mol.RHF().run()
pyscf_cc = mf.CCSD().run()
Ecorr_pyscf = pyscf_cc.e_corr

eri = ao2mo.restore(1, mf._eri, mf.mol.nao_nr())

# Set up cc object and run kernel for T-amplitude optimization (no bosons)
mycc = ebccsd.EBCCSD.fromUHFobj(mf, options={'tthresh': 1e-10, 'diis space': 12}, autogen_code=True)
ecorr = mycc.kernel()
assert(np.isclose(ecorr, pyscf_cc.e_corr))
mycc.solve_lambda()

# Generate 1 and 2 RDMs
dm1 = mycc.make_1rdm_f()
# Note that dm2[p,q,r,s] = <p^+ r^+ s q>
dm2 = mycc.make_2rdm_f()

# EOM computation of dd spectral moments
# dd moments as dd_moms[p,q,r,s,n] = <p^+ q (H-E)^n r^+ s> - \delta_{n0} <p^+ q r^+ s>
# First is the computation under the EOM approximation (i.e. excited state space is limited to S+D)
dd_moms_eom = mycc.make_dd_EOM_moms(1)
# Next is the approximation where the excited state space can include the reference state (with the GS contribution removed from non-hermitian 1RDMs for the zeroth moment)
dd_moms_with_ref_proj = mycc.make_dd_EOM_moms(1, include_ref_proj=True)
# Finally, there is the approximation which is the same above, but removing the GS contribution from hermitian 1RDMs for the zeroth moment.
# Both of these last two approximations should be exact for all orders for 2 electrons
dd_moms_with_ref_proj_herm = mycc.make_dd_EOM_moms(1, include_ref_proj=True, hermit_gs_contrib=True)
# Note that we can also choose to only create the dd-moments over an arbitrary subspace, given by a projector which can be passed in with optional argument 'pertspace'

### TESTS and CHECKS
umf = mf.to_uhf()
nspato = umf.mol.nao_nr()
na = sum(umf.mo_occ[0] > 0.0)
nb = sum(umf.mo_occ[1] > 0.0)
va = nspato - na
vb = nspato - nb
nocc = na + nb
nvir = va + vb
ntot = dm2.shape[0]

# 0. Find dd moment in an arbitrary 2-orbital subspace, and check that this is the same as generating the whole dd-moment and projecting after.
# Find random vectors spanning two orbital subspace
pertspace = np.zeros((ntot, 2))
orb_ind_1, orb_ind_2 = np.random.choice(ntot, size=2, replace=False) # Find two unique integer indices to span.
mask_inds = [False]*ntot
mask_inds[orb_ind_1] = True
mask_inds[orb_ind_2] = True 
pertspace[orb_ind_1, 0] = 1.
pertspace[orb_ind_2, 1] = 1.
# Unitarily mix the vectors to a new representation
angle = np.random.random()*2*np.pi
print('Testing subspace dd-moment evaluation, by choosing orbitals {} and {}, and unitarily mixing them by angle {} radians to check consistency...'.format(orb_ind_1,orb_ind_2,angle))
rot = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
pert_space_ = np.dot(pertspace, rot)
dd_moms_subspace = mycc.make_dd_EOM_moms(1, include_ref_proj=True, pertspace=pert_space_)
# Now, rotate the subspace dd moments back to the original (canonical) orbital space
dd_moms_subspace_canonical = np.einsum('pqrsn,wp,xq,yr,zs->wxyzn', dd_moms_subspace, pert_space_, pert_space_, pert_space_, pert_space_)
# Check that this is the same as the subspace of the full dd moments spanned by the orbital indices orb_ind_1 and orb_ind_2
orig_space_mask = np.ix_(mask_inds, mask_inds, mask_inds, mask_inds, [True]*dd_moms_with_ref_proj.shape[-1])
assert(np.allclose(dd_moms_subspace_canonical[orig_space_mask], dd_moms_with_ref_proj[orig_space_mask]))
print('Choosing custom perturbation subspace working correctly')

# 1. Check agreement between zeroth dd moment and moment from 2RDM
# First, convert the 2RDM into a product of single-excitations form
moms_dm2 = dm2 + np.einsum('qr,ps->pqrs', np.eye(dm2.shape[0]), dm1)
# Now, remove the ground state contribution
moms_dm2 -= np.einsum('pq,rs->pqrs', dm1, dm1)

# Look for the MSE in the elements of the zeroth moment, compared to the 'exact' solution
mse_dd_eom = np.linalg.norm(moms_dm2-dd_moms_eom[:,:,:,:,0]) / (dm2.shape[0]**2)
mse_dd_eom_ref = np.linalg.norm(moms_dm2-dd_moms_with_ref_proj[:,:,:,:,0]) / (dm2.shape[0]**2)
mse_dd_eom_ref_herm = np.linalg.norm(moms_dm2-dd_moms_with_ref_proj_herm[:,:,:,:,0]) / (dm2.shape[0]**2)

print('Mean squared error in the zeroth moment (EOM approximation) compared to "exact" CCSD zeroth moment: ',mse_dd_eom)
print('Mean squared error in the zeroth moment (EOM with reference proj approximation) compared to "exact" CCSD zeroth moment: ',mse_dd_eom_ref)
print('Mean squared error in the zeroth moment (EOM with reference proj and zeroth-order moment removing hermitized (rather than unhermitized) 1RDMs) compared to "exact" CCSD zeroth moment: ',mse_dd_eom_ref_herm)
print('Note that the latter two approximations should be exact for two-electron systems')
print('')
print('Testing exactness of zeroth dd-mom from 2RDM vs. EOM approximation with reference in projector in individual sectors (I think only ov blocks in the bra should be error for >2e):')
# Convert 2RDM to product of single excitation form
dm2_pose = dm2 + np.einsum('qr,ps->pqrs', np.eye(dm2.shape[0]), dm1)
# Also include back in the GS contribution to the zeroth moment of the approximation with the reference projector
dm2_pose_eom_ref = dd_moms_with_ref_proj_herm[:,:,:,:,0] + np.einsum('pq,rs->pqrs',dm1,dm1)
print('Is full matrix in agreement: ',np.allclose(dm2_pose_eom_ref,dm2_pose))
print('Testing vvvv agreement: ',np.allclose(dm2_pose_eom_ref[nocc:,nocc:,nocc:,nocc:],dm2_pose[nocc:,nocc:,nocc:,nocc:]))
print('Testing ovvv agreement: ',np.allclose(dm2_pose_eom_ref[:nocc,nocc:,nocc:,nocc:],dm2_pose[:nocc,nocc:,nocc:,nocc:]))
print('Testing vvvo agreement: ',np.allclose(dm2_pose_eom_ref[nocc:,nocc:,nocc:,:nocc],dm2_pose[nocc:,nocc:,nocc:,:nocc]))
print('Testing vvov agreement: ',np.allclose(dm2_pose_eom_ref[nocc:,nocc:,:nocc,nocc:],dm2_pose[nocc:,nocc:,:nocc,nocc:]))
print('Testing vovv agreement: ',np.allclose(dm2_pose_eom_ref[nocc:,:nocc,nocc:,nocc:],dm2_pose[nocc:,:nocc,nocc:,nocc:]))
print('Testing oovv agreement: ',np.allclose(dm2_pose_eom_ref[:nocc,:nocc,nocc:,nocc:],dm2_pose[:nocc,:nocc,nocc:,nocc:]))
print('Testing ovov agreement: ',np.allclose(dm2_pose_eom_ref[:nocc,nocc:,:nocc,nocc:],dm2_pose[:nocc,nocc:,:nocc,nocc:]))
print('Testing ovvo agreement: ',np.allclose(dm2_pose_eom_ref[:nocc,nocc:,nocc:,:nocc],dm2_pose[:nocc,nocc:,nocc:,:nocc]))
print('Testing voov agreement: ',np.allclose(dm2_pose_eom_ref[nocc:,:nocc,:nocc,nocc:],dm2_pose[nocc:,:nocc,:nocc,nocc:]))
print('Testing vovo agreement: ',np.allclose(dm2_pose_eom_ref[nocc:,:nocc,nocc:,:nocc],dm2_pose[nocc:,:nocc,nocc:,:nocc]))
print('Testing vvoo agreement: ',np.allclose(dm2_pose_eom_ref[nocc:,nocc:,:nocc,:nocc],dm2_pose[nocc:,nocc:,:nocc,:nocc]))
print('Testing vooo agreement: ',np.allclose(dm2_pose_eom_ref[nocc:,:nocc,:nocc,:nocc],dm2_pose[nocc:,:nocc,:nocc,:nocc]))
print('Testing ovoo agreement: ',np.allclose(dm2_pose_eom_ref[:nocc,nocc:,:nocc,:nocc],dm2_pose[:nocc,nocc:,:nocc,:nocc]))
print('Testing oovo agreement: ',np.allclose(dm2_pose_eom_ref[:nocc,:nocc,nocc:,:nocc],dm2_pose[:nocc,:nocc,nocc:,:nocc]))
print('Testing ooov agreement: ',np.allclose(dm2_pose_eom_ref[:nocc,:nocc,:nocc,nocc:],dm2_pose[:nocc,:nocc,:nocc,nocc:]))
print('Testing oooo agreement: ',np.allclose(dm2_pose_eom_ref[:nocc,:nocc,:nocc,:nocc],dm2_pose[:nocc,:nocc,:nocc,:nocc]))

# 2. Calculate the energy from the zeroth moment (using the 1RDM)
# Is there a better way to directly get the energy from ACFDT?
# Add back the GS projector contribution, and convert to normal ordered form of 2RDM
print('')
print('Calculating ground state energy from zeroth moment approximations')
rdm2_eom = dd_moms_eom[:,:,:,:,0] + np.einsum('pq,rs->pqrs',dm1, dm1) - np.einsum('qr,ps->pqrs', np.eye(dm2.shape[0]), dm1)
rdm2_ref = dd_moms_with_ref_proj[:,:,:,:,0] + np.einsum('pq,rs->pqrs',dm1, dm1) - np.einsum('qr,ps->pqrs', np.eye(dm2.shape[0]), dm1)
rdm2_ref_herm = dd_moms_with_ref_proj_herm[:,:,:,:,0] + np.einsum('pq,rs->pqrs',dm1, dm1) - np.einsum('qr,ps->pqrs', np.eye(dm2.shape[0]), dm1)

# Now get the hamiltonian terms in the right (GHF) ordering
# NOTE: ebcc returns RDMs in molecular spin-orbitals as occ_a, occ_b, virt_a, virt_b.
# Therefore, construct a list of spinorbital coefficients in this order
C = np.hstack((umf.mo_coeff[0][:,:na], umf.mo_coeff[1][:,:nb], umf.mo_coeff[0][:,na:], umf.mo_coeff[1][:,nb:]))
# Get full spinorbital integrals in this ordering
eri_g = ao2mo.full(eri, C, compact=False)
# zero out spin-forbidden sectors of the integrals
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
t_so = np.linalg.multi_dot((C.T, mf.get_hcore(), C))
t_so[np.ix_(mask_a, mask_b)] = 0.
t_so[np.ix_(mask_b, mask_a)] = 0.

# Contract for correlation energy from different RDM approximations from zeroth moments
E10b = np.einsum('pq,qp',t_so, dm1) + mol.energy_nuc()
e_eom = np.einsum('pqrs,pqrs', eri_g, rdm2_eom) * .5 + E10b - mf.energy_tot()
e_eom_ref = np.einsum('pqrs,pqrs', eri_g, rdm2_ref) * .5 + E10b - mf.energy_tot()
e_eom_ref_herm = np.einsum('pqrs,pqrs', eri_g, rdm2_ref_herm) * .5 + E10b - mf.energy_tot()
print('Etot from EOM approx: {}, while pyscf={} (error = {})'.format(e_eom, Ecorr_pyscf, abs(e_eom-Ecorr_pyscf)))
print('Etot from EOM+ref approx: {}, while pyscf={} (error = {})'.format(e_eom_ref, Ecorr_pyscf, abs(e_eom_ref-Ecorr_pyscf)))
print('Etot from EOM+ref+herm approx: {}, while pyscf={} (error = {})'.format(e_eom_ref_herm, Ecorr_pyscf, abs(e_eom_ref_herm-Ecorr_pyscf)))

# 3. Now (if the system is small enough), generate the whole H_bar matrix, and investigate the excitations compared to EOM-CC in pyscf
# Note that there is the potential for differences if the reference projector is used or not
if (nocc*nvir)**2 < 4000:
    # Testing full EOM hamiltonian roots compared to pyscf
    print('Completely diagonalizing Hbar-E_cc to get all excitation energies, in space of singles and doubles...')
    full_eom_h = ccsd_equations.gen_dd_eom_matrix(mycc)
    e = np.linalg.eigvals(full_eom_h)
    e_sorted = np.sort(e.real)
    print('First positive 30 eigenvalues from diagonalization of effective EOM hamiltonian (without reference state in projector): ')
    e_sorted_pos = e_sorted[e_sorted > 1.e-12]
    print(e_sorted_pos[:30])
    print('Roots from pyscf ee-eom')
    eee, cee = pyscf_cc.eeccsd(nroots=50)
    print(eee[:30])
    print('')
    print('Completely diagonalizing Hbar-E_cc to get all excitation energies, in space of reference, singles and doubles...')
    full_eom_h = ccsd_equations.gen_dd_eom_matrix(mycc, include_ref_proj=True)
    #print('First row and column of Hbar: ')
    #print(full_eom_h[:,0])
    #print('***')
    #print(full_eom_h[0,:])
    #e = np.linalg.eigvals(full_eom_h)
    e_sorted = np.sort(e.real)
    print('First positive 30 eigenvalues from diagonalization of effective EOM hamiltonian: ')
    e_sorted_pos = e_sorted[e_sorted > 1.e-12]
    #e_sorted_pos = e_sorted
    print(e_sorted_pos[:30])
    print('Roots from pyscf ee-eom')
    eee, cee = pyscf_cc.eeccsd(nroots=50)
    print(eee[:30])
