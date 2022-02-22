''' Compute the first hole and particle one-body spectral moments at CCSD level
    for the CCSD-11 model.. 
    Compare exact, contraction from RDMs and EOM approaches (see header of 18-sp_moms-ccsd.py for more info.'''
import numpy as np
import pyscf
from pyscf import gto, scf, cc, ao2mo
from ebcc import ebccsd, ccsd_equations

mol = pyscf.M(
    atom = 'N 0 0 0; N 0 0 1.2',
    basis = 'cc-pvdz')
mf = mol.RHF().run()
nbos = 8
nao = mf.mol.nao_nr()
np.random.seed(92)
gmat = np.random.random((nbos,nao,nao)) * 0.01
omega = 0.1+np.random.random((nbos)) * 10 
eri = ao2mo.restore(1, mf._eri, nao)

# Set up cc object and run kernel for T-amplitude optimization (no bosons)
mycc = ebccsd.EBCCSD.fromUHFobj(mf, options={'tthresh': 1e-10, 'diis space': 12}, autogen_code=True)
ecorr = mycc.kernel()
etot = ecorr + mf.e_tot - mycc.const
mycc.solve_lambda()

# Generate 1 and 2 RDMs
dm1 = mycc.make_1rdm_f()
dm2 = mycc.make_2rdm_f()

# Direct computation of IP and EA first spectral moments, as <c^+ (H-E) c> and <c (H-E) c^+> 
ip_mom = mycc.make_ip_1mom()
ea_mom = mycc.make_ea_1mom()

# EOM moments
ip_moms = mycc.make_ip_EOM_moms(1)
#ea_moms = mycc.make_ea_EOM_moms(1)

### TESTS and CHECKS

# 1. Compare the EOM CCSD-11 moments to the direct evaluation
umf = mf.to_uhf()
nspato = umf.mol.nao_nr()
na = sum(umf.mo_occ[0] > 0.0)
nb = sum(umf.mo_occ[1] > 0.0)
va = nspato - na
vb = nspato - nb
nocc = na + nb
nv = va + vb
print('Are the exact and EOM zeroth IP moments (1DMs) correct?',np.allclose(dm1, ip_moms[:,:,0]))
#print('Are the exact and EOM zeroth EA moments (hole DM) correct?', np.allclose(np.eye(dm1.shape[0])-dm1, ea_moms[:,:,0]))
print('Are the occ-occ blocks of exact and and EOM first IP moments correct?', np.allclose(ip_mom[:nocc,:nocc], ip_moms[:nocc,:nocc,1]))
print('Are the vir-vir blocks of exact and and EOM first IP moments correct?', np.allclose(ip_mom[nocc:,nocc:], ip_moms[nocc:,nocc:,1]))
print('Are the occ-vir blocks of exact and and EOM first IP moments correct?', np.allclose(ip_mom[:nocc,nocc:], ip_moms[:nocc,nocc:,1]))
#print('Are the occ-occ blocks of exact and and EOM first EA moments correct?', np.allclose(ea_mom[:nocc,:nocc], ea_moms[:nocc,:nocc,1]))
# Note that even for 2 electrons, I don't think that the EA-EOM has to be exactly correct.
#print('Are the vir-vir blocks of exact and and EOM first EA moments correct?', np.allclose(ea_mom[nocc:,nocc:], ea_moms[nocc:,nocc:,1]))
#print('Are the occ-vir blocks of exact and and EOM first EA moments correct?', np.allclose(ea_mom[:nocc,nocc:], ea_moms[:nocc,nocc:,1]))
if np.allclose(gmat, np.zeros_like(gmat)):
    # Check that at zero electron-boson coupling, that the moments reproduce the ccsd model?
    # Hack so we call the CCSD model function directly
    ccsd_ip_moms = ccsd_equations.hole_moms_eom(mycc, 1)
    assert(np.allclose(ccsd_ip_moms, ip_moms))
    print('In the limit of no boson coupling, moments are the same as the ccsd model.')

# 1. Compute total energy from RDMs and first hole moment
# First compute the MO GHF basis for the hamiltonian terms, in the appropriate ordering for the orbitals in ebcc 
# NOTE: ebcc returns RDMs in molecular spin-orbitals as occ_a, occ_b, virt_a, virt_b.
C = np.hstack((umf.mo_coeff[0][:,:na], umf.mo_coeff[1][:,:nb], umf.mo_coeff[0][:,na:], umf.mo_coeff[1][:,nb:]))
mask_a = [True]*na + [False]*nb + [True]*va + [False]*vb
mask_b = [not elem for elem in mask_a]
# Get full spinorbital integrals in this ordering
eri_g = ao2mo.full(eri, C, compact=False)
# zero out spin-forbidden sectors due to UHF
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

# Fermionic part of the energy from RDMs (obviously this isn't the total energy)
E_rdms = np.einsum('jk,kj->', t_so, dm1) + 0.5*np.einsum('ijkl,ijkl->', eri_g, dm2) + mol.energy_nuc()

# Fermionic energy from moments
# Note that the sign of the first hole moment is flipped cf. e.g. https://arxiv.org/pdf/1904.08019.pdf.
# This is due to the definition as <c^+ (H-E) c>, rather than the commutator expression (which is actually the negative of this).
E_ip_mom = 0.5*np.einsum('ij,ji->', t_so, dm1) - 0.5*np.trace(ip_mom) + mol.energy_nuc()
assert(np.isclose(E_ip_mom, E_rdms))
assert(np.allclose(ip_mom, ip_mom.T))
assert(np.allclose(ea_mom, ea_mom.T))
print('Fermionic energies from moments correct.')
E_ip_eom_mom = 0.5*np.einsum('ij,ji->', t_so, ip_moms[:,:,0]) - 0.5*np.trace(ip_moms[:,:,1]) + mol.energy_nuc()
if np.isclose(E_ip_eom_mom, etot):
    print('Fermionic energies from EOM moments correct and exactly equal to CCSD energy.')
else:
    print('Fermionic energy from EOM-CC moments compared to exact: ',E_ip_eom_mom, E_rdms,' with error: ',E_rdms-E_ip_eom_mom)
    print('Note that we do not expect energies from EOM to be exact without 3h2p excitations.')

# 2. Compute moments via contraction of 2-RDM.
# *** Note that the moments constructed in this way are not identical to the direct construction, due to the non-variational nature of CC theory.
#     However, both approaches should give the same total energy, and both are exact for 2e systems.
#     Are these differences related to LR-CC vs. EOM-CC greens functions?
ip_mom_contract2rdm = -np.einsum('jk,ik->ij', t_so, dm1)
ip_mom_contract2rdm -= np.einsum('jklm,iklm->ij', eri_g, dm2) 
# Hermitize
ip_mom_contract2rdm = 0.5*(ip_mom_contract2rdm + ip_mom_contract2rdm.T)
# Also construct EA moment
ea_mom_contract2rdm = t_so.T - np.einsum('ia,bi->ab', t_so, dm1)
ea_mom_contract2rdm += np.einsum('bakl,kl->ab', eri_g, dm1) - np.einsum('iabl,il->ab', eri_g, dm1)
ea_mom_contract2rdm += np.einsum('iakl,ilkb->ab', eri_g, dm2)
# Hermitize
ea_mom_contract2rdm = 0.5*(ea_mom_contract2rdm + ea_mom_contract2rdm.T)

# Energy should be the same
E_ip_mom_contract2rdm = 0.5*np.einsum('ij,ji->', t_so, dm1) - 0.5*np.trace(ip_mom_contract2rdm) + mol.energy_nuc()
assert(np.isclose(E_ip_mom_contract2rdm, E_rdms))

# The two approaches to the moments will only be identical for 2-electron systems? (Where they are both exact)
# Evaluate the difference.
diff = np.abs(ip_mom_contract2rdm - ip_mom)
ind = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
print('Mean and max difference (and value) between RDM and direct IP moment calculation: {} {} {}'.format(np.mean(diff), diff[ind], ip_mom[ind]))
diff = np.abs(ea_mom_contract2rdm - ea_mom)
ind = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
print('Mean and max difference (and value) between RDM and direct EA moment calculation: {} {} {}'.format(np.mean(diff), diff[ind], ea_mom[ind]))
