''' Compute the first hole and particle one-body spectral moments at CCSD level. 
    Compare to direct contraction from RDMs.'''
import numpy as np
import pyscf
from pyscf import gto, scf, cc, ao2mo
from ebcc import ebccsd

mol = pyscf.M(
    atom = 'N 0 0 0; N 0 0 1.2',
    basis = 'cc-pvdz')
mf = mol.RHF().run()
eri = ao2mo.restore(1, mf._eri, mf.mol.nao_nr())

# Set up cc object and run kernel for T-amplitude optimization (no bosons)
mycc = ebccsd.EBCCSD(mol, mf, eri, options={'tthresh': 1e-10, 'diis space': 12}, autogen_code=True)
etot, ecorr = mycc.kernel()
mycc.solve_lambda()

# Generate 1 and 2 RDMs
dm1 = mycc.make_1rdm_f()
dm2 = mycc.make_2rdm_f()

# Direct computation of IP and EA first spectral moments, as <c^+ (H-E) c> and <c (H-E) c^+> 
ip_mom = mycc.make_ip_1mom()
ea_mom = mycc.make_ea_1mom()

### TESTS and CHECKS

# 1. Compute total energy from RDMs and first hole moment
# First compute the MO GHF basis for the hamiltonian terms, in the appropriate ordering for the orbitals in ebcc 
# NOTE: ebcc returns RDMs in molecular spin-orbitals as occ_a, occ_b, virt_a, virt_b.
C = np.hstack((mycc.mf.mo_coeff[0][:,:mycc.na], mycc.mf.mo_coeff[1][:,:mycc.nb], mycc.mf.mo_coeff[0][:,mycc.na:], mycc.mf.mo_coeff[1][:,mycc.nb:]))
mask_a = [True]*mycc.na + [False]*mycc.nb + [True]*mycc.va + [False]*mycc.vb
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
t_so = np.linalg.multi_dot((C.T, mycc.mf.get_hcore(), C))
t_so[np.ix_(mask_a, mask_b)] = 0.
t_so[np.ix_(mask_b, mask_a)] = 0.

# E from RDMs
E_rdms = np.einsum('jk,kj->', t_so, dm1) + 0.5*np.einsum('ijkl,ijkl->', eri_g, dm2) + mol.energy_nuc()
assert(np.isclose(E_rdms, etot))

# E from moments
# Note that the sign of the first hole moment is flipped cf. e.g. https://arxiv.org/pdf/1904.08019.pdf.
# This is due to the definition as <c^+ (H-E) c>, rather than the commutator expression (which is actually the negative of this).
E_ip_mom = 0.5*np.einsum('ij,ji->', t_so, dm1) - 0.5*np.trace(ip_mom) + mol.energy_nuc()
assert(np.isclose(E_ip_mom, etot))
assert(np.allclose(ip_mom, ip_mom.T))
assert(np.allclose(ea_mom, ea_mom.T))

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
assert(np.isclose(E_ip_mom_contract2rdm, etot))

# The two approaches to the moments will only be identical for 2-electron systems? (Where they are both exact)
# Evaluate the difference.
diff = np.abs(ip_mom_contract2rdm - ip_mom)
ind = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
print('Mean and max difference (and value) between RDM and direct IP moment calculation: {} {} {}'.format(np.mean(diff), diff[ind], ip_mom[ind]))
diff = np.abs(ea_mom_contract2rdm - ea_mom)
ind = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
print('Mean and max difference (and value) between RDM and direct EA moment calculation: {} {} {}'.format(np.mean(diff), diff[ind], ea_mom[ind]))
