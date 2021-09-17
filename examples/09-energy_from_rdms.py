'''
Compute RDMs from ebcc, and check that contracting them with the hamiltonian
returns the total energy and agrees with pyscf. Also compute RDMs from approximate 
lambda amplitudes (L = T^+), which should also get correct energy.
'''

import numpy as np
import pyscf
from pyscf import ao2mo, scf
from pyscf import cc as pyscf_cc
from ebcc import ebccsd, utils

mol = pyscf.M(
    atom = [
    ['O' , (0. , 0.     , 0.)],
    ['H' , (0. , -0.757 , 0.587)],
    ['H' , (0. , 0.757  , 0.587)]],
    basis = 'cc-pvdz')
mf = mol.RHF().run()
# Generate reference pyscf CCSD values
mycc = mf.CCSD().run()
print('pyscf CCSD correlation energy', mycc.e_corr)
dm1 = mycc.make_rdm1()
dm2 = mycc.make_rdm2()

# Compute energy from pyscf density matrices
h1 = np.einsum('pi,pq,qj->ij', mf.mo_coeff.conj(), mf.get_hcore(), mf.mo_coeff)
eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape([mf.mo_coeff.shape[1]]*4)
# Compute sum of 1b, 2b and nuclear energy contributions
Etot_pyscf = np.einsum('pq,qp', h1, dm1) + np.einsum('pqrs,pqrs', eri, dm2) * .5 + mol.energy_nuc()
print('pyscf total energy from dms: {}'.format(Etot_pyscf))

# Get AO integrals for ebcc
eri = ao2mo.restore(1, mf._eri, mf.mol.nao_nr())
# Set up cc object and run kernel for T-amplitude optimization (no bosons)
cc = ebccsd.EBCCSD(mol, mf, eri, options={'tthresh': 1e-9, 'diis space': 12}, autogen_code=True)
etot, ecorr = cc.kernel()
print('EBCCSD correlation energy', cc.e_corr)
assert(np.allclose(etot, Etot_pyscf))

# Solve lambda equations
cc.solve_lambda()

# Generate 1 and 2 RDMs
dm1_eb = cc.make_1rdm_f()
dm2_eb = cc.make_2rdm_f()

# NOTE: ebcc returns RDMs in molecular spin-orbitals as occ_a, occ_b, virt_a, virt_b.
# Therefore, construct a list of spinorbital coefficients in this order
C = np.hstack((cc.mf.mo_coeff[0][:,:cc.na], cc.mf.mo_coeff[1][:,:cc.nb], cc.mf.mo_coeff[0][:,cc.na:], cc.mf.mo_coeff[1][:,cc.nb:]))
# Get full spinorbital integrals in this ordering
eri_g = ao2mo.full(eri, C, compact=False)
# zero out spin-forbidden sectors of the integrals
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
t_so = np.linalg.multi_dot((C.T, mf.get_hcore(), C))
t_so[np.ix_(mask_a, mask_b)] = 0.
t_so[np.ix_(mask_b, mask_a)] = 0.

# Contract for energy from RDMs
E1b = np.einsum('pq,qp',t_so, dm1_eb)
E2b = np.einsum('pqrs,pqrs', eri_g, dm2_eb) * .5
Enuc = mol.energy_nuc()
print('Etot from ebcc RDMs: {}'.format(E1b+E2b+Enuc))
E = E1b+E2b+Enuc
print('E(CCSD) = %s, reference %s' % (E, Etot_pyscf))
assert(np.allclose(E, Etot_pyscf))

# Note that even if the lambda equations aren't solved in ebcc,
# it should make the approximation that L = T^+, and still
# get the energy right
# Remove the optimized lambda amplitudes
cc.L1 = cc.L2 = None
# Get the density matrices again (now approximated without correct lambdas)
dm1_eb_approxL = cc.make_1rdm_f()
dm2_eb_approxL = cc.make_2rdm_f()

# Check the energy again
E1b = np.einsum('pq,qp',t_so, dm1_eb_approxL)
E2b = np.einsum('pqrs,pqrs', eri_g, dm2_eb_approxL) * .5
Enuc = mol.energy_nuc()
print('Etot from approximated ebcc RDMs: {}'.format(E1b+E2b+Enuc))
E = E1b+E2b+Enuc
print('E(CCSD) = %s, reference %s' % (E, Etot_pyscf))
assert(np.allclose(E, Etot_pyscf))
