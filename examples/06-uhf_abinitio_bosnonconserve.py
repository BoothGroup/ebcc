import numpy as np
import math
import pyscf
from pyscf import ao2mo, gto, scf
from ebcc import ebccsd

# Set up ab initio system with random e-b couplings
mol = pyscf.M(
    atom = 'F 0 0 0; F 0 0 2.1',  # in Angstrom
    basis = '6-31G',
    symmetry = True,
    verbose=1
)

mf = mol.UHF().run()
nmo = mf.mo_coeff[1].shape[1]
nbos = 5
# Boson energies and couplings to AO density
# Note that these couplings are for the boson *annihilation* operator (i.e. gmat[p,q,x] c_p^+ c_q b_x)
# with the creation operator appropriately transposed
gmat = np.random.random((nbos,nmo,nmo)) * 0.005
omega = np.random.random((nbos)) * 5.
G = np.random.random((nbos)) * 0.05
# Can alternatively represent G as a coupling to the fermionic density.
gmat_mod = gmat + np.einsum("I,pq->Ipq", G, np.eye(nmo)) / mol.nelectron

options = {'diis space': 12}
cc_noG = ebccsd.EBCCSD.fromUHFobj(mf, options = options, rank=(2, 1, 1), omega=omega, gmat=gmat_mod, shift=True, autogen_code=True)
e_corr = cc_noG.kernel()
print('EBCCSD correlation energy for rank 211 with boson-nonconserving term folded into fermion-boson coupling:',
      cc_noG.e_corr)
print('EBCCSD total energy', e_corr + mf.e_tot - cc_noG.const)

cc_G = ebccsd.EBCCSD.fromUHFobj(mf, options = options, rank=(2, 1, 1), omega=omega, gmat=gmat, shift=True, G=G, autogen_code=True)
e_corr = cc_G.kernel()
print('EBCCSD correlation energy for rank 211 with explicit boson-nonconserving term:', cc_G.e_corr)
print('EBCCSD total energy', e_corr + mf.e_tot - cc_G.const)
assert(np.allclose(cc_noG.e_corr, cc_G.e_corr))
print("Correlation energies match with and without explicit boson-nonconserving term!")