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

# AO eri array passed in
options = {'damp': 0.3}
cc = ebccsd.EBCCSD.fromUHFobj(mf, options = options, rank=(2,1,1), omega=omega, gmat=gmat, shift=True, autogen_code=True)
e_corr = cc.kernel()
print('EBCCSD correlation energy for rank 211:', cc.e_corr)
print('EBCCSD total energy', e_corr + mf.e_tot - cc.const)

options = {'diis space': 12}
cc_diis = ebccsd.EBCCSD.fromUHFobj(mf, options = options, rank=(2,1,1), omega=omega, gmat=gmat, shift=True, autogen_code=True)
e_corr = cc_diis.kernel()
print('EBCCSD correlation energy for rank 211:', cc_diis.e_corr)
print('EBCCSD total energy', e_corr + mf.e_tot - cc_diis.const)
assert(np.allclose(cc_diis.e_corr, cc.e_corr))
