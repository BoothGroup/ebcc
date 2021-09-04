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
gmat = np.random.random((nbos,nmo,nmo)) * 0.01
omega = np.random.random((nbos)) * 5.

# AO eri array passed in
eri = ao2mo.restore(1, mf._eri, nmo)
options = {'damp': 0.3}
cc = ebccsd.EBCCSD(mol, mf, eri, options = options, rank=(2,1,1), omega=omega, gmat=gmat, shift=True, autogen_code=False)
etot, e_corr = cc.kernel()
print('EBCCSD correlation energy', cc.e_corr)
print('EBCCSD total energy', etot)

options = {'diis space': 12}
cc_diis = ebccsd.EBCCSD(mol, mf, eri, options = options, rank=(2,1,1), omega=omega, gmat=gmat, shift=True, autogen_code=False)
etot, e_corr = cc_diis.kernel()
print('EBCCSD correlation energy', cc_diis.e_corr)
print('EBCCSD total energy', etot)
assert(np.allclose(cc_diis.e_corr, cc.e_corr))
