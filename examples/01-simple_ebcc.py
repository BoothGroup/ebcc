import numpy as np
import math
import pyscf
from pyscf import ao2mo, gto, scf
from ebcc import ebccsd

# Set up ab initio system with random e-b couplings
mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = '6-31G',
    symmetry = True,
    verbose=1
)

myhf = mol.RHF().run()
nmo = myhf.mo_coeff.shape[1]
nbos = 5
# Boson energies and couplings to AO density
gmat = np.random.random((nbos,nmo,nmo)) * 0.02
omega = np.random.random((nbos)) * 5.

# AO eri array passed in
eri = ao2mo.restore(1, myhf._eri, nmo)
# One boson in bosonic and coupling parts of ansatz
cc = ebccsd.EBCCSD(mol, myhf, eri, rank=(2,1,1), omega=omega, gmat=gmat)
etot, e_corr = cc.kernel()
print('EBCCSD correlation energy:   {}'.format(cc.e_corr))
print('EBCCSD total energy', etot)
