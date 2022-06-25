import numpy as np
import math
import pyscf
from pyscf import ao2mo, gto, scf
from ebcc import EBCC

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
# gmat can be specified either spatially or spin-resolved.
g = np.random.random((nbos,nmo,nmo)) * 0.03
omega = np.random.random((nbos)) * 5.

# One boson in bosonic and coupling parts of ansatz
cc = EBCC(myhf, rank=(2, 1, 1), omega=omega, g=g)
e_corr = cc.kernel()
print('EBCCSD correlation energy for rank 211:   {}'.format(e_corr))

# Improve ansatz to double boson excitations
cc = EBCC(myhf, rank=(2, 2, 1), omega=omega, g=g)
e_corr = cc.kernel()
print('EBCCSD correlation energy for rank 221:   {}'.format(e_corr))

# Improve ansatz to double boson excitations in coupling term too
cc = EBCC(myhf, rank=(2, 2, 2), omega=omega, g=g)
e_corr = cc.kernel()
print('EBCCSD correlation energy for rank 222:   {}'.format(e_corr))
