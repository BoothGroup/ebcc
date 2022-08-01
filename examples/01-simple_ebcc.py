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

import numpy as np
from pyscf import gto, scf
from ebcc import EBCC

mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

mf = scf.RHF(mol)
mf.kernel()

# Define boson energies and couplings to AO density:
nbos = 5
nmo = mf.mo_occ.size
g = np.random.random((nbos, nmo, nmo)) * 0.03
omega = np.random.random((nbos,)) * 5.0

# One-boson amplitudes and one-boson-one-fermion coupling:
ccsd = EBCC(mf, rank=("SD", "S", "S"), omega=omega, g=g)
ccsd.kernel()
print("%s correlation energy:" % ccsd.name, ccsd.e_corr)

# Include two-boson amplitudes:
ccsd = EBCC(mf, rank=("SD", "SD", "S"), omega=omega, g=g)
ccsd.kernel()
print("%s correlation energy:" % ccsd.name, ccsd.e_corr)

# Include two-boson-one-fermion coupling:
ccsd = EBCC(mf, rank=("SD", "SD", "SD"), omega=omega, g=g)
ccsd.kernel()
print("%s correlation energy:" % ccsd.name, ccsd.e_corr)
