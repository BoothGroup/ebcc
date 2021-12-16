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
gmat = np.random.random((nbos,nmo,nmo)) * 0.03
omega = np.random.random((nbos)) * 5.
# AO eri array passed in
eri = ao2mo.restore(1, myhf._eri, nmo)

# One boson in bosonic and coupling parts of ansatz
cc = ebccsd.EBCCSD(mol, myhf, eri, options={'diis space': 12}, rank=(2,1,1), omega=omega, gmat=gmat, autogen_code=True)
etot, e_corr = cc.kernel()
print('EBCCSD correlation energy for rank 211:   {}'.format(cc.e_corr))

# Improve ansatz to double boson excitations
cc = ebccsd.EBCCSD(mol, myhf, eri, options={'diis space': 12}, rank=(2,2,1), omega=omega, gmat=gmat, autogen_code=True)
etot, e_corr = cc.kernel()
print('EBCCSD correlation energy for rank 221:   {}'.format(cc.e_corr))

# Improce ansatz to double boson excitations in coupling term too
cc = ebccsd.EBCCSD(mol, myhf, eri, options={'diis space': 12}, rank=(2,2,2), omega=omega, gmat=gmat, autogen_code=True)
etot, e_corr = cc.kernel()
print('EBCCSD correlation energy for rank 222:   {}'.format(cc.e_corr))
