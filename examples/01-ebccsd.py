import numpy as np
from pyscf import gto, scf

from ebcc import EBCC

np.random.seed(123)

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

#    ,-------- Fermionic ansatz
#    | ,------ Bosonic excitation amplitudes
#    | | ,---- Rank of fermions in coupling term
#    | | | ,-- Rank of bosons in coupling term
#    v v v v
# ____ _ _ _
# CCSD-S-1-1: One-boson amplitudes and one-boson-one-fermion coupling
ccsd = EBCC(
    mf,
    ansatz="CCSD-S-1-1",
    omega=omega,
    g=g,
)
ccsd.kernel()

#    ,--------- Fermionic ansatz
#    |  ,------ Bosonic excitation amplitudes
#    |  | ,---- Rank of fermions in coupling term
#    |  | | ,-- Rank of bosons in coupling term
#    v  v v v
# ____ __ _ _
# CCSD-SD-1-1: Two-boson amplitudes and one-boson-one-fermion coupling
ccsd = EBCC(
    mf,
    ansatz="CCSD-SD-1-1",
    omega=omega,
    g=g,
)
ccsd.kernel()

#    ,--------- Fermionic ansatz
#    |  ,------ Bosonic excitation amplitudes
#    |  | ,---- Rank of fermions in coupling term
#    |  | | ,-- Rank of bosons in coupling term
#    v  v v v
# ____ __ _ _
# CCSD-SD-1-2: Two-boson amplitudes and two-boson-one-fermion coupling
ccsd = EBCC(
    mf,
    ansatz="CCSD-SD-1-2",
    omega=omega,
    g=g,
)
ccsd.kernel()
