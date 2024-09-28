"""
Example of customising the Hamiltonian using PySCF for `ebcc` calculations.
"""

import numpy as np
from pyscf import gto, scf

from ebcc import EBCC

# Hubbard parameter
n = 10
u = 4.0

# Define a fake molecule
mol = gto.M()
mol.nelectron = 10
mol.verbose = 4

# Define the 1-electron Hamiltonian
h1e = np.zeros((n, n))
for i in range(n - 1):
    h1e[i, i + 1] = h1e[i + 1, i] = -1.0
h1e[0, n - 1] = h1e[n - 1, 0] = -1.0  # Periodic boundary conditions

# Define the 2-electron Hamiltonian
h2e = np.zeros((n, n, n, n))
for i in range(n):
    h2e[i, i, i, i] = u

# Define a fake mean-field object
mf = scf.RHF(mol)
mf.get_hcore = lambda *args: h1e
mf.get_ovlp = lambda *args: np.eye(n)
mf._eri = h2e
mf.kernel()

# Run a EBCC calculation
ccsd = EBCC(mf, ansatz="CCSD")
ccsd.kernel()
