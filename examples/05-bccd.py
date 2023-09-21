"""
Example of a Brueckner orbital calculation using a CCSD reference.
"""

import numpy as np
from pyscf import gto, scf

from ebcc import EBCC

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

# Run a RHF calculation using PySCF
mf = scf.RHF(mol)
mf.kernel()

# Run a CCSD calculation
ccsd = EBCC(mf, ansatz="CCSD")
ccsd.kernel()

# Run a Brueckner orbital calculation using the CCSD reference
ccsd.brueckner(e_tol=1e-6, t_tol=1e-5)
