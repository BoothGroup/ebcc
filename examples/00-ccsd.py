"""
Example of a simple CCSD calculation.
"""

import numpy as np
from pyscf import gto, scf

from ebcc import EBCC

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

# Run a Hartree-Fock calculation using PySCF
mf = scf.RHF(mol)
mf.kernel()

# Run a CCSD calculation using EBCC
ccsd = EBCC(mf)
ccsd.kernel()
