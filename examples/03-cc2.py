"""
Example of a CC2 calculation.
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

# Run a CC2 calculation
cc2 = EBCC(mf, ansatz="CC2")
cc2.kernel()
