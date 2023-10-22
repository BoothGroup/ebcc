"""
Example of a density-fitted CCSD calculation.
"""

import numpy as np
from pyscf import gto, scf

from ebcc import REBCC

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

# Run a RHF calculation with DF using PySCF
mf = scf.RHF(mol)
mf = mf.density_fit()
mf.kernel()

# Run a DFCCSD calculation using EBCC - the ansatz is still specified
# as `"CCSD"`, but this uses a DFCCSD implementation. The T2 integrals
# are still stored in memory, but i.e. (vv|vv) and (vv|vo) integrals
# do not need to be.
ccsd = REBCC(mf, ansatz="CCSD")
ccsd.kernel()
