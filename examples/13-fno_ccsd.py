"""
Example of a simple FNO-CCSD calculation.
"""

import numpy as np
from pyscf import gto, scf

from ebcc import EBCC
from ebcc.space import construct_fno_space

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

# Run a Hartree-Fock calculation using PySCF
mf = scf.RHF(mol)
mf.kernel()

# Construct the FNOs
no_coeff, no_occ, no_space = construct_fno_space(mf, occ_tol=1e-3)

# Run a FNO-CCSD calculation using EBCC
ccsd = EBCC(mf, mo_coeff=no_coeff, mo_occ=no_occ, space=no_space)
ccsd.kernel()
