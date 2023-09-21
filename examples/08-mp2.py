"""
Example of a simple MP2 calculation.
"""

import numpy as np
from pyscf import gto, scf

from ebcc import REBCC

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

# Run a Hartree-Fock calculation using PySCF
mf = scf.RHF(mol)
mf.kernel()

# The CC solver can be used for Moller-Plesset perturbation theory,
# and EBCC will detect that convergence isn't necessary via the
# `ansatz.is_one_shot` attribute.
mp2 = REBCC(mf, ansatz="MP2")
mp2.kernel()

# Note that this is not the most efficient way to compute MP energies,
# as EBCC will still generate bare amplitudes stored in memory.
