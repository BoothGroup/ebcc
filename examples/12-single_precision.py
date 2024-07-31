"""
Example of single-precision calculations.
"""

import numpy as np
from pyscf import gto, scf

from ebcc import REBCC
from ebcc.precision import single_precision

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "N 0 0 0; N 0 0 1.1"
mol.basis = "cc-pvtz"
mol.build()

# Run a Hartree-Fock calculation using PySCF
mf = scf.RHF(mol)
mf.kernel()

# Run a CCSD calculation using EBCC at single precision
with single_precision():
    sp = REBCC(mf, ansatz="CCSD", e_tol=1e-6, t_tol=1e-4)
    sp.kernel()

# Use the amplitudes to finish convergence at double precision -
# the types should be promoted automatically
dp = REBCC(mf, ansatz="CCSD", e_tol=1e-8, t_tol=1e-6)
dp.amplitudes = sp.amplitudes
dp.kernel()
