"""
Example of a CCSDt' and CCSDt calculations with T3 amplitudes in an
active space.
"""

import numpy as np
from pyscf import gto, scf

from ebcc import REBCC, Space

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

# Run a RHF calculation using PySCF
mf = scf.RHF(mol)
mf.kernel()

# Define the occupied, frozen, and active spaces
occupied = mf.mo_occ > 0
frozen = np.zeros_like(occupied)
active = np.zeros_like(occupied)
active[mol.nelectron // 2 - 1] = True  # HOMO
active[mol.nelectron // 2] = True      # LUMO
space = Space(
    occupied,
    frozen,
    active,
)

# Run a CCSDt' calculation
ccsdt = REBCC(mf, ansatz="CCSDt'", space=space)
ccsdt.kernel()

# Run a CCSDt calculation
ccsdt = REBCC(mf, ansatz="CCSDt", space=space)
ccsdt.kernel()
