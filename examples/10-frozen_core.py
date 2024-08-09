"""
Example of running a CCSD calculation with frozen core orbitals.
"""

import numpy as np
from pyscf import gto, scf, lib

from ebcc import REBCC, Space

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

# Run a RHF calculation using PySCF
mf = scf.RHF(mol)
mf.kernel()

# Define the occupied, frozen, and active spaces. In `ebcc`, the active
# space refers to orbitals that are active within a correlated regime,
# frozen refers to orbitals that are not correlated, and any orbitals
# that are neither frozen nor active are considered correlated. The
# active space is reserved for methods such as CCSDt' (see example 4).
# For more details see `ebcc.space`.
occupied = mf.mo_occ > 0
frozen = np.zeros_like(occupied)
active = np.zeros_like(occupied)
frozen[:2] = True  # Freeze the two lowest energy orbitals
space = Space(occupied, frozen, active)

# Run a CCSD calculation, first without the frozen core
ccsd = REBCC(mf, ansatz="CCSD")
ccsd.kernel()

# Now run a CCSD calculation with the frozen core
ccsd_froz = REBCC(mf, ansatz="CCSD", space=space)
ccsd_froz.kernel()
