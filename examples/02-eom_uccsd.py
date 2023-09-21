"""
Example of a CCSD calculation using a UHF reference and a subsequent
EOM-CCSD calculation for the ionization potential.
"""

import numpy as np
from pyscf import gto, scf

from ebcc import UEBCC

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

# Run a UHF calculation using PySCF
mf = scf.UHF(mol)
mf.kernel()

# Run a UCCSD calculation
ccsd = UEBCC(mf, ansatz="CCSD")
ccsd.kernel()

# Run an EOM-CCSD calculation
eom = ccsd.ip_eom()
eom.kernel()
