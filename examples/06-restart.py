"""
Example of saving and restarting an EBCC calculation.
"""

import os
import numpy as np
from pyscf import gto, scf

from ebcc import REBCC

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

# Run a RHF calculation using PySCF
mf = scf.RHF(mol)
mf.kernel()

# Run a CC3 calculation that does not converge
cc3 = REBCC(mf, ansatz="CC3")
cc3.options.max_iter = 5
cc3.kernel()

# Save the calculation to a file
cc3.write("restart.h5")

# Load the calculation from the file
cc3 = REBCC.read("restart.h5")

# Run the calculation again, but this time with a higher max_iter
cc3.options.max_iter = 20
cc3.kernel()

# Delete the file
os.remove("restart.h5")
