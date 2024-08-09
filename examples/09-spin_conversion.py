"""
Example of converting restricted and unrestricted EBCC calculations
to spin-orbital (GHF) EBCC calculations.
"""

import numpy as np
from pyscf import gto, scf

from ebcc import REBCC, UEBCC, GEBCC

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

# Run a RHF calculation using PySCF
rhf = scf.RHF(mol)
rhf.kernel()

# Run a REBCC calculation
rcc = REBCC(rhf, ansatz="QCISD")
rcc.kernel()

# Convert to unrestricted and run kernel - unless the UEBCC solution
# breaks some symmetry this should converge immediately to the same
# solution as the REBCC calculation.
uebcc_from_rebcc = UEBCC.from_rebcc(rcc)
uebcc_from_rebcc.kernel()

# Conversion of REBCC to GEBCC goes via a UEBCC intermediate, here
# we just convert the UEBCC object we just created. Once again, in
# the absence of symmetry breaking this should converge immediately
# to the same solution as the REBCC calculation.
gebcc_from_uebcc = GEBCC.from_uebcc(uebcc_from_rebcc)
gebcc_from_uebcc.kernel()
