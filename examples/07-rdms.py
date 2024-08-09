"""
Example obtaining RDMs from EBCC.
"""

import numpy as np
from pyscf import gto, scf, lib

from ebcc import GEBCC

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

# Run a RHF calculation using PySCF
mf = scf.RHF(mol)
mf.kernel()

# Convert the RHF object to a GHF object (not necessary for density
# matrices, just simplifies the Hamiltonian)
mf = mf.to_ghf()

# Run a CCSD calculation
ccsd = GEBCC(mf, ansatz="CCSD")
ccsd.kernel()

# If the Λ amplitudes are not solved, EBCC will use the approximation
# Λ = T* and warn the user.
ccsd.solve_lambda()

# Fermionic RDMs
dm1 = ccsd.make_rdm1_f()
dm2 = ccsd.make_rdm2_f()

# Compare the energies
h1 = np.linalg.multi_dot((mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
h2 = ccsd.get_eris().array
e_rdm = lib.einsum("pq,qp->", h1, dm1)
e_rdm += lib.einsum("pqrs,pqrs->", h2, dm2) * 0.5
e_rdm += mol.energy_nuc()
assert np.allclose(e_rdm, ccsd.e_tot)
