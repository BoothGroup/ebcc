"""
Example of an ecCC calculation with T3 and T4 calculated externally.
"""

import numpy as np
from pyscf import gto, scf, fci, ao2mo

from ebcc import REBCC, Space
from ebcc.ext.fci import extract_amplitudes_restricted

# Define the molecule using PySCF
mol = gto.Mole()
mol.atom = "N 0 0 0; N 0 0 1.2"
mol.basis = "cc-pvdz"
mol.build()

# Run a RHF calculation using PySCF
mf = scf.RHF(mol)
mf.kernel()

# Define the occupied, frozen, and active spaces
occupied = mf.mo_occ > 0
frozen = np.zeros_like(occupied)
active = np.zeros_like(occupied)
nocc = np.sum(occupied)
active[nocc - 4 : nocc + 4] = True  # First four HOMOs and LUMOs
space = Space(
    occupied,
    frozen,
    active,
)

# Run an FCI calculation in the active space
mo = mo=mf.mo_coeff[:, space.active]
h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mo, mo)
h2e = ao2mo.kernel(mf._eri, mo, compact=False).reshape((mo.shape[-1],) * 4)
ci = fci.direct_spin1.FCI()
ci.kernel(h1e, h2e, space.nact, space.naocc * 2)

# Extract the amplitudes from the FCI calculation
amplitudes = extract_amplitudes_restricted(ci, space)

# Run an ecCC calculation
eccc = REBCC(mf, ansatz="CCSD", space=space)
eccc.external_correction(amplitudes, mixed_term_strategy="update")

# Alternatively, we can use tCC
tcc = REBCC(mf, ansatz="CCSD", space=space)
tcc.tailor(amplitudes)
