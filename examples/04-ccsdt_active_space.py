import numpy as np
from pyscf import gto, scf

from ebcc import REBCC, Space

mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

mf = scf.RHF(mol)
mf.kernel()

frozen = np.zeros_like(mf.mo_occ, dtype=bool)
active = np.zeros_like(mf.mo_occ, dtype=bool)
active[mol.nelectron // 2 - 1] = True  # HOMO
active[mol.nelectron // 2] = True      # LUMO
space = Space(
    mf.mo_occ > 0,
    frozen,
    active,
)

ccsdt = REBCC(mf, ansatz="CCSDt'", space=space)
ccsdt.kernel()
