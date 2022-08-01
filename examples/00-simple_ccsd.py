import numpy as np
from pyscf import gto, scf
from ebcc import EBCC

mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

mf = scf.RHF(mol)
mf.kernel()

ccsd = EBCC(mf)
ccsd.kernel()
print("CCSD correlation energy:", ccsd.e_corr)
