import numpy as np
from pyscf import gto, scf

from ebcc import UEBCC

mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

mf = scf.UHF(mol)
mf.kernel()

ccsd = UEBCC(mf)
ccsd.kernel()

eom = ccsd.ip_eom()
eom.kernel()
