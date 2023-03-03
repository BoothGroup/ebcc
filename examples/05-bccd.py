import numpy as np
from pyscf import gto, scf

from ebcc import EBCC, brueckner

mol = gto.Mole()
mol.atom = "H 0 0 0; Li 0 0 1.64"
mol.basis = "cc-pvdz"
mol.build()

mf = scf.RHF(mol)
mf.kernel()

ccsd = EBCC(mf, ansatz="CCSD")
bccd = ccsd.brueckner()

assert np.allclose(ccsd.t1, 0)
