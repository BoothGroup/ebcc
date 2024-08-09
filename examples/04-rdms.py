import numpy as np
from pyscf import gto, scf

from ebcc import REBCC

mol = gto.Mole()
mol.atom = "H 0 0 0; F 0 0 1.1"
mol.basis = "cc-pvdz"
mol.build()

mf = scf.RHF(mol)
mf.kernel()

ccsd = REBCC(mf)
ccsd.kernel()
ccsd.solve_lambda()

rdm1 = ccsd.make_rdm1_f()
rdm2 = ccsd.make_rdm2_f()

h1e = np.einsum("pq,pi,qj->ij", mf.get_hcore(), mf.mo_coeff, mf.mo_coeff)
h2e = ccsd.get_eris().xxxx

e_rdm1 =  np.einsum("pq,pq->", rdm1, h1e)
e_rdm2 =  np.einsum("pqrs,pqrs->", rdm2, h2e) / 2

print("E(rdm1) = %16.10f" % e_rdm1)
print("E(rdm2) = %16.10f" % e_rdm2)
print("E(tot)  = %16.10f" % (e_rdm1 + e_rdm2 + mol.energy_nuc()))
