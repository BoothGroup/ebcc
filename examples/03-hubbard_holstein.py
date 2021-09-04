import numpy as np
import math
import pyscf
from pyscf import ao2mo, gto, scf
from ebcc import ebccsd

# Set up fake 4-site Hubbard-Holstein model
U = 1.
L = 4 
w = 2.5
g = np.sqrt(0.15)

mol = gto.M()
mol.nelectron = L   # Half filling
mf = scf.RHF(mol)
h1 = np.zeros((L,L))
for i in range(L-1):
    h1[i,i+1] = h1[i+1,i] = -1.0
h1[L-1,0] = h1[0,L-1] = 1.0  # APBC

eri = np.zeros((L,L,L,L))
gmat = np.zeros((L, L, L))  # nbosons, spinorbs, spinorbs
omega = np.zeros((L))
for i in range(L):
    eri[i,i,i,i] = U
    gmat[i,i,i] = g  # Local coupling
omega.fill(w)
mf.get_hcore = lambda *args: h1
mf.get_ovlp = lambda *args: np.eye(L)
mf._eri = ao2mo.restore(8, eri, L)
mol.incore_anyway = True
mf.kernel()

cc = ebccsd.EBCCSD(mol, mf, eri, options={'diis space': 8}, rank=(2,2,2), omega=omega, gmat=gmat, shift=True)
etot, e_corr = cc.kernel()
print('EBCCSD correlation energy', cc.e_corr)
print('EBCCSD total energy', etot)
