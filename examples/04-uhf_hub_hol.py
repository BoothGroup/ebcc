import numpy as np
import math
import pyscf
from pyscf import ao2mo, gto, scf
from ebcc import ebccsd

# Set up fake Hubbard-Holstein model
U = 4.
L = 4 
w = 5.
breaksym = True
g = math.sqrt(0.25) 

mol = gto.M()
mol.nelectron = L   # Half filling
# UHF object
mf = scf.UHF(mol)
h1 = np.zeros((L,L))
for i in range(L-1):
    h1[i,i+1] = h1[i+1,i] = -1.0
h1[L-1,0] = h1[0,L-1] = -1.0  # PBC
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

# Get broken symmetry initial guess for UHF
dma = np.zeros((L,L))
dmb = np.zeros((L,L))
if breaksym:
    for i in range(L):
        if i % 2 == 0:
            dma[i,i] = 1.0
        else:
            dmb[i,i] = 1.0
else:
    for i in range(L): dma[i,i] = dmb[i,i] = 0.5
mf.kernel(dm0 = [dma, dmb])

cc = ebccsd.EBCCSD(mol, mf, eri, options={'diis space': 8}, rank=(2,1,1), omega=omega, gmat=gmat, shift=True, autogen_code=True)
etot, e_corr = cc.kernel()
print('EBCCSD correlation energy for rank 211:', cc.e_corr)
print('EBCCSD total energy', etot)

cc = ebccsd.EBCCSD(mol, mf, eri, options={'diis space': 8}, rank=(2,2,1), omega=omega, gmat=gmat, shift=True, autogen_code=True)
etot, e_corr = cc.kernel()
print('EBCCSD correlation energy for rank 221:', cc.e_corr)
print('EBCCSD total energy', etot)

cc = ebccsd.EBCCSD(mol, mf, eri, options={'diis space': 8}, rank=(2,2,2), omega=omega, gmat=gmat, shift=True, autogen_code=True)
etot, e_corr = cc.kernel()
print('EBCCSD correlation energy for rank 222:', cc.e_corr)
print('EBCCSD total energy', etot)
