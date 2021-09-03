import numpy as np
import math
import pyscf
from pyscf import ao2mo, gto, scf
import driver_utils
from ebcc import ebccsd
from epcc import epcc as epcc_ccsd
from epcc import hh_model
from epcc import fci

# Set up fake Hubbard-Holstein model
U = 1.
L = 4 
w = 1.5
g = np.sqrt(0.15)

mol = gto.M()
mol.nelectron = L   # Half filling
mf = scf.RHF(mol)
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
mf.kernel()

print('Running ebcc...')
#eri = ao2mo.incore.full(eri, mf.mo_coeff).reshape([L,]*4)
cc = ebccsd.EBCCSD(mol, mf, eri, rank=(2,2,2), omega=omega, gmat=gmat, shift=True, autogen_code=False)
etot, e_corr = cc.kernel()
print('EBCCSD correlation energy', cc.e_corr)
print('EBCCSD total energy', etot)

print('Running epcc via HH functionality...')
# Give it the same orbitals
hh_mod = hh_model.HHModel(L, L, L, U, w, g=g, bc='p', ca=mf.mo_coeff, cb=mf.mo_coeff, gij=None, breaksym=False)
# Note that even though this doesn't break symmetry in the initial guess, it is a UHF class
options = {"ethresh" : 1e-8,"tthresh" : 1e-7, "max_iter" : 500, "damp" : 0.3}
ecc, e_cor = epcc_ccsd.epccsd_2_s2(hh_mod, options)
e,c = hh_mod.fci(nphonon=0)
print('FCI energy with 0 boson occupation: {}'.format(e))
e,c = hh_mod.fci(nphonon=1)
print('FCI energy with 1 boson occupation: {}'.format(e))
e,c = hh_mod.fci(nphonon=2)
print('FCI energy with 2 boson occupation: {}'.format(e))
e,c = hh_mod.fci(nphonon=3)
print('FCI energy with 3 boson occupation: {}'.format(e))
e,c = hh_mod.fci(nphonon=4)
print('FCI energy with 4 boson occupation: {}'.format(e))
if np.allclose(e_cor,cc.e_corr):
    print('**********************************************************************')
    print('EXCELLENT: CCSD correlation energies agree between epcc and ebcc via HH model')
    print('**********************************************************************')
else:
    print('pyscf and ebcc correlation energies do not agree...')

print('Running epcc via fake ab initio model...')
mod = driver_utils.FAKE_EPCC_MODEL(mol, mf, eri, omega=omega, gmat=gmat, shift=True)
options = {"ethresh":1e-8, 'tthresh':1e-7, 'max_iter':500, 'damp':0.4}
ecc, e_cor = epcc_ccsd.epccsd_2_s2(mod, options)
if np.allclose(cc.e_corr, e_cor):
    print('**********************************************************************')
    print('EXCELLENT: CCSD correlation energies agree between epcc and ebcc via fake ab initio model')
    print('**********************************************************************')
else:
    print('epcc and ebcc correlation energies do not agree...')
