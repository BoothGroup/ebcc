import numpy as np
import math
import pyscf
from pyscf import ao2mo, gto, scf
import driver_utils
from ebcc import ebccsd
from epcc import ccsd as epcc_ccsd
from epcc import epcc as epcc_epccsd
from epcc import hh_model
from epcc import fci

# Set up ab initio system with random e-b couplings
mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
    basis = '6-31G',
    symmetry = True,
    verbose=1
)

myhf = mol.RHF().run()
nmo = myhf.mo_coeff.shape[1]
nbos = 5
gmat = np.random.random((nbos,nmo,nmo)) * 0.02
omega = np.random.random((nbos)) * 5.

print('Running ebcc...')
#eri = ao2mo.kernel(mol, myhf.mo_coeff, compact=False).reshape([nmo,]*4)
eri = ao2mo.restore(1, myhf._eri, nmo)
options = {"ethresh":1e-9, 'tthresh':1e-8, 'max_iter':500, 'damp':0.3}

for rank in [(2,0,0), (2,1,1), (2,2,1), (2,2,2)]:
    for shift in [True, False]:
        cc = ebccsd.EBCCSD(mol, myhf, eri, options=options, rank=rank, omega=omega, gmat=gmat, shift=shift, autogen_code=False)
        etot, e_corr = cc.kernel()
        print('EBCCSD correlation energy for rank {} and shift {}:   {}'.format(rank,shift,cc.e_corr))
        print('EBCCSD total energy', etot)

        cc_auto = ebccsd.EBCCSD(mol, myhf, eri, options=options, rank=rank, omega=omega, gmat=gmat, shift=shift, autogen_code=True)
        etot, e_corr = cc_auto.kernel()
        print('EBCCSD correlation energy with autogenerated code for rank {} and shift {}:   {}'.format(rank,shift,cc_auto.e_corr))
        print('EBCCSD total energy', etot)
        assert(np.allclose(cc.e_corr, e_corr))

        print('Running epcc via fake ab initio model...')
        mod = driver_utils.FAKE_EPCC_MODEL(mol, myhf, eri, omega=omega, gmat=gmat, shift=shift)
        if rank == (2,0,0):
            mod = driver_utils.FAKE_EPCC_MODEL(mol, myhf, eri, omega=None, gmat=None, shift=shift)
            ecc, e_cor, t1, t2 = epcc_ccsd.ccsd(mod, options)
        elif rank == (2,1,1):
            ecc, e_cor = epcc_epccsd.epcc(mod, options)
        elif rank == (2,2,1):
            ecc, e_cor = epcc_epccsd.epccsd_2_s1(mod, options)
        else:
            ecc, e_cor = epcc_epccsd.epccsd_2_s2(mod, options)
        assert(np.allclose(cc.e_corr, e_cor))
