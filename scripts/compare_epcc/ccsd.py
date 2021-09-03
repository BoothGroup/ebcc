import numpy as np
import pyscf
from pyscf import ao2mo
import driver_utils
from ebcc import ebccsd
from epcc import ccsd as epcc_ccsd

mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = mol.RHF().run()

mycc = mf.CCSD().run()
print('CCSD correlation energy', mycc.e_corr)

print('FINISHED PYSCF. Running ebcc...')

# Get integrals
eri = ao2mo.restore(1, mf._eri, mf.mol.nao_nr())
#eri = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape([mf.mol.nao_nr(),]*4)
cc = ebccsd.EBCCSD(mol, mf, eri, autogen_code=True)
etot, ecorr = cc.kernel()
print('EBCCSD correlation energy', cc.e_corr)
if np.allclose(cc.e_corr,mycc.e_corr):
    print('**********************************************************************')
    print('EXCELLENT: CCSD correlation energies agree between pyscf and ebcc')
    print('**********************************************************************')
else:
    print('pyscf and ebcc correlation energies do not agree...')
    1./0

print('Running epcc...')

mod = driver_utils.FAKE_EPCC_MODEL(mol, mf, eri)
options = {"ethresh":1e-8, 'tthresh':1e-7, 'max_iter':500, 'damp':0.4}
ecc, e_cor, t1, t2 = epcc_ccsd.ccsd(mod, options)
print(ecc)

if np.allclose(cc.e_corr, e_cor):
    print('**********************************************************************')
    print('EXCELLENT: CCSD correlation energies agree between epcc and ebcc')
    print('**********************************************************************')
else:
    print('epcc and ebcc correlation energies do not agree...')

#mod = Model(mol, 'rhf', xc)
#ecc, e_cor = epccsd_2_s2(mod, options)
#print(ecc)

