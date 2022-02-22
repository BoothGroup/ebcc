import numpy as np
import pyscf
from pyscf import ao2mo
from ebcc import ebccsd

mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = mol.RHF().run()

mycc = mf.CCSD().run()
print('CCSD correlation energy', mycc.e_corr)

# Get integrals

cc = ebccsd.EBCCSD.fromUHFobj(mf, options={'diis space': 12}, autogen_code=True)
ecorr = cc.kernel()
print('EBCCSD correlation energy', cc.e_corr)
if np.allclose(cc.e_corr,mycc.e_corr):
    print('**********************************************************************')
    print('EXCELLENT: CCSD correlation energies agree between pyscf and ebcc')
    print('**********************************************************************')
else:
    print('pyscf and ebcc correlation energies do not agree...')
