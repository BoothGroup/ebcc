'''
Compute RDMs from ebcc, in both shifted and unshifted boson representations 
'''

import numpy as np
import pyscf
from pyscf import ao2mo
from ebcc import ebccsd, utils

mol = pyscf.M(
    atom = [
    ['O' , (0. , 0.     , 0.)],
    ['H' , (0. , -0.757 , 0.587)],
    ['H' , (0. , 0.757  , 0.587)]],
    basis = 'cc-pvdz')
mf = mol.RHF().run()
nao = mf.mo_coeff.shape[1]
nbos = 5
np.random.seed(92)
gmat = np.random.random((nbos,nao,nao)) * 0.01
omega = 0.1+np.random.random((nbos)) * 10 
eri = ao2mo.restore(1, mf._eri, nao)

for rank in [(2,1,1)]:
    for shift in [True, False]:
        cc = ebccsd.EBCCSD(mol, mf, eri, options={'tthresh': 1e-8}, rank=rank, omega=omega, gmat=gmat, shift=shift, autogen_code=True)
        etot, e_corr = cc.kernel()
        print('EBCCSD correlation energy for rank {} and shift {}:   {}'.format(rank,shift,cc.e_corr))
        print('EBCCSD total energy', etot)
        
        cc.solve_lambda()
        
        # Create RDMs with respect to the original bosonic operators (no shift included) 
        dm1_eb = cc.make_1rdm_f()
        dm2_eb = cc.make_2rdm_f()
        dm1_bb = cc.make_1rdm_b()
        dm1_coup_cre, dm1_coup_ann = cc.make_eb_coup_rdm()
        assert(np.allclose(dm1_coup_cre,dm1_coup_ann.transpose((0,2,1))))
        dm1_b_sing_cre, dm1_b_sing_ann = cc.make_sing_b_dm()

        # We can also compute the bosonic operators in the basis where they have
        # been shifted to remove coupling to the static mean-field density
        dm1_bb = cc.make_1rdm_b(unshifted_bos=False)
        dm1_coup_cre, dm1_coup_ann = cc.make_eb_coup_rdm(unshifted_bos=False)
        assert(np.allclose(dm1_coup_cre,dm1_coup_ann.transpose((0,2,1))))
