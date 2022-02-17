''' Compute the EOM dd spectral moments at CCSD level. 
'''
import numpy as np
import pyscf
import itertools
import sys
from pyscf import gto, scf, cc, ao2mo
from ebcc import ebccsd, ccsd_equations

mol = pyscf.M(
    atom = 'Li 0 0 0; H 0 0 1.5',
    basis = 'sto-3g')
mf = mol.RHF().run()
pyscf_cc = mf.CCSD().run()

eri = ao2mo.restore(1, mf._eri, mf.mol.nao_nr())

# Set up cc object and run kernel for T-amplitude optimization (no bosons)
mycc = ebccsd.EBCCSD(mol, mf, eri, options={'tthresh': 1e-10, 'diis space': 12}, autogen_code=True)
etot, ecorr = mycc.kernel()
assert(np.isclose(ecorr, pyscf_cc.e_corr))
mycc.solve_lambda()

# Generate 1 and 2 RDMs
dm1 = mycc.make_1rdm_f()
# Note that dm2[p,q,r,s] = <p^+ r^+ s q>
dm2 = mycc.make_2rdm_f()

# EOM computation of dd spectral moments
# dd moments as dd_moms[p,q,r,s,n] = <p^+ q (H-E)^n r^+ s>
dd_moms = mycc.make_dd_EOM_moms(1)
dd_moms_with_ref_proj = ccsd_equations.dd_moms_eom(mycc, 1, include_ref_proj=True)

### TESTS and CHECKS
nocc = mycc.na + mycc.nb 
nvir = mycc.va + mycc.vb
ntot = dm2.shape[0]

# 1. Check agreement between zeroth dd moment and 2RDM (should at least be exact for 2-electrons)
# First, turn zeroth-order dd moment into normal ordered form of dm2 (is this correct?)
moms_dm2 = dd_moms[:,:,:,:,0]
moms_dm2 -= np.einsum('qr,ps->pqrs',np.eye(dm2.shape[0]),dm1)
print('Is full matrix in agreement: ',np.allclose(moms_dm2,dm2))
print('Testing vvvv agreement: ',np.allclose(moms_dm2[nocc:,nocc:,nocc:,nocc:],dm2[nocc:,nocc:,nocc:,nocc:]))
print('Testing ovvv agreement: ',np.allclose(moms_dm2[:nocc,nocc:,nocc:,nocc:],dm2[:nocc,nocc:,nocc:,nocc:]))
print('Testing vvvo agreement: ',np.allclose(moms_dm2[nocc:,nocc:,nocc:,:nocc],dm2[nocc:,nocc:,nocc:,:nocc]))
print('Testing vvov agreement: ',np.allclose(moms_dm2[nocc:,nocc:,:nocc,nocc:],dm2[nocc:,nocc:,:nocc,nocc:]))
print('Testing vovv agreement: ',np.allclose(moms_dm2[nocc:,:nocc,nocc:,nocc:],dm2[nocc:,:nocc,nocc:,nocc:]))
print('Testing oooo agreement: ',np.allclose(moms_dm2[:nocc,:nocc,:nocc,:nocc],dm2[:nocc,:nocc,:nocc,:nocc]))
print('***')
print('Writing out non-zero elements: ')
print('My dd oooo block: ')
for i,j,k,l in itertools.product(range(nocc),range(nocc),range(nocc),range(nocc)):
    if abs(moms_dm2[i,j,k,l]) > 1.e-12:
        print(i,j,k,l,moms_dm2[i,j,k,l])
#print(dd_moms[:nocc,:nocc,:nocc,:nocc,0])
print('***')
print('2dm oooo block: ')
for i,j,k,l in itertools.product(range(nocc),range(nocc),range(nocc),range(nocc)):
    if abs(dm2[i,j,k,l]) > 1.e-12:
        print(i,j,k,l,dm2[i,j,k,l])
#print(dm2_singexcits_[:nocc,:nocc,:nocc,:nocc])
print('****')
print('1dm oo block: ')
print(dm1[:nocc,:nocc])

print('')
print('Now testing including the reference state in the EOM projector...')
moms_dm2 = dd_moms_with_ref_proj[:,:,:,:,0]
moms_dm2 -= np.einsum('qr,ps->pqrs',np.eye(dm2.shape[0]),dm1)
print('Is full matrix in agreement: ',np.allclose(moms_dm2,dm2))
print('Testing vvvv agreement: ',np.allclose(moms_dm2[nocc:,nocc:,nocc:,nocc:],dm2[nocc:,nocc:,nocc:,nocc:]))
print('Testing ovvv agreement: ',np.allclose(moms_dm2[:nocc,nocc:,nocc:,nocc:],dm2[:nocc,nocc:,nocc:,nocc:]))
print('Testing vvvo agreement: ',np.allclose(moms_dm2[nocc:,nocc:,nocc:,:nocc],dm2[nocc:,nocc:,nocc:,:nocc]))
print('Testing vvov agreement: ',np.allclose(moms_dm2[nocc:,nocc:,:nocc,nocc:],dm2[nocc:,nocc:,:nocc,nocc:]))
print('Testing vovv agreement: ',np.allclose(moms_dm2[nocc:,:nocc,nocc:,nocc:],dm2[nocc:,:nocc,nocc:,nocc:]))
print('Testing oovv agreement: ',np.allclose(moms_dm2[:nocc,:nocc,nocc:,nocc:],dm2[:nocc,:nocc,nocc:,nocc:]))
print('Testing ovov agreement: ',np.allclose(moms_dm2[:nocc,nocc:,:nocc,nocc:],dm2[:nocc,nocc:,:nocc,nocc:]))
print('Testing ovvo agreement: ',np.allclose(moms_dm2[:nocc,nocc:,nocc:,:nocc],dm2[:nocc,nocc:,nocc:,:nocc]))
print('Testing voov agreement: ',np.allclose(moms_dm2[nocc:,:nocc,:nocc,nocc:],dm2[nocc:,:nocc,:nocc,nocc:]))
print('Testing vovo agreement: ',np.allclose(moms_dm2[nocc:,:nocc,nocc:,:nocc],dm2[nocc:,:nocc,nocc:,:nocc]))
print('Testing vvoo agreement: ',np.allclose(moms_dm2[nocc:,nocc:,:nocc,:nocc],dm2[nocc:,nocc:,:nocc,:nocc]))
print('Testing vooo agreement: ',np.allclose(moms_dm2[nocc:,:nocc,:nocc,:nocc],dm2[nocc:,:nocc,:nocc,:nocc]))
print('Testing ovoo agreement: ',np.allclose(moms_dm2[:nocc,nocc:,:nocc,:nocc],dm2[:nocc,nocc:,:nocc,:nocc]))
print('Testing oovo agreement: ',np.allclose(moms_dm2[:nocc,:nocc,nocc:,:nocc],dm2[:nocc,:nocc,nocc:,:nocc]))
print('Testing ooov agreement: ',np.allclose(moms_dm2[:nocc,:nocc,:nocc,nocc:],dm2[:nocc,:nocc,:nocc,nocc:]))
print('Testing oooo agreement: ',np.allclose(moms_dm2[:nocc,:nocc,:nocc,:nocc],dm2[:nocc,:nocc,:nocc,:nocc]))
print('***')
print('Writing out non-zero elements: ')
print('My dd oooo block: ')
for i,j,k,l in itertools.product(range(nocc),range(nocc),range(nocc),range(nocc)):
    if abs(moms_dm2[i,j,k,l]) > 1.e-12:
        print(i,j,k,l,moms_dm2[i,j,k,l])
#print(dd_moms[:nocc,:nocc,:nocc,:nocc,0])
print('***')
print('2dm oooo block: ')
for i,j,k,l in itertools.product(range(nocc),range(nocc),range(nocc),range(nocc)):
    if abs(dm2[i,j,k,l]) > 1.e-12:
        print(i,j,k,l,dm2[i,j,k,l])
#print(dm2_singexcits_[:nocc,:nocc,:nocc,:nocc])
print('****')
print('1dm oo block: ')
print(dm1[:nocc,:nocc])

if (nocc*nvir)**2 < 3000:
    # Testing full EOM hamiltonian roots compared to pyscf
    print('Completely diagonalizing Hbar-E_cc to get all excitation energies, in space of singles and doubles...')
    full_eom_h = ccsd_equations.gen_dd_eom_matrix(mycc)
    e = np.linalg.eigvals(full_eom_h)
    e_sorted = np.sort(e.real)
    print('First positive 30 eigenvalues from diagonalization of effective EOM hamiltonian: ')
    e_sorted_pos = e_sorted[e_sorted > 1.e-12]
    print(e_sorted_pos[:30])
    print('Roots from pyscf ee-eom')
    eee, cee = pyscf_cc.eeccsd(nroots=50)
    print(eee[:30])
    print('')
    print('Completely diagonalizing Hbar-E_cc to get all excitation energies, in space of reference, singles and doubles...')
    full_eom_h = ccsd_equations.gen_dd_eom_matrix(mycc, include_ref_proj=True)
    #print('First row and column of Hbar: ')
    #print(full_eom_h[:,0])
    #print('***')
    #print(full_eom_h[0,:])
    #e = np.linalg.eigvals(full_eom_h)
    e_sorted = np.sort(e.real)
    print('First positive 30 eigenvalues from diagonalization of effective EOM hamiltonian: ')
    e_sorted_pos = e_sorted[e_sorted > 1.e-12]
    #e_sorted_pos = e_sorted
    print(e_sorted_pos[:30])
    print('Roots from pyscf ee-eom')
    eee, cee = pyscf_cc.eeccsd(nroots=50)
    print(eee[:30])
