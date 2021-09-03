"""Module for slow, exact diagonalisation-based coupled electron-boson FCI code.
Based on the fci_slow.py code within pyscf.
"""


import numpy
from pyscf import tools
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf.fci import rdm
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci import fci_slow
from pyscf.fci import direct_ep


def contract_all(h1e, g2e, hep, hpp, ci0, nelec, norbs, nbosons, max_occ,
                ecore=0.0, adj_zero_pho=False):
    #ci1  = contract_1e(h1e, ci0, nelec, norbs, nbosons, max_occ)
    contrib1 = contract_2e(g2e, ci0, nelec, norbs, nbosons, max_occ)
    incbosons = (nbosons > 0 and max_occ > 0)

    if incbosons:
        contrib2 = contract_ep(hep, ci0, nelec, norbs, nbosons, max_occ,
                    adj_zero_pho=adj_zero_pho)
        contrib3 = contract_pp(hpp, ci0, nelec, norbs, nbosons, max_occ)

    cishape = make_shape(nelec, norbs, nbosons, max_occ)

    #print("1+2-body")
    #print(contrib1.reshape(cishape))
    #print("electron-phonon coupling")
    #print(contrib2.reshape(cishape))
    #print("phonon-phonon coupling")
    #print(contrib3.reshape(cishape))
    if incbosons:
        return contrib1 + contrib2 + contrib3
    else:
        return contrib1


def make_shape(nelec, norbs, nbosons, max_occ):
    """Construct the shape of a single FCI vector in the coupled electron-boson space.
    """
    neleca, nelecb = _unpack_nelec(nelec)
    na = cistring.num_strings(norbs, neleca)
    nb = cistring.num_strings(norbs, nelecb)
    return (na,nb)+(max_occ+1,)*nbosons

# Contract 1-electron integrals with fcivec.
def contract_1e(h1e, fcivec,  norb, nelec, nbosons, max_occ):
    raise NotImplementedError("1 electron contraction is currently"
                        "bugged for coupled electron-boson systems."
                        "This should instead be folded into a two-body operator.")
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

    cishape = make_shape(nelec, norb, nbosons, max_occ)

    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros(cishape, dtype=fcivec.dtype)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * ci0[str0] * h1e[a,i]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:,str1] += sign * ci0[:,str0] * h1e[a,i]
    return fcinew.reshape(fcivec.shape)

# Contract 2-electron integrals with fcivec.
def contract_2e(eri, fcivec, norb, nelec, nbosons, max_occ):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

    cishape = make_shape(nelec, norb, nbosons, max_occ)

    ci0 = fcivec.reshape(cishape)
    t1 = numpy.zeros((norb,norb)+cishape, dtype=fcivec.dtype)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1[a,i,:,str1] += sign * ci0[:,str0]

    t1 = lib.einsum('bjai,aiAB...->bjAB...', eri.reshape([norb]*4), t1)

    fcinew = numpy.zeros_like(ci0)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * t1[a,i,str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:,str1] += sign * t1[a,i,:,str0]
    return fcinew.reshape(fcivec.shape)

# Contract electron-phonon portion of the Hamiltonian.
def contract_ep(g, fcivec, norb, nelec, nbosons, max_occ, adj_zero_pho=False):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

    cishape = make_shape(nelec, norb, nbosons, max_occ)

    ci0 = fcivec.reshape(cishape)
    t1 = numpy.zeros((norb,norb)+cishape, dtype=fcivec.dtype)

    if adj_zero_pho:
        zfac = float(neleca+nelecb) / norb
        #print("Zfac=",zfac)
        adj_val = zfac * ci0
        for i in range(norb):
            t1[i,i] -= adj_val

    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1[a,i,str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1[a,i,:,str1] += sign * ci0[:,str0]
    # Now have contribution to a particular state via given excitation
    # channel; just need to apply bosonic (de)excitations.
    # Note that while the contribution {a^{+} i p} and {i a^{+} p^{+}}
    # are related via the Hermiticity of the Hamiltonian, no other properties
    # are guaranteed, so we cannot write this as p + p^{+}.

    # Contract intermediate with the electron-boson coupling.
    # If we need to remove zero phonon mode also need factor of -<N>

    # First, bosonic excitations.
    tex = numpy.einsum("nia,ia...->n...", g, t1)
    # Then bosonic deexcitations.
    tdex = numpy.einsum("nai,ia...->n...", g, t1)

    #print(norb,nelec, nbosons)
    #print(tex.shape,"Ex:",numpy.sum(tex**2))
    #print(tex)
    #print(tdex.shape, "Deex:",numpy.sum(tdex**2))
    #print(tdex)
    # The leading index tells us which bosonic degree of freedom is coupled
    # to in each case.
    fcinew = numpy.zeros_like(ci0)

    bos_cre = numpy.sqrt(numpy.arange(1,max_occ+1))

    for ibos in range(nbosons):
        for iocc in range(0, max_occ):
            ex_slice = slices_for_cre(ibos, nbosons, iocc)
            norm_slice = slices_for(ibos, nbosons, iocc)
            # NB bos_cre[iocc] = sqrt(iocc+1)
            fcinew[ex_slice] += tex[ibos][norm_slice] * bos_cre[iocc]

        for iocc in range(1, max_occ+1):
            dex_slice = slices_for_des(ibos, nbosons, iocc)
            norm_slice = slices_for(ibos, nbosons, iocc)
            # NB bos_cre[iocc] = sqrt(iocc+1)
            fcinew[dex_slice] += tdex[ibos][norm_slice] * bos_cre[iocc-1]

    return fcinew.reshape(fcivec.shape)

# Contract phonon-phonon portion of the Hamiltonian.

def contract_pp(hpp, fcivec, norb, nelec, nbosons, max_occ):
    """Arbitrary phonon-phonon coupling.
    """
    cishape = make_shape(nelec, norb, nbosons, max_occ)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros_like(ci0)

    phonon_cre = numpy.sqrt(numpy.arange(1,max_occ+1))
    t1 = numpy.zeros((nbosons,)+cishape, dtype=fcivec.dtype)
    for psite_id in range(nbosons):
        for i in range(max_occ):
            slices1 = slices_for_cre(psite_id, nbosons, i)
            slices0 = slices_for    (psite_id, nbosons, i)
            t1[(psite_id,)+slices0] += ci0[slices1] * phonon_cre[i]     # annihilation

    t1 = lib.dot(hpp, t1.reshape(nbosons,-1)).reshape(t1.shape)

    for psite_id in range(nbosons):
        for i in range(max_occ):
            slices1 = slices_for_cre(psite_id, nbosons, i)
            slices0 = slices_for    (psite_id, nbosons, i)
            fcinew[slices1] += t1[(psite_id,)+slices0] * phonon_cre[i]  # creation
    return fcinew.reshape(fcivec.shape)

def contract_pp_for_future(hpp, fcivec, norb, nelec, nbosons, max_occ):
    """Our bosons are decoupled; only have diagonal couplings,
    ie. to the boson number.
    """
    cishape = make_shape(nelec, norb, nbosons, max_occ)
    ci0 = fcivec.reshape(cishape)
    fcinew = numpy.zeros_like(ci0)

    for ibos in range(nbosons):
        for iocc in range(max_occ+1):
            slice1 = slices_for(ibos, nbosons, iocc)
            # This may need a sign change?
            # Two factors sqrt(iocc) from annihilation then creation.
            fcinew[slice1] += ci0[slice1] * iocc * hpp[ibos]
    return fcinew.reshape(fcivec.shape)

def slices_for(b_id, nbos, occ):
    slices = [slice(None,None,None)] * (2+nbos)  # +2 for electron indices
    slices[2+b_id] = occ
    return tuple(slices)
def slices_for_cre(b_id, nbos, occ):
    return slices_for(b_id, nbos, occ+1)
def slices_for_des(b_id, nbos, occ):
    return slices_for(b_id, nbos, occ-1)

def slices_for_occ_reduction(nbos, new_max_occ):
    slices = [slice(None,None,None)] * 2
    slices += [slice(0,new_max_occ+1)] * nbos    
    return tuple(slices)

def make_hdiag(h1e, g2e, hep, hpp, norb, nelec, nbosons, max_occ):
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    occslista = [tab[:neleca,0] for tab in link_indexa]
    occslistb = [tab[:nelecb,0] for tab in link_indexb]

    nelec_tot = neleca + nelecb

    # electron part
    cishape = make_shape(nelec, norb, nbosons, max_occ)
    hdiag = numpy.zeros(cishape)
    print(cishape)
    g2e = ao2mo.restore(1, g2e, norb)
    diagj = numpy.einsum('iijj->ij',g2e)
    diagk = numpy.einsum('ijji->ij',g2e)
    for ia,aocc in enumerate(occslista):
        for ib,bocc in enumerate(occslistb):
            e1 = h1e[aocc,aocc].sum() + h1e[bocc,bocc].sum()
            e2 = diagj[aocc][:,aocc].sum() + diagj[aocc][:,bocc].sum() \
               + diagj[bocc][:,aocc].sum() + diagj[bocc][:,bocc].sum() \
               - diagk[aocc][:,aocc].sum() - diagk[bocc][:,bocc].sum()
            hdiag[ia,ib] += e1 + e2*.5

    # No electron-phonon part?

    # phonon part
    if len(hpp.shape) == 2:
        hpp = hpp.diagonal()
    for b_id in range(nbosons):
        for i in range(max_occ+1):
            slices0 = slices_for(b_id, nbosons, i)
            hdiag[slices0] += hpp[b_id] * i

    return hdiag.ravel()

def kernel(h1e, g2e, hep, hpp, norb, nelec, nbosons, max_occ,
           tol=1e-9, max_cycle=100, verbose=0, ecore=0,
           returnhop = False, adj_zero_pho = False,
           **kwargs):
    h2e = fci_slow.absorb_h1e(h1e, g2e, norb, nelec, .5)

    cishape = make_shape(nelec, norb, nbosons, max_occ)
    ci0 = numpy.zeros(cishape)
    ci0.__setitem__((0,0) + (0,)*nbosons, 1)
    # Add noise for initial guess, remove it if problematic
    #ci0[0,:] += numpy.random.random(ci0[0,:].shape) * 1e-6
    #ci0[:,0] += numpy.random.random(ci0[:,0].shape) * 1e-6

    def hop(c):
        hc = contract_all(h1e, h2e, hep, hpp, c, norb,
                    nelec, nbosons, max_occ, adj_zero_pho=adj_zero_pho)
        return hc.reshape(-1)

    hdiag = make_hdiag(h1e, g2e, hep, hpp, norb, nelec, nbosons, max_occ)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    if returnhop:
        return hop,ci0,hdiag

    '''
    e, c = lib.davidson(hop, ci0.reshape(-1), precond,
                        tol=tol, max_cycle=max_cycle, verbose=verbose,
                        **kwargs)
    '''
    tol = 1e-10
    e, c = lib.davidson(hop, ci0.reshape(-1), precond,
                        tol=tol, max_cycle=max_cycle, verbose=6,
                        **kwargs)
    return e+ecore, c.reshape(cishape)

# dm_pq = <|p^+ q|>
def make_rdm1e(fcivec, norb, nelec):
    '''1-electron density matrix dm_pq = <|p^+ q|>'''
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    rdm1 = numpy.zeros((norb,norb))
    ci0 = fcivec.reshape(na,-1)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            rdm1[a,i] += sign * numpy.dot(ci0[str1],ci0[str0])

    ci0 = fcivec.reshape(na,nb,-1)
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            rdm1[a,i] += sign * numpy.einsum('ax,ax->', ci0[:,str1],ci0[:,str0])
    return rdm1


def make_rdm12e(fcivec, norb, nelec):
    '''1-electron and 2-electron density matrices
    dm_qp = <|q^+ p|>
    dm_{pqrs} = <|p^+ r^+ s q|>
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    ci0 = fcivec.reshape(na,nb,-1)
    rdm1 = numpy.zeros((norb,norb))
    rdm2 = numpy.zeros((norb,norb,norb,norb))

    for str0 in range(na):
        t1 = numpy.zeros((norb,norb,nb)+ci0.shape[2:])
        for a, i, str1, sign in link_indexa[str0]:
            t1[i,a,:] += sign * ci0[str1,:]

        for k, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                t1[i,a,k] += sign * ci0[str0,str1]

        rdm1 += numpy.einsum('mp,ijmp->ij', ci0[str0], t1)
        # i^+ j|0> => <0|j^+ i, so swap i and j
        #:rdm2 += numpy.einsum('ijmp,klmp->jikl', t1, t1)
        tmp = lib.dot(t1.reshape(norb**2,-1), t1.reshape(norb**2,-1).T)
        rdm2 += tmp.reshape((norb,)*4).transpose(1,0,2,3)
    rdm1, rdm2 = rdm.reorder_rdm(rdm1, rdm2, True)
    return rdm1, rdm2

def make_rdm12e_spinresolved(fcivec, norb, nelec):
    '''1-electron and 2-electron spin-resolved density matrices
    dm_qp = <|q^+ p|>
    dm_{pqrs} = <|p^+ r^+ s q|> 
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
    link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)

    ci0 = fcivec.reshape(na,nb,-1)
    nspinorb = 2*norb
    rdm1 = numpy.zeros((nspinorb,nspinorb))
    rdm2 = numpy.zeros((nspinorb,nspinorb,nspinorb,nspinorb))

    # We'll initially calculate <|p^+ q r^+ s|>, where p&q and r&s are
    # of the same spin, then use this to obtain the components where
    # each individual excitation is not spin-conserving, before using
    # this to finally generate the standard form of the 2rdm.
    for str0 in range(na):
        # Alpha excitation.
        t1 = numpy.zeros((nspinorb,nspinorb,nb)+ci0.shape[2:])
        for a, i, str1, sign in link_indexa[str0]:
            t1[2*i,2*a,:] += sign * ci0[str1,:]
        # Beta excitation.
        for k, tab in enumerate(link_indexb):
            for a, i, str1, sign in tab:
                t1[2*i+1,2*a+1,k] += sign * ci0[str0,str1]
        # Generate our spin-resolved 1rdm contribution.
        rdm1 += numpy.einsum('mp,ijmp->ij', ci0[str0], t1)
        # t1[i,a] = a^+ i |0>
        # i^+ j|0> => <0|j^+ i, so swap i and j
        #:rdm2 += numpy.einsum('ijmp,klmp->jikl', t1, t1)
        # Calc <0|i^+ a b^+ j |0>
        tmp = lib.dot(t1.reshape(nspinorb**2,-1), t1.reshape(nspinorb**2,-1).T)
        rdm2 += tmp.reshape((nspinorb,)*4).transpose(1,0,2,3)
    # Need to fill in components where have two single excitations which are
    # spin diallowed, but in combination excitations conserve spin.
    # From standard fermionic commutation relations
    #   <0|b1^+ a1 a2^+ b2|0> =
    #           - <0|a2^+ a1 b1^+ b2|0> + \delta_{a1a2} <0|b1^+ b2|0>
    rdm2[::2,1::2,1::2,::2] = -numpy.einsum("pqrs->rqps",rdm2[1::2,1::2,::2,::2]) +\
                    numpy.einsum("pq,rs->prsq",rdm1[::2,::2], numpy.identity(norb))
    rdm2[1::2,::2,::2,1::2] = -numpy.einsum("pqrs->rqps",rdm2[::2,::2,1::2,1::2]) +\
                    numpy.einsum("pq,rs->prsq",rdm1[1::2,1::2], numpy.identity(norb))
    # This should hopefully work with spinorbitals.
    rdm1, rdm2 = rdm.reorder_rdm(rdm1, rdm2, True)
    return rdm1, rdm2

def run(mf, hpp, hep, max_occ, returnhop=False, **kwargs):
    """run a calculation using a pyscf mf object.
    """
    from functools import reduce

    norb = mf.mo_coeff.shape[1]
    nelec = mf.mol.nelectron
    nbosons = hpp.shape[0]
    assert(hep.shape == (nbosons, norb, norb))

    h1e = reduce(numpy.dot, (mf.mo_coeff.T, mf.get_hcore(), mf.mo_coeff))
    eri = ao2mo.kernel(mf._eri, mf.mo_coeff, compact=False)
    eri = eri.reshape(norb,norb,norb,norb)
    if returnhop:
        hop0 = kernel(h1e, eri, hep, hpp, norb, nelec, nbosons, max_occ,
                        returnhop=True, **kwargs)

        #hop1 = fci_slow.kernel(h1e, eri, norb, nelec)#, returnhop=True)
        return hop0#, hop1
    res0 = kernel(h1e, eri, hep, hpp, norb, nelec, nbosons, max_occ,
                        **kwargs)

    res1 = fci_slow.kernel(h1e, eri, norb, nelec)
    return res0, res1

def run_hub_test(returnhop = False, **kwargs):
    return run_ep_hubbard(t=1.0, u=1.5, g=0.5, pp=0.1, nsite=2, nelec=2, nphonon=3,
                        returnhop=returnhop)


