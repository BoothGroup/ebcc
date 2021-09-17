import numpy as np
from pyscf import scf

def reorder_to_ghf(cc, mat):
    ''' Orbital dependent expectation values (e.g. RDMs) from ebcc are returned with
    a spinorbital index of occ_a, occ_b, virt_a, virt_b.

    This routine will reorder them to an energy-ordering of the spin-orbitals,
    disregarding spin, so they can be compared to a normal GHF implementation of
    these quantities, i.e. occupied, virtual, in energy/occupation ordering.

    The energy ordering comes from the original cc.mf object'''

    # Note cc.mf will always be a UHF object
    orbspin = scf.addons.get_ghf_orbspin(cc.mf.mo_energy, cc.mf.mo_occ, is_rhf=False)

    # orbspin will be a list of length #spinorbs, and will say whether we want to include a
    # alpha or beta spinorb in the ghf ordering
    assert(len(orbspin) == cc.nso)

    # This is a very non-pythonic way of solving this problem!
    def increment_alpha_ind(ind_a):
        # Initial index should be 0
        ind_a += 1
        if ind_a == cc.na:
            # Switch to the starting index of the virtual alpha block
            ind_a = cc.no
        return ind_a

    def increment_beta_ind(ind_b):
        # Initial index should be cc.na (which is where the occupied beta will start)
        ind_b += 1
        if ind_b == cc.no:
            # Switch to the starting index of the virtual beta block
            ind_b = cc.no + cc.va
        return ind_b

    idx = []
    ind_a = 0
    ind_b = cc.na
    for i in range(len(orbspin)):
        if orbspin[i] == 0:
            # Next orbital should be an alpha
            idx.append(ind_a)
            ind_a = increment_alpha_ind(ind_a)
        elif orbspin[i] == 1:
            # Next orbital should be an beta
            idx.append(ind_b)
            ind_b = increment_beta_ind(ind_b)

    ndim = mat.ndim 
    if ndim == 2:
        mat_reorder = mat[np.ix_(idx,idx)]
    elif ndim == 4:
        mat_reorder = mat[np.ix_(idx,idx,idx,idx)]
    else:
        raise NotImplementedError

    return mat_reorder

def reorder_to_uhf(cc, mat):
    ''' Orbital dependent expectation values (e.g. RDMs) from ebcc are returned with
    a spinorbital index of occ_a, occ_b, virt_a, virt_b.

    This routine will reorder them to separate spinned blocks in a UHF energy-ordering of the spin-orbitals,
    disregarding spin, so they can be compared to a normal GHF implementation of
    these quantities. The energy ordering comes from the original cc.mf object'''

    raise NotImplementedError
    return None

def block_diag(A, B):
    """Return a block diagonal matrix
       A 0
       0 B
    """
    ma = A.shape[0]
    mb = B.shape[0]
    na = A.shape[1]
    nb = B.shape[1]

    z1 = np.zeros((ma, nb))
    z2 = np.zeros((mb, na))
    M1 = np.hstack((A, z1))
    M2 = np.hstack((z2, B))
    return np.vstack((M1, M2))


def D1(ev, eo):
    """Create 4D tensor of energy denominators from
    2 1-d arrays"""
    D1 = ev[:, None] - eo[None, :]
    return D1


def D2(ev, eo):
    """Create 4D tensor of energy denominators from
    2 1-d arrays"""
    D2 = (ev[:, None, None, None] + ev[None, :, None, None]
          - eo[None, None, :, None] - eo[None, None, None, :])
    return D2


def D2u(eva, evb, eoa, eob):
    """Create 4D tensor of energy denominators from
    2 1-d arrays"""
    D2 = (eva[:, None, None, None] + evb[None, :, None, None]
          - eoa[None, None, :, None] - eob[None, None, None, :])
    return D2

class one_e_blocks:
    def __init__(self, oo, ov, vo, vv):
        self.oo = oo
        self.ov = ov
        self.vo = vo
        self.vv = vv

class two_e_blocks:
    def __init__(self,
                 vvvv=None, vvvo=None, vovv=None, vvoo=None,
                 vovo=None, oovv=None, vooo=None, ooov=None, oooo=None):
        self.vvvv = vvvv
        self.vvvo = vvvo
        self.vovv = vovv
        self.vvoo = vvoo
        self.vovo = vovo
        self.oovv = oovv
        self.vooo = vooo
        self.ooov = ooov
        self.oooo = oooo
        # Added by ghb
        self.ovov = vovo.copy().transpose((1, 0, 3, 2))
        self.ovvv = vovv.copy().transpose((1, 0, 3, 2))
        self.ovoo = vooo.copy().transpose((1, 0, 3, 2))
        self.vvov = vvvo.copy().transpose((1, 0, 3, 2)) 
