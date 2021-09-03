import numpy

def block_diag(A, B):
    """Return a block diagonal matrix
       A 0
       0 B
    """
    ma = A.shape[0]
    mb = B.shape[0]
    na = A.shape[1]
    nb = B.shape[1]

    z1 = numpy.zeros((ma, nb))
    z2 = numpy.zeros((mb, na))
    M1 = numpy.hstack((A, z1))
    M2 = numpy.hstack((z2, B))
    return numpy.vstack((M1, M2))


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
