"""General electron-boson coupled cluster.
"""

import functools
import itertools
import numpy as np
import scipy.linalg
from typing import Tuple
from types import SimpleNamespace
from pyscf import lib, ao2mo
from ebcc import util, rebcc


class Amplitudes(rebcc.Amplitudes):
    """Amplitude container class. Consists of a dictionary with keys
    that are strings of the name of each amplitude. Values are
    namespaces with keys indicating whether each fermionic dimension
    is alpha (`"a"`) or beta (`"b"`) spin, and values are arrays whose
    dimension depends on the particular amplitude. For purely bosonic
    amplitudes the values of `Amplitudes` are simply arrays, with no
    fermionic spins to index.
    """

    pass


class ERIs(rebcc.ERIs):
    """Electronic repulsion integral container class. Consists of a
    namespace containing blocks of the integrals, with keys that are
    length-4 strings of `"o"` or `"v"` signifying whether the
    corresponding dimension is occupied or virtual.
    """

    def __init__(self, ebcc, slices=None, mo_coeff=None):
        rebcc.ERIs.__init__(self, ebcc, slices=slices, mo_coeff=mo_coeff)

        mo_a = [mo[:self.mf.mol.nao] for mo in self.mo_coeff]
        mo_b = [mo[self.mf.mol.nao:] for mo in self.mo_coeff]

        eri  = ao2mo.kernel(self.mf._eri, mo_a)
        eri += ao2mo.kernel(self.mf._eri, mo_b)
        eri += ao2mo.kernel(self.mf._eri, mo_a[:2] + mo_b[2:])
        eri += ao2mo.kernel(self.mf._eri, mo_b[:2] + mo_a[2:])

        eri = ao2mo.addons.restore(1, eri, ebcc.nmo)
        eri = eri.reshape((ebcc.nmo,) * 4)
        eri = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)

        self.eri = eri

    def __getattr__(self, key):
        i, j, k, l = (self.slices[i][k] for i, k in enumerate(key))
        return self.eri[i, j, k, l]


@util.inherit_docstrings
class GEBCC(rebcc.REBCC):
    Amplitudes = Amplitudes
    ERIs = ERIs

    @staticmethod
    def _convert_mf(mf):
        return mf.to_ghf()

    def init_amps(self, eris=None):
        if eris is None:
            eris = self.get_eris()

        amplitudes = self.Amplitudes()
        e_ia = lib.direct_sum("i-a->ia", self.eo, self.ev)

        # Build T amplitudes:
        for n in self.rank_numeric[0]:
            if n == 1:
                amplitudes["t%d" % n] = self.fock.vo.T / e_ia
            elif n == 2:
                e_ijab = lib.direct_sum("ia,jb->ijab", e_ia, e_ia)
                amplitudes["t%d" % n] = eris.oovv / e_ijab
            else:
                amplitudes["t%d" % n] = np.zeros((self.nocc,) * n + (self.nvir,) * n)

        if not (self.rank[1] == self.rank[2] == ""):
            # Only true for real-valued couplings:
            h = self.g
            H = self.G

        # Build S amplitudes:
        for n in self.rank_numeric[1]:
            if n == 1:
                amplitudes["s%d" % n] = -H / self.omega
            else:
                amplitudes["s%d" % n] = np.zeros((self.nbos,) * n)

        # Build U amplitudes:
        for n in self.rank_numeric[2]:
            if n == 1:
                e_xia = lib.direct_sum("ia-x->xia", e_ia, self.omega)
                amplitudes["u1%d" % n] = h.bov / e_xia
            else:
                amplitudes["u1%d" % n] = np.zeros((self.nbos,) * n + (self.nocc, self.nvir))

        return amplitudes

    def get_mean_field_G(self):
        val = lib.einsum("Ipp->I", self.g.boo)
        val -= self.xi * self.omega

        if self.bare_G is not None:
            val += self.bare_G

        return val

    def get_eris(self):
        return self.ERIs(self)

    #def excitations_to_vector_ip(self, *excitations):
    #    """Construct a vector containing all of the excitation
    #    amplitudes used in the given ansatz.
    #    """

    #    vectors = []
    #    m = 0

    #    for n in self.rank_numeric[0]:
    #        shape = (self.nocc,) * n + (self.nvir,) * (n-1)
    #        occ = util.tril_indices_ndim(self.nocc, n)
    #        vir = util.tril_indices_ndim(self.nvir, n-1)
    #        excitation = excitations[m]
    #        excitation = excitation.reshape(shape)
    #        if n == 1:
    #            excitation = excitation[occ]
    #        else:
    #            excitation = excitation[occ][..., vir]
    #        vectors.append(excitation.ravel())
    #        m += 1

    #    for n in self.rank_numeric[1]:
    #        raise NotImplementedError

    #    for n in self.rank_numeric[2]:
    #        raise NotImplementedError

    #    return np.concatenate(vectors)

    #def excitations_to_vector_ea(self, *excitations):
    #    """Construct a vector containing all of the excitation
    #    amplitudes used in the given ansatz.
    #    """

    #    vectors = []
    #    m = 0

    #    for n in self.rank_numeric[0]:
    #        shape = (self.nvir,) * n + (self.nocc,) * (n-1)
    #        occ = util.tril_indices_ndim(self.nocc, n-1)
    #        vir = util.tril_indices_ndim(self.nvir, n)
    #        excitation = excitations[m]
    #        excitation = excitation.reshape(shape)
    #        excitation = excitation[vir][..., occ]
    #        vectors.append(excitation.ravel())
    #        m += 1

    #    for n in self.rank_numeric[1]:
    #        raise NotImplementedError

    #    for n in self.rank_numeric[2]:
    #        raise NotImplementedError

    #    return np.concatenate(vectors)

    #def vector_to_excitations_ip(self, vector):
    #    """Construct all of the excitation amplitudes used in the
    #    given ansatz from a vector.
    #    """

    #    excitations = []
    #    i0 = 0

    #    for n in self.rank_numeric[0]:
    #        shape = (self.nocc,) * n + (self.nvir,) * (n-1)
    #        excitation = np.zeros(shape)
    #        occ = util.tril_indices_ndim(self.nocc, n)
    #        vir = util.tril_indices_ndim(self.nvir, n-1)

    #        if n == 1:
    #            reduced_size = excitation[occ].size
    #            shape = excitation[occ].shape
    #            excitation[occ] = vector[i0:i0+reduced_size].reshape(shape)

    #            for pocc, so in util.permutations_with_signs(occ):
    #                pocc = tuple(pocc)
    #                excitation[pocc] = so * excitation[occ]

    #        else:
    #            reduced_size = excitation[occ][..., vir].size
    #            shape = excitation[occ][..., vir].shape
    #            excitation[occ][..., vir] = vector[i0:i0+reduced_size].reshape(shape)

    #            for pocc, so in util.permutations_with_signs(occ):
    #                pocc = tuple(pocc)
    #                for pvir, sv in util.permutations_with_signs(vir):
    #                    pvir = tuple(pvir)
    #                    excitation[pocc][..., pvir] = so * sv * excitation[occ][..., vir]

    #        excitations.append(excitation)
    #        i0 += reduced_size

    #    for n in self.rank_numeric[1]:
    #        raise NotImplementedError

    #    for n in self.rank_numeric[2]:
    #        raise NotImplementedError

    #    return tuple(excitations)

    #def vector_to_excitations_ea(self, vector):
    #    """Construct all of the excitation amplitudes used in the
    #    given ansatz from a vector.
    #    """

    #    excitations = []
    #    i0 = 0

    #    for n in self.rank_numeric[0]:
    #        shape = (self.nvir,) * n + (self.nocc,) * (n-1)
    #        occ = util.tril_indices_ndim(self.nocc, n-1)
    #        vir = util.tril_indices_ndim(self.nvir, n)
    #        excitation = np.zeros((self.nvir**n, self.nocc**(n-1)))
    #        excitation[np.ix_(*vir, *occ)] = vector[i0:i0+len(vir)*len(occ)]
    #        excitation = excitation.reshape(shape)

    #        for a in itertools.product(range(nvir), repeat=n):
    #            for i in itertools.product(range(nocc), repeat=n-1):
    #                sign = pow(-1, util.minimum_swaps(a) + util.minimum_swaps(i))
    #                inds = tuple(a) + tuple(i)
    #                ref = tuple(sorted(a)) + tuple(sorted(i))
    #                excitation[inds] = excitation[ref] * sign

    #        excitations.append(excitation)

    #    for n in self.rank_numeric[1]:
    #        raise NotImplementedError

    #    for n in self.rank_numeric[2]:
    #        raise NotImplementedError

    #    return tuple(excitations)

    @property
    def xi(self):
        if self.options.shift:
            xi = lib.einsum("Iii->I", self.g.boo)
            xi /= self.omega
            if self.bare_G is not None:
                xi += self.bare_G / self.omega
        else:
            xi = np.zeros_like(self.omega)
        return xi

    @property
    def name(self):
        return "GCC" + "-".join(self.rank).rstrip("-")



if __name__ == "__main__":
    from pyscf import gto, scf

    mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="6-31g", verbose=0)
    mf = scf.RHF(mol).run()

    ccsd = GEBCC(mf)
    ccsd.kernel()
