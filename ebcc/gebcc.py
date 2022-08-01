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

        mo_a = [mo[: self.mf.mol.nao] for mo in self.mo_coeff]
        mo_b = [mo[self.mf.mol.nao :] for mo in self.mo_coeff]

        eri = ao2mo.kernel(self.mf._eri, mo_a)
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
        for nf in self.rank_numeric[2]:
            if nf != 1:
                raise NotImplementedError
            for nb in self.rank_numeric[3]:
                if n == 1:
                    e_xia = lib.direct_sum("ia-x->xia", e_ia, self.omega)
                    amplitudes["u%d%d" % (nf, nb)] = h.bov / e_xia
                else:
                    amplitudes["u%d%d" % (nf, nb)] = np.zeros(
                        (self.nbos,) * nb + (self.nocc, self.nvir)
                    )

        return amplitudes

    def make_rdm2_f(self, eris=None, amplitudes=None, lambdas=None, hermitise=True):
        func, kwargs = self._load_function(
            "make_rdm2_f",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )

        dm = func(**kwargs)

        if hermitise:
            dm = 0.5 * (+dm.transpose(0, 1, 2, 3) + dm.transpose(2, 3, 0, 1))
            dm = 0.5 * (+dm.transpose(0, 1, 2, 3) + dm.transpose(1, 0, 3, 2))

        return dm

    def excitations_to_vector_ip(self, *excitations):
        vectors = []
        m = 0

        for n in self.rank_numeric[0]:
            if n == 1:
                vectors.append(excitations[m].ravel())
            elif n == 2:
                oinds = util.tril_indices_ndim(self.nocc, 2, include_diagonal=False)
                vectors.append(excitations[m][oinds].ravel())
            else:
                raise NotImplementedError
            m += 1

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        return np.concatenate(vectors)

    def excitations_to_vector_ea(self, *excitations):
        vectors = []
        m = 0

        for n in self.rank_numeric[0]:
            if n == 1:
                vectors.append(excitations[m].ravel())
            elif n == 2:
                vinds = util.tril_indices_ndim(self.nvir, 2, include_diagonal=False)
                vectors.append(excitations[m][vinds].ravel())
            else:
                raise NotImplementedError
            m += 1

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        return np.concatenate(vectors)

    def vector_to_excitations_ip(self, vector):
        excitations = []
        i0 = 0

        for n in self.rank_numeric[0]:
            if n == 1:
                size = self.nocc
                r = vector[i0 : i0 + size].copy()
            elif n == 2:
                o1, o2 = util.tril_indices_ndim(self.nocc, 2, include_diagonal=False)
                r = np.zeros((self.nocc, self.nocc, self.nvir))
                nocc2 = self.nocc * (self.nocc - 1) // 2
                size = nocc2 * self.nvir
                r[o1, o2] = vector[i0 : i0 + size].reshape(nocc2, self.nvir).copy()
                r[o2, o1] = -vector[i0 : i0 + size].reshape(nocc2, self.nvir).copy()
            else:
                raise NotImplementedError
            excitations.append(r)
            i0 += size

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        return tuple(excitations)

    def vector_to_excitations_ea(self, vector):
        excitations = []
        i0 = 0

        for n in self.rank_numeric[0]:
            if n == 1:
                size = self.nvir
                r = vector[i0 : i0 + size].copy()
            elif n == 2:
                v1, v2 = util.tril_indices_ndim(self.nvir, 2, include_diagonal=False)
                r = np.zeros((self.nvir, self.nvir, self.nocc))
                nvir2 = self.nvir * (self.nvir - 1) // 2
                size = nvir2 * self.nocc
                r[v1, v2] = vector[i0 : i0 + size].reshape(nvir2, self.nocc).copy()
                r[v2, v1] = -vector[i0 : i0 + size].reshape(nvir2, self.nocc).copy()
            else:
                raise NotImplementedError
            excitations.append(r)
            i0 += size

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        return tuple(excitations)

    def get_mean_field_G(self):
        val = lib.einsum("Ipp->I", self.g.boo)
        val -= self.xi * self.omega

        if self.bare_G is not None:
            val += self.bare_G

        return val

    def get_eris(self):
        return self.ERIs(self)

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
        return super().name.replace("R", "G", 1)


if __name__ == "__main__":
    from pyscf import gto, scf

    mol = gto.M(atom="H 0 0 0; F 0 0 1.1", basis="6-31g", verbose=0)
    mf = scf.RHF(mol).run()

    ccsd = GEBCC(mf)
    ccsd.kernel()
