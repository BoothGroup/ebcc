"""General electron-boson coupled cluster.
"""

import functools
import itertools
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import scipy.linalg
from pyscf import ao2mo, lib, scf

from ebcc import rebcc, uebcc, util


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

    @classmethod
    def from_uebcc(cls, ucc):
        """Initialise a GEBCC object from an UEBCC object."""

        raise NotImplementedError("UEBCC -> GEBCC conversion is a work in progress.")

        orbspin = scf.addons.get_ghf_orbspin(ucc.mf.mo_energy, ucc.mf.mo_occ, False)
        nocc = ucc.nocc[0] + ucc.nocc[1]
        slices = {"a": np.where(orbspin == 0)[0], "b": np.where(orbspin == 1)[0]}
        occs = {"a": np.where(orbspin[:nocc] == 0)[0], "b": np.where(orbspin[:nocc] == 1)[0]}
        virs = {"a": np.where(orbspin[nocc:] == 0)[0], "b": np.where(orbspin[nocc:] == 1)[0]}

        if ucc.bare_g is not None:
            if np.asarray(ucc.bare_g).ndim == 3:
                bare_g_a = bare_g_b = ucc.bare_g
            else:
                bare_g_a, bare_g_b = ucc.bare_g
            g = np.zeros((ucc.nbos, ucc.nmo * 2, ucc.nmo * 2))
            g[np.ix_(range(ucc.nbos), slices["a"], slices["a"])] = bare_g_a.copy()
            g[np.ix_(range(ucc.nbos), slices["b"], slices["b"])] = bare_g_b.copy()
        else:
            g = None

        gcc = cls(
            ucc.mf,
            log=ucc.log,
            fermion_excitations=ucc.fermion_excitations,
            boson_excitations=ucc.boson_excitations,
            fermion_coupling_rank=ucc.fermion_coupling_rank,
            boson_coupling_rank=ucc.boson_coupling_rank,
            omega=ucc.omega,
            g=g,
            G=ucc.bare_G,
            options=ucc.options,
        )

        gcc.e_corr = ucc.e_corr
        gcc.converged = ucc.converged
        gcc.converged_lambda = ucc.converged_lambda

        has_amps = ucc.amplitudes is not None
        has_lams = ucc.lambdas is not None

        if has_amps:
            Amplitudes = cls.Amplitudes()

            for n in ucc.rank_numeric[0]:
                for comb in uebcc.generate_spin_combinations(n):
                    done = set()
                    for perm, sign in util.permutations_with_signs(tuple(range(n))):
                        combn = util.permute_string(comb[:n], perm) + comb[n:]
                        if combn in done:
                            continue
                        mask = np.ix_(
                            *([occs[c] for c in combn[:n]] + [virs[c] for c in combn[n:]])
                        )
                        transpose = tuple(perm) + tuple(range(n, 2 * n))
                        amplitudes["t%d" % n][mask] = (
                            getattr(ucc.amplitudes["t%d" % n], comb).transpose(transpose).copy()
                            * sign
                        )
                        done.add(combn)

            # from pyscf.cc.addons import spatial2spin
            # ta = amplitudes["t1"]
            # tb = spatial2spin((ucc.amplitudes["t1"].aa, ucc.amplitudes["t1"].bb), orbspin=orbspin)
            # assert np.allclose(ta, tb)
            # ta = amplitudes["t2"]
            # tb = spatial2spin((ucc.amplitudes["t2"].aaaa, ucc.amplitudes["t2"].abab, ucc.amplitudes["t2"].bbbb), orbspin=orbspin)
            # assert np.allclose(ta, tb)

            for n in ucc.rank_numeric[1]:
                amplitudes["s%d" % n] = ucc.amplitudes["s%d" % n].copy()

            for nf in ucc.rank_numeric[2]:
                for nb in ucc.rank_numeric[3]:
                    for comb in uebcc.generate_spin_combinations(nf):
                        bmasks = [range(ucc.nbos)] * nb
                        fmasks = [occs[c] for c in comb[:nf]] + [virs[c] for c in comb[nf:]]
                        mask = np.ix_(*bmasks, *fmasks)
                        amplitudes["u%d%d" % (nf, nb)][mask] = getattr(
                            ucc.amplitudes["u%d%d" % (nf, nb)], comb
                        ).copy()
                    amplitudes["u%d%d" % (nf, nb)] = util.antisymmetrise_array(
                        amplitudes["u%d%d" % (nf, nb)], axes=tuple(range(nb, nb + nf))
                    )

            gcc.amplitudes = amplitudes

        if has_lams:
            lambas = gcc.init_lams()  # Easier this way - but have to build ERIs...

            for n in ucc.rank_numeric[0]:
                for comb in uebcc.generate_spin_combinations(n):
                    mask = np.ix_(*([virs[c] for c in comb[:n]] + [occs[c] for c in comb[n:]]))
                    lambdas["l%d" % n][mask] = getattr(ucc.lambdas["l%d" % n], comb).copy()
                lambdas["l%d" % n] = util.antisymmetrise_array(
                    lambdas["l%d" % n], axes=tuple(range(n))
                )

            for n in ucc.rank_numeric[1]:
                lambdas["ls%d" % n] = ucc.lambdas["ls%d" % n].copy()

            for nf in rcc.rank_numeric[2]:
                for nb in rcc.rank_numeric[3]:
                    for comb in uebcc.generate_spin_combinations(nf):
                        bmasks = [range(ucc.nbos)] * nb
                        fmasks = [virs[c] for c in comb[:nf]] + [occs[c] for c in comb[nf:]]
                        mask = np.ix_(*bmasks, *fmasks)
                        lambdas["lu%d%d" % (nf, nb)][mask] = getattr(
                            ucc.lambdas["lu%d%d" % (nf, nb)], comb
                        ).copy()
                    lambdas["lu%d%d" % (nf, nb)] = util.antisymmetrise_array(
                        lambdas["lu%d%d" % (nf, nb)], axes=tuple(range(nb, nb + nf))
                    )

            gcc.lambdas = lambdas

        return gcc

    @classmethod
    def from_rebcc(cls, rcc):
        """Initialise a GEBCC object from an REBCC object."""

        ucc = uebcc.UEBCC.from_rebcc(rcc)
        gcc = cls.from_uebcc(ucc)

        return gcc

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
