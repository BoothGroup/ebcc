"""General electron-boson coupled cluster.
"""

import functools
import itertools
from types import SimpleNamespace
from typing import Tuple

import numpy as np
import scipy.linalg
from pyscf import ao2mo, lib, scf

from ebcc import geom, rebcc, uebcc, util


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

    def __init__(self, ebcc, array=None, slices=None, mo_coeff=None):

        if array is None:
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
        else:
            eri = array

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

        # FIXME how will this behave with cluster spaces?
        orbspin = scf.addons.get_ghf_orbspin(ucc.mf.mo_energy, ucc.mf.mo_occ, False)
        nocc = sum(ucc.nocc)
        nvir = sum(ucc.nvir)
        nbos = ucc.nbos
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
            amplitudes = cls.Amplitudes()

            for n in ucc.rank_numeric[0]:
                amplitudes["t%d" % n] = np.zeros((nocc,) * n + (nvir,) * n)
                for comb in util.generate_spin_combinations(n):
                    done = set()
                    for perm, sign in util.permutations_with_signs(tuple(range(n))):
                        combn = util.permute_string(comb[:n], perm) + comb[n:]
                        if combn in done:
                            continue
                        mask = np.ix_(
                            *([occs[c] for c in combn[:n]] + [virs[c] for c in combn[n:]])
                        )
                        transpose = tuple(perm) + tuple(range(n, 2 * n))
                        amp = getattr(ucc.amplitudes["t%d" % n], comb).transpose(transpose) * sign
                        for perm, sign in util.permutations_with_signs(tuple(range(n))):
                            transpose = tuple(perm) + tuple(range(n, 2 * n))
                            if util.permute_string(comb[:n], perm) == comb[:n]:
                                amplitudes["t%d" % n][mask] += (
                                    amp.transpose(transpose).copy() * sign
                                )
                        done.add(combn)

            for n in ucc.rank_numeric[1]:
                amplitudes["s%d" % n] = ucc.amplitudes["s%d" % n].copy()

            for nf in ucc.rank_numeric[2]:
                for nb in ucc.rank_numeric[3]:
                    amplitudes["u%d%d" % (nf, nb)] = np.zeros(
                        (nbos,) * nb + (nocc,) * nf + (nvir,) * nf
                    )
                    for comb in util.generate_spin_combinations(nf):
                        done = set()
                        for perm, sign in util.permutations_with_signs(tuple(range(nf))):
                            combn = util.permute_string(comb[:nf], perm) + comb[nf:]
                            if combn in done:
                                continue
                            mask = np.ix_(
                                *([range(nbos)] * nb),
                                *([occs[c] for c in combn[:nf]] + [virs[c] for c in combn[nf:]]),
                            )
                            transpose = (
                                tuple(range(nb))
                                + tuple(p + nb for p in perm)
                                + tuple(p + nb + nf for p in perm)
                            )
                            amp = (
                                getattr(ucc.amplitudes["u%d%d" % (nf, nb)], comb).transpose(
                                    transpose
                                )
                                * sign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(nf))):
                                transpose = (
                                    tuple(range(nb))
                                    + tuple(p + nb for p in perm)
                                    + tuple(range(nb + nf, nb + 2 * nf))
                                )
                                if util.permute_string(comb[:nf], perm) == comb[:nf]:
                                    amplitudes["u%d%d" % (nf, nb)][mask] += (
                                        amp.transpose(transpose).copy() * sign
                                    )
                            done.add(combn)

            gcc.amplitudes = amplitudes

        if has_lams:
            lambdas = gcc.init_lams()  # Easier this way - but have to build ERIs...

            for n in ucc.rank_numeric[0]:
                lambdas["l%d" % n] = np.zeros((nvir,) * n + (nocc,) * n)
                for comb in util.generate_spin_combinations(n):
                    done = set()
                    for perm, sign in util.permutations_with_signs(tuple(range(n))):
                        combn = util.permute_string(comb[:n], perm) + comb[n:]
                        if combn in done:
                            continue
                        mask = np.ix_(
                            *([virs[c] for c in combn[:n]] + [occs[c] for c in combn[n:]])
                        )
                        transpose = tuple(perm) + tuple(range(n, 2 * n))
                        amp = getattr(ucc.lambdas["l%d" % n], comb).transpose(transpose) * sign
                        for perm, sign in util.permutations_with_signs(tuple(range(n))):
                            transpose = tuple(perm) + tuple(range(n, 2 * n))
                            if util.permute_string(comb[:n], perm) == comb[:n]:
                                lambdas["l%d" % n][mask] += amp.transpose(transpose).copy() * sign
                        done.add(combn)

            for n in ucc.rank_numeric[1]:
                lambdas["ls%d" % n] = ucc.lambdas["ls%d" % n].copy()

            for nf in ucc.rank_numeric[2]:
                for nb in ucc.rank_numeric[3]:
                    lambdas["lu%d%d" % (nf, nb)] = np.zeros(
                        (nbos,) * nb + (nvir,) * nf + (nocc,) * nf
                    )
                    for comb in util.generate_spin_combinations(nf):
                        done = set()
                        for perm, sign in util.permutations_with_signs(tuple(range(nf))):
                            combn = util.permute_string(comb[:nf], perm) + comb[nf:]
                            if combn in done:
                                continue
                            mask = np.ix_(
                                *([range(nbos)] * nb),
                                *([virs[c] for c in combn[:nf]] + [occs[c] for c in combn[nf:]]),
                            )
                            transpose = (
                                tuple(range(nb))
                                + tuple(p + nb for p in perm)
                                + tuple(p + nb + nf for p in perm)
                            )
                            amp = (
                                getattr(ucc.lambdas["lu%d%d" % (nf, nb)], comb).transpose(transpose)
                                * sign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(nf))):
                                transpose = (
                                    tuple(range(nb))
                                    + tuple(p + nb for p in perm)
                                    + tuple(range(nb + nf, nb + 2 * nf))
                                )
                                if util.permute_string(comb[:nf], perm) == comb[:nf]:
                                    lambdas["lu%d%d" % (nf, nb)][mask] += (
                                        amp.transpose(transpose).copy() * sign
                                    )
                            done.add(combn)

            gcc.lambdas = lambdas

        return gcc

    @classmethod
    def from_rebcc(cls, rcc):
        """Initialise a GEBCC object from an REBCC object."""

        ucc = uebcc.UEBCC.from_rebcc(rcc)
        gcc = cls.from_uebcc(ucc)

        return gcc

    def init_amps(self, eris=None):

        eris = self.get_eris(eris)

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
            dm = 0.5 * (dm.transpose(0, 1, 2, 3) + dm.transpose(2, 3, 0, 1))
            dm = 0.5 * (dm.transpose(0, 1, 2, 3) + dm.transpose(1, 0, 3, 2))

        return dm

    def excitations_to_vector_ip(self, *excitations):
        vectors = []
        m = 0

        for n in self.rank_numeric[0]:
            subscript = "i" * n + "a" * (n - 1)
            vectors.append(util.compress_axes(subscript, excitations[m]).ravel())
            m += 1

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        return np.concatenate(vectors)

    def excitations_to_vector_ee(self, *excitations):
        vectors = []
        m = 0

        for n in self.rank_numeric[0]:
            subscript = "i" * n + "a" * n
            vectors.append(util.compress_axes(subscript, excitations[m]).ravel())
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
            subscript = "i" * n + "a" * (n - 1)
            size = util.get_compressed_size(subscript, i=self.nocc, a=self.nvir)
            shape = tuple([self.nocc] * n + [self.nvir] * (n - 1))
            vn_tril = vector[i0 : i0 + size]
            vn = util.decompress_axes(subscript, vn_tril, shape=shape)
            excitations.append(vn)
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
            subscript = "a" * n + "i" * (n - 1)
            size = util.get_compressed_size(subscript, i=self.nocc, a=self.nvir)
            shape = tuple([self.nvir] * n + [self.nocc] * (n - 1))
            vn_tril = vector[i0 : i0 + size]
            vn = util.decompress_axes(subscript, vn_tril, shape=shape)
            excitations.append(vn)
            i0 += size

        for n in self.rank_numeric[1]:
            raise NotImplementedError

        for nf in self.rank_numeric[2]:
            for nb in self.rank_numeric[3]:
                raise NotImplementedError

        return tuple(excitations)

    def vector_to_excitations_ee(self, vector):
        excitations = []
        i0 = 0

        for n in self.rank_numeric[0]:
            subscript = "i" * n + "a" * n
            size = util.get_compressed_size(subscript, i=self.nocc, a=self.nvir)
            shape = tuple([self.nocc] * n + [self.nvir] * n)
            vn_tril = vector[i0 : i0 + size]
            vn = util.decompress_axes(subscript, vn_tril, shape=shape)
            excitations.append(vn)
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

    def get_eris(self, eris=None):
        """Get blocks of the ERIs.

        Parameters
        ----------
        eris : np.ndarray or ERIs, optional.
            Electronic repulsion integrals, either in the form of a
            dense array or an ERIs object. Default value is `None`.

        Returns
        -------
        eris : ERIs, optional
            Electronic repulsion integrals. Default value is generated
            using `self.ERIs()`.
        """
        if (eris is None) or isinstance(eris, np.ndarray):
            return self.ERIs(self, array=eris)
        else:
            return eris

    def ip_eom(self, options=None, **kwargs):
        return geom.IP_GEOM(self, options=options, **kwargs)

    def ea_eom(self, options=None, **kwargs):
        return geom.EA_GEOM(self, options=options, **kwargs)

    def ee_eom(self, options=None, **kwargs):
        return geom.EE_GEOM(self, options=options, **kwargs)

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

    # ccsd = GEBCC(mf)
    # ccsd.kernel()

    nbos = 5
    np.random.seed(1)
    g = np.random.random((nbos, mol.nao, mol.nao)) * 0.03
    g = g + g.transpose(0, 2, 1)
    omega = np.random.random((nbos)) * 0.5

    from ebcc import REBCC, UEBCC

    ccsd = REBCC(mf, e_tol=1e-10)
    ccsd.kernel()
    # ccsd.solve_lambda()

    ccsd = UEBCC.from_rebcc(ccsd)
    ccsd.kernel()
    # ccsd.solve_lambda()

    # ccsd = UEBCC(
    #    mf,
    #    # g=g,
    #    # omega=omega,
    #    e_tol=1e-10,
    #    # boson_excitations="S",
    #    # fermion_coupling_rank=1,
    #    # boson_coupling_rank=1,
    # )
    # ccsd.kernel()
    # ccsd.solve_lambda()

    # ccsd = GEBCC.from_uebcc(ccsd)
    # ccsd.kernel()
    # ccsd.solve_lambda()

    # geom = ccsd.ee_eom()
    # t = geom.moments(2, diagonal_only=True)
