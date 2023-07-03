"""General electron-boson coupled cluster.
"""

import functools
import itertools
from typing import Sequence

import numpy as np
import scipy.linalg
from pyscf import ao2mo, lib, scf

from ebcc import geom, rebcc, uebcc, util
from ebcc.brueckner import BruecknerGEBCC
from ebcc.eris import GERIs
from ebcc.fock import GFock
from ebcc.space import Space


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


@util.inherit_docstrings
class GEBCC(rebcc.REBCC):
    Amplitudes = Amplitudes
    ERIs = GERIs
    Brueckner = BruecknerGEBCC

    @staticmethod
    def _convert_mf(mf):
        if isinstance(mf, scf.ghf.GHF):
            return mf
        # NOTE: First convert to UHF - otherwise conversions from
        # RHF->GHF and UHF->GHF may have inconsistent ordering
        return mf.to_uhf().to_ghf()

    @classmethod
    def from_uebcc(cls, ucc):
        """Initialise a GEBCC object from an UEBCC object."""

        # FIXME test for frozen/active
        # FIXME won't work with active spaces
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

        occupied = np.zeros((nocc + nvir,), dtype=bool)
        occupied[slices["a"]] = ucc.space[0]._occupied.copy()
        occupied[slices["b"]] = ucc.space[1]._occupied.copy()
        frozen = np.zeros((nocc + nvir,), dtype=bool)
        frozen[slices["a"]] = ucc.space[0]._frozen.copy()
        frozen[slices["b"]] = ucc.space[1]._frozen.copy()
        active = np.zeros((nocc + nvir,), dtype=bool)
        active[slices["a"]] = ucc.space[0]._active.copy()
        active[slices["b"]] = ucc.space[1]._active.copy()
        space = Space(occupied, frozen, active)

        gcc = cls(
            ucc.mf,
            log=ucc.log,
            ansatz=ucc.ansatz,
            space=space,
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

            for name, key, n in ucc.ansatz.fermionic_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = np.zeros((space.ncocc,) * n + (space.ncvir,) * n)
                for comb in util.generate_spin_combinations(n, unique=True):
                    done = set()
                    for lperm, lsign in util.permutations_with_signs(tuple(range(n))):
                        for uperm, usign in util.permutations_with_signs(tuple(range(n))):
                            combn = util.permute_string(comb[:n], lperm)
                            combn += util.permute_string(comb[n:], uperm)
                            if combn in done:
                                continue
                            mask = np.ix_(
                                *([occs[c] for c in combn[:n]] + [virs[c] for c in combn[n:]])
                            )
                            transpose = tuple(lperm) + tuple(p + n for p in uperm)
                            amp = (
                                getattr(ucc.amplitudes[name], comb).transpose(transpose)
                                * lsign
                                * usign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(n))):
                                transpose = tuple(perm) + tuple(range(n, 2 * n))
                                if util.permute_string(comb[:n], perm) == comb[:n]:
                                    amplitudes[name][mask] += (
                                        amp.transpose(transpose).copy() * sign
                                    )
                            done.add(combn)

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = ucc.amplitudes[name].copy()

            for name, key, nf, nb in ucc.ansatz.coupling_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = np.zeros(
                    (nbos,) * nb + (space.ncocc,) * nf + (space.ncvir,) * nf
                )
                for comb in util.generate_spin_combinations(nf):
                    done = set()
                    for lperm, lsign in util.permutations_with_signs(tuple(range(nf))):
                        for uperm, usign in util.permutations_with_signs(tuple(range(nf))):
                            combn = util.permute_string(comb[:nf], lperm)
                            combn += util.permute_string(comb[nf:], uperm)
                            if combn in done:
                                continue
                            mask = np.ix_(
                                *([range(nbos)] * nb),
                                *(
                                    [occs[c] for c in combn[:nf]]
                                    + [virs[c] for c in combn[nf:]]
                                ),
                            )
                            transpose = (
                                tuple(range(nb))
                                + tuple(p + nb for p in lperm)
                                + tuple(p + nb + nf for p in uperm)
                            )
                            amp = (
                                getattr(ucc.amplitudes[name], comb).transpose(
                                    transpose
                                )
                                * lsign
                                * usign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(nf))):
                                transpose = (
                                    tuple(range(nb))
                                    + tuple(p + nb for p in perm)
                                    + tuple(range(nb + nf, nb + 2 * nf))
                                )
                                if util.permute_string(comb[:nf], perm) == comb[:nf]:
                                    amplitudes[name][mask] += (
                                        amp.transpose(transpose).copy() * sign
                                    )
                            done.add(combn)

            gcc.amplitudes = amplitudes

        if has_lams:
            lambdas = gcc.init_lams()  # Easier this way - but have to build ERIs...

            for name, key, n in ucc.ansatz.fermionic_cluster_ranks(spin_type=ucc.spin_type):
                lname = name.replace("t", "l")
                lambdas[lname] = np.zeros((space.ncvir,) * n + (space.ncocc,) * n)
                for comb in util.generate_spin_combinations(n, unique=True):
                    done = set()
                    for lperm, lsign in util.permutations_with_signs(tuple(range(n))):
                        for uperm, usign in util.permutations_with_signs(tuple(range(n))):
                            combn = util.permute_string(comb[:n], lperm)
                            combn += util.permute_string(comb[n:], uperm)
                            if combn in done:
                                continue
                            mask = np.ix_(
                                *([virs[c] for c in combn[:n]] + [occs[c] for c in combn[n:]])
                            )
                            transpose = tuple(lperm) + tuple(p + n for p in uperm)
                            amp = (
                                getattr(ucc.lambdas[lname], comb).transpose(transpose)
                                * lsign
                                * usign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(n))):
                                transpose = tuple(perm) + tuple(range(n, 2 * n))
                                if util.permute_string(comb[:n], perm) == comb[:n]:
                                    lambdas[lname][mask] += (
                                        amp.transpose(transpose).copy() * sign
                                    )
                            done.add(combn)

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                lambdas[lname] = ucc.lambdas[lname].copy()

            for name, key, nf, nb in ucc.ansatz.coupling_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                lambdas[lname] = np.zeros(
                    (nbos,) * nb + (space.ncvir,) * nf + (space.ncocc,) * nf
                )
                for comb in util.generate_spin_combinations(nf, unique=True):
                    done = set()
                    for lperm, lsign in util.permutations_with_signs(tuple(range(nf))):
                        for uperm, usign in util.permutations_with_signs(tuple(range(nf))):
                            combn = util.permute_string(comb[:nf], lperm)
                            combn += util.permute_string(comb[nf:], uperm)
                            if combn in done:
                                continue
                            mask = np.ix_(
                                *([range(nbos)] * nb),
                                *(
                                    [virs[c] for c in combn[:nf]]
                                    + [occs[c] for c in combn[nf:]]
                                ),
                            )
                            transpose = (
                                tuple(range(nb))
                                + tuple(p + nb for p in lperm)
                                + tuple(p + nb + nf for p in uperm)
                            )
                            amp = (
                                getattr(ucc.lambdas[lname], comb).transpose(
                                    transpose
                                )
                                * lsign
                                * usign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(nf))):
                                transpose = (
                                    tuple(range(nb))
                                    + tuple(p + nb for p in perm)
                                    + tuple(range(nb + nf, nb + 2 * nf))
                                )
                                if util.permute_string(comb[:nf], perm) == comb[:nf]:
                                    lambdas[lname][mask] += (
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

        # Build T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            if n == 1:
                ei = getattr(self, "e" + key[0])
                ea = getattr(self, "e" + key[1])
                e_ia = lib.direct_sum("i-a->ia", ei, ea)
                amplitudes[name] = getattr(self.fock, key) / e_ia
            elif n == 2:
                ei = getattr(self, "e" + key[0])
                ej = getattr(self, "e" + key[1])
                ea = getattr(self, "e" + key[2])
                eb = getattr(self, "e" + key[3])
                e_ia = lib.direct_sum("i-a->ia", ei, ea)
                e_jb = lib.direct_sum("i-a->ia", ej, eb)
                e_ijab = lib.direct_sum("ia,jb->ijab", e_ia, e_jb)
                amplitudes[name] = getattr(eris, key) / e_ijab
            else:
                shape = tuple(self.space.size(k) for k in key)
                amplitudes[name] = np.zeros(shape)

        if self.boson_ansatz:
            # Only true for real-valued couplings:
            h = self.g
            H = self.G

        # Build S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            if n == 1:
                amplitudes[name] = -H / self.omega
            else:
                shape = (self.nbos,) * n
                amplitudes[name] = np.zeros(shape)

        # Build U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            if n == 1:
                ei = getattr(self, "e" + key[1])
                ea = getattr(self, "e" + key[2])
                e_xia = lib.direct_sum("i-a-x->xia", ei, ea, self.omega)
                amplitudes[name] = getattr(h, key) / e_xia
            else:
                shape = (self.nbos,) * nb + tuple(self.space.size(k) for k in key[nb:])
                amplitudes[name] = np.zeros(shape)

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

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[:-1]
            vectors.append(util.compress_axes(key, excitations[m]).ravel())
            m += 1

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return np.concatenate(vectors)

    def excitations_to_vector_ee(self, *excitations):
        vectors = []
        m = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(util.compress_axes(key, excitations[m]).ravel())
            m += 1

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return np.concatenate(vectors)

    def vector_to_excitations_ip(self, vector):
        excitations = []
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[:-1]
            size = util.get_compressed_size(key, **{k: self.space.size(k) for k in set(key)})
            shape = tuple(self.space.size(k) for k in key)
            vn_tril = vector[i0 : i0 + size]
            vn = util.decompress_axes(key, vn_tril, shape=shape)
            excitations.append(vn)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return tuple(excitations)

    def vector_to_excitations_ea(self, vector):
        excitations = []
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[n:] + key[:n-1]
            size = util.get_compressed_size(key, **{k: self.space.size(k) for k in set(key)})
            shape = tuple(self.space.size(k) for k in key)
            vn_tril = vector[i0 : i0 + size]
            vn = util.decompress_axes(key, vn_tril, shape=shape)
            excitations.append(vn)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        return tuple(excitations)

    def vector_to_excitations_ee(self, vector):
        excitations = []
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            size = util.get_compressed_size(key, **{k: self.space.size(k) for k in set(key)})
            shape = tuple(self.space.size(k) for k in key)
            vn_tril = vector[i0 : i0 + size]
            vn = util.decompress_axes(key, vn_tril, shape=shape)
            excitations.append(vn)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            raise util.ModelNotImplemented

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
    def spin_type(self):
        return "G"


@util.inherit_docstrings
class SplitGEBCC(GEBCC, rebcc.SplitREBCC):

    @classmethod
    def from_uebcc(cls, ucc):
        raise NotImplementedError

    def init_amps(self, eris=None):
        eris = self.get_eris(eris)

        amplitudes = self.Amplitudes()

        def get_e_ia(key):
            ei = np.diag(getattr(self.fock, key[0] * 2))
            ea = np.diag(getattr(self.fock, key[1] * 2))
            return lib.direct_sum("i-a->ia", ei, ea)

        # Build T amplitudes:
        for n, keys in self.ansatz.split_cluster_ranks(spin=self.spin_type)[0]:
            amplitudes["t%d" % n] = util.Namespace()
            for key in keys:
                if n == 1:
                    e_ia = get_e_ia(key)
                    setattr(amplitudes["t%d" % n], key, getattr(self.fock, key) / e_ia)
                elif n == 2:
                    e_ijab = lib.direct_sum("ia,jb->ijab", get_e_ia(key[0] + key[2]), get_e_ia(key[1] + key[3]))
                    setattr(amplitudes["t%d" % n], key, getattr(eris, key) / e_ijab)
                else:
                    shape = tuple({"o": self.space.nocc, "O": self.space.naocc, "i": self.space.niocc}[k] for k in key[:n])
                    shape += tuple({"v": self.space.nvir, "V": self.space.navir, "a": self.space.nivir}[k] for k in key[n:])
                    setattr(amplitudes["t%d" % n], key, np.zeros(shape))

        if self.boson_ansatz:
            raise NotImplementedError

        return amplitudes
