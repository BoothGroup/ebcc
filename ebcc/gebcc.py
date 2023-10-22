"""General electron-boson coupled cluster."""

from pyscf import lib, scf

from ebcc import geom
from ebcc import numpy as np
from ebcc import uebcc, util
from ebcc.brueckner import BruecknerGEBCC
from ebcc.eris import GERIs
from ebcc.fock import GFock
from ebcc.precision import types
from ebcc.rebcc import REBCC
from ebcc.space import Space


@util.has_docstring
class GEBCC(REBCC, metaclass=util.InheritDocstrings):
    __doc__ = __doc__.replace("Restricted", "Generalised", 1)

    ERIs = GERIs
    Fock = GFock
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
        """Initialise a `GEBCC` object from an `UEBCC` object.

        Parameters
        ----------
        ucc : UEBCC
            The UEBCC object to initialise from.

        Returns
        -------
        gcc : GEBCC
            The GEBCC object.
        """

        orbspin = scf.addons.get_ghf_orbspin(ucc.mf.mo_energy, ucc.mf.mo_occ, False)
        nocc = ucc.space[0].nocc + ucc.space[1].nocc
        nvir = ucc.space[0].nvir + ucc.space[1].nvir
        nbos = ucc.nbos
        sa = np.where(orbspin == 0)[0]
        sb = np.where(orbspin == 1)[0]

        occupied = np.zeros((nocc + nvir,), dtype=bool)
        occupied[sa] = ucc.space[0]._occupied.copy()
        occupied[sb] = ucc.space[1]._occupied.copy()
        frozen = np.zeros((nocc + nvir,), dtype=bool)
        frozen[sa] = ucc.space[0]._frozen.copy()
        frozen[sb] = ucc.space[1]._frozen.copy()
        active = np.zeros((nocc + nvir,), dtype=bool)
        active[sa] = ucc.space[0]._active.copy()
        active[sb] = ucc.space[1]._active.copy()
        space = Space(occupied, frozen, active)

        slices = util.Namespace(
            a=util.Namespace(**{k: np.where(orbspin[space.mask(k)] == 0)[0] for k in "oOivVa"}),
            b=util.Namespace(**{k: np.where(orbspin[space.mask(k)] == 1)[0] for k in "oOivVa"}),
        )

        if ucc.bare_g is not None:
            if np.asarray(ucc.bare_g).ndim == 3:
                bare_g_a = bare_g_b = ucc.bare_g
            else:
                bare_g_a, bare_g_b = ucc.bare_g
            g = np.zeros((ucc.nbos, ucc.nmo * 2, ucc.nmo * 2))
            g[np.ix_(range(ucc.nbos), sa, sa)] = bare_g_a.copy()
            g[np.ix_(range(ucc.nbos), sb, sb)] = bare_g_b.copy()
        else:
            g = None

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
            amplitudes = util.Namespace()

            for name, key, n in ucc.ansatz.fermionic_cluster_ranks(spin_type=ucc.spin_type):
                shape = tuple(space.size(k) for k in key)
                amplitudes[name] = np.zeros(shape, dtype=types[float])
                for comb in util.generate_spin_combinations(n, unique=True):
                    done = set()
                    for lperm, lsign in util.permutations_with_signs(tuple(range(n))):
                        for uperm, usign in util.permutations_with_signs(tuple(range(n))):
                            combn = util.permute_string(comb[:n], lperm)
                            combn += util.permute_string(comb[n:], uperm)
                            if combn in done:
                                continue
                            mask = np.ix_(*[slices[s][k] for s, k in zip(combn, key)])
                            transpose = tuple(lperm) + tuple(p + n for p in uperm)
                            amp = (
                                getattr(ucc.amplitudes[name], comb).transpose(transpose)
                                * lsign
                                * usign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(n))):
                                transpose = tuple(perm) + tuple(range(n, 2 * n))
                                if util.permute_string(comb[:n], perm) == comb[:n]:
                                    amplitudes[name][mask] += amp.transpose(transpose).copy() * sign
                            done.add(combn)

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = ucc.amplitudes[name].copy()

            for name, key, nf, nb in ucc.ansatz.coupling_cluster_ranks(spin_type=ucc.spin_type):
                shape = (nbos,) * nb + tuple(space.size(k) for k in key[nb:])
                amplitudes[name] = np.zeros(shape, dtype=types[float])
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
                                *[slices[s][k] for s, k in zip(combn, key[nb:])],
                            )
                            transpose = (
                                tuple(range(nb))
                                + tuple(p + nb for p in lperm)
                                + tuple(p + nb + nf for p in uperm)
                            )
                            amp = (
                                getattr(ucc.amplitudes[name], comb).transpose(transpose)
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
                                    amplitudes[name][mask] += amp.transpose(transpose).copy() * sign
                            done.add(combn)

            gcc.amplitudes = amplitudes

        if has_lams:
            lambdas = gcc.init_lams()  # Easier this way - but have to build ERIs...

            for name, key, n in ucc.ansatz.fermionic_cluster_ranks(spin_type=ucc.spin_type):
                lname = name.replace("t", "l")
                shape = tuple(space.size(k) for k in key[n:] + key[:n])
                lambdas[lname] = np.zeros(shape, dtype=types[float])
                for comb in util.generate_spin_combinations(n, unique=True):
                    done = set()
                    for lperm, lsign in util.permutations_with_signs(tuple(range(n))):
                        for uperm, usign in util.permutations_with_signs(tuple(range(n))):
                            combn = util.permute_string(comb[:n], lperm)
                            combn += util.permute_string(comb[n:], uperm)
                            if combn in done:
                                continue
                            mask = np.ix_(*[slices[s][k] for s, k in zip(combn, key[n:] + key[:n])])
                            transpose = tuple(lperm) + tuple(p + n for p in uperm)
                            amp = (
                                getattr(ucc.lambdas[lname], comb).transpose(transpose)
                                * lsign
                                * usign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(n))):
                                transpose = tuple(perm) + tuple(range(n, 2 * n))
                                if util.permute_string(comb[:n], perm) == comb[:n]:
                                    lambdas[lname][mask] += amp.transpose(transpose).copy() * sign
                            done.add(combn)

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                lambdas[lname] = ucc.lambdas[lname].copy()

            for name, key, nf, nb in ucc.ansatz.coupling_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                shape = (nbos,) * nb + tuple(
                    space.size(k) for k in key[nb + nf :] + key[nb : nb + nf]
                )
                lambdas[lname] = np.zeros(shape, dtype=types[float])
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
                                *[
                                    slices[s][k]
                                    for s, k in zip(combn, key[nb + nf :] + key[nb : nb + nf])
                                ],
                            )
                            transpose = (
                                tuple(range(nb))
                                + tuple(p + nb for p in lperm)
                                + tuple(p + nb + nf for p in uperm)
                            )
                            amp = (
                                getattr(ucc.lambdas[lname], comb).transpose(transpose)
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
                                    lambdas[lname][mask] += amp.transpose(transpose).copy() * sign
                            done.add(combn)

            gcc.lambdas = lambdas

        return gcc

    @classmethod
    def from_rebcc(cls, rcc):
        """
        Initialise a `GEBCC` object from an `REBCC` object.

        Parameters
        ----------
        rcc : REBCC
            The REBCC object to initialise from.

        Returns
        -------
        gcc : GEBCC
            The GEBCC object.
        """

        ucc = uebcc.UEBCC.from_rebcc(rcc)
        gcc = cls.from_uebcc(ucc)

        return gcc

    @util.has_docstring
    def init_amps(self, eris=None):
        eris = self.get_eris(eris)
        amplitudes = util.Namespace()

        # Build T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            if n == 1:
                amplitudes[name] = getattr(self.fock, key) / self.energy_sum(key)
            elif n == 2:
                amplitudes[name] = getattr(eris, key) / self.energy_sum(key)
            else:
                shape = tuple(self.space.size(k) for k in key)
                amplitudes[name] = np.zeros(shape, dtype=types[float])

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
                amplitudes[name] = np.zeros(shape, dtype=types[float])

        # Build U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            if n == 1:
                amplitudes[name] = h[key] / self.energy_sum(key)
            else:
                shape = (self.nbos,) * nb + tuple(self.space.size(k) for k in key[nb:])
                amplitudes[name] = np.zeros(shape, dtype=types[float])

        return amplitudes

    @util.has_docstring
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

    @util.has_docstring
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

    @util.has_docstring
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

    @util.has_docstring
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

    @util.has_docstring
    def vector_to_excitations_ea(self, vector):
        excitations = []
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            key = key[n:] + key[: n - 1]
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

    @util.has_docstring
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

    @util.has_docstring
    def get_mean_field_G(self):
        val = lib.einsum("Ipp->I", self.g.boo)
        val -= self.xi * self.omega

        if self.bare_G is not None:
            val += self.bare_G

        return val

    def get_eris(self, eris=None):
        """
        Get blocks of the ERIs.

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

    @util.has_docstring
    def ip_eom(self, options=None, **kwargs):
        return geom.IP_GEOM(self, options=options, **kwargs)

    @util.has_docstring
    def ea_eom(self, options=None, **kwargs):
        return geom.EA_GEOM(self, options=options, **kwargs)

    @util.has_docstring
    def ee_eom(self, options=None, **kwargs):
        return geom.EE_GEOM(self, options=options, **kwargs)

    @property
    @util.has_docstring
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
    @util.has_docstring
    def spin_type(self):
        return "G"
