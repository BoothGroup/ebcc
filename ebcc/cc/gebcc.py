"""Generalised electron-boson coupled cluster."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyscf import scf

from ebcc import numpy as np
from ebcc import util
from ebcc.backend import _put
from ebcc.cc.base import BaseEBCC
from ebcc.core.precision import types
from ebcc.eom import EA_GEOM, EE_GEOM, IP_GEOM
from ebcc.ext.eccc import ExternalCorrectionGEBCC
from ebcc.ext.tcc import TailorGEBCC
from ebcc.ham.elbos import GElectronBoson
from ebcc.ham.eris import GERIs
from ebcc.ham.fock import GFock
from ebcc.ham.space import Space
from ebcc.opt.gbrueckner import BruecknerGEBCC

if TYPE_CHECKING:
    from typing import Any, Optional, TypeAlias, Union

    from numpy import floating
    from numpy.typing import NDArray
    from pyscf.scf.ghf import GHF
    from pyscf.scf.hf import SCF

    from ebcc.cc.rebcc import REBCC
    from ebcc.cc.uebcc import UEBCC
    from ebcc.util import Namespace

    T = floating

    ERIsInputType: TypeAlias = Union[GERIs, NDArray[T]]
    SpinArrayType: TypeAlias = NDArray[T]
    SpaceType: TypeAlias = Space


class GEBCC(BaseEBCC):
    """Restricted electron-boson coupled cluster."""

    # Types
    ERIs = GERIs
    Fock = GFock
    ElectronBoson = GElectronBoson
    Brueckner = BruecknerGEBCC
    ExternalCorrection = ExternalCorrectionGEBCC
    Tailor = TailorGEBCC

    # Attributes
    space: SpaceType
    amplitudes: Namespace[SpinArrayType]
    lambdas: Namespace[SpinArrayType]
    fock: GFock

    @property
    def spin_type(self) -> str:
        """Get a string representation of the spin type."""
        return "G"

    def ip_eom(self, **kwargs: Any) -> IP_GEOM:
        """Get the IP-EOM object.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            IP-EOM object.
        """
        return IP_GEOM(self, **kwargs)

    def ea_eom(self, **kwargs: Any) -> EA_GEOM:
        """Get the EA-EOM object.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            EA-EOM object.
        """
        return EA_GEOM(self, **kwargs)

    def ee_eom(self, **kwargs: Any) -> EE_GEOM:
        """Get the EE-EOM object.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            EE-EOM object.
        """
        return EE_GEOM(self, **kwargs)

    @staticmethod
    def _convert_mf(mf: SCF) -> GHF:
        """Convert a mean-field object to a GHF object.

        Note:
            Converts to UHF first to ensure consistent ordering.
        """
        if isinstance(mf, scf.ghf.GHF):
            return mf
        hf = mf.to_uhf().to_ghf()
        if hasattr(mf, "xc"):
            hf.e_tot = hf.energy_tot()
        return hf

    @classmethod
    def from_uebcc(cls, ucc: UEBCC) -> GEBCC:
        """Initialise a `GEBCC` object from an `UEBCC` object.

        Args:
            ucc: Unrestricted electron-boson coupled cluster object.

        Returns:
            GEBCC object.
        """
        orbspin = np.asarray(scf.addons.get_ghf_orbspin(ucc.mf.mo_energy, ucc.mf.mo_occ, False))
        nocc = ucc.space[0].nocc + ucc.space[1].nocc
        nvir = ucc.space[0].nvir + ucc.space[1].nvir
        nbos = ucc.nbos
        sa = np.where(orbspin == 0)[0]
        sb = np.where(orbspin == 1)[0]

        occupied = np.zeros((nocc + nvir,), dtype=np.bool_)
        occupied = _put(occupied, sa, np.copy(ucc.space[0]._occupied))
        occupied = _put(occupied, sb, np.copy(ucc.space[1]._occupied))
        frozen = np.zeros((nocc + nvir,), dtype=np.bool_)
        frozen = _put(frozen, sa, np.copy(ucc.space[0]._frozen))
        frozen = _put(frozen, sb, np.copy(ucc.space[1]._frozen))
        active = np.zeros((nocc + nvir,), dtype=np.bool_)
        active = _put(active, sa, np.copy(ucc.space[0]._active))
        active = _put(active, sb, np.copy(ucc.space[1]._active))
        space = Space(occupied, frozen, active)

        slices = util.Namespace(
            a=util.Namespace(**{k: np.where(orbspin[space.mask(k)] == 0)[0] for k in "oOivVa"}),
            b=util.Namespace(**{k: np.where(orbspin[space.mask(k)] == 1)[0] for k in "oOivVa"}),
        )

        g: Optional[NDArray[T]] = None
        if ucc.bare_g is not None:
            if ucc.bare_g.ndim == 3:
                bare_g_a = bare_g_b = ucc.bare_g
            else:
                bare_g_a, bare_g_b = ucc.bare_g
            g = np.zeros((ucc.nbos, ucc.nmo * 2, ucc.nmo * 2), dtype=types[float])
            g = _put(g, np.ix_(np.arange(ucc.nbos), sa, sa), np.copy(bare_g_a))
            g = _put(g, np.ix_(np.arange(ucc.nbos), sb, sb), np.copy(bare_g_b))

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

        has_amps = bool(ucc.amplitudes)
        has_lams = bool(ucc.lambdas)

        if has_amps:
            amplitudes: Namespace[SpinArrayType] = util.Namespace()

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
                                np.transpose(getattr(ucc.amplitudes[name], comb), transpose)
                                * lsign
                                * usign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(n))):
                                transpose = tuple(perm) + tuple(range(n, 2 * n))
                                if util.permute_string(comb[:n], perm) == comb[:n]:
                                    amplitudes[name] = _put(
                                        amplitudes[name],
                                        mask,
                                        amplitudes[name][mask]
                                        + np.transpose(amp, transpose) * sign,
                                    )
                            done.add(combn)

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                amplitudes[name] = np.copy(ucc.amplitudes[name])  # type: ignore

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
                                *([np.arange(nbos)] * nb),
                                *[slices[s][k] for s, k in zip(combn, key[nb:])],
                            )
                            transpose = (
                                tuple(range(nb))
                                + tuple(p + nb for p in lperm)
                                + tuple(p + nb + nf for p in uperm)
                            )
                            amp = (
                                np.transpose(getattr(ucc.amplitudes[name], comb), transpose)
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
                                    amplitudes[name] = _put(
                                        amplitudes[name],
                                        mask,
                                        amplitudes[name][mask]
                                        + np.transpose(amp, transpose) * sign,
                                    )
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
                                np.transpose(getattr(ucc.lambdas[lname], comb), transpose)
                                * lsign
                                * usign
                            )
                            for perm, sign in util.permutations_with_signs(tuple(range(n))):
                                transpose = tuple(perm) + tuple(range(n, 2 * n))
                                if util.permute_string(comb[:n], perm) == comb[:n]:
                                    lambdas[lname] = _put(
                                        lambdas[lname],
                                        mask,
                                        lambdas[lname][mask] + np.transpose(amp, transpose) * sign,
                                    )
                            done.add(combn)

            for name, key, n in ucc.ansatz.bosonic_cluster_ranks(spin_type=ucc.spin_type):
                lname = "l" + name
                lambdas[lname] = np.copy(ucc.lambdas[lname])  # type: ignore

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
                                *([np.arange(nbos)] * nb),
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
                                np.transpose(getattr(ucc.lambdas[lname], comb), transpose)
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
                                    lambdas[lname] = _put(
                                        lambdas[lname],
                                        mask,
                                        lambdas[lname][mask] + np.transpose(amp, transpose) * sign,
                                    )
                            done.add(combn)

            gcc.lambdas = lambdas

        return gcc

    @classmethod
    def from_rebcc(cls, rcc: REBCC) -> GEBCC:
        """Initialise a `GEBCC` object from an `REBCC` object.

        Args:
            rcc: Restricted electron-boson coupled cluster object.

        Returns:
            GEBCC object.
        """
        from ebcc.cc.uebcc import UEBCC

        ucc = UEBCC.from_rebcc(rcc)
        gcc = cls.from_uebcc(ucc)
        return gcc

    def init_space(self) -> SpaceType:
        """Initialise the fermionic space.

        Returns:
            Fermionic space. All fermionic degrees of freedom are assumed to be correlated.
        """
        space = Space(
            self.mo_occ > 0,
            np.zeros(self.mo_occ.shape, dtype=np.bool_),
            np.zeros(self.mo_occ.shape, dtype=np.bool_),
        )
        return space

    def _pack_codegen_kwargs(
        self, *extra_kwargs: dict[str, Any], eris: Optional[ERIsInputType] = None
    ) -> dict[str, Any]:
        """Pack all the keyword arguments for the generated code."""
        kwargs = dict(
            f=self.fock,
            v=self.get_eris(eris),
            g=self.g,
            G=self.G,
            w=np.diag(self.omega) if self.omega is not None else None,
            space=self.space,
            nocc=self.space.ncocc,
            nvir=self.space.ncvir,
            nbos=self.nbos,
        )

        for kw in extra_kwargs:
            if kw is not None:
                kwargs.update(kw)

        return kwargs

    def init_amps(self, eris: Optional[ERIsInputType] = None) -> Namespace[SpinArrayType]:
        """Initialise the cluster amplitudes.

        Args:
            eris: Electron repulsion integrals.

        Returns:
            Initial cluster amplitudes.
        """
        eris = self.get_eris(eris)
        amplitudes: Namespace[SpinArrayType] = util.Namespace()

        # Build T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            if n == 1:
                amplitudes[name] = getattr(self.fock, key) / self.energy_sum(key)
            elif n == 2:
                amplitudes[name] = getattr(eris, key) / self.energy_sum(key)
            else:
                shape = tuple(self.space.size(k) for k in key)
                amplitudes[name] = np.zeros(shape, dtype=types[float])

        # Build S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            if self.omega is None or self.G is None:
                raise ValueError("Bosonic parameters not set.")
            if n == 1:
                amplitudes[name] = -self.G / self.omega
            else:
                shape = (self.nbos,) * n
                amplitudes[name] = np.zeros(shape, dtype=types[float])

        # Build U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if self.omega is None or self.g is None:
                raise ValueError("Bosonic parameters not set.")
            if nf != 1:
                raise util.ModelNotImplemented
            if n == 1:
                amplitudes[name] = self.g[key] / self.energy_sum(key)
            else:
                shape = (self.nbos,) * nb + tuple(self.space.size(k) for k in key[nb:])
                amplitudes[name] = np.zeros(shape, dtype=types[float])

        return amplitudes

    def init_lams(
        self, amplitudes: Optional[Namespace[SpinArrayType]] = None
    ) -> Namespace[SpinArrayType]:
        """Initialise the cluster lambda amplitudes.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Initial cluster lambda amplitudes.
        """
        if not amplitudes:
            amplitudes = self.amplitudes
        lambdas: Namespace[SpinArrayType] = util.Namespace()

        # Build L amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            lname = name.replace("t", "l")
            perm = list(range(n, 2 * n)) + list(range(n))
            lambdas[lname] = np.transpose(amplitudes[name], perm)

        # Build LS amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            lname = "l" + name
            lambdas[lname] = amplitudes[name]

        # Build LU amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            lname = "l" + name
            perm = list(range(nb)) + [nb + 1, nb]
            lambdas[lname] = np.transpose(amplitudes[name], perm)

        return lambdas

    def update_amps(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
    ) -> Namespace[SpinArrayType]:
        """Update the cluster amplitudes.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.

        Returns:
            Updated cluster amplitudes.
        """
        amplitudes = self._get_amps(amplitudes=amplitudes)
        func, kwargs = self._load_function(
            "update_amps",
            eris=eris,
            amplitudes=amplitudes,
        )
        res: Namespace[SpinArrayType] = func(**kwargs)
        res = util.Namespace(**{key.rstrip("new"): val for key, val in res.items()})

        # Divide T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            res[name] /= self.energy_sum(key)
            res[name] += amplitudes[name]

        # Divide S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            res[name] /= self.energy_sum(key)
            res[name] += amplitudes[name]

        # Divide U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            if nf != 1:
                raise util.ModelNotImplemented
            res[name] /= self.energy_sum(key)
            res[name] += amplitudes[name]

        return res

    def update_lams(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        lambdas: Optional[Namespace[SpinArrayType]] = None,
        lambdas_pert: Optional[Namespace[SpinArrayType]] = None,
        perturbative: bool = False,
    ) -> Namespace[SpinArrayType]:
        """Update the cluster lambda amplitudes.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            lambdas_pert: Perturbative cluster lambda amplitudes.
            perturbative: Flag to include perturbative correction.

        Returns:
            Updated cluster lambda amplitudes.
        """
        # TODO active
        amplitudes = self._get_amps(amplitudes=amplitudes)
        lambdas = self._get_lams(lambdas=lambdas, amplitudes=amplitudes)
        if lambdas_pert is not None:
            lambdas.update(lambdas_pert)

        func, kwargs = self._load_function(
            "update_lams%s" % ("_perturbative" if perturbative else ""),
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        res: Namespace[SpinArrayType] = func(**kwargs)
        res = util.Namespace(**{key.rstrip("new"): val for key, val in res.items()})

        # Divide T amplitudes:
        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            res[name] /= self.energy_sum(key)
            if not perturbative:
                res[name] += lambdas[name]

        # Divide S amplitudes:
        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="l"):
            res[name] /= self.energy_sum(key)
            if not perturbative:
                res[name] += lambdas[name]

        # Divide U amplitudes:
        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            if nf != 1:
                raise util.ModelNotImplemented
            res[name] /= self.energy_sum(key)
            if not perturbative:
                res[name] += lambdas[name]

        if perturbative:
            res = Namespace(**{key + "pert": val for key, val in res.items()})

        return res

    def make_rdm1_f(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        lambdas: Optional[Namespace[SpinArrayType]] = None,
        hermitise: bool = True,
    ) -> SpinArrayType:
        r"""Make the one-particle fermionic reduced density matrix :math:`\langle i^+ j \rangle`.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            hermitise: Hermitise the density matrix.

        Returns:
            One-particle fermion reduced density matrix.
        """
        func, kwargs = self._load_function(
            "make_rdm1_f",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        dm: SpinArrayType = func(**kwargs)

        if hermitise:
            dm = (dm + np.transpose(dm)) * 0.5

        return dm

    def make_rdm2_f(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        lambdas: Optional[Namespace[SpinArrayType]] = None,
        hermitise: bool = True,
    ) -> SpinArrayType:
        r"""Make the two-particle fermionic reduced density matrix :math:`\langle i^+j^+lk \rangle`.

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            hermitise: Hermitise the density matrix.

        Returns:
            Two-particle fermion reduced density matrix.
        """
        func, kwargs = self._load_function(
            "make_rdm2_f",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        dm: SpinArrayType = func(**kwargs)

        if hermitise:
            dm = (np.transpose(dm, (0, 1, 2, 3)) + np.transpose(dm, (2, 3, 0, 1))) * 0.5
            dm = (np.transpose(dm, (0, 1, 2, 3)) + np.transpose(dm, (1, 0, 3, 2))) * 0.5

        return dm

    def make_eb_coup_rdm(
        self,
        eris: Optional[ERIsInputType] = None,
        amplitudes: Optional[Namespace[SpinArrayType]] = None,
        lambdas: Optional[Namespace[SpinArrayType]] = None,
        unshifted: bool = True,
        hermitise: bool = True,
    ) -> SpinArrayType:
        r"""Make the electron-boson coupling reduced density matrix.

        .. math::
            \langle b^+ i^+ j \rangle

        and

        .. math::
            \langle b i^+ j \rangle

        Args:
            eris: Electron repulsion integrals.
            amplitudes: Cluster amplitudes.
            lambdas: Cluster lambda amplitudes.
            unshifted: If `self.options.shift` is `True`, return the unshifted density matrix. Has
                no effect if `self.options.shift` is `False`.
            hermitise: Hermitise the density matrix.

        Returns:
            Electron-boson coupling reduced density matrix.
        """
        func, kwargs = self._load_function(
            "make_eb_coup_rdm",
            eris=eris,
            amplitudes=amplitudes,
            lambdas=lambdas,
        )
        dm_eb: SpinArrayType = func(**kwargs)

        if hermitise:
            dm_eb = np.array(
                [
                    (dm_eb[0] + np.transpose(dm_eb[1], (0, 2, 1))) * 0.5,
                    (dm_eb[1] + np.transpose(dm_eb[0], (0, 2, 1))) * 0.5,
                ]
            )

        if unshifted and self.options.shift:
            rdm1_f = self.make_rdm1_f(hermitise=hermitise)
            shift = util.einsum("x,ij->xij", self.xi, rdm1_f)
            dm_eb -= shift[None]

        return dm_eb

    def energy_sum(self, *args: str, signs_dict: Optional[dict[str, str]] = None) -> NDArray[T]:
        """Get a direct sum of energies.

        Args:
            *args: Energies to sum. Should specify a subscript only.
            signs_dict: Signs of the energies in the sum. Default sets `("o", "O", "i")` to be
                positive, and `("v", "V", "a", "b")` to be negative.

        Returns:
            Sum of energies.
        """
        (subscript,) = args
        n = 0

        def next_char() -> str:
            nonlocal n
            if n < 26:
                char = chr(ord("a") + n)
            else:
                char = chr(ord("A") + n)
            n += 1
            return char

        if signs_dict is None:
            signs_dict = {}
        for k, s in zip("vVaoOib", "---+++-"):
            if k not in signs_dict:
                signs_dict[k] = s

        energies = []
        for key in subscript:
            factor = 1 if signs_dict[key] == "+" else -1
            if key == "b":
                assert self.omega is not None
                energies.append(self.omega * types[float](factor))
            else:
                energies.append(np.diag(self.fock[key + key]) * types[float](factor))

        subscript = ",".join([next_char() for k in subscript])
        energy_sum = util.dirsum(subscript, *energies)

        return energy_sum

    def amplitudes_to_vector(self, amplitudes: Namespace[SpinArrayType]) -> NDArray[T]:
        """Construct a vector containing all of the amplitudes used in the given ansatz.

        Args:
            amplitudes: Cluster amplitudes.

        Returns:
            Cluster amplitudes as a vector.
        """
        vectors = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(np.ravel(amplitudes[name]))

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            vectors.append(np.ravel(amplitudes[name]))

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            vectors.append(np.ravel(amplitudes[name]))

        return np.concatenate(vectors)

    def vector_to_amplitudes(self, vector: NDArray[T]) -> Namespace[SpinArrayType]:
        """Construct a namespace of amplitudes from a vector.

        Args:
            vector: Cluster amplitudes as a vector.

        Returns:
            Cluster amplitudes.
        """
        amplitudes: Namespace[SpinArrayType] = util.Namespace()
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(spin_type=self.spin_type):
            shape = tuple(self.space.size(k) for k in key)
            size = util.prod(shape)
            amplitudes[name] = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type):
            shape = (self.nbos,) * n
            size = util.prod(shape)
            amplitudes[name] = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(spin_type=self.spin_type):
            shape = (self.nbos,) * nb + tuple(self.space.size(k) for k in key[nb:])
            size = util.prod(shape)
            amplitudes[name] = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        return amplitudes

    def lambdas_to_vector(self, lambdas: Namespace[SpinArrayType]) -> NDArray[T]:
        """Construct a vector containing all of the lambda amplitudes used in the given ansatz.

        Args:
            lambdas: Cluster lambda amplitudes.

        Returns:
            Cluster lambda amplitudes as a vector.
        """
        vectors = []

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            vectors.append(np.ravel(lambdas[name]))

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="l"):
            vectors.append(np.ravel(lambdas[name]))

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            vectors.append(np.ravel(lambdas[name]))

        return np.concatenate(vectors)

    def vector_to_lambdas(self, vector: NDArray[T]) -> Namespace[SpinArrayType]:
        """Construct a namespace of lambda amplitudes from a vector.

        Args:
            vector: Cluster lambda amplitudes as a vector.

        Returns:
            Cluster lambda amplitudes.
        """
        lambdas: Namespace[SpinArrayType] = util.Namespace()
        i0 = 0

        for name, key, n in self.ansatz.fermionic_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            shape = tuple(self.space.size(k) for k in key)
            size = util.prod(shape)
            lambdas[name] = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        for name, key, n in self.ansatz.bosonic_cluster_ranks(spin_type=self.spin_type, which="l"):
            shape = (self.nbos,) * n
            size = util.prod(shape)
            lambdas[name] = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        for name, key, nf, nb in self.ansatz.coupling_cluster_ranks(
            spin_type=self.spin_type, which="l"
        ):
            shape = (self.nbos,) * nb + tuple(self.space.size(k) for k in key[nb:])
            size = util.prod(shape)
            lambdas[name] = np.reshape(vector[i0 : i0 + size], shape)
            i0 += size

        return lambdas

    def get_mean_field_G(self) -> NDArray[T]:
        """Get the mean-field boson non-conserving term.

        Returns:
            Mean-field boson non-conserving term.
        """
        assert self.g is not None
        assert self.omega is not None
        # FIXME should this also sum in frozen orbitals?
        boo: NDArray[T] = self.g.boo
        val = util.einsum("Ipp->I", boo)
        val -= self.xi * self.omega
        if self.bare_G is not None:
            val += self.bare_G
        return val

    @property
    def xi(self) -> NDArray[T]:
        """Get the shift in the bosonic operators to diagonalise the photon Hamiltonian.

        Returns:
            Shift in the bosonic operators.
        """
        assert self.omega is not None
        if self.options.shift:
            assert self.g is not None
            boo: NDArray[T] = self.g.boo
            xi = util.einsum("Iii->I", boo)
            xi /= self.omega
            if self.bare_G is not None:
                xi += self.bare_G / self.omega
        else:
            xi = np.zeros(self.omega.shape)
        return xi

    @property
    def nmo(self) -> int:
        """Get the number of molecular orbitals.

        Returns:
            Number of molecular orbitals.
        """
        return self.space.nmo

    @property
    def nocc(self) -> int:
        """Get the number of occupied molecular orbitals.

        Returns:
            Number of occupied molecular orbitals.
        """
        return self.space.nocc

    @property
    def nvir(self) -> int:
        """Get the number of virtual molecular orbitals.

        Returns:
            Number of virtual molecular orbitals.
        """
        return self.space.nvir
